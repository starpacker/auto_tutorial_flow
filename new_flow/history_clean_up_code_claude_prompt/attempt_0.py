import os
import sys
import yaml
import torch
import torch.distributed as dist
from torch.fft import fft2, ifft2
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur
from math import prod
from copy import deepcopy

# --- Mocking/Importing External Dependencies based on context ---
# In a real scenario, these would be standard imports. 
# Based on the prompt, I will assume the environment has 'ptyrad' installed 
# or available in the path, but I will structure the code to use the logic 
# provided in the input snippets where possible to ensure self-containment 
# regarding the core logic.

from ptyrad.utils import vprint, time_sync, print_system_info, set_gpu_device, CustomLogger, imshift_batch, torch_phasor
from ptyrad.forward import multislice_forward_model_vec_all
from ptyrad.reconstruction import create_optimizer, prepare_recon, parse_sec_to_time_str, recon_loop, IndicesDataset
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss
from ptyrad.constraints import CombinedConstraint
from ptyrad.params import PtyRADParams

# --- 1. Data Loading and Preprocessing ---

def load_and_preprocess_data(params_path="PSO_reconstruct.yml", gpuid=0):
    """
    Loads data from disk/arguments and returns preprocessed tensors/arrays/objects.
    """
    print_system_info()
    
    # 1. Load Parameters
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"The specified file '{params_path}' does not exist.")
    
    vprint("### Loading params file ###")
    with open(params_path, "r", encoding='utf-8') as file:
        params_dict = yaml.safe_load(file)
    
    # Normalize constraint params (Legacy support logic from input)
    if params_dict.get('constraint_params') is not None:
        constraint_params = params_dict['constraint_params']
        normalized_params = {}
        for name, p in constraint_params.items():
            freq = p.get("freq", None)
            start_iter = p.get("start_iter", 1 if freq is not None else None)
            step = p.get("step", freq if freq is not None else 1)
            end_iter = p.get("end_iter", None)
            normalized_params[name] = {
                "start_iter": start_iter,
                "step": step,
                "end_iter": end_iter,
                **{k: v for k, v in p.items() if k not in ("freq", "step", "start_iter", "end_iter")},
            }
        params_dict['constraint_params'] = normalized_params

    # Validate params
    params = PtyRADParams(**params_dict).model_dump()
    params['params_path'] = params_path

    # 2. Setup Device and Logger
    logger = CustomLogger(log_file='ptyrad_log.txt', log_dir='auto', prefix_time='datetime', show_timestamp=True)
    device = set_gpu_device(gpuid=gpuid)

    # 3. Initialize Data (Measurements, Probe, Object guesses)
    # The Initializer class handles loading .mat files or generating synthetic data
    vprint("### Initializing Initializer ###")
    # We need a seed for reproducibility
    seed = None 
    initializer = Initializer(params['init_params'], seed=seed)
    init_variables = initializer.init_all().init_variables

    # 4. Initialize Loss and Constraints
    vprint("### Initializing loss function ###")
    loss_fn = CombinedLoss(params['loss_params'], device=device)
    
    vprint("### Initializing constraint function ###")
    constraint_fn = CombinedConstraint(params['constraint_params'], device=device, verbose=not params['recon_params']['if_quiet'])

    # Bundle everything into a data dictionary
    data = {
        "params": params,
        "device": device,
        "logger": logger,
        "init_variables": init_variables,
        "initializer_obj": initializer, # Kept for metadata access
        "loss_fn": loss_fn,
        "constraint_fn": constraint_fn
    }
    
    return data

# --- 2. Forward Operator ---

class PtychoAD(torch.nn.Module):
    """
    The physical model A in y = A(x).
    Encapsulates the physics of ptychography.
    """
    def __init__(self, init_variables, model_params, device='cuda', verbose=True):
        super(PtychoAD, self).__init__()
        self.device = device
        self.verbose = verbose
        
        # Physics parameters
        self.detector_blur_std = model_params['detector_blur_std']
        self.obj_preblur_std = model_params['obj_preblur_std']
        
        # On-the-fly measurement handling
        if init_variables.get('on_the_fly_meas_padded', None) is not None:
            self.meas_padded = torch.tensor(init_variables['on_the_fly_meas_padded'], dtype=torch.float32, device=device)
            self.meas_padded_idx = torch.tensor(init_variables['on_the_fly_meas_padded_idx'], dtype=torch.int32, device=device)
        else:
            self.meas_padded = None
        self.meas_scale_factors = init_variables.get('on_the_fly_meas_scale_factors', None)

        # Optimizer config parsing
        self.optimizer_params = model_params['optimizer_params']
        self.lr_params = {k: v['lr'] for k, v in model_params['update_params'].items()}
        self.start_iter = {k: v.get('start_iter') for k, v in model_params['update_params'].items()}
        self.end_iter = {k: v.get('end_iter') for k, v in model_params['update_params'].items()}

        # Optimizable Tensors (The "Latent Variables")
        self.opt_obja = nn.Parameter(torch.abs(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
        self.opt_objp = nn.Parameter(torch.angle(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
        self.opt_obj_tilts = nn.Parameter(torch.tensor(init_variables['obj_tilts'], dtype=torch.float32, device=device))
        self.opt_slice_thickness = nn.Parameter(torch.tensor(init_variables['slice_thickness'], dtype=torch.float32, device=device))
        self.opt_probe = nn.Parameter(torch.view_as_real(torch.tensor(init_variables['probe'], dtype=torch.complex64, device=device)))
        self.opt_probe_pos_shifts = nn.Parameter(torch.tensor(init_variables['probe_pos_shifts'], dtype=torch.float32, device=device))

        # Fixed Buffers
        self.register_buffer('omode_occu', torch.tensor(init_variables['omode_occu'], dtype=torch.float32, device=device))
        self.register_buffer('H', torch.tensor(init_variables['H'], dtype=torch.complex64, device=device))
        self.register_buffer('measurements', torch.tensor(init_variables['measurements'], dtype=torch.float32, device=device))
        self.register_buffer('crop_pos', torch.tensor(init_variables['crop_pos'], dtype=torch.int32, device=device))
        self.register_buffer('dx', torch.tensor(init_variables['dx'], dtype=torch.float32, device=device))
        self.register_buffer('lambd', torch.tensor(init_variables['lambd'], dtype=torch.float32, device=device))
        
        # State flags
        self.tilt_obj = bool(self.lr_params['obj_tilts'] != 0 or torch.any(self.opt_obj_tilts))
        self.shift_probes = bool(self.lr_params['probe_pos_shifts'] != 0)
        self.change_thickness = bool(self.lr_params['slice_thickness'] != 0)
        
        # Internal Grids
        self._create_grids()
        self._init_propagator_vars()
        
        # Register optimizable params
        self.optimizable_tensors = {
            'obja': self.opt_obja, 'objp': self.opt_objp, 'obj_tilts': self.opt_obj_tilts,
            'slice_thickness': self.opt_slice_thickness, 'probe': self.opt_probe,
            'probe_pos_shifts': self.opt_probe_pos_shifts
        }
        self._configure_optimizer_params()
        
        # Cache
        self._current_object_patches = None

    def _create_grids(self):
        probe = torch.view_as_complex(self.opt_probe)
        Npy, Npx = probe.shape[-2:]
        
        # Propagator Grid
        ygrid = (torch.arange(-Npy // 2, Npy // 2, device=self.device) + 0.5) / Npy
        xgrid = (torch.arange(-Npx // 2, Npx // 2, device=self.device) + 0.5) / Npx
        ky = torch.fft.ifftshift(2 * torch.pi * ygrid / self.dx)
        kx = torch.fft.ifftshift(2 * torch.pi * xgrid / self.dx)
        Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
        self.propagator_grid = torch.stack([Ky,Kx], dim=0)

        # ROI Grid
        self.rpy_grid, self.rpx_grid = torch.meshgrid(
            torch.arange(Npy, dtype=torch.int32, device=self.device), 
            torch.arange(Npx, dtype=torch.int32, device=self.device), indexing='ij'
        )
        
        # Shift Grid
        kpy, kpx = torch.meshgrid(torch.fft.fftfreq(Npy, dtype=torch.float32, device=self.device),
                                  torch.fft.fftfreq(Npx, dtype=torch.float32, device=device), indexing='ij')
        self.shift_probes_grid = torch.stack([kpy, kpx], dim=0)

    def _init_propagator_vars(self):
        dz = self.opt_slice_thickness.detach()
        Ky, Kx = self.propagator_grid 
        tilts_y = self.opt_obj_tilts[:,0,None,None] / 1e3
        tilts_x = self.opt_obj_tilts[:,1,None,None] / 1e3
        self.H_fixed_tilts_full = self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
        
        k = 2 * torch.pi / self.lambd
        self.Kz = torch.sqrt(k ** 2 - Kx ** 2 - Ky ** 2)

    def _configure_optimizer_params(self):
        self.optimizable_params = []
        for name, lr in self.lr_params.items():
            tensor = self.optimizable_tensors[name]
            tensor.requires_grad = (lr != 0) and (self.start_iter[name] == 1)
            if lr != 0:
                self.optimizable_params.append({'params': [tensor], 'lr': lr})

    def get_obj_patches(self, indices):
        opt_obj = torch.stack([self.opt_obja, self.opt_objp], dim=-1)
        obj_ROI_grid_y = self.rpy_grid[None,:,:] + self.crop_pos[indices, None, None, 0]
        obj_ROI_grid_x = self.rpx_grid[None,:,:] + self.crop_pos[indices, None, None, 1]
        object_patches = opt_obj[:,:,obj_ROI_grid_y,obj_ROI_grid_x,:].permute(2,0,1,3,4,5)
        
        if self.obj_preblur_std is not None and self.obj_preblur_std != 0:
            obj = object_patches.permute(5,0,1,2,3,4)
            obj_shape = obj.shape
            obj = obj.reshape(-1, obj_shape[-2], obj_shape[-1])
            object_patches = gaussian_blur(obj, kernel_size=5, sigma=self.obj_preblur_std).reshape(obj_shape).permute(1,2,3,4,5,0)
            
        return object_patches

    def get_probes(self, indices):
        probe = torch.view_as_complex(self.opt_probe)
        if self.shift_probes:
            return imshift_batch(probe, shifts=self.opt_probe_pos_shifts[indices], grid=self.shift_probes_grid)
        return torch.broadcast_to(probe, (indices.shape[0], *probe.shape))

    def get_propagators(self, indices):
        # Simplified logic for brevity, assuming standard case or full tilt optimization
        dz = self.opt_slice_thickness
        Ky, Kx = self.propagator_grid
        
        if self.tilt_obj:
            tilts = self.opt_obj_tilts if self.opt_obj_tilts.shape[0] == 1 else self.opt_obj_tilts[indices]
            tilts_y = tilts[:,0,None,None] / 1e3
            tilts_x = tilts[:,1,None,None] / 1e3
            
            if self.change_thickness:
                H_opt_dz = torch_phasor(dz * self.Kz)
                return H_opt_dz * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
            elif self.lr_params['obj_tilts'] != 0:
                return self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
            else:
                return self.H_fixed_tilts_full if self.opt_obj_tilts.shape[0] == 1 else self.H_fixed_tilts_full[indices]
        
        if self.change_thickness:
            return torch_phasor(dz * self.Kz)[None,]
            
        return self.H[None,]

    def get_measurements(self, indices=None):
        if indices is None: return self.measurements
        
        measurements = self.measurements[indices]
        if self.meas_padded is not None:
            pad_h1, pad_h2, pad_w1, pad_w2 = self.meas_padded_idx
            canvas = torch.zeros((measurements.shape[0], *self.meas_padded.shape[-2:]), dtype=measurements.dtype, device=self.device)
            canvas += self.meas_padded
            canvas[..., pad_h1:pad_h2, pad_w1:pad_w2] = measurements
            measurements = canvas
            
        if self.meas_scale_factors is not None:
            scale = tuple(self.meas_scale_factors)
            if any(f != 1 for f in scale):
                measurements = torch.nn.functional.interpolate(measurements[None,], scale_factor=scale, mode='bilinear')[0]
                measurements = measurements / prod(scale)
        return measurements

    def forward(self, indices):
        """
        The core forward operator.
        INPUT: indices (representing selection of latent variables x)
        OUTPUT: y_pred (predicted diffraction patterns)
        """
        object_patches = self.get_obj_patches(indices)
        probes = self.get_probes(indices)
        propagators = self.get_propagators(indices)
        
        # Physics calculation
        dp_fwd = multislice_forward_model_vec_all(object_patches, probes, propagators, omode_occu=self.omode_occu)
        
        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=self.detector_blur_std)
            
        self._current_object_patches = object_patches # Cache for loss
        return dp_fwd

def forward_operator(model, indices):
    """
    Wrapper to satisfy the functional requirement.
    """
    return model(indices)

# --- 3. Run Inversion ---

def run_inversion(data, accelerator=None):
    """
    Performs the optimization/solver loop.
    """
    params = data['params']
    device = data['device']
    init_variables = data['init_variables']
    initializer_obj = data['initializer_obj'] # Needed for prepare_recon
    loss_fn = data['loss_fn']
    constraint_fn = data['constraint_fn']
    logger = data['logger']
    
    vprint("### Starting Inversion ###")
    start_t = time_sync()

    # 1. Instantiate the Model (Physics)
    model = PtychoAD(init_variables, params['model_params'], device=device, verbose=True)
    
    # 2. Create Optimizer
    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
    
    # 3. Prepare Data Loaders / Batches
    # Note: Logic adapted from PtyRADSolver.reconstruct
    if accelerator is None:
        indices, batches, output_path = prepare_recon(model, initializer_obj.init, params)
    else:
        # Multi-GPU logic
        params['recon_params']['GROUP_MODE'] = 'random'
        indices, batches, output_path = prepare_recon(model, initializer_obj.init, params)
        ds = IndicesDataset(indices)
        dl = torch.utils.data.DataLoader(ds, batch_size=params['recon_params']['BATCH_SIZE']['size'], shuffle=True)
        batches = accelerator.prepare(dl)
        model, optimizer = accelerator.prepare(model, optimizer)

    # 4. Run Optimization Loop
    # The recon_loop internally calls model(indices), which is our forward_operator
    if logger is not None and logger.flush_file:
        logger.flush_to_file(log_dir=output_path)

    recon_loop(
        model, 
        initializer_obj.init, 
        params, 
        optimizer, 
        loss_fn, 
        constraint_fn, 
        indices, 
        batches, 
        output_path, 
        acc=accelerator
    )
    
    end_t = time_sync()
    solver_t = end_t - start_t
    vprint(f"### Inversion finished in {solver_t:.3f} sec ###")
    
    if logger is not None and logger.flush_file:
        logger.close()

    if dist.is_initialized():
        dist.destroy_process_group()

    return model

# --- 4. Evaluate Results ---

def evaluate_results(model, output_path=None):
    """
    Calculates metrics and visualizes/saves results.
    """
    vprint("### Evaluating Results ###")
    
    # In a real scenario, we would calculate PSNR/SSIM here against ground truth if available.
    # Since the provided code focuses on reconstruction without explicit ground truth in the snippet,
    # we will simulate the evaluation by extracting the final object.
    
    # Extract reconstructed object
    with torch.no_grad():
        obj_amp = model.opt_obja.cpu().numpy()
        obj_phase = model.opt_objp.cpu().numpy()
        
    vprint(f"Reconstructed Object Shape: {obj_amp.shape}")
    vprint("Evaluation complete. Results are stored in the model object.")
    
    # If we had ground truth, we would do:
    # psnr = calculate_psnr(obj_amp, ground_truth_amp)
    # vprint(f"PSNR: {psnr}")

# --- Main Execution Block ---

if __name__ == '__main__':
    # 1. Load Data
    data = load_and_preprocess_data(params_path="PSO_reconstruct.yml", gpuid=0)
    
    # 2. Run Inversion (Implicitly uses forward_operator via the model class)
    # Note: accelerator is None for single GPU execution as per default input
    res_model = run_inversion(data, accelerator=None)
    
    # 3. Evaluate
    evaluate_results(res_model)
    
    # 4. Success Signal
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")