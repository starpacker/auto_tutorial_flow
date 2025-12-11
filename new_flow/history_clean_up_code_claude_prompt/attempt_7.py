import os
import sys
import yaml
import torch
import torch.distributed as dist
from copy import deepcopy
from math import prod
from torch.fft import fft2, ifft2
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur

# --- Third-party / Library Imports ---
from ptyrad.utils import (
    print_system_info, set_gpu_device, CustomLogger, vprint, 
    time_sync, imshift_batch, torch_phasor
)
from ptyrad.params import PtyRADParams
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss
from ptyrad.constraints import CombinedConstraint
from ptyrad.reconstruction import (
    create_optimizer, prepare_recon, recon_loop, 
    IndicesDataset, parse_sec_to_time_str
)
from ptyrad.forward import multislice_forward_model_vec_all

# ==============================================================================
# 1. Data Loading and Preprocessing
# ==============================================================================

def load_and_preprocess_data(params_path: str, gpuid: int = 0):
    """
    Loads configuration, initializes system resources, and prepares initial data structures.
    
    Returns:
        dict: A context dictionary containing params, device, logger, and initialized data objects.
    """
    # 1. Setup Logging and System Info
    logger = CustomLogger(log_file='ptyrad_log.txt', log_dir='auto', prefix_time='datetime', show_timestamp=True)
    print_system_info()

    # 2. Load and Validate Parameters
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"File '{params_path}' not found.")
    
    vprint("### Loading params file ###")
    with open(params_path, "r", encoding='utf-8') as file:
        raw_params = yaml.safe_load(file)

    # Normalize constraint params (legacy support logic)
    if raw_params.get('constraint_params') is not None:
        c_params = raw_params['constraint_params']
        norm_params = {}
        for name, p in c_params.items():
            freq = p.get("freq", None)
            norm_params[name] = {
                "start_iter": p.get("start_iter", 1 if freq is not None else None),
                "step": p.get("step", freq if freq is not None else 1),
                "end_iter": p.get("end_iter", None),
                **{k: v for k, v in p.items() if k not in ("freq", "step", "start_iter", "end_iter")},
            }
        raw_params['constraint_params'] = norm_params

    # Validate via Pydantic model
    params = PtyRADParams(**raw_params).model_dump()
    params['params_path'] = params_path

    # 3. Setup Device
    device = set_gpu_device(gpuid=gpuid)

    # 4. Initialize Data (Measurements, Probe, Object, etc.)
    # The Initializer class handles loading .mat files or generating simulation data
    vprint("### Initializing Initializer ###")
    # Note: We pass seed=None to let params control it, or could pass a specific seed
    initializer = Initializer(params['init_params'], seed=None).init_all()
    
    # Return a context dictionary to pass to the next stage
    return {
        "params": params,
        "device": device,
        "logger": logger,
        "initializer": initializer,
        "accelerator": None # Placeholder if accelerate is used later
    }

# ==============================================================================
# 2. Forward Operator (Physics Model)
# ==============================================================================

class PtychoAD(torch.nn.Module):
    """
    The physical model A in y = A(x).
    Encapsulates the optimizable tensors and the forward physics.
    """
    def __init__(self, init_variables, model_params, device='cuda', verbose=True):
        super(PtychoAD, self).__init__()
        with torch.no_grad():
            vprint('### Initializing PtychoAD model ###', verbose=verbose)
            
            self.device = device
            self.verbose = verbose
            self.detector_blur_std = model_params['detector_blur_std']
            self.obj_preblur_std = model_params['obj_preblur_std']
            
            # Handle on-the-fly measurement padding
            if init_variables.get('on_the_fly_meas_padded', None) is not None:
                self.meas_padded = torch.tensor(init_variables['on_the_fly_meas_padded'], dtype=torch.float32, device=device)
                self.meas_padded_idx = torch.tensor(init_variables['on_the_fly_meas_padded_idx'], dtype=torch.int32, device=device)
            else:
                self.meas_padded = None
            self.meas_scale_factors = init_variables.get('on_the_fly_meas_scale_factors', None)

            # Parse learning rates
            self.optimizer_params = model_params['optimizer_params']
            self.start_iter = {}
            self.end_iter = {}
            self.lr_params = {}
            for key, p in model_params['update_params'].items():
                self.start_iter[key] = p.get('start_iter')
                self.end_iter[key] = p.get('end_iter')
                self.lr_params[key] = p['lr']
            
            # Optimizable parameters
            self.opt_obja = nn.Parameter(torch.abs(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
            self.opt_objp = nn.Parameter(torch.angle(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
            self.opt_obj_tilts = nn.Parameter(torch.tensor(init_variables['obj_tilts'], dtype=torch.float32, device=device))
            self.opt_slice_thickness = nn.Parameter(torch.tensor(init_variables['slice_thickness'], dtype=torch.float32, device=device))
            self.opt_probe = nn.Parameter(torch.view_as_real(torch.tensor(init_variables['probe'], dtype=torch.complex64, device=device)))
            self.opt_probe_pos_shifts = nn.Parameter(torch.tensor(init_variables['probe_pos_shifts'], dtype=torch.float32, device=device))
            
            # Buffers (Fixed constants or reference data)
            self.register_buffer('omode_occu', torch.tensor(init_variables['omode_occu'], dtype=torch.float32, device=device))
            self.register_buffer('H', torch.tensor(init_variables['H'], dtype=torch.complex64, device=device))
            self.register_buffer('measurements', torch.tensor(init_variables['measurements'], dtype=torch.float32, device=device))
            self.register_buffer('N_scan_slow', torch.tensor(init_variables['N_scan_slow'], dtype=torch.int32, device=device))
            self.register_buffer('N_scan_fast', torch.tensor(init_variables['N_scan_fast'], dtype=torch.int32, device=device))
            self.register_buffer('crop_pos', torch.tensor(init_variables['crop_pos'], dtype=torch.int32, device=device))
            self.register_buffer('slice_thickness', torch.tensor(init_variables['slice_thickness'], dtype=torch.float32, device=device))
            self.register_buffer('dx', torch.tensor(init_variables['dx'], dtype=torch.float32, device=device))
            self.register_buffer('dk', torch.tensor(init_variables['dk'], dtype=torch.float32, device=device))
            self.register_buffer('lambd', torch.tensor(init_variables['lambd'], dtype=torch.float32, device=device))
            
            self.random_seed = init_variables['random_seed']
            self.length_unit = init_variables['length_unit']
            self.scan_affine = init_variables['scan_affine']
            self.tilt_obj = bool(self.lr_params['obj_tilts'] != 0 or torch.any(self.opt_obj_tilts))
            self.shift_probes = bool(self.lr_params['probe_pos_shifts'] != 0)
            self.change_thickness = bool(self.lr_params['slice_thickness'] != 0)
            
            # Create grids
            self.create_grids()
            self.probe_int_sum = self.get_complex_probe_view().abs().pow(2).sum()

            # Setup optimizable tensors dict
            self.optimizable_tensors = {
                'obja': self.opt_obja, 'objp': self.opt_objp, 'obj_tilts': self.opt_obj_tilts,
                'slice_thickness': self.opt_slice_thickness, 'probe': self.opt_probe,
                'probe_pos_shifts': self.opt_probe_pos_shifts
            }
            self.create_optimizable_params_dict(self.lr_params, self.verbose)
            self.init_propagator_vars()
            self.init_compilation_iters()
            
            # Cache placeholder
            self._current_object_patches = None

    def get_complex_probe_view(self):
        return torch.view_as_complex(self.opt_probe)
        
    def create_grids(self):
        device = self.device
        probe = self.get_complex_probe_view()
        Npy, Npx = probe.shape[-2:]
        Noy, Nox = self.opt_objp.shape[-2:]
        
        ygrid = (torch.arange(-Npy // 2, Npy // 2, device=device) + 0.5) / Npy
        xgrid = (torch.arange(-Npx // 2, Npx // 2, device=device) + 0.5) / Npx
        ky = torch.fft.ifftshift(2 * torch.pi * ygrid / self.dx)
        kx = torch.fft.ifftshift(2 * torch.pi * xgrid / self.dx)
        Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
        self.propagator_grid = torch.stack([Ky,Kx], dim=0)
        
        rpy, rpx = torch.meshgrid(torch.arange(Npy, dtype=torch.int32, device=device), 
                                  torch.arange(Npx, dtype=torch.int32, device=device), indexing='ij')
        self.rpy_grid = rpy
        self.rpx_grid = rpx
        
        kpy, kpx = torch.meshgrid(torch.fft.fftfreq(Npy, dtype=torch.float32, device=device),
                                  torch.fft.fftfreq(Npx, dtype=torch.float32, device=device), indexing='ij')
        self.shift_probes_grid = torch.stack([kpy, kpx], dim=0)

    def create_optimizable_params_dict(self, lr_params, verbose=True):
        self.optimizable_params = []
        for param_name, lr in lr_params.items():
            if param_name not in self.optimizable_tensors:
                raise ValueError(f"Invalid param: {param_name}")
            self.optimizable_tensors[param_name].requires_grad = (lr != 0) and (self.start_iter[param_name] == 1)
            if lr != 0:
                self.optimizable_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})
        if verbose:
            self.print_model_summary()

    def init_propagator_vars(self):
        dz = self.opt_slice_thickness.detach()
        Ky, Kx = self.propagator_grid 
        tilts_y_full = self.opt_obj_tilts[:,0,None,None] / 1e3
        tilts_x_full = self.opt_obj_tilts[:,1,None,None] / 1e3
        self.H_fixed_tilts_full = self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y_full) + Kx * torch.tan(tilts_x_full)))
        self.k = 2 * torch.pi / self.lambd
        self.Kz = torch.sqrt(self.k ** 2 - Kx ** 2 - Ky ** 2)

    def init_compilation_iters(self):
        compilation_iters = {1}
        for param_name in self.optimizable_tensors.keys():
            s = self.start_iter.get(param_name)
            e = self.end_iter.get(param_name)
            if s and s >= 1: compilation_iters.add(s)
            if e and e >= 1: compilation_iters.add(e)
        self.compilation_iters = sorted(compilation_iters)

    def print_model_summary(self):
        vprint('### PtychoAD optimizable variables ###')
        for name, tensor in self.optimizable_tensors.items():
            vprint(f"{name.ljust(16)}: {str(tensor.shape).ljust(32)}, {str(tensor.dtype).ljust(16)}, device:{tensor.device}, grad:{str(tensor.requires_grad).ljust(5)}, lr:{self.lr_params[name]:.0e}")
        vprint(" ")

    def get_obj_ROI(self, indices):
        opt_obj = torch.stack([self.opt_obja, self.opt_objp], dim=-1)
        obj_ROI_grid_y = self.rpy_grid[None,:,:] + self.crop_pos[indices, None, None, 0]
        obj_ROI_grid_x = self.rpx_grid[None,:,:] + self.crop_pos[indices, None, None, 1]
        return opt_obj[:,:,obj_ROI_grid_y,obj_ROI_grid_x,:].permute(2,0,1,3,4,5)

    def get_obj_patches(self, indices):
        object_patches = self.get_obj_ROI(indices)
        if self.obj_preblur_std is None or self.obj_preblur_std == 0:
            return object_patches
        else:
            obj = object_patches.permute(5,0,1,2,3,4)
            obj_shape = obj.shape
            obj = obj.reshape(-1, obj_shape[-2], obj_shape[-1])
            return gaussian_blur(obj, kernel_size=5, sigma=self.obj_preblur_std).reshape(obj_shape).permute(1,2,3,4,5,0)

    def get_probes(self, indices):
        probe = self.get_complex_probe_view()
        if self.shift_probes:
            return imshift_batch(probe, shifts=self.opt_probe_pos_shifts[indices], grid=self.shift_probes_grid)
        else:
            return torch.broadcast_to(probe, (indices.shape[0], *probe.shape))

    def get_propagators(self, indices):
        tilt_obj = self.tilt_obj
        global_tilt = (self.opt_obj_tilts.shape[0] == 1)
        change_tilt = (self.lr_params['obj_tilts'] != 0)
        change_thickness = self.change_thickness
        
        dz = self.opt_slice_thickness
        Kz = self.Kz
        Ky, Kx = self.propagator_grid
        
        tilts = self.opt_obj_tilts if global_tilt else self.opt_obj_tilts[indices]
        tilts_y = tilts[:,0,None,None] / 1e3
        tilts_x = tilts[:,1,None,None] / 1e3
                
        if tilt_obj and change_thickness:
            H_opt_dz = torch_phasor(dz * Kz)
            return H_opt_dz * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
        elif tilt_obj and not change_thickness:
            if change_tilt:
                return self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
            else:
                return self.H_fixed_tilts_full if global_tilt else self.H_fixed_tilts_full[indices]
        elif not tilt_obj and change_thickness: 
            return torch_phasor(dz * Kz)[None,]
        else: 
            return self.H[None,]

    def get_forward_meas(self, object_patches, probes, propagators):
        dp_fwd = multislice_forward_model_vec_all(object_patches, probes, propagators, omode_occu=self.omode_occu)
        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=self.detector_blur_std)
        return dp_fwd
    
    def get_measurements(self, indices=None):
        measurements = self.measurements
        if indices is not None:
            measurements = self.measurements[indices]
            if self.meas_padded is not None:
                pad_h1, pad_h2, pad_w1, pad_w2 = self.meas_padded_idx
                canvas = torch.zeros((measurements.shape[0], *self.meas_padded.shape[-2:]), dtype=measurements.dtype, device=self.device)
                canvas += self.meas_padded
                canvas[..., pad_h1:pad_h2, pad_w1:pad_w2] = measurements
                measurements = canvas
            if self.meas_scale_factors is not None:
                scale_factor = tuple(self.meas_scale_factors)
                if any(factor != 1 for factor in scale_factor):
                    measurements = torch.nn.functional.interpolate(measurements[None,], scale_factor=scale_factor, mode='bilinear')[0]
                    measurements = measurements / prod(scale_factor)
        return measurements

    def clear_cache(self):
        self._current_object_patches = None

    def forward(self, indices):
        """
        The core forward operator A(x).
        Input: indices (representing latent variable selection)
        Output: Predicted diffraction patterns (y_pred)
        """
        object_patches = self.get_obj_patches(indices)
        probes = self.get_probes(indices)
        propagators = self.get_propagators(indices)
        dp_fwd = self.get_forward_meas(object_patches, probes, propagators)
        
        # Cache for loss calculation
        self._current_object_patches = object_patches
        return dp_fwd

def forward_operator(model: PtychoAD, indices: torch.Tensor):
    """
    Wrapper for the model's forward pass.
    """
    return model(indices)

# ==============================================================================
# 3. Inversion Loop
# ==============================================================================

def run_inversion(data_ctx: dict):
    """
    Performs the optimization loop.
    """
    params = data_ctx['params']
    device = data_ctx['device']
    initializer = data_ctx['initializer']
    logger = data_ctx['logger']
    accelerator = data_ctx['accelerator']
    
    verbose = not params['recon_params']['if_quiet']
    
    # 1. Initialize Loss and Constraints
    vprint("### Initializing loss function ###")
    loss_fn = CombinedLoss(params['loss_params'], device=device)
    
    vprint("### Initializing constraint function ###")
    constraint_fn = CombinedConstraint(params['constraint_params'], device=device, verbose=verbose)

    # 2. Instantiate the Model (Forward Operator)
    model = PtychoAD(initializer.init_variables, params['model_params'], device=device, verbose=verbose)
    
    # 3. Create Optimizer
    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)

    # 4. Prepare Data Loaders / Batches
    use_acc_device = (device is None and accelerator is not None)
    
    if not use_acc_device:
        indices, batches, output_path = prepare_recon(model, initializer, params)
    else:
        # Multi-GPU / Accelerate logic
        if params['model_params']['optimizer_params']['name'] == 'LBFGS' and accelerator.num_processes > 1:
            vprint("WARNING: LBFGS not supported for multiGPU. Switching to Adam.")
            params['model_params']['optimizer_params']['name'] = 'Adam'
            model.optimizer_params['name'] = 'Adam'
            optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
        
        params['recon_params']['GROUP_MODE'] = 'random'
        indices, batches, output_path = prepare_recon(model, initializer, params)
        ds = IndicesDataset(indices)
        dl = torch.utils.data.DataLoader(ds, batch_size=params['recon_params']['BATCH_SIZE']['size'], shuffle=True)
        batches = accelerator.prepare(dl)
        model, optimizer = accelerator.prepare(model, optimizer)

    # 5. Run Optimization Loop
    # Note: recon_loop internally calls model(indices), which is our forward_operator
    if logger is not None and logger.flush_file:
        logger.flush_to_file(log_dir=output_path)

    recon_loop(
        model, initializer, params, optimizer, 
        loss_fn, constraint_fn, indices, batches, output_path, 
        acc=accelerator
    )

    return {
        "model": model,
        "optimizer": optimizer,
        "output_path": output_path,
        "params": params,
        "logger": logger
    }

# ==============================================================================
# 4. Evaluation
# ==============================================================================

def evaluate_results(result_ctx: dict):
    """
    Evaluates and finalizes the reconstruction.
    """
    model = result_ctx['model']
    logger = result_ctx['logger']
    
    # In a real scenario, we might calculate PSNR/SSIM here if ground truth exists.
    # For this specific codebase, the evaluation is mostly logging completion and saving.
    
    vprint("### Evaluation / Finalization ###")
    
    if logger is not None and logger.flush_file:
        logger.close()
    
    if dist.is_initialized():
        dist.destroy_process_group()

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == '__main__':
    # Hardcoded path as per original script requirement
    PARAMS_FILE = "PSO_reconstruct.yml"
    
    # 1. Load Data
    data_context = load_and_preprocess_data(PARAMS_FILE, gpuid=0)
    
    # 2. Run Inversion (Implicitly calls forward_operator)
    results = run_inversion(data_context)
    
    # 3. Evaluate
    evaluate_results(results)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")