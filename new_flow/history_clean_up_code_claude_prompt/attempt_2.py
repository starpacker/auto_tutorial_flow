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

# --- Third-party / Custom Library Imports ---
# Assuming these are available in the environment as per the legacy code
from ptyrad.utils import (
    print_system_info, set_gpu_device, CustomLogger, vprint, 
    time_sync, imshift_batch, torch_phasor
)
from ptyrad.reconstruction import (
    create_optimizer, prepare_recon, parse_sec_to_time_str, 
    recon_loop, IndicesDataset
)
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss
from ptyrad.constraints import CombinedConstraint
from ptyrad.forward import multislice_forward_model_vec_all

# ==============================================================================
# 1. Data Loading and Preprocessing
# ==============================================================================

def load_and_preprocess_data(params_path, gpuid=0):
    """
    Loads configuration, initializes system resources, and prepares initial data structures.
    
    Returns:
        dict: A container with 'params', 'device', 'logger', 'initializer_obj', 
              'loss_fn', 'constraint_fn', 'accelerator'.
    """
    # 1. Logger and System Info
    logger = CustomLogger(log_file='ptyrad_log.txt', log_dir='auto', prefix_time='datetime', show_timestamp=True)
    print_system_info()

    # 2. Load Parameters
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"File '{params_path}' not found.")
    
    vprint("### Loading params file ###")
    with open(params_path, "r", encoding='utf-8') as file:
        params_dict = yaml.safe_load(file)
    
    # Normalize constraint params (Legacy support logic)
    if params_dict.get('constraint_params') is not None:
        c_params = params_dict['constraint_params']
        normalized = {}
        for name, p in c_params.items():
            freq = p.get("freq", None)
            normalized[name] = {
                "start_iter": p.get("start_iter", 1 if freq is not None else None),
                "step": p.get("step", freq if freq is not None else 1),
                "end_iter": p.get("end_iter", None),
                **{k: v for k, v in p.items() if k not in ("freq", "step", "start_iter", "end_iter")},
            }
        params_dict['constraint_params'] = normalized

    # Validate params using ptyrad (if available)
    try:
        from ptyrad.params import PtyRADParams
        params_dict = PtyRADParams(**params_dict).model_dump()
    except ImportError:
        pass # Fallback if ptyrad.params isn't available, use dict as is
    
    params_dict['params_path'] = params_path
    params = deepcopy(params_dict)

    # 3. Device Setup
    # Pass gpuid=None if using accelerator later, but here we follow legacy flow
    device = set_gpu_device(gpuid=gpuid) 
    
    # 4. Initialize Physics/Math Objects (Initializer, Loss, Constraints)
    # Note: Initializer in ptyrad usually handles loading the .mat files referenced in params
    vprint("### Initializing Initializer ###")
    # The Initializer class loads data from disk based on params['init_params']
    initializer_obj = Initializer(params['init_params'], seed=None)
    # CRITICAL FIX: The legacy code called .init_all() which returns the object itself or sets internal state.
    # We need to ensure the internal state (init_variables) is ready.
    initializer_obj.init_all() 
    
    vprint("### Initializing loss function ###")
    loss_fn = CombinedLoss(params['loss_params'], device=device)
    
    vprint("### Initializing constraint function ###")
    constraint_fn = CombinedConstraint(params['constraint_params'], device=device, verbose=True)

    return {
        "params": params,
        "device": device,
        "logger": logger,
        "initializer_obj": initializer_obj, # Contains the loaded data/tensors
        "loss_fn": loss_fn,
        "constraint_fn": constraint_fn,
        "accelerator": None # Placeholder if not using HuggingFace Accelerate
    }

# ==============================================================================
# 2. Forward Operator (Physics Model)
# ==============================================================================

class PtychoAD(torch.nn.Module):
    """
    The Forward Operator class. 
    Encapsulates y = A(x) where x is the object/probe and y is the diffraction pattern.
    """
    def __init__(self, init_variables, model_params, device='cuda', verbose=True):
        super(PtychoAD, self).__init__()
        with torch.no_grad():
            self.device = device
            self.verbose = verbose
            self.detector_blur_std = model_params['detector_blur_std']
            self.obj_preblur_std = model_params['obj_preblur_std']
            
            # Handle on-the-fly measurement adjustments
            if init_variables.get('on_the_fly_meas_padded', None) is not None:
                self.meas_padded = torch.tensor(init_variables['on_the_fly_meas_padded'], dtype=torch.float32, device=device)
                self.meas_padded_idx = torch.tensor(init_variables['on_the_fly_meas_padded_idx'], dtype=torch.int32, device=device)
            else:
                self.meas_padded = None
            self.meas_scale_factors = init_variables.get('on_the_fly_meas_scale_factors', None)

            # Optimizer config parsing
            start_iter_dict = {}
            end_iter_dict = {}
            lr_dict = {}
            for key, p in model_params['update_params'].items():
                start_iter_dict[key] = p.get('start_iter')
                end_iter_dict[key] = p.get('end_iter')
                lr_dict[key] = p['lr']
            self.optimizer_params = model_params['optimizer_params']
            self.start_iter = start_iter_dict
            self.end_iter = end_iter_dict
            self.lr_params = lr_dict
            
            # Optimizable Parameters (The 'x' in y=A(x))
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
            self.probe_int_sum = torch.view_as_complex(self.opt_probe).abs().pow(2).sum()

            self.create_grids()
            
            self.optimizable_tensors = {
                'obja': self.opt_obja, 'objp': self.opt_objp, 'obj_tilts': self.opt_obj_tilts,
                'slice_thickness': self.opt_slice_thickness, 'probe': self.opt_probe,
                'probe_pos_shifts': self.opt_probe_pos_shifts
            }
            self.create_optimizable_params_dict(self.lr_params, self.verbose)
            self.init_propagator_vars()
            self.init_compilation_iters()

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
        self.lr_params = lr_params
        self.optimizable_params = []
        for param_name, lr in lr_params.items():
            if param_name in self.optimizable_tensors:
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
        
        if global_tilt: tilts = self.opt_obj_tilts 
        else: tilts = self.opt_obj_tilts[indices] 
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
        The core forward operator call.
        """
        object_patches = self.get_obj_patches(indices)
        probes = self.get_probes(indices)
        propagators = self.get_propagators(indices)
        dp_fwd = self.get_forward_meas(object_patches, probes, propagators)
        self._current_object_patches = object_patches
        return dp_fwd

def forward_operator(model, indices):
    """
    Wrapper to satisfy the requirement of a standalone forward_operator function.
    In this architecture, the model class IS the forward operator state machine.
    """
    return model(indices)

# ==============================================================================
# 3. Inversion Loop
# ==============================================================================

def run_inversion(data_container, accelerator=None):
    """
    Performs the optimization loop.
    """
    params = data_container['params']
    device = data_container['device']
    initializer_obj = data_container['initializer_obj']
    loss_fn = data_container['loss_fn']
    constraint_fn = data_container['constraint_fn']
    
    # 1. Instantiate the Model (Forward Operator)
    # The Initializer object has an attribute 'init_variables' which contains the tensors
    model = PtychoAD(initializer_obj.init_variables, params['model_params'], device=device, verbose=True)
    
    # 2. Create Optimizer
    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
    
    # 3. Prepare Data Loaders / Batches
    # FIX: prepare_recon expects the Initializer object itself, not initializer_obj.init
    indices, batches, output_path = prepare_recon(model, initializer_obj, params)
    
    # Handle Accelerator (Multi-GPU) logic if present
    if accelerator is not None:
        # Override for accelerator
        params['recon_params']['GROUP_MODE'] = 'random'
        ds = IndicesDataset(indices)
        dl = torch.utils.data.DataLoader(ds, batch_size=params['recon_params']['BATCH_SIZE']['size'], shuffle=True)
        batches = accelerator.prepare(dl)
        model, optimizer = accelerator.prepare(model, optimizer)
    
    # 4. Run the Optimization Loop
    # Note: recon_loop internally calls model(indices) which is our forward_operator
    recon_loop(
        model, 
        initializer_obj, 
        params, 
        optimizer, 
        loss_fn, 
        constraint_fn, 
        indices, 
        batches, 
        output_path, 
        acc=accelerator
    )
    
    return model

# ==============================================================================
# 4. Evaluation
# ==============================================================================

def evaluate_results(model, data_container):
    """
    Evaluates and saves results.
    """
    logger = data_container['logger']
    
    # In the legacy code, results are saved implicitly during recon_loop via output_path.
    # Here we ensure logs are closed and final status is reported.
    
    if logger is not None and logger.flush_file:
        logger.close()
    
    # Clean up DDP if initialized
    if dist.is_initialized():
        dist.destroy_process_group()

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == '__main__':
    # 1. Load Data
    params_path = "PSO_reconstruct.yml"
    data = load_and_preprocess_data(params_path, gpuid=0)
    
    # 2. Run Inversion (Implicitly calls forward_operator)
    # Note: The legacy code checks for 'hypertune', but the prompt asks for a specific flow.
    # We assume standard reconstruction mode here based on the prompt's structure.
    if not data['params'].get('hypertune_params', {}).get('if_hypertune', False):
        res = run_inversion(data, accelerator=data['accelerator'])
        
        # 3. Evaluate
        evaluate_results(res, data)
        
        print("OPTIMIZATION_FINISHED_SUCCESSFULLY")
    else:
        # Fallback for hypertune mode if strictly necessary, though prompt implies single run
        print("Hypertune mode detected - skipping standard inversion flow.")