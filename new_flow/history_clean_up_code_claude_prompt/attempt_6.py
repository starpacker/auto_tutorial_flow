import os
import sys
import yaml
import torch
import torch.distributed as dist
from copy import deepcopy
from math import prod

# --- Third-party / Library Imports ---
# Assuming ptyrad is installed or available in the python path
from ptyrad.utils import print_system_info, set_gpu_device, CustomLogger, vprint, time_sync, torch_phasor, imshift_batch
from ptyrad.params import PtyRADParams
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss
from ptyrad.constraints import CombinedConstraint
from ptyrad.reconstruction import create_optimizer, prepare_recon, recon_loop, IndicesDataset, parse_sec_to_time_str
from ptyrad.forward import multislice_forward_model_vec_all

# Torch imports
import torch.nn as nn
from torch.fft import fft2, ifft2
from torchvision.transforms.functional import gaussian_blur

# ==============================================================================
# 1. Data Loading and Preprocessing
# ==============================================================================

def load_and_preprocess_data(params_path="PSO_reconstruct.yml", gpuid=0):
    """
    Loads configuration, initializes system resources, and prepares initial data structures.
    Returns a dictionary context containing everything needed for the inversion loop.
    """
    # 1. Logger and System Info
    logger = CustomLogger(log_file='ptyrad_log.txt', log_dir='auto', prefix_time='datetime', show_timestamp=True)
    print_system_info()

    # 2. Load Parameters
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"File '{params_path}' not found.")

    vprint("### Loading params file ###")
    with open(params_path, "r", encoding='utf-8') as file:
        raw_params = yaml.safe_load(file)
    
    # Normalize constraint params (legacy support logic from original code)
    if raw_params.get('constraint_params') is not None:
        c_params = raw_params['constraint_params']
        normalized = {}
        for name, p in c_params.items():
            freq = p.get("freq", None)
            start_iter = p.get("start_iter", 1 if freq is not None else None)
            step = p.get("step", freq if freq is not None else 1)
            end_iter = p.get("end_iter", None)
            normalized[name] = {
                "start_iter": start_iter, "step": step, "end_iter": end_iter,
                **{k: v for k, v in p.items() if k not in ("freq", "step", "start_iter", "end_iter")}
            }
        raw_params['constraint_params'] = normalized

    # Validate via Pydantic model
    params = PtyRADParams(**raw_params).model_dump()
    params['params_path'] = params_path

    # 3. Device Setup
    device = set_gpu_device(gpuid=gpuid)

    # 4. Initialize Data (Measurements, Probe, Object, etc.)
    # The Initializer class handles loading .mat files or generating synthetic data
    vprint("### Initializing Initializer ###")
    seed = None # Could be passed as arg if needed
    initializer = Initializer(params['init_params'], seed=seed).init_all()
    init_vars = initializer.init_variables

    # 5. Initialize Loss and Constraints
    vprint("### Initializing loss function ###")
    loss_fn = CombinedLoss(params['loss_params'], device=device)
    
    vprint("### Initializing constraint function ###")
    constraint_fn = CombinedConstraint(params['constraint_params'], device=device, verbose=True)

    # Return a context dictionary
    data_context = {
        "params": params,
        "device": device,
        "logger": logger,
        "initializer": initializer,
        "init_vars": init_vars,
        "loss_fn": loss_fn,
        "constraint_fn": constraint_fn,
        "accelerator": None # Placeholder if accelerator is needed later
    }
    return data_context


# ==============================================================================
# 2. Forward Operator (Physics Model)
# ==============================================================================

class PtychoAD(torch.nn.Module):
    """
    The PyTorch Module representing the physical model A(x).
    This class encapsulates the forward_operator logic and optimizable parameters.
    """
    def __init__(self, init_variables, model_params, device='cuda', verbose=True):
        super(PtychoAD, self).__init__()
        with torch.no_grad():
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

            # Optimizer setup helpers
            self.optimizer_params = model_params['optimizer_params']
            self.start_iter = {}
            self.end_iter = {}
            self.lr_params = {}
            for key, p in model_params['update_params'].items():
                self.start_iter[key] = p.get('start_iter')
                self.end_iter[key] = p.get('end_iter')
                self.lr_params[key] = p['lr']

            # Optimizable Parameters
            self.opt_obja = nn.Parameter(torch.abs(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
            self.opt_objp = nn.Parameter(torch.angle(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
            self.opt_obj_tilts = nn.Parameter(torch.tensor(init_variables['obj_tilts'], dtype=torch.float32, device=device))
            self.opt_slice_thickness = nn.Parameter(torch.tensor(init_variables['slice_thickness'], dtype=torch.float32, device=device))
            self.opt_probe = nn.Parameter(torch.view_as_real(torch.tensor(init_variables['probe'], dtype=torch.complex64, device=device)))
            self.opt_probe_pos_shifts = nn.Parameter(torch.tensor(init_variables['probe_pos_shifts'], dtype=torch.float32, device=device))

            # Buffers (Fixed Physics Constants/Grids)
            self.register_buffer('omode_occu', torch.tensor(init_variables['omode_occu'], dtype=torch.float32, device=device))
            self.register_buffer('H', torch.tensor(init_variables['H'], dtype=torch.complex64, device=device))
            self.register_buffer('measurements', torch.tensor(init_variables['measurements'], dtype=torch.float32, device=device))
            self.register_buffer('crop_pos', torch.tensor(init_variables['crop_pos'], dtype=torch.int32, device=device))
            self.register_buffer('dx', torch.tensor(init_variables['dx'], dtype=torch.float32, device=device))
            self.register_buffer('lambd', torch.tensor(init_variables['lambd'], dtype=torch.float32, device=device))
            
            # Flags
            self.tilt_obj = bool(self.lr_params['obj_tilts'] != 0 or torch.any(self.opt_obj_tilts))
            self.shift_probes = bool(self.lr_params['probe_pos_shifts'] != 0)
            self.change_thickness = bool(self.lr_params['slice_thickness'] != 0)
            
            # Logging lists
            self.loss_iters = []
            self.iter_times = []

            # Grid Creation
            self._create_grids()
            
            # Optimizable Tensors Dict
            self.optimizable_tensors = {
                'obja': self.opt_obja, 'objp': self.opt_objp, 'obj_tilts': self.opt_obj_tilts,
                'slice_thickness': self.opt_slice_thickness, 'probe': self.opt_probe,
                'probe_pos_shifts': self.opt_probe_pos_shifts
            }
            self._setup_optimizer_params()
            self._init_propagator_vars()

            # Cache
            self._current_object_patches = None

    def _create_grids(self):
        probe = torch.view_as_complex(self.opt_probe)
        Npy, Npx = probe.shape[-2:]
        ygrid = (torch.arange(-Npy // 2, Npy // 2, device=self.device) + 0.5) / Npy
        xgrid = (torch.arange(-Npx // 2, Npx // 2, device=self.device) + 0.5) / Npx
        ky = torch.fft.ifftshift(2 * torch.pi * ygrid / self.dx)
        kx = torch.fft.ifftshift(2 * torch.pi * xgrid / self.dx)
        Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
        self.propagator_grid = torch.stack([Ky, Kx], dim=0)

        rpy, rpx = torch.meshgrid(torch.arange(Npy, dtype=torch.int32, device=self.device), 
                                  torch.arange(Npx, dtype=torch.int32, device=self.device), indexing='ij')
        self.rpy_grid = rpy
        self.rpx_grid = rpx
        
        kpy, kpx = torch.meshgrid(torch.fft.fftfreq(Npy, dtype=torch.float32, device=self.device),
                                  torch.fft.fftfreq(Npx, dtype=torch.float32, device=self.device), indexing='ij')
        self.shift_probes_grid = torch.stack([kpy, kpx], dim=0)

    def _setup_optimizer_params(self):
        self.optimizable_params = []
        for param_name, lr in self.lr_params.items():
            self.optimizable_tensors[param_name].requires_grad = (lr != 0) and (self.start_iter[param_name] == 1)
            if lr != 0:
                self.optimizable_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})

    def _init_propagator_vars(self):
        dz = self.opt_slice_thickness.detach()
        Ky, Kx = self.propagator_grid 
        tilts_y_full = self.opt_obj_tilts[:,0,None,None] / 1e3
        tilts_x_full = self.opt_obj_tilts[:,1,None,None] / 1e3
        self.H_fixed_tilts_full = self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y_full) + Kx * torch.tan(tilts_x_full)))
        self.k = 2 * torch.pi / self.lambd
        self.Kz = torch.sqrt(self.k ** 2 - Kx ** 2 - Ky ** 2)

    def get_obj_patches(self, indices):
        opt_obj = torch.stack([self.opt_obja, self.opt_objp], dim=-1)
        obj_ROI_grid_y = self.rpy_grid[None,:,:] + self.crop_pos[indices, None, None, 0]
        obj_ROI_grid_x = self.rpx_grid[None,:,:] + self.crop_pos[indices, None, None, 1]
        object_roi = opt_obj[:,:,obj_ROI_grid_y,obj_ROI_grid_x,:].permute(2,0,1,3,4,5)
        
        if self.obj_preblur_std is None or self.obj_preblur_std == 0:
            return object_roi
        else:
            obj = object_roi.permute(5,0,1,2,3,4)
            obj_shape = obj.shape
            obj = obj.reshape(-1, obj_shape[-2], obj_shape[-1])
            return gaussian_blur(obj, kernel_size=5, sigma=self.obj_preblur_std).reshape(obj_shape).permute(1,2,3,4,5,0)

    def get_probes(self, indices):
        probe = torch.view_as_complex(self.opt_probe)
        if self.shift_probes:
            return imshift_batch(probe, shifts=self.opt_probe_pos_shifts[indices], grid=self.shift_probes_grid)
        else:
            return torch.broadcast_to(probe, (indices.shape[0], *probe.shape))

    def get_propagators(self, indices):
        # Simplified logic for brevity, matching original intent
        dz = self.opt_slice_thickness
        Ky, Kx = self.propagator_grid
        
        global_tilt = (self.opt_obj_tilts.shape[0] == 1)
        tilts = self.opt_obj_tilts if global_tilt else self.opt_obj_tilts[indices]
        tilts_y = tilts[:,0,None,None] / 1e3
        tilts_x = tilts[:,1,None,None] / 1e3

        if self.tilt_obj and self.change_thickness:
            H_opt_dz = torch_phasor(dz * self.Kz)
            return H_opt_dz * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
        elif self.tilt_obj and not self.change_thickness:
            if self.lr_params['obj_tilts'] != 0:
                return self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
            else:
                return self.H_fixed_tilts_full if global_tilt else self.H_fixed_tilts_full[indices]
        elif not self.tilt_obj and self.change_thickness:
            return torch_phasor(dz * self.Kz)[None,]
        else:
            return self.H[None,]

    def get_measurements(self, indices=None):
        if indices is None: return self.measurements
        measurements = self.measurements[indices]
        # Simplified on-the-fly logic
        if self.meas_padded is not None:
            pad_h1, pad_h2, pad_w1, pad_w2 = self.meas_padded_idx
            canvas = torch.zeros((measurements.shape[0], *self.meas_padded.shape[-2:]), dtype=measurements.dtype, device=self.device)
            canvas += self.meas_padded
            canvas[..., pad_h1:pad_h2, pad_w1:pad_w2] = measurements
            measurements = canvas
        if self.meas_scale_factors is not None:
             measurements = torch.nn.functional.interpolate(measurements[None,], scale_factor=tuple(self.meas_scale_factors), mode='bilinear')[0]
             measurements = measurements / prod(self.meas_scale_factors)
        return measurements

    def clear_cache(self):
        self._current_object_patches = None

    def forward(self, indices):
        """
        The core forward operator: x -> y_pred
        Here 'x' is implicitly the optimizable parameters (obj, probe) stored in the class.
        'indices' selects the specific batch of data.
        """
        object_patches = self.get_obj_patches(indices)
        probes = self.get_probes(indices)
        propagators = self.get_propagators(indices)
        
        dp_fwd = multislice_forward_model_vec_all(object_patches, probes, propagators, omode_occu=self.omode_occu)
        
        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=self.detector_blur_std)
            
        self._current_object_patches = object_patches
        return dp_fwd

def forward_operator(model, indices):
    """
    Wrapper to satisfy the requirement of a standalone forward_operator function.
    In this AD framework, the model holds the state (x).
    """
    return model(indices)


# ==============================================================================
# 3. Run Inversion (Optimization Loop)
# ==============================================================================

def run_inversion(data_context):
    """
    Performs the optimization loop.
    """
    params = data_context['params']
    device = data_context['device']
    init_vars = data_context['init_vars']
    initializer = data_context['initializer']
    loss_fn = data_context['loss_fn']
    constraint_fn = data_context['constraint_fn']
    logger = data_context['logger']
    accelerator = data_context['accelerator']

    vprint(f"### Starting Reconstruction ###")

    # 1. Instantiate the Model (Physics + Variables)
    model = PtychoAD(init_vars, params['model_params'], device=device, verbose=True)

    # 2. Create Optimizer
    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)

    # 3. Prepare Data Loaders / Indices
    # Note: prepare_recon is a legacy helper from ptyrad that sets up indices/batches
    indices, batches, output_path = prepare_recon(model, initializer, params)

    # 4. Run the Optimization Loop
    # The recon_loop function in ptyrad internally calls model(indices) -> forward_operator
    if logger is not None and logger.flush_file:
        logger.flush_to_file(log_dir=output_path)

    start_t = time_sync()
    
    # This loop modifies 'model' in-place
    recon_loop(
        model, initializer, params, optimizer, 
        loss_fn, constraint_fn, indices, batches, 
        output_path, acc=accelerator
    )
    
    end_t = time_sync()
    solver_t = end_t - start_t
    vprint(f"### Solver finished in {solver_t:.3f} sec ###")

    if logger is not None and logger.flush_file:
        logger.close()

    if dist.is_initialized():
        dist.destroy_process_group()

    return model


# ==============================================================================
# 4. Evaluate Results
# ==============================================================================

def evaluate_results(model, data_context):
    """
    Evaluates the reconstructed model.
    """
    vprint("### Evaluating Results ###")
    
    # Fix for the TypeError: Check if loss_iters contains tuples or floats
    if len(model.loss_iters) > 0:
        final_loss = model.loss_iters[-1]
        
        # Handle case where loss might be a tuple (total_loss, component_losses...)
        if isinstance(final_loss, (tuple, list)):
            # Assuming the first element is the total loss
            val_to_print = final_loss[0]
        else:
            val_to_print = final_loss
            
        vprint(f"Final Loss: {val_to_print:.6f}")
    else:
        vprint("No loss recorded.")

    # Basic result extraction (placeholder for more complex metrics like PSNR/SSIM)
    # In a real scenario, you would compare model.opt_obja against ground truth if available.
    
    # Save results logic could go here, but PtyRAD's recon_loop usually handles intermediate saving.
    # We can explicitly save the final model state here if needed.
    output_dir = os.path.dirname(data_context['params']['params_path'])
    save_path = os.path.join(output_dir, "final_model.pt")
    # torch.save(model.state_dict(), save_path) # Optional
    vprint("Evaluation complete.")


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == '__main__':
    # 1. Load Data
    data_ctx = load_and_preprocess_data(params_path="PSO_reconstruct.yml", gpuid=0)

    # 2. Run Inversion (Implicitly calls forward_operator via the model)
    res_model = run_inversion(data_ctx)

    # 3. Evaluate
    evaluate_results(res_model, data_ctx)

    # 4. Success Message
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")