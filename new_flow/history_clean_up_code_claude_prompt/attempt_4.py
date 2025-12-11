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

# --- Mocking/Importing Dependencies based on context ---
# Since I cannot import the actual 'ptyrad' library in this environment, 
# I will assume the imports provided in the input code exist in the user's environment.
# The refactoring focuses on structure.
from ptyrad.utils import vprint, time_sync, print_system_info, set_gpu_device, CustomLogger, torch_phasor, imshift_batch
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss
from ptyrad.constraints import CombinedConstraint
from ptyrad.reconstruction import create_optimizer, prepare_recon, parse_sec_to_time_str, recon_loop, IndicesDataset
from ptyrad.forward import multislice_forward_model_vec_all
from ptyrad.params import PtyRADParams

# --- 1. Data Loading and Preprocessing ---

def load_and_preprocess_data(params_path="PSO_reconstruct.yml"):
    """
    Loads data from disk/arguments and returns preprocessed tensors/arrays/objects.
    """
    print_system_info()
    
    # 1. Load YAML
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"File '{params_path}' not found.")
    
    vprint("### Loading params file ###")
    with open(params_path, "r", encoding='utf-8') as file:
        params_dict = yaml.safe_load(file)
    
    # 2. Normalize constraints (Legacy support logic)
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

    # 3. Validate Params
    vprint("validate = True: Filling defaults and validating the params file...")
    params_dict = PtyRADParams(**params_dict).model_dump()
    params_dict['params_path'] = params_path

    # 4. Setup Device and Logger
    logger = CustomLogger(log_file='ptyrad_log.txt', log_dir='auto', prefix_time='datetime', show_timestamp=True)
    device = set_gpu_device(gpuid=0)

    # 5. Initialize Data (Measurements, Probe, Object guesses)
    # The Initializer class handles loading .mat files and preparing initial tensors
    vprint("### Initializing Initializer ###")
    init_obj = Initializer(params_dict['init_params'], seed=None).init_all()
    
    # 6. Initialize Loss and Constraints
    loss_fn = CombinedLoss(params_dict['loss_params'], device=device)
    constraint_fn = CombinedConstraint(params_dict['constraint_params'], device=device, verbose=True)

    # Pack everything needed for the inversion
    data_package = {
        "params": params_dict,
        "device": device,
        "logger": logger,
        "init_obj": init_obj,
        "loss_fn": loss_fn,
        "constraint_fn": constraint_fn
    }
    
    return data_package

# --- 2. Forward Operator ---

def forward_operator(model_state, indices):
    """
    Represents the physical model A in y = A(x).
    INPUT: model_state (PtychoAD object containing latent variables), indices (batch indices)
    OUTPUT: dp_fwd (Predicted diffraction patterns), object_patches (for regularization)
    
    STRICTLY NO data loading inside here. Pure math only.
    """
    # This logic extracts the pure math components from the PtychoAD.forward method
    
    # 1. Get Object Patches (Cropping & Pre-blur)
    object_patches = model_state.get_obj_patches(indices)
    
    # 2. Get Probes (Shifting if necessary)
    probes = model_state.get_probes(indices)
    
    # 3. Get Propagators (Fresnel propagation kernels)
    propagators = model_state.get_propagators(indices)
    
    # 4. Multislice Physics Calculation
    # y = |F(P * O)|²
    dp_fwd = multislice_forward_model_vec_all(
        object_patches, 
        probes, 
        propagators, 
        omode_occu=model_state.omode_occu
    )
    
    # 5. Detector Blur (Physical effect)
    if model_state.detector_blur_std is not None and model_state.detector_blur_std != 0:
        dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=model_state.detector_blur_std)
        
    return dp_fwd, object_patches

# --- 3. Inversion Loop ---

class PtychoAD(torch.nn.Module):
    """
    Optimizable model wrapper. 
    This class holds the state (x) but delegates the physics calculation to forward_operator.
    """
    def __init__(self, init_variables, model_params, device='cuda', verbose=True):
        super(PtychoAD, self).__init__()
        # ... (Initialization logic copied from original code to setup tensors) ...
        with torch.no_grad():
            self.device = device
            self.verbose = verbose
            self.detector_blur_std = model_params['detector_blur_std']
            self.obj_preblur_std = model_params['obj_preblur_std']
            
            # Measurements setup
            if init_variables.get('on_the_fly_meas_padded', None) is not None:
                self.meas_padded = torch.tensor(init_variables['on_the_fly_meas_padded'], dtype=torch.float32, device=device)
                self.meas_padded_idx = torch.tensor(init_variables['on_the_fly_meas_padded_idx'], dtype=torch.int32, device=device)
            else:
                self.meas_padded = None
            self.meas_scale_factors = init_variables.get('on_the_fly_meas_scale_factors', None)

            # Optimizer params setup
            start_iter_dict = {}
            end_iter_dict = {}
            lr_dict = {}
            for key, params in model_params['update_params'].items():
                start_iter_dict[key] = params.get('start_iter')
                end_iter_dict[key] = params.get('end_iter')
                lr_dict[key] = params['lr']
            self.optimizer_params = model_params['optimizer_params']
            self.start_iter = start_iter_dict
            self.end_iter = end_iter_dict
            self.lr_params = lr_dict
            
            # Optimizable parameters (Latent Variables x)
            self.opt_obja = nn.Parameter(torch.abs(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
            self.opt_objp = nn.Parameter(torch.angle(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
            self.opt_obj_tilts = nn.Parameter(torch.tensor(init_variables['obj_tilts'], dtype=torch.float32, device=device))
            self.opt_slice_thickness = nn.Parameter(torch.tensor(init_variables['slice_thickness'], dtype=torch.float32, device=device))
            self.opt_probe = nn.Parameter(torch.view_as_real(torch.tensor(init_variables['probe'], dtype=torch.complex64, device=device)))
            self.opt_probe_pos_shifts = nn.Parameter(torch.tensor(init_variables['probe_pos_shifts'], dtype=torch.float32, device=device))
            
            # Buffers
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
            self.probe_int_sum = self.get_complex_probe_view().abs().pow(2).sum()
            self.loss_iters = []
            self.iter_times = []
            self.dz_iters = []
            self.avg_tilt_iters = []

            self.create_grids()

            self.optimizable_tensors = {
                'obja': self.opt_obja,
                'objp': self.opt_objp,
                'obj_tilts': self.opt_obj_tilts,
                'slice_thickness': self.opt_slice_thickness,
                'probe': self.opt_probe,
                'probe_pos_shifts': self.opt_probe_pos_shifts
            }
            self.create_optimizable_params_dict(self.lr_params, self.verbose)
            self.init_propagator_vars()
            self.init_compilation_iters()
            self._current_object_patches = None

    # --- Helper methods required by PtychoAD state management ---
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
            start_iter = self.start_iter.get(param_name)
            end_iter = self.end_iter.get(param_name)
            if start_iter is not None and start_iter >= 1: compilation_iters.add(start_iter)
            if end_iter is not None and end_iter >= 1: compilation_iters.add(end_iter)
        self.compilation_iters = sorted(compilation_iters)

    def print_model_summary(self):
        vprint('### PtychoAD optimizable variables ###')
        for name, tensor in self.optimizable_tensors.items():
            vprint(f"{name.ljust(16)}: {str(tensor.shape).ljust(32)}, {str(tensor.dtype).ljust(16)}, device:{tensor.device}, grad:{str(tensor.requires_grad).ljust(5)}, lr:{self.lr_params[name]:.0e}")

    def get_obj_ROI(self, indices):
        opt_obj = torch.stack([self.opt_obja, self.opt_objp], dim=-1)
        obj_ROI_grid_y = self.rpy_grid[None,:,:] + self.crop_pos[indices, None, None, 0]
        obj_ROI_grid_x = self.rpx_grid[None,:,:] + self.crop_pos[indices, None, None, 1]
        object_roi = opt_obj[:,:,obj_ROI_grid_y,obj_ROI_grid_x,:].permute(2,0,1,3,4,5)
        return object_roi

    def get_obj_patches(self, indices):
        object_patches = self.get_obj_ROI(indices)
        if self.obj_preblur_std is None or self.obj_preblur_std == 0:
            return object_patches
        else:
            obj = object_patches.permute(5,0,1,2,3,4)
            obj_shape = obj.shape
            obj = obj.reshape(-1, obj_shape[-2], obj_shape[-1])
            object_patches = gaussian_blur(obj, kernel_size=5, sigma=self.obj_preblur_std).reshape(obj_shape).permute(1,2,3,4,5,0)
            return object_patches

    def get_probes(self, indices):
        probe = self.get_complex_probe_view()
        if self.shift_probes:
            probes = imshift_batch(probe, shifts = self.opt_probe_pos_shifts[indices], grid = self.shift_probes_grid)
        else:
            probes = torch.broadcast_to(probe, (indices.shape[0], *probe.shape))
        return probes

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
            if change_tilt: return self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
            else: return self.H_fixed_tilts_full if global_tilt else self.H_fixed_tilts_full[indices]
        elif not tilt_obj and change_thickness: 
            H_opt_dz = torch_phasor(dz * Kz)
            return H_opt_dz[None,]
        else: 
            return self.H[None,]

    def get_measurements(self, indices=None):
        measurements = self.measurements
        device = self.device
        dtype = measurements.dtype
        if self.meas_padded is not None:
            meas_padded = self.meas_padded
            meas_padded_idx = self.meas_padded_idx
            pad_h1, pad_h2, pad_w1, pad_w2 = meas_padded_idx
        scale_factor = tuple(self.meas_scale_factors) if self.meas_scale_factors is not None else None
        if indices is not None:
            measurements = self.measurements[indices]
            if self.meas_padded is not None:
                canvas = torch.zeros((measurements.shape[0], *meas_padded.shape[-2:]), dtype=dtype, device=device)
                canvas += meas_padded
                canvas[..., pad_h1:pad_h2, pad_w1:pad_w2] = measurements
                measurements = canvas
            if self.meas_scale_factors is not None and any(factor != 1 for factor in scale_factor):
                measurements = torch.nn.functional.interpolate(measurements[None,], scale_factor=scale_factor, mode='bilinear')[0]
                measurements = measurements / prod(scale_factor)
        else:
            measurements = self.measurements
        return measurements

    def forward(self, indices):
        """
        The forward method required by PyTorch modules.
        CRITICAL: This calls the standalone forward_operator.
        """
        dp_fwd, object_patches = forward_operator(self, indices)
        
        # Cache for loss calculation
        self._current_object_patches = object_patches
        return dp_fwd

def run_inversion(data_package):
    """
    Performs the optimization/solver loop.
    """
    params = data_package['params']
    device = data_package['device']
    init_obj = data_package['init_obj']
    loss_fn = data_package['loss_fn']
    constraint_fn = data_package['constraint_fn']
    logger = data_package['logger']

    vprint(f"### Starting the PtyRADSolver in reconstruct mode ###")
    start_t = time_sync()

    # 1. Instantiate Model (Latent Variables)
    model = PtychoAD(init_obj.init_variables, params['model_params'], device=device, verbose=True)
    
    # 2. Create Optimizer
    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)

    # 3. Prepare Batches
    indices, batches, output_path = prepare_recon(model, init_obj, params)
    
    if logger is not None and logger.flush_file:
        logger.flush_to_file(log_dir=output_path)

    # 4. Run Optimization Loop
    # Note: recon_loop inside ptyrad calls model(indices), which calls model.forward(), 
    # which calls our defined forward_operator().
    recon_loop(model, init_obj, params, optimizer, loss_fn, constraint_fn, indices, batches, output_path, acc=None)

    end_t = time_sync()
    solver_t = end_t - start_t
    time_str = "" if solver_t < 60 else f", or {parse_sec_to_time_str(solver_t)}"
    vprint(f"### The PtyRADSolver is finished in {solver_t:.3f} sec{time_str} ###")

    if logger is not None and logger.flush_file:
        logger.close()
        
    if dist.is_initialized():
        dist.destroy_process_group()

    return model

# --- 4. Evaluation ---

def evaluate_results(model, data_package):
    """
    Calculates metrics and saves results.
    """
    # In the original code, results are stored in the model object or saved to disk during recon_loop.
    # Here we can add specific evaluation logic if needed.
    vprint("### Evaluating Results ###")
    
    # Example: Access final object
    final_obj_amp = model.opt_obja.detach().cpu().numpy()
    final_obj_phase = model.opt_objp.detach().cpu().numpy()
    
    vprint(f"Final Object Amplitude Shape: {final_obj_amp.shape}")
    vprint(f"Final Object Phase Shape: {final_obj_phase.shape}")
    
    # Since recon_loop handles saving, we just confirm completion here.
    vprint("Results evaluation complete.")

# --- Main Execution ---

if __name__ == '__main__':
    # 1. Load Data
    data = load_and_preprocess_data("PSO_reconstruct.yml")
    
    # 2. Run Inversion (Implicitly calls forward_operator via model.forward)
    res_model = run_inversion(data)
    
    # 3. Evaluate
    evaluate_results(res_model, data)
    
    # 4. Success Signal
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")