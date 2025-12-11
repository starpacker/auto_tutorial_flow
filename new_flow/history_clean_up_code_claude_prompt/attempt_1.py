import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.fft import fft2, ifft2
from torchvision.transforms.functional import gaussian_blur
from math import prod
from copy import deepcopy

# Import external ptyrad utilities as per the original script
from ptyrad.utils import print_system_info, set_gpu_device, CustomLogger, vprint, time_sync, torch_phasor, imshift_batch
from ptyrad.load import load_params as ptyrad_load_params
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss
from ptyrad.constraints import CombinedConstraint
from ptyrad.reconstruction import create_optimizer, prepare_recon, recon_loop, IndicesDataset, parse_sec_to_time_str
from ptyrad.forward import multislice_forward_model_vec_all

# -----------------------------------------------------------------------------
# 1. Load and Preprocess Data
# -----------------------------------------------------------------------------
def load_and_preprocess_data(params_path, gpuid=0):
    """
    Loads parameters, initializes system info, sets device, and prepares initial variables.
    """
    # Initialize Logger
    logger = CustomLogger(log_file='ptyrad_log.txt', log_dir='auto', prefix_time='datetime', show_timestamp=True)
    
    print_system_info()
    
    # Load Parameters
    # We use the provided helper functions from the original script context to ensure compatibility
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"The specified file '{params_path}' does not exist.")
    
    # Using ptyrad's load_params which handles validation internally
    params = ptyrad_load_params(params_path, validate=True)
    
    # Set Device
    device = set_gpu_device(gpuid=gpuid)
    
    # Initialize Variables (Physics/Geometry setup)
    # The Initializer class handles loading .mat files or generating synthetic data based on params
    vprint("### Initializing Initializer ###")
    initializer = Initializer(params['init_params'], seed=None)
    init_variables = initializer.init_all().init_variables
    vprint(" ")

    # Package everything needed for the inversion
    data = {
        'params': params,
        'device': device,
        'logger': logger,
        'init_variables': init_variables,
        'initializer_obj': initializer # Kept for reference in recon_loop if needed
    }
    
    return data

# -----------------------------------------------------------------------------
# 2. Forward Operator (Physics Model)
# -----------------------------------------------------------------------------
class PtychoAD(torch.nn.Module):
    """
    Main optimization class for ptychographic reconstruction using automatic differentiation (AD).
    Represents the physical model A in y = A(x).
    """

    def __init__(self, init_variables, model_params, device='cuda', verbose=True):
        super(PtychoAD, self).__init__()
        with torch.no_grad():
            
            vprint('### Initializing PtychoAD model ###', verbose=verbose)
            
            # Setup model behaviors
            self.device                 = device
            self.verbose                = verbose
            self.detector_blur_std      = model_params['detector_blur_std']
            self.obj_preblur_std        = model_params['obj_preblur_std']
            
            if init_variables.get('on_the_fly_meas_padded', None) is not None:
                self.meas_padded        = torch.tensor(init_variables['on_the_fly_meas_padded'], dtype=torch.float32, device=device)
                self.meas_padded_idx    = torch.tensor(init_variables['on_the_fly_meas_padded_idx'], dtype=torch.int32, device=device)
            else:
                self.meas_padded        = None
            self.meas_scale_factors     = init_variables.get('on_the_fly_meas_scale_factors', None)

            # Parse the learning rate and start iter for optimizable tensors
            start_iter_dict = {}
            end_iter_dict = {}
            lr_dict = {}
            for key, params in model_params['update_params'].items():
                start_iter_dict[key] = params.get('start_iter')
                end_iter_dict[key] = params.get('end_iter')
                lr_dict[key] = params['lr']
            self.optimizer_params       = model_params['optimizer_params']
            self.start_iter             = start_iter_dict
            self.end_iter               = end_iter_dict
            self.lr_params              = lr_dict
            
            # Optimizable parameters
            self.opt_obja               = nn.Parameter(torch.abs(torch.tensor(init_variables['obj'],    device=device)).to(torch.float32))
            self.opt_objp               = nn.Parameter(torch.angle(torch.tensor(init_variables['obj'],  device=device)).to(torch.float32))
            self.opt_obj_tilts          = nn.Parameter(torch.tensor(init_variables['obj_tilts'],                dtype=torch.float32, device=device))
            self.opt_slice_thickness    = nn.Parameter(torch.tensor(init_variables['slice_thickness'],          dtype=torch.float32, device=device))
            self.opt_probe              = nn.Parameter(torch.view_as_real(torch.tensor(init_variables['probe'], dtype=torch.complex64, device=device))) 
            self.opt_probe_pos_shifts   = nn.Parameter(torch.tensor(init_variables['probe_pos_shifts'],         dtype=torch.float32, device=device))
            
            # Buffers are used during forward pass
            self.register_buffer      ('omode_occu',      torch.tensor(init_variables['omode_occu'],       dtype=torch.float32, device=device))
            self.register_buffer      ('H',               torch.tensor(init_variables['H'],                dtype=torch.complex64, device=device))
            self.register_buffer      ('measurements',    torch.tensor(init_variables['measurements'],     dtype=torch.float32, device=device))
            self.register_buffer      ('N_scan_slow',     torch.tensor(init_variables['N_scan_slow'],      dtype=torch.int32, device=device))
            self.register_buffer      ('N_scan_fast',     torch.tensor(init_variables['N_scan_fast'],      dtype=torch.int32, device=device))
            self.register_buffer      ('crop_pos',        torch.tensor(init_variables['crop_pos'],         dtype=torch.int32, device=device))
            self.register_buffer      ('slice_thickness', torch.tensor(init_variables['slice_thickness'],  dtype=torch.float32, device=device))
            self.register_buffer      ('dx',              torch.tensor(init_variables['dx'],               dtype=torch.float32, device=device))
            self.register_buffer      ('dk',              torch.tensor(init_variables['dk'],               dtype=torch.float32, device=device))
            self.register_buffer      ('lambd',           torch.tensor(init_variables['lambd'],            dtype=torch.float32, device=device))
            
            self.random_seed            = init_variables['random_seed']
            self.length_unit            = init_variables['length_unit']
            self.scan_affine            = init_variables['scan_affine']
            self.tilt_obj               = bool(self.lr_params['obj_tilts']        != 0 or torch.any(self.opt_obj_tilts))
            self.shift_probes           = bool(self.lr_params['probe_pos_shifts'] != 0)
            self.change_thickness       = bool(self.lr_params['slice_thickness']  != 0)
            self.probe_int_sum          = self.get_complex_probe_view().abs().pow(2).sum()
            self.loss_iters             = []
            self.iter_times             = []
            self.dz_iters               = []
            self.avg_tilt_iters         = []

            # Create grids for shifting
            self.create_grids()

            # Create a dictionary to store the optimizable tensors
            self.optimizable_tensors = {
                'obja'            : self.opt_obja,
                'objp'            : self.opt_objp,
                'obj_tilts'       : self.opt_obj_tilts,
                'slice_thickness' : self.opt_slice_thickness,
                'probe'           : self.opt_probe,
                'probe_pos_shifts': self.opt_probe_pos_shifts}
            self.create_optimizable_params_dict(self.lr_params, self.verbose)

            # Initialize propagator-related variables
            self.init_propagator_vars()
            
            # Initialize iteration numbers that require torch.compile
            self.init_compilation_iters()
            
            vprint('### Done initializing PtychoAD model ###', verbose=verbose)
            vprint(' ', verbose=verbose)
            
    def get_complex_probe_view(self):
        return torch.view_as_complex(self.opt_probe)
        
    def create_grids(self):
        """ Create the grids for shifting probes, selecting obj ROI, and Fresnel propagator in a vectorized approach """
        device = self.device # FIX: Use self.device explicitly
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
        koy, kox = torch.meshgrid(torch.fft.fftfreq(Noy, dtype=torch.float32, device=device),
                                  torch.fft.fftfreq(Nox, dtype=torch.float32, device=device), indexing='ij')
        self.shift_probes_grid = torch.stack([kpy, kpx], dim=0) 
        self.shift_object_grid = torch.stack([koy, kox], dim=0) 
    
    def create_optimizable_params_dict(self, lr_params, verbose=True):
        self.lr_params = lr_params
        self.optimizable_params = []
        for param_name, lr in lr_params.items():
            if param_name not in self.optimizable_tensors:
                raise ValueError(f"WARNING: '{param_name}' is not a valid parameter name.")
            else:
                self.optimizable_tensors[param_name].requires_grad = (lr != 0) and (self.start_iter[param_name] ==1) 
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
            if start_iter is not None and start_iter >= 1:
                compilation_iters.add(start_iter)
            if end_iter is not None and end_iter >= 1:
                compilation_iters.add(end_iter)
        self.compilation_iters = sorted(compilation_iters)
        
    def print_model_summary(self):
        vprint('### PtychoAD optimizable variables ###')
        for name, tensor in self.optimizable_tensors.items():
            vprint(f"{name.ljust(16)}: {str(tensor.shape).ljust(32)}, {str(tensor.dtype).ljust(16)}, device:{tensor.device}, grad:{str(tensor.requires_grad).ljust(5)}, lr:{self.lr_params[name]:.0e}")
        total_var = sum(tensor.numel() for _, tensor in self.optimizable_tensors.items() if tensor.requires_grad)
        vprint(" ")        
        vprint('### Optimizable variables statitsics ###')
        vprint(f"Total measurement values  : {self.measurements.numel():,d}")
        vprint(f"Total optimizing variables: {total_var:,d}")
        vprint(f"Overdetermined ratio      : {self.measurements.numel()/total_var:.2f}")
        vprint(" ")
    
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
        
        dz       = self.opt_slice_thickness
        Kz       = self.Kz 
        Ky, Kx   = self.propagator_grid
        
        if global_tilt:
            tilts = self.opt_obj_tilts 
        else: 
            tilts = self.opt_obj_tilts[indices] 
        tilts_y  = tilts[:,0,None,None] / 1e3 
        tilts_x  = tilts[:,1,None,None] / 1e3
                
        if tilt_obj and change_thickness:
            H_opt_dz = torch_phasor(dz * Kz) 
            return H_opt_dz * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))

        elif tilt_obj and not change_thickness:
            if change_tilt:
                return self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
            else:
                return self.H_fixed_tilts_full if global_tilt else self.H_fixed_tilts_full[indices]
        
        elif not tilt_obj and change_thickness: 
            H_opt_dz = torch_phasor(dz * Kz)
            return H_opt_dz[None,]
            
        else: 
            return self.H[None,]

    def get_forward_meas(self, object_patches, probes, propagators):
        dp_fwd = multislice_forward_model_vec_all(object_patches, probes, propagators, omode_occu=self.omode_occu)
        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=self.detector_blur_std)
        return dp_fwd
    
    def get_measurements(self, indices=None):
        measurements = self.measurements
        device       = self.device
        dtype        = measurements.dtype
        if self.meas_padded is not None:
            meas_padded  = self.meas_padded
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
    
    def clear_cache(self):
        self._current_object_patches = None    
        
    def forward(self, indices):
        """ 
        Doing the forward pass and get an output diffraction pattern for each input index.
        This is the core Forward Operator.
        """
        object_patches = self.get_obj_patches(indices)
        probes         = self.get_probes(indices)
        propagators    = self.get_propagators(indices)
        dp_fwd         = self.get_forward_meas(object_patches, probes, propagators)
        
        # Keep the object_patches for later object-specific loss
        self._current_object_patches = object_patches
        
        return dp_fwd

def forward_operator(model, indices):
    """
    Wrapper to call the model's forward method.
    INPUT: model (PtychoAD instance), indices (Tensor)
    OUTPUT: y_pred (Tensor)
    """
    return model(indices)

# -----------------------------------------------------------------------------
# 3. Run Inversion
# -----------------------------------------------------------------------------
def run_inversion(data, accelerator=None):
    """
    Performs the optimization/solver loop.
    """
    params = data['params']
    device = data['device']
    init_variables = data['init_variables']
    initializer_obj = data['initializer_obj']
    logger = data['logger']
    
    vprint("### Starting Inversion ###")
    
    # Initialize Loss
    vprint("### Initializing loss function ###")
    loss_params = params['loss_params']
    loss_fn = CombinedLoss(loss_params, device=device)
    
    # Initialize Constraint
    vprint("### Initializing constraint function ###")
    constraint_params = params['constraint_params']
    constraint_fn = CombinedConstraint(constraint_params, device=device, verbose=True)
    
    # Initialize Model (Forward Operator)
    # Note: PtychoAD is instantiated here using the data loaded in step 1
    model = PtychoAD(init_variables, params['model_params'], device=device, verbose=True)
    
    # Initialize Optimizer
    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
    
    # Prepare Data Loaders / Batches
    # Note: In the original code, prepare_recon handles dataset creation
    indices, batches, output_path = prepare_recon(model, initializer_obj.init, params)
    
    # Handle Accelerator (Multi-GPU) logic if provided
    if accelerator is not None:
        # Simplified logic based on original code snippet for accelerator
        ds = IndicesDataset(indices)
        dl = torch.utils.data.DataLoader(ds, batch_size=params['recon_params']['BATCH_SIZE']['size'], shuffle=True)
        batches = accelerator.prepare(dl)
        model, optimizer = accelerator.prepare(model, optimizer)
    
    # Run the Reconstruction Loop
    # The recon_loop internally calls model(indices), which is our forward_operator
    if logger is not None and logger.flush_file:
        logger.flush_to_file(log_dir=output_path)

    recon_loop(model, initializer_obj.init, params, optimizer, loss_fn, constraint_fn, indices, batches, output_path, acc=accelerator)
    
    return model

# -----------------------------------------------------------------------------
# 4. Evaluate Results
# -----------------------------------------------------------------------------
def evaluate_results(model, data):
    """
    Evaluates and saves results.
    """
    vprint("### Evaluating Results ###")
    
    # In the original code context, the model itself holds the reconstructed state.
    # The recon_loop typically saves intermediate and final results to disk.
    # Here we can perform final logging or specific metric calculations if needed.
    
    # Example: Print final statistics
    if hasattr(model, 'loss_iters') and len(model.loss_iters) > 0:
        vprint(f"Final Loss: {model.loss_iters[-1]:.6f}")
    
    # Clean up resources
    if data['logger'] is not None and data['logger'].flush_file:
        data['logger'].close()
        
    if dist.is_initialized():
        dist.destroy_process_group()

# -----------------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Configuration
    PARAMS_PATH = "PSO_reconstruct.yml"
    
    # 1. Load Data
    data_container = load_and_preprocess_data(PARAMS_PATH, gpuid=0)
    
    # 2 & 3. Run Inversion (Implicitly uses Forward Operator)
    # Note: accelerator is None for single GPU execution as per default in main
    reconstructed_model = run_inversion(data_container, accelerator=None)
    
    # 4. Evaluate
    evaluate_results(reconstructed_model, data_container)
    
    print("OPTIMIZATION_FINISHED_SUCCESSFULLY")