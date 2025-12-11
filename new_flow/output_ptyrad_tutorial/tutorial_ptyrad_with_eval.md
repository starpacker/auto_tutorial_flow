

Here is Section 1 of the scientific tutorial, based on the provided paper.

---

# Section 1: Task Background and Paper Contributions

## 1.1 The Challenge of High-Resolution Phase Retrieval

Electron ptychography has fundamentally transformed atomic-resolution imaging. Unlike conventional Scanning Transmission Electron Microscopy (STEM), which often relies on incoherent intensity measurements (such as Annular Dark Field), ptychography is a phase retrieval technique. It reconstructs the complex exit wave of the specimen by processing a series of diffraction patterns recorded as a focused electron probe scans across the sample.

The theoretical foundation of ptychography rests on the redundancy of data collected from overlapping illumination spots. As the probe scans positions $\boldsymbol{\rho}_j$, the detector records the far-field diffraction intensity $I_j(\boldsymbol{k})$. The core task is to solve an inverse problem: recovering the object transmission function $O(\boldsymbol{r})$ and the probe function $P(\boldsymbol{r})$ from these intensity measurements.

However, standard 2D ptychography faces significant limitations when applied to thick or strongly scattering samples due to dynamical scattering (multiple scattering events). To address this, **Multislice Electron Ptychography (MEP)** was developed. MEP models the sample not as a single phase screen, but as a series of thin discrete slices spaced along the optical axis ($z$).

Mathematically, the exit wave $\psi_{j}^{(n)}$ after passing through the $n$-th slice of the object at scan position $j$ is calculated iteratively. The wave propagates from slice $n-1$ to slice $n$ and then interacts with the transmission function of slice $n$, denoted as $O^{(n)}$. This process involves a Fresnel propagator $\mathcal{M}$:

$$ \psi_{j}^{(n)}(\boldsymbol{k}) = \mathcal{F} \left[ O_{j}^{(n)}(\boldsymbol{r}) \cdot \mathcal{F}^{-1} \left[ \mathcal{M}_{\Delta z}(\boldsymbol{k}) \cdot \psi_{j}^{(n-1)}(\boldsymbol{k}) \right] \right] $$

where $\mathcal{F}$ denotes the Fourier transform. The propagator $\mathcal{M}_{\Delta z}$ typically takes the form:

$$ \mathcal{M}_{\Delta z}(\boldsymbol{k}) = \exp\left[-i\pi\lambda|\boldsymbol{k}|^{2}\Delta z \right] $$

While MEP enables 3D reconstruction and depth sectioning, it introduces immense computational complexity. The forward model is non-linear and computationally expensive, requiring optimization over millions of parameters (voxel values of the object, probe modes, positions). Furthermore, experimental imperfections—such as sample tilt, partial coherence, and scan distortions—must be incorporated into the model to achieve atomic resolution, often requiring manual derivation of gradients for each new parameter.

## 1.2 Theoretical Contributions of PtyRAD

The paper introduces **PtyRAD** (Ptychographic Reconstruction with Automatic Differentiation) to address the rigidity and computational cost of existing frameworks. The core contribution lies in shifting from analytical gradient derivation to an **Automatic Differentiation (AD)** paradigm.

### 1.2.1 The AD-Based Forward Model
In traditional iterative engines (e.g., ePIE or standard gradient descent), the update rules are derived analytically. If a researcher wants to optimize a new physical parameter (e.g., sample mistilt $\boldsymbol{\theta}_j$), they must manually derive $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}_j}$ and implement the specific update code.

PtyRAD redefines the reconstruction as a computational graph within PyTorch. The forward physical model $f$ predicts the diffraction intensity based on a set of optimizable parameters:

$$ I_{j}^{\mathrm{model}}(\boldsymbol{k}) = f(P, O, \boldsymbol{\rho}_{j}, \boldsymbol{\theta}_{j}, \Delta z) $$

where:
*   $P$ is the mixed-state probe.
*   $O$ is the mixed-state object (decomposed into amplitude and phase).
*   $\boldsymbol{\rho}_{j}$ represents probe positions (correcting for scan noise).
*   $\boldsymbol{\theta}_{j}$ represents local object tilts.
*   $\Delta z$ is the slice thickness.

By utilizing AD, PtyRAD automatically computes the gradients of the loss function $\mathcal{L}$ with respect to *all* these parameters simultaneously via the chain rule during the backward pass. This enables the optimization of complex physical phenomena—such as tilted multislice propagation—without explicit analytical derivation.

### 1.2.2 Mixed-State Formalism and Vectorization
To account for partial coherence and finite source size, PtyRAD implements a mixed-state formalism. The total intensity is an incoherent sum of independent modes:

$$ I_{j}^{\mathrm{model}}(\boldsymbol{k}) = \sum_{m=1}^{M} \sum_{n=1}^{N} |\mathcal{F}[\psi_{j}^{(m)(n)}(\boldsymbol{r})]|^{2} $$

where $m$ indexes the probe modes and $n$ indexes the object modes. Theoretically, PtyRAD differs from previous implementations by employing a **fully vectorized execution** of these modes. Instead of looping through modes sequentially, the probe modes are broadcasted against the object tensor. This approach leverages the massive parallelism of GPUs more effectively than sequential processing, which is the primary driver behind the reported **6–12$\times$ speedup** per iteration compared to other packages.

### 1.2.3 Real-Space Depth Regularization ($r_z$ filter)
A specific theoretical contribution of this work is a new method for regularizing depth in multislice reconstruction. 

Standard MEP uses a "missing wedge" filter in Fourier space ($k_z$-filter) to suppress artifacts arising from the poor information transfer along the optical axis. The filter typically takes the form:
$$ W(\boldsymbol{k}) = 1 - \frac{2}{\pi}\tan^{-1}\left(\frac{\beta^{2}|k_{z}|^{2}}{k_{x}^{2}+k_{y}^{2}+\epsilon}\right) $$
However, filtering in Fourier space imposes periodic boundary conditions, leading to "wrap-around" artifacts where intensity from the bottom slice bleeds into the top slice. This is particularly problematic for non-periodic samples like twisted 2D bilayers.

PtyRAD introduces a **Real-Space Depth Regularization ($r_z$ filter)**. Instead of modifying the reciprocal space representation, the object $O(\boldsymbol{r})$ is convolved with a 1D Gaussian kernel $G_{\sigma_z}$ along the $z$-axis:

$$ O^{\prime}(\boldsymbol{r}) = O(\boldsymbol{r}) * G_{\sigma_{z}}(r_{z}) $$

This enforces continuity and suppresses noise along the depth direction *without* enforcing periodicity, effectively mitigating wrap-around artifacts in vertical heterostructures.

## 1.3 Summary of Contributions
In summary, the theoretical and practical contributions of PtyRAD include:
1.  **Unified AD Framework:** Enabling simultaneous optimization of sample thickness, local tilts, and mixed states without manual gradient derivation.
2.  **Computational Efficiency:** Achieving up to 24$\times$ speedup via vectorized tensor operations and multi-GPU support.
3.  **Enhanced Regularization:** Introducing the real-space $r_z$ filter to resolve depth artifacts in non-periodic samples.
4.  **Bayesian Optimization:** Integrating a hyperparameter tuning workflow (using Optuna) to automate the selection of critical experimental parameters like defocus and learning rates.

**2. Observation Data Introduction and Acquisition Methods**

Following the establishment of regularization and optimization workflows, we must define the physical framework that generates the observation data. In computational imaging, particularly ptychography, the reconstruction quality is intrinsically bounded by the accuracy of the forward model—the mathematical description of how the electron probe interacts with the specimen to form a diffraction pattern.

This section details the theoretical basis of the PtyRAD framework, specifically the mixed-state multislice formalism. This model transforms the physical acquisition process into a differentiable computational graph, allowing us to utilize Automatic Differentiation (AD) to optimize not just the object potential, but also experimental parameters such as probe aberrations, local tilts, and sample thickness.

### 2.1 The Forward Physical Model

In Scanning Transmission Electron Microscopy (STEM) ptychography, a focused electron probe scans across a specimen, and a 2D diffraction pattern is recorded at every scan position. The forward model $f$ predicts the modeled diffraction intensity $I_{j}^{\mathrm{model}}$ at a specific probe position $\boldsymbol{\rho}_{j}$ based on the interaction between the probe $P$, the object $O$, and various geometric parameters:

$$I_{j}^{\mathrm{model}}(\boldsymbol{k}) = f(P, O, \boldsymbol{\rho}_{j}, \boldsymbol{\theta}_{j}, \Delta z)$$

Here, $\boldsymbol{k}$ represents the reciprocal space coordinates, $\boldsymbol{\theta}_{j}$ represents the local object misorientation (tilt), and $\Delta z$ represents the slice thickness. A critical innovation in PtyRAD is that parameters such as $\boldsymbol{\theta}_{j}$ and $\Delta z$ are treated as continuously optimizing variables within the AD framework, rather than static constants.

### 2.2 Mixed-State Multislice Formalism

To account for partial coherence in the illumination and dynamic excitations in the sample, PtyRAD employs a mixed-state formalism. Both the probe $P$ and the object $O$ are represented as mixtures of mutually incoherent states (modes). The total intensity is the incoherent sum of the intensities generated by the interaction of each probe mode $m$ with each object mode $n$:

$$I_{j}^{\mathrm{model}}(\boldsymbol{k}) = \sum_{m=1}^{M}\sum_{n=1}^{N} \left| \mathcal{F} \left[ \psi_{j}^{(m)(n)}(\boldsymbol{r}) \right] \right|^{2}$$

where $M$ and $N$ are the number of probe and object modes, respectively, $\mathcal{F}$ denotes the Fourier transform propagating the exit wave to the far-field detector, and $\psi_{j}^{(m)(n)}$ is the exit wave function in real space $\boldsymbol{r}$.

**The Multislice Propagation**
For thick samples where the projection approximation fails due to multiple scattering, the interaction is modeled using the multislice algorithm. The object $O$ is divided into $N_{z}^{O}$ discrete slices along the optical axis. The electron wave propagates sequentially through these slices.

The calculation for a specific scan position $j$ involves:
1.  **Cropping and Shifting:** An object patch $O_{j}^{(n)}$ is cropped from the full object array centered at $\boldsymbol{\rho}_{j}$. To handle sub-pixel scanning precision—critical for high-resolution reconstruction—a shift operator $S_{\Delta\boldsymbol{r}_{j}}$ is applied to the probe, where $\Delta\boldsymbol{r}_{j}$ represents the fractional pixel shift.
2.  **Sequential Interaction:** The wave transmits through an object slice and then propagates to the next slice via a propagator function $\mathcal{M}$.

The exit wave $\psi_{j}^{(m)(n)}$ after passing through all layers is given by the recursive interaction:

$$\psi_{j}^{(m)(n)} = O_{j,N^{O}}^{(n)} \cdots \mathcal{F}^{-1} \left[ \mathcal{M}_{\boldsymbol{\theta}_{j},\Delta z} \mathcal{F} \left[ O_{j,1}^{(n)} P_{j}^{(m)} \right] \right]$$

**Differentiable Propagator**
The multislice propagator $\mathcal{M}_{\boldsymbol{\theta}_{j},\Delta z}$ encapsulates the phase accumulation due to propagation through free space over distance $\Delta z$, modified by the local specimen tilt $\boldsymbol{\theta}_{j} = (\theta_{j,x}, \theta_{j,y})$:

$$\mathcal{M}_{\boldsymbol{\theta}_{j},\Delta z}(\boldsymbol{k}) = \exp\left[-i\pi\lambda|\boldsymbol{k}|^{2}\Delta z + 2\pi i\Delta z(k_{x}\tan\theta_{j,x} + k_{y}\tan\theta_{j,y})\right]$$

Because this propagator is defined analytically within the AD framework, the gradients with respect to thickness $\Delta z$ and tilt $\boldsymbol{\theta}_{j}$ are automatically computed during backpropagation, enabling the refinement of these physical parameters alongside the phase retrieval.

### 2.3 Optimization Objectives and Loss Functions

The goal of the reconstruction is to minimize the discrepancy between the experimentally measured diffraction patterns, $I^{\mathrm{meas}}$, and the forward model prediction, $I^{\mathrm{model}}$. The choice of loss function $\mathcal{L}$ dictates how the optimizer handles noise statistics.

**Noise Statistics**
PtyRAD implements negative log-likelihood functions tailored to specific noise environments:

1.  **Gaussian Loss ($\mathcal{L}_{\mathrm{Gaussian}}$):** Suitable for high-dose data where readout noise or Gaussian statistics dominate.
    $$\mathcal{L}_{\mathrm{Gaussian}} = \frac{\sqrt{\left\langle\left(I_{\mathrm{model}}^{p} - I_{\mathrm{meas}}^{p}\right)^{2}\right\rangle_{\mathcal{D},\mathcal{B}}}}{\langle I_{\mathrm{meas}}^{p}\rangle_{\mathcal{D},\mathcal{B}}}$$
    where $p$ is usually 0.5 (amplitude-based error).

2.  **Poisson Loss ($\mathcal{L}_{\mathrm{Poisson}}$):** Essential for low-dose, electron-counting data where shot noise is the primary error source.
    $$\mathcal{L}_{\mathrm{Poisson}} = -\frac{\left\langle I_{\mathrm{meas}} \log(I_{\mathrm{model}} + \epsilon) - I_{\mathrm{model}}\right\rangle_{\mathcal{D},\mathcal{B}}}{\langle I_{\mathrm{meas}}\rangle_{\mathcal{D},\mathcal{B}}}$$

**Total Objective Function**
To constrain the ill-posed nature of phase retrieval, particularly in 3D, the final objective function $\mathcal{L}_{\mathrm{total}}$ combines the data fidelity term with physical regularizers:

$$\mathcal{L}_{\mathrm{total}} = w_{1}\mathcal{L}_{\mathrm{data}} + w_{2}\mathcal{L}_{\mathrm{PACBED}} + w_{3}\mathcal{L}_{\mathrm{sparse}}$$

*   $\mathcal{L}_{\mathrm{PACBED}}$ enforces consistency with the Position-Averaged Convergent Beam Electron Diffraction pattern, stabilizing the probe intensity distribution.
*   $\mathcal{L}_{\mathrm{sparse}}$ applies an $L_1$-norm regularization (typically on the object phase), promoting "atomicity" and suppressing background noise, effectively deconvolving the probe tails from the atomic potentials.

By minimizing $\mathcal{L}_{\mathrm{total}}$, the framework simultaneously retrieves the object structure, probe aberration state, and geometric experimental parameters.

## 3. Detailed Explanation of the Physical Process

To effectively minimize the loss function $\mathcal{L}_{\mathrm{total}}$ described in the previous section, we must construct a differentiable forward model that accurately simulates the physical interaction between the electron probe and the specimen. In the PtyRAD framework, this process is mathematically formalized using a **mixed-state multislice ptychography model**. This approach accounts for partial coherence in the illumination, dynamical scattering through thick specimens, and geometric experimental imperfections such as sample tilt.

### 3.1 The Mixed-State Formalism

Experimental electron beams are rarely perfectly coherent, and specimens may exhibit dynamic states during the scan. To address this, the forward model represents the probe $P$ and the object $O$ as mixtures of mutually incoherent states.

Let $\boldsymbol{r} = (r_x, r_y)$ denote real-space coordinates and $\boldsymbol{k} = (k_x, k_y)$ denote reciprocal-space coordinates. The probe is defined as a set of $M$ orthogonal modes, $P^{(m)}$, and the object is defined as a set of $N$ modes, $O^{(n)}$. Because these states are treated as incoherent, the total modeled diffraction intensity $I_{j}^{\mathrm{model}}$ at a specific scan position $j$ is the sum of the intensities generated by every pairwise combination of probe and object modes:

$$I_{j}^{\mathrm{model}}(\boldsymbol{k}) = \sum_{m=1}^{M}\sum_{n=1}^{N} \left| \mathcal{F} \left[ \psi_{j}^{(m)(n)}(\boldsymbol{r}) \right] \right|^{2}$$

Here, $\mathcal{F}$ represents the Fourier transform, describing the Far-field propagation of the exit wave $\psi_{j}^{(m)(n)}$ from the sample exit surface to the detector plane. The core physical challenge lies in calculating this exit wave $\psi$ accurately.

### 3.2 The Multislice Propagation Algorithm

For thick samples, the single-phase-screen approximation fails because electrons undergo multiple scattering events (dynamical scattering) as they propagate through the electrostatic potential of the atoms. PtyRAD addresses this by implementing the **multislice algorithm**.

The 3D object potential $O^{(n)}$ is computationally divided into $N_z$ discrete slices along the optical axis ($z$). The electron wave function is transmitted through a slice, acquires a phase shift based on the potential in that slice, and then propagates through vacuum to the next slice.

At a scan position $\boldsymbol{\rho}_j$, the interaction begins with the incident probe mode $P^{(m)}$. To account for sub-pixel scanning precision—which is critical for high-resolution reconstruction—a shift operator $S_{\Delta\boldsymbol{r}_{j}}$ is applied to the probe relative to the integer-cropped object patch. The shifted probe at position $j$ is:

$$P_{j}^{(m)} = S_{\Delta\boldsymbol{r}_{j}} P^{(m)}$$

where $\Delta\boldsymbol{r}_{j} = \boldsymbol{\rho}_j - \text{round}(\boldsymbol{\rho}_j)$ represents the sub-pixel shift residual.

The wave propagation is then calculated iteratively. Let $\Psi_{k}$ be the wave function incident on slice $k$, and $O_{j,k}^{(n)}$ be the transmission function of the $k$-th slice of the object at position $j$. The wave transmits through the slice and propagates to the next slice ($k+1$) via the Fresnel propagator $\mathcal{M}$:

$$\Psi_{k+1} = \mathcal{F}^{-1} \left[ \mathcal{M}_{\boldsymbol{\theta}_{j}, \Delta z}(\boldsymbol{k}) \cdot \mathcal{F} \left[ O_{j,k}^{(n)} \cdot \Psi_{k} \right] \right]$$

This recursive process continues from the first slice ($k=1$) to the last slice ($k=N_z$), resulting in the final exit wave $\psi_{j}^{(m)(n)}$.

### 3.3 The Adaptive Propagator and Geometric Corrections

A distinct feature of the physical model in PtyRAD is the inclusion of geometric parameters directly into the propagation kernel. The standard Fresnel propagator is modified to account for **local sample mistilt** $\boldsymbol{\theta}_j = (\theta_{j,x}, \theta_{j,y})$ and variable slice thickness $\Delta z$.

The propagator $\mathcal{M}_{\boldsymbol{\theta}_{j}, \Delta z}$ in reciprocal space is given by:

$$\mathcal{M}_{\boldsymbol{\theta}_{j}, \Delta z}(\boldsymbol{k}) = \exp\left[-i\pi\lambda|\boldsymbol{k}|^{2}\Delta z + 2\pi i\Delta z(k_{x}\tan\theta_{j,x} + k_{y}\tan\theta_{j,y})\right]$$

This equation encapsulates two physical effects:
1.  **Diffraction:** The term $-i\pi\lambda|\boldsymbol{k}|^{2}\Delta z$ represents the parabolic approximation of the Ewald sphere, describing how the wave spreads over the propagation distance $\Delta z$.
2.  **Tilt Correction:** The term $2\pi i\Delta z(\dots)$ introduces a phase ramp that corrects for the geometric projection error caused by the sample being tilted relative to the optical axis.

By incorporating $\boldsymbol{\theta}_j$ and $\Delta z$ into the propagator, PtyRAD renders the diffraction physics differentiable with respect to these geometric parameters. Consequently, when the loss $\mathcal{L}_{\mathrm{total}}$ is backpropagated, the Automatic Differentiation (AD) engine computes gradients not only for the object pixels and probe values but also for the sample thickness and local tilts, allowing these experimental parameters to be refined simultaneously with the image reconstruction.

Here is Section 4: "Data Preprocessing" for your scientific tutorial on PtyRAD.

---

# 4. Data Preprocessing

Before the iterative reconstruction loop begins, the raw experimental data and simulation parameters must be ingested, validated, and transformed into a computational state suitable for optimization. This section outlines the mathematical basis for initialization and details the implementation of the data loading pipeline.

## 4.1 Conceptual Overview

In ptychography, we aim to recover an object function $O(\mathbf{r})$ and a probe function $P(\mathbf{r})$ from a set of intensity measurements $I_j(\mathbf{u})$ recorded in the far-field, where $j$ indexes the scanning positions.

The preprocessing stage is responsible for establishing the initial guess for the complex-valued wavefields. If we denote the initial object guess as $\psi_{obj}^{(0)}$ and the probe as $\psi_{probe}^{(0)}$, the initialization process can be formalized as mapping configuration parameters $\theta$ and raw measurements $Y$ to the initial state vector $\mathbf{x}^{(0)}$:

$$
\mathbf{x}^{(0)} = \mathcal{T}_{\text{init}}(Y, \theta)
$$

where $\mathcal{T}_{\text{init}}$ represents the initialization transform. This often involves:
1.  **Normalization:** Scaling raw photon counts to match the expected energy conservation of the forward model.
2.  **Phase Initialization:** Since detectors only record intensity ($I = |\mathcal{F}[\psi]|^2$), the initial phase $\phi^{(0)}$ is unknown. It is typically initialized as a flat field or random noise:
    $$ \psi_{obj}^{(0)}(\mathbf{r}) = A_{obj} \cdot e^{i \cdot 0} \quad \text{or} \quad A_{obj} \cdot e^{i \cdot \mathcal{U}(-\pi, \pi)} $$
3.  **Constraint Normalization:** Regularization terms (like Total Variation or modulus constraints) are defined as projection operators $\mathcal{P}_C$. The preprocessing step standardizes the hyperparameters $\lambda_k$ associated with these constraints to ensure consistent application during the iterative update:
    $$ \mathbf{x}^{(k+1)} \leftarrow \mathcal{P}_C(\mathbf{x}^{(k)} - \alpha \nabla \mathcal{L}) $$

## 4.2 Implementation Details

The implementation relies on a central orchestration function, `load_and_preprocess_data`, which acts as the bridge between static configuration files and the dynamic runtime environment. It handles hardware allocation, parameter validation, and the instantiation of loss and constraint manifolds.

### Function Explanations

Below is a detailed breakdown of the functions required to reproduce the preprocessing pipeline.

#### 1. `load_and_preprocess_data(params_path, gpuid=0)`

This is the primary entry point for the preprocessing module. It aggregates all setup steps into a single function call.

*   **Purpose:** To bootstrap the reconstruction environment. It loads the YAML configuration, sets up the GPU, initializes logging, and instantiates the core physics objects (Initializer, Loss, Constraints).
*   **Arguments:**
    *   `params_path` (str): Path to the `.yaml` configuration file.
    *   `gpuid` (int): The ID of the GPU device to use (default is 0).
*   **Process Flow:**
    1.  **System Info:** Calls `print_system_info()` to log hardware specs.
    2.  **Logging:** Instantiates `CustomLogger` to capture stdout/stderr to a file.
    3.  **Device Setup:** Calls `set_gpu_device` to pin the process to a specific GPU.
    4.  **Parameter Loading:** Reads the YAML file. Crucially, it calls `_normalize_constraint_params` to ensure legacy configuration files are compatible with the current solver.
    5.  **Validation:** Uses `PtyRADParams` (likely a Pydantic model) to validate the schema of the loaded parameters.
    6.  **Physics Initialization:** Instantiates the `Initializer` class. This class is responsible for actually loading the heavy `.mat` data files or generating synthetic data.
    7.  **Optimization Setup:** Instantiates `CombinedLoss` and `CombinedConstraint` based on the loaded parameters.
*   **Returns:** A `data_context` dictionary containing the fully initialized state (params, device, logger, loss functions, and initial variable tensors).

#### 2. `_normalize_constraint_params(constraint_params)`

This is a helper utility designed to sanitize input parameters for constraints.

*   **Purpose:** To standardize the scheduling of constraints. In iterative algorithms, constraints (like support or positivity) might only apply after a certain iteration or at specific intervals. This function converts various shorthand notations into a standard `{start, step, end}` format.
*   **Arguments:**
    *   `constraint_params` (dict): A dictionary where keys are constraint names and values are configuration dicts.
*   **Logic:**
    *   It iterates through each constraint.
    *   It looks for a `freq` key (legacy shorthand for frequency).
    *   It maps `freq` to `step`.
    *   It defaults `start_iter` to 1 and `end_iter` to `None` (infinity) if not specified.
    *   It preserves all other specific parameters (e.g., threshold values) while removing the processed scheduling keys.
*   **Returns:** A dictionary with normalized scheduling parameters.

---

## 4.3 Code Reproduction Guide

To reproduce the preprocessing module, you must implement the two functions described above. The code relies on specific external classes (`Initializer`, `CombinedLoss`, `PtyRADParams`, etc.) which are imported from the `ptyrad` package.

### Prerequisites
Ensure your environment has the following dependencies installed:
*   `PyYAML`
*   `torch`
*   `ptyrad` (The core package this tutorial is based on)

### Source Code
Save the following code in a file named `preprocessing.py`.

```python
import os
import yaml
import torch

# --- REQUIRED IMPORTS FROM PTYRAD ---
# These imports assume the ptyrad package structure is available in your environment.
from ptyrad.utils import (
    print_system_info, set_gpu_device, CustomLogger, vprint
)
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss
from ptyrad.constraints import CombinedConstraint
from ptyrad.params import PtyRADParams

def _normalize_constraint_params(constraint_params):
    """
    Helper to convert old constraint param format into a standardized
    start/step/end iteration format.
    
    Args:
        constraint_params (dict): Raw dictionary of constraint configurations.
        
    Returns:
        dict: Normalized configuration dictionary.
    """
    normalized_params = {}
    for name, p in constraint_params.items():
        # Extract scheduling logic
        freq = p.get("freq", None)
        
        # Logic: If freq is present, start at 1, otherwise use provided start_iter
        start_iter = p.get("start_iter", 1 if freq is not None else None)
        
        # Logic: If freq is present, use it as step, otherwise default to 1
        step = p.get("step", freq if freq is not None else 1)
        
        end_iter = p.get("end_iter", None)
        
        # Reconstruct dict with standardized keys, removing the old ones
        normalized_params[name] = {
            "start_iter": start_iter,
            "step": step,
            "end_iter": end_iter,
            # Include all other keys (e.g., 'alpha', 'threshold') that aren't scheduling keys
            **{k: v for k, v in p.items() if k not in ("freq", "step", "start_iter", "end_iter")},
        }
    return normalized_params

def load_and_preprocess_data(params_path, gpuid=0):
    """
    Loads configuration, initializes system, and prepares initial data structures.
    
    Args:
        params_path (str): Path to the .yaml configuration file.
        gpuid (int): GPU index to use.
        
    Returns:
        dict: A context dictionary containing params, device, logger, 
              initializer, variables, loss_fn, and constraint_fn.
    """
    # 1. System Setup
    print_system_info()
    # Initialize logger to write to ptyrad_log.txt
    logger = CustomLogger(log_file='ptyrad_log.txt', log_dir='auto', prefix_time='datetime', show_timestamp=True)
    device = set_gpu_device(gpuid=gpuid)

    # 2. Load Parameters
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"File '{params_path}' not found.")
    
    vprint(f"### Loading params file: {params_path} ###")
    with open(params_path, "r", encoding='utf-8') as file:
        raw_params = yaml.safe_load(file)
    
    # Normalize constraints (legacy support)
    if raw_params.get('constraint_params') is not None:
        raw_params['constraint_params'] = _normalize_constraint_params(raw_params['constraint_params'])
    
    # Validate params using the Pydantic model
    # .model_dump() converts the Pydantic object back to a standard dict
    params = PtyRADParams(**raw_params).model_dump()
    params['params_path'] = params_path

    # 3. Initialize Data (Measurements, Probe, Object guess)
    # The Initializer class handles loading .mat files or generating synthetic data
    vprint("### Initializing Initializer ###")
    # init_all() triggers the actual data loading/generation
    initializer = Initializer(params['init_params'], seed=None).init_all()
    init_variables = initializer.init_variables

    # 4. Initialize Loss and Constraints
    vprint("### Initializing loss function ###")
    loss_fn = CombinedLoss(params['loss_params'], device=device)
    
    vprint("### Initializing constraint function ###")
    constraint_fn = CombinedConstraint(params['constraint_params'], device=device, verbose=True)

    # Pack everything into a context dictionary
    data_context = {
        "params": params,
        "device": device,
        "logger": logger,
        "initializer": initializer,
        "init_variables": init_variables,
        "loss_fn": loss_fn,
        "constraint_fn": constraint_fn,
        "accelerator": None # Placeholder for distributed training contexts
    }
    
    return data_context
```

### Verification Strategy (Unit Test)

To verify that your reproduction is correct, you can run the following test script. This script mocks the complex `ptyrad` dependencies to ensure the logic within `load_and_preprocess_data` and `_normalize_constraint_params` functions correctly without needing the full physical dataset.

```python
import unittest
from unittest.mock import MagicMock, patch, mock_open
import yaml
import os

# Import the code you just wrote
from preprocessing import load_and_preprocess_data, _normalize_constraint_params

class TestPreprocessing(unittest.TestCase):

    def test_normalize_constraint_params(self):
        """Test if legacy 'freq' params are converted to 'step'."""
        input_params = {
            "tv_constraint": {"freq": 10, "alpha": 0.1},
            "positivity": {"start_iter": 5, "step": 2}
        }
        expected = {
            "tv_constraint": {"start_iter": 1, "step": 10, "end_iter": None, "alpha": 0.1},
            "positivity": {"start_iter": 5, "step": 2, "end_iter": None}
        }
        result = _normalize_constraint_params(input_params)
        self.assertEqual(result, expected)

    @patch("preprocessing.PtyRADParams")
    @patch("preprocessing.Initializer")
    @patch("preprocessing.CombinedLoss")
    @patch("preprocessing.CombinedConstraint")
    @patch("preprocessing.CustomLogger")
    @patch("preprocessing.set_gpu_device")
    @patch("preprocessing.print_system_info")
    def test_load_and_preprocess_flow(self, mock_print, mock_set_gpu, mock_logger, 
                                      mock_constraint, mock_loss, mock_init, mock_params):
        """
        Test the full loading flow with mocked dependencies.
        """
        # Setup Mocks
        mock_set_gpu.return_value = "cpu"
        
        # Mock the Pydantic model behavior
        mock_params_instance = MagicMock()
        mock_params_instance.model_dump.return_value = {
            "init_params": {}, "loss_params": {}, "constraint_params": {}
        }
        mock_params.return_value = mock_params_instance

        # Mock Initializer behavior
        mock_init_instance = MagicMock()
        mock_init_instance.init_all.return_value = mock_init_instance
        mock_init_instance.init_variables = "mock_variables"
        mock_init.return_value = mock_init_instance

        # Create a dummy yaml file
        dummy_yaml = """
        init_params: {}
        loss_params: {}
        constraint_params:
            test_const: {freq: 5}
        """
        
        with patch("builtins.open", mock_open(read_data=dummy_yaml)):
            with patch("os.path.exists", return_value=True):
                context = load_and_preprocess_data("dummy_config.yaml", gpuid=0)

        # Assertions
        self.assertIn("params", context)
        self.assertEqual(context["device"], "cpu")
        self.assertEqual(context["init_variables"], "mock_variables")
        
        # Verify normalization happened inside the function
        # The mock_params constructor should have received normalized constraints
        call_args = mock_params.call_args[1] # kwargs
        self.assertIn("constraint_params", call_args)
        self.assertEqual(call_args["constraint_params"]["test_const"]["step"], 5)

if __name__ == '__main__':
    unittest.main()
```

## 5. Forward Operator Implementation

Having verified the normalized constraints and parameter initialization in the previous section, we now move to the core of the reconstruction engine: the **Forward Operator**. In the context of iterative phase retrieval, the forward operator is the mathematical simulation that maps our current estimate of the physical parameters (the object and the probe) to the observable data (the diffraction patterns).

While previous generations of ptychography software required manual derivation of analytical gradients for every new parameter introduced, PtyRAD leverages the computational graph of PyTorch. By strictly defining the physical forward model using differentiable tensor operations, we enable the Automatic Differentiation (AD) engine to compute exact gradients for optimization via the chain rule.

This section details the theoretical construction of the **mixed-state multislice** forward model implemented in PtyRAD.

### 5.1. The Inverse Problem Formulation

The goal of ptychography is to recover the complex object function $O$ and the illumination probe $P$ from a set of measured diffraction intensities $I^{\mathrm{meas}}$. We treat this as an optimization problem where we minimize the discrepancy between the measured data and a simulated model $I^{\mathrm{model}}$.

The forward function $f$ simulates the physical interaction of the electron beam with the sample:

$$ I_{j}^{\mathrm{model}}(\boldsymbol{k}) = f(P, O, \boldsymbol{\rho}_{j}, \boldsymbol{\theta}_{j}, \Delta z) $$

where $j$ indexes the specific probe position in the scan. Crucially, in the PtyRAD framework, the inputs to $f$ are not limited to just the object and probe; they also include geometrical parameters such as the probe position $\boldsymbol{\rho}_{j}$, the local object mistilt $\boldsymbol{\theta}_{j}$, and the slice thickness $\Delta z$. Because these are inputs to the differentiable forward model, they can be refined simultaneously during reconstruction.

### 5.2. Mixed-State Formalism

To account for partial coherence in the illumination and dynamic states in the specimen, PtyRAD employs the mixed-state formalism. The physical model assumes that the total recorded intensity is an incoherent sum of mutually incoherent modes.

We define the probe $P$ as a set of $M$ orthogonal modes and the object $O$ as a set of $N$ modes. The modeled intensity at detector coordinates $\boldsymbol{k}$ is the summation of the Fourier intensities of the exit waves $\psi$ generated by every combination of probe and object modes:

$$ I_{j}^{\mathrm{model}}(\boldsymbol{k}) = \sum_{m=1}^{M} \sum_{n=1}^{N} \left| \mathcal{F} \left[ \psi_{j}^{(m)(n)}(\boldsymbol{r}) \right] \right|^{2} $$

Here, $\mathcal{F}$ denotes the Fourier transform, propagating the exit wave from the sample exit surface to the far-field detector plane. $\psi_{j}^{(m)(n)}$ represents the exit wave corresponding to the $m$-th probe mode and $n$-th object mode at scan position $j$.

### 5.3. Multislice Propagation

For thick samples where the projection approximation breaks down due to dynamical scattering (multiple scattering events), a single transmission function is insufficient. PtyRAD implements a **multislice** approach. The object is divided into $N_{z}^{O}$ discrete slices along the optical axis ($z$).

The interaction is modeled as a sequential process: the electron wave transmits through a thin slice of the object, propagates through free space to the next slice, transmits through that slice, and so on.

The exit wave $\psi_{j}^{(m)(n)}$ is computed by iterating through slices $l = 1 \dots N_{z}^{O}$. For a specific probe position, the interaction at the first slice is initialized by applying a sub-pixel shift operator $S_{\Delta\boldsymbol{r}_{j}}$ to the probe $P^{(m)}$ to align it with the object grid $O^{(n)}$:

$$ \psi_{j, 1} = O_{j, 1}^{(n)} \cdot S_{\Delta\boldsymbol{r}_{j}} P^{(m)} $$

For subsequent slices, the wave is propagated and then transmitted:

$$ \psi_{j, l+1} = O_{j, l+1}^{(n)} \cdot \mathcal{F}^{-1} \left[ \mathcal{M}_{\boldsymbol{\theta}_{j}, \Delta z} \cdot \mathcal{F} \left[ \psi_{j, l} \right] \right] $$

This recursive formulation allows the model to capture depth-dependent information, enabling 3D ptychography.

### 5.4. The Differentiable Propagator

A distinct feature of the PtyRAD implementation is the inclusion of geometric gradients within the propagator itself. The Fresnel free-space propagator $\mathcal{M}$ is modified to include terms for slice thickness $\Delta z$ and sample tilt $\boldsymbol{\theta}_{j} = (\theta_{j,x}, \theta_{j,y})$.

The propagator in reciprocal space is defined as:

$$ \mathcal{M}_{\boldsymbol{\theta}_{j},\Delta z}(\boldsymbol{k}) = \exp\left[ -i\pi\lambda|\boldsymbol{k}|^{2}\Delta z + 2\pi i\Delta z (k_{x}\tan\theta_{j,x} + k_{y}\tan\theta_{j,y}) \right] $$

The two terms in the exponential represent:
1.  **Diffraction:** $-i\pi\lambda|\boldsymbol{k}|^{2}\Delta z$ describes the parabolic approximation of the Fresnel diffraction over distance $\Delta z$, where $\lambda$ is the electron wavelength.
2.  **Tilt Correction:** $2\pi i\Delta z (\dots)$ introduces a phase ramp that accounts for the geometric shift caused by the beam propagating through a tilted coordinate system.

By implementing this equation using PyTorch tensors, the thickness $\Delta z$ and tilt $\boldsymbol{\theta}$ become scalar variables in the computational graph. During the backward pass (backpropagation), the AD engine computes $\frac{\partial \mathcal{L}}{\partial \Delta z}$ and $\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}}$, allowing the optimizer to correct for experimental inaccuracies in thickness and crystal orientation automatically.

Here is Section 6 of the tutorial, focusing on the core inverse algorithm loop.

---

# Section 6: Core Loop of Inverse Algorithm (Focus!)

## 6.1 Conceptual Overview

In ptychographic reconstruction, we solve an inverse problem to recover the complex-valued object $O(\mathbf{r})$ and the probe $P(\mathbf{r})$ from a set of diffraction intensity patterns $I_j$. This is formulated as an optimization problem where we seek to minimize a discrepancy metric (loss function) between the measured data and the output of our physical forward model.

We define our forward operator $\mathcal{F}$ as a multi-slice propagation model. For a specific scan position $\mathbf{r}_j$, the exit wave $\psi_j$ is modeled, and the predicted intensity at the detector is the magnitude squared of the Fourier transform of this exit wave.

The optimization objective is defined as:

$$
\underset{\theta}{\text{minimize}} \quad \mathcal{L}(\theta) = \sum_j \left\| \left| \mathfrak{F}\left[ \mathcal{F}_j(O, P, \dots) \right] \right| - \sqrt{I_j} \right\|^2 + \mathcal{R}(\theta)
$$

Where:
*   $\theta$: The set of learnable parameters, including the Object ($O$), Probe ($P$), scan positions, object tilts, and slice thickness.
*   $\mathfrak{F}$: The discrete Fourier Transform (modeling far-field propagation).
*   $\mathcal{F}_j$: The physical forward operator at scan index $j$.
*   $\sqrt{I_j}$: The amplitude of the measured diffraction pattern.
*   $\mathcal{R}(\theta)$: Regularization terms (constraints).

We solve this using **Automatic Differentiation (AD)**. By implementing the physics $\mathcal{F}$ as a differentiable computational graph (a PyTorch Module), we can compute gradients $\nabla_\theta \mathcal{L}$ via backpropagation and update parameters using standard optimizers (e.g., Adam):

$$
\theta_{t+1} \leftarrow \theta_t - \eta \cdot \text{Optimizer}(\nabla_\theta \mathcal{L})
$$

---

## 6.2 Implementation Details

Below is the detailed explanation of the implementation. We encapsulate the physical model and the state of the reconstruction in a class called `PtychoAD`. The optimization process is driven by `run_inversion`.

### 6.2.1 Dependencies and Imports

Ensure your environment includes these specific imports. This code relies on the `ptyrad` library for utility functions and the forward physics engine.

```python
import os
import yaml
import torch
import torch.nn as nn
from math import prod
from torchvision.transforms.functional import gaussian_blur
import torch.distributed as dist

# ptyrad specific imports
from ptyrad.utils import (
    print_system_info, set_gpu_device, CustomLogger, vprint, 
    time_sync, imshift_batch, torch_phasor
)
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss
from ptyrad.constraints import CombinedConstraint
from ptyrad.params import PtyRADParams
from ptyrad.reconstruction import (
    create_optimizer, prepare_recon, parse_sec_to_time_str, 
    recon_loop, IndicesDataset
)
from ptyrad.forward import multislice_forward_model_vec_all
```

### 6.2.2 The `PtychoAD` Class: Explaining Every Function

The `PtychoAD` class inherits from `torch.nn.Module`. This allows PyTorch to automatically track gradients for any `nn.Parameter` defined within it.

#### 1. `__init__(self, init_variables, model_params, device='cuda', verbose=True)`
**Purpose:** Initializes the reconstruction state, loads experimental data into GPU memory, and defines which physical parameters are optimizable.

*   **Data Loading:** It extracts initial guesses (`init_variables`) for the object, probe, and geometry.
*   **Parameters vs. Buffers:**
    *   Variables we want to optimize (Object, Probe, Tilts) are wrapped in `nn.Parameter`.
    *   Fixed constants (Measurements, Wavelength `lambd`, Pixel size `dx`) are registered as buffers using `register_buffer`. This ensures they move to the GPU with the model but are not updated by the optimizer.
*   **Optimizer Configuration:** It parses `model_params` to determine which parameters have a learning rate > 0 and sets flags (e.g., `self.tilt_obj`, `self.shift_probes`) to enable/disable specific physics branches during the forward pass to save computation.

#### 2. `get_complex_probe_view(self)`
**Purpose:** Helper to convert the stored probe tensor into complex format.
*   **Implementation:** The probe is often stored as a real tensor with shape `(..., 2)` (Real/Imaginary channels) for compatibility with certain optimizers. This function returns it as `torch.complex64`.

#### 3. `_create_grids(self)`
**Purpose:** Pre-calculates coordinate grids required for Fourier optics.
*   **Details:**
    *   Generates spatial grids ($x, y$) and frequency grids ($k_x, k_y$) based on the probe and object dimensions.
    *   `self.propagator_grid`: Stores ($K_y, K_x$) for calculating the Fresnel transfer function.
    *   `self.rpy_grid` / `self.rpx_grid`: Integer grids used for cropping the object (integer indexing).

#### 4. `_setup_optimizer_params(self)`
**Purpose:** Constructs the parameter groups for the optimizer.
*   **Logic:** Iterates through `self.lr_params`. If a learning rate is non-zero, it adds that specific parameter to the optimization list. It also sets `requires_grad=True` for those tensors. This allows for flexible "freezing" of parameters (e.g., optimize probe only, then object only).

#### 5. `_init_propagator_vars(self)`
**Purpose:** Pre-computes the static components of the Fresnel propagator.
*   **Physics:** The propagator in Fourier space is $H = \exp(i z \sqrt{k^2 - k_x^2 - k_y^2})$.
*   **Optimization:** It calculates the standard propagator `H`. It also pre-calculates a version `H_fixed_tilts_full` that includes phase ramps for object tilts, assuming the tilts don't change during this specific optimization phase.

#### 6. `_init_compilation_iters(self)`
**Purpose:** Performance tuning (JIT).
*   **Logic:** Identifies specific iteration numbers (start/end of parameter updates) where the computational graph structure might change. This is used by the training loop to trigger graph recompilation or cache clearing.

#### 7. `get_obj_patches(self, indices)`
**Purpose:** Extracts the specific region of the object illuminated by the probe at scan positions `indices`.
*   **Mechanism:**
    1.  Takes the integer crop positions (`self.crop_pos`) for the current batch of indices.
    2.  Uses `self.rpy_grid` and `self.rpx_grid` to create a mesh of indices corresponding to the Region of Interest (ROI).
    3.  Indexes into the master object `self.opt_obja` (amplitude) and `self.opt_objp` (phase).
    4.  **Pre-blur:** If `obj_preblur_std` is active, applies a Gaussian blur to the patch. This is often used to suppress high-frequency artifacts during early iterations.

#### 8. `get_probes(self, indices)`
**Purpose:** Returns the probe modes for the current batch.
*   **Mechanism:**
    *   If `self.shift_probes` is True, it applies sub-pixel shifts (`imshift_batch`) to the probe based on `self.opt_probe_pos_shifts`. This corrects for mechanical scan errors.
    *   Otherwise, it simply broadcasts the static probe to the batch size.

#### 9. `get_propagators(self, indices)`
**Purpose:** Computes the exact Fresnel propagation kernel for the current batch.
*   **Physics:** This function allows for **Multi-slice** capabilities.
    *   If `change_thickness` is True, the propagation distance $z$ is dynamic (`opt_slice_thickness`).
    *   If `tilt_obj` is True, it adds a phase ramp in Fourier space: $\exp(i(k_y \tan \theta_y + k_x \tan \theta_x)z)$.
    *   This dynamic calculation is critical for 3D or thick sample reconstruction.

#### 10. `get_measurements(self, indices=None)`
**Purpose:** Retrieves the ground-truth diffraction data for the loss calculation.
*   **Features:**
    *   **Padding:** Supports "on-the-fly" padding (`meas_padded`) if the detector size is larger than the computation window.
    *   **Scaling:** Applies `meas_scale_factors` (interpolation) if the experimental pixel size differs from the simulation grid.

#### 11. `clear_cache(self)`
**Purpose:** Manually clears internal temporary buffers (`_current_object_patches`) to free GPU memory between iterations.

#### 12. `forward(self, indices)`
**Purpose:** The main prediction step.
*   **Flow:**
    1.  Calls `get_obj_patches` to get measuring capabilities $O_{batch}$.
    2.  Calls `get_probes` to get $P_{batch}$.
    3.  Calls `get_propagators` to get transfer functions $H_{batch}$.
    4.  **Physics Engine:** Passes these into `multislice_forward_model_vec_all`. This external function performs the wave mixing $\psi_{exit} = O \cdot P$ and FFT propagation.
    5.  **Detector Model:** Optionally applies `detector_blur_std` (Gaussian blur) to the predicted exit wave to simulate detector point spread function (PSF).
    6.  Returns `dp_fwd` (Predicted exit wave at detector plane).

---

### 6.2.3 The Driver Function: `run_inversion`

This function orchestrates the training process.

#### `run_inversion(data_context)`
**Purpose:** Sets up the environment and executes the loop.
1.  **Unpacking:** Extracts configuration, loss functions, and the accelerator (for distributed training) from `data_context`.
2.  **Model Instantiation:** Creates an instance of `PtychoAD`.
3.  **Optimizer Creation:** Calls `create_optimizer` (typically Adam) passing the model's parameter groups.
4.  **Data Loader:**
    *   If running on a single GPU, it prepares simple index lists.
    *   If running with an `accelerator` (Multi-GPU), it wraps the indices in an `IndicesDataset` and a PyTorch `DataLoader`. This handles distributing different scan points to different GPUs.
5.  **Reconstruction Loop:** Calls `recon_loop`. This external function contains the standard boilerplate:
    ```python
    for i in range(n_epochs):
        optimizer.zero_grad()
        pred = model(indices)
        loss = loss_fn(pred, target)
        accelerator.backward(loss)
        optimizer.step()
    ```
6.  **Return:** Returns the trained model and optimizer state.

### 6.2.4 Reproducible Code Block

```python
import torch
import torch.nn as nn
from math import prod
from torchvision.transforms.functional import gaussian_blur
from ptyrad.utils import (
    print_system_info, set_gpu_device, CustomLogger, vprint, 
    time_sync, imshift_batch, torch_phasor
)
from ptyrad.reconstruction import (
    create_optimizer, prepare_recon, parse_sec_to_time_str, 
    recon_loop, IndicesDataset
)
from ptyrad.forward import multislice_forward_model_vec_all


class PtychoAD(torch.nn.Module):
    """
    The PyTorch Module representing the physical forward model.
    This class encapsulates the 'forward_operator' logic within its .forward() method.
    """
    def __init__(self, init_variables, model_params, device='cuda', verbose=True):
        super(PtychoAD, self).__init__()
        with torch.no_grad():
            self.device = device
            self.verbose = verbose
            self.detector_blur_std = model_params.get('detector_blur_std', 0)
            self.obj_preblur_std = model_params.get('obj_preblur_std', 0)
            
            # Initialize tracking lists
            self.loss_iters = []
            self.iter_times = []
            self.dz_iters = []
            self.avg_tilt_iters = []

            # Handle on-the-fly measurement padding
            if init_variables.get('on_the_fly_meas_padded', None) is not None:
                self.meas_padded = torch.tensor(init_variables['on_the_fly_meas_padded'], dtype=torch.float32, device=device)
                self.meas_padded_idx = torch.tensor(init_variables['on_the_fly_meas_padded_idx'], dtype=torch.int32, device=device)
            else:
                self.meas_padded = None
            self.meas_scale_factors = init_variables.get('on_the_fly_meas_scale_factors', None)

            # Optimizer setup parsing
            self.optimizer_params = model_params['optimizer_params']
            self.start_iter = {}
            self.end_iter = {}
            self.lr_params = {}
            for key, p in model_params['update_params'].items():
                self.start_iter[key] = p.get('start_iter', 0)
                self.end_iter[key] = p.get('end_iter', 10000)
                self.lr_params[key] = p.get('lr', 0)

            # Optimizable Parameters (wrapped in nn.Parameter)
            # We assume complex objects are stored as Ampliture (obja) and Phase (objp)
            self.opt_obja = nn.Parameter(torch.abs(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
            self.opt_objp = nn.Parameter(torch.angle(torch.tensor(init_variables['obj'], device=device)).to(torch.float32))
            
            # Geometry parameters
            self.opt_obj_tilts = nn.Parameter(torch.tensor(init_variables['obj_tilts'], dtype=torch.float32, device=device))
            self.opt_slice_thickness = nn.Parameter(torch.tensor(init_variables['slice_thickness'], dtype=torch.float32, device=device))
            self.opt_probe = nn.Parameter(torch.view_as_real(torch.tensor(init_variables['probe'], dtype=torch.complex64, device=device)))
            self.opt_probe_pos_shifts = nn.Parameter(torch.tensor(init_variables['probe_pos_shifts'], dtype=torch.float32, device=device))

            # Buffers (Fixed constants)
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

            # Metadata
            self.random_seed = init_variables['random_seed']
            self.length_unit = init_variables['length_unit']
            self.scan_affine = init_variables['scan_affine']
            
            # Logic flags for optimization branches
            self.tilt_obj = bool(self.lr_params.get('obj_tilts', 0) != 0 or torch.any(self.opt_obj_tilts != 0))
            self.shift_probes = bool(self.lr_params.get('probe_pos_shifts', 0) != 0)
            self.change_thickness = bool(self.lr_params.get('slice_thickness', 0) != 0)
            self.probe_int_sum = self.get_complex_probe_view().abs().pow(2).sum()

            # Internal Grids and Setup
            self._create_grids()
            self._init_propagator_vars()
            
            # Map strings to tensors
            self.optimizable_tensors = {
                'obja': self.opt_obja, 'objp': self.opt_objp,
                'obj_tilts': self.opt_obj_tilts, 'slice_thickness': self.opt_slice_thickness,
                'probe': self.opt_probe, 'probe_pos_shifts': self.opt_probe_pos_shifts
            }
            self._setup_optimizer_params()
            self._init_compilation_iters()

    def get_complex_probe_view(self):
        """Helper to view the stored real/imag probe as complex64."""
        return torch.view_as_complex(self.opt_probe)

    def _create_grids(self):
        """Generates Fourier space coordinates (ky, kx) and real space crop grids."""
        device = self.device
        probe = self.get_complex_probe_view()
        Npy, Npx = probe.shape[-2:]

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

    def _setup_optimizer_params(self):
        """Configures parameter groups based on learning rates."""
        self.optimizable_params = []
        for param_name, lr in self.lr_params.items():
            if param_name not in self.optimizable_tensors:
                # Some params might be purely virtual or unused in this specific model configuration
                continue 
            # Enable gradients only if LR > 0 and start_iter is immediate (1)
            self.optimizable_tensors[param_name].requires_grad = (lr != 0) and (self.start_iter[param_name] <= 1)
            if lr != 0:
                self.optimizable_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr})

    def _init_propagator_vars(self):
        """Pre-calculates static parts of the Fresnel kernel."""
        dz = self.opt_slice_thickness.detach()
        Ky, Kx = self.propagator_grid 
        tilts_y_full = self.opt_obj_tilts[:,0,None,None] / 1e3
        tilts_x_full = self.opt_obj_tilts[:,1,None,None] / 1e3
        
        # Calculate Propagator with fixed initial tilts
        self.H_fixed_tilts_full = self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y_full) + Kx * torch.tan(tilts_x_full)))
        self.k = 2 * torch.pi / self.lambd
        self.Kz = torch.sqrt(self.k ** 2 - Kx ** 2 - Ky ** 2)

    def _init_compilation_iters(self):
        """Determines when to trigger recompilation/cache clearing."""
        compilation_iters = {1}
        for param_name in self.optimizable_tensors.keys():
            s = self.start_iter.get(param_name)
            e = self.end_iter.get(param_name)
            if s and s >= 1: compilation_iters.add(s)
            if e and e >= 1: compilation_iters.add(e)
        self.compilation_iters = sorted(compilation_iters)

    def get_obj_patches(self, indices):
        """
        Crops the large object into small patches corresponding to the probe positions.
        """
        opt_obj = torch.stack([self.opt_obja, self.opt_objp], dim=-1)
        
        # Advanced indexing to crop ROI
        obj_ROI_grid_y = self.rpy_grid[None,:,:] + self.crop_pos[indices, None, None, 0]
        obj_ROI_grid_x = self.rpx_grid[None,:,:] + self.crop_pos[indices, None, None, 1]
        
        # Extract patches: (Batch, Y, X, Real/Imag)
        object_roi = opt_obj[:,:,obj_ROI_grid_y,obj_ROI_grid_x,:].permute(2,0,1,3,4,5)
        
        if self.obj_preblur_std is None or self.obj_preblur_std == 0:
            return object_roi
        else:
            # Apply optional Gaussian blur for conditioning
            obj = object_roi.permute(5,0,1,2,3,4)
            obj_shape = obj.shape
            obj = obj.reshape(-1, obj_shape[-2], obj_shape[-1])
            return gaussian_blur(obj, kernel_size=5, sigma=self.obj_preblur_std).reshape(obj_shape).permute(1,2,3,4,5,0)

    def get_probes(self, indices):
        """Retrieves probes, optionally applying sub-pixel shifts."""
        probe = self.get_complex_probe_view()
        if self.shift_probes:
            return imshift_batch(probe, shifts=self.opt_probe_pos_shifts[indices], grid=self.shift_probes_grid)
        return torch.broadcast_to(probe, (indices.shape[0], *probe.shape))

    def get_propagators(self, indices):
        """
        Constructs the Fresnel propagator.
        Handles dynamic logic:
        1. If thickness changes, Kz term is recalculated.
        2. If tilts change, phase ramp (tan theta) is recalculated.
        """
        tilt_obj = self.tilt_obj
        global_tilt = (self.opt_obj_tilts.shape[0] == 1)
        change_tilt = (self.lr_params.get('obj_tilts', 0) != 0)
        change_thickness = self.change_thickness
        
        dz = self.opt_slice_thickness
        Kz = self.Kz
        Ky, Kx = self.propagator_grid
        
        if global_tilt: tilts = self.opt_obj_tilts 
        else: tilts = self.opt_obj_tilts[indices] 
        tilts_y = tilts[:,0,None,None] / 1e3
        tilts_x = tilts[:,1,None,None] / 1e3
                
        if tilt_obj and change_thickness:
            # Full recalculation
            H_opt_dz = torch_phasor(dz * Kz)
            return H_opt_dz * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
        elif tilt_obj and not change_thickness:
            if change_tilt:
                # Recalculate tilt ramp only
                return self.H * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
            else:
                # Use cached fixed tilts
                return self.H_fixed_tilts_full if global_tilt else self.H_fixed_tilts_full[indices]
        elif not tilt_obj and change_thickness: 
            # Recalculate thickness only (normal incidence)
            return torch_phasor(dz * Kz)[None,]
        else: 
            # Static propagator
            return self.H[None,]

    def get_measurements(self, indices=None):
        """Returns observed data, handling padding and scale interpolation."""
        if indices is None: return self.measurements
        
        measurements = self.measurements[indices]
        
        # Handle larger detector canvas
        if self.meas_padded is not None:
            pad_h1, pad_h2, pad_w1, pad_w2 = self.meas_padded_idx
            canvas = torch.zeros((measurements.shape[0], *self.meas_padded.shape[-2:]), dtype=measurements.dtype, device=self.device)
            canvas += self.meas_padded
            canvas[..., pad_h1:pad_h2, pad_w1:pad_w2] = measurements
            measurements = canvas
            
        # Handle pixel size mismatch
        if self.meas_scale_factors is not None:
            scale = tuple(self.meas_scale_factors)
            if any(f != 1 for f in scale):
                measurements = torch.nn.functional.interpolate(measurements[None,], scale_factor=scale, mode='bilinear')[0]
                measurements = measurements / prod(scale)
        return measurements

    def clear_cache(self):
        self._current_object_patches = None

    def forward(self, indices):
        """
        The core forward operator: x -> y_pred
        Here 'x' is implicitly the internal state (object, probe) at 'indices'.
        """
        # 1. Prepare inputs
        object_patches = self.get_obj_patches(indices)
        probes = self.get_probes(indices)
        propagators = self.get_propagators(indices)
        
        # 2. Physics Simulation
        dp_fwd = multislice_forward_model_vec_all(object_patches, probes, propagators, omode_occu=self.omode_occu)
        
        # 3. Detector Modeling
        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=self.detector_blur_std)
            
        self._current_object_patches = object_patches
        return dp_fwd

def run_inversion(data_context):
    """
    Performs the optimization loop.
    """
    params = data_context['params']
    device = data_context['device']
    init_variables = data_context['init_variables']
    initializer = data_context['initializer']
    loss_fn = data_context['loss_fn']
    constraint_fn = data_context['constraint_fn']
    accelerator = data_context['accelerator']
    logger = data_context['logger']

    vprint("### Starting Reconstruction ###")

    # Instantiate the Model (Physics + State)
    model = PtychoAD(init_variables, params['model_params'], device=device, verbose=True)
    
    # Create Optimizer (usually Adam)
    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)

    # Prepare Data Loaders / Batches
    use_acc_device = (device is None and accelerator is not None)
    
    if not use_acc_device:
        # Single GPU / CPU path
        indices, batches, output_path = prepare_recon(model, initializer, params)
    else:
        # Multi-GPU / Accelerator logic
        if params['model_params']['optimizer_params']['name'] == 'LBFGS' and accelerator.num_processes > 1:
            vprint("WARNING: LBFGS not supported for multi-GPU. Switching to Adam.")
            params['model_params']['optimizer_params']['name'] = 'Adam'
            model.optimizer_params['name'] = 'Adam'
            optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
        
        params['recon_params']['GROUP_MODE'] = 'random'
        indices, batches, output_path = prepare_recon(model, initializer, params)
        
        # Use Dataset/Loader for distributed sampling
        ds = IndicesDataset(indices)
        dl = torch.utils.data.DataLoader(ds, batch_size=params['recon_params']['BATCH_SIZE']['size'], shuffle=True)
        batches = accelerator.prepare(dl)
        model, optimizer = accelerator.prepare(model, optimizer)

    # Logging setup
    if logger is not None and logger.flush_file:
        logger.flush_to_file(log_dir=output_path)

    # Run the main optimization loop (from ptyrad.reconstruction)
    # This loop implicitly calls model.forward() (our forward_operator)
    recon_loop(
        model, initializer, params, optimizer, 
        loss_fn, constraint_fn, indices, batches, 
        output_path, acc=accelerator
    )

    return {
        "model": model,
        "optimizer": optimizer,
        "output_path": output_path,
        "logger": logger
    }
```

Here is Section 7 of the scientific tutorial, covering the definition and implementation of evaluation metrics and the finalization of the reconstruction pipeline.

---

# Section 7: Definition and Implementation of Evaluation Metrics

In computational imaging tasks like ptycho-tomography, evaluation is often twofold. First, we must assess the convergence of the optimization process itself (loss behavior). Second, we must finalize the computational environment, ensuring that distributed resources are released and logs are persisted for post-hoc analysis (e.g., PSNR or SSIM calculation against ground truth, if available).

This section defines the mathematical context for evaluation and details the implementation of a cleanup and finalization routine.

## 7.1 Conceptual Overview

While the provided implementation focuses on the operational cleanup of the pipeline, the theoretical basis for evaluating reconstruction quality in complex amplitude imaging relies on measuring the discrepancy between the reconstructed object transmission function $O(\mathbf{r})$ and a ground truth $O_{GT}(\mathbf{r})$ (if available), or the consistency between the measured diffraction patterns $I_{exp}$ and the simulated patterns $I_{sim}$.

### 7.1.1 Image Quality Metrics
When a ground truth object is available (e.g., in simulation studies), the Peak Signal-to-Noise Ratio (PSNR) is typically defined for the phase or amplitude component:

$$
\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)
$$

Where $\text{MSE}$ is the Mean Squared Error:

$$
\text{MSE} = \frac{1}{mn} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i, j) - K(i, j)]^2
$$

In ptycho-tomography, since the object is complex-valued ($O = |O|e^{i\phi}$), we often evaluate the **Phase Error** specifically, as the phase shift contains the tomographic information:

$$
E_{\phi} = \sqrt{\frac{\sum_{\mathbf{r}} |\phi_{rec}(\mathbf{r}) - \phi_{gt}(\mathbf{r})|^2}{\sum_{\mathbf{r}} |\phi_{gt}(\mathbf{r})|^2}}
$$

### 7.1.2 Process Finalization
The computational evaluation also involves the correct termination of the distributed context. In multi-GPU environments formalized by `torch.distributed`, the process group $\mathcal{G}$ must be explicitly destroyed to release shared memory and TCP/IP sockets:

$$
\text{Cleanup}(\mathcal{G}) \implies \text{free}(\text{VRAM}) \land \text{close}(\text{logs})
$$

## 7.2 Implementation Details

The implementation of the evaluation phase is encapsulated in the `evaluate_results` function. This function acts as the destructor for the reconstruction session. It handles the extraction of the final model state, ensures the integrity of the logging system, and tears down the PyTorch distributed backend.

### Function Signature
The function accepts a single dictionary, `results`, which is the output of the reconstruction loop.

### Key Operations
1.  **Model Extraction**: It retrieves the trained `model` from the results dictionary.
2.  **Logger Management**: It accesses the custom logger instance. If file logging was enabled (`flush_file` is true), it explicitly closes the file handlers to ensure all buffered metrics (loss curves, timing data) are written to disk.
3.  **Distributed Cleanup**: It checks `torch.distributed.is_initialized()`. If the code is running in a Distributed Data Parallel (DDP) setting, it calls `dist.destroy_process_group()`. This is critical to prevent zombie processes on cluster nodes.

---

## 7.3 Code Reproduction Guide

To reproduce the `evaluate_results` function, follow the detailed logic below. This implementation is designed to be robust against single-GPU and multi-GPU setups.

### Required Imports
You must include `torch.distributed` for process group management and the specific utility function `vprint` from `ptyrad.utils` for verbose logging.

```python
import torch.distributed as dist
from ptyrad.utils import vprint
```

### Function Logic `evaluate_results(results)`

1.  **Input**:
    *   `results`: A dictionary containing keys `'model'` (the PyTorch model) and `'logger'` (an instance of `CustomLogger` or similar).

2.  **Procedure**:
    *   Extract `model` and `logger` from the `results` dictionary.
    *   Print a header message "### Evaluation & Cleanup ###" using `vprint`.
    *   **Logger Check**: Check if `logger` is not `None`. Inside this check, verify if the logger has an attribute `flush_file` that evaluates to True. If so, call the `close()` method on the logger.
    *   **Distributed Check**: Use `dist.is_initialized()` to check if the distributed backend is active. If it is, call `dist.destroy_process_group()` to cleanly terminate the process group.

### Final Source Code

```python
import torch.distributed as dist
from ptyrad.utils import vprint

def evaluate_results(results):
    """
    Finalizes the process, saves logs, and cleans up.
    
    Args:
        results (dict): A dictionary containing reconstruction results, specifically:
                        - 'model': The reconstructed PyTorch model/object.
                        - 'logger': The logging utility instance.
    """
    model = results['model']
    logger = results['logger']
    
    vprint("### Evaluation & Cleanup ###")
    
    # In a real scenario, we might calculate PSNR/SSIM here if ground truth exists.
    # For now, we just ensure logs are closed and distributed processes are cleaned up.
    
    if logger is not None and getattr(logger, 'flush_file', False):
        logger.close()
        
    if dist.is_initialized():
        dist.destroy_process_group()
```

### Verification Hints for Unit Tests
If you are writing a unit test for this function:
1.  **Mocking**: You must mock `torch.distributed.is_initialized` and `torch.distributed.destroy_process_group`.
2.  **Logger Stub**: Create a dummy class for the logger that has a `flush_file` attribute and a `close()` method.
3.  **Assertions**:
    *   Assert that `logger.close()` is called if `flush_file` is True.
    *   Assert that `dist.destroy_process_group()` is called **only if** `dist.is_initialized()` returns True.