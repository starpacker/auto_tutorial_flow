# FPM Paper Full Reproduction Tutorial

## Task Background and Paper Contributions
# 1. Task Background and Paper Contributions

Fourier Ptychographic Microscopy (FPM) is an advanced computational imaging technique that enhances the resolution of a standard optical microscope beyond its physical limitations by synthesizing a high-resolution image from multiple low-resolution images taken under varying illumination angles. This method combines principles of ptychography and Fourier optics, reconstructing the object's complex wavefront in both amplitude and phase, thereby providing rich spatial frequency information.

The core idea behind FPM is to iteratively reconstruct the high-resolution image in the Fourier domain through phase retrieval algorithms, which align and combine the low-resolution images captured by the microscope. This involves solving an optimization problem that estimates the best-fit high-resolution image that could give rise to the observed set of low-resolution images. 

However, traditional FPM algorithms are often limited by their high computational cost and large memory usage due to the exhaustive iterative processes and storage requirements for high-resolution data. This is where the introduction of Implicit Neural Representations (INRs) can play a transformative role. INRs represent continuous signals through neural networks, allowing for a compact and memory-efficient representation. In the context of FPM, INRs can significantly reduce the computational resources required for image reconstruction by leveraging their ability to encode spatial information in a compressed form.

The paper "FPM-INR" integrates implicit neural representations into the FPM framework, proposing a novel approach for image stack reconstruction that is both efficient and scalable. The key mathematical formulation involves transforming the problem of recovering the object's wavefront into a neural network optimization task, where the network weights implicitly model the spatial frequency components. This allows for the retrieval of high-frequency details without explicitly reconstructing the entire image stack in memory.

The contributions of the paper include a new methodology for coupling INRs with FPM, providing an end-to-end differentiable framework and demonstrating significant improvements in both speed and accuracy compared to traditional FPM approaches, along with the potential for real-time applications.

Key mathematical frameworks involved in this work include Fourier transforms, optimization techniques for phase retrieval, and neural network training protocols specific to INR modeling:

$$
\min_{\theta} \sum_{i} \| \mathcal{F}^{-1}( \mathbf{G}(\mathbf{k}_i) \odot \mathcal{F}(\mathbf{u}_\theta)) - \mathbf{I}_i \|^2
$$

where \(\mathcal{F}\) and \(\mathcal{F}^{-1}\) denote the Fourier and inverse Fourier transforms, \( \mathbf{G}(\mathbf{k}_i) \) represents the transfer function for each illumination angle, \(\odot\) is the element-wise multiplication, \(\mathbf{u}_\theta\) the implicit neural representation parameterized by weights \(\theta\), and \(\mathbf{I}_i\) the captured low-resolution images.

## Data Introduction and Acquisition Methods
## Data Introduction and Acquisition Methods

In Fourier Ptychographic Microscopy (FPM), the data acquisition process is critical as it forms the foundation for high-resolution image reconstruction. The core idea is to illuminate the sample with a sequence of angled light patterns, each generating a unique low-resolution image at the sensor. These images contain different portions of the sample's spatial frequency components. By capturing a comprehensive dataset using various illumination angles, we can assure the capture of complete frequency information required for synthesis of high-resolution images.

The primary motivation for this data acquisition step is to overcome the diffraction limit of the microscope's objective lens. Each captured image under a distinct illumination angle shifts parts of the high-resolution information into the observable passband of the lens, providing a piece of the puzzle. By later combining these pieces, a complete and enriched spatial frequency representation can be achieved, allowing for super-resolution effects.

The mathematical formulation underpinning this process involves the transfer function of the microscope for each angle of illumination, \( \mathbf{G}(\mathbf{k}_i) \), which determines how the spatial frequencies are shifted in each captured image. This is critical for the phase retrieval algorithm that operates in the Fourier domain. Mathematically, the captured image \( \mathbf{I}_i \) is modeled as:

$$
\mathbf{I}_i = \mathcal{F}^{-1}(\mathbf{G}(\mathbf{k}_i) \odot \mathbf{U}(\mathbf{k}))
$$

where \( \mathbf{U}(\mathbf{k}) \) represents the high-resolution Fourier domain image, and \(\odot\) is the multiplication operation that simulates how specific spatial frequency components are modulated by the transfer function. The goal is to obtain a comprehensive dataset that, through iterative computational methods, reveals these high-frequency components critical to image reconstruction.

Overall, the data acquisition process for FPM is meticulously designed to capture the necessary information that allows subsequent algorithms to reconstruct a high-fidelity image representation of the sample, pushing the observational capabilities well beyond conventional limits.

## Detailed Explanation of the Physical Process
## Detailed Explanation of the Physical Process

The physical process underlying Fourier Ptychographic Microscopy (FPM) is a sophisticated technique aimed at overcoming the limitations of conventional optical microscopy by achieving super-resolution imaging. This is done through strategic manipulation of light and careful data acquisition. The core idea is to use a sequence of controlled illumination angles to interact with the sample, ensuring that different spatial frequency components are brought into the observable range of the microscope’s sensor. This essentially shifts the high-resolution details into the captured images, which can be subsequently reconstructed into a high-fidelity representation of the sample.

Why is this step crucial? The diffraction limit of any optical system restricts the amount of detail that can be observed directly. This means high-frequency details of the sample are often lost or not captured by the sensor. FPM circumvents this limitation by employing angled illumination and computational reconstruction. The process begins by illuminating the sample under different angles, each corresponding to a specific shift in the spatial frequency domain. Each captured image \( \mathbf{I}_i \) records a subset of the frequencies, which, when properly combined, reconstruct the full spectrum necessary for super-resolution imaging.

The key mathematical principle involves the manipulation of frequency spectra using the transfer function \( \mathbf{G}(\mathbf{k}_i) \). This function dictates which frequencies are accessible under each illumination condition. Mathematically, this manipulation can be expressed as:

$$
\mathbf{I}_i = \mathcal{F}^{-1}(\mathbf{G}(\mathbf{k}_i) \odot \mathbf{U}(\mathbf{k}))
$$

where \(\mathcal{F}^{-1}\) represents the inverse Fourier transform, and \(\odot\) signifies element-wise multiplication. \(\mathbf{G}(\mathbf{k}_i)\) modulates the spatial frequency spectrum of the high-resolution image \(\mathbf{U}(\mathbf{k})\), effectively allowing the reconstruction from various image captures under different \( \mathbf{k}_i \).

This physical process of angular illumination and capture is vital for assembling the complete high-resolution image, transforming the achievable detail level well beyond the conventional constraints of optical microscopy.

## Data Preprocessing
## Data Preprocessing

Data preprocessing is a critical step in Fourier Ptychographic Microscopy (FPM), ensuring that the captured low-resolution images are prepared correctly for the subsequent reconstruction process. The core idea here is to convert the raw data into a structured format that can be utilized effectively by the computational algorithms, especially when integrated with Implicit Neural Representations (INRs). This involves calibrating the images, aligning them according to their corresponding illumination conditions, and setting up the necessary parameters to handle the spatial frequency data.

The necessity of preprocessing arises from the need to standardize data, minimize noise, and enhance the quality of inputs to the phase retrieval and reconstruction algorithms. This ensures that the reconstructed images are accurate representations of the sample. The preprocessing also involves transforming spatial domain data into the frequency domain, which aligns with the mathematical formulation of FPM where:

$$
\mathbf{I}_i = \mathcal{F}^{-1}(\mathbf{G}(\mathbf{k}_i) \odot \mathbf{U}(\mathbf{k}))
$$

Here, the captured low-resolution images \(\mathbf{I}_i\) are modeled in terms of their Fourier transforms and manipulation through the transfer function \(\mathbf{G}(\mathbf{k}_i)\).

A key mathematical aspect of this step involves calculating the necessary spatial frequency coordinates, illumination angles, and the Fourier domain representation to support accurate phase retrieval. The preprocessing also involves setting up spatial parameters like the numerical aperture, pixel size, and wavelength, which are critical for configuring the reconstruction algorithm correctly, thus bridging the physical setup with computational modeling.

# === Full Runnable Code Below ===

```python
def load_and_preprocess_data() -> dict:
    import mat73
    import scipy.io as sio
    import numpy as np
    import torch

    # Define constants for the sample, color, modes, and a 3D fitting flag
    sample = "BloodSmearTilt"
    color = "g"
    num_modes = 512
    fit_3D = True

    # Load data from a MATLAB file appropriate to the sample setting
    if fit_3D:
        data_struct = mat73.loadmat(f"data/{sample}/{sample}_{color}.mat")
    else:
        if sample == 'Siemens':
            data_struct = sio.loadmat(f"data/{sample}/{sample}_{color}.mat")
        else:
            data_struct = mat73.loadmat(f"data/{sample}/{sample}_{color}.mat")

    # Extract and convert the image matrix to a float32 type
    I = data_struct["I_low"].astype("float32")

    # Crop the image to a region of interest depending on the selected mode
    if fit_3D:
        I = I[0:int(num_modes*2), 0:int(num_modes*2), :]
    else:
        I = I[0:int(num_modes), 0:int(num_modes), :]

    # Establish raw measurement dimensions
    M = I.shape[0]
    N = I.shape[1]
    ID_len = I.shape[2]

    # Extract numerical apertures for illumination points
    NAs = data_struct["na_calib"].astype("float32")
    NAx = NAs[:, 0]
    NAy = NAs[:, 1]

    # Set the LED central wavelength based on the color channel
    if color == "r":
        wavelength = 0.632  # um
    elif color == "g":
        wavelength = 0.5126  # um
    elif color == "b":
        wavelength = 0.471  # um

    # Define LED spacing, free-space k-vector, and optical magnification
    D_led = 4000
    k0 = 2 * np.pi / wavelength
    mag = data_struct["mag"].astype("float32")

    # Determine pixel size at the image plane
    pixel_size = data_struct["dpix_c"].astype("float32")
    D_pixel = pixel_size / mag

    # Set the objective lens numerical aperture and calculate kmax
    NA = data_struct["na_cal"].astype("float32")
    kmax = NA * k0

    # Calculate the upsampling ratio and deduce pixel count in upsampled image
    MAGimg = 2 if sample != 'Siemens' else 3
    MM = int(M * MAGimg)
    NN = int(N * MAGimg)

    # Create spatial frequency coordinates mesh grid
    Fxx1, Fyy1 = np.meshgrid(np.arange(-NN / 2, NN / 2), np.arange(-MM / 2, MM / 2))
    Fxx1 = Fxx1[0, :] / (N * D_pixel) * (2 * np.pi)
    Fyy1 = Fyy1[:, 0] / (M * D_pixel) * (2 * np.pi)

    # Calculate light source angles in the frequency domain
    u = -NAx
    v = -NAy
    NAillu = np.sqrt(u**2 + v**2)
    order = np.argsort(NAillu)
    u = u[order]
    v = v[order]

    # Identify the pixel shifts for each LED illumination
    ledpos_true = np.zeros((ID_len, 2), dtype=int)
    for idx in range(ID_len):
        Fx1_temp = np.abs(Fxx1 - k0 * u[idx])
        ledpos_true[idx, 0] = np.argmin(Fx1_temp)
        Fy1_temp = np.abs(Fyy1 - k0 * v[idx])
        ledpos_true[idx, 1] = np.argmin(Fy1_temp)

    # Normalize the captured images to enhance contrast
    Isum = I[:, :, order] / np.max(I)

    # Calculate angular spectrum for propagation models
    if sample == 'Siemens':
        kxx, kyy = np.meshgrid(Fxx1[0, :M], Fxx1[0, :N])
    else:
        kxx, kyy = np.meshgrid(Fxx1[:M], Fxx1[:N])
    kxx, kyy = kxx - np.mean(kxx), kyy - np.mean(kyy)
    krr = np.sqrt(kxx**2 + kyy**2)
    mask_k = k0**2 - krr**2 > 0
    kzz_ampli = mask_k * np.abs(np.sqrt((k0**2 - krr.astype("complex64") ** 2)))
    kzz_phase = np.angle(np.sqrt((k0**2 - krr.astype("complex64") ** 2)))
    kzz = kzz_ampli * np.exp(1j * kzz_phase)

    # Define Pupil filter in the frequency domain
    Fx1, Fy1 = np.meshgrid(np.arange(-N / 2, N / 2), np.arange(-M / 2, M / 2))
    Fx2 = (Fx1 / (N * D_pixel) * (2 * np.pi)) ** 2
    Fy2 = (Fy1 / (M * D_pixel) * (2 * np.pi)) ** 2
    Fxy2 = Fx2 + Fy2
    Pupil0 = np.zeros((M, N))
    Pupil0[Fxy2 <= (kmax**2)] = 1

    # Convert computed arrays to PyTorch tensors
    Pupil0 = torch.from_numpy(Pupil0).view(1, 1, Pupil0.shape[0], Pupil0.shape[1])
    kzz = torch.from_numpy(kzz).unsqueeze(0)
    Isum = torch.from_numpy(Isum)

    # Configure depth of field and z-slice parameters for 3D models
    if fit_3D:
        DOF = 0.5 / NA**2
        delta_z = 0.8 * DOF
        z_max = 20.0
        z_min = -20.0
        num_z = int(np.ceil((z_max - z_min) / delta_z))
    else:
        z_min = 0.0
        z_max = 1.0

    # Assemble geometry and metadata relevant for reconstruction tasks
    geometry = {
        "grid": (M, N),
        "sources": ledpos_true,
        "receivers": None,  # Define if applicable
    }
    metadata = {
        "dt": None,  # Define if applicable
        "freq": None,  # Define if applicable
        "wavelength": wavelength,
        "D_led": D_led,
        "k0": k0,
        "mag": mag,
        "pixel_size": pixel_size,
        "D_pixel": D_pixel,
        "NA": NA,
        "kmax": kmax,
        "MAGimg": MAGimg,
        "z_min": z_min,
        "z_max": z_max,
        "num_z": num_z if fit_3D else None,
    }

    return {
        "observed_data": Isum,
        "initial_model": None,  # Define if applicable
        "geometry": geometry,
        "metadata": metadata,
    }
```


## Forward Operator Implementation
## Forward Operator Implementation

In the context of Fourier Ptychographic Microscopy (FPM) integrated with Implicit Neural Representations, the forward operator plays a crucial role in modeling how light propagates through and interacts with the sample. This process simulates the physical behavior of optical fields within the microscope system. The fundamental purpose of the forward operator is to predict the intensity of the optical field captured by the sensor from the sample, given its complex wavefront. By emulating the physical interaction, the forward operator provides critical feedback for the reconstruction algorithm, guiding it towards minimizing the difference between the observed and simulated data.

The core idea of the forward operator is to replicate how the optical field is transformed as it propagates through various components of the microscope setup. It involves transitioning the model, representing the sample in spatial coordinates, into the frequency domain via a Fourier transform. This switch allows the leveraging of convolutional properties, often necessary for applying system transfer functions. After applying appropriate masks and propagating the optical field, the result is transformed back to the spatial domain to evaluate the field intensity.

Mathematically, the forward operation starts with the Fourier transform of the model:

$$
O = \text{FFT}(\mathbf{model})
$$

Here, \(\mathbf{model}\) is the complex tensor encoding the object's amplitude and phase. The frequency domain representation \(O\) enables manipulation based on optical parameters. The operator then extracts a subsection of this spectrum according to the illumination setup using a mask and inverse Fourier transform to convert the data back to the spatial domain, resulting in:

$$
\mathbf{oI\_sub} = |\text{IFFT}(O_{\text{sub}} \odot \text{spectrum\_mask})|
$$

This intensity \(\mathbf{oI\_sub}\) can be directly compared against measured data to refine and validate reconstructions.

# === Full Runnable Code Below ===

```python
def forward_operator(model: torch.Tensor, geometry: dict) -> torch.Tensor:
    """
    Forward operator for simulating the optical field propagation in a brightfield microscope setup.

    Parameters:
    - model: A complex tensor representing the amplitude and phase of the optical field.
    - geometry: A dictionary containing the geometry and optical parameters such as:
        - 'led_num': List of LED indices.
        - 'x_0', 'y_0', 'x_1', 'y_1': Coordinates for sub-spectrum extraction.
        - 'spectrum_mask': Mask for the spectrum.
        - 'mag': Magnification factor.

    Returns:
    - oI_sub: The intensity of the optical field after propagation and sub-spectrum extraction.
    """

    # Perform Fourier transform to switch to frequency domain
    O = torch.fft.fftshift(torch.fft.fft2(model))
    
    # Calculate padding needed to match the spectrum mask size
    to_pad_x = (geometry['spectrum_mask'].shape[-2] * geometry['mag'] - O.shape[-2]) // 2
    to_pad_y = (geometry['spectrum_mask'].shape[-1] * geometry['mag'] - O.shape[-1]) // 2

    # Add zero padding around the frequency domain representation
    O = F.pad(O, (to_pad_x, to_pad_x, to_pad_y, to_pad_y, 0, 0), "constant", 0)

    # Extract sub-spectrum for each LED based on its coordinates
    O_sub = torch.stack(
        [O[:, geometry['x_0'][i]:geometry['x_1'][i], geometry['y_0'][i]:geometry['y_1'][i]] 
         for i in range(len(geometry['led_num']))], dim=1
    )

    # Apply the spectrum mask to simulate the system's optical transfer function
    O_sub = O_sub * geometry['spectrum_mask']
    
    # Perform inverse Fourier transform to return to the spatial domain
    o_sub = torch.fft.ifft2(torch.fft.ifftshift(O_sub))
    
    # Calculate and return the intensity of the optical field
    oI_sub = torch.abs(o_sub)

    return oI_sub
```

In this implementation, the function `forward_operator` accepts a model of the sample and geometrical parameters that dictate how the light propagates and interacts with the sample. A Fourier transform is used to switch to the frequency domain, pivotal for simulating wavefront interaction given limited measurements. Zero padding is employed to match with the spectrum mask, which represents the spatial frequency bandwidth constraints. After extracting relevant sub-spectra, an inverse transform returns the data to the spatial domain, enabling intensity computation, a critical step for comparing and refining the reconstruction process in FPM.

## Core Loop of Inverse Algorithm (Focus!)
## Core Loop of Inverse Algorithm

The inverse algorithm is central to reconstructing high-resolution images within Fourier Ptychographic Microscopy integrated with Implicit Neural Representations (FPM-INR). This process is pivotal as it iteratively refines the model parameters to minimize the discrepancy between simulated and observed low-resolution images, thereby improving the reconstructed object's fidelity. The core idea revolves around leveraging optimization techniques to adjust the neural network weights that implicitly encode the high-frequency components of the sample. The algorithm navigates through this complex optimization landscape by evaluating the predicted intensities against real measurements captured in experiments.

Mathematically, the inverse algorithm involves minimizing the difference between the measured intensity and the intensity predicted by the model. This is typically represented as:

$$
\min_{\theta}\sum_i \|\mathbf{oI\_cap}_i - \mathbf{oI\_sub}_i\|^2
$$

Here, \(\theta\) denotes the neural network parameters, \(\mathbf{oI\_cap}_i\) represents the observed intensity for each illumination angle, and \(\mathbf{oI\_sub}_i\) represents the simulated intensity derived from the model's forward pass.

The algorithm employs computational techniques such as gradient descent to iteratively update \(\theta\), ensuring convergence towards a solution where the simulated intensities align closely with the observed data. This is achieved through epochs of training where the model's outputs are continuously compared against the ground truth using loss functions like Smooth L1 Loss and Mean Squared Error (MSE).

# === Full Runnable Code Below ===

```python
def run_inversion():
    # Initialize model parameters and optimizer setup
    model = FullModel(
        w=MM, h=MM, num_feats=num_feats, x_mode=num_modes, y_mode=num_modes,
        z_min=z_min, z_max=z_max, ds_factor=cur_ds, use_layernorm=use_layernorm
    ).to(device)

    # Setup optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(
        lr=1e-3, params=filter(lambda p: p.requires_grad, model.parameters())
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_step, gamma=0.1
    )

    # Start loop over number of epochs for model training updates
    t = tqdm.trange(num_epochs)
    for epoch in t:
        # Randomize LED indices for batch processing
        led_idices = list(np.arange(ID_len))

        # Determine depth-of-field slices and initialize range
        if fit_3D:
            dzs = ((torch.randperm(num_z - 1)[: num_z // 2] +
                    torch.rand(num_z // 2)) * ((z_max - z_min) // (num_z - 1))
                   ).to(device) + z_min
            if epoch % 2 == 0:
                dzs = torch.linspace(z_min, z_max, num_z).to(device)
        else:
            dzs = torch.FloatTensor([0.0]).to(device)

        # Update model scaling factor if required, using coarse-to-fine approach
        if use_c2f and c2f_sche[epoch] < model.ds_factor:
            model.init_scale_grids(ds_factor=c2f_sche[epoch])
            print(f"ds_factor changed to {c2f_sche[epoch]}")
            model_fn = torch.jit.trace(model, dzs[0:1])

        # Compile model based on operating system
        if epoch == 0:
            if is_os == "Windows":
                model_fn = torch.jit.trace(model, dzs[0:1])
            elif is_os == "Linux":
                model_fn = torch.compile(model, backend="inductor")
            else:
                raise NotImplementedError

        # Process each depth slice and LED batches
        for dz in dzs:
            dz = dz.unsqueeze(0)
            for it in range(ID_len // led_batch_size):
                model.zero_grad()

                # Create spectrum phase mask for wavefront propagation simulation
                dfmask = torch.exp(
                    1j * kzz.repeat(dz.shape[0], 1, 1) *
                    dz[:, None, None].repeat(1, kzz.shape[1], kzz.shape[2])
                )
                led_num = led_idices[it * led_batch_size : (it + 1) * led_batch_size]
                dfmask = dfmask.unsqueeze(1).repeat(1, len(led_num), 1, 1)

                # Calculate amplitude and phase masks for spectrum corrections
                spectrum_mask_ampli = Pupil0.repeat(len(dz), len(led_num), 1, 1) * torch.abs(dfmask)
                spectrum_mask_phase = Pupil0.repeat(len(dz), len(led_num), 1, 1) * (
                    torch.angle(dfmask) + 0
                )
                spectrum_mask = spectrum_mask_ampli * torch.exp(1j * spectrum_mask_phase)

                # Autocast and compute outputs using AMP for precision enhancement
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                    img_ampli, img_phase = model_fn(dz)
                    img_complex = img_ampli * torch.exp(1j * img_phase)
                    uo, vo = ledpos_true[led_num, 0], ledpos_true[led_num, 1]
                    x_0, x_1 = vo - M // 2, vo + M // 2
                    y_0, y_1 = uo - N // 2, uo + N // 2

                    # Normalize and prepare captured intensity for comparison
                    oI_cap = torch.sqrt(Isum[:, :, led_num])
                    oI_cap = oI_cap.permute(2, 0, 1).unsqueeze(0).repeat(len(dz), 1, 1, 1)

                    # Simulate sub-spectrum using propogation model
                    oI_sub = get_sub_spectrum(
                        img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, MAGimg
                    )

                    # Calculate loss using smooth L1 and mean squared error
                    l1_loss = F.smooth_l1_loss(oI_cap, oI_sub)
                    loss = l1_loss
                    mse_loss = F.mse_loss(oI_cap, oI_sub)

                # Backpropagate loss and update model weights accordingly
                loss.backward()

                psnr = 10 * -torch.log10(mse_loss).item()
                t.set_postfix(Loss=f"{loss.item():.4e}", PSNR=f"{psnr:.2f}")
                optimizer.step()

        # Scheduler step to manage learning rate adjustments
        scheduler.step()

        # Save reconstructed amplitude and phase every few epochs
        if (epoch+1) % 10 == 0 or (epoch % 2 == 0 and epoch < 20) or epoch == num_epochs:
            if epoch == num_epochs - 1:
                np.save(f"{vis_dir}/last_amplitude.npy", img_ampli[0].float().cpu().detach().numpy())
                print(f"Saved last epoch amplitude data to {vis_dir}/last_amplitude.npy")
            amplitude = (img_ampli[0].float()).cpu().detach().numpy()
            phase = (img_phase[0].float()).cpu().detach().numpy()

            # Visualize amplitude and phase to monitor reconstruction progress
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
            im = axs[0].imshow(amplitude, cmap="gray")
            axs[0].axis("image")
            axs[0].set_title("Reconstructed amplitude")
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            im = axs[1].imshow(phase, cmap="gray")
            axs[1].axis("image")
            axs[1].set_title("Reconstructed phase")
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            plt.savefig(f"{vis_dir}/e_{epoch}.png")

        # Generate 3D visualizations of reconstructed fields if applicable
        if fit_3D and (epoch % 5 == 0 or epoch == num_epochs) and epoch > 0:
            dz = torch.linspace(z_min, z_max, 61).to(device).view(61)
            with torch.no_grad():
                out = []
                for z in torch.chunk(dz, 32):
                    img_ampli, img_phase = model(z)
                    _img_complex = img_ampli * torch.exp(1j * img_phase)
                    out.append(_img_complex)
                img_complex = torch.cat(out, dim=0)
            _imgs = img_complex.abs().cpu().detach().numpy()
            imgs = (_imgs - _imgs.min()) / (_imgs.max() - _imgs.min())
            save_gif(imgs, 'recon_amplitude.gif')
```

This detailed code block handles the core iterative process in reconstructing high-resolution images using the FPM-INR setup. The function `run_inversion` initializes the neural model, optimizer, and manages learning updates across epochs. It integrates advanced techniques such as adaptive scaling, dynamic phase masking, and learning rate scheduling to enhance convergence stability and reconstruction fidelity. By systematically applying these computational methods, the code ensures the model iteratively aligns with the real-world measurements collected during microscopy to achieve super-resolution effects.

## Training/Inversion Hyperparameters and Techniques
## Training/Inversion Hyperparameters and Techniques

The training and inversion process in Fourier Ptychographic Microscopy with Implicit Neural Representations (FPM-INR) is essential for achieving accurate super-resolution image reconstruction. This step is pivotal because it involves fine-tuning the model's hyperparameters and employing advanced techniques to ensure efficient convergence during the optimization process. The core idea is to systematically adjust parameters like learning rate, batch size, and scaling factors to optimize the model's performance in capturing high-frequency details that conventional microscopy misses.

The choice of hyperparameters such as learning rate, weight decay, and optimizer selection significantly impacts the model's ability to learn the complex mapping from low-resolution images to a high-resolution representation. A higher learning rate can lead to faster convergence but risks overshooting optimal solutions, while a lower rate generally offers more stability at the expense of speed. Weight decay helps prevent overfitting by penalizing large weights, thus maintaining generalization across the dataset.

Key mathematical concepts involved in this process include regularized loss functions like Smooth L1 Loss and Mean Squared Error (MSE), which measure the discrepancy between predicted and actual image intensities to guide weight updates. These functions can be defined as:

$$
\text{Smooth L1 Loss:} \quad \mathcal{L}_{\text{smooth}}(x, y) = \sum_i \frac{1}{2} (x_i - y_i)^2 \quad \text{if } |x_i - y_i| < 1
$$

$$
\text{Mean Squared Error:} \quad \mathcal{L}_{\text{mse}}(x, y) = \frac{1}{n} \sum_{i=1}^{n} (x_i - y_i)^2
$$

Another critical aspect is the use of learning rate schedulers, which adapt the learning rate dynamically during training, typically decreasing it by a fixed factor every few epochs to refine convergence as the model approaches its optimal configuration. Techniques such as automated mixed precision (AMP) are employed to speed up computations by reducing memory usage without compromising precision, crucial for handling large image datasets efficiently.

Overall, setting appropriate hyperparameters and incorporating robust training techniques ensures that the FPM-INR model effectively reconstructs high-resolution images while maintaining computational efficiency and accuracy. These strategies collectively facilitate resolving fine details beyond the diffraction limit, unlocking the full potential of super-resolution microscopy.

## Definition and Implementation of Evaluation Metrics
## Definition and Implementation of Evaluation Metrics

Evaluating the performance of the Fourier Ptychographic Microscopy with Implicit Neural Representations (FPM-INR) model is a crucial step to ensure its effectiveness in reconstructing high-resolution images. The core idea of this step is to quantitatively measure how well the reconstructed images align with the ground truth data in terms of clarity and fidelity. This involves using various metrics that assess image quality, such as Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), Root Mean Square Error (RMSE), and Relative L2 Norm. These metrics offer insights into the accuracy, perceptual quality, and structural details preserved in the reconstructed images.

PSNR essentially measures the ratio between the maximal possible power of a signal (image) and the power of distorting noise that affects its representation. It is computed as:

$$
\text{PSNR} = 20 \cdot \log_{10} \left( \frac{\text{MAX}_{I}}{\sqrt{\text{MSE}}} \right)
$$

where \(\text{MAX}_{I}\) is the maximum pixel value of the image and \(\text{MSE}\) is the Mean Squared Error between the observed and predicted images.

SSIM assesses the similarity between two images by considering changes in luminance, contrast, and structure, defined mathematically as:

$$
\text{SSIM}(x, y) = \frac{(2 \mu_x \mu_y + C_1)(2 \sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

where \(\mu\) and \(\sigma\) denote mean and variance, respectively, and \(C_1\), \(C_2\) are constants to stabilize the division.

RMSE measures the standard deviation of the prediction errors and is used to indicate the absolute differences between actual and predicted data, while Relative L2 Norm provides a normalized measure of error magnitude.

# === Full Runnable Code Below ===

```python
def evaluate_results(
    observed_data: torch.Tensor,
    predicted_data: torch.Tensor
):
    # Ensure the directory exists for saving the results
    os.makedirs("./results/", exist_ok=True)

    # Convert tensors to numpy arrays for metric computation
    observed_np = observed_data.cpu().numpy()
    predicted_np = predicted_data.cpu().numpy()

    # Compute Peak Signal-to-Noise Ratio (PSNR)
    psnr_value = psnr(observed_np, predicted_np, data_range=predicted_np.max() - predicted_np.min())

    # Compute Structural Similarity Index (SSIM)
    ssim_value = ssim(observed_np, predicted_np, data_range=predicted_np.max() - predicted_np.min())

    # Compute Root Mean Square Error (RMSE)
    rmse_value = np.sqrt(F.mse_loss(observed_data, predicted_data).item())

    # Compute Relative L2 Norm, normalized through division
    relative_l2 = F.mse_loss(observed_data, predicted_data).item() / F.mse_loss(observed_data, torch.zeros_like(observed_data)).item()

    # Print evaluated metrics for analysis
    print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, RMSE: {rmse_value:.4f}, Relative L2: {relative_l2:.4f}")

    # Visualize observed, predicted, and difference images using matplotlib
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

    # Plot observed data with a color bar for intensity reference
    im = axs[0].imshow(observed_np, cmap="gray")
    axs[0].axis("image")
    axs[0].set_title("Observed Data")
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Plot predicted data to visualize reconstruction quality
    im = axs[1].imshow(predicted_np, cmap="gray")
    axs[1].axis("image")
    axs[1].set_title("Predicted Data")
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Plot absolute difference to highlight reconstruction errors
    im = axs[2].imshow(np.abs(observed_np - predicted_np), cmap="gray")
    axs[2].axis("image")
    axs[2].set_title("Difference")
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Save the evaluation plots to the results directory
    plt.savefig("./results/evaluation.png")
    plt.close()
```

This code provides a comprehensive evaluation framework by leveraging image quality metrics to assess reconstructed images, aligning with theoretical measures of accuracy and similarity. Through visualization, researchers can intuitively discern the effectiveness of their FPM-INR model, finely tune operational parameters, and make informed improvements.

## Complete Experiment Reproduction Script
## Complete Experiment Reproduction Script

The experiment reproduction script is essential for validating the findings presented in the FPM-INR paper by replicating the methodology, results, and improvements reported. This step exemplifies the capability of Fourier ptychographic microscopy integrated with implicit neural representations to achieve high-resolution image reconstruction while minimizing computational overhead. The core idea behind this script is to ensure that every aspect of the experiment, from data loading to image reconstruction, is consistent and reproducible across different environments. This reproducibility is crucial in scientific research, as it allows independent verification and potential further exploration or enhancement of the proposed methodologies.

Key mathematical components involved in this script build upon the equations and methods outlined throughout the tutorial. For instance, the reconstruction process centers around minimizing the discrepancy between observed and simulated images, expressed mathematically as a regularized optimization problem:

$$
\min_{\theta} \sum_{i} \| \mathcal{F}^{-1}( \mathbf{G}(\mathbf{k}_i) \odot \mathcal{F}(\mathbf{u}_\theta)) - \mathbf{I}_i \|^2
$$

where \(\mathcal{F}\) and \(\mathcal{F}^{-1}\) are Fourier and inverse Fourier transforms, and \(\theta\) are the neural network parameters. The script operationalizes these equations, managing parameters such as the learning rate, iteration count, and data sources through a structured coding approach. It also applies advanced computing techniques such as automated mixed precision (AMP) for enhancing computational efficiency and leveraging GPU acceleration wherever possible.

By following this complete script, researchers can validate the paper's findings, assess the robustness of the proposed INR-based reconstruction method, and explore its applicability to various imaging challenges. This serves as a foundation for subsequent explorations into optimizing image reconstruction in Fourier ptychographic microscopy, fostering innovation in microscopy and computational imaging.

## Result Reproduction and Visualization Comparison
## Result Reproduction and Visualization Comparison

The "Result Reproduction and Visualization Comparison" step is crucial for validating the effectiveness of the Fourier Ptychographic Microscopy with Implicit Neural Representations (FPM-INR) model. The core idea here is to ensure that the outcomes achieved are not just theoretical predictions, but are reproducible and comparable to the original findings presented in the FPM-INR paper. This step is important because it enables independent verification and confirms the consistency and robustness of the proposed methodologies across different datasets and experimental conditions.

Why is this step critical? In scientific research, result reproduction is paramount for establishing the reliability and credibility of a novel approach. It also opens the door to further enhancements and adaptations as the fundamental accuracy and applicability of the approach are substantiated. Moreover, visualization comparison plays a key role in qualitatively assessing the improvements brought about by the FPM-INR model in terms of resolution, clarity, and fidelity to the underlying sample.

Key mathematical principles underpinning this section include the evaluation of image quality metrics such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM), beyond RMSE and Relative L2 Norm. These metrics provide quantitative benchmarks to compare the reconstructed images to the ground truth:

$$
\text{PSNR} = 20 \cdot \log_{10} \left( \frac{\text{MAX}_{I}}{\sqrt{\text{MSE}}} \right)
$$

$$
\text{SSIM}(x, y) = \frac{(2 \mu_x \mu_y + C_1)(2 \sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

Visualization comparison entails plotting the observed versus reconstructed images and differences to visually illustrate the system's efficacy. This step ensures that the high-resolution reconstructions achieved through FPM-INR are consistent with expectations and demonstrate marked improvement over traditional methods, reinforcing the methodology's contribution to the field of computational microscopy. Through meticulous result reproduction and visualization, the potential for real-world applications of FPM-INR in advanced imaging scenarios is substantiated.

