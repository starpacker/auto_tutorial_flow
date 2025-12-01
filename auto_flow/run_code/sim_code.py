import math
import warnings
import numpy as np
from numpy import zeros, log
import gc
import time
import pywt
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
import matplotlib.pyplot as plt

# Global array module (numpy by default, can be cupy for GPU)
xp = np
cp = None

# ============================================================================
# Helper Functions
# ============================================================================

def cart2pol(x, y):
    """Convert Cartesian to polar coordinates."""
    rho = xp.sqrt(x ** 2 + y ** 2)
    phi = xp.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    """Convert polar to Cartesian coordinates."""
    x = rho * xp.cos(phi)
    y = rho * xp.sin(phi)
    return (x, y)


def psf2otf(psf, outSize):
    """Convert point spread function to optical transfer function."""
    psfSize = xp.array(psf.shape)
    outSize = xp.array(outSize)
    padSize = xp.array(outSize - psfSize)
    psf = xp.pad(psf, ((0, int(padSize[0])), (0, int(padSize[1]))), 'constant')
    for i in range(len(psfSize)):
        psf = xp.roll(psf, -int(psfSize[i] / 2), i)
    otf = xp.fft.fftn(psf)
    nElem = xp.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * xp.log2(psfSize[k]) * nffts
    if xp.max(xp.abs(xp.imag(otf))) / xp.max(xp.abs(otf)) <= nOps * xp.finfo(xp.float32).eps:
        otf = xp.real(otf)
    return otf


def rliter(yk, data, otf):
    """Richardson-Lucy iteration helper."""
    rliter_val = xp.fft.fftn(data / xp.maximum(xp.fft.ifftn(otf * xp.fft.fftn(yk)), 1e-6))
    return rliter_val


def Gauss(sigma):
    """Generate Gaussian PSF kernel."""
    sigma = np.array(sigma, dtype='float32')
    s = sigma.size
    if s == 1:
        sigma = [sigma, sigma]
    sigma = np.array(sigma, dtype='float32')
    psfN = np.ceil(sigma / math.sqrt(8 * log(2)) * math.sqrt(-2 * log(0.0002))) + 1
    N = psfN * 2 + 1
    sigma = sigma / (2 * math.sqrt(2 * log(2)))
    dim = len(N)
    if dim > 1:
        N[1] = np.maximum(N[0], N[1])
        N[0] = N[1]
    if dim == 1:
        x = np.arange(-np.fix(N / 2), np.ceil(N / 2), dtype='float32')
        PSF = np.exp(-0.5 * (x * x) / (np.dot(sigma, sigma)))
        PSF = PSF / PSF.sum()
        return PSF
    if dim == 2:
        m = N[0]
        n = N[1]
        x = np.arange(-np.fix((n / 2)), np.ceil((n / 2)), dtype='float32')
        y = np.arange(-np.fix((m / 2)), np.ceil((m / 2)), dtype='float32')
        X, Y = np.meshgrid(x, y)
        s1 = sigma[0]
        s2 = sigma[1]
        PSF = np.exp(-(X * X) / (2 * np.dot(s1, s1)) - (Y * Y) / (2 * np.dot(s2, s2)))
        PSFsum = PSF.sum()
        PSF = PSF / PSFsum
        return PSF
    if dim == 3:
        m = N[0]
        n = N[1]
        k = N[2]
        x = np.arange(-np.fix(n / 2), np.ceil(n / 2), dtype='float32')
        y = np.arange(-np.fix(m / 2), np.ceil(m / 2), dtype='float32')
        z = np.arange(-np.fix(k / 2), np.ceil(k / 2), dtype='float32')
        [X, Y, Z] = np.meshgrid(x, y, z)
        s1 = sigma[0]
        s2 = sigma[1]
        s3 = sigma[2]
        PSF = np.exp(-(X * X) / (2 * s1 * s1) - (Y * Y) / (2 * s2 * s2) - (Z * Z) / (2 * s3 ** 2))
        PSFsum = PSF.sum()
        PSF = PSF / PSFsum
        return PSF


def deblur_core(data, kernel, iteration, rule):
    """Core deblurring function."""
    kernel = xp.array(kernel)
    kernel = kernel / sum(sum(kernel))
    kernel_initial = kernel
    [dx, dy] = data.shape

    B = math.floor(min(dx, dy) / 6)
    data = xp.pad(data, [int(B), int(B)], 'edge')
    yk = data
    xk = zeros((data.shape[0], data.shape[1]), dtype='float32')
    vk = zeros((data.shape[0], data.shape[1]), dtype='float32')
    otf = psf2otf(kernel_initial, data.shape)

    if rule == 2:
        # LandWeber deconv
        t = 1
        gamma1 = 1
        for i in range(0, iteration):
            if i == 0:
                xk_update = data
                xk = data + t * xp.fft.ifftn(xp.conj(otf)) * (xp.fft.fftn(data) - (otf * xp.fft.fftn(data)))
            else:
                gamma2 = 1 / 2 * (4 * gamma1 * gamma1 + gamma1 ** 4) ** (1 / 2) - gamma1 ** 2
                beta = -gamma2 * (1 - 1 / gamma1)
                yk_update = xk + beta * (xk - xk_update)
                yk = yk_update + t * xp.fft.ifftn(xp.conj(otf) * (xp.fft.fftn(data) - (otf * xp.fft.fftn(yk_update))))
                yk = xp.maximum(yk, 1e-6, dtype='float32')
                gamma1 = gamma2
                xk_update = xk
                xk = yk

    elif rule == 1:
        # Richardson-Lucy deconv
        for iter in range(0, iteration):
            xk_update = xk
            rliter1 = rliter(yk, data, otf)
            xk = yk * ((xp.fft.ifftn(xp.conj(otf) * rliter1)).real) / ((xp.fft.ifftn(xp.fft.fftn(xp.ones(data.shape)) * otf)).real)
            xk = xp.maximum(xk, 1e-6, dtype='float32')
            vk_update = vk
            vk = xp.maximum(xk - yk, 1e-6, dtype='float32')

            if iter == 0:
                alpha = 0
                yk = xk
                yk = xp.maximum(yk, 1e-6, dtype='float32')
                yk = xp.array(yk)
            else:
                alpha = sum(sum(vk_update * vk)) / (sum(sum(vk_update * vk_update)) + 1e-10)
                alpha = xp.maximum(xp.minimum(alpha, 1), 1e-6, dtype='float32')
                yk = xk + alpha * (xk - xk_update)
                yk = xp.maximum(yk, 1e-6, dtype='float32')
                yk[xp.isnan(yk)] = 1e-6

    yk[yk < 0] = 0
    yk = xp.array(yk, dtype='float32')
    data_decon = yk[B + 0:yk.shape[0] - B, B + 0: yk.shape[1] - B]

    return data_decon


def iterative_deconv(data, kernel, iteration, rule):
    """Iterative deconvolution wrapper."""
    if xp is not np:
        data = xp.asarray(data)
        kernel = xp.asarray(kernel)

    if data.ndim > 2:
        data_de = xp.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype='float32')
        for i in range(0, data.shape[2]):
            data_de[:, :, i] = (deblur_core(data[:, :, i], kernel, iteration, rule)).real
    else:
        data_de = (deblur_core(data, kernel, iteration, rule)).real

    if xp is not np:
        data_de = xp.asnumpy(data_de)

    return data_de


# Operation functions for sparse Hessian
def operation_xx(gsize):
    delta_xx = xp.array([[[1, -2, 1]]], dtype='float32')
    xxfft = xp.fft.fftn(delta_xx, gsize) * xp.conj(xp.fft.fftn(delta_xx, gsize))
    return xxfft


def operation_xy(gsize):
    delta_xy = xp.array([[[1, -1], [-1, 1]]], dtype='float32')
    xyfft = xp.fft.fftn(delta_xy, gsize) * xp.conj(xp.fft.fftn(delta_xy, gsize))
    return xyfft


def operation_xz(gsize):
    delta_xz = xp.array([[[1, -1]], [[-1, 1]]], dtype='float32')
    xzfft = xp.fft.fftn(delta_xz, gsize) * xp.conj(xp.fft.fftn(delta_xz, gsize))
    return xzfft


def operation_yy(gsize):
    delta_yy = xp.array([[[1], [-2], [1]]], dtype='float32')
    yyfft = xp.fft.fftn(delta_yy, gsize) * xp.conj(xp.fft.fftn(delta_yy, gsize))
    return yyfft


def operation_yz(gsize):
    delta_yz = xp.array([[[1], [-1]], [[-1], [1]]], dtype='float32')
    yzfft = xp.fft.fftn(delta_yz, gsize) * xp.conj(xp.fft.fftn(delta_yz, gsize))
    return yzfft


def operation_zz(gsize):
    delta_zz = xp.array([[[1]], [[-2]], [[1]]], dtype='float32')
    zzfft = xp.fft.fftn(delta_zz, gsize) * xp.conj(xp.fft.fftn(delta_zz, gsize))
    return zzfft


# Sparse iteration helper functions
def forward_diff(data, step, dim):
    """Forward difference operator."""
    assert dim <= 2
    r, n, m = np.shape(data)
    size = np.array((r, n, m))
    position = xp.zeros(3, dtype='float32')
    temp1 = xp.zeros(size + 1, dtype='float32')
    temp2 = xp.zeros(size + 1, dtype='float32')

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp1[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data
    temp2[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data

    size[dim] = size[dim] - 1
    temp2[0:size[0], 0:size[1], 0:size[2]] = data
    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] + 1

    out = temp1[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])]
    return -out


def back_diff(data, step, dim):
    """Backward difference operator."""
    assert dim <= 2
    r, n, m = np.shape(data)
    size = np.array((r, n, m))
    position = np.zeros(3, dtype='float32')
    temp1 = xp.zeros(size + 1, dtype='float32')
    temp2 = xp.zeros(size + 1, dtype='float32')

    temp1[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data
    temp2[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp2[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data

    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] - 1
    out = temp1[0:size[0], 0:size[1], 0:size[2]]
    return out


def shrink(x, L):
    """Shrinkage operator for L1 regularization."""
    s = xp.abs(x)
    xs = xp.sign(x) * xp.maximum(s - 1 / L, 0)
    return xs


def iter_xx(g, bxx, para, mu):
    gxx = back_diff(forward_diff(g, 1, 1), 1, 1)
    dxx = shrink(gxx + bxx, mu)
    bxx = bxx + (gxx - dxx)
    Lxx = para * back_diff(forward_diff(dxx - bxx, 1, 1), 1, 1)
    return Lxx, bxx


def iter_xy(g, bxy, para, mu):
    gxy = forward_diff(forward_diff(g, 1, 1), 1, 2)
    dxy = shrink(gxy + bxy, mu)
    bxy = bxy + (gxy - dxy)
    Lxy = para * back_diff(back_diff(dxy - bxy, 1, 2), 1, 1)
    return Lxy, bxy


def iter_xz(g, bxz, para, mu):
    gxz = forward_diff(forward_diff(g, 1, 1), 1, 0)
    dxz = shrink(gxz + bxz, mu)
    bxz = bxz + (gxz - dxz)
    Lxz = para * back_diff(back_diff(dxz - bxz, 1, 0), 1, 1)
    return Lxz, bxz


def iter_yy(g, byy, para, mu):
    gyy = back_diff(forward_diff(g, 1, 2), 1, 2)
    dyy = shrink(gyy + byy, mu)
    byy = byy + (gyy - dyy)
    Lyy = para * back_diff(forward_diff(dyy - byy, 1, 2), 1, 2)
    return Lyy, byy


def iter_yz(g, byz, para, mu):
    gyz = forward_diff(forward_diff(g, 1, 2), 1, 0)
    dyz = shrink(gyz + byz, mu)
    byz = byz + (gyz - dyz)
    Lyz = para * back_diff(back_diff(dyz - byz, 1, 0), 1, 2)
    return Lyz, byz


def iter_zz(g, bzz, para, mu):
    gzz = back_diff(forward_diff(g, 1, 0), 1, 0)
    dzz = shrink(gzz + bzz, mu)
    bzz = bzz + (gzz - dzz)
    Lzz = para * back_diff(forward_diff(dzz - bzz, 1, 0), 1, 0)
    return Lzz, bzz


def iter_sparse(gsparse, bsparse, para, mu):
    dsparse = shrink(gsparse + bsparse, mu)
    bsparse = bsparse + (gsparse - dsparse)
    Lsparse = para * (dsparse - bsparse)
    return Lsparse, bsparse


def sparse_hessian(f, iteration_num=1000, fidelity=150, sparsity=10, contiz=0.5, mu=1):
    """
    Sparse Hessian reconstruction.
    
    Source code for argmin_g { ||f-g ||_2^2 +||gxx||_1+||gxx||_1+||gyy||_1+lamdbaz*||gzz||_1+2*||gxy||_1
     +2*sqrt(lamdbaz)||gxz||_1+ 2*sqrt(lamdbaz)|||gyz||_1+2*sqrt(lamdbal1)|||g||_1}
    
    Parameters
    ----------
    f : ndarray
        Input image (can be N dimensional).
    iteration_num : int, optional
        The iteration of sparse hessian {default: 1000}
    fidelity : int, optional
        Fidelity {default: 150}
    contiz : float, optional
        Continuity along z-axial {default: 0.5}
    sparsity : int, optional
        Sparsity {default: 10}
    mu : float, optional
        Regularization parameter {default: 1}
    
    Returns
    -------
    g : ndarray
        Reconstructed image.
    """
    if xp is not cp:
        contiz = np.sqrt(contiz)
        f1 = f
    else:
        contiz = cp.sqrt(contiz)
        f1 = cp.asarray(f, dtype='float32')
    
    flage = 0
    f_flag = f.ndim
    
    if f_flag == 2:
        contiz = 0
        flage = 1
        f = xp.zeros((3, f.shape[0], f.shape[1]), dtype='float32')
        f = xp.array(f)
        for i in range(0, 3):
            f[i, :, :] = f1
    elif f_flag > 2:
        if f1.shape[0] < 3:
            contiz = 0
            f = xp.zeros((3, f.shape[1], f.shape[2]), dtype='float32')
            f[0:f1.shape[0], :, :] = f1
            for i in range(f1.shape[0], 3):
                f[i, :, :] = f[1, :, :]
        else:
            f = f1
    
    imgsize = xp.shape(f)

    print("Start the Sparse deconvolution...")
    
    # Calculate derivative operators
    xxfft = operation_xx(imgsize)
    yyfft = operation_yy(imgsize)
    zzfft = operation_zz(imgsize)
    xyfft = operation_xy(imgsize)
    xzfft = operation_xz(imgsize)
    yzfft = operation_yz(imgsize)

    operationfft = xxfft + yyfft + (contiz ** 2) * zzfft + 2 * xyfft + 2 * (contiz) * xzfft + 2 * (contiz) * yzfft
    normlize = (fidelity / mu) + (sparsity ** 2) + operationfft
    del xxfft, yyfft, zzfft, xyfft, xzfft, yzfft, operationfft
    gc.collect()
    
    # Initialize b
    bxx = xp.zeros(imgsize, dtype='float32')
    byy = bxx
    bzz = bxx
    bxy = bxx
    bxz = bxx
    byz = bxx
    bl1 = bxx
    
    # Initialize g
    g_update = xp.multiply(fidelity / mu, f)
    
    # Iteration
    tol = 1e-4
    residual_prev = xp.inf
    
    for iter in range(0, iteration_num):
        g_update = xp.fft.fftn(g_update)

        if iter == 0:
            g = xp.fft.ifftn(g_update / (fidelity / mu)).real
        else:
            g = xp.fft.ifftn(xp.divide(g_update, normlize)).real

        g_update = xp.multiply((fidelity / mu), f)

        Lxx, bxx = iter_xx(g, bxx, 1, mu)
        g_update = g_update + Lxx
        del Lxx
        gc.collect()

        Lyy, byy = iter_yy(g, byy, 1, mu)
        g_update = g_update + Lyy
        del Lyy
        gc.collect()

        Lzz, bzz = iter_zz(g, bzz, contiz ** 2, mu)
        g_update = g_update + Lzz
        del Lzz
        gc.collect()

        Lxy, bxy = iter_xy(g, bxy, 2, mu)
        g_update = g_update + Lxy
        del Lxy
        gc.collect()

        Lxz, bxz = iter_xz(g, bxz, 2 * contiz, mu)
        g_update = g_update + Lxz
        del Lxz
        gc.collect()

        Lyz, byz = iter_yz(g, byz, 2 * contiz, mu)
        g_update = g_update + Lyz
        del Lyz
        gc.collect()

        Lsparse, bl1 = iter_sparse(g, bl1, sparsity, mu)
        g_update = g_update + Lsparse
        del Lsparse
        gc.collect()

        # Check convergence
        if iter % 20 == 0:
            residual = xp.linalg.norm(f - g)
            rel_change = abs(residual - residual_prev) / (residual_prev + 1e-12)
            if rel_change < tol:
                print(f"Converged at iteration {iter}: residual change = {rel_change:.2e}")
                break
            residual_prev = residual

        if iter % 20 == 0:
            print('%d iterations done\r' % iter)

    g[g < 0] = 0

    del bxx, byy, bzz, bxy, byz, bl1, f, normlize, g_update
    gc.collect()

    return g[1, :, :] if flage else g


# Background estimation functions
def Low_frequency_resolve(coeffs, dlevel):
    """Extract low frequency components from wavelet coefficients."""
    cAn = coeffs[0]
    vec = []
    vec.append(cAn)
    for i in range(1, dlevel + 1):
        (cH, cV, cD) = coeffs[i]
        [cH_x, cH_y] = cH.shape
        cH_new = np.zeros((cH_x, cH_y))
        t = (cH_new, cH_new, cH_new)
        vec.append(t)
    return vec


def rm_1(Biter, x, y):
    """Remove extra pixels from wavelet reconstruction."""
    Biter_new = np.zeros((x, y), dtype=('uint8'))
    if x % 2 and y % 2 == 0:
        Biter_new[:, :] = Biter[0:x, :]
    elif x % 2 == 0 and y % 2:
        Biter_new[:, :] = Biter[:, 0:y]
    elif x % 2 and y % 2:
        Biter_new[:, :] = Biter[0:x, 0:y]
    else:
        Biter_new = Biter
    return Biter_new


def background_estimation(imgs, th=1, dlevel=7, wavename='db6', iter=3):
    """
    Background estimation using wavelet decomposition.
    
    Parameters
    ----------
    imgs : ndarray
        Input image (can be T × X × Y).
    th : int, optional
        If iteration {default: 1}
    dlevel : int, optional
        Decomposition level {default: 7}
    wavename : str, optional
        The selected wavelet function {default: 'db6'}
    iter : int, optional
        Iteration {default: 3}
    
    Returns
    -------
    Background : ndarray
        Estimated background.
    """
    try:
        [t, x, y] = imgs.shape
        Background = np.zeros((t, x, y))
        for taxial in range(t):
            img = imgs[taxial, :, :]
            for i in range(iter):
                initial = img
                res = initial
                coeffs = pywt.wavedec2(res, wavelet=wavename, level=dlevel)
                vec = Low_frequency_resolve(coeffs, dlevel)
                Biter = pywt.waverec2(vec, wavelet=wavename)
                Biter_new = rm_1(Biter, x, y)
                if th > 0:
                    eps = np.sqrt(np.abs(res)) / 2
                    ind = initial > (Biter_new + eps)
                    res[ind] = Biter_new[ind] + eps[ind]
                    coeffs1 = pywt.wavedec2(res, wavelet=wavename, level=dlevel)
                    vec = Low_frequency_resolve(coeffs1, dlevel)
                    Biter = pywt.waverec2(vec, wavelet=wavename)
                    Biter_new = rm_1(Biter, x, y)
                    Background[taxial, :, :] = Biter_new
    except ValueError:
        [x, y] = imgs.shape
        Background = np.zeros((x, y))
        for i in range(iter):
            initial = imgs
            res = initial
            coeffs = pywt.wavedec2(res, wavelet=wavename, level=dlevel)
            vec = Low_frequency_resolve(coeffs, dlevel)
            Biter = pywt.waverec2(vec, wavelet=wavename)
            Biter_new = rm_1(Biter, x, y)
            if th > 0:
                eps = np.sqrt(np.abs(res)) / 2
                ind = initial > (Biter_new + eps)
                res[ind] = Biter_new[ind] + eps[ind]
                coeffs1 = pywt.wavedec2(res, wavelet=wavename, level=dlevel)
                vec = Low_frequency_resolve(coeffs1, dlevel)
                Biter = pywt.waverec2(vec, wavelet=wavename)
                Biter_new = rm_1(Biter, x, y)
                Background = Biter_new
    return Background


# Upsampling functions
def spatial_upsample(SIMmovie, n=2):
    """Spatial upsampling by factor n."""
    if xp is not cp:
        SIMmovie = SIMmovie
    else:
        SIMmovie = cp.asarray(SIMmovie)

    k = SIMmovie.ndim
    if k > 2:
        [sz, sx, sy] = SIMmovie.shape
        for frames in range(0, sz):
            y = xp.zeros((sz, sx * n, sy * n), dtype='float32')
            y = xp.array(y)
            y[frames, 0:sx * n:n, 0:sy * n:n] = SIMmovie[frames, :, :]
            y = xp.array(y)
        return y
    else:
        [sx, sy] = SIMmovie.shape
        y = xp.zeros((sx * n, sy * n), dtype='float32')
        y[0:sx * n:n, 0:sy * n:n] = SIMmovie
        return y


def fInterp_2D(img, newsz):
    """2D Fourier interpolation."""
    imgsz = img.shape
    imgsz = xp.array(imgsz)
    newsz = xp.array(newsz)
    if (xp.sum(newsz == 0)) >= 1:
        img_ip = []
    isgreater = newsz >= imgsz
    isgreater = isgreater.astype(int)
    isgreater = xp.array(isgreater)
    incr = xp.zeros((2, 1), dtype='float32')
    for iDim in range(0, 2):
        if isgreater[0][iDim] == 1:
            incr[iDim] = 1
        else:
            incr = xp.floor(imgsz[iDim] / newsz[iDim]) + 1
    newsz[0][0] = int(newsz[0][0])
    a = newsz[0][0]
    b = newsz[0][1]
    nyqst = xp.ceil((imgsz + 1) / 2)
    B = float(a / imgsz[0] * b / imgsz[1])
    img = B * xp.fft.fft2(img)
    img_ip = xp.zeros((int(a), int(b)), dtype='complex')
    img_ip[0: int(nyqst[0]), 0: int(nyqst[1])] = img[0: int(nyqst[0]), 0: int(nyqst[1])]
    img_ip[a - (int(imgsz[0]) - int(nyqst[0])):a, 0:int(nyqst[1])] = img[int(nyqst[0]):int(imgsz[0]), 0:int(nyqst[1])]
    img_ip[0: int(nyqst[0]), a - (int(imgsz[1]) - int(nyqst[1])):a] = img[0: int(nyqst[0]), int(nyqst[1]): int(imgsz[1])]
    img_ip[a - (int(imgsz[0]) - int(nyqst[0])):a, a - (int(imgsz[1]) - int(nyqst[1])):a] = img[int(nyqst[0]):int(imgsz[0]), int(nyqst[1]):int(imgsz[1])]
    rm = xp.remainder(imgsz, 2)
    if int(rm[0]) == 0 & int(a) != int(imgsz[0]):
        img_ip[int(nyqst[0]), :] = img_ip[int(nyqst[0]), :] / 2
        img_ip[int(nyqst[0]) + int(a) - int(imgsz[0]), :] = img_ip[int(nyqst[0]), :]
    if int(rm[1]) == 0 & int(b) != int(imgsz[1]):
        img_ip[:, int(nyqst[1])] = img_ip[:, int(nyqst[1])] / 2
        img_ip[:, int(nyqst[1]) + int(b) - imgsz[1]] = img_ip[:, int(nyqst[1])]
    img_ip = xp.array(img_ip)
    img_ip = (xp.fft.ifft2(img_ip)).real
    img_ip = img_ip[0: int(a):int(incr[0]), 0:int(b): int(incr[1])]
    return img_ip


def fourier_upsample(imgstack, n=2):
    """
    Fourier interpolation upsampling.
    
    Parameters
    ----------
    imgstack : ndarray
        Input image (can be N dimensional).
    n : int, optional
        Magnification times {default: 2}
    
    Returns
    -------
    imgfl : ndarray
        Upsampled image.
    """
    if xp is not cp:
        imgstack = imgstack
    else:
        imgstack = cp.asarray(imgstack)

    n = n * xp.ones((1, 2))
    if imgstack.ndim < 3:
        z = 1
        sz = [imgstack.shape[0], imgstack.shape[1]]
        imgfl = xp.zeros((int(n[0][0]) * int(sz[0]), int(n[0][0]) * int(sz[1])))
    else:
        z = imgstack.shape[0]
        sz = [imgstack.shape[1], imgstack.shape[2]]
        imgfl = xp.zeros((z, int(n[0][0]) * int(sz[0]), int(n[0][0]) * int(sz[1])))

    for i in range(0, z):
        if imgstack.ndim < 3:
            img = imgstack
        else:
            img = imgstack[i, :, :]
        imgsz = [img.shape[0], img.shape[1]]
        imgsz = xp.array(imgsz)
        if ((imgsz[0] % 2)) == 1:
            sz = imgsz
        else:
            sz = imgsz - 1
        sz = xp.array(sz)
        idx = xp.ceil(sz / 2) + 1 + (n - 1) * xp.floor(sz / 2)
        padsize = [img.shape[0] / 2, img.shape[1] / 2]
        padsize = xp.array(padsize)
        k = xp.ceil(padsize)
        f = xp.floor(padsize)

        img = xp.pad(img, ((int(k[0]), 0), (int(k[1]), 0)), 'symmetric')
        img = xp.pad(img, ((0, int(f[0])), (0, int(f[1]))), 'symmetric')

        im_shape = n * (xp.array(img.shape))
        newsz = xp.floor(im_shape - (n - 1))
        imgl = fInterp_2D(img, newsz)
        if imgstack.ndim < 3:
            imgfl = imgl[int(idx[0][0]):int(n[0][0]) * int(imgsz[0]) + int(idx[0][0]),
                         int(idx[0][1]):int(idx[0][1]) + int(n[0][1]) * int(imgsz[1])]
        else:
            imgfl = xp.array(imgfl)
            imgfl[i, :, :] = imgl[int(idx[0][0]):int(n[0][0]) * int(imgsz[0]) + int(idx[0][0]),
                                  int(idx[0][1]):int(idx[0][1]) + int(n[0][1]) * int(imgsz[1])]

    return imgfl


# ============================================================================
# Main Functions
# ============================================================================

def load_and_preprocess_data(input_path):
    """
    Load and preprocess the input image data.
    
    Parameters
    ----------
    input_path : str
        Path to the input image file.
    
    Returns
    -------
    img : ndarray
        Preprocessed image data.
    scaler : float
        Original maximum value for rescaling.
    original_dtype : dtype
        Original data type of the image.
    """
    print(f"Loading image from: {input_path}")
    img = imread(input_path)
    original_dtype = img.dtype
    
    # Convert to float32 for processing
    img = np.array(img, dtype='float32')
    scaler = np.max(img)
    img = img / scaler
    
    # Background estimation and subtraction
    print("Estimating and subtracting background...")
    backgrounds = background_estimation(img / 2.5)
    img = img - backgrounds
    
    # Normalize
    img = img / (img.max())
    img[img < 0] = 0
    
    print(f"Image shape: {img.shape}, dtype: {original_dtype}")
    
    return img, scaler, original_dtype


def forward_operator(sigma, up_sample=0):
    """
    Construct the forward operator (PSF/kernel) for the imaging problem.
    
    Parameters
    ----------
    sigma : float or list
        The point spread function size in pixels.
    up_sample : int, optional
        Upsampling type:
        0: No upsampling
        1: Fourier upsampling
        2: Spatial upsampling
    
    Returns
    -------
    kernel : ndarray
        The Gaussian PSF kernel.
    up_sample_func : callable or None
        The upsampling function to apply, or None.
    """
    print(f"Constructing forward operator with sigma={sigma}")
    
    # Generate Gaussian PSF kernel
    if sigma:
        kernel = Gauss(sigma)
        print(f"PSF kernel shape: {kernel.shape}")
    else:
        kernel = None
        print("No PSF specified, skipping deconvolution")
    
    # Select upsampling function
    if up_sample == 1:
        up_sample_func = fourier_upsample
        print("Using Fourier upsampling (2x)")
    elif up_sample == 2:
        up_sample_func = spatial_upsample
        print("Using Spatial upsampling (2x)")
    else:
        up_sample_func = None
        print("No upsampling")
    
    return kernel, up_sample_func


def run_inversion(img, kernel, up_sample_func=None, sparse_iter=100, fidelity=150, 
                  sparsity=10, tcontinuity=0.5, deconv_iter=7, deconv_type=1):
    """
    Run the sparse deconvolution inversion algorithm.
    
    Parameters
    ----------
    img : ndarray
        Preprocessed input image.
    kernel : ndarray
        PSF kernel for deconvolution.
    up_sample_func : callable, optional
        Upsampling function to apply.
    sparse_iter : int, optional
        Number of sparse Hessian iterations {default: 100}
    fidelity : int, optional
        Fidelity parameter {default: 150}
    sparsity : int, optional
        Sparsity parameter {default: 10}
    tcontinuity : float, optional
        Continuity along z-axis {default: 0.5}
    deconv_iter : int, optional
        Number of deconvolution iterations {default: 7}
    deconv_type : int, optional
        Deconvolution type: 0=None, 1=Richardson-Lucy, 2=LandWeber {default: 1}
    
    Returns
    -------
    img_result : ndarray
        Reconstructed image.
    """
    # Apply upsampling if specified
    if up_sample_func is not None:
        print("Applying upsampling...")
        img = up_sample_func(img)
        img = img / (img.max())
    
    # Sparse Hessian reconstruction
    print(f"Running sparse Hessian reconstruction ({sparse_iter} iterations)...")
    start = time.process_time()
    gc.collect()
    
    img_sparse = sparse_hessian(img, sparse_iter, fidelity, sparsity, tcontinuity)
    
    end = time.process_time()
    print(f'Sparse-Hessian time: {end - start:.2f}s')
    
    img_sparse = img_sparse / (img_sparse.max())
    
    # Iterative deconvolution
    if deconv_type == 0 or kernel is None:
        print("Skipping iterative deconvolution")
        img_result = img_sparse
    else:
        print(f"Running iterative deconvolution (type={deconv_type}, {deconv_iter} iterations)...")
        start = time.process_time()
        img_result = iterative_deconv(img_sparse, kernel, deconv_iter, rule=deconv_type)
        end = time.process_time()
        print(f'Deconvolution time: {end - start:.2f}s')
    
    return img_result


def evaluate_results(img_recon, expected_path=None, output_path=None, scaler=1.0, original_dtype=None):
    """
    Evaluate and save the reconstruction results.
    
    Parameters
    ----------
    img_recon : ndarray
        Reconstructed image.
    expected_path : str, optional
        Path to expected output for comparison.
    output_path : str, optional
        Path to save the reconstructed image.
    scaler : float, optional
        Scaling factor to restore original intensity range.
    original_dtype : dtype, optional
        Original data type for saving.
    
    Returns
    -------
    metrics : dict
        Dictionary containing evaluation metrics (PSNR, SSIM, MSE).
    """
    # Rescale to original intensity range
    img_recon_scaled = scaler * img_recon
    
    # Save result if output path specified
    if output_path is not None:
        if original_dtype is not None:
            save_img = img_recon_scaled.astype(original_dtype)
        else:
            save_img = img_recon_scaled
        imsave(output_path, save_img)
        print(f"✅ Processing complete! Result saved to: {output_path}")
    
    metrics = {}
    
    # Compare with expected output if provided
    if expected_path is not None:
        try:
            expected = imread(expected_path)
            
            # Ensure shapes match
            if img_recon_scaled.shape != expected.shape:
                print(f"Warning: Shape mismatch - Reconstructed: {img_recon_scaled.shape}, Expected: {expected.shape}")
                return metrics
            
            # Calculate PSNR
            data_range = expected.max() - expected.min()
            psnr_val = peak_signal_noise_ratio(expected, img_recon_scaled, data_range=data_range)
            metrics['PSNR'] = psnr_val
            
            # Calculate SSIM
            channel_axis = None if len(expected.shape) == 2 else 2
            ssim_val = ssim(expected, img_recon_scaled, data_range=data_range, channel_axis=channel_axis)
            metrics['SSIM'] = ssim_val
            
            # Calculate MSE
            mse_val = np.mean((expected.astype(np.float64) - img_recon_scaled.astype(np.float64)) ** 2)
            metrics['MSE'] = mse_val
            
            # Calculate RMSE
            rmse_val = np.sqrt(mse_val)
            metrics['RMSE'] = rmse_val
            
            print(f"📊 PSNR: {psnr_val:.4f} dB")
            print(f"📊 SSIM: {ssim_val:.4f}")
            print(f"📊 MSE: {mse_val:.6f}")
            print(f"📊 RMSE: {rmse_val:.6f}")
            
        except FileNotFoundError:
            print(f"Warning: Expected output file not found: {expected_path}")
    
    return metrics


def sparse_deconv(img, sigma, sparse_iter=100, fidelity=150, sparsity=10, tcontinuity=0.5,
                  background=1, deconv_iter=7, deconv_type=1, up_sample=0):
    """
    Sparse deconvolution - main wrapper function.
    
    Parameters
    ----------
    img : ndarray
        Input image (can be T × X × Y).
    sigma : float or list
        The point spread function size in pixels.
    sparse_iter : int, optional
        Number of sparse Hessian iterations {default: 100}
    fidelity : int, optional
        Fidelity parameter {default: 150}
    sparsity : int, optional
        Sparsity parameter {default: 10}
    tcontinuity : float, optional
        Continuity along z-axis {default: 0.5}
    background : int, optional
        Background estimation flag {default: 1}
    deconv_iter : int, optional
        Number of deconvolution iterations {default: 7}
    deconv_type : int, optional
        Deconvolution type: 0=None, 1=Richardson-Lucy, 2=LandWeber {default: 1}
    up_sample : int, optional
        Upsampling type: 0=None, 1=Fourier, 2=Spatial {default: 0}
    
    Returns
    -------
    img_last : ndarray
        The sparse deconvolved image.
    """
    if not sigma:
        print("The PSF's sigma is not given, turning off the iterative deconv...")
        deconv_type = 0
    
    img = np.array(img, dtype='float32')
    scaler = np.max(img)
    img = img / scaler

    backgrounds = background_estimation(img / 2.5)
    img = img - backgrounds

    img = img / (img.max())
    img[img < 0] = 0

    if up_sample == 1:
        img = fourier_upsample(img)
    elif up_sample == 2:
        img = spatial_upsample(img)

    img = img / (img.max())

    start = time.process_time()
    gc.collect()

    img_sparse = sparse_hessian(img, sparse_iter, fidelity, sparsity, tcontinuity)
    end = time.process_time()
    print('sparse-hessian time %0.2fs' % (end - start))
    
    img_sparse = img_sparse / (img_sparse.max())
    
    if deconv_type == 0:
        img_last = img_sparse
        return scaler * img_last
    else:
        start = time.process_time()
        kernel = Gauss(sigma)
        img_last = iterative_deconv(img_sparse, kernel, deconv_iter, rule=deconv_type)
        end = time.process_time()
        print('deconv time %0.2fs' % (end - start))
        return scaler * img_last


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Configuration parameters
    PSF_FACTOR = 280 / 65
    SPARSE_ITER = 1000
    
    # Input and output file paths
    input_path = "001.tif"
    output_path = "output.tif"
    expected_output_path = "expected_output.tif"
    
    # Step 1: Load and preprocess data
    print("=" * 60)
    print("Step 1: Loading and preprocessing data")
    print("=" * 60)
    img, scaler, original_dtype = load_and_preprocess_data(input_path)
    
    # Step 2: Construct forward operator
    print("\n" + "=" * 60)
    print("Step 2: Constructing forward operator")
    print("=" * 60)
    kernel, up_sample_func = forward_operator(PSF_FACTOR, up_sample=0)
    
    # Step 3: Run inversion
    print("\n" + "=" * 60)
    print("Step 3: Running inversion")
    print("=" * 60)
    img_recon = run_inversion(
        img, 
        kernel, 
        up_sample_func=up_sample_func,
        sparse_iter=SPARSE_ITER,
        fidelity=150,
        sparsity=10,
        tcontinuity=0.5,
        deconv_iter=7,
        deconv_type=1
    )
    
    # Step 4: Evaluate results
    print("\n" + "=" * 60)
    print("Step 4: Evaluating results")
    print("=" * 60)
    metrics = evaluate_results(
        img_recon,
        expected_path=expected_output_path,
        output_path=output_path,
        scaler=scaler,
        original_dtype=original_dtype
    )
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
