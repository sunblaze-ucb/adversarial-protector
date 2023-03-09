import torch
# https://github.com/tomrunia/PyTorchSteerablePyramid/blob/e54981e7fcfd24263354d9c11fe70cb44457a594/steerable/math_utils.py#L25
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def get_expected_ft_single_channel_v1_2(images):
    fft = torch.rfft(images, signal_ndim=2, onesided = False)
    shift = batch_fftshift2d(fft)
    real, imaginary = torch.unbind(shift,-1)
    total = torch.sqrt(real**2 + imaginary**2)
    
    return total

def get_expected_ft_multi_channelv1_2(images):
    fft = torch.rfft(images, signal_ndim=3, onesided = False)
    shift = batch_fftshift2d(fft)
    real, imaginary = torch.unbind(shift,-1)
    total = torch.sqrt(real**2 + imaginary**2)
    return total

def get_expected_ft_single_channel(images):
    # fft = torch.fft.rfftn(images, dim=[-1,-2,-3])
    fft = torch.view_as_real(torch.fft.fftn(images, dim=[-1,-2]))
    #remove nans
    fft[fft != fft] = 0

    shift = batch_fftshift2d(fft)
    real, imaginary = torch.unbind(shift,-1)
    total = torch.sqrt(real**2 + imaginary**2)
    # total = total-total.min()
    # total = total/(total.max() + 1e-6)
    # assert(total.min() == 0)
    # assert(total.max() == 1)
    return total

def get_expected_ft_multi_channel(images):
    # fft = torch.fft.rfftn(images, dim=[-1,-2,-3])
    fft = torch.view_as_real(torch.fft.fftn(images, dim=[-1,-2,-3]))
    #remove nans
    fft[fft != fft] = 0
    
    shift = batch_fftshift2d(fft)
    real, imaginary = torch.unbind(shift,-1)

    total = torch.sqrt(real**2 + imaginary**2)
    # total = total-total.min()
    # total = total/(total.max() + 1e-6)
    return total

def get_fft(images):
    if images.shape[1] == 1: return get_expected_ft_single_channel(images)
    else: return get_expected_ft_multi_channel(images)
