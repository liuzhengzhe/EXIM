import numpy
import torch.nn as nn
import torch.nn.functional as F
import pytorch_wavelets.dwt.lowlevel as lowlevel
from pytorch_wavelets.dwt.transform2d import DWTForward
import torch
from torch.autograd import Function
import time

### modify from https://github.com/fbcotter/pytorch_wavelets/blob/master/pytorch_wavelets/dwt/lowlevel.py
from models.module.dwt_utils import prep_filt_sfb3d, prep_filt_afb3d


class DWTInverse3d_Laplacian(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
            g0_dep, g1_dep = g0_col, g1_col


        # Prepare the filters
        filts = prep_filt_sfb3d(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_dep', filts[0])
        self.register_buffer('g1_dep', filts[1])
        self.register_buffer('g0_col', filts[2])
        self.register_buffer('g1_col', filts[3])
        self.register_buffer('g0_row', filts[4])
        self.register_buffer('g1_row', filts[5])
        self.J = J
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward
        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:

            # 'Unpad' added dimensions
            ll_diff = SFB3D_Laplacian.apply(
                ll, self.g0_dep, self.g0_col, self.g0_row, mode)
            if ll_diff.shape[-3] > h.shape[-3]:
                ll_diff = ll_diff[...,:-1, :, :]
            if ll_diff.shape[-2] > h.shape[-2]:
                ll_diff = ll_diff[...,:-1,:]
            if ll_diff.shape[-1] > h.shape[-1]:
                ll_diff = ll_diff[...,:-1]

            ll = ll_diff + h

        return ll

class DWTInverse3d(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
            g0_dep, g1_dep = g0_col, g1_col


        # Prepare the filters
        filts = prep_filt_sfb3d(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_dep', filts[0])
        self.register_buffer('g1_dep', filts[1])
        self.register_buffer('g0_col', filts[2])
        self.register_buffer('g1_col', filts[3])
        self.register_buffer('g0_row', filts[4])
        self.register_buffer('g1_row', filts[5])
        self.J = J
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward
        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 7, ll.shape[-3], ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-3] > h.shape[-3]:
                ll = ll[...,:-1, :, :]
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]
            ll = SFB3D.apply(
                ll, h, self.g0_dep, self.g1_dep, self.g0_col, self.g1_col, self.g0_row, self.g1_row, mode)
        return ll

class DWTForward3d_Laplacian(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
            h0_dep, h1_dep = h0_col, h1_col


        # Prepare the filters
        filts = prep_filt_afb3d(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_dep', filts[0])
        self.register_buffer('h1_dep', filts[1])
        self.register_buffer('h0_col', filts[2])
        self.register_buffer('h1_col', filts[3])
        self.register_buffer('h0_row', filts[4])
        self.register_buffer('h1_row', filts[5])
        self.J = J
        self.mode = mode

        ## Need for inverse
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
            g0_dep, g1_dep = g0_col, g1_col


        # Prepare the filters
        filts = prep_filt_sfb3d(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_dep', filts[0])
        self.register_buffer('g1_dep', filts[1])
        self.register_buffer('g0_col', filts[2])
        self.register_buffer('g1_col', filts[3])
        self.register_buffer('g0_row', filts[4])
        self.register_buffer('g1_row', filts[5])

    def forward(self, x):
        """ Forward pass of the DWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll_new = AFB3D_Laplacian.apply(
                ll, self.h0_dep, self.h0_col, self.h0_row, mode)
            reversed_ll = SFB3D_Laplacian.apply(ll_new, self.g0_dep, self.g0_col, self.g0_row, mode)
            if ll.shape[-1] < reversed_ll.shape[-1]:
                reversed_ll = reversed_ll[..., :-1]
            if ll.shape[-2] < reversed_ll.shape[-2]:
                reversed_ll = reversed_ll[..., :-1, :]
            if ll.shape[-3] < reversed_ll.shape[-3]:
                reversed_ll = reversed_ll[..., :-1, :, :]
            yh.append(ll - reversed_ll)
            ll = ll_new

        return ll, yh

class DWTForward3d(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image
    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
            h0_dep, h1_dep = h0_col, h1_col


        # Prepare the filters
        filts = prep_filt_afb3d(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_dep', filts[0])
        self.register_buffer('h1_dep', filts[1])
        self.register_buffer('h0_col', filts[2])
        self.register_buffer('h1_col', filts[3])
        self.register_buffer('h0_row', filts[4])
        self.register_buffer('h1_row', filts[5])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.
        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.
        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = AFB3D.apply(
                ll, self.h0_dep, self.h1_dep, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            yh.append(high)

        return ll, yh


def afb1d_laplacian(x, h0, mode='zero', dim = -1):
    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 5
    s = [1, 1, 1]
    s[d - 2] = 2
    s = tuple(s)
    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1, 1, 1, 1, 1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)

    h = torch.cat([h0] * C, dim=0)

    assert mode in ['zero', 'constant']

    # Calculate the pad size
    outsize = pywt.dwt_coeff_len(N, L, mode=mode)
    p = 2 * (outsize - 1) - N + L

    # Sadly, pytorch only allows for same padding before and after, if
    # we need to do more padding after for odd length signals, have to
    # prepad

    padding_mode = None
    if mode == 'zero':
        padding_mode = 'zero'
    elif mode == 'constant':
        padding_mode = 'replicate'
    else:
        raise Exception("Unknown mode")

    if p % 2 == 1:
        pad = [0, 0, 0, 0, 0, 0]
        pad[(4 - d) * 2 + 1] = 1
        pad = tuple(pad)
        function_padding = 'constant' if padding_mode == 'zero' else padding_mode
        x = F.pad(x, pad, mode = function_padding)
    pad = [0, 0, 0]
    pad[d - 2] = p // 2
    pad = tuple(pad)
    # Calculate the high and lowpass
    if padding_mode == 'zero':
        lo = F.conv3d(x, h, padding=pad, stride=s, groups=C)
    else:
        pad_new = [ pad[2 - i // 2] for i in range(6)]
        x = F.pad(x, pad_new, mode = padding_mode)
        lo = F.conv3d(x, h, stride=s, groups=C)

    return lo

def afb1d(x, h0, h1, mode='zero', dim=-1):
    """ 1D analysis filter bank (along one dimension only) of an image
    Inputs:
        x (tensor): 5D input with the last two dimensions the spatial input
        h0 (tensor): 5D input for the lowpass filter. Should have shape (1, 1,
            h, 1, 1) or (1, 1, 1, w, 1) or (1, 1, 1, 1, d)
        h1 (tensor): 4D input for the highpass filter. Should have shape (1, 1,
            h, 1) or (1, 1, 1, w, 1) or (1, 1, 1, 1, d)
        mode (str): padding method can only be zero
        dim (int) - dimension of filtering. d=2 is for a vertical filter (called
            column filtering but filters across the rows). d=3 is for a
            horizontal filter, (called row filtering but filters across the
            columns).
    Returns:
        lohi: lowpass and highpass subbands concatenated along the channel
            dimension
    """

    C = x.shape[1]
    # Convert the dim to positive
    d = dim % 5
    s = [1, 1, 1]
    s[d-2] = 2
    s = tuple(s)
    N = x.shape[d]
    # If h0, h1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(h0, torch.Tensor):
        h0 = torch.tensor(np.copy(np.array(h0).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    if not isinstance(h1, torch.Tensor):
        h1 = torch.tensor(np.copy(np.array(h1).ravel()[::-1]),
                          dtype=torch.float, device=x.device)
    L = h0.numel()
    L2 = L // 2
    shape = [1,1,1,1,1]
    shape[d] = L
    # If h aren't in the right shape, make them so
    if h0.shape != tuple(shape):
        h0 = h0.reshape(*shape)
    if h1.shape != tuple(shape):
        h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)

    assert mode in ['zero', 'constant']

    # Calculate the pad size
    outsize = pywt.dwt_coeff_len(N, L, mode=mode)
    p = 2 * (outsize - 1) - N + L

    # Sadly, pytorch only allows for same padding before and after, if
    # we need to do more padding after for odd length signals, have to
    # prepad
    padding_mode = None
    if mode == 'zero':
        padding_mode = 'zero'
    elif mode == 'constant':
        padding_mode = 'replicate'
    else:
        raise Exception("Unknown mode")

    if p % 2 == 1:
        pad = [0, 0, 0, 0, 0, 0]
        pad[(4 - d)*2+1] = 1
        pad = tuple(pad)
        function_padding = 'constant' if padding_mode == 'zero' else padding_mode
        x = F.pad(x, pad, mode = function_padding)
    pad = [0, 0, 0]
    pad[d - 2] = p // 2
    pad = tuple(pad)
    # Calculate the high and lowpass
    if padding_mode == 'zero':
        lohi = F.conv3d(x, h, padding=pad, stride=s, groups=C)
    else:
        pad_new = [ pad[2 - i // 2] for i in range(6)]
        x = F.pad(x, pad_new, mode = padding_mode)
        lohi = F.conv3d(x, h, stride=s, groups=C)

    return lohi

def sfb1d_laplacian(lo, g0, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 5
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1,1]
    shape[d] = L
    N = 2*lo.shape[d]

    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)

    s = [1, 1, 1]
    s[d-2] = 2

    g0 = torch.cat([g0]*C,dim=0)

    assert mode in ['zero', 'constant']

    pad = [0, 0, 0]
    pad[d-2] = L - 2
    pad = tuple(pad)

    y = F.conv_transpose3d(lo, g0, stride=s, padding=pad, groups=C)

    return y

def sfb1d(lo, hi, g0, g1, mode='zero', dim=-1):
    """ 1D synthesis filter bank of an image tensor
    """
    C = lo.shape[1]
    d = dim % 5
    # If g0, g1 are not tensors, make them. If they are, then assume that they
    # are in the right order
    if not isinstance(g0, torch.Tensor):
        g0 = torch.tensor(np.copy(np.array(g0).ravel()),
                          dtype=torch.float, device=lo.device)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.tensor(np.copy(np.array(g1).ravel()),
                          dtype=torch.float, device=lo.device)
    L = g0.numel()
    shape = [1,1,1,1,1]
    shape[d] = L
    N = 2*lo.shape[d]

    # If g aren't in the right shape, make them so
    if g0.shape != tuple(shape):
        g0 = g0.reshape(*shape)
    if g1.shape != tuple(shape):
        g1 = g1.reshape(*shape)

    s = [1, 1, 1]
    s[d-2] = 2

    g0 = torch.cat([g0]*C,dim=0)
    g1 = torch.cat([g1]*C,dim=0)

    assert mode in ['zero', 'constant']

    pad = [0, 0, 0]
    pad[d-2] = L - 2
    pad = tuple(pad)

    y = F.conv_transpose3d(lo, g0, stride=s, padding=pad, groups=C) + \
        F.conv_transpose3d(hi, g1, stride=s, padding=pad, groups=C)

    return y

class SFB3D_Laplacian(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`
    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.
    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here
    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.
    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """
    @staticmethod
    def forward(ctx, low, g0_dep, g0_col, g0_row, mode):
        mode = lowlevel.int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(g0_dep, g0_col, g0_row)
        lll = low
        ## first level
        ll = sfb1d_laplacian(lll, g0_dep, mode=mode, dim=2)

        ## second level
        l = sfb1d_laplacian(ll, g0_col, mode=mode, dim=3)

        ## last level
        y = sfb1d_laplacian(l, g0_row, mode=mode, dim=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        dlow = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            g0_dep, g0_col, g0_row = ctx.saved_tensors
            dx = afb1d_laplacian(dy, g0_row, mode=mode, dim=4)
            dx = afb1d_laplacian(dx, g0_col, mode=mode, dim=3)
            dx = afb1d_laplacian(dx, g0_dep, mode = mode, dim = 2)
            s = dx.shape
            dlow = dx.reshape(s[0], -1, s[-3], s[-2], s[-1])
        return dlow, None, None, None, None, None

class SFB3D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`
    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.
    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        mode (int): use mode_to_int to get the int code here
    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.
    Returns:
        y: Tensor of shape (N, C*4, H, W)
    """
    @staticmethod
    def forward(ctx, low, highs, g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row, mode):
        mode = lowlevel.int_to_mode(mode)
        ctx.mode = mode
        ctx.save_for_backward(g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row)
        hll, lhl, hhl, llh, hlh, lhh, hhh = torch.unbind(highs, dim=2)
        lll = low
        ## first level
        ll = sfb1d(lll, hll, g0_dep, g1_dep, mode=mode, dim=2)
        hl = sfb1d(lhl, hhl, g0_dep, g1_dep, mode=mode, dim=2)
        lh = sfb1d(llh, hlh, g0_dep, g1_dep, mode=mode, dim=2)
        hh = sfb1d(lhh, hhh, g0_dep, g1_dep, mode=mode, dim=2)

        ## second level
        l = sfb1d(ll, hl, g0_col, g1_col, mode=mode, dim=3)
        h = sfb1d(lh, hh, g0_col, g1_col, mode=mode, dim=3)

        ## last level
        y = sfb1d(l, h, g0_row, g1_row, mode=mode, dim=4)
        return y

    @staticmethod
    def backward(ctx, dy):
        dlow, dhigh = None, None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            g0_dep, g1_dep, g0_col, g1_col, g0_row, g1_row = ctx.saved_tensors
            dx = afb1d(dy, g0_row, g1_row, mode=mode, dim=4)
            dx = afb1d(dx, g0_col, g1_col, mode=mode, dim=3)
            dx = afb1d(dx, g0_dep, g1_dep, mode = mode, dim = 2)
            s = dx.shape
            dx = dx.reshape(s[0], -1, 8, s[-3], s[-2], s[-1])
            dlow = dx[:,:,0].contiguous()
            dhigh = dx[:,:,1:].contiguous()
        return dlow, dhigh, None, None, None, None, None, None, None

class AFB3D_Laplacian(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`
    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.
    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        h0_dep: depth lowpass
        h1_dep: depth highpass
        mode (int): use mode_to_int to get the int code here
    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.
    Returns:
        y: Tensor of shape (N, C*4, H, W, D)
    """

    @staticmethod
    def forward(ctx, x, h0_dep, h0_col, h0_row, mode):
        ctx.save_for_backward(h0_dep, h0_col, h0_row)
        ctx.shape = x.shape[-3:]
        mode = lowlevel.int_to_mode(mode)
        ctx.mode = mode
        lohi_dim_last = afb1d_laplacian(x, h0_row, mode=mode, dim=4)
        lohi_dim_last_2 = afb1d_laplacian(lohi_dim_last, h0_col, mode=mode, dim=3)
        y = afb1d_laplacian(lohi_dim_last_2, h0_dep, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 1, s[-3], s[-2], s[-1])
        low = y[:, :, 0].contiguous()
        return low

    @staticmethod
    def backward(ctx, lll):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_dep, h0_row, h0_col = ctx.saved_tensors

            ## first level
            ll = sfb1d_laplacian(lll, h0_dep, mode=mode, dim=2)

            ## second level
            l = sfb1d_laplacian(ll, h0_col, mode=mode, dim=3)

            ## last level
            dx = sfb1d_laplacian(l, h0_row, mode = mode, dim = 4)

            if dx.shape[-3] > ctx.shape[-3]:
                dx = dx[:, :, :ctx.shape[-3]]
            if dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:, :, :, :ctx.shape[-2]]
            if dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:, :, :, :, :ctx.shape[-1]]

        return dx, None, None, None, None

class AFB3D(Function):
    """ Does a single level 2d wavelet decomposition of an input. Does separate
    row and column filtering by two calls to
    :py:func:`pytorch_wavelets.dwt.lowlevel.afb1d`
    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.
    Inputs:
        x (torch.Tensor): Input to decompose
        h0_row: row lowpass
        h1_row: row highpass
        h0_col: col lowpass
        h1_col: col highpass
        h0_dep: depth lowpass
        h1_dep: depth highpass
        mode (int): use mode_to_int to get the int code here
    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.
    Returns:
        y: Tensor of shape (N, C*4, H, W, D)
    """

    @staticmethod
    def forward(ctx, x, h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row, mode):
        ctx.save_for_backward(h0_dep, h1_dep, h0_col, h1_col, h0_row, h1_row)
        ctx.shape = x.shape[-3:]
        mode = lowlevel.int_to_mode(mode)
        ctx.mode = mode
        lohi_dim_last = afb1d(x, h0_row, h1_row, mode=mode, dim=4)
        lohi_dim_last_2 = afb1d(lohi_dim_last, h0_col, h1_col, mode=mode, dim=3)
        y = afb1d(lohi_dim_last_2, h0_dep, h1_dep, mode=mode, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 8, s[-3], s[-2], s[-1])
        low = y[:, :, 0].contiguous()
        highs = y[:, :, 1:].contiguous()
        return low, highs

    @staticmethod
    def backward(ctx, lll, highs):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0_dep, h1_dep, h0_row, h1_row, h0_col, h1_col = ctx.saved_tensors
            hll, lhl, hhl, llh, hlh, lhh, hhh = torch.unbind(highs, dim=2)

            ## first level
            ll = sfb1d(lll, hll, h0_dep, h1_dep, mode=mode, dim=2)
            hl= sfb1d(lhl, hhl, h0_dep, h1_dep, mode=mode, dim=2)
            lh = sfb1d(llh, hlh, h0_dep, h1_dep, mode=mode, dim=2)
            hh= sfb1d(lhh, hhh, h0_dep, h1_dep, mode=mode, dim=2)

            ## second level
            l = sfb1d(ll, hl, h0_col, h1_col, mode=mode, dim=3)
            h = sfb1d(lh, hh, h0_col, h1_col, mode=mode, dim=3)

            ## last level
            dx = sfb1d(l, h, h0_row, h1_row, mode = mode, dim = 4)

            if dx.shape[-3] > ctx.shape[-3]:
                dx = dx[:, :, :ctx.shape[-3]]
            if dx.shape[-2] > ctx.shape[-2]:
                dx = dx[:, :, :, :ctx.shape[-2]]
            if dx.shape[-1] > ctx.shape[-1]:
                dx = dx[:, :, :, :, :ctx.shape[-1]]

        return dx, None, None, None, None, None


###### TESTING CODE

from utils.debugger import MyDebugger
import os
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels
import mcubes
import pywt
from configs import config

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_weight_from_array(indices, level, low, highs, **kwargs):
    inidces_long = indices.long()
    if level == len(highs):
        return low[inidces_long[:, 0], 0, inidces_long[:, 1], inidces_long[:, 2], inidces_long[:, 3]].unsqueeze(1)
    else:
        return highs[level][inidces_long[:, 0], 0, inidces_long[:, 1], inidces_long[:, 2], inidces_long[:, 3]].unsqueeze(1)

if __name__ == '__main__':
    import time
    from models.network import create_coordinates
    debugger = MyDebugger('My-Wavelet-Testing', is_save_print_to_file = False)
    folder = r'Y:\proj51_backup\edward\ShapeNetCore.v1\03001627'
    cnt = 100
    # paths = [path for path in os.listdir(folder) if os.path.isdir(os.path.join(folder, path))]
    # paths = np.random.choice(paths, cnt, replace = False)
    paths = [
        #'Y:\proj51_backup\edward\ShapeNetCore.v1\03001627\2a98a638f675f46e7d44dc16af152638',
        # r'Y:\proj51_backup\edward\ShapeNetCore.v1\02828884\f98acd1dbe15f3c02056b4bd5d870b47',
        'E:\3D_models\dragon_recon\dragon_vrip.obj'
    ]
    resolution = 256
    recompute = False
    optimize = True
    test_multi = False
    use_sparse = True
    use_slice = False
    use_dense_conv = True
    use_mlp = True
    split_batch = True
    progressive = True
    num_points = 64 ** 3
    batch_num_points = 64 ** 3

    ## TESTING
    padding_mode = 'zero'
    wavelet_type = 'bior3.3'
    wavelet = pywt.Wavelet(wavelet_type)
    max_depth = pywt.dwt_max_level(data_len = resolution, filter_len=wavelet.dec_len)
    #max_depth = 4
    #max_depth = 1
    skip_level = []

    ### pytorch wavelet
    J = max_depth
    dwt_forward = DWTForward(J = max_depth, wave=wavelet, mode = padding_mode).to(device)
    dwt_forward_3d = DWTForward3d(J = max_depth, wave=wavelet, mode=padding_mode).to(device)
    dwt_inverse_3d = DWTInverse3d(J = max_depth, wave=wavelet, mode=padding_mode).to(device)
    dwt_forward_3d_lap = DWTForward3d_Laplacian(J = max_depth, wave=wavelet, mode=padding_mode).to(device)
    dwt_inverse_3d_lap = DWTInverse3d_Laplacian(J = max_depth, wave=wavelet, mode=padding_mode).to(device)
    from models.network import SparseComposer, MultiScaleMLP
    composer_parms = dwt_inverse_3d_lap if use_dense_conv else None
    dwt_sparse_composer = SparseComposer(input_shape = [resolution, resolution, resolution], J = max_depth, wave=wavelet, mode = padding_mode, inverse_dwt_module=composer_parms).to(device)
    weight_mlp_functions = MultiScaleMLP(config = config, data_num = 1, J = J).to(device)
    indices = create_coordinates(resolution).view((-1, 3))
    all_indices = indices
    indices = indices[torch.randperm(indices.size(0))]
    indices = indices.view((-1, num_points, 3)).int()
    all_indices_np = np.reshape(all_indices.cpu().numpy(), (resolution, resolution, resolution, 3))
    all_indices = all_indices.view((-1, num_points, 3)).int().to(device)

    ## sample coordinates
    sample_resolution = 64
    sample_voxel_size = resolution // sample_resolution
    sample_coordinates = create_coordinates(sample_resolution).int().cpu().numpy()
    sample_coordinates = np.reshape(sample_coordinates, (-1, 3))
    sample_near_surface = False
    surface_ratio = 0.5

    for path in paths:
        print(f'start {path}!')
        save_path = os.path.join(config.backup_path, os.path.basename(path) + f'_{resolution}.npy')
        if os.path.exists(save_path) and not recompute:
            voxels = np.load(save_path)
        else:
            start_time = time.time()
            if os.path.isdir(path):
                mesh = trimesh.load_mesh(os.path.join(path, 'model_flipped_manifold.obj'))
            else:
                mesh = trimesh.load_mesh(path)
            voxels = mesh_to_voxels(mesh, resolution)
            np.save(save_path, voxels)
            print(f"Compelete time : {time.time() - start_time} s!")
        print(f"saved path : {save_path}")

        ### compute samples
        sampled_voxels_values = numpy.concatenate([ voxels[sample_coordinates[i,0]*sample_voxel_size:sample_coordinates[i,0]*sample_voxel_size+sample_voxel_size,
                                                    sample_coordinates[i, 1]*sample_voxel_size:sample_coordinates[i, 1]*sample_voxel_size+ sample_voxel_size,
                                                    sample_coordinates[i, 2]*sample_voxel_size:sample_coordinates[i, 2]*sample_voxel_size+ sample_voxel_size
                                                    ][None, :, :, :] for i in range(sample_coordinates.shape[0])], axis = 0)
        sampled_voxels_indices = numpy.concatenate([ all_indices_np[sample_coordinates[i,0]*sample_voxel_size:sample_coordinates[i,0]*sample_voxel_size+sample_voxel_size,
                                                    sample_coordinates[i, 1]*sample_voxel_size:sample_coordinates[i, 1]*sample_voxel_size+ sample_voxel_size,
                                                    sample_coordinates[i, 2]*sample_voxel_size:sample_coordinates[i, 2]*sample_voxel_size+ sample_voxel_size
                                                    ][None, :, :, :, :] for i in range(sample_coordinates.shape[0])], axis = 0)

        samples_voxels_sign_changed = np.sign(np.max(sampled_voxels_values, axis = (1,2,3))) != np.sign(np.min(sampled_voxels_values, axis = (1, 2, 3)))
        sign_changed_voxel_indices = torch.from_numpy(sampled_voxels_indices[samples_voxels_sign_changed]).to(device).view((-1, 3)).int()
        sign_unchanged_voxel_indices = torch.from_numpy(sampled_voxels_indices[np.logical_not(samples_voxels_sign_changed)]).to(device).view((-1, 3)).int()


        voxels_cuda = torch.from_numpy(voxels).to(device).unsqueeze(0).unsqueeze(0)
        voxels_cuda_sliced = voxels_cuda[:, :, :1, :, :]
        #print(voxels_cuda_sliced.size())
        voxels_cuda_input = voxels_cuda_sliced[:, :, 0, :, :]

        low, highs = dwt_forward(voxels_cuda_input)
        voxels_np = voxels_cuda_input[0, 0].cpu().numpy()
        coeffs = pywt.wavedecn(voxels_np, wavelet, mode=padding_mode, level = J)

        #print(low, coeffs[0])
        #print(highs[0].size(), coeffs[1][0])

        ## another coefficient
        coeffs = pywt.wavedecn(voxels, wavelet, mode=padding_mode, level=1)
        low_cuda, highs_cuda = dwt_forward_3d(voxels_cuda)
        voxels_reconstruction = dwt_inverse_3d((low_cuda, highs_cuda))


        #print(low_cuda, coeffs[0])
        #print(highs_cuda[0], coeffs[1][0])

        low_cuda_lap, highs_cuda_lap = dwt_forward_3d_lap(voxels_cuda)

        start_time = time.time()
        batch_indices = indices[0].unsqueeze(0)
        sliced_reconstruction_values = dwt_sparse_composer(batch_indices, get_weight_from_array, low=low_cuda_lap,
                                                           highs=highs_cuda_lap)
        print(f"time for compose {time.time() - start_time}")

        ### testing reconstruction
        voxels_reconstruction_lap = dwt_inverse_3d_lap((low_cuda_lap, highs_cuda_lap))
        if test_multi:
            vertices, traingles = mcubes.marching_cubes(voxels_reconstruction_lap.detach().cpu().numpy()[0, 0], 0.0)
            mcubes.export_off(vertices, traingles, debugger.file_path(os.path.basename(path) + '_original.off'))
            for i in range(max_depth):
                highs_cuda_lap[i] = torch.zeros_like(highs_cuda_lap[i]).to(device)
                voxels_reconstruction_lap = dwt_inverse_3d_lap((low_cuda_lap, highs_cuda_lap))
                vertices, traingles = mcubes.marching_cubes(voxels_reconstruction_lap.detach().cpu().numpy()[0, 0], 0.0)
                mcubes.export_off(vertices, traingles, debugger.file_path(os.path.basename(path) + f'_{i}.off'))


        if optimize:
            ###
            low_pred = torch.zeros_like(low_cuda_lap, requires_grad= True)
            highs_pred = [torch.zeros_like(high_cuda_lap, requires_grad = True) for high_cuda_lap in highs_cuda_lap]

            ###
            lr = 1e-1
            if use_mlp:
                # if config.lr_weight_decay_after_stage:
                parms_list = []
                for layer in weight_mlp_functions.highs_layers:
                    parms_list.append({'params' : layer.parameters(), 'lr' : lr})
                parms_list.append({'params' : weight_mlp_functions.low_layer.parameters(), 'lr' : lr})
                parms_list.append({'params' : weight_mlp_functions.latent_codes.parameters(), 'lr' : lr})
                optimizer = torch.optim.Adam(parms_list, lr = lr)
                # else:
                #     optimizer = torch.optim.Adam(weight_mlp_functions.parameters(), lr)
            else:
                optimizer = torch.optim.Adam([low_pred] + highs_pred, lr = lr)
            iteration = 5000
            save_iteration = 500
            loss_f = torch.nn.MSELoss()

            stages = [0] if not progressive else list(range(max_depth+1))[::-1]
            for stage in stages:
                for i in range(iteration+1):
                    optimizer.zero_grad()
                    weight_mlp_functions.train()
                    if use_sparse:
                        if sample_near_surface:
                            sample_cnt = int(num_points * surface_ratio)
                            perm = torch.randperm(sign_changed_voxel_indices.size(0))
                            idx = perm[:sample_cnt]
                            sign_changed_samples = sign_changed_voxel_indices[idx]
                            perm = torch.randperm(sign_unchanged_voxel_indices.size(0))
                            idx = perm[:num_points - sample_cnt]
                            sign_unchanged_samples = sign_unchanged_voxel_indices[idx]
                            batch_indices = torch.cat((sign_changed_samples, sign_unchanged_samples), dim = 0).unsqueeze(0)
                        else:
                            batch_indices = indices[i%indices.size(0)].unsqueeze(0)
                        gt = voxels_cuda[0, 0, batch_indices.long()[:, :,0], batch_indices.long()[:, :, 1], batch_indices.long()[:, :, 2]].T
                        if use_mlp:
                            code_indices = torch.from_numpy(numpy.array([0])).to(device).long()
                            sliced_reconstruction_values = dwt_sparse_composer(batch_indices, weight_mlp_functions,
                                                                               code_indices=code_indices, stage = stage)
                        else:
                            sliced_reconstruction_values = dwt_sparse_composer(batch_indices, get_weight_from_array, low=low_pred,
                                                                               highs=highs_pred)
                        loss = loss_f(sliced_reconstruction_values, gt)
                    else:
                        voxels_reconstruction = dwt_inverse_3d_lap((low_pred, highs_pred))
                        if use_slice:
                            batch_indices = indices[i%indices.size(0)].unsqueeze(0)
                            voxels_reconstruction = voxels_reconstruction[0, 0, batch_indices.long()[:, :,0], batch_indices.long()[:, :, 1], batch_indices.long()[:, :, 2]].T
                            gt = voxels_cuda[0, 0, batch_indices.long()[:, :,0], batch_indices.long()[:, :, 1], batch_indices.long()[:, :, 2]].T
                        else:
                            gt = voxels_cuda
                        loss = loss_f(voxels_reconstruction, gt)
                    loss.backward()
                    print(f"Iteration {i} : {loss}")
                    optimizer.step()

                    if i % save_iteration == 0:
                        weight_mlp_functions.eval()
                        optimizer.zero_grad()
                        if use_mlp:
                            with torch.autograd.no_grad():
                                code_indices = torch.from_numpy(numpy.array([0])).to(device).long()
                                if split_batch:
                                    voxels_reconstruction = []
                                    batch_num = batch_num_points // num_points
                                    for j in range(all_indices.size(0) // batch_num):
                                        voxels_pred = dwt_sparse_composer(all_indices[j*batch_num:(j+1)*batch_num].view(1, -1, 3), weight_mlp_functions,
                                                                                           code_indices=code_indices, stage = stage).detach()
                                        voxels_reconstruction.append(voxels_pred)
                                    voxels_reconstruction = torch.cat(voxels_reconstruction, dim = 0)
                                else:
                                    voxels_reconstruction = dwt_sparse_composer(all_indices.view((1, -1, 3)), weight_mlp_functions,
                                                                                           code_indices=code_indices, stage = stage).detach()
                                voxels_reconstruction = voxels_reconstruction.view((1, 1, resolution, resolution, resolution))
                        else:
                            voxels_reconstruction = dwt_inverse_3d_lap((low_pred, highs_pred))
                        vertices, traingles = mcubes.marching_cubes(voxels_reconstruction.detach().cpu().numpy()[0, 0], 0.0)
                        mcubes.export_off(vertices, traingles, debugger.file_path(os.path.basename(path) + f'_recon_{stage}_{i}.off'))
                        print("done coversion")

                if config.lr_weight_decay_after_stage:
                    for stage_after in range(stage, max_depth):
                        optimizer.param_groups[stage_after]['lr'] = optimizer.param_groups[stage_after]['lr'] * config.lr_decay_rate_after_stage





