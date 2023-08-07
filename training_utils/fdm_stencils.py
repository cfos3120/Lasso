import torch
import numpy as np

def Navier_Stokes_Vorticity_torch_gradient(w, nu=1/500, t_interval=0.5, L=2*np.pi):
    assert w.shape[-1] == 1
    w = w.squeeze(-1)

    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)
    
    dt = t_interval / (nt-1)

    ## Calculate Cartesian Velocities using Spectral Methods
    w_h = torch.fft.fft2(w, dim=[1, 2])
    
    # Wavenumbers in y-direction
    k_max = nx//2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(nx, 1).repeat(1, nx).reshape(1,nx,nx,1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, nx).repeat(nx, 1).reshape(1,nx,nx,1)
    
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    # Calculate Velocity
    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])

    # create grid
    y = torch.tensor(np.linspace(0, L, nx+1)[:-1], dtype=torch.float, device=device)
    x = y

    wx = torch.zeros_like(w)
    wy = torch.zeros_like(w)
    wxx = torch.zeros_like(w)
    wyy = torch.zeros_like(w)

    # scheme
    wx, wy  = torch.gradient(w   , spacing = tuple([x, y]) , dim = [1,2]) #(batch, X-size, Y-Size, T-size, 1)
    wxx     = torch.gradient(wx  , spacing = tuple([x])    , dim = [1])[0]
    wyy     = torch.gradient(wy  , spacing = tuple([y])    , dim = [2])[0]
    
    # Laplacian
    wlap = wxx + wyy

    # Time derivative is calculated the same way and is kept constant across methods
    dw_dt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    fdm_derivatives = tuple([wx, wy, wxx, wyy, dw_dt])

    Dw = dw_dt + (ux*wx + uy*wy - nu*(wlap))[...,1:-1]

    return Dw, fdm_derivatives

def Navier_Stokes_Vorticity_spectral(w,nu=1/500, t_interval=0.5):
    assert w.shape[-1] == 1
    w = w.squeeze(-1)

    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    dt = t_interval / (nt-1)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
    
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    # Calculate Velocity
    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    wx = torch.fft.irfft2(wx_h[:, :, :k_max+1], dim=[1,2])
    wy = torch.fft.irfft2(wy_h[:, :, :k_max+1], dim=[1,2])
    wlap = torch.fft.irfft2(wlap_h[:, :, :k_max+1], dim=[1,2])

    dw_dt          = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    spectral_derivatives = tuple([ux, uy, wx, wy, wlap])
    Dw = dw_dt + (ux*wx + uy*wy - nu*wlap)[...,1:-1]

    return Dw, spectral_derivatives

def Navier_Stokes_Vorticity_periodic(w, nu=1/500, t_interval=0.5, L=2*np.pi, order=2):
    
    assert w.shape[-1] == 1
    w = w.squeeze(-1)

    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)
    
    dt = t_interval / (nt-1)

    ## Calculate Cartesian Velocities using Spectral Methods
    w_h = torch.fft.fft2(w, dim=[1, 2])
    
    # Wavenumbers in y-direction
    k_max = nx//2
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(nx, 1).repeat(1, nx).reshape(1,nx,nx,1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, nx).repeat(nx, 1).reshape(1,nx,nx,1)
    
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    # Calculate Velocity
    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])

    # create grid
    y = torch.tensor(np.linspace(0, L, nx+1)[:-1], dtype=torch.float, device=device)
    x = y

    wx = torch.zeros_like(w)
    wy = torch.zeros_like(w)
    wxx = torch.zeros_like(w)
    wyy = torch.zeros_like(w)

    # scheme selector
    if order == 2:
        # second order first derivative scheme
        # third order second derivative scheme
        dx = abs(x[1]-x[0])
        dy = dx 

        # gradients in internal zone
        wx[:, 1:-1, : , :]  = (w[:, 2: ,  : , :] - w[:, :-2 , :    , :]) / (2*dx)
        wy[:, : , 1:-1, :]  = (w[:,  : , 2: , :] - w[:, :   , :-2  , :]) / (2*dy)

        wxx[:, 1:-1, : , :] = (w[:, 2: ,  : , :] - 2*w[:, 1:-1 ,  :   , :] + w[:, :-2 , :    , :]) / (dx**2)
        wyy[:, : , 1:-1, :] = (w[:,  : , 2: , :] - 2*w[:,  :   , 1:-1 , :] + w[:, :   , :-2  , :]) / (dy**2)

        # gradients at boundary
        wx[:, 0 , :  , :]  = (w[:, 1 , : , :] - w[:, -1 , :  , :]) / (2*dx)
        wx[:, -1, :  , :]  = (w[:, 0 , : , :] - w[:, -2 , :  , :]) / (2*dx)
        wy[:, : , 0  , :]  = (w[:, : , 1 , :] - w[:, :  , -1 , :]) / (2*dy)
        wy[:, : , -1 , :]  = (w[:, : , 0 , :] - w[:, :  , -2 , :]) / (2*dy)

        wxx[:, 0 , :  , :]  = (w[:, 1 , : , :] - 2*w[:,  0  , :  , :] + w[:, -1 , :  , :]) / (dx**2)
        wxx[:, -1, :  , :]  = (w[:, 0 , : , :] - 2*w[:,  -1 , :  , :] + w[:, -2 , :  , :]) / (dx**2)
        wyy[:, : , 0  , :]  = (w[:, : , 1 , :] - 2*w[:,  :  , 0  , :] + w[:, :  , -1 , :]) / (dy**2)
        wyy[:, : , -1 , :]  = (w[:, : , 0 , :] - 2*w[:,  :  , -1 , :] + w[:, :  , -2 , :]) / (dy**2)

    elif order == 4:
        dx = abs(x[1]-x[0])
        dy = dx 

        # gradients in internal zone
        wx[:, 2:-2, : , :]  = (-w[:, 4: ,  : , :] + 8*w[:, 3:-1 ,  :   , :] - 8*w[: , 1:-3 , :   , :] + w[:, :-4 , :   , :]) / (12*dx)
        wy[:, : , 2:-2, :]  = (-w[:,  : , 4: , :] + 8*w[:,  :   , 3:-1 , :] - 8*w[: ,  :,   1:-3 , :] + w[:, : ,   :-4 , :]) / (12*dx)

        wxx[:, 2:-2, : , :] = (-w[:, 4: ,  : , :] + 16*w[:, 3:-1 ,  :   , :] - 30*w[:, 2:-2, : , :] + 16*w[: , 1:-3 , :   , :] - w[:, :-4 , :   , :]) / (12*dx**2)
        wyy[:, : , 2:-2, :] = (-w[:,  : , 4: , :] + 16*w[:,  :   , 3:-1 , :] - 30*w[:, : , 2:-2, :] + 16*w[: ,  :,   1:-3 , :] - w[:, : ,   :-4 , :]) / (12*dx**2)

        # gradients at boundary
        wx[:, 0 , :  , :] = (-w[:, 2 , : , :] + 8*w[:, 1 , : , :] - 8*w[: , -1 , : , :] + w[:, -2 , : , :]) / (12*dx)
        wx[:, 1 , :  , :] = (-w[:, 3 , : , :] + 8*w[:, 2 , : , :] - 8*w[: ,  0 , : , :] + w[:, -1 , : , :]) / (12*dx)
        wx[:, -1, :  , :] = (-w[:, 1 , : , :] + 8*w[:, 0 , : , :] - 8*w[: , -2 , : , :] + w[:, -3 , : , :]) / (12*dx)
        wx[:, -2, :  , :] = (-w[:, 0 , : , :] + 8*w[:, -1 ,: , :] - 8*w[: , -3 , : , :] + w[:, -4 , : , :]) / (12*dx)

        wy[:, : , 0 , : ] = (-w[:, : , 2 , :] + 8*w[:, : , 1 , :] - 8*w[: , : , -1 , :] + w[:, : , -2 , :]) / (12*dy)
        wy[:, : , 1 , : ] = (-w[:, : , 3 , :] + 8*w[:, : , 2 , :] - 8*w[: , : ,  0 , :] + w[:, : , -1 , :]) / (12*dy)
        wy[:, : , -1, : ] = (-w[:, : , 1 , :] + 8*w[:, : , 0 , :] - 8*w[: , : , -2 , :] + w[:, : , -3 , :]) / (12*dy)
        wy[:, : , -2, : ] = (-w[:, : , 0 , :] + 8*w[:, : , -1 ,:] - 8*w[: , : , -3 , :] + w[:, : , -4 , :]) / (12*dy)

        wxx[:, 0 , :  , :] = (-w[:, 2 , : , :] + 16*w[:, 1 , : , :] -30*w[:, 0 , :  , :] + 16*w[: , -1 , : , :] - w[:, -2 , : , :]) / (12*dx**2)
        wxx[:, 1 , :  , :] = (-w[:, 3 , : , :] + 16*w[:, 2 , : , :] -30*w[:, 1 , :  , :] + 16*w[: ,  0 , : , :] - w[:, -1 , : , :]) / (12*dx**2)
        wxx[:, -1, :  , :] = (-w[:, 1 , : , :] + 16*w[:, 0 , : , :] -30*w[:,-1 , :  , :] + 16*w[: , -2 , : , :] - w[:, -3 , : , :]) / (12*dx**2)
        wxx[:, -2, :  , :] = (-w[:, 0 , : , :] + 16*w[:, -1 ,: , :] -30*w[:,-2 , :  , :] + 16*w[: , -3 , : , :] - w[:, -4 , : , :]) / (12*dx**2)

        wyy[:, : , 0 , : ] = (-w[:, : , 2 , :] + 16*w[:, : , 1 , :] -30*w[:, : , 0  , :] + 16*w[: , : , -1 , :] - w[:, : , -2 , :]) / (12*dy**2)
        wyy[:, : , 1 , : ] = (-w[:, : , 3 , :] + 16*w[:, : , 2 , :] -30*w[:, : , 1  , :] + 16*w[: , : ,  0 , :] - w[:, : , -1 , :]) / (12*dy**2)
        wyy[:, : , -1, : ] = (-w[:, : , 1 , :] + 16*w[:, : , 0 , :] -30*w[:, : ,-1  , :] + 16*w[: , : , -2 , :] - w[:, : , -3 , :]) / (12*dy**2)
        wyy[:, : , -2, : ] = (-w[:, : , 0 , :] + 16*w[:, : , -1 ,:] -30*w[:, : ,-2  , :] + 16*w[: , : , -3 , :] - w[:, : , -4 , :]) / (12*dy**2)

    else:
        raise(ValueError)

    # Laplacian
    wlap = wxx + wyy

    # Time derivative is calculated the same way and is kept constant across methods
    dw_dt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    fdm_derivatives = tuple([wx, wy, wxx, wyy, dw_dt])

    Dw = dw_dt + (ux*wx + uy*wy - nu*(wlap))[...,1:-1]

    return Dw, fdm_derivatives

