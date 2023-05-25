import torch
import torch.nn.functional as F
import numpy as np

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def FDM_NS_vorticity(w, nu=1/40, t_interval=1.0):
    
    assert w.shape[-1] == 1
    w = w.squeeze(-1)

    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

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

    dt = t_interval / (nt-1)
    wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    Du1 = wt + (ux*wx + uy*wy - nu*wlap)[...,1:-1]
    return Du1

def FDM_NS_vorticity_v2(w, L =2*np.pi, nu=1/40, t_interval=1.0):
    
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

    # Calculate Differentials
    y = torch.tensor(np.linspace(0, L, nx+1)[:-1], dtype=torch.float, device=device)
    x = y
    dw_dx, dw_dy    = torch.gradient(w,  spacing = tuple([x, y]), dim = [1,2])
    dw_dxx         = torch.gradient(dw_dx,    spacing = tuple([x]), dim = [1])[0]
    dw_dyy         = torch.gradient(dw_dy,    spacing = tuple([y]), dim = [2])[0]
    dw_dt          = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)

    Du1 = dw_dt + (ux*dw_dx + uy*dw_dy - nu*(dw_dxx+dw_dyy))[...,1:-1]

    # Store the spectral form so we can calculate loss on it too.
    Du2 = dw_dt + (ux*wx + uy*wy - nu*wlap)[...,1:-1]
    return Du1, Du2

def FDM_NS_cartesian(u, nu=1/500, t_interval=1.0):

    assert u.shape[-1] == 2
    assert u.shape[0] > 1 #--> There is a bug with torch.gradient which doesnt handle small batches well on Artemis

    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)
    device = u.device

    L = 1.0

    # Assuming uniform periodic spatial grid NOTE: These need to line up with the grid function made for training.
    x = torch.tensor(np.linspace(0, L, nx+1)[:-1], dtype=torch.float, device=device)#<- keep the domain non-dimensional
    y = torch.tensor(np.linspace(0, L, nx+1)[:-1], dtype=torch.float, device=device)
    t = torch.tensor(np.linspace(0, t_interval, nt+1)[:-1], dtype=torch.float, device=device)

    # each of these (dV_dx etc.) should come with shape (Batch,x,y,t,Velocity direction)
    dV_dx, dV_dy, dV_dt = torch.gradient(u, spacing =tuple([x, y, t]), dim = [1,2,3])
    dV_dxx = torch.gradient(dV_dx, spacing = tuple([x]), dim = 1)[0]
    dV_dyy = torch.gradient(dV_dy, spacing = tuple([y]), dim = 2)[0]

    eqn_c = dV_dx[...,0] + dV_dy[...,1]
    #eqn_mx = nu * (dV_dxx[...,0] + dV_dyy[...,0]) - dV_dt[...,0] - u[...,0]*dV_dx[...,0] - u[...,1]*dV_dy[...,0]
    #eqn_my = nu * (dV_dxx[...,1] + dV_dyy[...,1]) - dV_dt[...,1] - u[...,0]*dV_dx[...,1] - u[...,1]*dV_dy[...,1]
    
    #eqn_mx = dV_dt[...,0] + u[...,0]*dV_dx[...,0] + u[...,1]*dV_dy[...,0] - nu * (dV_dxx[...,0] + dV_dyy[...,0])
    #eqn_my = dV_dt[...,1] + u[...,0]*dV_dx[...,1] + u[...,1]*dV_dy[...,1] - nu * (dV_dxx[...,1] + dV_dyy[...,1])
    
    # correction
    # Note we do not take the first and last timesteps as the gradient calculated would be less accurate (this is more inline with the voritcity version) 
    eqn_mx = (dV_dt[..., 0] + u[..., 0]*dV_dx[..., 0] + u[..., 1]*dV_dy[..., 0] - nu*(dV_dxx[..., 0] + dV_dyy[..., 0]))[...,1:-1] #+ outx[..., 2]
    eqn_my = (dV_dt[..., 1] + u[..., 0]*dV_dx[..., 1] + u[..., 1]*dV_dy[..., 1] - nu*(dV_dxx[..., 1] + dV_dyy[..., 1]))[...,1:-1] #+ outy[..., 2]
    eqn_c = (dV_dx[..., 0] + dV_dy[..., 1])[...,1:-1]

    return eqn_c, eqn_mx, eqn_my

def FDM_NS_cartesian_v2(u, L = 1.0, nu=1/500, t_interval=1.0):

    assert u.shape[-1] == 2
    #assert u.shape[0] > 1 #--> There is a bug with torch.gradient which doesnt handle small batches well on Artemis

    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)
    device = u.device

    dt = t_interval / (nt-1)

    y = torch.tensor(np.linspace(0, L, nx+1)[:-1], dtype=torch.float, device=device)
    x = y
    dUx_dx,dUx_dy   = torch.gradient(u[...,0],  spacing = tuple([x, y]), dim = [1,2])
    dUy_dx,dUy_dy   = torch.gradient(u[...,1],  spacing = tuple([x, y]), dim = [1,2])
    dUx_dxx         = torch.gradient(dUx_dx,    spacing = tuple([x]), dim = [1])[0]
    dUx_dyy         = torch.gradient(dUx_dy,    spacing = tuple([y]), dim = [2])[0]
    dUy_dxx         = torch.gradient(dUy_dx,    spacing = tuple([x]), dim = [1])[0]
    dUy_dyy         = torch.gradient(dUy_dy,    spacing = tuple([y]), dim = [2])[0]
    dUx_dt          = (u[:, :, :, 2:,0] - u[:, :, :, :-2,0]) / (2 * dt)
    dUy_dt          = (u[:, :, :, 2:,1] - u[:, :, :, :-2,1]) / (2 * dt)
    
    eqn_mx = dUx_dt + (u[..., 0]*dUx_dx + u[..., 1]*dUx_dy - 1/500*(dUx_dxx + dUx_dyy))[...,1:-1]
    eqn_my = dUy_dt + (u[..., 0]*dUy_dx + u[..., 1]*dUy_dy - 1/500*(dUy_dxx + dUy_dyy))[...,1:-1]
    eqn_c = (dUx_dx + dUy_dy)[...,1:-1]

    return eqn_c, eqn_mx, eqn_my

def FDM_NS_cartesian_hard(A):
    # This is based on the model producing a vector potential A, instead of output velocity V.
    # Thus gradients of A need to be calculated to enforce mass conservation.

    assert A.shape[-1] == 2
    assert A.shape[0] > 1 #--> There is a bug with torch.gradient which doesnt handle small batches well

    batchsize = A.size(0)
    nx = A.size(1)
    ny = A.size(2)
    nt = A.size(3)
    device = A.device

    # Assuming uniform periodic spatial grid NOTE: These need to line up with the grid function made for training.
    x = torch.arange(0,1.0,1.0/nx, device=device) #<- keep the domain non-dimensional
    y = torch.arange(0,1.0,1.0/ny, device=device)

    # each of these (dV_dx etc.) should come with shape (Batch,x,y,t,Velocity direction)
    dA_dx, dA_dy = torch.gradient(A, spacing =tuple([x, y]), dim = [1,2])

    V_tilde = dA_dx + dA_dy

    return V_tilde

def PINO_loss3d_decider(model_input, model_output, model_val, forcing_type, nu, t_interval):
    B = model_output.shape[0]
    S = model_output.shape[1]
    T = model_output.shape[3]
    C = model_output.shape[4]
    device = model_output.device

    # Set Loss function
    torch_zero = torch.zeros(1).to(device)

    # Intialize Losses
    loss_l2, loss_ic, loss_bc, loss_w, loss_c, loss_m1, loss_m2 = torch_zero, torch_zero, torch_zero, torch_zero, torch_zero, torch_zero, torch_zero
    
    # Inital condition is the same across all conditions and test types (3rd index onwards excludes the grid)
    u0 = model_input[:, :, :, 0, 3:] 

    loss_ic = F.mse_loss(model_output[:, :, :, 0, :], u0)

    if forcing_type == 'cartesian_to_vorticity_trial':
        
        # Here we will convert the output to vorticity form and run the backwards pass on that loss.
        x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.double).reshape(1, S).repeat(S, 1)
        forcing = -4 * (torch.cos(4*(x2))).reshape(1,S,S,1).repeat(B, 1, 1, T-2).to(device)
        
        y = torch.tensor(np.linspace(0, 1.0, S+1)[:-1]).to(device)
        x = y

        __,dUx_dy = torch.gradient(model_output[...,0], spacing = tuple([x, y]), dim = [1,2])
        dUy_dx,__ = torch.gradient(model_output[...,1], spacing = tuple([x, y]), dim = [1,2])
        model_output_curl = (dUy_dx - dUx_dy).unsqueeze(-1)

        Dw = FDM_NS_vorticity(model_output_curl, nu=nu, t_interval=t_interval)
        loss_w = F.mse_loss(Dw, forcing)
        
        # next we will calculate the cartesian losses for logging and further comparison
        eqn_c, eqn_mx, eqn_my = FDM_NS_cartesian(model_output, nu=nu, t_interval=t_interval)
        forcing_x = -1 * (torch.sin(4*(x2))).reshape(1,S,S,1).repeat(B, 1, 1, T-2).to(device)

        torch_zero_tensor = torch.zeros(eqn_c.shape, device=eqn_c.device)
        loss_c  = F.mse_loss(eqn_c, torch_zero_tensor)
        loss_m1 = F.mse_loss(eqn_mx, forcing_x)
        loss_m2 = F.mse_loss(eqn_my, torch_zero_tensor)

        # will still use the cartesian LP loss
        loss_l2 = F.mse_loss(model_output, model_val)

    elif forcing_type == 'cartesian_periodic_short':
        
        eqn_c, eqn_mx, eqn_my = FDM_NS_cartesian_v2(model_output, L=2*np.pi, nu=nu, t_interval=t_interval)
        
        torch_zero_tensor = torch.zeros(eqn_c.shape, device=eqn_c.device)
        x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
        
        forcing_x = (-1*torch.sin(4*(x2))).reshape(1,S,S,1).repeat(B, 1, 1, T-2).to(device)

        loss_c = F.mse_loss(eqn_c, torch_zero_tensor)
        loss_m1 = F.mse_loss(eqn_mx, forcing_x)
        loss_m2 = F.mse_loss(eqn_my, torch_zero_tensor)
        loss_l2 = F.mse_loss(model_output, model_val)
    elif forcing_type == 'vorticity_periodic_short':
        print('Performing FDM Vorticity ALl losses MSE')
        x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
        forcing = -4 * (torch.cos(4*(x2))).reshape(1,S,S,1).repeat(B, 1, 1, T-2).to(device)
        Dw1 , Dw2 = FDM_NS_vorticity_v2(model_output, nu=nu, t_interval=t_interval)

        loss_w = F.mse_loss(Dw1, forcing)
        loss_c = F.mse_loss(Dw2, forcing) # <- Storing in continuity equation, cause why not
        loss_l2 = F.mse_loss(model_output, model_val)
    else:
        raise Exception('Wrong Use case, need to rewrite code, and use different PINO_loss3d_decider')

    return loss_l2, loss_ic, loss_bc, loss_w, loss_c, loss_m1, loss_m2

def PINO_loss3d_decider_ss(model_input, model_output, model_val, forcing_type, nu, t_interval):
    
    B = model_output.shape[0]
    S = model_output.shape[1]
    T = model_output.shape[3]
    C = model_output.shape[4]
    device = model_output.device

    # Set Loss function
    lploss = LpLoss(size_average=True)
    torch_zero = torch.zeros(1).to(device)

    # Intialize Losses
    loss_l2, loss_ic, loss_bc, loss_w, loss_c, loss_m1, loss_m2 = torch_zero, torch_zero, torch_zero, torch_zero, torch_zero, torch_zero, torch_zero
    
    # Inital condition is the same across all conditions and test types (3rd index onwards excludes the grid)
    u0 = model_input[:, :, :, 0, 3:] 

    loss_ic = lploss(model_output[:, :, :, 0, :], u0)

    # Select what conditions to use for PDE Loss (vorticity, cartesian, cavity)
    if forcing_type == 'vorticity_periodic_short':
        x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
        forcing = -4 * (torch.cos(4*(x2))).reshape(1,S,S,1).repeat(B, 1, 1, T-2).to(device)
        Dw = FDM_NS_vorticity(model_output, nu=nu, t_interval=t_interval)

        loss_w = lploss(Dw, forcing)
        loss_l2 = lploss(model_output, model_val)

    elif forcing_type == 'cartesian_periodic_short':
        
        eqn_c, eqn_mx, eqn_my = FDM_NS_cartesian(model_output, nu=nu, t_interval=t_interval)
        
        torch_zero_tensor = torch.zeros(eqn_c.shape, device=eqn_c.device)
        x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
        
        # edit this for different problems (curl of sin(4y) is 4cos(4y) -> forcing in other direction is irrelevant for this problem) 
        forcing_x = (torch.sin(4*(x2))).reshape(1,S,S,1).repeat(B, 1, 1, T-2).to(device)
        forcing_y = torch_zero_tensor

        error_normalizer = 2*3
        #loss_c = lploss.abs(eqn_c, torch_zero_tensor)
        #loss_m1 = lploss.abs(eqn_mx, forcing_x)
        #loss_m2 = lploss.abs(eqn_my, forcing_y)
        loss_c = F.mse_loss(eqn_c, torch_zero_tensor) / error_normalizer
        loss_m1 = F.mse_loss(eqn_mx, forcing_x) /error_normalizer
        loss_m2 = F.mse_loss(eqn_my, forcing_y) /error_normalizer 
        loss_l2 = lploss(model_output, model_val)

    elif forcing_type == 'cartesian_to_vorticity_trial':
        
        # Here we will convert the output to vorticity form and run the backwards pass on that loss.
        x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
        forcing = -4 * (torch.cos(4*(x2))).reshape(1,S,S,1).repeat(B, 1, 1, T-2).to(device)
        
        y = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1]).to(device)
        x = y

        __,dUx_dy = torch.gradient(model_output[...,0], spacing = tuple([x, y]), dim = [1,2])
        dUy_dx,__ = torch.gradient(model_output[...,1], spacing = tuple([x, y]), dim = [1,2])
        model_output_curl = (dUy_dx - dUx_dy).unsqueeze(-1)

        Dw = FDM_NS_vorticity(model_output_curl, nu=nu, t_interval=t_interval)
        loss_w = lploss(Dw, forcing)
        
        # next we will calculate the cartesian losses for logging and further comparison
        eqn_c, eqn_mx, eqn_my = FDM_NS_cartesian(model_output, nu=nu, t_interval=t_interval)
        
        torch_zero_tensor = torch.zeros(eqn_c.shape, device=eqn_c.device)
        x2 = torch.tensor(np.linspace(0, 2*np.pi, S+1)[:-1], dtype=torch.float).reshape(1, S).repeat(S, 1)
        forcing_x = -1 * (torch.sin(4*(x2))).reshape(1,S,S,1).repeat(B, 1, 1, T-2).to(device)
        forcing_y = torch_zero_tensor

        loss_c = lploss.abs(eqn_c, torch_zero_tensor)
        loss_m1 = lploss.abs(eqn_mx, forcing_x)
        loss_m2 = lploss.abs(eqn_my, forcing_y)

        # will still use the cartesian LP loss
        loss_l2 = lploss(model_output, model_val)

    elif forcing_type == 'cavity':
        eqn_c, eqn_mx, eqn_my = FDM_NS_cartesian(model_output, nu=nu, t_interval=t_interval)

        torch_zero_tensor = torch.zeros(eqn_c.shape, device=eqn_c.device)
        E1 = F.mse_loss(eqn_c,torch_zero_tensor)
        E2 = F.mse_loss(eqn_mx,torch_zero_tensor)
        E3 = F.mse_loss(eqn_my,torch_zero_tensor)

        loss_bc1 = F.mse_loss(model_output[:,0,:,...], model_val[:,0,:,...])
        loss_bc2 = F.mse_loss(model_output[:,-1,:,...], model_val[:,-1,:,...])
        loss_bc3 = F.mse_loss(model_output[:,:,-1,...], model_val[:,:,-1,...])
        loss_bc4 = F.mse_loss(model_output[:,:,0,...], model_val[:,:,0,...])

        loss_bc = (loss_bc1+loss_bc2+loss_bc3+loss_bc4)/4

    elif forcing_type == 'cartesian_periodic_short_hard_loss':
        V_tilde = FDM_NS_cartesian(model_output)
        loss_l2 = lploss(V_tilde, model_val)

    else: 
        raise(ValueError)


    return loss_l2, loss_ic, loss_bc, loss_w, loss_c, loss_m1, loss_m2
        