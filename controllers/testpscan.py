import torch
import time
import math
import torch.nn.functional as F





def L_at_X(X):
    N, T, D = X.shape
    device = X.device
    X_ = X.transpose(0, 1)
    X_ = torch.cat([X_, torch.zeros(T - 1, N, D, device=device)], dim=0)

    L = torch.cat(
        [torch.ones(T, device=device), torch.zeros(T - 1, device=device)], dim=0
    )
    L = L.unsqueeze(1).unsqueeze(2)

    output = torch.fft.ifft(
        torch.fft.fft(L, dim=0) * torch.fft.fft(X_, dim=0), n=2 * T - 1, dim=0
    )
    output = output[:T, :, :].transpose(0, 1)
    return output


def U_at_A(A):
    N, T = A.shape
    device = A.device
    A_ = A.transpose(0, 1)
    A_ = torch.cat([A_, torch.zeros(T - 1, N, device=device)], dim=0)

    L_no_diag = torch.cat(
        [
            torch.zeros(1, device=device),
            torch.ones(T - 1, device=device),
            torch.zeros(T - 1, device=device),
        ],
        dim=0,
    )
    L_no_diag = L_no_diag.unsqueeze(1)

    L_no_diag_at_A = torch.fft.ifft(
        torch.fft.fft(L_no_diag, dim=0) * torch.fft.fft(A_, dim=0), n=2 * T - 1, dim=0
    )
    # Since we add T - 1 of padding zeros to A_log_T
    output = A_.sum(0).unsqueeze(0) - L_no_diag_at_A
    output = output[:T, :].transpose(0, 1)
    return output




def pscan_fft_efficient(A, X):
    N, T, D = X.shape
    device = X.device

    # A_log \in [N x T]
    A_log = torch.log(A.to(dtype=torch.cfloat))

    UA = U_at_A(A_log)
    W = UA
    W = W.real
    W_max = W.max()
    e_W = torch.exp(W - W_max)
    e_W = e_W.unsqueeze(-1)

    V = -UA + A_log
    V = V.real
    V_max = V.max()
    e_V = torch.exp(V - V_max)
    e_V = e_V.unsqueeze(-1)
    Y_ = e_V * L_at_X(e_W * X) * (torch.exp(V_max + W_max))

    # After exp we no longer have complex components
    Y_ = Y_.real
    Y_ = torch.cat([torch.zeros(N, 1, D, device=device), Y_[:, :-1, :]], dim=1)
    Y = Y_ + X
    return Y


N, T, D = 3, 9, 1

device = 'cuda'
rmin=0.2
rmax=.7
max_phase=6.283
u1 = torch.rand(N)
u2 = torch.rand(N)
nu_log = torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2))
theta_log = torch.log(max_phase * u2)
theta_log = torch.log(max_phase * u2)
Lambda_mod = torch.exp(-torch.exp(nu_log))
Lambda_re = Lambda_mod * torch.cos(torch.exp(theta_log))
Lambda_im = Lambda_mod * torch.sin(torch.exp(theta_log))
Lambda = torch.complex(Lambda_re, Lambda_im)  # Eigenvalues matrix
Lambda = Lambda.to(device)
Lambda = Lambda.unsqueeze(1)
Ac = torch.tile(Lambda, (1, T))



A = torch.randn(N, T).requires_grad_().cuda() + 1
X = torch.randn(N, T, D).requires_grad_().cuda()


udim=3

u = torch.randn(udim, T, D).requires_grad_().cuda()
B_re = torch.randn([N, udim]).requires_grad_().cuda() / math.sqrt(2 * udim)
B_im = torch.randn([N, udim]).requires_grad_().cuda() / math.sqrt(2 * udim)
b = torch.complex(B_re, B_im)



# Unsqueeze b to make its shape (N, V, 1, 1)
b_unsqueezed = b.unsqueeze(-1).unsqueeze(-1)

# Now broadcast b along dimensions T and D so it can be multiplied elementwise with u
b_broadcasted = b_unsqueezed.expand(N, udim, T, D)

# Expand u so that it can be multiplied along dimension N, resulting in shape (N, V, T, D)
u_broadcasted = u.unsqueeze(0).expand(N, udim, T, D)

# Elementwise multiplication and then sum over V (the second dimension)
x = torch.sum(b_broadcasted * u_broadcasted, dim=1)



t0= time.time()

qq=pscan_fft_efficient(Ac, x)

t1= time.time()

tscan = t1-t0




t0= time.time()


qq2=torch.complex(torch.zeros(N, T, D, dtype=torch.float).requires_grad_().cuda(),
                  torch.zeros(N, T, D, dtype=torch.float).requires_grad_().cuda())
qq2[:, 0, :] = X[:, 0, :]
for k in range(1, x.shape[1]):
    qq2[:, k, :] = Ac[:, k - 1].unsqueeze(1) * qq2[:, k - 1, :] + x[:, k-1, :]




t1 = time.time()

tnaiv = t1-t0

qq2

