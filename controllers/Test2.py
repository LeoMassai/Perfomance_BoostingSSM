import torch
import math
import time

class PScan(torch.autograd.Function):
    # Given A is NxTx1 and X is NxTxD, expands A and X in place in O(T),
    # and O(log(T)) if not core-bounded, so that
    #
    # Y[:, 0] = Y_init
    # Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
    #
    # can be computed as
    #
    # Y[:, t] = A[:, t] * Y_init + X[:, t]

    @staticmethod
    def expand_(A, X):
        if A.size(1) == 1:
            return
        T = 2 * (A.size(1) // 2)
        Aa = A[:, :T].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, :T].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
        Aa[:, :, 1].mul_(Aa[:, :, 0])
        PScan.expand_(Aa[:, :, 1], Xa[:, :, 1])
        Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
        Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
        if T < A.size(1):
            X[:, -1].add_(A[:, -1].mul(X[:, -2]))
            A[:, -1].mul_(A[:, -2])

    @staticmethod
    def acc_rev_(A, X):
        if X.size(1) == 1:
            return
        T = 2 * (X.size(1) // 2)
        Aa = A[:, -T:].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, -T:].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 0].add_(Aa[:, :, 1].mul(Xa[:, :, 1]))
        B = Aa[:, :, 0].clone()
        B[:, 1:].mul_(Aa[:, :-1, 1])
        PScan.acc_rev_(B, Xa[:, :, 0])
        Xa[:, :-1, 1].add_(Aa[:, 1:, 0].mul(Xa[:, 1:, 0]))
        if T < A.size(1):
            X[:, 0].add_(A[:, 1].mul(X[:, 1]))

    # A is NxT, X is NxTxD, Y_init is NxD
    #
    # returns Y of same shape as X, with
    #
    # Y[:, t] = A[:, 0] * Y_init   + X[:, 0] if t == 0
    #         = A[:, t] * Y[:, t-1] + X[:, t] otherwise

    @staticmethod
    def forward(ctx, A, X, Y_init):
        ctx.A = A[:, :, None].clone()
        ctx.Y_init = Y_init[:, None, :].clone()
        ctx.A_star = ctx.A.clone()
        ctx.X_star = X.clone()
        PScan.expand_(ctx.A_star, ctx.X_star)
        return ctx.A_star * ctx.Y_init + ctx.X_star

    @staticmethod
    def backward(ctx, grad_output):
        # ppprint(grad_output)
        U = grad_output * ctx.A_star
        A = ctx.A.clone()
        R = grad_output.clone()
        PScan.acc_rev_(A, R)
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
        return (Q * R).sum(-1), R, U.sum(dim=1)


pscan = PScan.apply

# A = torch.randn(3, 5, dtype=torch.float64).requires_grad_()
#
# X = torch.randn(3, 5, 1, dtype=torch.float64).requires_grad_()
#
# Y0 = torch.randn(3, 1, dtype=torch.float64).requires_grad_()
#
#
# y = Y0[:, None]
#
#
# for k in range(A.size(1)):
#     y = A[:, k, None].unsqueeze(1) * y + X[:, k,:].unsqueeze(1)
#
#     print(f"{k} -> {y}")
#
#
#
#
# Y = pscan(A, X, Y0)
#
# for k in range(A.size(1)):
#     print(f"{k} -> {Y[:, k]}")
# y = Y[:, -1]
#
# y



N, T, D = 3, 6, 2

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

Y0c = torch.randn(3, 1,  dtype=torch.float64).requires_grad_().cuda()

y = Y0c.unsqueeze(1)

for k in range(Ac.size(1)):
    y = Ac[:, k, None].unsqueeze(1) * y + x[:, k,:].unsqueeze(1)
    #print(f"{k} -> {y}")


t0= time.time()
Ys = pscan(Ac, x, Y0c)
t1= time.time()

tscan = t1-t0


# for k in range(Ac.size(1)):
#     print(f"{k} -> {Y[:, k]}")
# y = Y[:, -1]


#timing
y=torch.complex(torch.zeros(N, T, D, dtype=torch.float).requires_grad_().cuda(),
                  torch.zeros(N, T, D, dtype=torch.float).requires_grad_().cuda())

y[:,0,:] = Y0c

t0= time.time()
for k in range(1, Ac.size(1)):
    y[:,k,:] = Ac[:, k].unsqueeze(1) * y[:,k-1,:] + x[:, k-1,:]
t1= time.time()

trec = t1-t0

y