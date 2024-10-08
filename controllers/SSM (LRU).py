import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Define the model using nn.Sequential
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # First layer
            nn.SiLU(),  # Activation after the first layer
            nn.Linear(hidden_size, hidden_size),  # Hidden layer
            nn.ReLU(),  # Activation after hidden layer
            nn.Linear(hidden_size, output_size)  # Output layer (no activation)
        )

    def forward(self, x):
        if x.dim() == 3:
            # x is of shape (batch_size, sequence_length, input_size)
            batch_size, seq_length, input_size = x.size()

            # Flatten the batch and sequence dimensions to apply MLP across all elements
            x = x.view(batch_size * seq_length, input_size)  # Shape: (batch_size * sequence_length, input_size)

            # Apply the MLP to each feature vector
            x = self.model(x)

            # Reshape back to (batch_size, sequence_length, output_size)
            x = x.view(batch_size, seq_length, -1)  # Shape: (batch_size, sequence_length, output_size)
        else:
            x = self.model(x)

        return x


class LRU(nn.Module):
    def __init__(self, in_features, out_features, state_features, rmin=0.9, rmax=1, max_phase=6.283):
        super().__init__()
        self.out_features = out_features
        self.D = nn.Parameter(torch.randn([out_features, in_features]) / math.sqrt(in_features))
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod) - torch.square(Lambda_mod))))
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))
        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))
        self.state = torch.complex(torch.zeros(state_features), torch.zeros(state_features))

    def forward(self, input):
        self.state = self.state.to(self.B.device)
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        Lambda_re = Lambda_mod * torch.cos(torch.exp(self.theta_log))
        Lambda_im = Lambda_mod * torch.sin(torch.exp(self.theta_log))
        Lambda = torch.complex(Lambda_re, Lambda_im)  # Eigenvalues matrix
        Lambda = Lambda.to(self.state.device)
        gammas = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        gammas = gammas.to(self.state.device)
        output = torch.empty([i for i in input.shape[:-1]] + [self.out_features], device=self.B.device)
        # Handle input of (Batches,Seq_length, Input size)
        if input.dim() == 3:
            for i, batch in enumerate(input):
                out_seq = torch.empty(input.shape[1], self.out_features)
                for j, step in enumerate(batch):
                    self.state = (Lambda * self.state + gammas * self.B @ step.to(dtype=self.B.dtype))
                    out_step = 2 * (self.C @ self.state).real + self.D @ step
                    out_seq[j] = out_step
                self.state = torch.complex(torch.zeros_like(self.state.real), torch.zeros_like(self.state.real))
                output[i] = out_seq
        # Handle input of (Seq_length, Input size)
        if input.dim() == 2:
            for i, step in enumerate(input):
                self.state = (Lambda * self.state + gammas * self.B @ step.to(dtype=self.B.dtype))
                out_step = 2 * (self.C @ self.state).real + self.D @ step
                output[i] = out_step
            self.state = torch.complex(torch.zeros_like(self.state.real), torch.zeros_like(self.state.real))
        return output


class SSM(nn.Module):  # LRU + scaffolding
    def __init__(self, in_features, out_features, state_features, mlp_hidden_size=30, rmin=0.9, rmax=1,
                 max_phase=6.283):
        super().__init__()
        self.mlp = MLP(out_features, mlp_hidden_size, out_features)
        self.LRU = LRU(in_features, out_features, state_features, rmin, rmax, max_phase)
        self.model = nn.Sequential(self.LRU, self.mlp)
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, input):
        result = self.model(input) + self.lin(input)
        return result


class DeepLRU(nn.Module):  # implements a cascade of SSMs
    def __init__(self, N, in_features, out_features, mid_features, state_features):
        super().__init__()
        self.linin = nn.Linear(in_features, mid_features)
        self.linout = nn.Linear(mid_features, out_features)
        self.modelt = nn.ModuleList(
            [SSM(mid_features, mid_features, state_features) for j in range(N)])
        self.modelt.insert(0, self.linin)
        self.modelt.append(self.linout)
        self.model = nn.Sequential(*self.modelt)

    def forward(self, input):
        result = self.model(input)
        return result


class SSL2(nn.Module):
    def __init__(self, in_features, mid_features, out_features, state_features, rmin, rmax,
                 max_phase, mlp_hidden_size=30):
        super().__init__()
        #self.mlp = MLP(in_features, mlp_hidden_size, mid_features)
        self.LRU = LRU(mid_features, mid_features, state_features, rmin, rmax, max_phase)
        #self.model = nn.Sequential(self.mlp, self.LRU)
        self.lin_in = nn.Linear(in_features, mid_features)
        self.lin_out = nn.Linear(mid_features, out_features)
        self.lin_skip = nn.Linear(in_features, mid_features)
        self.silu = nn.SiLU()

    def forward(self, input):
        result = self.lin_out(self.silu(self.LRU(self.lin_in(input))) + self.lin_skip(input))
        return result


class DeepLRU2(nn.Module):
    def __init__(self, N, in_features, out_features, mid_features, state_features, rmin=0.9, rmax=1,
                 max_phase=6.283):
        super().__init__()
        self.linout = nn.Linear(in_features, out_features)
        modelt = nn.ModuleList(
            [SSL2(in_features, mid_features, in_features, state_features, rmin, rmax,
                  max_phase) for j in range(N)])
        modelt.append(self.linout)
        self.model = nn.Sequential(*modelt)

    def forward(self, input):
        result = self.model(input)
        return result
