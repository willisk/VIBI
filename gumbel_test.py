import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


def sample_gumbel(size):
    return -torch.log(-torch.log(torch.rand(size)))


def gumbel_reparametrize(log_p, temp, num_samples):
    assert log_p.ndim == 2
    B, C = log_p.shape  # (B, C)
    shape = (B, num_samples, C)
    g = sample_gumbel(shape).to(log_p.device)    # (B, N, C)
    return F.softmax((log_p.unsqueeze(1) + g) / temp, dim=-1)  # (B, N, C)


# this is only a, at most k-hot relaxation
def k_hot_relaxed(log_p, k, temp, num_samples):
    assert log_p.ndim == 2  # (B, C)
    B, C = log_p.shape
    shape = (k, B, C)
    k_log_p = log_p.unsqueeze(0).expand(shape).reshape((k * B, C))   # (k * B, C)
    k_hot = gumbel_reparametrize(k_log_p, temp, num_samples).reshape((k, B, num_samples, C))  # (k, B, N, C)
    k_hot, _ = k_hot.max(dim=0)  # (B, N, C)
    return k_hot  # (B, N, C)


from torch.distributions import RelaxedOneHotCategorical
from debug import debug

B, C = 1, 5

log_p = torch.randn((B, C))

p = F.softmax(log_p, dim=-1)
for i in range(B):
    plt.bar(range(C), p[i])
    plt.show()

k = 3
num_samples = 100
for t in 10.0 ** torch.arange(-3, 4):
    print(f'temperature: {round(t.item(), 3)}')
    q = k_hot_relaxed(log_p, k, t, num_samples)   # (B, N, C)
    # relaxed_one_hot = RelaxedOneHotCategorical(t, logits=log_p)
    # qz = relaxed_one_hot.sample((B, num_samples))
    for i in range(B):
        # plt.subplot(1, 2, 1)
        plt.bar(range(C), q[i].mean(dim=0))
        # plt.subplot(1, 2, 2)
        # plt.bar(range(C), qz[i].mean(dim=0))
        plt.show()
