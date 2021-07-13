import os
import sys
import math

from debug import debug

import torch
import torch.nn as nn
import torch.optim

from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, TensorDataset, random_split

import torch.nn.functional as F
import torchvision.transforms as T

from models import ResNet, resnet18, resnet34, Unet

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='MNIST')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'], default='ResNet_4x')
parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=1)
parser.add_argument('--resume_training', action='store_true', default=False)
parser.add_argument('--num_samples', type=int, default=4,
                    help='Number of samples used for estimating expectation over p(t|x)')
parser.add_argument('--beta', type=float, default=0, help='beta in objective J = I(y,t) - beta * I(x,t)')

if 'ipykernel_launcher' in sys.argv[0]:
    args = parser.parse_args([
        # '--dataset=MNIST',
        '--cuda',
        # '--cpu',
        # '--explainer_type=ResNet_2x',
        # '--beta=0.1',
        '--dataset=CIFAR10',
        '--explainer_type=ResNet_4x',
        '--beta=0',
        # '--resume_training',
    ])
else:
    args = parser.parse_args()

dataset = args.dataset
device = 'cuda' if args.cuda else 'cpu'

print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

if dataset == 'MNIST':
    transform = T.Compose([T.ToTensor(),
                           T.Normalize(0.1307, 0.3080)])
    train_set = MNIST('~/data', train=True, transform=transform, download=True)
    test_set = MNIST('~/data', train=False, transform=transform, download=True)
elif dataset == 'CIFAR10':
    transform = T.Compose([T.ToTensor(),
                           T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608))])
    train_set = CIFAR10('~/data', train=True, transform=transform, download=True)
    test_set = CIFAR10('~/data', train=False, transform=transform, download=True)

split = int(len(train_set) * 0.9)
train_set, val_set = random_split(train_set, [split, len(train_set) - split])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=8)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)


def calculate_mean_and_std(data_loader):
    mean = 0
    std = 0
    with torch.no_grad():
        for x, y in data_loader:
            mean += x.mean(dim=[0, 2, 3])
            std += x.std(dim=[0, 2, 3])
    mean /= len(data_loader)
    std /= len(data_loader)
    print(f'mean: {mean}')
    print(f'std: {std}')


# calculate_mean_and_std(train_loader)


##################################### Train Black Box #####################################


@torch.no_grad()
def test_accuracy(model, data_loader, name='test'):
    num_total = 0
    num_correct = 0
    model.eval()
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        num_correct += (out.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    print(f'{name} accuracy: {acc:.3f}')
    return acc


def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


model_ckpt = f'models/{dataset}_black_box.pt'
pred_ckpt = f'models/{dataset}_black_box_predictions.pt'
os.makedirs('models', exist_ok=True)

if os.path.exists(model_ckpt):
    black_box = torch.load(model_ckpt, map_location=device)
else:
    if dataset == 'MNIST':
        black_box = resnet18(1, 10)
        lr = 0.05
        num_epochs = 2
    elif dataset == 'CIFAR10':
        black_box = resnet34(3, 10)
        lr = 0.005
        num_epochs = 40
    print('Training black box model.')
    print(f'black_box params: \t{num_params(black_box) / 1000:.2f} K')
    optimizer = torch.optim.Adam(black_box.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    black_box.to(device)

    for epoch in range(num_epochs):
        black_box.train()
        for step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            logits = black_box(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            if step % 500 == 0:
                print(f'[{epoch}:{step:3d}] accuracy {acc:.3f}, loss {loss.item():.3f}')
        black_box.eval()
        test_accuracy(black_box, val_loader, 'valid')
    torch.save(black_box, model_ckpt)
if os.path.exists(pred_ckpt):
    bb_dataset = torch.load(pred_ckpt)
else:
    x_full = []
    y_full = []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            logits = black_box(x)
            y = F.softmax(logits, dim=1)
            x_full.append(x.cpu())
            y_full.append(y.cpu())
        x_full = torch.cat(x_full)
        y_full = torch.cat(y_full)
        bb_dataset = TensorDataset(x_full, y_full)
        torch.save(bb_dataset, pred_ckpt)

bb_loader = DataLoader(bb_dataset, batch_size=64, shuffle=True, num_workers=8)

# test_accuracy(black_box, test_loader, 'black_box test')


########################################## VIBI #########################################


def sample_gumbel(size):
    return -torch.log(-torch.log(torch.rand(size)))


def gumbel_reparametrize(log_p, temp, num_samples):
    assert log_p.ndim == 2
    B, C = log_p.shape                                              # (B, C)
    shape = (B, num_samples, C)
    g = sample_gumbel(shape).to(log_p.device)                       # (B, N, C)
    return F.softmax((log_p.unsqueeze(1) + g) / temp, dim=-1)       # (B, N, C)


# this is only a, at most k-hot relaxation
def k_hot_relaxed(log_p, k, temp, num_samples):
    assert log_p.ndim == 2
    B, C = log_p.shape                                              # (B, C)
    shape = (k, B, C)
    k_log_p = log_p.unsqueeze(0).expand(shape).reshape((k * B, C))  # (k* B, C)
    k_hot = gumbel_reparametrize(k_log_p, temp, num_samples)        # (k* B, N, C)
    k_hot = k_hot.reshape((k, B, num_samples, C))                   # (k, B, N, C)
    k_hot, _ = k_hot.max(dim=0)                                     # (B, N, C)
    return k_hot                                                    # (B, N, C)


# needed for when labels are not one-hot
def soft_cross_entropy(logits, y):
    return -(y * F.log_softmax(logits, dim=-1)).mean() * y.shape[-1]


class LogSoftmax2d(nn.Module):
    def forward(self, x):
        return F.log_softmax(x.reshape((len(x), -1)), dim=1).reshape(x.shape)


# %%

class VIBI(nn.Module):
    def __init__(self, explainer, approximator, k=4, num_samples=4, temp=1):
        super().__init__()

        self.explainer = explainer
        self.approximator = approximator

        self.k = k
        self.temp = temp
        self.num_samples = num_samples

        self.warmup = False

    def explain(self, x, mode='topk', num_samples=None):
        """Returns the relevance scores
        """

        k = self.k
        temp = self.temp
        N = num_samples or self.num_samples

        B, C, H, W = x.shape

        logits_z = self.explainer(x)                                            # (B, C, h, w)
        B, C, h, w = logits_z.shape
        logits_z = logits_z.reshape((B, -1))                                    # (B, C* h* w)

        if mode == 'distribution':  # return the distribution over explanation
            p_z = F.softmax(logits_z, dim=1).reshape((B, C, h, w))              # (B, C, h, w)
            p_z_upsampled = F.interpolate(p_z, (H, W), mode='nearest')          # (B, C, H, W)
            return p_z_upsampled
        elif mode == 'topk':    # return top k pixels from input
            _, top_k_idx = torch.topk(logits_z, k, dim=1)
            k_hot_z = torch.zeros_like(logits_z).scatter_(1, top_k_idx, 1)      # (B, C* h* w)
            k_hot_z = k_hot_z.reshape((B, C, h, w))                             # (B, C, h, w)
            k_hot_z_upsampled = F.interpolate(k_hot_z, (H, W), mode='nearest')  # (B, C, H, W)
            return k_hot_z_upsampled
        elif mode == 'sample':  # return (not always) k hot vector over pixels sampled from p_z
            k_hot_z = k_hot_relaxed(logits_z, k, temp, N)                       # (B, N, F)
            k_hot_z = k_hot_z.reshape((B * N, C, h, w))                         # (B* N, C, h, w)

            k_hot_z_upsampled = F.interpolate(k_hot_z, (H, W), mode='nearest')  # (B* N, C, H, W)
            k_hot_z_upsampled = k_hot_z_upsampled.reshape((B, N, C, H, W))      # (B, N, C, H, W)
            return k_hot_z_upsampled, logits_z
        elif mode == 'warmup':
            return logits_z

    def forward(self, x, mode='topk'):

        N = self.num_samples
        B, C, H, W = x.shape

        if mode == 'warmup':
            logits_z_dummy = torch.ones_like(x) / x.numel()                     # (B, C, H, W)
            logits_z_dummy = logits_z_dummy.reshape((B, -1))                    # (B, C* H* W)
            logits_y = self.approximator(x)                                     # (B, 10)
            return logits_z_dummy, logits_y

        if mode == 'sample':
            k_hot_z_upsampled, logits_z_flat = self.explain(x, mode=mode)       # (B, N, C, H, W), (B, C* h* w)

            t = x.unsqueeze(1) * k_hot_z_upsampled                              # (B, N, C, H, W)
            t = t.reshape((B * N, C, H, W))                                     # (B* N, C, H, W)
            logits_y = self.approximator(t)                                     # (B* N, 10)
            logits_y = logits_y.reshape((B, N, 10))                             # (B, N, 10)
            return logits_z_flat, logits_y
        elif mode == 'topk':
            k_hot_z_upsampled = self.explain(x, mode=mode)                      # (B, N, C, H, W)
            t = x * k_hot_z_upsampled                                           # (B, C, H, W)
            logits_y = self.approximator(t)                                     # (B, 10)
            return logits_y


import torchvision
import matplotlib.pyplot as plt


def to_zero_one(x):
    x_r = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)
    x_min = x_r.min(dim=1)[0].reshape(1, x.shape[1], 1, 1)
    x_max = x_r.max(dim=1)[0].reshape(1, x.shape[1], 1, 1)
    return (x - x_min) / (x_max - x_min)


@torch.no_grad()
def inspect_explanations(save_loc=None, mode='topk'):
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        z = vibi.explain(x, mode=mode)
        x = to_zero_one(x)

        # r = x * (1 - z) + (1 - x) * z + x * z
        # g = x * (1 - z) + (1 - x) * z
        # b = x - x * z
        # xpl = torch.cat([r, g, b], dim=1)
        # xpl = x * z + 0.5 * (1 - z)

        xpl = x * z + 0.5 * (1 - z)

        grid = torchvision.utils.make_grid(xpl.cpu())
        plt.figure(figsize=(14, 8))
        plt.axis('off')
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()
        if save_loc is not None:
            torchvision.utils.save_image(grid, save_loc)
        break


# %%
explainer_type = args.explainer_type
xpl_channels = args.xpl_channels
num_samples = args.num_samples
beta = args.beta

if dataset == 'MNIST':
    approximator = resnet18(1, 10)
    xpl_channels = 1

    if explainer_type == 'ResNet_4x':
        block_features = [64] * 2 + [128] * 3 + [256] * 4
        explainer = ResNet(1, block_features, 1, headless=True)
        k = 4   # 4x4 chunksize
    elif explainer_type == 'ResNet_2x':
        block_features = [64] * 10
        explainer = ResNet(1, block_features, 1, headless=True)
        k = 12   # 2x2 chunksize
    elif explainer_type == 'Unet':
        explainer = Unet(1, [64, 128, 256], 1)
        k = 16  # 1x1 chunksize
    else:
        raise ValueError
    lr = 0.05
    num_epochs = 3
    temp_warmup = 200

elif dataset == 'CIFAR10':
    approximator = resnet18(3, 10)

    if explainer_type == 'ResNet_8x':
        block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
        explainer = ResNet(3, block_features, xpl_channels, headless=True)
        k = 4   # 8x8 chunksize
    if explainer_type == 'ResNet_4x':
        block_features = [64] * 3 + [128] * 4 + [256] * 5
        explainer = ResNet(3, block_features, xpl_channels, headless=True)
        k = 8   # 4x4 chunksize, note: k=4 too small
    elif explainer_type == 'ResNet_2x':
        block_features = [64] * 4 + [128] * 5
        explainer = ResNet(3, block_features, xpl_channels, headless=True)
        k = 16   # 2x2 chunksize
    elif explainer_type == 'Unet':
        explainer = Unet(3, [64, 128, 256, 512], xpl_channels)
        k = 16  # 1x1 chunksize
    else:
        raise ValueError

    lr = 0.005
    temp_warmup = 4000
    num_epochs = 20

print(f'k = {k}')

result_dir = f'results/{dataset}_{explainer_type}_{xpl_channels}_vibi_k={k}_b={beta}'
images_fmt = f'results/{dataset}_{explainer_type}_{xpl_channels}_vibi_k={k}_b={beta}/{{}}.png'
model_ckpt = f'models/{dataset}_{explainer_type}_{xpl_channels}_vibi_k={k}_b={beta}.pt'

os.makedirs(result_dir, exist_ok=True)

if os.path.exists(model_ckpt):
    state_dict = torch.load(model_ckpt, map_location=device)
    vibi = state_dict['model']
    optimizer = state_dict['optimizer']
    init_epoch = state_dict['epoch']
    logs = state_dict['logs']
    best_acc = state_dict['acc']
    print(f'Model {model_ckpt} loaded, valid_acc {acc}')
else:
    init_epoch = 0
    best_acc = 0
    logs = []


if init_epoch == 0 or args.resume_training:
    vibi = VIBI(explainer, approximator, k=k, num_samples=args.num_samples)
    vibi.to(device)
    inspect_explanations()

    print('Training VIBI')
    print(f'{explainer_type:>10} explainer params:\t{num_params(vibi.explainer) / 1000:.2f} K')
    print(f'{type(approximator).__name__:>10} approximator params:\t{num_params(vibi.approximator) / 1000:.2f} K')

    optimizer = torch.optim.Adam(vibi.parameters(), lr=lr)

    for epoch in range(init_epoch, num_epochs):
        vibi.train()
        for step, (x, y) in enumerate(bb_loader, start=epoch * len(bb_loader)):
            t = step - temp_warmup
            vibi.temp = 10 / t if t > 1 else 10
            warmup = t < 1
            # warmup = False

            x, y = x.to(device), y.to(device)                   # (B, C, H, W), (B, 10)
            if warmup:  # note: warmup logits_z might have different shape (.., H, W) instead of (h, w)
                logits_z, logits_y = vibi(x, mode='warmup')     # (B, C, H, W), (B, 10)
            else:
                y = y.reshape(-1, 1, 10)  # (B, 1, 10)
                logits_z, logits_y = vibi(x, mode='sample')     # (B, C* h* w), (B, N, 10)

            logits_z = logits_z.log_softmax(dim=1)

            H_p_q = soft_cross_entropy(logits_y, y)
            KL_z_r = (torch.exp(logits_z) * logits_z).sum(dim=1).mean() + math.log(logits_z.shape[1])

            # if KL_z_r < 0.01:
            #     beta = beta * 0.1
            #     print('Decreasing beta')

            if warmup:    # warmup stage can kill signal
                loss = H_p_q
            else:
                loss = H_p_q + beta * KL_z_r

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not warmup:
                logits_y = logits_y.mean(dim=1)
                y = y.squeeze()

            acc = (logits_y.argmax(dim=1) == y.argmax(dim=1)).float().mean().item()

            metrics = {
                'accuracy': acc,
                'loss': loss.item(),
                'temp': vibi.temp,
                'H(p,q)': H_p_q.item(),
                'KL(z||r)': KL_z_r.item(),
            }

            logs.append(metrics)

            if step % len(bb_loader) % 50 == 0:
                print(f'[{epoch}/{init_epoch + num_epochs}:{step % len(bb_loader):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()])
                      + (' ~ warmup' if warmup else ''))

        vibi.eval()
        valid_acc = test_accuracy(vibi, val_loader, name='vibi valid top1')
        inspect_explanations(images_fmt.format(epoch))

        if valid_acc > best_acc:
            best_acc = valid_acc

            print(f'Saving model to {model_ckpt}')
            torch.save({'model': vibi, 'optimizer': optimizer, 'epoch': epoch,
                       'k': k, 'beta': beta, 'acc': best_acc, 'logs': logs}, model_ckpt)

test_accuracy(vibi, test_loader, name='vibi test top1')
inspect_explanations()


def transpose_dict(d):
    return {k: v for k, v in zip(d[0].keys(), zip(*[e.values() for e in d]))}


def pretty_plot(logs):
    vals = torch.Tensor(list(logs.values()))
    y_max = max(vals.mean(dim=1) + vals.std(dim=1) * 1.5)

    plt.figure(figsize=(14, 8))
    for m, values in logs.items():
        if m == 'accuracy':
            values = [v * y_max for v in values]
        plt.plot(range(len(values)), values, label=m)

    plt.ylim([0, y_max])
    plt.legend()
    plt.show()


logs_t = transpose_dict(logs)
pretty_plot(logs_t)
