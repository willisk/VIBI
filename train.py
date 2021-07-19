import os
import math
from collections import defaultdict

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
parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs for VIBI.')
parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'], default='ResNet_4x')
parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=1)
parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
parser.add_argument('--beta', type=float, default=0.0, help='beta in objective J = I(y,t) - beta * I(x,t).')
parser.add_argument('--num_samples', type=int, default=4,
                    help='Number of samples used for estimating expectation over p(t|x).')
parser.add_argument('--resume_training', action='store_true')
parser.add_argument('--save_best', action='store_true', help='Save only the best models (measured in valid accuracy).')
parser.add_argument('--save_images_every_epoch', action='store_true', help='Save explanation images every epoch.')

parser.add_argument('--jump_start', action='store_true', default=False)

import sys
if 'ipykernel' in sys.argv[0]:
    args = parser.parse_args([
        '--cuda',
        '--dataset=CIFAR10',
    ])
else:
    args = parser.parse_args()

print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

device = 'cuda' if args.cuda else 'cpu'
dataset = args.dataset

if dataset == 'MNIST':
    transform = T.Compose([T.ToTensor(),
                           T.Normalize(0.1307, 0.3080)])
    train_set = MNIST('~/data', train=True, transform=transform, download=True)
    test_set = MNIST('~/data', train=False, transform=transform, download=True)
    train_set_no_aug = train_set
elif dataset == 'CIFAR10':
    train_transform = T.Compose([T.RandomCrop(32, padding=4),
                                 T.RandomHorizontalFlip(),
                                 T.ToTensor(),
                                 T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608))])
    test_transform = T.Compose([T.ToTensor(),
                                T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608))])
    train_set = CIFAR10('~/data', train=True, transform=train_transform, download=True)
    test_set = CIFAR10('~/data', train=False, transform=test_transform, download=True)
    train_set_no_aug = CIFAR10('~/data', train=True, transform=test_transform, download=True)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8)
train_loader_full = DataLoader(train_set_no_aug, batch_size=64, shuffle=True, num_workers=8)


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
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (out.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    print(f'{name} accuracy: {acc:.3f}')
    return acc


def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


model_ckpt = f'models/{dataset}_black_box.pt'
bb_dataset_ckpt = f'models/{dataset}_black_box_predictions.pt'
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
                print(f'[{epoch}/{num_epochs}:{step:3d}] accuracy {acc:.3f}, loss {loss.item():.3f}')
        black_box.eval()
        test_accuracy(black_box, test_loader)
    torch.save(black_box, model_ckpt)
if os.path.exists(bb_dataset_ckpt):
    ckpt = torch.load(bb_dataset_ckpt)
    bb_train_set = ckpt['train_set']
    inspect_batch = ckpt['inspect_batch']
else:
    x_full = []
    y_full = []
    with torch.no_grad():
        for x, y in train_loader_full:
            x = x.to(device)
            logits = black_box(x)
            y = F.softmax(logits, dim=1)
            x_full.append(x.cpu())
            y_full.append(y.cpu())
        x_full = torch.cat(x_full)
        y_full = torch.cat(y_full)
        bb_train_set = TensorDataset(x_full, y_full)

        # find a good batch to inspect
        max_false_labeled = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if x.shape[0] == 64:
                out = black_box(x)
                num_false = (out.argmax(dim=1) != y).sum()
                if num_false > max_false_labeled:
                    max_false_labeled = num_false
                    inspect_batch = x.cpu(), y.cpu()

        torch.save({'train_set': bb_train_set, 'inspect_batch': inspect_batch}, bb_dataset_ckpt)


torch.manual_seed(1)
split = int(len(bb_train_set) * 0.9)
bb_train_set, bb_valid_set = random_split(bb_train_set, [split, len(bb_train_set) - split])

bb_train_loader = DataLoader(bb_train_set, batch_size=64, shuffle=True, num_workers=8)
bb_valid_loader = DataLoader(bb_valid_set, batch_size=64, shuffle=False, num_workers=8)

# test_accuracy(black_box, test_loader, 'black_box test')   # XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX

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
def soft_cross_entropy_loss(logits, y):
    return -(y * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()


def js_div(p, q):
    m = (p + q) / 2
    return 0.5 * (p * torch.log(p / m) + q * torch.log(q / m)).sum(dim=1) / math.log(2)


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
        elif mode == 'warmup':
            logits_z_dummy = torch.log(torch.ones_like(x) / x.numel())          # (B, C, H, W)
            logits_z_dummy = logits_z_dummy.reshape((B, -1))                    # (B, C* H* W)
            logits_y = self.approximator(x)                                     # (B, 10)
            return logits_z_dummy, logits_y


import torchvision
import matplotlib.pyplot as plt

k = args.k
beta = args.beta
num_samples = args.num_samples
xpl_channels = args.xpl_channels
explainer_type = args.explainer_type

if dataset == 'MNIST':
    approximator = resnet18(1, 10)
    xpl_channels = 1

    if explainer_type == 'ResNet_4x':
        block_features = [64] * 2 + [128] * 3 + [256] * 4
        explainer = ResNet(1, block_features, 1, headless=True)
    elif explainer_type == 'ResNet_2x':
        block_features = [64] * 10
        explainer = ResNet(1, block_features, 1, headless=True)
    elif explainer_type == 'Unet':
        explainer = Unet(1, [64, 128, 256], 1)
    else:
        raise ValueError
    lr = 0.05
    temp_warmup = 200

elif dataset == 'CIFAR10':
    approximator = resnet18(3, 10)

    if explainer_type == 'ResNet_8x':
        block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
        explainer = ResNet(3, block_features, xpl_channels, headless=True)
    if explainer_type == 'ResNet_4x':
        block_features = [64] * 3 + [128] * 4 + [256] * 5
        explainer = ResNet(3, block_features, xpl_channels, headless=True)
    elif explainer_type == 'ResNet_2x':
        block_features = [64] * 4 + [128] * 5
        explainer = ResNet(3, block_features, xpl_channels, headless=True)
    elif explainer_type == 'Unet':
        explainer = Unet(3, [64, 128, 256, 512], xpl_channels)
    else:
        raise ValueError

    lr = 0.005
    temp_warmup = 4000

model_ckpt = f'models/{dataset}_{explainer_type}_{xpl_channels}_k={k}_b={beta}.pt'
results_loc = f'results/{dataset}_{explainer_type}_{xpl_channels}_k={k}_b={beta}'
images_loc = f'{results_loc}/{{}}.png'
plot_loc = f'{results_loc}/logs.pdf'

load_ckpt = model_ckpt
if args.jump_start and not os.path.exists(model_ckpt):
    load_ckpt = f'models/{dataset}_{explainer_type}_{xpl_channels}_k={k}_b=0.0.pt'

os.makedirs(results_loc, exist_ok=True)

if os.path.exists(load_ckpt):
    state_dict = torch.load(load_ckpt, map_location=device)
    vibi = state_dict['model']
    optimizer = state_dict['optimizer']
    init_epoch = state_dict['epoch']
    logs = state_dict['logs']
    best_acc = state_dict['acc']
    print(f'Loading model {load_ckpt}, top1 valid acc {best_acc:.3f}')
else:
    init_epoch = 0
    best_acc = 0
    logs = defaultdict(list)

    vibi = VIBI(explainer, approximator, k=k, num_samples=args.num_samples)
    vibi.to(device)

    optimizer = torch.optim.Adam(vibi.parameters(), lr=lr)


def transpose_dict(d):
    if isinstance(d, dict):
        return [{k: v for k, v in zip(d.keys(), vals)} for vals in zip(*d.values())]
    elif isinstance(d, list):
        return {k: v for k, v in zip(d[0].keys(), zip(*[e.values() for e in d]))}


def pretty_plot(logs, smoothing=0, save_loc=None):
    vals = torch.Tensor(list(logs.values()))
    if smoothing:
        vals = F.conv1d(vals.reshape((len(vals), 1, -1)), torch.ones((1, 1, smoothing)) / smoothing).squeeze()
    y_max = max(vals.mean(dim=1) + vals.std(dim=1) * 1.5)
    x = torch.arange(vals.shape[1]) / len(bb_train_loader)

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(14, 8))
    plt.xlabel('epoch')
    main_axis = plt.gca()
    main_axis.set_ylim([0, y_max])
    scaled_axis = main_axis.twinx()
    scaled_axis.set_ylabel('%')
    scaled_axis.set_ylim([-10, 110])

    for i, (m, values) in enumerate(zip(logs, vals)):
        axis = main_axis
        if 'acc' in m or m == '1-JS(p,q)':
            values *= 100
            axis = scaled_axis
        axis.plot(x, values, label=m, c=color_cycle[i])

    legend = main_axis.legend(loc=2)
    legend.remove()
    scaled_axis.legend(loc=1)
    scaled_axis.add_artist(legend)

    if save_loc is not None:
        plt.savefig(save_loc, transparent=True)
        plt.savefig(save_loc.replace('.pdf', '.png'))

    plt.show()


def to_zero_one(x, scale_each=False):
    if scale_each:
        x_r = x.reshape(x.shape[0], -1)
        x_min = x_r.min(dim=1)[0].reshape(x.shape[0], 1, 1, 1)
        x_max = x_r.max(dim=1)[0].reshape(x.shape[0], 1, 1, 1)
    else:
        x_min = x.min()
        x_max = x.max()
    return (x - x_min) / (x_max - x_min)


@torch.no_grad()
def inspect_explanations(save_loc=None, mode='topk', highlight=True):
    x, y = inspect_batch
    x, y = x.to(device), y.to(device)

    bb_logits_y = black_box(x)
    bb_y_correct = (bb_logits_y.argmax(dim=1) == y).float()

    z = vibi.explain(x, mode=mode)
    if mode == 'distribution':
        mag = ((k / 4)**(1.3) * 8)
        z = (z * mag).clamp(0, 1)   # magnify

    x = to_zero_one(x)
    xpl = x * z + 0.5 * (1 - z)

    # r = x * (1 - z) + (1 - x) * z + x * z
    # g = x * (1 - z) + (1 - x) * z
    # b = x - x * z
    # xpl = torch.cat([r, g, b], dim=1)
    # xpl = x * z + 0.5 * (1 - z)

    if highlight:
        vibi_logits_y = vibi(x, mode='topk')
        js = js_div(bb_logits_y.softmax(dim=1), vibi_logits_y.softmax(dim=1))
        js = (1 - js) ** 0.5

        xpl = xpl + torch.zeros((1, 3, 1, 1)).to(device)

        xpl = F.pad(xpl, (1, 1, 1, 1), value=-1)
        for i, (bb_c, strength) in enumerate(zip(bb_y_correct, js)):
            xpl[i][0][xpl[i][0] == -1] = (1 - bb_c) * 0.5 * (strength + 0.5)
            xpl[i][1][xpl[i][1] == -1] = bb_c * 0.5 * (strength + 0.5)
            xpl[i][2][xpl[i][2] == -1] = 0

    grid = torchvision.utils.make_grid(xpl.cpu())
    plt.figure(figsize=(14, 8))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
    if save_loc is not None:
        torchvision.utils.save_image(grid, save_loc)


valid_acc = test_accuracy(vibi, bb_valid_loader, name='vibi valid top1')

if init_epoch == 0 or args.resume_training:

    print('Training VIBI')
    print(f'{explainer_type:>10} explainer params:\t{num_params(vibi.explainer) / 1000:.2f} K')
    print(f'{type(approximator).__name__:>10} approximator params:\t{num_params(vibi.approximator) / 1000:.2f} K')

    inspect_explanations()

    for epoch in range(init_epoch, init_epoch + args.num_epochs):
        vibi.train()
        step_start = epoch * len(bb_train_loader)
        for step, (x, y) in enumerate(bb_train_loader, start=step_start):
            t = step - temp_warmup
            vibi.temp = 10 / t if t > 1 else 10
            warmup = t < 1

            x, y = x.to(device), y.to(device)                   # (B, C, H, W), (B, 10)
            # note: in the case of upsampling, warmup logits_z have different shape: (.., H, W) instead of (.., h, w)
            if warmup:
                logits_z, logits_y = vibi(x, mode='warmup')     # (B, C, H, W), (B, 10)
            else:
                y = y.reshape(-1, 1, 10)  # (B, 1, 10)
                logits_z, logits_y = vibi(x, mode='sample')     # (B, C* h* w), (B, N, 10)

            logits_z = logits_z.log_softmax(dim=1)

            H_p_q = soft_cross_entropy_loss(logits_y, y)
            KL_z_r = (torch.exp(logits_z) * logits_z).sum(dim=1).mean() + math.log(logits_z.shape[1])

            if warmup:    # KL_z_r in warmup stage can kill signal
                loss = H_p_q
            else:
                loss = H_p_q + beta * KL_z_r

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not warmup:  # logits_y contain 'num_sample' samples (B, N, 10)
                logits_y = logits_y.mean(dim=1)
                y = y.squeeze()

            acc = (logits_y.argmax(dim=1) == y.argmax(dim=1)).float().mean().item()
            # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()

            metrics = {
                'acc': acc,
                'loss': loss.item(),
                'temp': vibi.temp,
                'H(p,q)': H_p_q.item(),
                # '1-JS(p,q)': JS_p_q,
                'KL(z||r)': KL_z_r.item(),
            }

            for m, v in metrics.items():
                logs[m].append(v)

            if step % len(bb_train_loader) % 50 == 0:
                print(f'[{epoch}/{init_epoch + args.num_epochs}:{step % len(bb_train_loader):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()])
                      + (' ~ warmup' if warmup else ''))

        vibi.eval()
        valid_acc_old = valid_acc
        valid_acc = test_accuracy(vibi, bb_valid_loader, name='vibi valid top1')
        interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(bb_train_loader)).tolist()
        logs['val_acc'].extend(interpolate_valid_acc)

        if args.save_images_every_epoch:
            inspect_explanations(save_loc=images_loc.format(f'top_k_{epoch + 1}'), mode='topk')
            inspect_explanations(save_loc=images_loc.format(epoch + 1), mode='distribution')
        pretty_plot(logs, smoothing=50, save_loc=plot_loc)

        if not args.save_best or valid_acc > best_acc:
            best_acc = valid_acc

            print(f'Saving model to {model_ckpt}')
            torch.save({'model': vibi, 'optimizer': optimizer, 'epoch': epoch + 1,
                       'k': k, 'beta': beta, 'acc': best_acc, 'logs': logs}, model_ckpt)

    if args.save_best:
        state_dict = torch.load(load_ckpt, map_location=device)['model']
        print(f'Loading best model {load_ckpt}, top1 valid acc {best_acc:.3f}')

print('top k explanation')
inspect_explanations(save_loc=images_loc.format('best_top_k'), mode='topk')
print('magnified distribution')
inspect_explanations(save_loc=images_loc.format('best_distribution'), mode='distribution')


# XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX XXX
# inspect_batch = next(iter(test_loader))
# 54 plane
# 59 bird
# 32 nice
grid = torchvision.utils.make_grid(to_zero_one(inspect_batch[0]).cpu())
torchvision.utils.save_image(grid, f'results/test_batch.png')

if args.dataset == 'CIFAR10':
    for i, inspect_batch in enumerate(test_loader):
        if i != 32:
            continue

        grid = torchvision.utils.make_grid(to_zero_one(inspect_batch[0]).cpu())
        torchvision.utils.save_image(grid, f'results/test_batch_{i}.png')

        print('top k explanation')
        inspect_explanations(save_loc=images_loc.format(f'test_top_k_{i}'), mode='topk')
        print('test magnified distribution')
        inspect_explanations(save_loc=images_loc.format(f'test_distribution_{i}'), mode='distribution')
        break

# pretty_plot(logs, smoothing=50)
pretty_plot(logs, smoothing=50, save_loc=plot_loc)
