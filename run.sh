#!/bin/bash


python train.py --cuda --num_epochs=3 --save_best --dataset=MNIST --explainer_type=Unet --k=16 --b=0
python train.py --cuda --num_epochs=3 --save_best --dataset=MNIST --explainer_type=Unet --k=16 --b=0.1
python train.py --cuda --num_epochs=3 --save_best --dataset=MNIST --explainer_type=Unet --k=16 --b=0.01

python train.py --cuda --num_epochs=3 --save_best --dataset=MNIST --explainer_type=ResNet_2x --k=12 --b=0
python train.py --cuda --num_epochs=3 --save_best --dataset=MNIST --explainer_type=ResNet_2x --k=12 --b=0.1
python train.py --cuda --num_epochs=3 --save_best --dataset=MNIST --explainer_type=ResNet_2x --k=12 --b=0.01

python train.py --cuda --num_epochs=3 --save_best --dataset=MNIST --explainer_type=ResNet_4x --k=4 --b=0
python train.py --cuda --num_epochs=3 --save_best --dataset=MNIST --explainer_type=ResNet_4x --k=4 --b=0.1
python train.py --cuda --num_epochs=3 --save_best --dataset=MNIST --explainer_type=ResNet_4x --k=4 --b=0.01


python train.py --cuda --num_epochs=20 --save_best --dataset=CIFAR10 --explainer_type=Unet --k=64 --b=0
python train.py --cuda --num_epochs=20 --save_best --dataset=CIFAR10 --explainer_type=Unet --k=64 --b=0.01
python train.py --cuda --num_epochs=20 --save_best --dataset=CIFAR10 --explainer_type=Unet --k=64 --b=0.001

python train.py --cuda --num_epochs=20 --save_best --dataset=CIFAR10 --explainer_type=Unet --k=64 --b=0.001 --xpl_channels=3

python train.py --cuda --num_epochs=20 --save_best --dataset=CIFAR10 --explainer_type=ResNet_2x --k=32 --b=0 
python train.py --cuda --num_epochs=20 --save_best --dataset=CIFAR10 --explainer_type=ResNet_2x --k=32 --b=0.01 
python train.py --cuda --num_epochs=20 --save_best --dataset=CIFAR10 --explainer_type=ResNet_2x --k=32 --b=0.001 

python train.py --cuda --num_epochs=20 --save_best --dataset=CIFAR10 --explainer_type=ResNet_4x --k=12 --b=0
python train.py --cuda --num_epochs=20 --save_best --dataset=CIFAR10 --explainer_type=ResNet_4x --k=12 --b=0.01
python train.py --cuda --num_epochs=20 --save_best --dataset=CIFAR10 --explainer_type=ResNet_4x --k=12 --b=0.001