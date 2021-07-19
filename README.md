# VIBI Experiments
This is a re-implementation of VIBI ([arxiv](https://arxiv.org/abs/1902.06918), [github](https://github.com/SeojinBang/VIBI))
including experiments for MNIST and CIFAR10 written in Python and PyTorch.

To run the experiments, first clone this repository and install requirements.
```
git clone https://github.com/willisk/VIBI
cd VIBI
pip install -r requirements.txt
```

Run all experiments shown in [results](results):
```
chmod +x run_experiments.sh
./run_experiments.sh
```

Otherwise run the script with passed arguments.
```
python train.py

optional arguments
  --dataset {MNIST,CIFAR10}
  --cuda                Enable cuda.
  --num_epochs NUM_EPOCHS
                        Number of training epochs for VIBI.
  --explainer_type {Unet,ResNet_2x,ResNet_4x,ResNet_8x}
  --xpl_channels {1,3}
  --k K                 Number of chunks.
  --beta BETA           beta in objective J = I(y,t) - beta * I(x,t).
  --num_samples NUM_SAMPLES
                        Number of samples used for estimating expectation over p(t|x).
  --resume_training     Recommence training vibi from last saved checkpoint.
  --save_best           Save only the best models (measured in valid accuracy).
  --save_images_every_epoch
                        Save explanation images every epoch.
  --jump_start          Use pretrained model with beta=0 as starting point.
```

# VIBI Overview
The goal is to create interpretable explanations for black-box models.
This is achieved by two neural network, the explainer and the approximator.
The explainer network produces a probability distribution over the input chunks, 
given an input image. A relaxed k-hot vector is sampled from this distribution.
This k-hot vector is used to create a masked input, which is then 
fed into the approximator network.
The approximator network aims to match the probability distribution of the
black-box model output.
The whole idea builds heavily on [L2X (Learning to explain)](https://arxiv.org/pdf/1802.07814.pdf).
The only difference is that VIBI's additional term effectively increases the entropy of the distribution `p(z)`,
whereas L2X only optimizes for minimizing the cross-entropy `H(p,q)` between the black-box model's predictions and the approximator.

# MNIST Example Results

| Test Batch      | Explanation Distribution | Top-k Explanation  |
| :-------------: | :----------------------: | :----------------: |
| ![MNIST_test_batch](https://user-images.githubusercontent.com/38631399/126166117-1d9235b3-04f8-4b9e-9dde-ccfb64f42942.png) | ![MNIST_best_distribution](https://user-images.githubusercontent.com/38631399/126187923-c3e56d3c-706a-4f56-864f-a15cbee1dc68.png) | ![MNIST_best_top_k](https://user-images.githubusercontent.com/38631399/126164103-d996b14f-2be2-49be-bcca-7943a2bdfc44.png) |

Using `explainer_model=Resnet4x`, `k=4`, `beta=0.01`.


# CIFAR10 Example Results

| Test Batch      | Explanation Distribution | Top-k Explanation  |
| :-------------: | :----------------------: | :----------------: |
| ![CIFAR10_test_batch_32](https://user-images.githubusercontent.com/38631399/126159047-139bb5b1-eef1-4826-a653-de87e1c6b12c.png) | ![CIFAR10_test_distribution_32](https://user-images.githubusercontent.com/38631399/126161587-3aa94d12-8db2-4bad-9f3b-107534924a43.png) | ![CIFAR10_test_top_k_32](https://user-images.githubusercontent.com/38631399/126161648-e2069c71-ed8d-4234-aba4-7bc2ae7bbcfe.png) |

Using `explainer_model=Unet`, `k=64`, `beta=0.001`.


Green boxes indicate that the black-box model's prediction is correct, red boxes indicate incorrect predictions.
The strength (calculated using `1 - JS(p,q)`) of the outlining color gives feedback on how well the approximator's prediction (using top-k) fits the black-box model's output.

