# [Learning to Prompt for Continual Learning (L2P)](tbd) Official Jax Implementation


## Introduction
![L2P illustration](method.jpg)

L2P is a novel continual learning technique which learns to dynamically prompt a pre-trained model to learn tasks sequentially under different task transitions. Different from mainstream rehearsal-based or architecture-based methods, L2P requires neither a rehearsal buffer nor test-time task identity. L2P can be generalized to various continual learning settings including the most challenging and realistic task-agnostic setting. L2P consistently outperforms prior state-of-the-art methods. Surprisingly, L2P achieves competitive results against rehearsal-based methods even without a rehearsal buffer.

Code is written by Zifeng Wang. Acknowledgement to https://github.com/google-research/nested-transformer.

## Enviroment setup
```
pip install -r requirements.txt
```

## Getting pretrained ViT model
ViT-B/16 model used in this paper can be downloaded at [here](gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz).


## Instructions on running L2P
We provide the configuration file to train and evaluate L2P on multiple benchmarks in `configs`.


To run our method on the Split CIFAR-100 dataset (class-incremental setting):

```
python -m main.py --my_config configs/cifar100_l2p.py --workdir=./cifar100_l2p --my_config.init_checkpoint=<ViT-saved-path/ViT-B_16.npz>
```

To run our method on the more complex Gaussian Scheduled CIFAR-100 dataset (task-agnostic setting):

```
python -m main.py --my_config configs/cifar100_gaussian_l2p.py --workdir=./cifar100_gaussian_l2p --my_config.init_checkpoint=<ViT-saved-path/ViT-B_16.npz>
```

Note: we run our experiments using 8 V100 GPUs or 4 TPUs, and we specify a per device batch size of 16 in the config files. This indicates that we use a total batch size of 128.


## Visualize results
TODO(zifengw): Add instructions and metrics explanation.
```
```

## Cite
TODO(zifengw): Add bib.

