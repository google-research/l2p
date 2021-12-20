# [Learning to Prompt for Continual Learning (L2P)](https://arxiv.org/pdf/2112.08654.pdf) Official Jax Implementation


## Introduction
![L2P illustration](method.jpg)

L2P is a novel continual learning technique which learns to dynamically prompt a pre-trained model to learn tasks sequentially under different task transitions. Different from mainstream rehearsal-based or architecture-based methods, L2P requires neither a rehearsal buffer nor test-time task identity. L2P can be generalized to various continual learning settings including the most challenging and realistic task-agnostic setting. L2P consistently outperforms prior state-of-the-art methods. Surprisingly, L2P achieves competitive results against rehearsal-based methods even without a rehearsal buffer.

Code is written by Zifeng Wang. Acknowledgement to https://github.com/google-research/nested-transformer.

This is not an officially supported Google product.

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
We use tensorboard to visualize the result. For example, if the working directory specified to run L2P is `workdir=./cifar100_l2p`, the command to check result is as follows:

```
tensorboard --logdir ./cifar100_l2p
```
Here are the important metrics to keep track of, and their corresponding meanings:

| Metric    | Description |
| ----------- | ----------- |
| accuracy_n      | Accuracy of the n-th task       |
| forgetting   | Average forgetting up until the current task       |
| avg_acc  | Average evaluation accuracy up until the current task      |



## Cite
```
@inproceedings{wang2021learning,
  title={Learning to Prompt for Continual Learning},
  author={Zifeng Wang and Zizhao Zhang and Chen-Yu Lee and Han Zhang and Ruoxi Sun and Xiaoqi Ren and Guolong Su and Vincent Perot and Jennifer Dy and Tomas Pfister},
  booktitle={arXiv preprint arXiv:2112.08654},
  year={2021}
}
```
