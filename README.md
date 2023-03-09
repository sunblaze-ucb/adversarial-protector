# Adversarial Training

Code base for the paper [Perturbation Type Categorization for Multiple Adversarial Perturbation
Robustness](https://proceedings.mlr.press/v180/maini22a/maini22a.pdf)

## Training a Model

```
CUDA_VISIBLE_DEVICES=0 python pc_train.py --config configs/CIFAR10_pipeline.json 
```

Additional Parameters:

> *model_id*: Unique id for the model
*opt_type*: SGD or Adam
*fft*: 0 or 1 (To use fourier transform or not)
*epochs*: Number of epochs to train
*num_iter*: Number of iterations for the attack
*model_type*: Type of model to train
*batch_size*: Batch size
*lr_max*: Maximum learning rate
*lr_mode*: 1 for linear, 2 for cosine
*droprate*: Dropout rate
*attacked_model_list*: List of models to attack
*attack_types*: List of attack types


## Evaluating a Model
```
python test.py --config configs/MNIST_small_step.json --num_iter 200 --model_type cnn_msd --path models/m_cnn/Baselines/max --mode base --restarts 2 --attack pgd --batch_size 500  --attack_types linf l1 l2 ddn
```

### How can I Cite this work?

```
@inproceedings{
maini2022perturbation,
title={Perturbation Type Categorization for Multiple Adversarial Perturbation Robustness},
author={Pratyush Maini and Xinyun Chen and Bo Li and Dawn Song},
booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
year={2022},
url={https://openreview.net/forum?id=BlbhyDUo9xc}
}
```