# Fast AutoAugment Implementation in Pytorch

Unofficial and Partial implementation of <a href="https://arxiv.org/abs/1905.00397">Fast AutoAugment</a> in Pytorch.



## Summary

- Fast AutoAugment (hereafter FAA) finds the optimal set of data augmentation operations via density matching using Bayesian optimization.
- FAA delivers comparable performance to <a href="https://arxiv.org/abs/1805.09501">AutoAugment</a> but in a much shorter period of time.
- Unlike AutoAugment that discretizes the search space, FAA can handle continuous search space directly.

<br>



## Getting Started

```bash
$ git clone https://github.com/junkwhinger/fastautoaugment_jsh.git
cd fastautoaugment_jsh
```



### Install dependencies

```bash
pip install -r requirements.txt
```



### Training

You can train or test the model with the baseline or optimal augmentation policies found by FAA with the following commands.

#### Test Only

```bash
# Baseline
python train.py --model_dir experiments/baseline --eval_only

# Fast AutoAugment
python train.py --model_dir experiments/fastautoaugment --eval_only
```



#### Training + Evaluation

```bash
# Baseline
python train.py --model_dir experiments/baseline

# Fast AutoAugment
python train.py --model_dir experiments/fastautoaugment
```



### Fast AutoAugment

You can run Fast AutoAugment with the following commands. It takes time.

- train_mode: train models on $D_M$s for 5 splits (takes roughly 4.5 hours)
- bayesian_mode: run bayesian optimiazation with HyperOpt to find the optimal policy (takes 3 hours)
- merge: aggregates the trials and combines the best policies from the splits. Writes the result as a file `optimal_policy.json`. To use the policy for training, please copy this file into your `experiments/fastautoaugment` folder.

```bash
# Train models on D_Ms & Bayesian Optimization & Merge
python search_fastautoaugment.py --train_mode --bayesian_mode

# Bayesian Optimization & Merge
python search_fastautoaugment.py --bayesian_mode

# Merge only
python search_fastautoaugment.py
```



### Checkpoints

Here are the checkpoints I made during the replication of the paper.

- for training and testing (baseline / fastautoaugment)
  - `experiments/baseline/best_model.torch`: a trained model for Baseline at epoch 200 
  - `experiments/baseline/params.json`: a hyper-parameter set for Baseline
  - `experiments/baseline/train.log`: a training log for Baseline 
- for FAA policy searching
  - `fastautoaugment/k0_t0_trials.pkl`: a pickled trial log for 0th split and 0th search width
  - `fastautoaugment/model_k_0.torch`: a model file that trained on D_M[0]
  - `fastautoaugment/optimal_policy.json`: an optimal policy json file from the search
  - `fastautoaugment/params.json`: a hyper-parameter set for FAA
  - `fastautoaugment/train.log`: a training log for FAA 

<br>



## Search Strategy

### Terminology

- A set of operations <img src="https://latex.codecogs.com/gif.latex?\mathbb{O}"/>
- Operation <img src="https://latex.codecogs.com/gif.latex?O"/>: an augmentation function (e.g. Cutout)
  - Probability <img src="https://latex.codecogs.com/gif.latex?p"/>: (attribute of an operation) the chance that the operation is turned on. This value ranges from 0 to 1, 0 being always off, 1 always on.
  - Magnitude <img src="https://latex.codecogs.com/gif.latex?\lambda"/>: (attribute of an operation) the amount that the operation transforms a given image. This value ranges from 0 to 1, and gets adjusted according to the corresponding range of its operation. For example, <img src="https://latex.codecogs.com/gif.latex?\lambda=0"/> for Rotate means Rotate -30 degree.
- Sub-policy <img src="https://latex.codecogs.com/gif.latex?\tau"/>: a random sequence of operations. The length of a sub-policy is determined by Search Width(<img src="https://latex.codecogs.com/gif.latex?T=2"/>). For example, a sub-policy that has Cutout and Rotate transforms a given image in 4 ways. 
- A set of sub-policies <img src="https://latex.codecogs.com/gif.latex?S"/>
- Policy <img src="https://latex.codecogs.com/gif.latex?\mathcal{T}"/>: a collection of sub-policies. FAA aims to find <img src="https://latex.codecogs.com/gif.latex?\mathcal{T}_*"/> that contains <img src="https://latex.codecogs.com/gif.latex?\mathcal{T}_*^{k}"/> from <img src="https://latex.codecogs.com/gif.latex?k"/>th split of the train dataset.
- <img src="https://latex.codecogs.com/gif.latex?N_{\tau}"/>: the number of consecutive operations in a sub policy
- <img src="https://latex.codecogs.com/gif.latex?N_{\mathcal{T}}"/>: the number of sub-policies in a policy
- <img src="https://latex.codecogs.com/gif.latex?K"/>: the number of folds used for stratified shuffling
- <img src="https://latex.codecogs.com/gif.latex?T"/>: search width
- <img src="https://latex.codecogs.com/gif.latex?B"/>: search depth
- <img src="https://latex.codecogs.com/gif.latex?N"/>: the number of top policies to select in each evaluation round

### Search Space

- FAA attempts to find the probability <img src="https://latex.codecogs.com/gif.latex?p"/> and magnitude <img src="https://latex.codecogs.com/gif.latex?\lambda"/> for the following 16 augmentation operations.
  -  ShearX, ShearY, TranslateX, TranslateY, 
    Rotate, AutoContrast, Invert, Equalize, 
    Solarize, Posterize, Contrast, Color, 
    Brightness, Sharpness, Cutout, Sample Pairing

### Algorithm

- Inputs
  - <img src="https://latex.codecogs.com/gif.latex?\theta"/>: network to train
  - <img src="https://latex.codecogs.com/gif.latex?D_{train}"/>: train dataset that contains 42675 images from cifar10.
  - <img src="https://latex.codecogs.com/gif.latex?K"/>: the number of cross validation folds. <img src="https://latex.codecogs.com/gif.latex?K=5"/> in FAA.
  - <img src="https://latex.codecogs.com/gif.latex?T"/>: search width. <img src="https://latex.codecogs.com/gif.latex?T=2"/> in FAA.
  - <img src="https://latex.codecogs.com/gif.latex?B"/>: search depth. <img src="https://latex.codecogs.com/gif.latex?B=200"/> in FAA.
  - <img src="https://latex.codecogs.com/gif.latex?N"/>: the number of top policies to keep. <img src="https://latex.codecogs.com/gif.latex?N=10"/> in FAA. 
- Step 1: Shuffle
  - Split <img src="https://latex.codecogs.com/gif.latex?D_{train}"/> into <img src="https://latex.codecogs.com/gif.latex?K"/> sets of <img src="https://latex.codecogs.com/gif.latex?D_M"/> and <img src="https://latex.codecogs.com/gif.latex?D_A"/> using the target labels.
- Step 2: Train
  - Train <img src="https://latex.codecogs.com/gif.latex?\theta"/> on each <img src="https://latex.codecogs.com/gif.latex?D_M"/>. FAA implemented Step 2 in parallel. In my implementation, it is done sequentially in a for loop.
    - Each model is trained from scratch without data augmentation.
    - I added `TF.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))`. 
- Step 3:  Explore-and-Exploit
  - Find the optimal set of sub-policies and probabilities and magnitudes of their operations. 
    FAA employs <a href="[http://hyperopt.github.io/hyperopt/](http://hyperopt.github.io/hyperopt/)">HyperOpt</a> for this step. I saved the sub-policies and their corresponding validation error on augmented <img src="https://latex.codecogs.com/gif.latex?D_A"/> in `Trials` for Step 4.
- Step 4. Merge
  - Select top <img src="https://latex.codecogs.com/gif.latex?N"/> policies for each <img src="https://latex.codecogs.com/gif.latex?K"/> split and <img src="https://latex.codecogs.com/gif.latex?T"/>. Combined the top policies into the final set policies that are used for re-training <img src="https://latex.codecogs.com/gif.latex?\theta"/> on <img src="https://latex.codecogs.com/gif.latex?D_{train}"/>



<br>

## Results

### CIFAR-10 Test Error (%)

Search: 7.5 GPU Hours on a single Tesla V100 16GB Memory machine
(FAA in paper took 3.5 GPU Hours)

| Model(CIFAR-10)  | Baseline(paper) | Baseline(mine) | FAA(paper/direct) | FAA(mine/direct) |
| ---------------- | --------------- | -------------- | ----------------- | ---------------- |
| Wide-ResNet-40-2 | 5.3             | 5.6            | 3.7               | 5.1              |

### Evaluation & Interpretation

- Replicating Baseline performance

  - Failed to replicate the Baseline performance of the paper despite the same hyper-parameter set I <i>tried</i> to follow.
  - During debugging the original code, I found some discrepancies regarding the dataset size that could have caused the issue (covered in-depth in ETC).
    - Revision needed on `train.py` and `model/data_loader.py`.

- Replicating Fast AutoAugment performance

  - [original] Failed to replicate Fast AutoAugment performance. The improvement on Test Error that I gained via FAA (-0.1) is much smaller than the paper's result(-1.6).
  - [revision_1] Achieved a partial improvement via FAA (-0.5) which is nearly one third of the paper's result.
    - I presume some hidden mistakes had impaired the test error.
    - Revision needed on `search_fastautoaugment.py`.

- The optimal policies  of revision_1 do show a tendency to keep the original image as the policies found in the first attempt.

  - Brightness, Contrast, Sharpness, Rotate, TranslateX, TranslateY magnitudes are around 0.5 that rarely modifies the original image.
  - Equalize and Invert functions are given near zero probabilities.
  
- However, the reported tendency is not as extreme as the previous optimal policies, leaving some rooms for potential useful augmentations.

- Plus, Cutout, Color, ShearY and Solarize found interesting probabilities and magnitudes compared to the previous attempt.

  ![trials_visualized](assets/trials_visualized_revision_1.png)

<br>



## Update on revision_1 

- Fixed a serious implementation mistake with the bayesian optimization part.
  - The original implementation passes a sub_policy to the function `bayesian_optimization`. 
    <img src="https://latex.codecogs.com/gif.latex?BayesOptim(\tau, \mathcal{L}(\theta | \tau(D_A)), B)"/>
  - The revision_1 implementation passes a list of augmentations to the function, and let it compose a policy that has 5 sub-policies.
    <img src="https://latex.codecogs.com/gif.latex?BayesOptim(\mathcal{T}, \mathcal{L}(\theta | \mathcal{T}(D_A)), B)"/>
- Simplified the use of HyperOpt and several helper functions.
- Added `SamplePairing` to `augmentation.py` (not used in revision_1 but might do in the future revisions).
- revision_1 took one day.



## Discrepencies between Paper and my Implementation

- I did not include SamplePairing from the set of augmentation operations to optimize.
- I did not use GradualWarmupScheduler for training <img src="https://latex.codecogs.com/gif.latex?\theta"/> on <img src="https://latex.codecogs.com/gif.latex?D_M"/>. 
  (I did for training Baseline and FAA final model) 
- I did not use parallel or distributed training using ray or horovod. 

<br>

## ETC

- Testing: FAA official implementation `python train.py -c confs/wresnet40x2_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar10`
  - It runs validation steps with the same 16 images every 10th epoch (AutoAugment set 7,325 images aside for validation).
  - The images used in the validation phase are augmented with the optimal policies, unlike my previous expectation that we do NOT augment the validation dataset for a normal training loop.
  - The image batches loaded from validloader are as follows:
    - ![validloader](assets/validloader.png)
  
- On FAA paper, Algorithm 1 decribed on page 5 can be somewhat misleading.
  - ![excerpt_1](assets/excerpt_1.png)
  - For the number of search width <img src="https://latex.codecogs.com/gif.latex?T"/>, we select top <img src="https://latex.codecogs.com/gif.latex?N"/> policies in <img src="https://latex.codecogs.com/gif.latex?B"/>. Hence with <img src="https://latex.codecogs.com/gif.latex?T=2"/> and <img src="https://latex.codecogs.com/gif.latex?N=10"/>, we end up with 20(2x10) top policies each split. However, on page 6, the paper says "Select the top N best policies for each split". Either one of these explanations should be corrected.
  
- The use of augmentation for training <img src="https://latex.codecogs.com/gif.latex?\theta"/> with <img src="https://latex.codecogs.com/gif.latex?D_M"/>

  - The first paragraph on page 5

    > Next, we train model parameter <img src="https://latex.codecogs.com/gif.latex?\theta"/> on <img src="https://latex.codecogs.com/gif.latex?D_M"/> from scratch without data augmentation.

  - The last paragraph on page 5

    > … (1) training model parameters on K-fold train data with default augmentation rules ...

  - I interpreted this part as using `ToTensor()` and `Normalize()`. Defining the default augmentation rules would help enhance clarity. 

- Typo

  - The last paragraph on page 8 (a a)

    > Moreover, the joint optimization of NAS and Fast AutoAugment is a a curious area in AutoML.

<br>

## Contact for Issues

- Junsik Hwang, junsik.whang@gmail.com



<br>

## References & Opensources

1. Fast AutoAugment
   - Paper: https://arxiv.org/abs/1905.00397
   - Codes: https://github.com/kakaobrain/fast-autoaugment
   - GradualWarmupScheduler: https://github.com/ildoonet/pytorch-gradual-warmup-lr
2. AutoAugment
   - Paper: https://arxiv.org/abs/1805.09501
   - Codes:https://github.com/tensorflow/models/tree/master/research/autoaugment
3. Wide Residual Network
   - Paper: https://arxiv.org/pdf/1605.07146v2.pdf
   - Codes: https://github.com/meliketoy/wide-resnet.pytorch
4. HyperOpt
   - Official Documentation: [http://hyperopt.github.io/hyperopt/](http://hyperopt.github.io/hyperopt/)
   - Tutorials: https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce
5. FloydHub (Cloud GPU)
   - Website: [http://floydhub.com/](http://floydhub.com/)