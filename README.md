# DeltaGAN-Few-Shot-Image-Generation


Code for our paper *"DeltaGAN: Towards Diverse Few-shot Image Generation with Sample-Specific Delta"*.

Created by [Yan Hong](https://github.com/hy-zpg),  [Li Niu\*](https://github.com/ustcnewly), Jianfu Zhang, Liqing Zhang.

Accepted by *ECCV2022*.


## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{HongDeltaGAN,
  title={DeltaGAN: Towards Diverse Few-shot Image Generation with Sample-Specific Delta},
  author={Hong, Yan and Niu, Li and Zhang, Jianfu and Zhang, Liqing},
  booktitle={ECCV},
  year={2022}
}
```

## Introduction

Few-shot image generation aims at generating images for a new category with only a few images, which can make fast adaptation to a new category especially for those newly emerging categories or long-tail categories. Few-shot image generation can be used for data augmentation, which benefits a wide range of downstream category-aware tasks like few-shot classification.Several state-of-the-art works have yielded impressive results, but the diversity is still limited. In this work, we propose a novel Delta Generative Adversarial Network (DeltaGAN), which consists of a reconstruction subnetwork and a generation subnetwork. The reconstruction subnetwork captures intra-category transformation, \emph{i.e.}, delta, between same-category pairs. The generation subnetwork generates sample-specific delta for an input image, which is combined with this input image to generate a new image within the same category. Besides, an adversarial delta matching loss is designed to link the above two subnetworks together. Extensive experiments on six benchmark datasets demonstrate the effectiveness of our proposed method.
<div align="center">
  <img src='https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation/blob/main/figures/framework_combine.png' align="center" width=800>
</div>




## Comparison Visualization
<div align="center">
  <img src='https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation/blob/main/figures/comparison_main.png' align="center" width=800>
</div>

## More Visualization
<div align="center">
  <img src='https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation/blob/main/figures/combo_less.png' align="center" width=800>
</div>

[//]: # (More generated reuslts to view [here]&#40;https://arxiv.org/pdf/2008.01999.pdf&#41;)







## Experiments

### Hardware& Software Dependency

- **Hardware**

  a single GPU or multiple GPUs

- **Software**

  Tensorflow-gpu (version >= 1.7)

  Opencv

  scipy

- Click [here](https://github.com/bcmi/F2GAN-Few-Shot-Image-Generation/three/main/requirements.txt) to view detailed software dependency


### Datasets Preparation
* The Download links can be found [here](https://github.com/bcmi/Awesome-Few-Shot-Image-Generation#Datasets)

- **Emnist**

  Categories/Samples: 38/ 106400

  Split: 28 seen classes, 10 unseen classes


- **VGGFace**

  Categories/Samples: 2299/ 229900

  Split: 1802 seen classes, 497 unseen classes

- **Flowers**

  Categories/Samples:** 102/ 8189

  Split: 85 seen classes, 17 unseen classes

- **Animal Faces**

  Categories/Samples: 149/ 214105

  Split: 119 seen classes, 30 unseen classes

- **NABirds**

  Categories/Samples: 555/ 48527

  Split: 444 seen classes, 111 unseen classes
  
- **Foods**

  Categories/Samples: 256/ 31395

  Split: 224 seen classes, 32 unseen classes

### Baselines

#### Few-shot Image Generation

* FIGR: Few-shot Image Generation with Reptile [paper](http://proceedings.mlr.press/v84/bartunov18a/bartunov18a.pdf)  [code](https://github.com/sbos/gmn)

* Few-shot Generative Modelling with Generative Matching Networks [paper](http://proceedings.mlr.press/v84/bartunov18a/bartunov18a.pdf)  [code](https://github.com/sbos/gmn)

* DAWSON: A do- main adaptive few shot generation framework [paper](https://arxiv.org/pdf/2001.00576)  [code](https://github.com/LC1905/musegan/)

* Data Augmentation Generative Adversarial Networks [paper](https://arxiv.org/pdf/1711.04340)  [code](https://github.com/AntreasAntoniou/DAGAN)

* F2GAN: Fusing-and-Filling GAN for Few-shot Image Generation [paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413561) [code](https://github.com/bcmi/F2GAN-Few-Shot-Image-Generation)

* Matchinggan: Matching-Based Few-Shot Image Generation[paper](https://ieeexplore.ieee.org/abstract/document/9102917/) [code](https://github.com/bcmi/MatchingGAN-Few-Shot-Image-Generation)

* LoFGAN: Fusing Local Representations for Few-shot Image Generation[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Gu_LoFGAN_Fusing_Local_Representations_for_Few-Shot_Image_Generation_ICCV_2021_paper.pdf)[code](https://github.com/edward3862/LoFGAN-pytorch)

#### Few-shot Image Classification
* Matching Networks for One Shot Learning [paper](https://arxiv.org/pdf/1606.04080.pdf)  [code](https://github.com/AntreasAntoniou/MatchingNetworks)

* Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks [paper](https://arxiv.org/pdf/1703.03400.pdf)  [code](https://github.com/cbfinn/maml)

* Learning to Compare: Relation Network for Few-Shot Learning [paper](https://arxiv.org/pdf/1711.06025.pdf)  [code](https://github.com/floodsung/LearningToCompare_FSL)

* DPGN: Distribution Propagation Graph Network for Few-shot Learning [paper](https://arxiv.org/pdf/2003.14247.pdf )  [code](https://github.com/megvii-research/DPGN)

* Meta-Transfer Learning for Few-Shot Learning [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Meta-Transfer_Learning_for_Few-Shot_Learning_CVPR_2019_paper.pdf)  [code](https://github.com/y2l/meta-transfer-learning-tensorflow)

* Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation [paper](https://arxiv.org/pdf/2001.08735.pdf)  [code](https://github.com/y2l/meta-transfer-learning-tensorflow)

* Delta-encoder: an effective sample synthesis methodfor few-shot object recognitio [paper](https://proceedings.neurips.cc/paper/2018/file/1714726c817af50457d810aae9d27a2e-Paper.pdf)[code](https://github.com/EliSchwartz/DeltaEncoder)






### Getting Started

### Installation

1.Clone this repository.

```
git clone https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation.git
```

2.Create python environment for *DeltaGAN* via pip.

```
pip install -r requirements.txt
```

### Data Preprecessing
1. Dwonloading the datasets, obtaining the path of corresponding datasets 'dataroot', and setting the path for preprocessed dataset 'storepath'

2. Runing the script, obtaining list [num_categories, num_samples_each_category, image_width, image_height, image_channel]
```
python data_preparation.py --dataroot --storepath --image_width 96 --channel 3 
```
* for Emnist: --image_width 28, --channel 1
* for other datasets: --image_width 96, --channel 3

3. Setting the storepath in the data_with_matchingclassifier.py for each dataset. For example, replacing the path in class EmnistDAGANDataset with the storepath of your preprocessed Emnist data.

4. If your need run the other datasets except for our selected five datasets, you can also follow the above data preprocessing




### Training


1.Train on EMNIST dataset

```
python train_dagan_with_matchingclassifier.py --dataset emnist --image_width 28 --batch_size 20  --experiment_title MMF2GAN/emnist1way3shot   --selected_classes 1 --support_number 2  --loss_G 1 --loss_D 1 --loss_CLA 1  --loss_recons_B 1 --loss_matching_G 0.01 --loss_matching_D 1 --loss_sim 1 
```

2.Train on VGGFce dataset

```
python train_dagan_with_matchingclassifier.py --dataset vggface --image_width 96 --batch_size 20  --experiment_title MMF2GAN/vggface1way3shot   --selected_classes 1 --support_number 3  --loss_G 1 --loss_D 1 --loss_CLA 1  --loss_recons_B 1 --loss_matching_G 0.01 --loss_matching_D 1 --loss_sim 1 
```

3.Train on Flowers dataset

```
python train_dagan_with_matchingclassifier.py --dataset flowers --image_width 96 --batch_size 20  --experiment_title MMF2GAN/flowers1way3shot   --selected_classes 1 --support_number 2  --loss_G 1 --loss_D 1 --loss_CLA 1  --loss_recons_B 1 --loss_matching_G 0.01 --loss_matching_D 1 --loss_sim 1 
```


4.Train on Animal Faces dataset

```
python train_dagan_with_matchingclassifier.py --dataset animals --image_width 96 --batch_size 20 --generation_layers 4 --generator_inner_layers 2 --num_generations 32 --experiment_title animals1way2shot42layersLikeStargenV2allNoSharedOneDUniformSameDimensionLowerx1MSDCombinedDx1x2  --selected_classes 1 --support_number 2  --loss_G 0.5 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_FSL 10 --loss_recons_B 1 --loss_matching_G 10 --loss_matching_D 0.1 --loss_sim 0 --z_dim 128  
```


### Trained Model


* EMNIST: [emnist_trained model](https://pan.baidu.com/s/1OAAr7SE4rBFmqTLjP0Gv4w), extracted code: usgq.

* VGGFace: [vggface_trained model](https://pan.baidu.com/s/1kl7xBu-2paKcifgluv01KA), extracted code: ece6.

* Flowers: [flowers_trained model](https://pan.baidu.com/s/1csEXu6UT0qpj8qW5G9Y5ew), extracted code: xfei.

* Animal Faces: [animals_trained model](https://pan.baidu.com/s/1ro1XljphBYRQaXoj4P6OhQ), extracted code: fdb2.

* NAbirds: [nabirds_trained model](https://pan.baidu.com/s/1ro1XljphBYRQaXoj4P6OhQ), extracted code: fdb2.


### Evaluation from three aspects including GAN metrics, low-data classification, and few-shot classification.

#### 1. Visualizing the generated images based on trained models, the generated images are stored in the path '--experiment_title'

**EMNIST generated images**
```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 10 --dataset emnist --image_width 28  --batch_size 30  --num_generations 128 --experiment_title EVALUATION_Augmented_emnist_DeltaGAN --selected_classes 1 --support_number 3   --restore_path   ./trained_models/emnist/  --continue_from_epoch 100
```

**VGGFace generated images**

```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 10 --dataset vggface --image_width 96  --batch_size 30  --num_generations 128 --experiment_title EVALUATION_Augmented_vggface_DeltaGAN --selected_classes 1 --support_number 3   --restore_path  path  ./trained_models/vggface/  --continue_from_epoch 100
```

**Flowers generated images**

```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 10 --dataset flowers --image_width 96  --batch_size 30  --num_generations 128 --experiment_title EVALUATION_Augmented_flowers_DeltaGAN --selected_classes 1 --support_number 3   --restore_path path   ./trained_models/flowers/  --continue_from_epoch  100

```

**Animal Faces generated images**

```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 10 --dataset animals --generation_layers 4 --generator_inner_layers 2 --image_width 96 --batch_size 30  --num_generations 100 --experiment_title EVALUATION_Augmented_animals --selected_classes 1 --support_number 3 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_recons_B 1 --loss_FSL 10 --loss_sim 1 --loss_matching_D 0.1 --loss_matching_G 10 --z_dim 128  --restore_path   --continue_from_epoch 
```

**NABirds generated images**

```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 10 --dataset nabirds --generation_layers 4 --generator_inner_layers 2 --image_width 96 --batch_size 30  --num_generations 100 --experiment_title EVALUATION_Augmented_animals --selected_classes 1 --support_number 3 --loss_G 1 --loss_D 1 --loss_KL 0 --loss_CLA 1 --loss_recons_B 1 --loss_FSL 10 --loss_sim 1 --loss_matching_D 0.1 --loss_matching_G 10 --z_dim 128  --restore_path   --continue_from_epoch 
```


#### 2. Testing the GAN metrics including IS, FID, and IPIPS for generated images, which is suitable for RGB 3-channel images like VGGFace, Flowers, Animal Faces, and NABirds datasets.


**VGGFace GAN metrics**


```
python GAN_metrcis_FID_IS_LPIPS.py  --dataroot_real ./EVALUATION/Augmented/vggface/DeltaGAN/visual_outputs_realimages/ --dataroot_fake  ./EVALUATION/Augmented/vggface/DeltaGAN/visual_outputs_forquality/  --image_width 128 --image_channel 3 --augmented_support 100  --dir ./EVALUATION/Augmented/vggface/DeltaGAN/visual_outputs_forquality/ --out ./EVALUATION/Augmented/vggface/DeltaGAN/GAN_METRICS.txt 

```

**Flowers GAN metrics**

```
python GAN_metrcis_FID_IS_LPIPS.py  --dataroot_real ./EVALUATION/Augmented/flowers/DeltaGAN/visual_outputs_realimages/ --dataroot_fake  ./EVALUATION/Augmented/flowers/DeltaGAN/visual_outputs_forquality/  --image_width 128 --image_channel 3 --augmented_support 100  --dir ./EVALUATION/Augmented/flowers/DeltaGAN/visual_outputs_forquality/ --out ./EVALUATION/Augmented/flowers/DeltaGAN/GAN_METRICS.txt 

```


**Animal Faces GAN metrics**

```
python GAN_metrcis_FID_IS_LPIPS.py  --dataroot_real ./EVALUATION/Augmented/animals/DeltaGAN/visual_outputs_realimages/ --dataroot_fake  ./EVALUATION/Augmented/animals/DeltaGAN/visual_outputs_forquality/  --image_width 128 --image_channel 3 --augmented_support 100  --dir ./EVALUATION/Augmented/animals/DeltaGAN/visual_outputs_forquality/ --out ./EVALUATION/Augmented/animals/DeltaGAN/GAN_METRICS.txt 

```


**NABirds GAN metrics**

```
python GAN_metrcis_FID_IS_LPIPS.py  --dataroot_real ./EVALUATION/Augmented/nabirds/DeltaGAN/visual_outputs_realimages/ --dataroot_fake  ./EVALUATION/Augmented/nabirds/DeltaGAN/visual_outputs_forquality/  --image_width 128 --image_channel 3 --augmented_support 100  --dir ./EVALUATION/Augmented/nabirds/DeltaGAN/visual_outputs_forquality/ --out ./EVALUATION/Augmented/nabirds/DeltaGAN/GAN_METRICS.txt 

```


#### 3. Testing the classification in low-data setting with augmented images.
take EMNIST as example, low-data classification with augmented images generated from our trained model

3.1. Gnerating augmented images using three conditional images
```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 10 --dataset emnist --image_width 28  --batch_size 30  --num_generations 512 --experiment_title EVALUATION_Augmented_emnist_DeltaGAN --selected_classes 1 --support_number 3   --restore_path path ./trained_models/emnist/ --continue_from_epoch 100
```

3.2. Preparing generated images: the generated images are stored in the 'storepath/visual_outputs_forclassifier' and setting the storepath for preprocessed data, running below script

```
python data_preparation.py --dataroot storepath/visual_outputs_forclassifier  --storepath --image_width 28 --channel 1 
```
3.3. Replacing the datapath in data_with_matchingclassifier_for_quality_and_classifier.py with the storepath for preprocessed data.

3.4. Running the script for low-data classification.

```
train_classifier_with_augmented_images.py --dataset emnist  --selected_classes testing_categories  --batch_size 16 --classification_total_epoch 50  --experiment_title AugmentedLowdataClassifier_emnist  --image_width 28  --image_height 28 --image_channel 1
```
--selected_classes: the number of total testing categories


#### 4. Testing the classification in few-shot setting with augmented images.
take EMNIST as example, NwayKshot classification with augmented images generated from our trained model

4.1. Gnerating augmented images using Kshot conditional images
```
python test_dagan_with_matchingclassifier_for_generation.py  --is_training 0 --is_all_test_categories 1 --is_generation_for_classifier 1  --general_classification_samples 10 --dataset emnist --image_width 28  --batch_size 30  --num_generations 128 --experiment_title EVALUATION_Augmented_emnist_emnist --selected_classes 1 --support_number K   --restore_path path ./trained_models/emnist/ --continue_from_epoch 100
```
setting the '--support_number' as K.

4.2. Preprocessing the generated images
```
python data_preparation.py --dataroot ./EVALUATION/Augmented/emnist/DeltaGAN/visual_outputs_forclassifier  --storepath ./EVALUATION/Augmented/emnist/DeltaGAN/  --image_width 28 --channel 1 
```

4.3. Replacing the datapath in data_with_matchingclassifier_for_quality_and_classifier.py with ./EVALUATION/Augmented/emnist/DeltaGAN/emnist.npy.

4.4. Running the script for few-shot classification.

```
train_classifier_with_augmented_images.py --dataset emnist  --selected_classes N  --batch_size 16 --classification_total_epoch 50  --experiment_title AugmentedFewshotClassifier_emnist  --image_width 28  --image_height 28 --image_channel 1
```
setting the '--selected_classes' as N.





### Results

To view more clear results, please click the belowing tables.

#### GAN metrics of Generated Images
<div align="center">
  <img src='https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation/blob/main/figures/table_1.png' align="center" width=800>
</div>





#### Few-shot Image Classification

<div align="center">
  <img src='https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation/blob/main/figures/table_2.png' align="center" width=800>
</div>





## Acknowledgement

Some of the codes are built upon [DAGAN](https://github.com/AntreasAntoniou/DAGAN). Thanks them for their great work!

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request!

*DeltaGAN* is freely available for non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.



