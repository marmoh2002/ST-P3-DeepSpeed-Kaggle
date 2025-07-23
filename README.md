# ST-P3

![pipeline](imgs/pipeline.png)

> **ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning**  
> Shengchao Hu, [Li Chen](https://scholar.google.com/citations?hl=en&user=ulZxvY0AAAAJ), Penghao Wu, [Hongyang Li](https://lihongyang.info/), [Junchi Yan](https://thinklab.sjtu.edu.cn/), Dacheng Tao.       
> - [arXiv Paper](https://arxiv.org/abs/2207.07601), ECCV 2022
> - Our [Blog](https://zhuanlan.zhihu.com/p/544387122) (in Chinese) 

## Introduction
This reposity is a fork of [the official PyTorch Lightning implementation for **ST-P3**.](https://github.com/OpenDriveLab/ST-P3)

This repo aims to optimize the training script of the original repo in order for it so be trainable on Kaggle's 2 x T4 GPU.
We make use of Lightning Deepspeed.
Since Kaggle resources are too small to train the entire model at once and or fit the entire trainval dataset at once, we work with the partitioned dataset (10 blobs each containing 85 scenes).
We also resorted to training perception module only, then freezing encoder weights and training the prediction & planning modules.
We are limited to a batch size of 1 and to using efficientnet-b0 rather than efficientnet-b4 due to memory limitations.

## Get Started
### Kaggle Notebook
```
[Kaggle Notebook: st-p3-deepspeed](https://www.kaggle.com/code/mariaamm/st-p3-deepspeed)
```
### Kaggle Datasets
You can find the Nuscenes TrainVal Dataset uploaded as Kaggle Datasets for easier use:

[nuscenes trainval metadata](https://www.kaggle.com/datasets/mariaamm/nuscenes-trainval-metadata)

[nuscenes trainval Blob 1](https://www.kaggle.com/datasets/mariaamm/nuscenes-blob1)

[nuscenes trainval Blob 2](https://www.kaggle.com/datasets/mariaamm/nuscenes-blob2-1746978131)

[nuscenes trainval Blob 3](https://www.kaggle.com/datasets/mariaamm/nuscenes-blob3-1746975231)

[nuscenes trainval Blob 4](https://www.kaggle.com/datasets/mariaamm/nuscenes-blob4-1746976536)

[nuscenes trainval Blob 5](https://www.kaggle.com/datasets/mariaamm/nuscenes-blob5-1746977412)

[nuscenes trainval Blob 6](https://www.kaggle.com/datasets/mariaamm/nuscenes-blob6-1746979118)

[nuscenes trainval Blob 7](https://www.kaggle.com/datasets/mariaamm/nuscenes-blob8-1746984595)

[nuscenes trainval Blob 8](https://www.kaggle.com/datasets/mariaamm/nuscenes-blob8-1746982718)

[nuscenes trainval Blob 9](https://www.kaggle.com/datasets/mariaamm/nuscenes-blob9-1746980948)

[nuscenes trainval Blob 10](https://www.kaggle.com/datasets/mariaamm/nuscenes-blob10-1746998516)


### Training

```
# perception module pretrain
bash scripts/train_perceive.sh ${configs} ${dataroot}

# prediction module training purpose, no need for e2e training
bash scripts/train_prediction.sh ${configs} ${dataroot} ${pretrained}

```
### Evaluation
To evaluate the model on nuScenes:
- Download the [nuScenes](https://www.nuscenes.org/download) dataset.
- Download the pretrained weights.

```
bash scripts/eval_plan.sh ${checkpoint} ${dataroot}
```
### Pre-trained models
- open-loop planning on nuScenes: [model](https://drive.google.com/file/d/1fPAzrgohTVeFfyXSUh5wUHB_US8v9HFa/view?usp=sharing).

## Citation

If you find their repo or our paper useful, please use the following citation:

```
@inproceedings{hu2022stp3,
 title={ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning}, 
 author={Shengchao Hu and Li Chen and Penghao Wu and Hongyang Li and Junchi Yan and Dacheng Tao},
 booktitle={European Conference on Computer Vision (ECCV)},
 year={2022}
}
```

## License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgement
We thank Xiangwei Geng for his support on the depth map generation, and fruitful discussions from [Xiaosong Jia](https://jiaxiaosong1002.github.io/). We have many thanks to [FIERY](https://github.com/wayveai/fiery) team for their exellent open source project.
