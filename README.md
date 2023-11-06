# EXIM: A Hybrid Explicit-Implicit Representation for Text-Guided 3D Shape Generation
## SIGGRAPH Asia 2023 & ACM TOG
## [Project Page](https://liuzhengzhe.github.io/EXIM.github.io/)
Code for the paper [EXIM: A Hybrid Explicit-Implicit Representation for Text-Guided 3D Shape Generation](https://arxiv.org/pdf/2311.01714.pdf).


**Authors**: Zhengzhe Liu, Jingyu Hu, Ka-Hei Hui, Xiaojuan Qi, Daniel Cohen-Or, Chi-Wing Fu

<img src="figure1.png" width="900"/>


## Installation

```
conda env create -f environment.yaml
conda activate EXIM
cd stage 2
python setup.py build_ext --inplace
cd data_processing/libmesh/
python setup.py build_ext --inplace
cd ../libvoxelize/
python setup.py build_ext --inplace
cd ../..
```

## Data Preparation


* Download our [models](https://drive.google.com/drive/folders/1JD4LFgEN9i2a9eeUU74TeuKIIvX9ozRZ)

unzip to "EXIM/data/"



##  3D Shape Generation


```
cd stage1
python test_chair.py
cd ../stage2
sh test.sh
```

<!---
* Table generation:
cd stage1
python test_table.py
edit test:sh: -checkpoint ../data/model/table/checkpoint_epoch_200.tar
stage2/models/data/voxelized_data_shapenet_test.py: uncomment Line 133
stage2/generation_iterator.py: uncomment Line 28
--->

##  Training

(1)  Stage 1

Download the [train data](https://drive.google.com/drive/folders/1JD4LFgEN9i2a9eeUU74TeuKIIvX9ozRZ).

Put to "EXIM/data/"

```
cd stage1
python trainer_new.py
```

(2)  Stage 2

* Download [Choy et. al. rendering data](https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/ShapeNet.zip) and [IF-Net data](https://drive.google.com/drive/folders/1QGhDW335L7ra31uw5U-0V7hB-viA0JXr): 03001627.tar.gz

Put to "EXIM/data/"

```
cd stage2
sh train.sh
```


## Evaluation

```
cd evaluation
```

##  Manipulation

Option1: Locate the interested region following [Diff-Edit](https://arxiv.org/abs/2210.11427).

```
cd manipulation

python mani_diffedit.py
```

Option2: Locate the interested region using Interactive System (Thanks to Ruihui Li, Ka-Hei Hui, and Jingyu Hu for developing this tool).


First, run the UI-Interface

```
python sample_point_cloud.py
cd UI-Interface
python label_interface.py
```

Load a input.obj.ply
Select a region you want to edit
The "selection.npy" file is saved in "UI-Interface/debug/"
Move the selection.npy to the "manipulation folder"

```
python mani_select.py

```

To train the manipulation model:

```
python seg.py
python trainer_new.py
```


## Acknowledgement

The code is built upon [Wavelet-Diffusion](https://github.com/edward1997104/Wavelet-Generation) and [DVR](https://github.com/autonomousvision/differentiable_volumetric_rendering)

## Contact
If you have any questions or suggestions about this repo, please feel free to contact me (liuzhengzhelzz@gmail.com).
