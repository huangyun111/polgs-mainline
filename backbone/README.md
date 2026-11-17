<h2 align="center">PolGS++: Physically-Based Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction</h2>
<h4 align="center">
    <a href="https://yu-fei-han.github.io/homepage/"><strong>Yufei Han</strong></a>
    ·<a href="https://fourson.github.io/"><strong>Chu Zhou</strong></a>
    ·<a href="https://youweilyu.github.io/"><strong>Youwei Lyu</strong></a>
    ·<strong>Qi Chen</strong>
    ·<a href="https://teacher.bupt.edu.cn/lisi/zh_CN/index.htm"><strong>Si Li</strong></a>
    ·
    <a href="https://camera.pku.edu.cn/"><strong>Boxin Shi</strong></a>
    ·
    <a href="https://sdmda.bupt.edu.cn/info/1061/1060.htm"><strong>Yunpeng Jia</strong></a>
    ·<a href="https://gh-home.github.io/"><strong>Heng Guo</strong></a>
    ·<a href="https://zhanyuma.cn/"><strong>Zhanyu Ma</strong></a>
</h3>
<h4 align="center">preprint </h3>
<p align="center">
  <br>
    <a href="https://arxiv.org/abs/2509.19726">
      <img src='https://img.shields.io/badge/arXiv-Paper-981E32?style=for-the-badge&Color=B31B1B' alt='arXiv PDF'>
    </a>
</p>
<div align="center">
</div>



## Environment Setup

This project was tested on Ubuntu 22.04.3 with CUDA 11.8 and Python 3.7.13. The reconstruction process for a single object takes approximately 10 minutes on an RTX 4090 GPU.

### 1. Clone the Repository
```shell
git clone https://github.com/PRIS-CV/PolGS_plus.git
cd PolGS_plus
```

### 2. Create and Activate Conda Environment
```shell
conda create -n polgs++ python=3.7.13 -y
conda activate polgs++
```

### 3. Install Dependencies
Install PyTorch with CUDA support:
```shell
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

Install other requirements:
```shell
pip install -r requirements.txt
```

Install custom submodules:
```shell
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/cubemapencoder
```

### 4. Install PyTorch3D
Please install PyTorch3D following the [official installation guide](https://github.com/facebookresearch/pytorch3d.git).


## Data Preparation

We evaluate our method on subsets of RMVP3D and SMVP3D from [NeRSP](https://github.com/PRIS-CV/NeRSP), [PANDORA](https://github.com/akshatdave/pandora), and [PISR](https://github.com/GCChen97/PISR) datasets. We also provide the preprocessed data for some of the datasets, which can be downloaded from [here](https://drive.google.com/file/d/119LOKGS0Y3QuSbo-Y7u-oNqX8xukAYxF/view?usp=drive_link).

### Dataset Configuration
- **SMVP3D**: 36 views (12×3 configuration) for training and testing
- **RMVP3D**: 31 views (half of the original 61 views) for training and testing  
- **PANDORA**: All available views for training and testing
- **PISR**: 20 views (half of the original 40 views) for training and testing
### Directory Structure
The data should be organized as follows:
```
data/
├── PANDORA/
│   ├── owl/
│   │   ├── image/                     # RGB images
│   │   ├── train_images_stokes/       # Stokes images
│   │   │   ├── 01_s0.hdr             
│   │   │   ├── 01_s0p1.hdr             
│   │   │   └── ...
│   │   ├── input_azimuth_maps/  # Mask files
│   │   └── cameras.npz               # Camera parameters
│   └── ...
├── RMVP3D/
│   ├── frog/
│   │   ├── image/                     # RGB images
│   │   ├── s0/                       # Stokes S0 images
│   │   ├── s1/                       # Stokes S1 images  
│   │   ├── s2/                       # Stokes S2 images
│   │   ├── input_azimuth_maps/  # Mask files
│   │   └── cameras.npz               # Camera parameters
│   └── ...
├── SMVP3D/
│   ├── snail/
│   │   └── ...
│   └── ...
└── PISR/
    ├── StandingRabbit/
    ├── LyingRabbit/
    └── ...
```

### Using Your Own Data
To test on your own polarimetric data:
1. Organize your data following the structure above
2. Create mask files for object segmentation 


## Training

### Prerequisites
Before training, ensure you have:
1. Prepared your data following the format described in the [Data Preparation](#data-preparation) section
2. Updated the data paths in `train.bash` to point to your dataset location

### Running Training
To train a scene, execute:
```shell
bash train.bash
```

## Acknowledgements

We gratefully acknowledge the following open-source projects that contributed to this work:

- [Gaussian Surfels](https://github.com/turandai/gaussian_surfels) - For foundational Gaussian surfels implementation
- [3DGS-DR](https://github.com/gapszju/3DGS-DR) - For 3D Gaussian Splatting with deferred rendering


## Bibtex
```
@inproceedings{han2025polgs,
  title={PolGS: Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction},
  author={Han, Yufei and Tie, Bowen and Guo, Heng and Lyu, Youwei and Li, Si and Shi, Boxin and Jia, Yunpeng and Ma, Zhanyu},
  year = {2025},
  booktitle = ICCV,
}
@misc{han2026polgs++,
  title={PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction}, 
  author={Yufei Han and Chu Zhou and Youwei Lyu and Qi Chen and Si Li and Boxin Shi and Yunpeng Jia and Heng Guo and Zhanyu Ma},
  year={2026},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2603.10801}, 
}
```