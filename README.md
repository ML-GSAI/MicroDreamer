# MicroDreamer
Official implementation of *[MicroDreamer: Zero-shot 3D Generation in ~20 Seconds by Score-based Iterative Reconstruction](http://arxiv.org/abs/2404.19525)*.



https://github.com/ML-GSAI/MicroDreamer/assets/91880347/8b5661f5-3100-4da7-b1c4-7fd7f69aba4e



## Installation

The codebase is built on [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian). For installation, 
```bash
conda create -n MicroDreamer python=3.11
conda activate MicroDreamer

pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# The commit hash we used
# d986da0d4cf2dfeb43b9a379b6e9fa0a7f3f7eea

# simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# The version we used
# pip install git+https://github.com/NVlabs/nvdiffrast/@0.3.1

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit/

# The version we used
# pip install git+https://github.com/ashawkey/kiuikit/@0.2.3

# To use ImageDream, also install:
pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream

# The commit hash we used
# 26c3972e586f0c8d2f6c6b297aa9d792d06abebb
```

## Usage

Image-to-3D:

```bash
### preprocess
# background removal and recentering, save rgba at 256x256
python process.py test_data/name.jpg

# save at a larger resolution
python process.py test_data/name.jpg --size 512

# process all jpg images under a dir
python process.py test_data

### training gaussian stage
# train 30 iters and export ckpt & coarse_mesh to logs
python main.py --config configs/image_sai.yaml input=test_data/name_rgba.png save_path=name

### training mesh stage
# auto load coarse_mesh and refine 3 iters, export fine_mesh to logs
python main2.py --config configs/image_sai.yaml input=test_data/name_rgba.png save_path=name
```

Image+Text-to-3D (ImageDream):

```bash
### training gaussian stage
python main.py --config configs/imagedream.yaml input=test_data/ghost_rgba.png prompt="a ghost eating hamburger" save_path=ghost
```

## More Results

https://github.com/ML-GSAI/MicroDreamer/assets/91880347/422db3e0-a4b5-44d0-9135-787ebc2c5c99

## Acknowledgement

This work is built on many amazing open source projects, thanks to all the authors!

- [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
- [threestudio](https://github.com/threestudio-project/threestudio)


## BibTeX

```
@article{chen2024microdreamer,
  title={MicroDreamer: Zero-shot 3D Generation in $$\backslash$sim $20 Seconds by Score-based Iterative Reconstruction},
  author={Chen, Luxi and Wang, Zhengyi and Li, Chongxuan and Gao, Tingting and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2404.19525},
  year={2024}
}
```
