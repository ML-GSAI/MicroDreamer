# MicroDreamer
Official implementation of "MicroDreamer: Zero-shot 3D Generation in ~20 Seconds by Score-based Iterative Reconstruction".


https://github.com/ML-GSAI/MicroDreamer/assets/91880347/6109542c-4b13-4afb-ae10-87e646efe187


## Installation

The codebase is built on [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian). For installation, 
```bash
conda create -n MicroDreamer python=3.11
conda activate MicroDreamer

pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit

# To use ImageDream, also install:
pip install git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream
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

## Acknowledgement

This work is built on many amazing open source projects, thanks to all the authors!

- [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
- [threestudio](https://github.com/threestudio-project/threestudio)


## BibTeX

