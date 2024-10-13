# MicroDreamer
Official implementation of *[MicroDreamer: Zero-shot 3D Generation in ~20 Seconds by Score-based Iterative Reconstruction](http://arxiv.org/abs/2404.19525)*.




https://github.com/user-attachments/assets/0a99424a-2e7a-47f0-9f0a-b6713b7686b5



## News
[10/2024] Add a new mesh export method from [LGM](https://github.com/3DTopia/LGM)


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
# train 20 iters and export ckpt & coarse_mesh to logs
python main.py --config configs/image_sai.yaml input=test_data/name_rgba.png save_path=name_rgba

### training mesh stage
# auto load coarse_mesh and refine 3 iters, export fine_mesh to logs
python main2.py --config configs/image_sai.yaml input=test_data/name_rgba.png save_path=name_rgba
```

Image+Text-to-3D (ImageDream):

```bash
### training gaussian stage
python main.py --config configs/imagedream.yaml input=test_data/ghost_rgba.png prompt="a ghost eating hamburger" save_path=ghost_rgba
```

Calculate for CLIP similarity:
```bash
PYTHONPATH='.' python scripts/cal_sim.py
```

## More Results



https://github.com/user-attachments/assets/8888a353-df16-4e19-ac1b-7ee37ece7ed1




https://github.com/user-attachments/assets/7e52a87b-d1f6-4e7b-a6b4-7732ea69613c





## Acknowledgement

This work is built on many amazing open source projects, thanks to all the authors!

- [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
- [LGM](https://github.com/3DTopia/LGM)
- [threestudio](https://github.com/threestudio-project/threestudio)


## BibTeX

```
@misc{chen2024microdreamerzeroshot3dgeneration,
      title={MicroDreamer: Zero-shot 3D Generation in $\sim$20 Seconds by Score-based Iterative Reconstruction}, 
      author={Luxi Chen and Zhengyi Wang and Zihan Zhou and Tingting Gao and Hang Su and Jun Zhu and Chongxuan Li},
      year={2024},
      eprint={2404.19525},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.19525}, 
}
```
