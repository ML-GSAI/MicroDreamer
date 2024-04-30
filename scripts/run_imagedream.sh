export CUDA_VISIBLE_DEVICES=0

# imagedream with no refinement

python main.py --config configs/imagedream.yaml input=test_data/ghost_rgba.png prompt="a ghost eating hamburger" save_path=ghost

