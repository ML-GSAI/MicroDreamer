export CUDA_VISIBLE_DEVICES=0

python main.py --config configs/image_sai.yaml input=test_data/anya_rgba.png save_path=anya_sai
python main2.py --config configs/image_sai.yaml input=test_data/anya_rgba.png save_path=anya_sai

python main.py --config configs/image_sai.yaml input=test_data/ghost_rgba.png save_path=ghost_sai
python main2.py --config configs/image_sai.yaml input=test_data/ghost_rgba.png save_path=ghost_sai

python main.py --config configs/image_sai.yaml input=test_data/astronaut_rgba.png save_path=astro_sai
python main2.py --config configs/image_sai.yaml input=test_data/astronaut_rgba.png save_path=astro_sai