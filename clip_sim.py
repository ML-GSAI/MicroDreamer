import cv2
import torch
import numpy as np
from torchvision import transforms as T
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from typing import Literal
from PIL import Image
import kiui


class CLIP:
    def __init__(self, device, model_name='openai/clip-vit-large-patch14'):

        self.device = device

        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode_image(self, image):
        # image: PIL, np.ndarray uint8 [H, W, 3]

        pixel_values = self.processor(
            images=image, return_tensors="pt").pixel_values.to(self.device)
        image_features = self.clip_model.get_image_features(
            pixel_values=pixel_values)

        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)  # normalize features

        return image_features

    def encode_text(self, text):
        # text: str

        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(
            self.device)
        text_features = self.clip_model.get_text_features(**inputs)

        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)  # normalize features

        return text_features


def read_image(
    path: str, 
    mode: Literal["float", "uint8", "pil", "torch", "tensor"] = "float", 
    order: Literal["RGB", "RGBA", "BGR", "BGRA"] = "RGB",
):

    if mode == "pil":
        return Image.open(path).convert(order)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # cvtColor
    if len(img.shape) == 3: # ignore if gray scale
        if order in ["RGB", "RGBA"]:
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            elif img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # mix background
        if img.shape[-1] == 4 and 'A' not in order:
            img = img.astype(np.float32) / 255
            img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:])

    # mode
    if mode == "uint8":
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        return img
    elif mode == "float":
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255
        return img
    elif mode in ["tensor", "torch"]:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255
        return torch.from_numpy(img)
    else:
        raise ValueError(f"Unknown read_image mode {mode}")


clip = CLIP('cuda', model_name='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

def cal_clip_sim(ref_path, novel_path_ls):
    
    ref_img = read_image(ref_path, mode='float')
    if ref_img.shape[-1] == 4:
        # rgba to white-bg rgb
        ref_img = ref_img[..., :3] * ref_img[..., 3:] + (1 - ref_img[..., 3:])
    ref_img = (ref_img * 255).astype(np.uint8)
    with torch.no_grad():
        ref_features = clip.encode_image(ref_img)

    results = []
    for novel_path in novel_path_ls:
        novel_img = read_image(novel_path, mode='float')
        if novel_img.shape[-1] == 4:
            # rgba to white-bg rgb
            novel_img = novel_img[..., :3] * novel_img[..., 3:] + (1 - novel_img[..., 3:])
        novel_img = (novel_img * 255).astype(np.uint8)
        with torch.no_grad():
            novel_features = clip.encode_image(novel_img)

        sim = (ref_features * novel_features).sum(dim=-1).mean().item()
        results.append(sim)

    avg_similarity = np.mean(results)
    return avg_similarity


def cal_clip_sim_text(ref_text, novel_path_ls):
    with torch.no_grad():
        ref_features = clip.encode_text(ref_text)

    results = []
    for novel_path in novel_path_ls:
        novel_img = read_image(novel_path, mode='float')
        if novel_img.shape[-1] == 4:
            # rgba to white-bg rgb
            novel_img = novel_img[..., :3] * novel_img[..., 3:] + (1 - novel_img[..., 3:])
        novel_img = (novel_img * 255).astype(np.uint8)
        with torch.no_grad():
            novel_features = clip.encode_image(novel_img)

        sim = (ref_features * novel_features).sum(dim=-1).mean().item()
        results.append(sim)

    avg_similarity = np.mean(results)
    return avg_similarity
