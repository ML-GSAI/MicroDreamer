from zero123 import Zero123Pipeline
from diffusers import DDIMScheduler
import torchvision.transforms.functional as TF

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from diffusers import DDIMInverseScheduler, DPMSolverMultistepScheduler

import sys
sys.path.append('./')


class Zero123(nn.Module):
    def __init__(self, device, fp16=True, t_range=[0.02, 0.98], model_key="ashawkey/stable-zero123-diffusers"):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.dtype = torch.float16 if fp16 else torch.float32

        # assert self.fp16, 'Only zero123 fp16 is supported for now.'

        self.pipe = Zero123Pipeline.from_pretrained(
            model_key,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)

        # stable-zero123 has a different camera embedding
        self.use_stable_zero123 = 'stable' in model_key

        self.pipe.image_encoder.eval()
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe.clip_camera_projection.eval()

        self.vae = self.pipe.vae
        self.unet = self.pipe.unet

        self.pipe.set_progress_bar_config(disable=True)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device)  # for convenience

        self.embeddings = None

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor in [0, 1]
        x = F.interpolate(x, (256, 256), mode='bilinear', align_corners=False)
        x_pil = [TF.to_pil_image(image) for image in x]
        x_clip = self.pipe.feature_extractor(images=x_pil, return_tensors="pt").pixel_values.to(
            device=self.device, dtype=self.dtype)
        c = self.pipe.image_encoder(x_clip).image_embeds
        v = self.encode_imgs(x.to(self.dtype)) / self.vae.config.scaling_factor
        self.embeddings = [c, v]

    def get_cam_embeddings(self, elevation, azimuth, radius, default_elevation=0):
        if self.use_stable_zero123:
            T = np.stack([np.deg2rad(elevation), np.sin(np.deg2rad(azimuth)), np.cos(
                np.deg2rad(azimuth)), np.deg2rad([90 + default_elevation] * len(elevation))], axis=-1)
        else:
            # original zero123 camera embedding
            T = np.stack([np.deg2rad(elevation), np.sin(np.deg2rad(
                azimuth)), np.cos(np.deg2rad(azimuth)), radius], axis=-1)
        T = torch.from_numpy(T).unsqueeze(1).to(
            dtype=self.dtype, device=self.device)  # [8, 1, 4]
        return T

    @torch.no_grad()
    def refine(self, pred_rgb, elevation, azimuth, radius,
               guidance_scale=2, steps=50, strength=0.8, default_elevation=0,
               ):

        batch_size = pred_rgb.shape[0]

        self.scheduler.set_timesteps(steps)

        if strength == 0:
            init_step = 0
            latents = torch.randn(
                (1, 4, 32, 32), device=self.device, dtype=self.dtype)
        else:
            init_step = int(steps * strength)
            pred_rgb_256 = F.interpolate(
                pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256.to(self.dtype))
            latents = self.scheduler.add_noise(latents, torch.randn_like(
                latents), self.scheduler.timesteps[init_step])

        T = self.get_cam_embeddings(
            elevation, azimuth, radius, default_elevation)
        cc_emb = torch.cat(
            [self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
        cc_emb = self.pipe.clip_camera_projection(cc_emb)
        cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

        vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
        vae_emb = torch.cat([vae_emb, torch.zeros_like(vae_emb)], dim=0)

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):

            x_in = torch.cat([latents] * 2)
            t_in = torch.cat(
                [t.view(1)] * 2).repeat(batch_size).to(self.device)

            noise_pred = self.unet(
                torch.cat([x_in, vae_emb], dim=1),
                t_in.to(self.unet.dtype),
                encoder_hidden_states=cc_emb,
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents)  # [1, 3, 256, 256]
        return imgs

    def train_step(self, pred_rgb, elevation, azimuth, radius, step_ratio=None, guidance_scale=3.0, as_latent=False, default_elevation=0, target_img=None, step=None, iter_steps=20, init_3d=False, inverse_ratio=0.6, ddim_eta=1.0):

        batch_size = pred_rgb.shape[0]
        if target_img is None:
            with torch.no_grad():
                if as_latent:
                    latents = F.interpolate(
                        pred_rgb, (32, 32), mode='bicubic', align_corners=False) * 2 - 1
                else:
                    pred_rgb_256 = F.interpolate(
                        pred_rgb, (256, 256), mode='bicubic', align_corners=False)
                    latents = self.encode_imgs(pred_rgb_256.to(self.dtype))

                t = torch.tensor(step, dtype=torch.long,
                                 device=self.device).unsqueeze(0)
                t_expand = t.repeat(batch_size)

                latents_noisy = latents
                inverse_scheduler = DDIMInverseScheduler.from_config(
                    self.pipe.scheduler.config)
                # inverse_scheduler = DDIMInverseScheduler(clip_sample=False)

                inverse_scheduler.set_timesteps(iter_steps)
                # scheduler = DPMSolverMultistepScheduler()
                # scheduler.config.algorithm_type = 'sde-dpmsolver++'
                # scheduler.config.solver_order = 1

                scheduler = DDIMScheduler.from_config(
                    self.pipe.scheduler.config)
                # scheduler=DDIMScheduler(clip_sample=False)

                scheduler.set_timesteps(iter_steps)

                # cond
                T = self.get_cam_embeddings(
                    elevation, azimuth, radius, default_elevation)
                cc_emb = torch.cat(
                    [self.embeddings[0].repeat(batch_size, 1, 1), T], dim=-1)
                cc_emb = self.pipe.clip_camera_projection(cc_emb)
                cc_emb = torch.cat([cc_emb, torch.zeros_like(cc_emb)], dim=0)

                vae_emb = self.embeddings[1].repeat(batch_size, 1, 1, 1)
                vae_emb = torch.cat(
                    [vae_emb, torch.zeros_like(vae_emb)], dim=0)

                @torch.no_grad()
                def pred_noise(latents, t_expand, uncond=False):

                    x_in = torch.cat([latents] * 2)
                    t_in = torch.cat([t_expand] * 2)
                    noise_pred = self.unet(
                        torch.cat([x_in, vae_emb], dim=1),
                        t_in.to(self.unet.dtype),
                        encoder_hidden_states=cc_emb,
                    ).sample
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    if uncond:
                        return noise_pred_uncond
                    noise_pred = noise_pred_uncond + guidance_scale * \
                        (noise_pred_cond - noise_pred_uncond)

                    return noise_pred

                t_stop = t[0]   # t_2 in paper
                ratio = torch.tensor(
                    (step/1000)*iter_steps*inverse_ratio, dtype=torch.long)
                t_spe = inverse_scheduler.timesteps[ratio]  # t_1 in paper
                if t_spe > 0:
                    latents_noisy = self.scheduler.add_noise(
                        latents_noisy, torch.randn_like(latents), t_spe)

                if init_3d:
                    latents_noisy = torch.randn_like(latents)
                    t_stop = 1000
                else:
                    for i, t_inv in enumerate(inverse_scheduler.timesteps[:-1]):
                        t_inv_prev = inverse_scheduler.timesteps[i+1]
                        if t_inv_prev <= t_spe:
                            continue
                        if t_inv_prev > t_stop:
                            break
                        t_inv_expand = t_inv.repeat(batch_size).to(self.device)
                        noise_pred = pred_noise(
                            latents_noisy, t_inv_expand, uncond=True)
                        latents_noisy = inverse_scheduler.step(
                            noise_pred, t_inv_prev, latents_noisy).prev_sample.clone().detach()

                for tt in scheduler.timesteps:
                    if tt > t_stop:
                        continue
                    tt_expand = tt.repeat(batch_size).to(self.device)
                    noise_pred = pred_noise(latents_noisy, tt_expand)
                    latents_noisy = scheduler.step(noise_pred, tt, latents_noisy, eta=ddim_eta).prev_sample.to(
                        latents.dtype).clone().detach()
                pred_latents = latents_noisy
                target_img = self.decode_latents(pred_latents)

        # real_target_img = F.interpolate(
        #     target_img, (pred_rgb.shape[-2], pred_rgb.shape[-1]), mode='bicubic', align_corners=False)
        # loss = F.l1_loss(pred_rgb, real_target_img.to(
        #     pred_rgb), reduction='sum')/pred_rgb.shape[0]

        return target_img

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs, mode=False):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        if mode:
            latents = posterior.mode()
        else:
            latents = posterior.sample()
        latents = latents * self.vae.config.scaling_factor

        return latents

