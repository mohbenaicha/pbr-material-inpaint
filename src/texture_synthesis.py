from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
)
from utils import *
import numpy as np
import os, cv2, torch
from PIL import Image
import argparse
import time


class TextureSynthesis:
    def __init__(self):
        (
            self.generator,
            self.vae,
            self.controlnet,
            self.pipe_controlnet,
            self.controlnet_conditioning_scale,
        ) = self.setup_components()

    def setup_components(self) -> tuple:
        # VAE setup
        generate_random_seed()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        generator = torch.Generator(device="cuda")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        ).to(device)

        # ControlNet setup
        controlnet_canny = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
        ).to(device)
        controlnet_conditioning_scale = 0.99

        # Inpainting Albedo Pipeline Setup
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet_canny,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )

        pipe.load_lora_weights(
            "./model_weights",
            weight_name="texture-synthesis-topdown-base-condensed.safetensors",
        )

        pipe.controlnet.to(memory_format=torch.channels_last)
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()

        generator = torch.Generator(device=device)  # Nice to have

        scale_and_apply_lora_weights_inplace(pipe, 9.7)

        for _, module in pipe.unet.named_modules():
            module.register_forward_hook(detect_nan_hook)

        return (
            generator,
            vae,
            controlnet_canny,
            pipe,
            controlnet_conditioning_scale,
        )

    def save_color_and_generate_guide_image(self, albedo_image: Image) -> Image:
        # Rescale and save the inpainted albedo image
        # Image -> np.array -> rescale -> Image
        albedo_image = rescale(albedo_image)
        albedo_image = Image.fromarray((albedo_image * 255).astype(np.uint8))
        albedo_image.save(str(os.path.join("output", "albedo_inpainted.png")))

        # Generate Canny edge map of the albedo for guiding other textures
        albedo_canny_image = cv2.Canny(
            (np.array(albedo_image) * 255).astype(np.uint8), 20, 70
        )
        albedo_canny_image = albedo_canny_image[:, :, None]
        albedo_canny_image = np.concatenate(
            [albedo_canny_image, albedo_canny_image, albedo_canny_image], axis=2
        )
        guide_image = Image.fromarray(albedo_canny_image)
        return guide_image

    def generate_images(
        self,
        sample_steps=11,
        guidance_scale=8,
        maps={"albedo"},
        user_image: Image = None,  # 1024x1024
        user_mask: Image = None,  # 1024x1024
        prompt="",
    ) -> dict:
        images = {}
        prompts = update_user_prompt(prompt)
        if not os.path.exists("output"):
            os.makedirs("output")
        print("Processing maps: ", maps, "\n\n")

        original_img = user_image.convert("RGB").resize((1024, 1024))
        mask = user_mask.convert("L").resize((1024, 1024))
        canny_mask = mask_to_canny(mask)  # grey scale mask
        canny_mask.save(str(os.path.join("output", "canny_mask.png")))
        time.sleep(1)

        base_img = self.pipe_controlnet(
            prompt=prompts["albedo"],
            negative_prompt=negativePrompts["albedo"],
            controlnet_conditioning_scale=0.8,
            image=canny_mask,
            num_inference_steps=sample_steps,
            generator=self.generator,
            guidance_scale=guidance_scale-1,
        ).images[0]


        print("Generated albedo texture using ControlNet Inpaint.")

        # --- Convert Albedo to Canny Edges ---
        image_canny_pil = color_to_canny(base_img)
        image_canny_pil.save(str(os.path.join("output", "canny_albedo.png")))
        
        color_image = self.pipe_controlnet(
            prompt=prompts["albedo"],
            negative_prompt=negativePrompts["albedo"],
            controlnet_conditioning_scale=0.99,
            image=image_canny_pil,
            num_inference_steps=sample_steps,
            generator=self.generator,
            guidance_scale=guidance_scale,
        ).images[0]
        
        
        base_img = Image.composite(
            color_image, original_img, mask.convert("L")
        )
        base_img.save(os.path.join("output", "albedo" + ".png"))

        images["albedo"] = base_img
        
        maps.discard("albedo") # no ned to process it again

        # Generate the other maps using the inpainted albedo and Canny edge map as guides
        for map_ in maps:
            print(f"Generating {map_} map.\n")
            print(f"Prompt: {prompts[map_]}\n")

            generated_map = self.pipe_controlnet(
                prompt=prompts[map_],
                negative_prompt=negativePrompts[map_],
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                image=image_canny_pil,
                num_inference_steps=sample_steps,
                generator=self.generator,
                guidance_scale=guidance_scale,
            )
            
            generated_map.images[0].save(str(os.path.join("output", map_ + ".png")))
            images[map_] = generated_map.images[0]
        return images

if __name__ == "__main__":
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Texture Synthesis using ControlNet")
        parser.add_argument("--user_image", type=str, required=True, help="Path to the user image")
        parser.add_argument("--user_mask", type=str, required=True, help="Path to the user mask")
        parser.add_argument("--prompt", type=str, required=True, help="Text prompt for texture synthesis")
        return parser.parse_args()

    args = parse_args()
    user_image_path = args.user_image
    user_mask_path = args.user_mask
    prompt = args.prompt

    user_image = Image.open(user_image_path).convert("RGB").resize((1024, 1024))
    user_mask = Image.open(user_mask_path).convert("L").resize((1024, 1024))
    texture_synthesis = TextureSynthesis()
    texture_synthesis.generate_images(
        sample_steps=11,
        guidance_scale=8,
        maps={"albedo", "normal", "rough", "ambientocl", "metal", "specular", "height"},
        user_image=user_image,
        user_mask=user_mask,
        prompt=prompt,
    )
