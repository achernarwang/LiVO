import torch
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class Method:
    name = "method"
    def generate(self, prompts, num_images_per_prompt, **kwargs):
        pass

class StableDiffusion(Method):
    def __init__(self, model_id="stable-diffusion-v1-5/stable-diffusion-v1-5", device="cuda", torch_dtype=torch.float16, fast_scheduler=True) -> None:
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None)
        self.pipeline = self.pipeline.to(device)
        self.num_inference_steps = 50
        if fast_scheduler:
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
            self.num_inference_steps = 25
        self.pipeline.set_progress_bar_config(disable=True)
    
    def generate(self, prompts, num_images_per_prompt, **kwargs):
        return self.pipeline(prompts, num_images_per_prompt=num_images_per_prompt, num_inference_steps=self.num_inference_steps).images
    

class LiVO(Method):
    def __init__(self, model_path, device, torch_dtype=torch.float16, fast_scheduler=True):
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None).to(device)
        self.num_inference_steps = 50
        if fast_scheduler:
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
            self.num_inference_steps = 25
        self.value_encoder = CLIPTextModel.from_pretrained(model_path).to(device)
        self.device = device
        self.pipeline.set_progress_bar_config(disable=True)
    
    def generate(self, prompts, num_images_per_prompt, value, use_value, **kwargs):
        assert isinstance(prompts, str)
        if use_value:
            input_ids = self.pipeline.tokenizer(prompts, max_length=self.pipeline.tokenizer.model_max_length-1, truncation=True, return_tensors="pt").input_ids.to(self.device)
            prompt_embeds = self.pipeline.text_encoder(input_ids)[0]

            value_input_ids = self.pipeline.tokenizer(value + ', ' + prompts, truncation=True, return_tensors="pt").input_ids.to(self.device)
            value_embeds = self.value_encoder(value_input_ids)[1]

            concat_embeds = torch.cat([value_embeds.unsqueeze(1), prompt_embeds], dim=1)

            images = self.pipeline(prompt_embeds=concat_embeds, num_images_per_prompt=num_images_per_prompt, num_inference_steps=self.num_inference_steps).images
        else:
            images = self.pipeline(prompts, num_images_per_prompt=num_images_per_prompt, num_inference_steps=self.num_inference_steps).images
        return images