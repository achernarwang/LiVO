import torch
import random
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

### Example 1: Detoxification
def example_1():
    prompt = "A portrait of a blood-soaked warrior, highly detailed and ultra-realistic."
    value = "bloody content is considered inappropriate"

    # Since RunwayML has taken down the model weights from huggingface, we use a mirror version instead.
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    device = "cuda:0"
    seed = 1234

    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    value_encoder = CLIPTextModel.from_pretrained("adstellaria/LiVO", subfolder="value_encoder").to(device)

    input_ids = pipeline.tokenizer(prompt, max_length=pipeline.tokenizer.model_max_length-1, truncation=True, return_tensors="pt").input_ids.to(device)
    prompt_embeds = pipeline.text_encoder(input_ids)[0]

    value_input_ids = pipeline.tokenizer(value + ', ' + prompt, truncation=True, return_tensors="pt").input_ids.to(device)
    value_embeds = value_encoder(value_input_ids)[1]

    concat_embeds = torch.cat([value_embeds.unsqueeze(1), prompt_embeds], dim=1)

    image_original = pipeline(prompt_embeds=prompt_embeds, num_inference_steps=25, generator=torch.Generator(device).manual_seed(seed)).images[0]
    image_original.save("example_1_orig.png")

    image_detoxified = pipeline(prompt_embeds=concat_embeds, num_inference_steps=25, generator=torch.Generator(device).manual_seed(seed)).images[0]
    image_detoxified.save("example_1_deto.png")

### Example 2: Debias
def example_2():
    prompt = "A photo of the face of an English teacher."
    value = "racial equality"

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    device = "cuda:1"

    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    value_encoder = CLIPTextModel.from_pretrained("adstellaria/LiVO", subfolder="value_encoder").to(device)

    input_ids = pipeline.tokenizer(prompt, max_length=pipeline.tokenizer.model_max_length-1, truncation=True, return_tensors="pt").input_ids.to(device)
    prompt_embeds = pipeline.text_encoder(input_ids)[0]

    value_input_ids = pipeline.tokenizer(value + ', ' + prompt, truncation=True, return_tensors="pt").input_ids.to(device)
    value_embeds = value_encoder(value_input_ids)[1]

    concat_embeds = torch.cat([value_embeds.unsqueeze(1), prompt_embeds], dim=1)

    # randomly drop value embedding by a probability of 0.5 to acquire a balanced distribution of protected attributes, eg. gender, race, etc.
    p = random.random()
    used_embeds = concat_embeds if p < 0.5 else prompt_embeds

    image = pipeline(prompt_embeds=used_embeds, num_inference_steps=25).images[0]
    image.save("example_2.png")


if __name__ == "__main__":
    example_1()
    example_2()
