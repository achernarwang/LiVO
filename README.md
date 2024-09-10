# Embedding an Ethical Mind:<br/>Aligning Text-to-Image Synthesis via Lightweight Value Optimization

The official implementation of ACM Multimedia 2024 accepted paper "Embedding an Ethical Mind: Aligning Text-to-Image Synthesis via Lightweight Value Optimization"

<div align=center>
<img src="assets/framework.png" width="100%"/>  
  
Illustration of LiVO.

</div>

## Abstract

Recent advancements in diffusion models trained on large-scale data have enabled the generation of indistinguishable human-level images, yet they often produce harmful content misaligned with human values, e.g., social bias, and offensive content. Despite extensive research on Large Language Models (LLMs), the challenge of Text-to-Image (T2I) model alignment remains largely unexplored. Addressing this problem, we propose LiVO (Lightweight Value Optimization), a novel lightweight method for aligning T2I models with human values. LiVO only optimizes a plug-and-play value encoder to integrate a specified value principle with the input prompt, allowing the control of generated images over both semantics and values. Specifically, we design a diffusion model-tailored preference optimization loss, which theoretically approximates the Bradley-Terry model used in LLM alignment but provides a more flexible trade-off between image quality and value conformity. To optimize the value encoder, we also develop a framework to automatically construct a text-image preference dataset of 86k (prompt, aligned image, violating image, value principle) samples. Without updating most model parameters and through adaptive value selection from the input prompt, LiVO significantly reduces harmful outputs and achieves faster convergence, surpassing several strong baselines and taking an initial step towards ethically aligned T2I models.

## Installation

Firstly, clone this repository to your local environment:

```shell
git clone https://github.com/achernarwang/LiVO.git
```

Then create a virtual python 3.10 environment using conda:

```shell
conda create -n livo python=3.10 -y
```

Finally, installing necessary dependencies in the created python environment:

```shell
conda activate livo
conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install accelerate datasets transformers diffusers -c conda-forge -y
pip install torchmetrics[image] openai tiktoken
```

## Inference

### Value Encoder

To use the value encoder, you could refer the example script below (more examples are provided at [value_encoder/inference_example.py](value_encoder/inference_example.py)):

```python
import torch
from transformers import CLIPTextModel
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

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
image_original.save("example_orig.png")

image_detoxified = pipeline(prompt_embeds=concat_embeds, num_inference_steps=25, generator=torch.Generator(device).manual_seed(seed)).images[0]
image_detoxified.save("example_deto.png")
```

To access the pretrained weights of the value encoder, you could also directly visit [this link](https://huggingface.co/achernarwang/LiVO).

### Value Retriever

Please check [value_retriever/retriever.py](value_retriever/retriever.py) for the implementation and example usage of the value retriever used in our work. You could also run the file by the following steps:

```shell
export OPENAI_API_KEY="<your_api_key>"
cd value_retriever
python retriever.py
```

## Training the Value Encoder

## Evaluation

## TODO

- [ ] Value encoder implementation
  - [x] inference
  - [ ] training
- [x] Value retriever implementation
- [x] Checkpoints for value encoder
- [ ] Evaluation code
- [ ] How-to guide
- [ ] Datasets (maybe, as we need to assess whether they are appropriate for public release)

We expect to complete the open-sourcing of this work before its official publication at ACM MM 2024, which is October 28, 2024.
  
## Acknowledgements

- Thanks to the [🤗 Diffusers](https://github.com/huggingface/diffusers) Library and everyone who ever contributed to it! We built our work upon this great open-source project.
- Thanks to the authors of [Fair Diffusion](https://github.com/ml-research/Fair-Diffusion), [Concept Ablation](https://github.com/nupurkmr9/concept-ablation), [Unified Concept Editing](https://github.com/rohitgandikota/unified-concept-editing) for their amazing research and the open source of their implementations! We use their implemetations as baseline models in our paper.
