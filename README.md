# Embedding an Ethical Mind:<br/>Aligning Text-to-Image Synthesis via Lightweight Value Optimization

The official implementation of ACM Multimedia 2024 accepted paper "Embedding an Ethical Mind: Aligning Text-to-Image Synthesis via Lightweight Value Optimization"

<div align=center>
<img src="assets/framework.png" width="100%"/>  
  
Illustration of LiVO.

</div>

## Abstract

Recent advancements in diffusion models trained on large-scale data have enabled the generation of indistinguishable human-level images, yet they often produce harmful content misaligned with human values, e.g., social bias, and offensive content. Despite extensive research on Large Language Models (LLMs), the challenge of Text-to-Image (T2I) model alignment remains largely unexplored. Addressing this problem, we propose LiVO (Lightweight Value Optimization), a novel lightweight method for aligning T2I models with human values. LiVO only optimizes a plug-and-play value encoder to integrate a specified value principle with the input prompt, allowing the control of generated images over both semantics and values. Specifically, we design a diffusion model-tailored preference optimization loss, which theoretically approximates the Bradley-Terry model used in LLM alignment but provides a more flexible trade-off between image quality and value conformity. To optimize the value encoder, we also develop a framework to automatically construct a text-image preference dataset of 86k (prompt, aligned image, violating image, value principle) samples. Without updating most model parameters and through adaptive value selection from the input prompt, LiVO significantly reduces harmful outputs and achieves faster convergence, surpassing several strong baselines and taking an initial step towards ethically aligned T2I models.

## TODO

- [ ] Value encoder implementation
- [ ] Value retriever implementation
- [ ] Checkpoints for value encoder
- [ ] Evaluation code
- [ ] How-to guide
- [ ] Datasets (maybe, as we need to assess whether they are appropriate for public release)

We expect to complete the open-sourcing of this work before its official publication at ACM MM 2024, which is October 28, 2024.
  
## Acknowledgements

- Thanks to the [🤗 Diffusers](https://github.com/huggingface/diffusers) Library and everyone who ever contributed to it! We built our work upon this great open-source project.
- Thanks to the authors of [Fair Diffusion](https://github.com/ml-research/Fair-Diffusion), [Concept Ablation](https://github.com/nupurkmr9/concept-ablation), [Unified Concept Editing](https://github.com/rohitgandikota/unified-concept-editing) for their amazing research and the open source of their implementations! We use their implemetations as baseline models in our paper.
