# Official repo for GenSAM
**GenSAM: Exploring Text Prompts without Visual Prompts in Segmentation**

[Jian Hu*](https://lwpyh.github.io/),, [Jiayi Lin*](https://scholar.google.com/citations?hl=en&view_op=list_works&gmla=AH70aAW7LfVn82AZckFTh_Y7mXPPLrqDH6LMWDXLBCTbnSLe39ue9Iiza6jy5HDuReAozF5HnWECuq9xlCXrlw&user=l4Fps4EAAAAJ), [Weitong Cai](https://lvgd.github.io/), [Shaogang Gong](http://www.eecs.qmul.ac.uk/~sgg/)

[[Arxiv]](https://arxiv.org/abs/) 
[[Project Page]](https://lwpyh.github.io/GenSAM/)


## Abstract

Camouflaged object detection (COD) approaches heavily rely on pixel-level annotated datasets. Weakly-supervised COD (WSCOD) approaches use sparse annotations like scribbles or points to reduce annotation effort, but this can lead to decreased accuracy. The Segment Anything Model (SAM) shows remarkable segmentation ability with sparse prompts like points. However, manual prompt is not always feasible, as it may not be accessible in real-world application. Additionally, it only provides localization information instead of semantic one, which can intrinsically cause ambiguity in interpreting the targets.In this work, we aim to eliminate the need for manual prompt.The key idea is to employ Cross-modal Chains of Thought Prompting (CCTP) to reason visual prompts using the semantic information given by a generic text prompt.To that end, we introduce a test-time adaptation per-instance mechanism called Generalizable SAM (GenSAM) to automatically enerate and optimize visual prompts the generic task prompt for WSCOD.In particular, CCTP maps a single generic text prompt onto image-specific consensus foreground and background heatmaps using vision-language models, acquiring reliable visual prompts. Moreover, to test-time adapt the visual prompts, we further propose Progressive Mask Generation (PMG) to iteratively reweight the input image, guiding the model to focus on the targets in a coarse-to-fine manner.Crucially, all network parameters are fixed, avoiding the need for additional training.Experiments demonstrate the superiority of GenSAM. Experiments on three benchmarks demonstrate that GenSAM outperforms point supervision approaches and achieves comparable results to scribble supervision ones, solely relying on general task descriptions as prompts.     


<img src='AIG_framework_v2.png'>

## Pipeline 
<!-- The prompt-dialogue of varies abilities are saved in [dataset](https://github.com/crystraldo/StableLLAVA/tree/main/dataset). -->

<!-- The synthesized prompt-dialogue datasets of various abilities are saved in [dataset](https://github.com/crystraldo/StableLLAVA/tree/main/dataset). Please follow the steps below to generate datasets with LLaVA format. -->

<!-- 1. Use [SD-XL](https://github.com/crystraldo/StableLLAVA/blob/main/stable_diffusion.py) to generate images as training images. It will take ~13s to generate one image on V100.-->
<!-- python stable_diffusion.py --prompt_path dataset/animal.json --save_path train_set/animal/-->
<!-- 2. Use [data_to_llava](https://github.com/crystraldo/StableLLAVA/blob/main/data_to_llava.py) to convert dataset format for LLaVA model training. -->
<!-- ```
python data_to_llava.py --image_path train_set/ --prompt_path dataset/ --save_path train_ano/
``` -->

 ## TO-DO LIST
- [ ] Update datasets and implementation scripts
- [ ] Keep incorporating more capabilities
- [ ] Demo and Codes
