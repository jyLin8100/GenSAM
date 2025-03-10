# :fire: GenSAM (AAAI 2024)

Code release of paper:

[**Relax Image-Specific Prompt Requirement in SAM: A Single Generic Prompt for Segmenting Camouflaged Objects**](https://arxiv.org/abs/2312.07374)

[Jian Hu*](https://lwpyh.github.io/), [Jiayi Lin*](https://jylin8100.github.io/), [Weitong Cai](https://lvgd.github.io/), [Shaogang Gong](http://www.eecs.qmul.ac.uk/~sgg/)

<a href='https://arxiv.org/abs/2312.07374'><img src='https://img.shields.io/badge/ArXiv-2312.07374-red' /></a> 
<a href='https://lwpyh.github.io/GenSAM/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='#demo'><img src='https://img.shields.io/badge/Replicate-Demo-violet'></a>

## :rocket: News
* **[2025.03.04]** We have integrated the Progressive Task Control module into our GenSAM++ method. You can try the updated demo_v1.ipynb to explore its functionality.
* **[2024.08.27]** Our new paper [Leveraging Hallucinations to Reduce Manual Prompt Dependency in Promptable Segmentation](https://lwpyh.github.io/ProMaC/) (NeurIPS24) is released, it extends the manual-free segmentation setting to more fields, and acheives SOTA performance.
* **[2023.12.25]** [Demo](#demo) of GenSAM is released.
* **[2023.12.12]** Model running instructions with LLaVA1 and LLaVA1.5 are released.
* **[2023.12.10]** LLaVA1 and LLaVA1.5 version GenSAM on CHAMELEON dataset is released.
<p align="center">
  <img src="demo_show.gif" width="100%" />
</p>
<p align="center">
  <img src="framework_GenSAM.gif" width="100%" />
</p>

<img src='supp_cod.png'>

## :bulb: Highlight

The Segment Anything Model (SAM) shows remarkable segmentation ability with sparse prompts like points. However, manual prompt is not always feasible, as it may not be accessible in real-world application. In this work, we aim to eliminate the need for manual prompt.The key idea is to employ Cross-modal Chains of Thought Prompting (CCTP) to reason visual prompts using the semantic information given by a generic text prompt. We introduce a test-time adaptation per-instance mechanism called Generalizable SAM (GenSAM) to automatically enerate and optimize visual prompts the generic task prompt.

A brief introduction of how we GenSAM do!
<img src='AIG_framework_v2.png'>
CCTP maps a single generic text prompt onto image-specific consensus foreground and background heatmaps using vision-language models, acquiring reliable visual prompts. Moreover, to test-time adapt the visual prompts, we further propose Progressive Mask Generation (PMG) to iteratively reweight the input image, guiding the model to focus on the targets in a coarse-to-fine manner.Crucially, all network parameters are fixed, avoiding the need for additional training.Experiments demonstrate the superiority of GenSAM. Experiments on three benchmarks demonstrate that GenSAM outperforms point supervision approaches and achieves comparable results to scribble supervision ones, solely relying on general task descriptions as prompts.     

## Quick Start
<!-- The prompt-dialogue of varies abilities are saved in [dataset](https://github.com/crystraldo/StableLLAVA/tree/main/dataset). -->

<!-- The synthesized prompt-dialogue datasets of various abilities are saved in [dataset](https://github.com/crystraldo/StableLLAVA/tree/main/dataset). Please follow the steps below to generate datasets with LLaVA format. -->

<!-- 1. Use [SD-XL](https://github.com/crystraldo/StableLLAVA/blob/main/stable_diffusion.py) to generate images as training images. It will take ~13s to generate one image on V100.-->
<!-- python stable_diffusion.py --prompt_path dataset/animal.json --save_path train_set/animal/-->
<!-- 2. Use [data_to_llava](https://github.com/crystraldo/StableLLAVA/blob/main/data_to_llava.py) to convert dataset format for LLaVA model training. -->
<!-- ```
python data_to_llava.py --image_path train_set/ --prompt_path dataset/ --save_path train_ano/
``` -->

### Download Dataset
1. Download the datasets from the follow links:
   
**Camouflaged Object Detection Dataset**
- **[COD10K](https://github.com/DengPingFan/SINet/)**
- **[CAMO](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)**
- **[CHAMELEON](https://www.polsl.pl/rau6/datasets/)**
2. Put it in ./data/.
### Running GenSAM on CHAMELON Dataset with LLaVA1/LLaVA1.5
1. When playing with LLaVA, this code was implemented with Python 3.8 and PyTorch 2.1.0. We recommend creating [virtualenv](https://virtualenv.pypa.io/) environment and installing all the dependencies, as follows:
```bash
# create virtual environment
virtualenv GenSAM_LLaVA
source GenSAM_LLaVA/bin/activate
# prepare LLaVA
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
cd ..
# prepare SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
pip install opencv-python imageio ftfy urllib3==1.26.6
```
2. Our GenSAM is a training-free test-time adaptation approach, so you can play with it by running:
```bash
python main.py --config config/CHAMELEON_LLaVA1.5.yaml   ###LLaVA1.5
python main.py --config config/CHAMELEON_LLaVA.yaml   ###LLaVA
```
if you want to visualize the output picture during test-time adaptation, you can running:
```bash
python main.py --config config/CHAMELEON_LLaVA1.5.yaml --visualization    ###LLaVA1.5
python main.py --config config/CHAMELEON_LLaVA.yaml --visualization    ###LLaVA
```
 ## Demo
 We further prepare a [jupyter notebook demo](https://github.com/jyLin8100/GenSAM/blob/main/demo_v1.ipynb) for visualization.
 1. Complete the following steps in the shell before opening the jupyter notebook. \
 The virtualenv environment named GenSAM_LLaVA needs to be created first following [Quick Start](#running-gensam-on-chamelon-dataset-with-llava1llava15).
```
pip install notebook 
pip install ipykernel ipywidgets
python -m ipykernel install --user --name GenSAM_LLaVA
```
 2. Open demo_v1.ipynb and select the 'GenSAM_LLaVA' kernel in the running notebook.
 



 ## TO-DO LIST
- [x] Update datasets and implementation scripts
- [ ] Keep incorporating more capabilities
- [ ] Demo and Codes


## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{hu2024relax,
  title={Relax Image-Specific Prompt Requirement in SAM: A Single Generic Prompt for Segmenting Camouflaged Objects},
  author={Hu, Jian and Lin, Jiayi and Gong, Shaogang and Cai, Weitong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={11},
  pages={12511--12518},
  year={2024}
}
```

## :cupid: Acknowledgements

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [BLIP2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
- [CLIP Surgery](https://github.com/xmed-lab/CLIP_Surgery)

