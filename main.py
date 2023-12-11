

from pickle import FALSE
from socket import IPPROTO_UDP
import torch
import cv2
import numpy as np
from PIL import Image
from  matplotlib import pyplot as plt
import argparse
import yaml
import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from utils_model import get_text_from_img, get_mask, fuse_mask
from utils_model import get_edge_img, DotDict, printd, mkdir
import os


## configs
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/mydemo.yaml')
    parser.add_argument('--visualization', action='store_true')

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
data_args = config['test_dataset']
model_args = DotDict(config)

## get data
dataset = datasets.make(data_args['dataset'])
dataset = datasets.make(data_args['wrapper'], args={'dataset': dataset})
loader = DataLoader(dataset, batch_size=data_args['batch_size'],
                    num_workers=8)
paths_img = dataset.dataset.paths_img
data_len = len(paths_img)
printd(f"dataset size:\t {len(paths_img)}")

## save dir
config_name = args.config.split("/")[-1][:-5]
save_path_dir = f'output_img/{config_name}/'
mkdir(save_path_dir)


## load pretrained model  
# CLIP surgery, SAM
from segment_anything import sam_model_registry, SamPredictor
from clip.clip_surgery_model import CLIPSurgery
import clip
sam = sam_model_registry[model_args.sam_model_type](checkpoint=model_args.sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)
clip_params={ 'attn_qkv_strategy':model_args.clip_attn_qkv_strategy}
clip_model, _ = clip.load(model_args.clip_model, device=device, params=clip_params)
clip_model.eval()
# VLM
llm_dict=None
if model_args.llm=='blip':
    from lavis.models import load_model_and_preprocess
    # blip_model_type="pretrain_opt2.7b"
    blip_model_type="pretrain_opt6.7b" 
    printd(f'loading BLIP ({blip_model_type})...')
    BLIP_model, BLIP_vis_processors, _ = load_model_and_preprocess(name="blip2_opt", 
                                                                   model_type=blip_model_type, 
                                                                   is_eval=True, 
                                                                   device=device)
    BLIP_dict = {"demo_data/9.jpg": 'lizard in the middle',}
    llm_dict = {
        'model': BLIP_model,
        'vis_processors':  BLIP_vis_processors,
    }
elif model_args.llm=='LLaVA':
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    disable_torch_init()
    print(f'llava pretrained model: {model_args.model_path}')
    model_path = os.path.expanduser(model_args.model_path)
    model_args.model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_args.model_base, model_args.model_name)
    if 'llama-2' in model_args.model_name.lower(): # from clip.py
        conv_mode = "llava_llama_2"
    elif "v1" in model_args.model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_args.model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    model_args.conv_mode = conv_mode
    llm_dict = {
        'model': model,
        'vis_processors':  image_processor,
        'tokenizer': tokenizer,
        'conv_mode': model_args.conv_mode,
        'temperature': model_args.temperature,
        'w_caption': model_args.LLaVA_w_caption,
    }
else:
    exit(f'unknow LLM: {model_args.llm}')


## metrics
import utils
metric_fn = utils.calc_cod
metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
val_metric1 = utils.Averager()
val_metric2 = utils.Averager()
val_metric3 = utils.Averager()
val_metric4 = utils.Averager()
recursive_times_l = []


## run model
printd('Start inference...')
for s_i, img_path, pairs in zip(range(data_len), paths_img, loader):
    printd(img_path)
    pil_img = Image.open(img_path).convert("RGB")

    ## infer GenSAM
    text, text_bg = get_text_from_img(pil_img, model_args.prompt_q, llm_dict, 
                                      model_args.use_gene_prompt, model_args.clip_use_bg_text, model_args)
    mask_l, mask_logit_origin_l, num_l, vis_dict = get_mask(pil_img, text, sam_predictor, clip_model, 
                                                            model_args, device,  
                                                            llm_dict=llm_dict, 
                                                            text_bg=text_bg,
                                                            is_visualization=args.visualization )
    recursive_times = len(mask_l)
    recursive_times_l.append(recursive_times)
    vis_mask_acc, vis_mask_logit_acc = fuse_mask(mask_logit_origin_l, 
                                                 sam_predictor.model.mask_threshold) # fuse masks from different iterations

    ## get metric
    tensor_gt = pairs['gt']
    inp_size = 1024
    mask_transform = transforms.Compose([
                    transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                ])

    # get metric of mask closest to fused mask 
    mask_delta_l = [np.sum((mask_i - vis_mask_acc)**2) for mask_i in mask_l]  # distance of each mask to fused one
    idxMaskSim = np.argmin(mask_delta_l)
    vis_tensor = Image.fromarray(mask_l[idxMaskSim].astype('uint8'))
    vis_tensor = mask_transform(vis_tensor)[0].view(1, 1, inp_size, inp_size)
    result1, result2, result3, result4 = metric_fn(vis_tensor, tensor_gt)
    val_metric1.add(result1, tensor_gt.shape[0])
    val_metric2.add(result2, tensor_gt.shape[0])
    val_metric3.add(result3, tensor_gt.shape[0])
    val_metric4.add(result4, tensor_gt.shape[0])
  

    ## visualization
    if args.visualization:
        vis_input_img = vis_dict['vis_input_img']
        vis_mask_l = vis_dict['vis_mask_l']
        points_l = vis_dict['points_l']
        labels_l = vis_dict['labels_l']
        img_name = img_path.split('/')[-1][:-4]
        vis_pt_l = [np.expand_dims(255*vis_mask_l[i], axis=2).repeat(3, axis=2) for i in range(len(vis_mask_l))]
        for i in range(len(vis_mask_l)):
            vis_input_img[i] = get_edge_img(255*vis_mask_l[i], vis_input_img[i],)
        for j in range(len(points_l)):
            for i, [x, y] in enumerate(points_l[j]):
                if labels_l[j][i] == 0:
                    clr = (50, 50, 255)
                elif labels_l[j][i] == 1:
                    clr = (255, 50, 50)
                else:
                    clr = (0, 255, 102)
                cv2.circle(vis_pt_l[j], (x, y), 6, clr, -1)
                cv2.circle(vis_input_img[j], (x, y), 6, clr, -1)
        for i in range(len(vis_input_img)-1):
            result_path = save_path_dir + img_name + f'_{i}.jpg'
            plt.imsave(result_path, vis_input_img[i])
            print(f'saving in {result_path}')


## print metric
printd('End inference...')
print(f'\ncloset to fuse (formated):\n\
            {round(val_metric4.item(),4):.3f}\t\
            {round(val_metric3.item(),4):.3f}\t\
            {round(val_metric2.item(),4):.3f}\t\
            {round(val_metric1.item(),4):.3f}\t')
print('average recursive times: ', np.mean(recursive_times_l ))