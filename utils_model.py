from asyncio import wait
import clip
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
import torch.nn.functional as F
import datetime
import os
BICUBIC = InterpolationMode.BICUBIC
eps = 1e-7


def fuse_mask(mask_logit_origin_l, sam_thr, fuse='avg'):

    num_mask = len(mask_logit_origin_l)
    if fuse=='avg':
        mask_logit_origin = sum(mask_logit_origin_l)/num_mask  #
        mask_logit = F.sigmoid(torch.from_numpy(mask_logit_origin)).numpy()
        mask = mask_logit_origin > sam_thr

    mask = mask.astype('uint8')
    mask_logit *= 255
    mask_logit = mask_logit.astype('uint8')

    return mask, mask_logit


def get_mask(pil_img, text, sam_predictor, clip_model, args, device='cuda', llm_dict=None, text_bg=None,
                reset_prompt_qkeys=False, new_prompt_qkeys_l=[], bg_cat_list=[], post_process_per_cat_fg=False,
                is_visualization=False):

    num_l = []
    mask_l = []
    mask_logit_origin_l = []
    prob_delta_list = []
    mask_logit_l = []
    vis_mask_logit_l = []
    bbox_list = []  # get the box prompt
    vis_dict = {}
    if is_visualization:
        vis_input_img = []
        vis_mask_l = []
        points_l = []
        labels_l = []

    ori_image = np.array(pil_img)
    sam_predictor.set_image(ori_image)

    cur_image = ori_image
    if is_visualization:  vis_input_img.append(cur_image.astype('uint8'))
    with torch.no_grad():
        flag_terminate = False
        for i in range(args.recursive+1):
            if i>=1 and args.update_text:
                cur_image_pil=pil_img
                text, text_bg = get_text_from_img(cur_image_pil, args.prompt_q, llm_dict,
                                                    args.use_gene_prompt, args.clip_use_bg_text, args,
                                                    reset_prompt_qkeys=reset_prompt_qkeys,
                                                    new_prompt_qkeys_l=new_prompt_qkeys_l,
                                                    bg_cat_list=bg_cat_list,
                                                    )
                print(f'iter {i} text:\t{text}, {text_bg}')
                if args.check_exist_each_iter and text==[]:
                    return None, mask_logit_origin_l, None, None, None, num_l, vis_dict
                    
            sm, sm_mean, sm_logit, clip_vis_dict = clip_surgery(cur_image, 
                                                                text, 
                                                                clip_model, 
                                                                args, device='cuda', 
                                                                text_bg=text_bg, 
                                                                is_visualization=is_visualization)

            # get positive points from individual maps (each sentence in the list), and negative points from the mean map
            points, labels, vis_radius, num = heatmap2points(sm, sm_mean, cur_image, args, is_visualization=is_visualization)

            # Inference SAM with points from CLIP Surgery
            if args.post_mode =='MaxIOUBoxSAMInput':
                if i==0:
                    if len(points) == 0:
                        x_min = 0
                        x_max = ori_image.shape[1]
                        y_min = 0
                        y_max = ori_image.shape[0]
                        bboxes = np.array([x_min, y_min, x_max, y_max])
                        mask_logit_origin, scores, logits = sam_predictor.predict(box=bboxes[None, :], multimask_output=True, return_logits=True,)
                    else:
                        mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=True, return_logits=True,)
                else:
                    if len(points) == 0:
                        mask_logit_origin, scores, logits = sam_predictor.predict(box=bbox_list[i-1][None, :],multimask_output=True, return_logits=True)
                    else:
                        mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels, point_coords=np.array(points), box=bbox_list[i-1][None, :],multimask_output=True, return_logits=True)
                mask = mask_logit_origin[np.argmax(scores)] > sam_predictor.model.mask_threshold
                #get bbox
                contours, _ = cv2.findContours(mask.copy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bboxes = []
                overlaps = []
                if len(contours)==0:
                    x_min = 0
                    x_max = mask_logit_origin[0].shape[1]
                    y_min = 0
                    y_max = mask_logit_origin[0].shape[0]
                    bboxes = np.array([x_min, y_min, x_max, y_max])
                else:
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        bbox = np.array([x, y, x + w, y + h])
                        bboxes.append(bbox)
                        overlap = (w * h) / np.sum(mask)
                        overlaps.append(overlap)
                    bboxes = np.array(bboxes)
                    overlaps = np.array(overlaps)
                    max_overlap_idx = np.argmax(overlaps)
                    max_bbox = bboxes[max_overlap_idx]
                    scaled_bbox = max_bbox.copy()
                    scaled_bbox[:2] -= np.floor((scaled_bbox[2:] - scaled_bbox[:2]) * 0.1).astype(int)
                    scaled_bbox[2:] += np.ceil((scaled_bbox[2:] - scaled_bbox[:2]) * 0.1).astype(int)
                    bboxes[max_overlap_idx] = scaled_bbox
                    bboxes = bboxes[max_overlap_idx]
                bbox_list.append(bboxes)

            mask_logit_origin = mask_logit_origin[np.argmax(scores)]
            mask = mask_logit_origin > sam_predictor.model.mask_threshold
            mask_logit = F.sigmoid(torch.from_numpy(mask_logit_origin)).numpy()

            delta_thr = 0.003

            if i>0:
                prob_delta_list.append(np.mean(np.abs(mask_logit-mask_logit_l[-1])))
                if i>1 and np.abs(prob_delta_list[-1]-prob_delta_list[-2])<delta_thr and prob_delta_list[-1]<delta_thr:
                    print('break (prob_delta_list & delta of prob_delta_list delta_thr)\t',i, delta_thr)
                    flag_terminate=True


            # update input image for next iter
            sm1 = sm_logit
            if args.clipInputEMA:
                cur_image = ori_image * sm1 * args.recursive_coef + cur_image * (1-args.recursive_coef)
            else:
                cur_image = cur_image * sm1 * args.recursive_coef + cur_image * (1-args.recursive_coef)

            vis_mask_logit_l.append((mask_logit * 255).astype('uint8'))
            # collect for visualization
            if is_visualization:
                vis_input_img.append(cur_image.astype('uint8'))

                vis_mask_l.append(mask.astype('uint8'))
                points_l.append(points)
                labels_l.append(labels)

            mask_logit_l.append(mask_logit)
            num_l.append(num)
            mask_l.append(mask)
            mask_logit_origin_l.append(mask_logit_origin)
            if flag_terminate:
                break

        if is_visualization:
            vis_dict = {
                    'vis_input_img': vis_input_img,
                    'vis_mask_l': vis_mask_l,
                    'points_l': points_l,
                    'labels_l': labels_l,
                    }

    return mask_l, mask_logit_origin_l, num_l, vis_dict


def clip_surgery(np_img, text, model, args, device='cuda', text_bg=None, is_visualization=False):
    if is_visualization:
        sm_sub_l, sm_bg_sub_l =[], []

    pil_img = Image.fromarray(np_img.astype(np.uint8))
    h, w = np_img.shape[:2]
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    image = preprocess(pil_img).unsqueeze(0).to(device)

    # CLIP architecture surgery acts on the image encoder
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)    # torch.Size([1, 197, 512])

    # Extract redundant features from an empty string
    redundant_features = clip.encode_text_with_prompt_ensemble(model, [args.rdd_str], device)  # torch.Size([1, 512])

    # Prompt ensemble for text features with normalization
    text_features = clip.encode_text_with_prompt_ensemble(model, text, device)  # torch.Size([x, 512])
    if args.clip_use_bg_text:
        text_bg_features = clip.encode_text_with_prompt_ensemble(model, text_bg, device)  # torch.Size([x, 512])


    def _norm_sm(_sm, h, w):
        side = int(_sm.shape[0] ** 0.5)
        _sm = _sm.reshape(1, 1, side, side)
        _sm = torch.nn.functional.interpolate(_sm, (h, w), mode='bilinear')[0, 0, :, :].unsqueeze(-1)
        _sm = (_sm - _sm.min()) / (_sm.max() - _sm.min())
        _sm = _sm.cpu().numpy()
        return _sm

    # Combine features after removing redundant features and min-max norm
    sm = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0, 1:, :]  # 整个输出：torch.Size([1, 197, x])  # 最后的1，是text这个list 的长度。
    sm_norm = (sm - sm.min(0, keepdim=True)[0]) / (sm.max(0, keepdim=True)[0] - sm.min(0, keepdim=True)[0])
    sm_mean = sm_norm.mean(-1, keepdim=True)
    if is_visualization:
        sm_sub_l = [_norm_sm(sm_norm[..., i:i+1], h, w) for i in range( sm_norm.size()[-1] )]
        sm_mean_fg = _norm_sm(sm_mean, h, w)

    sm_mean_bg, sm_mean_fg_bg=None, None
    if args.clip_use_bg_text:
        sm_bg = clip.clip_feature_surgery(image_features, text_bg_features, redundant_features)[0, 1:, :]  # 整个输出：torch.Size([1, 197, x])  # 最后的1，是text这个list 的长度。
        sm_norm_bg = (sm_bg - sm_bg.min(0, keepdim=True)[0]) / (sm_bg.max(0, keepdim=True)[0] - sm_bg.min(0, keepdim=True)[0])
        sm_mean_bg = sm_norm_bg.mean(-1, keepdim=True)
        if is_visualization:  sm_bg_sub_l = [_norm_sm(sm_norm_bg[...,i:i+1], h, w) for i in range(sm_norm_bg.size()[-1])]

        if args.clip_bg_strategy=='FgBgHm':
            sm_mean_fg_bg = sm_mean - sm_mean_bg
        else: # FgBgHmClamp
            sm_mean_fg_bg = torch.clamp(sm_mean - sm_mean_bg, 0, 1)

        sm_mean_fg_bg = (sm_mean_fg_bg - sm_mean_fg_bg.min(0, keepdim=True)[0]) / (sm_mean_fg_bg.max(0, keepdim=True)[0] - sm_mean_fg_bg.min(0, keepdim=True)[0])
        sm_mean_fg_bg_origin = sm_mean_fg_bg
        sm_mean = sm_mean_fg_bg_origin

    # expand similarity map to original image size, normalize. to apply to image for next iter

    sm1 = sm_mean
    sm_logit = _norm_sm(sm1, h, w)

    if is_visualization and args.clip_use_bg_text:
        sm_mean_bg = _norm_sm(sm_mean_bg, h, w)
        sm_mean_fg_bg = _norm_sm(sm_mean_fg_bg, h, w)
    # return sm, sm_mean, sm_logit, sm_mean_bg, sm_mean_fg_bg
    clip_vis_dict={}
    if is_visualization:
        clip_vis_dict={
            'sm_fg':	sm_mean_fg,
            'sm_bg':	sm_mean_bg,
            'sm_fg_bg':	sm_mean_fg_bg,
            'sm_sub_l':	sm_sub_l,
            'sm_bg_sub_l':	sm_bg_sub_l,}

    return sm, sm_mean, sm_logit, clip_vis_dict


template_q='Name of the {} in one word.'
template_bg_q='Name of the environment of the {} in one word.'
prompt_qkeys_dict={

    'TheCamo':          ['camouflaged animal'],
    'TheShadow':        ['shadow'],
    'TheGlass':         ['glass'],
    'ThePolyp':         ['polyp'],


    '3attriTheBgSyn':   ['concealed animal', 'hidden animal', 'unseen animal'],
    '3attriTheBgSynCamo':   ['camouflaged animal', 'disguised animal', 'hidden animal'],
    '3attriTheBgSynCamoSpec':   ['camouflaged species', 'disguised species', 'hidden species'],

    '3TheGlassSyn':     ['glass', 'window', 'mirror'],
    '3TheGlassSyn1':     ['glass', 'window', 'transparent material'],

    '3TheShadowSyn':    ['shadow', 'silhouette', 'profile'],
    '3TheShadowSyn1':    ['shadow', 'silhouette', 'outline'],

    '3ThePolypSyn':     ['polyp', 'appendage', 'tentacle'],
    '3ThePolypSyn1':    ['polyp', 'appendage', 'tumor'],
    '3ThePolypSyn2':    ['polyp', 'tumor', 'growth'],

    '1attriTheCamouflageBg_test': ['camouflaged animal'],
    '3attriTheBgSynCamo_test':   ['camouflaged animal', 'disguised animal', 'hidden animal'],

}
prompt_q_dict={}
for k, v in prompt_qkeys_dict.items():
    if prompt_q_dict.get(k) is None:
        prompt_q_dict[k] = [[template_q.format(key), template_bg_q.format(key)] for key in prompt_qkeys_dict[k]]
prompt_gene_dict={}
for k, v in prompt_qkeys_dict.items():
    if prompt_gene_dict.get(k) is None:
        prompt_gene_dict[k] = [prompt_qkeys_dict[k], ['environment']]


def get_text_from_img(pil_img, prompt_q, llm_dict, use_gene_prompt, get_bg_text, args,
                        reset_prompt_qkeys=False, new_prompt_qkeys_l=None,
                        bg_cat_list=[],
                        post_process_per_cat_fg=False):
    if use_gene_prompt:
        return prompt_gene_dict[args.prompt_q]
    else:  # use LLM model: BLIP2; LLaVA
        model = llm_dict['model']
        vis_processors = llm_dict['vis_processors']
        use_gene_prompt_fg=args.use_gene_prompt_fg
        if args.llm=='blip':
            return get_text_from_img_blip(pil_img, prompt_q,
                        model, vis_processors,
                        get_bg_text=get_bg_text,)
        elif args.llm=='LLaVA':
            tokenizer = llm_dict['tokenizer']
            conv_mode = llm_dict['conv_mode']
            temperature = llm_dict['temperature']
            w_caption = llm_dict['w_caption']
            if args.check_exist_each_iter: # only for multiple classes
                if not cat_exist(
                    pil_img, new_prompt_qkeys_l[0],
                    model, vis_processors, tokenizer,
                    ):
                    return [], []

            return get_text_from_img_llava(pil_img, prompt_q,
                        model, vis_processors, tokenizer,
                        get_bg_text=get_bg_text,
                        conv_mode=conv_mode,
                        temperature=temperature,
                        w_caption=w_caption,
                        use_gene_prompt_fg=use_gene_prompt_fg,
                        reset_prompt_qkeys=reset_prompt_qkeys,
                        new_prompt_qkeys_l=new_prompt_qkeys_l,
                        bg_cat_list=bg_cat_list)


def get_text_from_img_blip(pil_img, prompt_q=None, model=None, vis_processors=None, get_bg_text=False, device='cuda', ):

    image = vis_processors["eval"](pil_img).unsqueeze(0).to(device)
    blip_output = model.generate({"image": image})
    blip_output = blip_output[0].split('-')[0]
    context = [
        ("Image caption",blip_output),
    ]
    template = "Question: {}. Answer: {}."

    question_l = ["Name of hidden animal in one word."] if prompt_q is None else prompt_q_dict[prompt_q]
    text_list = []
    textbg_list = []
    for question in question_l:
        out_list = []
        prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question[0] + " Answer:"
        blip_output_forsecond = model.generate({"image": image, "prompt": prompt})
        blip_output_forsecond = blip_output_forsecond[0].split('-')[0].replace('\'','')
        if len(blip_output_forsecond)==0:    continue
        out_list.append(blip_output_forsecond)
        out_list = " ".join(out_list)
        text_list.append(out_list)

        if get_bg_text:
            ## get background text
            outbg_list = []
            prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question[0] + " Answer:" + blip_output_forsecond + ". Question: " + question[1] + " Answer:"
            blip_output_forsecond = model.generate({"image": image, "prompt": prompt})
            blip_output_forsecond = blip_output_forsecond[0].split('-')[0].replace('\'','')
            print(prompt)
            print(blip_output_forsecond)
            if 'Question' in blip_output_forsecond:
                blip_output_forsecond = blip_output_forsecond.split('Question')[0]
            blip_output_forsecond = blip_output_forsecond.split('.')[0]
                # while blip_output_forsecond[-1]==' ':
                #     blip_output_forsecond = blip_output_forsecond[:-1]
            if len(blip_output_forsecond)==0:     continue
            outbg_list.append(blip_output_forsecond)
            outbg_list = " ".join(outbg_list)

            textbg_list.append(outbg_list)

    print(f'caption: {blip_output}')
    text = text_list
    text_bg = textbg_list

    # deal with empty text
    if len(text)==0:
        text = prompt_gene_dict[prompt_q][0]
    if get_bg_text:
        def _same(l1, l2):
            l1_ = [i1.replace(' ','') for i1 in l1]
            l2_ = [i2.replace(' ','') for i2 in l2]
            return set(l1_)==set(l2_)
        if _same(text, text_bg):    text_bg=[]
        if len(text_bg)==0:
            text_bg = prompt_gene_dict[prompt_q][1]

    print(text, text_bg)
    return text, text_bg


def get_text_from_img_llava(
    pil_img, prompt_q,
    model, image_processor, tokenizer,
    get_bg_text=False,
    conv_mode='llava_v0',
    temperature=0.2,
    w_caption=False,
    use_gene_prompt_fg=False,
    reset_prompt_qkeys=False,
    new_prompt_qkeys_l=[],
    bg_cat_list=[]):
    '''
    input
    '''
    from transformers import TextStreamer
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    # from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
    if reset_prompt_qkeys:
        prompt_qkeys_l = new_prompt_qkeys_l
        question_l = [[template_q.format(key), template_bg_q.format(key)] for key in prompt_qkeys_l]
        prompt_gene_l = [prompt_qkeys_l, ['environment']]
        prompt_gene_fg_l = prompt_qkeys_l
        # print('prompt_qkeys_l: ', prompt_qkeys_l)
        # print('question_l: ', question_l)
        # print('prompt_gene_l: ', prompt_gene_l)
        # print('prompt_gene_fg_l: ', prompt_gene_fg_l)
    else:
        prompt_qkeys_l = prompt_qkeys_dict[prompt_q]
        question_l = prompt_q_dict[prompt_q]
        prompt_gene_l = prompt_gene_dict[prompt_q]
        prompt_gene_fg_l = prompt_gene_dict[prompt_q][0]
    text_list = []
    textbg_list = []

    image = pil_img #load_image(img_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    # get question index: caption:0, fg:1, bg:2
    fg_idx = 0
    bg_idx = 1
    if w_caption:
        fg_idx = 1
        bg_idx = 2

    disable_torch_init()
    for qi, qs in enumerate(question_l):

        if w_caption:
            q_keyword = prompt_qkeys_l[qi]
            caption_q = f'This image is from {q_keyword} detection task, describe the {q_keyword} in one sentence'
            qs=[caption_q] + qs

        image = pil_img #load_image(img_path)
        conv = conv_templates[conv_mode].copy() # 是否需要改一下system 提示词，换成caption？

        for i, inp in enumerate(qs):
            if i==fg_idx and use_gene_prompt_fg:
                text_list.append(prompt_gene_fg_l[qi])
                continue

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=1024,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs

            if w_caption and i==0:    continue
            if outputs.find('"') > 0:
                outputs = outputs.split('"')[1]
            elif outputs.find(' is an ') > 0:
                outputs = outputs.split(' is an ')[1]
            elif outputs.find(' is a ') > 0:
                outputs = outputs.split(' is a ')[1]
            outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '') #"<im_end>"
            outputs = outputs.replace('<|im_end|>', '')
            outputs = outputs.replace('</s>', '')
            if outputs[-1]=='.':    outputs = outputs[:-1]
            while outputs[0]==' ':  outputs=outputs[1:]

            if i==fg_idx:
                text_list.append(outputs)
                if not get_bg_text: break
            elif i==bg_idx:
                if outputs.upper() != text_list[-1].upper():
                    textbg_list.append(outputs)

    if len(textbg_list+bg_cat_list)==0:
        textbg_list=['background']
    return text_list, textbg_list+bg_cat_list


def heatmap2points(sm, sm_mean, np_img, args, attn_thr=-1, is_visualization=False):
    cv2_img = cv2.cvtColor(np_img.astype('uint8'), cv2.COLOR_RGB2BGR)
    if attn_thr < 0:
        attn_thr = args.attn_thr
    map_l=[]
    if args.use_adaptive_thr:
        p, l, map1, map = clip.similarity_map_to_points(sm_mean, cv2_img.shape[:2], cv2_img, t=attn_thr, down_sample=args.down_sample) # p: [pos (min->max), neg(max->min)]
        map_l.append(map)
        num = len(p) // 2
        map_pos = map[:num]
        map_neg = map[num:] # negatives in the second half

        if num == 1:
            labels = [0,1]
            points = [p[1]] + [p[0]]
        else:
            num_min, num_max = one_dimensional_kmeans_with_min_max(map_neg, 2)
            points_neg = p[num:num+num_min]
            points = points_neg
            labels = [0] * num_min
            num_min, num_max = one_dimensional_kmeans_with_min_max(map_pos, 2)
            points_pos = p[num-num_max:num]
            points = points + points_pos # positive in first half
            labels_1 = [1] * num_max
            labels.extend(labels_1)
    else:

        p, l, map, _ = clip.similarity_map_to_points(sm_mean, cv2_img.shape[:2], cv2_img, t=attn_thr,
                                                    down_sample=args.down_sample) # p: [pos (min->max), neg(max->min)]
        map_l.append(map)
        num = len(p) // 2
        points = p[num:] # negatives in the second half
        labels = [l[num:]]

        points = points + p[:num] # positive in first half
        labels.append(l[:num])
        labels = np.concatenate(labels, 0)
    vis_radius = []
    if is_visualization:
        vis_radius = [np.linspace(5,2,num)]
        vis_radius.append(np.linspace(2,5,num))
        vis_radius = np.concatenate(vis_radius, 0).astype('uint8')

    return points, labels, vis_radius, num


def get_dir_from_args(args, parent_dir='output_img/'):
    text_filename = f'{args.llm}Text'
    if args.update_text:
        text_filename += 'Update'
    parent_dir += f'{text_filename}/'

    exp_name = ''
    exp_name += f's{args.down_sample}_thr{args.attn_thr}'
    if args.recursive > 0:
        exp_name += f'_rcur{args.recursive}'
        if args.recursive_coef!=.3:
            exp_name += f'_{args.recursive_coef}'
    if args.rdd_str != '':
        exp_name += f'_rdd{args.rdd_str}'
    if args.clip_attn_qkv_strategy!='vv':
        exp_name += f'_qkv{args.clip_attn_qkv_strategy}'
        
    if args.clipInputEMA:  # darken
        exp_name += f'_clipInputEMA'

    if args.post_mode !='':
        exp_name += f'_post{args.post_mode}'
    if args.prompt_q!='Name of hidden animal in one word':
        exp_name += f'_prompt_q{args.prompt_q}'
        if args.use_gene_prompt:
            exp_name += 'Gene'
        if args.use_gene_prompt_fg:
            exp_name += 'GeneFg'
    if args.clip_use_bg_text:
        exp_name += f'_{args.clip_bg_strategy}'

    if args.llm=='LLaVA' and args.LLaVA_w_caption:
        exp_name += f'_shortCaption'


    save_path_dir = f'{parent_dir+exp_name}/'
    printd(f'{exp_name} ({args}')

    return save_path_dir


def one_dimensional_kmeans_with_min_max(data, k, max_iterations=100):
    np.random.seed(0)
    data = np.array(data)
    initial_centers = np.random.choice(data, size=k, replace=False)
    centers = initial_centers
    min_values = np.zeros(k)
    max_values = np.zeros(k)
    for _ in range(max_iterations):
        labels = np.argmin(np.abs(data[:, np.newaxis] - centers), axis=1)
        new_centers = np.array([data[labels == i].mean() for i in range(k)])
        for i in range(k):
            cluster_data = data[labels == i]
            min_values[i] = cluster_data.min()
            max_values[i] = cluster_data.max()
        if np.all(centers == new_centers):
            break
        centers = new_centers
    min_mean_cluster_index = np.argmin(min_values)
    max_mean_cluster_index = np.argmax(max_values)
    min_mean_cluster_count = np.sum(labels == min_mean_cluster_index)
    max_mean_cluster_count = np.sum(labels == max_mean_cluster_index)
    return min_mean_cluster_count, max_mean_cluster_count


#### utility ####
class DotDict:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def mkdir(path):
    if not os.path.isdir(path) and not os.path.exists(path):
        os.makedirs(path)

def printd(str):
    dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(dt+'\t '+str)

def get_edge_img_path(mask_path, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # edges = cv2.Canny(binary_mask, threshold1=30, threshold2=100)
    # kernel = np.ones((5, 5), np.uint8)
    # thicker_edges = cv2.dilate(edges, kernel, iterations=1)
    # coord=(thicker_edges==255)
    # img[binary_mask==255] = img[binary_mask==255]*0.8 + np.array([[[0,0,51]]])
    # img[...,2][coord]=255
    # return img
    return get_edge_img(binary_mask, img)

def get_edge_img(binary_mask, img):
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


    kernel = np.ones((5, 5), np.uint8)

    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    edges = cv2.Canny(binary_mask, threshold1=30, threshold2=100)
    thicker_edges = cv2.dilate(edges, kernel, iterations=1)
    coord=(thicker_edges==255)
    img[...,:][coord]=np.array([255, 200,200])
    coord_fg = (binary_mask==255)
    coord_bg = (binary_mask==0)

    r = 0.2
    img[...,0][coord_fg] = img[...,0][coord_fg] * (1-r) + 255 * r
    img[...,2][coord_bg] = img[...,2][coord_bg] * (1-r) + 255 * r
    img = np.clip(img,0,255) #.astype(np.uint8)

    return img

