import argparse
import json
from io import BytesIO
import collections
import requests
import torch
from PIL import Image
from tqdm import tqdm
import os
####
import numpy as np
from . import get_image


from .share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from .share4v.conversation import SeparatorStyle, conv_templates
from .share4v.mm_utils import (KeywordsStoppingCriteria,process_images,
                              get_model_name_from_path, tokenizer_image_token)
from .share4v.model.builder import load_pretrained_model
from .share4v.utils import disable_torch_init
import pdb


class Testsharegpt4v:
    def __init__(self, device=None):
        model_path="Lin-Chen/ShareGPT4V-7B"
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        model_path, None, model_name)        
        
        self.dtype = torch.float16
    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256,method=None, level=0):
        image = get_image(image)
        if method is not None and level!=0:
            image = Image.fromarray(d[method](np.asarray(image),level).astype(np.uint8))
        conv = self.conv.copy()
        # text=DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, 'mm_use_im_start_end', False):
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
                question = DEFAULT_IMAGE_TOKEN + '\n' + question

        text = question # + '\n<image>'
        # text = (text, image, self.image_process_mode)
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str  = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        output = self.do_generate([prompt], [image], stop_str=stop_str, dtype=self.dtype, temperature=0, max_new_tokens=max_new_tokens)[0]

        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256,method=None, level=0):
        images, prompts = [], []
        for image, question in zip(image_list, question_list):
            # print("jfk",image)
            if method is not None and level!=0:
                tmp=image.split('/')
                image=os.path.join('/mnt/petrelfs/zhanghao1/tiny_lvlm_new',tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])#,tmp[-1]
            elif 'attack' in image:
                tmp=image.split('/')
                image=os.path.join('/mnt/petrelfs/zhanghao1/attack_dataset',tmp[-2],tmp[-1])#,tmp[-1]
            else:
                tmp=image.split('/')
                image=os.path.join('/mnt/petrelfs/zhanghao1',tmp[-3],tmp[-2],tmp[-1])#,tmp[-1]
            # print("jkf",image,"22")
            image = Image.open(image)

            
            if self.model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                question = DEFAULT_IMAGE_TOKEN + '\n' + question

            conv = conv_templates['share4v_v1'].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()  

            prompts.append(prompt)
            images.append(image)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # stop_str  = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        outputs = self.do_generate(prompts, images, stop_str=stop_str, dtype=self.dtype, temperature=0, max_new_tokens=max_new_tokens,method=method, level=level,image_listnew=image_list.copy())

        return outputs

    @torch.no_grad()
    def do_generate(self, prompts, images, dtype=torch.float16, temperature=0, max_new_tokens=256, stop_str=None, keep_aspect_ratio=False,method=None, level=0,image_listnew=None):
       
        if 1:            
            # image_processor.preprocess
            images =process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=dtype)
            
        input_ids = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX) for prompt in prompts]
        keywords = [stop_str]
        # print(len(input_ids),input_ids[0].shape,"pppp")
        # for k in range(len(input_ids)):
        #     print(len(input_ids),input_ids[k].shape,"pppp")
        # print("nn",input_ids)
        stopping_criteria = None # [KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)] 
        # print(self.model)

        batch_size = len(input_ids)
        # print(len(input_ids[0]),"pkp")
        min_prompt_size = min([len(input_id) for input_id in input_ids])
        max_prompt_size = max([len(input_id) for input_id in input_ids])
        for i in range(len(input_ids)):
            padding_size = max_prompt_size - len(input_ids[i])
            # print(padding_size,self.tokenizer.pad_token_id)
            # input_ids[i].extend([self.tokenizer.pad_token_id] * padding_size)
            # print(input_ids[i],"00")
            # print(padding_size,input_ids[i],self.tokenizer.pad_token_id,"aa")
            # pdb.set_trace()
            input_ids[i] = [self.tokenizer.pad_token_id] * padding_size + input_ids[i]
            # print(input_ids[i],"0220")

        input_ids=torch.as_tensor(input_ids).to(self.model.device)
        
        
        with torch.inference_mode():
            output_ids = self.model.generate(
            input_ids,
            images=images,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=stopping_criteria)  
        # print(output_ids.shape,"op")  #4,74
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        # print(outputs,"kop")
        if stop_str is not None:
            for i in range(len(outputs)):
                tmp = outputs[i].strip()
                if tmp.endswith(stop_str):
                    tmp = tmp[:-len(stop_str)]
                outputs[i] = tmp.strip() 
                # print(outputs[i],i,"opop")
        # print(outputs,"op")
        return outputs