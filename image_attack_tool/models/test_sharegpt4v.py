import argparse
import json
from io import BytesIO
from PIL import Image
import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm

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

####
from utils.tools import has_word, remove_special_chars
from collections import defaultdict
from .additive_noise import AdditiveGaussianNoiseAttack
import pdb
from .patch_attack import PatchAttack
import time
import pdb
from .evolutionary_attack import EvolutionaryAttack

from .surfree import SurFree
import json
from skimage.transform import resize
####
import pdb
import eagerpy as ep
from torchvision.transforms.functional import to_pil_image
import os


def l2_distance(a, b):    
    return (np.sum((a/255.0 - b/255.0) ** 2))**0.5
def adversarial_ori_check(adversarial_ori, image, used_iterations, total_access):
    """
        adversarial_ori    : 初始对抗样本
        image              : 原始图像
        used_iterations    : 已经迭代过的次数
        total_access       : 总查询次数
    """
    if adversarial_ori is None:  
        return False, -200
    else:  
        temp_dist_ori = l2_distance(adversarial_ori, image)
        if temp_dist_ori > 0:   #说明不是直接成功的
            if total_access > used_iterations:  #次数还没有用完，说明可以继续进行运算
                # print("sd",total_access - used_iterations,total_access ,used_iterations)#sd 915 1500 85
                return True, total_access - used_iterations #0
            else:   #次数用完了，直接返回当前噪声幅度
                return False, temp_dist_ori

        else:  #没有攻击成功，不太可能发生
            return False, 0
def adversarial_patch_check(remain_access):
    #  判断patch是否把次数用完
    if remain_access == 0:   #说明次数已经用完
        return False
    else:
        return True

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
    def batch_generate(self, image_list, question_list, max_new_tokens=256,method=None, level=0,gt_answer=None,max_it=None,task_name=None):
        images, prompts = [], []
        for image, question in zip(image_list, question_list):           
            if method is not None and level!=0:
                tmp=image.split('/')
                image=os.path.join('/nvme/share/zhanghao/tiny_lvlm_new',tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])#,tmp[-1]
            image = Image.open(image).convert("RGB")

            
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
        imgs = images
        question_list= prompts
        aux_dist = []
        temp_adv_list = [] 
        index_attack = []
        attack_success = []
        for list_counter in range(4):
            aux_dist.append([])
            temp_adv_list.append([])
        for ind in range(len(gt_answer)):
            model_att =  self.do_generate    
            model_name = "sharegpt4v"
            vis_proc=[stop_str,method,level]
            
            image=np.asarray(imgs[ind].resize((224, 224), resample=Image.BICUBIC))
            tmp=image_list[ind].split('/')

            label = gt_answer[ind]
            attack=AdditiveGaussianNoiseAttack(model_att,task_name)

            start_time1 = time.time()
            adversarial_ori_unpack_1=attack(image, label, task_name,epsilons=100, unpack=False, question_list=question_list[ind], chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)#100
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls
            
            
            check_1, return_1 = adversarial_ori_check(adversarial_ori_1, image, total_prediction_calls_1, 1500)
            end_time1= time.time()

            if return_1==-200:
                index_attack.append(0)
                attack_success.append(0)
            elif return_1==0:
                index_attack.append(1)
                attack_success.append(0)#攻击了，但没有成功
            elif check_1==1 :
                index_attack.append(1)
                attack_success.append(1)
            else:
                index_attack.append(1)
                attack_success.append(1)
                return_1=0

            if check_1:   #允许攻击

                temp_adv_list[0] = adversarial_ori_1
                aux_dist[0].append(l2_distance(temp_adv_list[0], image))
            else:
                aux_dist[0].append(0)            
          
                
            patch_used_step = 0
            if check_1 :             
                start_time2 = time.time()
                attacker = PatchAttack(model_att,task_name,label)
                patch_adversarial_1, patch_used_step=attacker.attack(image, label, adversarial_ori_1, int(return_1), mode='untargeted', question_list=question_list[ind], chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)
               
                aux_dist[1].append(l2_distance(patch_adversarial_1, image))


                end_time2 = time.time()
            else:
                aux_dist[1].append(0)
                
            patch_dist = aux_dist[1][-1]
            patch_remain_access = int(return_1) - patch_used_step
            check_2 = adversarial_patch_check(int(return_1) - patch_used_step)  

            if check_1 and check_2:   
                initial_time = time.time()
                
                start_time3 = time.time()
                attacker = EvolutionaryAttack(model_att,task_name,label)#patch_adversarial_1
                temp_adv_list[2] =  attacker.attack(image, label, patch_adversarial_1, initial_time, time_limit=99999999, 
                  iterations=patch_remain_access, source_step=3e-3, spherical_step=1e-1, rescale_or_not=2, rate = 0.2, big_size=(image.shape[0],image.shape[1]), center_size=(image.shape[0]*40/64,image.shape[1]*40/64), mode='untargeted', question_list=question_list[ind], chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)
                
                
                aux_dist[2].append(l2_distance(temp_adv_list[2][0], image))

                end_time3 = time.time()                
            else:
                aux_dist[2].append(0)

           
            if check_1 and check_2: 
                start_time4= time.time()
                attacker = SurFree(steps=patch_remain_access, max_queries=patch_remain_access,task=task_name,label=label)
                config = json.load(open("models/config_example.json", "r"))
                new_image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
                new_starting_points = torch.tensor(patch_adversarial_1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
                temp_result = attacker(model_att, new_image, starting_points=new_starting_points, **config["run"], question_list=question_list[ind], chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)
               
                temp_adv_list[3] = temp_result[1][0][0].permute(1, 2, 0).cpu().numpy()
                iii,_=ep.astensor_(temp_result[1][0])
                new_image_raw=iii.raw
                new_image_tensor = torch.tensor(new_image_raw, dtype=torch.float32)
                inputs = to_pil_image(new_image_tensor.squeeze(0))
                zhlin=np.uint8(inputs)
               
                aux_dist[3].append(l2_distance(temp_adv_list[3], image))

                end_time4 = time.time()
            else:
                aux_dist[3].append(0) 

        batch_outputs = [aux_dist,index_attack,attack_success]
        return batch_outputs

    @torch.no_grad()
    def do_generate(self, prompts, images, dtype=torch.float16, temperature=0, max_new_tokens=256, stop_str=None, keep_aspect_ratio=False,method=None, level=0,image_listnew=None):
       
        if 1:  
            images =process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=dtype)
            
        input_ids = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX) for prompt in prompts]
        keywords = [stop_str]
        stopping_criteria = None # 
        batch_size = len(input_ids)
        min_prompt_size = min([len(input_id) for input_id in input_ids])
        max_prompt_size = max([len(input_id) for input_id in input_ids])
        for i in range(len(input_ids)):
            padding_size = max_prompt_size - len(input_ids[i])
            input_ids[i] = [self.tokenizer.pad_token_id] * padding_size + input_ids[i]

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
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        if stop_str is not None:
            for i in range(len(outputs)):
                tmp = outputs[i].strip()
                if tmp.endswith(stop_str):
                    tmp = tmp[:-len(stop_str)]
                outputs[i] = tmp.strip() 
        return outputs