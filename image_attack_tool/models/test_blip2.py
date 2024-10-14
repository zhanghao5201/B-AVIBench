import torch
import contextlib
from types import MethodType
from lavis.models import load_model_and_preprocess
from lavis.models.eva_vit import convert_weights_to_fp16
from . import get_image
import os
####
from utils.tools import has_word, remove_special_chars
from collections import defaultdict
from .additive_noise import AdditiveGaussianNoiseAttack
import pdb
from .patch_attack import PatchAttack
import time
from .evolutionary_attack import EvolutionaryAttack
# from new_foolbox_attacks.surfree_refinement import sf_refinement
from .surfree import SurFree
import json
import time
from torchvision import transforms
from skimage.transform import resize
import imageio
import eagerpy as ep
from torchvision.transforms.functional import to_pil_image
import numpy as np

def l2_distance(a, b):
    return (np.sum((a/255.0 - b/255.0) ** 2))**0.5
def adversarial_ori_check(adversarial_ori, image, used_iterations, total_access):
    #用来判断adversarial_ori是否正常，下一步该怎么做（是否还需要决策攻击）
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
        if temp_dist_ori > 0:  
            if total_access > used_iterations:                  
                return True, total_access - used_iterations #0
            else:   
                return False, temp_dist_ori
        else: 
            return False, 0
def adversarial_patch_check(remain_access):    
    if remain_access == 0:   
        return False
    else:
        return True

def new_maybe_autocast(self, dtype=None):
    return contextlib.nullcontext()
    enable_autocast = self.device != torch.device("cpu")
    if not enable_autocast:
        return contextlib.nullcontext()
    elif dtype is torch.bfloat16:
        if torch.cuda.is_bf16_supported():
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
            return torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        return torch.cuda.amp.autocast(dtype=dtype)


class TestBlip2:
    def __init__(self, device=None) -> None:
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device='cpu'
        )
        self.model.maybe_autocast = MethodType(new_maybe_autocast, self.model)

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float32 # torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            self.model.visual_encoder = self.model.visual_encoder.to(self.device, dtype=self.dtype)
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=30,method=None, level=0):
        image = get_image(image)        
        
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device, dtype=self.dtype)
        answer = self.model.generate({
            "image": image, "prompt": f"Question: {question} Answer:"
        }, max_length=max_new_tokens)

        return answer[0]
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=30,method=None, level=0,gt_answer=None,max_it=None,task_name=None):
        ###
        images=[]
        for image in image_list:
            if method is not None and level!=0:
                tmp=image.split('/')
                image=os.path.join('/nvme/share/zhanghao/tiny_lvlm_new',tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])
            images.append(image)
        image_list=images
        ####
        imgs = [get_image(img) for img in image_list]
        aux_dist = []
        temp_adv_list = [] 
        index_attack = []
        attack_success = []         
        for list_counter in range(4):
            aux_dist.append([])
            temp_adv_list.append([])
        
        for ind in range(len(gt_answer)):   
            model_att =  self.model    
            model_name = "blip2"
            vis_proc=self.vis_processors
            
            image=np.asarray(imgs[ind].resize((224, 224), resample=Image.BICUBIC))
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
                attack_success.append(0)
            elif check_1==1 :
                index_attack.append(1)
                attack_success.append(1)
            else:
                index_attack.append(1)
                attack_success.append(1)
                return_1=0
            
            if check_1:   
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
                attacker = EvolutionaryAttack(model_att,task_name,label)
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
                aux_dist[3].append(l2_distance(temp_adv_list[3], image))
                iii,_=ep.astensor_(temp_result[1][0])
                new_image_raw=iii.raw
                new_image_tensor = torch.tensor(new_image_raw, dtype=torch.float32)
                inputs = to_pil_image(new_image_tensor.squeeze(0))
                zhlin=np.uint8(inputs)
                end_time4 = time.time()
            else:
                aux_dist[3].append(0)  
        batch_outputs = [aux_dist,index_attack,attack_success]
        
        return batch_outputs