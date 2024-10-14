import os
import importlib
from gradio_client import Client

import clip
import torch
####
from utils.tools import has_word, remove_special_chars
from collections import defaultdict
from .additive_noise import AdditiveGaussianNoiseAttack
import pdb
from .patch_attack import PatchAttack
import time
from .evolutionary_attack import EvolutionaryAttack
from .surfree import SurFree
import json
import pdb
from skimage.transform import resize
####
from . import get_BGR_image, DATA_DIR
from . import llama_adapter_v2 as llama
import numpy as np
import eagerpy as ep
from torchvision.transforms.functional import to_pil_image



llama_dir = f'{DATA_DIR}/llama_checkpoints'
model_path = f'{DATA_DIR}/llama_checkpoints/llama_adapter_v2_LORA-BIAS-7B.pth' # llama_adapter_v2_BIAS-7B.pth, llama_adapter_v2_0518.pth


from gradio_client import Client

def l2_distance(a, b):
    # print(a[0].shape,"ffs",b.shape)
    return (np.sum((a/255.0 - b/255.0) ** 2))**0.5
def adversarial_ori_check(adversarial_ori, image, used_iterations, total_access):
    #用来判断adversarial_ori是否正常，下一步该怎么做（是否还需要决策攻击）
    """
        adversarial_ori    : 初始对抗样本
        image              : 原始图像
        used_iterations    : 已经迭代过的次数
        total_access       : 总查询次数
    """
    if adversarial_ori is None:   #本身就不存在，说明攻击失败,本身不用攻击(本身预测就是错的)
        return False, -200
    else:   #说明攻击成功了，需要计算噪声幅度
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
        
class TestLLamaAdapterV2_web:
    def __init__(self) -> None:
        self.model = Client("http://llama-adapter.opengvlab.com/")

    def generate(self, image: str, question: str, max_length=128, temperature=0.1, top_p=0.75):
        output = self.model.predict(image, question, max_length, temperature, top_p, fn_index=1)
        
        return output


class TestLLamaAdapterV2:
    def __init__(self, device=None) -> None:
        # choose from BIAS-7B, LORA-BIAS-7B
        model, preprocess = llama.load(model_path, llama_dir, device, max_seq_len=256, max_batch_size=16)
        model.eval()
        self.img_transform = preprocess
        self.model = model
        self.device = device

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256, temperature=0, top_p=0.75,method=None, level=0):
        imgs = [get_BGR_image(image)]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama.format_prompt(question)]
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens)
        result = results[0].strip()

        return result
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=128,method=None, level=0,gt_answer=None,max_it=None,task_name=None):
                ###
        images=[]
        for image in image_list:
            if method is not None and level!=0:
                tmp=image.split('/')
                image=os.path.join('/nvme/share/zhanghao/tiny_lvlm_new',tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])
            images.append(image)
        image_list=images

        imgs = [get_BGR_image(img) for img in image_list]
        aux_dist = []
        temp_adv_list = [] 
        index_attack = []
        attack_success = []
        for list_counter in range(4):
            aux_dist.append([])
            temp_adv_list.append([])
        for ind in range(len(gt_answer)):   
            model_att =  self.model    
            model_name = "adv2"
            vis_proc=self.img_transform            
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

            if check_1:   #允许攻击
                temp_adv_list[0] = adversarial_ori_1
                aux_dist[0].append(l2_distance(temp_adv_list[0], image))
                # index_valid[0].append(1)
            else:
                aux_dist[0].append(0)

            patch_used_step = 0
            if check_1 :   #允许攻击 patch attack
                start_time2 = time.time()
                attacker = PatchAttack(model_att,task_name,label)
                patch_adversarial_1, patch_used_step=attacker.attack(image, label, adversarial_ori_1, int(return_1), mode='untargeted', question_list=question_list[ind], chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)
                # patch_refinement(image, temp_adv_img, model, label, total_access, mode='untargeted')
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


           
            if check_1 and check_2:  #patch_adversarial_1
                start_time4= time.time()
                attacker = SurFree(steps=patch_remain_access, max_queries=patch_remain_access,task=task_name,label=label)
                config = json.load(open("models/config_example.json", "r"))
                new_image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
                new_starting_points = torch.tensor(patch_adversarial_1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
                temp_result = attacker(model_att, new_image, starting_points=new_starting_points, **config["run"], question_list=question_list[ind], chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)
               
                temp_adv_list[3] = temp_result[1][0][0].permute(1, 2, 0).cpu().numpy()
                aux_dist[3].append(l2_distance(temp_adv_list[3], image))
                end_time4 = time.time()
            else:
                aux_dist[3].append(0) 

        batch_outputs = [aux_dist,index_attack,attack_success]   
        return batch_outputs
    