import torch

from .vpgtrans.common.config import Config
from .vpgtrans.common.registry import registry
from .vpgtrans.conversation.conversation import Chat, CONV_VISION
from PIL import Image
import numpy as np
# imports modules for registration
from .vpgtrans.models import *
from .vpgtrans.processors import *
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
from skimage.transform import resize
####
from . import get_image, DATA_DIR
import eagerpy as ep
from torchvision.transforms.functional import to_pil_image
import os
CFG_PATH = 'models/vpgtrans/vpgtrans_demo.yaml'

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
        if temp_dist_ori > 0:   
            if total_access > used_iterations: 
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
        
class TestVPGTrans:
    def __init__(self, device=None):
        cfg = Config(CFG_PATH, DATA_DIR)
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cpu')
        vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.model, self.vis_processor = model, vis_processor
        self.model.llama_model = self.model.llama_model.float().to('cpu')
        self.chat = Chat(model, vis_processor, device='cpu')
        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.chat.device = self.device
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.chat.move_stopping_criteria_device(self.device, dtype=self.dtype)
    
    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256,method=None, level=0):
        chat_state = CONV_VISION.copy()
        img_list = []
        if image is not None:
            image = get_image(image)
            self.chat.upload_img(image, chat_state, img_list)
        self.chat.ask(question, chat_state)
        llm_message = self.chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=max_new_tokens)[0]

        return llm_message
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256,method=None, level=0,gt_answer=None,max_it=None,task_name=None):
                ###
        images=[]
        for image in image_list:
            if method is not None and level!=0:
                tmp=image.split('/')
                image=os.path.join('/nvme/share/zhanghao/tiny_lvlm_new',tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])
            images.append(image)
        image_list=images
        ####
        image_list = [get_image(image) for image in image_list]
        imgs_ys=image_list
        chat_list = [CONV_VISION.copy() for _ in range(len(image_list))]
        aux_dist = []
        temp_adv_list = []
        index_attack = []
        attack_success = []
        for list_counter in range(4):
            aux_dist.append([])
            temp_adv_list.append([])
        for ind in range(len(chat_list)):   
            model_att =  self.chat    
            model_name = "vpgtrans"
            image=np.asarray(image_list[ind].resize((224, 224), resample=Image.BICUBIC))
            label = gt_answer[ind]
            attack=AdditiveGaussianNoiseAttack(model_att,task_name)

            adversarial_ori_unpack_1=attack(image, label, task_name,epsilons=100, unpack=False, question_list=question_list[ind], chat_list=chat_list[ind], max_new_tokens=max_new_tokens,model_name=model_name)#100
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls
            check_1, return_1 = adversarial_ori_check(adversarial_ori_1, image, total_prediction_calls_1, 1500)

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
            else:
                aux_dist[0].append(0)

            patch_used_step = 0
            if check_1 :   
                attacker = PatchAttack(model_att,task_name,label)
                patch_adversarial_1, patch_used_step=attacker.attack(image, label, adversarial_ori_1, int(return_1), mode='untargeted', question_list=question_list[ind], chat_list=chat_list[ind], max_new_tokens=max_new_tokens,model_name=model_name)
                
                aux_dist[1].append(l2_distance(patch_adversarial_1, image))
                
            else:
                aux_dist[1].append(0)       
            
            patch_remain_access = int(return_1) - patch_used_step
            check_2 = adversarial_patch_check(int(return_1) - patch_used_step)  

            if check_1 and check_2:   #允许攻击 patch+boundary
                initial_time = time.time()                
                attacker = EvolutionaryAttack(model_att,task_name,label)#patch_adversarial_1
                temp_adv_list[2] =  attacker.attack(image, label, patch_adversarial_1, initial_time, time_limit=99999999, 
                  iterations=patch_remain_access, source_step=3e-3, spherical_step=1e-1, rescale_or_not=2, rate = 0.2, big_size=(image.shape[0],image.shape[1]), center_size=(image.shape[0]*40/64,image.shape[1]*40/64), mode='untargeted', question_list=question_list[ind], chat_list=chat_list[ind], max_new_tokens=max_new_tokens,model_name=model_name)
                
                aux_dist[2].append(l2_distance(temp_adv_list[2][0], image))
            else:
                aux_dist[2].append(0)
            
            if check_1 and check_2:  #patch_adversarial_1
                attacker = SurFree(steps=patch_remain_access, max_queries=patch_remain_access,task=task_name,label=label)
                config = json.load(open("models/config_example.json", "r"))
                new_image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
                new_starting_points = torch.tensor(patch_adversarial_1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
                temp_result = attacker(model_att, new_image, starting_points=new_starting_points, **config["run"], question_list=question_list[ind], chat_list=chat_list[ind], max_new_tokens=max_new_tokens,model_name=model_name)
                temp_adv_list[3] = temp_result[1][0][0].permute(1, 2, 0).cpu().numpy()
                aux_dist[3].append(l2_distance(temp_adv_list[3], image))
            else:
                aux_dist[3].append(0)

        batch_outputs = [aux_dist,index_attack,attack_success]    
        return batch_outputs