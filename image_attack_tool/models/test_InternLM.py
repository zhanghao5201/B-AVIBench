import torch
import os
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import io
import base64
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import Optional
import xlsxwriter
import pandas as pd
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision
####
from utils.tools import has_word, remove_special_chars
from collections import defaultdict
from .additive_noise import AdditiveGaussianNoiseAttack
import pdb
from .patch_attack import PatchAttack
import time
import pdb
from .evolutionary_attack import EvolutionaryAttack
# from new_foolbox_attacks.surfree_refinement import sf_refinement
from .surfree import SurFree
import json
from skimage.transform import resize
import numpy as np
import os
####
import eagerpy as ep
from torchvision.transforms.functional import to_pil_image
from . import get_image, DATA_DIR


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
            

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

stop_words_ids = [
                  torch.tensor([103027]).cuda(), ### end of human
                  torch.tensor([103028]).cuda(), ### end of bot
                 ]
stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)])
def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

    model.apply(_convert_weights_to_fp16)

def generate_answer(model, text, image_path,max_new_tokens=256):
    img_embeds = model.encode_img(image_path)
    prompt_segs = text.split('<ImageHere>')
    prompt_seg_tokens = [
        model.tokenizer(seg,
                             return_tensors='pt',
                             add_special_tokens=i == 0).
        to(model.internlm_model.model.embed_tokens.weight.device).input_ids
        for i, seg in enumerate(prompt_segs)
    ]
    prompt_seg_embs = [
        model.internlm_model.model.embed_tokens(seg)
        for seg in prompt_seg_tokens
    ]
    prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
    
    prompt_embs = torch.cat(prompt_seg_embs, dim=1)
    
    outputs = model.internlm_model.generate(
        inputs_embeds=prompt_embs,
        max_new_tokens=max_new_tokens,
        num_beams=5,
        do_sample=False,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=0,
        stopping_criteria=stopping_criteria,
    )
    #print (outputs)
    output_token = outputs[0]
    if output_token[0] == 0:
        output_token = output_token[1:]
    if output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token,
                                              add_special_tokens=False)

    output_text = output_text.split(model.eoa)[0]
    output_text = output_text.split('<|Bot|>')[-1].strip()
    return output_text
@torch.no_grad()
def batch_generate_answer_new(model, text, image_path,max_new_tokens=256):
    model.eval()
    prompt_embs_final=[]
    img_embeds_All = model.encode_img(image_path)
    pro_token=[]
    # print(text)
    for k in range(len(text)):
        prompt_segs = text[k].split('<ImageHere>')
        prompt_seg_tokens = [
            model.tokenizer(seg,
                             return_tensors='pt',
                             add_special_tokens=i == 0).
            to(model.internlm_model.model.embed_tokens.weight.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ] 
        pro_token.append(prompt_seg_tokens)       
    max_prompt_size = max([input_id[1].shape[1] for input_id in pro_token])
    for i in range(len(pro_token)):
        padding_size = max_prompt_size - pro_token[i][1].shape[1]        
        padding_ids =torch.tensor([0]).repeat(1, padding_size).to(pro_token[i][1][0].device,dtype=pro_token[i][1][0].dtype)
        pro_token[i][1] = torch.cat((padding_ids,pro_token[i][1]),dim=1)
    outputs_out=[]
    for k in range(len(text)):
        img_embeds=img_embeds_All[k].unsqueeze(0)
        prompt_seg_tokens = pro_token[k]   
        prompt_seg_embs = [
            model.internlm_model.model.embed_tokens(seg)
            for seg in prompt_seg_tokens
        ]
        prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
        prompt_embs = torch.cat(prompt_seg_embs, dim=1)        
        outputs = model.internlm_model.generate(
        inputs_embeds=prompt_embs.to(device=model.device,dtype=torch.float16),
        max_new_tokens=max_new_tokens,
        num_beams=5,
        do_sample=False,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=0,
        stopping_criteria=stopping_criteria,
        )
        output_token = outputs[0]
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = model.tokenizer.decode(output_token,
                                              add_special_tokens=False)

        output_text = output_text.split(model.eoa)[0]
        output_text = output_text.split('<|Bot|>')[-1].strip()
        outputs_out.append(output_text)

    return outputs_out

@torch.no_grad()
def batch_generate_answer(model, text, image_path,max_new_tokens=256):
    model.eval()
    prompt_embs_final=[]
    img_embeds_All = model.encode_img(image_path)
    pro_token=[]
    # print(text)
    for k in range(len(text)):
        prompt_segs = text[k].split('<ImageHere>')
        prompt_seg_tokens = [
            model.tokenizer(seg,
                             return_tensors='pt',
                             add_special_tokens=i == 0).
            to(model.internlm_model.model.embed_tokens.weight.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ] 
        pro_token.append(prompt_seg_tokens)
    for k in range(len(text)):
        img_embeds=img_embeds_All[k].unsqueeze(0)
        prompt_seg_tokens = pro_token[k]     
        prompt_seg_embs = [
            model.internlm_model.model.embed_tokens(seg)
            for seg in prompt_seg_tokens
        ]
        prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
        prompt_embs = torch.cat(prompt_seg_embs, dim=1)
        prompt_embs_final.append(prompt_embs)
    
    prompt_embs_final=torch.cat(prompt_embs_final,dim=0)
    outputs = model.internlm_model.generate(
        inputs_embeds=prompt_embs_final.to(device=model.device,dtype=torch.float16),
        max_new_tokens=max_new_tokens,
        num_beams=5,
        do_sample=False,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=0,
        stopping_criteria=stopping_criteria,
    )
    outputs_out=[]
    for m in range(len(outputs)):
        output_token = outputs[m]
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_token[output_token == 0] = 2 #####
        output_token[output_token == -1] = 2
        output_text = model.tokenizer.decode(output_token,
                                              add_special_tokens=False)
        output_text = output_text.split(model.eoa)[0]
        output_text = output_text.split('<|Bot|>')[-1].strip()
        outputs_out.append(output_text)
    return outputs_out

def convert_weights_to_fp32(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp32(l):
        if hasattr(l, 'weight') and l.weight is not None:
            if l.weight.dtype == torch.float16:
                l.weight = l.weight.to(torch.float32)
        if hasattr(l, 'bias') and l.bias is not None:
            if l.bias.dtype == torch.float16:
                l.bias = l.bias.to(torch.float32)

    model.apply(_convert_weights_to_fp32)

class TestInternLM:
    def __init__(self, device=None) -> None:
        self.model = AutoModel.from_pretrained('internlm/internlm-xcomposer-7b', trust_remote_code=True,cache_dir="/home/zhanghao1/.cache/huggingface/hub").cuda().eval()
        self.tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer-7b', trust_remote_code=True)
        self.model.tokenizer = self.tokenizer       

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
            convert_weights_to_fp16(self.model)
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
            self.model.vision_encoder = self.model.to(self.device, dtype=self.dtype)
        self.model = self.model.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256,method=None, level=0):
        if method is not None and level!=0:
            tmp=image.split('/')
            image=os.path.join('/nvme/share/zhanghao/tiny_lvlm_new',tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])
        else:
            tmp=image.split('/')
            image=os.path.join('/nvme/share/zhanghao/',tmp[-3],tmp[-2],tmp[-1])
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                text=f" <|User|>:<ImageHere> {question}" + self.model.eoh + " <|Bot|>:"
                output = generate_answer(self.model, text, image,max_new_tokens=max_new_tokens)
        
        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256,method=None, level=0,gt_answer=None,max_it=None,task_name=None):
                ###
        
        images=[]
        for image in image_list:
            if method is not None and level!=0:
                tmp=image.split('/')
                image=os.path.join('/nvme/share/zhanghao/tiny_lvlm_new',tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])
            else:
                tmp=image.split('/')
                image=os.path.join('/nvme/share/zhanghao/',tmp[-3],tmp[-2],tmp[-1])
            images.append(image)
        image_list=images
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
            model_name = "internlm"
            print("nn",image_list[ind]) 
            vis_proc=[generate_answer,self.model.eoh,task_name]

            image=np.asarray(imgs[ind].resize((224, 224), resample=Image.BICUBIC))
            tmp=image_list[ind].split('/')

            
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
                print("kld_2",end_time2-start_time2)
            else:
                aux_dist[1].append(0)
                
            patch_dist = aux_dist[1][-1]
            patch_remain_access = int(return_1) - patch_used_step
            check_2 = adversarial_patch_check(int(return_1) - patch_used_step)  
            print("dld",check_1,check_2,patch_used_step)

            if check_1 and check_2:   #允许攻击 patch+boundary
                initial_time = time.time()
                start_time3 = time.time()
                attacker = EvolutionaryAttack(model_att,task_name,label)#patch_adversarial_1
                temp_adv_list[2] =  attacker.attack(image, label, patch_adversarial_1, initial_time, time_limit=99999999, 
                  iterations=patch_remain_access, source_step=3e-3, spherical_step=1e-1, rescale_or_not=2, rate = 0.2, big_size=(image.shape[0],image.shape[1]), center_size=(image.shape[0]*40/64,image.shape[1]*40/64), mode='untargeted', question_list=question_list[ind], chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)
                
                aux_dist[2].append(l2_distance(temp_adv_list[2][0], image))
                end_time3 = time.time()
                
                print("kld_3",end_time3-start_time3)
            else:
                aux_dist[2].append(0)
                

           
            if check_1 and check_2:  #patch_adversarial_1
                start_time4= time.time()
                attacker = SurFree(steps=patch_remain_access, max_queries=patch_remain_access,task=task_name,label=label)
                config = json.load(open("models/config_example.json", "r"))
                new_image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
                new_starting_points = torch.tensor(patch_adversarial_1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
                # print(new_image.shape,"pps")
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




