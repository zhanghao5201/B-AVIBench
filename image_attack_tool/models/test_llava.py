import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPImageProcessor, CLIPVisionModel, StoppingCriteria
from .llava import LlavaMPTForCausalLM, LlavaLlamaForCausalLM, conv_templates, SeparatorStyle
from . import get_image

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
####
import eagerpy as ep
from torchvision.transforms.functional import to_pil_image
import numpy as np
import os
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

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

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def get_model_name(model_path):
    # get model name
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        model_name = model_paths[-2] + "_" + model_paths[-1]
    else:
        model_name = model_paths[-1]
    
    return model_name


def get_conv(model_name):
    if "llava" in model_name.lower():
        if "v1" in model_name.lower():
            template_name = "llava_v1"
        elif "mpt" in model_name.lower():
            template_name = "mpt_multimodal"
        else:
            template_name = "multimodal"
    elif "mpt" in model_name:
        template_name = "mpt_text"
    elif "koala" in model_name: # Hardcode the condition
        template_name = "bair_v1"
    elif "v1" in model_name:    # vicuna v1_1/v1_2
        template_name = "vicuna_v1_1"
    else:
        template_name = "v1"
    return conv_templates[template_name].copy()


def load_model(model_path, model_name, dtype=torch.float16, device='cpu'):
    # get tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if 'llava' in model_name.lower():
        if 'mpt' in model_name.lower():
            model = LlavaMPTForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)
    elif 'mpt' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True)

    # get image processor
    image_processor = None
    if 'llava' in model_name.lower():
        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=dtype)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.get_model().vision_tower[0]
        if vision_tower.device.type == 'meta':
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True).to(device=device)
            model.get_model().vision_tower[0] = vision_tower
        else:
            vision_tower.to(device=device, dtype=dtype)
        
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    model.to(device=device)

    return tokenizer, model, image_processor, context_len


class TestLLaVA:
    def __init__(self, device=None):
        model_path="liuhaotian/LLaVA-Lightning-MPT-7B-preview"
        model_name = get_model_name(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_model(model_path, model_name)
        self.conv = get_conv(model_name)
        self.image_process_mode = "Resize" # Crop, Resize, Pad

        if device is not None:
            self.move_to_device(device)
        
    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.dtype = torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        vision_tower = self.model.get_model().vision_tower[0]
        vision_tower.to(device=self.device, dtype=self.dtype)
        self.model.to(device=self.device, dtype=self.dtype)
    
    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256,method=None, level=0):
        image = get_image(image)
        if method is not None and level!=0:
            image = Image.fromarray(d[method](np.asarray(image),level).astype(np.uint8))
        conv = self.conv.copy()
        text = question + '\n<image>'
        text = (text, image, self.image_process_mode)
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        output = self.do_generate([prompt], [image], stop_str=stop_str, dtype=self.dtype, max_new_tokens=max_new_tokens)[0]

        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256,method=None, level=0,gt_answer=None,max_it=None,task_name=None):
        images, prompts = [], []
        for image, question in zip(image_list, question_list):
            if method is not None and level!=0:
                tmp=image.split('/')
                image=os.path.join('/nvme/share/zhanghao/tiny_lvlm_new',tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])#,tmp[-1]
            image = get_image(image)
            conv = self.conv.copy()
            text = question + '\n<image>'
            text = (text, image, self.image_process_mode)
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
            images.append(image)
        imgs = images
        question_list= prompts
        stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else conv.sep2
        aux_dist = []
        temp_adv_list = [] 
        index_attack = []
        attack_success = []
        for list_counter in range(4):
            aux_dist.append([])
            temp_adv_list.append([])
        for ind in range(len(gt_answer)):   
            model_att =  self.do_generate    
            model_name = "llava"
            vis_proc=[stop_str,method,level]
            
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
            if check_1 :   
                start_time2 = time.time()
                attacker = PatchAttack(model_att,task_name,label)
                patch_adversarial_1, patch_used_step=attacker.attack(image, label, adversarial_ori_1, int(return_1), mode='untargeted', question_list=question_list[ind], chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)
                
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
            else:
                aux_dist[2].append(0)           
            if check_1 and check_2:  #patch_adversarial_1
                start_time4= time.time()
                attacker = SurFree(steps=patch_remain_access, max_queries=patch_remain_access,task=task_name,label=label)
                config = json.load(open("models/config_example.json", "r"))
                new_image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
                new_starting_points = torch.tensor(patch_adversarial_1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
                temp_result = attacker(model_att, new_image, starting_points=new_starting_points, **config["run"], question_list=question_list[ind], chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)

                temp_adv_list[3] = np.array(temp_result[3])
                aux_dist[3].append(l2_distance(temp_adv_list[3], image))
                end_time4 = time.time()
                print("kld_4",end_time4-start_time4)
            else:
                aux_dist[3].append(0) 
        batch_outputs = [aux_dist,index_attack,attack_success]     
        return batch_outputs

    @torch.no_grad()
    def do_generate(self, prompts, images, dtype=torch.float16, temperature=0, max_new_tokens=256, stop_str=None, keep_aspect_ratio=False,method=None, level=0,image_listnew=None):
        if keep_aspect_ratio:
            new_images = []
            for image, prompt in zip(images, prompts):
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 448, 224
                shortest_edge = int(min(max_len / aspect_ratio, min_len))
                image = self.image_processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
                new_images.append(image.to(self.model.device, dtype=dtype))
                cur_token_len = (image.shape[1]//14) * (image.shape[2]//14)
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * cur_token_len
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token, 1)
            images = new_images
        else:            
            images = self.image_processor(images, return_tensors='pt')['pixel_values']
            images = images.to(self.model.device, dtype=dtype)
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256    # HACK: 256 is the max image token length hacked
            if getattr(self.model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompts = [prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token) for prompt in prompts]

        stop_idx = None
        if stop_str is not None:
            stop_idx = self.tokenizer(stop_str).input_ids
            if len(stop_idx) == 1:
                stop_idx = stop_idx[0]
            else:
                stop_idx = None

        input_ids = self.tokenizer(prompts).input_ids
        batch_size = len(input_ids)
        min_prompt_size = min([len(input_id) for input_id in input_ids])
        max_prompt_size = max([len(input_id) for input_id in input_ids])
        for i in range(len(input_ids)):
            padding_size = max_prompt_size - len(input_ids[i])
            # input_ids[i].extend([self.tokenizer.pad_token_id] * padding_size)
            input_ids[i] = [self.tokenizer.pad_token_id] * padding_size + input_ids[i]
        
        output_ids = []
        get_result = [False for _ in range(batch_size)]
        for i in range(max_new_tokens):
            if i == 0:
                out = self.model(
                    torch.as_tensor(input_ids).to(self.model.device),
                    use_cache=True,
                    images=images)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = self.model(input_ids=token,
                            use_cache=True,
                            attention_mask=torch.ones(batch_size, past_key_values[0][0].shape[-2] + 1, device=self.model.device),
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[:, -1]
            if temperature < 1e-4:
                token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            token = token.long().to(self.model.device)

            output_ids.append(token)
            for idx in range(len(token)):
                if token[idx] == stop_idx or token[idx] == self.tokenizer.eos_token_id:
                    get_result[idx] = True
            if all(get_result):
                break
        
        output_ids = torch.cat(output_ids, dim=1).long()
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if stop_str is not None:
            for i in range(len(outputs)):
                pos = outputs[i].rfind(stop_str)
                if pos != -1:
                    outputs[i] = outputs[i][:pos]
        
        return outputs