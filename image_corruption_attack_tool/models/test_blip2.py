import torch
import contextlib
from types import MethodType
from lavis.models import load_model_and_preprocess
from lavis.models.eva_vit import convert_weights_to_fp16
from . import get_image
import collections

import os

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
            # convert_weights_to_fp16(self.model.visual_encoder)
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
    def batch_generate(self, image_list, question_list, max_new_tokens=30,method=None, level=0):
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
        imgs = [self.vis_processors["eval"](x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device, dtype=self.dtype)
        # print(len(question_list))#8
        prompts = [f"Question: {question} Answer:" for question in question_list]
        # print(len(prompts),"lk")#8
        output = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)

        return output
    