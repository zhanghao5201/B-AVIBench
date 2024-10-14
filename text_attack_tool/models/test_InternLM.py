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
import collections
import pandas as pd
from torch.utils.data import Dataset
import torchvision
import os
from . import get_image, DATA_DIR



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
        temperature=1.0,
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
        # print(prompt_seg_tokens[1],prompt_seg_tokens[1].shape,"ssss")
    max_prompt_size = max([input_id[1].shape[1] for input_id in pro_token])
    # print(max_prompt_size,"aa")
    for i in range(len(pro_token)):
        padding_size = max_prompt_size - pro_token[i][1].shape[1]        
        # print(pro_token[i][1].shape[1],padding_size,max_prompt_size)
        padding_ids =torch.tensor([0]).repeat(1, padding_size).to(pro_token[i][1][0].device,dtype=pro_token[i][1][0].dtype)
        # print(padding_ids,"pl",pro_token[i][1])        
        pro_token[i][1] = torch.cat((padding_ids,pro_token[i][1]),dim=1)
        # print("s11",pro_token[i][1].shape,pro_token[i][1])

    # print(img_embeds.shape,"a0a")
    outputs_out=[]
    for k in range(len(text)):
        # img_embeds = model.encode_img(image_path[k])
        # print(img_embeds.shape,"aa")
        img_embeds=img_embeds_All[k].unsqueeze(0)
        # prompt_segs = text[k].split('<ImageHere>')
        prompt_seg_tokens = pro_token[k]
        # print(prompt_seg_tokens[1],len(prompt_seg_tokens))      
        prompt_seg_embs = [
            model.internlm_model.model.embed_tokens(seg)
            for seg in prompt_seg_tokens
        ]
        # print(prompt_seg_embs[0].shape, "mm", prompt_seg_embs[1].shape)
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
        temperature=1.0,
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

    # # print ("kl",len(outputs))
    # outputs_out=[]
    # for m in range(len(outputs)):
    #     output_token = outputs[m]
    #     if output_token[0] == 0:
    #         output_token = output_token[1:]
    #     if output_token[0] == 1:
    #         output_token = output_token[1:]
    #     # print(output_token,"op")
    #     output_token[output_token == 0] = 2 #####
    #     output_token[output_token == -1] = 2
    #     # print(output_token,output_token.shape,"kl")
    #     output_text = model.tokenizer.decode(output_token,
    #                                           add_special_tokens=False)
    #     # print("0",output_text)
    #     output_text = output_text.split(model.eoa)[0]
    #     output_text = output_text.split('<|Bot|>')[-1].strip()
    #     # print(output_text)
    #     outputs_out.append(output_text)
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
   

    # print(img_embeds.shape,"a0a")
    for k in range(len(text)):
        # img_embeds = model.encode_img(image_path[k])
        # print(img_embeds.shape,"aa")
        img_embeds=img_embeds_All[k].unsqueeze(0)
        # prompt_segs = text[k].split('<ImageHere>')
        prompt_seg_tokens = pro_token[k]
        # print(prompt_seg_tokens[1],len(prompt_seg_tokens))      
        prompt_seg_embs = [
            model.internlm_model.model.embed_tokens(seg)
            for seg in prompt_seg_tokens
        ]
        # print(prompt_seg_embs[0].shape, "mm", prompt_seg_embs[1].shape)
        prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
        prompt_embs = torch.cat(prompt_seg_embs, dim=1)
        prompt_embs_final.append(prompt_embs)
    
    prompt_embs_final=torch.cat(prompt_embs_final,dim=0)
    # print(prompt_embs_final.shape,prompt_embs_final.dtype,model.dtype,max_new_tokens)
    outputs = model.internlm_model.generate(
        inputs_embeds=prompt_embs_final.to(device=model.device,dtype=torch.float16),
        max_new_tokens=max_new_tokens,
        num_beams=5,
        do_sample=False,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1.0,
        stopping_criteria=stopping_criteria,
    )
    # print ("kl",len(outputs))
    outputs_out=[]
    for m in range(len(outputs)):
        output_token = outputs[m]
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        # print(output_token,"op")
        output_token[output_token == 0] = 2 #####
        output_token[output_token == -1] = 2
        # print(output_token,output_token.shape,"kl")
        output_text = model.tokenizer.decode(output_token,
                                              add_special_tokens=False)
        # print("0",output_text)
        output_text = output_text.split(model.eoa)[0]
        output_text = output_text.split('<|Bot|>')[-1].strip()
        # print(output_text)
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
        # model_path=CKPT_PATH
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
        # image = get_image(image)
        # vision_x = (self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0))
        # lang_x = self.model.text_tokenizer([f"<image> User: {question} GPT: <answer>"], return_tensors="pt")
        # generated_text = self.model.generate(
        #     vision_x=vision_x.to(self.model.device, dtype=self.dtype),
        #     lang_x=lang_x["input_ids"].to(self.model.device),
        #     attention_mask=lang_x["attention_mask"].to(self.model.device, dtype=self.dtype),
        #     max_new_tokens=max_new_tokens,
        #     num_beams=3,
        #     no_repeat_ngram_size=3,
        # )
        # output = self.model.text_tokenizer.decode(generated_text[0])
        # output = [x for x in output.split(' ') if not x.startswith('<')]
        # out_label = output.index('GPT:')
        # output = ' '.join(output[out_label + 1:])
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
                # texts=[f" <|User|>:<ImageHere> {question}" + self.model.eoh + " <|Bot|>:"]
                # output=batch_generate_answer(self.model,texts, [image],max_new_tokens=max_new_tokens)  
        
        return output
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256,method=None, level=0):
                ###
        
        images=[]
        for image in image_list:
            if method is not None and level!=0:
                tmp=image.split('/')
                image=os.path.join('/nvme/share/zhanghao/tiny_lvlm_new',tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])
            else:
                tmp=image.split('/')
                image=os.path.join('/nvme/share/zhanghao/',tmp[-3],tmp[-2],tmp[-1])
            # print(image)
            images.append(image)
        # print(images,"pp")
        # tmp=image.split('/')
    # image=os.path.join('/nvme/share/zhanghao/',tmp[-3],tmp[-2],tmp[-1])
        image_list=images
        total_output = []
        texts=[f" <|User|>:<ImageHere> {question}" + self.model.eoh + " <|Bot|>:" for question in question_list]
        # with torch.cuda.amp.autocast():
        #     with torch.no_grad():
        #         total_output=batch_generate_answer_new(self.model,texts, image_list,max_new_tokens=max_new_tokens)   
        for k in range(len(image_list)):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    # print(question_list[k],"a",image_list[k])
                    # response = self.model.generate(question_list[k], image_list[k])
                    total_output.append(generate_answer(self.model,texts[k], image_list[k],max_new_tokens=max_new_tokens))        
        # print(max_new_tokens,"opo")
        return total_output
    