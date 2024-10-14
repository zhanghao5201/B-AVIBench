import torch
from torch import nn
from huggingface_hub import hf_hub_download
from PIL import Image
import collections
from open_flamingo import create_model_and_transforms

d = collections.OrderedDict()

import os

class OFv2(nn.Module):
    def __init__(self, version: str='3BI',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        super().__init__()
        if version == '3BI':
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
                tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b-dolly",
                cross_attn_every_n_layers=1
            )
            # grab model checkpoint from huggingface hub
            checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "checkpoint.pt")
        elif version == '4BI':
            model, image_processor, tokenizer = create_model_and_transforms(
                clip_vision_encoder_path="ViT-L-14",
                clip_vision_encoder_pretrained="openai",
                lang_encoder_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
                tokenizer_path="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
                cross_attn_every_n_layers=2
                #/nvme/share/huggingface_cache
            )
            checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct", "checkpoint.pt",cache_dir="/home/xupeng/.cache/huggingface")
        else:
            raise ValueError(f'OpenFlamingo v2 {version} NOT supported yet!')
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        self.model = model.eval()
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        if device is not None:
            self.move_to_device(device)

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            # print("*************************************************************")
            self.dtype = torch.float16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        self.model = self.model.to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens: int=20,method=None, level=0,*args, **kwargs):
        # images=[]
        # for image in image_list:
        #     if method is not None and level!=0:
        #         tmp=image.split('/')
        #         image=os.path.join('/nvme/share/zhanghao/tiny_lvlm_new',tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])
        #     images.append(image)
        # image_list=images
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

        if type(image_list[0]) is not str:
            images = [Image.fromarray(x) for x in image_list]
        else:
            images = [Image.open(img).convert('RGB') for img in image_list]
        vision_x = [self.image_processor(x).unsqueeze(0).unsqueeze(0).unsqueeze(0) for x in images]
        vision_x = torch.cat(vision_x, dim=0).to(self.device, dtype=self.dtype)
        prompt_template = kwargs.get('prompt_template', 'OFv2_vqa')
        # print(prompt_template,"ops------")
        # print(vision_x.shape,"kl")
        # print(question_list,len(question_list))
        if prompt_template == 'OFv2_vqa':
            prompts = [f"<image>Question: {x} Short answer:" for x in question_list]
        else:
            prompts = [f"<image>{x}" for x in question_list]
        lang_x = self.tokenizer(
            prompts,
            return_tensors="pt", padding=True,
        ).to(self.device)
        # print("op1",lang_x.shape)
        generated_text = self.model.generate(
            vision_x=vision_x.to(self.device, dtype=self.dtype),
            lang_x=lang_x["input_ids"].to(self.device),
            attention_mask=lang_x["attention_mask"].to(self.device, dtype=self.dtype),
            max_new_tokens=max_new_tokens,
            num_beams=3,pad_token_id=self.tokenizer.eos_token_id
        )
        outputs = self.tokenizer.batch_decode(generated_text, skip_special_tokens=True)
        results = [y[len(x)-len('<image>'):].strip() for x, y in zip(prompts, outputs)]
        return results