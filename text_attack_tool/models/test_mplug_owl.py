import torch
from .mplug_owl.processing_mplug_owl import MplugOwlProcessor, MplugOwlImageProcessor
from .mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from .mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from transformers import AutoTokenizer
from . import get_image
import collections
import os


prompt_template = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {}\nAI:"


class TestMplugOwl:
    def __init__(self, device=None):
        model_path='MAGAer13/mplug-owl-llama-7b'
        self.model = MplugOwlForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
        self.image_processor = MplugOwlImageProcessor.from_pretrained(model_path)
        # self.tokenizer = MplugOwlTokenizer.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)
        
        # from peft import LoraConfig, get_peft_model
        # peft_config = LoraConfig(
        #     target_modules=r'.*language_model.*\.(q_proj|v_proj)', 
        #     inference_mode=False, 
        #     r=8,
        #     lora_alpha=32, 
        #     lora_dropout=0.05
        # )
        # self.model = get_peft_model(self.model, peft_config)
        # self.model.print_trainable_parameters()
        # exit()

        self.model.eval()
        if device is not None:
            self.move_to_device(device)
        
    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            self.device = device
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.device = 'cpu'
            self.dtype = torch.float32
        self.model.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def generate(self, image, question, max_new_tokens=256,method=None, level=0):
        prompts = [prompt_template.format(question)]
        image = get_image(image)
        inputs = self.processor(text=prompts, images=[image], return_tensors='pt')
        inputs = {k: v.to(self.device, dtype=self.dtype) if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': max_new_tokens
        }

        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        generated_text = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

        return generated_text

    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256,method=None, level=0):
                ###
        images=[]
        for image in image_list:
            if method is not None and level!=0:
                tmp=image.split('/')
                image=os.path.join('/nvme/share/zhanghao/tiny_lvlm_new',tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])
            images.append(image)
        image_list=images
        ####
        images = [get_image(image) for image in image_list]
        images = [self.image_processor(image, return_tensors='pt').pixel_values for image in images]
        images = torch.cat(images, dim=0).to(self.device, dtype=self.dtype)
        prompts = [prompt_template.format(question) for question in question_list]
        inputs = self.processor(text=prompts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = images

        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': max_new_tokens
        }

        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in res.tolist()]

        return outputs

