import os
import importlib
from gradio_client import Client
import collections
import clip
import torch
import os
from . import get_BGR_image, DATA_DIR
from . import llama_adapter_v2 as llama



llama_dir = f'{DATA_DIR}/llama_checkpoints'
model_path = f'{DATA_DIR}/llama_checkpoints/llama_adapter_v2_LORA-BIAS-7B.pth' # llama_adapter_v2_BIAS-7B.pth, llama_adapter_v2_0518.pth


from gradio_client import Client
class TestLLamaAdapterV2_web:
    def __init__(self) -> None:
        self.model = Client("http://llama-adapter.opengvlab.com/")

    def generate(self, image: str, question: str, max_length=128, temperature=0, top_p=0.75):
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
    def batch_generate(self, image_list, question_list, max_new_tokens=128,method=None, level=0):
                ###
        images=[]
        for image in image_list:
            if method is not None and level!=0:
                tmp=image.split('/')
                image=os.path.join('/nvme/share/zhanghao/tiny_lvlm_new',tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])
            images.append(image)
        image_list=images
        ####
        imgs = [get_BGR_image(img) for img in image_list]
        imgs = [self.img_transform(x) for x in imgs]
        imgs = torch.stack(imgs, dim=0).to(self.device)
        prompts = [llama.format_prompt(question) for question in question_list]
        # print("op",max_new_tokens)
        results = self.model.generate(imgs, prompts, max_gen_len=max_new_tokens)
        results = [result.strip() for result in results]

        return results

    