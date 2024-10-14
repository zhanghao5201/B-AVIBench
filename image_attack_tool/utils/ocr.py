import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle

from .tools import has_word, remove_special_chars


def evaluate_OCR(
    model,
    dataset,
    model_name,
    dataset_name,
    task_type,
    time,
    batch_size=1,
    answer_path='./answers',
    question='what is written in the image?',
    method=None, level=0
):
    predictions=[]
    attack_noise=[]
    attack_patch=[]
    attack_patch_boundary=[]
    attack_patch_SurFree=[]
    index_attack=[]
    attack_success=[]
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
     ###
    data_new=[]
    if model_name=="LLaVA15": 
        new_dataset_path = f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/LLaVA15/noise/{dataset_name}'
        if not os.path.exists(new_dataset_path):
            os.makedirs(new_dataset_path, exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/LLaVA15/patch/{dataset_name}', exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/LLaVA15/boundary/{dataset_name}', exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/LLaVA15/SurFree/{dataset_name}', exist_ok=True)
    elif model_name=="OFv2": 
        new_dataset_path = f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/OFv2/noise/{dataset_name}'
        if not os.path.exists(new_dataset_path):
            os.makedirs(new_dataset_path, exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/OFv2/patch/{dataset_name}', exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/OFv2/boundary/{dataset_name}', exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/OFv2/SurFree/{dataset_name}', exist_ok=True)
    elif model_name=="internlm-xcomposer": 
        new_dataset_path = f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/internlm-xcomposer/noise/{dataset_name}'
        if not os.path.exists(new_dataset_path):
            os.makedirs(new_dataset_path, exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/internlm-xcomposer/patch/{dataset_name}', exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/internlm-xcomposer/boundary/{dataset_name}', exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/internlm-xcomposer/SurFree/{dataset_name}', exist_ok=True)
    elif model_name=="Qwen": 
        new_dataset_path = f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/Qwen/noise/{dataset_name}'
        if not os.path.exists(new_dataset_path):
            os.makedirs(new_dataset_path, exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/Qwen/patch/{dataset_name}', exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/Qwen/boundary/{dataset_name}', exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/Qwen/SurFree/{dataset_name}', exist_ok=True)
    elif model_name=="moellava" : 
        new_dataset_path = f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/moellava/noise/{dataset_name}'
        if not os.path.exists(new_dataset_path):
            os.makedirs(new_dataset_path, exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/moellava/patch/{dataset_name}', exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/moellava/boundary/{dataset_name}', exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/moellava/SurFree/{dataset_name}', exist_ok=True)
    elif model_name=="sharegpt4v" : 
        new_dataset_path = f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/sharegpt4v/noise/{dataset_name}'
        if not os.path.exists(new_dataset_path):
            os.makedirs(new_dataset_path, exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/sharegpt4v/patch/{dataset_name}', exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/sharegpt4v/boundary/{dataset_name}', exist_ok=True)
            os.makedirs(f'/mnt/petrelfs/zhanghao1/tiny_attack_datasets/sharegpt4v/SurFree/{dataset_name}', exist_ok=True)
    ###
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = model.batch_generate(batch['image_path'], [question for _ in range(len(batch['image_path']))],method=method, level=level,gt_answer=batch['gt_answers'],task_name="ocr")
        index_attack=index_attack+outputs[1]
        attack_success=attack_success+outputs[2]
        attack_noise=attack_noise+outputs[0][0]
        attack_patch=attack_patch+outputs[0][1]
        attack_patch_boundary=attack_patch_boundary+outputs[0][2]
        attack_patch_SurFree=attack_patch_SurFree+outputs[0][3]
        ###
        if model_name=="LLaVA15" or model_name=="OFv2" or model_name=="internlm-xcomposer" or model_name=="Qwen":             
            for k in range(len(outputs[2])):
                if outputs[2][k]>0:
                    sample={}
                    sample['question']=batch['question'][k]
                    sample['gt_answers']=batch['gt_answers'][k]
                    sample['image_path']=batch['image_path'][k]
                    data_new.append(sample)
            with open(f"{new_dataset_path}/dataset.pkl", 'wb') as f:
                pickle.dump(data_new, f)
        ####        
    if sum(index_attack)!=0 and sum(attack_success)!=0:
        metrics = {
        'success_rate': sum(attack_success)/sum(index_attack),
        "attack_num": sum(index_attack),
        "attack_noise":sum(attack_noise)/sum(attack_success),
        "attack_patch":sum(attack_patch)/sum(attack_success),
        "attack_patch_boundary":sum(attack_patch_boundary)/sum(attack_success),
        "attack_patch_SurFree":sum(attack_patch_SurFree)/sum(attack_success),
    }
    else:
        metrics = {
        'success_rate': -100,
        "attack_num": sum(index_attack),
        "attack_noise":-100,
        "attack_patch":-100,
        "attack_patch_boundary":-100,
        "attack_patch_SurFree":-100,
    }
    return metrics