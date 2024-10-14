import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional
from collections import defaultdict
import pickle

from .tools import has_word, remove_special_chars


def evaluate_zero_shot_image_classification(
    model,
    dataset,
    model_name,
    dataset_name,
    task_type,
    time,
    batch_size=1,
    answer_path='answers',
    question='The photo of the',
    max_new_tokens=16,
    per_class_acc=True,
    method=None, level=0
):
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
     ###
    data_new=[]
    if model_name=="LLaVA15" : 
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
    i = 0
    if dataset_name=="OxfordIIITPet":
        question='What breed is the pet in the image?'
    elif dataset_name=="Flowers102":
        question='What breed is the flower in the image?'
    attack_noise=[]
    attack_patch=[]
    attack_patch_boundary=[]
    attack_patch_SurFree=[]
    index_attack=[]
    attack_success=[]
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = model.batch_generate(batch['image_path'], [question for _ in range(len(batch['image_path']))], max_new_tokens=max_new_tokens,method=method, level=level,gt_answer=batch['gt_answers'],max_it=10,task_name="cls")

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

    
def evaluate_zero_shot_image_classification_ys(
    model,
    dataset,
    model_name,
    dataset_name,
    task_type,
    time,
    batch_size=1,
    answer_path='answers',
    question='The photo of the',
    max_new_tokens=16,
    per_class_acc=True,
    method=None, level=0
):
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    i = 0
    if dataset_name=="OxfordIIITPet":
        question='What breed is the pet in the image?'
    elif dataset_name=="Flowers102":
        question='What breed is the flower in the image?'
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = model.batch_generate(batch['image_path'], [question for _ in range(len(batch['image_path']))], max_new_tokens=max_new_tokens,method=method, level=level)
        j = 0
        for image_path, gt_answer, output in zip(batch['image_path'], batch['gt_answers'], outputs):
            if type(image_path) is not str:
                image_path = f'batch#{i} sample#{j}'
            answer_dict={'question': question, 'answer': output,
            'gt_answers': gt_answer, 'image_path': image_path,
            'model_name': model_name, 'task_type': task_type}
            predictions.append(answer_dict)
            j += 1
        i += 1
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}_{method}_{level}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    correct = 0
    num = 0
    exact_match = 0
    per_class_dict = defaultdict(lambda : defaultdict(int))
    
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            answer = dict[i]['answer']
            answer = remove_special_chars(answer).lower()
            gt_answers = dict[i]['gt_answers']
            
            if type(gt_answers) is str:
                cls_name = gt_answers
                gt_answers = [remove_special_chars(gt_answers).lower()]
            else:
                cls_name = gt_answers[0]
                gt_answers = [remove_special_chars(x).lower() for x in gt_answers]
            per_class_dict[cls_name]['total'] += 1
            if any([has_word(answer, x) for x in gt_answers]):
                per_class_dict[cls_name]['correct'] += 1
                correct+=1
            if any([answer == x for x in gt_answers]):
                exact_match += 1
            num+=1
    acc_has_word = correct / num * 100
    acc_exact_match = exact_match / num * 100

    metrics = {
        'has_word': acc_has_word,
        'exact match': acc_exact_match,
    }
    if per_class_acc:
        num_classes = len(per_class_dict)
        acc_sum = 0.0
        for val in per_class_dict.values():
            acc_sum += val['correct'] / val['total']
        mean_per_class_acc = acc_sum / num_classes * 100
        metrics['mean_per_class_acc'] = mean_per_class_acc
        print(f'{dataset_name}_{method}_{level} of mean per-class: {mean_per_class_acc:.2f}%')
    return metrics