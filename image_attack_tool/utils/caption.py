import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle

from .cider import CiderScorer

"""
NOTE: caption prompt candidates
1. what is described in the image?
2. Generate caption of this image:
"""



def evaluate_Caption(
    model,
    dataset,
    model_name,
    dataset_name,
    task_type,
    time,
    batch_size=1,
    answer_path='./answers',
    question='what is described in the image?',
    max_new_tokens=16,
    method=None, level=0
):
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    attack_noise=[]
    attack_patch=[]
    attack_patch_boundary=[]
    attack_patch_SurFree=[]
    index_attack=[]
    attack_success=[]
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = model.batch_generate(batch['image_path'], [question for _ in range(len(batch['image_path']))], max_new_tokens=max_new_tokens,method=method, level=level,gt_answer=batch['gt_answers'],max_it=10,task_name="caption")
        index_attack=index_attack+outputs[1]
        attack_success=attack_success+outputs[2]
        attack_noise=attack_noise+outputs[0][0]
        attack_patch=attack_patch+outputs[0][1]
        attack_patch_boundary=attack_patch_boundary+outputs[0][2]
        attack_patch_SurFree=attack_patch_SurFree+outputs[0][3]

    if sum(index_attack)!=0:
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


