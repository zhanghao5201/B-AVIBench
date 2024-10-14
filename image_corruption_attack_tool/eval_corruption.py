import os
import json
import argparse
import datetime
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import torch
import numpy as np
import random
from models import get_model
from utils import dataset_task_dict
from tiny_datasets import dataset_class_dict, GeneralDataset
import socket
import deepspeed

method = ['Fog','Zoom_Blur','Glass_Blur','Gaussian_Noise','Shot_Noise','Impulse_Noise','Defocus_Blur','Motion_Blur','Snow',
'Frost','Brightness','Contrast','Elastic','Pixelate','JPEG','Speckle_Noise','Gaussian_Blur','Spatter','Saturate'] 
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)

    # datasets
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--sample_seed", type=int, default=20230719)

    # result_path
    parser.add_argument("--answer_path", type=str, default="./tiny_answers")

    args = parser.parse_args()
    return args


def sample_dataset(dataset, max_sample_num=5000, seed=0):
    if max_sample_num == -1:
        return dataset

    if len(dataset) > max_sample_num:
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        dataset = torch.utils.data.Subset(dataset, random_indices)
    return dataset

def initialize_distributed():
    os.environ['MASTER_IP'] = os.getenv('MASTER_ADDR', 'localhost')
    
    # 生成一个随机端口号
    port = random.randint(10000, 60000)
    
    # 检查随机端口号是否可用
    while True:
        try:
            # 创建一个临时的socket对象并尝试绑定到随机端口号
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
            break
        except OSError:
            # 如果端口号已被占用，则选择一个新的随机端口号
            port = random.randint(10000, 60000)
    
    # 将随机端口号设置为环境变量
    os.environ['MASTER_PORT'] = str(port)
    
    # 初始化分布式训练
    deepspeed.init_distributed(dist_backend='nccl')

def main(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    # os.environ['DS_INIT_PROCESS_PORT'] = str(10008)
    # 将随机端口号设置为环境变量    os.environ['MASTER_PORT'] = str(port)
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = f"{args.answer_path}/{args.model_name}"
    model = get_model(args.model_name, device=torch.device('cuda')) 
       

    result = {}
    dataset_names = args.dataset_name.split(',')
    for dataset_name in dataset_names:
        eval_function, task_type = dataset_task_dict[dataset_name]      
        
        for method_name in method:     
            for k in [0,1,3,5]:#                
                dataset = GeneralDataset(dataset_name) 
                metrics = eval_function(model, dataset, args.model_name, dataset_name, task_type, time, args.batch_size, answer_path=answer_path, method=method_name, level=k)
                result["{}_severity_{}_{}".format(dataset_name,method_name,k)] = metrics                
    result_path = os.path.join(os.path.join(answer_path, time), 'result.json')    
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()

    main(args)