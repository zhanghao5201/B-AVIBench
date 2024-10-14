import os
import json
import argparse
import datetime
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import torch
import numpy as np
from bat.attacks import SimBA
from models import get_model
from utils import dataset_task_dict
from tiny_datasets import dataset_class_dict, GeneralDataset

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


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    print(torch.__version__)
    model = get_model(args.model_name, device=torch.device('cuda'))
    # print(model)
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = f"{args.answer_path}/{args.model_name}"

    result = {}
    dataset_names = args.dataset_name.split(',')
    for dataset_name in dataset_names:
        print("ll",dataset_name)
        eval_function, task_type = dataset_task_dict[dataset_name]
        # dataset = dataset_class_dict[dataset_name]()
        # dataset = sample_dataset(dataset, args.sample_num, args.sample_seed)
        dataset = GeneralDataset(dataset_name)
        # for method_name in method:     
        
        metrics = eval_function(model, dataset, args.model_name, dataset_name, task_type, time, args.batch_size, answer_path=answer_path, method=None, level=0)
        print("***---",metrics)
        result["{}_severity_{}".format(dataset_name,0)] = metrics
                # result[dataset_name] = metrics

        result_path = os.path.join(os.path.join(answer_path, time), 'result_{}.json'.format(dataset_name))
        with open(result_path, "w") as f:
            f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)