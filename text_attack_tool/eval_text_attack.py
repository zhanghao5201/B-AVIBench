import os
import json
import argparse
import datetime
import torch
print(torch.__version__)
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor


import numpy as np
import pdb
from models import get_model
from utils import dataset_task_dict
from tiny_datasets import dataset_class_dict, GeneralDataset
import logging

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
    ####
    parser.add_argument('--attack', type=str, nargs='+', default='deepwordbug')
    parser.add_argument('--output_dir', type=str, default='./')                                                                        
    parser.add_argument('--prompt_selection', action='store_true')
    parser.add_argument('--shot', type=int, default=0)
    parser.add_argument('--generate_len', type=int, default=30)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--task", type=str, default='OCR')
    parser.add_argument('--query_budget', type=float, default=float("inf"))
    ####

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
########
def create_logger(log_path):

    logging.getLogger().handlers = []

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger
########
def prompt_selection(logger, inference_model, prompts,eval_function,dataset, model_name, dataset_name, task_type, time, batch_size, answer_path):
    prompt_dict = {}

    for prompt in prompts:
        acc= eval_function(inference_model, dataset, args.model_name, dataset_name, task_type, time, args.batch_size, answer_path=answer_path, question=prompt)
        prompt_dict[prompt] = acc
        logger.info("{:.2f}, {}\n".format(acc*100, prompt))

    sorted_prompts = sorted(prompt_dict.items(), key=lambda x:x[1], reverse=True)
    return sorted_prompts

def prompt_selection_ave(logger, inference_model, prompts,eval_function,dataset, model_name, dataset_name, task_type, time, batch_size, answer_path):
    prompt_dict = {}

    len_=len(prompts)
    print(len_)

    acc=[]
    for prompt in prompts:
        acc_= eval_function(inference_model, dataset, args.model_name, dataset_name, task_type, time, args.batch_size, answer_path=answer_path, question=prompt)
        acc.append(acc_)

    largest_nine = sorted(acc, reverse=True)[:9]
    average = sum(largest_nine) / len(largest_nine)
    ave=average
    logger.info("ave: {:.2f},\n".format(ave*100))

    return ave


def attack(args, inference_model, RESULTS_DIR,eval_function,dataset, model_name, dataset_name, task,task_type, time, batch_size, answer_path):
    from prompt_attack.attack import create_attack
    from prompt_attack.goal_function import create_goal_function
    if args.attack == "semantic":
        from prompts.lvlm_semantic_prompts import SEMANTIC_ADV_PROMPT_SET
        from prompts.zero_shot.lvlmtask import LVLM_PROMPT_SET
        if task=="classification" or task=="vqa":
            prompts_ys=LVLM_PROMPT_SET[dataset_name]
        else:
            prompts_ys=LVLM_PROMPT_SET[task]
        average_ys = prompt_selection_ave(args.logger, inference_model, prompts_ys,eval_function,dataset, model_name, dataset_name, task_type, time, batch_size, answer_path)

        if task=="classification" or task=="vqa":
            prompts_dict = SEMANTIC_ADV_PROMPT_SET[dataset_name]
        else:
            prompts_dict = SEMANTIC_ADV_PROMPT_SET[task]
        
        for language in prompts_dict.keys():
            prompts = prompts_dict[language]
            average = prompt_selection_ave(args.logger, inference_model, prompts,eval_function,dataset, model_name, dataset_name, task_type, time, batch_size, answer_path)

            acc_drop= average_ys - average
            args.logger.info("Language: {}, accys: {:.2f}, accnew: {:.2f}, accdrop: {:.2f}%\n".format(language, average_ys*100, average*100, acc_drop*100))
                
            with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
                f.write("Language: {}, accys: {:.2f}, accnew: {:.2f}, accdrop: {:.2f}%\n".format(language, average_ys*100, average*100, acc_drop*100))

    else:
        if args.shot == 0:
            from prompts.zero_shot.lvlmtask import LVLM_PROMPT_SET

        goal_function =create_goal_function(args, inference_model, eval_function,dataset, args.model_name, dataset_name,task, task_type, time, args.batch_size, answer_path=answer_path)

        if task=="classification" or task=="vqa":
            prompts=LVLM_PROMPT_SET[dataset_name]
        else:
            prompts=LVLM_PROMPT_SET[task]
        attack = create_attack(args, goal_function)        
 
        sorted_prompts = prompt_selection(args.logger, inference_model, prompts,eval_function,dataset, model_name, dataset_name, task_type, time, batch_size, answer_path)
        if args.prompt_selection:
            for prompt, acc in sorted_prompts:
                args.logger.info("Prompt: {}, acc: {:.2f}%\n".format(prompt, acc*100))
                with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
                    f.write("Prompt: {}, acc: {:.2f}%\n".format(prompt, acc*100))

        for init_prompt, init_acc in sorted_prompts[:3]:
            if init_acc > 0:
                init_acc, attacked_prompt, attacked_acc, dropped_acc = attack.attack(init_prompt)
                args.logger.info("Original prompt: {}".format(init_prompt))
                args.logger.info("Attacked prompt: {}".format(attacked_prompt.encode('utf-8')))
                args.logger.info("Original acc: {:.2f}%, attacked acc: {:.2f}%, dropped acc: {:.2f}%".format(init_acc*100, attacked_acc*100, dropped_acc*100))
                with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
                    f.write("Original prompt: {}\n".format(init_prompt))
                    f.write("Attacked prompt: {}\n".format(attacked_prompt.encode('utf-8')))
                    f.write("Original acc: {:.2f}%, attacked acc: {:.2f}%, dropped acc: {:.2f}%\n\n".format(init_acc*100, attacked_acc*100, dropped_acc*100))
            else:
                with open(RESULTS_DIR+args.save_file_name+".txt", "a+") as f:
                    f.write("Init acc is 0, skip this prompt\n")
                    f.write("Original prompt: {}\n".format(init_prompt))
                    f.write("Original acc: {:.2f}% \n\n".format(init_acc*100, init_prompt))


def main(args):
    model_lvlm = get_model(args.model_name, device=torch.device('cuda'))
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    answer_path = f"{args.answer_path}/{args.model_name}"

    result = {}
    dataset_names = args.dataset_name.split(',')
    print(args.attack)
    at_name_list=args.attack
    for at_name in at_name_list:
        args.attack=at_name
        for dataset_name in dataset_names:
            save_dir = f"{args.model_name}/{args.task}/{dataset_name}/"
            LOGS_DIR = os.path.join(args.output_dir, "textattack_logs/" + save_dir)
            RESULTS_DIR = os.path.join(args.output_dir, "textattack_results/" + save_dir)
            for DIR in [LOGS_DIR, RESULTS_DIR]:
                if not os.path.isdir(DIR):
                    os.makedirs(DIR)
            file_name = args.model_name + '_' + args.attack + "_gen_len_" + str(args.generate_len) + "_" + str(args.shot) + "_shot"
            args.save_file_name = file_name
            logger = create_logger(LOGS_DIR+file_name+".log")
            logger.info(args)
            eval_function, task_type = dataset_task_dict[dataset_name]
            dataset = GeneralDataset(dataset_name)            

            args.logger = logger
            attack(args, model_lvlm, RESULTS_DIR,eval_function,dataset, args.model_name, dataset_name, args.task, task_type, time, args.batch_size, answer_path=answer_path)

        result_path = os.path.join(os.path.join(answer_path, time), 'result.json')
        with open(result_path, "w") as f:
            f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)