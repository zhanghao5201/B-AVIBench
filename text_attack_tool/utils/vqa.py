import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import re
from .tools import VQAEval
import pdb

def evaluate_VQA(
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
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    for batch in tqdm(dataloader, desc="Running inference"):
        # print(dataset_name,task_type)
        if dataset_name=="IconQA":
            tmp_que=[]
            # print(question,"pl")
            # pdb.set_trace()
            for k in range(len(batch['question'])):
                tmp=batch['question'][k].split('Choose the best answer from the following choices:')
                # print(tmp,"00")
                tmp_f=question.format(tmp[0],tmp[1])
                # print(tmp_f,"001")
                tmp_que.append(tmp_f)
                # print(tmp_que,"ll")
        elif dataset_name=="ScienceQAIMG":
            tmp_que=[]
            # print(question,"pl")
            # pdb.set_trace()
            for k in range(len(batch['question'])):
                # print("oo",batch['question'][k])
                tmp=batch['question'][k].split('Choose the best answer from the following choices:')
                # print(tmp,"00")
                tmp_f=question.format(tmp[0],tmp[1])
                # print(tmp_f,"001")
                tmp_que.append(tmp_f)
                # print(tmp_que,"ll")
        elif dataset_name=="AOKVQAClose":
            tmp_que=[]
            # print(question,"pl")
            # pdb.set_trace()
            for k in range(len(batch['question'])):
                tmp=batch['question'][k].split('Choose the best answer from the following choices:')
                # print(tmp,"00")
                tmp_f=question.format(tmp[0],tmp[1])
                # print(tmp_f,"001")
                tmp_que.append(tmp_f)
                # print(tmp_que,"ll")
        elif dataset_name=="WHOOPSWeird":
            tmp_que=[]
            # print(question,"pl")
            # pdb.set_trace()
            for k in range(len(batch['question'])):
                # print(batch['question'][k],"nn")
                # tmp=batch['question'][k].split('Choose the best answer from the following choices:')
                # print(tmp,"00")
                tmp_f=question
                # print(tmp_f,"001")
                tmp_que.append(tmp_f)
                # print(tmp_que,"ll")
        else:#object
            tmp_que=[]
            # print(question,"pl")
            # pdb.set_trace()
            for k in range(len(batch['question'])):
                # print(batch['question'][k],"nn")
                pattern = r"Is there (.+?) in the"
                match = re.search(pattern, batch['question'][k])
                # print(match)
                if match:
                    extracted_content = match.group(1)
                    # print(extracted_content)
                else:
                    pdb.set_trace()
                # tmp=batch['question'][k].split()
                # print(tmp,"00")
                tmp_f=question.format(extracted_content)
                # print(tmp_f,"001")
                tmp_que.append(tmp_f)
        # print(batch['question'],"aa \n",tmp_que)
        outputs = model.batch_generate(batch['image_path'], tmp_que, method=method, level=level)
        for image_path, question_new, gt_answer, output in zip(batch['image_path'], tmp_que, batch['gt_answers'], outputs):
            answer_dict={'question': question_new, 'answer': output,
            'gt_answers': gt_answer, 'image_path': image_path,
            'model_name': model_name, 'task_type': task_type}
            predictions.append(answer_dict)
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}_{method}_{level}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    eval = VQAEval()
    correct = 0
    num = 0
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            if eval.evaluate(answer, gt_answers)==1:
                correct+=1
            num+=1
    # print(correct,num)
    print(f'{dataset_name}_{method}_{level}:{float(correct)/num}')##yes
    # pdb.set_trace()
    return float(correct)/num