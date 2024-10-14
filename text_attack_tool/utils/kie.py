import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader


class F1Scorer:
    def __init__(self):
        self.n_detected_words = 0
        self.n_gt_words = 0        
        self.n_match_words = 0

    def add_string(self, ref, pred):        
        pred_words = list(pred.split())
        ref_words = list(ref.split())
        self.n_gt_words += len(ref_words)
        self.n_detected_words += len(pred_words)
        for pred_w in pred_words:
            if pred_w in ref_words:
                self.n_match_words += 1
                ref_words.remove(pred_w)

    def score(self):
        prec = self.n_match_words / float(self.n_detected_words) * 100
        recall = self.n_match_words / float(self.n_gt_words) * 100
        
        if prec + recall==0:
            f1=0
        else:
            f1 = 2 * (prec * recall) / (prec + recall)
        return prec, recall, f1

    def result_string(self):
        prec, recall, f1 = self.score()
        return f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f}"


def evaluate_KIE(
    model,
    dataset,
    model_name,
    dataset_name,
    task_type,
    time,
    batch_size=1,
    answer_path='./answers',
    method=None, level=0
):
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    for batch in tqdm(dataloader, desc="Running inference"):
        outputs = model.batch_generate(batch['image_path'], batch['question'],method=method, level=level)
        for image_path, question, gt_answer, output in zip(batch['image_path'], batch['question'], batch['gt_answers'], outputs):
            answer_dict={'question': question, 'answer': output,
            'gt_answers': gt_answer, 'image_path': image_path,
            'model_name': model_name, 'task_type': task_type}
            predictions.append(answer_dict)
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}_{method}_{level}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        f1_scorer = F1Scorer()
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            # print(gt_answers)
            # if isinstance(gt_answers, list):
            #     print("cjfkxlfjjl**************************")
            if isinstance(gt_answers, list) :
                gt_answers =" ".join(gt_answers)
            # print(len(gt_answers),gt_answers)
            answer = dict[i]['answer']
            # print(len(answer),answer)
            f1_scorer.add_string(gt_answers, answer)
        prec, recall, f1 = f1_scorer.score()
    result = f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f}"
    #####F1：yes
    print(f'{dataset_name}_{method}_{level}: {result}')
    return f1#result
