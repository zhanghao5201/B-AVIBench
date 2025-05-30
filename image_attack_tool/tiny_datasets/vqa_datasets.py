import os
import json
import datasets
from torch.utils.data import Dataset

from task_datasets import DATA_DIR

from . import DATA_DIR

class TextVQADataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/TextVQA"

    def __init__(self):
        self.data = json.load(open(f"{self.data_root}/TextVQA_0.5.1_val.json", "r"))["data"]
        self.image_dir_path = self.data_root + '/train_images'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        img_path = os.path.join(self.image_dir_path, f"{self.data[idx]['image_id']}.jpg")
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class DocVQADataset(Dataset):
    data_root = f'{DATA_DIR}/VQA_Datasets/DocVQA/val'

    def __init__(self):
        ann_path = f"{self.data_root}/val_v1.0.json"
        self.data = json.load(open(ann_path, "r"))["data"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        img_path = os.path.join(self.data_root, self.data[idx]['image'])
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class OCRVQADataset(Dataset):
    data_root = f'{DATA_DIR}/VQA_Datasets/OCRVQA'

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        dataset = json.load(open(f'{self.data_root}/dataset.json', "r"))
        for idx, data in enumerate(dataset):
            if dataset[data]['split'] != 2:
                continue
            questions =  dataset[data]['questions']
            for index, question in enumerate(questions):
                img_name = dataset[data]['imageURL'].split('/')[-1]
                image_file = os.path.join(self.data_root, 'images', img_name)
                gt_answers = dataset[data]['answers'][index]
                self.image_list.append(image_file)
                self.answer_list.append(gt_answers)
                self.question_list.append(question)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class STVQADataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/STVQA"

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        data = json.load(open(f"{self.data_root}/train_task_3.json", "r"))['data']
        for i in range(len(data)):
            image_path = self.data_root + '/train_imgs/' + data[i]['dataset'] + '/' + data[i]['file_name']
            self.image_list.append(image_path)
            self.answer_list.append(data[i]['answers'])
            self.question_list.append(data[i]['question'])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}
    

class OKVQADataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/OKVQA"

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        questions = json.load(open(f"{self.data_root}/OpenEnded_mscoco_val2014_questions.json", "r"))['questions']
        question_dict = {x['question_id']: x['question'] for x in questions}
        annotations = json.load(open(f"{self.data_root}/mscoco_val2014_annotations.json", "r"))['annotations']
        for i in range(len(annotations)):
            question = question_dict[annotations[i]['question_id']]
            answers = [x['answer'] for x in annotations[i]['answers']]
            image_path = f"{self.data_root}/val2014/COCO_val2014_000000{annotations[i]['image_id']:06d}.jpg"
            self.answer_list.append(answers)
            self.image_list.append(image_path)
            self.question_list.append(question)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}

class AOKVQADataset(Dataset):
    data_root = f"{DATA_DIR}/A-OKVQA"

    def __init__(self, image_dir: str=f"{DATA_DIR}/MSCOCO/val2014", open_ended: bool=True):
        self.image_dir = image_dir
        self.open_ended = open_ended
        qa_json = f"{self.data_root}/aokvqa_v1p0_val.json"
        self.dataset = json.load(open(qa_json, 'r'))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        question = sample['question']
        image_id = sample['image_id']
        image_path = f'{self.image_dir}/COCO_val2014_{image_id:012d}.jpg'
        if self.open_ended:
            answers = sample['direct_answers']
        else: # close-set i.e., multi-choice selection
            choices = sample['choices']
            correct_choice_idx = sample['correct_choice_idx']
            answers = choices[correct_choice_idx]
            options = '\n'.join(['- ' + x for x in choices])
            question += f'\nChoose the best answer from the following choices:\n{options}\n'
        return {
            "image_path": image_path,
            "question": question,
            "gt_answers": answers,
        }

class AOKVQAOpenDataset(AOKVQADataset):

    def __init__(self, open_ended: bool = True):
        super().__init__(open_ended=open_ended)

class AOKVQACloseDataset(AOKVQADataset):

    def __init__(self, open_ended: bool = False):
        super().__init__(open_ended=open_ended)

class GQADataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/GQA"

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        annotations = json.load(open(f"{self.data_root}/questions/testdev_balanced_questions.json", "r"))
        for sample in annotations:
            sample = annotations[sample]
            image_path = f"{self.data_root}/images/{sample['imageId']}.jpg"
            self.image_list.append(image_path)
            self.answer_list.append(sample['answer'])
            self.question_list.append(sample['question'])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}
    

class VizWizDataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/VizWiz"

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        self.load_data(split='val')
        # self.load_data(split='train')

    def load_data(self, split='val'):
        annotations = json.load(open(f"{self.data_root}/{split}_grounding.json", "r"))
        for image_name in annotations:
            sample = annotations[image_name]
            image_path = f"{self.data_root}/{split}/{image_name}"
            self.image_list.append(image_path)
            self.answer_list.append(sample['answers'])
            self.question_list.append(sample['question'])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class VQAv2Dataset(Dataset):
    data_root = f"{DATA_DIR}/VQAv2"

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        image_dir = f"{DATA_DIR}/MSCOCO"
        questions = json.load(open(f"{self.data_root}/v2_OpenEnded_mscoco_val2014_questions.json", "r"))['questions']
        question_dict = {x['question_id']: x['question'] for x in questions}
        annotations = json.load(open(f"{self.data_root}/v2_mscoco_val2014_annotations.json", "r"))['annotations']
        for i in range(len(annotations)):
            question = question_dict[annotations[i]['question_id']]
            answers = [x['answer'] for x in annotations[i]['answers']]
            image_path = f"{image_dir}/val2014/COCO_val2014_000000{annotations[i]['image_id']:06d}.jpg"
            self.answer_list.append(answers)
            self.image_list.append(image_path)
            self.question_list.append(question)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class VQAv1Dataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/VQAv1"

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        questions = json.load(open(f"{self.data_root}/OpenEnded_mscoco_val2014_questions.json", "r"))['questions']
        question_dict = {x['question_id']: x['question'] for x in questions}
        annotations = json.load(open(f"{self.data_root}/mscoco_val2014_annotations.json", "r"))['annotations']
        for i in range(len(annotations)):
            question = question_dict[annotations[i]['question_id']]
            answers = [x['answer'] for x in annotations[i]['answers']]
            image_path = f"{self.data_root}/val2014/COCO_val2014_000000{annotations[i]['image_id']:06d}.jpg"
            self.answer_list.append(answers)
            self.image_list.append(image_path)
            self.question_list.append(question)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class VisdialDataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/Visdial"

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        data = json.load(open(f"{self.data_root}/visdial_1.0_val.json", "r"))['data']
        for sample in data['dialogs']:
            image_id = sample['image_id']
            image_path = f"{self.data_root}/images_val2018/VisualDialog_val2018_000000{image_id:06d}.jpg"
            # caption = sample['caption']
            dialog = sample['dialog']
            for qa in dialog:
                question = data['questions'][qa['question']]
                # answer = data['answers'][qa['answer']]
                answer_options = [data['answers'][x] for x in qa['answer_options']]
                self.answer_list.append(answer_options)
                self.image_list.append(image_path)
                self.question_list.append(question)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class MSCOCO_POPEDataset_adversarial(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/MSCOCO/val2014",
        ann_path= "utils_data/MSCOCO_POPE/coco_pope_adversarial1.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text']
        answers = self.data[idx]['label']
        # COCO_val2014_000000007991.jpg
        name = str(self.data[idx]['image'])
        img_path = os.path.join(self.image_dir_path,name)
        if os.path.isfile(img_path):
            return {
                "image_path": img_path,
                "question": question,
                "gt_answers": answers}
        else:
            print(img_path, 'not exist!!!')
            return self.__getitem__((idx + 1) % len(self))


class MSCOCO_POPEDataset_popular(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/MSCOCO/val2014",
        ann_path= "utils_data/MSCOCO_POPE/coco_pope_popular1.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text']
        answers = self.data[idx]['label']
        name = str(self.data[idx]['image'])
        
        img_path = os.path.join(self.image_dir_path,name)
        if os.path.isfile(img_path):
            return {
                "image_path": img_path,
                "question": question,
                "gt_answers": answers}
        else:
            print(img_path, 'not exist!!!')
            return self.__getitem__((idx + 1) % len(self))


class MSCOCO_POPEDataset(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/MSCOCO/val2014",
        ann_path= "utils_data/MSCOCO_POPE/coco_pope_random1.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text']
        answers = self.data[idx]['label']
        name = str(self.data[idx]['image'])
        img_path = os.path.join(self.image_dir_path,name)
        if os.path.isfile(img_path):
            return {
                "image_path": img_path,
                "question": question,
                "gt_answers": answers}
        else:
            print(img_path, 'not exist!!!')
            return self.__getitem__((idx + 1) % len(self))


class VCR1_OCDataset(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/VCR/vcr1images",
        ann_path= "utils_data/GVT/vcr1_oc.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text_in']
        answers = self.data[idx]['text_out']
        # COCO_val2014_000000007991.jpg
       
        img_path = os.path.join(self.image_dir_path,str(self.data[idx]['image_id']))
        if os.path.isfile(img_path):
            return {
                "image_path": img_path,
                "question": question,
                "gt_answers": answers}
        else:
            print(img_path, 'not exist!!!')
            return self.__getitem__((idx + 1) % len(self))


class VCR1_MCIDataset(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/VCR/vcr1images",
        ann_path= "utils_data/GVT/vcr1_mci.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text_in']
        answers = self.data[idx]['text_out']
        img_path = os.path.join(self.image_dir_path,str(self.data[idx]['image_id']))
        if os.path.isfile(img_path):
            return {
                "image_path": img_path,
                "question": question,
                "gt_answers": answers}
        else:
            print(img_path, 'not exist!!!')
            return self.__getitem__((idx + 1) % len(self))


class MSCOCO_MCIDataset(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/MSCOCO/val2014",
        ann_path= "utils_data/GVT/coco_mci.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text_in']
        answers = self.data[idx]['text_out']
        name = 'COCO_val2014_' + str(self.data[idx]['image_id']).zfill(len('000000007991')) + '.jpg'
        img_path = os.path.join(self.image_dir_path,name)
        if os.path.isfile(img_path):
            return {
                "image_path": img_path,
                "question": question,
                "gt_answers": answers}
        else:
            print(img_path, 'not exist!!!')
            return self.__getitem__((idx + 1) % len(self))


class MSCOCO_OCDataset(Dataset):
    def __init__(
        self,
        image_dir_path=f"{DATA_DIR}/MSCOCO/val2014",
        ann_path="utils_data/GVT/coco_oc.json"
    ):
        self.data = json.load(open(ann_path, "r"))
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['text_in']
        answers = self.data[idx]['text_out']
        name = 'COCO_val2014_' + str(self.data[idx]['image_id']).zfill(len('000000007991')) + '.jpg'
        img_path = os.path.join(self.image_dir_path,name)
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}



class VSRDataset(Dataset):
    data_root = f"{DATA_DIR}/VQA_Datasets/VSR"
    choices = ['No', 'Yes']

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []

        data = []
        with open(f"{self.data_root}/all_vsr_validated_data.jsonl", "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))
        for sample in data:
            image_path = f"{self.data_root}/images/{sample['image']}"
            question = f"Question: Is the following caption right? {sample['caption']}\n"
            options = '\n'.join(['- ' + x for x in self.choices])
            question += f'Choose the best answer from the following choices:\n{options}\n'
            answer = self.choices[sample['label']]
            self.answer_list.append(answer)
            self.image_list.append(image_path)
            self.question_list.append(question)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class HatefulMemes(Dataset):
    # TODO: prepare the list of Yes/No meaning words

    def __init__(self, data_root: str=f'{DATA_DIR}/hateful_memes') -> None:
        super().__init__()
        self.data_root = data_root
        jsonl_path = f'{self.data_root}/test_seen.jsonl'
        with open(jsonl_path, 'r') as f:
            self.dataset = [json.loads(x) for x in f.readlines()]
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> dict:
        sample = self.dataset[index]
        img = sample['img']
        image_path = f'{self.data_root}/{img}'
        text = sample['text']
        question = f"This is a meme with '{text}' written on it. Is it hateful?"
        gt_answers = "Yes" if sample["label"] == 1 else "No"
        return {
            'image_path': image_path,
            'question': question,
            'gt_answers': gt_answers,
        }




class IconQADataset(Dataset):
    split='test'
    data_root = f'{DATA_DIR}/VQA_Datasets/IconQA'

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []

        dataset_dir = f"{self.data_root}/dataset/{self.split}/choose_txt"
        for sample in os.listdir(dataset_dir):
            image_path = f"{dataset_dir}/{sample}/image.png"
            self.image_list.append(image_path)
            data = json.load(open(f"{dataset_dir}/{sample}/data.json", 'r'))
            question = f"Question: {data['question']}\n"
            options = '\n'.join(['- ' + x for x in data['choices']])
            question += f'Choose the best answer from the following choices:\n{options}\n'
            self.question_list.append(question)
            self.answer_list.append(data['choices'][data['answer']])

    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}



class ScienceQADataset(Dataset):
    split='test'
    options = ["A", "B", "C", "D", "E", "F", "G", "H"]
    data_root = f'{DATA_DIR}/VQA_Datasets/ScienceQA'

    def __init__(self):
        self.image_list = []
        self.question_list = []
        self.answer_list = []

        ann_path = f"{self.data_root}/{self.split}_anns.json"
        if os.path.exists(ann_path):
            dataset = json.load(open(ann_path, "r"))
            for sample in dataset:
                self.image_list.append(self.data_root + '/' + sample['image_path'])
                self.question_list.append(sample['question'])
                self.answer_list.append(sample['answer'])
        else:
            self.load_save_dataset()
    
    def load_save_dataset(self):
        # load dataset
        data = datasets.load_dataset('derek-thomas/ScienceQA', self.split)
        for sample in data[self.split]:
            if sample['image'] is None:
                continue
           
            question = f"Question: {sample['question']}\n" \
                       f"Options: {' '.join(sample['choices'])}\n"

            self.question_list.append(question)
            self.image_list.append(sample['image'].convert('RGB'))
            self.answer_list.append(sample['choices'][sample['answer']])

        # save dataset
        dataset = []
        for i in range(len(self.image_list)):
            img_file_name = f'{self.data_root}/{self.split}_imgs/{i:04d}.png'
            if not os.path.exists(img_file_name):
                self.image_list[i].save(img_file_name)
            self.image_list[i] = img_file_name
            dataset.append({
                'answer': self.answer_list[i],
                'image_path': self.image_list[i],
                'question': self.question_list[i]
            })
        with open(f"{self.data_root}/{self.split}_anns.json", "w") as f:
            f.write(json.dumps(dataset, indent=4))

    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers}


class ScienceQAIMGDataset(Dataset):
    
    def __init__(self) -> None:
        super().__init__()
        self.data_root = f'{DATA_DIR}/scienceqa'
        self.dataset = []
        jsonl_path = os.path.join(self.data_root, 'annos_test_image_2017samples.jsonl')
        with open(jsonl_path, 'r') as f:
            self.dataset = [json.loads(x) for x in f.readlines()]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> dict:
        sample = self.dataset[index]
        image_path = os.path.join(self.data_root, sample['image_path'])
        return {
            "image_path": image_path,
            "question": sample['question'],
            "gt_answers": sample['answer'],
        }
class ImageNetVC(Dataset):

    def __init__(
        self, task: str='shape', root: str=f'{DATA_DIR}/ImageNetVC'
    ) -> None:
        super().__init__()
        csv_path = os.path.join(root, f'{task}.csv')
        wid2label_path = os.path.join(root, 'ImageNet_mapping.txt')
        label2wid = {}
        with open(wid2label_path, 'r') as f:
            for line in f.readlines():
                # n01440764 tench, Tinca tinca
                wid = line[:9]
                for x in line[9:].split(','):
                    label2wid[x.strip()] = wid
        # category,question,answer
        annos = pd.read_csv(csv_path)
        img_dir = os.path.join(root, 'images')
        self.data = []
        for i, row in annos.iterrows():
            label = row['category']
            question = row['question']
            answer = row['answer']
            wid = label2wid[label]
            for image_path in sorted(list(Path(os.path.join(img_dir, wid)).glob('*'))):
                sample = {
                    'image_path': str(image_path),
                    'question': question,
                    'answer': answer
                }
                self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        sample = self.data[index]
        return {
            'image_path': sample['image_path'],
            'question': sample['question'],
            'gt_answers': sample['answer'],
        }
class RSVQALR(Dataset):

    def __init__(
        self, split: str='test', root: str=f'{DATA_DIR}/RSVQALR', q_type: str=None,
    ) -> None:
        super().__init__()
        self.split = split
        images_path = os.path.join(root, f'LR_split_{self.split}_images.json')
        images = json.load(open(images_path, 'r'))['images']
        img_dir = os.path.join(root, 'Images_LR')
        questions_path = os.path.join(root, f'LR_split_{self.split}_questions.json')
        questions = json.load(open(questions_path, 'r'))['questions']
        answers_path = os.path.join(root, f'LR_split_{self.split}_answers.json')
        answers = json.load(open(answers_path, 'r'))['answers']
        self.data = []
        for x in answers:
            if x['active']:
                ans = x['answer']
                ques_id = x['question_id']
                question = questions[ques_id]
                assert ques_id == question['id'], f'question id NOT valid!'
                if q_type is None:
                    continue
                if question['type'] != q_type:
                    continue
                ques = question['question']
                if q_type == 'presence':
                    ques = (
                        f"Question: {ques}\n\n"
                        'Choose the single most likely answer from the following choices <choice>:\n- Yes\n- No\n\n'
                        'The output format follows exactly as below:\nAnswer: <choice>')
                img_id = question['img_id']
                image = images[img_id]
                assert img_id == image['id'], f'image id NOT valid!'
                # image_name = image['original_name']
                image_path = os.path.join(img_dir, f'{img_id}.tif')
                sample = {
                    'image_path': image_path,
                    'question': ques,
                    'answer': ans
                }
                self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        sample = self.data[index]
        return {
            'image_path': sample['image_path'],
            'question': sample['question'],
            'gt_answers': sample['answer'],
        }

class RSVQALR(Dataset):

    def __init__(
        self, split: str='test', root: str=f'{DATA_DIR}/RSVQALR', q_type: str=None,
    ) -> None:
        super().__init__()
        self.split = split
        images_path = os.path.join(root, f'LR_split_{self.split}_images.json')
        images = json.load(open(images_path, 'r'))['images']
        img_dir = os.path.join(root, 'Images_LR')
        questions_path = os.path.join(root, f'LR_split_{self.split}_questions.json')
        questions = json.load(open(questions_path, 'r'))['questions']
        answers_path = os.path.join(root, f'LR_split_{self.split}_answers.json')
        answers = json.load(open(answers_path, 'r'))['answers']
        self.data = []
        for x in answers:
            if x['active']:
                ans = x['answer']
                ques_id = x['question_id']
                question = questions[ques_id]
                assert ques_id == question['id'], f'question id NOT valid!'
                if q_type is None:
                    continue
                if question['type'] != q_type:
                    continue
                ques = question['question']
                if q_type == 'presence':
                    ques = (
                        f"Question: {ques}\n\n"
                        'Choose the single most likely answer from the following choices <choice>:\n- Yes\n- No\n\n'
                        'The output format follows exactly as below:\nAnswer: <choice>')
                img_id = question['img_id']
                image = images[img_id]
                assert img_id == image['id'], f'image id NOT valid!'
                # image_name = image['original_name']
                image_path = os.path.join(img_dir, f'{img_id}.tif')
                sample = {
                    'image_path': image_path,
                    'question': ques,
                    'answer': ans
                }
                self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> dict:
        sample = self.data[index]
        return {
            'image_path': sample['image_path'],
            'question': sample['question'],
            'gt_answers': sample['answer'],
        }
