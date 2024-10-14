#coding=utf-8
#基于patch的攻击方法，一个patch一个patch进行攻击

import numpy as np
import time
import copy
import torch

####
from .distances import Distance
from .distances import MSE
from PIL import Image
from .tools import has_word, remove_special_chars
from .cider import CiderScorer
import pdb
from .tools import VQAEval
from models import llama_adapter_v2 as llama
import imageio
####
def softmax(logits):
    """Transforms predictions into probability values.

    Parameters
    ----------
    logits : array_like
        The logits predicted by the model.

    Returns
    -------
    `numpy.ndarray`
        Probability values corresponding to the logits.
    """

    assert logits.ndim == 1

    # for numerical reasons we subtract the max logit
    # (mathematically it doesn't matter!)
    # otherwise exp(logits) might become too large or too small
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)

class StopAttack(Exception):
    """Exception thrown to request early stopping of an attack
    if a given (optional!) threshold is reached."""

    pass
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
        if float(self.n_detected_words)!=0:
            prec = self.n_match_words / float(self.n_detected_words) * 100
            recall = self.n_match_words / float(self.n_gt_words) * 100
        else:
            prec = 0
            recall = self.n_match_words / float(self.n_gt_words) * 100
        
        if prec + recall==0:
            f1=0
        else:
            f1 = 2 * (prec * recall) / (prec + recall)
        return prec, recall, f1

    def result_string(self):
        prec, recall, f1 = self.score()
        return f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f}"


def crossentropy(label, logits):
    """Calculates the cross-entropy.

    Parameters
    ----------
    logits : array_like
        The logits predicted by the model.
    label : int
        The label describing the target distribution.

    Returns
    -------
    float
        The cross-entropy between softmax(logits) and onehot(label).

    """

    assert logits.ndim == 1

    # for numerical reasons we subtract the max logit
    # (mathematically it doesn't matter!)
    # otherwise exp(logits) might become too large or too small
    logits = logits - np.max(logits)
    e = np.exp(logits)
    s = np.sum(e)
    ce = np.log(s) - logits[label]
    return ce


def to_cuda(x): #将numpy转换为tensor在显卡上计算
    return torch.from_numpy(x).cuda()


def l2_distance(a, b):
    if type(b) != torch.Tensor:
        b = torch.ones_like(a).cuda() * b        

    dist = (torch.sum((torch.round(a)/255.0 - torch.round(b)/255.0) ** 2))**0.5

    return dist


def normalize_noise(direction, distance, original_image):
    norm_direction = direction/l2_distance(direction, 0)   #归一化

    clipped_direction = torch.clip(torch.round(norm_direction*distance + original_image), 0, 255) - original_image

    clipped_dist = l2_distance(clipped_direction, 0)

    return clipped_direction, clipped_dist



def scatter_draw(data):
    save_path = "/home/syc/adversarial_machine_learning/nips18-avc-attack-template__/"
    fig = plt.figure(figsize=(16,9))
    plt.scatter(data[1], data[0], s=1)
    plt.savefig(save_path+"data.png", bbox_inches='tight')



def clip(x, min_x=-1, max_x=1):
    x[x < min_x] = min_x
    x[x > max_x] = max_x
    return x

def value_mask_init(patch_num):    #初始化查询价值mask
    value_mask = torch.ones([patch_num, patch_num]).cuda()
    # value_mask[int(patch_num*0.25):int(patch_num*0.75) , int(patch_num*0.25):int(patch_num*0.75)] = 0.5

    return value_mask

def noise_mask_init(x, image, patch_num, patch_size):    #初始化噪声幅度mask
    noise = x - image
    noise_mask = torch.zeros([patch_num, patch_num]).cuda()
    for row_counter in range(patch_num):
        for col_counter in range(patch_num):
            noise_mask[row_counter][col_counter] = l2_distance(noise[(row_counter*patch_size):(row_counter*patch_size+patch_size) , (col_counter*patch_size):(col_counter*patch_size+patch_size) ], 0)

    return noise_mask


def translate(index, patch_num):  #将价值最高patch的行列输出出来
    best_row = index//patch_num
    best_col = index - patch_num*best_row

    return best_row, best_col




class Attacker:
    def __init__(self, model,task,label):
        self.model = model
        self.task=task
        self.__original_class=label

    def attack(self, inputs):
        return NotImplementedError

    def attack_target(self, inputs, targets):
        return NotImplementedError


class PatchAttack(Attacker):
    def __init__(self, model,task,label): 
        self.model = model
        self.task=task
        self.__original_class=label

    def predictions(self, inputs, question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None):
        if model_name=="TestMiniGPT4"  or model_name=="vpgtrans":
            chat_list_new=chat_list.copy()
            outputs = self.model.batch_answer([Image.fromarray(np.uint8(inputs))], [question_list], [chat_list_new],max_new_tokens=max_new_tokens)
            del chat_list_new
        elif model_name=="blip2":
            imgs = vis_proc["eval"](Image.fromarray(np.uint8(inputs))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = f"Question: {question_list} Answer:" 
            outputs = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        elif model_name=="instruct_blip":
            imgs = vis_proc["eval"](Image.fromarray(np.uint8(inputs))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = question_list 
            outputs = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        elif model_name=="adv2":
            imgs = vis_proc(Image.fromarray(np.uint8(inputs))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts =[ llama.format_prompt(question_list) ]
            outputs =[ self.model.generate(imgs, prompts, temperature=0,max_gen_len=max_new_tokens)[0].strip()]
        elif model_name=="panda":  
            Image.fromarray(np.uint8(inputs)).save("./panda2_{}.png".format(vis_proc[0]) )      
            image_list= "./panda2_{}.png".format(vis_proc[0])        
            outputs = [self.model(image_list, question_list, max_new_tokens)]
        elif model_name=="otter": 
            imgs = vis_proc([Image.fromarray(np.uint8(inputs))],return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0).to("cuda", dtype=torch.float16)
            prompts = [f"<image> User: {question_list} GPT: <answer>"]
            lang_x = self.model.text_tokenizer(prompts, return_tensors="pt", padding=True)
            generated_text = self.model.generate(
            vision_x=imgs,
            lang_x=lang_x["input_ids"].to("cuda"),
            attention_mask=lang_x["attention_mask"].to("cuda", dtype=torch.float16),
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
            )
            output = self.model.text_tokenizer.decode(generated_text[0])
            output = [x for x in output.split(' ') if not x.startswith('<')]
            out_label = output.index('GPT:')
            outputs = [' '.join(output[out_label + 1:])]
        elif model_name=="owl":
            prompt_template = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {}\nAI:"

            prompts = [prompt_template.format(question_list)]
            inputs = vis_proc[2](text=prompts, images=[Image.fromarray(np.uint8(inputs))], return_tensors='pt')
            inputs = {k: v.to("cuda", dtype=torch.float32) if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            generate_kwargs = {
            'do_sample': False,
            'top_k': 5,
            'max_length': max_new_tokens
            }
            with torch.no_grad():
                res = self.model.generate(**inputs, **generate_kwargs)
            outputs = [vis_proc[1].decode(res.tolist()[0], skip_special_tokens=True)]  
        elif model_name=="ofv2":
            vision_x = vis_proc[0](Image.fromarray(np.uint8(inputs))).unsqueeze(0).unsqueeze(0).unsqueeze(0).to("cuda", dtype=torch.float16)
            prompts = [f"<image>Question: {question_list} Short answer:"]            
            lang_x = vis_proc[1](
            prompts,
            return_tensors="pt", padding=True,
            ).to("cuda")
            generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"].to("cuda"),
            attention_mask=lang_x["attention_mask"].to("cuda", dtype=torch.float16),
            max_new_tokens=max_new_tokens,
            num_beams=3,pad_token_id=vis_proc[1].eos_token_id
            )
            outputs = vis_proc[1].batch_decode(generated_text, skip_special_tokens=True)
            outputs = [y[len(x)-len('<image>'):].strip() for x, y in zip(prompts, outputs)]
        elif model_name == "internlm":
            Image.fromarray(np.uint8(inputs)).save("./internlm2{}.png".format(vis_proc[2]))   
            image_list= "./internlm2{}.png".format(vis_proc[2])  
            texts=f" <|User|>:<ImageHere> {question_list}" + vis_proc[1] + " <|Bot|>:"
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs=[vis_proc[0](self.model,texts, image_list,max_new_tokens=max_new_tokens) ]  
            
        elif model_name=="llava" or model_name=="llava15" or model_name=="moellava" or model_name=="sharegpt4v":
            outputs = self.model([question_list],[Image.fromarray(np.uint8(inputs))],stop_str=vis_proc[0], dtype=torch.float16, max_new_tokens=max_new_tokens,method=vis_proc[1], level=vis_proc[2],image_listnew=None)
       
        if self.task=="cls" or self.task=="ocr":
            predict = remove_special_chars(outputs[0]).lower()            
        else:
            predict = outputs[0]
        temp_result=predict
        if 1:
            is_adversarial=False
            if self.task=="cls" or self.task=="ocr":        
                adv=True
                if len(self.__original_class)!=0 and not isinstance(self.__original_class, str):
                    for gt in self.__original_class:
                        is_adversarial_tmp=not (bool(has_word(temp_result, gt)) or bool(has_word(temp_result, gt+'s')) )
                        adv=adv and is_adversarial_tmp
                    is_adversarial= is_adversarial or adv
                else:
                    is_adversarial=not (bool(has_word(temp_result, self.__original_class)) or bool(has_word(temp_result, self.__original_class+'s')) )                    
            if self.task=="caption":                
                cider_scorer = CiderScorer(n=4, sigma=6.0)
                cider_scorer += (temp_result, self.__original_class)
                (score, scores) = cider_scorer.compute_score()                
                if scores==0:
                    is_adversarial=True
            elif self.task=="kie":
                f1_scorer = F1Scorer()
                if isinstance(self.__original_class, list) :
                    gt_answers =" ".join(self.__original_class)
                else:
                    gt_answers=self.__original_class
                f1_scorer.add_string(gt_answers, temp_result)
                prec, recall, f1 = f1_scorer.score()
                if f1==0:
                    is_adversarial=True                    
            elif self.task=="mrr":
                eval = VQAEval()
                mrr = eval.evaluate_MRR(temp_result, self.__original_class)
                if mrr==0:
                    is_adversarial=True  
            elif self.task=="vqa" or self.task=="vqachoice" or self.task=="imagenetvc":
                eval = VQAEval()
                mrr = eval.evaluate(temp_result, self.__original_class)
                if mrr==0:
                    is_adversarial=True     
        
        return predict,is_adversarial

    def distance(self, input1, input2, min_, max_):
        return np.mean((input1 - input2) ** 2) / ((max_ - min_) ** 2)

    def print_distance(self, distance):
        return np.sqrt(distance * 1*28*28)

    def log_step(self, step, distance, spherical_step, source_step, message=''):
        print('Step {}: {:.5f}, stepsizes = {:.1e}/{:.1e}: {}'.format(
            step,
            self.print_distance(distance),
            spherical_step,
            source_step,
            message))

    def patch_attack(
            self,
            original,    #原始图像
            label,       #原始标签
            starting_point,   #初始对抗样本
            iterations=1000,  #总的查询次数
            min_=0.0,         
            max_=255.0,
            mode='targeted', question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None):

        from numpy.linalg import norm
        from scipy import interpolate
        import collections

        #全部转换为torch来计算
        original = to_cuda(original)
        starting_point = to_cuda(starting_point)
        step = 0

        patch_num = 4   #横纵几等分
        patch_size = int(original.shape[0] / patch_num)


        success_num = 0    #成功和失败的次数
        fail_num = 0

        value_mask = value_mask_init(patch_num)
        noise_mask = noise_mask_init(starting_point, original, patch_num, patch_size)

        best_noise = starting_point - original
        current_min_noise = l2_distance(starting_point, original)

        #FIXME
        evolutionary_doc = np.zeros(iterations)   #记录下当前最小噪声   这个不管了，先不记录了

        while step < iterations:

            if torch.sum(value_mask * noise_mask) == 0:  #当前平分方法下没有可以查询的了
                #FIXME
                # print("*************-----------------")
                # pdb.set_trace()
                print("patch num * 2", step)
                patch_num *= 2

                if patch_num == 64:  
                    print("only", step)
                    break

                patch_size = int(original.shape[0] / patch_num)

                value_mask = value_mask_init(patch_num)
                noise_mask = noise_mask_init(best_noise, original, patch_num, patch_size)
            total_mask = value_mask*noise_mask
            best_index = torch.argmax(total_mask)
            best_row, best_col = translate(best_index, patch_num)
            temp_noise = copy.deepcopy(best_noise)
            temp_noise[(best_row*patch_size):(best_row*patch_size+patch_size) , (best_col*patch_size):(best_col*patch_size+patch_size) ] = 0
            candidate = torch.clip(torch.round(original + temp_noise), 0, 255)   
            if l2_distance(candidate, original) >= current_min_noise:
                value_mask[best_row, best_col] = 0
                continue
            if chat_list is not None:
                chat_list_new=chat_list.copy()
                temp_result,is_adversarial = self.predictions((candidate).cpu().numpy(), question_list=question_list, chat_list=chat_list_new, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)
                del chat_list_new
            else:
                temp_result,is_adversarial = self.predictions((candidate).cpu().numpy(), question_list=question_list, chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)

            if is_adversarial:
                current_min_noise = l2_distance(candidate, original)
                success_num += 1
                best_noise = candidate - original
                noise_mask[best_row, best_col] = l2_distance(best_noise[(best_row*patch_size):(best_row*patch_size+patch_size) , (best_col*patch_size):(best_col*patch_size+patch_size) ], 0)
            else:
                fail_num += 1
                value_mask[best_row, best_col] = 0
            step += 1 
        final_best_adv_example = best_noise+original
        final_best_adv_example = final_best_adv_example.cpu().numpy().astype(np.float32)
        print("success_num", success_num, step)
        return final_best_adv_example, step


    def attack(
            self, 
            image,
            label,
            starting_point, 
            iterations=1000,
            val_samples = 1000,
            min_=0.0, 
            max_=255.0,
            mode = 'untargeted',
            strategy = 0, question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None):

        if mode == 'untargeted':
            return self.patch_attack(image, label, starting_point, iterations, min_, max_, mode='untargeted', question_list=question_list, chat_list=chat_list, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)

       