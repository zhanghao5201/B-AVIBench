import numpy as np
import time
import copy
# from foolbox.utils import crossentropy, softmax

import eagerpy as ep

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
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def l2_distance(a, b):
    return (np.sum((a/255.0 - b/255.0) ** 2))**0.5

def ortho(noise):  
    noise_dim=noise.shape


    xr=(np.random.rand(noise_dim[0]))
    xo=xr-(np.sum(noise*xr)/np.sum(noise**2))*noise

    xo -= np.mean(xo)
    xo=np.reshape(xo, (1, 28, 28))

    return xo


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

def cw_loss_calculator(label, inputs):
    return np.max(inputs) - inputs[label]



def griddle(noise, rate):  
    noise_temp = np.round(noise)  
    noise_temp = np.abs(noise_temp)
    negative_rate = 1 - rate



    perturbed_num = np.sum(noise_temp != 0) 
    deleted = 0  

    for i in range(1, 256):
        if np.sum(noise_temp == i) != 0:  
            temp_deleted = deleted + np.sum(noise_temp == i)

            if temp_deleted/(perturbed_num * 1.0) >= negative_rate: 
                lottery_rate = (negative_rate*perturbed_num*1.0 - deleted)/(np.sum(noise_temp == i))

                temp_A = copy.deepcopy(noise_temp)
                temp_A[temp_A != i] = 0
                temp_B =  np.random.uniform(0, 1, np.shape(temp_A))
                temp_B[temp_B<lottery_rate] = 0
                temp_B[temp_B>=lottery_rate] = 1

                noise_temp = noise_temp - temp_A + temp_A*temp_B
                break

            else:
                noise_temp[noise_temp == i] = 0
                deleted = temp_deleted

    mask = copy.deepcopy(noise_temp)  
    mask[mask != 0] = 1


    return mask




def clip(x, min_x=-1, max_x=1):
    x[x < min_x] = min_x
    x[x > max_x] = max_x
    return x


def l2_distance(a, b):
    return (np.sum((np.round(a)/255.0 - np.round(b)/255.0) ** 2))**0.5


class Attacker:
    def __init__(self, model,task,label):
        self.model = model
        self.task=task
        self.__original_class=label

    def attack(self, inputs):
        return NotImplementedError

    def attack_target(self, inputs, targets):
        return NotImplementedError


class EvolutionaryAttack(Attacker):
    def __init__(self, model,task,label): 
        self.model = model
        self.task=task
        self.__original_class=label
    def ce_and_cw_loss(self, inputs, label):
        logits = self.model.forward_one(np.round(inputs).astype(np.float32))
        ce_loss = ep.crossentropy(label, logits)
        cw_loss = cw_loss_calculator(label, logits)

        return ce_loss, cw_loss

    def cw_prob_calculator(self, logits, label):

        predict_label = np.argmax(logits)
        exp_logits = np.exp(logits)
        prob = exp_logits/np.sum(exp_logits)

        if predict_label != label:
            cw_prob = np.max(prob) - prob[label]
        else:
            temp_prob = copy.deepcopy(prob)
            temp_prob[label] = -9999
            near_label = np.argmax(temp_prob)
            cw_prob = prob[near_label] - prob[label]
        return cw_prob

    def predictions(self, inputs, question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None):
        
        if model_name=="TestMiniGPT4"  or model_name=="vpgtrans":
            outputs = self.model.batch_answer([Image.fromarray(np.uint8(inputs))], [question_list], [chat_list],max_new_tokens=max_new_tokens)
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
            prompts = [llama.format_prompt(question_list) ]
            outputs =[self.model.generate(imgs, prompts, temperature=0,max_gen_len=max_new_tokens)[0].strip()]
        elif model_name=="panda":    
            Image.fromarray(np.uint8(inputs)).save("./panda3_{}.png".format(vis_proc[0]) )        
            image_list= "./panda3_{}.png".format(vis_proc[0])        
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
        elif model_name == "internlm":
            Image.fromarray(np.uint8(inputs)).save("./internlm3{}.png".format(vis_proc[2]))   
            image_list= "./internlm3{}.png".format(vis_proc[2])  
            texts=f" <|User|>:<ImageHere> {question_list}" + vis_proc[1] + " <|Bot|>:"                      
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs=[vis_proc[0](self.model,texts, image_list,max_new_tokens=max_new_tokens) ]  
            
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
        elif model_name=="llava" or model_name=="llava15" or model_name=="moellava" or model_name=="sharegpt4v":
            outputs = self.model([question_list],[Image.fromarray(np.uint8(inputs))],stop_str=vis_proc[0], dtype=torch.float16, max_new_tokens=max_new_tokens,method=vis_proc[1], level=vis_proc[2],image_listnew=None)
        elif model_name == "Qwen":
            Image.fromarray(np.uint8(inputs)).save("./Qwen3{}_{}.png".format(vis_proc[1][0],vis_proc[1][1])) 
            image_list= "./Qwen3{}_{}.png".format(vis_proc[1][0],vis_proc[1][1])
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                   outputs = [vis_proc[0](image_list, question_list, max_new_tokens)]   

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

    def evolutionary_attack(
            self,
            original,
            label,
            starting_point,
            initial_time,
            time_limit=10,
            iterations=1000,
            spherical_step=3e-2,
            source_step=1e-2,
            min_=0.0,
            max_=255.0,
            mode='targeted',
            rescale_or_not = False,
            rate = 0.2,
            step_decay_factor=0.99,
            big_size = (64,64),
            center_size = (40,40), question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None):

        from numpy.linalg import norm
        from scipy import interpolate
        import collections


        resize_factor = 4
        perturbed = starting_point   
        dis = self.distance(perturbed, original, min_, max_)  
        shape = [center_size[0], center_size[1]]
        big_shape = [big_size[0], big_size[1], 3]
        decay_factor = 0.99
        init_source_step = copy.deepcopy(source_step)
        init_spherical_step = copy.deepcopy(spherical_step)

        center_shape = [center_size, center_size, 3]

        pert_shape = [int(shape[0]/resize_factor), int(shape[1]/resize_factor), 3]   


        if rescale_or_not == 1 or rescale_or_not == 5 or rescale_or_not == 55 or rescale_or_not == 29:
            evolution_path = np.zeros(pert_shape , dtype=original.dtype)
            diagonal_covariance = np.ones(pert_shape, dtype=original.dtype)   
        elif rescale_or_not == 4:
            evolution_path = np.zeros(center_shape, dtype=original.dtype)
            diagonal_covariance = np.ones(center_shape, dtype=original.dtype) 
        else:
            evolution_path = np.zeros(big_shape, dtype=original.dtype)
            diagonal_covariance = np.ones(big_shape, dtype=original.dtype) 


        c = 0.001                                                   
        stats_step_adversarial = collections.deque(maxlen=20)

        neg_improve_num = 0

        evolutionary_doc = np.zeros(iterations)   
        best_dis = 0

        success_num = 0

        if rescale_or_not == 15 or rescale_or_not == 22 or rescale_or_not == 39 or rescale_or_not == 23 or rescale_or_not == 29 or rescale_or_not == 31 or rescale_or_not == 33 or rescale_or_not == 34 or rescale_or_not == 35:  # 修正均值,定义是否学习的flag变量
            amend_flag = False
            amend_list = []   

        if rescale_or_not == 16:
            amend_list = []   

        if rescale_or_not == 17:
            amend = 0   

        if rescale_or_not == 18 or rescale_or_not == 19:
            success_list = []
            fail_list = []

        if rescale_or_not == 37:
            last_50_success = 0

        if rescale_or_not == 21 or rescale_or_not == 24 or rescale_or_not == 25 or rescale_or_not == 26 or rescale_or_not == 27 or rescale_or_not == 28:
            success_noise_list = [perturbed - original]
            fail_noise_list = []

        if rescale_or_not == 28:
            temp_result, temp_logits = self.predictions(perturbed)
            success_prob = [self.cw_prob_calculator(temp_logits, label)]

        if rescale_or_not == 30 or rescale_or_not == 31:
            temp_result, temp_logits = self.predictions(perturbed)
            noise_list = [perturbed - original]
            prob_list = [self.cw_prob_calculator(temp_logits, label)]
            prob_saver = []
            sample_num = 10
            backup_perturbation = []   
            backup_prob = []

        if rescale_or_not == 33: 
            prob_est = 0

        
        for step in range(1, iterations + 1):
            unnormalized_source_direction = original - perturbed   
            source_norm = norm(unnormalized_source_direction)    
            clipper_counter = 0 

            if rescale_or_not == 2:  
                perturbation_large = np.random.normal(0, 1, big_shape)

            line_candidate = perturbed + source_step * unnormalized_source_direction               
            candidate = line_candidate + spherical_step * source_norm * perturbation_large / max(norm(perturbation_large), 1e-6)
            if norm(original - candidate)!=0 and  norm(original - line_candidate)!=0:
                candidate = original - (original - candidate) / norm(original - candidate) * norm(original - line_candidate)
                candidate = clip(candidate, min_, max_)
            else:
                candidate = original            

            if chat_list is not None:
                chat_list_new=chat_list.copy()
                temp_result,is_adversarial = self.predictions((candidate), question_list=question_list, chat_list=chat_list_new, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)
                del chat_list_new
            else:
                temp_result,is_adversarial = self.predictions((candidate), question_list=question_list, chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)

            stats_step_adversarial.appendleft(is_adversarial)
            if rescale_or_not == 30 or rescale_or_not == 31: 
                noise_list.append(candidate - original)
                this_prob = self.cw_prob_calculator(temp_logits, label)
                prob_list.append(this_prob)
            if is_adversarial: 
                improvement = l2_distance(original, perturbed) - l2_distance(original, candidate)
                if improvement < 0:
                    neg_improve_num += 1

                if rescale_or_not == 39 and improvement < 0:
                    temp_possibility = np.random.rand(1)[0]
                    if (1.0*step/iterations) < temp_possibility:

                        new_perturbed = None
                    else:
                        success_num += 1
                        new_perturbed = candidate
                        new_dis = self.distance(candidate, starting_point, min_, max_)

                        best_dis = new_dis

                elif rescale_or_not == 38 and improvement < 0:  
                    new_perturbed = None
                else:
                    success_num += 1
                    new_perturbed = candidate
                    new_dis = self.distance(candidate, starting_point, min_, max_)

                    best_dis = new_dis

                if rescale_or_not==1 or rescale_or_not==5 or rescale_or_not==4 or rescale_or_not==55 or rescale_or_not==29:
                    evolution_path = decay_factor * evolution_path + np.sqrt(1 - decay_factor ** 2) * perturbation
                elif rescale_or_not == 7:
                    evolution_path = decay_factor * evolution_path + np.sqrt(1 - decay_factor ** 2) * perturbation_large
                else:
                    evolution_path = decay_factor * evolution_path + np.sqrt(1 - decay_factor ** 2) * perturbation_large
                diagonal_covariance = (1 - c) * diagonal_covariance + c * (evolution_path ** 2)

                if rescale_or_not == 15 or rescale_or_not == 22 or rescale_or_not == 39 or rescale_or_not == 23 or rescale_or_not == 29 or rescale_or_not == 31 or rescale_or_not == 33 or rescale_or_not == 34 or rescale_or_not == 35:
                    if amend_flag == True:   
                        amend_flag = False
                        amend_list=[]

                if rescale_or_not == 18 or rescale_or_not == 19:
                    success_list.append(perturbation_large)

                if rescale_or_not == 21 or rescale_or_not == 24:
                    success_noise_list.append(candidate - original)

                if rescale_or_not == 26 or rescale_or_not == 27:
                    weight = self.cw_prob_calculator(temp_logits, label)
                    success_noise_list.append(weight*(candidate - original))

                if rescale_or_not == 28:
                    success_noise_list.append(candidate - original)
                    success_prob.append(self.cw_prob_calculator(temp_logits, label))

                if rescale_or_not == 30 or rescale_or_not == 31:
                    backup_perturbation = []
                    backup_prob = []

                if rescale_or_not == 32 or rescale_or_not == 22 or rescale_or_not == 35:  #step decay
                    source_step *= step_decay_factor
                    spherical_step *= step_decay_factor

                if rescale_or_not == 34:  
                    source_step = init_source_step
                    spherical_step = init_spherical_step

                if rescale_or_not == 33:  
                    prob_est += 0.01 

                if rescale_or_not == 37:
                    last_50_success += 1


            else:
                new_perturbed = None

                if rescale_or_not == 15 or rescale_or_not == 22 or rescale_or_not == 39 or rescale_or_not == 23 or rescale_or_not == 31:
                    if amend_flag == True:   
                        amend_list.append(perturbation_large)
                    else:  
                        amend_flag = True
                        amend_list.append(perturbation_large)

                if rescale_or_not == 34:  
                    if amend_flag == True:  
                        amend_list.append(perturbation_large)
                        if len(amend_list) == 50:   
                            source_step /= 1.5
                            spherical_step /= 1.5
                            amend_flag = False
                            amend_list=[]
                    else: 
                        amend_flag = True
                        amend_list.append(perturbation_large) 

                if rescale_or_not == 16:
                    amend_list.append(perturbation_large)

                if rescale_or_not == 17:
                    amend = 0.7*amend + 0.3*perturbation_large

                if rescale_or_not == 18 or rescale_or_not == 19:
                    fail_list.append(perturbation_large)

                if rescale_or_not == 21 or rescale_or_not == 24:
                    fail_noise_list.append(candidate - original)

                if rescale_or_not == 26 or rescale_or_not == 27:
                    weight = self.cw_prob_calculator(temp_logits, label)
                    fail_noise_list.append(weight*(candidate - original))

                if rescale_or_not == 29:
                    if amend_flag == True:  
                        amend_list.append(perturbation)
                    else: 
                        amend_flag = True
                        amend_list.append(perturbation)

            if rescale_or_not == 33 and step>0 and step%100 == 0:  
                if prob_est > 0.5:
                    source_step *= 1.5
                    spherical_step *= 1.5

                if prob_est < 0.2:
                    source_step /= 1.5
                    spherical_step /= 1.5

                prob_est = 0


            message = ''
            if new_perturbed is not None:
                abs_improvement = dis - new_dis
                rel_improvement = abs_improvement / dis
                message = 'd. reduced by {:.2f}% ({:.4e})'.format(
                    rel_improvement * 100, abs_improvement)

                perturbed = new_perturbed
                dis = new_dis
            evolutionary_doc[step-1] = l2_distance(original, perturbed)
            if len(stats_step_adversarial) == stats_step_adversarial.maxlen:
                p_step = np.mean(stats_step_adversarial)
                n_step = len(stats_step_adversarial)
                source_step *= np.exp(p_step - 0.2)
                stats_step_adversarial.clear()

        print("success_num, neg_improve_num", success_num, neg_improve_num)
        if rescale_or_not == 13: 
            print("clipper_counter", clipper_counter)

        
        if rescale_or_not == 21:   
            fail_noise_mean = np.mean(np.array(fail_noise_list), axis=0)
            success_noise_mean = np.mean(np.array(success_noise_list), axis=0)

            total_noise_list = fail_noise_list + success_noise_list
            total_noise_array = np.array(total_noise_list)
            total_noise_mean = np.mean(total_noise_array, axis=0)

            fail_noise_dist = np.sqrt(np.sum((np.array(fail_noise_list) - fail_noise_mean)**2)/len(fail_noise_list))
            success_noise_dist = np.sqrt(np.sum((np.array(success_noise_list) - success_noise_mean)**2)/len(success_noise_list))
            total_noise_dist = np.sqrt(np.sum((total_noise_array - total_noise_mean)**2)/len(total_noise_list))
            print("fail_dist", fail_noise_dist)
            print("success_dist", success_noise_dist)
            print("total_dist", total_noise_dist)   


        return perturbed, evolutionary_doc


    def attack(
            self, 
            image,
            label,
            starting_point, 
            initial_time,
            time_limit=10,
            iterations=1000, 
            spherical_step=3e-2, 
            source_step=1e-3,
            min_=0.0, 
            max_=255.0,
            rescale_or_not = False,
            rate = 0.2,
            step_decay_factor=0.99,
            big_size = (64,64),
            center_size = (40,40),
            mode = 'untargeted', question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None):

        if mode == 'untargeted':            
                return self.evolutionary_attack(
                    image, label, starting_point, initial_time, time_limit,
                    iterations, spherical_step, source_step, 
                    min_, max_, mode='untargeted', rescale_or_not=rescale_or_not, rate = rate, step_decay_factor=step_decay_factor, big_size=big_size, center_size=center_size, question_list=question_list, chat_list=chat_list, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)

       
