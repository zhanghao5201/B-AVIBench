"""
Provides a class that represents an adversarial example.

"""

import numpy as np
import numbers

from .distances import Distance
from .distances import MSE
from PIL import Image
from .tools import has_word, remove_special_chars
from .cider import CiderScorer
import pdb
from .tools import VQAEval
from models import llama_adapter_v2 as llama
import torch
import imageio
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


class Adversarial(object):
    """Defines an adversarial that should be found and stores the result.

    The :class:`Adversarial` class represents a single adversarial example
    for a given model, criterion and reference input. It can be passed to
    an adversarial attack to find the actual adversarial perturbation.

    Parameters
    ----------
    model : a :class:`Model` instance
        The model that should be fooled by the adversarial.
    criterion : a :class:`Criterion` instance
        The criterion that determines which inputs are adversarial.
    unperturbed : a :class:`numpy.ndarray`
        The unperturbed input to which the adversarial input should be as close as possible.
    original_class : int
        The ground-truth label of the unperturbed input.
    distance : a :class:`Distance` class
        The measure used to quantify how close inputs are.
    threshold : float or :class:`Distance`
        If not None, the attack will stop as soon as the adversarial
        perturbation has a size smaller than this threshold. Can be
        an instance of the :class:`Distance` class passed to the distance
        argument, or a float assumed to have the same unit as the
        the given distance. If None, the attack will simply minimize
        the distance as good as possible. Note that the threshold only
        influences early stopping of the attack; the returned adversarial
        does not necessarily have smaller perturbation size than this
        threshold; the `reached_threshold()` method can be used to check
        if the threshold has been reached.

    """

    def __init__(
        self,
        model,
        task,unperturbed,
        original_class,
        distance=MSE,
        threshold=None,
        verbose=False, question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None
    ):

        self.__model = model
        
        self.task = task
        self.__unperturbed = unperturbed
        self.__unperturbed_for_distance = unperturbed
        self.__original_class = original_class
        self.__distance = distance
        
        if threshold is not None and not isinstance(threshold, Distance):
            threshold = distance(value=threshold)
        self.__threshold = threshold

        self.verbose = verbose
        
        self.__best_adversarial = None
        self.__best_distance = distance(value=np.inf)
        self.__best_adversarial_output = None

        self._total_prediction_calls = 0
        self._total_gradient_calls = 0
        
        self._best_prediction_calls = 0
        self._best_gradient_calls = 0
        
        # check if the original input is already adversarial
        self._check_unperturbed(question_list=question_list, chat_list=chat_list, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)
   
    def _check_unperturbed(self, question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None):
        try:            
            if chat_list is not None:
                chat_list_new=chat_list.copy()
                self.forward_one(self.__unperturbed, question_list=question_list, chat_list=chat_list_new, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc,flag=1)
                del chat_list_new
            else:
                self.forward_one(self.__unperturbed, question_list=question_list, chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc,flag=1)
            
        except StopAttack:
            # if a threshold is specified and the unperturbed input is
            # misclassified, this can already cause a StopAttack
            # exception
            assert self.distance.value == 0.0

    def _reset(self):
        self.__best_adversarial = None
        self.__best_distance = self.__distance(value=np.inf)
        self.__best_adversarial_output = None

        self._best_prediction_calls = 0
        self._best_gradient_calls = 0

        self._check_unperturbed()

    @property
    def perturbed(self):
        """The best adversarial example found so far."""
        return self.__best_adversarial

    @property
    def output(self):
        """The model predictions for the best adversarial found so far.

        None if no adversarial has been found.
        """
        return self.__best_adversarial_output

    @property
    def adversarial_class(self):
        """The argmax of the model predictions for the best adversarial found so far.

        None if no adversarial has been found.
        """
        if self.output is None:
            return None
        return np.argmax(self.output)

    @property
    def distance(self):
        """The distance of the adversarial input to the original input."""
        return self.__best_distance

    @property
    def unperturbed(self):
        """The original input."""
        return self.__unperturbed

    @property
    def original_class(self):
        """The class of the original input (ground-truth, not model prediction)."""
        return self.__original_class

    @property
    def _model(self):  # pragma: no cover
        """Should not be used."""
        return self.__model

    @property
    def __task(self):  # pragma: no cover
        """Should not be used."""
        return self.__task

    @property
    def _distance(self):  # pragma: no cover
        """Should not be used."""
        return self.__distance

    def set_distance_dtype(self, dtype):
        assert dtype >= self.__unperturbed.dtype
        self.__unperturbed_for_distance = self.__unperturbed.astype(dtype, copy=False)

    def reset_distance_dtype(self):
        self.__unperturbed_for_distance = self.__unperturbed

    def normalized_distance(self, x):
        """Calculates the distance of a given input x to the original input.

        Parameters
        ----------
        x : `numpy.ndarray`
            The input x that should be compared to the original input.

        Returns
        -------
        :class:`Distance`
            The distance between the given input and the original input.

        """
        return self.__distance(self.__unperturbed_for_distance, x, bounds=self.bounds())

    def reached_threshold(self):
        """Returns True if a threshold is given and the currently
        best adversarial distance is smaller than the threshold."""        
        return self.__threshold is not None and self.__best_distance <= self.__threshold

    def __new_adversarial(self, x, predictions, in_bounds):
        x = x.copy()  # to prevent accidental inplace changes
        distance = self.normalized_distance(x)
        if in_bounds and self.__best_distance > distance:
            # new best adversarial
            if self.verbose:
                print("new best adversarial: {}".format(distance))

            self.__best_adversarial = x
            self.__best_distance = distance
            self.__best_adversarial_output = predictions

            self._best_prediction_calls = self._total_prediction_calls
            self._best_gradient_calls = self._total_gradient_calls

            if self.reached_threshold():
                raise StopAttack

            return True, distance
        return False, distance

    def __is_adversarial(self, x, predictions, in_bounds,flag=0):
        """Interface to criterion.is_adverarial that calls
        __new_adversarial if necessary.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            The input that should be checked.
        predictions : :class:`numpy.ndarray`
            A vector with the pre-softmax predictions for some input x.
        label : int
            The label of the unperturbed reference input.

        """        
        is_adversarial=False
        aa=0
        if self.task=="cls" or self.task=="ocr":                     
            adv=True            
            if len(self.__original_class)!=0 and not isinstance(self.__original_class, str):                
                for gt in self.__original_class:
                    is_adversarial_tmp=not (bool(has_word(predictions, gt)) or bool(has_word(predictions, gt+'s')) )
                    adv=adv and is_adversarial_tmp
                is_adversarial= is_adversarial or adv
            else:
                is_adversarial=not (bool(has_word(predictions, self.__original_class)) or bool(has_word(predictions, self.__original_class+'s')) )
        if self.task=="caption":
            cider_scorer = CiderScorer(n=4, sigma=6.0)
            cider_scorer += (predictions, self.__original_class)
            (score, scores) = cider_scorer.compute_score()
            if scores==0:
                is_adversarial=True
        elif self.task=="kie":
            f1_scorer = F1Scorer()
            if isinstance(self.__original_class, list) :
                gt_answers =" ".join(self.__original_class)
            else:
                gt_answers=self.__original_class
            f1_scorer.add_string(gt_answers, predictions)
            prec, recall, f1 = f1_scorer.score()
            aa=0
            if flag==1:
                if f1==0:
                    is_adversarial=True
                    aa=1
            else:
                if f1==0:
                    is_adversarial=True
            
                        # is_adversarial=True
        elif self.task=="mrr":
            eval = VQAEval()
            mrr = eval.evaluate_MRR(predictions, self.__original_class)
            if flag==1:
                if mrr==0:
                    is_adversarial=True
                    aa=1
            else:
                if mrr==0:
                    is_adversarial=True            
            
        elif self.task=="vqa" or self.task=="vqachoice" or self.task=="imagenetvc":
            eval = VQAEval()
            mrr = eval.evaluate(predictions, self.__original_class)
            if flag==1:
                if mrr==0:
                    is_adversarial=True
                    aa=1
            else:
                if mrr==0:
                    is_adversarial=True            


        
        assert isinstance(is_adversarial, bool) or isinstance(is_adversarial, np.bool_)
        if is_adversarial:
            is_best, distance = self.__new_adversarial(x, predictions, in_bounds)            
        else:
            is_best = False
            distance = None
        return is_adversarial, is_best, distance   

    def bounds(self):
        min_=0
        max_ = 255
        assert isinstance(min_, numbers.Number)
        assert isinstance(max_, numbers.Number)
        assert min_ < max_
        return min_, max_

    def in_bounds(self, input_):
        min_, max_ = self.bounds()
        return min_ <= input_.min() and input_.max() <= max_

    def channel_axis(self, batch):
        """Interface to model.channel_axis for attacks.

        Parameters
        ----------
        batch : bool
            Controls whether the index of the axis for a batch of inputs
            (4 dimensions) or a single input (3 dimensions) should be returned.

        """
        axis = self.__model.channel_axis()
        if not batch:
            axis = axis - 1
        return axis

    def forward_one(self, x, strict=True, return_details=False, question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None,flag=0):
        """Interface to model.forward_one for attacks.

        Parameters
        ----------
        x : `numpy.ndarray`
            Single input with shape as expected by the model
            (without the batch dimension).
        strict : bool
            Controls if the bounds for the pixel values should be checked.

        """
        in_bounds = self.in_bounds(x)
        assert not strict or in_bounds

        self._total_prediction_calls += 1
        if model_name=="TestMiniGPT4"  or model_name=="vpgtrans":
            chat_list_new=chat_list.copy()
            outputs = self.__model.batch_answer([Image.fromarray(np.uint8(x))], [question_list], [chat_list_new],max_new_tokens=max_new_tokens)
            del chat_list_new
        elif model_name=="blip2":   
            imgs = vis_proc["eval"](Image.fromarray(np.uint8(x))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = f"Question: {question_list} Answer:" 
            outputs = self.__model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        elif model_name=="instruct_blip":
            imgs = vis_proc["eval"](Image.fromarray(np.uint8(x))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = question_list 
            outputs = self.__model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        elif model_name=="adv2":
            imgs = vis_proc(Image.fromarray(np.uint8(x))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = [llama.format_prompt(question_list) ]
            outputs = [self.__model.generate(imgs, prompts, temperature=0, max_gen_len=max_new_tokens)[0].strip()]
        elif model_name=="panda":            
            Image.fromarray(np.uint8(x)).save("./panda_{}.png".format(vis_proc[0]))   
            image_list= "./panda_{}.png".format(vis_proc[0])      
            outputs = [self.__model(image_list, question_list, max_new_tokens)]
        elif model_name=="otter": 
            imgs = vis_proc([Image.fromarray(np.uint8(x))],return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0).to("cuda", dtype=torch.float16)
            prompts = [f"<image> User: {question_list} GPT: <answer>"]
            lang_x = self.__model.text_tokenizer(prompts, return_tensors="pt", padding=True)
            generated_text = self.__model.generate(
            vision_x=imgs,
            lang_x=lang_x["input_ids"].to("cuda"),
            attention_mask=lang_x["attention_mask"].to("cuda", dtype=torch.float16),
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
            )
            output = self.__model.text_tokenizer.decode(generated_text[0])
            output = [x for x in output.split(' ') if not x.startswith('<')]
            out_label = output.index('GPT:')
            outputs = [' '.join(output[out_label + 1:])]
        elif model_name=="owl":
            prompt_template = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {}\nAI:"

            prompts = [prompt_template.format(question_list)]
            inputs = vis_proc[2](text=prompts, images=[Image.fromarray(np.uint8(x))], return_tensors='pt')
            inputs = {k: v.to("cuda", dtype=torch.float32) if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            generate_kwargs = {
            'do_sample': False,            
            'top_k': 5,
            'max_length': max_new_tokens
            }
            with torch.no_grad():
                res = self.__model.generate(**inputs, **generate_kwargs)
            outputs = [vis_proc[1].decode(res.tolist()[0], skip_special_tokens=True)]
        elif model_name=="ofv2":
            vision_x = vis_proc[0](Image.fromarray(np.uint8(x))).unsqueeze(0).unsqueeze(0).unsqueeze(0).to("cuda", dtype=torch.float16)            
            prompts = [f"<image>Question: {question_list} Short answer:"]            
            lang_x = vis_proc[1](
            prompts,
            return_tensors="pt", padding=True,
            ).to("cuda")
            generated_text = self.__model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"].to("cuda"),
            attention_mask=lang_x["attention_mask"].to("cuda", dtype=torch.float16),
            max_new_tokens=max_new_tokens,
            num_beams=3,pad_token_id=vis_proc[1].eos_token_id
            )
            outputs = vis_proc[1].batch_decode(generated_text, skip_special_tokens=True)
            outputs = [y[len(x)-len('<image>'):].strip() for x, y in zip(prompts, outputs)]
        elif model_name == "internlm":
            Image.fromarray(np.uint8(x)).save("./internlm{}.png".format(vis_proc[2])) 
            image_list= "./internlm{}.png".format(vis_proc[2])  
            texts=f" <|User|>:<ImageHere> {question_list}" + vis_proc[1] + " <|Bot|>:"
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs=[vis_proc[0](self.__model,texts, image_list,max_new_tokens=max_new_tokens) ]  
        elif model_name == "Qwen":
            Image.fromarray(np.uint8(x)).save("./Qwen{}_{}.png".format(vis_proc[1][0],vis_proc[1][1])) 
            image_list= "./Qwen{}_{}.png".format(vis_proc[1][0],vis_proc[1][1])
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = [vis_proc[0](image_list, question_list, max_new_tokens)]   
        
        elif model_name=="llava" or model_name=="llava15" or model_name=="moellava" or model_name=="sharegpt4v":
            outputs = self.__model([question_list],[Image.fromarray(np.uint8(x))],stop_str=vis_proc[0], dtype=torch.float16, max_new_tokens=max_new_tokens,method=vis_proc[1], level=vis_proc[2],image_listnew=None)


        if self.task=="cls" or self.task=="ocr":
            predictions = remove_special_chars(outputs[0]).lower()
        else:
            predictions = outputs[0]


        is_adversarial, is_best, distance = self.__is_adversarial(
            x, predictions, in_bounds,flag=flag
        )
        
        if return_details:
            return predictions, is_adversarial, is_best, distance
        else:
            return predictions, is_adversarial

