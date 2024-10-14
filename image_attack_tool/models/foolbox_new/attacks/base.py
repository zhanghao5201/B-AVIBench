from typing import Callable, TypeVar, Any, Union, Optional, Sequence, List, Tuple, Dict
from typing_extensions import final, overload
from abc import ABC, abstractmethod
from collections.abc import Iterable
import eagerpy as ep

from ..models import Model

from ..criteria import Criterion
from ..criteria import Misclassification

from ..devutils import atleast_kd
import numpy as np
from ..distances import Distance
from PIL import Image
import pdb
import torch
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image
####
from models import llama_adapter_v2 as llama
from PIL import Image
from .tools import has_word, remove_special_chars
from .cider import CiderScorer
import pdb
from .tools import VQAEval
import imageio

####

T = TypeVar("T")
CriterionType = TypeVar("CriterionType", bound=Criterion)


# TODO: support manually specifying early_stop in __call__
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
    def result_string(self):
        prec, recall, f1 = self.score()
        return f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f}"

class Attack(ABC):
    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T, T]:
        ...

    @abstractmethod  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        **kwargs: Any,
    ) -> Union[Tuple[List[T], List[T], T], Tuple[T, T, T]]:
        # in principle, the type of criterion is Union[Criterion, T]
        # but we want to give subclasses the option to specify the supported
        # criteria explicitly (i.e. specifying a stricter type constraint)
        ...

    @abstractmethod
    def repeat(self, times: int) -> "Attack":
        ...

    def __repr__(self) -> str:
        args = ", ".join(f"{k.strip('_')}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({args})"


class AttackWithDistance(Attack):
    @property
    @abstractmethod
    def distance(self) -> Distance:
        ...

    def repeat(self, times: int) -> Attack:
        return Repeated(self, times)


class Repeated(AttackWithDistance):
    """Repeats the wrapped attack and returns the best result"""

    def __init__(self, attack: AttackWithDistance, times: int):
        if times < 1:
            raise ValueError(f"expected times >= 1, got {times}")  # pragma: no cover

        self.attack = attack
        self.times = times

    @property
    def distance(self) -> Distance:
        return self.attack.distance

    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T, T]:
        ...

    def __call__(  # noqa: F811
        self,
        model: Model,
        inputs: T,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        **kwargs: Any,
    ) -> Union[Tuple[List[T], List[T], T], Tuple[T, T, T]]:
        x, restore_type = ep.astensor_(inputs)
        del inputs

        criterion = get_criterion(criterion)

        was_iterable = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False

        N = len(x)
        K = len(epsilons)

        for i in range(self.times):
            # run the attack
            xps, xpcs, success = self.attack(
                model, x, criterion, epsilons=epsilons, **kwargs
            )
            assert len(xps) == K
            assert len(xpcs) == K
            for xp in xps:
                assert xp.shape == x.shape
            for xpc in xpcs:
                assert xpc.shape == x.shape
            assert success.shape == (K, N)

            if i == 0:
                best_xps = xps
                best_xpcs = xpcs
                best_success = success
                continue

            # TODO: test if stacking the list to a single tensor and
            # getting rid of the loop is faster

            for k, epsilon in enumerate(epsilons):
                first = best_success[k].logical_not()
                assert first.shape == (N,)
                if epsilon is None:
                    # if epsilon is None, we need the minimum

                    # TODO: maybe cache some of these distances
                    # and then remove the else part
                    closer = self.distance(x, xps[k]) < self.distance(x, best_xps[k])
                    assert closer.shape == (N,)
                    new_best = ep.logical_and(success[k], ep.logical_or(closer, first))
                else:
                    # for concrete epsilon, we just need a successful one
                    new_best = ep.logical_and(success[k], first)
                new_best = atleast_kd(new_best, x.ndim)
                best_xps[k] = ep.where(new_best, xps[k], best_xps[k])
                best_xpcs[k] = ep.where(new_best, xpcs[k], best_xpcs[k])

            best_success = ep.logical_or(success, best_success)

        best_xps_ = [restore_type(xp) for xp in best_xps]
        best_xpcs_ = [restore_type(xpc) for xpc in best_xpcs]
        if was_iterable:
            return best_xps_, best_xpcs_, restore_type(best_success)
        else:
            assert len(best_xps_) == 1
            assert len(best_xpcs_) == 1
            return (
                best_xps_[0],
                best_xpcs_[0],
                restore_type(best_success.squeeze(axis=0)),
            )

    def repeat(self, times: int) -> "Repeated":
        return Repeated(self.attack, self.times * times)


class FixedEpsilonAttack(AttackWithDistance):
    """Fixed-epsilon attacks try to find adversarials whose perturbation sizes
    are limited by a fixed epsilon"""

    @abstractmethod
    def run(
        self, model: Model, inputs: T,  *, epsilon: float, **kwargs: Any
    ) -> T:
        """Runs the attack and returns perturbed inputs.

        The size of the perturbations should be at most epsilon, but this
        is not guaranteed and the caller should verify this or clip the result.
        """
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T, T]:
        ...

    @final  # noqa: F811
    def __call__(  # type: ignore
        self,
        model: Model,
        inputs: T,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        **kwargs: Any,
    ) -> Union[Tuple[List[T], List[T], T], Tuple[T, T, T]]:

        x, restore_type = ep.astensor_(inputs)
        del inputs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(model)

        was_iterable = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False

        N = len(x)
        K = len(epsilons)

        # None means: just minimize, no early stopping, no limit on the perturbation size
        if any(eps is None for eps in epsilons):
            # TODO: implement a binary search
            raise NotImplementedError(
                "FixedEpsilonAttack subclasses do not yet support None in epsilons"
            )
        real_epsilons = [eps for eps in epsilons if eps is not None]
        del epsilons

        xps = []
        xpcs = []
        success = []
        for epsilon in real_epsilons:
            xp = self.run(model, x, criterion, epsilon=epsilon, **kwargs)

            # clip to epsilon because we don't really know what the attack returns;
            # alternatively, we could check if the perturbation is at most epsilon,
            # but then we would need to handle numerical violations;
            xpc = self.distance.clip_perturbation(x, xp, epsilon)
            is_adv = is_adversarial(xpc)

            xps.append(xp)
            xpcs.append(xpc)
            success.append(is_adv)

        # # TODO: the correction we apply here should make sure that the limits
        # # are not violated, but this is a hack and we need a better solution
        # # Alternatively, maybe can just enforce the limits in __call__
        # xps = [
        #     self.run(model, x, criterion, epsilon=epsilon, **kwargs)
        #     for epsilon in real_epsilons
        # ]

        # is_adv = ep.stack([is_adversarial(xp) for xp in xps])
        # assert is_adv.shape == (K, N)

        # in_limits = ep.stack(
        #     [
        #         self.distance(x, xp) <= epsilon
        #         for xp, epsilon in zip(xps, real_epsilons)
        #     ],
        # )
        # assert in_limits.shape == (K, N)

        # if not in_limits.all():
        #     # TODO handle (numerical) violations
        #     # warn user if run() violated the epsilon constraint
        #     import pdb

        #     pdb.set_trace()

        # success = ep.logical_and(in_limits, is_adv)
        # assert success.shape == (K, N)

        success_ = ep.stack(success)
        assert success_.shape == (K, N)

        xps_ = [restore_type(xp) for xp in xps]
        xpcs_ = [restore_type(xpc) for xpc in xpcs]

        if was_iterable:
            return xps_, xpcs_, restore_type(success_)
        else:
            assert len(xps_) == 1
            assert len(xpcs_) == 1
            return xps_[0], xpcs_[0], restore_type(success_.squeeze(axis=0))


class MinimizationAttack(AttackWithDistance):
    """Minimization attacks try to find adversarials with minimal perturbation sizes"""

    @abstractmethod
    def run(
        self,
        model: Model,
        inputs: T,
        *,
        early_stop: Optional[float] = None,task=None,label=None,
        **kwargs: Any,
    ) -> T:
        """Runs the attack and returns perturbed inputs.

        The size of the perturbations should be as small as possible such that
        the perturbed inputs are still adversarial. In general, this is not
        guaranteed and the caller has to verify this.
        """
        ...

    @overload
    def __call__(
        self,
        model: Model,
        inputs: T,
        *,
        epsilons: Sequence[Union[float, None]],
        **kwargs: Any,
    ) -> Tuple[List[T], List[T], T]:
        ...

    @overload  # noqa: F811
    def __call__(
        self,
        model: Model,
        inputs: T,
        *,
        epsilons: Union[float, None],
        **kwargs: Any,
    ) -> Tuple[T, T, T]:
        ...

    @final  # noqa: F811######yes
    def __call__(  # type: ignore
        self,
        model: Model,
        inputs: T,
        *,
        epsilons: Union[Sequence[Union[float, None]], float, None],
        question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None,        
        **kwargs: Any,
    ) -> Union[Tuple[List[T], List[T], T], Tuple[T, T, T]]:
        x, restore_type = ep.astensor_(inputs)
        del inputs

        is_adversarial = get_is_adversarial(model, question_list=question_list, chat_list=chat_list, max_new_tokens=max_new_tokens,model_name=model_name,task_name=self.task,label_name=self.original_class,vis_proc=vis_proc)

        was_iterable = True
        if not isinstance(epsilons, Iterable):
            epsilons = [epsilons]
            was_iterable = False

        N = len(x)
        K = len(epsilons)

        # None means: just minimize, no early stopping, no limit on the perturbation size
        if any(eps is None for eps in epsilons):
            early_stop = None
        else:
            early_stop = min(epsilons)

        xp = self.run(model, x, early_stop=early_stop, question_list=question_list, chat_list=chat_list, max_new_tokens=max_new_tokens,model_name=model_name,task_name=self.task,label_name=self.original_class,vis_proc=vis_proc,**kwargs)
       
        xpcs = []
        success = []
        for idx,epsilon in enumerate(epsilons):            
            if epsilon is None:
                xpc = xp
            else:
                xpc = self.distance.clip_perturbation(x, xp, epsilon)
            is_adv = is_adversarial(xpc, question_list=question_list, chat_list=chat_list, max_new_tokens=max_new_tokens,model_name=model_name,task_name=self.task,label_name=self.original_class,vis_proc=vis_proc)
            
            xpcs.append(xpc)
            
            is_adv=ep.astensor(torch.Tensor([is_adv])).bool()
            success.append(is_adv)

        success_ = ep.stack(success)
        assert success_.shape == (K, N)
        xp_ = restore_type(xp)
        xpcs_ = [restore_type(xpc) for xpc in xpcs]

        new_image_raw = xpc.raw
        new_image_tensor = torch.tensor(new_image_raw, dtype=torch.float32) 

        inputs_new = to_pil_image(new_image_tensor.to(torch.uint8).squeeze(0))

        if was_iterable:
            return [xp_] * K, xpcs_, restore_type(success_),inputs_new
        else:
            assert len(xpcs_) == 1
            return xp_, xpcs_[0], restore_type(success_.squeeze(axis=0))


class FlexibleDistanceMinimizationAttack(MinimizationAttack):
    def __init__(self, *, distance: Optional[Distance] = None):
        self._distance = distance

    @property
    def distance(self) -> Distance:
        if self._distance is None:
            # we delay the error until the distance is needed,
            # e.g. when __call__ is executed (that way, run
            # can be used without specifying a distance)
            raise ValueError(
                "unknown distance, please pass `distance` to the attack initializer"
            )
        return self._distance


def get_is_adversarial(
    model: Model,question_list=None, chat_list=None, max_new_tokens=None,model_name=None,task_name=None,label_name=None,vis_proc=None
) -> Callable[[ep.Tensor], ep.Tensor]:
    def is_adversarial(perturbed: ep.Tensor,question_list=None, chat_list=None, max_new_tokens=None,model_name=None,task_name=None,label_name=None,vis_proc=None) -> ep.Tensor:
       
        new_image_raw = perturbed.raw

        new_image_tensor = torch.tensor(new_image_raw, dtype=torch.float32)        
        
        inputs = to_pil_image(new_image_tensor.squeeze(0))


        if model_name=="TestMiniGPT4" or model_name=="vpgtrans":
            chat_list_new=chat_list.copy()
            outputs = model.batch_answer([inputs], [question_list], [chat_list_new], max_new_tokens=max_new_tokens)            
            del chat_list_new  
        elif model_name=="blip2":
            imgs = vis_proc["eval"](inputs).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = f"Question: {question_list} Answer:" 
            outputs = model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        elif model_name=="instruct_blip":
            imgs = vis_proc["eval"](inputs).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = question_list 
            outputs = model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        elif model_name=="adv2":
            imgs = vis_proc(inputs).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = [llama.format_prompt(question_list)] 
            outputs = [model.generate(imgs, prompts, max_gen_len=max_new_tokens)[0].strip()]
        elif model_name=="panda":            
            Image.fromarray(np.uint8(inputs)).save("./panda4_{}.png".format(vis_proc[0]) )
            image_list= "./panda4_{}.png".format(vis_proc[0])        
            outputs = [model(image_list, question_list, max_new_tokens)]
        elif model_name=="otter": 
            imgs = vis_proc([Image.fromarray(np.uint8(inputs))],return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0).to("cuda", dtype=torch.float16)
            prompts = [f"<image> User: {question_list} GPT: <answer>"]
            lang_x = model.text_tokenizer(prompts, return_tensors="pt", padding=True)
            generated_text = model.generate(
            vision_x=imgs,
            lang_x=lang_x["input_ids"].to("cuda"),
            attention_mask=lang_x["attention_mask"].to("cuda", dtype=torch.float16),
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
            )
            output = model.text_tokenizer.decode(generated_text[0])
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
            'do_sample': True,
            'top_k': 5,
            'max_length': max_new_tokens
            }
            with torch.no_grad():
                res = model.generate(**inputs, **generate_kwargs)
            outputs = [vis_proc[1].decode(res.tolist()[0], skip_special_tokens=True)]
        elif model_name=="ofv2":
            vision_x = vis_proc[0](Image.fromarray(np.uint8(inputs))).unsqueeze(0).unsqueeze(0).unsqueeze(0).to("cuda", dtype=torch.float16)
            
            prompts = [f"<image>Question: {question_list} Short answer:"]            
            lang_x = vis_proc[1](
            prompts,
            return_tensors="pt", padding=True,
            ).to("cuda")
            generated_text = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"].to("cuda"),
            attention_mask=lang_x["attention_mask"].to("cuda", dtype=torch.float16),
            max_new_tokens=max_new_tokens,
            num_beams=3,pad_token_id=vis_proc[1].eos_token_id
            )
            outputs = vis_proc[1].batch_decode(generated_text, skip_special_tokens=True)
            outputs = [y[len(x)-len('<image>'):].strip() for x, y in zip(prompts, outputs)]
        elif model_name == "internlm":
            Image.fromarray(np.uint8(inputs)).save("./internlm4{}.png".format(vis_proc[2]))
            image_list= "./internlm4{}.png".format(vis_proc[2])  
            texts=f" <|User|>:<ImageHere> {question_list}" + vis_proc[1] + " <|Bot|>:"
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs=[vis_proc[0](model,texts, image_list,max_new_tokens=max_new_tokens)]        
            
        elif model_name=="llava" or model_name=="llava15" or model_name=="moellava" or model_name=="sharegpt4v":
            outputs = model([question_list],[Image.fromarray(np.uint8(inputs))],stop_str=vis_proc[0], dtype=torch.float16, max_new_tokens=max_new_tokens,method=vis_proc[1], level=vis_proc[2],image_listnew=None)
       
        if task_name=="cls" or task_name=="ocr":
            predict = remove_special_chars(outputs[0]).lower()
        else:
            predict = outputs[0]
        temp_result=predict

        if 1:
            are_adversarial=False
            if task_name=="cls" or task_name=="ocr":   
                adv=True
                if len(label_name)!=0 and not isinstance(label_name, str):
                    for gt in label_name:
                        are_adversarial_tmp=not (bool(has_word(temp_result, gt)) or bool(has_word(temp_result, gt+'s')) )
                        adv=adv and are_adversarial_tmp
                    are_adversarial= are_adversarial or adv
                else:
                    are_adversarial=not (bool(has_word(temp_result, label_name)) or bool(has_word(temp_result, label_name+'s')) )

            if task_name=="caption":
                cider_scorer = CiderScorer(n=4, sigma=6.0)
                cider_scorer += (temp_result, label_name)
                (score, scores) = cider_scorer.compute_score()
                if scores==0:
                    are_adversarial=True
            elif task_name=="kie":
                f1_scorer = F1Scorer()
                if isinstance(label_name, list) :
                    gt_answers =" ".join(label_name)
                else:
                    gt_answers=label_name
                f1_scorer.add_string(gt_answers, temp_result)
                prec, recall, f1 = f1_scorer.score()
                if f1==0:
                    are_adversarial=True                    
            elif task_name=="mrr":
                eval = VQAEval()
                mrr = eval.evaluate_MRR(temp_result, label_name)
                if mrr==0:
                    are_adversarial=True 
            elif task_name=="vqa" or task_name=="vqachoice" or task_name=="imagenetvc":
                eval = VQAEval()
                mrr = eval.evaluate(temp_result, label_name)
                if mrr==0:
                    are_adversarial=True         
        
        return are_adversarial

    return is_adversarial


def get_criterion(criterion: Union[Criterion, Any]) -> Criterion:
    if isinstance(criterion, Criterion):
        return criterion
    else:
        return Misclassification(criterion)


def get_channel_axis(model: Model, ndim: int) -> Optional[int]:
    data_format = getattr(model, "data_format", None)
    if data_format is None:
        return None
    if data_format == "channels_first":
        return 1
    if data_format == "channels_last":
        return ndim - 1
    raise ValueError(
        f"unknown data_format, expected 'channels_first' or 'channels_last', got {data_format}"
    )


def raise_if_kwargs(kwargs: Dict[str, Any]) -> None:
    if kwargs:
        raise TypeError(
            f"attack got an unexpected keyword argument '{next(iter(kwargs.keys()))}'"
        )
