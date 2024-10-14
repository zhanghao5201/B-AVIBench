from abc import abstractmethod
from collections.abc import Iterable
import functools
import numpy as np
from .adversarial import Adversarial
import random
from .distances import Distance
from .distances import MSE
import warnings
rng = random.Random()
nprng = np.random.RandomState()
from models import llama_adapter_v2 as llama
import pdb
def set_seeds(seed):
    """Sets the seeds of both random number generators used by Foolbox.

    Parameters
    ----------
    seed : int
        The seed for both random number generators.

    """
    rng.seed(seed)
    nprng.seed(seed)


class AdditiveGaussianNoiseAttack():
    """Adds Gaussian noise to the input, gradually increasing
    the standard deviation until the input is misclassified.

    """
    def __init__(self, model,task, distance=MSE, threshold=None): 
        self._default_model = model
        self._default_task = task
        self._default_distance = distance
        self._default_threshold = threshold
   
    def __call__(self, input_or_adv, label=None, task_name=None,epsilons=1000, unpack=True, question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None):
       
        assert input_or_adv is not None

        if 1:
            if label is None:
                raise ValueError(
                    "Label must be passed when input_or_adv is"
                    " not an Adversarial instance"
                )
            else:
                model = self._default_model
                distance = self._default_distance
                threshold = self._default_threshold
                a = Adversarial(
                    model,
                    task_name,
                    input_or_adv,
                    label,
                    distance=distance,
                    threshold=threshold ,question_list=question_list, chat_list=chat_list, max_new_tokens=max_new_tokens,model_name=model_name ,vis_proc=vis_proc                
                )

        assert a is not None

        if a.distance.value == 0.0:
            warnings.warn(
                "Not running the attack because the original input"
                " is already misclassified and the adversarial thus"
                " has a distance of 0."
            )
        elif a.reached_threshold():
            warnings.warn(
                "Not running the attack because the given treshold"
                " is already reached"
            )
        else:
            try:
                _ = call_fn(a, label=label, unpack=unpack, epsilons=1000,question_list=question_list, chat_list=chat_list, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)

                assert _ is None, "decorated __call__ method must return None"
            except StopAttack: 
                logging.info("threshold reached, stopping attack")
        if a.perturbed is None:
            warnings.warn(
                " did not find an adversarial, maybe the model"
                " or the criterion is not supported by this"
                " attack."
            )

        if unpack:
            return a.perturbed            
        else:
            return a

    
def call_fn(input_or_adv, label=None, unpack=True, epsilons=1000,question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None):
        """Adds uniform or Gaussian noise to the input, gradually increasing
        the standard deviation until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        epsilons : int or Iterable[float]
            Either Iterable of noise levels or number of noise levels
            between 0 and 1 that should be tried.

        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        x = a.unperturbed
        bounds = a.bounds()
        min_, max_ = bounds

        if not isinstance(epsilons, Iterable):
            epsilons = np.linspace(0, 1, num=epsilons + 1)[1:]
        for epsilon in epsilons:
            noise = AdditiveGaussianNoise_sample_noise(epsilon, x, bounds)
            perturbed = x + epsilon * noise
            perturbed = np.clip(perturbed, min_, max_)
            _, is_adversarial = a.forward_one(perturbed,question_list=question_list, chat_list=chat_list, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)

            if is_adversarial:
                return #找到对抗样本就报错

def AdditiveGaussianNoise_sample_noise(epsilon, x, bounds):
        
        min_, max_ = bounds
        std = epsilon / np.sqrt(3) * (max_ - min_)
        noise = nprng.normal(scale=std, size=x.shape)
        noise = noise.astype(x.dtype)
        return noise
  