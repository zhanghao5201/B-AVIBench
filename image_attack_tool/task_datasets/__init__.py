DATA_DIR = '/nvme/share/xupeng/datasets'

from .ocr_datasets import ocrDataset
from .caption_datasets import NoCapsDataset, FlickrDataset
from .kie_datasets import SROIEDataset, FUNSDDataset, POIEDataset
from .embod_datasets import EmbodiedDataset
from .cls_datasets import ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset, OxfordIIITPet, Flowers102
from .whoops import WHOOPSCaptionDataset, WHOOPSVQADataset, WHOOPSWeirdDataset
from .vqa_datasets import (
    TextVQADataset, DocVQADataset, OCRVQADataset, STVQADataset,
    ScienceQADataset, OKVQADataset, GQADataset, VizWizDataset,
    VQAv2Dataset, VQAv1Dataset, VisdialDataset, IconQADataset,
    VSRDataset, SplitOCRVQADataset,ImageNetVC,
    MSCOCO_POPEDataset_random, MSCOCO_POPEDataset_popular, MSCOCO_POPEDataset_adversarial
)

from functools import partial


dataset_class_dict = {
    # Caption Datasets
    'NoCaps': NoCapsDataset,
    'Flickr': FlickrDataset,
    # KIE Datasets
    'SROIE': SROIEDataset,
    'FUNSD': FUNSDDataset,
    'POIE': POIEDataset,
    # VQA Datasets
    'TextVQA': TextVQADataset,
    'DocVQA': DocVQADataset,
    'OCRVQA': OCRVQADataset,
    'STVQA': STVQADataset,
    'ScienceQA': ScienceQADataset,
    'OKVQA': OKVQADataset,
    'GQA': GQADataset,
    'VizWiz': VizWizDataset,
    'VQAv2': VQAv2Dataset,
    'VQAv1': VQAv1Dataset,
    'Visdial': VisdialDataset,
    'IconQA': IconQADataset,
    'VSR': VSRDataset,
    # Embodied Datasets
    "MetaWorld": partial(EmbodiedDataset, dataset_name="MetaWorld"),
    "FrankaKitchen": partial(EmbodiedDataset, dataset_name="FrankaKitchen"),
    "Minecraft": partial(EmbodiedDataset, dataset_name="Minecraft"),
    "VirtualHome": partial(EmbodiedDataset, dataset_name="VirtualHome"),
    "MinecraftPolicy": partial(EmbodiedDataset, dataset_name="MinecraftPolicy"),
    # classification
    'ImageNet': ImageNetDataset,
    'CIFAR10': CIFAR10Dataset,
    'CIFAR100': CIFAR100Dataset,
    'OxfordIIITPet': OxfordIIITPet,
    'Flowers102': Flowers102,
    # whoops
    'WHOOPSCaption': WHOOPSCaptionDataset,
    'WHOOPSVQA': WHOOPSVQADataset,
    'WHOOPSWeird': WHOOPSWeirdDataset,
        # ImageNetVC
    'ImageNetVC_color': partial(ImageNetVC, task='color'),
    'ImageNetVC_shape': partial(ImageNetVC, task='shape'),
    'ImageNetVC_material': partial(ImageNetVC, task='material'),
    'ImageNetVC_component': partial(ImageNetVC, task='component'),
    'ImageNetVC_others': partial(ImageNetVC, task='others'),
    # Object Hallucination
    'MSCOCO_pope_random': MSCOCO_POPEDataset_random,
    'MSCOCO_pope_popular': MSCOCO_POPEDataset_popular,
    'MSCOCO_pope_adversarial': MSCOCO_POPEDataset_adversarial,
    # split OCRVQA
    "OCR0": partial(SplitOCRVQADataset, index=0),
    "OCR1": partial(SplitOCRVQADataset, index=1),
    "OCR2": partial(SplitOCRVQADataset, index=2),
    "OCR3": partial(SplitOCRVQADataset, index=3),
    "OCR4": partial(SplitOCRVQADataset, index=4),
    "OCR5": partial(SplitOCRVQADataset, index=5),
    "OCR6": partial(SplitOCRVQADataset, index=6),
    "OCR7": partial(SplitOCRVQADataset, index=7),
}
