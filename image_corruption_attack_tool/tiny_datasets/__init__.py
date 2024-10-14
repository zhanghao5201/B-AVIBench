DATA_DIR = '/mnt/petrelfs/zhanghao1'

import os
import pickle
from functools import partial
from torch.utils.data import Dataset

from .ocr_datasets import ocrDataset
from .caption_datasets import NoCapsDataset, FlickrDataset, COCOCaptionDataset, COCOCaptionKarpathyDataset
from .kie_datasets import SROIEDataset, FUNSDDataset, POIEDataset
from .embod_datasets import EmbodiedDataset
from .cls_datasets import ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset, OxfordIIITPet, Flowers102
from .whoops import WHOOPSCaptionDataset, WHOOPSVQADataset, WHOOPSWeirdDataset
from .vqa_datasets import (
    TextVQADataset, DocVQADataset, OCRVQADataset, STVQADataset,
    ScienceQADataset, OKVQADataset, GQADataset, VizWizDataset,
    VQAv2Dataset, VQAv1Dataset, VisdialDataset, IconQADataset,
    VSRDataset, VCR1_MCIDataset, VCR1_OCDataset, MSCOCO_MCIDataset,
    MSCOCO_OCDataset, MSCOCO_POPEDataset, MSCOCO_POPEDataset_adversarial,
    MSCOCO_POPEDataset_popular, AOKVQAOpenDataset, AOKVQACloseDataset, HatefulMemes, ScienceQAIMGDataset,
)

class GeneralDataset(Dataset):
    def __init__(
        self,
        dataset_name
    ):
        self.dataset_name = dataset_name
        print("oo",f"{DATA_DIR}/tiny_lvlm_datasets/{dataset_name}/dataset.pkl")
        self.dataset = pickle.load(open(f"{DATA_DIR}/tiny_lvlm_datasets/{dataset_name}/dataset.pkl", 'rb'))
        print(len(self.dataset),"kk")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # print("f",sample['image_path'])
        sample['image_path'] = f"{DATA_DIR}/{sample['image_path']}"
        # print(sample,"ff")
        return sample

class GeneralDataset_bias(Dataset):
    def __init__(
        self,
        dataset_name,time
    ):
        self.dataset_name = dataset_name
        if self.dataset_name.endswith("poor") or self.dataset_name.endswith("prone") or self.dataset_name.endswith("fatheaded") or self.dataset_name.endswith("uncreative") or self.dataset_name.endswith("artistic"):
            data_name=self.dataset_name.split('_')[0]
        else:
            data_name=self.dataset_name
        print("oo",f"/mnt/petrelfs/zhanghao1/attack_dataset/{data_name}/dataset.pkl")
        self.dataset = pickle.load(open(f"/mnt/petrelfs/zhanghao1/attack_dataset/{data_name}/dataset.pkl", 'rb'))
        self.time=time
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.dataset_name.endswith('poor') or self.dataset_name.endswith('prone') or self.dataset_name.endswith('fatheaded') or self.dataset_name.endswith('uncreative') or self.dataset_name.endswith('artistic'):
            sample['question'] = sample[self.dataset_name.split('_')[-1]][self.time]+"Your answer can only be one word, either 'no' or 'yes'."
        else:
            sample['question'] = sample['question'][self.time]+"Your answer can only be one word, either 'no' or 'yes'."# or 'Unable to answer' or 'uncertain'
            
        return sample


dataset_class_dict = {
    # Caption Datasets
    'NoCaps': NoCapsDataset,
    'Flickr': FlickrDataset,
    'MSCOCO_caption': COCOCaptionDataset,
    'MSCOCO_caption_karpathy': COCOCaptionKarpathyDataset,
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
    'ScienceQAIMG': ScienceQAIMGDataset,
    'OKVQA': OKVQADataset,
    'AOKVQAOpen': AOKVQAOpenDataset,
    'AOKVQAClose': AOKVQACloseDataset,
    'GQA': GQADataset,
    'VizWiz': VizWizDataset,
    'VQAv2': VQAv2Dataset,
    'VQAv1': VQAv1Dataset,
    'Visdial': VisdialDataset,
    'IconQA': IconQADataset,
    'VSR': VSRDataset,
    'HatefulMemes': HatefulMemes,    
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
    # VCR, POPE
    'VCR1_OC': VCR1_OCDataset,
    'VCR1_MCI': VCR1_MCIDataset,
    'MSCOCO_MCI': MSCOCO_MCIDataset,
    'MSCOCO_OC': MSCOCO_OCDataset,
    'MSCOCO_pope_random': MSCOCO_POPEDataset,
    'MSCOCO_pope_popular': MSCOCO_POPEDataset_popular,
    'MSCOCO_pope_adversarial': MSCOCO_POPEDataset_adversarial,    
    # OCR
    "COCO-Text": partial(ocrDataset, dataset_name="COCO-Text"),
    "CTW": partial(ocrDataset, dataset_name="CTW"),
    "CUTE80": partial(ocrDataset, dataset_name="CUTE80"),
    "HOST": partial(ocrDataset, dataset_name="HOST"),
    "IC13": partial(ocrDataset, dataset_name="IC13"),
    "IC15": partial(ocrDataset, dataset_name="IC15"),
    "IIIT5K": partial(ocrDataset, dataset_name="IIIT5K"),
    "SVTP": partial(ocrDataset, dataset_name="SVTP"),
    "SVT": partial(ocrDataset, dataset_name="SVT"),
    "Total-Text": partial(ocrDataset, dataset_name="Total-Text"),
    "WOST": partial(ocrDataset, dataset_name="WOST"),
    "WordArt": partial(ocrDataset, dataset_name="WordArt"),

}
