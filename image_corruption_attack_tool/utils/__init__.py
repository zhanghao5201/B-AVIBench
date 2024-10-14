from .ocr import evaluate_OCR
from .vqa import evaluate_VQA,evaluate_VQA_bias
from .caption import evaluate_Caption
from .kie import evaluate_KIE
from .mrr import evaluate_MRR
from .embodied import evaluate_embodied
from .classification import evaluate_zero_shot_image_classification


# african_funeral,american_funeral,chinese_funeral,japanese_funeral
# african_wedding,american_wedding,chinese_wedding,japanese_wedding
# black,white,yellow
# chef,dentist,doctor,lawyer,painter,police_officer,scientist,singer,software_developer,teacher,
# harassment,hate,self_harm,sexual,shocking,violence
dataset_task_dict = {
    # Caption Datasets
    'NoCaps': (evaluate_Caption, 'Caption'),
    'Flickr': (evaluate_Caption, 'Caption'),
    'MSCOCO_caption': (evaluate_Caption, 'Caption'),
    'MSCOCO_caption_karpathy': (evaluate_Caption, 'Caption'),
    # KIE Datasets
    'SROIE': (evaluate_KIE, 'VQA'),
    'FUNSD': (evaluate_KIE, 'VQA'),
    'POIE': (evaluate_KIE, 'VQA'),
    # VQA Datasets
    'TextVQA': (evaluate_VQA, 'VQA'),
    'DocVQA': (evaluate_VQA, 'VQA'),
    'OCRVQA': (evaluate_VQA, 'VQA'),
    'STVQA': (evaluate_VQA, 'VQA'),
    'OKVQA': (evaluate_VQA, 'VQA'),
    'AOKVQAOpen': (evaluate_VQA, 'VQA'),
    'AOKVQAClose': (evaluate_VQA, 'VQA'),
    'GQA': (evaluate_VQA, 'VQA'),
    'VizWiz': (evaluate_VQA, 'VQA'),
    'VQAv2': (evaluate_VQA, 'VQA'),
    'VQAv1': (evaluate_VQA, 'VQA'),
    'Visdial': (evaluate_MRR, 'VQA'),
    # VQA (binary answer)
    'VSR': (evaluate_VQA, 'Binary'),
    'HatefulMemes': (evaluate_VQA, 'Binary'),
    ###new
    'african_funeral': (evaluate_VQA_bias, 'Binary'),
    'american_funeral': (evaluate_VQA_bias, 'Binary'),
    'chinese_funeral': (evaluate_VQA_bias, 'Binary'),
    'japanese_funeral': (evaluate_VQA_bias, 'Binary'),
    'african_wedding': (evaluate_VQA_bias, 'Binary'),
    'american_wedding': (evaluate_VQA_bias, 'Binary'),
    'chinese_wedding': (evaluate_VQA_bias, 'Binary'),
    'japanese_wedding': (evaluate_VQA_bias, 'Binary'),
    'chef':(evaluate_VQA_bias, 'Binary'),
    'dentist':(evaluate_VQA_bias, 'Binary'),
    'doctor':(evaluate_VQA_bias, 'Binary'),
    'lawyer':(evaluate_VQA_bias, 'Binary'),
    'painter':(evaluate_VQA_bias, 'Binary'),
    'police_officer':(evaluate_VQA_bias, 'Binary'),
    'scientist':(evaluate_VQA_bias, 'Binary'),
    'singer':(evaluate_VQA_bias, 'Binary'),
    'software_developer':(evaluate_VQA_bias, 'Binary'),
    'teacher':(evaluate_VQA_bias, 'Binary'),
    'harassment':(evaluate_VQA_bias, 'Binary'),
    'hate':(evaluate_VQA_bias, 'Binary'),
    'self_harm':(evaluate_VQA_bias, 'Binary'),
    'sexual':(evaluate_VQA_bias, 'Binary'),
    'shocking':(evaluate_VQA_bias, 'Binary'),
    'violence': (evaluate_VQA_bias, 'Binary'),
    'black_poor': (evaluate_VQA_bias, 'Binary'),
    'black_prone': (evaluate_VQA_bias, 'Binary'),
    'black_fatheaded':(evaluate_VQA_bias, 'Binary'),
    'black_uncreative':(evaluate_VQA_bias, 'Binary'),
    'black_artistic':(evaluate_VQA_bias, 'Binary'),
    'white_poor':(evaluate_VQA_bias, 'Binary'),
    'white_prone': (evaluate_VQA_bias, 'Binary'),
    'white_fatheaded':(evaluate_VQA_bias, 'Binary'),
    'white_uncreative':(evaluate_VQA_bias, 'Binary'),
    'white_artistic':(evaluate_VQA_bias, 'Binary'),
    'yellow_poor':(evaluate_VQA_bias, 'Binary'),
    'yellow_prone': (evaluate_VQA_bias, 'Binary'),
    'yellow_fatheaded':(evaluate_VQA_bias, 'Binary'),
    'yellow_uncreative':(evaluate_VQA_bias, 'Binary'),
    'yellow_artistic':(evaluate_VQA_bias, 'Binary'),

    # VQA (multi choice)
    'IconQA': (evaluate_VQA, 'Multi'),
    'ScienceQA': (evaluate_VQA, 'Multi'),
    'ScienceQAIMG': (evaluate_VQA, 'Multi'),
    # Embodied Datasets
    "MetaWorld": (evaluate_embodied, 'Embodied'),
    "FrankaKitchen": (evaluate_embodied, 'Embodied'),
    "Minecraft": (evaluate_embodied, 'Embodied'),
    "VirtualHome": (evaluate_embodied, 'Embodied'),
    "MinecraftPolicy": (evaluate_embodied, 'Embodied'),
    # classification
    'ImageNet': (evaluate_zero_shot_image_classification, 'VQA'),
    'CIFAR10': (evaluate_zero_shot_image_classification, 'VQA'),
    'CIFAR100': (evaluate_zero_shot_image_classification, 'VQA'),
    'OxfordIIITPet': (evaluate_zero_shot_image_classification, 'VQA'),
    'Flowers102': (evaluate_zero_shot_image_classification, 'VQA'),
    # whoops
    'WHOOPSCaption': (evaluate_Caption, 'Caption'),
    'WHOOPSVQA': (evaluate_VQA, 'VQA'),
    'WHOOPSWeird': (evaluate_VQA, 'VQA'),
    # ImageNetVC
    'ImageNetVC_color': (evaluate_VQA, 'VQA'),
    'ImageNetVC_shape': (evaluate_VQA, 'VQA'),
    'ImageNetVC_material': (evaluate_VQA, 'VQA'),
    'ImageNetVC_component': (evaluate_VQA, 'VQA'),
    'ImageNetVC_others': (evaluate_VQA, 'VQA'),
     # Object Hallucination
    'MSCOCO_pope_random': (evaluate_VQA, 'VQA'),
    'MSCOCO_pope_popular': (evaluate_VQA, 'VQA'),
    'MSCOCO_pope_adversarial': (evaluate_VQA, 'VQA'),
    # VCR, POPE
    'VCR1_OC': (evaluate_VQA, 'VQA'),
    'VCR1_MCI': (evaluate_VQA, 'VQA'),
    'MSCOCO_MCI': (evaluate_VQA, 'VQA'),
    'MSCOCO_OC': (evaluate_VQA, 'VQA'),
    # OCR
    "COCO-Text": (evaluate_OCR, 'VQA'),
    "CTW": (evaluate_OCR, 'VQA'),
    "CUTE80": (evaluate_OCR, 'VQA'),
    "HOST": (evaluate_OCR, 'VQA'),
    "IC13": (evaluate_OCR, 'VQA'),
    "IC15": (evaluate_OCR, 'VQA'),
    "IIIT5K": (evaluate_OCR, 'VQA'),
    "SVTP": (evaluate_OCR, 'VQA'),
    "SVT": (evaluate_OCR, 'VQA'),
    "Total-Text": (evaluate_OCR, 'VQA'),
    "WOST": (evaluate_OCR, 'VQA'),
    "WordArt": (evaluate_OCR, 'VQA'),
}