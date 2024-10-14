# **News**
2024/10/11 Code is open source.

# **Data**
The basic data used for text attacks that we collated can be downloaded in [here](https://drive.google.com/file/d/1Lv8tYLtatYacuhxGdWaf4l11dng-fgOl/view?usp=drive_link).

# **Evaluation**
```bash
sh image_attack.sh $PARTITION $JOB_NAME $model_name $batch_size $dataset_name
# sh image_attack.sh gvembodied test_corrup moellava 16  ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape
# sh image_attack.sh gvembodied test_corrup moellava 16  ScienceQAIMG,IconQA,VSR
# sh image_attack.sh gvembodied test_corrup moellava 16  COCO-Text,CTW,CUTE80,HOST,IC13,IC15,IIIT5K,SVTP,SVT,Total-Text,WOST,WordArt
# sh image_attack.sh gvembodied test_corrup moellava 16  FUNSD,POIE,SROIE
# sh image_attack.sh gvembodied test_corrup moellava 16  MSCOCO_MCI,VCR1_MCI,MSCOCO_OC,VCR1_OC
# sh image_attack.sh gvembodied test_corrup moellava 16  MSCOCO_pope_random,MSCOCO_pope_adversarial,MSCOCO_pope_popular
# sh image_attack.sh gvembodied test_corrup moellava 16  AOKVQAClose,AOKVQAOpen,DocVQA,GQA,OCRVQA,OKVQA,STVQA,TextVQA,VizWiz,WHOOPSVQA,WHOOPSWeird,Visdial
# sh image_attack.sh gvembodied test_corrup moellava 16  CIFAR10,CIFAR100,Flowers102,ImageNet,OxfordIIITPet
```

**model_name**: BLIP2; MiniGPT-4; mPLUG-Owl; Otter; Otter-Image; InstructBLIP; VPGTrans; LLaVA; sharegpt4v; moellava; LLaVA15; LLaMA-Adapter-v2; internlm-xcomposer; PandaGPT; OFv2.


**dataset_name**:
Visual perception--image_cls:ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape.
Visual perception--MCI and OC: MSCOCO_MCI,VCR1_MCI,MSCOCO_OC,VCR1_OC.

Visual knowledge acquisition--KIE: FUNSD,POIE,SROIE;
Visual knowledge acquisition--OCR: COCO-Text,CTW,CUTE80,HOST,IC13,IC15,IIIT5K,SVTP,SVT,Total-Text,WOST,WordArt.

Visual reasoning--VQA: AOKVQAClose,AOKVQAOpen,DocVQA,GQA,OCRVQA,OKVQA,STVQA,TextVQA,WHOOPSVQA,WHOOPSWeird,Visdial,IconQA,VSR;
Visual reasoning--KGID: ScienceQAIMG,VizWiz.

Visual commonsense: ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape.

Object hallucination: MSCOCO_pope_random,MSCOCO_pope_adversarial,MSCOCO_pope_popular.


When the model is tested and "tiny_answers" is generated under this folder, the code at [here](https://github.com/zhanghao5201/B-AVIBench/tree/main/result_process/image_attack.py) can be used to process and analyze the results.

