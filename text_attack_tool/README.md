# **News**
2024/10/11 Code is open source.

# **Data**
The basic data used for text attacks that we collated can be downloaded in [here](https://drive.google.com/file/d/1Lv8tYLtatYacuhxGdWaf4l11dng-fgOl/view?usp=drive_link).

# **Evaluation**
```bash
sh task_attack.sh $PARTITION $JOB_NAME $model_name $batch_size $dataset_name
# sh image_corruption.sh gvembodied test_corrup moellava 16  ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape
# sh image_corruption.sh gvembodied test_corrup moellava 16  AOKVQAClose,AOKVQAOpen,DocVQA,GQA,OCRVQA,OKVQA,STVQA,TextVQA,VizWiz,WHOOPSVQA,WHOOPSWeird,Visdial
# sh image_corruption.sh gvembodied test_corrup moellava 16  FUNSD,POIE,SROIE 
# sh image_corruption.sh gvembodied test_corrup moellava 16  COCO-Text,CTW,CUTE80,HOST,IC13,IC15,IIIT5K,SVTP,SVT,Total-Text,WOST,WordArt
# sh image_corruption.sh gvembodied test_corrup moellava 16  ScienceQAIMG,IconQA,VSR  
# sh image_corruption.sh gvembodied test_corrup moellava 16  CIFAR10,CIFAR100,Flowers102,ImageNet,OxfordIIITPet
# sh image_corruption.sh gvembodied test_corrup moellava 16  NoCaps,Flickr,MSCOCO_caption_karpathy,WHOOPSCaption 
# sh image_corruption.sh gvembodied test_corrup moellava 16  MSCOCO_MCI,VCR1_MCI,MSCOCO_OC,VCR1_OC
# sh image_corruption.sh gvembodied test_corrup moellava 16  MSCOCO_pope_random,MSCOCO_pope_adversarial,MSCOCO_pope_popular
```

**model_name**: BLIP2; MiniGPT-4; mPLUG-Owl; Otter; Otter-Image; InstructBLIP; VPGTrans; LLaVA; sharegpt4v; moellava; LLaVA15; LLaMA-Adapter-v2; internlm-xcomposer; PandaGPT; OFv2.


**dataset_name and task**:
Visual perception--image_cls:ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape; Task: classification.

Visual knowledge acquisition--OCR: COCO-Text,CTW,CUTE80,HOST,IC13,IC15; Task: OCR.
Visual knowledge acquisition--Image Caption: NoCaps,Flickr,MSCOCO_caption_karpathy,WHOOPSCaption. Task: classification.

Visual reasoning--VQA: AOKVQAClose,WHOOPSWeird,IconQA; Task: vqa.
Visual reasoning--KGID: ScienceQAIMG; Task: vqa.

Object hallucination: MSCOCO_pope_random,MSCOCO_pope_adversarial,MSCOCO_pope_popular; Task: object.

**attack**:textbugger, deepwordbug, pruthi, bertattack, textfooler, pwws, checklist, stresstest, input-reduction, semantic.

When the model is tested and "tiny_answers" is generated under this folder, the code at [here](https://github.com/zhanghao5201/B-AVIBench/tree/main/result_process/text_attack.py) can be used to process and analyze the results.

