# **News**
2024/10/11 Code is open source.

# **Data**
The image corruption data used in B-AVIBench Evaluation that we made can be downloaded in [B-AVIBench_corruption_1](https://huggingface.co/zhanghao520/B-AVIBench_data/blob/main/B-AVIBench_corruption_1.tar.gz), [B-AVIBench_corruption_3](https://huggingface.co/zhanghao520/B-AVIBench_data/blob/main/B-AVIBench_corruption_3.tar.gz), [B-AVIBench_corruption_5](https://huggingface.co/zhanghao520/B-AVIBench_data/blob/main/B-AVIBench_corruption_5.tar.gz). You can download, and then unpack and move it to `$B-AVIBench/eval_data/corruption/B-AVIBench_corruption_1/`,`$B-AVIBench/eval_data/corruption/B-AVIBench_corruption_3/`, `$B-AVIBench/eval_data/corruption/B-AVIBench_corruption_5/`. 1,3, and 5 respectively represent the corresponding three corruption levels in the paper.

# **Evaluation**
```bash
sh image_corruption.sh $PARTITION $JOB_NAME $model_name $batch_size $dataset_name
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


**dataset_name**:
Visual perception--image_cls:ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape;
Visual perception--MCI and OC: MSCOCO_MCI,VCR1_MCI,MSCOCO_OC,VCR1_OC.

Visual knowledge acquisition--KIE: FUNSD,POIE,SROIE;
Visual knowledge acquisition--OCR: COCO-Text,CTW,CUTE80,HOST,IC13,IC15,IIIT5K,SVTP,SVT,Total-Text,WOST,WordArt;
Visual knowledge acquisition--Image Caption: NoCaps,Flickr,MSCOCO_caption_karpathy,WHOOPSCaption.

Visual reasoning--VQA: AOKVQAClose,AOKVQAOpen,DocVQA,GQA,OCRVQA,OKVQA,STVQA,TextVQA,WHOOPSVQA,WHOOPSWeird,Visdial,IconQA,VSR;
Visual reasoning--KGID: ScienceQAIMG,VizWiz.

Visual commonsense: ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape.

Object hallucination: MSCOCO_pope_random,MSCOCO_pope_adversarial,MSCOCO_pope_popular.

When the model is tested and "tiny_answers" is generated under this folder, the code at [here](https://github.com/zhanghao5201/B-AVIBench/tree/main/result_process/image_corruption.py) can be used to process and analyze the results.

