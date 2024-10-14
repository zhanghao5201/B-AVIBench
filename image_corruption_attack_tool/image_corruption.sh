source /mnt/petrelfs/zhanghao1/.bashrccog_2
source activate VLP_web_cog_2 


PARTITION=$1
JOB_NAME=$2
model_name=$3
batch_size=$4
dataset_name=$5
PYTHONPATH="/mnt/petrelfs/zhanghao1/zhanghao5201/B-AVIBench/image_corruption_attack_tool/src":$PYTHONPATH 
srun --partition=$PARTITION --time=1-10:10:00 --quotatype=reserved \
--mpi=pmi2 \
--gres=gpu:1 \
--job-name=${JOB_NAME} \
--kill-on-bad-exit=1 \
python eval_corruption.py --model_name $model_name --batch_size $batch_size --dataset_name $dataset_name

#modelname: 

# sh image_corruption.sh gvembodied test_corrup moellava 16  ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape
# sh image_corruption.sh gvembodied test_corrup moellava 16  AOKVQAClose,AOKVQAOpen,DocVQA,GQA,OCRVQA,OKVQA,STVQA,TextVQA,VizWiz,WHOOPSVQA,WHOOPSWeird,Visdial
# sh image_corruption.sh gvembodied test_corrup moellava 16  FUNSD,POIE,SROIE 
# sh image_corruption.sh gvembodied test_corrup moellava 16  COCO-Text,CTW,CUTE80,HOST,IC13,IC15,IIIT5K,SVTP,SVT,Total-Text,WOST,WordArt
# sh image_corruption.sh gvembodied test_corrup moellava 16  ScienceQAIMG,IconQA,VSR  
# sh image_corruption.sh gvembodied test_corrup moellava 16  CIFAR10,CIFAR100,Flowers102,ImageNet,OxfordIIITPet
# sh image_corruption.sh gvembodied test_corrup moellava 16  NoCaps,Flickr,MSCOCO_caption_karpathy,WHOOPSCaption 
# sh image_corruption.sh gvembodied test_corrup moellava 16  MSCOCO_MCI,VCR1_MCI,MSCOCO_OC,VCR1_OC
# sh image_corruption.sh gvembodied test_corrup moellava 16  MSCOCO_pope_random,MSCOCO_pope_adversarial,MSCOCO_pope_popular




