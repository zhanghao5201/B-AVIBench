source /mnt/petrelfs/zhanghao1/.bashrccog_2
source activate VLP_web_cog_2 


PARTITION=$1
JOB_NAME=$2
model_name=$3
batch_size=$4
dataset_name=$5
task=$6
attack=$7

PYTHONPATH="/mnt/petrelfs/zhanghao1/zhanghao5201/B-AVIBench/text_attack_tool/src":$PYTHONPATH 
srun --partition=$PARTITION  --time=1-10:10:00 --quotatype=reserved \
--mpi=pmi2 \
--gres=gpu:1 \
--job-name=${JOB_NAME} \
--kill-on-bad-exit=1 \
python eval_text_attack.py --model_name $model_name --batch_size $batch_size --dataset_name $dataset_name --task $task --attack $attack


# sh text_attack.sh gvembodied text_1 moellava 32 CIFAR10,CIFAR100,Flowers102,ImageNet,OxfordIIITPet classification textbugger
# sh text_attack.sh gvembodied text_1 moellava 32 CIFAR10,CIFAR100,Flowers102,ImageNet,OxfordIIITPet classification semantic



###dataset and task
# --dataset_name CIFAR10,CIFAR100,Flowers102,ImageNet,OxfordIIITPet --task classification
# --dataset_name COCO-Text,CTW,CUTE80,HOST,IC13,IC15 --task OCR
# --dataset_name NoCaps,Flickr,MSCOCO_caption_karpathy,WHOOPSCaption --task caption 
# --dataset_name IconQA,ScienceQAIMG,AOKVQAClose,WHOOPSWeird --task vqa 
# --dataset_name MSCOCO_pope_random,MSCOCO_pope_adversarial,MSCOCO_pope_popular --task object




###attack
#  --dataset_name COCO-Text,CTW,CUTE80,HOST,IC13,IC15 --task OCR --attack textbugger
#  --dataset_name COCO-Text,CTW,CUTE80,HOST,IC13,IC15 --task OCR --attack deepwordbug
#  --dataset_name COCO-Text,CTW,CUTE80,HOST,IC13,IC15 --task OCR --attack pruthi 
#  --dataset_name COCO-Text,CTW,CUTE80,HOST,IC13,IC15 --task OCR --attack bertattack
#  --dataset_name COCO-Text,CTW,CUTE80,HOST,IC13,IC15 --task OCR --attack textfooler
#  --dataset_name COCO-Text,CTW,CUTE80,HOST,IC13,IC15 --task OCR --attack pwws
#  --dataset_name COCO-Text,CTW,CUTE80,HOST,IC13,IC15 --task OCR --attack checklist
#  --dataset_name COCO-Text,CTW,CUTE80,HOST,IC13,IC15 --task OCR --attack stresstest
#  --dataset_name COCO-Text,CTW,CUTE80,HOST,IC13,IC15 --task OCR --attack input-reduction
#  --dataset_name COCO-Text,CTW,CUTE80,HOST,IC13,IC15 --task OCR --attack semantic

# textbugger, deepwordbug, Character-level: pruthi
# bertattack, textfooler, #Word-level pwws
# checklist, stresstest, #Sentence-level input-reduction
# semantic