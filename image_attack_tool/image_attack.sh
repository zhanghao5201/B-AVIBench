source /mnt/petrelfs/zhanghao1/.bashrccog_2
source activate VLP_web_cog_2

PARTITION=$1
JOB_NAME=$2
model_name=$3
batch_size=$4
dataset_name=$5
device=${device:-0}
PYTHONPATH="/mnt/petrelfs/zhanghao1/holistic_imageatt/src":$PYTHONPATH 
srun --partition=$PARTITION --time=1-10:10:00 --quotatype=reserved \
--mpi=pmi2 \
--gres=gpu:1 \
--job-name=${JOB_NAME} \
--kill-on-bad-exit=1 \
python eval_image_attack.py --model_name $model_name --device $device --batch_size $batch_size --dataset_name $dataset_name

#sh image_attack.sh gvembodied eval1 LLaVA15 16 COCO-Text


#