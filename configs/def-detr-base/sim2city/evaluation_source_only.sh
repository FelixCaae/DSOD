BATCH_SIZE=1
DATA_ROOT=./dataset
OUTPUT_DIR=./outputs/def-detr-base/sim2city/source_only/evaluation

CUDA_VISIBLE_DEVICES=1 python -u main.py \
--backbone resnet50 \
--enable_dino \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 4 \
--data_root ${DATA_ROOT} \
--source_dataset sim10k \
--target_dataset cityscapes \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--output_dir ${OUTPUT_DIR} \
--resume ./outputs/def-detr-base/sim2city/teaching_mask_dino_const/tch_epoch29.pth