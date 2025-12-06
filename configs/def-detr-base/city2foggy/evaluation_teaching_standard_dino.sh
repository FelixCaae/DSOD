BATCH_SIZE=1
DATA_ROOT=./dataset
OUTPUT_DIR=./outputs/def-detr-base/city2foggy/teaching_standard/evaluation

CUDA_VISIBLE_DEVICES=1 python -u main.py \
--enable_dino \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 9 \
--data_root ${DATA_ROOT} \
--source_dataset cityscapes \
--target_dataset foggy_cityscapes \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--output_dir ${OUTPUT_DIR} \
--resume ./city2foggy_source_only_29_53.pth