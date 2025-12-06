BATCH_SIZE=1
DATA_ROOT=./dataset
OUTPUT_DIR=./outputs/def-detr-base/city2foggy/source_only/evaluation

CUDA_VISIBLE_DEVICES=1 python -u main.py \
--enable_dino \
--fuse_type gate_add \
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
--resume ./city2foggy/teaching_standard_dino_gate_add_04_with_sup/model_best.pth