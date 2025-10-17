BATCH_SIZE=1
DATA_ROOT=./dataset
OUTPUT_DIR=./outputs/def-detr-base/city2foggy/teaching_standard/evaluation

 python -u main.py \
--resume ./outputs/def-detr-base/city2foggy/teaching_mask_dino_nonlinear_k15/model_last_tch.pth \
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
--enable_dino 

# % --resume ./city2foggy_source_only_29_53.pth
# % --enable_dino \
