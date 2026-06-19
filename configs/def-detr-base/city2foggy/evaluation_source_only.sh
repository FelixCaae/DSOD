BATCH_SIZE=1
DATA_ROOT=./dataset
OUTPUT_DIR=./outputs/def-detr-base/city2foggy/source_only/evaluation

CUDA_VISIBLE_DEVICES=1 python -u main.py \
--fuse_type add \
--enable_dino \
--dino_weight 0.4 \
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
--resume  ./city2foggy/teaching_standard_dino_new_w04_feat_align_gaussian_fix/model_best.pth