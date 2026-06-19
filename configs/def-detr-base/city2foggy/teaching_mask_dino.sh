N_GPUS=8
BATCH_SIZE=8
DATA_ROOT=./dataset
OUTPUT_DIR=./outputs/def-detr-base/city2foggy/teaching_mask_dino_new_smooth_feature_alignment

OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26503 \
--nproc_per_node=${N_GPUS} \
main.py \
--enable_dino \
--enable_smooth  \
--dino_weight 0.4 \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 9 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset cityscapes \
--target_dataset foggy_cityscapes \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--alpha_ema 0.999 \
--epoch 30 \
--epoch_lr_drop 80 \
--enable_feature_alignment \
--mode teaching_mask \
--threshold 0.3 \
--dynamic_update \
--max_update_iter 5 \
--only_class_loss \
--use_pseudo_label_weights \
--output_dir ${OUTPUT_DIR} \
--resume ./city2foggy_source_only_29_53.pth