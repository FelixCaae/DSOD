N_GPUS=8
BATCH_SIZE=8
DATA_ROOT=./dataset
OUTPUT_DIR=./outputs/def-detr-base/sim2city/teaching_mask_dino_with_w025_a05_no_decrease

OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26505 \
--nproc_per_node=${N_GPUS} \
main.py \
--dino_weight 0.25 \
--dino_alpha 0.5 \
--enable_dino \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 4 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset sim10k \
--target_dataset cityscapes \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--alpha_ema 0.999 \
--epoch 35 \
--epoch_lr_drop 80 \
--mode teaching_mask \
--threshold 0.3 \
--dynamic_update \
--max_update_iter 5 \
--only_class_loss \
--use_pseudo_label_weights \
--output_dir ${OUTPUT_DIR} \
--resume ./sim2city_source_only_48_90.pth