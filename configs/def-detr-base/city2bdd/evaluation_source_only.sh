N_GPUS=8
BATCH_SIZE=8
DATA_ROOT=./dataset
OUTPUT_DIR=./outputs/def-detr-base/city2bdd/teaching_mask_dino_w025_distill_thr04

OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26508 \
--nproc_per_node=${N_GPUS} \
main_v3.py \
--enable_dino \
--dino_weight 0.25 \
--dino_alpha 0.5 \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 9 \
--dropout 0.0 \
--data_root ${DATA_ROOT} \
--source_dataset cityscapes \
--target_dataset bdd100k \
--batch_size ${BATCH_SIZE} \
--eval_batch_size ${BATCH_SIZE} \
--lr 2e-4 \
--lr_backbone 2e-5 \
--lr_linear_proj 2e-5 \
--alpha_ema 0.9996 \
--epoch 2 \
--epoch_lr_drop 80 \
--mode eval \
--threshold 0.4 \
--dynamic_update \
--max_update_iter 5 \
--use_pseudo_label_weights \
--output_dir ${OUTPUT_DIR} \
--resume /gpfsdata/home/caizhi/DRU/output/model_best.pth

# --only_class_loss \
