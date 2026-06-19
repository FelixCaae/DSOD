N_GPUS=8
BATCH_SIZE=8
DATA_ROOT=./dataset
OUTPUT_DIR=./outputs/def-detr-base/city2bdd/teaching_mask_dino_w025_distill_thr04

OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26508 \
--nproc_per_node=${N_GPUS} \
main.py \
--dino_weight 0.25 \
--dino_alpha 0.5 \
--enable_dino \
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
--mode eval \
--output_dir ${OUTPUT_DIR} \
--resume ./outputs/def-detr-base/city2bdd/teaching_standard_dino_w04_smooth_sqrt_feat_align_gauss/model_best.pth
# --only_class_loss \
