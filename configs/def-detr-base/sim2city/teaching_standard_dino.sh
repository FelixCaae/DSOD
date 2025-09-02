N_GPUS=8
BATCH_SIZE=8
DATA_ROOT=./dataset
OUTPUT_DIR=./outputs/def-detr-base/sim2city/teaching_standard
OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26504 \
--nproc_per_node=${N_GPUS} \
main.py \
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
--epoch 10 \
--epoch_lr_drop 80 \
--mode teaching_standard \
--threshold 0.3 \
--fix_update_iter 1 \
--output_dir ${OUTPUT_DIR} \
--resume ./sim2city_source_only_48_90.pth