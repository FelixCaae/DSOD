N_GPUS=1
BATCH_SIZE=1
DATA_ROOT=./dataset
OUTPUT_DIR=./city2foggy/teaching_standard_dino_new

 OMP_NUM_THREADS=4 torchrun \
--rdzv_endpoint localhost:26505 \
--nproc_per_node=${N_GPUS} \
main.py \
# --fuse_type se \
# --enable_dino \
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
--epoch 10 \
--epoch_lr_drop 80 \
--mode eval \
--threshold 0.3 \
--fix_update_iter 1 \
--output_dir ${OUTPUT_DIR} \
--resume ./outputs/def-detr-base/city2foggy/teaching_mask_dino_nonlinear_k15/model_last_tch.pth \
# --resume ./city2foggy/teaching_standard_dino_new/model_best.pth
--tag test_dynamic_weight