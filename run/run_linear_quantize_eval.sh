export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore finetune.py     \
 -a qmobilenetv2                 \
 --resume checkpoints/imagenet_qmobilenetv2_lr010e30_ratio060/checkpoint.pth.tar        \
 --workers 32                    \
 --test_batch 64                \
 --gpu_id 0,1,2,3                \
 --free_high_bit False           \
 --linear_quantization           \
 --eval                          \
