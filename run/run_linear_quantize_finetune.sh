export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore finetune.py     \
 -a lsqmobilenetv2                 \
 -c checkpoints/refine/cifar100_qmobilenetv2_lr001e30_ratio0356fratio0431rs0      \
 --data_name cifar100            \
 --data data/cifar100/           \
 --epochs 150                     \
 --lr 0.01                        \
 --wd 1e-4                        \
 --train_batch 64               \
 --workers 32                    \
 --pretrained                    \
 --obs 2000                       \
 --bits 5                         \
 --linear_quantization           \
# --eval                         \