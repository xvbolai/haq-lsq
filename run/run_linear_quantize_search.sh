export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore rl_quantize.py     \
 --arch lsqmobilenetv2                \
 --dataset cifar100              \
 --dataset_root data/cifar100    \
 --suffix ratio0356bit26fratio0431rs0            \
 --preserve_ratio 0.356             \
 --flash_preserve_ratio 0.431        \
 --float_bit 8                      \
 --max_bit 6                        \
 --min_bit 2                        \
 --n_worker 32                      \
 --obs 0                         \
 --bits 5                            \
 --data_bsize 64                   \
 --train_size 20000                 \
 --val_size 10000                   \
 --reward_strategy 0                \
 --linear_quantization_plus         \
#   --linear_quantization              \
