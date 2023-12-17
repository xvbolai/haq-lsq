### haq-lsq的配置

#### the flash ratio and latency ratio

```python

strategy = [[8, -1], [2, 5], [2, 6], [2, 2], [2, 4], [2, 3], [2, 2], [2, 4], [2, 5], [2, 2], [2, 2], [2, 4], [2, 3], 
[2, 3], [2, 6], [2, 2], [2, 2], [2, 5], [2, 4], [2, 2], [2, 3], [2, 5], [2, 3], [2, 4], [2, 4], [2, 3], 
[2, 3], [2, 4], [2, 6], [2, 4], [2, 2], [2, 4], [2, 2], [2, 4], [2, 4], [2, 4], [2, 3], [2, 6], [2, 5], 
[2, 2], [2, 6], [2, 4], [2, 3], [2, 2], [2, 4], [2, 4], [2, 2], [2, 2], 
[2, 2], [2, 3], [2, 4], [2, 3], [8, 8]]
f_weight = [864, 288, 512, 1536, 864, 2304, 3456, 1296, 3456, 3456, 1296, 4608, 6144, 1728, 6144, 6144, 1728, 6144, 6144, 1728, 
12288, 24576, 3456, 24576, 24576, 3456, 24576, 24576, 3456, 24576, 24576, 3456, 36864, 55296, 5184, 55296, 55296, 5184, 
55296, 55296, 5184, 92160, 153600, 8640, 153600, 153600, 8640, 153600, 153600, 8640, 307200, 409600, 1280000]
min_weight = 0.
for i, n_bit in enumerate(strategy[1:-1]):
    min_weight += f_weight[i + 1] * 3.45

min_weight += f_weight[0] * 8 

weight = 0.
for i, n_bit in enumerate(strategy[1:-1]):
    weight += f_weight[i + 1] * 8.0     
         
weight += 8.0 * (f_weight[0])

print(min_weight/ weight)

import numpy as np
def get_lookuptable():
    
    import os
    fname = 'lib/simulator/lookup_tables/lsqmobilenetv2_cifar100_batch16_latency_table.npy'
    if os.path.isfile(fname):
        # print('load latency table : ', fname)
        latency_list = np.load(fname)
        # print(latency_list)
    else:
        # you can put your own simulator/lookuptable here
        raise NotImplementedError
    return latency_list.copy()

def cost(cost_lookuptable, min_bit):
    min_cost = 0
    for i in range(cost_lookuptable.shape[0]):
        if i == 0 or i == (cost_lookuptable.shape[0] - 1):
            min_cost += cost_lookuptable[i, -1, -1]
        else:
            min_cost += cost_lookuptable[i, int(min_bit - 1), int(min_bit - 1)]
    return min_cost

cost_lookuptable = get_lookuptable()

print(np.mean([cost(cost_lookuptable, 3)/cost(cost_lookuptable, 8), cost(cost_lookuptable, 4)/cost(cost_lookuptable, 8)]))

# 介于3和4之间的配置是 latency ratio 0.356 memory ratio 0.431
```

#### 介于3bit和4bit之间的配置

```txt
pretrain model is 5 bits
latency ratio 0.356 
memory ratio 0.431
```

```bash
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
```

#### 低比特数据如何从内存中取出

```c++
#include<string.h>
#include<stdio.h> 

typedef struct {
	char l:4;
	char h:4;
} bits_4;

int main() {
	char a = 0xEF;
	bits_4 b;
	memcpy(&b, &a, 1);
	printf("%d %d\n", b.h, b.l);
	return 0;
}
```

output

```txt
-2 -1
```


#### other
```txt
best_reward:  4.751680051575057
best_policy:  [[8, -1], [6, 6], [6, 6], [5, 6], [6, 6], [6, 5], [5, 6], [6, 5], [5, 6], [6, 4], [4, 3], [6, 6], [5, 4], [6, 6], [5, 6], [6, 6], [6, 6], [5, 6], [6, 6], [5, 5], [6, 5], [5, 6], [5, 4], [6, 6], [4, 6], [6, 2], [4, 4], [4, 5], [5, 4], [4, 5], [3, 5], [5, 5], [3, 5], [5, 5], [5, 4], [4, 5], [5, 5], [5, 5], [4, 5], [5, 5], [3, 5], [5, 4], [4, 5], [5, 3], [5, 2], [4, 4], [5, 5], [5, 3], [5, 5], [4, 3], [5, 4], [2, 5], [8, 8]]
```