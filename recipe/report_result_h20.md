# 训练评估总表

## RL训练评估（性能）
| RL训练评估             | 显卡型号 | gpus | step | s/it   |
| ---------------------- | -------- | ---- | ---- | -- | 
| Qwen3-VL-8B-Instruct   | H20      | 8     | 30   | 679.15 |
| Qwen3-VL-8B-Instruct   | H20      | 16   | 330   | 91.98 |

## RL训练评估（准确率）
| RL训练评估                 | 显卡型号 | 训练方式     | RL Step | MMMU-Pro准确率 |
| -------------------------- | -------- | ------------ | ------- | -------------- |
| Qwen3-VL-8B-Instruct       | H20      | 原始预训练模型 | -       | 0.5902         |
| Qwen3-VL-8B-Instruct 单机  | H20      | 单机RL       | 60      | 0.6249         |
| Qwen3-VL-8B-Instruct 多机  | H20      | 多机RL       | 120     | 0.6231         |

## LLM训练评估（性能）
| LLM训练评估    | 显卡型号   | gpus | step |time（ms） |  TFLOP/s/GPU | TFLOPS | MFU  | lm loss value | lm loss PPL |
| ------------- | -------- | ---- | ---- | ------ | ---------- | ----------- | ------ | ---- | ------------- | 
| Qwen3-8B      | H20      | 8    | 1000  |2494.6     | 1641.95       | 148    | 0.54 | 5.017401E+00  | 1.510183E+02 |
| Qwen3-8B      | H20      | 16   | 1000  |1415.9     | 1446.43      | 148    | 0.54 | 5.017763E+00  | 1.510729E+02 |

## MLM训练评估（性能）

| MLM训练评估                  | 显卡型号    | gpus | step | time（ms） | TFLOP/s/GPU | TFLOPS | MFU  | lm loss value | lm loss PPL |
| ---------------------------- | -------- | ---- | ---- | ---------- | ----------- | ------ | ---- | ------------- | ----------- |
| Qwen3-VL-30B-A3B-Instruct | H20         | 8    | 500  |5712.2     | 10.4        | 148    | 0.07 | 4.343679E-06  | 1.000004E+00 |
| Qwen3-VL-30B-A3B-Instruct | H20         | 16   | 500  |2925.6     | 10.2        | 148    | 0.07 | 4.313877E-06  | 1.000004E+00 |

## 示例：单机8卡 Qwen3-8B 训练 loss 曲线拟合（其他显卡与H20的训练曲线拟合）（需要对于每一个实验添加loss曲线的拟合过程）

![image-20260121111821356](https://raw.githubusercontent.com/daneren/picgo_photo/main/github/202601211118446.png)

## s/it、time（ms）、TFLOP/s/GPU、MFU、lm loss value、lm loss PPL数据说明（这部分仅为说明数据）

RL训练评估中选择第30个step的值来进行数据统计

![image-20260121110136991](https://raw.githubusercontent.com/daneren/picgo_photo/main/github/202601211101374.png)

LLM训练评估中选择第1000个step的值和训练最后的validation评估来进行数据统计

![image-20260121110435660](https://raw.githubusercontent.com/daneren/picgo_photo/main/github/202601211104728.png)

![image-20260121110509674](https://raw.githubusercontent.com/daneren/picgo_photo/main/github/202601211105789.png)

MLM训练评估中选择第500个step的值和训练最后的validation评估来进行数据统计

![image-20260121110705095](https://raw.githubusercontent.com/daneren/picgo_photo/main/github/202601211107195.png)
