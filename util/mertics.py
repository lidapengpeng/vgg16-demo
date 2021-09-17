# 准确率的计算方法
import torch


outputs = torch.tensor([[0.6, 0.7],
                        [0.3, 0.2]])
print(outputs.argmax(1))

targets = torch.tensor([1, 1])
print(outputs.argmax(1) == targets)

print((outputs.argmax(1) == targets).sum())

