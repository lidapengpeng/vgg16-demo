"""
1、 创建数据集
2、 加载数据集
3、 模型实例化
4、 创建损失函数和优化器
5、 设置训练网络
6、 保存模型
7、 可视化参数
"""
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.vgg16 import VGG16

# 1、指定数据集的路径和变换(50000,32,32,3)（10000,32,32,3）
train_data = datasets.CIFAR10(root='data/test', train=True,
                              transform=transforms.ToTensor(), download=False)
test_data = datasets.CIFAR10(root='data/test', train=False,
                             transform=transforms.ToTensor, download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
# 2、加载数据进入网络，指定数据集和大小
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64)

# 3、实例化网络
net_vgg16 = VGG16()

# 4、创建损失函数和优化器
loss_function = CrossEntropyLoss()  # 可以去官网查看该损失函数的输入输出维度
optimizer = optim.Adam(net_vgg16.parameters(), lr=0.001)

# 5、设置训练网络
epochs = 10

for i in range(epochs):
    print("-----------第{}轮训练-----------".format(i+1))
    # 训练阶段开始
    net_vgg16.train()
    for j, data in enumerate(train_dataloader):
        images, labels = data
        predict = net_vgg16(images)
        # 计算该batch的loss
        train_batch_loss = loss_function(predict, labels)
        
        # 对上一次计算的各个节点梯度清零，如果不清零，则优化器保存的是上一次的梯度
        optimizer.zero_grad()        
        # 计算出各个节点的梯度,为优化器提供依据
        train_batch_loss.backward()
        # 利用优化器调整更新各个节点的参数，依据是每个节点的梯度
        optimizer.step()
        if (j+1) % 2 == 0:
            print("训练次数：{},loss为:{}".format(j+1, train_batch_loss.item()))
    # 测试阶段开始
    total_test_loss = 0
    total_test_accuracy = 0
    net_vgg16.eval()
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            predict = net_vgg16(images)
            test_batch_loss = loss_function(predict, labels)
            total_test_loss += test_batch_loss
            total_test_accuracy = (predict.argmax(1) == labels).sum()
    print("整体测试集loss:{}".format(total_test_loss))
    print("整体测试集accuracy:{}".format(total_test_accuracy/test_data_size))

    # 保存模型
    if (i+1) % 2 == 0:
        torch.save(net_vgg16.state_dict(), './weights/train_vgg16_{}'.format(i))



