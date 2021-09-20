"""
Author: @dapeng
File_name: vgg16_v1.py
Description: 最基础写法，适合刚开始练手
"""
import torchvision
from torch import nn
from torch.nn import MaxPool2d


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # 定义搭建网络的模块
        # 送入Conv2d的必须是四维tensor,[batch, channel, width, height],此处的-1表示自行计算batch大小
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # -1*3*224*224 -> -1*64*224*224
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # -1*64*224*224 -> -1*64*224*224
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pooling_1 = MaxPool2d(kernel_size=2, stride=2)  # -1*64*224*224 -> -1*64*112*112

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # -1*64*112*112 -> -1*128*112*112
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # -1*128*112*112 -> -1*128*112*112
        self.relu4 = nn.ReLU(inplace=True)
        self.max_pooling_2 = MaxPool2d(kernel_size=2, stride=2)  # -1*128*112*112 -> -1*128*56*56

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # -1*128*56*56 -> -1*256*56*56
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # -1*256*56*56 -> -1*256*56*56
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # -1*256*56*56 -> -1*256*56*56
        self.relu7 = nn.ReLU(inplace=True)
        self.max_pooling_3 = MaxPool2d(kernel_size=2, stride=2)  # -1*256*56*56 -> -1*256*28*28

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # -1*256*28*28 -> -1*512*28*28
        self.relu8 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # -1*512*28*28 -> -1*512*28*28
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # -1*512*28*28 -> -1*512*28*28
        self.relu10 = nn.ReLU(inplace=True)
        self.max_pooling_4 = MaxPool2d(kernel_size=2, stride=2)  # -1*512*28*28 -> -1*512*14*14

        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # -1*512*14*14 -> -1*512*14*14
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # -1*512*14*14 -> -1*512*14*14
        self.relu12 = nn.ReLU(inplace=True)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # -1*512*14*14 -> -1*512*14*14
        self.relu13 = nn.ReLU(inplace=True)
        self.max_pooling_5 = MaxPool2d(kernel_size=2, stride=2)  # -1*512*14*14 -> -1*512*7*7

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))  # -1*512*7*7   -> -1*512*7*7
        # 此处要注意！在全联接层，需要将数据进行展平操作，利用torch.view()操作
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)  # -1*25088    -> -1*4096
        self.relu14 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)  # -1*4096     -> -1*4096
        self.relu15 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.fc3 = nn.Linear(4096, 1000)  # -1*4096     -> -1*1000

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pooling_1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.max_pooling_2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.max_pooling_3(x)

        x = self.conv8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.max_pooling_4(x)

        x = self.conv11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)
        x = self.conv13(x)
        x = self.relu13(x)
        x = self.max_pooling_5(x)

        x = self.avg_pool(x)

        x = x.view(-1, 512 * 7 * 7)
        x = self.fc1(x)
        x = self.relu14(x)
        x = self.fc2(x)
        x = self.relu15(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # 导入torchsummary查看网络结构，没有安装的话用此命令pip install torchsummary
    from torchsummary import summary
    model = VGG16()
    summary(model, (3, 224, 224))
    print(model)
    # 以下是官方模型输出，建议进行比较一下官方输出
    official_model = torchvision.models.vgg16()
    summary(official_model, (3, 224, 224))
    print(official_model)
