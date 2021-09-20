"""
Author: @dapeng
File_name: vgg16_v3.py
Description: 官方写法，改写了网络结构，
             加入初始化权重代码以及加载预训练权重代码
"""
import torch
import torchvision
from torch import nn
from torch.hub import load_state_dict_from_url


class VGG(nn.Module):
    def __init__(self, features, num_class=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
        # 默认不加载预训练权重，对权重初始化
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            # isinstance判断两个类型是否相同
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 只需记录卷积通道的输出通道维数即可
cfgs = {'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}

vgg_weights_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


def make_layers(cfg):
    layers = []
    input_channel = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels=input_channel, out_channels=v, kernel_size=3, stride=1, padding=1)]
            layers += [nn.ReLU()]
            input_channel = v
    return nn.Sequential(*layers)


# 传入不同参数，实例化不同的VGG模型,返回的模型是加载权重之后的模型
# 1.模型结构的解析
# 2.加载权重还是初始化权重
# 3.分类类别是多少
def _vgg(cfg, weight_url, pretrained, progress, **kwargs):
    # 此处顺序很重要，先设定模型的属性为Fales，再实例化模型，再加载预训练权重
    if pretrained:
        kwargs['init_weights'] = False
    features = make_layers(cfg)
    model = VGG(features, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(weight_url, progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg_16(pretrained=False, progress=True):
    return _vgg(cfgs['A'], vgg_weights_urls['vgg16'], pretrained=pretrained, progress=progress)


def vgg_19(pretrained=False, progress=True):
    return _vgg(cfgs['A'], vgg_weights_urls['vgg16'], pretrained=pretrained, progress=progress)


if __name__ == '__main__':
    # 导入torchsummary查看网络结构，没有安装的话用此命令pip install torchsummary
    from torchsummary import summary
    model = vgg_16()
    summary(model, (3, 224, 224), device='cpu')
    print(model)
    # 以下是官方模型输出，建议进行比较一下官方输出
    official_model = torchvision.models.vgg16()
    summary(official_model, (3, 224, 224), device='cpu')
    print(official_model)

