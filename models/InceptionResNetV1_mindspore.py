import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, load_checkpoint, load_param_into_net
import numpy as np

class BasicConv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=4):
        super(BasicConv2d, self).__init__()

        if padding != 4:
            padding = 4

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, pad_mode='pad', padding=padding, has_bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def construct(self, x):
        print(f'Input shape: {x.shape}, type: {x.dtype}')
        x = self.conv(x)
        print(f'After conv layer shape: {x.shape}, type: {x.dtype}')
        x = self.bn(x)
        print(f'After bn layer shape: {x.shape}, type: {x.dtype}')
        x = self.relu(x)
        return x

class Block35(nn.Cell):
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1)
        self.branch1 = nn.SequentialCell([
            BasicConv2d(320, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        ])
        self.branch2 = nn.SequentialCell([
            BasicConv2d(320, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1)
        ])
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=True)
        self.relu = nn.ReLU()

        self.resize = nn.ResizeBilinear()

    def construct(self, x):
        target_size = (58, 58)

        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)

        branch0 = self.resize(branch0, target_size)
        branch1 = self.resize(branch1, target_size)
        branch2 = self.resize(branch2, target_size)
                   
        mixed = np.concatenate([branch0, branch1, branch2], axis=1)
        up = self.conv2d(mixed)
        x += self.scale * up
        x = self.relu(x)
        return x

class Block17(nn.Cell):
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch_0 = BasicConv2d(192, 128, kernel_size=1)
        self.branch_1 = nn.SequentialCell([
            BasicConv2d(192, 128, kernel_size=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), padding=(3, 0))
        ])
        self.conv2d = BasicConv2d(384, 192, kernel_size=1)
        self.relu = ops.ReLU()

    def construct(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        mixed = np.concatenate([branch0, branch1], axis=1)
        up = self.conv2d(mixed)
        x += self.scale * up
        x = self.relu(x)
        return x

class Block8(nn.Cell):
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1)
        self.branch1 = nn.SequentialCell([
            BasicConv2d(2080, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), padding=(1, 0))
        ])
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=True)
        if not self.noReLU:
            self.relu = nn.ReLU()

    def construct(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        mixed = np.concatenate([branch0, branch1], axis=1)
        up = self.conv2d(mixed)
        x += self.scale * up
        if not self.noReLU:
            x = self.relu(x)
        return x

class Mixed_6a(nn.Cell):
    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2, padding=0)
        self.branch1 = nn.SequentialCell([
            BasicConv2d(320, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2, padding=0)
        ])
        self.branch2 = nn.MaxPool2d(3, stride=2, pad_mode='valid')

    def construct(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        x = np.concatenate([branch0, branch1, branch2], axis=1)
        return x

class Mixed_7a(nn.Cell):
    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.SequentialCell([
            BasicConv2d(1088, 256, kernel_size=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2, padding=0)
        ])
        self.branch1 = nn.SequentialCell([
            BasicConv2d(1088, 256, kernel_size=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2, padding=0)
        ])
        self.branch2 = nn.SequentialCell([
            BasicConv2d(1088, 256, kernel_size=1),
            BasicConv2d(256, 288, kernel_size=3, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2, padding=0)
        ])
        self.branch3 = nn.MaxPool2d(3, stride=2, pad_mode='valid')

    def construct(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        x = np.concatenate([branch0, branch1, branch2, branch3], axis=1)
        return x

class Mixed_5b(nn.Cell):
    def __init__(self):
        super(Mixed_5b, self).__init__()
        
        self.branch1x1 = nn.SequentialCell([
            BasicConv2d(192, 96, kernel_size=1)
        ])

        self.branch5x5 = nn.SequentialCell([
            BasicConv2d(192, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5)
        ])

        self.branch3x3dbl = nn.SequentialCell([
            BasicConv2d(192, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3),
            BasicConv2d(96, 96, kernel_size=3)
        ])

        self.branch_pool = nn.SequentialCell([
            nn.MaxPool2d(3, stride=1, pad_mode='same'),
            BasicConv2d(192, 64, kernel_size=1)
        ])

        self.resize = nn.ResizeBilinear()

    def construct(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3dbl = self.branch3x3dbl(x)
        branch_pool = self.branch_pool(x)

        print(f'Branch1x1 shape: {branch1x1.shape}')
        print(f'Branch5x5 shape: {branch5x5.shape}')
        print(f'Branch3x3dbl shape: {branch3x3dbl.shape}')
        print(f'Branch_pool shape: {branch_pool.shape}')

        # Resize all branches to the size of the largest branch (50, 50)
        target_size = (38, 38)
        
        branch1x1 = self.resize(branch1x1, target_size)
        branch5x5 = self.resize(branch5x5, target_size)
        branch3x3dbl = self.resize(branch3x3dbl, target_size)
        branch_pool = self.resize(branch_pool, target_size)
                                  
        print(f'Resized Branch1x1 shape: {branch1x1.shape}')
        print(f'Resized Branch5x5 shape: {branch5x5.shape}')
        print(f'Resized Branch3x3dbl shape: {branch3x3dbl.shape}')
        print(f'Resized Branch_pool shape: {branch_pool.shape}')
        outputs = ops.Concat(axis=1)([branch1x1, branch5x5, branch3x3dbl, branch_pool])
        return outputs

class InceptionResnetV1(nn.Cell):
    def __init__(self, num_classes=1001, dropout_keep_prob=0.8, include_top=True):
        super(InceptionResnetV1, self).__init__()
        self.include_top = include_top
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2) #0
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3) #1
        self.maxpool_3a = nn.MaxPool2d(3, stride=2, pad_mode='valid')
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2, pad_mode='valid')
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.SequentialCell([Block35(scale=0.17) for _ in range(10)])
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.SequentialCell([Block17(scale=0.10) for _ in range(20)])
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.SequentialCell([Block8(scale=0.2) for _ in range(9)])
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AvgPool2d(8, stride=1)
        self.dropout = nn.Dropout(keep_prob=dropout_keep_prob)
        self.last_linear = nn.Dense(2080, num_classes)

    def construct(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        if self.include_top:
            x = self.dropout(x)
            x = self.last_linear(x.view(x.size(0), -1))
        return x

def load_facenet_model(model_path):
    model = InceptionResnetV1()
    param_dict = load_checkpoint(model_path)
    if not param_dict:
        raise ValueError("The loaded parameter dict is empty. Please check the checkpoint file.")
    load_param_into_net(model, param_dict)
    print("Facenet model is loaded")
    return model

