"""Model definitions."""

import torch
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck
from efficientnet_pytorch import EfficientNet
from collections import OrderedDict
from .mobilenet import MobileNetV2
from .vgg import VGG
from ..data.diff_data_augmentation import Normalizer


class TransformNet(torch.nn.Module):
    def __init__(self, dnn, dnn_name, dataset_name):
        super(TransformNet, self).__init__()

        self.dnn_name = dnn_name
        # self.resize = torch.nn.UpsamplingBilinear2d(size=(224, 224))
        self.normalizer = Normalizer(dataset_name)
        ft, classifier = self._split(dnn)

        self.dnn = torch.nn.Sequential(OrderedDict([
            ('transform', torch.nn.Sequential(self.normalizer)),  # torch.nn.Sequential(self.resize, self.normalizer)),
            ('features', torch.nn.Sequential(*ft)),
            ('classifier', torch.nn.Sequential(*classifier))
        ]))

    def forward(self, x):
        z = self.dnn(x)
        return z

    def _split(self, dnn):
        layer_cake = list(dnn.children())
        last_layer = torch.nn.Sequential(*layer_cake[-1:])
        head_model = torch.nn.Sequential(*layer_cake[:-1])

        classifier = [last_layer, torch.nn.Flatten(1)]
        ft = [head_model] if 'squeezenet' in self.dnn_name else [head_model, torch.nn.Flatten(1)]
        return ft, classifier


def change_last_layer(model, model_name, dataset_name):
    num_classes = {'CIFAR10': 10, 'MNIST': 10, 'imagenet1k': 1000, 'TinyImageNet': 200, 'GTSRB': 43, 'Celeb': 8}

    if 'resnet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes[dataset_name], bias=True)
    elif 'VGG' in model_name:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes[dataset_name])
    elif 'squeezenet' in model_name:
        model.classifier[1] = torch.nn.Conv2d(in_channels=512, out_channels=num_classes[dataset_name],
                                              kernel_size=(1, 1), stride=(1, 1))
    else:
        raise ValueError('Model unknown. Cannot change last layer.')
    return model


def get_model(model_name, dataset_name, pretrained=False):
    """Retrieve an appropriate architecture."""
    if 'CIFAR' in dataset_name or 'MNIST' in dataset_name:
        if pretrained:
            # raise ValueError('Loading pretrained models is only supported for ImageNet.')
            try:
                if 'efficientnet-b' in model_name.lower():
                    model = EfficientNet.from_pretrained(model_name.lower())
                elif model_name == 'squeezenet':
                    model = torchvision.models.squeezenet1_1(pretrained=pretrained)
                else:
                    model = getattr(torchvision.models, model_name.lower())(pretrained=pretrained)
            except AttributeError:
                raise NotImplementedError(f'ImageNet model {model_name} not found at torchvision.models.')
            model = change_last_layer(model, model_name, dataset_name)
        else:
            in_channels = 1 if dataset_name == 'MNIST' else 3
            num_classes = 10 if dataset_name in ['CIFAR10', 'MNIST'] else 100
            if 'resnet' in model_name.lower():
                model = resnet_picker(model_name, dataset_name)
            elif model_name.lower() == 'squeezenet':
                model = torchvision.models.squeezenet1_1()
            elif model_name.lower() == 'Linear':
                model = linear_model(dataset_name, num_classes=num_classes)
            elif 'efficientnet-b' in model_name.lower():
                model = EfficientNet.from_name(model_name.lower())
            elif 'VGG' in model_name:
                model = VGG(model_name)
            elif model_name == 'MobileNetV2':
                model = MobileNetV2(num_classes=num_classes, train_dp=0, test_dp=0, droplayer=0, bdp=0)
            else:
                raise ValueError(f'Architecture {model_name} not implemented for dataset {dataset_name}.')
    elif 'TinyImageNet' in dataset_name:
        in_channels = 3
        num_classes = 200
        if pretrained:
            try:
                if 'efficientnet-b' in model_name.lower():
                    model = EfficientNet.from_pretrained(model_name.lower())
                elif model_name == 'squeezenet':
                    model = torchvision.models.squeezenet1_1(pretrained=pretrained)
                else:
                    model = getattr(torchvision.models, model_name.lower())(pretrained=pretrained)
            except AttributeError:
                raise NotImplementedError(f'ImageNet model {model_name} not found at torchvision.models.')
            model = change_last_layer(model, model_name, dataset_name)
        else:
            if 'VGG16' in model_name:
                model = VGG('VGG16-TI', in_channels=in_channels, num_classes=num_classes)
            elif 'resnet' in model_name.lower():
                model = resnet_picker(model_name, dataset_name)
            else:
                raise ValueError(f'Model {model_name} not implemented for TinyImageNet')

    elif 'GTSRB' in dataset_name:
        in_channels = 3
        num_classes = 43
        if pretrained:
            try:
                if 'efficientnet-b' in model_name.lower():
                    model = EfficientNet.from_pretrained(model_name.lower())
                elif model_name == 'squeezenet':
                    model = torchvision.models.squeezenet1_1(pretrained=pretrained)
                else:
                    model = getattr(torchvision.models, model_name.lower())(pretrained=pretrained)
            except AttributeError:
                raise NotImplementedError(f'ImageNet model {model_name} not found at torchvision.models.')
            model = change_last_layer(model, model_name, dataset_name)
        else:
            if 'VGG16' in model_name:
                model = VGG(model_name, num_classes=num_classes)
            elif 'resnet' in model_name.lower():
                model = resnet_picker(model_name, dataset_name)
            else:
                raise ValueError(f'Model {model_name} not implemented for GTSRB')
    elif 'Celeb' in dataset_name:
        in_channels = 3
        num_classes = 8
        if 'VGG16' in model_name:
            model = VGG('VGG16-TI', in_channels=in_channels, num_classes=num_classes)
        elif 'resnet' in model_name.lower():
            model = resnet_picker(model_name, dataset_name)
        else:
            raise ValueError(f'Model {model_name} not implemented for Celeb')
    elif 'ImageNet' in dataset_name:
        in_channels = 3
        num_classes = 1000
        if model_name == 'Linear':
            model = linear_model(dataset_name, num_classes=num_classes)
        else:
            if 'densenet' in model_name.lower():
                extra_args = dict(
                    memory_efficient=False)  # memory_efficient->checkpointing -> incompatible with autograd.grad
            else:
                extra_args = dict()

            try:
                model = getattr(torchvision.models, model_name.lower())(pretrained=pretrained, **extra_args)
            except AttributeError:
                raise NotImplementedError(f'ImageNet model {model_name} not found at torchvision.models.')
    model = TransformNet(model, model_name, dataset_name)
    return model


def linear_model(dataset, num_classes=10):
    """Define the simplest linear model."""
    if 'cifar' in dataset.lower():
        dimension = 3072
    elif 'mnist' in dataset.lower():
        dimension = 784
    elif 'imagenet' in dataset.lower():
        dimension = 150528
    elif 'tinyimagenet' in dataset.lower():
        dimension = 64 ** 2 * 3
    else:
        raise ValueError('Linear model not defined for dataset.')
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(dimension, num_classes))


def resnet_picker(arch, dataset):
    """Pick an appropriate resnet architecture for MNIST/CIFAR."""
    in_channels = 1 if dataset == 'MNIST' else 3
    num_classes = 10
    if dataset in ['CIFAR10', 'MNIST']:
        num_classes = 10
        initial_conv = [3, 1, 1]
    elif dataset in ['GTSRB']:
        num_classes = 43
        initial_conv = [3, 1, 1]
    elif dataset in ['Celeb']:
        num_classes = 8
        initial_conv = [7, 2, 3]
    elif dataset == 'CIFAR100':
        num_classes = 100
        initial_conv = [3, 1, 1]
    elif dataset == 'TinyImageNet':
        num_classes = 200
        initial_conv = [7, 2, 3]
    else:
        raise ValueError(f'Unknown dataset {dataset} for ResNet.')

    if arch == 'resnet20':
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16,
                      initial_conv=initial_conv)
    elif 'resnet20-' in arch and arch[-1].isdigit():
        width_factor = int(arch[-1])
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes,
                      base_width=16 * width_factor, initial_conv=initial_conv)
    elif arch == 'resnet28-10':
        return ResNet(torchvision.models.resnet.BasicBlock, [4, 4, 4], num_classes=num_classes, base_width=16 * 10,
                      initial_conv=initial_conv)
    elif arch == 'resnet32':
        return ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16,
                      initial_conv=initial_conv)
    elif arch == 'resnet32-10':
        return ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16 * 10,
                      initial_conv=initial_conv)
    elif arch == 'resnet44':
        return ResNet(torchvision.models.resnet.BasicBlock, [7, 7, 7], num_classes=num_classes, base_width=16,
                      initial_conv=initial_conv)
    elif arch == 'resnet56':
        return ResNet(torchvision.models.resnet.BasicBlock, [9, 9, 9], num_classes=num_classes, base_width=16,
                      initial_conv=initial_conv)
    elif arch == 'resnet110':
        return ResNet(torchvision.models.resnet.BasicBlock, [18, 18, 18], num_classes=num_classes, base_width=16,
                      initial_conv=initial_conv)
    elif arch == 'resnet18':
        return ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=64,
                      initial_conv=initial_conv)
    elif 'resnet18-' in arch:  # this breaks the usual notation, but is nicer for now!!
        new_width = int(arch.split('-')[1])
        return ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=new_width,
                      initial_conv=initial_conv)
    elif arch == 'resnet34':
        return ResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes, base_width=64,
                      initial_conv=initial_conv)
    elif arch == 'resnet50':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes, base_width=64,
                      initial_conv=initial_conv)
    elif arch == 'resnet101':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], num_classes=num_classes, base_width=64,
                      initial_conv=initial_conv)
    elif arch == 'resnet152':
        return ResNet(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], num_classes=num_classes, base_width=64,
                      initial_conv=initial_conv)
    else:
        raise ValueError(f'Invalid ResNet [{dataset}] model chosen: {arch}.')


class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR-like thingies.

    This is a minor modification of
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py,
    adding additional options.
    """

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=[False, False, False, False],
                 norm_layer=torch.nn.BatchNorm2d, strides=[1, 2, 2, 2], initial_conv=[3, 1, 1]):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # torch.nn.Module
        self._norm_layer = norm_layer

        self.dilation = 1
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=initial_conv[0],
                                     stride=initial_conv[1], padding=initial_conv[2], bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)

        layer_list = []
        width = self.inplanes
        for idx, layer in enumerate(layers):
            layer_list.append(
                self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2
        self.layers = torch.nn.Sequential(*layer_list)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the arch by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
