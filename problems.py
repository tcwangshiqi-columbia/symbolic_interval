import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import argparse
from convex_adversarial import epsilon_from_model, DualNetBounds
from convex_adversarial import Dense, DenseSequential
import math
import os

def model_wide(in_ch, out_width, k): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*k, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*k, 8*k, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*k*out_width*out_width,k*128),
        nn.ReLU(),
        nn.Linear(k*128, 10)
    )
    return model

def model_deep(in_ch, out_width, k, n1=8, n2=16, linear_size=100): 
    def group(inf, outf, N): 
        if N == 1: 
            conv = [nn.Conv2d(inf, outf, 4, stride=2, padding=1), 
                         nn.ReLU()]
        else: 
            conv = [nn.Conv2d(inf, outf, 3, stride=1, padding=1), 
                         nn.ReLU()]
            for _ in range(1,N-1):
                conv.append(nn.Conv2d(outf, outf, 3, stride=1, padding=1))
                conv.append(nn.ReLU())
            conv.append(nn.Conv2d(outf, outf, 4, stride=2, padding=1))
            conv.append(nn.ReLU())
        return conv

    conv1 = group(in_ch, n1, k)
    conv2 = group(n1, n2, k)

    '''
    model = nn.Sequential(
        *conv1, 
        *conv2,
        Flatten(),
        nn.Linear(n2*out_width*out_width,linear_size),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    '''
    model = nn.Sequential(
        *conv1
    )
    return model

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mnist_loaders(batch_size, shuffle_test=False): 
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

def fashion_mnist_loaders(batch_size): 
    mnist_train = datasets.MNIST("./fashion_mnist", train=True,
       download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./fashion_mnist", train=False,
       download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def mnist_model_wide(k): 
    return model_wide(1, 7, k)

def mnist_model_deep(k): 
    return model_deep(1, 7, k)

def mnist_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model

def replace_10_with_0(y): 
    return y % 10

def svhn_loaders(batch_size): 
    train = datasets.SVHN("./data", split='train', download=True, transform=transforms.ToTensor(), target_transform=replace_10_with_0)
    test = datasets.SVHN("./data", split='test', download=True, transform=transforms.ToTensor(), target_transform=replace_10_with_0)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def svhn_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).cuda()
    return model

def har_loaders(batch_size):     
    X_te = torch.from_numpy(np.loadtxt('./data/UCI HAR Dataset/test/X_test.txt')).float()
    X_tr = torch.from_numpy(np.loadtxt('./data/UCI HAR Dataset/train/X_train.txt')).float()
    y_te = torch.from_numpy(np.loadtxt('./data/UCI HAR Dataset/test/y_test.txt')-1).long()
    y_tr = torch.from_numpy(np.loadtxt('./data/UCI HAR Dataset/train/y_train.txt')-1).long()

    har_train = td.TensorDataset(X_tr, y_tr)
    har_test = td.TensorDataset(X_te, y_te)

    train_loader = torch.utils.data.DataLoader(har_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(har_test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

def har_500_model(): 
    model = nn.Sequential(
        nn.Linear(561, 500),
        nn.ReLU(),
        nn.Linear(500, 6)
    )
    return model

def har_500_250_model(): 
    model = nn.Sequential(
        nn.Linear(561, 500),
        nn.ReLU(),
        nn.Linear(500, 250),
        nn.ReLU(),
        nn.Linear(250, 6)
    )
    return model

def har_500_250_100_model(): 
    model = nn.Sequential(
        nn.Linear(561, 500),
        nn.ReLU(),
        nn.Linear(500, 250),
        nn.ReLU(),
        nn.Linear(250, 100),
        nn.ReLU(),
        nn.Linear(100, 6)
    )
    return model

def har_resnet_model(): 
    model = DenseSequential(
        Dense(nn.Linear(561, 561)), 
        nn.ReLU(), 
        Dense(nn.Sequential(), None, nn.Linear(561,561)),
        nn.ReLU(), 
        nn.Linear(561,6)
        )
    return model

def cifar_loaders(batch_size, shuffle_test=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./data', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./data', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader


def imagenet_loaders(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train = datasets.ImageFolder('./tiny-imagenet-200/train/',\
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
    '''
    train = datasets.CIFAR10('./data', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./data', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    '''
    test = datasets.ImageFolder('./tiny-imagenet-200/val/',\
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader



def cifar_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

def cifar_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    #return model
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

def cifar_model_resnet(N = 5, factor=10): 
    def  block(in_filters, out_filters, k, downsample): 
        if not downsample: 
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else: 
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)), 
            nn.ReLU(), 
            Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0), 
                  None, 
                  nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)), 
            nn.ReLU()
        ]
    conv1 = [nn.Conv2d(3,16,3,stride=1,padding=1), nn.ReLU()]
    conv2 = block(16,16*factor,3, True)
    
    for _ in range(N): 
        conv2.extend(block(16*factor,16*factor,3, False))
    
    conv3 = block(16*factor,32*factor,3, True)
    for _ in range(N-1): 
        conv3.extend(block(32*factor,32*factor,3, False))
    conv4 = block(32*factor,64*factor,3, True)
    for _ in range(N-1): 
        conv4.extend(block(64*factor,64*factor,3, False))
    layers = (
        conv1 + 
        conv2 + 
        conv3 + 
        conv4 +
        [Flatten(),
        nn.Linear(64*factor*4*4,1000), 
        nn.ReLU(), 
        nn.Linear(1000, 10)]
        )
    model = DenseSequential(
        *layers
    )
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None: 
                m.bias.data.zero_()
    return model



def imagenet_model_resnet(N = 5, factor=10): 
    def  block(in_filters, out_filters, k, downsample): 
        if not downsample: 
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else: 
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)), 
            nn.ReLU(), 
            Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0), 
                  None, 
                  nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)), 
            nn.ReLU()
        ]
    conv1 = [nn.Conv2d(3,16,3,stride=1,padding=1), nn.ReLU()]
    conv2 = block(16,16*factor,3, True)
    
    for _ in range(N): 
        conv2.extend(block(16*factor,16*factor,3, False))
    
    conv3 = block(16*factor,32*factor,3, True)
    for _ in range(N-1): 
        conv3.extend(block(32*factor,32*factor,3, False))
    conv4 = block(32*factor,64*factor,3, True)
    for _ in range(N-1): 
        conv4.extend(block(64*factor,64*factor,3, False))
    layers = (
        conv1 + 
        conv2 + 
        conv3 + 
        conv4 +
        [Flatten(),
        nn.Linear(64*factor*4*4,512), 
        nn.ReLU(), 
        nn.Linear(512, 200)]
        )
    model = DenseSequential(
        *layers
    )
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None: 
                m.bias.data.zero_()
    return model


def imagenet_model_resnet1(N = 5, factor=10): 
    def  block(in_filters, out_filters, k, downsample): 
        if not downsample: 
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else: 
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [
            Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)), 
            nn.ReLU(), 
            Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0), 
                  None, 
                  nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)), 
            nn.ReLU()
        ]
    conv1 = [nn.Conv2d(3,16*factor,7,stride=2,padding=3), nn.ReLU()]
    conv2 = block(16*factor,16*factor,3, True)
    
    for _ in range(N): 
        conv2.extend(block(16*factor,16*factor,3, False))
    
    conv3 = block(16*factor,32*factor,3, True)
    for _ in range(N-1): 
        conv3.extend(block(32*factor,32*factor,3, False))
    conv4 = block(32*factor,64*factor,3, True)
    for _ in range(N-1): 
        conv4.extend(block(64*factor,64*factor,3, False))
    layers = (
        conv1 + 
        conv2 + 
        conv3 + 
        conv4 +
        [Flatten(),
        nn.Linear(64*factor*4*4,512), 
        nn.ReLU(), 
        nn.Linear(512, 200)]
        )
    model = DenseSequential(
        *layers
    )
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None: 
                m.bias.data.zero_()
    return model



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model




def argparser(batch_size=50, epochs=20, seed=0, verbose=10, lr=1e-3, 
              epsilon=0.1, starting_epsilon=None, 
              l1_proj=None, delta=None, m=1, l1_eps=None, 
              l1_train='exact', l1_test='exact', 
              opt='sgd', momentum=0.9, weight_decay=5e-4): 
    # l1_proj = None
    # l1_eps = None

    parser = argparse.ArgumentParser()

    # optimizer settings
    parser.add_argument('--opt', default=opt)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)

    # epsilon settings
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
    parser.add_argument('--schedule_length', type=int, default=10)

    parser.add_argument('--k', type=int, default=1)

    # projection settings
    parser.add_argument('--l1_proj', type=int, default=l1_proj)
    parser.add_argument('--delta', type=float, default=delta)
    parser.add_argument('--m', type=int, default=m)
    parser.add_argument('--l1_train', default=l1_train)
    parser.add_argument('--l1_test', default=l1_test)
    parser.add_argument('--l1_eps', type=float, default=l1_eps)

    # model arguments
    parser.add_argument('--model', default='resnet')
    parser.add_argument('--model_factor', type=int, default=8)
    parser.add_argument('--cascade', type=int, default=1)
    parser.add_argument('--method', default="robust_new")
    parser.add_argument('--resnet_N', type=int, default=1)
    parser.add_argument('--resnet_factor', type=int, default=1)

    # other arguments
    parser.add_argument('--prefix')
    parser.add_argument('--load')
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument('--cuda_ids', default=None)

    parser.add_argument('--interval_weight', type=float, default=0.1)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--portion', type=float, default=0.1)
    parser.add_argument('--t', type=int, default=0)
    parser.add_argument('--evaluate', type=int, default=None)

    parser.add_argument('--parallel', action='store_true', default=False)

    
    args = parser.parse_args()
    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon 
    if args.prefix: 
        if args.model is not None: 
            args.prefix += '_'+args.model

        if args.method is not None: 
            args.prefix += '_'+args.method

        banned = ['verbose', 'prefix',
                  'resume', 'baseline', 'eval', 
                  'method', 'model', 'cuda_ids', 'load']
        if args.method == 'baseline':
            banned += ['epsilon', 'starting_epsilon', 'schedule_length', 
                       'l1_test', 'l1_train', 'm', 'l1_proj']

        # if not using adam, ignore momentum and weight decay
        if args.opt == 'adam': 
            banned += ['momentum', 'weight_decay']

        if args.m == 1: 
            banned += ['m']
        if args.cascade == 1: 
            banned += ['cascade']

        # if not using a model that uses model_factor, 
        # ignore model_factor
        if args.model not in ['wide', 'deep']: 
            banned += ['model_factor']

        if args.model != 'resnet': 
            banned += ['resnet_N', 'resnet_factor']

        for arg in sorted(vars(args)): 
            if arg not in banned and getattr(args,arg) is not None: 
                args.prefix += '_' + arg + '_' +str(getattr(args, arg))

        if args.schedule_length > args.epochs: 
            raise ValueError('Schedule length for epsilon ({}) is greater than '
                             'number of epochs ({})'.format(args.schedule_length, args.epochs))
    else: 
        args.prefix = 'temporary'

    if args.cuda_ids is not None: 
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids


    return args

def args2kwargs(args, X=None): 

    if args.l1_proj is not None: 
        kwargs = {
            'l1_proj' : args.l1_proj
        }
    else:
        kwargs = {
        }

        '''
        if not args.l1_eps:
            if args.delta: 
                args.l1_eps = epsilon_from_model(model, Variable(X.cuda()), args.l1_proj,
                                            args.delta, args.m)
                print(
        With probability {} and projection into {} dimensions and a max
        over {} estimates, we have epsilon={}.format(args.delta, args.l1_proj,
                                                        args.m, args.l1_eps))
            else: 
                args.l1_eps = 0
                print('No epsilon or \delta specified, using epsilon=0.')
        else:
            print('Specified l1_epsilon={}'.format(args.l1_eps))
        kwargs = {
            'l1_proj' : args.l1_proj, 
            # 'l1_eps' : args.l1_eps, 
            # 'm' : args.m
        }
        '''
    return kwargs