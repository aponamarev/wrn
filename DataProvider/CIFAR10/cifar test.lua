if not torch then require 'torch' end
local cifar = torch.load('/Users/aponamaryov/TorchTutorials/cifar10torchsmall/cifar10-train.t7')
print(cifar)