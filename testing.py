import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_shufflenetv2_plus():
    print("test shufflenetv2_plus")
    from src.model.backbone.shufflenetv2_plus import shufflenetv2_plus
    model = shufflenetv2_plus()
    num_params = 0.0
    for params in model.parameters():
        num_params += params.numel()
    print("Trainable parameters: {:.2f}M".format(num_params / 1000000.0))
    x = torch.ones(2,3,112,112)
    output = model(x)

def test_hrnet():
    print("test hrnet")
    from src.model.backbone.hrnet import hrnet
    model = hrnet()
    num_params = 0.0
    for params in model.parameters():
        num_params += params.numel()
    print("Trainable parameters: {:.2f}M".format(num_params / 1000000.0))
    x = torch.rand(2, 3, 256, 192)
    output = model(x)

if __name__ == "__main__":
    test_shufflenetv2_plus()
    test_hrnet()