## Latest Lightweight Neural Network Survey

Due to the needs of the project, I investigated 10 latest lightweight backbone networks,  the parameter amount is about 5M but get nice accuracy on ImageNet2012 validate dataset. They belong to FBNet,  HardNet, MixNet, MnasNet, ShuffleNet and EfficientNet model series.

| Index |      Model      | Max Channel | Params(M) | Top-1  Accuracy(val) |
| :---: | :-------------: | :---------: | :-------: | :------------------: |
|   1   |    Fbnet-cb     |    1984     |  5.5722   |        75.12%        |
|   2   |   hardnet39ds   |    1024     | 3.488228  |        72.08%        |
|   3   |   hardnet68ds   |    1024     | 4.181602  |        74.29%        |
|   4   |    mixnet_s     |    1536     | 4.134606  |        75.99%        |
|   5   |    mixnet_m     |    1536     | 5.014382  |        77.05%        |
|   6   |    mixnet_l     |    1584     | 7.329252  |        78.88%        |
|   7   |   mnasnet_a1    |    1280     | 3.887036  |        75.33%        |
|   8   |   mnasnet_b1    |    1280     | 4.383312  |        74.61%        |
|   9   |  ShuffleNet v2  |    1553     | 4.410194  |        72.69%        |
|  10   | efficientnet_b0 |    1280     | 5.288548  |        75.23%        |

![](/imgs/img1.png)

### Paper Download

- [FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://arxiv.org/abs/1812.03443)

- [HarDNet: A Low Memory Traffic Network](https://arxiv.org/abs/1909.00948)

- [MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/abs/1907.09595)

- [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626)

- [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)

### Prerequisites

- torch
- torchsummary
- ptflops

### Function

- torchsummary, for example

  ```
  ==>  efficientnet_b0
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Conv2d-1         [-1, 32, 112, 112]             864
         BatchNorm2d-2         [-1, 32, 112, 112]              64
               Swish-3         [-1, 32, 112, 112]               0
           ConvBlock-4         [-1, 32, 112, 112]               0
       EffiInitBlock-5         [-1, 32, 112, 112]               0
              Conv2d-6         [-1, 32, 112, 112]             288
         BatchNorm2d-7         [-1, 32, 112, 112]              64
               Swish-8         [-1, 32, 112, 112]               0
           ConvBlock-9         [-1, 32, 112, 112]               0
  AdaptiveAvgPool2d-10             [-1, 32, 1, 1]               0
  ......
    EffiInvResUnit-289            [-1, 320, 7, 7]               0
            Conv2d-290           [-1, 1280, 7, 7]         409,600
       BatchNorm2d-291           [-1, 1280, 7, 7]           2,560
             Swish-292           [-1, 1280, 7, 7]               0
         ConvBlock-293           [-1, 1280, 7, 7]               0
  AdaptiveAvgPool2d-294           [-1, 1280, 1, 1]               0
           Dropout-295                 [-1, 1280]               0
            Linear-296                 [-1, 1000]       1,281,000
  ================================================================
  Total params: 5,288,548
  Trainable params: 5,288,548
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 0.57
  Forward/backward pass size (MB): 226.52
  Params size (MB): 20.17
  Estimated Total Size (MB): 247.26
  ----------------------------------------------------------------
  ```

- flops and params counter, for example

  ```
  init_block_channels: 16 0.001
  FBNet(
    5.572 M, 100.000% Params, 0.404 GMac, 100.000% MACs,
    (features): Sequential(
      3.587 M, 64.377% Params, 0.402 GMac, 99.509% MACs,
      (init_block): FBNetInitBlock(
        0.001 M, 0.022% Params, 0.016 GMac, 3.926% MACs,
        (conv1): ConvBlock(
          0.0 M, 0.008% Params, 0.006 GMac, 1.491% MACs,
          (conv): Conv2d(0.0 M, 0.008% Params, 0.005 GMac, 1.342% MACs, 3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.099% MACs, 16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
          (activ): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.050% MACs, inplace=True)
        )
        (conv2): FBNetUnit(
          0.001 M, 0.013% Params, 0.01 GMac, 2.435% MACs,
          (exp_conv): ConvBlock(
            0.0 M, 0.005% Params, 0.004 GMac, 0.944% MACs,
            (conv): Conv2d(0.0 M, 0.005% Params, 0.003 GMac, 0.795% MACs, 16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.099% MACs, 16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
            (activ): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.050% MACs, inplace=True)
          )
  ......
      (final_block): ConvBlock(
        0.702 M, 12.604% Params, 0.035 GMac, 8.545% MACs,
        (conv): Conv2d(0.698 M, 12.533% Params, 0.034 GMac, 8.473% MACs, 352, 1984, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(0.004 M, 0.071% Params, 0.0 GMac, 0.048% MACs, 1984, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        (activ): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.024% MACs, inplace=True)
      )
      (final_pool): AvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.024% MACs, kernel_size=7, stride=1, padding=0)
    )
    (output): Linear(1.985 M, 35.623% Params, 0.002 GMac, 0.491% MACs, in_features=1984, out_features=1000, bias=True)
  )
  Computational complexity:       0.4 GMac
  Number of parameters:           5.57 M
  ```

  
