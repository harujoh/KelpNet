# KelpNet : Pure C# machine learning framework
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Build status](https://ci.appveyor.com/api/projects/status/a51hnuaat3ldsdmo?svg=true)](https://ci.appveyor.com/project/harujoh/kelpnet) [![codecov](https://codecov.io/gh/harujoh/KelpNet/branch/master/graph/badge.svg)](https://codecov.io/gh/harujoh/KelpNet)

```csharp
/* SampleCode */
FunctionStack<float> nn = new FunctionStack<float>(
    new Convolution2D<float>(1, 32, 5, pad: 2, name: "l1 Conv2D"),
    new ReLU<float>(name: "l1 ReLU"),
    new MaxPooling<float>(2, 2, name: "l1 MaxPooling"),
    new Convolution2D<float>(32, 64, 5, pad: 2, name: "l2 Conv2D"),
    new ReLU<float>(name: "l2 ReLU"),
    new MaxPooling<float>(2, 2, name: "l2 MaxPooling"),
    new Linear<float>(7 * 7 * 64, 1024, name: "l3 Linear"),
    new ReLU<float>(name: "l3 ReLU"),
    new Dropout<float>(name: "l3 DropOut"),
    new Linear<float>(1024, 10, name: "l4 Linear")
);
```

- Samples:
・[**XOR**](https://github.com/harujoh/KelpNet/blob/master/KelpNet.Sample/Samples/Sample1.cs)
・[**CNN**](https://github.com/harujoh/KelpNet/blob/master/KelpNet.Sample/Samples/Sample5.cs)
・[**AlexNet**](https://github.com/harujoh/KelpNet/blob/master/KelpNet.Sample/Samples/Sample19.cs)
・[**VGG**](https://github.com/harujoh/KelpNet/blob/master/KelpNet.Sample/Samples/Sample15.cs)
・[**ResNet**](https://github.com/harujoh/KelpNet/blob/master/KelpNet.Sample/Samples/Sample17.cs)
・[**Others**](https://github.com/harujoh/KelpNet/tree/master/KelpNet.Sample)
- SampleData:
・MNIST
・FashionMNIST
・CIFAR 10/100
- Importable:
・CaffeModel
・ChainerModel
・ONNXModel


## Features
- No libraries are used for matrix operations, so all algorithms are readable.
- Uses the same "Define by Run" approach as PyTorch and Keras.
- OpenCL is used for parallel processing, so processing can be parallelized not only on GPUs, but also on CPUs, FPGAs, and various other computing devices.
> * Additional installation of the corresponding driver may be required to use OpenCL.
> - Intel CPU or GPU: https://software.intel.com/en-us/articles/opencl-drivers
> - AMD CPU or GPU: http://www.amd.com/ja-jp/solutions/professional/hpc/opencl
> - Nvidia GPU: https://developer.nvidia.com/opencl

### Advantages of being built in C#.
- Easy to set up a development environment and easy to learn for beginners in programming.
- There are many options for visual representation of processing results, such as the .Net standard Form and Unity.
- Development for various platforms such as PCs, mobile devices, and embedded devices is possible.

## How to contact us
If you have any questions or concerns, even minor ones, please feel free to use Issue. 

If you want to communicate with us easily, please contact us via Twitter.  
You can also check the current development status on Twitter.  
Twitter: https://twitter.com/harujoh

## System Requirements
Libraries: .NET Standard 2.0 or 2.1  
Samples: .NET Framework 4.6.1  

## mplemented Functions
- Connections:
　・Convolution2D
　・Deconvolution2D
　・EmbedID
　・Linear
　・LSTM
- Activations:
　・ELU
　・LeakyReLU
　・ReLU
　・ReLU6
　・Sigmoid
　・Tanh
　・Softmax
　・Softplus
　・Swish
　・Mish
- Poolings:
　・AveragePooling2D
　・MaxPooling2D
- Normalize:
　・BatchNormalization
　・LRN
- Noise:
　・Dropout
　・StochasticDepth
- LossFunctions:
　・MeanSquaredError
　・SoftmaxCrossEntropy
- Optimizers:
　・AdaBound
　・AdaDelta
　・AdaGrad
　・Adam
　・AdamW
　・AMSBound
　・AMSGrad 
　・MomentumSGD
　・RMSprop
　・SGD
