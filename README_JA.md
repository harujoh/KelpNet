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


## 特徴
- PyTorch や Keras と同じ Define by Run を採用しています
- 行列演算にライブラリを使用していないため、全てのアルゴリズムが可読になっています
- 並列演算にOpenCLを採用しているため、GPUだけでなくCPUやFPGA等の様々な演算装置で処理を並列化できます
> ※OpenCLを使用するためには対応するドライバの追加インストールが必要になることがあります
> - Intel製 CPU or GPU: https://software.intel.com/en-us/articles/opencl-drivers
> - AMD製 CPU or GPU: http://www.amd.com/ja-jp/solutions/professional/hpc/opencl
> - Nvidia製 GPU: https://developer.nvidia.com/opencl

### C#で作られているメリット
- 開発環境の構築がカンタンで、プログラミングの初学者にも学びやすい言語です
- .Net標準のFormやUnity等、処理結果を視覚的に表現するための選択肢が豊富です
- PCやモバイル端末、組み込み機器等、様々なプラットフォームに向けた開発ができます

## 連絡方法
ご質問、ご要望は Issues へご登録をお願いします  
細やかなことでも構いませんので、何かお気づきの点が御座いましたら、お気軽にご利用ください  

手軽なやり取りをご希望の場合は Twitter からご連絡ください  
現在の開発状況なども Twitter でご確認いただけます  
Twitter: https://twitter.com/harujoh

## 動作環境
Libraries: .NET Standard 2.0 or 2.1  
Samples: .NET Framework 4.6.1  

## 実装済み関数
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
