# KelpNet : Pure C# deep learning framework
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Build status](https://ci.appveyor.com/api/projects/status/a51hnuaat3ldsdmo?svg=true)](https://ci.appveyor.com/project/harujoh/kelpnet) [![codecov](https://codecov.io/gh/harujoh/KelpNet/branch/master/graph/badge.svg)](https://codecov.io/gh/harujoh/KelpNet)

```java
/* SampleCode */
FunctionStack nn = new FunctionStack(
    new Convolution2D(1, 32, 5, pad: 2, name: "l1 Conv2D"),
    new ReLU(name: "l1 ReLU"),
    new MaxPooling(2, 2, name: "l1 MaxPooling"),
    new Convolution2D(32, 64, 5, pad: 2, name: "l2 Conv2D"),
    new ReLU(name: "l2 ReLU"),
    new MaxPooling(2, 2, name: "l2 MaxPooling"),
    new Linear(7 * 7 * 64, 1024, name: "l3 Linear"),
    new ReLU(name: "l3 ReLU"),
    new Dropout(name: "l3 DropOut"),
    new Linear(1024, 10, name: "l4 Linear")
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
・CIFAR 10/100
- Importable:
・CaffeModel
・ChainerModel


## 特徴
- 行列演算にライブラリを使用していないため、全てのアルゴリズムが可読になっています
- KerasやChainerに似た、関数を書き連ねるコーディングスタイルを採用しています
- 並列演算にOpenCLを採用しているため、GPUだけでなくCPUやFPGA等の様々な演算装置で処理を並列化できます
> ※OpenCLを使用するためには対応するドライバの追加インストールが必要になることがあります
> - Intel製 CPU or GPU: https://software.intel.com/en-us/articles/opencl-drivers
> - AMD製 CPU or GPU: http://www.amd.com/ja-jp/solutions/professional/hpc/opencl
> - Nvidia製 GPU: https://developer.nvidia.com/opencl

### C#で作られているメリット
- 開発環境の構築がカンタンで、プログラミングの初学者にも学びやすい言語です
- .Net標準のFormやUnity等、処理結果を視覚的に表現するための選択肢が豊富です
- PCや携帯、組み込み機器等、様々なプラットフォームに向けた開発ができます

## このライブラリについて
このライブラリの基幹部分はChainerを参考に実装されています。  
その為、パラメータが似ており、Chainerのサンプルを参考に開発をすることができます。

## 連絡方法
ご質問、ご要望は Issues へご登録頂くか Twitter からご連絡ください。  
細やかなことでも構いませんので、何かお気づきの点が御座いましたら、お気軽にご連絡ください。

Twitter: https://twitter.com/harujoh



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
　・Sigmoid
　・Tanh
　・Softmax
　・Softplus
　・Swish
- Poolings:
　・AveragePooling2D
　・MaxPooling2D
- Normalize:
　・BatchNormalization
　・LRN
- Noise:
　・DropOut
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
