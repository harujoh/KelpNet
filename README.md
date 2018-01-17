# KelpNet
KelpNetはC#で実装された深層学習のライブラリです

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

## 特徴
- 行列演算をライブラリに頼らないため全ソースが可読になっており、どこで何をしているかを全て観測できます
- KerasやChainerが採用している、層を書き連ねるコーディングスタイルを採用しています
- 並列演算にOpenCLを採用しているため、GPUだけでなくCPUやFPGA等の様々な演算装置で処理を並列化できます
> ※OpenCLを使用するためには対応するドライバの追加インストールが必要になることがあります
> - Intel製 CPU GPU: https://software.intel.com/en-us/articles/opencl-drivers
> - AMD製 CPU GPU: http://www.amd.com/ja-jp/solutions/professional/hpc/opencl
> - Nvidia製 GPU: https://developer.nvidia.com/opencl

### C#で作られているメリット
- 開発環境の構築が容易で、プログラミング初学者にも学びやすい言語です
- WindowsFormやUnity等、処理結果を視覚的に表示するための選択肢が豊富です
- PCや携帯、ブラウザ、組み込み機器と様々な環境に向けたアプリケーションの開発ができます

## このライブラリについて
このライブラリの基幹部分はChainerを参考に実装されています。
その為ほとんどの関数パラメータがChainerと同じになっており、Chainer向けのサンプルを参考に開発することが可能になっています。

## 連絡方法
ご質問、ご要望は日本語、英語を問わず Issues へご登録頂くか Twitter からご連絡ください。
細やかなことでも構いませんので、何かお気づきの点が御座いましたら、お気軽にお申し付けください。

Twitter: https://twitter.com/harujoh


## 実行環境
- KelpNet : .Net Standard 1.1
- KelpNet.Tools, KelpNet.Sample : .Net Framework 4.5

## License
- Apache License 2.0

## 実装済み関数
- Activations:
　・ELU
　・LeakyReLU
　・ReLU
　・Sigmoid
　・Tanh
　・Softmax
　・Softplus
　・Swish
- Connections:
　・Convolution2D
　・Deconvolution2D
　・EmbedID
　・Linear
　・LSTM
- Poolings:
　・AveragePooling
　・MaxPooling
- LossFunctions:
　・MeanSquaredError
　・SoftmaxCrossEntropy
- Optimizers:
　・AdaDelta
　・AdaGrad
　・Adam
　・MomentumSGD
　・RMSprop
　・SGD
- Normalize:
　・BatchNormalization
　・LRN
- Noise:
　・DropOut
　・StochasticDepth
