# KelpNet
KelpNetはC#で実装された深層学習のライブラリです

 - ChainerStyle
```csharp
FunctionStack nn = new FunctionStack(
    new Convolution2D(32, 64, 5, pad: 2, name: "l1 Conv2D"),
    new ReLU(name: "l1 ReLU"),
    new MaxPooling(2, 2, name: "l1 MaxPooling"),
    new Linear(7 * 7 * 64, 1024, name: "l2 Linear"),
    new ReLU(name: "l2 ReLU"),
    new Dropout(name: "l2 DropOut"),
    new Linear(1024, 10, name: "l3 Linear")
);
```
 - KerasStyle
```csharp
FunctionStack nn = new FunctionStack(
    new Convolution2D(32, 64, 5, pad: 2, name: "l1 Conv2D", activation: new ReLU(name: "l1 ReLU")),
    new MaxPooling(2, 2, name: "l1 MaxPooling"),
    new Linear(7 * 7 * 64, 1024, name: "l2 Linear", activation: new ReLU(name: "l2 ReLU")),
    new Dropout(name: "l2 DropOut"),
    new Linear(1024, 10, name: "l3 Linear")
);
```

## 特徴
- 行列演算をライブラリに頼らないため全ソースが可読になっており、どこで何をしているかを全て観測できます
- KerasやChainerが採用している、関数を積み重ねるように記述するコーディングスタイルを採用しています
- C#特有の記述を極力避けているため、C#以外のプログラマーでも読み切れるようになっています
- 並列演算にOpenCLを採用しているため、GPUだけでなくCPUやFPGA等の様々な演算装置で処理を並列化できます
> ※OpenCLを使用するためには対応するドライバの追加インストールが必要になることがあります
> - Intel製 CPU GPU: https://software.intel.com/en-us/articles/opencl-drivers
> - AMD製 CPU GPU: http://www.amd.com/ja-jp/solutions/professional/hpc/opencl
> - Nvidia製 GPU: https://developer.nvidia.com/opencl

### C#で作られているメリット
- 開発環境の構築が容易で、プログラミング初学者にも学びやすい言語です
- WindowsFormやUnity等、処理結果を視覚的に表示するための選択肢が豊富です
- PCや携帯、組み込み機器等、様々なプラットフォームに向けたアプリケーションの開発ができます

## このライブラリについて
このライブラリの基幹部分はChainerを参考に実装されています。
その為ほとんどの関数パラメータがChainerと同じになっており、Chainer向けのサンプルを参考に開発することが可能になっています。
> ※ただし四則演算や初等関数の自動微分が実装されていないため、自動微分が必要なサンプルは一工夫が必要です



## 連絡方法
ご質問、ご要望は Issues へご登録頂くか Twitter からご連絡ください。
細やかなことでも構いませんので、何かお気づきの点が御座いましたら、お気軽にご連絡ください。

Twitter: https://twitter.com/harujoh



## License
- KelpNet [Apache License 2.0]
- Cloo [MIT License] https://sourceforge.net/projects/cloo/

## 実装済み関数
- Activations:
　・ELU
　・LeakyReLU
　・ReLU
　・Sigmoid
　・Tanh
　・Softmax
　・Softplus
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
- Others:
　・DropOut
　・BatchNormalization
 
 
 最後に、このライブラリが誰かの学習の助けになれば幸いです
