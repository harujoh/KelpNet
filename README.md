# KelpNet
KelpNetはChainerを参考に全てC#で実装された深層学習のライブラリです

## 特徴
- 行列を使わずに実装しているので Deep Learning の学習コストを抑えることが出来ます
- KerasやChainerが採用している、関数を積み重ねるように記述するコーディングスタイルを採用しています
- 全ソースが公開されているため、どこで何をしているかを全て観測できます
- C#特有の記述を極力避けているため、C#以外のプログラマーでも、読み切れるようになっていると思います
- 並列計算にOpenCLを採用しているため様々な演算装置で処理の高速化が可能です

### C#で作られているメリット
- 開発環境の構築が簡単で、これからプログラミングを学ぶ人にとって学習の敷居を低くすることが出来ます
- WindowsFormやUnity等、ビジュアライズを行うための選択肢が豊富です
- 様々な環境で動作するアプリケーションを作成できるため、成果物の公開が簡単です

## このライブラリについて
このライブラリは、他に先行するライブラリと比較すると、まだまだ機能が少ない状態です。
また、私自身が深層学習を勉強中であり、間違っている点もあるかと思います。
細やかなことでも構いませんので、何かお気づきの点が御座いましたら、お気軽にご連絡ください。

また、Gitを目下勉強中で、なんらかのマナー違反などが目につきましたら、ご指摘いただけると助かります。


## 連絡方法
ご質問、ご要望はIssuesにご登録頂くかTwitterから適当なつぶやきにご返信ください

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
