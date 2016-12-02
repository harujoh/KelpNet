# Kelp.Net
Kelp.NetはChainerを参考に全てC#で実装された深層学習のライブラリです。


##特徴
- 『Chainer』や『Keras』のように、関数を書き連ねるように記述するモダンなスタイルを採用しています。
- 全てC#で記述されており、また依存する外部ライブラリが無いため、どこで何をしているかを全て追うことが可能です。
- 行列を使わずに実装しているので、Deep Learning の仕組みを把握するための学習コストを抑えることが出来ます。
- C#特有の記述を極力避けているため、C#以外のプログラマーでも、読み切れるようになっていると思います。

###C#で作られているメリット
- 開発環境の構築が容易で、これからプログラミングを学ぶ人にとっても敷居を低くすることが出来ます。
- CPU実行時において純粋なPythonで作られているライブラリと比較して3～10倍ほどのパフォーマンスを発揮します。
- デバッガですべての値が観測可能なため『値を確認しようとしたらメモリアドレスが書いてある』と言ったことがありません。
- WindowsFormやUnity等、ビジュアライズを行うための選択肢が豊富です。

##このライブラリについて
このライブラリは、他に先行するライブラリと比較すると、まだまだ機能が少ない状態です。
また私自身が深層学習を勉強中であり、間違っている点もあるかと思います。
細やかなことでも構いませんので、何かお気づきの点が御座いましたら、お気軽にご連絡ください。

また、Gitを目下勉強中で、なんらかのマナー違反などが目につきましたら、ご指摘いただけると助かります。


##連絡方法
ご質問、ご要望はTwitterから適当なつぶやきに返信を頂ければ反応が早いと思います。

TwitterID: harujoh


最後に、このライブラリが誰かの学習の助けになれば幸いです。


## License
Apache License 2.0


##実装済み関数
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
