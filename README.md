# Kelp.Net
Kelp.NetはChainerを参考にC#で実装された深層学習の学習用ライブラリです。


##特徴
- ChainerやKerasのように、層を積み上げるように記述するモダンなスタイルを採用しています。
- 全てC#で記述されており、依存する外部ライブラリも無いため、どこで何をしているかを全て追うことが可能です。
- 線形代数を用いず、ひたすらforを使って実装しているので、全容を把握するための学習コストを抑えることが出来ます。
- 極力C#特有の記述を避けているため、C#以外のプログラマーでも、比較的読めるようになっていると思います。

###C#で作られているメリット
- 開発環境の構築が容易で、これからプログラミングを学ぶ人にとっても敷居を低くすることが出来ます。
- WindowsFormやUnity等、ビジュアライズを行うための選択肢が豊富です。
- CPU実行時において純粋なPythonで作られているライブラリと比較して3～10倍ほどのパフォーマンスを発揮します。
- デバッガですべての値が観測可能なため『値を確認しようとしたらメモリアドレスが書いてある』と言ったことがありません。

##おことわり
このライブラリは私自身の学習用に作ったもので、他に先行するライブラリと比較すると、まだまだ機能が少ない状態です。
また間違っている点もあるかと思いますので、なにかお気づきの点がありましたら、お気軽にご連絡ください。

また、Gitを目下勉強中で、なんらかのマナー違反などが目につきましたら、ご指摘ください。


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
