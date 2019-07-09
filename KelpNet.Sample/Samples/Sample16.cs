using System;
using KelpNet.CL;
using KelpNet.Tools;

namespace KelpNet.Sample
{
    class Sample16
    {
        private const string MODEL_FILE_PATH = "Data/ChainerModel.npz";

        public static void Run()
        {
            //読み込みたいネットワークの構成を FunctionStack に書き連ね、各 Function のパラメータを合わせる
            //ここで必ず name を Chainer の変数名に合わせておくこと

            FunctionStack nn = new FunctionStack(
                new Convolution2D(1, 2, 3, name: "conv1", gpuEnable: true),//必要であればGPUフラグも忘れずに
                new ReLU(),
                new MaxPooling2D(2, 2),
                new Convolution2D(2, 2, 2, name: "conv2", gpuEnable: true),
                new ReLU(),
                new MaxPooling2D(2, 2),
                new Linear(8, 2, name: "fl3"),
                new ReLU(),
                new Linear(2, 2, name: "fl4")
            );

            /* Chainerでの宣言
            class NN(chainer.Chain):
                def __init__(self):
                    super(NN, self).__init__(
                        conv1 = L.Convolution2D(1,2,3),
                        conv2 = L.Convolution2D(2,2,2),
                        fl3 = L.Linear(8,2),
                        fl4 = L.Linear(2,2)
                    )

                def __call__(self, x):
                    h_conv1 = F.relu(self.conv1(x))
                    h_pool1 = F.max_pooling_2d(h_conv1, 2)
                    h_conv2 = F.relu(self.conv2(h_pool1))
                    h_pool2 = F.max_pooling_2d(h_conv2, 2)
                    h_fc1 = F.relu(self.fl3(h_pool2))
                    y = self.fl4(h_fc1)
                    return y
            */


            //パラメータを読み込み
            ChainerModelDataLoader.ModelLoad(MODEL_FILE_PATH, nn);

            //あとは通常通り使用する
            nn.SetOptimizer(new SGD(0.1));

            //入力データ
            NdArray x = new NdArray(new Real[,,]{{
                { 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.9, 0.2, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.1, 0.8, 0.5, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.3, 0.3, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0},
                { 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
            }});

            //教師信号
            Real[] t = { 0.0, 1.0 };

            //訓練を実施
            Trainer.Train(nn, x, t, new MeanSquaredError(), false);

            //結果表示用に退避
            Convolution2D l2 = (Convolution2D)nn.Functions[0];


            //Updateを実行するとgradが消費されてしまうため値を先に出力
            Console.WriteLine("gw1");
            Console.WriteLine(l2.Weight.ToString("Grad"));

            Console.WriteLine("gb1");
            Console.WriteLine(l2.Bias.ToString("Grad"));

            //更新
            nn.Update();

            Console.WriteLine("w1");
            Console.WriteLine(l2.Weight);

            Console.WriteLine("b1");
            Console.WriteLine(l2.Bias);
        }
    }
}
