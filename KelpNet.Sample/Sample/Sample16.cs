using System;
using KelpNet.CL;
using KelpNet.Tools;

//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Sample
{
    class Sample16
    {
        private const string MODEL_FILE_PATH = "Data/ChainerModel.npz";

        public static void Run()
        {
            //読み込みたいネットワークの構成を FunctionStack に書き連ね、各 Function のパラメータを合わせる
            //ここで必ず name を Chainer の変数名に合わせておくこと

            FunctionStack<Real> nn = new FunctionStack<Real>(
                new Convolution2D<Real>(1, 2, 3, name: "conv1", gpuEnable: true),//必要であればGPUフラグも忘れずに
                new ReLU<Real>(),
                new MaxPooling2D<Real>(2, 2),
                new Convolution2D<Real>(2, 2, 2, name: "conv2", gpuEnable: true),
                new ReLU<Real>(),
                new MaxPooling2D<Real>(2, 2),
                new Linear<Real>(8, 2, name: "fl3"),
                new ReLU<Real>(),
                new Linear<Real>(2, 2, name: "fl4")
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
            SGD<Real> sgd = new SGD<Real>(0.1);
            sgd.SetUp(nn);

            //入力データ
            NdArray<Real> x = new NdArray<Real>(new Real[,,]{{
                { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.9f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f},
                { 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.8f, 0.9f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f},
                { 0.0f, 0.0f, 0.0f, 0.1f, 0.8f, 0.5f, 0.8f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f},
                { 0.0f, 0.0f, 0.0f, 0.3f, 0.3f, 0.1f, 0.7f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f},
                { 0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.1f, 0.7f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f},
                { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.7f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f},
                { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.4f, 0.8f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f},
                { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.8f, 0.4f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f},
                { 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.8f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                { 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.8f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                { 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.7f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
                { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
            }});

            //教師信号
            NdArray<Real> t = new NdArray<Real>(new Real[]{ 0.0f, 1.0f });

            //訓練を実施
            Trainer.Train(nn, x, t, new MeanSquaredError<Real>());

            //結果表示用に退避
            Convolution2D<Real> l2 = (Convolution2D<Real>)nn.Functions[0];


            //Updateを実行するとgradが消費されてしまうため値を先に出力
            Console.WriteLine("gw1");
            Console.WriteLine(l2.Weight.ToString("Grad"));

            Console.WriteLine("gb1");
            Console.WriteLine(l2.Bias.ToString("Grad"));

            //更新
            sgd.Update();

            Console.WriteLine("w1");
            Console.WriteLine(l2.Weight);

            Console.WriteLine("b1");
            Console.WriteLine(l2.Bias);
        }
    }
}
