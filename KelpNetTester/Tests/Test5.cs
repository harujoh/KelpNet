using System;
using KelpNet.Common;
using KelpNet.Common.Tools;
using KelpNet.Functions;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Poolings;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //エクセルCNNの再現
    class Test5
    {
        public static void Run()
        {
            //各初期値を記述
            Real[,,,] initial_W1 =
                {
                    {{{1.0f,  0.5f, 0.0f}, { 0.5f, 0.0f, -0.5f}, {0.0f, -0.5f, -1.0f}}},
                    {{{0.0f, -0.1f, 0.1f}, {-0.3f, 0.4f,  0.7f}, {0.5f, -0.2f,  0.2f}}}
                };
            Real[] initial_b1 = { 0.5f, 1.0f };

            Real[,,,] initial_W2 =
                {
                    {{{-0.1f,  0.6f}, {0.3f, -0.9f}}, {{ 0.7f, 0.9f}, {-0.2f, -0.3f}}},
                    {{{-0.6f, -0.1f}, {0.3f,  0.3f}}, {{-0.5f, 0.8f}, { 0.9f,  0.1f}}}
                };
            Real[] initial_b2 = { 0.1f, 0.9f };

            Real[,] initial_W3 =
                {
                    {0.5f, 0.3f, 0.4f, 0.2f, 0.6f, 0.1f, 0.4f, 0.3f},
                    {0.6f, 0.4f, 0.9f, 0.1f, 0.5f, 0.2f, 0.3f, 0.4f}
                };
            Real[] initial_b3 = { 0.01f, 0.02f };

            Real[,] initial_W4 = { { 0.8f, 0.2f }, { 0.4f, 0.6f } };
            Real[] initial_b4 = { 0.02f, 0.01f };


            //入力データ
            Real[,,] x = {{
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
                }};

            //教師信号
            Real[] t = { 0.0f, 1.0f };


            //層の中身をチェックしたい場合は、層単体でインスタンスを持つ
            Convolution2D l2 = new Convolution2D(1, 2, 3, initialW: initial_W1, initialb: initial_b1, name: "l2 Conv2D");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                l2, //new Convolution2D(1, 2, 3, initialW: initial_W1, initialb: initial_b1),
                new ReLU(name: "l2 ReLU"),
                //new AveragePooling(2, 2),
                new MaxPooling(2, 2, name: "l2 Pooling"),
                new Convolution2D(2, 2, 2, initialW: initial_W2, initialb: initial_b2, name: "l3 Conv2D"),
                new ReLU(name: "l3 ReLU"),
                //new AveragePooling(2, 2),
                new MaxPooling(2, 2, name: "l3 Pooling"),
                new Linear(8, 2, initialW: initial_W3, initialb: initial_b3, name: "l4 Linear"),
                new ReLU(name: "l4 ReLU"),
                new Linear(2, 2, initialW: initial_W4, initialb: initial_b4, name: "l5 Linear")
            );

            //optimizerの宣言を省略するとデフォルトのSGD(0.1)が使用される
            nn.SetOptimizer(new SGD());

            //訓練を実施
            Trainer.Train(nn, x, t, new MeanSquaredError(), false);

            //Updateを実行するとgradが消費されてしまうため値を先に出力
            Console.WriteLine("gw1");
            Console.WriteLine(l2.gW);

            Console.WriteLine("gb1");
            Console.WriteLine(l2.gb);

            //更新
            nn.Update();

            Console.WriteLine("w1");
            Console.WriteLine(l2.W);

            Console.WriteLine("b1");
            Console.WriteLine(l2.b);
        }
    }
}
