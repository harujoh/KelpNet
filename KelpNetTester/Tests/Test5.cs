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
                    {{{(Real)1.0, (Real) 0.5, (Real)0.0}, { (Real)0.5, (Real)0.0, -(Real)0.5}, {(Real)0.0, -(Real)0.5, -(Real)1.0}}},
                    {{{(Real)0.0, -(Real)0.1, (Real)0.1}, {-(Real)0.3, (Real)0.4,  (Real)0.7}, {(Real)0.5, -(Real)0.2,  (Real)0.2}}}
                };
            Real[] initial_b1 = { (Real)0.5, (Real)1.0 };

            Real[,,,] initial_W2 =
                {
                    {{{-(Real)0.1,  (Real)0.6}, {(Real)0.3, -(Real)0.9}}, {{ (Real)0.7, (Real)0.9}, {-(Real)0.2, -(Real)0.3}}},
                    {{{-(Real)0.6, -(Real)0.1}, {(Real)0.3,  (Real)0.3}}, {{-(Real)0.5, (Real)0.8}, { (Real)0.9,  (Real)0.1}}}
                };
            Real[] initial_b2 = { (Real)0.1, (Real)0.9 };

            Real[,] initial_W3 =
                {
                    {(Real)0.5, (Real)0.3, (Real)0.4, (Real)0.2, (Real)0.6, (Real)0.1, (Real)0.4, (Real)0.3},
                    {(Real)0.6, (Real)0.4, (Real)0.9, (Real)0.1, (Real)0.5, (Real)0.2, (Real)0.3, (Real)0.4}
                };
            Real[] initial_b3 = { (Real)0.01, (Real)0.02 };

            Real[,] initial_W4 = { { (Real)0.8, (Real)0.2 }, { (Real)0.4, (Real)0.6 } };
            Real[] initial_b4 = { (Real)0.02, (Real)0.01 };


            //入力データ
            Real[,,] x = {{
                    { (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.2, (Real)0.9, (Real)0.2, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0},
                    { (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.2, (Real)0.8, (Real)0.9, (Real)0.1, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0},
                    { (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.1, (Real)0.8, (Real)0.5, (Real)0.8, (Real)0.1, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0},
                    { (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.3, (Real)0.3, (Real)0.1, (Real)0.7, (Real)0.2, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0},
                    { (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.1, (Real)0.0, (Real)0.1, (Real)0.7, (Real)0.2, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0},
                    { (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.1, (Real)0.7, (Real)0.1, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0},
                    { (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.4, (Real)0.8, (Real)0.1, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0},
                    { (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.8, (Real)0.4, (Real)0.1, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0},
                    { (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.2, (Real)0.8, (Real)0.3, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0},
                    { (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.1, (Real)0.8, (Real)0.2, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0},
                    { (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.1, (Real)0.7, (Real)0.2, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0},
                    { (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.3, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0, (Real)0.0}
                }};

            //教師信号
            Real[] t = { (Real)0.0, (Real)1.0 };


            //層の中身をチェックしたい場合は、層単体でインスタンスを持つ
            Convolution2D l2 = new Convolution2D(1, 2, 3, initialW: initial_W1, initialb: initial_b1, name: "l2 Conv2D");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                l2, //new Convolution2D(1, 2, 3, initialW: initial_W1, initialb: initial_b1),
                new ReLU(name: "l2 ReLU"),
                //new AveragePooling(2, 2, name: "l2 Pooling"),
                new MaxPooling(2, 2, name: "l2 Pooling"),
                new Convolution2D(2, 2, 2, initialW: initial_W2, initialb: initial_b2, name: "l3 Conv2D"),
                new ReLU(name: "l3 ReLU"),
                //new AveragePooling(2, 2, name: "l3 Pooling"),
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
