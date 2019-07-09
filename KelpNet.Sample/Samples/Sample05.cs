using System;
using KelpNet.CL;

namespace KelpNet.Sample
{
    //エクセルCNNの再現
    class Sample05
    {
        public static void Run()
        {
            //各初期値を記述
            Real[,,,] initial_W1 =
                {
                    {{{1.0,  0.5, 0.0}, { 0.5, 0.0, -0.5}, {0.0, -0.5, -1.0}}},
                    {{{0.0, -0.1, 0.1}, {-0.3, 0.4,  0.7}, {0.5, -0.2,  0.2}}}
                };
            Real[] initial_b1 = { 0.5, 1.0 };

            Real[,,,] initial_W2 =
                {
                    {{{-0.1,  0.6}, {0.3, -0.9}}, {{ 0.7, 0.9}, {-0.2, -0.3}}},
                    {{{-0.6, -0.1}, {0.3,  0.3}}, {{-0.5, 0.8}, { 0.9,  0.1}}}
                };
            Real[] initial_b2 = { 0.1, 0.9 };

            Real[,] initial_W3 =
                {
                    {0.5, 0.3, 0.4, 0.2, 0.6, 0.1, 0.4, 0.3},
                    {0.6, 0.4, 0.9, 0.1, 0.5, 0.2, 0.3, 0.4}
                };
            Real[] initial_b3 = { 0.01, 0.02 };

            Real[,] initial_W4 = { { 0.8, 0.2 }, { 0.4, 0.6 } };
            Real[] initial_b4 = { 0.02, 0.01 };


            //入力データ
            NdArray x = new NdArray(new Real[, ,]{{
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


            //層の中身をチェックしたい場合は、層単体でインスタンスを持つ
            Convolution2D l2 = new Convolution2D(1, 2, 3, initialW: initial_W1, initialb: initial_b1, name: "l2 Conv2D");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                l2, //new Convolution2D(1, 2, 3, initialW: initial_W1, initialb: initial_b1),
                new ReLU(name: "l2 ReLU"),
                //new AveragePooling(2, 2, name: "l2 AVGPooling"),
                new MaxPooling2D(2, 2, name: "l2 MaxPooling"),
                new Convolution2D(2, 2, 2, initialW: initial_W2, initialb: initial_b2, name: "l3 Conv2D"),
                new ReLU(name: "l3 ReLU"),
                //new AveragePooling(2, 2, name: "l3 AVGPooling"),
                new MaxPooling2D(2, 2, name: "l3 MaxPooling"),
                new Linear(8, 2, initialW: initial_W3, initialb: initial_b3, name: "l4 Linear"),
                new ReLU(name: "l4 ReLU"),
                new Linear(2, 2, initialW: initial_W4, initialb: initial_b4, name: "l5 Linear")
            );

            nn.SetOptimizer(new SGD(0.1));

            //訓練を実施
            Trainer.Train(nn, x, t, new MeanSquaredError(), false);

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
