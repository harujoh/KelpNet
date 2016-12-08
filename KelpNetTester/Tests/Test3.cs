using System;
using KelpNet;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //MLPによるSin関数の学習

    //学習対象の周期を増やしたり、サンプリング数(N)を増やすとスコアが悪化するので、
    //課題として挑戦してみると良いかもしれない
    class Test3
    {
        //学習回数
        const int EPOCH = 1000;

        //一周期の分割数
        const int N = 50;

        public static void Run()
        {
            double[][] trainData = new double[N][];
            double[][] trainLabel = new double[N][];

            for (int i = 0; i < N; i++)
            {
                //Sin波を一周期分用意
                double radian = -Math.PI + Math.PI * 2.0 * i / (N - 1);
                trainData[i] = new[] { radian };
                trainLabel[i] = new[] { Math.Sin(radian) };
            }

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Linear(1, 4, name: "l1 Linear"),
                new Tanh(name: "l1 Tanh"),
                new Linear(4, 1, name: "l2 Linear")
            );

            //optimizerの宣言
            nn.SetOptimizer(new SGD());

            //訓練ループ
            for (int i = 0; i < EPOCH; i++)
            {
                //誤差集計用
                double loss = 0.0;

                for (int j = 0; j < N; j++)
                {
                    //ネットワークは訓練を実行すると戻り値に誤差が返ってくる
                    loss += Trainer.Train(nn, trainData[j], trainLabel[j], LossFunctions.MeanSquaredError);
                }

                if (i % (EPOCH / 10) == 0)
                {
                    Console.WriteLine("loss:" + loss / N);
                    Console.WriteLine("");
                }
            }

            //訓練結果を表示
            Console.WriteLine("Test Start...");

            foreach (double[] val in trainData)
            {
                Console.WriteLine(val[0] + ":" + Trainer.Predict(nn,val).Data[0]);
            }
        }
    }
}
