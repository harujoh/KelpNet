using System;
using KelpNet;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //MLPでSin関数を学習する

    //学習対象の周期を増やしたり、サンプリング数(N)を増やすとスコアが悪化するので、
    //課題として挑戦してみると良いかもしれない
    class Test3
    {
        public static void Run()
        {
            //学習回数
            const int EPOCH = 1000;

            const int N = 50;
            double[][] trainData = new double[N][];
            double[][] trainLabel = new double[N][];

            for (int i = 0; i < N; i++)
            {
                //Sin波を一周期分用意
                trainData[i] = new[] { -Math.PI + Math.PI * 2.0 * i / (N - 1) };
                trainLabel[i] = new[] { Math.Sin(trainData[i][0]) };
            }

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Linear(1, 4),
                new Tanh(),
                new Linear(4, 1)
            );

            //optimizerの宣言を省略するとデフォルトのSGD(0.1)が使用される
            //nn.SetOptimizer(new SGD(0.1));


            //訓練ループ
            for (int i = 0; i < EPOCH; i++)
            {
                Console.WriteLine("training..." + i + "/" + EPOCH);

                //誤差集計用
                double loss = 0.0;

                for (int j = 0; j < N; j++)
                {
                    //ネットワークは訓練を実行すると戻り値に誤差が返ってくる
                    loss += nn.Train(trainData[j], trainLabel[j], LossFunctions.MeanSquaredError);

                    //今回は逐次更新
                    nn.Update();
                }

                Console.WriteLine("loss:" + loss / N);
            }

            //訓練結果を表示
            Console.WriteLine("Test Start...");

            foreach (var val in trainData)
            {
                Console.WriteLine(val[0] + ":" + nn.Predict(NdArray.FromArray(val)).Data[0]);
            }
        }
    }
}
