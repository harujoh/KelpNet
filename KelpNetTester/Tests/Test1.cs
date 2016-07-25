using System;
using KelpNet;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //MLPによるXORの学習
    class Test1
    {
        public static void Run()
        {
            //訓練回数
            const int learningCount = 10000;

            //訓練データ
            double[][] trainData =
            {
                new[] { 0.0, 0.0 },
                new[] { 1.0, 0.0 },
                new[] { 0.0, 1.0 },
                new[] { 1.0, 1.0 }
            };

            //訓練データラベル
            double[][] trainLabel =
            {
                new[] { 0.0},
                new[] { 1.0},
                new[] { 1.0},
                new[] { 0.0}
            };

            //ネットワークの構成は FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Linear(2, 2),
                new Sigmoid(),
                new Linear(2, 2)
            );

            //optimizerを宣言
            nn.SetOptimizer(new MomentumSGD());


            //訓練ループ
            Console.WriteLine("Training...");
            for (int i = 0; i < learningCount; i++)
            {
                for (int j = 0; j < trainData.Length; j++)
                {
                    //訓練実行時にロス関数を記述
                    nn.Train(trainData[j], trainLabel[j], LossFunctions.SoftmaxCrossEntropy);
                    nn.Update();
                }
            }


            //訓練結果を表示
            Console.WriteLine("Test Start...");
            foreach (var val in trainData)
            {
                var input = NdArray.FromArray(val);
                Console.WriteLine(input + ":" + nn.Predict(input));
            }
        }
    }
}
