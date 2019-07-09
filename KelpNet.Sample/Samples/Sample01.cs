using System;
using System.Linq;
using KelpNet.CL;

namespace KelpNet.Sample
{
    //MLPによるXORの学習
    class Sample01
    {
        public static void Run()
        {
            //訓練回数
            const int learningCount = 10000;

            //訓練データ
            Real[][] trainData =
            {
                new Real[] { 0, 0 },
                new Real[] { 1, 0 },
                new Real[] { 0, 1 },
                new Real[] { 1, 1 }
            };

            //訓練データラベル
            Real[][] trainLabel =
            {
                new Real[] { 0 },
                new Real[] { 1 },
                new Real[] { 1 },
                new Real[] { 0 }
            };

            //ネットワークの構成は FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Linear(2, 2, name: "l1 Linear"),
                new Sigmoid(name: "l1 Sigmoid"),
                new Linear(2, 2, name: "l2 Linear")
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
                    Trainer.Train(nn, trainData[j], trainLabel[j], new SoftmaxCrossEntropy());
                }
            }

            //訓練結果を表示
            Console.WriteLine("Test Start...");
            foreach (Real[] input in trainData)
            {
                NdArray result = nn.Predict(input)[0];
                int resultIndex = Array.IndexOf(result.Data, result.Data.Max());
                Console.WriteLine(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
            }

            //学習の終わったネットワークを保存
            ModelIO.Save(nn, "test.nn");

            //学習の終わったネットワークを読み込み
            Function testnn = ModelIO.Load("test.nn");

            Console.WriteLine("Test Start...");
            foreach (Real[] input in trainData)
            {
                NdArray result = testnn.Predict(input)[0];
                int resultIndex = Array.IndexOf(result.Data, result.Data.Max());
                Console.WriteLine(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
            }
        }
    }
}
