using System;
using System.Linq;
using KelpNet.CL;

//using Real = System.Double;
using Real = System.Single;

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
            int[][] trainLabel =
            {
                new [] { 0 },
                new [] { 1 },
                new [] { 1 },
                new [] { 0 }
            };

            //ネットワークの構成は FunctionStack に書き連ねる
            FunctionStack<Real> nn = new FunctionStack<Real>(
                new Linear<Real>(2, 2, name: "l1 Linear"),
                new Sigmoid<Real>(name: "l1 Sigmoid"),
                new Linear<Real>(2, 2, name: "l2 Linear")
            );

            //optimizerを宣言
            //nn.SetOptimizer(new MomentumSGD<Real>());

            //訓練ループ
            Console.WriteLine("Training...");
            for (int i = 0; i < learningCount; i++)
            {
                for (int j = 0; j < trainData.Length; j++)
                {
                    //訓練実行時にロス関数を記述
                    Trainer.Train(nn, new[] { trainData[j] }, new[] { trainLabel[j] }, new SoftmaxCrossEntropy<Real>(), new MomentumSGD<Real>());
                }
            }

            //訓練結果を表示
            Console.WriteLine("Test Start...");
            foreach (Real[] input in trainData)
            {
                NdArray<Real> result = nn.Predict(input)[0];
                int resultIndex = Array.IndexOf(result.Data, result.Data.Max());
                Console.WriteLine(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
            }

            //学習の終わったネットワークを保存
            ModelIO<Real>.Save(nn, "test.nn");

            //学習の終わったネットワークを読み込み
            Function<Real> testnn = ModelIO<Real>.Load("test.nn");

            Console.WriteLine("Test Start...");
            foreach (Real[] input in trainData)
            {
                NdArray<Real> result = testnn.Predict(input)[0];
                int resultIndex = Array.IndexOf(result.Data, result.Data.Max());
                Console.WriteLine(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
            }
        }
    }
}
