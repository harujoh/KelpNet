using System;
using KelpNet.Tools;

namespace KelpNet.Sample.Samples
{
    //MLPによるXORの学習
    class Sample01<T> where T : unmanaged, IComparable<T>
    {
        public static void Run()
        {
            //訓練回数
            const int learningCount = 10000;

            //訓練データ
            RealArray<T>[] trainData =
            {
                new Real<T>[] { 0, 0 },
                new Real<T>[] { 1, 0 },
                new Real<T>[] { 0, 1 },
                new Real<T>[] { 1, 1 }
            };

            //訓練データラベル
            RealArray<T>[] trainLabel =
            {
                new Real<T>[] { 0 },
                new Real<T>[] { 1 },
                new Real<T>[] { 1 },
                new Real<T>[] { 0 }
            };

            //ネットワークの構成は FunctionStack に書き連ねる
            FunctionStack<T> nn = new FunctionStack<T>(
                new Linear<T>(2, 2, name: "l1 Linear"),
                new Sigmoid<T>(name: "l1 Sigmoid"),
                new Linear<T>(2, 2, name: "l2 Linear")
            );

            //optimizerを宣言
            nn.SetOptimizer(new MomentumSGD<T>());

            //訓練ループ
            Console.WriteLine("Training...");
            for (int i = 0; i < learningCount; i++)
            {
                for (int j = 0; j < trainData.Length; j++)
                {
                    //訓練実行時にロス関数を記述
                    Trainer<T>.Train(nn, trainData[j], trainLabel[j], new SoftmaxCrossEntropy<T>());
                }
            }

            //訓練結果を表示
            Console.WriteLine("Test Start...");
            foreach (RealArray<T> input in trainData)
            {
                NdArray<T> result = nn.Predict(input)[0];
                //int resultIndex = Array.IndexOf(result.Data, result.Data.Max());
                int resultIndex = result.Data.MaxIndex();
                Console.WriteLine(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
            }

            //学習の終わったネットワークを保存
            ModelIO<T>.Save(nn, "test.nn");

            //学習の終わったネットワークを読み込み
            FunctionStack<T> testnn = ModelIO<T>.Load("test.nn");

            Console.WriteLine("Test Start...");
            foreach (RealArray<T> input in trainData)
            {
                NdArray<T> result = testnn.Predict(input)[0];
                //int resultIndex = Array.IndexOf(result.Data, result.Data.Max());
                int resultIndex = result.Data.MaxIndex();
                Console.WriteLine(input[0] + " xor " + input[1] + " = " + resultIndex + " " + result);
            }
        }
    }
}
