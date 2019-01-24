using System;

namespace KelpNet.Sample.Samples
{
    //MLPによるXORの学習【回帰版】 ※精度が悪く何度か実行しないと望んだ結果を得られない
    class Sample02<T> where T : unmanaged, IComparable<T>
    {
        public static void Run()
        {
            //訓練回数
            const int learningCount = 1000;

            //訓練データ
            Real<T>[][] trainData =
            {
                new Real<T>[] { 0, 0 },
                new Real<T>[] { 1, 0 },
                new Real<T>[] { 0, 1 },
                new Real<T>[] { 1, 1 }
            };

            //訓練データラベル
            Real<T>[][] trainLabel =
            {
                new Real<T>[] { 0 },
                new Real<T>[] { 1 },
                new Real<T>[] { 1 },
                new Real<T>[] { 0 }
            };

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack<T> nn = new FunctionStack<T>(
                new Linear<T>(2, 2, name: "l1 Linear"),
                new ReLU<T>(name: "l1 ReLU"),
                new Linear<T>(2, 1, name: "l2 Linear")
            );

            //optimizerを宣言(今回はAdam)
            nn.SetOptimizer(new Adam<T>());

            //訓練ループ
            Console.WriteLine("Training...");
            for (int i = 0; i < learningCount; i++)
            {
                //今回はロス関数にMeanSquaredErrorを使う
                Trainer<T>.Train(nn, trainData[0], trainLabel[0], new MeanSquaredError<T>(), false);
                Trainer<T>.Train(nn, trainData[1], trainLabel[1], new MeanSquaredError<T>(), false);
                Trainer<T>.Train(nn, trainData[2], trainLabel[2], new MeanSquaredError<T>(), false);
                Trainer<T>.Train(nn, trainData[3], trainLabel[3], new MeanSquaredError<T>(), false);

                //訓練後に毎回更新を実行しなければ、ミニバッチとして更新できる
                nn.Update();
            }

            //訓練結果を表示
            Console.WriteLine("Test Start...");
            foreach (Real<T>[] val in trainData)
            {
                NdArray<T> result = nn.Predict(val)[0];
                Console.WriteLine(val[0] + " xor " + val[1] + " = " + (result.Data[0] > 0.5 ? 1 : 0) + " " + result);
            }
        }
    }
}
