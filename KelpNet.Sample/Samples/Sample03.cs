using System;

namespace KelpNet.Sample.Samples
{
    //MLPによるSin関数の学習

    //学習対象の周期を増やしたり、サンプリング数(N)を増やすとスコアが悪化するので、
    //課題として挑戦してみると良いかもしれない
    class Sample03<T> where T : unmanaged, IComparable<T>
    {
        //学習回数
        const int EPOCH = 1000;

        //一周期の分割数
        const int N = 50;

        public static void Run()
        {
            RealArray<T>[] trainData = new RealArray<T>[N];
            RealArray<T>[] trainLabel = new RealArray<T>[N];

            for (int i = 0; i < N; i++)
            {
                //Sin波を一周期分用意
                Real<T> radian = (-Math.PI + Math.PI * 2.0 * i / (N - 1));
                trainData[i] = new[] { radian };
                trainLabel[i] = new Real<T>[] { Math.Sin(radian) };
            }

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack<T> nn = new FunctionStack<T>(
                new Linear<T>(1, 4, name: "l1 Linear"),
                new TanhActivation<T>(name: "l1 TanhActivation"),
                new Linear<T>(4, 1, name: "l2 Linear")
            );

            //optimizerの宣言
            nn.SetOptimizer(new SGD<T>());

            //訓練ループ
            for (int i = 0; i < EPOCH; i++)
            {
                //誤差集計用
                Real<T> loss = 0;

                for (int j = 0; j < N; j++)
                {
                    //ネットワークは訓練を実行すると戻り値に誤差が返ってくる
                    loss += Trainer<T>.Train(nn, trainData[j], trainLabel[j], new MeanSquaredError<T>());
                }

                if (i % (EPOCH / 10) == 0)
                {
                    Console.WriteLine("loss:" + loss / N);
                    Console.WriteLine("");
                }
            }

            //訓練結果を表示
            Console.WriteLine("Test Start...");

            foreach (RealArray<T> val in trainData)
            {
                Console.WriteLine(val[0] + ":" + nn.Predict(val)[0].Data[0]);
            }
        }
    }
}
