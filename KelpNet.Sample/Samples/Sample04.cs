using System;
using KelpNet.CL;
using KelpNet.Tools;

namespace KelpNet.Sample
{
    //MLPによるMNIST（手書き文字）の学習
    class Sample04
    {
        //ミニバッチの数
        const int BATCH_DATA_COUNT = 20;

        //一世代あたりの訓練回数
        const int TRAIN_DATA_COUNT = 3000; // = 60000 / 20

        //性能評価時のデータ数
        const int TEST_DATA_COUNT = 200;


        public static void Run()
        {
            //MNISTのデータを用意する
            Console.WriteLine("MNIST Data Loading...");
            MnistData mnistData = new MnistData();


            Console.WriteLine("Training Start...");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Linear(28 * 28, 1024, name: "l1 Linear"),
                new Sigmoid(name: "l1 Sigmoid"),
                new Linear(1024, 10, name: "l2 Linear")
            );

            //optimizerを宣言
            nn.SetOptimizer(new MomentumSGD());

            //三世代学習
            for (int epoch = 0; epoch < 3; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                //全体での誤差を集計
                Real totalLoss = 0;
                long totalLossCount = 0;

                //何回バッチを実行するか
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //訓練データからランダムにデータを取得
                    TestDataSet datasetX = mnistData.Train.GetRandomDataSet(BATCH_DATA_COUNT);

                    //バッチ学習を並列実行する
                    Real sumLoss = Trainer.Train(nn, datasetX, new SoftmaxCrossEntropy());
                    totalLoss = sumLoss;
                    totalLossCount++;

                    //20回バッチを動かしたら精度をテストする
                    if (i % 20 == 0)
                    {
                        Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                        //結果出力
                        Console.WriteLine("total loss " + totalLoss / totalLossCount);
                        Console.WriteLine("local loss " + sumLoss);

                        Console.WriteLine("\nTesting...");

                        //テストデータからランダムにデータを取得
                        TestDataSet datasetY = mnistData.Eval.GetRandomDataSet(TEST_DATA_COUNT);

                        //テストを実行
                        Real accuracy = Trainer.Accuracy(nn, datasetY);
                        Console.WriteLine("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
