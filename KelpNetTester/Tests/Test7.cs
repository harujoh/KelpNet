using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Normalization;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //バッチノーマライゼーションを使った15層MLPによるMNIST（手書き文字）の学習
    //参考　http://takatakamanbou.hatenablog.com/entry/2015/12/20/233232
    class Test7
    {
        //ミニバッチの数
        //ミニバッチにC#標準のParallelを使用しているため、大きくし過ぎると遅くなるので注意
        const int BatchDataCount = 128;

        //一世代あたりの訓練回数
        const int TrainDataCount = 50000; // = 60000 / 20

        //性能評価時のデータ数
        const int TestDataCount = 100;


        public static void Run()
        {
            //MNISTのデータを用意する
            Console.WriteLine("MNIST Data Loading...");
            MnistData mnistData = new MnistData();


            Console.WriteLine("Training Start...");

            //中間層の数
            const int N = 30; //参考先リンクと同様の1000でも動作するがCPUでは遅いので

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Linear(28 * 28, N), // L1
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L2
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L3
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L4
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L5
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L6
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L7
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L8
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L9
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L10
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L11
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L12
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L13
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, N), // L14
                new BatchNormalization(N),
                new ReLU(),
                new Linear(N, 10)    // L15
            );

            //この構成では学習が進まない
            //FunctionStack nn = new FunctionStack(
            //    new Linear(28 * 28, N), // L1
            //    new ReLU(),
            //    new Linear(N, N), // L2
            //    new ReLU(),
            //    new Linear(N, N), // L3
            //    new ReLU(),
            //    new Linear(N, N), // L4
            //    new ReLU(),
            //    new Linear(N, N), // L5
            //    new ReLU(),
            //    new Linear(N, N), // L6
            //    new ReLU(),
            //    new Linear(N, N), // L7
            //    new ReLU(),
            //    new Linear(N, N), // L8
            //    new ReLU(),
            //    new Linear(N, N), // L9
            //    new ReLU(),
            //    new Linear(N, N), // L10
            //    new ReLU(),
            //    new Linear(N, N), // L11
            //    new ReLU(),
            //    new Linear(N, N), // L12
            //    new ReLU(),
            //    new Linear(N, N), // L13
            //    new ReLU(),
            //    new Linear(N, N), // L14
            //    new ReLU(),
            //    new Linear(N, 10)    // L15
            //);


            //optimizerを宣言
            nn.SetOptimizer(new MomentumSGD(0.1, 0.9));

            //三世代学習
            for (int epoch = 0; epoch < 3; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                //全体での誤差を集計
                List<double> totalLoss = new List<double>();

                //何回バッチを実行するか
                for (int i = 1; i < TrainDataCount + 1; i++)
                {
                    Console.WriteLine("\nbatch count " + i + "/" + TrainDataCount);

                    //訓練データからランダムにデータを取得
                    var datasetX = mnistData.GetRandomXSet(BatchDataCount);

                    //バッチ学習を並列実行する
                    var sumLoss = nn.BatchTrain(datasetX.Data, datasetX.Label, LossFunctions.SoftmaxCrossEntropy, BatchDataCount);
                    totalLoss.AddRange(sumLoss);

                    //バッチ更新
                    nn.Update();

                    //結果出力
                    Console.WriteLine("total loss " + totalLoss.Average());
                    Console.WriteLine("local loss " + sumLoss.Average());

                    //20回バッチを動かしたら精度をテストする
                    if (i % 20 == 0)
                    {
                        Console.WriteLine("Testing...");

                        //テストデータからランダムにデータを取得
                        var datasetY = mnistData.GetRandomYSet(TestDataCount);

                        //テストを実行
                        var accuracy = nn.Accuracy(datasetY.Data, datasetY.Label);
                        Console.WriteLine("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
