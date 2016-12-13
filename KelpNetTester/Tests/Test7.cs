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
    //参考： http://takatakamanbou.hatenablog.com/entry/2015/12/20/233232
    class Test7
    {
        //ミニバッチの数
        //ミニバッチにC#標準のParallelを使用しているため、大きくし過ぎると遅くなるので注意
        const int BATCH_DATA_COUNT = 128;

        //一世代あたりの訓練回数
        const int TRAIN_DATA_COUNT = 50000;

        //性能評価時のデータ数
        const int TEST_DATA_COUNT = 200;

        //中間層の数
        const int N = 30; //参考先リンクと同様の1000でも動作するがCPUでは遅いので

        public static void Run()
        {
            //MNISTのデータを用意する
            Console.WriteLine("MNIST Data Loading...");
            MnistData mnistData = new MnistData();

            Console.WriteLine("Training Start...");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Linear(28 * 28, N, name: "l1 Linear"), // L1
                new BatchNormalization(N, name: "l1 BatchNorm"),
                new ReLU(name: "l1 ReLU"),
                new Linear(N, N, name: "l2 Linear"), // L2
                new BatchNormalization(N, name: "l2 BatchNorm"),
                new ReLU(name: "l2 ReLU"),
                new Linear(N, N, name: "l3 Linear"), // L3
                new BatchNormalization(N, name: "l3 BatchNorm"),
                new ReLU(name: "l3 ReLU"),
                new Linear(N, N, name: "l4 Linear"), // L4
                new BatchNormalization(N, name: "l4 BatchNorm"),
                new ReLU(name: "l4 ReLU"),
                new Linear(N, N, name: "l5 Linear"), // L5
                new BatchNormalization(N, name: "l5 BatchNorm"),
                new ReLU(name: "l5 ReLU"),
                new Linear(N, N, name: "l6 Linear"), // L6
                new BatchNormalization(N, name: "l6 BatchNorm"),
                new ReLU(name: "l6 ReLU"),
                new Linear(N, N, name: "l7 Linear"), // L7
                new BatchNormalization(N, name: "l7 BatchNorm"),
                new ReLU(name: "l7 ReLU"),
                new Linear(N, N, name: "l8 Linear"), // L8
                new BatchNormalization(N, name: "l8 BatchNorm"),
                new ReLU(name: "l8 ReLU"),
                new Linear(N, N, name: "l9 Linear"), // L9
                new BatchNormalization(N, name: "l9 BatchNorm"),
                new ReLU(name: "l9 ReLU"),
                new Linear(N, N, name: "l10 Linear"), // L10
                new BatchNormalization(N, name: "l10 BatchNorm"),
                new ReLU(name: "l10 ReLU"),
                new Linear(N, N, name: "l11 Linear"), // L11
                new BatchNormalization(N, name: "l11 BatchNorm"),
                new ReLU(name: "l11 ReLU"),
                new Linear(N, N, name: "l12 Linear"), // L12
                new BatchNormalization(N, name: "l12 BatchNorm"),
                new ReLU(name: "l12 ReLU"),
                new Linear(N, N, name: "l13 Linear"), // L13
                new BatchNormalization(N, name: "l13 BatchNorm"),
                new ReLU(name: "l13 ReLU"),
                new Linear(N, N, name: "l14 Linear"), // L14
                new BatchNormalization(N, name: "l14 BatchNorm"),
                new ReLU(name: "l14 ReLU"),
                new Linear(N, 10, name: "l15 Linear") // L15
            );

            //この構成では学習が進まない
            //FunctionStack nn = new FunctionStack(
            //    new Linear(28 * 28, N), // L1
            //    new ReLU(),
            //    new Linear(N, N), // L2
            //    new ReLU(),
            //
            //    (中略)
            //
            //    new Linear(N, N), // L14
            //    new ReLU(),
            //    new Linear(N, 10) // L15
            //);

            //optimizerを宣言
            nn.SetOptimizer(new AdaGrad());

            //三世代学習
            for (int epoch = 0; epoch < 3; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                //全体での誤差を集計
                List<double> totalLoss = new List<double>();

                //何回バッチを実行するか
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //訓練データからランダムにデータを取得
                    MnistDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT);

                    //学習を実行
                    double sumLoss = Trainer.BatchTrain(nn, datasetX.Data, datasetX.Label, new SoftmaxCrossEntropy());
                    totalLoss.Add(sumLoss);

                    //20回バッチを動かしたら精度をテストする
                    if (i % 20 == 0)
                    {
                        //結果出力
                        Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                        Console.WriteLine("total loss " + totalLoss.Average());
                        Console.WriteLine("local loss " + sumLoss);
                        Console.WriteLine("");
                        Console.WriteLine("Testing...");

                        //テストデータからランダムにデータを取得
                        MnistDataSet datasetY = mnistData.GetRandomYSet(TEST_DATA_COUNT);

                        //テストを実行
                        double accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        Console.WriteLine("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
