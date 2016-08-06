using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using KelpNet;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //MLPによるMNIST（手書き文字）の学習
    class Test4
    {
        //ミニバッチの数
        //ミニバッチにC#標準のParallelを使用しているため、大きくし過ぎると遅くなるので注意
        const int BatchDataCount = 20;

        //一世代あたりの訓練回数
        const int TrainDataCount = 3000; // = 60000 / 20

        //性能評価時のデータ数
        const int TestDataCount = 100;


        public static void Run()
        {
            //MNISTのデータを用意する
            Console.WriteLine("MNIST Data Loading...");
            MnistData mnistData = new MnistData();


            Console.WriteLine("Training Start...");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Linear(28 * 28, 1024),
                new Sigmoid(),
                new Linear(1024, 10)
            );

            //optimizerを宣言
            nn.SetOptimizer(new MomentumSGD());

            //三世代学習
            for (int epoch = 0; epoch < 3; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                //全体での誤差を集計
                List<double> totalLoss = new List<double>();

                //何回バッチを実行するか
                for (int i = 1; i < TrainDataCount+1; i++)
                {
                    Console.WriteLine("\nbatch count " + i + "/" + TrainDataCount);

                    //バッチごとの誤差を集計
                    List<double> sumLoss = new List<double>();

                    //訓練データからランダムにデータを取得
                    var datasetX = mnistData.GetRandomXSet(BatchDataCount);

                    //バッチ学習を並列実行する
                    Parallel.For(0, BatchDataCount, j =>
                    {
                        var loss = nn.Train(datasetX.Data[j], datasetX.Label[j], LossFunctions.SoftmaxCrossEntropy);
                        totalLoss.Add(loss);
                        sumLoss.Add(loss);
                    });

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
