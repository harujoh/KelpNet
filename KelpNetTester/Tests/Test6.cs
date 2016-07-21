using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using KelpNet;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
using KelpNet.Functions.Poolings;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //5層CNNによるMNIST（手書き文字）の認識
    //Test4と違うのはネットワークの構成とOptimizerだけです
    class Test6
    {
        //ミニバッチの数
        //ミニバッチにC#標準のParallelを使用しているため、大きくし過ぎると遅くなるので注意
        const int BatchDataCount = 20;

        //一世代あたりの訓練回数
        const int TrainDataCount = 3000; // = 60000 / 20

        //性能評価時のデータ数
        const int TeachDataCount = 100;

        public static void Run()
        {
            //MNISTのデータを用意する
            Console.WriteLine("MNIST Data Loading...");
            MnistData mnistData = new MnistData();


            Console.WriteLine("Training Start...");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Convolution2D(1, 32, 5, pad: 2),
                new ReLU(),
                new MaxPooling(2, 2),
                new Convolution2D(32, 64, 5, pad: 2),
                new ReLU(),
                new MaxPooling(2, 2),
                new Linear(7 * 7 * 64, 1024),
                new Dropout(),
                new ReLU(),
                new Linear(1024, 10)
            );

            //optimizerを宣言
            nn.Optimizer = new Adam(nn);

            //三世代学習
            for (int epoch = 1; epoch < 3; epoch++)
            {
                Console.WriteLine("epoch " + epoch);

                //全体での誤差を集計
                List<double> totalLoss = new List<double>();

                //何回バッチを実行するか
                for (int i = 0; i < TrainDataCount; i++)
                {
                    Console.WriteLine("\nbatch count " + (i + 1) + "/" + TrainDataCount);

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
                        var datasetY = mnistData.GetRandomYSet(TeachDataCount);

                        //テストを実行
                        var accuracy = nn.Accuracy(datasetY.Data, datasetY.Label);
                        Console.WriteLine("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
