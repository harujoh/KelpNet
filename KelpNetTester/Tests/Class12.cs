using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet;
using KelpNet.Common;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Normalization;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    //Decoupled Neural Interfaces using Synthetic GradientsによるMNIST（手書き文字）の学習
    // http://ralo23.hatenablog.com/entry/2016/10/22/233405
    class Test12
    {
        //ミニバッチの数
        //ミニバッチにC#標準のParallelを使用しているため、大きくし過ぎると遅くなるので注意
        const int BATCH_DATA_COUNT = 20;

        //一世代あたりの訓練回数
        const int TRAIN_DATA_COUNT = 3000; // = 60000 / 20

        //性能評価時のデータ数
        const int TEST_DATA_COUNT = 100;

        public static void Run()
        {
            //MNISTのデータを用意する
            Console.WriteLine("MNIST Data Loading...");
            MnistData mnistData = new MnistData();


            Console.WriteLine("Training Start...");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack Layer1 = new FunctionStack(
                new Linear(28 * 28, 256, name: "l1 Linear"),
                new BatchNormalization(256, name: "l1 Norm"),
                new ReLU(name: "l1 ReLU")
            );

            FunctionStack Layer2 = new FunctionStack(
                new Linear(256, 256, name: "l2 Linear"),
                new BatchNormalization(256, name: "l2 Norm"),
                new ReLU(name: "l2 ReLU")
            );

            FunctionStack Layer3 = new FunctionStack(
                new Linear(256, 256, name: "l3 Linear"),
                new BatchNormalization(256, name: "l3 Norm"),
                new ReLU(name: "l3 ReLU")
            );

            FunctionStack Layer4 = new FunctionStack(
                new Linear(256, 10, name: "l4 Linear")
            );

            //FunctionStack自身もFunctionとして積み上げられる
            FunctionStack nn = new FunctionStack
            (
                Layer1,
                Layer2,
                Layer3,
                Layer4
            );

            FunctionStack DNI1 = new FunctionStack(
                new Linear(256, 1024, name: "DNI1 Linear1"),
                new BatchNormalization(1024, name: "DNI1 Nrom1"),
                new ReLU(name: "DNI1 ReLU1"),
                new Linear(1024, 1024, name: "DNI1 Linear2"),
                new BatchNormalization(1024, name: "DNI1 Nrom2"),
                new ReLU(name: "DNI1 ReLU2"),
                new Linear(1024, 256, initialW: new double[1024, 256], name: "DNI1 Linear3")
            );

            FunctionStack DNI2 = new FunctionStack(
                new Linear(256, 1024, name: "DNI2 Linear1"),
                new BatchNormalization(1024, name: "DNI2 Nrom1"),
                new ReLU(name: "DNI2 ReLU1"),
                new Linear(1024, 1024, name: "DNI2 Linear2"),
                new BatchNormalization(1024, name: "DNI2 Nrom2"),
                new ReLU(name: "DNI2 ReLU2"),
                new Linear(1024, 256, initialW: new double[1024, 256], name: "DNI2 Linear3")
            );

            FunctionStack DNI3 = new FunctionStack(
                new Linear(256, 1024, name: "DNI3 Linear1"),
                new BatchNormalization(1024, name: "DNI3 Nrom1"),
                new ReLU(name: "DNI3 ReLU1"),
                new Linear(1024, 1024, name: "DNI3 Linear2"),
                new BatchNormalization(1024, name: "DNI3 Nrom2"),
                new ReLU(name: "DNI3 ReLU2"),
                new Linear(1024, 256, initialW: new double[1024, 256], name: "DNI3 Linear3")
            );

            //optimizerを宣言
            Layer1.SetOptimizer(new Adam());
            Layer2.SetOptimizer(new Adam());
            Layer3.SetOptimizer(new Adam());
            Layer4.SetOptimizer(new Adam());
            DNI1.SetOptimizer(new Adam());
            DNI2.SetOptimizer(new Adam());
            DNI3.SetOptimizer(new Adam());


            //三世代学習
            for (int epoch = 0; epoch < 20; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                //全体での誤差を集計
                List<double> totalLoss = new List<double>();

                List<double> DNI1totalLoss = new List<double>();

                List<double> DNI2totalLoss = new List<double>();

                List<double> DNI3totalLoss = new List<double>();


                var layer1ForwardResult = new List<NdArray[]>();
                var layer2ForwardResult = new List<NdArray[]>();
                var layer3ForwardResult = new List<NdArray[]>();

                var layer2BackwardResult = new List<NdArray[]>();
                var layer3BackwardResult = new List<NdArray[]>();
                var layer4BackwardResult = new List<NdArray[]>();

                var DNI1Result = new List<NdArray[]>();
                var DNI2Result = new List<NdArray[]>();
                var DNI3Result = new List<NdArray[]>();
                var datasetXResult = new List<int[][]>();

                //何回バッチを実行するか
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //訓練データからランダムにデータを取得
                    var datasetX = new MnistDataSet[10];
                    for (int j = 0; j < datasetX.Length; j++)
                    {
                        datasetX[j] = mnistData.GetRandomXSet(BATCH_DATA_COUNT);
                    }

                    //第一層を3～10回実行
                    var count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count; j++)
                    {
                        layer1ForwardResult.Add(Layer1.Forward(datasetX[j].Data));
                        datasetXResult.Add(datasetX[j].Label);

                        DNI1Result.Add(DNI1.Forward(layer1ForwardResult.Last()));

                        //第一層の傾きを取得
                        if (0.5 < Mother.Dice.NextDouble())
                        {
                            //第一層を更新
                            Layer1.Backward(DNI1Result.Last());
                            Layer1.Update();
                        }
                    }

                    //第二層を3～10回実行
                    count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count && layer1ForwardResult.Count > 0; j++)
                    {
                        layer2ForwardResult.Add(Layer2.Forward(layer1ForwardResult[0]));
                        layer1ForwardResult.RemoveAt(0);

                        DNI2Result.Add(DNI2.Forward(layer2ForwardResult.Last()));

                        //第二層の傾きを取得
                        if (0.5 < Mother.Dice.NextDouble())
                        {
                            //第二層を更新
                            layer2BackwardResult.Add(Layer2.Backward(DNI2Result.Last()));
                            Layer2.Update();
                        }
                    }

                    //第一層用のDNIの学習を3～10回実行
                    double DNI1loss = 0;
                    count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count && layer2BackwardResult.Count > 0 && DNI1Result.Count > 0; j++)
                    {
                        var DNI1lossResult = LossFunctions.MeanSquaredError(DNI1Result[0], layer2BackwardResult[0], out DNI1loss);
                        DNI1Result.RemoveAt(0);
                        layer2BackwardResult.RemoveAt(0);

                        DNI1.Backward(DNI1lossResult);
                        DNI1.Update();
                        DNI1totalLoss.Add(DNI1loss);
                    }

                    //第三層を3～10回実行
                    count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count && layer2ForwardResult.Count > 0; j++)
                    {
                        layer3ForwardResult.Add(Layer3.Forward(layer2ForwardResult[0]));
                        layer2ForwardResult.RemoveAt(0);

                        DNI3Result.Add(DNI3.Forward(layer3ForwardResult.Last()));

                        //第三層の傾きを取得
                        if (0.5 < Mother.Dice.NextDouble())
                        {
                            //第三層を更新
                            layer3BackwardResult.Add(Layer3.Backward(DNI3Result.Last()));
                            Layer3.Update();
                        }
                    }


                    //第二層用のDNIの学習を3～10回実行
                    double DNI2loss = 0;
                    count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count && layer3BackwardResult.Count > 0 && DNI2Result.Count > 0; j++)
                    {
                        var DNI2lossResult = LossFunctions.MeanSquaredError(DNI2Result[0], layer3BackwardResult[0], out DNI2loss);
                        DNI2Result.RemoveAt(0);
                        layer3BackwardResult.RemoveAt(0);

                        DNI2.Backward(DNI2lossResult);
                        DNI2.Update();
                        DNI2totalLoss.Add(DNI2loss);
                    }


                    double sumLoss = 0;
                    //第四層を3～10回実行
                    count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count && layer3ForwardResult.Count > 0 && datasetXResult.Count > 0; j++)
                    {
                        var layer4ForwardResult = Layer4.Forward(layer3ForwardResult[0]);

                        //第四層の傾きを取得
                        var lossResult = LossFunctions.SoftmaxCrossEntropy(layer4ForwardResult, datasetXResult[0], out sumLoss);
                        datasetXResult.RemoveAt(0);

                        //第四層を更新
                        layer4BackwardResult.Add(Layer4.Backward(lossResult));
                        Layer4.Update();
                        totalLoss.Add(sumLoss);
                    }

                    //第三層用のDNIの学習を実行
                    double DNI3loss = 0;
                    count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count && layer4BackwardResult.Count > 0 && DNI3Result.Count > 0; j++)
                    {
                        var DNI3lossResult = LossFunctions.MeanSquaredError(DNI3Result[0], layer4BackwardResult[0], out DNI3loss);
                        DNI3Result.RemoveAt(0);
                        layer4BackwardResult.RemoveAt(0);

                        DNI3.Backward(DNI3lossResult);
                        DNI3.Update();
                        DNI3totalLoss.Add(DNI3loss);
                    }

                    Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                    //結果出力
                    Console.WriteLine("total loss " + totalLoss.Average());
                    Console.WriteLine("local loss " + sumLoss);

                    Console.WriteLine("\nDNI1 total loss " + DNI1totalLoss.Average());
                    Console.WriteLine("DNI2 total loss " + DNI2totalLoss.Average());
                    Console.WriteLine("DNI3 total loss " + DNI3totalLoss.Average());

                    Console.WriteLine("\nDNI1 local loss " + DNI1loss);
                    Console.WriteLine("DNI2 local loss " + DNI2loss);
                    Console.WriteLine("DNI3 local loss " + DNI3loss);

                    //20回バッチを動かしたら精度をテストする
                    if (i % 20 == 0)
                    {
                        Console.WriteLine("\nTesting...");

                        //テストデータからランダムにデータを取得
                        var datasetY = mnistData.GetRandomYSet(TEST_DATA_COUNT);

                        //テストを実行
                        var accuracy = nn.Accuracy(datasetY.Data, datasetY.Label);
                        Console.WriteLine("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
