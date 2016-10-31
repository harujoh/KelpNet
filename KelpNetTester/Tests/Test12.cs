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
    // 教師信号にラベル情報を混ぜ込むcDNI
    // モデルDと表現されている全層のDecoupledを非同期で実行
    // http://ralo23.hatenablog.com/entry/2016/10/22/233405
    class Test12
    {
        //ミニバッチの数
        //ミニバッチにC#標準のParallelを使用しているため、大きくし過ぎると遅くなるので注意
        const int BATCH_DATA_COUNT = 256;

        //一世代あたりの訓練回数
        const int TRAIN_DATA_COUNT = 234; // = 60,000 / 256

        //性能評価時のデータ数
        const int TEST_DATA_COUNT = 100;

        class ResultDataSet
        {
            public NdArray[] Result;
            public int[][] Label;

            public ResultDataSet(NdArray[] result, int[][] label)
            {
                this.Result = result;
                this.Label = label;
            }

            public double[][] GetTrainData()
            {
                //第一層の傾きを取得
                var train = new double[BATCH_DATA_COUNT][];

                for (int k = 0; k < BATCH_DATA_COUNT; k++)
                {
                    train[k] = new double[256 + 10];
                    train[k][256 + this.Label[k][0]] = 1.0;
                    Buffer.BlockCopy(this.Result[k].Data, 0, train[k], 0, sizeof(double) * 256);
                }

                return train;
            }
        }

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

            FunctionStack cDNI1 = new FunctionStack(
                new Linear(256 + 10, 1024, name: "cDNI1 Linear1"),
                new BatchNormalization(1024, name: "cDNI1 Nrom1"),
                new ReLU(name: "cDNI1 ReLU1"),
                new Linear(1024, 256, initialW: new double[1024, 256], name: "DNI1 Linear3")
            );

            FunctionStack cDNI2 = new FunctionStack(
                new Linear(256, 1024, name: "cDNI2 Linear1"),
                new BatchNormalization(1024, name: "cDNI2 Nrom1"),
                new ReLU(name: "cDNI2 ReLU1"),
                new Linear(1024, 256, initialW: new double[1024, 256], name: "cDNI2 Linear3")
            );

            FunctionStack cDNI3 = new FunctionStack(
                new Linear(256, 1024, name: "cDNI3 Linear1"),
                new BatchNormalization(1024, name: "cDNI3 Nrom1"),
                new ReLU(name: "cDNI3 ReLU1"),
                new Linear(1024, 256, initialW: new double[1024, 256], name: "cDNI3 Linear3")
            );

            //optimizerを宣言
            Layer1.SetOptimizer(new Adam(0.00003));
            Layer2.SetOptimizer(new Adam(0.00003));
            Layer3.SetOptimizer(new Adam(0.00003));
            Layer4.SetOptimizer(new Adam(0.00003));

            cDNI1.SetOptimizer(new Adam(0.00003));
            cDNI2.SetOptimizer(new Adam(0.00003));
            cDNI3.SetOptimizer(new Adam(0.00003));


            for (int epoch = 0; epoch < 10; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                //全体での誤差を集計
                List<double> totalLoss = new List<double>();

                List<double> cDNI1totalLoss = new List<double>();

                List<double> cDNI2totalLoss = new List<double>();

                List<double> cDNI3totalLoss = new List<double>();

                var layer1ForwardResults = new List<ResultDataSet>();
                var layer2ForwardResults = new List<ResultDataSet>();
                var layer3ForwardResults = new List<ResultDataSet>();

                var layer2BackwardResult = new List<NdArray[]>();
                var layer3BackwardResult = new List<NdArray[]>();
                var layer4BackwardResult = new List<NdArray[]>();

                var cDNI1Result = new List<NdArray[]>();
                var cDNI2Result = new List<NdArray[]>();
                var cDNI3Result = new List<NdArray[]>();

                int cDNI1Count = 0;
                int cDNI2Count = 0;
                int cDNI3Count = 0;

                //何回バッチを実行するか
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //訓練データからランダムにデータを取得
                    var datasetX = new MnistDataSet[10];
                    for (int j = 0; j < datasetX.Length; j++)
                    {
                        datasetX[j] = mnistData.GetRandomXSet(BATCH_DATA_COUNT);
                    }

                    //第一層を2～6回実行
                    var count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count; j++)
                    {
                        var layer1ForwardResult = Layer1.Forward(datasetX[j].Data);
                        var layer1ResultDataSet = new ResultDataSet(layer1ForwardResult, datasetX[j].Label);
                        layer1ForwardResults.Add(layer1ResultDataSet);

                        ////第一層の傾きを取得
                        cDNI1Result.Add(cDNI1.Forward(layer1ResultDataSet.GetTrainData()));

                        //第一層を更新
                        Layer1.Backward(cDNI1Result.Last());
                        Layer1.Update();

                        cDNI1Count++;
                    }

                    //第二層を2～6回実行
                    count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count && layer1ForwardResults.Count > 0; j++)
                    {
                        var layer2ForwardResult = Layer2.Forward(layer1ForwardResults[0].Result);
                        var layer2ResultDataSet = new ResultDataSet(layer2ForwardResult, layer1ForwardResults[0].Label);
                        layer2ForwardResults.Add(layer2ResultDataSet);
                        layer1ForwardResults.RemoveAt(0);

                        //第二層の傾きを取得
                        cDNI2Result.Add(cDNI2.Forward(layer2ResultDataSet.GetTrainData()));

                        //第二層を更新
                        layer2BackwardResult.Add(Layer2.Backward(cDNI2Result.Last()));
                        Layer2.Update();

                        cDNI2Count++;
                    }

                    //第一層用のcDNIの学習を2～6回実行
                    double cDNI1loss = 0;
                    count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count && layer2BackwardResult.Count > 0 && cDNI1Result.Count > 0; j++)
                    {
                        var DNI1lossResult = LossFunctions.MeanSquaredError(cDNI1Result[0], layer2BackwardResult[0], out cDNI1loss);
                        cDNI1Result.RemoveAt(0);
                        layer2BackwardResult.RemoveAt(0);

                        cDNI1.Backward(DNI1lossResult);
                        cDNI1.Update();
                        cDNI1totalLoss.Add(cDNI1loss);
                    }

                    //第三層を2～6回実行
                    count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count && layer2ForwardResults.Count > 0; j++)
                    {
                        var layer3ForwardResult = Layer3.Forward(layer2ForwardResults[0].Result);
                        var layer3ResultDataSet = new ResultDataSet(layer3ForwardResult, layer2ForwardResults[0].Label);
                        layer3ForwardResults.Add(layer3ResultDataSet);
                        layer2ForwardResults.RemoveAt(0);

                        //第三層の傾きを取得
                        cDNI3Result.Add(cDNI3.Forward(layer3ResultDataSet.GetTrainData()));

                        //第三層を更新
                        layer3BackwardResult.Add(Layer3.Backward(cDNI3Result.Last()));
                        Layer3.Update();

                        cDNI3Count++;
                    }


                    //第二層用のcDNIの学習を2～6回実行
                    double cDNI2loss = 0;
                    count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count && layer3BackwardResult.Count > 0 && cDNI2Result.Count > 0; j++)
                    {
                        var DNI2lossResult = LossFunctions.MeanSquaredError(cDNI2Result[0], layer3BackwardResult[0], out cDNI2loss);
                        cDNI2Result.RemoveAt(0);
                        layer3BackwardResult.RemoveAt(0);

                        cDNI2.Backward(DNI2lossResult);
                        cDNI2.Update();
                        cDNI2totalLoss.Add(cDNI2loss);
                    }


                    double sumLoss = 0;
                    //第四層を2～6回実行
                    count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count && layer3ForwardResults.Count > 0; j++)
                    {
                        var layer4ForwardResult = Layer4.Forward(layer3ForwardResults[0].Result);

                        //第四層の傾きを取得
                        var lossResult = LossFunctions.SoftmaxCrossEntropy(layer4ForwardResult, layer3ForwardResults[0].Label, out sumLoss);
                        layer3ForwardResults.RemoveAt(0);

                        //第四層を更新
                        layer4BackwardResult.Add(Layer4.Backward(lossResult));
                        Layer4.Update();
                        totalLoss.Add(sumLoss);
                    }

                    //第三層用のcDNIの学習を実行
                    double cDNI3loss = 0;
                    count = Mother.Dice.Next(2, 6);
                    for (int j = 0; j < count && layer4BackwardResult.Count > 0 && cDNI3Result.Count > 0; j++)
                    {
                        var DNI3lossResult = LossFunctions.MeanSquaredError(cDNI3Result[0], layer4BackwardResult[0], out cDNI3loss);
                        cDNI3Result.RemoveAt(0);
                        layer4BackwardResult.RemoveAt(0);

                        cDNI3.Backward(DNI3lossResult);
                        cDNI3.Update();
                        cDNI3totalLoss.Add(cDNI3loss);
                    }

                    Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                    //結果出力
                    Console.WriteLine("total loss " + totalLoss.Average());
                    Console.WriteLine("local loss " + sumLoss);

                    Console.WriteLine("\ncDNI1[" + cDNI1Count + "] total loss " + cDNI1totalLoss.Average());
                    Console.WriteLine("cDNI2[" + cDNI2Count + "] total loss " + cDNI2totalLoss.Average());
                    Console.WriteLine("cDNI3[" + cDNI3Count + "] total loss " + cDNI3totalLoss.Average());

                    Console.WriteLine("\ncDNI1 local loss " + cDNI1loss);
                    Console.WriteLine("cDNI2 local loss " + cDNI2loss);
                    Console.WriteLine("cDNI3 local loss " + cDNI3loss);

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
