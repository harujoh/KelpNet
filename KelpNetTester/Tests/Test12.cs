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

            public NdArray[] GetTrainData()
            {
                //第一層の傾きを取得
                double[][] train = new double[BATCH_DATA_COUNT][];

                for (int k = 0; k < BATCH_DATA_COUNT; k++)
                {
                    train[k] = new double[256 + 10];
                    train[k][256 + this.Label[k][0]] = 1.0;
                    Buffer.BlockCopy(this.Result[k].Data, 0, train[k], 0, sizeof(double) * 256);
                }

                return NdArray.FromArray(train);
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
                new Linear(256+10, 1024, name: "cDNI2 Linear1"),
                new BatchNormalization(1024, name: "cDNI2 Nrom1"),
                new ReLU(name: "cDNI2 ReLU1"),
                new Linear(1024, 256, initialW: new double[1024, 256], name: "cDNI2 Linear3")
            );

            FunctionStack cDNI3 = new FunctionStack(
                new Linear(256+10, 1024, name: "cDNI3 Linear1"),
                new BatchNormalization(1024, name: "cDNI3 Nrom1"),
                new ReLU(name: "cDNI3 ReLU1"),
                new Linear(1024, 256, initialW: new double[1024, 256], name: "cDNI3 Linear3")
            );

            //optimizerを宣言
            Adam l1Adam = new Adam(Layer1.Parameters, 0.00003);
            Adam l2Adam = new Adam(Layer2.Parameters, 0.00003);
            Adam l3Adam = new Adam(Layer3.Parameters, 0.00003);
            Adam l4Adam = new Adam(Layer4.Parameters, 0.00003);

            Adam cDNI1Adam = new Adam(cDNI1.Parameters, 0.00003);
            Adam cDNI2Adam = new Adam(cDNI2.Parameters, 0.00003);
            Adam cDNI3Adam = new Adam(cDNI3.Parameters, 0.00003);


            for (int epoch = 0; epoch < 10; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                //全体での誤差を集計
                List<double> totalLoss = new List<double>();

                List<double> cDNI1totalLoss = new List<double>();

                List<double> cDNI2totalLoss = new List<double>();

                List<double> cDNI3totalLoss = new List<double>();


                //何回バッチを実行するか
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //訓練データからランダムにデータを取得
                    MnistDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT);

                    //第一層を実行
                    NdArray[] layer1ForwardResult = Layer1.Forward(NdArray.FromArray(datasetX.Data));
                    ResultDataSet layer1ResultDataSet = new ResultDataSet(layer1ForwardResult, datasetX.Label);

                    ////第一層の傾きを取得
                    NdArray[] cDNI1Result = cDNI1.Forward(layer1ResultDataSet.GetTrainData());

                    //第一層を更新
                    Layer1.Backward(cDNI1Result);
                    Layer1.Update(l1Adam);


                    //第二層を実行
                    NdArray[] layer2ForwardResult = Layer2.Forward(layer1ResultDataSet.Result);
                    ResultDataSet layer2ResultDataSet = new ResultDataSet(layer2ForwardResult, layer1ResultDataSet.Label);

                    //第二層の傾きを取得
                    NdArray[] cDNI2Result =cDNI2.Forward(layer2ResultDataSet.GetTrainData());

                    //第二層を更新
                    NdArray[] layer2BackwardResult = Layer2.Backward(cDNI2Result);
                    Layer2.Update(l2Adam);


                    //第一層用のcDNIの学習を実行
                    double cDNI1loss = 0;
                    NdArray[] DNI1lossResult = LossFunctions.MeanSquaredError(cDNI1Result, layer2BackwardResult, out cDNI1loss);

                    cDNI1.Backward(DNI1lossResult);
                    cDNI1.Update(cDNI1Adam);
                    cDNI1totalLoss.Add(cDNI1loss);


                    //第三層を実行
                    NdArray[] layer3ForwardResult = Layer3.Forward(layer2ResultDataSet.Result);
                    ResultDataSet layer3ResultDataSet = new ResultDataSet(layer3ForwardResult, layer2ResultDataSet.Label);

                    //第三層の傾きを取得
                    NdArray[] cDNI3Result = cDNI3.Forward(layer3ResultDataSet.GetTrainData());

                    //第三層を更新
                    NdArray[] layer3BackwardResult = Layer3.Backward(cDNI3Result);
                    Layer3.Update(l3Adam);


                    //第二層用のcDNIの学習を実行
                    double cDNI2loss = 0;
                    NdArray[] DNI2lossResult = LossFunctions.MeanSquaredError(cDNI2Result, layer3BackwardResult, out cDNI2loss);

                    cDNI2.Backward(DNI2lossResult);
                    cDNI2.Update(cDNI2Adam);
                    cDNI2totalLoss.Add(cDNI2loss);


                    //第四層を実行
                    NdArray[] layer4ForwardResult = Layer4.Forward(layer3ResultDataSet.Result);

                    //第四層の傾きを取得
                    double sumLoss = 0;
                    NdArray[] lossResult = LossFunctions.SoftmaxCrossEntropy(layer4ForwardResult, NdArray.FromArray(layer3ResultDataSet.Label), out sumLoss);

                    //第四層を更新
                    NdArray[] layer4BackwardResult = Layer4.Backward(lossResult);
                    Layer4.Update(l4Adam);
                    totalLoss.Add(sumLoss);


                    //第三層用のcDNIの学習を実行
                    double cDNI3loss = 0;
                    NdArray[] DNI3lossResult = LossFunctions.MeanSquaredError(cDNI3Result, layer4BackwardResult, out cDNI3loss);

                    cDNI3.Backward(DNI3lossResult);
                    cDNI3.Update(cDNI3Adam);
                    cDNI3totalLoss.Add(cDNI3loss);


                    Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                    //結果出力
                    Console.WriteLine("total loss " + totalLoss.Average());
                    Console.WriteLine("local loss " + sumLoss);

                    Console.WriteLine("\ncDNI1 total loss " + cDNI1totalLoss.Average());
                    Console.WriteLine("cDNI2 total loss " + cDNI2totalLoss.Average());
                    Console.WriteLine("cDNI3 total loss " + cDNI3totalLoss.Average());

                    Console.WriteLine("\ncDNI1 local loss " + cDNI1loss);
                    Console.WriteLine("cDNI2 local loss " + cDNI2loss);
                    Console.WriteLine("cDNI3 local loss " + cDNI3loss);

                    //20回バッチを動かしたら精度をテストする
                    if (i % 20 == 0)
                    {
                        Console.WriteLine("\nTesting...");

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
