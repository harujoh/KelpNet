using System;
using System.Linq;
using KelpNet.CL;
using KelpNet.Tools;

namespace KelpNet.Sample
{
    //Decoupled Neural Interfaces using Synthetic GradientsによるMNIST（手書き文字）の学習
    // 教師信号にラベル情報を混ぜ込むcDNI
    // モデルDと表現されている全層のDecoupledを非同期で実行
    // http://ralo23.hatenablog.com/entry/2016/10/22/233405
    class Sample12
    {
        //ミニバッチの数
        const int BATCH_DATA_COUNT = 256;

        //一世代あたりの訓練回数
        const int TRAIN_DATA_COUNT = 234; // = 60,000 / 256

        //性能評価時のデータ数
        const int TEST_DATA_COUNT = 100;

        class ResultDataSet
        {
            public readonly NdArray Result;
            public readonly NdArray Label;

            public ResultDataSet(NdArray result, NdArray label)
            {
                this.Result = result;
                this.Label = label;
            }

            public NdArray GetTrainData()
            {
                Real[] tmp = new Real[(256 + 10) * BATCH_DATA_COUNT];

                for (int i = 0; i < BATCH_DATA_COUNT; i++)
                {
                    tmp[256 + (int)this.Label.Data[i * this.Label.Length] + i * (256 + 10)] = 1;
                    Array.Copy(this.Result.Data, i * this.Result.Length, tmp, i * (256 + 10), 256);
                }

                return NdArray.Convert(tmp, new[] { 256 + 10 }, BATCH_DATA_COUNT);
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
                new Linear(1024, 256, initialW: new Real[1024, 256], name: "DNI1 Linear3")
            );

            FunctionStack cDNI2 = new FunctionStack(
                new Linear(256 + 10, 1024, name: "cDNI2 Linear1"),
                new BatchNormalization(1024, name: "cDNI2 Nrom1"),
                new ReLU(name: "cDNI2 ReLU1"),
                new Linear(1024, 256, initialW: new Real[1024, 256], name: "cDNI2 Linear3")
            );

            FunctionStack cDNI3 = new FunctionStack(
                new Linear(256 + 10, 1024, name: "cDNI3 Linear1"),
                new BatchNormalization(1024, name: "cDNI3 Nrom1"),
                new ReLU(name: "cDNI3 ReLU1"),
                new Linear(1024, 256, initialW: new Real[1024, 256], name: "cDNI3 Linear3")
            );

            //optimizerを宣言
            Layer1.SetOptimizer(new Adam(0.00003f));
            Layer2.SetOptimizer(new Adam(0.00003f));
            Layer3.SetOptimizer(new Adam(0.00003f));
            Layer4.SetOptimizer(new Adam(0.00003f));

            cDNI1.SetOptimizer(new Adam(0.00003f));
            cDNI2.SetOptimizer(new Adam(0.00003f));
            cDNI3.SetOptimizer(new Adam(0.00003f));

            for (int epoch = 0; epoch < 10; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                //全体での誤差を集計
                Real totalLoss = 0;
                Real cDNI1totalLoss = 0;
                Real cDNI2totalLoss = 0;
                Real cDNI3totalLoss = 0;

                long totalLossCount = 0;
                long cDNI1totalLossCount = 0;
                long cDNI2totalLossCount = 0;
                long cDNI3totalLossCount = 0;


                //何回バッチを実行するか
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //訓練データからランダムにデータを取得
                    TestDataSet datasetX = mnistData.Train.GetRandomDataSet(BATCH_DATA_COUNT);

                    //第一層を実行
                    NdArray layer1ForwardResult = Layer1.Forward(datasetX.Data)[0];
                    ResultDataSet layer1ResultDataSet = new ResultDataSet(layer1ForwardResult, datasetX.Label);

                    //第一層の傾きを取得
                    NdArray cDNI1Result = cDNI1.Forward(layer1ResultDataSet.GetTrainData())[0];

                    //第一層の傾きを適用
                    layer1ForwardResult.Grad = cDNI1Result.Data.ToArray();

                    //第一層を更新
                    Layer1.Backward(layer1ForwardResult);
                    layer1ForwardResult.ParentFunc = null;
                    Layer1.Update();

                    //第二層を実行
                    NdArray layer2ForwardResult = Layer2.Forward(layer1ResultDataSet.Result)[0];
                    ResultDataSet layer2ResultDataSet = new ResultDataSet(layer2ForwardResult, layer1ResultDataSet.Label);

                    //第二層の傾きを取得
                    NdArray cDNI2Result = cDNI2.Forward(layer2ResultDataSet.GetTrainData())[0];

                    //第二層の傾きを適用
                    layer2ForwardResult.Grad = cDNI2Result.Data.ToArray();

                    //第二層を更新
                    Layer2.Backward(layer2ForwardResult);
                    layer2ForwardResult.ParentFunc = null;


                    //第一層用のcDNIの学習を実行
                    Real cDNI1loss = new MeanSquaredError().Evaluate(cDNI1Result, new NdArray(layer1ResultDataSet.Result.Grad, cDNI1Result.Shape, cDNI1Result.BatchCount));

                    Layer2.Update();

                    cDNI1.Backward(cDNI1Result);
                    cDNI1.Update();

                    cDNI1totalLoss += cDNI1loss;
                    cDNI1totalLossCount++;

                    //第三層を実行
                    NdArray layer3ForwardResult = Layer3.Forward(layer2ResultDataSet.Result)[0];
                    ResultDataSet layer3ResultDataSet = new ResultDataSet(layer3ForwardResult, layer2ResultDataSet.Label);

                    //第三層の傾きを取得
                    NdArray cDNI3Result = cDNI3.Forward(layer3ResultDataSet.GetTrainData())[0];

                    //第三層の傾きを適用
                    layer3ForwardResult.Grad = cDNI3Result.Data.ToArray();

                    //第三層を更新
                    Layer3.Backward(layer3ForwardResult);
                    layer3ForwardResult.ParentFunc = null;

                    //第二層用のcDNIの学習を実行
                    Real cDNI2loss = new MeanSquaredError().Evaluate(cDNI2Result, new NdArray(layer2ResultDataSet.Result.Grad, cDNI2Result.Shape, cDNI2Result.BatchCount));

                    Layer3.Update();

                    cDNI2.Backward(cDNI2Result);
                    cDNI2.Update();

                    cDNI2totalLoss += cDNI2loss;
                    cDNI2totalLossCount++;

                    //第四層を実行
                    NdArray layer4ForwardResult = Layer4.Forward(layer3ResultDataSet.Result)[0];

                    //第四層の傾きを取得
                    Real sumLoss = new SoftmaxCrossEntropy().Evaluate(layer4ForwardResult, layer3ResultDataSet.Label);

                    //第四層を更新
                    Layer4.Backward(layer4ForwardResult);
                    layer4ForwardResult.ParentFunc = null;

                    totalLoss += sumLoss;
                    totalLossCount++;

                    //第三層用のcDNIの学習を実行
                    Real cDNI3loss = new MeanSquaredError().Evaluate(cDNI3Result, new NdArray(layer3ResultDataSet.Result.Grad, cDNI3Result.Shape, cDNI3Result.BatchCount));

                    Layer4.Update();

                    cDNI3.Backward(cDNI3Result);
                    cDNI3.Update();

                    cDNI3totalLoss += cDNI3loss;
                    cDNI3totalLossCount++;

                    Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                    //結果出力
                    Console.WriteLine("total loss " + totalLoss / totalLossCount);
                    Console.WriteLine("local loss " + sumLoss);

                    Console.WriteLine("\ncDNI1 total loss " + cDNI1totalLoss / cDNI1totalLossCount);
                    Console.WriteLine("cDNI2 total loss " + cDNI2totalLoss / cDNI2totalLossCount);
                    Console.WriteLine("cDNI3 total loss " + cDNI3totalLoss / cDNI3totalLossCount);

                    Console.WriteLine("\ncDNI1 local loss " + cDNI1loss);
                    Console.WriteLine("cDNI2 local loss " + cDNI2loss);
                    Console.WriteLine("cDNI3 local loss " + cDNI3loss);

                    //20回バッチを動かしたら精度をテストする
                    if (i % 20 == 0)
                    {
                        Console.WriteLine("\nTesting...");

                        //テストデータからランダムにデータを取得
                        TestDataSet datasetY = mnistData.Eval.GetRandomDataSet(TEST_DATA_COUNT);

                        //テストを実行
                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        Console.WriteLine("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
