using System;
using System.Linq;
using KelpNet.CL;
using KelpNet.Tools;

//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Sample
{
    //Decoupled Neural Interfaces using Synthetic GradientsによるMNIST（手書き文字）の学習
    // http://ralo23.hatenablog.com/entry/2016/10/22/233405
    class Sample11
    {
        //ミニバッチの数
        const int BATCH_DATA_COUNT = 200;

        //一世代あたりの訓練回数
        const int TRAIN_DATA_COUNT = 300; // = 60000 / 20

        //性能評価時のデータ数
        const int TEST_DATA_COUNT = 1000;

        public static void Run()
        {
            //MNISTのデータを用意する
            Console.WriteLine("MNIST Data Loading...");
            MnistData<Real> mnistData = new MnistData<Real>();


            Console.WriteLine("Training Start...");

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack<Real> Layer1 = new FunctionStack<Real>(
                new Linear<Real>(28 * 28, 256, name: "l1 Linear"),
                new BatchNormalization<Real>(256, name: "l1 Norm"),
                new ReLU<Real>(name: "l1 ReLU")
            );

            FunctionStack<Real> Layer2 = new FunctionStack<Real>(
                new Linear<Real>(256, 256, name: "l2 Linear"),
                new BatchNormalization<Real>(256, name: "l2 Norm"),
                new ReLU<Real>(name: "l2 ReLU")
            );

            FunctionStack<Real> Layer3 = new FunctionStack<Real>(
                new Linear<Real>(256, 256, name: "l3 Linear"),
                new BatchNormalization<Real>(256, name: "l3 Norm"),
                new ReLU<Real>(name: "l3 ReLU")
            );

            FunctionStack<Real> Layer4 = new FunctionStack<Real>(
                new Linear<Real>(256, 10, name: "l4 Linear")
            );

            //FunctionStack自身もFunctionとして積み上げられる
            FunctionStack<Real> nn = new FunctionStack<Real>
            (
                Layer1,
                Layer2,
                Layer3,
                Layer4
            );

            FunctionStack<Real> DNI1 = new FunctionStack<Real>(
                new Linear<Real>(256, 1024, name: "DNI1 Linear1"),
                new BatchNormalization<Real>(1024, name: "DNI1 Nrom1"),
                new ReLU<Real>(name: "DNI1 ReLU1"),
                new Linear<Real>(1024, 1024, name: "DNI1 Linear2"),
                new BatchNormalization<Real>(1024, name: "DNI1 Nrom2"),
                new ReLU<Real>(name: "DNI1 ReLU2"),
                new Linear<Real>(1024, 256, initialW: new Real[1024, 256], name: "DNI1 Linear3")
            );

            FunctionStack<Real> DNI2 = new FunctionStack<Real>(
                new Linear<Real>(256, 1024, name: "DNI2 Linear1"),
                new BatchNormalization<Real>(1024, name: "DNI2 Nrom1"),
                new ReLU<Real>(name: "DNI2 ReLU1"),
                new Linear<Real>(1024, 1024, name: "DNI2 Linear2"),
                new BatchNormalization<Real>(1024, name: "DNI2 Nrom2"),
                new ReLU<Real>(name: "DNI2 ReLU2"),
                new Linear<Real>(1024, 256, initialW: new Real[1024, 256], name: "DNI2 Linear3")
            );

            FunctionStack<Real> DNI3 = new FunctionStack<Real>(
                new Linear<Real>(256, 1024, name: "DNI3 Linear1"),
                new BatchNormalization<Real>(1024, name: "DNI3 Nrom1"),
                new ReLU<Real>(name: "DNI3 ReLU1"),
                new Linear<Real>(1024, 1024, name: "DNI3 Linear2"),
                new BatchNormalization<Real>(1024, name: "DNI3 Nrom2"),
                new ReLU<Real>(name: "DNI3 ReLU2"),
                new Linear<Real>(1024, 256, initialW: new Real[1024, 256], name: "DNI3 Linear3")
            );

            //optimizerを宣言
            Adam<Real> L1adam = new Adam<Real>();
            Adam<Real> L2adam = new Adam<Real>();
            Adam<Real> L3adam = new Adam<Real>();
            Adam<Real> L4adam = new Adam<Real>();

            L1adam.SetUp(Layer1);
            L2adam.SetUp(Layer2);
            L3adam.SetUp(Layer3);
            L4adam.SetUp(Layer4);

            Adam<Real> DNI1adam = new Adam<Real>();
            Adam<Real> DNI2adam = new Adam<Real>();
            Adam<Real> DNI3adam = new Adam<Real>();

            DNI1adam.SetUp(DNI1);
            DNI2adam.SetUp(DNI2);
            DNI3adam.SetUp(DNI3);

            //三世代学習
            for (int epoch = 0; epoch < 20; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

                Real totalLoss = 0;
                Real DNI1totalLoss = 0;
                Real DNI2totalLoss = 0;
                Real DNI3totalLoss = 0;

                long totalLossCount = 0;
                long DNI1totalLossCount = 0;
                long DNI2totalLossCount = 0;
                long DNI3totalLossCount = 0;

                //何回バッチを実行するか
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //訓練データからランダムにデータを取得
                    TestDataSet<Real> datasetX = mnistData.Train.GetRandomDataSet(BATCH_DATA_COUNT);

                    //第一層を実行
                    NdArray<Real> layer1ForwardResult = Layer1.Forward(datasetX.Data)[0];

                    //第一層の傾きを取得
                    NdArray<Real> DNI1Result = DNI1.Forward(layer1ForwardResult)[0];

                    //第一層の傾きを適用
                    layer1ForwardResult.Grad = DNI1Result.Data.ToArray();

                    //第一層を更新
                    Layer1.Backward(layer1ForwardResult);
                    layer1ForwardResult.ParentFunc = null; //Backwardを実行したので計算グラフを切っておく
                    L1adam.Update();

                    //第二層を実行
                    NdArray<Real> layer2ForwardResult = Layer2.Forward(layer1ForwardResult)[0];

                    //第二層の傾きを取得
                    NdArray<Real> DNI2Result = DNI2.Forward(layer2ForwardResult)[0];

                    //第二層の傾きを適用
                    layer2ForwardResult.Grad = DNI2Result.Data.ToArray();

                    //第二層を更新
                    Layer2.Backward(layer2ForwardResult);
                    layer2ForwardResult.ParentFunc = null;

                    //第一層用のDNIの学習を実行
                    Real DNI1loss = new MeanSquaredError<Real>().Evaluate(DNI1Result, new NdArray<Real>(layer1ForwardResult.Grad, DNI1Result.Shape, DNI1Result.BatchCount));

                    L2adam.Update();

                    DNI1.Backward(DNI1Result);
                    DNI1adam.Update();

                    DNI1totalLoss += DNI1loss;
                    DNI1totalLossCount++;

                    //第三層を実行
                    NdArray<Real> layer3ForwardResult = Layer3.Forward(layer2ForwardResult)[0];

                    //第三層の傾きを取得
                    NdArray<Real> DNI3Result = DNI3.Forward(layer3ForwardResult)[0];

                    //第三層の傾きを適用
                    layer3ForwardResult.Grad = DNI3Result.Data.ToArray();

                    //第三層を更新
                    Layer3.Backward(layer3ForwardResult);
                    layer3ForwardResult.ParentFunc = null;

                    //第二層用のDNIの学習を実行
                    Real DNI2loss = new MeanSquaredError<Real>().Evaluate(DNI2Result, new NdArray<Real>(layer2ForwardResult.Grad, DNI2Result.Shape, DNI2Result.BatchCount));

                    L3adam.Update();

                    DNI2.Backward(DNI2Result);
                    DNI2adam.Update();

                    DNI2totalLoss += DNI2loss;
                    DNI2totalLossCount++;

                    //第四層を実行
                    NdArray<Real> layer4ForwardResult = Layer4.Forward(layer3ForwardResult)[0];

                    //第四層の傾きを取得
                    Real sumLoss = new SoftmaxCrossEntropy<Real>().Evaluate(layer4ForwardResult, datasetX.Label);

                    //第四層を更新
                    Layer4.Backward(layer4ForwardResult);
                    layer4ForwardResult.ParentFunc = null;

                    totalLoss += sumLoss;
                    totalLossCount++;

                    //第三層用のDNIの学習を実行
                    Real DNI3loss = new MeanSquaredError<Real>().Evaluate(DNI3Result, new NdArray<Real>(layer3ForwardResult.Grad, DNI3Result.Shape, DNI3Result.BatchCount));

                    L4adam.Update();

                    DNI3.Backward(DNI3Result);
                    DNI3adam.Update();

                    DNI3totalLoss += DNI3loss;
                    DNI3totalLossCount++;

                    Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                    //結果出力
                    Console.WriteLine("total loss " + totalLoss / totalLossCount);
                    Console.WriteLine("local loss " + sumLoss);

                    Console.WriteLine("\nDNI1 total loss " + DNI1totalLoss / DNI1totalLossCount);
                    Console.WriteLine("DNI2 total loss " + DNI2totalLoss / DNI2totalLossCount);
                    Console.WriteLine("DNI3 total loss " + DNI3totalLoss / DNI3totalLossCount);

                    Console.WriteLine("\nDNI1 local loss " + DNI1loss);
                    Console.WriteLine("DNI2 local loss " + DNI2loss);
                    Console.WriteLine("DNI3 local loss " + DNI3loss);

                    //20回バッチを動かしたら精度をテストする
                    if (i % 20 == 0)
                    {
                        Console.WriteLine("\nTesting...");

                        //テストデータからランダムにデータを取得
                        TestDataSet<Real> datasetY = mnistData.Eval.GetRandomDataSet(TEST_DATA_COUNT);

                        //テストを実行
                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        Console.WriteLine("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
