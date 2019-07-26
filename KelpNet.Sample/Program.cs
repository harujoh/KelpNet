using System;
using KelpNet.Sample.Samples;

using RealType = System.Single;
//using RealType = System.Double;

namespace KelpNet.Sample
{
    //実行したいテストのコメントを外して下さい
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            //全て.Net Framework上で実行したい場合はこちらをコメントアウト
            //Weaver.Initialize(ComputeDeviceTypes.Gpu);
            //Weaver.Initialize(ComputeDeviceTypes.Cpu, 1); //複数デバイスがある場合は添字が必要

            //MLPによるXORの学習
            //Sample01<RealType>.Run();

            //MLPによるXORの学習【回帰版】
            //Sample02<RealType>.Run();

            //MLPによるSin関数の学習
            //Sample03<RealType>.Run();

            //MLPによるMNIST（手書き文字）の学習
            //Sample04<RealType>.Run();

            //5層CNNによるMNISTの学習
            Sample06<RealType>.Run();

            //BatchNormを使った15層MLPによるMNISTの学習
            //Sample07<RealType>.Run();

            //LSTMによるSin関数の学習
            //Sample08<RealType>.Run();

            //SimpleなRNNによるRNNLM
            //Sample09<RealType>.Run();

            //LSTMによるRNNLM
            //Sample10<RealType>.Run();

            //Decoupled Neural Interfaces using Synthetic GradientsによるMNISTの学習
            //Sample11<RealType>.Run();

            //Test11のDNIをcDNIとした
            //Sample12<RealType>.Run();

            //Deconvolution2Dのテスト(Winform)
            //new Sample13WinForm<RealType>().ShowDialog();

            //Test6を連結して実行
            //Sample14<RealType>.Run();

            //CaffeモデルのVGGを読み込んで画像分類をさせるテスト
            //Sample15<RealType>.Run(Sample15.VGGModel.VGG16); //VGG16またはVGG19を選択してください

            //ChainerモデルのTest5と同じ内容を読み込んで実行
            //Sample16<RealType>.Run();

            //CaffeモデルのRESNETを読み込んで画像分類をさせるテスト
            //Sample17<RealType>.Run(Sample17.ResnetModel.ResNet50);  //任意のResnetモデルを選択してください

            //CIFAR-10を5層CNNを使って学習する
            //Sample18<RealType>.Run(isCifar100:false, isFineLabel:false);

            //CaffeモデルのAlexNetを読み込んで画像分類をさせるテスト
            //Test19<RealType>.Run();

            Console.WriteLine("Done...");
            Console.Read();
        }
    }
}
