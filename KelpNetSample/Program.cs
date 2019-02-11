using System;
using KelpNet;
using KelpNetSample.Benchmarker;

namespace KelpNetSample 
{
    //実行したいテストのコメントを外して下さい
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            //全て.Net Framework上で実行したい場合はこちらをコメントアウト
            Weaver.Initialize(ComputeDeviceTypes.Gpu);
            //Weaver.Initialize(ComputeDeviceTypes.Cpu, 1); //複数デバイスがある場合は添字が必要

            //MLPによるXORの学習
            //Sample1.Run();

            //MLPによるXORの学習【回帰版】
            //Sample2.Run();

            //MLPによるSin関数の学習
            //Sample3.Run();

            //MLPによるMNIST（手書き文字）の学習
            //Sample4.Run();

            //エクセルCNNの再現
            //Sample5.Run();

            //5層CNNによるMNISTの学習
            //Sample6.Run();

            //BatchNormを使った15層MLPによるMNISTの学習
            //Sample7.Run();

            //LSTMによるSin関数の学習
            //Sample8.Run();

            //SimpleなRNNによるRNNLM
            //Sample9.Run();

            //LSTMによるRNNLM
            //Sample10.Run();

            //Decoupled Neural Interfaces using Synthetic GradientsによるMNISTの学習
            //Sample11.Run();

            //Test11のDNIをcDNIとした
            //Sample12.Run();

            //Deconvolution2Dのテスト(Winform)
            //new Sample13WinForm().ShowDialog();

            //Test6を連結して実行
            //Sample14.Run();

            //CaffeモデルのVGGを読み込んで画像分類をさせるテスト
            //Sample15.Run(Sample15.VGGModel.VGG16); //VGG16またはVGG19を選択してください

            //ChainerモデルのTest5と同じ内容を読み込んで実行
            //Sample16.Run();

            //CaffeモデルのRESNETを読み込んで画像分類をさせるテスト
            //Sample17.Run(Sample17.ResnetModel.ResNet50);  //任意のResnetモデルを選択してください

            //CIFAR-10を5層CNNを使って学習する
            //Sample18.Run(isCifar100:false, isFineLabel:false);

            //CaffeモデルのAlexNetを読み込んで画像分類をさせるテスト
            //Sample19.Run();

            //Linearの分割実行
            //SampleX.Run();

            //ベンチマーク
            SingleBenchmark.Run();

            Console.WriteLine("Test Done...");
            Console.Read();
        }
    }
}
