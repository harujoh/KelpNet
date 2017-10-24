using System;
using KelpNet.Common;
using KelpNetTester.Benchmarker;
using KelpNetTester.Tests;

namespace KelpNetTester
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
            //Test1.Run();

            //MLPによるXORの学習【回帰版】
            //Test2.Run();

            //MLPによるSin関数の学習
            //Test3.Run();

            //MLPによるMNIST（手書き文字）の学習
            //Test4.Run();

            //エクセルCNNの再現
            //Test5.Run();

            //5層CNNによるMNISTの学習
            //Test6.Run();

            //BatchNormを使った15層MLPによるMNISTの学習
            //Test7.Run();

            //LSTMによるSin関数の学習
            //Test8.Run();

            //SimpleなRNNによるRNNLM
            //Test9.Run();

            //LSTMによるRNNLM
            //Test10.Run();

            //Decoupled Neural Interfaces using Synthetic GradientsによるMNISTの学習
            //Test11.Run();

            //Test11のDNIをcDNIとした
            //Test12.Run();

            //Deconvolution2Dのテスト(Winform)
            //new Test13WinForm().ShowDialog();

            //Test6を連結して実行
            //Test14.Run();

            //CaffeモデルのVGG16を読み込んで画像分類をさせるテスト
            //Test15.Run();

            //ChainerモデルのTest5と同じ内容を読み込んで実行
            //Test16.Run();

            //CaffeモデルのRESNET-152を読み込んで画像分類をさせるテスト
            Test17.Run();

            //Linearの分割実行
            //TestX.Run();

            //ベンチマーク
            SingleBenchmark.Run();

            Console.WriteLine("Test Done...");
            Console.Read();
        }
    }
}
