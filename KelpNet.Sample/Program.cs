using System;
using KelpNet.CL;
using KelpNet.CL.Common;

namespace KelpNet.Sample
{
    //実行したいテストのコメントを外して下さい
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            //全て.Net Framework上で実行したい場合はこちらをコメントアウト
            OpenCL.Initialize();
            //OpenCL.Initialize(ComputeDeviceTypes.Cpu); //複数の種類のデバイスがある場合はデバイスタイプを指定
            //OpenCL.Initialize(ComputeDeviceTypes.Gpu, 1); //同じ種類のGPUやCPUが複数ある場合は添字を使用

            Console.WriteLine("Running on " + OpenCL.InfoString);

            //MLPによるXORの学習
            //Sample01.Run();

            //MLPによるXORの学習【回帰版】
            //Sample02.Run();

            //MLPによるSin関数の学習
            //Sample03.Run();

            //MLPによるMNIST（手書き文字）の学習
            //Sample04.Run();

            //エクセルCNNの再現
            //Sample05.Run();

            //5層CNNによるMNISTの学習
            //Sample06.Run();

            //BatchNormを使った15層MLPによるMNISTの学習
            //Sample07.Run();

            //LSTMによるSin関数の学習
            //Sample08.Run();

            //SimpleなRNNによるRNNLM
            //Sample09.Run();

            //LSTMによるRNNLM
            //Sample10.Run();

            //Decoupled Neural Interfaces using Synthetic GradientsによるMNISTの学習
            //Sample11.Run();

            //Test11のDNIをcDNIとした
            //Sample12.Run();

            //Deconvolution2Dのテスト(Winform)
            //new Sample13WinForm().ShowDialog();

            //Test6を連結してFashion-MNISTを学習
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

            Console.WriteLine("Done...");
            Console.Read();
        }
    }
}
