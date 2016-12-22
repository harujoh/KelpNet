using System;
using KelpNetTester.Tests;

namespace KelpNetTester
{
    class Program
    {
        //実行したいテストのコメントを外して下さい
        static void Main(string[] args)
        {
            //MLPによるXORの学習
            Test1.Run();

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

            //Deconvolution2Dのテスト
            //Test13.Run();

            //Linearの分割実行
            //TestX.Run();

            Console.WriteLine("Test Done...");
            Console.Read();
        }
    }
}
