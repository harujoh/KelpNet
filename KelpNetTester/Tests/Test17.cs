using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using CaffemodelLoader;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Poolings;
using TestDataManager;

namespace KelpNetTester.Tests
{
    //ResNetを読み込んで実行する
    class Test17
    {
        private const string DOWNLOAD_URL_MEAN = "https://onedrive.live.com/download?cid=4006CBB8476FF777&resid=4006CBB8476FF777%2117894&authkey=%21AAFW2%2DFVoxeVRck";
        private const string DOWNLOAD_URL_50   = "https://onedrive.live.com/download?cid=4006CBB8476FF777&resid=4006CBB8476FF777%2117895&authkey=%21AAFW2%2DFVoxeVRck";
        private const string DOWNLOAD_URL_101  = "https://onedrive.live.com/download?cid=4006CBB8476FF777&resid=4006CBB8476FF777%2117896&authkey=%21AAFW2%2DFVoxeVRck";
        private const string DOWNLOAD_URL_152  = "https://onedrive.live.com/download?cid=4006CBB8476FF777&resid=4006CBB8476FF777%2117897&authkey=%21AAFW2%2DFVoxeVRck";

        private const string MODEL_FILE_MEAN = "ResNet_mean.binaryproto";
        private const string MODEL_FILE_50 = "ResNet-50-model.caffemodel";
        private const string MODEL_FILE_101 = "ResNet-101-model.caffemodel";
        private const string MODEL_FILE_152 = "ResNet-152-model.caffemodel";

        private static readonly string[] Urls = { DOWNLOAD_URL_50, DOWNLOAD_URL_101, DOWNLOAD_URL_152 };
        private static readonly string[] FileNames = { MODEL_FILE_50, MODEL_FILE_101, MODEL_FILE_152 };

        private const string CLASS_LIST_PATH = "Data/synset_words.txt";

        public enum ResnetModel
        {
            ResNet50,
            ResNet101,
            ResNet152,
        }

        public static void Run(ResnetModel modelType)
        {
            OpenFileDialog ofd = new OpenFileDialog { Filter = "画像ファイル(*.jpg;*.png;*.gif;*.bmp)|*.jpg;*.png;*.gif;*.bmp|すべてのファイル(*.*)|*.*" };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                int resnetId = (int)modelType;

                Console.WriteLine("Mean Loading.");
                string meanFilePath = InternetFileDownloader.Donwload(DOWNLOAD_URL_MEAN, MODEL_FILE_MEAN);
                NdArray mean = CaffemodelDataLoader.ReadBinary(meanFilePath);

                Console.WriteLine("Model Loading.");
                string modelFilePath = InternetFileDownloader.Donwload(Urls[resnetId], FileNames[resnetId]);
                FunctionDictionary nn = CaffemodelDataLoader.LoadNetWork(modelFilePath);
                string[] classList = File.ReadAllLines(CLASS_LIST_PATH);

                //GPUを初期化
                foreach (KeyValuePair<string, FunctionSet> resNetFunctionBlock in nn.FunctionBlocks)
                {
                    foreach (Function function in resNetFunctionBlock.Value.Functions)
                    {
                        if (function is Convolution2D || function is Linear || function is MaxPooling)
                        {
                            ((IParallelizable)function).SetGpuEnable(true);
                        }
                    }

                    //ブロック単位で層の圧縮を実行
                    resNetFunctionBlock.Value.Compress();
                }

                Console.WriteLine("Model Loading done.");

                do
                {
                    //ネットワークへ入力する前に解像度を 224px x 224px x 3ch にしておく
                    Bitmap baseImage = new Bitmap(ofd.FileName);
                    Bitmap resultImage = new Bitmap(224, 224, PixelFormat.Format24bppRgb);
                    Graphics g = Graphics.FromImage(resultImage);
                    g.InterpolationMode = InterpolationMode.Bilinear;
                    g.DrawImage(baseImage, 0, 0, 224, 224);
                    g.Dispose();

                    NdArray imageArray = NdArrayConverter.Image2NdArray(resultImage, false, true);
                    imageArray -= mean;
                    imageArray.ParentFunc = null;

                    Console.WriteLine("Start predict.");
                    Stopwatch sw = Stopwatch.StartNew();
                    NdArray result = nn.Predict(imageArray)[0];
                    sw.Stop();

                    Console.WriteLine("Result Time : " +
                                      (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") +
                                      "μｓ");

                    int maxIndex = Array.IndexOf(result.Data, result.Data.Max());
                    Console.WriteLine("[" + result.Data[maxIndex] + "] : " + classList[maxIndex]);
                } while (ofd.ShowDialog() == DialogResult.OK);
            }
        }
    }
}
