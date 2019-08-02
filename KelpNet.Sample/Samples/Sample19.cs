using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using KelpNet.Tools;
using KelpNet.CL;

namespace KelpNet.Sample
{
    //CaffeモデルのAlexNetを読み込んで画像分類をさせるテスト
    class Sample19
    {
        private const string DOWNLOAD_URL = "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel";
        private const string MODEL_FILE = "bvlc_alexnet.caffemodel";
        private const string MODEL_FILE_HASH = "29eb495b11613825c1900382f5286963";
        private const string CLASS_LIST_PATH = "Data/synset_words.txt";

        public static void Run()
        {
            OpenFileDialog ofd = new OpenFileDialog
            {
                Filter = "画像ファイル(*.jpg;*.png;*.gif;*.bmp)|*.jpg;*.png;*.gif;*.bmp|すべてのファイル(*.*)|*.*"
            };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                Console.WriteLine("Model Loading.");
                string modelFilePath = InternetFileDownloader.Donwload(DOWNLOAD_URL, MODEL_FILE, MODEL_FILE_HASH);
                List<Function> alexNet = CaffemodelDataLoader.ModelLoad(modelFilePath);
                string[] classList = File.ReadAllLines(CLASS_LIST_PATH);

                //GPUを初期化
                for (int i = 0; i < alexNet.Count - 1; i++)
                {
                    if (alexNet[i] is CPU.Convolution2D || alexNet[i] is CPU.Linear || alexNet[i] is CPU.MaxPooling2D)
                    {
                        alexNet[i] = (Function)CLConverter.Convert(alexNet[i]);
                    }
                }

                FunctionStack nn = new FunctionStack(alexNet.ToArray());

                //層を圧縮
                nn.Compress();

                Console.WriteLine("Model Loading done.");

                do
                {
                    //ネットワークへ入力する前に解像度を 224px x 224px x 3ch にしておく
                    Bitmap baseImage = new Bitmap(ofd.FileName);
                    Bitmap resultImage = new Bitmap(227, 227, PixelFormat.Format24bppRgb);
                    Graphics g = Graphics.FromImage(resultImage);
                    g.DrawImage(baseImage, 0, 0, 227, 227);
                    g.Dispose();

                    Real[] bias = {-123.68, -116.779, -103.939}; //補正値のチャンネル順は入力画像に従う
                    NdArray imageArray = BitmapConverter.Image2NdArray(resultImage, false, true, bias);

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
