using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using CaffemodelLoader;
using KelpNet.Common;
using KelpNet.Common.Tools;
using KelpNet.Functions;
using TestDataManager;

namespace KelpNetTester.Tests
{
    //CaffeモデルのVGG16を読み込んで画像分類をさせるテスト
    class Test15
    {
        const string DOWNLOAD_URL = "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/";
        const string MODEL_FILE = "VGG_ILSVRC_16_layers.caffemodel";
        private const string CLASS_LIST_PATH = "data/synset_words.txt";

        public static void Run()
        {
            OpenFileDialog ofd = new OpenFileDialog { Filter = "画像ファイル(*.jpg;*.png)|*.jpg;*.png|すべてのファイル(*.*)|*.*" };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                var baseImage = new Bitmap(ofd.FileName);

                var modelFilePath = InternetFileDownloader.Donwload(DOWNLOAD_URL + MODEL_FILE, MODEL_FILE);
                var vgg16Net = CaffemodelDataLoader.ModelLoad(modelFilePath);
                var classList = File.ReadAllLines(CLASS_LIST_PATH);

                var nn = new FunctionStack(vgg16Net.ToArray());

                //ネットワークへ入力する前に解像度を 224 x 224 にしておく必要がある
                Bitmap resultImage = new Bitmap(224, 224, PixelFormat.Format24bppRgb);
                Graphics g = Graphics.FromImage(resultImage);

                //補間にニアレストネイバーを使用
                g.InterpolationMode = InterpolationMode.NearestNeighbor;

                //画像を拡大して描画する
                g.DrawImage(baseImage, 0, 0, 224, 224);
                g.Dispose();

                BatchArray image = new BatchArray(NdArrayConverter.Image2NdArray(resultImage));

                Stopwatch sw = Stopwatch.StartNew();
                var result = nn.Predict(image);
                sw.Stop();
                Console.WriteLine("Result Time : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                var maxIndex = result.Data.Select((val, idx) => new { V = val, I = idx }).Aggregate((max, working) => max.V > working.V ? max : working).I;

                Console.WriteLine("[" + result.Data[maxIndex] + "] : " + classList[maxIndex]);
            }
        }
    }
}
