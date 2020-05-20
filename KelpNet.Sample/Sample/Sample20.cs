using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using KelpNet.CL;
using KelpNet.Tools;

//using Real = System.Double;
using Real = System.Single;

namespace KelpNet.Sample
{
    class Sample20
    {
        private const string DOWNLOAD_URL = "https://github.com/onnx/models/blob/master/vision/classification/vgg/model/";
        private const string VGG16_MODEL_FILE = "vgg16-7.onnx";
        private const string VGG19_MODEL_FILE = "vgg19-7.onnx";
        private const string VGG16_MODEL_FILE_HASH = "a5a3fd73d345152852568509ebff19fc";
        private const string VGG19_MODEL_FILE_HASH = "";
        private const string CLASS_LIST_PATH = "Data/synset_words.txt";

        private static readonly string[] Urls = { DOWNLOAD_URL + VGG16_MODEL_FILE, DOWNLOAD_URL + VGG19_MODEL_FILE };
        private static readonly string[] FileNames = { VGG16_MODEL_FILE, VGG19_MODEL_FILE };
        private static readonly string[] Hashes = { VGG16_MODEL_FILE_HASH, VGG19_MODEL_FILE_HASH };

        public enum VGGModel
        {
            VGG16,
            VGG19
        }

        public static void Run(VGGModel modelType)
        {
            OpenFileDialog ofd = new OpenFileDialog { Filter = "画像ファイル(*.jpg;*.png;*.gif;*.bmp)|*.jpg;*.png;*.gif;*.bmp|すべてのファイル(*.*)|*.*" };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                int vggId = (int)modelType;

                Console.WriteLine("Model Loading.");
                string modelFilePath = InternetFileDownloader.Donwload(Urls[vggId], FileNames[vggId], Hashes[vggId]);

                List<Function<Real>> vggNet = OnnxmodelDataLoader.LoadNetWork<Real>(modelFilePath);
                while (vggNet.Remove(null)) { }

                string[] classList = File.ReadAllLines(CLASS_LIST_PATH);

                //GPUを初期化
                for (int i = 0; i < vggNet.Count - 1; i++)
                {
                    if (vggNet[i] is CPU.Convolution2D<Real> || vggNet[i] is CPU.Linear<Real> || vggNet[i] is CPU.MaxPooling2D<Real>)
                    {
                        vggNet[i] = (Function<Real>)CLConverter.Convert(vggNet[i]);
                    }
                }

                FunctionStack<Real> nn = new FunctionStack<Real>(vggNet.ToArray());

                //層を圧縮
                nn.Compress();

                Console.WriteLine("Model Loading done.");

                do
                {
                    //ネットワークへ入力する前に解像度を 224px x 224px x 3ch にしておく
                    Bitmap baseImage = new Bitmap(ofd.FileName);
                    Bitmap resultImage = new Bitmap(224, 224, PixelFormat.Format24bppRgb);
                    Graphics g = Graphics.FromImage(resultImage);
                    g.DrawImage(baseImage, 0, 0, 224, 224);
                    g.Dispose();

                    Real[] bias = new Real[] { -123.68f, -116.779f, -103.939f }; //補正値のチャンネル順は入力画像に従う(標準的なBitmapならRGB)
                    NdArray<Real> imageArray = BitmapConverter.Image2NdArray<Real>(resultImage, false, true, bias);

                    Console.WriteLine("Start predict.");
                    Stopwatch sw = Stopwatch.StartNew();
                    NdArray<Real> result = nn.Predict(imageArray)[0];
                    sw.Stop();

                    Console.WriteLine("Result Time : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                    int maxIndex = Array.IndexOf(result.Data, result.Data.Max());

                    Console.WriteLine("[" + result.Data[maxIndex] + "] : " + classList[maxIndex]);

                } while (ofd.ShowDialog() == DialogResult.OK);
            }
        }

    }
}
