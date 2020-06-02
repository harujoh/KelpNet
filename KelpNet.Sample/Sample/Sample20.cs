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
        private const string DOWNLOAD_URL = "https://github.com/onnx/models/raw/master/vision/classification/vgg/model/";
        private const string VGG16_MODEL_FILE = "vgg16-7.onnx";
        private const string VGG19_MODEL_FILE = "vgg19-7.onnx";
        private const string VGG16BN_MODEL_FILE = "vgg16-bn-7.onnx";
        private const string VGG19BN_MODEL_FILE = "vgg19-bn-7.onnx";
        private const string VGG16_MODEL_FILE_HASH = "a5a3fd73d345152852568509ebff19fc";
        private const string VGG19_MODEL_FILE_HASH = "3ddcbebbe6c937d504b09c48bf0ba371";
        private const string VGG16BN_MODEL_FILE_HASH = "fe75b896f4eaf071ff8460953ab7f4c6";
        private const string VGG19BN_MODEL_FILE_HASH = "e00e03427016cf30e1c3b39b4e0ca0d1";
        private const string CLASS_LIST_PATH = "Data/synset_words.txt";

        private static readonly string[] Urls = { DOWNLOAD_URL + VGG16_MODEL_FILE, DOWNLOAD_URL + VGG19_MODEL_FILE, DOWNLOAD_URL + VGG16BN_MODEL_FILE, DOWNLOAD_URL + VGG19BN_MODEL_FILE };
        private static readonly string[] FileNames = { VGG16_MODEL_FILE, VGG19_MODEL_FILE, VGG16BN_MODEL_FILE, VGG19BN_MODEL_FILE };
        private static readonly string[] Hashes = { VGG16_MODEL_FILE_HASH, VGG19_MODEL_FILE_HASH, VGG16BN_MODEL_FILE_HASH, VGG19BN_MODEL_FILE_HASH };

        public enum VGGModel
        {
            VGG16,
            VGG19,
            VGG16BN,
            VGG19NM
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

                    Real[] mean = new Real[]{ 0.485f, 0.456f, 0.406f };
                    Real[] std = new Real[] { 0.229f, 0.224f, 0.225f };

                    NdArray<Real> imageArray = BitmapConverter.Image2NdArray<Real>(resultImage);
                    int dataSize = imageArray.Shape[1] * imageArray.Shape[2];
                    for (int ch = 0; ch < imageArray.Shape[0]; ch++)
                    {
                        for (int i = 0; i < dataSize; i++)
                        {
                            imageArray.Data[ch * dataSize + i] = (imageArray.Data[ch * dataSize + i] - mean[ch]) / std[ch];
                        }
                    }

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
