using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using KelpNet.Tools;
using KelpNet.Tools.DataImporter.Models.Caffe;

namespace KelpNet.Sample.Samples
{
    //CaffeモデルのVGGを読み込んで画像分類をさせるテスト
    class Sample15<T> where T : unmanaged, IComparable<T>
    {
        private const string DOWNLOAD_URL = "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/";
        private const string VGG16_MODEL_FILE = "VGG_ILSVRC_16_layers.caffemodel";
        private const string VGG19_MODEL_FILE = "VGG_ILSVRC_19_layers.caffemodel";
        private const string VGG16_MODEL_FILE_HASH = "441315b0ff6932dbfde97731be7ca852";
        private const string VGG19_MODEL_FILE_HASH = "b5c644beabd7cf06bdd9065cfd674c97";
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
                List<Function<T>> vggNet = CaffemodelDataLoader<T>.ModelLoad(modelFilePath);

                string[] classList = File.ReadAllLines(CLASS_LIST_PATH);

                //GPUを初期化
                //for (int i = 0; i < vggNet.Count - 1; i++)
                //{
                //    if (vggNet[i] is Convolution2D || vggNet[i] is Linear || vggNet[i] is MaxPooling)
                //    {
                //        ((IParallelizable) vggNet[i]).SetGpuEnable(true);
                //    }
                //}

                FunctionStack<T> nn = new FunctionStack<T>(vggNet.ToArray());

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

                    RealArray<T> bias = new Real<T>[] { -123.68f, -116.779f, -103.939f }; //補正値のチャンネル順は入力画像に従う
                    NdArray<T> imageArray = NdArrayConverter<T>.Image2NdArray(resultImage, false, true, bias);

                    Console.WriteLine("Start predict.");
                    Stopwatch sw = Stopwatch.StartNew();
                    NdArray<T> result = nn.Predict(imageArray)[0];
                    sw.Stop();

                    Console.WriteLine("Result Time : " +
                                      (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") +
                                      "μｓ");

                    //int maxIndex = Array.IndexOf(result.Data, result.Data.Max());
                    int maxIndex = result.Data.MaxIndex();
                    Console.WriteLine("[" + result.Data[maxIndex] + "] : " + classList[maxIndex]);
                } while (ofd.ShowDialog() == DialogResult.OK);
            }
        }
    }
}
