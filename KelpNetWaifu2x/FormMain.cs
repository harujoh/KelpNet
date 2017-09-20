using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;

namespace KelpNetWaifu2x
{
    /* モデルファイルを https://github.com/nagadomi/waifu2x/tree/master/models/upconv_7/art よりダウンロードしてください*/
    /* サンプルは scale2.0x_model.json にて動作を確認しています*/

    public partial class FormMain : Form
    {
        FunctionStack nn;

        public FormMain()
        {
            InitializeComponent();

            //GPUを初期化
            Weaver.Initialize(ComputeDeviceTypes.Gpu);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog
            {
                Filter = "Jsonファイル(*.json)|*.json|すべてのファイル(*.*)|*.*",
            };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                int layerCounter = 1;

                var json = DynamicJson.Parse(File.ReadAllText(ofd.FileName));

                List<Function> functionList = new List<Function>();

                //Microsoft.CSharp.RuntimeBinder.RuntimeBinderExceptionは無視して下さい
                foreach (var data in json)
                {
                    Real[,,,] weightData = new Real[(int)data["nOutputPlane"], (int)data["nInputPlane"], (int)data["kW"], (int)data["kH"]];

                    for (int i = 0; i < weightData.GetLength(0); i++)
                    {
                        for (int j = 0; j < weightData.GetLength(1); j++)
                        {
                            for (int k = 0; k < weightData.GetLength(2); k++)
                            {
                                for (int l = 0; l < weightData.GetLength(3); l++)
                                {
                                    weightData[i, j, k, l] = data["weight"][i][j][k][l];
                                }
                            }
                        }
                    }

                    //padを行い入力と出力画像のサイズを合わせる
                    functionList.Add(new Convolution2D((int)data["nInputPlane"], (int)data["nOutputPlane"], (int)data["kW"], pad: (int)data["kW"] / 2, initialW: weightData, initialb: (Real[])data["bias"],name: "Convolution2D l" + layerCounter++));
                    functionList.Add(new LeakyReLU(0.1, name: "LeakyReLU l" + layerCounter++));
                }

                nn = new FunctionStack(functionList.ToArray());

                MessageBox.Show("読み込み完了");
            }
        }

        Bitmap _baseImage;
        private void button2_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog
            {
                Filter = "画像ファイル(*.jpg;*.png)|*.jpg;*.png|すべてのファイル(*.*)|*.*"
            };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                this._baseImage = new Bitmap(ofd.FileName);
                this.pictureBox1.Image = new Bitmap(this._baseImage);
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog
            {
                Filter = "pngファイル(*.png)|*.png|すべてのファイル(*.*)|*.*",
                FileName = "result.png"
            };

            if (sfd.ShowDialog() == DialogResult.OK)
            {
                Task.Factory.StartNew(() =>
                {
                    //ネットワークへ入力する前に予め拡大しておく必要がある
                    Bitmap resultImage = new Bitmap(this._baseImage.Width * 2, this._baseImage.Height * 2, PixelFormat.Format24bppRgb);
                    Graphics g = Graphics.FromImage(resultImage);

                    //補間にニアレストネイバーを使用
                    g.InterpolationMode = InterpolationMode.NearestNeighbor;

                    //画像を拡大して描画する
                    g.DrawImage(this._baseImage, 0, 0, this._baseImage.Width * 2, this._baseImage.Height * 2);
                    g.Dispose();

                    NdArray image = NdArrayConverter.Image2NdArray(resultImage);
                    BatchArray resultArray = this.nn.Predict(new BatchArray(image));
                    resultImage = NdArrayConverter.NdArray2Image(resultArray.GetNdArray(0));
                    resultImage.Save(sfd.FileName);
                    this.pictureBox1.Image = new Bitmap(resultImage);
                }
                    ).ContinueWith(_ =>
                    {
                        MessageBox.Show("変換完了");
                    });

                MessageBox.Show("変換処理は開始されました。\n『変換完了』のメッセージが表示されるまで、しばらくお待ち下さい\n※非常に時間がかかります（64x64の画像で三分ほど）");
            }
        }
    }
}
