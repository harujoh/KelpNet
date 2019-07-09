using System;
using System.Drawing;
using System.Windows.Forms;
using KelpNet.CL;
using KelpNet.Tools;

namespace KelpNet.Sample
{
    public partial class Sample13WinForm : Form
    {
        Deconvolution2D model;
        private Deconvolution2D decon_core;
        private SGD optimizer;
        MeanSquaredError meanSquaredError = new MeanSquaredError();
        private int counter = 0;

        public Sample13WinForm()
        {
            this.InitializeComponent();

            ClientSize = new Size(128 * 4, 128 * 4);

            //目標とするフィルタを作成（実践であればココは不明な値となる）
            this.decon_core = new Deconvolution2D(1, 1, 15, 1, 7, gpuEnable: true)
            {
                Weight = { Data = MakeOneCore() }
            };

            this.model = new Deconvolution2D(1, 1, 15, 1, 7, gpuEnable: true);

            this.optimizer = new SGD(learningRate: 0.01);
            this.model.SetOptimizer(this.optimizer);
        }

        static NdArray getRandomImage(int N = 1, int img_w = 128, int img_h = 128)
        {
            // ランダムに0.1％の点を作る
            Real[] img_p = new Real[N * img_w * img_h];

            for (int i = 0; i < img_p.Length; i++)
            {
                img_p[i] = Mother.Dice.Next(0, 10000);
                img_p[i] = img_p[i] < 10 ? 255 : 0;
            }

            return new NdArray(img_p, new[] { N, img_h, img_w }, 1);
        }

        //１つの球状の模様を作成（ガウスですが）
        static Real[] MakeOneCore()
        {
            int max_xy = 15;
            Real sig = 5;
            Real sig2 = sig * sig;
            Real c_xy = 7;
            Real[] core = new Real[max_xy * max_xy];

            for (int px = 0; px < max_xy; px++)
            {
                for (int py = 0; py < max_xy; py++)
                {
                    Real r2 = (px - c_xy) * (px - c_xy) + (py - c_xy) * (py - c_xy);
                    core[py * max_xy + px] = Math.Exp(-r2 / sig2) * 1;
                }
            }

            return core;
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            //移植元では同じ教育画像で教育しているが、より実践に近い学習に変更
            if (this.counter < 11)
            {
                //ランダムに点が打たれた画像を生成
                NdArray img_p = getRandomImage();

                //目標とするフィルタで学習用の画像を出力
                NdArray[] img_core = this.decon_core.Forward(img_p);

                //未学習のフィルタで画像を出力
                NdArray[] img_y = this.model.Forward(img_p);

                //img_yを暗黙的にNdArrayとして使用
                this.BackgroundImage = BitmapConverter.NdArray2Image(img_y[0])[0];

                Real loss = this.meanSquaredError.Evaluate(img_y, img_core);

                this.model.Backward(img_y);
                this.model.Update();

                this.Text = "[epoch" + this.counter + "] Loss : " + string.Format("{0:F4}", loss);

                this.counter++;
            }
            else
            {
                this.timer1.Enabled = false;
            }

        }
    }
}
