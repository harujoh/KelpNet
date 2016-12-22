using System;
using System.Drawing;
using System.Windows.Forms;
using KelpNet.Common;
using KelpNet.Common.Tools;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    public partial class Test13 : Form
    {
        Deconvolution2D model;
        private Deconvolution2D decon_core;
        private SGD optimizer;
        MeanSquaredError meanSquaredError = new MeanSquaredError();
        private int counter = 0;

        public Test13()
        {
            this.InitializeComponent();

            ClientSize = new Size(128*4, 128*4);

            //目標とするフィルタを作成（実践であればココは不明な値となる）
            this.decon_core = new Deconvolution2D(1, 1, 15, 1, 7)
            {
                W = { Data = MakeOneCore() }
            };

            this.model = new Deconvolution2D(1, 1, 15, 1, 7);

            this.optimizer = new SGD(learningRate: 0.00005); //大きいと発散する
            this.model.SetOptimizer(this.optimizer);
        }

        static NdArray getRandomImage(int N = 1, int img_w = 128, int img_h = 128)
        {
            // ランダムに0.1％の点を作る
            double[] img_p = new double[N * img_w * img_h];

            for (int i = 0; i < img_p.Length; i++)
            {
                img_p[i] = Mother.Dice.Next(0, 1000);
                if(img_p[i] != 990) img_p[i] = img_p[i] > 990 ? 1 : 0;
            }

            return new NdArray(img_p, new[] { N, img_h, img_w });
        }

        //１つの球状の模様を作成（ガウスですが）
        static double[] MakeOneCore()
        {
            int max_xy = 15;
            double sig = 5.0;
            double sig2 = sig * sig;
            double c_xy = 7;
            double[] core = new double[max_xy * max_xy];

            for (int px = 0; px < max_xy; px++)
            {
                for (int py = 0; py < max_xy; py++)
                {
                    double r2 = (px - c_xy) * (px - c_xy) + (py - c_xy) * (py - c_xy);
                    core[py * max_xy + px] = Math.Exp(-r2 / sig2) * 1;
                }
            }

            return core;
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            //移植元では同じ教育画像だけで教育しているが、こちらの方が実践に近いと思う
            if (this.counter < 31)
            {
                //ランダムに点が打たれた画像を生成
                NdArray img_p = getRandomImage();

                //目標とするフィルタで学習用の画像を出力
                NdArray img_core = this.decon_core.Forward(img_p);

                this.model.ClearGrads();

                //未学習のフィルタで画像を出力
                NdArray img_y = this.model.Forward(img_p);

                this.BackgroundImage = NdArrayConverter.NdArray2Image(img_y);

                double loss;
                NdArray gy = this.meanSquaredError.Evaluate(img_y, img_core, out loss);

                this.model.Backward(gy);

                this.model.Update();

                this.Text = "[epoch" + this.counter + "] Loss : " + loss.ToString("f4");

                this.counter++;
            }
            else
            {
                this.timer1.Enabled = false;
            }

        }
    }
}
