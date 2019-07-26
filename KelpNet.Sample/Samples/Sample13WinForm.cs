using System;
using System.Drawing;
using System.Windows.Forms;
using KelpNet.Tools;

namespace KelpNet.Sample.Samples
{
    public partial class Sample13WinForm<T> : Form where T : unmanaged, IComparable<T>
    {
        Deconvolution2D<T> model;
        private Deconvolution2D<T> decon_core;
        private SGD<T> optimizer;
        MeanSquaredError<T> meanSquaredError = new MeanSquaredError<T>();
        private int counter = 0;

        public Sample13WinForm()
        {
            this.InitializeComponent();

            ClientSize = new Size(128 * 4, 128 * 4);

            //目標とするフィルタを作成（実践であればココは不明な値となる）
            this.decon_core = new Deconvolution2D<T>(1, 1, 15, 1, trim:7)
            {
                Weight = { Data = MakeOneCore() }
            };

            this.model = new Deconvolution2D<T>(1, 1, 15, 1, trim:7);

            this.optimizer = new SGD<T>(learningRate: 0.01f); //大きいと発散する
            this.model.SetOptimizer(this.optimizer);
        }

        static NdArray<T> GetRandomImage(int N = 1, int img_w = 128, int img_h = 128)
        {
            // ランダムに0.1％の点を作る
            RealArray<T> img_p = new T[N * img_w * img_h];

            for (int i = 0; i < img_p.Length; i++)
            {
                img_p[i] = Mother<T>.Dice.Next(0, 10000);
                img_p[i] = img_p[i] < 10 ? 255 : 0;
            }

            return new NdArray<T>(img_p, new[] { N, img_h, img_w }, 1);
        }

        //１つの球状の模様を作成（ガウスですが）
        static RealArray<T> MakeOneCore()
        {
            int max_xy = 15;
            double sig = 5.0;
            double sig2 = sig * sig;
            double c_xy = 7.0;
            RealArray<T> core = new T[max_xy * max_xy];

            for (int px = 0; px < max_xy; px++)
            {
                for (int py = 0; py < max_xy; py++)
                {
                    double r2 = (px - c_xy) * (px - c_xy) + (py - c_xy) * (py - c_xy);
                    core[py * max_xy + px] = (Math.Exp(-r2 / sig2) * 1.0);
                }
            }

            return core;
        }

        private void Timer1_Tick(object sender, EventArgs e)
        {
            //移植元では同じ教育画像で教育しているが、より実践に近い学習に変更
            if (this.counter < 11)
            {
                //ランダムに点が打たれた画像を生成
                NdArray<T> img_p = GetRandomImage();

                //目標とするフィルタで学習用の画像を出力
                NdArray<T>[] img_core = this.decon_core.Forward(img_p);

                //未学習のフィルタで画像を出力
                NdArray<T>[] img_y = this.model.Forward(img_p);

                this.BackgroundImage = NdArrayConverter<T>.NdArray2Image(img_y[0]);

                Real<T> loss = this.meanSquaredError.Evaluate(img_y, img_core);

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
