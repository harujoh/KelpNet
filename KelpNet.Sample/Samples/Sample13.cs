using System;

namespace KelpNet.Sample.Samples
{
    //ある学習済みフィルタで出力された画像を元に、そのフィルタと同等のフィルタを獲得する
    //コンソール版
    //移植元 : http://qiita.com/samacoba/items/958c02f455ca5f3a475d
    class Sample13<T> where T : unmanaged, IComparable<T>
    {
        public static void Run()
        {
            //目標とするフィルタを作成（実践であればココは不明な値となる）
            Deconvolution2D<T> decon_core = new Deconvolution2D<T>(1, 1, 15, 1, trim:7)
            {
                Weight = { Data = MakeOneCore() }
            };

            Deconvolution2D<T> model = new Deconvolution2D<T>(1, 1, 15, 1, trim:7);

            SGD<T> optimizer = new SGD<T>(learningRate: 0.00005f); //大きいと発散する
            model.SetOptimizer(optimizer);
            MeanSquaredError<T> meanSquaredError = new MeanSquaredError<T>();

            //移植元では同じ教育画像で教育しているが、より実践に近い学習に変更
            for (int i = 0; i < 11; i++)
            {
                //ランダムに点が打たれた画像を生成
                NdArray<T> img_p = GetRandomImage();

                //目標とするフィルタで学習用の画像を出力
                NdArray<T>[] img_core = decon_core.Forward(img_p);

                //未学習のフィルタで画像を出力
                NdArray<T>[] img_y = model.Forward(img_p);

                Real<T> loss = meanSquaredError.Evaluate(img_y, img_core);

                model.Backward(img_y);
                model.Update();

                Console.WriteLine("epoch" + i + " : " + loss);
            }
        }

        static NdArray<T> GetRandomImage(int N = 1, int img_w = 128, int img_h = 128)
        {
            // ランダムに0.1％の点を作る
            Real<T>[] img_p = new Real<T>[N * img_w * img_h];

            for (int i = 0; i < img_p.Length; i++)
            {
                img_p[i] = Mother<T>.Dice.Next(0, 1000);
                img_p[i] = img_p[i] > 999 ? 0 : 1;
            }

            return new NdArray<T>(img_p, new[] { N, img_h, img_w }, 1);
        }

        //１つの球状の模様を作成（ガウスですが）
        static Real<T>[] MakeOneCore()
        {
            int max_xy = 15;
            double sig = 5;
            double sig2 = sig * sig;
            double c_xy = 7;
            Real<T>[] core = new Real<T>[max_xy * max_xy];

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
    }
}
