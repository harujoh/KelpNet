using System;

namespace KelpNet
{
    //乱数の素
    //C#ではRandomを複数同時にインスタンスすると似たような値しか吐かないため
    //一箇所でまとめて管理しておく必要がある
    public class Mother
    {
#if DEBUG
        //デバッグ時はシードを固定
        public static Random Dice = new Random(128);
#else
        public static Random Dice = new Random();
#endif
        static double Alpha, Beta, BoxMuller1, BoxMuller2;
        static bool Flip;
        public static double Mu = 0.0;
        public static double Sigma = 1.0;

        // 平均mu, 標準偏差sigmaの正規分布乱数を得る。Box-Muller法による。
        public static double RandomNormal()
        {
            if (!Flip)
            {
                Alpha = Dice.NextDouble();
                Beta = Dice.NextDouble() * Math.PI * 2;
                BoxMuller1 = Math.Sqrt(-2 * Math.Log(Alpha));
                BoxMuller2 = Math.Sin(Beta);
            }
            else
            {
                BoxMuller2 = Math.Cos(Beta);
            }

            Flip = !Flip;

            return Sigma * (BoxMuller1 * BoxMuller2) + Mu;
        }
    }
}
