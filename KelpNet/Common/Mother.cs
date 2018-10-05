using System;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
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
        static bool _flip;
        private static double _beta;
        static Real _boxMuller1;

        // 平均mu, 標準偏差sigmaの正規分布乱数を得る。Box-Muller法による。
        public static Real RandomNormal(Real sigma = 1.0f, Real mu = 0.0f)
        {
            Real boxMuller2;

            if (!_flip)
            {
                _boxMuller1 = (Real)Math.Sqrt(-2 * Math.Log(Dice.NextDouble()));
                _beta = Dice.NextDouble() * Math.PI * 2.0;
                boxMuller2 = (Real)Math.Sin(_beta);
            }
            else
            {
                boxMuller2 = (Real)Math.Cos(_beta);
            }

            _flip = !_flip;

            return sigma * (_boxMuller1 * boxMuller2) + mu;
        }
    }
}
