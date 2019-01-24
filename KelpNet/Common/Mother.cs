using System;

namespace KelpNet
{
    //乱数の素
    //C#ではRandomを複数同時にインスタンスすると似たような値しか吐かないため
    //一箇所でまとめて管理しておく必要がある
    public class Mother<T> where T : unmanaged, IComparable<T>
    {
#if DEBUG
        //デバッグ時はシードを固定
        public static Random Dice = new Random(128);
#else
        public static Random Dice = new Random();
#endif
        static bool _flip;
        private static double _beta;
        static Real<T> _boxMuller1;

        // 平均mu, 標準偏差sigmaの正規分布乱数を得る。Box-Muller法による。
        public static Real<T> RandomNormal(double sigma = 1.0, double mu = 0.0)
        {
            Real<T> boxMuller2;

            if (!_flip)
            {
                _boxMuller1 = Math.Sqrt(-2 * Math.Log(Dice.NextDouble()));
                _beta = Dice.NextDouble() * Math.PI * 2.0;
                boxMuller2 = Math.Sin(_beta);
            }
            else
            {
                boxMuller2 = Math.Cos(_beta);
            }

            _flip = !_flip;

            return sigma * (_boxMuller1 * boxMuller2) + mu;
        }

        public static Real<T> Next()
        {
            if (typeof(T) == typeof(float))
            {
                return Dice.NextDouble();
            }
            else if (typeof(T) == typeof(double))
            {
                return Dice.NextDouble();
            }
            else
            {
                throw new Exception();
            }
        }
    }
}
