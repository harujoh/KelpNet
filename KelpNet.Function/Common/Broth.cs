using System;

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
    //外部公開を必要としないinternalなクラスなので、それぞれの型で名称を変えずに済む
    internal class Broth
    {
        static bool _flip;
        static double _beta;
        static double _boxMuller1;

        // 平均mu, 標準偏差sigmaの正規分布乱数を得る。Box-Muller法による。
        public static Real RandomNormal(Real sigma = 1, Real mu = 0)
        {
            double boxMuller2;

            if (!_flip)
            {
                _beta = Mother.Dice.NextDouble() * Math.PI * 2;
                _boxMuller1 = Math.Sqrt(-2 * Math.Log(Mother.Dice.NextDouble()));
                boxMuller2 = Math.Sin(_beta);
            }
            else
            {
                boxMuller2 = Math.Cos(_beta);
            }

            _flip = !_flip;

            return sigma * (Real)(_boxMuller1 * boxMuller2) + mu;
        }

        public static Real Random()
        {
            return (Real)Mother.Dice.NextDouble();
        }

        public static Real Next()
        {
            return (Real)Mother.Dice.Next();
        }

        public static Real Next(int maxValue)
        {
            return (Real)Mother.Dice.Next(maxValue);
        }

        public static Real Next(int minValue, int maxValue)
        {
            return (Real)Mother.Dice.Next(minValue, maxValue);
        }
    }
}
