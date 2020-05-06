using System;

#if DOUBLE
using KelpMath = System.Math;
#elif NETSTANDARD2_1
using KelpMath = System.MathF;
#elif NETSTANDARD2_0
using KelpMath = KelpNet.MathF;
#endif

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
        static Real _beta;
        static Real _boxMuller1;

        // 平均mu, 標準偏差sigmaの正規分布乱数を得る。Box-Muller法による。
        public static Real RandomNormal(Real sigma = 1, Real mu = 0)
        {
            Real boxMuller2;

            if (!_flip)
            {
                _beta = Broth.Random() * KelpMath.PI * 2;
                _boxMuller1 = KelpMath.Sqrt(-2 * KelpMath.Log(Broth.Random()));
                boxMuller2 = KelpMath.Sin(_beta);
            }
            else
            {
                boxMuller2 = KelpMath.Cos(_beta);
            }

            _flip = !_flip;

            return sigma * _boxMuller1 * boxMuller2 + mu;
        }

        public static Real Random()
        {
            return (Real)Mother.Dice.NextDouble();
        }

        public static int Next()
        {
            return Mother.Dice.Next();
        }

        public static int Next(int maxValue)
        {
            return Mother.Dice.Next(maxValue);
        }

        public static int Next(int minValue, int maxValue)
        {
            return Mother.Dice.Next(minValue, maxValue);
        }
    }
}
