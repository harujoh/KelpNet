#if DOUBLE
using System;
using Real = System.Double;
#elif NETSTANDARD2_1
using Real = System.Single;
using Math = System.MathF;
#else
using Real = System.Single;
using Math = KelpNet.MathF;
#endif

namespace KelpNet
{
    //外部公開を必要としないinternalなクラスなので、それぞれの型で名称を変えずに済む
    internal class Broth
    {
        // 平均mu, 標準偏差sigmaの正規分布乱数を得る
        public static Real RandomNormal(Real sigma = 1, Real mu = 0)
        {
            Real boxMuller = Math.Sqrt(-2 * Math.Log(Broth.Random())) * Math.Sin(2 * Math.PI * Broth.Random());
            return sigma * boxMuller + mu;
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
