using System;
using System.Collections.Generic;
#if DOUBLE
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
#if !DOUBLE
    public class WeightDecay<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T Rate;

        public WeightDecay(T rate)
        {
            this.Rate = rate;

            switch (this)
            {
                case WeightDecay<float> weightDecayF:
                    weightDecayF.Update = () => WeightDecayF.Update(weightDecayF.Rate, weightDecayF.FunctionParameters);
                    break;

                case WeightDecay<double> weightDecayD:
                    weightDecayD.Update = () => WeightDecayD.Update(weightDecayD.Rate, weightDecayD.FunctionParameters);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class WeightDecayD
#else
    public static class WeightDecayF
#endif
    {
        public static void Update(Real rate, List<NdArray<Real>> functionParameters)
        {
            foreach (var functionParameter in functionParameters)
            {
                for (int i = 0; i < functionParameter.Data.Length; i++)
                {
                    functionParameter.Grad[i] += functionParameter.Data[i] * rate;
                }
            }
        }
    }
}
