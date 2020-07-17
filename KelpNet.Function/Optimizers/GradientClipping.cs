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
    //与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
    public class GradientClipping<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T Threshold;

        public GradientClipping(T threshold)
        {
            this.Threshold = threshold;

            switch (this)
            {
                case GradientClipping<float> gradientClippingF:
                    gradientClippingF.Update = () => GradientClippingF.Update(gradientClippingF.Threshold, gradientClippingF.FunctionParameters);
                    break;

                case GradientClipping<double> gradientClippingD:
                    gradientClippingD.Update = () => GradientClippingD.Update(gradientClippingD.Threshold, gradientClippingD.FunctionParameters);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class GradientClippingD
#else
    public static class GradientClippingF
#endif
    {
        public static void Update(Real threshold, List<NdArray<Real>> functionParameters)
        {
            foreach (NdArray<Real> functionParameter in functionParameters)
            {
                Real s = 0;

                for (int i = 0; i < functionParameter.Data.Length; i++)
                {
                    s += functionParameter.Grad[i] * functionParameter.Grad[i];
                }

                Real norm = Math.Sqrt(s);
                Real rate = threshold / norm;

                if (rate < 1)
                {
                    for (int i = 0; i < functionParameter.Data.Length; i++)
                    {
                        functionParameter.Grad[i] *= rate;
                    }
                }
            }
        }
    }
}
