using System;

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
    public class AdamW<T> : Adam<T> where T : unmanaged, IComparable<T>
    {
        public T WeightDecayRate;

        public AdamW(T? alpha = null, T? beta1 = null, T? beta2 = null, T? epsilon = null, T? eta = null, T weightDecayRate = default(T)) : base(alpha: alpha, beta1: beta1, beta2: beta2, epsilon: epsilon, eta: eta)
        {
            WeightDecayRate = weightDecayRate;

            switch (this)
            {
                case AdamW<float> adamWF:
                    adamWF.Update = () => OptimizerF.Update(adamWF);
                    adamWF.UpdateFunctionParameters = (i) => AdamWF.UpdateFunctionParameters(adamWF.Alpha, adamWF.WeightDecayRate, adamWF.Beta1, adamWF.Beta2, adamWF.Epsilon, adamWF.Eta, UpdateCount, adamWF.FunctionParameters[i], adamWF.m[i], adamWF.v[i]);
                    break;

                case AdamW<double> adamWD:
                    adamWD.Update = () => OptimizerD.Update(adamWD);
                    adamWD.UpdateFunctionParameters = (i) => AdamWD.UpdateFunctionParameters(adamWD.Alpha, adamWD.WeightDecayRate, adamWD.Beta1, adamWD.Beta2, adamWD.Epsilon, adamWD.Eta, UpdateCount, adamWD.FunctionParameters[i], adamWD.m[i], adamWD.v[i]);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class AdamWD
#else
    public static class AdamWF
#endif
    {
        public static void UpdateFunctionParameters(Real alpha, Real weightDecayRate, Real beta1, Real beta2, Real epsilon, Real eta, long updateCount, NdArray<Real> functionParameter, Real[] m, Real[] v)
        {
            Real alphaT = AdamParameter.GetAlphaT(alpha, beta1, beta2, updateCount);

            for (int i = 0; i < functionParameter.Data.Length; i++)
            {
                Real grad = functionParameter.Grad[i];

                m[i] += (1 - beta1) * (grad - m[i]);
                v[i] += (1 - beta2) * (grad * grad - v[i]);

                Real step = alphaT / (Math.Sqrt(v[i]) + epsilon);

                functionParameter.Data[i] -= eta * (step * m[i] + weightDecayRate * functionParameter.Data[i]);
            }
        }
    }
}
