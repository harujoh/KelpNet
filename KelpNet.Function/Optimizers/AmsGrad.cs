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
    public class AmsGrad<T> : Adam<T> where T : unmanaged, IComparable<T>
    {
        private List<T[]> vhat = new List<T[]>();

        public AmsGrad(T? alpha = null, T? beta1 = null, T? beta2 = null, T? epsilon = null, T? eta = null) : base(alpha: alpha, beta1: beta1, beta2: beta2, epsilon: epsilon, eta: eta)
        {
            switch (this)
            {
                case AmsGrad<float> amsGradF:
                    amsGradF.Update = () => OptimizerF.Update(amsGradF);
                    amsGradF.UpdateFunctionParameters = (i) => AmsGradF.UpdateFunctionParameters(amsGradF.Alpha, amsGradF.Beta1, amsGradF.Beta2, amsGradF.Epsilon, amsGradF.Eta, UpdateCount, amsGradF.FunctionParameters[i], amsGradF.m[i], amsGradF.v[i], amsGradF.vhat[i]);
                    break;

                case AmsGrad<double> amsGradD:
                    amsGradD.Update = () => OptimizerD.Update(amsGradD);
                    amsGradD.UpdateFunctionParameters = (i) => AmsGradD.UpdateFunctionParameters(amsGradD.Alpha, amsGradD.Beta1, amsGradD.Beta2, amsGradD.Epsilon, amsGradD.Eta, UpdateCount, amsGradD.FunctionParameters[i], amsGradD.m[i], amsGradD.v[i], amsGradD.vhat[i]);
                    break;
            }
        }

        protected override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.m.Add(new T[functionParameter.Data.Length]);
                this.v.Add(new T[functionParameter.Data.Length]);
                this.vhat.Add(new T[functionParameter.Data.Length]);
            }
        }
    }
#endif

#if DOUBLE
    public static class AmsGradD
#else
    public static class AmsGradF
#endif
    {
        public static void UpdateFunctionParameters(Real alpha, Real beta1, Real beta2, Real epsilon, Real eta, long updateCount, NdArray<Real> functionParameter, Real[] m, Real[] v, Real[] vhat)
        {
            Real alphaT = AdamParameter.GetAlphaT(alpha, beta1, beta2, updateCount);

            for (int i = 0; i < functionParameter.Data.Length; i++)
            {
                Real grad = functionParameter.Grad[i];

                m[i] += (1 - beta1) * (grad - m[i]);
                v[i] += (1 - beta2) * (grad * grad - v[i]);

                if (vhat[i] < v[i])
                {
                    vhat[i] = v[i];
                }

                Real step = alphaT / (Math.Sqrt(vhat[i]) + epsilon);

                functionParameter.Data[i] -= eta * step * m[i];
            }
        }
    }
}
