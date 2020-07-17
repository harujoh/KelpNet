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
    public class Adam<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T Alpha;
        public T Beta1;
        public T Beta2;
        public T Epsilon;
        public T Eta;

        public List<T[]> m = new List<T[]>();
        public List<T[]> v = new List<T[]>();

        public Adam(T? alpha = null, T? beta1 = null, T? beta2 = null, T? epsilon = null, T? eta = null)
        {
            this.Alpha = alpha ?? (TVal<T>)0.001;
            this.Beta1 = beta1 ?? (TVal<T>)0.9;
            this.Beta2 = beta2 ?? (TVal<T>)0.999;
            this.Epsilon = epsilon ?? (TVal<T>)1e-8;
            this.Eta = eta ?? (TVal<T>)1.0;

            switch (this)
            {
                case Adam<float> adamF:
                    adamF.Update = () => OptimizerF.Update(adamF);
                    adamF.UpdateFunctionParameters = (i) => AdamF.UpdateFunctionParameters(adamF.Alpha, adamF.Beta1, adamF.Beta2, adamF.Epsilon, adamF.Eta, adamF.UpdateCount, adamF.FunctionParameters[i], adamF.m[i], adamF.v[i]);
                    break;

                case Adam<double> adamD:
                    adamD.Update = () => OptimizerD.Update(adamD);
                    adamD.UpdateFunctionParameters = (i) => AdamD.UpdateFunctionParameters(adamD.Alpha, adamD.Beta1, adamD.Beta2, adamD.Epsilon, adamD.Eta, UpdateCount, adamD.FunctionParameters[i], adamD.m[i], adamD.v[i]);
                    break;
            }
        }

        protected override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.m.Add(new T[functionParameter.Data.Length]);
                this.v.Add(new T[functionParameter.Data.Length]);
            }
        }

        public override void Step()
        {
            for (int i = 0; i < this.Schedulers.Count; i++)
            {
                this.Alpha = this.Schedulers[i].Step(this.Alpha);
            }
        }
    }
#endif

    //外部公開しないため型スイッチを必要としない
    internal static class AdamParameter
    {
        public static Real GetAlphaT(Real alpha, Real beta1, Real beta2, long updateCount)
        {
            Real fix1 = 1 - Math.Pow(beta1, updateCount);
            Real fix2 = 1 - Math.Pow(beta2, updateCount);

            return alpha * Math.Sqrt(fix2) / fix1;
        }
    }

#if DOUBLE
    public static class AdamD
#else
    public static class AdamF
#endif
    {
        public static void UpdateFunctionParameters(Real alpha, Real beta1, Real beta2, Real epsilon, Real eta, long updateCount, NdArray<Real> functionParameter, Real[] m, Real[] v)
        {
            Real alphaT = AdamParameter.GetAlphaT(alpha, beta1, beta2, updateCount);

            for (int i = 0; i < functionParameter.Data.Length; i++)
            {
                Real grad = functionParameter.Grad[i];

                m[i] += (1 - beta1) * (grad - m[i]);
                v[i] += (1 - beta2) * (grad * grad - v[i]);

                Real step = alphaT / (Math.Sqrt(v[i]) + epsilon);

                functionParameter.Data[i] -= eta * step * m[i];
            }
        }
    }
}
