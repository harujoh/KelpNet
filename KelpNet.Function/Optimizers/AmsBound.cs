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
    public class AmsBound<T> : Adam<T> where T : unmanaged, IComparable<T>
    {
        public T InitialAlpha;

        public T Upper;
        public T Lower;

        public T FinalLr;
        public T Gamma;

        private List<T[]> vhat = new List<T[]>();

        public AmsBound(T? alpha = null, T? beta1 = null, T? beta2 = null, T? finalLr = null, T? gamma = null, T? epsilon = null, T? eta = null) : base(alpha: alpha, beta1: beta1, beta2: beta2, epsilon: epsilon, eta: eta)
        {
            this.InitialAlpha = alpha ?? (TVal<T>)0.001;
            this.FinalLr = finalLr ?? (TVal<T>)0.1;
            this.Gamma = gamma ?? (TVal<T>)1e-3;

            switch (this)
            {
                case AmsBound<float> amsBoundF:
                    amsBoundF.Update = () => OptimizerF.Update(amsBoundF);
                    amsBoundF.UpdateFunctionParameters = (i) => AmsBoundF.UpdateFunctionParameters(amsBoundF.Alpha, amsBoundF.InitialAlpha, amsBoundF.Gamma, amsBoundF.Beta1, amsBoundF.Beta2, amsBoundF.Epsilon, amsBoundF.Eta, UpdateCount, amsBoundF.FunctionParameters[i], amsBoundF.m[i], amsBoundF.v[i], amsBoundF.vhat[i], ref amsBoundF.FinalLr, out amsBoundF.Lower, out amsBoundF.Upper, amsBoundF.Clip);
                    break;

                case AmsBound<double> amsBoundD:
                    amsBoundD.Update = () => OptimizerD.Update(amsBoundD);
                    amsBoundD.UpdateFunctionParameters = (i) => AmsBoundD.UpdateFunctionParameters(amsBoundD.Alpha, amsBoundD.InitialAlpha, amsBoundD.Gamma, amsBoundD.Beta1, amsBoundD.Beta2, amsBoundD.Epsilon, amsBoundD.Eta, UpdateCount, amsBoundD.FunctionParameters[i], amsBoundD.m[i], amsBoundD.v[i], amsBoundD.vhat[i], ref amsBoundD.FinalLr, out amsBoundD.Lower, out amsBoundD.Upper, amsBoundD.Clip);
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

        public T Clip(T val)
        {
            if (val.CompareTo(Lower) <= 0) return Lower;
            if (val.CompareTo(Upper) >= 0) return Upper;
            return val;
        }
    }
#endif

    //外部公開しないため型スイッチを必要としない
    internal static class AmsBound
    {
        public static void UpdateBound(Real alpha, Real initialAlpha, Real gamma, long updateCount, ref Real finalLr, out Real lower, out Real upper)
        {
            finalLr = finalLr * alpha / initialAlpha;

            lower = finalLr * (1.0f - 1.0f / (gamma * updateCount + 1.0f));
            upper = finalLr * (1.0f + 1.0f / (gamma * updateCount));
        }
    }

#if DOUBLE
    public static class AmsBoundD
#else
    public static class AmsBoundF
#endif
    {
        public static void UpdateFunctionParameters(Real alpha, Real initialAlpha, Real gamma, Real beta1, Real beta2, Real epsilon, Real eta, long updateCount, NdArray<Real> functionParameter, Real[] m, Real[] v, Real[] vhat, ref Real finalLr, out Real lower, out Real upper, Func<Real, Real> clip)
        {
            Real alphaT = AdamParameter.GetAlphaT(alpha, beta1, beta2, updateCount);

            AmsBound.UpdateBound(alpha, initialAlpha, gamma, updateCount, ref finalLr, out lower, out upper);

            for (int i = 0; i < functionParameter.Data.Length; i++)
            {
                Real grad = functionParameter.Grad[i];

                m[i] += (1 - beta1) * (grad - m[i]);
                v[i] += (1 - beta2) * (grad * grad - v[i]);

                if (vhat[i] < v[i])
                {
                    vhat[i] = v[i];
                }

                Real step = clip(alphaT / (Math.Sqrt(vhat[i]) + epsilon));

                functionParameter.Data[i] -= eta * step * m[i];
            }
        }
    }
}
