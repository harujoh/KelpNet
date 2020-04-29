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
#if !DOUBLE
    public class AdaBound<T> : Adam<T> where T : unmanaged, IComparable<T>
    {
        public T InitialAlpha;

        public T Upper;
        public T Lower;

        public T FinalLr;
        public T Gamma;

        public AdaBound(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double finalLr = 0.1, double gamma = 1e-3, double epsilon = 1e-8, double eta = 1.0) : base(alpha: alpha, beta1: beta1, beta2: beta2, epsilon: epsilon, eta: eta)
        {
            switch (this)
            {
                case AdaBound<float> adaBoundF:
                    adaBoundF.InitialAlpha = (float)alpha;
                    adaBoundF.FinalLr = (float)finalLr;
                    adaBoundF.Gamma = (float)gamma;
                    adaBoundF.Update = () => OptimizerF.Update(adaBoundF);
                    break;

                case AdaBound<double> adaBoundD:
                    adaBoundD.InitialAlpha = alpha;
                    adaBoundD.FinalLr = finalLr;
                    adaBoundD.Gamma = gamma;
                    adaBoundD.Update = () => OptimizerD.Update(adaBoundD);
                    break;
            }
        }

        public override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdaBoundParameter<T>(functionParameter, this));
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
    internal static class AdaBound
    {
        public static void UpdateBound(Real Alpha, Real InitialAlpha, Real Gamma, long UpdateCount, ref Real FinalLr, out Real Lower, out Real Upper)
        {
            FinalLr = FinalLr * Alpha / InitialAlpha;

            Lower = FinalLr * (1.0f - 1.0f / (Gamma * UpdateCount + 1.0f));
            Upper = FinalLr * (1.0f + 1.0f / (Gamma * UpdateCount));
        }
    }

#if !DOUBLE
    public class AdaBoundParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly AdaBound<T> _optimizer;

        private readonly T[] m;
        private readonly T[] v;

        public AdaBoundParameter(NdArray<T> parameter, AdaBound<T> optimizer) : base(parameter)
        {
            this.m = new T[parameter.Data.Length];
            this.v = new T[parameter.Data.Length];

            this._optimizer = optimizer;

            switch (this)
            {
                case AdaBoundParameter<float> adamParameterF:
                    adamParameterF.UpdateFunctionParameters = () => AdaBoundParameterF.UpdateFunctionParameters(adamParameterF._optimizer.Alpha, adamParameterF._optimizer.InitialAlpha, adamParameterF._optimizer.Gamma, adamParameterF._optimizer.Beta1, adamParameterF._optimizer.Beta2, adamParameterF._optimizer.Epsilon, adamParameterF._optimizer.Eta, _optimizer.UpdateCount, adamParameterF.FunctionParameter, adamParameterF.m, adamParameterF.v, ref adamParameterF._optimizer.FinalLr, out adamParameterF._optimizer.Lower, out adamParameterF._optimizer.Upper, adamParameterF._optimizer.Clip);
                    break;

                case AdaBoundParameter<double> adamParameterD:
                    adamParameterD.UpdateFunctionParameters = () => AdaBoundParameterD.UpdateFunctionParameters(adamParameterD._optimizer.Alpha, adamParameterD._optimizer.InitialAlpha, adamParameterD._optimizer.Gamma, adamParameterD._optimizer.Beta1, adamParameterD._optimizer.Beta2, adamParameterD._optimizer.Epsilon, adamParameterD._optimizer.Eta, _optimizer.UpdateCount, adamParameterD.FunctionParameter, adamParameterD.m, adamParameterD.v, ref adamParameterD._optimizer.FinalLr, out adamParameterD._optimizer.Lower, out adamParameterD._optimizer.Upper, adamParameterD._optimizer.Clip);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class AdaBoundParameterD
#else
    public static class AdaBoundParameterF
#endif
    {
        public static void UpdateFunctionParameters(Real Alpha, Real InitialAlpha, Real Gamma, Real Beta1, Real Beta2, Real Epsilon, Real Eta, long UpdateCount, NdArray<Real> FunctionParameter, Real[] m, Real[] v, ref Real FinalLr, out Real Lower, out Real Upper, Func<Real, Real> Clip)
        {
            Real alphaT = AdamParameter.GetAlphaT(Alpha, Beta1, Beta2, UpdateCount);

            AdaBound.UpdateBound(Alpha, InitialAlpha, Gamma, UpdateCount, ref FinalLr, out Lower, out Upper);

            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                Real grad = FunctionParameter.Grad[i];

                m[i] += (1 - Beta1) * (grad - m[i]);
                v[i] += (1 - Beta2) * (grad * grad - v[i]);

                Real step = Clip(alphaT / (KelpMath.Sqrt(v[i]) + Epsilon));

                FunctionParameter.Data[i] -= Eta * step * m[i];
            }
        }
    }
}
