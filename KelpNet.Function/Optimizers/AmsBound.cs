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
    public class AmsBound<T> : Adam<T> where T : unmanaged, IComparable<T>
    {
        public T InitialAlpha;

        public T Upper;
        public T Lower;

        public T FinalLr;
        public T Gamma;

        public AmsBound(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double finalLr = 0.1, double gamma = 1e-3, double epsilon = 1e-8, double eta = 1.0) : base(alpha: alpha, beta1: beta1, beta2: beta2, epsilon: epsilon, eta: eta)
        {
            switch (this)
            {
                case AmsBound<float> amsBoundF:
                    amsBoundF.InitialAlpha = (float)alpha;
                    amsBoundF.FinalLr = (float)finalLr;
                    amsBoundF.Gamma = (float)gamma;
                    amsBoundF.Update = () => OptimizerF.Update(amsBoundF);
                    break;

                case AmsBound<double> amsBoundD:
                    amsBoundD.InitialAlpha = alpha;
                    amsBoundD.FinalLr = finalLr;
                    amsBoundD.Gamma = gamma;
                    amsBoundD.Update = () => OptimizerD.Update(amsBoundD);
                    break;
            }
        }

        public override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AmsBoundParameter<T>(functionParameter, this));
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
        public static void UpdateBound(Real Alpha, Real InitialAlpha, Real Gamma, long UpdateCount, ref Real FinalLr, out Real Lower, out Real Upper)
        {
            FinalLr = FinalLr * Alpha / InitialAlpha;

            Lower = FinalLr * (1.0f - 1.0f / (Gamma * UpdateCount + 1.0f));
            Upper = FinalLr * (1.0f + 1.0f / (Gamma * UpdateCount));
        }
    }

#if !DOUBLE
    public class AmsBoundParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly AmsBound<T> _optimizer;

        private readonly T[] m;
        private readonly T[] v;
        private readonly T[] vhat;

        public AmsBoundParameter(NdArray<T> parameter, AmsBound<T> optimizer) : base(parameter)
        {
            this.m = new T[parameter.Data.Length];
            this.v = new T[parameter.Data.Length];
            this.vhat = new T[parameter.Data.Length];

            this._optimizer = optimizer;

            switch (this)
            {
                case AmsBoundParameter<float> amsBoundParameterF:
                    amsBoundParameterF.UpdateFunctionParameters = () => AmsBoundParameterF.UpdateFunctionParameters(amsBoundParameterF._optimizer.Alpha, amsBoundParameterF._optimizer.InitialAlpha, amsBoundParameterF._optimizer.Gamma, amsBoundParameterF._optimizer.Beta1, amsBoundParameterF._optimizer.Beta2, amsBoundParameterF._optimizer.Epsilon, amsBoundParameterF._optimizer.Eta, _optimizer.UpdateCount, amsBoundParameterF.FunctionParameter, amsBoundParameterF.m, amsBoundParameterF.v, amsBoundParameterF.vhat, ref amsBoundParameterF._optimizer.FinalLr, out amsBoundParameterF._optimizer.Lower, out amsBoundParameterF._optimizer.Upper, amsBoundParameterF._optimizer.Clip);
                    break;

                case AmsBoundParameter<double> amsBoundParameterD:
                    amsBoundParameterD.UpdateFunctionParameters = () => AmsBoundParameterD.UpdateFunctionParameters(amsBoundParameterD._optimizer.Alpha, amsBoundParameterD._optimizer.InitialAlpha, amsBoundParameterD._optimizer.Gamma, amsBoundParameterD._optimizer.Beta1, amsBoundParameterD._optimizer.Beta2, amsBoundParameterD._optimizer.Epsilon, amsBoundParameterD._optimizer.Eta, _optimizer.UpdateCount, amsBoundParameterD.FunctionParameter, amsBoundParameterD.m, amsBoundParameterD.v, amsBoundParameterD.vhat, ref amsBoundParameterD._optimizer.FinalLr, out amsBoundParameterD._optimizer.Lower, out amsBoundParameterD._optimizer.Upper, amsBoundParameterD._optimizer.Clip);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class AmsBoundParameterD
#else
    public static class AmsBoundParameterF
#endif
    {
        public static void UpdateFunctionParameters(Real Alpha, Real InitialAlpha, Real Gamma, Real Beta1, Real Beta2, Real Epsilon, Real Eta, long UpdateCount, NdArray<Real> FunctionParameter, Real[] m, Real[] v, Real[] vhat, ref Real FinalLr, out Real Lower, out Real Upper, Func<Real, Real> Clip)
        {
            Real alphaT = AdamParameter.GetAlphaT(Alpha, Beta1, Beta2, UpdateCount);

            AmsBound.UpdateBound(Alpha, InitialAlpha, Gamma, UpdateCount, ref FinalLr, out Lower, out Upper);

            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                Real grad = FunctionParameter.Grad[i];

                m[i] += (1 - Beta1) * (grad - m[i]);
                v[i] += (1 - Beta2) * (grad * grad - v[i]);

                if (vhat[i] < v[i])
                {
                    vhat[i] = v[i];
                }

                Real step = Clip(alphaT / (KelpMath.Sqrt(vhat[i]) + Epsilon));

                FunctionParameter.Data[i] -= Eta * step * m[i];
            }
        }
    }
}
