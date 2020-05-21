using System;

#if DOUBLE
using Real = System.Double;
#elif NETSTANDARD2_1
using Real = System.Single;
using Math = System.MathF;
#elif NETSTANDARD2_0
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
                    break;

                case Adam<double> adamD:
                    adamD.Update = () => OptimizerD.Update(adamD);
                    break;
            }
        }

        public override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdamParameter<T>(functionParameter, this));
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

    public class AdamParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        public readonly Adam<T> _optimizer;

        public readonly T[] m;

        public readonly T[] v;

        public AdamParameter(NdArray<T> parameter, Adam<T> optimizer) : base(parameter)
        {
            this.m = new T[parameter.Data.Length];
            this.v = new T[parameter.Data.Length];

            this._optimizer = optimizer;

            switch (this)
            {
                case AdamParameter<float> adamParameterF:
                    adamParameterF.UpdateFunctionParameters = () => AdamParameterF.UpdateFunctionParameters(adamParameterF._optimizer.Alpha, adamParameterF._optimizer.Beta1, adamParameterF._optimizer.Beta2, adamParameterF._optimizer.Epsilon, adamParameterF._optimizer.Eta, _optimizer.UpdateCount, adamParameterF.FunctionParameter, adamParameterF.m, adamParameterF.v);
                    break;

                case AdamParameter<double> adamParameterD:
                    adamParameterD.UpdateFunctionParameters = () => AdamParameterD.UpdateFunctionParameters(adamParameterD._optimizer.Alpha, adamParameterD._optimizer.Beta1, adamParameterD._optimizer.Beta2, adamParameterD._optimizer.Epsilon, adamParameterD._optimizer.Eta, _optimizer.UpdateCount, adamParameterD.FunctionParameter, adamParameterD.m, adamParameterD.v);
                    break;
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
    public static class AdamParameterD
#else
    public static class AdamParameterF
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
