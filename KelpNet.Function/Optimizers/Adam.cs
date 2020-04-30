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
    public class Adam<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T Alpha;
        public T Beta1;
        public T Beta2;
        public T Epsilon;
        public T Eta;

        public Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double eta = 1.0)
        {
            switch (this)
            {
                case Adam<float> adamF:
                    adamF.Alpha = (float)alpha;
                    adamF.Beta1 = (float)beta1;
                    adamF.Beta2 = (float)beta2;
                    adamF.Epsilon = (float)epsilon;
                    adamF.Eta = (float)eta;
                    adamF.Update = () => OptimizerF.Update(adamF);
                    break;

                case Adam<double> adamD:
                    adamD.Alpha = alpha;
                    adamD.Beta1 = beta1;
                    adamD.Beta2 = beta2;
                    adamD.Epsilon = epsilon;
                    adamD.Eta = eta;
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
            Real fix1 = 1 - KelpMath.Pow(beta1, updateCount);
            Real fix2 = 1 - KelpMath.Pow(beta2, updateCount);

            return alpha * KelpMath.Sqrt(fix2) / fix1;
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

                Real step = alphaT / (KelpMath.Sqrt(v[i]) + epsilon);

                functionParameter.Data[i] -= eta * step * m[i];
            }
        }
    }
}
