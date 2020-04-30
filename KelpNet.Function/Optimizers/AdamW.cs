using System;
using System.Threading.Tasks;

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
    public class AdamW<T> : Adam<T> where T : unmanaged, IComparable<T>
    {
        public T WeightDecayRate;

        public AdamW(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double eta = 1.0, double weightDecayRate = 0) : base(alpha: alpha, beta1: beta1, beta2: beta2, epsilon: epsilon, eta: eta)
        {
            switch (this)
            {
                case AdamW<float> adamWF:
                    adamWF.WeightDecayRate = (float)weightDecayRate;
                    adamWF.Update = () => OptimizerF.Update(adamWF);
                    break;

                case AdamW<double> adamWD:
                    adamWD.WeightDecayRate = weightDecayRate;
                    adamWD.Update = () => OptimizerD.Update(adamWD);
                    break;
            }
        }

        public override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdamWParameter<T>(functionParameter, this));
            }
        }
    }

    public class AdamWParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly AdamW<T> _optimizer;

        private readonly T[] m;
        private readonly T[] v;

        public AdamWParameter(NdArray<T> parameter, AdamW<T> optimizer) : base(parameter)
        {
            this.m = new T[parameter.Data.Length];
            this.v = new T[parameter.Data.Length];

            this._optimizer = optimizer;

            switch (this)
            {
                case AdamWParameter<float> adamWParameterF:
                    adamWParameterF.UpdateFunctionParameters = () => AdamWParameterF.UpdateFunctionParameters(adamWParameterF._optimizer.Alpha, adamWParameterF._optimizer.WeightDecayRate, adamWParameterF._optimizer.Beta1, adamWParameterF._optimizer.Beta2, adamWParameterF._optimizer.Epsilon, adamWParameterF._optimizer.Eta, _optimizer.UpdateCount, adamWParameterF.FunctionParameter, adamWParameterF.m, adamWParameterF.v);
                    break;

                case AdamWParameter<double> adamWParameterD:
                    adamWParameterD.UpdateFunctionParameters = () => AdamWParameterD.UpdateFunctionParameters(adamWParameterD._optimizer.Alpha, adamWParameterD._optimizer.WeightDecayRate, adamWParameterD._optimizer.Beta1, adamWParameterD._optimizer.Beta2, adamWParameterD._optimizer.Epsilon, adamWParameterD._optimizer.Eta, _optimizer.UpdateCount, adamWParameterD.FunctionParameter, adamWParameterD.m, adamWParameterD.v);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class AdamWParameterD
#else
    public static class AdamWParameterF
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

                Real step = alphaT / (KelpMath.Sqrt(v[i]) + epsilon);

                functionParameter.Data[i] -= eta * (step * m[i] + weightDecayRate * functionParameter.Data[i]);
            }
        }
    }
}
