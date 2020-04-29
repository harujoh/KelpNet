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
    public class AmsGrad<T> : Adam<T> where T : unmanaged, IComparable<T>
    {
        public AmsGrad(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8,double eta = 1.0) : base(alpha: alpha, beta1: beta1, beta2: beta2, epsilon: epsilon, eta: eta)
        {
            switch (this)
            {
                case AmsGrad<float> amsGradF:
                    amsGradF.Update = () => OptimizerF.Update(amsGradF);
                    break;

                case AmsGrad<double> amsGradD:
                    amsGradD.Update = () => OptimizerD.Update(amsGradD);
                    break;
            }
        }

        public override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AmsGradParameter<T>(functionParameter, this));
            }
        }
    }

    public class AmsGradParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly AmsGrad<T> _optimizer;

        private readonly T[] m;
        private readonly T[] v;
        private readonly T[] vhat;

        public AmsGradParameter(NdArray<T> parameter, AmsGrad<T> optimizer) : base(parameter)
        {
            this.m = new T[parameter.Data.Length];
            this.v = new T[parameter.Data.Length];
            this.vhat = new T[parameter.Data.Length];

            this._optimizer = optimizer;

            switch (this)
            {
                case AmsGradParameter<float> amsGradParameterF:
                    amsGradParameterF.UpdateFunctionParameters = () => AmsGradParameterF.UpdateFunctionParameters(amsGradParameterF._optimizer.Alpha, amsGradParameterF._optimizer.Beta1, amsGradParameterF._optimizer.Beta2, amsGradParameterF._optimizer.Epsilon, amsGradParameterF._optimizer.Eta, _optimizer.UpdateCount, amsGradParameterF.FunctionParameter, amsGradParameterF.m, amsGradParameterF.v, amsGradParameterF.vhat);
                    break;

                case AmsGradParameter<double> amsGradParameterD:
                    amsGradParameterD.UpdateFunctionParameters = () => AmsGradParameterD.UpdateFunctionParameters(amsGradParameterD._optimizer.Alpha, amsGradParameterD._optimizer.Beta1, amsGradParameterD._optimizer.Beta2, amsGradParameterD._optimizer.Epsilon, amsGradParameterD._optimizer.Eta, _optimizer.UpdateCount, amsGradParameterD.FunctionParameter, amsGradParameterD.m, amsGradParameterD.v, amsGradParameterD.vhat);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class AmsGradParameterD
#else
    public static class AmsGradParameterF
#endif
    {
        public static void UpdateFunctionParameters(Real Alpha, Real Beta1, Real Beta2, Real Epsilon, Real Eta, long UpdateCount, NdArray<Real> FunctionParameter, Real[] m, Real[] v, Real[] vhat)
        {
            Real alphaT = AdamParameter.GetAlphaT(Alpha, Beta1, Beta2, UpdateCount);

            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                Real grad = FunctionParameter.Grad[i];

                m[i] += (1 - Beta1) * (grad - m[i]);
                v[i] += (1 - Beta2) * (grad * grad - v[i]);

                if (vhat[i] < v[i])
                {
                    vhat[i] = v[i];
                }

                Real step = alphaT / (KelpMath.Sqrt(vhat[i]) + Epsilon);

                FunctionParameter.Data[i] -= Eta * step * m[i];
            }
        }
    }
}
