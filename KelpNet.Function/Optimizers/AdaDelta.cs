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
    [Serializable]
    public class AdaDelta<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T Rho;
        public T Epsilon;

        public AdaDelta(double rho = 0.95, double epsilon = 1e-6)
        {
            switch (this)
            {
                case AdaDelta<float> adaDeltaF:
                    adaDeltaF.Rho = (float)rho;
                    adaDeltaF.Epsilon = (float)epsilon;
                    adaDeltaF.Update = () => OptimizerF.Update(adaDeltaF);
                    break;

                case AdaDelta<double> adaDeltaD:
                    adaDeltaD.Rho = rho;
                    adaDeltaD.Epsilon = epsilon;
                    adaDeltaD.Update = () => OptimizerD.Update(adaDeltaD);
                    break;
            }
        }

        public override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdaDeltaParameter<T>(functionParameter, this));
            }
        }
    }

    public class AdaDeltaParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly T[] msg;
        private readonly T[] msdx;
        private readonly AdaDelta<T> optimizer;

        public AdaDeltaParameter(NdArray<T> functionParameter, AdaDelta<T> optimizer) : base(functionParameter)
        {
            this.msg = new T[functionParameter.Data.Length];
            this.msdx = new T[functionParameter.Data.Length];
            this.optimizer = optimizer;

            switch (this)
            {
                case AdaDeltaParameter<float> adaDeltaParameterF:
                    adaDeltaParameterF.UpdateFunctionParameters = () => AdaDeltaParameterF.UpdateFunctionParameters(adaDeltaParameterF.msg, adaDeltaParameterF.msdx, adaDeltaParameterF.optimizer.Rho, adaDeltaParameterF.optimizer.Epsilon, adaDeltaParameterF.FunctionParameter);
                    break;

                case AdaDeltaParameter<double> adaDeltaParameterD:
                    adaDeltaParameterD.UpdateFunctionParameters = () => AdaDeltaParameterD.UpdateFunctionParameters(adaDeltaParameterD.msg, adaDeltaParameterD.msdx, adaDeltaParameterD.optimizer.Rho, adaDeltaParameterD.optimizer.Epsilon, adaDeltaParameterD.FunctionParameter);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class AdaDeltaParameterD
#else
    public static class AdaDeltaParameterF
#endif
    {
        public static void UpdateFunctionParameters(Real[] msg, Real[] msdx, Real Rho, Real Epsilon, NdArray<Real> FunctionParameter)
        {
            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                Real grad = FunctionParameter.Grad[i];
                msg[i] *= Rho;
                msg[i] += (1 - Rho) * grad * grad;

                Real dx = KelpMath.Sqrt((msdx[i] + Epsilon) / (msg[i] + Epsilon)) * grad;

                msdx[i] *= Rho;
                msdx[i] += (1 - Rho) * dx * dx;

                FunctionParameter.Data[i] -= dx;
            }
        }
    }
}
