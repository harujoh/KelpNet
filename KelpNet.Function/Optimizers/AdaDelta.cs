using System;
using System.Threading.Tasks;

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
    [Serializable]
    public class AdaDelta<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T Rho;
        public T Epsilon;

        public AdaDelta(T? rho = null, T? epsilon = null)
        {
            this.Rho = rho??(TVal<T>)0.95;
            this.Epsilon = epsilon??(TVal<T>)1e-6;

            switch (this)
            {
                case AdaDelta<float> adaDeltaF:
                    adaDeltaF.Update = () => OptimizerF.Update(adaDeltaF);
                    break;

                case AdaDelta<double> adaDeltaD:
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
        public static void UpdateFunctionParameters(Real[] msg, Real[] msdx, Real rho, Real epsilon, NdArray<Real> functionParameter)
        {
            for (int i = 0; i < functionParameter.Data.Length; i++)
            {
                Real grad = functionParameter.Grad[i];
                msg[i] *= rho;
                msg[i] += (1 - rho) * grad * grad;

                Real dx = Math.Sqrt((msdx[i] + epsilon) / (msg[i] + epsilon)) * grad;

                msdx[i] *= rho;
                msdx[i] += (1 - rho) * dx * dx;

                functionParameter.Data[i] -= dx;
            }
        }
    }
}
