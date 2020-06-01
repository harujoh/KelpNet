using System;
using System.Collections.Generic;

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

        private List<T[]> msg = new List<T[]>();
        private List<T[]> msdx = new List<T[]>();

        public AdaDelta(T? rho = null, T? epsilon = null)
        {
            this.Rho = rho??(TVal<T>)0.95;
            this.Epsilon = epsilon??(TVal<T>)1e-6;

            switch (this)
            {
                case AdaDelta<float> adaDeltaF:
                    adaDeltaF.Update = () => OptimizerF.Update(adaDeltaF);
                    adaDeltaF.UpdateFunctionParameters = (i) => AdaDeltaF.UpdateFunctionParameters(adaDeltaF.msg[i], adaDeltaF.msdx[i], adaDeltaF.Rho, adaDeltaF.Epsilon, adaDeltaF.FunctionParameters[i]);
                    break;

                case AdaDelta<double> adaDeltaD:
                    adaDeltaD.Update = () => OptimizerD.Update(adaDeltaD);
                    adaDeltaD.UpdateFunctionParameters = (i) => AdaDeltaD.UpdateFunctionParameters(adaDeltaD.msg[i], adaDeltaD.msdx[i], adaDeltaD.Rho, adaDeltaD.Epsilon, adaDeltaD.FunctionParameters[i]);
                    break;
            }
        }

        protected override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.msg.Add(new T[functionParameter.Data.Length]);
                this.msdx.Add(new T[functionParameter.Data.Length]);
            }
        }
    }
#endif

#if DOUBLE
    public static class AdaDeltaD
#else
    public static class AdaDeltaF
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
