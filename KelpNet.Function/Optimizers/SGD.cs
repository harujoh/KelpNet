using System;

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    public class SGD<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T LearningRate;

        public SGD(T? learningRate = null)
        {
            this.LearningRate = learningRate ?? (TVal<T>)0.01;

            switch (this)
            {
                case SGD<float> sgdF:
                    sgdF.Update = () => OptimizerF.Update(sgdF);
                    sgdF.UpdateFunctionParameters = (i) => SGDF.UpdateFunctionParameters(sgdF.LearningRate, sgdF.FunctionParameters[i]);
                    break;

                case SGD<double> sgdD:
                    sgdD.Update = () => OptimizerD.Update(sgdD);
                    sgdD.UpdateFunctionParameters = (i) => SGDD.UpdateFunctionParameters(sgdD.LearningRate, sgdD.FunctionParameters[i]);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class SGDD
#else
    public static class SGDF
#endif
    {
        public static void UpdateFunctionParameters(Real learningRate, NdArray<Real> functionParameter)
        {
            for (int i = 0; i < functionParameter.Data.Length; i++)
            {
                functionParameter.Data[i] -= learningRate * functionParameter.Grad[i];
            }
        }
    }

}
