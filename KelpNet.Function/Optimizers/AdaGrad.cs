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
    public class AdaGrad<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T LearningRate;
        public T Epsilon;

        private List<T[]> h = new List<T[]>();

        public AdaGrad(T? learningRate = null, T? epsilon = null)
        {
            this.LearningRate = learningRate??(TVal<T>)0.01;
            this.Epsilon = epsilon?? (TVal<T>)1e-8;

            switch (this)
            {
                case AdaGrad<float> adaGradF:
                    adaGradF.Update = () => OptimizerF.Update(adaGradF);
                    adaGradF.UpdateFunctionParameters = (i) => AdaGradF.UpdateFunctionParameters(adaGradF.LearningRate, adaGradF.Epsilon, adaGradF.h[i], adaGradF.FunctionParameters[i]);
                    break;

                case AdaGrad<double> adaGradD:
                    adaGradD.Update = () => OptimizerD.Update(adaGradD);
                    adaGradD.UpdateFunctionParameters = (i) => AdaGradD.UpdateFunctionParameters(adaGradD.LearningRate, adaGradD.Epsilon, adaGradD.h[i], adaGradD.FunctionParameters[i]);
                    break;
            }
        }

        protected override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.h.Add(new T[functionParameter.Data.Length]);
            }
        }
    }
#endif

#if DOUBLE
    public static class AdaGradD
#else
    public static class AdaGradF
#endif
    {
        public static void UpdateFunctionParameters(Real learningRate, Real epsilon, Real[] h, NdArray<Real> functionParameter)
        {
            for (int i = 0; i < functionParameter.Data.Length; i++)
            {
                Real grad = functionParameter.Grad[i];

                h[i] += grad * grad;

                functionParameter.Data[i] -= learningRate * grad / (Math.Sqrt(h[i]) + epsilon);
            }
        }
    }

}
