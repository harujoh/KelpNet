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
    public class RMSprop<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T LearningRate;
        public T Alpha;
        public T Epsilon;

        private List<T[]> ms = new List<T[]>();

        public RMSprop(T? learningRate = null, T? alpha = null, T? epsilon = null)
        {
            this.LearningRate = learningRate??(TVal<T>)0.01;
            this.Alpha = alpha?? (TVal<T>)0.99;
            this.Epsilon = epsilon?? (TVal<T>)1e-8;

            switch (this)
            {
                case RMSprop<float> rmsPropF:
                    rmsPropF.Update = () => OptimizerF.Update(rmsPropF);
                    rmsPropF.UpdateFunctionParameters = (i) => RMSpropF.UpdateFunctionParameters(rmsPropF.LearningRate, rmsPropF.Alpha, rmsPropF.Epsilon, rmsPropF.FunctionParameters[i], rmsPropF.ms[i]);
                    break;

                case RMSprop<double> rmsPropD:
                    rmsPropD.Update = () => OptimizerD.Update(rmsPropD);
                    rmsPropD.UpdateFunctionParameters = (i) => RMSpropD.UpdateFunctionParameters(rmsPropD.LearningRate, rmsPropD.Alpha, rmsPropD.Epsilon, rmsPropD.FunctionParameters[i], rmsPropD.ms[i]);
                    break;
            }
        }

        protected override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.ms.Add(new T[functionParameter.Data.Length]);
            }
        }
    }
#endif

#if DOUBLE
    public static class RMSpropD
#else
    public static class RMSpropF
#endif
    {
        public static void UpdateFunctionParameters(Real learningRate, Real alpha, Real epsilon, NdArray<Real> functionParameter, Real[] ms)
        {
            for (int i = 0; i < functionParameter.Data.Length; i++)
            {
                Real grad = functionParameter.Grad[i];
                ms[i] *= alpha;
                ms[i] += (1 - alpha) * grad * grad;

                functionParameter.Data[i] -= learningRate * grad / (Math.Sqrt(ms[i]) + epsilon);
            }
        }
    }
}
