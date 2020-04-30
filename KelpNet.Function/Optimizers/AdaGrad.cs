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
    public class AdaGrad<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T LearningRate;
        public T Epsilon;

        public AdaGrad(double learningRate = 0.01, double epsilon = 1e-8)
        {
            switch (this)
            {
                case AdaGrad<float> adagradF:
                    adagradF.LearningRate = (float)learningRate;
                    adagradF.Epsilon = (float)epsilon;
                    adagradF.Update = () => OptimizerF.Update(adagradF);
                    break;

                case AdaGrad<double> adagradD:
                    adagradD.LearningRate = learningRate;
                    adagradD.Epsilon = epsilon;
                    adagradD.Update = () => OptimizerD.Update(adagradD);
                    break;
            }
        }

        public override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdaGradParameter<T>(functionParameter, this));
            }
        }
    }

    public class AdaGradParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly AdaGrad<T> optimizer;
        private readonly T[] h;

        public AdaGradParameter(NdArray<T> functionParameter, AdaGrad<T> optimizer) : base(functionParameter)
        {
            this.h = new T[functionParameter.Data.Length];
            this.optimizer = optimizer;

            switch (this)
            {
                case AdaGradParameter<float> adaGradParameterF:
                    adaGradParameterF.UpdateFunctionParameters = () => AdaGradParameterF.UpdateFunctionParameters(adaGradParameterF.optimizer.LearningRate, adaGradParameterF.optimizer.Epsilon, adaGradParameterF.h, adaGradParameterF.FunctionParameter);
                    break;

                case AdaGradParameter<double> adaGradParameterD:
                    adaGradParameterD.UpdateFunctionParameters = () => AdaGradParameterD.UpdateFunctionParameters(adaGradParameterD.optimizer.LearningRate, adaGradParameterD.optimizer.Epsilon, adaGradParameterD.h, adaGradParameterD.FunctionParameter);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class AdaGradParameterD
#else
    public static class AdaGradParameterF
#endif
    {
        public static void UpdateFunctionParameters(Real learningRate, Real epsilon, Real[] h, NdArray<Real> functionParameter)
        {
            for (int i = 0; i < functionParameter.Data.Length; i++)
            {
                Real grad = functionParameter.Grad[i];

                h[i] += grad * grad;

                functionParameter.Data[i] -= learningRate * grad / (KelpMath.Sqrt(h[i]) + epsilon);
            }
        }
    }

}
