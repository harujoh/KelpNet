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
    public class RMSprop<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T LearningRate;
        public T Alpha;
        public T Epsilon;

        public RMSprop(double learningRate = 0.01, double alpha = 0.99, double epsilon = 1e-8)
        {
            switch (this)
            {
                case RMSprop<float> rmsPropF:
                    rmsPropF.LearningRate = (float)learningRate;
                    rmsPropF.Alpha = (float)alpha;
                    rmsPropF.Epsilon = (float)epsilon;
                    rmsPropF.Update = () => OptimizerF.Update(rmsPropF);
                    break;

                case RMSprop<double> rmsPropD:
                    rmsPropD.LearningRate = learningRate;
                    rmsPropD.Alpha = alpha;
                    rmsPropD.Epsilon = epsilon;
                    rmsPropD.Update = () => OptimizerD.Update(rmsPropD);
                    break;
            }
        }

        public override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new RMSpropParameter<T>(functionParameter, this));
            }
        }
    }

    public class RMSpropParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly RMSprop<T> optimizer;
        private readonly T[] ms;

        public RMSpropParameter(NdArray<T> parameter, RMSprop<T> optimizer) : base(parameter)
        {
            this.optimizer = optimizer;
            this.ms = new T[parameter.Data.Length];

            switch (this)
            {
                case RMSpropParameter<float> rmsPropParameterF:
                    rmsPropParameterF.UpdateFunctionParameters = () => RMSpropParameterF.UpdateFunctionParameters(rmsPropParameterF.optimizer.LearningRate, rmsPropParameterF.optimizer.Alpha, rmsPropParameterF.optimizer.Epsilon, rmsPropParameterF.FunctionParameter, rmsPropParameterF.ms);
                    break;

                case RMSpropParameter<double> rmsPropParameterF:
                    rmsPropParameterF.UpdateFunctionParameters = () => RMSpropParameterD.UpdateFunctionParameters(rmsPropParameterF.optimizer.LearningRate, rmsPropParameterF.optimizer.Alpha, rmsPropParameterF.optimizer.Epsilon, rmsPropParameterF.FunctionParameter, rmsPropParameterF.ms);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class RMSpropParameterD
#else
    public static class RMSpropParameterF
#endif
    {
        public static void UpdateFunctionParameters(Real learningRate, Real alpha, Real epsilon, NdArray<Real> functionParameter, Real[] ms)
        {
            for (int i = 0; i < functionParameter.Data.Length; i++)
            {
                Real grad = functionParameter.Grad[i];
                ms[i] *= alpha;
                ms[i] += (1 - alpha) * grad * grad;

                functionParameter.Data[i] -= learningRate * grad / (KelpMath.Sqrt(ms[i]) + epsilon);
            }
        }
    }
}
