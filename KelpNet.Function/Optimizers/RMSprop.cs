using System;

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

        public RMSprop(T? learningRate = null, T? alpha = null, T? epsilon = null)
        {
            this.LearningRate = learningRate??(TVal<T>)0.01;
            this.Alpha = alpha?? (TVal<T>)0.99;
            this.Epsilon = epsilon?? (TVal<T>)1e-8;

            switch (this)
            {
                case RMSprop<float> rmsPropF:
                    rmsPropF.Update = () => OptimizerF.Update(rmsPropF);
                    break;

                case RMSprop<double> rmsPropD:
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

                functionParameter.Data[i] -= learningRate * grad / (Math.Sqrt(ms[i]) + epsilon);
            }
        }
    }
}
