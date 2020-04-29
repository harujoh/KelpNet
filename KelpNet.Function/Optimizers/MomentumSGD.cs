using System;

#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    public class MomentumSGD<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public T LearningRate;
        public T Momentum;

        public MomentumSGD(double learningRate = 0.01, double momentum = 0.9)
        {
            switch (this)
            {
                case MomentumSGD<float> momentumSgdF:
                    momentumSgdF.LearningRate = (float)learningRate;
                    momentumSgdF.Momentum = (float)momentum;
                    momentumSgdF.Update = () => OptimizerF.Update(momentumSgdF);
                    break;

                case MomentumSGD<double> momentumSgdD:
                    momentumSgdD.LearningRate = learningRate;
                    momentumSgdD.Momentum = momentum;
                    momentumSgdD.Update = () => OptimizerD.Update(momentumSgdD);
                    break;
            }
        }

        public override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new MomentumSGDParameter<T>(functionParameter, this));
            }
        }
    }

    public class MomentumSGDParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly MomentumSGD<T> optimizer;

        private readonly T[] v;

        public MomentumSGDParameter(NdArray<T> functionParameter, MomentumSGD<T> optimizer) : base(functionParameter)
        {
            this.v = new T[functionParameter.Data.Length];
            this.optimizer = optimizer;

            switch (this)
            {
                case MomentumSGDParameter<float> momentumSgdParameterF:
                    momentumSgdParameterF.UpdateFunctionParameters = () => MomentumSGDParameterF.UpdateFunctionParameters(momentumSgdParameterF.optimizer.LearningRate, momentumSgdParameterF.optimizer.Momentum, momentumSgdParameterF.v, momentumSgdParameterF.FunctionParameter);
                    break;

                case MomentumSGDParameter<double> momentumSgdParameterD:
                    momentumSgdParameterD.UpdateFunctionParameters = () => MomentumSGDParameterD.UpdateFunctionParameters(momentumSgdParameterD.optimizer.LearningRate, momentumSgdParameterD.optimizer.Momentum, momentumSgdParameterD.v, momentumSgdParameterD.FunctionParameter);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class MomentumSGDParameterD
#else
    public static class MomentumSGDParameterF
#endif
    {
        public static void UpdateFunctionParameters(Real LearningRate, Real Momentum, Real[] v, NdArray<Real> FunctionParameter)
        {
            for (int i = 0; i < FunctionParameter.Data.Length; i++)
            {
                v[i] *= Momentum;
                v[i] -= LearningRate * FunctionParameter.Grad[i];

                FunctionParameter.Data[i] += v[i];
            }
        }
    }

}
