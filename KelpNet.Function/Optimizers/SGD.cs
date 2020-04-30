using System;
using System.Runtime.Serialization;
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

        public SGD(double learningRate = 0.01)
        {
            switch (this)
            {
                case SGD<float> sgdF:
                    sgdF.LearningRate = (float)learningRate;
                    sgdF.Update = () => OptimizerF.Update(sgdF);
                    break;

                case SGD<double> sgdD:
                    sgdD.LearningRate = learningRate;
                    sgdD.Update = () => OptimizerD.Update(sgdD);
                    break;
            }
        }

        public override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new SGDParameter<T>(functionParameter, this));
            }
        }
    }

    public class SGDParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly SGD<T> optimizer;

        public SGDParameter(NdArray<T> functionParameter, SGD<T> optimizer) : base(functionParameter)
        {
            this.optimizer = optimizer;

            switch (this)
            {
                case SGDParameter<float> sgdParameterF:
                    sgdParameterF.UpdateFunctionParameters = () => SGDParameterF.UpdateFunctionParameters(sgdParameterF.optimizer.LearningRate, sgdParameterF.FunctionParameter);
                    break;

                case SGDParameter<double> sgdParameterD:
                    sgdParameterD.UpdateFunctionParameters = () => SGDParameterD.UpdateFunctionParameters(sgdParameterD.optimizer.LearningRate, sgdParameterD.FunctionParameter);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class SGDParameterD
#else
    public static class SGDParameterF
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
