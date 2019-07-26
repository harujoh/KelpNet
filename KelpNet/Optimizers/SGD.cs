using System;

namespace KelpNet
{
    [Serializable]
    public class SGD<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public Real<T> LearningRate;

        public SGD(double learningRate = 0.1)
        {
            this.LearningRate = learningRate;
        }

        internal override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new SGDParameter<T>(functionParameter, this));
            }
        }
    }

    [Serializable]
    class SGDParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly SGD<T> optimizer;

        public SGDParameter(NdArray<T> functionParameter, SGD<T> optimizer) : base(functionParameter)
        {
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.DataLength; i++)
            {
                this.FunctionParameter.Data[i] -= this.optimizer.LearningRate * this.FunctionParameter.Grad[i];
            }
        }
    }

}
