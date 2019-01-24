using System;

namespace KelpNet
{
    [Serializable]
    public class AdaGrad<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public Real<T> LearningRate;
        public Real<T> Epsilon;

        public AdaGrad(double learningRate = 0.01f, double epsilon = 1e-8f)
        {
            this.LearningRate = learningRate;
            this.Epsilon = epsilon;
        }

        internal override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdaGradParameter<T>(functionParameter, this));
            }
        }
    }

    [Serializable]
    class AdaGradParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly AdaGrad<T> optimizer;
        private readonly Real<T>[] h;

        public AdaGradParameter(NdArray<T> functionParameter, AdaGrad<T> optimizer) : base(functionParameter)
        {
            this.h = new Real<T>[functionParameter.Data.Length];
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Data.Length; i++)
            {
                Real<T> grad = this.FunctionParameter.Grad[i];

                this.h[i] += grad * grad;

                this.FunctionParameter.Data[i] -= this.optimizer.LearningRate * grad / (Math.Sqrt(this.h[i]) + this.optimizer.Epsilon);
            }
        }
    }

}
