using System;

namespace KelpNet
{
    [Serializable]
    public class RMSprop<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public Real<T> LearningRate;
        public Real<T> Alpha;
        public Real<T> Epsilon;

        public RMSprop(double learningRate = 0.01, double alpha = 0.99, double epsilon = 1e-8)
        {
            this.LearningRate = learningRate;
            this.Alpha = alpha;
            this.Epsilon = epsilon;
        }

        internal override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new RMSpropParameter<T>(functionParameter, this));
            }
        }
    }

    [Serializable]
    class RMSpropParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly RMSprop<T> optimizer;
        private readonly Real<T>[] ms;

        public RMSpropParameter(NdArray<T> parameter, RMSprop<T> optimizer) : base(parameter)
        {
            this.optimizer = optimizer;
            this.ms = new Real<T>[parameter.Data.Length];
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Data.Length; i++)
            {
                Real<T> grad = this.FunctionParameter.Grad[i];
                this.ms[i] *= this.optimizer.Alpha;
                this.ms[i] += (1 - this.optimizer.Alpha) * grad * grad;

                this.FunctionParameter.Data[i] -= this.optimizer.LearningRate * grad / (Math.Sqrt(this.ms[i]) + this.optimizer.Epsilon);
            }
        }
    }

}
