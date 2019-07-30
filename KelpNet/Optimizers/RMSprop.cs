using System;
using System.Threading.Tasks;

namespace KelpNet
{
    [Serializable]
    public class RMSprop : Optimizer
    {
        public Real LearningRate;
        public Real Alpha;
        public Real Epsilon;

        public RMSprop(double learningRate = 0.01, double alpha = 0.99, double epsilon = 1e-8)
        {
            this.LearningRate = learningRate;
            this.Alpha = alpha;
            this.Epsilon = epsilon;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new RMSpropParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    public class RMSpropParameter : OptimizerParameter
    {
        private readonly RMSprop optimizer;
        private readonly Real[] ms;

        public RMSpropParameter(NdArray parameter, RMSprop optimizer) : base(parameter)
        {
            this.optimizer = optimizer;
            this.ms = new Real[parameter.Data.Length];
        }

        public override void UpdateFunctionParameters()
        {
            Parallel.For(0, FunctionParameter.Data.Length, i =>
            {
                Real grad = this.FunctionParameter.Grad[i];
                this.ms[i] *= this.optimizer.Alpha;
                this.ms[i] += (1 - this.optimizer.Alpha) * grad * grad;

                this.FunctionParameter.Data[i] -= this.optimizer.LearningRate * grad / (Math.Sqrt(this.ms[i]) + this.optimizer.Epsilon);
            });
        }
    }

}
