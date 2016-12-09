using System;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class RMSprop : Optimizer
    {
        public double LearningRate;
        public double Alpha;
        public double Epsilon;

        public RMSprop(double learningRate = 0.01, double alpha = 0.99, double epsilon = 1e-8)
        {
            this.LearningRate = learningRate;
            this.Alpha = alpha;
            this.Epsilon = epsilon;
        }

        public override void Initilise(FunctionParameter[] functionParameters)
        {
            this.OptimizerParameters = new OptimizerParameter[functionParameters.Length];

            for (int i = 0; i < this.OptimizerParameters.Length; i++)
            {
                this.OptimizerParameters[i] = new RMSpropParameter(functionParameters[i], this);
            }
        }
    }

    [Serializable]
    class RMSpropParameter : OptimizerParameter
    {
        private readonly RMSprop optimiser;
        private readonly double[] ms;

        public RMSpropParameter(FunctionParameter parameter, RMSprop optimiser) : base(parameter)
        {
            this.optimiser = optimiser;
            this.ms = new double[parameter.Length];
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                double grad = this.FunctionParameter.Grad.Data[i];
                this.ms[i] *= this.optimiser.Alpha;
                this.ms[i] += (1 - this.optimiser.Alpha) * grad * grad;

                this.FunctionParameter.Param.Data[i] -= this.optimiser.LearningRate * grad / (Math.Sqrt(this.ms[i]) + this.optimiser.Epsilon);
            }
        }
    }

}
