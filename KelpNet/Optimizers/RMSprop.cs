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

        public override void Initilise(OptimizeParameter[] functionParameters)
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

        public RMSpropParameter(OptimizeParameter parameter, RMSprop optimiser) : base(parameter)
        {
            this.optimiser = optimiser;
            this.ms = new double[parameter.Length];
        }

        public override void Update()
        {
            for (int i = 0; i < FunctionParameters.Length; i++)
            {
                double grad = FunctionParameters.Grad.Data[i];
                this.ms[i] *= this.optimiser.Alpha;
                this.ms[i] += (1 - this.optimiser.Alpha) * grad * grad;

                this.FunctionParameters.Param.Data[i] -= this.optimiser.LearningRate * grad / (Math.Sqrt(this.ms[i]) + this.optimiser.Epsilon);
            }
        }
    }

}
