using System;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class AdaGrad : Optimizer
    {
        public double LearningRate;
        public double Epsilon;

        public AdaGrad(double learningRate = 0.01, double epsilon = 1e-8)
        {
            this.LearningRate = learningRate;
            this.Epsilon = epsilon;
        }

        public override void Initilise(OptimizeParameter[] functionParameters)
        {
            this.OptimizerParameters = new OptimizerParameter[functionParameters.Length];

            for (int i = 0; i < this.OptimizerParameters.Length; i++)
            {
                this.OptimizerParameters[i] = new AdaGradParameter(functionParameters[i], this);
            }
        }
    }

    [Serializable]
    class AdaGradParameter : OptimizerParameter
    {
        private readonly AdaGrad optimiser;
        private readonly double[] h;

        public AdaGradParameter(OptimizeParameter functionParameter, AdaGrad optimiser) : base(functionParameter)
        {
            this.h = new double[functionParameter.Length];
            this.optimiser = optimiser;
        }

        public override void Update()
        {
            for (int i = 0; i < this.FunctionParameters.Length; i++)
            {
                double grad = this.FunctionParameters.Grad.Data[i];

                this.h[i] += grad * grad;

                this.FunctionParameters.Param.Data[i] -= this.optimiser.LearningRate * grad / (Math.Sqrt(this.h[i]) + this.optimiser.Epsilon);
            }
        }
    }

}
