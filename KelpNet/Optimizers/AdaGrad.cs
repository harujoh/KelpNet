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

        public override void Initilise(FunctionParameter[] functionParameters)
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

        public AdaGradParameter(FunctionParameter functionParameter, AdaGrad optimiser) : base(functionParameter)
        {
            this.h = new double[functionParameter.Length];
            this.optimiser = optimiser;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                double grad = this.FunctionParameter.Grad.Data[i];

                this.h[i] += grad * grad;

                this.FunctionParameter.Param.Data[i] -= this.optimiser.LearningRate * grad / (Math.Sqrt(this.h[i]) + this.optimiser.Epsilon);
            }
        }
    }

}
