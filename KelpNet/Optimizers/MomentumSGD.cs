using System;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class MomentumSGD : IOptimizer
    {
        public double LearningRate;
        public double Momentum;

        private readonly double[] v;

        public MomentumSGD(double learningRate = 0.01, double momentum = 0.9, int parameterLength = 0)
        {
            this.LearningRate = learningRate;
            this.Momentum = momentum;

            this.v = new double[parameterLength];
        }

        public IOptimizer Initialise(OptimizeParameter parameter)
        {
            return new MomentumSGD(this.LearningRate, this.Momentum, parameter.Length);
        }

        public void Update(OptimizeParameter parameter)
        {
            for (int i = 0; i < parameter.Length; i++)
            {
                this.v[i] *= this.Momentum;
                this.v[i] -= this.LearningRate * parameter.Grad.Data[i];

                parameter.Param.Data[i] += this.v[i];
            }
        }
    }
}
