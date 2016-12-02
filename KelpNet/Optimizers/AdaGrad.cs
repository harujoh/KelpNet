using System;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class AdaGrad : IOptimizer
    {
        public double LearningRate;
        public double Epsilon;

        private readonly double[] h;

        public AdaGrad(double learningRate = 0.01, double epsilon = 1e-8, int parameterLength = 0)
        {
            this.LearningRate = learningRate;
            this.Epsilon = epsilon;

            this.h = new double[parameterLength];
        }

        public IOptimizer Initialise(OptimizeParameter parameter)
        {
            return new AdaGrad(this.LearningRate, this.Epsilon, parameter.Length);
        }

        public void Update(OptimizeParameter parameter)
        {
            for (int i = 0; i < parameter.Length; i++)
            {
                double grad = parameter.Grad.Data[i];

                this.h[i] += grad * grad;

                parameter.Param.Data[i] -= this.LearningRate * grad / (Math.Sqrt(this.h[i]) + this.Epsilon);
            }
        }
    }
}
