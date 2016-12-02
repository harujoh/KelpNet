using System;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class RMSprop : IOptimizer
    {
        public double LearningRate;
        public double Alpha;
        public double Epsilon;

        private readonly double[] ms;

        public RMSprop(double learningRate = 0.01, double alpha = 0.99, double epsilon = 1e-8, int parameterLength = 0)
        {
            this.LearningRate = learningRate;
            this.Alpha = alpha;
            this.Epsilon = epsilon;

            this.ms = new double[parameterLength];
        }

        public IOptimizer Initialise(OptimizeParameter parameter)
        {
            return new RMSprop(this.LearningRate, this.Alpha, this.Epsilon, parameter.Length);
        }

        public void Update(OptimizeParameter parameter)
        {
            for (int i = 0; i < parameter.Length; i++)
            {
                double grad = parameter.Grad.Data[i];
                this.ms[i] *= this.Alpha;
                this.ms[i] += (1 - this.Alpha) * grad * grad;

                parameter.Param.Data[i] -= this.LearningRate * grad / (Math.Sqrt(this.ms[i]) + this.Epsilon);
            }
        }
    }
}
