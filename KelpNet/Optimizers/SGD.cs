using System;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class SGD : IOptimizer
    {
        public double LearningRate;

        public SGD(double learningRate = 0.1)
        {
            this.LearningRate = learningRate;
        }

        public IOptimizer Initialise(OptimizeParameter parameter)
        {
            return this;
        }

        public void Update(OptimizeParameter parameter)
        {
            for (int j = 0; j < parameter.Length; j++)
            {
                parameter.Param.Data[j] -= this.LearningRate * parameter.Grad.Data[j];
            }
        }
    }
}
