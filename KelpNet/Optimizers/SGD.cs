using System;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
    public class SGD : Optimizer
    {
        public double LearningRate;

        public SGD(OptimizeParameter[] parameters, double learningRate = 0.1) : base(parameters)
        {
            this.LearningRate = learningRate;
        }

        protected override void DoUpdate()
        {
#if DEBUG
            for (int i = 0; i < this.Parameters.Length; i++)
#else
            Parallel.For(0, this.Parameters.Length, i =>
#endif
            {
                for (int j = 0; j < this.Parameters[i].Length; j++)
                {
                    this.Parameters[i].Param.Data[j] -= this.LearningRate * this.Parameters[i].Grad.Data[j];
                }
            }
#if !DEBUG
            );
#endif
        }
    }
}
