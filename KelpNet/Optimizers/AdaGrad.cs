using System;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
    public class AdaGrad : Optimizer
    {
        private readonly double[][] h;

        public double LearningRate;
        public double Epsilon;

        public AdaGrad(OptimizeParameter[] parameters, double learningRate = 0.01, double epsilon = 1e-8):base(parameters)
        {
            this.LearningRate = learningRate;
            this.Epsilon = epsilon;

            this.h = new double[parameters.Length][];
            for (int i = 0; i < this.h.Length; i++)
            {
                this.h[i] = new double[parameters[i].Param.Length];
            }
        }

        protected override void DoUpdate()
        {
#if DEBUG
            for (int i = 0; i < this.Parameters.Length; i++)
#else
            Parallel.For(0, this.Parameters.Length, i => 
#endif
            {
                for (int k = 0; k < this.Parameters[i].Length; k++)
                {
                    double grad = this.Parameters[i].Grad.Data[k];

                    this.h[i][k] += grad * grad;

                    this.Parameters[i].Param.Data[k] -= this.LearningRate * grad / (Math.Sqrt(this.h[i][k]) + this.Epsilon);

                }
            }
#if !DEBUG
            );
#endif
        }
    }
}
