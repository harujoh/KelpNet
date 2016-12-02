using System;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
    public class RMSprop : Optimizer
    {
        private double[][] ms;

        public double LearningRate;
        public double Alpha;
        public double Epsilon;

        public RMSprop(OptimizeParameter[] parameters, double learningRate = 0.01, double alpha = 0.99, double epsilon = 1e-8) : base(parameters)
        {
            this.LearningRate = learningRate;
            this.Alpha = alpha;
            this.Epsilon = epsilon;

            this.ms = new double[parameters.Length][];
            for (int i = 0; i < this.ms.Length; i++)
            {
                this.ms[i] = new double[parameters[i].Param.Length];
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
                for (int j = 0; j < this.Parameters[i].Length; j++)
                {
                    double grad = this.Parameters[i].Grad.Data[j];
                    this.ms[i][j] *= this.Alpha;
                    this.ms[i][j] += (1 - this.Alpha) * grad * grad;

                    this.Parameters[i].Param.Data[j] -= this.LearningRate * grad / (Math.Sqrt(this.ms[i][j]) + this.Epsilon);
                }
            }
#if !DEBUG
            );
#endif
        }
    }
}
