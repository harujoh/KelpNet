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

        private double lr;
        private double alpha;
        private double eps;

        public RMSprop(double lr = 0.01,double alpha = 0.99,double eps = 1e-8)
        {
            this.lr = lr;
            this.alpha = alpha;
            this.eps = eps;
        }

        protected override void DoUpdate()
        {
#if DEBUG
            for (int i = 0; i < Parameters.Count; i++)
#else
            Parallel.For(0, this.Parameters.Count, i => 
#endif
            {
                for (int j = 0; j < this.Parameters[i].Length; j++)
                {
                    double grad = this.Parameters[i].Grad.Data[j];
                    this.ms[i][j] *= this.alpha;
                    this.ms[i][j] += (1 - this.alpha) * grad * grad;

                    this.Parameters[i].Param.Data[j] -= this.lr * grad / (Math.Sqrt(this.ms[i][j]) + this.eps);
                }
            }
#if !DEBUG
            );
#endif
        }

        protected override void Initialize()
        {
            this.ms = new double[this.Parameters.Count][];

            for (int i = 0; i < this.ms.Length; i++)
            {
                this.ms[i] = new double[this.Parameters[i].Param.Length];
            }
        }
    }
}
