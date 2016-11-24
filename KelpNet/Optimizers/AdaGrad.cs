using System;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
    public class AdaGrad : Optimizer
    {
        private double[][] h;

        private double lr;
        private double eps;

        public AdaGrad(double lr = 0.01, double eps = 1e-8)
        {
            this.lr = lr;
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
                for (int k = 0; k < this.Parameters[i].Length; k++)
                {
                    double grad = this.Parameters[i].Grad.Data[k];

                    this.h[i][k] += grad * grad;

                    this.Parameters[i].Param.Data[k] -= this.lr * grad / (Math.Sqrt(this.h[i][k]) + this.eps);

                }
            }
#if !DEBUG
            );
#endif
        }

        protected override void Initialize()
        {
            this.h = new double[this.Parameters.Count][];

            for (int i = 0; i < this.h.Length; i++)
            {
                this.h[i] = new double[this.Parameters[i].Param.Length];
            }

        }
    }
}
