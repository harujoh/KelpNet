using System;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
    public class Adam : Optimizer
    {
        private double alpha;
        private double beta1;
        private double beta2;
        private double eps;

        private double[][] m;
        private double[][] v;

        public Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
        {
            this.alpha = alpha;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
        }

        protected override void DoUpdate()
        {
            double fix1 = 1 - Math.Pow(this.beta1, this.UpdateCount);
            double fix2 = 1 - Math.Pow(this.beta2, this.UpdateCount);
            double lr = this.alpha * Math.Sqrt(fix2) / fix1;

#if DEBUG
            for (int i = 0; i < Parameters.Count; i++)
#else
            Parallel.For(0, this.Parameters.Count, i => 
#endif
            {
                for (int j = 0; j < this.Parameters[i].Length; j++)
                {
                    double grad = this.Parameters[i].Grad.Data[j];

                    this.m[i][j] += (1 - this.beta1) * (grad - this.m[i][j]);
                    this.v[i][j] += (1 - this.beta2) * (grad * grad - this.v[i][j]);

                    this.Parameters[i].Param.Data[j] -= lr *this.m[i][j] / (Math.Sqrt(this.v[i][j]) + this.eps);
                }
            }
#if !DEBUG
            );
#endif
        }

        protected override void Initialize()
        {
            this.m = new double[this.Parameters.Count][];
            this.v = new double[this.Parameters.Count][];

            for (int i = 0; i < this.Parameters.Count; i++)
            {
                this.m[i] = new double[this.Parameters[i].Param.Length];
                this.v[i] = new double[this.Parameters[i].Param.Length];
            }
        }
    }
}
