using System;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
    public class AdaDelta : Optimizer
    {
        private double[][] msg;
        private double[][] msdx;

        private double rho;
        private double eps;

        public AdaDelta(double rho = 0.95, double eps = 1e-6)
        {
            this.rho = rho;
            this.eps = eps;
        }

        protected override void DoUpdate()
        {
#if DEBUG
            for (int i = 0; i < Parameters.Count; i++)
#else
            Parallel.For(0, Parameters.Count, i => 
#endif
            {
                for (int j = 0; j < Parameters[i].Length; j++)
                {
                    double grad = Parameters[i].Grad.Data[j];
                    this.msg[i][j] *= this.rho;
                    this.msg[i][j] += (1 - this.rho) * grad * grad;

                    double dx = Math.Sqrt((this.msdx[i][j] + this.eps) / (this.msg[i][j] + this.eps)) * grad;

                    this.msdx[i][j] *= this.rho;
                    this.msdx[i][j] += (1 - this.rho) * dx * dx;

                    Parameters[i].Param.Data[j] -= dx;
                }
            }
#if !DEBUG
            );
#endif
        }

        protected override void Initialize()
        {
            this.msg = new double[Parameters.Count][];
            this.msdx = new double[Parameters.Count][];

            for (int i = 0; i < Parameters.Count; i++)
            {
                this.msg[i] = new double[Parameters[i].Param.Length];
                this.msdx[i] = new double[Parameters[i].Param.Length];
            }
        }
    }
}
