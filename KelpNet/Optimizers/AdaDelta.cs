using System;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
    public class AdaDelta : Optimizer
    {
        private readonly double[][] msg;
        private readonly double[][] msdx;

        public double rho;
        public double Epsilon;

        public AdaDelta(OptimizeParameter[] parameters, double rho = 0.95, double epsilon = 1e-6) : base(parameters)
        {
            this.rho = rho;
            this.Epsilon = epsilon;

            this.msg = new double[parameters.Length][];
            this.msdx = new double[parameters.Length][];

            for (int i = 0; i < parameters.Length; i++)
            {
                this.msg[i] = new double[parameters[i].Param.Length];
                this.msdx[i] = new double[parameters[i].Param.Length];
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
                    this.msg[i][j] *= this.rho;
                    this.msg[i][j] += (1 - this.rho) * grad * grad;

                    double dx = Math.Sqrt((this.msdx[i][j] + this.Epsilon) / (this.msg[i][j] + this.Epsilon)) * grad;

                    this.msdx[i][j] *= this.rho;
                    this.msdx[i][j] += (1 - this.rho) * dx * dx;

                    this.Parameters[i].Param.Data[j] -= dx;
                }
            }
#if !DEBUG
            );
#endif
        }
    }
}
