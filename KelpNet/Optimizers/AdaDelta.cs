using System;
using System.Threading.Tasks;

namespace KelpNet.Optimizers
{
    public class AdaDelta : Optimizer
    {
        private NdArray[][] msg;
        private NdArray[][] msdx;

        private double rho;
        private double eps;

        public AdaDelta(double rho= 0.95, double eps = 1e-6)
        {
            this.rho = rho;
            this.eps = eps;
        }

        protected override void DoUpdate(Function[] functions)
        {
            Parallel.For(0, functions.Length, i =>
            {
                for (int j = 0; j < functions[i].Parameters.Count; j++)
                {
                    for (int k = 0; k < functions[i].Parameters[j].Length; k++)
                    {
                        var grad = functions[i].Parameters[j].Grad.Data[k];
                        this.msg[i][j].Data[k] *= this.rho;
                        this.msg[i][j].Data[k] += (1 - this.rho) * grad * grad;

                        var dx = Math.Sqrt((this.msdx[i][j].Data[k] + this.eps)/(this.msg[i][j].Data[k] + this.eps))* grad;

                        this.msdx[i][j].Data[k] *= this.rho;
                        this.msdx[i][j].Data[k] += (1 - this.rho) * dx * dx;

                        functions[i].Parameters[j].Param.Data[k] -= dx;
                    }
                }
            });
        }

        protected override void Initialize(Function[] functions)
        {
            this.msg = new NdArray[functions.Length][];
            this.msdx = new NdArray[functions.Length][];

            for (int i = 0; i < functions.Length; i++)
            {
                this.msg[i] = new NdArray[functions[i].Parameters.Count];
                this.msdx[i] = new NdArray[functions[i].Parameters.Count];

                for (int j = 0; j < functions[i].Parameters.Count; j++)
                {
                    this.msg[i][j] = NdArray.ZerosLike(functions[i].Parameters[j].Param);
                    this.msdx[i][j] = NdArray.ZerosLike(functions[i].Parameters[j].Param);
                }
            }
        }
    }
}
