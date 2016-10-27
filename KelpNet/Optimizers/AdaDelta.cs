using System;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
    public class AdaDelta : Optimizer
    {
        private NdArray[] msg;
        private NdArray[] msdx;

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
                    var grad = Parameters[i].Grad.Data[j];
                    this.msg[i].Data[j] *= this.rho;
                    this.msg[i].Data[j] += (1 - this.rho) * grad * grad;

                    var dx = Math.Sqrt((this.msdx[i].Data[j] + this.eps) / (this.msg[i].Data[j] + this.eps)) * grad;

                    this.msdx[i].Data[j] *= this.rho;
                    this.msdx[i].Data[j] += (1 - this.rho) * dx * dx;

                    Parameters[i].Param.Data[j] -= dx;
                }
            }
#if !DEBUG
            );
#endif
        }

        protected override void Initialize()
        {
            this.msg = new NdArray[Parameters.Count];
            this.msdx = new NdArray[Parameters.Count];

            for (int i = 0; i < Parameters.Count; i++)
            {
                this.msg[i] = NdArray.ZerosLike(Parameters[i].Param);
                this.msdx[i] = NdArray.ZerosLike(Parameters[i].Param);
            }
        }
    }
}
