using System;
using System.Threading.Tasks;
using KelpNet.Common;

namespace KelpNet.Optimizers
{
    public class RMSprop : Optimizer
    {
        private NdArray[] ms;

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
            Parallel.For(0, Parameters.Count, i =>
            {
                for (int k = 0; k < Parameters[i].Length; k++)
                {
                    var grad = Parameters[i].Grad.Data[k];
                    this.ms[i].Data[k] *= this.alpha;
                    this.ms[i].Data[k] += (1 - this.alpha) * grad * grad;

                    Parameters[i].Param.Data[k] -= this.lr * grad / (Math.Sqrt(this.ms[i].Data[k]) + this.eps);
                }
            });
        }

        protected override void Initialize()
        {
            this.ms = new NdArray[Parameters.Count];

            for (int i = 0; i < ms.Length; i++)
            {
                this.ms[i] = NdArray.ZerosLike(Parameters[i].Param);
            }
        }
    }
}
