using System;
using KelpNet.Common;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
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
#if DEBUG
            for (int i = 0; i < Parameters.Count; i++)
#else
            Parallel.For(0, Parameters.Count, i => 
#endif
            {
                for (int j = 0; j < Parameters[i].Length; j++)
                {
                    var grad = Parameters[i].Grad.Data[j];
                    this.ms[i].Data[j] *= this.alpha;
                    this.ms[i].Data[j] += (1 - this.alpha) * grad * grad;

                    Parameters[i].Param.Data[j] -= this.lr * grad / (Math.Sqrt(this.ms[i].Data[j]) + this.eps);
                }
            }
#if !DEBUG
            );
#endif
        }

        protected override void Initialize()
        {
            this.ms = new NdArray[Parameters.Count];

            for (int i = 0; i < this.ms.Length; i++)
            {
                this.ms[i] = NdArray.ZerosLike(Parameters[i].Param);
            }
        }
    }
}
