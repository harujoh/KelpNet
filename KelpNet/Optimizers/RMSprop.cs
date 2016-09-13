using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace KelpNet.Optimizers
{
    public class RMSprop : Optimizer
    {
        private NdArray[][] ms;

        private double lr;
        private double alpha;
        private double eps;

        public RMSprop(double lr = 0.01,double alpha = 0.99,double eps = 1e-8)
        {
            this.lr = lr;
            this.alpha = alpha;
            this.eps = eps;
        }

        protected override void DoUpdate(List<Function> functions)
        {
            Parallel.For(0, functions.Count, i =>
            {
                for (int j = 0; j < functions[i].Parameters.Count; j++)
                {
                    for (int k = 0; k < functions[i].Parameters[j].Length; k++)
                    {
                        var grad = functions[i].Parameters[j].Grad.Data[k];
                        this.ms[i][j].Data[k] *= this.alpha;
                        this.ms[i][j].Data[k] += (1 - this.alpha) * grad * grad;

                        functions[i].Parameters[j].Param.Data[k] -= this.lr * grad / (Math.Sqrt(this.ms[i][j].Data[k]) + this.eps);
                    }
                }
            });
        }

        public override void Initialize(FunctionStack fs)
        {
            this.ms = new NdArray[fs.Functions.Count][];

            for (int i = 0; i < fs.Functions.Count; i++)
            {
                this.ms[i] = new NdArray[fs.Functions[i].Parameters.Count];

                for (int j = 0; j < fs.Functions[i].Parameters.Count; j++)
                {
                    this.ms[i][j] = NdArray.ZerosLike(fs.Functions[i].Parameters[j].Param);
                }
            }
        }
    }
}
