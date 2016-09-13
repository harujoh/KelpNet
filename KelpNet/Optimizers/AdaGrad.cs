using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace KelpNet.Optimizers
{
    public class AdaGrad : Optimizer
    {
        private NdArray[][] h;

        private double lr;
        private double eps;

        public AdaGrad(double lr = 0.01, double eps = 1e-8)
        {
            this.lr = lr;
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

                        this.h[i][j].Data[k] += grad * grad;

                        functions[i].Parameters[j].Param.Data[k] -= this.lr * grad / (Math.Sqrt(this.h[i][j].Data[k]) + this.eps);
                    }
                }
            });
        }

        public override void Initialize(FunctionStack fs)
        {
            this.h = new NdArray[fs.Functions.Count][];

            for (int i = 0; i < fs.Functions.Count; i++)
            {
                this.h[i] = new NdArray[fs.Functions[i].Parameters.Count];

                for (int j = 0; j < fs.Functions[i].Parameters.Count; j++)
                {
                    this.h[i][j] = NdArray.ZerosLike(fs.Functions[i].Parameters[j].Param);
                }
            }
        }
    }
}
