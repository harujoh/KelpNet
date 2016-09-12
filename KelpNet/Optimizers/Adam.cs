using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using KelpNet.Functions;

namespace KelpNet.Optimizers
{
    public class Adam : Optimizer
    {
        private double alpha;
        private double beta1;
        private double beta2;
        private double eps;

        double lr
        {
            get
            {
                double fix1 = 1 - Math.Pow(this.beta1, t);
                double fix2 = 1 - Math.Pow(this.beta2, t);
                return this.alpha * Math.Sqrt(fix2) / fix1;
            }
        }

        private NdArray[][] m;
        private NdArray[][] v;

        public Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
        {
            this.alpha = alpha;
            this.beta1 = beta1;
            this.beta2 = beta2;
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
                        double grad = functions[i].Parameters[j].Grad.Data[k];

                        this.m[i][j].Data[k] += (1 - this.beta1) * (grad - this.m[i][j].Data[k]);
                        this.v[i][j].Data[k] += (1 - this.beta2) * (grad * grad - this.v[i][j].Data[k]);

                        functions[i].Parameters[j].Param.Data[k] -= this.lr *this.m[i][j].Data[k] /
                                                                    (Math.Sqrt(this.v[i][j].Data[k]) + this.eps);
                    }
                }
            });
        }

        public override void Initialize(FunctionStack fs)
        {
            this.m = new NdArray[fs.Functions.Count][];
            this.v = new NdArray[fs.Functions.Count][];

            for (int i = 0; i < fs.Functions.Count; i++)
            {
                this.m[i] = new NdArray[fs.Functions[i].Parameters.Count];
                this.v[i] = new NdArray[fs.Functions[i].Parameters.Count];

                for (int j = 0; j < fs.Functions[i].Parameters.Count; j++)
                {
                    this.m[i][j] = NdArray.ZerosLike(fs.Functions[i].Parameters[j].Param);
                    this.v[i][j] = NdArray.ZerosLike(fs.Functions[i].Parameters[j].Param);
                }
            }
        }
    }
}
