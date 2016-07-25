using System;
using System.Collections.Generic;
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
                double fix1 = 1 - Math.Pow(this.beta1, this.t);
                double fix2 = 1 - Math.Pow(this.beta2, this.t);
                return this.alpha * Math.Sqrt(fix2) / fix1;
            }
        }

        private NdArray[] mW;
        private NdArray[] vW;

        private NdArray[] mb;
        private NdArray[] vb;

        public Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
        {
            this.alpha = alpha;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
        }

        protected override void DoUpdate(List<OptimizableFunction> optimizableFunctions)
        {
            for (int i = 0; i < optimizableFunctions.Count; i++)
            {
                for (int j = 0; j < optimizableFunctions[i].W.Length; j++)
                {
                    double gradW = optimizableFunctions[i].gW.Data[j];

                    mW[i].Data[j] += (1 - this.beta1) * (gradW - mW[i].Data[j]);
                    vW[i].Data[j] += (1 - this.beta2) * (gradW * gradW - vW[i].Data[j]);

                    optimizableFunctions[i].W.Data[j] -= lr * mW[i].Data[j] / (Math.Sqrt(vW[i].Data[j]) + this.eps);
                }

                if (optimizableFunctions[i].b != null)
                {
                    for (int j = 0; j < optimizableFunctions[i].b.Length; j++)
                    {
                        double gradb = optimizableFunctions[i].gb.Data[j];

                        mb[i].Data[j] += (1 - this.beta1) * (gradb - mb[i].Data[j]);
                        vb[i].Data[j] += (1 - this.beta2) * (gradb * gradb - vb[i].Data[j]);

                        optimizableFunctions[i].b.Data[j] -= lr * mb[i].Data[j] / (Math.Sqrt(vb[i].Data[j]) + this.eps);
                    }
                }
            }
        }

        public override void Initialize(FunctionStack fs)
        {
            this.mW = new NdArray[fs.OptimizableFunctions.Count];
            this.vW = new NdArray[fs.OptimizableFunctions.Count];

            this.mb = new NdArray[fs.OptimizableFunctions.Count];
            this.vb = new NdArray[fs.OptimizableFunctions.Count];

            for (int i = 0; i < fs.OptimizableFunctions.Count; i++)
            {
                this.mW[i] = NdArray.ZerosLike(fs.OptimizableFunctions[i].W);
                this.vW[i] = NdArray.ZerosLike(fs.OptimizableFunctions[i].W);

                if (fs.OptimizableFunctions[i].b != null)
                {
                    this.mb[i] = NdArray.ZerosLike(fs.OptimizableFunctions[i].b);
                    this.vb[i] = NdArray.ZerosLike(fs.OptimizableFunctions[i].b);
                }
            }
        }
    }
}
