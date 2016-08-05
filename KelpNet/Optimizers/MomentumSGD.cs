using System.Collections.Generic;
using KelpNet.Functions;

namespace KelpNet.Optimizers
{
    public class MomentumSGD : Optimizer
    {
        private double LearningRate;
        private double momentum;

        private NdArray[][] v;

        public MomentumSGD(double learningRate = 0.01, double momentum = 0.9)
        {
            this.LearningRate = learningRate;
            this.momentum = momentum;
        }

        protected override void DoUpdate(List<OptimizableFunction> optimizableFunctions)
        {
            for (int i = 0; i < optimizableFunctions.Count; i++)
            {
                for (int j = 0; j < optimizableFunctions[i].Parameters.Count; j++)
                {
                    for (int k = 0; k < optimizableFunctions[i].Parameters[j].Length; k++)
                    {
                        v[i][j].Data[k] *= this.momentum;
                        v[i][j].Data[k] -= this.LearningRate*optimizableFunctions[i].Parameters[j].Grad.Data[k];

                        optimizableFunctions[i].Parameters[j].Param.Data[k] += v[i][j].Data[k];
                    }
                }
            }
        }

        public override void Initialize(FunctionStack fs)
        {
            this.v = new NdArray[fs.OptimizableFunctions.Count][];

            for (int i = 0; i < fs.OptimizableFunctions.Count; i++)
            {
                this.v[i] = new NdArray[fs.OptimizableFunctions[i].Parameters.Count];

                for (int j = 0; j < fs.OptimizableFunctions[i].Parameters.Count; j++)
                {
                    this.v[i][j] = NdArray.ZerosLike(fs.OptimizableFunctions[i].Parameters[j].Param);
                }
            }
        }
    }
}
