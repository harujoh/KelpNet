using System.Collections.Generic;
using KelpNet.Functions;

namespace KelpNet.Optimizers
{
    public class SGD : Optimizer
    {
        public double LearningRate;

        public SGD(double learningRate=0.1)
        {
            this.LearningRate = learningRate;
        }

        protected override void DoUpdate(List<OptimizableFunction> optimizableFunctions)
        {
            foreach (var optimizableFunction in optimizableFunctions)
            {
                for (int i = 0; i < optimizableFunction.W.Length; i++)
                {
                    optimizableFunction.W.Data[i] -= this.LearningRate * optimizableFunction.gW.Data[i];
                }

                if (optimizableFunction.b != null)
                {
                    for (int i = 0; i < optimizableFunction.b.Length; i++)
                    {
                        optimizableFunction.b.Data[i] -= this.LearningRate*optimizableFunction.gb.Data[i];
                    }
                }
            }
        }

        public override void Initialize(FunctionStack fs)
        {
            //特に初期化処理は行わない
        }
    }
}
