using System.Collections.Generic;
using System.Threading.Tasks;

namespace KelpNet.Optimizers
{
    public class SGD : Optimizer
    {
        public double LearningRate;

        public SGD(double learningRate=0.1)
        {
            this.LearningRate = learningRate;
        }

        protected override void DoUpdate(List<Function> functions)
        {
            Parallel.ForEach(functions, function =>
            {
                foreach (Function.Parameter parameter in function.Parameters)
                {
                    for (int j = 0; j < parameter.Length; j++)
                    {
                        parameter.Param.Data[j] -= this.LearningRate*parameter.Grad.Data[j];
                    }
                }
            });
        }

        public override void Initialize(FunctionStack fs)
        {
            //特に初期化処理は行わない
        }
    }
}
