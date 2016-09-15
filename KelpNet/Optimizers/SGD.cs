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

        protected override void DoUpdate(Function[] functions)
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

        protected override void Initialize(Function[] functions)
        {
            //特に初期化処理は行わない
        }
    }
}
