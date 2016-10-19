#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    public class SGD : Optimizer
    {
        public double LearningRate;

        public SGD(double learningRate = 0.1)
        {
            this.LearningRate = learningRate;
        }

        protected override void DoUpdate()
        {
#if DEBUG
            foreach (var parameter in Parameters)
#else
            Parallel.ForEach(Parameters, parameter =>
#endif
            {
                for (int j = 0; j < parameter.Length; j++)
                {
                    parameter.Param.Data[j] -= this.LearningRate * parameter.Grad.Data[j];
                }
            }
#if !DEBUG
            );
#endif
        }
    }
}
