using System;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
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
                for (int i = 0; i < parameter.Length; i++)
                {
                    parameter.Param.Data[i] -= this.LearningRate * parameter.Grad.Data[i];
                }
            }
#if !DEBUG
            );
#endif
        }
    }
}
