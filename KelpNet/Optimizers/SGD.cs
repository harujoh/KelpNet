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
            for (int i = 0; i < this.Parameters.Count; i++)
#else
            Parallel.For(0, this.Parameters.Count, i =>
#endif
            {
                var parameter = Parameters[i];
                for (int j = 0; j < parameter.Length; j++)
                {
                    parameter.Param.Data[j] -= this.LearningRate*parameter.Grad.Data[j];
                }
            }
#if !DEBUG
            );
#endif
        }
    }
}
