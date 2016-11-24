using System;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
    public class MomentumSGD : Optimizer
    {
        private double LearningRate;
        private double momentum;

        private double[][] v;

        public MomentumSGD(double learningRate = 0.01, double momentum = 0.9)
        {
            this.LearningRate = learningRate;
            this.momentum = momentum;
        }

        protected override void DoUpdate()
        {
#if DEBUG
            for (int i = 0; i < Parameters.Count; i++)
#else
            Parallel.For(0, this.Parameters.Count, i =>
#endif
            {
                for (int k = 0; k < this.Parameters[i].Length; k++)
                {
                    this.v[i][k] *= this.momentum;
                    this.v[i][k] -= this.LearningRate * this.Parameters[i].Grad.Data[k];

                    this.Parameters[i].Param.Data[k] += this.v[i][k];
                }
            }
#if !DEBUG
            );
#endif
        }

        protected override void Initialize()
        {
            this.v = new double[this.Parameters.Count][];

            for (int i = 0; i < this.v.Length; i++)
            {
                this.v[i] = new double[this.Parameters[i].Param.Length];
            }
        }
    }
}
