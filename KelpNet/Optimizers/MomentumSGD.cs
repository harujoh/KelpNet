using System;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
    public class MomentumSGD : Optimizer
    {
        public double LearningRate;
        public double Momentum;

        private double[][] v;

        public MomentumSGD(OptimizeParameter[] parameters, double learningRate = 0.01, double momentum = 0.9) : base(parameters)
        {
            this.LearningRate = learningRate;
            this.Momentum = momentum;

            this.v = new double[parameters.Length][];
            for (int i = 0; i < this.v.Length; i++)
            {
                this.v[i] = new double[parameters[i].Param.Length];
            }
        }

        protected override void DoUpdate()
        {
#if DEBUG
            for (int i = 0; i < this.Parameters.Length; i++)
#else
            Parallel.For(0, this.Parameters.Length, i =>
#endif
            {
                for (int k = 0; k < this.Parameters[i].Length; k++)
                {
                    this.v[i][k] *= this.Momentum;
                    this.v[i][k] -= this.LearningRate * this.Parameters[i].Grad.Data[k];

                    this.Parameters[i].Param.Data[k] += this.v[i][k];
                }
            }
#if !DEBUG
            );
#endif
        }
    }
}
