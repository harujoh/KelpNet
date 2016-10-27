using System;
using KelpNet.Common;
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

        private NdArray[] v;

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
            Parallel.For(0, Parameters.Count, i => 
#endif
            {
                for (int k = 0; k < Parameters[i].Length; k++)
                {
                    this.v[i].Data[k] *= this.momentum;
                    this.v[i].Data[k] -= this.LearningRate*Parameters[i].Grad.Data[k];

                    Parameters[i].Param.Data[k] += this.v[i].Data[k];
                }
            }
#if !DEBUG
            );
#endif
        }

        protected override void Initialize()
        {
            this.v = new NdArray[Parameters.Count];

            for (int i = 0; i < this.v.Length; i++)
            {
                this.v[i] = NdArray.ZerosLike(Parameters[i].Param);
            }
        }
    }
}
