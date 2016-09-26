using System.Threading.Tasks;

namespace KelpNet.Optimizers
{
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
            Parallel.For(0, Parameters.Count, i =>
            {
                for (int k = 0; k < Parameters[i].Length; k++)
                {
                    this.v[i].Data[k] *= this.momentum;
                    this.v[i].Data[k] -= this.LearningRate*Parameters[i].Grad.Data[k];

                    Parameters[i].Param.Data[k] += this.v[i].Data[k];
                }
            });
        }

        protected override void Initialize()
        {
            this.v = new NdArray[Parameters.Count];

            for (int i = 0; i < v.Length; i++)
            {
                this.v[i] = NdArray.ZerosLike(Parameters[i].Param);
            }
        }
    }
}
