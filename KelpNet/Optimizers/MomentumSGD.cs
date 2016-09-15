using System.Threading.Tasks;

namespace KelpNet.Optimizers
{
    public class MomentumSGD : Optimizer
    {
        private double LearningRate;
        private double momentum;

        private NdArray[][] v;

        public MomentumSGD(double learningRate = 0.01, double momentum = 0.9)
        {
            this.LearningRate = learningRate;
            this.momentum = momentum;
        }

        protected override void DoUpdate(Function[] functions)
        {
            Parallel.For(0, functions.Length, i =>
            {
                for (int j = 0; j < functions[i].Parameters.Count; j++)
                {
                    for (int k = 0; k < functions[i].Parameters[j].Length; k++)
                    {
                        this.v[i][j].Data[k] *= this.momentum;
                        this.v[i][j].Data[k] -= this.LearningRate*functions[i].Parameters[j].Grad.Data[k];

                        functions[i].Parameters[j].Param.Data[k] += this.v[i][j].Data[k];
                    }
                }
            });
        }

        protected override void Initialize(Function[] functions)
        {
            this.v = new NdArray[functions.Length][];

            for (int i = 0; i < functions.Length; i++)
            {
                this.v[i] = new NdArray[functions[i].Parameters.Count];

                for (int j = 0; j < functions[i].Parameters.Count; j++)
                {
                    this.v[i][j] = NdArray.ZerosLike(functions[i].Parameters[j].Param);
                }
            }
        }
    }
}
