using System;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
    public class GradientClipping : Optimizer
    {
        public double Threshold;

        public GradientClipping(OptimizeParameter[] parameters, double threshold) : base(parameters)
        {
            this.Threshold = threshold;
        }

        protected override void DoUpdate()
        {
            //_sum_sqnorm
            double s = 0.0;

            for (int i = 0; i < this.Parameters.Length; i++)
            {
                for (int j = 0; j < this.Parameters[i].Length; j++)
                {
                    s += Math.Pow(this.Parameters[i].Grad.Data[j], 2);
                }
            }

            double norm = Math.Sqrt(s);
            double rate = this.Threshold / norm;

            if (rate < 1)
            {
#if DEBUG
                for (int i = 0; i < this.Parameters.Length; i++)
#else
                Parallel.For(0, this.Parameters.Length, i =>
#endif
                {
                    OptimizeParameter parameter = this.Parameters[i];

                    for (int j = 0; j < parameter.Length; j++)
                    {
                        parameter.Grad.Data[j] *= rate;
                    }
                }
#if !DEBUG
                );
#endif
            }

        }
    }
}
