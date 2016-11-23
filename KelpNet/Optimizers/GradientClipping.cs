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

        public GradientClipping(double threshold)
        {
            this.Threshold = threshold;
        }

        protected override void DoUpdate()
        {
            //_sum_sqnorm
            double s = 0.0;

            for (int i = 0; i < Parameters.Count; i++)
            {
                for (int j = 0; j < Parameters[i].Length; j++)
                {
                    s += Math.Pow(Parameters[i].Grad.Data[j], 2);
                }
            }

            var norm = Math.Sqrt(s);
            var rate = this.Threshold / norm;

            if (rate < 1)
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
