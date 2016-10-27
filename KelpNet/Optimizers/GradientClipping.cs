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

            foreach (var parameter in Parameters)
            {
                for (int i = 0; i < parameter.Length; i++)
                {
                    s += Math.Pow(parameter.Grad.Data[i], 2);
                }
            }

            var norm = Math.Sqrt(s);
            var rate = this.Threshold / norm;

            if (rate < 1)
            {
#if DEBUG
                foreach (var parameter in Parameters)
#else
                Parallel.ForEach(Parameters, parameter =>
#endif
                {
                    for (int i = 0; i < parameter.Length; i++)
                    {
                        parameter.Grad.Data[i] *= rate;
                    }
                }
#if !DEBUG
                );
#endif

            }

        }
    }
}
