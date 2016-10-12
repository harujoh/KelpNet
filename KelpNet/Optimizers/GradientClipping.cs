using System;
using System.Threading.Tasks;

namespace KelpNet.Optimizers
{
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
                for (int j = 0; j < parameter.Length; j++)
                {
                    s += Math.Pow(parameter.Grad.Data[j], 2);
                }
            }

            var norm = Math.Sqrt(s);
            var rate = this.Threshold / norm;

            if (rate < 1)
            {
                Parallel.ForEach(Parameters, parameter =>
                {
                    for (int j = 0; j < parameter.Length; j++)
                    {
                        parameter.Grad.Data[j] *= rate;
                    }
                });

            }

        }
    }
}
