using System;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class AdaDelta : IOptimizer
    {
        public double rho;
        public double Epsilon;

        private readonly double[] msg;
        private readonly double[] msdx;

        public AdaDelta(double rho = 0.95, double epsilon = 1e-6, int parameterLength = 0)
        {
            this.rho = rho;
            this.Epsilon = epsilon;

            this.msg = new double[parameterLength];
            this.msdx = new double[parameterLength];
        }

        public IOptimizer Initialise(OptimizeParameter parameter)
        {
            return new AdaDelta(this.rho, this.Epsilon, parameter.Length);
        }

        public void Update(OptimizeParameter parameter)
        {
            for (int i = 0; i < parameter.Length; i++)
            {
                double grad = parameter.Grad.Data[i];
                this.msg[i] *= this.rho;
                this.msg[i] += (1 - this.rho) * grad * grad;

                double dx = Math.Sqrt((this.msdx[i] + this.Epsilon) / (this.msg[i] + this.Epsilon)) * grad;

                this.msdx[i] *= this.rho;
                this.msdx[i] += (1 - this.rho) * dx * dx;

                parameter.Param.Data[i] -= dx;
            }
        }
    }
}
