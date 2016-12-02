using System;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class Adam : IOptimizer
    {
        public double Alpha;
        public double Beta1;
        public double Beta2;
        public double Epsilon;

        private long UpdateCount = 1;

        private readonly double[] m;
        private readonly double[] v;

        public Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, int parameterLength = 0)
        {
            this.Alpha = alpha;
            this.Beta1 = beta1;
            this.Beta2 = beta2;
            this.Epsilon = epsilon;

            this.m = new double[parameterLength];
            this.v = new double[parameterLength];
        }

        public IOptimizer Initialise(OptimizeParameter parameter)
        {
            return new Adam(this.Alpha, this.Beta1, this.Beta2, this.Epsilon, parameter.Length);
        }

        public void Update(OptimizeParameter parameter)
        {
            double fix1 = 1 - Math.Pow(this.Beta1, this.UpdateCount);
            double fix2 = 1 - Math.Pow(this.Beta2, this.UpdateCount);
            double lr = this.Alpha * Math.Sqrt(fix2) / fix1;

            for (int i = 0; i < parameter.Length; i++)
            {
                double grad = parameter.Grad.Data[i];

                this.m[i] += (1 - this.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this.Beta2) * (grad * grad - this.v[i]);

                parameter.Param.Data[i] -= lr * this.m[i] / (Math.Sqrt(this.v[i]) + this.Epsilon);
            }

            this.UpdateCount++;
        }
    }
}
