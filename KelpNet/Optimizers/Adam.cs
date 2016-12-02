using System;
#if !DEBUG
using System.Threading.Tasks;
#endif

namespace KelpNet.Optimizers
{
    [Serializable]
    public class Adam : Optimizer
    {
        public double Alpha;
        public double Beta1;
        public double Beta2;
        public double Epsilon;

        private double[][] m;
        private double[][] v;

        public Adam(OptimizeParameter[] parameters, double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8) : base(parameters)
        {
            this.Alpha = alpha;
            this.Beta1 = beta1;
            this.Beta2 = beta2;
            this.Epsilon = epsilon;

            this.m = new double[parameters.Length][];
            this.v = new double[parameters.Length][];

            for (int i = 0; i < parameters.Length; i++)
            {
                this.m[i] = new double[parameters[i].Param.Length];
                this.v[i] = new double[parameters[i].Param.Length];
            }
        }

        protected override void DoUpdate()
        {
            double fix1 = 1 - Math.Pow(this.Beta1, this.UpdateCount);
            double fix2 = 1 - Math.Pow(this.Beta2, this.UpdateCount);
            double lr = this.Alpha * Math.Sqrt(fix2) / fix1;

#if DEBUG
            for (int i = 0; i < this.Parameters.Length; i++)
#else
            Parallel.For(0, this.Parameters.Length, i =>
#endif
            {
                for (int j = 0; j < this.Parameters[i].Length; j++)
                {
                    double grad = this.Parameters[i].Grad.Data[j];

                    this.m[i][j] += (1 - this.Beta1) * (grad - this.m[i][j]);
                    this.v[i][j] += (1 - this.Beta2) * (grad * grad - this.v[i][j]);

                    this.Parameters[i].Param.Data[j] -= lr * this.m[i][j] / (Math.Sqrt(this.v[i][j]) + this.Epsilon);
                }
            }
#if !DEBUG
            );
#endif
        }
    }
}
