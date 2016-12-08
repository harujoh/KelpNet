using System;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class Adam : Optimizer
    {
        public double Alpha;
        public double Beta1;
        public double Beta2;
        public double Epsilon;

        public Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            this.Alpha = alpha;
            this.Beta1 = beta1;
            this.Beta2 = beta2;
            this.Epsilon = epsilon;
        }

        public override void Initilise(OptimizeParameter[] functionParameters)
        {
            this.OptimizerParameters = new OptimizerParameter[functionParameters.Length];

            for (int i = 0; i < this.OptimizerParameters.Length; i++)
            {
                this.OptimizerParameters[i] = new AdamParameter(functionParameters[i], this);
            }
        }
    }

    [Serializable]
    class AdamParameter : OptimizerParameter
    {
        private readonly Adam optimiser;

        private readonly double[] m;
        private readonly double[] v;

        public AdamParameter(OptimizeParameter parameter, Adam optimiser) : base(parameter)
        {
            this.m = new double[parameter.Length];
            this.v = new double[parameter.Length];

            this.optimiser = optimiser;
        }

        public override void Update()
        {
            double fix1 = 1 - Math.Pow(this.optimiser.Beta1, this.optimiser.UpdateCount);
            double fix2 = 1 - Math.Pow(this.optimiser.Beta2, this.optimiser.UpdateCount);
            double lr = this.optimiser.Alpha * Math.Sqrt(fix2) / fix1;

            for (int i = 0; i < FunctionParameters.Length; i++)
            {
                double grad = FunctionParameters.Grad.Data[i];

                this.m[i] += (1 - this.optimiser.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this.optimiser.Beta2) * (grad * grad - this.v[i]);

                this.FunctionParameters.Param.Data[i] -= lr * this.m[i] / (Math.Sqrt(this.v[i]) + this.optimiser.Epsilon);
            }
        }
    }

}
