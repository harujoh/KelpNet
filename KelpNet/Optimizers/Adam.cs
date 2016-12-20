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

        internal override void AddFunctionParameters(FunctionParameter[] functionParameters)
        {
            foreach (FunctionParameter functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdamParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    class AdamParameter : OptimizerParameter
    {
        private readonly Adam optimiser;

        private readonly double[] m;
        private readonly double[] v;

        public AdamParameter(FunctionParameter parameter, Adam optimiser) : base(parameter)
        {
            this.m = new double[parameter.Length];
            this.v = new double[parameter.Length];

            this.optimiser = optimiser;
        }

        public override void UpdateFunctionParameters()
        {
            double fix1 = 1 - Math.Pow(this.optimiser.Beta1, this.optimiser.UpdateCount);
            double fix2 = 1 - Math.Pow(this.optimiser.Beta2, this.optimiser.UpdateCount);
            double lr = this.optimiser.Alpha * Math.Sqrt(fix2) / fix1;

            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                double grad = this.FunctionParameter.Grad.Data[i];

                this.m[i] += (1 - this.optimiser.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this.optimiser.Beta2) * (grad * grad - this.v[i]);

                this.FunctionParameter.Param.Data[i] -= lr * this.m[i] / (Math.Sqrt(this.v[i]) + this.optimiser.Epsilon);
            }
        }
    }

}
