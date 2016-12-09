using System;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class AdaDelta : Optimizer
    {
        public double Rho;
        public double Epsilon;

        public AdaDelta(double rho = 0.95, double epsilon = 1e-6)
        {
            this.Rho = rho;
            this.Epsilon = epsilon;
        }

        public override void Initilise(FunctionParameter[] functionParameters)
        {
            this.OptimizerParameters = new OptimizerParameter[functionParameters.Length];

            for (int i = 0; i < this.OptimizerParameters.Length; i++)
            {
                this.OptimizerParameters[i] = new AdaDeltaParameter(functionParameters[i], this);
            }
        }
    }

    [Serializable]
    class AdaDeltaParameter : OptimizerParameter
    {
        private readonly double[] msg;
        private readonly double[] msdx;
        private readonly AdaDelta optimiser;

        public AdaDeltaParameter(FunctionParameter functionParameter, AdaDelta optimiser) : base(functionParameter)
        {
            this.msg = new double[functionParameter.Length];
            this.msdx = new double[functionParameter.Length];
            this.optimiser = optimiser;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                double grad = this.FunctionParameter.Grad.Data[i];
                this.msg[i] *= this.optimiser.Rho;
                this.msg[i] += (1 - this.optimiser.Rho) * grad * grad;

                double dx = Math.Sqrt((this.msdx[i] + this.optimiser.Epsilon) / (this.msg[i] + this.optimiser.Epsilon)) * grad;

                this.msdx[i] *= this.optimiser.Rho;
                this.msdx[i] += (1 - this.optimiser.Rho) * dx * dx;

                this.FunctionParameter.Param.Data[i] -= dx;
            }
        }
    }
}
