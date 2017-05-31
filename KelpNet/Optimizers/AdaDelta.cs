using System;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Optimizers;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class AdaDelta : Optimizer
    {
        public Real Rho;
        public Real Epsilon;

        public AdaDelta(Real? rho = null, Real? epsilon = null)
        {
            this.Rho = rho ?? (Real)0.95;
            this.Epsilon = epsilon ?? (Real)1e-6;
        }

        internal override void AddFunctionParameters(FunctionParameter[] functionParameters)
        {
            foreach (FunctionParameter functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdaDeltaParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    class AdaDeltaParameter : OptimizerParameter
    {
        private readonly Real[] msg;
        private readonly Real[] msdx;
        private readonly AdaDelta optimizer;

        public AdaDeltaParameter(FunctionParameter functionParameter, AdaDelta optimizer) : base(functionParameter)
        {
            this.msg = new Real[functionParameter.Length];
            this.msdx = new Real[functionParameter.Length];
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad.Data[i];
                this.msg[i] *= this.optimizer.Rho;
                this.msg[i] += (1 - this.optimizer.Rho) * grad * grad;

                Real dx = (Real)Math.Sqrt((this.msdx[i] + this.optimizer.Epsilon) / (this.msg[i] + this.optimizer.Epsilon)) * grad;

                this.msdx[i] *= this.optimizer.Rho;
                this.msdx[i] += (1 - this.optimizer.Rho) * dx * dx;

                this.FunctionParameter.Param.Data[i] -= dx;
            }
        }
    }
}
