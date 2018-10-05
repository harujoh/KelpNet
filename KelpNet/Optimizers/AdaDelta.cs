using System;

#if DOUBLE
using Real = System.Double;
namespace Double.KelpNet
#else
using Real = System.Single;
namespace KelpNet
#endif
{
    [Serializable]
    public class AdaDelta : Optimizer
    {
        public Real Rho;
        public Real Epsilon;

        public AdaDelta(Real rho = 0.95f, Real epsilon = 1e-6f)
        {
            this.Rho = rho;
            this.Epsilon = epsilon;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
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

        public AdaDeltaParameter(NdArray functionParameter, AdaDelta optimizer) : base(functionParameter)
        {
            this.msg = new Real[functionParameter.Data.Length];
            this.msdx = new Real[functionParameter.Data.Length];
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Data.Length; i++)
            {
                Real grad = this.FunctionParameter.Grad[i];
                this.msg[i] *= this.optimizer.Rho;
                this.msg[i] += (1 - this.optimizer.Rho) * grad * grad;

                Real dx = (Real)Math.Sqrt((this.msdx[i] + this.optimizer.Epsilon) / (this.msg[i] + this.optimizer.Epsilon)) * grad;

                this.msdx[i] *= this.optimizer.Rho;
                this.msdx[i] += (1 - this.optimizer.Rho) * dx * dx;

                this.FunctionParameter.Data[i] -= dx;
            }
        }
    }
}
