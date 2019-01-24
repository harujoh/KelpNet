using System;

namespace KelpNet
{
    [Serializable]
    public class AdaDelta<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public Real<T> Rho;
        public Real<T> Epsilon;

        public AdaDelta(double rho = 0.95f, double epsilon = 1e-6f)
        {
            this.Rho = rho;
            this.Epsilon = epsilon;
        }

        internal override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdaDeltaParameter<T>(functionParameter, this));
            }
        }
    }

    [Serializable]
    class AdaDeltaParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly Real<T>[] msg;
        private readonly Real<T>[] msdx;
        private readonly AdaDelta<T> optimizer;

        public AdaDeltaParameter(NdArray<T> functionParameter, AdaDelta<T> optimizer) : base(functionParameter)
        {
            this.msg = new Real<T>[functionParameter.Data.Length];
            this.msdx = new Real<T>[functionParameter.Data.Length];
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Data.Length; i++)
            {
                Real<T> grad = this.FunctionParameter.Grad[i];
                this.msg[i] *= this.optimizer.Rho;
                this.msg[i] += (1 - this.optimizer.Rho) * grad * grad;

                Real<T> dx = Math.Sqrt((this.msdx[i] + this.optimizer.Epsilon) / (this.msg[i] + this.optimizer.Epsilon)) * grad;

                this.msdx[i] *= this.optimizer.Rho;
                this.msdx[i] += (1 - this.optimizer.Rho) * dx * dx;

                this.FunctionParameter.Data[i] -= dx;
            }
        }
    }
}
