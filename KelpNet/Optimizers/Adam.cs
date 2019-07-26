using System;

namespace KelpNet
{
    [Serializable]
    public class Adam<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public Real<T> Alpha;
        public Real<T> Beta1;
        public Real<T> Beta2;
        public Real<T> Epsilon;

        public Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            this.Alpha = alpha;
            this.Beta1 = beta1;
            this.Beta2 = beta2;
            this.Epsilon = epsilon;
        }

        internal override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new AdamParameter<T>(functionParameter, this));
            }
        }
    }

    [Serializable]
    class AdamParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly Adam<T> _optimizer;

        private RealArray<T> m;
        private RealArray<T> v;

        public AdamParameter(NdArray<T> parameter, Adam<T> optimizer) : base(parameter)
        {
            this.m = new T[parameter.DataLength];
            this.v = new T[parameter.DataLength];

            this._optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            Real<T> fix1 = this._optimizer.Beta1;
            Real<T> fix2 = this._optimizer.Beta2;

            for (int i = 1; i < this._optimizer.UpdateCount; i++)
            {
                fix1 *= this._optimizer.Beta1;
                fix2 *= this._optimizer.Beta2;
            }

            fix1 = 1 - fix1;
            fix2 = 1 - fix2;

            Real<T> learningRate = this._optimizer.Alpha * Math.Sqrt(fix2) / fix1;

            for (int i = 0; i < FunctionParameter.DataLength; i++)
            {
                Real<T> grad = this.FunctionParameter.Grad[i];

                this.m[i] += (1 - this._optimizer.Beta1) * (grad - this.m[i]);
                this.v[i] += (1 - this._optimizer.Beta2) * (grad * grad - this.v[i]);

                this.FunctionParameter.Data[i] -= learningRate * this.m[i] / (Math.Sqrt(this.v[i]) + this._optimizer.Epsilon);
            }
        }
    }

}
