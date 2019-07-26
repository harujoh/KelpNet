using System;

namespace KelpNet
{
    [Serializable]
    public class MomentumSGD<T> : Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public Real<T> LearningRate;
        public Real<T> Momentum;

        public MomentumSGD(double learningRate = 0.01, double momentum = 0.9)
        {
            this.LearningRate = learningRate;
            this.Momentum = momentum;
        }

        internal override void AddFunctionParameters(NdArray<T>[] functionParameters)
        {
            foreach (NdArray<T> functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new MomentumSGDParameter<T>(functionParameter, this));
            }
        }
    }

    [Serializable]
    class MomentumSGDParameter<T> : OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        private readonly MomentumSGD<T> optimizer;
        private RealArray<T> v;

        public MomentumSGDParameter(NdArray<T> functionParameter, MomentumSGD<T> optimizer) : base(functionParameter)
        {
            this.v = new T[functionParameter.DataLength];
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.DataLength; i++)
            {
                this.v[i] *= this.optimizer.Momentum;
                this.v[i] -= this.optimizer.LearningRate * this.FunctionParameter.Grad[i];

                this.FunctionParameter.Data[i] += this.v[i];
            }
        }
    }

}
