using System;
using System.Threading.Tasks;

namespace KelpNet
{
    [Serializable]
    public class MomentumSGD : Optimizer
    {
        public Real LearningRate;
        public Real Momentum;

        public MomentumSGD(double learningRate = 0.01, double momentum = 0.9)
        {
            this.LearningRate = learningRate;
            this.Momentum = momentum;
        }

        internal override void AddFunctionParameters(NdArray[] functionParameters)
        {
            foreach (NdArray functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new MomentumSGDParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    public class MomentumSGDParameter : OptimizerParameter
    {
        private readonly MomentumSGD optimizer;
        private readonly Real[] v;

        public MomentumSGDParameter(NdArray functionParameter, MomentumSGD optimizer) : base(functionParameter)
        {
            this.v = new Real[functionParameter.Data.Length];
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            Parallel.For(0, FunctionParameter.Data.Length, i =>
            {
                this.v[i] *= this.optimizer.Momentum;
                this.v[i] -= this.optimizer.LearningRate * this.FunctionParameter.Grad[i];

                this.FunctionParameter.Data[i] += this.v[i];
            });
        }
    }

}
