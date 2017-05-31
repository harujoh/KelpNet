using System;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Optimizers;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class MomentumSGD : Optimizer
    {
        public Real LearningRate;
        public Real Momentum;

        public MomentumSGD(Real? learningRate = null, Real? momentum = null)
        {
            this.LearningRate = learningRate ?? (Real)0.01;
            this.Momentum = momentum ?? (Real)0.9;
        }

        internal override void AddFunctionParameters(FunctionParameter[] functionParameters)
        {
            foreach (FunctionParameter functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new MomentumSGDParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    class MomentumSGDParameter : OptimizerParameter
    {
        private readonly MomentumSGD optimizer;
        private readonly Real[] v;

        public MomentumSGDParameter(FunctionParameter functionParameter, MomentumSGD optimizer) : base(functionParameter)
        {
            this.v = new Real[functionParameter.Length];
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                this.v[i] *= this.optimizer.Momentum;
                this.v[i] -= this.optimizer.LearningRate * this.FunctionParameter.Grad.Data[i];

                this.FunctionParameter.Param.Data[i] += this.v[i];
            }
        }
    }

}
