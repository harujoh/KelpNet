using System;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class MomentumSGD : Optimizer
    {
        public double LearningRate;
        public double Momentum;

        public MomentumSGD( double learningRate = 0.01, double momentum = 0.9)
        {
            this.LearningRate = learningRate;
            this.Momentum = momentum;
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
        private readonly MomentumSGD optimiser;
        private readonly double[] v;

        public MomentumSGDParameter(FunctionParameter functionParameter, MomentumSGD optimiser) : base(functionParameter)
        {
            this.v = new double[functionParameter.Length];
            this.optimiser = optimiser;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                this.v[i] *= this.optimiser.Momentum;
                this.v[i] -= this.optimiser.LearningRate * this.FunctionParameter.Grad.Data[i];

                this.FunctionParameter.Param.Data[i] += this.v[i];
            }
        }
    }

}
