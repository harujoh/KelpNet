using System;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Optimizers;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class SGD : Optimizer
    {
        public Real LearningRate;

        public SGD(Real? learningRate = null)
        {
            this.LearningRate = learningRate ?? 0.1f;
        }

        internal override void AddFunctionParameters(FunctionParameter[] functionParameters)
        {
            foreach (FunctionParameter functionParameter in functionParameters)
            {
                this.OptimizerParameters.Add(new SGDParameter(functionParameter, this));
            }
        }
    }

    [Serializable]
    class SGDParameter : OptimizerParameter
    {
        private readonly SGD optimiser;

        public SGDParameter(FunctionParameter functionParameter, SGD optimiser) : base(functionParameter)
        {
            this.optimiser = optimiser;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                this.FunctionParameter.Param.Data[i] -= this.optimiser.LearningRate * this.FunctionParameter.Grad.Data[i];
            }
        }
    }

}
