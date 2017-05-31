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
            this.LearningRate = learningRate ?? (Real)0.1;
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
        private readonly SGD optimizer;

        public SGDParameter(FunctionParameter functionParameter, SGD optimizer) : base(functionParameter)
        {
            this.optimizer = optimizer;
        }

        public override void UpdateFunctionParameters()
        {
            for (int i = 0; i < this.FunctionParameter.Length; i++)
            {
                this.FunctionParameter.Param.Data[i] -= this.optimizer.LearningRate * this.FunctionParameter.Grad.Data[i];
            }
        }
    }

}
