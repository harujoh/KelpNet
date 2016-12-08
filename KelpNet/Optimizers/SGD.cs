using System;

namespace KelpNet.Optimizers
{
    [Serializable]
    public class SGD : Optimizer
    {
        public double LearningRate;

        public SGD(double learningRate = 0.1)
        {
            this.LearningRate = learningRate;
        }

        public override void Initilise(OptimizeParameter[] functionParameters)
        {
            this.OptimizerParameters = new OptimizerParameter[functionParameters.Length];

            for (int i = 0; i < this.OptimizerParameters.Length; i++)
            {
                this.OptimizerParameters[i] = new SGDParameter(functionParameters[i], this);
            }
        }
    }

    [Serializable]
    class SGDParameter : OptimizerParameter
    {
        private readonly SGD optimiser;

        public SGDParameter(OptimizeParameter functionParameter, SGD optimiser) : base(functionParameter)
        {
            this.optimiser = optimiser;
        }

        public override void Update()
        {
            for (int i = 0; i < this.FunctionParameters.Length; i++)
            {
                this.FunctionParameters.Param.Data[i] -= this.optimiser.LearningRate * this.FunctionParameters.Grad.Data[i];
            }
        }
    }

}
