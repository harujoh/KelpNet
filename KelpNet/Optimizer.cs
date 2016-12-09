using System;

namespace KelpNet
{
    //Optimizerの素となるクラスでパラメータを持つ
    [Serializable]
    public abstract class Optimizer
    {
        public long UpdateCount = 1;
        protected OptimizerParameter[] OptimizerParameters;

        public abstract void Initilise(FunctionParameter[] functionParameters);

        public void Update()
        {
            foreach (OptimizerParameter optimizersParameter in this.OptimizerParameters)
            {
                optimizersParameter.UpdateFunctionParameters();
            }

            this.UpdateCount++;
        }
    }

    //このクラスはFunctionParameterと1:1で作成される
    [Serializable]
    public abstract class OptimizerParameter
    {
        protected FunctionParameter FunctionParameter;

        protected OptimizerParameter(FunctionParameter functionParameter)
        {
            this.FunctionParameter = functionParameter;
        }

        public abstract void UpdateFunctionParameters();
    }
}
