using System;

namespace KelpNet
{
    //Optimizerの素となるクラスでパラメータを持つ
    [Serializable]
    public abstract class Optimizer
    {
        public long UpdateCount = 1;
        protected OptimizerParameter[] OptimizerParameters;

        public abstract void Initilise(OptimizeParameter[] functionParameters);

        public void Update()
        {
            foreach (OptimizerParameter optimizersParameter in this.OptimizerParameters)
            {
                optimizersParameter.Update();
            }

            this.UpdateCount++;
        }
    }

    //このクラスはFunctionParameterと1:1で作成される
    [Serializable]
    public abstract class OptimizerParameter
    {
        protected OptimizeParameter FunctionParameters;

        protected OptimizerParameter(OptimizeParameter functionParameter)
        {
            this.FunctionParameters = functionParameter;
        }

        public abstract void Update();
    }
}
