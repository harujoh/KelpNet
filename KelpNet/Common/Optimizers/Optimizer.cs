using System;
using System.Collections.Generic;

namespace KelpNet
{
    //Optimizerの素となるクラスでパラメータを持つ
    [Serializable]
    public abstract class Optimizer
    {
        public long UpdateCount = 0;
        protected List<OptimizerParameter> OptimizerParameters = new List<OptimizerParameter>();

        internal abstract void AddFunctionParameters(NdArray[] functionParameters);

        public List<Scheduler> Schedulers = new List<Scheduler>();

        public void SetUp(Function function)
        {
            AddFunctionParameters(function.Parameters);
        }

        public void Update()
        {
            if (this.OptimizerParameters[0].FunctionParameter.TrainCount > 0)
            {
                this.UpdateCount++;

                for (int i = 0; i < this.OptimizerParameters.Count; i++)
                {
                    //傾きの割引を実行
                    this.OptimizerParameters[i].FunctionParameter.Reduce();

                    this.OptimizerParameters[i].UpdateFunctionParameters();

                    this.OptimizerParameters[i].FunctionParameter.InitGrad();

                    //カウンタをリセット
                    this.OptimizerParameters[i].FunctionParameter.TrainCount = 0;
                }
            }
        }

        public virtual void Step()
        {
        }

        public void ResetParams()
        {
            for (int i = 0; i < this.OptimizerParameters.Count; i++)
            {
                if (this.OptimizerParameters[i].FunctionParameter.Grad != null)
                {
                    this.OptimizerParameters[i].FunctionParameter.InitGrad();
                }

                //カウンタをリセット
                this.OptimizerParameters[i].FunctionParameter.TrainCount = 0;
            }

            this.UpdateCount = 0;
        }
    }

    //このクラスはFunctionParameterと1:1で作成される
    [Serializable]
    public abstract class OptimizerParameter
    {
        public NdArray FunctionParameter;

        protected OptimizerParameter(NdArray functionParameter)
        {
            this.FunctionParameter = functionParameter;
        }

        public abstract void UpdateFunctionParameters();
    }
}
