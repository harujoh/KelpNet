using System;
using System.Collections.Generic;

namespace KelpNet
{
    //Optimizerの素となるクラスでパラメータを持つ
    [Serializable]
    public abstract class Optimizer<T> where T : unmanaged, IComparable<T>
    {
        public long UpdateCount = 0;
        public List<OptimizerParameter<T>> OptimizerParameters = new List<OptimizerParameter<T>>();

        public abstract void AddFunctionParameters(NdArray<T>[] functionParameters);

        public List<Scheduler<T>> Schedulers = new List<Scheduler<T>>();

        public Action Update;

        public List<Action> ResetStates = new List<Action>();

        public void SetUp(Function<T> function)
        {
            ResetStates.Add(function.ResetState);
            AddFunctionParameters(function.Parameters);
        }

        public virtual void Step()
        {
        }

        public void ResetState()
        {
            for (int i = 0; i < ResetStates.Count; i++)
            {
                ResetStates[i]();
            }
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
    public abstract class OptimizerParameter<T> where T : unmanaged, IComparable<T>
    {
        public NdArray<T> FunctionParameter;

        protected OptimizerParameter(NdArray<T> functionParameter)
        {
            this.FunctionParameter = functionParameter;
        }

        public Action UpdateFunctionParameters;
    }
}
