using System;

namespace KelpNet
{
    //Optimizerの基底クラス
    [Serializable]
    public abstract class Optimizer
    {
        //更新回数のカウント
        protected long UpdateCount = 1;
        protected OptimizeParameter[] Parameters;

        protected Optimizer(OptimizeParameter[] parameters)
        {
            this.Parameters = parameters;
        }

        //更新回数のカウントを取りつつ更新処理を呼び出す
        public void Update()
        {
            this.DoUpdate();
            this.UpdateCount++;
        }

        //カウントを取るために呼び変えしている
        protected abstract void DoUpdate();
    }
}
