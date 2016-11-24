using System;
using System.Collections.Generic;

namespace KelpNet
{
    //Optimizerの基底クラス
    [Serializable]
    public abstract class Optimizer
    {
        //更新回数のカウント
        protected long UpdateCount = 1;
        protected List<OptimizeParameter> Parameters = new List<OptimizeParameter>();

        //更新回数のカウントを取りつつ更新処理を呼び出す
        public void Update()
        {
            this.DoUpdate();
            this.UpdateCount++;
        }

        //カウントを取るために呼び変えしている
        protected abstract void DoUpdate();

        //更新対象となるパラメータを保存
        public void SetParameters(IEnumerable<OptimizeParameter> parameters)
        {
            this.Parameters.AddRange(parameters);
            this.Initialize();
        }

        protected virtual void Initialize()
        {            
        }
    }
}
