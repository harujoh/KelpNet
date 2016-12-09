using System;
using KelpNet.Common;

namespace KelpNet
{
    [Serializable]
    public class FunctionParameter
    {
        public string Name;
        public NdArray Param;
        public NdArray Grad;

        //Updateを行わずに実行されたBackwardの回数をカウントし、バッチ更新時に使用する
        public int TrainCount = 0;

        public readonly int Length = 0;

        public FunctionParameter(NdArray param, NdArray grad, string name)
        {
            this.Param = param;
            this.Grad = grad;
            this.Name = name;

            this.Length = this.Param.Length;
        }


        //傾きの補正
        public void Reduce()
        {
            for (int j = 0; j < this.Grad.Length; j++)
            {
                this.Grad.Data[j] /= this.TrainCount;
            }

            //カウンタをリセット
            this.TrainCount = 0;
        }

        //傾きの初期化
        public void ClearGrad()
        {
            //0埋め
            this.Grad.Fill(0);

            //カウンタをリセット
            this.TrainCount = 0;
        }

        public override string ToString()
        {
            return this.Name;
        }
    }
}
