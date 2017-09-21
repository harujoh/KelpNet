using System;

namespace KelpNet.Common.Functions
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

            this.Length = this.Param.Data.Length;
        }

        public void CountUp()
        {
            TrainCount++;
        }

        //傾きの補正
        public bool Reduce()
        {
            if (this.TrainCount > 0)
            {
                for (int i = 0; i < this.Grad.Data.Length; i++)
                {
                    this.Grad.Data[i] /= this.TrainCount;
                }

                return true;
            }

            return false;
        }

        //傾きの初期化
        public void ClearGrad()
        {
            //参照値であるためClearは使ってはいけない
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
