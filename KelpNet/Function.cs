using System;
using System.Collections.Generic;

namespace KelpNet
{
    //FunctionStackに積み上げるFunctionの基底クラス
    public abstract class Function
    {
        public struct Parameter
        {
            public NdArray Param;
            public NdArray Grad;

            public int Length
            {
                get
                {
                    return this.Param.Length;
                }
            }

            public Parameter(NdArray param, NdArray grad)
            {
                this.Param = param;
                this.Grad = grad;
            }
        }
        
        public List<Parameter> Parameters = new List<Parameter>();

        public int OutputCount;
        public int InputCount;

        public NdArray[] PrevInput = new NdArray[1];
        public NdArray[] PrevOutput = new NdArray[1];

        public abstract NdArray Forward(NdArray x, int batchId=0);
        public abstract NdArray Backward(NdArray gy, int batchId=0);

        //初期値が入力されなかった場合、この関数で初期化を行う
        protected void InitWeight(NdArray array, double masterScale = 1.0)
        {
            var localScale = 1 / Math.Sqrt(2);
            var fanIn = this.GetFans(array.Shape);
            var s = localScale * Math.Sqrt(2.0 / fanIn);

            for (int i = 0; i < array.Length; i++)
            {
                array.Data[i] = this.Normal(s) * masterScale;
            }
        }

        double Normal(double scale = 0.05)
        {
            Mother.Sigma = scale;
            return Mother.RandomNormal();
        }

        int GetFans(int[] shape)
        {
            int result = 1;

            for (int i = 1; i < shape.Length; i++)
            {
                result *= shape[i];
            }

            return result;
        }

    }
}
