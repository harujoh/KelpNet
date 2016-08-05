using System;
using System.Collections.Generic;

namespace KelpNet.Functions
{
    //重みと傾きがあるものを扱うクラス
    public abstract class OptimizableFunction : Function
    {
        public class Parameter
        {
            public NdArray Param;
            public NdArray Grad;
            public int Length
            {
                get
                {
                    return Param.Length;
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

        //初期値が入力されなかった場合、この関数で初期化を行う
        protected void InitWeight(NdArray array, double masterScale = 1.0)
        {
            var localScale = 1 / Math.Sqrt(2);
            var fanIn = GetFans(array.Shape);
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

        int GetFans(int[] Shape)
        {
            int result = 1;

            for (int i = 1; i < Shape.Length; i++)
            {
                result *= Shape[i];
            }

            return result;
        }
    }
}
