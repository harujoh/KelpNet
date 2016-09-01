using System;
using System.Collections.Generic;
using System.Threading.Tasks;

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


        public abstract NdArray Forward(NdArray x);
        public abstract NdArray Backward(NdArray gy, NdArray prevInput, NdArray prevOutput);

        public virtual NdArray[] BatchForward(NdArray[] x)
        {
            NdArray[] y = new NdArray[x.Length];

            Parallel.For(0, x.Length, i =>
            {
                y[i] = Forward(x[i]);
            });

            return y;
        }

        public virtual NdArray[] BatchBackward(NdArray[] gy, NdArray[] prevInput, NdArray[] prevOutput)
        {
            NdArray[] gx = new NdArray[gy.Length];

            Parallel.For(0, gy.Length, i =>
            {
                gx[i] = Backward(gy[i], prevInput[i], prevOutput[i]);
            });

            return gx;
        }

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
