using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Arrays
{
    [Serializable]
    public class Broadcast : SingleInputFunction
    {
        const string FUNCTION_NAME = "Broadcast";
        private readonly int[] Shape;

        public Broadcast(int[] shape, string name = FUNCTION_NAME) : base(name)
        {
            this.Shape = shape.ToArray();

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        NdArray ForwardCpu(NdArray val)
        {
            int[] resultShape;

            if (val.Shape.Length > this.Shape.Length)
            {
                //入力の方が大きい
                resultShape = val.Shape.ToArray();
                int offset = val.Shape.Length - this.Shape.Length;

                for (int i = offset; i < resultShape.Length; i++)
                {
                    if (resultShape[i] == 1)
                    {
                        resultShape[i] = this.Shape[i - offset];
                    }
#if DEBUG
                    else if (resultShape[i] != this.Shape[i - offset])
                    {
                        throw new Exception("変換不可能な組み合わせです");
                    }
#endif
                }
            }
            else
            {
                //指定の方が大きい
                resultShape = this.Shape.ToArray();
                int offset = this.Shape.Length - val.Shape.Length;

                for (int i = offset; i < resultShape.Length; i++)
                {
                    if (resultShape[i] == 1)
                    {
                        resultShape[i] = val.Shape[i - offset];
                    }
#if DEBUG
                    else if (resultShape[i] != val.Shape[i - offset])
                    {
                        throw new Exception("変換不可能な組み合わせです");
                    }
#endif
                }
            }

            NdArray result = new NdArray(resultShape, val.BatchCount, this);
            int indexOffset = result.Shape.Length - val.Shape.Length;

            for (int batchCount = 0; batchCount < result.BatchCount; batchCount++)
            {
                for (int i = 0; i < result.Length; i++)
                {
                    int[] baseIndex = result.GetDimensionsIndex(i);

                    //全て0が入った添字配列を用意
                    int[] tmpIndex = new int[val.Shape.Length];

                    for (int k = 0; k < tmpIndex.Length; k++)
                    {
                        if (val.Shape[k] > 1)
                        {
                            tmpIndex[k] = baseIndex[k + indexOffset];
                        }
                    }

                    result.Data[batchCount * result.Length + i] = val.Data[val.GetLocalIndex(tmpIndex, batchCount)];
                }
            }

            return result;
        }

        protected void BackwardCpu(NdArray y, NdArray x)
        {
            int ndim = x.Shape.Length;

            if (y.Shape.Length != ndim)
            {
                NdArray.Sum(y, false, Enumerable.Range(0, y.Shape.Length - ndim).ToArray());
            }

            List<int> axis = new List<int>();
            for (int i = 0; i < x.Shape.Length; i++)
            {
                if (x.Shape[i] == 1)
                {
                    axis.Add(i);
                }
            }

            if (axis.Count > 0)
            {
                NdArray result = NdArray.Sum(y, true, axis.ToArray());
                for (int i = 0; i < x.Grad.Length; i++)
                {
                    x.Grad[i] += result.Grad[i];
                }
            }
            else
            {
                for (int i = 0; i < x.Grad.Length; i++)
                {
                    x.Grad[i] += y.Grad[i];
                }
            }


        }
    }
}
