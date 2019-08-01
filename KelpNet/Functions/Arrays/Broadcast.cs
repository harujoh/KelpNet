using System;
using System.Collections.Generic;
using System.Linq;

namespace KelpNet
{
    [Serializable]
    public class Broadcast : SingleInputFunction
    {
        const string FUNCTION_NAME = "Broadcast";
        public int[] Shape;

        public Broadcast(int[] shape, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Shape = shape.ToArray();
        }

        public override NdArray SingleInputForward(NdArray val)
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
                    else if (this.Shape[i - offset] != 1 && resultShape[i] != this.Shape[i - offset])
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
                    else if (val.Shape[i - offset] != 1 && resultShape[i] != val.Shape[i - offset])
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

                    int tmpIndexLastIndex = val.Shape.Length - 1;
                    int valIndex = batchCount * val.Length;
                    int rankoffset = 1;

                    for (int j = 0; j < val.Shape.Length; j++)
                    {
                        if (val.Shape[tmpIndexLastIndex] > 1)
                        {
                            valIndex += baseIndex[tmpIndexLastIndex + indexOffset] * rankoffset;
                        }

                        rankoffset *= val.Shape[tmpIndexLastIndex--];
                    }

                    result.Data[batchCount * result.Length + i] = val.Data[valIndex];
                }
            }

            return result;
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            int ndim = x.Shape.Length;

            if (y.Shape.Length != ndim)
            {
                NdArray.Sum(y, Enumerable.Range(0, y.Shape.Length - ndim).ToArray(), false);
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
                NdArray result = NdArray.Sum(y, axis.ToArray(), true);
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
