using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Arrays
{
    public class Broadcast : SingleInputFunction
    {
        const string FUNCTION_NAME = "Broadcast";
        private int[] Shape;

        public Broadcast(int[] shape, string name = FUNCTION_NAME) : base(name)
        {
            Shape = shape.ToArray();

            SingleInputForward = ForwardCpu;
        }

        NdArray ForwardCpu(NdArray val)
        {
            int resultNdim = this.Shape.Length;
            int[] maxShape = this.Shape.ToArray();

            //次元数を揃えつつ最大の次元を探査して設定
            if (val.Shape.Length < resultNdim)
            {
                int offset = 0;

                for (int i = 0; i < resultNdim; i++)
                {
                    if (resultNdim - i - 1 < val.Shape.Length)
                    {
                        if (maxShape[i] < val.Shape[i - offset])
                        {
                            maxShape[i] = val.Shape[i - offset];
                        }
                    }
                    else
                    {
                        offset++;
                    }
                }
            }
            else
            {
                resultNdim = val.Shape.Length;
                maxShape = val.Shape.ToArray();

                for (int i = 0; i < resultNdim; i++)
                {
                    if (maxShape[i] < val.Shape[i])
                    {
                        maxShape[i] = val.Shape[i];
                    }
                }
            }

#if DEBUG
            for (int j = 0; j < val.Shape.Length; j++)
            {
                int dimOffset = resultNdim - val.Shape.Length;

                if (val.Shape[j] != 1 && val.Shape[j] != maxShape[j + dimOffset])
                {
                    throw new Exception("変換不可能な組み合わせです");
                }
            }
#endif

            NdArray result = new NdArray(maxShape);

            for (int batchCount = 0; batchCount < result.BatchCount; batchCount++)
            {
                for (int i = 0; i < result.Length; i++)
                {
                    int[] baseIndex = result.GetDimensionsIndex(i);

                    for (int j = 0; j < val.Length; j++)
                    {
                        //全て0が入った添字配列を用意
                        int[] tmpIndex = Enumerable.Repeat(0, val.Shape.Length).ToArray();

                        int offset = result.Shape.Length - val.Shape.Length;

                        for (int k = 0; k < tmpIndex.Length; k++)
                        {
                            if (val.Shape[k] > 1)
                            {
                                tmpIndex[k] = baseIndex[k + offset];
                            }
                        }

                        result.Data[batchCount * result.Length + i] += val.Data[val.GetLocalIndex(tmpIndex, batchCount)];
                    }
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
