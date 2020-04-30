using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet
{
#if !DOUBLE
    [Serializable]
    public class Broadcast<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Broadcast";
        public int[] Shape;

        public Broadcast(int[] shape, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Shape = shape.ToArray();

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            base.SingleInputForward = this.BroadcastForward;

            switch (this)
            {
                case Broadcast<float> broadcastF:
                    broadcastF.SingleOutputBackward = BroadcastF.SingleOutputBackward;
                    break;
                case Broadcast<double> broadcastD:
                    broadcastD.SingleOutputBackward = BroadcastD.SingleOutputBackward;
                    break;
            }
        }

        public NdArray<T> BroadcastForward(NdArray<T> val)
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

            NdArray<T> result = new NdArray<T>(resultShape, val.BatchCount, this);
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
    }
#endif

#if DOUBLE
    public static class BroadcastD
#else
    public static class BroadcastF
#endif
    {
        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x)
        {
            int ndim = x.Shape.Length;

            if (y.Shape.Length != ndim)
            {
                y.Sum(Enumerable.Range(0, y.Shape.Length - ndim).ToArray(), false);
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
                NdArray<Real> result = y.Sum(axis.ToArray(), true);
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
