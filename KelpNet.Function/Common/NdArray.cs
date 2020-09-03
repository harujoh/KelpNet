using System;
using System.Collections.Generic;
using System.Linq;

#if !DOUBLE
using Real = System.Single;
#else
using Real = System.Double;
#endif

namespace KelpNet
{
#if !DOUBLE
    public static class NdArrayF
#else
    public static class NdArrayD
#endif
    {
        public static void Backward(this NdArray<Real> ndArray)
        {
            if (ndArray.ParentFunc != null)
            {
                if (ndArray.Grad == null)
                {
                    ndArray.Grad = new Real[ndArray.Length * ndArray.BatchCount];

                    for (int i = 0; i < ndArray.Grad.Length; i++)
                    {
                        ndArray.Grad[i] = 1;
                    }
                }

                NdArray.Backward(ndArray);
            }
        }

        //傾きの補正
        public static void Reduce(this NdArray<Real> ndArray)
        {
            if (ndArray.TrainCount > 0)
            {
                for (int i = 0; i < ndArray.Grad.Length; i++)
                {
                    ndArray.Grad[i] /= ndArray.TrainCount;
                }
            }
        }

        private static NdArray<Real> ArrayFunc(NdArray<Real> input, Func<NdArray<Real>, int, NdArray<Real>> calcFunc, int[] axis = null, bool keepDims = false)
        {
#if DEBUG
            if (axis != null && axis.Length != axis.Distinct().ToArray().Length)
            {
                throw new Exception("指定された要素が重複しています");
            }

            if (axis != null && axis.Length != 0 && input.Shape.Length < axis.Max())
            {
                throw new Exception("指定された要素が範囲を超えています");
            }
#endif
            if (axis == null || axis.Length == 0)
            {
                axis = Enumerable.Range(0, input.Shape.Length).ToArray();
            }

            Array.Sort(axis);

            NdArray<Real> result = calcFunc(input, axis[0]);

            for (int i = 1; i < axis.Length; i++)
            {
                result = calcFunc(result, axis[i] - i);
            }

            if (keepDims)
            {
                int[] resultKeepDimShape = input.Shape.ToArray();

                for (int i = 0; i < axis.Length; i++)
                {
                    resultKeepDimShape[axis[i]] = 1;
                }

                result.Shape = resultKeepDimShape;
            }
            else if (input.Shape.Length == axis.Length)
            {
                result.Shape = new[] { 1 };
            }

            return result;
        }

        public static NdArray<Real> Sum(this NdArray<Real> input, int[] axis = null, bool keepDims = false)
        {
            return ArrayFunc(input, LocalSum, axis, keepDims);
        }

        private static NdArray<Real> LocalSum(NdArray<Real> input, int axis)
        {
            int[] resultShape = new int[input.Shape.Length - 1];

            for (int i = 0, j = 0; i < input.Shape.Length; i++)
            {
                if (i != axis)
                {
                    resultShape[j++] = input.Shape[i];
                }
            }

            NdArray<Real> result = new NdArray<Real>(resultShape, input.BatchCount);

            for (int i = 0; i < input.Length; i++)
            {
                List<int> index = new List<int>(input.GetDimensionsIndex(i));
                index.RemoveAt(axis);
                int localIndex = result.GetLocalIndex(0, index.ToArray());

                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    result.Data[batchCount * result.Length + localIndex] += input.Data[batchCount * input.Length + i];
                    if (input.Grad != null) result.Grad[batchCount * result.Length + localIndex] += input.Grad[batchCount * input.Length + i];
                }
            }

            return result;
        }

        public static NdArray<Real> Max(this NdArray<Real> input, int[] axis = null, bool keepDims = false)
        {
            return ArrayFunc(input, LocalMax, axis, keepDims);
        }

        private static NdArray<Real> LocalMax(NdArray<Real> input, int axis)
        {
            int[] resultShape = new int[input.Shape.Length - 1];

            for (int i = 0, j = 0; i < input.Shape.Length; i++)
            {
                if (i != axis)
                {
                    resultShape[j++] = input.Shape[i];
                }
            }

            NdArray<Real> result = new NdArray<Real>(resultShape, input.BatchCount);
            result.Fill(input.Data.Min());
            if (input.Grad != null) result.FillGrad(input.Grad.Min());

            for (int i = 0; i < input.Length; i++)
            {
                List<int> index = new List<int>(input.GetDimensionsIndex(i));
                index.RemoveAt(axis);
                int localIndex = result.GetLocalIndex(0, index.ToArray());

                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    if (result.Data[batchCount * result.Length + localIndex] < input.Data[batchCount * input.Length + i]) result.Data[batchCount * result.Length + localIndex] = input.Data[batchCount * input.Length + i];
                    if (input.Grad != null)
                    {
                        if (result.Grad[batchCount * result.Length + localIndex] < input.Grad[batchCount * input.Length + i])
                            result.Grad[batchCount * result.Length + localIndex] = input.Grad[batchCount * input.Length + i];
                    }
                }
            }

            return result;
        }


        public static NdArray<Real> Min(this NdArray<Real> input, int[] axis = null, bool keepDims = false)
        {
            return ArrayFunc(input, LocalMin, axis, keepDims);
        }

        private static NdArray<Real> LocalMin(NdArray<Real> input, int axis)
        {
            int[] resultShape = new int[input.Shape.Length - 1];

            for (int i = 0, j = 0; i < input.Shape.Length; i++)
            {
                if (i != axis)
                {
                    resultShape[j++] = input.Shape[i];
                }
            }

            NdArray<Real> result = new NdArray<Real>(resultShape, input.BatchCount);
            result.Fill(input.Data.Min());
            if (input.Grad != null) result.FillGrad(input.Grad.Min());

            for (int i = 0; i < input.Length; i++)
            {
                List<int> index = new List<int>(input.GetDimensionsIndex(i));
                index.RemoveAt(axis);
                int localIndex = result.GetLocalIndex(0, index.ToArray());

                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    if (result.Data[batchCount * result.Length + localIndex] > input.Data[batchCount * input.Length + i]) result.Data[batchCount * result.Length + localIndex] = input.Data[batchCount * input.Length + i];
                    if (input.Grad != null)
                    {
                        if (result.Grad[batchCount * result.Length + localIndex] > input.Grad[batchCount * input.Length + i])
                            result.Grad[batchCount * result.Length + localIndex] = input.Grad[batchCount * input.Length + i];
                    }
                }
            }

            return result;
        }
    }
}
