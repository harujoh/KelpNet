using System;
using System.Collections.Generic;

namespace KelpNet
{
    public class Concat<T> : MultiInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Concat";
        public int Axis;

        private readonly List<int[]> _prevInputSections = new List<int[]>();

        public Concat(int axis = 1, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Axis = axis;

            MultiInputForward = ForwardCpu;
            MultiOutputBackward = BackwardCpu;
        }

        private NdArray<T> ForwardCpu(params NdArray<T>[] xs)
        {
            int[] sections = new int[xs.Length - 1];
            int sizeOffset = xs[0].Shape[Axis];

            NdArray<T> resultNdArray = xs[0].Clone();
            resultNdArray.ParentFunc = this;

            for (int i = 1; i < xs.Length; i++)
            {
                //BackwardのSplitで使用しないため最後のshapeを保存しないロジックになっている
                sections[i - 1] = sizeOffset;
                sizeOffset += xs[i].Shape[Axis];

                resultNdArray = NdArray<T>.Concatenate(resultNdArray, xs[i], Axis);
            }

            _prevInputSections.Add(sections);

            return resultNdArray;
        }

        private void BackwardCpu(NdArray<T> y, NdArray<T>[] xs)
        {
            int[] prevInputShapes = this._prevInputSections[this._prevInputSections.Count - 1];
            this._prevInputSections.RemoveAt(this._prevInputSections.Count - 1);

            NdArray<T>[] result = NdArray<T>.Split(y, prevInputShapes, this.Axis);

            for (int i = 0; i < xs.Length; i++)
            {
                for (int j = 0; j < xs[i].Grad.Length; j++)
                {
                    xs[i].Grad[j] += result[i].Grad[j];
                }
            }
        }
    }
}
