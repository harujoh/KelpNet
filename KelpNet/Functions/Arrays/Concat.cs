using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Arrays
{
    public class Concat : MultiInputFunction
    {
        const string FUNCTION_NAME = "Concat";
        private int _axis;

        private readonly List<int[]> _prevInputSections = new List<int[]>();

        public Concat(int axis = 1, string name = FUNCTION_NAME) : base(name)
        {
            this._axis = axis;

            MultiInputForward = ForwardCpu;
            MultiOutputBackward = BackwardCpu;
        }

        public NdArray ForwardCpu(params NdArray[] xs)
        {
            int[] sections = new int[xs.Length - 1];
            int sizeOffset = xs[0].Shape[_axis];

            NdArray resultNdArray = xs[0].Clone();
            resultNdArray.ParentFunc = this;

            for (int i = 1; i < xs.Length; i++)
            {
                //BackwardのSplitで使用しないため最後のshapeを保存しないロジックになっている
                sections[i - 1] = sizeOffset;
                sizeOffset += xs[i].Shape[_axis];

                resultNdArray = NdArray.Concatenate(resultNdArray, xs[i], _axis);
            }

            _prevInputSections.Add(sections);

            return resultNdArray;
        }

        public void BackwardCpu(NdArray y, NdArray[] xs)
        {
            int[] prevInputShapes = this._prevInputSections[this._prevInputSections.Count - 1];
            this._prevInputSections.RemoveAt(this._prevInputSections.Count - 1);

            NdArray[] result = NdArray.Split(y, prevInputShapes, this._axis);

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
