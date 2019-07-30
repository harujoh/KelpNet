using System.Collections.Generic;

namespace KelpNet
{
    public class Concat : MultiInputFunction
    {
        const string FUNCTION_NAME = "Concat";
        public int Axis;

        private readonly List<int[]> _prevInputSections = new List<int[]>();

        public Concat(int axis, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Axis = axis;
        }

        protected override NdArray MultiInputForward(params NdArray[] xs)
        {
            int[] sections = new int[xs.Length - 1];
            int sizeOffset = xs[0].Shape[Axis];

            NdArray resultNdArray = xs[0].Clone();

            for (int i = 1; i < xs.Length; i++)
            {
                //BackwardのSplitで使用しないため最後のshapeを保存しないロジックになっている
                sections[i - 1] = sizeOffset;
                sizeOffset += xs[i].Shape[Axis];

                resultNdArray = NdArray.Concatenate(resultNdArray, xs[i], Axis);
            }

            resultNdArray.ParentFunc = this;

            _prevInputSections.Add(sections);

            return resultNdArray;
        }

        protected override void MultiOutputBackward(NdArray y, NdArray[] xs)
        {
            int[] prevInputShapes = this._prevInputSections[this._prevInputSections.Count - 1];
            this._prevInputSections.RemoveAt(this._prevInputSections.Count - 1);

            NdArray[] result = NdArray.Split(y, prevInputShapes, this.Axis);

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
