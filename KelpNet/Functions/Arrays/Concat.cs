using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Arrays
{
    public class Concat : MultiInputFunction
    {
        const string FUNCTION_NAME = "Concat";
        private int _axis;

        private readonly List<int[]> _prevInputShapes = new List<int[]>();

        public Concat(int axis = 1, string name = FUNCTION_NAME) : base(name)
        {
            this._axis = axis;

            MultiInputForward = ForwardCpu;
            MultiOutputBackward = BackwardCpu;
        }

        public NdArray ForwardCpu(params NdArray[] xs)
        {
            List<int> shapes = new List<int>();
            shapes.Add(xs[0].Shape[_axis]);
            int baseShapeSize = xs[0].Shape[_axis];

            NdArray[] resultNdArrays = xs[0].DivideArrays();

            for (int i = 1; i < xs.Length; i++)
            {
                baseShapeSize += xs[i].Shape[_axis];
                shapes.Add(baseShapeSize);
                var tmpNdArray = xs[i].DivideArrays();

                for (int j = 0; j < resultNdArrays.Length; j++)
                {
                    resultNdArrays[j] = NdArray.Concatenate(resultNdArrays[j], tmpNdArray[j], _axis);
                }
            }

            //BackwardのSplitで使用しないため最後のshapeは削除しておく
            shapes.RemoveAt(shapes.Count - 1);

            _prevInputShapes.Add(shapes.ToArray());

            return new NdArray(resultNdArrays, this);
        }

        public void BackwardCpu(NdArray y, NdArray[] xs)
        {
            int[] prevInputShapes = this._prevInputShapes[this._prevInputShapes.Count - 1];
            this._prevInputShapes.RemoveAt(this._prevInputShapes.Count - 1);

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
