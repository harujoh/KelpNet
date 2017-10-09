using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Arrays
{
    public class Concat : Function
    {
        const string FUNCTION_NAME = "Concat";
        private int _axis;

        private readonly List<int[][]> _prevInputShapes = new List<int[][]>();

        public Concat(int axis = 1, string name = FUNCTION_NAME, int inputCount = 0, int oututCount = 0) : base(name, inputCount, oututCount)
        {
            this._axis = axis;
        }

        public NdArray ForwardCpu(params NdArray[] xs)
        {
            List<int[]> shapes = new List<int[]>();
            shapes.Add(xs[0].Shape);

            NdArray[] resultNdArrays = xs[0].DivideArrays();

            for (int i = 1; i < xs.Length; i++)
            {
                shapes.Add(xs[i].Shape);
                var tmpNdArray = xs[i].DivideArrays();

                for (int j = 0; j < resultNdArrays.Length; j++)
                {
                    //resultNdArrays[j] = NdArray.Concatenate(resultNdArrays[j], tmpNdArray[j], _axis);
                }
            }

            _prevInputShapes.Add(shapes.ToArray());

            return new NdArray(resultNdArrays);
        }

        public NdArray[] BackwardCpu(NdArray gy)
        {
            int[][] prevInputShapes = this._prevInputShapes[this._prevInputShapes.Count - 1];
            this._prevInputShapes.RemoveAt(this._prevInputShapes.Count - 1);

            return new[] { gy };
        }
    }
}
