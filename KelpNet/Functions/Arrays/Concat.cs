using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Arrays
{
    class Concat : Function
    {
        const string FUNCTION_NAME = "Concat";
        private int _axis;

        private readonly List<int[]> _prevInputShapes = new List<int[]>();

        public Concat(int axis = 1, string name = FUNCTION_NAME, int inputCount = 0, int oututCount = 0) : base(name, inputCount, oututCount)
        {
            this._axis = axis;
        }

        public BatchArray[] ForwardCpu(BatchArray[] x)
        {
            _prevInputShapes.Add(x[0].Shape);

            return x;
        }

        public BatchArray[] BackwardCpu(BatchArray[] gh)
        {
            int[] prevOutputData = this._prevInputShapes[this._prevInputShapes.Count - 1];
            this._prevInputShapes.RemoveAt(this._prevInputShapes.Count - 1);

            BackwardCountUp();

            return gh;
        }
    }
}
