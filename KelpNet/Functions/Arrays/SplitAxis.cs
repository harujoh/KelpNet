using System.Linq;

namespace KelpNet
{
    public class SplitAxis : MultiOutputFunction
    {
        const string FUNCTION_NAME = "SplitAxis";
        public int Axis;
        public int[] Indices;

        public SplitAxis(int[] indices, int axis, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Indices = indices.ToArray();
            this.Axis = axis;
        }

        protected override NdArray[] SingleInputForward(NdArray x)
        {
            NdArray[] resultArrays = NdArray.Split(x, Indices, Axis);

            for (int i = 0; i < resultArrays.Length; i++)
            {
                resultArrays[i].ParentFunc = this;
            }

            return resultArrays;
        }

        protected override void MultiOutputBackward(NdArray[] ys, NdArray x)
        {
            NdArray resultNdArray = ys[0].Clone();

            for (int i = 1; i < ys.Length; i++)
            {
                resultNdArray = NdArray.Concatenate(resultNdArray, ys[i], Axis);
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += resultNdArray.Grad[i];
            }
        }
    }
}
