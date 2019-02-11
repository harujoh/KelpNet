using System.Linq;

namespace KelpNet
{
    public class SplitAxis : MultiOutputFunction
    {
        const string FUNCTION_NAME = "SplitAxis";
        public int Axis;
        public int[] Indices;

        public SplitAxis(int indices, int axis, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Indices = new[] { indices };
            this.Axis = axis;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        public SplitAxis(int[] indices, int axis, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.Indices = indices.ToArray();
            this.Axis = axis;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        private NdArray[] ForwardCpu(NdArray x)
        {
            NdArray[] resultArays = NdArray.Split(x, Indices, Axis);

            foreach (var resultArray in resultArays)
            {
                resultArray.ParentFunc = this;
            }

            return resultArays;
        }

        private void BackwardCpu(NdArray[] ys, NdArray x)
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
