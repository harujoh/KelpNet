using KelpNet.Common.Functions.Type;

namespace KelpNet.Common.Functions
{
    public class Splitter : MultiOutputFunction
    {
        const string FUNCTION_NAME = "Splitter";
        private readonly int _splitNum;

        public Splitter(int splitNum = 2, string name = FUNCTION_NAME) : base(name)
        {
            this._splitNum = splitNum;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        private NdArray[] ForwardCpu(NdArray x)
        {
            NdArray[] result = new NdArray[_splitNum];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = x;
            }

            return result;
        }

        private void BackwardCpu(NdArray[] ys, NdArray x)
        {
        }
    }
}
