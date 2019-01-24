using System;

namespace KelpNet
{
    [Serializable]
    public class SplitFunction<T> : MultiOutputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "SplitFunction";
        private readonly int _splitNum;

        public FunctionStack<T>[] SplitedFunctions;

        public SplitFunction(int splitNum = 2, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._splitNum = splitNum;
            SplitedFunctions = new FunctionStack<T>[splitNum];

            for (int i = 0; i < SplitedFunctions.Length; i++)
            {
                SplitedFunctions[i] = new FunctionStack<T>(new Function<T>[] { }, name + i, new[] { inputNames[0] }, new[] { outputNames[i] });
            }

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        private NdArray<T>[] ForwardCpu(NdArray<T> x)
        {
            NdArray<T>[] result = new NdArray<T>[_splitNum];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = SplitedFunctions[i].Forward(x)[0];
            }

            return result;
        }

        private void BackwardCpu(NdArray<T>[] ys, NdArray<T> x)
        {
        }

        public override NdArray<T>[] Predict(params NdArray<T>[] xs)
        {
            NdArray<T>[] result = new NdArray<T>[_splitNum];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = SplitedFunctions[i].Predict(xs[0])[0];
            }

            return result;
        }
    }
}
