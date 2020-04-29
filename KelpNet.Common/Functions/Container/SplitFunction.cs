using System;
using System.Runtime.Serialization;
using KelpNet.CPU;

namespace KelpNet
{
    [DataContract(Name = "SplitFunction", Namespace = "KelpNet")]
    public class SplitFunction<T> : MultiOutputFunction<T> where T:unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "SplitFunction";

        [DataMember]
        private readonly int _splitNum;

        [DataMember]
        public FunctionStack<T>[] SplitedFunctions;

        public SplitFunction(int splitNum = 2, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._splitNum = splitNum;
            SplitedFunctions = new FunctionStack<T>[splitNum];

            for (int i = 0; i < SplitedFunctions.Length; i++)
            {
                SplitedFunctions[i] = new FunctionStack<T>(new Function<T>[] { }, name + i, new[] { inputNames[0] }, new[] { outputNames[i] });
            }

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            SplitedFunctions = new FunctionStack<T>[this._splitNum];

            for (int i = 0; i < SplitedFunctions.Length; i++)
            {
                SplitedFunctions[i] = new FunctionStack<T>(new Function<T>[] { }, this.Name + i, new[] { this.InputNames[0] }, new[] { this.OutputNames[i] });
            }

            this.SingleInputForward = SingleInputForwardSF;
            this.MultiOutputBackward = MultiOutputBackwardSF;
            this.Predict = PredictSF;
        }

        protected NdArray<T>[] SingleInputForwardSF(NdArray<T> x)
        {
            NdArray<T>[] result = new NdArray<T>[_splitNum];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = SplitedFunctions[i].Forward(x)[0];
            }

            return result;
        }

        protected void MultiOutputBackwardSF(NdArray<T>[] ys, NdArray<T> x)
        {
        }

        public NdArray<T>[] PredictSF(params NdArray<T>[] xs)
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
