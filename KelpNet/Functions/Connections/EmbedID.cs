using System;

namespace KelpNet
{
    [Serializable]
    public class EmbedID<T> : SingleInputFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "EmbedID";

        public NdArray<T> Weight;

        public readonly int InputCount;
        public readonly int OutputCount;

        public EmbedID(int inputCount, int outputCount, Real<T>[,] initialW = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.InputCount = inputCount;
            this.OutputCount = outputCount;

            this.Weight = new NdArray<T>(inputCount, outputCount);
            this.Weight.Name = this.Name + " Weight";

            if (initialW == null)
            {
                Initializer<T>.InitWeight(this.Weight);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                this.Weight.Data = Real<T>.GetArray(initialW);
            }

            this.Parameters = new[] { this.Weight };

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        protected NdArray<T> NeedPreviousForwardCpu(NdArray<T> x)
        {
            RealArray<T> result = new T[x.DataLength * this.OutputCount];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    for (int j = 0; j < this.OutputCount; j++)
                    {
                        result[i * this.OutputCount + j + b * x.Length * this.OutputCount] = this.Weight.Data[(int)x.Data[i + b * x.Length] * this.OutputCount + j];
                    }
                }
            }

            return NdArray<T>.Convert(result, new[] { x.Length, this.OutputCount }, x.BatchCount, this);
        }

        protected void NeedPreviousBackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    for (int j = 0; j < this.OutputCount; j++)
                    {
                        this.Weight.Grad[(int)x.Data[i + b * x.Length] * this.OutputCount + j] += y.Grad[i + j + b * y.Length];
                    }
                }
            }
        }
    }
}
