using System;

namespace KelpNet
{
    [Serializable]
    public class EmbedID : SingleInputFunction
    {
        const string FUNCTION_NAME = "EmbedID";

        public NdArray Weight;

        public int InputCount;
        public int OutputCount;

        public EmbedID(int inputCount, int outputCount, Real[,] initialW = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.InputCount = inputCount;
            this.OutputCount = outputCount;

            this.Weight = new NdArray(inputCount, outputCount);
            this.Weight.Name = this.Name + " Weight";

            if (initialW == null)
            {
                Initializer.InitWeight(this.Weight);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                this.Weight.Data = Real.ToRealArray(initialW);
            }

            this.Parameters = new[] { this.Weight };
        }

        public override NdArray SingleInputForward(NdArray x)
        {
            Real[] result = new Real[x.Data.Length * this.OutputCount];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    for (int j = 0; j < this.OutputCount; j++)
                    {
                        result[i * this.OutputCount + j + b * x.Length * this.OutputCount] = this.Weight.Data[(int)x.Data[b * x.Length + i] * this.OutputCount + j];
                    }
                }
            }

            return NdArray.Convert(result, new[] { x.Length, this.OutputCount }, x.BatchCount, this);
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    for (int j = 0; j < this.OutputCount; j++)
                    {
                        this.Weight.Grad[(int)x.Data[b * x.Length + i] * this.OutputCount + j] += y.Grad[i * this.OutputCount + j + b * x.Length * this.OutputCount];
                    }
                }
            }
        }
    }
}
