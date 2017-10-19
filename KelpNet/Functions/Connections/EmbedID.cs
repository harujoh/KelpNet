using System;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class EmbedID : SingleInputFunction
    {
        public NdArray Weight;

        public readonly int InputCount;
        public readonly int OutputCount;

        public EmbedID(int inputCount, int outputCount, Real[,] initialW = null, string name = "EmbedID") : base(name)
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
                this.Weight.Data = Real.GetArray(initialW);
            }

            this.Parameters = new[] { this.Weight };

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        protected NdArray NeedPreviousForwardCpu(NdArray x)
        {
            Real[] result = new Real[x.Data.Length * this.OutputCount];

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

            return NdArray.Convert(result, new[] { x.Length, this.OutputCount }, x.BatchCount, this);
        }

        protected void NeedPreviousBackwardCpu(NdArray y, NdArray x)
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
