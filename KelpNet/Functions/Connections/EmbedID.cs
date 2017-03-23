using System;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Tools;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class EmbedID : NeedPreviousInputFunction
    {
        public NdArray W;
        public NdArray gW;

        public EmbedID(int inputCount, int outputCount, Array initialW = null, string name = "EmbedID", bool isGpu = false) : base(name, isGpu, inputCount, outputCount)
        {
            this.W = NdArray.Zeros(inputCount, outputCount);
            this.gW = NdArray.ZerosLike(this.W);

            if (initialW == null)
            {
                Initializer.InitWeight(this.W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                Buffer.BlockCopy(initialW, 0, this.W.Data, 0, sizeof(double) * initialW.Length);
            }

            this.Parameters = new[] { new FunctionParameter(this.W, this.gW, this.Name + " W") };
        }

        protected override BatchArray NeedPreviousForward(BatchArray x)
        {
            double[] result = new double[x.Data.Length * this.OutputCount];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    for (int j = 0; j < this.OutputCount; j++)
                    {
                        result[i * this.OutputCount + j + b * x.Length * this.OutputCount] = this.W.Data[(int)x.Data[i + b * x.Length] * this.OutputCount + j];
                    }
                }
            }

            return BatchArray.Convert(result, new[] { x.Length, this.OutputCount }, x.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput)
        {
            for (int b = 0; b < gy.BatchCount; b++)
            {
                for (int i = 0; i < prevInput.Length; i++)
                {
                    for (int j = 0; j < this.OutputCount; j++)
                    {
                        this.gW.Data[(int)prevInput.Data[i + b * prevInput.Length] * this.OutputCount + j] += gy.Data[i + j + b * gy.Length];
                    }
                }
            }

            return null;
        }
    }
}
