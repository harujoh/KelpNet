using System;
using KelpNet.Common;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class EmbedID : NeedPreviousDataFunction
    {
        public NdArray W;
        public NdArray gW;

        public EmbedID(int inputCount, int outputCount, Array initialW = null, string name = "EmbedID") : base(name)
        {
            this.W = NdArray.Zeros(inputCount, outputCount);
            this.gW = NdArray.ZerosLike(this.W);

            Parameters.Add(new OptimizeParameter(this.W, this.gW, Name + " W"));

            if (initialW == null)
            {
                InitWeight(this.W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                Buffer.BlockCopy(initialW, 0, this.W.Data, 0, sizeof(double) * initialW.Length);
            }

            OutputCount = outputCount;
            InputCount = inputCount;
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            NdArray result = NdArray.Zeros(x.Length, OutputCount);

            for (int j = 0; j < x.Length; j++)
            {
                for (int k = 0; k < OutputCount; k++)
                {
                    result.Data[j * OutputCount + k] = this.W.Data[(int)x.Data[j] * OutputCount + k];
                }
            }

            return result;
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            for (int j = 0; j < prevInput.Length; j++)
            {
                for (int k = 0; k < OutputCount; k++)
                {
                    this.gW.Data[(int)prevInput.Data[j] * OutputCount + k] += gy.Data[j + k];
                }
            }

            return NdArray.Zeros(1);
        }
    }
}
