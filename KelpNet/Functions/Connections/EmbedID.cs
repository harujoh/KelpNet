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

            for (int i = 0; i < x.Length; i++)
            {
                for (int j = 0; j < OutputCount; j++)
                {
                    result.Data[i * OutputCount + j] = this.W.Data[(int)x.Data[i] * OutputCount + j];
                }
            }

            return result;
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            for (int i = 0; i < prevInput.Length; i++)
            {
                for (int j = 0; j < OutputCount; j++)
                {
                    this.gW.Data[(int)prevInput.Data[i] * OutputCount + j] += gy.Data[i + j];
                }
            }

            return null;
        }
    }
}
