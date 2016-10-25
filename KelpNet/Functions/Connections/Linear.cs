using System;
using KelpNet.Common;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Linear : NeedPreviousDataFunction
    {
        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, string name = "Linear") : base(name)
        {
            this.W = NdArray.Zeros(outputCount, inputCount);
            this.gW = NdArray.ZerosLike(this.W);
            Parameters.Add(new OptimizeParameter(this.W, this.gW, Name + " W"));

            //Zeroバイアス
            this.b = NdArray.Zeros(outputCount);

            if (initialW == null)
            {
                InitWeight(this.W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                Buffer.BlockCopy(initialW, 0, this.W.Data, 0, sizeof(double) * initialW.Length);
            }

            if (!noBias)
            {
                this.gb = NdArray.ZerosLike(this.b);

                if (initialb != null)
                {
                    Buffer.BlockCopy(initialb, 0, this.b.Data, 0, sizeof(double) * initialb.Length);
                }

                Parameters.Add(new OptimizeParameter(this.b, this.gb, Name + " b"));
            }

            OutputCount = outputCount;
            InputCount = inputCount;
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] output = new double[OutputCount];

            for (int j = 0; j < OutputCount; j++)
            {
                for (int k = 0; k < InputCount; k++)
                {
                    output[j] += x.Data[k] * this.W.Get(j, k);
                }

                output[j] += this.b.Data[j];
            }

            return NdArray.FromArray(output);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            for (int j = 0; j < prevInput.Length; j++)
            {
                for (int k = 0; k < gy.Length; k++)
                {
                    this.gW.Data[this.gW.GetIndex(k, j)] += prevInput.Data[j] * gy.Data[k];
                }
            }

            double[] gxData = new double[InputCount];

            for (int j = 0; j < this.W.Shape[0]; j++)
            {
                for (int k = 0; k < this.W.Shape[1]; k++)
                {
                    gxData[k] += this.W.Get(j, k) * gy.Data[j];
                }
            }

            if (this.gb != null)
            {
                for (int j = 0; j < gy.Length; j++)
                {
                    this.gb.Data[j] += gy.Data[j];
                }
            }

            return new NdArray(gxData, new[] { 1, InputCount });
        }
    }
}
