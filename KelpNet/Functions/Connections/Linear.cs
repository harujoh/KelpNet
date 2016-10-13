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
            Parameters.Add(new OptimizeParameter(this.W, this.gW, this.Name + " W" ));

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

                Parameters.Add(new OptimizeParameter(this.b, this.gb, this.Name + " b"));
            }

            OutputCount = outputCount;
            InputCount = inputCount;
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            double[] output = new double[OutputCount];

            for (int i = 0; i < OutputCount; i++)
            {
                for (int j = 0; j < InputCount; j++)
                {
                    output[i] += x.Data[j] * this.W.Get(i, j);
                }

                output[i] += this.b.Data[i];
            }

            return NdArray.FromArray(output);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            for (int i = 0; i < prevInput.Length; i++)
            {
                for (int j = 0; j < gy.Length; j++)
                {
                    this.gW.Data[this.gW.GetIndex(j, i)] += prevInput.Data[i] * gy.Data[j];
                }
            }

            double[] gxData = new double[InputCount];

            for (int i = 0; i < this.W.Shape[0]; i++)
            {
                for (int j = 0; j < this.W.Shape[1]; j++)
                {
                    gxData[j] += this.W.Get(i, j) * gy.Data[i];
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
