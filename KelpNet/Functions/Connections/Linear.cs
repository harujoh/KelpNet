using System;
using System.Linq;
using KelpNet.Common;

namespace KelpNet.Functions.Connections
{
    [Serializable]
    public class Linear : NeedPreviousInputFunction
    {
        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, string name = "Linear") : base(name, inputCount, outputCount)
        {
            this.W = NdArray.Zeros(outputCount, inputCount);
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

            //noBias=trueでもbiasを用意して更新しない
            this.b = NdArray.Zeros(outputCount);
            this.gb = NdArray.ZerosLike(this.b);

            if (!noBias)
            {
                if (initialb != null)
                {
                    Buffer.BlockCopy(initialb, 0, this.b.Data, 0, sizeof(double) * initialb.Length);
                }

                Parameters.Add(new OptimizeParameter(this.b, this.gb, Name + " b"));
            }
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            //バイアスを最初から入れておく
            double[] output = this.b.Data.ToArray();

            for (int i = 0; i < OutputCount; i++)
            {
                int indexOffset = InputCount * i;

                for (int j = 0; j < InputCount; j++)
                {
                    output[i] += x.Data[j] * this.W.Data[indexOffset + j];
                }
            }

            return NdArray.FromArray(output);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput)
        {
            double[] gxData = new double[InputCount];

            for (int i = 0; i < gy.Length; i++)
            {
                int indexOffset = InputCount * i;
                double gyData = gy.Data[i];

                for (int j = 0; j < InputCount; j++)
                {
                    this.gW.Data[indexOffset + j] += prevInput.Data[j] * gyData;

                    gxData[j] += this.W.Data[indexOffset + j] * gyData;
                }

                this.gb.Data[i] += gyData;
            }

            return new NdArray(gxData, prevInput.Shape);
        }
    }
}
