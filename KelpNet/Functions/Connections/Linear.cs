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

            this.Parameters = new FunctionParameter[noBias ? 1 : 2];

            if (initialW == null)
            {
                InitWeight(this.W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                Buffer.BlockCopy(initialW, 0, this.W.Data, 0, sizeof(double) * initialW.Length);
            }

            this.Parameters[0] = new FunctionParameter(this.W, this.gW, this.Name + " W");

            //noBias=trueでもbiasを用意して更新しない
            this.b = NdArray.Zeros(outputCount);
            this.gb = NdArray.ZerosLike(this.b);

            if (!noBias)
            {
                if (initialb != null)
                {
                    Buffer.BlockCopy(initialb, 0, this.b.Data, 0, sizeof(double) * initialb.Length);
                }

                this.Parameters[1] = new FunctionParameter(this.b, this.gb, this.Name + " b");
            }
        }

        protected override NdArray NeedPreviousForward(NdArray x)
        {
            //バイアスを最初からコピーして入れておく
            double[] output = this.b.Data.ToArray();

            for (int i = 0; i < this.OutputCount; i++)
            {
                int indexOffset = this.InputCount * i;

                for (int j = 0; j < this.InputCount; j++)
                {
                    output[i] += x.Data[j] * this.W.Data[indexOffset + j];
                }
            }

            return NdArray.FromArray(output);
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput)
        {
            double[] gxData = new double[this.InputCount];

            for (int i = 0; i < gy.Length; i++)
            {
                int indexOffset = this.InputCount * i;
                double gyData = gy.Data[i];

                for (int j = 0; j < this.InputCount; j++)
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
