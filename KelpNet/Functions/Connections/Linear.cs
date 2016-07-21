using System;

namespace KelpNet.Functions.Connections
{
    public class Linear : OptimizableFunction, IPredictableFunction
    {
        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null)
        {
            this.W = NdArray.Empty(outputCount, inputCount);
            this.gW = NdArray.ZerosLike(W);

            if (initialW == null)
            {
                InitWeight(W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                Buffer.BlockCopy(initialW, 0, W.Data, 0, sizeof (double)*initialW.Length);
            }

            if (!noBias)
            {
                this.b = NdArray.Empty(outputCount);
                this.gb = NdArray.ZerosLike(b);

                if (initialb == null)
                {
                    InitWeight(b);
                }
                else
                {
                    Buffer.BlockCopy(initialb, 0, b.Data, 0, sizeof (double)*initialb.Length);
                }
            }

            this.OutputCount = outputCount;
            this.InputCount = inputCount;
        }

        public override NdArray Forward(NdArray x)
        {
            this.PrevInput = new NdArray(x);

            NdArray output = NdArray.Empty(1, this.OutputCount);
            NdArray bias = this.b != null ? b : NdArray.Zeros(OutputCount);

            for (int i = 0; i < this.OutputCount; i++)
            {
                for (int j = 0; j < this.W.Shape[1]; j++)
                {
                    output.Data[i] += x.Data[j] * this.W.Get(i, j);
                }

                output.Data[i] += bias.Get(i);
            }

            return output;
        }

        public override NdArray Backward(NdArray gy)
        {
            for (int i = 0; i < this.PrevInput.Length; i++)
            {
                for (int j = 0; j < gy.Length; j++)
                {
                    this.gW.Data[gW.GetIndex(j, i)] += this.PrevInput.Data[i] * gy.Data[j];
                }
            }

            NdArray gx = NdArray.Empty(1, this.InputCount);

            for (int i = 0; i < this.W.Shape[0]; i++)
            {
                for (int j = 0; j < this.W.Shape[1]; j++)
                {
                    gx.Data[j] += this.W.Get(i, j) * gy.Data[i];
                }
            }

            if (this.b != null)
            {
                for (int j = 0; j < gy.Length; j++)
                {
                    gb.Data[j] += gy.Data[j];
                }
            }

            return gx;
        }

        public NdArray Predict(NdArray input)
        {
            return Forward(input);
        }
    }
}
