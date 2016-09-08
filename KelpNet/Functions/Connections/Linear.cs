using System;

namespace KelpNet.Functions.Connections
{
    public class Linear : PredictableFunction
    {
        public NdArray W;
        public NdArray b;

        public NdArray gW;
        public NdArray gb;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null)
        {
            this.W = NdArray.Empty(outputCount, inputCount);
            this.gW = NdArray.ZerosLike(W);
            Parameters.Add(new Parameter(this.W, this.gW));

            if (initialW == null)
            {
                InitWeight(W);
            }
            else
            {
                //単純に代入しないのはサイズのチェックを兼ねるため
                Buffer.BlockCopy(initialW, 0, W.Data, 0, sizeof(double) * initialW.Length);
            }

            if (!noBias)
            {
                this.b = NdArray.Zeros(outputCount);
                this.gb = NdArray.ZerosLike(b);

                if (initialb != null)
                {
                    Buffer.BlockCopy(initialb, 0, b.Data, 0, sizeof (double)*initialb.Length);
                }

                Parameters .Add(new Parameter(this.b, this.gb));
            }

            this.OutputCount = outputCount;
            this.InputCount = inputCount;
        }

        public override NdArray Forward(NdArray x,int batchID=0)
        {
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

        public override NdArray Backward(NdArray gy, int batchID = 0)
        {
            for (int i = 0; i < PrevInput[batchID].Length; i++)
            {
                for (int j = 0; j < gy.Length; j++)
                {
                    this.gW.Data[gW.GetIndex(j, i)] += PrevInput[batchID].Data[i] * gy.Data[j];
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
    }
}
