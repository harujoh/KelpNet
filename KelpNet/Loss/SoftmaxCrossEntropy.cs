using System;

namespace KelpNet
{
    public class SoftmaxCrossEntropy : LossFunction
    {
        public override Real Evaluate(NdArray[] input, params NdArray[] teachSignal)
        {
            Real resultLoss = 0;

#if DEBUG
            if (input.Length != teachSignal.Length) throw new Exception("入力と教師信号のサイズが異なります");
#endif
            for (int k = 0; k < input.Length; k++)
            {
                Real localLoss = 0;

                int chLen = input[k].Shape[0];
                int dataLen = 1;
                for (int i = 1; i < input[k].Shape.Length; i++)
                {
                    dataLen *= input[k].Shape[i];
                }

                NdArray m = NdArray.Max(input[k], new[] { 0 }, true);
                NdArray y = new NdArray(input[k].Shape, input[k].BatchCount);

                for (int b = 0; b < input[k].BatchCount; b++)
                {
                    for (int i = 0; i < chLen; i++)
                    {
                        for (int j = 0; j < dataLen; j++)
                        {
                            int dataIndex = b * input[k].Length + i * dataLen + j;
                            y.Data[dataIndex] = Math.Exp(input[k].Data[dataIndex] - m.Data[b * m.Length + j]);
                        }
                    }
                }

                NdArray s = NdArray.Sum(y, new[] { 0 }, true);

                for (int i = 0; i < s.Data.Length; i++)
                {
                    m.Data[i] += Math.Log(s.Data[i]);
                }

                for (int b = 0; b < input[k].BatchCount; b++)
                {
                    for (int i = 0; i < chLen; i++)
                    {
                        for (int j = 0; j < dataLen; j++)
                        {
                            int dataIndex = b * input[k].Length + i * dataLen + j;
                            y.Data[dataIndex] = input[k].Data[dataIndex] - m.Data[b * m.Length + j];
                        }
                    }
                }

                Real[] log_p = new Real[teachSignal[k].Data.Length];

                for (int b = 0; b < teachSignal[k].BatchCount; b++)
                {
                    for (int j = 0; j < teachSignal[k].Length; j++)
                    {
                        int index = b * teachSignal[k].Length + j;
                        int tb = (int)teachSignal[k].Data[index];

                        log_p[index] = y.Data[b * y.Length + tb * dataLen + j];
                    }
                }

                Real coef = 1.0 / teachSignal[k].Data.Length;

                for (int i = 0; i < log_p.Length; i++)
                {
                    localLoss += log_p[i];
                }

                resultLoss += localLoss * -coef;


                Real[] gx = new Real[input[k].Data.Length];

                for (int i = 0; i < gx.Length; i++)
                {
                    gx[i] = Math.Exp(y.Data[i]) * coef;
                }

                for (int b = 0; b < teachSignal[k].BatchCount; b++)
                {
                    for (int j = 0; j < teachSignal[k].Length; j++)
                    {
                        int index = b * teachSignal[k].Length + j;
                        int tb = (int)teachSignal[k].Data[index];

                        gx[b * y.Length + tb * dataLen + j] -= coef;
                    }
                }

                input[k].Grad = gx;
            }

            resultLoss /= input.Length;

            return resultLoss;
        }
    }
}