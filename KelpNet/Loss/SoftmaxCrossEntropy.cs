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
                Real localloss = 0;
                Real[] gx = new Real[input[k].Data.Length];

                for (int b = 0; b < input[k].BatchCount; b++)
                {
                    Real maxIndex = 0;

                    for (int i = 0; i < teachSignal[k].Length; i++)
                    {
                        if (maxIndex < teachSignal[k].Data[i + b * teachSignal[k].Length])
                        {
                            maxIndex = teachSignal[k].Data[i + b * teachSignal[k].Length];
                        }
                    }

                    Real[] logY = new Real[input[k].Length];
                    Real y = 0;
                    Real m = input[k].Data[b * input[k].Length];

                    for (int i = 1; i < input[k].Length; i++)
                    {
                        if (m < input[k].Data[i + b * input[k].Length])
                        {
                            m = input[k].Data[i + b * input[k].Length];
                        }
                    }

                    for (int i = 0; i < input[k].Length; i++)
                    {
                        y += Math.Exp(input[k].Data[i + b * input[k].Length] - m);
                    }

                    m += Math.Log(y);

                    for (int i = 0; i < input[k].Length; i++)
                    {
                        logY[i] = input[k].Data[i + b * input[k].Length] - m;
                    }

                    localloss += -logY[(int)maxIndex];


                    for (int i = 0; i < logY.Length; i++)
                    {
                        gx[i + b * input[k].Length] = Math.Exp(logY[i]);
                    }

                    gx[(int)maxIndex + b * input[k].Length] -= 1;
                }

                resultLoss += localloss / input[k].BatchCount;
                input[k].Grad = gx;
            }

            resultLoss /= input.Length;

            return resultLoss;
        }
    }
}
