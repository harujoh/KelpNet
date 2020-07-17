using System;

#if DOUBLE
using Real = System.Double;
#elif NETSTANDARD2_1
using Real = System.Single;
using Math = System.MathF;
#else
using Real = System.Single;
using Math = KelpNet.MathF;
#endif

namespace KelpNet
{
#if !DOUBLE
    public class SoftmaxCrossEntropy<T> : LossFunction<T, int> where T : unmanaged, IComparable<T>
    {
        public SoftmaxCrossEntropy()
        {
            switch (this)
            {
                case SoftmaxCrossEntropy<float> softmaxCrossEntropyF:
                    softmaxCrossEntropyF.EvaluateFunc = SoftmaxCrossEntropyF.Evaluate;
                    break;

                case SoftmaxCrossEntropy<double> softmaxCrossEntropyD:
                    softmaxCrossEntropyD.EvaluateFunc = SoftmaxCrossEntropyD.Evaluate;
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class SoftmaxCrossEntropyD
#else
    public static class SoftmaxCrossEntropyF
#endif
    {
        public static Real Evaluate(NdArray<Real>[] input, NdArray<int>[] teachSignal)
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

                Real[] m = new Real[dataLen * input[k].BatchCount];

                for (int b = 0; b < input[k].BatchCount; b++)
                {
                    for (int i = 0; i < chLen; i++)
                    {
                        for (int j = 0; j < dataLen; j++)
                        {
                            int dataIndex = b * input[k].Length + i * dataLen + j;
                            if (m[b * m.Length / input[k].BatchCount + j] < input[k].Data[dataIndex])
                            {
                                m[b * m.Length / input[k].BatchCount + j] = input[k].Data[dataIndex];
                            }
                        }
                    }
                }

                Real[] y = new Real[input[k].Data.Length];
                Real[] s = new Real[dataLen * input[k].BatchCount];

                for (int b = 0; b < input[k].BatchCount; b++)
                {
                    for (int i = 0; i < chLen; i++)
                    {
                        for (int j = 0; j < dataLen; j++)
                        {
                            int dataIndex = b * input[k].Length + i * dataLen + j;
                            y[dataIndex] = Math.Exp(input[k].Data[dataIndex] - m[b * m.Length / input[k].BatchCount + j]);
                            s[b * m.Length / input[k].BatchCount + j] += y[dataIndex];
                        }
                    }
                }

                for (int i = 0; i < s.Length; i++)
                {
                    if (s[i] == 0) s[i] += Real.Epsilon;
                    m[i] += Math.Log(s[i]);
                }

                for (int b = 0; b < input[k].BatchCount; b++)
                {
                    for (int i = 0; i < chLen; i++)
                    {
                        for (int j = 0; j < dataLen; j++)
                        {
                            int dataIndex = b * input[k].Length + i * dataLen + j;
                            y[dataIndex] = input[k].Data[dataIndex] - m[b * m.Length / input[k].BatchCount + j];
                        }
                    }
                }

                Real[] log_p = new Real[teachSignal[k].Data.Length];

                for (int b = 0; b < teachSignal[k].BatchCount; b++)
                {
                    for (int j = 0; j < teachSignal[k].Length; j++)
                    {
                        int index = b * teachSignal[k].Length + j;
                        int tb = teachSignal[k].Data[index];

                        log_p[index] = y[b * y.Length / input[k].BatchCount + tb * dataLen + j];
                    }
                }

                Real coef = 1.0f / teachSignal[k].Data.Length;

                for (int i = 0; i < log_p.Length; i++)
                {
                    localLoss += log_p[i];
                }

                resultLoss += localLoss * -coef;


                Real[] gx = new Real[input[k].Data.Length];

                for (int i = 0; i < gx.Length; i++)
                {
                    gx[i] = Math.Exp(y[i]) * coef;
                }

                for (int b = 0; b < teachSignal[k].BatchCount; b++)
                {
                    for (int j = 0; j < teachSignal[k].Length; j++)
                    {
                        int index = b * teachSignal[k].Length + j;
                        int tb = teachSignal[k].Data[index];

                        gx[b * y.Length / input[k].BatchCount + tb * dataLen + j] -= coef;
                    }
                }

                input[k].Grad = gx;
            }

            resultLoss /= input.Length;

            return resultLoss;
        }
    }
}
