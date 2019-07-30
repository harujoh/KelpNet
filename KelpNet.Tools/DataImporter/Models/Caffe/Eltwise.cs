using System;
using System.Collections.Generic;

namespace KelpNet.Tools
{
    public class Eltwise : MultiInputFunction
    {
        private const string FUNCTION_NAME = "Eltwise";

        List<int[]> PrevOutputIndex = new List<int[]>();

        private EltwiseParameter.EltwiseOp _operation;
        private float[] _coeffs;

        public Eltwise(EltwiseParameter.EltwiseOp operation, float[] coeffs, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this._operation = operation;
            this._coeffs = coeffs;
        }

        protected override NdArray MultiInputForward(params NdArray[] xs)
        {
            Real[] result = new Real[xs[0].Data.Length];

            switch (_operation)
            {
                case EltwiseParameter.EltwiseOp.Prod:
                    Array.Copy(xs[0].Data, result, result.Length);
                    for (int i = 1; i < xs.Length; i++)
                    {
                        for (int j = 0; j < result.Length; j++)
                        {
                            result[j] *= xs[i].Data[j];
                        }
                    }
                    break;

                case EltwiseParameter.EltwiseOp.Sum:
                    if (this._coeffs != null)
                    {
                        for (int i = 0; i < xs.Length; i++)
                        {
                            for (int j = 0; j < result.Length; j++)
                            {
                                result[j] += xs[i].Data[j] * _coeffs[i];
                            }
                        }
                    }
                    else
                    {
                        for (int i = 0; i < xs.Length; i++)
                        {
                            for (int j = 0; j < result.Length; j++)
                            {
                                result[j] += xs[i].Data[j];
                            }
                        }
                    }
                    break;

                case EltwiseParameter.EltwiseOp.Max:
                    Array.Copy(xs[0].Data, result, result.Length);
                    int[] outputIndex = new int[result.Length];

                    for (int i = 1; i < xs.Length; i++)
                    {
                        for (int j = 0; j < result.Length; j++)
                        {
                            if (result[j] < xs[i].Data[j])
                            {
                                outputIndex[j] = i;
                                result[j] = xs[i].Data[j];
                            }
                        }
                    }

                    PrevOutputIndex.Add(outputIndex);
                    break;
            }

            return NdArray.Convert(result, xs[0].Shape, xs[0].BatchCount, this);
        }

        protected override void MultiOutputBackward(NdArray y, params NdArray[] xs)
        {
            Real[][] result = new Real[xs.Length][];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = new Real[xs[i].Data.Length];
            }

            switch (_operation)
            {
                case EltwiseParameter.EltwiseOp.Prod:
                    for (int i = 0; i < result.Length; i++)
                    {
                        Array.Copy(y.Grad, result[i], y.Grad.Length);
                        for (int j = 0; j < xs.Length; j++)
                        {
                            if (i != j)
                            {
                                for (int k = 0; k < result[i].Length; k++)
                                {
                                    result[i][k] *= xs[j].Data[k];
                                }
                            }
                        }
                    }
                    break;

                case EltwiseParameter.EltwiseOp.Sum:
                    for (int i = 0; i < result.Length; i++)
                    {
                        Array.Copy(y.Grad, result[i], result[i].Length);
                    }
                    break;

                case EltwiseParameter.EltwiseOp.Max:
                    var prevOutputIndex = PrevOutputIndex[PrevOutputIndex.Count - 1];
                    PrevOutputIndex.RemoveAt(PrevOutputIndex.Count - 1);

                    for (int i = 0; i < prevOutputIndex.Length; i++)
                    {
                        result[prevOutputIndex[i]][i] = y.Grad[i];
                    }
                    break;
            }

            for (int i = 0; i < xs.Length; i++)
            {
                for (int j = 0; j < xs[i].Grad.Length; j++)
                {
                    xs[i].Grad[j] += result[i][j];
                }
            }
        }
    }
}
