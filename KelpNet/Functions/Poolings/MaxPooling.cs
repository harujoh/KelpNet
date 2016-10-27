using System;
using KelpNet.Common;

namespace KelpNet.Functions.Poolings
{
    [Serializable]
    public class MaxPooling : NeedPreviousDataFunction
    {
        private int _kSize;
        private int _stride;
        private int _pad;

        public MaxPooling(int ksize, int stride = 1, int pad = 0, string name = "MaxPooling") : base(name)
        {
            this._kSize = ksize;
            this._stride = stride;
            this._pad = pad;
        }

        protected override NdArray NeedPreviousForward(NdArray input)
        {
            int outputSize = (int)Math.Floor((input.Shape[2] - this._kSize + this._pad * 2.0) / this._stride) + 1;
            NdArray result = NdArray.Zeros(input.Shape[0], outputSize, outputSize);
            result.Fill(double.MinValue);

            for (int j = 0; j < input.Shape[0]; j++)
            {
                for (int y = 0; y < outputSize; y++)
                {
                    for (int x = 0; x < outputSize; x++)
                    {
                        int resultIndex = result.GetIndex(j, x, y);
                        for (int dy = 0; dy < this._kSize; dy++)
                        {
                            for (int dx = 0; dx < this._kSize; dx++)
                            {
                                int inputIndexX = x * this._stride + dx - this._pad;
                                int inputIndexY = y * this._stride + dy - this._pad;

                                if (inputIndexX >= 0 && inputIndexX < input.Shape[1] &&
                                    inputIndexY >= 0 && inputIndexY < input.Shape[2])
                                {
                                    result.Data[result.GetIndex(j, x, y)] = Math.Max(result.Data[resultIndex], input.Get(j, inputIndexX, inputIndexY));
                                }
                            }
                        }
                    }
                }
            }

            return result;
        }

        protected override NdArray NeedPreviousBackward(NdArray gy, NdArray prevInput, NdArray prevOutput)
        {
            NdArray result = NdArray.ZerosLike(prevInput);
            gy.Shape = (int[])prevOutput.Shape.Clone();

            for (int j = 0; j < result.Shape[0]; j++)
            {
                for (int y = 0; y < gy.Shape[1]; y++)
                {
                    for (int x = 0; x < gy.Shape[2]; x++)
                    {
                        //前回の入力値と出力値を比較して、同じ値のものを見つける
                        this.setresult(j, y, x, gy.Data[gy.GetIndex(j, y, x)], prevInput, prevOutput, ref result);
                    }
                }
            }

            return result;
        }

        //同じ値を複数持つ場合、左上優先にして処理を打ち切る
        //他のライブラリの実装では乱数を取って同じ値の中からどれかを選ぶ物が多い
        void setresult(int i, int y, int x, double data, NdArray PrevInput, NdArray PrevOutput, ref NdArray result)
        {
            for (int dy = 0; dy < this._kSize; dy++)
            {
                for (int dx = 0; dx < this._kSize; dx++)
                {
                    int outputIndexY = y * this._stride + dy - this._pad;
                    int outputIndexX = x * this._stride + dx - this._pad;

                    if (outputIndexY >= 0 && outputIndexY < result.Shape[1] &&
                        outputIndexX >= 0 && outputIndexX < result.Shape[2])
                    {
                        if (PrevInput.Data[PrevInput.GetIndex(i, outputIndexY, outputIndexX)].Equals(PrevOutput.Data[PrevOutput.GetIndex(i, y, x)]))
                        {
                            result.Data[result.GetIndex(i, outputIndexY, outputIndexX)] = data;
                            return;
                        }
                    }
                }
            }
        }
    }
}
