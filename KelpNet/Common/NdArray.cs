using System;
using System.Linq;
using System.Text;

namespace KelpNet.Common
{
    [Serializable]
    public class NdArray
    {
        public Real[] Data;
        public Real[] Grad;

        public int[] Shape;

        public int BatchCount;
        public int Length;

        //Updateを行わずに実行されたBackwardの回数をカウントし、バッチ更新時に使用する
        public int TrainCount;

        public NdArray(Array data)
        {
            Real[] resultData = Real.GetArray(data);

            int[] resultShape = new int[data.Rank];
            for (int i = 0; i < data.Rank; i++)
            {
                resultShape[i] = data.GetLength(i);
            }

            this.Data = resultData;
            this.Shape = resultShape;
            this.Length = Data.Length;
            this.Grad = new Real[this.Length];
            this.BatchCount = 1;
            this.TrainCount = 0;
        }

        public NdArray(params int[] shape)
        {
            this.Data = new Real[ShapeToArrayLength(shape)];
            this.Shape = shape.ToArray();
            this.Length = Data.Length;
            this.BatchCount = 1;
            this.Grad = new Real[this.Length];
            this.TrainCount = 0;
        }

        public NdArray(Real[] data, int[] shape, int batchCount = 1)
        {
            this.Shape = shape.ToArray();
            this.Length = ShapeToArrayLength(this.Shape);
            this.BatchCount = batchCount;
            this.Data = data.ToArray();
            this.Grad = new Real[this.Length];
            this.TrainCount = 0;
        }

        public NdArray(int[] shape, int batchCount)
        {
            this.Shape = shape.ToArray();
            this.Length = ShapeToArrayLength(this.Shape);
            this.BatchCount = batchCount;
            this.Data = new Real[this.Length * batchCount];
            this.Grad = new Real[this.Length];
            this.TrainCount = 0;
        }

        static int ShapeToArrayLength(params int[] shapes)
        {
            int result = 1;

            foreach (int shape in shapes)
            {
                result *= shape;
            }

            return result;
        }

        public NdArray[] DivideArrays()
        {
            NdArray[] result = new NdArray[BatchCount];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = GetSingleArray(i);
            }

            return result;
        }

        public NdArray GetSingleArray(int i)
        {
            Real[] data = new Real[this.Length];
            Array.Copy(this.Data, i * this.Length, data, 0, this.Length);

            return new NdArray(data, this.Shape);
        }

        public static NdArray FromArrays(Array[] arrays)
        {
            int[] resultShape = new int[arrays[0].Rank];

            for (int i = 0; i < arrays[0].Rank; i++)
            {
                resultShape[i] = arrays[0].GetLength(i);
            }

            int length = arrays[0].Length;
            Real[] result = new Real[length * arrays.Length];

            for (int i = 0; i < arrays.Length; i++)
            {
                Array.Copy(Real.GetArray(arrays[i]), 0, result, length * i, length);
            }

            return new NdArray(result, resultShape, arrays.Length);
        }

        public static NdArray Convert(Real[] data, int[] shape, int batchCount)
        {
            return new NdArray(shape, batchCount) { Data = data };
        }

        public static NdArray ZerosLike(NdArray baseArray)
        {
            return new NdArray(baseArray.Shape.ToArray(), baseArray.BatchCount);
        }

        public void CountUp()
        {
            TrainCount++;
        }

        //傾きの補正
        public bool Reduce()
        {
            if (this.TrainCount > 0)
            {
                for (int i = 0; i < this.Grad.Length; i++)
                {
                    this.Grad[i] /= this.TrainCount;
                }

                return true;
            }

            return false;
        }

        //傾きの初期化
        public void ClearGrad()
        {
            this.Grad = new Real[this.Data.Length];

            //カウンタをリセット
            this.TrainCount = 0;
        }

        public override string ToString()
        {
            return ToString(this.Data);
        }

        public string ToString(string format)
        {
            if (format == "Grad")
                return ToString(this.Grad);
            return ToString(this.Data);
        }

        public string ToString(Real[] datas)
        {
            StringBuilder sb = new StringBuilder();

            int intMaxLength = 0;   //整数部の最大値
            int realMaxLength = 0;   //小数点以下の最大値
            bool isExponential = false; //指数表現にするか

            foreach (Real data in datas)
            {
                string[] divStr = data.ToString().Split('.');
                intMaxLength = Math.Max(intMaxLength, divStr[0].Length);
                if (divStr.Length > 1 && !isExponential)
                {
                    isExponential = divStr[1].Contains("E");
                }

                if (realMaxLength != 8 && divStr.Length == 2)
                {
                    realMaxLength = Math.Max(realMaxLength, divStr[1].Length);
                    if (realMaxLength > 8) realMaxLength = 8;
                }
            }

            //配列の約数を取得
            int[] commonDivisorList = new int[this.Shape.Length];

            //一個目は手動取得
            commonDivisorList[0] = this.Shape[this.Shape.Length - 1];
            for (int i = 1; i < this.Shape.Length; i++)
            {
                commonDivisorList[i] = commonDivisorList[i - 1] * this.Shape[this.Shape.Length - i - 1];
            }

            if (this.BatchCount > 1)
            {
                sb.Append("{");
            }

            for (int batch = 0; batch < this.BatchCount; batch++)
            {
                int indexOffset = batch * Length;
                //先頭の括弧
                for (int i = 0; i < this.Shape.Length; i++)
                {
                    sb.Append("[");
                }

                int closer = 0;
                for (int i = 0; i < Length; i++)
                {
                    string[] divStr;
                    if (isExponential)
                    {
                        divStr = string.Format("{0:0.00000000e+00}", datas[indexOffset + i]).Split('.');
                    }
                    else
                    {
                        divStr = datas[indexOffset + i].ToString().Split('.');
                    }

                    //最大文字数でインデントを揃える
                    for (int j = 0; j < intMaxLength - divStr[0].Length; j++)
                    {
                        sb.Append(" ");
                    }
                    sb.Append(divStr[0]);
                    if (realMaxLength != 0)
                    {
                        sb.Append(".");
                    }
                    if (divStr.Length == 2)
                    {
                        sb.Append(divStr[1].Length > 8 && !isExponential ? divStr[1].Substring(0, 8) : divStr[1]);
                        for (int j = 0; j < realMaxLength - divStr[1].Length; j++)
                        {
                            sb.Append(" ");
                        }
                    }
                    else
                    {
                        for (int j = 0; j < realMaxLength; j++)
                        {
                            sb.Append(" ");
                        }
                    }

                    //約数を調査してピッタリなら括弧を出力
                    if (i != Length - 1)
                    {
                        foreach (int commonDivisor in commonDivisorList)
                        {
                            if ((i + 1) % commonDivisor == 0)
                            {
                                sb.Append("]");
                                closer++;
                            }
                        }

                        sb.Append(" ");

                        if ((i + 1) % commonDivisorList[0] == 0)
                        {
                            //閉じ括弧分だけ改行を追加
                            for (int j = 0; j < closer; j++)
                            {
                                sb.Append("\n");
                            }
                            closer = 0;

                            //括弧前のインデント
                            foreach (int commonDivisor in commonDivisorList)
                            {
                                if ((i + 1) % commonDivisor != 0)
                                {
                                    sb.Append(" ");
                                }
                            }
                        }

                        foreach (int commonDivisor in commonDivisorList)
                        {
                            if ((i + 1) % commonDivisor == 0)
                            {
                                sb.Append("[");
                            }
                        }
                    }
                }

                //後端の括弧
                for (int i = 0; i < this.Shape.Length; i++)
                {
                    sb.Append("]");
                }

                if (batch < this.BatchCount - 1)
                {
                    sb.Append("},\n{");
                }
            }

            if (this.BatchCount > 1)
            {
                sb.Append("}");
            }

            return sb.ToString();
        }
    }
}
