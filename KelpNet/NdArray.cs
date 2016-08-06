using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace KelpNet
{
    //NumpyのNdArrayを模したクラス
    //N次元のArrayクラスを入力に取り、内部的には1次元配列として保持する事で動作を模倣している
    public class NdArray
    {
        public double[] Data;
        public int[] Shape;

        public NdArray(double[] data, int[] shape)
        {
            this.Data = data;
            this.Shape = shape;
        }

        public NdArray(NdArray ndArray)
        {
            this.Shape = new int[ndArray.Shape.Length];
            Array.Copy(ndArray.Shape, this.Shape, this.Shape.Length);

            this.Data = new double[this.Shape.Aggregate(1, (current, val) => current * val)];
            Array.Copy(ndArray.Data, this.Data, this.Data.Length);
        }

        public static NdArray EmptyLike(NdArray baseArray)
        {
            return new NdArray(new double[GetArrayLength(baseArray.Shape)], baseArray.Shape);
        }

        public static NdArray ZerosLike(NdArray baseArray)
        {
            int length = GetArrayLength(baseArray.Shape);
            double[] resutlArray = new double[length]; 

            for (int i = 0; i < length; i++)
            {
                resutlArray[i] = 0;
            }

            return new NdArray(resutlArray, baseArray.Shape);
        }

        public static NdArray Empty(params int[] shape)
        {
            return new NdArray(new double[GetArrayLength(shape)], shape);
        }

        public static NdArray Ones(params int[] shape)
        {
            int length = GetArrayLength(shape);
            double[] resutlArray = new double[length];

            for (int i = 0; i < length; i++)
            {
                resutlArray[i] = 1;
            }

            return new NdArray(resutlArray, shape);
        }

        public static NdArray Zeros(params int[] shape)
        {
            int length = GetArrayLength(shape);
            double[] resutlArray = new double[length];

            for (int i = 0; i < length; i++)
            {
                resutlArray[i] = 0;
            }

            return new NdArray(resutlArray, shape);
        }

        static int GetArrayLength(params int[] shapes)
        {
            int result = 1;

            foreach (int shape in shapes)
            {
                result *= shape;
            }

            return result;
        }

        public static NdArray FromArray(Array data)
        {
            double[] resultData;
            int[] resultShape;

            if (data.Rank == 1)
            {
                resultData = new double[data.Length];
                data.CopyTo(resultData, 0);

                resultShape = new[] { data.Length };
            }
            else
            {
                resultShape = new int[data.Rank];

                for (int i = 0; i < data.Rank; i++)
                {
                    resultShape[i] = data.GetLength(i);
                }

                resultData = new double[data.Length];

                //int -> doubleの指定ミスで例外がポコポコ出るので、ここで吸収
                if (data.GetType().GetElementType() != typeof(double))
                {
                    Type arrayType = data.GetType().GetElementType();
                    //一次元の長さの配列を用意
                    var array = Array.CreateInstance(arrayType, data.Length);
                    //一次元化して
                    Buffer.BlockCopy(data, 0, array, 0, Marshal.SizeOf(arrayType) * resultData.Length);

                    //型変換しつつコピー
                    array.CopyTo(resultData, 0);
                }
                else
                {
                    Buffer.BlockCopy(data, 0, resultData, 0, sizeof(double) * resultData.Length);
                }
            }

            return new NdArray(resultData, resultShape);
        }

        public int Length
        {
            get { return this.Data.Length; }
        }

        public double Get(params int[] indices)
        {
            return this.Data[this.GetIndex(indices)];
        }

        public void Set(int[] indices, double val)
        {
            this.Data[this.GetIndex(indices)] = val;
        }

        public void Fill(double val)
        {
            for (int i = 0; i < this.Data.Length; i++)
            {
                this.Data[i] = val;
            }
        }

        //N次元のIndexから１次元のIndexを取得する
        public int GetIndex(params int[] indices)
        {
#if DEBUG
            if (this.Shape.Length != indices.Length)
            {
                throw new Exception("次元数がマッチしていません");
            }
#endif

            int index = 0;

            for (int i = 0; i < indices.Length; i++)
            {
#if DEBUG
                if (this.Shape[i] <= indices[i])
                {
                    throw new Exception(i + "次元の添字が範囲を超えています");
                }
#endif

                int rankOffset = 1;

                for (int j = i + 1; j < this.Shape.Length; j++)
                {
                    rankOffset *= this.Shape[j];
                }

                index += indices[i] * rankOffset;
            }

            return index;
        }

        //Numpyっぽく値を文字列に変換して出力する
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            int intMaxLength = 0;   //整数部の最大値
            int realMaxLength = 0;   //小数点以下の最大値
            bool isExponential = false; //指数表現にするか

            for (int i = 0; i < this.Data.Length; i++)
            {
                string[] divStr = this.Data[i].ToString().Split('.');
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
            List<int> CommonDivisorList = new List<int>();

            //一個目は手動取得
            CommonDivisorList.Add(this.Shape[this.Shape.Length - 1]);
            for (int i = 1; i < this.Shape.Length; i++)
            {
                CommonDivisorList.Add(CommonDivisorList[CommonDivisorList.Count - 1] * this.Shape[this.Shape.Length - i - 1]);
            }

            //先頭の括弧
            for (int i = 0; i < this.Shape.Length; i++)
            {
                sb.Append("[");
            }

            int closer = 0;
            for (int i = 0; i < this.Length; i++)
            {
                string[] divStr;
                double doubleVal;
                if (isExponential && double.TryParse(this.Data[i].ToString(), NumberStyles.AllowExponent | NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out doubleVal))
                {
                    divStr = doubleVal.ToString("0.00000000e+00").Split('.');
                }
                else
                {
                    divStr = this.Data[i].ToString().Split('.');
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
                if (i != this.Length - 1)
                {
                    for (int j = 0; j < CommonDivisorList.Count; j++)
                    {
                        if ((i + 1) % CommonDivisorList[j] == 0)
                        {
                            sb.Append("]");
                            closer++;
                        }
                    }

                    sb.Append(" ");

                    if ((i + 1) % CommonDivisorList[0] == 0)
                    {
                        //閉じ括弧分だけ改行を追加
                        for (int j = 0; j < closer; j++)
                        {
                            sb.Append("\n");
                        }
                        closer = 0;

                        //括弧前のインデント
                        for (int j = 0; j < CommonDivisorList.Count; j++)
                        {
                            if ((i + 1) % CommonDivisorList[j] != 0)
                            {
                                sb.Append(" ");
                            }
                        }
                    }

                    for (int j = 0; j < CommonDivisorList.Count; j++)
                    {
                        if ((i + 1) % CommonDivisorList[j] == 0)
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

            return sb.ToString();
        }

    }
}
