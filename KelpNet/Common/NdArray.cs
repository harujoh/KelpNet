using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KelpNet.Common.Tools;

namespace KelpNet.Common
{
    //NumpyのNdArrayを模したクラス
    //N次元のArrayクラスを入力に取り、内部的には1次元配列として保持する事で動作を模倣している
    [Serializable]
    public struct NdArray
    {
        public Real[] Data;
        public int[] Shape;

        public NdArray(Real[] data, int[] shape)
        {
            //コンストラクタはコピーを作成する
            this.Data = data.ToArray();
            this.Shape = shape.ToArray();
        }

        public NdArray(NdArray ndArray)
        {
            //コンストラクタはコピーを作成する
            this.Data = ndArray.Data.ToArray();
            this.Shape = ndArray.Shape.ToArray();
        }

        public NdArray (params int[] shape)
        {
            Data = new Real[ShapeToArrayLength(shape)];
            Shape = shape.ToArray();
        }

        public int Rank
        {
            get { return this.Shape.Length; }
        }

        //データ部をコピーせずにインスタンスする
        public static NdArray Convert(Real[] data, int[] shape)
        {
            return new NdArray { Data = data, Shape = shape.ToArray() };
        }

        //データ部をコピーせずにインスタンスする
        public static NdArray Convert(Real[] data)
        {
            return new NdArray { Data = data, Shape = new[] { data.Length } };
        }

        public static NdArray ZerosLike(NdArray baseArray)
        {
            return new NdArray { Data = new Real[baseArray.Data.Length], Shape = baseArray.Shape.ToArray() };
        }

        public static int ShapeToArrayLength(params int[] shapes)
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
            Real[] resultData = Real.GetArray(data);

            int[] resultShape = new int[data.Rank];
            for (int i = 0; i < data.Rank; i++)
            {
                resultShape[i] = data.GetLength(i);
            }

            return new NdArray { Data = resultData, Shape = resultShape };
        }

        public void Clear()
        {
            this.Data = new Real[this.Data.Length];
        }

        public void Fill(Real val)
        {
            for (int i = 0; i < this.Data.Length; i++)
            {
                this.Data[i] = val;
            }
        }

        //Numpyっぽく値を文字列に変換して出力する
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            int intMaxLength = 0;   //整数部の最大値
            int realMaxLength = 0;   //小数点以下の最大値
            bool isExponential = false; //指数表現にするか

            foreach (Real data in this.Data)
            {
                string[] divStr = ((double)data).ToString().Split('.');
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

            //先頭の括弧
            for (int i = 0; i < this.Shape.Length; i++)
            {
                sb.Append("[");
            }

            int closer = 0;
            for (int i = 0; i < this.Data.Length; i++)
            {
                string[] divStr;
                if (isExponential)
                {
                    divStr = string.Format("{0:0.00000000e+00}", (double)this.Data[i]).Split('.');
                }
                else
                {
                    divStr = ((double)this.Data[i]).ToString().Split('.');
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
                if (i != this.Data.Length - 1)
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

            return sb.ToString();
        }

        //コピーを作成するメソッド
        public NdArray Clone()
        {
            return DeepCopyHelper.DeepCopy(this);
        }

        public NdArray Transpose(params int[] targetDimensions)
        {
#if DEBUG
            if (targetDimensions.Length != 0 && this.Shape.Length != targetDimensions.Length)
            {
                throw new Exception("次元数がマッチしていません");
            }

            for (int i = 0; i < targetDimensions.Length; i++)
            {
                //自身の中にダブりがないか
                for (int j = i + 1; j < targetDimensions.Length; j++)
                {
                    if (targetDimensions[i] == targetDimensions[j]) throw new Exception("指定された要素がダブっています");
                }
                //範囲チェック
                if (targetDimensions[i] >= this.Shape.Length || targetDimensions[i] < 0) throw new Exception("指定された要素が範囲を超えています");
            }
#endif
            //引数なしの場合単純に逆転させる
            if (targetDimensions.Length == 0)
            {
                targetDimensions = Enumerable.Range(0, this.Shape.Length).ToArray();
                Array.Reverse(targetDimensions);
            }

            //指定された次元のインデックスからShapeを作成する
            int[] transposedShape = new int[this.Shape.Length];
            for (int j = 0; j < this.Shape.Length; j++)
            {
                transposedShape[j] = this.Shape[targetDimensions[j]];
            }

            Real[] resultArray = new Real[this.Data.Length];

            for (int i = 0; i < resultArray.Length; i++)
            {
                //データインデックスから変換元の次元インデックスを取得
                int[] dimensionIndex = this.GetDimensionsIndex(i);

                //次元インデックスを転送先インデックスに変換
                int resultindex = 0;
                for (int j = this.Shape.Length - 1; j >= 0; j--)
                {
                    int rankOffset = 1;
                    for (int k = j + 1; k < this.Shape.Length; k++)
                    {
                        rankOffset *= this.Shape[k];
                    }

                    resultindex += dimensionIndex[targetDimensions[j]] * rankOffset;
                }

                resultArray[resultindex] = this.Data[i];
            }

            return NdArray.Convert(resultArray, transposedShape);
        }

        public NdArray Rollaxis(int axis, int start = 0)
        {
            int n = this.Shape.Length;
            if (axis < 0) axis += n;
            if (start < 0) start += n;
            if (axis == start) return this;

#if DEBUG
            string msg = "rollaxis: {0} ({1}) must be >=0 and < {2}";
            if (!(0 <= axis && axis < n)) throw new Exception(string.Format(msg, "axis", axis, n));
            if (!(0 <= start && start < n + 1)) throw new Exception(string.Format(msg, "start", start, n + 1));
#endif

            List<int> axes = new List<int>(Enumerable.Range(0, n).ToArray());
            axes.RemoveAt(axis);
            axes.Insert(start, axis);

            return this.Transpose(axes.ToArray());
        }

        public static NdArray Concatenate(NdArray a, NdArray b, int axis)
        {
            List<int> shapeList = new List<int>();
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (i == axis)
                {
                    shapeList.Add(a.Shape[i] + b.Shape[i]);
                }
                else if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("配列の大きさがマッチしていません");
                }
                else
                {
                    shapeList.Add(a.Shape[i]);
                }
            }

            NdArray result = new NdArray(shapeList.ToArray());

            for (int i = 0; i < a.Data.Length; i++)
            {
                result.Data[result.GetLocalIndex(a.GetDimensionsIndex(i))] = a.Data[i];
            }

            for (int i = 0; i < b.Data.Length; i++)
            {
                int[] tmpIndex = b.GetDimensionsIndex(i);
                tmpIndex[axis] += a.Shape[axis];

                result.Data[result.GetLocalIndex(tmpIndex)] = b.Data[i];
            }

            return result;
        }

        private int[] GetDimensionsIndex(int index)
        {
            int[] dimensionsIndex = new int[this.Shape.Length];

            for (int i = this.Shape.Length - 1; i >= 0; i--)
            {
                dimensionsIndex[i] = index % this.Shape[i];
                index = index / this.Shape[i];
            }

            return dimensionsIndex;
        }

        private int GetLocalIndex(int[] indices)
        {
            int index = 0;

            for (int i = 0; i < indices.Length; i++)
            {
                int rankOffset = 1;

                for (int j = i + 1; j < this.Shape.Length; j++)
                {
                    rankOffset *= this.Shape[j];
                }

                index += indices[i] * rankOffset;
            }

            return index;
        }
    }
}
