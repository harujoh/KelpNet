using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using KelpNet.Common.Functions;
using KelpNet.Functions.Mathmetrics.BasicMath;

namespace KelpNet.Common
{
    [Serializable]
    [DebuggerDisplay("{Name + ToString(\"Size\")}", Type = "{\"NdArray\" + ToString(\"Size\")}")]
    public class NdArray
    {
        public string Name = "NdArray";

        public Real[] Data;

        [NonSerialized]
        public Real[] Grad;

        //このNdArrayの各次元のサイズ
        public int[] Shape { private set; get; }

        //Shapeから算出されるLengthで、DataのLengthとは異なる
        public int Length { private set; get; }

        //関数によって使用された回数をカウントしBackward動作のタイミングを図る
        [NonSerialized]
        public int UseCount = 0;

        //自身が関数から生成された場合、その関数をここに保存する
        [NonSerialized]
        public Function ParentFunc;

        //各関数内でまとめて実行されるバッチの数を示し、Loss関数内の割引で使用される
        public int BatchCount;

        //Updateを行わずに実行されたBackwardの回数をカウントし、Optimizer実行時に使用する
        [NonSerialized]
        public int TrainCount;

        public NdArray(Array data, Function parentFunc = null)
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
            this.ParentFunc = parentFunc;
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

        public NdArray(Real[] data, int[] shape, int batchCount = 1, Function parentFunc = null)
        {
            this.Shape = shape.ToArray();
            this.Length = ShapeToArrayLength(this.Shape);
            this.BatchCount = batchCount;
            this.Data = data.ToArray();
            this.Grad = new Real[this.Length * batchCount];
            this.TrainCount = 0;
            this.ParentFunc = parentFunc;
        }

        public NdArray(int[] shape, int batchCount, Function parentFunc = null)
        {
            this.Shape = shape.ToArray();
            this.Length = ShapeToArrayLength(this.Shape);
            this.BatchCount = batchCount;
            this.Data = new Real[this.Length * batchCount];
            this.Grad = new Real[this.Length * batchCount];
            this.TrainCount = 0;
            this.ParentFunc = parentFunc;
        }

        public static NdArray FromArrays(Array[] arrays, Function parentFunc = null)
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

            return new NdArray(result, resultShape, arrays.Length, parentFunc);
        }

        public static NdArray Convert(Real[] data, int[] shape, int batchCount, Function parentFunc = null)
        {
            return new NdArray(shape, batchCount, parentFunc) { Data = data };
        }

        public static NdArray ZerosLike(NdArray baseArray)
        {
            return new NdArray(baseArray.Shape.ToArray(), baseArray.BatchCount);
        }

        public void Reshape(params int[] shape)
        {
#if DEBUG
            if (Length != ShapeToArrayLength(shape)) throw new Exception("指定されたShapeのサイズが現在のData.Lengthと等しくありません");
#endif
            Shape = shape.ToArray();
        }

        //バッチでまとまっているアレイをバラバラにして排出する
        public NdArray[] DivideArrays()
        {
            NdArray[] result = new NdArray[BatchCount];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = GetSingleArray(i);
            }

            return result;
        }

        //バッチ番号に対応するアレイを排出する
        public NdArray GetSingleArray(int i)
        {
            Real[] data = new Real[this.Length];
            Array.Copy(this.Data, i * this.Length, data, 0, this.Length);

            return new NdArray(data, this.Shape);
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

        public void Backward()
        {
            if (ParentFunc != null)
            {
                for (int i = 0; i < Grad.Length; i++)
                {
                    Grad[i] = 1;
                }

                Backward(this);
            }
        }

        internal static void Backward(NdArray y)
        {
            if (y.ParentFunc != null)
            {
                List<NdArray[]> prevInputs = y.ParentFunc.PrevInputs;
                NdArray[] xs = prevInputs[prevInputs.Count - 1];

                y.ParentFunc.Backward(y);

                for (int i = 0; i < xs.Length; i++)
                {
                    if (xs[i].UseCount == 0)
                    {
                        NdArray.Backward(xs[i]);
                    }
                }
            }
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
            switch (format)
            {
                case "Data":
                    return ToString(this.Data);

                case "Grad":
                    return ToString(this.Grad);

                case "Shape":
                    return "[" + string.Join(",", Shape) + "]";

                case "Size":
                    return "[" + string.Join(",", Shape) + "]" +
                           (BatchCount > 1 ? "x" + BatchCount + "batch" : string.Empty);

                case "Name":
                    return Name;

                default:
                    return Name;
            }
        }

        public string ToString(Real[] datas)
        {
            StringBuilder sb = new StringBuilder();

            int intMaxLength = 0; //整数部の最大値
            int realMaxLength = 0; //小数点以下の最大値
            bool isExponential = false; //指数表現にするか

            foreach (Real data in datas)
            {
                string[] divStr = ((double)data).ToString().Split('.');
                intMaxLength = Math.Max(intMaxLength, divStr[0].Length);

                if (divStr.Length > 1)
                {
                    isExponential |= divStr[1].Contains("E");
                }

                if (realMaxLength != 8 && divStr.Length == 2)
                {
                    realMaxLength = Math.Max(realMaxLength, divStr[1].Length);

                    if (realMaxLength > 8)
                    {
                        realMaxLength = 8;
                    }
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
                        divStr = string.Format("{0:0.00000000e+00}", (double)datas[indexOffset + i]).Split('.');
                    }
                    else
                    {
                        divStr = ((double)datas[indexOffset + i]).ToString().Split('.');
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


        public static NdArray operator +(NdArray a, NdArray b)
        {
            return new Add().Forward(a, b);
        }

        public static NdArray operator +(NdArray a, Real b)
        {
            return new AddConst().Forward(a, b);
        }

        public static NdArray operator +(Real a, NdArray b)
        {
            return new AddConst().Forward(b, a);
        }


        public static NdArray operator *(NdArray a, NdArray b)
        {
            return new Mul().Forward(a, b);
        }

        public static NdArray operator *(NdArray a, Real b)
        {
            return new MulConst().Forward(a, b);
        }

        public static NdArray operator *(Real a, NdArray b)
        {
            return new MulConst().Forward(b, a);
        }


        public static NdArray operator -(NdArray a, NdArray b)
        {
            return new Sub().Forward(a, b);
        }

        public static NdArray operator -(NdArray a, Real b)
        {
            return new SubConst().Forward(a, b);
        }

        public static NdArray operator -(Real a, NdArray b)
        {
            return new ConstSub().Forward(a, b);
        }


        public static NdArray operator /(NdArray a, NdArray b)
        {
            return new Div().Forward(a, b);
        }

        public static NdArray operator /(NdArray a, Real b)
        {
            return new DivConst().Forward(a, b);
        }

        public static NdArray operator /(Real a, NdArray b)
        {
            return new ConstDiv().Forward(a, b);
        }


        public static implicit operator NdArray(Real a)
        {
            return new NdArray(new[] { a });
        }

        public static implicit operator NdArray(long a)
        {
            return new NdArray(new[] { (Real)a });
        }

        //コピーを作成するメソッド
        public NdArray Clone()
        {
            return new NdArray
            {
                ParentFunc = ParentFunc,
                Data = Data.ToArray(),
                Grad = Grad.ToArray(),
                Shape = Shape.ToArray(),
                Name = Name,
                Length = Length,
                BatchCount = BatchCount,
                UseCount = UseCount,
                TrainCount = TrainCount
            };
        }

        public static NdArray Concatenate(NdArray a, NdArray b, int axis)
        {
            int[] shapeList = a.Shape.ToArray();
            shapeList[axis] += b.Shape[axis];

#if DEBUG
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("配列の大きさがマッチしていません");
                }
            }
#endif

            NdArray result = new NdArray(shapeList.ToArray(), a.BatchCount);

            for (int batchCount = 0; batchCount < a.BatchCount; batchCount++)
            {
                int aInputBatchoffset = batchCount * a.Length;
                int bInputBatchoffset = batchCount * b.Length;

                for (int i = 0; i < a.Length; i++)
                {
                    int resultindex = result.GetLocalIndex(a.GetDimensionsIndex(i), batchCount);

                    result.Data[resultindex] = a.Data[i + aInputBatchoffset];
                    result.Grad[resultindex] = a.Grad[i + aInputBatchoffset];
                }

                for (int i = 0; i < b.Length; i++)
                {
                    int[] tmpIndex = b.GetDimensionsIndex(i);
                    tmpIndex[axis] += a.Shape[axis];

                    int resultIndex = result.GetLocalIndex(tmpIndex, batchCount);

                    result.Data[resultIndex] = b.Data[i + bInputBatchoffset];
                    result.Grad[resultIndex] = b.Grad[i + bInputBatchoffset];
                }
            }

            return result;
        }

        private int[] GetDimensionsIndex(int index)
        {
            //バッチ分を補正
            int batchCount = index % Length;
            index -= Length * batchCount;

            int[] dimensionsIndex = new int[this.Shape.Length];

            for (int i = this.Shape.Length - 1; i >= 0; i--)
            {
                dimensionsIndex[i] = index % this.Shape[i];
                index = index / this.Shape[i];
            }

            return dimensionsIndex;
        }

        private int GetLocalIndex(int[] indices, int batchIndex)
        {
            int index = batchIndex * Length;

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