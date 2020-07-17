using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

#if DOUBLE
#elif NETSTANDARD2_1
using Math = System.MathF;
#else
using Math = KelpNet.MathF;
#endif

namespace KelpNet
{
    [Serializable]
    [DebuggerDisplay("{Name + ToString(\"Size\")}", Type = "{\"NdArray\" + ToString(\"Size\")}")]
    public class NdArray<T> where T : unmanaged, IComparable<T>
    {
        public string Name = "NdArray";

        public T[] Data;

        [NonSerialized]
        public T[] Grad;

        //各関数内でまとめて実行されるバッチの数を示し、Loss関数内の割引で使用される
        public int BatchCount = 1;

        //このNdArrayの各次元のサイズ
        public int[] Shape;

        //Shapeから算出されるLengthで、DataのLengthとは異なる
        public int Length;

        //関数によって使用された回数をカウントしBackward動作のタイミングを図る
        [NonSerialized]
        public int UseCount;

        //自身が関数から生成された場合、その関数をここに保存する
        [NonSerialized]
        public IFunction<T> ParentFunc;

        //Updateを行わずに実行されたBackwardの回数をカウントし、Optimizer実行時に使用する
        [NonSerialized]
        public int TrainCount;

        public NdArray(Array array, bool asBatch = false, IFunction<T> parentFunc = null)
        {
            this.Data = array.FlattenEx<T>();

            //1次元目をバッチとして扱うか？
            if (asBatch)
            {
                this.BatchCount = array.Rank > 1 ? array.GetLength(0) : array.Length;
                int[] resultShape = array.Rank > 1 ? new int[array.Rank - 1] : new[] { 1 };

                if (array.Rank > 1)
                {
                    for (int i = 0; i < resultShape.Length; i++)
                    {
                        resultShape[i] = array.GetLength(i + 1);
                    }
                }

                this.Shape = resultShape;
                this.Length = NdArray.ShapeToLength(this.Shape);
            }
            else
            {
                int[] resultShape = new int[array.Rank];

                for (int i = 0; i < array.Rank; i++)
                {
                    resultShape[i] = array.GetLength(i);
                }

                this.Shape = resultShape;
                this.Length = Data.Length;
            }

            this.ParentFunc = parentFunc;
        }

        public NdArray(params int[] shape)
        {
            this.Data = new T[NdArray.ShapeToLength(shape)];
            this.Shape = shape.ToArray();
            this.Length = Data.Length;
        }

        public NdArray(T[] data, int[] shape, int batchCount = 1, IFunction<T> parentFunc = null)
        {
            this.Shape = shape.ToArray();
            this.Length = data.Length / batchCount;
            this.BatchCount = batchCount;
            this.Data = data.ToArray();
            this.ParentFunc = parentFunc;
        }

        public NdArray(int[] shape, int batchCount, IFunction<T> parentFunc = null)
        {
            this.Shape = shape.ToArray();
            this.Length = NdArray.ShapeToLength(this.Shape);
            this.BatchCount = batchCount;
            this.Data = new T[this.Length * batchCount];
            this.ParentFunc = parentFunc;
        }

        //インデクサはあまり早くないので頻繁にアクセスする場合は使用をオススメしません。デバッグ用途向けと割り切ってください。
        public T this[int batchcount, params int[] indices]
        {
            get
            {
                return this.Data[this.GetLocalIndex(batchcount, indices)];
            }
            set
            {
                this.Data[this.GetLocalIndex(batchcount, indices)] = value;
            }
        }


        public void Reshape(params int[] shape)
        {
            int val = 0;
            int dimension = Length;

            //-1指定を算出
            if (shape.Contains(-1))
            {
                int minusIndex = -1;

                for (int i = 0; i < shape.Length; i++)
                {
                    if (shape[i] != -1)
                    {
                        val += Length % shape[i];

                        if (val == 0)
                        {
                            dimension /= shape[i];
                        }
                        else
                        {
                            throw new Exception("要素の指定が間違っています");
                        }
                    }
                    else
                    {
                        if (minusIndex != -1)
                        {
                            throw new Exception("-1が二つ以上指定されています");
                        }

                        minusIndex = i;
                    }
                }

                shape[minusIndex] = dimension;
            }
#if DEBUG
            else if (Length != NdArray.ShapeToLength(shape)) throw new Exception("指定されたShapeのサイズが現在のData.Lengthと等しくありません");
#endif

            Shape = shape.ToArray();
        }

        //バッチでまとまっているアレイをバラバラにして排出する
        public NdArray<T>[] DivideArrays()
        {
            NdArray<T>[] result = new NdArray<T>[BatchCount];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = GetSingleArray(i);
            }

            return result;
        }

        //バッチ番号に対応するアレイを排出する
        public NdArray<T> GetSingleArray(int i)
        {
            T[] data = new T[this.Length];
            Array.Copy(this.Data, i * this.Length, data, 0, this.Length);

            return new NdArray<T>(data, this.Shape);
        }

        public void CountUp()
        {
            TrainCount++;
        }

        //傾きの初期化
        public void InitGrad()
        {
            this.Grad = new T[this.Data.Length];
        }

        //傾きの初期化
        public void ClearGrad()
        {
            this.Grad = null;
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

        public string ToString(T[] datas)
        {
#if DEBUG
            if (Shape.Length == 0 && datas.Length != 1) throw new Exception();
#endif
            //単品データ
            if (Shape.Length == 0 && datas.Length == 1)
            {
                return datas[0].ToString();
            }

            StringBuilder sb = new StringBuilder();

            int intMaxLength = 0; //整数部の最大値
            int realMaxLength = 0; //小数点以下の最大値
            bool isExponential = false; //指数表現にするか

            foreach (T data in datas)
            {
                string[] divStr = data.ToString().Split('.');
                intMaxLength = (int)Math.Max(intMaxLength, divStr[0].Length);

                if (divStr.Length > 1)
                {
                    isExponential |= divStr[1].Contains("E");
                }

                if (realMaxLength != 8 && divStr.Length == 2)
                {
                    realMaxLength = (int)Math.Max(realMaxLength, divStr[1].Length);

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

                            if (BatchCount > 1) sb.Append(" ");

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

        public static NdArray<T> operator +(NdArray<T> a, NdArray<T> b)
        {
            return new Add<T>().Forward(a, b)[0];
        }

        public static NdArray<T> operator +(NdArray<T> a, T b)
        {
            return new AddConst<T>().Forward(a, b)[0];
        }

        public static NdArray<T> operator +(T a, NdArray<T> b)
        {
            return new ConstAdd<T>().Forward(a, b)[0];
        }


        public static NdArray<T> operator *(NdArray<T> a, NdArray<T> b)
        {
            return new Mul<T>().Forward(a, b)[0];
        }

        public static NdArray<T> operator *(NdArray<T> a, T b)
        {
            return new MulConst<T>().Forward(a, b)[0];
        }

        public static NdArray<T> operator *(T a, NdArray<T> b)
        {
            return new ConstMul<T>().Forward(a, b)[0];
        }


        public static NdArray<T> operator -(NdArray<T> a, NdArray<T> b)
        {
            return new Sub<T>().Forward(a, b)[0];
        }

        public static NdArray<T> operator -(NdArray<T> a, T b)
        {
            return new SubConst<T>().Forward(a, b)[0];
        }

        public static NdArray<T> operator -(T a, NdArray<T> b)
        {
            return new ConstSub<T>().Forward(a, b)[0];
        }


        public static NdArray<T> operator /(NdArray<T> a, NdArray<T> b)
        {
            return new Div<T>().Forward(a, b)[0];
        }

        public static NdArray<T> operator /(NdArray<T> a, T b)
        {
            return new DivConst<T>().Forward(a, b)[0];
        }

        public static NdArray<T> operator /(T a, NdArray<T> b)
        {
            return new ConstDiv<T>().Forward(a, b)[0];
        }

        public static implicit operator NdArray<T>(T[] a)
        {
            return new NdArray<T>(a);
        }

        public static implicit operator NdArray<T>(T a)
        {
            return new NdArray<T>(new[] { a });
        }

        //public static implicit operator NdArray<T>(long a)
        //{
        //    return new NdArray<T>(new[] { (T)a });
        //}

        //コピーを作成するメソッド
        public NdArray<T> Clone()
        {
            return new NdArray<T>(Data, Shape, BatchCount, ParentFunc)
            {
                Grad = Grad?.ToArray(),
                Name = Name,
                Length = Length,
                UseCount = UseCount,
                TrainCount = TrainCount
            };
        }

        public void Fill(T val)
        {
            for (int i = 0; i < Data.Length; i++)
            {
                Data[i] = val;
            }
        }

        public void FillGrad(T val)
        {
            for (int i = 0; i < Grad.Length; i++)
            {
                Grad[i] = val;
            }
        }

        public int[] GetDimensionsIndex(int index)
        {
            //バッチ分を補正
            int batchCount = index / this.Length;
            index -= this.Length * batchCount;

            int[] dimensionsIndex = new int[this.Shape.Length];

            for (int i = this.Shape.Length - 1; i >= 0; i--)
            {
                dimensionsIndex[i] = index % this.Shape[i];
                index /= this.Shape[i];
            }

            return dimensionsIndex;
        }

        public int GetLocalIndex(int batchIndex, params int[] indices)
        {
            int result = 0;
            int rankOffset = 1;

            for (int i = indices.Length - 1; i >= 0; i--)
            {
                result += indices[i] * rankOffset;
                rankOffset *= Shape[i];
            }

            return result + batchIndex * Length;
        }

    }

    public static class NdArray
    {
        public static int ShapeToLength(params int[] shapes)
        {
            int result = 1;

            foreach (int shape in shapes)
            {
                result *= shape;
            }

            return result;
        }

        //アレイ配列をバッチとして登録する
        public static NdArray<T> FromArrays<T>(T[][] arrays, IFunction<T> parentFunc = null) where T : unmanaged, IComparable<T>
        {
            int[] resultShape = new int[arrays[0].Rank];

            for (int i = 0; i < arrays[0].Rank; i++)
            {
                resultShape[i] = arrays[0].GetLength(i);
            }

            int length = arrays[0].Length;
            T[] result = new T[length * arrays.Length];

            for (int i = 0; i < arrays.Length; i++)
            {
                Array.Copy(arrays[i], 0, result, length * i, length);
            }

            return new NdArray<T>(result, resultShape, arrays.Length, parentFunc);
        }

        public static NdArray<T> FromArrays<T>(List<T[]> arrays, IFunction<T> parentFunc = null) where T : unmanaged, IComparable<T>
        {
            int[] resultShape = new int[arrays[0].Rank];

            for (int i = 0; i < arrays[0].Rank; i++)
            {
                resultShape[i] = arrays[0].GetLength(i);
            }

            int length = arrays[0].Length;
            T[] result = new T[length * arrays.Count];

            for (int i = 0; i < arrays.Count; i++)
            {
                Array.Copy(arrays[i], 0, result, length * i, length);
            }

            return new NdArray<T>(result, resultShape, arrays.Count, parentFunc);
        }

        public static NdArray<T> Convert<T>(T[] data, int[] shape, int batchCount, IFunction<T> parentFunc = null) where T : unmanaged, IComparable<T>
        {
            return new NdArray<T>(shape, batchCount, parentFunc) { Data = data };
        }

        public static NdArray<T> ZerosLike<T>(NdArray<T> baseArray) where T : unmanaged, IComparable<T>
        {
            return new NdArray<T>(baseArray.Shape, baseArray.BatchCount);
        }

        public static void Backward<T>(NdArray<T> y) where T : unmanaged, IComparable<T>
        {
            if (y.ParentFunc != null)
            {
                List<NdArray<T>[]> prevInputs = y.ParentFunc.PrevInputs;
                NdArray<T>[] xs = prevInputs[prevInputs.Count - 1];

                y.ParentFunc.Backward(y);

                for (int i = 0; i < xs.Length; i++)
                {
                    if (xs[i].UseCount == 0)
                    {
                        Backward(xs[i]);
                    }
                }
            }
        }

        public static NdArray<T>[] Split<T>(NdArray<T> array, int indices, int axis) where T : unmanaged, IComparable<T>
        {
            return Split(array, new[] { indices }, axis);
        }

        public static NdArray<T>[] Split<T>(NdArray<T> array, int[] indices, int axis) where T : unmanaged, IComparable<T>
        {
            int[] shapeOffets = new int[indices.Length + 1];        //入力されたindicesの先頭0を追加した配列
            int[] resultAxisShapes = new int[indices.Length + 1];   //分割後の指定軸のShape

            for (int i = 0; i < indices.Length; i++)
            {
                shapeOffets[i + 1] = indices[i];
                resultAxisShapes[i] = indices[i] - shapeOffets[i];
            }
            resultAxisShapes[indices.Length] = array.Shape[axis] - indices[indices.Length - 1];

            NdArray<T>[] resultArrays = new NdArray<T>[indices.Length + 1];

            for (int i = 0; i < resultArrays.Length; i++)
            {
                int[] resultShape = array.Shape.ToArray();
                resultShape[axis] = resultAxisShapes[i];
                resultArrays[i] = new NdArray<T>(resultShape, array.BatchCount);
                if (array.Grad != null) resultArrays[i].InitGrad();
            }

            for (int batchCount = 0; batchCount < array.BatchCount; batchCount++)
            {
                for (int i = 0; i < resultArrays.Length; i++)
                {
                    for (int j = 0; j < resultArrays[i].Length; j++)
                    {
                        int[] resultIndex = resultArrays[i].GetDimensionsIndex(j);
                        resultIndex[axis] += shapeOffets[i];
                        int localIndex = array.GetLocalIndex(batchCount, resultIndex);

                        resultArrays[i].Data[batchCount * resultArrays[i].Length + j] = array.Data[localIndex];
                        if (array.Grad != null) resultArrays[i].Grad[batchCount * resultArrays[i].Length + j] = array.Grad[localIndex];
                    }
                }
            }

            return resultArrays;
        }

        public static NdArray<T> Rollaxis<T>(NdArray<T> input, int axis, int start = 0) where T : unmanaged, IComparable<T>
        {
            int n = input.Shape.Length;
            if (axis < 0) axis += n;
            if (start < 0) start += n;

#if DEBUG
            string msg = "rollaxis: {0} ({1}) must be >=0 and < {2}";

            if (!(0 <= axis && axis < n)) throw new Exception(string.Format(msg, "axis", axis, n));
            if (!(0 <= start && start < n + 1)) throw new Exception(string.Format(msg, "start", start, n + 1));
#endif
            if (axis == start) return input;

            List<int> axes = new List<int>(Enumerable.Range(0, n).ToArray());
            axes.RemoveAt(axis);
            axes.Insert(start, axis);

            return Transpose(input, axes.ToArray());
        }

        public static NdArray<T> Transpose<T>(NdArray<T> input, params int[] dimensions) where T : unmanaged, IComparable<T>
        {
#if DEBUG
            if (input.Shape.Length != dimensions.Length)
            {
                throw new Exception("次元数がマッチしていません");
            }

            for (int i = 0; i < dimensions.Length; i++)
            {
                //自身の中にダブりがないか
                for (int j = i + 1; j < dimensions.Length; j++)
                {
                    if (dimensions[i] == dimensions[j]) throw new Exception("指定された要素がダブっています");
                }
                //範囲チェック
                if (dimensions[i] >= input.Shape.Length || dimensions[i] < 0) throw new Exception("指定された要素が範囲を超えています");
            }
#endif
            int[] transposedDimensions = new int[input.Shape.Length];

            for (int j = 0; j < input.Shape.Length; j++)
            {
                transposedDimensions[j] = input.Shape[dimensions[j]];
            }

            NdArray<T> resultMatrix = new NdArray<T>(transposedDimensions);

            for (int b = 0; b < input.BatchCount; b++)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    int[] indecies = input.GetDimensionsIndex(i);
                    int[] transposedIndex = new int[input.Shape.Length];

                    for (int j = 0; j < input.Shape.Length; j++)
                    {
                        transposedIndex[j] = indecies[dimensions[j]];
                    }

                    resultMatrix.Data[resultMatrix.GetLocalIndex(b, transposedIndex)] = input.Data[input.GetLocalIndex(b, indecies)];
                }
            }

            return resultMatrix;
        }

        public static NdArray<T> Concatenate<T>(NdArray<T> a, NdArray<T> b, int axis) where T : unmanaged, IComparable<T>
        {
            int[] shapeList = a.Shape.ToArray();
            shapeList[axis] += b.Shape[axis];

#if DEBUG
            for (int i = 0; i < a.Shape.Length; i++)
            {
                if (i != axis && a.Shape[i] != b.Shape[i])
                {
                    throw new Exception("配列の大きさがマッチしていません");
                }
            }

            if (a.BatchCount != b.BatchCount)
            {
                throw new Exception("バッチの大きさがマッチしていません");
            }

            if((a.Grad != null) != (b.Grad != null))
            {
                throw new Exception("Grad値の有無が揃っていません");
            }
#endif

            NdArray<T> result = new NdArray<T>(shapeList, a.BatchCount);
            if (a.Grad != null || b.Grad != null)
            {
                result.InitGrad();
            }

            for (int batchCount = 0; batchCount < a.BatchCount; batchCount++)
            {
                int aInputBatchoffset = batchCount * a.Length;
                int bInputBatchoffset = batchCount * b.Length;

                for (int i = 0; i < a.Length; i++)
                {
                    int resultindex = result.GetLocalIndex(batchCount, a.GetDimensionsIndex(i));

                    result.Data[resultindex] = a.Data[i + aInputBatchoffset];
                    if (a.Grad != null) result.Grad[resultindex] = a.Grad[i + aInputBatchoffset];
                }

                for (int i = 0; i < b.Length; i++)
                {
                    int[] tmpIndex = b.GetDimensionsIndex(i);
                    tmpIndex[axis] += a.Shape[axis];

                    int resultIndex = result.GetLocalIndex(batchCount, tmpIndex);

                    result.Data[resultIndex] = b.Data[i + bInputBatchoffset];
                    if (b.Grad != null) result.Grad[resultIndex] = b.Grad[i + bInputBatchoffset];
                }
            }

            return result;
        }

    }
}
