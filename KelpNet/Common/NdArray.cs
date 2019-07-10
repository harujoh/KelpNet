using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace KelpNet
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
        public int UseCount;

        //自身が関数から生成された場合、その関数をここに保存する
        [NonSerialized]
        public IFunction ParentFunc;

        //各関数内でまとめて実行されるバッチの数を示し、Loss関数内の割引で使用される
        public int BatchCount = 1;

        //Updateを行わずに実行されたBackwardの回数をカウントし、Optimizer実行時に使用する
        [NonSerialized]
        public int TrainCount;

        public NdArray(Array data, IFunction parentFunc = null)
        {
            Real[] resultData = Real.ToRealArray(data);

            int[] resultShape = new int[data.Rank];

            for (int i = 0; i < data.Rank; i++)
            {
                resultShape[i] = data.GetLength(i);
            }

            this.Data = resultData;
            this.Shape = resultShape;
            this.Length = Data.Length;
            this.ParentFunc = parentFunc;
        }

        public NdArray(params int[] shape)
        {
            this.Data = new Real[ShapeToArrayLength(shape)];
            this.Shape = shape.ToArray();
            this.Length = Data.Length;
        }

        public NdArray(Real[] data, int[] shape, int batchCount = 1, IFunction parentFunc = null)
        {
            this.Shape = shape.ToArray();
            this.Length = data.Length / batchCount;
            this.BatchCount = batchCount;
            this.Data = data.ToArray();
            this.ParentFunc = parentFunc;
        }

        public NdArray(int[] shape, int batchCount, IFunction parentFunc = null)
        {
            this.Shape = shape.ToArray();
            this.Length = ShapeToArrayLength(this.Shape);
            this.BatchCount = batchCount;
            this.Data = new Real[this.Length * batchCount];
            this.ParentFunc = parentFunc;
        }

        //アレイ配列をバッチとして登録する
        public static NdArray FromArrays(Array[] arrays, IFunction parentFunc = null)
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
                Array.Copy(Real.ToRealArray(arrays[i]), 0, result, length * i, length);
            }

            return new NdArray(result, resultShape, arrays.Length, parentFunc);
        }

        public static NdArray Convert(Real[] data, int[] shape, int batchCount, IFunction parentFunc = null)
        {
            return new NdArray(shape, batchCount, parentFunc) { Data = data };
        }

        public static NdArray ZerosLike(NdArray baseArray)
        {
            return new NdArray(baseArray.Shape, baseArray.BatchCount);
        }

        //インデクサはあまり早くないので頻繁にアクセスする場合は使用をオススメしません。デバッグ用途向けと割り切ってください。
        public Real this[int batchcount, params int[] indices]
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
            else if (Length != ShapeToArrayLength(shape)) throw new Exception("指定されたShapeのサイズが現在のData.Lengthと等しくありません");
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
            if (this.ParentFunc != null)
            {
                if (this.Grad == null)
                {
                    this.Grad = new Real[this.Length * this.BatchCount];

                    for (int i = 0; i < this.Grad.Length; i++)
                    {
                        Grad[i] = 1;
                    }
                }

                NdArray.Backward(this);
            }
        }

        public static void Backward(NdArray y)
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
        public void Reduce()
        {
            if (this.TrainCount > 0)
            {
                for (int i = 0; i < this.Grad.Length; i++)
                {
                    this.Grad[i] /= this.TrainCount;
                }
            }
        }

        //傾きの初期化
        public void InitGrad()
        {
            this.Grad = new Real[this.Data.Length];
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

        public string ToString(Real[] datas)
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
                        divStr = string.Format("{0:0.00000000e+00}", (Real)datas[indexOffset + i]).Split('.');
                    }
                    else
                    {
                        divStr = ((Real)datas[indexOffset + i]).ToString().Split('.');
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


        public static NdArray operator +(NdArray a, NdArray b)
        {
            return new Add().Forward(a, b)[0];
        }

        public static NdArray operator +(NdArray a, Real b)
        {
            return new AddConst().Forward(a, b)[0];
        }

        public static NdArray operator +(Real a, NdArray b)
        {
            return new AddConst().Forward(b, a)[0];
        }


        public static NdArray operator *(NdArray a, NdArray b)
        {
            return new Mul().Forward(a, b)[0];
        }

        public static NdArray operator *(NdArray a, Real b)
        {
            return new MulConst().Forward(a, b)[0];
        }

        public static NdArray operator *(Real a, NdArray b)
        {
            return new MulConst().Forward(b, a)[0];
        }


        public static NdArray operator -(NdArray a, NdArray b)
        {
            return new Sub().Forward(a, b)[0];
        }

        public static NdArray operator -(NdArray a, Real b)
        {
            return new SubConst().Forward(a, b)[0];
        }

        public static NdArray operator -(Real a, NdArray b)
        {
            return new ConstSub().Forward(a, b)[0];
        }


        public static NdArray operator /(NdArray a, NdArray b)
        {
            return new Div().Forward(a, b)[0];
        }

        public static NdArray operator /(NdArray a, Real b)
        {
            return new DivConst().Forward(a, b)[0];
        }

        public static NdArray operator /(Real a, NdArray b)
        {
            return new ConstDiv().Forward(a, b)[0];
        }

        public static implicit operator NdArray(Real[] a)
        {
            return new NdArray(a);
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
            return new NdArray(Data, Shape, BatchCount, ParentFunc)
            {
                Grad = Grad?.ToArray(),
                Name = Name,
                Length = Length,
                UseCount = UseCount,
                TrainCount = TrainCount
            };
        }

        public void Fill(Real val)
        {
            for (int i = 0; i < Data.Length; i++)
            {
                Data[i] = val;
            }
        }

        public void FillGrad(Real val)
        {
            for (int i = 0; i < Grad.Length; i++)
            {
                Grad[i] = val;
            }
        }

        private static NdArray ArrayFunc(NdArray input, Func<NdArray, int, NdArray> calcFunc, int[] axis = null, bool keepDims = false)
        {
#if DEBUG
            if (axis != null && axis.Length != axis.Distinct().ToArray().Length)
            {
                throw new Exception("指定された要素が重複しています");
            }

            if (axis != null && axis.Length != 0 && input.Shape.Length < axis.Max())
            {
                throw new Exception("指定された要素が範囲を超えています");
            }
#endif
            if (axis == null || axis.Length == 0)
            {
                axis = Enumerable.Range(0, input.Shape.Length).ToArray();
            }

            Array.Sort(axis);

            NdArray result = calcFunc(input, axis[0]);

            for (int i = 1; i < axis.Length; i++)
            {
                result = calcFunc(result, axis[i] - i);
            }

            if (keepDims)
            {
                int[] resultKeepDimShape = input.Shape.ToArray();

                for (int i = 0; i < axis.Length; i++)
                {
                    resultKeepDimShape[axis[i]] = 1;
                }

                result.Shape = resultKeepDimShape;
            }

            return result;
        }

        public static NdArray Sum(NdArray input, int[] axis = null, bool keepDims = false)
        {
            return ArrayFunc(input, LocalSum, axis, keepDims);
        }

        private static NdArray LocalSum(NdArray input, int axis)
        {
            int[] resultShape = new int[input.Shape.Length - 1];

            for (int i = 0, j = 0; i < input.Shape.Length; i++)
            {
                if (i != axis)
                {
                    resultShape[j++] = input.Shape[i];
                }
            }

            NdArray result = new NdArray(resultShape, input.BatchCount);

            for (int i = 0; i < input.Length; i++)
            {
                List<int> index = new List<int>(input.GetDimensionsIndex(i));
                index.RemoveAt(axis);
                int localIndex = result.GetLocalIndex(0, index.ToArray());

                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    result.Data[batchCount * result.Length + localIndex] += input.Data[batchCount * input.Length + i];
                    if (input.Grad != null) result.Grad[batchCount * result.Length + localIndex] += input.Grad[batchCount * input.Length + i];
                }
            }

            return result;
        }

        public static NdArray Max(NdArray input, int[] axis = null, bool keepDims = false)
        {
            return ArrayFunc(input, LocalMax, axis, keepDims);
        }

        private static NdArray LocalMax(NdArray input, int axis)
        {
            int[] resultShape = new int[input.Shape.Length - 1];

            for (int i = 0, j = 0; i < input.Shape.Length; i++)
            {
                if (i != axis)
                {
                    resultShape[j++] = input.Shape[i];
                }
            }

            NdArray result = new NdArray(resultShape, input.BatchCount);
            result.Fill(input.Data.Min());
            if (input.Grad != null) result.FillGrad(input.Grad.Min());

            for (int i = 0; i < input.Length; i++)
            {
                List<int> index = new List<int>(input.GetDimensionsIndex(i));
                index.RemoveAt(axis);
                int localIndex = result.GetLocalIndex(0, index.ToArray());

                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    if(result.Data[batchCount * result.Length + localIndex] < input.Data[batchCount * input.Length + i]) result.Data[batchCount * result.Length + localIndex] = input.Data[batchCount * input.Length + i];
                    if (input.Grad != null)
                    {
                        if(result.Grad[batchCount * result.Length + localIndex] < input.Grad[batchCount * input.Length + i])
                            result.Grad[batchCount * result.Length + localIndex] = input.Grad[batchCount * input.Length + i];
                    }
                }
            }

            return result;
        }


        public static NdArray Min(NdArray input, int[] axis = null, bool keepDims = false)
        {
            return ArrayFunc(input, LocalMin, axis, keepDims);
        }

        private static NdArray LocalMin(NdArray input, int axis)
        {
            int[] resultShape = new int[input.Shape.Length - 1];

            for (int i = 0, j = 0; i < input.Shape.Length; i++)
            {
                if (i != axis)
                {
                    resultShape[j++] = input.Shape[i];
                }
            }

            NdArray result = new NdArray(resultShape, input.BatchCount);
            result.Fill(input.Data.Min());
            if (input.Grad != null) result.FillGrad(input.Grad.Min());

            for (int i = 0; i < input.Length; i++)
            {
                List<int> index = new List<int>(input.GetDimensionsIndex(i));
                index.RemoveAt(axis);
                int localIndex = result.GetLocalIndex(0, index.ToArray());

                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    if (result.Data[batchCount * result.Length + localIndex] > input.Data[batchCount * input.Length + i]) result.Data[batchCount * result.Length + localIndex] = input.Data[batchCount * input.Length + i];
                    if (input.Grad != null)
                    {
                        if (result.Grad[batchCount * result.Length + localIndex] > input.Grad[batchCount * input.Length + i])
                            result.Grad[batchCount * result.Length + localIndex] = input.Grad[batchCount * input.Length + i];
                    }
                }
            }

            return result;
        }

        public static NdArray[] Split(NdArray array, int indices, int axis)
        {
            return Split(array, new[] { indices }, axis);
        }

        public static NdArray[] Split(NdArray array, int[] indices, int axis)
        {
            int[] shapeOffets = new int[indices.Length + 1];        //入力されたindicesの先頭0を追加した配列
            int[] resultAxisShapes = new int[indices.Length + 1];   //分割後の指定軸のShape

            for (int i = 0; i < indices.Length; i++)
            {
                shapeOffets[i + 1] = indices[i];
                resultAxisShapes[i] = indices[i] - shapeOffets[i];
            }
            resultAxisShapes[indices.Length] = array.Shape[axis] - indices[indices.Length - 1];

            NdArray[] resultArrays = new NdArray[indices.Length + 1];

            for (int i = 0; i < resultArrays.Length; i++)
            {
                int[] resultShape = array.Shape.ToArray();
                resultShape[axis] = resultAxisShapes[i];
                resultArrays[i] = new NdArray(resultShape, array.BatchCount);
                if(array.Grad != null) resultArrays[i].InitGrad();
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

        public static NdArray Rollaxis(NdArray input, int axis, int start = 0)
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

        public static NdArray Transpose(NdArray input, params int[] dimensions)
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

            NdArray resultMatrix = new NdArray(transposedDimensions);

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

        public static NdArray Concatenate(NdArray a, NdArray b, int axis)
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

            NdArray result = new NdArray(shapeList, a.BatchCount);
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

        internal int[] GetDimensionsIndex(int index)
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

        internal int GetLocalIndex(int batchIndex, params int[] indices)
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
}