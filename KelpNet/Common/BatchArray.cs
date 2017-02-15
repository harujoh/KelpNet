using System;
using System.Linq;

namespace KelpNet.Common
{
    [Serializable]
    public class BatchArray : NdArray
    {
        public int BatchCount;
        public int Length;

        private BatchArray()
        {
        }

        public BatchArray(Array data)
        {
            NdArray ndArray = NdArray.FromArray(data);
            this.Shape = ndArray.Shape;
            this.Length = ndArray.Data.Length;
            this.BatchCount = 1;
            this.Data = ndArray.Data;
        }

        public BatchArray(double[] data, int[] shape, int batchCount)
        {
            this.Shape = shape.ToArray();
            this.Length = ShapeToArrayLength(this.Shape);
            this.BatchCount = batchCount;
            this.Data = data.ToArray();
        }

        public BatchArray(int[] shape, int batchCount)
        {
            this.Shape = shape.ToArray();
            this.Length = ShapeToArrayLength(this.Shape);
            this.BatchCount = batchCount;
            this.Data = new double[this.Length * batchCount];
        }

        public BatchArray(NdArray ndArray)
        {
            this.Data = ndArray.Data.ToArray();
            this.Shape = ndArray.Shape.ToArray();
            this.Length = ndArray.Data.Length;
            this.BatchCount = 1;
        }

        public BatchArray(NdArray[] ndArray)
        {
            this.BatchCount = ndArray.Length;
            this.Shape = ndArray[0].Shape.ToArray();
            this.Length = ndArray[0].Data.Length;

            int arrayLength = 0;
            for (int i = 0; i < ndArray.Length; i++)
            {
                arrayLength += ndArray[i].Data.Length;
            }

            this.Data = new double[arrayLength];
            for (int i = 0; i < ndArray.Length; i++)
            {
                Buffer.BlockCopy(ndArray[i].Data, 0, this.Data, this.Length * i * sizeof(double), ndArray[i].Data.Length * sizeof(double));
            }
        }

        public static BatchArray FromArray(Array[] data)
        {
            NdArray[] result = new NdArray[data.Length];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = FromArray(data[i]);
            }

            return new BatchArray(result);
        }

        public static BatchArray Convert(double[] data, int[] shape, int batchCount)
        {
            return new BatchArray { Data = data, Shape = shape.ToArray(), BatchCount = batchCount, Length = data.Length / batchCount };
        }
    }
}
