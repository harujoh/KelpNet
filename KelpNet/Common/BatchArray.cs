using System;
using System.Linq;
using System.Text;
using Cloo;

namespace KelpNet.Common
{
    [Serializable]
    public struct BatchArray
    {
        public Real[] Data;
        public int[] Shape;

        public int BatchCount;
        public int Length;

        [NonSerialized]
        public ComputeBuffer<Real> GpuData;

        public BatchArray(Array data)
        {
            NdArray ndArray = NdArray.FromArray(data);
            this.Shape = ndArray.Shape;
            this.Length = ndArray.Data.Length;
            this.BatchCount = 1;
            this.Data = ndArray.Data;
            this.GpuData = null;
        }

        public BatchArray(Real[] data, int[] shape, int batchCount)
        {
            this.Shape = shape.ToArray();
            this.Length = NdArray.ShapeToArrayLength(this.Shape);
            this.BatchCount = batchCount;
            this.Data = data.ToArray();
            this.GpuData = null;
        }

        public BatchArray(int[] shape, int batchCount)
        {
            this.Shape = shape.ToArray();
            this.Length = NdArray.ShapeToArrayLength(this.Shape);
            this.BatchCount = batchCount;
            this.Data = new Real[this.Length * batchCount];
            this.GpuData = null;
        }

        public BatchArray(NdArray ndArray)
        {
            this.Data = ndArray.Data.ToArray();
            this.Shape = ndArray.Shape.ToArray();
            this.Length = ndArray.Data.Length;
            this.BatchCount = 1;
            this.GpuData = null;
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

            this.Data = new Real[arrayLength];
            for (int i = 0; i < ndArray.Length; i++)
            {
                Array.Copy(ndArray[i].Data, 0, this.Data, this.Length * i, ndArray[i].Data.Length);
            }
            this.GpuData = null;
        }

        public NdArray GetNdArray(int i)
        {
            Real[] data = new Real[this.Length];
            Array.Copy(this.Data, i * this.Length, data, 0, this.Length);

            return new NdArray(data, this.Shape);
        }

        public static BatchArray FromArrays(Array[] data)
        {
            NdArray[] result = new NdArray[data.Length];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NdArray.FromArray(data[i]);
            }

            return new BatchArray(result);
        }

        public static BatchArray Convert(Real[] data, int[] shape, int batchCount)
        {
            return new BatchArray(shape, batchCount) { Data = data};
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();

            if (this.BatchCount > 1)
            {
                sb.Append("{");
            }

            sb.Append(this.GetNdArray(0));

            for (int i = 1; i < this.BatchCount; i++)
            {
                if (this.BatchCount != 0)
                {
                    sb.Append("},\n{"+ this.GetNdArray(i));
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
