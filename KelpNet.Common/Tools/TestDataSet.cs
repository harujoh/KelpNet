using System;

namespace KelpNet
{
    public class TestDataSet<T> where T : unmanaged, IComparable<T>
    {
        public NdArray<T> Data;
        public NdArray<int> Label;

        public TestDataSet(NdArray<T> data, NdArray<int> label)
        {
            this.Data = data;
            this.Label = label;
        }
    }
}
