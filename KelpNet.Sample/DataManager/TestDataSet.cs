using System;

namespace KelpNet.Sample.DataManager
{
    public class TestDataSet<T> where T : unmanaged, IComparable<T>
    {
        public NdArray<T> Data;
        public NdArray<T> Label;

        public TestDataSet(NdArray<T> data, NdArray<T> label)
        {
            this.Data = data;
            this.Label = label;
        }
    }
}
