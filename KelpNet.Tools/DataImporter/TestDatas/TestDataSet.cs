namespace KelpNet.Tools
{
    public class TestDataSet
    {
        public NdArray Data;
        public NdArray Label;

        public TestDataSet(NdArray data, NdArray label)
        {
            this.Data = data;
            this.Label = label;
        }
    }
}
