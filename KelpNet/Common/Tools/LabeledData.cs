using System;

namespace KelpNet
{
    [Serializable]
    public class LabeledData
    {
        public Real[] Data;
        public Real Label;

        public LabeledData(Real[] data, Real label)
        {
            Data = data;
            Label = label;
        }

        public static LabeledData[] Convert(Real[][] data, Real[] label)
        {
#if DEBUG
            if (data.Length != label.Length) throw new Exception();
#endif
            LabeledData[] result = new LabeledData[data.Length];

            for (int i = 0; i < data.Length; i++)
            {
                result[i] = new LabeledData(data[i], label[i]);
            }

            return result;
        }
    }
}
