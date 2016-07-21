namespace MNISTLoader
{
    public class MnistDataLoader
    {
        public byte[] TrainLabel;
        public byte[][] TrainData;

        public byte[] TeachLabel;
        public byte[][] TeachData;

        public MnistDataLoader()
        {
            string trainLabelPath = "data/train-labels.idx1-ubyte";
            MnistLabelLoader trainLabelLoader = MnistLabelLoader.Load(trainLabelPath);
            TrainLabel = trainLabelLoader.labelList;

            string trainImagePath = "data/train-images.idx3-ubyte";
            MnistImageLoader trainImageLoader = MnistImageLoader.Load(trainImagePath);
            TrainData = trainImageLoader.bitmapList.ToArray();


            string teachLabelPath = "data/t10k-labels.idx1-ubyte";
            MnistLabelLoader teachLabelLoader = MnistLabelLoader.Load(teachLabelPath);
            this.TeachLabel = teachLabelLoader.labelList;

            string teachImagePath = "data/t10k-images.idx3-ubyte";
            MnistImageLoader teachImageLoader = MnistImageLoader.Load(teachImagePath);
            this.TeachData = teachImageLoader.bitmapList.ToArray();
        }
    }
}
