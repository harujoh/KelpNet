using System;
using System.IO;
using System.IO.Compression;
using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    public class ModelIO<T> where T : unmanaged, IComparable<T>
    {
        public static Type[] KnownTypes =
        {
            typeof(NdArray<T>),
            typeof(FunctionDictionary<T>),//Container
            typeof(FunctionStack<T>),
            typeof(DualInputFunction<T>),//Type
            typeof(MultiInputFunction<T>),
            typeof(MultiOutputFunction<T>),
            typeof(SingleInputFunction<T>),
            typeof(SplitFunction<T>),
            typeof(ELU<T>),//Activations
            typeof(Softmax<T>),
            typeof(Swish<T>),
            typeof(Broadcast<T>),//Arrays
            typeof(EmbedID<T>),//Connections
            typeof(LSTM<T>),
            typeof(AddBias<T>),//Mathmetrics
            typeof(MultiplyScale<T>),
            //typeof(StochasticDepth),//Noise
            typeof(BatchNormalization<T>),//Normalization
            typeof(AveragePooling2D<T>),//Poolings
            typeof(LeakyReLU<T>),//Parallizable
            typeof(ReLU<T>),
            typeof(Sigmoid<T>),
            typeof(TanhActivation<T>),
            typeof(Convolution2D<T>),
            typeof(Deconvolution2D<T>),
            typeof(Linear<T>),
            typeof(Dropout<T>),
            typeof(MaxPooling2D<T>)
        };

        public static void Save(Function<T> function, string fileName)
        {
            DataContractSerializer bf = new DataContractSerializer(typeof(Function<T>), new DataContractSerializerSettings { KnownTypes = KnownTypes, PreserveObjectReferences = true });

            //ZIP書庫を作成
            if (File.Exists(fileName))
            {
                File.Delete(fileName);
            }

            using (ZipArchive zipArchive = ZipFile.Open(fileName, ZipArchiveMode.Create))
            {
                ZipArchiveEntry entry = zipArchive.CreateEntry("Function");
                using (Stream stream = entry.Open())
                {
                    bf.WriteObject(stream, function);
                }
            }
        }

        public static Function<T> Load(string fileName)
        {
            DataContractSerializer bf = new DataContractSerializer(typeof(Function<T>), new DataContractSerializerSettings { KnownTypes = KnownTypes, PreserveObjectReferences = true });

            using (ZipArchive zipArchive = ZipFile.OpenRead(fileName))
            {
                ZipArchiveEntry zipData = zipArchive.GetEntry("Function");
                return (Function<T>)bf.ReadObject(zipData.Open());
            }
        }
    }
}