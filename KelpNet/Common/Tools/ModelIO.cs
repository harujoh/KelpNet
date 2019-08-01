using System;
using System.IO;
using System.IO.Compression;
using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    public class ModelIO
    {
        public static Type[] KnownTypes = new Type[]
        {
            typeof(Real),
            typeof(NdArray),
            typeof(FunctionDictionary),//Container
            typeof(FunctionStack),
            typeof(DualInputFunction),//Type
            typeof(MultiInputFunction),
            typeof(MultiOutputFunction),
            typeof(SingleInputFunction),
            typeof(SplitFunction),
            typeof(Optimizer),//Optimizer
            typeof(AdaBound),
            typeof(AdaDelta),
            typeof(AdaGrad),
            typeof(Adam),
            typeof(AdamW),
            typeof(AmsBound),
            typeof(AmsGrad),
            typeof(GradientClipping),
            typeof(MomentumSGD),
            typeof(RMSprop),
            typeof(SGD),
            typeof(OptimizerParameter),//OptimizerParameter
            typeof(AdaBoundParameter),
            typeof(AdaDeltaParameter),
            typeof(AdaGradParameter),
            typeof(AdamParameter),
            typeof(AdamWParameter),
            typeof(AmsBoundParameter),
            typeof(AmsGradParameter),
            typeof(GradientClippingParameter),
            typeof(MomentumSGDParameter),
            typeof(RMSpropParameter),
            typeof(SGDParameter),
            typeof(ELU),//Activations
            typeof(Softmax),
            typeof(Swish),
            typeof(Broadcast),//Arrays
            typeof(EmbedID),//Connections
            typeof(LSTM),
            typeof(AddBias),//Mathmetrics
            typeof(MultiplyScale),
            typeof(StochasticDepth),//Noise
            typeof(BatchNormalization),//Normalization
            typeof(AveragePooling2D),//Poolings
            typeof(LeakyReLU),//Parallizable
            typeof(ReLU),
            typeof(Sigmoid),
            typeof(TanhActivation),
            typeof(Convolution2D),
            typeof(Deconvolution2D),
            typeof(Linear),
            typeof(Dropout),
            typeof(MaxPooling2D)
        };

        public static void Save(Function function, string fileName)
        {
            DataContractSerializer bf = new DataContractSerializer(typeof(Function), KnownTypes);

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

        public static Function Load(string fileName)
        {
            DataContractSerializer bf = new DataContractSerializer(typeof(Function), KnownTypes);

            ZipArchiveEntry zipData = ZipFile.OpenRead(fileName).GetEntry("Function");
            Function result = (Function)bf.ReadObject(zipData.Open());

            if (result is FunctionStack functionStack)
            {
                InitFunctionStack(functionStack);
            }
            else if (result is FunctionDictionary functionDictionary)
            {
                foreach (FunctionStack functionBlock in functionDictionary.FunctionBlocks)
                {
                    InitFunctionStack(functionBlock);
                }
            }

            return result;
        }

        static void InitFunctionStack(FunctionStack functionStack)
        {
            foreach (Function function in functionStack.Functions)
            {
                function.ResetState();
            }
        }
    }
}