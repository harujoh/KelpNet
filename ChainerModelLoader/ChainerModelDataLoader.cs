using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Normalization;

namespace ChainerModelLoader
{
    public class ChainerModelDataLoader
    {
        public static void ModelLoad(string path, FunctionStack model)
        {
            var modelData = new NpzDictionary(path);

            foreach (var function in model.Functions)
            {
                SetParams(function, modelData);
            }
        }

        static void SetParams(Function func, NpzDictionary modelData)
        {
            if (func is Linear)
            {
                ((Linear)func).W.Data = Real.GetArray(modelData[func.Name + "/W.npy"]);
                ((Linear)func).b.Data = Real.GetArray(modelData[func.Name + "/b.npy"]);
            }
            else if (func is Convolution2D)
            {
                ((Convolution2D)func).W.Data = Real.GetArray(modelData[func.Name + "/W.npy"]);
                ((Convolution2D)func).b.Data = Real.GetArray(modelData[func.Name + "/b.npy"]);
            }
            else if (func is Deconvolution2D)
            {
                ((Deconvolution2D)func).W.Data = Real.GetArray(modelData[func.Name + "/W.npy"]);
                ((Deconvolution2D)func).b.Data = Real.GetArray(modelData[func.Name + "/b.npy"]);
            }
            else if (func is EmbedID)
            {
                ((EmbedID) func).W.Data = Real.GetArray(modelData[func.Name + "/W.npy"]);
            }
            else if (func is BatchNormalization)
            {
                ((BatchNormalization)func).Beta.Data = Real.GetArray(modelData[func.Name + "/beta.npy"]);
                ((BatchNormalization)func).Gamma.Data = Real.GetArray(modelData[func.Name + "/gamma.npy"]);
                ((BatchNormalization)func).AvgMean.Data = Real.GetArray(modelData[func.Name + "/avg_mean.npy"]);
                ((BatchNormalization)func).AvgVar.Data = Real.GetArray(modelData[func.Name + "/avg_var.npy"]);
            }
        }
    }
}
