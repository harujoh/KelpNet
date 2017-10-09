using System;
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
                Array.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), ((Linear)func).Weight.Data, ((Linear)func).Weight.Data.Length);
                Array.Copy(Real.GetArray(modelData[func.Name + "/b.npy"]), ((Linear)func).Bias.Data, ((Linear)func).Bias.Data.Length);
            }
            else if (func is Convolution2D)
            {
                Array.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), ((Convolution2D)func).Weight.Data, ((Convolution2D)func).Weight.Data.Length);
                Array.Copy(Real.GetArray(modelData[func.Name + "/b.npy"]), ((Convolution2D)func).Bias.Data, ((Convolution2D)func).Bias.Data.Length);
            }
            else if (func is Deconvolution2D)
            {
                Array.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), ((Deconvolution2D)func).Weight.Data, ((Deconvolution2D)func).Weight.Data.Length);
                Array.Copy(Real.GetArray(modelData[func.Name + "/b.npy"]), ((Deconvolution2D)func).Bias.Data, ((Deconvolution2D)func).Bias.Data.Length);
            }
            else if (func is EmbedID)
            {
                Array.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), ((EmbedID)func).Weight.Data, ((EmbedID)func).Weight.Data.Length);
            }
            else if (func is BatchNormalization)
            {
                Array.Copy(Real.GetArray(modelData[func.Name + "/beta.npy"]), ((BatchNormalization)func).Beta.Data, ((BatchNormalization)func).Beta.Data.Length);
                Array.Copy(Real.GetArray(modelData[func.Name + "/gamma.npy"]), ((BatchNormalization)func).Gamma.Data, ((BatchNormalization)func).Gamma.Data.Length);
                Array.Copy(Real.GetArray(modelData[func.Name + "/avg_mean.npy"]), ((BatchNormalization)func).AvgMean.Data, ((BatchNormalization)func).AvgMean.Data.Length);
                Array.Copy(Real.GetArray(modelData[func.Name + "/avg_var.npy"]), ((BatchNormalization)func).AvgVar.Data, ((BatchNormalization)func).AvgVar.Data.Length);
            }
        }
    }
}
