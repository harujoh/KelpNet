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
                Array.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), ((Linear)func).W.Data, ((Linear)func).W.Data.Length);
                Array.Copy(Real.GetArray(modelData[func.Name + "/b.npy"]), ((Linear)func).b.Data, ((Linear)func).b.Data.Length);
            }
            else if (func is Convolution2D)
            {
                Array.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), ((Convolution2D)func).W.Data, ((Convolution2D)func).W.Data.Length);
                Array.Copy(Real.GetArray(modelData[func.Name + "/b.npy"]), ((Convolution2D)func).b.Data, ((Convolution2D)func).b.Data.Length);
            }
            else if (func is Deconvolution2D)
            {
                Array.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), ((Deconvolution2D)func).W.Data, ((Deconvolution2D)func).W.Data.Length);
                Array.Copy(Real.GetArray(modelData[func.Name + "/b.npy"]), ((Deconvolution2D)func).b.Data, ((Deconvolution2D)func).b.Data.Length);
            }
            else if (func is EmbedID)
            {
                Array.Copy(Real.GetArray(modelData[func.Name + "/W.npy"]), ((EmbedID)func).W.Data, ((EmbedID)func).W.Data.Length);
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
