using System;
using KelpNet.CPU;

namespace KelpNet.Tools
{
    public class ChainerModelDataLoader
    {
        public static void ModelLoad<T>(string path, FunctionStack<T> model) where T : unmanaged, IComparable<T>
        {
            var modelData = new NpzDictionary(path);

            foreach (var function in model.Functions)
            {
                SetParams(function, modelData);
            }
        }

        static void SetParams<T>(Function<T> func, NpzDictionary modelData) where T : unmanaged, IComparable<T>
        {
            if (func is Linear<T>)
            {
                Linear<T> linear = (Linear<T>)func;

                linear.Weight.Data = modelData[func.Name + "/W.npy"].FlattenEx<T>();

                if (linear.Bias != null)
                {
                    linear.Bias.Data = modelData[func.Name + "/b.npy"].FlattenEx<T>();
                }
            }
            else if (func is Convolution2D<T>)
            {
                Convolution2D<T> conv2D = (Convolution2D<T>)func;

                conv2D.Weight.Data = modelData[func.Name + "/W.npy"].FlattenEx<T>();

                if (conv2D.Bias != null)
                {
                    conv2D.Bias.Data = modelData[func.Name + "/b.npy"].FlattenEx<T>();
                }
            }
            else if (func is Deconvolution2D<T>)
            {
                Deconvolution2D<T> deconv2D = (Deconvolution2D<T>)func;

                deconv2D.Weight.Data = modelData[func.Name + "/W.npy"].FlattenEx<T>();

                if (!deconv2D.NoBias)
                {
                    deconv2D.Bias.Data = modelData[func.Name + "/b.npy"].FlattenEx<T>();
                }
            }
            else if (func is EmbedID<T>)
            {
                EmbedID<T> embed = (EmbedID<T>)func;
                embed.Weight.Data = modelData[func.Name + "/W.npy"].FlattenEx<T>();
            }
            else if (func is BatchNormalization<T>)
            {
                BatchNormalization<T> bn = (BatchNormalization<T>)func;

                bn.Beta.Data = modelData[func.Name + "/beta.npy"].FlattenEx<T>();
                bn.Gamma.Data = modelData[func.Name + "/gamma.npy"].FlattenEx<T>();

                if (bn.Train)
                {
                    if (modelData.ContainsKey(func.Name + "/avg_mean.npy")) bn.AvgMean.Data = modelData[func.Name + "/avg_mean.npy"].FlattenEx<T>();
                    if (modelData.ContainsKey(func.Name + "/avg_var.npy")) bn.AvgVar.Data = modelData[func.Name + "/avg_var.npy"].FlattenEx<T>();
                }
            }
            else if (func is MultiplyScale<T>)
            {
                MultiplyScale<T> scale = (MultiplyScale<T>)func;

                scale.Weight.Data = modelData[func.Name + "/W.npy"].FlattenEx<T>();

                if (scale.BiasTerm)
                {
                    scale.Bias.Data = modelData[func.Name + "/bias/b.npy"].FlattenEx<T>();
                }
            }
            else if (func is LSTM<T>)
            {
                LSTM<T> lstm = (LSTM<T>)func;

                lstm.lateral.Weight.Data = modelData[func.Name + "/lateral/W.npy"].FlattenEx<T>();
                lstm.upward.Weight.Data = modelData[func.Name + "/upward/W.npy"].FlattenEx<T>();
                lstm.upward.Bias.Data = modelData[func.Name + "/upward/b.npy"].FlattenEx<T>();
            }
        }
    }
}
