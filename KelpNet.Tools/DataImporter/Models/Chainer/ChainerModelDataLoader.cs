using System;
using KelpNet.CPU;

namespace KelpNet.Tools
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
                Linear linear = (Linear)func;

                Array.Copy(Real.ToRealArray(modelData[func.Name + "/W.npy"]), linear.Weight.Data, linear.Weight.Data.Length);

                if (!linear.NoBias)
                {
                    Array.Copy(Real.ToRealArray(modelData[func.Name + "/b.npy"]), linear.Bias.Data, linear.Bias.Data.Length);
                }
            }
            else if (func is Convolution2D)
            {
                Convolution2D conv2D = (Convolution2D)func;

                Array.Copy(Real.ToRealArray(modelData[func.Name + "/W.npy"]), conv2D.Weight.Data, conv2D.Weight.Data.Length);

                if (!conv2D.NoBias)
                {
                    Array.Copy(Real.ToRealArray(modelData[func.Name + "/b.npy"]), conv2D.Bias.Data, conv2D.Bias.Data.Length);
                }
            }
            else if (func is Deconvolution2D)
            {
                Deconvolution2D deconv2D = (Deconvolution2D)func;

                Array.Copy(Real.ToRealArray(modelData[func.Name + "/W.npy"]), deconv2D.Weight.Data, deconv2D.Weight.Data.Length);

                if (!deconv2D.NoBias)
                {
                    Array.Copy(Real.ToRealArray(modelData[func.Name + "/b.npy"]), deconv2D.Bias.Data, deconv2D.Bias.Data.Length);
                }
            }
            else if (func is EmbedID)
            {
                EmbedID embed = (EmbedID)func;

                Array.Copy(Real.ToRealArray(modelData[func.Name + "/W.npy"]), embed.Weight.Data, embed.Weight.Data.Length);
            }
            else if (func is BatchNormalization)
            {
                BatchNormalization bn = (BatchNormalization)func;

                Array.Copy(Real.ToRealArray(modelData[func.Name + "/beta.npy"]), bn.Beta.Data, bn.Beta.Data.Length);
                Array.Copy(Real.ToRealArray(modelData[func.Name + "/gamma.npy"]), bn.Gamma.Data, bn.Gamma.Data.Length);

                if (bn.Train)
                {
                    if (modelData.ContainsKey(func.Name + "/avg_mean.npy")) Array.Copy(Real.ToRealArray(modelData[func.Name + "/avg_mean.npy"]), bn.AvgMean.Data, bn.AvgMean.Data.Length);
                    if (modelData.ContainsKey(func.Name + "/avg_var.npy")) Array.Copy(Real.ToRealArray(modelData[func.Name + "/avg_var.npy"]), bn.AvgVar.Data, bn.AvgVar.Data.Length);
                }
            }
            else if (func is MultiplyScale)
            {
                MultiplyScale scale = (MultiplyScale)func;

                Array.Copy(Real.ToRealArray(modelData[func.Name + "/W.npy"]), scale.Weight.Data, scale.Weight.Data.Length);

                if (scale.BiasTerm)
                {
                    Array.Copy(Real.ToRealArray(modelData[func.Name + "/bias/b.npy"]), scale.Bias.Data, scale.Bias.Data.Length);
                }
            }
            else if (func is LSTM)
            {
                LSTM lstm = (LSTM)func;

                Real[] lateral = Real.ToRealArray(modelData[func.Name + "/lateral/W.npy"]);
                Real[] upwardW = Real.ToRealArray(modelData[func.Name + "/upward/W.npy"]);
                Real[] upwardb = Real.ToRealArray(modelData[func.Name + "/upward/b.npy"]);

                Array.Copy(lateral, 0, lstm.lateral.Weight.Data, 0, lstm.lateral.Weight.Data.Length);
                Array.Copy(upwardW, 0, lstm.upward.Weight.Data, 0, lstm.upward.Weight.Data.Length);
                Array.Copy(upwardb, 0, lstm.upward.Bias.Data, 0, lstm.upward.Bias.Data.Length);
            }
        }
    }
}
