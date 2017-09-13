using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using KelpNet.Common.Functions;
using KelpNet.Functions;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
using KelpNet.Functions.Poolings;
using ProtoBuf;

namespace CaffemodelLoader
{
    class CaffemodelDataLoader
    {
        public static FunctionStack ModelLoad(string path)
        {
            List<Function> result = new List<Function>();

            using (FileStream stream = new FileStream(path, FileMode.Open))
            {
                var netparam = Serializer.Deserialize<NetParameter>(stream);

                foreach (var layer in netparam.Layers)
                {
                    var func = CreateFunction(layer);
                    if (func != null)
                    {
                        result.Add(func);
                    }
                }
            }

            return new FunctionStack(result.ToArray());
        }

        static Function CreateFunction(V1LayerParameter layer)
        {
            switch (layer.Type)
            {
                case V1LayerParameter.LayerType.Convolution:
                    return SetupConvolution2D(layer);

                case V1LayerParameter.LayerType.Dropout:
                    return new Dropout();

                case V1LayerParameter.LayerType.Pooling:
                    var poolingParam = layer.PoolingParam;

                    switch (poolingParam.Pool)
                    {
                        case PoolingParameter.PoolMethod.Max:
                            return new MaxPooling(0);

                        case PoolingParameter.PoolMethod.Ave:
                            return new AveragePooling(0);
                    }
                    break;

                case V1LayerParameter.LayerType.Relu:
                    return new ReLU();

                case V1LayerParameter.LayerType.InnerProduct:
                    return new Linear(0, 0);
            }

            Console.WriteLine("Skip the layer \"{0}\", since CaffemodelLoader does not support {0} layer", layer.Type);

            return null;
        }

        static Convolution2D SetupConvolution2D(V1LayerParameter layer)
        {
            var blobs = layer.Blobs;
            var param = layer.ConvolutionParam;

            var ksize = GetKernelSize(param);
            var stride = GetKernelStride(param);
            var pad = GetKernelPad(param);

            var num = GetNum(blobs[0]);
            var channels = GetChannels(blobs[0]);

            var nIn = channels * (int)param.Group;
            var nOut = num;

            var w = blobs[0].Datas;

            if (param.BiasTerm)
            {
                var b = blobs[1].Datas;
                return new Convolution2D(nIn, nOut, ksize, stride, pad, false, w, b);
            }

            return new Convolution2D(nIn, nOut, ksize, stride, pad);
        }

        static Size GetKernelSize(ConvolutionParameter param)
        {
            if (param.KernelH > 0)
            {
                return new Size((int)param.KernelW, (int)param.KernelH);
            }

            if (param.KernelSizes.Length == 1)
            {
                return new Size((int)param.KernelSizes[0], (int)param.KernelSizes[0]);
            }

            return new Size((int)param.KernelSizes[1], (int)param.KernelSizes[0]);
        }

        static Size GetKernelStride(ConvolutionParameter param)
        {
            if (param.StrideH > 0)
            {
                return new Size((int)param.StrideW, (int)param.StrideH);
            }

            if (param.Strides.Length == 1)
            {
                return new Size((int)param.Strides[0], (int)param.Strides[0]);
            }

            return new Size((int)param.Strides[1], (int)param.Strides[0]);
        }

        static Size GetKernelPad(ConvolutionParameter param)
        {
            if (param.PadH > 0)
            {
                return new Size((int)param.PadW, (int)param.PadH);
            }

            if (param.Pads.Length == 1)
            {
                return new Size((int)param.Pads[0], (int)param.Pads[0]);
            }

            return new Size((int)param.Pads[1], (int)param.Pads[0]);
        }

        static int GetNum(BlobProto brob)
        {
            if (brob.Num > 0)
            {
                return brob.Num;
            }

            return (int)brob.Shape.Dims[0];
        }

        static int GetChannels(BlobProto brob)
        {
            if (brob.Channels > 0)
            {
                return brob.Channels;
            }

            return (int)brob.Shape.Dims[1];
        }
    }
}
