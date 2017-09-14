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
    public class CaffemodelDataLoader
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
                    return SetupConvolution(layer);

                case V1LayerParameter.LayerType.Dropout:
                    return new Dropout(layer.DropoutParam.DropoutRatio, layer.Name);

                case V1LayerParameter.LayerType.Pooling:
                    return SetupPooling(layer);

                case V1LayerParameter.LayerType.Relu:
                    return layer.ReluParam != null ? new LeakyReLU(layer.ReluParam.NegativeSlope, layer.Name) : new LeakyReLU(name:layer.Name);

                case V1LayerParameter.LayerType.InnerProduct:
                    return SetupInnerProduct(layer);
            }

            Console.WriteLine("Skip the layer \"{0}\", since CaffemodelLoader does not support {0} layer", layer.Type);

            return null;
        }

        static Function SetupPooling(V1LayerParameter layer)
        {
            var param = layer.PoolingParam;
            var ksize = GetKernelSize(param);
            var stride = GetKernelStride(param);
            var pad = GetKernelPad(param);

            switch (param.Pool)
            {
                case PoolingParameter.PoolMethod.Max:
                    return new MaxPooling(ksize, stride, pad, layer.Name);

                case PoolingParameter.PoolMethod.Ave:
                    return new AveragePooling(ksize, stride, pad, layer.Name);
            }

            return null;
        }

        static Convolution2D SetupConvolution(V1LayerParameter layer)
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
                return new Convolution2D(nIn, nOut, ksize, stride, pad, !param.BiasTerm, w, b, layer.Name);
            }

            return new Convolution2D(nIn, nOut, ksize, stride, pad, !param.BiasTerm, w, name: layer.Name);
        }

        static Linear SetupInnerProduct(V1LayerParameter layer)
        {
            var param = layer.InnerProductParam;

            if (param.Axis != 1)
            {
                throw new Exception("Non-default axis in InnerProduct is not supported");
            }

            var blobs = layer.Blobs;
            var width = GetWidth(blobs[0]);
            var height = GetHeight(blobs[0]);
            var w = blobs[0].Datas;

            if (param.BiasTerm)
            {
                return new Linear(width, height, !param.BiasTerm, w, blobs[1].Datas, layer.Name);
            }

            return new Linear(width, height, !param.BiasTerm, w, name: layer.Name);
        }

        static int GetHeight(BlobProto blob)
        {
            if (blob.Height > 0)
                return blob.Height;

            if (blob.Shape.Dims.Length == 2)
                return (int)blob.Shape.Dims[0];

            if (blob.Shape.Dims.Length == 4)
                return (int)blob.Shape.Dims[2];

            throw new Exception(blob.Shape.Dims.Length + "-dimentional array is not supported");
        }

        static int GetWidth(BlobProto blob)
        {
            if (blob.Width > 0)
                return blob.Width;

            if (blob.Shape.Dims.Length == 2)
                return (int)blob.Shape.Dims[1];

            if (blob.Shape.Dims.Length == 4)
                return (int)blob.Shape.Dims[3];

            throw new Exception(blob.Shape.Dims.Length + "-dimentional array is not supported");
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

        static Size GetKernelSize(PoolingParameter param)
        {
            if (param.KernelH > 0)
            {
                return new Size((int)param.KernelW, (int)param.KernelH);
            }

            return new Size((int)param.KernelSize, (int)param.KernelSize);
        }

        static Size GetKernelStride(ConvolutionParameter param)
        {
            if (param.StrideH > 0)
            {
                return new Size((int)param.StrideW, (int)param.StrideH);
            }

            if (param.Strides == null || param.Strides.Length == 0)
            {
                return new Size(1, 1);
            }

            if (param.Strides.Length == 1)
            {
                return new Size((int)param.Strides[0], (int)param.Strides[0]);
            }

            return new Size((int)param.Strides[1], (int)param.Strides[0]);
        }

        static Size GetKernelStride(PoolingParameter param)
        {
            if (param.StrideH > 0)
            {
                return new Size((int)param.StrideW, (int)param.StrideH);
            }

            return new Size((int)param.Stride, (int)param.Stride);
        }


        static Size GetKernelPad(ConvolutionParameter param)
        {
            if (param.PadH > 0)
            {
                return new Size((int)param.PadW, (int)param.PadH);
            }

            if (param.Pads == null || param.Pads.Length == 0)
            {
                return new Size(1, 1);
            }

            if (param.Pads.Length == 1)
            {
                return new Size((int)param.Pads[0], (int)param.Pads[0]);
            }

            return new Size((int)param.Pads[1], (int)param.Pads[0]);
        }

        static Size GetKernelPad(PoolingParameter param)
        {
            if (param.PadH > 0)
            {
                return new Size((int)param.PadW, (int)param.PadH);
            }

            return new Size((int)param.Pad, (int)param.Pad);
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
