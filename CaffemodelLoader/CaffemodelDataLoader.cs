using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Arrays;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
using KelpNet.Functions.Normalization;
using KelpNet.Functions.Poolings;
using ProtoBuf;

namespace CaffemodelLoader
{
    public class CaffemodelDataLoader
    {
        public static List<Function> ModelLoad(string path)
        {
            List<Function> result = new List<Function>();

            using (FileStream stream = new FileStream(path, FileMode.Open))
            {
                NetParameter netparam = Serializer.Deserialize<NetParameter>(stream);

                foreach (V1LayerParameter layer in netparam.Layers)
                {
                    Function func = CreateFunction(layer);

                    if (func != null)
                    {
                        result.Add(func);
                    }
                }

                foreach (LayerParameter layer in netparam.Layer)
                {
                    Function func = CreateFunction(layer);

                    if (func != null)
                    {
                        result.Add(func);
                    }
                }
            }

            return result;
        }

        static Function CreateFunction(LayerParameter layer)
        {
            Console.WriteLine(layer.Type);
            switch (layer.Type)
            {
                case "LRN":
                    return SetupLRN(layer.LrnParam, layer.Name);

                case "Scale":
                    return null;

                case "Concat":
                    return SetupConcat(layer.ConcatParam, layer.Name);

                case "Eltwise":
                    return SetupEltwise(layer.EltwiseParam, layer.Name);

                case "BatchNorm":
                    return SetupBatchnorm(layer.BatchNormParam, layer.Blobs, layer.Name);

                case "Convolution":
                    return SetupConvolution(layer.ConvolutionParam, layer.Blobs, layer.Name);

                case "Dropout":
                    return new Dropout(layer.DropoutParam.DropoutRatio, layer.Name);

                case "Pooling":
                    return SetupPooling(layer.PoolingParam, layer.Name);

                case "ReLU":
                    return layer.ReluParam != null ? layer.ReluParam.NegativeSlope == 0 ? (Function)new ReLU(layer.Name) : (Function)new LeakyReLU(layer.ReluParam.NegativeSlope, layer.Name) : (Function)new LeakyReLU(name: layer.Name);

                case "InnerProduct":
                    return SetupInnerProduct(layer.InnerProductParam, layer.Blobs, layer.Name);

                case "Softmax":
                    return new Softmax();

                case "SoftmaxWithLoss":
                    return null;
            }

            Console.WriteLine("Skip the layer \"{0}\", since CaffemodelLoader does not support {0} layer", layer.Type);

            return null;
        }

        static Function CreateFunction(V1LayerParameter layer)
        {
            switch (layer.Type)
            {
                case V1LayerParameter.LayerType.Concat:
                    return SetupConcat(layer.ConcatParam, layer.Name);

                case V1LayerParameter.LayerType.Lrn:
                    return SetupLRN(layer.LrnParam, layer.Name);

                case V1LayerParameter.LayerType.Eltwise:
                    return SetupEltwise(layer.EltwiseParam, layer.Name);

                case V1LayerParameter.LayerType.Convolution:
                    return SetupConvolution(layer.ConvolutionParam, layer.Blobs, layer.Name);

                case V1LayerParameter.LayerType.Dropout:
                    return new Dropout(layer.DropoutParam.DropoutRatio, layer.Name);

                case V1LayerParameter.LayerType.Pooling:
                    return SetupPooling(layer.PoolingParam, layer.Name);

                case V1LayerParameter.LayerType.Relu:
                    return layer.ReluParam != null ? new LeakyReLU(layer.ReluParam.NegativeSlope, layer.Name) : new LeakyReLU(name: layer.Name);

                case V1LayerParameter.LayerType.InnerProduct:
                    return SetupInnerProduct(layer.InnerProductParam, layer.Blobs, layer.Name);

                case V1LayerParameter.LayerType.Softmax:
                    return new Softmax();

                case V1LayerParameter.LayerType.SoftmaxLoss:
                    return null;
            }

            Console.WriteLine("Skip the layer \"{0}\", since CaffemodelLoader does not support {0} layer", layer.Type);

            return null;
        }

        static Function SetupPooling(PoolingParameter param, string name)
        {
            Size ksize = GetKernelSize(param);
            Size stride = GetKernelStride(param);
            Size pad = GetKernelPad(param);

            switch (param.Pool)
            {
                case PoolingParameter.PoolMethod.Max:
                    return new MaxPooling(ksize, stride, pad, name);

                case PoolingParameter.PoolMethod.Ave:
                    return new AveragePooling(ksize, stride, pad, name);
            }

            return null;
        }

        static BatchNormalization SetupBatchnorm(BatchNormParameter param, List<BlobProto> blobs, string name)
        {
            double decay = param.MovingAverageFraction;
            double eps = param.Eps;
            int size = (int)blobs[0].Shape.Dims[0];

            float[] avgMean = blobs[0].Datas;
            float[] avgVar = blobs[1].Datas;

            if (blobs.Count >= 3)
            {
                float scalingFactor = blobs[2].Datas[0];

                for (int i = 0; i < avgMean.Length; i++)
                {
                    avgMean[i] /= scalingFactor;
                }

                for (int i = 0; i < avgVar.Length; i++)
                {
                    avgVar[i] /= scalingFactor;
                }
            }

            BatchNormalization batchNormalization = new BatchNormalization(size, decay, eps, avgMean, avgVar);

            return batchNormalization;
        }

        static Convolution2D SetupConvolution(ConvolutionParameter param, List<BlobProto> blobs, string name)
        {
            Size ksize = GetKernelSize(param);
            Size stride = GetKernelStride(param);
            Size pad = GetKernelPad(param);
            int num = GetNum(blobs[0]);
            int channels = GetChannels(blobs[0]);
            int nIn = channels * (int)param.Group;
            int nOut = num;
            float[] w = blobs[0].Datas;

            if (param.BiasTerm)
            {
                float[] b = blobs[1].Datas;
                return new Convolution2D(nIn, nOut, ksize, stride, pad, !param.BiasTerm, w, b, name);
            }

            return new Convolution2D(nIn, nOut, ksize, stride, pad, !param.BiasTerm, w, name: name);
        }

        static Linear SetupInnerProduct(InnerProductParameter param, List<BlobProto> blobs, string name)
        {
            if (param.Axis != 1)
            {
                throw new Exception("Non-default axis in InnerProduct is not supported");
            }

            int width = GetWidth(blobs[0]);
            int height = GetHeight(blobs[0]);
            float[] w = blobs[0].Datas;

            if (param.BiasTerm)
            {
                return new Linear(width, height, !param.BiasTerm, w, blobs[1].Datas, name);
            }

            return new Linear(width, height, !param.BiasTerm, w, name: name);
        }

        static LRN SetupLRN(LRNParameter param, string name)
        {
            return new LRN((int)param.LocalSize, param.K, param.Alpha / param.LocalSize, param.Beta, name);
        }

        static Eltwise SetupEltwise(EltwiseParameter param, string name)
        {
            return new Eltwise(param.Operation, param.Coeffs, name);
        }

        static Concat SetupConcat(ConcatParameter param, string name)
        {
            int axis = param.Axis;

            if (axis == 1 && param.ConcatDim != 1)
            {
                axis = (int)param.ConcatDim;
            }

            return new Concat(axis, name);
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

        public class Eltwise : Function
        {
            private const string FUNCTION_NAME = "Eltwise";

            List<NdArray[]> PrevInput = new List<NdArray[]>();
            List<int[]> PrevOutputIndex = new List<int[]>();

            private EltwiseParameter.EltwiseOp _operation;
            private float[] _coeffs;

            public Eltwise(EltwiseParameter.EltwiseOp operation, float[] coeffs, string name = FUNCTION_NAME) : base(name)
            {
                this._operation = operation;
                this._coeffs = coeffs;
            }

            public NdArray ForwardCPU(params NdArray[] xs)
            {
                PrevInput.Add(xs);

                Real[] result = new Real[xs[0].Data.Length];
                Array.Copy(xs[0].Data, result, result.Length);

                switch (_operation)
                {
                    case EltwiseParameter.EltwiseOp.Prod:
                        for (int i = 1; i < xs.Length; i++)
                        {
                            for (int j = 0; j < result.Length; j++)
                            {
                                result[j] *= xs[i].Data[j];
                            }
                        }
                        break;

                    case EltwiseParameter.EltwiseOp.Sum:
                        for (int i = 1; i < xs.Length; i++)
                        {
                            for (int j = 0; j < result.Length; j++)
                            {
                                result[j] += xs[i].Data[j] * _coeffs[i];
                            }
                        }
                        break;

                    case EltwiseParameter.EltwiseOp.Max:
                        int[] outputIndex = new int[result.Length];

                        for (int i = 1; i < xs.Length; i++)
                        {
                            for (int j = 0; j < result.Length; j++)
                            {
                                if (result[j] < xs[i].Data[j])
                                {
                                    outputIndex[j] = i;
                                    result[j] = xs[i].Data[j];
                                }
                            }
                        }

                        PrevOutputIndex.Add(outputIndex);
                        break;
                }

                return NdArray.Convert(result, xs[0].Shape, xs[0].BatchCount);
            }

            public NdArray[] BackwardCPU(NdArray gy)
            {
                var prevInput = PrevInput[PrevInput.Count - 1];
                PrevInput.RemoveAt(PrevInput.Count - 1);

                Real[][] result = new Real[prevInput.Length][];
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = new Real[prevInput[i].Length];
                }

                switch (_operation)
                {
                    case EltwiseParameter.EltwiseOp.Prod:
                        for (int i = 0; i < result.Length; i++)
                        {
                            Array.Copy(gy.Data, result[i], gy.Data.Length);
                            for (int j = 0; j < prevInput.Length; j++)
                            {
                                if (i != j)
                                {
                                    for (int k = 0; k < result[i].Length; k++)
                                    {
                                        result[i][k] *= prevInput[j].Data[k];
                                    }
                                }
                            }
                        }
                        break;

                    case EltwiseParameter.EltwiseOp.Sum:
                        for (int i = 0; i < result.Length; i++)
                        {
                            Array.Copy(gy.Data, result[i], result[i].Length);
                        }
                        break;

                    case EltwiseParameter.EltwiseOp.Max:
                        var prevOutputIndex = PrevOutputIndex[PrevOutputIndex.Count - 1];
                        PrevOutputIndex.RemoveAt(PrevOutputIndex.Count - 1);

                        for (int i = 0; i < prevOutputIndex.Length; i++)
                        {
                            result[prevOutputIndex[i]][i] = gy.Data[i];
                        }
                        break;
                }

                NdArray[] resultArrays = new NdArray[prevInput.Length];

                for (int i = 0; i < prevInput.Length; i++)
                {
                    resultArrays[i] = NdArray.Convert(result[i], prevInput[0].Shape, prevInput[0].BatchCount);
                }

                return resultArrays;
            }
        }
    }
}
