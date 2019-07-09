using System;
using System.Collections.Generic;
using System.IO;
using KelpNet.CPU;
using ProtoBuf;

namespace KelpNet.Tools
{
    public class CaffemodelDataLoader
    {
        //binaryprotoを読み込む
        public static NdArray ReadBinary(string path)
        {
            using (FileStream stream = new FileStream(path, FileMode.Open))
            {
                BlobProto bp = Serializer.Deserialize<BlobProto>(stream);

                NdArray result = new NdArray(new[] { bp.Channels, bp.Height, bp.Width }, bp.Num);

                if (bp.Datas != null)
                {
                    result.Data = Real.ToRealArray(bp.Datas);
                }

                if (bp.DoubleDatas != null)
                {
                    result.Data = Real.ToRealArray(bp.DoubleDatas);
                }

                if (bp.Diffs != null)
                {
                    result.Grad = Real.ToRealArray(bp.Diffs);
                }

                if (bp.DoubleDiffs != null)
                {
                    result.Grad = Real.ToRealArray(bp.DoubleDiffs);
                }

                return result;
            }
        }

        //分岐ありモデル用関数
        public static FunctionDictionary LoadNetWork(string path)
        {
            FunctionDictionary functionDictionary = new FunctionDictionary();

            using (FileStream stream = new FileStream(path, FileMode.Open))
            {
                NetParameter netparam = Serializer.Deserialize<NetParameter>(stream);

                foreach (V1LayerParameter layer in netparam.Layers)
                {
                    Function func = CreateFunction(layer);

                    if (func != null)
                    {
                        functionDictionary.Add(func);
                    }
                }

                foreach (LayerParameter layer in netparam.Layer)
                {
                    Function func = CreateFunction(layer);

                    if (func != null)
                    {
                        functionDictionary.Add(func);
                    }
                }
            }

            return functionDictionary;
        }

        //分岐なしモデル用関数
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
            switch (layer.Type)
            {
                case "Scale":
                    return SetupScale(layer.ScaleParam, layer.Blobs, layer.Bottoms, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Split":
                    return new SplitFunction(layer.Tops.Count, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Slice":
                    return SetupSlice(layer.SliceParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "LRN":
                    return SetupLRN(layer.LrnParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Concat":
                    return SetupConcat(layer.ConcatParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Eltwise":
                    return SetupEltwise(layer.EltwiseParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "BatchNorm":
                    return SetupBatchnorm(layer.BatchNormParam, layer.Blobs, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Convolution":
                    return SetupConvolution(layer.ConvolutionParam, layer.Blobs, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Dropout":
                    return new Dropout(layer.DropoutParam.DropoutRatio, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Pooling":
                    return SetupPooling(layer.PoolingParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "ReLU":
                    return layer.ReluParam != null ? layer.ReluParam.NegativeSlope == 0 ? (Function)new ReLU(layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray()) : (Function)new LeakyReLU(layer.ReluParam.NegativeSlope, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray()) : (Function)new ReLU(layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "InnerProduct":
                    return SetupInnerProduct(layer.InnerProductParam, layer.Blobs, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Softmax":
                    return new Softmax(layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case "Data":
                    return null; //読み飛ばし

                case "SoftmaxWithLoss":
                    return null; //読み飛ばし
            }

            Console.WriteLine("Skip the layer \"{0}\", since CaffemodelLoader does not support {0} layer", layer.Type);

            return null;
        }

        static Function CreateFunction(V1LayerParameter layer)
        {
            switch (layer.Type)
            {
                case V1LayerParameter.LayerType.Split:
                    return new SplitFunction(layer.Tops.Count, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Slice:
                    return SetupSlice(layer.SliceParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Concat:
                    return SetupConcat(layer.ConcatParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Lrn:
                    return SetupLRN(layer.LrnParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Eltwise:
                    return SetupEltwise(layer.EltwiseParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Convolution:
                    return SetupConvolution(layer.ConvolutionParam, layer.Blobs, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Dropout:
                    return new Dropout(layer.DropoutParam.DropoutRatio, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Pooling:
                    return SetupPooling(layer.PoolingParam, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Relu:
                    return layer.ReluParam != null ? layer.ReluParam.NegativeSlope == 0 ? (Function)new ReLU(layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray()) : (Function)new LeakyReLU(layer.ReluParam.NegativeSlope, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray()) : (Function)new ReLU(layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.InnerProduct:
                    return SetupInnerProduct(layer.InnerProductParam, layer.Blobs, layer.Name, layer.Bottoms.ToArray(), layer.Tops.ToArray());

                case V1LayerParameter.LayerType.Softmax:
                    return new Softmax();

                case V1LayerParameter.LayerType.Data:
                    return null; //読み飛ばし

                case V1LayerParameter.LayerType.SoftmaxLoss:
                    return null; //読み飛ばし
            }

            Console.WriteLine("Skip the layer \"{0}\", since CaffemodelLoader does not support {0} layer", layer.Type);

            return null;
        }

        static Function SetupScale(ScaleParameter param, List<BlobProto> blobs, List<string> bottoms, string name, string[] inputNames, string[] outputNames)
        {
            //Caffe及びChainerは暗黙的に1次元目をBacthとして利用しているため補正を行う
            int axis = param.Axis - 1;
            bool biasTerm = param.BiasTerm;

            if (bottoms.Count == 1)
            {
                //Scaleを作成
                int[] wShape = new int[blobs[0].Shape.Dims.Length];

                for (int i = 0; i < wShape.Length; i++)
                {
                    wShape[i] = (int)blobs[0].Shape.Dims[i];
                }

                return new MultiplyScale(axis, wShape, biasTerm, blobs[0].Datas, blobs[1].Datas, name, inputNames, outputNames);
            }
            else
            {
                //Biasを作成
                int[] shape = new int[blobs[0].Shape.Dims.Length];

                for (int i = 0; i < shape.Length; i++)
                {
                    shape[i] = (int)blobs[0].Shape.Dims[i];
                }

                return new AddBias(axis, shape, blobs[0].Datas, name);
            }
        }

        static Function SetupSlice(SliceParameter param, string name, string[] inputNames, string[] outputNames)
        {
            int[] slicePoints = new int[param.SlicePoints.Length];

            for (int i = 0; i < slicePoints.Length; i++)
            {
                slicePoints[i] = (int)param.SlicePoints[i];
            }

            //Caffe及びChainerは暗黙的に1次元目をBacthとして利用しているため補正を行う
            return new SplitAxis(slicePoints, param.Axis - 1, name, inputNames, outputNames);
        }

        static Function SetupPooling(PoolingParameter param, string name, string[] inputNames, string[] outputNames)
        {
            int[] ksize = GetKernelSize(param);
            int[] stride = GetKernelStride(param);
            int[] pad = GetKernelPad(param);

            switch (param.Pool)
            {
                case PoolingParameter.PoolMethod.Max:
                    return new MaxPooling2D(ksize, stride, pad, name: name, inputNames: inputNames, outputNames: outputNames);

                case PoolingParameter.PoolMethod.Ave:
                    return new AveragePooling2D(ksize, stride, pad, name, inputNames, outputNames);
            }

            return null;
        }

        static BatchNormalization SetupBatchnorm(BatchNormParameter param, List<BlobProto> blobs, string name, string[] inputNames, string[] outputNames)
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

            BatchNormalization batchNormalization = new BatchNormalization(size, decay, eps, name: name, inputNames: inputNames, outputNames: outputNames);
            batchNormalization.AvgMean.Data = Real.ToRealArray(avgMean);
            batchNormalization.AvgVar.Data = Real.ToRealArray(avgVar);

            return batchNormalization;
        }

        static Convolution2D SetupConvolution(ConvolutionParameter param, List<BlobProto> blobs, string name, string[] inputNames, string[] outputNames)
        {
            int[] ksize = GetKernelSize(param);
            int[] stride = GetKernelStride(param);
            int[] pad = GetKernelPad(param);
            int num = GetNum(blobs[0]);
            int channels = GetChannels(blobs[0]);
            int nIn = channels * (int)param.Group;
            int nOut = num;
            float[] w = blobs[0].Datas;

            if (param.BiasTerm)
            {
                float[] b = blobs[1].Datas;
                return new Convolution2D(nIn, nOut, ksize, stride, pad, !param.BiasTerm, w, b, name: name, inputNames: inputNames, outputNames: outputNames);
            }

            return new Convolution2D(nIn, nOut, ksize, stride, pad, !param.BiasTerm, w, name: name, inputNames: inputNames, outputNames: outputNames);
        }

        static Linear SetupInnerProduct(InnerProductParameter param, List<BlobProto> blobs, string name, string[] inputNames, string[] outputNames)
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
                return new Linear(width, height, !param.BiasTerm, w, blobs[1].Datas, name: name, inputNames: inputNames, outputNames: outputNames);
            }

            return new Linear(width, height, !param.BiasTerm, w, name: name);
        }

        static LRN SetupLRN(LRNParameter param, string name, string[] inputNames, string[] outputNames)
        {
            return new LRN((int)param.LocalSize, param.K, param.Alpha / param.LocalSize, param.Beta, name, inputNames, outputNames);
        }

        static Eltwise SetupEltwise(EltwiseParameter param, string name, string[] inputNames, string[] outputNames)
        {
            if (param != null)
            {
                return new Eltwise(param.Operation, param.Coeffs, name, inputNames, outputNames);
            }
            else
            {
                return new Eltwise(EltwiseParameter.EltwiseOp.Sum, null, name, inputNames, outputNames);
            }
        }

        static Concat SetupConcat(ConcatParameter param, string name, string[] inputNames, string[] outputNames)
        {
            int axis = param.Axis;

            if (axis == 1 && param.ConcatDim != 1)
            {
                axis = (int)param.ConcatDim;
            }

            //Caffe及びChainerは暗黙的に1次元目をBacthとして利用しているため補正を行う
            return new Concat(axis - 1, name, inputNames, outputNames);
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

        static int[] GetKernelSize(ConvolutionParameter param)
        {
            if (param.KernelH > 0)
            {
                return new []{ (int)param.KernelW, (int)param.KernelH };
            }

            if (param.KernelSizes.Length == 1)
            {
                return new []{(int)param.KernelSizes[0], (int)param.KernelSizes[0]};
            }

            return new []{(int)param.KernelSizes[1], (int)param.KernelSizes[0]};
        }

        static int[] GetKernelSize(PoolingParameter param)
        {
            if (param.KernelH > 0)
            {
                return new []{(int)param.KernelW, (int)param.KernelH};
            }

            return new []{(int)param.KernelSize, (int)param.KernelSize};
        }

        static int[] GetKernelStride(ConvolutionParameter param)
        {
            if (param.StrideH > 0)
            {
                return new []{(int)param.StrideW, (int)param.StrideH};
            }

            if (param.Strides == null || param.Strides.Length == 0)
            {
                return new []{1, 1};
            }

            if (param.Strides.Length == 1)
            {
                return new []{(int)param.Strides[0], (int)param.Strides[0]};
            }

            return new []{(int)param.Strides[1], (int)param.Strides[0]};
        }

        static int[] GetKernelStride(PoolingParameter param)
        {
            if (param.StrideH > 0)
            {
                return new []{(int)param.StrideW, (int)param.StrideH};
            }

            return new []{(int)param.Stride, (int)param.Stride};
        }


        static int[] GetKernelPad(ConvolutionParameter param)
        {
            if (param.PadH > 0)
            {
                return new []{(int)param.PadW, (int)param.PadH};
            }

            if (param.Pads == null || param.Pads.Length == 0)
            {
                return new []{1, 1};
            }

            if (param.Pads.Length == 1)
            {
                return new []{(int)param.Pads[0], (int)param.Pads[0]};
            }

            return new []{(int)param.Pads[1], (int)param.Pads[0]};
        }

        static int[] GetKernelPad(PoolingParameter param)
        {
            if (param.PadH > 0)
            {
                return new []{(int)param.PadW, (int)param.PadH};
            }

            return new []{(int)param.Pad, (int)param.Pad};
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
