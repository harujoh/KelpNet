using System;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
#if DOUBLE
using Real = System.Double;
#else
using Real = System.Single;
#endif

namespace KelpNet.CPU
{
#if !DOUBLE
    [DataContract(Name = "Deconvolution2D", Namespace = "KelpNet")]
    public class Deconvolution2D<T> : SingleInputFunction<T>, ICompressibleFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Deconvolution2D";

        [DataMember]
        public NdArray<T> Weight;

        [DataMember]
        public NdArray<T> Bias;


        [DataMember]
        public bool NoBias;


        [DataMember]
        public int KernelWidth;

        [DataMember]
        public int KernelHeight;

        [DataMember]
        public int StrideX;

        [DataMember]
        public int StrideY;

        [DataMember]
        public int PadX;

        [DataMember]
        public int PadY;


        [DataMember]
        public int InputCount;

        [DataMember]
        public int OutputCount;


        [DataMember]
        public ICompressibleActivation<T> Activation { get; set; }


        public Deconvolution2D(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        public Deconvolution2D(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.KernelWidth = kernelSize;
            this.KernelHeight = kernelSize;
            this.PadX = pad;
            this.PadY = pad;
            this.StrideX = stride;
            this.StrideY = stride;
            this.NoBias = noBias;

            this.Parameters = new NdArray<T>[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Activation = activation;

            this.Initialize(initialW, initialb);
            InitFunc(new StreamingContext());
        }

        public Deconvolution2D(int inputChannels, int outputChannels, int[] kernelSize, int[] subSample = null, int[] trim = null, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            if (subSample == null)
                subSample = new[] { 1, 1 };

            if (trim == null)
                trim = new[] { 0, 0 };

            this.KernelWidth = kernelSize[0];
            this.KernelHeight = kernelSize[1];
            this.PadX = trim[0];
            this.PadY = trim[1];
            this.NoBias = noBias;

            this.StrideX = subSample[0];
            this.StrideY = subSample[1];

            this.Parameters = new NdArray<T>[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Activation = activation;

            this.Initialize(initialW, initialb);
            InitFunc(new StreamingContext());
        }

        void Initialize(Array initialW = null, Array initialb = null)
        {
            this.Weight = new NdArray<T>(InputCount, OutputCount, this.KernelHeight, this.KernelWidth);
            this.Weight.Name = this.Name + " Weight";

            if (initialW == null)
            {
                Initializer.InitHeNorm(this.Weight);
            }
            else
            {
                this.Weight.Data = initialW.FlattenEx<T>();
            }

            this.Parameters[0] = this.Weight;


            if (!NoBias)
            {
                this.Bias = new NdArray<T>(OutputCount);
                this.Bias.Name = this.Name + " Bias";

                if (initialb != null)
                {
                    this.Bias.Data = initialb.FlattenEx<T>();
                }

                this.Parameters[1] = this.Bias;
            }
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Deconvolution2D<float> deconvolution2DF:
                    deconvolution2DF.SingleInputForward = x => Deconvolution2DF.SingleInputForward(x, deconvolution2DF.Weight, deconvolution2DF.Bias, deconvolution2DF.NoBias, deconvolution2DF.OutputCount, deconvolution2DF.KernelWidth, deconvolution2DF.KernelHeight, deconvolution2DF.StrideX, deconvolution2DF.StrideY, deconvolution2DF.PadX, deconvolution2DF.PadY, deconvolution2DF, deconvolution2DF.Activation);
                    deconvolution2DF.SingleOutputBackward = (y, x) => Deconvolution2DF.SingleOutputBackward(y, x, deconvolution2DF.Weight, deconvolution2DF.Bias, deconvolution2DF.NoBias, deconvolution2DF.InputCount, deconvolution2DF.OutputCount, deconvolution2DF.StrideX, deconvolution2DF.StrideY, deconvolution2DF.PadX, deconvolution2DF.PadY, deconvolution2DF.Activation);
                    break;

                case Deconvolution2D<double> deconvolution2DD:
                    deconvolution2DD.SingleInputForward = x => Deconvolution2DD.SingleInputForward(x, deconvolution2DD.Weight, deconvolution2DD.Bias, deconvolution2DD.NoBias, deconvolution2DD.OutputCount, deconvolution2DD.KernelWidth, deconvolution2DD.KernelHeight, deconvolution2DD.StrideX, deconvolution2DD.StrideY, deconvolution2DD.PadX, deconvolution2DD.PadY, deconvolution2DD, deconvolution2DD.Activation);
                    deconvolution2DD.SingleOutputBackward = (y, x) => Deconvolution2DD.SingleOutputBackward(y, x, deconvolution2DD.Weight, deconvolution2DD.Bias, deconvolution2DD.NoBias, deconvolution2DD.InputCount, deconvolution2DD.OutputCount, deconvolution2DD.StrideX, deconvolution2DD.StrideY, deconvolution2DD.PadX, deconvolution2DD.PadY, deconvolution2DD.Activation);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class Deconvolution2DD
#else
    public static class Deconvolution2DF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> input, NdArray<Real> weight, NdArray<Real> bias, bool noBias, int outputCount, int kernelWidth, int kernelHeight, int strideX, int strideY, int padX, int padY, IFunction<Real> deconv2d, ICompressibleActivation<Real> activation)
        {
            int outputHeight = (input.Shape[1] - 1) * strideY + kernelHeight - padY * 2;
            int outputWidth = (input.Shape[2] - 1) * strideX + kernelWidth - padX * 2;

            Real[] result = new Real[input.BatchCount * outputCount * outputWidth * outputHeight];

            int outSizeOffset = outputWidth * outputHeight;
            int inputSizeOffset = input.Shape[1] * input.Shape[2];
            int kSizeOffset = weight.Shape[2] * weight.Shape[3];

            for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
            {
                for (int och = 0; och < outputCount; och++)
                {
                    for (int oy = padY; oy < outputHeight + padY; oy++)
                    {
                        int iyLimit = oy / strideY + 1 < input.Shape[1] ? oy / strideY + 1 : input.Shape[1];
                        int iyStart = oy - weight.Shape[2] < 0 ? 0 : (oy - weight.Shape[2]) / strideY + 1;

                        for (int ox = padX; ox < outputWidth + padX; ox++)
                        {
                            int ixLimit = ox / strideX + 1 < input.Shape[2] ? ox / strideX + 1 : input.Shape[2];
                            int ixStart = ox - weight.Shape[3] < 0 ? 0 : (ox - weight.Shape[3]) / strideX + 1;

                            int outputIndex = batchCount * outputCount * outSizeOffset + och * outSizeOffset + (oy - padY) * outputWidth + ox - padX;

                            for (int ich = 0; ich < input.Shape[0]; ich++)
                            {
                                int inputIndexOffset = batchCount * input.Length + ich * inputSizeOffset;
                                int kernelIndexOffset = ich * weight.Shape[1] * kSizeOffset + och * kSizeOffset;

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int inputIndex = inputIndexOffset + iy * input.Shape[2] + ix;
                                        int kernelIndex = kernelIndexOffset + (oy - iy * strideY) * weight.Shape[3] + (ox - ix * strideX);

                                        result[outputIndex] += input.Data[inputIndex] * weight.Data[kernelIndex];
                                    }
                                }
                            }

                        }
                    }
                }
            }

            if (activation != null && !noBias)
            {
                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    for (int och = 0; och < outputCount; och++)
                    {
                        for (int oy = padY; oy < outputHeight + padY; oy++)
                        {
                            for (int ox = padX; ox < outputWidth + padX; ox++)
                            {
                                int outputIndex = batchCount * outputCount * outSizeOffset + och * outSizeOffset + (oy - padY) * outputWidth + ox - padX;

                                result[outputIndex] += bias.Data[och];
                                result[outputIndex] = activation.ForwardActivate(result[outputIndex]);
                            }
                        }
                    }
                }
            }
            else if (!noBias)
            {
                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    for (int och = 0; och < outputCount; och++)
                    {
                        for (int oy = padY; oy < outputHeight + padY; oy++)
                        {
                            for (int ox = padX; ox < outputWidth + padX; ox++)
                            {
                                int outputIndex = batchCount * outputCount * outSizeOffset + och * outSizeOffset + (oy - padY) * outputWidth + ox - padX;

                                result[outputIndex] += bias.Data[och];
                            }
                        }
                    }
                }
            }
            else if (activation != null)
            {
                for (int batchCount = 0; batchCount < input.BatchCount; batchCount++)
                {
                    for (int och = 0; och < outputCount; och++)
                    {
                        for (int oy = padY; oy < outputHeight + padY; oy++)
                        {
                            for (int ox = padX; ox < outputWidth + padX; ox++)
                            {
                                int outputIndex = batchCount * outputCount * outSizeOffset + och * outSizeOffset + (oy - padY) * outputWidth + ox - padX;

                                result[outputIndex] = activation.ForwardActivate(result[outputIndex]);
                            }
                        }
                    }
                }
            }

            return NdArray.Convert(result, new[] { outputCount, outputHeight, outputWidth }, input.BatchCount, deconv2d);
        }

        public static void CalcBiasGrad(Real[] gy, Real[] biasGrad, int[] gyShape, int batchCount)
        {
            int gyIndex = 0;

            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int och = 0; och < gyShape[0]; och++)
                {
                    for (int olocation = 0; olocation < gyShape[1] * gyShape[2]; olocation++)
                    {
                        biasGrad[och] += gy[gyIndex];
                        gyIndex++;
                    }
                }
            }
        }

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> weight, NdArray<Real> bias, bool noBias, int inputCount, int outputCount, int strideX, int strideY, int padX, int padY, ICompressibleActivation<Real> activation)
        {
            Real[] activatedgy = activation != null ? activation.GetActivatedgy(y) : y.Grad;
            if (!noBias)
            {
                CalcBiasGrad(activatedgy, bias.Grad, y.Shape, y.BatchCount);
            }

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int och = 0; och < outputCount; och++)
                {
                    int wOchOffset = och * weight.Shape[2] * weight.Shape[3];
                    int yChOffset = och * y.Shape[1] * y.Shape[2];

                    for (int oy = padY; oy < y.Shape[1] + padY; oy++)
                    {
                        int iyLimit = oy / strideY + 1 < x.Shape[1] ? oy / strideY + 1 : x.Shape[1];
                        int iyStart = oy - weight.Shape[2] < 0 ? 0 : (oy - weight.Shape[2]) / strideY + 1;

                        for (int ox = padX; ox < y.Shape[2] + padX; ox++)
                        {
                            int ixLimit = ox / strideX + 1 < x.Shape[2] ? ox / strideX + 1 : x.Shape[2];
                            int ixStart = ox - weight.Shape[3] < 0 ? 0 : (ox - weight.Shape[3]) / strideX + 1;

                            int gyIndex = batchCount * y.Length + yChOffset + (oy - padY) * y.Shape[2] + ox - padX;

                            for (int ich = 0; ich < inputCount; ich++)
                            {
                                int wIchOffset = ich * weight.Shape[1] * weight.Shape[2] * weight.Shape[3] + wOchOffset;
                                int xChOffset = batchCount * x.Length + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int xIndex = xChOffset + iy * x.Shape[2] + ix;
                                        int wIndex = wIchOffset + (oy - iy * strideY) * weight.Shape[3] + (ox - ix * strideX);

                                        weight.Grad[wIndex] += x.Data[xIndex] * activatedgy[gyIndex];
                                        x.Grad[xIndex] += weight.Data[wIndex] * activatedgy[gyIndex];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
