using System;
using System.Runtime.Serialization;

#if DOUBLE
using Real = System.Double;
#elif NETSTANDARD2_1
using Real = System.Single;
using Math = System.MathF;
#else
using Real = System.Single;
using Math = KelpNet.MathF;
#endif

namespace KelpNet.CPU
{
#if !DOUBLE
    [DataContract(Name = "Convolution2D", Namespace = "KelpNet")]
    public class Convolution2D<T> : SingleInputFunction<T>, ICompressibleFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Convolution2D";

        [DataMember]
        public NdArray<T> Weight;

        [DataMember]
        public NdArray<T> Bias;


        [DataMember]
        public int StrideX;

        [DataMember]
        public int StrideY;

        [DataMember]
        public int PadX;

        [DataMember]
        public int PadY;


        [DataMember]
        public ICompressibleActivation<T> Activation { get; set; }

        //OpenCLのため
        public Convolution2D(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            InitFunc(new StreamingContext());
        }

        public Convolution2D(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.StrideX = stride;
            this.StrideY = stride;
            this.PadX = pad;
            this.PadY = pad;

            this.Parameters = new NdArray<T>[noBias ? 1 : 2];

            this.Weight = new NdArray<T>(outputChannels, inputChannels, kernelSize, kernelSize);
            if (!noBias) this.Bias = new NdArray<T>(outputChannels);

            this.Activation = activation;

            this.Initialize(initialW, initialb);
            InitFunc(new StreamingContext());
        }

        public Convolution2D(int inputChannels, int outputChannels, int[] kernelSize, int[] stride = null, int[] pad = null, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            if (pad == null)
                pad = new[] { 0, 0 };

            if (stride == null)
                stride = new[] { 1, 1 };

            this.Weight = new NdArray<T>(outputChannels, inputChannels, kernelSize[1], kernelSize[0]);
            if (!noBias) this.Bias = new NdArray<T>(outputChannels);

            this.StrideX = stride[0];
            this.StrideY = stride[1];
            this.PadX = pad[0];
            this.PadY = pad[1];

            this.Parameters = new NdArray<T>[noBias ? 1 : 2];

            this.Activation = activation;

            this.Initialize(initialW, initialb);
            InitFunc(new StreamingContext());
        }

        public Convolution2D(Linear<T> linear) : base(linear.Name, linear.InputNames, linear.OutputNames)
        {
            this.StrideX = 1;
            this.StrideY = 1;
            this.PadX = 0;
            this.PadY = 0;

            this.Parameters = linear.Parameters;

            this.Weight = linear.Weight;
            this.Weight.Reshape(this.Weight.Shape[0], this.Weight.Shape[1], 1, 1);
            this.Bias = linear.Bias;
            this.Activation = linear.Activation;
            InitFunc(new StreamingContext());
        }

        void Initialize(Array initialW = null, Array initialb = null)
        {
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

            if (this.Bias != null)
            {
                this.Bias.Name = this.Name + " Bias";

                if (initialb != null)
                {
                    this.Bias.Data = initialb.FlattenEx<T>();
                }

                this.Parameters[1] = this.Bias;
            }

            InitFunc(new StreamingContext());
        }

        [OnDeserializing]
        void InitFunc(StreamingContext sc)
        {
            switch (this)
            {
                case Convolution2D<float> convolution2DF:
                    convolution2DF.SingleInputForward = x => Convolution2DF.SingleInputForward(x, convolution2DF.Weight, convolution2DF.Bias, convolution2DF.StrideX, convolution2DF.StrideY, convolution2DF.PadX, convolution2DF.PadY, convolution2DF.Activation, convolution2DF);
                    convolution2DF.SingleOutputBackward = (y, x) => Convolution2DF.SingleOutputBackward(y, x, convolution2DF.Weight, convolution2DF.Bias, convolution2DF.StrideX, convolution2DF.StrideY, convolution2DF.PadX, convolution2DF.PadY, convolution2DF.Activation);
                    break;

                case Convolution2D<double> convolution2DD:
                    convolution2DD.SingleInputForward = x => Convolution2DD.SingleInputForward(x, convolution2DD.Weight, convolution2DD.Bias, convolution2DD.StrideX, convolution2DD.StrideY, convolution2DD.PadX, convolution2DD.PadY, convolution2DD.Activation, convolution2DD);
                    convolution2DD.SingleOutputBackward = (y, x) => Convolution2DD.SingleOutputBackward(y, x, convolution2DD.Weight, convolution2DD.Bias, convolution2DD.StrideX, convolution2DD.StrideY, convolution2DD.PadX, convolution2DD.PadY, convolution2DD.Activation);
                    break;
            }
        }
    }
#endif

#if DOUBLE
    public static class Convolution2DD
#else
    public static class Convolution2DF
#endif
    {
        public static NdArray<Real> SingleInputForward(NdArray<Real> x, NdArray<Real> weight, NdArray<Real> bias, int strideX, int strideY, int padX, int padY, ICompressibleActivation<Real> activation, IFunction<Real> conv2d)
        {
            int outputCount = weight.Shape[0];
            int inputCount = weight.Shape[1];
            int kernelHeight = weight.Shape[2];
            int kernelWidth = weight.Shape[3];

            int outputHeight = (int)Math.Floor((x.Shape[1] - kernelHeight + padY * 2.0f) / strideY) + 1;
            int outputWidth = (int)Math.Floor((x.Shape[2] - kernelWidth + padX * 2.0f) / strideX) + 1;

            Real[] y = new Real[x.BatchCount * outputCount * outputHeight * outputWidth];

            for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
            {
                int yBatchOffset = batchCounter * outputCount * outputHeight * outputWidth;
                int xBatchOffset = batchCounter * x.Length;

                for (int och = 0; och < outputCount; och++)
                {
                    int kOchOffset = och * inputCount * kernelHeight * kernelWidth;

                    int yChOffset = yBatchOffset + och * outputHeight * outputWidth;

                    for (int oy = 0; oy < outputHeight * strideY; oy += strideY)
                    {
                        int iyStart = oy - padY < 0 ? 0 : oy - padY;
                        int iyLimit = kernelHeight + oy - padY < x.Shape[1] ? kernelHeight + oy - padY : x.Shape[1];

                        for (int ox = 0; ox < outputWidth * strideX; ox += strideX)
                        {
                            int ixStart = ox - padX < 0 ? 0 : ox - padX;
                            int ixLimit = kernelWidth + ox - padX < x.Shape[2] ? kernelWidth + ox - padX : x.Shape[2];

                            int yIndex = yChOffset + oy / strideY * outputWidth + ox / strideX;

                            for (int ich = 0; ich < inputCount; ich++)
                            {
                                int kIchOffset = kOchOffset + ich * kernelHeight * kernelWidth;

                                int xChOffset = xBatchOffset + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int wIndex = kIchOffset + (iy - oy + padY) * kernelWidth + ix - ox + padX;
                                        int xIndex = xChOffset + iy * x.Shape[2] + ix;

                                        y[yIndex] += x.Data[xIndex] * weight.Data[wIndex];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (activation != null && bias != null)
            {
                for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * outputCount * outputHeight * outputWidth;

                    for (int och = 0; och < outputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            y[resultIndex] += bias.Data[och];
                            y[resultIndex] = activation.ForwardActivate(y[resultIndex]);

                            resultIndex++;
                        }
                    }
                }
            }
            else if (bias != null)
            {
                for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * outputCount * outputHeight * outputWidth;

                    for (int och = 0; och < outputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            y[resultIndex] += bias.Data[och];
                            resultIndex++;
                        }
                    }
                }
            }
            else if (activation != null)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    y[i] = activation.ForwardActivate(y[i]);
                }
            }

            return NdArray.Convert(y, new[] { outputCount, outputHeight, outputWidth }, x.BatchCount, conv2d);
        }

        public static void CalcBiasGrad(Real[] gy, int[] gyShape, int batchCount, Real[] biasGrad)
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

        public static void SingleOutputBackward(NdArray<Real> y, NdArray<Real> x, NdArray<Real> weight, NdArray<Real> bias, int strideX, int strideY, int padX, int padY, ICompressibleActivation<Real> activation)
        {
            //int outputCount = weight.Shape[0];
            int inputCount = weight.Shape[1];
            int kernelHeight = weight.Shape[2];
            int kernelWidth = weight.Shape[3];

            Real[] activatedgy = activation != null ? activation.GetActivatedgy(y, x) : y.Grad;
            if (bias != null) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount, bias.Grad);

            for (int batchCounter = 0; batchCounter < y.BatchCount; batchCounter++)
            {
                int yBatchOffset = batchCounter * y.Length;
                int xBatchOffset = batchCounter * x.Length;

                for (int och = 0; och < y.Shape[0]; och++)
                {
                    int wOchOffset = och * inputCount * kernelHeight * kernelWidth;

                    int yChOffset = och * y.Shape[1] * y.Shape[2];

                    for (int oy = 0; oy < y.Shape[1] * strideY; oy += strideY)
                    {
                        int iyStart = oy - padY < 0 ? 0 : oy - padY;
                        int iyLimit = kernelHeight + oy - padY < x.Shape[1] ? kernelHeight + oy - padY : x.Shape[1];

                        for (int ox = 0; ox < y.Shape[2] * strideX; ox += strideX)
                        {
                            int ixStart = ox - padX < 0 ? 0 : ox - padX;
                            int ixLimit = kernelWidth + ox - padX < x.Shape[2] ? kernelWidth + ox - padX : x.Shape[2];

                            int gyIndex = yBatchOffset + yChOffset + oy / strideY * y.Shape[2] + ox / strideX;

                            for (int ich = 0; ich < x.Shape[0]; ich++)
                            {
                                int wIchOffset = wOchOffset + ich * kernelHeight * kernelWidth;

                                int xChOffset = xBatchOffset + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int wIndex = wIchOffset + (iy - oy + padY) * kernelWidth + ix - ox + padX;
                                        int xIndex = xChOffset + iy * x.Shape[2] + ix;

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
