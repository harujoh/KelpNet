using System;
using System.Runtime.Serialization;

namespace KelpNet.CPU
{
    [DataContract(Name = "Convolution2D", Namespace = "KelpNet")]
    public class Convolution2D : SingleInputFunction, ICompressibleFunction
    {
        const string FUNCTION_NAME = "Convolution2D";

        [DataMember]
        public NdArray Weight;

        [DataMember]
        public NdArray Bias;

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
        public ICompressibleActivation Activation { get; set; }

        public Convolution2D(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        public Convolution2D(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int pad = 0, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.KernelWidth = kernelSize;
            this.KernelHeight = kernelSize;
            this.StrideX = stride;
            this.StrideY = stride;
            this.PadX = pad;
            this.PadY = pad;
            this.NoBias = noBias;

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Activation = activation;

            this.Initialize(initialW, initialb);
        }

        public Convolution2D(int inputChannels, int outputChannels, int[] kernelSize, int[] stride = null, int[] pad = null, bool noBias = false, Array initialW = null, Array initialb = null, ICompressibleActivation activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            if (pad == null)
                pad = new[] { 0, 0 };

            if (stride == null)
                stride = new[] { 1, 1 };

            this.KernelWidth = kernelSize[0];
            this.KernelHeight = kernelSize[1];
            this.StrideX = stride[0];
            this.StrideY = stride[1];
            this.PadX = pad[0];
            this.PadY = pad[1];
            this.NoBias = noBias;

            this.Parameters = new NdArray[noBias ? 1 : 2];

            this.OutputCount = outputChannels;
            this.InputCount = inputChannels;

            this.Activation = activation;

            this.Initialize(initialW, initialb);
        }

        public Convolution2D(Linear linear) : base(linear.Name, linear.InputNames, linear.OutputNames)
        {
            this.KernelWidth = 1;
            this.KernelHeight = 1;
            this.StrideX = 1;
            this.StrideY = 1;
            this.PadX = 0;
            this.PadY = 0;

            this.Parameters = linear.Parameters;

            this.Weight = linear.Weight;
            this.Weight.Reshape(OutputCount, InputCount, this.KernelHeight, this.KernelWidth);
            this.Bias = linear.Bias;
            this.NoBias = linear.NoBias;
            this.Activation = linear.Activation;
        }

        void Initialize(Array initialW = null, Array initialb = null)
        {
            this.Weight = new NdArray(OutputCount, InputCount, this.KernelHeight, this.KernelWidth);
            this.Weight.Name = this.Name + " Weight";

            if (initialW == null)
            {
                Initializer.InitWeight(this.Weight);
            }
            else
            {
                this.Weight.Data = Real.ToRealArray(initialW);
            }

            this.Parameters[0] = this.Weight;

            if (!NoBias)
            {
                this.Bias = new NdArray(OutputCount);
                this.Bias.Name = this.Name + " Bias";

                if (initialb != null)
                {
                    this.Bias.Data = Real.ToRealArray(initialb);
                }

                this.Parameters[1] = this.Bias;
            }
        }

        public override NdArray SingleInputForward(NdArray x)
        {
            int outputHeight = (int)Math.Floor((x.Shape[1] - this.KernelHeight + this.PadY * 2.0) / this.StrideY) + 1;
            int outputWidth = (int)Math.Floor((x.Shape[2] - this.KernelWidth + this.PadX * 2.0) / this.StrideX) + 1;

            Real[] y = new Real[x.BatchCount * this.OutputCount * outputHeight * outputWidth];

            for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
            {
                int yBatchOffset = batchCounter * this.OutputCount * outputHeight * outputWidth;
                int xBatchOffset = batchCounter * x.Length;

                for (int och = 0; och < this.OutputCount; och++)
                {
                    int kOchOffset = och * this.InputCount * this.KernelHeight * this.KernelWidth;

                    int yChOffset = yBatchOffset + och * outputHeight * outputWidth;

                    for (int oy = 0; oy < outputHeight * this.StrideY; oy += this.StrideY)
                    {
                        int iyStart = oy - this.PadY < 0 ? 0 : oy - this.PadY;
                        int iyLimit = this.KernelHeight + oy - this.PadY < x.Shape[1] ? this.KernelHeight + oy - this.PadY : x.Shape[1];

                        for (int ox = 0; ox < outputWidth * this.StrideX; ox += this.StrideX)
                        {
                            int ixStart = ox - this.PadX < 0 ? 0 : ox - this.PadX;
                            int ixLimit = this.KernelWidth + ox - this.PadX < x.Shape[2] ? this.KernelWidth + ox - this.PadX : x.Shape[2];

                            int yIndex = yChOffset + oy / this.StrideY * outputWidth + ox / this.StrideX;

                            for (int ich = 0; ich < this.InputCount; ich++)
                            {
                                int kIchOffset = kOchOffset + ich * this.KernelHeight * this.KernelWidth;

                                int xChOffset = xBatchOffset + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int wIndex = kIchOffset + (iy - oy + this.PadY) * this.KernelWidth + ix - ox + this.PadX;
                                        int xIndex = xChOffset + iy * x.Shape[2] + ix;

                                        y[yIndex] += x.Data[xIndex] * this.Weight.Data[wIndex];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (this.Activation != null && !NoBias)
            {
                for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            y[resultIndex] += this.Bias.Data[och];
                            y[resultIndex] = this.Activation.ForwardActivate(y[resultIndex]);

                            resultIndex++;
                        }
                    }
                }
            }
            else if (!NoBias)
            {
                for (int batchCounter = 0; batchCounter < x.BatchCount; batchCounter++)
                {
                    int resultIndex = batchCounter * this.OutputCount * outputHeight * outputWidth;

                    for (int och = 0; och < this.OutputCount; och++)
                    {
                        for (int location = 0; location < outputHeight * outputWidth; location++)
                        {
                            y[resultIndex] += this.Bias.Data[och];
                            resultIndex++;
                        }
                    }
                }
            }
            else if (this.Activation != null)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    y[i] = this.Activation.ForwardActivate(y[i]);
                }
            }

            return NdArray.Convert(y, new[] { this.OutputCount, outputHeight, outputWidth }, x.BatchCount, this);
        }

        protected void CalcBiasGrad(Real[] gy, int[] gyShape, int batchCount)
        {
            int gyIndex = 0;

            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int och = 0; och < gyShape[0]; och++)
                {
                    for (int olocation = 0; olocation < gyShape[1] * gyShape[2]; olocation++)
                    {
                        this.Bias.Grad[och] += gy[gyIndex];

                        gyIndex++;
                    }
                }
            }
        }

        public override void SingleOutputBackward(NdArray y, NdArray x)
        {
            Real[] activatedgy = this.Activation != null ? this.GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.Shape, y.BatchCount);

            for (int batchCounter = 0; batchCounter < y.BatchCount; batchCounter++)
            {
                int yBatchOffset = batchCounter * y.Length;
                int xBatchOffset = batchCounter * x.Length;

                for (int och = 0; och < y.Shape[0]; och++)
                {
                    int wOchOffset = och * this.InputCount * this.KernelHeight * this.KernelWidth;

                    int yChOffset = och * y.Shape[1] * y.Shape[2];

                    for (int oy = 0; oy < y.Shape[1] * this.StrideY; oy += this.StrideY)
                    {
                        int iyStart = oy - this.PadY < 0 ? 0 : oy - this.PadY;
                        int iyLimit = this.KernelHeight + oy - this.PadY < x.Shape[1] ? this.KernelHeight + oy - this.PadY : x.Shape[1];

                        for (int ox = 0; ox < y.Shape[2] * this.StrideX; ox += this.StrideX)
                        {
                            int ixStart = ox - this.PadX < 0 ? 0 : ox - this.PadX;
                            int ixLimit = this.KernelWidth + ox - this.PadX < x.Shape[2] ? this.KernelWidth + ox - this.PadX : x.Shape[2];

                            int gyIndex = yBatchOffset + yChOffset + oy / this.StrideY * y.Shape[2] + ox / this.StrideX;

                            for (int ich = 0; ich < x.Shape[0]; ich++)
                            {
                                int wIchOffset = wOchOffset + ich * this.KernelHeight * this.KernelWidth;

                                int xChOffset = xBatchOffset + ich * x.Shape[1] * x.Shape[2];

                                for (int iy = iyStart; iy < iyLimit; iy++)
                                {
                                    for (int ix = ixStart; ix < ixLimit; ix++)
                                    {
                                        int wIndex = wIchOffset + (iy - oy + this.PadY) * this.KernelWidth + ix - ox + this.PadX;
                                        int xIndex = xChOffset + iy * x.Shape[2] + ix;

                                        this.Weight.Grad[wIndex] += x.Data[xIndex] * activatedgy[gyIndex];
                                        x.Grad[xIndex] += this.Weight.Data[wIndex] * activatedgy[gyIndex];
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
