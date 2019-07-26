using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace KelpNet
{
    [Serializable]
    public class Linear<T> : CompressibleFunction<T> where T : unmanaged, IComparable<T>
    {
        const string FUNCTION_NAME = "Linear";

        private const string PARAM_NAME = "/*ForwardActivate*/";
        private const string PARAM_VALUE = "gpuYSum = ForwardActivate(gpuYSum);";

        public NdArray<T> Weight;
        public NdArray<T> Bias;

        public readonly bool NoBias;

        public readonly int InputCount;
        public readonly int OutputCount;

        public Linear(int inputCount, int outputCount, bool noBias = false, Array initialW = null, Array initialb = null, CompressibleActivation<T> activation = null, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(FUNCTION_NAME, activation, new[] { new KeyValuePair<string, string>(PARAM_NAME, PARAM_VALUE) }, name, inputNames, outputNames)
        {
            this.OutputCount = outputCount;
            this.InputCount = inputCount;

            this.Weight = new NdArray<T>(outputCount, inputCount);
            this.Weight.Name = this.Name + " Weight";

            this.NoBias = noBias;

            this.Parameters = new NdArray<T>[noBias ? 1 : 2];

            if (initialW == null)
            {
                Initializer<T>.InitWeight(this.Weight);
            }
            else
            {
                this.Weight.Data = Real<T>.GetArray(initialW);
            }

            this.Parameters[0] = this.Weight;

            if (!noBias)
            {
                this.Bias = new NdArray<T>(outputCount);
                this.Bias.Name = this.Name + " Bias";

                if (initialb != null)
                {
                    this.Bias.Data = Real<T>.GetArray(initialb);
                }

                this.Parameters[1] = this.Bias;
            }
        }

        unsafe RealArray<T> GetBiasedValue(int batchCount)
        {
            //T[] y = new T[OutputCount * batchCount];
            IntPtr y = Marshal.AllocCoTaskMem(OutputCount * batchCount * sizeof(T));

            for (int i = 0; i < batchCount; i++)
            {
                Buffer.MemoryCopy((void*)this.Bias.Data.Ptr, (void*)(y + i * this.OutputCount * sizeof(T)), OutputCount * sizeof(T), OutputCount * sizeof(T));
                //Marshal.Copy(this.Bias.Data.Ptr, y, i * this.OutputCount, this.Bias.DataLength);
                //Array.Copy(this.Bias.Data, 0, y, i * this.OutputCount, this.Bias.DataLength);
            }

            return new RealArray<T>(y, OutputCount * batchCount);
        }

        protected override NdArray<T> NeedPreviousForwardCpu(NdArray<T> x)
        {
            RealArray<T> y = this.NoBias ? new T[OutputCount * x.BatchCount] : GetBiasedValue(x.BatchCount);

            for (int batchCount = 0; batchCount < x.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    for (int j = 0; j < this.InputCount; j++)
                    {
                        y[batchCount * this.OutputCount + i] += x.Data[batchCount * this.InputCount + j] * this.Weight.Data[i * this.InputCount + j];
                    }
                }
            }

            if (this.Activator != null)
            {
                for (int i = 0; i < OutputCount * x.BatchCount; i++)
                {
                    y[i] = this.Activator.ForwardActivate(y[i]);
                }
            }

            return NdArray<T>.Convert(y, new[] { OutputCount }, x.BatchCount, this);
        }

        RealArray<T> GetActivatedgy(NdArray<T> y)
        {
            RealArray<T> activatedgY = new T[y.DataLength];

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    int index = batchCount * this.OutputCount + i;
                    activatedgY[index] = this.Activator.BackwardActivate(y.Grad[index], y.Data[index]);
                }
            }

            return activatedgY;
        }

        void CalcBiasGrad(RealArray<T> gy, int batchCount)
        {
            for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    this.Bias.Grad[i] += gy[batchCounter * this.OutputCount + i];
                }
            }
        }

        protected override void NeedPreviousBackwardCpu(NdArray<T> y, NdArray<T> x)
        {
            RealArray<T> activatedgy = this.Activator != null ? GetActivatedgy(y) : y.Grad;
            if (!NoBias) CalcBiasGrad(activatedgy, y.BatchCount);

            for (int batchCount = 0; batchCount < y.BatchCount; batchCount++)
            {
                for (int i = 0; i < this.OutputCount; i++)
                {
                    Real<T> gyData = activatedgy[i + batchCount * this.OutputCount];

                    for (int j = 0; j < this.InputCount; j++)
                    {
                        this.Weight.Grad[i * this.InputCount + j] += x.Data[j + batchCount * this.InputCount] * gyData;
                        x.Grad[j + batchCount * this.InputCount] += this.Weight.Data[i * this.InputCount + j] * gyData;
                    }
                }
            }
        }

        public Convolution2D<T> AsConvolution2D()
        {
            return new Convolution2D<T>(this);
        }
    }
}
