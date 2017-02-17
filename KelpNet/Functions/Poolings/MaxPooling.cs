using System;
using System.Drawing;
using System.Linq;
using Cloo;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Poolings
{
    [Serializable]
    public class MaxPooling : NeedPreviousDataFunction
    {
        private int _kWidth;
        private int _kHeight;
        private int _padX;
        private int _padY;
        private int _stride;

        public MaxPooling(int ksize, int stride = 1, int pad = 0, string name = "MaxPooling") : base(name)
        {
            this._kHeight = ksize;
            this._kWidth = ksize;
            this._padY = pad;
            this._padX = pad;
            this._stride = stride;

            //カーネルを作成
            if (IsGpu)
            {
                initGPU();
            }
        }

        public MaxPooling(Size ksize, int stride = 1, Size pad = new Size(), string name = "MaxPooling") : base(name)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            this._kHeight = ksize.Height;
            this._kWidth = ksize.Width;
            this._padY = pad.Height;
            this._padX = pad.Width;
            this._stride = stride;

            //カーネルを作成
            if (IsGpu)
            {
                initGPU();
            }
        }

        void initGPU()
        {
            ForwardKernel = Weaver.CreateKernel(ForwardKernelSource, ForwardKernelName);
            //BackwardKernel = Weaver.CreateKernel("", "");
        }

        const string ForwardKernelName = "MaxPoolingForward";
        const string ForwardKernelSource =
@"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void MaxPoolingForward(
	__global double *gpuX,
	__global double *gpuY,
    int outputHeight, int outputWidth,
    int kHeight, int kWidth,
    int stride,
    int padY,int padX,
    int inputShape0, int inputShape1, int inputShape2)
{
	int b = get_global_id(0);

    int resultIndex = b * inputShape0 * outputHeight * outputWidth;
    int inputLength = inputShape0 * inputShape1 * inputShape2;

    for (int i = 0; i < inputShape0; i++)
    {
        int inputIndexOffset = i * inputShape1 * inputShape2;

        for (int y = 0; y < outputHeight; y++)
        {
            for (int x = 0; x < outputWidth; x++)
            {
                for (int dy = 0; dy < kHeight; dy++)
                {
                    int inputIndexY = y * stride + dy - padY;

                    if (inputIndexY >= 0 && inputIndexY < inputShape1)
                    {
                        for (int dx = 0; dx < kWidth; dx++)
                        {
                            int inputIndexX = x * stride + dx - padX;

                            if (inputIndexX >= 0 && inputIndexX < inputShape2)
                            {
                                int inputIndex = inputIndexOffset + inputIndexY * inputShape2 + inputIndexX + b * inputLength;

                                if(gpuY[resultIndex] < gpuX[inputIndex])
                                {
                                    gpuY[resultIndex] = gpuX[inputIndex];
                                }
                            }
                        }
                    }
                }

                resultIndex++;
            }
        }
    }
}";

        protected override BatchArray NeedPreviousForward(BatchArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - this._kHeight + this._padY * 2.0) / this._stride) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - this._kWidth + this._padX * 2.0) / this._stride) + 1;
            double[] result = Enumerable.Repeat(double.MinValue, input.Shape[0] * outputHeight * outputWidth  * input.BatchCount).ToArray();

            if (!IsGpu)
            {
                for (int b = 0; b < input.BatchCount; b++)
                {
                    int resultIndex = b * input.Shape[0] * outputHeight * outputWidth;

                    for (int i = 0; i < input.Shape[0]; i++)
                    {
                        int inputIndexOffset = i * input.Shape[1] * input.Shape[2];

                        for (int y = 0; y < outputHeight; y++)
                        {
                            for (int x = 0; x < outputWidth; x++)
                            {
                                for (int dy = 0; dy < this._kHeight; dy++)
                                {
                                    int inputIndexY = y * this._stride + dy - this._padY;

                                    if (inputIndexY >= 0 && inputIndexY < input.Shape[1])
                                    {
                                        for (int dx = 0; dx < this._kWidth; dx++)
                                        {
                                            int inputIndexX = x * this._stride + dx - this._padX;

                                            if (inputIndexX >= 0 && inputIndexX < input.Shape[2])
                                            {
                                                int inputIndex = inputIndexOffset + inputIndexY * input.Shape[2] + inputIndexX + b * input.Length;
                                                result[resultIndex] = Math.Max(result[resultIndex], input.Data[inputIndex]);
                                            }
                                        }
                                    }
                                }

                                resultIndex++;
                            }
                        }
                    }
                }
            }
            else
            {
                ComputeBuffer<double> gpuX = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, input.Data);
                ComputeBuffer<double> gpuY = new ComputeBuffer<double>(Weaver.Context, ComputeMemoryFlags.ReadWrite | ComputeMemoryFlags.CopyHostPointer, result);

                ForwardKernel.SetMemoryArgument(0, gpuX);
                ForwardKernel.SetMemoryArgument(1, gpuY);
                ForwardKernel.SetValueArgument(2, outputHeight);
                ForwardKernel.SetValueArgument(3, outputWidth);
                ForwardKernel.SetValueArgument(4, this._kHeight);
                ForwardKernel.SetValueArgument(5, this._kWidth);
                ForwardKernel.SetValueArgument(6, this._stride);
                ForwardKernel.SetValueArgument(7, this._padY);
                ForwardKernel.SetValueArgument(8, this._padX);
                ForwardKernel.SetValueArgument(9, input.Shape[0]);
                ForwardKernel.SetValueArgument(10, input.Shape[1]);
                ForwardKernel.SetValueArgument(11, input.Shape[2]);

                Weaver.CommandQueue.Execute
                (
                    ForwardKernel,
                    null,
                    new long[] { input.BatchCount },
                    null,
                    null
                );

                Weaver.CommandQueue.ReadFromBuffer(gpuY, ref result, true, null);

                gpuX.Dispose();
                gpuY.Dispose();
            }

            return BatchArray.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount);
        }

        protected override BatchArray NeedPreviousBackward(BatchArray gy, BatchArray prevInput, BatchArray prevOutput)
        {
            double[] result = new double[prevInput.Data.Length];

            for (int b = 0; b < gy.BatchCount; b++)
            {
                int index = b * gy.Length;

                for (int i = 0; i < prevInput.Shape[0]; i++)
                {
                    int prevInputIndexOffset = i * prevInput.Shape[1] * prevInput.Shape[2];

                    for (int y = 0; y < prevOutput.Shape[1]; y++)
                    {
                        for (int x = 0; x < prevOutput.Shape[2]; x++)
                        {
                            //前回の入力値と出力値を比較して、同じ値のものを見つける
                            this.SetResult(prevInputIndexOffset, y, x, gy.Data[index], prevInput, prevOutput.Data[index], b, ref result);
                            index++;
                        }
                    }
                }
            }

            return BatchArray.Convert(result, prevInput.Shape, prevInput.BatchCount);
        }

        //同じ値を複数持つ場合、左上優先にして処理を打ち切る
        //他のライブラリの実装では乱数を取って同じ値の中からどれかを選ぶ物が多い
        void SetResult(int prevInputIndexOffset, int y, int x, double data, BatchArray prevInput, double prevOutputData, int b, ref double[] result)
        {
            for (int dy = 0; dy < this._kHeight; dy++)
            {
                int outputIndexY = y * this._stride + dy - this._padY;

                if (outputIndexY >= 0 && outputIndexY < prevInput.Shape[1])
                {
                    for (int dx = 0; dx < this._kWidth; dx++)
                    {
                        int outputIndexX = x * this._stride + dx - this._padX;

                        if (outputIndexX >= 0 && outputIndexX < prevInput.Shape[2])
                        {
                            int prevInputIndex = prevInputIndexOffset + outputIndexY * prevInput.Shape[2] + outputIndexX + b * prevInput.Length;

                            if (prevInput.Data[prevInputIndex].Equals(prevOutputData))
                            {
                                result[prevInputIndex] = data;
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}
