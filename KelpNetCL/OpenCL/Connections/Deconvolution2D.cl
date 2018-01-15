__kernel void Deconvolution2DForward(
	const __global __read_only	Real* gpuX,
	const __global __read_only	Real* gpuW,
	const __global __read_only	Real* gpub,
	__global __write_only Real* gpuY,
	const int inputShape1,
	const int inputShape2,
	const int inputLength,
	const int outputWidth,
	const int outputHeight,
	const int subSampleX,
	const int subSampleY,
	const int trimX,
	const int trimY,
	const int kHeight,
	const int kWidth,
	const int OutputCount,
	const int InputCount)
{
	int batchCounter = get_global_id(0) / OutputCount;
	int och = get_global_id(0) % OutputCount;
	int oy = get_global_id(1) + trimY;
	int ox = get_global_id(2) + trimX;

	int iyLimit = oy / subSampleY + 1 < inputShape1 ? oy / subSampleY + 1 : inputShape1;
	int iyStart = oy - kHeight < 0 ? 0 : (oy - kHeight) / subSampleY + 1;

	int ixLimit = ox / subSampleX + 1 < inputShape2 ? ox / subSampleX + 1 : inputShape2;
	int ixStart = ox - kWidth < 0 ? 0 : (ox - kWidth) / subSampleX + 1;

	Real result = 0;

	for (int ich = 0; ich < InputCount; ich++)
	{
		int inputIndexOffset = batchCounter * inputLength + ich * inputShape1 * inputShape2;
		int kernelIndexOffset = och * InputCount * kHeight * kWidth + ich * kHeight * kWidth;

		for (int iy = iyStart; iy < iyLimit; iy++)
		{
			for (int ix = ixStart; ix < ixLimit; ix++)
			{
				int inputIndex = inputIndexOffset + iy * inputShape2 + ix;
				int kernelIndex = kernelIndexOffset + (oy - iy * subSampleY) * kWidth + (ox - ix * subSampleX);

				result += gpuX[inputIndex] * gpuW[kernelIndex];
			}
		}
	}

	int outputIndex = batchCounter * OutputCount * outputWidth * outputHeight + och * outputWidth * outputHeight + (oy - trimY) * outputWidth + ox - trimX;
	result += gpub[och];

	/*ForwardActivate*/

	gpuY[outputIndex] = result;
}

__kernel void Deconvolution2DgWBackward(
	const __global __read_only	Real* activatedgy,
	const __global __read_only	Real* gpuX,
	__global __read_write Real* gpugW,
	const int batchCounter,
	const int inputCount,
	const int gyLength,
	const int gyShape1,
	const int gyShape2,
	const int xShape1,
	const int xShape2,
	const int xLength,
	const int subSampleX,
	const int subSampleY,
	const int trimX,
	const int trimY,
	const int kHeight,
	const int kWidth)
{
	int och = get_global_id(0) / inputCount;
	int ich = get_global_id(0) % inputCount;
	int ky = get_global_id(1);
	int kx = get_global_id(2);

	int gyChannelOffest = och * gyShape1 * gyShape2;

	int xChannelOffest = ich * xShape1 * xShape2;

	int iyOffset = ky - trimY;
	int iyStart = iyOffset < 0 ? 0 : iyOffset;
	int iyLimit = gyShape1 < xShape1 * subSampleY + iyOffset ? gyShape1 : xShape1 * subSampleY + iyOffset;

	int ixOffset = kx - trimX;
	int ixStart = ixOffset < 0 ? 0 : ixOffset;
	int ixLimit = gyShape2 < xShape2 * subSampleX + ixOffset ? gyShape2 : xShape2 * subSampleX + ixOffset;

	int gwIndex = och * inputCount * kHeight * kWidth + ich * kHeight * kWidth + ky * kWidth + kx;

	Real localgW = gpugW[gwIndex];

	for (int batchCount = 0; batchCount < batchCounter; batchCount++)
	{
		int xIndexOffset = batchCount * xLength + xChannelOffest;
		int gyIndexOffset = batchCount * gyLength + gyChannelOffest;

		for (int iy = iyStart; iy < iyLimit; iy += subSampleY)
		{
			for (int ix = ixStart; ix < ixLimit; ix += subSampleX)
			{
				int gyIndex = gyIndexOffset + iy * gyShape2 + ix;
				int xIndex = xIndexOffset + (iy / subSampleY - iyOffset) * xShape2 + ix / subSampleX - ixOffset;

				localgW += gpuX[xIndex] * activatedgy[gyIndex];
			}
		}
	}

	gpugW[gwIndex] = localgW;
}

__kernel void Deconvolution2DgXBackward(
	const __global __read_only	Real* activatedgy,
	const __global __read_only	Real* gpuW,
	__global __write_only Real* gpugX,
	const int outputCount,
	const int inputCount,
	const int gyLength,
	const int gyShape1,
	const int gyShape2,
	const int xShape1,
	const int xShape2,
	const int xLength,
	const int subSampleX,
	const int subSampleY,
	const int trimX,
	const int trimY,
	const int kHeight,
	const int kWidth)
{
	int batchCounter = get_global_id(0) / inputCount;
	int ich = get_global_id(0) % inputCount;
	int iy = get_global_id(1) * subSampleY;
	int ix = get_global_id(2) * subSampleX;

	int kyOffset = iy - trimY;
	int kyStart = kyOffset < 0 ? 0 : kyOffset;
	int kyLimit = gyShape1 < kHeight + kyOffset ? gyShape1 : kHeight + kyOffset;

	int kxOffset = ix - trimX;
	int kxStart = kxOffset < 0 ? 0 : kxOffset;
	int kxLimit = gyShape2 < kWidth + kxOffset ? gyShape2 : kWidth + kxOffset;

	Real localgX = 0;

	for (int och = 0; och < outputCount; och++)
	{
		int gyIndexOffset = batchCounter * gyLength + och * gyShape1 * gyShape2;
		int wIndexOffset = ich * kHeight * kWidth + och * inputCount * kHeight * kWidth;

		for (int ky = kyStart; ky < kyLimit; ky++)
		{
			for (int kx = kxStart; kx < kxLimit; kx++)
			{
				int gyIndex = gyIndexOffset + ky * gyShape2 + kx;
				int wIndex = wIndexOffset + (ky - kyOffset) * kWidth + kx - kxOffset;

				localgX += gpuW[wIndex] * activatedgy[gyIndex];
			}
		}
	}

	int gxIndex = batchCounter * xLength + ich * xShape1 * xShape2 + iy * xShape2 + ix;
	gpugX[gxIndex] = localgX;
}