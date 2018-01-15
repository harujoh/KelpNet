__kernel void Convolution2DForward(
	const __global __read_only	Real* gpuX,
	const __global __read_only	Real* gpuW,
	const __global __read_only	Real* gpub,
	__global __write_only Real* gpuY,
	const int inputShape1,
	const int inputShape2,
	const int inputLength,
	const int outputWidth,
	const int outputHeight,
	const int strideX,
	const int strideY,
	const int padX,
	const int padY,
	const int kHeight,
	const int kWidth,
	const int OutputCount,
	const int InputCount)
{
	int batchCounter = get_global_id(0) / OutputCount;
	int och = get_global_id(0) % OutputCount;
	int oy = get_global_id(1) * strideY - padY;
	int ox = get_global_id(2) * strideX - padX;

	Real localResult = 0;

	gpuW += och * InputCount * kHeight* kWidth;
	gpuX += batchCounter * inputLength;

	int kyStartIndex = oy < 0 ? 0 : oy;
	int kyLimit = kHeight + oy < inputShape1 ? kHeight + oy : inputShape1;

	int kxStartIndex = ox < 0 ? 0 : ox;
	int kxLimit = kWidth + ox < inputShape2 ? kWidth + ox : inputShape2;

	for (int ich = 0; ich < InputCount; ich++)
	{
		for (int ky = kyStartIndex; ky < kyLimit; ky++)
		{
			for (int kx = kxStartIndex; kx < kxLimit; kx++)
			{
				int inputIndex = ich * inputShape1 * inputShape2 + ky * inputShape2 + kx;
				int wIndex = ich * kHeight * kWidth + (ky - oy) * kWidth + kx - ox;

				localResult += gpuX[inputIndex] * gpuW[wIndex];
			}
		}
	}

	int index = batchCounter * OutputCount * outputHeight * outputWidth + och * outputHeight * outputWidth + get_global_id(1) * outputWidth + get_global_id(2);

	localResult += gpub[och];

	/*ForwardActivate*/

	gpuY[index] = localResult;
}

__kernel void Convolution2DgWBackward(
	const __global __read_only	Real* activatedgy,
	const __global __read_only	Real* gpuX,
	__global __read_write Real* gpugW,
	const int batchCount,
	const int inputCount,
	const int gyShape0,
	const int gyShape1,
	const int gyShape2,
	const int xShape1,
	const int xShape2,
	const int xLength,
	const int strideX,
	const int strideY,
	const int padX,
	const int padY,
	const int kHeight,
	const int kWidth)
{
	int och = get_global_id(0) / inputCount;
	int ich = get_global_id(0) % inputCount;
	int ky = get_global_id(1);
	int kx = get_global_id(2);

	int outChOffset = och * inputCount * kHeight * kWidth;
	int gychOffset = och * gyShape1 * gyShape2;

	int iyStartIndex = ky - padY < 0 ? 0 : ky - padY;
	int iyLimit = gyShape1 * strideY + ky - padY < xShape1 ? gyShape1 * strideY + ky - padY : xShape1;

	int ixStartIndex = kx - padX < 0 ? 0 : kx - padX;
	int ixLimit = gyShape2 * strideX + kx - padX < xShape2 ? gyShape2 * strideX + kx - padX : xShape2;

	Real localgW = gpugW[outChOffset + ich * kHeight * kWidth + ky * kWidth + kx];

	for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
	{
		int gpuXIndex = batchCounter * xLength + ich * xShape1 * xShape2;
		int gyIndexOffset = batchCounter * gyShape0 * gyShape1 * gyShape2;

		for (int iy = iyStartIndex; iy < iyLimit; iy += strideY)
		{
			int oy = iy - ky + padY;

			for (int ix = ixStartIndex; ix < ixLimit; ix += strideX)
			{
				int ox = ix - kx + padX;

				int gyIndex = gyIndexOffset + gychOffset + oy * gyShape2 + ox;
				int inputIndex = gpuXIndex + iy * xShape2 + ix;

				localgW += gpuX[inputIndex] * activatedgy[gyIndex];
			}
		}
	}

	gpugW[outChOffset + ich * kHeight * kWidth + ky * kWidth + kx] = localgW;
}

__kernel void Convolution2DgXBackward(
	const __global __read_only	Real* activatedgy,
	const __global __read_only	Real* gpuW,
	__global __write_only Real* gpugX,
	const int outputCount,
	const int inputCount,
	const int gyShape0,
	const int gyShape1,
	const int gyShape2,
	const int xShape1,
	const int xShape2,
	const int xLength,
	const int strideX,
	const int strideY,
	const int padX,
	const int padY,
	const int kHeight,
	const int kWidth)
{
	int batchCounter = get_global_id(0) / inputCount;
	int ich = get_global_id(0) % inputCount;
	int iy = get_global_id(1) + padY;
	int ix = get_global_id(2) + padX;

	int kyStart = 0 <= iy - gyShape1 * strideY ? iy - gyShape1 * strideY + 1 : 0;
	int kyLimit = kHeight < iy + 1 ? kHeight : iy + 1;

	int kxStart = 0 <= ix - gyShape2 * strideX ? ix - gyShape2 * strideX + 1 : 0;
	int kxLimit = kWidth < ix + 1 ? kWidth : ix + 1;

	Real localgX = 0;

	for (int och = 0; och < outputCount; och++)
	{
		int gyIndexOffset = batchCounter * gyShape0 * gyShape1 * gyShape2 + och * gyShape1 * gyShape2;
		int wIndexOffset = ich * kHeight * kWidth + och * inputCount * kHeight * kWidth;

		for (int ky = kyStart; ky < kyLimit; ky++)
		{
			int kydiv = (iy - ky) / strideY;

			for (int kx = kxStart; kx < kxLimit; kx++)
			{
				int kxdiv = (ix - kx) / strideX;

				int gyIndex = gyIndexOffset + kydiv * gyShape2 + kxdiv;
				int wIndex = wIndexOffset + ky * kWidth + kx;

				localgX += gpuW[wIndex] * activatedgy[gyIndex];
			}
		}
	}

	gpugX[batchCounter * xLength + ich * xShape1 * xShape2 + (iy - padY) * xShape2 + (ix - padX)] = localgX;
}