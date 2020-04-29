__kernel void Deconvolution2DForward(
	const __global Real* gpuX,
	const __global Real* gpuW,
	const __global Real* gpub,
	__global Real* gpuY,
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
	const int outputCount,
	const int inputCount)
{
	int batchCounter = get_global_id(0) / outputCount;
	int och = get_global_id(0) % outputCount;
	int oy = get_global_id(1) + padY;
	int ox = get_global_id(2) + padX;

	int iyLimit = oy / strideY + 1 < inputShape1 ? oy / strideY + 1 : inputShape1;
	int iyStart = oy - kHeight < 0 ? 0 : (oy - kHeight) / strideY + 1;

	int ixLimit = ox / strideX + 1 < inputShape2 ? ox / strideX + 1 : inputShape2;
	int ixStart = ox - kWidth < 0 ? 0 : (ox - kWidth) / strideX + 1;

	Real result = 0;

	for (int ich = 0; ich < inputCount; ich++)
	{
		int inputIndexOffset = batchCounter * inputLength + ich * inputShape1 * inputShape2;
		int kernelIndexOffset = ich * outputCount * kHeight * kWidth + och * kHeight * kWidth;

		for (int iy = iyStart; iy < iyLimit; iy++)
		{
			for (int ix = ixStart; ix < ixLimit; ix++)
			{
				int inputIndex = inputIndexOffset + iy * inputShape2 + ix;
				int kernelIndex = kernelIndexOffset + (oy - iy * strideY) * kWidth + (ox - ix * strideX);

				result += gpuX[inputIndex] * gpuW[kernelIndex];
			}
		}
	}

	int outputIndex = batchCounter * outputCount * outputWidth * outputHeight + och * outputWidth * outputHeight + (oy - padY) * outputWidth + ox - padX;
	gpuY[outputIndex] = /*ForwardActivate*/(result + gpub[och]);
}

__kernel void Deconvolution2DgWBackward(
	const __global Real* activatedgy,
	const __global Real* gpuX,
	__global Real* gpugW,
	const int yBatchCount,
	const int outputCount,
	const int yLength,
	const int yShape1,
	const int yShape2,
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
	int ich = get_global_id(0) / outputCount;
	int och = get_global_id(0) % outputCount;
	int ky = get_global_id(1);
	int kx = get_global_id(2);

	int gwIndex = ich * outputCount * kHeight * kWidth + och * kHeight * kWidth + ky * kWidth + kx;
	Real localgW = gpugW[gwIndex];

    int xChOffset = ich * xShape1 * xShape2;
    int yChOffset = och * yShape1 * yShape2;

    int iyStart = 0 > padY - ky ? 0 : (padY - ky + strideY - 1) / strideY * strideY;
    int iyLimit = xShape1 * strideY < yShape1 + padY - ky ? xShape1 * strideY : yShape1 + padY - ky;

    int ixStart = 0 > padX - kx ? 0 : (padX - kx + strideX - 1) / strideX * strideX;
    int ixLimit = xShape2 * strideX < yShape2 + padX - kx ? xShape2 * strideX : yShape2 + padX - kx;

    for (int batchCounter = 0; batchCounter < yBatchCount; batchCounter++)
    {
		for (int iy = iyStart; iy < iyLimit; iy += strideY)
		{
			for (int ix = ixStart; ix < ixLimit; ix += strideX)
			{
				int outputIndex = batchCounter * yLength + yChOffset + (ky + iy - padY) * yShape2 + (kx + ix - padX);
				int resultIndex = batchCounter * xLength + xChOffset + iy / strideY * xShape2 + ix / strideX;

				localgW += gpuX[resultIndex] * activatedgy[outputIndex];
            }
        }
    }

	gpugW[gwIndex] = localgW;
}

__kernel void Deconvolution2DgXBackward(
	const __global Real* activatedgy,
	const __global Real* gpuW,
	__global Real* gpugX,
	const int outputCount,
	const int inputCount,
	const int yLength,
	const int yShape1,
	const int yShape2,
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
	int iy = get_global_id(1) * strideY;
	int ix = get_global_id(2) * strideX;

	Real localgX = 0;

    int inChOffset = ich * outputCount * kHeight * kWidth;

    int kyStartIndex = iy - padY < 0 ? 0 : iy - padY;
    int kyLimit = kHeight + iy - padY < yShape1 ? kHeight + iy - padY : yShape1;

    int kxStartIndex = ix - padX < 0 ? 0 : ix - padX;
    int kxLimit = kWidth + ix - padX < yShape2 ? kWidth + ix - padX : yShape2;

    for (int och = 0; och < outputCount; och++)
    {
        int outChOffset = och * kHeight * kWidth;
        int outputOffset = och * yShape1 * yShape2;

        for (int ky = kyStartIndex; ky < kyLimit; ky++)
        {
            for (int kx = kxStartIndex; kx < kxLimit; kx++)
            {
                int wIndex = inChOffset + outChOffset + (ky - iy + padY) * kWidth + kx - ix + padX;
                int outputIndex = batchCounter * yLength + outputOffset + ky * yShape2 + kx;

				localgX += gpuW[wIndex] * activatedgy[outputIndex];
            }
        }
    }

	int gxIndex = batchCounter * xLength + ich * xShape1 * xShape2 + iy/strideY * xShape2 + ix/strideX;
	gpugX[gxIndex] = localgX;
}