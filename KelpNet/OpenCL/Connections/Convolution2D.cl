__kernel void Convolution2DForward(
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

	gpuY[index] = /*ForwardActivate*/(localResult);
}

__kernel void Convolution2DgWBackward(
	const __global Real* activatedgy,
	const __global Real* gpuX,
	__global Real* gpugW,
	const int batchCount,
	const int inputCount,
	const int yShape1,
	const int yShape2,
	const int yLength,
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

    int yChOffset = och * yShape1 * yShape2;

    int xChOffset = ich * xShape1 * xShape2;

    int oyStart = 0 > padY - ky ? 0 : (padY - ky + strideY - 1) / strideY * strideY;
    int oyLimit = yShape1 * strideY < xShape1 - ky + padY ? yShape1 * strideY : xShape1 - ky + padY;

    int oxStart = 0 > padX - kx ? 0 : (padX - kx + strideX - 1) / strideX * strideX;
    int oxLimit = yShape2 * strideX < xShape2 - kx + padX ? yShape2 * strideX : xShape2 - kx + padX;

	int wIndex = och * inputCount * kHeight * kWidth + ich * kHeight * kWidth + ky * kWidth + kx;
	Real localgW = gpugW[wIndex];

    for (int batchCounter = 0; batchCounter < batchCount; batchCounter++)
    {
        int yBatchOffset = batchCounter * yLength;
        int xBatchOffset = batchCounter * xLength + xChOffset;

        for (int oy = oyStart; oy < oyLimit; oy += strideY)
        {
            for (int ox = oxStart; ox < oxLimit; ox += strideX)
            {
                int gyIndex = yBatchOffset + yChOffset + oy / strideY * yShape2 + ox / strideX;

                int xIndex = xBatchOffset + (ky + oy - padY) * xShape2 + kx + ox - padX;

				localgW += gpuX[xIndex] * activatedgy[gyIndex];
            }
        }
    }

	gpugW[wIndex] = localgW;
}

__kernel void Convolution2DgXBackward(
	const __global Real* activatedgy,
	const __global Real* gpuW,
	__global Real* gpugX,
	const int inputCount,
	const int yShape0,
	const int yShape1,
	const int yShape2,
	const int yLength,
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
	int iy = get_global_id(1) + padY + 1;
	int ix = get_global_id(2) + padX + 1;

    int yBatchOffset = batchCounter * yLength;

    int wIchOffset = ich * kHeight * kWidth;

    int oyStart = 0 > iy - kHeight ? 0 : (iy - kHeight + strideY - 1) / strideY * strideY;
    int oyLimit = yShape1 * strideY < iy ? yShape1 * strideY : iy;

    int oxStart = 0 > ix - kWidth ? 0 : (ix - kWidth + strideX - 1) / strideX * strideX;
    int oxLimit = yShape2 * strideX < ix ? yShape2 * strideX : ix;

	Real localgX = 0;

    for (int och = 0; och < yShape0; och++)
    {
        int wOchOffset = wIchOffset + och * inputCount * kHeight * kWidth;

        int yChOffset = och * yShape1 * yShape2;

        for (int oy = oyStart; oy < oyLimit; oy += strideY)
        {
            for (int ox = oxStart; ox < oxLimit; ox += strideX)
            {
                int gyIndex = yBatchOffset + yChOffset + oy / strideY * yShape2 + ox / strideX;
                int wIndex = wOchOffset + (iy - oy - 1) * kWidth + ix - ox - 1;

                localgX += gpuW[wIndex] * activatedgy[gyIndex];
	        }
        }
    }

	int resultIndex = batchCounter * xLength + ich * xShape1 * xShape2 + (iy - padY - 1) * xShape2 + ix - padX - 1;
	gpugX[resultIndex] = localgX;
}