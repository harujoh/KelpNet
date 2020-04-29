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
	const int yLength,
	const int yShape0,
	const int yShape1,
	const int yShape2,
	const int xLength,
	const int xShape0,
	const int xShape1,
	const int xShape2,
	const int strideX,
	const int strideY,
	const int padX,
	const int padY,
	const int kWidth,
	const int kHeight,
	const int kxStartPrevOffset,
	const int kyStartPrevOffset)
{
	int batchCounter = get_global_id(0) / xShape0;
	int ich = get_global_id(0) % xShape0;
	int iy = get_global_id(1);
	int ix = get_global_id(2);

    int yBatchOffset = batchCounter * yLength;

    int wIchOffset = ich * kHeight * kWidth;

	int kyStartOffset = (kyStartPrevOffset + iy) / strideY * strideY > 0 ? (kyStartPrevOffset + iy) / strideY * strideY : 0;
    int kyStart = (iy + padY) % strideY + kyStartOffset;
    int kyStop = iy + padY + 1 < kHeight ? iy + padY + 1 : kHeight;

	int kxStartOffset = (kxStartPrevOffset + ix) / strideX * strideX > 0 ? (kxStartPrevOffset + ix) / strideX * strideX : 0;
    int kxStart = (ix + padX) % strideX + kxStartOffset;
    int kxStop = ix + padX + 1 < kWidth ? ix + padX + 1 : kWidth;

	Real localgX = 0;

    for (int och = 0; och < yShape0; och++)
    {
        int wOchOffset = wIchOffset + och * xShape0 * kHeight * kWidth;
        int yChOffset = yBatchOffset + och * yShape1 * yShape2;

        for (int ky = kyStart; ky < kyStop; ky += strideY)
        {
            for (int kx = kxStart; kx < kxStop; kx += strideX)
            {
			    int oy = (iy + padY - ky) / strideY;
                int ox = (ix + padX - kx) / strideX;

                int wIndex = wOchOffset + ky * kWidth + kx;
                int gyIndex = yChOffset + oy * yShape2 + ox;

                localgX += gpuW[wIndex] * activatedgy[gyIndex];
	        }
        }
    }

    int xIndex = batchCounter * xLength + ich * xShape1 * xShape2 + iy * xShape2 + ix;
	gpugX[xIndex] = localgX;
}