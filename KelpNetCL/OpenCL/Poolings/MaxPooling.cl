__kernel void MaxPoolingForward(
	__global const Real *gpuX,
	__global int *gpuYindex,
	const int outputHeight, const int outputWidth,
	const int inputShape0, const int inputShape1, const int inputShape2,
	const int kHeight, const int kWidth,
	const int strideX, const int strideY,
	const int padY, const int padX)
{
	int b = get_global_id(0) / inputShape0;
	int i = get_global_id(0) % inputShape0;
	int y = get_global_id(1);
	int x = get_global_id(2);

	int indexOffset = b * inputShape0 * inputShape1 * inputShape2 + i * inputShape1 * inputShape2;

	int dyOffset = y * strideY - padY < 0 ? 0 : y * strideY - padY;
	int dyLimit = kHeight + dyOffset < inputShape1 ? kHeight + dyOffset : inputShape1;

	int dxOffset = x * strideX - padX < 0 ? 0 : x * strideX - padX;
	int dxLimit = kWidth + dxOffset < inputShape2 ? kWidth + dxOffset : inputShape2;

	int yIndex = indexOffset + dyOffset * inputShape2 + dxOffset;
	Real maxVal = gpuX[yIndex];

	for (int dy = dyOffset; dy < dyLimit; dy++)
	{
		for (int dx = dxOffset; dx < dxLimit; dx++)
		{
			int inputIndex = indexOffset + dy * inputShape2 + dx;

			if (maxVal < gpuX[inputIndex])
			{
				maxVal = gpuX[inputIndex];
				yIndex = inputIndex;
			}
		}
	}

	gpuYindex[b * inputShape0 * outputHeight * outputWidth + i * outputHeight * outputWidth + y * outputWidth + x] = yIndex;
}