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

    int inIndexY = y * strideY - padY;
    int dyLimit = kHeight < inputShape1 - inIndexY ? kHeight : inputShape1 - inIndexY;
    int dyStart = inIndexY < 0 ? -inIndexY : 0;

    int inIndexX = x * strideX - padX;
    int dxLimit = kWidth < inputShape2 - inIndexX ? kWidth : inputShape2 - inIndexX;
    int dxStart = inIndexX < 0 ? -inIndexX : 0;

    int inBaseIndex = b * inputShape0 * inputShape1 * inputShape2 + i * inputShape1 * inputShape2 + inIndexY * inputShape2 + inIndexX;
    int outIndex = b * inputShape0 * outputHeight * outputWidth + i * outputHeight * outputWidth + y * outputWidth + x;

	int yIndex = -1;
    Real maxVal = -INFINITY;

    for (int dy = dyStart; dy < dyLimit; dy++)
    {
        for (int dx = dxStart; dx < dxLimit; dx++)
        {
            int inputIndex = inBaseIndex + dy * inputShape2 + dx;

            if (maxVal < gpuX[inputIndex])
            {
                maxVal = gpuX[inputIndex];
                yIndex = inputIndex;
            }
        }
    }

	gpuYindex[b * inputShape0 * outputHeight * outputWidth + i * outputHeight * outputWidth + y * outputWidth + x] = yIndex;
}