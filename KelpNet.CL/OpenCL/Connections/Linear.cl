__kernel void LinearForward(
	__global const Real *gpuX,
	__global const Real *gpuW,
	__global	   Real *gpuY,
	const int OutputCount,
	const int InputCount)
{
	int i = get_global_id(0);
	int batchCount = get_global_id(1);

	gpuX += batchCount * InputCount;
	gpuW += i * InputCount;
	gpuY += i + batchCount * OutputCount;

	Real gpuYSum = *gpuY;

	for (int j = 0; j < InputCount; j++)
	{
		gpuYSum += gpuX[j] * gpuW[j];
	}
	
	*gpuY = /*ForwardActivate*/(gpuYSum);
}

__kernel void LineargWBackward(
	__global const Real *gpugY,
	__global const Real *gpuX,
	__global	   Real *gpugW,
	const int BatchCount,
	const int OutputCount,
	const int InputCount)
{
	int j = get_global_id(0);
	int i = get_global_id(1);

	gpugY += i;
	gpugW += i * InputCount + j;
	gpuX += j;

	Real tmpgW = *gpugW;

	for (int b = 0; b < BatchCount; b++)
	{
		tmpgW += gpuX[b * InputCount] * gpugY[b * OutputCount];
	}

	*gpugW = tmpgW;
}

__kernel void LineargXBackward(
	__global const Real *gpugY,
	__global const Real *gpuW,
	__global	   Real *gpugX,
	const int BatchCount,
	const int OutputCount,
	const int InputCount)
{
	int j = get_global_id(0);
	int b = get_global_id(1);

	gpuW += j;
	gpugX += b * InputCount + j;
	gpugY += b * OutputCount;

	Real tmpgX = 0;

	for (int i = 0; i < OutputCount; i++)
	{
		tmpgX += gpuW[i * InputCount] * gpugY[i];
	}

	*gpugX = tmpgX;
}
