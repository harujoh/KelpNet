__kernel void DropoutForward(
	__global const Real *gpuX,
	__global const Real *mask,
	__global Real *gpuY,
    int maskLength)
{
	int i = get_global_id(0);

    gpuY[i] = gpuX[i] * mask[i % maskLength];
}

__kernel void DropoutBackward(
	__global const Real *mask,
	__global Real *gpugX,
    int gyLength)
{
	int j = get_global_id(0);
	int b = get_global_id(1);

    gpugX[j + b * gyLength] *= mask[j];
}