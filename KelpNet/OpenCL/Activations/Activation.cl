__kernel void /*kernelNameBase*/Forward(
    const __global Real *gpuX,
	__global Real *gpuY)
{
	int i = get_global_id(0);

    gpuY[i] = ForwardActivate(gpuX[i]);
}

__kernel void /*kernelNameBase*/Backward(
    const __global Real *gpugY,
    const __global Real *gpuY,
    __global Real *gpugX)
{
	int i = get_global_id(0);
    
    gpugX[i] = BackwardActivate(gpuY[i], gpugY[i]);
}