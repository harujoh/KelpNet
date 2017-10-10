__kernel void /*kernelNameBase*/Forward(
    const __global read_only Real *gpuX,
	__global write_only Real *gpuY)
{
	int i = get_global_id(0);

    gpuY[i] = ForwardActivate(gpuX[i]);
}

__kernel void /*kernelNameBase*/Backward(
    const __global read_only Real *gpugY,
    const __global read_only Real *gpuY,
    __global write_only Real *gpugX)
{
	int i = get_global_id(0);
    
    gpugX[i] = BackwardActivate(gpuY[i], gpugY[i]);
}