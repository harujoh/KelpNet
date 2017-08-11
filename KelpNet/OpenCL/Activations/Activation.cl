__kernel void /*kernelNameBase*/Forward(__global Real *gpuY)
{
	int i = get_global_id(0);

    ForwardActivate(gpuY + i);
}

__kernel void /*kernelNameBase*/Backward(__global read_only Real *gpuY,
           __global Real *gpugX)
{
	int i = get_global_id(0);

    Real gpugY = gpugX[i];
    BackwardActivate(gpuY[i], &gpugY);
    
    gpugX[i] = gpugY;
}