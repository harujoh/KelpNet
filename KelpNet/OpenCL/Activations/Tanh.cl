void ForwardActivate(__global Real* gpuY)
{
	*gpuY = tanh(*gpuY);
}

void BackwardActivate(Real gpuY, Real* gpugX)
{
	*gpugX *= 1 - gpuY * gpuY;
}
