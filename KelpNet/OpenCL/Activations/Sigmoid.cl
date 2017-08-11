void ForwardActivate(__global Real* gpuY)
{
	*gpuY = 1 / (1 + exp(-*gpuY));
}

void BackwardActivate(Real gpuY, Real* gpugX)
{
	*gpugX *= gpuY * (1 - gpuY);
}
