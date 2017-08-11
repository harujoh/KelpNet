void ForwardActivate(__global Real* gpuY)
{
	if (*gpuY < 0.0)
	{
		*gpuY = 0.0;
	}
}

void BackwardActivate(Real gpuY, Real* gpugX)
{
	if (gpuY <= 0.0)
	{
		*gpugX = 0.0;
	}
}
