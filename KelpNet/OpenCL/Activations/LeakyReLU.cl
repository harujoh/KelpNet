void ForwardActivate(__global Real* gpuY)
{
	if (*gpuY < 0.0)
	{
		*gpuY *= /*slope*/ + 0;
	}
}

void BackwardActivate(Real gpuY, Real* gpugX)
{
	if (gpuY <= 0.0)
	{
		*gpugX = gpuY * /*slope*/ + 0;
	}
}