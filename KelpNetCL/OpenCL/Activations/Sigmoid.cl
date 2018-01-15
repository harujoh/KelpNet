Real ForwardActivate(Real gpuY)
{
	return 1 / (1 + exp(-gpuY));
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpugX * gpuY * (1 - gpuY);
}
