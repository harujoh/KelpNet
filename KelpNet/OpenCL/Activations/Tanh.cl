Real ForwardActivate(Real gpuY)
{
	return tanh(gpuY);
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpugX * (1 - gpuY * gpuY);
}
