Real ForwardActivate(Real gpuX)
{
	return tanh(gpuX);
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpugX * (1 - gpuY * gpuY);
}
