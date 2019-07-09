Real ForwardActivate(Real gpuX)
{
	return tanh(gpuX * 0.5) * 0.5 + 0.5;
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpugX * gpuY * (1 - gpuY);
}
