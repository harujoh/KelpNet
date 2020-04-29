Real ForwardActivate(Real gpuX)
{
	return gpuX < 0.0 ? 0.0 : gpuX;
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpuY <= 0.0 ? 0.0 : gpugX;
}
