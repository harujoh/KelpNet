Real ForwardActivate(Real gpuX)
{
	return gpuX < 0.0 ? gpuX * /*slope*/ + 0 : gpuX;
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpuY <= 0.0 ? gpuY * /*slope*/ + 0 : gpugX;
}