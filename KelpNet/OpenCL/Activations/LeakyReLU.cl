Real ForwardActivate(Real gpuY)
{
	return gpuY < 0.0 ? gpuY * /*slope*/ + 0 : gpuY;
}

Real BackwardActivate(Real gpuY, Real gpugX)
{
	return gpuY <= 0.0 ? gpuY * /*slope*/ + 0 : gpugX;
}