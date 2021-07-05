#pragma once
#include "Objects.h"

class BoundingBox {
	public:
		__device__ BoundingBox();
		__device__ BoundingBox(float3 pos, float3 min, float3 max);
		__device__ void returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance);

		Object** objectsInsideOfTheBox;
		float3 position;
		float3 minBound, maxBound;
};