#include "Objects.h"

__device__ MeshTriangle::MeshTriangle()
{
	baseV1 = float3();
	v1 = float3();
	baseV2 = float3();
	v2 = float3();
	baseV3 = float3();
	v3 = float3();
}
__device__ MeshTriangle::MeshTriangle(float3 v1, float3 v2, float3 v3)
{
	this->baseV1 = v1;
	this->v1 = v1;
	this->baseV2 = v2;
	this->v2 = v2;
	this->baseV3 = v3;
	this->v3 = v3;
}
__device__ float3 MeshTriangle::GetNormal() {
	return normalize(cross(v2 - v1, v3 - v1));
}
__device__ float3* MeshTriangle::returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance, float3 position) {
	float3 v1v2 = v2 - v1;
	float3 v1v3 = v3 - v1;

	float3 normal = this->GetNormal();
	float3 q = cross(rayDirection, v1v3);
	float a = dot(v1v2, q);

	// IF CHECKING FOR BACKFACING:
	/*
	if (dotProduct(normal, rayDirection) >= 0) {
		return Vector4(0,0,0,-1);
	}*/
	// If nearly parallel
	if (abs(a) <= Epsilon) {
		return NULL;
	}

	float3 s = (rayPoint - v1 - position) / a;
	float3 r = cross(s, v1v2);

	float b1 = dot(s, q);
	float b2 = dot(r, rayDirection);
	float b3 = 1.0f - b1 - b2;

	// Not Intersectig : Intersecting with outside of triangle:
	if (b1 < 0 || b2 < 0 || b3 < 0) {
		return NULL;
	}

	// If ray intersecs with positive direction, not negative.
	float t = dot(v1v3, r);
	if (t > 0) {
		*distance = t;
		*collided = true;
		float3 hitPoint = float3(rayPoint + t * rayDirection);
		return &hitPoint;
	}


	return NULL;
}