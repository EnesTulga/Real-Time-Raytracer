#include "BoundingBox.h"
#define RAYAABB_EPSILON 0.00001f
__device__ BoundingBox::BoundingBox() {
	this->minBound = make_float3(0,0,0);
	this->maxBound = make_float3(1, 1, 1);
	this->position = make_float3(0,0,0);
}
__device__ BoundingBox::BoundingBox(float3 pos, float3 min, float3 max) {
	this->minBound = min;
	this->maxBound = max;
	this->position = pos;
}

__device__ void BoundingBox::returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance) {
	/*float tmin = (position.x + minBound.x - rayPoint.x) / rayDirection.x;
	float tmax = (position.x + maxBound.x - rayPoint.x) / rayDirection.x;
	float temp;
	if (tmin > tmax) {
		temp = tmin;
		tmin = tmax;
		tmax = temp;
	}

	float tymin = (position.y + minBound.y - rayPoint.y) / rayDirection.y;
	float tymax = (position.y + maxBound.y - rayPoint.y) / rayDirection.y;

	if (tymin > tymax) {
		temp = tymin;
		tymin = tymax;
		tymax = temp;
	}

	if ((tmin > tymax) || (tymin > tmax)) {
		*collided = false;
		return;
	}

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (position.z + minBound.z - rayPoint.z) / rayDirection.z;
	float tzmax = (position.z + maxBound.z - rayPoint.z) / rayDirection.z;

	if (tzmin > tzmax) {
		temp = tzmin;
		tzmin = tzmax;
		tzmax = temp;
	}

	if ((tmin > tzmax) || (tzmin > tmax)) {
		*collided = false;
		return;
	}

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	*collided = true;*/
	bool Inside = true;
	float3 MinB = this->position + this->minBound;
	float3 MaxB = this->position + this->maxBound;
	float3 MaxT;
	MaxT.x = MaxT.y = MaxT.z = -1.0f;
	float3 coord = float3();
	// Find candidate planes.
	if (rayPoint.x < MinB.x)
	{
		coord.x = MinB.x;
		Inside = false;

		// Calculate T distances to candidate planes
		if (rayDirection.x)	MaxT.x = (MinB.x - rayPoint.x) / rayDirection.x;
	}
	else if (rayPoint.x > MaxB.x)
	{
		coord.x = MaxB.x;
		Inside = false;

		// Calculate T distances to candidate planes
		if (rayDirection.x)	MaxT.x = (MaxB.x - rayPoint.x) / rayDirection.x;
	}
	if (rayPoint.y < MinB.y)
	{
		coord.y = MinB.y;
		Inside = false;

		// Calculate T distances to candidate planes
		if (rayDirection.y)	MaxT.y = (MinB.y - rayPoint.y) / rayDirection.y;
	}
	else if (rayPoint.y > MaxB.y)
	{
		coord.y = MaxB.y;
		Inside = false;

		// Calculate T distances to candidate planes
		if (rayDirection.y)	MaxT.y = (MaxB.y - rayPoint.y) / rayDirection.y;
	}
	if (rayPoint.z < MinB.z)
	{
		coord.z = MinB.z;
		Inside = false;

		// Calculate T distances to candidate planes
		if (rayDirection.z)	MaxT.z = (MinB.z - rayPoint.z) / rayDirection.z;
	}
	else if (rayPoint.z > MaxB.z)
	{
		coord.z = MaxB.z;
		Inside = false;

		// Calculate T distances to candidate planes
		if (rayDirection.z)	MaxT.z = (MaxB.z - rayPoint.z) / rayDirection.z;
	}

	// Ray origin inside bounding box
	if (Inside)
	{
		coord = rayPoint;
		*collided = true;
		return;
	}

	// Get largest of the maxT's for final choice of intersection
	float maxTChoose = MaxT.x;
	int maxTIndex = 0;
	if (MaxT.y > MaxT.x) {
		maxTChoose = MaxT.y;
		maxTIndex = 1;
	}	
	if (MaxT.z > maxTChoose) {
		maxTChoose = MaxT.z;
		maxTIndex = 2;
	}	

	// Check final candidate actually inside box
	if ((unsigned int)maxTChoose & 0x80000000) {
		*collided = false;
		return;
	}
	if (maxTIndex != 0) {
		coord.x = rayPoint.x + maxTChoose * rayDirection.x;
#ifdef RAYAABB_EPSILON
		if (coord.x < MinB.x - RAYAABB_EPSILON || coord.x > MaxB.x + RAYAABB_EPSILON) {
			*collided = false;
			return;
		}
#else
		if (coord.x < MinB.x || coord.x > MaxB.x) {
			*collided = false;
			return;
		}
#endif
	}
	if (maxTIndex != 1) {
		coord.y = rayPoint.y + maxTChoose * rayDirection.y;
#ifdef RAYAABB_EPSILON
		if (coord.y < MinB.y - RAYAABB_EPSILON || coord.y > MaxB.y + RAYAABB_EPSILON) {
			*collided = false;
			return;
		}
#else
		if (coord.y < MinB.y || coord.y > MaxB.y) {
			*collided = false;
			return;
		}
#endif
	}
	if (maxTIndex != 2) {
		coord.z = rayPoint.z + maxTChoose * rayDirection.z;
#ifdef RAYAABB_EPSILON
		if (coord.z < MinB.z - RAYAABB_EPSILON || coord.z > MaxB.z + RAYAABB_EPSILON) {
			*collided = false;
			return;
		}
#else
		if (coord.z < MinB.z || coord.z > MaxB.z) {
			*collided = false;
			return;
		}
#endif
	}
	

	*collided = true;
	return;
}