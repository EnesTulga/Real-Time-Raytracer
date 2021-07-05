#include "Objects.h"

__device__ Sphere::Sphere()
	: Object()
{
	radius = 1;
}
__device__ Sphere::Sphere(float3 position, float3 rotation, float3 scale, float radius)
	: Object(position, rotation, scale)
{
	this->radius = radius;
}
__device__ float3 Sphere::GetNormal(float3 position) {
	return normalize(position - (this->position));
}
__device__ float3* Sphere::returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance) {
	float3 diffToRay = rayPoint - position;
	float t1 = diffToRay.x * diffToRay.x + diffToRay.y * diffToRay.y + diffToRay.z * diffToRay.z - radius * radius;
	float t2 = 2 * (rayDirection.x * diffToRay.x + rayDirection.y * diffToRay.y + rayDirection.z * diffToRay.z);
	float t3 = rayDirection.x * rayDirection.x + rayDirection.y * rayDirection.y + rayDirection.z * rayDirection.z;
	float d = t2 * t2 - (4.0 * t3 * t1);
	float t = 0;
	float tt = 0;
	if (d < 0)
	{
		*collided = false;
		return NULL;
	}
	else if (d == 0)
	{
		t = (-t2) / (2 * t3);
		if (t <= 0)
		{
			*collided = false;
			return NULL;
		}
		else {
			*collided = true;
			*distance = t;
			return &(rayPoint + t * rayDirection);
		}
	}
	else {
		t = (-t2 + sqrt(d)) / (2 * t3);
		tt = (-t2 - sqrt(d)) / (2 * t3);
		if (t > 0 && (tt <= 0 || t < tt)) {
			*collided = true;
			*distance = t;
			return &(rayPoint + t * rayDirection);
		}
		else if (tt > 0 && (t <= 0 || tt < t)) {
			*collided = true;
			*distance = tt;
			return &(rayPoint + tt * rayDirection);
		}
	}
	*collided = false;
	return NULL;
}