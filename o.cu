#include "Objects.h"

__device__ Object::Object() {
	position = float3();
	rotation = float3();
	scale = make_float3(1, 1, 1);
}
__device__ Object::Object(float3 position, float3 rotation, float3 scale) {
	this->position = position;
	this->rotation = rotation;
	this->scale = scale;
}
__device__ float3 Object::GetNormal(float3 position) {
	return float3();
}
__device__ float3* Object::returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance) {
	*collided = false;
	return NULL;
}