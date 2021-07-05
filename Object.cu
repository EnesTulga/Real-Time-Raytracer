#include "Objects.h"

__device__ Object::Object() {
	position = float3();
	rotation = float3();
	scale = make_float3(1, 1, 1);
	material = Material();
}
__device__ Object::Object(float3 position, float3 rotation, float3 scale) {
	this->position = position;
	this->rotation = rotation;
	this->scale = scale;
	material = Material();
}
__device__ float3 Object::GetNormal(float3 position) {
	return float3();
}
__device__ float3 Object::GetNormal(MeshTriangle mesh) {
	return float3();
}
__device__ float3 Object::GetRefractionPosition(float3 position, float3 normal, float3 direction) {
	return float3();
}
__device__ float3* Object::returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance) {
	*collided = false;
	return NULL;
}
__device__ float3* Object::returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance, MeshTriangle* collidedMesh) {
	*collided = false;
	return NULL;
}

__device__ void Object::Rotate(float x, float y, float z) {
	rotation.x += x;
	rotation.y += y;
	rotation.z += z;
}