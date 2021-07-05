#pragma once
#include "Material.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <helper_math.h>
class MeshTriangle;
#include "DataStructures.h"

#define Epsilon 1.19209e-07


class Object {
	public:
		__device__ Object();
		__device__ Object(float3 position, float3 rotation, float3 scale);
		__device__ virtual float3 GetNormal(float3 position);
		__device__ virtual float3 GetRefractionPosition(float3 position, float3 normal, float3 direction);
		__device__ virtual float3 GetNormal(MeshTriangle mesh);
		__device__ virtual float3* returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance);
		__device__ virtual float3* returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance, MeshTriangle* collidedMesh);
		__device__ virtual void Rotate(float x, float y, float z);

		float3 position, rotation, scale;
		Material material;
};
class Sphere :public Object {
public:
	__device__ Sphere();
	__device__ Sphere(float3 position, float3 rotation, float3 scale, float radius);
	__device__ virtual float3 GetNormal(float3 position);
	__device__ float3 GetRefractionPosition(float3 position, float3 normal, float3 direction);
	__device__ virtual float3* returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance);

	float radius;
};

class Triangle : public Object{
	public:
		__device__ Triangle();
		__device__ Triangle(float3 position, float3 rotation, float3 scale, float3 v1, float3 v2, float3 v3);
		__device__ virtual float3 GetNormal(float3 position);
		__device__ virtual float3* returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance);
		float3 v1, v2, v3;
};


class Model : public Object {
	public:
		__device__ Model();
		__device__ Model(float3 position, float3 rotation, float3 scale);
		__device__ virtual float3 GetNormal(MeshTriangle mesh);
		__device__ virtual float3* returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance, MeshTriangle* collidedMesh);
		__device__ virtual void Rotate(float x, float y, float z);
		__device__ void checkIntersection(float3 rayPoint, float3 rayDirection, bool* collided, float* distance, int index);

		DynamicMeshArray* meshList;
};
__global__ void RotateModel(Model* model);
__global__ void PiecedReturnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance, MeshTriangle* collidedMeshes,Model* model, int pieceCount);