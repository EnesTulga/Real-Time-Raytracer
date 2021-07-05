#include "Objects.h"

__device__ Model::Model()
	: Object()
{
	meshList = new DynamicMeshArray();
}
__device__ Model::Model(float3 position, float3 rotation, float3 scale)
	: Object(position, rotation, scale)
{
	meshList = new DynamicMeshArray();
}
__device__ float3 Model::GetNormal(MeshTriangle mesh) {
	return mesh.GetNormal();
}
__device__ void Model::checkIntersection(float3 rayPoint, float3 rayDirection, bool* collided, float* distance, int index) {
	
	MeshTriangle* currentElement = &(meshList->meshArray[index]);

	float3 v1v2 = currentElement->v2 - currentElement->v1;
	float3 v1v3 = currentElement->v3 - currentElement->v1;

	float3 q = cross(rayDirection, v1v3);
	float a = dot(v1v2, q);


	if (abs(a) <= Epsilon) {
		return;
	}

	float3 s = (rayPoint - currentElement->v1 - position) / a;
	float3 r = cross(s, v1v2);

	float b1 = dot(s, q);
	float b2 = dot(r, rayDirection);
	float b3 = 1.0f - b1 - b2;


	if (b1 < 0 || b2 < 0 || b3 < 0) {
		return;
	}

	float t = dot(v1v3, r);
	if (t > 0) {
		*(distance) = t;
		*(collided) = true;
	}
}
__device__ float3* Model::returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance, MeshTriangle* collidedMesh) {
	int pieceCount = 128;
	int threadsNumber = meshList->size / pieceCount;
	if (meshList->size % pieceCount != 0) {
		threadsNumber++;
	}
	*collided = false;
	PiecedReturnCollision<<<1, threadsNumber>>>(rayPoint, rayDirection, collided, distance, collidedMesh, (Model*)this, pieceCount);
	//Test << <1, threadsNumber >> > ();
	// wait for child to complete 
	cudaDeviceSynchronize();
	
	float3 hitPoint = float3(rayPoint + *distance * rayDirection);
	return &hitPoint;
}

__device__ void Model::Rotate(float x, float y, float z) {
	this->Object::Rotate(x, y, z);
	int numBlock = 2;
	int numThreadsPerBlock = 1024;
	RotateModel << <numBlock, numThreadsPerBlock >> > ((Model*)this);
}

__global__ void RotateModel(Model* model) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < model->meshList->size) {
		MeshTriangle* currentElement = &(model->meshList->meshArray[i]);
		float3 nv1 = currentElement->baseV1 - model->position;
		float3 nv2 = currentElement->baseV2 - model->position;
		float3 nv3 = currentElement->baseV3 - model->position;
		float degree = model->rotation.y * M_PI / 180;
		float sinRot = sin(degree);
		float cosRot = cos(degree);
		currentElement->v1.x = nv1.x * cosRot + nv1.z * sinRot;
		currentElement->v1.z = nv1.z * cosRot - nv1.x * sinRot;
		currentElement->v2.x = nv2.x * cosRot + nv2.z * sinRot;
		currentElement->v2.z = nv2.z * cosRot - nv2.x * sinRot;
		currentElement->v3.x = nv3.x * cosRot + nv3.z * sinRot;
		currentElement->v3.z = nv3.z * cosRot - nv3.x * sinRot;

		currentElement->v1.y = nv1.y;
		currentElement->v2.y = nv2.y;
		currentElement->v3.y = nv3.y;

		currentElement->v1 += model->position;
		currentElement->v2 += model->position;
		currentElement->v3 += model->position;
	}
	
}


__global__ void PiecedReturnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance, MeshTriangle* collidedMeshes, Model* model, int pieceCount) {
	int i = threadIdx.x * pieceCount;
	int to = min(i + pieceCount, model->meshList->size);
	
	*collided = false;
	return;
	while (i < to) {
		i++;
		continue;
		MeshTriangle* currentElement = &(model->meshList->meshArray[i]);
		
		float3 v1v2 = currentElement->v2 - currentElement->v1;
		float3 v1v3 = currentElement->v3 - currentElement->v1;

		float3 q = cross(rayDirection, v1v3);
		float a = dot(v1v2, q);

		
		if (abs(a) <= Epsilon) {
			i++;
			continue;
		}

		float3 s = (rayPoint - currentElement->v1 - model->position) / a;
		float3 r = cross(s, v1v2);

		float b1 = dot(s, q);
		float b2 = dot(r, rayDirection);
		float b3 = 1.0f - b1 - b2;

		
		if (b1 < 0 || b2 < 0 || b3 < 0) {
			i++;
			continue;
		}

		
		float t = dot(v1v3, r);
		if (t > 0) {
			__syncthreads();
			if (!*(collided)) {
				*(distance) = t;
				*(collided) = true;
				*collidedMeshes = *currentElement;
			}
			else if (t < *(distance)) {
				*(distance) = t;
				*collidedMeshes = *currentElement;
			}
			i++;
			continue;
		}

		i++;
		continue;
	}
}