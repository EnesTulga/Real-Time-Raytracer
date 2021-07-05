#pragma once
#include "Objects.h"

class MeshTriangle;

class MeshLinkedListElement {
	public:
		MeshLinkedListElement* next;
		MeshTriangle* data;
};
class MeshLinkedList {
	public:
		MeshLinkedListElement* root;
		int size;
};
class DynamicMeshArray;
class MeshTriangle {
public:
	__device__ MeshTriangle();
	__device__ MeshTriangle(float3 v1, float3 v2, float3 v3);
	__device__ float3 GetNormal();
	__device__ float3* returnCollision(float3 rayPoint, float3 rayDirection, bool* collided, float* distance, float3 position);

	float3 baseV1, baseV2, baseV3;
	float3 v1, v2, v3;
};
class DynamicMeshArray {
	public:
		int size;
		MeshTriangle* meshArray;
		__device__ DynamicMeshArray() {
			maxSize = 16;
			size = 0;
			meshArray = (MeshTriangle*)malloc(sizeof(MeshTriangle) * maxSize);
		}
		__device__ void Add(MeshTriangle* meshTriangle) {
			size++;
			int topLimit = size - 1;
			if (size >= maxSize) {
				maxSize *= 2;
				
				MeshTriangle* temp = (MeshTriangle*)malloc(sizeof(MeshTriangle) * maxSize);
				for (int i = 0; i < topLimit; i++) {
					temp[i] = meshArray[i];
				}
				free(meshArray);
				meshArray = temp;
			}
			meshArray[topLimit] = *meshTriangle;
		}
	private:
		int maxSize;
};