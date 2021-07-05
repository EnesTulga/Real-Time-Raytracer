#pragma once
#define M_PI           3.14159265358979323846
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <helper_math.h>

class Material {
public:
	__device__ Material();
	__device__ Material(float4 color);
	__device__ virtual float4 GetColorAtPosition(float3 position);
	__device__ virtual float4 GetColorAtPositionWithTime(float3 position, float time);

	float4 color;
	float4* texture;
	bool reflection;
	bool refraction;
	int textureWidth, textureHeight;
};
