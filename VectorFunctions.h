#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include <helper_math.h>

__device__ float3 operator+(const float3 &a, const float3 &b) {

	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}
__device__ float3 operator-(const float3 &a, const float3 &b) {

	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);

}