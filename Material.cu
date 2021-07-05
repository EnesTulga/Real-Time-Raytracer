#include "Material.h"

__device__ Material::Material()
{
	reflection = false;
	refraction = false;
	color = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
}
__device__ Material::Material(float4 c)
{
	reflection = false;
	refraction = false;
	color = c;
}
__device__ float4 Material::GetColorAtPosition(float3 position) {
	float u = 0.5f + atan2(position.x, position.z)/ (2 * M_PI);
	float v = 0.5 - asin(position.y)/M_PI;
	int x = u * (textureWidth - 1);
	int y = v * (textureHeight - 1);
	return texture[x + (y * textureWidth)];
}
__device__ float4 Material::GetColorAtPositionWithTime(float3 position, float time) {
	float u = 0.5f + atan2(position.x, position.z) / (2 * M_PI);
	float v = 0.5 - asin(position.y) / M_PI;
	u = u + time;
	u = u - (int)u;
	int x = u * (textureWidth - 1);
	int y = v * (textureHeight - 1);
	return texture[x + (y * textureWidth)];
}