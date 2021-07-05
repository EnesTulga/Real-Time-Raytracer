#define GLEW_STATIC
#define BUFFER_OFFSET(i) ((char *)NULL + (i))
#define M_PI           3.14159265358979323846
#define REFRESH_DELAY     16 //ms
#pragma once
#include <GL/glew.h>
#include <GL/glut.h>
#include <assert.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include <helper_math.h>
#include <helper_functions.h>
#include "Objects.h"
#include "BoundingBox.h"
#include "RGBAUtilities.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
float4** textureAccess;
GLuint pbo;
struct cudaGraphicsResource *cuda_pbo_resource;
Object** objects;
float4* testTexture;
float4* floorTexture;
BoundingBox ** boundingBoxes;
int testTextureWidth, testTextureHeight;

const unsigned int mesh_width = 512;
const unsigned int mesh_height = 512;

float g_fAnim = 0.0;

// declaration, forward
void cleanup();

//
void LoadModels(float3** verticiesPointer, int3** facesPointer, int* faceSizePointer, int* verticiesSizePointer);

// GL functionality
bool initGL(int *argc, char **argv);
void createPBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
cudaError_t createObjectsBuffer(Object*** objectsList, char **argv);
void deletePBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

int pieceCount = 16;
bool pressedKeys[10] = {false,false, false, false, false, false, false, false, false, false};
// rendering callbacks
void DisplayFunction();
void checkKeysDown(unsigned char key, int x, int y);
void checkKeysUp(unsigned char key, int x, int y);
void timerEvent(int value);
float3 cameraPos;
__device__ float4 Raycast(float3 cameraPoint, float3 direction, BoundingBox** objects, bool* collidedSomething, float* distance, Object** collidedObject, bool *planeCollided) {
	int i = 0;
	BoundingBox* box = *objects;
	bool boxCollided = false;
	float dis;
	while (box != NULL) {
		box->returnCollision(cameraPoint, direction, &boxCollided, &dis);
		if (boxCollided) {
			Object* object = *(box->objectsInsideOfTheBox);
			int k = 0;
			while (object != NULL) {
				float distance2 = 0;
				bool intersectionObject = false;
				object->returnCollision(cameraPoint, direction, &intersectionObject, &distance2);
				if (intersectionObject) {
					if (*collidedSomething) {
						if (distance2 < *distance) {
							*collidedObject = object;
							*distance = distance2;
						}
					}
					else {
						*collidedObject = object;
						*distance = distance2;
						*collidedSomething = true;
					}
				}
				k++;
				object = *(box->objectsInsideOfTheBox + k);
			}
		}

		i++;
		box = *(objects + i);
	}
	if (!*collidedSomething) {
		float3 n = make_float3(0, -1, 0);
		float denom = dot(n, direction);
		if (denom > 1e-6) {
			float3 p0l0 = make_float3(0, -0.5f, 0) - cameraPoint;
			float t = dot(p0l0, n) / denom;
			if (t > 0) {
				*planeCollided = true;
				*distance = t;
				return;
			}
		}
	}
}
__device__ float4 RaycastNoReflection(float3 cameraPoint, float3 direction, BoundingBox** objects, bool* collidedSomething, float* distance, Object** collidedObject, bool *planeCollided, int depth) {
	int i = 0;
	BoundingBox* box = *objects;
	bool boxCollided = false;
	float dis;
	while (box != NULL) {
		box->returnCollision(cameraPoint, direction, &boxCollided, &dis);
		if (boxCollided) {
			Object* object = *(box->objectsInsideOfTheBox);
			int k = 0;
			while (object != NULL) {
				float distance2 = 0;
				bool intersectionObject = false;
				object->returnCollision(cameraPoint, direction, &intersectionObject, &distance2);
				if (intersectionObject) {
					if (*collidedSomething) {
						if (distance2 < *distance) {
							*collidedObject = object;
							*distance = distance2;
						}
					}
					else {
						*collidedObject = object;
						*distance = distance2;
						*collidedSomething = true;
					}
				}
				k++;
				object = *(box->objectsInsideOfTheBox + k);
			}
		}

		i++;
		box = *(objects + i);
	}
	if (*collidedSomething && (*collidedObject)->material.reflection) {
		if (depth == 0) {
			*collidedSomething = false;
			float3 n = make_float3(0, -1, 0);
			float denom = dot(n, direction);
			if (denom > 1e-6) {
				float3 p0l0 = make_float3(0, -0.5f, 0) - cameraPoint;
				float t = dot(p0l0, n) / denom;
				if (t > 0) {
					*planeCollided = true;
					*distance = t;
					return;
				}
			}
			return;
		}
		else {
			float3 newCameraPoint = cameraPoint + direction * *distance;
			float3 normal = (*collidedObject)->GetNormal(newCameraPoint);
			float3 newDirection = normalize(direction - 2 * (dot(direction, normal)) * normal);
			Object* reflectionRayCollidedObject;
			*collidedSomething = false;
			RaycastNoReflection(newCameraPoint + newDirection * 0.01f, newDirection, objects, collidedSomething, distance, &reflectionRayCollidedObject, planeCollided, depth - 1);
		}
	}
	else if (!*collidedSomething) {
		float3 n = make_float3(0, -1, 0);
		float denom = dot(n, direction);
		if (denom > 1e-6) {
			float3 p0l0 = make_float3(0, -0.5f, 0) - cameraPoint;
			float t = dot(p0l0, n) / denom;
			if (t > 0) {
				*planeCollided = true;
				*distance = t;
				return;
			}
		}
	}
}
// Cuda functionality
float cameraRotate, cameraRotate2;
void Render(struct cudaGraphicsResource **pbo_resource);
__device__ int GetSkyboxIndex(float3 direction) {
	float u = 0.5f + atan2(direction.x, direction.z) / (2 * M_PI);
	float v = 0.5 - asin(direction.y) / M_PI;
	int uvX = u * 1499;
	int uvY = v * 750;
	return uvX + (uvY * 1500);
}
__device__ int GetFloorTextureIndex(float3 position) {
	float u = (position.x - 5 * ((int)(position.x / 5))) / 5;
	if (position.x < 0) {
		u *= -1;
	}
	
	float v = (position.z - 5 * (int)(position.z / 5))/5;
	if (position.z < 0) {
		v *= -1;
	}
	int uvX = u * 299;
	int uvY = v * 300;
	return uvX + (uvY * 300);
}
__global__ void rotateFirstObject(Object** objects);
__global__ void CalculatePixel(float4* pos, unsigned int width, unsigned int height, Object** objects, float time, float3 cameraPoint, bool* collidedList) {

	unsigned int pixelId = blockIdx.x;
	unsigned int x = blockIdx.x % height;
	unsigned int y = blockIdx.x / width;
	float distance;
	bool collided = false;
	float3 eyePoint = make_float3(cameraPoint.x + 0.2f, 0.1f * (((float)height) - 2 * y) / (float)height, 0.1f * (((float)width) - 2 * x) / (float)width);
	float3 direction = normalize(eyePoint - cameraPoint);
	Model* obj1 = (Model*)*objects;

	MeshTriangle* currentElement = &(obj1->meshList->meshArray[threadIdx.x]);

	float3 v1v2 = currentElement->v2 - currentElement->v1;
	float3 v1v3 = currentElement->v3 - currentElement->v1;

	float3 q = cross(direction, v1v3);
	float a = dot(v1v2, q);


	if (abs(a) > Epsilon) {

		float3 s = (cameraPoint - currentElement->v1 - obj1->position) / a;
		float3 r = cross(s, v1v2);
		
		float b1 = dot(s, q);
		float b2 = dot(r, direction);
		float b3 = 1.0f - b1 - b2;


		if (b1 >= 0 && b2 >= 0 && b3 >= 0) {
			float t = dot(v1v3, r);
			if (t > 0) {
				*(collidedList + pixelId) = true;
			}
		}
	}
	return;
	if (*(collidedList + pixelId)) {
		pos[pixelId] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	}
	else {
		pos[pixelId] = make_float4(0, 0, 0, 1.0f);
	}
	return;
	//obj1->checkIntersection(cameraPoint, direction, (collidedList + pixelId), &distance, threadIdx.x);

	//obj1->meshList->meshArray[threadIdx.x].returnCollision(cameraPoint, direction, (collidedList + pixelId), &distance, obj1->position);
	if (*(collidedList + pixelId)) {
		pos[pixelId] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	}
	else {
		pos[pixelId] = make_float4(0, 0, 0, 1.0f);
	}
}
__global__ void CalculatePixel(float4* pos, unsigned int width, unsigned int height, BoundingBox** objects, float3 cameraPoint, float timer, float rotateDegree, float rotateDegreeZ, float4** textures) {
	int x = threadIdx.x;
	int y = blockIdx.x;
	int posIndex = threadIdx.x + blockDim.x * blockIdx.x;
	float4 color = make_float4(0,0,0,0);
	//BoundingBox* box = *objects;
	//bool boxCollided = false;
	//float dis;
	//float3 eyePoint2 = make_float3(cameraPoint.x + 0.2f, cameraPoint.y + 0.1f * (((float)height) - 2 * y) / (float)height, cameraPoint.z + (0.1f * (((float)width) - 2 * x) ) / (float)width);
	//float3 direction2 = normalize(eyePoint2 - cameraPoint);
	//box->returnCollision(cameraPoint, direction2, &boxCollided, &dis);
	//if (boxCollided) {
	//	color = make_float4(1,1,1,1);
	//}
	int distributedRayNumber = 5;
	float3 lightPosition = make_float3(0, 5, -3 + 6 * sin(M_PI * ((timer / 10) - ((int)timer / 10))));
	for (int j = 0; j < distributedRayNumber; j++) {
		//break;
		float val1 = 0;
		float val2 = 0;
		if (j == 0) {
			val1 = -0.05f;
		}
		else if (j == 2) {
			val1 = 0.05f;
		}
		else if (j == 3) {
			val2 = 0.05f;
		}
		else if (j == 4) {
			val2 = -0.05f;
		}
		float distance;
		bool collidedSomething = false;
		float3 eyePoint = make_float3(0.2f, -(0.1f * (((float)height) - 2 * y) + val2) / (float)height, -(0.1f * (((float)width) - 2 * x) + val1) / (float)width);
		float degree = rotateDegree * M_PI / 180;
		float sinRot = sin(degree);
		float cosRot = cos(degree);
		degree = rotateDegreeZ * M_PI / 180;
		float sinRot2 = sin(degree);
		float cosRot2 = cos(degree);
		float newEyeX = sinRot2 * eyePoint.y + cosRot2 * (cosRot * eyePoint.x - sinRot * eyePoint.z);
		float newEyeY = cosRot2 * eyePoint.y - sinRot2 * (cosRot * eyePoint.x - sinRot * eyePoint.z);
		float newEyeZ = eyePoint.z * cosRot + eyePoint.x * sinRot;

		eyePoint.x = newEyeX +cameraPoint.x;
		eyePoint.y = newEyeY + cameraPoint.y;
		eyePoint.z = newEyeZ + cameraPoint.z;
		//eyePoint.z += cameraPoint.z;
		float3 direction = normalize(eyePoint - cameraPoint);
		Object* collidedObject = NULL;
		bool planeCollided = false;
		Raycast(cameraPoint, direction, objects, &collidedSomething, &distance, &collidedObject, &planeCollided);
		if (collidedSomething) {
			float3 newCameraPoint = cameraPoint + direction * distance;
			float3 normal = collidedObject->GetNormal(newCameraPoint);
			if (!collidedObject->material.reflection) {
				color = color + collidedObject->material.color / distributedRayNumber;
			}
			else {
				float3 newDirection = normalize(direction - 2 * (dot(direction, normal)) * normal);
				Object* reflectionRayCollidedObject;
				collidedSomething = false;
				RaycastNoReflection(newCameraPoint + newDirection * 0.01f, newDirection, objects, &collidedSomething, &distance, &reflectionRayCollidedObject, &planeCollided, 1);
				if (collidedSomething) {
					color = color + (reflectionRayCollidedObject->material.color * 0.93f) / distributedRayNumber;
				}
				else if (planeCollided) {
					color += (*(textures + 1))[GetFloorTextureIndex(newCameraPoint + newDirection * distance)] / distributedRayNumber;
				}
				else {
					color += (*textures)[GetSkyboxIndex(newDirection)] / distributedRayNumber;
				}
			}
			if (!collidedObject->material.reflection) {
				
				float3 shadowRayDirection = normalize(lightPosition - newCameraPoint);
				bool shadowRayCollided = false;
				Object* shadowRayCollidedObject;
				float shadowDot = dot(normal, shadowRayDirection);
				if (shadowDot > 0) {
					Raycast(newCameraPoint + shadowRayDirection * 0.01f, shadowRayDirection, objects, &shadowRayCollided, &distance, &shadowRayCollidedObject, &planeCollided);
					if (shadowRayCollided) {
						color -= shadowDot * shadowDot * shadowDot * make_float4(1 - color.x, 1 - color.y, 1 - color.z, 0) / distributedRayNumber;
					}
					else {
						color += (shadowDot * shadowDot * shadowDot * shadowDot / distributedRayNumber) * make_float4(1 - color.x, 1 - color.y, 1 - color.z, 0);
					}
					
				}
				else {
					color += (-shadowDot * shadowDot / distributedRayNumber) * color;
				}
			}
		}
		else if (planeCollided) {
			float3 planeHitPos = (cameraPoint + direction * distance);
			float3 shadowRayDirection = normalize(lightPosition - planeHitPos);
			bool shadowRayCollided = false;
			Object* shadowRayCollidedObject;
			//float shadowDot = dot(make_float3(0,1,0), shadowRayDirection);
			planeCollided = false;
			Raycast(planeHitPos + shadowRayDirection * 0.01f, shadowRayDirection, objects, &shadowRayCollided, &distance, &shadowRayCollidedObject, &planeCollided);
			if (shadowRayCollided) {
				//color -= make_float4(color.x,color.y, color.z, 0) / distributedRayNumber;
				float shadowDot = dot(shadowRayDirection, shadowRayCollidedObject->GetNormal(planeHitPos + shadowRayDirection * distance));
				color += (shadowDot * shadowDot * shadowDot * 0.75 + 1) * (*(textures + 1))[GetFloorTextureIndex(planeHitPos)] / distributedRayNumber;
			}
			else {
				color += (*(textures+1))[GetFloorTextureIndex(planeHitPos)] / distributedRayNumber;
			}
				
			
		}
		else {
			
			color += (*textures)[GetSkyboxIndex(direction)] / distributedRayNumber;
		}
	}
	
	color.w = 1;
	pos[posIndex] = color;
}
__global__ void CalculateTheFrameParallel(float4 *pos, unsigned int width, unsigned int height, Object** objects, float time, float3 cameraPoint)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int row = y * width;

	/*pos[row + x] = make_float4(0.5f + sin(time * 2)/2, 1.0f, 1.0f, 1.0f);

	
	Object* obj1 = *objects;

	bool collided = false;
	float distance = 0;
	float3 eyePoint = make_float3(cameraPoint.x + 0.2f, 0.1f * (((float)height) - 2 * y) / (float)height,0.1f * (((float)width) - 2 * x) / (float)width);
	float3 direction = normalize(eyePoint - cameraPoint);
	MeshTriangle collidedMesh;
	
	obj1->returnCollision(cameraPoint, direction, &collided, &distance, &collidedMesh);
	return;
	//obj1->returnCollision(cameraPoint, direction, &collided, &distance);
	if (collided) {
		//float3 hitPoint = cameraPoint + direction * distance;
		//pos[row + x] = obj1->material.GetColorAtPositionWithTime(normalize(hitPoint - obj1->position), (time- (int)time + ((int)time) % 2) / 2);
		pos[row + x] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
	}
	else {
		pos[row + x] = make_float4(0, 0, 0, 1.0f);
	}*/
	int objectSize = 2;
	//float3 cameraPoint = make_float3(-5, 0, 0);
	float3 eyePoint = make_float3(cameraPoint.x + 0.2f, 0.1f * (((float)height) - 2 * y) / (float)height, 0.1f * (((float)width) - 2 * x) / (float)width);
	float3 direction = normalize(eyePoint - cameraPoint);
	float3 lightPosition = make_float3(0, 5 * sin(time * 2), 5 * cos(time * 2));
	bool collided = false;
	float distance = 0;
	Object* collidedObject;
	for (int i = 0; i < objectSize; i++) {
		Object* object = *(objects + i);
		bool collided2 = false;
		float distance2;
		object->returnCollision(cameraPoint, direction, &collided2, &distance2);
		if (collided2) {
			if (collided) {
				if (distance2 < distance) {
					distance = distance2;
					collidedObject = object;
				}
			}
			else {
				distance = distance2;
				collidedObject = object;
				collided = true;
			}
		}
	}
	if (collided) {
		float3 hitPoint = cameraPoint + direction * distance;
		float3 normal = collidedObject->GetNormal(hitPoint);
		float3 reflectionDirection = normalize(direction - 2 * (dot(direction, normal)) * normal);
		float3 shadowRayDirection = lightPosition - hitPoint;
		float lightDistance = length(shadowRayDirection);
		shadowRayDirection = normalize(shadowRayDirection);

		bool collided4 = false;
		float distance4 = 0;
		Object* collidedObject3;
		float shadowRatio = 1.0f;
		float dt = dot(-1.0f * normal, shadowRayDirection);
		float angle = (180 / M_PI) * (float)acos(dt / (length(normal)  * lightDistance));
		if ((angle > 90 && angle < 270)) {
			shadowRatio = 2;
		}
		else {
			
			for (int i = 0; i < objectSize; i++) {
				Object* object = *(objects + i);
				if (object == collidedObject) {
					continue;
				}
				bool collided2 = false;
				float distance2;
				object->returnCollision(hitPoint, -shadowRayDirection, &collided2, &distance2);
				if (collided2 && distance2 < lightDistance) {
					if (collided4) {
						if (distance2 < distance4) {
							distance4 = distance2;
							shadowRatio = 10 / max((lightDistance - distance4), 1.0f);
						}
					}
					else {
						distance4 = distance2;
						collided4 = true;
						shadowRatio = 2;
					}
					
				}
			}
		}
		
		bool collided3 = false;
		float distance3 = 0;
		for (int i = 0; i < objectSize; i++) {
			Object* object = *(objects + i);
			if (object == collidedObject) {
				continue;
			}
			bool collided2 = false;
			float distance2;
			object->returnCollision(hitPoint, reflectionDirection, &collided2, &distance2);
			if (collided2) {
				if (collided3) {
					if (distance2 < distance3) {
						distance3 = distance2;
						collidedObject3 = object;
					}
				}
				else {
					distance3 = distance2;
					collidedObject3 = object;
					collided3 = true;
				}
			}
		}
		if (collided3) {
			pos[row + x] = (collidedObject->material.color * 0.25f + collidedObject3->material.color * 0.75f) / shadowRatio;
		}
		else {
			pos[row + x] = collidedObject->material.color / shadowRatio;
		}
		
	}
	else {
		pos[row + x] = make_float4(0, 0, 0, 1.0f);
	}
}
__global__ void load_objects_to_kernel(Object** objects, float4** testTexturePointer, int textureWidth, int textureHeight) {
	//(*objects) = new Sphere(make_float3(0, 0, -1.25f), float3(), make_float3(1, 1, 1), 1);
	//(*objects)->material.color.y = 0.0f;
	(*(objects + 0)) = new Sphere(make_float3(0,0,1), float3(), make_float3(1,1,1), 1.25f);
	(*(objects + 0))->material.color.x = 0.0f;
	(*(objects + 1)) = new Sphere(make_float3(0, 3, 0), float3(), make_float3(1, 1, 1), 1.5f);
	(*(objects + 1))->material.color.z = 0.0f;
	//(*objects)->material.texture = *testTexturePointer;
	//(*objects)->material.textureHeight = textureHeight;
	//(*objects)->material.textureWidth = textureWidth;
}
__global__ void load_model_to_kernel(Object** objects, float3* verticies, int3* faces, int faceSize) {
	(*objects) = new Model();
	Model* model = static_cast<Model*>(*objects);
	model->position = make_float3(0, 1, 0);
	if (faceSize == 0) {
		return;
	}
	int i = 0;
	while (i < faceSize) {
		MeshTriangle* currentElement = new MeshTriangle(verticies[faces[i].x], verticies[faces[i].y], verticies[faces[i].z]);
		model->meshList->Add(currentElement);
		i++;
	}
	model->meshList->size = faceSize;
}

__global__ void load_BoundingBox(Object** objects, BoundingBox ** boxes) {

	int i = 0;
	float4 colors[9] = { make_float4(1,0,0,1),make_float4(0,1,0,1),
						make_float4(0,0,1,1),make_float4(0.5f,0.5f,0,1),
						make_float4(0.5f,1,0.5f,1), make_float4(0,0.5f,0.5f,1),
						make_float4(0.75f,0.25f,0.5f,1), make_float4(1,0.25f,0.75f,1), make_float4(0.25f,0.75f,0,1) };
	while (i < 12) {
		float3 boxPos = make_float3(3 - 2 * (int)(i / 3), 0, 2 - 2 * (i%3));
		*(boxes + i) = new BoundingBox(boxPos, make_float3(-1, -0.5f, -1), make_float3(1, 0.5f, 1));
		(*(boxes + i))->objectsInsideOfTheBox = (Object**)malloc(sizeof(Object*) * 4);
		
		int j = 0;
		while (j < 4) {
			*((*(boxes+i))->objectsInsideOfTheBox + j) = new Sphere(boxPos + make_float3(0.5f - (int)(j/2), 0, 0.5f - (int)(j % 2)), float3(), make_float3(1, 1, 1), 0.5f);
			(*((*(boxes + i))->objectsInsideOfTheBox + j))->material.color = colors[(i*4 + j)%9];
			if ((i + j%2)%2 == 0) {
				(*((*(boxes + i))->objectsInsideOfTheBox + j))->material.reflection = true;
			}
			else if (i == 11 && j == 2) {
				(*((*(boxes + i))->objectsInsideOfTheBox + j))->material.refraction = true;
			}
			j++;
		}
		i++;
	}
	*(boxes + 12) = new BoundingBox(make_float3(0,3,0), make_float3(-0.5, -1.5f, -0.5), make_float3(0.5, 1.5f, 0.5));
	(*(boxes + 12))->objectsInsideOfTheBox = (Object**)malloc(sizeof(Object*) * 2);
	*((*(boxes + 12))->objectsInsideOfTheBox) = new Sphere(make_float3(0, 2, 0), float3(), make_float3(1, 1, 1), 0.5f);
	(*((*(boxes + 12))->objectsInsideOfTheBox))->material.reflection = true;
	*((*(boxes + 12))->objectsInsideOfTheBox + 1) = new Sphere(make_float3(0, 3.3f, 0), float3(), make_float3(1, 1, 1), 0.5f);
	(*((*(boxes + 12))->objectsInsideOfTheBox + 1))->material.color = colors[0];
}

__global__ void unload_objects_to_kernel(Object** objects) {
	int i = 0;
	while (*(objects + i)) {
		delete *(objects + i);
		i++;
	}
}

__global__ void rotateFirstObject(Object** objects) {
	(*objects)->Rotate(0, 0.016f * 360, 0);
}
__global__ void changePositionOfFirstObject(Object** objects, float time) {
	float incrementY = sin(time * 2);
	(*objects)->position.y = incrementY;
}
__global__ void ResetFrame(bool* collided, int width, float4* pbo) {
	int id = ((blockIdx.y*blockDim.y + threadIdx.y) * width + blockIdx.x*blockDim.x + threadIdx.x) * 16;
	int i = 0;
	bool noDrawed = true;
	while (i < 16) {
		if (noDrawed && *(collided + id + i)) {
			noDrawed = false;
			pbo[id / 16] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
		}
		*(collided + id + i) = false;
		i++;
	}
	if(noDrawed){
		pbo[id / 16] = make_float4(0,0,0,0);
	}
}
__global__ void Test(int pieceNumber, int width, int height, Object** objects, float3 cameraPoint, bool* collidedList) {
	unsigned int i = (threadIdx.x % pieceNumber) * 128;
	unsigned int to = i + 128;
	Model* model = (Model*)*objects;
	if (i >= model->meshList->size || to > model->meshList->size) {
		return;
	}
	unsigned int collidedId = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int x = (collidedId / pieceNumber) % height;
	unsigned int y = (collidedId / pieceNumber) / width;
	float3 eyePoint = make_float3(cameraPoint.x + 0.2f, 0.1f * (((float)height) - 2 * y) / (float)height, 0.1f * (((float)width) - 2 * x) / (float)width);
	float3 direction = normalize(eyePoint - cameraPoint);
	while (i < to) {
		MeshTriangle* currentElement = &(model->meshList->meshArray[i]);

		float3 v1v2 = currentElement->v2 - currentElement->v1;
		float3 v1v3 = currentElement->v3 - currentElement->v1;

		float3 q = cross(direction, v1v3);
		float a = dot(v1v2, q);


		if (abs(a) <= Epsilon) {
			i++;
			continue;
		}

		float3 s = (cameraPoint - currentElement->v1 - model->position) / a;
		float3 r = cross(s, v1v2);

		float b1 = dot(s, q);
		float b2 = dot(r, direction);
		float b3 = 1.0f - b1 - b2;


		if (b1 < 0 || b2 < 0 || b3 < 0) {
			i++;
			continue;
		}


		float t = dot(v1v3, r);
		if (t > 0) {
			collidedList[collidedId] = true;
			//i++;
			return;
		}
		i++;
	}
}
void process_Normal_Keys(int key, int x, int y);
int main(int argc, char **argv)
{
	cameraPos = make_float3(-5,0,0);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(512, 512);
	glutCreateWindow("single triangle");
	glewExperimental = GL_TRUE;
	if (glewInit()) {
		exit(EXIT_FAILURE);
	}
	glutDisplayFunc(DisplayFunction);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	//glutSpecialFunc(process_Normal_Keys);
	glutKeyboardFunc(checkKeysDown);
	glutKeyboardUpFunc(checkKeysUp);
#if defined (__APPLE__) || defined(MACOSX)
	atexit(cleanup);
#else
	glutCloseFunc(cleanup);
#endif
	glClearColor(0.0, 0.0, 0.0, 1.0);
	// create PBO
	createPBO(&pbo, &cuda_pbo_resource, cudaGraphicsMapFlagsWriteDiscard);
	createObjectsBuffer(&objects, argv);
	// run the cuda part
	Render(&cuda_pbo_resource);

	glutMainLoop();
	return 0; 
}
struct IHDR {
	uint32_t width;
	uint32_t height;
	uint8_t depth;
	uint8_t type;
	uint8_t compression;
	uint8_t filter;
	uint8_t interlace;
};
bool* pixelsCollided;
cudaError_t createObjectsBuffer(Object*** objectsList, char **argv) {
	cudaError_t cudaStatus;
	bool error = false;
	char* imageFilename = (char*)"C:\\Users\\UserName\\source\\repos\\Real-Time-Raytracer\\x64\\Release\\skybox2.png";
	read_png_file(imageFilename);
	uint8* image = getImage();
	testTextureWidth = getImageWidth();
	testTextureHeight = getImageLength();
	
	cudaStatus = cudaMalloc((void**)&textureAccess, sizeof(float4*) * 2);
	cudaStatus = cudaMalloc((void**)&testTexture, sizeof(float4) * testTextureWidth * testTextureHeight);
	cudaStatus = cudaMemcpy(textureAccess, &testTexture, sizeof(float4*), cudaMemcpyHostToDevice);
	int i = 0;
	int size = testTextureWidth * testTextureHeight;
	int channelSize = getSamplesPerPixel();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc failed 2! " << cudaStatus << std::endl;
		goto Error;
	}
	std::cout << channelSize << std::endl;
	while (i < size) {
		int currentIndex = i * channelSize;
		float r = image[currentIndex++] / 255.0f;
		float g = image[currentIndex++] / 255.0f;
		float b = image[currentIndex] / 255.0f;
		float a = 1.0f;
		float4 rgba = make_float4(r,g,b,a);

		cudaStatus = cudaMemcpy(&testTexture[i], &rgba, sizeof(float4), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Texture atanirken hata! " << cudaStatus << std::endl;
			goto Error;
		}
		i++;
	}
	std::cout << "Texture atamasi tamamlandi!" << std::endl;
	imageFilename = (char*)"C:\\Users\\UserName\\source\\repos\\Real-Time-Raytracer\\x64\\Release\\floor.png";
	read_png_file(imageFilename);
	uint8* image2 = getImage();
	testTextureWidth = getImageWidth();
	testTextureHeight = getImageLength();
	cudaStatus = cudaMalloc((void**)&floorTexture, sizeof(float4) * testTextureWidth * testTextureHeight);
	cudaStatus = cudaMemcpy((textureAccess+1), &floorTexture, sizeof(float4*), cudaMemcpyHostToDevice);
	i = 0;
	size = testTextureWidth * testTextureHeight;
	channelSize = getSamplesPerPixel();
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc failed 2.1 before floor texture" << cudaStatus << std::endl;
		goto Error;
	}
	std::cout << channelSize << std::endl;
	while (i < size) {
		int currentIndex = i * channelSize;
		float r = image2[currentIndex++] / 255.0f;
		float g = image2[currentIndex++] / 255.0f;
		float b = image2[currentIndex] / 255.0f;
		float a = 1.0f;
		float4 rgba = make_float4(r, g, b, a);

		cudaStatus = cudaMemcpy(&floorTexture[i], &rgba, sizeof(float4), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Floor Texture atanirken hata! " << cudaStatus << std::endl;
			goto Error;
		}
		i++;
	}
	std::cout << "Floor Texture atamasi tamamlandi!" << std::endl;
	cudaStatus = cudaMalloc((void**)objectsList, sizeof(Object*) * 1);
	cudaStatus = cudaMalloc((void**)&boundingBoxes, sizeof(BoundingBox*) * 1);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaMalloc failed 2.2!" << std::endl;
		goto Error;
	}
	float3* verticies, * verticiesDevice;
	int3* faces, * facesDevice;
	int faceSize, verticiesSize;
	LoadModels(&verticies, &faces, &faceSize, &verticiesSize);
	std::cout << "Verticies: " << verticiesSize << "/ Faces: " << faceSize << endl;
	cudaStatus = cudaMalloc((void**)&verticiesDevice, sizeof(float3) * verticiesSize);
	cudaStatus = cudaMemcpy(verticiesDevice, verticies, sizeof(float3) * verticiesSize, cudaMemcpyHostToDevice);
	cudaStatus = cudaMalloc((void**)&pixelsCollided, sizeof(bool) * mesh_width * mesh_height * pieceCount);
	dim3 block(32, 32);
	dim3 grid(mesh_width / block.x, mesh_height / block.y);
	//ResetFrame << <grid, block >> > (pixelsCollided, mesh_width);
	cudaStatus = cudaMalloc((void**)&facesDevice, sizeof(int3) * faceSize);
	cudaStatus = cudaMemcpy(facesDevice, faces, sizeof(int3) * faceSize, cudaMemcpyHostToDevice);
	//load_objects_to_kernel << < 1, 1 >> > (*objectsList, textureAccess, testTextureWidth, testTextureHeight);
	//load_model_to_kernel << < 1, 1 >> > (*objectsList, verticiesDevice, facesDevice, faceSize);
	load_BoundingBox<<<1,1>>>(objects, boundingBoxes);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		std::cout << "addKernel launch failed: \n" << cudaGetErrorString(cudaStatus) << std::endl;
		goto Error;
	}
Error:
	if (error) {
		std::cout << "Error var" << std::endl;
	}

	return cudaStatus;
}
void createPBO(GLuint *pbo, struct cudaGraphicsResource **pbo_res, unsigned int pbo_res_flags)
{
	assert(pbo);

	// create buffer object
	glGenBuffers(1, pbo);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, *pbo);

	// initialize buffer object
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
	glBufferData(GL_PIXEL_PACK_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

	// register this buffer object with CUDA
	cudaGraphicsGLRegisterBuffer(pbo_res, *pbo, pbo_res_flags);

}

void LoadModels(float3** verticiesPointer, int3** facesPointer, int* faceSizePointer, int* verticiesSizePointer) {
	string modelDirectory= "C:\\Users\\UserName\\source\\repos\\Real-Time-Raytracer\\x64\\Release\\models\\deer.obj";
	string line = "";
	ifstream objFile;

	objFile.open(modelDirectory, std::ifstream::in);
	if (!objFile) {
		std::cout << "Unable to open file" << endl;
		exit(1);
	}
	std::vector<float3> verticies;
	std::vector<int3> faces;
	while (getline(objFile, line)) {
		int size = line.size();
		
		if (size == 0) {
			continue;
		}
		int i = 0;
		if (line[i] == 'v') {
			if (line[++i] == ' ') {
				i++;
				string number = "";
				float position[3];
				int j = 0;
				while (j < 3) {
					if (line[i] != ' ' && line[i] != '/n') {
						number += line[i];
					}
					else if(number != ""){
						float value = std::stof(number);
						position[j] = -value/600;
						//position[j] = -value;
						number = "";
						j++;
					}
					i++;
				}
				if (j != 3) {
					continue;
				}
				float3 positionOfVertice = make_float3(position[0], position[1], position[2]);
				verticies.push_back(positionOfVertice);
			}
		}
		else if (line[i] == 'f') {
			if (line[++i] == ' ') {
				i++;
				string number = "";
				int face[3];
				int j = 0;
				while (j < 3) {
					if (line[i] != '/' && line[i] != '/n') {
						number += line[i];
					}
					else {
						int value = (std::stoi(number)) - 1;
						face[j] = value;
						number = "";
						j++;
						while (line[i] != ' ' || line[i] == '/n') {
							i++;
						}
					}
					i++;
				}
				if (j != 3) {
					continue;
				}
				int3 faceData = make_int3(face[0], face[1], face[2]);
				faces.push_back(faceData);
			}
		}
	}

	objFile.close();

	*verticiesPointer = verticies.data();
	*facesPointer = faces.data();
	*faceSizePointer = faces.size();
	*verticiesSizePointer = verticies.size();
}
int frameNumber = 0;
cudaError cudaStatus2;
void Render(struct cudaGraphicsResource **pbo_resource)
{
	frameNumber++;
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	cudaGraphicsMapResources(1, pbo_resource, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		*pbo_resource);

	dim3 block(32, 32);
	dim3 grid(mesh_width / block.x, mesh_height / block.y);
	
	cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);
	float elapsed = 0;
	
	
	
	cudaStatus2 = cudaGetLastError();
	if (cudaStatus2 != cudaSuccess) {
		std::cout << "Render rotate starting launch failed: \n" << cudaGetErrorString(cudaStatus2) << "/Frame : " << frameNumber << std::endl;
	}
	//rotateFirstObject << <1, 1 >> > (objects);
	int numBlock = 2;
	int numThreadsPerBlock = 1024;
	//RotateModel << <numBlock, numThreadsPerBlock >> > (objects);
	/*
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	std::cout << "Rotate Timing: " << elapsed / 1000.0f << '\n';
	elapsed = 0;*/
	cudaStatus2 = cudaGetLastError();
	if (cudaStatus2 != cudaSuccess) {
		std::cout << "Render timer starting launch failed: \n" << cudaGetErrorString(cudaStatus2) << "/Frame : " << frameNumber << std::endl;
	}
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	cudaStatus2 = cudaGetLastError();
	if (cudaStatus2 != cudaSuccess) {
		std::cout << "Render starting launch failed: \n" << cudaGetErrorString(cudaStatus2) << "/Frame : " << frameNumber << std::endl;
	}
	dim3 grid2(4096, 1, 1);
	dim3 block2(1024, 1, 1);
	//CalculatePixel << <grid2, block2 >> > (dptr, mesh_width, mesh_height, objects, g_fAnim, make_float3(-5, 0, 0), pixelsCollided);
	//CalculateTheFrameParallel << < grid, block >> > (dptr, mesh_width, mesh_height, objects, g_fAnim, make_float3(-5, 0, 0));
	CalculatePixel << <512, 512 >> > (dptr, 512, 512, boundingBoxes, cameraPos, g_fAnim, cameraRotate, cameraRotate2, textureAccess);
	//ResetFrame << <grid, block >> > (pixelsCollided, mesh_width, dptr);
	cudaStatus2 = cudaGetLastError();
	if (cudaStatus2 != cudaSuccess) {
		std::cout << "Render timer stop launch failed: \n" << cudaGetErrorString(cudaStatus2) << "/Frame : " << frameNumber << std::endl;
	}
	//Test << <grid2, block2 >> > ();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	//cudaStatus2 = cudaGetLastError();
	if (cudaStatus2 != cudaSuccess) {
		std::cout << "Render timer stop launch failed: \n" << cudaGetErrorString(cudaStatus2) << "/Frame : " << frameNumber << std::endl;
	}
	std::cout << "Ray Tracer Timing: " << elapsed / 1000.0f << "/ FPS: " << 1000.0f / elapsed  << '\n';
	// unmap buffer object
	cudaGraphicsUnmapResources(1, pbo_resource, 0);
}
void DisplayFunction(void)
{
	float xChange = 0, yChange = 0, zChange = 0;
	if (pressedKeys[0]) { 
		xChange += 0.08f;
	}
	if (pressedKeys[1]) { 
		xChange -= 0.08f;
	}
	if (pressedKeys[2]) { 
		zChange += 0.08f;
	}
	if (pressedKeys[3]) {
		zChange -= 0.08f;
	}
	if (pressedKeys[4]) { 
		yChange += 0.08f;
	}
	if (pressedKeys[5]) {
		yChange -= 0.08f;
	}
	if (pressedKeys[6]) { 
		cameraRotate += 0.016f * 90;
		cameraRotate = cameraRotate - 360 * (int)(cameraRotate / 360);
	}
	if (pressedKeys[7]) { 
		cameraRotate -= 0.016f * 90;
		cameraRotate = cameraRotate - 360 * (int)(cameraRotate / 360);
	}
	if (pressedKeys[8]) {
		cameraRotate2 += 0.016f * 90;
		cameraRotate2 = cameraRotate2 - 360 * (int)(cameraRotate2 / 360);
	}
	if (pressedKeys[9]) {
		cameraRotate2 -= 0.016f * 90;
		cameraRotate2 = cameraRotate2 - 360 * (int)(cameraRotate2 / 360);
	}
	float degree = cameraRotate * M_PI / 180;
	float sinRot = sin(degree);
	float cosRot = cos(degree);
	cameraPos.x += xChange * cosRot - zChange * sinRot;
	cameraPos.y += yChange;
	cameraPos.z += zChange * cosRot + xChange * sinRot;
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	Render(&cuda_pbo_resource);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glDrawPixels(512, 512, GL_RGBA, GL_FLOAT, BUFFER_OFFSET(0));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glutSwapBuffers();
}
void checkKeysDown(unsigned char key, int x, int y)
{
	if (!pressedKeys[0] && key == 'w') pressedKeys[0] = true;
	if (!pressedKeys[1] && key == 's') pressedKeys[1] = true;
	if (!pressedKeys[2] && key == 'd') pressedKeys[2] = true;
	if (!pressedKeys[3] && key == 'a') pressedKeys[3] = true;
	if (!pressedKeys[4] && key == 'e') pressedKeys[4] = true;
	if (!pressedKeys[5] && key == 'q') pressedKeys[5] = true;
	if (!pressedKeys[6] && key == 'l') pressedKeys[6] = true;
	if (!pressedKeys[7] && key == 'j') pressedKeys[7] = true;
	if (!pressedKeys[8] && key == 'o') pressedKeys[8] = true;
	if (!pressedKeys[9] && key == 'k') pressedKeys[9] = true;
	
}
void checkKeysUp(unsigned char key, int x, int y)
{
	if (key == 'w') pressedKeys[0] = false;
	if (key == 's') pressedKeys[1] = false;
	if (key == 'd') pressedKeys[2] = false;
	if (key == 'a') pressedKeys[3] = false;
	if (key == 'e') pressedKeys[4] = false;
	if (key == 'q') pressedKeys[5] = false;
	if (key == 'l') pressedKeys[6] = false;
	if (key == 'j') pressedKeys[7] = false;
	if (key == 'o') pressedKeys[8] = false;
	if (key == 'k') pressedKeys[9] = false;
}
void timerEvent(int value)
{
	if (glutGetWindow())
	{
		g_fAnim += 0.016f;

		//rotateFirstObject<<<1,1>>>(objects, 0.016f * 180);
		//changePositionOfFirstObject << <1, 1 >> > (objects, g_fAnim);
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

void deletePBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	cudaGraphicsUnregisterResource(vbo_res);

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}
void cleanup()
{

	if (pbo)
	{
		deletePBO(&pbo, cuda_pbo_resource);
	}
	unload_objects_to_kernel << <1, 1 >> > (objects);
	cudaFree(objects);
	cudaFree(pixelsCollided);
}


