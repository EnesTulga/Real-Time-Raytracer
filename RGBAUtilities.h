//
//  RGBAUtilities.h
//  RGBAUtilities
//
//  Initially created by Guillaume Cottenceau. 
//  Copyright 2002-2010 Guillaume Cottenceau.
//
//  Modified by Beau Johnston on 13/07/11.
//  Copyright 2011 Beau Johnston
//  This software may be freely redistributed under the terms
//  of the X11 license.
//

#ifndef RGBA_UTILS
#define RGBA_UTILS

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#define PNG_DEBUG 3

	//define our data types, the following numbers denote the number of bits
	//e.g. int8 uses a signed 8 bit
#define int8 signed char
#define int16 signed short
#define int32 signed int
#define int64 signed long
#define uint8 unsigned char
#define uint16 unsigned short
#define uint32 unsigned int
#define uint64 unsigned long

//supported for apple OSX and other variations of UNIX
#if defined (__APPLE__)  && defined (__MACH__)
#include <libpng/png.h>
#else
#include <png.h>
#endif

void abort_(const char * s, ...);
void read_png_file(char* file_name);
void write_png_file(char* file_name);
//an example on how to access pixel components of the file
void process_file(void);
//normalizing/denormalizing and testing normalization only works with the getImage/setImage functions and even then cannot currently be used due to rounding
uint8* normalizeImage(uint8*);
uint8* denormalizeImage(uint8*);
bool allPixelsAreNormal(uint8*);

uint8* getImage(void);
void setImage(uint8*);

//old less obvious way of doing things (nested uint8*[uint8*[]]) 2D array, useless for OpenCL
uint8* convolutedGetImage(void);
void convolutedSetImage(uint8*);

uint32 getImageLength(void);
uint32 getImageWidth(void);
uint32 getConfig(void);
uint32 getBitsPerSample(void);
uint32 getSamplesPerPixel(void);
uint32 getImageRowPitch(void);

#endif