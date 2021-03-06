//
//  RGBAUtilities.cpp
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

#include "RGBAUtilities.h"

void abort_(const char * s, ...)
{
	va_list args;
	va_start(args, s);
	vfprintf(stderr, s, args);
	fprintf(stderr, "\n");
	va_end(args);
	abort();
}

int x, y;

int imageWidth, imageHeight;
png_byte color_type;
png_byte bit_depth;

png_structp png_ptr;
png_infop info_ptr;
int number_of_passes;
png_bytep * row_pointers;

uint32 _imageLength, _imageWidth, _config, _bitsPerSample, _samplesPerPixel, _bitsPerPixel, _imageBitSize, _imageSize;
uint64 _linebytes;

void read_png_file(char* file_name)
{
	char header[8];    // 8 is the maximum size that can be checked

	/* open file and test for it being a png */
	FILE *fp = fopen(file_name, "rb");
	if (!fp)
		abort_("[read_png_file] File %s could not be opened for reading", file_name);
	fread(header, 1, 8, fp);
	//    if (png_sig_cmp(header, 0, 8))
	//        abort_("[read_png_file] File %s is not recognized as a PNG file", file_name);
	//    

		/* initialize stuff */
	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr)
		abort_("[read_png_file] png_create_read_struct failed");

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
		abort_("[read_png_file] png_create_info_struct failed");

	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[read_png_file] Error during init_io");

	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);

	png_read_info(png_ptr, info_ptr);

	imageWidth = png_get_image_width(png_ptr, info_ptr);
	imageHeight = png_get_image_height(png_ptr, info_ptr);
	color_type = png_get_color_type(png_ptr, info_ptr);
	bit_depth = png_get_bit_depth(png_ptr, info_ptr);

	number_of_passes = png_set_interlace_handling(png_ptr);
	png_read_update_info(png_ptr, info_ptr);

	_imageWidth = png_get_image_width(png_ptr, info_ptr);
	_imageLength = png_get_image_height(png_ptr, info_ptr);
	_config = png_get_color_type(png_ptr, info_ptr);
	_bitsPerSample = png_get_bit_depth(png_ptr, info_ptr); // = 8 bits
	_samplesPerPixel = png_get_channels(png_ptr, info_ptr); // = 4 bytes

	_bitsPerPixel = _samplesPerPixel * _bitsPerSample;
	_linebytes = _samplesPerPixel * _imageWidth; // = 640
	//_linebytes = png_get_rowbytes(png_ptr, info_ptr); = 640
	_imageBitSize = (sizeof(uint8) * _imageWidth * _imageLength * _samplesPerPixel);
	_imageSize = _imageWidth * _imageLength * _samplesPerPixel;
	//printf("linebytes = %i, expected %i\n",_linebytes,png_get_rowbytes(png_ptr, info_ptr));
	//printf("Image Height is %d", sizeof(png_bytep) * imageHeight);


	/* read file */
	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[read_png_file] Error during read_image");

	row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * imageHeight);
	for (y = 0; y < imageHeight; y++)
		row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png_ptr, info_ptr));

	png_read_image(png_ptr, row_pointers);

	fclose(fp);
}


void write_png_file(char* file_name)
{
	/* create file */
	FILE *fp = fopen(file_name, "wb");
	if (!fp)
		abort_("[write_png_file] File %s could not be opened for writing", file_name);


	/* initialize stuff */
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (!png_ptr)
		abort_("[write_png_file] png_create_write_struct failed");

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr)
		abort_("[write_png_file] png_create_info_struct failed");

	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[write_png_file] Error during init_io");

	png_init_io(png_ptr, fp);


	/* write header */
	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[write_png_file] Error during writing header");

	png_set_IHDR(png_ptr, info_ptr, imageWidth, imageHeight,
		bit_depth, color_type, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);


	/* write bytes */
	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[write_png_file] Error during writing bytes");

	png_write_image(png_ptr, row_pointers);


	/* end write */
	if (setjmp(png_jmpbuf(png_ptr)))
		abort_("[write_png_file] Error during end of write");

	png_write_end(png_ptr, NULL);

	/* cleanup heap allocation */
	for (y = 0; y < imageHeight; y++)
		free(row_pointers[y]);
	free(row_pointers);

	fclose(fp);
}


void process_file(void)
{
	if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_RGB)
		abort_("[process_file] input file is PNG_COLOR_TYPE_RGB but must be PNG_COLOR_TYPE_RGBA "
			"(lacks the alpha channel)");

	if (png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_RGBA)
		abort_("[process_file] color_type of input file must be PNG_COLOR_TYPE_RGBA (%d) (is %d)",
			PNG_COLOR_TYPE_RGBA, png_get_color_type(png_ptr, info_ptr));

	for (y = 0; y < imageHeight; y++) {
		png_byte* row = row_pointers[y];
		for (x = 0; x < imageWidth; x++) {
			png_byte* ptr = &(row[x * 4]);
			//printf("Pixel at position [ %d - %d ] has RGBA values: %d - %d - %d - %d\n",
			//       x, y, ptr[0], ptr[1], ptr[2], ptr[3]);

			/* set red value to 0 and green value to the blue one */
			ptr[0] = 0;
			ptr[1] = ptr[2];
		}
	}
}

uint8* normalizeImage(uint8* input) {
	//with 8 bits this obvously causes a rounding error, usually down to 0, solve this by storing as floats
	uint8* output = new uint8[_imageBitSize];

	for (int i = 0; i < _imageSize; i++) {
		output[i] = (input[i] / pow(2, _bitsPerSample));
	}
	delete input;
	return output;
}

uint8* denormalizeImage(uint8*input) {
	//with 8 bits this obvously causes a rounding error, usually down to 0, solve this by storing as floats
	uint8* output = new uint8[_imageBitSize];

	for (int i = 0; i < _imageSize; i++) {
		output[i] = (input[i] * pow(2, _bitsPerSample));
	}
	delete input;
	return output;
}

bool allPixelsAreNormal(uint8* image) {
#ifdef DEBUG
	printf("bits per sample is %i", _bitsPerSample);
#endif
	for (int i = 0; i < _imageSize; i++) {
		if (image[i] > 1) {
#ifdef DEBUG
			printf("Not normal at point %i with %d", i, image[i]);
#endif
			return false;
		}
	}
	return true;
}

uint8* convolutedGetImage(void) {
	uint8* image = new uint8[(sizeof(uint8) * imageHeight)];

	if (png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_RGBA)
		abort_("[process_file] color_type of input file must be PNG_COLOR_TYPE_RGBA (%d) (is %d)",
			PNG_COLOR_TYPE_RGBA, png_get_color_type(png_ptr, info_ptr));

	for (y = 0; y < imageHeight; y++) {
		uint8* row = row_pointers[y];

		for (x = 0; x < imageWidth; x++) {
			uint8* ptr = &(row[x * 4]);
			image[y*imageWidth + x] = *ptr;
		}
	}

	return image;
}

uint8* getImage(void) {
	uint8* image = new uint8[(sizeof(uint8) * _imageWidth * _imageLength * _samplesPerPixel)];
	for (y = 0; y < _imageLength; y++) {
		uint8* row = row_pointers[y];
		int origX = 0;
		for (x = 0; x < _linebytes; x += 4) {
			uint8* ptr = &(row[origX * 4]);
			image[y*_linebytes + x + 0] = ptr[0];
			image[y*_linebytes + x + 1] = ptr[1];
			image[y*_linebytes + x + 2] = ptr[2];
			image[y*_linebytes + x + 3] = ptr[3];
			origX++;
		}
	}

	return image;
}

void convolutedSetImage(uint8* image) {
	for (y = 0; y < imageHeight; y++) {
		uint8* row = row_pointers[y];
		for (x = 0; x < imageWidth; x++) {
			uint8* ptr = &(row[x * 4]);
			*ptr = image[y*imageWidth + x];
		}
	}

	delete image;

}

void setImage(uint8* image) {

	for (y = 0; y < _imageLength; y++) {
		uint8* row = row_pointers[y];
		int origX = 0;
		for (x = 0; x < _linebytes; x += 4) {
			uint8* ptr = &(row[origX * 4]);
			ptr[0] = image[y*_linebytes + x + 0];
			ptr[1] = image[y*_linebytes + x + 1];
			ptr[2] = image[y*_linebytes + x + 2];
			ptr[3] = image[y*_linebytes + x + 3];
			origX++;
		}
	}

	delete image;
}


uint32 getImageLength(void) {
	return _imageLength;
};

uint32 getImageWidth(void) {
	return _imageWidth;
};

uint32 getConfig(void) {
	return _config;
};

uint32 getBitsPerSample(void) {
	return _bitsPerSample;
};

uint32 getSamplesPerPixel(void) {
	return _samplesPerPixel;
};

uint32 getImageRowPitch(void) {
	return _samplesPerPixel * _imageWidth;
};
