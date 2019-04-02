// Copyright 2014-2018 Oxford University Innovation Limited and the authors of ORUtils

#pragma once

#if !defined(__METALC__) && !defined(__OPENCL_VERSION__)
#include <math.h>
#endif

#include "MathTypes.h"
#include "PlatformIndependence.h"

_CPU_AND_GPU_CODE_ inline Vector4f interpolateBilinear_4f(const CONSTPTR(Vector4f) *source, Vector2f position, Vector2i imgSize)
{
	const Vector2s p = TO_SHORT_FLOOR2(position);
	const Vector2f delta = position - TO_FLOAT2(p);

	Vector4f a = source[p.x + p.y * imgSize.x];
	Vector4f b = 0.0f, c = 0.0f, d = 0.0f;

	if (delta.x != 0) b = source[(p.x + 1) + p.y * imgSize.x];
	if (delta.y != 0) c = source[p.x + (p.y + 1) * imgSize.x];
	if (delta.x != 0 && delta.y != 0) d = source[(p.x + 1) + (p.y + 1) * imgSize.x];

	Vector4f result;
	result.x = (a.x * (1.0f - delta.x) * (1.0f - delta.y) + b.x * delta.x * (1.0f - delta.y) +
		c.x * (1.0f - delta.x) * delta.y + d.x * delta.x * delta.y);
	result.y = (a.y * (1.0f - delta.x) * (1.0f - delta.y) + b.y * delta.x * (1.0f - delta.y) +
		c.y * (1.0f - delta.x) * delta.y + d.y * delta.x * delta.y);
	result.z = (a.z * (1.0f - delta.x) * (1.0f - delta.y) + b.z * delta.x * (1.0f - delta.y) +
		c.z * (1.0f - delta.x) * delta.y + d.z * delta.x * delta.y);
	result.w = (a.w * (1.0f - delta.x) * (1.0f - delta.y) + b.w * delta.x * (1.0f - delta.y) +
		c.w * (1.0f - delta.x) * delta.y + d.w * delta.x * delta.y);

	return result;
}

_CPU_AND_GPU_CODE_ inline Vector4f interpolateBilinear_4fu(const CONSTPTR(Vector4u) *source, Vector2f position, Vector2i imgSize)
{
	const Vector2s p = TO_SHORT_FLOOR2(position);
	const Vector2f delta = position - TO_FLOAT2(p);

	Vector4u a = source[p.x + p.y * imgSize.x];
	Vector4u b = CREATE_VECTOR4u(0, 0, 0, 0), c = CREATE_VECTOR4u(0, 0, 0, 0), d = CREATE_VECTOR4u(0, 0, 0, 0);

	if (delta.x != 0 && p.x < imgSize.x - 1) b = source[(p.x + 1) + p.y * imgSize.x];
	if (delta.y != 0 && p.y < imgSize.y - 1) c = source[p.x + (p.y + 1) * imgSize.x];
	if (delta.x != 0 && delta.y != 0 && (p.x < imgSize.x - 1) && (p.y < imgSize.y - 1)) d = source[(p.x + 1) + (p.y + 1) * imgSize.x];

	Vector4f result;
	result.x = ((float)a.x * (1.0f - delta.x) * (1.0f - delta.y) + (float)b.x * delta.x * (1.0f - delta.y) +
		(float)c.x * (1.0f - delta.x) * delta.y + (float)d.x * delta.x * delta.y);
	result.y = ((float)a.y * (1.0f - delta.x) * (1.0f - delta.y) + (float)b.y * delta.x * (1.0f - delta.y) +
		(float)c.y * (1.0f - delta.x) * delta.y + (float)d.y * delta.x * delta.y);
	result.z = ((float)a.z * (1.0f - delta.x) * (1.0f - delta.y) + (float)b.z * delta.x * (1.0f - delta.y) +
		(float)c.z * (1.0f - delta.x) * delta.y + (float)d.z * delta.x * delta.y);
	result.w = ((float)a.w * (1.0f - delta.x) * (1.0f - delta.y) + (float)b.w * delta.x * (1.0f - delta.y) +
		(float)c.w * (1.0f - delta.x) * delta.y + (float)d.w * delta.x * delta.y);

	return result;
}

_CPU_AND_GPU_CODE_ inline Vector4f interpolateBilinear_4f_checkbounds(const CONSTPTR(Vector4f) *source, Vector2f position, Vector2i imgSize)
{
	const Vector2s p = TO_SHORT_FLOOR2(position);
	const Vector2f delta = position - TO_FLOAT2(p);

	Vector4f a = source[p.x + p.y * imgSize.x];
	Vector4f b = CREATE_VECTOR4f(0.0f, 0.0f, 0.0f, 0.0f), c = CREATE_VECTOR4f(0.0f, 0.0f, 0.0f, 0.0f), d = CREATE_VECTOR4f(0.0f, 0.0f, 0.0f, 0.0f);

	if (delta.x != 0 && p.x < imgSize.x - 1) b = source[(p.x + 1) + p.y * imgSize.x];
	if (delta.y != 0 && p.y < imgSize.y - 1) c = source[p.x + (p.y + 1) * imgSize.x];
	if (delta.x != 0 && delta.y != 0 && (p.x < imgSize.x - 1) && (p.y < imgSize.y - 1)) d = source[(p.x + 1) + (p.y + 1) * imgSize.x];

	Vector4f result;
	result.x = (a.x * (1.0f - delta.x) * (1.0f - delta.y) + b.x * delta.x * (1.0f - delta.y) +
		c.x * (1.0f - delta.x) * delta.y + d.x * delta.x * delta.y);
	result.y = (a.y * (1.0f - delta.x) * (1.0f - delta.y) + b.y * delta.x * (1.0f - delta.y) +
		c.y * (1.0f - delta.x) * delta.y + d.y * delta.x * delta.y);
	result.z = (a.z * (1.0f - delta.x) * (1.0f - delta.y) + b.z * delta.x * (1.0f - delta.y) +
		c.z * (1.0f - delta.x) * delta.y + d.z * delta.x * delta.y);
	result.w = (a.w * (1.0f - delta.x) * (1.0f - delta.y) + b.w * delta.x * (1.0f - delta.y) +
		c.w * (1.0f - delta.x) * delta.y + d.w * delta.x * delta.y);

	return result;
}

_CPU_AND_GPU_CODE_ inline float interpolateBilinear_1f(const CONSTPTR(float) *source, Vector2f position, Vector2i imgSize)
{
	const Vector2s p = TO_SHORT_FLOOR2(position);
	const Vector2f delta = position - TO_FLOAT2(p);

	float a = source[p.x + p.y * imgSize.x];
	float b = 0.0f, c = 0.0f, d = 0.0f;

	if (delta.x != 0) b = source[(p.x + 1) + p.y * imgSize.x];
	if (delta.y != 0) c = source[p.x + (p.y + 1) * imgSize.x];
	if (delta.x != 0 && delta.y != 0) d = source[(p.x + 1) + (p.y + 1) * imgSize.x];

	float result = (a * (1.0f - delta.x) * (1.0f - delta.y) + b * delta.x * (1.0f - delta.y) +
		c * (1.0f - delta.x) * delta.y + d * delta.x * delta.y);

	return result;
}

_CPU_AND_GPU_CODE_ inline float interpolateBilinear_1f_checkbounds(const CONSTPTR(float) *source, Vector2f position, Vector2i imgSize)
{
	const Vector2s p = TO_SHORT_FLOOR2(position);
	const Vector2f delta = position - TO_FLOAT2(p);
	
	float a = 0.0f; ;
	float b = 0.0f, c = 0.0f, d = 0.0f;
	
	if (p.x >= 0 && p.y >= 0) a = source[p.x + p.y * imgSize.x];
	if (delta.x != 0 && p.x < imgSize.x - 1) b = source[(p.x + 1) + p.y * imgSize.x];
	if (delta.y != 0 && p.y < imgSize.y - 1) c = source[p.x + (p.y + 1) * imgSize.x];
	if (delta.x != 0 && delta.y != 0 && (p.x < imgSize.x - 1) && (p.y < imgSize.y - 1))
		d = source[(p.x + 1) + (p.y + 1) * imgSize.x];
	
	float result = (a * (1.0f - delta.x) * (1.0f - delta.y) + b * delta.x * (1.0f - delta.y) +
					c * (1.0f - delta.x) * delta.y + d * delta.x * delta.y);
	
	return result;
}

_CPU_AND_GPU_CODE_ inline Vector2f interpolateBilinear_2f(const CONSTPTR(Vector2f) *source, Vector2f position, Vector2i imgSize)
{
	const Vector2s p = TO_SHORT_FLOOR2(position);
	const Vector2f delta = position - TO_FLOAT2(p);

	Vector2f a = source[p.x + p.y * imgSize.x];
	Vector2f b = 0.0f, c = 0.0f, d = 0.0f;

	if (delta.x != 0) b = source[(p.x + 1) + p.y * imgSize.x];
	if (delta.y != 0) c = source[p.x + (p.y + 1) * imgSize.x];
	if (delta.x != 0 && delta.y != 0) d = source[(p.x + 1) + (p.y + 1) * imgSize.x];

	if (a.x == 0.0f || b.x == 0.0f || c.x == 0.0f || d.x == 0.0f ||
		a.y == 0.0f || b.y == 0.0f || c.y == 0.0f || d.y == 0.0f) return 0.0f;

	Vector2f result;
	result.x = (a.x * (1.0f - delta.x) * (1.0f - delta.y) + b.x * delta.x * (1.0f - delta.y) +
		c.x * (1.0f - delta.x) * delta.y + d.x * delta.x * delta.y);
	result.y = (a.y * (1.0f - delta.x) * (1.0f - delta.y) + b.y * delta.x * (1.0f - delta.y) +
		c.y * (1.0f - delta.x) * delta.y + d.y * delta.x * delta.y);

	return result;
}

_CPU_AND_GPU_CODE_ inline Vector4f interpolateBilinear_4f_holes(const CONSTPTR(Vector4f) *source, Vector2f position, Vector2i imgSize)
{
	const Vector2s p = TO_SHORT_FLOOR2(position);
	const Vector2f delta = position - TO_FLOAT2(p);

	Vector4f a = source[p.x + p.y * imgSize.x];
	Vector4f b = source[(p.x + 1) + p.y * imgSize.x];
	Vector4f c = source[p.x + (p.y + 1) * imgSize.x];
	Vector4f d = source[(p.x + 1) + (p.y + 1) * imgSize.x];

	Vector4f result;
	if (a.w < 0 || b.w < 0 || c.w < 0 || d.w < 0)
	{
		result.x = 0; result.y = 0; result.z = 0; result.w = -1.0f;
		return result;
	}

	result.x = (a.x * (1.0f - delta.x) * (1.0f - delta.y) + b.x * delta.x * (1.0f - delta.y) +
		c.x * (1.0f - delta.x) * delta.y + d.x * delta.x * delta.y);
	result.y = (a.y * (1.0f - delta.x) * (1.0f - delta.y) + b.y * delta.x * (1.0f - delta.y) +
		c.y * (1.0f - delta.x) * delta.y + d.y * delta.x * delta.y);
	result.z = (a.z * (1.0f - delta.x) * (1.0f - delta.y) + b.z * delta.x * (1.0f - delta.y) +
		c.z * (1.0f - delta.x) * delta.y + d.z * delta.x * delta.y);
	result.w = (a.w * (1.0f - delta.x) * (1.0f - delta.y) + b.w * delta.x * (1.0f - delta.y) +
		c.w * (1.0f - delta.x) * delta.y + d.w * delta.x * delta.y);

	return result;
}

_CPU_AND_GPU_CODE_ inline float interpolateBilinear_1f_holes(const CONSTPTR(float) *source, Vector2f position, Vector2i imgSize)
{
	const Vector2s p = TO_SHORT_FLOOR2(position);
	const Vector2f delta = position - TO_FLOAT2(p);

	float a = source[p.x + p.y * imgSize.x];
	float b = 0.0f, c = 0.0f, d = 0.0f;

	if (delta.x != 0) b = source[(p.x + 1) + p.y * imgSize.x];
	if (delta.y != 0) c = source[p.x + (p.y + 1) * imgSize.x];
	if (delta.x != 0 && delta.y != 0) d = source[(p.x + 1) + (p.y + 1) * imgSize.x];

	if (a <= 0 || b <= 0 || c <= 0 || d <= 0) return -1;

	float result = (a * (1.0f - delta.x) * (1.0f - delta.y) + b * delta.x * (1.0f - delta.y) +
		c * (1.0f - delta.x) * delta.y + d * delta.x * delta.y);

	return result;
}

_CPU_AND_GPU_CODE_ inline void selectSubsampleWithHoles_4f(DEVICEPTR(Vector4f) *imageData_out, int x, int y, Vector2i newDims,
	const CONSTPTR(Vector4f) *imageData_in, Vector2i oldDims)
{
	imageData_out[x + y * newDims.x] = imageData_in[x * 2 + y * 2 * oldDims.x];
}

_CPU_AND_GPU_CODE_ inline void selectSubsampleWithHoles_1f(DEVICEPTR(float) *imageData_out, int x, int y, Vector2i newDims,
	const CONSTPTR(float) *imageData_in, Vector2i oldDims)
{
	imageData_out[x + y * newDims.x] = imageData_in[x * 2 + y * 2 * oldDims.x];
}

_CPU_AND_GPU_CODE_ inline void convertColourToNormalisedIntensity(DEVICEPTR(float) *imageData_out, int x, int y, Vector2i dims,
	const CONSTPTR(Vector4u) *imageData_in)
{
	const int linear_pos = y * dims.x + x;
	const Vector4u colour = imageData_in[linear_pos];

	imageData_out[linear_pos] = (0.299f * colour.x + 0.587f * colour.y + 0.114f * colour.z) / 255.f;
}

_CPU_AND_GPU_CODE_ inline void gradientX(DEVICEPTR(Vector4s) *grad, int x, int y, const CONSTPTR(Vector4u) *image, Vector2i imgSize)
{
	Vector4s d1, d2, d3, d_out;

	d1.x = image[(x + 1) + (y - 1) * imgSize.x].x - image[(x - 1) + (y - 1) * imgSize.x].x;
	d1.y = image[(x + 1) + (y - 1) * imgSize.x].y - image[(x - 1) + (y - 1) * imgSize.x].y;
	d1.z = image[(x + 1) + (y - 1) * imgSize.x].z - image[(x - 1) + (y - 1) * imgSize.x].z;

	d2.x = image[(x + 1) + (y)* imgSize.x].x - image[(x - 1) + (y)* imgSize.x].x;
	d2.y = image[(x + 1) + (y)* imgSize.x].y - image[(x - 1) + (y)* imgSize.x].y;
	d2.z = image[(x + 1) + (y)* imgSize.x].z - image[(x - 1) + (y)* imgSize.x].z;

	d3.x = image[(x + 1) + (y + 1) * imgSize.x].x - image[(x - 1) + (y + 1) * imgSize.x].x;
	d3.y = image[(x + 1) + (y + 1) * imgSize.x].y - image[(x - 1) + (y + 1) * imgSize.x].y;
	d3.z = image[(x + 1) + (y + 1) * imgSize.x].z - image[(x - 1) + (y + 1) * imgSize.x].z;

	d1.w = d2.w = d3.w = 2 * 255;

	d_out.x = (d1.x + 2 * d2.x + d3.x) / 8;
	d_out.y = (d1.y + 2 * d2.y + d3.y) / 8;
	d_out.z = (d1.z + 2 * d2.z + d3.z) / 8;
	d_out.w = (d1.w + 2 * d2.w + d3.w) / 8;

	grad[x + y * imgSize.x] = d_out;
}

_CPU_AND_GPU_CODE_ inline void gradientY(DEVICEPTR(Vector4s) *grad, int x, int y, const CONSTPTR(Vector4u) *image, Vector2i imgSize)
{
	Vector4s d1, d2, d3, d_out;

	d1.x = image[(x - 1) + (y + 1) * imgSize.x].x - image[(x - 1) + (y - 1) * imgSize.x].x;
	d1.y = image[(x - 1) + (y + 1) * imgSize.x].y - image[(x - 1) + (y - 1) * imgSize.x].y;
	d1.z = image[(x - 1) + (y + 1) * imgSize.x].z - image[(x - 1) + (y - 1) * imgSize.x].z;

	d2.x = image[(x)+(y + 1) * imgSize.x].x - image[(x)+(y - 1) * imgSize.x].x;
	d2.y = image[(x)+(y + 1) * imgSize.x].y - image[(x)+(y - 1) * imgSize.x].y;
	d2.z = image[(x)+(y + 1) * imgSize.x].z - image[(x)+(y - 1) * imgSize.x].z;

	d3.x = image[(x + 1) + (y + 1) * imgSize.x].x - image[(x + 1) + (y - 1) * imgSize.x].x;
	d3.y = image[(x + 1) + (y + 1) * imgSize.x].y - image[(x + 1) + (y - 1) * imgSize.x].y;
	d3.z = image[(x + 1) + (y + 1) * imgSize.x].z - image[(x + 1) + (y - 1) * imgSize.x].z;

	d1.w = d2.w = d3.w = 2 * 255;

	d_out.x = (d1.x + 2 * d2.x + d3.x) / 8;
	d_out.y = (d1.y + 2 * d2.y + d3.y) / 8;
	d_out.z = (d1.z + 2 * d2.z + d3.z) / 8;
	d_out.w = (d1.w + 2 * d2.w + d3.w) / 8;

	grad[x + y * imgSize.x] = d_out;
}

_CPU_AND_GPU_CODE_ inline void gradientXY(DEVICEPTR(Vector2f) *grad, int x, int y, const CONSTPTR(float) *image, Vector2i imgSize)
{
	Vector2f d1, d2, d3, d_out;

	// Compute gradient in the X direction
	d1.x = image[(y - 1) * imgSize.x + (x + 1)] - image[(y - 1) * imgSize.x + (x - 1)];
	d2.x = image[(y)* imgSize.x + (x + 1)] - image[(y)* imgSize.x + (x - 1)];
	d3.x = image[(y + 1) * imgSize.x + (x + 1)] - image[(y + 1) * imgSize.x + (x - 1)];

	// Compute gradient in the Y direction
	d1.y = image[(y + 1) * imgSize.x + (x - 1)] - image[(y - 1) * imgSize.x + (x - 1)];
	d2.y = image[(y + 1) * imgSize.x + (x)] - image[(y - 1) * imgSize.x + (x)];
	d3.y = image[(y + 1) * imgSize.x + (x + 1)] - image[(y - 1) * imgSize.x + (x + 1)];

	d_out.x = (d1.x + 2.f * d2.x + d3.x) / 8.f;
	d_out.y = (d1.y + 2.f * d2.y + d3.y) / 8.f;

	grad[y * imgSize.x + x] = d_out;
}

#ifndef __OPENCL_VERSION__

template <typename T>
_CPU_AND_GPU_CODE_ inline void boxFilter2x2(DEVICEPTR(T) *imageData_out, int x_out, int y_out, Vector2i newDims,
	const CONSTPTR(T) *imageData_in, int x_in, int y_in, Vector2i oldDims)
{
	T pixel_out = T((uchar)(0));

	pixel_out += imageData_in[(x_in + 0) + (y_in + 0) * oldDims.x] / 4;
	pixel_out += imageData_in[(x_in + 1) + (y_in + 0) * oldDims.x] / 4;
	pixel_out += imageData_in[(x_in + 0) + (y_in + 1) * oldDims.x] / 4;
	pixel_out += imageData_in[(x_in + 1) + (y_in + 1) * oldDims.x] / 4;

	imageData_out[x_out + y_out * newDims.x] = pixel_out;
}

template <typename T>
_CPU_AND_GPU_CODE_ inline void boxFilter1x2(DEVICEPTR(T) *imageData_out, int x_out, int y_out, Vector2i newDims,
	const CONSTPTR(T) *imageData_in, int x_in, int y_in, Vector2i oldDims)
{
	T pixel_out = T((uchar)(0));

	pixel_out += imageData_in[(x_in + 0) + (y_in + 0) * oldDims.x] / 2;
	pixel_out += imageData_in[(x_in + 1) + (y_in + 0) * oldDims.x] / 2;

	imageData_out[x_out + y_out * newDims.x] = pixel_out;
}

template <typename T>
_CPU_AND_GPU_CODE_ inline void boxFilter2x1(DEVICEPTR(T) *imageData_out, int x_out, int y_out, Vector2i newDims,
	const CONSTPTR(T) *imageData_in, int x_in, int y_in, Vector2i oldDims)
{
	T pixel_out = T((uchar)(0));

	pixel_out += imageData_in[(x_in + 0) + (y_in + 0) * oldDims.x] / 2;
	pixel_out += imageData_in[(x_in + 0) + (y_in + 1) * oldDims.x] / 2;

	imageData_out[x_out + y_out * newDims.x] = pixel_out;
}

template <typename T>
_CPU_AND_GPU_CODE_ inline void boxFilter1x1(DEVICEPTR(T) *imageData_out, int x_out, int y_out, Vector2i newDims,
	const CONSTPTR(T) *imageData_in, int x_in, int y_in, Vector2i oldDims)
{
	T pixel_out = T((uchar)(0));
	pixel_out += imageData_in[(x_in + 0) + (y_in + 0) * oldDims.x];
	imageData_out[x_out + y_out * newDims.x] = pixel_out;
}

#endif
