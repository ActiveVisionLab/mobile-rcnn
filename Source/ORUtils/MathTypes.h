// Copyright 2014-2018 Oxford University Innovation Limited and the authors of ORUtils

#pragma once

#ifndef NULL
#define NULL 0
#endif

#if !defined(__METALC__) && !defined(__OPENCL_VERSION__)

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

#include "Vector.h"
#include "Matrix.h"

typedef class ORUtils::Matrix3<float> Matrix3f;
typedef class ORUtils::Matrix4<float> Matrix4f;

typedef class ORUtils::Vector2<short> Vector2s;
typedef class ORUtils::Vector2<int> Vector2i;
typedef class ORUtils::Vector2<unsigned int> Vector2ui;
typedef class ORUtils::Vector2<uint> Vector2ui;
typedef class ORUtils::Vector2<unsigned char> Vector2u;
typedef class ORUtils::Vector2<float> Vector2f;
typedef class ORUtils::Vector2<double> Vector2d;

typedef class ORUtils::Vector3<short> Vector3s;
typedef class ORUtils::Vector3<double> Vector3d;
typedef class ORUtils::Vector3<int> Vector3i;
typedef class ORUtils::Vector3<unsigned int> Vector3ui;
typedef class ORUtils::Vector3<uchar> Vector3u;
typedef class ORUtils::Vector3<float> Vector3f;

typedef class ORUtils::Vector4<float> Vector4f;
typedef class ORUtils::Vector4<double> Vector4d;
typedef class ORUtils::Vector4<int> Vector4i;
typedef class ORUtils::Vector4<uint> Vector4ui;
typedef class ORUtils::Vector4<short> Vector4s;
typedef class ORUtils::Vector4<uchar> Vector4u;

#endif

#if defined(__METALC__)

typedef metal::float3x3 Matrix3f;
typedef metal::float4x4 Matrix4f;

typedef metal::short2 Vector2s;
typedef metal::int2 Vector2i;
typedef metal::uint2 Vector2ui;
typedef metal::uchar2 Vector2u;
typedef metal::float2 Vector2f;

typedef metal::short3 Vector3s;
typedef metal::int3 Vector3i;
typedef metal::uchar3 Vector3u;
typedef metal::float3 Vector3f;

typedef metal::float4 Vector4f;
typedef metal::int4 Vector4i;
typedef metal::uint4 Vector4ui;
typedef metal::short4 Vector4s;
typedef metal::uchar4 Vector4u;

#endif

#if defined(__OPENCL_VERSION__)

typedef short2 Vector2s;
typedef int2 Vector2i;
typedef uint2 Vector2ui;
typedef float2 Vector2f;

typedef short3 Vector3s;
typedef int3 Vector3i;
typedef uchar3 Vector3u;
typedef float3 Vector3f;

typedef float4 Vector4f;
typedef int4 Vector4i;
typedef short4 Vector4s;
typedef uchar4 Vector4u;

typedef float16 Matrix4f;

#endif

#include "MathUtils.h"
