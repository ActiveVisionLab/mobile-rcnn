// Copyright 2014-2018 Oxford University Innovation Limited and the authors of ORUtils

#pragma once

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290E-07F
#endif

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif


#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) (((a) < (b)) ? (b) : (a))
#endif

#ifndef ABS
#define ABS(a) (((a) < 0) ? (-(a)) : (a))
#endif

#ifndef FABS
#define FABS(a) fabs(a)
#endif

#ifndef ROUND
#define ROUND(x) ((x < 0) ? (x - 0.5f) : (x + 0.5f))
#endif

#if !defined(__OPENCL_VERSION__)
#ifndef PI
#define PI ((float)(3.1415926535897932384626433832795))
#endif
#else
#ifndef PI
#define PI ((float)(3.1415926535897932384626433832795))
#endif
#endif

#ifndef DEGTORAD
#define DEGTORAD ((float)(0.017453292519943295769236907684886))
#endif

#if !defined(__METALC__) && !defined(__OPENCL_VERSION__)

#ifndef CLAMP
#define CLAMP(x,a,b) MAX((a), MIN((b), (x)))
#endif

#ifndef CREATE_VECTOR2i
#define CREATE_VECTOR2i(x, y) (Vector2i((x), (y)))
#endif

#ifndef CREATE_VECTOR2ui
#define CREATE_VECTOR2ui(x, y) (Vector2ui((x), (y)))
#endif

#ifndef CREATE_VECTOR2f
#define CREATE_VECTOR2f(x, y) (Vector2f((x), (y)))
#endif

#ifndef CREATE_VECTOR3i
#define CREATE_VECTOR3i(x, y, z) (Vector3i((x), (y), (z)))
#endif

#ifndef CREATE_VECTOR3f
#define CREATE_VECTOR3f(x, y, z) (Vector3f((x), (y), (z)))
#endif

#ifndef CREATE_VECTOR3u
#define CREATE_VECTOR3u(x, y, z) (Vector3u((x), (y), (z)))
#endif

#ifndef CREATE_VECTOR4u
#define CREATE_VECTOR4u(x, y, z, w) (Vector4u((x), (y), (z), (w)))
#endif

#ifndef CREATE_VECTOR4i
#define CREATE_VECTOR4i(x, y, z, w) (Vector4i((x), (y), (z), (w)))
#endif

#ifndef CREATE_VECTOR4f
#define CREATE_VECTOR4f(x, y, z, w) (Vector4f((x), (y), (z), (w)))
#endif

#ifndef TO_INT_ROUND3
#define TO_INT_ROUND3(x) (x).toIntRound()
#endif

#ifndef TO_INT_ROUND4
#define TO_INT_ROUND4(x) (x).toIntRound()
#endif

#ifndef TO_INT_FLOOR3
#define TO_INT_FLOOR3(in) (in).toIntFloor()
#endif

#ifndef TO_INT_FLOOR3_WITH_REMAINDER
#define TO_INT_FLOOR3_WITH_REMAINDER(inted, coeffs, in) inted = (in).toIntFloor(coeffs)
#endif

#ifndef TO_SHORT_FLOOR2
#define TO_SHORT_FLOOR2(x) (x).toShortFloor()
#endif

#ifndef TO_SHORT_FLOOR3
#define TO_SHORT_FLOOR3(x) (x).toShortFloor()
#endif

#ifndef TO_UCHAR3
#define TO_UCHAR3(x) (x).toUChar()
#endif

#ifndef TO_UCHAR4
#define TO_UCHAR4(x) (x).toUChar()
#endif

#ifndef TO_FLOAT2
#define TO_FLOAT2(x) (x).toFloat()
#endif

#ifndef TO_FLOAT3
#define TO_FLOAT3(x) (x).toFloat()
#endif

#ifndef TO_FLOAT4
#define TO_FLOAT4(x) (x).toFloat()
#endif

#ifndef TO_VECTOR3
#define TO_VECTOR3(a) (a).toVector3()
#endif

#ifndef IS_EQUAL3
#define IS_EQUAL3(a,b) (((a).x == (b).x) && ((a).y == (b).y) && ((a).z == (b).z))
#endif

#ifndef TO_HOMOGENEOUS_SHORT4
#define TO_HOMOGENEOUS_SHORT4(in) Vector4s(in.x, in.y, in.z, 1)
#endif

#ifndef TO_HOMOGENEOUS_FLOAT4
#define TO_HOMOGENEOUS_FLOAT4(in) Vector4f(in.x, in.y, in.z, 1)
#endif

#ifndef MATRIXxVECTOR4
#define MATRIXxVECTOR4(m,v) (m * v)
#endif

inline bool portable_finite(float a)
{
	volatile float temp = a;
	if (temp != a) return false;
	if ((temp - a) != 0.0) return false;
	return true;
}

inline void matmul(const float *A, const float *b, float *x, int numRows, int numCols)
{
	for (int r = 0; r < numRows; ++r)
	{
		float res = 0.0f;
		for (int c = 0; c < numCols; ++c) res += A[r*numCols + c] * b[c];
		x[r] = res;
	}
}

#elif defined(__METALC__)

#ifndef CLAMP
#define CLAMP(x,a,b) clamp(x, a, b)
#endif

#ifndef FABS
#define FABS(a) fabs(a)
#endif

#ifndef TO_INT_ROUND3
#define TO_INT_ROUND3(p) (static_cast<metal::int3>(round(p)))
#endif

#ifndef TO_INT_ROUND4
#define TO_INT_ROUND4(p) (static_cast<metal::int4>(round(p)))
#endif

#ifndef TO_INT_FLOOR3
#define TO_INT_FLOOR3(p) (static_cast<metal::int3>(floor(p)))
#endif

#ifndef CREATE_VECTOR2i
#define CREATE_VECTOR2i(x, y) (Vector2i((x), (y)))
#endif

#ifndef CREATE_VECTOR2ui
#define CREATE_VECTOR2ui(x, y) (Vector2ui((x), (y)))
#endif

#ifndef CREATE_VECTOR2f
#define CREATE_VECTOR2f(x, y) (Vector2f((x), (y)))
#endif

#ifndef CREATE_VECTOR3i
#define CREATE_VECTOR3i(x, y, z) (Vector3i((x), (y), (z)))
#endif

#ifndef CREATE_VECTOR3f
#define CREATE_VECTOR3f(x, y, z) (Vector3f((x), (y), (z)))
#endif

#ifndef CREATE_VECTOR3u
#define CREATE_VECTOR3u(x, y, z) (Vector3u((x), (y), (z)))
#endif

#ifndef CREATE_VECTOR4u
#define CREATE_VECTOR4u(x, y, z, w) (Vector4u((x), (y), (z), (w)))
#endif

#ifndef CREATE_VECTOR4i
#define CREATE_VECTOR4i(x, y, z, w) (Vector4i((x), (y), (z), (w)))
#endif

#ifndef CREATE_VECTOR4f
#define CREATE_VECTOR4f(x, y, z, w) (Vector4f((x), (y), (z), (w)))
#endif

#ifndef TO_INT_FLOOR3_WITH_REMAINDER
#define TO_INT_FLOOR3_WITH_REMAINDER(inted, coeffs, in) { Vector3f flored(floor(in.x), floor(in.y), floor(in.z)); coeffs = in - flored; inted = Vector3i((int)flored.x, (int)flored.y, (int)flored.z); }
#endif

#ifndef TO_SHORT_FLOOR3
#define TO_SHORT_FLOOR3(x) (static_cast<metal::short3>(floor(x)))
#endif

#ifndef TO_SHORT_FLOOR2
#define TO_SHORT_FLOOR2(x) (static_cast<metal::short2>(floor(x)))
#endif

#ifndef TO_UCHAR3
#define TO_UCHAR3(x) (static_cast<metal::uchar3>(x))
#endif

#ifndef TO_UCHAR4
#define TO_UCHAR4(x) (static_cast<metal::uchar4>(x))
#endif

#ifndef TO_FLOAT2
#define TO_FLOAT2(x) (static_cast<metal::float2>(x))
#endif

#ifndef TO_FLOAT3
#define TO_FLOAT3(x) (static_cast<metal::float3>(x))
#endif

#ifndef TO_FLOAT4
#define TO_FLOAT4(x) (static_cast<metal::float4>(x))
#endif

#ifndef TO_VECTOR3
#define TO_VECTOR3(a) ((a).xyz)
#endif

#ifndef TO_HOMOGENEOUS_SHORT4
#define TO_HOMOGENEOUS_SHORT4(in) Vector4s(in.x, in.y, in.z, 1)
#endif

#ifndef TO_HOMOGENEOUS_FLOAT4
#define TO_HOMOGENEOUS_FLOAT4(in) Vector4f(in.x, in.y, in.z, 1)
#endif

#ifndef IS_EQUAL3
#define IS_EQUAL3(a,b) (((a).x == (b).x) && ((a).y == (b).y) && ((a).z == (b).z))
#endif

#ifndef MATRIXxVECTOR4
#define MATRIXxVECTOR4(m,v) (m * v)
#endif

#elif defined(__OPENCL_VERSION__)

#ifndef CLAMP
#define CLAMP(x,a,b) MAX((a), MIN((b), (x)))
#endif

#ifndef FABS
#define FABS(a) fabs(a)
#endif

#ifndef TO_VECTOR3
#define TO_VECTOR3(a) ((a).xyz)
#endif

#ifndef IS_EQUAL3
#define IS_EQUAL3(a,b) (((a).x == (b).x) && ((a).y == (b).y) && ((a).z == (b).z))
#endif

#ifndef CREATE_VECTOR2i
#define CREATE_VECTOR2i(x, y) ((Vector2i)((x), (y)))
#endif

#ifndef CREATE_VECTOR2ui
#define CREATE_VECTOR2ui(x, y) ((Vector2ui)((x), (y)))
#endif

#ifndef CREATE_VECTOR2f
#define CREATE_VECTOR2f(x, y) ((Vector2f)((x), (y)))
#endif

#ifndef CREATE_VECTOR3i
#define CREATE_VECTOR3i(x, y, z) ((Vector3i)((x), (y), (z)))
#endif

#ifndef CREATE_VECTOR3f
#define CREATE_VECTOR3f(x, y, z) ((Vector3f)((x), (y), (z)))
#endif

#ifndef CREATE_VECTOR3u
#define CREATE_VECTOR3u(x, y, z) ((Vector3u)((x), (y), (z)))
#endif

#ifndef CREATE_VECTOR4u
#define CREATE_VECTOR4u(x, y, z, w) ((Vector4u)((x), (y), (z), (w)))
#endif

#ifndef CREATE_VECTOR4i
#define CREATE_VECTOR4i(x, y, z, w) ((Vector4i)((x), (y), (z), (w)))
#endif

#ifndef CREATE_VECTOR4f
#define CREATE_VECTOR4f(x, y, z, w) ((Vector4f)((x), (y), (z), (w)))
#endif

#ifndef TO_UCHAR3
#define TO_UCHAR3(p) ((Vector3u)((p).x, (p).y, (p).z))
#endif

#ifndef TO_UCHAR4
#define TO_UCHAR4(p) ((Vector4u)((p).x, (p).y, (p).z, (p).w))
#endif

#ifndef TO_FLOAT2
#define TO_FLOAT2(p) ((Vector2f)((p).x, (p).y))
#endif

#ifndef TO_FLOAT3
#define TO_FLOAT3(p) ((Vector3f)((p).x, (p).y, (p).z))
#endif

#ifndef TO_FLOAT4
#define TO_FLOAT4(p) ((Vector4f)((p).x, (p).y, (p).z, (p).w))
#endif

#ifndef TO_SHORT_FLOOR2
#define TO_SHORT_FLOOR2(p) ((Vector2s)(floor((p).x), floor((p).y)))
#endif

#ifndef TO_SHORT_FLOOR3
#define TO_SHORT_FLOOR3(p) ((Vector3s)(floor((p).x), floor((p).y), floor((p).z)))
#endif

#ifndef TO_INT_FLOOR3
#define TO_INT_FLOOR3(p) ((Vector3i)(floor((p).x), floor((p).y), floor((p).z)))
#endif

#ifndef TO_INT_FLOOR3_WITH_REMAINDER
#define TO_INT_FLOOR3_WITH_REMAINDER(inted, coeffs, in) { Vector3f flored = (Vector3f)(floor(in.x), floor(in.y), floor(in.z)); coeffs = in - flored; inted = (Vector3i)((int)flored.x, (int)flored.y, (int)flored.z); }
#endif

#ifndef TO_INT_ROUND3
#define TO_INT_ROUND3(p) ((Vector3i)(round((p).x), round((p).y), round((p).z)))
#endif

#ifndef TO_HOMOGENEOUS_SHORT4
#define TO_HOMOGENEOUS_SHORT4(p) ((Vector4s)((p).x, (p).y, (p).z, 1))
#endif

#ifndef TO_HOMOGENEOUS_FLOAT4
#define TO_HOMOGENEOUS_FLOAT4(p) ((Vector4f)((p).x, (p).y, (p).z, 1.0f))
#endif

#ifndef MATRIXxVECTOR4
#define MATRIXxVECTOR4(m,v) multMatrixVector4(&m, &v)
#endif

inline Vector4f multMatrixVector4(const Matrix4f *m, const Vector4f *v)
{
	Vector4f r;

	r.x = m->s0 * v->x + m->s4 * v->y + m->s8 * v->z + m->sc * v->w;
	r.y = m->s1 * v->x + m->s5 * v->y + m->s9 * v->z + m->sd * v->w;
	r.z = m->s2 * v->x + m->s6 * v->y + m->sa * v->z + m->se * v->w;
	r.w = m->s3 * v->x + m->s7 * v->y + m->sb * v->z + m->sf * v->w;

	return r;
}

#endif
