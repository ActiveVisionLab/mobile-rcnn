// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "RoIAlign_Implementation_CUDNN.h"
#include "../../RoIAlign.h"

using namespace LNTLib;

__global__ void roialign_implementation_device(float** fpnFeatures, float *output, Vector4f *boxes, int *memberships, ushort C, int localBoxOffset,
	float samplingRatio, Vector2f *spatialScales, Vector2i *inSizes, int noFPNLevels, Vector2i outSize);

template <typename T>
_CPU_AND_GPU_CODE_ T roialign_interpolate(const T* data, const int width, const int height, T y, T x);

void RoIAlign_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	RoIAlign::NodeParams params = ((RoIAlign*)node)->Params();
	TensorInfo outputGeometry = node->OutputGeometry();

	// allocate boxes and membership arrays
	if (boxes == NULL || memberships == NULL)
	{
		boxes = new ORUtils::MemoryBlock<Vector4f>(params.boxes->DataSize(), MEMORYDEVICE_CUDA);
		memberships = new ORUtils::MemoryBlock<int>(params.memberships->DataSize(), MEMORYDEVICE_CUDA);
		spatialScales = new ORUtils::MemoryBlock<Vector2f>(params.noFPNLevels, MEMORYDEVICE_CUDA);

		spatialScales_host = new Vector2f[params.noFPNLevels];
		for (int scaleId = 0; scaleId < params.noFPNLevels; scaleId++)
			spatialScales_host[scaleId] = params.fpnLevelSizes[scaleId].toFloat() / params.inputImageSize.toFloat();

		ORcudaSafeCall(cudaMalloc((void**)&spatialScales_device, sizeof(Vector2f) * params.noFPNLevels));
		ORcudaSafeCall(cudaMemcpy(spatialScales_device, spatialScales_host, params.noFPNLevels * sizeof(Vector2f), cudaMemcpyHostToDevice));

		ORcudaSafeCall(cudaMalloc((void**)&fpnLevelSizes_device, sizeof(Vector2i) * params.noFPNLevels));
		ORcudaSafeCall(cudaMemcpy(fpnLevelSizes_device, params.fpnLevelSizes, params.noFPNLevels * sizeof(Vector2i), cudaMemcpyHostToDevice));

		fpnLevels_host = new float*[params.noFPNLevels];
		ORcudaSafeCall(cudaMalloc(&fpnLevels_device, sizeof(float*) * params.noFPNLevels));
	}

	// copy boxes and memberships to device memory
	boxes->SetFrom(params.boxes, MEMCPYDIR_CPU_TO_CUDA);
	memberships->SetFrom(params.memberships, MEMCPYDIR_CPU_TO_CUDA);

	// copy fpn level pointers to device memory (note -- they were already GPU points, just stored inside CPU vectors)
	auto baseFeatures = ((RoIAlign*)node)->GetBaseFeatures();
	for (int levelId = 0; levelId < params.noFPNLevels; levelId++)
		fpnLevels_host[levelId] = baseFeatures[levelId]->Data(MEMORYDEVICE_CUDA);
	ORcudaSafeCall(cudaMemcpy(fpnLevels_device, fpnLevels_host, params.noFPNLevels * sizeof(float*), cudaMemcpyHostToDevice));

	dim3 blockSize(4, 4, 4);
	dim3 gridSize((unsigned int)ceil((float)params.resolution.x / (float)blockSize.x), (unsigned int)ceil((float)params.resolution.y / (float)blockSize.y),
		outputGeometry.n * outputGeometry.c / blockSize.z);

	roialign_implementation_device << <gridSize, blockSize >> > (fpnLevels_device, output->Data(MEMORYDEVICE_CUDA), boxes->Data(MEMORYDEVICE_CUDA),
		memberships->Data(MEMORYDEVICE_CUDA), outputGeometry.c, params.currentBatchOffset,
		(float)params.samplingRatio, spatialScales_device, fpnLevelSizes_device, params.noFPNLevels,
		Vector2i(params.resolution.x, params.resolution.y));

	ORcudaKernelCheck;
}

void RoIAlign_Implementation_CUDNN::Allocate(LNTLib::Device *device)
{
	Implementation_CUDNN::Allocate(device);

	boxes = NULL;
	memberships = NULL;
	fpnLevels_host = NULL;
	spatialScales_host = NULL;
	fpnLevelSizes_device = NULL;
	spatialScales_device = NULL;
}

void RoIAlign_Implementation_CUDNN::Deallocate()
{
	RoIAlign::NodeParams params = ((RoIAlign*)node)->Params();

	delete boxes;
	delete memberships;

	delete fpnLevels_host;
	delete spatialScales_host;

	if (fpnLevelSizes_device) ORcudaSafeCall(cudaFree(fpnLevelSizes_device));
	if (spatialScales_device) ORcudaSafeCall(cudaFree(spatialScales_device));

	Implementation_CUDNN::Deallocate();
}

__global__ void roialign_implementation_device(float** fpnFeatures, float *output, Vector4f *boxes, int *memberships, ushort C, int localBoxOffset,
	float samplingRatio, Vector2f *spatialScales, Vector2i *inSizes, int noFPNLevels, Vector2i outSize)
{
	Vector3i gid(
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
		threadIdx.z + blockIdx.z * blockDim.z);

	if (gid.x >= outSize.x || gid.y >= outSize.y) return;

	const ushort n = gid.z / C;
	const ushort c = gid.z % C;

	const int fpnLevelId = memberships[n + localBoxOffset];

	const Vector2i inSize = inSizes[fpnLevelId];
	const float *input = fpnFeatures[fpnLevelId];

	const Vector4f roi_scaled = boxes[n + localBoxOffset] * spatialScales[fpnLevelId].x;

	// Force malformed ROIs to be 1x1
	const float roi_width = fmax(roi_scaled.z - roi_scaled.x, 1.0f);
	const float roi_height = fmax(roi_scaled.w - roi_scaled.y, 1.0f);

	const int roi_bin_grid_w = samplingRatio > 0 ? samplingRatio : ceil(roi_width / (float)outSize.x);
	const int roi_bin_grid_h = samplingRatio > 0 ? samplingRatio : ceil(roi_height / (float)outSize.y);

	const float bin_size_w = roi_width / (roi_bin_grid_w * (float)outSize.x);
	const float bin_size_h = roi_height / (roi_bin_grid_h * (float)outSize.y);

	const float count = roi_bin_grid_w * roi_bin_grid_h;

	float output_val = 0.0f;
	for (int iy = 0; iy < roi_bin_grid_h; iy++) for (int ix = 0; ix < roi_bin_grid_w; ix++)
	{
		const float x = roi_scaled.x + ((float)(gid.x * roi_bin_grid_w + ix) + 0.5f) * bin_size_w;
		const float y = roi_scaled.y + ((float)(gid.y * roi_bin_grid_h + iy) + 0.5f) * bin_size_h;

		if (x >= -1 && x <= inSize.x && y >= -1 && y <= inSize.y)
			output_val += roialign_interpolate(input + c * (inSize.x * inSize.y), inSize.x, inSize.y, x, y);
	}

	output[gid.x + gid.y * outSize.x + gid.z * outSize.x * outSize.y] = output_val / count;
}

template <typename T>
_CPU_AND_GPU_CODE_ T roialign_interpolate(const T* data, const int width, const int height, T x, T y)
{
	if (y <= 0) y = 0;
	if (x <= 0) x = 0;

	int x_low = (int)x;
	int y_low = (int)y;
	int y_high;
	int x_high;

	if (y_low >= height - 1) { y_high = y_low = height - 1; y = (T)y_low; }
	else { y_high = y_low + 1; }

	if (x_low >= width - 1) { x_high = x_low = width - 1; x = (T)x_low; }
	else { x_high = x_low + 1; }

	T ly = y - y_low;
	T lx = x - x_low;
	T hy = 1.0f - ly, hx = 1.0f - lx;

	// do bilinear interpolation
	T v1 = data[y_low * width + x_low];
	T v2 = data[y_low * width + x_high];
	T v3 = data[y_high * width + x_low];
	T v4 = data[y_high * width + x_high];
	T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

	T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

	return val;
}

#endif