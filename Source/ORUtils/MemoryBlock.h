// Copyright 2014-2018 Oxford University Innovation Limited and the authors of ORUtils

#pragma once

#include "MemoryDeviceDefs.h"
#include "PlatformIndependence.h"

#ifndef __METALC__

#ifdef COMPILE_WITH_CUDA
#include "CUDADefines.h"
#endif

#ifdef COMPILE_WITH_OPENCL
#include "OpenCLContext.h"
#endif

#ifdef COMPILE_WITH_METAL
#include "MetalContext.h"
#endif

#include <stdlib.h>
#include <string.h>

namespace ORUtils
{
	// ugly hack so that we can create completely generic memory blocks without needing any templates
	class GenericMemoryBlock
	{
		
	};
	
	/** \brief
	Represents memory blocks, templated on the data type
	*/
	template <typename T>
	class MemoryBlock : public GenericMemoryBlock
	{
	protected:
		/** Total number of allocated entries in the data array. */
		int dataSize;

		bool isAllocated_CPU, isAllocated_CUDA, isMetalCompatible;

		/** Pointer to memory on CPU host. */
		DEVICEPTR(T)* data_cpu;

		/** Pointer to memory on GPU, if available. */
		DEVICEPTR(T)* data_cuda;

#ifdef COMPILE_WITH_OPENCL
		void *data_openclBuffer;
		void *data_openclImage;
#endif

#ifdef COMPILE_WITH_METAL
		void *data_metalBuffer;
		void *data_metalImage;
#endif
	public:
		/** Get the data pointer on CPU or GPU. */
		inline DEVICEPTR(T)* Data(MemoryDeviceType memoryType)
		{
			switch (memoryType)
			{
			case MEMORYDEVICE_CPU: return data_cpu;
			case MEMORYDEVICE_CUDA: return data_cuda;
			}

			return 0;
		}

		/** Get the data pointer on CPU or GPU. */
		inline const DEVICEPTR(T)* Data(MemoryDeviceType memoryType) const
		{
			switch (memoryType)
			{
			case MEMORYDEVICE_CPU: return data_cpu;
			case MEMORYDEVICE_CUDA: return data_cuda;
			}

			return 0;
		}

#ifdef COMPILE_WITH_OPENCL
		inline cl_mem Data_OpenCL() const { return (cl_mem)data_openclBuffer; }
		inline cl_mem Data_OpenCLImage() const { return (cl_mem)data_openclImage; }

		inline void InitOpenCLImage(void *oclImageDesc, void* oclImageFormat) {
			OpenCLContext::Instance()->InitOpenCLImage(&data_openclImage, oclImageDesc, oclImageFormat);
		}
#endif
#ifdef COMPILE_WITH_METAL
		inline const void *Data_Metal() const { return data_metalBuffer; }
		inline void *Data_MetalImage() { return data_metalImage; }

		/* TEMP FIX */
		void UpdateHostFromMetalImage(int width, int height, int channels) {}

		inline void InitMetalImage(void *imageDesc)
		{
			initMetalImage((void**)&data_metalImage, imageDesc);
		}
		
		inline void ReleaseMetalImage()
		{
			if (data_metalImage != NULL)
				freeMetalImage((void**)&data_metalImage);
		}
#endif
		inline int DataSize() const { return dataSize; }

		inline bool HasData(MemoryDeviceType memoryType) const
		{
			if (memoryType == MEMORYDEVICE_CPU && isAllocated_CPU) return true;
			if (memoryType == MEMORYDEVICE_CUDA && isAllocated_CUDA) return true;
			return false;
		}

#ifdef COMPILE_WITH_OPENCL
		inline void Map() { OpenCLContext::Instance()->Map((void**)&data_openclBuffer, (void**)&data_cpu, (int)(dataSize * sizeof(T))); }
		inline void Map(int newDataSize) { OpenCLContext::Instance()->Map((void**)&data_openclBuffer, (void**)&data_cpu, (int)(newDataSize * sizeof(T))); }
		inline void UnMap() { OpenCLContext::Instance()->UnMap((void**)&data_openclBuffer, (void**)&data_cpu); }
#endif

		/** Initialize an empty memory block of the given size,
		on CPU only or GPU only or on both. CPU might also use the
		Metal compatible allocator (i.e. with 16384 alignment).
		*/
		MemoryBlock(int dataSize, bool allocate_CPU, bool allocate_CUDA, bool metalCompatible = true)
		{
			this->isAllocated_CPU = false;
			this->isAllocated_CUDA = false;
			this->isMetalCompatible = false;

#ifndef NDEBUG // When building in debug mode always allocate both on the CPU and the GPU
			if (allocate_CUDA) allocate_CPU = true;
#endif

			Allocate(dataSize, allocate_CPU, allocate_CUDA, metalCompatible);
			Clear();
		}

		/** Initialize an empty memory block of the given size, either
		on CPU only or on GPU only. CPU will be Metal compatible if Metal
		is enabled.
		*/
		MemoryBlock(int dataSize, MemoryDeviceType memoryType)
		{
			this->isAllocated_CPU = false;
			this->isAllocated_CUDA = false;
			this->isMetalCompatible = false;

			switch (memoryType)
			{
			case MEMORYDEVICE_CPU: Allocate(dataSize, true, false, true); break;
			case MEMORYDEVICE_CUDA:
			{
#ifndef NDEBUG // When building in debug mode always allocate both on the CPU and the GPU
				Allocate(dataSize, true, true, true);
#else
				Allocate(dataSize, false, true, true);
#endif
				break;
			}
			}

			Clear();
		}

		/** Resize a memory block, loosing all old data.
		Essentially any previously allocated data is
		released, new memory is allocated.
		*/
		void ChangeDataSize(int newDataSize, bool noResize = false, bool ignoreOldSize = false)
		{
			if (noResize && (dataSize > newDataSize || ignoreOldSize))
				this->dataSize = newDataSize;
			else
				if (dataSize != newDataSize)
				{
					this->dataSize = newDataSize;

					bool allocate_CPU = this->isAllocated_CPU;
					bool allocate_CUDA = this->isAllocated_CUDA;
					bool metalCompatible = this->isMetalCompatible;

					this->Free();
					this->Allocate(newDataSize, allocate_CPU, allocate_CUDA, metalCompatible);
				}
		}

		/** Set all image data to the given @p defaultValue. */
		virtual void Clear(unsigned char defaultValue = 0)
		{
			if (isAllocated_CPU) memset(data_cpu, defaultValue, dataSize * sizeof(T));
#ifdef COMPILE_WITH_CUDA
			if (isAllocated_CUDA) ORcudaSafeCall(cudaMemset(data_cuda, defaultValue, dataSize * sizeof(T)));
#endif
		}

		/** Set all image data to the given @p defaultValue. */
		virtual void Clear(int clearSize, unsigned char defaultValue)
		{
			if (isAllocated_CPU) memset(data_cpu, defaultValue, clearSize * sizeof(T));
#ifdef COMPILE_WITH_CUDA
			if (isAllocated_CUDA) ORcudaSafeCall(cudaMemset(data_cuda, defaultValue, clearSize * sizeof(T)));
#endif
		}

		/** Transfer data from CPU to GPU, if possible. */
		void UpdateDeviceFromHost() const {
#ifdef COMPILE_WITH_CUDA
			if (isAllocated_CUDA && isAllocated_CPU)
				ORcudaSafeCall(cudaMemcpy(data_cuda, data_cpu, dataSize * sizeof(T), cudaMemcpyHostToDevice));
#endif
#ifdef COMPILE_WITH_OPENCL
			if ((data_openclBuffer != NULL) && (data_cpu != NULL)) {
				OpenCLContext::Instance()->UpdateDeviceFromHost((cl_mem)data_openclBuffer, data_cpu, dataSize * sizeof(T));
			}
#endif
		}
		/** Transfer data from GPU to CPU, if possible. */
		void UpdateHostFromDevice() const {
#ifdef COMPILE_WITH_CUDA
			if (isAllocated_CUDA && isAllocated_CPU)
				ORcudaSafeCall(cudaMemcpy(data_cpu, data_cuda, dataSize * sizeof(T), cudaMemcpyDeviceToHost));
#endif
#ifdef COMPILE_WITH_OPENCL
			if ((data_openclBuffer != NULL) && (data_cpu != NULL)) {
				OpenCLContext::Instance()->UpdateHostFromDevice((cl_mem)data_openclBuffer, data_cpu, dataSize * sizeof(T));
			}
#endif
		}

		/** Copy data */
		void SetFrom(const MemoryBlock<T> *source, MemoryCopyDirection memoryCopyDirection)
		{
			switch (memoryCopyDirection)
			{
			case MEMCPYDIR_CPU_TO_CPU:
				memcpy(this->data_cpu, source->data_cpu, source->dataSize * sizeof(T));
				break;
#ifdef COMPILE_WITH_CUDA
			case MEMCPYDIR_CPU_TO_CUDA:
				ORcudaSafeCall(cudaMemcpyAsync(this->data_cuda, source->data_cpu, source->dataSize * sizeof(T), cudaMemcpyHostToDevice));
				break;
			case MEMCPYDIR_CUDA_TO_CPU:
				ORcudaSafeCall(cudaMemcpy(this->data_cpu, source->data_cuda, source->dataSize * sizeof(T), cudaMemcpyDeviceToHost));
				break;
			case MEMCPYDIR_CUDA_TO_CUDA:
				ORcudaSafeCall(cudaMemcpyAsync(this->data_cuda, source->data_cuda, source->dataSize * sizeof(T), cudaMemcpyDeviceToDevice));
				break;
#endif
			default: break;
			}
		}


		/** Copy data */
		void SetFrom(const T *source, int size, MemoryCopyDirection memoryCopyDirection)
		{
			switch (memoryCopyDirection)
			{
			case MEMCPYDIR_CPU_TO_CPU:
				memcpy(this->data_cpu, source, size * sizeof(T));
				break;
#ifdef COMPILE_WITH_CUDA
			case MEMCPYDIR_CPU_TO_CUDA:
				ORcudaSafeCall(cudaMemcpyAsync(this->data_cuda, source, size * sizeof(T), cudaMemcpyHostToDevice));
				break;
			case MEMCPYDIR_CUDA_TO_CPU:
				ORcudaSafeCall(cudaMemcpy(this->data_cpu, source, size * sizeof(T), cudaMemcpyDeviceToHost));
				break;
			case MEMCPYDIR_CUDA_TO_CUDA:
				ORcudaSafeCall(cudaMemcpyAsync(this->data_cuda, source, size * sizeof(T), cudaMemcpyDeviceToDevice));
				break;
#endif
			default: break;
			}
		}

		/** Get an individual element of the memory block from either the CPU or GPU. */
		T GetElement(int n, MemoryDeviceType memoryType) const
		{
			switch (memoryType)
			{
			case MEMORYDEVICE_CPU:
			{
				return this->data_cpu[n];
			}
#ifdef COMPILE_WITH_CUDA
			case MEMORYDEVICE_CUDA:
			{
				T result;
				ORcudaSafeCall(cudaMemcpy(&result, this->data_cuda + n, sizeof(T), cudaMemcpyDeviceToHost));
				return result;
			}
#endif
			default: throw std::runtime_error("Invalid memory type");
			}
		}

		virtual ~MemoryBlock() { this->Free(); }

		/** Allocate image data of the specified size. If the
		data has been allocated before, the data is freed.
		*/
		void Allocate(int dataSize, bool allocate_CPU, bool allocate_CUDA, bool metalCompatible)
		{
			Free();

			this->dataSize = dataSize;

			if (allocate_CPU)
			{
				int allocType = 0;

#ifdef COMPILE_WITH_CUDA
				if (allocate_CUDA) allocType = 1;
#endif
#ifdef COMPILE_WITH_METAL
				if (metalCompatible) allocType = 2;
#endif
#ifdef COMPILE_WITH_OPENCL
				allocType = 3;
#endif
				switch (allocType)
				{
				case 0:
					if (dataSize == 0) data_cpu = NULL;
					else data_cpu = new T[dataSize];
					break;
				case 1:
#ifdef COMPILE_WITH_CUDA
					if (dataSize == 0) data_cpu = NULL;
					else ORcudaSafeCall(cudaMallocHost((void**)&data_cpu, dataSize * sizeof(T)));
#endif
					break;
				case 2:
#ifdef COMPILE_WITH_METAL
					if (dataSize == 0) data_cpu = NULL;
					else allocateMetalData((void**)&data_cpu, (void**)&data_metalBuffer, (int)(dataSize * sizeof(T)), true);
#endif
					break;
				case 3:
#ifdef COMPILE_WITH_OPENCL
					if (dataSize == 0) data_cpu = NULL;
					else OpenCLContext::Instance()->AllocateOpenCLData((void**)&data_openclBuffer, (void**)&data_cpu, (int)(dataSize * sizeof(T)));
#endif
					break;
				}

				this->isAllocated_CPU = allocate_CPU;
				this->isMetalCompatible = metalCompatible;
			}

			if (allocate_CUDA)
			{
#ifdef COMPILE_WITH_CUDA
				if (dataSize == 0) data_cuda = NULL;
				else ORcudaSafeCall(cudaMalloc((void**)&data_cuda, dataSize * sizeof(T)));
				this->isAllocated_CUDA = allocate_CUDA;
#endif
			}


#ifdef COMPILE_WITH_OPENCL
            data_openclImage = NULL;
#endif

#ifdef COMPILE_WITH_METAL
			data_metalImage = NULL;
#endif
		}

		void Free()
		{
			if (isAllocated_CPU)
			{
				int allocType = 0;

#ifdef COMPILE_WITH_CUDA
				if (isAllocated_CUDA) allocType = 1;
#endif
#ifdef COMPILE_WITH_METAL
				if (isMetalCompatible) allocType = 2;
#endif
#ifdef COMPILE_WITH_OPENCL
				allocType = 3;
#endif
				switch (allocType)
				{
				case 0:
					if (data_cpu != NULL) delete[] data_cpu;
					break;
				case 1:
#ifdef COMPILE_WITH_CUDA
					if (data_cpu != NULL) ORcudaSafeCall(cudaFreeHost(data_cpu));
#endif
					break;
				case 2:
#ifdef COMPILE_WITH_METAL
					if (data_cpu != NULL) freeMetalData((void**)&data_cpu, (void**)&data_metalBuffer, (int)(dataSize * sizeof(T)), true);
					if (data_metalImage != NULL) freeMetalImage((void**)&data_metalImage);
#endif
					break;
				case 3:
#ifdef COMPILE_WITH_OPENCL
					if (data_cpu != NULL) OpenCLContext::Instance()->FreeOpenCLData((void**)&data_openclBuffer, (void**)&data_cpu);
					if (data_openclImage != NULL) OpenCLContext::Instance()->FreeOpenCLData((void**)&data_openclImage, (void**)&data_openclImage);
#endif
					break;
				}

				isMetalCompatible = false;
				isAllocated_CPU = false;
			}

			if (isAllocated_CUDA)
			{
#ifdef COMPILE_WITH_CUDA
				if (data_cuda != NULL) ORcudaSafeCall(cudaFree(data_cuda));
#endif
				isAllocated_CUDA = false;
			}
		}

		void Swap(MemoryBlock<T>& rhs)
		{
			std::swap(this->dataSize, rhs.dataSize);
			std::swap(this->data_cpu, rhs.data_cpu);
			std::swap(this->data_cuda, rhs.data_cuda);
#ifdef COMPILE_WITH_METAL
			std::swap(this->data_metalBuffer, rhs.data_metalBuffer);
			std::swap(this->data_metalImage, rhs.data_metalImage);
#endif
#ifdef COMPILE_WITH_OPENCL
			std::swap(this->data_openclBuffer, rhs.data_openclBuffer);
			std::swap(this->data_openclImage, rhs.data_openclImage);
#endif
			std::swap(this->isAllocated_CPU, rhs.isAllocated_CPU);
			std::swap(this->isAllocated_CUDA, rhs.isAllocated_CUDA);
			std::swap(this->isMetalCompatible, rhs.isMetalCompatible);
		}

		// Suppress the default copy constructor and assignment operator
		MemoryBlock(const MemoryBlock&);
		MemoryBlock& operator=(const MemoryBlock&);
	};
}

#endif
