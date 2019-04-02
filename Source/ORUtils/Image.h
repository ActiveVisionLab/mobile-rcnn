// Copyright 2014-2018 Oxford University Innovation Limited and the authors of ORUtils

#pragma once

#include "MemoryBlock.h"
#include "Vector.h"

#ifndef __METALC__

namespace ORUtils
{
	/** \brief
	Represents images, templated on the pixel type
	*/
	template <typename T>
	class Image : public MemoryBlock < T >
	{
	private:
		/** Size of the image in pixels. */
		Vector2<int> noDims;

	public:

		/** Initialize an empty image of the given size, either
		on CPU only or on both CPU and GPU.
		*/
		Image(Vector2<int> noDims, bool allocate_CPU, bool allocate_CUDA, bool metalCompatible = true)
			: MemoryBlock<T>(noDims.x * noDims.y, allocate_CPU, allocate_CUDA, metalCompatible)
		{
			this->noDims = noDims;
		}

		Image(bool allocate_CPU, bool allocate_CUDA, bool metalCompatible = true)
			: MemoryBlock<T>(0, allocate_CPU, allocate_CUDA, metalCompatible)
		{
			this->noDims = Vector2<int>(0, 0);
		}

		Image(Vector2<int> noDims, MemoryDeviceType memoryType)
			: MemoryBlock<T>(noDims.x * noDims.y, memoryType)
		{
			this->noDims = noDims;
		}

		/** Resize an image, loosing all old image data.
		Essentially any previously allocated data is
		released, new memory is allocated.
		*/
		void ChangeDims(Vector2<int> newDims, bool noResize = false, bool ignoreOldDims = false)
		{
			if (noResize && ((noDims.x > newDims.x && noDims.y > newDims.y) || ignoreOldDims))
				this->noDims = newDims;
			else
				if (newDims != noDims)
				{
					this->noDims = newDims;

					bool allocate_CPU = this->isAllocated_CPU;
					bool allocate_CUDA = this->isAllocated_CUDA;
					bool metalCompatible = this->isMetalCompatible;

					this->Free();
					this->Allocate(newDims.x * newDims.y, allocate_CPU, allocate_CUDA, metalCompatible);
				}
		}

		void Clear2D(Vector2<int> clearSize)
		{
			int totalClearSize = clearSize.x * clearSize.y;
			MemoryBlock<T>::Clear(totalClearSize, 0);
		}

		Vector2<int> NoDims() const { return noDims; }

		void SetFrom(const Image<T> *source, MemoryCopyDirection memoryCopyDirection)
		{
			switch (memoryCopyDirection)
			{
			case MEMCPYDIR_CPU_TO_CPU:
				MemoryBlock<T>::SetFrom(source->Data(MEMORYDEVICE_CPU), source->DataSize(), memoryCopyDirection);
				break;
#ifdef COMPILE_WITH_CUDA
			case MEMCPYDIR_CPU_TO_CUDA:
				MemoryBlock<T>::SetFrom(source->Data(MEMORYDEVICE_CPU), source->DataSize(), memoryCopyDirection);
				break;
			case MEMCPYDIR_CUDA_TO_CPU:
				MemoryBlock<T>::SetFrom(source->Data(MEMORYDEVICE_CUDA), source->DataSize(), memoryCopyDirection);
				break;
			case MEMCPYDIR_CUDA_TO_CUDA:
				MemoryBlock<T>::SetFrom(source->Data(MEMORYDEVICE_CUDA), source->DataSize(), memoryCopyDirection);
				break;
#endif
			default: break;
			}
		}

		void SetFrom(const T *source, int size, MemoryCopyDirection memoryCopyDirection)
		{
			MemoryBlock<T>::SetFrom(source, size, memoryCopyDirection);
		}

		void Swap(Image<T>& rhs)
		{
			MemoryBlock<T>::Swap(rhs);
			std::swap(this->noDims, rhs.noDims);
		}

		// Suppress the default copy constructor and assignment operator
		Image(const Image&);
		Image& operator=(const Image&);
	};
}

#endif
