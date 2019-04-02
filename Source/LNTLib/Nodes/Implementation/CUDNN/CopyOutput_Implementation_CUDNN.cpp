// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet
#ifdef COMPILE_WITH_CUDNN

#include "CopyOutput_Implementation_CUDNN.h"
#include "../../CopyOutput.h"

using namespace LNTLib;

void CopyOutput_Implementation_CUDNN::Forward(const std::vector< ORUtils::MemoryBlock<float>* > &inputs)
{
	this->output->SetFrom(inputs[0], MEMCPYDIR_CUDA_TO_CUDA);
}

#endif