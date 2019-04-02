// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

#include "Network.h"

#include <exception>
#include <iostream>
#include <string>

using namespace LNTLib;

void Network::Create(const char *fileName_prototype)
{
	try
	{
		char protoLine[1000];

		FILE *f_proto = fopen(fileName_prototype, "r");

		while (true)
		{
			if (fgets(protoLine, 1000, f_proto) == NULL) break;

			LNTLib::Node *node = this->CreateNode(protoLine);
			this->AddNode(node);

			fgets(protoLine, 1000, f_proto);

			LNTLib::Connection *connection = this->CreateConnection(protoLine, node);
			this->AddConnection(connection);
		}

		fclose(f_proto);
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
}

Node* Network::CreateNode(char *nodeStructure)
{
	char nodeType[20];
	LNTLib::Node *node = NULL;

	try
	{
		int numChar, numCharTotal = 0; char name[100];
		sscanf(nodeStructure, "%s %s %n", nodeType, name, &numChar);
		numCharTotal += numChar;

		if (strcmp(nodeType, "IMAGEINPUT") == 0) node = CreateNode_ImageInput(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "CONV") == 0) node = CreateNode_Conv(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "CONVDEPTHWISE") == 0) node = CreateNode_ConvDepthWise(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "CONVRELU") == 0) node = CreateNode_ConvRelu(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "CONVRELUDEPTHWISE") == 0) node = CreateNode_ConvReluDepthWise(name, nodeStructure, numChar, numCharTotal);
        else if (strcmp(nodeType, "CONVRELU6") == 0) node = CreateNode_ConvRelu6(name, nodeStructure, numChar, numCharTotal);
        else if (strcmp(nodeType, "CONVRELU6DEPTHWISE") == 0) node = CreateNode_ConvRelu6DepthWise(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "CONVTRANS") == 0) node = CreateNode_ConvTrans(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "CONVTRANSDEPTHWISE") == 0) node = CreateNode_ConvTransDepthWise(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "COPYOUTPUT") == 0) node = CreateNode_CopyOutput(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "RELU") == 0)  node = CreateNode_Relu(name, nodeStructure, numChar, numCharTotal);
        else if (strcmp(nodeType, "RELU6") == 0)  node = CreateNode_Relu6(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "RESHAPE") == 0)  node = CreateNode_Reshape(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "ROIALIGN") == 0)  node = CreateNode_RoIAlign(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "SIGMOID") == 0)  node = CreateNode_Sigmoid(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "POOLING") == 0) node = CreateNode_Pooling(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "CONCAT") == 0) node = CreateNode_Concat(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "SOFTMAX") == 0) node = CreateNode_Softmax(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "SUM") == 0) node = CreateNode_Sum(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "BATCHNORM") == 0) node = CreateNode_BatchNorm(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "CROP") == 0) node = CreateNode_Crop(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "BIAS") == 0) node = CreateNode_Bias(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "POWER") == 0) node = CreateNode_Power(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "INNERPRODUCT") == 0) node = CreateNode_InnerProduct(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "UPSAMPLE") == 0) node = CreateNode_Upsample(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "NORM") == 0) node = CreateNode_Norm(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "IMAGEOUTPUT") == 0) node = CreateNode_ImageOutput(name, nodeStructure, numChar, numCharTotal);
		else if (strcmp(nodeType, "IMAGERESIZE") == 0) node = CreateNode_ImageResize(name, nodeStructure, numChar, numCharTotal);

	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return NULL;
	}

	return node;
}

Node* Network::CreateNode_ImageInput(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	ImageInput::NodeParams params;

	sscanf(nodeStructure + numCharTotal, "size: %d %d %d %d %n", &params.size.x, &params.size.y, &params.size.z, &params.size.w, &numChar);
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "clipRect: %d %d %d %d %n", &params.clipRect.x, &params.clipRect.y, &params.clipRect.z, &params.clipRect.w, &numChar);
	numCharTotal += numChar;

	return new ImageInput(name, params, device);
}

Node* Network::CreateNode_ImageOutput(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	ImageOutput::NodeParams params;

	sscanf(nodeStructure + numCharTotal, "clipRect: %d %d %d %d %n", &params.clipRect.x, &params.clipRect.y, &params.clipRect.z, &params.clipRect.w, &numChar);
	numCharTotal += numChar;

	return new ImageOutput(name, params, device);
}

Node* Network::CreateNode_ImageResize(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	ImageResize::NodeParams params;

	sscanf(nodeStructure + numCharTotal, "size: %d %d %d %d %n", &params.size.x, &params.size.y, &params.size.z, &params.size.w, &numChar);
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "clipRect: %d %d %d %d %n", &params.clipRect.x, &params.clipRect.y, &params.clipRect.z, &params.clipRect.w, &numChar);
	numCharTotal += numChar;

	return new ImageResize(name, params, device);
}

Node* Network::CreateNode_Conv(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	Conv::NodeParams params; int buff;

	sscanf(nodeStructure + numCharTotal, "size: %d %d %d %d %n", &params.kernelSize.x, &params.kernelSize.y, &params.kernelSize.z, &params.noOutputs, &numChar);
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "hasBias: %d %n", &buff, &numChar); params.hasBiases = buff ? 1 : 0;
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "stride: %d %d %n", &params.stride.y, &params.stride.x, &numChar); // Y X
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "pad: %d %d %d %d %n", &params.padding.x, &params.padding.y, &params.padding.z, &params.padding.w, &numChar); // T B L R
	numCharTotal += numChar;

	params.hasRelu = false;
	params.isDepthWise = false;

	return new Conv(name, params, device);
}

Node* Network::CreateNode_ConvDepthWise(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	Conv *node = (Conv*)Network::CreateNode_Conv(name, nodeStructure, numChar, numCharTotal);
	node->SetDepthWise(true);
	
	return node;
}

Node* Network::CreateNode_ConvRelu(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	Conv *node = (Conv*)Network::CreateNode_Conv(name, nodeStructure, numChar, numCharTotal);
	node->SetHasRelu(true);
	
	return node;
}

Node* Network::CreateNode_Reshape(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	Reshape::NodeParams params;
	
	sscanf(nodeStructure + numCharTotal, "out: %d %d %d %n", &params.size_out.x, &params.size_out.y, &params.size_out.z, &numChar);
	numCharTotal += numChar;
	
	return new Reshape(name, params, device);
}

Node* Network::CreateNode_RoIAlign(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	RoIAlign::NodeParams params;
	
	sscanf(nodeStructure + numCharTotal, "resolution: %d %d %n", &params.resolution.x, &params.resolution.y, &numChar);
	numCharTotal += numChar;
	
	sscanf(nodeStructure + numCharTotal, "channels: %d %n", &params.noChannels, &numChar);
	numCharTotal += numChar;
		
	sscanf(nodeStructure + numCharTotal, "sampling_ratio: %d %n", &params.samplingRatio, &numChar);
	numCharTotal += numChar;
	
	params.boxes = NULL;
	params.noBoxes = 0;
	params.noFPNLevels = 0;
	
	return new RoIAlign(name, params, device);
}

Node* Network::CreateNode_ConvReluDepthWise(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	Conv *node = (Conv*)Network::CreateNode_Conv(name, nodeStructure, numChar, numCharTotal);
	node->SetDepthWise(true);
	node->SetHasRelu(true);
	
	return node;
}

Node* Network::CreateNode_ConvRelu6(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
    Conv *node = (Conv*)Network::CreateNode_Conv(name, nodeStructure, numChar, numCharTotal);
    node->SetHasRelu(true);
    node->SetReluMax(6.0f);
    
    return node;
}

Node* Network::CreateNode_ConvRelu6DepthWise(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
    Conv *node = (Conv*)Network::CreateNode_Conv(name, nodeStructure, numChar, numCharTotal);
    node->SetDepthWise(true);
    node->SetHasRelu(true);
    node->SetReluMax(6.0f);
    
    return node;
}

Node* Network::CreateNode_InnerProduct(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	InnerProduct::NodeParams params; int buff;

	sscanf(nodeStructure + numCharTotal, "size: %d %d %d %d %n", &params.kernelSize.x, &params.kernelSize.y, &params.kernelSize.z, &params.noOutputs, &numChar);
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "hasBias: %d %n", &buff, &numChar); params.hasBiases = buff ? 1 : 0;
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "stride: %d %d %n", &params.stride.y, &params.stride.x, &numChar); // Y X
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "pad: %d %d %d %d %n", &params.padding.x, &params.padding.y, &params.padding.z, &params.padding.w, &numChar); // T B L R
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "newShape: %d %d %d %d %n", &params.newShape.n, &params.newShape.c, &params.newShape.h, &params.newShape.w, &numChar);
	numCharTotal += numChar;

	return new InnerProduct(name, params, device);
}

Node* Network::CreateNode_ConvTrans(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	ConvTransposed::NodeParams params; int buff;

	sscanf(nodeStructure + numCharTotal, "size: %d %d %d %d %n", &params.kernelSize.x, &params.kernelSize.y, &params.kernelSize.z, &params.noOutputs, &numChar);
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "hasBias: %d %n", &buff, &numChar); params.hasBiases = buff ? 1 : 0;
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "upsample: %d %d %n", &params.upsample.y, &params.upsample.x, &numChar); // Y X
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "crop: %d %d %d %d %n", &params.crop.x, &params.crop.y, &params.crop.z, &params.crop.w, &numChar); // T B L R
	numCharTotal += numChar;
	
	params.isDepthWise = false;

	return new ConvTransposed(name, params, device);
}

Node* Network::CreateNode_ConvTransDepthWise(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	ConvTransposed *node = (ConvTransposed*)Network::CreateNode_ConvTrans(name, nodeStructure, numChar, numCharTotal);
	node->SetDepthWise(true);
	
	return node;
}

Node* Network::CreateNode_CopyOutput(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	return new CopyOutput(name, device);
}

Node* Network::CreateNode_Bias(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	Bias::NodeParams params;

	char method[10];
	sscanf(nodeStructure + numCharTotal, "method: %s %n", method, &numChar);
	numCharTotal += numChar;

	params.method = Bias::Bias_Weights;

	if (strcmp(method, "weights") == 0) params.method = Bias::Bias_Weights;
	else if (strcmp(method, "features") == 0) params.method = Bias::Bias_Features;

	return new Bias(name, params, device);
}

Node* Network::CreateNode_Power(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	Power::NodeParams params;

	sscanf(nodeStructure + numCharTotal, "shift: %f %n", &params.shift, &numChar);
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "scale: %f %n", &params.scale, &numChar);
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "power: %f %n", &params.power, &numChar);
	numCharTotal += numChar;

	return new Power(name, params, device);
}

Node* Network::CreateNode_Relu6(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	Relu *node = new Relu(name, device);
    node->SetReluMax(6.0f);
    
    return node;
}

Node* Network::CreateNode_Relu(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
    return new Relu(name, device);
}

Node* Network::CreateNode_Sigmoid(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	return new Sigmoid(name, device);
}

Node* Network::CreateNode_Norm(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	return new Norm(name, device);
}

Node* Network::CreateNode_Pooling(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	Pooling::NodeParams params; int buff;

	sscanf(nodeStructure + numCharTotal, "size: %d %d %n", &params.poolSize.y, &params.poolSize.x, &numChar);
	numCharTotal += numChar;

	char method[10];
	sscanf(nodeStructure + numCharTotal, "method: %s %n", method, &numChar);
	numCharTotal += numChar;

	if (strcmp(method, "max") == 0) params.method = Pooling::Pooling_Max;
	else if (strcmp(method, "avg") == 0) params.method = Pooling::Pooling_Avg;

	sscanf(nodeStructure + numCharTotal, "stride: %d %d %n", &params.stride.y, &params.stride.x, &numChar); // Y X
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "pad: %d %d %d %d %n", &params.padding.x, &params.padding.y, &params.padding.z, &params.padding.w, &numChar); // T B L R
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "remove_extra: %d %n", &buff, &numChar); params.remove_extra = buff > 0;
	numCharTotal += numChar;

	return new Pooling(name, params, device);
}

Node* Network::CreateNode_Upsample(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	Upsample::NodeParams params;

	sscanf(nodeStructure + numCharTotal, "upsample: %d %d %n", &params.upsample.x, &params.upsample.y, &numChar);
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "kernel: %d %d %n", &params.kernel.x, &params.kernel.y, &numChar);
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "crop: %d %n", &params.crop, &numChar);
	numCharTotal += numChar;

	sscanf(nodeStructure + numCharTotal, "rect: %d %d %n", &params.rect.x, &params.rect.y, &numChar);
	numCharTotal += numChar;

	char method[50];
	sscanf(nodeStructure + numCharTotal, "method: %s %n", method, &numChar);
	numCharTotal += numChar;

	if (strcmp(method, "bilinear") == 0) params.method = Upsample::Upsample_Bilinear;
	else if (strcmp(method, "sarojbilinear") == 0) params.method = Upsample::Upsample_SarojBilinear;
	else if (strcmp(method, "nearest") == 0) params.method = Upsample::Upsample_Nearest;

	return new Upsample(name, params, device);
}

Node* Network::CreateNode_Concat(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	return new Concat(name, device);
}

Node* Network::CreateNode_Softmax(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	return new Softmax(name, device);
}

Node* Network::CreateNode_Sum(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	return new Sum(name, device);
}

Node* Network::CreateNode_BatchNorm(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	return new BatchNorm(name, device);
}

Node* Network::CreateNode_Crop(char *name, char *nodeStructure, int numChar, int numCharTotal)
{
	Crop::NodeParams params;

	sscanf(nodeStructure + numCharTotal, "topLeft: %d %d %n", &params.topLeft.x, &params.topLeft.y, &numChar);
	numCharTotal += numChar;

	return new Crop(name, params, device);
}

Connection* Network::CreateConnection(char* connectionStructure, LNTLib::Node *node)
{
	int numChar, numCharTotal = 0; int noInputs; char targetName[100];

	sscanf(connectionStructure, "\tinput from %d%n", &noInputs, &numChar);
	numCharTotal += numChar + 2; // to skip :

	LNTLib::Connection *connection = new LNTLib::Connection(noInputs, node);

	for (int inputId = 0; inputId < noInputs; inputId++)
	{
		sscanf(connectionStructure + numCharTotal, "%s %n", targetName, &numChar);
		numCharTotal += numChar;

		if (strcmp(targetName, "previous") == 0)
		{
			connection->AddNode(this->Node((int)this->NoNodes() - 2), inputId);
		}
		else
		{
			// search for node
			for (size_t i = 0; i < this->NoNodes(); i++)
			{
				if (strcmp(this->Node(i)->Name().c_str(), targetName) == 0)
				{
					connection->AddNode(this->Node(i), inputId);
					break;
				}
			}
		}
	}

	TensorInfo *inputGeometries = NULL;
	if (noInputs > 0)
	{
		inputGeometries = new TensorInfo[noInputs];

		for (int inputId = 0; inputId < noInputs; inputId++)
			inputGeometries[inputId] = connection->Input(inputId)->OutputGeometry();
	}

	node->SetGeometry(inputGeometries, noInputs);

	return connection;
}

void Network::Allocate()
{
	// allocate memory for each node using the designated implementation
	for (size_t nodeId = 0; nodeId < this->NoNodes(); nodeId++)
	{
		LNTLib::Node *node = this->Node(nodeId);
		node->Allocate(device);
	}
}

void Network::Deallocate()
{
	for (size_t nodeId = 0; nodeId < this->NoNodes(); nodeId++)
		this->Node(nodeId)->Deallocate();
}

void Network::ReadWeights(const char *fileName_weights)
{
	FILE *f = fopen(fileName_weights, "rb");

	for (size_t nodeId = 0; nodeId < this->NoNodes(); nodeId++)
		this->Node(nodeId)->ReadWeights(f);

	fclose(f);
}
