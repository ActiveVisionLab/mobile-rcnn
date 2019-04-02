// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

// Brief: 
//   Base class for all LNTNetworks. LNTNetworks own a list of LNTNodes
//   and LNTConnections. Network implements the method to pass data 
//   through the LNTNodes using the LNTConnections. Subclasses of 
//   LNTNetworks are repsonsible for getting data to and from the network.

#pragma once

#include <vector>

#include "Node.h"
#include "Connection.h"

#include "../Nodes/ImageInput.h"
#include "../Nodes/ImageResize.h"
#include "../Nodes/ImageOutput.h"
#include "../Nodes/Conv.h"
#include "../Nodes/ConvTransposed.h"
#include "../Nodes/CopyOutput.h"
#include "../Nodes/Relu.h"
#include "../Nodes/Reshape.h"
#include "../Nodes/RoIAlign.h"
#include "../Nodes/Sigmoid.h"
#include "../Nodes/Pooling.h"
#include "../Nodes/Power.h"
#include "../Nodes/Concat.h"
#include "../Nodes/Softmax.h"
#include "../Nodes/Sum.h"
#include "../Nodes/BatchNorm.h"
#include "../Nodes/Bias.h"
#include "../Nodes/Crop.h"
#include "../Nodes/InnerProduct.h"
#include "../Nodes/Upsample.h"
#include "../Nodes/Norm.h"

#include "../../ORUtils/NVTimer.h"
#include "../../ORUtils/ImageTypes.h"

namespace LNTLib
{
	class Network
	{
	public:
		virtual void Process(ORUtils::GenericMemoryBlock *input, bool waitForComputeToFinish = true)
		{
			std::vector<ORUtils::GenericMemoryBlock *> inputs;
			inputs.push_back(input);

			this->Process(inputs, waitForComputeToFinish);
		}

		virtual void Process(std::vector< ORUtils::GenericMemoryBlock* > inputs, bool waitForComputeToFinish)
		{
			InputToNetwork(inputs);
			Forward(waitForComputeToFinish);
			NetworkToResult();
		}

		// Brief:
		//   Adds node to the end of the network.
		inline void AddNode(LNTLib::Node *node) { nodes.push_back(node); }

		// Brief:
		//  Adds connection to the end of the network.
		inline void AddConnection(LNTLib::Connection *connection) { connections.push_back(connection); }

		// Brief:
		//   Getter a node.
		inline LNTLib::Node* Node(size_t nodeID) { return nodes[nodeID]; }

		// Brief:
		//	 Get a connection
		inline LNTLib::Connection* Connection(size_t nodeId) { return connections[nodeId]; }

		// Brief:
		//   Getter for number of nodes.
		inline size_t NoNodes() { return nodes.size(); }

		// Brief:
		//   Implemented for overriding in derived classes.
		virtual ~Network() { 
			if (hasInternalNodes) 
				this->Deallocate();
		}

		inline LNTLib::Device* Device() { return device; }

	protected:
		// Brief:
		//   Iniits current inputs to null.
		Network(const char *f_proto, const char *f_weights, LNTLib::Device *device, int verboseLevel = 0) {
			this->verboseLevel = verboseLevel;
			this->device = device;

			hasInternalNodes = false;
			if (f_proto != NULL && f_weights != NULL)
			{
				this->Create(f_proto);
				this->Allocate();
				this->ReadWeights(f_weights);
				hasInternalNodes = true;
			}
		}

		virtual void InputToNetwork(std::vector< ORUtils::GenericMemoryBlock* > inputs) = 0;
		virtual void NetworkToResult() = 0;

		// Brief:
		//   Forwards data through each node in the network.
		virtual void Forward(bool waitForComputeToFinish)
		{
			StopWatchInterface *timerNetwork = NULL;
			StopWatchInterface *timerNode = NULL;

			if (verboseLevel > 0)
			{
				sdkCreateTimer(&timerNetwork); sdkResetTimer(&timerNetwork);
				if (verboseLevel > 1) { sdkCreateTimer(&timerNode); sdkResetTimer(&timerNode); }
				printf("processing network...\n"); sdkStartTimer(&timerNetwork);
			}

			device->StartCompute();

			size_t noNodes = nodes.size();
			std::vector < ORUtils::MemoryBlock<float>* > currentInputs;

			for (size_t nodeId = 0; nodeId < noNodes; nodeId++)
			{
				LNTLib::Node *currentNode = nodes[nodeId];
				LNTLib::Connection *currentConnection = connections[nodeId];

				size_t noInputs = currentConnection->NoInputs();

				currentInputs.resize(noInputs);

				for (size_t inputId = 0; inputId < noInputs; inputId++)
					currentInputs[inputId] = currentConnection->Input(inputId)->Output();

				if (verboseLevel > 1)
				{
					printf("\tprocessing node: %s...", currentNode->Name().c_str());
					sdkStartTimer(&timerNode);
				}

				currentNode->Forward(currentInputs);

				if (verboseLevel > 1)
				{
					sdkStopTimer(&timerNode);
					printf("done, time elapsed: %f\n", sdkGetTimerValue(&timerNode));

					sdkResetTimer(&timerNode);
				}
			}

			if (waitForComputeToFinish) device->FinishCompute();

			if (verboseLevel > 0)
			{
				sdkStopTimer(&timerNetwork); printf("done, time elapsed: %f\n", sdkGetTimerValue(&timerNetwork));
				sdkResetTimer(&timerNetwork); sdkDeleteTimer(&timerNetwork);
				if (verboseLevel > 1) sdkDeleteTimer(&timerNode);
			}
		}

	protected:
		int verboseLevel;
		
		LNTLib::Device* device;

		std::vector<LNTLib::Node *>nodes;
		std::vector<LNTLib::Connection *>connections;

	private:
		bool hasInternalNodes;

		LNTLib::Node* CreateNode(char *nodeStructure);
		LNTLib::Connection* CreateConnection(char* connectionStructure, LNTLib::Node *node);

		LNTLib::Node* CreateNode_ImageInput(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Bias(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Power(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Conv(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_ConvDepthWise(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_ConvRelu(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_ConvRelu6(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_ConvReluDepthWise(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_ConvRelu6DepthWise(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_RoIAlign(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Sigmoid(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_ConvTrans(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_ConvTransDepthWise(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_CopyOutput(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Relu(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Relu6(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Reshape(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Pooling(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Concat(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Softmax(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Sum(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_BatchNorm(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Crop(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_InnerProduct(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Upsample(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_Norm(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_ImageOutput(char *name, char *nodeStructure, int numChar, int numCharTotal);
		LNTLib::Node* CreateNode_ImageResize(char *name, char *nodeStructure, int numChar, int numCharTotal);

		void Create(const char *fileName_prototype);

		void Allocate();
		void Deallocate();

		void ReadWeights(const char *fileName_weights);
	}; // class Network
}
