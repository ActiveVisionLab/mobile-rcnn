// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

#pragma once

#include "../Core/Node.h"
#include "../Core/Connection.h"

#include <vector>

namespace LNTLib
{
	class Composite : public Node
	{
	public:
		Composite(std::string name) : Node(name) { }

		inline void Allocate(LNTLib::Device *device)
		{
			for (size_t nodeID = 0; nodeID < nodes.size(); nodeID++) {
				nodes[nodeID]->Allocate(device);
			}
		}

		inline void Deallocate()
		{
			for (size_t nodeID = 0; nodeID < nodes.size(); nodeID++) {
				nodes[nodeID]->Deallocate();
			}
		}

		inline void ReadWeights(FILE *f)
		{
			for (size_t nodeID = 0; nodeID < nodes.size(); nodeID++) {
				nodes[nodeID]->ReadWeights(f);
			}
		}

		inline ORUtils::MemoryBlock<float>* Output()
		{
			return nodes[nodes.size() - 1]->Output();
		}

		void Output(ORUtils::MemoryBlock<float>** res, Implementation::OutputOrder outputOrder, int offsetOut) {
			return nodes[nodes.size() - 1]->Output(res, outputOrder, offsetOut);
		}

		inline TensorInfo OutputGeometry()
		{
			return nodes[nodes.size() - 1]->OutputGeometry();
		}

		inline TensorInfo* InputGeometries()
		{
			return nodes[0]->InputGeometries();
		}

	protected:
		std::vector<Node *> nodes;
		std::vector<Connection *> connections;
	};
}
