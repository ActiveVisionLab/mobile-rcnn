// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

#pragma once

#include "Node.h"

namespace LNTLib
{
	class Connection
	{
	public:
		// Brief:
		//   Sets output node and allocates storage for input nodes.
		//   Input nodes must be added via AddNode.
		Connection(int noInputs, Node *output) : inputs(NULL), noInputs(noInputs), output(output)
		{
			if (noInputs > 0)
			{
				inputs = new Node*[noInputs];
			}
		}

		// Brief:
		//   Releases all input nodes.
		~Connection()
		{
			delete[] inputs;
		}

		// Brief:
		//   Adds an input node to the list of input nodes.
		void AddNode(Node *node, int nodeID)
		{
			inputs[nodeID] = node;
		}

		// Brief:
		//   Getter for input.
		inline Node* Input(size_t inputID) { return inputs[inputID]; }

		// Brief:
		//   Getter for noInputs.
		inline int NoInputs() { return noInputs; }

		// Brief:
		//   Getter for output.
		inline Node* Output() { return output; }

	private:
		Node * *inputs; int noInputs;
		Node *output;

	}; // class Connection
} // namespace LNTLib
