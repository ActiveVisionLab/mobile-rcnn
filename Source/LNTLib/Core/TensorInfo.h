// Copyright 2015-2018 Oxford University Innovation and the authors of LightNet

#pragma once

namespace LNTLib
{
	class TensorInfo
	{
	public:
		int n, c, w, h;

		TensorInfo(int i_n, int i_c, int i_w, int i_h)
		{
			n = i_n; c = i_c; w = i_w; h = i_h;
		}

		TensorInfo()
		{
			n = 0; c = 0; w = 0; h = 0;
		}
	};
}
