{
    "targets": [{
        "target_name": "node-mobilercnn",
        "cflags!": [ "-fno-exceptions" ],
        "cflags_cc!": [ "-fno-exceptions" ],
        "sources": [
            "cppsrc/main.cpp"
        ],
        "copies":
        [
			{
				'destination': '<(module_root_dir)',
				'files': ['<(module_root_dir)/../../../Library/bin/windows-vc15/LNTLibRedist.dll']
			}
        ],		
        'include_dirs': [
            "<!@(node -p \"require('node-addon-api').include\")",
			"../../../Library/include",
			"S:/SDK/opencv/build/include"
        ],
        'libraries': [
			"../../../../Library/lib/windows-vc15/LNTLibRedist.lib",
			"S:/SDK/opencv/build/x64/vc15/lib/opencv_world341.lib"
		],
        'dependencies': [
            "<!(node -p \"require('node-addon-api').gyp\")"
        ],
        'defines': [ 'NAPI_DISABLE_CPP_EXCEPTIONS' ]
    }]
}