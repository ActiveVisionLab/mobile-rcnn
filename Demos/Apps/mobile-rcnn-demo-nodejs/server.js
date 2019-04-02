const mobilercnn = require('./build/Release/node-mobilercnn.node');

// initializes the node
mobilercnn.initialize('../../../Models/MobileNetsV2/320_FPN', 0, 640, 480)

var http = require('http');
var port = process.env.PORT || 1337;

// we can now pull from the camera
console.log('Listening ...');

http.createServer(function (req, res) {
	console.log('----');
	
	// captures one image from the camera
	mobilercnn.captureCameraImage();
	
	// runs the detector on the image
	mobilercnn.runDetector();
	
	// draws the result on the image
	// image is jpeg compressed
	var image = mobilercnn.resultImage();
	
	// sends the image to the browser
	res.writeHead(200, { 'Content-Type': 'image/jpeg' });
    res.end(Buffer.from(image));
	
	// prints summary to console
	var noDetections = mobilercnn.noDetections();
	console.log('no detections: ' + noDetections);
	for (var i = 0; i < noDetections; i++) {
		// current box is mobilercnn.box(i)
		// current classId is mobilercnn.classId(i)
		// current score is mobilercnn.score(i)
		// current mask is mobilercnn.mask(i)
		
		var box = mobilercnn.box(i);
		var box_print = '[' + box[0].toFixed(3) + ', ' + box[1].toFixed(3) + ', ' + box[2].toFixed(3) + ', ' + box[3].toFixed(3) + ']';
		
		console.log(
			'found class ' + mobilercnn.classId(i) + 
			' with score ' + mobilercnn.score(i).toFixed(3)
			+ ' at location ' + box_print);
	}
	
	console.log('----');
}).listen(port);

// this should be run somewhere
//mobilercnn.shutdown()