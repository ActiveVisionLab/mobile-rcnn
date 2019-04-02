//index.js
const mobilercnn = require('./build/Release/node-mobilercnn.node');

/*
console.log('addon',mobilercnn);

var http = require('http');
var port = process.env.PORT || 1337;

http.createServer(function (req, res) {
	mobilercnn.captureCameraImage();
	mobilercnn.runDetector();
	var image = mobilercnn.getResultImage();

	res.writeHead(200, { 'Content-Type': 'image/jpeg' });
    res.end(image);
}).listen(port);
*/

module.exports = mobilercnn;