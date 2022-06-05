
try {
function timeInterval(bP) {
	if (!rT[bP]) rT[bP] = new Date();
	else {
		rT.i[rT.i.length] = bP + '=' + ( (new Date()).getTime() - rT[bP].getTime() );
		delete rT[bP];
	}
}

var rT = {};
rT.i = [];
timeInterval('renderbrowser');
rT.addHandler = function(object, event, handler, useCapture) {
	if (object.addEventListener) object.addEventListener(event, handler, useCapture ? useCapture : false);
	else if (object.attachEvent) object.attachEvent('on' + event, handler);
};
rT.radar_type = 'main';
if (window.name.indexOf('srvt') != -1) {
	rT.srvt = ( new Date() ).getTime() - parseInt( window.name.substr(5) );
	window.name = '';
}

if (!window.patron) {
  rT.addHandler(document, 'click', function() {
  	if (!window.name.length || window.name.indexOf('srvt') > -1)
  		window.name = 'srvt=' + (new Date()).getTime();
  });
  rT.addHandler(document, 'keypress', function() {
  	if (window.name.indexOf('srvt') > -1) window.name = '';
  });
}

rT.img0 = new Image();
rT.img0.src = '//mail.radar.imgsmail.ru/update?' +
'p=mail0' +
(rT.radar_type != '' ? ('&t=' + rT.radar_type) : '') +
(rT.srvt ? ('&i=srvt:' + rT.srvt) : '') +
'&v=0&rnd=456498499' +
'';

} catch(e) {}
function timeInterval2(bP) {
    if (!rT[bP]) {
		rT[bP] = new Date();
    } else {
		var rT_cnt = rT.i.length;
		var found  = false;
		for (var j = 0; j < rT_cnt; j++) {
			var rT_array = rT.i[j].split("=");
			if (rT_array[0] == bP) {
				rT.i[j] = bP + '=' + (parseInt(rT_array[1]) + (new Date()).getTime() - rT[bP].getTime() );
				found = true;
			}
		}
		if (!found) {
			rT.i[rT.i.length] = bP + '=' + ( (new Date()).getTime() - rT[bP].getTime() );
		}
		delete rT[bP];
    }
}

