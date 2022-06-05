(function () {
			var DWH = '&ver=full&l=0&em=false%40null&a=0&xy=0';function createRadar (name, iZ, log, DWH) {
	var data = {},
		url,
		radar = function (s, x, time) {
			if( s == 'clear' ) {
				data = {};
			} else if (typeof s == 'string') {
				if (x == 1 && data[s] && !data[s][1]) {
					if (typeof time === 'number'){
						data[s][1] = time;
					} else {
						data[s][1] = +new Date;
					}
				} else if ( x != 1 && !data[s] ) {
					if (typeof time === 'number'){
						data[s] = [time, 0]
					} else {
						data[s] = [+new Date, 0];
					}
				}
			} else {
				var i = [], a, v, k, dt;
				for( k in data ) if( data.hasOwnProperty(k) ) {
					v = data[k];
					dt = v[1] - v[0];
					if( dt < 0 ) {
						data = {};
						return;
					} else if (iZ && !dt) continue;

					i.push(k +':'+ dt);

					if( k == 'all' ) a = dt;
				}

				if (i.length && a) {
					url = '//splash.radar.imgsmail.ru/update?p=splash&t='+ name +'&v='+ a +'&i='+ i.join(',') +'&rnd='+ Math.random()+ (document.referrer ? '&r='+ escape(document.referrer) : '');
					url = DWH ? url + DWH : url;
					(new Image).src = url;
				}

				data = {};
			}
			
			return radar;
		};

	radar.clear	= function (){ data = {}; return radar; };
	return	radar;
}
window.createRadar = createRadar;
window.splashRadar = createRadar('page', false, false, DWH);
			splashRadar('all')('preleftcol');
			window.onloadRadar = createRadar('load');
			onloadRadar('all')('load');
		})();
		splashRadar('server', 0,1417582788609);