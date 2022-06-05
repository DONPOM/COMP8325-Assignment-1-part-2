
				(function(){
					if (window.performance && window.performance.timing){
						var t = window.performance.timing;
						if (t.connectStart && t.connectEnd){
							splashRadar('connect', 0, t.connectStart)('connect', 1, t.connectEnd);
						}
						if (t.domainLookupStart && t.domainLookupEnd){
							var domainLookup = 'domainLookup' + (t.domainLookupEnd - t.domainLookupStart == 0 ? '0' : '');
							splashRadar(domainLookup, 0, t.domainLookupStart)(domainLookup, 1, t.domainLookupEnd);
						}
						if (t.requestStart && t.responseStart){
							splashRadar('request', 0, t.requestStart)('request', 1, t.responseStart);
						}
						if (t.responseStart && t.responseEnd){
							splashRadar('response', 0, t.responseStart)('response', 1, t.responseEnd);
						}
					}
				})();
				splashRadar('server', 1,1417582788621);
				splashRadar(true);

				var loadTimerEnd = new Date().valueOf(),
					loadTimer = parseInt(Math.round((loadTimerEnd - loadTimerStart) / 10) * 10, 10);

				(function() {
					if (mr.retina) {var info = [];
						info.push('retina');
						info.push('host=' + location.host);
						info.push('desktop');info.push('ratio=' + mr.dpr);
						//(new Image()).src = '//gstat.imgsmail.ru/gstat?ua=1&logme=' + encodeURIComponent(info.join('&')) + '&rnd=' + (new Date()).getTime() + Math.random();
					}

					if (mr.scriptLoaded) {
						mr.s_cookie[(mr.retina ? 'set' : 'remove') + 'Global']('rt', 1);
						mr.s_cookie[(mr.dpr != void 0 && mr.dpr > 1 ? 'set' : 'remove') + 'Global']('dpr', mr.dpr);
					}
				})();
				var aRadar = createRadar('additional');
				aRadar('all');
				aRadar('additionalJs');
				aRadar('additionalJSLoad');