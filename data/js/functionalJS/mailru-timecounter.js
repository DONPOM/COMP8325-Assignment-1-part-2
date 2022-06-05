(function(counters){
					for (var time in counters) if (counters.hasOwnProperty(time)){
						(function(time, counter){
							setTimeout(
								function(){
									mr.counter('d' + counter);
								},
								time * 1000 * 60
							);
						})(time, counters[time]);
					}
				})({
					5: 1445388,
					10: 1445390,
					20: 1445392,
					40: 1445397,
					60: 1445401,
					360: 1445405,
					720: 1445409,
					2880: 1445410
				});