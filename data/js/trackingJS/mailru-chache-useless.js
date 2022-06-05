function checkCache(data){
			var TS = parseInt(mr.timestamp / 1000, 10)

			var cached = TS > data.TS,
				modified = false;

			if (!cached){
				modified = data.IMS_TS !== 0;
			}
			if (cached){
				mr.counter('d1290319');
				splashRadar('cache_fromCache', 0, 0)('cache_fromCache', 1, 1);
			} else {
				if (modified){
					mr.counter('d1290330');
					splashRadar('cache_dirtyCache', 0, 0)('cache_dirtyCache', 1, 1);
				} else {
					mr.counter('d1290328');
					splashRadar('cache_clearCache', 0, 0)('cache_clearCache', 1, 1);
				}
			}
		};