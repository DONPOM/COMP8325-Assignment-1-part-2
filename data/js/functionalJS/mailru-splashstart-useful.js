splashRadar('headline', 1);
		if (window.__PH){
			var __mailUpdated, __okUpdated;
			__PH.once('update', function(e, next){
				__mailUpdated = e.data;
				next();
			});
			__PH.once('OKUpdate', function(e, next){
				__okUpdated = e.data;
				next();
			});
		}