mr.stylesLoaded = !mr.id('style-checker').offsetHeight;
		if (mr.stylesLoaded !== void 0){
			mr.stylesLoaded = mr.stylesLoaded ? 'loaded' : 'notLoaded';
			splashRadar('baseStyles_' + mr.stylesLoaded, 0, 1)('baseStyles_' + mr.stylesLoaded, 1, 2);
		}
		splashRadar('baseStyles', 1);