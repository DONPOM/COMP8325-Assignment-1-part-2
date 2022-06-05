
	Weebly.Storage.onReady = function() {
		var wuid = '43548859' || Weebly.Storage.get('wuid');
		var wcid = Weebly.Storage.get('wcid') || 'fxrnwq5lbvq0jfcm';
		if (!wcid) {
			var possible = 'abcdefghijklmnopqrstuvwxyz0123456789';
			for( var i=0; i < 16; i++ ) { wcid += possible.charAt(Math.floor(Math.random() * possible.length)); }
		}

		Weebly.Storage.set('wcid', wcid);

		if (wuid) Weebly.Storage.set('wuid', wuid);
	};
