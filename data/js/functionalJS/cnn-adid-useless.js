
        function cnnad_getTld (hostname)
        {
               var data = hostname.split(".");
               if (data.length >= 2)
               {
                       return (data[data.length-2] + "." + data[data.length-1]);
               }
               return(null);
        }
 
        var cnnDocDomain = cnnad_getTld(location.hostname);
        if(cnnDocDomain) {document.domain = cnnDocDomain;}

	function cnnGetId(query) {
		var keyValPairs = query.split('&');
		if(!keyValPairs) {
			keyValPairs = new Array();
			keyValPairs[keyValPairs.length]=query;
		}
		for(var counter=0; counter < keyValPairs.length;counter++) {
			var keyVal = keyValPairs[counter].split('=');
			if(keyVal[0]=='domId') {
				//alert("returning id: " + keyVal[1]);
				return keyVal[1];
			}
		}
	}
			
	function cnnSendData2() {
		var docId = cnnGetId(decodeURIComponent(location.pathname.substring(9)));
		var height = 90;
		var width = 120;
		var targetWindow = top;
		try
		{
			if(parent.cnnad_showAd)
			{
				targetWindow = parent;
			}
		} catch(err) {} 
		targetWindow.cnnad_showAd(docId);
		if ((height > 0)&&(width > 0))
		{
			targetWindow.cnnad_setAdSize(docId,height,width);
		}
	}
