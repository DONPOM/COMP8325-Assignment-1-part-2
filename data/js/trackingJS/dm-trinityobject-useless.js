
            var obj = {
		"Right":"b506a1bee96d3ef2f26f",
        	//"Middle":"2fab6b8ba53fe108415c",
		"Top":"5fa5ae00098d1d80e09b"
	    	};

	    if(navigator.appVersion.indexOf("MSIE 9.")!=-1 || navigator.appVersion.indexOf("MSIE 8.")!=-1) 
	    {
		obj = { "Right":"b506a1bee96d3ef2f26f"};
	    }

	    window.load_trinity = function(callback) {
      	    //Timeout in case of unexpected slow load
	    var st = new Date().getTime();
	    var td = navigator.userAgent.match(/webOS|Android|iPhone|iPad|iPod|BlackBerry|Windows Phone|IEMobile/i) ? 2000 : 1500;
	    var ran = false;
       	    var f = (function() {
            	try {
                     if (new Date().getTime() >= st + td) {
                           if (!ran)
                                  callback();
                           ran = true;
                           return;
                     }
              	} catch (e) {
              	}
              	setTimeout(f, 2);
       	    	});
       	    f();
	    //Loading the trinity object
            var sbi_script = document.createElement("script");
            sbi_script.async = true;
            sbi_script.type = "text/javascript";
            sbi_script.src = "http://apex.go.sonobi.com/trinity.js?key_maker=" + encodeURIComponent(JSON.stringify(obj)) + "&s=" + Math.floor(Math.random() * 1000);
	    if (callback == null) {
            	callback = function() {};
       	    	}
            if (sbi_script.readyState) {
            	sbi_script.onreadystatechange = function() {
                     if (sbi_script.readyState == "loaded" || sbi_script.readyState == "complete") {
                           sbi_script.onreadystatechange = null;
                           if (!ran)
                                  callback();
                           ran = true;
                     }
            	};
	    } else {
            	sbi_script.onload = function() {
                    if (!ran)
                    	callback();
                    ran = true;
              	};
           }	
           var node = document.getElementsByTagName("script")[0];
           node.parentNode.insertBefore(sbi_script, node);
	}	
	  
	var trinity_response = window.load_trinity(function(){});
        