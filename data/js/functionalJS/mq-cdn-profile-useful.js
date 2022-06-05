
    if (typeof jQuery !== 'undefined') {  
        $(window).load( function() {
            if (document.createStyleSheet){
                document.createStyleSheet("http://content.mqcdn.com/winston-624/cdn/loader.css.pre$profile=winstonA");
            } else {
                $("head").append($("<link rel='stylesheet' href='http://content.mqcdn.com/winston-624/cdn/loader.css.pre$profile=winstonA' type='text/css' media='screen' />"));
            }
            
            var referrer = window.SITECONFIG.referringUrl || null;
        	
        	if (referrer) {
            	$('#backToAol').attr({ href: unescape(referrer) });
        	}
        });
    }
