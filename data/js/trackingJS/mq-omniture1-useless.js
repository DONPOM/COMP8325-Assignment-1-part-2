
    	/*<![CDATA[*/
    	bN_cfg = (function(d){
    		var host = location.hostname,
    			isProd = true,
    			cfg = {
					click: 0,
					h: [ host ],
					p: [],
					upgradeIE: false,
					view: 0
				};
			
			if (isProd) {
				if (host.indexOf('.co.uk')) {
					cfg.b = 'b.' + host.replace(/^www\./, '');
				}
			} else {
				cfg.b = 'b.aol.com';
			}
			
			return cfg;
		})(document);
		
    	function runOmni() {
	        try{
	        /* Disable Omniture optional features */
	            s_265.trackDownloadLinks=false
	            s_265.trackExternalLinks=false
	            s_265.trackInlineStats=false
	            s_265.linkLeaveQueryString=false
	            s_265.trackFormList=false
	            s_265.trackPageName=false
	            s_265.useCommerce=false
	
	            s_265.pfxID = (SITECONFIG.isOSM || ((DOMAIN && DOMAIN.country.toLowerCase()) != "us")) ? "mqi" : "map";
	            s_265.pageName = "";
	            s_265.server = "";
	            s_265.channel =  (DOMAIN)? m3.Omniture.getOmniChannel(DOMAIN.country) : "mq.mq"; 
	            s_265.pageType = "";
	            s_265.linkInternalFilters = "javascript:,mapquest.com,mqcdn.com";
                s_265.prop1 = SITECONFIG.isOSM ? "MQOSM" : ( ( (DOMAIN && DOMAIN.country.toLowerCase()) != "us") ? "MQ"+DOMAIN.country.toUpperCase() : 'MQ10MQmain');
                s_265.prop2 = SITECONFIG.isOSM ? "mq.osm" : 'MQ10mq.main';
	            s_265.prop12 = m3.Omniture.generateProp12();
	            if((!s_265.prop16) || (s_265.prop16 == "")) { s_265.prop16 = s_265.getQueryParam('cid'); }
	            
	            s_265.mmxgo = true;
	        	s_265.mmxcustom = null;
	        	
	        }catch(e){}
	    }
        /*]]>*/
    