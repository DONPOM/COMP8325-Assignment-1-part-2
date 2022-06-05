/* ---------------------------------------------------------------------------------
   file-start: js/ads/adfox.reload_code.embeds.js 
*/

/**
	Namespace module;
	methods:
		window.ADFOX.getNSReference
	properties:
		window.ADFOX
	Modify: 21-06-2013
*/
(function(){
	window.ADFOX = (window.ADFOX) || {};
	window.ADFOX.getNSReference = (window.ADFOX.getNSReference) || function(referenceString, value)
	{
		var referenceArray = referenceString.split('.'), 
			currentElement = window.ADFOX; 
			if (referenceArray[0] === "ADFOX") { 
				referenceArray = referenceArray.slice(1); 
			} 
			for (var i = 0, len = referenceArray.length; i < len ; i += 1) { 
				if (typeof currentElement[referenceArray[i]] === "undefined") { 
					currentElement[referenceArray[i]] = {}; 
				}
				currentElement = currentElement[referenceArray[i]]; 
			}
		return currentElement;
	};
})();

/**
	Embed code module;
	API in window.ADFOX.RELOAD_CODE:
		methods:
			loadBanner
				@param {String} placeholderId
				@param {String} requestSrc
				@param {Number} sessionId
			clearSession
			getSessionId
	API in window.ADFOX.RELOAD_CODE.embeds:
		properties:
			'<id áàííåðíîãî ìåñòà RELOAD_CODE>':{
				placeholderId: '<id placeholder adfox>',
				requestSrc: '<ññûëêà çàïðîñà çà áàíåðîì>'
			}
			
*/
(function(){
	var 
		tgEmbeds = window.ADFOX.getNSReference('RELOAD_CODE.embeds'),
		tgNS = window.ADFOX.getNSReference('RELOAD_CODE'),
		pageReferrer = (typeof(document.referrer) != 'undefined') ? escape(document.referrer) : '';
		
	/**
		API method;
		returns id for current session, id is the same for all request;
	*/
	function getSessionId()
	{
		tgNS.sessionId = (tgNS.sessionId) ? tgNS.sessionId : (Math.floor(Math.random() * 1000000))
		return tgNS.sessionId;
	}
	
	/**
		API method;
		clears current session; Subsequent getSessionId call will create new id;
	*/
	function clearSession()
	{
		tgNS.sessionId = 0;
	}
	
	/**
		Private method;
		sends new banner request, reuses existing iframe;
		@param {String} placeholderId
		@param {String} requestSrc
	*/
	function sendRequest(placeholderId, requestSrc){
		var iframeDocument = null,
			iframeId = 'AdFox_iframe_' + placeholderId;
		
		try{
			if(document.all && !window.opera){
				iframeDocument = window.frames[iframeId].document;
			}
			else if(document.getElementById){
				iframeDocument = document.getElementById(iframeId).contentDocument;
			}
		}
		catch(e){}
		
		if(iframeDocument){
			iframeDocument.write('<scr'+'ipt type="text/javascript" src="'+requestSrc+'"><\/scr'+'ipt>');
		}
	}
	
	/**
		API method;
		forms parameters for banner request, call sendRequest method;
		@param {String} placeholderId
		@param {String} requestSrc
		@param {Number} sessionId
	*/
	function loadBanner(placeholderId, requestSrc, sessionId){
		var 
			addate = new Date(),
			dl = escape(document.location);

		var dynamicParameters = 
			'&amp;pt=b' +
			'&amp;prr=' + pageReferrer + //closure;
			'&amp;pr1=' + placeholderId + 
			'&amp;pr=' + sessionId +  
			'&amp;pd=' + addate.getDate() + 
			'&amp;pw=' + addate.getDay() + 
			'&amp;pv=' + addate.getHours() + 
			'&amp;dl=' + dl;

		document.getElementById('AdFox_banner_'+placeholderId).innerHTML = '';;
		
		sendRequest(placeholderId, requestSrc + dynamicParameters);
	};

	function initBanner(bannerPlaceId,requestSrc) {
		var 
			pr1 = Math.floor(Math.random() * 1000000),
			placeholderHtml = '<div id="AdFox_banner_'+pr1+'"></div>',
			iframeHtml = '<div style="visibility:hidden;position:absolute;top:0;right:0;"><iframe id="AdFox_iframe_'+pr1+'" width=1 height=1 marginwidth=0 marginheight=0 scrolling=no frameborder=0></iframe></div>',
			html = placeholderHtml + '\n' + iframeHtml;

		tgEmbeds[bannerPlaceId] = {
				'placeholderId': pr1,
				'requestSrc': requestSrc
			};

		return {
			placeholderHtml: placeholderHtml,
			sessionId: getSessionId(),
			iframeHtml: iframeHtml,
			html: html,
			pr1: pr1
		}
	}
	
	function reloadBanners(bannerPlaceId) {
		var tgNS = window.ADFOX.getNSReference('RELOAD_CODE'),
			tgEmbeds = window.ADFOX.getNSReference('RELOAD_CODE.embeds');
			
		tgNS.clearSession();

		if (typeof(bannerPlaceId) == "object" || !bannerPlaceId)  {//EXAMPLE 1; reload all banners;
			for(var currentEmbed in tgEmbeds) {
				tgNS.loadBanner(
					tgEmbeds[currentEmbed].placeholderId,
					tgEmbeds[currentEmbed].requestSrc,
					tgNS.getSessionId()
				);
			}
		} else {//EXAMPLE 2; reload any banner;
			tgNS.loadBanner(
				tgEmbeds[bannerPlaceId].placeholderId,
				tgEmbeds[bannerPlaceId].requestSrc,
				tgNS.getSessionId()
			);
		}
	}

	/**
		Open API methods in namespace;
	*/
	
	tgNS.loadBanner = loadBanner;
	tgNS.getSessionId = getSessionId;
	tgNS.clearSession = clearSession;
	tgNS.initBanner = initBanner;
	tgNS.reloadBanners = reloadBanners;
})();

/*
** Èìÿ ôóíêöèè, ïðè âûçîâå êîòîðîé áóäåò ïåðåçàãðóæàòüñÿ áàííåð.
** Åñëè íè ÷åãî íå óêàçûâàòü äîñòóï ê ôóíêöèÿ áóäåò äîñòóïíà â ïðîñòðàíñòâå èìåí ADFOX (ïóòü ê ôóíêöèè) "window.ADFOX.RELOAD_CODE.reloadBanners".
** Ïîóìîë÷àíèþ ïåðåäàåòñÿ "adfox_reloadBanner".
*/
(function(functionName){
	if (functionName) {
		window[functionName] = window.ADFOX.RELOAD_CODE.reloadBanners;
	}		
})('adfox_reloadBanner');

/* file-end: js/ads/adfox.reload_code.embeds.js 
----------------------------------------------------------------------------------*/
/* ---------------------------------------------------------------------------------
   file-start: js/ads/adfox.asyn.code.scroll.js 
*/

function AdFox_getWindowSize() {
var winWidth,winHeight;
	if( typeof( window.innerWidth ) == 'number' ) {
		//Non-IE
		winWidth = window.innerWidth;
		winHeight = window.innerHeight;
	} else if( document.documentElement && ( document.documentElement.clientWidth || document.documentElement.clientHeight ) ) {
		//IE 6+ in 'standards compliant mode'
		winWidth = document.documentElement.clientWidth;
		winHeight = document.documentElement.clientHeight;
	} else if( document.body && ( document.body.clientWidth || document.body.clientHeight ) ) {
		//IE 4 compatible
		winWidth = document.body.clientWidth;
		winHeight = document.body.clientHeight;
	}
	return {"width":winWidth, "height":winHeight};
}//END function AdFox_getWindowSize

function AdFox_getElementPosition(elemId){
    var elem;
    
    if (document.getElementById) {
		elem = document.getElementById(elemId);
	}
	else if (document.layers) {
		elem = document.elemId;
	}
	else if (document.all) {
		elem = document.all.elemId;
	}
    var w = elem.offsetWidth;
    var h = elem.offsetHeight;	
    var l = 0;
    var t = 0;
	
    while (elem)
    {
        l += elem.offsetLeft;
        t += elem.offsetTop;
        elem = elem.offsetParent;
    }

    return {"left":l, "top":t, "width":w, "height":h};
} //END function AdFox_getElementPosition

function AdFox_getBodyScrollTop(){
	return self.pageYOffset || (document.documentElement && document.documentElement.scrollTop) || (document.body && document.body.scrollTop);
} //END function AdFox_getBodyScrollTop

function AdFox_getBodyScrollLeft(){
	return self.pageXOffset || (document.documentElement && document.documentElement.scrollLeft) || (document.body && document.body.scrollLeft);
}//END function AdFox_getBodyScrollLeft

function AdFox_Scroll(elemId,elemSrc){
   var winPos = AdFox_getWindowSize();
   var winWidth = winPos.width;
   var winHeight = winPos.height;
   var scrollY = AdFox_getBodyScrollTop();
   var scrollX =  AdFox_getBodyScrollLeft();
   var divId = 'AdFox_banner_'+elemId;
   var ltwhPos = AdFox_getElementPosition(divId);
   var lPos = ltwhPos.left;
   var tPos = ltwhPos.top;

   if(scrollY+winHeight+5 >= tPos && scrollX+winWidth+5 >= lPos){
      AdFox_getCodeScript(1,elemId,elemSrc);
	  }else{
	     setTimeout('AdFox_Scroll('+elemId+',"'+elemSrc+'");',100);
	  }
}//End function AdFox_Scroll;

/* file-end: js/ads/adfox.asyn.code.scroll.js 
----------------------------------------------------------------------------------*/
