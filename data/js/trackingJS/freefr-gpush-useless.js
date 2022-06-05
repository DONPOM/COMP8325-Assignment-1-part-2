 


  var _paq = _paq || [];

	//Mise en conformit� CNIL
	//limitation conservation cookies analytics internes 13 mois
		_paq.push([function() { 
			var self = this; 
			function getOriginalVisitorCookieTimeout() { 
			 var now = new Date(), 
			 nowTs = Math.round(now.getTime() / 1000), 
			 visitorInfo = self.getVisitorInfo(); 
			 var createTs = parseInt(visitorInfo[2]); 
			 var cookieTimeout = 33696000; 
			 var originalTimeout = createTs + cookieTimeout - nowTs; 
			 return originalTimeout; 
			 } 
			 this.setVisitorCookieTimeout( getOriginalVisitorCookieTimeout() ); 
		 }]); 
	// END CNIL

var niceUrl = 'www.free.fr'+document.URL.split('free.fr')[1].split('.html')[0]; 
  _paq.push(["setDocumentTitle",niceUrl]);
  _paq.push(["setCookieDomain", "*.free.fr"]);
  _paq.push(["setDomains", ["*.free.fr"]]);
  //_paq.push(["setDoNotTrack", true]);
  _paq.push(['trackPageView']);
  _paq.push(['enableLinkTracking']);
  (function() {
    var u=(("https:" == document.location.protocol) ? "https" : "http") + "://statsweb.proxad.net/";
    _paq.push(['setTrackerUrl', u+'piwik.php']);
    _paq.push(['setSiteId', 2]);
    var d=document, g=d.createElement('script'), s=d.getElementsByTagName('script')[0]; g.type='text/javascript';
    g.defer=true; g.async=true; g.src=u+'piwik.js'; s.parentNode.insertBefore(g,s);
  })();


			$('.clicka,.ss,.bottombar,button').click(function(){
				_paq.push(['setCustomVariable',1,'HomeLinks',$(this).attr('id')]);
			})
			




$(function(){
	
	$('#OptOut').on('click',function(){
		_paq.push(['setDoNotTrack', true]);
		_paq.push(['disableCookies']);
		_paq.push(['trackPageView']);
	})
})