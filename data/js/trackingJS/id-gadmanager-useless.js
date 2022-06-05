
//appendTag function
function appendTag(tagSource, scope, callBack) {
	var script = document.createElement('script');
	script.type = 'text/javascript';
	script.async = 'true';
	script.src = tagSource;
        if(callBack) script.onload = callBack;
	if(scope) scope.appendChild(script);
	else document.getElementsByTagName('head')[0].appendChild(script);
};
//Tag Manager
function _loadTagManager(dataLayerVar, iD) {
	window[dataLayerVar] = window[dataLayerVar] || [];
	window[dataLayerVar].push({
		'gtm.start': new Date().getTime(),
		event: 'gtm.js'
	});
	var gtmScript = '//www.googletagmanager.com/gtm.js?id=' + iD +'&l='+ dataLayerVar;
	appendTag(gtmScript);
}
//Call the Tag Management Script
if(!window.coreDataLayer) _loadTagManager('coreDataLayer', 'GTM-5PCVLF');
//End Tag Manager
