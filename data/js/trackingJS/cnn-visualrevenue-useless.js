
/*globals MainLocalObj*/
$j(window).load(function () {
'use strict';
MainLocalObj.init();
});
<!--Visual Revenue Reader Response Tracking Script (v6) -->
var _vrq = _vrq || [];
_vrq.push(['id', 396]);
_vrq.push(['automate', true]);
_vrq.push(['track', function(){}]);
(function(d, a){
var s = d.createElement(a),
x = d.getElementsByTagName(a)[0];
s.async = true;
s.src = 'http://a.visualrevenue.com/vrs.js';
x.parentNode.insertBefore(s, x);
})(document, 'script');
<!-- End of VR RR Tracking Script - All rights reserved -->
/**
* TODO:
* - HTML Comments need to be replaced with javascript comments.
* - This needs to be wrapped in its own function so that the variables
* do not mess with the global namespace.
* - The wrapped function needs to be moved into a separate javascript
* file so that it can be cached and the function called here.
* - All variables need to be defined at the top of the function scope
* - The if and else if logic can be combined since they set the same
* value.
* - The !== and === equality forms need to be used.
*/
<!-- Visual Revenue script for blog and partner links -->
var all_links = document.getElementsByTagName("a");
for (var i = 0 ; i < all_links.length ; i++) {
var href = all_links[i].getAttribute('href');
if (!href || all_links[i].getAttribute('onclick')) continue;
var boolOnClick = true;
/**
* All relative URLs and ones beginning with money.cnn.com,
* www.cnn.com, blogs.cnn.com or edition.cnn.com do not get the
* _vrtrack function applied to the onclick handler since those
* URLs all have the VR tracking markup on them so they do not
* need to be manually counted. All other URLs need to be
* manually tracked using the _vrtrack function.
*/
if (href.charAt(0) == "/") {
boolOnClick = false;
} else {
if (href.indexOf('money.cnn.com') !== -1 || href.indexOf("www.cnn.com") != -1 || href.indexOf("blogs.cnn.com") != -1 || href.indexOf("edition.cnn.com") != -1) {
boolOnClick = false;
}
}
if (boolOnClick == true) {
var cleanLink = href.split('&hpt=')[0].split('?hpt=')[0];
all_links[i].setAttribute('onclick', '_vrtrack({ track_url : \'' + cleanLink + '\' })');
}
}
<!-- End Visual Revenue script for blog and partner links -->
$j(document).ready(function () {
'use strict';
if (navigator.userAgent.indexOf('iPad') > -1) {
$j('#makeHPLink').hide();
}
loadChartbeat("homepage", "");
limitBinList($j('.cnn_sectbincntnt2'),90);
limitBinList($j('.cnn_sbin4mpopbin'),123);
});