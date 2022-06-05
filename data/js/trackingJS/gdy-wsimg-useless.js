
// defer load liveperson
$(window).load(function(){
$('body').append($('<div id="lpButtonDiv"></div>'))
var scriptResource = document.createElement('script');
scriptResource.src = "https://img1.wsimg.com/fos/liveperson/js/liveperson_20141013a.min.js";
var styleResource = document.createElement('link');
styleResource.rel = "stylesheet";
styleResource.type = "text/css";
styleResource.href = "https://img1.wsimg.com/fos/liveperson/css/chat-window_20140205.css";
var script = document.getElementsByTagName('script')[0];
script.parentNode.insertBefore(scriptResource, script);
script.parentNode.insertBefore(styleResource, script);
});
