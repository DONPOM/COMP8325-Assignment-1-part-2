
function IncludeJavaScript(jsFile, onLoadCallback) {
    var head = document.getElementsByTagName('head')[0] || document.documentElement;
    var s = document.createElement('script');
    s.type = 'text/javascript';
    s.async = true;
    s.src = jsFile;
    if (onLoadCallback) {
        s.onload = s.onreadystatechange = function() {
            if (!s.readyState || s.readyState == 'loaded' || s.readyState == 'complete') {
                s.onreadystatechange = null;
                onLoadCallback(s);
            }
        };
    }
    head.appendChild(s);
}
