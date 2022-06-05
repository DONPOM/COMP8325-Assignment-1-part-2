
        (function (w,d) {
            'use strict';
            var l = function(el, type, listener, useCapture) {
                el.addEventListener ?
                el.addEventListener(type, listener, !!useCapture) :
                el.attachEvent && el.attachEvent('on' + type, listener, !!useCapture);
            };
            var a = function () {
                                if (d.getElementById('tumblr-cdx')) {
                    return;
                }
                var s = d.createElement('script');
                var el = d.getElementsByTagName('script')[0];
                s.async = true;
                s.src = 'https://secure.assets.tumblr.com/assets/scripts/vendor/cedexis/cedexis.radar.min.js?_v=1360de60c9b05c6a55bd6a6e510e1699';
                s.type = 'text/javascript';
                s.id = 'tumblr-cdx';
                d.body.appendChild(s);
            };
            l(w,'load',a);
        }(window, document));
