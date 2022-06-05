

    var navigationData = {"is_logged_in":"422403","is_https":true,"is_creator":false,"message_amount":false,"image":{"fullsize_url":"\/\/graph.facebook.com\/100001546764538\/picture?type=square","thumbnail_url":"\/\/graph.facebook.com\/100001546764538\/picture?type=square"},"login_url":"\/login?ru=http%3A%2F%2Fwww.patreon.com%2Fhome%3Frgsn%3D1","display_name":"Shohae Yu"};

    angular.module('App.services').config(["NavigationProvider", function(NavigationProvider) {

        NavigationProvider.setNavigationData(navigationData);

    }]);


    (function (e, b) {
        if (!b.__SV) {
            var a, f, i, g;
            window.mixpanel = b;
            a = e.createElement("script");
            a.type = "text/javascript";
            a.async = !0;
            a.src = ("https:" === e.location.protocol ? "https:" : "http:") + 
                '//cdn.mxpnl.com/libs/mixpanel-2.2.min.js';
            f = e.getElementsByTagName("script")[0];
            f.parentNode.insertBefore(a, f);
            b._i = [];
            b.init = function (a, e, d) {
                function f(b, h) {
                    var a = h.split(".");
                    2 == a.length && (b = b[a[0]], h = a[1]);
                    b[h] = function () {
                        b.push([h].concat(Array.prototype.slice.call(arguments, 0)))
                    }
                }

                var c = b;
                "undefined" !== typeof d ? c = b[d] = [] : d = "mixpanel";
                c.people = c.people || [];
                c.toString = function (b) {
                    var a = "mixpanel";
                    "mixpanel" !== d && (a += "." + d);
                    b || (a += " (stub)");
                    return a
                };
                c.people.toString = function () {
                    return c.toString(1) + ".people (stub)"
                };
                i =["disable",
                    "track",
                    "track_pageview",
                    "track_links",
                    "track_forms",
                    "register",
                    "register_once",
                    "alias",
                    "unregister",
                    "identify",
                    "name_tag",
                    "set_config",
                    "people.set",
                    "people.set_once",
                    "people.increment",
                    "people.append",
                    "people.track_charge",
                    "people.clear_charges",
                    "people.delete_user"];
                for (g = 0; g < i.length; g++)f(c, i[g]);
                b._i.push([a, e, d])
            };
            b.__SV = 1.2
        }
    })(document, window.mixpanel || []);
    mixpanel.init("cb7506f85e445f00907587f5033f7c6b");