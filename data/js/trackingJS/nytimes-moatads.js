/*Copyright (c) 2011-2014 Moat Inc. All Rights Reserved.*/
try {
    (function(t, r) {
        function da(a, l) {
            for (var m = [a], e = 1; e <= l; e++)
                m.push(a + e), m.push(a - e);
            return m[r.floor(r.random() * m.length)]
        }
        function ea() {
            var a = da(8, 2);
            this.inSample = function() {
                return 0 == r.floor(r.random() * a)
            };
            this.getMultiplier = function() {
                return a
            }
        }
        function fa(a) {
            var l = new ea;
            if (J && l.inSample()) {
                var m = A.document;
                (a = m.getElementsByTagName("script")[0]) || (a = m.getElementsByTagName("head")[0]);
                if (a) {
                    var m = m.createElement("script"), e = "//" + q.b.j + "/moatpg195nulK89/moatad.js", d = {
                        i: "NYTIMES2"
                    };
                    (l = l.getMultiplier()) &&
                    (d.cm = l);
                    e += "#" + q.l.e(d);
                    m.src = e;
                    m.async=!0;
                    m.type = "text/javascript";
                    a.parentNode.insertBefore(m, a);
                    a=!0
                } else 
                    a=!1;
                return a
            }
            return !1
        }
        function Z() {
            for (var a = 0; a < F.length; a++)
                window.clearTimeout(F[a]);
            for (a = 0; a < M.length; a++)
                window.clearInterval(M[a]);
            for (var l in w)
                w.hasOwnProperty && w.hasOwnProperty(l) && w[l] && (window.clearTimeout(w[l].tid), w[l]=!1);
            F = [];
            M = []
        }
        function $() {
            for (var a in u)
                if (u.hasOwnProperty(a)) {
                    var l = u[a];
                    q.h.r(l);
                    q.c.i(l)
                }
            Z()
        }
        var q = {};
        (function(a) {
            function l(a) {
                return a && a.document &&
                a.location && a[c + b] && a[h + f]
            }
            function m(a) {
                if (null == a || l(a))
                    return !1;
                var k = a.length;
                return 1 === a.nodeType && k?!0 : "string" === typeof a || e(a) || 0 === k || "number" === typeof k && 0 < k && k - 1 in a
            }
            function e(g) {
                return "[object Array]" === a.a.af.toString.call(g)
            }
            a.a = {};
            a.a.a = 3E3;
            a.a.b = function() {
                var a = /Firefox\/(\d+)/.exec(navigator.userAgent);
                return a ? 14 < parseInt(a[1], 10) : !1
            }();
            a.a.c = function() {
                var g;
                g = /^(?:[a-z]+:\/\/|:?\/?\/)?(?:www\.)?([^\/:]*)/i;
                g=!a.b.a && a.b.b ? (g = a.b.b.match(g)) && g[1] ? g[1] : a.b.c.location.hostname :
                a.b.c.location.hostname;
                var k = g.match(/.*\.([^\.]*\..*)/i);
                return k && k[1] ? decodeURIComponent(k[1]) : decodeURIComponent(g)
            };
            a.a.d = function(a, k) {
                for (var n = [a], b = 1; b <= k; b++)
                    n.push(a + b), n.push(a - b);
                n = n[r.floor(r.random() * n.length)];
                b = r.floor(r.random() * n);
                return {
                    multiplier: n,
                    sample: 0 == b
                }
            };
            a.a.e = function(g, b) {
                var n = a.a.d(g, b);
                a.a.e = function() {
                    return n
                };
                return n
            };
            a.a.f = function() {
                var g = a.a.g();
                return 5 === g || 6 === g || 7 === g
            };
            a.a.g = function() {
                if (!navigator)
                    return null;
                var a = navigator.userAgent;
                return "Microsoft Internet Explorer" ==
                navigator.appName ? parseInt(a.replace(/^.*MSIE (\d+).*$/, "$1"), 10) : "Netscape" == navigator.appName && (a = a.match(/Trident\/.*rv:(\d+)/)) ? parseInt(a[1], 10) : null
            };
            a.a.h = function(a, b) {
                for (var n = 0; n < a.length; n++)
                    if (a[n] === b)
                        return !0;
                return !1
            };
            a.a.i = function() {
                function a(g) {
                    g = g.match(/[\d]+/g);
                    g.length = 3;
                    return g.join(".")
                }
                var b=!1, n = "";
                if (navigator.plugins && navigator.plugins.length) {
                    var c = navigator.plugins["Shockwave Flash"];
                    c && (b=!0, c.description && (n = a(c.description)));
                    navigator.plugins["Shockwave Flash 2.0"] &&
                    (b=!0, n = "2.0.0.11")
                } else if (navigator.mimeTypes && navigator.mimeTypes.length)(b = (c = navigator.mimeTypes["application/x-shockwave-flash"]) && c.enabledPlugin) && (n = a(c.enabledPlugin.description));
                else 
                    try {
                        c = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7"), b=!0, n = a(c.GetVariable("$version"))
                } catch (e) {
                    try {
                        c = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6"), b=!0, n = "6.0.21"
                    } catch (d) {
                        try {
                            c = new ActiveXObject("ShockwaveFlash.ShockwaveFlash"), b=!0, n = a(c.GetVariable("$version"))
                        } catch (K) {}
                    }
                }
                return b ? n : "0"
            };
            a.a.j = function(a, b) {
                for (var c = 0; c < a.length; c++)
                    if ( - 1 < b.indexOf(a[c]))
                        return a[c];
                return !1
            };
            a.a.k = function(a) {
                return new t - a.de
            };
            a.a.l = function(a) {
                return a.replace(/^http:/, "").replace(/^\/\//, "").replace(/^www[^.]*\./, "").split("/")[0]
            };
            a.a.m = function(g, b, c) {
                if (("undefined" === typeof c ||!c) && g && (c = a.c.a(g), !c))
                    return;
                if (g && g.nodeType)
                    if ("undefined" === typeof Node) {
                        if (1 != g.nodeType)
                            return 
                    } else if (g.nodeType != Node.ELEMENT_NODE)
                        return;
                if (c.getComputedStyle)
                    return c.getComputedStyle(g, "")[b];
                for (c = b.indexOf("-"); - 1 <
                c;)
                    b = c == b.length - 1 ? b.substr(0, c) : b.substr(0, c) + b.charAt(c + 1).toUpperCase() + b.substr(c + 2), c = b.indexOf("-");
                return g.currentStyle ? g.currentStyle[b] : g.style[b]
            };
            a.a.n = function(g) {
                if (!g)
                    return !1;
                g = a.a.m(g, "background-image");
                var b;
                g && (b = (b = g.match("url\\((.*)\\)")) && b[1]);
                return b
            };
            a.a.o = function(g, b) {
                if (!g)
                    return !1;
                var c = [g], e=!1;
                a.a.forEach("number" === typeof b ? b : 50, function() {
                    if ((e = g.parentNode || g.parentElement ||!1) && 1 == e.nodeType)
                        g = e, c.push(g);
                    else 
                        return !1
                });
                return c
            };
            a.a.p = function(g, b) {
                var c = "number" ===
                typeof b ? b: 50, e = [], d = a.c.a(g);
                if (d)
                    e.push(d);
                else 
                    return e;
                a.a.forEach(c, function() {
                    var b = a.d.a(g, d);
                    return b && (d = a.c.a(b)) ? (e.push(d), !0) : !1
                });
                return e
            };
            a.a.q = function() {
                return null !== /iPad|iPhone|iPod|Silk|Kindle|Android|BlackBerry|PlayBook|BB10|Windows Phone/.exec(navigator.userAgent)
            };
            a.a.r = function() {
                return null !== /iPhone|iPod/.exec(navigator.userAgent)
            };
            a.a.s = function() {
                var a = navigator.userAgent;
                return a.match(/iPhone|iPod|iPad/)&&!a.match(/Safari/i)
            };
            a.a.t = function() {
                var a = window;
                try {
                    return "undefined" !==
                    typeof a.external && "undefined" !== typeof a.external.InPrivateFilteringEnabled && a.external.InPrivateFilteringEnabled() || "undefined" !== typeof a.external && "undefined" !== typeof a.external.msTrackingProtectionEnabled && a.external.msTrackingProtectionEnabled() || "undefined" !== typeof a._gaUserPrefs && a._gaUserPrefs.ioo && a._gaUserPrefs.ioo() || "undefined" !== typeof navigator.doNotTrack && ("yes" === navigator.doNotTrack ||!0 === navigator.doNotTrack) || "undefined" !== typeof a._gaUserPrefs&&!0 === a._gaUserPrefs
                } catch (b) {
                    return !1
                }
            };
            a.a.getAttribute = function(a, b) {
                return a[b] || a.getAttribute(b)
            };
            a.a.u = function(a) {
                var b = 0, c;
                for (c in a)
                    a.hasOwnProperty(c) && (b += 1);
                return b
            };
            var d = [function(a) {
                if (!a || "IFRAME" !== a.nodeName)
                    return !1;
                var b = a.offsetHeight;
                return isNaN(b) || 15 < b || "google_conversion_frame" !== a.name?!1 : !0
            }
            ];
            a.a.v = function(b, c) {
                return !a.a.w(b) || a.a.filter(d, function(a) {
                    return a(b)
                }).length ||!0 === b[y]?!1 : !0
            };
            a.a.w = function(b) {
                return b.offsetWidth * b.offsetHeight >= a.a.a
            };
            a.a.x = function(a, b) {
                var c=!1, e = b.body, d = e && e.firstChild;
                e && d && (e.insertBefore(a, d), c=!0);
                return c
            };
            a.a.y = function(a, b) {
                if ("string" !== typeof a ||!b ||!b.insertBefore ||!b.ownerDocument)
                    return !1;
                var c = b.ownerDocument.createElement("script");
                c.type = "text/javascript";
                c.async=!0;
                b.insertBefore(c, b.childNodes[0]);
                c.src = a;
                return !0
            };
            a.a.z = function() {
                var a;
                return function() {
                    if (!a)
                        a: {
                        for (var b = document.getElementsByTagName("script"), c = b.length - 1; - 1 < c; c--)
                            if ((a = b[c]) && a.src && a.src.indexOf&&-1 !== a.src.indexOf(ga + "/moatad.js") && ("undefined" === typeof a[y] ||!0 !== a[y])) {
                                a[y] =
                                !0;
                                break a
                            }
                            a = void 0
                    }
                    return a ? (a[y]=!0, a) : null
                }
            }();
            a.a.aa = function(a) {
                try {
                    return - 1 !== (a.src || a.getAttribute("src")).indexOf("psd=1")
                } catch (b) {
                    return !1
                }
            };
            a.a.ab = function(b) {
                if (v && v.yh && v.yh.qa && v.yh.qa()&&!a.a.q())
                    try {
                        var c = v.yh.qb(b), n = '<object type="application/x-shockwave-flash" classid="clsid:d27cdb6e-ae6d-11cf-96b8-444553540000" codebase="http://download.macromedia.com/pub/shockwave/cabs/flash/swflash.cab#version=8,0,0,0" width="1" height="1" id="moatCap"><param name="allowScriptAccess" value="always" /><param name="allowFullScreen" value="false" /><param name="movie" value="' +
                        c.src + '" /><param name="quality" value="high" /><param name="bgcolor" value="#ffffff" /><embed type="application/x-shockwave-flash" src="' + c.src + '" quality="high" bgcolor="#ffffff" width="1" height="1" id="moatCapEmbed" align="middle" allowScriptAccess="always" allowFullScreen="false" type="application/x-shockwave-flash" pluginspage="http://www.adobe.com/go/getflashplayer" /></object>', e = A.document.createElement("div");
                        e.style.position = "absolute";
                        e.style.left = "-9999px";
                        e.style.top = "-9999px";
                        a.e.a(function() {
                            if (a.a.x(e,
                            A.document))
                                return e.innerHTML = n, !0
                            }, 3, 100)
                } catch (d) {}
            };
            a.a.ac = function(a) {
                for (var b = [], c = 0; c < a.length; c++)
                    b.push(a[c]);
                return b
            };
            a.a.previousElementSibling = function(a) {
                if (a.previousElementSibling)
                    return a.previousElementSibling;
                for (; a = a.previousSibling;)
                    if (1 === a.nodeType)
                        return a
            };
            a.a.ad = function(a, b, c) {
                "undefined" !== typeof c && (a[b] = c)
            };
            a.a.filter = function(a, b) {
                for (var c = [], e = 0; e < a.length; e++)
                    b(a[e]) && c.push(a[e]);
                return c
            };
            a.a.ae = function(a, b) {
                for (var c = [], e = 0; e < b.length; e++)
                    c.push(a(b[e]));
                return c
            };
            a.a.indexOf = function(a, b) {
                if ({}.toString.call(a).match(/\[object array\]/i)) {
                    var c;
                    for (c = 0; c < a.length; c++)
                        if (a[c] === b)
                            return c;
                    return - 1
                }
                if ("string" === typeof a)
                    return a.indexOf(b)
            };
            a.a.bind = function(a, b) {
                return function() {
                    a[b].apply(a, arguments)
                }
            };
            var c = "ale", b = "rt", h = "setInte", f = "rval";
            a.a.af = {};
            a.a.ag = function(a, b) {
                if (a && b && b.childNodes) {
                    var c = b.childNodes;
                    0 < c.length ? b.insertBefore(a, c[0]) : b.appendChild(a)
                }
            };
            a.a.ah = function(b, c) {
                if (!b ||!c)
                    return !1;
                var e = a.a.ai(c);
                if (!e)
                    return !1;
                a.a.hasChildNodes(e) ?
                e.insertBefore(b, e.childNodes[0]) : e.appendChild(b);
                return e
            };
            a.a.hasChildNodes = function(a) {
                return a && a.childNodes && 0 < a.childNodes.length
            };
            a.a.ai = function(b) {
                if (!b)
                    return !1;
                if ("OBJECT" !== b.nodeName && "EMBED" !== b.nodeName)
                    return b;
                b = a.a.o(b);
                var c=!1;
                a.a.forEach(b, function(a) {
                    if (a && "OBJECT" !== a.nodeName && "EMBED" !== a.nodeName)
                        return c = a, !1
                });
                return c
            };
            a.a.aj = function(a, b) {
                if ("undefined" === typeof a)
                    return !1;
                for (var c = 0, e = b.length; c < e; c++)
                    if ("string" == typeof b[c] && (a = a[b[c]], "undefined" === typeof a))
                        return !1;
                return a
            };
            a.a.ak = function(a) {
                return encodeURIComponent(a.moatClientLevel1 + ":" + a.moatClientLevel2 + ":" + a.moatClientLevel3 + ":" + a.moatClientLevel4)
            };
            a.a.al = function(a) {
                return u && "undefined" !== typeof a && u[a] ? u[a] : !1
            };
            a.a.am = e;
            a.a.an = l;
            a.a.ao = m;
            a.a.forEach = function(a, b, c, e) {
                var d, s = typeof a;
                if (a)
                    if ("function" === s)
                        for (d in a) {
                            if ("prototype" != d && "length" != d && "name" != d && (e ||!a.hasOwnProperty || a.hasOwnProperty(d)) && (s = b.call(c, a[d], d), "boolean" === typeof s&&!s))
                                break
                        } else if ("number" === s)
                            for (d = 0; d < a && (s = b.call(c,
                            a, d), "boolean" !== typeof s || s); d++);
                        else if ("function" === typeof a.every)
                            a.every(function(a, e, d) {
                                a = b.call(c, a, e);
                                return !("boolean" === typeof a&&!a)
                            });
                        else if (m(a))
                            for (d = 0; d < a.length && (s = b.call(c, a[d], d), "boolean" !== typeof s || s); d++);
                        else 
                            for (d in a)
                                if (e || a.hasOwnProperty(d))
                                    if (s = b.call(c, a[d], d), "boolean" === typeof s&&!s)
                                        break;
                return a
            };
            a.a.ap = function(b) {
                return b && a.a.filter(b.children, function(a) {
                    return 1 === a.nodeType
                })
            }
        })(q);
        (function(a) {
            function l(a) {
                try {
                    var d = typeof a.location.toString;
                    if ("undefined" ===
                    d || "unknown" === d)
                        return !0;
                    var c = typeof a.document;
                    return "undefined" === c || "unknown" === c?!0 : !1
                } catch (b) {
                    return !0
                }
            }
            a.b = {};
            a.b.d = "MoatSuperV5";
            a.b.e = 0;
            a.b.f = 0 < Object.prototype.toString.call(window.HTMLElement).indexOf("Constructor");
            a.b.g =- 1 !== navigator.userAgent.toLowerCase().indexOf("firefox");
            a.b.h =- 1 !== navigator.userAgent.indexOf("MSIE");
            a.b.i = (new t).getTime();
            a.b.protocol = function() {
                var a;
                try {
                    a = window.location.protocol
                } catch (d) {
                    a = document.createElement("a"), a.href = "", a = a.protocol
                }
                return a
            }();
            a.b.j =
            ("https:" === a.b.protocol ? "z" : "js") + ".moatads.com";
            a.b.k = window != window.parent;
            var m = l(window.parent);
            a.b.l = a.b.k&&!m;
            a.b.a = m?!1 : !l(window.top);
            a.b.c = a.b.a ? window.top : a.b.l ? window.parent : window;
            a.b.b = a.b.c.document.referrer
        })(q);
        (function(a) {
            function l(a) {
                return function() {
                    var d=!1;
                    return function(c) {
                        try {
                            return a(c)
                        } catch (b) {
                            if (!d) {
                                d=!0;
                                c = (new t).getTime();
                                window["Moat#ETS"] || (window["Moat#ETS"] = c);
                                window["Moat#EMC"] || (window["Moat#EMC"] = 0);
                                var h = 36E5 <= c - window["Moat#ETS"], f = b.name + " in closure: " +
                                b.message + ", stack=" + b.stack;
                                if (!h && 10 > window["Moat#EMC"]) {
                                    window["Moat#EMC"]++;
                                    try {
                                        var g = "//v4.moatads.com/pixel.gif?e=24&d=data%3Adata%3Adata%3Adata&i=" + escape("NYTIMES2") + "&ac=1&k=" + escape(f) + "&ar=" + escape("4bb5684-clean") + "&j=" + escape(document.referrer) + "&cs=" + (new t).getTime(), k = new Image(1, 1);
                                        k.src = g
                                    } catch (n) {}
                                } else if (h) {
                                    window["Moat#EMC"] = 1;
                                    window["Moat#ETS"] = c;
                                    try {
                                        g = "//v4.moatads.com/pixel.gif?e=24&d=data%3Adata%3Adata%3Adata&i=" + escape("NYTIMES2") + "&ac=1&k=" + escape(f) + "&ar=" + escape("4bb5684-clean") +
                                        "&j=" + escape(document.referrer) + "&cs=" + (new t).getTime(), k = new Image(1, 1), k.src = g
                                    } catch (aa) {}
                                }
                            }
                        }
                    }
                }()
            }
            a.e = {};
            var m = {};
            a.e.b = m;
            a.e.c = function(e, d) {
                if (!e || "string" !== typeof d ||!e[d] || e == window)
                    return !1;
                if ("string" === typeof e.nodeName && ("OBJECT" === e.nodeName || "EMBED" === e.nodeName)) {
                    var c = a && a.f && a.f[d];
                    if (c && c !== e[d])
                        return c
                }
                return !1
            };
            a.e.d = function(e, d, c, b) {
                var h, f = l(c), g;
                e.addEventListener ? (c = "addEventListener", h = "") : (c = "attachEvent", h = "on");
                if (g = a.e.c(e, c))
                    try {
                        g.call(e, h + d, f, !1)
                } catch (k) {
                    e[c](h + d,
                    f, !1)
                } else 
                    e[c](h + d, f, !1);
                !1 !== b && (m[d + b] = f)
            };
            a.e.e = function(e, d, c, b) {
                var h, f = d + b, g;
                if (!e)
                    return delete m[f], !1;
                c=!1 !== b ? m[f] : c;
                e.removeEventListener ? (b = "removeEventListener", h = "") : (b = "detachEvent", h = "on");
                if (g = a.e.c(e, b))
                    try {
                        g.call(e, h + d, c, !1)
                } catch (k) {
                    e[b](h + d, c, !1)
                } else 
                    e[b](h + d, c, !1);
                delete m[f]
            };
            a.e.f = function(a, d) {
                a = l(a);
                var c = setInterval(a, d);
                M.push(c);
                return c
            };
            a.e.g = function(a, d) {
                a = l(a);
                var c = setTimeout(a, d);
                F.push(c);
                return c
            };
            a.e.h = function(e, d, c, b) {
                if (!b)
                    return !1;
                b += "";
                w[b] && window.clearTimeout(w[b].tid);
                w[b] = {};
                w[b].callback = l(e);
                w[b].params = d;
                w[b].interval = c;
                w[b].tid = a.e.g(a.e.i(b), c);
                callbacks = null
            };
            a.e.i = function(e) {
                return function() {
                    if (!w ||!w[e])
                        return !1;
                    var d = w[e].callback(w[e].params);
                    if ("boolean" === typeof d&&!1 === d)
                        return window.clearTimeout(w[e].tid), w[e]=!1;
                    w[e].tid = a.e.g(a.e.i(e), w[e].interval);
                    time = e = null
                }
            };
            a.e.j = function() {
                return w
            };
            a.e.a = function(e, d, c, b) {
                var h = 0, f = function() {
                    h += 1;
                    !0 !== e() && (h < d ? a.e.g(f, c) : "function" === typeof b && b())
                };
                f()
            };
            a.e.k = l
        })(q);
        (function(a) {
            function l() {
                this.height =
                this.width = this.absTop = this.absLeft = 0;
                this.update = function(a) {
                    this.absLeft = a.left + b("left", a.win);
                    this.absTop = a.top + b("top", a.win);
                    this.width = a.width;
                    this.height = a.height
                }
            }
            function m(b, c) {
                var d = b.zr;
                f.hasOwnProperty(d) || (f[d] = new l);
                var e = c || new a.g.h(b.aa);
                f[d].update(e)
            }
            function e(a, b) {
                for (var c = [], d = 0; d < b.length; d++)
                    c.push(a(b[d]));
                return c
            }
            function d(b) {
                var c, d = a.c.b(b.el, b.win), d = e(function(b) {
                    return new a.g.h(b)
                }, d);
                d.unshift(b);
                var f = d.length;
                b = new a.g.h(b.el, a.b.c);
                for (var h = 0; h < f; h++) {
                    var s =
                    d[h];
                    0 === h ? c = s : (c.changeReferenceFrame(s), b.changeReferenceFrame(s));
                    s = s.getViewportRect(h < f - 1 ? d[h + 1] : !1);
                    c = a.g.j(c, s)
                }
                return {
                    visibleRect: c,
                    cumulRect: b,
                    parentArea: d[d.length - 1].calcArea()
                }
            }
            function c(a, b, c, d) {
                a = r.max(a, c);
                b = r.min(b, d);
                return b > a ? [a, b] : [0, 0]
            }
            function b(a, b) {
                b || (b = window);
                var c = b.document.documentElement, d = b.document.body;
                return "left" === a ? b.pageXOffset || c && c.scrollLeft || d && d.scrollLeft : b.pageYOffset || c && c.scrollTop || d && d.scrollTop
            }
            a.g = {};
            var h = a.b.a, f = {};
            a.g.a = .5;
            a.g.b = .3;
            a.g.c = 236425;
            a.g.d = function(b, c) {
                var e;
                e = new a.g.h(b.aa, c);
                m(b, e);
                var f = e.height, h = e.width, s = e.calcArea();
                if (0 === s)
                    e = {
                        area: s,
                        percv: 0
                    };
                else {
                    var K = d(e), D = K.visibleRect.calcArea(), G = D / s, l;
                    b:
                    {
                        var x = K.cumulRect, H = K.cumulRect.getViewportRect();
                        if (0 > x.top && 0 < x.bottom)
                            l =- x.top;
                        else if (0 <= x.top && x.top <= H.height)
                            l = 0;
                        else {
                            l = {
                                yMin: - 1,
                                yMax: - 1
                            };
                            break b
                        }
                        if (0 <= x.bottom && x.bottom <= H.height)
                            x = x.height;
                        else if (x.bottom > H.height && x.top < H.height)
                            x = x.height - (x.bottom - H.height);
                        else {
                            l = {
                                yMin: - 1,
                                yMax: - 1
                            };
                            break b
                        }
                        l = {
                            yMin: l,
                            yMax: x
                        }
                    }
                    e = {
                        area: s,
                        visibleArea: D,
                        percv: G,
                        yMinMax: l,
                        elGeo: {
                            elHeight: f,
                            elWidth: h,
                            foldTop: K.cumulRect.top,
                            totalArea: K.parentArea
                        },
                        rect: e.rect
                    }
                }
                f = a.g.e(e, b);
                e.isVisible = e.percv >= f;
                e.isFullyVisible = 1 == e.percv;
                e.elGeo && (e.elGeo.threshold = f);
                return e
            };
            a.g.e = function(b, c) {
                return b.area >= a.g.c ? (c.viewstats.isBigAd=!0, a.g.b) : a.g.a
            };
            a.g.f = function() {
                return h
            };
            a.g.g = function(a, b) {
                h && f.hasOwnProperty(b) || m(a);
                return f[b]
            };
            a.g.i = function(a) {
                var b, c, d, e = 0, f = 0;
                try {
                    b = a.document, c = b.documentElement, d = b.body
                } catch (h) {
                    return {
                        width: e,
                        height: f,
                        left: 0,
                        right: e,
                        top: 0,
                        bottom: f
                    }
                }
                "undefined" !== typeof a.innerWidth ? (e = a.innerWidth, f = a.innerHeight) : "CSS1Compat" === b.compatMode && 5 !== b.documentMode ||!d || "undefined" === typeof d.clientWidth ? c && "undefined" !== typeof c.clientWidth && (e = c.clientWidth, f = c.clientHeight) : (e = d.clientWidth, f = d.clientHeight);
                return {
                    width: e,
                    height: f,
                    left: 0,
                    right: e,
                    top: 0,
                    bottom: f
                }
            };
            a.g.h = function(b, c, d) {
                this.rect = d || b.getBoundingClientRect();
                d = ["left", "right", "top", "bottom"];
                for (var e = 0; e < d.length; e++) {
                    var f = d[e];
                    this[f] = this.rect[f]
                }
                this.width =
                this.right - this.left;
                this.height = this.bottom - this.top;
                this.el = b;
                this.win = c || b && a.c.a(b);
                this.changeReferenceFrame = function(a) {
                    this.left += a.left;
                    this.right += a.left;
                    this.top += a.top;
                    this.bottom += a.top
                };
                this.calcArea = function() {
                    return (this.right - this.left) * (this.bottom - this.top)
                };
                this.getViewportRect = function(b) {
                    var c = a.g.i(this.win);
                    b && (b.width < c.width && (c.width = b.width, c.right = c.left + c.width), b.height < c.height && (c.height = b.height, c.bottom = c.top + c.height));
                    return c
                }
            };
            a.g.j = function(b, d) {
                var e = c(b.left,
                b.right, d.left, d.right), f = c(b.top, b.bottom, d.top, d.bottom);
                return new a.g.h(void 0, void 0, {
                    left : e[0], right : e[1], top : f[0], bottom : f[1]
                })
            };
            a.g.k = b
        })(q);
        (function(a) {
            a.h = {};
            var l = a.a.g(), m = null !== l, e =- 1 !== navigator.userAgent.indexOf("Chrome"), d=!1, c = function() {
                var a = navigator.appVersion.match(/Windows NT (\d\.\d)/);
                return a ? parseFloat(a[1]) : - 1
            }(), b = 6.2 <= c;
            a.h.a = l;
            a.h.b = m;
            a.h.c = e;
            a.h.d = c;
            a.h.e = b;
            var h = {
                FRAME_RATE: "fr",
                STAGE_WIDTH: "sd",
                ACTIVE_STAGE_WIDTH: "asd",
                THROTTLE: "td",
                RAPID_THROTTLE: "rtd"
            };
            a.h.f =
            h;
            a.h.g = 0 < Object.prototype.toString.call(window.HTMLElement).indexOf("Constructor");
            a.h.h=!1;
            if (a.h.g)
                try {
                    var f = parseInt(navigator.userAgent.match(/Version\/(\d)/)[1], 10);
                    a.h.h = 5 < f
            } catch (g) {}
            a.h.i=!1;
            a.h.j = function(b) {
                if (b.currentFocusState) {
                    var c, d;
                    if ("center" != b.config.name && (c = b.manager.getPixelByName("center")) && (d = c.viewstates[b.manager.getTargetViewState()])&&!d.inview)
                        b.skipWidthCheck=!0;
                    else {
                        var e, f, h;
                        "undefined" !== typeof b.manager.adNum && (e = a.a.al(b.manager.adNum)) && (f = a.i.a(e, !1), h = a.i.b(e,
                        1));
                        f && h && "center" != b.config.name ? b.killWidthCheck=!0 : !f || "bottomLeft" != b.config.name && "topRight" != b.config.name ? b.skipWidthCheck=!1 : b.killWidthCheck=!0
                    }
                } else 
                    b.skipWidthCheck=!0
            };
            a.h.k = function(b) {
                return {
                    insertableFunc: a.h.l,
                    pixels: [{
                        name: "center",
                        id: "moatPx" + b.zr + "_" + r.ceil(1E6 * r.random()),
                        target: b.aa,
                        container: b.aa.parentNode,
                        position: {
                            top: "50%",
                            left: "50%"
                        },
                        onWidthCheck: a.h.j
                    }, {
                        name: "topLeft",
                        id: "moatPx" + b.zr + "_" + r.ceil(1E6 * r.random()),
                        target: b.aa,
                        container: b.aa.parentNode,
                        position: {
                            top: "0px",
                            left: "0px"
                        },
                        onWidthCheck: a.h.j
                    }, {
                        name: "bottomRight",
                        id: "moatPx" + b.zr + "_" + r.ceil(1E6 * r.random()),
                        target: b.aa,
                        container: b.aa.parentNode,
                        position: {
                            top: "100%",
                            left: "100%"
                        },
                        onWidthCheck: a.h.j
                    }
                    ]
                }
            };
            a.h.m = function(b, c) {
                var d=!1, e=!1;
                a.a.forEach(b.pixels, function(a) {
                    "0px" == a.config.position.left && "0px" == a.config.position.top && a.measurable && a.viewstates && a.viewstates[c] && (d=!0);
                    "100%" == a.config.position.left && "100%" == a.config.position.top && a.measurable && a.viewstates && a.viewstates[c] && (e=!0)
                });
                return d && e?!0 :
                !1
            };
            a.h.n = function(b, c) {
                var d=!1;
                a.a.forEach(b.pixels, function(a) {
                    if (a.config && "50%" == a.config.position.left && "50%" == a.config.position.top && a.viewstates && a.viewstates[c])
                        return d=!0, !1
                });
                return d
            };
            a.h.o = function(b, c) {
                var d=!1;
                a.a.forEach(b.pixels, function(a) {
                    if (a.config && a.viewstates && a.viewstates[c] && (d = a.viewstates[c].inview))
                        return !1
                });
                return d
            };
            a.h.p = function(b, c) {
                var d=!1;
                a.a.forEach(b.pixels, function(a) {
                    if (a.config && "50%" == a.config.position.left && "50%" == a.config.position.top && a.viewstates &&
                    a.viewstates[c])
                        return d = a.viewstates[c].inview, !1
                });
                return d
            };
            a.h.q = function(b, c) {
                if (!b.inview)
                    return !1;
                var d=!1, e=!1;
                a.a.forEach(b.pixels, function(a) {
                    "0px" == a.config.position.left && "0px" == a.config.position.top && a.measurable && a.viewstates && a.viewstates[c] && (d = a.viewstates[c].inview);
                    "100%" == a.config.position.left && "100%" == a.config.position.top && a.measurable && a.viewstates && a.viewstates[c] && (e = a.viewstates[c].inview)
                });
                return d && e?!0 : !1
            };
            a.h.r = function(a) {
                a.periscopeManager && a.periscopeManager.killAllPixels()
            };
            a.h.s = function(b) {
                a.h.t(b.periscopeConfig) || (b.periscopeConfig = a.h.k(b));
                b.periscopeManager = new a.h.u(b.periscopeConfig, b.zr);
                a.h.v = b.periscopeManager.insertSuccessful;
                return b.periscopeManager.insertSucceeded
            };
            a.h.v=!1;
            a.h.l = function() {
                return !a.a.q() && (m || e || a.h.h)
            };
            a.h.w = function(a) {
                return a.periscopeManager ? a.periscopeManager.getViewStats() : {
                    isVisible: !1
                }
            };
            a.h.x = function() {};
            a.h.t = function(b) {
                if ("object" !== typeof b || "function" !== typeof b.insertableFunc ||!a.a.am(b.pixels) || 0 == b.pixels.length)
                    return !1;
                var c=!1;
                a.a.forEach(b.pixels, function(a) {
                    a.id && a.target && a.container && "object" === typeof a.position && "string" === typeof a.position.top && "string" === typeof a.position.left || (c=!0)
                });
                return !c
            };
            a.h.y = function() {
                return b && m && 10 <= l
            };
            a.h.z = /([0-9]+(?:\.[0-9]+)?)(\%|px)/i;
            a.h.aa = function() {
                var b = {};
                return function(c) {
                    if ("string" !== typeof c)
                        return !1;
                    if ("undefined" !== typeof b[c])
                        return b[c];
                    var d, e;
                    (d = c.match(a.h.z)) && 3 == d.length && (e = d[2], d =- 1 != d[1].indexOf(".") ? parseInt(d[1], 10) : parseFloat(d[1], 10));
                    if ("number" !==
                    typeof d)
                        return !1;
                    b[c] = {
                        val: d,
                        type: e
                    };
                    return b[c]
                }
            }();
            a.h.ab = function(b, c) {
                this.config = b;
                this.measurable = this.inserted=!1;
                this.viewstates = {};
                this.manager = c;
                this.killed=!1;
                this.cbNames = [];
                this.killWidthCheck = this.skipWidthCheck=!1;
                this.loopIds = [];
                this.getPeriscopeAssetURI = function() {
                    return a.b.protocol + "//" + a.b.j + "/swf/p6.v3.swf"
                };
                this.insertIntoDOM = function() {
                    if (this.inserted)
                        return !1;
                    var b, c;
                    c = m ? b = 2 : b = 1;
                    var d = "position: absolute; width: " + b + "px; height: " + c + "px; z-index: -9999;";
                    a.h.h && (d += "-webkit-transform: translate3d(0, 0, 0);");
                    var e = this.config.id, f = this.getPeriscopeAssetURI(), h = this.calcPosition();
                    if (!h)
                        return !1;
                    var h = d + "left: " + h.left + "px; top: " + h.top + "px;", g = d + "left: 0px; top: 0px;";
                    this.targetDoc = d = this.config.target.ownerDocument;
                    var n = "defaultView"in d ? d.defaultView: d.parentWindow, s = "Moat#PSCB" + r.floor(1E8 * r.random());
                    n[s] = {
                        fn: this.onStateChange,
                        ct: this
                    };
                    this.cbNames.push(s);
                    n = "sco=" + encodeURIComponent(s) + "&tvs=" + this.manager.getTargetViewState();
                    s = d.createElement("div");
                    s.id = "moatPxDiv" + r.ceil(1E6 * r.random());
                    s.style.width =
                    "0px";
                    s.style.height = "0px";
                    s.style.position = "absolute";
                    s.style.top = "0px";
                    s.style.left = "0px";
                    this.wrapperDiv = s;
                    a.h.y() ? (g = function(a, b, c) {
                        var d = document.createElement("param");
                        d.setAttribute("name", b);
                        d.setAttribute("value", c);
                        a.appendChild(d)
                    }, d = d.createElement("OBJECT"), d.setAttribute("data", f), d.setAttribute("id", e), d.setAttribute("name", e), d.setAttribute("style", h), d.setAttribute("width", b + ""), d.setAttribute("height", c + ""), g(d, "flashvars", n), g(d, "wmode", "transparent"), g(d, "bgcolor", ""), g(d, "allowscriptaccess",
                    "always"), a.a.ah(s, this.config.container), s.appendChild(d)) : (a.a.ah(s, this.config.container), s.innerHTML = '<object type="application/x-shockwave-flash" classid="clsid:d27cdb6e-ae6d-11cf-96b8-444553540000" codebase="http://download.macromedia.com/pub/shockwave/cabs/flash/swflash.cab#version=8,0,0,0" width="' + b + '" height="' + c + '" style="' + h + '" id="' + e + '"><param name="allowScriptAccess" value="always" /><param name="allowFullScreen" value="false" /><param name="movie" value="' + f + '" /><param name="quality" value="low" /><param name="hasPriority" value="true" /><param name="FlashVars" value="' +
                    n + '" /><param name="bgcolor" value="" /><param name="wmode" value="transparent" /><embed type="application/x-shockwave-flash" src="' + f + '" quality="low" flashvars="' + n + '" bgcolor="" wmode="transparent" width="' + b + '" height="' + c + '" id="' + e + 'e" name="' + e + '" style="' + g + '" hasPriority="true" allowscriptaccess="always" allowFullScreen="false" type="application/x-shockwave-flash" pluginspage="http://www.adobe.com/go/getflashplayer" /></object>');
                    return this.inserted=!0
                };
                this.startIntervals = function() {
                    var b =
                    this.getPixelElement();
                    if (!b)
                        return !1;
                    if (8 == l && (this.manager.getTargetViewState() === h.STAGE_WIDTH || this.manager.getTargetViewState() === h.ACTIVE_STAGE_WIDTH)) {
                        var c = "positionToggle#" + this.config.id;
                        this.loopIds.push(c);
                        a.e.j()[c] || (this.positionTogglingEnabled=!0, this.positionOffsets || (this.positionOffsets = {}), a.e.h(this.positionToggle, {
                            pxSwf: b,
                            pxRef: this
                        }, 100, c))
                    }
                    this.manager.getTargetViewState() === h.STAGE_WIDTH && (c = "stageWidthLoop#" + this.config.id, this.loopIds.push(c), a.e.j()[c] || a.e.h(this.stageWidthToggle,
                    {
                        pxSwf: b,
                        pxRef: this,
                        originalWidth: b.style.width,
                        widthRe: /^[0-9\.]+/i,
                        updates: 0
                    }, 150, c));
                    this.manager.getTargetViewState() === h.ACTIVE_STAGE_WIDTH && (c = "activeStageWidthLoop#" + this.config.id, this.loopIds.push(c), a.e.j()[c] || a.e.h(this.stageWidthToggle, {
                        pxSwf: b,
                        pxRef: this,
                        originalWidth: b.style.width,
                        widthRe: /^[0-9\.]+/i,
                        updates: 0,
                        active: !0,
                        onWidthCheck: this.config.onWidthCheck
                    }, 200, c))
                };
                this.stageWidthToggle = function(a) {
                    if (!a.pxSwf ||!a.pxSwf.parentNode ||!a.pxRef)
                        return !1;
                    if (a.onWidthCheck) {
                        a.onWidthCheck(a.pxRef);
                        if (a.pxRef.killWidthCheck)
                            return a.pxRef.kill(), !1;
                        if (a.pxRef.skipWidthCheck)
                            return !0
                    }
                    var b;
                    if (a.parsedWidth || (b = a.pxSwf.style.width.match(a.widthRe)))
                        a.parsedWidth || (a.parsedWidth = parseInt(b[0], 10)), 1 == a.updates ? (a.updates = 0, a.pxSwf.style.width = a.originalWidth, b = a.parsedWidth) : (a.updates = 1, b = 1 < a.parsedWidth ? a.parsedWidth - 1 : a.parsedWidth + 1, a.pxSwf.style.width = b + "px"), a.active && a.pxSwf.currentPW && a.pxSwf.currentPW(b)
                };
                this.positionToggle = function(a) {
                    if (!a.pxRef ||!a.pxRef.element)
                        return !1;
                    0 > a.pxRef.positionOffsets.yOffset ?
                    (a.pxRef.positionOffsets.yOffset = 0, a.pxRef.positionOffsets.xOffset = 0) : (a.pxRef.positionOffsets.yOffset =- 2E3, a.pxRef.positionOffsets.xOffset =- 2E3);
                    a.pxRef.updatePosition(!0)
                };
                this.onStateChange = function(b) {
                    if (!this.measurable) {
                        this.measurable=!0;
                        var c;
                        b && b[0] && b[0].rev && (c = b[0].rev.match(a.h.ac)) && 3 == c.length && (a.h.ad = c[1], a.h.ae = c[2]);
                        this.updateFocusState();
                        this.startIntervals()
                    }
                    this.inserted && this.killed ? (this.killed=!1, this.updateFocusState(), this.startIntervals()) : (a.a.forEach(b, function(a) {
                        this.viewstates[a.name] =
                        a
                    }, this), this.manager.onStateChange(this, b))
                };
                this.calcPosition = function() {
                    var b = {}, c = this.config.position.left, d = a.h.aa(this.config.position.top), c = a.h.aa(c), e = 0, f = 0;
                    this.config.positionTarget ? this.config.positionTargetWindow ? this.targetRect = new a.g.h(this.config.positionTarget, this.config.positionTargetWindow) : (this.targetRect = new a.g.h(this.config.positionTarget), this.config.positionTargetWindow = this.targetRect.win) : (this.targetRect = this.targetRect ? new a.g.h(this.config.target, this.targetRect.win) :
                    new a.g.h(this.config.target), 0 == this.targetRect.left && 0 == this.targetRect.right && 0 == this.targetRect.top && 0 == this.targetRect.bottom && "EMBED" == this.targetRect.el.nodeName && null == this.targetRect.el.offsetParent && this.config.target.parentNode && (this.targetRect = new a.g.h(this.config.target.parentNode), this.config.positionTarget = this.config.target.parentNode));
                    var e = a.g.k("left", this.targetRect.win), f = a.g.k("top", this.targetRect.win), h;
                    this.wrapperDiv && (h = this.wrapperDiv.offsetParent) && "BODY" !== h.nodeName ?
                    this.offsetRect = this.offsetRect ? new a.g.h(h, this.offsetRect.win) : new a.g.h(h) : this.offsetRect = {
                        left: - e,
                        top: - f
                    };
                    if (!d ||!c)
                        return !1;
                    if ("%" == d.type)
                        b.relativeTop = d.val / 100 * this.targetRect.height, b.top = this.targetRect.top - this.offsetRect.top + d.val / 100 * this.targetRect.height;
                    else if ("px" == d.type)
                        b.relativeTop = d.val, b.top = this.targetRect.top - this.offsetRect.top + d.val;
                    else 
                        return !1;
                    if ("%" == c.type)
                        b.relativeLeft = c.val / 100 * this.targetRect.width, b.left = this.targetRect.left - this.offsetRect.left + c.val / 100 * this.targetRect.width;
                    else if ("px" == c.type)
                        b.relativeLeft = c.val, b.left = this.targetRect.left - this.offsetRect.left + c.val;
                    else 
                        return !1;
                    return b
                };
                this.maxPositionUpdateInterval = 200;
                this.updatePosition = function(a) {
                    var b = (new t).getTime();
                    if (!this.element ||!a && b - this.lastPositionUpdate < this.maxPositionUpdateInterval)
                        return !1;
                    this.lastPositionUpdate = b;
                    a = this.calcPosition();
                    if (!a)
                        return !1;
                    this.positionOffsets && (a.left += this.positionOffsets.xOffset || 0, a.top += this.positionOffsets.yOffset || 0);
                    this.element.style.left = this.width +
                    a.relativeLeft > this.targetRect.width ? r.floor(a.left - this.width) + "px" : 0 == a.relativeLeft ? r.floor(a.left) + "px" : r.floor(a.left - .5 * this.width) + "px";
                    this.element.style.top = this.height + a.relativeTop > this.targetRect.height ? r.floor(a.top - this.height) + "px" : 0 == a.relativeTop ? r.floor(a.top) + "px" : r.floor(a.top - .5 * this.height) + "px";
                    return !0
                };
                this.updateFocusState = function() {
                    var a = this.getPixelElement();
                    if (a && this.measurable)
                        try {
                            a.updateFocusState(this.currentFocusState)
                    } catch (b) {}
                };
                this.kill = function() {
                    var b = this.getPixelElement(),
                    c = a.c.a(b);
                    c && b && b.dataMoatTIDS && a.a.forEach(b.dataMoatTIDS, function(a) {
                        c.clearTimeout(a)
                    });
                    a.a.forEach(this.loopIds, function(a) {
                        w && w[a] && (window.clearTimeout(w[a].tid), w[a]=!1)
                    });
                    for (var b = 0, d = this.cbNames.length; b < d; b++) {
                        window[this.cbNames[b]] = null;
                        try {
                            delete window[this.cbNames[b]]
                        } catch (e) {}
                    }
                    return this.wrapperDiv && this.wrapperDiv.parentNode ? (this.wrapperDiv.parentNode.removeChild(this.wrapperDiv), this.killed=!0, this.inserted=!1, this.element = this.wrapperDiv = null, !0) : !1
                };
                this.getPixelElement =
                function() {
                    var a, b=!1, c = this.config.id;
                    if (this.targetDoc) {
                        a = this.targetDoc.getElementById(c);
                        if (b=!!(a && a.isPxSwf && a.isPxSwf()))
                            return a;
                        a = this.targetDoc.getElementById(c + "e");
                        if (b=!!(a && a.isPxSwf && a.isPxSwf()))
                            return a
                    }
                    return !1
                };
                if ("embed" === b.container.nodeName || "object" === b.container.nodeName) {
                    var d;
                    a.a.forEach(a.a.o(b.container), function(a) {
                        if ("embed" !== a.nodeName && "object" !== a.nodeName)
                            return d = a, !1
                    });
                    if (!d)
                        return !1;
                    this.config.container = d
                }
                if (!this.insertIntoDOM())
                    return !1;
                this.element = this.targetDoc.getElementById(this.config.id);
                if (!this.element)
                    return !1;
                var e = new a.g.h(this.element);
                this.width = e.width;
                this.height = e.height;
                if (!this.updatePosition())
                    return !1;
                this.currentFocusState = a.focus.pageIsVisible();
                this.focusCheckingLoop = function(b) {
                    if (!b.pxRef)
                        return !1;
                    b = b.pxRef;
                    b.currentFocusState != a.focus.pageIsVisible() && (b.currentFocusState=!b.currentFocusState, b.killed || b.updateFocusState())
                };
                this.rebuildOnFocusLoss = function() {
                    m && (this.currentFocusState || this.killed ||!this.inserted ? this.currentFocusState && this.killed&&!this.inserted &&
                    (this.insertIntoDOM.call(this), (this.element = this.targetDoc.getElementById(this.config.id)) && this.updatePosition()) : this.kill())
                };
                this.positionUpdateLoop = function(a) {
                    if (!a.pxRef)
                        return !1;
                    a.pxRef.killed || a.pxRef.updatePosition()
                };
                var e = "focusCheckingLoop#" + this.config.id, f = "positionUpdateLoop#" + this.config.id;
                this.loopIds.push(e);
                this.loopIds.push(f);
                a.e.h(this.focusCheckingLoop, {
                    pxRef: this
                }, 200, e);
                a.e.h(this.positionUpdateLoop, {
                    pxRef: this
                }, 500, f);
                this.inserted=!0;
                this.insertionTime = (new t).getTime()
            };
            a.h.ac = /([0-9a-z]+-[a-z]+)-(.*)/i;
            a.h.u = function(b, c) {
                this.insertedAllSuccessfully = this.insertSuccessful=!1;
                this.pixels = [];
                this.adNum = c;
                this.fullyMeasurable = this.measurable = this.reachedAnyInview = this.anyInview = this.anyMeasurable = this.reachedFullyInview = this.fullyInview = this.reachedInview = this.inview=!1;
                this.getPixelByName = function(b) {
                    var c;
                    a.a.forEach(this.pixels, function(a) {
                        if (a.config.name && a.config.name == b)
                            return c = a, !1
                    });
                    return c ||!1
                };
                this.getTargetViewState = function() {
                    var b = h.FRAME_RATE;
                    m && (b =
                    h.ACTIVE_STAGE_WIDTH);
                    a.h.h && (b = h.ACTIVE_STAGE_WIDTH);
                    return b
                };
                this.onStateChange = function(b, c) {
                    var e = this.getTargetViewState(), f = a.focus.pageIsVisible(), h = "undefined" != typeof u && u[this.adNum];
                    this.anyMeasurable || (this.anyMeasurable=!0);
                    this.fullyMeasurable || (this.fullyMeasurable = a.h.m(this, e));
                    this.measurable || (this.measurable = a.h.n(this, e), a.h.af = (new t).getTime());
                    if (1 == c.length) {
                        if (c[0].name != e)
                            return !1
                    } else {
                        var g=!0;
                        a.a.forEach(c, function(a) {
                            if (a.name == e)
                                return g=!1
                        });
                        if (g ||!f)
                            return !1
                    }
                    this.measurable &&
                    ((this.anyInview = a.h.o(this, e))&&!this.reachedAnyInview && (this.reachedAnyInview=!0), (this.inview = a.h.p(this, e))&&!this.reachedInview && (this.reachedInview=!0), !d && a.h.h && h && (d=!0, a.c.c(h)));
                    this.fullyMeasurable && (this.fullyInview = a.h.q(this, e))&&!this.reachedFullyInview && (this.reachedFullyInview=!0);
                    h && a.i.c(h)
                };
                this.getViewStats = function() {
                    return {
                        isVisible: this.inview,
                        isFullyVisible: this.fullyInview
                    }
                };
                this.killAllPixels = function() {
                    a.a.forEach(this.pixels, function(a) {
                        a.kill()
                    })
                };
                if (b.insertableFunc()) {
                    var e =
                    0;
                    a.a.forEach(b.pixels, function(b, c) {
                        this.pixels.push(new a.h.ab(b, this));
                        this.pixels[c].inserted && (e++, this.insertSuccessful=!0)
                    }, this);
                    this.insertedAllSuccessfully = e === this.pixels.length
                }
            }
        })(q);
        (function(a) {
            function l(a, b, c) {
                a.IR5.MIN[c] = r.min(b, a.IR5.MIN[c]) || b || 1;
                a.IR5.MAX[c] = r.max(b, a.IR5.MAX[c]) || b
            }
            function m(a, b) {
                b.be = r.max("undefined" !== typeof b.be ? b.be : 0, a - b.bf);
                "undefined" === typeof b.by && 500 <= b.be && (b.by = b.bk)
            }
            function e(b) {
                b.az === a.j.a.d.a ? b.az = a.j.a.d.b : b.az === a.j.a.d.b && (b.az = a.j.a.d.a)
            }
            function d(b) {
                b.ba === a.j.a.d.a ? b.ba = a.j.a.d.b : b.ba === a.j.a.d.b && (b.ba = a.j.a.d.a)
            }
            function c(b) {
                b.ax === a.j.a.b.a ? b.ax = a.j.a.b.b : b.ax === a.j.a.b.b && (b.ax = a.j.a.b.a)
            }
            function b(b) {
                b.ay === a.j.a.c.a ? b.ay = a.j.a.c.b : b.ay === a.j.a.c.b && (b.ay = a.j.a.c.a)
            }
            function h(a, b) {
                "undefined" === typeof b.bk && (b.bk = 0);
                b.bk += a - b.bo;
                b.bo = a
            }
            function f(a, b) {
                "undefined" === typeof b.bl && (b.bl = 0);
                b.bl += a - b.bp;
                b.bp = a
            }
            function g(a, b) {
                "undefined" === typeof b.bg && (b.bg = 0);
                "undefined" === typeof b.bc && (b.bc = 0);
                b.bu = a - b.bs;
                b.bu > b.bc && (b.bc =
                b.bu);
                b.bg += a - b.bq;
                500 <= b.bc && "undefined" === typeof b.bw && (b.bk += a - b.bo, b.bw = b.bk);
                b.bq = a
            }
            function k(a, b) {
                "undefined" === typeof b.bh && (b.bh = 0);
                "undefined" === typeof b.bd && (b.bd = 0);
                b.bv = a - b.bt;
                b.bv > b.bd && (b.bd = b.bv);
                b.bh += a - b.br;
                500 <= b.bd && "undefined" === typeof b.bx && (b.bl += a - b.bp, b.bx = b.bl);
                b.br = a
            }
            a.j = {};
            a.j.a = {};
            a.j.a.a = {};
            a.j.a.a.a = 0;
            a.j.a.a.b = 1;
            a.j.a.b = {};
            a.j.a.b.a = 0;
            a.j.a.b.b = 1;
            a.j.a.c = {};
            a.j.a.c.a = 0;
            a.j.a.c.b = 1;
            a.j.a.d = {};
            a.j.a.d.a = 0;
            a.j.a.d.b = 1;
            a.j.a.e = {};
            a.j.a.e.a = 0;
            a.j.a.e.b = 1;
            a.j.a.f = {};
            a.j.a.f.a =
            0;
            a.j.a.f.b = 1;
            a.j.a.f.c = 2;
            a.j.b = function(a) {
                l(a, a.aj, "x");
                l(a, a.ak, "y");
                a.IR5.AREA = (a.IR5.MAX.x - a.IR5.MIN.x) * (a.IR5.MAX.y - a.IR5.MIN.y)
            };
            a.j.c = function(b) {
                function c() {
                    500 <= (new t).getTime() - U && (a.j.d({
                        type: "park"
                    }, b), clearInterval(e), b.au = a.j.a.a.a)
                }
                var d = b.au;
                if (d === a.j.a.a.a) {
                    U = (new t).getTime();
                    var e = a.e.f(c, 50);
                    b.au = a.j.a.a.b
                } else 
                    d === a.j.a.a.b && (U = (new t).getTime())
            };
            a.j.e = function(b) {
                function c() {
                    3E3 <= (new t).getTime() - V && (a.j.f({
                        type: "park"
                    }, b), clearInterval(e), b.av = a.j.a.a.a)
                }
                var d = b.av;
                if (d ===
                a.j.a.a.a) {
                    V = (new t).getTime();
                    var e = a.e.f(c, 50);
                    b.av = a.j.a.a.b
                } else 
                    d === a.j.a.a.b && (V = (new t).getTime())
            };
            a.j.g = function(b, c) {
                var d = b.type;
                if (c.az === a.j.a.d.a) {
                    if ("mouseover" === d || "mousemove" === d)
                        c.bo = (new t).getTime(), e(c)
                } else if (c.az === a.j.a.d.b) {
                    "mousemove" === d && h((new t).getTime(), c);
                    if ("mouseout" === d || "blur" === d)
                        h((new t).getTime(), c), e(c);
                    "scooper" === d && h((new t).getTime(), c)
                }
            };
            a.j.h = function(b, c) {
                var e = b.type;
                if (c.ba === a.j.a.d.a) {
                    if ("mouseover" === e || "mousemove" === e)
                        c.bp = (new t).getTime(),
                        d(c)
                } else if (c.ba === a.j.a.d.b) {
                    "mousemove" === e && f((new t).getTime(), c);
                    if ("mouseout" === e || "blur" === e)
                        f((new t).getTime(), c), d(c);
                    "scooper" === e && f((new t).getTime(), c)
                }
            };
            a.j.d = function(b, d) {
                if (2 != d.an) {
                    var e = b.type;
                    if (d.ax === a.j.a.b.a) {
                        if ("mouseover" === e || "mousemove" === e)
                            d.bq = (new t).getTime(), d.bs = (new t).getTime(), c(d)
                        } else 
                            d.ax === a.j.a.b.b && ("mousemove" !== e && "mouseout" !== e || g((new t).getTime(), d), "park" === e && g((new t).getTime() - 500, d), "mouseout" !== e && "park" !== e || c(d))
                }
            };
            a.j.f = function(c, d) {
                if (2 !=
                d.an) {
                    var e = c.type;
                    if (d.ay === a.j.a.c.a) {
                        if ("mouseover" === e || "mousemove" === e)
                            d.br = (new t).getTime(), d.bt = (new t).getTime(), b(d)
                        } else 
                            d.ay === a.j.a.c.b && ("mousemove" !== e && "mouseout" !== e || k((new t).getTime(), d), "park" === e && k((new t).getTime() - 3E3, d), "mouseout" !== e && "park" !== e || b(d))
                }
            };
            a.j.i = function(b, c) {
                var d = b.type;
                if (c.bb == a.j.a.e.a) {
                    if ("mouseover" == d || "mousemove" == d)
                        c.bf = (new t).getTime(), c.bb = a.j.a.e.b
                } else 
                    c.bb == a.j.a.e.b && ("mouseout" == d ? (m((new t).getTime(), c), c.bb = a.j.a.e.a) : "mousemove" != d && "scooper" !=
                    d || m((new t).getTime(), c))
            }
        })(q);
        (function(a) {
            function l(b) {
                if (!a.focus.pageIsPrerendered()) {
                    a.e.e(document, g, l, "pr");
                    for (var c in u)
                        u.hasOwnProperty(c) && a.c.d(u[c])
                }
            }
            function m(a) {
                "undefined" == typeof b && (b = a)
            }
            a.focus = {};
            var e = navigator.userAgent, e =- 1 < e.indexOf("Safari")&&-1 == e.indexOf("Chrome")&&-1 == e.indexOf("Chromium")&&!B, d = (eval("/*@cc_on!@*/false") || document.documentMode)&&!B, c = "undefined" !== typeof document.hasFocus, b, h = {
                visible: 0,
                hidden: 1,
                prerender: 2
            }, f, g, k, n;
            "undefined" !== typeof document.hidden ?
            (f = "hidden", g = "visibilitychange") : "undefined" !== typeof document.mozHidden ? (f = "mozHidden", g = "mozvisibilitychange") : "undefined" !== typeof document.msHidden ? (f = "msHidden", g = "msvisibilitychange") : "undefined" !== typeof document.webkitHidden && (f = "webkitHidden", g = "webkitvisibilitychange");
            for (var r = ["v", "mozV", "msV", "webkitV"], q = 0; q < r.length; q++) {
                var s = r[q] + "isibilityState";
                if ("undefined" !== typeof document[s]) {
                    k = s;
                    break
                }
            }
            n = "undefined" !== typeof f;
            var K, D;
            "undefined" !== typeof window.requestAnimationFrame ? K =
            "requestAnimationFrame" : "undefined" !== typeof window.webkitRequestAnimationFrame && (K = "webkitRequestAnimationFrame");
            D = e && "string" == typeof K&&!n;
            var G = (new t).getTime();
            D && function H() {
                G = (new t).getTime();
                window[K](H)
            }();
            a.focus.getFocusMethod = function() {
                return b
            };
            a.focus.visibilitychange = g;
            a.focus.setFocusListeners = function() {};
            a.focus.getQueryString = function(a) {
                a = {};
                a.em = b;
                P && (a.eo = 1);
                "undefined" != typeof k && (a.en = h[document[k]]);
                return a
            };
            a.focus.supportsPageVisAPI = function() {
                return n
            };
            a.focus.pageIsVisible =
            function() {
                if (a.focus.supportsPageVisAPI())
                    return m(0), !document[f];
                if (D)
                    return m(1), 100 > (new t).getTime() - G;
                if (d && c)
                    return m(2), document.hasFocus();
                m(3);
                return !0
            };
            a.focus.pageIsPrerendered = function() {
                return "undefined" !== typeof k ? "prerender" == document[k] : !1
            };
            a.focus.pageIsVisible();
            a.focus.pageIsPrerendered() && a.e.d(document, g, l, "pr");
            var P = a.focus.pageIsPrerendered()
        })(q);
        (function(a) {
            function l(b, c, d, e, f) {
                ("remove" === f ? a.e.e : a.e.d)(b, c, function(c) {
                    c = c || this.event;
                    var e = c.currentTarget || b, f = u[e[R]];
                    if (f) {
                        var h;
                        h = c;
                        var g = e.getBoundingClientRect();
                        h =- 1 != h.type.indexOf("touch") ? {
                            x: parseInt(h.changedTouches[0].clientX - g.left, 10),
                            y: parseInt(h.changedTouches[0].clientY - g.top, 10)
                        } : {
                            x: parseInt(h.clientX - g.left, 10),
                            y: parseInt(h.clientY - g.top, 10)
                        };
                        f.aj = h.x;
                        f.ak = h.y;
                        f.dm || (f.cb = 2 == a.focus.getFocusMethod() ? f.counters.laxDwell.tCur : f.counters.strictDwell.tCur, f.dm = 1);
                        d.call(b, c, e, f)
                    }
                }, e)
            }
            function m(a, b, f) {
                l(a, "click", n, b, f);
                l(a, "mousedown", d, b, f);
                B ? l(a, "touchstart", c, b, f) : (l(a, "mousemove", e, b, f), l(a, "mouseover",
                g, b, f), l(a, "mouseout", k, b, f))
            }
            function e(b, c, d) {
                B || d.aj === d.al && d.ak === d.am || (a.j.d(b, d), a.j.f(b, d), a.j.g(b, d), a.j.i(b, d), a.j.h(b, d), a.j.b(d), a.j.c(d), a.j.e(d), 0 === d.ar.length && (d.ai = O(d)), d.ar.push(d.aj), d.as.push(d.ak), d.at.push(a.a.k(d)), d.al = d.aj, d.am = d.ak);
                d.ai !== O(d) && 1 < d.ar.length && q(d, "onMouseMove")
            }
            function d(b, c, d) {
                b = {
                    e: 2
                };
                b.q = d.aq[2]++;
                b.x = d.aj;
                b.y = d.ak;
                a.l.a(d, b)
            }
            function c(b, c, d) {
                b = {
                    e: 23
                };
                b.q = d.aq[23]++;
                b.x = d.aj;
                b.y = d.ak;
                c = (new t).getTime();
                if ("undefined" === typeof d.ct)
                    d.ct = c;
                else {
                    var e =
                    c - d.ct;
                    d.ct = c;
                    d.cu = r.min(d.cu, e) || e
                }
                b.bz = void 0;
                a.l.a(d, b)
            }
            function b(b, c, d) {
                if (2 == b.an) {
                    var e = c.e, g = b.ck;
                    g == a.j.a.f.a && 6 === e ? (h(b, 0), b.cl = a.a.k(b), b.ck = a.j.a.f.b) : g == a.j.a.f.b ? 22 === e ? (f(b, c), h(b, d), b.cl = a.a.k(b)) : 7 === e && (1E3 < a.a.k(b) - b.cl ? (clearTimeout(b.cm), c.e = 22, f(b, c), b.cn = 0, b.ck = a.j.a.f.a) : b.ck = a.j.a.f.c) : g == a.j.a.f.c && (6 == e ? (1E3 < a.a.k(b) - b.cl && (clearTimeout(b.cm), b.cn = 0, b.cl = a.a.k(b), h(b, 0)), b.ck = a.j.a.f.b) : 22 == e && (f(b, c), b.ck = a.j.a.f.a, b.cn = 0))
                }
            }
            function h(c, d) {
                var e = 5 > c.cn ? 1E3: 2 * d, f = {
                    e: 22
                };
                c.cm = a.e.g(function() {
                    b(c, f, e)
                }, e)
            }
            function f(b, c) {
                c.q = b.aq[c.e]++;
                c.m = a.a.k(b);
                b.cl = c.m;
                a.l.a(b, c);
                b.cn++
            }
            function g(c, d, e) {
                a.j.d(c, e);
                a.j.f(c, e);
                a.j.g(c, e);
                a.j.i(c, e);
                a.j.h(c, e);
                b(e, {
                    e: 6
                }, 0)
            }
            function k(c, d, e) {
                a.j.d(c, e);
                a.j.f(c, e);
                a.j.g(c, e);
                a.j.i(c, e);
                a.j.h(c, e);
                b(e, {
                    e: 7
                }, 0)
            }
            function n(b, c, d) {
                b = {
                    e: 3
                };
                b.q = d.aq[3]++;
                b.x = d.aj;
                b.y = d.ak;
                a.l.a(d, b)
            }
            function q(b, c) {
                b.ai = O(b);
                var d = {
                    e: 1
                };
                d.q = b.aq[1]++;
                d.x = b.ar.join("a");
                d.y = b.as.join("a");
                for (var e = a.a.k(b), f = b.at, h = [], g = 0; g < f.length; g++)
                    isNaN(f[g]) ||
                    h.push(r.abs(f[g] - e));
                d.c = h.join("a");
                d.m = e;
                a.l.a(b, d);
                b.ar = [];
                b.as = [];
                b.at = []
            }
            function O(b) {
                return r.floor(a.a.k(b) / 1E3)
            }
            a.k = {};
            a.k.a = function(a, b) {
                if (!0 === a.isSkin)
                    for (var c = 0; c < a.elements.length; c++)
                        m(a.elements[c], a.zr + "-" + c, b);
                else 
                    m(a.aa, a.zr, b)
            };
            a.k.b = function(b) {
                for (var c in u)
                    u.hasOwnProperty(c) && (b = u[c]) && (a.j.g({
                        type: "scooper"
                    }, b), a.j.i({
                        type: "scooper"
                    }, b), a.j.h({
                        type: "scooper"
                    }, b), 1 < b.ar.length && b.ai !== O(b) && q(b, "scooper"))
            }
        })(q);
        (function(a) {
            a.m = {};
            var l = "undefined" !== typeof window.devicePixelRatio,
            m = l && r.round(1E3 * window.devicePixelRatio), e = a.b.a && l && "undefined" !== typeof window.innerHeight && "undefined" !== typeof window.outerHeight && r.round(window.devicePixelRatio * (a.b.c.outerHeight - a.b.c.innerHeight)), d = "undefined" !== typeof window.mozInnerScreenX && "undefined" !== typeof window.screenX;
            a.m.a = function(c, b) {
                var d, e, g = {
                    isVisible: !1,
                    isFullyVisible: !1
                };
                try {
                    var k = c.aa.getBoundingClientRect(), n = b || c.WINDOW || a.c.a(c.aa), m = a.g.i(n), q = a.g.j(k, m), s = n.mozInnerScreenX, r = n.mozInnerScreenY, D = window.screenX, G =
                    window.screenY, t = a.g.j({
                        left: q.left + s,
                        right: q.right + s,
                        top: q.top + r,
                        bottom: q.bottom + r
                    }, {
                        left: D,
                        right: D + window.outerWidth,
                        top: G + 117 / (l ? window.devicePixelRatio : 1),
                        bottom: G + window.outerHeight
                    }), x = k.width * k.height;
                    d = {
                        area: x,
                        percv: (t.right - t.left) * (t.bottom - t.top) / x
                    };
                    e = a.g.e(d, c)
                } catch (H) {}
                "undefined" !== typeof d && "undefined" !== typeof e && (g.isVisible = d.percv >= e, g.isFullyVisible = 1 == d.percv);
                return g
            };
            a.m.b = function() {
                return d
            };
            a.m.c = function() {
                var a = {};
                a.dl = Number(d);
                "number" !== typeof m || isNaN(m) || (a.dm = m);
                "number" !== typeof e || isNaN(e) || (a.dn = e);
                return a
            }
        })(q);
        (function(a) {
            function l(a) {
                var b = 0, d;
                return function() {
                    var e = (new t).getTime();
                    150 < e - b && (d = a.apply(this, arguments), b = e);
                    return d
                }
            }
            function m(c, b, d, f) {
                var g = [], k = 0, n = 0, l = 0, m = 0, s = 0, q = 0, D = (new t).getTime(), G=!1, P=!1, x=!1, H=!1, X, u, v, w = 0, A = 0, y = 0, B = 0, M=!1, Q = a.b.a, J;
                if (0 === f)
                    var E = "as", z = "ag", F = "an", C = "ck", L = "kw", N = "ah", W = "aj", U = "pg", V = "pf", R = "gi", S = "gf", T = "gg", M=!0;
                else 
                    1 === f ? (E = "cc", z = "bw", F = "bx", C = "ci", L = "jz", N = "bu", W = "dj") : 2 === f ? (E = "cg", z = "ce", F =
                    "cf", C = "cj", L = "ts", N = "ah", W = "dk", R = "gj", S = "gb", T = "ge") : 3 === f && (E = "cg", z = "ce", F = "cf", C = "cj", L = "ts", N = "ah", W = "dk", R = "gi", S = "gf", T = "gg");
                this.addListener = function(a) {
                    g.push(a)
                };
                this.hadOTS = function() {
                    return x
                };
                this.hadVideo2SecOTS = function() {
                    return _hadVideo2SecOts
                };
                this.getInViewTime = function() {
                    return k
                };
                this.visible = function() {
                    return G
                };
                this.fullyVisible = function() {
                    return P
                };
                this.getFullInviewTimeTotal = function() {
                    return n
                };
                this.update = function(l) {
                    var m = k || 0, aa = n || 0, O=!1, I = c(l);
                    l.elementRect = I.rect;
                    var B = I.isVisible, E = I.isFullyVisible, z = b(l), F = (new t).getTime(), C = r.max(r.min(F - D, 1E3), 0);
                    D = F;
                    F=!d || a.focus.pageIsVisible(2 === f || 3 === f);
                    B = B && F&&!z;
                    E = E && F&&!z;
                    if (B && G)
                        k += C, s += C;
                    else if (B || G)
                        z = r.round(C / 2), k += z, s += z;
                    if (E && P)
                        n += C, q += C;
                    else if (E || P)
                        z = r.round(C / 2), n += z, q += z;
                    !x && 1E3 <= s && (O = x=!0, u = k, this.timeToView = X = l.counters.query()[N]);
                    !H && 1E3 <= q && (O = H=!0);
                    "undefined" === typeof v && (1E3 >= l.counters.query().bu ? B && (v=!0) : v=!1);
                    if ((l.FOLDMEASURABLE = Q) && "undefined" === typeof J && 2 !== f && 3 !== f && I.elGeo) {
                        var z =
                        e().y + I.elGeo.foldTop, L = I.elGeo.threshold * I.elGeo.elHeight;
                        _isBTFCheck = z > a.i.d() - L;
                        0 < I.elGeo.totalArea && (J = _isBTFCheck, l.BELOWTHEFOLDAD = J)
                    }
                    l = I.yMinMax;
                    M && F && "undefined" !== typeof l && I.elGeo && 0 <= l.yMax && 0 <= l.yMin && 0 < I.visibleArea && (w = r.max(l.yMax, w), A = r.min(l.yMin, A), y = r.min(1, r.max((w - A) / I.elGeo.elHeight, y)));
                    B || (s = 0);
                    E || (q = 0);
                    G = B;
                    P = E;
                    a.a.forEach(g, function(a) {
                        if (a.onInViewTimeCount)
                            a.onInViewTimeCount(C, k - m);
                        if (a.onFullyInViewTimeCount)
                            a.onFullyInViewTimeCount(C, n - aa)
                    });
                    return O
                };
                this.getQS = function(a) {
                    l >
                    k && (l = k);
                    m > n && (m = n);
                    a[E] = Number(x);
                    a[z] = k;
                    a[F] = l;
                    if (0 === f || 2 === f || 3 === f)
                        H && (a[R] = 1), a[S] = n, a[T] = m;
                    "undefined" !== typeof u && (a[C] = u);
                    "undefined" !== typeof X && (a[L] = X);
                    "undefined" !== typeof v && (a[W] = Number(v));
                    if (!0 === M) {
                        var b = r.round(100 * y), c = r.round(100 * B);
                        a[U] = b;
                        a[V] = c;
                        B = y
                    }
                    "undefined" !== typeof J && (a.ib = Number(J));
                    l = k;
                    m = n
                }
            }
            function e() {
                var c = a.b.c, b = c.document;
                return {
                    y: void 0 !== c.pageYOffset ? c.pageYOffset: (b.documentElement || b.body.parentNode || b.body).scrollTop
                }
            }
            a.i = {};
            var d = {};
            a.i.d = function() {
                return B ?
                a.g.i(a.b.c).height : 750
            };
            a.i.e = function(c) {
                var b = c.zr;
                d[b] = {};
                c.viewstats = {
                    isBigAd: !1
                };
                if (a.g.f()) {
                    var e = l(a.g.d), f;
                    f = new m(e, a.c.e, !0, 0);
                    d[b].strict = f;
                    e = new m(e, a.c.e, !1, 1);
                    d[b].lax = e
                }
                !0 !== c.isSkin && a.m && a.m.b() ? (c = new m(a.m.a, a.c.e, !0, 3), d[b].pscope = c) : a.h && a.h.l() && (c = new m(a.h.w, a.c.e, !0, 2), d[b].pscope = c)
            };
            a.i.c = function(c) {
                var b = d[c.zr], e, f;
                for (f in b)
                    b.hasOwnProperty(f) && b[f].update(c) && (e=!0);
                e && a.c.f(c);
                a.n.a(c) && a.c.c(c)
            };
            a.i.f = function(c, b, d) {
                "unefined" == typeof d && (d=!1);
                var e = a.i && a.i.g(c.zr);
                c = (a.m && a.m.b() || a.h && a.h.v) && e && e.pscope.getInViewTime() >= b;
                b = d ? a.g && a.g.f() && e && e.strict && e.strict.getInViewTime() >= b : a.g && a.g.f() && e && e.lax && e.lax.getInViewTime() >= b;
                return a.b.a ? b : c
            };
            a.i.h = function(c, b, d) {
                c = a.i.g(c.zr);
                d = d ? "hadVideo2SecOTS" : "hadOTS";
                var e = (a.m && a.m.b() || a.h && a.h.v) && c && c.pscope[d]();
                if ("strict" === b)
                    var g = a.g && a.g.f() && c && c.strict && c.strict[d]();
                else 
                    "lax" === b && (g = a.g && a.g.f() && c && c.lax && c.lax[d]());
                return a.b.a ? g : e
            };
            a.i.a = function(c, b) {
                var d = a.i.g(c.zr);
                return a.h && a.h.v && d && d.pscope[b ?
                "hadVideo2SecOTS": "hadOTS"]()
            };
            a.i.b = function(c, b) {
                var d = a.i && a.i.g(c.zr);
                return a.h && a.h.v && d && d.pscope.getFullInviewTimeTotal() >= b
            };
            a.i.i = function(c, b, d) {
                "undefined" == typeof d && (d=!1);
                var e = a.i && a.i.g(c.zr);
                c = (a.m && a.m.b() || a.h && a.h.v) && e && e.pscope.getFullInviewTimeTotal() >= b;
                b = d ? a.g && a.g.f() && e && e.strict && e.strict.getFullInviewTimeTotal() >= b : a.g && a.g.f() && e && e.lax && e.lax.getFullInviewTimeTotal() >= b;
                return a.b.a ? b : c
            };
            a.i.j = function(a) {
                delete d[a]
            };
            a.i.g = function(a) {
                return d[a]
            };
            a.i.k = function(a) {
                var b =
                null;
                (a = d[a]) && a.strict ? b = a.strict : a && a.pscope && (b = a.pscope);
                return b
            };
            a.i.l = function(a) {
                var b = {}, e = d[a], f;
                for (f in e)
                    e.hasOwnProperty(f) && e[f].getQS(b);
                (a = u[a]) && a.viewstats.isBigAd && (b.el = 1);
                return b
            }
        })(q);
        (function(a) {
            a.o = {};
            a.o.a = function(a, m) {
                var e;
                m.outerHTML ? e = m.outerHTML : (e = document.createElement("div"), e.appendChild(m.cloneNode(!0)), e = e.innerHTML);
                e = [/flashvars\s*=\s*(".*?"|'.*?')/i.exec(e), /name\s*=\s*["']flashvars["']\s*value\s*=\s*(".*?"|'.*?')/i.exec(e), /value\s*=\s*(".*?"|'.*?')\s*name\s*=\s*["']flashvars["']/i.exec(e),
                a];
                for (var d, c, b = {}, h = 0; h < e.length; h++) {
                    if ((d = e[h]) && "object" === typeof d && d[1])
                        d = d[1].replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/^"/g, "").replace(/"$/g, "").replace(/^'/g, "").replace(/'$/g, "");
                    else if ("string" === typeof d)
                        d = d.split("?")[1];
                    else 
                        continue;
                    if (d) {
                        d = d.split("&");
                        for (var f = 0; f < d.length; f++)
                            c = d[f].split("="), 2 > c.length || "function" == c[0].slice(0, 8) || (b[c[0]] = [c[1]])
                        }
                }
                return b
            }
        })(q);
        (function(a) {
            function l(d) {
                var c = {
                    window: 0,
                    transparent: 1,
                    opaque: 2,
                    direct: 3,
                    gpu: 4
                };
                if ("EMBED" === d.tagName)
                    var b =
                    a.a.getAttribute(d, "wmode");
                else if ("OBJECT" === d.tagName) {
                    d = d.getElementsByTagName("param");
                    for (var e = 0; e < d.length; e++) {
                        var f = d[e], g = a.a.getAttribute(f, "name"), f = a.a.getAttribute(f, "value");
                        if ("wmode" === g) {
                            b = f;
                            break
                        }
                    }
                }
                return b && c[b.toLowerCase()] || 5
            }
            function m(d) {
                var c = d.el, b = d.url, h = d.flashVars, f = d.adIds, g = d.opt_props;
                (new t).getTime();
                this.ao = f;
                this.FIND_AD_TRIES = f.numTries || 0;
                var k;
                try {
                    if (c) {
                        var n = c, m;
                        "DIV" === n.tagName && ((n = c.getElementsByTagName("EMBED")[0]) || (n = c.getElementsByTagName("OBJECT")[0]),
                        n || (n = c.getElementsByTagName("IMG")[0]), n || (n = c));
                        1 === n.nodeType && "IMG" !== n.nodeName && "EMBED" !== n.nodeName && "OBJECT" !== n.nodeName && (n = c.getElementsByTagName("EMBED")[0] || c.getElementsByTagName("OBJECT")[0] || c.getElementsByTagName("IMG")[0] || c);
                        if ("OBJECT" === n.tagName) {
                            for (var q = 0; q < n.children.length; q++)
                                if ("movie" === n.children[q].name || "Movie" === n.children[q].name)
                                    m = n.children[q].value;
                            n.object && n.object.Movie ? m = n.object.Movie : n.data&&-1 !== n.data.indexOf("swf") && (m = n.data)
                        }
                        "EMBED" !== n.tagName && "IMG" !==
                        n.tagName ||!n.src || (m = n.src);
                        var s = a.o.a(m, n);
                        k = {
                            adURL: m,
                            flashVars: s
                        }
                    } else 
                        k=!1
                } catch (r) {
                    k=!1
                }
                if (k && k.adURL && h)
                    for (p in k.flashVars)
                        k.flashVars.hasOwnProperty(p) && (h[p] = k.flashVars[p]);
                k && k.flashVars && (h = k.flashVars);
                if ("string" !== typeof b || "DIV" === b)
                    b = k && k.adURL || "-";
                b && 0 !== b.toLowerCase().indexOf("http:") && 0 !== b.toLowerCase().indexOf("https:") && ("/" === b[0] ? b = window.location.protocol + "//" + window.location.host + b : (k = window.location.pathname.split("/").slice(0, - 1).join("/"), b = window.location.protocol +
                "//" + window.location.host + "/" + k + (k ? "/" : "") + b));
                "IFRAME" !== c.tagName && "IMG" !== c.tagName&&-1 === b.indexOf("googlesyndication") && (b = b.split("?")[0]);
                this.zr = f.adNum;
                u[this.zr] = this;
                a.n.d(this.zr, [c]);
                this.ae = b;
                this.aa = c;
                this.WINDOW = a.c.a(this.aa);
                this.INITIAL_WIDTH = c.offsetWidth;
                this.INITIAL_HEIGHT = c.offsetHeight;
                "undefined" === typeof h && (h = {});
                v.i[C]=!0;
                this.eg = [];
                this.ee = {};
                g && g.IS_PAGE_LEVEL || (this.ed = {}, a.h.s(this));
                a.c.g(this);
                this.ag = h;
                this.ah = void 0;
                this.ai = 0;
                this.an = this.am = this.al = this.ak = this.aj =
                void 0;
                this.ar = [];
                this.as = [];
                this.at = [];
                this.av = this.au = a.j.a.a.a;
                this.ax = a.j.a.b.a;
                this.ay = a.j.a.c.a;
                this.ba = this.az = a.j.a.d.a;
                this.bb = a.j.a.e.a;
                this.by = this.bx = this.bw = this.bv = this.bu = this.bt = this.bs = this.br = this.bq = this.bp = this.bo = this.bm = this.bl = this.bk = this.bi = this.bh = this.bg = this.bf = this.be = this.bd = this.bc = void 0;
                this.ca = this.bz=!1;
                this.cb = this.cu = this.ct = void 0;
                this.cc =+ new t + 12E4;
                this.BEACON_LAST_SENT_AT =+ new t;
                this.cl = this.cm = void 0;
                this.cn = 0;
                this.ck = a.j.a.f.a;
                this.cd=!1;
                this.cy = void 0;
                this.dt =
                !1;
                this.db = void 0;
                this.cf = this.ce=!1;
                this.af = Number(this.ef);
                this.eq=!1;
                this.ds = this.ch = this.dr = this.cg = 0;
                this.dq = this.bn = void 0;
                this.IR5 = {
                    MIN: {
                        x: void 0,
                        y: void 0
                    },
                    MAX: {
                        x: void 0,
                        y: void 0
                    },
                    AREA: 0
                };
                a.i.e(this);
                this.dm = 0;
                this.ep = this.dd=!1;
                this.aq = {};
                this.aq.g = 0;
                this.aq[1] = 0;
                this.aq[2] = 0;
                this.aq[3] = 0;
                this.aq[13] = 0;
                this.aq[0] = 0;
                this.aq[4] = 0;
                this.aq[5] = 0;
                this.aq[6] = 0;
                this.aq[7] = 0;
                this.aq[9] = 0;
                this.aq[8] = 0;
                this.aq[15] = 0;
                this.aq[16] = 0;
                this.aq[21] = 0;
                this.aq[22] = 0;
                this.aq[23] = 0;
                this.INVIEW_TIME_THRESHHOLDS =
                [5, 10, 15, 30, 60];
                this.DWELL_TIME_THRESHHOLDS = [5, 10, 15, 30, 60];
                this.an = d.adType || e(c);
                0 === this.an && (this.WMODE = l(c));
                a.n.c(this)
            }
            function e(a) {
                return "IFRAME" === a.tagName ? 2 : "IMG" === a.tagName ? 1 : "EMBED" === a.tagName || "OBJECT" === a.tagName ? 0 : 3
            }
            a.n = {};
            a.n.b = function(d, c, b, h, f, g, k) {
                var n;
                n = 1 == arguments.length ? arguments[0] : {
                    el: d,
                    url: c,
                    flashVars: h,
                    adIds: f,
                    opt_props: k
                };
                if (g) {
                    if ("function" === typeof g)
                        return g(d, c, h, f);
                    new t;
                    g.em=!0;
                    u[g.zr] = g;
                    d[R] = g.zr;
                    d[y]=!0;
                    g.aa = d;
                    g.INITIAL_WIDTH = d.offsetWidth;
                    g.INITIAL_HEIGHT =
                    d.offsetHeight;
                    g.ae = c;
                    g.an = e(d);
                    0 === g.an && (g.WMODE = l(d));
                    g.ag = h || {};
                    a.k.a(g);
                    n = {
                        e: 0
                    };
                    n.q = g.aq[0]++;
                    a.l.a(g, n);
                    return g
                }
                return new m(n)
            };
            a.n.c = function(d) {
                d.de = d.ao.startTime;
                d.RAND = d.ao.rand;
                (new t).getTime();
                d.dd=!0;
                a.i.c(d);
                var c = {
                    e: 0
                };
                c.q = d.aq[0]++;
                a.l.a(d, c);
                a.k.a(d)
            };
            a.n.a = function(d) {
                var c =+ new t, b = c - d.BEACON_LAST_SENT_AT;
                if (0 < d.DWELL_TIME_THRESHHOLDS.length) {
                    var e = 1E3 * d.DWELL_TIME_THRESHHOLDS[0];
                    if (d.counters.strictDwell.tCur >= e) {
                        d.DWELL_TIME_THRESHHOLDS.shift();
                        var f = d.INVIEW_TIME_THRESHHOLDS.length ?
                        d.INVIEW_TIME_THRESHHOLDS[0]: 60;
                        if (e < f)
                            return !1;
                        if (5E3 < b)
                            return !0
                    }
                }
                return 0 < d.INVIEW_TIME_THRESHHOLDS.length && (f = 1E3 * d.INVIEW_TIME_THRESHHOLDS[0], a.i.f(d, f)) ? (d.INVIEW_TIME_THRESHHOLDS.shift(), !0) : 0 === d.DWELL_TIME_THRESHHOLDS.length && c > d.cc ? (d.cc*=2, !0) : !1
            };
            a.n.d = function(a, c) {
                for (var b = 0; b < c.length; b++) {
                    var e = c[b];
                    e[R] = a;
                    e[y]=!0
                }
            }
        })(q);
        (function(a) {
            function l(e, d, c) {
                if (!e)
                    return !1;
                for (var b=!1, h = e.getElementsByTagName("DIV"), f = 0; f < h.length; f++) {
                    var g = h[f];
                    if (g.id && ( - 1 < g.id.indexOf("pickleads_wrapper") ||
                    g.id.match(/afp_[\d]*_unexpanded/))) {
                        var k = g.offsetHeight;
                        if (1 < g.offsetWidth && 1 < k) {
                            var n;
                            if (g.id.match(/afp_[\d]*_unexpanded/))
                                for (var l = g.getElementsByTagName("div"), k = 0; k < l.length; k++)
                                    if (l[k].id.match(/afp_[\d]*_unexpanded_image/)) {
                                        l[k].className.match("hidden") && (b=!0);
                                        var m = a.a.m(l[k], "background-image");
                                        m && (n = (n = m.match("url\\((.*)\\)")) && n[1])
                                    }
                            n = n || "DIV";
                            if (!b && (k = a.n.b(g, n, !1, void 0, d, c)))
                                return k
                        }
                    }
                }
                f = a.a.g();
                b = null !== f && 11 > f;
                if (!b)
                    for (k = e.getElementsByTagName("embed"), f = 0; f < k.length; f++)
                        if (n =
                        k[f], !0 !== n[y]&&-1 === n.id.indexOf("moatPx") && a.a.v(n) && n.getAttribute("src")) {
                            var s = n.getAttribute("src");
                            e = a.o.a(s, n);
                            return a.n.b(n, s, !1, e, d, c)
                        }
                k = e.getElementsByTagName("object");
                for (f = 0; f < k.length; f++)
                    if (h = k[f], a.a.v(h) && ("undefined" === typeof h[y] ||!0 !== h[y])&&-1 == h.id.indexOf("moatPx")) {
                        for (e = 0; e < h.children.length; e++)
                            if ("movie" === h.children[e].name || "Movie" === h.children[e].name)
                                for (s = h.children[e].value, k = 0; k < h.children.length; k++) {
                                    if (!b && "EMBED" === h.children[k].tagName) {
                                        n = h.children[k];
                                        if ("undefined" !==
                                        typeof n[y]&&!0 === n[y]||-1 != n.id.indexOf("moatPx"))
                                            continue;
                                            if (a.a.v(n))
                                                return e = a.o.a(s, n), a.n.b(n, s, !1, e, d, c)
                                            }
                                            if ("OBJECT" === h.children[k].tagName) {
                                                var q = h.children[k];
                                                if (a.a.v(q)&&!0 !== q[y]&&-1 === q.id.indexOf("moatPx"))
                                                    return a.n.b(q, void 0, !1, void 0, d, c)
                                                }
                                        }
                                        h.object && h.object.Movie ? s = h.object.Movie : h.data && (s = h.data);
                                        e = a.o.a(s, h);
                                        return a.n.b(h, s, !1, e, d, c)
                            }
                if (k = a.p.h(e, d, c))
                    return k;
                s = e.getElementsByTagName("img");
                for (f = 0; f < s.length; f++)
                    if (k = s[f], "undefined" === typeof k[y] ||!0 !== k[y]) {
                        if (a.a.v(k) &&
                        "" !== k.src&&-1 === document.location.href.indexOf(k.src))
                            return k.id&&-1 < k.id.indexOf("checkm8") && k.src&&-1 < k.src.indexOf("transparent_pixel.png") ? (e.MOATCHECKM8ORIGIN=!0, !1) : a.n.b(k, k.getAttribute("src"), !1, void 0, d, c);
                            try {
                                q = k.src
                            } catch (D) {
                                q = k.getAttribute && k.getAttribute("src")
                            }
                            L["1"] = k.offsetWidth + "x" + k.offsetHeight + ":" + q
                        }
                if (s = e.getElementsByTagName("canvas")[0])
                    if (d = a.n.b(s.parentNode, s.parentNode.nodeName, !1, void 0, d, c))
                        return d;
                return !1
            }
            function m(e, d) {
                for (var c = [], b = e.getElementsByTagName("iframe"),
                h, f = 0; f < b.length; f++)
                    if (h = b[f], !h[y] && a.a.v(h)) {
                        var g = a.d.c(h)?!1 : !0;
                        (1 === d && g || 2 === d&&!g) && c.push(h)
                    }
                return c
            }
            a.p = {};
            a.p.a = function(e, d, c, b, h) {
                var f = a.p.b, g = 0;
                a.e.a(function() {
                    d.numTries = g++;
                    return f(e, d, c, b)&&!0
                }, a.p.c, 500, h)
            };
            a.p.h = function(e, d, c) {
                e = m(e, 1);
                if (e[0])
                    return a.n.b(e[0], e[0].src, !1, void 0, d, c)
                };
            a.p.g = function(e, d, c) {
                var b;
                e = a.p.e(e, a.p.i);
                for (var h = 0; h < e.length; h++)
                    if (b = e[h], (b = a.d.c(b)) && b.documentElement && (b = l(b.documentElement, d, c)))
                        return b
                        };
            a.p.j = function(e, d, c) {
                var b, h;
                e = a.p.e(e,
                a.p.i);
                for (var f = 0; f < e.length; f++)
                    if (h = e[f], (h = a.d.c(h)) && h.documentElement) {
                        if (b = h.getElementById("flite-1"))
                            if (b = a.n.b(b, b.nodeName, !1, void 0, d, c))
                                return b;
                                if (b = l(h.documentElement, d, c))
                                    return b
                    }
                };
            a.p.d = l;
            a.p.b = function(e, d, c, b) {
                var h = a.p.d;
                if ("undefined" === typeof e)
                    return !1;
                if ("HEAD" === e.tagName) {
                    var f = e.parentNode;
                    "HTML" === f.tagName && (f = f.getElementsByTagName("body"), 0 < f.length && (e = f[0]))
                }
                d = a.p.f(d);
                if ((f = h(e, d, c)) || (f = a.p.g(e, d, c)) || ba && (b = a.d.b(e), f = a.p.d(b, d, c)))
                    return f;
                try {
                    if (ba && J) {
                        var g =
                        a.d.b(e), k;
                        a:
                        {
                            var n = g.ownerDocument.getElementById("checkm8Cont_0");
                            if (null === n)
                                k = null;
                            else {
                                for (var l = n, m = 0; (l = l.parentElement) && 10 > m;)
                                    if (m++, "TopAd" == l.id) {
                                        k = n;
                                        break a
                                    }
                                k = void 0
                            }
                        }
                        if ("TopAd" == g.id && null !== k && (f = a.n.b(k, k.nodeName, !1, void 0, d, c)))
                            return f
                    }
                } catch (q) {}
                if ((ha || b) && (b = b || a.d.b(e)) && (!a.b.a || "BODY" !== b.nodeName || a.c.a(b) != a.b.c) && (f = h(b, d, c)))
                    return f;
                e = e.getElementsByTagName("DIV");
                for (b = 0; b < e.length; b++)
                    if (g = e[b], g.id&&-1 < g.id.indexOf("checkm8") && (k = g.offsetHeight, g = g.offsetWidth, 1 < g &&
                    1 < k)) {
                        a:
                        {
                            n = document.getElementsByTagName("DIV");
                            for (f = 0; f < n.length; f++)
                                if (l = n[f], l.id&&-1 < l.id.indexOf("checkm8Cont")&&!0 !== l.MOATCHECKM8ORIGIN) {
                                    var r = l.style;
                                    if (r && (m = r.width, r = r.height, m && r&&-1 < m.indexOf(g)&&-1 < r.indexOf(k))) {
                                        g = l;
                                        break a
                                    }
                                }
                                g=!1
                            }
                            if (g) {
                                if (f = h(g, d, c))
                                    return f;
                                    break
                            }
                    }
                return !1
            };
            a.p.e = m;
            a.p.k = 1;
            a.p.i = 2;
            a.p.l = 500;
            a.p.c = 20;
            a.p.m = {
                object: 1,
                embed: 1,
                img: 1,
                iframe: 1
            }
        })(q);
        (function(a) {
            function l(e) {
                try {
                    if ("string" === typeof e) {
                        e = e.toLowerCase();
                        var d = e.split(".")[0];
                        if (d && a.a.h("travel travel2 theater theater2 homepage blogs health blog".split(" "),
                        d)) {
                            if ("travel2" === d || "theater2" === d || "blogs" === d)
                                d = d.slice(0, - 1);
                            if ("homepage" !== d && "blog" === d) {
                                var c = e.split("/");
                                c[1] && (d = c[2] ? d + "-" + c[1] + ("frontpage" === c[2] ? "-SectionFront" : "-Article") : d + "-" + c[1] + "-Other")
                            }
                            return d
                        }
                        var b = a.a.j("www.nytimes.com/yr/mo/day/ video.nytimes.com/video/playlist/ www.nytimes.com/top/news/ www.nytimes.com/archive/article/ www.nytimes.com/aponline/ www.nytimes.com/ref/ www.nytimes.com/slideshow/yr/mo/day/ global.nytimes.com/pages/ global.nytimes.com/yr/mo/day/ www.nytimes.com/interactive/yr/mo/day/ www.nytimes.com/pages/ prototype.nytimes.com/yr/mo/day/ ipad.nytimes.com/sectionfront/section/ ipad.nytimes.com/sectionfront/blogs/ ipad.nytimes.com/article/section/ ipad.nytimes.com/article/blog/ mraid.ipad/article/ mraid.ipad/sectionfront/ iphone.mobile.nytimes.com/yr/mo/day/ iphone.mobile.nytimes.com/pages/ mobile.nytimes.com/pages/ mobile.nytimes.com/yr/mo/day/ mweb.nytimes.com/yr/mo/day/ mweb.nytimes.com/pages/ www.nytimes.com/recommendations/".split(" "),
                        e);
                        if (b) {
                            var h = e.split(b)[1], f = h.split("/")[0];
                            if (f) {
                                var d = ["ipad.nytimes.com", "mraid.ipad"], c = ["iphone.nytimes.com"], g = ["mobile.nytimes.com", "mweb.nytimes.com"];
                                - 1 < e.indexOf("prototype.nytimes.com") ? f = "prototype-" + f : a.a.j(d, e) ? f = "iPadNewsApp-" + f : a.a.j(c, e) ? f = "iPhoneNewsApp-" + f : a.a.j(g, e) && (f = "MobileWeb-" + f);
                                if (f.match(/blogs?/))
                                    var k = h.split("/"), f = "directory.html" === k[k.length - 1] ? f + "-SectionFront": f + "-Other";
                                else 
                                    f = (isSectionFront = a.a.j(["/pages/", "/sectionfront/", "/frontpage"], b)) ? f + "-SectionFront" :
                                    f + "-Article";
                                "www.nytimes.com/recommendations/" === b && (f =- 1 != f.indexOf("module") ? "recommendations-" + f.split("-")[0] : "recommendations-" + f);
                                return f
                            }
                        }
                    }
                    return "other"
                } catch (n) {
                    return "other"
                }
            }
            function m(a, d) {
                for (var c = {}, b = a.split("&"), h = 0; h < b.length; h++) {
                    var f = b[h].split("=");
                    if ("string" === typeof f[1]) {
                        var g = f, k, n = k = f[1];
                        try {
                            for (var m = 0; 100 > m && (k = decodeURIComponent(k), n != k)&&!k.match(/^http(s)?\:/); m++)
                                n = k
                        } catch (q) {}
                        k = k.replace(/(^\s+|\s+$)/g, "");
                        g[1] = k
                    } else 
                        f[1] = "";
                    c[f[0]] = f[1]
                }
                c.parsedSection = l(c.moatClientSlicer1);
                c.zMoatVER&&-1 < c.zMoatVER.indexOf("nyt5") && (c.parsedSection = "nyt5-" + c.parsedSection);
                "undefined" !== typeof d && (c.clientZone = "undefined" !== d ? d : "");
                return c
            }
            a.q = {};
            a.q.a = function(a, d) {
                if (!a)
                    return !1;
                if ("undefined" === typeof a.startTime || d)
                    a.startTime = (new t).getTime();
                if ("undefined" === typeof a.rand || d)
                    a.rand = r.floor(r.random() * r.pow(10, 8));
                a.adNum = v.zr;
                v.zr++
            };
            a.q.b = function(e) {
                if (!e)
                    return !1;
                var d;
                d = /^(?:[a-z]+:\/\/|:?\/?\/)?(?:www\.)?([^\/:]*)/i;
                d=!a.b.a && a.b.b ? (d = a.b.b.match(d)) && d[1] ? d[1] : a.b.c.location.hostname :
                a.b.c.location.hostname;
                var c = d.match(/.*\.([^\.]*\..*)/i);
                e.moatClientSlicer1 = c && c[1] ? decodeURIComponent(c[1]) : decodeURIComponent(d);
                e.moatClientSlicer2 = decodeURIComponent(d);
                return e
            };
            a.q.c = function(e) {
                return a.q.d(e)
            };
            a.q.d = function(a) {
                try {
                    var d = a.className, c = a.getAttribute("src");
                    d.split("\n").join(" ");
                    if ( - 1 !== d.indexOf("moatfooter"))
                        return !1;
                    var b = c.split("?"), h = c.split("#");
                    a=!1;
                    1 < b.length && 1 < h.length && b[1].length < h[1].length && (a=!0);
                    if (1 == b.length || a)
                        b = h;
                    return 1 < b.length ? m(b[1], "undefined") :
                    !1
                } catch (f) {
                    return !1
                }
            };
            a.q.e = function(a) {
                try {
                    var d = a && a.className.replace("amp;", "").split("?")[1];
                    return d && m(d)
                } catch (c) {
                    return !1
                }
            };
            a.q.f = function(a) {
                try {
                    var d = zoneRegEx.exec(a.innerHTML);
                    0 < d.length && (zone = d[1]);
                    return zone
                } catch (c) {}
            };
            a.q.g = function(a) {
                return (a = unescape(unescape(unescape(unescape(a.innerHTML)))).match(/~fdr=(\d*).*?\/.*?;(\d*)/)) ? {
                    adid: a && a[1] || "-",
                    cid: a && a[2] || "-"
                } : !1
            };
            a.q.h = function(a, d) {
                return d || {}
            }
        })(q);
        (function(a) {
            function l(c, b) {
                var d, e = [], g, k = a.a.f() ? 2048: 7750, n = b || {};
                g = {};
                for (d in c)
                    c.hasOwnProperty(d) && (1 != c.e || "x" !== d && "y" !== d && "c" !== d ? e.push(encodeURIComponent(d) + "=" + encodeURIComponent(c[d])) : g[d] = c[d].split("a"));
                d = e.join("&");
                var e = k - d.length, l = 0;
                if ("undefined" !== typeof g.x) {
                    for (var m = 0, q = 0; q < g.x.length; q++)
                        if (m += g.x[q].length + (g.y[q] ? g.y[q].length : 0) + (g.c[q] ? g.c[q].length : 0), m < e)
                            l++;
                        else 
                            break;
                    0 < l && (d += "&x=" + g.x.slice(0, l - 1).join("a"), d += "&y=" + g.y.slice(0, l - 1).join("a"), d += "&c=" + g.c.slice(0, l - 1).join("a"))
                }
                for (var r in n)
                    n.hasOwnProperty(r) && (g = "&" + encodeURIComponent(r) +
                    "=" + encodeURIComponent(n[r]), g.length + d.length < k && (d += g));
                return d
            }
            function m(a, b) {
                for (var d in b)
                    b.hasOwnProperty(d) && (a[d] = b[d])
            }
            a.l = {};
            var e = ["zMoatPMode", "zMoatPMVer"], d = {
                34: "bd.moatads.com",
                36: "bd.moatads.com"
            };
            a.l.b = function(c, b, h, f) {
                a.q.a(b, f);
                var g = {};
                g.e = c;
                m(g, h);
                g.i = E;
                g.cm = a.a.e(7, 3).multiplier;
                if (11 === c) {
                    h = [];
                    for (var k in L)
                        L.hasOwnProperty(k) && h.push(k + "=" + L[k]);
                    g.k = h.join("&")
                }
                g.e in ca || (g.bq = a.b.e, g.f = Number(!J), g.j = a.b.b, g.o = 3, g.t = b.startTime, g.de = b.rand, g.m = 0, g.ar = S, a.a.ad(g, "ai",
                v.z), g.q = v.m++, g.cb = B ? 1 : 0, g.cu = T, g.r = a.h.ae, m(g, a.focus.getQueryString()), a.q.h(b, g), "undefined" !== typeof b && (g.d = b.moatClientLevel1 + ":" + b.moatClientLevel2 + ":" + b.moatClientLevel3 + ":" + b.moatClientLevel4, a.a.forEach(e, function(a) {
                    g[a] = b[a] || "-"
                }), g.bd = b.parsedSection, g.zMoatSite = b.moatClientSlicer1, g.zMoatAT = b.zMoatAT || "-"), g.ac = 1, g.it = a.p.l, k = l(g), k += "&cs=0", h = d[c] ? a.b.protocol + "//" + d[c] : z, c = d[c] ? a.b.protocol + "//" + d[c] : Y, a.a.e(7, 3).sample && v.yh.yi(k, h, c))
            };
            a.l.c = function(c) {
                !0 !== c.em && (delete u[c.zr],
                clearTimeout(c.cc), a.p.a(Q.parentNode, c.ao, c))
            };
            a.l.a = function(c, b) {
                if (!c ||!0 === c.ep)
                    return !1;
                if (!a.c.h(c))
                    return a.c.i(c), !1;
                (3 > c.aa.offsetHeight || 3 > c.aa.offsetWidth ||!c.aa.parentNode)&&!1 !== c.aa.findAgain && a.l.c(c);
                if ("undefined" !== typeof c.ao && (2 !== c.an || 1 !== b.e && 3 !== b.e)&&!(b.e in ca)) {
                    b.lo = c.FIND_AD_TRIES;
                    if ("string" === typeof c.ae) {
                        var h;
                        h = a.b.h ? 700 : 1200;
                        b.ak = c.ae.length <= h ? c.ae : c.ae.slice(0, h)
                    } else 
                        b.ak = "-";
                    c.ah && (b.a = c.ah);
                    var f = c.ag;
                    h = {};
                    if (9 === b.e && 2 === b.q || 25 === b.e) {
                        for (var g in f)
                            f.hasOwnProperty(g) &&
                            "" !== g && "undefined" !== typeof f[g]&&-1 === g.indexOf("dvContains")&&-1 === g.indexOf("indexOf")&&-1 === g.toLowerCase().indexOf("clicktag") && (h["z" + g] = f[g]);
                        b.e = 25
                    }
                    0 === c.an && (b.dc = c.WMODE);
                    0 !== b.e && a.c.j(c);
                    c.bi > c.bg && (c.bg = c.bi);
                    c.bm > c.bk && (c.bk = c.bm);
                    b.i = E;
                    b.bq = a.b.e;
                    b.g = c.aq.g++;
                    g = c.INITIAL_WIDTH;
                    b.h = c.INITIAL_HEIGHT;
                    b.w = g;
                    b.cm = a.a.e(7, 3).multiplier;
                    b.f = Number(!J);
                    b.j = a.b.b;
                    b.o = 3;
                    b.t = c.de;
                    b.de = c.RAND;
                    b.cu = T;
                    b.m = b.m || a.a.k(c);
                    b.ar = S;
                    b.cb = B ? 1 : 0;
                    b.r = a.h.ae;
                    m(b, a.m.c());
                    a.b.a && (b.gh = 1);
                    a.m && a.m.b() ? (b.ch =
                    1, b.gh = 1) : a.h && a.h.v ? (b.ct = a.h.ad, c && c.periscopeManager ? (c.periscopeManager.measurable && (b.ch = 1), c.periscopeManager.fullyMeasurable && c.ao&&!c.ao.skin && (b.ga = 1)) : b.ch = 1, "undefined" !== typeof a.h.af && c && c.ao && c.ao.startTime&&!isNaN(c.ao.startTime) && (g = a.h.af - c.ao.startTime, b.fg = 0 <= g ? g : 0)) : b.ch = 0;
                    m(b, a.i.l(c.zr));
                    m(b, a.focus.getQueryString());
                    m(b, c.counters.getQs());
                    a.a.ad(b, "ai", v.z);
                    a.a.ad(b, "ap", c.cb);
                    a.a.ad(b, "ax", c.bg);
                    a.a.ad(b, "ay", c.bi);
                    a.a.ad(b, "az", c.bk);
                    a.a.ad(b, "ba", c.bm);
                    a.a.ad(b, "aw", c.bc);
                    a.a.ad(b, "bg", c.bd);
                    a.a.ad(b, "be", c.be);
                    a.a.ad(b, "bc", c.bw);
                    a.a.ad(b, "bf", c.by);
                    a.a.ad(b, "bh", c.bx);
                    a.a.ad(b, "bz", c.cu);
                    b.cl = r.round(100 * c.IR5.AREA / (b.w * b.h));
                    b.au = c.aq[2] - 1;
                    b.av = c.aq[3] - 1;
                    b.by = c.aq[23] - 1;
                    b.at = c.dm;
                    a.q.h(c.ao, b);
                    b.d = c.ao.moatClientLevel1 + ":" + c.ao.moatClientLevel2 + ":" + c.ao.moatClientLevel3 + ":" + c.ao.moatClientLevel4;
                    b.bd = c.ao.parsedSection;
                    b.zMoatSite = c.ao.moatClientSlicer1;
                    b.zMoatAT = c.ao.zMoatAT || "-";
                    a.a.forEach(e, function(a) {
                        b[a] = c.ao[a] || "-"
                    });
                    b.ab = c.an;
                    b.ac = 1;
                    b.zMoatSeg = Number(3E4 <=
                    b.ah && b.hasOwnProperty("ai"));
                    b.it = a.p.l;
                    c.bi = c.bg;
                    c.bm = c.bk;
                    h = l(b, h);
                    h += "&cs=0";
                    g = d[b.e] ? a.b.protocol + "//" + d[b.e] : z;
                    f = d[b.e] ? a.b.protocol + "//" + d[b.e] : Y;
                    a.a.e(7, 3).sample && v.yh.yi(h, g, f)
                }
            };
            a.l.d = function(a, b) {
                if (2 !== a.an || 1 !== b.e && 3 !== b.e) {
                    var d = ia;
                    (new Image(1, 1)).src = d
                }
            };
            a.l.e = function(a, b) {
                return l(a, b)
            };
            a.l.f = function(c) {
                var b = {
                    e: 16
                };
                b.q = c.aq[16]++;
                a.l.a(c, b)
            }
        })(q);
        (function(a) {
            function l(a, d, c, b) {
                var h = (new t).getTime();
                this.tLast = this.tCur = 0;
                this.update = function(c) {
                    var d = (new t).getTime();
                    if (a(c)) {
                        c =
                        r.min(d - h, 1E3);
                        var k = typeof b;
                        this.tCur += c;
                        "number" === k ? this.tCur > b && (this.tCur = b) : "function" === k && (c = b(), "number" === typeof c && this.tCur > c && (this.tCur = c))
                    }
                    h = d
                };
                this.getQs = function(a) {
                    a = this.query(a);
                    this.tLast = this.tCur;
                    return a
                };
                this.query = function(a) {
                    a = a || {};
                    this.tLast > this.tCur && (this.tLast = this.tCur);
                    a[d] = this.tCur;
                    a[c] = this.tLast;
                    return a
                }
            }
            function m() {
                if (a.focus.pageIsVisible() && "undefined" === typeof v.z) {
                    v.z = new t - T;
                    a:
                    {
                        var e = void 0, d;
                        for (d in u)
                            if (u.hasOwnProperty(d) && (e = u[d]) && "undefined" !==
                            typeof e.ao) {
                                if (e.ce)
                                    break a;
                                    var c = {
                                        e: 4
                                    };
                                    c.q = e.aq[4]++;
                                    c.ai = v.z;
                                    a.l.a(e, c);
                                    e.ce=!0
                            }
                        a.e.e(A, "scroll", m, "onScroll")
                    }
                }
            }
            a.c = {};
            a.c.k = function() {
                a.e.d(A, "scroll", m, "onScroll");
                a.focus.setFocusListeners()
            };
            a.c.h = function(e, d) {
                try {
                    var c = e.aa, b = a.a.o(c, 5), h = b && (6 == b.length || 1 <= b.length && "HTML" === b[b.length - 1].nodeName);
                    d = d || e.WINDOW || a.c.a(c);
                    return c && d && h?!0 : !1
                } catch (f) {
                    return !1
                }
            };
            a.c.l = function() {
                var e;
                return function() {
                    for (var d = 0, c = M.length; d < c; d++)
                        if (M[d] === e)
                            return;
                    e = a.e.f(function() {
                        a.c.j();
                        for (var b in u)
                            u.hasOwnProperty(b) &&
                            a.i.c(u[b])
                    }, 200)
                }
            }();
            a.c.j = function() {
                var e, d;
                for (d in u)
                    u.hasOwnProperty(d) && (e = u[d], a.c.h(e, e.WINDOW) ? e.counters.update(e) : a.c.i(e))
            };
            a.c.i = function(e) {
                if (!0 !== e.em)
                    a.l.c(e);
                else {
                    clearTimeout(e.cc);
                    a.h.r(e);
                    a.e.e(A, "scroll", m, "onScroll");
                    e.ep=!0;
                    delete u[e.zr];
                    a.k.a(e, "remove");
                    a.i.j(e.zr);
                    e.aa = null;
                    e = 0;
                    for (prop in u)
                        u.hasOwnProperty && u.hasOwnProperty(prop) && e++;
                    0 === e && Z()
                }
            };
            a.c.f = function(e) {
                e.eq || (e.eq=!0);
                var d = {
                    e: 5
                };
                d.q = e.aq[5]++;
                a.l.a(e, d)
            };
            a.c.e = function(e) {
                var d, c;
                c = e.aa;
                e.elementRect ? (d =
                e.elementRect.right - e.elementRect.left, e = e.elementRect.bottom - e.elementRect.top) : (d = c.offsetWidth, e = c.offsetHeight);
                return 3 > d || 3 > e || a.focus.pageIsPrerendered()?!0 : !1
            };
            a.c.m = function(a) {
                var d = 1;
                screen.deviceXDPI ? d = screen.deviceXDPI / screen.systemXDPI : a.devicePixelRatio && "undefined" !== typeof a.mozInnerScreenX && (d = a.devicePixelRatio);
                return {
                    w: d * screen.width,
                    h: d * screen.height
                }
            };
            a.c.a = function(a) {
                try {
                    var d = a && a.ownerDocument;
                    return d && (d.defaultView || d.parentWindow)
                } catch (c) {
                    return !1
                }
            };
            a.c.g = function(e) {
                e.counters =
                {};
                e.counters.laxDwell = new l(function() {
                    return !a.focus.pageIsPrerendered()
                }, "bu", "cd");
                e.counters.strictDwell = new l(a.focus.pageIsVisible, "ah", "am");
                e.counters.query = function() {
                    var a = {}, c;
                    for (c in this)
                        if (this.hasOwnProperty(c)) {
                            var b = this[c];
                            "function" === typeof b.query && b.query(a)
                        }
                    return a
                };
                e.counters.getQs = function() {
                    var a = {}, c;
                    for (c in this)
                        if (this.hasOwnProperty(c)) {
                            var b = this[c];
                            "function" === typeof b.getQs && b.getQs(a)
                        }
                    return a
                };
                e.counters.update = function(a) {
                    for (var c in this)
                        if (this.hasOwnProperty(c)) {
                            var b =
                            this[c];
                            "function" === typeof b.update && b.update(a)
                        }
                }
            };
            a.c.b = function(e, d) {
                for (var c = [], b, h = 0; h < ja; h++)
                    if (d != d.parent) {
                        if (b = a.d.a(e, d))
                            c.push(b);
                        else 
                            break;
                            d = d.parent;
                            e = b
                    } else 
                        break;
                return c
            };
            a.c.n = function() {
                v.z = void 0;
                v.zs=!1;
                a.e.e(A, "scroll", m, "onScroll")
            };
            a.c.o = function() {
                for (var e in u)
                    if (u.hasOwnProperty(e)) {
                        var d = u[e];
                        if (d) {
                            a.r && a.r.a && a.r.a(d);
                            var c = {
                                e: 21
                            };
                            c.q = d.aq[21]++;
                            a.l.a(d, c)
                        }
                    }
                };
            a.c.c = function(e) {
                var d = {
                    e: 9
                };
                d.q = e.aq[9]++;
                e.BEACON_LAST_SENT_AT =+ new t;
                a.l.a(e, d)
            }
        })(q);
        (function(a) {
            a.d =
            {};
            a.d.c = function(l) {
                try {
                    if (l.moatHostileIframe)
                        return null;
                    if (l.src && l.src.slice && "http" === l.src.slice(0, 4) && a.a.l(l.src) != a.a.l(A.location.toString()))
                        return l.moatHostileIframe=!0, null;
                    var m = l && (l.contentDocument || l.contentWindow && l.contentWindow.document);
                    if (m && "string" === typeof m.location.toString())
                        return m;
                    l.moatHostileIframe=!0;
                    return null
                } catch (e) {
                    return l.moatHostileIframe=!0, null
                }
            };
            a.d.a = function(l, m) {
                m = m || a.c.a(l);
                try {
                    return m && m.frameElement
                } catch (e) {
                    return !1
                }
            };
            a.d.b = function(l) {
                return (l =
                a.d.a(l)) ? l.parentNode : null
            }
        })(q);
        (function(a) {
            a.s = {};
            a.s.a = function(l) {
                l[a.b.d] = l[a.b.d] || {
                    zs: !1,
                    zr: 0,
                    h: 0,
                    m: 0,
                    i: {}
                }
            }
        })(q);
        (function(a) {
            var l = function(a, e) {
                function d(a, c, d) {
                    a && f.push({
                        qs: a,
                        jsd: c,
                        fld: d
                    });
                    if (0 === g && 0 < f.length)
                        if (g += 1, a = f.shift(), a.fld && q && l && l.sendMessage)
                            try {
                                l.sendMessage(a)
                    } catch (e) {
                        q=!1, b(a)
                    } else 
                        b(a)
                }
                function c() {
                    try {
                        return new s(1, 1)
                    } catch (a) {
                        var b = window.document.createElement("img");
                        b.height = 1;
                        b.width = 1;
                        return b
                    }
                }
                function b(a) {
                    var d = c();
                    d.toSend = a;
                    d.onerror = function() {
                        var a =
                        this.toSend;
                        k += 1;
                        var c = (a.jsd + "/pixel.gif?" + a.qs).length;
                        2 > k ? b(a) : G && c > P && h()
                    };
                    d.onload = function() {
                        h()
                    };
                    d.src = a.jsd + "/pixel.gif?" + a.qs
                }
                function h() {
                    0 < g && (g -= 1, d())
                }
                var f = [], g = 0, k = 0, l=!1, q=!1, r=!1, s, t = e[a];
                t.yh = {};
                t = t.yh;
                s = e.Image;
                t.yi = function(a, b, c) {
                    d(a, b, c)
                };
                t.yk = function(b) {
                    l=!0;
                    var c = a + ".yh.", d = {}, e;
                    try {
                        e = window.location.protocol
                    } catch (f) {
                        e = document.createElement("a"), e.href = "", e = e.protocol
                    }
                    d.src = "https:" === e ? "https://z.moatads.com/swf/MessageSenderV2.swf" : "http://s.moatads.com/swf/MessageSenderV2.swf";
                    d.flashVars = "r=" + c + "zb&s=" + c + "zc&e=" + c + "zd&td=" + b;
                    return d
                };
                t.yj = function() {
                    return !1 === l
                };
                t.qb = function(a) {
                    r=!0;
                    a = {};
                    var b;
                    try {
                        b = window.location.protocol
                    } catch (c) {
                        b = document.createElement("a"), b.href = "", b = b.protocol
                    }
                    a.src = "https:" === b ? "https://z.moatads.com/swf/cap.swf" : "http://s.moatads.com/swf/cap.swf";
                    return a
                };
                t.qa = function() {
                    return !1 === r
                };
                t.zb = function() {
                    try {
                        if (!0 === l) {
                            var a = e.document.getElementById("moatMessageSender");
                            a&&!a.sendMessage && (a = e.document.getElementById("moatMessageSenderEmbed"));
                            a && a.sendMessage && (q=!0, l = a)
                        }
                    } catch (b) {}
                };
                t.zc = function() {
                    try {
                        h()
                    } catch (a) {}
                };
                t.zd = function(a) {
                    try {
                        q=!1, a && a.jsd && f.push(a), h()
                    } catch (b) {}
                };
                var D, G, P = 2083;
                try {
                    D = document.createElement("div"), D.innerHTML = "\x3c!--[if IE 8]>x<![endif]--\x3e", G = "x" === D.innerHTML
                } catch (x) {
                    G=!1
                }
            }.toString();
            a.t = {};
            a.t.a = function() {
                return "undefined" !== typeof A.eval && (A.eval("(function(win){ win['Moat#EVA'] = true; })(window)"), "undefined" !== typeof A["Moat#EVA"])?!0 : !1
            };
            a.t.b = function(m) {
                if (!v.yh)
                    if (a.t.a())
                        m.eval("(" + l +
                        ")('" + a.b.d + "',window)");
                    else {
                        var e = m.document.createElement("script");
                        e.type = "text/javascript";
                        e.text = "(" + l + ")('" + a.b.d + "',window)";
                        m.document.body.insertBefore(e, m.document.body.childNodes[0])
                    }
            }
        })(q);
        (function(a) {
            a.u = {};
            var l = function(a) {
                for (var b = [], c = 0, d = a.length; c < d; c++)
                    b.push(new RegExp(a[c]));
                return b
            }(".*(Silk).*;.*(Kindle).*;.*(Opera) Mini.*;.*(OPR)/[0-9]+.*;.*(AOLBuild).*;.*(Chrome).*;.*(CriOS).*;.*(PlayBook).*;.*(Android).*;.*(BlackBerry).*;.*(BB10).*;.*(Safari).*;.*(Opera).*;.*(IEMobile).*;.*(Trident)/[0-9.]*.*rv:[0-9.]*.*;.*(MSIE).*;.*(Firefox).*;.*(NetNewsWire).*;.*FBAN/(FBForIPhone).*;.*(Moozilla).*;.* (Mobile)/[0-9A-Z]+.*".split(";"));
            a.u.a = function(a) {
                a = a || navigator.userAgent || "";
                a: {
                    for (var b = 0, c = l.length; b < c; b++) {
                        var d = l[b].exec(a);
                        if (d) {
                            a = d[1];
                            break a
                        }
                    }
                    a = "-"
                }
                a = {
                    CriOS: "Chrome",
                    AOLBuild: "MSIE",
                    Trident: "MSIE",
                    Mobile: "InAppContent",
                    BB10: "BlackBerry",
                    PlayBook: "BlackBerryPlaybook",
                    NetNewsWire: "RSSreader",
                    OPR: "Opera"
                }
                [a] || a;
                var b = {
                    msie: "MSIE",
                    chrome: "Chrome",
                    firefox: "Firefox",
                    opera: "Opera",
                    safari: "Safari"
                }, c = {
                    iphone: "iPhone",
                    kindle: "Kindle",
                    ipad: "iPad",
                    ipod: "iPod",
                    android: "Android",
                    blackberry: "BlackBerry"
                }, e;
                for (e in c)
                    if (c.hasOwnProperty(e) &&
                    (d = new RegExp(e, "i"), d.test(a)))
                        return c[e];
                for (var f in b)
                    if (b.hasOwnProperty(f) && (d = new RegExp(f, "i"), d.test(a)))
                        return b[f];
                return !1
            };
            a.u.b = function() {
                var c=!!window.opera, d = "undefined" !== typeof InstallTrigger || "MozAppearance"in document.documentElement.style, e = 0 < Object.prototype.toString.call(window.HTMLElement).indexOf("Constructor"), f=!!window.chrome&&!!window.chrome.webstore, g=!+"\v1"||!!document.documentMode||!!window.ActiveXObject || "-ms-scroll-limit"in document.documentElement.style && "-ms-ime-align"in
                document.documentElement.style, h = {
                    Chrome: f,
                    Firefox: d,
                    Opera: c,
                    Safari: e,
                    MSIE: g
                };
                a.u.c = [!0===!!window.opera ? 1: 0, "undefined" !== typeof InstallTrigger===!0 ? 1: 0, !0===!!window.sidebar ? 1: 0, !0 === "MozAppearance"in document.documentElement.style ? 1: 0, !0 === 0 < Object.prototype.toString.call(window.HTMLElement).indexOf("Constructor") ? 1: 0, !0===!!window.chrome ? 1: 0, !0===!(!window.chrome ||!window.chrome.webstore) ? 1: 0, !0===!+"\v1" ? 1: 0, !0===!!document.documentMode ? 1: 0, !0===!!window.ActiveXObject ? 1: 0, !0 === "-ms-scroll-limit"in
                document.documentElement.style ? 1: 0, !0 === "-ms-ime-align"in document.documentElement.style ? 1: 0];
                a.u.d = [!0 === f ? 1: 0, !0 === d ? 1: 0, !0 === c ? 1: 0, !0 === e ? 1: 0, !0 === g ? 1: 0];
                for (b in h)
                    if (h.hasOwnProperty(b) && h[b])
                        return b;
                return !1
            };
            a.u.e = a.u.a();
            a.u.f = a.u.b();
            var m = ["Chrome", "Firefox", "Opera", "Safari", "MSIE"];
            a.u.g = a.a.indexOf(m, a.u.f);
            a.u.h = a.a.indexOf(m, a.u.e);
            a.u.i = a.u.e && a.u.f && 0 <= a.u.h && a.u.e != a.u.f;
            for (var e = [[1, 25], [7, 1], [1, 25], [ - 74, 1], [1, 9], [ - 24, 1], [2, 1], [1, 3], [2, 1], [1, 4], [2, 1], [1, 1], [11, 1], [1, 6], [27, 1],
            [2, 1], [1, 3], [27, 1], [1, 3], [ - 92, 1]], d = 65, c = "", b = 0, h = function(a) {
                for (var b = "", d = 0; d < a.length; d++)
                    var e = a.charCodeAt(d)^85, b = b + String.fromCharCode(e);
                a = b;
                for (var b = "", f = e = d = 0, g = 0, h = 0, k = 0; k < a.length; ++k)
                    for (h = a.charCodeAt(k), g = 255 < h ? 0 : 1; 2 > g; ++g)
                        d = 0 === g ? d | (h & 65280) / 256<<e : d | (h & 255)<<e, e += 8, 13 < e && (f = d & 8191, 88 < f ? (d>>=13, e -= 13) : (f = d & 16383, d>>=14, e -= 14), b += c.charAt(f%91), b += c.charAt(f / 91 | 0));
                0 < e && (b += c.charAt(d%91), 7 < e || 90 < d) && (b += c.charAt(d / 91 | 0));
                return b
            }, m = function(b, c) {
                var d = [];
                a.a.forEach(b, function(a,
                b) {
                    if (a && "string" === typeof(c ? a[c] : a)) {
                        var e = (c ? a[c] : a).split("&").join("%26").split("=").join("%3D");
                        d.push(("number" === typeof b ? "" : b + "=") + e)
                    }
                }, null, !0);
                d.sort();
                return d.join("&")
            }, b = 0; b < e.length; b++)
                for (var f = 0; f < e[b][1]; f++)
                    c += String.fromCharCode(d), d += e[b][0];
            var c = c + String.fromCharCode(d), d = a.b.c, e = d.document, g = e.body, f = d.screen, k = d.navigator;
            a.u.j = f.availWidth;
            a.u.k = f.availHeight;
            a.u.l = f.width;
            a.u.m = f.height;
            a.u.n = d.innerWidth || e.documentElement.clientWidth || g.clientWidth;
            a.u.o = d.innerHeight ||
            e.documentElement.clientHeight || g.clientHeight;
            a.u.p = d.outerWidth || g.offsetWidth;
            a.u.q = d.outerHeight || g.offsetHeight;
            a.u.r = d.screenX || d.screenLeft || d.screenX;
            a.u.s = d.screenY || d.screenTop || d.screenY;
            a.u.t = h(m(k));
            a.u.u = h(m(k.plugins, "name"));
            a.u.v = (new t).getTimezoneOffset();
            a.u.d = a.u.d.join("");
            a.u.c = a.u.c.join("");
            a.u.w = 0;
            a.u.x = 0;
            var m = function(a) {
                for (var b = "", d = 0; d < a.length; d++)
                    a.hasOwnProperty(d) && (b += c[a[d]]);
                return b
            }, d = m([48, 30, 27, 29, 43, 34, 47, 30, 43]), f = m([30, 47, 26, 37, 46, 26, 45, 30]), k = m([43, 30,
            44, 41, 40, 39, 44, 30]), n = [d, f].join("-"), q = [n, k].join("-");
            m([38, 40, 46, 44, 30, 38, 40, 47, 30]);
            var r = function() {
                var b = {};
                b.qr = a.u.w;
                b.qo = a.u.x;
                a.l.b(36, a.u.y, b)
            }, s = function() {
                var b = {};
                b.ud = a.u.g;
                b.up = a.u.h;
                b.qa = a.u.l;
                b.qb = a.u.m;
                b.qc = a.u.r;
                b.qd = a.u.s;
                b.qf = a.u.n;
                b.qe = a.u.o;
                b.qh = a.u.p;
                b.qg = a.u.q;
                b.qi = a.u.j;
                b.qj = a.u.k;
                b.qk = a.u.z;
                b.ql = a.u.u;
                b.qm = a.u.v;
                b.qn = a.u.t;
                b.qo = a.u.x;
                b.qp = a.u.d;
                b.qq = a.u.c;
                b.qr = a.u.w;
                a.l.b(34, a.u.y, b)
            }, u = function() {
                var b = e.getElementById("moatCap");
                if (!b ||!b.gc && (b = e.getElementById("moatCapEmbed"),
                !b ||!b.gc))
                    return !1;
                a.u.z = h(unescape(b.gc()));
                s();
                var c = {
                    e: 35
                };
                c.i = E;
                c.ac = 1;
                c.bq = a.b.e;
                c.ar = S;
                c.t = a.u.y.startTime;
                c.de = a.u.y.rand;
                c.bo = a.u.y.moatClientSlicer1;
                c.bd = a.u.y.moatClientSlicer2;
                c.d = a.u.y.moatClientLevel1 + ":" + a.u.y.moatClientLevel2 + ":" + a.u.y.moatClientLevel3 + ":" + a.u.y.moatClientLevel4;
                c = {
                    fld: "bd.moatads.com",
                    qs: a.l.e(c)
                };
                b.sm(c);
                return !0
            };
            a.u.aa = function(b) {
                a.u.y = b;
                a.e.d(e, n, function(b) {
                    a.e.e(e, n, null, "mswe");
                    a.u.w = 1;
                    r()
                }, "mswe");
                a.e.d(e, q, function(b) {
                    a.e.e(e, q, null, "mswer");
                    a.u.w = 1;
                    r()
                },
                "mswer");
                a.e.d(g, "mousemove", function(b) {
                    a.e.e(g, "mousemove", null, "msmm");
                    a.u.x = 1;
                    r()
                }, "msmm");
                a.e.a(u, 10, 200, function() {
                    a.u.z = "0";
                    s()
                })
            }
        })(q);
        (function(a) {
            function l(b) {
                if (b.currentFocusState) {
                    var c, d;
                    if ("center" != b.config.name && (c = b.manager.getPixelByName("center")) && (d = c.viewstates[b.manager.getTargetViewState()])&&!d.inview)
                        b.skipWidthCheck=!0;
                    else {
                        var e, k, l;
                        "undefined" !== typeof b.manager.adNum && (e = a.a.al(b.manager.adNum)) && (k = a.i.a(e, !1), l = a.i.b(e, 1));
                        k && l && "center" != b.config.name ? b.killWidthCheck =
                        !0 : !k || "bottomLeft" != b.config.name && "topRight" != b.config.name ? b.skipWidthCheck=!1 : b.killWidthCheck=!0
                    }
                } else 
                    b.skipWidthCheck=!0
            }
            function m() {
                d && (a.h.k = function(b) {
                    return {
                        insertableFunc: a.h.l,
                        pixels: [{
                            name: "center",
                            id: "moatPx" + b.zr + "_" + r.ceil(1E6 * r.random()),
                            target: b.aa,
                            container: b.aa.parentNode,
                            position: {
                                top: "50%",
                                left: "50%"
                            },
                            onWidthCheck: l
                        }, {
                            name: "topLeft",
                            id: "moatPx" + b.zr + "_" + r.ceil(1E6 * r.random()),
                            target: b.aa,
                            container: b.aa.parentNode,
                            position: {
                                top: "0px",
                                left: "0px"
                            },
                            onWidthCheck: l
                        }, {
                            name: "topRight",
                            id: "moatPx" + b.zr + "_" + r.ceil(1E6 * r.random()),
                            target: b.aa,
                            container: b.aa.parentNode,
                            position: {
                                top: "0%",
                                left: "100%"
                            },
                            onWidthCheck: l
                        }, {
                            name: "bottomRight",
                            id: "moatPx" + b.zr + "_" + r.ceil(1E6 * r.random()),
                            target: b.aa,
                            container: b.aa.parentNode,
                            position: {
                                top: "100%",
                                left: "100%"
                            },
                            onWidthCheck: l
                        }, {
                            name: "bottomLeft",
                            id: "moatPx" + b.zr + "_" + r.ceil(1E6 * r.random()),
                            target: b.aa,
                            container: b.aa.parentNode,
                            position: {
                                top: "100%",
                                left: "0%"
                            },
                            onWidthCheck: l
                        }
                        ]
                    }
                }, a.h.n = function(b, c) {
                    var d = 0;
                    a.a.forEach(b.pixels, function(a) {
                        a.measurable &&
                        a.viewstates && a.viewstates[c] && d++
                    });
                    return 5 == d
                }, a.h.m = a.h.n, a.h.p = function(b, c) {
                    var d = 0, e, k;
                    "undefined" !== typeof b.adNum && (e = a.a.al(b.adNum)) && (k = a.i.a(e, !1));
                    e=!1;
                    k ? e = (k = b.getPixelByName("center")) && k.viewstates && k.viewstates[c] && k.viewstates[c].inview : (a.a.forEach(b.pixels, function(a) {
                        a.viewstates && a.viewstates[c] && a.viewstates[c].inview && d++
                    }), e = 3 <= d);
                    return e
                }, a.h.q = function(b, c) {
                    if (!b.inview)
                        return !1;
                    var d = 0;
                    a.a.forEach(b.pixels, function(a) {
                        a.viewstates && a.viewstates[c] && a.viewstates[c].inview &&
                        ("topLeft" != a.config.name && "bottomRight" != a.config.name || d++)
                    });
                    return 2 == d
                })
            }
            var e = ["1PX", "5PX"], d=!1, c=!1;
            a.p.f = function(a) {
                c || (d = .5 < r.random(), m(), c=!0);
                a.zMoatPMode = d ? e[1] : e[0];
                a.zMoatPMVer = "VER2";
                return a
            }
        })(q);
        if (q.a.t())
            return !1;
        var E = "NYTIMES2", ga = "nyt952824751", S = "4bb5684-clean", T = q.b.i, ba = q.b.k, y = "moatFound" + E, R = "__moat__" + E, ia = "", ha = q.b.l, J = q.b.a, Y = "afs.moatads.com", z, C = 0, U, V, L = {}, ja = 50, F = [], w = {}, M = [], u = {}, N=!1, ca = {
            15: "",
            12: "",
            6: "",
            7: ""
        };
        "string" === typeof q.b.protocol && (z = ("https:" ===
        q.b.protocol ? q.b.protocol : "http:") + "//v4.moatads.com");
        z || (z = "//v4.moatads.com");
        var A = q.b.c;
        q.s.a(A);
        var v = A[q.b.d];
        window[q.b.d] = v;
        q.t.b(A);
        q.a.l(J ? A.location.href : A.document.referrer) || q.a.l(window.location.href);
        var Q = q.a.z();
        if (!Q)
            return !1;
        q.f = Q;
        q.a.aa(Q);
        var B = q.a.q();
        (function(a) {
            C = v.h;
            v.h++;
            v.i[C]=!1;
            !1 === v.zs && (q.c.k(), v.zs=!0);
            q.c.l();
            q.e.d(window, "unload", function() {
                N || (q.c.o(), N=!0)
            }, !1);
            q.e.d(window, "beforeunload", function() {
                N || (q.c.o(), N=!0)
            }, !1);
            q.e.f(q.k.b, 100);
            q.a.f() && q.e.g($, 3E5);
            "undefined" === typeof a && (a = q.q.c(Q));
            var l = Q.parentNode;
            q.e.g(function() {
                !1 === v.i[C] && (q.l.b(11, a), $())
            }, 1E4);
            q.l.b(17, a);
            q.p.a(l, a);
            1 >= r.random() && (q.a.ab(Y), q.u.aa(a));
            v.alreadyRequestedContent || (v.alreadyRequestedContent=!0, fa(a))
        })()
    })(Date, Math)
} catch (e$$52) {
    var ct = (new Date).getTime();
    window["Moat#ETS"] || (window["Moat#ETS"] = ct);
    window["Moat#EMC"] || (window["Moat#EMC"] = 0);
    var et = ct - window["Moat#ETS"], hourElapsed = 36E5 <= et, msg = e$$52.name + " in closure: " + e$$52.message + ", stack=" + e$$52.stack;
    if (!hourElapsed &&
    10 > window["Moat#EMC"]) {
        window["Moat#EMC"]++;
        try {
            var pxSrc = "//v4.moatads.com/pixel.gif?e=24&d=data%3Adata%3Adata%3Adata&i=" + escape("NYTIMES2") + "&ac=1&k=" + escape(msg) + "&ar=" + escape("4bb5684-clean") + "&j=" + escape(document.referrer) + "&cs=" + (new Date).getTime(), px = new Image(1, 1);
            px.src = pxSrc
        } catch (e$$53) {}
    } else if (hourElapsed) {
        window["Moat#EMC"] = 1;
        window["Moat#ETS"] = ct;
        try {
            pxSrc = "//v4.moatads.com/pixel.gif?e=24&d=data%3Adata%3Adata%3Adata&i=" + escape("NYTIMES2") + "&ac=1&k=" + escape(msg) + "&ar=" + escape("4bb5684-clean") +
            "&j=" + escape(document.referrer) + "&cs=" + (new Date).getTime(), px = new Image(1, 1), px.src = pxSrc
        } catch (e$$54) {}
    }
};


