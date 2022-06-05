/*Copyright (c) 2011-2014 Moat Inc. All Rights Reserved.*/
try {
    (function(q, s) {
        function da() {
            for (var a = 0; a < A.length; a++)
                window.clearTimeout(A[a]);
            for (a = 0; a < F.length; a++)
                window.clearInterval(F[a]);
            for (var k in t)
                t.hasOwnProperty && t.hasOwnProperty(k) && t[k] && (window.clearTimeout(t[k].tid), t[k]=!1);
            A = [];
            F = []
        }
        function ea() {
            for (var a in r)
                r.hasOwnProperty(a) && n.c.i(r[a]);
            da()
        }
        var n = {};
        (function(a) {
            function k(a) {
                return a && a.document && a.location && a[b + d] && a[e + f]
            }
            function h(a) {
                if (null == a || k(a))
                    return !1;
                var b = a.length;
                return 1 === a.nodeType && b?!0 : "string" === typeof a ||
                g(a) || 0 === b || "number" === typeof b && 0 < b && b - 1 in a
            }
            function g(b) {
                return "[object Array]" === a.a.aa.toString.call(b)
            }
            a.a = {};
            a.a.a = 3E3;
            a.a.b = function() {
                var a = /Firefox\/(\d+)/.exec(navigator.userAgent);
                return a ? 14 < parseInt(a[1], 10) : !1
            }();
            a.a.c = function() {
                var b;
                b = /^(?:[a-z]+:\/\/|:?\/?\/)?(?:www\.)?([^\/:]*)/i;
                b=!a.b.a && a.b.b ? (b = a.b.b.match(b)) && b[1] ? b[1] : a.b.c.location.hostname : a.b.c.location.hostname;
                var c = b.match(/.*\.([^\.]*\..*)/i);
                return c && c[1] ? decodeURIComponent(c[1]) : decodeURIComponent(b)
            };
            a.a.d =
            function(a, b) {
                for (var c = [a], d = 1; d <= b; d++)
                    c.push(a + d), c.push(a - d);
                c = c[s.floor(s.random() * c.length)];
                d = s.floor(s.random() * c);
                return {
                    multiplier: c,
                    sample: 0 == d
                }
            };
            a.a.e = function(b, c) {
                var d = a.a.d(b, c);
                a.a.e = function() {
                    return d
                };
                return d
            };
            a.a.f = function() {
                var b = a.a.g();
                return 5 === b || 6 === b || 7 === b
            };
            a.a.g = function() {
                if (!navigator)
                    return null;
                var a = navigator.userAgent;
                return "Microsoft Internet Explorer" == navigator.appName ? parseInt(a.replace(/^.*MSIE (\d+).*$/, "$1"), 10) : "Netscape" == navigator.appName && (a = a.match(/Trident\/.*rv:(\d+)/)) ?
                parseInt(a[1], 10) : null
            };
            a.a.h = function(a, b) {
                for (var c = 0; c < a.length; c++)
                    if (a[c] === b)
                        return !0;
                return !1
            };
            a.a.i = function() {
                function a(b) {
                    b = b.match(/[\d]+/g);
                    b.length = 3;
                    return b.join(".")
                }
                var b=!1, c = "";
                if (navigator.plugins && navigator.plugins.length) {
                    var d = navigator.plugins["Shockwave Flash"];
                    d && (b=!0, d.description && (c = a(d.description)));
                    navigator.plugins["Shockwave Flash 2.0"] && (b=!0, c = "2.0.0.11")
                } else if (navigator.mimeTypes && navigator.mimeTypes.length)(b = (d = navigator.mimeTypes["application/x-shockwave-flash"]) &&
                d.enabledPlugin) && (c = a(d.enabledPlugin.description));
                else 
                    try {
                        d = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7"), b=!0, c = a(d.GetVariable("$version"))
                } catch (e) {
                    try {
                        d = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6"), b=!0, c = "6.0.21"
                    } catch (f) {
                        try {
                            d = new ActiveXObject("ShockwaveFlash.ShockwaveFlash"), b=!0, c = a(d.GetVariable("$version"))
                        } catch (g) {}
                    }
                }
                return b ? c : "0"
            };
            a.a.j = function(a) {
                return new q - a.de
            };
            a.a.k = function(a) {
                return a.replace(/^http:/, "").replace(/^\/\//, "").replace(/^www[^.]*\./, "").split("/")[0]
            };
            a.a.l = function(b, c) {
                if (!b)
                    return !1;
                var d = [b];
                a.a.forEach("number" === typeof c ? c : 50, function() {
                    if (b.parentNode && 1 == b.parentNode.nodeType)
                        b = b.parentNode, d.push(b);
                    else 
                        return !1
                });
                return d
            };
            a.a.m = function(b, c) {
                var d = "number" === typeof c ? c: 50, e = [], f = a.c.a(b);
                if (f)
                    e.push(f);
                else 
                    return e;
                a.a.forEach(d, function() {
                    var c = a.d.a(b, f);
                    return c && (f = a.c.a(c)) ? (e.push(f), !0) : !1
                });
                return e
            };
            a.a.n = function() {
                return null !== /iPad|iPhone|iPod|Silk|Kindle|Android|BlackBerry|PlayBook|BB10|Windows Phone/.exec(navigator.userAgent)
            };
            a.a.o = function() {
                return null !== /iPhone|iPod/.exec(navigator.userAgent)
            };
            a.a.p = function() {
                var a = navigator.userAgent;
                return a.match(/iPhone|iPod|iPad/)&&!a.match(/Safari/i)
            };
            a.a.q = function() {
                var a = window;
                try {
                    return "undefined" !== typeof a.external && "undefined" !== typeof a.external.InPrivateFilteringEnabled && a.external.InPrivateFilteringEnabled() || "undefined" !== typeof a.external && "undefined" !== typeof a.external.msTrackingProtectionEnabled && a.external.msTrackingProtectionEnabled() || "undefined" !== typeof a._gaUserPrefs &&
                    a._gaUserPrefs.ioo && a._gaUserPrefs.ioo() || "undefined" !== typeof navigator.doNotTrack && ("yes" === navigator.doNotTrack ||!0 === navigator.doNotTrack) || "undefined" !== typeof a._gaUserPrefs&&!0 === a._gaUserPrefs
                } catch (b) {
                    return !1
                }
            };
            a.a.getAttribute = function(a, b) {
                return a[b] || a.getAttribute(b)
            };
            a.a.r = function(a) {
                var b = 0, c;
                for (c in a)
                    a.hasOwnProperty(c) && (b += 1);
                return b
            };
            var c = [function(a) {
                if (!a || "IFRAME" !== a.nodeName)
                    return !1;
                var b = a.offsetHeight;
                return isNaN(b) || 15 < b || "google_conversion_frame" !== a.name?!1 : !0
            }
            ];
            a.a.s = function(b, d) {
                return !a.a.t(b) || a.a.filter(c, function(a) {
                    return a(b)
                }).length ||!0 === b[w]?!1 : !0
            };
            a.a.t = function(b) {
                return b.offsetWidth * b.offsetHeight >= a.a.a
            };
            a.a.u = function(a, b) {
                if ("string" !== typeof a ||!b ||!b.insertBefore ||!b.ownerDocument)
                    return !1;
                var c = b.ownerDocument.createElement("script");
                c.type = "text/javascript";
                c.async=!0;
                b.insertBefore(c, b.childNodes[0]);
                c.src = a;
                return !0
            };
            a.a.v = function() {
                var a;
                return function() {
                    if (!a)
                        a: {
                        for (var b = document.getElementsByTagName("script"), c = b.length - 1; - 1 <
                        c; c--)
                            if ((a = b[c]) && a.src && a.src.indexOf&&-1 !== a.src.indexOf(ja + "/moatad.js") && ("undefined" === typeof a[w] ||!0 !== a[w])) {
                                a[w]=!0;
                                break a
                            }
                            a = void 0
                    }
                    return a ? (a[w]=!0, a) : null
                }
            }();
            a.a.w = function(a) {
                try {
                    return - 1 !== (a.src || a.getAttribute("src")).indexOf("psd=1")
                } catch (b) {
                    return !1
                }
            };
            a.a.x = function(a) {
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
            a.a.y =
            function(a, b, c) {
                "undefined" !== typeof c && (a[b] = c)
            };
            a.a.filter = function(a, b) {
                for (var c = [], d = 0; d < a.length; d++)
                    b(a[d]) && c.push(a[d]);
                return c
            };
            a.a.z = function(a, b) {
                for (var c = [], d = 0; d < b.length; d++)
                    c.push(a(b[d]));
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
            var b = "ale", d = "rt", e = "setInte", f = "rval";
            a.a.aa = {};
            a.a.ab = function(a, b) {
                if (a && b && b.childNodes) {
                    var c = b.childNodes;
                    0 < c.length ? b.insertBefore(a, c[0]) : b.appendChild(a)
                }
            };
            a.a.ac = function(b, c) {
                if (!b ||!c)
                    return !1;
                var d = a.a.ad(c);
                if (!d)
                    return !1;
                a.a.hasChildNodes(d) ? d.insertBefore(b, d.childNodes[0]) : d.appendChild(b);
                return d
            };
            a.a.hasChildNodes = function(a) {
                return a && a.childNodes && 0 < a.childNodes.length
            };
            a.a.ad = function(b) {
                if (!b)
                    return !1;
                if ("OBJECT" !== b.nodeName && "EMBED" !== b.nodeName)
                    return b;
                b = a.a.l(b);
                var c=!1;
                a.a.forEach(b, function(a) {
                    if (a && "OBJECT" !==
                    a.nodeName && "EMBED" !== a.nodeName)
                        return c = a, !1
                });
                return c
            };
            a.a.ae = function(a, b) {
                if ("undefined" === typeof a)
                    return !1;
                for (var c = 0, d = b.length; c < d; c++)
                    if ("string" == typeof b[c] && (a = a[b[c]], "undefined" === typeof a))
                        return !1;
                return a
            };
            a.a.af = function(a) {
                return encodeURIComponent(a.moatClientLevel1 + ":" + a.moatClientLevel2 + ":" + a.moatClientLevel3 + ":" + a.moatClientLevel4)
            };
            a.a.ag = function(a) {
                return r && "undefined" !== typeof a && r[a] ? r[a] : !1
            };
            a.a.ah = g;
            a.a.ai = k;
            a.a.aj = h;
            a.a.forEach = function(a, b, c, d) {
                var e, f = typeof a;
                if (a)
                    if ("function" === f)
                        for (e in a) {
                            if ("prototype" != e && "length" != e && "name" != e && (d ||!a.hasOwnProperty || a.hasOwnProperty(e)) && (f = b.call(c, a[e], e), "boolean" === typeof f&&!f))
                                break
                        } else if ("number" === f)
                            for (e = 0; e < a && (f = b.call(c, a, e), "boolean" !== typeof f || f); e++);
                        else if ("function" === typeof a.every)
                            a.every(function(a, d, e) {
                                a = b.call(c, a, d);
                                return !("boolean" === typeof a&&!a)
                            });
                        else if (h(a))
                            for (e = 0; e < a.length && (f = b.call(c, a[e], e), "boolean" !== typeof f || f); e++);
                        else 
                            for (e in a)
                                if (d || a.hasOwnProperty(e))
                                    if (f = b.call(c,
                                    a[e], e), "boolean" === typeof f&&!f)
                                        break;
                return a
            };
            a.a.ak = function(b) {
                return b && a.a.filter(b.children, function(a) {
                    return 1 === a.nodeType
                })
            }
        })(n);
        (function(a) {
            function k(a) {
                try {
                    var c = typeof a.location.toString;
                    if ("undefined" === c || "unknown" === c)
                        return !0;
                    var b = typeof a.document;
                    return "undefined" === b || "unknown" === b?!0 : !1
                } catch (d) {
                    return !0
                }
            }
            a.b = {};
            a.b.d = "MoatSuperV5";
            a.b.e = 1;
            a.b.f = 0 < Object.prototype.toString.call(window.HTMLElement).indexOf("Constructor");
            a.b.g =- 1 !== navigator.userAgent.toLowerCase().indexOf("firefox");
            a.b.h =- 1 !== navigator.userAgent.indexOf("MSIE");
            a.b.i = (new q).getTime();
            a.b.protocol = function() {
                var a;
                try {
                    a = window.location.protocol
                } catch (c) {
                    a = document.createElement("a"), a.href = "", a = a.protocol
                }
                return a
            }();
            a.b.j = ("https:" === a.b.protocol ? "z" : "js") + ".moatads.com";
            a.b.k = window != window.parent;
            var h = k(window.parent);
            a.b.l = a.b.k&&!h;
            a.b.a = h?!1 : !k(window.top);
            a.b.c = a.b.a ? window.top : a.b.l ? window.parent : window;
            a.b.b = a.b.c.document.referrer
        })(n);
        (function(a) {
            function k(a) {
                return function() {
                    var c=!1;
                    return function(b) {
                        try {
                            return a(b)
                        } catch (d) {
                            if (!c) {
                                c =
                                !0;
                                b = (new q).getTime();
                                window["Moat#ETS"] || (window["Moat#ETS"] = b);
                                window["Moat#EMC"] || (window["Moat#EMC"] = 0);
                                var e = 36E5 <= b - window["Moat#ETS"], f = d.name + " in closure: " + d.message + ", stack=" + d.stack;
                                if (!e && 10 > window["Moat#EMC"]) {
                                    window["Moat#EMC"]++;
                                    try {
                                        var m = "//tu.moatads.com/pixel.gif?e=24&d=data%3Adata%3Adata%3Adata&i=" + escape("TURNER1") + "&ac=1&k=" + escape(f) + "&ar=" + escape("a8a845a-clean") + "&j=" + escape(document.referrer) + "&cs=" + (new q).getTime(), l = new Image(1, 1);
                                        l.src = m
                                    } catch (B) {}
                                } else if (e) {
                                    window["Moat#EMC"] =
                                    1;
                                    window["Moat#ETS"] = b;
                                    try {
                                        m = "//tu.moatads.com/pixel.gif?e=24&d=data%3Adata%3Adata%3Adata&i=" + escape("TURNER1") + "&ac=1&k=" + escape(f) + "&ar=" + escape("a8a845a-clean") + "&j=" + escape(document.referrer) + "&cs=" + (new q).getTime(), l = new Image(1, 1), l.src = m
                                    } catch (T) {}
                                }
                            }
                        }
                    }
                }()
            }
            a.e = {};
            var h = {};
            a.e.a = h;
            a.e.b = function(g, c) {
                if (!g || "string" !== typeof c ||!g[c] || g == window)
                    return !1;
                if ("string" === typeof g.nodeName && ("OBJECT" === g.nodeName || "EMBED" === g.nodeName)) {
                    var b = a && a.f && a.f[c];
                    if (b && b !== g[c])
                        return b
                }
                return !1
            };
            a.e.c = function(g, c, b, d) {
                var e, f = k(b), m;
                g.addEventListener ? (b = "addEventListener", e = "") : (b = "attachEvent", e = "on");
                if (m = a.e.b(g, b))
                    try {
                        m.call(g, e + c, f, !1)
                } catch (l) {
                    g[b](e + c, f, !1)
                } else 
                    g[b](e + c, f, !1);
                !1 !== d && (h[c + d] = f)
            };
            a.e.d = function(g, c, b, d) {
                var e, f = c + d, m;
                if (!g)
                    return delete h[f], !1;
                b=!1 !== d ? h[f] : b;
                g.removeEventListener ? (d = "removeEventListener", e = "") : (d = "detachEvent", e = "on");
                if (m = a.e.b(g, d))
                    try {
                        m.call(g, e + c, b, !1)
                } catch (l) {
                    g[d](e + c, b, !1)
                } else 
                    g[d](e + c, b, !1);
                delete h[f]
            };
            a.e.e = function(a, c) {
                a = k(a);
                var b =
                setInterval(a, c);
                F.push(b);
                return b
            };
            a.e.f = function(a, c) {
                a = k(a);
                var b = setTimeout(a, c);
                A.push(b);
                return b
            };
            a.e.g = function(g, c, b, d) {
                if (!d)
                    return !1;
                d += "";
                t[d] && window.clearTimeout(t[d].tid);
                t[d] = {};
                t[d].callback = k(g);
                t[d].params = c;
                t[d].interval = b;
                t[d].tid = a.e.f(a.e.h(d), b);
                callbacks = null
            };
            a.e.h = function(g) {
                return function() {
                    if (!t ||!t[g])
                        return !1;
                    var c = t[g].callback(t[g].params);
                    if ("boolean" === typeof c&&!1 === c)
                        return window.clearTimeout(t[g].tid), t[g]=!1;
                    t[g].tid = a.e.f(a.e.h(g), t[g].interval);
                    time = g =
                    null
                }
            };
            a.e.i = function() {
                return t
            };
            a.e.j = function(g, c, b, d) {
                var e = 0, f = function() {
                    e += 1;
                    !0 !== g() && (e < c ? a.e.f(f, b) : "function" === typeof d && d())
                };
                f()
            };
            a.e.k = k
        })(n);
        (function(a) {
            function k(a, c) {
                for (var e = [], f = 0; f < c.length; f++)
                    e.push(a(c[f]));
                return e
            }
            function h(b) {
                var c, e = a.c.b(b.el, b.win), e = k(function(b) {
                    return new a.g.g(b)
                }, e);
                e.unshift(b);
                var f = e.length;
                b = new a.g.g(b.el, a.b.c);
                for (var g = 0; g < f; g++) {
                    var l = e[g];
                    0 === g ? c = l : (c.changeReferenceFrame(l), b.changeReferenceFrame(l));
                    l = l.getViewportRect(g < f - 1 ? e[g +
                    1] : !1);
                    c = a.g.i(c, l)
                }
                return {
                    visibleRect: c,
                    cumulRect: b,
                    parentArea: e[e.length - 1].calcArea()
                }
            }
            function g(a, c, e, f) {
                a = s.max(a, e);
                c = s.min(c, f);
                return c > a ? [a, c] : [0, 0]
            }
            a.g = {};
            var c = a.b.a;
            a.g.a = .5;
            a.g.b = .3;
            a.g.c = 236425;
            a.g.d = function(b, c) {
                var e;
                e = new a.g.g(b.aa, c);
                var f = e.height, g = e.width, l = e.calcArea();
                if (0 === l)
                    e = {
                        area: l,
                        percv: 0
                    };
                else {
                    var B = h(e), T = B.visibleRect.calcArea(), k = T / l, G;
                    b:
                    {
                        var v = B.cumulRect, J = B.cumulRect.getViewportRect();
                        if (0 > v.top && 0 < v.bottom)
                            G =- v.top;
                        else if (0 <= v.top && v.top <= J.height)
                            G = 0;
                        else {
                            G =
                            {
                                yMin: - 1,
                                yMax: - 1
                            };
                            break b
                        }
                        if (0 <= v.bottom && v.bottom <= J.height)
                            v = v.height;
                        else if (v.bottom > J.height && v.top < J.height)
                            v = v.height - (v.bottom - J.height);
                        else {
                            G = {
                                yMin: - 1,
                                yMax: - 1
                            };
                            break b
                        }
                        G = {
                            yMin: G,
                            yMax: v
                        }
                    }
                    e = {
                        area: l,
                        visibleArea: T,
                        percv: k,
                        yMinMax: G,
                        elGeo: {
                            elHeight: f,
                            elWidth: g,
                            foldTop: B.cumulRect.top,
                            totalArea: B.parentArea
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
                return b.area >= a.g.c ? (c.viewstats.isBigAd=!0, a.g.b) :
                a.g.a
            };
            a.g.f = function() {
                return c
            };
            a.g.h = function(a) {
                var c, e, f, g = 0, l = 0;
                try {
                    c = a.document, e = c.documentElement, f = c.body
                } catch (B) {
                    return {
                        width: g,
                        height: l,
                        left: 0,
                        right: g,
                        top: 0,
                        bottom: l
                    }
                }
                "undefined" !== typeof a.innerWidth ? (g = a.innerWidth, l = a.innerHeight) : "CSS1Compat" === c.compatMode && 5 !== c.documentMode ||!f || "undefined" === typeof f.clientWidth ? e && "undefined" !== typeof e.clientWidth && (g = e.clientWidth, l = e.clientHeight) : (g = f.clientWidth, l = f.clientHeight);
                return {
                    width: g,
                    height: l,
                    left: 0,
                    right: g,
                    top: 0,
                    bottom: l
                }
            };
            a.g.g =
            function(b, c, e) {
                this.rect = e || b.getBoundingClientRect();
                e = ["left", "right", "top", "bottom"];
                for (var f = 0; f < e.length; f++) {
                    var g = e[f];
                    this[g] = this.rect[g]
                }
                this.width = this.right - this.left;
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
                    var c = a.g.h(this.win);
                    b && (b.width <
                    c.width && (c.width = b.width, c.right = c.left + c.width), b.height < c.height && (c.height = b.height, c.bottom = c.top + c.height));
                    return c
                }
            };
            a.g.i = function(b, c) {
                var e = g(b.left, b.right, c.left, c.right), f = g(b.top, b.bottom, c.top, c.bottom);
                return new a.g.g(void 0, void 0, {
                    left : e[0], right : e[1], top : f[0], bottom : f[1]
                })
            };
            a.g.j = function(a, c) {
                c || (c = window);
                var e = c.document.documentElement, f = c.document.body;
                return "left" === a ? c.pageXOffset || e && e.scrollLeft || f && f.scrollLeft : c.pageYOffset || e && e.scrollTop || f && f.scrollTop
            }
        })(n);
        (function(a) {
            function k(a,
            b, c) {
                a.IR5.MIN[c] = s.min(b, a.IR5.MIN[c]) || b || 1;
                a.IR5.MAX[c] = s.max(b, a.IR5.MAX[c]) || b
            }
            function h(a, b) {
                b.be = s.max("undefined" !== typeof b.be ? b.be : 0, a - b.bf);
                "undefined" === typeof b.by && 500 <= b.be && (b.by = b.bk)
            }
            function g(b) {
                b.az === a.h.a.d.a ? b.az = a.h.a.d.b : b.az === a.h.a.d.b && (b.az = a.h.a.d.a)
            }
            function c(b) {
                b.ba === a.h.a.d.a ? b.ba = a.h.a.d.b : b.ba === a.h.a.d.b && (b.ba = a.h.a.d.a)
            }
            function b(b) {
                b.ax === a.h.a.b.a ? b.ax = a.h.a.b.b : b.ax === a.h.a.b.b && (b.ax = a.h.a.b.a)
            }
            function d(b) {
                b.ay === a.h.a.c.a ? b.ay = a.h.a.c.b : b.ay === a.h.a.c.b &&
                (b.ay = a.h.a.c.a)
            }
            function e(a, b) {
                "undefined" === typeof b.bk && (b.bk = 0);
                b.bk += a - b.bo;
                b.bo = a
            }
            function f(a, b) {
                "undefined" === typeof b.bl && (b.bl = 0);
                b.bl += a - b.bp;
                b.bp = a
            }
            function m(a, b) {
                "undefined" === typeof b.bg && (b.bg = 0);
                "undefined" === typeof b.bc && (b.bc = 0);
                b.bu = a - b.bs;
                b.bu > b.bc && (b.bc = b.bu);
                b.bg += a - b.bq;
                500 <= b.bc && "undefined" === typeof b.bw && (b.bk += a - b.bo, b.bw = b.bk);
                b.bq = a
            }
            function l(a, b) {
                "undefined" === typeof b.bh && (b.bh = 0);
                "undefined" === typeof b.bd && (b.bd = 0);
                b.bv = a - b.bt;
                b.bv > b.bd && (b.bd = b.bv);
                b.bh += a - b.br;
                500 <= b.bd && "undefined" === typeof b.bx && (b.bl += a - b.bp, b.bx = b.bl);
                b.br = a
            }
            a.h = {};
            a.h.a = {};
            a.h.a.a = {};
            a.h.a.a.a = 0;
            a.h.a.a.b = 1;
            a.h.a.b = {};
            a.h.a.b.a = 0;
            a.h.a.b.b = 1;
            a.h.a.c = {};
            a.h.a.c.a = 0;
            a.h.a.c.b = 1;
            a.h.a.d = {};
            a.h.a.d.a = 0;
            a.h.a.d.b = 1;
            a.h.a.e = {};
            a.h.a.e.a = 0;
            a.h.a.e.b = 1;
            a.h.a.f = {};
            a.h.a.f.a = 0;
            a.h.a.f.b = 1;
            a.h.a.f.c = 2;
            a.h.b = function(a) {
                k(a, a.aj, "x");
                k(a, a.ak, "y");
                a.IR5.AREA = (a.IR5.MAX.x - a.IR5.MIN.x) * (a.IR5.MAX.y - a.IR5.MIN.y)
            };
            a.h.c = function(b) {
                function c() {
                    500 <= (new q).getTime() - U && (a.h.d({
                        type: "park"
                    }, b), clearInterval(e),
                    b.au = a.h.a.a.a)
                }
                var d = b.au;
                if (d === a.h.a.a.a) {
                    U = (new q).getTime();
                    var e = a.e.e(c, 50);
                    b.au = a.h.a.a.b
                } else 
                    d === a.h.a.a.b && (U = (new q).getTime())
            };
            a.h.e = function(b) {
                function c() {
                    3E3 <= (new q).getTime() - V && (a.h.f({
                        type: "park"
                    }, b), clearInterval(e), b.av = a.h.a.a.a)
                }
                var d = b.av;
                if (d === a.h.a.a.a) {
                    V = (new q).getTime();
                    var e = a.e.e(c, 50);
                    b.av = a.h.a.a.b
                } else 
                    d === a.h.a.a.b && (V = (new q).getTime())
            };
            a.h.g = function(b, c) {
                var d = b.type;
                if (c.az === a.h.a.d.a) {
                    if ("mouseover" === d || "mousemove" === d)
                        c.bo = (new q).getTime(), g(c)
                } else if (c.az ===
                a.h.a.d.b) {
                    "mousemove" === d && e((new q).getTime(), c);
                    if ("mouseout" === d || "blur" === d)
                        e((new q).getTime(), c), g(c);
                    "scooper" === d && e((new q).getTime(), c)
                }
            };
            a.h.h = function(b, d) {
                var e = b.type;
                if (d.ba === a.h.a.d.a) {
                    if ("mouseover" === e || "mousemove" === e)
                        d.bp = (new q).getTime(), c(d)
                } else if (d.ba === a.h.a.d.b) {
                    "mousemove" === e && f((new q).getTime(), d);
                    if ("mouseout" === e || "blur" === e)
                        f((new q).getTime(), d), c(d);
                    "scooper" === e && f((new q).getTime(), d)
                }
            };
            a.h.d = function(c, d) {
                if (2 != d.an) {
                    var e = c.type;
                    if (d.ax === a.h.a.b.a) {
                        if ("mouseover" ===
                        e || "mousemove" === e)
                            d.bq = (new q).getTime(), d.bs = (new q).getTime(), b(d)
                        } else 
                            d.ax === a.h.a.b.b && ("mousemove" !== e && "mouseout" !== e || m((new q).getTime(), d), "park" === e && m((new q).getTime() - 500, d), "mouseout" !== e && "park" !== e || b(d))
                }
            };
            a.h.f = function(b, c) {
                if (2 != c.an) {
                    var e = b.type;
                    if (c.ay === a.h.a.c.a) {
                        if ("mouseover" === e || "mousemove" === e)
                            c.br = (new q).getTime(), c.bt = (new q).getTime(), d(c)
                        } else 
                            c.ay === a.h.a.c.b && ("mousemove" !== e && "mouseout" !== e || l((new q).getTime(), c), "park" === e && l((new q).getTime() - 3E3, c), "mouseout" !==
                            e && "park" !== e || d(c))
                }
            };
            a.h.i = function(b, c) {
                var d = b.type;
                if (c.bb == a.h.a.e.a) {
                    if ("mouseover" == d || "mousemove" == d)
                        c.bf = (new q).getTime(), c.bb = a.h.a.e.b
                } else 
                    c.bb == a.h.a.e.b && ("mouseout" == d ? (h((new q).getTime(), c), c.bb = a.h.a.e.a) : "mousemove" != d && "scooper" != d || h((new q).getTime(), c))
            }
        })(n);
        (function(a) {
            function k(b) {
                if (!a.focus.pageIsPrerendered()) {
                    a.e.d(document, m, k, "pr");
                    for (var c in r)
                        r.hasOwnProperty(c) && a.c.c(r[c])
                }
            }
            function h(a) {
                "undefined" == typeof d && (d = a)
            }
            a.focus = {};
            var g = navigator.userAgent, g =
            - 1 < g.indexOf("Safari")&&-1 == g.indexOf("Chrome")&&-1 == g.indexOf("Chromium")&&!C, c = (eval("/*@cc_on!@*/false") || document.documentMode)&&!C, b = "undefined" !== typeof document.hasFocus, d, e = {
                visible: 0,
                hidden: 1,
                prerender: 2
            }, f, m, l, B;
            "undefined" !== typeof document.hidden ? (f = "hidden", m = "visibilitychange") : "undefined" !== typeof document.mozHidden ? (f = "mozHidden", m = "mozvisibilitychange") : "undefined" !== typeof document.msHidden ? (f = "msHidden", m = "msvisibilitychange") : "undefined" !== typeof document.webkitHidden && (f = "webkitHidden",
            m = "webkitvisibilitychange");
            for (var n = ["v", "mozV", "msV", "webkitV"], s = 0; s < n.length; s++) {
                var G = n[s] + "isibilityState";
                if ("undefined" !== typeof document[G]) {
                    l = G;
                    break
                }
            }
            B = "undefined" !== typeof f;
            var v, J;
            "undefined" !== typeof window.requestAnimationFrame ? v = "requestAnimationFrame" : "undefined" !== typeof window.webkitRequestAnimationFrame && (v = "webkitRequestAnimationFrame");
            J = g && "string" == typeof v&&!B;
            var fa = (new q).getTime();
            J && function R() {
                fa = (new q).getTime();
                window[v](R)
            }();
            a.focus.getFocusMethod = function() {
                return d
            };
            a.focus.visibilitychange = m;
            a.focus.setFocusListeners = function() {};
            a.focus.getQueryString = function(a) {
                a = {};
                a.em = d;
                t && (a.eo = 1);
                "undefined" != typeof l && (a.en = e[document[l]]);
                return a
            };
            a.focus.supportsPageVisAPI = function() {
                return B
            };
            a.focus.pageIsVisible = function() {
                if (a.focus.supportsPageVisAPI())
                    return h(0), !document[f];
                if (J)
                    return h(1), 100 > (new q).getTime() - fa;
                if (c && b)
                    return h(2), document.hasFocus();
                h(3);
                return !0
            };
            a.focus.pageIsPrerendered = function() {
                return "undefined" !== typeof l ? "prerender" == document[l] :
                !1
            };
            a.focus.pageIsVisible();
            a.focus.pageIsPrerendered() && a.e.c(document, m, k, "pr");
            var t = a.focus.pageIsPrerendered()
        })(n);
        (function(a) {
            function k(b, c, d, e, f) {
                ("remove" === f ? a.e.d : a.e.c)(b, c, function(c) {
                    c = c || this.event;
                    var e = c.currentTarget || b, f = r[e[K]];
                    if (f) {
                        var g;
                        g = c;
                        var m = e.getBoundingClientRect();
                        g =- 1 != g.type.indexOf("touch") ? {
                            x: parseInt(g.changedTouches[0].clientX - m.left, 10),
                            y: parseInt(g.changedTouches[0].clientY - m.top, 10)
                        } : {
                            x: parseInt(g.clientX - m.left, 10),
                            y: parseInt(g.clientY - m.top, 10)
                        };
                        f.aj = g.x;
                        f.ak = g.y;
                        f.dm || (f.cb = 2 == a.focus.getFocusMethod() ? f.counters.laxDwell.tCur : f.counters.strictDwell.tCur, f.dm = 1);
                        d.call(b, c, e, f)
                    }
                }, e)
            }
            function h(a, d, e) {
                k(a, "click", B, d, e);
                k(a, "mousedown", c, d, e);
                C ? k(a, "touchstart", b, d, e) : (k(a, "mousemove", g, d, e), k(a, "mouseover", m, d, e), k(a, "mouseout", l, d, e))
            }
            function g(b, c, d) {
                C || d.aj === d.al && d.ak === d.am || (a.h.d(b, d), a.h.f(b, d), a.h.g(b, d), a.h.i(b, d), a.h.h(b, d), a.h.b(d), a.h.c(d), a.h.e(d), 0 === d.ar.length && (d.ai = L(d)), d.ar.push(d.aj), d.as.push(d.ak), d.at.push(a.a.j(d)), d.al =
                d.aj, d.am = d.ak);
                d.ai !== L(d) && 1 < d.ar.length && n(d, "onMouseMove")
            }
            function c(b, c, d) {
                b = {
                    e: 2
                };
                b.q = d.aq[2]++;
                b.x = d.aj;
                b.y = d.ak;
                a.j.a(d, b)
            }
            function b(b, c, d) {
                b = {
                    e: 23
                };
                b.q = d.aq[23]++;
                b.x = d.aj;
                b.y = d.ak;
                c = (new q).getTime();
                if ("undefined" === typeof d.ct)
                    d.ct = c;
                else {
                    var e = c - d.ct;
                    d.ct = c;
                    d.cu = s.min(d.cu, e) || e
                }
                b.bz = void 0;
                a.j.a(d, b)
            }
            function d(b, c, d) {
                if (2 == b.an) {
                    var g = c.e, m = b.ck;
                    m == a.h.a.f.a && 6 === g ? (e(b, 0), b.cl = a.a.j(b), b.ck = a.h.a.f.b) : m == a.h.a.f.b ? 22 === g ? (f(b, c), e(b, d), b.cl = a.a.j(b)) : 7 === g && (1E3 < a.a.j(b) - b.cl ? (clearTimeout(b.cm),
                    c.e = 22, f(b, c), b.cn = 0, b.ck = a.h.a.f.a) : b.ck = a.h.a.f.c) : m == a.h.a.f.c && (6 == g ? (1E3 < a.a.j(b) - b.cl && (clearTimeout(b.cm), b.cn = 0, b.cl = a.a.j(b), e(b, 0)), b.ck = a.h.a.f.b) : 22 == g && (f(b, c), b.ck = a.h.a.f.a, b.cn = 0))
                }
            }
            function e(b, c) {
                var e = 5 > b.cn ? 1E3: 2 * c, f = {
                    e: 22
                };
                b.cm = a.e.f(function() {
                    d(b, f, e)
                }, e)
            }
            function f(b, c) {
                c.q = b.aq[c.e]++;
                c.m = a.a.j(b);
                b.cl = c.m;
                a.j.a(b, c);
                b.cn++
            }
            function m(b, c, e) {
                a.h.d(b, e);
                a.h.f(b, e);
                a.h.g(b, e);
                a.h.i(b, e);
                a.h.h(b, e);
                d(e, {
                    e: 6
                }, 0)
            }
            function l(b, c, e) {
                a.h.d(b, e);
                a.h.f(b, e);
                a.h.g(b, e);
                a.h.i(b, e);
                a.h.h(b,
                e);
                d(e, {
                    e: 7
                }, 0)
            }
            function B(b, c, d) {
                b = {
                    e: 3
                };
                b.q = d.aq[3]++;
                b.x = d.aj;
                b.y = d.ak;
                a.j.a(d, b)
            }
            function n(b, c) {
                b.ai = L(b);
                var d = {
                    e: 1
                };
                d.q = b.aq[1]++;
                d.x = b.ar.join("a");
                d.y = b.as.join("a");
                for (var e = a.a.j(b), f = b.at, g = [], m = 0; m < f.length; m++)
                    isNaN(f[m]) || g.push(s.abs(f[m] - e));
                d.c = g.join("a");
                d.m = e;
                a.j.a(b, d);
                b.ar = [];
                b.as = [];
                b.at = []
            }
            function L(b) {
                return s.floor(a.a.j(b) / 1E3)
            }
            a.i = {};
            a.i.a = function(a, b) {
                if (!0 === a.isSkin)
                    for (var c = 0; c < a.elements.length; c++)
                        h(a.elements[c], a.zr + "-" + c, b);
                else 
                    h(a.aa, a.zr, b)
            };
            a.i.b = function(b) {
                for (var c in r)
                    r.hasOwnProperty(c) &&
                    (b = r[c]) && (a.h.g({
                        type: "scooper"
                    }, b), a.h.i({
                        type: "scooper"
                    }, b), a.h.h({
                        type: "scooper"
                    }, b), 1 < b.ar.length && b.ai !== L(b) && n(b, "scooper"))
            }
        })(n);
        (function(a) {
            a.k = {};
            var k = "undefined" !== typeof window.devicePixelRatio, h = k && s.round(1E3 * window.devicePixelRatio), g = a.b.a && k && "undefined" !== typeof window.innerHeight && "undefined" !== typeof window.outerHeight && s.round(window.devicePixelRatio * (a.b.c.outerHeight - a.b.c.innerHeight)), c = "undefined" !== typeof window.mozInnerScreenX && "undefined" !== typeof window.screenX;
            a.k.a =
            function(b, c) {
                var e, f, g = {
                    isVisible: !1,
                    isFullyVisible: !1
                };
                try {
                    var l = b.aa.getBoundingClientRect(), h = c || b.WINDOW || a.c.a(b.aa), n = a.g.h(h), q = a.g.i(l, n), s = h.mozInnerScreenX, v = h.mozInnerScreenY, r = window.screenX, t = window.screenY, u = a.g.i({
                        left: q.left + s,
                        right: q.right + s,
                        top: q.top + v,
                        bottom: q.bottom + v
                    }, {
                        left: r,
                        right: r + window.outerWidth,
                        top: t + 117 / (k ? window.devicePixelRatio : 1),
                        bottom: t + window.outerHeight
                    }), O = l.width * l.height;
                    e = {
                        area: O,
                        percv: (u.right - u.left) * (u.bottom - u.top) / O
                    };
                    f = a.g.e(e, b)
                } catch (R) {}
                "undefined" !==
                typeof e && "undefined" !== typeof f && (g.isVisible = e.percv >= f, g.isFullyVisible = 1 == e.percv);
                return g
            };
            a.k.b = function() {
                return c
            };
            a.k.c = function() {
                var a = {};
                a.dl = Number(c);
                "number" !== typeof h || isNaN(h) || (a.dm = h);
                "number" !== typeof g || isNaN(g) || (a.dn = g);
                return a
            }
        })(n);
        (function(a) {
            function k(a) {
                var c = 0, e;
                return function() {
                    var f = (new q).getTime();
                    150 < f - c && (e = a.apply(this, arguments), c = f);
                    return e
                }
            }
            function h(b, c, e, f) {
                var m = [], l = 0, k = 0, h = 0, n = 0, r = 0, v = 0, t = (new q).getTime(), u=!1, w=!1, O=!1, R=!1, ba, y, C, E = 0, F = 0, I = 0,
                A = 0, N=!1, S = a.b.a, W;
                if (0 === f)
                    var x = "as", z = "ag", X = "an", Y = "ck", Z = "kw", $ = "ah", aa = "aj", U = "pg", V = "pf", K = "gi", P = "gf", Q = "gg", N=!0;
                else 
                    1 === f ? (x = "cc", z = "bw", X = "bx", Y = "ci", Z = "jz", $ = "bu", aa = "dj") : 2 === f ? (x = "cg", z = "ce", X = "cf", Y = "cj", Z = "ts", $ = "ah", aa = "dk", K = "gj", P = "gb", Q = "ge") : 3 === f && (x = "cg", z = "ce", X = "cf", Y = "cj", Z = "ts", $ = "ah", aa = "dk", K = "gi", P = "gf", Q = "gg");
                this.addListener = function(a) {
                    m.push(a)
                };
                this.hadOTS = function() {
                    return O
                };
                this.hadVideo2SecOTS = function() {
                    return _hadVideo2SecOts
                };
                this.getInViewTime = function() {
                    return l
                };
                this.visible = function() {
                    return u
                };
                this.fullyVisible = function() {
                    return w
                };
                this.getFullInviewTimeTotal = function() {
                    return k
                };
                this.update = function(h) {
                    var n = l || 0, T = k || 0, L=!1, D = b(h);
                    h.elementRect = D.rect;
                    var x = D.isVisible, z = D.isFullyVisible, H = c(h), A = (new q).getTime(), M = s.max(s.min(A - t, 1E3), 0);
                    t = A;
                    A=!e || a.focus.pageIsVisible(2 === f || 3 === f);
                    x = x && A&&!H;
                    z = z && A&&!H;
                    if (x && u)
                        l += M, r += M;
                    else if (x || u)
                        H = s.round(M / 2), l += H, r += H;
                    if (z && w)
                        k += M, v += M;
                    else if (z || w)
                        H = s.round(M / 2), k += H, v += H;
                    !O && 1E3 <= r && (L = O=!0, y = l, this.timeToView =
                    ba = h.counters.query()[$]);
                    !R && 1E3 <= v && (L = R=!0);
                    "undefined" === typeof C && (1E3 >= h.counters.query().bu ? x && (C=!0) : C=!1);
                    if ((h.FOLDMEASURABLE = S) && "undefined" === typeof W && 2 !== f && 3 !== f && D.elGeo) {
                        var H = g().y + D.elGeo.foldTop, K = D.elGeo.threshold * D.elGeo.elHeight;
                        _isBTFCheck = H > a.l.a() - K;
                        0 < D.elGeo.totalArea && (W = _isBTFCheck, h.BELOWTHEFOLDAD = W)
                    }
                    h = D.yMinMax;
                    N && A && "undefined" !== typeof h && D.elGeo && 0 <= h.yMax && 0 <= h.yMin && 0 < D.visibleArea && (E = s.max(h.yMax, E), F = s.min(h.yMin, F), I = s.min(1, s.max((E - F) / D.elGeo.elHeight, I)));
                    x || (r = 0);
                    z || (v = 0);
                    u = x;
                    w = z;
                    a.a.forEach(m, function(a) {
                        if (a.onInViewTimeCount)
                            a.onInViewTimeCount(M, l - n);
                        if (a.onFullyInViewTimeCount)
                            a.onFullyInViewTimeCount(M, k - T)
                    });
                    return L
                };
                this.getQS = function(a) {
                    h > l && (h = l);
                    n > k && (n = k);
                    a[x] = Number(O);
                    a[z] = l;
                    a[X] = h;
                    if (0 === f || 2 === f || 3 === f)
                        R && (a[K] = 1), a[P] = k, a[Q] = n;
                    "undefined" !== typeof y && (a[Y] = y);
                    "undefined" !== typeof ba && (a[Z] = ba);
                    "undefined" !== typeof C && (a[aa] = Number(C));
                    if (!0 === N) {
                        var b = s.round(100 * I), c = s.round(100 * A);
                        a[U] = b;
                        a[V] = c;
                        A = I
                    }
                    "undefined" !== typeof W && (a.ib =
                    Number(W));
                    h = l;
                    n = k
                }
            }
            function g() {
                var b = a.b.c, c = b.document;
                return {
                    y: void 0 !== b.pageYOffset ? b.pageYOffset: (c.documentElement || c.body.parentNode || c.body).scrollTop
                }
            }
            a.l = {};
            var c = {};
            a.l.a = function() {
                return C ? a.g.h(a.b.c).height : 750
            };
            a.l.b = function(b) {
                var d = b.zr;
                c[d] = {};
                b.viewstats = {
                    isBigAd: !1
                };
                if (a.g.f()) {
                    var e = k(a.g.d), f;
                    f = new h(e, a.c.d, !0, 0);
                    c[d].strict = f;
                    e = new h(e, a.c.d, !1, 1);
                    c[d].lax = e
                }
                !0 !== b.isSkin && a.k && a.k.b() ? (b = new h(a.k.a, a.c.d, !0, 3), c[d].pscope = b) : a.m && a.m.a() && (b = new h(a.m.b, a.c.d, !0, 2), c[d].pscope =
                b)
            };
            a.l.c = function(b) {
                var d = c[b.zr], e, f;
                for (f in d)
                    d.hasOwnProperty(f) && d[f].update(b) && (e=!0);
                e && a.c.e(b);
                a.n.a(b) && a.c.f(b)
            };
            a.l.d = function(b, c, e) {
                "unefined" == typeof e && (e=!1);
                var f = a.l && a.l.e(b.zr);
                b = (a.k && a.k.b() || a.m && a.m.c) && f && f.pscope.getInViewTime() >= c;
                c = e ? a.g && a.g.f() && f && f.strict && f.strict.getInViewTime() >= c : a.g && a.g.f() && f && f.lax && f.lax.getInViewTime() >= c;
                return a.b.a ? c : b
            };
            a.l.f = function(b, c, e) {
                b = a.l.e(b.zr);
                e = e ? "hadVideo2SecOTS" : "hadOTS";
                var f = (a.k && a.k.b() || a.m && a.m.c) && b && b.pscope[e]();
                if ("strict" === c)
                    var g = a.g && a.g.f() && b && b.strict && b.strict[e]();
                else 
                    "lax" === c && (g = a.g && a.g.f() && b && b.lax && b.lax[e]());
                return a.b.a ? g : f
            };
            a.l.g = function(b, c) {
                var e = a.l.e(b.zr);
                return a.m && a.m.c && e && e.pscope[c ? "hadVideo2SecOTS": "hadOTS"]()
            };
            a.l.h = function(b, c) {
                var e = a.l && a.l.e(b.zr);
                return a.m && a.m.c && e && e.pscope.getFullInviewTimeTotal() >= c
            };
            a.l.i = function(b, c, e) {
                "undefined" == typeof e && (e=!1);
                var f = a.l && a.l.e(b.zr);
                b = (a.k && a.k.b() || a.m && a.m.c) && f && f.pscope.getFullInviewTimeTotal() >= c;
                c = e ? a.g && a.g.f() &&
                f && f.strict && f.strict.getFullInviewTimeTotal() >= c : a.g && a.g.f() && f && f.lax && f.lax.getFullInviewTimeTotal() >= c;
                return a.b.a ? c : b
            };
            a.l.j = function(a) {
                delete c[a]
            };
            a.l.e = function(a) {
                return c[a]
            };
            a.l.k = function(a) {
                var d = null;
                (a = c[a]) && a.strict ? d = a.strict : a && a.pscope && (d = a.pscope);
                return d
            };
            a.l.l = function(a) {
                var d = {}, e = c[a], f;
                for (f in e)
                    e.hasOwnProperty(f) && e[f].getQS(d);
                (a = r[a]) && a.viewstats.isBigAd && (d.el = 1);
                return d
            }
        })(n);
        (function(a) {
            a.o = {};
            a.o.a = function(a, h) {
                var g;
                h.outerHTML ? g = h.outerHTML : (g = document.createElement("div"),
                g.appendChild(h.cloneNode(!0)), g = g.innerHTML);
                g = [/flashvars\s*=\s*(".*?"|'.*?')/i.exec(g), /name\s*=\s*["']flashvars["']\s*value\s*=\s*(".*?"|'.*?')/i.exec(g), /value\s*=\s*(".*?"|'.*?')\s*name\s*=\s*["']flashvars["']/i.exec(g), a];
                for (var c, b, d = {}, e = 0; e < g.length; e++) {
                    if ((c = g[e]) && "object" === typeof c && c[1])
                        c = c[1].replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/^"/g, "").replace(/"$/g, "").replace(/^'/g, "").replace(/'$/g, "");
                    else if ("string" === typeof c)
                        c = c.split("?")[1];
                    else 
                        continue;
                    if (c) {
                        c = c.split("&");
                        for (var f = 0; f < c.length; f++)
                            b = c[f].split("="), 2 > b.length || "function" == b[0].slice(0, 8) || (d[b[0]] = [b[1]])
                        }
                }
                return d
            }
        })(n);
        (function(a) {
            function k(c) {
                var b = {
                    window: 0,
                    transparent: 1,
                    opaque: 2,
                    direct: 3,
                    gpu: 4
                };
                if ("EMBED" === c.tagName)
                    var d = a.a.getAttribute(c, "wmode");
                else if ("OBJECT" === c.tagName) {
                    c = c.getElementsByTagName("param");
                    for (var e = 0; e < c.length; e++) {
                        var f = c[e], g = a.a.getAttribute(f, "name"), f = a.a.getAttribute(f, "value");
                        if ("wmode" === g) {
                            d = f;
                            break
                        }
                    }
                }
                return d && b[d.toLowerCase()] || 5
            }
            function h(c) {
                var b = c.el,
                d = c.url, e = c.flashVars, f = c.adIds;
                (new q).getTime();
                this.ao = f;
                this.FIND_AD_TRIES = f.numTries || 0;
                var m;
                try {
                    if (b) {
                        var l = b, h;
                        "DIV" === l.tagName && ((l = b.getElementsByTagName("EMBED")[0]) || (l = b.getElementsByTagName("OBJECT")[0]), l || (l = b.getElementsByTagName("IMG")[0]), l || (l = b));
                        1 === l.nodeType && "IMG" !== l.nodeName && "EMBED" !== l.nodeName && "OBJECT" !== l.nodeName && (l = b.getElementsByTagName("EMBED")[0] || b.getElementsByTagName("OBJECT")[0] || b.getElementsByTagName("IMG")[0] || b);
                        if ("OBJECT" === l.tagName) {
                            for (var n = 0; n <
                            l.children.length; n++)
                                if ("movie" === l.children[n].name || "Movie" === l.children[n].name)
                                    h = l.children[n].value;
                            l.object && l.object.Movie ? h = l.object.Movie : l.data&&-1 !== l.data.indexOf("swf") && (h = l.data)
                        }
                        "EMBED" !== l.tagName && "IMG" !== l.tagName ||!l.src || (h = l.src);
                        var s = a.o.a(h, l);
                        m = {
                            adURL: h,
                            flashVars: s
                        }
                    } else 
                        m=!1
                } catch (t) {
                    m=!1
                }
                if (m && m.adURL && e)
                    for (p in m.flashVars)
                        m.flashVars.hasOwnProperty(p) && (e[p] = m.flashVars[p]);
                m && m.flashVars && (e = m.flashVars);
                if ("string" !== typeof d || "DIV" === d)
                    d = m && m.adURL || "-";
                d && 0 !== d.toLowerCase().indexOf("http:") &&
                0 !== d.toLowerCase().indexOf("https:") && ("/" === d[0] ? d = window.location.protocol + "//" + window.location.host + d : (m = window.location.pathname.split("/").slice(0, - 1).join("/"), d = window.location.protocol + "//" + window.location.host + "/" + m + (m ? "/" : "") + d));
                "IFRAME" !== b.tagName && "IMG" !== b.tagName&&-1 === d.indexOf("googlesyndication") && (d = d.split("?")[0]);
                this.zr = f.adNum;
                r[this.zr] = this;
                a.n.d(this.zr, [b]);
                this.ae = d;
                this.aa = b;
                this.WINDOW = a.c.a(this.aa);
                this.INITIAL_WIDTH = b.offsetWidth;
                this.INITIAL_HEIGHT = b.offsetHeight;
                "undefined" === typeof e && (e = {});
                u.i[N]=!0;
                this.eg = [];
                this.ee = {};
                a.c.g(this);
                this.ag = e;
                this.ah = void 0;
                this.ai = 0;
                this.an = this.am = this.al = this.ak = this.aj = void 0;
                this.ar = [];
                this.as = [];
                this.at = [];
                this.av = this.au = a.h.a.a.a;
                this.ax = a.h.a.b.a;
                this.ay = a.h.a.c.a;
                this.ba = this.az = a.h.a.d.a;
                this.bb = a.h.a.e.a;
                this.by = this.bx = this.bw = this.bv = this.bu = this.bt = this.bs = this.br = this.bq = this.bp = this.bo = this.bm = this.bl = this.bk = this.bi = this.bh = this.bg = this.bf = this.be = this.bd = this.bc = void 0;
                this.ca = this.bz=!1;
                this.cb = this.cu =
                this.ct = void 0;
                this.cc =+ new q + 12E4;
                this.BEACON_LAST_SENT_AT =+ new q;
                this.cl = this.cm = void 0;
                this.cn = 0;
                this.ck = a.h.a.f.a;
                this.cd=!1;
                this.cy = void 0;
                this.dt=!1;
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
                a.l.b(this);
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
                this.aq[5] =
                0;
                this.aq[6] = 0;
                this.aq[7] = 0;
                this.aq[9] = 0;
                this.aq[8] = 0;
                this.aq[15] = 0;
                this.aq[16] = 0;
                this.aq[21] = 0;
                this.aq[22] = 0;
                this.aq[23] = 0;
                this.INVIEW_TIME_THRESHHOLDS = [5, 10, 15, 30, 60];
                this.DWELL_TIME_THRESHHOLDS = [5, 10, 15, 30, 60];
                this.an = c.adType || g(b);
                0 === this.an && (this.WMODE = k(b));
                a.n.c(this)
            }
            function g(a) {
                return "IFRAME" === a.tagName ? 2 : "IMG" === a.tagName ? 1 : "EMBED" === a.tagName || "OBJECT" === a.tagName ? 0 : 3
            }
            a.n = {};
            a.n.b = function(c, b, d, e, f, m, l) {
                var n;
                n = 1 == arguments.length ? arguments[0] : {
                    el: c,
                    url: b,
                    flashVars: e,
                    adIds: f,
                    opt_props: l
                };
                if (m) {
                    if ("function" === typeof m)
                        return m(c, b, e, f);
                    new q;
                    m.em=!0;
                    r[m.zr] = m;
                    c[K] = m.zr;
                    c[w]=!0;
                    m.aa = c;
                    m.INITIAL_WIDTH = c.offsetWidth;
                    m.INITIAL_HEIGHT = c.offsetHeight;
                    m.ae = b;
                    m.an = g(c);
                    0 === m.an && (m.WMODE = k(c));
                    m.ag = e || {};
                    a.i.a(m);
                    n = {
                        e: 0
                    };
                    n.q = m.aq[0]++;
                    a.j.a(m, n);
                    return m
                }
                return new h(n)
            };
            a.n.c = function(c) {
                c.de = c.ao.startTime;
                c.RAND = c.ao.rand;
                (new q).getTime();
                c.dd=!0;
                a.l.c(c);
                var b = {
                    e: 0
                };
                b.q = c.aq[0]++;
                a.j.a(c, b);
                a.i.a(c)
            };
            a.n.a = function(c) {
                var b =+ new q, d = b - c.BEACON_LAST_SENT_AT;
                if (0 < c.DWELL_TIME_THRESHHOLDS.length) {
                    var e =
                    1E3 * c.DWELL_TIME_THRESHHOLDS[0];
                    if (c.counters.strictDwell.tCur >= e) {
                        c.DWELL_TIME_THRESHHOLDS.shift();
                        var f = c.INVIEW_TIME_THRESHHOLDS.length ? c.INVIEW_TIME_THRESHHOLDS[0]: 60;
                        if (e < f)
                            return !1;
                        if (5E3 < d)
                            return !0
                    }
                }
                return 0 < c.INVIEW_TIME_THRESHHOLDS.length && (f = 1E3 * c.INVIEW_TIME_THRESHHOLDS[0], a.l.d(c, f)) ? (c.INVIEW_TIME_THRESHHOLDS.shift(), !0) : 0 === c.DWELL_TIME_THRESHHOLDS.length && b > c.cc ? (c.cc*=2, !0) : !1
            };
            a.n.d = function(a, b) {
                for (var d = 0; d < b.length; d++) {
                    var e = b[d];
                    e[K] = a;
                    e[w]=!0
                }
            }
        })(n);
        (function(a) {
            function k(g,
            c, b) {
                if (!g)
                    return !1;
                var d = a.a.g(), d = null !== d && 11 > d;
                if (!d)
                    for (var e = g.getElementsByTagName("embed"), f = 0; f < e.length; f++) {
                        var m = e[f];
                        if (!0 !== m[w]&&-1 === m.id.indexOf("moatPx") && a.a.s(m) && m.getAttribute("src")) {
                            var l = m.getAttribute("src"), h = a.o.a(l, m);
                            return a.n.b(m, l, !1, h, c, b)
                        }
                    }
                m = g.getElementsByTagName("object");
                for (f = 0; f < m.length; f++)
                    if (e = m[f], a.a.s(e) && ("undefined" === typeof e[w] ||!0 !== e[w])&&-1 == e.id.indexOf("moatPx")) {
                        for (h = 0; h < e.children.length; h++)
                            if ("movie" === e.children[h].name || "Movie" === e.children[h].name)
                                for (l =
                                e.children[h].value, g = 0; g < e.children.length; g++) {
                                    if (!d && "EMBED" === e.children[g].tagName) {
                                        m = e.children[g];
                                        if ("undefined" !== typeof m[w]&&!0 === m[w]||-1 != m.id.indexOf("moatPx"))
                                            continue;
                                            if (a.a.s(m))
                                                return h = a.o.a(l, m), a.n.b(m, l, !1, h, c, b)
                                            }
                                            if ("OBJECT" === e.children[g].tagName && (f = e.children[g], a.a.s(f)&&!0 !== f[w]&&-1 === f.id.indexOf("moatPx")))
                                                return a.n.b(f, void 0, !1, void 0, c, b)
                                }
                                e.object && e.object.Movie ? l = e.object.Movie : e.data && (l = e.data);
                                h = a.o.a(l, e);
                                return a.n.b(e, l, !1, h, c, b)
                            }
                if (l = a.p.g(g, c, b))
                    return l;
                l = g.getElementsByTagName("img");
                for (f = 0; f < l.length; f++)
                    if (d = l[f], "undefined" === typeof d[w] ||!0 !== d[w]) {
                        if (a.a.s(d) && "" !== d.src&&-1 === document.location.href.indexOf(d.src))
                            return a.n.b(d, d.getAttribute("src"), !1, void 0, c, b);
                            try {
                                h = d.src
                            } catch (k) {
                                h = d.getAttribute && d.getAttribute("src")
                            }
                            S["1"] = d.offsetWidth + "x" + d.offsetHeight + ":" + h
                        }
                return !1
            }
            function h(g, c) {
                for (var b = [], d = g.getElementsByTagName("iframe"), e, f = 0; f < d.length; f++)
                    if (e = d[f], !e[w] && a.a.s(e)) {
                        var m = a.d.c(e)?!1 : !0;
                        (1 === c && m || 2 === c&&!m) && b.push(e)
                    }
                return b
            }
            a.p = {};
            a.p.a = function(g, c, b, d, e) {
                var f = a.p.b, m = 0;
                a.e.j(function() {
                    c.numTries = m++;
                    return f(g, c, b, d)&&!0
                }, a.p.c, 500, e)
            };
            a.p.g = function(g, c, b) {
                g = h(g, 1);
                if (g[0])
                    return a.n.b(g[0], g[0].src, !1, void 0, c, b)
                };
            a.p.f = function(g, c, b) {
                var d;
                g = a.p.e(g, a.p.h);
                for (var e = 0; e < g.length; e++)
                    if (d = g[e], (d = a.d.c(d)) && d.documentElement && (d = k(d.documentElement, c, b)))
                        return d
                };
            a.p.i = function(g, c, b) {
                var d, e;
                g = a.p.e(g, a.p.h);
                for (var f = 0; f < g.length; f++)
                    if (e = g[f], (e = a.d.c(e)) && e.documentElement) {
                        if (d = e.getElementById("flite-1"))
                            if (d =
                            a.n.b(d, d.nodeName, !1, void 0, c, b))
                                return d;
                                if (d = k(e.documentElement, c, b))
                                    return d
                    }
                };
            a.p.d = k;
            a.p.b = function(g, c, b, d) {
                var e = a.p.d;
                if ("undefined" === typeof g)
                    return !1;
                if ("HEAD" === g.tagName) {
                    var f = g.parentNode;
                    "HTML" === f.tagName && (f = f.getElementsByTagName("body"), 0 < f.length && (g = f[0]))
                }
                return (f = e(g, c, b)) || (f = a.p.f(g, c, b)) || (ka || d) && (d = d || a.d.b(g)) && (!a.b.a || "BODY" !== d.nodeName || a.c.a(d) != a.b.c) && (f = e(d, c, b)) ? f : !1
            };
            a.p.e = h;
            a.p.j = 1;
            a.p.h = 2;
            a.p.k = 500;
            a.p.c = 20;
            a.p.l = {
                object: 1,
                embed: 1,
                img: 1,
                iframe: 1
            }
        })(n);
        (function(a) {
            function k(a) {
                var b =
                a;
                try {
                    for (var d = 0; 100 > d && (a = decodeURIComponent(a), b != a)&&!a.match(/^http(s)?\:/); d++)
                        b = a
                } catch (e) {}
                return a.replace(/(^\s+|\s+$)/g, "")
            }
            function h(c, b) {
                function d(a, b) {
                    var c = (new RegExp(b + "=(.*?)&")).exec(a);
                    if (c)
                        return c[1]
                }
                var e;
                if (la) {
                    var f = a.d.a(b);
                    f && (e = f.src)
                } else 
                    for (b.getElementsByTagName("script"), f = 0; f < g.length; f++) {
                        var g = g[0];
                        if (g.src&&-1 < g.src.indexOf("js.ng")) {
                            e = g.src;
                            break
                        }
                    }
                e && (c.zMoatSite = d(e, "site") || "-", c.zMoatPosition = d(e, "position") || d(e, "pos") || d(e, "positions") || "-", f = k(d(e, "pagetype") ||
                "-"), c.zMoatRollup = d(e, "rollup") || "-", f && "-" != f && "" != f && (c.zMoatRollup = [c.zMoatRollup, f].join("-")), c.zMoatSection = d(e, "_section") || "-", c.zMoatSubsection = d(e, "subsection") || "-");
                a.b.a && "beta.next.cnn.com" === a.b.c.location.hostname && (c.zMoatSite = "beta.next.cnn.com")
            }
            function g(c, b) {
                for (var d = {}, e = c.split("&"), f = 0; f < e.length; f++) {
                    var g = e[f].split("=");
                    g[1] = "string" === typeof g[1] ? k(g[1]) : "";
                    d[g[0]] = g[1]
                }
                h(d, E.parentNode);
                (f = a.b.c.document.getElementById("moatab")) && (d.zMoatTarget = f.innerHTML.trim());
                e = a.b.c.optimizely;
                g = [];
                if (e) {
                    var l = e.activeExperiments;
                    if (l && 0 < l.length)
                        for (f = 0; f < l.length; f++)
                            try {
                                var n = e.data.experiments[l[f]].name, q = e.variationNamesMap[l[f]];
                                n&&-1 < n.indexOf("moat") && q && g.push([n, q].join("-").split(" ").join("-"))
                            } catch (r) {}
                }
                0 < g.length && (d.zAudience = g.join(";"));
                "undefined" !== typeof b && (d.clientZone = "undefined" !== b ? b : "");
                return d
            }
            a.q = {};
            a.q.a = function(a, b) {
                if ("undefined" === typeof a.startTime || b)
                    a.startTime = (new q).getTime();
                if ("undefined" === typeof a.rand || b)
                    a.rand = s.floor(s.random() *
                    s.pow(10, 8));
                a.adNum = u.zr;
                u.zr++
            };
            a.q.b = function(c) {
                if (!c)
                    return !1;
                var b;
                b = /^(?:[a-z]+:\/\/|:?\/?\/)?(?:www\.)?([^\/:]*)/i;
                b=!a.b.a && a.b.b ? (b = a.b.b.match(b)) && b[1] ? b[1] : a.b.c.location.hostname : a.b.c.location.hostname;
                var d = b.match(/.*\.([^\.]*\..*)/i);
                c.moatClientSlicer1 = d && d[1] ? decodeURIComponent(d[1]) : decodeURIComponent(b);
                c.moatClientSlicer2 = decodeURIComponent(b);
                return c
            };
            a.q.c = function(c) {
                return a.q.d(c)
            };
            a.q.d = function(a) {
                try {
                    var b = a.className, d = a.getAttribute("src");
                    b.split("\n").join(" ");
                    if ( - 1 !== b.indexOf("moatfooter"))
                        return !1;
                    var e = d.split("?"), f = d.split("#");
                    a=!1;
                    1 < e.length && 1 < f.length && e[1].length < f[1].length && (a=!0);
                    if (1 == e.length || a)
                        e = f;
                    return 1 < e.length ? g(e[1], "undefined") : !1
                } catch (h) {
                    return !1
                }
            };
            a.q.e = function(a) {
                try {
                    var b = a && a.className.replace("amp;", "").split("?")[1];
                    return b && g(b)
                } catch (d) {
                    return !1
                }
            };
            a.q.f = function(a) {
                try {
                    var b = zoneRegEx.exec(a.innerHTML);
                    0 < b.length && (zone = b[1]);
                    return zone
                } catch (d) {}
            };
            a.q.g = function(a) {
                return (a = unescape(unescape(unescape(unescape(a.innerHTML)))).match(/~fdr=(\d*).*?\/.*?;(\d*)/)) ?
                {
                    adid: a && a[1] || "-",
                    cid: a && a[2] || "-"
                } : !1
            };
            a.q.h = function(a, b) {
                return b || {}
            }
        })(n);
        (function(a) {
            function k(c, b) {
                var d, e = [], f, g = a.a.f() ? 2048: 7750, h = b || {};
                f = {};
                for (d in c)
                    c.hasOwnProperty(d) && (1 != c.e || "x" !== d && "y" !== d && "c" !== d ? e.push(encodeURIComponent(d) + "=" + encodeURIComponent(c[d])) : f[d] = c[d].split("a"));
                d = e.join("&");
                var e = g - d.length, k = 0;
                if ("undefined" !== typeof f.x) {
                    for (var n = 0, q = 0; q < f.x.length; q++)
                        if (n += f.x[q].length + (f.y[q] ? f.y[q].length : 0) + (f.c[q] ? f.c[q].length : 0), n < e)
                            k++;
                        else 
                            break;
                    0 < k && (d += "&x=" +
                    f.x.slice(0, k - 1).join("a"), d += "&y=" + f.y.slice(0, k - 1).join("a"), d += "&c=" + f.c.slice(0, k - 1).join("a"))
                }
                for (var r in h)
                    h.hasOwnProperty(r) && (f = "&" + encodeURIComponent(r) + "=" + encodeURIComponent(h[r]), f.length + d.length < g && (d += f));
                return d
            }
            function h(a, b) {
                for (var d in b)
                    b.hasOwnProperty(d) && (a[d] = b[d])
            }
            a.j = {};
            var g = {
                34: "bd.moatads.com",
                36: "bd.moatads.com"
            };
            a.j.b = function(c, b, d, e) {
                a.q.a(b, e);
                e = {};
                e.e = c;
                h(e, d);
                e.i = x;
                e.cm = P;
                if (11 === c) {
                    d = [];
                    for (var f in S)
                        S.hasOwnProperty(f) && d.push(f + "=" + S[f]);
                    e.k = d.join("&")
                }
                e.e in
                ga || (e.bq = a.b.e, e.f = Number(!Q), e.j = a.b.b, e.o = 3, e.t = b.startTime, e.de = b.rand, e.m = 0, e.ar = ha, a.a.y(e, "ai", u.z), e.q = u.m++, e.cb = C ? 1 : 0, e.cu = ca, h(e, a.focus.getQueryString()), a.q.h(b, e), "undefined" !== typeof b && (e.d = b.moatClientLevel1 + ":" + b.moatClientLevel2 + ":-:-", e.zMoatST = b.zMoatSite, e.zMoatPS = b.zMoatPosition, e.zMoatRL = b.zMoatRollup, e.zMoatSC = b.zMoatSection, e.zMoatSSC = b.zMoatSubsection, b.zMoatTarget && (e.zMoatTarget = b.zMoatTarget), b.zAudience && (e.zAudience = b.zAudience)), e.ac = 1, e.it = a.p.k, b = k(e), u.yh.yi(b +
                "&cs=0", g[c] ? a.b.protocol + "//" + g[c] : I, g[c] ? a.b.protocol + "//" + g[c] : ia))
            };
            a.j.c = function(c) {
                !0 !== c.em && (delete r[c.zr], clearTimeout(c.cc), a.p.a(E.parentNode, c.ao, c))
            };
            a.j.a = function(c, b) {
                if (!c ||!0 === c.ep)
                    return !1;
                if (!a.c.h(c))
                    return a.c.i(c), !1;
                if ("undefined" !== typeof c.ao && (2 !== c.an || 1 !== b.e && 3 !== b.e)&&!(b.e in ga)) {
                    b.lo = c.FIND_AD_TRIES;
                    if ("string" === typeof c.ae) {
                        var d;
                        d = a.b.h ? 700 : 1200;
                        b.ak = c.ae.length <= d ? c.ae : c.ae.slice(0, d)
                    } else 
                        b.ak = "-";
                    c.ah && (b.a = c.ah);
                    var e = c.ag;
                    d = {};
                    if (9 === b.e && 2 === b.q || 25 ===
                    b.e) {
                        for (var f in e)
                            e.hasOwnProperty(f) && "" !== f && "undefined" !== typeof e[f]&&-1 === f.indexOf("dvContains")&&-1 === f.indexOf("indexOf")&&-1 === f.toLowerCase().indexOf("clicktag") && (d["z" + f] = e[f]);
                        b.e = 25
                    }
                    0 === c.an && (b.dc = c.WMODE);
                    0 !== b.e && a.c.j(c);
                    c.bi > c.bg && (c.bg = c.bi);
                    c.bm > c.bk && (c.bk = c.bm);
                    b.i = x;
                    b.bq = a.b.e;
                    b.g = c.aq.g++;
                    f = c.INITIAL_WIDTH;
                    b.h = c.INITIAL_HEIGHT;
                    b.w = f;
                    b.cm = P;
                    b.f = Number(!Q);
                    b.j = a.b.b;
                    b.o = 3;
                    b.t = c.de;
                    b.de = c.RAND;
                    b.cu = ca;
                    b.m = b.m || a.a.j(c);
                    b.ar = ha;
                    b.cb = C ? 1 : 0;
                    h(b, a.k.c());
                    a.b.a && (b.gh = 1);
                    a.k && a.k.b() ?
                    (b.ch = 1, b.gh = 1) : a.m && a.m.c ? (b.ct = a.m.d, c && c.periscopeManager ? (c.periscopeManager.measurable && (b.ch = 1), c.periscopeManager.fullyMeasurable && c.ao&&!c.ao.skin && (b.ga = 1)) : b.ch = 1, "undefined" !== typeof a.m.e && c && c.ao && c.ao.startTime&&!isNaN(c.ao.startTime) && (f = a.m.e - c.ao.startTime, b.fg = 0 <= f ? f : 0)) : b.ch = 0;
                    h(b, a.l.l(c.zr));
                    h(b, a.focus.getQueryString());
                    h(b, c.counters.getQs());
                    a.a.y(b, "ai", u.z);
                    a.a.y(b, "ap", c.cb);
                    a.a.y(b, "ax", c.bg);
                    a.a.y(b, "ay", c.bi);
                    a.a.y(b, "az", c.bk);
                    a.a.y(b, "ba", c.bm);
                    a.a.y(b, "aw", c.bc);
                    a.a.y(b,
                    "bg", c.bd);
                    a.a.y(b, "be", c.be);
                    a.a.y(b, "bc", c.bw);
                    a.a.y(b, "bf", c.by);
                    a.a.y(b, "bh", c.bx);
                    a.a.y(b, "bz", c.cu);
                    b.cl = s.round(100 * c.IR5.AREA / (b.w * b.h));
                    b.au = c.aq[2] - 1;
                    b.av = c.aq[3] - 1;
                    b.by = c.aq[23] - 1;
                    b.at = c.dm;
                    a.q.h(c.ao, b);
                    b.d = c.ao.moatClientLevel1 + ":" + c.ao.moatClientLevel2 + ":-:-";
                    b.zMoatST = c.ao.zMoatSite;
                    b.zMoatPS = c.ao.zMoatPosition;
                    b.zMoatRL = c.ao.zMoatRollup;
                    b.zMoatSC = c.ao.zMoatSection;
                    b.zMoatSSC = c.ao.zMoatSubsection;
                    c.ao.zMoatTarget && (b.zMoatTarget = c.ao.zMoatTarget);
                    c.ao.zAudience && (b.zAudience = c.ao.zAudience);
                    b.ab = c.an;
                    b.ac = 1;
                    b.it = a.p.k;
                    c.bi = c.bg;
                    c.bm = c.bk;
                    d = k(b, d);
                    u.yh.yi(d + "&cs=0", g[b.e] ? a.b.protocol + "//" + g[b.e] : I, g[b.e] ? a.b.protocol + "//" + g[b.e] : ia)
                }
            };
            a.j.d = function(a, b) {
                if (2 !== a.an || 1 !== b.e && 3 !== b.e) {
                    var d = ma;
                    (new Image(1, 1)).src = d
                }
            };
            a.j.e = function(a, b) {
                return k(a, b)
            };
            a.j.f = function(c) {
                var b = {
                    e: 16
                };
                b.q = c.aq[16]++;
                a.j.a(c, b)
            }
        })(n);
        (function(a) {
            function k(a, c, b, d) {
                var e = (new q).getTime();
                this.tLast = this.tCur = 0;
                this.update = function(b) {
                    var c = (new q).getTime();
                    if (a(b)) {
                        b = s.min(c - e, 1E3);
                        var h = typeof d;
                        this.tCur +=
                        b;
                        "number" === h ? this.tCur > d && (this.tCur = d) : "function" === h && (b = d(), "number" === typeof b && this.tCur > b && (this.tCur = b))
                    }
                    e = c
                };
                this.getQs = function(a) {
                    a = this.query(a);
                    this.tLast = this.tCur;
                    return a
                };
                this.query = function(a) {
                    a = a || {};
                    this.tLast > this.tCur && (this.tLast = this.tCur);
                    a[c] = this.tCur;
                    a[b] = this.tLast;
                    return a
                }
            }
            function h() {
                if (a.focus.pageIsVisible() && "undefined" === typeof u.z) {
                    u.z = new q - ca;
                    a:
                    {
                        var g = void 0, c;
                        for (c in r)
                            if (r.hasOwnProperty(c) && (g = r[c]) && "undefined" !== typeof g.ao) {
                                if (g.ce)
                                    break a;
                                    var b = {
                                        e: 4
                                    };
                                    b.q = g.aq[4]++;
                                    b.ai = u.z;
                                    a.j.a(g, b);
                                    g.ce=!0
                            }
                        a.e.d(y, "scroll", h, "onScroll")
                    }
                }
            }
            a.c = {};
            a.c.k = function() {
                a.e.c(y, "scroll", h, "onScroll");
                a.focus.setFocusListeners()
            };
            a.c.h = function(g, c) {
                try {
                    var b = g.aa, d = a.a.l(b, 5), e = d && (6 == d.length || 1 <= d.length && "HTML" === d[d.length - 1].nodeName);
                    c = c || g.WINDOW || a.c.a(b);
                    return b && c && e?!0 : !1
                } catch (f) {
                    return !1
                }
            };
            a.c.l = function() {
                var g;
                return function() {
                    for (var c = 0, b = F.length; c < b; c++)
                        if (F[c] === g)
                            return;
                    g = a.e.e(function() {
                        a.c.j();
                        for (var b in r)
                            r.hasOwnProperty(b) && a.l.c(r[b])
                    },
                    200)
                }
            }();
            a.c.j = function() {
                var g, c;
                for (c in r)
                    r.hasOwnProperty(c) && (g = r[c], a.c.h(g, g.WINDOW) ? g.counters.update(g) : a.c.i(g))
            };
            a.c.i = function(g) {
                clearTimeout(g.cc);
                a.e.d(y, "scroll", h, "onScroll");
                g.ep=!0;
                delete r[g.zr];
                a.i.a(g, "remove");
                a.l.j(g.zr);
                g.aa = null;
                g = 0;
                for (prop in r)
                    r.hasOwnProperty && r.hasOwnProperty(prop) && g++;
                0 === g && da()
            };
            a.c.e = function(g) {
                g.eq || (g.eq=!0);
                var c = {
                    e: 5
                };
                c.q = g.aq[5]++;
                a.j.a(g, c)
            };
            a.c.d = function(g) {
                var c, b;
                b = g.aa;
                g.elementRect ? (c = g.elementRect.right - g.elementRect.left, g = g.elementRect.bottom -
                g.elementRect.top) : (c = b.offsetWidth, g = b.offsetHeight);
                return 3 > c || 3 > g || a.focus.pageIsPrerendered()?!0 : !1
            };
            a.c.m = function(a) {
                var c = 1;
                screen.deviceXDPI ? c = screen.deviceXDPI / screen.systemXDPI : a.devicePixelRatio && "undefined" !== typeof a.mozInnerScreenX && (c = a.devicePixelRatio);
                return {
                    w: c * screen.width,
                    h: c * screen.height
                }
            };
            a.c.a = function(a) {
                try {
                    var c = a && a.ownerDocument;
                    return c && (c.defaultView || c.parentWindow)
                } catch (b) {
                    return !1
                }
            };
            a.c.g = function(g) {
                g.counters = {};
                g.counters.laxDwell = new k(function() {
                    return !a.focus.pageIsPrerendered()
                },
                "bu", "cd");
                g.counters.strictDwell = new k(a.focus.pageIsVisible, "ah", "am");
                g.counters.query = function() {
                    var a = {}, b;
                    for (b in this)
                        if (this.hasOwnProperty(b)) {
                            var d = this[b];
                            "function" === typeof d.query && d.query(a)
                        }
                    return a
                };
                g.counters.getQs = function() {
                    var a = {}, b;
                    for (b in this)
                        if (this.hasOwnProperty(b)) {
                            var d = this[b];
                            "function" === typeof d.getQs && d.getQs(a)
                        }
                    return a
                };
                g.counters.update = function(a) {
                    for (var b in this)
                        if (this.hasOwnProperty(b)) {
                            var d = this[b];
                            "function" === typeof d.update && d.update(a)
                        }
                }
            };
            a.c.b =
            function(g, c) {
                for (var b = [], d, e = 0; e < na; e++)
                    if (c != c.parent) {
                        if (d = a.d.a(g, c))
                            b.push(d);
                        else 
                            break;
                            c = c.parent;
                            g = d
                    } else 
                        break;
                return b
            };
            a.c.n = function() {
                u.z = void 0;
                u.zs=!1;
                a.e.d(y, "scroll", h, "onScroll")
            };
            a.c.o = function() {
                for (var g in r)
                    if (r.hasOwnProperty(g)) {
                        var c = r[g];
                        if (c) {
                            a.r && a.r.a && a.r.a(c);
                            var b = {
                                e: 21
                            };
                            b.q = c.aq[21]++;
                            a.j.a(c, b)
                        }
                    }
                };
            a.c.f = function(g) {
                var c = {
                    e: 9
                };
                c.q = g.aq[9]++;
                g.BEACON_LAST_SENT_AT =+ new q;
                a.j.a(g, c)
            }
        })(n);
        (function(a) {
            a.d = {};
            a.d.c = function(k) {
                try {
                    if (k.moatHostileIframe)
                        return null;
                    if (k.src && k.src.slice && "http" === k.src.slice(0, 4) && a.a.k(k.src) != a.a.k(y.location.toString()))
                        return k.moatHostileIframe=!0, null;
                    var h = k && (k.contentDocument || k.contentWindow && k.contentWindow.document);
                    if (h && "string" === typeof h.location.toString())
                        return h;
                    k.moatHostileIframe=!0;
                    return null
                } catch (g) {
                    return k.moatHostileIframe=!0, null
                }
            };
            a.d.a = function(k, h) {
                h = h || a.c.a(k);
                try {
                    return h && h.frameElement
                } catch (g) {
                    return !1
                }
            };
            a.d.b = function(k) {
                return (k = a.d.a(k)) ? k.parentNode : null
            }
        })(n);
        (function(a) {
            a.s = {};
            a.s.a =
            function(k) {
                k[a.b.d] = k[a.b.d] || {
                    zs: !1,
                    zr: 0,
                    h: 0,
                    m: 0,
                    i: {}
                }
            }
        })(n);
        (function(a) {
            var k = function(a, g) {
                function c(a, b, c) {
                    a && f.push({
                        qs: a,
                        jsd: b,
                        fld: c
                    });
                    if (0 === m && 0 < f.length)
                        if (m += 1, a = f.shift(), a.fld && n && k && k.sendMessage)
                            try {
                                k.sendMessage(a)
                    } catch (e) {
                        n=!1, d(a)
                    } else 
                        d(a)
                }
                function b() {
                    try {
                        return new q(1, 1)
                    } catch (a) {
                        var b = window.document.createElement("img");
                        b.height = 1;
                        b.width = 1;
                        return b
                    }
                }
                function d(a) {
                    var c = b();
                    c.toSend = a;
                    c.onerror = function() {
                        var a = this.toSend;
                        l += 1;
                        var b = (a.jsd + "/pixel.gif?" + a.qs).length;
                        2 >
                        l ? d(a) : t && b > u && e()
                    };
                    c.onload = function() {
                        e()
                    };
                    c.src = a.jsd + "/pixel.gif?" + a.qs
                }
                function e() {
                    0 < m && (m -= 1, c())
                }
                var f = [], m = 0, l = 0, k=!1, n=!1, q, r = g[a];
                r.yh = {};
                r = r.yh;
                q = g.Image;
                r.yi = function(a, b, d) {
                    c(a, b, d)
                };
                r.yk = function(b) {
                    k=!0;
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
                r.yj = function() {
                    return !1 === k
                };
                r.zb = function() {
                    try {
                        if (!0 === k) {
                            var a = g.document.getElementById("moatMessageSender");
                            a&&!a.sendMessage && (a = g.document.getElementById("moatMessageSenderEmbed"));
                            a && a.sendMessage && (n=!0, k = a)
                        }
                    } catch (b) {}
                };
                r.zc = function() {
                    try {
                        e()
                    } catch (a) {}
                };
                r.zd = function(a) {
                    try {
                        n=!1, a && a.jsd && f.push(a), e()
                    } catch (b) {}
                };
                var s, t, u = 2083;
                try {
                    s = document.createElement("div"), s.innerHTML = "\x3c!--[if IE 8]>x<![endif]--\x3e", t = "x" === s.innerHTML
                } catch (w) {
                    t=!1
                }
            }.toString();
            a.t = {};
            a.t.a = function() {
                return "undefined" !==
                typeof y.eval && (y.eval("(function(win){ win['Moat#EVA'] = true; })(window)"), "undefined" !== typeof y["Moat#EVA"])?!0 : !1
            };
            a.t.b = function(h) {
                if (!u.yh)
                    if (a.t.a())
                        h.eval("(" + k + ")('" + a.b.d + "',window)");
                    else {
                        var g = h.document.createElement("script");
                        g.type = "text/javascript";
                        g.text = "(" + k + ")('" + a.b.d + "',window)";
                        h.document.body.insertBefore(g, h.document.body.childNodes[0])
                    }
            }
        })(n);
        if (n.a.q())
            return !1;
        var x = "TURNER1", ja = "turnerd8523", ha = "a8a845a-clean", ca = n.b.i, la = n.b.k, w = "moatFound" + x, K = "__moat__" + x, P = function(a,
        k) {
            for (var h = [a], g = 1; g <= k; g++)
                h.push(a + g), h.push(a - g);
            return h[s.floor(s.random() * h.length)]
        }(5, 2);
        if (0 != s.floor(s.random() * P))
            return !1;
        var ma = "", ka = n.b.l, Q = n.b.a, ia = "tu.moatads.com", I, N = 0, U, V, S = {}, na = 50, A = [], t = {}, F = [], r = {}, z=!1, ga = {
            15: "",
            12: "",
            6: "",
            7: ""
        };
        "string" === typeof n.b.protocol && (I = ("https:" === n.b.protocol ? n.b.protocol : "http:") + "//tu.moatads.com");
        I || (I = "//tu.moatads.com");
        var y = n.b.c;
        n.s.a(y);
        var u = y[n.b.d];
        window[n.b.d] = u;
        n.t.b(y);
        n.a.k(Q ? y.location.href : y.document.referrer) || n.a.k(window.location.href);
        var E = n.a.v();
        if (!E)
            return !1;
        n.f = E;
        n.a.w(E);
        var C = n.a.n();
        (function(a) {
            N = u.h;
            u.h++;
            u.i[N]=!1;
            !1 === u.zs && (n.c.k(), u.zs=!0);
            n.c.l();
            n.e.c(window, "unload", function() {
                z || (n.c.o(), z=!0)
            }, !1);
            n.e.c(window, "beforeunload", function() {
                z || (n.c.o(), z=!0)
            }, !1);
            n.e.e(n.i.b, 100);
            n.a.f() && n.e.f(ea, 3E5);
            "undefined" === typeof a && (a = n.q.c(E));
            var k = E.parentNode;
            n.e.f(function() {
                !1 === u.i[N] && (n.j.b(11, a), ea())
            }, 1E4);
            n.j.b(17, a);
            n.p.a(k, a)
        })()
    })(Date, Math)
} catch (e$$44) {
    var ct = (new Date).getTime();
    window["Moat#ETS"] || (window["Moat#ETS"] =
    ct);
    window["Moat#EMC"] || (window["Moat#EMC"] = 0);
    var et = ct - window["Moat#ETS"], hourElapsed = 36E5 <= et, msg = e$$44.name + " in closure: " + e$$44.message + ", stack=" + e$$44.stack;
    if (!hourElapsed && 10 > window["Moat#EMC"]) {
        window["Moat#EMC"]++;
        try {
            var pxSrc = "//tu.moatads.com/pixel.gif?e=24&d=data%3Adata%3Adata%3Adata&i=" + escape("TURNER1") + "&ac=1&k=" + escape(msg) + "&ar=" + escape("a8a845a-clean") + "&j=" + escape(document.referrer) + "&cs=" + (new Date).getTime(), px = new Image(1, 1);
            px.src = pxSrc
        } catch (e$$45) {}
    } else if (hourElapsed) {
        window["Moat#EMC"] =
        1;
        window["Moat#ETS"] = ct;
        try {
            pxSrc = "//tu.moatads.com/pixel.gif?e=24&d=data%3Adata%3Adata%3Adata&i=" + escape("TURNER1") + "&ac=1&k=" + escape(msg) + "&ar=" + escape("a8a845a-clean") + "&j=" + escape(document.referrer) + "&cs=" + (new Date).getTime(), px = new Image(1, 1), px.src = pxSrc
        } catch (e$$46) {}
    }
};

