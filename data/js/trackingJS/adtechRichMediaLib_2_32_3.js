/*!
 * ADTECH Rich Media JS Library 2_32_3 Copyright 2010 AOL Platforms.
 */
var com = com || {};
if (com.adtech = com.adtech || {}, com.adtech.IframeUtils_2_32_3 = function() {}, com.adtech.IframeUtils_2_32_3.IFRAME_AD_CONFIG_FILE = "iframeadconfig.js", com.adtech.IframeUtils_2_32_3.DELIM = "|ifv|", com.adtech.IframeUtils_2_32_3.ARRAY_DELIM = "|ae|", com.adtech.IframeUtils_2_32_3.ADTIME_MACRO = "_ADTIME_", com.adtech.IframeUtils_2_32_3.adServerVars = {}, com.adtech.IframeUtils_2_32_3.pubVars = {}, com.adtech.IframeUtils_2_32_3.geoData = {}, com.adtech.IframeUtils_2_32_3.tagVars = {}, com.adtech.IframeUtils_2_32_3.clickRedirect = "", com.adtech.IframeUtils_2_32_3.displayWindowTarget = null, com.adtech.IframeUtils_2_32_3.postAdConfigProcessedCallback = null, com.adtech.IframeUtils_2_32_3.isInIframe = function() {
    return ("undefined" == typeof adtechCanvasAdPreview ||!adtechCanvasAdPreview) && self != top
}, com.adtech.IframeUtils_2_32_3.isInFriendlyIframe = function(a) {
    var b=!1;
    try {
        var c = this.determineDisplayWindowTarget(a).document;
        c && (b=!0)
    } catch (d) {}
    return b
}, com.adtech.IframeUtils_2_32_3.isBreakoutAdType = function(a) {
    for (var b in a.assetContainers)
        if (a.assetContainers.hasOwnProperty(b)) {
            var c = a.assetContainers[b];
            switch (c.type) {
            case com.adtech.AssetContainerFactory_2_32_3.NO_CONTENT:
            case com.adtech.AssetContainerFactory_2_32_3.FLOATING_DIV:
                return !0;
            case com.adtech.AssetContainerFactory_2_32_3.INLINE_DIV:
                if (c.isExpandable)
                    return !0
            }
        }
    return !1
}, com.adtech.IframeUtils_2_32_3.getTargetParentIframeEle = function(a) {
    var b = a.document.getElementsByTagName("iframe");
    for (i = 0; i < b.length; i++) {
        var c = b[i];
        if (a == parent && c.contentWindow == self || a == parent.parent && c.contentWindow == self.parent || a == parent.parent.parent && c.contentWindow == self.parent.parent)
            return c
    }
}, com.adtech.IframeUtils_2_32_3.registerAdOnTargetParent = function(a, b) {
    this.displayWindowTarget = a;
    var c = this.getTargetParentIframeEle(a);
    if ("undefined" != typeof a.adtechAdCallbacks && a.adtechAdCallbacks[this.ADTIME_MACRO] && com.adtech.Utils_2_32_3.isArray(a.adtechAdCallbacks[this.ADTIME_MACRO]) && (a.adtechAdCallbacks[b.adServerVars.uid] = a.adtechAdCallbacks[this.ADTIME_MACRO].slice(), delete a.adtechAdCallbacks[this.ADTIME_MACRO]), a.adtechAdCallback && a.adtechAdCallback[this.ADTIME_MACRO] && (a.adtechAdCallback[b.adServerVars.uid] = a.adtechAdCallback[this.ADTIME_MACRO], delete a.adtechAdCallback[this.ADTIME_MACRO]), "undefined" == typeof a.adtechAdManager_2_32_3) {
        if (a.adtechAdQueue = a.adtechAdQueue || [], a.adtechAdTargetIframeQueue = a.adtechAdTargetIframeQueue || {}, a.adtechAdQueue.push(b), a.adtechAdTargetIframeQueue[b.adServerVars.uid] = c, "undefined" == typeof a.adtechAdManagerReqs ||!a.adtechAdManagerReqs["2_32_3"]) {
            a.adtechAdManagerReqs = a.adtechAdManagerReqs || {}, a.adtechAdManagerReqs["2_32_3"]=!0;
            var d = a.document.createElement("script");
            d.src = b.rmLibUrl, c.parentNode.insertBefore(d, c)
        }
    } else 
        a.adtechAdManager_2_32_3.registerAd(b, c, !1)
}, com.adtech.IframeUtils_2_32_3.determineDisplayWindowTarget = function(a) {
    var b = com.adtech.Utils_2_32_3.getConfigOverride(a, "displayWindowTarget");
    if (null != b)
        return "undefined" != typeof adtechIframeHashArray && self != top && (b = b == top ? top : b.parent), b;
    for (var c = top, d = null, e = parent; null != e;) {
        try {
            var f = e.document;
            f && (d = e)
        } catch (g) {}
        e = e == top ? null : e.parent
    }
    return d || c
}, com.adtech.IframeUtils_2_32_3.topOrSelf = function() {
    var a=!1;
    try {
        var b = top.document;
        b && (a=!0)
    } catch (c) {}
    return a ? top : self
}, com.adtech.IframeUtils_2_32_3.getDisplayWindowTarget = function() {
    return this.displayWindowTarget ? this.displayWindowTarget : this.topOrSelf()
}, com.adtech.IframeUtils_2_32_3.registerCallback = function(a) {
    this.postAdConfigProcessedCallback = a
}, com.adtech.IframeUtils_2_32_3.processAdConfig = function(a) {
    a.adServerVars = this.adServerVars, a.pubVars = this.pubVars, a.geoData = this.geoData, a.tagVars = this.tagVars, a.clickRedirect = this.clickRedirect, "function" == typeof adtechConfigureAdFLP ? adtechConfigureAdFLP() : this.postAdConfigProcessedCallback && this.postAdConfigProcessedCallback(a.adServerVars.id), this.registerAdOnTargetParent(this.determineDisplayWindowTarget(a), a)
}, com.adtech.IframeUtils_2_32_3.loadAdJsFile = function() {
    var a = "scr";
    document.write("<" + a + 'ipt src="' + this.adServerVars.configBaseURL + this.IFRAME_AD_CONFIG_FILE + '" type="text/javascript"></' + a + "ipt>")
}, com.adtech.IframeUtils_2_32_3.setPropertiesOnObject = function(a, b) {
    for (var c = b.split("&"), d = 0; d < c.length; d++) {
        var e = c[d].split("=");
        a[decodeURIComponent(e[0])] = decodeURIComponent(e[1])
    }
}, com.adtech.IframeUtils_2_32_3.setIframeObjectsFromHashArray = function() {
    this.setPropertiesOnObject(this.adServerVars, adtechIframeHashArray[1]), this.setPropertiesOnObject(this.pubVars, adtechIframeHashArray[2]), this.setPropertiesOnObject(this.geoData, adtechIframeHashArray[3]), this.clickRedirect = adtechIframeHashArray[4], this.setPropertiesOnObject(this.tagVars, adtechIframeHashArray[5])
}, com.adtech.IframeUtils_2_32_3.getStringFromSimpleJsonObject = function(a) {
    var b = "";
    for (var c in a)
        a.hasOwnProperty(c) && (b += this.applyHashEncoding(c) + "=" + this.applyHashEncoding(a[c]) + "&");
    return b.substring(0, b.length - 1)
}, com.adtech.IframeUtils_2_32_3.applyHashEncoding = function(a) {
    for (var b = com.adtech.Utils_2_32_3.isFirefox() ? 2 : 1; b;)
        a = encodeURIComponent(a), b--;
    return a
}, com.adtech.IframeUtils_2_32_3.loadIframeBuster = function(a, b) {
    var c = self.document.referrer;
    if ("" == c)
        try {
            c = parent.location.href
    } catch (d) {}
    try {
        var e = parent.document.referrer;
        e && (c = e)
    } catch (d) {}
    var f = c.match(/https?:\/\/[^:\/]*/i);
    0 != a.indexOf("/") && (a = "/" + a), a += "#" + b.rmLibUrl + this.DELIM + this.getStringFromSimpleJsonObject(b.adServerVars) + this.DELIM + this.getStringFromSimpleJsonObject(b.pubVars) + this.DELIM + this.getStringFromSimpleJsonObject(b.geoData) + this.DELIM + b.clickRedirect + this.DELIM + this.getStringFromSimpleJsonObject(b.tagVars), document.write('<iframe src="' + f + a + '" width="0" height="0" frameborder="0" scrolling="no"></iframe>')
}, com.adtech.SafeframeUtils_2_32_3 = function() {}, com.adtech.SafeframeUtils_2_32_3.isInSafeframe = function() {
    try {
        var a = window.sfAPI || $sf.ext;
        if (a)
            return !0
    } catch (b) {}
    return !1
}, com.adtech.SafeframeUtils_2_32_3.isInSafeframeSandbox = function() {
    return this.getSafeframeSandbox()?!0 : !1
}, com.adtech.SafeframeUtils_2_32_3.getSafeframeSandbox = function() {
    return window.adtechSafeframeSandbox
}, com.adtech.SafeframeUtils_2_32_3.register = function(a, b, c) {
    try {
        this.sfAPI = parent.sfAPI || parent.$sf.ext, this.sfAPI && this.sfAPI.register(a, b, c)
    } catch (d) {}
}, com.adtech.SafeframeUtils_2_32_3.getInViewPercentage = function() {
    try {
        if (this.sfAPI) {
            var a = this.sfAPI.geom();
            return 100 * a.self.iv
        }
    } catch (b) {}
}, com.adtech.SafeframeUtils_2_32_3.getMaximumAdArea = function() {
    try {
        if (this.sfAPI) {
            var a = this.sfAPI.geom();
            return (a.self.b - a.self.t) * (a.self.r - a.self.l)
        }
    } catch (b) {}
}, com.adtech.SafeframeUtils_2_32_3.expand = function(a, b) {
    this.adjustSafeframeContainer(a, b), a.contractedX && a.contractedX != a.contentWidth - a.contractedWidth && (b.r = 0), a.contractedY && a.contractedY != a.contentHeight - a.contractedHeight && (b.b = 0);
    try {
        this.sfAPI && this.sfAPI.expand(b)
    } catch (c) {}
}, com.adtech.SafeframeUtils_2_32_3.prototype.safeframeObjFromClips = function(a, b, c) {
    var d = /rect\s*\(\s*(\w+)px\s*,\s*(\w+)px\s*,\s*(\w+)px\s*,\s*(\w+)px\s*\)/i, e = d.exec(a), f = d.exec(b), g = new Object;
    return g.push = c, g.t = Math.abs(parseInt(f[1]) - parseInt(e[1])), g.r = Math.abs(parseInt(f[2]) - parseInt(e[2])), g.b = Math.abs(parseInt(f[3]) - parseInt(e[3])), g.l = Math.abs(parseInt(f[4]) - parseInt(e[4])), g
}, com.adtech.SafeframeUtils_2_32_3.collapse = function(a) {
    this.adjustSafeframeContainer(a);
    try {
        this.sfAPI && this.sfAPI.collapse()
    } catch (b) {}
}, com.adtech.SafeframeUtils_2_32_3.adjustSafeframeContainer = function(a, b) {
    var c = this.getSafeframeSandbox();
    return b ? (c.style.position = "relative", b.l && (c.style.left =- b.l + (a.contentWidth - this.sfAPI.geom().self.w) + "px", a.anchorDiv.style.left = a.contractedX + "px"), void(b.t && (c.style.top =- b.t + (a.contentHeight - this.sfAPI.geom().self.h) + "px", a.anchorDiv.style.top = a.contractedY + "px"))) : (c.style.position = "static", c.style.left = "auto", c.style.top = "auto", a.anchorDiv.style.left = "auto", void(a.anchorDiv.style.top = "auto"))
}, com.adtech.SafeframeUtils_2_32_3.createSafeframeSandbox = function(a) {
    var b = "adtechSafeFrameSandbox_" + a.adServerVars.uid, c = a.assetContainers.main.contentWidth, d = a.assetContainers.main.contentHeight, e = "width:" + c + "px;height:" + d + "px;border:0px;";
    document.write('<iframe id="' + b + '" style="' + e + '" seamless="seamless" scrolling="no" frameborder="0" allowtransparency="true" allowFullScreen="true" mozallowFullScreen="true" webkitAllowFullScreen="true"></iframe>');
    var f = document.getElementById(b), g = f.contentWindow, h = g.document, i = "scr";
    h.open(), g.adtechSafeframeSandbox = f, a.overrides = a.overrides || {}, a.overrides.displayWindowTarget = g, g.adtechAdQueue = [a], g.adtechAdManagerReqs = {
        "2_32_3": !0
    }, window.adtechAdCallbacks && (g.adtechAdCallbacks = window.adtechAdCallbacks), h.write('<html><head></head><body style="padding:0px;margin:0px;"><' + i + 'ipt type="text/javascript" src="' + a.rmLibUrl + '"></' + i + "ipt></body><html>"), h.close()
}, /*!  SWFObject v2.2 <http://code.google.com/p/swfobject/>
  is released under the MIT License <http://www.opensource.org/licenses/mit-license.php>
*/
com.adtech.swfobject_2_32_3 = function() {
    function a() {
        if (!G) {
            try {
                var a = y.getElementsByTagName("body")[0].appendChild(k("span"));
                a.parentNode.removeChild(a)
            } catch (c) {
                return 
            }
            G=!0, b();
            for (var d = B.length, e = 0; d > e; e++)
                B[e]()
        }
    }
    function b() {
        if (!D) {
            D=!0;
            for (var a = C.length, b = 0; a > b; b++)
                C[b]()
        }
    }
    function c(a) {
        G ? a() : B[B.length] = a
    }
    function d(a) {
        D || G ? a() : C[C.length] = a
    }
    function e(a) {
        var b = com.adtech.IframeUtils_2_32_3, c = b && b.isInIframe() && b.isInFriendlyIframe() ? b.topOrSelf(): x;
        if (typeof x.addEventListener != r)
            c.addEventListener("load", a, !1);
        else if (typeof y.addEventListener != r)
            c.document.addEventListener("load", a, !1);
        else if (typeof x.attachEvent != r)
            l(c, "onload", a);
        else if ("function" == typeof c.onload) {
            var d = c.onload;
            c.onload = function() {
                d(), a()
            }
        } else 
            c.onload = a
    }
    function f(a, b, c) {
        var d, e = j(c);
        if (!e) {
            var h = arguments.length > 3 ? arguments[3] + 1: 0, i = this, l = function() {
                f.apply(i, [a, b, c, h])
            };
            3 == arguments.length ? setTimeout(l, 10) : 5 > h && setTimeout(l, 50)
        }
        if (I.wk && I.wk < 312)
            return d;
        if (e)
            if (typeof a.id == r && (a.id = c), I.ie && I.win) {
                var m = "";
                for (var n in a)
                    a[n] != Object.prototype[n] && ("data" == n.toLowerCase() ? b.movie = a[n] : "styleclass" == n.toLowerCase() ? m += ' class="' + a[n] + '"' : "classid" != n.toLowerCase() && (m += " " + n + '="' + a[n] + '"'));
                    var o = "";
                    for (var p in b)
                        b[p] != Object.prototype[p] && (o += '<param name="' + p + '" value="' + b[p] + '" />');
                        e.outerHTML = '<object classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000"' + m + ">" + o + "</object>", E[E.length] = a.id, d = j(a.id)
            } else {
                var q = k(s);
                q.setAttribute("type", v);
                for (var t in a)
                    a[t] != Object.prototype[t] && ("styleclass" == t.toLowerCase() ? q.setAttribute("class", a[t]) : "classid" != t.toLowerCase() && q.setAttribute(t, a[t]));
                    for (var u in b)
                        b[u] != Object.prototype[u] && "movie" != u.toLowerCase() && g(q, u, b[u]);
                        e.parentNode.replaceChild(q, e), d = q
            }
        return d
    }
    function g(a, b, c) {
        var d = k("param");
        d.setAttribute("name", b), d.setAttribute("value", c), a.appendChild(d)
    }
    function h(a) {
        var b = j(a);
        b && "OBJECT" == b.nodeName && (I.ie && I.win ? (b.style.display = "none", function() {
            4 == b.readyState ? i(a) : setTimeout(arguments.callee, 10)
        }()) : b.parentNode.removeChild(b))
    }
    function i(a) {
        var b = j(a);
        if (b) {
            for (var c in b)
                "function" == typeof b[c] && (b[c] = null);
            b.parentNode.removeChild(b)
        }
    }
    function j(a) {
        var b = null;
        try {
            b = y.getElementById(a)
        } catch (c) {}
        return b
    }
    function k(a) {
        return y.createElement(a)
    }
    function l(a, b, c) {
        a.attachEvent(b, c), F[F.length] = [a, b, c]
    }
    function m(a) {
        a += "";
        var b = I.pv, c = a.split(".");
        return c[0] = parseInt(c[0], 10), c[1] = parseInt(c[1], 10) || 0, c[2] = parseInt(c[2], 10) || 0, b[0] > c[0] || b[0] == c[0] && b[1] > c[1] || b[0] == c[0] && b[1] == c[1] && b[2] >= c[2]?!0 : !1
    }
    function n(a, b, c, d) {
        if (!I.ie ||!I.mac) {
            var e = y.getElementsByTagName("head")[0];
            if (e) {
                var f = c && "string" == typeof c ? c: "screen";
                if (d && (p = null, q = null), !p || q != f) {
                    var g = k("style");
                    g.setAttribute("type", "text/css"), g.setAttribute("media", f), p = e.appendChild(g), I.ie && I.win && typeof y.styleSheets != r && y.styleSheets.length > 0 && (p = y.styleSheets[y.styleSheets.length - 1]), q = f
                }
                I.ie && I.win ? p && typeof p.addRule == s && p.addRule(a, b) : p && typeof y.createTextNode != r && p.appendChild(y.createTextNode(a + " {" + b + "}"))
            }
        }
    }
    function o(a, b) {
        if (H) {
            var c = b ? "inherit": "hidden";
            G && j(a) ? j(a).style.visibility = c : n("#" + a, "visibility:" + c)
        }
    }
    {
        var p, q, r = "undefined", s = "object", t = "Shockwave Flash", u = "ShockwaveFlash.ShockwaveFlash", v = "application/x-shockwave-flash", w = "onreadystatechange", x = window, y = document, z = navigator, A=!1, B = [], C = [], D=!1, E = [], F = [], G=!1, H=!0, I = function() {
            var a = typeof y.getElementById != r && typeof y.getElementsByTagName != r && typeof y.createElement != r, b = z.userAgent.toLowerCase(), c = z.platform.toLowerCase(), d = /win/.test(c ? c : b), e = /mac/.test(c ? c : b), f = /webkit/.test(b) ? parseFloat(b.replace(/^.*webkit\/(\d+(\.\d+)?).*$/, "$1")): !1, g=!1, h = [0, 0, 0], i = null;
            if (typeof z.plugins != r && typeof z.plugins[t] == s)
                i = z.plugins[t].description, !i || typeof z.mimeTypes != r && z.mimeTypes[v]&&!z.mimeTypes[v].enabledPlugin || (A=!0, g=!1, i = i.replace(/^.*\s+(\S+\s+\S+$)/, "$1"), h[0] = parseInt(i.replace(/^(.*)\..*$/, "$1"), 10), h[1] = parseInt(i.replace(/^.*\.(.*)\s.*$/, "$1"), 10), h[2] = /[a-zA-Z]/.test(i) ? parseInt(i.replace(/^.*[a-zA-Z]+(.*)$/, "$1"), 10) : 0);
            else if (typeof x.ActiveXObject != r)
                try {
                    var j = new ActiveXObject(u);
                    j && (i = j.GetVariable("$version"), i && (g=!0, i = i.split(" ")[1].split(","), h = [parseInt(i[0], 10), parseInt(i[1], 10), parseInt(i[2], 10)]))
            } catch (k) {}
            return {
                w3: a,
                pv: h,
                wk: f,
                ie: g,
                win: d,
                mac: e,
                string: b
            }
        }();
        !function() {
            I.w3 && ((typeof y.readyState != r && "complete" == y.readyState || typeof y.readyState == r && (y.getElementsByTagName("body")[0] || y.body)) && a(), G || (typeof y.addEventListener != r && y.addEventListener("DOMContentLoaded", a, !1), I.ie && I.win && (y.attachEvent(w, function() {
                "complete" == y.readyState && (y.detachEvent(w, arguments.callee), a())
            }), x == top&&!function() {
                if (!G) {
                    try {
                        y.documentElement.doScroll("left")
                    } catch (b) {
                        return void setTimeout(arguments.callee, 0)
                    }
                    a()
                }
            }()), I.wk&&!function() {
                return G ? void 0 : /loaded|complete/.test(y.readyState) ? void a() : void setTimeout(arguments.callee, 0)
            }(), e(a)))
        }(), function() {
            I.ie && I.win && window.attachEvent("onunload", function() {
                for (var a = F.length, b = 0; a > b; b++)
                    F[b][0].detachEvent(F[b][1], F[b][2]);
                for (var c = E.length, d = 0; c > d; d++)
                    h(E[d]);
                for (var e in I)
                    I[e] = null;
                I = null;
                for (var f in com.adtech.swfobject)
                    com.adtech.swfobject[f] = null;
                com.adtech.swfobject = null
            })
        }()
    }
    return {
        embedSWF: function(a, b, c, e, g, h, i, j, k) {
            var l = {
                success: !1,
                id: b
            };
            I.w3&&!(I.wk && I.wk < 312) && a && b && c && e && g ? (o(b, !1), d(function() {
                c += "", e += "";
                var d = {};
                if (j && typeof j === s)
                    for (var n in j)
                        d[n] = j[n];
                d.data = a, d.width = c, d.height = e;
                var p = {};
                if (i && typeof i === s)
                    for (var q in i)
                        p[q] = i[q];
                if (h && typeof h === s)
                    for (var t in h)
                        typeof p.flashvars != r ? p.flashvars += "&" + t + "=" + h[t] : p.flashvars = t + "=" + h[t];
                if (m(g)) {
                    var u = f(d, p, b);
                    d.id == b && o(b, !0), l.success=!0, l.ref = u
                } else 
                    o(b, !0);
                k && k(l)
            })) : k && k(l)
        },
        switchOffAutoHideShow: function() {
            H=!1
        },
        ua: I,
        getFlashPlayerVersion: function() {
            return {
                major: I.pv[0],
                minor: I.pv[1],
                release: I.pv[2]
            }
        },
        hasFlashPlayerVersion: m,
        createSWF: function(a, b, c) {
            return I.w3 ? f(a, b, c) : void 0
        },
        removeSWF: function(a) {
            I.w3 && h(a)
        },
        createCSS: function(a, b, c, d) {
            I.w3 && n(a, b, c, d)
        },
        setDOMContext: function(a) {
            y = a
        },
        forceSWFRender: function() {
            b()
        },
        addDomLoadEvent: c,
        addLoadEvent: e
    }
}(), "object" != typeof JSON && (JSON = {}), function() {
    "use strict";
    function f(a) {
        return 10 > a ? "0" + a : a
    }
    function quote(a) {
        return escapable.lastIndex = 0, escapable.test(a) ? '"' + a.replace(escapable, function(a) {
            var b = meta[a];
            return "string" == typeof b ? b : "\\u" + ("0000" + a.charCodeAt(0).toString(16)).slice( - 4)
        }) + '"' : '"' + a + '"'
    }
    function str(a, b) {
        var c, d, e, f, g, h = gap, i = b[a];
        switch (i && "object" == typeof i && "function" == typeof i.toJSON && (i = i.toJSON(a)), "function" == typeof rep && (i = rep.call(b, a, i)), typeof i) {
        case"string":
            return quote(i);
        case"number":
            return isFinite(i) ? String(i) : "null";
        case"boolean":
        case"null":
            return String(i);
        case"object":
            if (!i)
                return "null";
            if (gap += indent, g = [], "[object Array]" === Object.prototype.toString.apply(i)) {
                for (f = i.length, c = 0; f > c; c += 1)
                    g[c] = str(c, i) || "null";
                return e = 0 === g.length ? "[]" : gap ? "[\n" + gap + g.join(",\n" + gap) + "\n" + h + "]" : "[" + g.join(",") + "]", gap = h, e
            }
            if (rep && "object" == typeof rep)
                for (f = rep.length, c = 0; f > c; c += 1)
                    "string" == typeof rep[c] && (d = rep[c], e = str(d, i), e && g.push(quote(d) + (gap ? ": " : ":") + e));
            else 
                for (d in i)
                    Object.prototype.hasOwnProperty.call(i, d) && (e = str(d, i), e && g.push(quote(d) + (gap ? ": " : ":") + e));
            return e = 0 === g.length ? "{}" : gap ? "{\n" + gap + g.join(",\n" + gap) + "\n" + h + "}" : "{" + g.join(",") + "}", gap = h, e
        }
    }
    "function" != typeof Date.prototype.toJSON && (Date.prototype.toJSON = function() {
        return isFinite(this.valueOf()) ? this.getUTCFullYear() + "-" + f(this.getUTCMonth() + 1) + "-" + f(this.getUTCDate()) + "T" + f(this.getUTCHours()) + ":" + f(this.getUTCMinutes()) + ":" + f(this.getUTCSeconds()) + "Z" : null
    }, String.prototype.toJSON = Number.prototype.toJSON = Boolean.prototype.toJSON = function() {
        return this.valueOf()
    });
    var cx, escapable, gap, indent, meta, rep;
    "function" != typeof JSON.stringify && (escapable = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g, meta = {
        "\b": "\\b",
        "	": "\\t",
        "\n": "\\n",
        "\f": "\\f",
        "\r": "\\r",
        '"': '\\"',
        "\\": "\\\\"
    }, JSON.stringify = function(a, b, c) {
        var d;
        if (gap = "", indent = "", "number" == typeof c)
            for (d = 0; c > d; d += 1)
                indent += " ";
        else 
            "string" == typeof c && (indent = c);
        if (rep = b, b && "function" != typeof b && ("object" != typeof b || "number" != typeof b.length))
            throw new Error("JSON.stringify");
        return str("", {
            "": a
        })
    }), "function" != typeof JSON.parse && (cx = /[\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g, JSON.parse = function(text, reviver) {
        function walk(a, b) {
            var c, d, e = a[b];
            if (e && "object" == typeof e)
                for (c in e)
                    Object.prototype.hasOwnProperty.call(e, c) && (d = walk(e, c), void 0 !== d ? e[c] = d : delete e[c]);
            return reviver.call(a, b, e)
        }
        var j;
        if (text = String(text), cx.lastIndex = 0, cx.test(text) && (text = text.replace(cx, function(a) {
            return "\\u" + ("0000" + a.charCodeAt(0).toString(16)).slice( - 4)
        })), /^[\],:{}\s]*$/.test(text.replace(/\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, "@").replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, "]").replace(/(?:^|:|,)(?:\s*\[)+/g, "")))
            return j = eval("(" + text + ")"), "function" == typeof reviver ? walk({
                "": j
            }, "") : j;
        throw new SyntaxError("JSON.parse")
    })
}(), com.adtech.Utils_2_32_3 = function() {}, com.adtech.Utils_2_32_3.IE_9_TRIDENT_VERSION = 5, com.adtech.Utils_2_32_3.IE_8_TRIDENT_VERSION = 4, com.adtech.Utils_2_32_3.JSON_CD_POST_REQUEST_FIELD = "event", com.adtech.Utils_2_32_3.isBrowserW3CCompliant = function() {
    var a = "undefined", b = document;
    return typeof b.getElementById != a && typeof b.getElementsByTagName != a && typeof b.createElement != a
}, com.adtech.Utils_2_32_3.getUserAgent = function() {
    return navigator.userAgent.toLowerCase()
}, com.adtech.Utils_2_32_3.getAppVersion = function() {
    return navigator.appVersion.toLowerCase()
}, com.adtech.Utils_2_32_3.getAppName = function() {
    return navigator.appName.toLowerCase()
}, com.adtech.Utils_2_32_3.isLinux = function() {
    return this.getAppVersion().indexOf("linux")>-1
}, com.adtech.Utils_2_32_3.isWin = function() {
    return this.getAppVersion().indexOf("windows")>-1
}, com.adtech.Utils_2_32_3.isMac = function() {
    return this.getAppVersion().indexOf("mac")>-1
}, com.adtech.Utils_2_32_3.isiOS = function() {
    return this.getUserAgent().match(/(ipad|iphone|ipod)/i)?!0 : !1
}, com.adtech.Utils_2_32_3.getiOSVersion = function() {
    if (!this.isiOS())
        return null;
    var a = navigator.appVersion.match(/os (\d+)_(\d+)_?(\d+)?/i);
    return a ? {
        major: parseInt(a[1], 10),
        minor: parseInt(a[2], 10)
    } : null
}, com.adtech.Utils_2_32_3.fileIsImage = function(a) {
    var b = ["gif", "jpeg", "jpg", "png"];
    try {
        var c = a.split(".")
    } catch (d) {
        return !1
    }
    if (c.length > 1)
        for (var e = c[c.length - 1].toLowerCase(), f = 0; f < b.length; f++)
            if (e == b[f])
                return !0;
    return !1
}, com.adtech.Utils_2_32_3.isIE = function() {
    var a = this.getUserAgent();
    if ("netscape" == this.getAppName()) {
        var b = new RegExp("trident/.*rv:([0-9]{1,}[.0-9]{0,})");
        if (null != b.exec(a))
            return 1
    }
    return a.indexOf("msie")>-1
}, com.adtech.Utils_2_32_3.isFirefox = function() {
    return this.getUserAgent().indexOf("firefox")>-1
}, com.adtech.Utils_2_32_3.isWebkit = function() {
    return this.getUserAgent().indexOf("applewebkit")>-1
}, com.adtech.Utils_2_32_3.isSafari = function() {
    return this.getUserAgent().indexOf("safari")>-1&&!this.isChrome()
}, com.adtech.Utils_2_32_3.isChrome = function() {
    return "object" == typeof window.chrome
}, com.adtech.Utils_2_32_3.getIEVersion = function() {
    if (!this.isIE())
        return null;
    var a = this.getUserAgent();
    if ("netscape" == this.getAppName()) {
        var b = new RegExp("trident/.*rv:([0-9]{1,}[.0-9]{0,})");
        if (null != b.exec(a))
            return {
                major: parseInt(RegExp.$1, 10),
                minor: parseInt(RegExp.$1, 10)
            }
    }
    var c = a.match(/msie\s+(\d*\.?\d+)/i);
    return c ? {
        major: parseInt(c[1], 10),
        minor: parseInt(c[2], 10)
    } : null
}, com.adtech.Utils_2_32_3.isIEQuirksMode = function() {
    return this.isIE() && document.compatMode && "BackCompat" == document.compatMode
}, com.adtech.Utils_2_32_3.isPluginEnabled = function(a, b, c) {
    var d = this.getUserAgent();
    if (/msie\s[^\s]*\smac/.test(d))
        return !1;
    var e = this.getIEVersion(), f = e ? e.major: 0, g = /trident\/(\d*\.?\d+)/.exec(d), h = null !== g && f >= 7 ? parseInt(g[1]): 0;
    switch (a) {
    case com.adtech.ContentFactory_2_32_3.FLASH:
        try {
            return com.adtech.swfobject_2_32_3.hasFlashPlayerVersion(b)
        } catch (i) {
            return !1
        }
    case com.adtech.ContentFactory_2_32_3.HTML:
        return (!this.isIE() || f && h >= c) && "undefined" != typeof window.postMessage && this.userAgentSupportsJSON();
    default:
        return !0
    }
}, com.adtech.Utils_2_32_3.userAgentSupportsJSON = function() {
    return window.JSON && JSON.parse && JSON.stringify
}, com.adtech.Utils_2_32_3.extend = function(a, b) {
    var c = function() {};
    c.prototype = b.prototype, a.prototype = new c, a.prototype.constructor = a, a.supa = b.prototype, b.prototype.constructor == Object.prototype.constructor && (b.prototype.constructor = b)
}, com.adtech.Utils_2_32_3.isArray = function(a) {
    return "array" == Object.prototype.toString.call(a).slice(8, - 1).toLowerCase()
}, com.adtech.Utils_2_32_3.inArray = function(a, b) {
    if (this.isArray(b)) {
        for (var c = 0; c < b.length; c++)
            if (b[c] == a)
                return !0
    } else 
        for (var d in b)
            if (b.hasOwnProperty(d) && b[d] == a)
                return !0;
    return !1
}, com.adtech.Utils_2_32_3.getMatchingObjectFromArray = function(a, b, c) {
    for (var d = 0; d < a.length; d++) {
        var e = a[d];
        if ("undefined" != typeof e[b] && e[b] === c)
            return e
    }
    return null
}, com.adtech.Utils_2_32_3.dispatchElementFocusEvent = function(a, b) {
    b&&!a.windowInFocus ? (a.windowInFocus=!0, a.dispatchEvent(com.adtech.RichMediaEvent_2_32_3.FOCUS)) : !b && a.windowInFocus && (a.windowInFocus=!1, a.dispatchEvent(com.adtech.RichMediaEvent_2_32_3.BLUR))
}, com.adtech.Utils_2_32_3.dispatchTouchStartEvent = function(a) {
    a.isTouchDevice || (a.isTouchDevice=!0, a.dispatchEvent(com.adtech.RichMediaEvent_2_32_3.TOUCH_START))
}, com.adtech.Utils_2_32_3.addCustomEventHandler = function(a, b, c) {
    if (a instanceof com.adtech.EventDispatcher_2_32_3)
        for (var d = 0; d < c.length; d++) {
            var e = c[d];
            e.custom && a.addEventListener(e.name, b.customEventHandler, b)
        }
}, com.adtech.Utils_2_32_3.createClosure = function(a, b) {
    var c = [].slice.call(arguments, 2), d = function() {
        var a = arguments.callee, b = [].slice.call(arguments), c = a.extraArgs.concat(b, [a]);
        return a.handler.apply(a.target, c)
    };
    return d.extraArgs = c, d.target = a, d.handler = b, d
}, com.adtech.Utils_2_32_3.registerNativeEventHandler = function(a, b, c) {
    "function" == typeof a.addEventListener ? a.addEventListener(b, c, !1) : a.attachEvent("on" + b, c)
}, com.adtech.Utils_2_32_3.deregisterNativeEventHandler = function(a, b, c) {
    "function" == typeof a.removeEventListener ? a.removeEventListener(b, c, !1) : a.detachEvent("on" + b, c)
}, com.adtech.Utils_2_32_3.getPageOffsets = function(a) {
    var b = a ? com.adtech.IframeUtils_2_32_3.topOrSelf().document: document, c = b.pageXOffset || b.documentElement.scrollLeft || b.body.scrollLeft, d = b.pageYOffset || b.documentElement.scrollTop || b.body.scrollTop;
    return {
        x: c,
        y: d
    }
}, com.adtech.Utils_2_32_3.getViewportDims = function(a) {
    var b = a ? com.adtech.IframeUtils_2_32_3.topOrSelf().document: document, c = b.documentElement.clientWidth, d = "CSS1Compat" === b.compatMode && c || b.body.clientWidth || c, e = b.documentElement.clientHeight, f = "CSS1Compat" === b.compatMode && e || b.body.clientHeight || e;
    return this.isiOS() && f < window.innerHeight && (f = window.innerHeight), {
        w: d,
        h: f
    }
}, com.adtech.Utils_2_32_3.calculateAbsolutePosition = function(a) {
    for (var b = 0, c = 0; a;)
        b += a.offsetTop, c += a.offsetLeft, a = a.offsetParent;
    return {
        x: c,
        y: b
    }
}, com.adtech.Utils_2_32_3.getDocumentHeight = function() {
    var a = document;
    return Math.max(Math.max(a.body.scrollHeight, a.documentElement.scrollHeight), Math.max(a.body.offsetHeight, a.documentElement.offsetHeight), Math.max(a.body.clientHeight, a.documentElement.clientHeight))
}, com.adtech.Utils_2_32_3.getDocumentWidth = function() {
    var a = document;
    return Math.max(Math.max(a.body.scrollWidth, a.documentElement.scrollWidth), Math.max(a.body.offsetWidth, a.documentElement.offsetWidth), Math.max(a.body.clientWidth, a.documentElement.clientWidth))
}, com.adtech.Utils_2_32_3.getComputedStyle = function(a, b) {
    try {
        return "undefined" != typeof a.currentStyle ? a.currentStyle[b] : window.getComputedStyle(a, null)[b]
    } catch (c) {
        return null
    }
}, com.adtech.Utils_2_32_3.canUseFixedPositioning = function() {
    return !(this.isiOS() && this.getiOSVersion().major < 5 || this.getUserAgent().indexOf("opera mini")>-1 || this.isIE() && this.getIEVersion().major < 8)
}, com.adtech.Utils_2_32_3.requestJs = function(a) {
    try {
        var b = document.createElement("script");
        b.src = a, b.charset = "utf-8", document.body.appendChild(b)
    } catch (c) {
        var d = "<scr", e = "ipt>";
        document.write(d + 'ipt type="text/javascript" charset="utf-8" src="' + a + '"></scr' + e)
    }
}, com.adtech.Utils_2_32_3.canRenderRich = function(a) {
    var b = a.assetContainers, c=!0, d = this.getConfigOverride(a, "HTMLOnIE8") ? this.IE_8_TRIDENT_VERSION : this.IE_9_TRIDENT_VERSION;
    for (var e in b)
        if (b.hasOwnProperty(e)) {
            var f = b[e].type == com.adtech.AssetContainerFactory_2_32_3.NO_CONTENT ? "": b[e].contentType;
            if (!this.isPluginEnabled(f, b[e].pluginVersion, d)) {
                c=!1;
                break
            }
        }
    return this.passesRenderRichExclusions(a) && this.isBrowserW3CCompliant() && c
}, com.adtech.Utils_2_32_3.passesRenderRichExclusions = function(a) {
    var b=!1;
    for (var c in a.assetContainers)
        if (a.assetContainers.hasOwnProperty(c)) {
            var d = a.assetContainers[c];
            if (d.contentType == com.adtech.ContentFactory_2_32_3.FLASH && "window" == d.wmode) {
                b=!0;
                break
            }
        }
    return b && this.isLinux()&&!this.isWebkit()?!1 : !0
}, com.adtech.Utils_2_32_3.calculatePercentValue = function(a, b) {
    return Math.round(a / 100 * b)
}, com.adtech.Utils_2_32_3.getVideoReportingIdFromEventName = function(a) {
    var b = "Video ";
    return a.indexOf(b) == a.lastIndexOf(b) && 0 == a.indexOf(b) ? "" : a.substring(0, a.lastIndexOf(" " + b)) + " "
}, com.adtech.Utils_2_32_3.getConfigOverride = function(a, b) {
    return "object" == typeof a && "object" == typeof a.overrides && void 0 !== a.overrides[b] ? a.overrides[b] : null
}, com.adtech.Utils_2_32_3.getQueryStringValue = function(a) {
    if (this.queryStringMap && this.queryStringMap[a])
        return this.queryStringMap[a];
    if (this.queryStringMap)
        return null;
    this.queryStringMap = {};
    var b = window.location.search.substr(1).split("&");
    if (self != top)
        try {
            b.concat(top.location.search.substr(1).split("&"))
    } catch (c) {}
    for (var d = 0; d < b.length; ++d) {
        var e = b[d].split("=");
        2 == e.length && (this.queryStringMap[e[0]] = decodeURIComponent(e[1].replace(/\+/g, " ")))
    }
    return this.getQueryStringValue(a)
}, com.adtech.Utils_2_32_3.mergeObjects = function(a, b) {
    for (var c in b)
        b.hasOwnProperty(c) && (a[c] = b[c])
}, com.adtech.Utils_2_32_3.isEmptyObj = function(a) {
    for (var b in a)
        if (a.hasOwnProperty(b))
            return !1;
    return !0
}, com.adtech.Utils_2_32_3.displayError = function(a) {
    "undefined" != typeof console && console.warn("[PTLA]", a)
}, com.adtech.Utils_2_32_3.removeChildSafe = function(a) {
    for (; a.childNodes.length > 0;)
        com.adtech.Utils_2_32_3.removeChildSafe(a.childNodes[a.childNodes.length - 1]);
    a.parentNode.removeChild(a)
}, com.adtech.Utils_2_32_3.createCORSRequest = function(a, b) {
    var c = new XMLHttpRequest;
    return "withCredentials"in c ? (c.open(a, b, !0), c.setRequestHeader("Content-Type", "text/plain")) : c = null, c
}, com.adtech.Utils_2_32_3.createIframeFormPostRequest = function(a, b) {
    var c = document.createElement("iframe"), d = document.createElement("form"), e = document.createElement("div"), f = document.createElement("input");
    e.style.setAttribute ? e.style.setAttribute("cssText", "display:none;") : e.setAttribute("style", "display:none;"), d.setAttribute("action", a), d.setAttribute("method", "POST"), d.setAttribute("enctype", "text/plain"), d.setAttribute("encoding", "text/plain"), f.setAttribute("type", "hidden"), f.setAttribute("name", com.adtech.Utils_2_32_3.JSON_CD_POST_REQUEST_FIELD), f.setAttribute("value", JSON.stringify(b)), d.appendChild(f), e.appendChild(c), document.body.insertBefore(e, document.body.firstChild);
    var g = c.contentDocument || c.contentWindow.document;
    g.open(), g.write("<!DOCTYPE html><html><body></body></html>"), g.close(), com.adtech.Utils_2_32_3.registerNativeEventHandler(c, "load", function() {
        com.adtech.Utils_2_32_3.removeChildSafe(e)
    }), g.body.appendChild(d), g.getElementsByTagName("form")[0].submit()
}, com.adtech.Utils_2_32_3.makeJsonPostCdRequest = function(a, b) {
    var c = com.adtech.Utils_2_32_3.createCORSRequest("POST", a), d = com.adtech.Utils_2_32_3.JSON_CD_POST_REQUEST_FIELD + "=";
    return c ? (c.onerror = function() {
        com.adtech.Utils_2_32_3.createIframeFormPostRequest(a, b)
    }, d += JSON.stringify(b), void c.send(d)) : void com.adtech.Utils_2_32_3.createIframeFormPostRequest(a, b)
}, com.adtech.Utils_2_32_3.replaceTokens = function(a, b) {
    for (var c in b)
        b.hasOwnProperty(c) && (a = a.replace(c, b[c]));
    return a
}, com.adtech.Utils_2_32_3.replaceCharAt = function(a, b, c) {
    return a.substr(0, b) + c + a.substr(b + 1, a.length)
}, com.adtech.ADRUtils_2_32_3 = function() {}, com.adtech.ADRUtils_2_32_3.TIMER_STR = "Timer", com.adtech.ADRUtils_2_32_3.eventNameToObject = function(a) {
    var b = {
        section: null,
        component: null,
        content: null,
        detail: null,
        action: null
    };
    try {
        eventNameBits = a.split(": ")
    } catch (c) {
        return b
    }
    var d = eventNameBits[0].split(" - ");
    switch (2 == d.length && (b.section = d[0], b.component = d[1], eventNameBits.shift()), eventNameBits.length) {
    case 1:
        b.action = eventNameBits[0];
        break;
    case 2:
        b.content = eventNameBits[0], b.action = eventNameBits[1];
        break;
    case 3:
        b.content = eventNameBits[0], b.detail = eventNameBits[1], b.action = eventNameBits[2]
    }
    return b
}, com.adtech.ADRUtils_2_32_3.convertTimerName = function(a) {
    return a.match(/[^\s]Timer$/) ? (a = a.replace(com.adtech.ADRUtils_2_32_3.TIMER_STR, ""), com.adtech.Utils_2_32_3.replaceCharAt(a, 0, a.charAt(0).toUpperCase())) : a.match(/Video View$/) ? a.replace(/\sView$/, "") : a
}, com.adtech.ADRUtils_2_32_3.processADRValue = function(a) {
    for (key in a)
        a.hasOwnProperty(key) && (a[key] && a[key].constructor == {}.constructor ? a[key] = com.adtech.ADRUtils_2_32_3.processADRValue(a[key]) : (a[key] = "undefined" != typeof a[key] && null !== a[key] ? a[key] : "", a[key] = com.adtech.Utils_2_32_3.isArray(a[key]) ? a[key] : a[key] + ""));
    return a
}, com.adtech.EventDispatcher_2_32_3 = function() {
    this.eventTypes = {}
}, com.adtech.EventDispatcher_2_32_3.prototype.addEventListener = function(a, b, c) {
    "undefined" == typeof this.eventTypes[a] && (this.eventTypes[a] = []), com.adtech.Utils_2_32_3.inArray(b, this.eventTypes[a]) || this.eventTypes[a].push("undefined" == typeof c ? b : {
        ref: c,
        handler: b
    })
}, com.adtech.EventDispatcher_2_32_3.prototype.removeEventListener = function(a, b, c) {
    if ("undefined" != typeof this.eventTypes[a])
        for (var d = this.eventTypes[a].length, e = 0; d > e; e++) {
            var f = this.eventTypes[a][e], g = "undefined" == typeof c ? "function": "object";
            if (g == typeof f)
                switch (g) {
                case"function":
                    f == b && this.eventTypes[a].splice(e, 1);
                    break;
                case"object":
                    f.handler == b && f.ref == c && this.eventTypes[a].splice(e, 1)
                }
            }
    }, com.adtech.EventDispatcher_2_32_3.prototype.hasEventListener = function(a, b, c) {
    if ("undefined" == typeof this.eventTypes[a] || 0 == this.eventTypes[a].length)
        return !1;
    for (var d = this.eventTypes[a].length, e = 0; d > e; e++) {
        var f = this.eventTypes[a][e], g = "undefined" == typeof c ? "function": "object";
        if (g == typeof f)
            switch (g) {
            case"function":
                if (f === b)
                    return !0;
                    break;
                case"object":
                    if (f.handler === b && f.ref === c)
                        return !0
            }
    }
    return !1
}, com.adtech.EventDispatcher_2_32_3.prototype.dispatchEvent = function(a) {
    if ("string" == typeof a && (a = new com.adtech.RichMediaEvent_2_32_3(a)), "string" == typeof a.type && "undefined" != typeof this.eventTypes[a.type]) {
        ("undefined" == typeof a.target || null === a.target) && (a.target = this), a.currentTarget = this;
        for (var b = this.eventTypes[a.type].slice(0), c = 0; c < b.length; c++) {
            var d = b[c];
            "object" == typeof d ? d.handler.call(d.ref, a) : d(a)
        }
    }
}, com.adtech.RichMediaEvent_2_32_3 = function(a) {
    this.type = a, this.target = null, this.currentTarget = null
}, com.adtech.RichMediaEvent_2_32_3.DOM_LOAD = "DOMLoad", com.adtech.RichMediaEvent_2_32_3.ENGAGEMENT = "engagement", com.adtech.RichMediaEvent_2_32_3.VIEWABLE_IMPRESSION = "viewableImpression", com.adtech.RichMediaEvent_2_32_3.INTERACTIVE_IMPRESSION = "interactiveImpression", com.adtech.RichMediaEvent_2_32_3.VIDEO_IMPRESSION = "videoImpression", com.adtech.RichMediaEvent_2_32_3.QUALIFIED_ROLLOVER = "qualifiedRollover", com.adtech.RichMediaEvent_2_32_3.BACKUP_VIEW = "backupView", com.adtech.RichMediaEvent_2_32_3.UNSUPPORTED_CLIENT = "unsupportedClient", com.adtech.RichMediaEvent_2_32_3.BACKUP_VIEWABLE_IMPRESSION = "backupViewableImpression", com.adtech.RichMediaEvent_2_32_3.INDETERMINABLE_VIEWABILITY = "indeterminableViewability", com.adtech.RichMediaEvent_2_32_3.MOUSE_MOVE = "mouseMove", com.adtech.RichMediaEvent_2_32_3.MOUSE_OUT = "mouseOut", com.adtech.RichMediaEvent_2_32_3.MOUSE_OVER = "mouseOver", com.adtech.RichMediaEvent_2_32_3.PAGE_LOAD = "pageLoad", com.adtech.RichMediaEvent_2_32_3.PAGE_SCROLL = "pageScroll", com.adtech.RichMediaEvent_2_32_3.BLUR = "blur", com.adtech.RichMediaEvent_2_32_3.ORIENTATION_CHANGE = "orientationChange", com.adtech.RichMediaEvent_2_32_3.TOUCH_START = "touchStart", com.adtech.RichMediaEvent_2_32_3.FOCUS = "focus", com.adtech.RichMediaEvent_2_32_3.RENDER = "render", com.adtech.RichMediaEvent_2_32_3.REPORT = "reporting", com.adtech.RichMediaEvent_2_32_3.PAGE_RESIZE = "pageResize", com.adtech.RichMediaEvent_2_32_3.SERVE = "serve", com.adtech.RichMediaEvent_2_32_3.IN_VIEWPORT = "inViewport", com.adtech.RichMediaEvent_2_32_3.OUT_VIEWPORT = "outViewport", com.adtech.RichMediaEvent_2_32_3.EXPANDED = "ADTECH.expanded", com.adtech.RichMediaEvent_2_32_3.CONTRACTED = "ADTECH.contracted", com.adtech.RichMediaEvent_2_32_3.LOADED = "ADTECH.loaded", com.adtech.RichMediaEvent_2_32_3.CLICK = "click", com.adtech.RichMediaEvent_2_32_3.CLOSE = "close", com.adtech.RichMediaEvent_2_32_3.HIDE = "hide", com.adtech.RichMediaEvent_2_32_3.SHOW = "show", com.adtech.RichMediaEvent_2_32_3.READY = "ready", com.adtech.RichMediaEvent_2_32_3.ERROR = "error", com.adtech.RichMediaEvent_2_32_3.MRAID_READY = "MRAIDReady", com.adtech.RichMediaEvent_2_32_3.MRAID_ERROR = "MRAIDError", com.adtech.RichMediaEvent_2_32_3.MRAID_VIEWABLE_CHANGE = "viewableChange", com.adtech.RichMediaEvent_2_32_3.MRAID_STATE_CHANGE = "stateChange", com.adtech.RichMediaEvent_2_32_3.MRAID_KEYBOARD_CHANGE = "keyboardChange", com.adtech.RichMediaEvent_2_32_3.MRAID_LOCATION_CHANGE = "locationChange", com.adtech.RichMediaEvent_2_32_3.MRAID_HEADING_CHANGE = "headingChange", com.adtech.RichMediaEvent_2_32_3.MRAID_NETWORK_CHANGE = "networkChange", com.adtech.RichMediaEvent_2_32_3.MRAID_ORIENTATION_CHANGE = "orientationChange", com.adtech.RichMediaEvent_2_32_3.MRAID_SCREEN_CHANGE = "screenChange", com.adtech.RichMediaEvent_2_32_3.MRAID_SIZE_CHANGE = "sizeChange", com.adtech.RichMediaEvent_2_32_3.MRAID_TILT_CHANGE = "tiltChange", com.adtech.RichMediaEvent_2_32_3.MRAID_SHAKE = "shake", com.adtech.RichMediaEvent_2_32_3.MRAID_RESPONSE = "response", com.adtech.RichMediaEvent_2_32_3.CONFIG_CHANGE = "ADTECH.configChange", com.adtech.RichMediaEvent_2_32_3.DYNAMIC = "PICTELA.dynamic", com.adtech.RichMediaEvent_2_32_3.prototype.property = function(a, b) {
    return "type" != a && "target" != a && (this[a] = b), this
}, com.adtech.Logger_2_32_3 = function(a) {
    this.adManager = a, this.pixelRequests = []
}, com.adtech.Logger_2_32_3.HELIOS_TRACKING_VERSION = "3.0", com.adtech.Logger_2_32_3.LIVE_STATS_BASE_URL = "http://rt.pictela.net", com.adtech.Logger_2_32_3.AD_LOGGER_BASE_URL = "http://ads.pictela.net/a/log", com.adtech.Logger_2_32_3.REPORT_ERRORS = {
    ZERO_ID: "1000"
}, com.adtech.Logger_2_32_3.prototype.log = function(a) {
    if (this.filterForLiveStats(a), "undefined" == typeof a.clickName) {
        if ("undefined" != typeof a.thirdPartyUrl)
            return void this.performRequest(a.thirdPartyUrl);
        if (a.networkId&&!(a.networkId.indexOf("_")>-1)) {
            this.validateEvent(a);
            var b = "undefined" != typeof a.eventValues ? ";EventVals=" + a.eventValues.join(","): "", c = "undefined" != typeof a.eventIds ? a.eventIds.join(","): a.eventId, d = "";
            "undefined" != typeof a.thirdPartyPlacementId && (d += ";kvu.3rd-plc=" + a.thirdPartyPlacementId), "undefined" != typeof a.thirdPartyCreativeId && (d += ";kvu.3rd-creative=" + a.thirdPartyCreativeId);
            var e = a.proto + "://" + a.host + "/custrmevent/" + com.adtech.Logger_2_32_3.HELIOS_TRACKING_VERSION + "/" + a.networkId + "." + a.subNetworkId + "/" + a.placementId + "/" + a.pageId + "/" + a.size + "/AdId=" + a.adId + ";CreativeId=" + a.creativeId + ";BnId=" + a.bannerId + ";misc=" + + new Date + ";refsequenceid=" + a.sequenceId32 + ";refseqid2=" + a.sequenceId64 + ";EventIds=" + c + b + d;
            this.performRequest(e)
        }
    }
}, com.adtech.Logger_2_32_3.prototype.filterForLiveStats = function(a) {
    if (a.target.liveStatsEnabled && (a.interaction || a.clickName)) {
        var b = com.adtech.Logger_2_32_3.LIVE_STATS_BASE_URL + "/" + a.adId + "/" + a.eventId + "/" + + new Date;
        this.performRequest(b)
    }
}, com.adtech.Logger_2_32_3.prototype.performRequest = function(a) {
    var b = this.pixelRequests.length;
    this.pixelRequests[b] = new Image, this.pixelRequests[b].src = a
}, com.adtech.Logger_2_32_3.prototype.validateEvent = function(a) {
    var b=!1;
    try {
        if ("undefined" != typeof a.eventIds) {
            for (var c = a.eventIds.length, d = 0, e = 0; c > e; e++) {
                var f = a.eventIds[e] / 1;
                f > 0 && d++
            }
            d == c && (b=!0)
        } else {
            var f = a.eventId / 1;
            f > 0 && (b=!0)
        }
    } catch (g) {}
    b || this.performRequest(com.adtech.Logger_2_32_3.AD_LOGGER_BASE_URL + "/" + a.canvasId + "/" + com.adtech.Logger_2_32_3.REPORT_ERRORS.ZERO_ID)
}, com.adtech.ADRLogger_2_32_3 = function(a) {
    this.adrRequests = [], com.adtech.ADRLogger_2_32_3.supa.constructor.call(this, a)
}, com.adtech.Utils_2_32_3.extend(com.adtech.ADRLogger_2_32_3, com.adtech.Logger_2_32_3), com.adtech.ADRLogger_2_32_3.ADR_DATA_FIELD = "adrData", com.adtech.ADRLogger_2_32_3.prototype.log = function(a) {
    "object" == typeof a[com.adtech.ADRLogger_2_32_3.ADR_DATA_FIELD] ? (com.adtech.Utils_2_32_3.makeJsonPostCdRequest(a.eventEndpoint, a[com.adtech.ADRLogger_2_32_3.ADR_DATA_FIELD]), this.adrRequests[this.adrRequests.length] = a[com.adtech.ADRLogger_2_32_3.ADR_DATA_FIELD]) : com.adtech.ADRLogger_2_32_3.supa.log.call(this, a)
}, com.adtech.ADRDataLoader_2_32_3 = function(a) {
    this.config = a.adrConfig, this.servingProto = a.adServerVars.servingProto, this.mainContainer = a.assetContainers.main, this.view = a.preview ? "preview" : "content", this.conditionalMap = {
        ct: a.pubVars.clickPixel || null,
        clk: a.pubVars.clickRedirect || null
    }, this.arbitaryConditionalMap = a.tagVars[com.adtech.ADRDataLoader_2_32_3.ARBITRARY_PAIRS_KEY] || null, this.dataLoadTimeout = null, this.timeoutErrorState=!1, this.data = null, com.adtech.ADRDataLoader_2_32_3.supa.constructor.call(this)
}, com.adtech.ADRDataLoader_2_32_3.CONTENT_ENDPOINT_PATH = "{advcode}/creatives/{creativeid}/{view}", com.adtech.ADRDataLoader_2_32_3.ADR_DATA_TYPES = {
    PACKAGE: "package",
    PRODUCTS: "products",
    PROMOS: "promotions",
    OBJ_GRPS: "objectgroups",
    USER_DATA: "advuserdata"
}, com.adtech.ADRDataLoader_2_32_3.CONTENT_PROPERTIES = "CP", com.adtech.ADRDataLoader_2_32_3.CONTENT_MAP = "CM", com.adtech.ADRDataLoader_2_32_3.CONTENT_META_KEY = "__meta", com.adtech.ADRDataLoader_2_32_3.CONTENT_INDEX_KEY = "__idx", com.adtech.ADRDataLoader_2_32_3.DEFAULT_TRACKING_ID_PREFIX = "Item", com.adtech.ADRDataLoader_2_32_3.PRODUCT_ITEM_CLICK_URLS_KEY = "ctUrls", com.adtech.ADRDataLoader_2_32_3.PRODUCT_ITEM_IMP_URLS_KEY = "itUrls", com.adtech.ADRDataLoader_2_32_3.BASE_PROPERTIES = "baseProperties", com.adtech.ADRDataLoader_2_32_3.ARBITRARY_PAIRS_KEY = "dcoKvs", com.adtech.ADRDataLoader_2_32_3.DATA_LOAD_TIMEOUT = 5e3, com.adtech.ADRDataLoader_2_32_3.DEFAULT_PRODUCT_COUNT = 5, com.adtech.Utils_2_32_3.extend(com.adtech.ADRDataLoader_2_32_3, com.adtech.EventDispatcher_2_32_3), com.adtech.ADRDataLoader_2_32_3.prototype.getConfig = function() {
    return this.config
}, com.adtech.ADRDataLoader_2_32_3.prototype.load = function() {
    if (!this.config ||!this.config.map)
        return this.dispatchEvent(com.adtech.RichMediaEvent_2_32_3.ERROR), void com.adtech.Utils_2_32_3.displayError("Missing data map from the advert configuration.");
    var a = {
        "{advcode}": this.config.advertiserCode,
        "{creativeid}": this.config.creativeId,
        "{view}": this.view
    }, b = this.config.contentAPIBaseUrl.replace(/^https?/i, this.servingProto) + com.adtech.ADRDataLoader_2_32_3.CONTENT_ENDPOINT_PATH, c = com.adtech.Utils_2_32_3.replaceTokens(b, a), d = "ADRDataLoadedHandler_" + + new Date, e = this.config.productCount ? this.config.productCount: com.adtech.ADRDataLoader_2_32_3.DEFAULT_PRODUCT_COUNT;
    window[d] = com.adtech.Utils_2_32_3.createClosure(this, this.dataLoadedHandler), c += "?width=" + this.mainContainer.width + "&height=" + this.mainContainer.height, c += "&format=jsonp&pcount=" + e + "&callback=" + d, com.adtech.Utils_2_32_3.requestJs(this.addConditionalParams(c));
    var f = this;
    this.dataLoadTimeout = setTimeout(function() {
        f.dataLoadFailHandler()
    }, com.adtech.ADRDataLoader_2_32_3.DATA_LOAD_TIMEOUT)
}, com.adtech.ADRDataLoader_2_32_3.prototype.addConditionalParams = function(a) {
    for (var b in this.conditionalMap)
        this.conditionalMap.hasOwnProperty(b) && this.conditionalMap[b] && (a += "&" + b + "=" + encodeURIComponent(this.conditionalMap[b]));
    if (this.arbitaryConditionalMap)
        for (var b in this.arbitaryConditionalMap)
            this.arbitaryConditionalMap.hasOwnProperty(b) && this.arbitaryConditionalMap[b] && (a += "&" + b + "=" + encodeURIComponent(this.arbitaryConditionalMap[b]));
    return a
}, com.adtech.ADRDataLoader_2_32_3.prototype.getGroupProperties = function(a) {
    for (var b, c, d = [], e = 0; e < a.length; e++) {
        var f = a[e], g = f.adrType == com.adtech.ADRDataLoader_2_32_3.ADR_DATA_TYPES.PACKAGE ? com.adtech.ADRDataLoader_2_32_3.ADR_DATA_TYPES.OBJ_GRPS: f.adrType, h = this.data[g];
        switch (f.adrType) {
        case com.adtech.ADRDataLoader_2_32_3.ADR_DATA_TYPES.PACKAGE:
            var i = com.adtech.Utils_2_32_3.getMatchingObjectFromArray(h, "id", f.id);
            null != i && (b = this.getPackageItems(i.objects, f.mapping), c = this.extractContentMeta(i.objects));
            break;
        case com.adtech.ADRDataLoader_2_32_3.ADR_DATA_TYPES.PRODUCTS:
            b = this.getProductFeedItems(h, f.mapping);
            break;
        case com.adtech.ADRDataLoader_2_32_3.ADR_DATA_TYPES.PROMOS:
            b = this.getPromotionItem(h, f.mapping);
            break;
        default:
            d.push(f)
        }
    }
    if (com.adtech.Utils_2_32_3.isArray(b))
        for (var j = 0; j < b.length; j++)
            com.adtech.Utils_2_32_3.mergeObjects(b[j], this.getBaseProperties(d).baseProperties);
    else 
        "undefined" == typeof b ? (b = [{}
        ], com.adtech.Utils_2_32_3.mergeObjects(b[0], this.getBaseProperties(d).baseProperties), com.adtech.Utils_2_32_3.isEmptyObj(b[0]) && (b = null)) : com.adtech.Utils_2_32_3.mergeObjects(b, this.getBaseProperties(d).baseProperties);
    return {
        groupObject: b,
        contentMap: c
    }
}, com.adtech.ADRDataLoader_2_32_3.prototype.getBaseProperties = function(a) {
    for (var b = {}, c = {}, d = 0; d < a.length; d++) {
        var e = a[d], f = e.adrType == com.adtech.ADRDataLoader_2_32_3.ADR_DATA_TYPES.PACKAGE ? com.adtech.ADRDataLoader_2_32_3.ADR_DATA_TYPES.OBJ_GRPS: e.adrType, g = this.data[f];
        if ("undefined" != typeof g) {
            var h = com.adtech.Utils_2_32_3.getMatchingObjectFromArray(g, "id", e.id);
            if (null != h) {
                for (var i in e.mapping)
                    if (e.mapping.hasOwnProperty(i)) {
                        b[i] = h.objects[0][e.mapping[i]];
                        var j = this.extractContentMeta(h.objects);
                        j.length > 0 && (c[i] = j[0])
                    }
            } else if (com.adtech.Utils_2_32_3.isArray(g))
                for (var i in e.mapping)
                    e.mapping.hasOwnProperty(i) && (b[i] = g[0][e.mapping[i]]);
            else 
                for (var i in e.mapping)
                    e.mapping.hasOwnProperty(i) && (b[i] = g[e.mapping[i]])
                }
    }
    return {
        baseProperties: b,
        contentMap: c
    }
}, com.adtech.ADRDataLoader_2_32_3.prototype.getContentPropertyValues = function() {
    var a = {}, b = {};
    for (var c in this.config.map)
        if (this.config.map.hasOwnProperty(c)) {
            var d = this.config.map[c];
            if (c != com.adtech.ADRDataLoader_2_32_3.BASE_PROPERTIES) {
                var e = this.getGroupProperties(d);
                e.groupObject && (a[c] = e.groupObject, b[c] = e.contentMap)
            } else {
                var f = this.getBaseProperties(d);
                com.adtech.Utils_2_32_3.mergeObjects(a, f.baseProperties), com.adtech.Utils_2_32_3.mergeObjects(b, f.contentMap)
            }
        }
    var g = {};
    return g[com.adtech.ADRDataLoader_2_32_3.CONTENT_PROPERTIES] = a, g[com.adtech.ADRDataLoader_2_32_3.CONTENT_MAP] = b, g
}, com.adtech.ADRDataLoader_2_32_3.prototype.getThirdPartyTrackingData = function() {
    var a = {};
    if (a[com.adtech.Advert_2_32_3.TRACK_EVENT_VIEW] = [], a[com.adtech.Advert_2_32_3.TRACK_EVENT_CLICKS_ALL] = [], "object" == typeof this.data.trackers) {
        if (this.data.trackers.adview)
            for (var b = 0; b < this.data.trackers.adview.length; b++)
                a[com.adtech.Advert_2_32_3.TRACK_EVENT_VIEW].push(this.data.trackers.adview[b]);
        if (this.data.trackers.click)
            for (var b = 0; b < this.data.trackers.click.length; b++)
                a[com.adtech.Advert_2_32_3.TRACK_EVENT_CLICKS_ALL].push(this.data.trackers.click[b])
    }
    return a
}, com.adtech.ADRDataLoader_2_32_3.prototype.getPackageItems = function(a, b) {
    for (var c = [], d = 0; d < a.length; d++) {
        var e = {};
        for (var f in b)
            b.hasOwnProperty(f) && (e[f] = a[d][b[f]]);
        this.addDynamicItemProperties(e, d), c.push(e)
    }
    return c
}, com.adtech.ADRDataLoader_2_32_3.prototype.getPromotionItem = function(a, b) {
    var c = {};
    if (null !== a)
        for (var d in b)
            b.hasOwnProperty(d) && (c[d] = a[b[d]]);
    return c
}, com.adtech.ADRDataLoader_2_32_3.prototype.getProductFeedItems = function(a, b) {
    for (var c = a, d = c.length, e = 0; d > e; e++)
        for (var f in b)
            if (b.hasOwnProperty(f)) {
                var g = b[f];
                c[e].hasOwnProperty(g) && (c[e][f] = c[e][g], this.addDynamicItemProperties(c[e], e))
            }
    return c
}, com.adtech.ADRDataLoader_2_32_3.prototype.addDynamicItemProperties = function(a, b) {
    a[com.adtech.ADRDataLoader_2_32_3.CONTENT_INDEX_KEY] = b;
    var c = this.config.iqCollectionIdPrefix ? this.config.iqCollectionIdPrefix: com.adtech.ADRDataLoader_2_32_3.DEFAULT_TRACKING_ID_PREFIX, d = 5 > b ? b + 1: "6+";
    a[com.adtech.Advert_2_32_3.CONTENT_PROPERTY_TRACKING_KEY] = c + " " + d
}, com.adtech.ADRDataLoader_2_32_3.prototype.extractContentMeta = function(a) {
    for (var b = [], c = 0; c < a.length; c++) {
        var d = a[c];
        "object" == typeof d && d.hasOwnProperty(com.adtech.ADRDataLoader_2_32_3.CONTENT_META_KEY) && b.push(d[com.adtech.ADRDataLoader_2_32_3.CONTENT_META_KEY])
    }
    return b
}, com.adtech.ADRDataLoader_2_32_3.prototype.dataLoadFailHandler = function() {
    com.adtech.Utils_2_32_3.displayError("ADR response maximum timeout exceeded."), this.timeoutErrorState=!0, this.dispatchEvent(com.adtech.RichMediaEvent_2_32_3.ERROR)
}, com.adtech.ADRDataLoader_2_32_3.prototype.dataLoadedHandler = function(a) {
    if (!this.timeoutErrorState) {
        clearTimeout(this.dataLoadTimeout), this.data = a;
        try {
            if ("object" != typeof a || a.errors)
                throw com.adtech.Utils_2_32_3.displayError(a.errors), new Error
        } catch (b) {
            return void this.dispatchEvent(com.adtech.RichMediaEvent_2_32_3.ERROR)
        }
        var c = this.getContentPropertyValues(), d = new com.adtech.RichMediaEvent_2_32_3(com.adtech.RichMediaEvent_2_32_3.LOADED).property("contentProperties", c[com.adtech.ADRDataLoader_2_32_3.CONTENT_PROPERTIES]).property("contentMap", c[com.adtech.ADRDataLoader_2_32_3.CONTENT_MAP]).property("dcoData", a.dcoData || {}).property("reqClickUrl", a.reqClickUrl || null).property("thirdPartyTracking", this.getThirdPartyTrackingData()).property("eventEndpoint", this.config.eventEndpoint);
        this.dispatchEvent(d)
    }
}, com.adtech.ADREventRecordGenerator_2_32_3 = function() {
    this.engagementSequence = 0
}, com.adtech.ADREventRecordGenerator_2_32_3.IMPRESSION = "impression", com.adtech.ADREventRecordGenerator_2_32_3.USER_INTERACTION = "userInteraction", com.adtech.ADREventRecordGenerator_2_32_3.OTHER = "other", com.adtech.ADREventRecordGenerator_2_32_3.CLICK = "click", com.adtech.ADREventRecordGenerator_2_32_3.VIEWABLE_IMPRESSION = "viewableImpression", com.adtech.ADREventRecordGenerator_2_32_3.INTERACTIVE_IMPRESSION = "interactiveImpression", com.adtech.ADREventRecordGenerator_2_32_3.BACKUP_VIEWABLE_IMPRESSION = "backupViewableImpression", com.adtech.ADREventRecordGenerator_2_32_3.CONTENT_VIEW = "contentView", com.adtech.ADREventRecordGenerator_2_32_3.COMPONENT_VIEW = "componentView", com.adtech.ADREventRecordGenerator_2_32_3.OTHER_USER_ACTION = "otherUserAction", com.adtech.ADREventRecordGenerator_2_32_3.CONTENT_DURATION = "contentDuration", com.adtech.ADREventRecordGenerator_2_32_3.INTERACTION_DURATION = "interactionDuration", com.adtech.ADREventRecordGenerator_2_32_3.IMPRESSION_DURATION = "impressionDuration", com.adtech.ADREventRecordGenerator_2_32_3.prototype.generateADRData = function(a) {
    return a.property(com.adtech.ADRLogger_2_32_3.ADR_DATA_FIELD, com.adtech.ADRUtils_2_32_3.processADRValue(this.getADRData(a)))
}, com.adtech.ADREventRecordGenerator_2_32_3.prototype.getADRData = function(a) {
    var b = {
        eventType: a.adrType,
        width: a.width,
        height: a.height
    };
    switch (a.adrType) {
    case com.adtech.ADREventRecordGenerator_2_32_3.IMPRESSION:
    case com.adtech.ADREventRecordGenerator_2_32_3.VIEWABLE_IMPRESSION:
    case com.adtech.ADREventRecordGenerator_2_32_3.INTERACTIVE_IMPRESSION:
    case com.adtech.ADREventRecordGenerator_2_32_3.BACKUP_VIEWABLE_IMPRESSION:
        b.eventHeader = this.getADREventHeader(a);
        break;
    case com.adtech.ADREventRecordGenerator_2_32_3.CONTENT_VIEW:
    case com.adtech.ADREventRecordGenerator_2_32_3.COMPONENT_VIEW:
        b.engagementHeader = this.getADREngagementHeader(a);
        break;
    case com.adtech.ADREventRecordGenerator_2_32_3.USER_INTERACTION:
    case com.adtech.ADREventRecordGenerator_2_32_3.OTHER_USER_ACTION:
        b.actionHeader = this.getADRActionHeader(a);
        break;
    case com.adtech.ADREventRecordGenerator_2_32_3.CLICK:
        b.actionHeader = this.getADRActionHeader(a), b.url = a.url;
        break;
    case com.adtech.ADREventRecordGenerator_2_32_3.CONTENT_DURATION:
        b.engagementHeader = this.getADREngagementHeader(a), b.percentage = a.percentage, b.duration = a.duration, b.name = a.name;
        break;
    case com.adtech.ADREventRecordGenerator_2_32_3.INTERACTION_DURATION:
        b.actionHeader = this.getADRActionHeader(a), b.duration = a.duration;
        break;
    case com.adtech.ADREventRecordGenerator_2_32_3.IMPRESSION_DURATION:
        b.eventHeader = this.getADREventHeader(a), b.name = com.adtech.ADRUtils_2_32_3.convertTimerName(a.name), b.duration = a.duration;
        break;
    case com.adtech.ADREventRecordGenerator_2_32_3.OTHER:
        b.eventHeader = this.getADREventHeader(a), b.name = a.name
    }
    return b
}, com.adtech.ADREventRecordGenerator_2_32_3.prototype.getADREventHeader = function(a) {
    var b = com.adtech.IframeUtils_2_32_3.topOrSelf() == top, c = com.adtech.Utils_2_32_3.getViewportDims(b), d = {
        adId: a.canvasId,
        formatId: a.formatId,
        templateId: a.templateId,
        productFamilyId: a.productFamilyId
    }, e = {
        regionId: a.regionId,
        networkId: a.networkId,
        advertiserId: a.advertiserId,
        campaignId: a.campaignId,
        flightId: a.adId,
        publisherId: a.publisherId,
        sectionId: null,
        placementId: a.placementId,
        bannerId: a.bannerId
    }, f = {
        placementId: a.thirdPartyPlacementId,
        creativeId: a.thirdPartyCreativeId,
        segments: [],
        companyId: null,
        userId: null
    };
    return {
        dcoData: a.adrEventData,
        APID: null,
        UTID: null,
        device: null,
        browser: null,
        os: null,
        pageUrl: b ? top.location.href: null,
        mobileAppName: null,
        mobileDeviceId: null,
        iframeDomain: null,
        deliveryFramework: null,
        language: null,
        screenResolution: null,
        viewportSize: c.w + "x" + c.h,
        deviceOrientation: null,
        viewable: null,
        positionOnPage: null,
        eventTimestamp: null,
        callerIP: null,
        serverIP: null,
        userAgent: null,
        extension: {},
        pictelaData: d,
        adtechData: e,
        externalData: f
    }
}, com.adtech.ADREventRecordGenerator_2_32_3.prototype.getADREngagementHeader = function(a) {
    var b = com.adtech.ADRUtils_2_32_3.eventNameToObject(a.name);
    return {
        eventHeader: this.getADREventHeader(a),
        advertiserContentId: a.advertiserContentId,
        contentName: a.contentName || b.content,
        contentTypeId: a.contentTypeId,
        contentIndex: null,
        componentId: null,
        componentName: b.component,
        componentProductId: null,
        sectionName: b.section,
        sectionIndex: null,
        sequenceNo: this.engagementSequence++,
        data: {}
    }
}, com.adtech.ADREventRecordGenerator_2_32_3.prototype.getADRActionHeader = function(a) {
    var b = com.adtech.ADRUtils_2_32_3.eventNameToObject(a.name);
    return {
        engagementHeader: this.getADREngagementHeader(a),
        x: a.x,
        y: a.y,
        actionTypeId: null,
        actionCategoryId: null,
        actionName: b.action,
        actionId: null,
        detail: [],
        value: null
    }
}, com.adtech.DynamicEventDispatcher_2_32_3 = function() {
    com.adtech.DynamicEventDispatcher_2_32_3.supa.constructor.call(this)
}, com.adtech.Utils_2_32_3.extend(com.adtech.DynamicEventDispatcher_2_32_3, com.adtech.EventDispatcher_2_32_3), com.adtech.DynamicEventDispatcher_2_32_3.prototype.dispatchEvent = function(a) {
    "string" == typeof a && "undefined" != typeof this.eventTypes[com.adtech.RichMediaEvent_2_32_3.DYNAMIC] && "undefined" == typeof this.eventTypes[a] && (a = new com.adtech.RichMediaEvent_2_32_3(com.adtech.RichMediaEvent_2_32_3.DYNAMIC).property("dynamicValue", a)), com.adtech.DynamicEventDispatcher_2_32_3.supa.dispatchEvent.call(this, a)
}, com.adtech.AdManager_2_32_3 = function() {
    this.adverts = {}, this.globalEventBus = new com.adtech.EventDispatcher_2_32_3, this.renderingInFiF=!1, this.renderingInSafeframe=!1, this.advertClass = com.adtech.Advert_2_32_3, this.createLogger(), this.init()
}, com.adtech.AdManager_2_32_3.PAGE_LOAD_TIMEOUT = 4e3, com.adtech.AdManager_2_32_3.prototype.init = function() {
    var a = this, b = com.adtech.swfobject_2_32_3;
    b.addDomLoadEvent(function() {
        a.dispatchDOMLoadEvent(), a.checkPageLoadStatus()
    }), b.addLoadEvent(function() {
        a.dispatchPageLoadEvent()
    })
}, com.adtech.AdManager_2_32_3.prototype.addWindowEventListeners = function() {
    if (!this.windowEventListenersAdded) {
        this.windowEventListenersAdded=!0;
        var a = this.renderingInFiF ? com.adtech.IframeUtils_2_32_3.topOrSelf(): window;
        com.adtech.Utils_2_32_3.registerNativeEventHandler(a, "scroll", com.adtech.Utils_2_32_3.createClosure(this, this.scrollHandler)), com.adtech.Utils_2_32_3.registerNativeEventHandler(a, "resize", com.adtech.Utils_2_32_3.createClosure(this, this.resizeHandler)), com.adtech.Utils_2_32_3.registerNativeEventHandler(a, "orientationchange", com.adtech.Utils_2_32_3.createClosure(this, this.orientationChangeHandler)), a.hasFocus&&!a.hasFocus() ? this.blurHandler() : this.focusHandler(), com.adtech.Utils_2_32_3.registerNativeEventHandler(a, "blur", com.adtech.Utils_2_32_3.createClosure(this, this.blurHandler)), com.adtech.Utils_2_32_3.registerNativeEventHandler(a, "focus", com.adtech.Utils_2_32_3.createClosure(this, this.focusHandler)), com.adtech.Utils_2_32_3.registerNativeEventHandler(a.document, "touchstart", com.adtech.Utils_2_32_3.createClosure(this, this.touchStartHandler))
    }
}, com.adtech.AdManager_2_32_3.prototype.createLogger = function() {
    this.logger = "undefined" == typeof com.adtech.debugLogger_2_32_3 ? new com.adtech.ADRLogger_2_32_3(this) : new com.adtech.debugLogger_2_32_3
}, com.adtech.AdManager_2_32_3.prototype.checkPageLoadStatus = function() {
    if (!this.globalEventBus.pageLoaded) {
        var a = this;
        this.pageLoadStatusTimeout = setTimeout(function() {
            a.dispatchPageLoadEvent(), clearTimeout(a.pageLoadStatusTimeout)
        }, com.adtech.AdManager_2_32_3.PAGE_LOAD_TIMEOUT)
    }
}, com.adtech.AdManager_2_32_3.prototype.dispatchDOMLoadEvent = function() {
    this.globalEventBus.DOMLoaded || (this.globalEventBus.DOMLoaded=!0, this.globalEventBus.dispatchEvent(com.adtech.RichMediaEvent_2_32_3.DOM_LOAD))
}, com.adtech.AdManager_2_32_3.prototype.dispatchPageLoadEvent = function() {
    this.globalEventBus.pageLoaded || (this.globalEventBus.pageLoaded=!0, this.globalEventBus.dispatchEvent(com.adtech.RichMediaEvent_2_32_3.PAGE_LOAD))
}, com.adtech.AdManager_2_32_3.prototype.initialiseAd = function(a, b, c) {
    this.createGlobalAdvertAccessor(), this.addWindowEventListeners(), this.adverts[a.adServerVars.uid] = new this.advertClass(b, c);
    var d = this.adverts[a.adServerVars.uid];
    d.addEventListener(com.adtech.RichMediaEvent_2_32_3.REPORT, this.logger.log, this.logger);
    var e = a.adServerVars.uid;
    this.checkCustomCallbacks(e, !1), d.init(a, this.globalEventBus, this.renderingInFiF, this.renderingInSafeframe), this.checkCustomCallbacks(e, !0), "undefined" != typeof adtechAdCallback && adtechAdCallback[e] && this.invokeAdvertCallback(e, adtechAdCallback[e], !0)
}, com.adtech.AdManager_2_32_3.prototype.checkCustomCallbacks = function(a, b) {
    if ("undefined" != typeof adtechAdCallbacks && adtechAdCallbacks[a] && com.adtech.Utils_2_32_3.isArray(adtechAdCallbacks[a]))
        for (var c = 0; c < adtechAdCallbacks[a].length; c++)
            this.invokeAdvertCallback(a, adtechAdCallbacks[a][c], b)
}, com.adtech.AdManager_2_32_3.prototype.invokeAdvertCallback = function(a, b, c) {
    switch (typeof b) {
    case"function":
        c && b(this.adverts[a]);
        break;
    case"object":
        var d = c ? "init": "preInit";
        try {
            b[d](this.adverts[a])
        } catch (e) {}
    }
}, com.adtech.AdManager_2_32_3.prototype.registerAd = function(a, b, c) {
    var d = com.adtech.Utils_2_32_3.getConfigOverride(a, "displayWindowTarget"), e = a.tagVars.forceDefault;
    if (a.tagVars.forceDefault = a.tagVars.forceBackup?!0 : !1, com.adtech.SafeframeUtils_2_32_3.isInSafeframe()
        )return void com.adtech.SafeframeUtils_2_32_3.createSafeframeSandbox(a);
    if (b || c ||!com.adtech.SafeframeUtils_2_32_3.isInSafeframeSandbox() && d == self)
        return void this.initialiseAd(a, b, c);
    if (!com.adtech.SafeframeUtils_2_32_3.isInSafeframeSandbox() && com.adtech.IframeUtils_2_32_3.isInIframe() && com.adtech.IframeUtils_2_32_3.isBreakoutAdType(a))
        if (com.adtech.IframeUtils_2_32_3.isInFriendlyIframe(a))
            com.adtech.IframeUtils_2_32_3.registerAdOnTargetParent(com.adtech.IframeUtils_2_32_3.determineDisplayWindowTarget(a), a);
        else if (e)
            a.tagVars.forceDefault=!0, this.initialiseAd(a);
        else {
            var f = a.pubVars.iframeBusterPath ? a.pubVars.iframeBusterPath: "adtech/iframeproxy.html";
            com.adtech.IframeUtils_2_32_3.loadIframeBuster(f, a)
        } else 
            this.renderingInSafeframe = com.adtech.SafeframeUtils_2_32_3.isInSafeframeSandbox(), this.renderingInSafeframe || (this.renderingInFiF = com.adtech.IframeUtils_2_32_3.isInIframe() && com.adtech.IframeUtils_2_32_3.isInFriendlyIframe(a)), this.initialiseAd(a)
}, com.adtech.AdManager_2_32_3.prototype.createGlobalAdvertAccessor = function() {
    com.adtech.getAdvertById = function(a) {
        for (var b in window)
            if (window.hasOwnProperty(b) && b.match(/^adtechAdManager/g) && window[b].adverts) {
                var c = window[b].adverts;
                for (var d in c)
                    if (c.hasOwnProperty(d) && c[d].canvasId == a)
                        return c[d]
            }
        return null
    }
}, com.adtech.AdManager_2_32_3.prototype.registerAds = function(a) {
    for (var b = a.length, c = b - 1; c>-1; c--)
        if ( - 1 != a[c].rmLibUrl.indexOf("2_32_3")) {
            var d = a.splice(c, 1)[0], e = "undefined" != typeof adtechAdTargetIframeQueue && adtechAdTargetIframeQueue[d.adServerVars.uid] ? adtechAdTargetIframeQueue[d.adServerVars.uid]: !1, f = "undefined" != typeof adtechAdTargetDivQueue && adtechAdTargetDivQueue[d.adServerVars.uid] ? adtechAdTargetDivQueue[d.adServerVars.uid]: !1;
            this.registerAd(d, e, f)
        }
}, com.adtech.AdManager_2_32_3.prototype.getAdvert = function(a) {
    return "undefined" != typeof this.adverts[a] ? this.adverts[a] : null
}, com.adtech.AdManager_2_32_3.prototype.dispatchViewPortEvent = function(a) {
    offsets = com.adtech.Utils_2_32_3.getPageOffsets(this.renderingInFiF), dims = com.adtech.Utils_2_32_3.getViewportDims(this.renderingInFiF), this.globalEventBus.dispatchEvent(new com.adtech.RichMediaEvent_2_32_3(a).property("offsetX", offsets.x).property("offsetY", offsets.y).property("width", dims.w).property("height", dims.h))
}, com.adtech.AdManager_2_32_3.prototype.scrollHandler = function() {
    this.dispatchViewPortEvent(com.adtech.RichMediaEvent_2_32_3.PAGE_SCROLL)
}, com.adtech.AdManager_2_32_3.prototype.resizeHandler = function() {
    this.dispatchViewPortEvent(com.adtech.RichMediaEvent_2_32_3.PAGE_RESIZE)
}, com.adtech.AdManager_2_32_3.prototype.orientationChangeHandler = function() {
    this.dispatchViewPortEvent(com.adtech.RichMediaEvent_2_32_3.ORIENTATION_CHANGE)
}, com.adtech.AdManager_2_32_3.prototype.blurHandler = function() {
    com.adtech.Utils_2_32_3.dispatchElementFocusEvent(this.globalEventBus, !1)
}, com.adtech.AdManager_2_32_3.prototype.focusHandler = function() {
    com.adtech.Utils_2_32_3.dispatchElementFocusEvent(this.globalEventBus, !0)
}, com.adtech.AdManager_2_32_3.prototype.touchStartHandler = function() {
    com.adtech.Utils_2_32_3.dispatchTouchStartEvent(this.globalEventBus)
}, com.adtech.Advert_2_32_3 = function(a, b) {
    com.adtech.Advert_2_32_3.supa.constructor.call(this), this.assetContainers = {}, this.rendered=!1, this.reportedEvents = [], this.richView=!0, this.targetIframe = a, this.targetDiv = b, this.containerMouseOverCount = 0, this.timersFlushed=!1, this.macroMap = {}, this.windowInFocus=!0, this.viewEventsLogged=!1, this.isTouchDevice=!1, this.interactionEventBuffer = [], this.hasQualifiedRollover=!1, this.userHasInteracted=!1, this.interactionStartTime = null, this.playingVideos = {}, this.logTimers=!0, this.forceRenderBackup=!1, this.isBuySightAd=!1, this.adrEventData = null, this.adrContentMap = {}, this.tagVarsPropertyMap = {
        forceDefault: "forceRenderBackup",
        placementId: "thirdPartyPlacementId",
        creativeId: "thirdPartyCreativeId"
    }, this.eventBus = new com.adtech.DynamicEventDispatcher_2_32_3, this.rmEvent = com.adtech.RichMediaEvent_2_32_3
}, com.adtech.Utils_2_32_3.extend(com.adtech.Advert_2_32_3, com.adtech.EventDispatcher_2_32_3), com.adtech.Advert_2_32_3.prototype.init = function(a, b, c, d) {
    this.canRenderRich = com.adtech.Utils_2_32_3.canRenderRich(a), this.renderingInFiF = c, this.renderingInSafeframe = d, this.id = a.adServerVars.id;
    for (var e = ["canvasId", "servicesUrl", "servicesHosts", "adServerVars", "pubVars", "clickTrackerUrl", "clickthroughs", "clickRedirect", "contentVariables", "contentProperties", "geoData", "tagVars", "assets", "thirdPartyTracking", "events", "timers", "dataFeeds", "liveStatsEnabled", "polls", "fallback", "formatId", "productFamilyId", "templateId", "dynamic"], f = 0; f < e.length; f++) {
        var g = e[f];
        a.hasOwnProperty(g) && (this[g] = a[g])
    }
    this.debugMode = com.adtech.Utils_2_32_3.getQueryStringValue("canvasDebugAdId") == this.canvasId.toString()?!0 : !1, this.processTagVars(), this.pubVars.clickPixel && this.addToThirdPartyTracking(this.pubVars.clickPixel, com.adtech.Advert_2_32_3.TRACK_TYPE_PIXEL, com.adtech.Advert_2_32_3.TRACK_EVENT_CLICKS_ALL), b.isTouchDevice ? this.setTouchDeviceProperties() : b.addEventListener(this.rmEvent.TOUCH_START, this.touchStartHandler, this);
    var h = com.adtech.Utils_2_32_3.getConfigOverride(a, "engagementThreshold");
    this.engagementThreshold = null !== h ? h : com.adtech.Advert_2_32_3.ENGAGEMENT_THRESHOLD;
    var i = com.adtech.Utils_2_32_3.getConfigOverride(a, "qualifiedRolloverThreshold");
    this.qualifiedRolloverThreshold = null !== i ? i : com.adtech.Advert_2_32_3.QUALIFIED_ROLLOVER_THRESHOLD;
    var j = com.adtech.Utils_2_32_3.getConfigOverride(a, "viewableImpressionThreshold");
    if (this.viewableImpressionThreshold = null !== j ? j : com.adtech.Advert_2_32_3.VIEWABLE_IMPRESSION_THRESHOLD, this.trackViewOnEngagement = com.adtech.Utils_2_32_3.getConfigOverride(a, "trackViewOnEngagement")===!0?!0 : !1, this.disableMouseOverEngagement = com.adtech.Utils_2_32_3.getConfigOverride(a, "disableMouseOverEngagement")===!0?!0 : !1, this.initTimers(), this.eventActions = a.eventHandlers, this.setMacroMap(), this.setDebugOverrides(), !this.forceRenderBackup && this.canRenderRich) {
        var k = a.assetContainers;
        for (var l in k)
            if (k.hasOwnProperty(l)) {
                this.playingVideos[l] = [];
                var m = com.adtech.AssetContainerFactory_2_32_3.getContainer(k[l].type, this, a, l, this.eventBus, b);
                m.addEventListener(this.rmEvent.RENDER, this.renderEventHandler, this), m.addEventListener(this.rmEvent.MOUSE_OVER, this.mouseOverEventHandler, this), m.addEventListener(this.rmEvent.MOUSE_OUT, this.mouseOutEventHandler, this), m.addEventListener(this.rmEvent.MOUSE_OVER, this.customEventHandler, this), m.addEventListener(this.rmEvent.MOUSE_OUT, this.customEventHandler, this), m.addEventListener(this.rmEvent.IN_VIEWPORT, this.containerViewHandler, this), m.addEventListener(this.rmEvent.OUT_VIEWPORT, this.containerViewHandler, this), m.addEventListener(this.rmEvent.CLOSE, this.containerCloseHandler, this), m.addEventListener(this.rmEvent.HIDE, this.containerHideHandler, this), m.isExpandable && (m.addEventListener(this.rmEvent.EXPANDED, this.containerExpansionStateChangeHandler, this), m.addEventListener(this.rmEvent.CONTRACTED, this.containerExpansionStateChangeHandler, this)), this.assetContainers[l] = m
            }
        a.dynamic && this.eventBus.addEventListener(this.rmEvent.DYNAMIC, this.customEventHandler, this), this.addVideoEventHandlers(), com.adtech.Utils_2_32_3.addCustomEventHandler(this.eventBus, this, a.events)
    } else if (this.richView=!1, a.assetContainers.main.type == com.adtech.AssetContainerFactory_2_32_3.INLINE_DIV) {
        a.assetContainers.main.content = this.fallback, a.assetContainers.main.contentType = com.adtech.ContentFactory_2_32_3.HTML, a.assetContainers.main.x = 0, a.assetContainers.main.y = 0, a.assetContainers.main.contentWidth = a.assetContainers.main.width, a.assetContainers.main.contentHeight = a.assetContainers.main.height, a.assetContainers.main.contractedX = 0, a.assetContainers.main.contractedY = 0, a.assetContainers.main.isExpandable=!1;
        var n = com.adtech.AssetContainerFactory_2_32_3.getContainer(a.assetContainers.main.type, this, a, "main", this.eventBus, b);
        n.addEventListener(this.rmEvent.IN_VIEWPORT, this.containerViewHandler, this), n.addEventListener(this.rmEvent.OUT_VIEWPORT, this.containerViewHandler, this), n.addEventListener(this.rmEvent.RENDER, this.backupImageRenderEventHandler, this), this.assetContainers.main = n
    } else 
        this.reportEvent(this.rmEvent.UNSUPPORTED_CLIENT);
    if (this.canMeasureViewExposure = self == top || this.renderingInSafeframe || c && top == parent && com.adtech.IframeUtils_2_32_3.topOrSelf() != self, this.canMeasureViewExposure && (this.windowInFocus = b.windowInFocus, b.addEventListener(this.rmEvent.BLUR, this.windowFocusEventHandler, this), b.addEventListener(this.rmEvent.FOCUS, this.windowFocusEventHandler, this), b.addEventListener(this.rmEvent.SCROLL, this.windowScrollEventHandler, this)), this.richView && a.dynamic && a.adrConfig) {
        this.adrEventRecordGenerator = new com.adtech.ADREventRecordGenerator_2_32_3(this);
        var o = new com.adtech.ADRDataLoader_2_32_3(a);
        o.addEventListener(com.adtech.RichMediaEvent_2_32_3.LOADED, this.adrDataLoadedHandler, this), o.addEventListener(com.adtech.RichMediaEvent_2_32_3.ERROR, this.adrDataErrorHandler, this), o.load()
    } else 
        this.eventBus.dispatchEvent(this.rmEvent.SERVE)
}, com.adtech.Advert_2_32_3.prototype.generateReportEvent = function(a) {
    var b = "", c = "", d=!1;
    "undefined" != typeof a && (b = a.name, c = a.id, d = a.interaction);
    var e = new this.rmEvent(this.rmEvent.REPORT).property("canvasId", this.canvasId).property("adId", this.id).property("eventId", c).property("eventName", b).property("proto", this.adServerVars.servingProto).property("host", this.adServerVars.servingHost).property("bannerId", this.adServerVars.bannerId).property("networkId", this.adServerVars.networkId).property("subNetworkId", this.adServerVars.subNetworkId).property("creativeId", this.adServerVars.creativeId).property("placementId", this.adServerVars.placementId).property("size", this.adServerVars.adSize).property("pageId", this.adServerVars.pageId).property("sequenceId32", this.adServerVars.sequenceId32).property("sequenceId64", this.adServerVars.sequenceId64).property("interaction", d);
    return "undefined" != typeof this.thirdPartyCreativeId && e.property("thirdPartyCreativeId", this.thirdPartyCreativeId), "undefined" != typeof this.thirdPartyPlacementId && e.property("thirdPartyPlacementId", this.thirdPartyPlacementId), e
}, com.adtech.Advert_2_32_3.prototype.reportEvent = function(a) {
    a = "string" == typeof a ? new this.rmEvent(a) : a;
    for (var b = 0; b < this.events.length; b++) {
        var c = this.events[b];
        if (c.name == a.type && c.isLoggable) {
            var d = a.type != this.rmEvent.ENGAGEMENT && a.type != this.rmEvent.INTERACTIVE_IMPRESSION && c.interaction&&!c.forceLog?!0 : !1;
            if (d&&!this.hasQualifiedRollover)
                return void(this.interactionEventBuffer[0] = a);
            if (c.video&&!c.interaction&&!a.forceLog) {
                var e = com.adtech.Utils_2_32_3.getVideoReportingIdFromEventName(a.type), f = e + com.adtech.Advert_2_32_3.VIDEO_EVENT_VIEW, g=!1;
                for (var h in this.assetContainers)
                    if (this.assetContainers.hasOwnProperty(h) && com.adtech.Utils_2_32_3.inArray(f, this.playingVideos[h])) {
                        g=!0;
                        break
                    }
                if (!g)
                    return 
            }
            if (c.cumulative ||!c.cumulative&&!com.adtech.Utils_2_32_3.inArray(a.type, this.reportedEvents)) {
                this.thirdPartyEventTrackingCheck(c);
                var i = this.generateReportEvent(c);
                if ("object" == typeof a.meta && "object" == typeof a.meta.reportingData) {
                    var j = [];
                    for (var k in a.meta.reportingData)
                        if (a.meta.reportingData.hasOwnProperty(k)) {
                            var l = a.meta.reportingData[k];
                            if ("number" == typeof l) {
                                j.push(parseInt(l));
                                break
                            }
                        }
                    j.length > 0 && (i = i.property("eventValues", j))
                }
                switch (this.dispatchEvent(i), a.type) {
                case com.adtech.ADREventRecordGenerator_2_32_3.INTERACTIVE_IMPRESSION:
                case com.adtech.ADREventRecordGenerator_2_32_3.VIEWABLE_IMPRESSION:
                case com.adtech.ADREventRecordGenerator_2_32_3.BACKUP_VIEWABLE_IMPRESSION:
                    this.reportADREvent(a.type)
                }
                if (c.custom) {
                    var m = {
                        name: a.type
                    };
                    c.interaction && this.hasQualifiedRollover ? (this.extractADRMetaFromEvent(a.meta, m), this.reportADREvent(com.adtech.ADREventRecordGenerator_2_32_3.USER_INTERACTION, m)) : this.reportADREvent(com.adtech.ADREventRecordGenerator_2_32_3.OTHER, m)
                }
                this.reportedEvents.push(a.type), a.type == this.rmEvent.ENGAGEMENT && (this.eventBus.dispatchEvent(this.rmEvent.ENGAGEMENT), this.trackViewOnEngagement && this.trackViewEvents()), c.interaction && a.type != this.rmEvent.ENGAGEMENT && (com.adtech.Utils_2_32_3.inArray(this.rmEvent.ENGAGEMENT, this.reportedEvents) || this.reportEvent(this.rmEvent.ENGAGEMENT), this.hasQualifiedRollover&&!com.adtech.Utils_2_32_3.inArray(this.rmEvent.INTERACTIVE_IMPRESSION, this.reportedEvents) && (this.userHasInteracted=!0, this.startTimer(com.adtech.Advert_2_32_3.INTERACTION_TIMER), this.reportEvent(this.rmEvent.INTERACTIVE_IMPRESSION), this.eventBus.dispatchEvent(this.rmEvent.INTERACTIVE_IMPRESSION)))
            }
        }
    }
}, com.adtech.Advert_2_32_3.prototype.trackViewEvents = function() {
    if (!this.viewEventsLogged) {
        this.viewEventsLogged=!0;
        var a = com.adtech.Advert_2_32_3.OPT_TRACKING_KEYS;
        "undefined" != typeof this.thirdPartyCreativeId && (this.adServerVars.viewCounter += a.creativeId + "=" + this.thirdPartyCreativeId + ";"), "undefined" != typeof this.thirdPartyPlacementId && (this.adServerVars.viewCounter += a.placementId + "=" + this.thirdPartyPlacementId + ";"), this.thirdPartyPixelLog(this.adServerVars.viewCounter), this.thirdPartyEventTrackingCheck(com.adtech.Advert_2_32_3.TRACK_EVENT_VIEW), this.thirdPartyPixelLog(this.pubVars.viewCountUrl), this.reportADREvent(com.adtech.ADREventRecordGenerator_2_32_3.IMPRESSION)
    }
}, com.adtech.Advert_2_32_3.prototype.logViewabilityAvailability = function() {
    this.canMeasureViewExposure || (this.reportEvent(this.rmEvent.INDETERMINABLE_VIEWABILITY), this.eventBus.dispatchEvent(this.rmEvent.IN_VIEWPORT))
}, com.adtech.Advert_2_32_3.prototype.initTimers = function() {
    for (var a = 0; a < this.timers.length; a++)
        switch (this.timers[a].meta = {
            requestNum: 1,
            previousVal: 0
        }, this.timers[a].name) {
        case com.adtech.Advert_2_32_3.DISPLAY_TIMER:
        case com.adtech.Advert_2_32_3.VIEW_TIMER:
            this.timers[a].timeElapsed = 0
        }
}, com.adtech.Advert_2_32_3.prototype.checkDisplayTimerValue = function() {
    var a = this.getTimer(com.adtech.Advert_2_32_3.DISPLAY_TIMER), b = "number" == typeof a.timeElapsed && a.timeElapsed ? a.timeElapsed: (new Date).getTime() - a.startTime;
    b >= com.adtech.Advert_2_32_3.TIMER_LOGGING_THRESHOLD && (this.logTimers=!1)
}, com.adtech.Advert_2_32_3.prototype.flushTimers = function() {
    if (this.checkDisplayTimerValue(), !this.timersFlushed && this.logTimers) {
        this.timersFlushed=!0;
        for (var a = [], b = [], c = 0; c < this.timers.length; c++) {
            var d = this.timers[c];
            this.stopTimer(d.name), this.isExcludedTimer(name) || "number" == typeof d.timeElapsed && d.timeElapsed > 0 && (a.push(d.id), b.push(this.getTimerLogPayloadValue(d)), d.name == com.adtech.Advert_2_32_3.DISPLAY_TIMER && this.reportADREvent(com.adtech.ADREventRecordGenerator_2_32_3.IMPRESSION_DURATION, {
                name: com.adtech.Advert_2_32_3.DISPLAY_TIMER,
                duration: this.getTimerLogPayloadValue(d)
            }))
        }
        if (a.length > 0) {
            var e = this.generateReportEvent().property("timerEvent", !0).property("eventValues", b).property("eventIds", a);
            this.dispatchEvent(e)
        }
    }
}, com.adtech.Advert_2_32_3.prototype.isExcludedTimer = function(a) {
    var b = [com.adtech.Advert_2_32_3.ENGAGEMENT_TIMER, com.adtech.Advert_2_32_3.INTERACTION_TIMER];
    return this.isTouchDevice && com.adtech.Utils_2_32_3.inArray(a, b)?!0 : !1
}, com.adtech.Advert_2_32_3.prototype.logTimerValue = function(a) {
    if (this.checkDisplayTimerValue(), !this.isExcludedTimer(a)) {
        var b = this.getTimer(a);
        if (this.logTimers && b && "number" == typeof b.startTime && b.startTime > 0) {
            b.name === com.adtech.Advert_2_32_3.INTERACTION_TIMER && null != this.interactionStartTime && (b.timeElapsed = b.startTime - this.interactionStartTime, this.interactionStartTime = null);
            var c = this.getTimerLogPayloadValue(b);
            if (c > 0) {
                var d = this.generateReportEvent().property("timerEvent", !0).property("eventValues", [c]).property("eventIds", [b.id]);
                this.dispatchEvent(d), b.name === com.adtech.Advert_2_32_3.INTERACTION_TIMER && this.reportADREvent(com.adtech.ADREventRecordGenerator_2_32_3.INTERACTION_DURATION, {
                    name: com.adtech.Advert_2_32_3.INTERACTION_TIMER,
                    duration: c
                })
            }
        }
    }
}, com.adtech.Advert_2_32_3.prototype.getTimerLogPayloadValue = function(a) {
    this.stopTimer(a.name);
    var b = a.meta.requestNum * a.timeElapsed + a.meta.previousVal;
    return a.meta.requestNum++, a.meta.previousVal = a.timeElapsed, a.timeElapsed = 0, b
}, com.adtech.Advert_2_32_3.prototype.startTimer = function(a) {
    if (this.richView) {
        var b = this.getTimer(a);
        !b || "undefined" != typeof b.isRunning && b.isRunning || (b.startTime = (new Date).getTime(), a == com.adtech.Advert_2_32_3.ENGAGEMENT_TIMER && (b.startTime = b.startTime - this.engagementThreshold), b.isRunning=!0, "undefined" == typeof b.timeElapsed && (b.timeElapsed = 0))
    }
}, com.adtech.Advert_2_32_3.prototype.stopTimer = function(a) {
    var b = this.getTimer(a);
    b && b.isRunning && (b.timeElapsed += (new Date).getTime() - b.startTime, b.isRunning=!1)
}, com.adtech.Advert_2_32_3.prototype.getTimer = function(a) {
    for (var b = 0; b < this.timers.length; b++)
        if (this.timers[b].name == a)
            return this.timers[b];
    return null
}, com.adtech.Advert_2_32_3.prototype.startViewTimer = function() {
    var a = this;
    this.viewableImpressionCheckTimer = setTimeout(function() {
        a.reportEvent(a.rmEvent.VIEWABLE_IMPRESSION), a.richView || a.reportEvent(a.rmEvent.BACKUP_VIEWABLE_IMPRESSION)
    }, this.viewableImpressionThreshold), this.startTimer(com.adtech.Advert_2_32_3.VIEW_TIMER)
}, com.adtech.Advert_2_32_3.prototype.stopViewTimer = function() {
    this.viewableImpressionCheckTimer && clearTimeout(this.viewableImpressionCheckTimer), this.stopTimer(com.adtech.Advert_2_32_3.VIEW_TIMER)
}, com.adtech.Advert_2_32_3.prototype.startEngagementTimer = function() {
    this.disableMouseOverEngagement || this.reportEvent(this.rmEvent.ENGAGEMENT), this.startTimer(com.adtech.Advert_2_32_3.ENGAGEMENT_TIMER)
}, com.adtech.Advert_2_32_3.prototype.addVideoEventHandlers = function() {
    this.videoReportingIdentifiers = [];
    for (var a = 0; a < this.events.length; a++) {
        var b = this.events[a];
        if ("undefined" != typeof b.video && b.video) {
            this.eventBus.addEventListener(b.name, this.videoEventHandler, this);
            var c = com.adtech.Utils_2_32_3.getVideoReportingIdFromEventName(b.name);
            com.adtech.Utils_2_32_3.inArray(c, this.videoReportingIdentifiers) || this.videoReportingIdentifiers.push(c)
        }
    }
}, com.adtech.Advert_2_32_3.prototype.getBackupImageHTML = function(a) {
    var b = "adtclk" + this.adServerVars.uid;
    this.createFallbackClickHandler(b);
    var c = this.getContentVariable(com.adtech.Advert_2_32_3.ALT_TEXT_VARIABLE_KEY);
    return '<a href="javascript:' + b + '();void(0);" target="_self"><img src="' + this.getFileUrl(a) + '" alt="' + c.replace(/"/g, "&quot;") + '" style="border:0px;"/></a>'
}, com.adtech.Advert_2_32_3.prototype.createFallbackClickHandler = function(a) {
    var b = this, c = this.clickthroughs[com.adtech.Advert_2_32_3.BACKUP_CLICK] ? com.adtech.Advert_2_32_3.BACKUP_CLICK: com.adtech.Advert_2_32_3.DEFAULT_CLICK;
    window[a] = function() {
        b.click(c)
    }
}, com.adtech.Advert_2_32_3.prototype.getServicesHosts = function() {
    return {
        GET: this.servicesHosts[this.adServerVars.servingProto],
        POST: this.servicesHosts.post
    }
}, com.adtech.Advert_2_32_3.prototype.getAssetContainers = function() {
    return this.assetContainers
}, com.adtech.Advert_2_32_3.prototype.getAssetContainer = function(a) {
    return "undefined" != typeof this.assetContainers[a] ? this.assetContainers[a] : null
}, com.adtech.Advert_2_32_3.prototype.getContentVariable = function(a) {
    return void 0 !== this.contentVariables[a] ? this.contentVariables[a] : null
}, com.adtech.Advert_2_32_3.prototype.getContent = function(a) {
    return void 0 == this.contentProperties || void 0 == this.contentProperties[a] ? null : this.replaceContentPropertyFileValue(this.contentProperties[a])
}, com.adtech.Advert_2_32_3.prototype.replaceContentPropertyFileValue = function(a) {
    switch (typeof a) {
    case"string":
        return this.getFileUrl(a);
    case"object":
        return this.replaceContentPropertyFileValues(a);
    default:
        return a
    }
}, com.adtech.Advert_2_32_3.prototype.replaceContentPropertyFileValues = function(a) {
    if (com.adtech.Utils_2_32_3.isArray(a)) {
        for (var b = [], c = 0; c < a.length; c++)
            b.push(this.replaceContentPropertyFileValue(a[c]));
        return b
    }
    var d = {}, e = [com.adtech.Advert_2_32_3.CONTENT_PROPERTY_TRACKING_KEY, com.adtech.Advert_2_32_3.CONTENT_PROPERTY_PARENT_TRACKING_KEY];
    for (var f in a)
        if (a.hasOwnProperty(f)) {
            if (com.adtech.Utils_2_32_3.inArray(f, e)) {
                d[f] = a[f];
                continue
            }
            d[f] = this.replaceContentPropertyFileValue(a[f])
        }
    return d
}, com.adtech.Advert_2_32_3.prototype.contentView = function(a, b) {
    if ("object" == typeof b) {
        var c = this.getContent(a), d = this.adrContentMap[a];
        c && d && (com.adtech.Utils_2_32_3.isArray(d) && b.hasOwnProperty(com.adtech.ADRDataLoader_2_32_3.CONTENT_INDEX_KEY) && (d = d[b[com.adtech.ADRDataLoader_2_32_3.CONTENT_INDEX_KEY]]), "object" == typeof d && this.reportADREvent(com.adtech.ADREventRecordGenerator_2_32_3.CONTENT_VIEW, {
            advertiserContentId: d.cid,
            contentTypeId: d.tid,
            contentName: d.name,
            name: a
        }));
        var e = b[com.adtech.ADRDataLoader_2_32_3.PRODUCT_ITEM_IMP_URLS_KEY];
        if (com.adtech.Utils_2_32_3.isArray(e))
            for (var f = 0; f < e.length; f++)
                this.thirdPartyPixelLog(e[f])
    }
}, com.adtech.Advert_2_32_3.prototype.contentClick = function(a, b) {
    if (com.adtech.Utils_2_32_3.isArray(this.getContent(a)) && "object" == typeof b) {
        var c = b[com.adtech.ADRDataLoader_2_32_3.PRODUCT_ITEM_CLICK_URLS_KEY];
        if (com.adtech.Utils_2_32_3.isArray(c))
            for (var d = 0; d < c.length; d++)
                this.thirdPartyPixelLog(c[d])
    }
}, com.adtech.Advert_2_32_3.prototype.contentIsDynamic = function(a, b) {
    if (this.adrDataLoader) {
        var c = this.adrDataLoader.getConfig().map;
        for (var d in c)
            if (c.hasOwnProperty(d) && (d == a ||!b && d == com.adtech.ADRDataLoader_2_32_3.BASE_PROPERTIES))
                for (var e = c[d], f = 0, g = e.length; g > f; f++) {
                    var h = e[f], i = d == com.adtech.ADRDataLoader_2_32_3.BASE_PROPERTIES ? a: b;
                    if (h.mapping && h.mapping[i])
                        return !0
                }
    }
    return !1
}, com.adtech.Advert_2_32_3.prototype.getGeoVariable = function(a) {
    return void 0 !== this.geoData[a] ? this.geoData[a] : null
}, com.adtech.Advert_2_32_3.prototype.getDataFeedUrl = function(a) {
    return void 0 !== this.dataFeeds && void 0 !== this.dataFeeds[a] ? this.dataFeeds[a] : null
}, com.adtech.Advert_2_32_3.prototype.getScreenGrabUrl = function(a) {
    return this.screenGrabs && this.screenGrabs[a] ? this.screenGrabs[a] : null
}, com.adtech.Advert_2_32_3.prototype.getPollResultUrl = function(a) {
    return this.polls && this.polls[a] ? this.polls[a] : null
}, com.adtech.Advert_2_32_3.prototype.close = function() {
    for (var a in this.assetContainers)
        this.assetContainers.hasOwnProperty(a) && this.assetContainers[a].close()
}, com.adtech.Advert_2_32_3.prototype.click = function(a) {
    this.executeClick(a)
}, com.adtech.Advert_2_32_3.prototype.dynamicClick = function(a, b) {
    this.executeClick(a, b)
}, com.adtech.Advert_2_32_3.prototype.getFileUrl = function(a) {
    if (a)
        for (var b = 0; b < this.assets.length; b++) {
            var c = this.assets[b];
            if (c.name == a)
                return this.adServerVars.assetBaseURL + c.url
        }
    return a
}, com.adtech.Advert_2_32_3.prototype.getFileUrlById = function(a, b) {
    for (var c = 0; c < this.assets.length; c++) {
        var d = this.assets[c];
        if (d.id === a)
            return this.adServerVars.assetBaseURL + d.url
    }
    return b
}, com.adtech.Advert_2_32_3.prototype.executeClick = function(a, b) {
    if (a = this.verifyClickIdentifier(a), "undefined" != typeof this.clickthroughs[a]) {
        var c = this.clickthroughs[a], d = this.generateClickUrl(a, b);
        switch (this.logClick(a, b), c.target) {
        case"_self":
            top.location.href = d;
            break;
        default:
            window.open(d, this.adServerVars.uid + + new Date, c.features)
        }
    }
}, com.adtech.Advert_2_32_3.prototype.generateClickUrl = function(a, b) {
    var c = "undefined" != typeof adtechDisableClickTracking?!0 : !1;
    if (a = this.verifyClickIdentifier(a), "undefined" != typeof this.clickthroughs[a]) {
        var d = this.clickthroughs[a], e = d.id;
        a == com.adtech.Advert_2_32_3.BACKUP_CLICK && (d = this.clickthroughs[com.adtech.Advert_2_32_3.DEFAULT_CLICK], e = this.clickthroughs[com.adtech.Advert_2_32_3.BACKUP_CLICK].id);
        var f = "undefined" != typeof b && b ? b: d.dest;
        if (!c) {
            f = this.addThirdPartyRedirectsToClickUrl(a, f);
            var g = this.generateClickRedirectUrl({
                id: e
            });
            g && (f = g + escape(f)), "undefined" != typeof this.pubVars.clickRedirect && this.pubVars.clickRedirect && (f = this.replaceServerMacros(this.pubVars.clickRedirect) + f)
        }
        return f
    }
    return ""
}, com.adtech.Advert_2_32_3.prototype.generateClickRedirectUrl = function(a) {
    var b = "";
    if ("undefined" != typeof this.clickRedirect && this.clickRedirect) {
        b = this.clickRedirect.replace(/_CLK_ID_/g, a.id);
        var c = ";link=", d = new RegExp(c, "g"), e = com.adtech.Advert_2_32_3.OPT_TRACKING_KEYS;
        "undefined" != typeof this.thirdPartyCreativeId && (b = b.replace(d, ";" + e.creativeId + "=" + this.thirdPartyCreativeId + c)), "undefined" != typeof this.thirdPartyPlacementId && (b = b.replace(d, ";" + e.placementId + "=" + this.thirdPartyPlacementId + c))
    }
    return b
}, com.adtech.Advert_2_32_3.prototype.verifyClickIdentifier = function(a) {
    return "undefined" != typeof a && null !== a && a ? a : "default"
}, com.adtech.Advert_2_32_3.prototype.logClick = function(a, b, c) {
    var d = this.generateClickUrl(a, b);
    a == com.adtech.Advert_2_32_3.BACKUP_CLICK && (a = com.adtech.Advert_2_32_3.DEFAULT_CLICK);
    var e = this.clickthroughs[a];
    this.thirdPartyClickTrackingCheck(a), this.reportEvent(this.rmEvent.INTERACTIVE_IMPRESSION), this.reportEvent(this.rmEvent.ENGAGEMENT);
    var f = {
        url: "undefined" != typeof b && b ? b: e.dest,
        name: a,
        x: null,
        y: null
    };
    this.extractADRMetaFromEvent(c, f), this.reportADREvent(com.adtech.ADREventRecordGenerator_2_32_3.CLICK, f), this.userHasInteracted=!0, this.hasQualifiedRollover=!0;
    var g = this.generateReportEvent().property("clickName", a).property("clickUrl", d).property("eventId", e.id);
    this.dispatchEvent(g);
    var h = {
        type: this.rmEvent.CLICK,
        data: {
            identifier: a,
            destinationUrl: d,
            meta: c || null
        }
    };
    this.dispatchEvent(h), this.isBuySightAd && this.canRenderRich && this.requestBuySightThirdPartyClickPixels(a)
}, com.adtech.Advert_2_32_3.prototype.thirdPartyEventTrackingCheck = function(a) {
    for (var b = "string" == typeof a ? a : a.name, c = 0; c < this.thirdPartyTracking.length; c++) {
        var d = this.thirdPartyTracking[c];
        (d.onEvent == com.adtech.Advert_2_32_3.TRACK_EVENT_ALL || d.onEvent == com.adtech.Advert_2_32_3.TRACK_EVENT_VIEW && b == com.adtech.Advert_2_32_3.TRACK_EVENT_VIEW || d.onEvent == com.adtech.Advert_2_32_3.TRACK_EVENT_INTERACTION && b == this.rmEvent.ENGAGEMENT || d.onEvent == com.adtech.Advert_2_32_3.TRACK_EVENT_SPECIFIC_EVENT && d.eventName == b || d.onEvent == com.adtech.Advert_2_32_3.TRACK_EVENT_INTERACTIONS_ALL && b != this.rmEvent.INTERACTIVE_IMPRESSION && b != this.rmEvent.ENGAGEMENT && "object" == typeof a && a.interaction) && this.makeThirdPartyRequest(d)
    }
}, com.adtech.Advert_2_32_3.prototype.thirdPartyClickTrackingCheck = function(a) {
    for (var b = 0; b < this.thirdPartyTracking.length; b++) {
        var c = this.thirdPartyTracking[b];
        !this.clickMustMakeThirdPartyRequest(c, a) || c.type != com.adtech.Advert_2_32_3.TRACK_TYPE_PIXEL && c.type != com.adtech.Advert_2_32_3.TRACK_TYPE_JS || this.makeThirdPartyRequest(c)
    }
}, com.adtech.Advert_2_32_3.prototype.addThirdPartyRedirectsToClickUrl = function(a, b) {
    for (var c = 0; c < this.thirdPartyTracking.length; c++) {
        var d = this.thirdPartyTracking[c];
        this.clickMustMakeThirdPartyRequest(d, a) && d.type == com.adtech.Advert_2_32_3.TRACK_TYPE_REDIRECT && d.url && (b = this.replaceServerMacros(d.url) + escape(b))
    }
    return b
}, com.adtech.Advert_2_32_3.prototype.clickMustMakeThirdPartyRequest = function(a, b) {
    return a.onEvent == com.adtech.Advert_2_32_3.TRACK_EVENT_CLICKS_ALL || a.onEvent == com.adtech.Advert_2_32_3.TRACK_EVENT_ALL || a.onEvent == com.adtech.Advert_2_32_3.TRACK_EVENT_SPECIFIC_CLICK && a.eventName == b?!0 : !1
}, com.adtech.Advert_2_32_3.prototype.makeThirdPartyRequest = function(a) {
    var b = this.replaceServerMacros(a.url);
    switch (a.type) {
    case com.adtech.Advert_2_32_3.TRACK_TYPE_PIXEL:
        this.thirdPartyPixelLog(b);
        break;
    case com.adtech.Advert_2_32_3.TRACK_TYPE_JS:
        com.adtech.Utils_2_32_3.requestJs(b)
    }
}, com.adtech.Advert_2_32_3.prototype.processTagVars = function() {
    for (var a in this.tagVars)
        if (this.tagVars.hasOwnProperty(a))
            switch (a) {
            case com.adtech.Advert_2_32_3.BUYSIGHT_KEY:
                this.applyTrackingIdsToTagData(a), this.processBuysightData(this.tagVars[a]);
                break;
            default:
                this.tagVarsPropertyMap.hasOwnProperty(a) && (this[this.tagVarsPropertyMap[a]] = this.tagVars[a])
            }
}, com.adtech.Advert_2_32_3.prototype.applyDynamicContent = function(a) {
    for (var b in this.contentProperties)
        if (this.contentProperties.hasOwnProperty(b) && a.hasOwnProperty(b)) {
            var c = typeof this.contentProperties[b];
            switch (c) {
            case"string":
            case"number":
            case"boolean":
                this.contentProperties[b] = a[b];
                break;
            case"object":
                if (com.adtech.Utils_2_32_3.isArray(this.contentProperties[b]))
                    for (var d = 0; d < a[b].length; d++)
                        if (this.contentProperties[b][d])
                            com.adtech.Utils_2_32_3.mergeObjects(this.contentProperties[b][d], a[b][d]);
                        else {
                            this.contentProperties[b][d] = {};
                            for (var e in this.contentProperties[b][0])
                                this.contentProperties[b][d][e] = this.contentProperties[b][0][e];
                                com.adtech.Utils_2_32_3.mergeObjects(this.contentProperties[b][d], a[b][d])
                            } else 
                                com.adtech.Utils_2_32_3.mergeObjects(this.contentProperties[b], a[b][0])
                            }
                }
    }, com.adtech.Advert_2_32_3.prototype.applyTrackingIdsToTagData = function(a) {
    var b = this.tagVars[a];
    try {
        for (var c in this.contentProperties)
            if (com.adtech.Utils_2_32_3.isArray(this.contentProperties[c]) && b.hasOwnProperty(c) && com.adtech.Utils_2_32_3.isArray(b[c]))
                for (var d = 0; d < this.contentProperties[c].length; d++)
                    b[c][d][com.adtech.Advert_2_32_3.CONTENT_PROPERTY_TRACKING_KEY] = this.contentProperties[c][d][com.adtech.Advert_2_32_3.CONTENT_PROPERTY_TRACKING_KEY], this.contentProperties[c][d][com.adtech.Advert_2_32_3.CONTENT_PROPERTY_PARENT_TRACKING_KEY] && (b[c][d][com.adtech.Advert_2_32_3.CONTENT_PROPERTY_PARENT_TRACKING_KEY] = this.contentProperties[c][d][com.adtech.Advert_2_32_3.CONTENT_PROPERTY_PARENT_TRACKING_KEY]);
        for (var c in b)
            b.hasOwnProperty(c) && (this.contentProperties[c] = b[c])
        } catch (e) {
        this.contentProperties[a] = b
    }
}, com.adtech.Advert_2_32_3.prototype.processBuysightData = function(a) {
    this.isBuySightAd=!0;
    var b = a.items[0];
    if ("1" !== a.status && this.canRenderRich)
        try {
            for (var c = 0; c < a.items.length; c++) {
                var d = "items" + com.adtech.Advert_2_32_3.CONTENT_SEPARATOR + a.items[c][com.adtech.Advert_2_32_3.CONTENT_PROPERTY_TRACKING_KEY] + com.adtech.Advert_2_32_3.CONTENT_SEPARATOR + "landingUrl";
                this.clickthroughs[d] && (this.clickthroughs[d].dest = a.items[c].landingUrl)
            }
            this.clickthroughs[com.adtech.Advert_2_32_3.DEFAULT_CLICK].dest = a.fallbackClickUrl ? a.fallbackClickUrl : b.landingUrl, this.addToThirdPartyTracking(b.impressionUrl, com.adtech.Advert_2_32_3.TRACK_TYPE_PIXEL, com.adtech.Advert_2_32_3.TRACK_EVENT_VIEW);
            for (var e in a.customEvents) {
                if (a.customEvents.hasOwnProperty(e))
                    var f = com.adtech.Utils_2_32_3.isArray(a.customEvents[e]) ? a.customEvents[e]: [a.customEvents[e]];
                    for (var c = 0; c < f.length; c++)
                        this.addToThirdPartyTracking(f[c], com.adtech.Advert_2_32_3.TRACK_TYPE_PIXEL, com.adtech.Advert_2_32_3.TRACK_EVENT_SPECIFIC_EVENT, e)
                    }
        } catch (g) {
        return this.renderBuysightFallback(a), void(window.console && console.error("buysightDataKey could not be processed! Fallback rendered."))
    } else 
        this.renderBuysightFallback(a);
    for (var c = 0; c < b.itUrls.length; c++)
        this.addToThirdPartyTracking(b.itUrls[c], com.adtech.Advert_2_32_3.TRACK_TYPE_PIXEL, com.adtech.Advert_2_32_3.TRACK_EVENT_VIEW)
}, com.adtech.Advert_2_32_3.prototype.renderBuysightFallback = function(a) {
    for (var b = a.items[0].ctUrls, c = 0; c < b.length; c++)
        this.addToThirdPartyTracking(b[c], com.adtech.Advert_2_32_3.TRACK_TYPE_PIXEL, com.adtech.Advert_2_32_3.TRACK_EVENT_CLICKS_ALL);
    this.fallback = a.fallbackImageUrl, this.clickthroughs[com.adtech.Advert_2_32_3.DEFAULT_CLICK].dest = a.fallbackClickUrl ? a.fallbackClickUrl : a.items[0].landingUrl;
    var d = a.fallbackImpressionUrl ? a.fallbackImpressionUrl: a.items[0].impressionUrl;
    this.addToThirdPartyTracking(d, com.adtech.Advert_2_32_3.TRACK_TYPE_PIXEL, com.adtech.Advert_2_32_3.TRACK_EVENT_VIEW), this.forceRenderBackup=!0
}, com.adtech.Advert_2_32_3.prototype.requestBuySightThirdPartyClickPixels = function(a) {
    if (a.indexOf(com.adtech.Advert_2_32_3.CONTENT_SEPARATOR) > 0) {
        var b = a.split(com.adtech.Advert_2_32_3.CONTENT_SEPARATOR);
        if (b.length >= 3)
            for (var c = b[b.length - 2], d = this.getContent("items"), e = 0; e < d.length; e++) {
                var f = d[e];
                if (f[com.adtech.Advert_2_32_3.CONTENT_PROPERTY_TRACKING_KEY] == c) {
                    for (var g = 0; g < f.ctUrls.length; g++)
                        this.thirdPartyPixelLog(f.ctUrls[g]);
                        break
                }
            }
    }
}, com.adtech.Advert_2_32_3.prototype.addToThirdPartyTracking = function(a, b, c, d) {
    var e = {
        url: a,
        type: b,
        onEvent: c,
        eventName: d
    };
    this.thirdPartyTracking.push(e)
}, com.adtech.Advert_2_32_3.prototype.setMacroMap = function() {
    this.macroMap[com.adtech.Advert_2_32_3.MACRO_AD_TIME] = this.adServerVars.uid;
    var a = this.tagVars[com.adtech.Advert_2_32_3.CONFIG_TRACKING_MACROS_KEY];
    if ("object" == typeof a)
        for (var b in a)
            a.hasOwnProperty(b) && (this.macroMap["##" + b + "##"] = a[b])
    }, com.adtech.Advert_2_32_3.prototype.replaceServerMacros = function(a) {
    for (var b in this.macroMap)
        this.macroMap.hasOwnProperty(b) && (a = a.replace(b, this.macroMap[b]));
    return a.replace(com.adtech.Advert_2_32_3.MACRO_TIMESTAMP, + new Date)
}, com.adtech.Advert_2_32_3.prototype.thirdPartyPixelLog = function(a) {
    if (/^http(s)?:\/\//i.test(a)) {
        var b = this.generateReportEvent().property("thirdPartyUrl", a);
        this.dispatchEvent(b)
    }
}, com.adtech.Advert_2_32_3.prototype.checkInPageUnit = function() {
    this.getAssetContainer("main").anchorDiv.parentNode || (clearInterval(this.inpageUnitCheckInterval), this.close())
}, com.adtech.Advert_2_32_3.prototype.checkVisibleContainerCount = function() {
    var a = 0;
    for (var b in this.assetContainers)
        this.assetContainers.hasOwnProperty(b) && this.assetContainers[b].visibleinViewport&&!this.assetContainers[b].closed && a++;
    a && this.windowInFocus ? (this.eventBus.dispatchEvent(this.rmEvent.IN_VIEWPORT), this.startViewTimer()) : (this.eventBus.dispatchEvent(this.rmEvent.OUT_VIEWPORT), this.stopViewTimer())
}, com.adtech.Advert_2_32_3.prototype.setTouchDeviceProperties = function() {
    this.isTouchDevice=!0, this.qualifiedRolloverHandler()
}, com.adtech.Advert_2_32_3.prototype.setDebugOverrides = function() {
    if (this.debugMode) {
        var a = com.adtech.Utils_2_32_3.getQueryStringValue("canvasAssetBasePath");
        a && (this.adServerVars.assetBaseURL = "http://localhost/" + a + "/")
    }
}, com.adtech.Advert_2_32_3.prototype.flushInteractionEventBuffer = function() {
    this.interactionEventBuffer = []
}, com.adtech.Advert_2_32_3.prototype.flushMouseOverEventTimers = function() {
    this.logTimerValue(com.adtech.Advert_2_32_3.ENGAGEMENT_TIMER), this.userHasInteracted && 0 === this.getPlayingVideosCount() && this.logTimerValue(com.adtech.Advert_2_32_3.INTERACTION_TIMER)
}, com.adtech.Advert_2_32_3.prototype.addToPlayingVideos = function(a, b) {
    com.adtech.Utils_2_32_3.inArray(a, this.playingVideos[b]) || (this.playingVideos[b].push(a), this.startTimer(a))
}, com.adtech.Advert_2_32_3.prototype.removeFromPlayingVideos = function(a, b) {
    for (var c = 0; c < this.playingVideos[b].length; c++)
        if (this.playingVideos[b][c] === a) {
            this.playingVideos[b].splice(c, 1), this.logTimerValue(a);
            break
        }
    0 === this.getPlayingVideosCount() && 0 === this.containerMouseOverCount && this.logTimerValue(com.adtech.Advert_2_32_3.INTERACTION_TIMER)
}, com.adtech.Advert_2_32_3.prototype.removeAllPlayingVideosFromContainer = function(a) {
    var b = [], c = [];
    for (var d in this.playingVideos)
        if (this.playingVideos.hasOwnProperty(d))
            for (var e = 0; e < this.playingVideos[d]; e++)
                com.adtech.Utils_2_32_3.inArray(this.playingVideos[d][e], allRunningVideosIds) && c.push(this.playingVideos[d][e]), b.push(this.playingVideos[d][e]);
    for (var f = 0; f < this.playingVideos[a].length; f++) {
        var g = this.playingVideos[a][f];
        com.adtech.Utils_2_32_3.inArray(g, c) || this.removeFromPlayingVideos(g, a)
    }
    this.playingVideos[a] = []
}, com.adtech.Advert_2_32_3.prototype.getPlayingVideosCount = function() {
    var a = 0;
    for (var b in this.playingVideos)
        this.playingVideos.hasOwnProperty(b) && (a += this.playingVideos[b].length);
    return a
}, com.adtech.Advert_2_32_3.prototype.checkContainersExpansionState = function() {
    for (var a in this.assetContainers)
        if (this.assetContainers.hasOwnProperty(a)&&!this.assetContainers[a].closed && this.assetContainers[a].isExpanded)
            return void this.startTimer(com.adtech.Advert_2_32_3.EXPANSION_TIMER);
    this.logTimerValue(com.adtech.Advert_2_32_3.EXPANSION_TIMER)
}, com.adtech.Advert_2_32_3.prototype.extractADRMetaFromEvent = function(a, b) {
    a && "number" == typeof a.__mouseX && "number" == typeof a.__mouseY && (b.x = a.__mouseX, b.y = a.__mouseY)
}, com.adtech.Advert_2_32_3.prototype.reportADREvent = function(a, b) {
    if (this.richView && "undefined" != typeof this.adrEventRecordGenerator) {
        var c = this.generateReportEvent().property("adrType", a).property("adrEventData", this.adrEventData).property("formatId", this.formatId).property("productFamilyId", this.productFamilyId).property("templateId", this.templateId).property("regionId", this.adServerVars.regionId).property("advertiserId", this.adServerVars.advertiserId).property("campaignId", this.adServerVars.campaignId).property("publisherId", this.adServerVars.publisherId).property("regionId", this.adServerVars.regionId).property("width", this.getAssetContainer("main").width).property("height", this.getAssetContainer("main").height).property("eventEndpoint", this.adrEventEndpoint);
        for (var d in b)
            b.hasOwnProperty(d) && c.property(d, b[d]);
        this.dispatchEvent(this.adrEventRecordGenerator.generateADRData(c))
    }
}, com.adtech.Advert_2_32_3.prototype.containerViewHandler = function() {
    this.canMeasureViewExposure && this.checkVisibleContainerCount()
}, com.adtech.Advert_2_32_3.prototype.windowFocusEventHandler = function(a) {
    this.windowInFocus = a.target.windowInFocus, this.checkVisibleContainerCount()
}, com.adtech.Advert_2_32_3.prototype.windowScrollEventHandler = function(a) {
    this.windowInFocus = a.target.windowInFocus=!0
}, com.adtech.Advert_2_32_3.prototype.touchStartHandler = function() {
    this.setTouchDeviceProperties()
}, com.adtech.Advert_2_32_3.prototype.containerCloseHandler = function(a) {
    this.removeAllPlayingVideosFromContainer(a.target.id), this.checkContainersExpansionState();
    for (var b in this.assetContainers)
        if (this.assetContainers.hasOwnProperty(b)&&!this.assetContainers[b].closed)
            return;
    this.flushTimers()
}, com.adtech.Advert_2_32_3.prototype.containerHideHandler = function(a) {
    this.removeAllPlayingVideosFromContainer(a.target.id)
}, com.adtech.Advert_2_32_3.prototype.mouseOverEventHandler = function() {
    if (0 === this.containerMouseOverCount) {
        var a = this;
        this.interactionEventBufferFlushTimer && clearTimeout(this.interactionEventBufferFlushTimer), this.flushMouseOverEventsTimeout && clearTimeout(this.flushMouseOverEventsTimeout), this.engagementCheckTimer = setTimeout(function() {
            a.startEngagementTimer()
        }, this.engagementThreshold), this.qualifiedRolloverEventTimer = setTimeout(function() {
            a.qualifiedRolloverHandler()
        }, this.qualifiedRolloverThreshold)
    }
    this.containerMouseOverCount++
}, com.adtech.Advert_2_32_3.prototype.mouseOutEventHandler = function() {
    if (1 === this.containerMouseOverCount) {
        var a = this;
        this.engagementCheckTimer && (this.stopTimer(com.adtech.Advert_2_32_3.ENGAGEMENT_TIMER), clearTimeout(this.engagementCheckTimer)), this.qualifiedRolloverEventTimer && clearTimeout(this.qualifiedRolloverEventTimer), this.flushMouseOverEventsTimeout = setTimeout(function() {
            a.flushMouseOverEventTimers()
        }, com.adtech.Advert_2_32_3.MOUSE_OVER_TIMERS_FLUSH_TIMEOUT), this.hasQualifiedRollover || (this.interactionEventBufferFlushTimer = setTimeout(function() {
            a.flushInteractionEventBuffer()
        }, 500))
    }
    this.containerMouseOverCount--
}, com.adtech.Advert_2_32_3.prototype.qualifiedRolloverHandler = function() {
    for (this.hasQualifiedRollover=!0; this.interactionEventBuffer.length > 0;)
        this.reportEvent(this.interactionEventBuffer.shift());
    this.userHasInteracted ? this.startTimer(com.adtech.Advert_2_32_3.INTERACTION_TIMER) : this.interactionStartTime = (new Date).getTime() - this.qualifiedRolloverThreshold
}, com.adtech.Advert_2_32_3.prototype.backupImageRenderEventHandler = function() {
    this.rendered || (this.rendered=!0, this.reportEvent(this.rmEvent.BACKUP_VIEW), this.logViewabilityAvailability(), this.trackViewOnEngagement || this.trackViewEvents())
}, com.adtech.Advert_2_32_3.prototype.containerExpansionStateChangeHandler = function() {
    this.checkContainersExpansionState()
}, com.adtech.Advert_2_32_3.prototype.renderEventHandler = function() {
    if (!this.rendered) {
        this.rendered=!0;
        var a = this;
        for (var b in this.assetContainers)
            this.assetContainers.hasOwnProperty(b) && this.assetContainers[b].removeEventListener(this.rmEvent.RENDER, this.renderEventHandler, this);
        this.dispatchEvent(this.rmEvent.RENDER), this.trackViewOnEngagement ? this.reportEvent(this.rmEvent.RENDER) : this.trackViewEvents(), this.logViewabilityAvailability(), this.startTimer(com.adtech.Advert_2_32_3.DISPLAY_TIMER), this.checkContainersExpansionState(), com.adtech.Utils_2_32_3.registerNativeEventHandler(window.self, "beforeunload", com.adtech.Utils_2_32_3.createClosure(this, this.flushTimers));
        var c = this.getAssetContainer("main");
        if (c && c instanceof com.adtech.DivContainer_2_32_3&&!("undefined" != typeof com.adtech.FloatingDivContainer_2_32_3 && c instanceof com.adtech.FloatingDivContainer_2_32_3)) {
            var d = 0;
            for (var b in this.assetContainers)
                this.assetContainers.hasOwnProperty(b) && d++;
            d > 1 && (this.inpageUnitCheckInterval = setInterval(function() {
                a.checkInPageUnit()
            }, com.adtech.Advert_2_32_3.INPAGE_UNIT_CHECK_INTERVAL))
        }
    }
}, com.adtech.Advert_2_32_3.prototype.videoEventHandler = function(a) {
    for (var b = 0; b < this.videoReportingIdentifiers.length; b++) {
        var c = this.videoReportingIdentifiers[b], d = c + com.adtech.Advert_2_32_3.VIDEO_EVENT_VIEW, e = a.target instanceof com.adtech.HtmlContent_2_32_3 ? a.target.parent.id: a.target.id;
        switch (a.type) {
        case c + com.adtech.Advert_2_32_3.VIDEO_EVENT_START:
            this.reportEvent(new this.rmEvent(this.rmEvent.VIDEO_IMPRESSION).property("forceLog", !0));
        case c + com.adtech.Advert_2_32_3.VIDEO_EVENT_RESUME:
            this.addToPlayingVideos(d, e);
            break;
        case c + com.adtech.Advert_2_32_3.VIDEO_EVENT_PAUSE:
        case c + com.adtech.Advert_2_32_3.VIDEO_EVENT_COMPLETE:
            com.adtech.Utils_2_32_3.inArray(d, this.playingVideos[e]) && (a.forceLog=!0), this.removeFromPlayingVideos(d, e)
        }
    }
}, com.adtech.Advert_2_32_3.prototype.adrDataLoadedHandler = function(a) {
    this.adrDataLoader = a.target, this.adrEventData = a.dcoData, this.adrEventEndpoint = a.eventEndpoint;
    for (var b = a.thirdPartyTracking[com.adtech.Advert_2_32_3.TRACK_EVENT_VIEW], c = a.thirdPartyTracking[com.adtech.Advert_2_32_3.TRACK_EVENT_CLICKS_ALL], d = 0; d < b.length; d++)
        this.addToThirdPartyTracking(b[d], com.adtech.Advert_2_32_3.TRACK_TYPE_PIXEL, com.adtech.Advert_2_32_3.TRACK_EVENT_VIEW);
    for (var d = 0; d < c.length; d++)
        this.addToThirdPartyTracking(c[d], com.adtech.Advert_2_32_3.TRACK_TYPE_PIXEL, com.adtech.Advert_2_32_3.TRACK_EVENT_CLICKS_ALL);
    this.applyDynamicContent(a.contentProperties), this.adrContentMap = a.contentMap, a.reqClickUrl && (this.pubVars.clickRedirect = a.reqClickUrl), this.eventBus.dispatchEvent(this.rmEvent.SERVE)
}, com.adtech.Advert_2_32_3.prototype.adrDataErrorHandler = function() {
    this.richView=!1;
    var a = this.getAssetContainer("main");
    a instanceof com.adtech.DivContainer_2_32_3&&!(a instanceof com.adtech.FloatingDivContainer_2_32_3) && (a.anchorDiv.innerHTML = this.getBackupImageHTML(this.fallback), this.trackViewOnEngagement || this.trackViewEvents(), this.reportEvent(this.rmEvent.BACKUP_VIEW), this.rendered=!0, this.dispatchEvent(this.rmEvent.RENDER))
}, com.adtech.Advert_2_32_3.prototype.customEventHandler = function(a) {
    this.reportEvent(a);
    for (var b = 0; b < this.eventActions.length; b++) {
        var c = this.eventActions[b];
        if (c.name == a.type) {
            var d, e = a.target instanceof com.adtech.HtmlContent_2_32_3 ? a.target.parent.id: a.target.id;
            switch (c.target) {
            case com.adtech.Advert_2_32_3.ACTION_TARGET_SELF:
                d = e;
                break;
            default:
                d = c.target
            }
            if (c.source == com.adtech.Advert_2_32_3.ACTION_TARGET_WILDCARD || c.source == e || c.source == com.adtech.Advert_2_32_3.ACTION_TARGET_SELF && e == d) {
                var f = "object" == typeof a.meta ? [a.meta]: [], g = this.getAssetContainer(d);
                g[c.action].apply(g, f)
            }
        }
    }
}, com.adtech.Advert_2_32_3 = com.adtech.Advert_2_32_3 || function() {}, com.adtech.Advert_2_32_3.ENGAGEMENT_THRESHOLD = 2e3, com.adtech.Advert_2_32_3.QUALIFIED_ROLLOVER_THRESHOLD = 500, com.adtech.Advert_2_32_3.VIEWABLE_IMPRESSION_THRESHOLD = 1e3, com.adtech.Advert_2_32_3.MOUSE_OVER_TIMERS_FLUSH_TIMEOUT = 1e3, com.adtech.Advert_2_32_3.TIMER_LOGGING_THRESHOLD = 9e5, com.adtech.Advert_2_32_3.INPAGE_UNIT_CHECK_INTERVAL = 1e3, com.adtech.Advert_2_32_3.TRACK_TYPE_PIXEL = "1_PIXEL", com.adtech.Advert_2_32_3.TRACK_TYPE_JS = "JS_URL", com.adtech.Advert_2_32_3.TRACK_TYPE_REDIRECT = "REDIRECT_URL", com.adtech.Advert_2_32_3.TRACK_EVENT_VIEW = "VIEW", com.adtech.Advert_2_32_3.TRACK_EVENT_INTERACTION = "INTERACTION", com.adtech.Advert_2_32_3.TRACK_EVENT_SPECIFIC_EVENT = "SPECIFIC_EVENT", com.adtech.Advert_2_32_3.TRACK_EVENT_SPECIFIC_CLICK = "SPECIFIC_CLICK", com.adtech.Advert_2_32_3.TRACK_EVENT_CLICKS_ALL = "CLICKS_ALL", com.adtech.Advert_2_32_3.TRACK_EVENT_ALL = "EVENTS_ALL", com.adtech.Advert_2_32_3.TRACK_EVENT_INTERACTIONS_ALL = "INTERACTIONS_ALL", com.adtech.Advert_2_32_3.DISPLAY_TIMER = "displayTimer", com.adtech.Advert_2_32_3.VIEW_TIMER = "viewTimer", com.adtech.Advert_2_32_3.ENGAGEMENT_TIMER = "engagementTimer", com.adtech.Advert_2_32_3.INTERACTION_TIMER = "interactionTimer", com.adtech.Advert_2_32_3.EXPANSION_TIMER = "expansionTimer", com.adtech.Advert_2_32_3.DEFAULT_CLICK = "default", com.adtech.Advert_2_32_3.BACKUP_CLICK = "backupImageClickthrough", com.adtech.Advert_2_32_3.ALT_TEXT_VARIABLE_KEY = "Backup Alt Text", com.adtech.Advert_2_32_3.ACTION_TARGET_SELF = "self", com.adtech.Advert_2_32_3.ACTION_TARGET_WILDCARD = "*", com.adtech.Advert_2_32_3.MACRO_AD_TIME = "_ADTIME_", com.adtech.Advert_2_32_3.MACRO_TIMESTAMP = "_TIMESTAMP_", com.adtech.Advert_2_32_3.CONFIG_TRACKING_MACROS_KEY = "trackingMacros", com.adtech.Advert_2_32_3.BUYSIGHT_KEY = "buysightDataKey", com.adtech.Advert_2_32_3.CONTENT_SEPARATOR = ": ", com.adtech.Advert_2_32_3.CONTENT_PROPERTY_TRACKING_KEY = "Tracking Id", com.adtech.Advert_2_32_3.CONTENT_PROPERTY_PARENT_TRACKING_KEY = "Parent Tracking Id", com.adtech.Advert_2_32_3.VIDEO_EVENT_VIEW = "Video View", com.adtech.Advert_2_32_3.VIDEO_EVENT_START = "Video Start", com.adtech.Advert_2_32_3.VIDEO_EVENT_RESUME = "Video Resume", com.adtech.Advert_2_32_3.VIDEO_EVENT_REPLAY = "Video Replay", com.adtech.Advert_2_32_3.VIDEO_EVENT_PAUSE = "Video Pause", com.adtech.Advert_2_32_3.VIDEO_EVENT_PLAY = "Video Play", com.adtech.Advert_2_32_3.VIDEO_EVENT_COMPLETE = "Video Complete", com.adtech.Advert_2_32_3.OPT_TRACKING_KEYS = {
    creativeId: "kvu.3rd-creative",
    placementId: "kvu.3rd-plc"
}, com.adtech.AssetContainerAlignmentOption_2_32_3 = function() {}, com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_LEFT = "pageLeft", com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_TOP = "pageTop", com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_RIGHT = "pageRight", com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_BOTTOM = "pageBottom", com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_CENTER = "pageCenter", com.adtech.AssetContainerAlignmentOption_2_32_3.AD_SLOT_LEFT = "adSlotLeft", com.adtech.AssetContainerAlignmentOption_2_32_3.AD_SLOT_TOP = "adSlotTop", com.adtech.AssetContainer_2_32_3 = function(a, b, c, d, e) {
    com.adtech.AssetContainer_2_32_3.supa.constructor.call(this), this.parent = a, this.windowContext = window, this.id = c, this.visibleinViewport=!1, this.visible=!1, this.rendered=!1, this.floatOffsetLeft = 0, this.floatOffsetTop = 0, this.closed=!1, this.globalEventBus = e, this.autoCloseTimeout = null, this.adEventBus = d, this.advertAnchorId = com.adtech.AssetContainer_2_32_3.ANCHOR_ID_PREFIX + b.adServerVars.id + "_" + b.adServerVars.uid, this.mouseIsOver=!1, this.content = null, this.rmEvent = com.adtech.RichMediaEvent_2_32_3, this.init(b, d)
}, com.adtech.AssetContainer_2_32_3.ANCHOR_ID_PREFIX = "adtechAdContainer_", com.adtech.AssetContainer_2_32_3.CONTENT_ID_PREFIX = "adtechAdContent_", com.adtech.AssetContainer_2_32_3.DWELL_MIN_VIEWABLE_PERCENTAGE = 50, com.adtech.AssetContainer_2_32_3.ACCEPTABLE_VIEW_AREA = 37500, com.adtech.Utils_2_32_3.extend(com.adtech.AssetContainer_2_32_3, com.adtech.EventDispatcher_2_32_3), com.adtech.AssetContainer_2_32_3.prototype.init = function(a) {
    var b = a.assetContainers[this.id];
    for (var c in b)
        if (b.hasOwnProperty(c))
            switch (c) {
            case"isExpandable":
                this[c] = b.type == com.adtech.AssetContainerFactory_2_32_3.INLINE_DIV?!(b.contentWidth === b.contractedWidth && b.contentHeight === b.contractedHeight) : b[c];
                break;
            case"isResponsive":
                this[c] = b[c]===!0 && b.contentType == com.adtech.ContentFactory_2_32_3.HTML && b.type == com.adtech.AssetContainerFactory_2_32_3.INLINE_DIV;
                break;
            default:
                this[c] = b[c]
            }
    this.uid = this.id.replace(/[ \/\.-]/g, "_") + "_" + a.adServerVars.uid, this.contentId = com.adtech.AssetContainer_2_32_3.CONTENT_ID_PREFIX + this.uid, this.addRenderEventListeners(), this.content = com.adtech.ContentFactory_2_32_3.getContent(this.contentType, this, a, this.id, this.adEventBus, this.globalEventBus, this.contentId), com.adtech.Utils_2_32_3.getConfigOverride(a, "forcePageLoad") && this.content.forcePageLoad(), com.adtech.Utils_2_32_3.addCustomEventHandler(this.adEventBus, this, a.events)
}, com.adtech.AssetContainer_2_32_3.prototype.addRenderEventListeners = function() {
    this.globalEventBus.addEventListener(this.rmEvent.DOM_LOAD, this.coreLibEventHandler, this), this.globalEventBus.addEventListener(this.rmEvent.PAGE_LOAD, this.coreLibEventHandler, this), this.adEventBus.addEventListener(this.rmEvent.SERVE, this.coreLibEventHandler, this), this.adEventBus.addEventListener(this.rmEvent.ENGAGEMENT, this.coreLibEventHandler, this)
}, com.adtech.AssetContainer_2_32_3.prototype.checkRenderEvent = function(a) {
    this.rendered || a != this.renderEvent || this.render()
}, com.adtech.AssetContainer_2_32_3.prototype.render = function() {
    this.rendered || this.closed || (this.rendered=!0, this.visible=!0, this.dispatchEvent(this.rmEvent.RENDER))
}, com.adtech.AssetContainer_2_32_3.prototype.getContent = function() {
    return this.content
}, com.adtech.AssetContainer_2_32_3.prototype.dispatchMouseOverEvent = function() {
    this.mouseIsOver || (this.dispatchEvent(this.rmEvent.MOUSE_OVER), this.mouseIsOver=!0)
}, com.adtech.AssetContainer_2_32_3.prototype.dispatchMouseOutEvent = function() {
    this.mouseIsOver && (this.dispatchEvent(this.rmEvent.MOUSE_OUT), this.mouseIsOver=!1)
}, com.adtech.AssetContainer_2_32_3.prototype.dispatchInViewportEvent = function() {
    this.visibleinViewport || (this.visibleinViewport=!0, this.dispatchEvent(this.rmEvent.IN_VIEWPORT))
}, com.adtech.AssetContainer_2_32_3.prototype.dispatchOutViewportEvent = function() {
    this.visibleinViewport && (this.visibleinViewport=!1, this.dispatchEvent(this.rmEvent.OUT_VIEWPORT))
}, com.adtech.AssetContainer_2_32_3.prototype.setFloatingOffsets = function() {
    var a = com.adtech.Utils_2_32_3.getViewportDims(), b = com.adtech.Utils_2_32_3.calculateAbsolutePosition(document.getElementById(this.advertAnchorId));
    switch (this.xRel) {
    case com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_RIGHT:
        this.floatOffsetLeft = a.w;
        break;
    case com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_CENTER:
        var c = a.w / 2;
        this.floatOffsetLeft = this.x < 0 && c + this.x < 0?-this.x : c;
        break;
    case com.adtech.AssetContainerAlignmentOption_2_32_3.AD_SLOT_LEFT:
        this.floatOffsetLeft = b.x;
        break;
    default:
        this.floatOffsetLeft = 0
    }
    switch (this.yRel) {
    case com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_BOTTOM:
        this.floatOffsetTop = a.h;
        break;
    case com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_CENTER:
        var d = a.h / 2;
        this.floatOffsetTop = this.y < 0 && d + this.y < 0?-this.y : d;
        break;
    case com.adtech.AssetContainerAlignmentOption_2_32_3.AD_SLOT_TOP:
        this.floatOffsetTop = b.y;
        break;
    default:
        this.floatOffsetTop = 0
    }
}, com.adtech.AssetContainer_2_32_3.prototype.show = function() {
    this.visible=!0
}, com.adtech.AssetContainer_2_32_3.prototype.hide = function() {
    this.visible=!1, this.dispatchEvent(this.rmEvent.HIDE)
}, com.adtech.AssetContainer_2_32_3.prototype.expand = function() {
    this.isExpandable && (this.isExpanded=!0, this.dispatchEvent(this.rmEvent.EXPANDED))
}, com.adtech.AssetContainer_2_32_3.prototype.contract = function() {
    this.isExpandable && (this.isExpanded=!1, this.dispatchEvent(this.rmEvent.CONTRACTED))
}, com.adtech.AssetContainer_2_32_3.prototype.reload = function() {}, com.adtech.AssetContainer_2_32_3.prototype.close = function() {
    this.dispatchMouseOutEvent(), this.content.remove(), this.content = null, this.rendered=!1, this.closed=!0, this.dispatchEvent(this.rmEvent.CLOSE)
}, com.adtech.AssetContainer_2_32_3.prototype.coreLibEventHandler = function(a) {
    this.checkRenderEvent(a.type), a.type == this.rmEvent.ENGAGEMENT && this.autoCloseTimeout && clearTimeout(this.autoCloseTimeout)
}, com.adtech.AssetContainer_2_32_3.prototype.customEventHandler = function(a) {
    a.target != this.content && this.checkRenderEvent(a.type)
}, com.adtech.DivContainer_2_32_3 = function(a, b, c, d, e) {
    this.isExpanded=!1, this.visible=!1, this.anchorDiv = null, this.div = null, this.anchorDivAttemptIntervals = [10, 20, 50, 100, 200, 500], this.anchorStartWidth = 1, this.anchorStartHeight = 1, this.dimensionUnits = "px", this.preventRender=!1, this.forceRender=!1, com.adtech.DivContainer_2_32_3.supa.constructor.call(this, a, b, c, d, e), this.setAnchorDivStyle(), this.createAnchorDiv()
}, com.adtech.Utils_2_32_3.extend(com.adtech.DivContainer_2_32_3, com.adtech.AssetContainer_2_32_3), com.adtech.DivContainer_2_32_3.SCALE_UP = "u", com.adtech.DivContainer_2_32_3.SCALE_DOWN = "d", com.adtech.DivContainer_2_32_3.ANIMATION_PERIOD = 40, com.adtech.DivContainer_2_32_3.ZINDEX_BUMP = 10, com.adtech.DivContainer_2_32_3.DOM_LOAD_TIMEOUT = 2e3, com.adtech.DivContainer_2_32_3.prototype.init = function(a) {
    com.adtech.DivContainer_2_32_3.supa.init.call(this, a), this.parent.renderingInFiF && com.adtech.IframeUtils_2_32_3.topOrSelf() != self && (this.containingIframe = com.adtech.IframeUtils_2_32_3.getTargetParentIframeEle(top)), this.setCoreLibPageEventListeners(!0), this.isExpandable && (this.isExpanded=!this.startContracted), this.setClipStyle();
    var b = com.adtech.Utils_2_32_3.getConfigOverride(a, "domLoadTimeout");
    this.DOMLoadTimeout = null != b ? b : com.adtech.DivContainer_2_32_3.DOM_LOAD_TIMEOUT, this.parent.renderingInSafeframe && com.adtech.SafeframeUtils_2_32_3.register(this.width, this.height, com.adtech.Utils_2_32_3.createClosure(this, this.statusEventHandler))
}, com.adtech.DivContainer_2_32_3.prototype.getAnchorDivDisplayStyle = function() {
    return com.adtech.Utils_2_32_3.isIEQuirksMode() ? "inline" : "inline-block"
}, com.adtech.DivContainer_2_32_3.prototype.setClipStyle = function() {
    this.initialClipStyle = this.isExpandable && this.startContracted ? this.getClippedStyle() : this.getUnclippedStyle()
}, com.adtech.DivContainer_2_32_3.prototype.setCoreLibPageEventListeners = function(a) {
    for (var b = [this.rmEvent.PAGE_RESIZE, this.rmEvent.PAGE_SCROLL, this.rmEvent.ORIENTATION_CHANGE], c = 0; c < b.length; c++)
        a ? this.globalEventBus.addEventListener(b[c], this.coreLibEventHandler, this) : this.globalEventBus.removeEventListener(b[c], this.coreLibEventHandler, this)
}, com.adtech.DivContainer_2_32_3.prototype.calculateAbsolutePosition = function() {
    var a = com.adtech.Utils_2_32_3.calculateAbsolutePosition(this.containingIframe ? this.containingIframe : this.div);
    this.absTop = a.y, this.absLeft = a.x
}, com.adtech.DivContainer_2_32_3.prototype.getVisibleDims = function() {
    return this.containingIframe ? {
        w: parseInt(com.adtech.Utils_2_32_3.getComputedStyle(this.containingIframe, "width")),
        h: parseInt(com.adtech.Utils_2_32_3.getComputedStyle(this.containingIframe, "height"))
    } : (this.isResponsive && (this.width = com.adtech.Utils_2_32_3.getComputedStyle(this.div, "width"), this.height = com.adtech.Utils_2_32_3.getComputedStyle(this.div, "height")), this.isExpanded ? {
        w: this.contentWidth,
        h: this.contentHeight
    } : {
        w: this.width,
        h: this.height
    })
}, com.adtech.DivContainer_2_32_3.prototype.getVisiblePos = function() {
    return this.isExpanded ? {
        l: this.absLeft,
        t: this.absTop
    } : {
        l: this.absLeft + this.contractedX,
        t: this.absTop + this.contractedY
    }
}, com.adtech.DivContainer_2_32_3.prototype.setInitialVisibilityState = function() {
    if (this.parent.renderingInSafeframe)
        this.checkAssetVisibility();
    else {
        var a = com.adtech.Utils_2_32_3.getViewportDims(this.parent.renderingInFiF), b = com.adtech.Utils_2_32_3.getPageOffsets(this.parent.renderingInFiF);
        this.calculateAbsolutePosition(), this.checkAssetVisibility(b.x, b.y, a.w, a.h)
    }
}, com.adtech.DivContainer_2_32_3.prototype.checkRenderEvent = function(a) {
    this.rendered || a != this.renderEvent || this.globalEventBus.DOMLoaded || this.addDOMLoadRenderTimeout(), com.adtech.DivContainer_2_32_3.supa.checkRenderEvent.call(this, a)
}, com.adtech.DivContainer_2_32_3.prototype.addDOMLoadRenderTimeout = function() {
    if (!this.globalEventBus.DOMLoaded&&!this.DOMLoadRenderTimeout) {
        var a = this;
        this.DOMLoadRenderTimeout = setTimeout(function() {
            a.forceRendering()
        }, this.DOMLoadTimeout)
    }
}, com.adtech.DivContainer_2_32_3.prototype.forceRendering = function() {
    this.forceRender=!0, "undefined" != typeof com.adtech.swfobject_2_32_3 && com.adtech.swfobject_2_32_3.forceSWFRender()
}, com.adtech.DivContainer_2_32_3.prototype.checkAssetVisibility = function(a, b, c, d) {
    if (this.rendered) {
        var e, f;
        if (this.parent.renderingInSafeframe) {
            var g = com.adtech.SafeframeUtils_2_32_3.getInViewPercentage();
            e = com.adtech.SafeframeUtils_2_32_3.getMaximumAdArea(), f = g / 100 * e
        } else {
            var h = this.getVisibleDims();
            if (a >= this.absLeft + h.w || this.absLeft - a > c)
                var i = 0;
            else 
                var j = a < this.absLeft ? 0: Math.abs(this.absLeft - a), k = Math.max(this.absLeft + h.w - c - a, 0), i = h.w - j - k;
            if (b >= this.absTop + h.h || this.absTop - b > d)
                var l = 0;
            else 
                var m = b < this.absTop ? 0: Math.abs(this.absTop - b), n = Math.max(this.absTop + h.h - d - b, 0), l = h.h - m - n;
            e = h.w * h.h, f = i * l
        }
        this.dispatchViewportEvents(f, e)
    }
}, com.adtech.DivContainer_2_32_3.prototype.dispatchViewportEvents = function(a, b) {
    var c = com.adtech.AssetContainer_2_32_3.DWELL_MIN_VIEWABLE_PERCENTAGE / 100;
    a >= b * c || a >= com.adtech.AssetContainer_2_32_3.ACCEPTABLE_VIEW_AREA || this.stickyY ? this.dispatchInViewportEvent() : this.dispatchOutViewportEvent()
}, com.adtech.DivContainer_2_32_3.prototype.setAnchorDivStyle = function() {
    var a = this.width, b = this.height;
    this.pushesContent&&!this.startContracted && (a = this.contentWidth, b = this.contentHeight), this.isResponsive && (this.dimensionUnits = "%", a = b = this.contentWidth = this.contentHeight = 100), this.anchorDivStyle = "position:relative;width:" + a + this.dimensionUnits + ";height:" + b + this.dimensionUnits + ";display:" + this.getAnchorDivDisplayStyle() + ";z-index:" + this.zIndex + ";", this.anchorStartWidth = a, this.anchorStartHeight = b
}, com.adtech.DivContainer_2_32_3.prototype.createAnchorDiv = function() {
    var a = '<div id="' + this.advertAnchorId + '" style="' + this.anchorDivStyle + '"></div>', b = document.getElementById(this.advertAnchorId);
    if (b)
        this.anchorDiv = b;
    else {
        var c = this.getTargetIframeReference();
        if (c) {
            if (this.anchorDiv = document.createElement("div"), this.anchorDiv.id = this.advertAnchorId, this.anchorDiv.style.position = "relative", this.anchorDiv.style.width = this.anchorStartWidth + this.dimensionUnits, this.anchorDiv.style.height = this.anchorStartHeight + this.dimensionUnits, this.anchorDiv.style.zIndex = this.zIndex, this.type == com.adtech.AssetContainerFactory_2_32_3.INLINE_DIV) {
                this.anchorDiv.style.margin = "0";
                var d = {
                    marginTop: "marginTop",
                    marginRight: "marginRight",
                    marginBottom: "marginBottom",
                    marginLeft: "marginLeft",
                    paddingTop: "marginTop",
                    paddingRight: "marginRight",
                    paddingBottom: "marginBottom",
                    paddingLeft: "marginLeft"
                };
                for (var e in d)
                    if (d.hasOwnProperty(e)) {
                        var f = parseInt(com.adtech.Utils_2_32_3.getComputedStyle(c, e));
                        if (!isNaN(f)) {
                            var g = parseInt(this.anchorDiv.style[d[e]]);
                            this.anchorDiv.style[d[e]] = g + parseInt(f) + "px"
                        }
                    }
                this.anchorDiv.style.display = this.getAnchorDivDisplayStyle(), com.adtech.Utils_2_32_3.isIE()&&!com.adtech.Utils_2_32_3.isIEQuirksMode() && (this.anchorDiv.style.clear = "none"), c.style.width = c.style.height = "0px", c.style.display = "none"
            } else 
                this.anchorDiv.style.display = "none";
            c.parentNode.insertBefore(this.anchorDiv, c)
        } else 
            document.write(a), this.anchorDiv = document.getElementById(this.advertAnchorId)
    }
}, com.adtech.DivContainer_2_32_3.prototype.getTargetIframeReference = function() {
    var a = this.parent.targetIframe;
    return a = "string" == typeof a ? document.getElementById(a) : a
}, com.adtech.DivContainer_2_32_3.prototype.createContentDiv = function() {
    this.isExpandable && this.isMultiDirectionalExpand && this.setMDEOverrides();
    var a = "position:absolute;width:" + this.contentWidth + this.dimensionUnits + ";height:" + this.contentHeight + this.dimensionUnits + ";left:" + this.x + "px;top:" + this.y + "px;";
    this.isExpandable && (a += "clip:" + this.initialClipStyle + ";z-index:" + (this.zIndex + 1) + ";");
    var b = '<div id="' + this.contentId + '" style="' + a + '"></div>', c = parseInt(this.parent.pubVars.overflowFixLevel);
    (isNaN(c) || 0 == c) && (c = 1), this.fixParentOverflow(c), this.anchorDiv.innerHTML = b
}, com.adtech.DivContainer_2_32_3.prototype.getClippedStyle = function() {
    return "rect(" + this.contractedY + "px, " + (this.contractedX + this.contractedWidth) + "px, " + (this.contractedY + this.contractedHeight) + "px, " + this.contractedX + "px)"
}, com.adtech.DivContainer_2_32_3.prototype.getUnclippedStyle = function(a) {
    var b = this.contentWidth, c = this.contentHeight;
    return "object" == typeof a && ("number" == typeof a.width && (b = a.width), "number" == typeof a.height && (c = a.height)), "rect(0px, " + b + "px, " + c + "px, 0px)"
}, com.adtech.DivContainer_2_32_3.prototype.setMDEOverrides = function() {
    var a = com.adtech.Utils_2_32_3.calculateAbsolutePosition(this.anchorDiv), b = (com.adtech.Utils_2_32_3.getDocumentWidth(), com.adtech.Utils_2_32_3.getDocumentHeight()), c = this.content.adConfig.assetContainers.main;
    this.expandsInPlaneY() && (a.y + this.contentHeight < b || a.y < this.contentHeight - this.height ? this.y = this.contractedY = c.y = c.contractedY = 0 : (this.y = c.y = this.height - this.contentHeight, this.contractedY = c.contractedY = Math.abs(this.y), this.pushesContent=!1)), this.expandsInPlaneX() && (a.x > this.contentWidth - this.width ? (this.x = c.x = this.width - this.contentWidth, this.contractedX = c.contractedX = Math.abs(this.x)) : this.x = this.contractedX = c.x = c.contractedX = 0), this.setClipStyle()
}, com.adtech.DivContainer_2_32_3.prototype.expandsInPlaneY = function() {
    return this.isExpandable && this.contentHeight > this.height
}, com.adtech.DivContainer_2_32_3.prototype.expandsInPlaneX = function() {
    return this.isExpandable && this.contentWidth > this.width
}, com.adtech.DivContainer_2_32_3.prototype.fixParentOverflow = function(a) {
    if (!isNaN(a) && 0 != a)
        for (var b = this.anchorDiv.parentNode; b && 0 != a;)
            this.pushesContent ? b.style.height = "auto" : b.style.overflow = "visible", b = b.parentNode, a--
}, com.adtech.DivContainer_2_32_3.prototype.render = function() {
    if (!this.rendered&&!this.closed) {
        if (!this.anchorDiv && (this.anchorDiv = document.getElementById(this.advertAnchorId), !this.anchorDiv)) {
            var a = this;
            return this.anchorDivAttemptIntervals.length > 0 ? void setTimeout(com.adtech.Utils_2_32_3.createClosure(a, a.render), this.anchorDivAttemptIntervals.shift()) : void 0
        }
        this.createContentDiv(), this.div = document.getElementById(this.contentId), this.addMouseListeners(), this.isExpandable && this.isMultiDirectionalExpand&&!this.globalEventBus.DOMLoaded ? this.globalEventBus.addEventListener(this.rmEvent.DOM_LOAD, this.content.render, this.content) : this.content.render(), this.content.addEventListener(com.adtech.RichMediaEvent_2_32_3.READY, com.adtech.Utils_2_32_3.createClosure(this, this.contentReadyHandler)), com.adtech.DivContainer_2_32_3.supa.render.call(this), this.setInitialVisibilityState(), this.isExpandable && this.isExpanded && this.parent.renderingInSafeframe && com.adtech.SafeframeUtils_2_32_3.expand(this, com.adtech.SafeframeUtils_2_32_3.prototype.safeframeObjFromClips(this.getClippedStyle(), this.getUnclippedStyle(), this.pushesContent));
        var a = this;
        setTimeout(com.adtech.Utils_2_32_3.createClosure(a, a.setInitialVisibilityState), com.adtech.Advert_2_32_3.VIEWABLE_IMPRESSION_THRESHOLD - 300)
    }
}, com.adtech.DivContainer_2_32_3.prototype.addMouseListeners = function() {
    this.contentType == com.adtech.ContentFactory_2_32_3.FLASH && "window" == this.wmode ? (this.content.handleMouseEvents(), this.content.addEventListener(this.rmEvent.MOUSE_OVER, this.dispatchMouseOverEvent, this), this.content.addEventListener(this.rmEvent.MOUSE_OUT, this.dispatchMouseOutEvent, this)) : (com.adtech.Utils_2_32_3.registerNativeEventHandler(this.div, "mouseover", com.adtech.Utils_2_32_3.createClosure(this, this.dispatchMouseOverEvent)), com.adtech.Utils_2_32_3.registerNativeEventHandler(this.div, "mouseout", com.adtech.Utils_2_32_3.createClosure(this, this.dispatchMouseOutEvent)))
}, com.adtech.DivContainer_2_32_3.prototype.anchorDivScaleStep = function() {
    var a = this.anchorDivScaleMode == com.adtech.DivContainer_2_32_3.SCALE_UP?!0 : !1, b = parseInt(this.anchorDiv.style.height), c = a ? b + this.animationStepDelta : b - this.animationStepDelta;
    if (this.currentAnimationStep < this.totalAnimationSteps)
        this.resizeContainer(c), this.anchorDiv.style.height = c + "px", this.currentAnimationStep++;
    else {
        var d = a ? this.contentHeight: this.height;
        this.resizeContainer(d), this.anchorDiv.style.height = d + "px", this.resetAnimationInterval(), a || this.finaliseContract()
    }
}, com.adtech.DivContainer_2_32_3.prototype.resetAnimationInterval = function() {
    this.animationIntervalTimer && (clearInterval(this.animationIntervalTimer), this.animationIntervalTimer = null)
}, com.adtech.DivContainer_2_32_3.prototype.animateAnchorDivScaling = function() {
    this.heightDelta = this.contentHeight - this.height, this.animationStepDelta = Math.floor(this.heightDelta * com.adtech.DivContainer_2_32_3.ANIMATION_PERIOD / (1e3 * this.animationDuration)), this.totalAnimationSteps = Math.floor(this.heightDelta / this.animationStepDelta), this.currentAnimationStep = 0, this.animationIntervalTimer = setInterval(com.adtech.Utils_2_32_3.createClosure(this, this.anchorDivScaleStep), com.adtech.DivContainer_2_32_3.ANIMATION_PERIOD)
}, com.adtech.DivContainer_2_32_3.prototype.expandAnchorDiv = function() {
    this.expandAnimationDuration ? (this.resetAnimationInterval(), this.anchorDivScaleMode = com.adtech.DivContainer_2_32_3.SCALE_UP, this.animationDuration = this.expandAnimationDuration, this.animateAnchorDivScaling()) : this.anchorDiv.style.height = this.contentHeight + "px"
}, com.adtech.DivContainer_2_32_3.prototype.contractAnchorDiv = function() {
    this.contractAnimationDuration ? (this.resetAnimationInterval(), this.anchorDivScaleMode = com.adtech.DivContainer_2_32_3.SCALE_DOWN, this.animationDuration = this.contractAnimationDuration, this.animateAnchorDivScaling()) : (this.anchorDiv.style.height = this.height + "px", this.finaliseContract())
}, com.adtech.DivContainer_2_32_3.prototype.resizeContainer = function() {}, com.adtech.DivContainer_2_32_3.prototype.show = function() {
    com.adtech.DivContainer_2_32_3.supa.show.call(this)
}, com.adtech.DivContainer_2_32_3.prototype.hide = function() {
    com.adtech.DivContainer_2_32_3.supa.hide.call(this)
}, com.adtech.DivContainer_2_32_3.prototype.expand = function(a) {
    if (this.rendered && this.isExpandable&&!this.isExpanded) {
        this.pushesContent&&!this.parent.renderingInSafeframe && this.expandAnchorDiv(), com.adtech.DivContainer_2_32_3.supa.expand.call(this);
        var b = this.getUnclippedStyle(a);
        this.div.style.clip = b, "undefined" != typeof com.adtech.FlashContent_2_32_3 && this.content instanceof com.adtech.FlashContent_2_32_3 && this.content.resizeObject(this.rectObjFromClip(b)), this.anchorDiv.style.zIndex = this.zIndex + com.adtech.DivContainer_2_32_3.ZINDEX_BUMP, this.div.style.zIndex = this.zIndex + 1 + com.adtech.DivContainer_2_32_3.ZINDEX_BUMP, this.calculateAbsolutePosition(), this.parent.renderingInSafeframe && com.adtech.SafeframeUtils_2_32_3.expand(this, com.adtech.SafeframeUtils_2_32_3.prototype.safeframeObjFromClips(this.getClippedStyle(), this.getUnclippedStyle(a), this.pushesContent))
    }
}, com.adtech.DivContainer_2_32_3.prototype.rectObjFromClip = function(a) {
    var b = /rect\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\)/i, c = b.exec(a);
    return {
        top: c[1],
        right: c[2],
        bottom: c[3],
        left: c[4]
    }
}, com.adtech.DivContainer_2_32_3.prototype.contract = function() {
    if (this.rendered && this.isExpandable && this.isExpanded) {
        if (this.pushesContent&&!this.parent.renderingInSafeframe)
            return void this.contractAnchorDiv();
        this.finaliseContract()
    }
}, com.adtech.DivContainer_2_32_3.prototype.reload = function() {
    this.rendered ? (this.content.remove(), this.content.forcePageLoad(), this.content.render(), this.show()) : this.render()
}, com.adtech.DivContainer_2_32_3.prototype.finaliseContract = function() {
    com.adtech.DivContainer_2_32_3.supa.contract.call(this);
    var a = this.getClippedStyle();
    this.div.style.clip = a, "undefined" != typeof com.adtech.FlashContent_2_32_3 && this.content instanceof com.adtech.FlashContent_2_32_3 && this.content.resizeObject(this.rectObjFromClip(a)), this.anchorDiv.style.zIndex = this.zIndex, this.div.style.zIndex = this.zIndex + 1, this.calculateAbsolutePosition(), this.parent.renderingInSafeframe && com.adtech.SafeframeUtils_2_32_3.collapse(this)
}, com.adtech.DivContainer_2_32_3.prototype.close = function() {
    if (this.rendered&&!this.closed) {
        this.setCoreLibPageEventListeners(!1), com.adtech.DivContainer_2_32_3.supa.close.call(this);
        try {
            this.div.parentNode == this.anchorDiv ? this.anchorDiv.parentNode.removeChild(this.anchorDiv) : (this.div.parentNode.removeChild(this.div), this.anchorDiv.parentNode.removeChild(this.anchorDiv))
        } catch (a) {}
    }
}, com.adtech.DivContainer_2_32_3.prototype.statusEventHandler = function(a) {
    "geom-update" == a && this.checkAssetVisibility()
}, com.adtech.DivContainer_2_32_3.prototype.coreLibEventHandler = function(a) {
    if (com.adtech.DivContainer_2_32_3.supa.coreLibEventHandler.call(this, a), this.rendered)
        switch (a.type) {
        case this.rmEvent.PAGE_SCROLL:
        case this.rmEvent.PAGE_RESIZE:
            this.calculateAbsolutePosition(), this.checkAssetVisibility(a.offsetX, a.offsetY, a.width, a.height);
        case this.rmEvent.DOM_LOAD:
        case this.rmEvent.PAGE_LOAD:
            this.setInitialVisibilityState()
        }
}, com.adtech.DivContainer_2_32_3.prototype.contentReadyHandler = function() {
    if ("undefined" != typeof com.adtech.FlashContent_2_32_3 && this.content instanceof com.adtech.FlashContent_2_32_3 && this.isExpandable && this.startContracted && this.content.resizeObject(this.rectObjFromClip(this.getClippedStyle())), com.adtech.Utils_2_32_3.isChrome()) {
        var a = this.div.style.left, b = this.div.style.top;
        this.div.style.left = "", this.div.style.top = "";
        var c = this;
        setTimeout(function() {
            c.div.style.left = a, c.div.style.top = b
        }, 10)
    }
}, com.adtech.FloatingDivContainer_2_32_3 = function(a, b, c, d, e) {
    this.isPercentageRendered=!1, com.adtech.FloatingDivContainer_2_32_3.supa.constructor.call(this, a, b, c, d, e)
}, com.adtech.Utils_2_32_3.extend(com.adtech.FloatingDivContainer_2_32_3, com.adtech.DivContainer_2_32_3), com.adtech.FloatingDivContainer_2_32_3.MODAL_COLOUR_CONTENT_VARIABLE = "Overlay background colour", com.adtech.FloatingDivContainer_2_32_3.MODAL_OPACITY_CONTENT_VARIABLE = "Overlay opacity", com.adtech.FloatingDivContainer_2_32_3.DEFAULT_MODAL_COLOUR = "#000000", com.adtech.FloatingDivContainer_2_32_3.DEFAULT_MODAL_OPACITY = "0.4", com.adtech.FloatingDivContainer_2_32_3.prototype.init = function(a) {
    var b = a.assetContainers[this.id];
    ("%" == b.contentWidthUnit || "%" == b.contentHeightUnit) && (a.assetContainers[this.id].isExpandable=!1), com.adtech.FloatingDivContainer_2_32_3.supa.init.call(this, a)
}, com.adtech.FloatingDivContainer_2_32_3.prototype.setInitialDimensions = function() {
    var a = com.adtech.Utils_2_32_3.getViewportDims();
    this.originalContentWidth = this.contentWidth, this.originalContentHeight = this.contentHeight, this.contentWidth = "%" != this.contentWidthUnit || this.isExpandable ? this.contentWidth : com.adtech.Utils_2_32_3.calculatePercentValue(a.w, this.contentWidth), this.contentHeight = "%" != this.contentHeightUnit || this.isExpandable ? this.contentHeight : com.adtech.Utils_2_32_3.calculatePercentValue(a.h, this.contentHeight), "%" != this.contentHeightUnit && "%" != this.contentWidthUnit || this.isExpandable || (this.isPercentageRendered=!0, this.initialClipStyle = this.getUnclippedStyle())
}, com.adtech.FloatingDivContainer_2_32_3.prototype.checkRenderEvent = function(a) {
    if (!this.preventRender&&!this.rendered && a == this.renderEvent) {
        if (!this.globalEventBus.DOMLoaded)
            return this.renderEvent = this.rmEvent.DOM_LOAD, void this.addDOMLoadRenderTimeout();
        com.adtech.FloatingDivContainer_2_32_3.supa.checkRenderEvent.call(this, a)
    }
}, com.adtech.FloatingDivContainer_2_32_3.prototype.forceRendering = function() {
    com.adtech.FloatingDivContainer_2_32_3.supa.forceRendering.call(this), this.render()
}, com.adtech.FloatingDivContainer_2_32_3.prototype.setAnchorDivStyle = function() {
    var a = targetHeight = 1;
    this.anchorDivStyle = "width:" + a + "px;height:" + targetHeight + "px;position:relative;"
}, com.adtech.FloatingDivContainer_2_32_3.prototype.createContentDiv = function() {
    if (this.modal && this.createModalLayer(), this.setFloatingOffsets(), this.setInitialDimensions(), this.checkPublisherPositionalOverrides(), this.fixedPosition = com.adtech.Utils_2_32_3.canUseFixedPositioning() && this.stickyX && this.stickyY?!0 : !1, this.fixedPosition && this.xRel == com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_RIGHT)
        var a = null, b =- this.x - this.contentWidth + "px";
    else 
        var a = this.floatOffsetLeft + this.x + "px", b = null;
    if (this.fixedPosition && this.yRel == com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_BOTTOM)
        var c = null, d =- this.y - this.contentHeight + "px";
    else 
        var c = this.floatOffsetTop + this.y + "px", d = null;
    this.createDiv(this.contentId, this.fixedPosition ? "fixed" : "absolute", this.contentWidth + "px", this.contentHeight + "px", a, c, this.zIndex, this.initialClipStyle, b, d), this.checkCloseTimeout()
}, com.adtech.FloatingDivContainer_2_32_3.prototype.createDiv = function(a, b, c, d, e, f, g, h, i, j) {
    var k = document.createElement("div");
    return k.setAttribute("id", a), k.style.position = b, k.style.width = c, k.style.height = d, e && (k.style.left = e), i && (k.style.right = i), f && (k.style.top = f), j && (k.style.bottom = j), k.style.zIndex = g, k.style.padding = "0px", k.style.margin = "0px", h && (k.style.clip = h), this.forceRender ? document.body.insertBefore(k, document.body.firstChild) : document.body.appendChild(k), k
}, com.adtech.FloatingDivContainer_2_32_3.prototype.createModalLayer = function() {
    var a = com.adtech.Utils_2_32_3.getViewportDims();
    this.modalDiv = this.createDiv(this.contentId + "_modal", "fixed", a.w + "px", a.h + "px", "0px", "0px", this.zIndex - 1);
    var b = this.parent.getContentVariable(com.adtech.FloatingDivContainer_2_32_3.MODAL_OPACITY_CONTENT_VARIABLE), c = this.parent.getContentVariable(com.adtech.FloatingDivContainer_2_32_3.MODAL_COLOUR_CONTENT_VARIABLE), d = b || com.adtech.FloatingDivContainer_2_32_3.DEFAULT_MODAL_OPACITY, e = c || com.adtech.FloatingDivContainer_2_32_3.DEFAULT_MODAL_COLOUR;
    this.modalDiv.style.opacity = d, this.modalDiv.style.filter = "alpha(opacity:" + 100 * parseFloat(d) + ")", this.modalDiv.style.backgroundColor = 0 != e.indexOf("#") ? "#" + e : e, com.adtech.Utils_2_32_3.registerNativeEventHandler(this.modalDiv, "click", com.adtech.Utils_2_32_3.createClosure(this, this.modalClickHandler))
}, com.adtech.FloatingDivContainer_2_32_3.prototype.adjustModalDiv = function(a) {
    this.modalDiv.style.width = a.width + "px", this.modalDiv.style.height = a.height + "px"
}, com.adtech.FloatingDivContainer_2_32_3.prototype.checkPublisherPositionalOverrides = function() {
    if (this.xRel == com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_LEFT) {
        var a = parseInt(this.parent.pubVars.overrideX);
        this.x = isNaN(a) ? this.x : a
    }
    if (this.yRel == com.adtech.AssetContainerAlignmentOption_2_32_3.PAGE_TOP) {
        var b = parseInt(this.parent.pubVars.overrideY);
        this.y = isNaN(b) ? this.y : b
    }
}, com.adtech.FloatingDivContainer_2_32_3.prototype.addStickyOffsets = function(a) {
    this.stickyY && (this.floatOffsetTop += a.offsetY), this.stickyX && (this.floatOffsetLeft += a.offsetX)
}, com.adtech.FloatingDivContainer_2_32_3.prototype.resize = function() {
    var a = com.adtech.Utils_2_32_3.getViewportDims();
    "%" == this.contentWidthUnit && (this.contentWidth = com.adtech.Utils_2_32_3.calculatePercentValue(a.w, this.originalContentWidth), this.div.style.width = this.contentWidth + "px"), "%" == this.contentHeightUnit && (this.contentHeight = com.adtech.Utils_2_32_3.calculatePercentValue(a.h, this.originalContentHeight), this.div.style.height = this.contentHeight + "px"), this.visible && (this.div.style.clip = this.getUnclippedStyle())
}, com.adtech.FloatingDivContainer_2_32_3.prototype.reposition = function() {
    if (this.rendered&&!this.closed) {
        var a = com.adtech.AssetContainerAlignmentOption_2_32_3;
        (this.stickyX || this.xRel == a.AD_SLOT_LEFT) && (this.div.style.left = this.floatOffsetLeft + this.x + "px"), (this.stickyY || this.yRel == a.AD_SLOT_TOP) && (this.div.style.top = this.floatOffsetTop + this.y + "px")
    }
}, com.adtech.FloatingDivContainer_2_32_3.prototype.expandAnchorDiv = function() {}, com.adtech.FloatingDivContainer_2_32_3.prototype.contractAnchorDiv = function() {}, com.adtech.FloatingDivContainer_2_32_3.prototype.render = function() {
    return this.forceRender || this.globalEventBus.DOMLoaded ? void com.adtech.FloatingDivContainer_2_32_3.supa.render.call(this) : void this.globalEventBus.addEventListener(this.rmEvent.DOM_LOAD, com.adtech.Utils_2_32_3.createClosure(this, this.render))
}, com.adtech.FloatingDivContainer_2_32_3.prototype.checkCloseTimeout = function() {
    var a = "0" == this.parent.pubVars.closeTimeout ? 0: this.parent.pubVars.closeTimeout, b = a || this.closeTimeout, c = parseInt(b);
    if (!isNaN(c) && c) {
        var d = this, e = this.isTiedToAnInPageUnit() ? this.rmEvent.HIDE: this.rmEvent.CLOSE;
        this.autoCloseTimeout = setTimeout(com.adtech.Utils_2_32_3.createClosure(d.content, d.content.dispatchAdvertLevelEvent, new d.rmEvent(e)), 1e3 * c)
    }
}, com.adtech.FloatingDivContainer_2_32_3.prototype.isTiedToAnInPageUnit = function() {
    for (var a in this.parent.assetContainers)
        if (this.parent.assetContainers.hasOwnProperty(a)) {
            var b = this.parent.assetContainers[a];
            if (this.id != b.id&&!(b instanceof com.adtech.FloatingDivContainer_2_32_3))
                return !0
        }
    return !1
}, com.adtech.FloatingDivContainer_2_32_3.prototype.show = function() {
    return this.rendered ? void(!this.visible && this.rendered && (this.div.style.clip = this.preHiddenClipStyle, this.div.style.zIndex++, this.modalDiv && (this.modalDiv.style.visibility = "visible"), this.content.contentObject.style.display = "block", com.adtech.FloatingDivContainer_2_32_3.supa.show.call(this))) : void this.render()
}, com.adtech.FloatingDivContainer_2_32_3.prototype.hide = function() {
    this.rendered && this.visible && (this.preHiddenClipStyle = this.div.style.clip, this.div.style.clip = "rect(0px, 0px, 0px, 0px)", this.div.style.zIndex--, this.modalDiv && (this.modalDiv.style.visibility = "hidden"), com.adtech.FloatingDivContainer_2_32_3.supa.hide.call(this))
}, com.adtech.FloatingDivContainer_2_32_3.prototype.close = function() {
    if (this.rendered&&!this.closed) {
        if (this.setCoreLibPageEventListeners(!1), this.closeDelayTimeout)
            return;
        this.div.style.visibility = "hidden";
        var a = this;
        this.closeDelayTimeout = setTimeout(function() {
            com.adtech.DivContainer_2_32_3.supa.close.call(a), a.div.parentNode.removeChild(a.div), a.modalDiv && a.modalDiv.parentNode.removeChild(a.modalDiv)
        }, 100)
    }
}, com.adtech.FloatingDivContainer_2_32_3.prototype.modalClickHandler = function() {
    this.content.dispatchAdvertLevelEvent(new this.rmEvent(this.rmEvent.CLOSE))
}, com.adtech.FloatingDivContainer_2_32_3.prototype.coreLibEventHandler = function(a) {
    if (this.xRel == com.adtech.AssetContainerAlignmentOption_2_32_3.AD_SLOT_LEFT || this.yRel == com.adtech.AssetContainerAlignmentOption_2_32_3.AD_SLOT_TOP || this.stickyX || this.stickyY)
        switch (a.type) {
        case this.rmEvent.PAGE_SCROLL:
        case this.rmEvent.PAGE_RESIZE:
        case this.rmEvent.ORIENTATION_CHANGE:
            if (this.setFloatingOffsets(), !this.fixedPosition) {
                this.addStickyOffsets(a);
                try {
                    this.reposition()
                } catch (b) {}
            }
        }
    this.isPercentageRendered && this.rendered && (a.type == this.rmEvent.PAGE_RESIZE || a.type == this.rmEvent.PAGE_SCROLL) && this.resize(), a.type == this.rmEvent.ENGAGEMENT && null != this.autoCloseTimeout && clearTimeout(this.autoCloseTimeout), this.modalDiv && a.type == this.rmEvent.PAGE_RESIZE && this.adjustModalDiv(a), com.adtech.FloatingDivContainer_2_32_3.supa.coreLibEventHandler.call(this, a)
}, com.adtech.NoContentContainer_2_32_3 = function(a, b, c, d, e) {
    com.adtech.NoContentContainer_2_32_3.supa.constructor.call(this, a, b, c, d, e)
}, com.adtech.Utils_2_32_3.extend(com.adtech.NoContentContainer_2_32_3, com.adtech.AssetContainer_2_32_3), com.adtech.NoContentContainer_2_32_3.prototype.init = function(a) {
    var b = a.assetContainers[this.id];
    for (var c in b)
        b.hasOwnProperty(c) && (this[c] = b[c]);
    this.visibleinViewport=!0, this.addRenderEventListeners()
}, com.adtech.AssetContainerFactory_2_32_3 = function() {}, com.adtech.AssetContainerFactory_2_32_3.CHROMELESS_POPUP = "chromelessPopup", com.adtech.AssetContainerFactory_2_32_3.POPUP = "popup", com.adtech.AssetContainerFactory_2_32_3.INLINE_DIV = "inlineDiv", com.adtech.AssetContainerFactory_2_32_3.FLOATING_DIV = "floatingDiv", com.adtech.AssetContainerFactory_2_32_3.NO_CONTENT = "noContent", com.adtech.AssetContainerFactory_2_32_3.getContainer = function(a, b, c, d, e, f) {
    switch (a) {
    case this.CHROMELESS_POPUP:
        var g = "undefined" == typeof com.adtech.debugLogger_2_32_3 ? com.adtech.ChromelessWindowContainer_2_32_3: com.adtech.PreviewChromelessWindowContainer_2_32_3;
        return new g(b, c, d, e, f);
    case this.POPUP:
        return new com.adtech.WindowContainer_2_32_3(b, c, d, e, f);
    case this.INLINE_DIV:
        return new com.adtech.DivContainer_2_32_3(b, c, d, e, f);
    case this.FLOATING_DIV:
        return new com.adtech.FloatingDivContainer_2_32_3(b, c, d, e, f);
    case this.NO_CONTENT:
        return new com.adtech.NoContentContainer_2_32_3(b, c, d, e, f);
    default:
        return null
    }
}, com.adtech.HtmlContent_2_32_3 = function(a, b, c, d, e, f) {
    com.adtech.HtmlContent_2_32_3.supa.constructor.call(this), this.parent = a, this.eventBus = d, this.containerObject = null, this.shouldHandleMouseEvents=!1, this.mouseIsOver = null, this.contentReady=!1, this.outgoingMessageQueue = [], this.contentObject = null, this.globalEventBus = e, this.rendered=!1, this.DOMContext = document, this.advertRef = this.parent.parent, this.targetId = f, this.pageLoadForced=!1, this.isStaticContent=!1, this.init(b, c)
}, com.adtech.Utils_2_32_3.extend(com.adtech.HtmlContent_2_32_3, com.adtech.EventDispatcher_2_32_3), com.adtech.HtmlContent_2_32_3.QUEUED_MESSAGE_PUSH_DELAY = 50, com.adtech.HtmlContent_2_32_3.prototype.init = function(a, b) {
    var c = a.assetContainers[b];
    this.content = c.content, this.langVersion = c.langVersion ? c.langVersion : 0, this.width = this.renderWidthAsPercentage(c) ? "100%" : c.contentWidth, this.height = this.renderHeightAsPercentage(c) ? "100%" : c.contentHeight;
    var d = com.adtech.RichMediaEvent_2_32_3;
    this.globalEventBus.addEventListener(d.PAGE_LOAD, this.customEventHandler, this), com.adtech.Utils_2_32_3.addCustomEventHandler(this.eventBus, this, a.events);
    for (var e = [d.ENGAGEMENT, d.IN_VIEWPORT, d.OUT_VIEWPORT, d.INTERACTIVE_IMPRESSION], f = 0; f < e.length; f++)
        this.eventBus.addEventListener(e[f], this.customEventHandler, this);
    this.setAdConfig(a), this.parent.addEventListener(d.MOUSE_OVER, this.customEventHandler, this), this.parent.addEventListener(d.MOUSE_OUT, this.customEventHandler, this)
}, com.adtech.HtmlContent_2_32_3.prototype.setAdConfig = function(a) {
    this.adConfig = a, "undefined" != typeof com.adtech.debugLogger_2_32_3 && (this.adConfig.adtechDisableClickTracking=!0)
}, com.adtech.HtmlContent_2_32_3.prototype.renderWidthAsPercentage = function(a) {
    return a.isResponsive || this.containerIsFloatingDiv(a)&&!a.isExpandable && "%" == a.contentWidthUnit
}, com.adtech.HtmlContent_2_32_3.prototype.renderHeightAsPercentage = function(a) {
    return a.isResponsive || this.containerIsFloatingDiv(a)&&!a.isExpandable && "%" == a.contentHeightUnit
}, com.adtech.HtmlContent_2_32_3.prototype.containerIsFloatingDiv = function(a) {
    return a.type == com.adtech.AssetContainerFactory_2_32_3.FLOATING_DIV
}, com.adtech.HtmlContent_2_32_3.prototype.setDOMContext = function(a) {
    this.DOMContext = a
}, com.adtech.HtmlContent_2_32_3.prototype.render = function() {
    this.containerObject = this.DOMContext.getElementById(this.targetId), this.renderContent(), this.rendered=!0, this.dispatchEvent(com.adtech.RichMediaEvent_2_32_3.RENDER)
}, com.adtech.HtmlContent_2_32_3.prototype.forcePageLoad = function() {
    this.dispatchToContent(this.createContentEvent(new com.adtech.RichMediaEvent_2_32_3(com.adtech.RichMediaEvent_2_32_3.PAGE_LOAD))), this.pageLoadForced=!0
}, com.adtech.HtmlContent_2_32_3.prototype.getWindowScript = function() {
    return ""
}, com.adtech.HtmlContent_2_32_3.prototype.renderContent = function() {
    if (com.adtech.Utils_2_32_3.fileIsImage(this.content))
        this.isStaticContent=!0, this.containerObject.innerHTML = this.advertRef.getBackupImageHTML(this.content), this.contentInitialised();
    else {
        var a = this.targetId + "_iframe";
        this.addPostMessageListener();
        var b = this.advertRef.getFileUrl(this.content);
        this.globalEventBus.pageLoaded && this.forcePageLoad(), this.contentOrigin = b.split("/").splice(0, 3).join("/");
        var c = document.createElement("iframe");
        c.src = b, c.seamless = "seamless", c.width = this.width, c.height = this.height, c.id = a, c.style.padding = "0px", c.style.border = "0px none transparent", c.style.width = this.width.toString().match("%") ? this.width : this.width + "px", c.style.height = this.height.toString().match("%") ? this.height : this.height + "px", c.scrolling = "no", c.frameBorder = "no", c.allowtransparency=!0, c.setAttribute("allowFullScreen", "true"), c.setAttribute("mozallowFullScreen", "true"), c.setAttribute("webkitAllowFullScreen", "true"), com.adtech.Utils_2_32_3.isWebkit() && (c.style.visibility = "hidden", setTimeout(function() {
            c.style.visibility = "visible"
        }, 200)), this.containerObject.appendChild(c), this.contentObject = this.DOMContext.getElementById(a)
    }
}, com.adtech.HtmlContent_2_32_3.prototype.getContentEnvironment = function() {
    return com.adtech.HtmlContent_2_32_3.ENVIRONMENT_WEB
}, com.adtech.HtmlContent_2_32_3.prototype.remove = function() {
    "undefined" != typeof this.containerObject && this.rendered && (this.containerObject.innerHTML = "", this.rendered=!1, this.contentReady=!1, this.adContentWindow && (this.adContentWindow = null))
}, com.adtech.HtmlContent_2_32_3.prototype.handleMouseEvents = function() {
    this.shouldHandleMouseEvents=!0
}, com.adtech.HtmlContent_2_32_3.prototype.contentInitialised = function() {
    this.contentReady=!0, this.outgoingMessageQueue.length > 0 && this.sendQueuedMessagesToContent(), this.dispatchEvent(com.adtech.RichMediaEvent_2_32_3.READY)
}, com.adtech.HtmlContent_2_32_3.prototype.addToOutgoingQueue = function(a) {
    this.outgoingMessageQueue[this.outgoingMessageQueue.length] = a
}, com.adtech.HtmlContent_2_32_3.prototype.sendQueuedMessagesToContent = function() {
    this.dispatchToContent(this.outgoingMessageQueue.shift()), this.outgoingMessageQueue.length > 0 && setTimeout(com.adtech.Utils_2_32_3.createClosure(this, this.sendQueuedMessagesToContent), com.adtech.HtmlContent_2_32_3.QUEUED_MESSAGE_PUSH_DELAY)
}, com.adtech.HtmlContent_2_32_3.prototype.createContentEventProperties = function(a) {
    var b = {}, c = ["type", "target", "currentTarget"];
    for (var d in a)
        if (a.hasOwnProperty(d)) {
            if (com.adtech.Utils_2_32_3.inArray(d, c) || "function" == typeof a[d])
                continue;
                b[d] = a[d]
        }
    return b
}, com.adtech.HtmlContent_2_32_3.prototype.dispatchToContent = function(a) {
    !this.isStaticContent && this.contentReady ? this.sendMessage(a) : this.addToOutgoingQueue(a)
}, com.adtech.HtmlContent_2_32_3.prototype.dispatchAdvertLevelEvent = function(a) {
    a.target = this, this.eventBus.dispatchEvent(a)
}, com.adtech.HtmlContent_2_32_3.prototype.createContentEvent = function(a) {
    if (this.isStaticContent)
        return null;
    var b = {
        type: a.type
    }, c = this.createContentEventProperties(a);
    for (var d in c)
        c.hasOwnProperty(d) && (b[d] = c[d]);
    return this.constructMessage(com.adtech.HtmlContent_2_32_3.CMD_TYPE_DISPATCH, b)
}, com.adtech.HtmlContent_2_32_3.prototype.addPostMessageListener = function() {
    this.contentOrigin || com.adtech.Utils_2_32_3.registerNativeEventHandler(window, "message", com.adtech.Utils_2_32_3.createClosure(this, this.postMessageHandler))
}, com.adtech.HtmlContent_2_32_3.prototype.constructMessage = function(a, b) {
    var c = a;
    return c += void 0 != b ? com.adtech.HtmlContent_2_32_3.OBJECT_DELIM + JSON.stringify(b) : ""
}, com.adtech.HtmlContent_2_32_3.prototype.sendMessage = function(a) {
    "undefined" != typeof this.adContentWindow && this.adContentWindow.postMessage(a, this.contentOrigin)
}, com.adtech.HtmlContent_2_32_3.prototype.sendEnvironmentInfo = function() {
    var a = {
        env: this.getContentEnvironment()
    };
    this.sendMessage(this.constructMessage(com.adtech.HtmlContent_2_32_3.CMD_TYPE_INIT, a))
}, com.adtech.HtmlContent_2_32_3.prototype.sendConfig = function() {
    this.adConfig.clickDestinations = {}, this.adConfig.overrides && delete this.adConfig.overrides, this.adConfig.rmLibVersion = "2_32_3";
    var a = {
        adConfig: JSON.stringify(this.adConfig)
    };
    this.sendMessage(this.constructMessage(com.adtech.HtmlContent_2_32_3.CMD_TYPE_INIT, a)), this.contentInitialised()
}, com.adtech.HtmlContent_2_32_3.prototype.constructMessageObject = function(a) {
    var b = {}, c = a.split(com.adtech.HtmlContent_2_32_3.OBJECT_DELIM);
    return b.cmd = c[0], c.length > 1 && (b.payload = JSON.parse(c[1])), b
}, com.adtech.HtmlContent_2_32_3.prototype.logClick = function(a, b, c) {
    this.advertRef.logClick(a, b, c)
}, com.adtech.HtmlContent_2_32_3.prototype.touchStart = function() {
    this.advertRef.setTouchDeviceProperties()
}, com.adtech.HtmlContent_2_32_3.prototype.contentView = function(a, b) {
    this.advertRef.contentView(a, b)
}, com.adtech.HtmlContent_2_32_3.prototype.contentClick = function(a, b) {
    this.advertRef.contentClick(a, b)
}, com.adtech.HtmlContent_2_32_3.prototype.shouldAcceptMessage = function(a) {
    return a.origin == this.contentOrigin && this.contentObject && this.contentObject.contentWindow && this.contentObject.contentWindow == a.source?!0 : !1
}, com.adtech.HtmlContent_2_32_3.prototype.mouseOverHandler = function() {
    this.mouseIsOver || (this.mouseIsOver=!0, this.dispatchEvent(com.adtech.RichMediaEvent_2_32_3.MOUSE_OVER))
}, com.adtech.HtmlContent_2_32_3.prototype.mouseOutHandler = function() {
    this.mouseIsOver && (this.mouseIsOver=!1, this.dispatchEvent(com.adtech.RichMediaEvent_2_32_3.MOUSE_OUT))
}, com.adtech.HtmlContent_2_32_3.prototype.postMessageHandler = function(a) {
    if (this.shouldAcceptMessage(a)) {
        this.adContentWindow = this.adContentWindow || a.source;
        var b = this.constructMessageObject(a.data);
        switch (b.cmd) {
        case com.adtech.HtmlContent_2_32_3.CMD_TYPE_INIT:
            this.sendEnvironmentInfo(), this.sendConfig();
            break;
        case com.adtech.HtmlContent_2_32_3.CMD_TYPE_DISPATCH:
            var a = new com.adtech.RichMediaEvent_2_32_3(b.payload.type);
            "object" == typeof b.payload.meta && (a.meta = b.payload.meta), this.dispatchAdvertLevelEvent(a);
            break;
        case com.adtech.HtmlContent_2_32_3.CMD_TYPE_EXEC:
            this[b.payload.method].apply(this, b.payload.params)
        }
    }
}, com.adtech.HtmlContent_2_32_3.prototype.customEventHandler = function(a) {
    a.target == this || this.pageLoadForced && a.type == com.adtech.RichMediaEvent_2_32_3.PAGE_LOAD || this.dispatchToContent(this.createContentEvent(a))
}, com.adtech.HtmlContent_2_32_3 = com.adtech.HtmlContent_2_32_3 || function() {}, com.adtech.HtmlContent_2_32_3.CMD_TYPE_DISPATCH = "dispatch", com.adtech.HtmlContent_2_32_3.CMD_TYPE_EXEC = "exec", com.adtech.HtmlContent_2_32_3.CMD_TYPE_INIT = "init", com.adtech.HtmlContent_2_32_3.CMD_TYPE_MRAID = "CMD_MRAID", com.adtech.HtmlContent_2_32_3.PROPERTY_DELIM = "A$T$P", com.adtech.HtmlContent_2_32_3.OBJECT_DELIM = "A#T#O", com.adtech.HtmlContent_2_32_3.KEY_VALUE_DELIM = "#:#", com.adtech.HtmlContent_2_32_3.ENVIRONMENT_WEB = "WEB", com.adtech.HtmlContent_2_32_3.ENVIRONMENT_MRAID = "MRAID", com.adtech.HtmlContent_2_32_3.ENVIRONMENT_ORMMA = "ORMMA", com.adtech.NoContent_2_32_3 = function(a, b, c, d, e, f) {
    com.adtech.NoContent_2_32_3.supa.constructor.call(this, a, b, c, d, e, f)
}, com.adtech.Utils_2_32_3.extend(com.adtech.NoContent_2_32_3, com.adtech.HtmlContent_2_32_3), com.adtech.NoContent_2_32_3.prototype.renderContent = function() {}, com.adtech.FlashContent_2_32_3 = function(a, b, c, d, e, f) {
    this.params = {}, this.objectResizingEnabled=!1, com.adtech.FlashContent_2_32_3.supa.constructor.call(this, a, b, c, d, e, f)
}, com.adtech.Utils_2_32_3.extend(com.adtech.FlashContent_2_32_3, com.adtech.HtmlContent_2_32_3), com.adtech.FlashContent_2_32_3.prototype.init = function(a, b) {
    com.adtech.FlashContent_2_32_3.supa.init.call(this, a, b);
    var c = a.assetContainers[b];
    this.externalInterfaceHandler = this.targetId + "_handler", this.params.menu = "false", this.params.allowscriptaccess = "always", this.params.allowFullScreen = "true", this.params.wmode = c.wmode, this.flashVersion = c.pluginVersion, this.createFlashVars(a);
    var d = com.adtech.Utils_2_32_3;
    if (d.isMac() && (d.isSafari() || d.isFirefox())) {
        var e = this.generateAlignmentValue(c);
        e && (this.objectResizingEnabled=!0, this.flashVars.salign = e, this.flashVars.scaleMode = "noScale", this.startContracted = c.startContracted)
    }
}, com.adtech.FlashContent_2_32_3.prototype.generateAlignmentValue = function(a) {
    var b = "";
    return 0 == a.contractedY ? b = "T" : a.contractedY == a.contentHeight - a.contractedHeight && (b = "B"), 0 == a.contractedX ? b += "L" : a.contractedX == a.contentWidth - a.contractedWidth && (b += "R"), 2 == b.length ? b : ""
}, com.adtech.FlashContent_2_32_3.prototype.createFlashVars = function(a) {
    var b = com.adtech.HtmlContent_2_32_3;
    this.flashVars = {
        renderingDomain: self.location.hostname,
        keyValueDelim: b.KEY_VALUE_DELIM,
        propDelim: b.PROPERTY_DELIM,
        objDelim: b.OBJECT_DELIM,
        clickTargets: this.serializeClickTargets(a.clickthroughs),
        preview: a.preview,
        canvas_id: a.canvasId,
        flight_id: a.adServerVars.id,
        banner_id: a.adServerVars.bannerId
    }
}, com.adtech.FlashContent_2_32_3.prototype.forcePageLoad = function() {
    this.flashVars.pageLoaded = "true", this.pageLoadForced=!0
}, com.adtech.FlashContent_2_32_3.prototype.serializeClickTargets = function(a) {
    var b = com.adtech.HtmlContent_2_32_3, c = "";
    for (var d in a)
        if (a.hasOwnProperty(d)) {
            "" != c && (c += b.OBJECT_DELIM);
            var e = a[d];
            c += "id" + b.KEY_VALUE_DELIM + escape(d) + b.PROPERTY_DELIM + "target" + b.KEY_VALUE_DELIM + escape(e.target) + b.PROPERTY_DELIM + "features" + b.KEY_VALUE_DELIM + escape(e.features)
        }
    return escape(c)
}, com.adtech.FlashContent_2_32_3.prototype.setDOMContext = function(a) {
    com.adtech.FlashContent_2_32_3.supa.setDOMContext.call(this, a), com.adtech.swfobject_2_32_3.setDOMContext(a)
}, com.adtech.FlashContent_2_32_3.prototype.renderContent = function() {
    var a = this.targetId + "_swf";
    this.containerObject.innerHTML = '<div id="' + a + '"></div>', this.attributes = {
        id: a,
        name: a,
        style: "display:block"
    }, this.flashVars.uid = this.targetId, this.globalEventBus.pageLoaded && this.forcePageLoad(), this.addExternalInterfaceHandler();
    var b = com.adtech.swfobject_2_32_3, c = this.advertRef.getFileUrl(this.content);
    - 1 != b.ua.string.toUpperCase().indexOf("AOL") && (c += "?" + Math.random()), b.embedSWF(c, a, this.width, this.height, this.flashVersion, this.flashVars, this.params, this.attributes, com.adtech.Utils_2_32_3.createClosure(this, this.embedHandler))
}, com.adtech.FlashContent_2_32_3.prototype.addExternalInterfaceHandler = function() {
    var a = this;
    try {
        this.parent.windowContext[this.targetId + "_handler"] = function(b) {
            return a.flashEventHandler(b)
        }
    } catch (b) {}
}, com.adtech.FlashContent_2_32_3.prototype.getWindowScript = function() {
    if ("function" != typeof window[this.externalInterfaceHandler]) {
        var a = this;
        window[this.externalInterfaceHandler] = function(b) {
            return a.flashEventHandler(b)
        }
    }
    return '<script type="text/javascript">function ' + this.externalInterfaceHandler + "(data) {return parent." + this.externalInterfaceHandler + "(data);}</script>"
}, com.adtech.FlashContent_2_32_3.prototype.dispatchToContent = function(a) {
    if (this.contentReady)
        try {
            this.contentObject.jsEventHandler(a)
    } catch (b) {
        this.addToOutgoingQueue(a)
    } else 
        this.addToOutgoingQueue(a)
}, com.adtech.FlashContent_2_32_3.prototype.createContentEvent = function(a) {
    var b = this.createContentEventProperties(a);
    return {
        command: com.adtech.HtmlContent_2_32_3.CMD_TYPE_DISPATCH,
        type: a.type,
        args: b
    }
}, com.adtech.FlashContent_2_32_3.prototype.flashInitialised = function(a) {
    this.componentCapabilities = a ? a : null, this.contentInitialised()
}, com.adtech.FlashContent_2_32_3.prototype.getConfigValue = function(a) {
    var b = a.split("."), c = b.length;
    if (c > 0) {
        for (var d = this.advertRef, e = 0; c > e; e++)
            try {
                d = d[b[e]]
        } catch (f) {}
        try {
            return d.toString()
        } catch (f) {}
    }
    return ""
}, com.adtech.FlashContent_2_32_3.prototype.getOffsetX = function() {
    return this.adConfig.assetContainers.main.contractedX
}, com.adtech.FlashContent_2_32_3.prototype.getOffsetY = function() {
    return this.adConfig.assetContainers.main.contractedY
}, com.adtech.FlashContent_2_32_3.prototype.contentIsDynamic = function(a, b) {
    return this.advertRef.contentIsDynamic(a, b)
}, com.adtech.FlashContent_2_32_3.prototype.contentClick = function(a, b) {
    var c = {};
    if ("object" == typeof b)
        for (var d in b) {
            var e = d.split(com.adtech.HtmlContent_2_32_3.PROPERTY_DELIM).join(" ");
            c[e] = b[d], delete b[d]
        }
    com.adtech.FlashContent_2_32_3.supa.contentClick.call(this, a, c)
}, com.adtech.FlashContent_2_32_3.prototype.resizeObject = function(a) {
    if (this.objectResizingEnabled && null != this.componentCapabilities)
        if (this.contentReady)
            this.contentObject.style.marginTop = a.top, this.contentObject.style.marginLeft = a.left, this.contentObject.width = this.contentObject.style.width = parseInt(a.right, 10) - parseInt(a.left, 10) + "px", this.contentObject.height = this.contentObject.style.height = parseInt(a.bottom, 10) - parseInt(a.top, 10) + "px";
        else {
            var b = com.adtech.Utils_2_32_3.createClosure(this, this.resizeObject);
            this.addEventListener(com.adtech.RichMediaEvent_2_32_3.READY, b, a)
        }
}, com.adtech.FlashContent_2_32_3.prototype.shouldLaunchWinViaExtInterface = function() {
    var a = parseInt(this.langVersion);
    return a >= 3 && parseInt(com.adtech.Utils_2_32_3.getIEVersion().major) >= 7 ? "true" : "false"
}, com.adtech.FlashContent_2_32_3.prototype.embedHandler = function(a) {
    this.contentObject = a.ref, this.shouldHandleMouseEvents && (com.adtech.Utils_2_32_3.registerNativeEventHandler(this.contentObject, "mouseover", com.adtech.Utils_2_32_3.createClosure(this, this.mouseOverHandler)), com.adtech.Utils_2_32_3.registerNativeEventHandler(this.contentObject, "mouseout", com.adtech.Utils_2_32_3.createClosure(this, this.mouseOutHandler)))
}, com.adtech.FlashContent_2_32_3.prototype.flashEventHandler = function(a) {
    switch (a.command) {
    case com.adtech.HtmlContent_2_32_3.CMD_TYPE_DISPATCH:
        var b = new com.adtech.RichMediaEvent_2_32_3(a.type);
        "object" == typeof a.meta && (b.meta = a.meta), this.dispatchAdvertLevelEvent(b);
        break;
    case com.adtech.HtmlContent_2_32_3.CMD_TYPE_EXEC:
        var c = a.args ? a.args: [];
        if ("function" == typeof this[a.type])
            return this[a.type].apply(this, c);
        var d = this.advertRef[a.type].apply(this.advertRef, c);
        return com.adtech.Utils_2_32_3.isArray(d) ? [].slice.call(d) : d
    }
}, com.adtech.ScriptContent_2_32_3 = function(a, b, c, d, e, f) {
    com.adtech.ScriptContent_2_32_3.supa.constructor.call(this, a, b, c, d, e, f)
}, com.adtech.Utils_2_32_3.extend(com.adtech.ScriptContent_2_32_3, com.adtech.HtmlContent_2_32_3), com.adtech.ScriptContent_2_32_3.prototype.renderContent = function() {
    return this.replaceKeyValues(), /^\s*<\s*iframe/i.test(this.content) ? void(this.containerObject.innerHTML = this.content) : (this.tmpContainerObjId = this.targetId + "_tmpContainer", document.write('<div id="' + this.tmpContainerObjId + '" style="position:absolute;clip:rect(0px, 1px, 1px, 0px)">'), document.write(this.content), document.write("</div>"), void(this.globalEventBus.DOMLoaded ? this.moveThirdPartyContent() : this.globalEventBus.addEventListener(com.adtech.RichMediaEvent_2_32_3.PAGE_LOAD, this.moveThirdPartyContent, this)))
}, com.adtech.ScriptContent_2_32_3.prototype.moveThirdPartyContent = function() {
    for (var a = document.getElementById(this.tmpContainerObjId), b = a.childNodes.length, c = [], d = 0; b > d; d++) {
        var e = a.childNodes[d];
        if ("SCRIPT" != e.nodeName && "NOSCRIPT" != e.nodeName) {
            if ("IMG" == e.nodeName) {
                var f = parseInt(e.getAttribute("width")), g = parseInt(e.style.width);
                if (!isNaN(g) && 1 >= g ||!isNaN(f) && 1 >= f)
                    continue
            }
            c.push(e)
        }
    }
    for (var h = c.length, i = 0; h > i; i++) {
        var e = c[i];
        e.parentNode.removeChild(e), this.containerObject.appendChild(e)
    }
    a.style.display = "none"
}, com.adtech.ScriptContent_2_32_3.prototype.replaceKeyValues = function() {
    if (this.adConfig.pubVars.kvs)
        for (var a = this.adConfig.pubVars.kvs.split("#~~#"), b = 0; b < a.length; b++) {
            var c = a[b].split("#=#");
            if (2 == c.length) {
                var d = new RegExp("_" + c[0] + "_", "g");
                this.content = this.content.replace(d, c[1])
            }
        }
    }, com.adtech.ContentFactory_2_32_3 = function() {}, com.adtech.ContentFactory_2_32_3.HTML = "html", com.adtech.ContentFactory_2_32_3.FLASH = "flash", com.adtech.ContentFactory_2_32_3.SCRIPT = "script", com.adtech.ContentFactory_2_32_3.NO_CONTENT = "noContent", com.adtech.ContentFactory_2_32_3.getContent = function(a, b, c, d, e, f, g) {
    switch (a) {
    case this.HTML:
        return new com.adtech.HtmlContent_2_32_3(b, c, d, e, f, g);
    case this.FLASH:
        return new com.adtech.FlashContent_2_32_3(b, c, d, e, f, g);
    case this.SCRIPT:
        return new com.adtech.ScriptContent_2_32_3(b, c, d, e, f, g);
    case this.NO_CONTENT:
        return new com.adtech.NoContent_2_32_3(b, c, d, e, f, g)
    }
}, "undefined" != typeof adtechIframeHashArray && self != top)
    com.adtech.IframeUtils_2_32_3.setIframeObjectsFromHashArray(), com.adtech.IframeUtils_2_32_3.loadAdJsFile();
else if ("undefined" == typeof adtechAdManager_2_32_3) {
    var adtechAdManager_2_32_3 = new com.adtech.AdManager_2_32_3;
    "undefined" != typeof adtechAdQueue && adtechAdManager_2_32_3.registerAds(adtechAdQueue)
}
