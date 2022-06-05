/* From: production-mt-wfe-58-160-use1 : 14147 */
var swfobject = function() {
    var aq = "undefined", aD = "object", ab = "Shockwave Flash", X = "ShockwaveFlash.ShockwaveFlash", aE = "application/x-shockwave-flash", ac = "SWFObjectExprInst", ax = "onreadystatechange", af = window, aL = document, aB = navigator, aa = false, Z = [aN], aG = [], ag = [], al = [], aJ, ad, ap, at, ak = false, aU = false, aH, an, aI = true, ah = function() {
        var a = typeof aL.getElementById != aq && typeof aL.getElementsByTagName != aq && typeof aL.createElement != aq, e = aB.userAgent.toLowerCase(), c = aB.platform.toLowerCase(), h = c ? /win/.test(c): /win/.test(e), j = c ? /mac/.test(c): /mac/.test(e), g = /webkit/.test(e) ? parseFloat(e.replace(/^.*webkit\/(\d+(\.\d+)?).*$/, "$1")): false, d=!+"\v1", f = [0, 0, 0], k = null;
        if (typeof aB.plugins != aq && typeof aB.plugins[ab] == aD) {
            k = aB.plugins[ab].description;
            if (k&&!(typeof aB.mimeTypes != aq && aB.mimeTypes[aE]&&!aB.mimeTypes[aE].enabledPlugin)) {
                aa = true;
                d = false;
                k = k.replace(/^.*\s+(\S+\s+\S+$)/, "$1");
                f[0] = parseInt(k.replace(/^(.*)\..*$/, "$1"), 10);
                f[1] = parseInt(k.replace(/^.*\.(.*)\s.*$/, "$1"), 10);
                f[2] = /[a-zA-Z]/.test(k) ? parseInt(k.replace(/^.*[a-zA-Z]+(.*)$/, "$1"), 10) : 0
            }
        } else {
            if (typeof af.ActiveXObject != aq) {
                try {
                    var i = new ActiveXObject(X);
                    if (i) {
                        k = i.GetVariable("$version");
                        if (k) {
                            d = true;
                            k = k.split(" ")[1].split(",");
                            f = [parseInt(k[0], 10), parseInt(k[1], 10), parseInt(k[2], 10)]
                        }
                    }
                } catch (b) {}
            }
        }
        return {
            w3: a,
            pv: f,
            wk: g,
            ie: d,
            win: h,
            mac: j
        }
    }(), aK = function() {
        if (!ah.w3) {
            return 
        }
        if ((typeof aL.readyState != aq && aL.readyState == "complete") || (typeof aL.readyState == aq && (aL.getElementsByTagName("body")[0] || aL.body))) {
            aP()
        }
        if (!ak) {
            if (typeof aL.addEventListener != aq) {
                aL.addEventListener("DOMContentLoaded", aP, false)
            }
            if (ah.ie && ah.win) {
                aL.attachEvent(ax, function() {
                    if (aL.readyState == "complete") {
                        aL.detachEvent(ax, arguments.callee);
                        aP()
                    }
                });
                if (af == top) {
                    (function() {
                        if (ak) {
                            return 
                        }
                        try {
                            aL.documentElement.doScroll("left")
                        } catch (a) {
                            setTimeout(arguments.callee, 0);
                            return 
                        }
                        aP()
                    })()
                }
            }
            if (ah.wk) {
                (function() {
                    if (ak) {
                        return 
                    }
                    if (!/loaded|complete/.test(aL.readyState)) {
                        setTimeout(arguments.callee, 0);
                        return 
                    }
                    aP()
                })()
            }
            aC(aP)
        }
    }();
    function aP() {
        if (ak) {
            return 
        }
        try {
            var b = aL.getElementsByTagName("body")[0].appendChild(ar("span"));
            b.parentNode.removeChild(b)
        } catch (a) {
            return 
        }
        ak = true;
        var d = Z.length;
        for (var c = 0; c < d; c++) {
            Z[c]()
        }
    }
    function aj(a) {
        if (ak) {
            a()
        } else {
            Z[Z.length] = a
        }
    }
    function aC(a) {
        if (typeof af.addEventListener != aq) {
            af.addEventListener("load", a, false)
        } else {
            if (typeof aL.addEventListener != aq) {
                aL.addEventListener("load", a, false)
            } else {
                if (typeof af.attachEvent != aq) {
                    aM(af, "onload", a)
                } else {
                    if (typeof af.onload == "function") {
                        var b = af.onload;
                        af.onload = function() {
                            b();
                            a()
                        }
                    } else {
                        af.onload = a
                    }
                }
            }
        }
    }
    function aN() {
        if (aa) {
            Y()
        } else {
            am()
        }
    }
    function Y() {
        var d = aL.getElementsByTagName("body")[0];
        var b = ar(aD);
        b.setAttribute("type", aE);
        var a = d.appendChild(b);
        if (a) {
            var c = 0;
            (function() {
                if (typeof a.GetVariable != aq) {
                    var e = a.GetVariable("$version");
                    if (e) {
                        e = e.split(" ")[1].split(",");
                        ah.pv = [parseInt(e[0], 10), parseInt(e[1], 10), parseInt(e[2], 10)]
                    }
                } else {
                    if (c < 10) {
                        c++;
                        setTimeout(arguments.callee, 10);
                        return 
                    }
                }
                d.removeChild(b);
                a = null;
                am()
            })()
        } else {
            am()
        }
    }
    function am() {
        var g = aG.length;
        if (g > 0) {
            for (var h = 0; h < g; h++) {
                var c = aG[h].id;
                var l = aG[h].callbackFn;
                var a = {
                    success: false,
                    id: c
                };
                if (ah.pv[0] > 0) {
                    var i = aS(c);
                    if (i) {
                        if (ao(aG[h].swfVersion)&&!(ah.wk && ah.wk < 312)) {
                            ay(c, true);
                            if (l) {
                                a.success = true;
                                a.ref = av(c);
                                l(a)
                            }
                        } else {
                            if (aG[h].expressInstall && au()) {
                                var e = {};
                                e.data = aG[h].expressInstall;
                                e.width = i.getAttribute("width") || "0";
                                e.height = i.getAttribute("height") || "0";
                                if (i.getAttribute("class")) {
                                    e.styleclass = i.getAttribute("class")
                                }
                                if (i.getAttribute("align")) {
                                    e.align = i.getAttribute("align")
                                }
                                var f = {};
                                var d = i.getElementsByTagName("param");
                                var k = d.length;
                                for (var j = 0; j < k; j++) {
                                    if (d[j].getAttribute("name").toLowerCase() != "movie") {
                                        f[d[j].getAttribute("name")] = d[j].getAttribute("value")
                                    }
                                }
                                ae(e, f, c, l)
                            } else {
                                aF(i);
                                if (l) {
                                    l(a)
                                }
                            }
                        }
                    }
                } else {
                    ay(c, true);
                    if (l) {
                        var b = av(c);
                        if (b && typeof b.SetVariable != aq) {
                            a.success = true;
                            a.ref = b
                        }
                        l(a)
                    }
                }
            }
        }
    }
    function av(b) {
        var d = null;
        var c = aS(b);
        if (c && c.nodeName == "OBJECT") {
            if (typeof c.SetVariable != aq) {
                d = c
            } else {
                var a = c.getElementsByTagName(aD)[0];
                if (a) {
                    d = a
                }
            }
        }
        return d
    }
    function au() {
        return !aU && ao("6.0.65") && (ah.win || ah.mac)&&!(ah.wk && ah.wk < 312)
    }
    function ae(f, d, h, e) {
        aU = true;
        ap = e || null;
        at = {
            success: false,
            id: h
        };
        var a = aS(h);
        if (a) {
            if (a.nodeName == "OBJECT") {
                aJ = aO(a);
                ad = null
            } else {
                aJ = a;
                ad = h
            }
            f.id = ac;
            if (typeof f.width == aq || (!/%$/.test(f.width) && parseInt(f.width, 10) < 310)) {
                f.width = "310"
            }
            if (typeof f.height == aq || (!/%$/.test(f.height) && parseInt(f.height, 10) < 137)) {
                f.height = "137"
            }
            aL.title = aL.title.slice(0, 47) + " - Flash Player Installation";
            var b = ah.ie && ah.win ? "ActiveX": "PlugIn", c = "MMredirectURL=" + af.location.toString().replace(/&/g, "%26") + "&MMplayerType=" + b + "&MMdoctitle=" + aL.title;
            if (typeof d.flashvars != aq) {
                d.flashvars += "&" + c
            } else {
                d.flashvars = c
            }
            if (ah.ie && ah.win && a.readyState != 4) {
                var g = ar("div");
                h += "SWFObjectNew";
                g.setAttribute("id", h);
                a.parentNode.insertBefore(g, a);
                a.style.display = "none";
                (function() {
                    if (a.readyState == 4) {
                        a.parentNode.removeChild(a)
                    } else {
                        setTimeout(arguments.callee, 10)
                    }
                })()
            }
            aA(f, d, h)
        }
    }
    function aF(a) {
        if (ah.ie && ah.win && a.readyState != 4) {
            var b = ar("div");
            a.parentNode.insertBefore(b, a);
            b.parentNode.replaceChild(aO(a), b);
            a.style.display = "none";
            (function() {
                if (a.readyState == 4) {
                    a.parentNode.removeChild(a)
                } else {
                    setTimeout(arguments.callee, 10)
                }
            })()
        } else {
            a.parentNode.replaceChild(aO(a), a)
        }
    }
    function aO(b) {
        var d = ar("div");
        if (ah.win && ah.ie) {
            d.innerHTML = b.innerHTML
        } else {
            var e = b.getElementsByTagName(aD)[0];
            if (e) {
                var a = e.childNodes;
                if (a) {
                    var f = a.length;
                    for (var c = 0; c < f; c++) {
                        if (!(a[c].nodeType == 1 && a[c].nodeName == "PARAM")&&!(a[c].nodeType == 8)) {
                            d.appendChild(a[c].cloneNode(true))
                        }
                    }
                }
            }
        }
        return d
    }
    function aA(e, g, c) {
        var d, a = aS(c);
        if (ah.wk && ah.wk < 312) {
            return d
        }
        if (a) {
            if (typeof e.id == aq) {
                e.id = c
            }
            if (ah.ie && ah.win) {
                var f = "";
                for (var i in e) {
                    if (e[i] != Object.prototype[i]) {
                        if (i.toLowerCase() == "data") {
                            g.movie = e[i]
                        } else {
                            if (i.toLowerCase() == "styleclass") {
                                f += ' class="' + e[i] + '"'
                            } else {
                                if (i.toLowerCase() != "classid") {
                                    f += " " + i + '="' + e[i] + '"'
                                }
                            }
                        }
                    }
                }
                var h = "";
                for (var j in g) {
                    if (g[j] != Object.prototype[j]) {
                        h += '<param name="' + j + '" value="' + g[j] + '" />'
                    }
                }
                a.outerHTML = '<object classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000"' + f + ">" + h + "</object>";
                ag[ag.length] = e.id;
                d = aS(e.id)
            } else {
                var b = ar(aD);
                b.setAttribute("type", aE);
                for (var k in e) {
                    if (e[k] != Object.prototype[k]) {
                        if (k.toLowerCase() == "styleclass") {
                            b.setAttribute("class", e[k])
                        } else {
                            if (k.toLowerCase() != "classid") {
                                b.setAttribute(k, e[k])
                            }
                        }
                    }
                }
                for (var l in g) {
                    if (g[l] != Object.prototype[l] && l.toLowerCase() != "movie") {
                        aQ(b, l, g[l])
                    }
                }
                a.parentNode.replaceChild(b, a);
                d = b
            }
        }
        return d
    }
    function aQ(b, d, c) {
        var a = ar("param");
        a.setAttribute("name", d);
        a.setAttribute("value", c);
        b.appendChild(a)
    }
    function aw(a) {
        var b = aS(a);
        if (b && b.nodeName == "OBJECT") {
            if (ah.ie && ah.win) {
                b.style.display = "none";
                (function() {
                    if (b.readyState == 4) {
                        aT(a)
                    } else {
                        setTimeout(arguments.callee, 10)
                    }
                })()
            } else {
                b.parentNode.removeChild(b)
            }
        }
    }
    function aT(a) {
        var b = aS(a);
        if (b) {
            for (var c in b) {
                if (typeof b[c] == "function") {
                    b[c] = null
                }
            }
            b.parentNode.removeChild(b)
        }
    }
    function aS(a) {
        var c = null;
        try {
            c = aL.getElementById(a)
        } catch (b) {}
        return c
    }
    function ar(a) {
        return aL.createElement(a)
    }
    function aM(a, c, b) {
        a.attachEvent(c, b);
        al[al.length] = [a, c, b]
    }
    function ao(a) {
        var b = ah.pv, c = a.split(".");
        c[0] = parseInt(c[0], 10);
        c[1] = parseInt(c[1], 10) || 0;
        c[2] = parseInt(c[2], 10) || 0;
        return (b[0] > c[0] || (b[0] == c[0] && b[1] > c[1]) || (b[0] == c[0] && b[1] == c[1] && b[2] >= c[2])) ? true : false
    }
    function az(b, f, a, c) {
        if (ah.ie && ah.mac) {
            return 
        }
        var e = aL.getElementsByTagName("head")[0];
        if (!e) {
            return 
        }
        var g = (a && typeof a == "string") ? a: "screen";
        if (c) {
            aH = null;
            an = null
        }
        if (!aH || an != g) {
            var d = ar("style");
            d.setAttribute("type", "text/css");
            d.setAttribute("media", g);
            aH = e.appendChild(d);
            if (ah.ie && ah.win && typeof aL.styleSheets != aq && aL.styleSheets.length > 0) {
                aH = aL.styleSheets[aL.styleSheets.length - 1]
            }
            an = g
        }
        if (ah.ie && ah.win) {
            if (aH && typeof aH.addRule == aD) {
                aH.addRule(b, f)
            }
        } else {
            if (aH && typeof aL.createTextNode != aq) {
                aH.appendChild(aL.createTextNode(b + " {" + f + "}"))
            }
        }
    }
    function ay(a, c) {
        if (!aI) {
            return 
        }
        var b = c ? "visible": "hidden";
        if (ak && aS(a)) {
            aS(a).style.visibility = b
        } else {
            az("#" + a, "visibility:" + b)
        }
    }
    function ai(b) {
        var a = /[\\\"<>\.;]/;
        var c = a.exec(b) != null;
        return c && typeof encodeURIComponent != aq ? encodeURIComponent(b) : b
    }
    var aR = function() {
        if (ah.ie && ah.win) {
            window.attachEvent("onunload", function() {
                var a = al.length;
                for (var b = 0; b < a; b++) {
                    al[b][0].detachEvent(al[b][1], al[b][2])
                }
                var d = ag.length;
                for (var c = 0; c < d; c++) {
                    aw(ag[c])
                }
                for (var e in ah) {
                    ah[e] = null
                }
                ah = null;
                for (var f in swfobject) {
                    swfobject[f] = null
                }
                swfobject = null
            })
        }
    }();
    return {
        registerObject: function(a, e, c, b) {
            if (ah.w3 && a && e) {
                var d = {};
                d.id = a;
                d.swfVersion = e;
                d.expressInstall = c;
                d.callbackFn = b;
                aG[aG.length] = d;
                ay(a, false)
            } else {
                if (b) {
                    b({
                        success: false,
                        id: a
                    })
                }
            }
        },
        getObjectById: function(a) {
            if (ah.w3) {
                return av(a)
            }
        },
        embedSWF: function(k, e, h, f, c, a, b, i, g, j) {
            var d = {
                success: false,
                id: e
            };
            if (ah.w3&&!(ah.wk && ah.wk < 312) && k && e && h && f && c) {
                ay(e, false);
                aj(function() {
                    h += "";
                    f += "";
                    var q = {};
                    if (g && typeof g === aD) {
                        for (var o in g) {
                            q[o] = g[o]
                        }
                    }
                    q.data = k;
                    q.width = h;
                    q.height = f;
                    var n = {};
                    if (i && typeof i === aD) {
                        for (var p in i) {
                            n[p] = i[p]
                        }
                    }
                    if (b && typeof b === aD) {
                        for (var l in b) {
                            if (typeof n.flashvars != aq) {
                                n.flashvars += "&" + l + "=" + b[l]
                            } else {
                                n.flashvars = l + "=" + b[l]
                            }
                        }
                    }
                    if (ao(c)) {
                        var m = aA(q, n, e);
                        if (q.id == e) {
                            ay(e, true)
                        }
                        d.success = true;
                        d.ref = m
                    } else {
                        if (a && au()) {
                            q.data = a;
                            ae(q, n, e, j);
                            return 
                        } else {
                            ay(e, true)
                        }
                    }
                    if (j) {
                        j(d)
                    }
                })
            } else {
                if (j) {
                    j(d)
                }
            }
        },
        switchOffAutoHideShow: function() {
            aI = false
        },
        ua: ah,
        getFlashPlayerVersion: function() {
            return {
                major: ah.pv[0],
                minor: ah.pv[1],
                release: ah.pv[2]
            }
        },
        hasFlashPlayerVersion: ao,
        createSWF: function(a, b, c) {
            if (ah.w3) {
                return aA(a, b, c)
            } else {
                return undefined
            }
        },
        showExpressInstall: function(b, a, d, c) {
            if (ah.w3 && au()) {
                ae(b, a, d, c)
            }
        },
        removeSWF: function(a) {
            if (ah.w3) {
                aw(a)
            }
        },
        createCSS: function(b, a, c, d) {
            if (ah.w3) {
                az(b, a, c, d)
            }
        },
        addDomLoadEvent: aj,
        addLoadEvent: aC,
        getQueryParamValue: function(b) {
            var a = aL.location.search || aL.location.hash;
            if (a) {
                if (/\?/.test(a)) {
                    a = a.split("?")[1]
                }
                if (b == null) {
                    return ai(a)
                }
                var c = a.split("&");
                for (var d = 0; d < c.length; d++) {
                    if (c[d].substring(0, c[d].indexOf("=")) == b) {
                        return ai(c[d].substring((c[d].indexOf("=") + 1)))
                    }
                }
            }
            return ""
        },
        expressInstallCallback: function() {
            if (aU) {
                var a = aS(ac);
                if (a && aJ) {
                    a.parentNode.replaceChild(aJ, a);
                    if (ad) {
                        ay(ad, true);
                        if (ah.ie && ah.win) {
                            aJ.style.display = "block"
                        }
                    }
                    if (ap) {
                        ap(at)
                    }
                }
                aU = false
            }
        }
    }
}();
var FlashObject = swfobject;
(function(d) {
    d.each(["backgroundColor", "borderBottomColor", "borderLeftColor", "borderRightColor", "borderTopColor", "color", "outlineColor"], function(f, e) {
        d.fx.step[e] = function(g) {
            if (g.state == 0 || g.start.constructor != Array || g.end.constructor != Array) {
                g.start = c(g.elem, e);
                g.end = b(g.end)
            }
            g.elem.style[e] = "rgb(" + [Math.max(Math.min(parseInt((g.pos * (g.end[0] - g.start[0])) + g.start[0]), 255), 0), Math.max(Math.min(parseInt((g.pos * (g.end[1] - g.start[1])) + g.start[1]), 255), 0), Math.max(Math.min(parseInt((g.pos * (g.end[2] - g.start[2])) + g.start[2]), 255), 0)].join(",") + ")"
        }
    });
    function b(f) {
        var e;
        if (f && f.constructor == Array && f.length == 3) {
            return f
        }
        if (e = /rgb\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*\)/.exec(f)) {
            return [parseInt(e[1]), parseInt(e[2]), parseInt(e[3])]
        }
        if (e = /rgb\(\s*([0-9]+(?:\.[0-9]+)?)\%\s*,\s*([0-9]+(?:\.[0-9]+)?)\%\s*,\s*([0-9]+(?:\.[0-9]+)?)\%\s*\)/.exec(f)) {
            return [parseFloat(e[1]) * 2.55, parseFloat(e[2]) * 2.55, parseFloat(e[3]) * 2.55]
        }
        if (e = /#([a-fA-F0-9]{2})([a-fA-F0-9]{2})([a-fA-F0-9]{2})/.exec(f)) {
            return [parseInt(e[1], 16), parseInt(e[2], 16), parseInt(e[3], 16)]
        }
        if (e = /#([a-fA-F0-9])([a-fA-F0-9])([a-fA-F0-9])/.exec(f)) {
            return [parseInt(e[1] + e[1], 16), parseInt(e[2] + e[2], 16), parseInt(e[3] + e[3], 16)]
        }
        return a[d.trim(f).toLowerCase()]
    }
    function c(g, e) {
        var f;
        do {
            f = d.curCSS(g, e);
            if (f != "" && f != "transparent" || d.nodeName(g, "body")) {
                break
            }
            e = "backgroundColor"
        }
        while (g = g.parentNode);
        return b(f)
    }
    var a = {
        aqua: [0, 255, 255],
        azure: [240, 255, 255],
        beige: [245, 245, 220],
        black: [0, 0, 0],
        blue: [0, 0, 255],
        brown: [165, 42, 42],
        cyan: [0, 255, 255],
        darkblue: [0, 0, 139],
        darkcyan: [0, 139, 139],
        darkgrey: [169, 169, 169],
        darkgreen: [0, 100, 0],
        darkkhaki: [189, 183, 107],
        darkmagenta: [139, 0, 139],
        darkolivegreen: [85, 107, 47],
        darkorange: [255, 140, 0],
        darkorchid: [153, 50, 204],
        darkred: [139, 0, 0],
        darksalmon: [233, 150, 122],
        darkviolet: [148, 0, 211],
        fuchsia: [255, 0, 255],
        gold: [255, 215, 0],
        green: [0, 128, 0],
        indigo: [75, 0, 130],
        khaki: [240, 230, 140],
        lightblue: [173, 216, 230],
        lightcyan: [224, 255, 255],
        lightgreen: [144, 238, 144],
        lightgrey: [211, 211, 211],
        lightpink: [255, 182, 193],
        lightyellow: [255, 255, 224],
        lime: [0, 255, 0],
        magenta: [255, 0, 255],
        maroon: [128, 0, 0],
        navy: [0, 0, 128],
        olive: [128, 128, 0],
        orange: [255, 165, 0],
        pink: [255, 192, 203],
        purple: [128, 0, 128],
        violet: [128, 0, 128],
        red: [255, 0, 0],
        silver: [192, 192, 192],
        white: [255, 255, 255],
        yellow: [255, 255, 0]
    }
})(jQuery);
var FloatingPrompt = {
    type: "bottom",
    embed_id: null,
    container: null,
    html_or_url: "",
    default_class: "prompting_overlay",
    embed: function(o, l, d, c, t) {
        if (!this.container) {
            this.container = document.createElement("div");
            Dom.addClass(this.container, this.default_class);
            this.embed_id = Dom.generateId(this.container);
            document.body.appendChild(this.container)
        } else {
            this.container.className = this.default_class
        }
        if (!o.id) {
            Dom.generateId(o)
        }
        this.type = c || this.type;
        t = t || {};
        var p = t.after_cb || function() {}, g = t.arrow_style || "", m = t.timeout_remove || 100, f = t.fp_intersects, i = t.pre_cb || function() {}, j = t.ignore_arrow || false, s = t.class_name || "", k = t.forget_intersect, n = t.base_coords || undefined;
        if (f) {
            o.setAttribute("on_hover", "yes")
        }
        if (t.width) {
            this.container.style.width = t.width + "px"
        }
        if (s) {
            Dom.addClass(this.container, s)
        }
        this.container.setAttribute("embedded_to", o.id);
        var r = function() {
            o.removeAttribute("on_hover");
            var w = this, x = function() {
                if (w.parentNode&&!o.getAttribute("on_hover")) {
                    p();
                    if (w.getAttribute("embedded_to") == o.id) {
                        E.removeListener(w, "mouseout");
                        E.removeListener(w, "mouseover");
                        w.style.display = "none"
                    }
                    o.removeAttribute("floating_id")
                }
            };
            if (o.getAttribute("fp_intersects")) {
                setTimeout(x, m)
            } else {
                x()
            }
        };
        o.setAttribute("floating_id", this.embed_id);
        E.on(o, "mouseout", function() {
            this.removeAttribute("on_hover");
            var w = this, x = function() {
                if (Dom.get(w.getAttribute("floating_id"))&&!w.getAttribute("on_hover")) {
                    p();
                    if (Dom.get(w.getAttribute("floating_id")).getAttribute("embedded_to") == o.id) {
                        E.removeListener(w.getAttribute("floating_id"), "mouseout");
                        E.removeListener(w.getAttribute("floating_id"), "mouseover");
                        Dom.get(w.getAttribute("floating_id")).style.display = "none"
                    }
                    w.removeAttribute("floating_id")
                }
            };
            if (o.getAttribute("fp_intersects")) {
                setTimeout(x, m)
            } else {
                x()
            }
        });
        var h = "", b = "", e = t.add_xy || [0, 0];
        switch (this.type) {
        case"bottom":
            h = ' style="top:-5px;left:-14px;" ';
            b = "right-top";
            break;
        case"top":
            h = "";
            b = "left-top";
            break;
        case"top-right":
            h = "";
            b = "right-top";
            break
        }
        h = g || h;
        if (undefined === f&&!k) {
            var v = Y.util.Region.getRegion(o);
            v.left -= 1;
            v.right += 1;
            v.top -= 1;
            v.bottom += 1
        }
        if ("" != l) {
            this.container.innerHTML=!j ? this._GetContent(h, l) : l;
            this.container.style.display = "block";
            var q = this;
            HPUtil.ShowNearElement(b, o, this.container, e, function() {
                i(q, r)
            }, n);
            if (undefined === f&&!k) {
                var a = Y.util.Region.getRegion(this.container).intersect(v) ? 1: 0;
                o.setAttribute("fp_intersects", a);
                if (a) {
                    o.setAttribute("on_hover", "yes");
                    E.on(this.container, "mouseout", r);
                    E.on([this.container, o], "mouseover", function() {
                        o.setAttribute("on_hover", "yes");
                        return false
                    })
                }
            } else {
                if (!k) {
                    o.setAttribute("fp_intersects", f.toString());
                    if (f) {
                        E.on(this.container, "mouseout", r);
                        E.on([this.container, o], "mouseover", function() {
                            o.setAttribute("on_hover", "yes");
                            return false
                        })
                    }
                }
            }
        } else {
            var u = this;
            C.asyncRequest("GET", d, {
                success: function(x) {
                    u.container.innerHTML=!j ? u._GetContent(h, x.responseText) : x.responseText;
                    u.container.style.display = "block";
                    HPUtil.ShowNearElement(b, o, u.container, e, function() {
                        i(u, r)
                    }, n);
                    if (undefined === f&&!k) {
                        var w = Y.util.Region.getRegion(u.contaner).intersect(v) ? 1: 0;
                        o.setAttribute("fp_intersects", w);
                        if (w) {
                            o.setAttribute("on_hover", "yes");
                            E.on(u.container, "mouseout", r);
                            E.on(u.container, "mouseover", function() {
                                o.setAttribute("on_hover", "yes");
                                return false
                            })
                        }
                    } else {
                        if (!k) {
                            o.setAttribute("fp_intersects", f.toString());
                            if (f) {
                                E.on(u.container, "mouseout", r);
                                E.on(u.container, "mouseover", function() {
                                    o.setAttribute("on_hover", "yes");
                                    return false
                                })
                            }
                        }
                    }
                },
                failure: function() {
                    HPError.e()
                }
            })
        }
    },
    _GetContent: function(b, a) {
        switch (this.type) {
        case"bottom":
            return '<div class="btm_embed_arrow" ' + b + "></div>" + a;
            break;
        case"top":
        case"top-right":
            return a + '<div class="top_embed_arrow" ' + b + "></div>";
            break
        }
    }
};
huff.unit("tmp/floating_promt", function(a) {
    a(FloatingPrompt)
});
window._ || (window._ = function(a) {
    return a
});
window.ngettext || (window.ngettext = function(a) {
    return a
});
window.sprintf || (window.sprintf = function(a) {
    return a
});
var QuickSubscribeUser = {
    user_connected: false,
    pop: function(c, b, g, j) {
        var d = c.split("_");
        c = d[0];
        g = g || 1;
        if (!j) {
            j = "subscribe_user_email"
        }
        if (jQuery("#" + j).val() == "") {
            alert(__("Please enter your email address."));
            $(j).focus();
            return false
        }
        if (!HPUtil.checkEmail(jQuery("#" + j).val())) {
            alert(__("Oops! Please enter your email address again."));
            $(j).focus();
            return false
        }
        if (jQuery("#newsletter-choices").length == 0) {
            HuffConnect.showLoader()
        }
        var i = jQuery("#" + j).val();
        var a = "/users/favorite-bloggers/get_subscribe_user.php";
        var f = b + "=1&daily=1&fid=" + b + "&vertical=" + c + "&ajax=1&update=1&mode=add";
        if (g) {
            var e = {
                post_data: f + "&third_party_user=1",
                url: a
            };
            f += "&email=" + encodeURIComponent(i);
            jQuery.ajax({
                url: a,
                type: "POST",
                data: f,
                success: function(m, l, k) {
                    QuickSubscribeUser.Success(m, l, k, e);
                    return 
                },
                error: QuickSubscribeUser.Fail
            })
        } else {
            jQuery("#huff_modal_common_inner").html("<iframe id='alert_email_iframe' frameborder=no src='http://www.huffingtonpost.com/users/favorite-bloggers/get_subscribe_user.php?email=" + encodeURIComponent(i) + "&iframe=1&fid=" + b + "' style='width:544px; height:110px; visibility:hidden; overflow:hidden;' onload='QuickSubscribeUser.setIframeSize()'></iframe>")
        }
        return false
    },
    checkAll: function() {
        var b = false;
        if ($("check_all").checked == true) {
            b = true
        }
        var c = $("unsub_form");
        for (var a = 0; a < c.length; a++) {
            if (c.elements[a].type == "checkbox") {
                c.elements[a].checked = b
            }
        }
    },
    checkInput: function(a) {
        if (a.value == __("Enter email address")) {
            a.value = "";
            jQuery(a).removeClass("color_ACACAC")
        }
    },
    Success: function(e, d, c, f) {
        var a = HuffConnect.Login;
        if (c.status != 200) {
            if (QuickSubscribeUser.user_connected) {
                a.onLoginSuccess()
            }
            return 
        }
        var b = function() {};
        if (QuickSubscribeUser.user_connected) {
            b = function() {
                a.onLoginSuccess()
            }
        }
        if (jQuery("#newsletter-choices").length != 0) {
            jQuery("#newsletter-choices").html(e)
        } else {
            HuffConnect.socialModal({
                inner_html: e,
                onclose: b
            })
        }
        return 
    },
    Fail: function() {
        HuffConnect.hideModal();
        var a = HuffConnect.Login;
        if (QuickSubscribeUser.user_connected) {
            a.onLoginSuccess()
        }
        return 
    },
    Subscribe: function(b) {
        if (jQuery("#newsletter-choices").length != 0) {
            jQuery("#subscribe_loader").css("display", "none")
        } else {
            jQuery("#subscribe_loader").css("display", "block")
        }
        var g = jQuery("#unsub_form").get(0);
        var d = new Array();
        var a = 0;
        for (var c = 0; c < g.length; c++) {
            if (g.elements[c].type == "checkbox") {
                d[a] = g.elements[c].name;
                a += 1;
                if (g.elements[c].checked == true) {
                    g.elements[c].value = 1
                } else {
                    g.elements[c].value = 0
                }
            }
        }
        var f = "update=1&email=" + encodeURIComponent(b);
        for (c = 0; c < d.length; c++) {
            var e = $(d[c] + "");
            if (e != null) {
                f += "&" + escape(e.name) + "=" + escape(e.value)
            }
        }
        jQuery.ajax({
            url: "/subscription/get_email_alerts.php",
            type: "POST",
            data: f,
            success: QuickSubscribeUser.SubscribeSuccessEmail,
            error: QuickSubscribeUser.SubscribeFailEmail
        });
        return false
    },
    SubscribeSuccessEmail: function(e, d, c) {
        if (c.status != 200) {
            return 
        }
        var b = jQuery("#subscribe_alerts_div");
        var a = "<div class='center' style='text-align:left;padding-left:159px;padding-top:10px;background-color:#FAFAFA;border-bottom:1px solid #EDEDED;border-top:1px solid #EDEDED;'><h3 style='font-size:18px;padding-bottom:5px;'>" + __("Thank You") + "</h3><p class='note padding_5'>" + __("Your subscription settings have been updated") + "</p></div><p class='privacy' style='margin-top:10px;'><a href='/privacy'>" + __("Privacy Policy") + "</a></p>";
        if (b.length) {
            b.html(a)
        } else {
            if (QuickSubscribeUser.user_connected) {
                HCL.onLoginSuccess()
            }
            HuffConnect.hideModal()
        }
        return 
    },
    SubscribeFailEmail: function() {
        if (QuickSubscribeUser.user_connected) {
            HCL.onLoginSuccess()
        }
        HPError.e()
    },
    setIframeSize: function() {
        h = document.getElementById("alert_email_iframe").myheight || 770;
        try {
            document.getElementById("alert_email_iframe").style.height = document.getElementById("alert_email_iframe").contentDocument.body.clientHeight + "px"
        } catch (a) {
            document.getElementById("alert_email_iframe").style.height = h + "px"
        }
        document.getElementById("alert_email_iframe").style.visibility = "visible"
    },
    pop2: function(a, d) {
        d = d || "";
        var c = "";
        var e = "f32=1&" + d + "=1";
        var b = "/users/alerts/signup.php?" + e;
        HuffConnect.socialModal({
            url: b,
            modal_head_image: "",
            modal_subhead_image: "",
            width: 595
        })
    },
    getAlerts: function(b, c, a, d) {
        d = d || false;
        if (b.email.value == "") {
            alert(__("Please enter your email address."));
            b.email.focus();
            return false
        }
        if (!(b.email && HPUtil.checkEmail(b.email.value))) {
            alert(__("Please specify a valid e-mail."));
            b.email.focus();
            return false
        }
        if (d) {
            HuffConnect.showLoader()
        }
        post_data = "email=" + encodeURIComponent(b.email.value);
        if (c) {
            post_data += "&hide_close_button=1"
        }
        if (a) {
            post_data += "&daily_brief=1"
        }
        YAHOO.util.Connect.asyncRequest("POST", "/subscription/get_email_alerts.php", {
            success: QuickSubscribeUser.SubscribeSuccess,
            failure: QuickSubscribeUser.SubscribeFail
        }, post_data)
    },
    SubscribeSuccess: function(a) {
        HuffConnect.socialModal({
            inner_html: a.responseText
        })
    },
    SubscribeFail: function(a) {
        HuffConnect.hideModal();
        HPError.e()
    }
};
huff.unit("tmp/quick_subscribe_user", function(a) {
    a(QuickSubscribeUser)
});
window._ || (window._ = function(a) {
    return a
});
window.ngettext || (window.ngettext = function(a) {
    return a
});
window.sprintf || (window.sprintf = function(a) {
    return a
});
var QuickFan = {
    fan_timeout: null,
    pop_similar: function(a) {
        HPTrack.trackPageview("/t/a/similar_bloggers");
        YAHOO.util.Connect.asyncRequest("GET", "/users/favorite-bloggers/get_similar_bloggers.php?author=" + a, {
            success: QuickFan.GetSimilarSuccess,
            failure: QuickFan.GetSimilarFail
        }, {});
        return false
    },
    GetSimilarSuccess: function(a) {
        HuffConnect.socialModal({
            inner_html: a.responseText,
            modal_subhead_image: "",
            modal_head_image: "",
            width: 600
        })
    },
    GetSimilarFail: function(a) {
        HPError.e();
        HuffConnect.hideModal()
    },
    pop: function(a) {
        if (HuffCookies.getUserName()) {
            QuickFan.becomeFan(a)
        } else {
            HPConnect.Login()
        }
    },
    become_fan_mouseover: function() {
        if (QuickFan.fan_timeout) {
            clearTimeout(QuickFan.fan_timeout)
        }
    },
    become_fan_hovercard: function(c, e, b, g, a, d) {
        a = a || 27;
        g = g || [0, 0];
        var c = c || window.event;
        var f = c.target || c.srcElement;
        var h = c.type.toLowerCase();
        d = d || e;
        if (f.getAttribute("floating_id")) {
            return false
        }
        QuickFan.fan_timeout = setTimeout(function() {
            f.setAttribute("on_hover", "yes");
            var i = "";
            if (!HuffCookies.getUserName()) {
                i = '<div class="become_fan_tooltip"><div class="float_left bold"><img src="/contributors/' + b + '/headshot.jpg" class="float_left margin_right_5" /><div class="float_left become_fan_text">' + d.charAt(0).toUpperCase() + d.substr(1, d.length - 1) + '<div class="margin_top_5"><img onclick="return QuickFan.hovercard_handle(this, \'hp_blogger_' + e.replace(/'/g, "\\'") + '\');" class="cursor_pointer social fan-bloger" src="/images/blank.gif"></div></div></div><div class="clear_first"></div><div class="become_fan_corner"><div class="first"></div><div class="second"></div></div></div>'
            } else {
                i = '<div class="become_fan_tooltip"><div class="fp_loader">' + __("Loading...") + '</div><div class="become_fan_corner"><div class="first"></div><div class="second"></div></div></div>'
            }
            FloatingPrompt.embed(f, i, undefined, "top", {
                ignore_arrow: true,
                width: 180,
                add_xy: [ - 72 + g[0], g[1] + (HuffCookies.getUserName()?-50 : - 72)],
                class_name: "clear-overlay",
                fp_intersects: 1
            });
            if (HuffCookies.getUserName()) {
                var j = "author_name=hp_blogger_" + encodeURIComponent(e);
                C.asyncRequest("POST", "/web-services/author/friendship.php", {
                    success: function(n) {
                        var k = JSON.parse(n.responseText);
                        var l = k.friendship ? "unfan-bloger": "fan-bloger";
                        var m = '<div class="become_fan_tooltip"><div class="float_left bold"><img src="/contributors/' + b + '/headshot.jpg" class="float_left margin_right_5" /><div class="float_left become_fan_text">' + d.charAt(0).toUpperCase() + d.substr(1, d.length - 1) + '<div class="margin_top_5"><img onclick="return QuickFan.hovercard_handle(this, \'hp_blogger_' + e.replace(/'/g, "\\'") + '\');" class="cursor_pointer social ' + l + '" src="/images/blank.gif"></div></div></div><div class="clear_first"></div><div class="become_fan_corner"><div class="first"></div><div class="second"></div></div></div>';
                        FloatingPrompt.container.style.top = (parseInt(FloatingPrompt.container.style.top) - a).toString() + "px";
                        FloatingPrompt.container.innerHTML = m
                    },
                    failure: function() {
                        HPError.e()
                    },
                    timeout: 10000
                }, j)
            }
        }, 1000);
        return false
    },
    hovercard_handle: function(b, c) {
        if (!HuffCookies.getUserName()) {
            ConnectOverview.showBenefits({
                provider: "signup"
            });
            return false
        }
        var a=!jQuery(b).hasClass("fan-bloger");
        jQuery.ajax({
            url: "/users/favorite-bloggers/fan_action.php",
            data: "fan=" + c + "&action=" + (a ? "delete" : "add"),
            type: "POST",
            success: function(d) {
                if ( - 1 !== d.indexOf("::")) {
                    if (!a) {
                        jQuery(b).removeClass("fan-bloger").addClass("unfan-bloger")
                    } else {
                        jQuery(b).removeClass("unfan-bloger").addClass("fan-bloger")
                    }
                }
            },
            error: function() {
                HPError.e()
            }
        });
        return false
    },
    becomeFan: function(a) {
        jQuery.ajax({
            url: "/users/favorite-bloggers/fan_action.php",
            data: "fan=" + a + "&action=add",
            type: "POST",
            success: QuickFan.Success,
            error: QuickFan.Fail
        });
        return 
    },
    Success: function(e, d, c) {
        if (c.status != 200) {
            HuffConnect.hideModal();
            return 
        }
        resp = e;
        action = resp.substring(0, 3);
        if (action == "add") {
            var b = resp.split("::");
            SNProject.track(b[2], "user_follow");
            var f = b[3];
            var a = b[4];
            QuickFan.pop_email_alerts(b[1], f, a)
        } else {
            if (resp == "notificationsaved") {
                jQuery("#quick_alert_div").html("<div style='margin:30px 0;' class='arial_16 bold center color_444444'>" + __("Social preferences successfully saved") + "</div>");
                setTimeout(function() {
                    HuffConnect.hideModal()
                }, 4000)
            } else {
                if (action != "mov") {
                    return QuickFan.Fail()
                }
            }
        }
        return false
    },
    Fail: function() {
        HPError.e();
        HuffConnect.hideModal();
        return 
    },
    emailAlerts: function(b, c, d, a) {
        d = d || false;
        a = a || false;
        QuickFan.pop_email_alerts(b, d, a, c);
        return 
    },
    pop_email_alerts: function(c, e, a, d) {
        e = e || false;
        a = a || false;
        d = d || c;
        if (HuffCookies.getUserName()) {
            var b = "/users/favorite-bloggers/qet_email_alerts.php?blogger=" + c + "&blogger_name=" + d;
            if (e) {
                b += "&fan_success=" + e
            }
            if (a) {
                b += "&fan_reason=" + a
            }
            HuffConnect.showLoader();
            jQuery.ajax({
                type: "GET",
                url: b,
                success: QuickFan.GetEmailAlertsSuccess,
                error: QuickFan.GetEmailAlertsFail
            })
        } else {
            ConnectOverview.showBenefits({
                provider: "signup",
                message: __("You must be logged in to subscribe."),
                onSuccess: function() {
                    QuickFan.user_connected = true;
                    QuickFan.pop_email_alerts(c, e, a)
                }
            })
        }
    },
    GetEmailAlertsSuccess: function(e, d, c) {
        if (c.status != 200) {
            HuffConnect.hideModal();
            return 
        }
        var a = {
            inner_html: e,
            width: 670,
            modal_head_image: "",
            modal_subhead_image: ""
        };
        if (QuickFan.user_connected) {
            a.onclose = function() {
                HuffConnect.Login.onLoginSuccess()
            }
        }
        HuffConnect.socialModal(a);
        if (HPConfig.current_vertical_name&&!(/homepage/.test(HPConfig.current_vertical_name))) {
            var b = HuffPoUtil.trim(HPConfig.current_vertical_name);
            b = b.replace("-", " ");
            b = b.toLowerCase().replace(/\w+/g, function(f) {
                return f.charAt(0).toUpperCase() + f.substr(1)
            });
            jQuery("#quick_vertical_alert").html('<input type="checkbox" name="' + b + ' Brief" value="1" checked /><span class="">&nbsp;' + __("Sign me up for the HuffPost %s Newsletter", b) + "</span>").removeClass("hidden")
        }
        return;
        if (typeof(QuickFan._HeaderText) != "undefined") {
            $("header_id").innerHTML = QuickFan._HeaderText;
            delete QuickFan._HeaderText
        }
    },
    GetEmailAlertsFail: function() {
        HPError.e();
        HuffConnect.hideModal();
        return 
    },
    sendForm: function(b) {
        var e = jQuery("#quick_email_alerts_form").get(0);
        var b = b || false;
        if (!b) {
            if (!HPUtil.checkEmail(e.blogger_alert_email.value)) {
                jQuery("#qa_error_notify").html(__("Please submit a valid email address"));
                jQuery("#qa_error_notify").removeClass("hidden");
                e.blogger_alert_email.value = "";
                e.blogger_alert_email.focus();
                return false
            }
        }
        var d = "";
        d += jQuery("#save").attr("name") + "=" + jQuery("#save").val() + "&";
        d += e.blogger_alert_email.name + "=" + e.blogger_alert_email.value + "&";
        d += document.getElementById("notify[blogger]").name + "=" + document.getElementById("notify[blogger]").value + "&";
        d += document.getElementById("blogger_name").name + "=" + document.getElementById("blogger_name").value + "&";
        for (var a = 0; a < e.length; a++) {
            if (e[a].type == "checkbox") {
                var c = (e[a].checked) ? 1: 0;
                d += "&" + e[a].name + "=" + c
            }
        }
        jQuery("#quick_alert_button").addClass("hidden");
        jQuery("#quick_alert_loader").removeClass("hidden");
        jQuery.ajax({
            url: e.action,
            type: "POST",
            data: d,
            success: QuickFan.Success,
            error: QuickFan.Fail
        });
        return false
    },
    focusOnBox: function(a) {
        if (jQuery(a).val() == __("Type your email address here...")) {
            jQuery(a).val("")
        }
        return 
    },
    blurOnBox: function(a) {
        if (jQuery(a).val() == "") {
            jQuery(a).val(__("Type your email address here..."))
        }
        return 
    }
};
huff.unit("tmp/quick_fan", function(a) {
    a(QuickFan)
});
window._ || (window._ = function(a) {
    return a
});
window.ngettext || (window.ngettext = function(a) {
    return a
});
window.sprintf || (window.sprintf = function(a) {
    return a
});
HPConfig.current_web_address = HPConfig.current_web_address.replace(/^http:\/\//i, "");
HPConfig.current_web_domain = HPConfig.current_web_domain.replace(/^http:\/\//i, "");
var Provider = {
    auto_link_email: true,
    force_login: false,
    user_provider_info: {
        facebook: {},
        twitter: {},
        yahoo: {},
        google: {},
        "google-plus": {},
        linkedin: {},
        aol: {},
        winlive: {}
    },
    login_config: {
        facebook: {},
        twitter: {
            w: 800,
            h: 680
        },
        yahoo: {
            w: 500,
            h: 500
        },
        google: {
            w: 850,
            h: 600
        },
        "google-plus": {
            w: 520,
            h: 600
        },
        linkedin: {
            w: 600,
            h: 500
        },
        aol: {
            w: 520,
            h: 500
        },
        winlive: {
            w: 500,
            h: 500
        }
    },
    SignUp: function(e, h) {
        var d = "";
        if ("object" == typeof e) {
            for (var c in e) {
                d = "&" + c + "=" + e[c] + ","
            }
            d = d.substr(0, d.length - 1)
        }
        var g = HPConfig.current_web_address;
        var b = "";
        if (HPConfig.site) {
            g = HPConfig.huffsite;
            b = "&multid=" + HPConfig.site
        }
        var f = "";
        if (Provider.force_login) {
            f = "&force_login=true"
        }
        var a = "http://" + g + "/users/signup/provider/index.php?type=" + h + d + "&" + Math.random() + b + f;
        PopupManager.open(a, Provider.login_config[h].w, Provider.login_config[h].h);
        PopupManager.onClose = function() {
            Provider.Connect(h)
        }
    },
    Connect: function(a) {
        if ("undefined" != typeof(Provider.user_provider_info[a].id)) {
            HuffConnect.loadConnectModal({
                provider: a
            })
        }
        return 
    },
    QuickLogin: function(g) {
        var a = linkSocialAccount;
        if ("undefined" == typeof g) {
            g = "yahoo"
        }
        var f = HPConfig.current_web_address;
        var c = "";
        if (HPConfig.site) {
            f = HPConfig.huffsite;
            c = "&multid=" + HPConfig.site
        }
        if (window.HPUtil && window.HPUtil.isHPLiveHost()) {
            c = "&multid=hpsn"
        } else {
            if (window.HPUtil && window.HPUtil.isHPBetaLiveHost()) {
                c = "&multid=hpsnbeta"
            }
        }
        var d = jQuery.I18N.getLocale() || false;
        if (d) {
            c += "&locale=" + d
        }
        var e = "";
        if (Provider.force_login) {
            e = "&force_login=true"
        }
        var b = "http://" + f + "/users/login/provider/index.php?type=" + g + "&" + Math.random() + c + e;
        if (a.linkOptions.disable_connect) {
            b += "&disable_connect=1"
        }
        if ("new_user" == Provider.user_provider_info[g].Status) {
            Provider.LoginConnect(g);
            return 
        }
        PopupManager.open(b, Provider.login_config[g].w, Provider.login_config[g].h);
        PopupManager.onClose = function() {
            Provider.LoginConnect(g)
        }
    },
    resetProviderStatus: function(a) {
        Provider.user_provider_info[a] = {}
    },
    LoginConnect: function(c) {
        var b = HuffConnect.Login;
        var a = linkSocialAccount;
        if ("undefined" != typeof(Provider.user_provider_info[c].Status)) {
            if (!a.linkOptions.disable_connect) {
                if (Provider.user_provider_info[c].Status == "existed_user") {
                    huff.use("connect/auth", function(d) {
                        d.Login.loginSuccess()
                    })
                } else {
                    if (Provider.user_provider_info[c].Status == "new_user") {
                        huff.use("connect/provider", function(d) {
                            d.loadModal(c)
                        })
                    } else {
                        if (Provider.user_provider_info[c].Status == "self_closed_user") {
                            huff.use("connect/provider", function(d) {
                                d.loadModal(c)
                            })
                        } else {
                            if ("undefined" !== typeof Provider.user_provider_info[c].Status) {
                                huff.use("connect/provider", function(d) {
                                    d.loadModal(c)
                                })
                            } else {
                                HPError.e("something went wrong")
                            }
                        }
                    }
                }
            } else {
                if (typeof a.linkOptions.callback == "function") {
                    a.linkOptions.callback(Provider.user_provider_info[c].Status)
                }
            }
        } else {
            HPError.e("something else went wrong")
        }
    }
};
var linkSocialAccount = {
    provider: "",
    facebookInfo: {},
    gfcInfo: {},
    onLinkSuccess: false,
    onLinkFailed: false,
    linkOptions: {},
    check_fans: true,
    checkLoginStatus: function(h, c, g, b) {
        this.linkOptions = g || {};
        b = b || false;
        if ("function" == typeof(b)) {
            b()
        } else {
            HPFB.force_login = false;
            HPFB.overridden_permissions = false;
            HPFB.onAuthenticateRequestSuccess = false
        }
        var e = SNProject;
        if ("undefined" === typeof h) {
            return 
        }
        if (c && typeof c.onLinkSuccess == "function") {
            this.onLinkSuccess = c.onLinkSuccess
        }
        this.onLinkFailed = false;
        if (c && typeof c.onLinkFailed == "function") {
            this.onLinkFailed = c.onLinkFailed
        }
        if (HuffCookies.getUserName()) {
            if (HuffPrefs.get(h + "")&&!this.linkOptions.reauth) {
                if (c && typeof c.linked === "function") {
                    c.linked()
                }
                if ("facebook" === h) {
                    var a = this;
                    var f = new Date().getTime();
                    var d = function(i) {
                        a.provider = "facebook";
                        if (!i.model.is_verified) {
                            linkSocialAccount.FacebookPromtLogin()
                        }
                    };
                    jQuery.ajax({
                        url: "/conversations/users/" + HuffCookies.getUserId() + "?app_token=d6dc44cc3ddeffb09b8957cf270a845d&_=" + f,
                        type: "GET",
                        success: d,
                        error: HuffConnect.Login.FacebookFail
                    })
                }
                return 
            }
            if ("facebook" === h) {
                this.provider = "facebook";
                linkSocialAccount.FacebookPromtLogin()
            } else {
                if ("undefined" != typeof Provider.login_config[h]) {
                    this.provider = h;
                    linkSocialAccount.PromptLogin()
                }
            }
        } else {
            if ("facebook" === h) {
                HPFB.login(e._tryToJoinUser)
            } else {
                if ("undefined" != typeof Provider.login_config[h]) {
                    Provider.QuickLogin(h)
                }
            }
        }
    },
    FacebookPromtLogin: function() {
        var a = HuffConnect.Login;
        HPFB.waitForSession(function() {
            if ("connected" == HPFB.user_status) {
                var b = new Date().getTime();
                var c = HPFB.getSessionForServer();
                jQuery.ajax({
                    url: "/users/login/provider/check_user_status.php",
                    type: "POST",
                    data: "provider=facebook" + (c ? "&" + c : ""),
                    success: linkSocialAccount.FacebookSuccess,
                    error: a.FacebookFail
                })
            } else {
                if ("notConnected" == HPFB.user_status) {
                    return 
                }
            }
        })
    },
    FacebookSuccess: function(e, d, c) {
        var b = HuffConnect.Login;
        var a = JSON.parse(e);
        switch (a.msg) {
        case"existed_user":
        case"current_user":
        case"new_user":
            linkSocialAccount.facebookInfo.fb_id = a.fb_id || "";
            linkSocialAccount.facebookInfo.profile_image = a.fb_id ? "http://graph.facebook.com/" + a.fb_id + "/picture?type=square" : "";
            linkSocialAccount.facebookInfo.user_id = a.user_id || "";
            linkSocialAccount.facebookInfo.user_name = a.user_name || "";
            linkSocialAccount.facebookInfo.full_name = a.full_name || (a.user_name || "");
            linkSocialAccount.facebookInfo.email = a.facebook_email || "";
            linkSocialAccount.facebookInfo.Status = a.msg;
            HuffConnect.hideModal();
            if (linkSocialAccount.linkOptions.disable_connect === true) {
                linkSocialAccount.linkOptions.callback("facebook", a.msg)
            } else {
                linkSocialAccount.PromptForConnectPop()
            }
            break;
        case"unverified_user":
            unverified_fb_error_el = jQuery("#connect_modal_v2 .facebook_link_tpl #main_notify");
            unverified_fb_error_el.fadeIn();
        default:
            b.FacebookFail(a);
            break
        }
        return 
    },
    PromptLogin: function() {
        var c = HPConfig.current_web_address;
        var b = "";
        if (HPConfig.site) {
            c = HPConfig.huffsite;
            b = "&multid=" + HPConfig.site
        }
        if (window.HPUtil && window.HPUtil.isHPLiveHost()) {
            b = "&multid=hpsn"
        } else {
            if (window.HPUtil && window.HPUtil.isHPBetaLiveHost()) {
                b = "&multid=hpsnbeta"
            }
        }
        var a = "http://" + c + "/users/login/provider/link_login.php?provider=" + linkSocialAccount.provider + "&" + Math.random() + b;
        if (this.linkOptions.reauth) {
            a += "&reauth=1"
        } else {
            if (this.linkOptions.disable_connect) {
                a += "&disable_connect=1"
            }
        }
        if (!this.linkOptions.popup) {
            if (Provider.user_provider_info[linkSocialAccount.provider] && Provider.user_provider_info[linkSocialAccount.provider].Status) {
                linkSocialAccount.PromptLoginConnect()
            } else {
                if ("undefined" != typeof Provider.login_config[linkSocialAccount.provider]) {
                    PopupManager.open(a, Provider.login_config[linkSocialAccount.provider].w, Provider.login_config[linkSocialAccount.provider].h);
                    PopupManager.onClose = function() {
                        linkSocialAccount.PromptLoginConnect()
                    }
                }
            }
        } else {
            setTimeout(function() {
                linkSocialAccount.linkOptions.popup.location = a;
                if ("twitter" == linkSocialAccount.provider) {
                    linkSocialAccount.linkOptions.popup.resizeTo(800, 550)
                }
                PopupManager.onClose = function() {
                    linkSocialAccount.PromptLoginConnect()
                }
            }, 100)
        }
    },
    PromptLoginConnect: function() {
        var a = linkSocialAccount;
        if (Provider.user_provider_info[a.provider].Status && "undefined" != typeof(Provider.user_provider_info[a.provider])) {
            if (a.linkOptions.disable_connect) {
                if (typeof a.linkOptions.callback == "function") {
                    a.linkOptions.callback();
                    return 
                }
            } else {
                linkSocialAccount.PromptForConnectPop()
            }
        } else {
            HuffConnect.hideModal();
            HPError.e("something went wrong")
        }
        return 
    },
    PromptForConnectPop: function() {
        HuffConnect.hideModal();
        if (linkSocialAccount.provider == "facebook") {
            var a = linkSocialAccount.facebookInfo.Status
        } else {
            var a = Provider.user_provider_info[linkSocialAccount.provider].Status
        }
        var c = this.getPromptPopHtml(a);
        if (c != "" || linkSocialAccount.provider == "facebook") {
            var d = {
                inner_html: c,
                width: 550
            };
            if (/existed_user/.test(a)) {
                d.cb = linkSocialAccount.callback;
                if (linkSocialAccount.provider == "facebook") {
                    var b = linkSocialAccount.facebookInfo.user_id
                } else {
                    var b = Provider.user_provider_info[linkSocialAccount.provider].user_id
                }
                if (b == HPUser.user_id()&&!this.linkOptions.reauth) {
                    alert("You are already linked to this " + linkSocialAccount.provider.charAt(0).toUpperCase() + linkSocialAccount.provider.substr(1, linkSocialAccount.provider.length) + " account!");
                    return 
                } else {
                    if (linkSocialAccount.provider == "facebook") {
                        alert(_("Your Facebook account is already linked to a different HuffPost account. Please log out and then sign into HuffPost using your Facebook account."));
                        return 
                    }
                }
            }
            if (this.linkOptions.reauth) {
                linkSocialAccount.ConnectProvider();
                return 
            }
            if (linkSocialAccount.provider == "facebook") {
                huff.use("connect/provider", function(e) {
                    if (!/existed_user/.test(a)) {
                        e.showVerifyModal()
                    }
                })
            } else {
                HuffConnect.socialModal(d)
            }
        } else {
            HuffConnect.hideModal();
            HPError.e("something went wrong")
        }
        return 
    },
    getPromptPopHtml: function(a) {
        var b = "";
        var c = HuffCookies.getUserName().replace("hp_blogger_", "").replace("_", " ");
        var d = linkSocialAccount.provider.charAt(0).toUpperCase() + linkSocialAccount.provider.substr(1);
        if (linkSocialAccount.provider != "facebook") {
            if (/new_user/.test(a)) {
                b += '<div class="align_left modal_link_your_social_account"><div class="connect_pop_head">' + sprintf(_("Link your %s account?"), d) + '</div><p class="connect_pop_name">' + sprintf(_("Hi %s"), c) + ',</p><br /><p class="connect_pop_main_text">' + _("Would you like to link your account to your Huffington Post account?") + '</p><br /><p class="connect_pop_sub_text">' + sprintf(_("By linking your %s account to your Huffington Post account, you will be able to log in to Huffpost using your %s credentials. You will also gain access to new features and be able to more easily see what your %s friends are up to on this site.</p>"), d, d, d) + '<br /><div id="user_link_reaction"><div class="connect_pop_yes" onclick="linkSocialAccount.ConnectProvider(); return false;">' + _("Yes") + '</div><div class="connect_pop_no" onclick="linkSocialAccount.closePromptPop(); return false;">' + _("No") + '</div></div><div class="clear"></div><br /><br /><p class="connect_pop_notyou">' + sprintf(_("Not %s? Please %sclick here%s to log out, then log back in with your correct account."), c, '<a href="/users/logout/" onclick="QuickLogin.SocialLogout(\'/users/logout/\'); return false;">', "</a>") + "</p></div></div><br />"
            } else {
                if (/existed_user/.test(a)) {
                    oldUserName = Provider.user_provider_info[linkSocialAccount.provider].user_name;
                    b += '<div class="align_left modal_link_your_social_account"><div class="connect_pop_head">' + sprintf(_("Do you wish to re-link your %s account?"), d) + '</div><p class="connect_pop_name">' + sprintf(_("Hi %s"), c) + ',</p><br /><p class="connect_pop_sub_text">' + sprintf(_("This %s account is already linked to the Huffington Post username '%s'. Would you like to instead associate it with your current account '%s'?"), d, oldUserName, c) + '</p><br /><div id="user_link_reaction"><div class="connect_pop_yes" onclick="linkSocialAccount.ConnectProvider(); return false;">' + _("Yes") + '</div><div class="connect_pop_no" onclick="linkSocialAccount.closePromptPop(); return false;">' + _("No") + '</div></div><div class="clear"></div></div><br /><br /></div>'
                }
            }
        }
        return b
    },
    closePromptPop: function() {
        HuffConnect.hideModal(linkSocialAccount.onLinkFailed);
        return 
    },
    ConnectProvider: function() {
        var i = HuffConnect.Login;
        var d = linkSocialAccount;
        var a = jQuery("#huffpo_snn_is_loading");
        var e = jQuery("#user_link_reaction");
        if (a.length) {
            a.css("display", "block")
        }
        if (e.length) {
            e.css("display", "none")
        }
        var b = 0;
        var h = 0;
        var j = "";
        var k = "";
        var g = "";
        var f = "";
        var c = new Date().getTime();
        switch (d.provider) {
        case"facebook":
            b = d.facebookInfo.fb_id;
            f = d.facebookInfo.Status;
            h = d.facebookInfo.user_id || "";
            j = d.facebookInfo.user_name || "";
            k = "&" + HPFB.getSessionForServer() + "&" + jQuery("#quicksignup_form").serialize();
            break;
        case"twitter":
            b = Provider.user_provider_info[d.provider].id;
            f = Provider.user_provider_info[d.provider].Status;
            h = Provider.user_provider_info[d.provider].user_id || "";
            j = Provider.user_provider_info[d.provider].user_name || "";
            k = "&name=" + Provider.user_provider_info.twitter.screen_name + "&t=" + encodeURIComponent(Provider.user_provider_info.twitter.oauth_token) + "&s=" + encodeURIComponent(Provider.user_provider_info.twitter.oauth_token_secret) + "&img=" + escape(Provider.user_provider_info.twitter.profile_image);
            break;
        case"yahoo":
        case"google":
        case"google-plus":
        case"linkedin":
            b = Provider.user_provider_info[d.provider].id;
            f = Provider.user_provider_info[d.provider].Status;
            h = Provider.user_provider_info[d.provider].user_id || "";
            j = Provider.user_provider_info[d.provider].user_name || "";
            k = "&access_token=" + encodeURIComponent(Provider.user_provider_info[d.provider].access_token) + "&img=" + escape(Provider.user_provider_info[d.provider].profile_image);
            break;
        case"aol":
            b = Provider.user_provider_info[d.provider].id;
            h = Provider.user_provider_info[d.provider].user_id || "";
            j = Provider.user_provider_info[d.provider].user_name || "";
            f = Provider.user_provider_info[d.provider].Status + "&img=" + escape(Provider.user_provider_info.aol.profile_image);
            break;
        case"winlive":
            b = Provider.user_provider_info[d.provider].WinliveId;
            f = Provider.user_provider_info[d.provider].Status;
            h = Provider.user_provider_info[d.provider].user_id || "";
            j = Provider.user_provider_info[d.provider].user_name || "";
            k = "&WinliveAccessToken=" + encodeURIComponent(Provider.user_provider_info[d.provider].WinliveAccessToken) + "&WinliveAccessTokenExpiresAt=" + encodeURIComponent(Provider.user_provider_info[d.provider].WinliveAccessTokenExpiresAt) + "&WinliveRefreshToken=" + encodeURIComponent(Provider.user_provider_info[d.provider].WinliveRefreshToken) + "&img=" + escape(Provider.user_provider_info[d.provider].profile_image);
            break;
        default:
            HuffConnect.hideModal();
            HPError.e(_("something went wrong"));
            return 
        }
        g = "provider=" + d.provider + "&puid=" + b + k + "&r=" + c;
        g += "&status=" + f + "&uid=" + h + "&uname=" + j;
        if (b != "0") {
            jQuery.ajax({
                url: "/users/login/provider/link_provider.php",
                type: "POST",
                data: g,
                success: linkSocialAccount.LinkSuccess,
                error: i.FacebookFail
            })
        } else {
            if (d.linkOptions.reauth && b == "0") {
                if (typeof linkSocialAccount.onLinkSuccess == "function") {
                    linkSocialAccount.onLinkSuccess()
                }
            }
        }
        return 
    },
    LinkSuccess: function(c, b, a) {
        if (jQuery("#huffpo_snn_is_loading").length) {
            jQuery("#huffpo_snn_is_loading").css("display", "none")
        }
        if (jQuery("#user_link_reaction").length) {
            jQuery("#user_link_reaction").html(c)
        }
        if (jQuery("#user_relink_reaction").length) {
            jQuery("#user_relink_reaction").css("display", "block")
        }
        switch (c) {
        case"success":
            huff.use("connect/auth", function(d) {
                d.FireEvent("auth.onVerify")
            });
            if (!linkSocialAccount.linkOptions.reauth) {
                HuffConnect.hideModal()
            }
            if (linkSocialAccount.check_fans) {
                HuffCookies.setCookie("check_for_fans", 1)
            }
            if (typeof linkSocialAccount.onLinkSuccess == "function") {
                return linkSocialAccount.onLinkSuccess()
            } else {
                if (typeof IS_NEWSGLIDE === "boolean" && IS_NEWSGLIDE) {
                    $(document).trigger("userlogedin");
                    HuffConnect.hideModal()
                } else {
                    location.href = location.href
                }
            }
            break;
        case"error":
            linkSocialAccount.closePromptPop();
            break;
        default:
            linkSocialAccount.closePromptPop();
            break
        }
    },
    callback: function() {
        HuffConnect.hideModal();
        if (typeof IS_NEWSGLIDE === "boolean" && IS_NEWSGLIDE) {
            $(document).trigger("userlogedin");
            HuffConnect.hideModal()
        } else {
            location.href = location.href
        }
    }
};
window._ || (window._ = function(a) {
    return a
});
window.ngettext || (window.ngettext = function(a) {
    return a
});
window.sprintf || (window.sprintf = function(a) {
    return a
});
PlaceTools = function(a, c, b) {
    ps = c.getElementsByTagName("P");
    id = a.id;
    if (ps.length > 2) {
        ps[2].parentNode.insertBefore(a, ps[2]);
        HuffPoUtil.show(a.id)
    } else {
        if (ps.length > 1) {
            ps[1].parentNode.insertBefore(a, ps[1]);
            HuffPoUtil.show(a.id)
        } else {
            if (ps.length == 1 && ps[0].innerHTML.match(/(<br.?>\s*?<br.?>)/)) {
                outerHTML = '<div class="' + a.className + '" id="' + a.id + '">' + a.innerHTML + "</div>";
                a.parentNode.removeChild(a);
                ps[0].innerHTML = ps[0].innerHTML.replace(/(<br.?>\s*?<br.?>)/, "<br><br>" + outerHTML);
                HuffPoUtil.show(a.id)
            }
        }
    }
    y1 = HuffPoUtil.getRegion(id).top;
    y2 = HuffPoUtil.getRegion(b).top;
    if (y1 && y2 && (y1 + 150) >= y2) {
        HuffPoUtil.hide(id)
    }
};
function addBookmark(a, b) {
    b = b || __("HuffingtonPost");
    a = (!a) ? location.href : a;
    b = (!b) ? document.title : b;
    if ((typeof window.sidebar == "object") && (typeof window.sidebar.addPanel == "function")) {} else {
        if (typeof window.external == "object") {
            window.external.AddFavorite(a, b)
        } else {
            if (window.opera && document.createElement) {
                return true
            } else {
                return false
            }
        }
    }
    return false
}
function addBookmark_mac(a, b) {
    if (document.all) {
        window.external.AddFavorite(a, b)
    } else {
        window.sidebar.addPanel(a, b)
    }
}
SharePost = {
    share_form_ajax_response: false,
    cache: {},
    contacts: {
        yahoo: {},
        google: {},
        aol: {},
        winlive: {}
    },
    process_type: "regular",
    files_loaded: false,
    contacts_details: new Array(),
    pop: function(c, b, a) {
        this.trackingPixelOnShareAction();
        this.emailForm(c, b, a);
        return 
    },
    fill_me: function(f, d) {
        var a = document.forms.email_share_form.your_name.value;
        var b = a.split(" ");
        var c = "";
        var e = 0;
        for (z = 0; (z < b.length && e++<3); z++) {
            c = c + " " + b[z].replace(/[^a-zA-Z]/g, "")
        }
        a = c.replace(/^\s+|\s+$/g, "")
    },
    emailForm: function(c, b, a) {
        this.trackingPixelOnShareAction();
        this.share_id = c;
        this.big_news_title = a;
        HPTrack.trackPageview("/t/a/initiate_share");
        if (typeof Badges != "undefined") {
            Badges.trackShare({
                eid: c,
                badge: "email"
            })
        }
        var d = {
            share_id: c || "",
            vertical: b || "",
            title: this.big_news_title || ""
        };
        this.loadModal(d);
        return 
    },
    loadModal: function(b) {
        this.config = b;
        if (this.cache[this.config.share_id]) {
            var a = {
                width: 650,
                inner_html: this.cache[this.config.share_id].html,
                modal_head_image: "",
                modal_subhead_image: "",
                onshow: function() {
                    HuffPoUtil.EvalScript(SharePost.cache[SharePost.config.share_id].html);
                    FB.XFBML.parse(jQuery(".email_share_block").get(0), function() {})
                }
            };
            HuffConnect.socialModal(a)
        } else {
            HuffConnect.showLoader();
            this.loadForm()
        }
        this.process_type = "regular";
        return 
    },
    loadForm: function() {
        var a = "/social/entry_share_v2.php";
        var b = this;
        jQuery.ajax({
            url: a,
            type: "GET",
            data: "share_id=" + this.config.share_id,
            success: function(e, d, c) {
                if (c.status != 200) {
                    HuffConnect.hideModal();
                    return 
                }
                b.loadFormSuccess(e)
            },
            error: function() {
                HuffConnect.hideModal();
                HPError.e();
                return 
            }
        });
        return 
    },
    loadFormSuccess: function(b) {
        oResp = JSON.parse(b);
        if (oResp.status && /success/.test(oResp.status) && oResp.html) {
            oResp.html = "<div class='align_left'>" + oResp.html + "</div>";
            var a = function() {
                var c = jQuery("#email_share_title");
                var e = jQuery("#email_share_title_success");
                if (c.length) {
                    var d = "";
                    if (SharePost.config && SharePost.config.title) {
                        d = SharePost.config.title
                    } else {
                        if (d == "" && jQuery("#title_permalink")) {
                            d = jQuery("#title_permalink").html()
                        } else {
                            if (jQuery("#title_permalink_bold")) {
                                d = jQuery("#title_permalink_bold").html()
                            }
                        }
                    }
                    c.html(d);
                    e.html('"' + d + '"')
                }
                FB.XFBML.parse(jQuery(".email_share_block").get(0), function() {});
                HuffPoUtil.EvalScript(oResp.html);
                SharePost.cache[SharePost.config.share_id] = {
                    html: oResp.html
                };
                return 
            };
            HuffConnect.socialModal({
                width: 650,
                inner_html: oResp.html,
                modal_head_image: "",
                modal_subhead_image: "",
                onshow: a
            })
        } else {
            HuffConnect.hideModal()
        }
        return 
    },
    selectOption: function(f) {
        var h = ["gmail", "yahoo", "aol", "winlive"];
        var d = document.getElementById("email_share_radio_" + f);
        var j = jQuery("#single_box_" + f);
        var g = jQuery("#mailbox_arrow_" + f);
        HPTrack.trackEvent("Email Share", "Select Import Provider", f);
        for (var e = 0; e < h.length; e++) {
            var c = jQuery("#single_box_" + h[e]);
            var b = jQuery("#mailbox_arrow_" + h[e]);
            var a = jQuery("#email_share_radio_" + h[e]);
            if (c) {
                jQuery(c).removeClass("selected_box")
            }
            if (b) {
                jQuery(b).addClass("hidden")
            }
            if (a) {
                a.checked = false
            }
        }
        if (d) {
            d.checked = true
        }
        if (j.length) {
            j.addClass("selected_box")
        }
        if (g.length) {
            g.removeClass("hidden")
        }
        if (f) {
            this.importContacts(f)
        }
        this.hideMessage();
        return 
    },
    send: function() {
        var c = this.validateForm();
        if (typeof c == "object") {
            jQuery("#es_send_button").addClass("hidden");
            jQuery("#es_send_status").removeClass("hidden");
            var b = "/send/send.php", f = escape(c.id);
            var d = "id=" + f + "&your_name=" + encodeURIComponent(c.your_name.substr(0, 30)) + "&your_email=" + encodeURIComponent(c.your_email.substr(0, 320)) + "&email_list=" + encodeURIComponent(c.email_list) + "&modulous_email_pre=" + (HPConfig.modulous_email_pre || "") + "&modulous_email_linked=" + (HPConfig.modulous_email_linked || "") + "&modulous_email_post=" + (HPConfig.modulous_email_post || "");
            if (f==-1) {
                d += "&url=" + encodeURIComponent(window.location.href) + "&title=" + encodeURIComponent(document.title)
            }
            if ((typeof social_campaign !== "undefined") && (typeof social_campaign_creative !== "undefined") && (typeof social_campaign_creative.email !== "undefined")) {
                var a = "";
                if ((typeof social_campaign_creative.email.link !== "undefined") && (typeof social_campaign_creative.email.link !== "function") && (social_campaign_creative.email.link != "")) {
                    a += (typeof social_campaign_creative.email.pre_link !== "undefined") ? encodeURIComponent(social_campaign_creative.email.pre_link.replace(/\\+"/g, '"')) : "";
                    a += '<a href="' + social_campaign_creative.email.link + '">' + ((typeof social_campaign_creative.email.text !== "undefined") ? encodeURIComponent(social_campaign_creative.email.text.replace(/\\+"/g, '"')) : "") + "</a>";
                    a += (typeof social_campaign_creative.email.post_link !== "undefined") ? encodeURIComponent(social_campaign_creative.email.post_link.replace(/\\+"/g, '"')) : ""
                } else {
                    a += encodeURIComponent(social_campaign_creative.email.replace(/\\+"/g, '"'))
                }
                d += "&share_ad_text=" + a;
                d += "&share_link_ref=" + encodeURIComponent(social_campaign)
            } else {
                if ((typeof social_campaign != "undefined") && HPAds.social_campaigns && HPAds.social_campaigns[social_campaign] && HPAds.social_campaigns[social_campaign].email_message) {
                    d += "&share_ad_text=" + encodeURIComponent(HPAds.social_campaigns[social_campaign].email_message.replace(/\\+"/g, '"'));
                    d += "&share_link_ref=" + encodeURIComponent(social_campaign)
                } else {
                    if (SharePost.share_email_comm_text) {
                        d += "&share_ad_text=" + encodeURIComponent(SharePost.share_email_comm_text)
                    }
                }
            }
            d += "&send_this=1";
            var e = jQuery(location).attr("search") || "";
            e = decodeURIComponent(e);
            e = e.replace(/\s/g, "");
            e = e.substr(0, 255);
            if (e !== "") {
                d += "&get=" + encodeURIComponent(e)
            }
            d += "&banners=" + HPAds.get_banners_str();
            jQuery.ajax({
                type: "POST",
                url: b,
                data: d,
                dataType: "json",
                error: function() {
                    SharePost.sendFailure()
                },
                success: SharePost.sendSuccess
            })
        }
        return 
    },
    sendSuccess: function(a) {
        if (a && a.status && /success/.test(a.status)) {
            jQuery("#es_share_started").addClass("hidden");
            jQuery("#es_share_ended").removeClass("hidden");
            jQuery("#es_send_status").addClass("hidden");
            jQuery("#es_send_button").removeClass("hidden");
            jQuery("#es_error_dialogue").addClass("hidden");
            var d = jQuery("#es_share_ended .title");
            if (d.html() == '""') {
                d.html('"' + document.title + '"')
            }
            var b = window.document.email_share_form;
            if (b.email_list) {
                b.email_list.value = ""
            }
            if (b.your_name) {
                b.your_name.value = ""
            }
            if (b.your_email) {
                b.your_email.value = ""
            }
            SharePost.selectAll(false);
            window.location.hash = "#es_share_ended"
        } else {
            var c = a.message || false;
            SharePost.sendFailure(c)
        }
        return 
    },
    sendFailure: function(a) {
        jQuery("#es_send_status").addClass("hidden");
        jQuery("#es_send_button").removeClass("hidden");
        SharePost.displayMessage(a);
        return 
    },
    validateForm: function() {
        var b = window.document.email_share_form;
        if (this.process_type == "regular") {
            if (b.email_list.value == "") {
                SharePost.displayMessage(__("Send to email(s) box cannot be empty"));
                b.email_list.focus();
                return 
            } else {
                if (b.your_name.value == "") {
                    SharePost.displayMessage(__("Please enter your name"));
                    b.your_name.focus();
                    return 
                } else {
                    if (b.your_email.value == "") {
                        SharePost.displayMessage(__("Please enter your email"));
                        b.your_email.focus();
                        return 
                    }
                }
            }
            var a = {
                id: b.email_share_id.value,
                your_name: b.your_name.value,
                your_email: b.your_email.value,
                email_list: b.email_list.value
            }
        } else {
            if (this.process_type == "import") {
                if (b.email_list.value == "") {
                    var d = __("Please enter the email addresses");
                    if (this.total_contacts_found == 0) {
                        d = __("No contacts found")
                    }
                    SharePost.displayMessage(d);
                    return 
                }
                var c = (this.contacts[this.current_option]);
                var a = {
                    id: b.email_share_id.value,
                    your_name: c.name,
                    your_email: c.email,
                    email_list: b.email_list.value
                }
            }
        }
        return a
    },
    displayMessage: function(b) {
        if (b) {
            var a = b
        } else {
            var a = __("Unable to process!")
        }
        jQuery("#es_error_dialogue").html(a);
        jQuery("#es_error_dialogue").removeClass("hidden");
        return 
    },
    hideMessage: function() {
        jQuery("#es_error_dialogue").addClass("hidden");
        return 
    },
    importContacts: function(d) {
        if (d == "gmail") {
            d = "google"
        }
        if (this.contacts[d].linked) {
            this.importContactsSuccess(d)
        } else {
            if (HPConfig.site) {
                domain = "http://" + HPConfig.huffsite;
                multid = "&multid=" + HPConfig.site
            } else {
                domain = HuffPoUtil.getHostName();
                multid = ""
            }
            var b = domain + "/users/contacts/index.php?provider=" + d + "&nofilter=1" + multid;
            var a = 500, c = 500;
            if (d == "gmail") {
                a = 860, c = 550
            }
            PopupManager.open(b, a, c);
            PopupManager.onClose = function() {
                SharePost.importContactsSuccess(d);
                huff.use("provider/logout", function(e) {
                    e.OnAfterContactsImport(d)
                })
            }
        }
        return 
    },
    importContactsSuccess: function(f) {
        jQuery("#es_share_started").removeClass("hidden");
        jQuery("#es_share_ended").addClass("hidden");
        if (this.contacts[f].linked) {
            var d = f;
            if (d == "google") {
                d = "gmail"
            }
            var c = jQuery("#es_mailbox_rad_" + d);
            var g = jQuery("#email_form_left_section");
            var b = jQuery("#es_provider");
            var l = jQuery("#es_pic");
            var j = jQuery("#es_user_data");
            var k = (this.contacts[f]);
            c.html("").addClass("selected_mailbox email_share_pixie");
            b.html(f.charAt(0).toUpperCase() + f.substr(1));
            if (k.avatar) {
                l.html('<div class="mailbox_avatar float_left" style="margin-right:2px;"><img width="27" height="27" src="' + k.avatar + '" /></div>')
            } else {
                l.html('<div class="email_share_pixie mailbox_avatar float_left avatar_pic_' + f + '"></div>')
            }
            j.html(k.name + " (" + k.email + ")");
            if ("string" === typeof k.contacts) {
                k.contacts = JSON.parse(k.contacts)
            }
            if (k.contacts && k.contacts.length) {
                var h = "";
                h += '<div class="es_margins">';
                h += '	<div class="es_form_label uppercase float_left">' + __("SEND TO EMAIL(S)") + "</div>";
                h += '	<div class="email_share_pixie es_required_ico float_left"></div>';
                h += '	<div class="clear"></div>';
                h += '	<div class="es_input_box"><input type="text" class="email_share_box email_listing_box" name="email_list" id="cbox_email_listing" /></div>';
                h += '<input type="hidden" id="project-id" />';
                h += '	<div class="es_form_small_text float_left es_helper_text">' + __("Separate multiple addresses with comma") + "</div>";
                h += '	<div class="clear"></div>';
                h += "</div>";
                h += "<div class='contact_list'>";
                this.contacts_details = new Array();
                for (var e = 0; e < k.contacts.length; e++) {
                    var a = cropped_email = "";
                    if (k.contacts[e].name) {
                        a = k.contacts[e].name
                    }
                    if (a.length > 12) {
                        a = a.substr(0, 9) + "..."
                    }
                    if (k.contacts[e].email) {
                        cropped_email = k.contacts[e].email
                    }
                    if (cropped_email.length > 22) {
                        cropped_email = cropped_email.substr(0, 19) + "..."
                    }
                    this.contacts_details.push(k.contacts[e].email);
                    h += "<div class='es_single_contact'>";
                    h += "<div class='es_contact_names float_left'>";
                    h += "<span class='es_username' title='" + k.contacts[e].name + "'>" + a + "</span>&nbsp;&nbsp;<span class='es_email' title='" + k.contacts[e].email + "'>" + cropped_email + "</span>";
                    h += "</div>";
                    h += "<div class='es_contact_cbox float_right'>";
                    h += "<input type='checkbox' name='" + k.contacts[e].email + "' value='" + k.contacts[e].email + "' onclick='SharePost.selectOne(this)' />";
                    h += "</div>";
                    h += "</div>";
                    h += "<div class='clear es_contacts_border'></div>"
                }
                h += "</div>";
                h += "<div class='es_selectall'><input type='checkbox' name='select_all_contacts' value='all' onclick='SharePost.selectAll(this.checked);' />&nbsp;" + __("Select All") + "</div>";
                g.html(h);
                this.total_contacts_found = k.contacts.length;
                this.loadFiles()
            } else {
                if ("" != typeof this.contacts[f].error && "" !== this.contacts[f].error) {
                    g.html("<div class='es_provider_err'>" + this.contacts[f].error + "</div>");
                    this.total_contacts_found = 0
                } else {
                    g.html("<div class='es_list_empty'>" + __("No contacts found!") + "</div>");
                    this.total_contacts_found = 0
                }
            }
            this.process_type = "import";
            this.current_option = f
        } else {
            this.selectOption()
        }
        return 
    },
    loadFiles: function() {
        if (!this.files_loaded) {
            var b = ["/include/lib/jquery/jquery.ui.core.min.js", "/include/lib/jquery/jquery.ui.widget.min.js", "/include/lib/jquery/jquery.ui.position.min.js", "/include/lib/jquery/jquery.ui.autocomplete.min.js"];
            var a = function() {
                SharePost.files_loaded = true;
                SharePost.showOptions()
            };
            if (HuffPoUtil && typeof(HuffPoUtil.loadAndRun) == "function") {
                HuffPoUtil.loadAndRun(b, a)
            }
        } else {
            this.showOptions()
        }
        return 
    },
    showOptions: function() {
        if (!this.files_loaded) {
            return 
        }
        jQuery.noConflict();
        (function(a) {
            a(function() {
                var b = SharePost.contacts_details;
                function c(e) {
                    return e.split(/,\s*/)
                }
                function d(e) {
                    return c(e).pop()
                }
                a("#cbox_email_listing").bind("keydown", function(e) {
                    if (e.keyCode === a.ui.keyCode.TAB && a(this).data("autocomplete").menu.active) {
                        e.preventDefault()
                    }
                }).autocomplete({
                    minLength: 0,
                    source: function(f, e) {
                        e(a.ui.autocomplete.filter(b, d(f.term)))
                    },
                    focus: function() {
                        return false
                    },
                    select: function(f, g) {
                        var e = c(this.value);
                        e.pop();
                        e.push(g.item.value);
                        SharePost.checkOption(g.item.value);
                        e.push("");
                        this.value = e.join(", ");
                        return false
                    },
                    open: function(e, f) {
                        a(".ui-autocomplete").css("z-index", 900);
                        return 
                    }
                })
            })
        })(jQuery);
        return 
    },
    checkOption: function(b) {
        var a = window.document.email_share_form;
        if (a[b] && a[b].type == "checkbox") {
            a[b].checked = true
        }
        return 
    },
    selectOne: function(b) {
        var d = jQuery("#cbox_email_listing");
        var c = b.value + ", ";
        if (b.checked) {
            var a = d.val();
            d.val(a + c)
        } else {
            d.val(d.val().replace(c, ""))
        }
        d.focus();
        this.setCursorPostion();
        return 
    },
    selectAll: function(e) {
        var d = window.document.email_share_form;
        var f = jQuery("#cbox_email_listing");
        var b = new Array();
        for (var c = 0; c < d.length; c++) {
            if (d[c].type == "checkbox" && d[c].value != "all") {
                d[c].checked = e;
                b.push(d[c].value)
            }
        }
        if (d.select_all_contacts) {
            d.select_all_contacts.checked = e
        }
        if (e) {
            var a = f.val();
            a += b.join(", ") + ", ";
            f.val(a)
        } else {
            f.val("")
        }
        f.focus();
        this.setCursorPostion();
        return 
    },
    setCursorPostion: function() {
        return 
    },
    trackingPixelOnShareAction: function() {
        if (typeof social_campaign_tracking != "undefined" && social_campaign_tracking.email_share_action) {
            if (typeof social_campaign_tracking.email_share_action == "object") {
                for (tracking_url in social_campaign_tracking.email_share_action) {
                    if ((typeof social_campaign_tracking.email_share_action[tracking_url] == "string") && (social_campaign_tracking.email_share_action[tracking_url] != "")) {
                        HPUtil.trackerImg(social_campaign_tracking.email_share_action[tracking_url], document.body)
                    }
                }
            } else {
                HPUtil.trackerImg(social_campaign_tracking.email_share_action, document.body)
            }
            if (social_campaign && HPAds.social_campaigns && HPAds.social_campaigns[social_campaign]) {
                HPUtil.trackerImg("http://pixel.quantserve.com/pixel/p-4azu6n0QT1rbs.gif?labels=social_action." + social_campaign + ".email", document.body)
            } else {
                if (this.tracking_flight_name) {
                    HPUtil.trackerImg("http://pixel.quantserve.com/pixel/p-4azu6n0QT1rbs.gif?labels=social_action." + this.tracking_flight_name + ".email", document.body)
                }
            }
        }
        HPUtil.updateInfluenceCookie("emailed", 1)
    }
};
function prepare_and_show_blogs_table_new(m) {
    if (blogs_amount == 0) {
        return 
    }
    var s = "";
    var o = "";
    var r = 0;
    var c = 0;
    var h = 0;
    m--;
    var d = rows_in_the_blogs_table * tds_in_the_blogs_table;
    var a = m * d + 1;
    d = d + a - 1;
    if (m > 1 && null == document.getElementById("related_blogs_" + a) && typeof prepare_and_show_blogs_table_new.pagination_requested == "undefined") {
        prepare_and_show_blogs_table_new.pagination_requested = true;
        var e = document.getElementById("related_blogs");
        e.innerHTML = "";
        e.style.background = "transparent url('http://s.huffpost.com/images/loader.gif') no-repeat center top";
        e.style.height = "60px";
        var n = "/entries/pagination_load.php?module=linked_entries&entry_id=" + HPUtil.GetEntryID() + "&blog_id=3&load_from=" + loaded_blogs_amount + "&showed_expanded_blogs_amount=" + showed_expanded_blogs_amount;
        jQuery.ajax({
            type: "GET",
            url: n,
            dataType: "text",
            error: function() {},
            success: function(t) {
                if (t) {
                    var j = document.getElementById("related_blogs");
                    j.style.background = "transparent";
                    j.style.height = "auto";
                    document.getElementById("related_blogs_ajax_loaded").innerHTML = t;
                    prepare_and_show_blogs_table_new(m + 1)
                }
            }
        });
        return 
    }
    for (i = a; i <= d; i++) {
        var l = i > blogs_amount ? "": document.getElementById("related_blogs_" + i).innerHTML;
        if ("undefined" !== typeof html && (html.search("div onclick=")>-1) && (r == 0)) {
            r = 1;
            o += '<div class="sponsored_preview"><div class="commercial_title commercial_title_' + HPConfig.current_vertical_name + '">SPONSORED FEATURE</div><div style="clear: both; line-height: 10px; height: 10px;"></div>'
        }
        if ("undefined" !== typeof html && (html.search("div onclick=")==-1) && (r == 1) && (c == 0)) {
            c = 1;
            o += "</div>"
        }
        o += l;
        if ((0 == i%2) && (i != a)) {
            o += '<div class="clear"></div>'
        }
        if (i > blogs_amount) {
            break
        }
    }
    h = 0;
    document.getElementById("related_blogs").innerHTML = o;
    o = "";
    if (blogs_amount <= (rows_in_the_blogs_table * tds_in_the_blogs_table)) {
        return 
    }
    i--;
    var q = "";
    if (m > 0) {
        q = q + '&nbsp;<a href="#" onclick="prepare_and_show_blogs_table_new(' + 1 + '); return false;">&laquo;&nbsp;' + __("First") + "</a>&nbsp;";
        var f = m;
        q = q + '&nbsp;<a href="#" onclick="prepare_and_show_blogs_table_new(' + f + '); return false;">&nbsp;' + __("Prev") + "</a>&nbsp;"
    } else {
        q = q + "&nbsp;&laquo;&nbsp;" + __("First") + "&nbsp;&nbsp;&nbsp;" + __("Prev") + "&nbsp;"
    }
    var b = m + 1;
    var g;
    var p = show_pages_amount / 2;
    a = b - p + 1;
    if (a < 1) {
        a = 1
    }
    d = a + p;
    if ((d - a + 1) < show_pages_amount) {
        d = d + show_pages_amount - (d - a + 1)
    }
    if (d > pages_amount) {
        d = pages_amount
    }
    if ((d - a + 1) < show_pages_amount) {
        a = a - (show_pages_amount - (d - a + 1))
    }
    if (a < 1) {
        a = 1
    }
    for (g = a; g <= d; g++) {
        if (g > pages_amount) {
            break
        }
        if (g == b) {
            q = q + "&nbsp;" + g + "&nbsp;"
        } else {
            q = q + '&nbsp;<a href="#" onclick="prepare_and_show_blogs_table_new(' + g + '); return false;">' + g + "</a>&nbsp;"
        }
    }
    if (i < blogs_amount) {
        var k = m + 2;
        q = q + '&nbsp;<a href="#" onclick="prepare_and_show_blogs_table_new(' + k + '); return false;">' + __("Next") + "&nbsp;</a>&nbsp;";
        q = q + '&nbsp;<a href="#" onclick="prepare_and_show_blogs_table_new(' + pages_amount + '); return false;">' + __("Last") + "&nbsp;&raquo;</a>&nbsp"
    } else {
        q = q + "&nbsp;" + __("Next") + "&nbsp;&nbsp;&nbsp;" + __("Last") + "&nbsp;&raquo;&nbsp"
    }
    document.getElementById("blogs_pages").innerHTML = __("More Blog Posts:") + "&nbsp;" + q + "<br />"
}
function prepare_and_show_news_table_original(m) {
    if (news_amount == 0) {
        return 
    }
    var s = "";
    var o = "";
    var h = 0;
    var r = 0;
    var c = 0;
    m--;
    var d = rows_in_the_news_table * tds_in_the_news_table;
    var a = m * d + 1;
    d = d + a - 1;
    if (m > 1 && null == document.getElementById("related_news_" + a) && typeof prepare_and_show_news_table_original.pagination_requested == "undefined") {
        prepare_and_show_news_table_original.pagination_requested = true;
        var e = document.getElementById("related_news");
        e.innerHTML = "";
        e.style.background = "transparent url('http://s.huffpost.com/images/loader.gif') no-repeat center top";
        e.style.height = "60px";
        var n = "/entries/pagination_load.php?module=linked_entries&entry_id=" + HPUtil.GetEntryID() + "&blog_id=2&load_from=" + loaded_news_amount;
        jQuery.ajax({
            type: "GET",
            url: n,
            dataType: "text",
            error: function() {},
            success: function(t) {
                if (t) {
                    var j = document.getElementById("related_news");
                    j.style.background = "transparent";
                    j.style.height = "auto";
                    document.getElementById("related_news_ajax_loaded").innerHTML = t;
                    prepare_and_show_news_table_original(m + 1)
                }
            }
        });
        return 
    }
    for (i = a; i <= d; i++) {
        var k = i > news_amount ? "": document.getElementById("related_news_" + i).innerHTML;
        k = k.split("height: 10px;").join("height: 0;");
        if ((k.indexOf('onclick="HPAds')>-1) && (r == 0)) {
            r = 1;
            o += '<div class="sponsored_preview"><div class="commercial_title commercial_title_' + HPConfig.current_vertical_name + '">SPONSORED FEATURE</div><div style="clear: both; line-height: 10px; height: 10px;"></div>'
        }
        if ((k.indexOf('onclick="HPAds')==-1) && (r == 1) && (c == 0)) {
            c = 1;
            o += "</div>"
        }
        o += k;
        if (i > news_amount) {
            break
        }
    }
    document.getElementById("related_news").innerHTML = o;
    o = "";
    h = 0;
    if (news_amount <= (rows_in_the_news_table * tds_in_the_news_table)) {
        return 
    }
    i--;
    var q = "";
    if (m > 0) {
        q = q + '&nbsp;<a href="#" onclick="prepare_and_show_news_table_original(' + 1 + '); return false;">&laquo;&nbsp;' + __("First") + "</a>&nbsp;";
        var f = m;
        q = q + '&nbsp;<a href="#" onclick="prepare_and_show_news_table_original(' + f + '); return false;">&nbsp;' + __("Prev") + "</a>&nbsp;"
    } else {
        q = q + "&nbsp;&laquo;&nbsp;" + __("First") + "&nbsp;&nbsp;&nbsp;" + __("Prev") + "&nbsp;"
    }
    var b = m + 1;
    var g;
    var p = show_pages_amount / 2;
    a = b - p + 1;
    if (a < 1) {
        a = 1
    }
    d = a + p;
    if ((d - a + 1) < show_pages_amount) {
        d = d + show_pages_amount - (d - a + 1)
    }
    if (d > news_pages_amount) {
        d = news_pages_amount
    }
    if ((d - a + 1) < show_pages_amount) {
        a = a - (show_pages_amount - (d - a + 1))
    }
    if (a < 1) {
        a = 1
    }
    for (g = a; g <= d; g++) {
        if (g > news_pages_amount) {
            break
        }
        if (g == b) {
            q = q + "&nbsp;" + g + "&nbsp;"
        } else {
            q = q + '&nbsp;<a href="#" onclick="prepare_and_show_news_table_original(' + g + '); return false;">' + g + "</a>&nbsp;"
        }
    }
    if (i < news_amount) {
        var l = m + 2;
        q = q + '&nbsp;<a href="#" onclick="prepare_and_show_news_table_original(' + l + '); return false;">' + __("Next") + "&nbsp;</a>&nbsp;";
        q = q + '&nbsp;<a href="#" onclick="prepare_and_show_news_table_original(' + news_pages_amount + '); return false;">' + __("Last") + "&nbsp;&raquo;</a>&nbsp"
    } else {
        q = q + "&nbsp;" + __("Next") + "&nbsp;&nbsp;&nbsp;" + __("Last") + "&nbsp;&raquo;&nbsp"
    }
    document.getElementById("news_pages").innerHTML = __("More News Posts:") + "&nbsp;" + q + "<br />"
};
var MostPopular = {
    items: [],
    curIdx: 0,
    timerTick: 0,
    timerSkip: false,
    timerID: null,
    firstElement: null,
    counter: 0,
    container: {},
    newTopImage: {},
    isMSIE: false,
    findFirstChild: function(a) {
        if (!a) {
            return 
        }
        for (k = 0; k < a.childNodes.length; k++) {
            if (a.childNodes[k].id) {
                return a.childNodes[k]
            }
        }
        return a.firstChild
    },
    insertAfter_update: function(c, a) {
        var b = a.parentNode;
        if (b.lastchild == a) {
            b.appendChild(c)
        } else {
            b.insertBefore(c, a.nextSibling)
        }
    },
    tick: function() {
        this.timerTick++;
        if (this.timerTick%15) {
            return 
        }
        this.timerTick = 0;
        if (this.timerSkip) {
            this.timerSkip = false;
            return 
        }
        this.up()
    },
    startclock: function() {
        var a = this;
        this.timerID = setInterval(function() {
            a.tick()
        }, 1000)
    },
    down: function() {
        this.newTopImage = document.getElementById("image_" + this.items[this.curIdx].id);
        if (this.newTopImage && this.newTopImage.src.length < 1 && this.newTopImage.alt) {
            this.newTopImage.src = this.newTopImage.alt
        }
        this.fetchImage(this.newTopImage);
        this.container.insertBefore(document.getElementById(this.items[this.curIdx].id), this.container.firstChild);
        this.newTopImage = document.getElementById("image_" + this.items[this.curIdx].id);
        if (this.newTopImage && this.newTopImage.src.length < 1 && this.newTopImage.alt) {
            this.newTopImage.src = this.newTopImage.alt
        }
        this.fetchImage(this.newTopImage);
        var a = this.isMSIE ? 20: 0;
        jQuery("#" + this.items[this.curIdx].id).css("height", a);
        jQuery("#" + this.items[this.curIdx].id).animate({
            height: "+=67px"
        }, 900);
        this.curIdx = ((this.curIdx + 1)%this.items.length);
        this.timerTick = 0;
        this.timerSkip = true
    },
    up: function() {
        this.curIdx -= 1;
        if (this.curIdx < 0) {
            this.curIdx = this.items.length - 1
        }
        if (this.items[this.curIdx - 4]) {
            this.newTopImage = document.getElementById("image_" + this.items[this.curIdx - 4].id);
            if (this.newTopImage && this.newTopImage.src.length < 1 && this.newTopImage.alt) {
                this.newTopImage.src = this.newTopImage.alt
            }
            this.fetchImage(this.newTopImage)
        }
        var a = this.findFirstChild(this.container);
        setTimeout("MostPopular.insertAfter_update(MostPopular.findFirstChild(MostPopular.container), MostPopular.container.lastChild)", 400);
        jQuery("#" + this.findFirstChild(this.container).id).animate({
            height: "-=67px"
        }, 600);
        setTimeout(function() {
            jQuery("#" + a.id).animate({
                paddingBottom: "-=15px"
            }, 200)
        }, 600);
        setTimeout(function() {
            jQuery("#" + a.id).animate({
                height: "+=67px",
                paddingBottom: "+=15px"
            }, 500)
        }, 850);
        this.timerTick = 0;
        this.timerSkip = true
    },
    fetchImage: function(b) {
        if (b && b.longdesc) {
            if (this.loadFrom == "local" && (url_match = /.*(images|dev.assets).huffingtonpost.com\/gen\/(\d+)\/(.*)/.exec(b.longdesc))) {
                var d = url_match[2];
                var e = url_match[3];
                var c = (url_match[1] == "images") ? "http://www.huffingtonpost.com": "";
                var a = c + "/imagecrop/" + this.chunk_split(d, 2, "/") + "/" + d + "/" + e;
                b.attr("src", a)
            } else {
                b.attr("src", b.longdesc)
            }
        }
    },
    chunk_split: function(d, a, b) {
        var c = 0;
        var e = new String();
        while (c + a < d.length) {
            e += d.substring(c, c + a) + b;
            c += a
        }
        if (c < d.length) {
            e += d.substring(c)
        }
        return e
    },
    init: function() {
        var a = [];
        this.container = document.getElementById("most_popular_list");
        tmp = jQuery("#most_popular_list li.entry");
        for (i = 4; i < tmp.length; i++) {
            a.push(tmp[i])
        }
        for (i = 0; i < 4; i++) {
            a.push(tmp[i])
        }
        for (i = a.length - 5; i >= 0; i--) {
            this.items.push(a[i])
        }
        for (i = a.length - 1; i >= a.length - 4; i--) {
            this.items.push(a[i])
        }
        this.startclock();
        isDOM = document.getElementById;
        isOpera = window.opera && isDOM;
        this.isMSIE = document.all && document.all.item&&!isOpera
    }
};
/*
Copyright (c) 2008 Ryan Grove <ryan@wonko.com>. All rights reserved.
Licensed under the BSD License:
http://www.opensource.org/licenses/bsd-license.html
Version: 1.0.4
*/
var LazyLoad = function() {
    var E = document, D = null, A = [], C;
    function B() {
        if (C) {
            return 
        }
        var G = navigator.userAgent, F;
        C = {
            gecko: 0,
            ie: 0,
            webkit: 0
        };
        F = G.match(/AppleWebKit\/(\S*)/);
        if (F && F[1]) {
            C.webkit = parseFloat(F[1])
        } else {
            F = G.match(/MSIE\s([^;]*)/);
            if (F && F[1]) {
                C.ie = parseFloat(F[1])
            } else {
                if ((/Gecko\/(\S*)/).test(G)) {
                    C.gecko = 1;
                    F = G.match(/rv:([^\s\)]*)/);
                    if (F && F[1]) {
                        C.gecko = parseFloat(F[1])
                    }
                }
            }
        }
    }
    return {
        load: function(K, L, J, I) {
            var H = E.getElementsByTagName("head")[0], G, F;
            if (K) {
                K = K.constructor === Array ? K : [K];
                for (G = 0; G < K.length; ++G) {
                    A.push({
                        url: K[G],
                        callback: G === K.length - 1 ? L: null,
                        obj: J,
                        scope: I
                    })
                }
            }
            if (D ||!(D = A.shift())) {
                return 
            }
            B();
            F = E.createElement("script");
            F.src = D.url;
            if (C.ie) {
                F.onreadystatechange = function() {
                    if (this.readyState === "loaded" || this.readyState === "complete") {
                        LazyLoad.requestComplete()
                    }
                }
            } else {
                if (C.gecko || C.webkit >= 420) {
                    F.onload = LazyLoad.requestComplete;
                    F.onerror = LazyLoad.requestComplete
                }
            }
            H.appendChild(F);
            if (!C.ie&&!C.gecko&&!(C.webkit >= 420)) {
                F = E.createElement("script");
                F.appendChild(E.createTextNode("LazyLoad.requestComplete();"));
                H.appendChild(F)
            }
        },
        loadOnce: function(N, O, L, P, G) {
            var H = [], I = E.getElementsByTagName("script"), M, J, K, F;
            N = N.constructor === Array ? N : [N];
            for (M = 0; M < N.length; ++M) {
                K = false;
                F = N[M];
                for (J = 0; J < I.length; ++J) {
                    if (F === I[J].src) {
                        K = true;
                        break
                    }
                }
                if (!K) {
                    H.push(F)
                }
            }
            if (H.length > 0) {
                LazyLoad.load(H, O, L, P)
            } else {
                if (G) {
                    if (L) {
                        if (P) {
                            O.call(L)
                        } else {
                            O.call(window, L)
                        }
                    } else {
                        O.call()
                    }
                }
            }
        },
        requestComplete: function() {
            if (D.callback) {
                if (D.obj) {
                    if (D.scope) {
                        D.callback.call(D.obj)
                    } else {
                        D.callback.call(window, D.obj)
                    }
                } else {
                    D.callback.call()
                }
            }
            D = null;
            if (A.length) {
                LazyLoad.load()
            }
        }
    }
}();
var HPUser = {
    is_logged_in: function() {
        if (HuffCookies.getCookie("huffpost_user") && HuffCookies.getCookie("huffpost_user_id")) {
            return true
        }
        return false
    },
    user_id: function() {
        return HuffCookies.getUserId()
    },
    username: HuffCookies.getUserName(),
    user_guid: HuffCookies.getUserGuid(),
    logout: function(d, b) {
        SNProject.track(HuffCookies.getUserId(), "user_log_out", 1);
        if ($("avatar_logged_in")) {
            $("avatar_logged_in").style.display = "none"
        }
        if ($("wendybird_user")) {
            $("wendybird_user").style.display = "none"
        }
        if ($("not_logged_user")) {
            $("not_logged_user").style.display = "block"
        }
        var a = document.getElementById("fbook_main_text_notloggedin");
        if (a) {
            a.style.display = "block"
        }
        a = document.getElementById("join_login_fbook_notloggedin");
        if (a) {
            a.style.display = "block"
        }
        a = document.getElementById("fConnect_img_container");
        if (a) {
            a.style.display = "block"
        }
        a = document.getElementById("fbook_main_text_loggedin");
        if (a) {
            a.style.display = "none"
        }
        a = document.getElementById("join_login_fbook_loggedin");
        if (a) {
            a.style.display = "none"
        }
        a = document.getElementById("fbook_main_text_name");
        if (a) {
            a.innerHTML = HuffCookies.getUserName().replace(/[\+_]/g, " ")
        }
        a = document.getElementById("fConnect_img_container");
        if (a) {
            a.style.display = "block"
        }
        var f = new Array("huffpost_user", "huffpost_photo_mode", "huffpost_snp_status", "huffpost_user_id", "huffpost_user_guid");
        for (var e = 0; e < f.length; e++) {
            HuffCookies.destroyCookie(f[e])
        }
        a = $("facebook_friends_unit_wrapper");
        if (a) {
            HuffPoUtil.hide("facebook_friends_unit_wrapper")
        }
        a = $("snn_entry_inside_module");
        if (a) {
            HuffPoUtil.hide(a)
        }
        SNPModule.showHpModules();
        return true
    },
    getUserCookies: function(a) {
        a = a || false;
        var b = {
            huffpost_user_guid: false,
            huffpost_user_id: false,
            huffpost_user: false,
            huffpost_levels: false,
            huffpost_prefs: false,
            huffpost_photo_mode: false
        };
        if (a) {
            return b
        }
        var c = HuffCookies;
        b.huffpost_user_guid = HuffCookies.getUserGuid() || false;
        b.huffpost_user_id = HuffCookies.getUserId() || false;
        b.huffpost_user = HuffCookies.getUserName() || false;
        b.huffpost_levels = c.getCookie("huffpost_levels") || false;
        b.huffpost_prefs = c.getCookie("huffpost_prefs") || false;
        b.huffpost_photo_mode = c.getCookie("huffpost_photo_mode") || false;
        return b
    }
};
var SocialFriends = {
    page_config: {
        current_page: {
            mutual: 0,
            favored: 0,
            fans: 0
        },
        total_pages: {
            mutual: 0,
            favored: 0,
            fans: 0
        }
    },
    user_ids: "",
    ignore_ids: "",
    happy_join: false,
    init: function() {
        this.page_config.total_pages.mutual = parseInt(jQuery("#sf_mutual_total_pages").html()) || 0;
        this.page_config.total_pages.favored = parseInt(jQuery("#sf_favored_total_pages").html()) || 0;
        this.page_config.total_pages.fans = parseInt(jQuery("#sf_fans_total_pages").html()) || 0;
        this.user_ids = jQuery("#sf_fan_uids").html() || "";
        return 
    },
    previous_page: function(b) {
        var c = this.page_config.current_page[b] - 1;
        var a = "#social_friends_" + b + "_page_";
        if (jQuery(a + c).length) {
            jQuery(a + this.page_config.current_page[b]).addClass("hidden");
            jQuery(a + c).removeClass("hidden");
            this.page_config.current_page[b] = c
        }
        this.setPage(b);
        return 
    },
    next_page: function(b) {
        var c = this.page_config.current_page[b] + 1;
        var a = "#social_friends_" + b + "_page_";
        if (jQuery(a + c).length) {
            jQuery(a + this.page_config.current_page[b]).addClass("hidden");
            jQuery(a + c).removeClass("hidden");
            this.page_config.current_page[b] = c
        }
        this.setPage(b);
        return 
    },
    setPage: function(b) {
        var a = this.page_config.current_page[b] + 1;
        if (a == 1) {
            jQuery(".sf_" + b + "_prev").addClass("sf_prev_disabled").removeClass("sf_prev_enabled");
            jQuery(".sf_" + b + "_next").addClass("sf_next_enabled").removeClass("sf_next_disabled")
        } else {
            if (a == this.page_config.total_pages[b]) {
                jQuery(".sf_" + b + "_next").addClass("sf_next_disabled").removeClass("sf_next_enabled");
                jQuery(".sf_" + b + "_prev").addClass("sf_prev_enabled").removeClass("sf_prev_disabled")
            } else {
                jQuery(".sf_" + b + "_prev").addClass("sf_prev_enabled").removeClass("sf_prev_disabled");
                jQuery(".sf_" + b + "_next").addClass("sf_next_enabled").removeClass("sf_next_disabled")
            }
        }
        jQuery("#sf_" + b + "_current_page").html(a);
        return 
    },
    selectFriends: function(a) {
        var c = jQuery("#follower_" + a);
        var b=!(jQuery(c).hasClass("friend_to_fan_checked"));
        jQuery(c).toggleClass("friend_to_fan_checked", b);
        if (!b) {
            this.user_ids = this.user_ids.replace(a + ",", "");
            this.ignore_ids += a + ","
        } else {
            this.user_ids += a + ",";
            this.ignore_ids = this.ignore_ids.replace(a + ",", "")
        }
        return 
    },
    saveFriends: function() {
        if (this.user_ids == "" && this.ignore_ids == "") {
            return 
        }
        var b = {
            user_ids: this.user_ids,
            ignore_ids: this.ignore_ids
        };
        var a = "/users/social_news_project/followers_to_friends.php";
        jQuery(".sf_save_friends_button").addClass("hidden");
        jQuery(".sf_save_friends_loader").removeClass("hidden");
        jQuery.ajax({
            url: a,
            data: b,
            type: "POST",
            success: function(e, d, c) {
                if (c.status != 200) {
                    return 
                }
                var g = JSON.parse(e);
                if (/success/.test(g.status)) {
                    SocialFriends.user_ids = SocialFriends.ignore_ids = "";
                    var f = g.friends_message + "" + g.ignored_message;
                    jQuery(".sf_fans_block").html(f)
                }
            },
            error: function() {
                HPError.e()
            }
        })
    },
    acceptFollower: function(c, b, a) {
        jQuery.ajax({
            url: "/users/friend_relations.php",
            data: {
                fan_id: c,
                action: "accept"
            },
            type: "POST",
            success: function(f, e, d) {
                if (/failure/.test(f)) {
                    HPError.e()
                }
                if (b) {
                    b(c)
                }
            },
            error: function() {
                if (a) {
                    b(a)
                }
            }
        })
    },
    declineFollower: function(c, b, a) {
        jQuery.ajax({
            url: "/users/friend_relations.php",
            data: {
                fan_id: c,
                action: "ignore"
            },
            type: "POST",
            success: function(f, e, d) {
                if (b) {
                    b(c)
                }
            },
            error: function() {
                if (a) {
                    b(a)
                }
            }
        })
    }
};
var ProviderFriends = {
    inited: false,
    init: function(a) {
        if (this.inited) {
            return 
        }
        if (jQuery("#su2_total_mutual_pages").length) {
            this.total_mutual_pages = parseInt(jQuery("#su2_total_mutual_pages").html())
        }
        if (jQuery("#su2_total_following_pages").length) {
            this.total_following_pages = parseInt(jQuery("#su2_total_following_pages").html())
        }
        if (jQuery("#su2_total_followers_pages").length) {
            this.total_followers_pages = parseInt(jQuery("#su2_total_followers_pages").html())
        }
        this.current_mutual_page = this.current_following_page = this.current_followers_page = 1;
        this.inited = true;
        return 
    },
    nextMutualPage: function() {
        this.init();
        var a = this.current_mutual_page + 1;
        if (a > this.total_mutual_pages) {
            return 
        }
        this.showPage("mutual", a);
        this.setMutualPagination(a);
        return 
    },
    prevMutualPage: function() {
        this.init();
        var a = this.current_mutual_page - 1;
        if (a < 1) {
            return 
        }
        this.showPage("mutual", a);
        this.setMutualPagination(a);
        return 
    },
    setMutualPagination: function(b) {
        var a = jQuery("#su_modal2_prev_link_mutual");
        var c = jQuery("#su_modal2_next_link_mutual");
        if (b <= 1) {
            a.addClass("su_modal2_prev_disabled").removeClass("su_modal2_prev_enabled");
            c.addClass("su_modal2_next_enabled").removeClass("su_modal2_next_disabled");
            this.current_mutual_page = 1
        }
        if (b > 1 && b < this.total_mutual_pages) {
            a.addClass("su_modal2_prev_enabled").removeClass("su_modal2_prev_disabled");
            c.addClass("su_modal2_next_enabled").removeClass("su_modal2_next_disabled");
            this.current_mutual_page = b
        }
        if (b >= this.total_mutual_pages) {
            a.addClass("su_modal2_prev_enabled").removeClass("su_modal2_prev_disabled");
            c.addClass("su_modal2_next_disabled").removeClass("su_modal2_next_enabled");
            this.current_mutual_page = this.total_mutual_pages
        }
        return 
    },
    nextFollowingPage: function() {
        this.init();
        var a = this.current_following_page + 1;
        if (a > this.total_following_pages) {
            return 
        }
        this.showPage("following", a);
        this.setFollowingPagination(a);
        return 
    },
    prevFollowingPage: function() {
        this.init();
        var a = this.current_following_page - 1;
        if (a < 1) {
            return 
        }
        this.showPage("following", a);
        this.setFollowingPagination(a);
        return 
    },
    setFollowingPagination: function(b) {
        var a = jQuery("#su_modal2_prev_link_following");
        var c = jQuery("#su_modal2_next_link_following");
        if (b <= 1) {
            a.addClass("su_modal2_prev_disabled").removeClass("su_modal2_prev_enabled");
            c.addClass("su_modal2_next_enabled").removeClass("su_modal2_next_disabled");
            this.current_following_page = 1
        }
        if (b > 1 && b < this.total_following_pages) {
            a.addClass("su_modal2_prev_enabled").removeClass("su_modal2_prev_disabled");
            c.addClass("su_modal2_next_enabled").removeClass("su_modal2_next_disabled");
            this.current_following_page = b
        }
        if (b >= this.total_following_pages) {
            a.addClass("su_modal2_prev_enabled").removeClass("su_modal2_prev_disabled");
            c.addClass("su_modal2_next_disabled").removeClass("su_modal2_next_enabled");
            this.current_following_page = this.total_following_pages
        }
        return 
    },
    nextFollowersPage: function() {
        this.init();
        var a = this.current_followers_page + 1;
        if (a > this.total_followers_pages) {
            return 
        }
        this.showPage("followers", a);
        this.setFollowersPagination(a);
        return 
    },
    prevFollowersPage: function() {
        this.init();
        var a = this.current_followers_page - 1;
        if (a < 1) {
            return 
        }
        this.showPage("followers", a);
        this.setFollowersPagination(a);
        return 
    },
    setFollowersPagination: function(b) {
        var a = jQuery("#su_modal2_prev_link_followers");
        var c = jQuery("#su_modal2_next_link_followers");
        if (b <= 1) {
            a.addClass("su_modal2_prev_disabled").removeClass("su_modal2_prev_enabled");
            c.addClass("su_modal2_next_enabled").removeClass("su_modal2_next_disabled");
            this.current_followers_page = 1
        }
        if (b > 1 && b < this.total_followers_pages) {
            a.addClass("su_modal2_prev_enabled").removeClass("su_modal2_prev_disabled");
            c.addClass("su_modal2_next_enabled").removeClass("su_modal2_next_disabled");
            this.current_followers_page = b
        }
        if (b >= this.total_followers_pages) {
            a.addClass("su_modal2_prev_enabled").removeClass("su_modal2_prev_disabled");
            c.addClass("su_modal2_next_disabled").removeClass("su_modal2_next_enabled");
            this.current_followers_page = this.total_followers_pages
        }
        return 
    },
    showPage: function(b, c) {
        if (b == "mutual") {
            total = this.total_mutual_pages
        }
        if (b == "following") {
            total = this.total_following_pages
        }
        if (b == "followers") {
            total = this.total_followers_pages
        }
        for (var a = 1; a <= total; a++) {
            if (a == c) {
                jQuery("#su2_" + b + "_page_" + a).css("display", "block")
            } else {
                jQuery("#su2_" + b + "_page_" + a).css("display", "none")
            }
        }
    },
    setFollowersToFans: function(b, d) {
        var a = d.firstChild;
        var f = jQuery("#followers_to_fans_box");
        var e = b + "|";
        if (f.length) {
            var c = f.attr("value")
        }
        if (jQuery(a).hasClass("friend_to_fan_checked_small")) {
            jQuery(a).addClass("friend_to_fan_checked_small_disabled").removeClass("friend_to_fan_checked_small");
            f.attr("value", c.replace(e, ""))
        } else {
            jQuery(a).addClass("friend_to_fan_checked_small").removeClass("friend_to_fan_checked_small_disabled");
            f.attr("value", f.attr("value") + e)
        }
        this.updateTwitterFriendCount();
        return 
    },
    checkAllTwitterFollower: function(e) {
        var d = jQuery("#followers_to_fans_box");
        d.attr("value", e);
        this.updateTwitterFriendCount();
        var c = e.split("|");
        for (var a = 0; a < c.length; a++) {
            var b = jQuery("#twitter_check_" + c[a]);
            if (b.length) {
                if (b.hasClass("friend_to_fan_checked_small_disabled")) {
                    b.addClass("friend_to_fan_checked_small").removeClass("friend_to_fan_checked_small_disabled")
                }
            }
        }
        return 
    },
    unCheckAllTwitterFollower: function(e) {
        var d = jQuery("#followers_to_fans_box");
        d.attr("value", "");
        this.updateTwitterFriendCount();
        var c = e.split("|");
        for (var a = 0; a < c.length; a++) {
            var b = jQuery("#twitter_check_" + c[a]);
            if (b.length) {
                if (b.hasClass("friend_to_fan_checked_small")) {
                    b.addClass("friend_to_fan_checked_small_disabled").removeClass("friend_to_fan_checked_small")
                }
            }
        }
        return 
    },
    updateTwitterFriendCount: function() {
        var d = jQuery("#followers_to_fans_box");
        var c = d.attr("value");
        var a = c.split("|");
        var b = a.length - 1;
        jQuery("twitter_selected_count").html(b);
        return 
    },
    setCustomFans: function(b, d) {
        var a = d.firstChild;
        var f = jQuery("#custom_friends_box");
        var e = b + "|";
        if (f.length) {
            var c = f.attr("value")
        }
        if (jQuery(a).hasClass("friend_to_fan_checked_small")) {
            jQuery(a).addClass("friend_to_fan_checked_small_disabled").removeClass("friend_to_fan_checked_small");
            f.attr("value", c.replace(e, ""))
        } else {
            jQuery(a).addClass("friend_to_fan_checked_small").removeClass("friend_to_fan_checked_small_disabled");
            f.attr("value", f.attr("value") + e)
        }
        this.updateGooleFriendCount();
        return 
    },
    checkAllCustom: function(e) {
        var d = jQuery("#custom_friends_box");
        d.attr("value", e);
        this.updateGooleFriendCount();
        var c = e.split("|");
        for (var a = 0; a < c.length; a++) {
            var b = jQuery("#google_check_" + c[a]);
            if (b.length) {
                if (b.hasClass("friend_to_fan_checked_small_disabled")) {
                    b.addClass("friend_to_fan_checked_small").removeClass("friend_to_fan_checked_small_disabled")
                }
            }
        }
        return 
    },
    unCheckAllCustom: function(e) {
        var d = jQuery("#custom_friends_box");
        d.attr("value", "");
        this.updateGooleFriendCount();
        var c = e.split("|");
        for (var a = 0; a < c.length; a++) {
            var b = jQuery("#google_check_" + c[a]);
            if (b.length) {
                if (b.hasClass("friend_to_fan_checked_small")) {
                    b.addClass("friend_to_fan_checked_small_disabled").removeClass("friend_to_fan_checked_small")
                }
            }
        }
        return 
    },
    updateGooleFriendCount: function() {
        var d = jQuery("#custom_friends_box");
        var c = d.attr("value");
        var a = c.split("|");
        var b = a.length - 1;
        jQuery("google_selected_count").html(b);
        return 
    }
};
window._ || (window._ = new Function("return arguments[0]"));
window.ngettext || (window.ngettext = new Function("return arguments[0]"));
window.sprintf || (window.sprintf = new Function("return arguments[0]"));
var Recommendations = {
    page_entries_loading: false,
    entries_loaded: false,
    max_page: 1,
    loaded: false,
    fb_session: null,
    is_inited: false,
    load: function(b) {
        if (this.loaded) {
            return 
        }
        if ("undefined" !== typeof b.session) {
            console.log("Someone called Recommendations.load() with session param instead of fb_session which is the new style")
        }
        this.fb_session = b.fb_session;
        if ("unknown" != b.status && ( - 1 !== location.href.indexOf("SNN=fb_recommendations")||-1 !== location.href.indexOf("snn=2") || "fb_recommendations" == HuffCookies.getCookie("snn_version") || (b.fb_session && b.fb_session.userID&&!isNaN(b.fb_session.userID))) && b.fb_session && b.fb_session.userID&&!isNaN(b.fb_session.userID)) {
            HuffCookies.setCookie("snn_version", "fb_recommendations", 2000);
            E.onAvailable("facebook_friends_unit_wrapper_fb", function() {
                Dom.get("facebook_friends_unit_wrapper_fb").style.display = "block";
                huff.emit("content/update", "#facebook_friends_unit_wrapper_fb")
            });
            E.onAvailable("fb_recommendations_cont_loaded", function() {
                E.on("fb_recom_help", "mouseover", function() {
                    if (!this.getAttribute("floating_id")) {
                        var c = '<div class="relative">' + _('We\'re using Facebook to personalize your experience. The stories below reflect what you like on Facebook and around the web.Send us feedback at <a href="mailto:beta@huffingtonpost.com">beta@huffingtonpost.com</a>') + '<div class="bottom_bg relative"></div></div>';
                        var d = [ - 156, - 118];
                        FloatingPrompt.embed(this, c, undefined, "top", {
                            fp_intersects: 1,
                            timeout_remove: 2000,
                            ignore_arrow: true,
                            width: 251,
                            add_xy: d,
                            class_name: "fp_fb_recom"
                        })
                    }
                });
                Recommendations.init();
                Recommendations.loaded = true
            })
        } else {
            var a = this;
            if ($("avatar_logged_in_snn") && typeof FB !== "undefined" && this.fb_session.userID && HuffCookies.getUserName()) {
                $("avatar_logged_in_snn").innerHTML = sprintf('<a href="/social/%s/"><img src="http://graph.facebook.com/%s/picture" width="20"></a>', HuffCookies.getUserName(), this.fb_session.userID);
                FB.api({
                    method: "fql.query",
                    query: "SELECT name FROM user WHERE uid=" + this.fb_session.userID
                }, function(c) {
                    if (c[0] && c[0].name) {
                        $("wendybird_user_name_snn").innerHTML = sprintf(_('Welcome, <a href="/social/%s/">%s</a>'), HuffCookies.getUserName(), c[0].name)
                    }
                })
            }
        }
    },
    init: function() {
        if (this.is_inited) {
            return 
        }
        this.is_inited = true;
        this.snp_block = new YAHOO.SNP.RecommendBlock({
            cur_obj: this,
            id: "snp_recom_block",
            update_max_page: true
        })
    },
    buildHtml: function(f) {
        var e = "";
        var a = this.snp_block.paginator.currentPage;
        e += '<div id="hp_snp_recom_block_page_' + a + '"  data-beacon=\'{"p":{"lnid":"page' + a + "\"}}'>";
        var b = 0;
        var c = HPUtil.GetEntryID();
        for (var d = 0; d < f.entries.length; ++d) {
            if (c === f.entries[d].entry_id || b === 3) {
                continue
            }
            var g = f.entries[d];
            if ("dev" == HPConfig.inst_type) {
                g.entry_link = g.entry_link.replace(/beta.*?\//, "www.huffingtonpost.com/")
            }
            e += '<div class="float_left hp_snp_recom_entry" id="tag_fb_recom_' + g.specified_tag + '"                 onmouseover="HPFB.likeButton_v2(this, \'' + g.fb_button + '\', false);" style="' + (b != 0 ? "padding-top:7px;" : "") + "                " + (2 > b ? "padding-bottom:7px;border-bottom:1px dotted #afafaf;" : "") + '"  data-beacon=\'{"p":{"mpid":' + b + ', "plid":' + g.entry_id + "}}'>";
            e += '<div class="snp_most_popular_entry_image">';
            if (g.image) {
                e += '<a href="' + g.entry_link + '" class="track_page_article white_bg">                     <img style="margin-right:5px;" class="img_border" src="' + g.image + '"/></a>'
            }
            e += "</div>";
            e += '<div class="snp_most_popular_entry_desc">';
            e += '<a class="track_page_article white_bg black bold arial_13" href="' + g.entry_link + '">                 ' + g.entry_title + '</a> <br class="clear_first" />';
            e += g.fb_like_html;
            if ("undefined" !== g._tag_data) {
                e += '<div class="floatright relative hp_fb_tag_cont">';
                e += '<div class="hp_fb_recom_tag_name float_left relative overflow_hidden margin_right_5 overflow_hidden                     align_right hp_fb_recom_fblike_' + g.fb_button + '">';
                e += '<a class="capitalize track_page_tag white_bg bold underline line_height_22 hp_fb_recom_tag_link" href="/tag/' + g._tag_data.tag_name + '">' + g._tag_data.tag_name + "</a>";
                e += "</div>";
                e += '<div class="float_left margin_top_5 relative" style="z-index:102;">';
                e += '<a title="' + _("Remove this entry from my likes") + '" class="track_action_remove" href="#fb_recommendations" onclick="                     Recommendations.removeTag(' + g.specified_tag + ", " + g.entry_id + ')"><img                     src="/images/facebook/recommendations/delete-icon.png" width="13" height="13" /></a>';
                e += "</div>";
                e += "</div>"
            }
            e += "</div>";
            e += "</div>";
            ++b
        }
        e += "</div>";
        return e
    },
    loadSNPBlock: function() {
        var b = this;
        if ($("avatar_logged_in_snn") && typeof FB !== "undefined" && this.fb_session.userID && HuffCookies.getUserName()) {
            $("avatar_logged_in_snn").innerHTML = sprintf('<a href="/social/%s/"><img src="http://graph.facebook.com/%s/picture" width="20"></a>', HuffCookies.getUserName(), this.fb_session.userID);
            FB.api({
                method: "fql.query",
                query: "SELECT name FROM user WHERE uid=" + this.fb_session.userID
            }, function(h) {
                if (h[0] && h[0].name) {
                    $("wendybird_user_name_snn").innerHTML = sprintf(_('Welcome, <a href="/social/%s/">%s</a>'), HuffCookies.getUserName(), h[0].name)
                }
            })
        }
        var g = "&uid=" + this.fb_session.userID + "&access_token=" + encodeURIComponent(this.fb_session.accessToken);
        var a = 0, d = false, f = HPUtil.getUrlVar("username"), b = this;
        var e = HuffCookies.getUserName() ? "&logged_username=" + HuffCookies.getUserName(): "";
        var c = function() {
            if (d) {
                return 
            }
            d = true;
            C.asyncRequest("GET", "/ajax/users/recommendations/load_entries.php?" + g + "&try=" + a, {
                success: function(j) {
                    d = false;
                    var i = jQuery.parseJSON(j.responseText);
                    if ("undefined" !== typeof i.loading) {
                        ++a;
                        if (5 <= a) {
                            clearInterval(b.pollingID);
                            Dom.get("facebook_friends_unit_wrapper_fb").style.display = "none";
                            Dom.get("hp_snp_recom_block_all_pages").innerHTML = "<h4>" + jQuery.I18N.gettext("There are no stories for you") + "</h4>";
                            Dom.get("hp_snp_recom_block_all_pages").style.textAlign = "center";
                            Dom.get("hp_snp_recom_block_main").style.background = "transparent"
                        }
                        return 
                    }
                    if ("undefined" !== typeof i.cmd && "hide_block" === i.cmd && $("facebook_friends_unit_wrapper_fb")) {
                        $("facebook_friends_unit_wrapper_fb").style.display = "none"
                    }
                    clearInterval(b.pollingID);
                    var h = "/ajax/users/recommendations/load_entries.php?" + g;
                    b.max_page = i.c_pages;
                    if (b.max_page > 1) {
                        b.snp_block.init_pagination(1, b.max_page, h, {
                            is_loaded_callback: function(k) {
                                return "undefined" === typeof k.loading
                            },
                            main_block_to_hide: $("facebook_friends_unit_wrapper_fb"),
                            render_content_after_requested: function(k) {
                                return b.buildHtml(k)
                            },
                            recognize_cmd: function(k) {
                                return "undefined" !== typeof k.cmd ? k.cmd : ""
                            },
                            cPages: function(k) {
                                return k.c_pages
                            },
                            dataType: "json"
                        });
                        Dom.removeClass("snp_recom_block_pagination", "hidden")
                    }
                    Dom.get("hp_snp_recom_block_all_pages").innerHTML = b.buildHtml(i);
                    HPUtil.ImageLoader.foldCheck("hp_snp_recom_block_page_1");
                    Dom.get("hp_snp_recom_block_main").style.background = "transparent";
                    Dom.get("hp_snp_recom_block_main").style.height = "auto";
                    HPUtil.ImageLoader.foldCheck("hp_snp_recom_block_page_1");
                    HPTrack.Module.Impressions.queue("recommend")
                },
                failure: function() {
                    d = false;
                    clearInterval(b.pollingID);
                    Dom.get("facebook_friends_unit_wrapper_fb").style.display = "none";
                    Dom.get("hp_snp_recom_block_all_pages").innerHTML = "<h4>" + jQuery.I18N.gettext("There are no stories for you") + "</h4>";
                    Dom.get("hp_snp_recom_block_all_pages").style.textAlign = "center";
                    Dom.get("hp_snp_recom_block_main").style.background = "transparent";
                    HPError.e()
                },
                timeout: 10000
            })
        };
        c();
        b.pollingID = window.setInterval(c, 5000)
    },
    removeTag: function(a, b) {
        var c = this;
        var d = "entry_id=" + b;
        var e = "&uid=" + c.fb_session.userID + "&access_token=" + encodeURIComponent(c.fb_session.accessToken);
        C.asyncRequest("POST", "/users/recommendations/remove_tag.php?" + d + e + "&tag=" + a, {
            success: function(i) {
                var j = c.snp_block.paginator;
                var g = j.currentPage;
                var f = Dom.get("tag_fb_recom_" + a).parentNode;
                var h = false;
                while (1) {
                    if (Dom.get("hp_snp_recom_block_page_" + (++g).toString())) {
                        Dom.get(Dom.get("hp_snp_recom_block_page_" + g)).parentNode.removeChild(Dom.get("hp_snp_recom_block_page_" + g));
                        continue
                    }
                    break
                }
                Dom.get(Dom.get("hp_snp_recom_block_page_" + j.currentPage)).parentNode.removeChild(Dom.get("hp_snp_recom_block_page_" + j.currentPage));
                j.Goto(j.currentPage)
            },
            failure: function() {
                HPError.e()
            }
        })
    }
};
function AddSiteMode() {
    window.SiteMode = {
        buttons_to_names: {},
        UPDATE_ICON_FRAMES: 3,
        notification_showing: false,
        simulateUpdate: function() {
            Refresh.loadingStartUI();
            Refresh.update("news_col_1.html?45435&vertical=" + Refresh.vertical + (HPAds.doubleclick ? "&doubleclick=" + escape(HPAds.doubleclick) : ""))
        },
        verticalIsPinned: function() {
            if (typeof HPConfig == "undefined") {
                return false
            }
            if (!window.sessionStorage.pinMode) {
                window.sessionStorage.pinMode = HPConfig.current_vertical_name
            }
            if (window.sessionStorage.pinMode == HPConfig.current_vertical_name) {
                return true
            }
            return false
        },
        init: function() {
            try {
                if (window.external.msIsSiteMode()) {
                    var h = window.external.msIsSiteModeFirstRun(false);
                    if (h) {
                        var c = window.localStorage;
                        var k = c.siteModePromoType || "SiteModeC";
                        c.removeItem("siteModePromoType");
                        var a = "NA";
                        if (h == 1) {
                            a = "Drag-drop"
                        } else {
                            if (h == 2) {
                                a = "Shortcut Icon"
                            }
                        }
                        HPTrack.trackEvent("Sitemode", a, "Via promo " + k)
                    }
                    E.addListener(window, "focus", this.replayUpdate, this);
                    E.addListener(window, "focus", this.hideNotification, this);
                    this.updateButtons();
                    if (!this.verticalIsPinned()) {
                        return 
                    }
                    var n = [];
                    var g = Dom.getElementsByClassName("big_news_item", "a", "topnav_big_news_module");
                    for (var b = 0, f = g.length; b < f; b++) {
                        var m = g[b].innerHTML || "";
                        var d = g[b].className.match(/bn_v_([\w\-_]+)/);
                        var l = {
                            text: m,
                            href: g[b].href
                        };
                        if (d && d.length > 1) {
                            l.vert = d[1]
                        }
                        if (m) {
                            n.push(l)
                        }
                    }
                    this.updateJumpList(n)
                }
            } catch (j) {}
        },
        modalStyle: function() {},
        buttonHandlers: {
            twitter: function() {
                window.open("http://twitter.com/share?url=" + encodeURIComponent(window.location), "twitter_share", "width=500,height=350")
            },
            facebook: function() {
                window.open("http://www.facebook.com/sharer.php?u=" + encodeURIComponent(window.location), "facebook_share", "width=500,height=350")
            }
        },
        updateButtons: function() {
            var a = window.external.msSiteModeAddThumbBarButton("/images/sitemode/facebook.ico", "Facebook");
            var c = window.external.msSiteModeAddThumbBarButton("/images/sitemode/twitter.ico", "Twitter");
            this.buttons_to_names[a] = "facebook";
            this.buttons_to_names[c] = "twitter";
            function b(f) {
                var h = SiteMode;
                var g = f.buttonID;
                var d = h.buttons_to_names[g] || false;
                if (!d) {
                    return 
                }
                h.buttonHandlers[d]()
            }
            if (document.addEventListener) {
                document.addEventListener("msthumbnailclick", b, false)
            } else {
                if (document.attachEvent) {
                    document.attachEvent("onmsthumbnailclick", b)
                }
            }
            window.external.msSiteModeShowThumbBar()
        },
        updateJumpList: function(d) {
            if (d && d.length) {
                var a = new Date();
                window.external.msSiteModeCreateJumplist("Big News - " + (parseInt(a.getMonth()) + 1) + "/" + a.getDate() + "/" + a.getFullYear());
                for (var c = d.length - 1; c >= 0; c--) {
                    var b = "/images/sitemode/verts/" + (d[c].vert || "homepage") + ".ico";
                    window.external.msSiteModeAddJumpListItem(d[c].text, d[c].href, b)
                }
                window.external.msSiteModeShowJumplist()
            } else {
                window.external.msSiteModeClearJumplist()
            }
        },
        notifyUser: function(b) {
            try {
                if (window.external.msIsSiteMode()) {
                    var c = "New headlines!";
                    if (b.is_splash) {
                        c = "New top story!"
                    }
                    SiteMode.animator = window.setInterval(function() {
                        SiteMode.cycleNotification(c)
                    }, 500);
                    this.notification_showing = true
                }
            } catch (a) {}
        },
        cycleNotification: function(a) {
            var b = SiteMode;
            if (!b.icon_index || b.icon_index == b.UPDATE_ICON_FRAMES) {
                b.icon_index = 1
            } else {
                b.icon_index++
            }
            window.external.msSiteModeSetIconOverlay("/images/sitemode/update" + b.icon_index + ".ico", a)
        },
        hideNotification: function() {
            try {
                if (window.external.msIsSiteMode()) {
                    if (SiteMode.animator) {
                        window.clearInterval(SiteMode.animator)
                    }
                    window.external.msSiteModeClearIconOverlay()
                }
            } catch (a) {}
        },
        replayUpdate: function() {
            if (!(window.Refresh && Refresh.last_block)) {
                return 
            }
            Refresh.animateBlock(Refresh.last_block);
            HPUtil.ScrollTo(Refresh.last_block);
            Refresh.last_block = false
        }
    }
}
try {
    if (window.external.msIsSiteMode()) {
        AddSiteMode()
    }
} catch (e) {};
function SplashSlideshow() {
    this.slideshow_id = 0;
    this.navFirstEl = this.navLastEl = null;
    this.animating = false;
    this.aSlides = {};
    this.all_images = [];
    this.images = [];
    this.domain = "";
    this.carousel = false;
    this.play = true;
    this.current_image = 0;
    this.current_number = 0;
    this.loopcount = 3;
    this.currentloop = 0;
    this.delay = 8
}
SplashSlideshow.prototype.SlideShow = function(a) {
    this.carousel = true;
    this.current_number = this.aSlides[a].number;
    this.SwitchImage(0)
}, SplashSlideshow.prototype.SwitchImage = function(d) {
    var c = this;
    if (d == this.c_slides) {
        d = 0;
        this.currentloop++
    }
    this.current_number = d;
    if (this.current_number > 0) {
        this.MoveLeftStep()
    }
    if (this.current_number == 0 & this.currentloop > 0) {
        this.MoveLeftStep()
    }
    var f = d - 1;
    if (d == 0) {
        var f = this.c_slides - 1
    }
    var e = "slide_image_" + this.images[f];
    var a = "slide_image_" + this.images[d];
    var b = this.images[this.current_number];
    this.image_id = b;
    if ("image" == this.aSlides[b].content_type) {
        Dom.get("slide_image_" + this.slideshow_id).src = this.domain + "gadgets/slideshows/" + this.slideshow_id + "/slide_" + this.slideshow_id + "_" + b + "_splash.jpg";
        Dom.get("image_cont_" + this.slideshow_id).style.display = "block";
        Dom.get("video_cont_" + this.slideshow_id).style.display = "none";
        c.clearThumbnails();
        Dom.setStyle(a, "opacity", 0.5);
        if (this.aSlides[b].title) {
            Dom.get("slide_image_" + this.slideshow_id).title = this.aSlides[b].title
        } else {
            Dom.get("slide_image_" + this.slideshow_id).title = ""
        }
    } else {
        Dom.get("image_cont_" + this.slideshow_id).style.display = "none";
        Dom.get("video_cont_" + this.slideshow_id).style.display = "block";
        if ( - 1 !== this.aSlides[b].video_code.indexOf("<script")) {
            HPUtil.EvalScript(this.aSlides[b].video_code)
        }
        if (HPBrowser.isIE6()) {
            Dom.get("video_cont_" + this.slideshow_id).innerHTML = HPUtil.getCorrectVideoContentForIE6(this.aSlides[b].video_code)
        } else {
            Dom.get("video_cont_" + this.slideshow_id).innerHTML = this.aSlides[b].video_code
        }
    }
    d++;
    if (this.currentloop < this.loopcount) {
        this.timer = setTimeout(function() {
            splash_slideshow.SwitchImage(++c.current_number)
        }, (this.delay * 1000))
    }
}, SplashSlideshow.prototype.gotoTheNextImage = function() {
    if (this.animating) {
        return 
    }
    this.animating = true;
    var a = this;
    (function(b) {
        if (a.current_number < a.c_slides - 1) {
            a.current_number++
        } else {
            a.current_number = 0
        }
        a.clearThumbnails();
        b("#slide_image_" + a.images[a.current_number]).animate({
            opacity: 0.5
        }, 150);
        b("#slide_image_" + a.slideshow_id).attr("src", a.domain + "gadgets/slideshows/" + a.slideshow_id + "/slide_" + a.slideshow_id + "_" + a.images[a.current_number] + "_splash.jpg").load(function() {
            a.animating = false
        })
    })(jQuery)
}, SplashSlideshow.prototype.gotoThePrevImage = function() {
    if (this.animating) {
        return 
    }
    this.animating = true;
    var a = this;
    (function(b) {
        if (a.current_number > 0) {
            a.current_number--
        } else {
            a.current_number = a.c_slides - 1
        }
        a.clearThumbnails();
        b("#slide_image_" + a.images[a.current_number]).animate({
            opacity: 0.5
        }, 150);
        b("#slide_image_" + a.slideshow_id).attr("src", a.domain + "gadgets/slideshows/" + a.slideshow_id + "/slide_" + a.slideshow_id + "_" + a.images[a.current_number] + "_splash.jpg").load(function() {
            a.animating = false
        })
    })(jQuery)
}, SplashSlideshow.prototype.Initialize = function(c) {
    if (window.location.search.match("tigertail")) {
        huff.css("hp_modules/module.splash_slideshow", function() {
            return true
        })
    }
    var b = this;
    huff.jquery(function(d) {
        d("#image_cont_" + b.slideshow_id).click(function() {})
    });
    splash_slideshow.changeOpac(this.current_image, 0.7);
    for (var a in this.aSlides) {
        if (!this.aSlides.hasOwnProperty(a)) {
            continue
        }
        this.images[this.aSlides[a].number] = a
    }
    this.slideshow_id = c;
    this.anim_container = Dom.get("slideshow_splash_navigation_slides_container");
    HuffPoUtil.ImageLoader.foldCheck(this.anim_container, true);
    E.on(Dom.getElementsByClassName("slide_image", "img", this.anim_container), "mouseover", function() {
        var f = (new RegExp(/(\d+)_small/)).exec(this.src)[1];
        if (!this.getAttribute("floating_id") && b.aSlides[f].caption) {
            var e = HPUtil.Strip_Tags(b.aSlides[f].caption);
            var d = '<div style="position:absolute;width:200px;"><span style="' + (e.length ? "padding-right:2px;" : "") + '">' + e.substr(0, 28) + '</span></div><div><div class="floatleft"></div><div class="clear"></div><div class="floatright" style="font-weight:bold; height:20px"></div></div>';
            FloatingPrompt.embed(this, d, undefined, "top", {
                width: 200,
                add_xy: [0, - 40],
                class_name: "fp_splash_slideshow"
            })
        }
    });
    E.on(Dom.getElementsByClassName("slide_image", "img", this.anim_container), "click", function() {
        var d = (new RegExp(/(\d+)_small/)).exec(this.src);
        b.current_number = b.aSlides[d[1]].number;
        if (b.carousel == true) {
            clearTimeout(b.timer);
            b.image_id = d[1];
            Dom.get("slide_loading_spinner_" + b.slideshow_id).style.display = "none";
            b.timer = setTimeout(function() {
                b.SwitchImage(b.aSlides[d[1]].number + 1)
            }, 15000)
        }
        b.clearThumbnails();
        splash_slideshow.changeOpac(this.id, 0.7);
        if ("image" == b.aSlides[d[1]].content_type) {
            if (b.aSlides[d[1]].title) {
                Dom.get("slide_image_" + c).title = b.aSlides[d[1]].title
            } else {
                Dom.get("slide_image_" + c).title = ""
            }
            Dom.get("slide_image_" + c).src = this.src.replace("_small", "_splash");
            Dom.get("image_cont_" + c).style.display = "block";
            Dom.get("video_cont_" + c).style.display = "none";
            b.clearThumbnails();
            splash_slideshow.changeOpac(this.id, 0.7)
        } else {
            Dom.get("image_cont_" + c).style.display = "none";
            Dom.get("video_cont_" + c).style.display = "block";
            if ( - 1 !== b.aSlides[d[1]].video_code.indexOf("<script")) {
                HPUtil.EvalScript(b.aSlides[d[1]].video_code)
            }
            if (HPBrowser.isIE6()) {
                Dom.get("video_cont_" + c).innerHTML = HPUtil.getCorrectVideoContentForIE6(b.aSlides[d[1]].video_code)
            } else {
                Dom.get("video_cont_" + c).innerHTML = b.aSlides[d[1]].video_code
            }
        }
        b.current_image = this.id
    });
    if (this.c_slides > 9) {
        E.on(Dom.get("slideshow_splash_navigation_left_button"), "click", function(d) {
            E.preventDefault(d);
            this.MoveLeft();
            b.gotoThePrevImage();
            HuffPoUtil.ImageLoader.foldCheck(this.anim_container, true, [ - 100 * 8, 0])
        }, {}, this);
        E.on(Dom.get("slideshow_splash_navigation_right_button"), "click", function(d) {
            E.preventDefault(d);
            this.MoveRight();
            b.gotoTheNextImage();
            HuffPoUtil.ImageLoader.foldCheck(this.anim_container, true, [ - 100 * 8, 0])
        }, {}, this)
    }
}, SplashSlideshow.prototype.FindFirstAndLast = function() {
    this.navFirstEl = this.anim_container.firstChild;
    this.navLastEl = this.anim_container.lastChild
}, SplashSlideshow.prototype.MoveLeft = function() {
    if (this.animating) {
        return 
    }
    var b = this;
    this.animating = true;
    this.all_images = this.anim_container.getElementsByTagName("img");
    for (var a = 0; a < 1; ++a) {
        this.FindFirstAndLast();
        this.anim_container.removeChild(this.navLastEl);
        this.anim_container.insertBefore(this.navLastEl, this.navFirstEl)
    }
    this.FindFirstAndLast();
    HuffPoUtil.ImageLoader.foldCheck(this.anim_container, true, [ - 100 * 8, 0]);
    this.animating = false
}, SplashSlideshow.prototype.MoveLeftStep = function() {
    this.FindFirstAndLast();
    this.anim_container.removeChild(this.navFirstEl);
    this.anim_container.insertBefore(this.navFirstEl, this.navLastEl.nextSibling);
    HuffPoUtil.ImageLoader.foldCheck(this.anim_container, true, [ - 100 * 8, 0])
}, SplashSlideshow.prototype.MoveRight = function() {
    if (this.animating) {
        return 
    }
    var b = this;
    this.animating = true;
    for (var a = 0; a < 1; ++a) {
        this.FindFirstAndLast();
        this.anim_container.removeChild(this.navFirstEl);
        this.anim_container.insertBefore(this.navFirstEl, this.navLastEl.nextSibling)
    }
    this.FindFirstAndLast();
    HuffPoUtil.ImageLoader.foldCheck(this.anim_container, true, [100 * 8, 0]);
    this.animating = false
}, SplashSlideshow.prototype.changeOpac = function(c, b) {
    element = Dom.get(c);
    var a = element.style;
    a.opacity = (b);
    a.MozOpacity = (b);
    a.KhtmlOpacity = (b);
    a.filter = "alpha(opacity=" + b * 100 + ")"
}, SplashSlideshow.prototype.fade = function(b, c, a) {
    time = 0.5;
    direction = {
        from: c,
        to: a
    };
    this.animation = new Y.util.Anim($(b), {
        opacity: direction
    }, time, Y.util.Easing.easeOut);
    this.animation.onComplete.subscribe(this.changeBackGround);
    this.animation.animate()
}, SplashSlideshow.prototype.changeBackGround = function() {
    if (!splash_slideshow.carousel) {
        Dom.get("slide_loading_spinner_" + splash_slideshow.slideshow_id).style.display = "none";
        return true
    }
    Dom.get("slide_loading_spinner_" + splash_slideshow.slideshow_id).style.zIndex =- 100;
    Dom.get("slide_loading_spinner_" + splash_slideshow.slideshow_id).style.padding = "0px";
    Dom.get("back_ground").src = splash_slideshow.domain + "gadgets/slideshows/" + splash_slideshow.slideshow_id + "/slide_" + splash_slideshow.slideshow_id + "_" + splash_slideshow.image_id + "_splash.jpg";
    Dom.setAttribute(Dom.get("back_ground"), "width", "900");
    Dom.setAttribute(Dom.get("back_ground"), "height", "360")
}, SplashSlideshow.prototype.clearThumbnails = function() {
    var a = this;
    huff.jquery(function(c) {
        var d = c("#slideshow_splash_navigation_slides_container").children();
        for (var b = 0; b < d.length; b++) {
            a.changeOpac(d[b], 1)
        }
    })
};
var PopupManager = {
    popup_window: null,
    interval: null,
    interval_time: 80,
    onClose: null,
    onCloseParams: null,
    waitForPopupClose: function() {
        if (PopupManager.isPopupClosed()) {
            PopupManager.destroyPopup();
            if ("function" == typeof(PopupManager.onClose)) {
                PopupManager.onClose(PopupManager.onCloseParams);
                PopupManager.onClose = null
            }
        }
    },
    destroyPopup: function() {
        this.popup_window = null;
        window.clearInterval(this.interval);
        this.interval = null;
        this.callback = null
    },
    isPopupClosed: function() {
        return (!this.popup_window || (this.popup_window.closed === true || typeof(this.popup_window.closed) === "undefined"))
    },
    open: function(b, c, a) {
        b = HuffPoUtil.trim(b);
        this.popup_window = window.open(b, "_blank", this.getWindowParams(c, a));
        this.popup_window.name = "hp_popup_" + new Date().getTime();
        this.interval = window.setInterval(this.waitForPopupClose, this.interval_time);
        if (!this.popup_window || this.popup_window.closed) {
            alert("Please accept the pop-up for this, check your pop-up blocker");
            return false
        }
        return this.popup_window
    },
    getWindowParams: function(c, b) {
        var a = this.getCenterCoords(c, b);
        return "width=" + c + ",height=" + b + ",status=1,location=1,resizable=yes,left=" + a.x + ",top=" + a.y
    },
    getCenterCoords: function(b, a) {
        var f = this.getParentCoords();
        var e = this.getWindowInnerSize();
        var d = f.width + Math.max(0, Math.floor((e.width - b) / 2));
        var c = f.height + Math.max(0, Math.floor((e.height - a) / 2));
        return {
            x: d,
            y: c
        }
    },
    getWindowInnerSize: function() {
        var a = 0;
        var b = 0;
        if ("innerWidth" in window) {
            a = window.innerWidth;
            b = window.innerHeight
        } else {
            var c = null;
            if (("BackCompat" === window.document.compatMode) && ("body" in window.document)) {
                c = window.document.body
            } else {
                if ("documentElement" in window.document) {
                    c = window.document.documentElement
                }
            }
            if (c !== null) {
                a = c.offsetWidth;
                b = c.offsetHeight
            }
        }
        return {
            width: a,
            height: b
        }
    },
    getParentCoords: function() {
        var a = 0;
        var b = 0;
        if ("screenLeft" in window) {
            a = window.screenLeft;
            b = window.screenTop
        } else {
            if ("screenX" in window) {
                a = window.screenX;
                b = window.screenY
            }
        }
        return {
            width: a,
            height: b
        }
    }
};
var social_campaign = false;
var social_campaign_ref_param = "comm_ref";
var social_creative_ref_param = "comm_crv";
function Badges(a) {
    this.unique_id = "";
    if (a.unique_id) {
        this.unique_id = a.unique_id
    }
    this.lazy_slices = [];
    this.holders = {};
    this.holder_id = "";
    this.holder = false;
    if (a.holder_id) {
        this.holder_id = a.holder_id;
        this.holder = document.getElementById(this.holder_id);
        if (!this.holder) {
            this.holder = false
        }
    }
    document.getElementById(this.holder_id);
    this.entry_params = {};
    if (a.entry_params) {
        this.entry_params = a.entry_params
    }
    if (!this.entry_params.hasOwnProperty("url")) {
        this.entry_params.url = window.location.href
    }
    this.comment_params = {};
    if (a.comment_params) {
        this.comment_params = a.comment_params
    }
    this.global_object_name = "";
    if (a.global_name) {
        this.global_object_name = a.global_name
    }
    this.force_link_to_share = false;
    if (a.force_link_to_share) {
        this.force_link_to_share = a.force_link_to_share
    }
    this.complete_callback_func_name = false;
    if (a.complete_callback_func_name) {
        this.complete_callback_func_name = a.complete_callback_func_name
    }
    this.complete_callback = false;
    if (a.complete_callback) {
        this.complete_callback = a.complete_callback
    }
    this.share_details_callback = false;
    if (a.share_details_callback) {
        this.share_details_callback = a.share_details_callback
    }
    this.additional_panel_classes = "";
    if (a.additional_panel_classes) {
        this.additional_panel_classes = a.additional_panel_classes
    }
    this.panel_layout = 1;
    if (a.panel_layout) {
        this.panel_layout = a.panel_layout
    }
    this.human_readable = 0;
    if (a.human_readable) {
        this.human_readable = 1
    }
    this.slices = {};
    this.slice_params = {};
    this.panel_border_style = "standard";
    this.redirect_domains = [];
    this.redirect_domains.push("www.aol.com");
    this.slices_id = [];
    this.prev_amounts = [];
    this.current_amounts = [];
    this.current_delta = [0, 0, 0, 0];
    if (HPConfig.inst_type === "dev") {
        this.share_badges_test = 0
    }
    this.layout_bottom = false;
    if (a.hasOwnProperty("layout_bottom")) {
        this.layout_bottom = a.layout_bottom
    }
    HPFB.ensureInit(function() {
        var b = function(f, d, e) {
            var c = HPUtil.GetEntryID(d);
            if ("Liked" == f || "Recommended" == f) {
                Badges.trackShare({
                    eid: c,
                    badge: "fblike"
                });
                SNProject.track(c, "entry_like")
            } else {
                if ("Uniked" == f || "Unecommended" == f) {
                    SNProject.track(c, "entry_unlike")
                }
            }
            return {
                module: Badges.getModuleName()
            }
        };
        HPFB.Tracking.Subscribe(["Liked", "Unliked", "Recommended", "Unrecommended"], ["badges"], true, b)
    })
}
Badges.getModuleName = function(a) {
    var b = "Badges";
    if ("undefined" !== typeof a && "undefined" !== typeof a.badge_version && 2 == a.badge_version) {
        b = "Badges v2"
    }
    if (a && a.hasOwnProperty("layout_bottom") && a.layout_bottom) {
        b += " - Bottom"
    }
    return b
};
Badges.trackEvent = function(a) {
    var b = Badges.getModuleName(a);
    HPTrack.trackEvent(b, a.action, a.label)
};
Badges.trackedShares = [];
Badges.trackShare = function(c) {
    var a = c.badge;
    var b = c.eid;
    if (!(a && b)) {
        return 
    }
    if ( - 1 === jQuery.inArray(a, Badges.trackedShares)) {
        var g = "";
        if ("undefined" != typeof(HPAds)) {
            g = HPAds.get_banners_str()
        }
        jQuery.getJSON("/include/share_track.php?a=post&eid=" + b + "&b=" + a + "&banners=" + g);
        Badges.trackedShares.push(a);
        try {
            var d = "";
            if ("facebook" === a) {
                d = "fb_share"
            } else {
                if ("fblike" === a) {
                    d = "fb_like"
                } else {
                    if ("twitter" === a) {
                        d = a
                    } else {
                        if ("stumble" === a) {
                            d = a
                        } else {
                            if ("email" === a) {
                                d = a
                            } else {
                                if ("meneame" === a) {
                                    d = a
                                } else {
                                    if ("hatena" === a) {
                                        d = a
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (d) {
                bN.set("dL_shareType", d, 1);
                bN.set("dL_shareURL", escape(window.location.href), 1);
                bN.ping("aol-share")
            }
        } catch (f) {}
    }
};
Badges.getUniqName = function() {
    if (typeof(this.last_uniq_id) === "undefined") {
        this.last_uniq_id = 0
    }
    this.last_uniq_id++;
    var c = new Date();
    var b = {};
    var a = Math.floor((c.getTime() / (6000 * 12 * 5)));
    b.unique_id = a + "_" + this.last_uniq_id;
    b.obj_name = "Badges_" + b.unique_id;
    return b
};
Badges.getInstance = function(a) {
    if (!a) {
        return false
    }
    if (!a.badge_div_id) {
        return false
    }
    a.force_class = a.use_glamorous || false;
    var d = Badges.getUniqName();
    a.is_glamorous = a.is_glamorous || false;
    window[d.obj_name] = new Badges({
        unique_id: d.unique_id,
        holder_id: a.badge_div_id,
        complete_callback_func_name: "",
        share_details_callback: false,
        additional_panel_classes: "",
        entry_params: {
            id: a.entry_id,
            url: a.entry_url,
            title: a.entry_title,
            vertical_name: a.vertical_name
        },
        global_name: "window." + d.obj_name
    });
    var c = {
        1: "facebook_glamorous",
        2: "retweet_glamorous",
        3: "email_glamorous",
        4: "google_plusone_glamorous",
        5: "linkedin_glamorous",
        6: "pinterest_glamorous",
        7: "meneame_glamorous",
        8: "hatena_glamorous"
    };
    if (jQuery.browser.mozilla && (jQuery.browser.version <= "1.9.2") || (jQuery.browser.safari && (jQuery.browser.version <= "526.12.2"))) {
        c[4] = "comment_glamorous"
    }
    if (typeof(a.active_badges) !== "undefined") {
        c = {};
        for (var b = 0; b < a.active_badges.length; b++) {
            c[b + 1] = a.active_badges[b]
        }
    }
    window[d.obj_name].setPanelBorderStyle(a.force_class || "glamorous_4");
    window[d.obj_name].setSlices(c);
    window[d.obj_name].panel_layout = 8;
    Badges.launch(a.badge_div_id, d)
};
Badges.launch = function(a, b) {
    jQuery("#" + a).ready(function() {
        window[b.obj_name].start()
    })
};
Badges.prototype = {};
Badges.prototype.setPanelBorderStyle = function(a) {
    this.panel_border_style = a
};
Badges.prototype.setSlices = function(a) {
    this.slices = a
};
Badges.prototype.getSlices = function() {
    return this.slices
};
Badges.prototype.applyHolderStyle = function() {
    var a = jQuery(this.holder);
    a.addClass("badges_v2 " + this.panel_border_style + " " + this.additional_panel_classes);
    this.entry_params.force_fb_like && a.addClass("fb_like_contain");
    a.addClass("badge_layout_" + this.panel_layout);
    HPConfig.blog_id == 3 && this.panel_layout == 10 && a.addClass("v2")
};
Badges.prototype.getShareArticleTitle = function() {
    var a = this.entry_params.title;
    return escape(a)
};
Badges.prototype.getArticleLink = function(c, d) {
    var b = this.entry_params.url;
    if (c) {
        b = HuffPoUtil.AddStringToQueryString(b, c)
    }
    if (typeof social_campaign !== "undefined") {
        b = HuffPoUtil.AddStringToQueryString(b, social_campaign_ref_param + "=" + social_campaign);
        if ((typeof social_campaign_creative !== "undefined") && (typeof social_campaign_creative.facebook !== "undefined")) {
            b = HuffPoUtil.AddStringToQueryString(b, social_creative_ref_param + "=" + social_campaign_creative.facebook)
        }
    }
    var a = HuffCookies.getUserId();
    if (a && d) {
        b += "#sb=" + a + ",b=" + d
    }
    return escape(b)
};
Badges.prototype.getCommentLink = function(c, d) {
    var b = this.comment_params.url;
    if (c) {
        b = HuffPoUtil.AddStringToQueryString(b, c)
    }
    var a = HuffCookies.getUserId();
    if (a && d) {
        b += "#sb=" + a + ",b=" + d
    }
    return escape(b)
};
Badges.prototype.getShareArticleLink = function(a, b) {
    if (this.force_link_to_share) {
        return this.force_link_to_share
    }
    return this.getArticleLink(a, b)
};
Badges.prototype.getShareCommentLink = function(a, b) {
    if (this.force_link_to_share) {
        return this.force_link_to_share
    }
    return this.getCommentLink(a, b)
};
Badges.prototype.getShareCommentData = function() {
    var a = {
        title: "",
        link: ""
    };
    if (!this.share_details_callback) {
        return a
    }
    var b = window[this.share_details_callback](a);
    if (!b) {
        return a
    }
    return b
};
Badges.prototype.slicesCallback = function(cb_params) {
    var me, func_name, slice_holder_id, i, slice_name;
    if (typeof(cb_params) !== "object") {
        return 
    }
    me = eval(cb_params.global_name);
    for (i = 0; i < cb_params.slice_names.length; i++) {
        slice_name = cb_params.slice_names[i];
        func_name = "sliceHandler_" + slice_name;
        if (typeof(me[func_name]) == "function" && cb_params.slice_params[slice_name]) {
            slice_holder_id = me.slice_params[slice_name].slice_holder_id;
            me[func_name](slice_holder_id, cb_params.slice_params[slice_name])
        }
    }
    if (me.comment_params && me.comment_params.id) {
        jQuery(".hp_network_badge").html("");
        jQuery(".hp_network_badge").each(function() {
            var className = jQuery(this).attr("class");
            if (className.indexOf("facebook")!==-1) {
                jQuery(this).addClass("no_count_facebook_badge comment_share_badge")
            } else {
                if (className.indexOf("retweet")!==-1) {
                    jQuery(this).addClass("no_count_retweet_badge comment_share_badge")
                } else {
                    if (className.indexOf("stumble")!==-1) {
                        jQuery(this).addClass("no_count_stumble_badge comment_share_badge")
                    }
                }
            }
        });
        return 
    }
};
Badges.prototype.start = function() {
    if (!this.holder) {
        return false
    }
    this.applyHolderStyle();
    var c, o, v, l, u, h, n, s, j = [];
    for (c in this.slices) {
        if (this.slices.hasOwnProperty(c)) {
            v = this.slices[c];
            l = this.holder_id + "_slice_" + c;
            this.slice_params[v] = {
                slice_holder_id: l
            };
            o = document.createElement("div");
            o.setAttribute("id", l);
            jQuery(o).addClass("slice slice_" + c);
            this.holder.appendChild(o);
            h = "sliceHandler_" + v + "_needParams";
            if (typeof(this[h]) !== "undefined") {
                this.lazy_slices.push(v)
            } else {
                n = "sliceHandler_" + v;
                if (typeof(this[n]) == "function") {
                    this[n](l, false)
                }
            }
            j.push(v.substr(0, 2))
        }
    }
    var f = "";
    if (this.lazy_slices.length > 0) {
        var r = "";
        for (s = 0; s < this.lazy_slices.length; s++) {
            if (r !== "") {
                r += ","
            }
            r += this.lazy_slices[s]
        }
        u = document.createElement("script");
        u.setAttribute("type", "text/javascript");
        this.holder.appendChild(u);
        f = "/badge/badges_json_v2.php?sn=" + r + "&gn=" + this.global_object_name + "&eu=" + this.getArticleLink() + "&id=" + this.entry_params.id + "&ebi" + HPConfig.blog_id + "&entry_design=" + HPConfig.entry_design + "&cb=" + this.global_object_name + ".slicesCallback&ng=" + ((HPConfig.ipad_application === true) ? 1 : 0);
        if (typeof(this.comment_params.id) !== "undefined") {
            f += "&cd=" + this.comment_params.id + "&cu=" + this.comment_params.url
        }
        for (s = 0; s < this.redirect_domains.length; s++) {
            if (this.redirect_domains[s] == HPConfig.current_web_domain) {
                f = "http://www.huffingtonpost.com" + f;
                break
            }
        }
        u.setAttribute("src", f)
    }
    if (this.complete_callback_func_name) {
        window[this.complete_callback_func_name]({
            badge_object: this,
            badge_holder_id: this.holder_id
        })
    }
    if (this.complete_callback) {
        this.complete_callback({
            badge_object: this,
            badge_holder_id: this.holder_id
        })
    }
    var b = "badges_v1";
    if (this.panel_layout == 8) {
        b = "badges_v2"
    }
    HPTrack.Module.Impressions.queue(b + "/" + j.join("_"));
    if (this.comment_params && this.comment_params.id) {
        return 
    }
    if (f.length > 0 && false) {
        var t = ".slicesCallback";
        var d = ".slicesAmount_badges_glamorous";
        f = f.replace(new RegExp(t, "g"), d);
        var p = 0, k = 0;
        slices_id = this.slices_id;
        var q = this;
        var m = function(w, y) {
            p += 1;
            if (typeof w !== "undefined") {
                for (s = 0; s < w.length; s++) {
                    var i = y[s];
                    if (isNaN(i)) {
                        i = 0
                    }
                    i += Math.floor(p * w[s]);
                    jQuery("#" + slices_id[s]).html(HuffPoUtil.number_format(i))
                }
            }
        };
        var e = function(i) {
            if (HPConfig.inst_type === "dev") {
                this.share_badges_test = getEmulData()
            }
            jQuery.getScript(f);
            p = 0;
            if (typeof i !== "undefined") {
                clearTimeout(a)
            }
            var y = function() {
                clearInterval(k);
                k = setInterval(m, 1000, this.current_delta, this.current_amounts)
            };
            var w = this;
            var z = function() {
                y.apply(w, [])
            };
            setTimeout(z, 2000)
        };
        var g = function() {
            e.apply(q, [])
        };
        var a = setTimeout(g, 30000);
        setInterval(g, 60000, true)
    }
    return true
};
Badges.prototype.HpBuildIfr = function(b, c, d) {
    var a = document.createElement("iframe");
    a.height = c;
    a.width = b;
    a.src = d;
    a.frameBorder = "0";
    a.scrolling = "no";
    a.style.border = "0px";
    a.setAttribute("frameBorder", "0");
    return a
};
Badges.prototype.newWindow = function(c, d, a, b, g) {
    d = d || 750;
    a = a || 325;
    b = b || (screen.width / 2) - (d / 2);
    g = g || (screen.height / 2) - (a / 2) - 150;
    var e = "scrollbars=0,width=" + d + ",height=" + a + ",top=" + g + ",left=" + b;
    var f = window.open(c, "badge_v2_share_window", e);
    f.focus();
    return f
};
Badges.prototype.trackBadgeClick = function(c) {
    var f = c.slice_name;
    if (typeof(c.badge_version) === "undefined") {
        c.badge_version = 1
    }
    var e = function(g) {
        g = g.charAt(0).toUpperCase() + g.substr(1);
        if (g == "retweet") {
            g = "Twitter"
        } else {
            if (g == "retweet_glamorous") {
                g = "Twitter"
            } else {
                if (g == "pinterest") {
                    g = "Pinterest"
                } else {
                    if (g == "pinterest_glamorous") {
                        g = "Pinterest"
                    } else {
                        if (g == "meneame") {
                            g = "Meneame"
                        } else {
                            if (g == "meneame_glamorous") {
                                g = "Meneame"
                            } else {
                                if (g == "hatena_glamorous") {
                                    g = "Hatena"
                                }
                            }
                        }
                    }
                }
            }
        }
        return g
    };
    var b = function(g) {
        if (g == "retweet") {
            return "twitter"
        } else {
            if (g == "retweet_glamorous") {
                return "twitter"
            } else {
                if (g == "facebook") {
                    return "fb"
                } else {
                    if (g == "facebook_glamorous") {
                        return "fb"
                    } else {
                        if (g == "pinterest") {
                            g = "pinterest"
                        } else {
                            if (g == "pinterest_glamorous") {
                                g = "pinterest"
                            } else {
                                if (g == "meneame") {
                                    g = "Meneame"
                                } else {
                                    if (g == "meneame_glamorous") {
                                        g = "Meneame"
                                    } else {
                                        if (g == "hatena_glamorous") {
                                            g = "Hatena"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return false
    };
    var d = function(g) {
        if (g == "retweet") {
            return "twitter"
        } else {
            if (g == "retweet_glamorous") {
                return "twitter"
            } else {
                if (g == "facebook") {
                    return "facebook"
                } else {
                    if (g == "facebook_glamorous") {
                        return "facebook"
                    } else {
                        if (g == "pinterest") {
                            g = "pinterest"
                        } else {
                            if (g == "pinterest_glamorous") {
                                g = "pinterest"
                            } else {
                                if (g == "meneame") {
                                    g = "Meneame"
                                } else {
                                    if (g == "meneame_glamorous") {
                                        g = "Meneame"
                                    } else {
                                        if (g == "hatena_glamorous") {
                                            g = "Hatena"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return false
    };
    if (f) {
        Badges.trackEvent({
            badge_version: c.badge_version,
            action: "Click",
            label: e(f),
            layout_bottom: this.layout_bottom
        });
        var a = d(f);
        if (a) {
            if ((f == "retweet") && (typeof social_campaign_tracking !== "undefined") && social_campaign_tracking.twitter_share_action) {
                if (typeof social_campaign_tracking.twitter_share_action == "object") {
                    for (tracking_url in social_campaign_tracking.twitter_share_action) {
                        if ((typeof social_campaign_tracking.twitter_share_action[tracking_url] == "string") && (social_campaign_tracking.twitter_share_action[tracking_url] != "")) {
                            HPUtil.trackerImg(social_campaign_tracking.twitter_share_action[tracking_url], document.body)
                        }
                    }
                } else {
                    if (social_campaign_tracking.twitter_share_action != "") {
                        HPUtil.trackerImg(social_campaign_tracking.twitter_share_action, document.body)
                    }
                }
                if ((typeof social_campaign != "undefined") && HPAds.social_campaigns && HPAds.social_campaigns[social_campaign]) {
                    a = b(f);
                    HPUtil.trackerImg("http://pixel.quantserve.com/pixel/p-4azu6n0QT1rbs.gif?labels=social_action." + social_campaign + "." + a, document.body)
                } else {
                    if (this.tracking_flight_name) {
                        a = b(f);
                        HPUtil.trackerImg("http://pixel.quantserve.com/pixel/p-4azu6n0QT1rbs.gif?labels=social_action." + this.tracking_flight_name + "." + a, document.body)
                    }
                }
            } else {
                if ((f == "facebook") && (typeof social_campaign_tracking !== "undefined") && social_campaign_tracking.facebook_share_action) {
                    if (typeof social_campaign_tracking.facebook_share_action == "object") {
                        for (tracking_url in social_campaign_tracking.facebook_share_action) {
                            if ((typeof social_campaign_tracking.facebook_share_action[tracking_url] == "string") && (social_campaign_tracking.facebook_share_action[tracking_url] != "")) {
                                HPUtil.trackerImg(social_campaign_tracking.facebook_share_action[tracking_url], document.body)
                            }
                        }
                    } else {
                        if (social_campaign_tracking.facebook_share_action != "") {
                            HPUtil.trackerImg(social_campaign_tracking.facebook_share_action, document.body)
                        }
                    }
                }
            }
        }
    }
    HPUtil.updateInfluenceCookie("shared", 1)
};
Badges.prototype.getPinterestCounts = function(d, a) {
    var c = document.createElement("script");
    var b = document.getElementsByTagName("head")[0];
    c.setAttribute("async", "true");
    var e = "http://api.pinterest.com/v1/urls/count.json";
    c.setAttribute("src", e + "?callback=" + a + "&url=" + d);
    b.appendChild(c)
};
window.pinterestGetCounts = function(b) {
    if (b.error) {
        return 
    }
    var c = b.url;
    var a = b.count;
    if (a > 999 && a <= 999999) {
        a = Math.floor(a / 1000) + "K+"
    } else {
        if (a > 999999 && a <= 999999999) {
            a = Math.floor(a / 1000000) + "M+"
        } else {
            if (a > 999999999) {
                a = "++"
            }
        }
    }
    jQuery("#badges_v2_1 .badge_v2_pinterest").html(a)
};
Badges.prototype.sliceHandler_pinterest_glamorous = function(e, d) {
    this.slices_id.push("badge_v2_pinterest_" + this.unique_id);
    var b = jQuery(document.createElement("a"));
    b.attr("id", "badge_v2_pinterest_" + this.unique_id);
    b.attr("href", "javascript:void(0);");
    b.addClass("badge_v2_pinterest hp_network_badge");
    b.html(0).appendTo(document.getElementById(e));
    this.getPinterestCounts(this.entry_params.url, "pinterestGetCounts");
    var c = this;
    b.click({
        scope: this,
        handler: this.userFunc_pinterest_share
    }, function(a) {
        a.data.handler.apply(a.data.scope)
    }).click({
        handler: this.trackBadgeClick
    }, function(a) {
        a.data.handler({
            badge_version: d.badge_version,
            slice_name: "pinterest"
        });
        return false
    })
};
Badges.prototype.userFunc_pinterest_share = function(k) {
    var p = "", d = jQuery("img.pinit").attr("src"), n = this.unique_id, q = document.getElementsByTagName("meta"), m = q.length, h = this.entry_params.url.replace(/\?.*/, "");
    if (!d) {
        jQuery("#blog_author_info").find("img").each(function(c) {
            d = jQuery(this).attr("src")
        });
        jQuery("meta").each(function(c) {
            var i = jQuery(this).attr("content");
            if (i && i.indexOf("i.huffpost.com") >= 0) {
                var e = jQuery(this).attr("content");
                if (e) {
                    d = e
                }
            }
        })
    }
    if (jQuery(".pinitdesc").length == 0) {
        for (var b = 0; b < m; b++) {
            if (!q[b].name) {
                continue
            }
            if (q[b].name == "description") {
                p = q[b].content
            }
        }
    } else {
        p = jQuery(".pinitdesc").html()
    }
    if (jQuery(".piniturl").length > 0) {
        h = encodeURIComponent(jQuery(".piniturl").attr("href"))
    }
    var a = 500;
    var j = encodeURIComponent(p);
    if (j.length > a) {
        j = j.substr(0, a - 3) + "..."
    }
    var l = "http://pinterest.com/pin/create/button/?url=" + h + "&media=" + d + "&description=" + j;
    var o = this.newWindow(l, 665, 600), g = this;
    var f = function() {
        setTimeout(function() {
            if (o == null) {
                return 
            }
            if (o.closed) {
                g.getPinterestCounts(h, "pinterestGetCounts")
            } else {
                f()
            }
        }, 25)
    };
    f();
    return false
};
Badges.prototype.sliceHandler_linkedin_glamorous_needParams = true;
Badges.prototype.sliceHandler_linkedin_glamorous = function(f, e) {
    var d = parseInt(e.linkedin_amount);
    if (isNaN(d)) {
        d = 0
    }
    this.slices_id.push("badge_v2_retweet_" + this.unique_id);
    this.current_amounts.push(d);
    this.prev_amounts.push(d);
    var b = jQuery(document.createElement("a"));
    b.attr("id", "badge_v2_linkedin_" + this.unique_id);
    b.attr("href", "javascript:void(0);");
    b.addClass("badge_v2_linkedin hp_network_badge");
    b.html(HuffPoUtil.number_format(d)).appendTo(document.getElementById(f));
    e.badge_version = 8;
    var c = this;
    b.bind("click", function() {
        c.userFunc_linkedin_share();
        c.trackBadgeClick({
            badge_version: 8,
            slice_name: "linkedin"
        })
    })
};
Badges.prototype.userFunc_linkedin_share = function() {
    var c = jQuery("meta[name=description]").attr("content") || "";
    var b = 500;
    if (c.length > b) {
        c = c.substr(0, b - 3) + "..."
    }
    var e = this.getShareArticleTitle();
    if (e.length > b) {
        e = e.substr(0, b - 3) + "..."
    }
    var d = parseInt(jQuery("#badge_v2_linkedin_" + this.unique_id).html()) + 1;
    jQuery("#badge_v2_linkedin_" + this.unique_id).html(d);
    var a = {
        entry_id: this.entry_params.id,
        url: this.getShareArticleLink(),
        title: e,
        summary: encodeURIComponent(c)
    };
    Sharer.Linkedin.share(a);
    return false
};
Badges.prototype.sliceHandler_facebook_glamorous_needParams = true;
Badges.prototype.sliceHandler_facebook_glamorous = function(e, d) {
    var b = document.createElement("a");
    b.setAttribute("id", "badge_v2_facebook_" + this.unique_id);
    b.setAttribute("href", "javascript:void(0);");
    jQuery(b).addClass("badge_v2_facebook_ipad_app hp_network_badge");
    var c = parseInt(d.share_amount);
    if (isNaN(c)) {
        c = 0
    }
    if (this.human_readable) {
        b.innerHTML = HuffPoUtil.int2size(c)
    } else {
        b.innerHTML = HuffPoUtil.number_format(c)
    }
    this.slices_id.push("badge_v2_facebook_" + this.unique_id);
    this.current_amounts.push(c);
    this.prev_amounts.push(c);
    document.getElementById(e).appendChild(b);
    this.userFunc_facebook_add_click_listener(b, d)
};
Badges.prototype.sliceHandler_facebook_techcrunch_needParams = true;
Badges.prototype.sliceHandler_facebook_techcrunch = function(d, c) {
    var a = document.createElement("li");
    var b = document.createElement("a");
    b.setAttribute("id", "badge_v2_facebook_" + this.unique_id);
    jQuery(b).addClass("facebook");
    b.innerHTML = HuffPoUtil.number_format(c.share_amount);
    a.appendChild(b);
    document.getElementById(d).appendChild(a);
    this.userFunc_facebook_add_click_listener(b, c)
};
Badges.prototype.userFunc_facebook_add_click_listener = function(c, e) {
    var a = this;
    var d = "facebook";
    var b = this.getShareArticleLink("utm_hp_ref=fb&src=sp", d);
    jQuery(c).bind("click", function() {
        a.userFunc_facebook_share_url(b);
        a.trackBadgeClick({
            badge_version: 9,
            slice_name: d
        });
        var f = HPUtil.GetEntryID(b);
        if (typeof(SNProject) === "object") {
            SNProject.track(f, "entry_facebook_share")
        }
        return false
    })
};
Badges.prototype.userFunc_facebook_share_url = function(a) {
    this.userFunc_facebook_show_sharer(a);
    return false
};
Badges.prototype.userFunc_facebook_show_sharer = function(a) {
    if (social_campaign && HPAds.social_campaigns && HPAds.social_campaigns[social_campaign] && HPAds.social_campaigns[social_campaign].fb_100x20) {
        var b = document.getElementsByTagName("link");
        for (x in b) {
            if (b[x].rel && b[x].rel == "image_src") {
                b[x].href = "http://www.huffingtonpost.com/ads/creative/image_compositor.php?overlay=" + escape(HPAds.social_campaigns[social_campaign].fb_100x20) + "&image=" + escape(b[x].href)
            }
        }
    }
    FB.ui({
        method: "feed",
        link: decodeURIComponent(a)
    });
    Badges.trackShare({
        eid: this.entry_params.id,
        badge: "facebook"
    })
};
Badges.prototype.sliceHandler_retweet_glamorous_needParams = true;
Badges.prototype.sliceHandler_retweet_glamorous = function(g, d) {
    var e = parseInt(d.views_amount) || 0;
    this.slices_id.push("badge_v2_retweet_" + this.unique_id);
    this.current_amounts.push(e);
    this.prev_amounts.push(e);
    var h = e;
    if (this.human_readable) {
        h = HuffPoUtil.int2size(e)
    } else {
        h = HuffPoUtil.number_format(e)
    }
    var f = jQuery("<a />").attr("id", "badge_v2_retweet_" + this.unique_id).addClass("badge_v2_retweet_ipad_app hp_network_badge").html(h).appendTo(document.getElementById(g));
    var i = this.entry_params.custom_tweet_text || this.entry_params.title || "";
    var c = function(s) {
        var v = this.entry_params.custom_tweet_text || this.entry_params.title || "";
        var r, t = window.social_campaign, o = window.social_campaign_creative, l = HPAds.social_campaigns;
        if ((typeof(o) !== "undefined") && (typeof(o.constructor) == "function") && o.twitter && v.search(o.twitter)==-1) {
            var k = o.twitter_optout_language ? o.twitter_optout_language: "";
            r = {
                text: k,
                hash: o.twitter
            }
        } else {
            if (t && l && l[t] && l[t].hashtag && v.search(l[t].hashtag)==-1) {
                r = {
                    text: l[t].twitter_optout_language,
                    hash: l[t].hashtag
                }
            } else {
                if (this.entry_params.tweet_comm_hash && v.search(this.entry_params.tweet_comm_hash)==-1) {
                    r = {
                        text: this.entry_params.tweet_comm_text,
                        hash: this.entry_params.tweet_comm_hash
                    }
                }
            }
        }
        var p = (HPConfig.twitter_signature ? HPConfig.twitter_signature : "@HuffingtonPost").replace("@", "");
        if ("undefined" !== typeof this.entry_params.entry_author && "HuffPost Partner Studio" === this.entry_params.entry_author) {
            p = "@HuffPostPartner"
        }
        var u = " " + s;
        var q = "";
        if (r) {
            v = r.text ? r.text : v;
            q = " " + r.hash
        }
        var n = " via @" + p;
        if (v.length > 140 - q.length - u.length - n.length) {
            v = v.substring(0, 140 - q.length - u.length - n.length - 3) + "..."
        }
        v += q + u;
        var m = "en";
        if (typeof jQuery.I18N !== "undefined") {
            m = jQuery.I18N.getLocale().substr(0, 2)
        }
        v = v.replace(/<[^>]+>/ig, "").replace(/\s+/, " ");
        f.attr("href", "https://twitter.com/intent/tweet?text=" + encodeURIComponent(v) + "&via=" + p + "&lang=" + m)
    };
    var j = this, b = this.entry_params.id;
    var a = d.short_url || decodeURIComponent(this.getShareArticleLink("utm_hp_ref=twitter")) || "";
    if (a && a.length > 30) {
        HPUtil.bitly_url(a, function(k) {
            j.short_url = k
        })
    } else {
        this.short_url = a
    }
    f.click(function() {
        c.call(j, j.short_url);
        Badges.trackShare({
            eid: b,
            badge: "twitter"
        });
        j.trackBadgeClick({
            badge_version: d.badge_version,
            slice_name: "retweet"
        })
    })
};
Badges.prototype.sliceHandler_retweet_techcrunch_needParams = true;
Badges.prototype.sliceHandler_retweet_techcrunch = function(d, c) {
    var a = document.createElement("li");
    var b = jQuery(document.createElement("a"));
    b.attr("id", "badge_v2_retweet_" + this.unique_id).addClass("twitter").html(HuffPoUtil.number_format(c.views_amount)).appendTo(a).parent().appendTo(document.getElementById(d));
    c.badge_version = 9;
    this.userFunc_retweet_prepare_button(b, c)
};
Badges.prototype.userFunc_retweet_prepare_button = function(c, g) {
    var b = 140;
    var f = this.entry_params.custom_tweet_text || this.entry_params.title || "";
    if (f.length > b) {
        f = f.substr(0, b - 3) + "..."
    }
    this.tweet_descr = {
        is_ready: false,
        maxLimit: b,
        shortUrl: g.short_url || "",
        tweet_text: f
    };
    var e = function(h, i) {
        Badges.trackShare({
            eid: i.eid,
            badge: "twitter"
        })
    };
    var a = {
        eid: this.entry_params.id
    };
    var d = this;
    c.click(function() {
        e(d, a)
    }).click({
        scope: this,
        handler: this.userFunc_post_tweet
    }, function(h) {
        h.data.handler.apply(h.data.scope)
    }).click({
        handler: this.trackBadgeClick
    }, function(h) {
        h.data.handler({
            badge_version: g.badge_version,
            slice_name: "retweet"
        });
        return false
    })
};
Badges.prototype.userFunc_post_tweet = function() {
    var d = false, i;
    if ((typeof social_campaign != "undefined") && (typeof social_campaign_creative == "object") && (typeof social_campaign_creative.twitter !== "undefined") && (this.tweet_descr.tweet_text.search(social_campaign_creative.twitter)==-1)) {
        var e = (typeof social_campaign_creative.twitter_optout_language != "undefined") ? social_campaign_creative.twitter_optout_language: "";
        d = {
            text: e,
            hash: social_campaign_creative.twitter
        }
    } else {
        if (social_campaign && HPAds.social_campaigns && HPAds.social_campaigns[social_campaign] && HPAds.social_campaigns[social_campaign].hashtag && (this.tweet_descr.tweet_text.search(HPAds.social_campaigns[social_campaign].hashtag)==-1)) {
            d = {
                text: HPAds.social_campaigns[social_campaign].twitter_optout_language,
                hash: HPAds.social_campaigns[social_campaign].hashtag
            }
        } else {
            if (this.entry_params.tweet_comm_hash && (this.tweet_descr.tweet_text.search(this.entry_params.tweet_comm_hash)==-1)) {
                d = {
                    text: this.entry_params.tweet_comm_text,
                    hash: this.entry_params.tweet_comm_hash
                }
            }
        }
    }
    var g = "";
    switch (this.tweet_descr.url_mode) {
    case"comment_share":
        var c = this.getShareCommentData();
        g = c.link;
        break;
    case"comment":
        g = this.getShareCommentLink();
        break;
    default:
        g = this.getShareArticleLink("utm_hp_ref=twitter");
        break
    }
    var f = false;
    if (this.tweet_descr.shortUrl === "" || social_campaign) {
        i = [{
            insert_type: "add_to_end",
            make_short: true,
            url: g
        }
        ]
    } else {
        i = [{
            insert_type: "add_to_end",
            url: this.tweet_descr.shortUrl
        }
        ];
        f = this.tweet_descr.shortUrl
    }
    var b = function() {
        Badges.trackEvent({
            action: "Shared",
            label: "Twitter"
        })
    };
    var a = 25;
    var h = "";
    if (HuffCookies.getUserId()&&!HuffPrefs.get("twitter")) {
        b = Badges.prototype.userFunc_twitter_show_promo_modal_window;
        a = 1000;
        h = this.holder_id
    }
    Sharer.Twitter.share({
        tweet_text: this.tweet_descr.tweet_text,
        commercial_info: d,
        links: i,
        on_success_callback: b,
        on_success_timeout: a,
        on_success_callback_params: h,
        badge_holder_id: this.holder_id,
        entry_url: this.entry_params.url,
        entry_short_url: f
    });
    return false
};
Badges.prototype.userFunc_twitter_show_promo_modal_window = function(c) {
    HuffConnect.hideModal();
    var b = '<div style="font-size: 14pt; font-weight: bold">' + _("Do you want to connect to HuffPost using Twitter?") + "</div>";
    b += '<a class="login_light_link" ';
    b += 'href="/social/join.html?autojoin=1" ';
    b += "onclick=\"linkSocialAccount.checkLoginStatus('twitter'); return false;\"";
    b += 'target="_blank">';
    b += '<span class="twitter-darker">' + _("Connect with Twitter") + "</span>";
    b += "</a>";
    var a = {
        inner_html: b,
        modal_head_image: "",
        modal_subhead_image: "",
        position: "fixed"
    };
    if (c && "quickread_badges" == c) {
        a.onclose = function() {
            HPAds.ad_reload("quickread", "qr_ad");
            return 
        };
        HuffConnect.socialModal(a)
    } else {
        HuffConnect.socialModal(a)
    }
    return 
};
Badges.prototype.sliceHandler_comment_glamorous_needParams = true;
Badges.prototype.sliceHandler_comment_glamorous = function(e, d) {
    var b = document.createElement("a");
    jQuery(b).addClass("badge_v2_comments");
    b.setAttribute("href", "javascript:void(0);");
    b.setAttribute("id", "badge_v2_comments_" + this.unique_id);
    jQuery(b).addClass("badge_v2_comments_ipad_app");
    var c = parseInt(d.comments_amount);
    if (isNaN(c)) {
        c = 0
    }
    if (this.human_readable) {
        b.innerHTML = HuffPoUtil.int2size(c)
    } else {
        b.innerHTML = HuffPoUtil.number_format(c)
    }
    b.innerHTML = c;
    this.slices_id.push("badge_v2_comments_" + this.unique_id);
    this.current_amounts.push(c);
    this.prev_amounts.push(c);
    if (HPConfig.ipad_application === true) {
        jQuery(b).bind("click", function() {
            var g = jQuery("a[name=comments]");
            var a = jQuery("div#container-for-entry");
            if (!g ||!a) {
                return 
            }
            var f = "slow";
            var h =- 45;
            a.animate({
                scrollTop: g.offset().top + h
            }, f)
        })
    } else {
        jQuery(b).bind("click", function() {
            jQuery("html,body").animate({
                scrollTop: jQuery("a[name=comments]").offset().top
            }, 1700)
        })
    }
    document.getElementById(e).appendChild(b);
    if (typeof(document.badgeCommentAmountIncrease) === "undefined") {
        document.badge_comment_panels = [];
        document.badgeCommentRegisterPanel = function(a) {
            document.badge_comment_panels.push(a)
        };
        document.badgeCommentAmountIncrease = function(g) {
            var a, f;
            for (f in document.badge_comment_panels) {
                if (document.badge_comment_panels.hasOwnProperty(f)) {
                    a = document.badge_comment_panels[f];
                    a.userFunc_set_comments_amount(g)
                }
            }
        }
    }
    document.badgeCommentRegisterPanel(this);
    jQuery(b).click(jQuery.proxy(function() {
        this.trackBadgeClick({
            badge_version: 2,
            slice_name: "comments"
        })
    }, this));
    this.userFunc_set_comments_amount = function(f) {
        var a = document.getElementById("badge_v2_comments_" + this.unique_id);
        if (!a) {
            return 
        }
        while (a.childNodes.length >= 1) {
            a.removeChild(a.firstChild)
        }
        a.appendChild(document.createTextNode(f))
    }
};
Badges.prototype.sliceHandler_digg_small = function(c, b) {
    var a = HuffPoUtil.getNode("#" + c);
    this._innerHandler_digg_small(a, b, {});
    return 
};
Badges.prototype._innerHandler_digg_small = function(c, e, b) {
    var a = b || {};
    var d = {
        url: this.entry_params.url,
        title: this.entry_params.title
    };
    this._createDiggButton(c, d, a);
    this._digg_track(c);
    return 
};
Badges.prototype._createDiggButton = function(c, e, b) {
    var d = document.createElement("a");
    d.href = "http://digg.com/submit?url=" + escape(e.url) + "&title=" + encodeURIComponent(e.title);
    d.className = "DiggThisButton DiggThisButtonMedium";
    if (b && b.class_name) {
        var a = document.createElement("div");
        a.className = b.class_name;
        a.appendChild(d);
        c.appendChild(a)
    } else {
        c.appendChild(d)
    }
    HPUtil.loadAndRun("http://widgets.digg.com/buttons.js")
};
Badges.prototype._digg_track = function(a) {
    jQuery(a).bind("click", function(d) {
        var b = HuffPoUtil.GetEntryID();
        if (!b) {
            return 
        }
        var c = d.target || d.srcElement;
        if (c.className && c.className.indexOf("db-") === 0) {
            this.trackEvent("Click", "Digg")
        }
        Badges.trackShare({
            eid: b,
            badge: "digg"
        })
    })
};
Badges.prototype.sliceHandler_stumble_needParams = true;
Badges.prototype.sliceHandler_stumble = function(d, a) {
    var c = document.createElement("div");
    if (a.stumble_views > 9) {
        jQuery(c).addClass("badge_v2_stumble_count_div");
        c.innerHTML = a.stumble_views
    } else {
        jQuery(c).addClass("badge_v2_stumble_no_count_div")
    }
    var b = this;
    jQuery(c).bind("click", {
        url: "http://www.stumbleupon.com/toolbar/badge_click.php?r=" + this.getShareArticleLink(null, "stumble"),
        eid: this.entry_params.id
    }, function(f) {
        Badges.trackEvent({
            action: "Click",
            label: "Stumble",
            layout_bottom: b.layout_bottom
        });
        Badges.trackShare({
            eid: f.data.eid,
            badge: "stumble"
        });
        b.newWindow(f.data.url, 830, 500);
        return false
    });
    HuffPoUtil.getNode("#" + d + "").appendChild(c)
};
Badges.prototype.sliceHandler_stumble_comment_needParams = true;
Badges.prototype.sliceHandler_stumble_comment = function(d, a) {
    var c = document.createElement("div");
    if (a.stumble_views > 9) {
        jQuery(c).addClass("badge_v2_stumble_count_div");
        c.innerHTML = a.stumble_views
    } else {
        jQuery(c).addClass("badge_v2_stumble_no_count_div")
    }
    var b = this;
    jQuery(c).bind("click", {
        url: "http://www.stumbleupon.com/toolbar/badge_click.php?r=" + this.getShareCommentLink(null, "stumble")
    }, function(f) {
        Badges.trackEvent({
            action: "Click",
            label: "Stumble",
            layout_bottom: b.layout_bottom
        });
        b.newWindow(f.data.url);
        return false
    });
    HuffPoUtil.getNode("#" + d + "").appendChild(c)
};
Badges.prototype.sliceHandler_email_glamorous_needParams = true;
Badges.prototype.sliceHandler_email_glamorous = function(e, d) {
    var b = document.createElement("a"), a = parseInt(d.emails_amount);
    if (isNaN(a)) {
        a = 0
    }
    b.setAttribute("id", "badge_v2_email_" + this.unique_id);
    jQuery(b).addClass("badge_v2_email_ipad_app");
    if (this.human_readable) {
        b.innerHTML = HuffPoUtil.int2size(a)
    } else {
        b.innerHTML = HuffPoUtil.number_format(a)
    }
    this.slices_id.push("badge_v2_email_" + this.unique_id);
    this.current_amounts.push(a);
    this.prev_amounts.push(a);
    var c = this.layout_bottom;
    if (HPConfig.ipad_application === true) {
        b.href = "mailto:?subject=" + escape(this.entry_params.title) + "&body=" + this.entry_params.url;
        jQuery(b).bind("click", {
            eid: this.entry_params.id,
            human_readable: this.human_readable
        }, function(f) {
            if (f.data.human_readable) {
                b.innerHTML = HuffPoUtil.int2size(a + 1)
            } else {
                b.innerHTML = HuffPoUtil.number_format(a + 1)
            }
            Badges.trackEvent({
                badge_version: 2,
                action: "Click",
                label: "Email",
                layout_bottom: c
            });
            Badges.trackShare({
                eid: f.data.eid,
                badge: "email"
            });
            return false
        })
    } else {
        b.href = "javascript:void(0)";
        jQuery(b).bind("click", {
            eid: this.entry_params.id,
            human_readable: this.human_readable
        }, function(f) {
            if (f.data.human_readable) {
                b.innerHTML = HuffPoUtil.int2size(a + 1)
            } else {
                b.innerHTML = HuffPoUtil.number_format(a + 1)
            }
            SharePost.emailForm(f.data.eid, "", "");
            Badges.trackEvent({
                badge_version: 8,
                action: "Click",
                label: "Email",
                layout_bottom: c
            });
            return false
        })
    }
    document.getElementById(e).appendChild(b)
};
Badges.prototype.sliceHandler_comments_techcrunch_needParams = true;
Badges.prototype.sliceHandler_comments_techcrunch = function(d, c) {
    var a = document.createElement("li");
    var b = document.createElement("a");
    b.setAttribute("id", "badge_v2_comments_" + this.unique_id);
    b.setAttribute("href", this.entry_params.url + "#comments");
    jQuery(b).addClass("message");
    b.innerHTML = HuffPoUtil.number_format(c.comments_amount);
    a.appendChild(b);
    document.getElementById(d).appendChild(a)
};
Badges.prototype.sliceHandler_meneame_glamorous_needParams = true;
Badges.prototype.sliceHandler_meneame_glamorous = function(h, f) {
    var b = window.location.href, e = document.title, d = "badge_v2_meneame", a = d + "_" + this.unique_id, g = this, c = f.meneame_amount;
    isNaN(c) && (c = 0);
    if (this.human_readable) {
        c = HuffPoUtil.int2size(c)
    } else {
        c = HuffPoUtil.number_format(c)
    }
    this.slices_id.push(a);
    this.current_amounts.push(f.share_amount);
    this.prev_amounts.push(f.share_amount);
    huff.use("jquery", function(j) {
        var i = j('<a href="http://www.meneame.net/submit.php?url=' + encodeURIComponent(b) + "&title=" + encodeURIComponent(e) + '"></a>');
        i.addClass(d).attr("id", a).html(c);
        h = j("#" + h);
        h.append(i);
        i.click(function(k) {
            k.preventDefault();
            g.newWindow(j(this).attr("href"), 790, 600);
            Badges.trackShare({
                eid: HPUtil.GetEntryID(b),
                badge: "meneame"
            });
            i.html(++c)
        }).click({
            handler: g.trackBadgeClick
        }, function(k) {
            k.data.handler({
                badge_version: f.badge_version,
                slice_name: "meneame"
            })
        })
    })
};
Badges.prototype.sliceHandler_hatena_glamorous_needParams = true;
Badges.prototype.sliceHandler_hatena_glamorous = function(d, b) {
    var e = this.entry_params.url.split("?", 1)[0] || window.location.href;
    e = e.split("#", 1)[0];
    var i = this.entry_params.title || document.title;
    var a = e, f = i, g = "badge_v2_hatena", h = g + "_" + this.unique_id, c = this;
    huff.use("jquery", function(k) {
        var j = function() {
            var l = k('<div><a href="http://b.hatena.ne.jp/entry/' + a + '" class="hatena-bookmark-button" data-hatena-bookmark-title="' + f + '" data-hatena-bookmark-layout="vertical-balloon" title="' + f + '"><img src="http://b.st-hatena.com/images/entry-button/button-only.gif" alt="' + f + '" width="20" height="20" style="border: none;" /></a></div>');
            l.addClass(g).attr("id", h);
            d = k("#" + d);
            d.append(l);
            l.parent().addClass("hatena_glamorous");
            return 
        };
        HPUtil.loadAndRun("http://b.st-hatena.com/js/bookmark_button.js", function() {
            j()
        })
    })
};
Badges.prototype.sliceHandler_google_plusone_glamorous_needParams = true;
Badges.prototype.sliceHandler_google_plusone_glamorous = function(c, a) {
    var b = this.entry_params.url.split("?", 1)[0];
    (window.IS_NEWSGLIDE || window.IS_CHROMEAPP)&&!window.huff && (window.huff = {});
    if (window.IS_NEWSGLIDE || window.IS_CHROMEAPP) {
        (function(d) {
            c = d("#" + c);
            d.getScript("https://apis.google.com/js/plusone.js?lang=en-US", function(e) {
                if (!(window.gapi && gapi.plusone)) {
                    return 
                }
                c.html('<div id="plusone-div"></div>').parent().addClass("google_plus1_badge");
                gapi.plusone.render("plusone-div", {
                    size: "tall",
                    href: b
                })
            })
        }(jQuery))
    } else {
        huff.use("jquery", function(d) {
            c = d("#" + c);
            var e = "en-US";
            if (typeof jQuery.I18N !== "undefined") {
                e = jQuery.I18N.getLocale().replace("_", "-")
            }
            window.___gcfg = {
                lang: e
            };
            d.getScript("https://apis.google.com/js/plusone.js", function(f) {
                if (!(window.gapi && gapi.plusone)) {
                    return 
                }
                c.html('<div id="plusone-div"></div>').parent().addClass("google_plus1_badge");
                jQuery("#plusone-div").parent().addClass("google-plusone");
                gapi.plusone.render("plusone-div", {
                    size: "tall",
                    href: b,
                    callback: function(g) {
                        try {
                            if ("on" === g.state) {
                                bN.set("dL_shareType", "google+", 1);
                                bN.set("dL_shareURL", escape(g.href), 1);
                                bN.ping("dL_share")
                            }
                        } catch (h) {}
                    }
                })
            })
        })
    }
};
Badges.prototype.slicesAmount_badges_glamorous = function(cb_params) {
    var me, func_name, i, slice_name;
    if (typeof(cb_params) !== "object") {
        return 
    }
    this.prev_amounts = this.current_amounts;
    this.current_amounts = [];
    this.current_delta = [];
    me = eval(cb_params.global_name);
    for (i = 0; i < cb_params.slice_names.length; i++) {
        slice_name = cb_params.slice_names[i];
        func_name = "sliceAmount_" + slice_name;
        if (typeof(me[func_name]) == "function" && cb_params.slice_params[slice_name]) {
            me[func_name](cb_params.slice_params[slice_name])
        }
    }
    for (i = 0; i < this.current_amounts.length; i++) {
        if (this.current_amounts[i] > 0) {
            var delta = (this.current_amounts[i] - this.prev_amounts[i]) / 60;
            this.current_delta[i] = (delta > 0 ? delta : 0)
        } else {
            this.current_delta[i] = 0
        }
    }
};
Badges.prototype.sliceAmount_facebook_glamorous_needParams = true;
Badges.prototype.sliceAmount_facebook_glamorous = function(a) {
    if (HPConfig.inst_type === "dev") {
        count_amount = (a.share_amount) ? share_badges_test : 0
    } else {
        count_amount = parseInt(a.share_amount)
    }
    if (isNaN(count_amount)) {
        count_amount = 0
    }
    jQuery("#" + this.slices_id[0]).html(HuffPoUtil.number_format(count_amount));
    this.current_amounts.push(count_amount)
};
Badges.prototype.sliceAmount_retweet_glamorous_needParams = true;
Badges.prototype.sliceAmount_retweet_glamorous = function(a) {
    count_amount = parseInt(a.views_amount);
    if (isNaN(count_amount)) {
        count_amount = 0
    }
    jQuery("#" + this.slices_id[1]).html(HuffPoUtil.number_format(count_amount));
    this.current_amounts.push(count_amount)
};
Badges.prototype.sliceAmount_email_glamorous_needParams = true;
Badges.prototype.sliceAmount_email_glamorous = function(a) {
    count_amount = parseInt(a.emails_amount);
    if (isNaN(count_amount)) {
        count_amount = 0
    }
    jQuery("#" + this.slices_id[2]).html(HuffPoUtil.number_format(count_amount));
    this.current_amounts.push(count_amount)
};
Badges.prototype.sliceAmount_comment_glamorous_needParams = true;
Badges.prototype.sliceAmount_comment_glamorous = function(a) {
    count_amount = parseInt(a.comments_amount);
    if (isNaN(count_amount)) {
        count_amount = 0
    }
    jQuery("#" + this.slices_id[3]).html(HuffPoUtil.number_format(count_amount));
    this.current_amounts.push(count_amount)
};
function getEmulData() {
    var a = new Date();
    var b = a.getTime().toString();
    share_badges_test = Math.floor(b.substr(10));
    return share_badges_test
};
window._ || (window._ = function(a) {
    return a
});
window.ngettext || (window.ngettext = function(a) {
    return a
});
window.sprintf || (window.sprintf = function(a) {
    return a
});
var UserLevels = {
    BadgePopup: {
        timeout: 0,
        badge_popup_timer_id: 0,
        showBadgePopup: function(d) {
            if (HPConfig.ipad_application) {
                this.leftside_popup = true
            }
            badge_el = d.badge_el || false;
            badge_el_offset_x = d.badge_el_offset_x || (this.leftside_popup?-312 : 28);
            badge_el_offset_y = d.badge_el_offset_y || 55;
            user_name = d.user_name || false;
            user_type = d.user_type || false;
            user_level = d.user_level || 0;
            gift_points = d.gift_points || 0;
            gift_name = d.gift_name || "";
            comments_cnt = d.comments_cnt || 0;
            fans_cnt = d.fans_cnt || 0;
            friends_cnt = d.friends_cnt || 0;
            following_cnt = d.following_cnt || 0;
            shares_cnt = d.shares_cnt || 0;
            twitter_id = d.twitter_id || 0;
            facebook_id = d.facebook_id || 0;
            level_aquired = d.level_aquired === false ? false : true;
            hide_learn_more = d.hide_learn_more || false;
            sponsor_display_name = d.sponsor_display_name || "";
            author_created_on = d.author_created_on || "";
            author_nickname = d.author_nickname || "";
            if (this.badge_popup_timer_id) {
                clearTimeout(this.badge_popup_timer_id);
                this.badge_popup_timer_id = 0
            }
            var j = document.getElementById("badge_popup");
            var g = document.getElementById("badge_popup_text");
            var i = jQuery("#badge_popup .badge_popup_top");
            if (!j ||!badge_el) {
                return 
            }
            var m = "", k = "";
            if (user_type == "sponsor") {
                level_aquired = false
            }
            if (level_aquired) {
                k = '<span class="' + user_type + '_title title">' + user_name
            }
            var e = '<span class="' + user_type + '_name"><b>' + user_name + "</b></span>";
            switch (user_type) {
            case"networker":
                if (level_aquired) {
                    k += " " + sprintf(_("is a Level %s Networker!"), user_level) + "</span>";
                    switch (user_level) {
                    case 1:
                        m += sprintf(_("%s has earned the Level 1 Networker Badge! %s has <b>%s friends and %s followers</b>."), e, e, friends_cnt, fans_cnt);
                        m += "<br/><br/>" + _("Networkers with a Level 1 Badge are very socially connected users on the site—with lots of friends and followers.");
                        break;
                    case 2:
                        var c = "";
                        if ((twitter_id > 0) && (facebook_id > 0)) {
                            c = "Twitter " + _("and") + " Facebook"
                        } else {
                            if (twitter_id > 0) {
                                c = "Twitter"
                            } else {
                                if (facebook_id > 0) {
                                    c = "Facebook"
                                }
                            }
                        }
                        m += sprintf(_("%s has earned the Level 2 Networker Badge! With <b>%s friends and %s followers</b>, %s has comments featured in red. %s has created a HuffPost account using %s."), e, friends_cnt, fans_cnt, e, e, c);
                        m += "<br/><br/>" + _("Networkers with a Level 2 Badge are among the most socially connected users on the site—with tons of friends and followers. They have also created a HuffPost account using Twitter or Facebook, and their comments are featured in red.");
                        break
                    }
                } else {
                    m += _("Networkers are very socially connected users on the site—with lots of friends and followers! Users who connect with Facebook and Twitter can earn higher levels of this badge.")
                }
                break;
            case"superuser":
                if (level_aquired) {
                    k += " " + sprintf(_("is a Level %s Superuser!"), user_level) + "</span>";
                    switch (user_level) {
                    case 1:
                        m += sprintf(_("%s has earned the Level 1 Superuser Badge! %s has <b>%s comments</b> and <b>%s shares</b> to a social network."), e, e, comments_cnt, shares_cnt);
                        break;
                    case 2:
                        var c = "";
                        if ((twitter_id > 0) && (facebook_id > 0)) {
                            c = "Twitter " + _("and") + " Facebook"
                        } else {
                            if (twitter_id > 0) {
                                c = "Twitter"
                            } else {
                                if (facebook_id > 0) {
                                    c = "Facebook"
                                }
                            }
                        }
                        m += sprintf(_("%s has earned the Level 2 Superuser Badge! With <b>%s comments</b> and <b>%s shares</b> to a social network, %s has comments featured in purple. %s has created a HuffPost account using %s."), e, comments_cnt, shares_cnt, e, e, c);
                        break
                    }
                } else {
                    m += _("Superusers are very engaged users on the site-with lots of comments and shares! Users who connect with Facebook and Twitter can earn higher levels of this badge.")
                }
                break;
            case"moderator":
                if (level_aquired) {
                    k += " " + sprintf(_("is a Level %s Community Moderator!"), user_level) + "</span>";
                    switch (user_level) {
                    case 1:
                        m += sprintf(_("%s has earned the Level 1 Community Moderator Badge! %s flagged at least 20 comments that we deleted and has a high ratio of good flags to mistaken flags, so <span class='%s_name'>%s's</span> flags carry five times the weight as standard flags."), e, e, user_type, user_name);
                        break;
                    case 2:
                        m += sprintf(_("%s has earned the Level 2 Community Moderator Badge! %s flagged at least <b>100 comments</b> that we deleted and has a very high ratio of good flags to mistaken flags, so %s is trusted to delete inappropriate comments from the site. %s has maintained the ability to delete inappropriate comments by handling it responsibly."), e, e, e, e);
                        break
                    }
                } else {
                    m += _("Moderators have flagged at least 20 comments that we deleted and have a strong ratio of good flags to mistaken flags. Their flags carry five times the weight as standard flags. Users who earn higher levels of the Moderator badge can directly delete comments.")
                }
                break;
            case"expert":
                if (level_aquired) {
                    k = '<span class="' + user_type + '_title title">';
                    k += sprintf(_("%s is a Politics Pundit!"), e) + "</span>";
                    m += sprintf(_("%s has earned a Politics Pundit Badge. HuffPost Pundits are our most engaged and thought-provoking commenters. Pundit Badges are awarded based on a strong history of insightful comments."), e)
                }
                break;
            case"predictor":
                if (level_aquired) {
                    switch (user_level) {
                    case 1:
                        k = '<span class="' + user_type + '_title title">';
                        k += sprintf(_("%s is a Predictor!"), e) + "</span>";
                        m += sprintf(_("%s has an amazing track record predicting the future on HuffPost's Predict the News -- play now!"), e);
                        break;
                    case 11:
                        k = '<span class="' + user_type + '_title title">';
                        k += sprintf(_("%s is a Level 1 Olympics Predictor!"), e) + "</span>";
                        break;
                    case 12:
                        k = '<span class="' + user_type + '_title title">';
                        k += sprintf(_("%s is a Level 2 Olympics Predictor!"), e) + "</span>";
                        break;
                    case 13:
                        k = '<span class="' + user_type + '_title title">';
                        k += sprintf(_("%s is a Level 3 Olympics Predictor!"), e) + "</span>";
                        break;
                    case 14:
                        k = '<span class="' + user_type + '_title title">';
                        k += sprintf(_("%s is a Level 1 Olympics Predictor Winner!"), e) + "</span>";
                        break;
                    case 15:
                        k = '<span class="' + user_type + '_title title">';
                        k += sprintf(_("%s is a Level 2 Olympics Predictor Winner!"), e) + "</span>";
                        break;
                    case 16:
                        k = '<span class="' + user_type + '_title title">';
                        k += sprintf(_("%s is a Level 3 Olympics Predictor Winner!"), e) + "</span>";
                        break
                    }
                }
                break;
            case"pledge":
                if (level_aquired) {
                    switch (user_level) {
                    case 1:
                        k = '<span class="' + user_type + '_title title">';
                        k += sprintf(_("%s has taken the Third World America pledge!"), e) + "</span>";
                        m += sprintf(_("%s pledges to rebuild the American Dream and prevent a Third World America."), e);
                        break
                    }
                }
                break;
            case"sponsor":
                k = '<span class="sponsor_badge_title title">' + _("Sponsor Badge") + "</span>";
                m += '<span class="arial_16 bold">' + _("What is this?") + "</span><br/>" + sprintf(_("This comments thread is brought to you by <span class='sponsor_name'>%s</span>."), sponsor_display_name) + " <br/><br/>" + _("We offer our advertisers the opportunity to facilitate meaningful discussion around issues that are relevant to them and our community.<br/><br/>All sponsored comments are clearly marked as such and solely reflect the views of the advertiser. If you have feedback about sponsored comments, please <a href='#' class='sponsor_link'>contact us</a>.");
                break;
            case"betatester":
                k = '<span class="' + user_type + '_title title">';
                k += sprintf(_("%s is a Beta Tester!"), e) + "</span>";
                m += e + " ... " + _("Need a correction");
                break;
            case"blogger":
                k = '<span class="' + user_type + '_title title">';
                k += sprintf(_("%s has been a HuffPost Blogger since %s."), e, author_created_on) + "</span>";
                if (author_nickname) {
                    m += sprintf(_("Click %shere%s to see all of %s's articles."), '<a target="_hplink" class="' + user_type + '_link" href="http://www.huffingtonpost.com/' + author_nickname + '">', "</a>", e)
                }
                break;
            case"staff":
                k = '<span class="' + user_type + '_title title">';
                k += sprintf(_("%s has been a HuffPost employee since %s."), e, author_created_on) + "</span>";
                if (author_nickname) {
                    m += sprintf(_("Click %shere%s to see all of %s's articles."), '<a target="_hplink" class="' + user_type + '_link" href="http://www.huffingtonpost.com/' + author_nickname + '">', "</a>", e)
                }
                break;
            case"crimesolver":
                if (level_aquired) {
                    if (user_level <= 12) {
                        k = '<span class="crimesolver_badge_title title">';
                        k += sprintf(_("%s is a Level %s Crime Solver!"), e, user_level) + "</span>";
                        if ("undefined" != typeof(badge_promo_ad) && badge_promo_ad["crime-solver"]) {
                            var l = badge_promo_ad["crime-solver"];
                            m = k + "<br/><br/>";
                            if (l.commercial_text != "") {
                                m += l.commercial_text + "<br/><br/>"
                            }
                            k = '</td><td align="right" width="100%">' + l.custom_html
                        }
                        m += sprintf(_("%s has taken one or more Crime quizzes and earned the Crime Solver badge."), e);
                        m += "<br/><br/>" + sprintf(_("Find crime quizzes to win this badge %shere%s."), '<a href="/news/tnt-quizzes" class="bold underline">', "</a>")
                    }
                }
                break
            }
            if (gift_name.length > 0) {
                k += sprintf(_("'s comments have received a total of <b>%s</b> %s gift badges."), gift_points, gift_name) + "</span>"
            }
            if (!hide_learn_more) {
                m += '<br/><br/><a target="_hplink" class="' + user_type + '_link" href="http://www.huffingtonpost.com/p/frequently-asked-question.html#socialnews">' + __("Learn more about these badges") + "</a>"
            }
            i.html('<table><tr><td width="64px"><div class="badge_sprite float_left big_badge_' + user_type + user_level + '"></div></td><td valign="middle">' + k + "</td></tr></table>");
            g.innerHTML = m;
            var b = jQuery(badge_el).offset();
            var h = b.left - parseInt(jQuery(".badge_popup_corner.bottom").css("margin-left")) + jQuery(badge_el).width() / 2;
            var f = b.top - jQuery(j).height();
            if (jQuery(window).scrollTop() < f) {
                jQuery(".badge_popup_corner.bottom").show().css({
                    visibility: "visible"
                });
                jQuery(".badge_popup_corner.top").hide().css({
                    visibility: "hidden"
                })
            } else {
                f = b.top + jQuery(badge_el).height();
                jQuery(".badge_popup_corner.top").show().css({
                    visibility: "visible"
                });
                jQuery(".badge_popup_corner.bottom").hide().css({
                    visibility: "hidden"
                })
            }
            j.style.display = "block";
            var a = jQuery(j);
            a.offset({
                left: h,
                top: f
            });
            HuffPoUtil.bind(a, "mouseover", this.onMouseOverBadgePopup, null, this);
            HuffPoUtil.bind(a, "mouseout", this.onMouseOutBadgePopup, null, this)
        },
        getActivePopupDiv: function() {
            var a = document.getElementById("badge_popup");
            if (!a || (a.style.display == "none")) {
                return null
            }
            return a
        },
        hideBadgePopup: function() {
            UserLevels.BadgePopup.badge_popup_timer_id = 0;
            var a = this.getActivePopupDiv();
            if (!a) {
                return 
            }
            jQuery(a).unbind();
            a.style.display = "none";
            jQuery(a).offset({
                left: - 1000,
                top: - 1000
            })
        },
        cancelBadgePopupHiding: function() {
            if (this.badge_popup_timer_id) {
                clearTimeout(this.badge_popup_timer_id);
                this.badge_popup_timer_id = 0
            }
        },
        onMouseOutBadge: function() {
            this.cancelBadgePopupHiding();
            this.badge_popup_timer_id = setTimeout("UserLevels.BadgePopup.hideBadgePopup()", this.timeout)
        },
        onMouseOverBadgePopup: function(a) {
            this.cancelBadgePopupHiding()
        },
        onMouseOutBadgePopup: function(a) {
            this.cancelBadgePopupHiding();
            this.badge_popup_timer_id = setTimeout("UserLevels.BadgePopup.hideBadgePopup()", this.timeout)
        },
        onBadgeClick: function(a) {
            new_window = window.open(_("http://www.huffingtonpost.com/p/frequently-asked-question.html#socialnews"), "");
            new_window.focus()
        }
    },
    GiftBadgePopup: {
        timeout: 100,
        gift_badge_popup_timer_id: 0,
        badge_allowance: 0,
        badge_types: [],
        inited: false,
        setTypes: function(a, b) {
            this.badge_allowance = a;
            this.badge_types = b;
            if (this.badge_types.length == 0) {
                this.badge_allowance = 0
            }
        },
        init: function() {
            if (!this.inited) {
                try {
                    var a = jQuery.parseJSON(jQuery("#gift_badge_allowance").attr("allowance"));
                    var c = jQuery.parseJSON(jQuery("#gift_badge_types").attr("types"));
                    this.setTypes(a, c);
                    this.inited = true
                } catch (b) {
                    this.setTypes(0, [])
                }
            }
        },
        showBadgePopup: function(c) {
            this.init();
            this.leftside_popup = c.leftside_popup || false;
            if (HPConfig.ipad_application) {
                this.leftside_popup = true
            }
            is_own_comment = c.is_own_comment || false;
            badge_el = c.badge_el || false;
            comment_icons_id = c.comment_icons_id || false;
            user_name = c.user_name || false;
            user_id = c.user_id || 0;
            entry_id = c.entry_id || 0;
            comment_id = c.comment_id || 0;
            current_user_name = c.current_user_name || "";
            current_user_id = c.current_user_id || 0;
            vertical = c.vertical || "";
            if (is_own_comment || UserLevels.GiftBadgePopup.badge_allowance == 0) {
                return 
            }
            if (this.gift_badge_popup_timer_id) {
                clearTimeout(this.gift_badge_popup_timer_id);
                this.gift_badge_popup_timer_id = 0
            }
            var k = document.getElementById("gift_badge_popup");
            var f = document.getElementById("gift_badge_popup_text");
            var j = document.getElementById("gift_badge_popup_top");
            if (!k ||!badge_el) {
                return 
            }
            var l = "";
            l += "<p>" + sprintf(_("You can give <b>%s</b> the following badges"), user_name) + ":</p>";
            common_params = "'" + comment_icons_id + "'," + current_user_id + ",'" + vertical + "'," + comment_id + "," + entry_id + "," + user_id + ",'" + user_name + "'";
            var m = "";
            m += '<div class="gift_badge_icons">';
            var h = UserLevels.GiftBadgePopup.badge_types.length;
            for (var d = 0; d < h; d++) {
                m += '<div class="gift_badge" onclick="UserLevels.GiftBadgePopup.giftBadgeAction(' + common_params + "," + d + ');return false;">';
                m += '<div class="badge_sprite badge_' + UserLevels.GiftBadgePopup.badge_types[d]["badge_name"].toLowerCase() + '0"></div>';
                m += UserLevels.GiftBadgePopup.badge_types[d]["display_name"];
                m += "</div>"
            }
            m += "</div>";
            l += m;
            if (UserLevels.GiftBadgePopup.badge_allowance == 1) {
                l += "<p>" + _("You can only give one badge a day, so give it to the best comment you see.") + "</p>"
            } else {
                l += "<p>" + sprintf(_("You have %s badges left to give, so give them to the best comments you see."), UserLevels.GiftBadgePopup.badge_allowance) + "</p>"
            }
            l += '<p class="gift_badge_learn_more"><a target="_hplink" href="http://www.huffingtonpost.com/macgregor-thomson/gift-badges_b_927585.html">' + _("Learn more about these badges") + "</a></p>";
            f.innerHTML = l;
            badge_el_offset_x = (this.leftside_popup ? jQuery(badge_el).width() + 5 : 5);
            badge_el_offset_y = (this.leftside_popup ? 0 : jQuery(k).height() + 10);
            var b = jQuery(badge_el).offset();
            var g = b.left + badge_el_offset_x;
            var e = b.top - badge_el_offset_y;
            k.style.display = "block";
            var a = jQuery(k);
            a.offset({
                left: g,
                top: e
            });
            HuffPoUtil.bind(a, "mouseover", this.onMouseOverBadgePopup, null, this);
            HuffPoUtil.bind(a, "mouseout", this.onMouseOutBadgePopup, null, this)
        },
        getActivePopupDiv: function() {
            var a = document.getElementById("gift_badge_popup");
            if (!a || (a.style.display == "none")) {
                return null
            }
            return a
        },
        hideBadgePopup: function() {
            UserLevels.GiftBadgePopup.timer_id = 0;
            var a = this.getActivePopupDiv();
            if (!a) {
                return 
            }
            jQuery(a).unbind();
            a.style.display = "none";
            jQuery(a).offset({
                left: - 1000,
                top: - 1000
            })
        },
        cancelBadgePopupHiding: function() {
            if (this.gift_badge_popup_timer_id) {
                clearTimeout(this.gift_badge_popup_timer_id);
                this.gift_badge_popup_timer_id = 0
            }
        },
        onMouseOutBadge: function() {
            this.cancelBadgePopupHiding();
            this.gift_badge_popup_timer_id = setTimeout("UserLevels.GiftBadgePopup.hideBadgePopup()", this.timeout)
        },
        onMouseOverBadgePopup: function(a) {
            this.cancelBadgePopupHiding()
        },
        onMouseOutBadgePopup: function(a) {
            this.cancelBadgePopupHiding();
            this.gift_badge_popup_timer_id = setTimeout("UserLevels.GiftBadgePopup.hideBadgePopup()", this.timeout)
        },
        onBadgeClick: function(a) {
            new_window = window.open("http://www.huffingtonpost.com/p/frequently-asked-question.html#socialnews", "");
            new_window.focus()
        },
        giftBadgeAction: function(j, b, d, c, g, k, i, e) {
            var a = UserLevels.GiftBadgePopup.badge_types[e]["badge_id"];
            var f = UserLevels.GiftBadgePopup.badge_types[e]["badge_name"].toLowerCase();
            var h = UserLevels.GiftBadgePopup.badge_types[e]["display_name"];
            if (typeof Comments !== "undefined") {
                Comments.trackEvent("GiftBadgeAction")
            }
            jQuery.ajax({
                url: "/commentsv3/giftBadgeAction.php?badge_id=" + a + "&entry_id=" + g + "&comment_id=" + c + "&user_id=" + b + "&comment_user_id=" + k,
                success: function(q) {
                    var p = document.getElementById("gift_badge_popup_text");
                    result = parseInt(q);
                    if (isNaN(result)) {
                        return 
                    }
                    if (result < 0) {
                        p.innerHTML = _("Error occurred while trying to award gift badge, please try again later.")
                    } else {
                        if (result == 0) {
                            p.innerHTML = sprintf(_("You have already awarded the %s badge to this comment."), h)
                        } else {
                            UserLevels.GiftBadgePopup.badge_allowance--;
                            var m = document.getElementById(j);
                            var t = m ? m.innerHTML: "";
                            if (m && t.indexOf(f) < 0) {
                                newInnerHTML = '<div class="badge_sprite badge_' + f + '0" ';
                                newInnerHTML += 'onmouseout="UserLevels.BadgePopup.onMouseOutBadge();" ';
                                newInnerHTML += 'onmouseover="UserLevels.BadgePopup.showBadgePopup({ ';
                                newInnerHTML += "badge_el: this, ";
                                newInnerHTML += "user_name: '" + i + "', ";
                                newInnerHTML += "user_type: '" + f + "', ";
                                newInnerHTML += "user_level: 0, ";
                                newInnerHTML += "gift_points: 1, ";
                                newInnerHTML += "gift_name: '" + h + "', ";
                                newInnerHTML += '});" ';
                                newInnerHTML += "onclick=\"UserLevels.BadgePopup.onBadgeClick('" + i + "');\">";
                                newInnerHTML += "</div>";
                                var r = jQuery(m).children(".badge_sprite").length;
                                if (r%4 == 0 && r != 0) {
                                    newInnerHTML = "<br/>" + newInnerHTML
                                }
                                m.innerHTML = m.innerHTML + newInnerHTML
                            }
                            var n = "<p>" + sprintf(_("You have awarded <b>%s</b>'s comment the <b>%s</b> gift badge"), i, h) + "</p>";
                            var o = HuffPrefs;
                            var l = o.get("twitter");
                            var s = o.get("facebook");
                            if (l || s) {
                                n += "<br/><br/><p>" + _("Share your action with your friends on") + ":</p><br/><br/>";
                                if (s) {
                                    n += '<div class="gift-badge-share-button gift-badge-share-fb" onclick="UserLevels.Share.facebook(' + c + ",'" + i + "','" + h + "');return false\"></div>"
                                }
                                if (l) {
                                    n += '<div class="gift-badge-share-button gift-badge-share-tw" onclick="UserLevels.Share.twitter(' + c + ",'" + i + "','" + h + "');return false\"></div>"
                                }
                                n += '<div class="clear"></div>'
                            }
                            p.innerHTML = n
                        }
                    }
                },
                failure: function() {
                    HPError.throwError("giftBadgeAction ajax error")
                }
            })
        }
    },
    Share: {
        facebook: function(b, h, e) {
            var d = sprintf(_("%s's Comment On Huffington Post"), h);
            var f = "";
            var a = sprintf(_("I just gave %s a %s gift badge for their comment on The Huffington Post. Check out their comment. It's awesome!"), h, e);
            var g = "#comment_share_" + b + "_data_link";
            var i = jQuery(g).text();
            var c = jQuery("#comment_" + b).children(".comment_top").children(".comment_top_inner").children(".author_photo").children("img").attr("src");
            if (c.indexOf("facebook.com")!=-1) {
                c = HPConfig.image
            }
            Sharer.Facebook.share({
                facebook_post: {
                    message: a,
                    attachment: {
                        name: d,
                        description: f,
                        href: i,
                        media: [{
                            type: "image",
                            src: c,
                            href: i
                        }
                        ]
                    },
                    callback: function(j) {}
                },
                links: []
            });
            return false
        },
        twitter: function(a, e, d) {
            var c = "#comment_share_" + a + "_data_link";
            var b = jQuery(c).text();
            var f = sprintf(_("I just gave @%s a %s gift badge on @huffingtonpost "), e, d);
            Sharer.Twitter.share({
                tweet_text: f,
                links: [{
                    url: b,
                    make_short: true,
                    insert_type: "add_to_end"
                }
                ],
                on_success_callback: function() {
                    HuffConnect.hideModal();
                    Sharer.Twitter.connect_modal()
                },
                on_success_callback_params: 0,
                on_success_timeout: 1000,
                add_via: false
            });
            return false
        }
    },
    Agreement: {
        participation: false,
        badges_box_enabled: false,
        box_animation: false,
        activate: function(a) {
            if (!HuffCookies.getUserId()) {
                return 
            }
            if (HPConfig.user_badges_participation) {
                this.participation = true
            }
            if (a.slider == true) {
                this.Slider.activate(this)
            }
            if (a.checkbox) {
                this.Checkbox.activate(this, a.checkbox.listen_on_id, a.checkbox.input_id)
            }
        },
        set_participation: function(a) {
            this.participation = a;
            this.update_badges_box()
        },
        flip_participation: function() {
            this.set_participation(!this.get_participation())
        },
        get_participation: function() {
            return this.participation
        },
        update_badges_box: function() {
            var c = document.getElementById("all_user_levels");
            if (!c) {
                return 
            }
            var b = {
                height: 0,
                marginBottom: 0
            };
            if (this.get_participation()) {
                c.style.height = "0px";
                c.style.display = "block";
                var a = 134 + Math.floor(jQuery(".data .item", c).length / 4) * 50;
                b = {
                    height: a,
                    marginBottom: 15
                }
            }
            if (!this.box_animation) {
                jQuery(c).animate(b, {
                    duration: 500,
                    complete: function() {
                        if (parseInt(jQuery(this).css("height")) < 10) {
                            this.style.display = "none"
                        }
                    }
                })
            }
        },
        update_server_side: function() {
            jQuery.ajax({
                url: "/users/user_level_json.php?action=badges_participate&state=" + (this.participation == true ? "true" : "false"),
                cache: false
            })
        },
        Checkbox: {
            A: false,
            activated: false,
            checkbox: false,
            nodeName: false,
            activate: function(a, b, c) {
                this.A = a;
                if (!HuffCookies.getUserId()) {
                    return 
                }
                this.checkbox = document.getElementById(c);
                if (!this.checkbox) {
                    return 
                }
                if (HPBrowser.isIE6() || HPBrowser.isIE7()) {
                    if (!this.A.get_participation()) {
                        this.checkbox.setAttribute("checked", false)
                    } else {
                        this.checkbox.setAttribute("checked", true)
                    }
                } else {
                    if (!this.A.get_participation()) {
                        this.checkbox.checked = false
                    } else {
                        this.checkbox.checked = true
                    }
                }
                HuffPoUtil.bind("#" + b, "click", this.catch_click, null, this);
                this.activated = true
            },
            catch_click: function(a) {
                console.log("click");
                this.nodeName = (a.srcElement || a.target).nodeName.toLowerCase();
                this.A.flip_participation();
                this.Hint.show();
                this.change_state();
                this.A.Slider.change_state();
                this.A.update_server_side()
            },
            change_state: function() {
                if (!this.activated ||!this.checkbox) {
                    return 
                }
                if (this.nodeName != "input") {
                    if (HPBrowser.isIE6() || HPBrowser.isIE7()) {
                        if (!this.A.get_participation()) {
                            this.checkbox.setAttribute("checked", false)
                        } else {
                            this.checkbox.setAttribute("checked", true)
                        }
                    } else {
                        if (!this.A.get_participation()) {
                            this.checkbox.checked = false
                        } else {
                            this.checkbox.checked = true
                        }
                    }
                }
            },
            Hint: {
                activated: false,
                hint: false,
                animation: false,
                animation_attributes: {},
                hint_timer: false,
                activate: function() {
                    if (this.activated) {
                        return 
                    }
                    if (!this.hint) {
                        this.hint = document.getElementById("badge_participating_updated");
                        if (!this.hint) {
                            return 
                        }
                    }
                    this.activated = true
                },
                animate: function() {
                    jQuery(this.hint).animate(this.animation_attributes, {
                        duration: 1500,
                        complete: function() {
                            if (parseInt(jQuery(this).css("opacity")) < 0.5) {
                                this.style.display = "none"
                            }
                        }
                    })
                },
                show: function() {
                    this.activate();
                    if (!this.activated) {
                        return 
                    }
                    this.animation_attributes = {
                        opacity: 1
                    };
                    this.animate();
                    this.hint.style.display = "block";
                    clearTimeout(this.hint_timer);
                    this.hint_timer = setTimeout("UserLevels.Agreement.Checkbox.Hint.hide()", 3000)
                },
                hide: function() {
                    this.activate();
                    if (!this.activated) {
                        return 
                    }
                    clearTimeout(this.hint_timer);
                    this.animation_attributes = {
                        opacity: 0
                    };
                    this.animate()
                }
            }
        },
        Slider: {
            A: false,
            activated: false,
            slider_holder: false,
            amination: false,
            animation_attributes: {},
            activate: function(a) {
                this.A = a;
                this.slider_holder = document.getElementById("badges_participate_slider");
                if (this.A.get_participation()) {
                    this.slider_holder.style.backgroundPosition = "left 0"
                } else {
                    this.slider_holder.style.backgroundPosition = "right 0"
                }
                this.Tip.activate();
                HuffPoUtil.bind("#badges_participate_slider", "click", this.catch_click, null, this);
                document.getElementById("badges_participate_switcher").style.display = "block";
                this.activated = true
            },
            catch_click: function() {
                this.A.flip_participation();
                if (!this.A.get_participation()) {
                    this.Tip.show()
                } else {
                    this.Tip.quickly_hide()
                }
                this.change_state();
                this.A.Checkbox.change_state();
                this.A.update_server_side()
            },
            change_state: function() {
                if (!this.activated ||!this.slider_holder) {
                    return 
                }
                if (this.A.get_participation()) {
                    jQuery(this.slider_holder).css("backgroundPosition", "left 0")
                } else {
                    jQuery(this.slider_holder).css("backgroundPosition", "right 0")
                }
            },
            animate: function() {},
            Tip: {
                state: false,
                activated: false,
                tip: false,
                triangle: false,
                close_button: false,
                animation: false,
                tri_animation: false,
                animation_attributes: {},
                tri_animation_attributes: {},
                tip_timer: false,
                activate: function() {
                    if (this.activated) {
                        return 
                    }
                    if (!this.tip) {
                        this.tip = document.getElementById("badges_participate_tip");
                        if (!this.tip) {
                            return 
                        }
                    }
                    if (!this.close_button) {
                        this.close_button = document.getElementById("badges_participate_tip_close");
                        if (!this.close_button) {
                            return 
                        }
                    }
                    if (!this.triangle) {
                        this.triangle = document.getElementById("badges_participate_tip_triangle");
                        if (!this.triangle) {
                            return 
                        }
                    }
                    this.tip.style.display = "none";
                    this.triangle.style.display = "none";
                    HuffPoUtil.bind(this.close_button, "click", this.quickly_hide, null, this);
                    this.activated = true
                },
                animate: function() {
                    jQuery(this.tip).animate(this.animation_attributes, {
                        duration: 1500,
                        complete: function() {
                            if (parseInt(jQuery(this).css("opacity")) < 0.5) {
                                this.style.display = "none"
                            }
                        }
                    });
                    jQuery(this.triangle).animate(this.tri_animation_attributes, {
                        duration: 1500,
                        complete: function() {
                            if (parseInt(jQuery(this).css("opacity")) < 0.5) {
                                this.style.display = "none"
                            }
                        }
                    })
                },
                show: function() {
                    this.activate();
                    if (!this.activated) {
                        return 
                    }
                    jQuery(this.tip).css("display", "block").css("opacity", "0");
                    jQuery(this.triangle).css("display", "block").css("opacity", "0");
                    this.animation_attributes = {
                        opacity: 1
                    };
                    this.tri_animation_attributes = {
                        opacity: 1
                    };
                    this.animate();
                    clearTimeout(this.tip_timer);
                    this.tip_timer = setTimeout("UserLevels.Agreement.Slider.Tip.hide()", 5000)
                },
                quickly_hide: function() {
                    this.tip.style.display = "none";
                    this.triangle.style.display = "none"
                },
                hide: function() {
                    this.activate();
                    if (!this.activated) {
                        return 
                    }
                    clearTimeout(this.tip_timer);
                    this.animation_attributes = {
                        opacity: 0
                    };
                    this.tri_animation_attributes = {
                        opacity: 0
                    };
                    this.animate()
                }
            }
        }
    },
    update_beta_tester_mode: function() {
        jQuery.ajax({
            url: "/users/user_level_json.php?action=badges_betatester&state=" + (isBT ? "false" : "true"),
            cache: false,
            dataType: "html",
            success: function(a, b) {
                isBT = (a === "true") ? true : false
            }
        })
    }
};
var AwardCommentBadge = {
    badges: "",
    giftBadges: "",
    ieResized: {},
    init: function() {
        this.fansystem();
        this.pagination();
        this.bindEventsToGiftBadges();
        this.ieFix(1);
        jQuery("#more_badges").live("click", function() {
            if (jQuery("div.comments-badges-single-badge-hidden:visible").length) {
                jQuery("div.comments-badges-single-badge-hidden").hide(200);
                jQuery("#more_badges").text("more..")
            } else {
                jQuery("div.comments-badges-single-badge-hidden").show(200);
                jQuery("#more_badges").text("less..")
            }
        })
    },
    favComment: function(c, a, b) {
        var d = document.getElementById("best_" + c);
        if (d && (d.innerHTML.indexOf("Marked")==-1) && (d.innerHTML.indexOf("Favorited")==-1)) {
            this.trackEvent("Favorite");
            jQuery.ajax({
                url: "/include/flagComment.php?type=best&blog_id=" + b + "&cmt_id=" + c + "&entry_id=" + a,
                success: function(e) {
                    result = parseInt(e);
                    if (isNaN(result)) {
                        return 
                    }
                    if (result < 0) {
                        d.innerHTML = "Already Favorited (" + ( - result) + ")"
                    } else {
                        d.innerHTML = "Favorited (" + result + ")"
                    }
                },
                failure: function() {}
            })
        }
    },
    fansystem: function() {
        huff.use("fansystem", function(a) {
            jQuery(document).delegate(".become_fan_link", "click", function(b) {
                b.preventDefault();
                if (!jQuery(this).length) {
                    return 
                }
                var c = jQuery.trim(jQuery(this).attr("id").replace("become_fan_of_", "")) || "";
                a.becomeFan(c);
                return 
            });
            jQuery(document).delegate(".remove_fan_link", "click", function(d) {
                d.preventDefault();
                if (!jQuery(this).length) {
                    return 
                }
                var b = jQuery.trim(jQuery(this).attr("id").replace("remove_fan_of_", "")) || "";
                var c = b.split("_");
                a.removeFan(c[0], c[1]);
                return 
            })
        })
    },
    pagination: function() {
        if (!jQuery("div.comments-badges-arrow-right") ||!jQuery("div.comments-badges-arrow-left")) {
            return false
        }
        jQuery("div.comments-badges-arrow-right").live("click", function() {
            var a = jQuery("div.comments-badges-tread").children("div[class!='clear']:visible");
            var c = a.next("div[class!='clear']");
            var b = jQuery("div.comments-badges-pages-counter").children().first();
            if (!c.length) {
                var c = jQuery("div.comments-badges-tread").children("div[class!='clear']").first();
                b.text(1)
            } else {
                b.text(b.text() * 1 + 1)
            }
            a.hide();
            c.show();
            AwardCommentBadge.ieFix(b.text() * 1)
        });
        jQuery("div.comments-badges-arrow-left").live("click", function(c) {
            var a = jQuery("div.comments-badges-tread").children("div[class!='clear']:visible");
            var b = a.prev("div[class!='clear']");
            var d = jQuery("div.comments-badges-pages-counter").children().first();
            if (!b.length) {
                var b = jQuery("div.comments-badges-tread").children("div[class!='clear']").last();
                d.text(jQuery("div.comments-badges-pages-counter").children().last().text() * 1)
            } else {
                d.text(d.text() * 1 - 1)
            }
            a.hide();
            b.show();
            AwardCommentBadge.ieFix(d.text() * 1)
        })
    },
    bindEventsToGiftBadges: function() {
        if (!jQuery("a.share_favorite_comment_popup_trigger")) {
            return false
        }
        jQuery.each(jQuery("a.share_favorite_comment_popup_trigger"), function(a, d) {
            var b = jQuery(d);
            if (AwardCommentBadge.giftBadges[b.attr("comment_id")]&&!b.attr("isGiftBadgeBinded")) {
                var c = AwardCommentBadge.giftBadges[b.attr("comment_id")];
                b.bind("click", function() {
                    UserLevels.BadgePopup.onBadgeClick(c.comment_user_name)
                });
                b.bind("mouseout", function() {
                    UserLevels.GiftBadgePopup.onMouseOutBadge()
                });
                b.bind("mouseover", function() {
                    c.badge_el = b;
                    UserLevels.GiftBadgePopup.showBadgePopup(c)
                });
                b.attr("isGiftBadgeBinded", true)
            }
        })
    },
    ieFix: function(b) {
        if (!this.ieResized[b] && jQuery.browser.version <= 7 && jQuery.browser.msie) {
            this.ieResized[b] = true;
            var a = (jQuery(".comments-badges-tread:visible").height() / 2) - (jQuery(".comments-badges-right-block:visible").height() / 2) - 12;
            jQuery(".comments-badges-right-block:visible").css("margin-top", a + "px");
            jQuery(".comments-badges-monster-badge:visible").css("margin-top", a + "px")
        }
    }
};
window._ || (window._ = function(a) {
    return a
});
window.ngettext || (window.ngettext = function(a) {
    return a
});
window.sprintf || (window.sprintf = function(a) {
    return a
});
var Sharer = {
    check_incomplete_sessions: function() {
        this.Twitter.check_incomplete_session()
    },
    Facebook: {
        links: [],
        facebook_post: {
            message: "",
            attachment: {
                name: "",
                description: "",
                href: ""
            },
            action_links: [{
                text: _("Join HuffPost Social News now!"),
                href: HuffPoUtil.getHostName() + "/social/?r=" + escape(HuffCookies.getUserGuid())
            }
            ],
            target_id: null,
            user_message_prompt: null,
            callback: null,
            auto_publish: false,
            actor_id: null
        },
        share: function(b) {
            for (var a in b.facebook_post) {
                if (b.facebook_post.hasOwnProperty(a)) {
                    this.facebook_post[a] = b.facebook_post[a]
                }
            }
            if (b.links) {
                Sharer.Links.process(b.links, Sharer.Facebook.real_share, Sharer.Facebook);
                return false
            }
            this.real_share()
        },
        real_share: function() {
            HPFB.ensureInit(function() {
                HPFB.waitForSession(function() {
                    if (Sharer.Facebook.facebook_post.attachment.description) {
                        var a = Sharer.Facebook.facebook_post.attachment.description;
                        a = Sharer.Links.expand_text_with_links(a);
                        Sharer.Facebook.facebook_post.attachment.description = a
                    }
                    HPFB.streamPublish(Sharer.Facebook.facebook_post.message, Sharer.Facebook.facebook_post.attachment, Sharer.Facebook.facebook_post.action_links, Sharer.Facebook.facebook_post.target_id, Sharer.Facebook.facebook_post.user_message_prompt, Sharer.Facebook.facebook_post.callback, Sharer.Facebook.facebook_post.auto_publish, Sharer.Facebook.facebook_post.actor_id)
                })
            })
        }
    },
    Twitter: {
        hp_cookie_continue: "hp_sharer_twitter_sharing",
        hp_cookie_continue_obj: "hp_sharer_twitter_share_text",
        twitter_messg_limit: 140,
        modal_element_ids: {},
        oauthInfo: {},
        user_supply_vars: ["tweet_text", "commercial_info", "links", "callbacks", "badge_holder_id", "entry_url", "entry_short_url", "add_via", "on_success_callback", "on_success_callback_params", "on_success_timeout"],
        tweet_text: "",
        links: [],
        add_via: true,
        commercial_info: false,
        callbacks: false,
        badge_holder_id: null,
        entry_url: null,
        entry_short_url: false,
        check_incomplete_session: function() {
            if (HuffCookies.getCookie(this.hp_cookie_continue)) {
                this.load_last_share_session();
                HuffCookies.destroyCookie(this.hp_cookie_continue);
                if (HuffCookies.getUserId() && HuffPrefs.get("twitter")) {
                    this.check_credentials()
                }
            }
        },
        load_last_share_session: function() {
            var d = HuffCookies.getCookie(this.hp_cookie_continue_obj);
            if (!d) {
                return 
            }
            var b = JSON.parse(d);
            if (!b) {
                return 
            }
            var c, a;
            for (a = 0; a < this.user_supply_vars.length; a++) {
                c = this.user_supply_vars[a];
                if (b[c]) {
                    this[c] = b[c]
                }
            }
            HuffCookies.destroyCookie(this.hp_cookie_continue_obj)
        },
        save_session: function() {
            HuffCookies.setCookie(this.hp_cookie_continue, 1, 2);
            var a = {}, c, b;
            for (b = 0; b < this.user_supply_vars.length; b++) {
                c = this.user_supply_vars[b];
                if (this[c]) {
                    a[c] = this[c]
                }
            }
            HuffCookies.setCookie(this.hp_cookie_continue_obj, JSON.stringify(a), 2)
        },
        get_theme_descr: function(c) {
            var b = {
                ids: {},
                html: ""
            };
            if (c == "basic_twitter" || c == "basic_twitter_frontpage") {
                SNProject.initModal("tweet");
                b.ids = {
                    error_msg: c + "_modal_window_error_msg",
                    comm_text: c + "_modal_window_comm_text",
                    comm_checkbox: c + "_modal_window_share_checkbox",
                    textarea: c + "_modal_window_text_area",
                    chars_left_title: c + "_modal_window_chars_left",
                    close_button: c + "_modal_window_close_button",
                    submit_button: c + "_modal_window_submit_button",
                    submit_button_hider: c + "_modal_window_submit_button_hider",
                    loader_icon: c + "_modal_window_loader_icon",
                    status_text: c + "_modal_window_status_text",
                    comm_block: c + "_modal_window_comm_block"
                };
                var a = this.oauthInfo.twitter_pic || HuffCookies.getCurrentUserPhotoUrl();
                b.html = "						<div id='" + b.ids.chars_left_title + "' class='counter_container float_right'></div>                            <div class='clear'></div>						    <div class='avatar'" + (a ? " style='background: url(\"" + a + "\") no-repeat scroll 0pt 0pt transparent;'></div>" : "></div>") + "<div class='textarea_container'><textarea id='" + b.ids.textarea + "' class='textarea'></textarea></div>						<div style='clear: both;'></div>						<div id='" + b.ids.comm_block + "' style='display: none;'>							<div class='checkbox_container'>							<input id='" + b.ids.comm_checkbox + "' type='checkbox'>							</div>							<div id='" + b.ids.comm_text + "' class='checkbox_label_container'></div>							<div style='clear: both;'></div>						</div>						<div class='submit_container'>							<div id='" + b.ids.loader_icon + "' class='loader_icon'></div>							<div id='" + b.ids.error_msg + "' class='error_msg'></div>							<div id='" + b.ids.submit_button + "' class='submit_button'>&nbsp;</div>							<div id='" + b.ids.status_text + "' class='status_text'></div>						</div>"
            }
            return b
        },
        share: function(d) {
            var b, a;
            var c = this.user_supply_vars.length;
            for (a = 0; a < c; a++) {
                b = this.user_supply_vars[a];
                if (d.hasOwnProperty(b)) {
                    this[b] = d[b]
                }
            }
            if (this.callbacks !== false) {
                if (!this.callbacks.scope) {
                    this.callbacks.scope = null
                }
                if (!this.callbacks.timeout) {
                    this.callbacks.timeout = 25
                }
                if (!this.callbacks.params) {
                    this.callbacks.params = null
                }
                if (!this.callbacks.success) {
                    this.callbacks.success = null
                }
                if (!this.callbacks.failure) {
                    this.callbacks.failure = null
                }
            }
            Sharer.Twitter.check_credentials()
        },
        check_credentials: function() {
            if ((HuffCookies.getUserId() && HuffPrefs.get("twitter")) || this.oauthInfo.token) {
                if (this.links) {
                    Sharer.Links.process(this.links, Sharer.Twitter.show_modal_window, Sharer.Twitter);
                    return false
                }
                this.show_modal_window()
            } else {
                linkSocialAccount.checkLoginStatus("twitter", false, {
                    disable_connect: true,
                    callback: function() {
                        var a = Sharer.Twitter;
                        if (Provider.user_provider_info.twitter.oauth_token && Provider.user_provider_info.twitter.oauth_token_secret) {
                            a.oauthInfo = {
                                token: Provider.user_provider_info.twitter.oauth_token,
                                secret: Provider.user_provider_info.twitter.oauth_token_secret,
                                twitter_pic: Provider.user_provider_info.twitter.profile_image || "http://s.huffpost.com/images/profile/user_placeholder.gif"
                            };
                            a.check_credentials()
                        } else {
                            a.connect_modal()
                        }
                        return false
                    }
                })
            }
        },
        connect_modal: function() {
            if (HPConfig.ipad_application === false) {
                if (HuffCookies.getUserId()&&!HuffPrefs.get("twitter")) {
                    Badges.prototype.userFunc_twitter_show_promo_modal_window(this.badge_holder_id)
                } else {
                    if (HuffCookies.getUserId() == null) {
                        setTimeout(function() {
                            ConnectOverview.showBenefits({
                                provider: "twitter"
                            })
                        }, 200)
                    }
                }
            }
            return 
        },
        show_modal_window: function() {
            var a = this.twitter_messg_limit;
            if (this.commercial_info) {
                a -= (this.commercial_info.hash.length + 1)
            }
            this.tweet_text = Sharer.Links.expand_text_with_links(this.tweet_text, a);
            if (this.commercial_info) {
                this.tweet_text = this.tweet_text + " " + this.commercial_info.hash
            }
            var b = "";
            if (HuffPoUtil.GetEntryID(location.href)) {
                b = "basic_twitter"
            } else {
                b = "basic_twitter_frontpage"
            }
            var d = this.get_theme_descr(b);
            var c = this;
            huff.use("modal", function(e) {
                e.show({
                    theme: "tweet_modal",
                    content: d.html,
                    onclose: Sharer.Twitter.connect_modal,
                    onshow: function() {
                        c.modal_element_ids = d.ids;
                        c.setTwitterModal()
                    }
                })
            });
            return 
        },
        setTwitterModal: function() {
            var c = HuffPoUtil.getNode("#" + this.modal_element_ids.textarea);
            if (c) {
                c.value = this.tweet_text;
                if (this.add_via) {
                    c.value += " via " + HPConfig.twitter_signature
                }
                HuffPoUtil.bind(c, "keyup", this.key_up_handler, null, this);
                HuffPoUtil.bind(c, "change", this.key_up_handler, null, this);
                HuffPoUtil.bind(c, "keydown", this.key_up_handler, null, this);
                HuffPoUtil.bind(c, "keypress", this.key_up_handler, null, this)
            }
            this.key_up_handler();
            var b = HuffPoUtil.getNode("#" + this.modal_element_ids.submit_button);
            if (b) {
                HuffPoUtil.bind(b, "click", this.submit_handler, null, this);
                if (typeof(HPBrowser) !== "undefined") {
                    if (HPBrowser.isIE6() || HPBrowser.isIE7()) {
                        jQuery(b).css("left", "220px")
                    }
                }
            }
            if (this.commercial_info) {
                jQuery(this.modal_element_ids.comm_block).css("display", "block");
                var a = HuffPoUtil.getNode("#" + this.modal_element_ids.comm_text);
                if (a) {
                    a.innerHTML = this.commercial_info.text;
                    var d = HuffPoUtil.getNode("#" + this.modal_element_ids.comm_checkbox);
                    d.checked = true;
                    HuffPoUtil.bind(a, "click", this.commercial_checkbox_handler, null, this);
                    HuffPoUtil.bind(d, "click", this.commercial_checkbox_handler, null, this)
                }
            }
        },
        commercial_checkbox_handler: function(d) {
            var a = HuffPoUtil.getNode("#" + this.modal_element_ids.textarea);
            var b = HuffPoUtil.getNode("#" + this.modal_element_ids.comm_checkbox);
            var c = this.commercial_info.hash;
            if (b.disabled) {
                return 
            }
            var f = (d.srcElement || d.target).nodeName.toLowerCase();
            if (f != "input") {
                if (HPBrowser.isIE6() || HPBrowser.isIE7()) {
                    if (b.checked) {
                        b.setAttribute("checked", false)
                    } else {
                        b.setAttribute("checked", true)
                    }
                } else {
                    if (b.checked) {
                        b.checked = false
                    } else {
                        b.checked = true
                    }
                }
            }
            if (b.checked) {
                a.value += " " + c
            } else {
                a.value = a.value.replace(" " + c, "")
            }
        },
        key_up_handler: function() {
            var d = this.twitter_messg_limit;
            var b = HuffPoUtil.getNode("#" + this.modal_element_ids.textarea);
            var c = HuffPoUtil.getNode("#" + this.modal_element_ids.chars_left_title);
            var a = d - b.value.length;
            c.innerHTML = a;
            c.style.color = a < 0 ? "#D40D12" : "#999999"
        },
        submit_handler: function() {
            var c = HuffPoUtil.getNode("#" + this.modal_element_ids.textarea);
            var f = HuffPoUtil.getNode("#" + this.modal_element_ids.error_msg);
            var e = c.value;
            if (e == "") {
                f.innerHTML = _("Please enter some text!");
                c.focus();
                return false
            }
            if (e.length > this.twitter_messg_limit) {
                f.innerHTML = _("Tweet limit exceeded!");
                c.focus();
                return false
            }
            this.show_sibmitting_proccess();
            var d = "tweet=" + encodeURIComponent(e);
            if (this.oauthInfo.token) {
                d += "&user_ot=" + this.oauthInfo.token + "&user_os=" + this.oauthInfo.secret
            }
            var b = this.modal_element_ids;
            var a = {
                onLinkSuccess: function() {
                    Sharer.Twitter.oauthInfo = {
                        token: Provider.user_provider_info.twitter.oauth_token,
                        secret: Provider.user_provider_info.twitter.oauth_token_secret
                    };
                    Sharer.Twitter.no_popup = true;
                    Sharer.Twitter.modal_element_ids = b;
                    Sharer.Twitter.submit_handler();
                    return false
                }
            };
            d += "&eid=" + HPUtil.GetEntryID(window.location.href);
            d += "&banners=" + HPAds.get_banners_str();
            jQuery.ajax({
                type: "POST",
                url: "/badge/post_to_twitter.php",
                data: d,
                dataType: "json",
                error: function(g) {
                    Sharer.Twitter.show_failure_final();
                    HPError.e()
                },
                success: function(g) {
                    if (g.status && g.message) {
                        if (/success/.test(g.status)) {
                            Sharer.Twitter.show_successfull_final()
                        } else {
                            if (/duplicate/.test(g.message)) {
                                Sharer.Twitter.show_duplicate_final()
                            } else {
                                if (/invalid_token/.test(g.message)) {} else {
                                    Sharer.Twitter.show_failure_final()
                                }
                            }
                        }
                        return 
                    }
                }
            });
            return false
        },
        show_sibmitting_proccess: function() {
            var b = HuffPoUtil.getNode("#" + this.modal_element_ids.submit_button);
            var c = HuffPoUtil.getNode("#" + this.modal_element_ids.loader_icon);
            var a = HuffPoUtil.getNode("#" + this.modal_element_ids.textarea);
            var d = HuffPoUtil.getNode("#" + this.modal_element_ids.comm_checkbox);
            var e = HuffPoUtil.getNode("#" + this.modal_element_ids.status_text);
            if (HPBrowser.isIE6() || HPBrowser.isIE7()) {
                c.style.display = "block"
            }
            e.innerHTML = "";
            HuffPoUtil.getNode("#" + this.modal_element_ids.error_msg).innerHTML = "";
            b.style.display = "none";
            c.style.display = "block";
            a.disabled = true;
            if (d) {
                d.disabled = true
            }
        },
        show_successfull_final: function() {
            var a = HuffPoUtil.getNode("#" + this.modal_element_ids.loader_icon);
            var b = HuffPoUtil.getNode("#" + this.modal_element_ids.status_text);
            if (HPBrowser.isIE6() || HPBrowser.isIE7()) {
                a.style.display = "none"
            }
            a.style.display = "none";
            b.innerHTML = _("Posted to Twitter!");
            b.style.display = "block";
            if (typeof this.on_success_callback == "function") {
                setTimeout(function() {
                    Sharer.Twitter.on_success_callback(Sharer.Twitter.on_success_callback_params)
                }, Sharer.Twitter.on_success_timeout)
            } else {
                HuffConnect.hideModal();
                setTimeout(function() {
                    Sharer.Twitter.connect_modal()
                }, 1000)
            }
        },
        show_duplicate_final: function() {
            var b = HuffPoUtil.getNode("#" + this.modal_element_ids.submit_button);
            var a = HuffPoUtil.getNode("#" + this.modal_element_ids.textarea);
            var d = HuffPoUtil.getNode("#" + this.modal_element_ids.comm_checkbox);
            var c = HuffPoUtil.getNode("#" + this.modal_element_ids.loader_icon);
            var e = HuffPoUtil.getNode("#" + this.modal_element_ids.status_text);
            if (HPBrowser.isIE6() || HPBrowser.isIE7()) {
                c.style.display = "none"
            }
            c.style.display = "none";
            e.innerHTML = _("You already shared this Tweet");
            e.style.display = "block";
            b.style.display = "block";
            a.disabled = false;
            if (d) {
                d.disabled = false
            }
        },
        show_failure_final: function() {
            var a = HuffPoUtil.getNode("#" + this.modal_element_ids.loader_icon);
            var b = HuffPoUtil.getNode("#" + this.modal_element_ids.status_text);
            if (HPBrowser.isIE6() || HPBrowser.isIE7()) {
                a.style.display = "none"
            }
            a.style.display = "none";
            b.innerHTML = _("Unable to process your request!");
            b.style.display = "block"
        }
    },
    Links: {
        links: [],
        not_completed: true,
        need_to_short: 0,
        shorten: 0,
        check_for_complete: function() {
            if (this.shorten == this.need_to_short) {
                this.not_completed = false;
                return true
            }
            return false
        },
        bitly_shorten_complete: function(d) {
            var j = "", a, e, h, b, c, f = "", g;
            for (a in d.results) {
                b = d.results[a];
                c = a;
                break
            }
            for (h in b) {
                j += h + ":" + b[h].toString() + "\n";
                if (h == "shortUrl") {
                    f = b[h].toString()
                }
            }
            for (e = 0; e < Sharer.Links.links.length; e++) {
                g = Sharer.Links.links[e];
                if (unescape(g.url) == c) {
                    g.long_url = unescape(g.url);
                    g.url = f;
                    Sharer.Links.shorten++;
                    break
                }
            }
        },
        process: function(c, a, g) {
            var f, b, e, d, h;
            if (c) {
                this.links = c
            }
            for (f = 0; f < this.links.length; f++) {
                b = this.links[f];
                if (b.make_short && b.url) {
                    d = "http://api.bit.ly/shorten?version=2.0.1&longUrl=" + ((b.url == unescape(b.url)) ? escape(b.url) : b.url) + "&login=" + HPConfig.bit_ly_key.user_name + "&apiKey=" + HPConfig.bit_ly_key.user_key + "&callback=Sharer.Links.bitly_shorten_complete";
                    h = document.createElement("script");
                    h.setAttribute("type", "text/javascript");
                    document.getElementsByTagName("head")[0].appendChild(h);
                    h.setAttribute("src", d);
                    this.need_to_short++
                }
            }
            this.wait_for_completed(a, g, 10)
        },
        wait_for_completed: function(a, c, b) {
            if (this.check_for_complete()) {
                a.call(c);
                return 
            }
            var d = this;
            setTimeout(function() {
                d.wait_for_completed(a, c, b)
            }, b)
        },
        expand_text_with_links: function(e, c) {
            var b = e, d, a;
            if (!e) {
                return ""
            }
            c = c || false;
            for (d = 0; d < this.links.length; d++) {
                a = this.links[d];
                if (a.insert_type == "add_to_end") {
                    if (c && (b.length > (c - (a.url.length + 1)))) {
                        b = b.substr(0, (c - (a.url.length + 1)) - 3) + "..."
                    }
                    b = b + " " + a.url
                } else {
                    if (a.insert_type == "replace_mock") {
                        b = b.replace(a.mock, a.url);
                        if (c && (b.length > c)) {
                            b = b.substr(0, c - 3) + "..."
                        }
                    } else {
                        if (a.insert_type == "add_to_begining") {
                            b = a.url + " " + b;
                            if (c && (b.length > c)) {
                                b = b.substr(0, c - 3) + "..."
                            }
                        }
                    }
                }
            }
            return b
        }
    },
    trackedShares: [],
    Digg: {
        digg_it: function(a, f) {
            var b = "http://digg.com/submit?phase=2&title=" + a + "&url=" + (f ? escape(f) : escape(window.location));
            window.open(b, "digg_window", "width=650,height=520,status=yes,location=no,left=0,top=0");
            try {
                if ( - 1 === jQuery.inArray("digg", Badges.trackedShares)) {
                    var c = f ? f: window.location.href;
                    jQuery.post("/include/social_track.php", {
                        t: "digg",
                        entry: HPUtil.GetEntryID(c),
                        banners: HPAds.get_banners_str()
                    });
                    Sharer.trackedShares.push("digg");
                    bN.set("dL_shareType", "digg", 1);
                    bN.set("dL_shareURL", escape(c), 1);
                    bN.ping("aol-share")
                }
            } catch (d) {}
        }
    },
    Linkedin: {
        UserData: null,
        AccessTokens: null,
        popup_count: 0,
        popups: new Array(),
        share: function(b) {
            this.entry_id = b.entry_id || null;
            this.share_url = b.url || null;
            this.title = b.title || null;
            this.summary = b.summary || null;
            if (!this.entry_id ||!this.share_url) {
                return false
            }
            this.update_linkedin_data(this.entry_id);
            if (this.popup_count > 0) {
                this.popups[this.popup_count - 1].close()
            }
            if ((HuffCookies.getUserId() && HuffPrefs.get("linkedin")) || (this.AccessTokens && this.AccessTokens.oauth_token)) {
                var a = "http://www.linkedin.com/shareArticle?mini=true&source=The%20Huffington%20Post&url=" + this.share_url + "&title=" + this.title + "&summary=" + this.summary;
                this.popups[this.popup_count++] = window.open(a, "_blank", PopupManager.getWindowParams(520, 570))
            } else {
                var a = HuffPoUtil.getHostName() + "/users/linkedin/index.php?action=auth&step=1&url=" + this.share_url + "&title=" + this.title + "&summary=" + this.summary;
                this.popups[this.popup_count++] = window.open(a, "_blank", PopupManager.getWindowParams(750, 325))
            }
            var c = this;
            window.onbeforeunload = function() {
                c.update_linkedin_data(c.entry_id);
                c.popups[c.popup_count - 1].close()
            }
        },
        update_linkedin_data: function(a) {
            if (!(this.UserData || this.AccessTokens) && QuickLogin.LinkedinData) {
                this.UserData = JSON.parse(QuickLogin.LinkedinData.UserData);
                this.AccessTokens = JSON.parse(QuickLogin.LinkedinData.AccessTokens);
                Badges.trackShare({
                    eid: a,
                    badge: "linkedin"
                })
            }
        }
    }
};
Sharer.hp_amazing_video_module_fb_share = function(d, b, c, a, e) {
    Sharer.Facebook.share({
        facebook_post: {
            message: d,
            attachment: {
                name: b,
                description: c,
                href: a,
                media: [{
                    type: "image",
                    src: e,
                    href: a
                }
                ]
            },
            callback: function() {}
        },
        links: []
    })
};
jQuery(function() {
    Sharer.check_incomplete_sessions()
});
window._ || (window._ = function(a) {
    return a
});
window.ngettext || (window.ngettext = function(a) {
    return a
});
window.sprintf || (window.sprintf = function(a) {
    return a
});
var HuffConnect = {
    loadConnectModal: function(a) {
        HPConnect.Login();
        return 
    },
    SignUp: {
        Facebook: function(a) {
            setTimeout(function() {
                HPFB.ensureInit(function() {
                    HPFB.waitForSession(function() {
                        HPFB.getFBInfo(function(b) {
                            Provider.user_provider_info.facebook = {};
                            Provider.user_provider_info.facebook.profile_image = b[0].pic_square;
                            Provider.user_provider_info.facebook.screen_name = b[0].name;
                            Provider.user_provider_info.facebook.id = b[0].uid;
                            Provider.user_provider_info.facebook.email = b[0].email || "";
                            if (/proxymail.facebook.com/.test(Provider.user_provider_info.facebook.email)) {
                                Provider.user_provider_info.facebook.email = ""
                            }
                            Provider.user_provider_info.facebook.Status = a;
                            setTimeout(function() {
                                huff.use("connect/auth", "connect/modal", "connect/provider", function(d, e, c) {
                                    c.loadModal("facebook")
                                })
                            }, 500);
                            return 
                        })
                    }, {
                        fb_signup: true
                    })
                })
            }, 1000)
        }
    },
    Login: {
        is_become_fan: false,
        callbacks: {},
        onLoginSuccess: function() {
            huff.use("connect/auth", function(a) {
                a.Login.updateLoginStatus()
            })
        },
        FacebookSuccess: function(a, d) {
            if (HPUtil.isMobileweb() && window.Mobile && window.Mobile.Auth) {
                var c = d && d.autologin;
                window.Mobile.Auth.provider_callback("facebook", a.msg, c);
                return 
            }
            document.cookie = document.cookie;
            var e = HuffCookies;
            if (d && d.autologin) {
                e.setCookie("autologin", "1", 2)
            }
            switch (a.msg) {
            case"success":
                var b = function() {
                    e.destroyCookie("autologin");
                    e.set("snn_popup_needed", "1", "1");
                    var f = window.QuickLogin || false;
                    if (f && "function" == typeof f.FacebookLoginCallback) {
                        HuffConnect.Login.registerCallback("after_facebook_login", f.FacebookLoginCallback, "facebook")
                    }
                    HuffConnect.Login.callback("after_facebook_login");
                    return 
                };
                if (d && d.autologin) {
                    b()
                } else {
                    huff.use("conf", function(f) {
                        f.get("auth_method", function(g) {
                            g = g || "frame";
                            if ("direct" === g&&!HPUtil.isDotComVersion()) {
                                huff.use("util/user", function(h) {
                                    h.doFacebookAuth();
                                    return 
                                })
                            } else {
                                b()
                            }
                        })
                    })
                }
                break;
            case"self_closed_user":
            case"new_user":
                if (linkSocialAccount.linkOptions.disable_connect === true) {
                    linkSocialAccount.linkOptions.callback("facebook", "new_user")
                } else {
                    HuffConnect.SignUp.Facebook(a.msg)
                }
                break;
            case"permanent_removed":
            case"unapprove_removed":
            case"permanent_banned_removed":
                HuffConnect.SignUp.Facebook(a.msg);
                break;
            case"prompt_for_connect":
                HuffConnect.hideModal();
                linkSocialAccount.provider = "facebook";
                linkSocialAccount.FacebookPromtLogin();
                break;
            default:
                this.FacebookFail(a);
                break
            }
            return 
        },
        FacebookFail: function(b) {
            var a = HuffConnect.Login;
            if (b.argument && b.argument.autologin) {
                HuffCookies.setCookie("autologin", "1", 2)
            }
            if (b.tld != null) {
                HPError.d("ServiceLoginFail", b)
            } else {
                if (b.is_error) {
                    HPError.d("ServiceLoginFail2", b)
                }
            }
            if (typeof(a.is_become_fan) != "undefined") {
                a.is_become_fan = false
            }
            return 
        },
        registerCallback: function(a, c, b) {
            if ("undefined" == typeof(b)) {
                b = false
            }
            if (a && "function" == typeof c) {
                this.callbacks[a] = {
                    Func: c,
                    ProviderName: b
                }
            }
            return 
        },
        callback: function(a) {
            if (a && "undefined" != typeof this.callbacks[a] && "function" == typeof this.callbacks[a].Func) {
                if (!HPUtil.isDotComVersion()) {
                    window.XDAuth && XDAuth.syncRemoteWithLocal("login", [{
                        args: [],
                        on_success: HuffConnect.Login.callbacks[a].Func
                    }
                    ])
                } else {
                    this.callbacks[a].Func()
                }
                return 
            } else {
                if (!HPUtil.isDotComVersion()) {
                    var b = function() {
                        HuffConnect.hideModal();
                        HuffConnect.Login.onLoginSuccess();
                        return 
                    };
                    if ("after_facebook_login" == a) {
                        window.XDAuth && XDAuth.syncRemoteWithLocal("login", [{
                            args: [],
                            on_success: b
                        }
                        ])
                    } else {
                        window.XDAuth && XDAuth.syncRemoteWithLocal("login", [{
                            args: [],
                            on_success: b
                        }
                        ])
                    }
                } else {
                    HuffConnect.hideModal();
                    this.onLoginSuccess()
                }
                return 
            }
        }
    },
    SocialLogout: function(c, a, b) {
        if ("undefined" != typeof b) {
            b = false
        }
        if (a !== false) {
            a = true
        }
        if (!c) {
            c = ""
        }
        SNProject.track(HuffCookies.getUserId(), "user_log_out");
        huff.use("provider/logout", function(d) {
            d.Main(c, a, b)
        });
        return 
    },
    loader_active: false,
    socialModal: function(c) {
        if (HuffConnect.loader_active) {
            HuffConnect.loader_active = false;
            HuffConnect.hideModal()
        }
        c = c || {};
        if (c.modal_head_image || "" == c.modal_head_image) {
            c.modal_head_image = c.modal_head_image
        } else {
            c.modal_head_image = '<img width="457" height="40" alt="" src="' + __("/images/social-profile/lightbox/huffpo_logo_lightbox.png") + '"/>';
            if (HPConfig && HPConfig.site) {
                switch (HPConfig.site) {
                case"stylelist":
                    c.modal_head_image = '<img class="margin_5_0" height="42" alt="" src="' + __("/images/social-profile/lightbox/stylelist_logo_lightbox.png") + '"/>';
                    break;
                case"france":
                    c.modal_head_image = '<img width="500" height="40" alt="" src="' + __("/images/v/france/social-profile/lightbox/social-news-lightbox.png") + '"/>';
                    break;
                case"canada-quebec":
                    c.modal_head_image = '<img width="500" height="40" alt="" src="' + __("/images/v/canada-quebec/social-profile/lightbox/social-news-lightbox.png") + '"/>';
                    break;
                case"spain":
                case"voces":
                    c.modal_head_image = '<img width="500" height="50" alt="" src="' + __("/images/v/spain/social-profile/lightbox/social-news-lightbox.png") + '"/>';
                    break;
                case"italy":
                    c.modal_head_image = '<img width="500" height="50" alt="" src="' + __("/images/v/italy/social-profile/lightbox/social-news-lightbox.png") + '"/>';
                    break;
                case"brazil":
                    c.modal_head_image = '<img width="500" height="50" alt="" src="' + __("/images/social-profile/hpsn-title.png") + '"/>';
                    break
                }
            }
        }
        if (c.modal_subhead_image || "" == c.modal_subhead_image) {
            c.modal_subhead_image = c.modal_subhead_image
        } else {
            c.modal_subhead_image = '<img height="22" alt="" src="' + __("/images/social/benefits/social_logo_inner.png") + '"/>';
            if (HPConfig && HPConfig.site) {
                switch (HPConfig.site) {
                case"canada-quebec":
                case"stylelist":
                case"spain":
                case"voces":
                case"italy":
                case"brazil":
                    c.modal_subhead_image = "";
                    break
                }
            }
        }
        c.inner_wrapper_class = (c.inner_wrapper_class) ? c.inner_wrapper_class : "";
        c.width = (c.width) ? c.width : 680;
        c.onshow = ("function" == typeof c.onshow) ? c.onshow : false;
        c.onclose = ("function" == typeof c.onclose) ? c.onclose : false;
        c.inner_html = c.inner_html || "";
        c.url = c.url || "";
        c.type = c.type || "GET";
        c.cache = (c.cache === false) ? false : true;
        c.position = c.position || "absolute";
        c.content = "";
        var a = "";
        var b = "";
        if ((c.modal_head_image || c.modal_subhead_image) && HPConfig.current_vertical_id != 43) {
            a += '<div class="hp_blue_lb_head_image">' + c.modal_head_image + "</div>";
            if ("france" != HPConfig.site && HPConfig.current_vertical_name != "canada-quebec") {
                a += '<div class="hp_blue_lb_head_subimage margin_top_5">' + c.modal_subhead_image + "</div>"
            }
            a += '<div class="hp_blue_lb_hr"></div>'
        }
        b += '<div><div class="' + c.inner_wrapper_class + '">';
        b += a;
        b += '<div class="social_modal_content">' + c.inner_html + "</div>";
        b += "</div></div>";
        c.content += b;
        if (!c.inner_html && c.url) {
            if (HuffConnect.socialModalCache && HuffConnect.socialModalCache[c.url]) {
                c.inner_html = HuffConnect.socialModalCache[c.url];
                delete (c.url);
                HuffConnect.socialModal(c);
                return 
            } else {
                c.content = '<div style="width:32px; height:32px; background:url(http://s.huffpost.com/images/loader.gif);margin:33px auto;"></div>';
                jQuery.ajax({
                    url: c.url,
                    type: c.type,
                    success: function(f, e, d) {
                        HuffConnect.hideModal();
                        if (d.status != 200) {
                            return 
                        }
                        if (c.cache) {
                            HuffConnect.socialModalCache = {};
                            HuffConnect.socialModalCache[c.url] = f
                        }
                        c.inner_html = f;
                        delete (c.url);
                        HuffConnect.socialModal(c);
                        return 
                    },
                    error: function() {
                        HPError.e();
                        return 
                    }
                })
            }
        }
        huff.use("modal", function(d) {
            var e = {
                content: c.content,
                width: (c.url) ? 300: c.width,
                onshow: (!c.url) ? c.onshow: false,
                onclose: (!c.url) ? c.onclose: false,
                position: (!c.url) ? c.position: "fixed"
            };
            d.show(e)
        });
        return 
    },
    hideModal: function(a) {
        a = a || false;
        huff.use("modal", function(b) {
            b.hide();
            if ("function" == typeof(a)) {
                a()
            }
            return 
        })
    },
    showLoader: function() {
        huff.use("modal", function(a) {
            a.show({
                content: '<div style="width:32px; height:32px;margin:33px auto;"><img src="http://s.huffpost.com/images/loader.gif"></div>',
                width: 300
            });
            return 
        });
        HuffConnect.loader_active = true;
        return 
    },
    showMask: function() {
        jQuery('<div id="hp_modal_mask" style="position:fixed; z-index: 800; left:0px; top:0px;width:100%; height:100%; background:#000; opacity:0.7; filter:alpha(opacity=70);"></div>').prependTo("body");
        return 
    },
    hideMask: function() {
        jQuery("#hp_modal_mask").remove();
        return 
    },
    ShowEmbed: function() {
        this.ChangeEmbedState("visible", selector)
    },
    HideEmbed: function() {
        this.ChangeEmbedState("hidden", selector)
    },
    ChangeEmbedState: function(g, b) {
        if (typeof b == "undefined") {
            var f = document.getElementsByTagName("object");
            var e = document.getElementsByTagName("embed");
            for (var d = 0; d < e.length; d++) {
                e[d].style.visibility = g;
                if (e[d].getAttribute("belongs_modal")) {
                    e[d].style.display = "visible" == g ? "block" : "none"
                }
            }
        } else {
            f = jQuery(b)
        }
        for (d = 0; d < f.length; d++) {
            f[d].style.visibility = g;
            if (f[d].getAttribute("belongs_modal")) {
                f[d].style.display = "visible" == g ? "block" : "none"
            }
        }
        var c = ["curtainunit", "ad_leaderboard_top", "ad_store_leaderboard_top"];
        for (d = 0; d < c.length; d++) {
            var a = jQuery("#" + c[d]);
            if (a.length) {
                a.css("visibility", g);
                if (a.attr("belongs_modal")) {
                    a.css("display", "visible" == g ? "block" : "none")
                }
            }
        }
    }
};
var ConnectOverview = {
    config: null,
    current_page: null,
    interval_id: null,
    auto_paging_timeout_id: null,
    showBenefits: function(c) {
        HuffConnect.hideModal();
        var a = c.provider || null;
        var b = "";
        this.configureModal(a);
        this.clearIntervals();
        return 
    },
    configureModal: function(e, c) {
        var d = _("/images/social/benefits/") + e + "/";
        huff.use("user", "connect/auth", "connect/modal");
        switch (e) {
        case"facebook":
            var a = {
                type: "facebook",
                text: _("Connect to the Huffington Post through Facebook"),
                slides: [d + "facebook_benefit1.jpg", d + "facebook_benefit2.jpg", d + "facebook_benefit3.jpg"]
            };
            break;
        case"twitter":
            var a = {
                type: "twitter",
                text: _("Connect to the Huffington Post through Twitter"),
                slides: [d + "twitter_benefit1.jpg", d + "twitter_benefit2.jpg", d + "twitter_benefit3.jpg", d + "twitter_benefit4.jpg"]
            };
            break;
        default:
            HPConnect.Login();
            return;
            break
        }
        return;
        if (a) {
            this.config = a;
            var b = this.getHtml(c);
            this.showLightbox(b)
        }
        return 
    },
    showLightbox: function(c) {
        c = '<div class="social_benefit_modal">' + c + "</div>";
        var b = function() {
            ConnectOverview.auto_paging_timeout_id = setTimeout(function() {
                ConnectOverview.startAutoPaging()
            }, 5000)
        };
        var d = false;
        var a = "fixed";
        HuffConnect.socialModal({
            inner_html: c,
            width: 680,
            onclose: d,
            onshow: b,
            position: a
        });
        return 
    },
    closeLightbox: function() {
        HuffConnect.hideModal()
    },
    showPage: function(e) {
        for (var b = 0, d = this.config.slides.length; b < d; b++) {
            var c = jQuery("#benefit_page_" + b);
            var a = jQuery("#benefit_slide_" + b);
            if (b == e) {
                a.removeClass("display_none");
                c.removeClass("active_page_link").addClass("passive_page_link").unbind("click");
                this.current_page = e
            } else {
                a.addClass("display_none");
                c.removeClass("passive_page_link").addClass("active_page_link").unbind("click").click({
                    page: b
                }, function(f) {
                    ConnectOverview.showPage(f.data.page)
                })
            }
        }
        return 
    },
    prevPage: function() {
        var a = this.current_page - 1;
        if (a < 0) {
            a = (this.config.slides.length - 1)
        }
        this.showPage(a)
    },
    nextPage: function() {
        var a = this.current_page + 1;
        if (a >= this.config.slides.length) {
            a = 0
        }
        this.showPage(a)
    },
    autoPaging: function() {
        this.nextPage();
        return 
    },
    getHtml: function(j) {
        var b = this.config;
        var e = "";
        var f = "";
        var c = "";
        var a = 1;
        var h = _("/images/social/benefits/") + b.type + "_button_big.png";
        c += '<div class="float_left benefit_bg benft_left_arr" onclick="ConnectOverview.prevPage();"></div>';
        for (var d = 0; d < b.slides.length; d++) {
            var g = (d === 0) ? "": "display_none";
            c += '<div class="' + g + ' single_slide float_left" id="benefit_slide_' + d + '"><img id="benefit_img_src_' + d + '" src="' + b.slides[d] + '" /></div>';
            f += '<span class="benefit_pages ' + (d == 0 ? "passive" : "active") + '_page_link" id="benefit_page_' + d + '">' + a + "</span>";
            this.current_page = d;
            a++
        }
        c += '<div class="float_left benefit_bg benft_right_arr" onclick="ConnectOverview.nextPage();"></div>';
        c += '<div class="clear" style="clear:both;"></div>';
        e += '<div class="social_benefits social_benefits_lightbox" id="benefits_lb">';
        e += '<div class="social_link_text center">' + b.text + "</div>";
        e += '<div class="center"><a href="javascript:void(0);" onclick="linkSocialAccount.checkLoginStatus(\'' + b.type + '\')"><img src="' + h + '" /></a></div>';
        e += '<div class="benefit_paging" style="text-align:right">' + f + '<div class="clear"></div></div>';
        e += '<div class="benefit_slides" id="slideshow_div">' + c + "</div>";
        e += '<div class="clear"></div><br class="clear" />';
        e += "</div>";
        return e
    },
    startAutoPaging: function() {
        this.interval_id = self.setInterval(function() {
            ConnectOverview.autoPaging()
        }, 5000)
    },
    clearIntervals: function(b) {
        var a = b || this;
        if (a.auto_paging_timeout_id) {
            clearTimeout(a.auto_paging_timeout_id)
        }
        if (a.interval_id) {
            clearInterval(a.interval_id)
        }
    },
    showQuickModal: function(e) {
        var c = e.charAt(0).toUpperCase() + e.substr(1, e.length - 1);
        var b = "";
        var a = "";
        var d = "";
        switch (e) {
        case"facebook":
            a = _("/images/fb-large.gif");
            d = '<span class="fb-large">' + _("Sign in with Facebook") + "</span>";
            break;
        case"twitter":
            a = _("/images/Sign-in-with-Twitter-darker.png");
            d = '<span class="twitter-darker">' + _("Connect with Twitter") + "</span>";
            break;
        default:
            return 
        }
        b += '<div style="font-size: 14pt; font-weight: bold">' + sprintf(_("Do you want to connect to HuffPost using %s"), c) + "?</div>";
        b += '<a class="login_light_link" ';
        b += 'href="/social/join.html?autojoin=1" ';
        b += "onclick=\"linkSocialAccount.checkLoginStatus('" + e + "'); return false;\"";
        b += 'target="_blank">';
        b += d;
        b += "</a>";
        HuffConnect.socialModal({
            inner_html: b,
            modal_subhead_image: ""
        });
        return 
    }
};
BingPageModule = {
    app_id: "6AE728F21E27915214035B81C99B54592DE4D651",
    offset: 0,
    limit: {
        Video: 4,
        Image: 3,
        Web: 2
    },
    count_image: 3,
    count_web: 2,
    search_type: "",
    search_text: "",
    bing_container: "",
    search_url: "",
    web_url: "",
    videos_url: "",
    images_url: "",
    hp_url: "",
    atdm_imp_pixel: "",
    imr_imp_pixel: "",
    bing_df_tag: "",
    interclick_url: "http://idcs.interclick.com/Segment.aspx?sid=697a8088-0d41-4f72-9fb6-5bc7ecc099e8",
    version: "",
    ord: "",
    track_click: function() {
        HPTrack.trackPageview("/t/a/bing/2/out/" + escape(BingPageModule.search_text));
        jQuery.ajax({
            type: "post",
            url: "/ajax/bing/clicks_tracking.php",
            data: "keyword=" + escape(this.bingText.value) + "&vertical_id=" + HPConfig.current_vertical_id
        })
    },
    entry_extra_sort_bing: "",
    bing_metrics_bing_qtr_click_video: "",
    bingText: null,
    bingButton: null,
    bing_logo: null,
    bing_hp: null,
    bing_search_script: null,
    bing_item_web: null,
    bing_item_video: null,
    bing_item_image: null,
    entries: null,
    a_bing_videos: null,
    a_bing_web: null,
    a_bing_images: null,
    head: null
};
BingPageModule.setBingListeners = function() {
    E.removeListener(BingPageModule.bingButton);
    E.removeListener(BingPageModule.bing_logo);
    E.removeListener(BingPageModule.see_more);
    E.removeListener(BingPageModule.bing_hp);
    E.addListener(BingPageModule.bing_hp, "click", function() {
        this.track_click();
        window.open(BingPageModule.hp_url)
    }, this, true);
    E.addListener(BingPageModule.bing_logo, "click", function() {
        this.track_click();
        window.open(this.search_url + "?q=" + escape(BingPageModule.bingText.value))
    }, this, true);
    E.addListener(BingPageModule.bingButton, "click", function() {
        this.track_click();
        BingPageModule.bingSearch();
        window.open(this.search_url + "?q=" + escape(BingPageModule.bingText.value))
    }, this, true);
    E.addListener(BingPageModule.see_more, "click", function() {
        switch (BingPageModule.search_type) {
        case"Video":
            imr = BingPageModule.imr_videos_url;
            s_url = this.videos_url + "?q=" + escape(BingPageModule.bingText.value);
            break;
        case"Image":
            imr = BingPageModule.imr_images_url;
            s_url = this.images_url + "?q=" + escape(BingPageModule.bingText.value);
            break;
        case"Web":
            imr = BingPageModule.imr_web_url;
            s_url = this.web_url + "?q=" + escape(BingPageModule.bingText.value);
            break
        }
        this.track_click();
        window.open(s_url)
    }, this, true)
};
BingPageModule.bingStart = function() {
    BingPageModule.bing_container = $("bing_container");
    BingPageModule.bingText = $("sb_form_q");
    BingPageModule.bingButton = $("sb_form_go");
    BingPageModule.bing_logo = $("bing_logo");
    BingPageModule.bing_search_script = $("bing_search_script");
    BingPageModule.bing_item_web = $("bingItemWeb");
    BingPageModule.bing_item_video = $("bingItemVideo");
    BingPageModule.bing_item_image = $("bingItemImage");
    BingPageModule.entries = $("entries");
    BingPageModule.a_bing_videos = $("a_bing_videos");
    BingPageModule.a_bing_web = $("a_bing_web");
    BingPageModule.a_bing_images = $("a_bing_images");
    BingPageModule.head = document.getElementsByTagName("head");
    BingPageModule.search_type = "Video";
    var b = Math.random() * 100;
    var a = new Date();
    b4_june_30_2011 = a.getMonth() < 6 && a.getYear() < 2012;
    if (b4_june_30_2011 && b <= 80) {
        BingPageModule.initVersion();
        this.bingSearch();
        HPTrack.trackPageview("/t/a/bing/2/impression/" + escape(this.search_text))
    }
};
BingPageModule.initVersion = function() {
    BingPageModule.version = Math.random() * 100;
    if (BingPageModule.version < 50) {
        BingPageModule.version = "v0";
        BingPageModule.search_url = BingPageModule.search_url.v0 + BingPageModule.ord;
        BingPageModule.web_url = BingPageModule.web_url.v0 + BingPageModule.ord;
        BingPageModule.videos_url = BingPageModule.videos_url.v0 + BingPageModule.ord;
        BingPageModule.images_url = BingPageModule.images_url.v0 + BingPageModule.ord;
        BingPageModule.hp_url = BingPageModule.hp_url.v0 + BingPageModule.ord;
        BingPageModule.bing_metrics_bing_qtr_click_video = BingPageModule.bing_metrics_bing_qtr_click_video.v0 + BingPageModule.ord;
        BingPageModule.atdm_imp_pixel = BingPageModule.atdm_imp_pixel.v0;
        BingPageModule.imr_imp_pixel = BingPageModule.imr_imp_pixel.v0;
        document.getElementById("bing_bottom").innerHTML = '<div style="padding: 12px 0px 0px 445px;"><a id="see_more" href="javascript: void();" ><img id="see_more_image" border="0" src="/images/bing_new/see_more_v.gif" alt="See more" /></a></div>'
    } else {
        if ((BingPageModule.version >= 50) && (BingPageModule.version < 75)) {
            BingPageModule.version = "v1";
            BingPageModule.search_url = BingPageModule.search_url.v1 + BingPageModule.ord;
            BingPageModule.web_url = BingPageModule.web_url.v1 + BingPageModule.ord;
            BingPageModule.videos_url = BingPageModule.videos_url.v1 + BingPageModule.ord;
            BingPageModule.images_url = BingPageModule.images_url.v1 + BingPageModule.ord;
            BingPageModule.hp_url = BingPageModule.hp_url.v1 + BingPageModule.ord;
            BingPageModule.bing_metrics_bing_qtr_click_video = BingPageModule.bing_metrics_bing_qtr_click_video.v1 + BingPageModule.ord;
            BingPageModule.atdm_imp_pixel = BingPageModule.atdm_imp_pixel.v1;
            BingPageModule.imr_imp_pixel = BingPageModule.imr_imp_pixel.v1;
            var a = document.createElement("a");
            a.id = "bing_hp";
            a.style.cursor = "pointer";
            a.href = "javascript: void();";
            a.innerHTML = '<div class="HomepageTop"></div>';
            document.getElementById("bing_header").appendChild(a);
            document.getElementById("bing_bottom").innerHTML = '<div style="padding: 12px 0px 0px 445px;"><a id="see_more" href="javascript: void();" ><img id="see_more_image" border="0" src="/images/bing_new/see_more_v.gif" alt="See more" /></a></div>'
        } else {
            BingPageModule.version = "v2";
            BingPageModule.search_url = BingPageModule.search_url.v2 + BingPageModule.ord;
            BingPageModule.web_url = BingPageModule.web_url.v2 + BingPageModule.ord;
            BingPageModule.videos_url = BingPageModule.videos_url.v2 + BingPageModule.ord;
            BingPageModule.images_url = BingPageModule.images_url.v2 + BingPageModule.ord;
            BingPageModule.hp_url = BingPageModule.hp_url.v2 + BingPageModule.ord;
            BingPageModule.bing_metrics_bing_qtr_click_video = BingPageModule.bing_metrics_bing_qtr_click_video.v2 + BingPageModule.ord;
            BingPageModule.atdm_imp_pixel = BingPageModule.atdm_imp_pixel.v2;
            BingPageModule.imr_imp_pixel = BingPageModule.imr_imp_pixel.v2;
            document.getElementById("bing_bottom").innerHTML = '<div style="padding: 6px 0px 0px 340px;">                                                                <a  id="bing_hp" style="cursor: pointer;" href="javascript: void();" ><img border="0" src="/images/bing_new/homepage_bottom.gif" alt="See more" /></a>                                                                <a id="see_more" href="javascript: void();" ><img id="see_more_image" border="0" src="/images/bing_new/se_more_new.gif" alt="See more" /></a>                                                            </div>'
        }
    }
    BingPageModule.bing_hp = $("bing_hp");
    BingPageModule.see_more = $("see_more");
    BingPageModule.see_more_image = $("see_more_image")
};
BingPageModule.bingSearch = function() {
    $("bingItemWeb").className = "WebPassive";
    $("bingItemVideo").className = "VideoPassive";
    $("bingItemImage").className = "ImagePassive";
    $("bingItem" + BingPageModule.search_type).className = BingPageModule.search_type + "Active";
    BingPageModule.setBingListeners();
    BingPageModule.entries.innerHTML = "<i>Loading...</i>";
    if (BingPageModule.bingText.value == "") {
        BingPageModule.entries.innerHTML = "There is no result..."
    } else {
        BingPageModule.bingText.value = BingPageModule.bingText.value.replace(/\&FORM\=SOAPGN/, "");
        this.search_text = BingPageModule.bingText.value;
        var b = "/images/bing_new/see_more_";
        switch (BingPageModule.search_type) {
        case"Video":
            b = b + "v.gif";
            break;
        case"Image":
            b = b + "i.gif";
            break;
        case"Web":
            b = b + "r.gif";
            break
        }
        if (BingPageModule.version != "v2") {
            BingPageModule.see_more_image.src = b
        }
        var c = "http://api.search.live.net/json.aspx?Appid=" + BingPageModule.app_id + "&Query=" + escape(BingPageModule.bingText.value);
        searchString = c + "&Sources=" + BingPageModule.search_type + "&" + BingPageModule.search_type + ".count=" + BingPageModule.limit[BingPageModule.search_type] + "&" + BingPageModule.search_type + ".offset=" + BingPageModule.offset + "&Version=2.1&" + BingPageModule.search_type + ".SortBy=" + BingPageModule.entry_extra_sort_bing + "&JsonType=callback&SafeSearch=Strict&JsonCallback=BingPageModule.Search" + BingPageModule.search_type + "Completed";
        if (BingPageModule.bing_search_script != null) {
            BingPageModule.head[0].removeChild(BingPageModule.bing_search_script)
        }
        var a = document.createElement("script");
        a.id = "bing_search_script";
        a.src = searchString;
        BingPageModule.head[0].appendChild(a);
        BingPageModule.bing_search_script = document.getElementById("bing_search_script")
    }
};
BingPageModule.TabClick = function(b) {
    var a;
    switch (b) {
    case"Video":
        a = BingPageModule.videos_url + "?q=" + escape(BingPageModule.bingText.value);
        break;
    case"Image":
        a = BingPageModule.images_url + "?q=" + escape(BingPageModule.bingText.value);
        break;
    case"Web":
        a = BingPageModule.web_url + "?q=" + escape(BingPageModule.bingText.value);
        break
    }
    BingPageModule.track_click();
    window.open(a, "bing_web_popup");
    return false
};
BingPageModule.SearchImageCompleted = function(c) {
    resultHTML = "<div>";
    var e = c && c.SearchResponse && c.SearchResponse.Image && c.SearchResponse.Image.Results || [];
    if (!e.length) {
        BingPageModule.entries.innerHTML = "There is no result...";
        return 
    } else {
        BingPageModule.bing_container.style.display = "block";
        if (BingPageModule.atdm_imp_pixel != "") {
            HPUtil.trackerImg(BingPageModule.atdm_imp_pixel, BingPageModule.bing_container);
            BingPageModule.atdm_imp_pixel = ""
        }
        if (BingPageModule.imr_imp_pixel != "") {
            HPUtil.trackerImg(BingPageModule.imr_imp_pixel, BingPageModule.bing_container);
            BingPageModule.imr_imp_pixel = ""
        }
        if (BingPageModule.bing_df_tag != "") {
            HPUtil.scriptTag(BingPageModule.bing_df_tag, BingPageModule.bing_container);
            BingPageModule.bing_df_tag = ""
        }
        if (BingPageModule.interclick_url != "") {
            HPUtil.trackerImg(BingPageModule.interclick_url, BingPageModule.bing_container);
            BingPageModule.interclick_url = ""
        }
    }
    var j = new Array();
    var a = new Array();
    var d = 0;
    for (var f = 0; f < BingPageModule.limit.Image; ++f) {
        j[f] = e[f].Thumbnail.Width;
        a[f] = e[f].Thumbnail.Height;
        d = d + e[f].Thumbnail.Height
    }
    d = Math.ceil(d / BingPageModule.limit.Image);
    var k = 0;
    for (f = 0; f < BingPageModule.limit.Image; ++f) {
        j[f] = Math.ceil(j[f] / (a[f] / d));
        a[f] = d;
        k = k + j[f]
    }
    var b = 495;
    var h = "margin-right: 10px;";
    if (k < (b - 50)) {
        h = "margin-left: 15px; margin-right: 15px;";
        b = 435
    }
    if (k < (b - 100)) {
        h = "margin-left: 25px; margin-right: 25px;";
        b = 375
    }
    var g = b / k;
    for (f = 0; f < BingPageModule.limit.Image; ++f) {
        resultHTML = resultHTML + '<div class="imageItem" style="' + h + '"><a href="' + BingPageModule.images_url + "?q=" + escape(BingPageModule.bingText.value) + '" target="blank" onclick="BingPageModule.track_click(); HPUtil.trackerImg(BingPageModule.imr_images_url, document.body); window.open(this.href, \'bing_image_popup\'); return false;"><img src="' + e[f].Thumbnail.Url + '" border="0" width="' + Math.ceil(j[f] * g) + '" height="' + Math.ceil(a[f] * g) + '" /></a></div>'
    }
    resultHTML = resultHTML + '<div style="clear: both;"></div></div>';
    BingPageModule.entries.innerHTML = resultHTML
};
BingPageModule.SearchWebCompleted = function(a) {
    resultHTML = "";
    var c = a && a.SearchResponse && a.SearchResponse.Web && a.SearchResponse.Web.Results || [];
    if (!c.length) {
        BingPageModule.entries.innerHTML = "There is no result...";
        return 
    } else {
        BingPageModule.bing_container.style.display = "block";
        if (BingPageModule.imp_pixel_common != "") {
            HPUtil.trackerImg(BingPageModule.imp_pixel_common, BingPageModule.bing_container);
            BingPageModule.imp_pixel_common = ""
        }
        if (BingPageModule.imp_pixel != "") {
            HPUtil.trackerImg(BingPageModule.imp_pixel, BingPageModule.bing_container);
            BingPageModule.imp_pixel = ""
        }
        if (BingPageModule.bing_df_tag != "") {
            HPUtil.scriptTag(BingPageModule.bing_df_tag, BingPageModule.bing_container);
            BingPageModule.bing_df_tag = ""
        }
        if (BingPageModule.interclick_url != "") {
            HPUtil.trackerImg(BingPageModule.interclick_url, BingPageModule.bing_container);
            BingPageModule.interclick_url = ""
        }
    }
    for (var b = 0; b < BingPageModule.limit.Web; ++b) {
        if ("undefined" == typeof(c[b])) {
            break
        }
        resultHTML = resultHTML + '<div><a style="text-decoration: underline; font-weight: bold; color: #5462c3;" href="' + BingPageModule.web_url + "?q=" + escape(BingPageModule.bingText.value) + '" target="blank" onclick="BingPageModule.track_click(); HPUtil.trackerImg(BingPageModule.imr_web_url, document.body); window.open(this.href, \'bing_web_popup\'); return false;">' + c[b].Title + '</a><br /><div style="width: 520px; margin-bottom:3px;">' + c[b].Description + '&nbsp</div><span style="color: #688156">' + c[b].DisplayUrl + '</span>&nbsp;<span style="color: #4e7cc0;">-</span>&nbsp;<a style="color: #4e7cc0;" href="' + BingPageModule.web_url + "?q=" + escape(BingPageModule.bingText.value) + '" target="blank" onclick="BingPageModule.track_click(); HPUtil.trackerImg(BingPageModule.imr_web_url, document.body); window.open(this.href, \'bing_web2_popup\'); return false;">Cached page</a><div style="height: 10px;"></div></div>'
    }
    BingPageModule.entries.innerHTML = resultHTML
};
BingPageModule.SearchVideoCompleted = function(a) {
    resultHTML = "<div>";
    var c = a && a.SearchResponse && a.SearchResponse.Video && a.SearchResponse.Video.Results || [];
    if (!c.length) {
        BingPageModule.entries.innerHTML = "There is no result...";
        return 
    } else {
        if (c.length >= BingPageModule.limit.Video) {
            BingPageModule.bing_container.style.display = "block";
            if (BingPageModule.atdm_imp_pixel != "") {
                HPUtil.trackerImg(BingPageModule.atdm_imp_pixel, BingPageModule.bing_container);
                BingPageModule.atdm_imp_pixel = ""
            }
            if (BingPageModule.imr_imp_pixel != "") {
                HPUtil.trackerImg(BingPageModule.imr_imp_pixel, BingPageModule.bing_container);
                BingPageModule.imr_imp_pixel = ""
            }
            if (BingPageModule.bing_df_tag != "") {
                HPUtil.scriptTag(BingPageModule.bing_df_tag, BingPageModule.bing_container);
                BingPageModule.bing_df_tag = ""
            }
            if (BingPageModule.interclick_url != "") {
                HPUtil.trackerImg(BingPageModule.interclick_url, BingPageModule.bing_container);
                BingPageModule.interclick_url = ""
            }
        }
    }
    for (var b = 0; b < BingPageModule.limit.Video; ++b) {
        if ("undefined" == typeof(c[b])) {
            break
        }
        var d = "";
        if (c[b].ClickThroughPageUrl == null) {
            d = c[b].PlayUrl
        } else {
            d = c[b].ClickThroughPageUrl
        }
        d = d.toString().replace(/http:\/\/www\.bing\.com\/videos/gi, BingPageModule.bing_metrics_bing_qtr_click_video);
        d = d.replace(/\&FORM\=SOAPGN/gi, "");
        d = d.replace(/\&scope\=video/gi, "");
        resultHTML = resultHTML + '<div class="videoItem' + ((b == (BingPageModule.count_video - 1)) ? "" : " viRightMargin") + '"><a href="' + d + '" target="blank" onclick="BingPageModule.track_click(); HPUtil.trackerImg(BingPageModule.imr_videos_url, document.body);"><div><div class="bingImage" style="margin-bottom: 3px; width:123px; height: 92px;   background: url(\'' + c[b].StaticThumbnail.Url + '\') no-repeat center black;"><img src="http://s.huffpost.com/images/bing/icon-play.png" style="position: relative; top: 80%; left: 40%" width="16" height="16" alt="Play" /></div></div><div class="bing_video_title">';
        if (c[b].SourceTitle == null) {} else {
            resultHTML = resultHTML + '<div><b style="color:#5462c3;">' + c[b].Title + "</b></div>"
        }
        resultHTML = resultHTML + '<div style="margin-top: 3px;"><b>' + c[b].SourceTitle + "</b></div></div></a></div>"
    }
    resultHTML = resultHTML + '<div style="clear: both;"></div></div>';
    BingPageModule.entries.innerHTML = resultHTML
};
BingPageModule.clearString = function() {
    BingPageModule.search_text = BingPageModule.bingText.value;
    BingPageModule.bingText.value = ""
};
BingPageModule.setString = function() {
    if (BingPageModule.bingText.value == "") {
        BingPageModule.bingText.value = this.search_text
    }
    BingPageModule.setBingListeners()
};
BingPageModule.setStartPage = function(a) {
    if (document.all) {
        a.setHomePage("http://www.bing.com")
    }
    return false
};
var Constant = {
    GET_LISTS_TWEETS_URL: "/ajax/twitter/list_tweets.php",
    GET_TWEETS_URL: "http://search.twitter.com/search.json",
    GET_USER_TIMELINES: "/twitter/get_user_timelines.php",
    FRIENDSHIP_EXISTS_URL: "/twitter/friendship_exists.php",
    RETWEET_TWEET_URL: "/twitter/retweet_tweet.php",
    FOLLOW_LIST_URL: "/twitter/ajax_follow_tlist.php",
    FRIENDSHIP_URL: "/twitter/friendship.php",
    PUBLIC_INFO_URL: "/twitter/public_info.php",
    IS_USER_LISTS_SUBSCRIBER_URL: "/twitter/is_user_lists_subscriber.php"
};
var HPTwitter = {
    lists: [],
    getListsTweets: function(e, g, h, c, d) {
        if (!e ||!e.length) {
            return 
        }
        if (typeof(e) == "string") {
            e = [e]
        }
        if (!g) {
            g = {}
        }
        var a = e.length;
        for (var b = a; b--;) {
            if (!g[e[b]]) {
                g[e[b]] = {}
            }
            g[e[b]].since_id = g[e[b]].since_id || 1
        }
        var f = "lists=" + JSON.stringify(g);
        if (d) {
            f += d
        }
        this._sendRequest("GET", Constant.GET_LISTS_TWEETS_URL, f, h, c)
    },
    followList: function(d, c, a, b) {
        this._sendFollowUnfollowRequest(true, d, c, a, b)
    },
    unfollowList: function(d, c, a, b) {
        this._sendFollowUnfollowRequest(false, d, c, a, b)
    },
    getTweets: function(d, e) {
        d = d || "";
        var b = [];
        if (e) {
            if (e.lang) {
                b.push("lang=" + e.lang)
            }
            if (e.callback) {
                b.push("callback=" + e.callback)
            }
            if (e.locale) {
                b.push("locale=" + e.locale)
            }
            if (e.rpp) {
                b.push("rpp=" + e.rpp)
            }
            if (e.page) {
                b.push("page=" + e.page)
            }
            if (e.since_id) {
                b.push("since_id=" + e.since_id)
            }
            if (e.geocode) {
                b.push("geocode=" + e.geocode)
            }
            if (e.show_user) {
                b.push("show_user=" + e.show_user)
            }
        }
        var c = "?q=" + escape(d);
        if (b.length) {
            c += "&" + b.join("&")
        }
        var a = Constant.GET_TWEETS_URL + c;
        HuffPoUtil.loadAndRun(a)
    },
    retweetTweet: function(a, c, d) {
        if (!a) {
            d();
            return false
        }
        var b = "tweet_id=" + a;
        C.asyncRequest("POST", Constant.RETWEET_TWEET_URL, {
            success: c,
            failure: d
        }, b);
        return true
    },
    favorTweet: function(a, b, c) {
        if (!a) {
            c();
            return false
        }
    },
    getUserTimelines: function(d, e, f, b) {
        d = d || "";
        e = parseInt(e, 10);
        var a = isNaN(e) ? 0: e;
        var c = "screen_name=" + d + "&chunk_size=" + a;
        this._sendRequest("GET", Constant.GET_USER_TIMELINES, c, f, b)
    },
    _sendFollowUnfollowRequest: function(c, f, e, a, b) {
        var d = "list=" + f + "&action=" + (c ? "post" : "delete");
        this._sendRequest("GET", Constant.FOLLOW_LIST_URL, d, e, b, a, false)
    },
    getFollowStatus: function(b, d, a) {
        if (!b) {
            return 
        }
        var c = "user_b=" + b;
        this._sendRequest("GET", Constant.FRIENDSHIP_EXISTS_URL, c, d, a, undefined, false)
    },
    createFriendship: function(d, e, b, a) {
        if (!d) {
            return 
        }
        var c = "id=" + d + "&action=create";
        this._sendRequest("POST", Constant.FRIENDSHIP_URL, c, e, b, a)
    },
    destroyFriendship: function(d, e, b, a) {
        if (!d) {
            return 
        }
        var c = "id=" + d + "&action=destroy";
        this._sendRequest("POST", Constant.FRIENDSHIP_URL, c, e, b, a)
    },
    getPublicInfo: function(d, e, b, a) {
        if (!d) {
            return 
        }
        var c = "id=" + d;
        this._sendRequest("GET", Constant.PUBLIC_INFO_URL, c, e, b, a)
    },
    isUserListsSubscriber: function(a, d, b) {
        if (!a) {
            return 
        }
        if (!YAHOO.lang.isArray(a)) {
            a = [a]
        }
        var c = "lists=" + JSON.stringify(a);
        this._sendRequest("POST", Constant.IS_USER_LISTS_SUBSCRIBER_URL, c, d, b, undefined, false)
    },
    _sendRequest: function(d, a, g, h, c, j, e) {
        if (!d) {
            d = d.toUpperCase()
        }
        var i = c || this;
        var b;
        var f = {
            success: function(n) {
                var k = n.argument[0];
                var m = n.argument[1];
                var l = [n.responseText];
                if (m) {
                    l.push(m)
                }
                if (typeof(k) === "function") {
                    k.apply(i, l)
                }
            },
            argument: [h, j],
            scope: i
        };
        if (d == "POST") {
            b = g
        } else {
            if (d == "GET") {
                a = a + "?" + g;
                if (e === false || e === true) {
                    f.cache = e
                }
            }
        }
        C.asyncRequest(d, a, f, b)
    }
};
HPTwitter.Widgets = {};
(function(a, b) {
    a.Widgets.activateFollowButton = function(d) {
        var c = b("#" + d.node_id + " .container");
        var e = b("#frame-" + d.node_id);
        e.html('<iframe allowtransparency="true" frameborder="0" scrolling="no" src="http://platform.twitter.com/widgets/follow_button.html?' + d.query + '" ' + d.style + "></iframe>");
        e.find("iframe").bind("load", function() {
            b(this).parent().removeClass("hidden");
            c.remove()
        })
    };
    (function(i) {
        if (window.__twitterIntentHandler) {
            return 
        }
        var g = /twitter\.com(\:\d{2,4})?\/intent\/(\w+)/, j = "scrollbars=yes,resizable=yes,toolbar=no,location=yes", h = 550, d = 520, f = screen.height, e = screen.width;
        var c = function(p) {
            var o = b(p.target), k, n, l;
            while (o.get(0) && o.get(0).nodeName.toLowerCase() !== "a") {
                o = o.parent()
            }
            if (o.get(0) && o.get(0).nodeName.toLowerCase() === "a" && o.attr("href") !== undefined) {
                k = o.attr("href").match(g);
                if (k) {
                    n = Math.round((e / 2) - (h / 2));
                    l = 0;
                    if (f > d) {
                        l = Math.round((f / 2) - (d / 2))
                    }
                    window.open(o.attr("href"), "intent", j + ",width=" + h + ",height=" + d + ",left=" + n + ",top=" + l);
                    p.preventDefault()
                }
            }
        };
        i.click(c);
        window.__twitterIntentHandler = true
    })(b(document))
})(HPTwitter, jQuery);
(function(c, j) {
    function k(a) {
        return !c(a).parents().andSelf().filter(function() {
            return c.curCSS(this, "visibility") === "hidden" || c.expr.filters.hidden(this)
        }).length
    }
    c.ui = c.ui || {};
    if (!c.ui.version) {
        c.extend(c.ui, {
            version: "@VERSION",
            keyCode: {
                ALT: 18,
                BACKSPACE: 8,
                CAPS_LOCK: 20,
                COMMA: 188,
                COMMAND: 91,
                COMMAND_LEFT: 91,
                COMMAND_RIGHT: 93,
                CONTROL: 17,
                DELETE: 46,
                DOWN: 40,
                END: 35,
                ENTER: 13,
                ESCAPE: 27,
                HOME: 36,
                INSERT: 45,
                LEFT: 37,
                MENU: 93,
                NUMPAD_ADD: 107,
                NUMPAD_DECIMAL: 110,
                NUMPAD_DIVIDE: 111,
                NUMPAD_ENTER: 108,
                NUMPAD_MULTIPLY: 106,
                NUMPAD_SUBTRACT: 109,
                PAGE_DOWN: 34,
                PAGE_UP: 33,
                PERIOD: 190,
                RIGHT: 39,
                SHIFT: 16,
                SPACE: 32,
                TAB: 9,
                UP: 38,
                WINDOWS: 91
            }
        });
        c.fn.extend({
            _focus: c.fn.focus,
            focus: function(a, b) {
                return typeof a === "number" ? this.each(function() {
                    var d = this;
                    setTimeout(function() {
                        c(d).focus();
                        b && b.call(d)
                    }, a)
                }) : this._focus.apply(this, arguments)
            },
            scrollParent: function() {
                var a;
                a = c.browser.msie && /(static|relative)/.test(this.css("position")) || /absolute/.test(this.css("position")) ? this.parents().filter(function() {
                    return /(relative|absolute|fixed)/.test(c.curCSS(this,
                    "position", 1)) && /(auto|scroll)/.test(c.curCSS(this, "overflow", 1) + c.curCSS(this, "overflow-y", 1) + c.curCSS(this, "overflow-x", 1))
                }).eq(0) : this.parents().filter(function() {
                    return /(auto|scroll)/.test(c.curCSS(this, "overflow", 1) + c.curCSS(this, "overflow-y", 1) + c.curCSS(this, "overflow-x", 1))
                }).eq(0);
                return /fixed/.test(this.css("position")) ||!a.length ? c(document) : a
            },
            zIndex: function(a) {
                if (a !== j)
                    return this.css("zIndex", a);
                if (this.length) {
                    a = c(this[0]);
                    for (var b; a.length && a[0] !== document;) {
                        b = a.css("position");
                        if (b === "absolute" || b === "relative" || b === "fixed") {
                            b = parseInt(a.css("zIndex"), 10);
                            if (!isNaN(b) && b !== 0)
                                return b
                        }
                        a = a.parent()
                    }
                }
                return 0
            },
            disableSelection: function() {
                return this.bind((c.support.selectstart ? "selectstart" : "mousedown") + ".ui-disableSelection", function(a) {
                    a.preventDefault()
                })
            },
            enableSelection: function() {
                return this.unbind(".ui-disableSelection")
            }
        });
        c.each(["Width", "Height"], function(a, b) {
            function d(f, g, l, m) {
                c.each(e, function() {
                    g -= parseFloat(c.curCSS(f, "padding" + this, true)) || 0;
                    if (l)
                        g -= parseFloat(c.curCSS(f,
                        "border" + this + "Width", true)) || 0;
                    if (m)
                        g -= parseFloat(c.curCSS(f, "margin" + this, true)) || 0
                });
                return g
            }
            var e = b === "Width" ? ["Left", "Right"]: ["Top", "Bottom"], h = b.toLowerCase(), i = {
                innerWidth: c.fn.innerWidth,
                innerHeight: c.fn.innerHeight,
                outerWidth: c.fn.outerWidth,
                outerHeight: c.fn.outerHeight
            };
            c.fn["inner" + b] = function(f) {
                if (f === j)
                    return i["inner" + b].call(this);
                return this.each(function() {
                    c(this).css(h, d(this, f) + "px")
                })
            };
            c.fn["outer" + b] = function(f, g) {
                if (typeof f !== "number")
                    return i["outer" + b].call(this, f);
                return this.each(function() {
                    c(this).css(h,
                    d(this, f, true, g) + "px")
                })
            }
        });
        c.extend(c.expr[":"], {
            data: function(a, b, d) {
                return !!c.data(a, d[3])
            },
            focusable: function(a) {
                var b = a.nodeName.toLowerCase(), d = c.attr(a, "tabindex");
                if ("area" === b) {
                    b = a.parentNode;
                    d = b.name;
                    if (!a.href ||!d || b.nodeName.toLowerCase() !== "map")
                        return false;
                    a = c("img[usemap=#" + d + "]")[0];
                    return !!a && k(a)
                }
                return (/input|select|textarea|button|object/.test(b)?!a.disabled : "a" == b ? a.href ||!isNaN(d) : !isNaN(d)) && k(a)
            },
            tabbable: function(a) {
                var b = c.attr(a, "tabindex");
                return (isNaN(b) || b >= 0) && c(a).is(":focusable")
            }
        });
        c(function() {
            var a = document.body, b = a.appendChild(b = document.createElement("div"));
            c.extend(b.style, {
                minHeight: "100px",
                height: "auto",
                padding: 0,
                borderWidth: 0
            });
            c.support.minHeight = b.offsetHeight === 100;
            c.support.selectstart = "onselectstart"in b;
            a.removeChild(b).style.display = "none"
        });
        c.extend(c.ui, {
            plugin: {
                add: function(a, b, d) {
                    a = c.ui[a].prototype;
                    for (var e in d) {
                        a.plugins[e] = a.plugins[e] || [];
                        a.plugins[e].push([b, d[e]])
                    }
                },
                call: function(a, b, d) {
                    if ((b = a.plugins[b]) && a.element[0].parentNode)
                        for (var e = 0; e < b.length; e++)
                            a.options[b[e][0]] &&
                            b[e][1].apply(a.element, d)
                }
            },
            contains: function(a, b) {
                return document.compareDocumentPosition ? a.compareDocumentPosition(b) & 16 : a !== b && a.contains(b)
            },
            hasScroll: function(a, b) {
                if (c(a).css("overflow") === "hidden")
                    return false;
                var d = b && b === "left" ? "scrollLeft": "scrollTop", e = false;
                if (a[d] > 0)
                    return true;
                a[d] = 1;
                e = a[d] > 0;
                a[d] = 0;
                return e
            },
            isOverAxis: function(a, b, d) {
                return a > b && a < b + d
            },
            isOver: function(a, b, d, e, h, i) {
                return c.ui.isOverAxis(a, d, h) && c.ui.isOverAxis(b, e, i)
            }
        })
    }
})(jQuery);

(function(b, j) {
    if (b.cleanData) {
        var k = b.cleanData;
        b.cleanData = function(a) {
            for (var c = 0, d; (d = a[c]) != null; c++)
                b(d).triggerHandler("remove");
            k(a)
        }
    } else {
        var l = b.fn.remove;
        b.fn.remove = function(a, c) {
            return this.each(function() {
                if (!c)
                    if (!a || b.filter(a, [this]).length)
                        b("*", this).add([this]).each(function() {
                            b(this).triggerHandler("remove")
                        });
                return l.call(b(this), a, c)
            })
        }
    }
    b.widget = function(a, c, d) {
        var e = a.split(".")[0], f;
        a = a.split(".")[1];
        f = e + "-" + a;
        if (!d) {
            d = c;
            c = b.Widget
        }
        b.expr[":"][f] = function(h) {
            return !!b.data(h,
            a)
        };
        b[e] = b[e] || {};
        b[e][a] = function(h, g) {
            arguments.length && this._createWidget(h, g)
        };
        c = new c;
        c.options = b.extend(true, {}, c.options);
        b[e][a].prototype = b.extend(true, c, {
            namespace: e,
            widgetName: a,
            widgetEventPrefix: b[e][a].prototype.widgetEventPrefix || a,
            widgetBaseClass: f
        }, d);
        b.widget.bridge(a, b[e][a])
    };
    b.widget.bridge = function(a, c) {
        b.fn[a] = function(d) {
            var e = typeof d === "string", f = Array.prototype.slice.call(arguments, 1), h = this;
            d=!e && f.length ? b.extend.apply(null, [true, d].concat(f)) : d;
            if (e && d.charAt(0) === "_")
                return h;
            e ? this.each(function() {
                var g = b.data(this, a), i = g && b.isFunction(g[d]) ? g[d].apply(g, f): g;
                if (i !== g && i !== j) {
                    h = i;
                    return false
                }
            }) : this.each(function() {
                var g = b.data(this, a);
                g ? g.option(d || {})._init() : b.data(this, a, new c(d, this))
            });
            return h
        }
    };
    b.Widget = function(a, c) {
        arguments.length && this._createWidget(a, c)
    };
    b.Widget.prototype = {
        widgetName: "widget",
        widgetEventPrefix: "",
        options: {
            disabled: false
        },
        _createWidget: function(a, c) {
            b.data(c, this.widgetName, this);
            this.element = b(c);
            this.options = b.extend(true, {}, this.options,
            this._getCreateOptions(), a);
            var d = this;
            this.element.bind("remove." + this.widgetName, function() {
                d.destroy()
            });
            this._create();
            this._trigger("create");
            this._init()
        },
        _getCreateOptions: function() {
            return b.metadata && b.metadata.get(this.element[0])[this.widgetName]
        },
        _create: function() {},
        _init: function() {},
        destroy: function() {
            this.element.unbind("." + this.widgetName).removeData(this.widgetName);
            this.widget().unbind("." + this.widgetName).removeAttr("aria-disabled").removeClass(this.widgetBaseClass + "-disabled ui-state-disabled")
        },
        widget: function() {
            return this.element
        },
        option: function(a, c) {
            var d = a;
            if (arguments.length === 0)
                return b.extend({}, this.options);
            if (typeof a === "string") {
                if (c === j)
                    return this.options[a];
                d = {};
                d[a] = c
            }
            this._setOptions(d);
            return this
        },
        _setOptions: function(a) {
            var c = this;
            b.each(a, function(d, e) {
                c._setOption(d, e)
            });
            return this
        },
        _setOption: function(a, c) {
            this.options[a] = c;
            if (a === "disabled")
                this.widget()[c ? "addClass": "removeClass"](this.widgetBaseClass + "-disabled ui-state-disabled").attr("aria-disabled", c);
            return this
        },
        enable: function() {
            return this._setOption("disabled", false)
        },
        disable: function() {
            return this._setOption("disabled", true)
        },
        _trigger: function(a, c, d) {
            var e = this.options[a];
            c = b.Event(c);
            c.type = (a === this.widgetEventPrefix ? a : this.widgetEventPrefix + a).toLowerCase();
            d = d || {};
            if (c.originalEvent) {
                a = b.event.props.length;
                for (var f; a;) {
                    f = b.event.props[--a];
                    c[f] = c.originalEvent[f]
                }
            }
            this.element.trigger(c, d);
            return !(b.isFunction(e) && e.call(this.element[0], c, d) === false || c.isDefaultPrevented())
        }
    }
})(jQuery);

(function(c) {
    c.ui = c.ui || {};
    var n = /left|center|right/, o = /top|center|bottom/, t = c.fn.position, u = c.fn.offset;
    c.fn.position = function(b) {
        if (!b ||!b.of)
            return t.apply(this, arguments);
        b = c.extend({}, b);
        var a = c(b.of), d = a[0], g = (b.collision || "flip").split(" "), e = b.offset ? b.offset.split(" "): [0, 0], h, k, j;
        if (d.nodeType === 9) {
            h = a.width();
            k = a.height();
            j = {
                top: 0,
                left: 0
            }
        } else if (d.setTimeout) {
            h = a.width();
            k = a.height();
            j = {
                top: a.scrollTop(),
                left: a.scrollLeft()
            }
        } else if (d.preventDefault) {
            b.at = "left top";
            h = k = 0;
            j = {
                top: b.of.pageY,
                left: b.of.pageX
            }
        } else {
            h = a.outerWidth();
            k = a.outerHeight();
            j = a.offset()
        }
        c.each(["my", "at"], function() {
            var f = (b[this] || "").split(" ");
            if (f.length === 1)
                f = n.test(f[0]) ? f.concat(["center"]) : o.test(f[0]) ? ["center"].concat(f) : ["center", "center"];
            f[0] = n.test(f[0]) ? f[0] : "center";
            f[1] = o.test(f[1]) ? f[1] : "center";
            b[this] = f
        });
        if (g.length === 1)
            g[1] = g[0];
        e[0] = parseInt(e[0], 10) || 0;
        if (e.length === 1)
            e[1] = e[0];
        e[1] = parseInt(e[1], 10) || 0;
        if (b.at[0] === "right")
            j.left += h;
        else if (b.at[0] === "center")
            j.left += h / 2;
        if (b.at[1] === "bottom")
            j.top +=
            k;
        else if (b.at[1] === "center")
            j.top += k / 2;
        j.left += e[0];
        j.top += e[1];
        return this.each(function() {
            var f = c(this), l = f.outerWidth(), m = f.outerHeight(), p = parseInt(c.curCSS(this, "marginLeft", true)) || 0, q = parseInt(c.curCSS(this, "marginTop", true)) || 0, v = l + p + (parseInt(c.curCSS(this, "marginRight", true)) || 0), w = m + q + (parseInt(c.curCSS(this, "marginBottom", true)) || 0), i = c.extend({}, j), r;
            if (b.my[0] === "right")
                i.left -= l;
            else if (b.my[0] === "center")
                i.left -= l / 2;
            if (b.my[1] === "bottom")
                i.top -= m;
            else if (b.my[1] === "center")
                i.top -=
                m / 2;
            i.left = Math.round(i.left);
            i.top = Math.round(i.top);
            r = {
                left: i.left - p,
                top: i.top - q
            };
            c.each(["left", "top"], function(s, x) {
                c.ui.position[g[s]] && c.ui.position[g[s]][x](i, {
                    targetWidth: h,
                    targetHeight: k,
                    elemWidth: l,
                    elemHeight: m,
                    collisionPosition: r,
                    collisionWidth: v,
                    collisionHeight: w,
                    offset: e,
                    my: b.my,
                    at: b.at
                })
            });
            c.fn.bgiframe && f.bgiframe();
            f.offset(c.extend(i, {
                using: b.using
            }))
        })
    };
    c.ui.position = {
        fit: {
            left: function(b, a) {
                var d = c(window);
                d = a.collisionPosition.left + a.collisionWidth - d.width() - d.scrollLeft();
                b.left =
                d > 0 ? b.left - d : Math.max(b.left - a.collisionPosition.left, b.left)
            },
            top: function(b, a) {
                var d = c(window);
                d = a.collisionPosition.top + a.collisionHeight - d.height() - d.scrollTop();
                b.top = d > 0 ? b.top - d : Math.max(b.top - a.collisionPosition.top, b.top)
            }
        },
        flip: {
            left: function(b, a) {
                if (a.at[0] !== "center") {
                    var d = c(window);
                    d = a.collisionPosition.left + a.collisionWidth - d.width() - d.scrollLeft();
                    var g = a.my[0] === "left"?-a.elemWidth : a.my[0] === "right" ? a.elemWidth : 0, e = a.at[0] === "left" ? a.targetWidth : - a.targetWidth, h =- 2 * a.offset[0];
                    b.left +=
                    a.collisionPosition.left < 0 ? g + e + h : d > 0 ? g + e + h : 0
                }
            },
            top: function(b, a) {
                if (a.at[1] !== "center") {
                    var d = c(window);
                    d = a.collisionPosition.top + a.collisionHeight - d.height() - d.scrollTop();
                    var g = a.my[1] === "top"?-a.elemHeight : a.my[1] === "bottom" ? a.elemHeight : 0, e = a.at[1] === "top" ? a.targetHeight : - a.targetHeight, h =- 2 * a.offset[1];
                    b.top += a.collisionPosition.top < 0 ? g + e + h : d > 0 ? g + e + h : 0
                }
            }
        }
    };
    if (!c.offset.setOffset) {
        c.offset.setOffset = function(b, a) {
            if (/static/.test(c.curCSS(b, "position")))
                b.style.position = "relative";
            var d = c(b),
            g = d.offset(), e = parseInt(c.curCSS(b, "top", true), 10) || 0, h = parseInt(c.curCSS(b, "left", true), 10) || 0;
            g = {
                top: a.top - g.top + e,
                left: a.left - g.left + h
            };
            "using"in a ? a.using.call(b, g) : d.css(g)
        };
        c.fn.offset = function(b) {
            var a = this[0];
            if (!a ||!a.ownerDocument)
                return null;
            if (b)
                return this.each(function() {
                    c.offset.setOffset(this, b)
                });
            return u.call(this)
        }
    }
})(jQuery);

(function(d) {
    d.widget("ui.autocomplete", {
        options: {
            appendTo: "body",
            delay: 300,
            minLength: 1,
            position: {
                my: "left top",
                at: "left bottom",
                collision: "none"
            },
            source: null
        },
        pending: 0,
        _create: function() {
            var a = this, b = this.element[0].ownerDocument, f;
            this.element.addClass("ui-autocomplete-input").attr("autocomplete", "off").attr({
                role: "textbox",
                "aria-autocomplete": "list",
                "aria-haspopup": "true"
            }).bind("keydown.autocomplete", function(c) {
                if (!(a.options.disabled || a.element.attr("readonly"))) {
                    f = false;
                    var e = d.ui.keyCode;
                    switch (c.keyCode) {
                    case e.PAGE_UP:
                        a._move("previousPage", c);
                        break;
                    case e.PAGE_DOWN:
                        a._move("nextPage", c);
                        break;
                    case e.UP:
                        a._move("previous", c);
                        c.preventDefault();
                        break;
                    case e.DOWN:
                        a._move("next", c);
                        c.preventDefault();
                        break;
                    case e.ENTER:
                    case e.NUMPAD_ENTER:
                        if (a.menu.active) {
                            f = true;
                            c.preventDefault()
                        }
                    case e.TAB:
                        if (!a.menu.active)
                            return;
                        a.menu.select(c);
                        break;
                    case e.ESCAPE:
                        a.element.val(a.term);
                        a.close(c);
                        break;
                    default:
                        clearTimeout(a.searching);
                        a.searching = setTimeout(function() {
                            if (a.term != a.element.val()) {
                                a.selectedItem =
                                null;
                                a.search(null, c)
                            }
                        }, a.options.delay);
                        break
                    }
                }
            }).bind("keypress.autocomplete", function(c) {
                if (f) {
                    f = false;
                    c.preventDefault()
                }
            }).bind("focus.autocomplete", function() {
                if (!a.options.disabled) {
                    a.selectedItem = null;
                    a.previous = a.element.val()
                }
            }).bind("blur.autocomplete", function(c) {
                if (!a.options.disabled) {
                    clearTimeout(a.searching);
                    a.closing = setTimeout(function() {
                        a.close(c);
                        a._change(c)
                    }, 150)
                }
            });
            this._initSource();
            this.response = function() {
                return a._response.apply(a, arguments)
            };
            this.menu = d("<ul></ul>").addClass("ui-autocomplete").appendTo(d(this.options.appendTo ||
            "body", b)[0]).mousedown(function(c) {
                var e = a.menu.element[0];
                d(c.target).closest(".ui-menu-item").length || setTimeout(function() {
                    d(document).one("mousedown", function(g) {
                        g.target !== a.element[0] && g.target !== e&&!d.ui.contains(e, g.target) && a.close()
                    })
                }, 1);
                setTimeout(function() {
                    clearTimeout(a.closing)
                }, 13)
            }).menu({
                focus: function(c, e) {
                    var g = e.item.data("item.autocomplete");
                    false !== a._trigger("focus", c, {
                        item: g
                    }) && /^key/.test(c.originalEvent.type) && a.element.val(g.value)
                },
                selected: function(c, e) {
                    var g = e.item.data("item.autocomplete"),
                    h = a.previous;
                    if (a.element[0] !== b.activeElement) {
                        a.element.focus();
                        a.previous = h;
                        setTimeout(function() {
                            a.previous = h;
                            a.selectedItem = g
                        }, 1)
                    }
                    false !== a._trigger("select", c, {
                        item: g
                    }) && a.element.val(g.value);
                    a.term = a.element.val();
                    a.close(c);
                    a.selectedItem = g
                },
                blur: function() {
                    a.menu.element.is(":visible") && a.element.val() !== a.term && a.element.val(a.term)
                }
            }).zIndex(this.element.zIndex() + 1).css({
                top: 0,
                left: 0
            }).hide().data("menu");
            d.fn.bgiframe && this.menu.element.bgiframe()
        },
        destroy: function() {
            this.element.removeClass("ui-autocomplete-input").removeAttr("autocomplete").removeAttr("role").removeAttr("aria-autocomplete").removeAttr("aria-haspopup");
            this.menu.element.remove();
            d.Widget.prototype.destroy.call(this)
        },
        _setOption: function(a, b) {
            d.Widget.prototype._setOption.apply(this, arguments);
            a === "source" && this._initSource();
            if (a === "appendTo")
                this.menu.element.appendTo(d(b || "body", this.element[0].ownerDocument)[0]);
            a === "disabled" && b && this.xhr && this.xhr.abort()
        },
        _initSource: function() {
            var a = this, b, f;
            if (d.isArray(this.options.source)) {
                b = this.options.source;
                this.source = function(c, e) {
                    e(d.ui.autocomplete.filter(b, c.term))
                }
            } else if (typeof this.options.source ===
            "string") {
                f = this.options.source;
                this.source = function(c, e) {
                    a.xhr && a.xhr.abort();
                    a.xhr = d.ajax({
                        url: f,
                        data: c,
                        dataType: "json",
                        success: function(g, h, i) {
                            i === a.xhr && e(g);
                            a.xhr = null
                        },
                        error: function(g) {
                            g === a.xhr && e([]);
                            a.xhr = null
                        }
                    })
                }
            } else 
                this.source = this.options.source
        },
        search: function(a, b) {
            a = a != null ? a : this.element.val();
            this.term = this.element.val();
            if (a.length < this.options.minLength)
                return this.close(b);
            clearTimeout(this.closing);
            if (this._trigger("search", b) !== false)
                return this._search(a)
        },
        _search: function(a) {
            this.pending++;
            this.element.addClass("ui-autocomplete-loading");
            this.source({
                term: a
            }, this.response)
        },
        _response: function(a) {
            if (!this.options.disabled && a && a.length) {
                a = this._normalize(a);
                this._suggest(a);
                this._trigger("open")
            } else 
                this.close();
            this.pending--;
            this.pending || this.element.removeClass("ui-autocomplete-loading")
        },
        close: function(a) {
            clearTimeout(this.closing);
            if (this.menu.element.is(":visible")) {
                this.menu.element.hide();
                this.menu.deactivate();
                this._trigger("close", a)
            }
        },
        _change: function(a) {
            this.previous !==
            this.element.val() && this._trigger("change", a, {
                item: this.selectedItem
            })
        },
        _normalize: function(a) {
            if (a.length && a[0].label && a[0].value)
                return a;
            return d.map(a, function(b) {
                if (typeof b === "string")
                    return {
                        label: b,
                        value: b
                    };
                return d.extend({
                    label: b.label || b.value,
                    value: b.value || b.label
                }, b)
            })
        },
        _suggest: function(a) {
            var b = this.menu.element.empty().zIndex(this.element.zIndex() + 1);
            this._renderMenu(b, a);
            this.menu.deactivate();
            this.menu.refresh();
            b.show();
            this._resizeMenu();
            b.position(d.extend({
                of: this.element
            }, this.options.position))
        },
        _resizeMenu: function() {
            var a = this.menu.element;
            a.outerWidth(Math.max(a.width("").outerWidth(), this.element.outerWidth()))
        },
        _renderMenu: function(a, b) {
            var f = this;
            d.each(b, function(c, e) {
                f._renderItem(a, e)
            })
        },
        _renderItem: function(a, b) {
            return d("<li></li>").data("item.autocomplete", b).append(d("<a></a>").text(b.label)).appendTo(a)
        },
        _move: function(a, b) {
            if (this.menu.element.is(":visible"))
                if (this.menu.first() && /^previous/.test(a) || this.menu.last() && /^next/.test(a)) {
                    this.element.val(this.term);
                    this.menu.deactivate()
                } else 
                    this.menu[a](b);
            else 
                this.search(null, b)
        },
        widget: function() {
            return this.menu.element
        }
    });
    d.extend(d.ui.autocomplete, {
        escapeRegex: function(a) {
            return a.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, "\\$&")
        },
        filter: function(a, b) {
            var f = RegExp(d.ui.autocomplete.escapeRegex(b), "i");
            return d.grep(a, function(c) {
                return f.test(c.label || c.value || c)
            })
        }
    })
})(jQuery);
(function(d) {
    d.widget("ui.menu", {
        _create: function() {
            var a = this;
            this.element.addClass("ui-menu ui-widget ui-widget-content ui-corner-all").attr({
                role: "listbox",
                "aria-activedescendant": "ui-active-menuitem"
            }).click(function(b) {
                if (d(b.target).closest(".ui-menu-item a").length) {
                    b.preventDefault();
                    a.select(b)
                }
            });
            this.refresh()
        },
        refresh: function() {
            var a = this;
            this.element.children("li:not(.ui-menu-item):has(a)").addClass("ui-menu-item").attr("role", "menuitem").children("a").addClass("ui-corner-all").attr("tabindex",
            - 1).mouseenter(function(b) {
                a.activate(b, d(this).parent())
            }).mouseleave(function() {
                a.deactivate()
            })
        },
        activate: function(a, b) {
            this.deactivate();
            if (this.hasScroll()) {
                var f = b.offset().top - this.element.offset().top, c = this.element.attr("scrollTop"), e = this.element.height();
                if (f < 0)
                    this.element.attr("scrollTop", c + f);
                else 
                    f >= e && this.element.attr("scrollTop", c + f - e + b.height())
            }
            this.active = b.eq(0).children("a").addClass("ui-state-hover").attr("id", "ui-active-menuitem").end();
            this._trigger("focus", a, {
                item: b
            })
        },
        deactivate: function() {
            if (this.active) {
                this.active.children("a").removeClass("ui-state-hover").removeAttr("id");
                this._trigger("blur");
                this.active = null
            }
        },
        next: function(a) {
            this.move("next", ".ui-menu-item:first", a)
        },
        previous: function(a) {
            this.move("prev", ".ui-menu-item:last", a)
        },
        first: function() {
            return this.active&&!this.active.prevAll(".ui-menu-item").length
        },
        last: function() {
            return this.active&&!this.active.nextAll(".ui-menu-item").length
        },
        move: function(a, b, f) {
            if (this.active) {
                a = this.active[a + "All"](".ui-menu-item").eq(0);
                a.length ? this.activate(f, a) : this.activate(f, this.element.children(b))
            } else 
                this.activate(f, this.element.children(b))
        },
        nextPage: function(a) {
            if (this.hasScroll())
                if (!this.active || this.last())
                    this.activate(a, this.element.children(".ui-menu-item:first"));
                else {
                    var b = this.active.offset().top, f = this.element.height(), c = this.element.children(".ui-menu-item").filter(function() {
                        var e = d(this).offset().top - b - f + d(this).height();
                        return e < 10 && e>-10
                    });
                    c.length || (c = this.element.children(".ui-menu-item:last"));
                    this.activate(a,
                    c)
                } else 
                    this.activate(a, this.element.children(".ui-menu-item").filter(!this.active || this.last() ? ":first" : ":last"))
        },
        previousPage: function(a) {
            if (this.hasScroll())
                if (!this.active || this.first())
                    this.activate(a, this.element.children(".ui-menu-item:last"));
                else {
                    var b = this.active.offset().top, f = this.element.height();
                    result = this.element.children(".ui-menu-item").filter(function() {
                        var c = d(this).offset().top - b + f - d(this).height();
                        return c < 10 && c>-10
                    });
                    result.length || (result = this.element.children(".ui-menu-item:first"));
                    this.activate(a, result)
                } else 
                    this.activate(a, this.element.children(".ui-menu-item").filter(!this.active || this.first() ? ":last" : ":first"))
        },
        hasScroll: function() {
            return this.element.height() < this.element.attr("scrollHeight")
        },
        select: function(a) {
            this._trigger("selected", a, {
                item: this.active
            })
        }
    })
})(jQuery);

(function(b) {
    if (!document.defaultView ||!document.defaultView.getComputedStyle) {
        var d = jQuery.curCSS;
        jQuery.curCSS = function(g, e, h) {
            if (e === "background-position") {
                e = "backgroundPosition"
            }
            if (e !== "backgroundPosition" ||!g.currentStyle || g.currentStyle[e]) {
                return d.apply(this, arguments)
            }
            var f = g.style;
            if (!h && f && f[e]) {
                return f[e]
            }
            return d(g, "backgroundPositionX", h) + " " + d(g, "backgroundPositionY", h)
        }
    }
    var c = b.fn.animate;
    b.fn.animate = function(e) {
        if ("background-position" in e) {
            e.backgroundPosition = e["background-position"];
            delete e["background-position"]
        }
        if ("backgroundPosition" in e) {
            e.backgroundPosition = "(" + e.backgroundPosition
        }
        return c.apply(this, arguments)
    };
    function a(f) {
        f = f.replace(/left|top/g, "0px");
        f = f.replace(/right|bottom/g, "100%");
        f = f.replace(/([0-9\.]+)(\s|\)|$)/g, "$1px$2");
        var e = f.match(/(-?[0-9\.]+)(px|\%|em|pt)\s(-?[0-9\.]+)(px|\%|em|pt)/);
        return [parseFloat(e[1], 10), e[2], parseFloat(e[3], 10), e[4]]
    }
    b.fx.step.backgroundPosition = function(f) {
        if (!f.bgPosReady) {
            var h = b.curCSS(f.elem, "backgroundPosition");
            if (!h) {
                h = "0px 0px"
            }
            h = a(h);
            f.start = [h[0], h[2]];
            var e = a(f.options.curAnim.backgroundPosition);
            f.end = [e[0], e[2]];
            f.unit = [e[1], e[3]];
            f.bgPosReady = true
        }
        var g = [];
        g[0] = ((f.end[0] - f.start[0]) * f.pos) + f.start[0] + f.unit[0];
        g[1] = ((f.end[1] - f.start[1]) * f.pos) + f.start[1] + f.unit[1];
        f.elem.style.backgroundPosition = g[0] + " " + g[1]
    }
})(jQuery);
(function(d) {
    function g(a) {
        var b = a || window.event, i = [].slice.call(arguments, 1), c = 0, h = 0, e = 0;
        a = d.event.fix(b);
        a.type = "mousewheel";
        if (a.wheelDelta)
            c = a.wheelDelta / 120;
        if (a.detail)
            c =- a.detail / 3;
        e = c;
        if (b.axis !== undefined && b.axis === b.HORIZONTAL_AXIS) {
            e = 0;
            h =- 1 * c
        }
        if (b.wheelDeltaY !== undefined)
            e = b.wheelDeltaY / 120;
        if (b.wheelDeltaX !== undefined)
            h =- 1 * b.wheelDeltaX / 120;
        i.unshift(a, c, h, e);
        return d.event.handle.apply(this, i)
    }
    var f = ["DOMMouseScroll", "mousewheel"];
    d.event.special.mousewheel = {
        setup: function() {
            if (this.addEventListener)
                for (var a =
                f.length; a;)
                    this.addEventListener(f[--a], g, false);
            else 
                this.onmousewheel = g
        },
        teardown: function() {
            if (this.removeEventListener)
                for (var a = f.length; a;)
                    this.removeEventListener(f[--a], g, false);
            else 
                this.onmousewheel = null
        }
    };
    d.fn.extend({
        mousewheel: function(a) {
            return a ? this.bind("mousewheel", a) : this.trigger("mousewheel")
        },
        unmousewheel: function(a) {
            return this.unbind("mousewheel", a)
        }
    })
})(jQuery);

var HPEventModule = {
    js_main_modules_loaded: {},
    js_events_loaded: {},
    yui_version_default: "2.7.0",
    events_dependencies: {
        slideshow_participate: {
            immediately: {
                yui: {},
                our: ["quickslideshowparticipate", "hpimagecrop"],
                script: ["hp_browser", "_tmp/lightbox_search", "_tmp/common"]
            },
            delayed: {
                yui: {
                    button: "default"
                }
            }
        },
        quiz_share: {
            immediately: {
                our: ["comments"]
            }
        }
    },
    modules_loaded: {},
    yui_modules_loaded: {},
    Load: function(d, e, b, a) {
        b = b || this;
        a = a || [];
        e = e || (function() {});
        if (this.js_events_loaded[d]) {
            e.apply(b, a)
        } else {
            var c = this;
            this._LoadModules(d, function() {
                e.apply(b, a)
            })
        }
    },
    Wait: function(d, f, e, a, b) {
        if (e) {
            e()
        }
        a = (undefined === a) ? true : a;
        b = b || this;
        var c = this;
        HPUtil.WaitForCondition.apply(b, [function() {
            setTimeout(function() {
                f.apply(b)
            }, 100)
        }, 1, function() {
            return !a ? c.js_main_modules_loaded[d] : c.js_events_loaded[d]
        }
        ])
    },
    _LoadModules: function(g, l, k) {
        l = l || (function() {});
        k = k || "immediately";
        var c = this.events_dependencies[g][k];
        var b = "/assets/js.php?" + HPConfig.timestamp_for_clearing_js + "&f=";
        var a = [], e = [], j = "";
        if (c.our) {
            for (var d = 0; d < c.our.length; ++d) {
                j = c.our[d];
                if (this.modules_loaded[j]) {
                    continue
                }
                a[a.length] = "modules/" + j + ".js"
            }
        }
        if (c.yui) {
            for (var j in c.yui) {
                if (this.yui_modules_loaded[j]) {
                    continue
                }
                a[a.length] = "yui_" + ("default" !== c.yui[j] ? c.yui[j] : this.yui_version_default) + "/" + j + "/" + j + "-min.js"
            }
        }
        if (c.script) {
            e = c.script.map(function(i) {
                return i + ".js"
            })
        }
        if (a.length + e.length) {
            a = a.sort();
            if (c.yui && typeof YAHOO === "undefined") {
                e.unshift("yui.js")
            }
            b += e.concat(a).join("%2C");
            var h = this;
            var f = function() {
                var n = "";
                for (var m = 0; m < a.length; ++m) {
                    n = (new RegExp(/\/(.*?)\.js$/)).exec(a[m]);
                    if ( - 1 !== a[m].indexOf("yui_")) {
                        h.yui_modules_loaded[n[1]] = true
                    } else {
                        h.modules_loaded[n[1]] = true
                    }
                }
                l();
                if ("immediately" == k && h.events_dependencies[g]["delayed"]) {
                    h.js_main_modules_loaded[g] = 1;
                    h._LoadModules(g, null, "delayed")
                } else {
                    h.js_events_loaded[g] = 1
                }
            };
            HPUtil.loadAndRun(b, f)
        }
    }
};
huff.unit("tmp/event_module.js", function(a) {
    a(HPEventModule)
});
YAHOO.namespace("HPDocStatus");
var HPDocStatus = YAHOO.HPDocStatus;
HPDocStatus.on_focus = true;
HPDocStatus.setFocusHandler = function(a) {
    if (!a) {
        hpcallback = function() {
            HPDocStatus.on_focus = true
        }
    } else {
        hpcallback = function() {
            a();
            HPDocStatus.on_focus = true
        }
    }
    if (HPBrowser.isIEOld()) {
        window.onfocusin = hpcallback
    } else {
        YAHOO.util.Event.addListener(window, "focus", hpcallback)
    }
};
HPDocStatus.setBlurHandler = function(a) {
    if (!a) {
        hpcallback = function() {
            HPDocStatus.on_focus = false
        }
    } else {
        hpcallback = function() {
            a();
            HPDocStatus.on_focus = false
        }
    }
    if (HPBrowser.isIEOld()) {
        window.onfocusout = hpcallback
    } else {
        YAHOO.util.Event.addListener(window, "blur", hpcallback)
    }
};
huff.unit("tmp/doc_status", function(a) {
    a(HPDocStatus)
});
var ViewTracker = {
    VerticalType: - 1,
    VerticalTypeViews: null,
    AddView: function(b, a) {
        if (!this.VerticalTypeViews) {
            this.Init()
        }
        if (!b) {
            b = this.VerticalType
        }
        if (!a) {
            a = 1
        }
        if (this.VerticalTypeViews[b]) {
            this.VerticalTypeViews[b] += a
        } else {
            this.VerticalTypeViews[b] = a
        }
        HuffCookies.set("huffpo_type_views", JSON.stringify(this.VerticalTypeViews), 30 * 24)
    },
    Init: function() {
        var a = HuffCookies.get("huffpo_type_views");
        if (a) {
            this.VerticalTypeViews = JSON.parse(a)
        }
        if (!this.VerticalTypeViews) {
            this.VerticalTypeViews = {}
        }
    },
    GetMostViewedVertical: function() {
        var a = 0;
        var b =- 1;
        for (var c in this.VerticalTypeViews) {
            if (parseInt(c) == "NaN") {
                continue
            }
            if (a < this.VerticalTypeViews[c]) {
                a = this.VerticalTypeViews[c];
                b = c
            }
        }
        return b
    }
};
huff.unit("tmp/view_tracker", function(a) {
    a(ViewTracker)
});
var Slider = {
    Next: function(a) {
        if (Slider.Positions[a] < Slider.Lengths[a] - 1) {
            Slider.Positions[a]++
        } else {
            return false
        }
        Slider.LoadImage(a);
        Dom.setStyle(Dom.getElementsByClassName("slider_slide", "div", "slider_" + a), "display", "none");
        Dom.setStyle("slider_" + a + "_slide_" + Slider.Positions[a], "display", "block");
        if (Slider.Positions[a] == Slider.Lengths[a] - 1) {
            Dom.addClass("slider_right_" + a, "slider_off")
        }
        Dom.removeClass("slider_left_" + a, "slider_off")
    },
    Previous: function(a) {
        if (Slider.Positions[a] > 0) {
            Slider.Positions[a]--
        } else {
            return false
        }
        Slider.LoadImage(a);
        Dom.setStyle(Dom.getElementsByClassName("slider_slide", "div", "slider_" + a), "display", "none");
        Dom.setStyle("slider_" + a + "_slide_" + Slider.Positions[a], "display", "block");
        if (Slider.Positions[a] == 0) {
            Dom.addClass("slider_left_" + a, "slider_off")
        }
        Dom.removeClass("slider_right_" + a, "slider_off")
    },
    LoadImage: function(a) {
        this_slide = $("slider_" + a + "_img_" + Slider.Positions[a]);
        if (this_slide&&!this_slide.src < 2 && this_slide.alt) {
            this_slide.src = this_slide.alt
        }
    },
    Positions: new Array(),
    Lengths: new Array()
};
huff.unit("tmp/slider", function(a) {
    a(Slider)
});
StructuredImage = function(a) {
    if (!(decon = /<HH--(DEV--)?PHOTO--([A-Z\-_0-9]*)--(\d+)--HH>/.exec(a))) {
        return false
    }
    this.keywords = decon[2];
    this.id = decon[3];
    this.domain = (decon[1]) ? "dev.assets.huffingtonpost.com" : "i.huffpost.com";
    this.path = "http://" + this.domain + "/gen/" + this.id + "/thumbs/"
};
StructuredImage.prototype.Url = function(a, b) {
    return this.path + a + "-" + this.keywords + "-" + b + ".jpg"
};
huff.unit("tmp/structured_image.js", function(a) {
    a(StructuredImage)
});
var CurtainModule = {
    collapse: function() {
        $("main_container").style.display = "none";
        HuffCookies.set("huffpost_curtain", 1)
    }
};
huff.unit("tmp/curtain", function(a) {
    a(CurtainModule)
});
function Paginator(d) {
    if (!d.hasOwnProperty("paginator_id")) {
        return 
    }
    if (!d.hasOwnProperty("name")) {
        return 
    }
    if (!d.hasOwnProperty("entry_class")) {
        return 
    }
    if (!d.hasOwnProperty("entries_id")) {
        return 
    }
    if (!d.hasOwnProperty("entries_per_page")) {
        this.entries_per_page = 5
    }
    this.entries_id = d.entries_id;
    this.paginator_id = d.paginator_id;
    this.name = d.name;
    this.entry_class = d.entry_class;
    this.entries_per_page = this.entries_per_page ? this.entries_per_page : d.entries_per_page;
    this.entries = Dom.getElementsByClassName(this.entry_class);
    this.entries_tmp = [];
    var f = document.createElement("div");
    for (var e = 0; e < this.entries.length; e++) {
        if (this.entries[e]) {
            var f = document.createElement("div");
            f.appendChild(this.entries[e]);
            this.entries_tmp.push(f.innerHTML)
        }
    }
    this.BuildPaginator = a;
    this.RenderPage = c;
    this.UpdateFadeInOut = b;
    this.RenderPage(1);
    Dom.setStyle(this.entries_id, "display", "block");
    function c(i) {
        var g = (i - 1) * this.entries_per_page;
        var l = g + this.entries_per_page - 1;
        var h = "";
        for (var k = g; k <= l; k++) {
            if (this.entries_tmp[k]) {
                h += this.entries_tmp[k]
            }
        }
        this.UpdateFadeInOut(this.entries_id, h);
        var j = this.BuildPaginator(i);
        Dom.get(this.paginator_id).innerHTML = j
    }
    function a(m) {
        var r = "Pages: ";
        var q = 5;
        var h = this.entries_per_page;
        var n = this.entries_tmp.length;
        if (1 < (m - q)) {
            r += '<a href="javascript:' + this.name + '.RenderPage(1);">1</a> ';
            if (1 < (m - q - 1)) {
                r += "... "
            }
        }
        var j = Math.ceil(n / h);
        var l = m - q;
        if (l < 1) {
            l = 1
        }
        var o = m + q;
        if (o > j) {
            o = j
        }
        for (var k = l; k <= o; k++) {
            var g = (k - 1) * 4;
            if (k == m) {
                r += k + " "
            } else {
                r += '<a href="javascript:' + this.name + ".RenderPage(" + k + ');">' + k + "</a> "
            }
        }
        if ((m + q) < j) {
            if ((m + q + 1) < j) {
                r += "... "
            }
            r += '<a href="javascript:' + this.name + ".RenderPage(" + j + ');">' + j + "</a> "
        }
        return r
    }
    function b(j, i) {
        var h = new YAHOO.util.Anim($(j), {
            opacity: {
                to: 0
            }
        }, 0.5);
        var g = function(m, l) {
            $(j).innerHTML = i;
            var k = new YAHOO.util.Anim($(j), {
                opacity: {
                    to: 1
                }
            }, 0.5);
            k.animate()
        };
        h.onComplete.subscribe(g);
        h.animate()
    }
}
huff.unit("tmp/paginator", function(a) {
    a()
});
var OnFocusTimer = function() {
    var active = true;
    var _functionProxy = function(code) {
        if (typeof(code) == "string") {
            return function() {
                if (active) {
                    eval(code)
                }
            }
        } else {
            if (typeof(code) == "function") {
                return function() {
                    if (active) {
                        code()
                    }
                }
            }
        }
        return function() {}
    };
    var _onFocus = function() {
        active = true
    };
    var _onBlur = function() {
        active = false
    };
    YAHOO.util.Event.on(window, "focus", _onFocus);
    YAHOO.util.Event.on(window, "blur", _onBlur);
    return {
        setTimeout: function(code, timeout) {
            return setTimeout(_functionProxy(code), timeout)
        },
        setInterval: function(code, interval) {
            return setInterval(_functionProxy(code), interval)
        }
    }
}();
huff.unit("tmp/on_focus_timer", function(a) {
    a(OnFocusTimer)
});
window.sprintf || (window.sprintf = function(a) {
    return a
});
window._ || (window._ = function(a) {
    return a
});
var TwitterUtil = {
    elapsedTime: function(e, c) {
        var b = e;
        var a;
        var f;
        f = new Date(e).getTime();
        if (c) {
            b = this.fixDate(b);
            a = (new Date().getTime() - new Date(b).getTime()) / 1000;
            if (isNaN(a)) {
                a = (new Date().getTime() - f) / 1000
            }
        } else {
            a = (new Date().getTime() - f) / 1000
        }
        var d = function(h) {
            return h == 1
        };
        if (a < 60) {
            return _("less than a minute ago")
        }
        if (a < 60 * 60) {
            var g = Math.floor(a / 60);
            return d(g) ? sprintf(_("%s minute ago"), g) : sprintf(_("%s minutes ago"), g)
        }
        if (a < 60 * 60 * 24) {
            var g = Math.floor(a / 60 / 60);
            return d(g) ? sprintf(_("%s hour ago"), g) : sprintf(_("%s hours ago"), g)
        }
        if (a < 60 * 60 * 24 * 7) {
            var g = Math.floor(a / 60 / 60 / 24);
            return d(g) ? sprintf(_("%s day ago"), g) : sprintf(_("%s days ago"), g)
        }
        if (a < 60 * 60 * 24 * 31) {
            var g = Math.floor(a / 60 / 60 / 24 / 7);
            return d(g) ? sprintf(_("%s week ago"), g) : sprintf(_("%s weeks ago"), g)
        }
        if (a < 60 * 60 * 24 * 365) {
            var g = Math.floor(a / 60 / 60 / 24 / 31);
            return d(g) ? sprintf(_("%s month ago"), g) : sprintf(_("%s months ago"), g)
        }
        var g = Math.floor(a / 60 / 60 / 24 / 365);
        return d(g) ? sprintf(_("%s year ago"), g) : sprintf(_("%s years ago"), g)
    },
    fixDate: function(e) {
        if (HPBrowser.isIEOld()) {
            var b = e.split(" ");
            var c = b.pop();
            return b.slice(0, 3).concat([c]).concat(b.slice(3)).join(" ")
        } else {
            return e
        }
    },
    linkifyTextLinks: function(a) {
        return a.replace(/[a-z]+:\/\/[a-z0-9-_]+\.[a-z0-9-_:~%&\?\/.=]+[^:\.,\)\s*$]/ig, function(b) {
            return '<a href="' + b + '" target="_blank">' + ((b.length > 25) ? b.substr(0, 24) + "..." : b) + "</a>"
        })
    },
    twittify: function(b) {
        var a = function(d) {
            return d.replace(/(^|[^\w]+)\@([a-zA-Z0-9_]{1,15})/g, function(e, g, f) {
                return g + '<a href="http://twitter.com/' + f + '" target="_blank" class="twitter-anywhere-user">@' + f + "</a>"
            })
        };
        var c = function(d) {
            return d.replace(/(^|[^\w'"]+)\#([a-zA-Z0-9_]+)/g, function(e, g, f) {
                return g + '#<a href="http://search.twitter.com/search?q=%23' + f + '" target="_blank">' + f + "</a>"
            })
        };
        return c(a(this.linkifyTextLinks(b)))
    },
    arrayChunk: function(c, e) {
        e = parseInt(e, 10);
        e = isNaN(e) ||!e ? 1 : e;
        var d = [];
        var a = c.length;
        for (var b = 0; b < a; b += e) {
            d.push(c.slice(b, b + e))
        }
        return d
    },
    ucwords: function(a) {
        a = a || "";
        return a.toLowerCase().replace(/\w+/g, function(b) {
            return b.charAt(0).toUpperCase() + b.substr(1)
        })
    }
};
huff.unit("tmp/twitter_util", function(a) {
    a(TwitterUtil)
});
huff.unit("tmp/aol_search", function(a) {
    (function(b) {
        a(window.AOLSearch = {
            changeTab: function(d, c) {
                if (c == "news") {
                    b("#aol-web-search").hide();
                    b("#aol-news-search").show();
                    if (HPConfig.page_type == "frontpage") {
                        b(".aol-tabs > li > a").removeClass("vertical_color");
                        b(d).addClass("vertical_color")
                    } else {
                        b(".aol-tabs > li > a").removeClass("active");
                        b(d).addClass("active")
                    }
                } else {
                    if (c == "web") {
                        b("#aol-news-search").hide();
                        b("#aol-web-search").show();
                        if (HPConfig.page_type == "frontpage") {
                            b(".aol-tabs > li > a").removeClass("vertical_color");
                            b(d).addClass("vertical_color")
                        } else {
                            b(".aol-tabs > li > a").removeClass("active");
                            b(d).addClass("active")
                        }
                    }
                }
            },
            focusInput: function(c) {
                if (c.value == "Search News and Topics" || c.value == "Search the Web" || c.value == "*Please add search phrase" || c.value == "Search the Huffington Post" || c.value == "Search AOL Healthy Living") {
                    b("#" + c.id).attr("value", "").removeClass("red").removeClass("color_a1a1a1").addClass("color_111111")
                }
            },
            blurInput: function(c) {
                if (c.value == "") {
                    if (c.id[0] == "w") {
                        b("#" + c.id).attr("value", "Search the Web").removeClass("red").addClass("color_a1a1a1")
                    } else {
                        if (HPConfig.current_vertical_name == "healthy-living") {
                            b("#" + c.id).attr("value", "Search AOL Healthy Living").removeClass("red").addClass("color_a1a1a1")
                        } else {
                            if (c.id[0] == "h") {
                                b("#" + c.id).attr("value", "Search the Huffington Post").removeClass("red").addClass("color_a1a1a1")
                            } else {
                                b("#" + c.id).attr("value", "Search News and Topics").removeClass("red").addClass("color_a1a1a1")
                            }
                        }
                    }
                }
            },
            submitForm: function(d, c) {
                if (b(d).attr("value") == "" || b(d).attr("value") == "Search News and Topics" || b(d).attr("value") == "Search the Web" || b(d).attr("value") == "Search the Huffington Post" || b(d).attr("value") == "Search AOL Healthy Living" || b(d).attr("value") == "*Please add search phrase") {
                    b(d).attr("value", "*Please add search phrase").removeClass("color_a1a1a1").addClass("red");
                    return false
                } else {
                    c.submit()
                }
            }
        })
    })(jQuery)
});
var ElectionWidgets = {
    per_page: 1,
    pages: 4,
    cur_page: 1,
    showed_pages: [],
    goState: function(c) {
        var d = c.options.length;
        for (var b = 0; b < d; b++) {
            if (c.options[b].selected) {
                var a = "http://www.pollster.com/polls/" + c.options[b].value.toLowerCase();
                window.location.href = a;
                break
            }
        }
    },
    changeTab: function(b, d, f, c) {
        if (typeof b == "undefined" || typeof d == "undefined" || typeof f == "undefined" || typeof c == "undefined") {
            return false
        }
        var f = f.childNodes;
        for (var e = 0; e < f.length; e++) {
            if (f[e].nodeType == 1) {
                if (f[e].id == b) {
                    Dom.removeClass(b, "election-nonactive-tab");
                    Dom.addClass(b, "election-active-tab")
                } else {
                    Dom.removeClass(f[e].id, "election-active-tab");
                    Dom.addClass(f[e].id, "election-nonactive-tab")
                }
            }
        }
        var a = c.childNodes;
        for (e = 0; e < a.length; e++) {
            if (a[e].nodeType == 1) {
                if (a[e].id == d) {
                    Dom.setStyle(d, "display", "block")
                } else {
                    Dom.setStyle(a[e].id, "display", "none")
                }
            }
        }
    }
};
huff.unit("tmp/election_widgets", function(a) {
    a(ElectionWidgets)
});
function getSlide(b, a) {
    if (a == "INPUT" || a == "TEXTAREA") {
        return false
    }
    if (HPConfig.slideshow_type == "pollajax") {
        switch (b) {
        case"prev":
            window["oSlideshowPollAjax_" + HPConfig.slideshow_id].prev();
            break;
        case"next":
            window["oSlideshowPollAjax_" + HPConfig.slideshow_id].next();
            break
        }
        return false
    } else {
        if (HPConfig.slideshow_type == "fullscreen") {
            switch (b) {
            case"prev":
                if (!window.slideshow_fullscreen) {
                    window["FullScreen_" + HPConfig.slideshow_id].openSlideShow();
                    return false
                }
                window["FullScreen_" + HPConfig.slideshow_id].prevPhoto();
                break;
            case"next":
                if (!window.slideshow_fullscreen) {
                    window["FullScreen_" + HPConfig.slideshow_id].openSlideShow();
                    return false
                }
                window["FullScreen_" + HPConfig.slideshow_id].nextPhoto();
                break;
            case"esc":
                if (window.slideshow_fullscreen) {
                    window["FullScreen_" + HPConfig.slideshow_id].closeSlideShow();
                    return 
                }
                break;
            case"fullscreen":
                if (!window.slideshow_fullscreen) {
                    window["FullScreen_" + HPConfig.slideshow_id].openSlideShow();
                    return 
                }
                break;
            case"up":
                E.removeListener(window, "mousemove");
                HPUtil.show("fs_page_layout_" + HPConfig.slideshow_id);
                break;
            case"down":
                E.on(window, "mousemove", window["FullScreen_" + HPConfig.slideshow_id].showPageLayout, null, window["FullScreen_" + HPConfig.slideshow_id]);
                HPUtil.hide("fs_page_layout_" + HPConfig.slideshow_id);
                break
            }
            return false
        } else {
            if (HPConfig.slideshow_type == "adbigshot") {
                switch (b) {
                case"prev":
                    if (!window.slideshow_fullscreen) {
                        window["BigShot_" + HPConfig.slideshow_id].showBigShot();
                        return false
                    }
                    window["BigShot_" + HPConfig.slideshow_id].prevPhoto();
                    break;
                case"next":
                    if (!window.slideshow_fullscreen) {
                        window["BigShot_" + HPConfig.slideshow_id].showBigShot();
                        return false
                    }
                    window["BigShot_" + HPConfig.slideshow_id].nextPhoto();
                    break;
                case"esc":
                    if (window.slideshow_fullscreen) {
                        window["BigShot_" + HPConfig.slideshow_id].closeSlideShow();
                        return 
                    }
                    break;
                case"fullscreen":
                    if (!window.slideshow_fullscreen) {
                        window["BigShot_" + HPConfig.slideshow_id].openSlideShow();
                        return 
                    }
                    break
                }
                return false
            }
        }
    }
}
var HPHotKeys = {
    init: function() {
        var f = new YAHOO.util.KeyListener(document, {
            ctrl: false,
            keys: 37
        }, function(h, g) {
            getSlide("prev", (g[1].srcElement || g[1].target).nodeName)
        });
        var e = new YAHOO.util.KeyListener(document, {
            ctrl: false,
            keys: 39
        }, function(h, g) {
            getSlide("next", (g[1].srcElement || g[1].target).nodeName)
        });
        f.enable();
        e.enable();
        if (HPConfig.slideshow_type == "fullscreen") {
            var d = new YAHOO.util.KeyListener(document, {
                ctrl: false,
                keys: 27
            }, function(h, g) {
                getSlide("esc", (g[1].srcElement || g[1].target).nodeName)
            });
            var c = new YAHOO.util.KeyListener(document, {
                ctrl: false,
                keys: 38
            }, function(h, g) {
                getSlide("up", (g[1].srcElement || g[1].target).nodeName)
            });
            var a = new YAHOO.util.KeyListener(document, {
                ctrl: false,
                keys: 40
            }, function(h, g) {
                getSlide("down", (g[1].srcElement || g[1].target).nodeName)
            });
            d.enable();
            c.enable();
            a.enable();
            if (!HPBrowser.isIEOld()) {
                var b = new YAHOO.util.KeyListener(document, {
                    ctrl: false,
                    keys: 70
                }, function(h, g) {
                    getSlide("fullscreen", (g[1].srcElement || g[1].target).nodeName)
                });
                b.enable()
            }
        }
    }
};
huff.unit("tmp/hot_keys", function(a) {
    a(HPHotKeys)
});

/* From: production-mt-wfe-58-160-use1 : 14147 */
/* 7926f30f9fc7acf803f8ecc84650ac50ef058260 */
