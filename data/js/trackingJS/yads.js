window.YAHOO = window.YAHOO || {};
window.YAHOO.JP = window.YAHOO.JP || {};
window.YAHOO.JP.anemos = window.YAHOO.JP.anemos || {};
window.YAHOO.JP.anemos.punycode = (function() {
    var z = function(e) {
        var c, b, a, d, f = encoded = "";
        for (c = 0, b = e.length, d = 0; c < b; c++) {
            a = e.charCodeAt(c);
            if (a == 47) {
                break
            }
            if (d == 0) {
                if (a < 65 || a > 90 && a < 97 || a > 122) {
                    if (c + 3 < b && e.substr(c, 3) == "://") {
                        c += 2
                    }
                    d = 1
                }
            }
        }
        if (c != 0) {
            d = D(e.substr(0, c));
            if (d == "") {
                return ""
            }
        } else {
            d = ""
        }
        if (b != c) {
            d += A(e.substr(c, b - c))
        }
        f = d;
        for (c = 0, b = f.length; c < b; c++) {
            a = f.charCodeAt(c);
            if (a <= 126) {
                encoded += f.charAt(c)
            } else {
                d = "0" + a.toString(16);
                encoded += "%" + d.substr(d.length - 2, 2)
            }
        }
        encoded = x(encoded);
        return encoded
    };
    var D = function(a) {
        var f, i, b, e, d, c = "", h;
        for (f = 0, i = a.length, e = d = 0; f < i; f++) {
            b = a.charCodeAt(f);
            if (b != 45 && b < 48 || b > 57 && b < 65 || b > 90 && b < 97 || b > 122 && b <= 255) {
                if (e != 0) {
                    var g = a.substr(d, f - d);
                    if (e == 2) {
                        h = B(g);
                        if (h == "") {
                            return ""
                        }
                        g = "xn--" + h;
                        g = g.toLowerCase()
                    }
                    c += g;
                    d = f;
                    e = 0
                }
            } else {
                if (e == 0) {
                    c += a.substr(d, f - d);
                    d = f;
                    e = 1
                }
                if (b > 255) {
                    e = 2
                }
            }
        }
        if (e != 2) {
            c += a.substr(d, f - d)
        } else {
            h = B(a.substr(d, f - d));
            if (h == "") {
                return ""
            }
            g = "xn--" + h;
            g = g.toLowerCase();
            c += g
        }
        return c
    };
    var B = function(f) {
        if (typeof(f) == "string") {
            var b = f;
            f = new Array;
            for (var c = 0; c < b.length; c++) {
                f.push(b.charCodeAt(c))
            }
        }
        var d = w(f);
        if (d == "") {
            return ""
        }
        for (var e = 0; e < d.length; ++e) {
            var a = d[e];
            if (a >= 0 && a <= 127) {
                d[e] = String.fromCharCode(a)
            } else {
                break
            }
        }
        return d.join("")
    };
    var H = 72;
    var u = 128;
    var v = 45;
    var E = 2147483647;
    var G = 36;
    var I = 1;
    var C = 26;
    var t = 38;
    var F = 700;
    var s = function(a) {
        return a < 26 ? a + 97 : a + 22
    };
    var y = function(a, b, c) {
        var d;
        a = c ? parseInt(a / F) : a>>1;
        a += parseInt(a / b);
        for (d = 0; a > ((G - I) * C)>>1; d += G) {
            a = parseInt(a / (G - I))
        }
        return parseInt(d + (G - I + 1) * a / (a + t))
    };
    var w = function(d) {
        var g = new Array;
        var k = u;
        var c = 0;
        var e = H;
        for (j = 0; j < d.length; ++j) {
            if (d[j] < 128) {
                g.push(d[j])
            }
        }
        var a = g.length;
        var f = a;
        if (a > 0) {
            g.push(v)
        }
        while (f < d.length) {
            var i = E;
            for (j = 0; j < d.length; ++j) {
                if (d[j] >= k && d[j] < i) {
                    i = d[j]
                }
            }
            if (i - k > (E - c) / (f + 1)) {
                return ""
            }
            c += (i - k) * (f + 1);
            k = i;
            for (j = 0; j < d.length; ++j) {
                if (d[j] < k&&++c == 0) {
                    return ""
                }
                if (d[j] == k) {
                    var l = c;
                    for (var h = G; ; h += G) {
                        var b = h <= e ? I: h >= e + C ? C: h - e;
                        if (l < b) {
                            break
                        }
                        g.push(s(b + (l - b)%(G - b)));
                        l = parseInt((l - b) / (G - b))
                    }
                    g.push(s(l));
                    e = y(c, f + 1, f == a);
                    c = 0;
                    ++f
                }
            }
            ++c, ++k
        }
        return g
    };
    var A = function(c) {
        var a, b = "";
        for (var d = 0; d < c.length; d++) {
            a = c.charCodeAt(d);
            if (a <= 127) {
                b += c.charAt(d)
            } else {
                if (a >= 128 && a <= 2047) {
                    b += String.fromCharCode(((a>>6) & 31) | 192);
                    b += String.fromCharCode((a & 63) | 128)
                } else {
                    b += String.fromCharCode((a>>12) | 224);
                    b += String.fromCharCode(((a>>6) & 63) | 128);
                    b += String.fromCharCode((a & 63) | 128)
                }
            }
        }
        return b
    };
    var x = function(e) {
        if (window.encodeURIComponent) {
            return encodeURIComponent(e)
        } else {
            var a, c, b, d;
            a = "";
            for (c = 0; c < e.length; c++) {
                b = e.charAt(c);
                d = e.charCodeAt(c);
                if (b == " ") {
                    a += "+"
                } else {
                    if (d == 42 || d == 45 || d == 46 || d == 95 || ((d >= 48) && (d <= 57)) || ((d >= 65) && (d <= 90)) || ((d >= 97) && (d <= 122))) {
                        a = a + b
                    } else {
                        if ((d >= 0) && (d <= 127)) {
                            b = "0" + d.toString(16);
                            a += "%" + b.substr(b.length - 2)
                        } else {
                            if (d > 1048575) {
                                a += "%" + (240 + ((d & 1835008)>>18)).toString(16);
                                a += "%" + (128 + ((d & 258048)>>12)).toString(16);
                                a += "%" + (128 + ((d & 4032)>>6)).toString(16);
                                a += "%" + (128 + (d & 63)).toString(16)
                            } else {
                                if (d > 2047) {
                                    a += "%" + (224 + ((d & 61440)>>12)).toString(16);
                                    a += "%" + (128 + ((d & 4032)>>6)).toString(16);
                                    a += "%" + (128 + (d & 63)).toString(16)
                                } else {
                                    a += "%" + (192 + ((d & 1984)>>6)).toString(16);
                                    a += "%" + (128 + (d & 63)).toString(16)
                                }
                            }
                        }
                    }
                }
            }
            return a
        }
    };
    return {
        encodeURL: z
    }
}());
if (typeof(YJ_YADS) == "undefined") {
    var YJ_YADS = {}
}
window.YAHOO = window.YAHOO || {};
window.YAHOO.JP = window.YAHOO.JP || {};
window.YAHOO.JP.anemos = window.YAHOO.JP.anemos || {};
window.YAHOO.JP.anemos.yads = window.YAHOO.JP.anemos.yads || {};
YJ_YADS.gAdController = window.YJ_YADS.gAdController || {
    objs: [],
    curProd: 0,
    bind: function(b, a) {
        return function() {
            b.apply(a, arguments)
        }
    },
    isBeaconUrl: function(a) {
        var b = /^(https?:)?\/\/yadsbeacon\.yahoo\.co\.jp\/imps/;
        return b.test(a)
    },
    loadBeacon: function() {
        if (typeof(this.objs[this.curProd]) == "undefined" ||!this.isBeaconUrl(this.objs[this.curProd].ib_url)) {
            return 
        }
        var a = document.createElement("IMG");
        a.src = this.objs[this.curProd].ib_url
    },
    notifyFrameLoaded: function() {
        if (window == window.top ||!window.top.postMessage) {
            return 
        }
        function c(i) {
            return /^https?:\/\/(yads\.yahoo\.co\.jp|yads\.c\.yimg\.jp)\//.test(i)
        }
        function a(i) {
            return i.indexOf("http://rd.ane.yahoo.co.jp/rd?") === 0
        }
        var k = window.yads_vimps_hint;
        if (k) {
            var n = k.vimpsMode;
            var e = k.adProdSetCode;
            var p = k.requestId;
            var o = "";
            if (k.viewableInfo && k.viewableInfo.length > 0) {
                var f = "data-ydntxt-item";
                var h = [];
                if (document.querySelectorAll) {
                    var w = document.querySelectorAll("[" + f + "]");
                    var s = w.length;
                    for (var r = 0; r < s; r++) {
                        h.push(w[r])
                    }
                } else {
                    var d = document.getElementsByTagName("*");
                    var s = d.length;
                    for (var r = 0; r < s; r++) {
                        if (d[r].hasAttribute(f)) {
                            h.push(d[r])
                        }
                    }
                }
                var q = [];
                if (h.length > 0) {
                    var s = h.length;
                    for (var r = 0; r < s; r++) {
                        var v = h[r].getAttribute(f);
                        var t = parseInt(v, 10);
                        if (isNaN(t) || t === 0) {} else {
                            q[(t - 1)] = h[r]
                        }
                    }
                    if (q.length !== h.length) {
                        q = h
                    }
                } else {
                    var l = document.getElementsByTagName("a");
                    var s = l.length;
                    for (var r = 0; r < s; r++) {
                        if (a(l[r].href)) {
                            q.push(l[r])
                        }
                    }
                    if (q.length === 0) {
                        var m = document.getElementsByTagName("iframe");
                        var s = m.length;
                        for (var r = 0; r < s; r++) {
                            if (c(m[r].src)) {
                                q.push(m[r])
                            }
                        }
                    }
                }
                if (k.viewableInfo.length === q.length) {
                    var s = q.length;
                    for (var r = 0; r < s; r++) {
                        var u = q[r];
                        var b = u.getBoundingClientRect();
                        if (o.length > 0) {
                            o += "|"
                        }
                        o += Math.round(b.top) + "/" + Math.round(b.right) + "/" + Math.round(b.bottom) + "/" + Math.round(b.left) + "/" + k.viewableInfo[r]
                    }
                } else {
                    o = "ERR001"
                }
            } else {
                o = "ERR002"
            }
            var g = n + "," + e + "," + p + "," + o;
            window.top.postMessage("frameLoaded," + g, "*")
        } else {
            var e = (function() {
                var i = location.search.match(/[\?&]s=([^&]+)/);
                if (i && i.length === 2) {
                    return i[1]
                }
                return ""
            })();
            window.top.postMessage("frameLoaded,-1," + e + ",non_ydn,", "*")
        }
    },
    getAdElements: function(e, a) {
        if (e.getElementsByClassName) {
            return e.getElementsByClassName(a)
        } else {
            var b = [];
            var f = e.all ? e.all: e.getElementsByTagName("*");
            for (var c = 0, d = f.length; c < d; c++) {
                if (f.item(c).className == a) {
                    b.push(f.item(c))
                }
            }
            return b
        }
    },
    needToErasePr: function() {
        var a = this.getAdElements(document, "yads_ydn_frame_image");
        var b = this.getAdElements(document, "yads_ydn_frame_text");
        if (a.length == 0) {
            if (b.length == 0) {
                return false
            }
            return true
        } else {
            return false
        }
    },
    switchPrTag: function() {
        var a = document.getElementById("yads-pr-label");
        if (!a || typeof(this.objs[this.curProd]) == "undefined") {
            return 
        }
        if (this.objs[this.curProd].pr_flag == "1") {
            a.style.display = "block"
        } else {
            a.style.display = "none";
            return 
        }
        if (this.needToErasePr()) {
            a.style.display = "none"
        }
    },
    stringifyMessage: function(a) {
        switch (a.cmd) {
        case"eraseFrame":
            return a.cmd + "," + a.frameSrc;
        case"resizeFrame":
            return a.cmd + "," + a.width + "," + a.height + "," + a.frameSrc;
        default:
            return ""
        }
    },
    parseMessage: function(b) {
        var c, a;
        c = b.split(",");
        a = c[0];
        switch (a) {
        case"eraseFrame":
            return {
                cmd: "eraseFrame",
                frameSrc: c.slice(1).join()
            };
        case"resizeFrame":
            return {
                cmd: "resizeFrame",
                width: c[1],
                height: c[2],
                frameSrc: c.slice(3).join()
            };
        default:
            return {
                cmd: ""
            }
        }
    },
    onMessageCallback: function(b) {
        var c, d, a;
        d = /yahoo\.co\.jp(:[0-9]+)?$/;
        if (!(b.origin === "http://i.yimg.jp" || b.origin === "https://s.yimg.jp" || d.test(b.origin))) {
            return 
        }
        a = gAdController.parseMessage(b.data);
        switch (a.cmd) {
        case"eraseFrame":
            YJ_YADS.gAdController.eraseFrame(a, b.source);
            break;
        case"resizeFrame":
            YJ_YADS.gAdController.resizeFrame(a, b.source);
            break;
        default:
            break
        }
    },
    resizeFrame: function(a, c) {
        var b = this.findFrame(c);
        if (b) {
            b.width = a.width;
            b.height = a.height
        }
    },
    eraseFrame: function(a, d) {
        var c = this.findFrame(d), b;
        if (c) {
            c.style.display = "none";
            b = this.cache.yads_noad_hook[c.id];
            YJ_YADS.innerFuncs.executeCallback(b, [{
                reason: 2
            }
            ])
        }
    },
    findFrame: function(d) {
        var c = document.getElementsByTagName("IFRAME");
        for (var b = 0, a = c.length; b < a; b++) {
            if (d == c[b].contentWindow) {
                return c[b]
            }
        }
    },
    start: function(f, c, e) {
        this.objs = f;
        this.prodNums = this.objs.length;
        this.curProd = 0;
        if (c && window != window.top && window.top.postMessage) {
            var d = c.vimps_mode;
            var b = c.pattern_code;
            window.top.postMessage("frameStarted," + d + "," + b, "*")
        }
        if (this.prodNums > 0) {
            var h = this.bind(this.adDone, this);
            if (window.addEventListener) {
                window.addEventListener("load", h, false)
            } else {
                if (window.attachEvent) {
                    window.attachEvent("onload", h)
                }
            }
            var g = this.objs[this.curProd];
            if (window != window.parent) {
                if (typeof(g.width) != "undefined" && typeof(g.height) != "undefined") {
                    var a = g.height;
                    if (g.pr_flag == 1 && a.indexOf("%")==-1) {
                        a = parseInt(a) + 15
                    }
                    window.parent.postMessage(this.stringifyMessage({
                        cmd: "resizeFrame",
                        frameSrc: window.location.href,
                        width: g.width,
                        height: a
                    }), "*")
                }
            }
            document.write(g.ad_tag)
        } else {
            this.passback()
        }
    },
    cache: (function() {
        if (YJ_YADS.gAdController && YJ_YADS.gAdController.cache) {
            return YJ_YADS.gAdController.cache
        } else {
            return {}
        }
    })(),
    passback: function() {
        var b;
        this.curProd++;
        if (this.curProd >= this.prodNums) {
            if (window != window.parent) {
                window.parent.postMessage(this.stringifyMessage({
                    cmd: "eraseFrame",
                    frameSrc: window.location.href
                }), "*")
            }
            return 
        }
        b = this.objs[this.curProd];
        if (window != window.parent) {
            if (typeof(b.width) != "undefined" && typeof(b.height) != "undefined") {
                var a = b.height;
                if (b.pr_flag == 1 && a.indexOf("%")==-1) {
                    a = parseInt(a) + 15
                }
                window.parent.postMessage(this.stringifyMessage({
                    cmd: "resizeFrame",
                    frameSrc: window.location.href,
                    width: b.width,
                    height: a
                }), "*")
            }
        }
        document.write(b.ad_tag)
    },
    adDone: function() {
        this.switchPrTag();
        this.loadBeacon();
        this.notifyFrameLoaded()
    }
};
window.gCRITEO = YJ_YADS.gAdController;
window.gAdController = YJ_YADS.gAdController;
window.YAHOO.JP.anemos.yads.gAdController = YJ_YADS.gAdController;
YJ_YADS.openIframe = function(b, a) {
    if (a && document.getElementById(a)) {
        document.getElementById(a).innerHTML = b
    } else {
        document.write(b)
    }
};
window.yadsOpenIframe = YJ_YADS.openIframe;
window.YAHOO.JP.anemos.yads.openIframe = YJ_YADS.openIframe;
YJ_YADS.requestAdTag = function(b, a) {
    YJ_YADS.gAdController.start(b, a, 1)
};
window.yadsRequestAdTag = YJ_YADS.requestAdTag;
window.YAHOO.JP.anemos.yads.requestAdTag = YJ_YADS.requestAdTag;
YJ_YADS.renderAd = function(e, r, l, v, k, b, u) {
    var g = "";
    var p = true;
    if (e) {
        var f = e.length;
        for (var n = 0; n < f; n++) {
            if (e[n] != "") {
                p = false;
                break
            }
        }
    }
    if (p && l == "js_frame") {
        if (typeof(window.gAdController) != "undefined") {
            window.gAdController.passback()
        }
        return 
    }
    if (p && u && (typeof(u) == "string")) {
        YJ_YADS.innerFuncs.executeCallback(u, [{
            reason: 2
        }
        ]);
        return 
    }
    function t() {
        try {
            var x = window.top.YJ_UADF;
            if (!x ||!x.YADSViewable) {
                return 
            }
            var w = window.top.YJ_UADF.YADSViewable
        } catch (y) {
            return 
        }
        var i = (function() {
            var B = document.getElementsByTagName("script");
            var D = B.length - 1;
            while (D >= 0) {
                var C = B[D];
                if (/\/\/yads\.yahoo\.co\.jp\/(oi|tag)\?/.test(C.src)) {
                    var A = C.src.match(/[\?&]s=([^&]+)/);
                    if (A && A.length === 2) {
                        return [C, A[1]]
                    }
                }
                D--
            }
            return null
        })();
        if (!i) {
            return 
        }
        var z = {
            tagElem: i[0],
            tagId: i[1]
        };
        if (k) {
            if (r) {
                z.type = "multi";
                z.parentIdList = r
            } else {
                if (v) {
                    z.type = "single_with_parent";
                    z.parentId = v
                } else {
                    z.type = "doc_write"
                }
            }
        } else {
            if (l === "js_single") {
                if (v) {
                    z.type = "single_with_parent";
                    z.parentId = v
                } else {
                    z.type = "doc_write"
                }
            } else {
                if (l === "js_multi") {
                    z.type = "multi";
                    z.parentIdList = r
                } else {
                    return 
                }
            }
        }
        w.notifyRendered(z)
    }
    function s() {
        if (k && (typeof(k) == "string")) {
            var w = k.split(".");
            var y = window;
            for (var x = 0; x < w.length; x++) {
                if (!(y = y[w[x]])) {
                    return 
                }
            }
            if (typeof(y) == "function") {
                if (g) {
                    y(e, r, l, g)
                } else {
                    y(e, r, l, v)
                }
            }
            t();
            return 
        }
        if (l == "js_frame") {
            if (e.length == 0) {
                return 
            }
            document.write(e[0])
        } else {
            if (l == "js_single") {
                if (e.length == 0) {
                    return 
                }
                if (v && document.getElementById(v)) {
                    document.getElementById(v).innerHTML = e[0]
                } else {
                    document.write(e[0])
                }
                t()
            } else {
                if (l == "js_multi") {
                    for (var x = 0; x < r.length; ++x) {
                        if (document.getElementById(r[x]) && e[x]) {
                            document.getElementById(r[x]).innerHTML = e[x]
                        }
                    }
                    t()
                }
            }
        }
    }
    if (b) {
        var h = location.protocol == "https:" ? "https://s.yimg.jp": "http://i.yimg.jp";
        var m = h + "/images/listing/tool/yads/";
        var d = Math.floor(Math.random() * 999999999);
        var o = m + b;
        var c = "yads_preloaded_js_" + d;
        if (l == "js_frame") {
            g = "yads_dummy_parent_" + d;
            document.write('<div id="' + g + '"></div>');
            document.write('<script id="' + c + '" type="text/javascript" src="' + o + '"><\/script>')
        } else {
            if (l == "js_single") {
                if (v && document.getElementById(v)) {
                    var a = document.createElement("SCRIPT");
                    a.id = c;
                    a.type = "text/javascript";
                    a.src = o;
                    document.body.appendChild(a)
                } else {
                    g = "yads_dummy_parent_" + d;
                    document.write('<div id="' + g + '"></div>');
                    document.write('<script id="' + c + '" type="text/javascript" src="' + o + '"><\/script>')
                }
            } else {
                if (l == "js_multi") {
                    var a = document.createElement("SCRIPT");
                    a.id = c;
                    a.type = "text/javascript";
                    a.src = o;
                    document.body.appendChild(a)
                }
            }
        }
        var q = document.getElementById(c);
        if (window.ActiveXObject) {
            q.onreadystatechange = function() {
                if (q.readyState == "complete" || q.readyState == "loaded") {
                    s()
                }
            }
        } else {
            q.onload = function() {
                s()
            }
        }
    } else {
        s()
    }
};
window.yadsRenderAd = YJ_YADS.renderAd;
window.YAHOO.JP.anemos.yads.renderAd = YJ_YADS.renderAd;
YJ_YADS.renderAd_v2 = function(e, p) {
    var r = "";
    if (!p) {
        return 
    }
    var l = p.output_type;
    var h = p.callback;
    var d = p.js_file_name;
    var v = p.noad_callback;
    var n = p.adprodset_code;
    var u = p.request_id;
    var t = p.vimps_mode;
    var k = "";
    var o = [];
    var g = [];
    var D = [];
    var B = p.tracking_url;
    var z = true;
    if (e) {
        var m = e.length;
        for (var x = 0; x < m; x++) {
            if (e[x] != "") {
                z = false;
                break
            }
        }
    }
    if (z && l == "js_frame") {
        if (typeof(window.gAdController) != "undefined") {
            window.gAdController.passback()
        }
        return 
    }
    if (z && v && (typeof(v) == "string")) {
        YJ_YADS.innerFuncs.executeCallback(v, [{
            reason: 2
        }
        ]);
        return 
    }
    for (var x = 0, q = p.pos_info.length; x < q; x++) {
        o.push(p.pos_info[x].p_elem);
        g.push(p.pos_info[x].ib_url);
        if (p.pos_info[x].viewable_info) {
            D.push(p.pos_info[x].viewable_info)
        }
    }
    if (l == "js_single" || l == "js_frame") {
        if (o[0] && o[0] != "") {
            k = o[0]
        } else {
            o = null
        }
    }
    function E() {
        try {
            var F = window.top.YJ_UADF;
            if (!F ||!F.YADSViewable) {
                return 
            }
            var i = window.top.YJ_UADF.YADSViewable
        } catch (G) {
            return 
        }
        var H = {
            vimpsMode: t,
            adProdSetCode: n,
            requestId: u
        };
        if (h && h.length > 0) {
            if (o && o.length > 1) {
                H.type = "multi";
                H.parentIdList = o;
                H.viewableInfoList = D
            } else {
                if (k) {
                    H.type = "single_with_parent";
                    H.parentId = k;
                    H.viewableInfo = D[0]
                } else {
                    H.type = "doc_write";
                    H.viewableInfo = D[0]
                }
            }
        } else {
            if (l === "js_single") {
                if (k) {
                    H.type = "single_with_parent";
                    H.parentId = k
                } else {
                    H.type = "doc_write"
                }
                H.viewableInfo = D[0]
            } else {
                if (l === "js_multi") {
                    H.type = "multi";
                    H.parentIdList = o;
                    H.viewableInfoList = D
                } else {
                    return 
                }
            }
        }
        if (H.type === "doc_write") {
            H.docWriteCaller = (function() {
                var I = document.getElementsByTagName("script");
                var K = I.length - 1;
                while (K >= 0) {
                    var J = I[K];
                    if (/\/\/yads\.yahoo\.co\.jp\/(oi|tag)\?/.test(J.src)) {
                        return J
                    }
                    K--
                }
                return null
            })()
        }
        i.notifyRendered(H)
    }
    function f() {
        var i = {
            type: "frame",
            vimpsMode: t,
            adProdSetCode: n,
            requestId: u,
            viewableInfo: D[0]
        };
        if (i.viewableInfo && i.viewableInfo.length > 0) {
            window.yads_vimps_hint = i
        }
    }
    function C() {
        function G(L) {
            for (var P = 0, N = L.length; P < N; P++) {
                var O = L[P];
                if (typeof(O) == "undefined" ||!window.gAdController.isBeaconUrl(O)) {
                    continue
                }
                var M = document.createElement("IMG");
                M.src = O
            }
        }
        function J(T) {
            if (!T || T.length == 0) {
                return 
            }
            var L = (window.navigator.userAgent.toLowerCase().indexOf("msie")>-1);
            var S = location.protocol;
            for (var P = 0, M = T.length; P < M; P++) {
                var R = T[P];
                if (typeof(R) == "undefined") {
                    continue
                }
                var U = (new RegExp("^(https?:)(//.*)")).exec(R);
                if (!U) {
                    continue
                }
                var V = U[1];
                var Q = U[2];
                var N;
                if (V === "https:") {
                    if (S === "http:" || S === "https:") {
                        N = S + Q
                    }
                } else {
                    if (V === "http:") {
                        if (S === "http:") {
                            N = R
                        } else {
                            continue
                        }
                    }
                }
                var O = document.createElement("IMG");
                O.src = N;
                if (L) {
                    O.width = 0;
                    O.height = 0;
                    O.style.display = "none";
                    document.body.appendChild(O)
                }
            }
        }
        if (h && (typeof(h) == "string")) {
            var F = h.split(".");
            var K = window;
            for (var I = 0, H = F.length; I < H; I++) {
                if (!(K = K[F[I]])) {
                    return 
                }
            }
            if (typeof(K) == "function") {
                if (r) {
                    K(e, o, l, r)
                } else {
                    K(e, o, l, k)
                }
                G(g);
                J(B)
            }
            if (l == "js_single" || l == "js_multi") {
                E()
            } else {
                if (l == "js_frame") {
                    f()
                }
            }
            return 
        }
        if (l == "js_frame") {
            if (e.length == 0) {
                return 
            }
            document.write(e[0]);
            G(g);
            J(B);
            f()
        } else {
            if (l == "js_single") {
                if (e.length == 0) {
                    E();
                    return 
                }
                if (k && document.getElementById(k)) {
                    document.getElementById(k).innerHTML = e[0]
                } else {
                    document.write(e[0])
                }
                G(g);
                J(B);
                E()
            } else {
                if (l == "js_multi") {
                    for (var I = 0; I < o.length; ++I) {
                        if (document.getElementById(o[I]) && e[I]) {
                            document.getElementById(o[I]).innerHTML = e[I]
                        }
                    }
                    G(g);
                    J(B);
                    E()
                }
            }
        }
    }
    if (d) {
        var s = location.protocol == "https:" ? "https://s.yimg.jp": "http://i.yimg.jp";
        var w = s + "/images/listing/tool/yads/";
        var c = Math.floor(Math.random() * 999999999);
        var y = w + d;
        var b = "yads_preloaded_js_" + c;
        if (l == "js_frame") {
            r = "yads_dummy_parent_" + c;
            document.write('<div id="' + r + '"></div>');
            document.write('<script id="' + b + '" type="text/javascript" src="' + y + '"><\/script>')
        } else {
            if (l == "js_single") {
                if (k && document.getElementById(k)) {
                    var a = document.createElement("SCRIPT");
                    a.id = b;
                    a.type = "text/javascript";
                    a.src = y;
                    document.body.appendChild(a)
                } else {
                    r = "yads_dummy_parent_" + c;
                    document.write('<div id="' + r + '"></div>');
                    document.write('<script id="' + b + '" type="text/javascript" src="' + y + '"><\/script>')
                }
            } else {
                if (l == "js_multi") {
                    var a = document.createElement("SCRIPT");
                    a.id = b;
                    a.type = "text/javascript";
                    a.src = y;
                    document.body.appendChild(a)
                }
            }
        }
        var A = document.getElementById(b);
        if (window.ActiveXObject) {
            A.onreadystatechange = function() {
                if (A.readyState == "complete" || A.readyState == "loaded") {
                    C()
                }
            }
        } else {
            A.onload = function() {
                C()
            }
        }
    } else {
        C()
    }
};
window.yadsRenderAd_v2 = YJ_YADS.renderAd_v2;
window.YAHOO.JP.anemos.yads.renderAd_v2 = YJ_YADS.renderAd_v2;
YJ_YADS.renderStaticContent = function(a) {
    if (!a ||!a.html) {
        return 
    }
    if (a.p_elem && document.getElementById(a.p_elem)) {
        document.getElementById(a.p_elem).innerHTML += a.html
    } else {
        document.write(a.html)
    }
};
window.yadsRenderStaticContent = YJ_YADS.renderStaticContent;
window.YAHOO.JP.anemos.yads.renderStaticContent = YJ_YADS.renderStaticContent;
YJ_YADS.handleError = function(a) {
    if (typeof(console) != "undefined") {
        console.log(a)
    }
    return 
};
window.yadsHandleError = YJ_YADS.handleError;
window.YAHOO.JP.anemos.yads.handleError = YJ_YADS.handleError;
YJ_YADS.innerFuncs = YJ_YADS.innerFuncs || {
    _findSpaceId: function(c) {
        for (var b = 0, a = c.childNodes.length; b < a; ++b) {
            var e = c.childNodes.item(b);
            switch (e.nodeType) {
            case 1:
                var d = this._findSpaceId(e);
                if (d) {
                    return d
                }
                break;
            case 8:
                if (e.data.match(/SpaceID=([0-9]+)/)) {
                    return RegExp.$1
                }
                break;
            default:
            }
        }
        return null
    },
    findSpaceId: function() {
        if (window.yads_ad_space_cache != null) {
            return window.yads_ad_space_cache
        }
        if (typeof(window.yads_ad_space) !== "undefined") {
            return window.yads_ad_space
        }
        if (location.href.indexOf("http") == 0 && location.href.indexOf(".yahoo.co.jp") > 0 && document.body) {
            if (window.yads_ad_space_cache = this._findSpaceId(document.body)) {
                return window.yads_ad_space_cache
            } else {
                return ""
            }
        } else {
            return ""
        }
    },
    findLastScript: function(a) {
        if (a.nodeName.toUpperCase() == "SCRIPT") {
            return a
        } else {
            if (a.lastChild) {
                return this.findLastScript(a.lastChild)
            } else {
                return null
            }
        }
    },
    getTagPos: function(c) {
        if (!document.body) {
            return ""
        }
        var b = document.createElement("SPAN");
        if (window.yads_parent_element && document.getElementById(window.yads_parent_element)) {
            document.getElementById(window.yads_parent_element).appendChild(b)
        } else {
            var a = this.findLastScript(document);
            if (!a) {
                return ""
            }
            a.parentNode.appendChild(b)
        }
        var d = b.getBoundingClientRect();
        var e = {
            l: Math.round(d.left + (document.body.scrollLeft || document.documentElement.scrollLeft)),
            t: Math.round(d.top + (document.body.scrollTop || document.documentElement.scrollTop))
        };
        if (c) {
            return e.l + "x" + e.t
        } else {
            return e
        }
    },
    uaIsIE: function(a) {
        var d = window.navigator;
        if (d.userAgent.toLowerCase().indexOf("msie")>-1) {
            for (var c = 0, b = a.length; c < b; c++) {
                if (d.appVersion.toLowerCase().indexOf("msie " + a[c])>-1) {
                    return true
                }
            }
        }
        return false
    },
    uaIsFirefox: function() {
        var a = window.navigator;
        if (a.userAgent.toLowerCase().indexOf("firefox")>-1) {
            return true
        }
        return false
    },
    iframeIsEnabled: function() {
        return !this.uaIsIE([2, 3, 4, 5])
    },
    flashIsEnabled: function() {
        if (typeof(ActiveXObject) == "undefined") {
            try {
                return navigator.mimeTypes["application/x-shockwave-flash"].enabledPlugin ? true : false
            } catch (b) {
                return false
            }
        } else {
            try {
                var a = new ActiveXObject("ShockwaveFlash.ShockwaveFlash")
            } catch (b) {
                return false
            }
            return true
        }
    },
    enabledFlashVer: function() {
        if (typeof(ActiveXObject) == "undefined") {
            try {
                var a = navigator.mimeTypes["application/x-shockwave-flash"].enabledPlugin;
                return parseInt(a.description.match(/[0-9]+\./))
            } catch (b) {
                return 0
            }
        } else {
            try {
                var a = new ActiveXObject("ShockwaveFlash.ShockwaveFlash");
                return parseInt(a.GetVariable("$version").match(/[0-9]+\,/))
            } catch (b) {
                return 0
            }
        }
    },
    getTimestamp: function() {
        return parseInt((new Date).getTime())
    },
    encodeLocation: function(a) {
        return YAHOO.JP.anemos.punycode.encodeURL(a)
    },
    getLocation: function() {
        var b = location.href;
        if (window.parent != window) {
            var a = document.referrer;
            if (a) {
                b = a
            }
        }
        return this.encodeLocation(b)
    },
    getPvTs: function() {
        var a = (function() {
            var b = window;
            while (b.top != b) {
                try {
                    var c = b.parent.document.URL
                } catch (d) {
                    break
                }
                b = b.parent
            }
            return b
        })();
        if (typeof(a.yads_pv_timestamp) == "undefined") {
            a.yads_pv_timestamp = yadsInnerFuncs.getTimestamp() + "-" + Math.floor(Math.random() * 10000000)
        }
        return a.yads_pv_timestamp
    },
    getFrId: function() {
        var a = 0;
        if (typeof(gAdController.cache.yads_sequence_num) != "undefined") {
            a = gAdController.cache.yads_sequence_num
        }
        return "yads_" + Math.floor(Math.random() * 10000000) + "-" + a
    },
    executeCallback: function(f, e) {
        if (!f || typeof(f) != "string") {
            return 
        }
        var a = f.split(".");
        var c = a.length;
        var d = window;
        for (var b = 0; b < c; b++) {
            if (!(d = d[a[b]])) {
                return 
            }
        }
        if (typeof(d) == "function") {
            d.apply(this, e)
        }
    },
    resolveRelativeReference: function(b, c) {
        function d(i) {
            var r = i.indexOf("?");
            var s = i.indexOf("#");
            return (r < s) ? r : s
        }
        var o;
        o = (new RegExp("^(https?://)(.*)")).exec(b);
        if (!o) {
            return ""
        }
        var p = o[1];
        var m = o[2];
        var q;
        var h;
        h = d(m);
        if (h < 0) {
            q = m
        } else {
            q = m.substr(0, h)
        }
        var l, g;
        h = d(c);
        if (h < 0) {
            l = c;
            g = ""
        } else {
            l = c.substr(0, h);
            g = c.substr(h)
        }
        var n = q.split("/");
        var f = l.split("/");
        if (!n ||!f) {
            return ""
        }
        var k = n[0];
        var a = [];
        if (n.length > 1) {
            a = n.slice(1, n.length - 1)
        }
        for (var e = 0; e < f.length; e++) {
            if (f[e] == "") {
                if (e == 0) {
                    a = []
                } else {
                    break
                }
            } else {
                if (f[e] == ".") {
                    continue
                } else {
                    if (f[e] == "..") {
                        if (a.length > 0) {
                            a = a.slice(0, a.length - 1)
                        }
                    } else {
                        a.push(f[e])
                    }
                }
            }
        }
        return p + k + "/" + a.join("/") + g
    },
    getCanonicalUrl: function() {
        var a = (function() {
            var f = document.getElementsByTagName("link");
            if (!f || f.length == 0) {
                return ""
            }
            for (var g = 0, e = f.length; g < e; g++) {
                if (f[g].rel == "canonical" && typeof(f[g].href) != "undefined") {
                    return f[g].href
                }
            }
            return ""
        })();
        if (a == "" || a.indexOf("?") == 0 || a.indexOf("#") == 0) {
            return ""
        }
        var c = (new RegExp("^(.*?)://")).exec(a);
        if (c && c.length == 2) {
            var b = c[1];
            if (b == "http" || b == "https") {
                return a
            } else {
                return ""
            }
        }
        var d = (function() {
            var e = document.getElementsByTagName("base");
            if (e.length == 0) {
                return ""
            }
            if (typeof(e[0].href) == "undefined") {
                return ""
            }
            return e[0].href
        })();
        if (d == "") {
            d = location.href
        } else {
            if (!(new RegExp("^https?://")).test(d)) {
                return ""
            }
        }
        return YJ_YADS.innerFuncs.resolveRelativeReference(d, a)
    }
};
window.yadsInnerFuncs = YJ_YADS.innerFuncs;
window.YAHOO.JP.anemos.yads.innerFuncs = YJ_YADS.innerFuncs;
YJ_YADS.InputParameter = function() {
    this.PARAMETER_LIST = [{
        name: "s",
        tagName: "yads_ad_ds",
        init: ""
    }, {
        name: "t",
        init: "j"
    }, {
        name: "ssl",
        init: (location.protocol == "https:" ? "1" : "0")
    }, {
        name: "fr_id",
        init: YJ_YADS.innerFuncs.getFrId()
    }, {
        name: "p_elem",
        tagName: "yads_parent_element",
        init: ""
    }, {
        name: "noad_cb",
        tagName: "yads_noad_callback",
        init: ""
    }, {
        name: "xd_support",
        init: (window.postMessage ? "1" : "0")
    }, {
        name: "fl_support",
        init: YJ_YADS.innerFuncs.enabledFlashVer()
    }, {
        name: "fr_support",
        init: (YJ_YADS.innerFuncs.iframeIsEnabled() ? "1" : "0")
    }, {
        name: "enc",
        tagName: "yads_page_encode",
        init: (document.charset || document.characterSet).toUpperCase()
    }, {
        name: "pv_ts",
        init: YJ_YADS.innerFuncs.getPvTs()
    }, {
        name: "tag_path",
        tagName: "yads_entry_tag",
        init: (location.protocol == "https:" ? "https:" : "http:") + "//yads.yahoo.co.jp/tag"
    }, {
        name: "oi_path",
        tagName: "yads_entry_oi",
        init: (location.protocol == "https:" ? "https:" : "http:") + "//yads.yahoo.co.jp/oi"
    }, {
        name: "w",
        tagName: "yads_page_kw",
        init: "",
        noEnc: true
    }, {
        name: "type",
        tagName: "yads_type_tag",
        init: ""
    }, {
        name: "page",
        tagName: "yads_ad_page",
        init: "1"
    }, {
        name: "sid",
        tagName: "yads_ad_space",
        init: YJ_YADS.innerFuncs.findSpaceId()
    }, {
        name: "i",
        tagName: "yads_page_id",
        init: ""
    }, {
        name: "rid",
        tagName: "yads_split_id",
        init: ""
    }, {
        name: "u",
        tagName: "yads_page_url",
        init: YJ_YADS.innerFuncs.getLocation(),
        noEnc: true
    }, {
        name: "cu",
        tagName: "yads_crawl_url",
        init: ""
    }, {
        name: "canu",
        init: (function() {
            var a = YJ_YADS.innerFuncs.getCanonicalUrl();
            if (a.length > 600) {
                return ""
            } else {
                return a
            }
        })()
    }, {
        name: "ref",
        tagName: "yads_ref_url",
        init: document.referrer
    }, {
        name: "debug",
        tagName: "yads_ad_debug",
        init: ""
    }, {
        name: "tflg",
        tagName: "yads_ad_test",
        init: ""
    }, {
        name: "f_path",
        tagName: "yads_iframe_url",
        init: ""
    }, {
        name: "tagpos",
        init: YJ_YADS.innerFuncs.getTagPos(true)
    }, {
        name: "age",
        tagName: "yads_user_age",
        init: ""
    }, {
        name: "gen",
        tagName: "yads_user_gen",
        init: ""
    }
    ];
    this.PARAMETER_NUM = this.PARAMETER_LIST.length;
    this.params = [];
    this.initialized = false
};
YJ_YADS.InputParameter.prototype.init = function(b) {
    if (this.initialized) {
        return 
    }
    if (typeof(b) == "undefined") {
        return 
    }
    this.userInput = b;
    for (var a = 0; a < this.PARAMETER_NUM; a++) {
        var d = this.PARAMETER_LIST[a];
        var c = typeof(d.noEnc) != "undefined" ? d.noEnc: false;
        if (typeof(d.tagName) == "undefined") {
            this.params.push({
                name: d.name,
                val: d.init,
                noEnc: c
            })
        } else {
            var e = b[d.tagName];
            this.params.push({
                name: d.name,
                val: typeof(e) != "undefined" ? e: d.init,
                noEnc: c
            })
        }
    }
    this.initialized = true
};
YJ_YADS.InputParameter.prototype.setParam = function(b, c) {
    if (!this.initialized) {
        return 
    }
    for (var a = 0; a < this.PARAMETER_NUM; a++) {
        if (this.params[a].name == b) {
            this.params[a].val = c
        }
    }
};
YJ_YADS.InputParameter.prototype.getParamByName = function(b) {
    if (!this.initialized) {
        return null
    }
    for (var a = 0; a < this.PARAMETER_NUM; a++) {
        if (this.params[a].name == b) {
            return this.params[a]
        }
    }
    return null
};
YJ_YADS.InputParameter.prototype.getAllParams = function() {
    if (!this.initialized) {
        return null
    }
    return this.params
};
YJ_YADS.InputParameter.prototype.getQueryString = function() {
    if (!this.initialized) {
        return ""
    }
    var b = [];
    for (var a = 0; a < this.PARAMETER_NUM; a++) {
        var c = this.params[a];
        if (c.val == "") {
            continue
        }
        if (c.noEnc) {
            b.push(c.name + "=" + c.val)
        } else {
            b.push(c.name + "=" + window.encodeURIComponent(c.val))
        }
    }
    return b.join("&")
};
YJ_YADS.InputParameter.prototype.clearUserInput = function() {
    if (typeof(this.userInput) == "undefined") {
        return 
    }
    for (var a = 0; a < this.PARAMETER_NUM; a++) {
        var b = this.PARAMETER_LIST[a];
        if (typeof(b.tagName) != "undefined") {
            this.userInput[b.tagName] = undefined
        }
    }
};
(function() {
    if (typeof(window.yads_ad_ds) == "undefined") {
        return 
    }
    if (location.protocol == "http:") {
        var f = (function() {
            var o = new Date();
            var n = o.getMonth() + 1;
            n = n < 10 ? "0" + n : n.toString();
            var p = o.getDate();
            p = p < 10 ? "0" + p : p.toString();
            return o.getFullYear() + n + p
        })();
        var k = "01";
        var l = "http://i.yimg.jp/images/listing/tool/yads/uadf/yads_vimps-1.0.0.js?" + f + k;
        document.write('<script type="text/javascript" language="JavaScript" src="' + l + '"><\/script>')
    }
    var a = YJ_YADS.gAdController.cache;
    var d = a.yads_sequence_num;
    if (typeof(d) == "undefined") {
        d = 0;
        a.yads_sequence_num = 0
    } else {
        d++;
        a.yads_sequence_num = d
    }
    var i = new YJ_YADS.InputParameter();
    i.init(window);
    var e = i.getParamByName("fr_id"), c = i.getParamByName("noad_cb"), b = i.getParamByName("tag_path");
    if (typeof(a.yads_onmessage_added) == "undefined") {
        a.yads_onmessage_added = true;
        if (window.addEventListener) {
            window.addEventListener("message", YJ_YADS.gAdController.onMessageCallback, false)
        } else {
            if (window.attachEvent) {
                window.attachEvent("onmessage", YJ_YADS.gAdController.onMessageCallback)
            }
        }
    }
    if (typeof(a.yads_noad_hook) == "undefined") {
        a.yads_noad_hook = {}
    }
    a.yads_noad_hook[e.val] = c.val;
    var g = b.val + "?" + i.getQueryString();
    var h = '<script src="' + g + '"><\/script>';
    document.write(h);
    i.clearUserInput()
})();
