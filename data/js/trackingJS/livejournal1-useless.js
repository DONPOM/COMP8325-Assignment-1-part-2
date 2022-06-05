var ready = function() {
    function a() {
        if (!e.isReady) {
            try {
                document.documentElement.doScroll("left")
            } catch (b) {
                setTimeout(a, 1);
                return 
            }
            e.ready()
        }
    }
    var c, b, d = {
        "[object Boolean]": "boolean",
        "[object Number]": "number",
        "[object String]": "string",
        "[object Function]": "function",
        "[object Array]": "array",
        "[object Date]": "date",
        "[object RegExp]": "regexp",
        "[object Object]": "object"
    }, e = {
        isReady: !1,
        readyWait: 1,
        holdReady: function(b) {
            b ? e.readyWait++ : e.ready(!0)
        },
        ready: function(b) {
            if (!0 === b&&!--e.readyWait ||!0 !== b&&!e.isReady) {
                if (!document.body)
                    return setTimeout(e.ready,
                    1);
                e.isReady=!0;
                !0 !== b && 0<--e.readyWait || c.resolveWith(document, [e])
            }
        },
        bindReady: function() {
            if (!c) {
                c = e._Deferred();
                if ("complete" === document.readyState)
                    return setTimeout(e.ready, 1);
                if (document.addEventListener)
                    document.addEventListener("DOMContentLoaded", b, !1), window.addEventListener("load", e.ready, !1);
                else if (document.attachEvent) {
                    document.attachEvent("onreadystatechange", b);
                    window.attachEvent("onload", e.ready);
                    var d=!1;
                    try {
                        d = null == window.frameElement
                    } catch (p) {}
                    document.documentElement.doScroll &&
                    d && a()
                }
            }
        },
        _Deferred: function() {
            var b = [], a, c, d, q = {
                done: function() {
                    if (!d) {
                        var c = arguments, g, r, h, l, k;
                        a && (k = a, a = 0);
                        g = 0;
                        for (r = c.length; g < r; g++)
                            h = c[g], l = e.type(h), "array" === l ? q.done.apply(q, h) : "function" === l && b.push(h);
                        k && q.resolveWith(k[0], k[1])
                    }
                    return this
                },
                resolveWith: function(e, q) {
                    if (!d&&!a&&!c) {
                        q = q || [];
                        c = 1;
                        try {
                            for (; b[0];)
                                b.shift().apply(e, q)
                            } finally {
                            a = [e, q], c = 0
                        }
                    }
                    return this
                },
                resolve: function() {
                    q.resolveWith(this, arguments);
                    return this
                },
                isResolved: function() {
                    return !(!c&&!a)
                },
                cancel: function() {
                    d = 1;
                    b =
                    [];
                    return this
                }
            };
            return q
        },
        type: function(b) {
            return null == b ? String(b) : d[Object.prototype.toString.call(b)] || "object"
        }
    };
    document.addEventListener ? b = function() {
        document.removeEventListener("DOMContentLoaded", b, !1);
        e.ready()
    } : document.attachEvent && (b = function() {
        "complete" === document.readyState && (document.detachEvent("onreadystatechange", b), e.ready())
    });
    return function(b) {
        e.bindReady();
        e.type(b);
        c.done(b)
    }
}();
function http_build_query(a, c, b) {
    var d, e, f = [], p = this, g = function(b, a, c) {
        var d, e = [];
        !0 === a ? a = "1" : !1 === a && (a = "0");
        if (null != a) {
            if ("object" === typeof a) {
                for (d in a)
                    null != a[d] && e.push(g(b + "[" + d + "]", a[d], c));
                return e.join(c)
            }
            if ("function" !== typeof a)
                return p.urlencode(b) + "=" + p.urlencode(a);
            throw Error("There was an error processing for http_build_query().");
        }
        return ""
    };
    b || (b = "&");
    for (e in a)
        d = a[e], c&&!isNaN(e) && (e = String(c) + e), d = g(e, d, b), "" != d && f.push(d);
    return f.join(b)
}
function urlencode(a) {
    a = (a + "").toString();
    return encodeURIComponent(a).replace(/!/g, "%21").replace(/'/g, "%27").replace(/\(/g, "%28").replace(/\)/g, "%29").replace(/\*/g, "%2A").replace(/%20/g, "+")
}
function md5(a) {
    var c = function(b, a) {
        var c, d, e, g, f;
        e = b & 2147483648;
        g = a & 2147483648;
        c = b & 1073741824;
        d = a & 1073741824;
        f = (b & 1073741823) + (a & 1073741823);
        return c & d ? f^2147483648^e^g : c | d ? f & 1073741824 ? f^3221225472^e^g : f^1073741824^e^g : f^e^g
    }, b = function(b, a, d, e, g, f, h) {
        b = c(b, c(c(a & d|~a & e, g), h));
        return c(b<<f | b>>>32 - f, a)
    }, d = function(b, a, d, e, f, g, h) {
        b = c(b, c(c(a & e | d&~e, f), h));
        return c(b<<g | b>>>32 - g, a)
    }, e = function(b, a, d, e, g, f, h) {
        b = c(b, c(c(a^d^e, g), h));
        return c(b<<f | b>>>32 - f, a)
    }, f = function(b, a, d, e, f, g, h) {
        b = c(b, c(c(d^(a |
        ~e), f), h));
        return c(b<<g | b>>>32 - g, a)
    }, p = function(b) {
        var a = "", c = "", d;
        for (d = 0; 3 >= d; d++)
            c = b>>>8 * d & 255, c = "0" + c.toString(16), a += c.substr(c.length - 2, 2);
        return a
    }, g = [], n, q, u, v, r, h, l, k, m;
    a = this.utf8_encode(a);
    g = function(b) {
        var a, c = b.length;
        a = c + 8;
        for (var d = 16 * ((a - a%64) / 64 + 1), e = Array(d - 1), g = 0, f = 0; f < c;)
            a = (f - f%4) / 4, g = f%4 * 8, e[a]|=b.charCodeAt(f)<<g, f++;
        a = (f - f%4) / 4;
        e[a]|=128<<f%4 * 8;
        e[d - 2] = c<<3;
        e[d - 1] = c>>>29;
        return e
    }(a);
    h = 1732584193;
    l = 4023233417;
    k = 2562383102;
    m = 271733878;
    a = g.length;
    for (n = 0; n < a; n += 16)
        q = h, u = l, v = k,
        r = m, h = b(h, l, k, m, g[n + 0], 7, 3614090360), m = b(m, h, l, k, g[n + 1], 12, 3905402710), k = b(k, m, h, l, g[n + 2], 17, 606105819), l = b(l, k, m, h, g[n + 3], 22, 3250441966), h = b(h, l, k, m, g[n + 4], 7, 4118548399), m = b(m, h, l, k, g[n + 5], 12, 1200080426), k = b(k, m, h, l, g[n + 6], 17, 2821735955), l = b(l, k, m, h, g[n + 7], 22, 4249261313), h = b(h, l, k, m, g[n + 8], 7, 1770035416), m = b(m, h, l, k, g[n + 9], 12, 2336552879), k = b(k, m, h, l, g[n + 10], 17, 4294925233), l = b(l, k, m, h, g[n + 11], 22, 2304563134), h = b(h, l, k, m, g[n + 12], 7, 1804603682), m = b(m, h, l, k, g[n + 13], 12, 4254626195), k = b(k, m, h, l, g[n + 14], 17,
        2792965006), l = b(l, k, m, h, g[n + 15], 22, 1236535329), h = d(h, l, k, m, g[n + 1], 5, 4129170786), m = d(m, h, l, k, g[n + 6], 9, 3225465664), k = d(k, m, h, l, g[n + 11], 14, 643717713), l = d(l, k, m, h, g[n + 0], 20, 3921069994), h = d(h, l, k, m, g[n + 5], 5, 3593408605), m = d(m, h, l, k, g[n + 10], 9, 38016083), k = d(k, m, h, l, g[n + 15], 14, 3634488961), l = d(l, k, m, h, g[n + 4], 20, 3889429448), h = d(h, l, k, m, g[n + 9], 5, 568446438), m = d(m, h, l, k, g[n + 14], 9, 3275163606), k = d(k, m, h, l, g[n + 3], 14, 4107603335), l = d(l, k, m, h, g[n + 8], 20, 1163531501), h = d(h, l, k, m, g[n + 13], 5, 2850285829), m = d(m, h, l, k, g[n + 2],
        9, 4243563512), k = d(k, m, h, l, g[n + 7], 14, 1735328473), l = d(l, k, m, h, g[n + 12], 20, 2368359562), h = e(h, l, k, m, g[n + 5], 4, 4294588738), m = e(m, h, l, k, g[n + 8], 11, 2272392833), k = e(k, m, h, l, g[n + 11], 16, 1839030562), l = e(l, k, m, h, g[n + 14], 23, 4259657740), h = e(h, l, k, m, g[n + 1], 4, 2763975236), m = e(m, h, l, k, g[n + 4], 11, 1272893353), k = e(k, m, h, l, g[n + 7], 16, 4139469664), l = e(l, k, m, h, g[n + 10], 23, 3200236656), h = e(h, l, k, m, g[n + 13], 4, 681279174), m = e(m, h, l, k, g[n + 0], 11, 3936430074), k = e(k, m, h, l, g[n + 3], 16, 3572445317), l = e(l, k, m, h, g[n + 6], 23, 76029189), h = e(h, l, k, m,
        g[n + 9], 4, 3654602809), m = e(m, h, l, k, g[n + 12], 11, 3873151461), k = e(k, m, h, l, g[n + 15], 16, 530742520), l = e(l, k, m, h, g[n + 2], 23, 3299628645), h = f(h, l, k, m, g[n + 0], 6, 4096336452), m = f(m, h, l, k, g[n + 7], 10, 1126891415), k = f(k, m, h, l, g[n + 14], 15, 2878612391), l = f(l, k, m, h, g[n + 5], 21, 4237533241), h = f(h, l, k, m, g[n + 12], 6, 1700485571), m = f(m, h, l, k, g[n + 3], 10, 2399980690), k = f(k, m, h, l, g[n + 10], 15, 4293915773), l = f(l, k, m, h, g[n + 1], 21, 2240044497), h = f(h, l, k, m, g[n + 8], 6, 1873313359), m = f(m, h, l, k, g[n + 15], 10, 4264355552), k = f(k, m, h, l, g[n + 6], 15, 2734768916), l =
        f(l, k, m, h, g[n + 13], 21, 1309151649), h = f(h, l, k, m, g[n + 4], 6, 4149444226), m = f(m, h, l, k, g[n + 11], 10, 3174756917), k = f(k, m, h, l, g[n + 2], 15, 718787259), l = f(l, k, m, h, g[n + 9], 21, 3951481745), h = c(h, q), l = c(l, u), k = c(k, v), m = c(m, r);
    return (p(h) + p(l) + p(k) + p(m)).toLowerCase()
}
function utf8_encode(a) {
    if (null === a || "undefined" === typeof a)
        return "";
    a += "";
    var c = "", b, d, e = 0;
    b = d = 0;
    for (var e = a.length, f = 0; f < e; f++) {
        var p = a.charCodeAt(f), g = null;
        128 > p ? d++ : g = 127 < p && 2048 > p ? String.fromCharCode(p>>6 | 192, p & 63 | 128) : String.fromCharCode(p>>12 | 224, p>>6 & 63 | 128, p & 63 | 128);
        null !== g && (d > b && (c += a.slice(b, d)), c += g, b = d = f + 1)
    }
    d > b && (c += a.slice(b, e));
    return c
}
function str_replace(a, c, b, d) {
    var e = 0, f = 0, p = "", g = "", n = 0, q = 0;
    a = [].concat(a);
    c = [].concat(c);
    var u = "[object Array]" === Object.prototype.toString.call(c), v = "[object Array]" === Object.prototype.toString.call(b);
    b = [].concat(b);
    d && (this.window[d] = 0);
    e = 0;
    for (n = b.length; e < n; e++)
        if ("" !== b[e])
            for (f = 0, q = a.length; f < q; f++)
                p = b[e] + "", g = u ? void 0 !== c[f] ? c[f] : "" : c[0], b[e] = p.split(a[f]).join(g), d && b[e] !== p && (this.window[d] += (p.length - b[e].length) / a[f].length);
    return v ? b : b[0]
}
function parse_url(a, c) {
    var b;
    b = "source scheme authority userInfo user pass host port relative path directory file query fragment".split(" ");
    for (var d = this.php_js && this.php_js.ini || {}, e = d["phpjs.parse_url.mode"] && d["phpjs.parse_url.mode"].local_value || "php", f = {
        php: /^(?:([^:\/?#]+):)?(?:\/\/()(?:(?:()(?:([^:@]*):?([^:@]*))?@)?([^:\/?#]*)(?::(\d*))?))?()(?:(()(?:(?:[^?#\/]*\/)*)()(?:[^?#]*))(?:\?([^#]*))?(?:#(.*))?)/,
        strict: /^(?:([^:\/?#]+):)?(?:\/\/((?:(([^:@]*):?([^:@]*))?@)?([^:\/?#]*)(?::(\d*))?))?((((?:[^?#\/]*\/)*)([^?#]*))(?:\?([^#]*))?(?:#(.*))?)/,
        loose: /^(?:(?![^:@]+:[^:@\/]*@)([^:\/?#.]+):)?(?:\/\/\/?)?((?:(([^:@]*):?([^:@]*))?@)?([^:\/?#]*)(?::(\d*))?)(((\/(?:[^?#](?![^?#\/]*\.[^?#\/.]+(?:[?#]|$)))*\/?)?([^?#\/]*))(?:\?([^#]*))?(?:#(.*))?)/
    }, f = f[e].exec(a), p = {}, g = 14; g--;)
        f[g] && (p[b[g]] = f[g]);
    if (c)
        return p[c.replace("PHP_URL_", "").toLowerCase()];
    if ("php" !== e) {
        var n = d["phpjs.parse_url.queryKey"] && d["phpjs.parse_url.queryKey"].local_value || "queryKey", f = /(?:^|&)([^&=]*)=?([^&]*)/g;
        p[n] = {};
        b = p[b[12]] || "";
        b.replace(f, function(b, a, c) {
            a && (p[n][a] =
            c)
        })
    }
    delete p.source;
    return p
}
function vsl_now() {
    return (new Date).getTime()
}
function vsl_getRuntime() {
    return vsl_now() - vsl_startTime
}
function parseQuery(a) {
    var c = a.replace(/^[^\?]+\??/, "");
    a = {};
    var b, d;
    if (0 < c.length)
        for (c = c.split("&"), d = 0; d < c.length; d++)
            b = c[d], b = b.split("="), a[b[0].toLowerCase()] = b[1];
    return a
}
function shuffle(a) {
    var c = a.length, b, d;
    if (0 === c)
        return !1;
    for (; --c;)
        b = Math.floor(Math.random() * (c + 1)), d = a[c], a[c] = a[b], a[b] = d;
    return a
}
function buildFdrvUrlTParam() {
    var a = (new Date).getUTCDate().toString(), c = "";
    "undefined" !== typeof anyclip.videoSenseLite.version && 0 < anyclip.videoSenseLite.version.length && (c += anyclip.videoSenseLite.version);
    return md5(a + c)
}
function buildPlaylistByRatio(a, c, b, d) {
    var e=!1;
    if ("undefined" === typeof d || 1 == parseInt(d))
        e=!0;
    if ("undefined" === typeof b ||!b)
        return e ? shuffle(a) : a;
    d = [];
    b = b.split("|");
    var f = parseInt(b[0]), p = parseInt(b[1]);
    if (100 == f && 0 == p)
        return e ? shuffle(a) : a;
    if (0 == f && 100 == p)
        return e ? shuffle(c) : c;
    var g = b = 1, n = 0, q = 0;
    e && (a = shuffle(a), c = shuffle(c));
    f > p ? (b = Math.round(f / p), g = 1) : f < p && (b = 1, g = Math.round(p / f));
    for (; n < a.length;) {
        for (e = 0; e < b; e++)
            "undefined" !== typeof a[n] && d.push(a[n]), n++;
        for (e = 0; e < g; e++)
            "undefined" !== typeof c[q] &&
            d.push(c[q]), q++, q >= c.length && (q = 0)
    }
    return d
}
function createCORSRequest(a, c) {
    var b = new XMLHttpRequest;
    "withCredentials"in b ? b.open(a, c, !0) : "undefined" != typeof XDomainRequest ? (b = new XDomainRequest, b.open(a, c)) : b = null;
    return b
}
function vsl_xdr(a, c, b, d, e) {
    var f;
    XMLHttpRequest ? (f = new XMLHttpRequest, "withCredentials"in f && (f.open(c, a, !0), f.onerror = e, f.onreadystatechange = function() {
        4 === f.readyState && (200 <= f.status && 400 > f.status ? d(f.responseText) : e(Error("Response returned with non-OK status")))
    }, f.send(b))) : XDomainRequest ? (f = new XDomainRequest, f.open(c, a), f.onerror = e, f.onload = function() {
        d(f.responseText)
    }, f.send(b)) : e(Error("CORS not supported"))
}
function getClipUrl(a, c) {
    var b = {}, d = {}, b = parse_url(a);
    d.code = c;
    d.scheme = b.scheme;
    d.filename = b.path.substring(b.path.lastIndexOf("/") + 1);
    d.server = b.host + str_replace("/" + d.filename, "", b.path);
    vsl_playlist.push(d)
}
function setrawcookie(a, c, b, d, e, f) {
    "string" === typeof b && /^\d+$/.test(b) && (b = parseInt(b, 10));
    b instanceof Date ? b = b.toGMTString() : "number" === typeof b && (b = (new Date(1E3 * b)).toGMTString());
    a = [a + "=" + c];
    c = {};
    var p = "";
    c = {
        expires: b,
        path: d,
        domain: e
    };
    for (p in c)
        c.hasOwnProperty(p) && c[p] && a.push(p + "=" + c[p]);
    return f && a.push("secure"), this.window.document.cookie = a.join(";"), !0
}
function setcookie(a, c, b, d, e, f) {
    return this.setrawcookie(a, encodeURIComponent(c), b, d, e, f)
}
function getCookie(a) {
    var c, b, d, e = document.cookie.split(";");
    for (c = 0; c < e.length; c++)
        if (b = e[c].substr(0, e[c].indexOf("=")), d = e[c].substr(e[c].indexOf("=") + 1), b = b.replace(/^\s+|\s+$/g, ""), b == a)
            return unescape(d)
}
function myTrim(a) {
    return a.replace(/^\s+|\s+$/gm, "")
}
function vsClipEnded() {
    return this
}(function() {
    var a = 0, c = function() {
        _.extend(this, Backbone.Events);
        _.bindAll(this, "embed", "eventsCallback", "_constructor", "setAttributes", "setDefaults", "addEventsToFlashvars");
        this._constructor.apply(this, arguments);
        return this
    };
    c.prototype = {
        _swfPath: null,
        _swfComponent: null,
        _id: null,
        _placementId: null,
        _swf: null,
        _flash_params: null,
        _flash_flashvars: null,
        _flash_attributes: null,
        _flash_swfVersionStr: null,
        _flash_xiSwfUrlStr: null,
        _flash_callbackFn: null,
        _flash_width: null,
        _flash_height: null,
        _qaEnable: !1
    };
    c.prototype.eventsCallback =
    function(b) {
        this.trigger("vsl_report", b);
        return this
    };
    c.prototype._constructor = function() {
        this._id = "_vctv_flashobj_" + a++;
        this.setDefaults();
        arguments.length && this.setAttributes.apply(this, arguments);
        window[this._id + "_events"] = this.eventsCallback;
        return this
    };
    c.prototype.setDefaults = function() {
        var b = this;
        this._swfPath = "assets/swf/";
        this._flash_params = {
            quality: "high",
            allowScriptAccess: "always",
            allowFullScreen: !0,
            wmode: "transparent"
        };
        this._flash_flashvars = {
            eventsFunc: this._id + "_events"
        };
        this._flash_attributes =
        {
            id: this._id,
            name: this._id
        };
        this._flash_swfVersionStr = "10.0.0";
        this._flash_callbackFn = function(a) {
            player = b._swf = a.ref;
            b.trigger("vsl_ready")
        };
        this._flash_height = this._flash_width = "100%";
        return this
    };
    c.prototype.addEventsToFlashvars = function() {
        this._flash_flashvars.eventsFunc = this._id + "_events"
    };
    c.prototype.loadClip = function(b, a, c, f, p, g, n, q, u, v, r, h) {
        null != this._swf && this._swf.loadClip(b, a, c, f, p, g, n, q, u, v, r, h)
    };
    c.prototype.setAttributes = function() {
        var b = arguments, a = this;
        if (b.length)
            return "string" ===
            typeof b[0] && (a["_" + b[0]] = b[1] || null), "object" === typeof b[0] && _.each(b[0], function(b, c) {
                a["_" + c] = b
            }), this
    };
    c.prototype.embed = function() {
        var a = "?", c = this._placementId || null;
        if (null != c) {
            var e = {
                t: buildFdrvUrlTParam()
            };
            this._qaEnable && (e.uiqa=!0);
            a += http_build_query(e);
            swfobject.embedSWF(this._swfPath + a, c, this._flash_width, this._flash_height, this._flash_swfVersionStr, this._flash_xiSwfUrlStr, this._flash_flashvars, this._flash_params, this._flash_attributes, this._flash_callbackFn);
            return this
        }
    };
    c.prototype.dispose =
    function() {
        if (null != this._swf) {
            try {
                this._swf.dispose()
            } catch (a) {
                console.log("error disposing flash, ignoring", a)
            }
            this._swf = null
        }
    };
    window.FDRV = c
})();
window.console || (console = {
    log: function() {},
    warn: function() {}
});
var LiveRailVPAID, vpaidFrame, vpaidLoader, playerWrap, player, currentAdUnit, ad_wrap, adType = "", playerVol = "", anyclip = anyclip || {}, vsl_fdrv = new FDRV, vsl_ie7=!1, vsl_custom=!1, vsl_customClip = [], vsl_customCode = [];
if (/MSIE\s([\d.]+)/.test(navigator.userAgent)) {
    var version = new Number(RegExp.$1);
    8 > version && (vsl_ie7=!0)
}
anyclip.videoSenseLite = {};
anyclip.videoSenseLite.cdnServerUrl = "anyclip.fcod.llnwd.net/a8895/e1";
anyclip.videoSenseLite.stage = {};
_.extend(anyclip.videoSenseLite.stage, Backbone.Events);
anyclip.videoSenseLite.player = {
    swfVersionStr: "10.0.0",
    flashvars: {
        JSMethodsPrefix: "vsl_",
        pub_keyword: "anyclip_com",
        partner: "730867",
        ref: window.location.href
    },
    flashparams: {
        wmode: "transparent",
        quality: "high",
        allowScriptAccess: "always",
        allowFullScreen: !0
    }
};
anyclip.videoSenseLite.clips = "iRYUmhtbm Oc9V4nthtt4 FCrw4Yht7b zzmt7Juht7u xOfQmhnJb ivuDtJhnub 4issbhtJJ F_CoYunhtu4 FjczuYhtnJ _FJm22n7ht7t rtLL4724mhYbY lidIntttmhbb44 AOE_b24Y2hbbnm ukn4un2u7hYu2 C4cZmhYJn Fu3imhtYb ZVjumhbmnb FIxvmhtn4 yDxomhY22 uVzUmhbbJu Immmmht27 _m1Cmhu7t YjIy2huYJ xOfQmhbmt4 1yt2mh7mt ivuD7buhbJtb ivuDbY72htJm CCjDmhuun oEombmhb4bu FvEjbn7nhnbm 3OEQmhtmt AY1Vmh7t7 AY1V7tnhu2Y".split(" ");
anyclip.videoSenseLite.trailers = "iRYUmhtbm Oc9V4nthtt4 FCrw4Yht7b zzmt7Juht7u xOfQmhnJb ivuDtJhnub 4issbhtJJ F_CoYunhtu4 FjczuYhtnJ _FJm22n7ht7t rtLL4724mhYbY lidIntttmhbb44 AOE_b24Y2hbbnm ukn4un2u7hYu2 C4cZmhYJn Fu3imhtYb ZVjumhbmnb FIxvmhtn4 yDxomhY22 uVzUmhbbJu Immmmht27 _m1Cmhu7t YjIy2huYJ xOfQmhbmt4 1yt2mh7mt ivuD7buhbJtb ivuDbY72htJm CCjDmhuun oEombmhb4bu FvEjbn7nhnbm 3OEQmhtmt AY1Vmh7t7 AY1V7tnhu2Y".split(" ");
anyclip.videoSenseLite.clipData = {
    iRYUmhtbm: {
        u: "maleficent-light-and-dark",
        ut: "maleficent-light-and-dark",
        s: "89",
        m: "UQ9N"
    },
    Oc9V4nthtt4: {
        u: "x-men-days-of-future-past-michael-fassbender",
        ut: "xmen-days-of-future-past-michael-fassbender",
        s: "22",
        m: "WMIL"
    },
    FCrw4Yht7b: {
        u: "godzilla-bryan-cranston",
        ut: "godzilla-bryan-cranston",
        s: "22",
        m: "GPYJ"
    },
    zzmt7Juht7u: {
        u: "x-men-days-of-future-past-hugh-jackman",
        ut: "xmen-days-of-future-past-hugh-jackman",
        s: "52",
        m: "AA06"
    },
    xOfQmhnJb: {
        u: "maleficent-european-red-carpet",
        ut: "maleficent-european-red-carpet",
        s: "89",
        m: "SWOZ"
    },
    ivuDtJhnub: {
        u: "despicable-me-2-interview",
        ut: "despicable-me-2---behind-the-scenes",
        s: "60",
        m: "Ub5X"
    },
    "4issbhtJJ": {
        u: "how-to-train-your-dragon-2-gerard-butler-1",
        ut: "how-to-train-your-dragon-2-gerard-butler",
        s: "27",
        m: "4UBB"
    },
    F_CoYunhtu4: {
        u: "godzilla-elizabeth-olsen-1",
        ut: "godzilla-elizabeth-olsen",
        s: "22",
        m: "GCPF"
    },
    FjczuYhtnJ: {
        u: "neighbors-seth-rogen",
        ut: "neighbors-seth-rogen",
        s: "4",
        m: "GEMA"
    },
    _FJm22n7ht7t: {
        u: "blended-adam-and-drew",
        ut: "blended-adam-and-drew",
        s: "22",
        m: "CG20"
    },
    rtLL4724mhYbY: {
        u: "police-car-race",
        ut: "fast-five",
        s: "4",
        m: "Y6kk"
    },
    lidIntttmhbb44: {
        u: "the-jokers-monologue",
        ut: "the-dark-knight",
        s: "22",
        m: "xUsc"
    },
    AOE_b24Y2hbbnm: {
        u: "300",
        ut: "300",
        s: "22",
        m: "DWVC"
    },
    ukn4un2u7hYu2: {
        u: "fireworks",
        ut: "harry-potter-and-the-order-of-the-phoenix",
        s: "22",
        m: "5r74"
    },
    C4cZmhYJn: {
        u: "the-model-for-hardscrabble",
        ut: "monsters-university---hardscrabble",
        s: "60",
        m: "P4Md"
    },
    Fu3imhtYb: {
        u: "a-gag-session",
        ut: "monsters-university---gag-session",
        s: "60",
        m: "G5HU"
    },
    ZVjumhbmnb: {
        u: "monsters-drama-class",
        ut: "monsters-university---drama-class",
        s: "60",
        m: "dLE5"
    },
    FIxvmhtn4: {
        u: "scaring-the-new-kids",
        ut: "monsters-university---movie-night",
        s: "60",
        m: "GcSb"
    },
    yDxomhY22: {
        u: "learning-about-children",
        ut: "monsters-university---recon",
        s: "60",
        m: "RXSF"
    },
    uVzUmhbbJu: {
        u: "picked-for-the-scare-team",
        ut: "monsters-university---rivalry",
        s: "60",
        m: "5LAN"
    },
    Immmmht27: {
        u: "designing-the-university",
        ut: "monsters-university---designing-the-campus",
        s: "60",
        m: "c000"
    },
    _m1Cmhu7t: {
        u: "walking-with-dinosaurs-3d-behavior",
        ut: "walking-with-dinosaurs-3d---behavior",
        s: "52",
        m: "C0aP"
    },
    YjIy2huYJ: {
        u: "walking-with-dinosaurs-origins",
        ut: "walking-with-dinosaurs-3d---origins",
        s: "52",
        m: "9EcR"
    },
    xOfQmhbmt4: {
        u: "european-red-carpet",
        ut: "maleficent-european-red-carpet",
        s: "89",
        m: "SWOZ"
    },
    "1yt2mh7mt": {
        u: "x-men-sound-bite-peter-dinklage",
        ut: "xmen-days-of-future-past-peter-dinklage",
        s: "52",
        m: "aR63"
    },
    ivuD7buhbJtb: {
        u: "despicable-me-2",
        ut: "despicable-me-2---behind-the-scenes",
        s: "60",
        m: "Ub5X"
    },
    ivuDbY72htJm: {
        u: "despicable-me-2-interview1",
        ut: "despicable-me-2---behind-the-scenes",
        s: "60",
        m: "Ub5X"
    },
    CCjDmhuun: {
        u: "the-lone-ranger-johnny-depp-interview-trailer",
        ut: "the-lone-ranger---johnny-depp-interview",
        s: "60",
        m: "PPEX"
    },
    oEombmhb4bu: {
        u: "the-lone-ranger-gore-verbinski-trailer",
        ut: "the-lone-ranger---gore-verbinski",
        s: "60",
        m: "FVF0"
    },
    FvEjbn7nhnbm: {
        u: "fast-and-furious",
        ut: "fast-furious-6---b-roll-5",
        s: "60",
        m: "GbVE"
    },
    "3OEQmhtmt": {
        u: "jon-hamm-interview",
        ut: "million-dollar-arm-jon-hamm",
        s: "89",
        m: "HWVZ"
    },
    AY1Vmh7t7: {
        u: "x-men-sound-bite-patrick-stewart",
        ut: "xmen-days-of-future-past-patrick-stewart",
        s: "52",
        m: "D9aL"
    },
    AY1V7tnhu2Y: {
        u: "x-men-sound-bite-more-patrick-stewart",
        ut: "xmen-days-of-future-past-patrick-stewart",
        s: "52",
        m: "D9aL"
    }
};
"undefined" !== typeof loadCustomClips && (anyclip.videoSenseLite.clips = customClips.slice(0));
anyclip.videoSenseLite.trailers = anyclip.videoSenseLite.clips;
var vsl_request_id = md5(String(Math.floor(1E6 * Math.random() + 1)) + String((new Date).getTime())), vsl_startTime = vsl_now(), vsl_clipCounter = 0, vsl_refDomain = window.location.href;
0 < vsl_refDomain.indexOf("/") && (vsl_refDomain = vsl_refDomain.split("/").slice(2).join("/"));
0 < vsl_refDomain.indexOf("/") && (vsl_refDomain = vsl_refDomain.substr(0, vsl_refDomain.indexOf("/")));
for (var vsl_scripts = document.getElementsByTagName("script"), vsl_myScript = "", vsl_i = 0; vsl_i < vsl_scripts.length && (vsl_myScript = vsl_scripts[vsl_i], - 1 === vsl_myScript.src.indexOf("vslu.js")); vsl_i++);
var vsl_params = parseQuery(vsl_myScript.src);
if ("undefined" !== typeof vsl_params.nc && 1 == vsl_params.nc)
    var vsl_session_id = "nocookiesessionid", vsl_user_id = "nocookieuserid";
else {
    vsl_session_id = getCookie("vsl_sessionid");
    if (null == vsl_session_id || "" == vsl_session_id)
        vsl_session_id = md5(String(Math.floor(1E6 * Math.random() + 1)) + String((new Date).getTime())), setcookie("vsl_sessionid", vsl_session_id, null, "/");
    vsl_user_id = getCookie("vsl_userid");
    if (null == vsl_user_id || "" == vsl_user_id)
        vsl_user_id = md5(String(Math.floor(1E6 * Math.random() + 1)) + String((new Date).getTime())),
        setcookie("vsl_userid", vsl_user_id, Math.floor((new Date).getTime() / 1E3) + 31536E3, "/")
}
var vsl_ifl = 0;
if (window.top != window.self)
    for (var vsl_window = window.self; window.top != vsl_window;)
        vsl_window = vsl_window.parent, vsl_ifl++;
var logInfo = {
    category: "flat_" + vsl_params.type + "_2.0.0",
    ref: window.location.href.substr(0, 100),
    refDomain: vsl_refDomain,
    user_id: vsl_user_id,
    session_id: vsl_session_id,
    request_id: vsl_request_id,
    client_id: vsl_params.cid,
    site_id: vsl_params.sid,
    player_type: vsl_params.type,
    placement_id: vsl_params.uid,
    ifl: vsl_ifl,
    partner_id: 730867,
    pub_keyword: "unknown",
    mode: "flat",
    hasFlash: swfobject.hasFlashPlayerVersion("10.0.0") ? 1: 0,
    autoplay: "y",
    ib: "1"
}, x;
for (x in vsl_params)
    if ( - 1 !== x.indexOf("utm")) {
        "undefined" === typeof vsl_params.utm && (vsl_params.utm = {});
        var t = x.charAt(3);
        "_" == t && (t = x.charAt(4));
        switch (t) {
        case "s":
            vsl_params.utm.utm_source = vsl_params[x];
            break;
        case "m":
            vsl_params.utm.utm_medium = vsl_params[x];
            break;
        case "c":
            vsl_params.utm.utm_campaign = vsl_params[x];
            break;
        case "t":
            vsl_params.utm.utm_term = vsl_params[x];
            break;
        case "n":
            vsl_params.utm.utm_content = vsl_params[x]
        }
    }
"undefined" === typeof vsl_params.type && (vsl_params.type = "lite");
"undefined" === typeof vsl_params.vr && (vsl_params.vr = "prod");
var vsl_player_path = "http://player.anyclip.com/" + vsl_params.vr + "/";
"uilrlite" === vsl_params.type && (anyclip.videoSenseLite.player.url = vsl_player_path + "litePlayer/LitePlayer_LiveRail_UI.swf", vsl_ie7 && (anyclip.videoSenseLite.player.url = vsl_player_path + "litePlayer/LitePlayer_LiveRail_FDRV.swf"), anyclip.videoSenseLite.player.flashvars.streamingProtocol = "rtmp", anyclip.videoSenseLite.player.flashvars.streamingServer = anyclip.videoSenseLite.cdnServerUrl);
"engage" === vsl_params.type && (anyclip.videoSenseLite.player.url = vsl_player_path + "litePlayer/LitePlayer_LiveRail_Engagement.swf", anyclip.videoSenseLite.player.flashvars.streamingProtocol = "rtmp", anyclip.videoSenseLite.player.flashvars.streamingServer = anyclip.videoSenseLite.cdnServerUrl);
"flrlite" === vsl_params.type && (anyclip.videoSenseLite.player.url = vsl_player_path + "litePlayer/LitePlayer_LiveRail_FDRV.swf", anyclip.videoSenseLite.player.flashvars.streamingProtocol = "rtmp", anyclip.videoSenseLite.player.flashvars.streamingServer = anyclip.videoSenseLite.cdnServerUrl);
"lrlite" === vsl_params.type && (anyclip.videoSenseLite.player.url = vsl_player_path + "litePlayer/LitePlayer_LiveRail_FDRV.swf", anyclip.videoSenseLite.player.flashvars.streamingProtocol = "rtmp", anyclip.videoSenseLite.player.flashvars.streamingServer = anyclip.videoSenseLite.cdnServerUrl);
"vlrlite" === vsl_params.type && (anyclip.videoSenseLite.player.url = vsl_player_path + "litePlayer/LitePlayer_LiveRail_FDRV_VPAID.swf", anyclip.videoSenseLite.player.flashvars.streamingProtocol = "rtmp", anyclip.videoSenseLite.player.flashvars.streamingServer = anyclip.videoSenseLite.cdnServerUrl);
"tlrlite" === vsl_params.type && (anyclip.videoSenseLite.player.url = vsl_player_path + "litePlayer/LitePlayer_LiveRail_FDRV_Test.swf", anyclip.videoSenseLite.player.flashvars.streamingProtocol = "rtmp", anyclip.videoSenseLite.player.flashvars.streamingServer = anyclip.videoSenseLite.cdnServerUrl);
"mbclrlite" === vsl_params.type && (anyclip.videoSenseLite.player.url = vsl_player_path + "litePlayer/LitePlayer_LiveRail_MBC.swf", anyclip.videoSenseLite.player.flashvars.streamingProtocol = "rtmp", anyclip.videoSenseLite.player.flashvars.streamingServer = anyclip.videoSenseLite.cdnServerUrl);
"lite" === vsl_params.type && (anyclip.videoSenseLite.player.url = vsl_player_path + "litePlayer/LitePlayer_AdapTV.swf", anyclip.videoSenseLite.player.flashvars.streamingProtocol = "rtmp", anyclip.videoSenseLite.player.flashvars.streamingServer = anyclip.videoSenseLite.cdnServerUrl);
"unified" === vsl_params.type && (anyclip.videoSenseLite.player.url = "http://player.anyclip.com/acp4/vs/unified/AnyClipPlayer4.swf");
"plugins" === vsl_params.type && (anyclip.videoSenseLite.player.url = "http://player.anyclip.com/acp4/vs/nonunified/AnyClipPlayer4.swf");
var vsl_playlist = [], vsl_playlist = "undefined" !== typeof vsl_params.tr && 1 == parseInt(vsl_params.tr) ? shuffle(anyclip.videoSenseLite.trailers): buildPlaylistByRatio(anyclip.videoSenseLite.clips, anyclip.videoSenseLite.trailers, anyclip.videoSenseLite.clipsTrailersRatio, vsl_params.sh);
if ("undefined" !== typeof vsl_params.clip) {
    vsl_playlist = [];
    vsl_custom=!0;
    if ("undefined" !== typeof vsl_params.code)
        if ( - 1 === vsl_params.code.indexOf("|"))
            vsl_customCode.push(vsl_params.code);
        else 
            for (var codeArr = vsl_params.code.split("|"), codeArrCounter = 0; codeArrCounter < codeArr.length; codeArrCounter++)
                vsl_customCode.push(codeArr[codeArrCounter]);
    if ( - 1 === vsl_params.clip.indexOf("|"))
        getClipUrl(vsl_params.clip, vsl_customCode[0]);
    else 
        for (var clipArr = vsl_params.clip.split("|"), clipArrCounter = 0; clipArrCounter <
        clipArr.length; clipArrCounter++)
            getClipUrl(clipArr[clipArrCounter], vsl_customCode[clipArrCounter])
}
anyclip.videoSenseLite.items = anyclip.videoSenseLite.items || {};
anyclip.videoSenseLite.items["item_" + vsl_params.uid] = vsl_params;
vsl_width = 300;
vsl_height = 250;
"undefined" !== typeof vsl_params.w && (vsl_width = parseInt(vsl_params.w));
"undefined" !== typeof vsl_params.h && (vsl_height = parseInt(vsl_params.h));
vsl_params.w = vsl_width;
vsl_params.h = vsl_height;
"undefined" === typeof vsl_params.ig && (vsl_params.ig = 0);
"undefined" !== typeof vsl_params.utm && vsl_params.utm.utm_source && (logInfo.utm = vsl_params.utm.utm_source);
parent != window && (logInfo.refParent = document.referrer.substr(0, 100));
"undefined" !== typeof vsl_params.q && null != vsl_params.q && (logInfo.query = vsl_params.q);
"undefined" !== typeof vsl_params.pub ? (logInfo.pub_keyword = vsl_params.pub, anyclip.videoSenseLite.player.flashvars.pub_keyword = vsl_params.pub) : anyclip.videoSenseLite.player.flashvars.pub_keyword = "unknown";
if ("undefined" !== typeof vsl_params.pr)
    if ( - 1 === vsl_params.pr.indexOf("|"))
        logInfo.partner_id = vsl_params.pr, anyclip.videoSenseLite.player.flashvars.partner = vsl_params.pr;
    else {
        for (var prArr = vsl_params.pr.split("|"), prString = "", prComma = "", prCounter = 0; prCounter < prArr.length; prCounter++)
            prString += prComma + prArr[prCounter], prComma = ",";
            logInfo.partner_id = prString;
            anyclip.videoSenseLite.player.flashvars.partner = prString
    } else 
        anyclip.videoSenseLite.player.flashvars.partner = 730867;
"undefined" !== typeof vsl_params.pid && (logInfo.publisher_id = vsl_params.pid, anyclip.videoSenseLite.player.flashvars.pubId = vsl_params.pid);
"undefined" !== typeof vsl_params.utmc && "undefined" !== typeof vsl_params.utmt && (anyclip.videoSenseLite.player.flashvars.context = encodeURIComponent('{"utm_campaign": "' + vsl_params.utmc + '" ,"utm_term": "' + vsl_params.utmt + '"}'));
anyclip.videoSenseLite.player.flashvars.utmSource = "undefined" !== typeof vsl_params.utms ? vsl_params.utms : "unknown";
var urlParams = parent != window ? parseQuery(document.referrer): parseQuery(window.location.href);
"undefined" !== typeof urlParams.utm_source && (anyclip.videoSenseLite.player.flashvars.utmSource = urlParams.utm_source);
"undefined" !== typeof vsl_params.ref && "" !== myTrim(vsl_params.ref) ? (logInfo.refCustom = vsl_params.ref.substr(0, 100), anyclip.videoSenseLite.player.flashvars.ref = vsl_params.ref) : (anyclip.videoSenseLite.player.flashvars.ref = parent != window ? document.referrer : window.location.href, anyclip.videoSenseLite.player.flashvars.ref = encodeURIComponent(anyclip.videoSenseLite.player.flashvars.ref));
vsl_params.iasRef = window.location.href;
parent != window && (vsl_params.iasRef = document.referrer);
"undefined" !== typeof vsl_params.sl && (anyclip.videoSenseLite.player.flashvars.showLogo = vsl_params.sl);
anyclip.videoSenseLite.player.flashvars.muteonload=!0;
"undefined" !== typeof vsl_params.sn && 1 == parseInt(vsl_params.sn) && (anyclip.videoSenseLite.player.flashvars.muteonload=!1);
"undefined" !== typeof vsl_params.postrolltimerdelay && (anyclip.videoSenseLite.player.flashvars.postrollTimerDelay = vsl_params.postrolltimerdelay);
"undefined" !== typeof vsl_params.prerolltimerdelay && (anyclip.videoSenseLite.player.flashvars.prerollTimerDelay = vsl_params.prerolltimerdelay);
anyclip.videoSenseLite.player.flashvars.playerId = "acp_" + vsl_params.uid;
anyclip.videoSenseLite.player.flashvars.placementId = vsl_params.uid;
"undefined" !== typeof vsl_params.lrtd && (anyclip.videoSenseLite.player.flashvars.lrTimeoutDelivery = vsl_params.lrtd);
"undefined" !== typeof vsl_params.lrta && (anyclip.videoSenseLite.player.flashvars.lrTimeoutAdsource = vsl_params.lrta);
"undefined" !== typeof vsl_params.lrwl && (anyclip.videoSenseLite.player.flashvars.lrWrapperLimit = vsl_params.lrwl);
"undefined" !== typeof vsl_params.lrts && (anyclip.videoSenseLite.player.flashvars.lrTimeoutStream = vsl_params.lrts);
"undefined" !== typeof vsl_params.lrtvi && (anyclip.videoSenseLite.player.flashvars.lrTimeoutVpaidInit = vsl_params.lrtvi);
"undefined" !== typeof vsl_params.lrtvs && (anyclip.videoSenseLite.player.flashvars.lrTimeoutVpaidStart = vsl_params.lrtvs);
function myLogAction(a, c) {
    var b = document.createElement("img");
    b.width = 1;
    b.height = 1;
    b.style.display = "none";
    b.src = "qa" == anyclip.videoSenseLite.items["item_" + a].vr ? "http://statsqa.anyclip.com/fictivious.png?placement_id=" + a + "&action=globalError&label=" + c + "&value=" + vsl_getRuntime() : "http://stats.anyclip.com/fictivious.png?placement_id=" + a + "&action=globalError&label=" + c + "&value=" + vsl_getRuntime();
    b.onerror = function() {};
    b.onload = function() {}
}
window.onerror = function(a, c) {
    if (500 === Math.floor(1E3 * Math.random() + 1)) {
        console.log("windowError");
        if ("undefined" == typeof c || "" == c)
            c = window.location.href;
        if ("undefined" !== typeof widget)
            try {
                vsl_reportEvent("globalError", null, a + " url:" + c)
            } catch (b) {
            myLogAction(vsl_params.uid, b.message)
        } else 
            console.log("myLogAction"), myLogAction(vsl_params.uid, a + " url:" + c)
    }
};
function ThrowError() {
    window.NotExistingFunction()
}(function() {})();
function vsl_buildUrl(a) {
    var c = "";
    return c = null !== anyclip.videoSenseLite.clipData[a].u ? "http://www.anyclip.com/movies/" + anyclip.videoSenseLite.clipData[a].ut + "/" + anyclip.videoSenseLite.clipData[a].u + "/" : "http://www.anyclip.com/movies/" + anyclip.videoSenseLite.clipData[a].ut + "/"
}
function vsl_getClipName(a) {
    var c = "";
    null !== anyclip.videoSenseLite.clipData[a].u && (c = str_replace("-", " ", anyclip.videoSenseLite.clipData[a].u));
    return c
}
function vsl_getTitleName(a) {
    var c = "";
    null !== anyclip.videoSenseLite.clipData[a].ut && (c = str_replace("-", " ", anyclip.videoSenseLite.clipData[a].ut));
    return c
}
function getTitleCode(a) {
    return "undefined" !== typeof anyclip.videoSenseLite.clipData[a] && "undefined" !== typeof anyclip.videoSenseLite.clipData[a].m ? anyclip.videoSenseLite.clipData[a].m : "unknown"
}
function getStudioId(a) {
    return "undefined" !== typeof anyclip.videoSenseLite.clipData[a] && "undefined" !== typeof anyclip.videoSenseLite.clipData[a].s ? anyclip.videoSenseLite.clipData[a].s : "unknown"
}
vsl_fdrv.on("vsl_ready", function() {
    vsl_reportEvent(vsl_params.uid, "playerTemplateRenderd", vsl_getRuntime(), "");
    "vlrlite" === vsl_params.type && (vpaidFrame = document.createElement("iframe"), vpaidFrame.style.display = "none", vpaidFrame.onload = function() {
        vpaidLoader = vpaidFrame.contentWindow.document.createElement("script");
        vpaidLoader.src = "http://cdn-static.liverail.com/js/LiveRail.AdManager-1.0.js";
        vsl_reportEvent(vsl_params.uid, "lrLoading", vsl_getRuntime(), vsl_playlist[vsl_clipCounter]);
        vpaidLoader.onload =
        function() {
            vsl_reportEvent(vsl_params.uid, "lrLoaded", vsl_getRuntime(), vsl_playlist[vsl_clipCounter]);
            LiveRailVPAID = vpaidFrame.contentWindow.getVPAIDAd();
            LiveRailVPAID.handshakeVersion("2.0");
            onVPAIDLoad();
            initAd("in", 120, 0, 300, 250);
            adType = "Preroll";
            vsl_reportEvent(vsl_params.uid, "adPrerollInit-" + (vsl_clipCounter + 1).toString(), vsl_getRuntime(), vsl_playlist[vsl_clipCounter])
        };
        vpaidFrame.contentWindow.document.body.appendChild(vpaidLoader)
    }, document.body.appendChild(vpaidFrame))
});
vsl_fdrv.on("vsl_report", function(a) {
    console.log(a);
    var c = this;
    if ("initAd" === a.eventName) {
        var b = a.parameters[0];
        initAd(b.adUnit, b.clipDuration, b.videoPosition, b.width, b.height);
        a.eventName = "ad" + adType + "Init"
    }
    "adErrorOrTimeout" === a.eventName || "clipFailed" === a.eventName ? vsl_reportEvent(a.placementId, a.eventName + "-" + (vsl_clipCounter + 1).toString(), vsl_getRuntime(), vsl_playlist[vsl_clipCounter], a.label) : vsl_reportEvent(a.placementId, a.eventName + "-" + (vsl_clipCounter + 1).toString(), vsl_getRuntime(), vsl_playlist[vsl_clipCounter]);
    switch (a.eventName) {
    case "adPrerollFailed":
        "undefined" !== typeof anyclip.videoSenseLite.items["item_" + a.placementId].pb && (b = anyclip.videoSenseLite.items["item_" + a.placementId].pb, a.eventName = "loadPassbackTag", vsl_reportEvent(a.placementId, a.eventName + "-" + (vsl_clipCounter + 1).toString(), vsl_getRuntime(), vsl_playlist[vsl_clipCounter]), document.getElementById("player_wrap_" + a.placementId).innerHTML = '<iframe frameborder=0 width=300 height=250 marginheight=0 marginwidth=0 scrolling=no src="http://player.anyclip.com/video-sense/html/' +
        b + '.html"></iframe>');
        break;
    case "readyToNextClip":
        if (vsl_clipCounter++, vsl_clipCounter === vsl_playlist.length && (vsl_clipCounter = 0), a = anyclip.videoSenseLite.items["item_" + a.placementId].type, vsClipEnded(vsl_playlist, vsl_clipCounter), "uilrlite" == a) {
            var d = createCORSRequest("get", "http://apis.anyclip.com/api/clip/public/" + vsl_playlist[vsl_clipCounter] + "?format=JSON");
            d ? (d.onload = function() {
                var a = JSON.parse(d.responseText);
                a.url = typeof("undefined" !== a.uniqueName) ? "http://www.anyclip.com/movies/" + a.title.uniqueName +
                "/" + a.uniqueName + "/" : "http://www.anyclip.com/movies/" + a.title.uniqueName + "/";
                c.loadClip("rtmp", anyclip.videoSenseLite.cdnServerUrl, vsl_playlist[vsl_clipCounter] + ".mp4", vsl_playlist[vsl_clipCounter], a.name, vsl_clipCounter + 1, a.url, a.title.poster, encodeURIComponent(JSON.stringify(a.title.buyLinks)), a.title.name, getTitleCode(vsl_playlist[vsl_clipCounter]), getStudioId(vsl_playlist[vsl_clipCounter]))
            }, d.send()) : (a = vsl_getClipName(vsl_playlist[vsl_clipCounter]), b = vsl_getTitleName(vsl_playlist[vsl_clipCounter]),
            a = str_replace("-", " ", a), b = str_replace("-", " ", b), c.loadClip("rtmp", anyclip.videoSenseLite.cdnServerUrl, vsl_playlist[vsl_clipCounter] + ".mp4", vsl_playlist[vsl_clipCounter], a, vsl_clipCounter + 1, vsl_buildUrl(vsl_playlist[vsl_clipCounter]), null, null, b, getTitleCode(vsl_playlist[vsl_clipCounter]), getStudioId(vsl_playlist[vsl_clipCounter])))
        } else 
            a = vsl_getClipName(vsl_playlist[vsl_clipCounter]), b = vsl_getTitleName(vsl_playlist[vsl_clipCounter]), a = str_replace("-", " ", a), b = str_replace("-", " ", b), this.loadClip("rtmp",
            anyclip.videoSenseLite.cdnServerUrl, vsl_playlist[vsl_clipCounter] + ".mp4", vsl_playlist[vsl_clipCounter], a, vsl_clipCounter + 1, vsl_buildUrl(vsl_playlist[vsl_clipCounter]), null, null, b, getTitleCode(vsl_playlist[vsl_clipCounter]), getStudioId(vsl_playlist[vsl_clipCounter]))
    }
});
!0 === vsl_custom && (vsl_fdrv.off("vsl_report"), vsl_fdrv.on("vsl_report", function(a) {
    vsl_reportEvent(a.placementId, a.eventName + "-" + (vsl_clipCounter + 1).toString(), vsl_getRuntime(), vsl_playlist[vsl_clipCounter].code);
    switch (a.eventName) {
    case "readyToNextClip":
        vsl_clipCounter++, vsl_clipCounter === vsl_playlist.length && (vsl_clipCounter = 0), "wpc.6acd.edgecastcdn.net/006ACD/tok" === vsl_playlist[vsl_clipCounter].server && (vsl_playlist[vsl_clipCounter].server = "anyclip.fcod.llnwd.net/a8895/e4", vsl_playlist[vsl_clipCounter].scheme =
        "rtmp"), this.loadClip(vsl_playlist[vsl_clipCounter].scheme, vsl_playlist[vsl_clipCounter].server, vsl_playlist[vsl_clipCounter].filename, vsl_playlist[vsl_clipCounter].code, null, vsl_clipCounter + 1, null, null, null, null, null, null)
    }
}));
function initAd(a, c, b, d, e) {
    var f = vsl_playlist[vsl_clipCounter], f = {
        slot: playerWrap,
        LR_SCHEMA: "vast2-vpaid",
        LR_ENVIRONMENT: "flash",
        LR_VIDEO_POSITION: 0,
        LR_ADUNIT: a,
        LR_PUBLISHER_ID: 7042,
        LR_TAGS: anyclip.videoSenseLite.player.flashvars.pub_keyword,
        LR_DURATION: 60,
        LR_PARTNERS: anyclip.videoSenseLite.player.flashvars.partner,
        LR_CONTENT: "1",
        LR_AUTOPLAY: "1",
        LR_VIDEO_ID: f,
        LR_DESCRIPTION: vsl_getTitleName(f) + " " + vsl_getClipName(f),
        LR_VERTICALS: anyclip.videoSenseLite.player.flashvars.utmSource,
        LR_TITLE: vsl_getClipName(f),
        LR_VIDEO_URL: anyclip.videoSenseLite.player.flashvars.streamingProtocol + "://" + anyclip.videoSenseLite.player.flashvars.streamingServer + "/" + f + ".mp4",
        LR_URL: anyclip.videoSenseLite.player.flashvars.ref,
        LR_VIDEO_AMID: getStudioId(f) + "_" + getTitleCode(f)
    };
    "ov" != a ? ("undefined" !== typeof b && (f.LR_VIDEO_POSITION = b), adType = 0 == b ? "Preroll" : Math.round(b) == Math.round(c) ? "Postroll" : "Midroll") : adType = "Overlay";
    currentAdUnit = a;
    "undefined" !== typeof vsl_params.pr && (f.LR_PARTNERS = vsl_params.pr);
    LiveRailVPAID.initAd(d, e,
    "normal", 600, {}, f)
}
function onVPAIDLoad() {
    document.getElementById("mute_btn").onclick = function() {
        0 === LiveRailVPAID.getAdVolume() ? (player.unmute(), LiveRailVPAID.setAdVolume(1), this.src = "http://player.anyclip.com/video-sense/img/mute_off.png") : (player.mute(), LiveRailVPAID.setAdVolume(0), this.src = "http://player.anyclip.com/video-sense/img/mute_on.png")
    };
    LiveRailVPAID.subscribe(function() {
        LiveRailVPAID.startAd()
    }, "AdLoaded");
    LiveRailVPAID.subscribe(function() {
        vsl_reportEvent(vsl_params.uid, "ad" + adType + "Started-" + (vsl_clipCounter +
        1).toString(), vsl_getRuntime(), vsl_playlist[vsl_clipCounter]);
        "ov" !== currentAdUnit && (player.isMuted() ? (LiveRailVPAID.setAdVolume(0), document.getElementById("mute_btn").src = "http://player.anyclip.com/video-sense/img/mute_on.png") : (LiveRailVPAID.setAdVolume(1), document.getElementById("mute_btn").src = "http://player.anyclip.com/video-sense/img/mute_off.png"), document.getElementById("mute_btn").style.display = "block");
        adWrap = playerWrap.lastChild;
        "ov" == currentAdUnit ? adWrap.className = "overlayAd" : (adWrap.className =
        "instreamAd", player.pause())
    }, "AdStarted");
    LiveRailVPAID.subscribe(function() {
        document.getElementById("mute_btn").style.display = "none";
        vsl_reportEvent(vsl_params.uid, "ad" + adType + "Ended-" + (vsl_clipCounter + 1).toString(), vsl_getRuntime(), vsl_playlist[vsl_clipCounter]);
        setTimeout(function() {
            var a = playerWrap.getElementsByTagName("div");
            for (j = 0; j < a.length; j++)
                a[j].className = "overlayAd"
        }, 1E3);
        player.instreamAdEnded()
    }, "AdStopped");
    LiveRailVPAID.subscribe(function() {
        document.getElementById("mute_btn").style.display =
        "none";
        vsl_reportEvent(vsl_params.uid, "ad" + adType + "Failed-" + (vsl_clipCounter + 1).toString(), vsl_getRuntime(), vsl_playlist[vsl_clipCounter]);
        if ("undefined" !== typeof anyclip.videoSenseLite.items["item_" + vsl_params.uid].pb) {
            var a = anyclip.videoSenseLite.items["item_" + vsl_params.uid].pb;
            vsl_reportEvent(vsl_params.uid, "loadPassbackTag-" + (vsl_clipCounter + 1).toString(), vsl_getRuntime(), vsl_playlist[vsl_clipCounter]);
            document.getElementById("player_wrap_" + vsl_params.uid).innerHTML = '<iframe frameborder=0 width=300 height=250 marginheight=0 marginwidth=0 scrolling=no src="http://player.anyclip.com/video-sense/html/' +
            a + '.html"></iframe>'
        } else 
            setTimeout(function() {
                var a = playerWrap.getElementsByTagName("div");
                for (j = 0; j < a.length; j++)
                    a[j].className = "overlayAd"
                }, 1E3), player.instreamAdFailed()
    }, "AdError")
}
if ("undefined" !== typeof vsl_params.ua) {
    var _qevents = _qevents || [];
    (function() {
        var a = document.createElement("script");
        a.src = ("https:" == document.location.protocol ? "https://secure" : "http://edge") + ".quantserve.com/quant.js";
        a.async=!0;
        a.type = "text/javascript";
        var c = document.getElementsByTagName("script")[0];
        c.parentNode.insertBefore(a, c)
    })();
    _qevents.push({
        qacct: "p-VWwkLrB4fwGx1"
    })
} else {
    var qacctImg = document.createElement("img");
    qacctImg.width = 1;
    qacctImg.height = 1;
    qacctImg.style.display = "none";
    qacctImg.src =
    "http://pixel.quantserve.com/pixel/p-VWwkLrB4fwGx1.gif"
}(function() {
    var a = document.createElement("script"), c = "unknown";
    null != vsl_params.utm_source && (c = vsl_params.utm_source);
    a.src = "http://pixel.adsafeprotected.com/jload?anId=6889&advId=" + anyclip.videoSenseLite.player.flashvars.partner + "&campId=" + vsl_params.uid + "&pubId=" + anyclip.videoSenseLite.player.flashvars.pub_keyword + "&chanId=1&placementId=" + c;
    a.async=!0;
    a.type = "text/javascript";
    c = document.getElementsByTagName("script")[0];
    c.parentNode.insertBefore(a, c)
})();
for (var ga_is_loaded=!1, vsl_i = 0; vsl_i < vsl_scripts.length; vsl_i++)
    if (vsl_gaScript = vsl_scripts[vsl_i], - 1 !== vsl_gaScript.src.indexOf("ga.js")) {
        ga_is_loaded=!0;
        break
    }
if ("undefined" !== typeof vsl_params.ua&&!ga_is_loaded) {
    var ga = document.createElement("script");
    ga.type = "text/javascript";
    ga.async=!0;
    ga.src = ("https:" == document.location.protocol ? "https://ssl" : "http://www") + ".google-analytics.com/ga.js";
    var s = document.getElementsByTagName("script")[0];
    s.parentNode.insertBefore(ga, s)
}
var _gaq = _gaq || [];
"undefined" !== typeof vsl_params.ua && ("undefined" !== typeof vsl_params.utm ? _gaq.push(["b._setAccount", "UA-" + vsl_params.ua], ["b._set", "campaignParams", http_build_query(vsl_params.utm)], ["b._trackPageview"]) : _gaq.push(["b._setAccount", "UA-" + vsl_params.ua], ["b._trackPageview"]));
function vsl_reportEvent(a, c, b, d, e) {
    if ("undefined" !== typeof anyclip.videoSenseLite.items["item_" + a].ua) {
        var f = ["b._trackEvent", "flat_" + anyclip.videoSenseLite.items["item_" + a].type + "_2.0.0", c + "-" + a, void 0 != e && null != e ? e: d, void 0 != b && null != b ? b: vsl_getRuntime()];
        _gaq.push(f)
    }
    f = "";
    b = {
        action: c,
        label: void 0 != e && null != e ? e: d,
        value: void 0 != b && null != b ? b: vsl_getRuntime(),
        clipCode: void 0 != d && null != d ? d: null
    };
    f = http_build_query(logInfo) + "&" + http_build_query(b);
    b = document.createElement("img");
    b.width = 1;
    b.height =
    1;
    b.style.display = "none";
    b.src = "qa" == anyclip.videoSenseLite.items["item_" + a].vr ? "http://statsqa.anyclip.com/fictivious.png?" + f : "http://stats.anyclip.com/fictivious.png?" + f;
    b.onerror = function() {};
    b.onload = function() {};
    a = vsl_playlist[vsl_clipCounter];
    anyclip.videoSenseLite.clipData[a] ? (b = anyclip.videoSenseLite.clipData[a], a = b.ut, b = b.u) : b = a = "Unknown";
    d = "";
    e = logInfo.player_type;
    var f = encodeURIComponent(window.location.href), p = document.title, g = logInfo.refParent ? logInfo.refParent: logInfo.ref;
    0 === c.indexOf("clipStarted") ?
    d = "03" : 0 === c.indexOf("adPrerollStarted") ? d = "09" : 0 === c.indexOf("adMidrollStarted") ? d = "11" : 0 === c.indexOf("adPostrollStarted") && (d = "10");
    d && document.createElement("img").setAttribute("src", "http://b.scorecardresearch.com/p?c1=1&c2=17694432&c3=" + a + "&c4=" + b + "&c5=" + d + "&c6=" + e + "&c7=" + f + "&c8=" + p + "&c9=" + g + "&cv=2.0")
}
anyclip.videoSenseLite.stage.on("checkPlayerType", function() {
    if (vsl_custom)
        anyclip.videoSenseLite.stage.trigger("loadCustomPlayer");
    else if ("uilrlite" == vsl_params.type) {
        var a = createCORSRequest("get", "http://apis.anyclip.com/api/clip/public/" + vsl_playlist[vsl_clipCounter] + "?format=JSON");
        a ? (a.onload = function() {
            var c = JSON.parse(a.responseText);
            anyclip.videoSenseLite.clipData[c.code] = {};
            anyclip.videoSenseLite.clipData[c.code] = c;
            anyclip.videoSenseLite.clipData[c.code].u = c.uniqueName;
            anyclip.videoSenseLite.clipData[c.code].ut =
            c.title.uniqueName;
            anyclip.videoSenseLite.clipData[c.code].s = c.title.studioId;
            anyclip.videoSenseLite.clipData[c.code].m = c.title.code.substr(1, 4);
            anyclip.videoSenseLite.player.flashvars.buyLinks = encodeURIComponent(JSON.stringify(c.title.buyLinks));
            anyclip.videoSenseLite.player.flashvars.titlePoster = c.title.poster;
            anyclip.videoSenseLite.player.flashvars.titleName = c.title.name;
            anyclip.videoSenseLite.player.flashvars.clipName = c.name;
            anyclip.videoSenseLite.player.flashvars.titleCode = getTitleCode(vsl_playlist[vsl_clipCounter]);
            anyclip.videoSenseLite.player.flashvars.studioID = getStudioId(vsl_playlist[vsl_clipCounter]);
            anyclip.videoSenseLite.stage.trigger("loadPlayer")
        }, a.send()) : anyclip.videoSenseLite.stage.trigger("loadPlayer")
    } else 
        anyclip.videoSenseLite.player.flashvars.titleName = vsl_getTitleName(vsl_playlist[vsl_clipCounter]), anyclip.videoSenseLite.player.flashvars.clipName = vsl_getClipName(vsl_playlist[vsl_clipCounter]), anyclip.videoSenseLite.player.flashvars.titleCode = getTitleCode(vsl_playlist[vsl_clipCounter]), anyclip.videoSenseLite.player.flashvars.studioID =
        getStudioId(vsl_playlist[vsl_clipCounter]), anyclip.videoSenseLite.stage.trigger("loadPlayer")
});
anyclip.videoSenseLite.stage.on("loadPlayer", function() {
    ready(function() {
        var a = document.createElement("link");
        a.type = "text/css";
        a.rel = "stylesheet";
        a.href = "http://player.anyclip.com/video-sense/css/vs2.css";
        a.media = "screen";
        document.getElementsByTagName("head")[0].appendChild(a);
        var a = document.createElement("div"), c = document.createElement("img"), b = document.createElement("img"), d = document.createElement("div"), e = document.createElement("div"), f = document.createElement("img");
        f.id = "mute_btn";
        f.style.position =
        "absolute";
        f.style.top = "11px";
        f.style.right = "10px";
        f.style.zIndex = "9999999999";
        f.style.display = "none";
        f.style.cursor = "pointer";
        f.src = "http://player.anyclip.com/video-sense/img/mute_on.png";
        a.id = "ac_results_" + vsl_params.uid;
        a.className = "ac_results_id";
        c.id = "ac_stats_" + vsl_params.uid;
        c.width = 1;
        c.height = 1;
        c.style.display = "none";
        b.id = "ac_mg_stats_" + vsl_params.uid;
        b.width = 1;
        b.height = 1;
        b.style.display = "none";
        d.className = "player_wrap";
        d.id = "player_wrap_" + vsl_params.uid;
        d.style.width = vsl_width + "px";
        d.style.height =
        vsl_height + "px";
        d.style.overflow = "hidden";
        d.style.backgroundColor = "#000000";
        e.id = "acp_" + vsl_params.uid;
        vsl_myScript.parentNode.style.position = "relative";
        vsl_myScript.parentNode.style.width = vsl_width + "px";
        vsl_myScript.parentNode.style.height = vsl_height + "px";
        d.appendChild(e);
        vsl_myScript.parentNode.appendChild(a);
        vsl_myScript.parentNode.appendChild(c);
        vsl_myScript.parentNode.appendChild(b);
        vsl_myScript.parentNode.appendChild(d);
        vsl_myScript.parentNode.appendChild(f);
        vsl_reportEvent(vsl_params.uid, "widgetLoaded",
        vsl_getRuntime(), "");
        anyclip.videoSenseLite.player.flashvars.clipId = vsl_playlist[vsl_clipCounter];
        anyclip.videoSenseLite.player.flashvars.clipPageUrl = vsl_buildUrl(vsl_playlist[vsl_clipCounter]);
        anyclip.videoSenseLite.player.flashvars.player_position = vsl_clipCounter + 1;
        anyclip.videoSenseLite.player.flashvars.streamName = vsl_playlist[vsl_clipCounter] + ".mp4";
        anyclip.videoSenseLite.player.flashvars.titleCode = getTitleCode(vsl_playlist[vsl_clipCounter]);
        anyclip.videoSenseLite.player.flashvars.studioID = getStudioId(vsl_playlist[vsl_clipCounter]);
        playerWrap = document.getElementById("player_wrap_" + vsl_params.uid);
        vsl_fdrv.setAttributes({
            flash_width: vsl_width,
            flash_height: vsl_height,
            flash_flashvars: anyclip.videoSenseLite.player.flashvars,
            placementId: "acp_" + vsl_params.uid,
            swfPath: anyclip.videoSenseLite.player.url
        });
        vsl_fdrv.addEventsToFlashvars();
        var p=!0;
        if (vsl_ie7)
            vsl_reportEvent(vsl_params.uid, "iasError", vsl_getRuntime(), "IE7");
        else {
            var g = createCORSRequest("get", "http://api.adsafeprotected.com/db/client/27012/itgrl.json?adsafe_url=" + vsl_params.iasRef);
            g ? (g.onload = function() {
                if (0 < g.responseText.indexOf('bsc":')) {
                    var a = JSON.parse(g.responseText), a = JSON.parse(g.responseText), b = "", c = "";
                    if ("undefined" !== typeof a.bsc) {
                        for (var d in a.bsc)
                            c += b + a.bsc[d], b = ",", a.bsc[d] < vsl_params.ig && (p=!1);
                        p ? (vsl_reportEvent(vsl_params.uid, "iasPassed", vsl_getRuntime(), c), vsl_fdrv.embed()) : vsl_reportEvent(vsl_params.uid, "iasFailed", vsl_getRuntime(), c)
                    } else 
                        vsl_reportEvent(vsl_params.uid, "iasError", vsl_getRuntime(), "No data"), vsl_fdrv.embed()
                } else 
                    vsl_reportEvent(vsl_params.uid,
                    "iasError", vsl_getRuntime(), this.responseText), vsl_fdrv.embed()
            }, g.onerror = function() {
                vsl_reportEvent(vsl_params.uid, "iasError", vsl_getRuntime(), "Error in request");
                vsl_fdrv.embed()
            }, g.send()) : (vsl_reportEvent(vsl_params.uid, "iasError", vsl_getRuntime(), "Error in request"), vsl_fdrv.embed())
        }
    })
});
anyclip.videoSenseLite.stage.on("loadCustomPlayer", function() {
    ready(function() {
        var a = document.createElement("div"), c = document.createElement("img"), b = document.createElement("img"), d = document.createElement("div"), e = document.createElement("div");
        a.id = "ac_results_" + vsl_params.uid;
        a.className = "ac_results_id";
        c.id = "ac_stats_" + vsl_params.uid;
        c.width = 1;
        c.height = 1;
        c.style.display = "none";
        b.id = "ac_mg_stats_" + vsl_params.uid;
        b.width = 1;
        b.height = 1;
        b.style.display = "none";
        d.className = "player_wrap";
        d.id = "player_wrap_" +
        vsl_params.uid;
        d.style.width = vsl_width + "px";
        d.style.height = vsl_height + "px";
        d.style.overflow = "hidden";
        d.style.backgroundColor = "#000000";
        e.id = "acp_" + vsl_params.uid;
        vsl_myScript.parentNode.style.position = "relative";
        vsl_myScript.parentNode.style.width = vsl_width + "px";
        vsl_myScript.parentNode.style.height = vsl_height + "px";
        d.appendChild(e);
        vsl_myScript.parentNode.appendChild(a);
        vsl_myScript.parentNode.appendChild(c);
        vsl_myScript.parentNode.appendChild(b);
        vsl_myScript.parentNode.appendChild(d);
        vsl_reportEvent(vsl_params.uid,
        "widgetLoaded", vsl_getRuntime(), "");
        anyclip.videoSenseLite.player.flashvars.clipId = vsl_playlist[vsl_clipCounter].code;
        anyclip.videoSenseLite.player.flashvars.player_position = vsl_clipCounter + 1;
        anyclip.videoSenseLite.player.flashvars.streamingProtocol = vsl_playlist[vsl_clipCounter].scheme;
        anyclip.videoSenseLite.player.flashvars.streamingServer = vsl_playlist[vsl_clipCounter].server;
        "wpc.6acd.edgecastcdn.net/006ACD/tok" === anyclip.videoSenseLite.player.flashvars.streamingServer && (anyclip.videoSenseLite.player.flashvars.streamingServer =
        "anyclip.fcod.llnwd.net/a8895/e4", anyclip.videoSenseLite.player.flashvars.streamingProtocol = "rtmp");
        anyclip.videoSenseLite.player.flashvars.streamName = vsl_playlist[vsl_clipCounter].filename;
        anyclip.videoSenseLite.player.flashvars.titleCode = getTitleCode(vsl_playlist[vsl_clipCounter]);
        anyclip.videoSenseLite.player.flashvars.studioID = getStudioId(vsl_playlist[vsl_clipCounter]);
        vsl_fdrv.setAttributes({
            flash_width: vsl_width,
            flash_height: vsl_height,
            flash_flashvars: anyclip.videoSenseLite.player.flashvars,
            placementId: "acp_" + vsl_params.uid,
            swfPath: anyclip.videoSenseLite.player.url
        });
        vsl_fdrv.addEventsToFlashvars();
        var f=!0;
        if (vsl_ie7)
            vsl_reportEvent(vsl_params.uid, "iasError", vsl_getRuntime(), "IE7");
        else {
            var p = createCORSRequest("get", "http://api.adsafeprotected.com/db/client/27012/itgrl.json?adsafe_url=" + vsl_params.iasRef);
            p ? (p.onload = function() {
                if (0 < p.responseText.indexOf('bsc":')) {
                    var a = JSON.parse(p.responseText), a = JSON.parse(p.responseText), b = "", c = "";
                    if ("undefined" !== typeof a.bsc) {
                        for (var d in a.bsc)
                            c +=
                            b + a.bsc[d], b = ",", a.bsc[d] < vsl_params.ig && (f=!1);
                        f ? (vsl_reportEvent(vsl_params.uid, "iasPassed", vsl_getRuntime(), c), vsl_fdrv.embed()) : vsl_reportEvent(vsl_params.uid, "iasFailed", vsl_getRuntime(), c)
                    } else 
                        vsl_reportEvent(vsl_params.uid, "iasError", vsl_getRuntime(), "No data"), vsl_fdrv.embed()
                } else 
                    vsl_reportEvent(vsl_params.uid, "iasError", vsl_getRuntime(), this.responseText), vsl_fdrv.embed()
            }, p.onerror = function() {
                vsl_reportEvent(vsl_params.uid, "iasError", vsl_getRuntime(), "Error in request");
                vsl_fdrv.embed()
            },
            p.send()) : (vsl_reportEvent(vsl_params.uid, "iasError", vsl_getRuntime(), "Error in request"), vsl_fdrv.embed())
        }
    })
});
anyclip.videoSenseLite.stripHtml = function(a) {
    var c = document.createElement("DIV");
    c.innerHTML = a;
    return c.textContent || c.innerText
};
anyclip.videoSenseLite.countInstances = function(a, c) {
    return a.split(c).length
};
anyclip.videoSenseLite.getMetaDescription = function() {
    for (var a = "", c = document.getElementsByTagName("meta"), b = 0, d = c.length; b < d; b++)
        if ("description" == c[b].name.toLowerCase()) {
            a = "" + c[b].content;
            break
        }
    return a.toLowerCase()
};
anyclip.videoSenseLite.getParsedKeywords = function(a, c, b) {
    a = a.replace(/[^a-zA-Z0-9]/g, " ");
    a = a.split(" ");
    myText = c + b;
    keywords = {};
    for (c = 0; c < a.length; c++)
        if (b = a[c], (b = b.trim()) && 2 < b.length&&!anyclip.videoSenseLite.stops[b]) {
            var d = anyclip.videoSenseLite.countInstances(myText, b);
            keywords[b] ? keywords[b]++ : keywords[b] = d
        }
    a = [];
    for (var e in keywords)
        a.push([e, keywords[e]]);
    a.sort(function(a, b) {
        return b[1] - a[1]
    });
    e = "";
    c = 0;
    b = "";
    for (x in a)
        a[x][1] > c && (c = a[x][1]), a[x][1] > .8 * c && (e += b + a[x][0]), b = " ";
    return e
};
anyclip.videoSenseLite.getParagrapthsFromText = function() {
    for (var a = document.getElementsByTagName("p"), c = "", b = "", d = 0; d < a.length; d++)
        b += c + anyclip.videoSenseLite.stripHtml(a[d].innerHTML), c = " ";
    return b.toLowerCase()
};
anyclip.videoSenseLite.stops = {
    about: 1,
    above: 1,
    across: 1,
    after: 1,
    afterwards: 1,
    again: 1,
    against: 1,
    all: 1,
    almost: 1,
    alone: 1,
    along: 1,
    already: 1,
    also: 1,
    although: 1,
    always: 1,
    among: 1,
    amongst: 1,
    amoungst: 1,
    amount: 1,
    and: 1,
    another: 1,
    any: 1,
    anyhow: 1,
    anyone: 1,
    anything: 1,
    anyway: 1,
    anywhere: 1,
    are: 1,
    around: 1,
    back: 1,
    became: 1,
    because: 1,
    become: 1,
    becomes: 1,
    becoming: 1,
    been: 1,
    before: 1,
    beforehand: 1,
    behind: 1,
    being: 1,
    below: 1,
    beside: 1,
    besides: 1,
    between: 1,
    beyond: 1,
    bill: 1,
    both: 1,
    bottom: 1,
    but: 1,
    call: 1,
    can: 1,
    cannot: 1,
    cant: 1,
    computer: 1,
    con: 1,
    could: 1,
    couldnt: 1,
    cry: 1,
    describe: 1,
    detail: 1,
    done: 1,
    down: 1,
    due: 1,
    during: 1,
    each: 1,
    eight: 1,
    either: 1,
    eleven: 1,
    "else": 1,
    elsewhere: 1,
    empty: 1,
    enough: 1,
    etc: 1,
    even: 1,
    ever: 1,
    every: 1,
    everyone: 1,
    everything: 1,
    everywhere: 1,
    except: 1,
    few: 1,
    fifteen: 1,
    fify: 1,
    fill: 1,
    find: 1,
    fire: 1,
    first: 1,
    five: 1,
    "for": 1,
    former: 1,
    formerly: 1,
    forty: 1,
    found: 1,
    four: 1,
    from: 1,
    front: 1,
    full: 1,
    further: 1,
    get: 1,
    give: 1,
    had: 1,
    has: 1,
    hasnt: 1,
    have: 1,
    hence: 1,
    her: 1,
    here: 1,
    hereafter: 1,
    hereby: 1,
    herein: 1,
    hereupon: 1,
    hers: 1,
    herse: 1,
    him: 1,
    himse: 1,
    his: 1,
    how: 1,
    however: 1,
    hundred: 1,
    inc: 1,
    indeed: 1,
    interest: 1,
    into: 1,
    its: 1,
    itse: 1,
    keep: 1,
    last: 1,
    latter: 1,
    latterly: 1,
    least: 1,
    less: 1,
    ltd: 1,
    made: 1,
    many: 1,
    may: 1,
    meanwhile: 1,
    might: 1,
    mill: 1,
    mine: 1,
    more: 1,
    moreover: 1,
    most: 1,
    mostly: 1,
    move: 1,
    much: 1,
    must: 1,
    myse: 1,
    name: 1,
    namely: 1,
    neither: 1,
    never: 1,
    nevertheless: 1,
    next: 1,
    nine: 1,
    nobody: 1,
    none: 1,
    noone: 1,
    nor: 1,
    not: 1,
    nothing: 1,
    now: 1,
    nowhere: 1,
    off: 1,
    often: 1,
    once: 1,
    one: 1,
    only: 1,
    onto: 1,
    other: 1,
    others: 1,
    otherwise: 1,
    our: 1,
    ours: 1,
    ourselves: 1,
    out: 1,
    over: 1,
    own: 1,
    part: 1,
    per: 1,
    perhaps: 1,
    please: 1,
    put: 1,
    rather: 1,
    same: 1,
    see: 1,
    seem: 1,
    seemed: 1,
    seeming: 1,
    seems: 1,
    serious: 1,
    several: 1,
    she: 1,
    should: 1,
    show: 1,
    side: 1,
    since: 1,
    sincere: 1,
    six: 1,
    sixty: 1,
    some: 1,
    somehow: 1,
    someone: 1,
    something: 1,
    sometime: 1,
    sometimes: 1,
    somewhere: 1,
    still: 1,
    such: 1,
    system: 1,
    take: 1,
    ten: 1,
    than: 1,
    that: 1,
    the: 1,
    their: 1,
    them: 1,
    themselves: 1,
    then: 1,
    thence: 1,
    there: 1,
    thereafter: 1,
    thereby: 1,
    therefore: 1,
    therein: 1,
    thereupon: 1,
    these: 1,
    they: 1,
    thick: 1,
    thin: 1,
    third: 1,
    "this": 1,
    those: 1,
    though: 1,
    three: 1,
    through: 1,
    throughout: 1,
    thru: 1,
    thus: 1,
    together: 1,
    too: 1,
    top: 1,
    toward: 1,
    towards: 1,
    twelve: 1,
    twenty: 1,
    two: 1,
    under: 1,
    until: 1,
    upon: 1,
    very: 1,
    via: 1,
    was: 1,
    well: 1,
    were: 1,
    what: 1,
    whatever: 1,
    when: 1,
    whence: 1,
    whenever: 1,
    where: 1,
    whereafter: 1,
    whereas: 1,
    whereby: 1,
    wherein: 1,
    whereupon: 1,
    wherever: 1,
    whether: 1,
    which: 1,
    "while": 1,
    whither: 1,
    who: 1,
    whoever: 1,
    whole: 1,
    whom: 1,
    whose: 1,
    why: 1,
    will: 1,
    "with": 1,
    within: 1,
    without: 1,
    would: 1,
    yet: 1,
    you: 1,
    your: 1,
    yours: 1,
    yourself: 1,
    yourselves: 1
};
"undefined" === typeof vsl_params.se || 1 !== parseInt(vsl_params.se) || vsl_ie7 || (vsl_params.q = anyclip.videoSenseLite.getParsedKeywords(document.title.toLowerCase(), anyclip.videoSenseLite.getMetaDescription(), anyclip.videoSenseLite.getParagrapthsFromText()));
if ("undefined" === typeof vsl_params.q || vsl_ie7)
    anyclip.videoSenseLite.stage.trigger("checkPlayerType");
else {
    var request = createCORSRequest("get", "http://apis.anyclip.com/api/search/public/widget?format=JSON&maxResults=5&q=" + vsl_params.q);
    request ? (request.onload = function() {
        for (var a = JSON.parse(request.responseText).clips.result, c = [], b = 0; b < a.length; b++)
            c.push(a[b].code), anyclip.videoSenseLite.clipData[a[b].code] = {}, anyclip.videoSenseLite.clipData[a[b].code].u = a[b].uniqueName, anyclip.videoSenseLite.clipData[a[b].code].ut =
            a[b].title.uniqueName, anyclip.videoSenseLite.clipData[a[b].code].m = a[b].title.code.substr(1, 4);
        vsl_playlist = shuffle(c);
        anyclip.videoSenseLite.stage.trigger("checkPlayerType")
    }, request.send()) : anyclip.videoSenseLite.stage.trigger("checkPlayerType")
};


