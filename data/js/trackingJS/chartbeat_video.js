(function() {
    var d = void 0, g = true, i = null, j = false;
    function k() {
        return function() {}
    }
    function m(a) {
        return function() {
            return this[a]
        }
    }
    function n(a) {
        return function() {
            return a
        }
    }
    var p, q = this;
    function aa(a) {
        var b = typeof a;
        if (b == "object")
            if (a) {
                if (a instanceof Array)
                    return "array";
                else if (a instanceof Object)
                    return b;
                    var c = Object.prototype.toString.call(a);
                    if (c == "[object Window]")
                        return "object";
                        if (c == "[object Array]" || typeof a.length == "number" && typeof a.splice != "undefined" && typeof a.propertyIsEnumerable != "undefined"&&!a.propertyIsEnumerable("splice"))
                            return "array";
                            if (c == "[object Function]" || typeof a.call != "undefined" && typeof a.propertyIsEnumerable != "undefined"&&!a.propertyIsEnumerable("call"))
                                return "function"
            } else 
                return "null";
                else if (b == "function" && typeof a.call == "undefined")
    return "object";
return b
}
function ba(a, b, c) {
    return a.call.apply(a.bind, arguments)
}
function ca(a, b, c) {
    if (!a)
        throw Error();
    if (arguments.length > 2) {
        var e = Array.prototype.slice.call(arguments, 2);
        return function() {
            var c = Array.prototype.slice.call(arguments);
            Array.prototype.unshift.apply(c, e);
            return a.apply(b, c)
        }
    } else 
        return function() {
            return a.apply(b, arguments)
        }
}
function r(a, b, c) {
    r = Function.prototype.bind && Function.prototype.bind.toString().indexOf("native code")!=-1 ? ba : ca;
    return r.apply(i, arguments)
}
function da(a) {
    var b = ea;
    function c() {}
    c.prototype = b.prototype;
    a.aa = b.prototype;
    a.prototype = new c;
    a.prototype.constructor = a
};
function fa(a, b) {
    var c;
    c = new Image(1, 1);
    if (b)
        c.onerror = b;
    c.src = a
}
function ga() {
    for (var a = "", b = 0; b < 16; b++)
        a += Math.random();
    return a
}
function ha(a, b) {
    var c = "", e = ia(encodeURIComponent(a));
    e.splice(b || 5, e.length);
    s(e, function(a) {
        if (a == 0)
            a = "A";
        else {
            a>>>=0;
            for (var b = "", e; a > 0;)
                e = a%64, b = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-".charAt(e) + b, a>>>=6;
            a = b
        }
        c += a
    });
    return c
}
function ia(a) {
    a += String.fromCharCode(128);
    for (var b = [1518500249, 1859775393, 2400959708, 3395469782], c = 1732584193, e = 4023233417, f = 2562383102, h = 271733878, l = 3285377520, y = [], o, F, J, O, G, R = Math.ceil((a.length / 4 + 2) / 16), V = [R], P = 0, L; P < R; P++) {
        V[P] = [];
        for (o = 0; o < 16; o++)
            V[P][o] = a.charCodeAt(P * 64 + o * 4)<<24 | a.charCodeAt(P * 64 + o * 4 + 1)<<16 | a.charCodeAt(P * 64 + o * 4 + 2)<<8 | a.charCodeAt(P * 64 + o * 4 + 3)
    }
    P = (a.length - 1) * 8;
    a = R - 1;
    V[a][14] = Math.floor(P / Math.pow(2, 32));
    V[a][15] = P & 4294967295;
    for (P = 0; P < R; P++) {
        for (L = 0; L < 16; L++)
            y[L] = V[P][L];
        for (L = 16; L < 80; L++)
            y[L] = (y[L - 3]^y[L - 8]^y[L - 14]^y[L - 16])<<1 | (y[L - 3]^y[L - 8]^y[L - 14]^y[L - 16])>>>31;
        a = c;
        o = e;
        F = f;
        J = h;
        O = l;
        for (L = 0; L < 80; L++)
            G = Math.floor(L / 20), G = (a<<5 | a>>>27) + (G == 0 ? o & F^~o & J : G == 1 ? o^F^J : G == 2 ? o & F^o & J^F & J : o^F^J) + O + b[G] + y[L] & 4294967295, O = J, J = F, F = o<<30 | o>>>2, o = a, a = G;
        c = c + a & 4294967295;
        e = e + o & 4294967295;
        f = f + F & 4294967295;
        h = h + J & 4294967295;
        l = l + O & 4294967295
    }
    return [c, e, f, h, l]
}
function t(a, b) {
    return function() {
        a.apply(b, arguments)
    }
}
function u(a, b, c) {
    a.addEventListener ? a.addEventListener(b, c, j) : a.attachEvent && a.attachEvent("on" + b, c)
}
function ja(a, b, c) {
    a.removeEventListener ? a.removeEventListener(b, c, j) : a.detachEvent && a.detachEvent("on" + b, c)
}
function ka(a) {
    return typeof a === "number"
}
function la(a) {
    return typeof a === "string"
}
function ma(a) {
    return typeof a === "function"
}
function na(a) {
    a = new Date( + a);
    return Date.UTC(a.getFullYear(), a.getMonth(), a.getDate())
}
function v() {
    return (new Date).getTime()
}
function oa(a) {
    return encodeURIComponent(a)
}
function s(a, b) {
    if ((!!a && a.constructor === Object) === g)
        for (var c in a) {
            if (a.hasOwnProperty(c) && b(a[c], c) === j)
                break
        } else {
        c = 0;
        for (var e = a.length; c < e; c++)
            if (b(a[c], c) === j)
                break
        }
}
function pa(a, b) {
    for (var c = 0, e = a.length; c < e; c++)
        if (b(a[c]))
            return c;
    return - 1
}
function qa(a) {
    return a === d ? 0 : v() - a
}
function ra(a, b) {
    var c, e, f = "c" + Math.round(Math.random() * 1E9);
    e = 250;
    a += (a.indexOf("?")>-1 ? "&" : "?") + (c || "callback") + "=_cb_call." + f;
    c = document.createElement("script");
    var h = sa(f, c);
    c.id = f;
    c.type = "text/javascript";
    c.charset = "UTF-8";
    c.src = a;
    var l = document.getElementsByTagName("head");
    if (l = l && l[0]) {
        var y = q._cb_call || (q._cb_call = {});
        e = setTimeout(ta(b, h), e);
        y[f] = ua(b, h, e);
        l.appendChild(c)
    }
}
function ta(a, b) {
    return function() {
        b();
        a(d, "TIMEOUT")
    }
}
function ua(a, b, c) {
    return function(e) {
        b(c);
        a(e)
    }
}
function sa(a, b) {
    return function(c) {
        c && clearTimeout(c);
        setTimeout(function() {
            b && b.parentNode && b.parentNode.removeChild(b)
        }, 0);
        var e = q._cb_call, f = j;
        if (e && (c ? delete e[a] : e[a] = va, s(e, function() {
            f = g;
            return j
        }), !f))
            try {
                delete q._cb_call
        } catch (h) {
            q._cb_call = d
        }
    }
}
function va() {}
function wa(a, b) {
    if (a === b)
        return 0;
    if (a.length === 0)
        return b.length;
    if (b.length === 0)
        return a.length;
    for (var c = [], e = 0, f = b.length; e <= f; e++)
        c[e] = [e];
    for (var h = 0, l = a.length; h <= l; h++)
        c[0][h] = h;
    for (var y, o, F, e = 1; e <= f; e++)
        for (h = 1; h <= l; h++)
            y = e - 1, o = h - 1, F = c[y][o], b.charAt(y) == a.charAt(o) ? c[e][h] = F : (o = c[e][o] + 1, y = c[y][h] + 1, F += 2, c[e][h] = Math.min(o, y, F));
    return c[b.length][a.length]
}
function xa(a) {
    return a === d ? 0 : v() - a
}
function ya(a) {
    return Math.round(a===-1 || isNaN(a)?-1 : a * 1E3)
};
function za(a) {
    var b = {};
    a && (a.charAt(0) == "?" && (a = a.substring(1)), a = a.replace("+", " "), s(a.split(/[&;]/g), function(a) {
        a = a.split("=");
        b[decodeURIComponent(a[0])] = decodeURIComponent(a[1])
    }));
    return b
}
function Aa(a, b, c) {
    var e = "", f = q.location.href.split("?");
    f.length && (f = za(f[1]), b = c || b, f[b] && (e = "&" + a + "=" + f[b]));
    return e
}
function Ba(a, b) {
    return !b ? g : a === "http:" && b === "80" || a === "https:" && b === "443"
}
function Ca(a) {
    var b = {
        hostname: "",
        pathname: "",
        search: "",
        protocol: "",
        port: "",
        hash: ""
    };
    if (!a)
        return b;
    var c = document.createElement("a"), e = q.location;
    if (!/^https?:/.test(a) && a.indexOf("javascript:") !== 0)
        if (a.indexOf("//") === 0)
            a = e.protocol + a;
        else if (a.indexOf("/") === 0)
            var f = Ba(e.protocol, e.port) ? "": e.port, a = e.protocol + "//" + e.hostname + (f ? ":" + f : "") + a;
        else {
            var f = document.baseURI || e.href, h = f.indexOf("?");
            h===-1 && (h = f.indexOf("#"));
            if (h===-1)
                h = f.length;
                h = f.lastIndexOf("/", h);
                a = h===-1 ? "/" + a : f.substr(0, h) + "/" +
                a
        }
    c.href = a;
    b.hostname = c.hostname;
    b.pathname = c.pathname;
    b.search = c.search;
    b.protocol = c.protocol;
    b.port = c.port;
    b.hash = c.hash;
    if (b.pathname.charAt(0) !== "/")
        b.pathname = "/" + b.pathname;
    if (b.hostname === "")
        b.hostname = e.hostname;
    if (b.protocol === "")
        b.protocol = e.protocol;
    if (b.protocol === "javascript:")
        b.pathname = "", b.hostname = "", b.port = "", b.hash = "";
    if (Ba(b.protocol, b.port))
        b.port = "";
    return b
}
function Da(a) {
    var b = a.protocol;
    a.hostname && (b += "//" + a.hostname, a.port && (b += ":" + a.port));
    return b + a.pathname + a.search + a.hash
};
function w(a, b, c) {
    a[b] = a[b] || c
}
function Ea(a, b) {
    for (var c = q[a] || []; c.length;)
        b(c.shift());
    q[a] = {
        push: b
    }
}
function Fa(a) {
    s(document.getElementsByTagName("script"), function(b) {
        if (b.src.match(/chartbeat.js/))
            return b = za(b.src.split("?")[1]), w(a, "uid", b.uid), w(a, "domain", b.domain), j
    })
}
function Ga(a, b, c) {
    return a[c] ? "&g" + b + "=" + encodeURIComponent(a[c]) : ""
}
function Ha(a) {
    var b = [];
    s(a, function(a, e) {
        e.charAt(0) == "_" && b.push(e + "=" + a)
    });
    return b.length ? "&" + b.join("&") : ""
};
var x = {};
x.q = function(a) {
    var b = q._sf_async_config;
    if (!a && b && b.noCookies)
        return i;
    if (x.q.Aa !== d)
        return x.q.Aa;
    var a = v() + "", c, e;
    try {
        if ((e = q.localStorage).setItem(a, a), c = e.getItem(a) === a, e.removeItem(a), c)
            return x.q.Aa = e
    } catch (f) {}
    return x.q.Aa = i
};
x.e = function(a) {
    var b = x.q();
    if (!b)
        return "";
    var c = b.getItem(a + "_expires");
    return c && (c =+ c, !isNaN(c) && v() > c) ? (x.remove(a), "") : b.getItem(a) || ""
};
x.create = function(a, b, c) {
    var e = x.q();
    if (e) {
        var f = new Date;
        f.setTime(v() + c * 1E3);
        try {
            e.setItem(a, b), e.setItem(a + "_expires", f.getTime())
        } catch (h) {}
    }
};
x.remove = function(a) {
    var b = x.q();
    b && (b.removeItem(a), b.removeItem(a + "_expires"))
};
function Ia() {
    var a = document.createElement("script");
    a.async = g;
    a.src = (q.location.protocol || "http:") + "//static.chartbeat.com/js/inpage.js";
    var b = document.getElementsByTagName("script")[0];
    b.parentNode.insertBefore(a, b)
}
function Ja(a) {
    if (/[\/|\.]chartbeat\.com$/.test(a.origin)) {
        var b = x.q(g), c = String(a.data);
        b && c.indexOf("_cb_ip") == 0 && (b.setItem("_cb_ip", "1"), a.source.postMessage(1, a.origin), Ia())
    }
};
var z = {}, Ka = 1;
function A(a, b, c) {
    Ka++;
    z[a] = z[a] || {};
    z[a][Ka] = [b, c];
    return Ka
}
function La(a) {
    if (la(a))
        z[a] = d, delete z[a];
    else if (ka(a)) {
        var b = g;
        s(z, function(c) {
            s(c, function(e, f) {
                if (parseInt(f, 10) === a)
                    return c[f] = d, delete c[f], b = j
            });
            return b
        })
    }
}
function Ma(a, b) {
    if (z[a]) {
        var c = arguments.length > 1 ? Array.prototype.slice.call(arguments, 1): [];
        s(z[a], function(a) {
            var b;
            a && a.length === 2 && (b = a[0], a = a[1], b.apply(a, c))
        })
    }
};
var B = {};
B.A = function() {
    B.fa ? B.pa("pageload") : (B.Hb = [{
        target: q,
        event: "scroll"
    }, {
        target: document.body,
        event: "keydown"
    }, {
        target: document.body,
        event: "mousemove"
    }, {
        target: q,
        event: "resize"
    }, {
        target: document.body,
        event: "mousedown"
    }
    ], B.wa = i, B.Ta = j, B.ea = i, B.ob = {}, s(B.Hb, function(a) {
        var b = a.event;
        u(a.target, b, function(a) {
            B.pa.call(B, b, a)
        })
    }), A("focus", function() {
        B.pa("focus")
    }), B.pa("pageload"), u(document.body, "click", function(a) {
        a: if (a = a || window.event) {
            if (a = a.target || a.srcElement, a.tagName !== "A")
                if (a.parentNode)
                    if (a.parentNode.tagName === "A")
                        a =
                        a.parentNode;
                    else if (a.parentNode.parentNode && a.parentNode.parentNode.tagName === "A")
                        a = a.parentNode.parentNode;
                    else {
                        a = d;
                        break a
                    } else {
                        a = d;
                        break a
                    }
        } else 
            a = d;
        a && Ma("anchor", a)
    }), B.fa = g)
};
B.Mb = function() {
    var a, b = B.ob.keydown;
    if (b === d)
        return j;
    b = v() - b;
    return b <= (a || 15E3) && b >= 0
};
B.cb = 100;
B.pa = function(a, b) {
    if (!b)
        b = window.event;
    if (b && a === "keydown") {
        var c = b.keyCode ? b.keyCode: b.which;
        if (c === 32 || c > 36 && c < 41)
            a = "scroll"
    }
    B.Ec(a);
    B.wa === i ? B.Bb() : B.Ta = g
};
B.Ec = function(a) {
    B.ob[a] = v()
};
B.Bb = function() {
    B.wa = q.setTimeout(B.Qb, B.cb);
    Ma("activity");
    B.ea !== i && q.clearTimeout(B.ea);
    B.ea = q.setTimeout(function() {
        Ma("inactive");
        q.clearTimeout(B.ea);
        B.ea = i
    }, 5E3 + B.cb)
};
B.Qb = function() {
    q.clearTimeout(B.wa);
    B.wa = i;
    if (B.Ta)
        B.Ta = j, B.Bb()
};
var C = {};
C.Nb = function() {
    try {
        C.create("_cb_test", "1", 1);
        var a = C.e("_cb_test");
        C.remove("_cb_test");
        return a === "1"
    } catch (b) {
        return j
    }
};
C.e = function(a) {
    a += "=";
    var b = "";
    s(document.cookie.split(";"), function(c) {
        for (; c.charAt(0) === " ";)
            c = c.substring(1, c.length);
        if (c.indexOf(a) === 0)
            return b = c.substring(a.length, c.length), j
    });
    return b
};
C.create = function(a, b, c) {
    var e = q._sf_async_config;
    if (!e ||!e.noCookies)
        e = new Date, e.setTime(v() + c * 1E3), document.cookie = a + "=" + b + ("; expires=" + e.toGMTString()) + "; path=/"
};
C.remove = function(a) {
    C.e(a) && C.create(a, "", - 86400)
};
function Na(a) {
    this.ta = a || "";
    this.Rb = x.q() !== i || C.Nb();
    this.Sa = j;
    Oa(this)
}
p = Na.prototype;
p.isSupported = m("Rb");
function Oa(a) {
    if (!C.e("_cb_ls")) {
        var b = x.q() !== i, c = C.e("_SUPERFLY_nosample");
        c && s(["", "_v_", "_p_"], function(b) {
            a.create(b + "_SUPERFLY_nosample", c, 600, g)
        });
        var e = C.e("_chartbeat3");
        e && (a.create("_v__chartbeat3", e, 2592E3, g), C.remove("_chartbeat3"));
        b && ((b = C.e("_chartbeat2")) && a.create("_chartbeat2", b, 94608E3, g), (b = C.e("_chartbeat_uuniq")) && a.create("_chartbeat_uuniq", b, 94608E3, g), (b = C.e("_chartbeat5")) && a.create("_chartbeat5", b, 60, g));
        C.create("_cb_ls", "1", 2592E3)
    }
}
function Pa(a) {
    var b = a.Sa;
    a.Sa = j;
    return b
}
p.create = function(a, b, c, e) {
    a = e ? a : this.ta + a;
    (x.q() ? x : C).create(a, b, c);
    x.q() && C.create(a, b, c)
};
p.update = function(a, b, c, e, f, h) {
    a = e ? a : this.ta + a;
    f = la(f) ? f : "::";
    e = (e = this.e(a, g)) ? e.split(f) : [];
    ka(h) && e.length >= h && e.splice(0, e.length - h + 1);
    e.push(b);
    this.create(a, e.join(f), c, g)
};
p.e = function(a, b) {
    var a = b ? a: this.ta + a, c = (x.q() ? x : C).e(a);
    if (!c && x.q() && (c = C.e(a)) && C.e("_cb_ls")) {
        this.Sa = g;
        var e;
        switch (a) {
        case "_SUPERFLY_nosample":
            e = 600;
            break;
        case "_chartbeat4":
            e = 3600;
            break;
        case "_cb_cp":
            e = 3600;
            break;
        case "_chartbeat3":
            e = 2592E3;
            break;
        default:
            e = 94608E3
        }
        x.create(a, c, e)
    }
    return c
};
p.remove = function(a, b) {
    a = b ? a : this.ta + a;
    (x.q() ? x : C).remove(a);
    x.q() && C.remove(a)
};
function Qa(a, b) {
    var c, b = b || "*";
    c = c || document;
    if ("querySelectorAll"in c)
        return c.querySelectorAll(b + "[" + a + "]");
    var e = [];
    c = c.getElementsByTagName(b);
    for (var f = c.length; f--;)
        c[f].getAttribute(a) && e.push(c[f]);
    return e
}
function Ra(a, b, c) {
    return a === d ? j : c === d && a.getAttribute(b) || a.getAttribute(b) === c ? a : a === document.body ? j : Ra(a.parentNode, b, c)
}
function Sa(a) {
    var b, c;
    b = "pageYOffset";
    c = "scrollTop";
    if (a && (a = Qa("data-cb-scroll-element")) && a.length)
        return a[0][c];
    if (ka(q[b]))
        return q[b];
    else if (document.body && document.body[c])
        return document.body[c];
    else if (document.documentElement[c])
        return document.documentElement[c];
    return 0
}
function Ta() {
    var a = document, a = a[a.compatMode === "CSS1Compat" ? "documentElement": "body"].clientHeight || 0;
    window.innerHeight && (a = Math.min(window.innerHeight, a));
    return a
}
function Ua(a) {
    a = "scroll" + a;
    return Math.max(document.body[a], document.documentElement[a]) || 0
}
function Va(a, b) {
    var c;
    la(b) ? (b = b.toLowerCase(), c = function(a) {
        return (a = a.nodeName) && a.toLowerCase() === b
    }) : c = b;
    for (; a && a !== document.documentElement;) {
        if (c(a))
            return a;
        a = a.parentNode
    }
    return i
};
var D = function() {
    var a, b;
    s(["", "moz", "o", "ms", "webkit"], function(c) {
        a = (c + "Hidden").charAt(0).toLowerCase() + (c + "Hidden").slice(1);
        if (typeof document[a] === "boolean")
            return b = c, j
    });
    var c = {};
    if (b !== d)
        c.lb = a, c.ab = (b + "VisibilityState").charAt(0).toLowerCase() + (b + "VisibilityState").slice(1), c.xa = b + "visibilitychange";
    return c
}();
D.A = function() {
    if (!D.fa)
        D.na = D.ub(), D.xa && document.addEventListener && document.addEventListener(D.xa, D.zc, j), D.ib("focus", "onfocusin", D.Ja), D.ib("blur", "onfocusout", D.rb), D.na && D.Ja(), D.fa = g
};
D.Lc = function() {
    return D.na
};
D.Ja = function() {
    D.na = g;
    Ma("focus")
};
D.rb = function() {
    D.na = j;
    Ma("blur")
};
D.ib = function(a, b, c) {
    q.addEventListener ? q.addEventListener(a, c, j) : document.attachEvent && document.attachEvent(b, c)
};
D.ub = function() {
    var a = g;
    document.hasFocus && (a = document.hasFocus());
    var b = j;
    D.lb && (b = document[D.lb]);
    return a&&!b
};
D.zc = function() {
    D.ub() ? D.Ja() : D.rb()
};
function Wa(a) {
    var b = i;
    if (a && (b = Xa()))
        return b;
    var c = q.location, b = Ya(c.pathname), a = c.search || "", a = a.replace(/PHPSESSID=[^&]+/, ""), e = /&utm_[^=]+=[^&]+/ig;
    (c = e.exec(c.search)) && (a = a.replace(e, ""));
    e = /\?utm_[^=]+=[^&]+(.*)/i;
    (c = e.exec(a)) && (a = a.replace(e, c[1] != "" ? "?" + c[1] : ""));
    return b + a
}
function Xa() {
    var a = i;
    s(document.getElementsByTagName("link"), function(b) {
        if (b.rel == "canonical")
            return b = Ca(b.href), a = Ya(b.pathname) + b.search + b.hash, j
    });
    return a
}
for (var Za = {}, $a = 0; $a < 81; $a++)
    Za["0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-._~!$&'()*+,;=:@[]".charCodeAt($a)] = g;
function ab(a, b) {
    if (a === "%")
        return "%25";
    var c = parseInt(b, 16);
    return Za[c] ? String.fromCharCode(c) : "%" + b.toUpperCase()
}
function Ya(a) {
    if (!la(a))
        return a;
    a = a.replace(/%([0-9A-Fa-f]{2})?/g, ab);
    return a = a.replace(/[^0-9A-Za-z\-._~!$&'()*+,;=:@\/\[\]?#%]+/g, encodeURIComponent)
};
var E = {
    Jb: {
        IDLE: 0,
        Jc: 1,
        Ic: 2,
        Ib: 3
    },
    H: 0
};
E.A = function() {
    if (!E.fa)
        A("activity", E.Vb, E), A("inactive", E.Yb, E), A("focus", E.Xb, E), A("blur", E.Wb, E), E.fa = g
};
E.h = function() {
    return E.H
};
E.Vb = function() {
    E.H === 1 ? E.U(3) : E.H === 0 && E.U(2)
};
E.Yb = function() {
    E.H === 3 ? E.U(1) : E.H === 2 && E.U(0)
};
E.Xb = function() {
    (E.H === 0 || E.H === 2) && E.U(3)
};
E.Wb = function() {
    E.H === 3 ? E.U(2) : E.H === 1 && E.U(0)
};
E.U = function(a) {
    E.H = a;
    Ma("state", a)
};
function bb(a, b) {
    this.Ea = a || d;
    this.Ka = b || d;
    this.Y = this.Xa = 0
}
p = bb.prototype;
p.A = function() {
    this.Y = this.Xa = 0;
    this.ia = i;
    this.Cc = A("state", this.kb, this);
    this.kb(E.h())
};
p.total = function() {
    this.Xa += this.Y;
    this.Y = 0;
    return this.Xa
};
p.kb = function(a) {
    q.clearInterval(this.ia);
    this.ia = i;
    if (a === E.Jb.Ib)
        this.ia = q.setInterval(t(this.bc, this), 1E3)
};
p.bc = function() {
    if (this.Ea === d || this.Ea())
        this.Y++, this.Ka && this.Ka()
};
p.terminate = function() {
    q.clearInterval(this.ia);
    this.ia = i;
    La(this.Cc)
};
p.ca = function() {
    this.terminate();
    this.Ka = this.Ea = d
};
function ea() {
    this.b = q._sf_async_config || {};
    this.Da = t(this.qc, this);
    this.X = [];
    this.g = new Na(this.Cb);
    this.T = j;
    this.O = new bb;
    this.Ma = t(this.ra, this);
    u(q, "unload", this.Ma);
    cb(this) || this.A()
}
function cb(a) {
    if (D.ab && document[D.ab] === "prerender") {
        a.T = g;
        var b = t(function() {
            if (this.T && document[D.ab] !== "prerender")
                this.T = j, this.A(), db(this), window.setTimeout(function() {
                    document.removeEventListener(D.xa, b, j)
                }, 100)
        }, a);
        document.addEventListener(D.xa, b, j);
        return g
    }
    return j
}
p = ea.prototype;
p.A = function() {
    this.Fa = this.Z = 0;
    this.va = v();
    this.$ = ha(eb(this));
    this.Qa = g;
    this.Ga = 72E5;
    var a =+ this.b.sessionLength;
    if (!isNaN(a) && this.Cb !== "_p_")
        this.Ga = a * 6E4;
    this.O.A();
    E.A();
    B.A();
    D.A();
    this.ha = 0;
    this.Lb = A("activity", this.dc, this)
};
function db(a) {
    if (!a.g.e("_SUPERFLY_nosample")&&!a.T)
        a.cc ? a.Ra() : (a.cc = g, !q._sf_async_config&&!q._cbq ? (a.qa = t(a.Ra, a), u(q, "load", a.qa)) : a.Ra())
}
p.Ra = function() {
    var a = q._sf_startpt, b = q._sf_endpt;
    if (ka(a))
        this.sa = ka(b) ? b - a : v() - a;
    B.A();
    this.ga = setInterval(t(this.K, this), 15E3);
    this.K()
};
p.K = function() {
    var a = this.O.Y, a = this.b.reading&&+this.b.reading || a > 0;
    this.Fa < this.Z&&!a ? this.Fa++ : (a ? this.Z = 0 : this.za(), this.Fa = 0, this.X.push(0), this.X.length > 18 && this.X.shift(), v() - this.va >= this.Ga ? this.terminate() : this.vb())
};
p.qc = function() {
    this.X.push(1);
    var a = 0;
    s(this.X, function(b) {
        a += b
    });
    a < 3 ? (this.Qa = g, this.za()) : (this.terminate(), this.g.create("_SUPERFLY_nosample", "1", 600))
};
p.ra = k();
p.dc = function() {
    var a = fb(this);
    this.ha = Math.max(this.ha, a)
};
function fb(a) {
    return Sa(!!a.b.scrollElement)
}
p.za = function() {
    var a = this.Z;
    this.Z = a = a ? Math.min(a * 2, 16) : 1
};
p.Fc = function(a, b) {
    if (!this.T)
        this.ra(), this.terminate(), this.$a = q.location.protocol + "//" + this.b.domain + this.b.path, this.b.path = Ya(a), b && (this.b.title = b), this.A(), db(this)
};
function gb(a) {
    if (a.$a)
        return g;
    a = (document.referrer || "").indexOf("://" + q.location.host + "/");
    return a!=-1 && a < 9
}
function hb(a) {
    a = a.$a;
    if (!a && (a = document.referrer || "")) {
        var b = Ca(a);
        if (b.protocol === "http:" || b.protocol === "https:")
            b.pathname = Ya(b.pathname), a = Da(b)
    }
    return encodeURIComponent(a)
}
function ib(a) {
    a = a.b.title.slice(0, 100);
    return encodeURIComponent(a)
}
function eb(a) {
    var b = q.navigator, c = q.window.screen, e = [b.userAgent, b.platform, (new Date).getTimezoneOffset(), c.width + c.height + c.colorDepth];
    s(b.plugins, function(a) {
        e.push(a.name + a.description + a.filename + a[0].type)
    });
    b = q.performance;
    e = e.concat([b && b.now ? b.now(): "", hb(a), document.title, q.location.href, v(), Ua("Width"), Ua("Height"), ga()]);
    return e.join()
}
function jb(a) {
    var b = "", c = a.g.e("_chartbeat4");
    c && (s(c.split("::"), function(a) {
        b += "&z=" + encodeURIComponent(a)
    }), a.g.remove("_chartbeat4"));
    return b
}
p.terminate = function() {
    this.O.terminate();
    La(this.Lb);
    this.qa !== d && ja(q, "load", this.qa);
    clearInterval(this.ga)
};
p.ca = function() {
    this.terminate();
    ja(q, "unload", this.Ma);
    this.Ma = this.qa = this.g = this.X = this.Da = this.b = i;
    this.O.ca();
    this.O = i
};
function kb(a) {
    for (var b = a.offsetLeft, c = a.offsetTop, e = j, f = a.offsetParent; a && a !== d && a !== document.body;) {
        if (a === f)
            b += a.offsetLeft, c += a.offsetTop, f = a.offsetParent;
        var h = a, l = d;
        l = q.getComputedStyle ? (h = q.getComputedStyle(h, i)) && (h.position || h.getPropertyValue("position")) : h.currentStyle ? h.currentStyle.position : h.style && h.style.position;
        if ((l || "") === "fixed") {
            e = g;
            break
        }
        a = a.parentElement
    }
    e && (b += 0, c += 0);
    return {
        x: b,
        y: c
    }
};
function lb(a, b) {
    for (var c = b || document.documentElement, e = [], f = i, h = a, l, y, o, F, J, O; h && h !== c;) {
        l = h.nodeName.toLowerCase();
        f = h;
        y = f.nodeName;
        if ((h = h.parentNode) && h !== document.documentElement) {
            o = h.children;
            F = 0;
            for (J = 0, O = o.length; J < O; J++) {
                if (f === o[J]) {
                    l += "[" + (1 + J - F) + "]";
                    break
                }
                o[J].nodeName !== y && F++
            }
        }
        e.unshift(l)
    }
    return e.join("/")
};
function H() {
    ea.call(this);
    Ea("_cbq", t(this.yb, this));
    A("anchor", this.ic, this);
    "postMessage"in window && u(q, "message", t(this.Zb, this))
}
da(H);
p = H.prototype;
p.A = function() {
    H.aa.A.call(this);
    this.ya = i;
    Fa(this.b);
    var a = q.location;
    w(this.b, "pingServer", "ping.chartbeat.net");
    w(this.b, "title", document.title);
    w(this.b, "domain", a.host);
    this.b.path = this.b.path ? Ya(this.b.path) : Wa(!!this.b.useCanonical);
    this.S = a.host.replace(/^www\./, "");
    this.b.domain = this.b.domain.replace(/^www\./, "");
    a = (this.g.e("_chartbeat2", g) || "").split(".");
    a.length > 4 && (a = []);
    var b = v(), c, e = "1", f = a&&+a[2];
    c = a && a[3];
    if (a && f && c)
        if (e = Math.abs((na(b) - na(f)) / 864E5)) {
            e = Math.min(e, 16) - 1;
            for (f = ""; e--;)
                f +=
                0;
                e = (c + f + "1").slice( - 16)
        } else 
            e = c;
    c = e;
    var e = (this.g.e("_chartbeat2", g) || "").split("."), h = v(), f = h-+(e[1] || 0);
    h-=+(e[2] || 0);
    this.Ia = e[0] && f > 18E5 && h < 2592E6 ? 0 : 1;
    a[0] || (a[0] = this.b.utoken || ha(eb(this), 3), a[1] = b);
    a[2] = b;
    a[3] = c;
    this.Ya = a[0];
    this.P = a.join(".");
    this.g.create("_chartbeat2", this.P, 94608E3, g);
    this.b.utoken = this.Ya;
    var l;
    b =+ a[1];
    c =+ a[2];
    if ((a = a[3]) && b && c)
        l = (Math.min((Math.abs((na(c) - na(b)) / 864E5) || 0) + 1, 16, a.length) - 1).toString(16), l += ("0000" + parseInt(a, 2).toString(16)).slice( - 4);
    this.jb = l
};
p.Kb = function(a) {
    this.ya = a
};
p.ra = function() {
    this.g.update("_chartbeat4", ["t=" + this.$, "E=" + this.O.total(), "x=" + fb(this), "c=" + Math.round((v() - this.va) / 600) / 100, "y=" + Ua("Height"), "w=" + Ta()].join("&"), 3600)
};
p.vb = function() {
    var a = this.b, b = fb(this);
    this.ha = Math.max(this.ha, b);
    var c = Math.round((v() - this.va) / 600) / 100, e = 0, f = 0, h = 0, l = this.O.Y;
    B.Mb() ? f = 1 : this.b.reading&&+this.b.reading || l > 0 || c < 0.09 ? e = 1 : h = 1;
    var y = "", o = "", F = "", J = "", O = "", G = gb(this);
    if (this.Qa) {
        this.Qa = j;
        var y = (G ? "&v=" : "&r=") + hb(this), o = "&i=" + ib(this), R = this.b.hudTrackable;
        R !== d && (F = "&L=" + (R ? "1" : "0"));
        if (G && (G = mb(this)))
            y = "&v=" + encodeURIComponent(G.path), J = "&K=" + [G.left, G.top, G.Hc, encodeURIComponent(G.Gc)].join("::"), G.qb > 1 && (J += "&l1=" + G.qb);
        a.alias &&
        (O = "&PA=" + encodeURIComponent(a.alias));
        this.$a && (y += "&vp=1")
    }
    G = this.sa ? "&b=" + this.sa : "";
    R = this.ya ? "&A=" + this.ya : "";
    a = (q.location.protocol || "http:") + "//" + a.pingServer + "/ping?h=" + encodeURIComponent(a.domain) + "&p=" + encodeURIComponent(a.path) + "&u=" + this.Ya + "&d=" + encodeURIComponent(this.S) + "&g=" + a.uid + Ga(a, 0, "sections") + Ga(a, 1, "authors") + Ga(a, 3, "sponsorName") + Ga(a, 4, "type") + (!a.noCookies && this.g.isSupported() ? "&n=" + this.Ia : "&nc=1") + (this.jb ? "&f=" + this.jb : "") + "&c=" + c + "&x=" + b + "&m=" + this.ha + "&y=" + Ua("Height") +
    "&o=" + Ua("Width") + "&w=" + Ta() + "&j=" + Math.round((this.Z + 2) * 15E3 / 1E3) + "&R=" + e + "&W=" + f + "&I=" + h + "&E=" + this.O.total() + "&e=" + l + y + J + O + G + R + Aa("C", "utm_campaign", a.campaignTag) + Aa("M", "utm_medium", a.mediumTag) + "&t=" + this.$ + "&V=39" + jb(this) + o + F + "&tz=" + (new Date).getTimezoneOffset() + (Pa(this.g) ? "&l=1" : "") + Ha(a) + "&_";
    this.T || fa(a, this.Da)
};
p.Zb = function(a) {
    var b = this.b;
    if (a.origin === "http://" + (b.playerdomain || this.S)) {
        var c = a.data;
        la(c) && c.indexOf("cbqpush::") === 0 ? (a = c.split("::"), a.length == 3 && (a = a.slice(1), a[0].indexOf("_") === 0 && this.yb(a))) : c == "cbdata?" && (b = "domain=" + encodeURIComponent(b.domain) + "&path=" + encodeURIComponent(b.path) + "&title=" + ib(this) + "&referrer=" + hb(this) + "&internal=" + (gb(this) ? "1" : "0") + "&subdomain=" + encodeURIComponent(this.S) + "&utoken=" + this.Ya, a.source.postMessage(b, "*"))
    }
};
p.yb = function(a) {
    this.b[a[0]] = a[1];
    this.Z = 0
};
function nb(a) {
    a = a.replace(/-{2,}/g, "-");
    a = Ca(a);
    a.pathname = Ya(a.pathname);
    return a
}
p.ic = function(a) {
    var b = a.href || "", b = nb(b);
    if (!(b.hostname !== q.location.hostname || b.protocol.indexOf("http") !== 0)) {
        var b = Da(b), c = Va(a, function(a) {
            return a.id
        }), e = lb(a, c);
        c && (e && (e = "/" + e), e = "*[@id='" + c.id + "']" + e);
        c = ha(e);
        e = encodeURIComponent(e).replace(/-/g, "%2D");
        e.length > 512 && (e = "");
        a = kb(a);
        this.g.update("_chartbeat5", [a.x, a.y, encodeURIComponent(this.b.path), encodeURIComponent(b), c, e].join(","), 60, j, d, 5)
    }
};
function mb(a) {
    var b = a.g.e("_chartbeat5");
    if (!b)
        return i;
    var c = b.split("::"), b = c.length, e, f;
    if (b === 1)
        e = c[0].split(","), f = 0;
    else {
        var h, l = nb(q.location.href), y = Da(l);
        s(c, function(a, b) {
            var c = a.split(","), l = wa(y, decodeURIComponent(c[3]));
            if (l === 0)
                return e = c, f = b, j;
            if (h === d || l < h)
                h = l, e = c, f = b
        })
    }
    c.splice(f, 1);
    a.g.create("_chartbeat5", c.join("::"), 60);
    (a = e[5]) ? (a = a.replace(/%2D/g, "-"), a = decodeURIComponent(a)) : a = "";
    return {
        left: e[0],
        top: e[1],
        path: decodeURIComponent(e[2]),
        Hc: e[4] || "",
        Gc: a,
        qb: b
    }
};
var I = "ct", K = "s1";
function M(a) {
    this.a = a;
    this.d = this.f = j;
    this.c = v();
    this.J = this.M = d;
    this.D()
}
p = M.prototype;
p.D = function() {
    this.a.currentTime() > 0&&!this.a.paused&&!this.a.ended && this.C();
    this.a.on("play", t(this.C, this));
    this.a.ready(t(this.G, this))
};
p.G = function() {
    this.f = g
};
p.C = function() {
    this.d = g
};
p.j = m("f");
p.v = function() {
    var a = "";
    return a = (a = this.a.options()["data-media"]) && a.hasOwnProperty("title") ? a.title : this.a.el().attributes.title && this.a.el().attributes.title.value
};
p.i = function() {
    return this.a.src() || ""
};
p.o = function() {
    return I
};
p.k = n("");
p.w = function() {
    var a = this.a.duration();
    return ya(a)
};
p.h = function() {
    return !this.d ? K : this.a.ended() ? "s4" : this.a.paused() ? "s3" : "s2"
};
p.p = function() {
    var a = this.a.currentTime();
    return ya(a)
};
p.n = n( - 1);
p.u = function() {
    return this.a.poster() || ""
};
p.r = k();
p.m = k();
p.z = function() {
    return xa(this.c)
};
p.s = m("M");
p.l = m("J");
p.t = n("BCN");
M.prototype.getStrategyName = M.prototype.t;
M.prototype.isReady = M.prototype.j;
M.prototype.getTitle = M.prototype.v;
M.prototype.getVideoPath = M.prototype.i;
M.prototype.getContentType = M.prototype.o;
M.prototype.getAdPosition = M.prototype.k;
M.prototype.getTotalDuration = M.prototype.w;
M.prototype.getState = M.prototype.h;
M.prototype.getCurrentPlayTime = M.prototype.p;
M.prototype.getBitrate = M.prototype.n;
M.prototype.getThumbnailPath = M.prototype.u;
M.prototype.getPlayerType = M.prototype.r;
M.prototype.getAutoplayType = M.prototype.m;
M.prototype.getViewStartTime = M.prototype.z;
M.prototype.getSections = M.prototype.s;
M.prototype.getAuthors = M.prototype.l;
M.verify = function(a) {
    return a.controlBar !== d
};
function N(a) {
    this.Fb = this.S = d;
    this.f = j;
    this.xb = this.fb =- 1;
    this.N = i;
    this.a = a;
    this.d = j;
    this.Na = g;
    this.c = v();
    this.Oa = "";
    for (var a = window.experienceJSON.data.configuredProperties.plugins.list.split(" "), b = a.length, c = 0; c < b; c++) {
        var e = a[c];
        if ((e.indexOf("chartbeat") >= 0 || e.indexOf("localhost") >= 0) && e.indexOf(".js") >= 0) {
            this.Oa = e;
            break
        }
    }
    if (this.Oa)
        a = za(this.Oa), this.S = a.domain, this.Fb = a.uid;
    a = window.brightcove;
    this.a = a.api.getExperience();
    b = this.a.getModule(a.api.modules.APIModules.EXPERIENCE);
    b.getReady() ?
    this.tb() : b.addEventListener(a.player.events.ExperienceEvent.TEMPLATE_READY, r(this.tb, this))
}
p = N.prototype;
p.tb = function() {
    var a = window.brightcove, b = this.a.getModule(a.api.modules.APIModules.VIDEO_PLAYER), c = this;
    b.getVideoDuration(function(a) {
        c.fb = Math.round(a)
    });
    b.getCurrentVideo(function(a) {
        c.N = a
    });
    a = a.api.events.MediaEvent;
    b.addEventListener(a.PROGRESS, r(this.sc, this));
    b.addEventListener(a.STOP, r(this.tc, this));
    b.addEventListener(a.COMPLETE, r(this.jc, this));
    b.addEventListener(a.BEGIN, r(this.C, this));
    this.f = g
};
p.sc = function(a) {
    this.xb = a.position;
    this.Q = this.Na = j
};
p.tc = function() {
    this.Na = g
};
p.jc = function() {
    this.Q = g
};
p.C = function() {
    this.Q = j;
    if (!this.d)
        this.d = g
};
p.j = m("f");
p.v = function() {
    var a = "";
    this.N && (a = this.N.displayName);
    return a
};
p.i = function() {
    var a = "";
    this.N && (a = "video@" + this.S + "/" + this.N.id);
    return a
};
p.o = function() {
    return I
};
p.k = n("");
p.w = m("fb");
p.h = function() {
    return !this.d ? K : this.Q ? "s4" : this.Na ? "s3" : "s2"
};
p.p = m("xb");
p.n = n( - 1);
p.u = function() {
    var a = "";
    this.N && (a = this.N.thumbnailURL);
    return a
};
p.r = k();
p.z = function() {
    return isNaN(this.c) ? 0 : qa(this.c)
};
p.l = k();
p.s = function() {
    var a;
    this.N && (a = this.N.tags);
    return a
};
p.m = n("unkn");
p.t = n("BC");
p.Tb = m("S");
p.Ub = m("Fb");
N.prototype.isReady = N.prototype.j;
N.prototype.getTitle = N.prototype.v;
N.prototype.getVideoPath = N.prototype.i;
N.prototype.getContentType = N.prototype.o;
N.prototype.getAdPosition = N.prototype.k;
N.prototype.getTotalDuration = N.prototype.w;
N.prototype.getState = N.prototype.h;
N.prototype.getCurrentPlayTime = N.prototype.p;
N.prototype.getBitrate = N.prototype.n;
N.prototype.getThumbnailPath = N.prototype.u;
N.prototype.getPlayerType = N.prototype.r;
N.prototype.getViewStartTime = N.prototype.z;
N.prototype.getAuthors = N.prototype.l;
N.prototype.getSections = N.prototype.s;
N.prototype.getAutoplayType = N.prototype.m;
N.verify = function(a) {
    return a === window.brightcove
};
N.prototype.getStrategyName = N.prototype.t;
N.prototype.getHost = N.prototype.Tb;
N.prototype.getUid = N.prototype.Ub;
function Q() {
    this.a = window.flowplayer();
    this.d = this.f = j;
    this.c = v();
    this.J = this.M = d;
    this.Eb = this.Ua = "";
    this.Ha = d;
    this.D()
}
var ob = /url\(['"]?([^'"\)]*)['"]?\)/;
p = Q.prototype;
p.hb = function() {
    var a = this.i(), b = Qa("name", "param"), c = d;
    s(b, function(b) {
        if (b.attributes.name.value === "flashvars" && b.value.indexOf(a)!==-1)
            return c = b, j
    });
    if (c === d)
        return j;
    b = Ra(c, "data-swf");
    if (b === j)
        return j;
    var e = b.style.background;
    if (e) {
        if (e = e.match(ob))
            this.Ua = e[1]
    } else 
        this.Ua = b.attributes.poster && b.attributes.poster.value;
    this.Eb = b.title;
    this.Ha !== d && clearInterval(this.Ha);
    return g
};
p.D = function() {
    this.a.Nc ? this.G() : this.a.bind("ready", r(this.G, this));
    this.a.bind("resume", r(this.C, this));
    this.a.Mc && this.C()
};
p.G = function() {
    if (this.hb() === j)
        this.Ha = setInterval(r(this.hb, this), 1E3);
    this.f = g
};
p.C = function() {
    if (!this.d)
        this.d = g
};
p.j = m("f");
p.v = m("Eb");
p.i = function() {
    return this.a.video.url
};
p.o = function() {
    return I
};
p.k = n("");
p.w = function() {
    return this.a.video.duration
};
p.h = function() {
    return !this.d ? K : this.a.Kc ? "s4" : this.a.paused ? "s3" : "s2"
};
p.p = function() {
    return this.a.video.time
};
p.n = n( - 1);
p.u = m("Ua");
p.r = k();
p.m = k();
p.z = function() {
    return isNaN(this.c) ? 0 : qa(this.c)
};
p.s = m("M");
p.l = m("J");
p.t = n("FP");
Q.prototype.getStrategyName = Q.prototype.t;
Q.prototype.isReady = Q.prototype.j;
Q.prototype.getTitle = Q.prototype.v;
Q.prototype.getVideoPath = Q.prototype.i;
Q.prototype.getContentType = Q.prototype.o;
Q.prototype.getAdPosition = Q.prototype.k;
Q.prototype.getTotalDuration = Q.prototype.w;
Q.prototype.getState = Q.prototype.h;
Q.prototype.getCurrentPlayTime = Q.prototype.p;
Q.prototype.getBitrate = Q.prototype.n;
Q.prototype.getThumbnailPath = Q.prototype.u;
Q.prototype.getPlayerType = Q.prototype.r;
Q.prototype.getAutoplayType = Q.prototype.m;
Q.prototype.getViewStartTime = Q.prototype.z;
Q.prototype.getSections = Q.prototype.s;
Q.prototype.getAuthors = Q.prototype.l;
Q.verify = function() {
    return !!window.flowplayer
};
function S(a) {
    this.a = a;
    this.d = this.f = j;
    this.c = v();
    this.J = this.M = d;
    this.D()
}
p = S.prototype;
p.D = function() {
    this.a.readyState > 2 && this.G();
    this.a.currentTime > 0&&!this.a.paused&&!this.a.ended && this.C();
    this.a.addEventListener("canplay", t(this.G, this));
    this.a.addEventListener("playing", t(this.C, this))
};
p.G = function() {
    this.f = g
};
p.C = function() {
    this.d = g
};
p.j = m("f");
p.v = function() {
    return this.a.attributes.title && this.a.attributes.title.value || ""
};
p.i = function() {
    return this.a.currentSrc || ""
};
p.o = function() {
    return I
};
p.k = n("");
p.w = function() {
    return ya(this.a.duration)
};
p.h = function() {
    return !this.d ? K : this.a.ended ? "s4" : this.a.paused ? "s3" : "s2"
};
p.p = function() {
    return ya(this.a.currentTime)
};
p.n = n( - 1);
p.u = function() {
    return this.a.poster || ""
};
p.r = k();
p.m = k();
p.z = function() {
    return xa(this.c)
};
p.s = m("M");
p.l = m("J");
p.t = n("H5");
S.prototype.getStrategyName = S.prototype.t;
S.prototype.isReady = S.prototype.j;
S.prototype.getTitle = S.prototype.v;
S.prototype.getVideoPath = S.prototype.i;
S.prototype.getContentType = S.prototype.o;
S.prototype.getAdPosition = S.prototype.k;
S.prototype.getTotalDuration = S.prototype.w;
S.prototype.getState = S.prototype.h;
S.prototype.getCurrentPlayTime = S.prototype.p;
S.prototype.getBitrate = S.prototype.n;
S.prototype.getThumbnailPath = S.prototype.u;
S.prototype.getPlayerType = S.prototype.r;
S.prototype.getAutoplayType = S.prototype.m;
S.prototype.getViewStartTime = S.prototype.z;
S.prototype.getSections = S.prototype.s;
S.prototype.getAuthors = S.prototype.l;
S.verify = function(a) {
    var b = a instanceof HTMLElement && a.nodeName === "VIDEO", a = a.getAttribute && a.getAttribute("data-account") !== i;
    return b&&!a
};
function T(a) {
    this.a = a();
    this.d = this.Q = this.f = j;
    this.c = v();
    this.D()
}
p = T.prototype;
p.D = function() {
    this.a.onReady(r(this.G, this));
    this.a.onTime(r(this.vc, this));
    this.a.onComplete(r(this.wc, this))
};
p.G = function() {
    this.f = g
};
p.vc = function() {
    this.a.getDuration();
    this.Q = j;
    if (!this.d)
        this.d = g
};
p.wc = function() {
    this.Q = g
};
p.j = function() {
    return this.f || this.a.getState() !== d
};
p.v = function() {
    var a = pb(this);
    return !a ? "" : a.title
};
function pb(a) {
    if (a.a.getPlaylistIndex === d)
        return a.a.getPlaylistItem();
    var b = a.a.getPlaylist();
    if (!b)
        return i;
    var c = b.length, a = a.a.getPlaylistIndex();
    return c === 0 || c <= a ? i : b[a]
}
p.i = function() {
    var a = pb(this);
    return !a ? "" : a.file
};
p.o = function() {
    return I
};
p.k = n("");
p.w = function() {
    var a = this.a.getDuration();
    if (a > 0)
        return Math.round(a * 1E3)
};
p.h = function() {
    if (!this.d)
        return K;
    var a = this.a.getState();
    return this.Q ? "s4" : a === "PAUSED" ? "s3" : "s2"
};
p.p = function() {
    return this.a.getPosition()
};
p.n = n( - 1);
p.u = function() {
    var a = pb(this);
    return !a ? "" : a.image
};
p.r = k();
p.z = function() {
    return isNaN(this.c) ? 0 : qa(this.c)
};
p.l = k();
p.s = k();
p.m = n("unkn");
p.t = n("JW");
T.prototype.getStrategyName = T.prototype.t;
T.prototype.isReady = T.prototype.j;
T.prototype.getTitle = T.prototype.v;
T.prototype.getVideoPath = T.prototype.i;
T.prototype.getContentType = T.prototype.o;
T.prototype.getAdPosition = T.prototype.k;
T.prototype.getTotalDuration = T.prototype.w;
T.prototype.getState = T.prototype.h;
T.prototype.getCurrentPlayTime = T.prototype.p;
T.prototype.getBitrate = T.prototype.n;
T.prototype.getThumbnailPath = T.prototype.u;
T.prototype.getPlayerType = T.prototype.r;
T.prototype.getViewStartTime = T.prototype.z;
T.prototype.getAuthors = T.prototype.l;
T.prototype.getSections = T.prototype.s;
T.prototype.getAutoplayType = T.prototype.m;
T.verify = function(a) {
    return a === window.jwplayer
};
function U(a) {
    this.a = a;
    this.Gb = "";
    this.F = K;
    this.R = I;
    this.I = "";
    this.c = v();
    this.D()
}
p = U.prototype;
p.D = function() {
    this.a.kBind("playerStateChange.cb", r(this.yc, this));
    this.a.kBind("playerUpdatePlayhead.cb", r(this.rc, this));
    this.a.kBind("playerPlayEnd.cb", r(this.xc, this));
    this.a.kBind("adStart.cb", r(this.fc, this));
    this.a.kBind("adEnd.cb", r(this.ec, this));
    this.a.kBind("switchingChangeComplete.cb", r(this.uc, this))
};
p.yc = function(a) {
    switch (a) {
    case "playing":
        this.F = "s2";
        break;
    case "buffering":
        this.F = "s3";
        break;
    case "paused":
        this.F = "s3"
    }
};
p.rc = function() {
    this.F = "s2";
    this.a.kUnbind("playerUpdatePlayhead.cb")
};
p.xc = function() {
    this.F = "s4"
};
p.fc = function(a) {
    this.R = "ad";
    switch (a) {
    case "pre":
        this.I = "a1";
        break;
    case "post":
        this.I = "a3";
        break;
    case "mid":
        this.I = "a2";
        break;
    case "main":
        this.I = "a5"
    }
};
p.ec = function() {
    this.R = I
};
p.uc = function(a, b) {
    this.Ob = b
};
p.j = function() {
    return this.a.evaluate("{mediaProxy.entry.name}") != i
};
p.v = function() {
    return this.a.evaluate("{mediaProxy.entry.name}")
};
p.i = function() {
    if (!this.j())
        return this.Gb;
    var a = this.a.evaluate("{mediaProxy.entry.id}");
    return (this.Gb = a) || ""
};
p.o = m("R");
p.k = m("I");
p.w = function() {
    var a = this.a.evaluate("{duration}");
    return ya(a)
};
p.h = m("F");
p.p = function() {
    var a = this.a.evaluate("{video.player.currentTime}");
    return ya(a)
};
p.n = m("Ob");
p.u = function() {
    return this.a.evaluate("{mediaProxy.entry.thumbnailUrl}") || ""
};
p.r = k();
p.m = function() {
    var a = this.a.evaluate("{configProxy.flashvars.autoPlay}"), b = this.a.evaluate("{configProxy.flashvars.playlistAPI.autoContinue}"), c = this.a.evaluate("{playlistAPI.dataProvider.selectedIndex}");
    return b && c > 0 ? "cont" : a ? "auto" : "man"
};
p.z = function() {
    return isNaN(this.c) ? 0 : xa(this.c)
};
p.s = function() {
    var a = this.a.evaluate("{mediaProxy.entry.tags}");
    if (a)
        return a.split(",")
};
p.l = function() {
    var a = this.a.evaluate("{mediaProxy.entry.creatorId}");
    if (a)
        return [encodeURIComponent(a)]
};
p.t = n("KT");
U.prototype.getStrategyName = U.prototype.t;
U.prototype.isReady = U.prototype.j;
U.prototype.getTitle = U.prototype.v;
U.prototype.getVideoPath = U.prototype.i;
U.prototype.getContentType = U.prototype.o;
U.prototype.getAdPosition = U.prototype.k;
U.prototype.getTotalDuration = U.prototype.w;
U.prototype.getState = U.prototype.h;
U.prototype.getCurrentPlayTime = U.prototype.p;
U.prototype.getBitrate = U.prototype.n;
U.prototype.getThumbnailPath = U.prototype.u;
U.prototype.getPlayerType = U.prototype.r;
U.prototype.getAutoplayType = U.prototype.m;
U.prototype.getViewStartTime = U.prototype.z;
U.prototype.getSections = U.prototype.s;
U.prototype.getAuthors = U.prototype.l;
U.verify = function(a) {
    return typeof a.kBind == "function"
};
function W(a) {
    this.eb = d;
    this.a = a;
    this.pb = d;
    this.f = j;
    this.W = d;
    this.d = j;
    this.J = this.M = this.c = d;
    this.eb = q.OO;
    this.pb = this.a.mb;
    this.c = v();
    this.D()
}
p = W.prototype;
p.D = function() {
    var a = this.pb, b = this.eb.EVENTS;
    a.subscribe(b.CONTENT_TREE_FETCHED, "OoyalaStrategyPlugin", r(this.G, this));
    a.subscribe(b.WILL_PLAY_ADS, "OoyalaStrategyPlugin", r(this.hc, this));
    a.subscribe(b.PLAYING, "OoyalaStrategyPlugin", r(this.C, this));
    a.subscribe(b.ADS_PLAYED, "OoyalaStrategyPlugin", r(this.gc, this));
    a.subscribe(b.METADATA_FETCHED, "OoyalaStrategyPlugin", r(this.pc, this));
    a.subscribe(b.EMBED_CODE_CHANGED, "OoyalaStrategyPlugin", r(this.kc, this))
};
p.G = function() {
    this.f = g
};
p.hc = function() {
    var a = this.a.getPlayheadTime(), b = this.a.getDuration();
    isNaN(a) && (a = 0);
    this.W = a === 0 ? "a1" : qb(a, b) ? "a3" : "a2"
};
p.C = function() {
    this.d = g
};
p.gc = function() {
    this.W = d
};
p.pc = k();
p.j = function() {
    var a = this.a.getState();
    if (a === "ready" || a === "playing" || a === "paused")
        this.f = g;
    return this.f
};
p.v = function() {
    var a = this.a.getItem();
    return a ? a.title : ""
};
p.i = function() {
    return (this.a.getItem() || {}).embedCode || this.a.embedCode || ""
};
p.o = function() {
    return this.W === d ? I : "ad"
};
p.k = function() {
    return this.W ? this.W : ""
};
p.w = function() {
    var a = this.a.getDuration();
    return a===-1 || isNaN(a)?-1 : a * 1E3
};
p.h = function() {
    if (!this.d)
        return K;
    if (this.a.getState() === "paused")
        return "s3";
    var a = this.a.getPlayheadTime(), b = this.a.getDuration();
    return qb(a, b) ? "s4" : "s2"
};
function qb(a, b) {
    return a === b || Math.floor(a) === Math.floor(b) || Math.ceil(a) === Math.ceil(b) || Math.ceil(a) === Math.floor(b) || Math.floor(a) === Math.ceil(b)
}
p.kc = function() {
    this.W = d
};
p.p = function() {
    var a = this.a.getPlayheadTime();
    return a===-1 || isNaN(a)?-1 : a * 1E3
};
p.n = function() {
    return this.a.getTargetBitrate()
};
p.u = function() {
    var a = this.a.getItem();
    return !a ? "" : a.promo || a.promo_image || a.thumbnail_image
};
p.r = k();
p.m = k();
p.z = function() {
    return isNaN(this.c) ? 0 : xa(this.c || 0)
};
p.s = m("M");
p.l = m("J");
p.t = n("OO");
W.prototype.getStrategyName = W.prototype.t;
W.prototype.isReady = W.prototype.j;
W.prototype.getTitle = W.prototype.v;
W.prototype.getVideoPath = W.prototype.i;
W.prototype.getContentType = W.prototype.o;
W.prototype.getAdPosition = W.prototype.k;
W.prototype.getTotalDuration = W.prototype.w;
W.prototype.getState = W.prototype.h;
W.prototype.getCurrentPlayTime = W.prototype.p;
W.prototype.getBitrate = W.prototype.n;
W.prototype.getThumbnailPath = W.prototype.u;
W.prototype.getPlayerType = W.prototype.r;
W.prototype.getAutoplayType = W.prototype.m;
W.prototype.getViewStartTime = W.prototype.z;
W.prototype.getSections = W.prototype.s;
W.prototype.getAuthors = W.prototype.l;
W.verify = function(a) {
    return a.hasOwnProperty("mb") && a.hasOwnProperty("subscribe") && a.hasOwnProperty("modules") && a.hasOwnProperty("adPlaying")
};
function X(a) {
    this.a = a;
    this.L = i;
    this.F = K;
    this.R = I;
    this.I = "";
    this.c = v();
    this.a.controller.plugInLoaded && this.a.controller.plugInLoaded(this)
}
p = X.prototype;
p.nb = function(a) {
    this.D(a.controller)
};
p.D = function(a) {
    a.addEventListener("OnLoadRelease", t(this.sb, this));
    a.addEventListener("OnLoadReleaseUrl", t(this.sb, this));
    a.addEventListener("OnMediaPlay", t(this.mc, this));
    a.addEventListener("OnMediaStart", t(this.oc, this));
    a.addEventListener("OnMediaPause", t(this.La, this));
    a.addEventListener("OnMediaComplete", t(this.lc, this));
    a.addEventListener("OnMediaError", t(this.La, this));
    a.addEventListener("OnMediaBuffer", t(this.La, this));
    a.addEventListener("OnMediaPlaying", t(this.nc, this))
};
p.sb = function(a) {
    if (a.data && a.data.pid && a.data.title)
        this.L = a.data
};
p.mc = function() {
    this.F = "s2"
};
p.oc = function(a) {
    this.F = "s2";
    a = a.data.baseClip;
    a.isAd ? (this.R = "ad", this.I = a.isMid ? "a2" : "a1") : (this.R = I, this.I = "")
};
p.La = function() {
    if (this.F != K)
        this.F = "s3"
};
p.lc = function() {
    this.F = "s4"
};
p.nc = function(a) {
    this.Sb = a.currentTime
};
p.j = function() {
    return this.L != i
};
p.v = function() {
    return this.L.title
};
p.i = function() {
    return !this.L ? "" : this.L.pid
};
p.o = m("R");
p.k = m("I");
p.w = function() {
    return this.L.duration
};
p.h = m("F");
p.p = m("Sb");
p.n = function() {
    return this.L.bitrate
};
p.u = function() {
    return this.L.defaultThumbnailUrl
};
p.r = k();
p.m = k();
p.z = function() {
    return isNaN(this.c) ? 0 : this.c === d ? 0 : v() - this.c
};
p.s = function() {
    var a = [];
    s(this.L.categories, function(b) {
        a.push(encodeURIComponent(b.name))
    });
    return a
};
p.l = k();
p.t = n("TP");
X.prototype.nb.initialize = X.prototype.nb;
X.prototype.getStrategyName = X.prototype.t;
X.prototype.isReady = X.prototype.j;
X.prototype.getTitle = X.prototype.v;
X.prototype.getVideoPath = X.prototype.i;
X.prototype.getContentType = X.prototype.o;
X.prototype.getAdPosition = X.prototype.k;
X.prototype.getTotalDuration = X.prototype.w;
X.prototype.getState = X.prototype.h;
X.prototype.getCurrentPlayTime = X.prototype.p;
X.prototype.getBitrate = X.prototype.n;
X.prototype.getThumbnailPath = X.prototype.u;
X.prototype.getPlayerType = X.prototype.r;
X.prototype.getAutoplayType = X.prototype.m;
X.prototype.getViewStartTime = X.prototype.z;
X.prototype.getSections = X.prototype.s;
X.prototype.getAuthors = X.prototype.l;
X.verify = function(a) {
    return typeof a.controller !== "undefined"
};
function Y(a, b) {
    this.a = a;
    this.Db = b;
    this.da = window.location != window.parent.location;
    this.wb = j;
    this.Dc = v();
    this.Cb = "_v_";
    ea.call(this)
}
da(Y);
var rb = q.QUICK_TEST ? 1E4: 6E4, sb = "autoDetect", tb = {};
p = Y.prototype;
p.A = function() {
    this.Ia = (this.P = this.g.e("_chartbeat3")) ? 0 : 1;
    this.bb = this.Ab = g;
    this.Ca = j;
    this.ua = this.Pa = this.gb = this.V = d;
    this.ba = 15E3;
    this.ka = d;
    this.Ac = this.Va = this.Wa = this.la = this.ma = 0;
    var a = q.location;
    this.zb = j;
    if (!this.P || this.P.length === 0)
        this.P = ha(eb(this), 3);
    this.g.create("_chartbeat3", this.P, 2592E3);
    var b;
    (b = this.Db) && ma(b) ? (this.B = new b(this.a), b = g) : b = j;
    if (b) {
        w(this.b, "pingServer", "ping.chartbeat.net");
        w(this.b, "title", document.title);
        w(this.b, "domain", a.host);
        w(this.b, "path", Wa(!!this.b.useCanonical));
        this.B.getPlayerType && w(this.b, "video_type", this.B.getPlayerType());
        this.gb = setInterval(t(this.Pb, this), 1E3);
        if (this.da === g)
            this.oa = t(this.$b, this), u(window, "message", this.oa), this.ka = setInterval(this.Bc, 1E3);
        Y.aa.A.call(this)
    }
};
p.za = k();
p.K = function() {
    if (this.B) {
        if (!this.B.getVideoPath())
            this.bb = g;
        if (!this.bb)
            return v() - this.va >= this.Ga && clearInterval(this.V), Y.aa.K.call(this);
        if (this.B.isReady())
            this.Ca = this.bb = j, this.Pa = ub(this), vb(this), this.V = setInterval(t(this.Ba, this), 100), Y.aa.K.call(this);
        else if (!this.Ca) {
            this.Ca = g;
            if (this.Ac++>150)
                throw "Error: Video strategy never reported its ready state";
            clearInterval(this.ga);
            this.ga = setInterval(t(this.K, this), 100)
        }
    }
};
p.vb = function() {
    if (this.B) {
        var a = this.b, b = this.B, c = b.getState(), e = b.getVideoPath(), f = this.da ? []: q._cbq || (q._cbq = []), h = b.getViewStartTime(), l = b.getThumbnailPath(), y = b.getAutoplayType ? b.getAutoplayType(): d, h = h===-1?-1 : h / 1E3, l = l ? encodeURIComponent(l) : d;
        if (!tb.path || tb.path === e || c === "s2" && tb.state !== "s2")
            switch (tb = {
                path: e,
                state: c
            }, f.push(["_vi", oa(b.getTitle())]), f.push(["_vp", e]), f.push(["_vdd", encodeURIComponent("video@" + (a.videodomain || a.domain))]), f.push(["_vs", b.getState()]), b.getContentType && f.push(["_vt",
            b.getContentType()]), b.getAdPosition && f.push(["_vap", b.getAdPosition()]), f.push(["_vtn", oa(b.getThumbnailPath())]), f.push(["_vd", b.getTotalDuration()]), b.getState()) {
            case K:
            case "s2":
            case "s3":
                f.push(["R", "1"]);
                break;
            case "s4":
                f.push(["R", "0"])
            }
        a = wb(this) + Z("_vt", b.getContentType ? b.getContentType() : "") + Z("_vap", b.getAdPosition ? b.getAdPosition() : "") + Z("_vs", c) + Z("_vbr", b.getBitrate ? b.getBitrate() : "") + Z("_vvs", h) + Z("_vpt", b.getCurrentPlayTime()) + Z("_vtn", l) + Z("_vaup", y || "unkn") + Z("_vplt", a.video_type) +
        Z("_vce", this.Va) + Z("c", Math.round(qa(this.Dc) / 600) / 100) + "&W=0&R=" + (c === "s2" ? "1" : "0") + "&I=" + (c !== "s2" ? "1" : "0") + "&E=" + this.Wa + "&j=" + (this.ba / 1E3 + 15) + xb(this) + "&tz=" + (new Date).getTimezoneOffset() + (Pa(this.g) ? "&l=1" : "") + "&_";
        this.da === g && s(f, function(a) {
            window.top.postMessage("cbqpush::" + a.join("::"), "*")
        });
        this.T || fa(a, this.Da)
    }
};
function Z(a, b) {
    return b !== d && b !== "" ? "&" + a + "=" + b : ""
}
p.ra = function() {
    this.g.update("_cb_cp", this.$, 3600, j, ",")
};
function xb(a) {
    var b = a.g.e("_cb_cp");
    a.g.remove("_cb_cp");
    return b ? "&D=" + b : ""
}
function yb(a) {
    return a.b.subdomain ? a.b.subdomain : q.location.host.replace(/^www\./, "")
}
function wb(a) {
    var b = a.b, c = a.B, e = q.location, f = c.getUid && c.getUid() || b.uid, h = b.videodomain || c.getHost && c.getHost() || b.domain, h = encodeURIComponent("video@" + h), l = "", y = a.sa ? "&b=" + a.sa: "", o = a.b.useSubDomains === j ? a.b.path: yb(a) + a.b.path, F = c.getSections && c.getSections(), J = c.getAuthors && c.getAuthors(), O = Ga(b, 0, "sections"), G = Ga(b, 1, "authors"), F = F === d || b.videoPageGroups ? O: "&g0=" + F.join(","), J = J === d || b.videoPageGroups ? G: "&g1=" + J.join(","), G = "", O = a.B.getVideoPath();
    if (a.ua === d)
        a.ua = O;
    else if (O != a.ua)
        G = Z("D",
        a.$), a.$ = ha(eb(a)), a.ua = O, a.ma = 0, a.la = 0, a.Wa = 0, a.Va = 0;
    if (a.Ab || G || a.da === g && a.wb === j)
        a.Ab = j, l = (gb(a) ? "&v=" : "&r=") + hb(a);
    a.Wa += a.ma;
    a.ma = 0;
    a.Va += a.la;
    a.la = 0;
    vb(a);
    var R = O = "", V = "", P = "";
    if (!a.da === g || a.zb === g)
        O = "&_vi=" + ib(a), R = "&_vp=" + encodeURIComponent(o), V = "&_vh=" + oa(yb(a)), P = "&_vdd=" + encodeURIComponent(b.domain);
    var o=!b.noCookies && a.g.isSupported(), L = c.getStrategyName ? c.getStrategyName() : "";
    return (e.protocol || "http:") + "//" + b.pingServer + "/ping?h=" + h + "&g=" + f + "&p=" + oa(c.getVideoPath()) + "&i=" + oa(c.getTitle()) +
    F + J + "&u=" + a.P + "&t=" + a.$ + "&x=0&y=0&V=39" + Z("VS", L) + (o ? "&n=" + a.Ia : "&nc=1") + y + l + G + Z("_vd", c.getTotalDuration()) + O + R + V + P
}
p.Ba = function() {
    var a = ub(this);
    if (this.Pa != a)
        this.Pa = a, vb(this), this.K()
    };
function vb(a) {
    var b = a.ba;
    a.ba = zb(a) === g ? 5E3 : a.B.getState() === K ? rb : 15E3;
    if (b != a.ba)
        b = a.ba, clearInterval(a.ga), a.ga = setInterval(t(a.K, a), b)
}
function ub(a) {
    return a.B.getState() + a.B.getAdPosition()
}
function zb(a) {
    return a.B.getState() === "s2" || a.B.getAdPosition() !== "" && a.B.getViewAdPlayTime!==-1
}
p.Pb = function() {
    zb(this) === g && (this.ma += 1, this.B.getContentType() === I && (this.la += 1))
};
p.Bc = function() {
    window.parent.postMessage("cbdata?", "*")
};
p.$b = function(a) {
    if (a.source === window.parent) {
        var b = a.data;
        if (a.data.indexOf("domain=") === 0) {
            this.zb = g;
            var c = {
                domain: g,
                path: g,
                title: g,
                referrer: g,
                internal: g,
                subdomain: g,
                token: g
            }, e = this;
            s(b.split("&"), function(a) {
                var a = a.split("="), b = a[0], a = a[1];
                c[b] && (e.b[b] = decodeURIComponent(a))
            });
            ja(window, "message", this.oa);
            clearInterval(this.ka);
            this.ka = d;
            this.wb = g;
            this.K()
        }
    }
};
p.terminate = function() {
    clearInterval(this.gb);
    clearInterval(this.ka);
    clearInterval(this.V);
    ja(window, "message", this.oa);
    Y.aa.terminate.call(this)
};
p.ca = function() {
    Y.aa.ca.call(this);
    this.Db = this.a = this.oa = i
};
function $(a) {
    this.a = a;
    this.f = g;
    this.d = j;
    this.c = v();
    this.ja = this.V = this.J = this.M = d;
    Ab(this, Bb(this));
    this.D()
}
p = $.prototype;
p.D = function() {
    this.V = setInterval(r(this.Ba, this), 100);
    var a = this.a.getPlayerState();
    (a === 1 || a === 2 || a === 0) && this.C()
};
p.Ba = function() {
    if (this.a.getPlayerState() === 1)
        clearInterval(this.V), this.V = d, this.C()
};
p.G = function() {
    this.f = g
};
p.C = function() {
    if (!this.d)
        this.d = g
};
function Bb(a) {
    var b;
    return (b = (a = a.a.getVideoUrl()) && a.match(/[?&]v=([^&]+)/), a = b) && a[1]
}
function Ab(a, b) {
    ra("//gdata.youtube.com/feeds/api/videos/" + b + "?v=2&alt=json", t(function(a) {
        if (b == Bb(this))
            this.ja = a ? {
                id: b,
                title: a && a.entry && a.entry.title && a.entry.title.$t || ""
            } : {
                id: b,
                title: ""
            }
    }, a))
}
p.j = function() {
    return this.f&&!!this.ja
};
p.v = function() {
    return this.ja && this.ja.id === Bb(this) ? this.ja.title : ""
};
p.i = function() {
    return Bb(this) || this.a.getVideoUrl()
};
p.o = function() {
    return I
};
p.k = n("");
p.w = function() {
    var a = this.a.getDuration();
    return ka(a) ? a * 1E3 : a
};
p.h = function() {
    if (!this.d)
        return K;
    var a = this.a.getPlayerState();
    return a === 0 ? "s4" : a === 2 ? "s3" : "s2"
};
p.p = function() {
    return this.a.getCurrentTime()
};
p.n = n( - 1);
p.u = function() {
    return "http://img.youtube.com/vi/" + Bb(this) + "/default.jpg"
};
p.r = k();
p.m = k();
p.z = function() {
    return isNaN(this.c) ? 0 : qa(this.c)
};
p.s = m("M");
p.l = m("J");
p.t = n("YT");
$.prototype.getStrategyName = $.prototype.t;
$.prototype.isReady = $.prototype.j;
$.prototype.getTitle = $.prototype.v;
$.prototype.getVideoPath = $.prototype.i;
$.prototype.getContentType = $.prototype.o;
$.prototype.getAdPosition = $.prototype.k;
$.prototype.getTotalDuration = $.prototype.w;
$.prototype.getState = $.prototype.h;
$.prototype.getCurrentPlayTime = $.prototype.p;
$.prototype.getBitrate = $.prototype.n;
$.prototype.getThumbnailPath = $.prototype.u;
$.prototype.getPlayerType = $.prototype.r;
$.prototype.getAutoplayType = $.prototype.m;
$.prototype.getViewStartTime = $.prototype.z;
$.prototype.getSections = $.prototype.s;
$.prototype.getAuthors = $.prototype.l;
$.verify = function(a) {
    return ma(a.getDuration) && ma(a.getVideoUrl) && ma(a.getVideoEmbedCode)
};
function Cb() {
    q._cbv_strategies || (q._cbv_strategies = []);
    q._cbv_strategies.push(W);
    q._cbv_strategies.push(S);
    q._cbv_strategies.push(Q);
    q._cbv_strategies.push($);
    q._cbv_strategies.push(T);
    q._cbv_strategies.push(N);
    q._cbv_strategies.push(M);
    q._cbv_strategies.push(U);
    q._cbv_strategies.push(X);
    this.Za = [];
    this.b = q._sf_async_config || {};
    Ea("_cbv", t(this.ac, this));
    this.b[sb] !== j && Db()
}
Cb.prototype.ac = function(a) {
    if (aa(a) == "array" && typeof a[0] == "string") {
        var b = a[0];
        if (b === "autoDetectYouTubeIframes") {
            Eb();
            return 
        } else if (b === "removePlayer") {
            Fb(this, a[1]);
            return 
        }
    }
    Gb(this, a)
};
function Gb(a, b) {
    var c = j;
    s(q._cbv_strategies, function(a) {
        if (a.verify(b) === g)
            return c = a, j
    });
    c === j || Hb(a, b, c)
}
function Fb(a, b) {
    var c = a.Za, e = q.pSUPERFLY_video, f = pa(c, function(a) {
        return b === a
    });
    f!==-1 && c.splice(f, 1);
    f = pa(e, function(a) {
        return a.a === b
    });
    f!==-1 && e.splice(f, 1)[0].ca()
}
function Hb(a, b, c) {
    var e = j;
    s(a.Za, function(a) {
        if (b === a)
            return e = g, j
    });
    e || (c = new Y(b, c), q.pSUPERFLY_video.push(c), db(c), a.Za.push(b))
}
function Db() {
    window.jwplayer && window.jwplayer().onReady ? window._cbv.push(window.jwplayer) : window.flowplayer ? window._cbv.push(window.flowplayer) : window.brightcove ? window._cbv.push(window.brightcove) : window.kWidget ? window.kWidget.addReadyCallback(function(a) {
        window._cbv.push(document.getElementById(a))
    }) : window.$pdk ? window._cbv.push(window.$pdk) : s(document.getElementsByTagName("video"), function(a) {
        a.getAttribute("data-account") == i ? window._cbv.push(a) : window.videojs && (a = window.videojs(a.getAttribute("id"))) &&
        window._cbv.push(a)
    })
}
function Eb() {
    function a() {
        function a(b) {
            q._cbv.push(b.target)
        }
        var b = q.YT.Player;
        s(document.getElementsByTagName("iframe"), function(f) {
            var h = f.src, l = h && h.indexOf("www.youtube.com/embed/");
            h && l >= 0 && l < 9 && new b(f, {
                events: {
                    onReady: a
                }
            })
        })
    }
    if (q.YT && q.YT.Player)
        a();
    else {
        var b = q.onYouTubeIframeAPIReady;
        q.onYouTubeIframeAPIReady = function() {
            if (b) {
                try {
                    b.apply(q, arguments)
                } catch (c) {}
                a()
            }
        }
    }
};
if (window.location == window.parent.location&&!q.pSUPERFLY) {
    var Ib = new H;
    q.pSUPERFLY = Ib;
    H.prototype.virtualPage = H.prototype.Fc;
    H.prototype.activity = H.prototype.Kb;
    db(Ib);
    var Jb = x.q(g);
    if (Jb) {
        u(q, "message", Ja);
        var Kb;
        if (Kb = Jb.getItem("_cb_ip")) {
            var Lb = q.location;
            Kb=!(!/^([^.]+[.])?chartbeat\.com$/.test(Lb.hostname) ? 0 : /^\/publishing\/(overlay|hud|mab)\//.test(Lb.pathname))
        }
        Kb && Ia()
    }
}
q.pSUPERFLY_video || (q.pSUPERFLY_video = []);
new Cb;
})();

