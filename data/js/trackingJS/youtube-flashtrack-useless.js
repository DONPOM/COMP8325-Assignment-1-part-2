(function() {
    var l, n = this, q = function(a) {
        return void 0 !== a
    }, t = function(a, b) {
        for (var c = a.split("."), d = b || n, e; e = c.shift();)
            if (null != d[e])
                d = d[e];
            else 
                return null;
        return d
    }, aa = function() {}, ba = function(a) {
        var b = typeof a;
        if ("object" == b)
            if (a) {
                if (a instanceof Array)
                    return "array";
                    if (a instanceof Object)
                        return b;
                        var c = Object.prototype.toString.call(a);
                        if ("[object Window]" == c)
                            return "object";
                            if ("[object Array]" == c || "number" == typeof a.length && "undefined" != typeof a.splice && "undefined" != typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("splice"))
                                return "array";
                                if ("[object Function]" == c || "undefined" != typeof a.call && "undefined" != typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("call"))
                                    return "function"
            } else 
                return "null";
                else if ("function" == b && "undefined" == typeof a.call)
    return "object";
return b
}, u = function(a) {
    return "array" == ba(a)
}, v = function(a) {
    return "string" == typeof a
}, ca = function(a) {
    return "number" == typeof a
}, w = function(a) {
    return "function" == ba(a)
}, da = function(a) {
    var b = typeof a;
    return "object" == b && null != a || "function" == b
}, ea = function(a, b, c) {
    return a.call.apply(a.bind,
    arguments)
}, fa = function(a, b, c) {
    if (!a)
        throw Error();
    if (2 < arguments.length) {
        var d = Array.prototype.slice.call(arguments, 2);
        return function() {
            var c = Array.prototype.slice.call(arguments);
            Array.prototype.unshift.apply(c, d);
            return a.apply(b, c)
        }
    }
    return function() {
        return a.apply(b, arguments)
    }
}, x = function(a, b, c) {
    x = Function.prototype.bind&&-1 != Function.prototype.bind.toString().indexOf("native code") ? ea : fa;
    return x.apply(null, arguments)
}, ga = function(a, b) {
    var c = Array.prototype.slice.call(arguments, 1);
    return function() {
        var b =
        c.slice();
        b.push.apply(b, arguments);
        return a.apply(this, b)
    }
}, y = function() {
    return + new Date
}, ha = function(a, b) {
    var c = a.split("."), d = n;
    c[0]in d ||!d.execScript || d.execScript("var " + c[0]);
    for (var e; c.length && (e = c.shift());)
        !c.length && q(b) ? d[e] = b : d[e] ? d = d[e] : d = d[e] = {}
}, z = function(a, b) {
    function c() {}
    c.prototype = b.prototype;
    a.M = b.prototype;
    a.prototype = new c;
    a.prototype.constructor = a;
    a.Hb = function(a, c, f) {
        return b.prototype[c].apply(a, Array.prototype.slice.call(arguments, 2))
    }
};
var A = function() {
    this.G = this.G;
    this.N = this.N
};
A.prototype.G=!1;
A.prototype.O = function() {
    this.G || (this.G=!0, this.k())
};
A.prototype.k = function() {
    if (this.N)
        for (; this.N.length;)
            this.N.shift()()
};
var ia = function(a) {
    for (var b = 0, c = arguments.length; b < c; ++b) {
        var d = arguments[b], e = d, f = ba(e);
        "array" == f || "object" == f && "number" == typeof e.length ? ia.apply(null, d) : d && "function" == typeof d.O && d.O()
    }
};
var B = function(a, b) {
    this.type = a;
    this.a = this.target = b;
    this.kb=!0
};
B.prototype.O = function() {};
B.prototype.preventDefault = function() {
    this.kb=!1
};
var ja = function(a, b, c, d, e) {
    this.name = a;
    this.k = b;
    this.type = c;
    this.j = d;
    this.m = e;
    this.chargeable=!1
}, ka = function(a) {
    switch (a) {
    case "Exit":
        return "Exit";
    case "Count":
        return "Counter";
    case "Start":
    case "Stop":
        return "Timer";
    default:
        throw "Unsupported event action";
    }
};
var C = function(a, b, c) {
    B.call(this, a);
    this.c = b || void 0;
    this.data = c
};
z(C, B);
var la = function(a, b, c, d) {
    B.call(this, a);
    this.c = b;
    this.g = c;
    this.n=!!d
};
z(la, B);
var ma = function(a, b, c, d) {
    "Count" == b && (d=!0);
    la.call(this, a, b, !0, d);
    this.e = c
};
z(ma, la);
var na = function(a, b, c, d) {
    ma.call(this, a, b, c);
    this.videoName = d
};
z(na, ma);
var oa = function(a, b, c, d, e) {
    la.call(this, a, b, !1, e);
    this.k = c;
    this.m = d
};
z(oa, la);
var pa = function(a, b) {
    B.call(this, "reportCustomVariable");
    this.c = a;
    this.e = b
};
z(pa, B);
var qa = function(a, b, c, d) {
    oa.call(this, a, "Exit", b, !0);
    this.j = "null" != c ? c : null;
    this.e = "null" != d ? d : null
};
z(qa, oa);
var ra = function(a) {
    B.call(this, "invokeExternalJSFunction");
    this.c = a
};
z(ra, B);
var sa = function(a) {
    B.call(this, "setThrottlingWindow");
    this.c = a
};
z(sa, B);
var ta = function() {
    B.call(this, "setDynamicServeId")
};
z(ta, B);
var va = function(a, b, c) {
    B.call(this, "setTimerAdjustment");
    this.k = a;
    this.c = b;
    this.e = c
};
z(va, B);
var wa = function(a) {
    B.call(this, "registerChargeableEventName");
    this.c = a
};
z(wa, B);
var xa = function(a) {
    if (Error.captureStackTrace)
        Error.captureStackTrace(this, xa);
    else {
        var b = Error().stack;
        b && (this.stack = b)
    }
    a && (this.message = String(a))
};
z(xa, Error);
xa.prototype.name = "CustomError";
var ya;
var D = function(a) {
    return /^[\s\xa0]*$/.test(a)
}, za = function(a) {
    return a.replace(/^[\s\xa0]+|[\s\xa0]+$/g, "")
}, Ha = function(a) {
    if (!Aa.test(a))
        return a;
    - 1 != a.indexOf("&") && (a = a.replace(Ba, "&amp;"));
    - 1 != a.indexOf("<") && (a = a.replace(Ca, "&lt;"));
    - 1 != a.indexOf(">") && (a = a.replace(Da, "&gt;"));
    - 1 != a.indexOf('"') && (a = a.replace(Ea, "&quot;"));
    - 1 != a.indexOf("'") && (a = a.replace(Fa, "&#39;"));
    - 1 != a.indexOf("\x00") && (a = a.replace(Ga, "&#0;"));
    return a
}, Ba = /&/g, Ca = /</g, Da = />/g, Ea = /"/g, Fa = /'/g, Ga = /\x00/g, Aa = /[\x00&<>"']/,
E = function(a, b) {
    return - 1 != a.indexOf(b)
}, F = function(a) {
    return null == a ? "" : String(a)
}, G = function(a, b) {
    for (var c = 0, d = za(String(a)).split("."), e = za(String(b)).split("."), f = Math.max(d.length, e.length), g = 0; 0 == c && g < f; g++) {
        var h = d[g] || "", k = e[g] || "", m = RegExp("(\\d*)(\\D*)", "g"), r = RegExp("(\\d*)(\\D*)", "g");
        do {
            var p = m.exec(h) || ["", "", ""], s = r.exec(k) || ["", "", ""];
            if (0 == p[0].length && 0 == s[0].length)
                break;
            c = Ia(0 == p[1].length ? 0 : parseInt(p[1], 10), 0 == s[1].length ? 0 : parseInt(s[1], 10)) || Ia(0 == p[2].length, 0 == s[2].length) ||
            Ia(p[2], s[2])
        }
        while (0 == c)
        }
    return c
}, Ia = function(a, b) {
    return a < b?-1 : a > b ? 1 : 0
}, Ja = function(a) {
    return String(a).replace(/\-([a-z])/g, function(a, c) {
        return c.toUpperCase()
    })
}, Ka = function(a) {
    var b = v(void 0) ? "undefined".replace(/([-()\[\]{}+?*.$\^|,:#<!\\])/g, "\\$1").replace(/\x08/g, "\\x08"): "\\s";
    return a.replace(new RegExp("(^" + (b ? "|[" + b + "]+" : "") + ")([a-z])", "g"), function(a, b, e) {
        return b + e.toUpperCase()
    })
};
var La = Array.prototype, Ma = function(a, b) {
    if (v(a))
        return v(b) && 1 == b.length ? a.indexOf(b, 0) : - 1;
    for (var c = 0; c < a.length; c++)
        if (c in a && a[c] === b)
            return c;
    return - 1
}, H = function(a, b, c) {
    for (var d = a.length, e = v(a) ? a.split("") : a, f = 0; f < d; f++)
        f in e && b.call(c, e[f], f, a)
}, Na = function(a, b) {
    for (var c = a.length, d = [], e = 0, f = v(a) ? a.split("") : a, g = 0; g < c; g++)
        if (g in f) {
            var h = f[g];
            b.call(void 0, h, g, a) && (d[e++] = h)
        }
    return d
}, Oa = function(a, b) {
    for (var c = a.length, d = Array(c), e = v(a) ? a.split("") : a, f = 0; f < c; f++)
        f in e && (d[f] = b.call(void 0,
        e[f], f, a));
    return d
}, Qa = function(a, b) {
    for (var c = a.length, d = v(a) ? a.split("") : a, e = 0; e < c; e++)
        if (e in d && b.call(void 0, d[e], e, a))
            return !0;
    return !1
}, Ra = function(a, b) {
    for (var c = a.length, d = v(a) ? a.split("") : a, e = 0; e < c; e++)
        if (e in d&&!b.call(void 0, d[e], e, a))
            return !1;
    return !0
}, Sa = function(a, b) {
    var c;
    t: {
        c = a.length;
        for (var d = v(a) ? a.split("") : a, e = 0; e < c; e++)
            if (e in d && b.call(void 0, d[e], e, a)) {
                c = e;
                break t
            }
        c =- 1
    }
    return 0 > c ? null : v(a) ? a.charAt(c) : a[c]
}, I = function(a, b) {
    var c = Ma(a, b), d;
    (d = 0 <= c) && La.splice.call(a, c, 1);
    return d
}, Ta = function(a) {
    return La.concat.apply(La, arguments)
}, Ua = function(a) {
    var b = a.length;
    if (0 < b) {
        for (var c = Array(b), d = 0; d < b; d++)
            c[d] = a[d];
        return c
    }
    return []
}, Va = function(a, b) {
    var c = {};
    H(a, function(d, e) {
        c[b.call(void 0, d, e, a)] = d
    });
    return c
};
var Wa = function(a) {
    return a
}, Xa = function(a) {
    var b = 1, b = b || 0;
    return function() {
        return a.apply(this, Array.prototype.slice.call(arguments, 0, b))
    }
};
var Ya = function(a) {
    a = String(a);
    if (/^\s*$/.test(a) ? 0 : /^[\],:{}\s\u2028\u2029]*$/.test(a.replace(/\\["\\\/bfnrtu]/g, "@").replace(/"[^"\\\n\r\u2028\u2029\x00-\x08\x0a-\x1f]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, "]").replace(/(?:^|:|,)(?:[\s\u2028\u2029]*\[)+/g, "")))
        try {
            return eval("(" + a + ")")
    } catch (b) {}
    throw Error("Invalid JSON string: " + a);
}, ab = function(a) {
    var b = [];
    Za(new $a, a, b);
    return b.join("")
}, $a = function() {}, Za = function(a, b, c) {
    switch (typeof b) {
    case "string":
        bb(b, c);
        break;
    case "number":
        c.push(isFinite(b) &&
        !isNaN(b) ? b : "null");
        break;
    case "boolean":
        c.push(b);
        break;
    case "undefined":
        c.push("null");
        break;
    case "object":
        if (null == b) {
            c.push("null");
            break
        }
        if (u(b)) {
            var d = b.length;
            c.push("[");
            for (var e = "", f = 0; f < d; f++)
                c.push(e), Za(a, b[f], c), e = ",";
            c.push("]");
            break
        }
        c.push("{");
        d = "";
        for (e in b)
            Object.prototype.hasOwnProperty.call(b, e) && (f = b[e], "function" != typeof f && (c.push(d), bb(e, c), c.push(":"), Za(a, f, c), d = ","));
        c.push("}");
        break;
    case "function":
        break;
    default:
        throw Error("Unknown type: " + typeof b);
    }
}, cb = {
    '"': '\\"',
    "\\": "\\\\",
    "/": "\\/",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\x0B": "\\u000b"
}, db = /\uffff/.test("\uffff") ? /[\\\"\x00-\x1f\x7f-\uffff]/g: /[\\\"\x00-\x1f\x7f-\xff]/g, bb = function(a, b) {
    b.push('"', a.replace(db, function(a) {
        if (a in cb)
            return cb[a];
        var b = a.charCodeAt(0), e = "\\u";
        16 > b ? e += "000" : 256 > b ? e += "00" : 4096 > b && (e += "0");
        return cb[a] = e + b.toString(16)
    }), '"')
};
var eb = function(a, b) {
    for (var c in a)
        b.call(void 0, a[c], c, a)
}, fb = function(a, b) {
    var c = {}, d;
    for (d in a)
        b.call(void 0, a[d], d, a) && (c[d] = a[d]);
    return c
}, gb = function(a, b) {
    var c = {}, d;
    for (d in a)
        c[d] = b.call(void 0, a[d], d, a);
    return c
}, hb = function(a) {
    var b = [], c = 0, d;
    for (d in a)
        b[c++] = a[d];
    return b
}, ib = function(a, b) {
    for (var c in a)
        if (a[c] == b)
            return !0;
    return !1
}, jb = function(a) {
    for (var b in a)
        return !1;
    return !0
}, kb = "constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" "),
lb = function(a, b) {
    for (var c, d, e = 1; e < arguments.length; e++) {
        d = arguments[e];
        for (c in d)
            a[c] = d[c];
        for (var f = 0; f < kb.length; f++)
            c = kb[f], Object.prototype.hasOwnProperty.call(d, c) && (a[c] = d[c])
    }
}, mb = function(a) {
    var b = arguments.length;
    if (1 == b && u(arguments[0]))
        return mb.apply(null, arguments[0]);
    if (b%2)
        throw Error("Uneven number of arguments");
    for (var c = {}, d = 0; d < b; d += 2)
        c[arguments[d]] = arguments[d + 1];
    return c
};
var ob = function(a) {
    if (D(a))
        return "";
    a = Ya(a);
    var b;
    a.Profile && a.Profile[0] ? b = Oa(a.Profile, nb)[0] : b = {};
    eb(fb(a, function(a, b) {
        return "_profileid" != b && "Profile" != b
    }), function(a, d) {
        b[d + "Array"] = Oa(a, nb)
    });
    return ab(b)
}, nb = function(a) {
    return gb(fb(a, function(a, c) {
        return "_index" != c
    }), function(a) {
        if (da(a)) {
            var c;
            i:
            {
                for (c in a)
                    if (!u.call(void 0, a[c])) {
                        c=!1;
                        break i
                    }
                c=!0
            }
            if (c) {
                c = 0;
                for (var d in a)
                    c++;
                c = 1 == c
            }
            if (c) {
                i:
                {
                    for (var e in a) {
                        d = e;
                        break i
                    }
                    d = void 0
                }
                a = Oa(a[d], Xa(ga(mb, d)))
            }
        }
        return a
    })
}, qb = function(a) {
    if (D(a))
        return "";
    var b = Ya(a);
    a = "<layout>";
    var c = {}, d;
    for (d in b)
        if ("_profileid" != d)
            for (var e = b[d].length, f = 0; f < e; f++) {
                var g = "Profile" == d ? "Profile": "Profile." + d + "Array[" + f + "]";
                a += '<registeredObject id="' + Ha(g) + '">';
                for (var h in b[d][f])
                    if ("_index" != h) {
                        var k = b[d][f][h], m = pb(h, k);
                        if (m)
                            a += m;
                        else {
                            var m = g + "." + h, r = c, p = void 0;
                            for (p in k)
                                if (da(k[p]))
                                    for (var s in k[p]) {
                                        var Pa = m + "Array[" + s + "]";
                                        r[Pa] || (r[Pa] = {});
                                        r[Pa][p] = k[p][s]
                                    } else 
                                        r[m] || (r[m] = {}), r[m][p] = k[p]
                        }
                    }
                    a += "</registeredObject>"
            }
    for (var ua in c) {
        a += '<registeredObject id="' +
        Ha(ua) + '">';
        for (var Ed in c[ua])(d = pb(Ed, c[ua][Ed])
            ) && (a += d);
        a += "</registeredObject>"
    }
    return a + "</layout>"
}, pb = function(a, b) {
    if (u(b)) {
        var c;
        c = null;
        0 < b.length && (c = rb(b[0]));
        if (c) {
            c = '<propertyArray id="' + a + 'Array" type="' + c + '">';
            for (var d = 0; d < b.length; d++)
                c += "<propertyArrayElement>", c += sb(b[d]), c += "</propertyArrayElement>";
            c = c + "</propertyArray>"
        } else 
            c = null;
        return c
    }
    c = rb(b);
    if (!c)
        return null;
    d = sb(b);
    return '<property id="' + Ha(a) + '" type="' + c + '">' + d + "</property>"
}, sb = function(a) {
    return "<![CDATA[" + String(a).replace(/]]\x3e/g,
    "]]]]\x3e<![CDATA[>") + "]]\x3e"
}, rb = function(a) {
    return ca(a) ? 0 == a%1 ? "long" : "double" : "boolean" == typeof a ? "bool" : v(a) ? "string" : null
};
var tb = function(a, b, c, d) {
    this.e = a;
    this.a = b;
    this.c = c ? c : "0.0.0";
    this.k = d || "always"
}, ub = {
    wc: "transparent",
    oc: "opaque",
    yc: "window"
}, vb = function(a) {
    return ib(ub, a) ? a : null
};
var J = function(a, b) {
    this.x = q(a) ? a : 0;
    this.y = q(b) ? b : 0
};
J.prototype.clone = function() {
    return new J(this.x, this.y)
};
J.prototype.ceil = function() {
    this.x = Math.ceil(this.x);
    this.y = Math.ceil(this.y);
    return this
};
J.prototype.floor = function() {
    this.x = Math.floor(this.x);
    this.y = Math.floor(this.y);
    return this
};
J.prototype.round = function() {
    this.x = Math.round(this.x);
    this.y = Math.round(this.y);
    return this
};
var K = function(a, b) {
    this.width = a;
    this.height = b
};
K.prototype.clone = function() {
    return new K(this.width, this.height)
};
K.prototype.ceil = function() {
    this.width = Math.ceil(this.width);
    this.height = Math.ceil(this.height);
    return this
};
K.prototype.floor = function() {
    this.width = Math.floor(this.width);
    this.height = Math.floor(this.height);
    return this
};
K.prototype.round = function() {
    this.width = Math.round(this.width);
    this.height = Math.round(this.height);
    return this
};
var L;
t: {
    var wb = n.navigator;
    if (wb) {
        var xb = wb.userAgent;
        if (xb) {
            L = xb;
            break t
        }
    }
    L = ""
};
var yb, zb, Ab, Bb = E(L, "Opera") || E(L, "OPR"), M = E(L, "Trident") || E(L, "MSIE"), N = E(L, "Gecko")&&!E(L.toLowerCase(), "webkit")&&!(E(L, "Trident") || E(L, "MSIE")), O = E(L.toLowerCase(), "webkit"), Cb = O && E(L, "Mobile"), Db = n.navigator || null, Eb = Db && Db.platform || "";
yb = E(Eb, "Mac");
zb = E(Eb, "Win");
var Fb = L;
Ab=!!Fb && E(Fb, "Android");
var Gb = n.navigator || null, Hb=!!Gb && E(Gb.appVersion || "", "X11"), Ib = function() {
    var a = n.document;
    return a ? a.documentMode : void 0
}, Jb = function() {
    var a = "", b;
    if (Bb && n.opera)
        return a = n.opera.version, w(a) ? a() : a;
    N ? b = /rv\:([^\);]+)(\)|;)/ : M ? b = /\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/ : O && (b = /WebKit\/(\S+)/);
    b && (a = (a = b.exec(L)) ? a[1] : "");
    return M && (b = Ib(), b > parseFloat(a)) ? String(b) : a
}(), Kb = {}, P = function(a) {
    return Kb[a] || (Kb[a] = 0 <= G(Jb, a))
}, Lb = n.document, Mb = Lb && M ? Ib() || ("CSS1Compat" == Lb.compatMode ? parseInt(Jb, 10) : 5) : void 0;
!N&&!M || M && M && 9 <= Mb || N && P("1.9.1");
M && P("9");
var Pb = function(a) {
    return a ? new Nb(Ob(a)) : ya || (ya = new Nb)
}, Qb = function(a, b) {
    return v(b) ? a.getElementById(b) : b
}, Sb = function(a, b) {
    eb(b, function(b, d) {
        "style" == d ? a.style.cssText = b : "class" == d ? a.className = b : "for" == d ? a.htmlFor = b : d in Rb ? a.setAttribute(Rb[d], b) : 0 == d.lastIndexOf("aria-", 0) || 0 == d.lastIndexOf("data-", 0) ? a.setAttribute(d, b) : a[d] = b
    })
}, Rb = {
    cellpadding: "cellPadding",
    cellspacing: "cellSpacing",
    colspan: "colSpan",
    frameborder: "frameBorder",
    height: "height",
    maxlength: "maxLength",
    role: "role",
    rowspan: "rowSpan",
    type: "type",
    usemap: "useMap",
    valign: "vAlign",
    width: "width"
}, Tb = function(a) {
    return a.parentWindow || a.defaultView
}, Ub = function(a) {
    return a && a.parentNode ? a.parentNode.removeChild(a) : null
}, Ob = function(a) {
    return 9 == a.nodeType ? a : a.ownerDocument || a.document
}, Nb = function(a) {
    this.a = a || n.document || document
};
Nb.prototype.P = function(a) {
    return Qb(this.a, a)
};
Nb.prototype.appendChild = function(a, b) {
    a.appendChild(b)
};
Nb.prototype.c = Ub;
Nb.prototype.contains = function(a, b) {
    if (a.contains && 1 == b.nodeType)
        return a == b || a.contains(b);
    if ("undefined" != typeof a.compareDocumentPosition)
        return a == b || Boolean(a.compareDocumentPosition(b) & 16);
    for (; b && a != b;)
        b = b.parentNode;
    return b == a
};
var Vb = function() {
    return O ? "Webkit" : N ? "Moz" : M ? "ms" : Bb ? "O" : null
};
var Wb=!1, Xb = "", Yb = function(a) {
    a = a.match(/[\d]+/g);
    if (!a)
        return "";
    a.length = 3;
    return a.join(".")
};
if (navigator.plugins && navigator.plugins.length) {
    var Zb = navigator.plugins["Shockwave Flash"];
    Zb && (Wb=!0, Zb.description && (Xb = Yb(Zb.description)));
    navigator.plugins["Shockwave Flash 2.0"] && (Wb=!0, Xb = "2.0.0.11")
} else if (navigator.mimeTypes && navigator.mimeTypes.length) {
    var $b = navigator.mimeTypes["application/x-shockwave-flash"];
    (Wb = $b && $b.enabledPlugin) && (Xb = Yb($b.enabledPlugin.description))
} else 
    try {
        var ac = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7"), Wb=!0, Xb = Yb(ac.GetVariable("$version"))
} catch (bc) {
    try {
        ac =
        new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6"), Wb=!0, Xb = "6.0.21"
    } catch (cc) {
        try {
            ac = new ActiveXObject("ShockwaveFlash.ShockwaveFlash"), Wb=!0, Xb = Yb(ac.GetVariable("$version"))
        } catch (dc) {}
    }
}
var ec = Xb;
var fc, gc, hc, ic, jc, kc, lc;
lc = kc = jc = ic = hc = gc = fc=!1;
var mc = L;
mc && ( - 1 != mc.indexOf("Firefox") ? fc=!0 : - 1 != mc.indexOf("Camino") ? gc=!0 : - 1 != mc.indexOf("iPad") ? ic=!0 : - 1 != mc.indexOf("iPhone")||-1 != mc.indexOf("iPod") ? hc=!0 : - 1 != mc.indexOf("Chrome") ? kc=!0 : - 1 != mc.indexOf("Android") ? jc=!0 : - 1 != mc.indexOf("Safari") && (lc=!0));
var nc = fc, oc = gc, pc = hc, qc = ic, rc = jc, sc = kc, tc = lc;
var uc = function(a) {
    return (a = a.exec(L)) ? a[1] : ""
}, vc = function() {
    if (nc)
        return uc(/Firefox\/([0-9.]+)/);
    if (M || Bb)
        return Jb;
    if (sc)
        return uc(/Chrome\/([0-9.]+)/);
    if (tc)
        return uc(/Version\/([0-9.]+)/);
    if (pc || qc) {
        var a;
        if (a = /Version\/(\S+).*Mobile\/(\S+)/.exec(L))
            return a[1] + "." + a[2]
    } else {
        if (rc)
            return (a = uc(/Android\s+([0-9.]+)/)) ? a : uc(/Version\/([0-9.]+)/);
        if (oc)
            return uc(/Camino\/([0-9.]+)/)
    }
    return ""
}(), wc = function(a) {
    return 0 <= G(vc, a)
};
var xc = function(a) {
    return (M || N || O || Bb ||!1)&&!Cb && 0 <= G(ec, a)
}, zc = function(a, b, c) {
    return !window.postMessage || b && 0 < b.length && (!a || null == window.Modernizr)?!1 : a ? Ra(b, function(a) {
        if ("svgFilters" == a) {
            if (!(M && P("10") && M && 10 <= Mb || sc && wc("18") || nc && wc("11")))
                return !1
        } else if ("svgFeImage" == a) {
            if (!(M && P("10") && M && 10 <= Mb || sc && wc("29")))
                return !1
        } else if (!t(a))
            return !1;
        return !0
    }) : (!!document.createElement("canvas").getContext || M && wc("9")) && (!c || yc())
}, yc = function() {
    var a = (O || "CSS1Compat" != document.compatMode ? document.body ||
    document.documentElement : document.documentElement).style;
    return q(a.animationName) || q(a[Vb() + "AnimationName"])
}, Ac = function(a) {
    return 0 <= G(ec, "9.115") || 0 <= G(ec, "9.28") && "window" == a
};
var Bc = function(a, b, c, d) {
    this.top = a;
    this.right = b;
    this.bottom = c;
    this.left = d
};
l = Bc.prototype;
l.clone = function() {
    return new Bc(this.top, this.right, this.bottom, this.left)
};
l.contains = function(a) {
    return this && a ? a instanceof Bc ? a.left >= this.left && a.right <= this.right && a.top >= this.top && a.bottom <= this.bottom : a.x >= this.left && a.x <= this.right && a.y >= this.top && a.y <= this.bottom : !1
};
l.ceil = function() {
    this.top = Math.ceil(this.top);
    this.right = Math.ceil(this.right);
    this.bottom = Math.ceil(this.bottom);
    this.left = Math.ceil(this.left);
    return this
};
l.floor = function() {
    this.top = Math.floor(this.top);
    this.right = Math.floor(this.right);
    this.bottom = Math.floor(this.bottom);
    this.left = Math.floor(this.left);
    return this
};
l.round = function() {
    this.top = Math.round(this.top);
    this.right = Math.round(this.right);
    this.bottom = Math.round(this.bottom);
    this.left = Math.round(this.left);
    return this
};
var Cc = function(a, b, c, d) {
    this.left = a;
    this.top = b;
    this.width = c;
    this.height = d
};
l = Cc.prototype;
l.clone = function() {
    return new Cc(this.left, this.top, this.width, this.height)
};
l.contains = function(a) {
    return a instanceof Cc ? this.left <= a.left && this.left + this.width >= a.left + a.width && this.top <= a.top && this.top + this.height >= a.top + a.height : a.x >= this.left && a.x <= this.left + this.width && a.y >= this.top && a.y <= this.top + this.height
};
l.ceil = function() {
    this.left = Math.ceil(this.left);
    this.top = Math.ceil(this.top);
    this.width = Math.ceil(this.width);
    this.height = Math.ceil(this.height);
    return this
};
l.floor = function() {
    this.left = Math.floor(this.left);
    this.top = Math.floor(this.top);
    this.width = Math.floor(this.width);
    this.height = Math.floor(this.height);
    return this
};
l.round = function() {
    this.left = Math.round(this.left);
    this.top = Math.round(this.top);
    this.width = Math.round(this.width);
    this.height = Math.round(this.height);
    return this
};
var Ec = function(a, b, c) {
    if (v(b))(b = Dc(a, b)) && (a.style[b] = c);
    else 
        for (var d in b) {
            c = a;
            var e = b[d], f = Dc(c, d);
            f && (c.style[f] = e)
        }
}, Dc = function(a, b) {
    var c = Ja(b);
    if (void 0 === a.style[c]) {
        var d = Vb() + Ka(c);
        if (void 0 !== a.style[d])
            return d
    }
    return c
}, Fc = function(a) {
    var b;
    t: {
        b = Ob(a);
        if (b.defaultView && b.defaultView.getComputedStyle && (b = b.defaultView.getComputedStyle(a, null))) {
            b = b.position || b.getPropertyValue("position") || "";
            break t
        }
        b = ""
    }
    return b || (a.currentStyle ? a.currentStyle.position : null) || a.style && a.style.position
},
Gc = function(a) {
    if (M&&!(M && 8 <= Mb))
        return a.offsetParent;
    var b = Ob(a), c = Fc(a), d = "fixed" == c || "absolute" == c;
    for (a = a.parentNode; a && a != b; a = a.parentNode)
        if (c = Fc(a), d = d && "static" == c && a != b.documentElement && a != b.body, !d && (a.scrollWidth > a.clientWidth || a.scrollHeight > a.clientHeight || "fixed" == c || "absolute" == c || "relative" == c))
            return a;
    return null
}, Ic = function(a, b, c) {
    if (b instanceof K)
        c = b.height, b = b.width;
    else if (void 0 == c)
        throw Error("missing height argument");
    a.style.width = Hc(b, !0);
    a.style.height = Hc(c, !0)
}, Hc =
function(a, b) {
    "number" == typeof a && (a = (b ? Math.round(a) : a) + "px");
    return a
};
var Jc = function(a) {
    Jc[" "](a);
    return a
};
Jc[" "] = aa;
var Kc=!M || M && 9 <= Mb, Lc = M&&!P("9");
!O || P("528");
N && P("1.9b") || M && P("8") || Bb && P("9.5") || O && P("528");
N&&!P("8") || M && P("9");
var Mc = function(a, b) {
    B.call(this, a ? a.type : "");
    this.a = this.target = null;
    this.button = this.clientY = this.clientX = 0;
    this.metaKey = this.shiftKey = this.ctrlKey=!1;
    this.c = this.state = null;
    if (a) {
        this.type = a.type;
        this.target = a.target || a.srcElement;
        this.a = b;
        var c = a.relatedTarget;
        if (c && N)
            try {
                Jc(c.nodeName)
            } catch (d) {}
        this.clientX = void 0 !== a.clientX ? a.clientX : a.pageX;
        this.clientY = void 0 !== a.clientY ? a.clientY : a.pageY;
        this.button = a.button;
        this.ctrlKey = a.ctrlKey;
        this.shiftKey = a.shiftKey;
        this.metaKey = a.metaKey;
        this.state =
        a.state;
        this.c = a;
        a.defaultPrevented && this.preventDefault()
    }
};
z(Mc, B);
Mc.prototype.preventDefault = function() {
    Mc.M.preventDefault.call(this);
    var a = this.c;
    if (a.preventDefault)
        a.preventDefault();
    else if (a.returnValue=!1, Lc)
        try {
            if (a.ctrlKey || 112 <= a.keyCode && 123 >= a.keyCode)
                a.keyCode =- 1
    } catch (b) {}
};
var Nc = "closure_listenable_" + (1E6 * Math.random() | 0), Oc = 0;
var Pc = function(a, b, c, d, e) {
    this.ga = a;
    this.a = null;
    this.src = b;
    this.type = c;
    this.Fa=!!d;
    this.Ja = e;
    this.ib=++Oc;
    this.ta = this.Ga=!1
}, Qc = function(a) {
    a.ta=!0;
    a.ga = null;
    a.a = null;
    a.src = null;
    a.Ja = null
};
var Rc = function(a) {
    this.src = a;
    this.a = {};
    this.c = 0
};
Rc.prototype.add = function(a, b, c, d, e) {
    var f = a.toString();
    a = this.a[f];
    a || (a = this.a[f] = [], this.c++);
    var g = Sc(a, b, d, e);
    - 1 < g ? (b = a[g], c || (b.Ga=!1)) : (b = new Pc(b, this.src, f, !!d, e), b.Ga = c, a.push(b));
    return b
};
Rc.prototype.remove = function(a, b, c, d) {
    a = a.toString();
    if (!(a in this.a))
        return !1;
    var e = this.a[a];
    b = Sc(e, b, c, d);
    return - 1 < b ? (Qc(e[b]), La.splice.call(e, b, 1), 0 == e.length && (delete this.a[a], this.c--), !0) : !1
};
var Tc = function(a, b) {
    var c = b.type;
    if (!(c in a.a))
        return !1;
    var d = I(a.a[c], b);
    d && (Qc(b), 0 == a.a[c].length && (delete a.a[c], a.c--));
    return d
}, Uc = function(a, b, c, d, e) {
    a = a.a[b.toString()];
    b =- 1;
    a && (b = Sc(a, c, d, e));
    return - 1 < b ? a[b] : null
}, Sc = function(a, b, c, d) {
    for (var e = 0; e < a.length; ++e) {
        var f = a[e];
        if (!f.ta && f.ga == b && f.Fa==!!c && f.Ja == d)
            return e
    }
    return - 1
};
var Vc = "closure_lm_" + (1E6 * Math.random() | 0), Wc = {}, Xc = 0, Yc = function(a, b, c, d, e) {
    if (u(b)) {
        for (var f = 0; f < b.length; f++)
            Yc(a, b[f], c, d, e);
        return null
    }
    c = Zc(c);
    if (a && a[Nc])
        a = a.c(b, c, d, e);
    else {
        if (!b)
            throw Error("Invalid event type");
        var f=!!d, g = $c(a);
        g || (a[Vc] = g = new Rc(a));
        c = g.add(b, c, !1, d, e);
        c.a || (d = ad(), c.a = d, d.src = a, d.ga = c, a.addEventListener ? a.addEventListener(b.toString(), d, f) : a.attachEvent(bd(b.toString()), d), Xc++);
        a = c
    }
    return a
}, ad = function() {
    var a = cd, b = Kc ? function(c) {
        return a.call(b.src, b.ga, c)
    }
    : function(c) {
        c =
        a.call(b.src, b.ga, c);
        if (!c)
            return c
    };
    return b
}, dd = function(a, b, c, d, e) {
    if (u(b))
        for (var f = 0; f < b.length; f++)
            dd(a, b[f], c, d, e);
    else 
        c = Zc(c), a && a[Nc] ? a.T(b, c, d, e) : a && (a = $c(a)) && (b = Uc(a, b, c, !!d, e)) && ed(b)
}, ed = function(a) {
    if (ca(a) ||!a || a.ta)
        return !1;
    var b = a.src;
    if (b && b[Nc])
        return Tc(b.ea, a);
    var c = a.type, d = a.a;
    b.removeEventListener ? b.removeEventListener(c, d, a.Fa) : b.detachEvent && b.detachEvent(bd(c), d);
    Xc--;
    (c = $c(b)) ? (Tc(c, a), 0 == c.c && (c.src = null, b[Vc] = null)) : Qc(a);
    return !0
}, bd = function(a) {
    return a in Wc ? Wc[a] :
    Wc[a] = "on" + a
}, gd = function(a, b, c, d) {
    var e = 1;
    if (a = $c(a))
        if (b = a.a[b.toString()])
            for (b = b.concat(), a = 0; a < b.length; a++) {
                var f = b[a];
                f && f.Fa == c&&!f.ta && (e&=!1 !== fd(f, d))
            }
    return Boolean(e)
}, fd = function(a, b) {
    var c = a.ga, d = a.Ja || a.src;
    a.Ga && ed(a);
    return c.call(d, b)
}, cd = function(a, b) {
    if (a.ta)
        return !0;
    if (!Kc) {
        var c = b || t("window.event"), d = new Mc(c, this), e=!0;
        if (!(0 > c.keyCode || void 0 != c.returnValue)) {
            t:
            {
                var f=!1;
                if (0 == c.keyCode)
                    try {
                        c.keyCode =- 1;
                        break t
                } catch (g) {
                    f=!0
                }
                if (f || void 0 == c.returnValue)
                    c.returnValue=!0
            }
            c =
            [];
            for (f = d.a; f; f = f.parentNode)
                c.push(f);
            for (var f = a.type, h = c.length - 1; 0 <= h; h--)
                d.a = c[h], e&=gd(c[h], f, !0, d);
            for (h = 0; h < c.length; h++)
                d.a = c[h], e&=gd(c[h], f, !1, d)
            }
        return e
    }
    return fd(a, new Mc(b, this))
}, $c = function(a) {
    a = a[Vc];
    return a instanceof Rc ? a : null
}, hd = "__closure_events_fn_" + (1E9 * Math.random()>>>0), Zc = function(a) {
    if (w(a))
        return a;
    a[hd] || (a[hd] = function(b) {
        return a.handleEvent(b)
    });
    return a[hd]
};
var Q = function() {
    A.call(this);
    this.ea = new Rc(this);
    this.X = this;
    this.F = null
};
z(Q, A);
Q.prototype[Nc]=!0;
Q.prototype.Q = function(a) {
    this.F = a
};
Q.prototype.addEventListener = function(a, b, c, d) {
    Yc(this, a, b, c, d)
};
Q.prototype.removeEventListener = function(a, b, c, d) {
    dd(this, a, b, c, d)
};
var R = function(a, b) {
    var c, d = a.F;
    if (d) {
        c = [];
        for (var e = 1; d; d = d.F)
            c.push(d), ++e
    }
    var d = a.X, e = b, f = e.type || e;
    if (v(e))
        e = new B(e, d);
    else if (e instanceof B)
        e.target = e.target || d;
    else {
        var g = e, e = new B(f, d);
        lb(e, g)
    }
    var g=!0, h;
    if (c)
        for (var k = c.length - 1; 0 <= k; k--)
            h = e.a = c[k], g = id(h, f, !0, e) && g;
    h = e.a = d;
    g = id(h, f, !0, e) && g;
    g = id(h, f, !1, e) && g;
    if (c)
        for (k = 0; k < c.length; k++)
            h = e.a = c[k], g = id(h, f, !1, e) && g
};
Q.prototype.k = function() {
    Q.M.k.call(this);
    if (this.ea) {
        var a = this.ea, b = 0, c;
        for (c in a.a) {
            for (var d = a.a[c], e = 0; e < d.length; e++)
                ++b, Qc(d[e]);
            delete a.a[c];
            a.c--
        }
    }
    this.F = null
};
Q.prototype.c = function(a, b, c, d) {
    return this.ea.add(String(a), b, !1, c, d)
};
Q.prototype.T = function(a, b, c, d) {
    return this.ea.remove(String(a), b, c, d)
};
var id = function(a, b, c, d) {
    b = a.ea.a[String(b)];
    if (!b)
        return !0;
    b = b.concat();
    for (var e=!0, f = 0; f < b.length; ++f) {
        var g = b[f];
        if (g&&!g.ta && g.Fa == c) {
            var h = g.ga, k = g.Ja || g.src;
            g.Ga && Tc(a.ea, g);
            e=!1 !== h.call(k, d) && e
        }
    }
    return e && 0 != d.kb
};
var jd = function(a) {
    n.setTimeout(function() {
        throw a;
    }, 0)
}, kd, ld = function() {
    var a = n.MessageChannel;
    "undefined" === typeof a && "undefined" !== typeof window && window.postMessage && window.addEventListener && (a = function() {
        var a = document.createElement("iframe");
        a.style.display = "none";
        a.src = "";
        document.documentElement.appendChild(a);
        var b = a.contentWindow, a = b.document;
        a.open();
        a.write("");
        a.close();
        var c = "callImmediate" + Math.random(), d = "file:" == b.location.protocol ? "*": b.location.protocol + "//" + b.location.host, a = x(function(a) {
            if (("*" ==
            d || a.origin == d) && a.data == c)
                this.port1.onmessage()
        }, this);
        b.addEventListener("message", a, !1);
        this.port1 = {};
        this.port2 = {
            postMessage: function() {
                b.postMessage(c, d)
            }
        }
    });
    if ("undefined" !== typeof a&&!E(L, "Trident")&&!E(L, "MSIE")) {
        var b = new a, c = {}, d = c;
        b.port1.onmessage = function() {
            if (q(c.next)) {
                c = c.next;
                var a = c.nb;
                c.nb = null;
                a()
            }
        };
        return function(a) {
            d.next = {
                nb: a
            };
            d = d.next;
            b.port2.postMessage(0)
        }
    }
    return "undefined" !== typeof document && "onreadystatechange"in document.createElement("script") ? function(a) {
        var b = document.createElement("script");
        b.onreadystatechange = function() {
            b.onreadystatechange = null;
            b.parentNode.removeChild(b);
            b = null;
            a();
            a = null
        };
        document.documentElement.appendChild(b)
    } : function(a) {
        n.setTimeout(a, 0)
    }
};
var rd = function(a, b) {
    md || nd();
    od || (md(), od=!0);
    pd.push(new qd(a, b))
}, md, nd = function() {
    if (n.Promise && n.Promise.resolve) {
        var a = n.Promise.resolve();
        md = function() {
            a.then(sd)
        }
    } else 
        md = function() {
            var a = sd;
            !w(n.setImmediate) || n.Window && n.Window.prototype.setImmediate == n.setImmediate ? (kd || (kd = ld()), kd(a)) : n.setImmediate(a)
        }
}, od=!1, pd = [], sd = function() {
    for (; pd.length;) {
        var a = pd;
        pd = [];
        for (var b = 0; b < a.length; b++) {
            var c = a[b];
            try {
                c.a.call(c.c)
            } catch (d) {
                jd(d)
            }
        }
    }
    od=!1
}, qd = function(a, b) {
    this.a = a;
    this.c = b
};
var td = function(a) {
    a.prototype.then = a.prototype.then;
    a.prototype.$goog_Thenable=!0
}, ud = function(a) {
    if (!a)
        return !1;
    try {
        return !!a.$goog_Thenable
    } catch (b) {
        return !1
    }
};
var wd = function(a, b) {
    this.c = 0;
    this.j = void 0;
    this.a = this.g = null;
    this.e = this.k=!1;
    try {
        var c = this;
        a.call(b, function(a) {
            vd(c, 2, a)
        }, function(a) {
            vd(c, 3, a)
        })
    } catch (d) {
        vd(this, 3, d)
    }
};
wd.prototype.then = function(a, b, c) {
    return xd(this, w(a) ? a : null, w(b) ? b : null, c)
};
td(wd);
var zd = function(a, b) {
    a.a && a.a.length || 2 != a.c && 3 != a.c || yd(a);
    a.a || (a.a = []);
    a.a.push(b)
}, xd = function(a, b, c, d) {
    var e = {
        Ea: null,
        eb: null,
        fb: null
    };
    e.Ea = new wd(function(a, g) {
        e.eb = b ? function(c) {
            try {
                var e = b.call(d, c);
                a(e)
            } catch (m) {
                g(m)
            }
        } : a;
        e.fb = c ? function(b) {
            try {
                var e = c.call(d, b);
                a(e)
            } catch (m) {
                g(m)
            }
        } : g
    });
    e.Ea.g = a;
    zd(a, e);
    return e.Ea
};
wd.prototype.m = function(a) {
    this.c = 0;
    vd(this, 2, a)
};
wd.prototype.n = function(a) {
    this.c = 0;
    vd(this, 3, a)
};
var vd = function(a, b, c) {
    if (0 == a.c) {
        if (a == c)
            b = 3, c = new TypeError("Promise cannot resolve to itself");
        else {
            if (ud(c)) {
                a.c = 1;
                c.then(a.m, a.n, a);
                return 
            }
            if (da(c))
                try {
                    var d = c.then;
                    if (w(d)) {
                        Ad(a, c, d);
                        return 
                    }
            } catch (e) {
                b = 3, c = e
            }
        }
        a.j = c;
        a.c = b;
        yd(a);
        3 != b || Bd(a, c)
    }
}, Ad = function(a, b, c) {
    a.c = 1;
    var d=!1, e = function(b) {
        d || (d=!0, a.m(b))
    }, f = function(b) {
        d || (d=!0, a.n(b))
    };
    try {
        c.call(b, e, f)
    } catch (g) {
        f(g)
    }
}, yd = function(a) {
    a.k || (a.k=!0, rd(a.o, a))
};
wd.prototype.o = function() {
    for (; this.a && this.a.length;) {
        var a = this.a;
        this.a = [];
        for (var b = 0; b < a.length; b++) {
            var c = a[b], d = this.j;
            if (2 == this.c)
                c.eb(d);
            else {
                if (c.Ea)
                    for (var e = void 0, e = this; e && e.e; e = e.g)
                        e.e=!1;
                c.fb(d)
            }
        }
    }
    this.k=!1
};
var Bd = function(a, b) {
    a.e=!0;
    rd(function() {
        a.e && Cd.call(null, b)
    })
}, Cd = jd;
var Dd = function(a, b) {
    Q.call(this);
    this.j = a || 1;
    this.e = b || n;
    this.m = x(this.o, this);
    this.n = y()
};
z(Dd, Q);
Dd.prototype.g=!1;
Dd.prototype.a = null;
Dd.prototype.o = function() {
    if (this.g) {
        var a = y() - this.n;
        0 < a && a < .8 * this.j ? this.a = this.e.setTimeout(this.m, this.j - a) : (this.a && (this.e.clearTimeout(this.a), this.a = null), R(this, "tick"), this.g && (this.a = this.e.setTimeout(this.m, this.j), this.n = y()))
    }
};
var Fd = function(a) {
    a.g=!0;
    a.a || (a.a = a.e.setTimeout(a.m, a.j), a.n = y())
};
Dd.prototype.k = function() {
    Dd.M.k.call(this);
    this.g=!1;
    this.a && (this.e.clearTimeout(this.a), this.a = null);
    delete this.e
};
var Gd = function(a, b, c) {
    if (w(a))
        c && (a = x(a, c));
    else if (a && "function" == typeof a.handleEvent)
        a = x(a.handleEvent, a);
    else 
        throw Error("Invalid listener argument");
    return 2147483647 < b?-1 : n.setTimeout(a, b || 0)
};
var Hd = {
    vc: "top",
    tc: "right",
    Vb: "bottom",
    nc: "left"
}, Id = function(a) {
    return ib(Hd, a) ? a : null
};
var Jd = function(a) {
    a = a ? a.toLowerCase() : "";
    switch (a) {
    case "normal":
        return "normal";
    case "lightbox":
        return "lightbox";
    case "pushdown":
        return "pushdown"
    }
    return null
};
var Kd = function(a, b) {
    this.a = a;
    this.expansionMode = b;
    this.c = "pushdown" == b
};
var Ld = function(a, b, c, d, e) {
    this.a = a;
    this.j = b;
    this.k =- 1 < b;
    this.m = c;
    this.c = d;
    this.e =- 1 < d;
    this.g = e
};
var Md = function() {};
var Nd = function() {};
var Od = function() {
    this.e = this.a = this.c=!1
};
var Qd = function(a, b) {
    this.top = parseInt(a, 10);
    this.c = Pd(a);
    this.left = parseInt(b, 10);
    this.a = Pd(b)
}, Pd = function(a) {
    return Sa(Rd, function(b) {
        var c = String(b).toLowerCase();
        b = String(a.substr(a.length - b.length, b.length)).toLowerCase();
        return 0 == (c < b?-1 : c == b ? 0 : 1)
    }) || "px"
}, Rd = ["px", "%", "pxc"];
var Sd = "StopIteration"in n ? n.StopIteration: Error("StopIteration"), Td = function() {};
Td.prototype.next = function() {
    throw Sd;
};
Td.prototype.g = function() {
    return this
};
var S = function(a, b) {
    this.c = {};
    this.a = [];
    this.k = this.e = 0;
    var c = arguments.length;
    if (1 < c) {
        if (c%2)
            throw Error("Uneven number of arguments");
        for (var d = 0; d < c; d += 2)
            Ud(this, arguments[d], arguments[d + 1])
    } else if (a) {
        if (a instanceof S)
            d = a.za(), c = a.ya();
        else {
            var c = [], e = 0;
            for (d in a)
                c[e++] = d;
            d = c;
            c = hb(a)
        }
        for (e = 0; e < d.length; e++)
            Ud(this, d[e], c[e])
    }
};
S.prototype.ya = function() {
    Vd(this);
    for (var a = [], b = 0; b < this.a.length; b++)
        a.push(this.c[this.a[b]]);
    return a
};
S.prototype.za = function() {
    Vd(this);
    return this.a.concat()
};
S.prototype.clear = function() {
    this.c = {};
    this.k = this.e = this.a.length = 0
};
S.prototype.remove = function(a) {
    return Wd(this.c, a) ? (delete this.c[a], this.e--, this.k++, this.a.length > 2 * this.e && Vd(this), !0) : !1
};
var Vd = function(a) {
    if (a.e != a.a.length) {
        for (var b = 0, c = 0; b < a.a.length;) {
            var d = a.a[b];
            Wd(a.c, d) && (a.a[c++] = d);
            b++
        }
        a.a.length = c
    }
    if (a.e != a.a.length) {
        for (var e = {}, c = b = 0; b < a.a.length;)
            d = a.a[b], Wd(e, d) || (a.a[c++] = d, e[d] = 1), b++;
        a.a.length = c
    }
}, Xd = function(a, b) {
    return Wd(a.c, b) ? a.c[b] : void 0
}, Ud = function(a, b, c) {
    Wd(a.c, b) || (a.e++, a.a.push(b), a.k++);
    a.c[b] = c
};
S.prototype.forEach = function(a, b) {
    for (var c = this.za(), d = 0; d < c.length; d++) {
        var e = c[d];
        a.call(b, Xd(this, e), e, this)
    }
};
S.prototype.clone = function() {
    return new S(this)
};
S.prototype.g = function(a) {
    Vd(this);
    var b = 0, c = this.a, d = this.c, e = this.k, f = this, g = new Td;
    g.next = function() {
        for (; ;) {
            if (e != f.k)
                throw Error("The map has changed since the iterator was created");
            if (b >= c.length)
                throw Sd;
            var g = c[b++];
            return a ? g : d[g]
        }
    };
    return g
};
var Wd = function(a, b) {
    return Object.prototype.hasOwnProperty.call(a, b)
};
var Yd = function(a, b) {
    switch (a) {
    case "left":
        switch (b) {
        case "mtfDuration":
            return "mtfLeftDuration";
        case "wmode":
            return "mtfLeftWmode";
        case "mtfTop":
        case "mtfLeft":
            return "mtfLeftFloat";
        default:
            return null
        }
    case "right":
        switch (b) {
        case "mtfDuration":
            return "mtfRightDuration";
        case "wmode":
            return "mtfRightWmode";
        case "mtfTop":
        case "mtfLeft":
            return "mtfRightFloat";
        default:
            return null
        }
    case "bottom":
        switch (b) {
        case "mtfDuration":
            return "mtfBottomDuration";
        case "wmode":
            return "mtfBottomWmode";
        case "mtfTop":
        case "mtfLeft":
            return "mtfBottomFloat";
        default:
            return null
        }
    case "top":
        switch (b) {
        case "mtfDuration":
            return "mtfTopDuration";
        case "wmode":
            return "mtfTopWmode";
        case "mtfTop":
        case "mtfLeft":
            return "mtfTopFloat";
        default:
            return null
        }
    default:
        return b
    }
};
var Zd = function(a, b, c, d, e) {
    this.id = a;
    this.k = b;
    this.p = c;
    this.o = "BANNER" == c || "EXPANDABLE" == c;
    this.width = d.width;
    this.height = d.height;
    this.j = null;
    this.q = e;
    this.G = 0;
    this.A = this.B = this.n = this.g = this.N = this.a = this.e = this.v = this.c = this.m = null
}, $d = {
    zIndex: "zindex",
    expansionMode: "expansionMode",
    hideObjects: "hideObjects",
    duration: "mtfDuration",
    wmode: "wmode",
    top: "mtfTop",
    left: "mtfLeft",
    "multiFloat.top.duration": "mtfTopDuration",
    "multiFloat.top.wmode": "mtfTopWmode",
    "multiFloat.top.position": "mtfTopFloat",
    "multiFloat.right.duration": "mtfRightDuration",
    "multiFloat.right.wmode": "mtfRightWmode",
    "multiFloat.right.position": "mtfRightFloat",
    "multiFloat.bottom.duration": "mtfBottomDuration",
    "multiFloat.bottom.wmode": "mtfBottomWmode",
    "multiFloat.bottom.position": "mtfBottomFloat",
    "multiFloat.left.duration": "mtfLeftDuration",
    "multiFloat.left.wmode": "mtfLeftWmode",
    "multiFloat.left.position": "mtfLeftFloat"
}, ae = function(a) {
    var b = a.a && a.a.a;
    if (b) {
        var c = b.left, d = b.top;
        3 >= Math.abs(c) && (b.left = 0);
        3 >= Math.abs(d) && (b.top = 0);
        3 >= Math.abs(c + b.width - a.width) && (0 == b.left ?
        b.width = a.width : b.left = a.width - b.width);
        3 >= Math.abs(d + b.height - a.height) && (0 == b.top ? b.height = a.height : b.top = a.height - b.height)
    }
}, be = function(a) {
    return !!a.c && 0 <= G(a.c.c, "1.0.0") && 0 > G(a.c.c, "2.0.0")
};
Zd.prototype.toString = function() {
    return "[PrimaryFile " + this.id + "]"
};
var ce = function(a, b) {
    if ("zindex"in b) {
        var c = parseInt(b.zindex, 10);
        isNaN(c) || (a.G = c)
    }
    "expansionMode"in b&&!D(F(b.expansionMode)) && null !== a.a && (c = Jd(b.expansionMode), null !== c && (a.a.expansionMode = c));
    "hideObjects"in b&&!D(F(b.hideObjects)) && null !== a.g && (a.g.a = "true" == b.hideObjects);
    c = null;
    if (null !== a.e) {
        var c = a.e.g || null, d = Yd(c, "mtfDuration"), d = parseInt(b[d], 10);
        isNaN(d) || (a.e.c = d, a.e.e=!0);
        d = [];
        if (null === c)
            d[0] = b.mtfTop, d[1] = b.mtfLeft;
        else {
            var e = Yd(c, "mtfTop");
            null != b[e] && (d = b[e].split(","))
        }
        2 <= d.length &&
        (e = parseInt(d[0], 10), isNaN(e) || (a.e.a.top = e, a.e.a.c = Pd(d[0])), e = parseInt(d[1], 10), isNaN(e) || (a.e.a.left = e, a.e.a.a = Pd(d[1])))
    }
    a.c && (c = Yd(c, "wmode"), c = vb(b[c]), null !== c && (a.c.a = c))
};
var de = function(a) {
    this.a = a
};
ha("studio.common.mde.Direction", de);
de.prototype.toString = function() {
    return (this.a & 2 ? "b" : "t") + (this.a & 1 ? "r" : "l")
};
var T = function(a) {
    A.call(this);
    this.e = a;
    this.a = {}
};
z(T, A);
var ee = [];
T.prototype.c = function(a, b, c, d) {
    u(b) || (b && (ee[0] = b.toString()), b = ee);
    for (var e = 0; e < b.length; e++) {
        var f = Yc(a, b[e], c || this.handleEvent, d ||!1, this.e || this);
        if (!f)
            break;
        this.a[f.ib] = f
    }
    return this
};
T.prototype.T = function(a, b, c, d, e) {
    if (u(b))
        for (var f = 0; f < b.length; f++)
            this.T(a, b[f], c, d, e);
    else 
        c = c || this.handleEvent, e = e || this.e || this, c = Zc(c), d=!!d, b = a && a[Nc] ? Uc(a.ea, String(b), c, d, e) : a ? (a = $c(a)) ? Uc(a, b, c, d, e) : null : null, b && (ed(b), delete this.a[b.ib]);
    return this
};
var fe = function(a) {
    eb(a.a, ed);
    a.a = {}
};
T.prototype.k = function() {
    T.M.k.call(this);
    fe(this)
};
T.prototype.handleEvent = function() {
    throw Error("EventHandler.handleEvent not implemented");
};
var ge = void 0, je = function() {
    void 0 === ge && (ge = Sa(he, ie) || "none");
    return ge
}, ie = function(a) {
    switch (a) {
    case "gdn":
        return window.IN_ADSENSE_IFRAME && CreativeToolset;
    case "yahoo":
        return window.Y && window.Y.SandBox && window.Y.SandBox.vendor;
    case "msn":
        var b;
        t:
        {
            try {
                b = window.$WLXRmAd || window.parent && window.parent.$WLXRmAd;
                break t
            } catch (c) {}
            b = void 0
        }
        return !!b;
    case "safe":
        return !!t("$sf.ext");
    case "mraid":
        return window.mraid;
    default:
        return !1
    }
}, he = "gdn mraid safe yahoo msn none".split(" ");
var ke = function() {
    try {
        if ("" != n.document.referrer)
            return n.document.referrer;
        if (n.location.ancestorOrigins && n.location.ancestorOrigins[0])
            return n.location.ancestorOrigins[0];
        if (top != n)
            return n.parent.location.href
    } catch (a) {}
    return n.location.href
};
var me = function(a, b) {
    var c = Array.prototype.slice.call(arguments), d = c.shift();
    if ("undefined" == typeof d)
        throw Error("[goog.string.format] Template required");
    return d.replace(/%([0\-\ \+]*)(\d+)?(\.(\d+))?([%sfdiu])/g, function(a, b, d, h, k, m, r, p) {
        if ("%" == m)
            return "%";
        var s = c.shift();
        if ("undefined" == typeof s)
            throw Error("[goog.string.format] Not enough arguments");
        arguments[0] = s;
        return le[m].apply(null, arguments)
    })
}, le = {
    s: function(a, b, c) {
        return isNaN(c) || "" == c || a.length >= c ? a : a =- 1 < b.indexOf("-", 0) ? a + Array(c -
        a.length + 1).join(" ") : Array(c - a.length + 1).join(" ") + a
    },
    f: function(a, b, c, d, e) {
        d = a.toString();
        isNaN(e) || "" == e || (d = a.toFixed(e));
        var f;
        f = 0 > a ? "-" : 0 <= b.indexOf("+") ? "+" : 0 <= b.indexOf(" ") ? " " : "";
        0 <= a && (d = f + d);
        if (isNaN(c) || d.length >= c)
            return d;
        d = isNaN(e) ? Math.abs(a).toString() : Math.abs(a).toFixed(e);
        a = c - d.length - f.length;
        return d = 0 <= b.indexOf("-", 0) ? f + d + Array(a + 1).join(" ") : f + Array(a + 1).join(0 <= b.indexOf("0", 0) ? "0" : " ") + d
    },
    d: function(a, b, c, d, e, f, g, h) {
        return le.f(parseInt(a, 10), b, c, d, 0, f, g, h)
    }
};
le.i = le.d;
le.u = le.d;
var ne = /^(?:([^:/?#.]+):)?(?:\/\/(?:([^/?#]*)@)?([^/#?]*?)(?::([0-9]+))?(?=[/#?]|$))?([^?#]+)?(?:\?([^#]*))?(?:#(.*))?$/, pe = function(a) {
    if (oe) {
        oe=!1;
        var b = n.location;
        if (b) {
            var c = b.href;
            if (c && (c = (c = pe(c)[3] || null) ? decodeURI(c) : c) && c != b.hostname)
                throw oe=!0, Error();
        }
    }
    return a.match(ne)
}, oe = O, qe = function(a, b) {
    var c = [a, "&", b];
    if (c[1]) {
        var d = c[0], e = d.indexOf("#");
        0 <= e && (c.push(d.substr(e)), c[0] = d = d.substr(0, e));
        e = d.indexOf("?");
        0 > e ? c[1] = "?" : e == d.length - 1 && (c[1] = void 0)
    }
    return c.join("")
}, re = /#|$/;
var se = function(a, b) {
    var c;
    a instanceof se ? (this.ia = q(b) ? b : a.ia, te(this, a.ha), this.Na = a.Na, this.ua = a.ua, ue(this, a.Ma), this.La = a.La, ve(this, a.a.clone()), this.Ka = a.Ka) : a && (c = pe(String(a))) ? (this.ia=!!b, te(this, c[1] || "", !0), this.Na = we(c[2] || ""), this.ua = we(c[3] || "", !0), ue(this, c[4]), this.La = we(c[5] || "", !0), ve(this, c[6] || "", !0), this.Ka = we(c[7] || "")) : (this.ia=!!b, this.a = new xe(null, 0, this.ia))
};
l = se.prototype;
l.ha = "";
l.Na = "";
l.ua = "";
l.Ma = null;
l.La = "";
l.Ka = "";
l.ia=!1;
l.toString = function() {
    var a = [], b = this.ha;
    b && a.push(ye(b, ze, !0), ":");
    if (b = this.ua) {
        a.push("//");
        var c = this.Na;
        c && a.push(ye(c, ze, !0), "@");
        a.push(encodeURIComponent(String(b)).replace(/%25([0-9a-fA-F]{2})/g, "%$1"));
        b = this.Ma;
        null != b && a.push(":", String(b))
    }
    if (b = this.La)
        this.ua && "/" != b.charAt(0) && a.push("/"), a.push(ye(b, "/" == b.charAt(0) ? Ae : Be, !0));
    (b = this.a.toString()) && a.push("?", b);
    (b = this.Ka) && a.push("#", ye(b, Ce));
    return a.join("")
};
l.clone = function() {
    return new se(this)
};
var te = function(a, b, c) {
    a.ha = c ? we(b, !0) : b;
    a.ha && (a.ha = a.ha.replace(/:$/, ""))
}, ue = function(a, b) {
    if (b) {
        b = Number(b);
        if (isNaN(b) || 0 > b)
            throw Error("Bad port number " + b);
        a.Ma = b
    } else 
        a.Ma = null
}, ve = function(a, b, c) {
    b instanceof xe ? (a.a = b, De(a.a, a.ia)) : (c || (b = ye(b, Ee)), a.a = new xe(b, 0, a.ia))
}, we = function(a, b) {
    return a ? b ? decodeURI(a) : decodeURIComponent(a) : ""
}, ye = function(a, b, c) {
    return v(a) ? (a = encodeURI(a).replace(b, Fe), c && (a = a.replace(/%25([0-9a-fA-F]{2})/g, "%$1")), a) : null
}, Fe = function(a) {
    a = a.charCodeAt(0);
    return "%" +
    (a>>4 & 15).toString(16) + (a & 15).toString(16)
}, ze = /[#\/\?@]/g, Be = /[\#\?:]/g, Ae = /[\#\?]/g, Ee = /[\#\?@]/g, Ce = /#/g, xe = function(a, b, c) {
    this.a = a || null;
    this.c=!!c
}, He = function(a) {
    if (!a.K && (a.K = new S, a.$ = 0, a.a))
        for (var b = a.a.split("&"), c = 0; c < b.length; c++) {
            var d = b[c].indexOf("="), e = null, f = null;
            0 <= d ? (e = b[c].substring(0, d), f = b[c].substring(d + 1)) : e = b[c];
            e = decodeURIComponent(e.replace(/\+/g, " "));
            e = Ge(a, e);
            a.add(e, f ? decodeURIComponent(f.replace(/\+/g, " ")) : "")
        }
};
l = xe.prototype;
l.K = null;
l.$ = null;
l.add = function(a, b) {
    He(this);
    this.a = null;
    a = Ge(this, a);
    var c = Xd(this.K, a);
    c || Ud(this.K, a, c = []);
    c.push(b);
    this.$++;
    return this
};
l.remove = function(a) {
    He(this);
    a = Ge(this, a);
    return Wd(this.K.c, a) ? (this.a = null, this.$ -= Xd(this.K, a).length, this.K.remove(a)) : !1
};
l.clear = function() {
    this.K = this.a = null;
    this.$ = 0
};
l.za = function() {
    He(this);
    for (var a = this.K.ya(), b = this.K.za(), c = [], d = 0; d < b.length; d++)
        for (var e = a[d], f = 0; f < e.length; f++)
            c.push(b[d]);
    return c
};
l.ya = function(a) {
    He(this);
    var b = [];
    if (v(a)) {
        var c = a;
        He(this);
        c = Ge(this, c);
        Wd(this.K.c, c) && (b = Ta(b, Xd(this.K, Ge(this, a))))
    } else 
        for (a = this.K.ya(), c = 0; c < a.length; c++)
            b = Ta(b, a[c]);
    return b
};
l.toString = function() {
    if (this.a)
        return this.a;
    if (!this.K)
        return "";
    for (var a = [], b = this.K.za(), c = 0; c < b.length; c++)
        for (var d = b[c], e = encodeURIComponent(String(d)), d = this.ya(d), f = 0; f < d.length; f++) {
            var g = e;
            "" !== d[f] && (g += "=" + encodeURIComponent(String(d[f])));
            a.push(g)
        }
    return this.a = a.join("&")
};
l.clone = function() {
    var a = new xe;
    a.a = this.a;
    this.K && (a.K = this.K.clone(), a.$ = this.$);
    return a
};
var Ge = function(a, b) {
    var c = String(b);
    a.c && (c = c.toLowerCase());
    return c
}, De = function(a, b) {
    b&&!a.c && (He(a), a.a = null, a.K.forEach(function(a, b) {
        var e = b.toLowerCase();
        b != e && (this.remove(b), this.remove(e), 0 < a.length && (this.a = null, Ud(this.K, Ge(this, e), Ua(a)), this.$ += a.length))
    }, a));
    a.c = b
};
var Ie = function(a, b, c) {
    ja.call(this, a, b, "Exit", !1, !0);
    this.c = c;
    this.g = "_blank";
    this.e = null
};
z(Ie, ja);
var Je = function(a) {
    this.e = a.sourceFilename;
    this.a = a.quality;
    this.c = a.mimetype
};
var Ke = "undefined_type", Me = function() {
    return Le || (Le = "200_58_" + Ke)
}, Ne = 0, Le = null;
var Oe = {
    rc: "varName",
    ac: "creativeId",
    Ub: "assetId",
    Zb: "click",
    ec: "clickN",
    Xb: "assets",
    xc: "vcData",
    dc: "exitEvents",
    $b: "googleDiscoveryUrl",
    ic: "adSiteUrl",
    uc: "isGCNAd",
    hc: "td",
    qc: "assetType",
    gc: "isFlashFullScreenEnabled",
    Wb: "br",
    pc: "os",
    Sb: "sn",
    Rb: "sid",
    Ib: "adv",
    Kb: "buy",
    Pb: "pid",
    Jb: "aid",
    Mb: "cid",
    Qb: "rid",
    Ob: "kid",
    Nb: "geo",
    Lb: "randomNumber",
    cc: "dcData",
    lc: "ispushdown",
    fc: "expEnv",
    mc: "layoutsConfig",
    sc: "rv",
    bc: "customMetaData"
}, Pe = function(a) {
    this.a = a
}, Qe = function(a) {
    switch (a) {
    case "BANNER":
        return "Inpage";
    case "FLOATING":
        return "Floating";
    case "EXPANDABLE":
        return "Expanding";
    case "OVERLAY":
        return "Overlay";
    default:
        return ""
    }
}, Ue = function(a, b) {
    var c, d, e = [], f = a.a.v;
    c = [];
    d = [];
    if ("FLASH" == b.k)
        c = ["video/x-flv"], d = ["video/mp4"];
    else {
        var g = document.createElement("VIDEO");
        g && g.canPlayType && (g.canPlayType("video/webm").replace(/no/, "") && d.push("video/webm"), (g.canPlayType("video/mp4").replace(/no/, "") || Ab) && d.push("video/mp4"))
    }
    g = a.a;
    if (!g.o) {
        g.o = {};
        for (var h = 0; h < g.v.length; h++) {
            var k = g.v[h], m = k.name.toLowerCase();
            if (k.k) {
                var k = k.k, r = k.e.toLowerCase(), p = g.o[r];
                p || (p = {}, g.o[r] = p);
                r = p[k.a];
                r || (r = {}, p[k.a] = r);
                r[k.c] = m
            } else 
                g.o[m] || (g.o[m] = {})
            }
    }
    var g = g.o, h = Re(a), s;
    for (s in g) {
        (m = Se(g, s, h, c)) || (m = Se(g, s, h, d));
        k = m || s;
        m = null;
        for (p = 0; p < f.length; p++)
            f[p].name.toLowerCase() === k && (m = f[p]);
        if (m) {
            Te(a, e, s, m);
            for (var Pa in g[s]) {
                var k = g[s][Pa], ua;
                for (ua in k)
                    p = k[ua], g[p] || Te(a, e, p, m)
                }
        }
    }
    return be(b) ? e.join("&") : escape(e.join("&"))
}, Te = function(a, b, c, d) {
    d.e ? (b.push(Ve("PRO_" + c, d.a)), null != d.c && b.push(Ve("STR_" + c, d.c))) : b.push(Ve(c,
    /^https?:/.test(d.a) ? d.a : We(a.a) + d.a))
}, Re = function(a) {
    var b = parseInt;
    a = "?" + (a.a.a.geo || "");
    var c = a.search(re), d;
    i: {
        for (d = 0; 0 <= (d = a.indexOf("bw", d)) && d < c;) {
            var e = a.charCodeAt(d - 1);
            if (38 == e || 63 == e)
                if (e = a.charCodeAt(d + 2), !e || 61 == e || 38 == e || 35 == e)
                    break i;
            d += 3
        }
        d =- 1
    }
    if (0 > d)
        a = null;
    else {
        e = a.indexOf("&", d);
        if (0 > e || e > c)
            e = c;
        d += 3;
        a = decodeURIComponent(a.substr(d, e - d).replace(/\+/g, " "))
    }
    switch (b(a, 10) || 0) {
    case 0:
        return 2;
    case 1:
        return 1;
    case 2:
        return 3;
    default:
        return 5
    }
}, Se = function(a, b, c, d) {
    if (!d.length ||!a[b] ||
    jb(a[b]))
        return null;
    var e = function(c, e) {
        for (; 1 <= c && 5 >= c;) {
            var f = a[b][c];
            if (f)
                for (var m = 0; m < d.length; m++)
                    if (f[d[m]])
                        return f[d[m]];
            c += e
        }
        return null
    }, f = e(c, - 1);
    f || (f = e(c + 1, 1));
    return f
}, Xe = function(a) {
    var b = [];
    a = a.a.Xa;
    for (var c = 0; c < a.length; c++) {
        var d = a[c];
        b.push(escape(me("name:%s,vfp_low:%s,vfp_mid:%s,vfp_high:%s,pfp_low:%s,pfp_mid:%s,pfp_high:%s", escape(d.m), escape(d.k || ""), escape(d.j || ""), escape(d.c || ""), escape(d.e || ""), escape(d.g || ""), escape(d.a || ""))))
    }
    return b.join("{DELIM}")
}, Ye = function(a,
b) {
    var c = [], d = a.a.T, e;
    for (e in d) {
        var f = d[e], g = me("name:%s,url:%s,target:%s", escape(f.name), escape(f.c).replace(/\+/g, "%2B"), f.e ? "popup" : escape(f.g));
        if (/,/.test(f.c)&&!be(b) || b.c && 0 <= G(b.c.c, "2.0.0"))
            g = escape(g);
        c.push(g)
    }
    return c.join("{DELIM}")
}, Ze = function() {
    var a;
    if (a=!/^http/.test(window.location.href)) {
        var b = window.parent, c = window;
        try {
            a = b.document.domain == c.document.domain
        } catch (d) {
            a=!1
        }
    }
    a = new se(a ? window.parent.location.href : window.location.href);
    return a.ha + "://" + a.ua
}, $e = function(a) {
    if (a.a.c) {
        if (!D(F(a.a.X)) &&
        0 != a.a.X.indexOf("%"))
            return a.a.X;
        if ((a = document.getElementsByTagName("base")) && a.length&&!D(F(a[0].href)))
            return a[0].href
    }
    return ke()
}, af = function(a, b) {
    var c = [], d;
    for (d in b)
        c.push(U(a, d, b[d]));
    return c.join("&")
};
Pe.prototype.c = function(a) {
    return encodeURIComponent(a).replace(/[!'()*]/g, Xa(escape))
};
var U = function(a, b, c) {
    return a.c(b) + "=" + a.c(String(c))
}, Ve = function(a, b) {
    return escape(a) + "=" + escape(String(b))
};
var bf = function(a, b) {
    null != a && this.a.apply(this, arguments)
};
bf.prototype.c = "";
bf.prototype.a = function(a, b, c) {
    this.c += a;
    if (null != b)
        for (var d = 1; d < arguments.length; d++)
            this.c += arguments[d];
    return this
};
bf.prototype.clear = function() {
    this.c = ""
};
bf.prototype.toString = function() {
    return this.c
};
var cf = function(a) {
    return yb && (O&&!P("534") || tc&&!wc("5.1") || N&&!P("11")) || zb && "window" == a.toLowerCase()
};
var df = {
    DISPLAY_TIMER: [null, "Timer", !0, !0],
    INTERACTION_TIMER: [null, "Timer", !0, !0],
    INTERACTIVE_IMPRESSION: [null, "Counter", !0, !1],
    MANUAL_CLOSE: [null, "Counter", !0, !0],
    BACKUP_IMAGE_IMPRESSION: [null, "Counter", !0, !0],
    EXPAND_TIMER: [null, "Timer", !0, !1],
    VIDEO_PLAY: [null, "Counter", !0, !1],
    VIDEO_VIEW_TIMER: [null, "Timer", !0, !0],
    VIDEO_COMPLETE: [null, "Counter", !0, !1],
    VIDEO_INTERACTION: [null, "Counter", !0, !0],
    VIDEO_PAUSE: [null, "Counter", !0, !0],
    VIDEO_MUTE: [null, "Counter", !0, !0],
    VIDEO_REPLAY: [null, "Counter", !0, !0],
    VIDEO_MIDPOINT: [null, "Counter", !0, !0],
    VIDEO_STOP: [null, "Counter", !0, !0],
    VIDEO_UNMUTE: [null, "Counter", !0, !0],
    FULL_SCREEN: [null, "Counter", !0, !0],
    DYNAMIC_CREATIVE_IMPRESSION: [null, "Counter", !0, !0],
    HTML5_CREATIVE_IMPRESSION: [null, "Counter", !0, !0]
}, ef = function(a) {
    var b = df[a];
    return b ? new ja(a, b[0], b[1], b[2], b[3]) : null
};
var ff = {
    DISPLAY_TIMER: "DISPLAY_TIMER",
    INTERACTION_TIMER: "INTERACTION_TIMER",
    3: "INTERACTION_TIMER",
    EVENT_USER_INTERACTION: "INTERACTIVE_IMPRESSION",
    EVENT_MANUAL_CLOSE: "MANUAL_CLOSE",
    EXPAND_TIMER: "EXPAND_TIMER",
    EVENT_VIDEO_PLAY: "VIDEO_PLAY",
    EVENT_VIDEO_VIEW_TIMER: "VIDEO_VIEW_TIMER",
    EVENT_VIDEO_COMPLETE: "VIDEO_COMPLETE",
    EVENT_VIDEO_INTERACTION: "VIDEO_INTERACTION",
    EVENT_VIDEO_PAUSE: "VIDEO_PAUSE",
    EVENT_VIDEO_MUTE: "VIDEO_MUTE",
    EVENT_VIDEO_REPLAY: "VIDEO_REPLAY",
    EVENT_VIDEO_MIDPOINT: "VIDEO_MIDPOINT",
    EVENT_VIDEO_STOP: "VIDEO_STOP",
    EVENT_VIDEO_UNMUTE: "VIDEO_UNMUTE",
    EVENT_FULLSCREEN: "FULL_SCREEN"
};
var hf = function(a, b) {
    switch (a) {
    case "logEvent":
        return gf(b);
    case "logVideoEvent":
        return new na("logVideoEvent", b[0], ff[b[2]], b[1]);
    case "logEventFlushCounters":
        return b[0] = "Exit", gf(b, !0);
    case "logExitFlushEventsOpenPopup":
        return new qa("logExitFlushEventsOpenPopup", b[1], b[4], b[5] || null);
    case "reportCustomVariable":
        return new pa(b[0], b[1]);
    case "flushCounters":
        return new B("flushCounters");
    case "conduitInitialized":
        return new C("conduitInitialized");
    case "expandAsset":
        return new C("expandAsset", null,
        1 < b.length ? b[1] : null);
    case "expandRequested":
        return new C("expandRequested", null, 1 < b.length ? b[1] : null);
    case "expandFinished":
    case "collapseAsset":
    case "collapseRequested":
    case "collapseFinished":
    case "tellAssetHide":
        return new C(a);
    case "tellCompanionAssetHide":
        return new C("tellCompanionAssetHide", b[0]);
    case "tellCompanionAssetShow":
        return new C("tellCompanionAssetShow", b[0]);
    case "invokeExternalJSFunction":
        return new ra(b[0]);
    case "setThrottlingWindow":
        return new sa(parseFloat(b[0]));
    case "setDynamicServeId":
        return new ta;
    case "setTimerAdjustment":
        return new va(ff[b[0]], parseFloat(b[1]), parseFloat(b[2]));
    case "registerChargeableEventName":
        return new wa(b[0]);
    case "fullscreenExpandRequested":
        return new C(a, null, b);
    case "isFullscreenSupported":
    case "queryFullscreenDimensions":
    case "fullscreenExpandFinished":
    case "fullscreenCollapseRequested":
    case "fullscreenCollapseFinished":
        return new C(a)
    }
    return null
}, gf = function(a, b) {
    var c = a[0], d = a[1];
    ka(c);
    var e = jf(a[3]);
    return jf(a[4]) ? new oa("logEvent", c, d, e, b) : new ma("logEvent",
    c, ff[d], b)
}, jf = function(a) {
    return null != a ? "true" == a.toString().toLowerCase() : !1
};
var lf = function(a) {
    Q.call(this);
    this.a = this.j = null;
    this.e = new T(this);
    this.n=!1;
    kf || (this.a = new Dd(100), this.e.c(this.a, "tick", this.g), Fd(this.a), this.e.c(document, "DOMContentLoaded", this.g), this.e.c(document, "readystatechange", this.g), this.e.c(window, "load", this.m), this.g(), null != a && ca(a) && Gd(this.m, a, this))
};
z(lf, Q);
var kf=!1;
lf.prototype.g = function() {
    var a;
    if (a = M) {
        a=!1;
        try {
            a = null == window.frameElement
        } catch (b) {}
    }
    a && document.documentElement.doScroll ? a = mf() : "undefined" === document.readyState ? document.getElementsByTagName ? (a = document.getElementsByTagName("*"), a = 0 < a.length ? a[a.length - 1] : null, this.j && a && this.j == a ? a=!0 : (this.j = a, a=!1)) : a=!1 : a = "complete" === document.readyState;
    a && this.m()
};
var mf = function() {
    if (!document.documentElement.doScroll)
        return !1;
    try {
        return document.documentElement.doScroll("left"), !0
    } catch (a) {
        return !1
    }
};
lf.prototype.k = function() {
    nf(this);
    lf.M.k.call(this)
};
var nf = function(a) {
    a.e.O();
    a.a && (a.a.O(), a.a = null)
};
lf.prototype.m = function() {
    this.n || (kf = this.n=!0, nf(this), R(this, new B("ready")))
};
var of = function() {};
of.getInstance = function() {
    return of.a ? of.a : of.a = new of
};
of.prototype.a = 0;
var V = function(a) {
    Q.call(this);
    this.m = a || Pb();
    this.B = null;
    this.fa=!1;
    this.g = null;
    this.o = void 0;
    this.v = this.A = this.p = null
};
z(V, Q);
V.prototype.da = of.getInstance();
V.prototype.n = function() {
    return this.B || (this.B = ":" + (this.da.a++).toString(36))
};
V.prototype.P = function() {
    return this.g
};
var pf = function(a) {
    a.o || (a.o = new T(a));
    return a.o
};
V.prototype.Q = function(a) {
    if (this.p && this.p != a)
        throw Error("Method not supported");
    V.M.Q.call(this, a)
};
V.prototype.I = function() {
    this.g = this.m.a.createElement("div")
};
var qf = function(a, b, c) {
    if (a.fa)
        throw Error("Component already rendered");
    a.g || a.I();
    b ? b.insertBefore(a.g, c || null) : a.m.a.body.appendChild(a.g);
    a.p&&!a.p.fa || a.sa()
};
V.prototype.Ua = function(a) {
    this.g = a
};
V.prototype.sa = function() {
    this.fa=!0;
    rf(this, function(a) {
        !a.fa && a.P() && a.sa()
    })
};
V.prototype.xa = function() {
    rf(this, function(a) {
        a.fa && a.xa()
    });
    this.o && fe(this.o);
    this.fa=!1
};
V.prototype.k = function() {
    this.fa && this.xa();
    this.o && (this.o.O(), delete this.o);
    rf(this, function(a) {
        a.O()
    });
    this.g && Ub(this.g);
    this.p = this.g = this.v = this.A = null;
    V.M.k.call(this)
};
var rf = function(a, b) {
    a.A && H(a.A, b, void 0)
};
V.prototype.removeChild = function(a, b) {
    if (a) {
        var c = v(a) ? a: a.n(), d;
        this.v && c ? (d = this.v, d = (c in d ? d[c] : void 0) || null) : d = null;
        a = d;
        if (c && a) {
            d = this.v;
            c in d && delete d[c];
            I(this.A, a);
            b && (a.xa(), a.g && Ub(a.g));
            c = a;
            if (null == c)
                throw Error("Unable to set parent component");
            c.p = null;
            V.M.Q.call(c, null)
        }
    }
    if (!a)
        throw Error("Child is not in parent component");
    return a
};
var sf = function(a) {
    V.call(this, a)
};
z(sf, V);
sf.prototype.n = function() {
    return this.B || (this.B = Me() + "_" + (Ne++).toString(36))
};
sf.prototype.toString = function() {
    return this.n()
};
var tf = function(a) {
    rf(a, function(a) {
        tf(a)
    })
}, uf = function(a) {
    rf(a, function(a) {
        uf(a)
    })
}, vf = function(a) {
    rf(a, function(a) {
        vf(a)
    })
};
sf.prototype.Ha = function() {
    rf(this, function(a) {
        a.Ha()
    })
};
sf.prototype.va = function(a, b) {
    Ic(this.P(), a, b);
    rf(this, function(a) {
        a.va("100%", "100%")
    })
};
var wf = function(a, b, c) {
    V.call(this, c);
    this.j = a;
    this.D = b;
    if (b = a.a && a.a.a)
        b = a.a.a, b = new K(b.width, b.height);
    this.W = b || new K(a.width, a.height);
    this.J=!1;
    this.L = [];
    this.R = null
};
z(wf, sf);
l = wf.prototype;
l.Eb = function() {
    for (this.J=!0; 0 < this.L.length;) {
        var a = this.L.shift();
        xf(this, a.name, a.value)
    }
};
l.Fb = function() {
    this.J=!1
};
l.Ua = function(a) {
    wf.M.Ua.call(this, a);
    a.style.position = "relative";
    Ic(a, this.W.width, this.W.height);
    "boxSizing"in a.style && (a.style.boxSizing = "content-box");
    var b = this.D.D;
    (null == b ? 0 : b.e && 1 == this.D.k.length) && yf(b, a)
};
l.sa = function() {
    var a = pf(this);
    a.c(this, "conduitInitialized", this.Eb);
    a.c(this, "RESET", this.Fb);
    wf.M.sa.call(this);
    var b = pf(this);
    this.R = new lf(5E3);
    kf ? this.V() : b.c(this.R, "ready", this.V);
    b = this.P();
    a.c(b, "mouseover", this.Db);
    a.c(b, "mouseout", this.lb);
    a.c(this, "logEvent", this.mb);
    a.c(this, "logExitFlushEventsOpenPopup", this.mb)
};
l.Db = function() {
    zf(this, "isMouseOver", "1")
};
l.lb = function() {
    zf(this, "isMouseOver", "0")
};
l.mb = function(a) {
    "Exit" == ka(a.c) && this.lb()
};
var zf = function(a, b, c) {
    a.J ? xf(a, b, c) : a.L.push({
        name: b,
        value: c
    })
};
var W = function(a, b, c) {
    wf.call(this, a, b, c);
    this.H = "FLASH_" + this.n();
    this.Z = a.c.e;
    this.q = null;
    this.S=!(!a.e ||!a.e.m);
    this.e = this.U = this.C = this.a = null;
    b.cb && (a = a.A, a = ("FLASH_CONFIGURABLE_XML" == a || "FLASH_LAYOUTS" == a ? qb : ob)(b.N), b.$a = a)
};
z(W, wf);
W.prototype.I = function() {
    W.M.I.call(this);
    this.Ua(this.P())
};
W.prototype.V = function() {
    zf(this, "mtfContinue", "1")
};
W.prototype.sa = function() {
    W.M.sa.call(this);
    pf(this).c(this, "conduitInitialized", this.ba);
    var a = this.j, b = this.D, c = this.n(), d = {};
    if (d.src = a.q) {
        var e = a.q, e = e.slice(0, e.lastIndexOf("/") + 1);
        /^\/+$/.test(e) || (e = e.replace(/\/+$/, ""));
        d.base = e
    }
    d.width = a.width;
    d.height = a.height;
    d.wmode = a.c.a;
    d.allowScriptAccess = a.c.k;
    b = new Pe(b);
    e = (e = a.c) ? 3 == e.e ? escape : b.c : Wa;
    c = [U(b, "varName", c), U(b, "creativeId", b.a.m), U(b, "assetId", a.id), U(b, "rv", "200_58"), U(b, "assetType", Qe(a.p)), U(b, "layoutsConfig", a.B || b.a.$a || ""), U(b,
    "click", b.a.B), U(b, "clickN", b.a.A), U(b, "assets", Ue(b, a)), U(b, "vcData", Xe(b)), U(b, "exitEvents", Ye(b, a)), U(b, "sn", b.a.a.sn || ""), U(b, "sid", b.a.a.sid || ""), U(b, "adv", b.a.a.adv || ""), U(b, "buy", b.a.a.buy || ""), U(b, "pid", b.a.a.pid || ""), U(b, "aid", b.a.a.aid || ""), U(b, "cid", b.a.a.cid || ""), U(b, "rid", b.a.a.rid || ""), b.a.a.geo || "", U(b, "br", N ? "ff" : sc ? "cr" : tc ? "sf" : M ? "ie" : "ot"), U(b, "isFlashFullScreenEnabled", Ac(a.c && a.c.a || "transparent")), U(b, "td", Ze()), U(b, "adSiteUrl", $e(b)), U(b, "googleDiscoveryUrl", b.a.c ? "http://pagead2.googlesyndication.com/pagead/ads?client=dclk-3pas-query&output=xml&geo=true&adtest=on" :
    "http://pagead2.googlesyndication.com/pagead/ads?client=dclk-3pas-query&output=xml&geo=true"), U(b, "dcData", e(String(b.a.N))), U(b, "customMetaData", af(b, Af(b.a))), U(b, "ispushdown", !(!a.a ||!a.a.c)), U(b, "expEnv", ("gdn" == je() ? "adsense" : "basic").toString()), af(b, Af(b.a))].join("&");
    d.flashvars = c;
    a.c && cf(a.c.a) && (d.flashvars += "&scaleMode=noScale");
    d.id = d.name = this.H;
    "window" == d.wmode && Tb(this.m.a).IN_ADSENSE_IFRAME && (d.wmode = "opaque");
    if (a = this.j.a && this.j.a.a)
        this.C = null, cf(d.wmode) && (a = new Bc(a.top, a.left +
        a.width, a.top + a.height, a.left), c = "", 0 == a.top ? c += "T" : a.bottom == this.j.height && (c += "B"), 0 == a.left ? c += "L" : a.right == this.j.width && (c += "R"), this.C = 1 >= c.length ? null : c), this.C && (d.scale = "noscale", d.salign = this.C);
    var a = this.P(), c = {}, b = {}, e=!1, f;
    for (f in d) {
        var g = d[f];
        switch (f.toLowerCase()) {
        case "codebase":
        case "pluginspage":
        case "type":
        case "classid":
        case "minversion":
            break;
        case "src":
        case "movie":
            M ? b.movie = g : c.data = g;
            break;
        case "querystring":
        case "flashvars":
            M ? b.FlashVars = g : c.FlashVars = g;
            break;
        case "width":
        case "height":
        case "align":
        case "vspace":
        case "hspace":
        case "class":
        case "title":
        case "accesskey":
        case "name":
        case "id":
        case "tabindex":
        case "alt":
        case "onmouseover":
        case "onmouseout":
        case "swliveconnect":
            c[f] =
            g;
            break;
        case "allowscriptaccess":
        case "base":
            M ? b[f] = g : c[f] = g;
            break;
        case "wmode":
            e = Ac(g);
            M ? b[f] = g : c[f] = g;
            break;
        default:
            b[f] = g
        }
    }
    e && (M ? (b.allowFullScreenInteractive = "true", b.allowFullScreen = "true") : (c.allowFullScreenInteractive = "true", c.allowFullScreen = "true"));
    d = new bf;
    d.a("<object");
    M&&!P(11) ? d.a(' classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000"') : d.a(' type="application/x-shockwave-flash"');
    for (var h in c)
        d.a(" "), d.a(h), d.a('="'), d.a(c[h]), d.a('"');
    d.a(">");
    for (var k in b)
        d.a('<param name="'),
        d.a(k), d.a('" value="'), d.a(b[k]), d.a('"/>');
    d.a("</object>");
    a.innerHTML = d.toString();
    this.e = null;
    this.a = a ? a.firstChild : null;
    this.a.style.outline = "none";
    this.a.style.position = "relative";
    this.a.style.zIndex = 10;
    d = this.H;
    h = this.m;
    k = x(this.aa, this);
    f = d + "_DoFSCommand";
    Tb(h.a)[f] = k;
    k = null;
    M && (k = h.a.createElement("script"), k.event = "FSCommand(command,args)", k.htmlFor = d, k.text = f + "(command, args)", d = null, f = h.a, f.querySelectorAll && f.querySelector ? f = f.querySelectorAll("HEAD") : f = f.getElementsByTagName("HEAD"),
    f && 0 != f.length ? d = f[0] : d = h.a.documentElement, h.appendChild(d, k));
    this.U = k
};
W.prototype.ba = function() {
    Bf(this);
    if (this.S || tc && wc("6.1"))
        this.q = new Dd(300), pf(this).c(this.q, "tick", this.ca), Fd(this.q);
    this.a && (this.a.style.opacity = 1)
};
var Bf = function(a) {
    a.q && (a.q.O(), a.q = null)
};
W.prototype.ca = function() {
    try {
        100 == this.a.PercentLoaded()&&!this.a.IsPlaying() && this.S && (R(this, new C("tellAssetHide")), Bf(this))
    } catch (a) {
        t:
        {
            if (Bf(this), R(this, new C("RESET")), !this.e) {
                var b = this.D.H;
                if (!b ||!b.a)
                    break t;
                this.e = this.m.a.createElement("IMG");
                this.e.style.position = "absolute";
                this.e.style.zIndex = 1;
                var c = this.j.a && this.j.a.a || new Cc(0, 0, this.j.width, this.j.height);
                this.e.style.width = c.width;
                this.e.style.height = c.height;
                this.e.style.left = c.left;
                this.e.style.top = c.top;
                this.e.src = b.a;
                this.P().appendChild(this.e)
            }
            this.a &&
            (this.a.style.opacity = 0)
        }
    }
};
W.prototype.xa = function() {
    Bf(this);
    var a = this.m, b = this.U, c = Tb(a.a), d = this.H + "_DoFSCommand";
    if (c)
        try {
            delete c[d]
    } catch (e) {
        c[d] = null
    }
    b && a.c(b);
    this.e && (this.m.c(this.e), this.e = null);
    if (M) {
        this.a.style.display = "none";
        for (var f in this.a)
            w(this.a[f]) && (this.a[f] = null)
    }
    this.m.c(this.a);
    this.a = null;
    this.P().innerHTML = "";
    W.M.xa.call(this)
};
W.prototype.va = function(a, b) {
    W.M.va.call(this, a, b);
    Ic(this.a, "100%", "100%")
};
W.prototype.aa = function(a, b) {
    if ("mtfCommands" == a)
        for (var c = b.split("&"), d = 0; d < c.length; d++) {
            for (var e = c[d].split("="), f = unescape(e[0]), e = unescape(e[1]).split("#mtf#"), g = 0; g < e.length; g++) {
                var h = unescape(e[g] || "");
                e[g] = "" == za(h) ? null : h
            }(f = hf(f, e)) && R(this, f)
        }
};
var xf = function(a, b, c) {
    try {
        3 > a.Z ? a.a.SetVariable("_root." + b, c) : a.a.changeInputVariable(b, c)
    } catch (d) {}
};
W.prototype.Ha = function() {
    W.M.Ha.call(this);
    zf(this, "mde_expandedState", "collapsed")
};
var Cf = function(a, b, c) {
    a.addEventListener ? a.addEventListener(b, c, !1) : a.attachEvent && a.attachEvent("on" + b, c)
};
var Df = document;
var Ef = null;
var Ff = function(a) {
    this.a=!!a
};
Ff.prototype.a=!1;
var Gf = function(a, b) {
    if (!a.a&&!D(F(b))) {
        window.studioV2_image_requests || (window.studioV2_image_requests = []);
        var c = document.createElement("img");
        c.src = b;
        window.studioV2_image_requests.push(c)
    }
};
var Hf = function(a, b) {
    this.name = a.name;
    this.a = a.url;
    b && (this.a = this.a.replace(/^http:\/\//, "https://"));
    this.e = a.isVideo ||!1;
    this.c = a.streamingUrl || null;
    this.k = a.transcodeInformation ? new Je(a.transcodeInformation) : null
};
var If = function() {};
ha("studio.common.Environment", If);
var Jf = {
    LIVE: 1,
    LOCAL: 2,
    BROWSER: 4,
    IN_APP: 8,
    LAYOUTS_PREVIEW: 16
};
If.Type = Jf;
If.hasType = function(a) {
    return (0 & a) == a
};
var Kf = function(a) {
    return (a = a.match(/([^:]+):([^:]*):([^:]*):(.+)/)) && 5 == a.length ? {
        type: a[1],
        name: unescape(a[2]),
        videoName: unescape(a[3]),
        trigger: a[4]
    } : null
}; /*
 Portions of this code are from MochiKit, received by
 The Closure Authors under the MIT license. All other code is Copyright
 2005-2009 The Closure Authors. All Rights Reserved.
*/
var Lf = function(a, b) {
    this.k = [];
    this.G = b || null;
    this.a = this.c=!1;
    this.e = void 0;
    this.o = this.p = this.j=!1;
    this.g = 0;
    this.m = null;
    this.q = 0
};
Lf.prototype.n = function(a, b) {
    this.j=!1;
    Mf(this, a, b)
};
var Mf = function(a, b, c) {
    a.c=!0;
    a.e = c;
    a.a=!b;
    Nf(a)
}, Pf = function(a) {
    if (a.c) {
        if (!a.o)
            throw new Of;
        a.o=!1
    }
}, Qf = function(a, b, c) {
    a.k.push([b, c, void 0]);
    a.c && Nf(a)
};
Lf.prototype.then = function(a, b, c) {
    var d, e, f = new wd(function(a, b) {
        d = a;
        e = b
    });
    Qf(this, d, function(a) {
        e(a)
    });
    return f.then(a, b, c)
};
td(Lf);
var Rf = function(a) {
    return Qa(a.k, function(a) {
        return w(a[1])
    })
}, Nf = function(a) {
    if (a.g && a.c && Rf(a)) {
        var b = a.g, c = Sf[b];
        c && (n.clearTimeout(c.Da), delete Sf[b]);
        a.g = 0
    }
    a.m && (a.m.q--, delete a.m);
    for (var b = a.e, d = c=!1; a.k.length&&!a.j;) {
        var e = a.k.shift(), f = e[0], g = e[1], e = e[2];
        if (f = a.a ? g : f)
            try {
                var h = f.call(e || a.G, b);
                q(h) && (a.a = a.a && (h == b || h instanceof Error), a.e = b = h);
                ud(b) && (d=!0, a.j=!0)
            } catch (k) {
            b = k, a.a=!0, Rf(a) || (c=!0)
        }
    }
    a.e = b;
    d && (h = x(a.n, a, !0), d = x(a.n, a, !1), b instanceof Lf ? (Qf(b, h, d), b.p=!0) : b.then(h, d));
    c &&
    (b = new Tf(b), Sf[b.Da] = b, a.g = b.Da)
}, Of = function() {
    xa.call(this)
};
z(Of, xa);
Of.prototype.message = "Deferred has already fired";
Of.prototype.name = "AlreadyCalledError";
var Tf = function(a) {
    this.Da = n.setTimeout(x(this.c, this), 0);
    this.a = a
};
Tf.prototype.c = function() {
    delete Sf[this.Da];
    throw this.a;
};
var Sf = {};
var Xf = function(a) {
    var b = {}, c = b.document || document, d = document.createElement("SCRIPT"), e = {
        a: d,
        Cb: void 0
    }, f = new Lf(0, e), g = null, h = null != b.timeout ? b.timeout: 5E3;
    0 < h && (g = window.setTimeout(function() {
        Uf(d, !0);
        var b = new Vf(1, "Timeout reached for loading script " + a);
        Pf(f);
        Mf(f, !1, b)
    }, h), e.Cb = g);
    d.onload = d.onreadystatechange = function() {
        d.readyState && "loaded" != d.readyState && "complete" != d.readyState || (Uf(d, b.Gb ||!1, g), Pf(f), Mf(f, !0, null))
    };
    d.onerror = function() {
        Uf(d, !0, g);
        var b = new Vf(0, "Error while loading script " +
        a);
        Pf(f);
        Mf(f, !1, b)
    };
    Sb(d, {
        type: "text/javascript",
        charset: "UTF-8",
        src: a
    });
    Wf(c).appendChild(d)
}, Wf = function(a) {
    var b = a.getElementsByTagName("HEAD");
    return b && 0 != b.length ? b[0] : a.documentElement
}, Uf = function(a, b, c) {
    null != c && n.clearTimeout(c);
    a.onload = aa;
    a.onerror = aa;
    a.onreadystatechange = aa;
    b && window.setTimeout(function() {
        Ub(a)
    }, 0)
}, Vf = function(a, b) {
    var c = "Jsloader error (code #" + a + ")";
    b && (c += ": " + b);
    xa.call(this, c)
};
z(Vf, xa);
var Yf = function(a, b) {
    this.a = a || "";
    this.e=!D(this.a) && 8 != b;
    this.c=!1
}, yf = function(a, b) {
    b.setAttribute("id", "DfaVisibilityIdentifier_" + y());
    b.className = "GoogleActiveViewClass";
    b._avi_ = a.a;
    a.c || (Xf("//pagead2.googlesyndication.com/pagead/js/lidar.js"), a.c=!0)
};
var Zf = function(a, b, c) {
    this.a = a;
    this.c = b;
    this.e=!!c
}, $f = {
    Yb: "CLICK",
    kc: "IMPRESSION_JS",
    jc: "IMPRESSION_IMG",
    Tb: "ARTWORK_IMPRESSION"
}, ag = function(a) {
    if (a.e) {
        var b = a.a;
        if (!a.c.a&&!D(F(b)) && null !== b && (Df.body ? (Ef || (a = Df.createElement("iframe"), a.style.display = "none", a.id = "anonIframe", Ef = a, Df.body.appendChild(a)), a=!0) : a=!1, a)) {
            a = Ef.contentWindow;
            a.google_image_requests || (a.google_image_requests = []);
            var c = a.document.createElement("img");
            c.src = b;
            a.google_image_requests.push(c)
        }
    } else 
        Gf(a.c, a.a)
};
var bg = function(a) {
    var b;
    b = a.toLowerCase();
    a = [];
    var c = 0;
    /^https?:\/\//.test(b) ? (a[0] = 0, c = "s" == b.charAt(4) ? 8 : 7) : a[0] =- 1;
    a[1] = c;
    b = [b.indexOf(":", c), b.indexOf("/", c), b.indexOf("?", c), b.indexOf("#", c)];
    for (c = 0; c < b.length; ++c)
        if ( - 1 == b[c])
            a[c + 2] =- 1;
        else {
            for (var d=!0, e = c + 1; e < b.length; ++e) 
                - 1 < b[e] && b[c] > b[e] && (d=!1);
                a[c + 2] = d ? b[c] : - 1
        }
    this.a = a
};
var cg = function() {
    this.oa=!1;
    this.m = this.e = this.Ra = "";
    this.p = new K(0, 0);
    this.Oa=!1;
    this.N = "";
    this.Ya = this.q=!1;
    this.G = new K(0, 0);
    this.c=!1;
    this.S = this.J = "";
    this.R = [];
    this.Wa=!1;
    this.X = "";
    this.a = {};
    this.j = "";
    this.qa=!1;
    this.ma = this.la = this.ba = "";
    this.pa = this.F=!1;
    this.V = 4;
    this.Pa = this.$a = null;
    this.ra = "";
    this.C = {
        Aa: "",
        Ba: ""
    };
    this.g = {};
    this.B = this.W = this.Q = "";
    this.A = 1;
    this.Qa = this.Za = this.U = "";
    this.da = null;
    this.na = "";
    this.ab=!1;
    this.k = [];
    this.v = [];
    this.Xa = [];
    this.wa = {};
    this.Sa = {};
    this.T = {};
    this.ka = {};
    this.aa = {};
    this.n = {};
    this.I=!1;
    this.Z = {};
    this.D = this.H = this.L = this.ja = this.ca = null;
    this.Z.Exit = this.T;
    this.Z.Timer = this.ka;
    this.Z.Counter = this.aa
}, dg = 0;
cg.prototype.hb=!1;
var eg = function(a, b) {
    if (D(F(a)))
        return "";
    var c = new bg(a), d;
    d = c.a[3];
    if ( - 1 == d)
        return a;
    t: {
        for (var e = 4; e < c.a.length; ++e)
            if ( - 1 != c.a[e]) {
                c = c.a[e];
                break t
            }
        c =- 1
    }
    e =- 1 == c ? a.substring(d) : a.substring(d, c);
    if ( - 1 == e.toLowerCase().indexOf(";" + b.toLowerCase() + "="))
        e += (";" == e.charAt(e.length) ? "" : ";") + b + "=1";
    else 
        var f = "", g = e.toLowerCase().indexOf(";" + b.toLowerCase() + "="), h = e.indexOf("=", g), f = e.indexOf(";", g + 1), f =- 1 == f ? e.substring(h + 1) : e.substring(h + 1, f), e = e.replace(e.substring(g, h + 1) + f, ";" + b + "=1");
    return a.substring(0,
    d) + e + ( - 1 == c ? "" : a.substring(c))
};
cg.prototype.toString = function() {
    return "Creative_" + this.m
};
cg.prototype.cb = function() {
    return this.Oa
};
var Af = function(a) {
    var b = {}, c;
    for (c in a.a)
        c in Oe || (b[c] = a.a[c]);
    return b
}, We = function(a) {
    return "https:" == window.location.protocol ? a.C.Ba : a.C.Aa
}, fg = function(a, b) {
    if (q(b))
        return b in a.g ? Ua(a.g[b]) : [];
    var c = [], d;
    for (d in a.g)
        c = Ta(c, a.g[d]);
    return c
}, gg = function(a, b) {
    "HTML5" == b.k && (a.hb=!0);
    var c = b.e ? b.e.g: null;
    c && (a.wa[c] = b);
    a.k.push(b)
}, hg = function(a, b) {
    var c = [];
    H(a.k, function(a) {
        a.k == b && c.push(a)
    });
    return c
}, kg = function(a, b, c) {
    var d = ig(a, b);
    d && (d.k = c, a.Sa[b] = d, jg(a, d))
}, mg = function(a, b) {
    var c =
    a.I;
    lg(a, "EXPAND_TIMER", b);
    a.I = c
}, lg = function(a, b, c) {
    return (b = a.n[ng(b, "Counter") || ""] || a.n[ng(b, "Timer") || ""] || ig(a, b)) ? (b.chargeable = c, a.I=!0) : !1
}, jg = function(a, b) {
    if (b) {
        var c;
        (c = b.j ? og(a, b.name) : b.a ? pg(a, b.a.Ia, b.a.jb) : ng(b.name, b.type)) && (a.n[c] = b)
    }
}, qg = function(a, b) {
    var c = b.c, c = c.replace(/%eaid!/g, a.a.aid || "").replace(/%ebuy!/g, a.a.buy || "").replace(/%epid!/g, a.a.pid || "").replace(/%esid!/g, a.a.sid || "").replace(/%erid!/g, a.a.rid || "").replace(/%ecid!/g, a.a.cid || "").replace(/%ekid!/g, a.a.kid || "").replace(/%eadv!/g,
    a.a.adv || "").replace(/%erv!/g, a.J).replace(/%s/g, a.a.sn || "").replace(/%g/g, a.a.geo || "").replace(/%n/g, a.a.randomNumber || ""), d;
    for (d in a.da)
        var e = a.da[d], f = "(%pKEY=!;|%%PATTERN:KEY%%)".replace(/KEY/g, d), c = c.replace(new RegExp(f, "g"), e);
    b.c = c;
    c = a.Qa;
    D(F(c)) || (b.c = qe(b.c, c));
    a.T[b.name] = b;
    jg(a, b)
}, og = function(a, b) {
    var c = ig(a, b);
    return c ? [b, escape(""), escape(""), c.type].join(":") : null
}, ng = function(a, b) {
    return ["CUSTOM_EVENT", escape(a), escape(""), b].join(":")
}, pg = function(a, b, c) {
    return (a = ig(a, b)) && [b,
    escape(""), escape(c), a.type].join(":")
}, sg = function(a, b) {
    if (!b)
        return null;
    if ("CUSTOM_EVENT" == b.type || b.videoName) {
        if ("CUSTOM_EVENT" == b.type&&!b.videoName) {
            var c = b.trigger;
            if (a.Z[c])
                var c = a.Z[c], d = b.name, c = c[d] && c[d].k || null;
            else 
                c = null;
            return c
        }
        if ("CUSTOM_EVENT" != b.type && b.videoName) {
            t:
            switch (c = b.videoName, d = ig(a, b.type), d.type) {
            case "Timer":
                c = rg(a.ka, d, c);
                c = null != c ? c.k : null;
                break t;
            case "Counter":
                c = rg(a.aa, d, c);
                c = null != c ? c.k : null;
                break t;
            default:
                c = null
            }
            return c
        }
    } else 
        return (c = ig(a, b.type)) ? c.k :
        null;
    return null
}, ig = function(a, b) {
    return a.Sa[b] || (a.Sa[b] = ef(b))
}, rg = function(a, b, c) {
    for (var d in a) {
        var e = a[d];
        if (e.a && e.a.jb == c && e.a.Ia == b.name)
            return e
    }
    return null
}, tg = function(a) {
    for (var b = 0; b < a.k.length; b++)
        if (a.k[b].o)
            return !0;
    return !1
};
var vg = function(a) {
    this.q = a.eventReportingUrl || "";
    this.L = a.clickUrl || "";
    this.o = a.clickUrlTimesToEscape || 1;
    this.e = a.clickEventTagUrl || "";
    this.T = a.impressionUrl || "";
    this.A = a.geoData || "";
    this.I = a.siteName || "";
    this.H = a.siteId || "";
    this.j = a.adId || "";
    this.n = a.buyId || "";
    this.p = a.creativeId || "";
    this.N = a.placementId || "";
    this.m = a.advertiserId || "";
    this.B = a.keyValueOrdinal || "";
    this.D = a.renderingVersion || "";
    this.C = a.renderingId || "";
    this.F = a.randomNumber || "";
    this.J = a.stringReportingUrl || "";
    this.c = a.bookingTimeMetaData ||
    {};
    this.g = new ug(a.tag || {});
    this.R = a.urlToGetKeywordsFor || "";
    this.G = a.exitSuffix || "";
    this.Q = a.dynamicData || "";
    this.v = a.exitUrlPatternMacroValues;
    this.k = Jf[a.renderingEnvironment || "BROWSER"];
    this.a = a.placementDimensions || {
        w: "0",
        h: "0"
    };
    this.S = a.swiffyRuntimeUrl || ""
}, ug = function(a) {
    this.k = a.adContainerElementId;
    this.q = a.hideObjects;
    this.c = a.preferHtml5Artwork;
    this.g = a.adSenseKeywords;
    this.j = a.adSenseLatitude;
    this.m = a.adSenseLongitude;
    this.n = a.publisherSideFilePath;
    this.e = a.runtimeMetaData || {};
    this.expansionMode =
    Jd(a.expansionMode);
    this.o=!!a.renderFloatInplace;
    this.p=!!a.tryToWriteHtmlInline;
    this.a = {};
    this.a.top = new wg(t("multiFloat.top.duration", a), t("multiFloat.top.wmode", a), t("multiFloat.top.position", a) ? t("multiFloat.top.position", a).split(",") : []);
    this.a.right = new wg(t("multiFloat.right.duration", a), t("multiFloat.right.wmode", a), t("multiFloat.right.position", a) ? t("multiFloat.right.position", a).split(",") : []);
    this.a.bottom = new wg(t("multiFloat.bottom.duration", a), t("multiFloat.bottom.wmode", a), t("multiFloat.bottom.position",
    a) ? t("multiFloat.bottom.position", a).split(",") : []);
    this.a.left = new wg(t("multiFloat.left.duration", a), t("multiFloat.left.wmode", a), t("multiFloat.left.position", a) ? t("multiFloat.left.position", a).split(",") : [])
}, wg = function(a, b, c) {
    this.top = 1 < c.length ? c[0] : "";
    this.left = 1 < c.length ? c[1] : ""
};
var xg = function(a) {
    this.e = a.exitUrl;
    this.target = a.target;
    this.a = a.imageUrl;
    this.width = a.width;
    this.height = a.height;
    this.c = a.backupDisplayActivityUrl;
    this.k = a.thirdPartyBackupImpressionUrl
};
var yg = function(a) {
    this.id = a.id;
    this.D = a.uniqueId;
    this.c = new vg(a.adServerData || {});
    this.m = new Yf(a.adServerData && a.adServerData.activeViewClkStr, this.c.k);
    this.N = a.isPreviewEnvironment;
    this.a = a.thirdPartyImpressionUrls || [];
    this.j = a.thirdPartyArtworkImpressionUrl;
    this.G = a.hasFlashAsset;
    this.k = a.hasHtmlAsset;
    this.Ca = a.requiresCss3Animations;
    this.flashVersion = a.flashVersion;
    this.A = a.httpMediaServer || "";
    this.B = a.httpsMediaServer || "";
    this.q = a.csiStart;
    this.o = a.csiAdRespTime;
    this.p = a.csiEvents;
    this.e =
    a.dimensions;
    this.F = a.preloaderUrl;
    this.v = a.hasModernizrFeatureChecks;
    this.g = a.html5FeatureChecks;
    this.n = a.backupImage && new xg(a.backupImage) || null;
    this.C = a.hasSwiffyHtmlAsset
};
var zg = function(a, b, c, d) {
    this.a = d;
    d=!1;
    this.a && this.a.Ia && (d = ef(this.a.Ia).m);
    ja.call(this, a, b, c, !1, d)
};
z(zg, ja);
var Ag = function(a, b, c, d, e) {
    this.e = a;
    this.c = b;
    this.g = c ? c.replace(/^.*\/\//, "") : null;
    this.j = d;
    this.k = {};
    this.n = Math.floor(1E3 * Math.random());
    this.m = e
}, Bg = function(a, b, c, d) {
    a.k[b] = {
        time: d || y(),
        Ta: c
    }
}, Dg = function(a) {
    if (a.a()) {
        var b = Cg(a, a.k);
        Gf(a.m, b)
    }
}, Cg = function(a, b) {
    var c = "&adi=" + (a.g ? "csd_" + a.g + "," : "") + "gt_" + Me() + (0 <= a.c && 1E5 > a.c ? "&srt=" + a.c : ""), d = (self.location.protocol && "https:" == self.location.protocol.toString().toLowerCase() ? "https://s0.2mdn.net" : "http://s0.2mdn.net") + "/csi?v=2&s=rmad", e = [],
    f = [], g = [], h;
    for (h in b) {
        var k = b[h];
        if (k.Ta) {
            var m = b[k.Ta];
            m && (m = k.time - m.time, f.push(h + "." + (0 < m ? m : 0)), k = k.time - a.e, g.push(0 < k ? k : 0))
        } else 
            k = k.time - a.e, e.push(h + "." + (0 < k ? k : 0))
    }
    e = "" + ((e.length ? "&rt=" + e.join(",") : "") + (f.length ? "&it=" + f.join(",") : "") + (g.length ? "&irt=" + g.join(",") : ""));
    return d + e + c
};
Ag.prototype.a = function() {
    return 0 == this.n&&!this.j
};
var Eg = function(a, b, c, d, e, f) {
    Ag.call(this, a, b, c, d, e);
    this.o = f
};
z(Eg, Ag);
var Fg = function(a, b) {
    b[a] = {
        time: y(),
        Ta: void 0
    };
    return b
}, Gg = function(a, b, c) {
    q(b) ? a.a() && (b = Cg(a, b), D(F(c)) || (b += "&e=" + c + "," + Me()), Gf(a.m, b)) : Dg(a)
};
Eg.prototype.a = function() {
    return !this.j && this.o
};
var Hg = function(a) {
    switch (a) {
    case 0:
        return "mtf_nd_ru";
    case 1:
        return "mtf_nd_rbu";
    case 2:
        return "mtf_nd_rmb";
    case 3:
        return "mtf_nd_rfv";
    case 4:
        return "mtf_nd_rnf";
    default:
        throw Error("Unknown noDisplayReason");
    }
};
var Ig = function(a) {
    return a.gb && xc(a.flashVersion) || a.Bb && zc(!!a.html5Features, a.html5Features, a.Ca)?!0 : !1
}, Jg = function(a, b, c, d, e, f) {
    if (b) {
        c = Qb(document, c);
        var g = [], g = g.concat(['<a target="', b.target, '" href="', b.e, '">']), g = g.concat(['<img src="', b.a, '" ']), g = g.concat(['width="', b.width, '" height="', b.height, '" ']);
        g.push('border="0"></a>');
        c.innerHTML = g.join("");
        d = new Ff(d);
        Gf(d, b.c);
        Gf(d, b.k);
        for (b = 0; b < e.length; b++)
            Gf(d, e[b]);
        if (e = window.DARTDebugEventHandler) {
            e = new e;
            try {
                e.handleEventActivity("BACKUP_IMAGE_IMPRESSION",
                "Counter", 1, 0, !1, a)
            } catch (h) {}
        }
        f.e && yf(f, c)
    }
};
var Kg = function(a) {
    this.m = a.reportingIdentifier;
    this.k = a.lowBandwidthVideo || null;
    this.j = a.mediumBandwidthVideo || null;
    this.c = a.highBandwidthVideo || null;
    this.e = a.lowBandwidthFallbackVideo || null;
    this.g = a.mediumBandwidthFallbackVideo || null;
    this.a = a.highBandwidthFallbackVideo || null
};
var Lg = function(a, b) {
    this.jb = a;
    this.Ia = b
}, Ng = function(a) {
    var b = a.split(":");
    if (2 == b.length && (a = b[0], b = Mg[b[1]]))
        return new Lg(a, b)
}, Mg = {
    Complete: "VIDEO_COMPLETE",
    Interaction: "VIDEO_INTERACTION",
    MidPoint: "VIDEO_MIDPOINT",
    Mute: "VIDEO_MUTE",
    Pause: "VIDEO_PAUSE",
    Play: "VIDEO_PLAY",
    Replay: "VIDEO_REPLAY",
    Stop: "VIDEO_STOP",
    Unmute: "VIDEO_UNMUTE",
    ViewTime: "VIDEO_VIEW_TIMER"
};
var Og = null, Pg = null, Rg = function(a, b) {
    var c = new cg, d = new yg(a.creativeDto);
    if (!c.oa) {
        c.oa=!0;
        var e = d.c, f = e.g;
        c.e = d.id || c.e;
        c.m = d.D || c.e || c.m;
        c.p = d.e ? new K(parseInt(d.e.width, 10) || c.p.width, parseInt(d.e.height, 10) || c.p.height) : c.p;
        c.N = e.Q || c.N;
        c.Ya = d.C || c.Ya;
        c.G = e.a ? new K(parseInt(e.a.w, 10) || c.G.width, parseInt(e.a.h, 10) || c.G.height) : c.G;
        c.c = d.N || c.c;
        c.J = e.D || c.J;
        c.S = d.flashVersion || c.S;
        c.R = d.g || c.R;
        c.ob = d.Ca || c.ob;
        c.X = e.R || c.X;
        for (var g in e.c)
            c.a[g] = e.c[g];
        c.a.sn = e.I;
        c.a.sid = e.H;
        c.a.aid = e.j;
        c.a.cid =
        e.p;
        c.a.buy = e.n;
        c.a.pid = e.N;
        c.a.adv = e.m;
        c.a.kid = e.B;
        c.a.rid = e.C;
        c.a.geo = e.A;
        c.a.randomNumber = e.F;
        for (var h in f.e)
            c.a[h] = f.e[h];
        c.j = f.k || c.j;
        c.qa = f.c || c.qa;
        c.ba = f.g || c.ba;
        c.la = f.j || c.la;
        c.ma = f.m || c.ma;
        c.F = f.o || c.F;
        c.pa = f.p || c.pa;
        c.V = e.k || c.V;
        c.ra = d.F || c.ra;
        c.C = {
            Aa: d.A || c.C.Aa,
            Ba: d.B || c.C.Ba
        };
        g = new Ff(c.c);
        D(F(e.e)) || (c.g.CLICK = [new Zf(e.e, g)]);
        if (u(d.a) && 0 < d.a.length)
            for (c.g.IMPRESSION_IMG = [], h = 0; h < d.a.length; ++h)
                c.g.IMPRESSION_IMG.push(new Zf(d.a[h], g));
        D(F(d.j)) || (c.g.ARTWORK_IMPRESSION = [new Zf(d.j,
        g)]);
        c.Q = e.q || c.Q;
        c.W = e.J || c.W;
        c.B = e.L || c.B;
        c.A = parseInt(e.o, 10) || c.A;
        c.U = e.T || c.U;
        c.Za = e.S || c.Za;
        c.Qa = e.G || c.Qa;
        c.da = e.v || c.da;
        c.na = f.n || c.na;
        c.ca = d.q || c.ca;
        c.ja = d.o || c.ja;
        c.L = d.p || c.L;
        c.H = d.n || c.H;
        c.D = d.m || c.D
    }
    c.Pa = b.customScriptUrl;
    c.Oa = b.isDynamic;
    c.Wa = b.delayedImpression;
    var e = b.standardEventIds, k;
    for (k in e)
        kg(c, k, e[k]);
    e = b.exitEvents;
    for (k = 0; k < e.length; k++)
        f = e[k], g = new Ie(f.name, f.reportingId, f.url), g.g = f.targetWindow, g.e = f.windowProperties, qg(c, g);
    f = b.timerEvents;
    for (k = 0; k < f.length; k++)
        e =
        f[k], g = c, e = Qg(e, "Timer"), g.ka[e.name] = e, jg(g, e);
    f = b.counterEvents;
    for (k = 0; k < f.length; k++)
        e = f[k], g = c, e = Qg(e, "Counter"), g.aa[e.name] = e, jg(g, e);
    e = "https:" == window.location.protocol;
    f = b.childFiles;
    for (k = 0; k < f.length; k++)
        g = c, h = new Hf(f[k], e), g.v.push(h), g.o = null;
    f = b.videoFiles;
    for (k = 0; k < f.length; k++)
        g = c, h = new Hf(f[k], e), g.v.push(h), g.o = null;
    e = b.videoEntries;
    for (k = 0; k < e.length; k++)
        c.Xa.push(new Kg(e[k]));
    k=!1;
    e = xc(d.flashVersion);
    f = zc(d.v, d.g, d.Ca);
    g = d.c.g.c || tc && wc("6.1");
    d=!d.G ||!e || g && d.k && f ? d.k && f ?
    "HTML5" : null : "FLASH";
    e = b.primaryAssets;
    for (g = f = 0; g < e.length; ++g) {
        e[g].floatingDisplayTypeData&&!D(F(e[g].floatingDisplayTypeData.alignment)) && f++;
        var m = e[g], r = a.creativeDto.adServerData.tag;
        h = new Zd(m.id, m.artworkType, m.displayType, new K(parseInt(m.width, 10) || 0, parseInt(m.height, 10) || 0), (We(c) || "") + m.servingPath);
        h.G = m.zIndex;
        h.n = m.customCss;
        h.j = m.location || h.j;
        h.B = m.layoutsConfig;
        h.A = m.layoutsApi;
        var p = m.flashArtworkTypeData;
        p && (h.c = new tb(p.actionscriptVersion, p.wmode, D(F(p.sdkVersion)) ? "0.0.0" :
        p.sdkVersion, p.allowScriptAccess));
        m.htmlArtworkTypeData && (h.v = new Md);
        if (p = m.floatingDisplayTypeData) {
            var s = p.position, s = new Qd(s.top + s.topUnit, s.left + s.leftUnit);
            h.m = Id(p.alignment);
            h.e = new Ld(s, p.startTime, p.hasAutoEnd, p.endTime, h.m)
        }
        if (p = m.expandingDisplayTypeData)
            s = p.collapsedRect, h.a = new Kd(new Cc(s.left, s.top, s.width, s.height), p.isPushdown ? "pushdown" : Jd(p.expansionMode) || "normal");
        m.imageGalleryTypeData && (h.N = new Nd);
        m = m.pageSettings;
        p = new Od;
        m && (p.c = m.hideDropdowns ||!1, p.a = m.hideObjects ||
        !1, p.e = m.updateZIndex ||!1);
        h.g = p;
        if (null == r)
            r = {};
        else {
            m = {};
            p = void 0;
            for (p in $d)
                s = t(p, r), q(s) && (m[$d[p]] = s);
            r = m
        }
        ce(h, r);
        ae(h);
        h.k == d && gg(c, h)
    }
    1 < f && (k=!0);
    c.q = k;
    return c
}, Qg = function(a, b) {
    var c;
    a.videoData && (c = a.videoData, c = new Lg(c.associatedVideoEntryReportingIdentifier, c.associatedStandardVideoEvent));
    return new zg(a.name, a.reportingId, b, c)
}, Tg = function(a) {
    var b = Sg(a), c = a.j;
    null != document.body&&!tg(a) || null != Qb(document, c) ? b.Va() : Yc(window, "load", function() {
        null != Qb(document, c) ? b.Va() : Gd(b.Va, 0,
        b)
    })
}, Vg = function() {
    var a = [], b = t("studioV2.creatives") || {}, c;
    for (c in b) {
        var d = b[c], e = d.creativeDefinition;
        if (e)
            for (d = d.adResponses; 0 < d.length;) {
                var f = d.shift(), g = f.creativeDto;
                if (g.rendererName == Ke && "200_58" == g.templateVersion)
                    a.push(Rg(f, e));
                else {
                    d.unshift(f);
                    break
                }
            }
    }
    b = [];
    c = t("window.dclkStudioV3.creatives") || [];
    for (e = 0; e < c.length; e++) {
        var d = c[e], g = f = new cg, h = d;
        if (!g.oa) {
            g.oa=!0;
            g.Ra = h.renderingLibraryData.version;
            var k = h.displayConfigParameters;
            g.e = h.creativeParameters.cid || g.e;
            g.m = g.e ? g.e + "_" +
            g.Ra + "_" + dg++ : g.m;
            g.p = new K(parseInt(h.width, 10) || g.p.width, parseInt(h.height, 10) || g.p.height);
            g.N = h.dynamicData || g.N;
            g.G = new K(parseInt(h.slotWidth, 10) || g.G.width, parseInt(h.slotHeight, 10) || g.G.height);
            g.c = h.previewMode || g.c;
            g.J = h.creativeParameters.rv || g.J;
            g.S = h.flashVersion || g.S;
            g.R = h.html5Features || g.R;
            g.ab = h.translated_layout;
            g.a = h.creativeParameters;
            g.j = k.ad_container_id || g.j;
            g.qa = "true" == k.displayHTML5 || g.qa;
            g.ba = k.as_kw || g.ba;
            g.la = k.as_lat || g.la;
            g.ma = k.as_lng || g.ma;
            g.F = "true" == k.mtfRenderFloatInplace ||
            g.F;
            g.pa = "true" == k.tryToWriteHtmlInline || g.pa;
            g.V = Jf[k.dcapp] || g.V;
            var m;
            t:
            {
                m = h.primaryFiles;
                if (null != m)
                    for (var r = 0; r < m.length; ++r)
                        if ("PRE_LOADER" == m[r].renderAs) {
                            m = m[r].url;
                            break t
                        }
                m = ""
            }
            g.ra = m || g.ra;
            g.C = {
                Aa: "",
                Ba: ""
            };
            if (null != h.thirdPartyUrls) {
                m = h.thirdPartyUrls;
                for (var r = new Ff(g.c), p = 0; p < m.length; ++p) {
                    var s;
                    s = m[p].type;
                    s = ib($f, s) ? s : null;
                    null != s && (s in g.g || (g.g[s] = []), g.g[s].push(new Zf(m[p].url, r, "true" == m[p].scrub)))
                }
            }
            g.Q = eg(h.eventTrackingBaseUrl, "met") || g.Q;
            g.W = eg(h.eventTrackingBaseUrl, "stragg") ||
            g.W;
            g.B = h.clickUrl || g.B;
            g.A = parseInt(h.creativeParameters.clickN, 10) || g.A;
            g.U = h.impressionUrl || g.U;
            g.na = k.mtfIFPath || g.na;
            g.D = new Yf(h.clickString, g.V)
        }
        f.Oa = D(F(d.dynamicData));
        for (g = 0; g < d.standardEvents.length; g++)
            h = d.standardEvents[g], kg(f, h.name, h.reportingId);
        for (g = 0; g < d.exitEvents.length; g++)
            h = d.exitEvents[g], k = new Ie(h.name, h.reportingId, h.destinationUrl), k.g = h.targetWindow, k.e = h.windowProperties, qg(f, k);
        for (g = 0; g < d.timerEvents.length; g++)
            h = d.timerEvents[g], k = Ng(h.name), h = new zg(h.name, h.reportingId,
            "Timer", k), k = f, k.ka[h.name] = h, jg(k, h);
        for (g = 0; g < d.counterEvents.length; g++)
            h = d.counterEvents[g], k = Ng(h.name), h = new zg(h.name, h.reportingId, "Counter", k), k = f, k.aa[h.name] = h, jg(k, h);
        h = d.primaryFiles;
        for (g = 0; g < h.length; g++) {
            k = h[g];
            m = d.displayConfigParameters;
            r = new Zd(f.e + "_" + g, k.type, k.renderAs, new K(parseInt(k.width, 10) || 0, parseInt(k.height, 10) || 0), k.url);
            r.G = parseInt(k.zIndex, 10);
            r.n = k.customCss;
            r.j = k.location || r.j;
            if (p = k.flashProperties)
                r.c = new tb(parseInt(p.actionScriptVersion, 10), vb(p.wmode) || "transparent",
                "0.0.0", p.allowScriptAccess || "");
            k.htmlProperties && (r.v = new Md);
            if (p = k.floatingDisplayProperties)
                s = new Qd(p.top, p.left), r.m = Id(p.label), r.e = new Ld(s, parseInt(p.startTime, 10), !1, parseInt(p.duration, 10), r.m);
            if (p = k.expandingDisplayProperties)
                r.a = new Kd(new Cc(parseInt(p.collapsedRectLeft, 10), parseInt(p.collapsedRectTop, 10), parseInt(p.collapsedRectWidth, 10), parseInt(p.collapsedRectHeight, 10)), Jd(p.expansionMode) || "normal");
            r.g = new Od;
            r.g.a = k.hideFlashObjects;
            null != m && ce(r, m);
            ae(r);
            gg(f, r)
        }
        d = f;
        t:
        {
            f = d;
            g = f.k;
            for (h = 0; h < g.length; h++)
                if (!Og.a(g[h])) {
                    f=!1;
                    break t
                }
            f = "200_58" == f.Ra
        }
        f && b.push(d)
    }
    a = a.concat(b);
    for (b = 0; b < a.length; b++)
        Ug(a[b])
}, Ug = function(a) {
    Pg && a.L && (a.L.ge = Pg);
    var b = {
        gb: 0 < hg(a, "FLASH").length,
        flashVersion: a.S,
        Bb: 0 < hg(a, "HTML5").length,
        html5Features: a.R,
        Ca: !1
    };
    if (Ig(b))
        Tg(a);
    else if (a.H) {
        var c = [];
        H(fg(a, "IMPRESSION_IMG"), function(a) {
            c.push(a.a)
        });
        Jg(a.e, a.H, a.j, a.c, c, a.D)
    } else if (a.q) {
        var d = {};
        a = new Eg(a.ca || 0, a.ja || 0, We(a), a.c, new Ff(a.c), a.q);
        var e = b.gb ? M || N || O || Bb ? Cb ? 2: 0 <= G(ec, b.flashVersion) ?
        0: 3: 1: 4;
        4 != e || xc(b.flashVersion) || (d = Fg(Hg(e), d), e = M || N || O || Bb ? Cb ? 2 : 0 <= G(ec, b.flashVersion) ? 0 : 3 : 1);
        b = Hg(e);
        Gg(a, Fg(b, d))
    }
}, Xg = function(a) {
    Wg[a] && (Wg[a].O(), delete Wg[a])
};
var Yg = function(a) {
    var b;
    a.mozVisibilityState ? b = "mozvisibilitychange" : a.webkitVisibilityState ? b = "webkitvisibilitychange" : a.visibilityState && (b = "visibilitychange");
    return b
}, Zg = function(a, b) {
    if (3 == ({
        visible: 1,
        hidden: 2,
        prerender: 3,
        preview: 4
    }
    [b.webkitVisibilityState || b.mozVisibilityState || b.visibilityState || ""] || 0))
        return !1;
    a();
    return !0
}, $g = function(a) {
    var b = document;
    if (!Zg(a, b)) {
        var c=!1, d = Yg(b), e = function() {
            if (!c && Zg(a, b)) {
                c=!0;
                var f = e;
                b.removeEventListener ? b.removeEventListener(d, f, !1) : b.detachEvent &&
                b.detachEvent("on" + d, f)
            }
        };
        d && Cf(b, d, e)
    }
};
var ah = function(a, b) {
    B.call(this, a);
    this.c = b
};
z(ah, B);
var bh = "EXPAND EXPAND_REQUEST EXPAND_FINISH COLLAPSE COLLAPSE_REQUEST COLLAPSE_FINISH EXPAND_FULL_SCREEN COLLAPSE_FULL_SCREEN SHOW HIDE".split(" ");
var X = function(a, b) {
    this.c = b;
    this.B = new T(this);
    this.B.c(this.c, bh, this.W);
    this.a = a;
    this.v = [];
    this.q = [];
    this.m = [];
    this.e = [];
    this.p = [];
    this.o = [];
    this.j = [];
    this.g = []
};
z(X, A);
X.prototype.k = function() {
    this.B.O();
    this.g = this.j = this.e = this.o = this.p = this.m = this.q = this.v = null;
    X.M.k.call(this)
};
X.prototype.W = function(a) {
    if (a.c == this.a) {
        var b = null;
        switch (a.type) {
        case "SHOW":
            b = this.v;
            break;
        case "HIDE":
            b = this.q;
            break;
        case "COLLAPSE_REQUEST":
            b = this.j;
            break;
        case "COLLAPSE_FINISH":
            b = this.g;
            break;
        case "COLLAPSE":
            b = this.e;
            break;
        case "EXPAND":
            b = this.m;
            break;
        case "EXPAND_REQUEST":
            b = this.p;
            break;
        case "EXPAND_FINISH":
            b = this.o
        }
        b && ch(this, b)
    }
};
var ch = function(a, b) {
    H(b, function(a) {
        a(this)
    }, a)
}, dh = function(a) {
    return a.c.g[a.a] || null
};
X.prototype.A = function() {
    return this.a.p
};
X.prototype.getType = X.prototype.A;
X.prototype.n = function() {
    return this.a.id
};
X.prototype.getId = X.prototype.n;
X.prototype.V = function() {
    var a = dh(this);
    return a && (a = a.P()) ? (a = new J(a.offsetLeft, a.offsetTop), {
        x: a.x,
        y: a.y
    }) : null
};
X.prototype.getPosition = X.prototype.V;
X.prototype.ja = function(a, b) {
    var c = dh(this);
    if (c) {
        var c = c.P(), d, e, f = N && (yb || Hb) && P("1.9");
        a instanceof J ? (d = a.x, e = a.y) : (d = a, e = b);
        c.style.left = Hc(d, f);
        c.style.top = Hc(e, f)
    }
};
X.prototype.setPosition = X.prototype.ja;
X.prototype.ka = function(a) {
    var b = dh(this);
    b && Ec(b.P(), a)
};
X.prototype.setStyle = X.prototype.ka;
X.prototype.U = function() {
    var a = dh(this);
    if (a) {
        var b = a.P();
        if (b) {
            var c, d = Ob(b), e = Fc(b), f = N && d.getBoxObjectFor&&!b.getBoundingClientRect && "absolute" == e && (c = d.getBoxObjectFor(b)) && (0 > c.screenX || 0 > c.screenY), a = new J(0, 0), g;
            c = d ? Ob(d) : document;
            (g=!M || M && 9 <= Mb) || (g = "CSS1Compat" == Pb(c).a.compatMode);
            g = g ? c.documentElement : c.body;
            if (b != g)
                if (b.getBoundingClientRect) {
                    i:
                    {
                        var h;
                        try {
                            h = b.getBoundingClientRect()
                        } catch (k) {
                            c = {
                                left: 0,
                                top: 0,
                                right: 0,
                                bottom: 0
                            };
                            break i
                        }
                        M && b.ownerDocument.body && (c = b.ownerDocument, h.left -=
                        c.documentElement.clientLeft + c.body.clientLeft, h.top -= c.documentElement.clientTop + c.body.clientTop);
                        c = h
                    }
                    h = Pb(d).a;
                    d = O || "CSS1Compat" != h.compatMode ? h.body || h.documentElement : h.documentElement;
                    h = Tb(h);
                    d = M && P("10") && h.pageYOffset != d.scrollTop ? new J(d.scrollLeft, d.scrollTop) : new J(h.pageXOffset || d.scrollLeft, h.pageYOffset || d.scrollTop);
                    a.x = c.left + d.x;
                    a.y = c.top + d.y
                } else if (d.getBoxObjectFor&&!f)
                    c = d.getBoxObjectFor(b), d = d.getBoxObjectFor(g), a.x = c.screenX - d.screenX, a.y = c.screenY - d.screenY;
                else {
                    c = b;
                    do {
                        a.x +=
                        c.offsetLeft;
                        a.y += c.offsetTop;
                        c != b && (a.x += c.clientLeft || 0, a.y += c.clientTop || 0);
                        if (O && "fixed" == Fc(c)) {
                            a.x += d.body.scrollLeft;
                            a.y += d.body.scrollTop;
                            break
                        }
                        c = c.offsetParent
                    }
                    while (c && c != b);
                    if (Bb || O && "absolute" == e)
                        a.y -= d.body.offsetTop;
                        for (c = b; (c = Gc(c)) && c != d.body && c != g;)
                            a.x -= c.scrollLeft, Bb && "TR" == c.tagName || (a.y -= c.scrollTop)
                        }
            return {
                x: a.x,
                y: a.y
            }
        }
    }
    return null
};
X.prototype.getPagePosition = X.prototype.U;
X.prototype.F = function() {
    return {
        width: this.a.width,
        height: this.a.height
    }
};
X.prototype.getDimension = X.prototype.F;
X.prototype.va = function(a, b) {
    var c = dh(this);
    c && c.va(a, b)
};
X.prototype.setDimension = X.prototype.va;
X.prototype.S = function() {
    var a = this.a.a && this.a.a.a;
    return a ? {
        width: a.width,
        height: a.height
    } : this.F()
};
X.prototype.getCollapsedDimension = X.prototype.S;
X.prototype.ra = function(a, b, c, d) {
    var e = dh(this);
    e && (e.P().style.clip = ["rect(", a, "px ", b, "px ", c, "px ", d, "px)"].join(""), e.P().style.position = "absolute")
};
X.prototype.setClip = X.prototype.ra;
X.prototype.R = function() {
    var a = dh(this);
    return a ? a.P() : null
};
X.prototype.getContainerElement = X.prototype.R;
X.prototype.pa = function() {
    var a = dh(this);
    return a ? a.a : null
};
X.prototype.getAssetElement = X.prototype.pa;
X.prototype.T = function(a) {
    this.v.push(a)
};
X.prototype.addShowCallback = X.prototype.T;
X.prototype.ma = function(a) {
    I(this.v, a)
};
X.prototype.removeShowCallback = X.prototype.ma;
X.prototype.Q = function(a) {
    this.q.push(a)
};
X.prototype.addHideCallback = X.prototype.Q;
X.prototype.la = function(a) {
    I(this.q, a)
};
X.prototype.removeHideCallback = X.prototype.la;
X.prototype.I = function(a) {
    this.m.push(a)
};
X.prototype.addExpandCallback = X.prototype.I;
X.prototype.aa = function(a) {
    I(this.m, a)
};
X.prototype.removeExpandCallback = X.prototype.aa;
X.prototype.L = function(a) {
    this.p.push(a)
};
X.prototype.addExpandRequestCallback = X.prototype.L;
X.prototype.ba = function(a) {
    I(this.p, a)
};
X.prototype.removeExpandRequestCallback = X.prototype.ba;
X.prototype.J = function(a) {
    this.o.push(a)
};
X.prototype.addExpandFinishCallback = X.prototype.J;
X.prototype.da = function(a) {
    I(this.o, a)
};
X.prototype.removeExpandFinishCallback = X.prototype.da;
X.prototype.C = function(a) {
    this.e.push(a)
};
X.prototype.addCollapseCallback = X.prototype.C;
X.prototype.Z = function(a) {
    I(this.e, a)
};
X.prototype.removeCollapseCallback = X.prototype.Z;
X.prototype.H = function(a) {
    this.j.push(a)
};
X.prototype.addCollapseRequestCallback = X.prototype.H;
X.prototype.ca = function(a) {
    I(this.j, a)
};
X.prototype.removeCollapseRequestCallback = X.prototype.ca;
X.prototype.D = function(a) {
    this.g.push(a)
};
X.prototype.addCollapseFinishCallback = X.prototype.D;
X.prototype.X = function(a) {
    I(this.g, a)
};
X.prototype.removeCollapseFinishCallback = X.prototype.X;
X.prototype.wa = function() {
    eh(this.c, this.a)
};
X.prototype.show = X.prototype.wa;
X.prototype.qa = function() {
    this.c.p(this.a)
};
X.prototype.hide = X.prototype.qa;
X.prototype.oa = function() {
    fh(this.c, this.a)
};
X.prototype.expand = X.prototype.oa;
X.prototype.na = function() {
    gh(this.c, this.a)
};
X.prototype.collapse = X.prototype.na;
var Y = function(a, b) {
    this.a = a;
    this.c = b
};
z(Y, A);
Y.prototype.g = function() {
    return this.a.name
};
Y.prototype.getName = Y.prototype.g;
Y.prototype.j = function() {
    var a = this.a.a;
    return (/^https?/.test(a) ? "" : this.c) + a
};
Y.prototype.getUrl = Y.prototype.j;
Y.prototype.m = function() {
    return this.a.e
};
Y.prototype.isVideo = Y.prototype.m;
Y.prototype.e = function() {
    return this.a.c
};
Y.prototype.getStreamingUrl = Y.prototype.e;
var hh = function(a, b) {
    this.a = a;
    this.e = b
};
hh.prototype.k = function() {
    return this.a.name
};
hh.prototype.getName = hh.prototype.k;
hh.prototype.g = function() {
    return this.a.c
};
hh.prototype.getUrl = hh.prototype.g;
hh.prototype.c = function(a, b) {
    this.e.bb(new qa("logExitFlushEventsOpenPopup", this.a.name, a || null, b || null))
};
hh.prototype.fireExit = hh.prototype.c;
var Z = function(a, b) {
    A.call(this);
    this.a = a;
    this.c = ih(a.k, b);
    this.j = Va(this.c, function(a) {
        return a.n()
    });
    this.e = jh(a.v, We(a) || "");
    this.g = kh(a.T, b);
    this.AssetTypes = lh;
    mh(this)
};
z(Z, A);
var ih = function(a, b) {
    return Oa(a, function(a) {
        return new X(a, b)
    })
}, jh = function(a, b) {
    return Oa(a, function(a) {
        return new Y(a, b)
    })
}, kh = function(a, b) {
    return Oa(hb(a), function(a) {
        return new hh(a, b)
    })
}, mh = function(a) {
    var b = t("studioV2.api.creatives") || [];
    ha("studioV2.api.creatives", b);
    b.push(a)
};
Z.prototype.k = function() {
    var a = t("studioV2.api.creatives");
    a && u(a) && I(a, this);
    ia(this.c, this.e, this.g);
    Z.M.k.call(this)
};
Z.prototype.n = function() {
    return "1.0"
};
Z.prototype.getApiVersion = Z.prototype.n;
Z.prototype.B = function() {
    return this.a.m
};
Z.prototype.getCreativeId = Z.prototype.B;
Z.prototype.L = function() {
    return this.a.a.sid || ""
};
Z.prototype.getSiteId = Z.prototype.L;
Z.prototype.Q = function() {
    return this.a.a.sn || ""
};
Z.prototype.getSiteName = Z.prototype.Q;
Z.prototype.D = function() {
    return this.a.a.aid || ""
};
Z.prototype.getAdId = Z.prototype.D;
Z.prototype.I = function() {
    return this.a.a.buy || ""
};
Z.prototype.getBuyId = Z.prototype.I;
Z.prototype.m = function() {
    return this.a.a.cid || ""
};
Z.prototype.getAdserverCreativeId = Z.prototype.m;
Z.prototype.J = function() {
    return this.a.a.pid || ""
};
Z.prototype.getPlacementId = Z.prototype.J;
Z.prototype.H = function() {
    return this.a.a.adv || ""
};
Z.prototype.getAdvertiserId = Z.prototype.H;
Z.prototype.q = function() {
    return this.c
};
Z.prototype.getAssets = Z.prototype.q;
Z.prototype.A = function() {
    return this.e
};
Z.prototype.getChildAssets = Z.prototype.A;
Z.prototype.o = function(a) {
    return this.c[a] || null
};
Z.prototype.getAssetAt = Z.prototype.o;
Z.prototype.p = function(a) {
    return this.j[a] || null
};
Z.prototype.getAssetById = Z.prototype.p;
Z.prototype.C = function(a) {
    return Sa(this.c, function(b) {
        return b.A() == a
    })
};
Z.prototype.getFirstAssetByType = Z.prototype.C;
Z.prototype.v = function(a) {
    return Na(this.c, function(b) {
        return b.A() == a
    })
};
Z.prototype.getAssetsByType = Z.prototype.v;
Z.prototype.F = function() {
    return this.g
};
Z.prototype.getExits = Z.prototype.F;
var lh = {
    INPAGE: "BANNER",
    EXPANDING: "EXPANDABLE",
    FLOAT: "FLOATING",
    OVERLAY: "OVERLAY"
};
var nh = function() {
    this.a = [];
    this.c = []
};
l = nh.prototype;
l.addReporter = function(a) {
    H(this.a, function(b) {
        b.newReporterCallback(a);
        a.newReporterCallback(b)
    });
    H(this.c, function(b) {
        a.registerChargeableEventName(b)
    });
    this.a.push(a)
};
l.reportEvents = function(a) {
    H(this.a, function(b) {
        b.reportEvents(a)
    })
};
l.registerChargeableEventName = function(a) {
    H(this.a, function(b) {
        b.registerChargeableEventName(a)
    });
    this.c.push(a)
};
l.logCustomVariable = function(a, b) {
    H(this.a, function(c) {
        c.logCustomVariable(a, b)
    })
};
l.getType = function() {
    return "UNIFIED_DISPATCHER"
};
l.getConfig = function() {
    return {
        reportingApiVersion: 2
    }
};
l.newReporterCallback = aa;
l.supportsChargeableEvents = function() {
    var a=!1;
    H(this.a, function(b) {
        b.supportsChargeableEvents() && (a=!0)
    });
    return a
};
var oh = function(a) {
    this.e = a.Q;
    this.c = a.W;
    this.a = a;
    this.k = new Ff
};
l = oh.prototype;
l.reportEvents = function(a) {
    for (var b = [], c = 0; c < a.length; c++) {
        var d = a[c], e = sg(this.a, d.unifiedReportingEvent);
        if (e) {
            var f = b, d = ["eid", c + 1, "=", e, ";ecn", c + 1, "=", d.count, ";etm", c + 1, "=", d.time, ";"].join("");
            0 == f.length && f.push("");
            e = f.pop();
            950 < e.length + d.length && (f.push(e), e = "");
            e += d;
            f.push(e)
        }
    }
    if (b.length)
        for (a = y(), c = 0; c < b.length; c++)
            document.createElement("img").src = this.e + "&timestamp=" + a + ";" + b[c]
};
l.registerChargeableEventName = function(a) {
    var b = this.a.I;
    lg(this.a, a, !0)&&!b && mg(this.a, !1)
};
l.logCustomVariable = function(a, b) {
    if (this.c && 0 < this.c.length) {
        var c = [this.c, "&timestamp=", y(), ";str=", a, ";strtype=", b].join("");
        Gf(this.k, c)
    }
};
l.getType = function() {
    return "STUDIO"
};
l.getConfig = function() {
    return {
        reportingApiVersion: 2
    }
};
l.newReporterCallback = function(a) {
    a.supportsChargeableEvents()&&!this.a.I && mg(this.a, !0)
};
l.supportsChargeableEvents = function() {
    return !1
};
var ph = function(a) {
    this.a = {};
    this.c = {};
    this.g = {};
    this.v = {};
    this.D =- 1;
    this.C=!1;
    this.L=!a.c;
    this.n = null;
    this.A=!1;
    this.j = this.B(a);
    this.q = a;
    this.F = 5E3;
    this.o = {};
    this.p = {};
    this.m = {}
};
z(ph, A);
var qh = function(a) {
    a.C=!0;
    a.D = y()
}, rh = function(a, b, c) {
    if (c ||!a.v[b])
        a.v[b] || (a.a[b] = 0, a.v[b]=!0), a.a[b]++, (b = a.q.n[b]) && b.chargeable && a.e()
}, sh = function(a, b, c) {
    if (!(b in a.g)) {
        for (var d = y(), e = a.o[b]; e && 0 < e.length;)
            d += e.pop();
        a.g[b] = d;
        c && (a.m[c] = a.m[c] || [], a.m[c].push(b));
        rh(a, b, !1)
    }
}, th = function(a, b, c) {
    if (b in a.g) {
        for (var d = a.g[b], e = y(), f = a.p[b]; f && 0 < f.length;)
            e += f.pop();
        d = e - d;
        0 > d && (d = 0);
        a.c[b] = a.c[b] || 0;
        a.c[b] += d;
        delete a.g[b];
        c && a.m[c] && I(a.m[c], b)
    }
}, uh = function(a) {
    for (var b in a.g)
        th(a, b)
};
ph.prototype.e = function() {
    var a = [], b;
    if (b=!this.A)
        t: {
        for (var c in this.a)
            if (b = this.q.n[c], 0 < this.a[c] && b && b.chargeable) {
                b=!0;
                break t
            }
            b=!1
    }
    c = b;
    if (!(b = c)) {
        if (!(b = this.C&&!this.G && (!this.L ||!this.n || y() - this.n > this.F)))
            t: {
            for (var d in this.a)
                if ((b = this.q.n[d]) && b.m) {
                    b=!0;
                    break t
                }
                b=!1
        }
        b = b&&!(12E5 < y() - this.D)
    }
    if (b) {
        for (var e in this.g)
            th(this, e), sh(this, e);
        for (e in this.a)
            d = 0, b = this.a[e], this.c[e] && (d = Math.floor(this.c[e] / 1E3), this.c[e] -= 1E3 * d), this.a[e] = 0, (0 < b || 0 < d) && a.push({
                unifiedReportingEvent: Kf(e),
                count: b,
                time: d
            });
        c && (this.A=!0);
        this.n = y();
        this.j.reportEvents(a)
    }
};
ph.prototype.k = function() {
    uh(this);
    this.e();
    ph.M.k.call(this)
};
var vh = function(a) {
    ph.call(this, a)
};
z(vh, ph);
vh.prototype.B = function(a) {
    var b;
    "gdn" == je() ? (b = t("googlecreative.reporting.sharedReporter"), b || (b = new nh, ha("googlecreative.reporting.sharedReporter", b)), a.ab || b.addReporter(new oh(a))) : b = new oh(a);
    return b
};
vh.prototype.J = aa;
vh.prototype.I = aa;
vh.prototype.H = function(a, b) {
    this.j.logCustomVariable(a, b)
};
var wh = function(a) {
    this.c = a;
    this.a = "undefined" != typeof DARTDebugEventHandler && DARTDebugEventHandler ? new DARTDebugEventHandler : null
};
l = wh.prototype;
l.reportEvents = function(a) {
    if (this.a)
        for (var b = 0; b < a.length; ++b) {
            var c = a[b], d = c.unifiedReportingEvent, d = this.c.n[[d.type, escape(d.name), escape(d.videoName), d.trigger].join(":")];
            try {
                this.a.handleEventActivity(d.name, d.type, c.count, c.time, !d.j, this.c.e)
            } catch (e) {}
    }
};
l.registerChargeableEventName = aa;
l.logCustomVariable = function(a, b) {
    try {
        this.a.handleCustomVariable(unescape(a), b, this.c.e)
    } catch (c) {}
};
l.getType = function() {
    return "OUTPUT_CONSOLE"
};
l.getConfig = function() {
    return {
        reportingApiVersion: 2
    }
};
l.newReporterCallback = aa;
l.supportsChargeableEvents = function() {
    return !1
};
var xh = function(a) {
    ph.call(this, a)
};
z(xh, ph);
xh.prototype.B = function(a) {
    return new wh(a)
};
xh.prototype.J = function(a, b, c) {
    var d = this.j;
    a = d.c.n[a];
    if (d.a)
        try {
            d.a.handleEventAction(b, a.name, c, a.type, !a.j, d.c.e)
    } catch (e) {}
};
xh.prototype.I = function(a) {
    var b = this.j;
    try {
        b.a.handleCustomJSExecution(a, b.c.e)
    } catch (c) {}
};
xh.prototype.H = function(a, b) {
    this.j.logCustomVariable(a, b)
};
var $ = function(a, b) {
    Q.call(this);
    this.a = a;
    this.R = b;
    this.e = a.c ? new xh(a) : new vh(a);
    this.g = {};
    this.m = {};
    this.A = new T(this);
    this.D = this.H=!1;
    this.n = this.I = 0;
    this.o = [];
    this.q = {};
    this.J = new Z(this.a, this);
    this.B = new Ff(this.a.c);
    this.j = null;
    this.v = void 0;
    var c = a.ca, d = a.ja;
    null != c && null != d && (a.q ? this.j = new Eg(c, d, We(a), a.c, this.B, a.q) : this.j = new Ag(c, d, We(a), a.c, this.B), yh(this, "ge", "gb"), yh(this, "pe", "pb"));
    M && this.A.c(self, "unload", this.O)
};
z($, Q);
var zh = [10, 20, 50, 120, 240], Wg = {}, yh = function(a, b, c) {
    var d = a.a.L;
    a.j && d && d[c] && d[b] && (Bg(a.j, c, null, d[c]), Bg(a.j, b, c, d[b]))
};
l = $.prototype;
l.pb = function(a) {
    a = this.m[a.a];
    this.j && (0 == this.a.k.length || a.o) && (Bg(this.j, "fe", "fb"), Dg(this.j))
};
l.xb = function() {
    this.n++;
    Ah(this)
};
l.ub = function() {
    Bh(this, new ma("logEvent", "Count", "FULL_SCREEN"))
};
l.zb = function(a) {
    "boolean" == typeof a.g && a.g ? Bh(this, a) : Ch(this, a);
    a.n && this.e.e()
};
l.wb = function() {
    this.e.e()
};
var Ch = function(a, b) {
    var c = ka(b.c), c = ng(b.k, c);
    null != c && Dh(a, c, b.c, b.m, b.a)
}, Dh = function(a, b, c, d, e) {
    if (null !== b)
        switch (a.e.J(b, c, d), c) {
        case "Exit":
            for (c = fg(a.a, "CLICK"), e = c.length - 1; 0 <= e; --e)
                ag(c[e]);
            case "Count":
                rh(a.e, b, d);
                break;
            case "Start":
                sh(a.e, b, e);
                break;
            case "Stop":
                th(a.e, b, e)
        }
}, Bh = function(a, b) {
    if ("DISPLAY_TIMER" == b.e) {
        if ("Start" == b.c) {
            a.H || ($g(x(a.U, a)), a.H=!0);
            if (0 == a.I++) {
                var c = og(a.a, "DISPLAY_TIMER");
                c && Dh(a, c, "Start", !1)
            }
            a.n--
        }
    } else (c = og(a.a, b.e)
        ) && Dh(a, c, b.c, !1, b.a)
};
l = $.prototype;
l.Ab = function(a) {
    Bh(this, a);
    var b = pg(this.a, a.e, a.videoName);
    null != b && Dh(this, b, a.c, !1, a.a)
};
l.bb = function(a) {
    var b = this.a.T[a.k], c = null != a.j ? a.j: b.c;
    null != a.e&&!D(F(a.e)) && (c = qe(c, a.e));
    var d = this.a.B, e = this.a.A;
    if (!D(F(d))&&-1 < d.indexOf("?"))
        for (var e = ca(e) ? e : 1, f = 0; f < e; f++)
            c = escape(c);
    window.open(d + c, b.g || "_blank", (null != b.e ? b.e : void 0) || "");
    Ch(this, a);
    this.e.e()
};
l.vb = function(a) {
    var b = this.m[a.a];
    switch (a.type) {
    case "expandAsset":
        fh(this, b);
        break;
    case "expandRequested":
        fh(this, b);
        break;
    case "expandFinished":
        if (a = this.g[b])
            uf(a), R(this, new ah("EXPAND_FINISH", b));
        break;
    case "collapseAsset":
        gh(this, b);
        break;
    case "collapseRequested":
        gh(this, b);
        break;
    case "collapseFinished":
        if (a = this.g[b])
            a.Ha(), R(this, new ah("COLLAPSE_FINISH", b));
        break;
    case "tellAssetHide":
        this.p(b)
    }
};
l.sb = function(a) {
    var b;
    t: {
        b = this.a;
        var c = a.c;
        if (c && (c = Id(c)) && b.wa[c]) {
            b = b.wa[c];
            break t
        }
        for (c = 0; c < b.k.length; c++) {
            var d = b.k[c];
            if (this.m[a.a] != d) {
                b = d;
                break t
            }
        }
        b = null
    }
    if (b)
        switch (a.type) {
        case "tellCompanionAssetShow":
            eh(this, b);
            break;
        case "tellCompanionAssetHide":
            this.p(b)
        }
};
l.rb = function(a) {
    a = unescape(a.c);
    this.a.c && this.e.I(a);
    try {
        eval(a)
    } catch (b) {}
};
l.yb = function(a) {
    a = parseInt(a.c, 10);
    !isNaN(a) && 0 < a && (this.e.F = a)
};
l.qb = function(a) {
    var b = og(this.a, a.k);
    if (b) {
        var c = this.e, d = a.c;
        a = a.e;
        var e = c.o[b] || (c.o[b] = []), b = c.p[b] || (c.p[b] = []);
        0 != d && e.push(d);
        0 != a && b.push(a)
    }
};
l.tb = function(a) {
    this.e.H(a.c, a.e)
};
l.Va = function() {
    this.a.Pa && Xf(this.a.Pa);
    this.a.q && this.j && Gg(this.j, Fg("mtf_br", {}), this.a.e);
    for (var a = this.a.k, b = 0; b < a.length; b++) {
        var c = a[b], d = c.e;
        eh(this, c, d&&!d.k?-1 : 1E3 * (d && d.k && 0 < d.j && d.j || 0))
    }
};
var eh = function(a, b, c) {
    c ? 0 < c && (a.n++, a.o.push(Gd(x(a.C, a, b), c))) : (a.n++, a.C(b))
};
$.prototype.C = function(a) {
    if (!(a in this.g)) {
        var b;
        b = this.a;
        if (b = this.R.a(a) ? new W(a, b) : null) {
            this.g[a] = b;
            this.m[b] = a;
            b = this.g[a];
            var c = this.A;
            c.c(b, "conduitInitialized", this.pb);
            c.c(b, "logEvent", this.zb);
            c.c(b, "logVideoEvent", this.Ab);
            c.c(b, "logExitFlushEventsOpenPopup", this.bb);
            c.c(b, "expandAsset expandRequested expandFinished collapseAsset collapseRequested collapseFinished tellAssetHide".split(" "), this.vb);
            c.c(b, ["tellCompanionAssetShow", "tellCompanionAssetHide"], this.sb);
            c.c(b, "invokeExternalJSFunction",
            this.rb);
            c.c(b, "setThrottlingWindow", this.yb);
            c.c(b, "reportCustomVariable", this.tb);
            c.c(b, "setTimerAdjustment", this.qb);
            c.c(b, "flushCounters", this.wb);
            c.c(b, "RESET", this.xb);
            c.c(b, "registerChargeableEventName", this.L);
            c.c(b, "fullscreenExpandFinished", this.ub);
            this.j && (0 == this.a.k.length || a.o) && Bg(this.j, "fb");
            if (a.o || this.a.F) {
                if (b = Qb(document, this.a.j))
                    if (this.a.F && (b.style.position = "relative"), a.a && a.a.c && (this.v = b.style.height, b.style.height = "auto"), qf(this.g[a], b), a.n) {
                        b = this.g[a];
                        for (var c = a.n.split(";"),
                        d = 0; d < c.length; d++) {
                            var e = c[d].split(":");
                            2 <= e.length && Ec(b.P(), e[0], e[1])
                        }
                    }
                } else 
                    document.body && document.body.firstChild ? (b = document.body.firstChild, qf(this.g[a], b.parentNode, b)) : document.body ? qf(this.g[a], document.body) : document.documentElement && qf(this.g[a], document.documentElement);
            R(this, new ah("SHOW", a))
        }
    }
    a.e && (b = a.e, b = 1E3 * (b && b.e && 0 < b.c && b.c||-1), 0 < b && (b = Gd(x(this.p, this, a), b), this.o.push(b), this.q[a] = b));
    this.D || ($g(x(this.S, this)), this.D=!0)
};
$.prototype.p = function(a) {
    var b = this.g[a];
    if (b) {
        for (var c = this.e, d = c.m[b] || []; 0 < d.length;)
            th(c, d.pop(), b);
        delete this.m[b];
        b.O();
        delete this.g[a];
        this.q[a] && (n.clearTimeout(this.q[a]), delete this.q[a]);
        R(this, new ah("HIDE", a));
        Ah(this)
    }
};
var fh = function(a, b) {
    var c = a.g[b];
    c && (tf(c), R(a, new ah("EXPAND", b)), R(a, new ah("EXPAND_REQUEST", b)))
}, gh = function(a, b) {
    var c = a.g[b];
    c && (vf(c), R(a, new ah("COLLAPSE", b)), R(a, new ah("COLLAPSE_REQUEST", b)))
}, Eh = function(a) {
    Bh(a, new ma("logEvent", "Count", "HTML5_CREATIVE_IMPRESSION"));
    a.e.e()
}, Fh = function(a) {
    Bh(a, new ma("logEvent", "Count", "DYNAMIC_CREATIVE_IMPRESSION"));
    a.e.e()
};
$.prototype.S = function() {
    Gh(this)
};
var Gh = function(a) {
    qh(a.e);
    a.a.q && a.j && Gg(a.j, Fg("mtf_fi", {}), a.a.e);
    a.a.hb && Eh(a);
    a.a.cb() && Fh(a);
    a.a.Wa && Gf(a.B, a.a.U);
    H(["IMPRESSION_IMG", "ARTWORK_IMPRESSION"], function(a) {
        H(fg(this.a, a), function(a) {
            ag(a)
        }, this)
    }, a)
};
$.prototype.U = function() {
    var a = [];
    if (this.a.c)
        for (var b = zh[zh.length - 1] / 2, c = 1; c <= b; c++)
            a[a.length] = 2 * c;
    else 
        a = zh;
    b = x(this.e.e, this.e);
    for (c = 0; c < a.length; c++)
        this.o.push(Gd(b, 1E3 * a[c]))
};
var Ah = function(a) {
    if (0==--a.I) {
        var b = og(a.a, "DISPLAY_TIMER");
        b && Dh(a, b, "Stop", !1);
        uh(a.e);
        0 == a.n && a.O()
    }
};
$.prototype.L = function(a) {
    this.e.j.registerChargeableEventName(a.c)
};
$.prototype.k = function() {
    q(this.v) && (Qb(document, this.a.j).style.height = this.v);
    this.A.O();
    for (this.J.O(); 0 < this.o.length;) {
        var a = this.o.pop();
        n.clearTimeout(a)
    }
    for (var a = this.a.k, b = 0; b < a.length; b++) {
        var c = a[b];
        c in this.g && (this.g[c].O(), delete this.g[c])
    }
    this.m = null;
    this.e.O();
    this.e = null;
    $.M.k.call(this)
};
var Hh = function() {}, Ih = function(a, b) {
    return u(a) ? 0 <= Ma(a, b) : a == b
};
Hh.prototype.a = function() {
    return !1
};
var Sg = function(a) {
    var b = Og, c = a.m;
    Wg[c] || (Wg[c] = new $(a, b));
    return Wg[c]
};
var Jh = function() {};
z(Jh, Hh);
Jh.prototype.a = function(a) {
    var b = Ih("BANNER", a.p), c=!1, c = q("FLASH") ? Ih("FLASH", a.k) : !0;
    return b && c
};
for (var Og = new Jh, Kh = document.getElementsByTagName("noscript"), Lh = 0; Lh < Kh.length; Lh++)
    Kh[Lh].style.display = "none";
if (!t("studioV2.loadedLibraries.200_58.flash_inpage.bootstrap")) {
    Pg = y();
    ha("studioV2.loadedLibraries.200_58.flash_inpage.bootstrap", Vg);
    ha("studioV2.loadedLibraries.200_58.flash_inpage.unload", Xg);
    var Ke = "flash_inpage", Mh = t("studioV2.defer");
    Mh && w(Mh) ? Mh(Vg) : Vg()
};
})()
