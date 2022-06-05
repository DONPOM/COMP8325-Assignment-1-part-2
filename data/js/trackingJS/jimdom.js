(function() {
    function f(a, e) {
        function g(b) {
            if (g[b] !== D)
                return g[b];
            var h;
            if ("bug-string-char-index" == b)
                h = "a" != "a"[0];
            else if ("json" == b)
                h = g("json-stringify") && g("json-parse");
            else {
                var l;
                if ("json-stringify" == b) {
                    h = e.stringify;
                    var c = "function" == typeof h && E;
                    if (c) {
                        (l = function() {
                            return 1
                        }).toJSON = l;
                        try {
                            c = "0" === h(0) && "0" === h(new m) && '""' == h(new n) && h(z) === D && h(D) === D && h() === D && "1" === h(l) && "[1]" == h([l]) && "[null]" == h([D]) && "null" == h(null) && "[null,null,null]" == h([D, z, null]) && '{"a":[1,true,false,null,"\\u0000\\b\\n\\f\\r\\t"]}' ==
                            h({
                                a: [l, !0, !1, null, "\x00\b\n\f\r\t"]
                            }) && "1" === h(null, l) && "[\n 1,\n 2\n]" == h([1, 2], null, 1) && '"-271821-04-20T00:00:00.000Z"' == h(new r( - 864E13)) && '"+275760-09-13T00:00:00.000Z"' == h(new r(864E13)) && '"-000001-01-01T00:00:00.000Z"' == h(new r( - 621987552E5)) && '"1969-12-31T23:59:59.999Z"' == h(new r( - 1))
                        } catch (a) {
                            c=!1
                        }
                    }
                    h = c
                }
                if ("json-parse" == b) {
                    h = e.parse;
                    if ("function" == typeof h)
                        try {
                            if (0 === h("0")&&!h(!1)) {
                                l = h('{"a":[1,true,false,null,"\\u0000\\b\\n\\f\\r\\t"]}');
                                var d = 5 == l.a.length && 1 === l.a[0];
                                if (d) {
                                    try {
                                        d=!h('"\t"')
                                    } catch (p) {}
                                    if (d)
                                        try {
                                            d =
                                            1 !== h("01")
                                        } catch (k) {}
                                        if (d)
                                            try {
                                                d = 1 !== h("1.")
                                            } catch (f) {}
                                        }
                                    }
                    } catch (v) {
                        d=!1
                    }
                    h = d
                }
            }
            return g[b]=!!h
        }
        a || (a = d.Object());
        e || (e = d.Object());
        var m = a.Number || d.Number, n = a.String || d.String, s = a.Object || d.Object, r = a.Date || d.Date, J = a.SyntaxError || d.SyntaxError, y = a.TypeError || d.TypeError, G = a.Math || d.Math, c = a.JSON || d.JSON;
        "object" == typeof c && c && (e.stringify = c.stringify, e.parse = c.parse);
        var s = s.prototype, z = s.toString, A, H, D, E = new r( - 0xc782b5b800cec);
        try {
            E =- 109252 == E.getUTCFullYear() && 0 === E.getUTCMonth() && 1 === E.getUTCDate() &&
            10 == E.getUTCHours() && 37 == E.getUTCMinutes() && 6 == E.getUTCSeconds() && 708 == E.getUTCMilliseconds()
        } catch (P) {}
        if (!g("json")) {
            var K = g("bug-string-char-index");
            if (!E)
                var C = G.floor, Q = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], x = function(b, h) {
                    return Q[h] + 365 * (b - 1970) + C((b - 1969 + (h =+ (1 < h))) / 4) - C((b - 1901 + h) / 100) + C((b - 1601 + h) / 400)
                };
            (A = s.hasOwnProperty) || (A = function(b) {
                var h = {}, l;
                (h.__proto__ = null, h.__proto__ = {
                    toString: 1
                }, h).toString != z ? A = function(b) {
                    var h = this.__proto__;
                    b = b in (this.__proto__ = null, this);
                    this.__proto__ =
                    h;
                    return b
                } : (l = h.constructor, A = function(b) {
                    var h = (this.constructor || l).prototype;
                    return b in this&&!(b in h && this[b] === h[b])
                });
                h = null;
                return A.call(this, b)
            });
            H = function(b, h) {
                var l = 0, c, d, a;
                (c = function() {
                    this.valueOf = 0
                }).prototype.valueOf = 0;
                d = new c;
                for (a in d)
                    A.call(d, a) && l++;
                c = d = null;
                l ? H = 2 == l ? function(b, h) {
                    var l = {}, d = "[object Function]" == z.call(b), c;
                    for (c in b)
                        d && "prototype" == c || A.call(l, c) ||!(l[c] = 1) ||!A.call(b, c) || h(c)
                } : function(b, h) {
                    var l = "[object Function]" == z.call(b), c, d;
                    for (c in b)
                        l && "prototype" ==
                        c ||!A.call(b, c) || (d = "constructor" === c) || h(c);
                    (d || A.call(b, c = "constructor")) && h(c)
                } : (d = "valueOf toString toLocaleString propertyIsEnumerable isPrototypeOf hasOwnProperty constructor".split(" "), H = function(b, h) {
                    var l = "[object Function]" == z.call(b), c, a=!l && "function" != typeof b.constructor && q[typeof b.hasOwnProperty] && b.hasOwnProperty || A;
                    for (c in b)
                        l && "prototype" == c ||!a.call(b, c) || h(c);
                    for (l = d.length; c = d[--l]; a.call(b, c) 
                        && h(c));
                });
                return H(b, h)
            };
            if (!g("json-stringify")) {
                var R = {
                    92: "\\\\",
                    34: '\\"',
                    8: "\\b",
                    12: "\\f",
                    10: "\\n",
                    13: "\\r",
                    9: "\\t"
                }, F = function(b, h) {
                    return ("000000" + (h || 0)).slice( - b)
                }, N = function(b) {
                    for (var h = '"', c = 0, l = b.length, d=!K || 10 < l, a = d && (K ? b.split("") : b); c < l; c++) {
                        var e = b.charCodeAt(c);
                        switch (e) {
                        case 8:
                        case 9:
                        case 10:
                        case 12:
                        case 13:
                        case 34:
                        case 92:
                            h += R[e];
                            break;
                        default:
                            if (32 > e) {
                                h += "\\u00" + F(2, e.toString(16));
                                break
                            }
                            h += d ? a[c] : b.charAt(c)
                        }
                    }
                    return h + '"'
                }, p = function(b, h, c, l, d, a, e) {
                    var k, g, f, m, t, n, v, s, q;
                    try {
                        k = h[b]
                    } catch (O) {}
                    if ("object" == typeof k && k)
                        if (g = z.call(k), "[object Date]" != g || A.call(k,
                        "toJSON"))
                            "function" == typeof k.toJSON && ("[object Number]" != g && "[object String]" != g && "[object Array]" != g || A.call(k, "toJSON")) && (k = k.toJSON(b));
                        else if (k>-1 / 0 && k < 1 / 0) {
                            if (x) {
                                m = C(k / 864E5);
                                for (g = C(m / 365.2425) + 1970 - 1; x(g + 1, 0) <= m; g++);
                                for (f = C((m - x(g, 0)) / 30.42); x(g, f + 1) <= m; f++);
                                m = 1 + m - x(g, f);
                                t = (k%864E5 + 864E5)%864E5;
                                n = C(t / 36E5)%24;
                                v = C(t / 6E4)%60;
                                s = C(t / 1E3)%60;
                                t%=1E3
                            } else 
                                g = k.getUTCFullYear(), f = k.getUTCMonth(), m = k.getUTCDate(), n = k.getUTCHours(), v = k.getUTCMinutes(), s = k.getUTCSeconds(), t = k.getUTCMilliseconds();
                                k = (0 >= g || 1E4 <= g ? (0 > g ? "-" : "+") + F(6, 0 > g?-g : g) : F(4, g)) + "-" + F(2, f + 1) + "-" + F(2, m) + "T" + F(2, n) + ":" + F(2, v) + ":" + F(2, s) + "." + F(3, t) + "Z"
                        } else 
                            k = null;
                    c && (k = c.call(h, b, k));
                    if (null === k)
                        return "null";
                    g = z.call(k);
                    if ("[object Boolean]" == g)
                        return "" + k;
                    if ("[object Number]" == g)
                        return k>-1 / 0 && k < 1 / 0 ? "" + k : "null";
                    if ("[object String]" == g)
                        return N("" + k);
                    if ("object" == typeof k) {
                        for (b = e.length; b--;)
                            if (e[b] === k)
                                throw y();
                        e.push(k);
                        q = [];
                        h = a;
                        a += d;
                        if ("[object Array]" == g) {
                            f = 0;
                            for (b = k.length; f < b; f++)
                                g = p(f, k, c, l, d, a, e), q.push(g === D ? "null" :
                                g);
                            b = q.length ? d ? "[\n" + a + q.join(",\n" + a) + "\n" + h + "]" : "[" + q.join(",") + "]" : "[]"
                        } else 
                            H(l || k, function(b) {
                                var h = p(b, k, c, l, d, a, e);
                                h !== D && q.push(N(b) + ":" + (d ? " " : "") + h)
                            }), b = q.length ? d ? "{\n" + a + q.join(",\n" + a) + "\n" + h + "}" : "{" + q.join(",") + "}" : "{}";
                        e.pop();
                        return b
                    }
                };
                e.stringify = function(b, h, c) {
                    var l, d, a, k;
                    if (q[typeof h] && h)
                        if ("[object Function]" == (k = z.call(h)))
                            d = h;
                        else if ("[object Array]" == k) {
                            a = {};
                            for (var e = 0, g = h.length, f; e < g; f = h[e++], (k = z.call(f), "[object String]" == k || "[object Number]" == k) && (a[f] = 1));
                        }
                    if (c)
                        if ("[object Number]" ==
                        (k = z.call(c))) {
                            if (0 < (c -= c%1))
                                for (l = "", 10 < c && (c = 10); l.length < c; l += " ");
                        } else 
                            "[object String]" == k && (l = 10 >= c.length ? c : c.slice(0, 10));
                    return p("", (f = {}, f[""] = b, f), d, a, l, "", [])
                }
            }
            if (!g("json-parse")) {
                var b = n.fromCharCode, h = {
                    92: "\\",
                    34: '"',
                    47: "/",
                    98: "\b",
                    116: "\t",
                    110: "\n",
                    102: "\f",
                    114: "\r"
                }, l, O, v = function() {
                    l = O = null;
                    throw J();
                }, L = function() {
                    for (var c = O, d = c.length, a, k, e, g, p; l < d;)
                        switch (p = c.charCodeAt(l), p) {
                        case 9:
                        case 10:
                        case 13:
                        case 32:
                            l++;
                            break;
                        case 123:
                        case 125:
                        case 91:
                        case 93:
                        case 58:
                        case 44:
                            return a =
                            K ? c.charAt(l) : c[l], l++, a;
                        case 34:
                            a = "@";
                            for (l++; l < d;)
                                if (p = c.charCodeAt(l), 32 > p)
                                    v();
                                else if (92 == p)
                                    switch (p = c.charCodeAt(++l), p) {
                                    case 92:
                                    case 34:
                                    case 47:
                                    case 98:
                                    case 116:
                                    case 110:
                                    case 102:
                                    case 114:
                                        a += h[p];
                                        l++;
                                        break;
                                    case 117:
                                        k=++l;
                                        for (e = l + 4; l < e; l++)
                                            p = c.charCodeAt(l), 48 <= p && 57 >= p || 97 <= p && 102 >= p || 65 <= p && 70 >= p || v();
                                            a += b("0x" + c.slice(k, l));
                                            break;
                                        default:
                                            v()
                                    } else {
                                        if (34 == p)
                                            break;
                                            p = c.charCodeAt(l);
                                            for (k = l; 32 <= p && 92 != p && 34 != p;)
                                                p = c.charCodeAt(++l);
                                                a += c.slice(k, l)
                                            }
                                            if (34 == c.charCodeAt(l))
                                                return l++, a;
                                                v();
                                            default:
                                                k =
                                                l;
                                                45 == p && (g=!0, p = c.charCodeAt(++l));
                                                if (48 <= p && 57 >= p) {
                                                    for (48 == p && (p = c.charCodeAt(l + 1), 48 <= p && 57 >= p) && v(); l < d && (p = c.charCodeAt(l), 48 <= p && 57 >= p); l++);
                                                    if (46 == c.charCodeAt(l)) {
                                                        for (e=++l; e < d && (p = c.charCodeAt(e), 48 <= p && 57 >= p); e++);
                                                        e == l && v();
                                                        l = e
                                                    }
                                                    p = c.charCodeAt(l);
                                                    if (101 == p || 69 == p) {
                                                        p = c.charCodeAt(++l);
                                                        43 != p && 45 != p || l++;
                                                        for (e = l; e < d && (p = c.charCodeAt(e), 48 <= p && 57 >= p); e++);
                                                        e == l && v();
                                                        l = e
                                                    }
                                                    return + c.slice(k, l)
                                                }
                                                g && v();
                                                if ("true" == c.slice(l, l + 4))
                                                    return l += 4, !0;
                                                    if ("false" == c.slice(l, l + 5))
                                                        return l += 5, !1;
                                                        if ("null" == c.slice(l,
                                                        l + 4))
                                                            return l += 4, null;
                                                            v()
                                            }
                    return "$"
                }, S = function(b) {
                    var h, c;
                    "$" == b && v();
                    if ("string" == typeof b) {
                        if ("@" == (K ? b.charAt(0) : b[0]))
                            return b.slice(1);
                        if ("[" == b) {
                            for (h = []; ; c || (c=!0)
                                ) {
                                b = L();
                                if ("]" == b)
                                    break;
                                c && ("," == b ? (b = L(), "]" == b && v()) : v());
                                "," == b && v();
                                h.push(S(b))
                            }
                            return h
                        }
                        if ("{" == b) {
                            for (h = {}; ; c || (c=!0)
                                ) {
                                b = L();
                                if ("}" == b)
                                    break;
                                c && ("," == b ? (b = L(), "}" == b && v()) : v());
                                "," != b && "string" == typeof b && "@" == (K ? b.charAt(0) : b[0]) && ":" == L() || v();
                                h[b.slice(1)] = S(L())
                            }
                            return h
                        }
                        v()
                    }
                    return b
                }, U = function(b, h, c) {
                    c = T(b, h, c);
                    c ===
                    D ? delete b[h] : b[h] = c
                }, T = function(b, h, c) {
                    var l = b[h], d;
                    if ("object" == typeof l && l)
                        if ("[object Array]" == z.call(l))
                            for (d = l.length; d--;)
                                U(l, d, c);
                        else 
                            H(l, function(b) {
                                U(l, b, c)
                            });
                    return c.call(b, h, l)
                };
                e.parse = function(b, h) {
                    var c, d;
                    l = 0;
                    O = "" + b;
                    c = S(L());
                    "$" != L() && v();
                    l = O = null;
                    return h && "[object Function]" == z.call(h) ? T((d = {}, d[""] = c, d), "", h) : c
                }
            }
        }
        e.runInContext = f;
        return e
    }
    var a = "function" === typeof define && define.amd, q = {
        "function": !0,
        object: !0
    }, n = q[typeof exports] && exports&&!exports.nodeType && exports, d = q[typeof window] &&
    window || this, e = n && q[typeof module] && module&&!module.nodeType && "object" == typeof global && global;
    !e || e.global !== e && e.window !== e && e.self !== e || (d = e);
    if (n&&!a)
        f(d, n);
    else {
        var g = d.JSON, m = d.JSON3, s=!1, r = f(d, d.JSON3 = {
            noConflict: function() {
                s || (s=!0, d.JSON = g, d.JSON3 = m, g = m = null);
                return r
            }
        });
        d.JSON = {
            parse: r.parse,
            stringify: r.stringify
        }
    }
    a && define(function() {
        return r
    })
}).call(this);
(function(f) {
    "function" === typeof define && define.amd ? define(["jquery"], f) : "object" === typeof exports ? f(require("jquery")) : f(jimdoGen002)
})(function(f) {
    function a(a) {
        return d.raw ? a : encodeURIComponent(a)
    }
    function q(a, g) {
        var m;
        if (d.raw)
            m = a;
        else 
            a: {
            var s = a;
            0 === s.indexOf('"') && (s = s.slice(1, - 1).replace(/\\"/g, '"').replace(/\\\\/g, "\\"));
            try {
                s = decodeURIComponent(s.replace(n, " "));
                m = d.json ? JSON.parse(s) : s;
                break a
            } catch (q) {}
            m = void 0
        }
        return f.isFunction(g) ? g(m) : m
    }
    var n = /\+/g, d = f.cookie = function(e, g, m) {
        if (void 0 !==
        g&&!f.isFunction(g)
            ) {
            m = f.extend({}, d.defaults, m);
            if ("number" === typeof m.expires) {
                var n = m.expires, r = m.expires = new Date;
                r.setTime( + r + 864E5 * n)
            }
            return document.cookie = [a(e), "\x3d", a(d.json ? JSON.stringify(g) : String(g)), m.expires ? "; expires\x3d" + m.expires.toUTCString(): "", m.path ? "; path\x3d" + m.path: "", m.domain ? "; domain\x3d" + m.domain: "", m.secure ? "; secure": ""].join("")
        }
        m = e ? void 0 : {};
        for (var n = document.cookie ? document.cookie.split("; ") : [], r = 0, k = n.length; r < k; r++) {
            var t = n[r].split("\x3d"), w;
            w = t.shift();
            w =
            d.raw ? w : decodeURIComponent(w);
            t = t.join("\x3d");
            if (e && e === w) {
                m = q(t, g);
                break
            }
            e || void 0 === (t = q(t)) || (m[w] = t)
        }
        return m
    };
    d.defaults = {};
    f.removeCookie = function(d, a) {
        if (void 0 === f.cookie(d)
            )return !1;
        f.cookie(d, "", f.extend({}, a, {
            expires: - 1
        }));
        return !f.cookie(d)
    }
});
(function(f) {
    var a = f.Base64, q;
    "undefined" !== typeof module && module.exports && (q = require("buffer").Buffer);
    var n = function(c) {
        for (var d = {}, a = 0, k = c.length; a < k; a++)
            d[c.charAt(a)] = a;
        return d
    }("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"), d = String.fromCharCode, e = function(c) {
        if (2 > c.length) {
            var a = c.charCodeAt(0);
            return 128 > a ? c : 2048 > a ? d(192 | a>>>6) + d(128 | a & 63) : d(224 | a>>>12 & 15) + d(128 | a>>>6 & 63) + d(128 | a & 63)
        }
        a = 65536 + 1024 * (c.charCodeAt(0) - 55296) + (c.charCodeAt(1) - 56320);
        return d(240 | a>>>18 & 7) +
        d(128 | a>>>12 & 63) + d(128 | a>>>6 & 63) + d(128 | a & 63)
    }, g = /[\uD800-\uDBFF][\uDC00-\uDFFFF]|[^\x00-\x7F]/g, m = function(c) {
        return c.replace(g, e)
    }, s = function(c) {
        var a = [0, 2, 1][c.length%3];
        c = c.charCodeAt(0)<<16 | (1 < c.length ? c.charCodeAt(1) : 0)<<8 | (2 < c.length ? c.charCodeAt(2) : 0);
        return ["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".charAt(c>>>18), "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".charAt(c>>>12 & 63), 2 <= a ? "\x3d": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".charAt(c>>>
        6 & 63), 1 <= a ? "\x3d": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".charAt(c & 63)].join("")
    }, r = f.btoa ? function(c) {
        return f.btoa(c)
    }
    : function(c) {
        return c.replace(/[\s\S]{1,3}/g, s)
    }, k = q ? function(c) {
        return (new q(c)).toString("base64")
    }
    : function(c) {
        return r(m(c))
    }, t = function(c, a) {
        return a ? k(c).replace(/[+\/]/g, function(c) {
            return "+" == c ? "-" : "_"
        }).replace(/=/g, "") : k(c)
    }, w = /[\u00c0-\u00df][\u0080-\u00bf]|[\u00e0-\u00ef][\u0080-\u00bf]{2}|[\u00f0-\u00f7][\u0080-\u00bf]{3}/g, B = function(c) {
        switch (c.length) {
        case 4:
            return c =
            ((7 & c.charCodeAt(0))<<18 | (63 & c.charCodeAt(1))<<12 | (63 & c.charCodeAt(2))<<6 | 63 & c.charCodeAt(3)) - 65536, d((c>>>10) + 55296) + d((c & 1023) + 56320);
        case 3:
            return d((15 & c.charCodeAt(0))<<12 | (63 & c.charCodeAt(1))<<6 | 63 & c.charCodeAt(2));
        default:
            return d((31 & c.charCodeAt(0))<<6 | 63 & c.charCodeAt(1))
        }
    }, u = function(c) {
        return c.replace(w, B)
    }, M = function(c) {
        var a = c.length, k = a%4;
        c = (0 < a ? n[c.charAt(0)]<<18 : 0) | (1 < a ? n[c.charAt(1)]<<12 : 0) | (2 < a ? n[c.charAt(2)]<<6 : 0) | (3 < a ? n[c.charAt(3)] : 0);
        c = [d(c>>>16), d(c>>>8 & 255), d(c & 255)];
        c.length -=
        [0, 0, 2, 1][k];
        return c.join("")
    }, I = f.atob ? function(c) {
        return f.atob(c)
    }
    : function(c) {
        return c.replace(/[\s\S]{1,4}/g, M)
    }, J = q ? function(c) {
        return (new q(c, "base64")).toString()
    }
    : function(c) {
        return u(I(c))
    }, y = function(c) {
        return J(c.replace(/[-_]/g, function(c) {
            return "-" == c ? "+" : "/"
        }).replace(/[^A-Za-z0-9\+\/]/g, ""))
    };
    f.Base64 = {
        VERSION: "2.1.5",
        atob: I,
        btoa: r,
        fromBase64: y,
        toBase64: t,
        utob: m,
        encode: t,
        encodeURI: function(c) {
            return t(c, !0)
        },
        btou: u,
        decode: y,
        noConflict: function() {
            var c = f.Base64;
            f.Base64 = a;
            return c
        }
    };
    if ("function" === typeof Object.defineProperty) {
        var G = function(c) {
            return {
                value: c,
                enumerable: !1,
                writable: !0,
                configurable: !0
            }
        };
        f.Base64.extendString = function() {
            Object.defineProperty(String.prototype, "fromBase64", G(function() {
                return y(this)
            }));
            Object.defineProperty(String.prototype, "toBase64", G(function(c) {
                return t(this, c)
            }));
            Object.defineProperty(String.prototype, "toBase64URI", G(function() {
                return t(this, !0)
            }))
        }
    }
})(this);
this.Meteor && (Base64 = global.Base64);
(function(f, a, q) {
    "undefined" !== typeof module && module.exports ? module.exports = q() : "function" === typeof define && define.amd ? define(q) : a[f] = q()
})("Fingerprint", this, function() {
    var f = function(a) {
        var f, n;
        f = Array.prototype.forEach;
        n = Array.prototype.map;
        this.each = function(a, e, g) {
            if (null !== a)
                if (f && a.forEach === f)
                    a.forEach(e, g);
                else if (a.length ===+ a.length)
                    for (var m = 0, n = a.length; m < n && e.call(g, a[m], m, a) !== {}; m++);
                else 
                    for (m in a)
                        if (a.hasOwnProperty(m) && e.call(g, a[m], m, a) === {})
                            break
        };
        this.map = function(a, e, g) {
            var f =
            [];
            if (null == a)
                return f;
            if (n && a.map === n)
                return a.map(e, g);
            this.each(a, function(a, d, k) {
                f[f.length] = e.call(g, a, d, k)
            });
            return f
        };
        "object" == typeof a ? (this.hasher = a.hasher, this.screen_resolution = a.screen_resolution, this.canvas = a.canvas, this.ie_activex = a.ie_activex) : "function" == typeof a && (this.hasher = a)
    };
    f.prototype = {
        get: function() {
            var a = [];
            a.push(navigator.userAgent);
            a.push(navigator.language);
            a.push(screen.colorDepth);
            this.screen_resolution && "undefined" !== typeof this.getScreenResolution() && a.push(this.getScreenResolution().join("x"));
            a.push((new Date).getTimezoneOffset());
            a.push(this.hasSessionStorage());
            a.push(this.hasLocalStorage());
            a.push(!!window.indexedDB);
            document.body ? a.push(typeof document.body.addBehavior) : a.push("undefined");
            a.push(typeof window.openDatabase);
            a.push(navigator.cpuClass);
            a.push(navigator.platform);
            a.push(navigator.doNotTrack);
            a.push(this.getPluginsString());
            this.canvas && this.isCanvasSupported() && a.push(this.getCanvasFingerprint());
            return this.hasher ? this.hasher(a.join("###"), 31) : this.murmurhash3_32_gc(a.join("###"),
            31)
        },
        murmurhash3_32_gc: function(a, f) {
            var n, d, e, g, m;
            n = a.length & 3;
            d = a.length - n;
            e = f;
            for (m = 0; m < d;)
                g = a.charCodeAt(m) & 255 | (a.charCodeAt(++m) & 255)<<8 | (a.charCodeAt(++m) & 255)<<16 | (a.charCodeAt(++m) & 255)<<24, ++m, g = 3432918353 * (g & 65535) + ((3432918353 * (g>>>16) & 65535)<<16) & 4294967295, g = g<<15 | g>>>17, g = 461845907 * (g & 65535) + ((461845907 * (g>>>16) & 65535)<<16) & 4294967295, e^=g, e = e<<13 | e>>>19, e = 5 * (e & 65535) + ((5 * (e>>>16) & 65535)<<16) & 4294967295, e = (e & 65535) + 27492 + (((e>>>16) + 58964 & 65535)<<16);
            g = 0;
            switch (n) {
            case 3:
                g^=(a.charCodeAt(m +
                2) & 255)<<16;
            case 2:
                g^=(a.charCodeAt(m + 1) & 255)<<8;
            case 1:
                g^=a.charCodeAt(m) & 255, g = 3432918353 * (g & 65535) + ((3432918353 * (g>>>16) & 65535)<<16) & 4294967295, g = g<<15 | g>>>17, e^=461845907 * (g & 65535) + ((461845907 * (g>>>16) & 65535)<<16) & 4294967295
            }
            e^=a.length;
            e^=e>>>16;
            e = 2246822507 * (e & 65535) + ((2246822507 * (e>>>16) & 65535)<<16) & 4294967295;
            e^=e>>>13;
            e = 3266489909 * (e & 65535) + ((3266489909 * (e>>>16) & 65535)<<16) & 4294967295;
            return (e^e>>>16)>>>0
        },
        hasLocalStorage: function() {
            try {
                return !!window.localStorage
            } catch (a) {
                return !0
            }
        },
        hasSessionStorage: function() {
            try {
                return !!window.sessionStorage
            } catch (a) {
                return !0
            }
        },
        isCanvasSupported: function() {
            var a = document.createElement("canvas");
            return !(!a.getContext ||!a.getContext("2d"))
        },
        isIE: function() {
            return "Microsoft Internet Explorer" === navigator.appName || "Netscape" === navigator.appName && /Trident/.test(navigator.userAgent)?!0 : !1
        },
        getPluginsString: function() {
            return this.isIE() && this.ie_activex ? this.getIEPluginsString() : this.getRegularPluginsString()
        },
        getRegularPluginsString: function() {
            return this.map(navigator.plugins, function(a) {
                var f = this.map(a, function(a) {
                    return [a.type,
                    a.suffixes].join("~")
                }).join(",");
                return [a.name, a.description, f].join("::")
            }, this).join(";")
        },
        getIEPluginsString: function() {
            return window.ActiveXObject ? this.map("ShockwaveFlash.ShockwaveFlash;AcroPDF.PDF;PDF.PdfCtrl;QuickTime.QuickTime;rmocx.RealPlayer G2 Control;rmocx.RealPlayer G2 Control.1;RealPlayer.RealPlayer(tm) ActiveX Control (32-bit);RealVideo.RealVideo(tm) ActiveX Control (32-bit);RealPlayer;SWCtl.SWCtl;WMPlayer.OCX;AgControl.AgControl;Skype.Detection".split(";"), function(a) {
                try {
                    return new ActiveXObject(a),
                    a
                } catch (f) {
                    return null
                }
            }).join(";") : ""
        },
        getScreenResolution: function() {
            return [screen.height, screen.width]
        },
        getCanvasFingerprint: function() {
            var a = document.createElement("canvas"), f = a.getContext("2d");
            f.textBaseline = "top";
            f.font = "14px 'Arial'";
            f.textBaseline = "alphabetic";
            f.fillStyle = "#f60";
            f.fillRect(125, 1, 62, 20);
            f.fillStyle = "#069";
            f.fillText("http://valve.github.io", 2, 15);
            f.fillStyle = "rgba(102, 204, 0, 0.7)";
            f.fillText("http://valve.github.io", 4, 17);
            return a.toDataURL()
        }
    };
    return f
});
(function(f) {
    var a = decodeURIComponent, q = f.deparam = function(n, d) {
        var e = {};
        f.each(n.replace(/\+/g, " ").split("\x26"), function(g, m) {
            var n = m.split("\x3d"), r = a(n[0]);
            if (r) {
                var n = a(n[1] || ""), k = r.split("]["), t = k.length - 1, w = 0, B = e;
                0 <= k[0].indexOf("[") && /\]$/.test(k[t]) ? (k[t] = k[t].replace(/\]$/, ""), k = k.shift().split("[").concat(k), t++) : t = 0;
                f.isFunction(d) ? n = d(r, n) : d && (n = q.reviver(r, n));
                if (t)
                    for (; w <= t; w++)
                        r = "" !== k[w] ? k[w] : B.length, w < t ? B = B[r] = B[r] || (isNaN(k[w + 1]) ? {} : []) : B[r] = n;
                else 
                    f.isArray(e[r]) ? e[r].push(n) :
                    e[r] = r in e ? [e[r], n] : n
            }
        });
        return e
    };
    q.reviver = function(a, d) {
        var e = {
            "true": !0,
            "false": !1,
            "null": null,
            undefined: void 0
        };
        return + d + "" === d?+d : d in e ? e[d] : d
    }
})(jimdoGen002);
(function(f, a) {
    "function" === typeof define && define.amd ? define(["exports"], a) : "object" === typeof exports ? a(exports) : a(f.PubSub = {})
})("object" === typeof window && window || this, function(f) {
    function a(a) {
        for (var d in a)
            if (a.hasOwnProperty(d))
                return !0;
        return !1
    }
    function q(a) {
        return function() {
            throw a;
        }
    }
    function n(a, d, e) {
        try {
            a(d, e)
        } catch (f) {
            setTimeout(q(f), 0)
        }
    }
    function d(a, d, e) {
        a(d, e)
    }
    function e(a, e, f, g) {
        var m = s[e];
        g = g ? d : n;
        var q;
        if (s.hasOwnProperty(e))
            for (q in m)
                m.hasOwnProperty(q) && g(m[q], a, f)
    }
    function g(a,
    d, f) {
        return function() {
            var g = String(a), n = g.lastIndexOf(".");
            for (e(a, a, d, f);
            - 1 !== n;
            )g = g.substr(0, n), n = g.lastIndexOf("."), e(a, g, d)
        }
    }
    function m(d, e, f, n) {
        e = g(d, e, n);
        d = String(d);
        n = Boolean(s.hasOwnProperty(d) && a(s[d]));
        for (var m = d.lastIndexOf("."); !n&&-1 !== m;)
            d = d.substr(0, m), m = d.lastIndexOf("."), n = Boolean(s.hasOwnProperty(d) && a(s[d]));
        if (!n)
            return !1;
        !0 === f ? e() : setTimeout(e, 0);
        return !0
    }
    var s = {}, r =- 1;
    f.publish = function(a, d) {
        return m(a, d, !1, f.immediateExceptions)
    };
    f.publishSync = function(a, d) {
        return m(a, d,
        !0, f.immediateExceptions)
    };
    f.subscribe = function(a, d) {
        if ("function" !== typeof d)
            return !1;
        s.hasOwnProperty(a) || (s[a] = {});
        var e = "uid_" + String(++r);
        s[a][e] = d;
        return e
    };
    f.clearAllSubscriptions = function() {
        s = {}
    };
    f.unsubscribe = function(a) {
        var d = "string" === typeof a && s.hasOwnProperty(a), e=!d && "string" === typeof a, g = "function" === typeof a, f=!1, n, m;
        if (d)
            delete s[a];
        else {
            for (n in s)
                if (s.hasOwnProperty(n))
                    if (d = s[n], e && d[a]) {
                        delete d[a];
                        f = a;
                        break
                    } else if (g)
                        for (m in d)
                            d.hasOwnProperty(m) && d[m] === a && (delete d[m], f=!0);
            return f
        }
    }
});
(function() {
    var f = {}.hasOwnProperty, a = function(a, d) {
        function e() {
            this.constructor = a
        }
        for (var g in d)
            f.call(d, g) && (a[g] = d[g]);
        e.prototype = d.prototype;
        a.prototype = new e;
        a.__super__ = d.prototype;
        return a
    }, q = function(a, d) {
        return function() {
            return a.apply(d, arguments)
        }
    };
    (function(f) {
        return function(a, e, f) {
            return "object" === typeof exports && "object" === typeof module ? module.exports = f() : "function" === typeof define && define.amd ? define(f) : a[e] = f(a, a[e])
        }("object" === typeof f ? f : this, "jimdom", function(d, e) {
            var g, m, s,
            r, k, t, w, B, u, M, I, J, y, G, c, z, A, H, D, E, P, K, C, Q, x, R, F, N;
            r = function() {
                function a(b) {
                    this.topic = b
                }
                a.BASE = "jimdom";
                a.prototype.getTopic = function(b) {
                    var h;
                    h = this.topic ? "." + this.topic : "";
                    return "" + a.BASE + h + (b ? "." + b : "")
                };
                return a
            }();
            A = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                b.prototype.send = function(b, a) {
                    PubSub.publish(this.getTopic(a), b)
                };
                return b
            }(r);
            c = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                b.prototype.tell = function(b,
                a) {
                    if (!(b instanceof y))
                        throw {
                            message: "first argument must be of type Historian",
                            type: "IllegalArgumentException"
                        };
                    PubSub.subscribe(this.getTopic(a), function(a, c) {
                        a = a.substr(("" + r.BASE + ".").length);
                        b.ear(a, c)
                    })
                };
                return b
            }(r);
            y = function() {
                function a() {
                    this.ear = q(this.ear, this);
                    this.stories = []
                }
                a.TOPIC_KEY = "t";
                a.EVENT_TSTAMP_KEY = "d";
                a.MESSAGE_KEY = "m";
                a.EVENT_ID_KEY = "i";
                a.USER_DATA_KEY = "u";
                a.STORY_KEY = "s";
                a.prototype.stories = null;
                a.prototype.addStory = function(b) {
                    if (!(b instanceof x))
                        throw {
                            message: "first argument must be of type Story",
                            type: "IllegalArgumentException"
                        };
                    if (b)
                        return this.stories.push(b)
                };
                a.prototype.prepare = function(b, h) {
                    var c, d, e, f, g, k;
                    c = {};
                    if (this.stories.length)
                        for (c[a.STORY_KEY] = {}, g = this.stories, e = 0, f = g.length; e < f; e++)
                            d = g[e], d.hasData() && (c[a.STORY_KEY][d.getType()] = d.getData());
                    c[a.EVENT_ID_KEY] = F.generate();
                    c[a.TOPIC_KEY] = b;
                    c[a.MESSAGE_KEY] = h;
                    c[a.EVENT_TSTAMP_KEY] = null != (k = "function" === typeof Date.now ? Date.now() : void 0) ? k : (new Date).getTime();
                    return c
                };
                a.prototype.ear = function(b, a) {
                    this.write(this.prepare(b,
                    a))
                };
                a.prototype.write = function(b) {
                    throw {
                        message: "This method is not implemented",
                        type: "NotImplementedException"
                    };
                };
                return a
            }();
            D = function(c) {
                function b() {}
                a(b, c);
                b.prototype.addStory = function() {};
                b.prototype.prepare = function() {};
                b.prototype.ear = function() {};
                b.prototype.write = function() {};
                return b
            }(y);
            k = function(c) {
                function b(a) {
                    this.write = a;
                    b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                return b
            }(y);
            w = function(c) {
                function b(a) {
                    this.prefix = a;
                    this.write = q(this.write, this);
                    b.__super__.constructor.apply(this,
                    arguments)
                }
                a(b, c);
                b.prototype.write = function(b) {
                    this.prefix ? console.log(this.prefix, b) : console.log(b)
                };
                return b
            }(y);
            u = function(c) {
                function b(a) {
                    this.url = a;
                    b.__super__.constructor.apply(this, arguments);
                    if (!this.url)
                        throw {
                            message: "first argument is mandatory",
                            type: "MissingArgumentException"
                        };
                }
                a(b, c);
                b.prototype.abort = function() {
                    throw {
                        message: "This method is not implemented",
                        type: "NotImplementedException"
                    };
                };
                return b
            }(y);
            C = function() {
                function a() {}
                a.prototype.serialize = function(b, a) {
                    a(b)
                };
                return a
            }();
            P = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                b.prototype.serialize = function(b, a) {
                    a(null, d.param(b))
                };
                return b
            }(C);
            s = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                b.prototype.serialize = function(b, a) {
                    var c;
                    try {
                        c = encodeURIComponent(Base64.encode(JSON.stringify(b))), a(null, c)
                    } catch (d) {
                        a(d)
                    }
                };
                return b
            }(C);
            E = function(c) {
                function b(a, c, d) {
                    this.url = a;
                    this.$target = c;
                    null == d && (d = P);
                    b.__super__.constructor.call(this, this.url);
                    "function" ===
                    typeof d && (this.serializer = new d)
                }
                a(b, c);
                b.MAX_PIXELS = 10;
                b.prototype.getTarget = function() {
                    null == this.$target && (this.$target = d('\x3cdiv style\x3d"display: none !important; position: absolute; left: -1000px; top: -1000px;"\x3e').appendTo(d("body")));
                    return this.$target
                };
                b.prototype.clean = function() {
                    var a;
                    a = this.getTarget().children();
                    a.length > b.MAX_PIXELS && a.slice(0, b.MAX_PIXELS).remove()
                };
                b.prototype.write = function(b) {
                    this.serializer.serialize(b, function(b) {
                        return function(a, c) {
                            var h;
                            if (a)
                                throw a;
                            h = "" + b.url + "?" + c;
                            b.clean();
                            b.getTarget().append(d("\x3cimg src\x3d'" + h + "'\x3e"))
                        }
                    }(this))
                };
                b.prototype.abort = function() {
                    this.getTarget().empty()
                };
                return b
            }(u);
            m = function(c) {
                function b() {
                    b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                b.prototype.xhrs = [];
                b.prototype.write = function(b) {
                    this.xhrs.push(d.ajax({
                        url: this.url,
                        data: b,
                        type: "POST"
                    }))
                };
                b.prototype.abort = function() {
                    for (var b; b = this.xhrs.shift();)
                        b.abort()
                };
                return b
            }(u);
            I = function(c) {
                function b(a, c, e) {
                    this.historian = a;
                    this.maxEvents = null !=
                    c ? c : b.GATHER_EVENTS;
                    this.cleanBeforeUnload = null != e ? e : !0;
                    this.push = q(this.push, this);
                    if (!this.historian)
                        throw {
                            message: "first argument is mandatory",
                            type: "MissingArgumentException"
                        };
                    if (!(this.historian instanceof y))
                        throw {
                            message: "first argument must be of type Historian",
                            type: "IllegalArgumentException"
                        };
                    b.__super__.constructor.apply(this, arguments);
                    this.cache = [];
                    d(f).on("beforeunload", function(b) {
                        return function() {
                            b.cleanBeforeUnload ? b.push(!0) : b.abort()
                        }
                    }(this))
                }
                a(b, c);
                b.GATHER_EVENTS = 5;
                b.prototype.cache =
                null;
                b.prototype.push = function(b) {
                    if (this.cache.length && (this.cache.length >= this.maxEvents || b))
                        return b = this.cache.splice(0, this.maxEvents), this.historian.write(this.prepare("gathered", b)), b.length
                };
                b.prototype.write = function(b) {
                    this.cache.push(b);
                    this.push()
                };
                return b
            }(y);
            B = function() {
                function a(b, c) {
                    this.historian = b;
                    this.interval = null != c ? c : a.INTERVAL_SECONDS;
                    this.whip = q(this.whip, this);
                    if (!(this.historian instanceof I))
                        throw {
                            message: "first argument must be of type GatheringHistorian",
                            type: "IllegalArgumentException"
                        };
                    setInterval(this.whip, 1E3 * this.interval)
                }
                a.INTERVAL_SECONDS = 10;
                a.prototype.last = null;
                a.prototype.whip = function() {
                    this.last = new Date;
                    this.historian.push(!0)
                };
                return a
            }();
            F = function() {
                function a() {}
                a.generate = function() {
                    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(b) {
                        var a;
                        a = 16 * Math.random() | 0;
                        return ("x" === b ? a : a & 3 | 8).toString(16)
                    })
                };
                return a
            }();
            x = function() {
                function a() {}
                a.prototype.getWindow = function() {
                    return f
                };
                a.prototype.getDocument = function() {
                    return e
                };
                a.prototype.getNavigator =
                function() {
                    return f.navigator
                };
                a.prototype.getData = function() {
                    throw {
                        message: "This method is not implemented",
                        type: "NotImplementedException"
                    };
                };
                a.prototype.hasData = function() {
                    return !0
                };
                a.prototype.getType = function() {
                    throw {
                        message: "This method is not implemented",
                        type: "NotImplementedException"
                    };
                };
                return a
            }();
            Q = function(c) {
                function b(a) {
                    this.options = null != a ? a : {};
                    b.__super__.constructor.apply(this, arguments);
                    this.id = this._setOrRenew()
                }
                a(b, c);
                b.STORAGE_KEY = "shd";
                b.STORY_TYPE = "shd";
                b.DEFAULTS = {
                    expires: 365,
                    path: "/"
                };
                b.prototype._getOptions = function() {
                    return d.extend({}, b.DEFAULTS, this.options)
                };
                b.prototype._get = function() {
                    var a;
                    return null != (a = d.cookie(b.STORAGE_KEY)) ? a : null
                };
                b.prototype._setOrRenew = function() {
                    var a, c;
                    a = null != (c = this._get()) ? c : F.generate();
                    d.cookie(b.STORAGE_KEY, a, this._getOptions());
                    return a
                };
                b.prototype.detach = function() {
                    d.removeCookie(b.STORAGE_KEY, this._getOptions())
                };
                b.prototype.getData = function() {
                    return this.id
                };
                b.prototype.getType = function() {
                    return b.STORY_TYPE
                };
                return b
            }(x);
            g = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                b.STORY_TYPE = "aff";
                b.REGEX_HASH = /ref=a?(\d+)/;
                b.REGEX_QUERY = /[?|&]ref=a?(\d+)/;
                b.COOKIE_NAME = "jimdoRefAffiliate";
                b.prototype._hasDataInHash = function() {
                    return b.REGEX_HASH.test(this.getWindow().location.hash)
                };
                b.prototype._getDataFromHash = function() {
                    var a;
                    a = this.getWindow().location.hash.match(b.REGEX_HASH);
                    return 0 < (null != a ? a.length : void 0) ? a[1] : null
                };
                b.prototype._hasDataInQuery = function() {
                    return b.REGEX_QUERY.test(this.getWindow().location.search)
                };
                b.prototype._getDataFromQuery = function() {
                    var a;
                    a = this.getWindow().location.search.match(b.REGEX_QUERY);
                    return 0 < (null != a ? a.length : void 0) ? a[1] : null
                };
                b.prototype.hasData = function() {
                    return this._hasDataInHash() || this._hasDataInQuery()
                };
                b.prototype.getData = function() {
                    var b;
                    (b = this._getDataFromHash()) || (b = this._getDataFromQuery() || null);
                    return b
                };
                b.prototype.getType = function() {
                    return b.STORY_TYPE
                };
                return b
            }(x);
            z = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                b.STORY_TYPE =
                "loc";
                b.prototype.getData = function() {
                    return this.getWindow().location.href
                };
                b.prototype.getType = function() {
                    return b.STORY_TYPE
                };
                return b
            }(x);
            K = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                b.STORY_TYPE = "ref";
                b.prototype.hasData = function() {
                    return null !== this.getData()
                };
                b.prototype.getData = function() {
                    return this.getDocument().referrer || null
                };
                b.prototype.getType = function() {
                    return b.STORY_TYPE
                };
                return b
            }(x);
            N = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this,
                    arguments)
                }
                a(b, c);
                b.STORY_TYPE = "utm";
                b.UTM_PREFIX = "utm_";
                b.prototype.getType = function() {
                    return b.STORY_TYPE
                };
                b.prototype.hasData = function() {
                    return RegExp("(\\?|\x26)" + b.UTM_PREFIX + "\\w+").test(this.getWindow().location.href)
                };
                b._unique = function(b) {
                    var a, c, d, e, f;
                    a = {};
                    c = [];
                    e = 0;
                    for (f = b.length; e < f; e++)
                        d = b[e], a[d] || (c.push(d), a[d]=!0);
                    return c
                };
                b.prototype.flattenValue = function(a) {
                    var c;
                    d.isArray(a) && (c = b._unique(a), 1 === c.length && (a = c[0]));
                    return a
                };
                b.prototype.extractData = function(a) {
                    var c, d, e, f, g, k;
                    g = {};
                    d=!1;
                    for (e in a)
                        k = a[e], - 1 !== e.indexOf(b.UTM_PREFIX) && (f = e.substr(b.UTM_PREFIX.length), d|=c=!!f, c && (k = this.flattenValue(k), "string" === typeof k && 0 === k.length && (k = null), g[f] = k));
                    return d ? g : null
                };
                b.prototype.getData = function() {
                    var b, a;
                    b = d.deparam(this.getWindow().location.search.substr(1));
                    a = this.extractData(b);
                    a || (b = this.getWindow().location.hash.indexOf("?"), - 1 !== b && (b = this.getWindow().location.hash.substr(b + 1), a = this.extractData(d.deparam(b))));
                    return a
                };
                return b
            }(x);
            G = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this,
                    arguments)
                }
                a(b, c);
                b.STORY_TYPE = "jd";
                b.WEBSITE_KEY = "website";
                b.PAGE_KEY = "page";
                b.MOBILE_KEY = "isMobile";
                b.SUBGROUP_KEY = "subgroupId";
                b.ENV = "env";
                b.ENV_KEY_LC = "lc";
                b.ENV_KEY_CMS = "cms";
                b.ENV_KEY_HELP = "help";
                b.ENV_KEY_LP = "lp";
                b.prototype.getType = function() {
                    return b.STORY_TYPE
                };
                b.prototype._isCMS = function() {
                    var b;
                    return !(null == (b = this._getJimdoData()) ||!b.isCMS)
                };
                b.prototype._isHelpCenter = function() {
                    var b;
                    return !(null == (b = this._getJimdoData()) ||!b.isJimdoHelpCenter)
                };
                b.prototype._isLC = function() {
                    var b;
                    return !(null ==
                    (b = this._getJimdoData()) ||!b.isLcJimdoCom)
                };
                b.prototype._isLandingpage = function() {
                    var b;
                    return !(null == (b = this._getJimdoData()) ||!b.isLandingpage)
                };
                b.prototype.hasData = function() {
                    return null !== this.getData()
                };
                b.prototype._getJimdoData = function() {
                    var b, a, c, e;
                    return null != (b = null != (a = null != (c = this.getWindow().jimdoData) ? c : d.jimdoData) ? a : null != (e = this.getWindow().jimdoGen002) ? e.jimdoData : void 0) ? b : null
                };
                b._isEmpty = function(b) {
                    for (var a in b)
                        return !1;
                    return !0
                };
                b.prototype.getData = function() {
                    var a, c, d,
                    e, f, g;
                    a = this._getJimdoData();
                    if (!a)
                        return null;
                    g = {};
                    c = this._isCMS();
                    d = this._isHelpCenter();
                    f = this._isLC();
                    (e = c || d || f) && a.websiteId && (g[b.WEBSITE_KEY] = a.websiteId);
                    e && a.pageId && (g[b.PAGE_KEY] = a.pageId);
                    "function" === typeof a.isMobile && (g[b.MOBILE_KEY]=!!a.isMobile());
                    null != a["subgroupId.obfuscated"] && (g[b.SUBGROUP_KEY] = a["subgroupId.obfuscated"]);
                    a = null;
                    f ? a = b.ENV_KEY_LC : d ? a = b.ENV_KEY_HELP : this._isLandingpage() ? a = b.ENV_KEY_LP : c && (a = b.ENV_KEY_CMS);
                    a && (g[b.ENV] = a);
                    return b._isEmpty(g) ? null : g
                };
                return b
            }(x);
            t = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                b.STORY_TYPE = "T";
                b.prototype.getType = function() {
                    return b.STORY_TYPE
                };
                b.pad = function(b, a) {
                    var c, d;
                    null == a && (a = 2);
                    c = Math.abs(b);
                    d = Math.max(0, a - Math.floor(c).toString().length);
                    d = Math.pow(10, d).toString().substr(1);
                    0 > b && (d = "-" + d);
                    return d + c
                };
                b.prototype._getDate = function() {
                    return new Date
                };
                b.prototype.getData = function() {
                    var a, c, d, e, f, g, k, m, n;
                    a = this._getDate();
                    n = a.getFullYear();
                    f = b.pad(1 + a.getMonth());
                    c = b.pad(a.getDate());
                    d = b.pad(a.getHours());
                    e = b.pad(a.getMinutes());
                    k = b.pad(a.getSeconds());
                    g = a.getMilliseconds();
                    m =- 1 * a.getTimezoneOffset();
                    a = b.pad(Math.floor(m / 60));
                    0 <= a && (a = "+" + a);
                    m = b.pad(Math.abs(m%60));
                    return "" + n + "-" + f + "-" + c + "T" + d + ":" + e + ":" + k + "." + g + a + ":" + m
                };
                return b
            }(x);
            R = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                b.STORY_TYPE = "tz";
                b.prototype.getType = function() {
                    return b.STORY_TYPE
                };
                b.prototype.getData = function() {
                    return - 1 * (new Date).getTimezoneOffset()
                };
                return b
            }(x);
            H = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this,
                    arguments)
                }
                a(b, c);
                b.STORY_TYPE = "nav";
                b.prototype.getType = function() {
                    return b.STORY_TYPE
                };
                b.prototype.hasData = function() {
                    var b;
                    return !(null == (b = this.getNavigator()) ||!b.userAgent)
                };
                b.prototype.getData = function() {
                    return this.hasData() ? {
                        userAgent: this.getNavigator().userAgent
                    } : null
                };
                return b
            }(x);
            M = function(c) {
                function b() {
                    return b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                b.STORY_TYPE = "fp";
                b.prototype.getType = function() {
                    return b.STORY_TYPE
                };
                b.prototype.getData = function() {
                    return (new (this.getWindow().Fingerprint)).get()
                };
                return b
            }(x);
            J = function(c) {
                function b() {
                    b.__super__.constructor.call(this, b.PARAM_NAME)
                }
                a(b, c);
                b.STORY_TYPE = "gclid";
                b.PARAM_NAME = "gclid";
                b.prototype.getType = function() {
                    return b.STORY_TYPE
                };
                return b
            }(function(c) {
                function b(a) {
                    this.param = a;
                    b.__super__.constructor.apply(this, arguments)
                }
                a(b, c);
                b.prototype.param = null;
                b.prototype.hasData = function() {
                    var b;
                    b = this.getWindow().location.href;
                    return RegExp("(\\?|\x26)" + this.param + "[\x3d|\\?|\x26]").test(b) || RegExp("(\\?|\x26)" + this.param + "$").test(b)
                };
                b.prototype.getData =
                function() {
                    var b, a;
                    b = d.deparam(this.getWindow().location.search.substr(1));
                    return "undefined" !== typeof b[this.param] ? null != (a = b[this.param]) ? a : "" : null
                };
                return b
            }(x));
            return function(a, b) {
                var d;
                d = {
                    Messenger: A,
                    King: c,
                    historian: {
                        Interface: y,
                        Custom: k,
                        Ajax: m,
                        Pixel: E,
                        Debug: w,
                        Gathering: I,
                        Noop: D
                    },
                    DrillMaster: B,
                    story: {
                        Interface: x,
                        Shadow: Q,
                        Affiliate: g,
                        Referrer: K,
                        Utm: N,
                        Location: z,
                        JimdoData: G,
                        Navigator: H,
                        Fingerprint: M,
                        DateTime: t,
                        TimeZone: R,
                        GClick: J
                    },
                    serializer: {
                        Default: C,
                        Base64: s,
                        QueryString: P
                    }
                };
                a && (d.noConflict =
                function() {
                    if ("undefined" !== typeof b)
                        a.jimdom = b;
                    else 
                        try {
                            delete a.jimdom
                    } catch (c) {
                        a.jimdom = void 0
                    }
                    try {
                        delete d.noConflict
                    } catch (e) {
                        d.noConflict = void 0
                    }
                    return d
                });
                return d
            }
        }(jimdoGen002, document))
    })(window)
}).call(this);
(function(f, a, q, n) {
    var d = "cc-sa-iframe" === a.name, e = "wrapper" === (a._jimdoFrameId || a.name), g = void 0 !== f.jimdoData, m = g && f.jimdoData.isLcJimdoCom, s = g && f.jimdoData.isLandingpage, r = g && f.jimdoData.isJimdoHelpCenter, k = "";
    g && f.jimdoData.isCMS ? (k = "cms", d ? k += ".sac" : e && (k += ".wrapper")) : r ? k = "help" : m ? k = "lc" : s && (k = "lp");
    var t = {
        prod: "d253phb3c6v9f8",
        staging: "dc6l4zqdcuxep"
    }, w = {
        jimdoenv: "prod",
        logoutput: !1
    }, B = function() {
        var a = f("#jimdom");
        return f.extend({}, w, a.data("options"))
    }(), u = {}, M = function() {
        var a;
        if (B.logoutput)
            a =
            new n.historian.Debug("jimdom debug");
        else if ("dev" === B.jimdoenv)
            a = new n.historian.Noop;
        else {
            a = n.historian.Pixel;
            var d = B.jimdoenv, e = t[d];
            if (!e)
                throw Error("[jimddom] No subdomain configured for environment " + d);
            a = new a("//" + e + ".cloudfront.net/t.gif", null, n.serializer.Base64)
        }
        return a
    }, I = function() {
        if (!u.loadEventHistorian) {
            var a = u.loadEventHistorian = M();
            G(a);
            a.addStory(new n.story.Affiliate);
            a.addStory(new n.story.Utm);
            a.addStory(new n.story.GClick)
        }
        return u.loadEventHistorian
    }, J = function() {
        if (!u.multiEventHistorian) {
            var a =
            u.multiEventHistorian = M();
            G(a)
        }
        return u.multiEventHistorian
    }, y = function() {
        u.king || (u.king = new n.King("general"), u.king.tell(J()))
    }, G = function(c) {
        c.addStory(u.shadowStory || (u.shadowStory = new n.story.Shadow({
            domain: "prod" !== B.jimdoenv ? a.location.hostname.substr(a.location.hostname.indexOf(".")): ".jimdo.com"
        })));
        c.addStory(u.referrerStory || (u.referrerStory = new n.story.Referrer));
        c.addStory(u.jimdoDataStory || (u.jimdoDataStory = new n.story.JimdoData));
        c.addStory(u.timezoneStory || (u.timezoneStory = new n.story.TimeZone));
        c.addStory(u.fingerprintStory || (u.fingerprintStory = new n.story.Fingerprint))
    }, e = f.jimdom;
    f.jimdom = {
        event: function(a, d) {
            d = k ? k + (d ? "." + d : "") : d;
            u.generalMessenger || (y(), u.generalMessenger = new n.Messenger("general"));
            u.generalMessenger.send(a, d)
        },
        on: function(a, d) {
            y();
            u.king.tell(new n.historian.Custom(a), d)
        },
        onLoad: function() {
            d || ((new n.King("load")).tell(I()), (new n.Messenger("load")).send("ready"))
        }
    };
    e && q && (q.play(e.onLoad, f.jimdom.onLoad), q.play(e.event, f.jimdom.event))
})(jimdoGen002, window, window.ReplayJS,
window.jimdom.noConflict());
(function(f) {
    (!(window.jimdoData || window.sessionData || {}).isCMS || f.frameManager && "function" === typeof f.frameManager.inWrapper && f.frameManager.inWrapper()) && f(function() {
        f.jimdom.onLoad()
    })
})(jimdoGen002);
