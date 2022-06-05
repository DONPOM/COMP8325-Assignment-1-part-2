var DARLA, $sf, Y;
!function(t) {
    function e(t) {
        return t && typeof t == G ? F : B
    }
    function n(t) {
        return e(t) == B || t instanceof xe == B ? B : F
    }
    function r(t, e) {
        var n = [], r, i;
        if (e = o(e, 0), t) {
            try {
                n = we[q](V, t), i = F
            } catch (r) {
                i = B, n = []
            }
            i ? e > 0 && n[ue] >= e && (n = n.slice(e)) : e > 0 && (n = new we(e))
        }
        return n
    }
    function i(t) {
        var e = typeof t, n;
        if (e == z)
            return t;
        if (e == K&&!t)
            return "0";
        if (e == G && t && t.join)
            return t.join(ae);
        if (!t)
            return ae;
        try {
            t += ae
        } catch (n) {
            t = ae
        }
        return t
    }
    function o(t, e, n, r) {
        var i, o;
        if (typeof t != K)
            try {
                i = ge(i), isNaN(i) && (i = parseFloat(t)), t = i
        } catch (o) {
            t = ge.NaN
        }
        return r == V && (r = Le), n == V && (n = Ae), (isNaN(t) || n > t || t > r) && e != V ? e : t
    }
    function c(t, e, r, i, o, a, s) {
        var f, l, d, h, p, m, y, g;
        if (!n(e))
            return t;
        t || (t = {}), f = s ? t : e;
        for (d in f)
            try {
                if (l = e[d], h = typeof l, m = F, y = d in t, r&&!u(Ve, e, V, d))
                    continue;
                    if (i && h == Y)
                        continue;
                        if (y && (2 !== o || a ? o && (m = B) : m = h == G ? F : B), !m)
                            continue;
                            if (h == G && l) {
                                if (a)
                                    continue;
                                    p = 2 === o && y ? t[d] : {}, l.tagName || l.nodeType ? (l = "#node", DARLA.note && DARLA.note(558)) : l = c(p, l, r, i, o, B, s)
                                }
                                t[d] = l
                        } catch (g) {
            continue
        }
        return t
    }
    function a(t) {
        var e, n, r;
        try {
            t && typeof t == Y && (t instanceof Function ? n = F : (r = t[ie](), r && (r = new t.constructor("return window; ")(), n=!(!r ||!r.top))))
        } catch (e) {
            r = e
        }
        return e = t = r = V, !!n
    }
    function s(t, e) {
        function n(t) {
            var e = t[ue], n = t[0], r = t, i;
            return n && 1 == e && (i = s(n), i[ue] && (r = i)), r
        }
        var i = [], c;
        return t && typeof t == G && (i = t instanceof we ? t : r(t), i = n(i), c = i[ue], e = o(e, 0, 0), e && c && c - 1 >= e && (i = i.slice(e)), i = n(i)), i
    }
    function f(t, e, n) {
        var r = F, i, o, c;
        try {
            if (t && (o = typeof t, o == G || o == Y))
                for (i in t)
                    if (!(e&&!u(Ve, t, V, i) || n && typeof t[PROP] == Y)) {
                        r = B;
                        break
                    }
        } catch (c) {
            r = F
        }
        return r
    }
    function u(t, n, r) {
        var i, o, c, f = arguments, u = f[ue], l = 3, d = [], h = 0, p = 0;
        if (a(t)) {
            u > l && (d = s(f, l)), e(n) || (n = V), h = new be;
            try {
                i = t[q](n, d)
            } catch (c) {
                o = c
            }
            p = new be
        } else 
            o = new Error("bad ref"), o[K] =- 2147418113;
        if (r)
            try {
                r.time = p - h, r.err = o || V
        } catch (c) {}
        return i
    }
    function l(t, e) {
        var n = arguments, i = n[ue], o = i > 2 ? r(arguments, 2): V, c = function() {
            var n = r(arguments);
            return n = o ? n.concat(o) : n, t[q](e || V, n)
        };
        return c
    }
    function d() {
        return (new be).getTime()
    }
    function h() {
        return ve.round(100 * ve.random())
    }
    function p(t) {
        var e = ae;
        return t && (e = i(t)), e && (e = e[fe](/^\s\s*/, ae)[fe](/\s\s*$/, ae)), e
    }
    function m(t) {
        return escape(t)
    }
    function y(t) {
        return unescape(t)
    }
    function g(t, e, n, r, i, o) {
        var c = RegExp, a = t, s = r || ae, f = k(s, "g")>-1, u = "(?:(?!\\1)[^\\\\]|\\\\.)*\\1", l = s[fe](/g/g, ""), d = "g" + l, h = [], p = [], m = B, y, g, v, b, w, x, _, L, A, R, S, E;
        if (n || n === B || (m = F), n === B && (o = F), e)
            if (e && n && e != n) {
                try {
                    S = new c(e + "|" + n, d), E = new c(e, l)
                } catch (_) {
                    S = E = V
                }
                if (S && E)
                    do {
                        for (g = v = 0, A = R = y = V; w = S.exec(a);)
                            if (x = w[0], L = w.index, E.test(x))
                                v++?o || p.push(L) : (A || (A = x), b = S.lastIndex);
                            else if (v&&!--v) {
                                if (y = a.slice(b, L), R = x, A && R && (y = [A, y, R].join(ae), A = R = V), h.push(y), i && i > 0 && h[ue] === i)
                                    return h;
                                    if (!f)
                                        return h
                            }
                            f && p.length && (a = a.slice(p.shift()), g = 1)
                        }
                        while (g || v && (S.lastIndex = b))
                        } else {
                            if (1 === e[ue] && (e = "\\" + e), m)
                                try {
                                    S = new c("([" + e + "])" + u, d)
                                } catch (_) {
                                    S = V
                                } else 
                                    try {
                                        S = new c(e, d)
                                    } catch (_) {
                                        S = V
                                    }
                                    S && (w = a.match(S), w && w[ue] && (h = we.apply(V, w), i && i > 0 && h[ue] > i && (h[ue] = i)))
                                }
                                return h
    }
    function v(t, e, n) {
        var r = [], i, o, c;
        try {
            e = p(e), e = e[Z](), 0 == e.search(/([A-Za-z0-9_]+)/) && (e = e[fe](/([\:\-])/g, "\\$1"), "doctype" == e ? (i = "<(\\!" + e + ")+" + Ye, o = B) : (i = "<(" + e + ")+" + Ye, e in Ge && (c = Ge[e], c.end || (o = B)), o !== B && (o = "<\\/(" + e + ")>")), r = g(t, i, o, "gim", n))
        } catch (a) {
            r = []
        }
        return r
    }
    function b(t, e) {
        var n = new RegExp("(" + e + ")+(>+|\\s+|(\\={1,1}[\\\"']{0,1}([^\\\"']*)[\\\"']{0,1})+)", "i"), r = "", i, o, c;
        try {
            i = t.match(n), i && (c = i[1] || "", r = i[4]||!!c)
        } catch (o) {
            r = ""
        }
        return r
    }
    function w(e, n, r, o) {
        var a = r && typeof r == G ? r: t, s = 0, f = ".", u = V, l, d, h, m, y, g, v, b;
        if (e)
            if (e = i(e), n = n && typeof n == G ? n : V, k(e, f))
                for (l = e.split(f); d = l[s++];)
                    try {
                        d = p(d), y = d in a, h = a[d], m = typeof h, v=!!(m == Y || h && m == G), g=!(!o ||!v), u = s == l[ue] ? v && n ? a[d] = c(h, n, B, B, o) : g ? h : a[d] = h || n || {} : g ? h : a[d] = h || {}, a = u
        } catch (b) {
            a = u = V
        } else 
            h = a[e], m = typeof h, v=!!(m == Y || h && m == G), u = a[e] = v && n ? c(h, n, B, B, o) : h || n || {};
        return u
    }
    function x(t, e, n, r, o) {
        var a, s, f, u, l, d, h = this, p, m, g, v, b, w = B, _, R, S;
        if (!(h instanceof x))
            return new x(t, e, n, r, o);
        if (!arguments[ue])
            return h;
        if (t && typeof t == G)
            return c(new x(ae, e, n, r, o), t);
        if (t = i(t), e = i(e) || I, n = i(n) || j, !t)
            return h;
        for (e != P && n != P && A(t, 0) == P && (t = L(t, 1)), A(t, 0) == e && (t = L(t, 1)), p = t.split(e), b = p[ue], a = 0; b--;)
            if (u = p[a++], v = B, w = B, u) {
                if (m = u.split(n), R = m[ue], R > 2) {
                    if (g = y(m[0]), m.shift(), o)
                        if (l = g + n, s = k(t, l), R = l[ue], d = L(t, s + R), l = e + e, S = l[ue], f = k(d, l), - 1 != f) {
                            d = t.substr(s + R, f + S), _ = new x(d, e, n, r, o), d = ae, R = 0;
                            for (d in _)
                                R++;
                                R > 0 && (a += R - 1), u = _
                        } else 
                            u = y(m.join(n));
                    else 
                        u = y(m.join(n));
                        w = F
                } else 
                    2 == R && (g = y(m[0]), u = y(m[1]), w = F);
                    w && (r ? g in h || (h[g] = u, v = F) : (h[g] = u, v = F), o && v && g && u && typeof u != G && (k(u, e) >= 0 || k(u, n) >= 0) && (h[g] = new x(u, e, n, r, o)))
            }
    }
    function _(t, e, n, r) {
        var o, c = [], a = this, s, f;
        t = t || I, e = e || j;
        for (o in a)
            f = a[o], s = typeof f, f && s == Y || (f && s == G && (f.tagName || f.nodeType ? (DARLA.note && DARLA.note(559), f = "#node") : f = _[q](f, [t, e, n, r])), n && (o = m(o)), r || (f = m(f)), c.push(o, e, f, t));
        return c[ue] && (c[c[ue] - 1] = ""), i(c)
    }
    function k(t, e, n) {
        return n ? t.lastIndexOf(e) : t.indexOf(e)
    }
    function L(t, e, n) {
        return arguments[ue] > 2 ? t.substring(e, n) : t.substring(e)
    }
    function A(t, e) {
        return t.charAt(e)
    }
    function R(t, e, n) {
        var r = t && t.match(e);
        return n == V ? r || V : r && r[n] || V
    }
    function S(t) {
        var e = 0;
        return parseFloat(t[fe](/\./g, function() {
            return 1 == e++?ae: "."
        }))
    }
    function E(t, e) {
        return t.test(e)
    }
    function T() {
        var e, n;
        try {
            e = Ie ? new t[re](Ie) : new XMLHttpRequest
        } catch (n) {
            e = V
        }
        return e || V
    }
    function M() {
        var e = Xe, n = d(), r, i, o;
        if (n - je >= Se || Xe === V) {
            try {
                e=!(t != top ||!Fe[ee + "Enabled"])
            } catch (o) {
                e = V
            }
            if (e === V)
                try {
                    r = "sf_ck_test_" + n + "_" + h(), i = r + "=1", t[ne][ee] = i, e =- 1 != k(t[ne][ee], i), e && (t[ne][ee] = r + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT")
                } catch (o) {
                e = B
            }
            je = n, Xe = e
        }
        return e
    }
    function C() {
        var e = Ue, n = d(), r, o, c, a, s, f, u;
        if (n - $e >= Se || Ue === V) {
            try {
                a = Fe.plugins, f = Q + " " + te, s = a && a[f] || V, s && (e = s.description, e = e[fe](/^.*\s+(\S+\s+\S+$)/, "$1"), r = e[fe](/^(.*)\..*$/, "$1"), o = e[fe](/^.*\.(.*)\s.*$/, "$1"), c =- 1 != k(e, "r") ? e[fe](/^.*r(.*)$/, "$1") : 0, e = r + "." + o + "." + c)
            } catch (u) {
                e = 0
            }
            if (_e&&!e) {
                f = Q + te + "." + Q + te;
                try {
                    s = new t[re](f), s.AllowScriptAccess = "always", s && (e = s.GetVariable("$version"), e ? (e = e.split(" ")[1].split(","), e = e[0] + "." + e[1] + "." + e[2]) : e = 0)
                } catch (u) {
                    e = 0
                }
                if (!e)
                    try {
                        s = new t[re](f + ".6"), e = "6.0.21"
                } catch (u) {
                    e = 0
                }
            }
            $e = n, Ue = e
        }
        return i(e)
    }
    function N(t) {
        var e = {}, n;
        if (!t && We)
            return We;
        e.ie = e.opera = e[ke] = e.webkit = e.safari = e.chrome = e.air = e.ipod = e.ipad = e.iphone = e.android = e.webos = e.silk = e.nodejs = e.phanomjs = 0, e.mobile = e.ios = e.os = V, e.accel = B, e.caja = Fe && Fe.cajaVersion, t = t || Be || ae, t && (E(/windows|win32/i, t) ? e.os = "windows" : E(/macintosh|mac_powerpc/i, t) ? e.os = "macintosh" : E(/android/i, t) ? e.os = "android" : E(/symbos/i, t) ? e.os = "symbos" : E(/linux/i, t) ? e.os = "linux" : E(/rhino/i, t) && (e.os = "rhino"), E(/KHTML/, t) && (e.webkit = 1), E(/IEMobile|XBLWP7/, t) && (e.mobile = "windows"), E(/Fennec/, t) && (e.mobile = ke), n = R(t, /AppleWebKit\/([^\s]*)/, 1), n && (e.webkit = S(n), e.safari = e.webkit, E(/PhantomJS/, t) && (n = R(t, /PhantomJS\/([^\s]*)/, 1), n && (e.phantomjs = S(n))), E(/ Mobile\//, t) || E(/iPad|iPod|iPhone/, t) ? (e.mobile = "Apple", n = R(t, /OS ([^\s]*)/, 1), n = n && S(n[fe]("_", ".")), e.ios = n, e.ipad = e.ipod = e.iphone = 0, n = R(t, /iPad|iPod|iPhone/, 0), n && (e[n[Z]()] = e.ios)) : (n = R(t, /NokiaN[^\/]*|Android \d\.\d|webOS\/\d\.\d/, 0), n && (e.mobile = n), E(/webOS/, t) && (e.mobile = "WebOS", n = R(t, /webOS\/([^\s]*);/, 1), n && (e.webos = S(n))), E(/ Android/, t) && (e.mobile = "Android", n = R(t, /Android ([^\s]*);/, 1), n && (e.android = S(n))), E(/Silk/, t) && (n = R(t, /Silk\/([^\s]*)\)/, 1), n && (e.silk = S(n)), e.android || (e.android = 2.34, e.os = "Android"), E(/Accelerated=true/, t) && (e.accel=!0))), n = R(t, /(Chrome|CrMo)\/([^\s]*)/), n && n[1] && n[2] ? (e.chrome = S(n[2]), e.safari = 0, "CrMo" === n[1] && (e.mobile = "chrome")) : (n = R(t, /AdobeAIR\/([^\s]*)/), n && (e.air = n[0]))), e.webkit || (n = R(t, /Opera[\s\/]([^\s]*)/, 1), n ? (e.opera = S(n), n = R(t, /Opera Mini[^;]*/, 0), n && (e.mobile = n)) : (n = R(t, /MSIE ([^;]*)|Trident.*; rv:([0-9.]+)/), n ? (n = n[1] || n[2], e.ie = S(n)) : (n = R(t, /Gecko\/([^\s]*)/), n && (e[ke] = 1, n = R(t, /rv:([^\s\)]*)/, 1), n && (e[ke] = S(n)))))));
        try {
            typeof process == G && process.versions && process.versions.node && (e.os = process.platform, e.nodejs = S(process.versions.node))
        } catch (r) {
            e.nodejs = 0
        }
        return e
    }
    function H(t, e) {
        var n = 0, r, o, c, a, s, f;
        e || (e = {}, e[pe] = e[le] = e[de] = ae);
        try {
            if (ze) {
                if (r = ze[le], c = ze[de], s = ze[pe], o = ze[ye], a = R(t, Ce), a && (n = a.lastIndex, a = a[0]), E(Oe, a))
                    n = k(s, $, 1), n != se && (a = L(s, 0, n) + a);
                else if (E(He, a))
                    a = a[fe](He, ae), n = k(s, $, 1), n != se && (a = L(s, 0, n + 1) + a);
                else {
                    for (; Ne.exec(a) && (n = k(s, $, 1), n != se);)
                        s = L(s, 0, n), a = a[fe](RegExp.$1, ae);
                    a = i([s, $, a])
                }
                e[le] = r, e[de] = c, e[pe] = a
            }
        } catch (f) {}
        return e
    }
    function O(t, e) {
        var n = ae, r = ae, o = ae, c = ae, a = ae, s, f = ae, u = 0;
        return t && (t.search(Me) || (H(t, e), t = i([e[le], W, e[de], $, e[pe]])), s = R(t, Ee), s && (n = s[1] || ae, r = s[2] || ae, f = s[3] || ae, o = s[5] || ae, c = s[6] || ae, a = s[7] || ae), n && (n = n[fe](De, ae), "file" == n[Z]() && r && o && A(o, 0) == X && (o = r + o, r = ae)), f && (f = f[fe](De, ae)), c && A(c, 0) == P && (c = L(c, 1)), a && A(a, 0) == U && (a = L(a, 1)), o && A(o, 0) == $ && (o = L(o, 1)), a && (u = k(a, P), u != se && (c = L(a, u + 1), a = L(a, 0, u)))), e = e || {}, e[le] = n, e[de] = r, e[ye] = f, e[pe] = o, e[he] = c, e[me] = a, e
    }
    function D(t, e, n, r, o, c) {
        if (!(this instanceof D))
            return new D(t, e, n, r, o, c);
        var a = this, s = arguments[ue], f, u = B, l, d, h;
        if (a[le] = a[de] = a[pe] = a[me] = a[ye] = ae, a[he] = V, !s)
            return a;
        t = i(t), 1 != s && (!t || e || n || r || o || c) ? ( - 1 == k(t, X) && (t += X), f = R(t, Te, 0), f && (l = a[le] = f, a[le] = a[le][fe](/\:/g, ae)), e = i(e), e ? a[de] = e : (d = {}, O(t, d), d[de] && (a[de] = d[de])), c && R(c, /^\d+/) ? a[ye] = c : (d = {}, O(t, d), d[ye] && (a[ye] = d[ye])), n = i(n), n ? (u=!!R(n, Me), u && H(n, a), u && (e != a[de] && (a[de] = e), l != a[le] && (a[le] = l), c != a[ye] && (a[ye] = c)), a[pe] = n) : (d = {}, O(t, d), d[pe] && (a[pe] = d[pe])), o ? a[me] = o : (d = {}, O(t, d), d[me] && (a[me] = d[me])), r ? a[he] = r : (d = {}, O(t, d), d[he] && (a[he] = d[he]))) : O(t, a), k(a[pe], U) || (a[me] = L(a[pe], 1), a[pe] = ae);
        try {
            a[he] && "string" == typeof a[he] && (a[he] = x(a[he], I, j))
        } catch (h) {
            a[he] = V
        }
    }
    var P = "?", I = "&", j = "=", $ = "/", X = ":", U = "#", W = X + $ + $, F=!0, B=!1, V = null, G = "object", Y = "function", z = "string", J = "undefined", K = "number", Z = "toLowerCase", q = "apply", Q = "Shockwave", te = "Flash", ee = "cookie", ne = "document", re = "ActiveXObject", ie = "toString", oe = "valueOf", ce = "prototype", ae = "", se =- 1, fe = "replace", ue = "length", le = "protocol", de = "host", he = "params", pe = "path", me = "hash", ye = "port", ge = t && t.Number, ve = t && t.Math, be = t && t.Date, we = t && t.Array, xe = t && t.Object, _e = B, ke = "gecko", Le = ge && ge.MAX_VALUE || 9007199254740992, Ae = se * Le, Re = 2048, Se = 6e4, Ee = /^(http\:|https\:|file\:|ftp\:)(?:\/)+([-\w\.]+)(\:\d+)?(([^\s\?#]*)(\?\S[^#]*)*(#\S*)*)/i, Te = /http\:|https\:|file\:|ftp:\:/gi, Me = /^(\.\.\/|\.\/|\/)/, Ce = /\S[^\?#]*/, Ne = /(^\.\.\/)/, He = /(^\.\/)/, Oe = /(^\/)/, De = /\:/g, Pe = 0, Ie = ae, je = 0, $e = 0, Xe = V, Ue = V, We = V, Fe = t && t.navigator, Be = Fe && Fe.userAgent || ae, Ve = xe[ce].hasOwnProperty, Ge = {
        img: {
            end: 0,
            type: 0
        },
        script: {
            end: 1,
            type: 1
        },
        style: {
            end: 1,
            type: 2
        },
        iframe: {
            end: 1,
            type: 3
        },
        object: {
            end: 1,
            type: 4
        },
        embed: {
            end: 1,
            type: 5
        },
        param: {
            end: 0,
            type: 6
        },
        video: {
            end: 1,
            type: 7
        },
        audio: {
            end: 1,
            type: 8
        },
        track: {
            end: 0,
            type: 9
        },
        source: {
            end: 0,
            type: 10
        },
        applet: {
            end: 1,
            type: 11
        },
        base: {
            end: 0,
            type: 12
        },
        link: {
            end: 0,
            type: 13
        },
        meta: {
            end: 0,
            type: 14
        },
        title: {
            end: 1,
            type: 15
        },
        html: {
            end: 1,
            type: 16
        },
        head: {
            end: 1,
            type: 17
        },
        body: {
            end: 1,
            type: 18
        },
        frameset: {
            end: 1,
            type: 19
        },
        frame: {
            end: 0,
            type: 20
        },
        doctype: {
            end: 0,
            type: 21
        },
        noscript: {
            end: 1,
            type: 22
        }
    }, Ye = "((?:\\s+[\\:\\-A-Za-z0-9_]+(?:\\s*=\\s*(?:(?:\\\"[^\\\"]*\\\")|(?:'[^']*')|[^>\\s]+))?)*)\\s*(\\/?)>", ze, Je, Ke, Ze, qe;
    qe = x[ce], qe[ie] = qe[oe] = _, function() {
        function e(t) {
            var e = ae, n = t || this, r = n[le], o = n[de], c = n[ye];
            return r ? "file" != r ? o ? c&&!R(c, /^\d+/) ? B : (r += X, r.search(Te) ? B : (e = i([r, W, o, c ? X + c: ae, $, n[pe]]), e[ue] > Re ? B : R(o, /^[-\w\.]+/i)?!!e : B)) : B : !(!r ||!n[pe]) : B
        }
        function n(e) {
            var n, r, i;
            if (e) {
                try {
                    r = t[ne].referrer
                } catch (i) {
                    r = ""
                }
                r != Ze ? (Ze = r, n = Je = new D(r)) : n = Je
            } else {
                try {
                    r = t.location.href
                } catch (i) {
                    r = ae
                }
                if (!r)
                    try {
                        r = t[ne].URL
                } catch (i) {
                    r = ae
                }
                r != Ke ? (Ke = r, n = ze = new D(r)) : n = ze
            }
            return n
        }
        function r() {
            return n()
        }
        function o() {
            return n(1)
        }
        var c = {};
        D.MAX_LENGTH = Re, D.validate = function(t) {
            return e(new D(t))
        }, c.isValid = e, c.isSSL = function(t) {
            var e = t || this, n = B, r;
            if (e && e instanceof D)
                try {
                    n = 0 == e[le].search(/https/i)
            } catch (r) {
                n = B
            }
            return n
        }, c[ie] = c[oe] = function() {
            var t, n = this, r, o, c, a, s, f;
            try {
                if (!e(n))
                    return ae
            } catch (f) {
                return ae
            }
            return o = n[he], r = n[pe], c = n[me], a = n[ye], t = [n[le], W, n[de]], a && t.push(X, a), r && R(r, /\/|\w+/g) && t.push($, r), o && o instanceof x && (o = i(o) || ae, s = o[ue], s && A(o, s - 1) == I && (o = L(o, 0, s - 1)), t.push(P, o)), c && t.push(U, c), i(t)
        }, c.toStripString = function() {
            var t = ae, e = this, n = e[le], r = e[pe], o = e[de];
            return n && r && o && (t = i([n, W, o, $, r])), t
        }, c.toHostString = function(t, n) {
            var r = ae, o = this, c = [], a, s;
            return o instanceof D && e(o) && (a = o[de] || ae, a && c.push(a), t !== B && (s = o[le] || ae, s && c.unshift(s, W)), n && a && (s = o[ye] || ae, s && c.push(X, s)), r = i(c)), r
        }, D[ce] = c, r(), o(), D.loc = r, D.ref = o, D.convertRelative = H
    }(), function() {
        var e = "Msxml2", n = ".XMLHTTP", r = [e + n + ".6.0", e + n + ".3.0", e + n, "Microsoft" + n], i = V, o = 0, c, a, s;
        try {
            c = t[re], _e=!!c
        } catch (s) {
            c = V, _e = B
        }
        if (c)
            for (; a = r[o++];)
                try {
                    i = new c(a), i && (Ie = a)
        } catch (s) {
            Ie = ae
        } finally {
            i = V
        }
        c = i = V, We = N(), We.parse = N, _e = _e ? _e : !!We.ie
    }(), w("DARLA", {
        isIE: _e,
        cookiesOn: M,
        flashVersion: C,
        xhr: T
    }, V, F), w("DARLA.Lang", {
        ParamHash: x,
        URL: D,
        cstr: i,
        cnum: o,
        mix: c,
        time: d,
        rand: h,
        def: w,
        trim: p,
        convertArgs: s,
        canCall: a,
        callSafe: u,
        rbind: l,
        empty: f,
        findTags: v,
        findAttribute: b,
        cbool: function(t) {
            var e = t, n = typeof e;
            return e && n == G && (e = i(e), e = e ? e[Z]() : e, n = z), n == z && "0" === e || "false" === e || "no" === e || e === J || "null" === e ? B : !!e
        },
        guid: function(t) {
            return i([t || ae, "_", Pe++, "_" + d(), "_", h()])
        },
        noop: function() {},
        rtrue: function() {
            return F
        },
        rfalse: function() {
            return B
        },
        rnull: function() {
            return V
        },
        ar: r,
        obj: function() {
            return {}
        },
        ns: function(e, n) {
            var r = /(\[(.{1,})\])|(\.\w+)/gm, o = /\[(('|")?)((\s|.)*?)(('|")?)\]/gm, c = /(\[.*)|(\..*)/g, a = /\./gm, s = 0, f = ae, u = B, l = B, d, h, m, y, g, v, b;
            if (h = n = n || t, e && (e = i(e)))
                if (e = p(e), m = R(e, r)) {
                    for (f = e[fe](c, ae), m.unshift(f); y = m[s++];) {
                        y = y[fe](o, "$3")[fe](a, ae);
                        try {
                            if (g = h[y], v = typeof g, !(g && v == G || v == Y)) {
                                l = F, u = B;
                                break
                            }
                            h = h[y]
                        } catch (b) {
                            l = F, u = B;
                            break
                        }
                    }
                    l || (u = F)
                } else 
                    try {
                        y = e, g = h[y], v = typeof g, (g && v == G || v == Y) && (h = g, u = F)
            } catch (b) {
                u = B
            }
            return d = u ? h : B
        },
        sto: function(t, e) {
            return setTimeout(t, e)
        },
        cto: function(t) {
            return clearTimeout(t)
        },
        es: m,
        ues: y
    }, V, F), w("$sf.lib", {
        cookiesOn: M,
        isIE: _e,
        flashVersion: C,
        lang: DARLA.Lang
    }, V, F), w("DARLA.Dom.UA", We, V, F)
}(window), function(t) {
    function e(t) {
        var e, n;
        try {
            n = xn(t && t.nodeType, - 1)
        } catch (e) {
            n =- 1
        }
        return n
    }
    function n(t) {
        return 1 === e(t)
    }
    function r(e) {
        kn(t[He][Ae], t[He], Te, ge, r, Me), kn(t[He][Ae], t[He], Te, xe, i, Me), $(t, be, i), Vn = Ce
    }
    function i(e) {
        kn(t[He][Ae], t[He], Te, ge, r, Me), kn(t[He][Ae], t[He], Te, xe, i, Me), $(t, be, i), Vn = Ce
    }
    function o() {
        var e, n, r, i;
        if (Bn && (Rn(Bn), Bn = 0), Wn >= Fn && (Un = Te, Vn = Ce), Vn === Te) {
            try {
                e = t[He].body, n = ee("*")[ae], r = e && e.childNodes && e.childNodes[ae] || 0, i = e && e.lastChild
            } catch (c) {
                $n = Xn = 0, Un = Te
            }
            $n && n == $n && Xn && r == Xn && i == Un ? (Un = Te, Vn = Ce) : ($n = n, Xn = r, Un = i, Wn += 1, Bn = An(o, Se))
        } else 
            Un = Te
    }
    function c(e) {
        var n = Me, r = Me, i = be + "ed", c = "complete", a, s;
        if (Bn && (clearTimeout(Bn), Bn = 0), 1 != e && Vn)
            Un = Te, e ? 2 == e && (n = Ce) : n = Ce;
        else {
            try {
                a = t[He].readyState
            } catch (s) {
                a = ""
            }
            a ? (Un = Te, e ? 2 == e ? a == i || a == c ? n = Vn = Ce : (r = Ce, n = Vn = Me) : a != i && a != c ? (n = Ce, r = Ce) : (r = Ce, n = Me) : a == i || a == c ||!Mn || Mn && Nn > 8 && "interactive" == a ? n = Vn = Ce : (r = Ce, n = Vn = Me)) : (r = Ce, 1 == e && (n = Ce))
        }
        return r && ($n = Xn = Wn = 0, Un = Te, o()), n
    }
    function a(t) {
        var e = wn(t && t.id), n;
        n = e && Gn[e], n && ($(t, be, n), Gn[e] = Te, delete Gn[e])
    }
    function s(t, e) {
        var n, r;
        bn.canCall(e) && (n = function(i) {
            var o = i[Pn] || i[In];
            a(o), o && e && kn(e, o, Te, i), o = t = e = n = r = Te
        }, r = t.id, a(t), r && (Gn[r] = n), j(t, be, n)), n = Te
    }
    function f(t, e, n, r) {
        return Tn = Sn.XMsgHost, t && Tn && Tn[t] && Tn[t](e, n, r)
    }
    function u(t) {
        var e = t, n;
        try {
            e = t && "string" == typeof t ? R(t) || t : t
        } catch (n) {
            e = t
        }
        return e
    }
    function l(t, e, n) {
        var r = Me, i = h(t), o;
        if (o = _(i), e = e || "", o && i != top)
            try {
                e || Mn ? (o.open("text/html", le), o.write(e), n || o.close()) : i.location[le](ie), r = Ce
        } catch (c) {
            r = Me
        }
        return t = o = i = Te, r
    }
    function d(e) {
        var n = Te;
        return e = e || t, e && e[Dn] ? n = e : (e = _(e), e && e[Dn] && (n = e)), n
    }
    function h(t) {
        var e, n, r, i, o, c, a = 0, s, f;
        try {
            if (e = t.contentWindow || Te, !e)
                for (r = _(t), n = r && A(r), i = n && n.frames || []; o = i[a++];) {
                    try {
                        c = o.frameElement
                    } catch (f) {
                        c = Te
                    }
                    if (c && c == t) {
                        e = o;
                        break
                    }
                }
        } catch (s) {
            e = Te
        }
        return e
    }
    function p(e, n, r, i, o) {
        var c, s, l, d, h, p;
        e = e || {}, d = e.id, s = d && u(d), h = te(s), s = h ? s : Te, l = h == oe ? s : Te, l ? (f("detach", l), a(l), p = O(l), c = y(l, e, n, r, o), M(c, we, Te), M(c, "onreadystatechange", Te)) : (i && ("string" == typeof i && (i = u(i)), te(i) && (p = i)), !p && s && (p = O(s)), n = wn(n) || C(s) || "", c = w(e, n, r, o));
        try {
            p ? l ? p[de](c, l) : s ? p[de](c, s) : H(p, c) : H(t[He].body, c)
        } catch (m) {}
        return c = s = e = l = p = r = Te, R(d)
    }
    function m(t, e, n) {
        var r = Me;
        return (t = u(t)) ? (e = e || "", r = l(t, e, n), t = Te, r) : r
    }
    function y(t, e, n, r, i) {
        return b(t, e, n, r, i)
    }
    function g(t, e, n, r, i, o) {
        var c = Ln(v, Te, e, n, r, i, o);
        An(c, Mn ? 75 : 1)
    }
    function v(t, e, n, r, i) {
        var o, c, a, u;
        t && D(t) && (Mn ? (a = O(t), c = t.cloneNode(Me), c.src = e, u = x("div"), u.innerHTML = c.outerHTML, c = u.firstChild, s(c, n), r && f(he, c, r, i), a[de](c, t)) : (o = h(t), s(t, n), r && f(he, t, r, i), o.location[le](e)))
    }
    function b(t, e, n, r, i, o) {
        var c = ["<", oe, " "], a = "", l = Me, d, h, p, m, y, v, b, w, _, k, L;
        if (o)
            y = t;
        else {
            if (t = u(t), te(t) != oe)
                return Te;
            y = t.cloneNode(Me)
        }
        e = e || {}, _e in e && M(y, _e, Te), ke in e && M(y, ke, Te), _ = e[_e] = wn(e[_e]) || M(t, _e) || ie, k = e[ke] = wn(e[ke]) || M(t, ke) || "", a = i && f("prep", e), a && (k = wn(a)), o || (M(y, "width", Te), M(y, "height", Te)), n && (m = C(y), m && ";" != m.charAt(m[ae] - 1) && (m += ";"), C(y, [m, wn(n)])), l = _ != ie && bn.cbool(e.async) && Sn.loading(), l && (e[_e] = ie, delete e.async), m = x("div"), H(m, y), b = m.innerHTML, w = b[le](/<iframe(.*?)>(.*?)<\/iframe>/gim, "$1"), k && c.push(ke, '="', k, '" '), w && c.push(w), c.push(" ></", oe, ">"), delete e[ke], m.innerHTML = wn(c), v = m.firstChild;
        for (d in e)
            p = e[d], h = typeof p, ("function" == h || p && h == Ne) && (p = ""), M(v, d, p);
        return v.id || (v.id = "darla_" + oe + "_" + jn, jn++), M(v, "FRAMEBORDER", "no"), M(v, Ge, "no"), M(v, "ALLOWTRANSPARENCY", Ce), M(v, "HIDEFOCUS", Ce), M(v, "TABINDEX", - 1), M(v, "MARGINWIDTH", 0), M(v, "MARGINHEIGHT", 0), l ? (L = Ln(g, Te, v, _, r, a, i), s(v, L)) : (s(v, r), a && f(he, v, a, i)), a = i = y = r = t = m = null, v
    }
    function w(t, e, n, r) {
        return b(x(oe), t, e, n, r, Ce)
    }
    function x(e, n) {
        return (arguments[ae] > 1 && _(n) || t[He]).createElement(e)
    }
    function _(t) {
        var n = Te, r;
        try {
            t && (r = e(t), n = 9 == r ? t : t[He] || t.ownerDocument || Te)
        } catch (i) {
            n = Te
        }
        return n
    }
    function k(e) {
        var n = e && _(e) || t[He], r = n[$e], i = n[Oe];
        return r&&!Cn && "CSS1Compat" != r && (i = n.body), i
    }
    function L(e, n) {
        var r = t[He].domain, i = bn.URL.loc().host, o;
        if (e&&-1 != i.indexOf(e)&&-1 != e.indexOf("."))
            try {
                t[He].domain = e, r = e
        } catch (o) {}
        return r != i || n || (r = ""), r
    }
    function A(t) {
        var e = Te, n;
        try {
            t && (e = t[me] || t[ye] || Te, e || (n = _(t), e = n && (n[me] || n[ye]) || Te))
        } catch (r) {
            e = Te
        }
        return e
    }
    function R(e) {
        var n = arguments, r = n[ae], i, o = Te, c;
        i = r > 1 ? _(n[1]) : t[He];
        try {
            o = e && i.getElementById(e) || Te
        } catch (c) {
            o = Te
        }
        return o
    }
    function S(n, r) {
        var i = Te, o;
        try {
            i = t[He].elementFromPoint(n, r), 1 !== e(i) && (i = Te)
        } catch (o) {
            i = Te
        }
        return i
    }
    function E(e, n) {
        var r, i, o, c, a = "{", s = "}";
        try {
            r = ee("head")[0], - 1 == e.indexOf(a)&&-1 == e.indexOf(s) ? (i = x("link"), i.type = "text/css", i.rel = "stylesheet", i.href = e, n && (i.id = n), H(r, i)) : (i = x("style"), i.type = "text/css", n && (i.id = n), H(r, i), c = i.styleSheet, c && c.addRule ? c[Xe] = e : (o = t[He].createTextNode(e), H(i, o)))
        } catch (f) {}
    }
    function T(t, e, n) {
        try {
            arguments[ae] > 2 ? n === Te ? t[ue](e, 0) : (n = wn(n), "class" == e[Pe]() ? t.className = n : t[fe](e, n, 0)) : n = wn(t[se](e, 0))
        } catch (r) {
            n = ""
        }
        return n
    }
    function M(t, e, n) {
        try {
            arguments[ae] > 2 ? n === Te ? t[ue](e) : (n = wn(n), "class" == e[Pe]() ? t.className = n : t[fe](e, n)) : n = wn(t[se](e))
        } catch (r) {
            n = ""
        }
        return n
    }
    function C(t, e) {
        var n;
        try {
            n = t.style, arguments[ae] > 1 ? n[Xe] = wn(e) : e = n[Xe]
        } catch (r) {
            e = ""
        }
        return e
    }
    function N(t, e, n) {
        var r = 1 == e ? "inherit": 2 == e ? "visible": "hidden", i = t && t.style, o =- 1;
        i && (i.visibility = r, (0 === n || 1 === n || 2 === n || 3 === n) && (1 == e || 2 == e ? 1 == n ? o = "block" : 2 == n ? o = "inline-block" : 3 == n && (o = "inline") : o = "none"), - 1 != o && (i.display = o))
    }
    function H(t, e) {
        return t.appendChild(e)
    }
    function O(t) {
        return t && (t[Ke] || t.parentElement)
    }
    function D(e, n) {
        return n = arguments[ae] > 1 ? _(e) : t[He], n && e && q(n[Oe], e)
    }
    function P(t) {
        return t && (t.text || t.innerHTML || t.innerText) || ""
    }
    function I(t, e, n) {
        var r = new Image;
        return r[we] = r.onerror = function() {
            kn(e, r), kn(n, r), r[we] = r.onerror = Te, e = n = r = Te
        }, r[_e] = t, r
    }
    function j(t, e, n, r) {
        var i = Me, o = {}, c;
        if (r = r || Me, kn(t && t[Le], t, o, e, n, r), o.err || (i = Ce), !i && Mn)
            try {
                t.attachEvent(ve + e, n), i = Ce
        } catch (c) {
            i = Me
        }
        return t = n = Te, i
    }
    function $(t, e, n, r) {
        var i = Me, o = {}, c;
        if (r = r || Me, kn(t && t[Ae], t, o, e, n, r), o.err || (i = Ce), !i && Mn)
            try {
                t.detachEvent(ve + e, n)
        } catch (c) {
            i = Me
        }
        return t = n = Te, i
    }
    function X(t, e) {
        var n, r = "", i;
        try {
            n = t[Ue]
        } catch (i) {
            n = Te
        }
        if (arguments[ae] > 1 && e && n)
            try {
                r = n[e]
        } catch (i) {
            r = ""
        } else 
            r = n;
        return r
    }
    function U(t, e) {
        var n, r = "", i;
        try {
            n = A(t)[We](t, Te)
        } catch (i) {
            n = Te
        }
        if (arguments[ae] > 1 && e && n)
            try {
                r = n[e]
        } catch (i) {
            r = ""
        } else 
            r = n;
        return r
    }
    function W(t) {
        var e = [ - 1, - 1, - 1, - 1], n = [Be + "Top", Be + "Right", Be + "Bottom", Be + "Left"], r = 0, i, o;
        if (!t)
            return e;
        for (; o = n[r];)
            i = t[o], B(i) && (i = xn(i, - 1), i >= 0 && (e[r] = i)), r++;
        return e
    }
    function F(t) {
        var e = [ - 1, - 1, - 1, - 1], n = t && t[Be], r = 0, i, o;
        if (n&&-1 != n[pe](/\d+/g))
            for (n = n[le](/\w+\(([^\)]*?)\)/g, "$1"), e = n.split(" "), e = e[ae] <= 1 ? e.split(",") : e, o = e[ae], r = 0; o--;)
                i = e[r], e[r] = B(i) ? xn(i, - 1) : - 1, r++;
        return e
    }
    function B(t) {
        return t = wn(t), t&&-1 == t[pe](/\D+/g) ? Ce : t&&-1 != t[pe](/px/gi) ? Ce : void 0
    }
    function V(t, e, n) {
        var r = 0, i = 0, o = /^t(?:able|d|h|r|head|foot)$/i;
        return n = n || Yn(t), n && (r = n.borderTopWidth, i = n.borderLeftWidth, r = B(r) ? xn(r, 0) : 0, i = B(i) ? xn(i, 0) : 0, On && o.test(te(t)) && (r = i = 0)), e = e || {
            t: 0,
            l: 0
        }, e.t += r, e.l += i, e
    }
    function G(e) {
        var n = {
            x: 0,
            y: 0,
            w: 0,
            h: 0
        }, r = {
            scrollLeft: 0,
            scrollTop: 0,
            scrollWidth: 0,
            scrollHeight: 0
        }, i, o, c, a, s = 0, f = 0;
        return i = _(e) || t[He], o = i[Oe] || r, a = i.body || r, c = i.defaultView, c && (s = xn(c.pageXOffset, 0), f = xn(c.pageYOffset, 0)), n.x = mn(o[ln], a[ln], s), n.y = mn(o[un], a[un], f), n.w = mn(o[sn], a[sn], 0), n.h = mn(o[fn], a[fn], 0), n
    }
    function Y(e) {
        var r = {
            t: 0,
            l: 0,
            r: 0,
            b: 0,
            w: 0,
            h: 0,
            z: 0
        }, i = "getBoundingClientRect", o = 0, c = 0, a = 0, s = 0, f = Me, u = _(e) || t[He], l = u[$e], d = u.documentMode || 0, h, p, m, y, g, v, b, w, x, L, A;
        if (n(e))
            try {
                if (g = Yn(e), h = k(e), p = G(e), r.l = e[tn] || 0, r.t = e[Qe] || 0, m = e, y = Te, f = On || Hn > 519, A = e === h, !A && i && e[i])
                    Mn && (!d || d > 0 && 8 > d || l === Fe) && (b = h.clientLeft, w = h.clientTop), x = e[i](), r.t = x.top, r.l = x.left, (b || w) && (r.l -= b, r.t -= w), (p.y || p.x) && (!En.ios || En.ios >= 4.2) && (r.l += p.x, r.t += p.y);
                else {
                    for (; (m = m[qe]) && n(m) && y !== m;)
                        b = m[tn], w = m[Qe], r.t += w, r.l += b, f && (r = V(m, r)), y = m;
                        if ("fixed" != g.position) {
                            for (m = e, y = Te; (m = m[Ke]) && n(m) && y !== m && m != h;)
                                o = m[un], c = m[ln], On && (v = Yn(m), "visible" != v[Je] && (r = V(m, r, v))), (o || c) && (r.l -= c, r.t -= o), y = m;
                                r.l += p.x, r.t += p.y
                        } else 
                            r.l += p.x, r.t += p.y
                }
                e == h ? (s = e[on], a = e[rn]) : (s = e[nn], a = e[en]), r.b = r.t + s, r.r = r.l + a, r.w = mn(a, 0), r.h = mn(s, 0), r.z = g.zIndex
        } catch (L) {
            r = {
                t: 0,
                l: 0,
                r: 0,
                b: 0,
                w: 0,
                h: 0,
                z: 0
            }
        }
        return r
    }
    function z(e) {
        var n = e && A(e) || t, r = n[an] || 0, i = n[cn] || 0, o = n.screenY || n.screenTop || 0, c = r + o, a = n.screenX || n.screenLeft || 0, s = i + a, f = k(e);
        return r || i ||!f || (r = f[on] || 0, i = f[rn] || 0, s = a + i, c = o + r), {
            t: o,
            l: a,
            b: c,
            r: s,
            w: i,
            h: r
        }
    }
    function J(t) {
        var e = k(t), n = 0, r = 0;
        return e && (n = e[sn] || 0, r = e[fn] || 0), {
            t: 0,
            l: 0,
            b: r,
            r: n,
            w: n,
            h: r
        }
    }
    function K(e, n, r, i, o) {
        var c = e && O(e), a = _(e), s = k(e), f = Y(e), u = Y(s), l = G(s), d = J(e), h = {
            t: 0,
            l: 0,
            r: 0,
            b: 0,
            w: 0,
            h: 0
        }, p = {
            t: 0,
            l: 0,
            r: 0,
            b: 0,
            xs: 0,
            ys: 0,
            xiv: 0,
            yiv: 0,
            iv: 0,
            w: 0,
            h: 0
        }, m = [], y = Me, g = Me, v = Me, b = {
            left: Te,
            right: Te,
            top: Te,
            bottom: Te
        }, w, x, L, A, R, S, E, T, M, C, N, H, D, P, I, j, $, X, U, W, F, B, V, z, K, q, Q, ee, ne, re, ie, oe, ce, se, fe, ue, le, de, he, pe, me, ye, ge, ve, be;
        if (be = a && a.body || Te, n = n && "object" == typeof n ? n : {}, i = xn(i, 0, 0), o = xn(o, 0, 0), !f.h && o && (f.h = o, f.b = f.t + o), !f.w && i && (f.w = i, f.r = f.l + i), c)
            for (w = u.t, x = u.l, L = u.r, A = u.b; (E = Yn(c)) && (y = c == s, g = c == be, ve = E.position, ye = "fixed" == ve, "block" == E.display || "absolute" == ve || "none" != E["float"] || "none" != E.clear ? (he = Y(c), ye&&!me && (me = he, ge = c), C = he.t, N = he.l, H = he.r, D = he.b, W = E[Je + "X"], F = E[Je + "Y"], B = E[Je], V = y ? [ - 1, - 1, - 1, - 1] : zn(E), pe = Me, y ? (P = l.w, $ = l.h) : (P = c[sn], $ = c[fn]), I = c[en], X = c[nn], j = c[rn], U = c[on], !S && I > j && (S = I - j), !R && X > U && (R = X - U), y ? (P > j && (N = 0, H = (t[cn] || 0 || I) + l.x, N > x && (x = N), L > H && (L = H)), $ > U && (C = 0, D = (t[an] || 0 || X) + l.y, C > w && (w = C), A > D && (A = D))) : g || (S && H - N == I && (H -= S), R && D - C == X && (D -= R), (W == ze || W == Ve || W == Ye || B == ze || B == Ve || B == Ye) && (N > x && (x = N, b.left = c), L > H && (L = H, b.right = c), W == Ve || B == Ve ? (m.push(c), pe = Ce) : (W == Ye || B == Ye) && P > j && (m.push(c), pe = Ce)), V[3] > 0 && (re = N + V[3], re > x && (x = re, b.left = c)), V[1] > 0 && (ie = H + V[1], L > ie && (L = ie, b.right = c)), (F == ze || F == Ve || F == Ye || B == ze || B == Ve || B == Ye) && (C > w && (w = C, b.top = c), A > D && (A = D, b.bottom = c), pe || (F == Ve || B == Ve ? (m.push(c), pe = Ce) : (F == Ye || B == Ye) && $ > U && (m.push(c), pe = Ce))), V[0] > 0 && (Q = C + V[0], Q > w && (w = Q, b.top = c)), V[2] > 0 && (ee = he.t + V[2], A > ee && (A = ee, b.bottom = c)))) : ye&&!me && (me = Y(c), ge = c), !y) && (c = O(c), c && te(c)););
        return h = {
            t: mn(w, 0),
            l: mn(x, 0),
            r: mn(L, 0),
            b: mn(A, 0)
        }, h.w = mn(h.r - h.l, 0), h.h = mn(h.b - h.t, 0), N = f.l, H = f.r, C = f.t, D = f.b, T = H - N, M = D - C, re = h.l, ie = h.r, Q = h.t, ee = h.b, oe = ie - re, ne = ee - Q, K = yn(D, ee) - mn(C, Q), q = l.y, 0 > K && q && Q == q && Q > C && Q > D ? (fe = yn(D + q, ee) - mn(C + q, Q), fe > 0 && ne >= fe ? (K = fe, C = f.t = C + q, D = f.b = D + q) : (K = 0 > K ? 0 : K, K = K > M ? M : K)) : (K = 0 > K ? 0 : K, K = K > M ? M : K), z = yn(H, ie) - mn(N, re), q = l.x, 0 > z && q && re == q && re > N && ee > D ? (fe = yn(H + q, ie) - mn(N + q, re), fe > 0 && oe >= fe ? (fe = z, N = f.l = N + q, H = f.r = H + q) : (z = 0 > z ? 0 : z, z = z > T ? T : z)) : (z = 0 > z ? 0 : z, z = z > T ? T : z), p.t = C > Q ? C >= ee ? 0 : mn(C - Q, 0) : 0, p.b = ee > D ? Q >= D ? 0 : mn(ee - D, 0) : 0, p.l = N > re ? N >= ie ? 0 : C >= ee ? 0 : Q >= D ? 0 : mn(N - re, 0) : 0, p.r = ie > H ? re >= H ? 0 : C >= ee ? 0 : mn(ie - H, 0) : 0, p.w = mn(p.r - p.l, 0), p.h = mn(p.b - p.t, 0), p.xiv = T > 0 ? xn((z / T)[dn](2)) : 0, p.yiv = M > 0 ? xn((K / M)[dn](2)) : 0, p.iv = T > 0 || M > 0 ? xn((z * K / (T * M))[dn](2)) : 0, p.civ = 0, r && (le = p.iv, le > .3 && (de = Z(e, i, o), ce = de[ae], se = xn(de.on, 0), se && (ue = p.civ = 1 - xn((se / ce)[dn](2), 0), le > ue && (p.iv = ue)))), n.rect = f, n.clipRect = h, n.docRect = d, m[ae] ? (n.isRoot = Me, n.canScroll = Ce, p.xs=!!R, p.ys=!!S) : u.b >= h.b || u.r >= h.r ? (n.isRoot = Ce, p.xs=!!(d.w > u.w && R), p.ys=!!(d.h > u.h && S), n.canScroll = l.w > u.w || l.h > u.h) : p.ys = p.xs = n.isRoot = n.canScroll = Me, m[ae] ? (n.isRoot = Me, n.canScroll = Ce, p.xs=!!R, p.ys=!!S) : u.b >= h.b || u.r >= h.r ? (n.isRoot = Ce, p.xs=!!(d.w > u.w && R), p.ys=!!(d.h > u.h && S), n.canScroll = l.w > u.w || l.h > u.h) : p.ys = p.xs = n.isRoot = n.canScroll = Me, n.canScroll && n.isRoot && (0 == p.t && (p.t = yn(f.t - l.y, f.t), p.t = mn(p.t, 0), p.t > 0 && (v = Ce)), v = Me, 0 == p.l && (p.l = yn(f.l - l.x, f.l), p.l = mn(p.l, 0), p.l > 0 && (v = Ce))), n.scrollNodes = m, n.clipNodes = b, n.expRect = p, n.fixedRect = me || Te, n.fixedNode = ge || Te, p
    }
    function Z(t, e, n) {
        var r = Y(t), i = "iframe" == te(t), o = _(t), c = k(o), a = r.t, s = r.l, f = hn, u = [], l = 0, d, h, p, m, y, g, v, b, w, x, L, A, R, E, T, M;
        if (u.on = 0, e = xn(e, 0, 0), n = xn(n, 0, 0), a&&!r.h && n && (r.h = n, r.b = a + n), s&&!r.w && e && (r.w = e, elrect.r = s + e), d = r.w, h = r.h, p = gn(d / f), m = gn(h / f), y = p, g = m, 1 >= d || 1 >= h || 1 > p || 1 > m)
            return u;
        if (M = G(), T = M.y, E = M.x, A = s + d, R = a + h, o && c) {
            for (; d > y;) {
                for (g = m; h > g;)
                    v = s + y, b = a + g, A >= v && R >= b && u.push({
                        x: v,
                        y: b,
                        on: 0
                    }), g += m;
                y += p
            }
            for (; w = u[l++];)
                if (v = mn(w.x, 0), b = mn(w.y, 0), v = mn(w.x - E, 0), v = yn(v, w.x), b = mn(w.y - T, 0), b = yn(b, w.y), L = Te, 0 != v)
                    if (0 != b) {
                        if (L = S(v, b), L && L !== c && L !== t&&!q(L, t)) {
                            if (!i && q(t, L))
                                continue;
                                x = L.id, x || (x = bn.guid("geom_inter"), L.id = x), w.on = x, u.on++
                        }
                    } else 
                        T > 0 ? (L = L || S(w.x, w.y), L && L !== c && (L === t || i && q(t, L)) ? (x = L.id, x || (x = bn.guid("geom_inter"), L.id = x)) : (w.on = "!y-offscreen", u.on++)) : (w.on = "!y-offscreen", u.on++);
                    else 
                        E > 0 ? (L = S(w.x, w.y), L && L !== c && (L === t || i && q(t, L)) ? (x = L.id, x || (x = bn.guid("geom_inter"), L.id = x)) : (w.on = "!x-offscreen", u.on++)) : (w.on = "!x-offscreen", u.on++)
        }
        return u
    }
    function q(n, r) {
        var i = Me, o = e(n), c = e(r);
        if (1 == o&&-1 != c)
            if (n[Ie])
                if (Cn || 1 == c)
                    i = n[Ie](r);
                else 
                    for (; r;) {
                        if (n === r) {
                            i = Ce;
                            break
                        }
                        if (r = r[Ke], r == t[He][Oe])
                            break
                    } else 
                        n[je] && (i = n === r||!!(16 & n[je](r)));
        return i
    }
    function Q(t) {
        var e = Me, n, r = te(t) == oe;
        r && (f("detach", t), a(t), l(t) || M(t, _e, ie));
        try {
            n = O(t), n && (n.removeChild(t), e = Ce, Mn && r && ne())
        } catch (i) {}
        return t = n = Te, e
    }
    function te(t) {
        return 1 === e(t) && t.tagName[Pe]() || ""
    }
    function ee(e, n, r) {
        var i = [], o = "getElementsByTagName", c = "getElementsByClassName", a, s, f, u, l = Te;
        try {
            if (n || (n = t[He]), n)
                if (r) {
                    if ("string" == typeof r) {
                        if (n[c]) {
                            if (a = n[c](r), f = a && a[ae], u = 0, f)
                                for (; s = a[u++];)
                                    te(s) == e && i.push(s)
                                } else if (n[o] && (a = n[o](e), f = a && a[ae], u = 0, l = e ? new RegExp("(?:^|\\s+)" + r + "(?:\\s+|$)") : Te, f && l))
                                    for (; s = a[u++];) 
                                        - 1 != s.className.search(l) && i.push(s)
                                    } else if (n[c])
                                        i = n[c](e);
                                    else if (n[o] && (a = n[o]("*"), f = a && a[ae], u = 0, l = e ? new RegExp("(?:^|\\s+)" + e + "(?:\\s+|$)") : Te, f && l))
                                        for (; s = a[u++];) 
                                            - 1 != s.className.search(l) && i.push(s)
                } else 
                    n[o] && (i = n[o](e))
            } catch (d) {
            i = []
        }
        return i
    }
    function ne() {
        Mn && De in t && (Ee && Rn(Ee), Ee = An(function() {
            try {
                t[De]()
            } catch (e) {}
        }, ce))
    }
    function re(t) {
        var e = Me;
        return (e = kn(Sn.ready)) ? void kn(t) : void An(function() {
            re(t), t = Te
        }, 50)
    }
    var ie = "about:blank", oe = "iframe", ce = 3e3, ae = "length", se = "getAttribute", fe = "setAttribute", ue = "removeAttribute", le = "replace", de = le + "Child", he = "attach", pe = "search", me = "parentWindow", ye = "defaultView", ge = "DOMContentLoaded", ve = "on", be = "load", we = ve + be, xe = "pageshow", _e = "src", ke = "name", Le = "addEventListener", Ae = "removeEventListener", Re = {
        preventDefault: 0,
        stopImmediatePropagation: 0,
        stopPropagation: 0,
        preventBubble: 0
    }, Se = 10, Ee = 0, Te = null, Me=!1, Ce=!0, Ne = "object", He = "document", Oe = "documentElement", De = "CollectGarbage", Pe = "toLowerCase", Ie = "contains", je = "compareDocumentPosition", $e = "compatMode", Xe = "cssText", Ue = "currentStyle", We = "getComputedStyle", Fe = "BackCompat", Be = "clip", Ve = "scroll", Ge = "SCROLLING", Ye = "auto", ze = "hidden", Je = "overflow", Ke = "parentNode", Ze = "offset", qe = Ze + "Parent", Qe = Ze + "Top", tn = Ze + "Left", en = Ze + "Width", nn = Ze + "Height", rn = "clientWidth", on = "clientHeight", cn = "innerWidth", an = "innerHeight", sn = Ve + "Width", fn = Ve + "Height", un = Ve + "Top", ln = Ve + "Left", dn = "toFixed", hn = 5, pn = t.Math, mn = pn.max, yn = pn.min, gn = pn.round, vn = t.DARLA, bn = vn && vn.Lang, wn = bn && bn.cstr, xn = bn && bn.cnum, _n = bn && bn.def, kn = bn && bn.callSafe, Ln = bn && bn.rbind, An = bn && bn.sto, Rn = bn && bn.cto, Sn = vn && vn.Dom, En = Sn && Sn.UA, Tn = Te, Mn = vn && vn.isIE, Cn = En && En.opera, Nn = En && En.ie, Hn = En && En.webkit, On = En && En.gecko, Dn = "postMessage", Pn = Mn ? "srcElement" : "target", In = Mn ? "target" : "currentTarget", jn = 0, $n = 0, Xn = 0, Un = Te, Wn = 0, Fn = 300, Bn = 0, Vn = Te, Gn = {}, Yn, zn, Jn;
    bn && (Mn || Nn ? (Jn = x(oe), M(Jn, Ge, "no"), "no" != M(Jn, Ge) && (M = T), zn = W, Yn = X) : (zn = F, Yn = U), Sn = _n("Dom", {
        elt: R,
        doc: _,
        docNode: k,
        inDoc: D,
        dm: L,
        img: I,
        txt: P,
        make: x,
        view: A,
        css: C,
        attr: M,
        vis: N,
        append: H,
        purge: Q,
        par: O,
        tags: ee,
        tagName: te,
        gc: ne,
        attach: j,
        detach: $,
        wait: re,
        makeCss: E,
        contains: q,
        currentStyle: Yn,
        ready: function() {
            return c(0)
        },
        loading: function() {
            return c(1)
        },
        complete: function() {
            return c(2)
        },
        evtCancel: function(t) {
            var e = "", n;
            if (t) {
                try {
                    t.returnValue = Me
                } catch (n) {}
                try {
                    t.cancelBubble = Ce
                } catch (n) {}
                try {
                    t.stopped = Ce
                } catch (n) {}
                for (e in Re)
                    if (Re[e])
                        try {
                            t[e]()
                } catch (n) {}
            }
            return Me
        },
        evtTgt: function(t) {
            var e = Te;
            try {
                e = t ? t[Pn] || t[In] : Te
            } catch (n) {
                e = Te
            }
            return e
        }
    }, vn, 1), _n("HTML5", {
        listen: function(t, e, n) {
            var r = d(e);
            r && (n ? $(r, "message", t) : j(r, "message", t)), r = e = t = Te
        },
        post: function(t, e, n) {
            var r = d(e), i = Me;
            try {
                n = n || "*", r && t && (r[Dn](wn(t), n), i = Ce)
            } catch (o) {
                i = Me
            }
            return e = r = Te, i
        },
        canPostMsg: function() {
            var e = Me;
            try {
                e=!!t[Dn]
            } catch (n) {
                e = Me
            }
            return e
        }
    }, Sn, 1), _n("IFrames", {
        replace: p,
        write: m,
        make: w,
        clone: y,
        view: h
    }, Sn, 1), _n("Geom", {
        winSize: z,
        rect: Y,
        docRect: J,
        docScroll: G,
        bounds: K,
        overlaps: Z,
        atPt: S
    }, Sn, 1), _n("$sf.lib.dom", Sn, Te, Ce), function() {
        var e = "createEvent", n = "UIEvent", o = "", c;
        if (c = kn(t[He][e], t[He], Te, n), c = c ? c : kn(t[He][e], t[He], Te, n + "s"))
            for (o in Re)
                c[o] && (Re[o] = 1);
        c = Te, kn(t[He][Le], t[He], Te, ge, r, Me), kn(t[He][Le], t[He], Te, xe, i, Me), j(t, be, i)
    }())
}(window), function() {
    function t(t) {
        I && w.cto(I), A.length = 0, S = "", y = m = g = O = T = E = A = l
    }
    function e(t) {
        "iframe" == _.tagName(this) && _.attr(this, "name", l), S && (S = ""), n()
    }
    function n() {
        var t, n;
        S || (t = A.length ? A.shift() : l, t && (n = function() {
            var r;
            r = _.elt(h), r || (r = _.make("div"), r.id = h, r.className = "darla", _.append(g.body, r)), S = "darla_host_xmsg_proxy_frame", t.proxyID = S, t.dm = H, k.replace({
                id: S,
                src: N,
                name: t,
                "class": "darla"
            }, "display:none", e, h), n = t = l
        }, _.wait(n)))
    }
    function r() {
        var t, e;
        if (!T) {
            try {
                b ? t = m.opener : (e = m.frameElement, t = e && e.backChannel)
            } catch (n) {
                t = l
            }
            if (t && t.sendToHost)
                if (T = t, b)
                    m.opener = l;
                else 
                    try {
                        t = x({
                            msg: "darla:detach-ff",
                            id: M,
                            guid: j
                        }), T.sendToHost(t)
                    } catch (n) {}
        }
        T&&!T.sendToClient && E && (T.sendToClient = E)
    }
    function i() {
        return I && w.cto(I), S ? (o(), O ? (S = "", void(A.length && a(A.shift()))) : void(I = w.sto(i, p))) : void 0
    }
    function o(t) {
        var e = "DARLA.Dom.XMsgProxyFB", n = w.ns, r = 0, i = 0, o = d, c, a, s, f, u;
        try {
            o = O && w.canCall(O._send_to_host_via_rev_proxy)
        } catch (f) {
            o = d
        }
        if (!o && y) {
            try {
                s = t && t.name == D && n(e, t) || l
            } catch (f) {
                s = l
            }
            if (!s)
                try {
                    s = n(e, y[D]) || l
            } catch (f) {
                s = l
            }
            if (!s)
                try {
                    for (c = y.frames, i = c.length, r; i > r; r++)
                        try {
                            if (a = c[r], s = a.name == D && n(e, a) || l)
                                break
                            } catch (u) {
                                s = l
                            }
            } catch (f) {
                s = l
            }
        }
        return s && (O = s), O
    }
    function c(t, e) {
        var n = o(t);
        n && (S && (S = ""), I && (w.cto(I), I = 0), s(e, w.noop), S || I ||!A.length || a(A.shift()))
    }
    function a(t) {
        var e = t && t.guid;
        if (e && j && e == j)
            if (r(), T)
                T.sendToHost(t);
            else if (D) {
                if (S)
                    return void A.push(t);
                    if (o(), O) {
                        I && (w.cto(I), I = 0), S = D;
                        try {
                            O._send_to_host_via_rev_proxy(m, j, t), I = w.sto(i, p)
                        } catch (c) {
                            S = ""
                        }
                    }
                    O = t = l
                } else 
                    L._receive = s, A.push(t), n()
        }
    function s(t, e) {
        var r = t && t.guid, i = d, o = d, c, a, s, f;
        if (r && j && r == j && E)
            if (O)
                try {
                    E(t)
        } catch (u) {} else {
            a = t.proxyID, c = a && _.elt(a), s = k.view(c);
            try {
                w.canCall(e) && (f = new e.constructor("return parent")(), o = f == s)
            } catch (u) {
                o = d
            }
            if (i = c && a && S && S == a, o)
                try {
                    E(t)
                } catch (u) {}
            i && (S = "", n())
        }
    }
    function f(t, e) {
        var n = _ && _.XMsgClient;
        n && C && t && e && t == j && (E = e, r(), T || w.sto(function() {
            var t = {
                cmd: "fbinit",
                a: "msg",
                pos: C,
                guid: j,
                dm: H,
                id: M
            };
            a(x(t)), t = l
        }, 1))
    }
    function u(t) {
        var e = w.URL.loc(), n = e.toHostString(), r, i;
        !R && t && t instanceof x && (j = t.guid || "", M = t.id || "", C = t.pos || "", N = t.proxyPath, H = t.dm || "", R = j && M ? t : l, 0 == n.indexOf("file") && (n = ""), r = N && N.substring(0, N.indexOf("/", 9)), i = t.revProxy, r == n && i && ("string" == typeof i ? (P = i, i = x(i)) : P = i.toString(), D = i.proxyID), i && D && (L._rev_proxy_receive = c), delete t.revProxy)
    }
    var l = null, d=!1, h = "darla_xmsgfb_proxy_box", p = 175, m = window, y = top, g = document, v = m.DARLA, b = v && v.isIE, w = v && v.Lang, x = w && w.ParamHash, _ = v && v.Dom, k = _ && _.IFrames, L = l, A = [], R = l, S = "", E = l, T = l, M = "", C = "", N = "", H = "", O = l, D = "", P = "", I = 0, j = "";
    w && _ && (L = w.def("XMsgClientFB", {
        init: u,
        attach: f,
        send: a
    }, _, 1), _.attach(m, "unload", t))
}(), function() {
    function t() {
        x && g.listen(e, l, f), k = d = s, y.detach(l, "unload", t), l = t = s
    }
    function e(t) {
        var e = t && t.data || "";
        y.evtCancel(t), i(e&&-1 != e.indexOf("guid=") && m(e), t), t = s
    }
    function n(t, e, n) {
        return v = y.XMsgClientFB, t && v && v[t] && v[t](e, n)
    }
    function r(t) {
        return g.post(t, d, "file" == L ? "*" : L)
    }
    function i(t, e) {
        var n = t && t.guid, r = t && t.msg, i;
        if (n && b && b == n && k && r)
            try {
                p.sto(function() {
                    try {
                        k(r)
                    } catch (i) {
                        i = s
                    }
                    t = e = r = n = s
                }, 1)
        } catch (i) {}
    }
    function o(t, e) {
        b && b == t && p.canCall(e) && (x || (n("init", _), n("attach", b, i)), k = e)
    }
    function c(t, e) {
        var i, o = u;
        if (e) {
            if (t && b && b == t) {
                if (i = m({
                    msg: e,
                    id: w,
                    guid: t
                }), x && r(i.toString()))
                    return;
                o || n("send", i)
            }
            i = s
        }
    }
    function a(n) {
        !_ && n && n instanceof m && (w = n.id || "", L = n.host || "", x = p.cbool(n.html5), b = n.guid || "", b && w && L && (g.listen(e, l), y.attach(l, "unload", t), _ = n))
    }
    var s = null, f=!0, u=!1, l = window, d = l.parent, h = l.DARLA, p = h && h.Lang, m = p && p.ParamHash, y = h && h.Dom, g = y && y.HTML5, v = s, b = "", w = "", x = u, _ = s, k = s, L = "";
    p && g && y && p.def("XMsgClient", {
        init: a,
        attach: o,
        send: c
    }, y)
}(), function(t) {
    function e(t, e) {
        var n, r, i = "1", o = "0";
        return e && ee.test(e) && (e = e[q](te.sf_ver, "1.0"), e = e[q](te.sf_ck_on, t && t.ckOn || o), n = te.sf_flash_ver, n && n.test(e) && (e = e[q](n, j.flashVersion())), n = te.sf_tp_ck_on, n && n.test(e) && (e = e[q](n, j.cookiesOn())), n = te.sf_cur_iv, n && n.test(e) && (e = e[q](n, R() || o)), n = te.sf_exp_ovr, n && n.test(e) && (e = e[q](n, p(Re) ? i : o)), n = te.sf_exp_push, n && n.test(e) && (e = e[q](n, p(ne) ? i : o)), n = te.sf_rd_ck, n && n.test(e) && (e = e[q](n, p("read-cookie") ? i : o)), n = te.sf_wrt_ck, n && n.test(e) && (e = e[q](n, p("write-cookie") ? i : o)), n = te.sf_bg, n && n.test(e) && (e = e[q](n, p("bg") ? i : o)), n = te.sf_lyr, n && n.test(e) && (e = e[q](n, p("lyr") ? i : o)), r = be && be.self || {}, e = e[q](te.sf_x, r.l || o), e = e[q](te.sf_y, r.t || o), e = e[q](te.sf_w, r.w || o), e = e[q](te.sf_h, r.h || o), r = be && be.par || {}, e = e[q](te.sf_par_x, r.l || o), e = e[q](te.sf_par_y, r.t || o), e = e[q](te.sf_par_w, r.w || o), e = e[q](te.sf_par_h, r.h || o), e = e[q](te.sf_par_root, r && be.win && be.win.w === r.w ? i : o), e = e[q](te.sf_host, Y(fe))), e
    }
    function n(t, e) {
        Ke && K(Ke, V, H, t, e)
    }
    function r(t, e) {
        X.sto(function() {
            Ke && K(Ke, V, H, t, e)
        }, 1)
    }
    function i(t) {
        var e = O, n, r, i, o, c;
        for (n in t)
            if (r = t[n], o = K($, H, H, r)) {
                try {
                    if (i = typeof r, i == P || "function" == i)
                        continue
                    } catch (c) {
                        e = D;
                        break
                    }
                    e = D;
                    break
            }
        return e
    }
    function o(t) {
        var e = O, n, r, o, c, a, s, f;
        return 1 == arguments.length && t && (n = t.self || t, o = t.ref || t.par || o, c = t.exp || c, a = t.win || a, t = n), t && (n = {
            t: z(t.t),
            l: z(t.l),
            r: z(t.r),
            b: z(t.b),
            iv: z(t.iv),
            xiv: z(t.xiv),
            yiv: z(t.yiv)
        }), o && (r = {
            t: z(o.t),
            l: z(o.l),
            r: z(o.r),
            b: z(o.b)
        }), c && (s = {
            t: z(c.t),
            l: z(c.l),
            r: z(c.r),
            b: z(c.b)
        }, c.iv && (n.iv = z(c.iv)), c.xiv && (n.xiv = z(c.xiv)), c.yiv && (n.yiv = z(c.yiv))), a && (f = {
            t: z(a.t),
            l: z(a.l),
            r: z(a.r),
            b: z(a.b)
        }), e = n && r && s && f && i(n) && i(r) && i(s) && i(f)
    }
    function c() {
        var t = I && I.body, e = I && I.documentElement, n = t && t.offsetWidth || t && t.clientWidth || e && e.offsetWidth || e && e.clientWidth || le;
        return n
    }
    function a() {
        var t = I && I.body, e = I && I.documentElement, n = t && t.offsetHeight || t && t.clientHeight || e && e.offsetHeight || e && e.clientHeight || de;
        return n
    }
    function s(t) {
        try {} catch (e) {}
    }
    function f(t) {
        var e = D, r, i, c, a, s;
        return t = t && U(t) || H, i = t && t.geom, r = t && t.cmd, c = t && t.info, s = t && "hf"in t ? t.hf : H, s != H && (we = J(s)), i && (i = U(W(i), H, H, D, O), o(i) ? be = i : i = H), c && (c = U(W(c), H, H, D, O)), r && (a = {
            cmd: r,
            info: c || {},
            value: t.value || c && c.value || "",
            reason: t.reason || ""
        }, r == Ie ? (e = O, Ge && (ze = H, Ge = D, Je = O, x(), Je = D, n(Ie, a))) : r == Pe ? (e = O, Ge && (ze = H, Ge = D, n(Ie, a))) : r == Re || r == ne ? (e = O, ze && (ze = H, Ge = O, n(De + "ed", a))) : r == je ? (e = O, ze && (Ye = a, ze = H, n(je, a))) : r == $e ? (e = O, ze && (Ye = ze = H, n("clear-bg", a))) : r == Xe ? (e = O, ze && (ze = H, n(Xe, a))) : r == Ue ? (e = O, ze && ze.cmd === Ce && (ze = H), n(Ue, a)) : r == Ne ? (e = O, n(Ne, a)) : r == Ee ? (e = O, n(Ee, a)) : r == Oe || "focus-update" == r ? n(Oe, a) : "write-cookie" == r ? (ze && (ze = H), n("write-cookie", a)) : "read-cookie" == r ? (ze && (ze = H), n("read-cookie", a)) : "meta-update" == r ? t.meta && (Le = U(W(t.meta), H, H, D, O), n("meta-update", a)) : r == We && (e = O, ze && (ze = H, a.cmd = t.cmd_failed || a.cmd || "", n(We, a)))), t = H, e
    }
    function u(t, e) {
        var n = G("fc_align"), r = n.style, i, o, c = "position:absolute;";
        i = t ? "right:0px;" : "left:0px;", o = e ? "bottom:0px;" : "top:0px;", r.cssText = c + i + o, n = r = H
    }
    function l(t, e, i) {
        var o = X.guid("pnd_cmd"), c = X.time();
        return ze ? void r(We, "awaiting completion of previous action") : (i !== O && (ze = {
            id: o,
            sent: c,
            cmd: e
        }, X.sto(function() {
            ze && ze.id == o && (ze.cmd == Re && (Je = O, x(), Je = D), ze = H, n(We, {
                cmd: e,
                reason: "timeout"
            })), o = c = e = t = H
        }, Fe)), B && B.send(he, t))
    }
    function d(t, e, n) {
        return !ge && he ? (t = z(t, 0, 0), e = z(e, 0, 0), le = t, de = e, ge = O, ve = X.guid("sf_ext_reg"), Ke = X.canCall(n) ? n : H, ve) : void 0
    }
    function h(t, e) {
        var n = t && typeof t || "", r = D, i = X.time(), o = Te, c = D, a, s, f, u;
        if ("string" == n)
            Ze || 0 != t.indexOf("fdb") ? qe || ge || 0 != t.indexOf("noad") || (qe = O, r = O) : (Ze = O, r = O);
        else if ("object" == n) {
            if (Ze || "fdb" != t.cmd ? qe && ge && "noad" == t.cmd && (qe = O, r = O) : (Ze = O, r = O), t.data && 1 == t.data.nodeType) {
                try {
                    a = t.data, "script" == F.tagName(a) && (s = F.txt(a), s = X.trim(s), f = new Function(" return " + s), u = f())
                } catch (d) {
                    u = H
                }
                f = H, u ? t.data = u : delete t.data
            }
            t = Y(U(t))
        } else 
            t = Y(t);
        return t && (!r && tn > Qe && (o = i - tn), r || (tn = i), (ge && p(Ee) && o >= Te || r) && (l(Y(["cmd=", Ee, "&guid=", he, "&a=msg&pos=", me, "&msg=", escape(t)]), Ee, !0), c = O)), c
    }
    function p(t) {
        var e = Y(t), n = ue || re, r = H;
        return e ? (t in n ? r = n[t] : "*"in n && (r = n["*"]), J(r)) : X.mix({}, n, O, O, O)
    }
    function m(t, e, n) {
        var r = D, i = D, o = D, s = n ? ne: Re, f = ["cmd=", s, "&pos=", me, "&guid=", he], d = 0, h = 0, m, y, g, v, b, w, x;
        if (ge&&!ze&&!Ge&&!(_e || ke || n&&!p(ne))) {
            if (t && typeof t == P) {
                if (m = z(t.r, 0, 0), y = z(t.b, 0, 0), g = z(t.t, 0, 0), v = z(t.l, 0, 0), t.push) {
                    if (!p(ne))
                        return;
                    s = ne, f[1] = s
                }
                !m && v && (r = O, d =- 1 * v), m&&!v && (d = m), !y && g && (i = O, h =- 1 * g), y&&!g && (h = y), o = g && y || v && m ? D : O, c(), a(), o ? (u(r, i), f.push("&dx=", d, "&dy=", h), l(Y(f), s)) : (b = G("fc_align"), w = b && b.style, x = ["position:absolute;"], g && y ? x.push("top:0px;") : g ? x.push("bottom:0px;") : y && x.push("top:0px;"), v && m ? x.push("left:0px;") : v ? x.push("right:0px;") : y && x.push("left:0px;"), w && (w.cssText = Y(x)), f.push("&exp_obj=", escape(U(t))), l(Y(f), s))
            } else {
                if (t = z(t, 0), e = z(e, 0), 0 == t && 0 == e)
                    return;
                r = 0 > t, i = 0 > e, u(r, i), f.push("&dx=", t, "&dy=", e), l(Y(f), s)
            }
            return O
        }
    }
    function y(t) {
        var e = ["w", "h", "html"], n = ["r", "l", "t", "b", "center", "fixed"], r = [], i, o, c;
        if (p("lyr")) {
            for (o in e) {
                if (i = e[o], !(i in t))
                    return !1;
                c = t[i], c = "html" == i ? escape(c) : c, r.push("&", i, "=", Y(c))
            }
            o = "";
            for (o in n)
                i = n[o], i in t && r.push("&", i, "=", Y(t[i]));
            r = Y(r), l(Y(["cmd=", Me, "&guid=", he, "&a=msg&pos=", me, r]), Me)
        }
    }
    function g() {
        l(Y(["cmd=", Ce, "&guid=", he, "&a=msg&pos=", me, "&options=none"]), Ce)
    }
    function v(t) {
        p("lyr") && l(Y(["cmd=", Ne, "&guid=", he, "&a=msg&pos=", me, "&msg=", Y(t)]), Ne, !0)
    }
    function b(t) {
        var e;
        return p(Se) ? (e = w(t, ""), Ye || e && "1" != e.clear ? (e.cmd = Se, e.guid = he, e.pos = me, l(Y(e), Se), O) : (r(We, Se), D)) : (r(We, Se), D)
    }
    function w(t, e) {
        var n = "&clear=1", r = 0, i;
        if (t && typeof t == P) {
            n = [];
            for (i in t) {
                if (name = i, "color" == name && (n[r] = "&" + e + "color=" + t[i]), "imgsrc" == name && (n[r] = "&" + e + "imgsrc=" + t[i]), "posX" == name) {
                    if ($(t[i]))
                        return;
                    n[r] = "&" + e + "posX=" + t[i]
                }
                if ("posY" == name) {
                    if ($(t[i]))
                        return;
                    n[r] = "&" + e + "posY=" + t[i]
                }
                if ("repeatX" == name && (n[r] = "&" + e + "repeatX=" + t[i]), "repeatY" == name && (n[r] = "&" + e + "repeatY=" + t[i]), "fixed" == name && (n[r] = "&" + e + "fixed=" + t[i]), "left_pane" == name && null != t[i] && (n[r] = w(t[i], "left_"), r++), "right_pane" == name && null != t[i] && (n[r] = w(t[i], "right_"), r++), "href" == name && (n[r] = "&" + e + "href=" + t[i]), "t" == name) {
                    if ($(t[i]))
                        return;
                    n[r] = "&" + e + "t=" + t[i]
                }
                if ("l" == name) {
                    if ($(t[i]))
                        return;
                    n[r] = "&" + e + "l=" + t[i]
                }
                if ("b" == name) {
                    if ($(t[i]))
                        return;
                    n[r] = "&" + e + "b=" + t[i]
                }
                if ("r" == name) {
                    if ($(t[i]))
                        return;
                    n[r] = "&" + e + "r=" + t[i]
                }
                "tgt" == name && (n[r] = "&" + e + "tgt=" + t[i]), r += 1
            }
            n = Y(n)
        }
        return U(n, H, H, O, O)
    }
    function x() {
        return Je || ge && Ge&&!ze ? (u(0, 0), O) : D
    }
    function _() {
        x() && l(Y(["cmd=", He, "&pos=", me, "&guid=", he]), He)
    }
    function k(t, e) {
        var n = 1 == arguments.length, i = n ? "read-cookie": "write-cookie", o = ["cmd=", i, "&pos=", me, "&guid=", he], c = t || "", a, s, f;
        return !n && e && (typeof e == P ? (c = e.name || t || "", a = e.value || "", f = e.expires || "", s = e.domain || "") : (c = t || "", a = Y(e) || "", f = "", s = ""), o.push("&value=", a, "&exp=", f, "&dm=", s)), c ? (o.push("&cookie=", c), l(Y(o), i), O) : (r(We, i), D)
    }
    function L(t, e) {
        var n = "", r, i;
        return Le ? (r = Le.ownerKey || "", "host" === t ? T() : "from" === t ? E() : e && e !== r ? n : t && e && prop_name === e ? n : t && r && t === r ? n : "ownerKey" === t ? n : "y" === e ? n : (e ? r === e && (i = Le[e]) : i = Le, n = i && t in i ? i[t] : n)) : n
    }
    function A() {
        return be
    }
    function R() {
        var t = z(be && be.self && be.self.iv, - 1, 0), e = 0;
        return t >= 0 && (e = Math.floor(100 * t)), e
    }
    function S() {
        return !!we
    }
    function E() {
        return xe
    }
    function T() {
        return Y(X.URL.ref())
    }
    function M() {
        var t = "collapsed", e;
        return ze ? (e = ze.cmd || "", e == Re || e == ne ? t = "expanding" : e == He ? t = "collapsing" : Ge && (t = "expanded")) : Ge && (t = "expanded"), t
    }
    function C() {
        throw t.Y = H, new Error("api not supported by this site")
    }
    function N(t) {
        var n, r, i, c, a;
        if (!ye) {
            ye = O;
            try {
                t.geom && (i = U(W(t.geom), H, H, D, O), o(i) && (be = i)), t.meta && (Le = U(W(t.meta), H, H, D, O)), he = t.guid, Ae = t.conf, pe = J(Ae.debug), me = t.pos, Be = z(Ae.w, 0), Ve = z(Ae.h, 0), we = J(t.hf), xe = Y(t.fromURL), _e = J(t.flexW), ke = J(t.flexH), i = "supports"in Ae ? Ae.supports : 0, i && "string" == typeof i && (i = J(i)), i === D ? (Ae.supports = D, V.expand = V.collapse = V.bg = C, V.lyr.open = V.lyr.close = V.lyr.msg = C, re[Re] = 0, ue = re) : ue = U(W(Ae.supports), H, H, O, O), (_e || ke) && (ue[Re] = ue[ne] = D)
            } catch (u) {}
            try {
                n = X.URL.ref(), r = n.toHostString()
            } catch (u) {
                n = r = ""
            }
            if (!r ||!he ||!me)
                return;
            return c = t.html, c && (c = a = W(c), t.unescaped = O, a = e(t, a), t.html = a), B && B.attach(he, f), s
        }
    }
    if (!t ||!t.top || t != t.top) {
        var H = null, O=!0, D=!1, P = "object", I = document, j = t.DARLA, $ = isNaN, X = j && j.Lang, U = X && X.ParamHash, W = X && X.ues, F = j && j.Dom, B = F && F.XMsgClient, V = H, G = F && F.elt, Y = X && X.cstr, z = X && X.cnum, J = X && X.cbool, K = X && X.callSafe, Z = RegExp, q = "replace", Q = "sf_", te = [Q + "ver", Q + "ck_on", Q + "flash_ver", Q + "tp_ck_on", Q + "exp_ovr", Q + "exp_push", Q + "rd_ck", Q + "wrt_ck", Q + "bg", Q + "lyr", Q + "cur_iv", Q + "x", Q + "y", Q + "w", Q + "h", Q + "par_x", Q + "par_y", Q + "par_w", Q + "par_h", Q + "par_root", Q + "host"], ee = function() {
            var t = "\\})", e = te.join(t + "|"), n = Z("(" + Q + ")", "g"), r, i, o, c, a, s;
            for (e = e[q](n, "(\\$\\{$1"), e += t, r = Z(e, "g"), i = e.split("|"), c = 0, o = {}; a = te[c];)
                s = i[c++], s = Z(s, "g"), o[a] = s;
            return te = o, r
        }(), ne = "exp-push", re = function() {
            var t = {};
            return t[Re] = 1, t[ne] = t.bg = t.lyr = t["read-cookie"] = t["write-cookie"] = 0, t
        }(), ie = {
            t: 0,
            l: 0,
            r: 0,
            b: 0,
            w: 0,
            h: 0,
            iv: 0,
            xiv: 0,
            yiv: 0
        }, oe = {
            t: 0,
            l: 0,
            r: 0,
            b: 0,
            w: 0,
            h: 0
        }, ce = {
            t: 0,
            l: 0,
            r: 0,
            b: 0,
            w: 0,
            h: 0
        }, ae = {
            t: 0,
            l: 0,
            r: 0,
            b: 0,
            w: 0,
            h: 0
        }, se = {
            self: ie,
            par: oe,
            win: ce,
            exp: ae
        }, fe = "", ue = H, le = 0, de = 0, he = H, pe = 0, me = "", ye = D, ge = D, ve = "", be = se, we = D, xe = "", _e = D, ke = D, Le = H, Ae = H, Re = "exp-ovr", Se = "bg", Ee = "cmsg", Te = 500, Me = "lyr", Ce = Me + "-close", Ne = Me + "-msg", He = "COLLAPSE", Oe = "geom-update", De = "expand", Pe = "collapse", Ie = Pe + "d", je = Se, $e = "clear-bg", Xe = Me, Ue = Ce, We = "failed", Fe = 1e4, Be = 0, Ve = 0, Ge = D, Ye = H, ze = H, Je = D, Ke = H, Ze = D, qe = D, Qe = X && X.time(), tn = 0;
        V = {
            init: N,
            register: d,
            expand: m,
            collapse: _,
            geom: A,
            bg: b,
            msg: h,
            lyr: {
                open: y,
                close: g,
                msg: v
            },
            supports: p,
            inViewPercentage: R,
            winHasFocus: S,
            fromURL: E,
            hostURL: T
        }, X.def("Y.SandBox.vendor", V), X.def("$sf.ext", {
            register: d,
            expand: m,
            collapse: _,
            geom: A,
            supports: p,
            cookie: k,
            meta: L,
            fromURL: E,
            hostURL: T,
            status: M,
            inViewPercentage: R,
            winHasFocus: S,
            msg: h
        })
    }
}(window), function(t) {
    function e(e) {
        var n, r = window, c = 1, a = 0, f;
        try {
            e = e || r.event || {}
        } catch (f) {
            e = {
                type: B
            }, a = 1
        }
        for (; n = tn.shift();)
            try {
                n(e)
        } catch (f) {}
        try {
            Be && Be.call(t)
        } catch (f) {}
        en[F] = Qe[F] = 0, Be = Fe = Ve = Ge = P;
        try {
            $e && (r[Oe] = $e, r[De] = Ue)
        } catch (f) {}
        try {
            Xe && (r[Ne] = Xe, r[He] = We)
        } catch (f) {}
        Ye || Te(r, $, s), Te(r, V, o), Te(r, B, i);
        try {
            r.Unloader = P
        } catch (f) {}
        try {
            r[K] = P
        } catch (f) {}
        try {
            Je && (ke(Je), Je = 0)
        } catch (f) {}
        try {
            Ke && (ke(Ke), Ke = 0)
        } catch (f) {}
        try {
            Ze && (ke(Ze), Ze = 0)
        } catch (f) {}
        return r = t = e = $e = Xe = Ue = We = se = fe = ye = me = Ie = je = n = P, c
    }
    function n() {
        if (ue)
            try {
                Ze && (ke(Ze), Ze = 0), t[Y] && t[Y] != Ge && (Ge = t[Y], t[Y] = P), t[G] && t[G] != Be && (Be = t[G], t[G] = P), t[J] && t[J] != Ve && (Ve = t[J], t[J] = P), t[K] = v, Ze = _e(n, oe)
        } catch (e) {}
    }
    function r() {
        if (ue) {
            try {
                Ke&&-1 != Ke && (ke(Ke), Ke = 0)
            } catch (n) {}
            try {
                if (t == top&&-1 != Ke)
                    return Ke =- 1, he.gc(), o({}), void e()
            } catch (n) {
                Ke =- 1
            }
            try {
                Ke || (Ke = _e(r, ce))
            } catch (n) {}
        }
    }
    function i(t) {
        e(t)
    }
    function o(e) {
        var n = 0, r, i;
        if (Qe[F])
            for (; r = Qe[n++];)
                try {
                    r(e), e.returnValue && (e.returnValue = j)
        } catch (i) {}
        try {
            Ve && Ve.call(t)
        } catch (i) {}
    }
    function c() {
        var t = Ae("iframe"), e = 0, n = le.URL, r = n && n.ref(), i = r && r.host, o, c;
        if (i)
            for (; o = t[e++];)
                if (c = o.src, c = c ? new n(c) : "", c && c == i && "darla" != o.className&&-1 == c.indexOf("msg"))
                    try {
                        Se(o), b("render-err", Ie.pos, "Invalid iframe load detected")
                    } catch (a) {}
    }
    function a() {
        var t = document.body, e = t.clientWidth, n = t.clientHeight, r = 0, i = {
            w: - 1,
            h: - 1
        }, o, c, a, s, f, u, l, d;
        for (s = t.childNodes; a = s[r++];)
            if (1 == a.nodeType && (f = a.tagName || "", f = f.toLowerCase(), "script" != f && "style" != f && "link" != f)) {
                if (a == t)
                    break;
                    u = DARLA.Dom.Geom.rect(a), l = Math.min(u.r, e), d = Math.max(u.l, 0), o = Math.min(u.w, l - d), l = Math.min(u.b, n), d = Math.max(u.t, 0), c = Math.min(u.h, l - d), o > 0 && c > 0 && o > i.w && c > i.h && (i.w = o, i.h = c)
            }
        return i.w <= 0 && (i.w = e), i.h <= 0 && (i.h = n), i
    }
    function s(e) {
        var r = 0, i = 0, o = Ae("a"), f = o[F], u = je && je.tgt || "_blank", l, d, h, p, m, y = Re("fc_align"), g, w, x;
        if (!Ye) {
            for (Ye = I, Te(t, $, s), t[K] = v; (l = o[r++]) && (d = Me(l, "target"), d != u && Me(l, "target", u), !(r > 10)););
            if (y)
                for (o = Ae("img"), r = 0; l = o[r++];) {
                    try {
                        h = Me(l, "width") + "", p = Me(l, "height") + "", "0" !== h && "1" !== h || "0" !== p && "1" !== p ? i++ : (m = he.currentStyle(l), m && "none" != m.display && he.par(l) == y && (l.style.display = "none"))
                    } catch (_) {}
                    if (r > 20)
                        break
            }
            Ie.fif || (g = Ae("div")[F], 2 >= g && 0 == f && 0 == i && (x = Ae("span")[F], w = Ae("iframe")[F], on = Ae("*")[F], 0 == x && 0 == w && 10 >= on - rn && b("render-err", Ie.pos, "No visible content detected")), b("content-size", Ie.pos, be(de(a()))));
            try {
                _e(c, 100), _e(n, oe)
            } catch (_) {}
            try {
                Fe && Fe.call(t)
            } catch (_) {}
            try {
                nn && nn(e)
            } catch (_) {}
        }
    }
    function f(e) {
        var n = 0, r, i = I, o = j;
        if (en[F])
            for (; r = en[n++];) {
                try {
                    i = r.call(t, e)
                } catch (c) {
                    i = I
                }
                if (i === j || e.returnValue === j || e.stopped) {
                    o = I;
                    break
                }
        }
        try {
            !o && Ge && (i = Ge.call(t, e))
        } catch (c) {}
        return i
    }
    function u(e, n, r, i) {
        var o = j, c, a, s;
        if (r ? (a = Ue, s = We) : (a = $e, s = Xe), i)
            try {
                c = s.call(t, e, n, j), o = I
        } catch (f) {
            o = j, c = P
        }
        if (!o && a) {
            try {
                c = a(e, n), o = I
            } catch (f) {
                o = j, c = P
            }
            if (!o)
                try {
                    c = a.call(t, e, n), o = I
            } catch (f) {
                o = j, c = P
            }
        }
        return c
    }
    function l(t, e) {
        return m(t, e, I)
    }
    function d(t, e) {
        return m(t, e)
    }
    function h(t, e) {
        return y(t, e, I)
    }
    function p(t, e) {
        return y(t, e)
    }
    function m(t, e, n) {
        var r = j, i;
        switch (t = be(t)[te]()) {
        case B:
        case G:
            tn.push(e);
            break;
        case V:
        case J:
            Qe.push(e);
            break;
        case U:
        case Y:
            en.push(e);
            break;
        default:
            r = I
        }
        return i = r ? u(t, e, j, n) : I
    }
    function y(t, e, n) {
        var r = 0, i, o;
        switch (t = be(t)[te]()) {
        case B:
        case G:
            o = tn;
            break;
        case V:
        case J:
            o = Qe;
            break;
        case U:
        case Y:
            o = en
        }
        if (o) {
            if (o[F])
                for (; i = o[r];) {
                    if (i === e) {
                        o.splice(r, 1);
                        break
                    }
                    r++
                }
        } else 
            u(t, e, I, n)
    }
    function g() {
        Je && (ke(Je), Je = 0), le.empty(qe) || (qe = de(qe), b("js-err", qe), qe[F] = P)
    }
    function v(t, e, n) {
        return qe || (qe = {}), qe[le.guid("jse")] = {
            l: Ye,
            m: be(["Message: ", t, "\nURL:", e, "\nLine:", n, "\n"])
        }, Je && (ke(Je), Je = 0), Je = _e(g, ie), I
    }
    function b(e) {
        var n = [e, ze], r = arguments, i = r[F], o, c;
        try {
            o = t[ae][ee]
        } catch (a) {
            o = P
        }
        if (o) {
            n = n.concat(le.convertArgs(r, 1));
            try {
                o.msg(n)
            } catch (a) {}
        } else if (me) {
            for (c = ["a=handoff&cmd=msg&pos=", Ie.pos, "&msg=", e, "&guid=", ze, "&args="], idx = 1; i > idx; ++idx)
                c.push(xe(xe(r[idx])), ",");
            c.push("&"), me.send(ze, be(c))
        }
        o = P
    }
    function w() {
        return l
    }
    function x() {
        return h
    }
    function _() {
        return d
    }
    function k() {
        return p
    }
    function L(t) {
        Ve = t
    }
    function A() {
        return Ve
    }
    function R(t) {
        Be = t
    }
    function S() {
        return Be
    }
    function E(t) {
        Fe = t
    }
    function T() {
        return Fe
    }
    function M(t) {
        Ge = t
    }
    function C(t) {
        return Ge
    }
    function N(t) {
        var e = j;
        if (t) {
            if ($e && (t[Oe] = d, t[De] = p), Xe && (t[Ne] = l, t[He] = h), t[Z])
                try {
                    Xe && (t[Z](Ne, w), t[q](Ne, ge), t[Z](He, x), t[q](He, ge)), $e && (t[Z](Oe, _), t[q](Oe, ge), t[Z](De, k), t[q](De, ge)), t[Z](z, T), t[q](z, E), t[Z](G, S), t[q](G, R), t[Z](Y, C), t[q](Y, M), t[Z](J, A), t[q](J, L), e = I
            } catch (n) {
                e = j
            }
            if (Ce[Q]) {
                try {
                    $e && (Ce[Q](t, Oe, {
                        get: _,
                        set: ge
                    }), Ce[Q](t, De, {
                        get: k,
                        set: ge
                    })), Xe && (Ce[Q](t, Ne, {
                        get: w,
                        set: ge
                    }), Ce[Q](t, He, {
                        get: x,
                        set: ge
                    }))
                } catch (n) {
                    e || (e = j)
                }
                try {
                    Ce[Q](t, z, {
                        get: T,
                        set: E
                    }), Ce[Q](t, G, {
                        get: S,
                        set: R
                    }), Ce[Q](t, Y, {
                        get: C,
                        set: M
                    }), Ce[Q](t, J, {
                        get: A,
                        set: L
                    }), e = I
                } catch (n) {
                    e || (e = j)
                }
            }
        }
        return e
    }
    function H(t) {
        return he.evtCancel(t)
    }
    function O() {
        var n, r, c;
        if (je.bg && he.makeCss(be(["#fc { background-color: ", je.bg, "; }"]), "darla_bg_css"), r = je.tgt, r || (r = je.tgt = "_blank"), "_blank" != r && se) {
            for (; Se(Ae("base")[0]););
            n = he.make("base"), Me(n, "target", r), Le(Ae("head")[0], n)
        }
        if (ue && ($e = t[Oe], Ue = t[De]), Xe = t[Ne], We = t[He], Ee(t, V, o), Ee(t, B, i), Ee(t, $, s), Ee(t, U, f, I), Ee(t, "scroll", H, I), N(t), N(t.__proto__), N(t.Window && t.Window.prototype), t.Unloader = {
            fire: e
        }, t[K] = v, Ie.fif) {
            try {
                Me(t.frameElement, "name", P)
            } catch (c) {}
            t.inFIF = t.isAJAX = I
        } else 
            me && me.init(Ie), ye && (nn || (nn = ye.init(Ie)))
    }
    function D() {
        var e, i;
        if (i = be(je.css), i && (i = we(i), he.makeCss(i, "darla_custom_css")), e = be(Ie.html)) {
            Ie.unescaped || (e = we(e));
            try {
                se.write(e)
            } catch (o) {
                v("Error while rendering content: " + o[U])
            }
            t[K] = v
        }
        r(), n()
    }
    if (!t ||!t.top || t != t.top) {
        var P = null, I=!0, j=!1, $ = "load", X = "on", U = "message", W = "before", F = "length", B = "un" + $, V = W + B, G = X + B, Y = X + U, z = X + $, J = X + V, K = X + "error", Z = "__defineGetter__", q = "__defineSetter__", Q = "defineProperty", te = "toLowerCase", ee = "DARLA", ne = "URL", re = "Lang", ie = 1500, oe = 750, ce = 5e3, ae = "parent", se = document, fe = t.DARLA, ue = fe && fe.isIE, le = fe && fe[re], de = le && le.ParamHash, he = fe && fe.Dom, pe = he && he.IFrames, me = he && he.XMsgClient, ye = le && le.ns && le.ns("Y.SandBox.vendor"), ge = le && le.noop, ve = le && le.cnum, be = le && le.cstr, we = le && le.ues, xe = le && le.es, _e = le && le.sto, ke = le && le.cto, Le = he && he.append, Ae = he && he.tags, Re = he && he.elt, Se = he && he.purge, Ee = he && he.attach, Te = he && he.detach, Me = he && he.attr, Ce = Object, Ne = "addEventListener", He = "removeEventListener", Oe = "attachEvent", De = "detachEvent", Pe = "referrer", Ie, je, $e, Xe, Ue, We, Fe, Be, Ve, Ge, Ye = j, ze = "", Je = 0, Ke = 0, Ze = 0, qe = P, Qe = [], tn = [], en = [], nn = P, rn = Ae("*")[F], on = P;
        !function() {
            var e = /^https?\:\/\/([^\/\?]*)(yahoo|tumblr|flickr)\.(net|com)(\:\d+)?([\/\?]|$)/, n, r, i, o, c, a, s, f, u, l, d, h, p, m, y, g, v, b;
            if (top != t && top == t[ae] && de && ve && be && pe && Le && Ae && Re && Se && Ee && Te)
                try {
                    if (g = t.name, - 1 == g.indexOf("guid="))
                        return;
                        if (n = le[ne], Ie = de(g, P, P, I, I), je = Ie.conf, ze = Ie.guid, !ze ||!je)
                            return;
                            Ie.fif = l = ve(Ie.fif, 0, 0);
                            try {
                                h = t[ae][ee][re][ne].loc()
                            } catch (b) {
                                h = P
                            }
                            if (l&&!h)
                                return;
                                if (!l && h) {
                                    if (g = be(h.protocol), !g)
                                        return;
                                        if ( - 1 == g.indexOf("file"))
                                            return;
                                            m = I
                                }
                                if (l&&!e.test(h))
                                    return;
                                    try {
                                        d = n.ref(), p = d.toString()
                                    } catch (b) {
                                        d = P, p = ""
                                    }
                                    try {
                                        y = n.loc()
                                    } catch (b) {
                                        y = P
                                    }
                                    try {
                                        r = be(Ie.hostURL), r = r ? new n(we(r)) : ""
                                    } catch (b) {
                                        r = P
                                    }
                                    if (l && p && (!r || r&&!r.host)&&!m && (r = d, Ie.host = r.protocol + "://" + r.host), !(r && p || m))
                                        return;
                                        if (!l&&!m&&-1 == Ie.host.indexOf(d.host))
                                            return;
                                            if (!y ||!m&&!y.host)
                                                return;
                                                if (!(l || m || Ie.srcHost&&-1 != Ie.srcHost.indexOf(y.host)))
                                                    return;
                                                    if (r && d && (i = be(r.params || ""), o = be(d.params || ""), c = r.path || "", s = d.path || "", !m && r.host && d.host)) {
                                                        if (r.host != d.host)
                                                            return;
                                                            if (r.protocol != d.protocol)
                                                                return;
                                                                if (i != o || c != s) {
                                                                    a = be(r), u = function() {
                                                                        return a
                                                                    }, n.ref = function() {
                                                                        return r
                                                                    };
                                                                    try {
                                                                        Ce[Q](se, Pe, {
                                                                            get: u,
                                                                            set: ge,
                                                                            configurable: j
                                                                        }), f = I
                                                                    } catch (b) {
                                                                        f = j
                                                                    }
                                                                    if (!f)
                                                                        try {
                                                                            se[Z](Pe, u), se[q](Pe, ge)
                                                                        } catch (b) {}
                                                                    }
                                                            }
                                                            t.name = Ie.id || "", O(), D()
                                                } catch (v) {}
        }()
    }
}(window);
