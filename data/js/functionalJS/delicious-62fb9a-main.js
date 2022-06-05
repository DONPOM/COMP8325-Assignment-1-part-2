/*!
 * (c) Delicious Team.
 *
 * delicious-web - v4.0.1 (11-24-2014)
 * http://delicious.com/
 */
(function() {
    function a(a, b, c) {
        for (var d = (c || 0) - 1, e = a ? a.length : 0; ++d < e;)
            if (a[d] === b)
                return d;
        return - 1
    }
    function b(b, c) {
        var d = typeof c;
        if (b = b.cache, "boolean" == d || null == c)
            return b[c] ? 0 : - 1;
        "number" != d && "string" != d && (d = "object");
        var e = "number" == d ? c: r + c;
        return b = (b = b[d]) && b[e], "object" == d ? b && a(b, c)>-1 ? 0 : - 1 : b ? 0 : - 1
    }
    function c(a) {
        var b = this.cache, c = typeof a;
        if ("boolean" == c || null == a)
            b[a]=!0;
        else {
            "number" != c && "string" != c && (c = "object");
            var d = "number" == c ? a: r + a, e = b[c] || (b[c] = {});
            "object" == c ? (e[d] || (e[d] = [])).push(a) : e[d]=!0
        }
    }
    function d(a) {
        return a.charCodeAt(0)
    }
    function e(a, b) {
        for (var c = a.criteria, d = b.criteria, e =- 1, f = c.length; ++e < f;) {
            var g = c[e], h = d[e];
            if (g !== h) {
                if (g > h || "undefined" == typeof g)
                    return 1;
                if (h > g || "undefined" == typeof h)
                    return - 1
            }
        }
        return a.index - b.index
    }
    function f(a) {
        var b =- 1, d = a.length, e = a[0], f = a[d / 2 | 0], g = a[d - 1];
        if (e && "object" == typeof e && f && "object" == typeof f && g && "object" == typeof g)
            return !1;
        var h = i();
        h["false"] = h["null"] = h["true"] = h.undefined=!1;
        var j = i();
        for (j.array = a, j.cache = h, j.push = c; ++b < d;)
            j.push(a[b]);
        return j
    }
    function g(a) {
        return "\\" + V[a]
    }
    function h() {
        return o.pop() || []
    }
    function i() {
        return p.pop() || {
            array: null,
            cache: null,
            criteria: null,
            "false": !1,
            index: 0,
            "null": !1,
            number: null,
            object: null,
            push: null,
            string: null,
            "true": !1,
            undefined: !1,
            value: null
        }
    }
    function j(a) {
        a.length = 0, o.length < t && o.push(a)
    }
    function k(a) {
        var b = a.cache;
        b && k(b), a.array = a.cache = a.criteria = a.object = a.number = a.string = a.value = null, p.length < t && p.push(a)
    }
    function l(a, b, c) {
        b || (b = 0), "undefined" == typeof c && (c = a ? a.length : 0);
        for (var d =- 1, e = c - b || 0, f = Array(0 > e ? 0 : e); ++d < e;)
            f[d] = a[b + d];
        return f
    }
    function m(c) {
        function o(a) {
            return a && "object" == typeof a&&!Zd(a) && Hd.call(a, "__wrapped__") ? a : new p(a)
        }
        function p(a, b) {
            this.__chain__=!!b, this.__wrapped__ = a
        }
        function t(a) {
            function b() {
                if (d) {
                    var a = l(d);
                    Id.apply(a, arguments)
                }
                if (this instanceof b) {
                    var f = X(c.prototype), g = c.apply(f, a || arguments);
                    return Eb(g) ? g : f
                }
                return c.apply(e, a || arguments)
            }
            var c = a[0], d = a[2], e = a[4];
            return Yd(b, a), b
        }
        function V(a, b, c, d, e) {
            if (c) {
                var f = c(a);
                if ("undefined" != typeof f)
                    return f
            }
            var g = Eb(a);
            if (!g)
                return a;
            var i = Ad.call(a);
            if (!R[i])
                return a;
            var k = Wd[i];
            switch (i) {
            case K:
            case L:
                return new k( + a);
            case N:
            case Q:
                return new k(a);
            case P:
                return f = k(a.source, z.exec(a)), f.lastIndex = a.lastIndex, f
            }
            var m = Zd(a);
            if (b) {
                var n=!d;
                d || (d = h()), e || (e = h());
                for (var o = d.length; o--;)
                    if (d[o] == a)
                        return e[o];
                f = m ? k(a.length) : {}
            } else 
                f = m ? l(a) : ee({}, a);
            return m && (Hd.call(a, "index") && (f.index = a.index), Hd.call(a, "input") && (f.input = a.input)), b ? (d.push(a), e.push(f), (m ? Yb : he)(a, function(a, g) {
                f[g] = V(a, b, c, d, e)
            }), n && (j(d), j(e)), f) : f
        }
        function X(a) {
            return Eb(a) ? Nd(a) : {}
        }
        function Y(a, b, c) {
            if ("function" != typeof a)
                return Zc;
            if ("undefined" != typeof b && "prototype"in a) {
                var d = a.__bindData__;
                if ("undefined" == typeof d && (Xd.funcNames && (d=!a.name), d = d ||!Xd.funcDecomp, !d)) {
                    var e = Fd.call(a);
                    Xd.funcNames || (d=!A.test(e)), d || (d = E.test(e), Yd(a, d))
                }
                if (d===!1 || d!==!0 && 1 & d[1])
                    return a;
                switch (c) {
                case 1:
                    return function(c) {
                        return a.call(b, c)
                    };
                case 2:
                    return function(c, d) {
                        return a.call(b, c, d)
                    };
                case 3:
                    return function(c, d, e) {
                        return a.call(b, c, d, e)
                    };
                case 4:
                    return function(c, d, e, f) {
                        return a.call(b, c, d, e, f)
                    }
                }
                return Ic(a, b)
            }
            return a
        }
        function Z(a) {
            function b() {
                var a = i ? g: this;
                if (e) {
                    var o = l(e);
                    Id.apply(o, arguments)
                }
                if ((f || k) && (o || (o = l(arguments)), f && Id.apply(o, f), k && o.length < h))
                    return d|=16, Z([c, m ? d: - 4 & d, o, null, g, h]);
                if (o || (o = arguments), j && (c = a[n]), this instanceof b) {
                    a = X(c.prototype);
                    var p = c.apply(a, o);
                    return Eb(p) ? p : a
                }
                return c.apply(a, o)
            }
            var c = a[0], d = a[1], e = a[2], f = a[3], g = a[4], h = a[5], i = 1 & d, j = 2 & d, k = 4 & d, m = 8 & d, n = c;
            return Yd(b, a), b
        }
        function $(c, d) {
            var e =- 1, g = ib(), h = c ? c.length : 0, i = h >= s && g === a, j = [];
            if (i) {
                var l = f(d);
                l ? (g = b, d = l) : i=!1
            }
            for (; ++e < h;) {
                var m = c[e];
                g(d, m) < 0 && j.push(m)
            }
            return i && k(d), j
        }
        function ab(a, b, c, d) {
            for (var e = (d || 0) - 1, f = a ? a.length : 0, g = []; ++e < f;) {
                var h = a[e];
                if (h && "object" == typeof h && "number" == typeof h.length && (Zd(h) || mb(h))) {
                    b || (h = ab(h, b, c));
                    var i =- 1, j = h.length, k = g.length;
                    for (g.length += j; ++i < j;)
                        g[k++] = h[i]
                } else 
                    c || g.push(h)
            }
            return g
        }
        function bb(a, b, c, d, e, f) {
            if (c) {
                var g = c(a, b);
                if ("undefined" != typeof g)
                    return !!g
            }
            if (a === b)
                return 0 !== a || 1 / a == 1 / b;
            var i = typeof a, k = typeof b;
            if (!(a !== a || a && U[i] || b && U[k]))
                return !1;
            if (null == a || null == b)
                return a === b;
            var l = Ad.call(a), m = Ad.call(b);
            if (l == I && (l = O), m == I && (m = O), l != m)
                return !1;
            switch (l) {
            case K:
            case L:
                return + a ==+ b;
            case N:
                return a!=+a ? b!=+b : 0 == a ? 1 / a == 1 / b : a ==+ b;
            case P:
            case Q:
                return a == vd(b)
            }
            var n = l == J;
            if (!n) {
                var o = Hd.call(a, "__wrapped__"), p = Hd.call(b, "__wrapped__");
                if (o || p)
                    return bb(o ? a.__wrapped__ : a, p ? b.__wrapped__ : b, c, d, e, f);
                if (l != O)
                    return !1;
                var q = a.constructor, r = b.constructor;
                if (q != r&&!(Db(q) && q instanceof q && Db(r) && r instanceof r) && "constructor"in a && "constructor"in b)
                    return !1
            }
            var s=!e;
            e || (e = h()), f || (f = h());
            for (var t = e.length; t--;)
                if (e[t] == a)
                    return f[t] == b;
            var u = 0;
            if (g=!0, e.push(a), f.push(b), n) {
                if (t = a.length, u = b.length, g = u == t, g || d)
                    for (; u--;) {
                        var v = t, w = b[u];
                        if (d)
                            for (; v--&&!(g = bb(a[v], w, c, d, e, f)););
                        else if (!(g = bb(a[u], w, c, d, e, f)))
                            break
                    }
            } else 
                ge(b, function(b, h, i) {
                    return Hd.call(i, h) ? (u++, g = Hd.call(a, h) && bb(a[h], b, c, d, e, f)) : void 0
                }), g&&!d && ge(a, function(a, b, c) {
                    return Hd.call(c, b) ? g=--u>-1 : void 0
                });
            return e.pop(), f.pop(), s && (j(e), j(f)), g
        }
        function cb(a, b, c, d, e) {
            (Zd(b) ? Yb : he)(b, function(b, f) {
                var g, h, i = b, j = a[f];
                if (b && ((h = Zd(b)) || ie(b))) {
                    for (var k = d.length; k--;)
                        if (g = d[k] == b) {
                            j = e[k];
                            break
                        }
                    if (!g) {
                        var l;
                        c && (i = c(j, b), (l = "undefined" != typeof i) && (j = i)), l || (j = h ? Zd(j) ? j : [] : ie(j) ? j : {}), d.push(b), e.push(j), l || cb(j, b, c, d, e)
                    }
                } else 
                    c && (i = c(j, b), "undefined" == typeof i && (i = b)), "undefined" != typeof i && (j = i);
                a[f] = j
            })
        }
        function db(a, b) {
            return a + Ed(Vd() * (b - a + 1))
        }
        function eb(c, d, e) {
            var g =- 1, i = ib(), l = c ? c.length : 0, m = [], n=!d && l >= s && i === a, o = e || n ? h() : m;
            if (n) {
                var p = f(o);
                i = b, o = p
            }
            for (; ++g < l;) {
                var q = c[g], r = e ? e(q, g, c): q;
                (d?!g || o[o.length - 1] !== r : i(o, r) < 0) && ((e || n) && o.push(r), m.push(q))
            }
            return n ? (j(o.array), k(o)) : e && j(o), m
        }
        function fb(a) {
            return function(b, c, d) {
                var e = {};
                c = o.createCallback(c, d, 3);
                var f =- 1, g = b ? b.length : 0;
                if ("number" == typeof g)
                    for (; ++f < g;) {
                        var h = b[f];
                        a(e, h, c(h, f, b), b)
                    } else 
                        he(b, function(b, d, f) {
                            a(e, b, c(b, d, f), f)
                        });
                return e
            }
        }
        function gb(a, b, c, d, e, f) {
            var g = 1 & b, h = 2 & b, i = 4 & b, j = 16 & b, k = 32 & b;
            if (!h&&!Db(a))
                throw new wd;
            j&&!c.length && (b&=-17, j = c=!1), k&&!d.length && (b&=-33, k = d=!1);
            var m = a && a.__bindData__;
            if (m && m!==!0)
                return m = l(m), m[2] && (m[2] = l(m[2])), m[3] && (m[3] = l(m[3])), g&&!(1 & m[1]) && (m[4] = e), !g && 1 & m[1] && (b|=8), i&&!(4 & m[1]) && (m[5] = f), j && Id.apply(m[2] || (m[2] = []), c), k && Ld.apply(m[3] || (m[3] = []), d), m[1]|=b, gb.apply(null, m);
            var n = 1 == b || 17 === b ? t: Z;
            return n([a, b, c, d, e, f])
        }
        function hb(a) {
            return ae[a]
        }
        function ib() {
            var b = (b = o.indexOf) === rc ? a: b;
            return b
        }
        function jb(a) {
            return "function" == typeof a && Bd.test(a)
        }
        function kb(a) {
            var b, c;
            return a && Ad.call(a) == O && (b = a.constructor, !Db(b) || b instanceof b) ? (ge(a, function(a, b) {
                c = b
            }), "undefined" == typeof c || Hd.call(a, c)) : !1
        }
        function lb(a) {
            return be[a]
        }
        function mb(a) {
            return a && "object" == typeof a && "number" == typeof a.length && Ad.call(a) == I ||!1
        }
        function nb(a, b, c, d) {
            return "boolean" != typeof b && null != b && (d = c, c = b, b=!1), V(a, b, "function" == typeof c && Y(c, d, 1))
        }
        function ob(a, b, c) {
            return V(a, !0, "function" == typeof b && Y(b, c, 1))
        }
        function pb(a, b) {
            var c = X(a);
            return b ? ee(c, b) : c
        }
        function qb(a, b, c) {
            var d;
            return b = o.createCallback(b, c, 3), he(a, function(a, c, e) {
                return b(a, c, e) ? (d = c, !1) : void 0
            }), d
        }
        function rb(a, b, c) {
            var d;
            return b = o.createCallback(b, c, 3), tb(a, function(a, c, e) {
                return b(a, c, e) ? (d = c, !1) : void 0
            }), d
        }
        function sb(a, b, c) {
            var d = [];
            ge(a, function(a, b) {
                d.push(b, a)
            });
            var e = d.length;
            for (b = Y(b, c, 3); e--&&b(d[e--], d[e], a)
                !==!1;
            );
            return a
        }
        function tb(a, b, c) {
            var d = _d(a), e = d.length;
            for (b = Y(b, c, 3); e--;) {
                var f = d[e];
                if (b(a[f], f, a)===!1)
                    break
            }
            return a
        }
        function ub(a) {
            var b = [];
            return ge(a, function(a, c) {
                Db(a) && b.push(c)
            }), b.sort()
        }
        function vb(a, b) {
            return a ? Hd.call(a, b) : !1
        }
        function wb(a) {
            for (var b =- 1, c = _d(a), d = c.length, e = {}; ++b < d;) {
                var f = c[b];
                e[a[f]] = f
            }
            return e
        }
        function xb(a) {
            return a===!0 || a===!1 || a && "object" == typeof a && Ad.call(a) == K ||!1
        }
        function yb(a) {
            return a && "object" == typeof a && Ad.call(a) == L ||!1
        }
        function zb(a) {
            return a && 1 === a.nodeType ||!1
        }
        function Ab(a) {
            var b=!0;
            if (!a)
                return b;
            var c = Ad.call(a), d = a.length;
            return c == J || c == Q || c == I || c == O && "number" == typeof d && Db(a.splice)?!d : (he(a, function() {
                return b=!1
            }), b)
        }
        function Bb(a, b, c, d) {
            return bb(a, b, "function" == typeof c && Y(c, d, 2))
        }
        function Cb(a) {
            return Pd(a)&&!Qd(parseFloat(a))
        }
        function Db(a) {
            return "function" == typeof a
        }
        function Eb(a) {
            return !!a&&!!U[typeof a]
        }
        function Fb(a) {
            return Hb(a) && a!=+a
        }
        function Gb(a) {
            return null === a
        }
        function Hb(a) {
            return "number" == typeof a || a && "object" == typeof a && Ad.call(a) == N ||!1
        }
        function Ib(a) {
            return a && "object" == typeof a && Ad.call(a) == P ||!1
        }
        function Jb(a) {
            return "string" == typeof a || a && "object" == typeof a && Ad.call(a) == Q ||!1
        }
        function Kb(a) {
            return "undefined" == typeof a
        }
        function Lb(a, b, c) {
            var d = {};
            return b = o.createCallback(b, c, 3), he(a, function(a, c, e) {
                d[c] = b(a, c, e)
            }), d
        }
        function Mb(a) {
            var b = arguments, c = 2;
            if (!Eb(a))
                return a;
            if ("number" != typeof b[2] && (c = b.length), c > 3 && "function" == typeof b[c - 2])
                var d = Y(b[--c - 1], b[c--], 2);
            else 
                c > 2 && "function" == typeof b[c - 1] && (d = b[--c]);
            for (var e = l(arguments, 1, c), f =- 1, g = h(), i = h(); ++f < c;)
                cb(a, e[f], d, g, i);
            return j(g), j(i), a
        }
        function Nb(a, b, c) {
            var d = {};
            if ("function" != typeof b) {
                var e = [];
                ge(a, function(a, b) {
                    e.push(b)
                }), e = $(e, ab(arguments, !0, !1, 1));
                for (var f =- 1, g = e.length; ++f < g;) {
                    var h = e[f];
                    d[h] = a[h]
                }
            } else 
                b = o.createCallback(b, c, 3), ge(a, function(a, c, e) {
                    b(a, c, e) || (d[c] = a)
                });
            return d
        }
        function Ob(a) {
            for (var b =- 1, c = _d(a), d = c.length, e = nd(d); ++b < d;) {
                var f = c[b];
                e[b] = [f, a[f]]
            }
            return e
        }
        function Pb(a, b, c) {
            var d = {};
            if ("function" != typeof b)
                for (var e =- 1, f = ab(arguments, !0, !1, 1), g = Eb(a) ? f.length : 0; ++e < g;) {
                    var h = f[e];
                    h in a && (d[h] = a[h])
                } else 
                    b = o.createCallback(b, c, 3), ge(a, function(a, c, e) {
                        b(a, c, e) && (d[c] = a)
                    });
            return d
        }
        function Qb(a, b, c, d) {
            var e = Zd(a);
            if (null == c)
                if (e)
                    c = [];
                else {
                    var f = a && a.constructor, g = f && f.prototype;
                    c = X(g)
                }
            return b && (b = o.createCallback(b, d, 4), (e ? Yb : he)(a, function(a, d, e) {
                return b(c, a, d, e)
            })), c
        }
        function Rb(a) {
            for (var b =- 1, c = _d(a), d = c.length, e = nd(d); ++b < d;)
                e[b] = a[c[b]];
            return e
        }
        function Sb(a) {
            for (var b = arguments, c =- 1, d = ab(b, !0, !1, 1), e = b[2] && b[2][b[1]] === a ? 1 : d.length, f = nd(e); ++c < e;)
                f[c] = a[d[c]];
            return f
        }
        function Tb(a, b, c) {
            var d =- 1, e = ib(), f = a ? a.length : 0, g=!1;
            return c = (0 > c ? Sd(0, f + c) : c) || 0, Zd(a) ? g = e(a, b, c)>-1 : "number" == typeof f ? g = (Jb(a) ? a.indexOf(b, c) : e(a, b, c))>-1 : he(a, function(a) {
                return ++d >= c?!(g = a === b) : void 0
            }), g
        }
        function Ub(a, b, c) {
            var d=!0;
            b = o.createCallback(b, c, 3);
            var e =- 1, f = a ? a.length : 0;
            if ("number" == typeof f)
                for (; ++e < f && (d=!!b(a[e], e, a)););
            else 
                he(a, function(a, c, e) {
                    return d=!!b(a, c, e)
                });
            return d
        }
        function Vb(a, b, c) {
            var d = [];
            b = o.createCallback(b, c, 3);
            var e =- 1, f = a ? a.length : 0;
            if ("number" == typeof f)
                for (; ++e < f;) {
                    var g = a[e];
                    b(g, e, a) && d.push(g)
                } else 
                    he(a, function(a, c, e) {
                        b(a, c, e) && d.push(a)
                    });
            return d
        }
        function Wb(a, b, c) {
            b = o.createCallback(b, c, 3);
            var d =- 1, e = a ? a.length : 0;
            if ("number" != typeof e) {
                var f;
                return he(a, function(a, c, d) {
                    return b(a, c, d) ? (f = a, !1) : void 0
                }), f
            }
            for (; ++d < e;) {
                var g = a[d];
                if (b(g, d, a))
                    return g
            }
        }
        function Xb(a, b, c) {
            var d;
            return b = o.createCallback(b, c, 3), Zb(a, function(a, c, e) {
                return b(a, c, e) ? (d = a, !1) : void 0
            }), d
        }
        function Yb(a, b, c) {
            var d =- 1, e = a ? a.length : 0;
            if (b = b && "undefined" == typeof c ? b : Y(b, c, 3), "number" == typeof e)
                for (; ++d < e && b(a[d], d, a)!==!1;);
            else 
                he(a, b);
            return a
        }
        function Zb(a, b, c) {
            var d = a ? a.length: 0;
            if (b = b && "undefined" == typeof c ? b : Y(b, c, 3), "number" == typeof d)
                for (; d--&&b(a[d], d, a)!==!1;);
            else {
                var e = _d(a);
                d = e.length, he(a, function(a, c, f) {
                    return c = e ? e[--d] : --d, b(f[c], c, f)
                })
            }
            return a
        }
        function $b(a, b) {
            var c = l(arguments, 2), d =- 1, e = "function" == typeof b, f = a ? a.length : 0, g = nd("number" == typeof f ? f : 0);
            return Yb(a, function(a) {
                g[++d] = (e ? b : a[b]).apply(a, c)
            }), g
        }
        function _b(a, b, c) {
            var d =- 1, e = a ? a.length : 0;
            if (b = o.createCallback(b, c, 3), "number" == typeof e)
                for (var f = nd(e); ++d < e;)
                    f[d] = b(a[d], d, a);
            else 
                f = [], he(a, function(a, c, e) {
                    f[++d] = b(a, c, e)
                });
            return f
        }
        function ac(a, b, c) {
            var e =- 1 / 0, f = e;
            if ("function" != typeof b && c && c[b] === a && (b = null), null == b && Zd(a))
                for (var g =- 1, h = a.length; ++g < h;) {
                    var i = a[g];
                    i > f && (f = i)
                } else 
                    b = null == b && Jb(a) ? d : o.createCallback(b, c, 3), Yb(a, function(a, c, d) {
                        var g = b(a, c, d);
                        g > e && (e = g, f = a)
                    });
            return f
        }
        function bc(a, b, c) {
            var e = 1 / 0, f = e;
            if ("function" != typeof b && c && c[b] === a && (b = null), null == b && Zd(a))
                for (var g =- 1, h = a.length; ++g < h;) {
                    var i = a[g];
                    f > i && (f = i)
                } else 
                    b = null == b && Jb(a) ? d : o.createCallback(b, c, 3), Yb(a, function(a, c, d) {
                        var g = b(a, c, d);
                        e > g && (e = g, f = a)
                    });
            return f
        }
        function cc(a, b, c, d) {
            if (!a)
                return c;
            var e = arguments.length < 3;
            b = o.createCallback(b, d, 4);
            var f =- 1, g = a.length;
            if ("number" == typeof g)
                for (e && (c = a[++f]); ++f < g;)
                    c = b(c, a[f], f, a);
            else 
                he(a, function(a, d, f) {
                    c = e ? (e=!1, a) : b(c, a, d, f)
                });
            return c
        }
        function dc(a, b, c, d) {
            var e = arguments.length < 3;
            return b = o.createCallback(b, d, 4), Zb(a, function(a, d, f) {
                c = e ? (e=!1, a) : b(c, a, d, f)
            }), c
        }
        function ec(a, b, c) {
            return b = o.createCallback(b, c, 3), Vb(a, function(a, c, d) {
                return !b(a, c, d)
            })
        }
        function fc(a, b, c) {
            if (a && "number" != typeof a.length && (a = Rb(a)), null == b || c)
                return a ? a[db(0, a.length - 1)] : n;
            var d = gc(a);
            return d.length = Td(Sd(0, b), d.length), d
        }
        function gc(a) {
            var b =- 1, c = a ? a.length : 0, d = nd("number" == typeof c ? c : 0);
            return Yb(a, function(a) {
                var c = db(0, ++b);
                d[b] = d[c], d[c] = a
            }), d
        }
        function hc(a) {
            var b = a ? a.length: 0;
            return "number" == typeof b ? b : _d(a).length
        }
        function ic(a, b, c) {
            var d;
            b = o.createCallback(b, c, 3);
            var e =- 1, f = a ? a.length : 0;
            if ("number" == typeof f)
                for (; ++e < f&&!(d = b(a[e], e, a)););
            else 
                he(a, function(a, c, e) {
                    return !(d = b(a, c, e))
                });
            return !!d
        }
        function jc(a, b, c) {
            var d =- 1, f = Zd(b), g = a ? a.length : 0, l = nd("number" == typeof g ? g : 0);
            for (f || (b = o.createCallback(b, c, 3)), Yb(a, function(a, c, e) {
                var g = l[++d] = i();
                f ? g.criteria = _b(b, function(b) {
                    return a[b]
                }) : (g.criteria = h())[0] = b(a, c, e), g.index = d, g.value = a
            }), g = l.length, l.sort(e); g--;) {
                var m = l[g];
                l[g] = m.value, f || j(m.criteria), k(m)
            }
            return l
        }
        function kc(a) {
            return a && "number" == typeof a.length ? l(a) : Rb(a)
        }
        function lc(a) {
            for (var b =- 1, c = a ? a.length : 0, d = []; ++b < c;) {
                var e = a[b];
                e && d.push(e)
            }
            return d
        }
        function mc(a) {
            return $(a, ab(arguments, !0, !0, 1))
        }
        function nc(a, b, c) {
            var d =- 1, e = a ? a.length : 0;
            for (b = o.createCallback(b, c, 3); ++d < e;)
                if (b(a[d], d, a))
                    return d;
            return - 1
        }
        function oc(a, b, c) {
            var d = a ? a.length: 0;
            for (b = o.createCallback(b, c, 3); d--;)
                if (b(a[d], d, a))
                    return d;
            return - 1
        }
        function pc(a, b, c) {
            var d = 0, e = a ? a.length: 0;
            if ("number" != typeof b && null != b) {
                var f =- 1;
                for (b = o.createCallback(b, c, 3); ++f < e && b(a[f], f, a);)
                    d++
            } else if (d = b, null == d || c)
                return a ? a[0] : n;
            return l(a, 0, Td(Sd(0, d), e))
        }
        function qc(a, b, c, d) {
            return "boolean" != typeof b && null != b && (d = c, c = "function" != typeof b && d && d[b] === a ? null : b, b=!1), null != c && (a = _b(a, c, d)), ab(a, b)
        }
        function rc(b, c, d) {
            if ("number" == typeof d) {
                var e = b ? b.length: 0;
                d = 0 > d ? Sd(0, e + d) : d || 0
            } else if (d) {
                var f = Ac(b, c);
                return b[f] === c ? f : - 1
            }
            return a(b, c, d)
        }
        function sc(a, b, c) {
            var d = 0, e = a ? a.length: 0;
            if ("number" != typeof b && null != b) {
                var f = e;
                for (b = o.createCallback(b, c, 3); f--&&b(a[f], f, a);
                )d++
            } else 
                d = null == b || c ? 1 : b || d;
            return l(a, 0, Td(Sd(0, e - d), e))
        }
        function tc() {
            for (var c = [], d =- 1, e = arguments.length, g = h(), i = ib(), l = i === a, m = h(); ++d < e;) {
                var n = arguments[d];
                (Zd(n) || mb(n)) && (c.push(n), g.push(l && n.length >= s && f(d ? c[d] : m)))
            }
            var o = c[0], p =- 1, q = o ? o.length : 0, r = [];
            a: for (; ++p < q;) {
                var t = g[0];
                if (n = o[p], (t ? b(t, n) : i(m, n)) < 0) {
                    for (d = e, (t || m).push(n); --d;)
                        if (t = g[d], (t ? b(t, n) : i(c[d], n)) < 0)
                            continue a;
                    r.push(n)
                }
            }
            for (; e--;)
                t = g[e], t && k(t);
            return j(g), j(m), r
        }
        function uc(a, b, c) {
            var d = 0, e = a ? a.length: 0;
            if ("number" != typeof b && null != b) {
                var f = e;
                for (b = o.createCallback(b, c, 3); f--&&b(a[f], f, a);
                )d++
            } else if (d = b, null == d || c)
                return a ? a[e - 1] : n;
            return l(a, Sd(0, e - d))
        }
        function vc(a, b, c) {
            var d = a ? a.length: 0;
            for ("number" == typeof c && (d = (0 > c ? Sd(0, d + c) : Td(c, d - 1)) + 1); d--;)
                if (a[d] === b)
                    return d;
            return - 1
        }
        function wc(a) {
            for (var b = arguments, c = 0, d = b.length, e = a ? a.length : 0; ++c < d;)
                for (var f =- 1, g = b[c]; ++f < e;)
                    a[f] === g && (Kd.call(a, f--, 1), e--);
            return a
        }
        function xc(a, b, c) {
            a =+ a || 0, c = "number" == typeof c ? c : + c || 1, null == b && (b = a, a = 0);
            for (var d =- 1, e = Sd(0, Cd((b - a) / (c || 1))), f = nd(e); ++d < e;)
                f[d] = a, a += c;
            return f
        }
        function yc(a, b, c) {
            var d =- 1, e = a ? a.length : 0, f = [];
            for (b = o.createCallback(b, c, 3); ++d < e;) {
                var g = a[d];
                b(g, d, a) && (f.push(g), Kd.call(a, d--, 1), e--)
            }
            return f
        }
        function zc(a, b, c) {
            if ("number" != typeof b && null != b) {
                var d = 0, e =- 1, f = a ? a.length : 0;
                for (b = o.createCallback(b, c, 3); ++e < f && b(a[e], e, a);)
                    d++
            } else 
                d = null == b || c ? 1 : Sd(0, b);
            return l(a, d)
        }
        function Ac(a, b, c, d) {
            var e = 0, f = a ? a.length: e;
            for (c = c ? o.createCallback(c, d, 1) : Zc, b = c(b); f > e;) {
                var g = e + f>>>1;
                c(a[g]) < b ? e = g + 1 : f = g
            }
            return e
        }
        function Bc() {
            return eb(ab(arguments, !0, !0))
        }
        function Cc(a, b, c, d) {
            return "boolean" != typeof b && null != b && (d = c, c = "function" != typeof b && d && d[b] === a ? null : b, b=!1), null != c && (c = o.createCallback(c, d, 3)), eb(a, b, c)
        }
        function Dc(a) {
            return $(a, l(arguments, 1))
        }
        function Ec() {
            for (var a =- 1, b = arguments.length; ++a < b;) {
                var c = arguments[a];
                if (Zd(c) || mb(c))
                    var d = d ? eb($(d, c).concat($(c, d))): c
            }
            return d || []
        }
        function Fc() {
            for (var a = arguments.length > 1 ? arguments : arguments[0], b =- 1, c = a ? ac(me(a, "length")) : 0, d = nd(0 > c ? 0 : c); ++b < c;)
                d[b] = me(a, b);
            return d
        }
        function Gc(a, b) {
            var c =- 1, d = a ? a.length : 0, e = {};
            for (!b && d&&!Zd(a[0]) && (b = []); ++c < d;) {
                var f = a[c];
                b ? e[f] = b[c] : f && (e[f[0]] = f[1])
            }
            return e
        }
        function Hc(a, b) {
            if (!Db(b))
                throw new wd;
            return function() {
                return --a < 1 ? b.apply(this, arguments) : void 0
            }
        }
        function Ic(a, b) {
            return arguments.length > 2 ? gb(a, 17, l(arguments, 2), null, b) : gb(a, 1, null, null, b)
        }
        function Jc(a) {
            for (var b = arguments.length > 1 ? ab(arguments, !0, !1, 1) : ub(a), c =- 1, d = b.length; ++c < d;) {
                var e = b[c];
                a[e] = gb(a[e], 1, null, null, a)
            }
            return a
        }
        function Kc(a, b) {
            return arguments.length > 2 ? gb(b, 19, l(arguments, 2), null, a) : gb(b, 3, null, null, a)
        }
        function Lc() {
            for (var a = arguments, b = a.length; b--;)
                if (!Db(a[b]))
                    throw new wd;
            return function() {
                for (var b = arguments, c = a.length; c--;)
                    b = [a[c].apply(this, b)];
                return b[0]
            }
        }
        function Mc(a, b) {
            return b = "number" == typeof b ? b : + b || a.length, gb(a, 4, null, null, null, b)
        }
        function Nc(a, b, c) {
            var d, e, f, g, h, i, j, k = 0, l=!1, m=!0;
            if (!Db(a))
                throw new wd;
            if (b = Sd(0, b) || 0, c===!0) {
                var o=!0;
                m=!1
            } else 
                Eb(c) && (o = c.leading, l = "maxWait"in c && (Sd(b, c.maxWait) || 0), m = "trailing"in c ? c.trailing : m);
            var p = function() {
                var c = b - (oe() - g);
                if (0 >= c) {
                    e && Dd(e);
                    var l = j;
                    e = i = j = n, l && (k = oe(), f = a.apply(h, d), !i&&!e && (d = h = null))
                } else 
                    i = Jd(p, c)
            }, q = function() {
                i && Dd(i), e = i = j = n, (m || l !== b) && (k = oe(), f = a.apply(h, d), !i&&!e && (d = h = null))
            };
            return function() {
                if (d = arguments, g = oe(), h = this, j = m && (i ||!o), l===!1)
                    var c = o&&!i;
                else {
                    !e&&!o && (k = g);
                    var n = l - (g - k), r = 0 >= n;
                    r ? (e && (e = Dd(e)), k = g, f = a.apply(h, d)) : e || (e = Jd(q, n))
                }
                return r && i ? i = Dd(i) : !i && b !== l && (i = Jd(p, b)), c && (r=!0, f = a.apply(h, d)), r&&!i&&!e && (d = h = null), f
            }
        }
        function Oc(a) {
            if (!Db(a))
                throw new wd;
            var b = l(arguments, 1);
            return Jd(function() {
                a.apply(n, b)
            }, 1)
        }
        function Pc(a, b) {
            if (!Db(a))
                throw new wd;
            var c = l(arguments, 2);
            return Jd(function() {
                a.apply(n, c)
            }, b)
        }
        function Qc(a, b) {
            if (!Db(a))
                throw new wd;
            var c = function() {
                var d = c.cache, e = b ? b.apply(this, arguments): r + arguments[0];
                return Hd.call(d, e) ? d[e] : d[e] = a.apply(this, arguments)
            };
            return c.cache = {}, c
        }
        function Rc(a) {
            var b, c;
            if (!Db(a))
                throw new wd;
            return function() {
                return b ? c : (b=!0, c = a.apply(this, arguments), a = null, c)
            }
        }
        function Sc(a) {
            return gb(a, 16, l(arguments, 1))
        }
        function Tc(a) {
            return gb(a, 32, null, l(arguments, 1))
        }
        function Uc(a, b, c) {
            var d=!0, e=!0;
            if (!Db(a))
                throw new wd;
            return c===!1 ? d=!1 : Eb(c) && (d = "leading"in c ? c.leading : d, e = "trailing"in c ? c.trailing : e), S.leading = d, S.maxWait = b, S.trailing = e, Nc(a, b, S)
        }
        function Vc(a, b) {
            return gb(b, 16, [a])
        }
        function Wc(a) {
            return function() {
                return a
            }
        }
        function Xc(a, b, c) {
            var d = typeof a;
            if (null == a || "function" == d)
                return Y(a, b, c);
            if ("object" != d)
                return bd(a);
            var e = _d(a), f = e[0], g = a[f];
            return 1 != e.length || g !== g || Eb(g) ? function(b) {
                for (var c = e.length, d=!1; c--&&(d = bb(b[e[c]], a[e[c]], null, !0)););
                return d
            } : function(a) {
                var b = a[f];
                return g === b && (0 !== g || 1 / g == 1 / b)
            }
        }
        function Yc(a) {
            return null == a ? "" : vd(a).replace(de, hb)
        }
        function Zc(a) {
            return a
        }
        function $c(a, b, c) {
            var d=!0, e = b && ub(b);
            b && (c || e.length) || (null == c && (c = b), f = p, b = a, a = o, e = ub(b)), c===!1 ? d=!1 : Eb(c) && "chain"in c && (d = c.chain);
            var f = a, g = Db(f);
            Yb(e, function(c) {
                var e = a[c] = b[c];
                g && (f.prototype[c] = function() {
                    var b = this.__chain__, c = this.__wrapped__, g = [c];
                    Id.apply(g, arguments);
                    var h = e.apply(a, g);
                    if (d || b) {
                        if (c === h && Eb(h))
                            return this;
                        h = new f(h), h.__chain__ = b
                    }
                    return h
                })
            })
        }
        function _c() {
            return c._ = zd, this
        }
        function ad() {}
        function bd(a) {
            return function(b) {
                return b[a]
            }
        }
        function cd(a, b, c) {
            var d = null == a, e = null == b;
            if (null == c && ("boolean" == typeof a && e ? (c = a, a = 1) : !e && "boolean" == typeof b && (c = b, e=!0)), d && e && (b = 1), a =+ a || 0, e ? (b = a, a = 0) : b =+ b || 0, c || a%1 || b%1) {
                var f = Vd();
                return Td(a + f * (b - a + parseFloat("1e-" + ((f + "").length - 1))), b)
            }
            return db(a, b)
        }
        function dd(a, b) {
            if (a) {
                var c = a[b];
                return Db(c) ? a[b]() : c
            }
        }
        function ed(a, b, c) {
            var d = o.templateSettings;
            a = vd(a || ""), c = fe({}, c, d);
            var e, f = fe({}, c.imports, d.imports), h = _d(f), i = Rb(f), j = 0, k = c.interpolate || D, l = "__p += '", m = ud((c.escape || D).source + "|" + k.source + "|" + (k === B ? y : D).source + "|" + (c.evaluate || D).source + "|$", "g");
            a.replace(m, function(b, c, d, f, h, i) {
                return d || (d = f), l += a.slice(j, i).replace(F, g), c && (l += "' +\n__e(" + c + ") +\n'"), h && (e=!0, l += "';\n" + h + ";\n__p += '"), d && (l += "' +\n((__t = (" + d + ")) == null ? '' : __t) +\n'"), j = i + b.length, b
            }), l += "';\n";
            var p = c.variable, q = p;
            q || (p = "obj", l = "with (" + p + ") {\n" + l + "\n}\n"), l = (e ? l.replace(v, "") : l).replace(w, "$1").replace(x, "$1;"), l = "function(" + p + ") {\n" + (q ? "" : p + " || (" + p + " = {});\n") + "var __t, __p = '', __e = _.escape" + (e ? ", __j = Array.prototype.join;\nfunction print() { __p += __j.call(arguments, '') }\n" : ";\n") + l + "return __p\n}";
            var r = "\n/*\n//# sourceURL=" + (c.sourceURL || "/lodash/template/source[" + H++ + "]") + "\n*/";
            try {
                var s = qd(h, "return " + l + r).apply(n, i)
            } catch (t) {
                throw t.source = l, t
            }
            return b ? s(b) : (s.source = l, s)
        }
        function fd(a, b, c) {
            a = (a =+ a)>-1 ? a : 0;
            var d =- 1, e = nd(a);
            for (b = Y(b, c, 1); ++d < a;)
                e[d] = b(d);
            return e
        }
        function gd(a) {
            return null == a ? "" : vd(a).replace(ce, lb)
        }
        function hd(a) {
            var b=++q;
            return vd(null == a ? "" : a) + b
        }
        function id(a) {
            return a = new p(a), a.__chain__=!0, a
        }
        function jd(a, b) {
            return b(a), a
        }
        function kd() {
            return this.__chain__=!0, this
        }
        function ld() {
            return vd(this.__wrapped__)
        }
        function md() {
            return this.__wrapped__
        }
        c = c ? _.defaults(W.Object(), c, _.pick(W, G)) : W;
        var nd = c.Array, od = c.Boolean, pd = c.Date, qd = c.Function, rd = c.Math, sd = c.Number, td = c.Object, ud = c.RegExp, vd = c.String, wd = c.TypeError, xd = [], yd = td.prototype, zd = c._, Ad = yd.toString, Bd = ud("^" + vd(Ad).replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/toString| for [^\]]+/g, ".*?") + "$"), Cd = rd.ceil, Dd = c.clearTimeout, Ed = rd.floor, Fd = qd.prototype.toString, Gd = jb(Gd = td.getPrototypeOf) && Gd, Hd = yd.hasOwnProperty, Id = xd.push, Jd = c.setTimeout, Kd = xd.splice, Ld = xd.unshift, Md = function() {
            try {
                var a = {}, b = jb(b = td.defineProperty) && b, c = b(a, a, a) && b
            } catch (d) {}
            return c
        }(), Nd = jb(Nd = td.create) && Nd, Od = jb(Od = nd.isArray) && Od, Pd = c.isFinite, Qd = c.isNaN, Rd = jb(Rd = td.keys) && Rd, Sd = rd.max, Td = rd.min, Ud = c.parseInt, Vd = rd.random, Wd = {};
        Wd[J] = nd, Wd[K] = od, Wd[L] = pd, Wd[M] = qd, Wd[O] = td, Wd[N] = sd, Wd[P] = ud, Wd[Q] = vd, p.prototype = o.prototype;
        var Xd = o.support = {};
        Xd.funcDecomp=!jb(c.WinRTError) && E.test(m), Xd.funcNames = "string" == typeof qd.name, o.templateSettings = {
            escape: /<%-([\s\S]+?)%>/g,
            evaluate: /<%([\s\S]+?)%>/g,
            interpolate: B,
            variable: "",
            imports: {
                _: o
            }
        }, Nd || (X = function() {
            function a() {}
            return function(b) {
                if (Eb(b)) {
                    a.prototype = b;
                    var d = new a;
                    a.prototype = null
                }
                return d || c.Object()
            }
        }());
        var Yd = Md ? function(a, b) {
            T.value = b, Md(a, "__bindData__", T)
        }
        : ad, Zd = Od || function(a) {
            return a && "object" == typeof a && "number" == typeof a.length && Ad.call(a) == J ||!1
        }, $d = function(a) {
            var b, c = a, d = [];
            if (!c)
                return d;
            if (!U[typeof a])
                return d;
            for (b in c)
                Hd.call(c, b) && d.push(b);
            return d
        }, _d = Rd ? function(a) {
            return Eb(a) ? Rd(a) : []
        }
        : $d, ae = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        }, be = wb(ae), ce = ud("(" + _d(be).join("|") + ")", "g"), de = ud("[" + _d(ae).join("") + "]", "g"), ee = function(a, b, c) {
            var d, e = a, f = e;
            if (!e)
                return f;
            var g = arguments, h = 0, i = "number" == typeof c ? 2: g.length;
            if (i > 3 && "function" == typeof g[i - 2])
                var j = Y(g[--i - 1], g[i--], 2);
            else 
                i > 2 && "function" == typeof g[i - 1] && (j = g[--i]);
            for (; ++h < i;)
                if (e = g[h], e && U[typeof e])
                    for (var k =- 1, l = U[typeof e] && _d(e), m = l ? l.length : 0; ++k < m;)
                        d = l[k], f[d] = j ? j(f[d], e[d]) : e[d];
            return f
        }, fe = function(a, b, c) {
            var d, e = a, f = e;
            if (!e)
                return f;
            for (var g = arguments, h = 0, i = "number" == typeof c ? 2 : g.length; ++h < i;)
                if (e = g[h], e && U[typeof e])
                    for (var j =- 1, k = U[typeof e] && _d(e), l = k ? k.length : 0; ++j < l;)
                        d = k[j], "undefined" == typeof f[d] && (f[d] = e[d]);
            return f
        }, ge = function(a, b, c) {
            var d, e = a, f = e;
            if (!e)
                return f;
            if (!U[typeof e])
                return f;
            b = b && "undefined" == typeof c ? b : Y(b, c, 3);
            for (d in e)
                if (b(e[d], d, a)===!1)
                    return f;
            return f
        }, he = function(a, b, c) {
            var d, e = a, f = e;
            if (!e)
                return f;
            if (!U[typeof e])
                return f;
            b = b && "undefined" == typeof c ? b : Y(b, c, 3);
            for (var g =- 1, h = U[typeof e] && _d(e), i = h ? h.length : 0; ++g < i;)
                if (d = h[g], b(e[d], d, a)===!1)
                    return f;
            return f
        }, ie = Gd ? function(a) {
            if (!a || Ad.call(a) != O)
                return !1;
            var b = a.valueOf, c = jb(b) && (c = Gd(b)) && Gd(c);
            return c ? a == c || Gd(a) == c : kb(a)
        }
        : kb, je = fb(function(a, b, c) {
            Hd.call(a, c) ? a[c]++ : a[c] = 1
        }), ke = fb(function(a, b, c) {
            (Hd.call(a, c) ? a[c] : a[c] = []).push(b)
        }), le = fb(function(a, b, c) {
            a[c] = b
        }), me = _b, ne = Vb, oe = jb(oe = pd.now) && oe || function() {
            return (new pd).getTime()
        }, pe = 8 == Ud(u + "08") ? Ud: function(a, b) {
            return Ud(Jb(a) ? a.replace(C, "") : a, b || 0)
        };
        return o.after = Hc, o.assign = ee, o.at = Sb, o.bind = Ic, o.bindAll = Jc, o.bindKey = Kc, o.chain = id, o.compact = lc, o.compose = Lc, o.constant = Wc, o.countBy = je, o.create = pb, o.createCallback = Xc, o.curry = Mc, o.debounce = Nc, o.defaults = fe, o.defer = Oc, o.delay = Pc, o.difference = mc, o.filter = Vb, o.flatten = qc, o.forEach = Yb, o.forEachRight = Zb, o.forIn = ge, o.forInRight = sb, o.forOwn = he, o.forOwnRight = tb, o.functions = ub, o.groupBy = ke, o.indexBy = le, o.initial = sc, o.intersection = tc, o.invert = wb, o.invoke = $b, o.keys = _d, o.map = _b, o.mapValues = Lb, o.max = ac, o.memoize = Qc, o.merge = Mb, o.min = bc, o.omit = Nb, o.once = Rc, o.pairs = Ob, o.partial = Sc, o.partialRight = Tc, o.pick = Pb, o.pluck = me, o.property = bd, o.pull = wc, o.range = xc, o.reject = ec, o.remove = yc, o.rest = zc, o.shuffle = gc, o.sortBy = jc, o.tap = jd, o.throttle = Uc, o.times = fd, o.toArray = kc, o.transform = Qb, o.union = Bc, o.uniq = Cc, o.values = Rb, o.where = ne, o.without = Dc, o.wrap = Vc, o.xor = Ec, o.zip = Fc, o.zipObject = Gc, o.collect = _b, o.drop = zc, o.each = Yb, o.eachRight = Zb, o.extend = ee, o.methods = ub, o.object = Gc, o.select = Vb, o.tail = zc, o.unique = Cc, o.unzip = Fc, $c(o), o.clone = nb, o.cloneDeep = ob, o.contains = Tb, o.escape = Yc, o.every = Ub, o.find = Wb, o.findIndex = nc, o.findKey = qb, o.findLast = Xb, o.findLastIndex = oc, o.findLastKey = rb, o.has = vb, o.identity = Zc, o.indexOf = rc, o.isArguments = mb, o.isArray = Zd, o.isBoolean = xb, o.isDate = yb, o.isElement = zb, o.isEmpty = Ab, o.isEqual = Bb, o.isFinite = Cb, o.isFunction = Db, o.isNaN = Fb, o.isNull = Gb, o.isNumber = Hb, o.isObject = Eb, o.isPlainObject = ie, o.isRegExp = Ib, o.isString = Jb, o.isUndefined = Kb, o.lastIndexOf = vc, o.mixin = $c, o.noConflict = _c, o.noop = ad, o.now = oe, o.parseInt = pe, o.random = cd, o.reduce = cc, o.reduceRight = dc, o.result = dd, o.runInContext = m, o.size = hc, o.some = ic, o.sortedIndex = Ac, o.template = ed, o.unescape = gd, o.uniqueId = hd, o.all = Ub, o.any = ic, o.detect = Wb, o.findWhere = Wb, o.foldl = cc, o.foldr = dc, o.include = Tb, o.inject = cc, $c(function() {
            var a = {};
            return he(o, function(b, c) {
                o.prototype[c] || (a[c] = b)
            }), a
        }(), !1), o.first = pc, o.last = uc, o.sample = fc, o.take = pc, o.head = pc, he(o, function(a, b) {
            var c = "sample" !== b;
            o.prototype[b] || (o.prototype[b] = function(b, d) {
                var e = this.__chain__, f = a(this.__wrapped__, b, d);
                return e || null != b && (!d || c && "function" == typeof b) ? new p(f, e) : f
            })
        }), o.VERSION = "2.4.1", o.prototype.chain = kd, o.prototype.toString = ld, o.prototype.value = md, o.prototype.valueOf = md, Yb(["join", "pop", "shift"], function(a) {
            var b = xd[a];
            o.prototype[a] = function() {
                var a = this.__chain__, c = b.apply(this.__wrapped__, arguments);
                return a ? new p(c, a) : c
            }
        }), Yb(["push", "reverse", "sort", "unshift"], function(a) {
            var b = xd[a];
            o.prototype[a] = function() {
                return b.apply(this.__wrapped__, arguments), this
            }
        }), Yb(["concat", "slice", "splice"], function(a) {
            var b = xd[a];
            o.prototype[a] = function() {
                return new p(b.apply(this.__wrapped__, arguments), this.__chain__)
            }
        }), o
    }
    var n, o = [], p = [], q = 0, r =+ new Date + "", s = 75, t = 40, u = " 	\f \n\r\u2028\u2029 ᠎             　", v = /\b__p \+= '';/g, w = /\b(__p \+=) '' \+/g, x = /(__e\(.*?\)|\b__t\)) \+\n'';/g, y = /\$\{([^\\}]*(?:\\.[^\\}]*)*)\}/g, z = /\w*$/, A = /^\s*function[ \n\r\t]+\w/, B = /<%=([\s\S]+?)%>/g, C = RegExp("^[" + u + "]*0+(?=.$)"), D = /($^)/, E = /\bthis\b/, F = /['\n\r\t\u2028\u2029\\]/g, G = ["Array", "Boolean", "Date", "Function", "Math", "Number", "Object", "RegExp", "String", "_", "attachEvent", "clearTimeout", "isFinite", "isNaN", "parseInt", "setTimeout"], H = 0, I = "[object Arguments]", J = "[object Array]", K = "[object Boolean]", L = "[object Date]", M = "[object Function]", N = "[object Number]", O = "[object Object]", P = "[object RegExp]", Q = "[object String]", R = {};
    R[M]=!1, R[I] = R[J] = R[K] = R[L] = R[N] = R[O] = R[P] = R[Q]=!0;
    var S = {
        leading: !1,
        maxWait: 0,
        trailing: !1
    }, T = {
        configurable: !1,
        enumerable: !1,
        value: null,
        writable: !1
    }, U = {
        "boolean": !1,
        "function": !0,
        object: !0,
        number: !1,
        string: !1,
        undefined: !1
    }, V = {
        "\\": "\\",
        "'": "'",
        "\n": "n",
        "\r": "r",
        "	": "t",
        "\u2028": "u2028",
        "\u2029": "u2029"
    }, W = U[typeof window] && window || this, X = U[typeof exports] && exports&&!exports.nodeType && exports, Y = U[typeof module] && module&&!module.nodeType && module, Z = Y && Y.exports === X && X, $ = U[typeof global] && global;
    $ && ($.global === $ || $.window === $) && (W = $);
    var _ = m();
    "function" == typeof define && "object" == typeof define.amd && define.amd ? (W._ = _, define("underscore", [], function() {
        return _
    })) : X && Y ? Z ? (Y.exports = _)._ = _ : X._ = _ : W._ = _
}).call(this), function(a, b) {
    function c(a) {
        var b = a.length, c = kb.type(a);
        return kb.isWindow(a)?!1 : 1 === a.nodeType && b?!0 : "array" === c || "function" !== c && (0 === b || "number" == typeof b && b > 0 && b - 1 in a)
    }
    function d(a) {
        var b = zb[a] = {};
        return kb.each(a.match(mb) || [], function(a, c) {
            b[c]=!0
        }), b
    }
    function e(a, c, d, e) {
        if (kb.acceptData(a)) {
            var f, g, h = kb.expando, i = a.nodeType, j = i ? kb.cache: a, k = i ? a[h]: a[h] && h;
            if (k && j[k] && (e || j[k].data) || d !== b || "string" != typeof c)
                return k || (k = i ? a[h] = bb.pop() || kb.guid++ : h), j[k] || (j[k] = i ? {} : {
                    toJSON: kb.noop
                }), ("object" == typeof c || "function" == typeof c) && (e ? j[k] = kb.extend(j[k], c) : j[k].data = kb.extend(j[k].data, c)), g = j[k], e || (g.data || (g.data = {}), g = g.data), d !== b && (g[kb.camelCase(c)] = d), "string" == typeof c ? (f = g[c], null == f && (f = g[kb.camelCase(c)])) : f = g, f
        }
    }
    function f(a, b, c) {
        if (kb.acceptData(a)) {
            var d, e, f = a.nodeType, g = f ? kb.cache: a, i = f ? a[kb.expando]: kb.expando;
            if (g[i]) {
                if (b && (d = c ? g[i] : g[i].data)) {
                    kb.isArray(b) ? b = b.concat(kb.map(b, kb.camelCase)) : b in d ? b = [b] : (b = kb.camelCase(b), b = b in d ? [b] : b.split(" ")), e = b.length;
                    for (; e--;)
                        delete d[b[e]];
                    if (c?!h(d) : !kb.isEmptyObject(d)
                        )return 
                }(c || (delete g[i].data, h(g[i]))) && (f ? kb.cleanData([a], !0) : kb.support.deleteExpando || g != g.window ? delete g[i] : g[i] = null)
            }
        }
    }
    function g(a, c, d) {
        if (d === b && 1 === a.nodeType) {
            var e = "data-" + c.replace(Bb, "-$1").toLowerCase();
            if (d = a.getAttribute(e), "string" == typeof d) {
                try {
                    d = "true" === d?!0 : "false" === d?!1 : "null" === d ? null : + d + "" === d?+d : Ab.test(d) ? kb.parseJSON(d) : d
                } catch (f) {}
                kb.data(a, c, d)
            } else 
                d = b
        }
        return d
    }
    function h(a) {
        var b;
        for (b in a)
            if (("data" !== b ||!kb.isEmptyObject(a[b])) && "toJSON" !== b)
                return !1;
        return !0
    }
    function i() {
        return !0
    }
    function j() {
        return !1
    }
    function k() {
        try {
            return Y.activeElement
        } catch (a) {}
    }
    function l(a, b) {
        do 
            a = a[b];
        while (a && 1 !== a.nodeType);
        return a
    }
    function m(a, b, c) {
        if (kb.isFunction(b))
            return kb.grep(a, function(a, d) {
                return !!b.call(a, d, a) !== c
            });
        if (b.nodeType)
            return kb.grep(a, function(a) {
                return a === b !== c
            });
        if ("string" == typeof b) {
            if (Qb.test(b))
                return kb.filter(b, a, c);
            b = kb.filter(b, a)
        }
        return kb.grep(a, function(a) {
            return kb.inArray(a, b) >= 0 !== c
        })
    }
    function n(a) {
        var b = Ub.split("|"), c = a.createDocumentFragment();
        if (c.createElement)
            for (; b.length;)
                c.createElement(b.pop());
        return c
    }
    function o(a, b) {
        return kb.nodeName(a, "table") && kb.nodeName(1 === b.nodeType ? b : b.firstChild, "tr") ? a.getElementsByTagName("tbody")[0] || a.appendChild(a.ownerDocument.createElement("tbody")) : a
    }
    function p(a) {
        return a.type = (null !== kb.find.attr(a, "type")) + "/" + a.type, a
    }
    function q(a) {
        var b = ec.exec(a.type);
        return b ? a.type = b[1] : a.removeAttribute("type"), a
    }
    function r(a, b) {
        for (var c, d = 0; null != (c = a[d]); d++)
            kb._data(c, "globalEval", !b || kb._data(b[d], "globalEval"))
    }
    function s(a, b) {
        if (1 === b.nodeType && kb.hasData(a)) {
            var c, d, e, f = kb._data(a), g = kb._data(b, f), h = f.events;
            if (h) {
                delete g.handle, g.events = {};
                for (c in h)
                    for (d = 0, e = h[c].length; e > d; d++)
                        kb.event.add(b, c, h[c][d])
                    }
            g.data && (g.data = kb.extend({}, g.data))
        }
    }
    function t(a, b) {
        var c, d, e;
        if (1 === b.nodeType) {
            if (c = b.nodeName.toLowerCase(), !kb.support.noCloneEvent && b[kb.expando]) {
                e = kb._data(b);
                for (d in e.events)
                    kb.removeEvent(b, d, e.handle);
                b.removeAttribute(kb.expando)
            }
            "script" === c && b.text !== a.text ? (p(b).text = a.text, q(b)) : "object" === c ? (b.parentNode && (b.outerHTML = a.outerHTML), kb.support.html5Clone && a.innerHTML&&!kb.trim(b.innerHTML) && (b.innerHTML = a.innerHTML)) : "input" === c && bc.test(a.type) ? (b.defaultChecked = b.checked = a.checked, b.value !== a.value && (b.value = a.value)) : "option" === c ? b.defaultSelected = b.selected = a.defaultSelected : ("input" === c || "textarea" === c) && (b.defaultValue = a.defaultValue)
        }
    }
    function u(a, c) {
        var d, e, f = 0, g = typeof a.getElementsByTagName !== W ? a.getElementsByTagName(c || "*"): typeof a.querySelectorAll !== W ? a.querySelectorAll(c || "*"): b;
        if (!g)
            for (g = [], d = a.childNodes || a; null != (e = d[f]); f++)
                !c || kb.nodeName(e, c) ? g.push(e) : kb.merge(g, u(e, c));
        return c === b || c && kb.nodeName(a, c) ? kb.merge([a], g) : g
    }
    function v(a) {
        bc.test(a.type) && (a.defaultChecked = a.checked)
    }
    function w(a, b) {
        if (b in a)
            return b;
        for (var c = b.charAt(0).toUpperCase() + b.slice(1), d = b, e = yc.length; e--;)
            if (b = yc[e] + c, b in a)
                return b;
        return d
    }
    function x(a, b) {
        return a = b || a, "none" === kb.css(a, "display") ||!kb.contains(a.ownerDocument, a)
    }
    function y(a, b) {
        for (var c, d, e, f = [], g = 0, h = a.length; h > g; g++)
            d = a[g], d.style && (f[g] = kb._data(d, "olddisplay"), c = d.style.display, b ? (!f[g] && "none" === c && (d.style.display = ""), "" === d.style.display && x(d) && (f[g] = kb._data(d, "olddisplay", C(d.nodeName)))) : f[g] || (e = x(d), (c && "none" !== c ||!e) && kb._data(d, "olddisplay", e ? c : kb.css(d, "display"))));
        for (g = 0; h > g; g++)
            d = a[g], d.style && (b && "none" !== d.style.display && "" !== d.style.display || (d.style.display = b ? f[g] || "" : "none"));
        return a
    }
    function z(a, b, c) {
        var d = rc.exec(b);
        return d ? Math.max(0, d[1] - (c || 0)) + (d[2] || "px") : b
    }
    function A(a, b, c, d, e) {
        for (var f = c === (d ? "border" : "content") ? 4 : "width" === b ? 1 : 0, g = 0; 4 > f; f += 2)
            "margin" === c && (g += kb.css(a, c + xc[f], !0, e)), d ? ("content" === c && (g -= kb.css(a, "padding" + xc[f], !0, e)), "margin" !== c && (g -= kb.css(a, "border" + xc[f] + "Width", !0, e))) : (g += kb.css(a, "padding" + xc[f], !0, e), "padding" !== c && (g += kb.css(a, "border" + xc[f] + "Width", !0, e)));
        return g
    }
    function B(a, b, c) {
        var d=!0, e = "width" === b ? a.offsetWidth : a.offsetHeight, f = kc(a), g = kb.support.boxSizing && "border-box" === kb.css(a, "boxSizing", !1, f);
        if (0 >= e || null == e) {
            if (e = lc(a, b, f), (0 > e || null == e) && (e = a.style[b]), sc.test(e))
                return e;
            d = g && (kb.support.boxSizingReliable || e === a.style[b]), e = parseFloat(e) || 0
        }
        return e + A(a, b, c || (g ? "border" : "content"), d, f) + "px"
    }
    function C(a) {
        var b = Y, c = uc[a];
        return c || (c = D(a, b), "none" !== c && c || (jc = (jc || kb("<iframe frameborder='0' width='0' height='0'/>").css("cssText", "display:block !important")).appendTo(b.documentElement), b = (jc[0].contentWindow || jc[0].contentDocument).document, b.write("<!doctype html><html><body>"), b.close(), c = D(a, b), jc.detach()), uc[a] = c), c
    }
    function D(a, b) {
        var c = kb(b.createElement(a)).appendTo(b.body), d = kb.css(c[0], "display");
        return c.remove(), d
    }
    function E(a, b, c, d) {
        var e;
        if (kb.isArray(b))
            kb.each(b, function(b, e) {
                c || Ac.test(a) ? d(a, e) : E(a + "[" + ("object" == typeof e ? b : "") + "]", e, c, d)
            });
        else if (c || "object" !== kb.type(b))
            d(a, b);
        else 
            for (e in b)
                E(a + "[" + e + "]", b[e], c, d)
    }
    function F(a) {
        return function(b, c) {
            "string" != typeof b && (c = b, b = "*");
            var d, e = 0, f = b.toLowerCase().match(mb) || [];
            if (kb.isFunction(c))
                for (; d = f[e++];)
                    "+" === d[0] ? (d = d.slice(1) || "*", (a[d] = a[d] || []).unshift(c)) : (a[d] = a[d] || []).push(c)
        }
    }
    function G(a, b, c, d) {
        function e(h) {
            var i;
            return f[h]=!0, kb.each(a[h] || [], function(a, h) {
                var j = h(b, c, d);
                return "string" != typeof j || g || f[j] ? g?!(i = j) : void 0 : (b.dataTypes.unshift(j), e(j), !1)
            }), i
        }
        var f = {}, g = a === Rc;
        return e(b.dataTypes[0]) ||!f["*"] && e("*")
    }
    function H(a, c) {
        var d, e, f = kb.ajaxSettings.flatOptions || {};
        for (e in c)
            c[e] !== b && ((f[e] ? a : d || (d = {}))[e] = c[e]);
        return d && kb.extend(!0, a, d), a
    }
    function I(a, c, d) {
        for (var e, f, g, h, i = a.contents, j = a.dataTypes; "*" === j[0];)
            j.shift(), f === b && (f = a.mimeType || c.getResponseHeader("Content-Type"));
        if (f)
            for (h in i)
                if (i[h] && i[h].test(f)) {
                    j.unshift(h);
                    break
                }
        if (j[0]in d)
            g = j[0];
        else {
            for (h in d) {
                if (!j[0] || a.converters[h + " " + j[0]]) {
                    g = h;
                    break
                }
                e || (e = h)
            }
            g = g || e
        }
        return g ? (g !== j[0] && j.unshift(g), d[g]) : void 0
    }
    function J(a, b, c, d) {
        var e, f, g, h, i, j = {}, k = a.dataTypes.slice();
        if (k[1])
            for (g in a.converters)
                j[g.toLowerCase()] = a.converters[g];
        for (f = k.shift(); f;)
            if (a.responseFields[f] && (c[a.responseFields[f]] = b), !i && d && a.dataFilter && (b = a.dataFilter(b, a.dataType)), i = f, f = k.shift(), f)
                if ("*" === f)
                    f = i;
                else if ("*" !== i && i !== f) {
                    if (g = j[i + " " + f] || j["* " + f], !g)
                        for (e in j)
                            if (h = e.split(" "), h[1] === f && (g = j[i + " " + h[0]] || j["* " + h[0]])) {
                                g===!0 ? g = j[e] : j[e]!==!0 && (f = h[0], k.unshift(h[1]));
                                break
                            }
                            if (g!==!0)
                                if (g && a["throws"])
                                    b = g(b);
                                else 
                                    try {
                                        b = g(b)
                                    } catch (l) {
                                        return {
                                            state: "parsererror",
                                            error: g ? l: "No conversion from " + i + " to " + f
                                        }
                                    }
                                }
        return {
            state: "success",
            data: b
        }
    }
    function K() {
        try {
            return new a.XMLHttpRequest
        } catch (b) {}
    }
    function L() {
        try {
            return new a.ActiveXObject("Microsoft.XMLHTTP")
        } catch (b) {}
    }
    function M() {
        return setTimeout(function() {
            $c = b
        }), $c = kb.now()
    }
    function N(a, b, c) {
        for (var d, e = (ed[b] || []).concat(ed["*"]), f = 0, g = e.length; g > f; f++)
            if (d = e[f].call(c, b, a))
                return d
    }
    function O(a, b, c) {
        var d, e, f = 0, g = dd.length, h = kb.Deferred().always(function() {
            delete i.elem
        }), i = function() {
            if (e)
                return !1;
            for (var b = $c || M(), c = Math.max(0, j.startTime + j.duration - b), d = c / j.duration || 0, f = 1 - d, g = 0, i = j.tweens.length; i > g; g++)
                j.tweens[g].run(f);
            return h.notifyWith(a, [j, f, c]), 1 > f && i ? c : (h.resolveWith(a, [j]), !1)
        }, j = h.promise({
            elem: a,
            props: kb.extend({}, b),
            opts: kb.extend(!0, {
                specialEasing: {}
            }, c),
            originalProperties: b,
            originalOptions: c,
            startTime: $c || M(),
            duration: c.duration,
            tweens: [],
            createTween: function(b, c) {
                var d = kb.Tween(a, j.opts, b, c, j.opts.specialEasing[b] || j.opts.easing);
                return j.tweens.push(d), d
            },
            stop: function(b) {
                var c = 0, d = b ? j.tweens.length: 0;
                if (e)
                    return this;
                for (e=!0; d > c; c++)
                    j.tweens[c].run(1);
                return b ? h.resolveWith(a, [j, b]) : h.rejectWith(a, [j, b]), this
            }
        }), k = j.props;
        for (P(k, j.opts.specialEasing);
        g > f;
        f++)if (d = dd[f].call(j, a, k, j.opts))
            return d;
        return kb.map(k, N, j), kb.isFunction(j.opts.start) && j.opts.start.call(a, j), kb.fx.timer(kb.extend(i, {
            elem: a,
            anim: j,
            queue: j.opts.queue
        })), j.progress(j.opts.progress).done(j.opts.done, j.opts.complete).fail(j.opts.fail).always(j.opts.always)
    }
    function P(a, b) {
        var c, d, e, f, g;
        for (c in a)
            if (d = kb.camelCase(c), e = b[d], f = a[c], kb.isArray(f) && (e = f[1], f = a[c] = f[0]), c !== d && (a[d] = f, delete a[c]), g = kb.cssHooks[d], g && "expand"in g) {
                f = g.expand(f), delete a[d];
                for (c in f)
                    c in a || (a[c] = f[c], b[c] = e)
            } else 
                b[d] = e
    }
    function Q(a, b, c) {
        var d, e, f, g, h, i, j = this, k = {}, l = a.style, m = a.nodeType && x(a), n = kb._data(a, "fxshow");
        c.queue || (h = kb._queueHooks(a, "fx"), null == h.unqueued && (h.unqueued = 0, i = h.empty.fire, h.empty.fire = function() {
            h.unqueued || i()
        }), h.unqueued++, j.always(function() {
            j.always(function() {
                h.unqueued--, kb.queue(a, "fx").length || h.empty.fire()
            })
        })), 1 === a.nodeType && ("height"in b || "width"in b) && (c.overflow = [l.overflow, l.overflowX, l.overflowY], "inline" === kb.css(a, "display") && "none" === kb.css(a, "float") && (kb.support.inlineBlockNeedsLayout && "inline" !== C(a.nodeName) ? l.zoom = 1 : l.display = "inline-block")), c.overflow && (l.overflow = "hidden", kb.support.shrinkWrapBlocks || j.always(function() {
            l.overflow = c.overflow[0], l.overflowX = c.overflow[1], l.overflowY = c.overflow[2]
        }));
        for (d in b)
            if (e = b[d], ad.exec(e)) {
                if (delete b[d], f = f || "toggle" === e, e === (m ? "hide" : "show"))
                    continue;
                    k[d] = n && n[d] || kb.style(a, d)
            }
        if (!kb.isEmptyObject(k)) {
            n ? "hidden"in n && (m = n.hidden) : n = kb._data(a, "fxshow", {}), f && (n.hidden=!m), m ? kb(a).show() : j.done(function() {
                kb(a).hide()
            }), j.done(function() {
                var b;
                kb._removeData(a, "fxshow");
                for (b in k)
                    kb.style(a, b, k[b])
            });
            for (d in k)
                g = N(m ? n[d] : 0, d, j), d in n || (n[d] = g.start, m && (g.end = g.start, g.start = "width" === d || "height" === d ? 1 : 0))
                }
    }
    function R(a, b, c, d, e) {
        return new R.prototype.init(a, b, c, d, e)
    }
    function S(a, b) {
        var c, d = {
            height: a
        }, e = 0;
        for (b = b ? 1 : 0; 4 > e; e += 2 - b)
            c = xc[e], d["margin" + c] = d["padding" + c] = a;
        return b && (d.opacity = d.width = a), d
    }
    function T(a) {
        return kb.isWindow(a) ? a : 9 === a.nodeType ? a.defaultView || a.parentWindow : !1
    }
    var U, V, W = typeof b, X = a.location, Y = a.document, Z = Y.documentElement, $ = a.jQuery, _ = a.$, ab = {}, bb = [], cb = "1.10.2", db = bb.concat, eb = bb.push, fb = bb.slice, gb = bb.indexOf, hb = ab.toString, ib = ab.hasOwnProperty, jb = cb.trim, kb = function(a, b) {
        return new kb.fn.init(a, b, V)
    }, lb = /[+-]?(?:\d*\.|)\d+(?:[eE][+-]?\d+|)/.source, mb = /\S+/g, nb = /^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g, ob = /^(?:\s*(<[\w\W]+>)[^>]*|#([\w-]*))$/, pb = /^<(\w+)\s*\/?>(?:<\/\1>|)$/, qb = /^[\],:{}\s]*$/, rb = /(?:^|:|,)(?:\s*\[)+/g, sb = /\\(?:["\\\/bfnrt]|u[\da-fA-F]{4})/g, tb = /"[^"\\\r\n]*"|true|false|null|-?(?:\d+\.|)\d+(?:[eE][+-]?\d+|)/g, ub = /^-ms-/, vb = /-([\da-z])/gi, wb = function(a, b) {
        return b.toUpperCase()
    }, xb = function(a) {
        (Y.addEventListener || "load" === a.type || "complete" === Y.readyState) && (yb(), kb.ready())
    }, yb = function() {
        Y.addEventListener ? (Y.removeEventListener("DOMContentLoaded", xb, !1), a.removeEventListener("load", xb, !1)) : (Y.detachEvent("onreadystatechange", xb), a.detachEvent("onload", xb))
    };
    kb.fn = kb.prototype = {
        jquery: cb,
        constructor: kb,
        init: function(a, c, d) {
            var e, f;
            if (!a)
                return this;
            if ("string" == typeof a) {
                if (e = "<" === a.charAt(0) && ">" === a.charAt(a.length - 1) && a.length >= 3 ? [null, a, null] : ob.exec(a), e && (e[1] ||!c)) {
                    if (e[1]) {
                        if (c = c instanceof kb ? c[0] : c, kb.merge(this, kb.parseHTML(e[1], c && c.nodeType ? c.ownerDocument || c : Y, !0)), pb.test(e[1]) && kb.isPlainObject(c))
                            for (e in c)
                                kb.isFunction(this[e]) ? this[e](c[e]) : this.attr(e, c[e]);
                        return this
                    }
                    if (f = Y.getElementById(e[2]), f && f.parentNode) {
                        if (f.id !== e[2])
                            return d.find(a);
                        this.length = 1, this[0] = f
                    }
                    return this.context = Y, this.selector = a, this
                }
                return !c || c.jquery ? (c || d).find(a) : this.constructor(c).find(a)
            }
            return a.nodeType ? (this.context = this[0] = a, this.length = 1, this) : kb.isFunction(a) ? d.ready(a) : (a.selector !== b && (this.selector = a.selector, this.context = a.context), kb.makeArray(a, this))
        },
        selector: "",
        length: 0,
        toArray: function() {
            return fb.call(this)
        },
        get: function(a) {
            return null == a ? this.toArray() : 0 > a ? this[this.length + a] : this[a]
        },
        pushStack: function(a) {
            var b = kb.merge(this.constructor(), a);
            return b.prevObject = this, b.context = this.context, b
        },
        each: function(a, b) {
            return kb.each(this, a, b)
        },
        ready: function(a) {
            return kb.ready.promise().done(a), this
        },
        slice: function() {
            return this.pushStack(fb.apply(this, arguments))
        },
        first: function() {
            return this.eq(0)
        },
        last: function() {
            return this.eq( - 1)
        },
        eq: function(a) {
            var b = this.length, c =+ a + (0 > a ? b : 0);
            return this.pushStack(c >= 0 && b > c ? [this[c]] : [])
        },
        map: function(a) {
            return this.pushStack(kb.map(this, function(b, c) {
                return a.call(b, c, b)
            }))
        },
        end: function() {
            return this.prevObject || this.constructor(null)
        },
        push: eb,
        sort: [].sort,
        splice: [].splice
    }, kb.fn.init.prototype = kb.fn, kb.extend = kb.fn.extend = function() {
        var a, c, d, e, f, g, h = arguments[0] || {}, i = 1, j = arguments.length, k=!1;
        for ("boolean" == typeof h && (k = h, h = arguments[1] || {}, i = 2), "object" != typeof h&&!kb.isFunction(h) && (h = {}), j === i && (h = this, --i); j > i; i++)
            if (null != (f = arguments[i]))
                for (e in f)
                    a = h[e], d = f[e], h !== d && (k && d && (kb.isPlainObject(d) || (c = kb.isArray(d))) ? (c ? (c=!1, g = a && kb.isArray(a) ? a : []) : g = a && kb.isPlainObject(a) ? a : {}, h[e] = kb.extend(k, g, d)) : d !== b && (h[e] = d));
        return h
    }, kb.extend({
        expando: "jQuery" + (cb + Math.random()).replace(/\D/g, ""),
        noConflict: function(b) {
            return a.$ === kb && (a.$ = _), b && a.jQuery === kb && (a.jQuery = $), kb
        },
        isReady: !1,
        readyWait: 1,
        holdReady: function(a) {
            a ? kb.readyWait++ : kb.ready(!0)
        },
        ready: function(a) {
            if (a===!0?!--kb.readyWait : !kb.isReady) {
                if (!Y.body)
                    return setTimeout(kb.ready);
                kb.isReady=!0, a!==!0&&--kb.readyWait > 0 || (U.resolveWith(Y, [kb]), kb.fn.trigger && kb(Y).trigger("ready").off("ready"))
            }
        },
        isFunction: function(a) {
            return "function" === kb.type(a)
        },
        isArray: Array.isArray || function(a) {
            return "array" === kb.type(a)
        },
        isWindow: function(a) {
            return null != a && a == a.window
        },
        isNumeric: function(a) {
            return !isNaN(parseFloat(a)) && isFinite(a)
        },
        type: function(a) {
            return null == a ? String(a) : "object" == typeof a || "function" == typeof a ? ab[hb.call(a)] || "object" : typeof a
        },
        isPlainObject: function(a) {
            var c;
            if (!a || "object" !== kb.type(a) || a.nodeType || kb.isWindow(a))
                return !1;
            try {
                if (a.constructor&&!ib.call(a, "constructor")&&!ib.call(a.constructor.prototype, "isPrototypeOf"))
                    return !1
            } catch (d) {
                return !1
            }
            if (kb.support.ownLast)
                for (c in a)
                    return ib.call(a, c);
            for (c in a);
            return c === b || ib.call(a, c)
        },
        isEmptyObject: function(a) {
            var b;
            for (b in a)
                return !1;
            return !0
        },
        error: function(a) {
            throw new Error(a)
        },
        parseHTML: function(a, b, c) {
            if (!a || "string" != typeof a)
                return null;
            "boolean" == typeof b && (c = b, b=!1), b = b || Y;
            var d = pb.exec(a), e=!c && [];
            return d ? [b.createElement(d[1])] : (d = kb.buildFragment([a], b, e), e && kb(e).remove(), kb.merge([], d.childNodes))
        },
        parseJSON: function(b) {
            return a.JSON && a.JSON.parse ? a.JSON.parse(b) : null === b ? b : "string" == typeof b && (b = kb.trim(b), b && qb.test(b.replace(sb, "@").replace(tb, "]").replace(rb, ""))) ? new Function("return " + b)() : void kb.error("Invalid JSON: " + b)
        },
        parseXML: function(c) {
            var d, e;
            if (!c || "string" != typeof c)
                return null;
            try {
                a.DOMParser ? (e = new DOMParser, d = e.parseFromString(c, "text/xml")) : (d = new ActiveXObject("Microsoft.XMLDOM"), d.async = "false", d.loadXML(c))
            } catch (f) {
                d = b
            }
            return (!d ||!d.documentElement || d.getElementsByTagName("parsererror").length) && kb.error("Invalid XML: " + c), d
        },
        noop: function() {},
        globalEval: function(b) {
            b && kb.trim(b) && (a.execScript || function(b) {
                a.eval.call(a, b)
            })(b)
        },
        camelCase: function(a) {
            return a.replace(ub, "ms-").replace(vb, wb)
        },
        nodeName: function(a, b) {
            return a.nodeName && a.nodeName.toLowerCase() === b.toLowerCase()
        },
        each: function(a, b, d) {
            var e, f = 0, g = a.length, h = c(a);
            if (d) {
                if (h)
                    for (; g > f && (e = b.apply(a[f], d), e!==!1); f++);
                else 
                    for (f in a)
                        if (e = b.apply(a[f], d), e===!1)
                            break
            } else if (h)
                for (; g > f && (e = b.call(a[f], f, a[f]), e!==!1); f++);
            else 
                for (f in a)
                    if (e = b.call(a[f], f, a[f]), e===!1)
                        break;
            return a
        },
        trim: jb&&!jb.call(" ") ? function(a) {
            return null == a ? "" : jb.call(a)
        }
        : function(a) {
            return null == a ? "" : (a + "").replace(nb, "")
        },
        makeArray: function(a, b) {
            var d = b || [];
            return null != a && (c(Object(a)) ? kb.merge(d, "string" == typeof a ? [a] : a) : eb.call(d, a)), d
        },
        inArray: function(a, b, c) {
            var d;
            if (b) {
                if (gb)
                    return gb.call(b, a, c);
                for (d = b.length, c = c ? 0 > c ? Math.max(0, d + c) : c : 0; d > c; c++)
                    if (c in b && b[c] === a)
                        return c
            }
            return - 1
        },
        merge: function(a, c) {
            var d = c.length, e = a.length, f = 0;
            if ("number" == typeof d)
                for (; d > f; f++)
                    a[e++] = c[f];
            else 
                for (; c[f] !== b;)
                    a[e++] = c[f++];
            return a.length = e, a
        },
        grep: function(a, b, c) {
            var d, e = [], f = 0, g = a.length;
            for (c=!!c; g > f; f++)
                d=!!b(a[f], f), c !== d && e.push(a[f]);
            return e
        },
        map: function(a, b, d) {
            var e, f = 0, g = a.length, h = c(a), i = [];
            if (h)
                for (; g > f; f++)
                    e = b(a[f], f, d), null != e && (i[i.length] = e);
            else 
                for (f in a)
                    e = b(a[f], f, d), null != e && (i[i.length] = e);
            return db.apply([], i)
        },
        guid: 1,
        proxy: function(a, c) {
            var d, e, f;
            return "string" == typeof c && (f = a[c], c = a, a = f), kb.isFunction(a) ? (d = fb.call(arguments, 2), e = function() {
                return a.apply(c || this, d.concat(fb.call(arguments)))
            }, e.guid = a.guid = a.guid || kb.guid++, e) : b
        },
        access: function(a, c, d, e, f, g, h) {
            var i = 0, j = a.length, k = null == d;
            if ("object" === kb.type(d)) {
                f=!0;
                for (i in d)
                    kb.access(a, c, i, d[i], !0, g, h)
            } else if (e !== b && (f=!0, kb.isFunction(e) || (h=!0), k && (h ? (c.call(a, e), c = null) : (k = c, c = function(a, b, c) {
                return k.call(kb(a), c)
            })), c))
                for (; j > i; i++)
                    c(a[i], d, h ? e : e.call(a[i], i, c(a[i], d)));
            return f ? a : k ? c.call(a) : j ? c(a[0], d) : g
        },
        now: function() {
            return (new Date).getTime()
        },
        swap: function(a, b, c, d) {
            var e, f, g = {};
            for (f in b)
                g[f] = a.style[f], a.style[f] = b[f];
            e = c.apply(a, d || []);
            for (f in b)
                a.style[f] = g[f];
            return e
        }
    }), kb.ready.promise = function(b) {
        if (!U)
            if (U = kb.Deferred(), "complete" === Y.readyState)
                setTimeout(kb.ready);
            else if (Y.addEventListener)
                Y.addEventListener("DOMContentLoaded", xb, !1), a.addEventListener("load", xb, !1);
            else {
                Y.attachEvent("onreadystatechange", xb), a.attachEvent("onload", xb);
                var c=!1;
                try {
                    c = null == a.frameElement && Y.documentElement
                } catch (d) {}
                c && c.doScroll && function e() {
                    if (!kb.isReady) {
                        try {
                            c.doScroll("left")
                        } catch (a) {
                            return setTimeout(e, 50)
                        }
                        yb(), kb.ready()
                    }
                }()
            }
        return U.promise(b)
    }, kb.each("Boolean Number String Function Array Date RegExp Object Error".split(" "), function(a, b) {
        ab["[object " + b + "]"] = b.toLowerCase()
    }), V = kb(Y), function(a, b) {
        function c(a, b, c, d) {
            var e, f, g, h, i, j, k, l, o, p;
            if ((b ? b.ownerDocument || b : O) !== G && F(b), b = b || G, c = c || [], !a || "string" != typeof a)
                return c;
            if (1 !== (h = b.nodeType) && 9 !== h)
                return [];
            if (I&&!d) {
                if (e = tb.exec(a))
                    if (g = e[1]) {
                        if (9 === h) {
                            if (f = b.getElementById(g), !f ||!f.parentNode)
                                return c;
                                if (f.id === g)
                                    return c.push(f), c
                        } else if (b.ownerDocument && (f = b.ownerDocument.getElementById(g)) && M(b, f) && f.id === g)
                            return c.push(f), c
                    } else {
                        if (e[2])
                            return ab.apply(c, b.getElementsByTagName(a)), c;
                            if ((g = e[3]) && x.getElementsByClassName && b.getElementsByClassName)
                                return ab.apply(c, b.getElementsByClassName(g)), c
                    }
                if (x.qsa && (!J ||!J.test(a))) {
                    if (l = k = N, o = b, p = 9 === h && a, 1 === h && "object" !== b.nodeName.toLowerCase()) {
                        for (j = m(a), (k = b.getAttribute("id")) ? l = k.replace(wb, "\\$&") : b.setAttribute("id", l), l = "[id='" + l + "'] ", i = j.length; i--;)
                            j[i] = l + n(j[i]);
                        o = nb.test(a) && b.parentNode || b, p = j.join(",")
                    }
                    if (p)
                        try {
                            return ab.apply(c, o.querySelectorAll(p)), c
                    } catch (q) {} finally {
                        k || b.removeAttribute("id")
                    }
                }
            }
            return v(a.replace(jb, "$1"), b, c, d)
        }
        function d() {
            function a(c, d) {
                return b.push(c += " ") > z.cacheLength && delete a[b.shift()], a[c] = d
            }
            var b = [];
            return a
        }
        function e(a) {
            return a[N]=!0, a
        }
        function f(a) {
            var b = G.createElement("div");
            try {
                return !!a(b)
            } catch (c) {
                return !1
            } finally {
                b.parentNode && b.parentNode.removeChild(b), b = null
            }
        }
        function g(a, b) {
            for (var c = a.split("|"), d = a.length; d--;)
                z.attrHandle[c[d]] = b
        }
        function h(a, b) {
            var c = b && a, d = c && 1 === a.nodeType && 1 === b.nodeType && (~b.sourceIndex || X) - (~a.sourceIndex || X);
            if (d)
                return d;
            if (c)
                for (; c = c.nextSibling;)
                    if (c === b)
                        return - 1;
            return a ? 1 : - 1
        }
        function i(a) {
            return function(b) {
                var c = b.nodeName.toLowerCase();
                return "input" === c && b.type === a
            }
        }
        function j(a) {
            return function(b) {
                var c = b.nodeName.toLowerCase();
                return ("input" === c || "button" === c) && b.type === a
            }
        }
        function k(a) {
            return e(function(b) {
                return b =+ b, e(function(c, d) {
                    for (var e, f = a([], c.length, b), g = f.length; g--;)
                        c[e = f[g]] && (c[e]=!(d[e] = c[e]))
                })
            })
        }
        function l() {}
        function m(a, b) {
            var d, e, f, g, h, i, j, k = S[a + " "];
            if (k)
                return b ? 0 : k.slice(0);
            for (h = a, i = [], j = z.preFilter; h;) {
                (!d || (e = lb.exec(h))) && (e && (h = h.slice(e[0].length) || h), i.push(f = [])), d=!1, (e = mb.exec(h)) && (d = e.shift(), f.push({
                    value: d,
                    type: e[0].replace(jb, " ")
                }), h = h.slice(d.length));
                for (g in z.filter)(e = rb[g].exec(h)
                    ) && (!j[g] || (e = j[g](e))) && (d = e.shift(), f.push({
                        value: d,
                        type: g,
                        matches: e
                    }), h = h.slice(d.length));
                if (!d)
                    break
            }
            return b ? h.length : h ? c.error(a) : S(a, i).slice(0)
        }
        function n(a) {
            for (var b = 0, c = a.length, d = ""; c > b; b++)
                d += a[b].value;
            return d
        }
        function o(a, b, c) {
            var d = b.dir, e = c && "parentNode" === d, f = Q++;
            return b.first ? function(b, c, f) {
                for (; b = b[d];)
                    if (1 === b.nodeType || e)
                        return a(b, c, f)
            } : function(b, c, g) {
                var h, i, j, k = P + " " + f;
                if (g) {
                    for (; b = b[d];)
                        if ((1 === b.nodeType || e) && a(b, c, g))
                            return !0
                } else 
                    for (; b = b[d];)
                        if (1 === b.nodeType || e)
                            if (j = b[N] || (b[N] = {}), (i = j[d]) && i[0] === k) {
                                if ((h = i[1])===!0 || h === y)
                                    return h===!0
                            } else if (i = j[d] = [k], i[1] = a(b, c, g) || y, i[1]===!0)
                                return !0
            }
        }
        function p(a) {
            return a.length > 1 ? function(b, c, d) {
                for (var e = a.length; e--;)
                    if (!a[e](b, c, d))
                        return !1;
                return !0
            } : a[0]
        }
        function q(a, b, c, d, e) {
            for (var f, g = [], h = 0, i = a.length, j = null != b; i > h; h++)(f = a[h]) 
                && (!c || c(f, d, e)) && (g.push(f), j && b.push(h));
            return g
        }
        function r(a, b, c, d, f, g) {
            return d&&!d[N] && (d = r(d)), f&&!f[N] && (f = r(f, g)), e(function(e, g, h, i) {
                var j, k, l, m = [], n = [], o = g.length, p = e || u(b || "*", h.nodeType ? [h] : h, []), r=!a ||!e && b ? p : q(p, m, a, h, i), s = c ? f || (e ? a : o || d) ? [] : g : r;
                if (c && c(r, s, h, i), d)
                    for (j = q(s, n), d(j, [], h, i), k = j.length; k--;)(l = j[k]) 
                        && (s[n[k]]=!(r[n[k]] = l));
                if (e) {
                    if (f || a) {
                        if (f) {
                            for (j = [], k = s.length; k--;)(l = s[k]) 
                                && j.push(r[k] = l);
                            f(null, s = [], j, i)
                        }
                        for (k = s.length; k--;)(l = s[k]) 
                            && (j = f ? cb.call(e, l) : m[k])>-1 && (e[j]=!(g[j] = l))
                        }
                } else 
                    s = q(s === g ? s.splice(o, s.length) : s), f ? f(null, g, s, i) : ab.apply(g, s)
            })
        }
        function s(a) {
            for (var b, c, d, e = a.length, f = z.relative[a[0].type], g = f || z.relative[" "], h = f ? 1 : 0, i = o(function(a) {
                return a === b
            }, g, !0), j = o(function(a) {
                return cb.call(b, a)>-1
            }, g, !0), k = [function(a, c, d) {
                return !f && (d || c !== D) || ((b = c).nodeType ? i(a, c, d) : j(a, c, d))
            }
            ]; e > h; h++)
                if (c = z.relative[a[h].type])
                    k = [o(p(k), c)];
                else {
                    if (c = z.filter[a[h].type].apply(null, a[h].matches), c[N]) {
                        for (d=++h; e > d&&!z.relative[a[d].type]; d++);
                        return r(h > 1 && p(k), h > 1 && n(a.slice(0, h - 1).concat({
                            value: " " === a[h - 2].type ? "*": ""
                        })).replace(jb, "$1"), c, d > h && s(a.slice(h, d)), e > d && s(a = a.slice(d)), e > d && n(a))
                    }
                    k.push(c)
                }
            return p(k)
        }
        function t(a, b) {
            var d = 0, f = b.length > 0, g = a.length > 0, h = function(e, h, i, j, k) {
                var l, m, n, o = [], p = 0, r = "0", s = e && [], t = null != k, u = D, v = e || g && z.find.TAG("*", k && h.parentNode || h), w = P += null == u ? 1: Math.random() || .1;
                for (t && (D = h !== G && h, y = d); null != (l = v[r]); r++) {
                    if (g && l) {
                        for (m = 0; n = a[m++];)
                            if (n(l, h, i)) {
                                j.push(l);
                                break
                            }
                        t && (P = w, y=++d)
                    }
                    f && ((l=!n && l) && p--, e && s.push(l))
                }
                if (p += r, f && r !== p) {
                    for (m = 0; n = b[m++];)
                        n(s, o, h, i);
                    if (e) {
                        if (p > 0)
                            for (; r--;)
                                !s[r]&&!o[r] && (o[r] = $.call(j));
                        o = q(o)
                    }
                    ab.apply(j, o), t&&!e && o.length > 0 && p + b.length > 1 && c.uniqueSort(j)
                }
                return t && (P = w, D = u), s
            };
            return f ? e(h) : h
        }
        function u(a, b, d) {
            for (var e = 0, f = b.length; f > e; e++)
                c(a, b[e], d);
            return d
        }
        function v(a, b, c, d) {
            var e, f, g, h, i, j = m(a);
            if (!d && 1 === j.length) {
                if (f = j[0] = j[0].slice(0), f.length > 2 && "ID" === (g = f[0]).type && x.getById && 9 === b.nodeType && I && z.relative[f[1].type]) {
                    if (b = (z.find.ID(g.matches[0].replace(xb, yb), b) || [])[0], !b)
                        return c;
                    a = a.slice(f.shift().value.length)
                }
                for (e = rb.needsContext.test(a) ? 0 : f.length; e--&&(g = f[e], !z.relative[h = g.type]);)
                    if ((i = z.find[h]) && (d = i(g.matches[0].replace(xb, yb), nb.test(f[0].type) && b.parentNode || b))) {
                        if (f.splice(e, 1), a = d.length && n(f), !a)
                            return ab.apply(c, d), c;
                            break
                    }
            }
            return C(a, j)(d, b, !I, c, nb.test(a)), c
        }
        var w, x, y, z, A, B, C, D, E, F, G, H, I, J, K, L, M, N = "sizzle" +- new Date, O = a.document, P = 0, Q = 0, R = d(), S = d(), T = d(), U=!1, V = function(a, b) {
            return a === b ? (U=!0, 0) : 0
        }, W = typeof b, X = 1<<31, Y = {}.hasOwnProperty, Z = [], $ = Z.pop, _ = Z.push, ab = Z.push, bb = Z.slice, cb = Z.indexOf || function(a) {
            for (var b = 0, c = this.length; c > b; b++)
                if (this[b] === a)
                    return b;
            return - 1
        }, db = "checked|selected|async|autofocus|autoplay|controls|defer|disabled|hidden|ismap|loop|multiple|open|readonly|required|scoped", eb = "[\\x20\\t\\r\\n\\f]", fb = "(?:\\\\.|[\\w-]|[^\\x00-\\xa0])+", gb = fb.replace("w", "w#"), hb = "\\[" + eb + "*(" + fb + ")" + eb + "*(?:([*^$|!~]?=)" + eb + "*(?:(['\"])((?:\\\\.|[^\\\\])*?)\\3|(" + gb + ")|)|)" + eb + "*\\]", ib = ":(" + fb + ")(?:\\(((['\"])((?:\\\\.|[^\\\\])*?)\\3|((?:\\\\.|[^\\\\()[\\]]|" + hb.replace(3, 8) + ")*)|.*)\\)|)", jb = new RegExp("^" + eb + "+|((?:^|[^\\\\])(?:\\\\.)*)" + eb + "+$", "g"), lb = new RegExp("^" + eb + "*," + eb + "*"), mb = new RegExp("^" + eb + "*([>+~]|" + eb + ")" + eb + "*"), nb = new RegExp(eb + "*[+~]"), ob = new RegExp("=" + eb + "*([^\\]'\"]*)" + eb + "*\\]", "g"), pb = new RegExp(ib), qb = new RegExp("^" + gb + "$"), rb = {
            ID: new RegExp("^#(" + fb + ")"),
            CLASS: new RegExp("^\\.(" + fb + ")"),
            TAG: new RegExp("^(" + fb.replace("w", "w*") + ")"),
            ATTR: new RegExp("^" + hb),
            PSEUDO: new RegExp("^" + ib),
            CHILD: new RegExp("^:(only|first|last|nth|nth-last)-(child|of-type)(?:\\(" + eb + "*(even|odd|(([+-]|)(\\d*)n|)" + eb + "*(?:([+-]|)" + eb + "*(\\d+)|))" + eb + "*\\)|)", "i"),
            bool: new RegExp("^(?:" + db + ")$", "i"),
            needsContext: new RegExp("^" + eb + "*[>+~]|:(even|odd|eq|gt|lt|nth|first|last)(?:\\(" + eb + "*((?:-\\d)?\\d*)" + eb + "*\\)|)(?=[^-]|$)", "i")
        }, sb = /^[^{]+\{\s*\[native \w/, tb = /^(?:#([\w-]+)|(\w+)|\.([\w-]+))$/, ub = /^(?:input|select|textarea|button)$/i, vb = /^h\d$/i, wb = /'|\\/g, xb = new RegExp("\\\\([\\da-f]{1,6}" + eb + "?|(" + eb + ")|.)", "ig"), yb = function(a, b, c) {
            var d = "0x" + b - 65536;
            return d !== d || c ? b : 0 > d ? String.fromCharCode(d + 65536) : String.fromCharCode(d>>10 | 55296, 1023 & d | 56320)
        };
        try {
            ab.apply(Z = bb.call(O.childNodes), O.childNodes), Z[O.childNodes.length].nodeType
        } catch (zb) {
            ab = {
                apply: Z.length ? function(a, b) {
                    _.apply(a, bb.call(b))
                }
                : function(a, b) {
                    for (var c = a.length, d = 0; a[c++] = b[d++];);
                    a.length = c - 1
                }
            }
        }
        B = c.isXML = function(a) {
            var b = a && (a.ownerDocument || a).documentElement;
            return b ? "HTML" !== b.nodeName : !1
        }, x = c.support = {}, F = c.setDocument = function(a) {
            var b = a ? a.ownerDocument || a: O, c = b.defaultView;
            return b !== G && 9 === b.nodeType && b.documentElement ? (G = b, H = b.documentElement, I=!B(b), c && c.attachEvent && c !== c.top && c.attachEvent("onbeforeunload", function() {
                F()
            }), x.attributes = f(function(a) {
                return a.className = "i", !a.getAttribute("className")
            }), x.getElementsByTagName = f(function(a) {
                return a.appendChild(b.createComment("")), !a.getElementsByTagName("*").length
            }), x.getElementsByClassName = f(function(a) {
                return a.innerHTML = "<div class='a'></div><div class='a i'></div>", a.firstChild.className = "i", 2 === a.getElementsByClassName("i").length
            }), x.getById = f(function(a) {
                return H.appendChild(a).id = N, !b.getElementsByName ||!b.getElementsByName(N).length
            }), x.getById ? (z.find.ID = function(a, b) {
                if (typeof b.getElementById !== W && I) {
                    var c = b.getElementById(a);
                    return c && c.parentNode ? [c] : []
                }
            }, z.filter.ID = function(a) {
                var b = a.replace(xb, yb);
                return function(a) {
                    return a.getAttribute("id") === b
                }
            }) : (delete z.find.ID, z.filter.ID = function(a) {
                var b = a.replace(xb, yb);
                return function(a) {
                    var c = typeof a.getAttributeNode !== W && a.getAttributeNode("id");
                    return c && c.value === b
                }
            }), z.find.TAG = x.getElementsByTagName ? function(a, b) {
                return typeof b.getElementsByTagName !== W ? b.getElementsByTagName(a) : void 0
            } : function(a, b) {
                var c, d = [], e = 0, f = b.getElementsByTagName(a);
                if ("*" === a) {
                    for (; c = f[e++];)
                        1 === c.nodeType && d.push(c);
                    return d
                }
                return f
            }, z.find.CLASS = x.getElementsByClassName && function(a, b) {
                return typeof b.getElementsByClassName !== W && I ? b.getElementsByClassName(a) : void 0
            }, K = [], J = [], (x.qsa = sb.test(b.querySelectorAll)) && (f(function(a) {
                a.innerHTML = "<select><option selected=''></option></select>", a.querySelectorAll("[selected]").length || J.push("\\[" + eb + "*(?:value|" + db + ")"), a.querySelectorAll(":checked").length || J.push(":checked")
            }), f(function(a) {
                var c = b.createElement("input");
                c.setAttribute("type", "hidden"), a.appendChild(c).setAttribute("t", ""), a.querySelectorAll("[t^='']").length && J.push("[*^$]=" + eb + "*(?:''|\"\")"), a.querySelectorAll(":enabled").length || J.push(":enabled", ":disabled"), a.querySelectorAll("*,:x"), J.push(",.*:")
            })), (x.matchesSelector = sb.test(L = H.webkitMatchesSelector || H.mozMatchesSelector || H.oMatchesSelector || H.msMatchesSelector)) && f(function(a) {
                x.disconnectedMatch = L.call(a, "div"), L.call(a, "[s!='']:x"), K.push("!=", ib)
            }), J = J.length && new RegExp(J.join("|")), K = K.length && new RegExp(K.join("|")), M = sb.test(H.contains) || H.compareDocumentPosition ? function(a, b) {
                var c = 9 === a.nodeType ? a.documentElement: a, d = b && b.parentNode;
                return a === d||!!d && 1 === d.nodeType&&!!(c.contains ? c.contains(d) : a.compareDocumentPosition && 16 & a.compareDocumentPosition(d))
            } : function(a, b) {
                if (b)
                    for (; b = b.parentNode;)
                        if (b === a)
                            return !0;
                return !1
            }, V = H.compareDocumentPosition ? function(a, c) {
                if (a === c)
                    return U=!0, 0;
                var d = c.compareDocumentPosition && a.compareDocumentPosition && a.compareDocumentPosition(c);
                return d ? 1 & d ||!x.sortDetached && c.compareDocumentPosition(a) === d ? a === b || M(O, a)?-1 : c === b || M(O, c) ? 1 : E ? cb.call(E, a) - cb.call(E, c) : 0 : 4 & d?-1 : 1 : a.compareDocumentPosition?-1 : 1
            } : function(a, c) {
                var d, e = 0, f = a.parentNode, g = c.parentNode, i = [a], j = [c];
                if (a === c)
                    return U=!0, 0;
                if (!f ||!g)
                    return a === b?-1 : c === b ? 1 : f?-1 : g ? 1 : E ? cb.call(E, a) - cb.call(E, c) : 0;
                if (f === g)
                    return h(a, c);
                for (d = a; d = d.parentNode;)
                    i.unshift(d);
                for (d = c; d = d.parentNode;)
                    j.unshift(d);
                for (; i[e] === j[e];)
                    e++;
                return e ? h(i[e], j[e]) : i[e] === O?-1 : j[e] === O ? 1 : 0
            }, b) : G
        }, c.matches = function(a, b) {
            return c(a, null, null, b)
        }, c.matchesSelector = function(a, b) {
            if ((a.ownerDocument || a) !== G && F(a), b = b.replace(ob, "='$1']"), !(!x.matchesSelector ||!I || K && K.test(b) || J && J.test(b)))
                try {
                    var d = L.call(a, b);
                    if (d || x.disconnectedMatch || a.document && 11 !== a.document.nodeType)
                        return d
            } catch (e) {}
            return c(b, G, null, [a]).length > 0
        }, c.contains = function(a, b) {
            return (a.ownerDocument || a) !== G && F(a), M(a, b)
        }, c.attr = function(a, c) {
            (a.ownerDocument || a) !== G && F(a);
            var d = z.attrHandle[c.toLowerCase()], e = d && Y.call(z.attrHandle, c.toLowerCase()) ? d(a, c, !I): b;
            return e === b ? x.attributes ||!I ? a.getAttribute(c) : (e = a.getAttributeNode(c)) && e.specified ? e.value : null : e
        }, c.error = function(a) {
            throw new Error("Syntax error, unrecognized expression: " + a)
        }, c.uniqueSort = function(a) {
            var b, c = [], d = 0, e = 0;
            if (U=!x.detectDuplicates, E=!x.sortStable && a.slice(0), a.sort(V), U) {
                for (; b = a[e++];)
                    b === a[e] && (d = c.push(e));
                for (; d--;)
                    a.splice(c[d], 1)
            }
            return a
        }, A = c.getText = function(a) {
            var b, c = "", d = 0, e = a.nodeType;
            if (e) {
                if (1 === e || 9 === e || 11 === e) {
                    if ("string" == typeof a.textContent)
                        return a.textContent;
                    for (a = a.firstChild; a; a = a.nextSibling)
                        c += A(a)
                    } else if (3 === e || 4 === e)
                    return a.nodeValue
            } else 
                for (; b = a[d]; d++)
                    c += A(b);
            return c
        }, z = c.selectors = {
            cacheLength: 50,
            createPseudo: e,
            match: rb,
            attrHandle: {},
            find: {},
            relative: {
                ">": {
                    dir: "parentNode",
                    first: !0
                },
                " ": {
                    dir: "parentNode"
                },
                "+": {
                    dir: "previousSibling",
                    first: !0
                },
                "~": {
                    dir: "previousSibling"
                }
            },
            preFilter: {
                ATTR: function(a) {
                    return a[1] = a[1].replace(xb, yb), a[3] = (a[4] || a[5] || "").replace(xb, yb), "~=" === a[2] && (a[3] = " " + a[3] + " "), a.slice(0, 4)
                },
                CHILD: function(a) {
                    return a[1] = a[1].toLowerCase(), "nth" === a[1].slice(0, 3) ? (a[3] || c.error(a[0]), a[4] =+ (a[4] ? a[5] + (a[6] || 1) : 2 * ("even" === a[3] || "odd" === a[3])), a[5] =+ (a[7] + a[8] || "odd" === a[3])) : a[3] && c.error(a[0]), a
                },
                PSEUDO: function(a) {
                    var c, d=!a[5] && a[2];
                    return rb.CHILD.test(a[0]) ? null : (a[3] && a[4] !== b ? a[2] = a[4] : d && pb.test(d) && (c = m(d, !0)) && (c = d.indexOf(")", d.length - c) - d.length) && (a[0] = a[0].slice(0, c), a[2] = d.slice(0, c)), a.slice(0, 3))
                }
            },
            filter: {
                TAG: function(a) {
                    var b = a.replace(xb, yb).toLowerCase();
                    return "*" === a ? function() {
                        return !0
                    } : function(a) {
                        return a.nodeName && a.nodeName.toLowerCase() === b
                    }
                },
                CLASS: function(a) {
                    var b = R[a + " "];
                    return b || (b = new RegExp("(^|" + eb + ")" + a + "(" + eb + "|$)")) && R(a, function(a) {
                        return b.test("string" == typeof a.className && a.className || typeof a.getAttribute !== W && a.getAttribute("class") || "")
                    })
                },
                ATTR: function(a, b, d) {
                    return function(e) {
                        var f = c.attr(e, a);
                        return null == f ? "!=" === b : b ? (f += "", "=" === b ? f === d : "!=" === b ? f !== d : "^=" === b ? d && 0 === f.indexOf(d) : "*=" === b ? d && f.indexOf(d)>-1 : "$=" === b ? d && f.slice( - d.length) === d : "~=" === b ? (" " + f + " ").indexOf(d)>-1 : "|=" === b ? f === d || f.slice(0, d.length + 1) === d + "-" : !1) : !0
                    }
                },
                CHILD: function(a, b, c, d, e) {
                    var f = "nth" !== a.slice(0, 3), g = "last" !== a.slice( - 4), h = "of-type" === b;
                    return 1 === d && 0 === e ? function(a) {
                        return !!a.parentNode
                    } : function(b, c, i) {
                        var j, k, l, m, n, o, p = f !== g ? "nextSibling": "previousSibling", q = b.parentNode, r = h && b.nodeName.toLowerCase(), s=!i&&!h;
                        if (q) {
                            if (f) {
                                for (; p;) {
                                    for (l = b; l = l[p];)
                                        if (h ? l.nodeName.toLowerCase() === r : 1 === l.nodeType)
                                            return !1;
                                    o = p = "only" === a&&!o && "nextSibling"
                                }
                                return !0
                            }
                            if (o = [g ? q.firstChild: q.lastChild], g && s) {
                                for (k = q[N] || (q[N] = {}), j = k[a] || [], n = j[0] === P && j[1], m = j[0] === P && j[2], l = n && q.childNodes[n]; l=++n && l && l[p] || (m = n = 0) 
                                    || o.pop();
                                )if (1 === l.nodeType&&++m && l === b) {
                                    k[a] = [P, n, m];
                                    break
                                }
                            } else if (s && (j = (b[N] || (b[N] = {}))[a]) && j[0] === P)
                                m = j[1];
                            else 
                                for (; (l=++n && l && l[p] || (m = n = 0) || o.pop()) 
                                    && ((h ? l.nodeName.toLowerCase() !== r : 1 !== l.nodeType)||!++m || (s && ((l[N] || (l[N] = {}))[a] = [P, m]), l !== b));
                            );
                            return m -= e, m === d || m%d === 0 && m / d >= 0
                        }
                    }
                },
                PSEUDO: function(a, b) {
                    var d, f = z.pseudos[a] || z.setFilters[a.toLowerCase()] || c.error("unsupported pseudo: " + a);
                    return f[N] ? f(b) : f.length > 1 ? (d = [a, a, "", b], z.setFilters.hasOwnProperty(a.toLowerCase()) ? e(function(a, c) {
                        for (var d, e = f(a, b), g = e.length; g--;)
                            d = cb.call(a, e[g]), a[d]=!(c[d] = e[g])
                    }) : function(a) {
                        return f(a, 0, d)
                    }) : f
                }
            },
            pseudos: {
                not: e(function(a) {
                    var b = [], c = [], d = C(a.replace(jb, "$1"));
                    return d[N] ? e(function(a, b, c, e) {
                        for (var f, g = d(a, null, e, []), h = a.length; h--;)(f = g[h]) 
                            && (a[h]=!(b[h] = f))
                    }) : function(a, e, f) {
                        return b[0] = a, d(b, null, f, c), !c.pop()
                    }
                }),
                has: e(function(a) {
                    return function(b) {
                        return c(a, b).length > 0
                    }
                }),
                contains: e(function(a) {
                    return function(b) {
                        return (b.textContent || b.innerText || A(b)).indexOf(a)>-1
                    }
                }),
                lang: e(function(a) {
                    return qb.test(a || "") || c.error("unsupported lang: " + a), a = a.replace(xb, yb).toLowerCase(), function(b) {
                        var c;
                        do 
                            if (c = I ? b.lang : b.getAttribute("xml:lang") || b.getAttribute("lang"))
                                return c = c.toLowerCase(), c === a || 0 === c.indexOf(a + "-");
                        while ((b = b.parentNode) && 1 === b.nodeType);
                        return !1
                    }
                }),
                target: function(b) {
                    var c = a.location && a.location.hash;
                    return c && c.slice(1) === b.id
                },
                root: function(a) {
                    return a === H
                },
                focus: function(a) {
                    return a === G.activeElement && (!G.hasFocus || G.hasFocus())&&!!(a.type || a.href||~a.tabIndex)
                },
                enabled: function(a) {
                    return a.disabled===!1
                },
                disabled: function(a) {
                    return a.disabled===!0
                },
                checked: function(a) {
                    var b = a.nodeName.toLowerCase();
                    return "input" === b&&!!a.checked || "option" === b&&!!a.selected
                },
                selected: function(a) {
                    return a.parentNode && a.parentNode.selectedIndex, a.selected===!0
                },
                empty: function(a) {
                    for (a = a.firstChild; a; a = a.nextSibling)
                        if (a.nodeName > "@" || 3 === a.nodeType || 4 === a.nodeType)
                            return !1;
                    return !0
                },
                parent: function(a) {
                    return !z.pseudos.empty(a)
                },
                header: function(a) {
                    return vb.test(a.nodeName)
                },
                input: function(a) {
                    return ub.test(a.nodeName)
                },
                button: function(a) {
                    var b = a.nodeName.toLowerCase();
                    return "input" === b && "button" === a.type || "button" === b
                },
                text: function(a) {
                    var b;
                    return "input" === a.nodeName.toLowerCase() && "text" === a.type && (null == (b = a.getAttribute("type")) || b.toLowerCase() === a.type)
                },
                first: k(function() {
                    return [0]
                }),
                last: k(function(a, b) {
                    return [b - 1]
                }),
                eq: k(function(a, b, c) {
                    return [0 > c ? c + b: c]
                }),
                even: k(function(a, b) {
                    for (var c = 0; b > c; c += 2)
                        a.push(c);
                    return a
                }),
                odd: k(function(a, b) {
                    for (var c = 1; b > c; c += 2)
                        a.push(c);
                    return a
                }),
                lt: k(function(a, b, c) {
                    for (var d = 0 > c ? c + b : c; --d >= 0;)
                        a.push(d);
                    return a
                }),
                gt: k(function(a, b, c) {
                    for (var d = 0 > c ? c + b : c; ++d < b;)
                        a.push(d);
                    return a
                })
            }
        }, z.pseudos.nth = z.pseudos.eq;
        for (w in{
            radio: !0,
            checkbox: !0,
            file: !0,
            password: !0,
            image: !0
        })
            z.pseudos[w] = i(w);
        for (w in{
            submit: !0,
            reset: !0
        })
            z.pseudos[w] = j(w);
        l.prototype = z.filters = z.pseudos, z.setFilters = new l, C = c.compile = function(a, b) {
            var c, d = [], e = [], f = T[a + " "];
            if (!f) {
                for (b || (b = m(a)), c = b.length; c--;)
                    f = s(b[c]), f[N] ? d.push(f) : e.push(f);
                f = T(a, t(e, d))
            }
            return f
        }, x.sortStable = N.split("").sort(V).join("") === N, x.detectDuplicates = U, F(), x.sortDetached = f(function(a) {
            return 1 & a.compareDocumentPosition(G.createElement("div"))
        }), f(function(a) {
            return a.innerHTML = "<a href='#'></a>", "#" === a.firstChild.getAttribute("href")
        }) || g("type|href|height|width", function(a, b, c) {
            return c ? void 0 : a.getAttribute(b, "type" === b.toLowerCase() ? 1 : 2)
        }), (!x.attributes ||!f(function(a) {
            return a.innerHTML = "<input/>", a.firstChild.setAttribute("value", ""), "" === a.firstChild.getAttribute("value")
        })) && g("value", function(a, b, c) {
            return c || "input" !== a.nodeName.toLowerCase() ? void 0 : a.defaultValue
        }), f(function(a) {
            return null == a.getAttribute("disabled")
        }) || g(db, function(a, b, c) {
            var d;
            return c ? void 0 : (d = a.getAttributeNode(b)) && d.specified ? d.value : a[b]===!0 ? b.toLowerCase() : null
        }), kb.find = c, kb.expr = c.selectors, kb.expr[":"] = kb.expr.pseudos, kb.unique = c.uniqueSort, kb.text = c.getText, kb.isXMLDoc = c.isXML, kb.contains = c.contains
    }(a);
    var zb = {};
    kb.Callbacks = function(a) {
        a = "string" == typeof a ? zb[a] || d(a) : kb.extend({}, a);
        var c, e, f, g, h, i, j = [], k=!a.once && [], l = function(b) {
            for (e = a.memory && b, f=!0, h = i || 0, i = 0, g = j.length, c=!0; j && g > h; h++)
                if (j[h].apply(b[0], b[1])===!1 && a.stopOnFalse) {
                    e=!1;
                    break
                }
            c=!1, j && (k ? k.length && l(k.shift()) : e ? j = [] : m.disable())
        }, m = {
            add: function() {
                if (j) {
                    var b = j.length;
                    !function d(b) {
                        kb.each(b, function(b, c) {
                            var e = kb.type(c);
                            "function" === e ? (!a.unique ||!m.has(c)) && j.push(c) : c && c.length && "string" !== e && d(c)
                        })
                    }(arguments), c ? g = j.length : e && (i = b, l(e))
                }
                return this
            },
            remove: function() {
                return j && kb.each(arguments, function(a, b) {
                    for (var d; (d = kb.inArray(b, j, d))>-1;)
                        j.splice(d, 1), c && (g >= d && g--, h >= d && h--)
                }), this
            },
            has: function(a) {
                return a ? kb.inArray(a, j)>-1 : !!j&&!!j.length
            },
            empty: function() {
                return j = [], g = 0, this
            },
            disable: function() {
                return j = k = e = b, this
            },
            disabled: function() {
                return !j
            },
            lock: function() {
                return k = b, e || m.disable(), this
            },
            locked: function() {
                return !k
            },
            fireWith: function(a, b) {
                return j && (!f || k) && (b = b || [], b = [a, b.slice ? b.slice(): b], c ? k.push(b) : l(b)), this
            },
            fire: function() {
                return m.fireWith(this, arguments), this
            },
            fired: function() {
                return !!f
            }
        };
        return m
    }, kb.extend({
        Deferred: function(a) {
            var b = [["resolve", "done", kb.Callbacks("once memory"), "resolved"], ["reject", "fail", kb.Callbacks("once memory"), "rejected"], ["notify", "progress", kb.Callbacks("memory")]], c = "pending", d = {
                state: function() {
                    return c
                },
                always: function() {
                    return e.done(arguments).fail(arguments), this
                },
                then: function() {
                    var a = arguments;
                    return kb.Deferred(function(c) {
                        kb.each(b, function(b, f) {
                            var g = f[0], h = kb.isFunction(a[b]) && a[b];
                            e[f[1]](function() {
                                var a = h && h.apply(this, arguments);
                                a && kb.isFunction(a.promise) ? a.promise().done(c.resolve).fail(c.reject).progress(c.notify) : c[g + "With"](this === d ? c.promise() : this, h ? [a] : arguments)
                            })
                        }), a = null
                    }).promise()
                },
                promise: function(a) {
                    return null != a ? kb.extend(a, d) : d
                }
            }, e = {};
            return d.pipe = d.then, kb.each(b, function(a, f) {
                var g = f[2], h = f[3];
                d[f[1]] = g.add, h && g.add(function() {
                    c = h
                }, b[1^a][2].disable, b[2][2].lock), e[f[0]] = function() {
                    return e[f[0] + "With"](this === e ? d : this, arguments), this
                }, e[f[0] + "With"] = g.fireWith
            }), d.promise(e), a && a.call(e, e), e
        },
        when: function(a) {
            var b, c, d, e = 0, f = fb.call(arguments), g = f.length, h = 1 !== g || a && kb.isFunction(a.promise) ? g: 0, i = 1 === h ? a: kb.Deferred(), j = function(a, c, d) {
                return function(e) {
                    c[a] = this, d[a] = arguments.length > 1 ? fb.call(arguments) : e, d === b ? i.notifyWith(c, d) : --h || i.resolveWith(c, d)
                }
            };
            if (g > 1)
                for (b = new Array(g), c = new Array(g), d = new Array(g); g > e; e++)
                    f[e] && kb.isFunction(f[e].promise) ? f[e].promise().done(j(e, d, f)).fail(i.reject).progress(j(e, c, b)) : --h;
            return h || i.resolveWith(d, f), i.promise()
        }
    }), kb.support = function(b) {
        var c, d, e, f, g, h, i, j, k, l = Y.createElement("div");
        if (l.setAttribute("className", "t"), l.innerHTML = "  <link/><table></table><a href='/a'>a</a><input type='checkbox'/>", c = l.getElementsByTagName("*") || [], d = l.getElementsByTagName("a")[0], !d ||!d.style ||!c.length)
            return b;
        f = Y.createElement("select"), h = f.appendChild(Y.createElement("option")), e = l.getElementsByTagName("input")[0], d.style.cssText = "top:1px;float:left;opacity:.5", b.getSetAttribute = "t" !== l.className, b.leadingWhitespace = 3 === l.firstChild.nodeType, b.tbody=!l.getElementsByTagName("tbody").length, b.htmlSerialize=!!l.getElementsByTagName("link").length, b.style = /top/.test(d.getAttribute("style")), b.hrefNormalized = "/a" === d.getAttribute("href"), b.opacity = /^0.5/.test(d.style.opacity), b.cssFloat=!!d.style.cssFloat, b.checkOn=!!e.value, b.optSelected = h.selected, b.enctype=!!Y.createElement("form").enctype, b.html5Clone = "<:nav></:nav>" !== Y.createElement("nav").cloneNode(!0).outerHTML, b.inlineBlockNeedsLayout=!1, b.shrinkWrapBlocks=!1, b.pixelPosition=!1, b.deleteExpando=!0, b.noCloneEvent=!0, b.reliableMarginRight=!0, b.boxSizingReliable=!0, e.checked=!0, b.noCloneChecked = e.cloneNode(!0).checked, f.disabled=!0, b.optDisabled=!h.disabled;
        try {
            delete l.test
        } catch (m) {
            b.deleteExpando=!1
        }
        e = Y.createElement("input"), e.setAttribute("value", ""), b.input = "" === e.getAttribute("value"), e.value = "t", e.setAttribute("type", "radio"), b.radioValue = "t" === e.value, e.setAttribute("checked", "t"), e.setAttribute("name", "t"), g = Y.createDocumentFragment(), g.appendChild(e), b.appendChecked = e.checked, b.checkClone = g.cloneNode(!0).cloneNode(!0).lastChild.checked, l.attachEvent && (l.attachEvent("onclick", function() {
            b.noCloneEvent=!1
        }), l.cloneNode(!0).click());
        for (k in{
            submit: !0,
            change: !0,
            focusin: !0
        })
            l.setAttribute(i = "on" + k, "t"), b[k + "Bubbles"] = i in a || l.attributes[i].expando===!1;
        l.style.backgroundClip = "content-box", l.cloneNode(!0).style.backgroundClip = "", b.clearCloneStyle = "content-box" === l.style.backgroundClip;
        for (k in kb(b))
            break;
        return b.ownLast = "0" !== k, kb(function() {
            var c, d, e, f = "padding:0;margin:0;border:0;display:block;box-sizing:content-box;-moz-box-sizing:content-box;-webkit-box-sizing:content-box;", g = Y.getElementsByTagName("body")[0];
            g && (c = Y.createElement("div"), c.style.cssText = "border:0;width:0;height:0;position:absolute;top:0;left:-9999px;margin-top:1px", g.appendChild(c).appendChild(l), l.innerHTML = "<table><tr><td></td><td>t</td></tr></table>", e = l.getElementsByTagName("td"), e[0].style.cssText = "padding:0;margin:0;border:0;display:none", j = 0 === e[0].offsetHeight, e[0].style.display = "", e[1].style.display = "none", b.reliableHiddenOffsets = j && 0 === e[0].offsetHeight, l.innerHTML = "", l.style.cssText = "box-sizing:border-box;-moz-box-sizing:border-box;-webkit-box-sizing:border-box;padding:1px;border:1px;display:block;width:4px;margin-top:1%;position:absolute;top:1%;", kb.swap(g, null != g.style.zoom ? {
                zoom: 1
            } : {}, function() {
                b.boxSizing = 4 === l.offsetWidth
            }), a.getComputedStyle && (b.pixelPosition = "1%" !== (a.getComputedStyle(l, null) || {}).top, b.boxSizingReliable = "4px" === (a.getComputedStyle(l, null) || {
                width: "4px"
            }).width, d = l.appendChild(Y.createElement("div")), d.style.cssText = l.style.cssText = f, d.style.marginRight = d.style.width = "0", l.style.width = "1px", b.reliableMarginRight=!parseFloat((a.getComputedStyle(d, null) || {}).marginRight)), typeof l.style.zoom !== W && (l.innerHTML = "", l.style.cssText = f + "width:1px;padding:1px;display:inline;zoom:1", b.inlineBlockNeedsLayout = 3 === l.offsetWidth, l.style.display = "block", l.innerHTML = "<div></div>", l.firstChild.style.width = "5px", b.shrinkWrapBlocks = 3 !== l.offsetWidth, b.inlineBlockNeedsLayout && (g.style.zoom = 1)), g.removeChild(c), c = l = e = d = null)
        }), c = f = g = h = d = e = null, b
    }({});
    var Ab = /(?:\{[\s\S]*\}|\[[\s\S]*\])$/, Bb = /([A-Z])/g;
    kb.extend({
        cache: {},
        noData: {
            applet: !0,
            embed: !0,
            object: "clsid:D27CDB6E-AE6D-11cf-96B8-444553540000"
        },
        hasData: function(a) {
            return a = a.nodeType ? kb.cache[a[kb.expando]] : a[kb.expando], !!a&&!h(a)
        },
        data: function(a, b, c) {
            return e(a, b, c)
        },
        removeData: function(a, b) {
            return f(a, b)
        },
        _data: function(a, b, c) {
            return e(a, b, c, !0)
        },
        _removeData: function(a, b) {
            return f(a, b, !0)
        },
        acceptData: function(a) {
            if (a.nodeType && 1 !== a.nodeType && 9 !== a.nodeType)
                return !1;
            var b = a.nodeName && kb.noData[a.nodeName.toLowerCase()];
            return !b || b!==!0 && a.getAttribute("classid") === b
        }
    }), kb.fn.extend({
        data: function(a, c) {
            var d, e, f = null, h = 0, i = this[0];
            if (a === b) {
                if (this.length && (f = kb.data(i), 1 === i.nodeType&&!kb._data(i, "parsedAttrs"))) {
                    for (d = i.attributes; h < d.length; h++)
                        e = d[h].name, 0 === e.indexOf("data-") && (e = kb.camelCase(e.slice(5)), g(i, e, f[e]));
                    kb._data(i, "parsedAttrs", !0)
                }
                return f
            }
            return "object" == typeof a ? this.each(function() {
                kb.data(this, a)
            }) : arguments.length > 1 ? this.each(function() {
                kb.data(this, a, c)
            }) : i ? g(i, a, kb.data(i, a)) : null
        },
        removeData: function(a) {
            return this.each(function() {
                kb.removeData(this, a)
            })
        }
    }), kb.extend({
        queue: function(a, b, c) {
            var d;
            return a ? (b = (b || "fx") + "queue", d = kb._data(a, b), c && (!d || kb.isArray(c) ? d = kb._data(a, b, kb.makeArray(c)) : d.push(c)), d || []) : void 0
        },
        dequeue: function(a, b) {
            b = b || "fx";
            var c = kb.queue(a, b), d = c.length, e = c.shift(), f = kb._queueHooks(a, b), g = function() {
                kb.dequeue(a, b)
            };
            "inprogress" === e && (e = c.shift(), d--), e && ("fx" === b && c.unshift("inprogress"), delete f.stop, e.call(a, g, f)), !d && f && f.empty.fire()
        },
        _queueHooks: function(a, b) {
            var c = b + "queueHooks";
            return kb._data(a, c) || kb._data(a, c, {
                empty: kb.Callbacks("once memory").add(function() {
                    kb._removeData(a, b + "queue"), kb._removeData(a, c)
                })
            })
        }
    }), kb.fn.extend({
        queue: function(a, c) {
            var d = 2;
            return "string" != typeof a && (c = a, a = "fx", d--), arguments.length < d ? kb.queue(this[0], a) : c === b ? this : this.each(function() {
                var b = kb.queue(this, a, c);
                kb._queueHooks(this, a), "fx" === a && "inprogress" !== b[0] && kb.dequeue(this, a)
            })
        },
        dequeue: function(a) {
            return this.each(function() {
                kb.dequeue(this, a)
            })
        },
        delay: function(a, b) {
            return a = kb.fx ? kb.fx.speeds[a] || a : a, b = b || "fx", this.queue(b, function(b, c) {
                var d = setTimeout(b, a);
                c.stop = function() {
                    clearTimeout(d)
                }
            })
        },
        clearQueue: function(a) {
            return this.queue(a || "fx", [])
        },
        promise: function(a, c) {
            var d, e = 1, f = kb.Deferred(), g = this, h = this.length, i = function() {
                --e || f.resolveWith(g, [g])
            };
            for ("string" != typeof a && (c = a, a = b), a = a || "fx"; h--;)
                d = kb._data(g[h], a + "queueHooks"), d && d.empty && (e++, d.empty.add(i));
            return i(), f.promise(c)
        }
    });
    var Cb, Db, Eb = /[\t\r\n\f]/g, Fb = /\r/g, Gb = /^(?:input|select|textarea|button|object)$/i, Hb = /^(?:a|area)$/i, Ib = /^(?:checked|selected)$/i, Jb = kb.support.getSetAttribute, Kb = kb.support.input;
    kb.fn.extend({
        attr: function(a, b) {
            return kb.access(this, kb.attr, a, b, arguments.length > 1)
        },
        removeAttr: function(a) {
            return this.each(function() {
                kb.removeAttr(this, a)
            })
        },
        prop: function(a, b) {
            return kb.access(this, kb.prop, a, b, arguments.length > 1)
        },
        removeProp: function(a) {
            return a = kb.propFix[a] || a, this.each(function() {
                try {
                    this[a] = b, delete this[a]
                } catch (c) {}
            })
        },
        addClass: function(a) {
            var b, c, d, e, f, g = 0, h = this.length, i = "string" == typeof a && a;
            if (kb.isFunction(a))
                return this.each(function(b) {
                    kb(this).addClass(a.call(this, b, this.className))
                });
            if (i)
                for (b = (a || "").match(mb) || []; h > g; g++)
                    if (c = this[g], d = 1 === c.nodeType && (c.className ? (" " + c.className + " ").replace(Eb, " ") : " ")) {
                        for (f = 0; e = b[f++];)
                            d.indexOf(" " + e + " ") < 0 && (d += e + " ");
                            c.className = kb.trim(d)
                    }
            return this
        },
        removeClass: function(a) {
            var b, c, d, e, f, g = 0, h = this.length, i = 0 === arguments.length || "string" == typeof a && a;
            if (kb.isFunction(a))
                return this.each(function(b) {
                    kb(this).removeClass(a.call(this, b, this.className))
                });
            if (i)
                for (b = (a || "").match(mb) || []; h > g; g++)
                    if (c = this[g], d = 1 === c.nodeType && (c.className ? (" " + c.className + " ").replace(Eb, " ") : "")) {
                        for (f = 0; e = b[f++];)
                            for (; d.indexOf(" " + e + " ") >= 0;)
                                d = d.replace(" " + e + " ", " ");
                                c.className = a ? kb.trim(d) : ""
                    }
            return this
        },
        toggleClass: function(a, b) {
            var c = typeof a;
            return "boolean" == typeof b && "string" === c ? b ? this.addClass(a) : this.removeClass(a) : this.each(kb.isFunction(a) ? function(c) {
                kb(this).toggleClass(a.call(this, c, this.className, b), b)
            } : function() {
                if ("string" === c)
                    for (var b, d = 0, e = kb(this), f = a.match(mb) || []; b = f[d++];)
                        e.hasClass(b) ? e.removeClass(b) : e.addClass(b);
                else (c === W || "boolean" === c) 
                    && (this.className && kb._data(this, "__className__", this.className), this.className = this.className || a===!1 ? "" : kb._data(this, "__className__") || "")
            })
        },
        hasClass: function(a) {
            for (var b = " " + a + " ", c = 0, d = this.length; d > c; c++)
                if (1 === this[c].nodeType && (" " + this[c].className + " ").replace(Eb, " ").indexOf(b) >= 0)
                    return !0;
            return !1
        },
        val: function(a) {
            var c, d, e, f = this[0];
            {
                if (arguments.length)
                    return e = kb.isFunction(a), this.each(function(c) {
                        var f;
                        1 === this.nodeType && (f = e ? a.call(this, c, kb(this).val()) : a, null == f ? f = "" : "number" == typeof f ? f += "" : kb.isArray(f) && (f = kb.map(f, function(a) {
                            return null == a ? "" : a + ""
                        })), d = kb.valHooks[this.type] || kb.valHooks[this.nodeName.toLowerCase()], d && "set"in d && d.set(this, f, "value") !== b || (this.value = f))
                    });
                if (f)
                    return d = kb.valHooks[f.type] || kb.valHooks[f.nodeName.toLowerCase()], d && "get"in d && (c = d.get(f, "value")) !== b ? c : (c = f.value, "string" == typeof c ? c.replace(Fb, "") : null == c ? "" : c)
            }
        }
    }), kb.extend({
        valHooks: {
            option: {
                get: function(a) {
                    var b = kb.find.attr(a, "value");
                    return null != b ? b : a.text
                }
            },
            select: {
                get: function(a) {
                    for (var b, c, d = a.options, e = a.selectedIndex, f = "select-one" === a.type || 0 > e, g = f ? null : [], h = f ? e + 1 : d.length, i = 0 > e ? h : f ? e : 0; h > i; i++)
                        if (c = d[i], !(!c.selected && i !== e || (kb.support.optDisabled ? c.disabled : null !== c.getAttribute("disabled")) || c.parentNode.disabled && kb.nodeName(c.parentNode, "optgroup"))) {
                            if (b = kb(c).val(), f)
                                return b;
                                g.push(b)
                        }
                    return g
                },
                set: function(a, b) {
                    for (var c, d, e = a.options, f = kb.makeArray(b), g = e.length; g--;)
                        d = e[g], (d.selected = kb.inArray(kb(d).val(), f) >= 0) && (c=!0);
                    return c || (a.selectedIndex =- 1), f
                }
            }
        },
        attr: function(a, c, d) {
            var e, f, g = a.nodeType;
            if (a && 3 !== g && 8 !== g && 2 !== g)
                return typeof a.getAttribute === W ? kb.prop(a, c, d) : (1 === g && kb.isXMLDoc(a) || (c = c.toLowerCase(), e = kb.attrHooks[c] || (kb.expr.match.bool.test(c) ? Db : Cb)), d === b ? e && "get"in e && null !== (f = e.get(a, c)) ? f : (f = kb.find.attr(a, c), null == f ? b : f) : null !== d ? e && "set"in e && (f = e.set(a, d, c)) !== b ? f : (a.setAttribute(c, d + ""), d) : void kb.removeAttr(a, c))
        },
        removeAttr: function(a, b) {
            var c, d, e = 0, f = b && b.match(mb);
            if (f && 1 === a.nodeType)
                for (; c = f[e++];)
                    d = kb.propFix[c] || c, kb.expr.match.bool.test(c) ? Kb && Jb ||!Ib.test(c) ? a[d]=!1 : a[kb.camelCase("default-" + c)] = a[d]=!1 : kb.attr(a, c, ""), a.removeAttribute(Jb ? c : d)
        },
        attrHooks: {
            type: {
                set: function(a, b) {
                    if (!kb.support.radioValue && "radio" === b && kb.nodeName(a, "input")) {
                        var c = a.value;
                        return a.setAttribute("type", b), c && (a.value = c), b
                    }
                }
            }
        },
        propFix: {
            "for": "htmlFor",
            "class": "className"
        },
        prop: function(a, c, d) {
            var e, f, g, h = a.nodeType;
            if (a && 3 !== h && 8 !== h && 2 !== h)
                return g = 1 !== h ||!kb.isXMLDoc(a), g && (c = kb.propFix[c] || c, f = kb.propHooks[c]), d !== b ? f && "set"in f && (e = f.set(a, d, c)) !== b ? e : a[c] = d : f && "get"in f && null !== (e = f.get(a, c)) ? e : a[c]
        },
        propHooks: {
            tabIndex: {
                get: function(a) {
                    var b = kb.find.attr(a, "tabindex");
                    return b ? parseInt(b, 10) : Gb.test(a.nodeName) || Hb.test(a.nodeName) && a.href ? 0 : - 1
                }
            }
        }
    }), Db = {
        set: function(a, b, c) {
            return b===!1 ? kb.removeAttr(a, c) : Kb && Jb ||!Ib.test(c) ? a.setAttribute(!Jb && kb.propFix[c] || c, c) : a[kb.camelCase("default-" + c)] = a[c]=!0, c
        }
    }, kb.each(kb.expr.match.bool.source.match(/\w+/g), function(a, c) {
        var d = kb.expr.attrHandle[c] || kb.find.attr;
        kb.expr.attrHandle[c] = Kb && Jb ||!Ib.test(c) ? function(a, c, e) {
            var f = kb.expr.attrHandle[c], g = e ? b: (kb.expr.attrHandle[c] = b) != d(a, c, e) ? c.toLowerCase(): null;
            return kb.expr.attrHandle[c] = f, g
        } : function(a, c, d) {
            return d ? b : a[kb.camelCase("default-" + c)] ? c.toLowerCase() : null
        }
    }), Kb && Jb || (kb.attrHooks.value = {
        set: function(a, b, c) {
            return kb.nodeName(a, "input") ? void(a.defaultValue = b) : Cb && Cb.set(a, b, c)
        }
    }), Jb || (Cb = {
        set: function(a, c, d) {
            var e = a.getAttributeNode(d);
            return e || a.setAttributeNode(e = a.ownerDocument.createAttribute(d)), e.value = c += "", "value" === d || c === a.getAttribute(d) ? c : b
        }
    }, kb.expr.attrHandle.id = kb.expr.attrHandle.name = kb.expr.attrHandle.coords = function(a, c, d) {
        var e;
        return d ? b : (e = a.getAttributeNode(c)) && "" !== e.value ? e.value : null
    }, kb.valHooks.button = {
        get: function(a, c) {
            var d = a.getAttributeNode(c);
            return d && d.specified ? d.value : b
        },
        set: Cb.set
    }, kb.attrHooks.contenteditable = {
        set: function(a, b, c) {
            Cb.set(a, "" === b?!1 : b, c)
        }
    }, kb.each(["width", "height"], function(a, b) {
        kb.attrHooks[b] = {
            set: function(a, c) {
                return "" === c ? (a.setAttribute(b, "auto"), c) : void 0
            }
        }
    })), kb.support.hrefNormalized || kb.each(["href", "src"], function(a, b) {
        kb.propHooks[b] = {
            get: function(a) {
                return a.getAttribute(b, 4)
            }
        }
    }), kb.support.style || (kb.attrHooks.style = {
        get: function(a) {
            return a.style.cssText || b
        },
        set: function(a, b) {
            return a.style.cssText = b + ""
        }
    }), kb.support.optSelected || (kb.propHooks.selected = {
        get: function(a) {
            var b = a.parentNode;
            return b && (b.selectedIndex, b.parentNode && b.parentNode.selectedIndex), null
        }
    }), kb.each(["tabIndex", "readOnly", "maxLength", "cellSpacing", "cellPadding", "rowSpan", "colSpan", "useMap", "frameBorder", "contentEditable"], function() {
        kb.propFix[this.toLowerCase()] = this
    }), kb.support.enctype || (kb.propFix.enctype = "encoding"), kb.each(["radio", "checkbox"], function() {
        kb.valHooks[this] = {
            set: function(a, b) {
                return kb.isArray(b) ? a.checked = kb.inArray(kb(a).val(), b) >= 0 : void 0
            }
        }, kb.support.checkOn || (kb.valHooks[this].get = function(a) {
            return null === a.getAttribute("value") ? "on" : a.value
        })
    });
    var Lb = /^(?:input|select|textarea)$/i, Mb = /^key/, Nb = /^(?:mouse|contextmenu)|click/, Ob = /^(?:focusinfocus|focusoutblur)$/, Pb = /^([^.]*)(?:\.(.+)|)$/;
    kb.event = {
        global: {},
        add: function(a, c, d, e, f) {
            var g, h, i, j, k, l, m, n, o, p, q, r = kb._data(a);
            if (r) {
                for (d.handler && (j = d, d = j.handler, f = j.selector), d.guid || (d.guid = kb.guid++), (h = r.events) || (h = r.events = {}), (l = r.handle) || (l = r.handle = function(a) {
                    return typeof kb === W || a && kb.event.triggered === a.type ? b : kb.event.dispatch.apply(l.elem, arguments)
                }, l.elem = a), c = (c || "").match(mb) || [""], i = c.length; i--;)
                    g = Pb.exec(c[i]) || [], o = q = g[1], p = (g[2] || "").split(".").sort(), o && (k = kb.event.special[o] || {}, o = (f ? k.delegateType : k.bindType) || o, k = kb.event.special[o] || {}, m = kb.extend({
                        type: o,
                        origType: q,
                        data: e,
                        handler: d,
                        guid: d.guid,
                        selector: f,
                        needsContext: f && kb.expr.match.needsContext.test(f),
                        namespace: p.join(".")
                    }, j), (n = h[o]) || (n = h[o] = [], n.delegateCount = 0, k.setup && k.setup.call(a, e, p, l)!==!1 || (a.addEventListener ? a.addEventListener(o, l, !1) : a.attachEvent && a.attachEvent("on" + o, l))), k.add && (k.add.call(a, m), m.handler.guid || (m.handler.guid = d.guid)), f ? n.splice(n.delegateCount++, 0, m) : n.push(m), kb.event.global[o]=!0);
                a = null
            }
        },
        remove: function(a, b, c, d, e) {
            var f, g, h, i, j, k, l, m, n, o, p, q = kb.hasData(a) && kb._data(a);
            if (q && (k = q.events)) {
                for (b = (b || "").match(mb) || [""], j = b.length; j--;)
                    if (h = Pb.exec(b[j]) || [], n = p = h[1], o = (h[2] || "").split(".").sort(), n) {
                        for (l = kb.event.special[n] || {}, n = (d ? l.delegateType : l.bindType) || n, m = k[n] || [], h = h[2] && new RegExp("(^|\\.)" + o.join("\\.(?:.*\\.|)") + "(\\.|$)"), i = f = m.length; f--;)
                            g = m[f], !(!e && p !== g.origType || c && c.guid !== g.guid || h&&!h.test(g.namespace) || d && d !== g.selector && ("**" !== d ||!g.selector) || (m.splice(f, 1), g.selector && m.delegateCount--, !l.remove ||!l.remove.call(a, g)));
                            i&&!m.length && ((!l.teardown || l.teardown.call(a, o, q.handle)===!1) && kb.removeEvent(a, n, q.handle), delete k[n])
                    } else 
                        for (n in k)
                            kb.event.remove(a, n + b[j], c, d, !0);
                kb.isEmptyObject(k) && (delete q.handle, kb._removeData(a, "events"))
            }
        },
        trigger: function(c, d, e, f) {
            var g, h, i, j, k, l, m, n = [e || Y], o = ib.call(c, "type") ? c.type: c, p = ib.call(c, "namespace") ? c.namespace.split("."): [];
            if (i = l = e = e || Y, 3 !== e.nodeType && 8 !== e.nodeType&&!Ob.test(o + kb.event.triggered) && (o.indexOf(".") >= 0 && (p = o.split("."), o = p.shift(), p.sort()), h = o.indexOf(":") < 0 && "on" + o, c = c[kb.expando] ? c : new kb.Event(o, "object" == typeof c && c), c.isTrigger = f ? 2 : 3, c.namespace = p.join("."), c.namespace_re = c.namespace ? new RegExp("(^|\\.)" + p.join("\\.(?:.*\\.|)") + "(\\.|$)") : null, c.result = b, c.target || (c.target = e), d = null == d ? [c] : kb.makeArray(d, [c]), k = kb.event.special[o] || {}, f ||!k.trigger || k.trigger.apply(e, d)!==!1)) {
                if (!f&&!k.noBubble&&!kb.isWindow(e)) {
                    for (j = k.delegateType || o, Ob.test(j + o) || (i = i.parentNode); i; i = i.parentNode)
                        n.push(i), l = i;
                    l === (e.ownerDocument || Y) && n.push(l.defaultView || l.parentWindow || a)
                }
                for (m = 0; (i = n[m++])&&!c.isPropagationStopped();)
                    c.type = m > 1 ? j : k.bindType || o, g = (kb._data(i, "events") || {})[c.type] && kb._data(i, "handle"), g && g.apply(i, d), g = h && i[h], g && kb.acceptData(i) && g.apply && g.apply(i, d)===!1 && c.preventDefault();
                if (c.type = o, !f&&!c.isDefaultPrevented() && (!k._default || k._default.apply(n.pop(), d)===!1) && kb.acceptData(e) && h && e[o]&&!kb.isWindow(e)) {
                    l = e[h], l && (e[h] = null), kb.event.triggered = o;
                    try {
                        e[o]()
                    } catch (q) {}
                    kb.event.triggered = b, l && (e[h] = l)
                }
                return c.result
            }
        },
        dispatch: function(a) {
            a = kb.event.fix(a);
            var c, d, e, f, g, h = [], i = fb.call(arguments), j = (kb._data(this, "events") || {})[a.type] || [], k = kb.event.special[a.type] || {};
            if (i[0] = a, a.delegateTarget = this, !k.preDispatch || k.preDispatch.call(this, a)!==!1) {
                for (h = kb.event.handlers.call(this, a, j), c = 0; (f = h[c++])&&!a.isPropagationStopped();)
                    for (a.currentTarget = f.elem, g = 0; (e = f.handlers[g++])&&!a.isImmediatePropagationStopped();)(!a.namespace_re || a.namespace_re.test(e.namespace)
                        ) && (a.handleObj = e, a.data = e.data, d = ((kb.event.special[e.origType] || {}).handle || e.handler).apply(f.elem, i), d !== b && (a.result = d)===!1 && (a.preventDefault(), a.stopPropagation()));
                return k.postDispatch && k.postDispatch.call(this, a), a.result
            }
        },
        handlers: function(a, c) {
            var d, e, f, g, h = [], i = c.delegateCount, j = a.target;
            if (i && j.nodeType && (!a.button || "click" !== a.type))
                for (; j != this; j = j.parentNode || this)
                    if (1 === j.nodeType && (j.disabled!==!0 || "click" !== a.type)) {
                        for (f = [], g = 0; i > g; g++)
                            e = c[g], d = e.selector + " ", f[d] === b && (f[d] = e.needsContext ? kb(d, this).index(j) >= 0 : kb.find(d, this, null, [j]).length), f[d] && f.push(e);
                            f.length && h.push({
                                elem: j,
                                handlers: f
                            })
                    }
            return i < c.length && h.push({
                elem: this,
                handlers: c.slice(i)
            }), h
        },
        fix: function(a) {
            if (a[kb.expando])
                return a;
            var b, c, d, e = a.type, f = a, g = this.fixHooks[e];
            for (g || (this.fixHooks[e] = g = Nb.test(e) ? this.mouseHooks : Mb.test(e) ? this.keyHooks : {}), d = g.props ? this.props.concat(g.props) : this.props, a = new kb.Event(f), b = d.length; b--;)
                c = d[b], a[c] = f[c];
            return a.target || (a.target = f.srcElement || Y), 3 === a.target.nodeType && (a.target = a.target.parentNode), a.metaKey=!!a.metaKey, g.filter ? g.filter(a, f) : a
        },
        props: "altKey bubbles cancelable ctrlKey currentTarget eventPhase metaKey relatedTarget shiftKey target timeStamp view which".split(" "),
        fixHooks: {},
        keyHooks: {
            props: "char charCode key keyCode".split(" "),
            filter: function(a, b) {
                return null == a.which && (a.which = null != b.charCode ? b.charCode : b.keyCode), a
            }
        },
        mouseHooks: {
            props: "button buttons clientX clientY fromElement offsetX offsetY pageX pageY screenX screenY toElement".split(" "),
            filter: function(a, c) {
                var d, e, f, g = c.button, h = c.fromElement;
                return null == a.pageX && null != c.clientX && (e = a.target.ownerDocument || Y, f = e.documentElement, d = e.body, a.pageX = c.clientX + (f && f.scrollLeft || d && d.scrollLeft || 0) - (f && f.clientLeft || d && d.clientLeft || 0), a.pageY = c.clientY + (f && f.scrollTop || d && d.scrollTop || 0) - (f && f.clientTop || d && d.clientTop || 0)), !a.relatedTarget && h && (a.relatedTarget = h === a.target ? c.toElement : h), !a.which && g !== b && (a.which = 1 & g ? 1 : 2 & g ? 3 : 4 & g ? 2 : 0), a
            }
        },
        special: {
            load: {
                noBubble: !0
            },
            focus: {
                trigger: function() {
                    if (this !== k() && this.focus)
                        try {
                            return this.focus(), !1
                    } catch (a) {}
                },
                delegateType: "focusin"
            },
            blur: {
                trigger: function() {
                    return this === k() && this.blur ? (this.blur(), !1) : void 0
                },
                delegateType: "focusout"
            },
            click: {
                trigger: function() {
                    return kb.nodeName(this, "input") && "checkbox" === this.type && this.click ? (this.click(), !1) : void 0
                },
                _default: function(a) {
                    return kb.nodeName(a.target, "a")
                }
            },
            beforeunload: {
                postDispatch: function(a) {
                    a.result !== b && (a.originalEvent.returnValue = a.result)
                }
            }
        },
        simulate: function(a, b, c, d) {
            var e = kb.extend(new kb.Event, c, {
                type: a,
                isSimulated: !0,
                originalEvent: {}
            });
            d ? kb.event.trigger(e, null, b) : kb.event.dispatch.call(b, e), e.isDefaultPrevented() && c.preventDefault()
        }
    }, kb.removeEvent = Y.removeEventListener ? function(a, b, c) {
        a.removeEventListener && a.removeEventListener(b, c, !1)
    } : function(a, b, c) {
        var d = "on" + b;
        a.detachEvent && (typeof a[d] === W && (a[d] = null), a.detachEvent(d, c))
    }, kb.Event = function(a, b) {
        return this instanceof kb.Event ? (a && a.type ? (this.originalEvent = a, this.type = a.type, this.isDefaultPrevented = a.defaultPrevented || a.returnValue===!1 || a.getPreventDefault && a.getPreventDefault() ? i : j) : this.type = a, b && kb.extend(this, b), this.timeStamp = a && a.timeStamp || kb.now(), this[kb.expando]=!0, void 0) : new kb.Event(a, b)
    }, kb.Event.prototype = {
        isDefaultPrevented: j,
        isPropagationStopped: j,
        isImmediatePropagationStopped: j,
        preventDefault: function() {
            var a = this.originalEvent;
            this.isDefaultPrevented = i, a && (a.preventDefault ? a.preventDefault() : a.returnValue=!1)
        },
        stopPropagation: function() {
            var a = this.originalEvent;
            this.isPropagationStopped = i, a && (a.stopPropagation && a.stopPropagation(), a.cancelBubble=!0)
        },
        stopImmediatePropagation: function() {
            this.isImmediatePropagationStopped = i, this.stopPropagation()
        }
    }, kb.each({
        mouseenter: "mouseover",
        mouseleave: "mouseout"
    }, function(a, b) {
        kb.event.special[a] = {
            delegateType: b,
            bindType: b,
            handle: function(a) {
                var c, d = this, e = a.relatedTarget, f = a.handleObj;
                return (!e || e !== d&&!kb.contains(d, e)) && (a.type = f.origType, c = f.handler.apply(this, arguments), a.type = b), c
            }
        }
    }), kb.support.submitBubbles || (kb.event.special.submit = {
        setup: function() {
            return kb.nodeName(this, "form")?!1 : void kb.event.add(this, "click._submit keypress._submit", function(a) {
                var c = a.target, d = kb.nodeName(c, "input") || kb.nodeName(c, "button") ? c.form: b;
                d&&!kb._data(d, "submitBubbles") && (kb.event.add(d, "submit._submit", function(a) {
                    a._submit_bubble=!0
                }), kb._data(d, "submitBubbles", !0))
            })
        },
        postDispatch: function(a) {
            a._submit_bubble && (delete a._submit_bubble, this.parentNode&&!a.isTrigger && kb.event.simulate("submit", this.parentNode, a, !0))
        },
        teardown: function() {
            return kb.nodeName(this, "form")?!1 : void kb.event.remove(this, "._submit")
        }
    }), kb.support.changeBubbles || (kb.event.special.change = {
        setup: function() {
            return Lb.test(this.nodeName) ? (("checkbox" === this.type || "radio" === this.type) && (kb.event.add(this, "propertychange._change", function(a) {
                "checked" === a.originalEvent.propertyName && (this._just_changed=!0)
            }), kb.event.add(this, "click._change", function(a) {
                this._just_changed&&!a.isTrigger && (this._just_changed=!1), kb.event.simulate("change", this, a, !0)
            })), !1) : void kb.event.add(this, "beforeactivate._change", function(a) {
                var b = a.target;
                Lb.test(b.nodeName)&&!kb._data(b, "changeBubbles") && (kb.event.add(b, "change._change", function(a) {
                    this.parentNode&&!a.isSimulated&&!a.isTrigger && kb.event.simulate("change", this.parentNode, a, !0)
                }), kb._data(b, "changeBubbles", !0))
            })
        },
        handle: function(a) {
            var b = a.target;
            return this !== b || a.isSimulated || a.isTrigger || "radio" !== b.type && "checkbox" !== b.type ? a.handleObj.handler.apply(this, arguments) : void 0
        },
        teardown: function() {
            return kb.event.remove(this, "._change"), !Lb.test(this.nodeName)
        }
    }), kb.support.focusinBubbles || kb.each({
        focus: "focusin",
        blur: "focusout"
    }, function(a, b) {
        var c = 0, d = function(a) {
            kb.event.simulate(b, a.target, kb.event.fix(a), !0)
        };
        kb.event.special[b] = {
            setup: function() {
                0 === c++&&Y.addEventListener(a, d, !0)
            },
            teardown: function() {
                0===--c && Y.removeEventListener(a, d, !0)
            }
        }
    }), kb.fn.extend({
        on: function(a, c, d, e, f) {
            var g, h;
            if ("object" == typeof a) {
                "string" != typeof c && (d = d || c, c = b);
                for (g in a)
                    this.on(g, c, d, a[g], f);
                return this
            }
            if (null == d && null == e ? (e = c, d = c = b) : null == e && ("string" == typeof c ? (e = d, d = b) : (e = d, d = c, c = b)), e===!1)
                e = j;
            else if (!e)
                return this;
            return 1 === f && (h = e, e = function(a) {
                return kb().off(a), h.apply(this, arguments)
            }, e.guid = h.guid || (h.guid = kb.guid++)), this.each(function() {
                kb.event.add(this, a, e, d, c)
            })
        },
        one: function(a, b, c, d) {
            return this.on(a, b, c, d, 1)
        },
        off: function(a, c, d) {
            var e, f;
            if (a && a.preventDefault && a.handleObj)
                return e = a.handleObj, kb(a.delegateTarget).off(e.namespace ? e.origType + "." + e.namespace : e.origType, e.selector, e.handler), this;
            if ("object" == typeof a) {
                for (f in a)
                    this.off(f, c, a[f]);
                return this
            }
            return (c===!1 || "function" == typeof c) && (d = c, c = b), d===!1 && (d = j), this.each(function() {
                kb.event.remove(this, a, d, c)
            })
        },
        trigger: function(a, b) {
            return this.each(function() {
                kb.event.trigger(a, b, this)
            })
        },
        triggerHandler: function(a, b) {
            var c = this[0];
            return c ? kb.event.trigger(a, b, c, !0) : void 0
        }
    });
    var Qb = /^.[^:#\[\.,]*$/, Rb = /^(?:parents|prev(?:Until|All))/, Sb = kb.expr.match.needsContext, Tb = {
        children: !0,
        contents: !0,
        next: !0,
        prev: !0
    };
    kb.fn.extend({
        find: function(a) {
            var b, c = [], d = this, e = d.length;
            if ("string" != typeof a)
                return this.pushStack(kb(a).filter(function() {
                    for (b = 0; e > b; b++)
                        if (kb.contains(d[b], this))
                            return !0
                        }));
            for (b = 0; e > b; b++)
                kb.find(a, d[b], c);
            return c = this.pushStack(e > 1 ? kb.unique(c) : c), c.selector = this.selector ? this.selector + " " + a : a, c
        },
        has: function(a) {
            var b, c = kb(a, this), d = c.length;
            return this.filter(function() {
                for (b = 0; d > b; b++)
                    if (kb.contains(this, c[b]))
                        return !0
            })
        },
        not: function(a) {
            return this.pushStack(m(this, a || [], !0))
        },
        filter: function(a) {
            return this.pushStack(m(this, a || [], !1))
        },
        is: function(a) {
            return !!m(this, "string" == typeof a && Sb.test(a) ? kb(a) : a || [], !1).length
        },
        closest: function(a, b) {
            for (var c, d = 0, e = this.length, f = [], g = Sb.test(a) || "string" != typeof a ? kb(a, b || this.context) : 0; e > d; d++)
                for (c = this[d]; c && c !== b; c = c.parentNode)
                    if (c.nodeType < 11 && (g ? g.index(c)>-1 : 1 === c.nodeType && kb.find.matchesSelector(c, a))) {
                        c = f.push(c);
                        break
                    }
            return this.pushStack(f.length > 1 ? kb.unique(f) : f)
        },
        index: function(a) {
            return a ? "string" == typeof a ? kb.inArray(this[0], kb(a)) : kb.inArray(a.jquery ? a[0] : a, this) : this[0] && this[0].parentNode ? this.first().prevAll().length : - 1
        },
        add: function(a, b) {
            var c = "string" == typeof a ? kb(a, b): kb.makeArray(a && a.nodeType ? [a] : a), d = kb.merge(this.get(), c);
            return this.pushStack(kb.unique(d))
        },
        addBack: function(a) {
            return this.add(null == a ? this.prevObject : this.prevObject.filter(a))
        }
    }), kb.each({
        parent: function(a) {
            var b = a.parentNode;
            return b && 11 !== b.nodeType ? b : null
        },
        parents: function(a) {
            return kb.dir(a, "parentNode")
        },
        parentsUntil: function(a, b, c) {
            return kb.dir(a, "parentNode", c)
        },
        next: function(a) {
            return l(a, "nextSibling")
        },
        prev: function(a) {
            return l(a, "previousSibling")
        },
        nextAll: function(a) {
            return kb.dir(a, "nextSibling")
        },
        prevAll: function(a) {
            return kb.dir(a, "previousSibling")
        },
        nextUntil: function(a, b, c) {
            return kb.dir(a, "nextSibling", c)
        },
        prevUntil: function(a, b, c) {
            return kb.dir(a, "previousSibling", c)
        },
        siblings: function(a) {
            return kb.sibling((a.parentNode || {}).firstChild, a)
        },
        children: function(a) {
            return kb.sibling(a.firstChild)
        },
        contents: function(a) {
            return kb.nodeName(a, "iframe") ? a.contentDocument || a.contentWindow.document : kb.merge([], a.childNodes)
        }
    }, function(a, b) {
        kb.fn[a] = function(c, d) {
            var e = kb.map(this, b, c);
            return "Until" !== a.slice( - 5) && (d = c), d && "string" == typeof d && (e = kb.filter(d, e)), this.length > 1 && (Tb[a] || (e = kb.unique(e)), Rb.test(a) && (e = e.reverse())), this.pushStack(e)
        }
    }), kb.extend({
        filter: function(a, b, c) {
            var d = b[0];
            return c && (a = ":not(" + a + ")"), 1 === b.length && 1 === d.nodeType ? kb.find.matchesSelector(d, a) ? [d] : [] : kb.find.matches(a, kb.grep(b, function(a) {
                return 1 === a.nodeType
            }))
        },
        dir: function(a, c, d) {
            for (var e = [], f = a[c]; f && 9 !== f.nodeType && (d === b || 1 !== f.nodeType ||!kb(f).is(d));)
                1 === f.nodeType && e.push(f), f = f[c];
            return e
        },
        sibling: function(a, b) {
            for (var c = []; a; a = a.nextSibling)
                1 === a.nodeType && a !== b && c.push(a);
            return c
        }
    });
    var Ub = "abbr|article|aside|audio|bdi|canvas|data|datalist|details|figcaption|figure|footer|header|hgroup|mark|meter|nav|output|progress|section|summary|time|video", Vb = / jQuery\d+="(?:null|\d+)"/g, Wb = new RegExp("<(?:" + Ub + ")[\\s/>]", "i"), Xb = /^\s+/, Yb = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\w:]+)[^>]*)\/>/gi, Zb = /<([\w:]+)/, $b = /<tbody/i, _b = /<|&#?\w+;/, ac = /<(?:script|style|link)/i, bc = /^(?:checkbox|radio)$/i, cc = /checked\s*(?:[^=]|=\s*.checked.)/i, dc = /^$|\/(?:java|ecma)script/i, ec = /^true\/(.*)/, fc = /^\s*<!(?:\[CDATA\[|--)|(?:\]\]|--)>\s*$/g, gc = {
        option: [1, "<select multiple='multiple'>", "</select>"],
        legend: [1, "<fieldset>", "</fieldset>"],
        area: [1, "<map>", "</map>"],
        param: [1, "<object>", "</object>"],
        thead: [1, "<table>", "</table>"],
        tr: [2, "<table><tbody>", "</tbody></table>"],
        col: [2, "<table><tbody></tbody><colgroup>", "</colgroup></table>"],
        td: [3, "<table><tbody><tr>", "</tr></tbody></table>"],
        _default: kb.support.htmlSerialize ? [0, "", ""]: [1, "X<div>", "</div>"]
    }, hc = n(Y), ic = hc.appendChild(Y.createElement("div"));
    gc.optgroup = gc.option, gc.tbody = gc.tfoot = gc.colgroup = gc.caption = gc.thead, gc.th = gc.td, kb.fn.extend({
        text: function(a) {
            return kb.access(this, function(a) {
                return a === b ? kb.text(this) : this.empty().append((this[0] && this[0].ownerDocument || Y).createTextNode(a))
            }, null, a, arguments.length)
        },
        append: function() {
            return this.domManip(arguments, function(a) {
                if (1 === this.nodeType || 11 === this.nodeType || 9 === this.nodeType) {
                    var b = o(this, a);
                    b.appendChild(a)
                }
            })
        },
        prepend: function() {
            return this.domManip(arguments, function(a) {
                if (1 === this.nodeType || 11 === this.nodeType || 9 === this.nodeType) {
                    var b = o(this, a);
                    b.insertBefore(a, b.firstChild)
                }
            })
        },
        before: function() {
            return this.domManip(arguments, function(a) {
                this.parentNode && this.parentNode.insertBefore(a, this)
            })
        },
        after: function() {
            return this.domManip(arguments, function(a) {
                this.parentNode && this.parentNode.insertBefore(a, this.nextSibling)
            })
        },
        remove: function(a, b) {
            for (var c, d = a ? kb.filter(a, this) : this, e = 0; null != (c = d[e]); e++)
                !b && 1 === c.nodeType && kb.cleanData(u(c)), c.parentNode && (b && kb.contains(c.ownerDocument, c) && r(u(c, "script")), c.parentNode.removeChild(c));
            return this
        },
        empty: function() {
            for (var a, b = 0; null != (a = this[b]); b++) {
                for (1 === a.nodeType && kb.cleanData(u(a, !1)); a.firstChild;)
                    a.removeChild(a.firstChild);
                a.options && kb.nodeName(a, "select") && (a.options.length = 0)
            }
            return this
        },
        clone: function(a, b) {
            return a = null == a?!1 : a, b = null == b ? a : b, this.map(function() {
                return kb.clone(this, a, b)
            })
        },
        html: function(a) {
            return kb.access(this, function(a) {
                var c = this[0] || {}, d = 0, e = this.length;
                if (a === b)
                    return 1 === c.nodeType ? c.innerHTML.replace(Vb, "") : b;
                if (!("string" != typeof a || ac.test(a) ||!kb.support.htmlSerialize && Wb.test(a) ||!kb.support.leadingWhitespace && Xb.test(a) || gc[(Zb.exec(a) || ["", ""])[1].toLowerCase()])) {
                    a = a.replace(Yb, "<$1></$2>");
                    try {
                        for (; e > d; d++)
                            c = this[d] || {}, 1 === c.nodeType && (kb.cleanData(u(c, !1)), c.innerHTML = a);
                        c = 0
                    } catch (f) {}
                }
                c && this.empty().append(a)
            }, null, a, arguments.length)
        },
        replaceWith: function() {
            var a = kb.map(this, function(a) {
                return [a.nextSibling, a.parentNode]
            }), b = 0;
            return this.domManip(arguments, function(c) {
                var d = a[b++], e = a[b++];
                e && (d && d.parentNode !== e && (d = this.nextSibling), kb(this).remove(), e.insertBefore(c, d))
            }, !0), b ? this : this.remove()
        },
        detach: function(a) {
            return this.remove(a, !0)
        },
        domManip: function(a, b, c) {
            a = db.apply([], a);
            var d, e, f, g, h, i, j = 0, k = this.length, l = this, m = k - 1, n = a[0], o = kb.isFunction(n);
            if (o ||!(1 >= k || "string" != typeof n || kb.support.checkClone) && cc.test(n))
                return this.each(function(d) {
                    var e = l.eq(d);
                    o && (a[0] = n.call(this, d, e.html())), e.domManip(a, b, c)
                });
            if (k && (i = kb.buildFragment(a, this[0].ownerDocument, !1, !c && this), d = i.firstChild, 1 === i.childNodes.length && (i = d), d)) {
                for (g = kb.map(u(i, "script"), p), f = g.length; k > j; j++)
                    e = i, j !== m && (e = kb.clone(e, !0, !0), f && kb.merge(g, u(e, "script"))), b.call(this[j], e, j);
                if (f)
                    for (h = g[g.length - 1].ownerDocument, kb.map(g, q), j = 0; f > j; j++)
                        e = g[j], dc.test(e.type || "")&&!kb._data(e, "globalEval") && kb.contains(h, e) && (e.src ? kb._evalUrl(e.src) : kb.globalEval((e.text || e.textContent || e.innerHTML || "").replace(fc, "")));
                i = d = null
            }
            return this
        }
    }), kb.each({
        appendTo: "append",
        prependTo: "prepend",
        insertBefore: "before",
        insertAfter: "after",
        replaceAll: "replaceWith"
    }, function(a, b) {
        kb.fn[a] = function(a) {
            for (var c, d = 0, e = [], f = kb(a), g = f.length - 1; g >= d; d++)
                c = d === g ? this : this.clone(!0), kb(f[d])[b](c), eb.apply(e, c.get());
            return this.pushStack(e)
        }
    }), kb.extend({
        clone: function(a, b, c) {
            var d, e, f, g, h, i = kb.contains(a.ownerDocument, a);
            if (kb.support.html5Clone || kb.isXMLDoc(a) ||!Wb.test("<" + a.nodeName + ">") ? f = a.cloneNode(!0) : (ic.innerHTML = a.outerHTML, ic.removeChild(f = ic.firstChild)), !(kb.support.noCloneEvent && kb.support.noCloneChecked || 1 !== a.nodeType && 11 !== a.nodeType || kb.isXMLDoc(a)))
                for (d = u(f), h = u(a), g = 0; null != (e = h[g]); ++g)
                    d[g] && t(e, d[g]);
            if (b)
                if (c)
                    for (h = h || u(a), d = d || u(f), g = 0; null != (e = h[g]); g++)
                        s(e, d[g]);
                else 
                    s(a, f);
            return d = u(f, "script"), d.length > 0 && r(d, !i && u(a, "script")), d = h = e = null, f
        },
        buildFragment: function(a, b, c, d) {
            for (var e, f, g, h, i, j, k, l = a.length, m = n(b), o = [], p = 0; l > p; p++)
                if (f = a[p], f || 0 === f)
                    if ("object" === kb.type(f))
                        kb.merge(o, f.nodeType ? [f] : f);
                    else if (_b.test(f)) {
                        for (h = h || m.appendChild(b.createElement("div")), i = (Zb.exec(f) || ["", ""])[1].toLowerCase(), k = gc[i] || gc._default, h.innerHTML = k[1] + f.replace(Yb, "<$1></$2>") + k[2], e = k[0]; e--;)
                            h = h.lastChild;
                            if (!kb.support.leadingWhitespace && Xb.test(f) && o.push(b.createTextNode(Xb.exec(f)[0])), !kb.support.tbody)
                                for (f = "table" !== i || $b.test(f) ? "<table>" !== k[1] || $b.test(f) ? 0 : h : h.firstChild, e = f && f.childNodes.length; e--;)
                                    kb.nodeName(j = f.childNodes[e], "tbody")&&!j.childNodes.length && f.removeChild(j);
                                    for (kb.merge(o, h.childNodes), h.textContent = ""; h.firstChild;)
                                        h.removeChild(h.firstChild);
                                        h = m.lastChild
                    } else 
                        o.push(b.createTextNode(f));
            for (h && m.removeChild(h), kb.support.appendChecked || kb.grep(u(o, "input"), v), p = 0; f = o[p++];)
                if ((!d||-1 === kb.inArray(f, d)) && (g = kb.contains(f.ownerDocument, f), h = u(m.appendChild(f), "script"), g && r(h), c))
                    for (e = 0; f = h[e++];)
                        dc.test(f.type || "") && c.push(f);
            return h = null, m
        },
        cleanData: function(a, b) {
            for (var c, d, e, f, g = 0, h = kb.expando, i = kb.cache, j = kb.support.deleteExpando, k = kb.event.special; null != (c = a[g]); g++)
                if ((b || kb.acceptData(c)) && (e = c[h], f = e && i[e])) {
                    if (f.events)
                        for (d in f.events)
                            k[d] ? kb.event.remove(c, d) : kb.removeEvent(c, d, f.handle);
                            i[e] && (delete i[e], j ? delete c[h] : typeof c.removeAttribute !== W ? c.removeAttribute(h) : c[h] = null, bb.push(e))
                }
        },
        _evalUrl: function(a) {
            return kb.ajax({
                url: a,
                type: "GET",
                dataType: "script",
                async: !1,
                global: !1,
                "throws": !0
            })
        }
    }), kb.fn.extend({
        wrapAll: function(a) {
            if (kb.isFunction(a))
                return this.each(function(b) {
                    kb(this).wrapAll(a.call(this, b))
                });
            if (this[0]) {
                var b = kb(a, this[0].ownerDocument).eq(0).clone(!0);
                this[0].parentNode && b.insertBefore(this[0]), b.map(function() {
                    for (var a = this; a.firstChild && 1 === a.firstChild.nodeType;)
                        a = a.firstChild;
                    return a
                }).append(this)
            }
            return this
        },
        wrapInner: function(a) {
            return this.each(kb.isFunction(a) ? function(b) {
                kb(this).wrapInner(a.call(this, b))
            } : function() {
                var b = kb(this), c = b.contents();
                c.length ? c.wrapAll(a) : b.append(a)
            })
        },
        wrap: function(a) {
            var b = kb.isFunction(a);
            return this.each(function(c) {
                kb(this).wrapAll(b ? a.call(this, c) : a)
            })
        },
        unwrap: function() {
            return this.parent().each(function() {
                kb.nodeName(this, "body") || kb(this).replaceWith(this.childNodes)
            }).end()
        }
    });
    var jc, kc, lc, mc = /alpha\([^)]*\)/i, nc = /opacity\s*=\s*([^)]*)/, oc = /^(top|right|bottom|left)$/, pc = /^(none|table(?!-c[ea]).+)/, qc = /^margin/, rc = new RegExp("^(" + lb + ")(.*)$", "i"), sc = new RegExp("^(" + lb + ")(?!px)[a-z%]+$", "i"), tc = new RegExp("^([+-])=(" + lb + ")", "i"), uc = {
        BODY: "block"
    }, vc = {
        position: "absolute",
        visibility: "hidden",
        display: "block"
    }, wc = {
        letterSpacing: 0,
        fontWeight: 400
    }, xc = ["Top", "Right", "Bottom", "Left"], yc = ["Webkit", "O", "Moz", "ms"];
    kb.fn.extend({
        css: function(a, c) {
            return kb.access(this, function(a, c, d) {
                var e, f, g = {}, h = 0;
                if (kb.isArray(c)) {
                    for (f = kc(a), e = c.length; e > h; h++)
                        g[c[h]] = kb.css(a, c[h], !1, f);
                    return g
                }
                return d !== b ? kb.style(a, c, d) : kb.css(a, c)
            }, a, c, arguments.length > 1)
        },
        show: function() {
            return y(this, !0)
        },
        hide: function() {
            return y(this)
        },
        toggle: function(a) {
            return "boolean" == typeof a ? a ? this.show() : this.hide() : this.each(function() {
                x(this) ? kb(this).show() : kb(this).hide()
            })
        }
    }), kb.extend({
        cssHooks: {
            opacity: {
                get: function(a, b) {
                    if (b) {
                        var c = lc(a, "opacity");
                        return "" === c ? "1" : c
                    }
                }
            }
        },
        cssNumber: {
            columnCount: !0,
            fillOpacity: !0,
            fontWeight: !0,
            lineHeight: !0,
            opacity: !0,
            order: !0,
            orphans: !0,
            widows: !0,
            zIndex: !0,
            zoom: !0
        },
        cssProps: {
            "float": kb.support.cssFloat ? "cssFloat": "styleFloat"
        },
        style: function(a, c, d, e) {
            if (a && 3 !== a.nodeType && 8 !== a.nodeType && a.style) {
                var f, g, h, i = kb.camelCase(c), j = a.style;
                if (c = kb.cssProps[i] || (kb.cssProps[i] = w(j, i)), h = kb.cssHooks[c] || kb.cssHooks[i], d === b)
                    return h && "get"in h && (f = h.get(a, !1, e)) !== b ? f : j[c];
                if (g = typeof d, "string" === g && (f = tc.exec(d)) && (d = (f[1] + 1) * f[2] + parseFloat(kb.css(a, c)), g = "number"), !(null == d || "number" === g && isNaN(d) || ("number" === g&&!kb.cssNumber[i] && (d += "px"), !kb.support.clearCloneStyle && "" === d && 0 === c.indexOf("background") && (j[c] = "inherit"), h && "set"in h && (d = h.set(a, d, e)) === b)))
                    try {
                        j[c] = d
                } catch (k) {}
            }
        },
        css: function(a, c, d, e) {
            var f, g, h, i = kb.camelCase(c);
            return c = kb.cssProps[i] || (kb.cssProps[i] = w(a.style, i)), h = kb.cssHooks[c] || kb.cssHooks[i], h && "get"in h && (g = h.get(a, !0, d)), g === b && (g = lc(a, c, e)), "normal" === g && c in wc && (g = wc[c]), "" === d || d ? (f = parseFloat(g), d===!0 || kb.isNumeric(f) ? f || 0 : g) : g
        }
    }), a.getComputedStyle ? (kc = function(b) {
        return a.getComputedStyle(b, null)
    }, lc = function(a, c, d) {
        var e, f, g, h = d || kc(a), i = h ? h.getPropertyValue(c) || h[c]: b, j = a.style;
        return h && ("" === i&&!kb.contains(a.ownerDocument, a) && (i = kb.style(a, c)), sc.test(i) && qc.test(c) && (e = j.width, f = j.minWidth, g = j.maxWidth, j.minWidth = j.maxWidth = j.width = i, i = h.width, j.width = e, j.minWidth = f, j.maxWidth = g)), i
    }) : Y.documentElement.currentStyle && (kc = function(a) {
        return a.currentStyle
    }, lc = function(a, c, d) {
        var e, f, g, h = d || kc(a), i = h ? h[c]: b, j = a.style;
        return null == i && j && j[c] && (i = j[c]), sc.test(i)&&!oc.test(c) && (e = j.left, f = a.runtimeStyle, g = f && f.left, g && (f.left = a.currentStyle.left), j.left = "fontSize" === c ? "1em" : i, i = j.pixelLeft + "px", j.left = e, g && (f.left = g)), "" === i ? "auto" : i
    }), kb.each(["height", "width"], function(a, b) {
        kb.cssHooks[b] = {
            get: function(a, c, d) {
                return c ? 0 === a.offsetWidth && pc.test(kb.css(a, "display")) ? kb.swap(a, vc, function() {
                    return B(a, b, d)
                }) : B(a, b, d) : void 0
            },
            set: function(a, c, d) {
                var e = d && kc(a);
                return z(a, c, d ? A(a, b, d, kb.support.boxSizing && "border-box" === kb.css(a, "boxSizing", !1, e), e) : 0)
            }
        }
    }), kb.support.opacity || (kb.cssHooks.opacity = {
        get: function(a, b) {
            return nc.test((b && a.currentStyle ? a.currentStyle.filter : a.style.filter) || "") ? .01 * parseFloat(RegExp.$1) + "" : b ? "1" : ""
        },
        set: function(a, b) {
            var c = a.style, d = a.currentStyle, e = kb.isNumeric(b) ? "alpha(opacity=" + 100 * b + ")": "", f = d && d.filter || c.filter || "";
            c.zoom = 1, (b >= 1 || "" === b) && "" === kb.trim(f.replace(mc, "")) && c.removeAttribute && (c.removeAttribute("filter"), "" === b || d&&!d.filter) || (c.filter = mc.test(f) ? f.replace(mc, e) : f + " " + e)
        }
    }), kb(function() {
        kb.support.reliableMarginRight || (kb.cssHooks.marginRight = {
            get: function(a, b) {
                return b ? kb.swap(a, {
                    display: "inline-block"
                }, lc, [a, "marginRight"]) : void 0
            }
        }), !kb.support.pixelPosition && kb.fn.position && kb.each(["top", "left"], function(a, b) {
            kb.cssHooks[b] = {
                get: function(a, c) {
                    return c ? (c = lc(a, b), sc.test(c) ? kb(a).position()[b] + "px" : c) : void 0
                }
            }
        })
    }), kb.expr && kb.expr.filters && (kb.expr.filters.hidden = function(a) {
        return a.offsetWidth <= 0 && a.offsetHeight <= 0 ||!kb.support.reliableHiddenOffsets && "none" === (a.style && a.style.display || kb.css(a, "display"))
    }, kb.expr.filters.visible = function(a) {
        return !kb.expr.filters.hidden(a)
    }), kb.each({
        margin: "",
        padding: "",
        border: "Width"
    }, function(a, b) {
        kb.cssHooks[a + b] = {
            expand: function(c) {
                for (var d = 0, e = {}, f = "string" == typeof c ? c.split(" ") : [c]; 4 > d; d++)
                    e[a + xc[d] + b] = f[d] || f[d - 2] || f[0];
                return e
            }
        }, qc.test(a) || (kb.cssHooks[a + b].set = z)
    });
    var zc = /%20/g, Ac = /\[\]$/, Bc = /\r?\n/g, Cc = /^(?:submit|button|image|reset|file)$/i, Dc = /^(?:input|select|textarea|keygen)/i;
    kb.fn.extend({
        serialize: function() {
            return kb.param(this.serializeArray())
        },
        serializeArray: function() {
            return this.map(function() {
                var a = kb.prop(this, "elements");
                return a ? kb.makeArray(a) : this
            }).filter(function() {
                var a = this.type;
                return this.name&&!kb(this).is(":disabled") && Dc.test(this.nodeName)&&!Cc.test(a) && (this.checked ||!bc.test(a))
            }).map(function(a, b) {
                var c = kb(this).val();
                return null == c ? null : kb.isArray(c) ? kb.map(c, function(a) {
                    return {
                        name: b.name,
                        value: a.replace(Bc, "\r\n")
                    }
                }) : {
                    name: b.name,
                    value: c.replace(Bc, "\r\n")
                }
            }).get()
        }
    }), kb.param = function(a, c) {
        var d, e = [], f = function(a, b) {
            b = kb.isFunction(b) ? b() : null == b ? "" : b, e[e.length] = encodeURIComponent(a) + "=" + encodeURIComponent(b)
        };
        if (c === b && (c = kb.ajaxSettings && kb.ajaxSettings.traditional), kb.isArray(a) || a.jquery&&!kb.isPlainObject(a))
            kb.each(a, function() {
                f(this.name, this.value)
            });
        else 
            for (d in a)
                E(d, a[d], c, f);
        return e.join("&").replace(zc, "+")
    }, kb.each("blur focus focusin focusout load resize scroll unload click dblclick mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave change select submit keydown keypress keyup error contextmenu".split(" "), function(a, b) {
        kb.fn[b] = function(a, c) {
            return arguments.length > 0 ? this.on(b, null, a, c) : this.trigger(b)
        }
    }), kb.fn.extend({
        hover: function(a, b) {
            return this.mouseenter(a).mouseleave(b || a)
        },
        bind: function(a, b, c) {
            return this.on(a, null, b, c)
        },
        unbind: function(a, b) {
            return this.off(a, null, b)
        },
        delegate: function(a, b, c, d) {
            return this.on(b, a, c, d)
        },
        undelegate: function(a, b, c) {
            return 1 === arguments.length ? this.off(a, "**") : this.off(b, a || "**", c)
        }
    });
    var Ec, Fc, Gc = kb.now(), Hc = /\?/, Ic = /#.*$/, Jc = /([?&])_=[^&]*/, Kc = /^(.*?):[ \t]*([^\r\n]*)\r?$/gm, Lc = /^(?:about|app|app-storage|.+-extension|file|res|widget):$/, Mc = /^(?:GET|HEAD)$/, Nc = /^\/\//, Oc = /^([\w.+-]+:)(?:\/\/([^\/?#:]*)(?::(\d+)|)|)/, Pc = kb.fn.load, Qc = {}, Rc = {}, Sc = "*/".concat("*");
    try {
        Fc = X.href
    } catch (Tc) {
        Fc = Y.createElement("a"), Fc.href = "", Fc = Fc.href
    }
    Ec = Oc.exec(Fc.toLowerCase()) || [], kb.fn.load = function(a, c, d) {
        if ("string" != typeof a && Pc)
            return Pc.apply(this, arguments);
        var e, f, g, h = this, i = a.indexOf(" ");
        return i >= 0 && (e = a.slice(i, a.length), a = a.slice(0, i)), kb.isFunction(c) ? (d = c, c = b) : c && "object" == typeof c && (g = "POST"), h.length > 0 && kb.ajax({
            url: a,
            type: g,
            dataType: "html",
            data: c
        }).done(function(a) {
            f = arguments, h.html(e ? kb("<div>").append(kb.parseHTML(a)).find(e) : a)
        }).complete(d && function(a, b) {
            h.each(d, f || [a.responseText, b, a])
        }), this
    }, kb.each(["ajaxStart", "ajaxStop", "ajaxComplete", "ajaxError", "ajaxSuccess", "ajaxSend"], function(a, b) {
        kb.fn[b] = function(a) {
            return this.on(b, a)
        }
    }), kb.extend({
        active: 0,
        lastModified: {},
        etag: {},
        ajaxSettings: {
            url: Fc,
            type: "GET",
            isLocal: Lc.test(Ec[1]),
            global: !0,
            processData: !0,
            async: !0,
            contentType: "application/x-www-form-urlencoded; charset=UTF-8",
            accepts: {
                "*": Sc,
                text: "text/plain",
                html: "text/html",
                xml: "application/xml, text/xml",
                json: "application/json, text/javascript"
            },
            contents: {
                xml: /xml/,
                html: /html/,
                json: /json/
            },
            responseFields: {
                xml: "responseXML",
                text: "responseText",
                json: "responseJSON"
            },
            converters: {
                "* text": String,
                "text html": !0,
                "text json": kb.parseJSON,
                "text xml": kb.parseXML
            },
            flatOptions: {
                url: !0,
                context: !0
            }
        },
        ajaxSetup: function(a, b) {
            return b ? H(H(a, kb.ajaxSettings), b) : H(kb.ajaxSettings, a)
        },
        ajaxPrefilter: F(Qc),
        ajaxTransport: F(Rc),
        ajax: function(a, c) {
            function d(a, c, d, e) {
                var f, l, s, t, v, x = c;
                2 !== u && (u = 2, i && clearTimeout(i), k = b, h = e || "", w.readyState = a > 0 ? 4 : 0, f = a >= 200 && 300 > a || 304 === a, d && (t = I(m, w, d)), t = J(m, t, w, f), f ? (m.ifModified && (v = w.getResponseHeader("Last-Modified"), v && (kb.lastModified[g] = v), v = w.getResponseHeader("etag"), v && (kb.etag[g] = v)), 204 === a || "HEAD" === m.type ? x = "nocontent" : 304 === a ? x = "notmodified" : (x = t.state, l = t.data, s = t.error, f=!s)) : (s = x, (a ||!x) && (x = "error", 0 > a && (a = 0))), w.status = a, w.statusText = (c || x) + "", f ? p.resolveWith(n, [l, x, w]) : p.rejectWith(n, [w, x, s]), w.statusCode(r), r = b, j && o.trigger(f ? "ajaxSuccess" : "ajaxError", [w, m, f ? l: s]), q.fireWith(n, [w, x]), j && (o.trigger("ajaxComplete", [w, m]), --kb.active || kb.event.trigger("ajaxStop")))
            }
            "object" == typeof a && (c = a, a = b), c = c || {};
            var e, f, g, h, i, j, k, l, m = kb.ajaxSetup({}, c), n = m.context || m, o = m.context && (n.nodeType || n.jquery) ? kb(n): kb.event, p = kb.Deferred(), q = kb.Callbacks("once memory"), r = m.statusCode || {}, s = {}, t = {}, u = 0, v = "canceled", w = {
                readyState: 0,
                getResponseHeader: function(a) {
                    var b;
                    if (2 === u) {
                        if (!l)
                            for (l = {}; b = Kc.exec(h);)
                                l[b[1].toLowerCase()] = b[2];
                        b = l[a.toLowerCase()]
                    }
                    return null == b ? null : b
                },
                getAllResponseHeaders: function() {
                    return 2 === u ? h : null
                },
                setRequestHeader: function(a, b) {
                    var c = a.toLowerCase();
                    return u || (a = t[c] = t[c] || a, s[a] = b), this
                },
                overrideMimeType: function(a) {
                    return u || (m.mimeType = a), this
                },
                statusCode: function(a) {
                    var b;
                    if (a)
                        if (2 > u)
                            for (b in a)
                                r[b] = [r[b], a[b]];
                        else 
                            w.always(a[w.status]);
                    return this
                },
                abort: function(a) {
                    var b = a || v;
                    return k && k.abort(b), d(0, b), this
                }
            };
            if (p.promise(w).complete = q.add, w.success = w.done, w.error = w.fail, m.url = ((a || m.url || Fc) + "").replace(Ic, "").replace(Nc, Ec[1] + "//"), m.type = c.method || c.type || m.method || m.type, m.dataTypes = kb.trim(m.dataType || "*").toLowerCase().match(mb) || [""], null == m.crossDomain && (e = Oc.exec(m.url.toLowerCase()), m.crossDomain=!(!e || e[1] === Ec[1] && e[2] === Ec[2] && (e[3] || ("http:" === e[1] ? "80" : "443")) === (Ec[3] || ("http:" === Ec[1] ? "80" : "443")))), m.data && m.processData && "string" != typeof m.data && (m.data = kb.param(m.data, m.traditional)), G(Qc, m, c, w), 2 === u)
                return w;
            j = m.global, j && 0 === kb.active++&&kb.event.trigger("ajaxStart"), m.type = m.type.toUpperCase(), m.hasContent=!Mc.test(m.type), g = m.url, m.hasContent || (m.data && (g = m.url += (Hc.test(g) ? "&" : "?") + m.data, delete m.data), m.cache===!1 && (m.url = Jc.test(g) ? g.replace(Jc, "$1_=" + Gc++) : g + (Hc.test(g) ? "&" : "?") + "_=" + Gc++)), m.ifModified && (kb.lastModified[g] && w.setRequestHeader("If-Modified-Since", kb.lastModified[g]), kb.etag[g] && w.setRequestHeader("If-None-Match", kb.etag[g])), (m.data && m.hasContent && m.contentType!==!1 || c.contentType) && w.setRequestHeader("Content-Type", m.contentType), w.setRequestHeader("Accept", m.dataTypes[0] && m.accepts[m.dataTypes[0]] ? m.accepts[m.dataTypes[0]] + ("*" !== m.dataTypes[0] ? ", " + Sc + "; q=0.01" : "") : m.accepts["*"]);
            for (f in m.headers)
                w.setRequestHeader(f, m.headers[f]);
            if (!m.beforeSend || m.beforeSend.call(n, w, m)!==!1 && 2 !== u) {
                v = "abort";
                for (f in{
                    success: 1,
                    error: 1,
                    complete: 1
                })
                    w[f](m[f]);
                if (k = G(Rc, m, c, w)) {
                    w.readyState = 1, j && o.trigger("ajaxSend", [w, m]), m.async && m.timeout > 0 && (i = setTimeout(function() {
                        w.abort("timeout")
                    }, m.timeout));
                    try {
                        u = 1, k.send(s, d)
                    } catch (x) {
                        if (!(2 > u))
                            throw x;
                        d( - 1, x)
                    }
                } else 
                    d( - 1, "No Transport");
                return w
            }
            return w.abort()
        },
        getJSON: function(a, b, c) {
            return kb.get(a, b, c, "json")
        },
        getScript: function(a, c) {
            return kb.get(a, b, c, "script")
        }
    }), kb.each(["get", "post"], function(a, c) {
        kb[c] = function(a, d, e, f) {
            return kb.isFunction(d) && (f = f || e, e = d, d = b), kb.ajax({
                url: a,
                type: c,
                dataType: f,
                data: d,
                success: e
            })
        }
    }), kb.ajaxSetup({
        accepts: {
            script: "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript"
        },
        contents: {
            script: /(?:java|ecma)script/
        },
        converters: {
            "text script": function(a) {
                return kb.globalEval(a), a
            }
        }
    }), kb.ajaxPrefilter("script", function(a) {
        a.cache === b && (a.cache=!1), a.crossDomain && (a.type = "GET", a.global=!1)
    }), kb.ajaxTransport("script", function(a) {
        if (a.crossDomain) {
            var c, d = Y.head || kb("head")[0] || Y.documentElement;
            return {
                send: function(b, e) {
                    c = Y.createElement("script"), c.async=!0, a.scriptCharset && (c.charset = a.scriptCharset), c.src = a.url, c.onload = c.onreadystatechange = function(a, b) {
                        (b ||!c.readyState || /loaded|complete/.test(c.readyState)) && (c.onload = c.onreadystatechange = null, c.parentNode && c.parentNode.removeChild(c), c = null, b || e(200, "success"))
                    }, d.insertBefore(c, d.firstChild)
                },
                abort: function() {
                    c && c.onload(b, !0)
                }
            }
        }
    });
    var Uc = [], Vc = /(=)\?(?=&|$)|\?\?/;
    kb.ajaxSetup({
        jsonp: "callback",
        jsonpCallback: function() {
            var a = Uc.pop() || kb.expando + "_" + Gc++;
            return this[a]=!0, a
        }
    }), kb.ajaxPrefilter("json jsonp", function(c, d, e) {
        var f, g, h, i = c.jsonp!==!1 && (Vc.test(c.url) ? "url" : "string" == typeof c.data&&!(c.contentType || "").indexOf("application/x-www-form-urlencoded") && Vc.test(c.data) && "data");
        return i || "jsonp" === c.dataTypes[0] ? (f = c.jsonpCallback = kb.isFunction(c.jsonpCallback) ? c.jsonpCallback() : c.jsonpCallback, i ? c[i] = c[i].replace(Vc, "$1" + f) : c.jsonp!==!1 && (c.url += (Hc.test(c.url) ? "&" : "?") + c.jsonp + "=" + f), c.converters["script json"] = function() {
            return h || kb.error(f + " was not called"), h[0]
        }, c.dataTypes[0] = "json", g = a[f], a[f] = function() {
            h = arguments
        }, e.always(function() {
            a[f] = g, c[f] && (c.jsonpCallback = d.jsonpCallback, Uc.push(f)), h && kb.isFunction(g) && g(h[0]), h = g = b
        }), "script") : void 0
    });
    var Wc, Xc, Yc = 0, Zc = a.ActiveXObject && function() {
        var a;
        for (a in Wc)
            Wc[a](b, !0)
    };
    kb.ajaxSettings.xhr = a.ActiveXObject ? function() {
        return !this.isLocal && K() || L()
    } : K, Xc = kb.ajaxSettings.xhr(), kb.support.cors=!!Xc && "withCredentials"in Xc, Xc = kb.support.ajax=!!Xc, Xc && kb.ajaxTransport(function(c) {
        if (!c.crossDomain || kb.support.cors) {
            var d;
            return {
                send: function(e, f) {
                    var g, h, i = c.xhr();
                    if (c.username ? i.open(c.type, c.url, c.async, c.username, c.password) : i.open(c.type, c.url, c.async), c.xhrFields)
                        for (h in c.xhrFields)
                            i[h] = c.xhrFields[h];
                    c.mimeType && i.overrideMimeType && i.overrideMimeType(c.mimeType), !c.crossDomain&&!e["X-Requested-With"] && (e["X-Requested-With"] = "XMLHttpRequest");
                    try {
                        for (h in e)
                            i.setRequestHeader(h, e[h])
                    } catch (j) {}
                    i.send(c.hasContent && c.data || null), d = function(a, e) {
                        var h, j, k, l;
                        try {
                            if (d && (e || 4 === i.readyState))
                                if (d = b, g && (i.onreadystatechange = kb.noop, Zc && delete Wc[g]), e)
                                    4 !== i.readyState && i.abort();
                                else {
                                    l = {}, h = i.status, j = i.getAllResponseHeaders(), "string" == typeof i.responseText && (l.text = i.responseText);
                                    try {
                                        k = i.statusText
                                    } catch (m) {
                                        k = ""
                                    }
                                    h ||!c.isLocal || c.crossDomain ? 1223 === h && (h = 204) : h = l.text ? 200 : 404
                                }
                        } catch (n) {
                            e || f( - 1, n)
                        }
                        l && f(h, k, l, j)
                    }, c.async ? 4 === i.readyState ? setTimeout(d) : (g=++Yc, Zc && (Wc || (Wc = {}, kb(a).unload(Zc)), Wc[g] = d), i.onreadystatechange = d) : d()
                },
                abort: function() {
                    d && d(b, !0)
                }
            }
        }
    });
    var $c, _c, ad = /^(?:toggle|show|hide)$/, bd = new RegExp("^(?:([+-])=|)(" + lb + ")([a-z%]*)$", "i"), cd = /queueHooks$/, dd = [Q], ed = {
        "*": [function(a, b) {
            var c = this.createTween(a, b), d = c.cur(), e = bd.exec(b), f = e && e[3] || (kb.cssNumber[a] ? "" : "px"), g = (kb.cssNumber[a] || "px" !== f&&+d) && bd.exec(kb.css(c.elem, a)), h = 1, i = 20;
            if (g && g[3] !== f) {
                f = f || g[3], e = e || [], g =+ d || 1;
                do 
                    h = h || ".5", g/=h, kb.style(c.elem, a, g + f);
                while (h !== (h = c.cur() / d) && 1 !== h&&--i)
                }
            return e && (g = c.start =+ g||+d || 0, c.unit = f, c.end = e[1] ? g + (e[1] + 1) * e[2] : + e[2]), c
        }
        ]
    };
    kb.Animation = kb.extend(O, {
        tweener: function(a, b) {
            kb.isFunction(a) ? (b = a, a = ["*"]): a = a.split(" ");
            for (var c,
            d = 0,
            e = a.length;
            e > d;
            d++)c = a[d],
            ed[c] = ed[c] || [],
            ed[c].unshift(b)
        }, prefilter : function(a, b) {
            b ? dd.unshift(a) : dd.push(a)
        }
    }), kb.Tween = R, R.prototype = {
        constructor: R,
        init: function(a, b, c, d, e, f) {
            this.elem = a, this.prop = c, this.easing = e || "swing", this.options = b, this.start = this.now = this.cur(), this.end = d, this.unit = f || (kb.cssNumber[c] ? "" : "px")
        },
        cur: function() {
            var a = R.propHooks[this.prop];
            return a && a.get ? a.get(this) : R.propHooks._default.get(this)
        },
        run: function(a) {
            var b, c = R.propHooks[this.prop];
            return this.pos = b = this.options.duration ? kb.easing[this.easing](a, this.options.duration * a, 0, 1, this.options.duration) : a, this.now = (this.end - this.start) * b + this.start, this.options.step && this.options.step.call(this.elem, this.now, this), c && c.set ? c.set(this) : R.propHooks._default.set(this), this
        }
    }, R.prototype.init.prototype = R.prototype, R.propHooks = {
        _default: {
            get: function(a) {
                var b;
                return null == a.elem[a.prop] || a.elem.style && null != a.elem.style[a.prop] ? (b = kb.css(a.elem, a.prop, ""), b && "auto" !== b ? b : 0) : a.elem[a.prop]
            },
            set: function(a) {
                kb.fx.step[a.prop] ? kb.fx.step[a.prop](a) : a.elem.style && (null != a.elem.style[kb.cssProps[a.prop]] || kb.cssHooks[a.prop]) ? kb.style(a.elem, a.prop, a.now + a.unit) : a.elem[a.prop] = a.now
            }
        }
    }, R.propHooks.scrollTop = R.propHooks.scrollLeft = {
        set: function(a) {
            a.elem.nodeType && a.elem.parentNode && (a.elem[a.prop] = a.now)
        }
    }, kb.each(["toggle", "show", "hide"], function(a, b) {
        var c = kb.fn[b];
        kb.fn[b] = function(a, d, e) {
            return null == a || "boolean" == typeof a ? c.apply(this, arguments) : this.animate(S(b, !0), a, d, e)
        }
    }), kb.fn.extend({
        fadeTo: function(a, b, c, d) {
            return this.filter(x).css("opacity", 0).show().end().animate({
                opacity: b
            }, a, c, d)
        },
        animate: function(a, b, c, d) {
            var e = kb.isEmptyObject(a), f = kb.speed(b, c, d), g = function() {
                var b = O(this, kb.extend({}, a), f);
                (e || kb._data(this, "finish")) && b.stop(!0)
            };
            return g.finish = g, e || f.queue===!1 ? this.each(g) : this.queue(f.queue, g)
        },
        stop: function(a, c, d) {
            var e = function(a) {
                var b = a.stop;
                delete a.stop, b(d)
            };
            return "string" != typeof a && (d = c, c = a, a = b), c && a!==!1 && this.queue(a || "fx", []), this.each(function() {
                var b=!0, c = null != a && a + "queueHooks", f = kb.timers, g = kb._data(this);
                if (c)
                    g[c] && g[c].stop && e(g[c]);
                else 
                    for (c in g)
                        g[c] && g[c].stop && cd.test(c) && e(g[c]);
                for (c = f.length; c--;)
                    f[c].elem === this && (null == a || f[c].queue === a) && (f[c].anim.stop(d), b=!1, f.splice(c, 1));
                (b ||!d) && kb.dequeue(this, a)
            })
        },
        finish: function(a) {
            return a!==!1 && (a = a || "fx"), this.each(function() {
                var b, c = kb._data(this), d = c[a + "queue"], e = c[a + "queueHooks"], f = kb.timers, g = d ? d.length: 0;
                for (c.finish=!0, kb.queue(this, a, []), e && e.stop && e.stop.call(this, !0), b = f.length; b--;)
                    f[b].elem === this && f[b].queue === a && (f[b].anim.stop(!0), f.splice(b, 1));
                for (b = 0; g > b; b++)
                    d[b] && d[b].finish && d[b].finish.call(this);
                delete c.finish
            })
        }
    }), kb.each({
        slideDown: S("show"),
        slideUp: S("hide"),
        slideToggle: S("toggle"),
        fadeIn: {
            opacity: "show"
        },
        fadeOut: {
            opacity: "hide"
        },
        fadeToggle: {
            opacity: "toggle"
        }
    }, function(a, b) {
        kb.fn[a] = function(a, c, d) {
            return this.animate(b, a, c, d)
        }
    }), kb.speed = function(a, b, c) {
        var d = a && "object" == typeof a ? kb.extend({}, a): {
            complete: c ||!c && b || kb.isFunction(a) && a,
            duration: a,
            easing: c && b || b&&!kb.isFunction(b) && b
        };
        return d.duration = kb.fx.off ? 0 : "number" == typeof d.duration ? d.duration : d.duration in kb.fx.speeds ? kb.fx.speeds[d.duration] : kb.fx.speeds._default, (null == d.queue || d.queue===!0) && (d.queue = "fx"), d.old = d.complete, d.complete = function() {
            kb.isFunction(d.old) && d.old.call(this), d.queue && kb.dequeue(this, d.queue)
        }, d
    }, kb.easing = {
        linear: function(a) {
            return a
        },
        swing: function(a) {
            return .5 - Math.cos(a * Math.PI) / 2
        }
    }, kb.timers = [], kb.fx = R.prototype.init, kb.fx.tick = function() {
        var a, c = kb.timers, d = 0;
        for ($c = kb.now(); d < c.length; d++)
            a = c[d], !a() && c[d] === a && c.splice(d--, 1);
        c.length || kb.fx.stop(), $c = b
    }, kb.fx.timer = function(a) {
        a() && kb.timers.push(a) && kb.fx.start()
    }, kb.fx.interval = 13, kb.fx.start = function() {
        _c || (_c = setInterval(kb.fx.tick, kb.fx.interval))
    }, kb.fx.stop = function() {
        clearInterval(_c), _c = null
    }, kb.fx.speeds = {
        slow: 600,
        fast: 200,
        _default: 400
    }, kb.fx.step = {}, kb.expr && kb.expr.filters && (kb.expr.filters.animated = function(a) {
        return kb.grep(kb.timers, function(b) {
            return a === b.elem
        }).length
    }), kb.fn.offset = function(a) {
        if (arguments.length)
            return a === b ? this : this.each(function(b) {
                kb.offset.setOffset(this, a, b)
            });
        var c, d, e = {
            top: 0,
            left: 0
        }, f = this[0], g = f && f.ownerDocument;
        if (g)
            return c = g.documentElement, kb.contains(c, f) ? (typeof f.getBoundingClientRect !== W && (e = f.getBoundingClientRect()), d = T(g), {
                top: e.top + (d.pageYOffset || c.scrollTop) - (c.clientTop || 0),
                left: e.left + (d.pageXOffset || c.scrollLeft) - (c.clientLeft || 0)
            }) : e
    }, kb.offset = {
        setOffset: function(a, b, c) {
            var d = kb.css(a, "position");
            "static" === d && (a.style.position = "relative");
            var e, f, g = kb(a), h = g.offset(), i = kb.css(a, "top"), j = kb.css(a, "left"), k = ("absolute" === d || "fixed" === d) && kb.inArray("auto", [i, j])>-1, l = {}, m = {};
            k ? (m = g.position(), e = m.top, f = m.left) : (e = parseFloat(i) || 0, f = parseFloat(j) || 0), kb.isFunction(b) && (b = b.call(a, c, h)), null != b.top && (l.top = b.top - h.top + e), null != b.left && (l.left = b.left - h.left + f), "using"in b ? b.using.call(a, l) : g.css(l)
        }
    }, kb.fn.extend({
        position: function() {
            if (this[0]) {
                var a, b, c = {
                    top: 0,
                    left: 0
                }, d = this[0];
                return "fixed" === kb.css(d, "position") ? b = d.getBoundingClientRect() : (a = this.offsetParent(), b = this.offset(), kb.nodeName(a[0], "html") || (c = a.offset()), c.top += kb.css(a[0], "borderTopWidth", !0), c.left += kb.css(a[0], "borderLeftWidth", !0)), {
                    top: b.top - c.top - kb.css(d, "marginTop", !0),
                    left: b.left - c.left - kb.css(d, "marginLeft", !0)
                }
            }
        },
        offsetParent: function() {
            return this.map(function() {
                for (var a = this.offsetParent || Z; a&&!kb.nodeName(a, "html") && "static" === kb.css(a, "position");)
                    a = a.offsetParent;
                return a || Z
            })
        }
    }), kb.each({
        scrollLeft: "pageXOffset",
        scrollTop: "pageYOffset"
    }, function(a, c) {
        var d = /Y/.test(c);
        kb.fn[a] = function(e) {
            return kb.access(this, function(a, e, f) {
                var g = T(a);
                return f === b ? g ? c in g ? g[c] : g.document.documentElement[e] : a[e] : void(g ? g.scrollTo(d ? kb(g).scrollLeft() : f, d ? f : kb(g).scrollTop()) : a[e] = f)
            }, a, e, arguments.length, null)
        }
    }), kb.each({
        Height: "height",
        Width: "width"
    }, function(a, c) {
        kb.each({
            padding: "inner" + a,
            content: c,
            "": "outer" + a
        }, function(d, e) {
            kb.fn[e] = function(e, f) {
                var g = arguments.length && (d || "boolean" != typeof e), h = d || (e===!0 || f===!0 ? "margin" : "border");
                return kb.access(this, function(c, d, e) {
                    var f;
                    return kb.isWindow(c) ? c.document.documentElement["client" + a] : 9 === c.nodeType ? (f = c.documentElement, Math.max(c.body["scroll" + a], f["scroll" + a], c.body["offset" + a], f["offset" + a], f["client" + a])) : e === b ? kb.css(c, d, h) : kb.style(c, d, e, h)
                }, c, g ? e : b, g, null)
            }
        })
    }), kb.fn.size = function() {
        return this.length
    }, kb.fn.andSelf = kb.fn.addBack, "object" == typeof module && module && "object" == typeof module.exports ? module.exports = kb : (a.jQuery = a.$ = kb, "function" == typeof define && define.amd && define("jquery", [], function() {
        return kb
    }))
}(window), function(a, b) {
    if ("function" == typeof define && define.amd)
        define("backbone", ["underscore", "jquery", "exports"], function(c, d, e) {
            a.Backbone = b(a, e, c, d)
        });
    else if ("undefined" != typeof exports) {
        var c = require("underscore");
        b(a, exports, c)
    } else 
        a.Backbone = b(a, {}, a._, a.jQuery || a.Zepto || a.ender || a.$)
}(this, function(a, b, c, d) {
    {
        var e = a.Backbone, f = [], g = (f.push, f.slice);
        f.splice
    }
    b.VERSION = "1.1.2", b.$ = d, b.noConflict = function() {
        return a.Backbone = e, this
    }, b.emulateHTTP=!1, b.emulateJSON=!1;
    var h = b.Events = {
        on: function(a, b, c) {
            if (!j(this, "on", a, [b, c]) ||!b)
                return this;
            this._events || (this._events = {});
            var d = this._events[a] || (this._events[a] = []);
            return d.push({
                callback: b,
                context: c,
                ctx: c || this
            }), this
        },
        once: function(a, b, d) {
            if (!j(this, "once", a, [b, d]) ||!b)
                return this;
            var e = this, f = c.once(function() {
                e.off(a, f), b.apply(this, arguments)
            });
            return f._callback = b, this.on(a, f, d)
        },
        off: function(a, b, d) {
            var e, f, g, h, i, k, l, m;
            if (!this._events ||!j(this, "off", a, [b, d]))
                return this;
            if (!a&&!b&&!d)
                return this._events = void 0, this;
            for (h = a ? [a] : c.keys(this._events), i = 0, k = h.length; k > i; i++)
                if (a = h[i], g = this._events[a]) {
                    if (this._events[a] = e = [], b || d)
                        for (l = 0, m = g.length; m > l; l++)
                            f = g[l], (b && b !== f.callback && b !== f.callback._callback || d && d !== f.context) && e.push(f);
                            e.length || delete this._events[a]
                }
            return this
        },
        trigger: function(a) {
            if (!this._events)
                return this;
            var b = g.call(arguments, 1);
            if (!j(this, "trigger", a, b))
                return this;
            var c = this._events[a], d = this._events.all;
            return c && k(c, b), d && k(d, arguments), this
        },
        stopListening: function(a, b, d) {
            var e = this._listeningTo;
            if (!e)
                return this;
            var f=!b&&!d;
            !d && "object" == typeof b && (d = this), a && ((e = {})[a._listenId] = a);
            for (var g in e)
                a = e[g], a.off(b, d, this), (f || c.isEmpty(a._events)) && delete this._listeningTo[g];
            return this
        }
    }, i = /\s+/, j = function(a, b, c, d) {
        if (!c)
            return !0;
        if ("object" == typeof c) {
            for (var e in c)
                a[b].apply(a, [e, c[e]].concat(d));
            return !1
        }
        if (i.test(c)) {
            for (var f = c.split(i), g = 0, h = f.length; h > g; g++)
                a[b].apply(a, [f[g]].concat(d));
            return !1
        }
        return !0
    }, k = function(a, b) {
        var c, d =- 1, e = a.length, f = b[0], g = b[1], h = b[2];
        switch (b.length) {
        case 0:
            for (; ++d < e;)(c = a[d])
                .callback.call(c.ctx);
            return;
        case 1:
            for (; ++d < e;)(c = a[d])
                .callback.call(c.ctx, f);
            return;
        case 2:
            for (; ++d < e;)(c = a[d])
                .callback.call(c.ctx, f, g);
            return;
        case 3:
            for (; ++d < e;)(c = a[d])
                .callback.call(c.ctx, f, g, h);
            return;
        default:
            for (; ++d < e;)(c = a[d])
                .callback.apply(c.ctx, b);
            return 
        }
    }, l = {
        listenTo: "on",
        listenToOnce: "once"
    };
    c.each(l, function(a, b) {
        h[b] = function(b, d, e) {
            var f = this._listeningTo || (this._listeningTo = {}), g = b._listenId || (b._listenId = c.uniqueId("l"));
            return f[g] = b, !e && "object" == typeof d && (e = this), b[a](d, e, this), this
        }
    }), h.bind = h.on, h.unbind = h.off, c.extend(b, h);
    var m = b.Model = function(a, b) {
        var d = a || {};
        b || (b = {}), this.cid = c.uniqueId("c"), this.attributes = {}, b.collection && (this.collection = b.collection), b.parse && (d = this.parse(d, b) || {}), d = c.defaults({}, d, c.result(this, "defaults")), this.set(d, b), this.changed = {}, this.initialize.apply(this, arguments)
    };
    c.extend(m.prototype, h, {
        changed: null,
        validationError: null,
        idAttribute: "id",
        initialize: function() {},
        toJSON: function() {
            return c.clone(this.attributes)
        },
        sync: function() {
            return b.sync.apply(this, arguments)
        },
        get: function(a) {
            return this.attributes[a]
        },
        escape: function(a) {
            return c.escape(this.get(a))
        },
        has: function(a) {
            return null != this.get(a)
        },
        set: function(a, b, d) {
            var e, f, g, h, i, j, k, l;
            if (null == a)
                return this;
            if ("object" == typeof a ? (f = a, d = b) : (f = {})[a] = b, d || (d = {}), !this._validate(f, d))
                return !1;
            g = d.unset, i = d.silent, h = [], j = this._changing, this._changing=!0, j || (this._previousAttributes = c.clone(this.attributes), this.changed = {}), l = this.attributes, k = this._previousAttributes, this.idAttribute in f && (this.id = f[this.idAttribute]);
            for (e in f)
                b = f[e], c.isEqual(l[e], b) || h.push(e), c.isEqual(k[e], b) ? delete this.changed[e] : this.changed[e] = b, g ? delete l[e] : l[e] = b;
            if (!i) {
                h.length && (this._pending = d);
                for (var m = 0, n = h.length; n > m; m++)
                    this.trigger("change:" + h[m], this, l[h[m]], d)
            }
            if (j)
                return this;
            if (!i)
                for (; this._pending;)
                    d = this._pending, this._pending=!1, this.trigger("change", this, d);
            return this._pending=!1, this._changing=!1, this
        },
        unset: function(a, b) {
            return this.set(a, void 0, c.extend({}, b, {
                unset : !0
            }))
        },
        clear: function(a) {
            var b = {};
            for (var d in this.attributes)
                b[d] = void 0;
            return this.set(b, c.extend({}, a, {
                unset: !0
            }))
        },
        hasChanged: function(a) {
            return null == a?!c.isEmpty(this.changed) : c.has(this.changed, a)
        },
        changedAttributes: function(a) {
            if (!a)
                return this.hasChanged() ? c.clone(this.changed) : !1;
            var b, d=!1, e = this._changing ? this._previousAttributes : this.attributes;
            for (var f in a)
                c.isEqual(e[f], b = a[f]) || ((d || (d = {}))[f] = b);
            return d
        },
        previous: function(a) {
            return null != a && this._previousAttributes ? this._previousAttributes[a] : null
        },
        previousAttributes: function() {
            return c.clone(this._previousAttributes)
        },
        fetch: function(a) {
            a = a ? c.clone(a) : {}, void 0 === a.parse && (a.parse=!0);
            var b = this, d = a.success;
            return a.success = function(c) {
                return b.set(b.parse(c, a), a) ? (d && d(b, c, a), void b.trigger("sync", b, c, a)) : !1
            }, L(this, a), this.sync("read", this, a)
        },
        save: function(a, b, d) {
            var e, f, g, h = this.attributes;
            if (null == a || "object" == typeof a ? (e = a, d = b) : (e = {})[a] = b, d = c.extend({
                validate: !0
            }, d), e&&!d.wait) {
                if (!this.set(e, d))
                    return !1
            } else if (!this._validate(e, d))
                return !1;
            e && d.wait && (this.attributes = c.extend({}, h, e)), void 0 === d.parse && (d.parse=!0);
            var i = this, j = d.success;
            return d.success = function(a) {
                i.attributes = h;
                var b = i.parse(a, d);
                return d.wait && (b = c.extend(e || {}, b)), c.isObject(b)&&!i.set(b, d)?!1 : (j && j(i, a, d), void i.trigger("sync", i, a, d))
            }, L(this, d), f = this.isNew() ? "create" : d.patch ? "patch" : "update", "patch" === f && (d.attrs = e), g = this.sync(f, this, d), e && d.wait && (this.attributes = h), g
        },
        destroy: function(a) {
            a = a ? c.clone(a) : {};
            var b = this, d = a.success, e = function() {
                b.trigger("destroy", b, b.collection, a)
            };
            if (a.success = function(c) {
                (a.wait || b.isNew()) && e(), d && d(b, c, a), b.isNew() || b.trigger("sync", b, c, a)
            }, this.isNew())
                return a.success(), !1;
            L(this, a);
            var f = this.sync("delete", this, a);
            return a.wait || e(), f
        },
        url: function() {
            var a = c.result(this, "urlRoot") || c.result(this.collection, "url") || K();
            return this.isNew() ? a : a.replace(/([^\/])$/, "$1/") + encodeURIComponent(this.id)
        },
        parse: function(a) {
            return a
        },
        clone: function() {
            return new this.constructor(this.attributes)
        },
        isNew: function() {
            return !this.has(this.idAttribute)
        },
        isValid: function(a) {
            return this._validate({}, c.extend(a || {}, {
                validate: !0
            }))
        },
        _validate: function(a, b) {
            if (!b.validate ||!this.validate)
                return !0;
            a = c.extend({}, this.attributes, a);
            var d = this.validationError = this.validate(a, b) || null;
            return d ? (this.trigger("invalid", this, d, c.extend(b, {
                validationError: d
            })), !1) : !0
        }
    });
    var n = ["keys", "values", "pairs", "invert", "pick", "omit"];
    c.each(n, function(a) {
        m.prototype[a] = function() {
            var b = g.call(arguments);
            return b.unshift(this.attributes), c[a].apply(c, b)
        }
    });
    var o = b.Collection = function(a, b) {
        b || (b = {}), b.model && (this.model = b.model), void 0 !== b.comparator && (this.comparator = b.comparator), this._reset(), this.initialize.apply(this, arguments), a && this.reset(a, c.extend({
            silent: !0
        }, b))
    }, p = {
        add: !0,
        remove: !0,
        merge: !0
    }, q = {
        add: !0,
        remove: !1
    };
    c.extend(o.prototype, h, {
        model: m,
        initialize: function() {},
        toJSON: function(a) {
            return this.map(function(b) {
                return b.toJSON(a)
            })
        },
        sync: function() {
            return b.sync.apply(this, arguments)
        },
        add: function(a, b) {
            return this.set(a, c.extend({
                merge: !1
            }, b, q))
        },
        remove: function(a, b) {
            var d=!c.isArray(a);
            a = d ? [a] : c.clone(a), b || (b = {});
            var e, f, g, h;
            for (e = 0, f = a.length; f > e; e++)
                h = a[e] = this.get(a[e]), h && (delete this._byId[h.id], delete this._byId[h.cid], g = this.indexOf(h), this.models.splice(g, 1), this.length--, b.silent || (b.index = g, h.trigger("remove", h, this, b)), this._removeReference(h, b));
            return d ? a[0] : a
        },
        set: function(a, b) {
            b = c.defaults({}, b, p), b.parse && (a = this.parse(a, b));
            var d=!c.isArray(a);
            a = d ? a ? [a] : [] : c.clone(a);
            var e, f, g, h, i, j, k, l = b.at, n = this.model, o = this.comparator && null == l && b.sort!==!1, q = c.isString(this.comparator) ? this.comparator: null, r = [], s = [], t = {}, u = b.add, v = b.merge, w = b.remove, x=!o && u && w ? [] : !1;
            for (e = 0, f = a.length; f > e; e++) {
                if (i = a[e] || {}, g = i instanceof m ? h = i : i[n.prototype.idAttribute || "id"], j = this.get(g))
                    w && (t[j.cid]=!0), v && (i = i === h ? h.attributes : i, b.parse && (i = j.parse(i, b)), j.set(i, b), o&&!k && j.hasChanged(q) && (k=!0)), a[e] = j;
                else if (u) {
                    if (h = a[e] = this._prepareModel(i, b), !h)
                        continue;
                    r.push(h), this._addReference(h, b)
                }
                h = j || h, x && (h.isNew() ||!t[h.id]) && x.push(h), t[h.id]=!0
            }
            if (w) {
                for (e = 0, f = this.length; f > e; ++e)
                    t[(h = this.models[e]).cid] || s.push(h);
                s.length && this.remove(s, b)
            }
            if (r.length || x && x.length)
                if (o && (k=!0), this.length += r.length, null != l)
                    for (e = 0, f = r.length; f > e; e++)
                        this.models.splice(l + e, 0, r[e]);
                else {
                    x && (this.models.length = 0);
                    var y = x || r;
                    for (e = 0, f = y.length; f > e; e++)
                        this.models.push(y[e])
                }
            if (k && this.sort({
                silent: !0
            }), !b.silent) {
                for (e = 0, f = r.length; f > e; e++)(h = r[e])
                    .trigger("add", h, this, b);
                (k || x && x.length) && this.trigger("sort", this, b)
            }
            return d ? a[0] : a
        },
        reset: function(a, b) {
            b || (b = {});
            for (var d = 0, e = this.models.length; e > d; d++)
                this._removeReference(this.models[d], b);
            return b.previousModels = this.models, this._reset(), a = this.add(a, c.extend({
                silent: !0
            }, b)), b.silent || this.trigger("reset", this, b), a
        },
        push: function(a, b) {
            return this.add(a, c.extend({
                at: this.length
            }, b))
        },
        pop: function(a) {
            var b = this.at(this.length - 1);
            return this.remove(b, a), b
        },
        unshift: function(a, b) {
            return this.add(a, c.extend({
                at: 0
            }, b))
        },
        shift: function(a) {
            var b = this.at(0);
            return this.remove(b, a), b
        },
        slice: function() {
            return g.apply(this.models, arguments)
        },
        get: function(a) {
            return null == a ? void 0 : this._byId[a] || this._byId[a.id] || this._byId[a.cid]
        },
        at: function(a) {
            return this.models[a]
        },
        where: function(a, b) {
            return c.isEmpty(a) ? b ? void 0 : [] : this[b ? "find": "filter"](function(b) {
                for (var c in a)
                    if (a[c] !== b.get(c))
                        return !1;
                return !0
            })
        },
        findWhere: function(a) {
            return this.where(a, !0)
        },
        sort: function(a) {
            if (!this.comparator)
                throw new Error("Cannot sort a set without a comparator");
            return a || (a = {}), c.isString(this.comparator) || 1 === this.comparator.length ? this.models = this.sortBy(this.comparator, this) : this.models.sort(c.bind(this.comparator, this)), a.silent || this.trigger("sort", this, a), this
        },
        pluck: function(a) {
            return c.invoke(this.models, "get", a)
        },
        fetch: function(a) {
            a = a ? c.clone(a) : {}, void 0 === a.parse && (a.parse=!0);
            var b = a.success, d = this;
            return a.success = function(c) {
                var e = a.reset ? "reset": "set";
                d[e](c, a), b && b(d, c, a), d.trigger("sync", d, c, a)
            }, L(this, a), this.sync("read", this, a)
        },
        create: function(a, b) {
            if (b = b ? c.clone(b) : {}, !(a = this._prepareModel(a, b)))
                return !1;
            b.wait || this.add(a, b);
            var d = this, e = b.success;
            return b.success = function(a, c) {
                b.wait && d.add(a, b), e && e(a, c, b)
            }, a.save(null, b), a
        },
        parse: function(a) {
            return a
        },
        clone: function() {
            return new this.constructor(this.models)
        },
        _reset: function() {
            this.length = 0, this.models = [], this._byId = {}
        },
        _prepareModel: function(a, b) {
            if (a instanceof m)
                return a;
            b = b ? c.clone(b) : {}, b.collection = this;
            var d = new this.model(a, b);
            return d.validationError ? (this.trigger("invalid", this, d.validationError, b), !1) : d
        },
        _addReference: function(a) {
            this._byId[a.cid] = a, null != a.id && (this._byId[a.id] = a), a.collection || (a.collection = this), a.on("all", this._onModelEvent, this)
        },
        _removeReference: function(a) {
            this === a.collection && delete a.collection, a.off("all", this._onModelEvent, this)
        },
        _onModelEvent: function(a, b, c, d) {
            ("add" !== a && "remove" !== a || c === this) && ("destroy" === a && this.remove(b, d), b && a === "change:" + b.idAttribute && (delete this._byId[b.previous(b.idAttribute)], null != b.id && (this._byId[b.id] = b)), this.trigger.apply(this, arguments))
        }
    });
    var r = ["forEach", "each", "map", "collect", "reduce", "foldl", "inject", "reduceRight", "foldr", "find", "detect", "filter", "select", "reject", "every", "all", "some", "any", "include", "contains", "invoke", "max", "min", "toArray", "size", "first", "head", "take", "initial", "rest", "tail", "drop", "last", "without", "difference", "indexOf", "shuffle", "lastIndexOf", "isEmpty", "chain", "sample"];
    c.each(r, function(a) {
        o.prototype[a] = function() {
            var b = g.call(arguments);
            return b.unshift(this.models), c[a].apply(c, b)
        }
    });
    var s = ["groupBy", "countBy", "sortBy", "indexBy"];
    c.each(s, function(a) {
        o.prototype[a] = function(b, d) {
            var e = c.isFunction(b) ? b: function(a) {
                return a.get(b)
            };
            return c[a](this.models, e, d)
        }
    });
    var t = b.View = function(a) {
        this.cid = c.uniqueId("view"), a || (a = {}), c.extend(this, c.pick(a, v)), this._ensureElement(), this.initialize.apply(this, arguments), this.delegateEvents()
    }, u = /^(\S+)\s*(.*)$/, v = ["model", "collection", "el", "id", "attributes", "className", "tagName", "events"];
    c.extend(t.prototype, h, {
        tagName: "div",
        $: function(a) {
            return this.$el.find(a)
        },
        initialize: function() {},
        render: function() {
            return this
        },
        remove: function() {
            return this.$el.remove(), this.stopListening(), this
        },
        setElement: function(a, c) {
            return this.$el && this.undelegateEvents(), this.$el = a instanceof b.$ ? a : b.$(a), this.el = this.$el[0], c!==!1 && this.delegateEvents(), this
        },
        delegateEvents: function(a) {
            if (!a&&!(a = c.result(this, "events")))
                return this;
            this.undelegateEvents();
            for (var b in a) {
                var d = a[b];
                if (c.isFunction(d) || (d = this[a[b]]), d) {
                    var e = b.match(u), f = e[1], g = e[2];
                    d = c.bind(d, this), f += ".delegateEvents" + this.cid, "" === g ? this.$el.on(f, d) : this.$el.on(f, g, d)
                }
            }
            return this
        },
        undelegateEvents: function() {
            return this.$el.off(".delegateEvents" + this.cid), this
        },
        _ensureElement: function() {
            if (this.el)
                this.setElement(c.result(this, "el"), !1);
            else {
                var a = c.extend({}, c.result(this, "attributes"));
                this.id && (a.id = c.result(this, "id")), this.className && (a["class"] = c.result(this, "className"));
                var d = b.$("<" + c.result(this, "tagName") + ">").attr(a);
                this.setElement(d, !1)
            }
        }
    }), b.sync = function(a, d, e) {
        var f = x[a];
        c.defaults(e || (e = {}), {
            emulateHTTP: b.emulateHTTP,
            emulateJSON: b.emulateJSON
        });
        var g = {
            type: f,
            dataType: "json"
        };
        if (e.url || (g.url = c.result(d, "url") || K()), null == e.data && d && ("create" === a || "update" === a || "patch" === a) && (g.contentType = "application/json", g.data = JSON.stringify(e.attrs || d.toJSON(e))), e.emulateJSON && (g.contentType = "application/x-www-form-urlencoded", g.data = g.data ? {
            model: g.data
        } : {}), e.emulateHTTP && ("PUT" === f || "DELETE" === f || "PATCH" === f)) {
            g.type = "POST", e.emulateJSON && (g.data._method = f);
            var h = e.beforeSend;
            e.beforeSend = function(a) {
                return a.setRequestHeader("X-HTTP-Method-Override", f), h ? h.apply(this, arguments) : void 0
            }
        }
        "GET" !== g.type&&!e.emulateJSON && (g.processData=!1), "PATCH" === g.type && w && (g.xhr = function() {
            return new ActiveXObject("Microsoft.XMLHTTP")
        });
        var i = e.xhr = b.ajax(c.extend(g, e));
        return d.trigger("request", d, i, e), i
    };
    var w=!("undefined" == typeof window ||!window.ActiveXObject || window.XMLHttpRequest && (new XMLHttpRequest).dispatchEvent), x = {
        create: "POST",
        update: "PUT",
        patch: "PATCH",
        "delete": "DELETE",
        read: "GET"
    };
    b.ajax = function() {
        return b.$.ajax.apply(b.$, arguments)
    };
    var y = b.Router = function(a) {
        a || (a = {}), a.routes && (this.routes = a.routes), this._bindRoutes(), this.initialize.apply(this, arguments)
    }, z = /\((.*?)\)/g, A = /(\(\?)?:\w+/g, B = /\*\w+/g, C = /[\-{}\[\]+?.,\\\^$|#\s]/g;
    c.extend(y.prototype, h, {
        initialize: function() {},
        route: function(a, d, e) {
            c.isRegExp(a) || (a = this._routeToRegExp(a)), c.isFunction(d) && (e = d, d = ""), e || (e = this[d]);
            var f = this;
            return b.history.route(a, function(c) {
                var g = f._extractParameters(a, c);
                f.execute(e, g), f.trigger.apply(f, ["route:" + d].concat(g)), f.trigger("route", d, g), b.history.trigger("route", f, d, g)
            }), this
        },
        execute: function(a, b) {
            a && a.apply(this, b)
        },
        navigate: function(a, c) {
            return b.history.navigate(a, c), this
        },
        _bindRoutes: function() {
            if (this.routes) {
                this.routes = c.result(this, "routes");
                for (var a, b = c.keys(this.routes); null != (a = b.pop());)
                    this.route(a, this.routes[a])
            }
        },
        _routeToRegExp: function(a) {
            return a = a.replace(C, "\\$&").replace(z, "(?:$1)?").replace(A, function(a, b) {
                return b ? a : "([^/?]+)"
            }).replace(B, "([^?]*?)"), new RegExp("^" + a + "(?:\\?([\\s\\S]*))?$")
        },
        _extractParameters: function(a, b) {
            var d = a.exec(b).slice(1);
            return c.map(d, function(a, b) {
                return b === d.length - 1 ? a || null : a ? decodeURIComponent(a) : null
            })
        }
    });
    var D = b.History = function() {
        this.handlers = [], c.bindAll(this, "checkUrl"), "undefined" != typeof window && (this.location = window.location, this.history = window.history)
    }, E = /^[#\/]|\s+$/g, F = /^\/+|\/+$/g, G = /msie [\w.]+/, H = /\/$/, I = /#.*$/;
    D.started=!1, c.extend(D.prototype, h, {
        interval: 50,
        atRoot: function() {
            return this.location.pathname.replace(/[^\/]$/, "$&/") === this.root
        },
        getHash: function(a) {
            var b = (a || this).location.href.match(/#(.*)$/);
            return b ? b[1] : ""
        },
        getFragment: function(a, b) {
            if (null == a)
                if (this._hasPushState ||!this._wantsHashChange || b) {
                    a = decodeURI(this.location.pathname + this.location.search);
                    var c = this.root.replace(H, "");
                    a.indexOf(c) || (a = a.slice(c.length))
                } else 
                    a = this.getHash();
            return a.replace(E, "")
        },
        start: function(a) {
            if (D.started)
                throw new Error("Backbone.history has already been started");
            D.started=!0, this.options = c.extend({
                root: "/"
            }, this.options, a), this.root = this.options.root, this._wantsHashChange = this.options.hashChange!==!1, this._wantsPushState=!!this.options.pushState, this._hasPushState=!!(this.options.pushState && this.history && this.history.pushState);
            var d = this.getFragment(), e = document.documentMode, f = G.exec(navigator.userAgent.toLowerCase()) && (!e || 7 >= e);
            if (this.root = ("/" + this.root + "/").replace(F, "/"), f && this._wantsHashChange) {
                var g = b.$('<iframe src="javascript:0" tabindex="-1">');
                this.iframe = g.hide().appendTo("body")[0].contentWindow, this.navigate(d)
            }
            this._hasPushState ? b.$(window).on("popstate", this.checkUrl) : this._wantsHashChange && "onhashchange"in window&&!f ? b.$(window).on("hashchange", this.checkUrl) : this._wantsHashChange && (this._checkUrlInterval = setInterval(this.checkUrl, this.interval)), this.fragment = d;
            var h = this.location;
            if (this._wantsHashChange && this._wantsPushState) {
                if (!this._hasPushState&&!this.atRoot())
                    return this.fragment = this.getFragment(null, !0), this.location.replace(this.root + "#" + this.fragment), !0;
                this._hasPushState && this.atRoot() && h.hash && (this.fragment = this.getHash().replace(E, ""), this.history.replaceState({}, document.title, this.root + this.fragment))
            }
            return this.options.silent ? void 0 : this.loadUrl()
        },
        stop: function() {
            b.$(window).off("popstate", this.checkUrl).off("hashchange", this.checkUrl), this._checkUrlInterval && clearInterval(this._checkUrlInterval), D.started=!1
        },
        route: function(a, b) {
            this.handlers.unshift({
                route: a,
                callback: b
            })
        },
        checkUrl: function() {
            var a = this.getFragment();
            return a === this.fragment && this.iframe && (a = this.getFragment(this.getHash(this.iframe))), a === this.fragment?!1 : (this.iframe && this.navigate(a), void this.loadUrl())
        },
        loadUrl: function(a) {
            return a = this.fragment = this.getFragment(a), c.any(this.handlers, function(b) {
                return b.route.test(a) ? (b.callback(a), !0) : void 0
            })
        },
        navigate: function(a, b) {
            if (!D.started)
                return !1;
            b && b!==!0 || (b = {
                trigger: !!b
            });
            var c = this.root + (a = this.getFragment(a || ""));
            if (a = a.replace(I, ""), this.fragment !== a) {
                if (this.fragment = a, "" === a && "/" !== c && (c = c.slice(0, - 1)), this._hasPushState)
                    this.history[b.replace ? "replaceState": "pushState"]({}, document.title, c);
                else {
                    if (!this._wantsHashChange)
                        return this.location.assign(c);
                    this._updateHash(this.location, a, b.replace), this.iframe && a !== this.getFragment(this.getHash(this.iframe)) && (b.replace || this.iframe.document.open().close(), this._updateHash(this.iframe.location, a, b.replace))
                }
                return b.trigger ? this.loadUrl(a) : void 0
            }
        },
        _updateHash: function(a, b, c) {
            if (c) {
                var d = a.href.replace(/(javascript:|#).*$/, "");
                a.replace(d + "#" + b)
            } else 
                a.hash = "#" + b
        }
    }), b.history = new D;
    var J = function(a, b) {
        var d, e = this;
        d = a && c.has(a, "constructor") ? a.constructor : function() {
            return e.apply(this, arguments)
        }, c.extend(d, e, b);
        var f = function() {
            this.constructor = d
        };
        return f.prototype = e.prototype, d.prototype = new f, a && c.extend(d.prototype, a), d.__super__ = e.prototype, d
    };
    m.extend = o.extend = y.extend = t.extend = D.extend = J;
    var K = function() {
        throw new Error('A "url" property or function must be specified')
    }, L = function(a, b) {
        var c = b.error;
        b.error = function(d) {
            c && c(a, d, b), a.trigger("error", a, d, b)
        }
    };
    return b
}), function() {
    var a = function() {
        var a = {}, b = {}, c = function() {
            return function() {}
        }, d = function(a, d) {
            var e = {
                id: a,
                exports: {}
            };
            d(e.exports, c(), e);
            var f = b[a] = e.exports;
            return f
        }, e = function(c) {
            if (b.hasOwnProperty(c))
                return b[c];
            if (a.hasOwnProperty(c))
                return d(c, a[c]);
            throw new Error('Cannot find module "' + c + '"')
        };
        return e.register = function(b, c) {
            a[b] = c
        }, e
    }();
    a.register("chaplin/application", function(b, c, d) {
        var e, f, g, h, i, j, k, l, m;
        m = a("underscore"), f = a("backbone"), l = a("chaplin/mediator"), h = a("chaplin/dispatcher"), j = a("chaplin/views/layout"), g = a("chaplin/composer"), k = a("chaplin/lib/router"), i = a("chaplin/lib/event_broker"), d.exports = e = function() {
            function a(a) {
                null == a && (a = {}), this.initialize(a)
            }
            return a.extend = f.Model.extend, m.extend(a.prototype, i), a.prototype.title = "", a.prototype.dispatcher = null, a.prototype.layout = null, a.prototype.router = null, a.prototype.composer = null, a.prototype.started=!1, a.prototype.initialize = function(a) {
                if (null == a && (a = {}), this.started)
                    throw new Error("Application#initialize: App was already started");
                return this.initRouter(a.routes, a), this.initDispatcher(a), this.initLayout(a), this.initComposer(a), this.initMediator(), this.start()
            }, a.prototype.initDispatcher = function(a) {
                return this.dispatcher = new h(a)
            }, a.prototype.initLayout = function(a) {
                var b;
                return null == a && (a = {}), null == (b = a.title) && (a.title = this.title), this.layout = new j(a)
            }, a.prototype.initComposer = function(a) {
                return null == a && (a = {}), this.composer = new g(a)
            }, a.prototype.initMediator = function() {
                return l.seal()
            }, a.prototype.initRouter = function(a, b) {
                return this.router = new k(b), "function" == typeof a ? a(this.router.match) : void 0
            }, a.prototype.start = function() {
                return this.router.startHistory(), this.started=!0, "function" == typeof Object.freeze ? Object.freeze(this) : void 0
            }, a.prototype.disposed=!1, a.prototype.dispose = function() {
                var a, b, c, d, e;
                if (!this.disposed) {
                    for (a = "function" == typeof Object.isFrozen ? Object.isFrozen(this) : void 0, c = ["dispatcher", "layout", "router", "composer"], d = 0, e = c.length; e > d; d++)
                        b = c[d], null != this[b] && (this[b].dispose(), a || delete this[b]);
                    return this.disposed=!0, "function" == typeof Object.freeze ? Object.freeze(this) : void 0
                }
            }, a
        }()
    }), a.register("chaplin/mediator", function(b, c, d) {
        var e, f, g, h, i, j, k = [].slice;
        e = a("backbone"), j = a("underscore"), h = a("chaplin/lib/support"), i = a("chaplin/lib/utils"), g = {}, g.subscribe = e.Events.on, g.unsubscribe = e.Events.off, g.publish = e.Events.trigger, g._callbacks = null, f = g._handlers = {}, g.setHandler = function(a, b, c) {
            return f[a] = {
                instance: c,
                method: b
            }
        }, g.execute = function() {
            var a, b, c, d, e;
            if (d = arguments[0], a = 2 <= arguments.length ? k.call(arguments, 1) : [], e=!1, "object" == typeof d ? (e = d.silent, c = d.name) : c = d, b = f[c], b)
                return b.method.apply(b.instance, a);
            if (!e)
                throw new Error("mediator.execute: " + c + " handler is not defined")
        }, g.removeHandlers = function(a) {
            var b, c, d, e;
            if (a || (g._handlers = {}), i.isArray(a))
                for (d = 0, e = a.length; e > d; d++)
                    c = a[d], delete f[c];
            else 
                for (c in f)
                    b = f[c], b.instance === a && delete f[c]
        }, i.readonly(g, "subscribe", "unsubscribe", "publish", "setHandler", "execute", "removeHandlers"), g.seal = function() {
            return h.propertyDescriptors && Object.seal ? Object.seal(g) : void 0
        }, i.readonly(g, "seal"), d.exports = g
    }), a.register("chaplin/dispatcher", function(b, c, d) {
        var e, f, g, h, i, j;
        j = a("underscore"), e = a("backbone"), h = a("chaplin/mediator"), i = a("chaplin/lib/utils"), g = a("chaplin/lib/event_broker"), d.exports = f = function() {
            function a() {
                this.initialize.apply(this, arguments)
            }
            return a.extend = e.Model.extend, j.extend(a.prototype, g), a.prototype.previousRoute = null, a.prototype.currentController = null, a.prototype.currentRoute = null, a.prototype.currentParams = null, a.prototype.currentQuery = null, a.prototype.initialize = function(a) {
                return null == a && (a = {}), this.settings = j.defaults(a, {
                    controllerPath: "controllers/",
                    controllerSuffix: "_controller"
                }), this.subscribeEvent("router:match", this.dispatch)
            }, a.prototype.dispatch = function(a, b, c) {
                var d, e, f = this;
                return b = b ? j.extend({}, b) : {}, c = c ? j.extend({}, c) : {}, null == c.query && (c.query = {}), c.changeURL!==!1 && (c.changeURL=!0), c.forceStartup!==!0 && (c.forceStartup=!1), !c.forceStartup && (null != (d = this.currentRoute) ? d.controller : void 0) === a.controller && (null != (e = this.currentRoute) ? e.action : void 0) === a.action && j.isEqual(this.currentParams, b) && j.isEqual(this.currentQuery, c.query) ? void 0 : this.loadController(a.controller, function(d) {
                    return f.controllerLoaded(a, b, c, d)
                })
            }, a.prototype.loadController = function(a, b) {
                var c, d;
                return c = a + this.settings.controllerSuffix, d = this.settings.controllerPath + c, ("undefined" != typeof define && null !== define ? define.amd : void 0) ? require([d], b) : b(require(d))
            }, a.prototype.controllerLoaded = function(a, b, c, d) {
                var e;
                return this.previousRoute = this.currentRoute, this.currentRoute = j.extend({}, a, {
                    previous: i.beget(this.previousRoute)
                }), e = new d(b, this.currentRoute, c), this.executeBeforeAction(e, this.currentRoute, b, c)
            }, a.prototype.executeAction = function(a, b, c, d) {
                return this.currentController && (this.publishEvent("beforeControllerDispose", this.currentController), this.currentController.dispose(c, b, d)), this.currentController = a, this.currentParams = c, this.currentQuery = d.query, a[b.action](c, b, d), a.redirected ? void 0 : (this.adjustURL(b, c, d), this.publishEvent("dispatcher:dispatch", this.currentController, c, b, d))
            }, a.prototype.executeBeforeAction = function(a, b, c, d) {
                var e, f, g, h = this;
                if (e = a.beforeAction, f = function() {
                    return a.redirected || h.currentRoute && b !== h.currentRoute ? void a.dispose() : h.executeAction(a, b, c, d)
                }, !e)
                    return void f();
                if ("function" != typeof e)
                    throw new TypeError("Controller#beforeAction: function expected. Old object-like form is not supported.");
                return g = a.beforeAction(c, b, d), g && g.then ? g.then(f) : f()
            }, a.prototype.adjustURL = function(a, b, c) {
                var d;
                if (null != a.path)
                    return d = a.path + (a.query ? "?" + a.query : ""), c.changeURL ? h.execute("router:changeURL", d, c) : void 0
            }, a.prototype.disposed=!1, a.prototype.dispose = function() {
                return this.disposed ? void 0 : (this.unsubscribeAllEvents(), this.disposed=!0, "function" == typeof Object.freeze ? Object.freeze(this) : void 0)
            }, a
        }()
    }), a.register("chaplin/composer", function(b, c, d) {
        var e, f, g, h, i, j, k;
        k = a("underscore"), e = a("backbone"), i = a("chaplin/mediator"), j = a("chaplin/lib/utils"), g = a("chaplin/lib/composition"), h = a("chaplin/lib/event_broker"), d.exports = f = function() {
            function a() {
                this.initialize.apply(this, arguments)
            }
            return a.extend = e.Model.extend, k.extend(a.prototype, h), a.prototype.compositions = null, a.prototype.initialize = function(a) {
                return null == a && (a = {}), this.compositions = {}, i.setHandler("composer:compose", this.compose, this), i.setHandler("composer:retrieve", this.retrieve, this), this.subscribeEvent("dispatcher:dispatch", this.cleanup)
            }, a.prototype.compose = function(a, b, c) {
                return "function" == typeof b ? c || b.prototype.dispose ? b.prototype instanceof g ? this._compose(a, {
                    composition: b,
                    options: c
                }) : this._compose(a, {
                    options: c,
                    compose: function() {
                        var a, c;
                        return this.item = new b(this.options), a = this.item.autoRender, c = void 0 === a ||!a, c && "function" == typeof this.item.render ? this.item.render() : void 0
                    }
                }) : this._compose(a, {
                    compose: b
                }) : "function" == typeof c ? this._compose(a, {
                    compose: c,
                    options: b
                }) : this._compose(a, b)
            }, a.prototype._compose = function(a, b) {
                var c, d, e, f;
                if ("function" != typeof b.compose && null == b.composition)
                    throw new Error("Composer#compose was used incorrectly");
                return null != b.composition ? c = new b.composition(b.options) : (c = new g(b.options), c.compose = b.compose, b.check && (c.check = b.check)), d = this.compositions[a], e=!1, d && d.check(c.options) ? d.stale(!1) : (d && d.dispose(), f = c.compose(c.options), e = "function" == typeof(null != f ? f.then : void 0), c.stale(!1), this.compositions[a] = c), e ? f : this.compositions[a].item
            }, a.prototype.retrieve = function(a) {
                var b;
                return b = this.compositions[a], b&&!b.stale() ? b.item : void 0
            }, a.prototype.cleanup = function() {
                var a, b, c;
                c = this.compositions;
                for (b in c)
                    a = c[b], a.stale() ? (a.dispose(), delete this.compositions[b]) : a.stale(!0)
            }, a.prototype.dispose = function() {
                var a, b, c;
                if (!this.disposed) {
                    this.unsubscribeAllEvents(), i.removeHandlers(this), c = this.compositions;
                    for (b in c)
                        a = c[b], a.dispose();
                    return delete this.compositions, this.disposed=!0, "function" == typeof Object.freeze ? Object.freeze(this) : void 0
                }
            }, a
        }()
    }), a.register("chaplin/controllers/controller", function(b, c, d) {
        var e, f, g, h, i, j, k = [].slice, l = {}.hasOwnProperty;
        j = a("underscore"), e = a("backbone"), g = a("chaplin/lib/event_broker"), h = a("chaplin/lib/helpers"), i = a("chaplin/mediator"), d.exports = f = function() {
            function a() {
                this.initialize.apply(this, arguments)
            }
            return a.extend = e.Model.extend, j.extend(a.prototype, e.Events), j.extend(a.prototype, g), a.prototype.view = null, a.prototype.redirected=!1, a.prototype.initialize = function() {}, a.prototype.beforeAction = function() {}, a.prototype.adjustTitle = function(a) {
                return i.execute("adjustTitle", a)
            }, a.prototype.compose = function() {
                var a;
                return a = 1 === arguments.length ? "retrieve" : "compose", i.execute.apply(i, ["composer:" + a].concat(k.call(arguments)))
            }, a.prototype.redirectTo = function(a, b, c) {
                return this.redirected=!0, h.redirectTo(a, b, c)
            }, a.prototype.disposed=!1, a.prototype.dispose = function() {
                var a, b;
                if (!this.disposed) {
                    for (b in this)
                        l.call(this, b) && (a = this[b], a && "function" == typeof a.dispose && (a.dispose(), delete this[b]));
                    return this.unsubscribeAllEvents(), this.stopListening(), this.disposed=!0, "function" == typeof Object.freeze ? Object.freeze(this) : void 0
                }
            }, a
        }()
    }), a.register("chaplin/models/collection", function(b, c, d) {
        var e, f, g, h, i, j, k = {}.hasOwnProperty, l = function(a, b) {
            function c() {
                this.constructor = a
            }
            for (var d in b)
                k.call(b, d) && (a[d] = b[d]);
            return c.prototype = b.prototype, a.prototype = new c, a.__super__ = b.prototype, a
        };
        j = a("underscore"), e = a("backbone"), g = a("chaplin/lib/event_broker"), h = a("chaplin/models/model"), i = a("chaplin/lib/utils"), d.exports = f = function(a) {
            function b() {
                return b.__super__.constructor.apply(this, arguments)
            }
            return l(b, a), j.extend(b.prototype, g), b.prototype.model = h, b.prototype.serialize = function() {
                return this.map(i.serialize)
            }, b.prototype.disposed=!1, b.prototype.dispose = function() {
                var a, b, c, d;
                if (!this.disposed) {
                    for (this.trigger("dispose", this), this.reset([], {
                        silent: !0
                    }), this.unsubscribeAllEvents(), this.stopListening(), this.off(), b = ["model", "models", "_byId", "_byCid", "_callbacks"], c = 0, d = b.length; d > c; c++)
                        a = b[c], delete this[a];
                    return this.disposed=!0, "function" == typeof Object.freeze ? Object.freeze(this) : void 0
                }
            }, b
        }(e.Collection)
    }), a.register("chaplin/models/model", function(b, c, d) {
        var e, f, g, h, i, j, k, l = {}.hasOwnProperty, m = function(a, b) {
            function c() {
                this.constructor = a
            }
            for (var d in b)
                l.call(b, d) && (a[d] = b[d]);
            return c.prototype = b.prototype, a.prototype = new c, a.__super__ = b.prototype, a
        };
        k = a("underscore"), e = a("backbone"), j = a("chaplin/lib/utils"), f = a("chaplin/lib/event_broker"), h = function(a, b, c) {
            var d, f, g, h, k, l, m, n;
            d = j.beget(b), null == c && (c = {}), c[a.cid]=!0;
            for (f in b)
                if (k = b[f], k instanceof e.Model)
                    d[f] = i(k, a, c);
                else if (k instanceof e.Collection) {
                    for (h = [], n = k.models, l = 0, m = n.length; m > l; l++)
                        g = n[l], h.push(i(g, a, c));
                        d[f] = h
                }
            return delete c[a.cid], d
        }, i = function(a, b, c) {
            var d;
            return a === b || a.cid in c ? null : (d = "function" == typeof a.getAttributes ? a.getAttributes() : a.attributes, h(a, d, c))
        }, d.exports = g = function(a) {
            function b() {
                return b.__super__.constructor.apply(this, arguments)
            }
            return m(b, a), k.extend(b.prototype, f), b.prototype.getAttributes = function() {
                return this.attributes
            }, b.prototype.serialize = function() {
                return h(this, this.getAttributes())
            }, b.prototype.disposed=!1, b.prototype.dispose = function() {
                var a, b, c, d;
                if (!this.disposed) {
                    for (this.trigger("dispose", this), this.unsubscribeAllEvents(), this.stopListening(), this.off(), b = ["collection", "attributes", "changed", "_escapedAttributes", "_previousAttributes", "_silent", "_pending", "_callbacks"], c = 0, d = b.length; d > c; c++)
                        a = b[c], delete this[a];
                    return this.disposed=!0, "function" == typeof Object.freeze ? Object.freeze(this) : void 0
                }
            }, b
        }(e.Model)
    }), a.register("chaplin/views/layout", function(b, c, d) {
        var e, f, g, h, i, j, k, l, m, n = function(a, b) {
            return function() {
                return a.apply(b, arguments)
            }
        }, o = {}.hasOwnProperty, p = function(a, b) {
            function c() {
                this.constructor = a
            }
            for (var d in b)
                o.call(b, d) && (a[d] = b[d]);
            return c.prototype = b.prototype, a.prototype = new c, a.__super__ = b.prototype, a
        };
        m = a("underscore"), f = a("backbone"), k = a("chaplin/mediator"), j = a("chaplin/lib/helpers"), l = a("chaplin/lib/utils"), g = a("chaplin/lib/event_broker"), i = a("chaplin/views/view"), e = f.$, d.exports = h = function(a) {
            function b(a) {
                null == a && (a = {}), this.openLink = n(this.openLink, this), this.globalRegions = [], this.title = a.title, a.regions && (this.regions = a.regions), this.settings = m.defaults(a, {
                    titleTemplate: function(a) {
                        var b;
                        return b = a.subtitle ? "" + a.subtitle + " – " : "", b + a.title
                    },
                    openExternalToBlank: !1,
                    routeLinks: "a, .go-to",
                    skipRouting: ".noscript",
                    scrollTo: [0, 0]
                }), k.setHandler("region:show", this.showRegion, this), k.setHandler("region:register", this.registerRegionHandler, this), k.setHandler("region:unregister", this.unregisterRegionHandler, this), k.setHandler("region:find", this.regionByName, this), k.setHandler("adjustTitle", this.adjustTitle, this), b.__super__.constructor.apply(this, arguments), this.settings.routeLinks && this.startLinkRouting()
            }
            return p(b, a), b.prototype.el = "body", b.prototype.keepElement=!0, b.prototype.title = "", b.prototype.globalRegions = null, b.prototype.listen = {
                "beforeControllerDispose mediator": "scroll"
            }, b.prototype.scroll = function() {
                var a;
                return a = this.settings.scrollTo, a ? window.scrollTo(a[0], a[1]) : void 0
            }, b.prototype.adjustTitle = function(a) {
                var b, c = this;
                return null == a && (a = ""), b = this.settings.titleTemplate({
                    title: this.title,
                    subtitle: a
                }), setTimeout(function() {
                    return document.title = b, c.publishEvent("adjustTitle", a, b)
                }, 50), b
            }, b.prototype.startLinkRouting = function() {
                var a;
                return (a = this.settings.routeLinks) ? e ? this.$el.on("click", a, this.openLink) : this.delegate("click", a, this.openLink) : void 0
            }, b.prototype.stopLinkRouting = function() {
                var a;
                return a = this.settings.routeLinks, e ? a ? this.$el.off("click", a) : void 0 : this.undelegate("click", a, this.openLink)
            }, b.prototype.isExternalLink = function(a) {
                var b, c;
                return "_blank" === a.target || "external" === a.rel || "http:" !== (b = a.protocol) && "https:" !== b && "file:" !== b || (c = a.hostname) !== location.hostname && "" !== c
            }, b.prototype.openLink = function(a) {
                var b, c, d, g, h, i;
                if (!(l.modifierKeyPressed(a) || (b = e ? a.currentTarget : a.delegateTarget, g = "A" === b.nodeName, d = b.getAttribute("href") || b.getAttribute("data-href") || null, null == d || "" === d || "#" === d.charAt(0) || (h = this.settings.skipRouting, i = typeof h, "function" === i&&!h(d, b) || "string" === i && (e ? e(b).is(h) : f.utils.matchesSelector(b, h))))))
                    return (c = g && this.isExternalLink(b)) ? void(this.settings.openExternalToBlank && (a.preventDefault(), window.open(d))) : (j.redirectTo({
                        url: d
                    }), void a.preventDefault())
            }, b.prototype.registerRegionHandler = function(a, b, c) {
                return null != b ? this.registerGlobalRegion(a, b, c) : this.registerGlobalRegions(a)
            }, b.prototype.registerGlobalRegion = function(a, b, c) {
                return this.unregisterGlobalRegion(a, b), this.globalRegions.unshift({
                    instance: a,
                    name: b,
                    selector: c
                })
            }, b.prototype.registerGlobalRegions = function(a) {
                var b, c, d, e, f, g;
                for (g = l.getAllPropertyVersions(a, "regions"), e = 0, f = g.length; f > e; e++) {
                    d = g[e];
                    for (b in d)
                        c = d[b], this.registerGlobalRegion(a, b, c)
                }
            }, b.prototype.unregisterRegionHandler = function(a, b) {
                return null != b ? this.unregisterGlobalRegion(a, b) : this.unregisterGlobalRegions(a)
            }, b.prototype.unregisterGlobalRegion = function(a, b) {
                var c, d;
                return c = a.cid, this.globalRegions = function() {
                    var a, e, f, g;
                    for (f = this.globalRegions, g = [], a = 0, e = f.length; e > a; a++)
                        d = f[a], (d.instance.cid !== c || d.name !== b) && g.push(d);
                    return g
                }.call(this)
            }, b.prototype.unregisterGlobalRegions = function(a) {
                var b;
                return this.globalRegions = function() {
                    var c, d, e, f;
                    for (e = this.globalRegions, f = [], c = 0, d = e.length; d > c; c++)
                        b = e[c], b.instance.cid !== a.cid && f.push(b);
                    return f
                }.call(this)
            }, b.prototype.regionByName = function(a) {
                var b, c, d, e;
                for (e = this.globalRegions, c = 0, d = e.length; d > c; c++)
                    if (b = e[c], b.name === a&&!b.instance.stale)
                        return b
            }, b.prototype.showRegion = function(a, b) {
                var c;
                if (c = this.regionByName(a), !c)
                    throw new Error("No region registered under " + a);
                return b.container = "" === c.selector ? e ? c.instance.$el : c.instance.el : c.instance.noWrap ? e ? e(c.instance.container).find(c.selector) : c.instance.container.querySelector(c.selector) : c.instance[e ? "$": "find"](c.selector)
            }, b.prototype.dispose = function() {
                var a, c, d, e;
                if (!this.disposed) {
                    for (this.stopLinkRouting(), e = ["globalRegions", "title", "route"], c = 0, d = e.length; d > c; c++)
                        a = e[c], delete this[a];
                    return k.removeHandlers(this), b.__super__.dispose.apply(this, arguments)
                }
            }, b
        }(i)
    }), a.register("chaplin/views/view", function(b, c, d) {
        var e, f, g, h, i, j, k, l, m, n, o = {}.hasOwnProperty, p = function(a, b) {
            function c() {
                this.constructor = a
            }
            for (var d in b)
                o.call(b, d) && (a[d] = b[d]);
            return c.prototype = b.prototype, a.prototype = new c, a.__super__ = b.prototype, a
        }, q = [].indexOf || function(a) {
            for (var b = 0, c = this.length; c > b; b++)
                if (b in this && this[b] === a)
                    return b;
            return - 1
        };
        n = a("underscore"), f = a("backbone"), k = a("chaplin/mediator"), g = a("chaplin/lib/event_broker"), m = a("chaplin/lib/utils"), e = f.$, j = function() {
            return Function.prototype.bind ? function(a, b) {
                return a.bind(b)
            } : n.bind ? n.bind : void 0
        }(), l = function() {
            return e ? function(a, b) {
                return a.html(b)
            } : function(a, b) {
                return a.innerHTML = b
            }
        }(), i = function() {
            return e ? function(a) {
                var b;
                return b = e(a.container), "function" == typeof a.containerMethod ? a.containerMethod(b, a.el) : b[a.containerMethod](a.el)
            } : function(a) {
                var b;
                return b = "string" == typeof a.container ? document.querySelector(a.container) : a.container, "function" == typeof a.containerMethod ? a.containerMethod(b, a.el) : b[a.containerMethod](a.el)
            }
        }(), d.exports = h = function(a) {
            function b(a) {
                var c, d, f, g, h = this;
                if (a)
                    for (c in a)
                        d = a[c], q.call(this.optionNames, c) >= 0 && (this[c] = d);
                g = this.render, this.render = function() {
                    return h.disposed?!1 : (g.apply(h, arguments), h.autoAttach && h.attach.apply(h, arguments), h)
                }, this.subviews = [], this.subviewsByName = {}, this.noWrap && (this.region && (f = k.execute("region:find", this.region), null != f && (this.el = null != f.instance.container ? null != f.instance.region ? e(f.instance.container).find(f.selector) : f.instance.container : f.instance.$(f.selector))), this.container && (this.el = this.container)), b.__super__.constructor.apply(this, arguments), this.delegateListeners(), this.model && this.listenTo(this.model, "dispose", this.dispose), this.collection && this.listenTo(this.collection, "dispose", function(a) {
                    return a && a !== h.collection ? void 0 : h.dispose()
                }), null != this.regions && k.execute("region:register", this), this.autoRender && this.render()
            }
            return p(b, a), n.extend(b.prototype, g), b.prototype.autoRender=!1, b.prototype.autoAttach=!0, b.prototype.container = null, b.prototype.containerMethod = e ? "append" : "appendChild", b.prototype.regions = null, b.prototype.region = null, b.prototype.stale=!1, b.prototype.noWrap=!1, b.prototype.keepElement=!1, b.prototype.subviews = null, b.prototype.subviewsByName = null, b.prototype.optionNames = ["autoAttach", "autoRender", "container", "containerMethod", "region", "regions", "noWrap"], b.prototype.delegate = function(a, c, d) {
                var e, g, h, i, k, l;
                if (f.View.prototype.delegate)
                    return b.__super__.delegate.apply(this, arguments);
                if ("string" != typeof a)
                    throw new TypeError("View#delegate: first argument must be a string");
                if (2 === arguments.length)
                    i = c;
                else {
                    if (3 !== arguments.length)
                        throw new TypeError("View#delegate: only two or three arguments are allowed");
                    if (l = c, "string" != typeof l)
                        throw new TypeError("View#delegate: second argument must be a string");
                    i = d
                }
                if ("function" != typeof i)
                    throw new TypeError("View#delegate: handler argument must be function");
                return k = function() {
                    var b, c, d, e;
                    for (d = a.split(" "), e = [], b = 0, c = d.length; c > b; b++)
                        g = d[b], e.push("" + g + ".delegate" + this.cid);
                    return e
                }.call(this), h = k.join(" "), e = j(i, this), this.$el.on(h, l || null, e), e
            }, b.prototype._delegateEvents = function(a) {
                var b, c, d, e, g, h, i;
                if (2 === f.View.prototype.delegateEvents.length)
                    return f.View.prototype.delegateEvents.call(this, a, !0);
                for (e in a) {
                    if (i = a[e], d = "function" == typeof i ? i : this[i], !d)
                        throw new Error("Method '" + d + "' does not exist");
                    g = e.match(/^(\S+)\s*(.*)$/), c = "" + g[1] + ".delegateEvents" + this.cid, h = g[2], b = j(d, this), this.$el.on(c, h || null, b)
                }
            }, b.prototype.delegateEvents = function(a, b) {
                var c, d, e, f;
                if (b || this.undelegateEvents(), a)
                    return this._delegateEvents(a);
                for (f = m.getAllPropertyVersions(this, "events"), d = 0, e = f.length; e > d; d++) {
                    if (c = f[d], "function" == typeof c)
                        throw new TypeError("View#delegateEvents: functions are not supported");
                    this._delegateEvents(c)
                }
            }, b.prototype.undelegate = function(a, c, d) {
                var e, g, h, i, j;
                if (f.View.prototype.undelegate)
                    return b.__super__.undelegate.apply(this, arguments);
                if (a) {
                    if ("string" != typeof a)
                        throw new TypeError("View#undelegate: first argument must be a string");
                    if (2 === arguments.length)
                        "string" == typeof c ? j = c : h = c;
                    else if (3 === arguments.length) {
                        if (j = c, "string" != typeof j)
                            throw new TypeError("View#undelegate: second argument must be a string");
                        h = d
                    }
                    return i = function() {
                        var b, c, d, f;
                        for (d = a.split(" "), f = [], b = 0, c = d.length; c > b; b++)
                            e = d[b], f.push("" + e + ".delegate" + this.cid);
                        return f
                    }.call(this), g = i.join(" "), this.$el.off(g, j || null)
                }
                return this.$el.off(".delegate" + this.cid)
            }, b.prototype.delegateListeners = function() {
                var a, b, c, d, e, f, g, h, i;
                if (this.listen)
                    for (h = m.getAllPropertyVersions(this, "listen"), f = 0, g = h.length; g > f; f++) {
                        e = h[f];
                        for (b in e) {
                            if (c = e[b], "function" != typeof c && (c = this[c]), "function" != typeof c)
                                throw new Error("View#delegateListeners: " + ("" + c + " must be function"));
                                i = b.split(" "), a = i[0], d = i[1], this.delegateListener(a, d, c)
                            }
                    }
                }, b.prototype.delegateListener = function(a, b, c) {
                var d;
                "model" === b || "collection" === b ? (d = this[b], d && this.listenTo(d, a, c)) : "mediator" === b ? this.subscribeEvent(a, c) : b || this.on(a, c, this)
            }, b.prototype.registerRegion = function(a, b) {
                return k.execute("region:register", this, a, b)
            }, b.prototype.unregisterRegion = function(a) {
                return k.execute("region:unregister", this, a)
            }, b.prototype.unregisterAllRegions = function() {
                return k.execute({
                    name: "region:unregister",
                    silent: !0
                }, this)
            }, b.prototype.subview = function(a, b) {
                var c, d;
                return d = this.subviews, c = this.subviewsByName, a && b ? (this.removeSubview(a), d.push(b), c[a] = b, b) : a ? c[a] : void 0
            }, b.prototype.removeSubview = function(a) {
                var b, c, d, e, f, g, h;
                if (a) {
                    if (g = this.subviews, b = this.subviewsByName, "string" == typeof a)
                        d = a, h = b[d];
                    else {
                        h = a;
                        for (e in b)
                            if (f = b[e], f === h) {
                                d = e;
                                break
                            }
                    }
                    if (d && h && h.dispose)
                        return h.dispose(), c = m.indexOf(g, h), - 1 !== c && g.splice(c, 1), delete b[d]
                }
            }, b.prototype.getTemplateData = function() {
                var a, b;
                return a = this.model ? m.serialize(this.model) : this.collection ? {
                    items: m.serialize(this.collection),
                    length: this.collection.length
                } : {}, b = this.model || this.collection, b && "function" == typeof b.isSynced&&!("synced"in a) && (a.synced = b.isSynced()), a
            }, b.prototype.getTemplateFunction = function() {
                throw new Error("View#getTemplateFunction must be overridden")
            }, b.prototype.render = function() {
                var a, b, c;
                if (this.disposed)
                    return !1;
                if (c = this.getTemplateFunction(), "function" == typeof c)
                    if (b = c(this.getTemplateData()), this.noWrap) {
                        if (a = document.createElement("div"), a.innerHTML = b, a.children.length > 1)
                            throw new Error("There must be a single top-level element when using `noWrap`.");
                            this.undelegateEvents(), this.setElement(a.firstChild, !0)
                    } else 
                        l(e ? this.$el : this.el, b);
                return this
            }, b.prototype.attach = function() {
                return null != this.region && k.execute("region:show", this.region, this), this.container&&!document.body.contains(this.el) ? (i(this), this.trigger("addedToDOM")) : void 0
            }, b.prototype.disposed=!1, b.prototype.dispose = function() {
                var a, b, c, d, e, f, g, h;
                if (!this.disposed) {
                    for (this.unregisterAllRegions(), h = this.subviews, d = 0, f = h.length; f > d; d++)
                        c = h[d], c.dispose();
                    for (this.unsubscribeAllEvents(), this.off(), this.keepElement ? (this.undelegateEvents(), this.undelegate(), this.stopListening()) : this.remove(), b = ["el", "$el", "options", "model", "collection", "subviews", "subviewsByName", "_callbacks"], e = 0, g = b.length; g > e; e++)
                        a = b[e], delete this[a];
                    return this.disposed=!0, "function" == typeof Object.freeze ? Object.freeze(this) : void 0
                }
            }, b
        }(f.View)
    }), a.register("chaplin/views/collection_view", function(b, c, d) {
        var e, f, g, h, i, j, k, l, m, n, o, p, q = function(a, b) {
            return function() {
                return a.apply(b, arguments)
            }
        }, r = {}.hasOwnProperty, s = function(a, b) {
            function c() {
                this.constructor = a
            }
            for (var d in b)
                r.call(b, d) && (a[d] = b[d]);
            return c.prototype = b.prototype, a.prototype = new c, a.__super__ = b.prototype, a
        };
        p = a("underscore"), f = a("backbone"), h = a("chaplin/views/view"), o = a("chaplin/lib/utils"), e = f.$, k = function(a, b) {
            var c, d, e, g;
            if (!b)
                return a;
            for (g = [], d = 0, e = a.length; e > d; d++)
                c = a[d], f.utils.matchesSelector(c, b) && g.push(c);
            return g
        }, n = function() {
            return e ? function(a, b) {
                return a.toggle(b)
            } : function(a, b) {
                return a.style.display = b ? "" : "none"
            }
        }(), i = function() {
            return e ? function(a, b) {
                return a.addClass(b)
            } : function(a, b) {
                return a.classList.add(b)
            }
        }(), m = function() {
            return e ? function(a, b, c) {
                return b ? i(a, c) : a.css("opacity", 0)
            } : function(a, b, c) {
                return b ? i(a, c) : a.style.opacity = 0
            }
        }(), j = function() {
            return e ? function(a, b) {
                return a.animate({
                    opacity: 1
                }, b)
            } : function(a, b) {
                return a.style.transition = "opacity " + b / 1e3 + "s", a.opacity = 1
            }
        }(), l = function() {
            return e ? function(a, b, c, d, e) {
                var f, g, h, i, j;
                return h = c > 0 && d > c, i = function(a) {
                    return 0 === a || c === a
                }, h || e ? (f = a.children(e), g = f.length, f[c] !== b ? i(g) ? a.append(b) : 0 === c ? f.eq(c).before(b) : f.eq(c - 1).after(b) : void 0) : (j = i(d) ? "append" : "prepend", a[j](b))
            } : function(a, b, c, d, e) {
                var f, g, h, i, j;
                return h = c > 0 && d > c, i = function(a) {
                    return 0 === a || c === a
                }, h || e ? (f = k(a.children, e), g = f.length, f[c] !== b ? i(g) ? a.appendChild(b) : 0 === c ? a.insertBefore(b, f[c]) : (j = f[c - 1], a.lastChild === j ? a.appendChild(b) : a.insertBefore(b, j.nextElementSibling)) : void 0) : i(d) ? a.appendChild(b) : a.insertBefore(b, a.firstChild)
            }
        }(), d.exports = g = function(a) {
            function b() {
                this.renderAllItems = q(this.renderAllItems, this), this.toggleFallback = q(this.toggleFallback, this), this.itemsReset = q(this.itemsReset, this), this.itemRemoved = q(this.itemRemoved, this), this.itemAdded = q(this.itemAdded, this), this.visibleItems = [], b.__super__.constructor.apply(this, arguments)
            }
            return s(b, a), b.prototype.itemView = null, b.prototype.autoRender=!0, b.prototype.renderItems=!0, b.prototype.animationDuration = 500, b.prototype.useCssAnimation=!1, b.prototype.animationStartClass = "animated-item-view", b.prototype.animationEndClass = "animated-item-view-end", b.prototype.listSelector = null, b.prototype.$list = null, b.prototype.fallbackSelector = null, b.prototype.$fallback = null, b.prototype.loadingSelector = null, b.prototype.$loading = null, b.prototype.itemSelector = void 0, b.prototype.filterer = null, b.prototype.filterCallback = function(a, b) {
                return e && a.$el.stop(!0, !0), n(e ? a.$el : a.el, b)
            }, b.prototype.visibleItems = null, b.prototype.optionNames = h.prototype.optionNames.concat(["renderItems", "itemView"]), b.prototype.initialize = function(a) {
                return null == a && (a = {}), this.addCollectionListeners(), null != a.filterer ? this.filter(a.filterer) : void 0
            }, b.prototype.addCollectionListeners = function() {
                return this.listenTo(this.collection, "add", this.itemAdded), this.listenTo(this.collection, "remove", this.itemRemoved), this.listenTo(this.collection, "reset sort", this.itemsReset)
            }, b.prototype.getTemplateData = function() {
                var a;
                return a = {
                    length: this.collection.length
                }, "function" == typeof this.collection.isSynced && (a.synced = this.collection.isSynced()), a
            }, b.prototype.getTemplateFunction = function() {}, b.prototype.render = function() {
                return b.__super__.render.apply(this, arguments), e ? this.$list = this.listSelector ? this.$(this.listSelector) : this.$el : this.list = this.listSelector ? this.find(this.listSelector) : this.el, this.initFallback(), this.initLoadingIndicator(), this.renderItems ? this.renderAllItems() : void 0
            }, b.prototype.itemAdded = function(a, b, c) {
                return this.insertView(a, this.renderItem(a), c.at)
            }, b.prototype.itemRemoved = function(a) {
                return this.removeViewForItem(a)
            }, b.prototype.itemsReset = function() {
                return this.renderAllItems()
            }, b.prototype.initFallback = function() {
                return this.fallbackSelector ? (e ? this.$fallback = this.$(this.fallbackSelector) : this.fallback = this.find(this.fallbackSelector), this.on("visibilityChange", this.toggleFallback), this.listenTo(this.collection, "syncStateChange", this.toggleFallback), this.toggleFallback()) : void 0
            }, b.prototype.toggleFallback = function() {
                var a;
                return a = 0 === this.visibleItems.length && ("function" == typeof this.collection.isSynced ? this.collection.isSynced() : !0), n(e ? this.$fallback : this.fallback, a)
            }, b.prototype.initLoadingIndicator = function() {
                return this.loadingSelector && "function" == typeof this.collection.isSyncing ? (e ? this.$loading = this.$(this.loadingSelector) : this.loading = this.find(this.loadingSelector), this.listenTo(this.collection, "syncStateChange", this.toggleLoadingIndicator), this.toggleLoadingIndicator()) : void 0
            }, b.prototype.toggleLoadingIndicator = function() {
                var a;
                return a = 0 === this.collection.length && this.collection.isSyncing(), n(e ? this.$loading : this.loading, a)
            }, b.prototype.getItemViews = function() {
                var a, b, c, d;
                if (a = {}, this.subviews.length > 0) {
                    d = this.subviewsByName;
                    for (b in d)
                        c = d[b], "itemView:" === b.slice(0, 9) && (a[b.slice(9)] = c)
                }
                return a
            }, b.prototype.filter = function(a, b) {
                var c, d, e, f, g, h, i, j, k = this;
                if (this.filterer = a, b && (this.filterCallback = b), null == b && (b = this.filterCallback), c = function() {
                    var a;
                    if (k.subviews.length > 0)
                        for (a in k.subviewsByName)
                            if ("itemView:" === a.slice(0, 9))
                                return !0;
                    return !1
                }(), c)
                    for (j = this.collection.models, e = h = 0, i = j.length; i > h; e=++h) {
                        if (f = j[e], d = "function" == typeof a ? a(f, e) : !0, g = this.subview("itemView:" + f.cid), !g)
                            throw new Error("CollectionView#filter: no view found for " + f.cid);
                            this.filterCallback(g, d), this.updateVisibleItems(g.model, d, !1)
                    }
                return this.trigger("visibilityChange", this.visibleItems)
            }, b.prototype.renderAllItems = function() {
                var a, b, c, d, e, f, g, h, i, j, k;
                for (d = this.collection.models, this.visibleItems = [], e = {}, g = 0, i = d.length; i > g; g++)
                    c = d[g], f = this.subview("itemView:" + c.cid), f && (e[c.cid] = f);
                k = this.getItemViews();
                for (a in k)
                    r.call(k, a) && (f = k[a], a in e || this.removeSubview("itemView:" + a));
                for (b = h = 0, j = d.length; j > h; b=++h)
                    c = d[b], f = this.subview("itemView:" + c.cid), f ? this.insertView(c, f, b, !1) : this.insertView(c, this.renderItem(c), b);
                return 0 === d.length ? this.trigger("visibilityChange", this.visibleItems) : void 0
            }, b.prototype.renderItem = function(a) {
                var b;
                return b = this.subview("itemView:" + a.cid), b || (b = this.initItemView(a), this.subview("itemView:" + a.cid, b)), b.render(), b
            }, b.prototype.initItemView = function(a) {
                if (this.itemView)
                    return new this.itemView({
                        autoRender: !1,
                        model: a
                    });
                throw new Error("The CollectionView#itemView property must be defined or the initItemView() must be overridden.")
            }, b.prototype.insertView = function(a, b, c, d) {
                var f, g, h, k, n = this;
                return null == d && (d=!0), 0 === this.animationDuration && (d=!1), "number" != typeof c && (c = this.collection.indexOf(a)), g = "function" == typeof this.filterer ? this.filterer(a, c) : !0, f = e ? b.$el : b.el, g && d && m(f, this.useCssAnimation, this.animationStartClass), this.filterer && this.filterCallback(b, g), h = this.collection.length, k = e ? this.$list : this.list, l(k, f, c, h, this.itemSelector), b.trigger("addedToParent"), this.updateVisibleItems(a, g), g && d && (this.useCssAnimation ? setTimeout(function() {
                    return i(f, n.animationEndClass)
                }, 0) : j(f, this.animationDuration)), b
            }, b.prototype.removeViewForItem = function(a) {
                return this.updateVisibleItems(a, !1), this.removeSubview("itemView:" + a.cid)
            }, b.prototype.updateVisibleItems = function(a, b, c) {
                var d, e, f;
                return null == c && (c=!0), e=!1, f = o.indexOf(this.visibleItems, a), d =- 1 !== f, b&&!d ? (this.visibleItems.push(a), e=!0) : !b && d && (this.visibleItems.splice(f, 1), e=!0), e && c && this.trigger("visibilityChange", this.visibleItems), e
            }, b.prototype.dispose = function() {
                var a, c, d, e;
                if (!this.disposed) {
                    for (c = ["$list", "$fallback", "$loading", "visibleItems"], d = 0, e = c.length; e > d; d++)
                        a = c[d], delete this[a];
                    return b.__super__.dispose.apply(this, arguments)
                }
            }, b
        }(h)
    }), a.register("chaplin/lib/route", function(b, c, d) {
        var e, f, g, h, i, j, k = function(a, b) {
            return function() {
                return a.apply(b, arguments)
            }
        }, l = {}.hasOwnProperty;
        j = a("underscore"), e = a("backbone"), g = a("chaplin/lib/event_broker"), f = a("chaplin/controllers/controller"), i = a("chaplin/lib/utils"), d.exports = h = function() {
            function a(a, b, c, d) {
                var e;
                if (this.pattern = a, this.controller = b, this.action = c, this.handler = k(this.handler, this), this.addParamName = k(this.addParamName, this), "string" != typeof this.pattern)
                    throw new Error("Route: RegExps are not supported.        Use strings with :names and `constraints` option of route");
                if (this.options = d ? j.extend({}, d) : {}, null != this.options.name && (this.name = this.options.name), this.name&&-1 !== this.name.indexOf("#"))
                    throw new Error('Route: "#" cannot be used in name');
                if (null == (e = this.name) && (this.name = this.controller + "#" + this.action), this.paramNames = [], this.action in f.prototype)
                    throw new Error("Route: You should not use existing controller properties as action names");
                this.createRegExp(), "function" == typeof Object.freeze && Object.freeze(this)
            }
            var b;
            return a.extend = e.Model.extend, j.extend(a.prototype, g), b = /[-[\]{}()+?.,\\^$|#\s]/g, a.prototype.matches = function(a) {
                var b, c, d, e, f, g, h;
                if ("string" == typeof a)
                    return a === this.name;
                for (d = 0, h = ["name", "action", "controller"], f = 0, g = h.length; g > f; f++)
                    if (c = h[f], d++, e = a[c], e && e !== this[c])
                        return !1;
                return b = 1 === d && ("action" === c || "controller" === c), !b
            }, a.prototype.reverse = function(a, b) {
                var c, d, e, f, g, h, j;
                if (a = this.normalizeParams(a), a===!1)
                    return !1;
                for (e = this.pattern, j = this.paramNames, g = 0, h = j.length; h > g; g++)
                    c = j[g], f = a[c], e = e.replace(RegExp("[:*]" + c, "g"), f);
                return b ? "object" == typeof b ? (d = i.queryParams.stringify(b), e += d ? "?" + d : "") : e += ("?" === b[0] ? "" : "?") + b : e
            }, a.prototype.normalizeParams = function(a) {
                var b, c, d, e, f, g;
                if (i.isArray(a)) {
                    if (a.length < this.paramNames.length)
                        return !1;
                    for (d = {}, g = this.paramNames, b = e = 0, f = g.length; f > e; b=++e)
                        c = g[b], d[c] = a[b];
                    if (!this.testConstraints(d))
                        return !1;
                    a = d
                } else if (null == a && (a = {}), !this.testParams(a))
                    return !1;
                return a
            }, a.prototype.testConstraints = function(a) {
                var b, c, d;
                if (c = this.options.constraints)
                    for (d in c)
                        if (l.call(c, d) && (b = c[d], !b.test(a[d])))
                            return !1;
                return !0
            }, a.prototype.testParams = function(a) {
                var b, c, d, e;
                for (e = this.paramNames, c = 0, d = e.length; d > c; c++)
                    if (b = e[c], void 0 === a[b])
                        return !1;
                return this.testConstraints(a)
            }, a.prototype.createRegExp = function() {
                var a;
                return a = this.pattern.replace(b, "\\$&").replace(/(?::|\*)(\w+)/g, this.addParamName), this.regExp = RegExp("^" + a + "(?=\\?|$)")
            }, a.prototype.addParamName = function(a, b) {
                return this.paramNames.push(b), ":" === a.charAt(0) ? "([^/?]+)" : "(.*?)"
            }, a.prototype.test = function(a) {
                var b, c;
                return c = this.regExp.test(a), c ? (b = this.options.constraints, b ? this.testConstraints(this.extractParams(a)) : !0) : !1
            }, a.prototype.handler = function(a, b) {
                var c, d, e, f, g, h;
                return b = b ? j.extend({}, b) : {}, "object" == typeof a ? (f = i.queryParams.stringify(b.query), d = a, e = this.reverse(d)) : (h = a.split("?"), e = h[0], f = h[1], null == f ? f = "" : b.query = i.queryParams.parse(f), d = this.extractParams(e)), c = j.extend({}, d, this.options.params), g = {
                    path: e,
                    action: this.action,
                    controller: this.controller,
                    name: this.name,
                    query: f
                }, this.publishEvent("router:match", g, c, b)
            }, a.prototype.extractParams = function(a) {
                var b, c, d, e, f, g, h, i;
                for (f = {}, d = this.regExp.exec(a), i = d.slice(1), b = g = 0, h = i.length; h > g; b=++g)
                    c = i[b], e = this.paramNames.length ? this.paramNames[b] : b, f[e] = c;
                return f
            }, a
        }()
    }), a.register("chaplin/lib/router", function(b, c, d) {
        var e, f, g, h, i, j, k, l, m = function(a, b) {
            return function() {
                return a.apply(b, arguments)
            }
        };
        l = a("underscore"), e = a("backbone"), j = a("chaplin/mediator"), f = a("chaplin/lib/event_broker"), g = a("chaplin/lib/history"), h = a("chaplin/lib/route"), k = a("chaplin/lib/utils"), d.exports = i = function() {
            function a(a) {
                var b;
                this.options = null != a ? a : {}, this.match = m(this.match, this), b = "file:" !== window.location.protocol, l.defaults(this.options, {
                    pushState: b,
                    root: "/"
                }), this.removeRoot = new RegExp("^" + k.escapeRegExp(this.options.root) + "(#)?"), this.subscribeEvent("!router:route", this.oldEventError), this.subscribeEvent("!router:routeByName", this.oldEventError), this.subscribeEvent("!router:changeURL", this.oldURLEventError), j.setHandler("router:route", this.route, this), j.setHandler("router:reverse", this.reverse, this), j.setHandler("router:changeURL", this.changeURL, this), this.createHistory()
            }
            return a.extend = e.Model.extend, l.extend(a.prototype, f), a.prototype.oldEventError = function() {
                throw new Error("!router:route and !router:routeByName events were removed.  Use `Chaplin.helpers.redirectTo`")
            }, a.prototype.oldURLEventError = function() {
                throw new Error('!router:changeURL event was removed.  Use mediator.execute("router:changeURL")')
            }, a.prototype.createHistory = function() {
                return e.history = new g
            }, a.prototype.startHistory = function() {
                return e.history.start(this.options)
            }, a.prototype.stopHistory = function() {
                return e.History.started ? e.history.stop() : void 0
            }, a.prototype.findHandler = function(a) {
                var b, c, d, f;
                for (f = e.history.handlers, c = 0, d = f.length; d > c; c++)
                    if (b = f[c], a(b))
                        return b
            }, a.prototype.match = function(a, b, c) {
                var d, f, g, i;
                if (null == c && (c = {}), 2 === arguments.length && "object" == typeof b) {
                    if (c = b, f = c.controller, d = c.action, !f ||!d)
                        throw new Error("Router#match must receive either target or options.controller & options.action")
                } else {
                    if (f = c.controller, d = c.action, f || d)
                        throw new Error("Router#match cannot use both target and options.controller / options.action");
                    i = b.split("#"), f = i[0], d = i[1]
                }
                return g = new h(a, f, d, c), e.history.handlers.push({
                    route: g,
                    callback: g.handler
                }), g
            }, a.prototype.route = function(a, b, c) {
                var d, e;
                if (b = b ? k.isArray(b) ? b.slice() : l.extend({}, b) : {}, "object" == typeof a && (e = a.url), null != e ? (e = e.replace(this.removeRoot, ""), d = this.findHandler(function(a) {
                    return a.route.test(e)
                }), c = b, b = null) : (c = c ? l.extend({}, c) : {}, d = this.findHandler(function(c) {
                    return c.route.matches(a) && (b = c.route.normalizeParams(b))?!0 : !1
                })), d)
                    return l.defaults(c, {
                        changeURL: !0
                    }), d.callback(e || b, c), !0;
                throw new Error("Router#route: request was not routed")
            }, a.prototype.reverse = function(a, b, c) {
                var d, f, g, h, i, j, k;
                if (h = this.options.root, null != b && "object" != typeof b)
                    throw new TypeError("Router#reverse: params must be an array or an object");
                for (f = e.history.handlers, j = 0, k = f.length; k > j; j++)
                    if (d = f[j], d.route.matches(a) && (g = d.route.reverse(b, c), g!==!1))
                        return i = h ? h + g : g;
                throw new Error("Router#reverse: invalid route specified")
            }, a.prototype.changeURL = function(a, b) {
                var c;
                return null == b && (b = {}), c = {
                    trigger: b.trigger===!0,
                    replace: b.replace===!0
                }, e.history.navigate(a, c)
            }, a.prototype.disposed=!1, a.prototype.dispose = function() {
                return this.disposed ? void 0 : (this.stopHistory(), delete e.history, this.unsubscribeAllEvents(), j.removeHandlers(this), this.disposed=!0, "function" == typeof Object.freeze ? Object.freeze(this) : void 0)
            }, a
        }()
    }), a.register("chaplin/lib/history", function(b, c, d) {
        var e, f, g, h, i, j, k, l = {}.hasOwnProperty, m = function(a, b) {
            function c() {
                this.constructor = a
            }
            for (var d in b)
                l.call(b, d) && (a[d] = b[d]);
            return c.prototype = b.prototype, a.prototype = new c, a.__super__ = b.prototype, a
        };
        k = a("underscore"), e = a("backbone"), i = /^[#\/]|\s+$/g, h = /^\/+|\/+$/g, g = /msie [\w.]+/, j = /\/$/, f = function(a) {
            function b() {
                return b.__super__.constructor.apply(this, arguments)
            }
            return m(b, a), b.prototype.getFragment = function(a, b) {
                var c;
                return null == a && (this._hasPushState ||!this._wantsHashChange || b ? (a = this.location.pathname + this.location.search, c = this.root.replace(j, ""), a.indexOf(c) || (a = a.substr(c.length))) : a = this.getHash()), a.replace(i, "")
            }, b.prototype.start = function(a) {
                var b, c, d;
                if (e.History.started)
                    throw new Error("Backbone.history has already been started");
                return e.History.started=!0, this.options = k.extend({}, {
                    root: "/"
                }, this.options, a), this.root = this.options.root, this._wantsHashChange = this.options.hashChange!==!1, this._wantsPushState = Boolean(this.options.pushState), this._hasPushState = Boolean(this.options.pushState && this.history && this.history.pushState), c = this.getFragment(), this.root = ("/" + this.root + "/").replace(h, "/"), this._hasPushState ? e.$(window).on("popstate", this.checkUrl) : this._wantsHashChange && "onhashchange"in window ? e.$(window).on("hashchange", this.checkUrl) : this._wantsHashChange && (this._checkUrlInterval = setInterval(this.checkUrl, this.interval)), this.fragment = c, d = this.location, b = d.pathname.replace(/[^\/]$/, "$&/") === this.root, this._wantsHashChange && this._wantsPushState&&!this._hasPushState&&!b ? (this.fragment = this.getFragment(null, !0), this.location.replace(this.root + "#" + this.fragment), !0) : (this._wantsPushState && this._hasPushState && b && d.hash && (this.fragment = this.getHash().replace(i, ""), this.history.replaceState({}, document.title, this.root + this.fragment)), this.options.silent ? void 0 : this.loadUrl())
            }, b
        }(e.History), d.exports = e.$ ? f : e.History
    }), a.register("chaplin/lib/delayer", function(a, b, c) {
        var d;
        d = {
            setTimeout: function(a, b, c) {
                var d, e, f, g = this;
                return null == (f = this.timeouts) && (this.timeouts = {}), this.clearTimeout(a), e = function() {
                    return delete g.timeouts[a], c()
                }, d = setTimeout(e, b), this.timeouts[a] = d, d
            },
            clearTimeout: function(a) {
                this.timeouts && null != this.timeouts[a] && (clearTimeout(this.timeouts[a]), delete this.timeouts[a])
            },
            clearAllTimeouts: function() {
                var a, b, c;
                if (this.timeouts) {
                    c = this.timeouts;
                    for (b in c)
                        a = c[b], this.clearTimeout(b)
                }
            },
            setInterval: function(a, b, c) {
                var d, e;
                return this.clearInterval(a), null == (e = this.intervals) && (this.intervals = {}), d = setInterval(c, b), this.intervals[a] = d, d
            },
            clearInterval: function(a) {
                this.intervals && this.intervals[a] && (clearInterval(this.intervals[a]), delete this.intervals[a])
            },
            clearAllIntervals: function() {
                var a, b, c;
                if (this.intervals) {
                    c = this.intervals;
                    for (b in c)
                        a = c[b], this.clearInterval(b)
                }
            },
            clearDelayed: function() {
                this.clearAllTimeouts(), this.clearAllIntervals()
            }
        }, "function" == typeof Object.freeze && Object.freeze(d), c.exports = d
    }), a.register("chaplin/lib/event_broker", function(b, c, d) {
        var e, f, g = [].slice;
        f = a("chaplin/mediator"), e = {
            subscribeEvent: function(a, b) {
                if ("string" != typeof a)
                    throw new TypeError("EventBroker#subscribeEvent: type argument must be a string");
                if ("function" != typeof b)
                    throw new TypeError("EventBroker#subscribeEvent: handler argument must be a function");
                return f.unsubscribe(a, b, this), f.subscribe(a, b, this)
            },
            unsubscribeEvent: function(a, b) {
                if ("string" != typeof a)
                    throw new TypeError("EventBroker#unsubscribeEvent: type argument must be a string");
                if ("function" != typeof b)
                    throw new TypeError("EventBroker#unsubscribeEvent: handler argument must be a function");
                return f.unsubscribe(a, b)
            },
            unsubscribeAllEvents: function() {
                return f.unsubscribe(null, null, this)
            },
            publishEvent: function() {
                var a, b;
                if (b = arguments[0], a = 2 <= arguments.length ? g.call(arguments, 1) : [], "string" != typeof b)
                    throw new TypeError("EventBroker#publishEvent: type argument must be a string");
                return f.publish.apply(f, [b].concat(g.call(a)))
            }
        }, "function" == typeof Object.freeze && Object.freeze(e), d.exports = e
    }), a.register("chaplin/lib/support", function(a, b, c) {
        var d;
        d = {
            propertyDescriptors: function() {
                var a;
                if ("function" != typeof Object.defineProperty || "function" != typeof Object.defineProperties)
                    return !1;
                try {
                    return a = {}, Object.defineProperty(a, "foo", {
                        value: "bar"
                    }), "bar" === a.foo
                } catch (b) {
                    return !1
                }
            }()
        }, c.exports = d
    }), a.register("chaplin/lib/composition", function(b, c, d) {
        var e, f, g, h, i, j = {}.hasOwnProperty;
        i = a("underscore"), e = a("backbone"), g = a("chaplin/lib/event_broker"), h = Object.prototype.hasOwnProperty, d.exports = f = function() {
            function a(a) {
                null != a && (this.options = i.extend({}, a)), this.item = this, this.initialize(this.options)
            }
            return a.extend = e.Model.extend, i.extend(a.prototype, e.Events), i.extend(a.prototype, g), a.prototype.item = null, a.prototype.options = null, a.prototype._stale=!1, a.prototype.initialize = function() {}, a.prototype.compose = function() {}, a.prototype.check = function(a) {
                return i.isEqual(this.options, a)
            }, a.prototype.stale = function(a) {
                var b, c;
                if (null == a)
                    return this._stale;
                this._stale = a;
                for (c in this)
                    b = this[c], b && b !== this && "object" == typeof b && h.call(b, "stale") && (b.stale = a)
            }, a.prototype.disposed=!1, a.prototype.dispose = function() {
                var a, b, c, d, e;
                if (!this.disposed) {
                    for (b in this)
                        j.call(this, b) && (a = this[b], a && "function" == typeof a.dispose && a !== this && (a.dispose(), delete this[b]));
                    for (this.unsubscribeAllEvents(), this.stopListening(), c = ["redirected"], d = 0, e = c.length; e > d; d++)
                        b = c[d], delete this[b];
                    return this.disposed=!0, "function" == typeof Object.freeze ? Object.freeze(this) : void 0
                }
            }, a
        }()
    }), a.register("chaplin/lib/sync_machine", function(a, b, c) {
        var d, e, f, g, h, i, j, k, l, m;
        for (h = "unsynced", f = "syncing", e = "synced", d = "syncStateChange", g = {
            _syncState: h,
            _previousSyncState: null,
            syncState: function() {
                return this._syncState
            },
            isUnsynced: function() {
                return this._syncState === h
            },
            isSynced: function() {
                return this._syncState === e
            },
            isSyncing: function() {
                return this._syncState === f
            },
            unsync: function() {
                var a;
                ((a = this._syncState) === f || a === e) && (this._previousSync = this._syncState, this._syncState = h, this.trigger(this._syncState, this, this._syncState), this.trigger(d, this, this._syncState))
            },
            beginSync: function() {
                var a;
                ((a = this._syncState) === h || a === e) && (this._previousSync = this._syncState, this._syncState = f, this.trigger(this._syncState, this, this._syncState), this.trigger(d, this, this._syncState))
            },
            finishSync: function() {
                this._syncState === f && (this._previousSync = this._syncState, this._syncState = e, this.trigger(this._syncState, this, this._syncState), this.trigger(d, this, this._syncState))
            },
            abortSync: function() {
                this._syncState === f && (this._syncState = this._previousSync, this._previousSync = this._syncState, this.trigger(this._syncState, this, this._syncState), this.trigger(d, this, this._syncState))
            }
        }, m = [h, f, e, d], j = function(a) {
            return g[a] = function(b, c) {
                return null == c && (c = this), this.on(a, b, c), this._syncState === a ? b.call(c) : void 0
            }
        }, k = 0, l = m.length; l > k; k++)
            i = m[k], j(i);
        "function" == typeof Object.freeze && Object.freeze(g), c.exports = g
    }), a.register("chaplin/lib/utils", function(b, c, d) {
        var e, f, g, h = [].slice, i = [].indexOf || function(a) {
            for (var b = 0, c = this.length; c > b; b++)
                if (b in this && this[b] === a)
                    return b;
            return - 1
        }, j = {}.hasOwnProperty;
        g = a("underscore"), e = a("chaplin/lib/support"), f = {
            beget: function() {
                var a;
                return "function" == typeof Object.create ? Object.create : (a = function() {}, function(b) {
                    return a.prototype = b, new a
                })
            }(),
            indexOf: function() {
                return Array.prototype.indexOf ? function(a, b) {
                    return a.indexOf(b)
                } : g.indexOf ? g.indexOf : void 0
            }(),
            isArray: Array.isArray || g.isArray,
            serialize: function(a) {
                if ("function" == typeof a.serialize)
                    return a.serialize();
                if ("function" == typeof a.toJSON)
                    return a.toJSON();
                throw new TypeError("utils.serialize: Unknown data was passed")
            },
            readonly: function() {
                var a;
                return e.propertyDescriptors ? (a = {
                    writable: !1,
                    enumerable: !0,
                    configurable: !1
                }, function() {
                    var b, c, d, e, f;
                    for (b = arguments[0], d = 2 <= arguments.length ? h.call(arguments, 1) : [], e = 0, f = d.length; f > e; e++)
                        c = d[e], a.value = b[c], Object.defineProperty(b, c, a);
                    return !0
                }) : function() {
                    return !1
                }
            }(),
            getPrototypeChain: function(a) {
                var b, c, d, e;
                for (b = [a.constructor.prototype]; a = null != (c = null != (d = a.constructor) ? d.__super__ : void 0) ? c : null != (e = a.constructor) ? e.superclass : void 0;)
                    b.push(a);
                return b.reverse()
            },
            getAllPropertyVersions: function(a, b) {
                var c, d, e, g, h, j;
                for (d = [], j = f.getPrototypeChain(a), g = 0, h = j.length; h > g; g++)
                    c = j[g], e = c[b], e && i.call(d, e) < 0 && d.push(e);
                return d
            },
            upcase: function(a) {
                return a.charAt(0).toUpperCase() + a.substring(1)
            },
            escapeRegExp: function(a) {
                return String(a || "").replace(/([.*+?^=!:${}()|[\]\/\\])/g, "\\$1")
            },
            modifierKeyPressed: function(a) {
                return a.shiftKey || a.altKey || a.ctrlKey || a.metaKey
            },
            queryParams: {
                stringify: function(a) {
                    var b, c, d, e, g, h, i, k;
                    e = "", g = function(a, b) {
                        return null != b ? "&" + a + "=" + encodeURIComponent(b) : ""
                    };
                    for (d in a)
                        if (j.call(a, d))
                            if (h = a[d], c = encodeURIComponent(d), f.isArray(h))
                                for (i = 0, k = h.length; k > i; i++)
                                    b = h[i], e += g(c, b);
                            else 
                                e += g(c, h);
                    return e && e.substring(1)
                },
                parse: function(a) {
                    var b, c, d, e, f, g, h, i, j;
                    if (f = {}, !a)
                        return f;
                    for (e = a.split("&"), h = 0, i = e.length; i > h; h++)
                        d = e[h], d.length && (j = d.split("="), c = j[0], g = j[1], c.length && (c = decodeURIComponent(c), g = decodeURIComponent(g), b = f[c], b ? b.push ? b.push(g) : f[c] = [b, g] : f[c] = g));
                    return f
                }
            }
        }, "function" == typeof Object.seal && Object.seal(f), d.exports = f
    }), a.register("chaplin/lib/helpers", function(b, c, d) {
        var e, f;
        f = a("chaplin/mediator"), e = {
            reverse: function(a, b, c) {
                return f.execute("router:reverse", a, b, c)
            },
            redirectTo: function(a, b, c) {
                return f.execute("router:route", a, b, c)
            }
        }, d.exports = e
    }), a.register("chaplin", function(b, c, d) {
        d.exports = {
            Application: a("chaplin/application"),
            mediator: a("chaplin/mediator"),
            Dispatcher: a("chaplin/dispatcher"),
            Controller: a("chaplin/controllers/controller"),
            Composer: a("chaplin/composer"),
            Composition: a("chaplin/lib/composition"),
            Collection: a("chaplin/models/collection"),
            Model: a("chaplin/models/model"),
            Layout: a("chaplin/views/layout"),
            View: a("chaplin/views/view"),
            CollectionView: a("chaplin/views/collection_view"),
            Route: a("chaplin/lib/route"),
            Router: a("chaplin/lib/router"),
            Delayer: a("chaplin/lib/delayer"),
            EventBroker: a("chaplin/lib/event_broker"),
            helpers: a("chaplin/lib/helpers"),
            support: a("chaplin/lib/support"),
            SyncMachine: a("chaplin/lib/sync_machine"),
            utils: a("chaplin/lib/utils")
        }
    });
    var b = function(b, c) {
        a.register("backbone", function(a, c, d) {
            d.exports = b
        }), a.register("underscore", function(a, b, d) {
            d.exports = c
        })
    };
    if ("function" == typeof define && define.amd)
        define("chaplin", ["backbone", "underscore"], function(c, d) {
            return b(c, d), a("chaplin")
        });
    else if ("object" == typeof module && module && module.exports)
        b(require("backbone"), require("underscore")), module.exports = a("chaplin");
    else {
        if ("function" != typeof require)
            throw new Error("Chaplin requires Common.js or AMD modules");
        b(window.Backbone, window._ || window.Backbone.utils), window.Chaplin = a("chaplin")
    }
}(), function() {
    define("lib/session", [], function() {
        return {}
    })
}.call(this);
var Handlebars = {};
if (function(a, b) {
    a.VERSION = "1.0.0", a.COMPILER_REVISION = 4, a.REVISION_CHANGES = {
        1: "<= 1.0.rc.2",
        2: "== 1.0.0-rc.3",
        3: "== 1.0.0-rc.4",
        4: ">= 1.0.0"
    }, a.helpers = {}, a.partials = {};
    var c = Object.prototype.toString, d = "[object Function]", e = "[object Object]";
    a.registerHelper = function(b, d, f) {
        if (c.call(b) === e) {
            if (f || d)
                throw new a.Exception("Arg not supported with multiple helpers");
            a.Utils.extend(this.helpers, b)
        } else 
            f && (d.not = f), this.helpers[b] = d
    }, a.registerPartial = function(b, d) {
        c.call(b) === e ? a.Utils.extend(this.partials, b) : this.partials[b] = d
    }, a.registerHelper("helperMissing", function(a) {
        if (2 === arguments.length)
            return b;
        throw new Error("Missing helper: '" + a + "'")
    }), a.registerHelper("blockHelperMissing", function(b, e) {
        var f = e.inverse || function() {}, g = e.fn, h = c.call(b);
        return h === d && (b = b.call(this)), b===!0 ? g(this) : b===!1 || null == b ? f(this) : "[object Array]" === h ? b.length > 0 ? a.helpers.each(b, e) : f(this) : g(b)
    }), a.K = function() {}, a.createFrame = Object.create || function(b) {
        a.K.prototype = b;
        var c = new a.K;
        return a.K.prototype = null, c
    }, a.logger = {
        DEBUG: 0,
        INFO: 1,
        WARN: 2,
        ERROR: 3,
        level: 3,
        methodMap: {
            0: "debug",
            1: "info",
            2: "warn",
            3: "error"
        },
        log: function(b, c) {
            if (a.logger.level <= b) {
                var d = a.logger.methodMap[b];
                "undefined" != typeof console && console[d] && console[d].call(console, c)
            }
        }
    }, a.log = function(b, c) {
        a.logger.log(b, c)
    }, a.registerHelper("each", function(b, e) {
        var f, g = e.fn, h = e.inverse, i = 0, j = "", k = c.call(b);
        if (k === d && (b = b.call(this)), e.data && (f = a.createFrame(e.data)), b && "object" == typeof b)
            if (b instanceof Array)
                for (var l = b.length; l > i; i++)
                    f && (f.index = i), j += g(b[i], {
                        data: f
                    });
            else 
                for (var m in b)
                    b.hasOwnProperty(m) && (f && (f.key = m), j += g(b[m], {
                        data: f
                    }), i++);
        return 0 === i && (j = h(this)), j
    }), a.registerHelper("if", function(b, e) {
        var f = c.call(b);
        return f === d && (b = b.call(this)), !b || a.Utils.isEmpty(b) ? e.inverse(this) : e.fn(this)
    }), a.registerHelper("unless", function(b, c) {
        return a.helpers["if"].call(this, b, {
            fn: c.inverse,
            inverse: c.fn
        })
    }), a.registerHelper("with", function(b, e) {
        var f = c.call(b);
        return f === d && (b = b.call(this)), a.Utils.isEmpty(b) ? void 0 : e.fn(b)
    }), a.registerHelper("log", function(b, c) {
        var d = c.data && null != c.data.level ? parseInt(c.data.level, 10): 1;
        a.log(d, b)
    });
    var f = function() {
        function a() {
            this.yy = {}
        }
        var b = {
            trace: function() {},
            yy: {},
            symbols_: {
                error: 2,
                root: 3,
                program: 4,
                EOF: 5,
                simpleInverse: 6,
                statements: 7,
                statement: 8,
                openInverse: 9,
                closeBlock: 10,
                openBlock: 11,
                mustache: 12,
                partial: 13,
                CONTENT: 14,
                COMMENT: 15,
                OPEN_BLOCK: 16,
                inMustache: 17,
                CLOSE: 18,
                OPEN_INVERSE: 19,
                OPEN_ENDBLOCK: 20,
                path: 21,
                OPEN: 22,
                OPEN_UNESCAPED: 23,
                CLOSE_UNESCAPED: 24,
                OPEN_PARTIAL: 25,
                partialName: 26,
                params: 27,
                hash: 28,
                dataName: 29,
                param: 30,
                STRING: 31,
                INTEGER: 32,
                BOOLEAN: 33,
                hashSegments: 34,
                hashSegment: 35,
                ID: 36,
                EQUALS: 37,
                DATA: 38,
                pathSegments: 39,
                SEP: 40,
                $accept: 0,
                $end: 1
            },
            terminals_: {
                2: "error",
                5: "EOF",
                14: "CONTENT",
                15: "COMMENT",
                16: "OPEN_BLOCK",
                18: "CLOSE",
                19: "OPEN_INVERSE",
                20: "OPEN_ENDBLOCK",
                22: "OPEN",
                23: "OPEN_UNESCAPED",
                24: "CLOSE_UNESCAPED",
                25: "OPEN_PARTIAL",
                31: "STRING",
                32: "INTEGER",
                33: "BOOLEAN",
                36: "ID",
                37: "EQUALS",
                38: "DATA",
                40: "SEP"
            },
            productions_: [0, [3, 2], [4, 2], [4, 3], [4, 2], [4, 1], [4, 1], [4, 0], [7, 1], [7, 2], [8, 3], [8, 3], [8, 1], [8, 1], [8, 1], [8, 1], [11, 3], [9, 3], [10, 3], [12, 3], [12, 3], [13, 3], [13, 4], [6, 2], [17, 3], [17, 2], [17, 2], [17, 1], [17, 1], [27, 2], [27, 1], [30, 1], [30, 1], [30, 1], [30, 1], [30, 1], [28, 1], [34, 2], [34, 1], [35, 3], [35, 3], [35, 3], [35, 3], [35, 3], [26, 1], [26, 1], [26, 1], [29, 2], [21, 1], [39, 3], [39, 1]],
            performAction: function(a, b, c, d, e, f) {
                var g = f.length - 1;
                switch (e) {
                case 1:
                    return f[g - 1];
                case 2:
                    this.$ = new d.ProgramNode([], f[g]);
                    break;
                case 3:
                    this.$ = new d.ProgramNode(f[g - 2], f[g]);
                    break;
                case 4:
                    this.$ = new d.ProgramNode(f[g - 1], []);
                    break;
                case 5:
                    this.$ = new d.ProgramNode(f[g]);
                    break;
                case 6:
                    this.$ = new d.ProgramNode([], []);
                    break;
                case 7:
                    this.$ = new d.ProgramNode([]);
                    break;
                case 8:
                    this.$ = [f[g]];
                    break;
                case 9:
                    f[g - 1].push(f[g]), this.$ = f[g - 1];
                    break;
                case 10:
                    this.$ = new d.BlockNode(f[g - 2], f[g - 1].inverse, f[g - 1], f[g]);
                    break;
                case 11:
                    this.$ = new d.BlockNode(f[g - 2], f[g - 1], f[g - 1].inverse, f[g]);
                    break;
                case 12:
                    this.$ = f[g];
                    break;
                case 13:
                    this.$ = f[g];
                    break;
                case 14:
                    this.$ = new d.ContentNode(f[g]);
                    break;
                case 15:
                    this.$ = new d.CommentNode(f[g]);
                    break;
                case 16:
                    this.$ = new d.MustacheNode(f[g - 1][0], f[g - 1][1]);
                    break;
                case 17:
                    this.$ = new d.MustacheNode(f[g - 1][0], f[g - 1][1]);
                    break;
                case 18:
                    this.$ = f[g - 1];
                    break;
                case 19:
                    this.$ = new d.MustacheNode(f[g - 1][0], f[g - 1][1], "&" === f[g - 2][2]);
                    break;
                case 20:
                    this.$ = new d.MustacheNode(f[g - 1][0], f[g - 1][1], !0);
                    break;
                case 21:
                    this.$ = new d.PartialNode(f[g - 1]);
                    break;
                case 22:
                    this.$ = new d.PartialNode(f[g - 2], f[g - 1]);
                    break;
                case 23:
                    break;
                case 24:
                    this.$ = [[f[g - 2]].concat(f[g - 1]), f[g]];
                    break;
                case 25:
                    this.$ = [[f[g - 1]].concat(f[g]), null];
                    break;
                case 26:
                    this.$ = [[f[g - 1]], f[g]];
                    break;
                case 27:
                    this.$ = [[f[g]], null];
                    break;
                case 28:
                    this.$ = [[f[g]], null];
                    break;
                case 29:
                    f[g - 1].push(f[g]), this.$ = f[g - 1];
                    break;
                case 30:
                    this.$ = [f[g]];
                    break;
                case 31:
                    this.$ = f[g];
                    break;
                case 32:
                    this.$ = new d.StringNode(f[g]);
                    break;
                case 33:
                    this.$ = new d.IntegerNode(f[g]);
                    break;
                case 34:
                    this.$ = new d.BooleanNode(f[g]);
                    break;
                case 35:
                    this.$ = f[g];
                    break;
                case 36:
                    this.$ = new d.HashNode(f[g]);
                    break;
                case 37:
                    f[g - 1].push(f[g]), this.$ = f[g - 1];
                    break;
                case 38:
                    this.$ = [f[g]];
                    break;
                case 39:
                    this.$ = [f[g - 2], f[g]];
                    break;
                case 40:
                    this.$ = [f[g - 2], new d.StringNode(f[g])];
                    break;
                case 41:
                    this.$ = [f[g - 2], new d.IntegerNode(f[g])];
                    break;
                case 42:
                    this.$ = [f[g - 2], new d.BooleanNode(f[g])];
                    break;
                case 43:
                    this.$ = [f[g - 2], f[g]];
                    break;
                case 44:
                    this.$ = new d.PartialNameNode(f[g]);
                    break;
                case 45:
                    this.$ = new d.PartialNameNode(new d.StringNode(f[g]));
                    break;
                case 46:
                    this.$ = new d.PartialNameNode(new d.IntegerNode(f[g]));
                    break;
                case 47:
                    this.$ = new d.DataNode(f[g]);
                    break;
                case 48:
                    this.$ = new d.IdNode(f[g]);
                    break;
                case 49:
                    f[g - 2].push({
                        part: f[g],
                        separator: f[g - 1]
                    }), this.$ = f[g - 2];
                    break;
                case 50:
                    this.$ = [{
                        part: f[g]
                    }
                    ]
                }
            },
            table: [{
                3: 1,
                4: 2,
                5: [2, 7],
                6: 3,
                7: 4,
                8: 6,
                9: 7,
                11: 8,
                12: 9,
                13: 10,
                14: [1, 11],
                15: [1, 12],
                16: [1, 13],
                19: [1, 5],
                22: [1, 14],
                23: [1, 15],
                25: [1, 16]
            }, {
                1: [3]
            }, {
                5: [1, 17]
            }, {
                5: [2, 6],
                7: 18,
                8: 6,
                9: 7,
                11: 8,
                12: 9,
                13: 10,
                14: [1, 11],
                15: [1, 12],
                16: [1, 13],
                19: [1, 19],
                20: [2, 6],
                22: [1, 14],
                23: [1, 15],
                25: [1, 16]
            }, {
                5: [2, 5],
                6: 20,
                8: 21,
                9: 7,
                11: 8,
                12: 9,
                13: 10,
                14: [1, 11],
                15: [1, 12],
                16: [1, 13],
                19: [1, 5],
                20: [2, 5],
                22: [1, 14],
                23: [1, 15],
                25: [1, 16]
            }, {
                17: 23,
                18: [1, 22],
                21: 24,
                29: 25,
                36: [1, 28],
                38: [1, 27],
                39: 26
            }, {
                5: [2, 8],
                14: [2, 8],
                15: [2, 8],
                16: [2, 8],
                19: [2, 8],
                20: [2, 8],
                22: [2, 8],
                23: [2, 8],
                25: [2, 8]
            }, {
                4: 29,
                6: 3,
                7: 4,
                8: 6,
                9: 7,
                11: 8,
                12: 9,
                13: 10,
                14: [1, 11],
                15: [1, 12],
                16: [1, 13],
                19: [1, 5],
                20: [2, 7],
                22: [1, 14],
                23: [1, 15],
                25: [1, 16]
            }, {
                4: 30,
                6: 3,
                7: 4,
                8: 6,
                9: 7,
                11: 8,
                12: 9,
                13: 10,
                14: [1, 11],
                15: [1, 12],
                16: [1, 13],
                19: [1, 5],
                20: [2, 7],
                22: [1, 14],
                23: [1, 15],
                25: [1, 16]
            }, {
                5: [2, 12],
                14: [2, 12],
                15: [2, 12],
                16: [2, 12],
                19: [2, 12],
                20: [2, 12],
                22: [2, 12],
                23: [2, 12],
                25: [2, 12]
            }, {
                5: [2, 13],
                14: [2, 13],
                15: [2, 13],
                16: [2, 13],
                19: [2, 13],
                20: [2, 13],
                22: [2, 13],
                23: [2, 13],
                25: [2, 13]
            }, {
                5: [2, 14],
                14: [2, 14],
                15: [2, 14],
                16: [2, 14],
                19: [2, 14],
                20: [2, 14],
                22: [2, 14],
                23: [2, 14],
                25: [2, 14]
            }, {
                5: [2, 15],
                14: [2, 15],
                15: [2, 15],
                16: [2, 15],
                19: [2, 15],
                20: [2, 15],
                22: [2, 15],
                23: [2, 15],
                25: [2, 15]
            }, {
                17: 31,
                21: 24,
                29: 25,
                36: [1, 28],
                38: [1, 27],
                39: 26
            }, {
                17: 32,
                21: 24,
                29: 25,
                36: [1, 28],
                38: [1, 27],
                39: 26
            }, {
                17: 33,
                21: 24,
                29: 25,
                36: [1, 28],
                38: [1, 27],
                39: 26
            }, {
                21: 35,
                26: 34,
                31: [1, 36],
                32: [1, 37],
                36: [1, 28],
                39: 26
            }, {
                1: [2, 1]
            }, {
                5: [2, 2],
                8: 21,
                9: 7,
                11: 8,
                12: 9,
                13: 10,
                14: [1, 11],
                15: [1, 12],
                16: [1, 13],
                19: [1, 19],
                20: [2, 2],
                22: [1, 14],
                23: [1, 15],
                25: [1, 16]
            }, {
                17: 23,
                21: 24,
                29: 25,
                36: [1, 28],
                38: [1, 27],
                39: 26
            }, {
                5: [2, 4],
                7: 38,
                8: 6,
                9: 7,
                11: 8,
                12: 9,
                13: 10,
                14: [1, 11],
                15: [1, 12],
                16: [1, 13],
                19: [1, 19],
                20: [2, 4],
                22: [1, 14],
                23: [1, 15],
                25: [1, 16]
            }, {
                5: [2, 9],
                14: [2, 9],
                15: [2, 9],
                16: [2, 9],
                19: [2, 9],
                20: [2, 9],
                22: [2, 9],
                23: [2, 9],
                25: [2, 9]
            }, {
                5: [2, 23],
                14: [2, 23],
                15: [2, 23],
                16: [2, 23],
                19: [2, 23],
                20: [2, 23],
                22: [2, 23],
                23: [2, 23],
                25: [2, 23]
            }, {
                18: [1, 39]
            }, {
                18: [2, 27],
                21: 44,
                24: [2, 27],
                27: 40,
                28: 41,
                29: 48,
                30: 42,
                31: [1, 45],
                32: [1, 46],
                33: [1, 47],
                34: 43,
                35: 49,
                36: [1, 50],
                38: [1, 27],
                39: 26
            }, {
                18: [2, 28],
                24: [2, 28]
            }, {
                18: [2, 48],
                24: [2, 48],
                31: [2, 48],
                32: [2, 48],
                33: [2, 48],
                36: [2, 48],
                38: [2, 48],
                40: [1, 51]
            }, {
                21: 52,
                36: [1, 28],
                39: 26
            }, {
                18: [2, 50],
                24: [2, 50],
                31: [2, 50],
                32: [2, 50],
                33: [2, 50],
                36: [2, 50],
                38: [2, 50],
                40: [2, 50]
            }, {
                10: 53,
                20: [1, 54]
            }, {
                10: 55,
                20: [1, 54]
            }, {
                18: [1, 56]
            }, {
                18: [1, 57]
            }, {
                24: [1, 58]
            }, {
                18: [1, 59],
                21: 60,
                36: [1, 28],
                39: 26
            }, {
                18: [2, 44],
                36: [2, 44]
            }, {
                18: [2, 45],
                36: [2, 45]
            }, {
                18: [2, 46],
                36: [2, 46]
            }, {
                5: [2, 3],
                8: 21,
                9: 7,
                11: 8,
                12: 9,
                13: 10,
                14: [1, 11],
                15: [1, 12],
                16: [1, 13],
                19: [1, 19],
                20: [2, 3],
                22: [1, 14],
                23: [1, 15],
                25: [1, 16]
            }, {
                14: [2, 17],
                15: [2, 17],
                16: [2, 17],
                19: [2, 17],
                20: [2, 17],
                22: [2, 17],
                23: [2, 17],
                25: [2, 17]
            }, {
                18: [2, 25],
                21: 44,
                24: [2, 25],
                28: 61,
                29: 48,
                30: 62,
                31: [1, 45],
                32: [1, 46],
                33: [1, 47],
                34: 43,
                35: 49,
                36: [1, 50],
                38: [1, 27],
                39: 26
            }, {
                18: [2, 26],
                24: [2, 26]
            }, {
                18: [2, 30],
                24: [2, 30],
                31: [2, 30],
                32: [2, 30],
                33: [2, 30],
                36: [2, 30],
                38: [2, 30]
            }, {
                18: [2, 36],
                24: [2, 36],
                35: 63,
                36: [1, 64]
            }, {
                18: [2, 31],
                24: [2, 31],
                31: [2, 31],
                32: [2, 31],
                33: [2, 31],
                36: [2, 31],
                38: [2, 31]
            }, {
                18: [2, 32],
                24: [2, 32],
                31: [2, 32],
                32: [2, 32],
                33: [2, 32],
                36: [2, 32],
                38: [2, 32]
            }, {
                18: [2, 33],
                24: [2, 33],
                31: [2, 33],
                32: [2, 33],
                33: [2, 33],
                36: [2, 33],
                38: [2, 33]
            }, {
                18: [2, 34],
                24: [2, 34],
                31: [2, 34],
                32: [2, 34],
                33: [2, 34],
                36: [2, 34],
                38: [2, 34]
            }, {
                18: [2, 35],
                24: [2, 35],
                31: [2, 35],
                32: [2, 35],
                33: [2, 35],
                36: [2, 35],
                38: [2, 35]
            }, {
                18: [2, 38],
                24: [2, 38],
                36: [2, 38]
            }, {
                18: [2, 50],
                24: [2, 50],
                31: [2, 50],
                32: [2, 50],
                33: [2, 50],
                36: [2, 50],
                37: [1, 65],
                38: [2, 50],
                40: [2, 50]
            }, {
                36: [1, 66]
            }, {
                18: [2, 47],
                24: [2, 47],
                31: [2, 47],
                32: [2, 47],
                33: [2, 47],
                36: [2, 47],
                38: [2, 47]
            }, {
                5: [2, 10],
                14: [2, 10],
                15: [2, 10],
                16: [2, 10],
                19: [2, 10],
                20: [2, 10],
                22: [2, 10],
                23: [2, 10],
                25: [2, 10]
            }, {
                21: 67,
                36: [1, 28],
                39: 26
            }, {
                5: [2, 11],
                14: [2, 11],
                15: [2, 11],
                16: [2, 11],
                19: [2, 11],
                20: [2, 11],
                22: [2, 11],
                23: [2, 11],
                25: [2, 11]
            }, {
                14: [2, 16],
                15: [2, 16],
                16: [2, 16],
                19: [2, 16],
                20: [2, 16],
                22: [2, 16],
                23: [2, 16],
                25: [2, 16]
            }, {
                5: [2, 19],
                14: [2, 19],
                15: [2, 19],
                16: [2, 19],
                19: [2, 19],
                20: [2, 19],
                22: [2, 19],
                23: [2, 19],
                25: [2, 19]
            }, {
                5: [2, 20],
                14: [2, 20],
                15: [2, 20],
                16: [2, 20],
                19: [2, 20],
                20: [2, 20],
                22: [2, 20],
                23: [2, 20],
                25: [2, 20]
            }, {
                5: [2, 21],
                14: [2, 21],
                15: [2, 21],
                16: [2, 21],
                19: [2, 21],
                20: [2, 21],
                22: [2, 21],
                23: [2, 21],
                25: [2, 21]
            }, {
                18: [1, 68]
            }, {
                18: [2, 24],
                24: [2, 24]
            }, {
                18: [2, 29],
                24: [2, 29],
                31: [2, 29],
                32: [2, 29],
                33: [2, 29],
                36: [2, 29],
                38: [2, 29]
            }, {
                18: [2, 37],
                24: [2, 37],
                36: [2, 37]
            }, {
                37: [1, 65]
            }, {
                21: 69,
                29: 73,
                31: [1, 70],
                32: [1, 71],
                33: [1, 72],
                36: [1, 28],
                38: [1, 27],
                39: 26
            }, {
                18: [2, 49],
                24: [2, 49],
                31: [2, 49],
                32: [2, 49],
                33: [2, 49],
                36: [2, 49],
                38: [2, 49],
                40: [2, 49]
            }, {
                18: [1, 74]
            }, {
                5: [2, 22],
                14: [2, 22],
                15: [2, 22],
                16: [2, 22],
                19: [2, 22],
                20: [2, 22],
                22: [2, 22],
                23: [2, 22],
                25: [2, 22]
            }, {
                18: [2, 39],
                24: [2, 39],
                36: [2, 39]
            }, {
                18: [2, 40],
                24: [2, 40],
                36: [2, 40]
            }, {
                18: [2, 41],
                24: [2, 41],
                36: [2, 41]
            }, {
                18: [2, 42],
                24: [2, 42],
                36: [2, 42]
            }, {
                18: [2, 43],
                24: [2, 43],
                36: [2, 43]
            }, {
                5: [2, 18],
                14: [2, 18],
                15: [2, 18],
                16: [2, 18],
                19: [2, 18],
                20: [2, 18],
                22: [2, 18],
                23: [2, 18],
                25: [2, 18]
            }
            ],
            defaultActions: {
                17: [2, 1]
            },
            parseError: function(a) {
                throw new Error(a)
            },
            parse: function(a) {
                function b() {
                    var a;
                    return a = c.lexer.lex() || 1, "number" != typeof a && (a = c.symbols_[a] || a), a
                }
                var c = this, d = [0], e = [null], f = [], g = this.table, h = "", i = 0, j = 0, k = 0;
                this.lexer.setInput(a), this.lexer.yy = this.yy, this.yy.lexer = this.lexer, this.yy.parser = this, "undefined" == typeof this.lexer.yylloc && (this.lexer.yylloc = {});
                var l = this.lexer.yylloc;
                f.push(l);
                var m = this.lexer.options && this.lexer.options.ranges;
                "function" == typeof this.yy.parseError && (this.parseError = this.yy.parseError);
                for (var n, o, p, q, r, s, t, u, v, w = {}; ;) {
                    if (p = d[d.length - 1], this.defaultActions[p] ? q = this.defaultActions[p] : ((null === n || "undefined" == typeof n) && (n = b()), q = g[p] && g[p][n]), "undefined" == typeof q ||!q.length ||!q[0]) {
                        var x = "";
                        if (!k) {
                            v = [];
                            for (s in g[p])
                                this.terminals_[s] && s > 2 && v.push("'" + this.terminals_[s] + "'");
                            x = this.lexer.showPosition ? "Parse error on line " + (i + 1) + ":\n" + this.lexer.showPosition() + "\nExpecting " + v.join(", ") + ", got '" + (this.terminals_[n] || n) + "'" : "Parse error on line " + (i + 1) + ": Unexpected " + (1 == n ? "end of input" : "'" + (this.terminals_[n] || n) + "'"), this.parseError(x, {
                                text: this.lexer.match,
                                token: this.terminals_[n] || n,
                                line: this.lexer.yylineno,
                                loc: l,
                                expected: v
                            })
                        }
                    }
                    if (q[0]instanceof Array && q.length > 1)
                        throw new Error("Parse Error: multiple actions possible at state: " + p + ", token: " + n);
                    switch (q[0]) {
                    case 1:
                        d.push(n), e.push(this.lexer.yytext), f.push(this.lexer.yylloc), d.push(q[1]), n = null, o ? (n = o, o = null) : (j = this.lexer.yyleng, h = this.lexer.yytext, i = this.lexer.yylineno, l = this.lexer.yylloc, k > 0 && k--);
                        break;
                    case 2:
                        if (t = this.productions_[q[1]][1], w.$ = e[e.length - t], w._$ = {
                            first_line: f[f.length - (t || 1)].first_line,
                            last_line: f[f.length - 1].last_line,
                            first_column: f[f.length - (t || 1)].first_column,
                            last_column: f[f.length - 1].last_column
                        }, m && (w._$.range = [f[f.length - (t || 1)].range[0], f[f.length - 1].range[1]]), r = this.performAction.call(w, h, j, i, this.yy, q[1], e, f), "undefined" != typeof r)
                            return r;
                        t && (d = d.slice(0, - 1 * t * 2), e = e.slice(0, - 1 * t), f = f.slice(0, - 1 * t)), d.push(this.productions_[q[1]][0]), e.push(w.$), f.push(w._$), u = g[d[d.length - 2]][d[d.length - 1]], d.push(u);
                        break;
                    case 3:
                        return !0
                    }
                }
                return !0
            }
        }, c = function() {
            var a = {
                EOF: 1,
                parseError: function(a, b) {
                    if (!this.yy.parser)
                        throw new Error(a);
                    this.yy.parser.parseError(a, b)
                },
                setInput: function(a) {
                    return this._input = a, this._more = this._less = this.done=!1, this.yylineno = this.yyleng = 0, this.yytext = this.matched = this.match = "", this.conditionStack = ["INITIAL"], this.yylloc = {
                        first_line: 1,
                        first_column: 0,
                        last_line: 1,
                        last_column: 0
                    }, this.options.ranges && (this.yylloc.range = [0, 0]), this.offset = 0, this
                },
                input: function() {
                    var a = this._input[0];
                    this.yytext += a, this.yyleng++, this.offset++, this.match += a, this.matched += a;
                    var b = a.match(/(?:\r\n?|\n).*/g);
                    return b ? (this.yylineno++, this.yylloc.last_line++) : this.yylloc.last_column++, this.options.ranges && this.yylloc.range[1]++, this._input = this._input.slice(1), a
                },
                unput: function(a) {
                    var b = a.length, c = a.split(/(?:\r\n?|\n)/g);
                    this._input = a + this._input, this.yytext = this.yytext.substr(0, this.yytext.length - b - 1), this.offset -= b;
                    var d = this.match.split(/(?:\r\n?|\n)/g);
                    this.match = this.match.substr(0, this.match.length - 1), this.matched = this.matched.substr(0, this.matched.length - 1), c.length - 1 && (this.yylineno -= c.length - 1);
                    var e = this.yylloc.range;
                    return this.yylloc = {
                        first_line: this.yylloc.first_line,
                        last_line: this.yylineno + 1,
                        first_column: this.yylloc.first_column,
                        last_column: c ? (c.length === d.length ? this.yylloc.first_column : 0) + d[d.length - c.length].length - c[0].length: this.yylloc.first_column - b
                    }, this.options.ranges && (this.yylloc.range = [e[0], e[0] + this.yyleng - b]), this
                },
                more: function() {
                    return this._more=!0, this
                },
                less: function(a) {
                    this.unput(this.match.slice(a))
                },
                pastInput: function() {
                    var a = this.matched.substr(0, this.matched.length - this.match.length);
                    return (a.length > 20 ? "..." : "") + a.substr( - 20).replace(/\n/g, "")
                },
                upcomingInput: function() {
                    var a = this.match;
                    return a.length < 20 && (a += this._input.substr(0, 20 - a.length)), (a.substr(0, 20) + (a.length > 20 ? "..." : "")).replace(/\n/g, "")
                },
                showPosition: function() {
                    var a = this.pastInput(), b = new Array(a.length + 1).join("-");
                    return a + this.upcomingInput() + "\n" + b + "^"
                },
                next: function() {
                    if (this.done)
                        return this.EOF;
                    this._input || (this.done=!0);
                    var a, b, c, d, e;
                    this._more || (this.yytext = "", this.match = "");
                    for (var f = this._currentRules(), g = 0; g < f.length && (c = this._input.match(this.rules[f[g]]), !c || b&&!(c[0].length > b[0].length) || (b = c, d = g, this.options.flex)); g++);
                    return b ? (e = b[0].match(/(?:\r\n?|\n).*/g), e && (this.yylineno += e.length), this.yylloc = {
                        first_line: this.yylloc.last_line,
                        last_line: this.yylineno + 1,
                        first_column: this.yylloc.last_column,
                        last_column: e ? e[e.length - 1].length - e[e.length - 1].match(/\r?\n?/)[0].length: this.yylloc.last_column + b[0].length
                    }, this.yytext += b[0], this.match += b[0], this.matches = b, this.yyleng = this.yytext.length, this.options.ranges && (this.yylloc.range = [this.offset, this.offset += this.yyleng]), this._more=!1, this._input = this._input.slice(b[0].length), this.matched += b[0], a = this.performAction.call(this, this.yy, this, f[d], this.conditionStack[this.conditionStack.length - 1]), this.done && this._input && (this.done=!1), a ? a : void 0) : "" === this._input ? this.EOF : this.parseError("Lexical error on line " + (this.yylineno + 1) + ". Unrecognized text.\n" + this.showPosition(), {
                        text: "",
                        token: null,
                        line: this.yylineno
                    })
                },
                lex: function() {
                    var a = this.next();
                    return "undefined" != typeof a ? a : this.lex()
                },
                begin: function(a) {
                    this.conditionStack.push(a)
                },
                popState: function() {
                    return this.conditionStack.pop()
                },
                _currentRules: function() {
                    return this.conditions[this.conditionStack[this.conditionStack.length - 1]].rules
                },
                topState: function() {
                    return this.conditionStack[this.conditionStack.length - 2]
                },
                pushState: function(a) {
                    this.begin(a)
                }
            };
            return a.options = {}, a.performAction = function(a, b, c, d) {
                switch (c) {
                case 0:
                    return b.yytext = "\\", 14;
                case 1:
                    if ("\\" !== b.yytext.slice( - 1) && this.begin("mu"), "\\" === b.yytext.slice( - 1) && (b.yytext = b.yytext.substr(0, b.yyleng - 1), this.begin("emu")), b.yytext)
                        return 14;
                    break;
                case 2:
                    return 14;
                case 3:
                    return "\\" !== b.yytext.slice( - 1) && this.popState(), "\\" === b.yytext.slice( - 1) && (b.yytext = b.yytext.substr(0, b.yyleng - 1)), 14;
                case 4:
                    return b.yytext = b.yytext.substr(0, b.yyleng - 4), this.popState(), 15;
                case 5:
                    return 25;
                case 6:
                    return 16;
                case 7:
                    return 20;
                case 8:
                    return 19;
                case 9:
                    return 19;
                case 10:
                    return 23;
                case 11:
                    return 22;
                case 12:
                    this.popState(), this.begin("com");
                    break;
                case 13:
                    return b.yytext = b.yytext.substr(3, b.yyleng - 5), this.popState(), 15;
                case 14:
                    return 22;
                case 15:
                    return 37;
                case 16:
                    return 36;
                case 17:
                    return 36;
                case 18:
                    return 40;
                case 19:
                    break;
                case 20:
                    return this.popState(), 24;
                case 21:
                    return this.popState(), 18;
                case 22:
                    return b.yytext = b.yytext.substr(1, b.yyleng - 2).replace(/\\"/g, '"'), 31;
                case 23:
                    return b.yytext = b.yytext.substr(1, b.yyleng - 2).replace(/\\'/g, "'"), 31;
                case 24:
                    return 38;
                case 25:
                    return 33;
                case 26:
                    return 33;
                case 27:
                    return 32;
                case 28:
                    return 36;
                case 29:
                    return b.yytext = b.yytext.substr(1, b.yyleng - 2), 36;
                case 30:
                    return "INVALID";
                case 31:
                    return 5
                }
            }, a.rules = [/^(?:\\\\(?=(\{\{)))/, /^(?:[^\x00]*?(?=(\{\{)))/, /^(?:[^\x00]+)/, /^(?:[^\x00]{2,}?(?=(\{\{|$)))/, /^(?:[\s\S]*?--\}\})/, /^(?:\{\{>)/, /^(?:\{\{#)/, /^(?:\{\{\/)/, /^(?:\{\{\^)/, /^(?:\{\{\s*else\b)/, /^(?:\{\{\{)/, /^(?:\{\{&)/, /^(?:\{\{!--)/, /^(?:\{\{![\s\S]*?\}\})/, /^(?:\{\{)/, /^(?:=)/, /^(?:\.(?=[}\/ ]))/, /^(?:\.\.)/, /^(?:[\/.])/, /^(?:\s+)/, /^(?:\}\}\})/, /^(?:\}\})/, /^(?:"(\\["]|[^"])*")/, /^(?:'(\\[']|[^'])*')/, /^(?:@)/, /^(?:true(?=[}\s]))/, /^(?:false(?=[}\s]))/, /^(?:-?[0-9]+(?=[}\s]))/, /^(?:[^\s!"#%-,\.\/;->@\[-\^`\{-~]+(?=[=}\s\/.]))/, /^(?:\[[^\]]*\])/, /^(?:.)/, /^(?:$)/], a.conditions = {
                mu: {
                    rules: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                    inclusive: !1
                },
                emu: {
                    rules: [3],
                    inclusive: !1
                },
                com: {
                    rules: [4],
                    inclusive: !1
                },
                INITIAL: {
                    rules: [0, 1, 2, 31],
                    inclusive: !0
                }
            }, a
        }();
        return b.lexer = c, a.prototype = b, b.Parser = a, new a
    }();
    a.Parser = f, a.parse = function(b) {
        return b.constructor === a.AST.ProgramNode ? b : (a.Parser.yy = a.AST, a.Parser.parse(b))
    }, a.AST = {}, a.AST.ProgramNode = function(b, c) {
        this.type = "program", this.statements = b, c && (this.inverse = new a.AST.ProgramNode(c))
    }, a.AST.MustacheNode = function(a, b, c) {
        this.type = "mustache", this.escaped=!c, this.hash = b;
        var d = this.id = a[0], e = this.params = a.slice(1), f = this.eligibleHelper = d.isSimple;
        this.isHelper = f && (e.length || b)
    }, a.AST.PartialNode = function(a, b) {
        this.type = "partial", this.partialName = a, this.context = b
    }, a.AST.BlockNode = function(b, c, d, e) {
        var f = function(b, c) {
            if (b.original !== c.original)
                throw new a.Exception(b.original + " doesn't match " + c.original)
        };
        f(b.id, e), this.type = "block", this.mustache = b, this.program = c, this.inverse = d, this.inverse&&!this.program && (this.isInverse=!0)
    }, a.AST.ContentNode = function(a) {
        this.type = "content", this.string = a
    }, a.AST.HashNode = function(a) {
        this.type = "hash", this.pairs = a
    }, a.AST.IdNode = function(b) {
        this.type = "ID";
        for (var c = "", d = [], e = 0, f = 0, g = b.length; g > f; f++) {
            var h = b[f].part;
            if (c += (b[f].separator || "") + h, ".." === h || "." === h || "this" === h) {
                if (d.length > 0)
                    throw new a.Exception("Invalid path: " + c);
                ".." === h ? e++ : this.isScoped=!0
            } else 
                d.push(h)
        }
        this.original = c, this.parts = d, this.string = d.join("."), this.depth = e, this.isSimple = 1 === b.length&&!this.isScoped && 0 === e, this.stringModeValue = this.string
    }, a.AST.PartialNameNode = function(a) {
        this.type = "PARTIAL_NAME", this.name = a.original
    }, a.AST.DataNode = function(a) {
        this.type = "DATA", this.id = a
    }, a.AST.StringNode = function(a) {
        this.type = "STRING", this.original = this.string = this.stringModeValue = a
    }, a.AST.IntegerNode = function(a) {
        this.type = "INTEGER", this.original = this.integer = a, this.stringModeValue = Number(a)
    }, a.AST.BooleanNode = function(a) {
        this.type = "BOOLEAN", this.bool = a, this.stringModeValue = "true" === a
    }, a.AST.CommentNode = function(a) {
        this.type = "comment", this.comment = a
    };
    var g = ["description", "fileName", "lineNumber", "message", "name", "number", "stack"];
    a.Exception = function() {
        for (var a = Error.prototype.constructor.apply(this, arguments), b = 0; b < g.length; b++)
            this[g[b]] = a[g[b]]
    }, a.Exception.prototype = new Error, a.SafeString = function(a) {
        this.string = a
    }, a.SafeString.prototype.toString = function() {
        return this.string.toString()
    };
    var h = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
        "`": "&#x60;"
    }, i = /[&<>"'`]/g, j = /[&<>"'`]/, k = function(a) {
        return h[a] || "&amp;"
    };
    a.Utils = {
        extend: function(a, b) {
            for (var c in b)
                b.hasOwnProperty(c) && (a[c] = b[c])
        },
        escapeExpression: function(b) {
            return b instanceof a.SafeString ? b.toString() : null == b || b===!1 ? "" : (b = b.toString(), j.test(b) ? b.replace(i, k) : b)
        },
        isEmpty: function(a) {
            return a || 0 === a ? "[object Array]" === c.call(a) && 0 === a.length?!0 : !1 : !0
        }
    };
    var l = a.Compiler = function() {}, m = a.JavaScriptCompiler = function() {};
    l.prototype = {
        compiler: l,
        disassemble: function() {
            for (var a, b, c, d = this.opcodes, e = [], f = 0, g = d.length; g > f; f++)
                if (a = d[f], "DECLARE" === a.opcode)
                    e.push("DECLARE " + a.name + "=" + a.value);
                else {
                    b = [];
                    for (var h = 0; h < a.args.length; h++)
                        c = a.args[h], "string" == typeof c && (c = '"' + c.replace("\n", "\\n") + '"'), b.push(c);
                        e.push(a.opcode + " " + b.join(" "))
                }
            return e.join("\n")
        },
        equals: function(a) {
            var b = this.opcodes.length;
            if (a.opcodes.length !== b)
                return !1;
            for (var c = 0; b > c; c++) {
                var d = this.opcodes[c], e = a.opcodes[c];
                if (d.opcode !== e.opcode || d.args.length !== e.args.length)
                    return !1;
                for (var f = 0; f < d.args.length; f++)
                    if (d.args[f] !== e.args[f])
                        return !1
            }
            if (b = this.children.length, a.children.length !== b)
                return !1;
            for (c = 0; b > c; c++)
                if (!this.children[c].equals(a.children[c]))
                    return !1;
            return !0
        },
        guid: 0,
        compile: function(a, b) {
            this.children = [], this.depths = {
                list: []
            }, this.options = b;
            var c = this.options.knownHelpers;
            if (this.options.knownHelpers = {
                helperMissing: !0,
                blockHelperMissing: !0,
                each: !0,
                "if": !0,
                unless: !0,
                "with": !0,
                log: !0
            }, c)
                for (var d in c)
                    this.options.knownHelpers[d] = c[d];
            return this.program(a)
        },
        accept: function(a) {
            return this[a.type](a)
        },
        program: function(a) {
            var b, c = a.statements;
            this.opcodes = [];
            for (var d = 0, e = c.length; e > d; d++)
                b = c[d], this[b.type](b);
            return this.isSimple = 1 === e, this.depths.list = this.depths.list.sort(function(a, b) {
                return a - b
            }), this
        },
        compileProgram: function(a) {
            var b, c = (new this.compiler).compile(a, this.options), d = this.guid++;
            this.usePartial = this.usePartial || c.usePartial, this.children[d] = c;
            for (var e = 0, f = c.depths.list.length; f > e; e++)
                b = c.depths.list[e], 2 > b || this.addDepth(b - 1);
            return d
        },
        block: function(a) {
            var b = a.mustache, c = a.program, d = a.inverse;
            c && (c = this.compileProgram(c)), d && (d = this.compileProgram(d));
            var e = this.classifyMustache(b);
            "helper" === e ? this.helperMustache(b, c, d) : "simple" === e ? (this.simpleMustache(b), this.opcode("pushProgram", c), this.opcode("pushProgram", d), this.opcode("emptyHash"), this.opcode("blockValue")) : (this.ambiguousMustache(b, c, d), this.opcode("pushProgram", c), this.opcode("pushProgram", d), this.opcode("emptyHash"), this.opcode("ambiguousBlockValue")), this.opcode("append")
        },
        hash: function(a) {
            var b, c, d = a.pairs;
            this.opcode("pushHash");
            for (var e = 0, f = d.length; f > e; e++)
                b = d[e], c = b[1], this.options.stringParams ? (c.depth && this.addDepth(c.depth), this.opcode("getContext", c.depth || 0), this.opcode("pushStringParam", c.stringModeValue, c.type)) : this.accept(c), this.opcode("assignToHash", b[0]);
            this.opcode("popHash")
        },
        partial: function(a) {
            var b = a.partialName;
            this.usePartial=!0, a.context ? this.ID(a.context) : this.opcode("push", "depth0"), this.opcode("invokePartial", b.name), this.opcode("append")
        },
        content: function(a) {
            this.opcode("appendContent", a.string)
        },
        mustache: function(a) {
            var b = this.options, c = this.classifyMustache(a);
            "simple" === c ? this.simpleMustache(a) : "helper" === c ? this.helperMustache(a) : this.ambiguousMustache(a), this.opcode(a.escaped&&!b.noEscape ? "appendEscaped" : "append")
        },
        ambiguousMustache: function(a, b, c) {
            var d = a.id, e = d.parts[0], f = null != b || null != c;
            this.opcode("getContext", d.depth), this.opcode("pushProgram", b), this.opcode("pushProgram", c), this.opcode("invokeAmbiguous", e, f)
        },
        simpleMustache: function(a) {
            var b = a.id;
            "DATA" === b.type ? this.DATA(b) : b.parts.length ? this.ID(b) : (this.addDepth(b.depth), this.opcode("getContext", b.depth), this.opcode("pushContext")), this.opcode("resolvePossibleLambda")
        },
        helperMustache: function(a, b, c) {
            var d = this.setupFullMustacheParams(a, b, c), e = a.id.parts[0];
            if (this.options.knownHelpers[e])
                this.opcode("invokeKnownHelper", d.length, e);
            else {
                if (this.options.knownHelpersOnly)
                    throw new Error("You specified knownHelpersOnly, but used the unknown helper " + e);
                this.opcode("invokeHelper", d.length, e)
            }
        },
        ID: function(a) {
            this.addDepth(a.depth), this.opcode("getContext", a.depth);
            var b = a.parts[0];
            b ? this.opcode("lookupOnContext", a.parts[0]) : this.opcode("pushContext");
            for (var c = 1, d = a.parts.length; d > c; c++)
                this.opcode("lookup", a.parts[c])
        },
        DATA: function(b) {
            if (this.options.data=!0, b.id.isScoped || b.id.depth)
                throw new a.Exception("Scoped data references are not supported: " + b.original);
            this.opcode("lookupData");
            for (var c = b.id.parts, d = 0, e = c.length; e > d; d++)
                this.opcode("lookup", c[d])
        },
        STRING: function(a) {
            this.opcode("pushString", a.string)
        },
        INTEGER: function(a) {
            this.opcode("pushLiteral", a.integer)
        },
        BOOLEAN: function(a) {
            this.opcode("pushLiteral", a.bool)
        },
        comment: function() {},
        opcode: function(a) {
            this.opcodes.push({
                opcode: a,
                args: [].slice.call(arguments, 1)
            })
        },
        declare: function(a, b) {
            this.opcodes.push({
                opcode: "DECLARE",
                name: a,
                value: b
            })
        },
        addDepth: function(a) {
            if (isNaN(a))
                throw new Error("EWOT");
            0 !== a && (this.depths[a] || (this.depths[a]=!0, this.depths.list.push(a)))
        },
        classifyMustache: function(a) {
            var b = a.isHelper, c = a.eligibleHelper, d = this.options;
            if (c&&!b) {
                var e = a.id.parts[0];
                d.knownHelpers[e] ? b=!0 : d.knownHelpersOnly && (c=!1)
            }
            return b ? "helper" : c ? "ambiguous" : "simple"
        },
        pushParams: function(a) {
            for (var b, c = a.length; c--;)
                b = a[c], this.options.stringParams ? (b.depth && this.addDepth(b.depth), this.opcode("getContext", b.depth || 0), this.opcode("pushStringParam", b.stringModeValue, b.type)) : this[b.type](b)
        },
        setupMustacheParams: function(a) {
            var b = a.params;
            return this.pushParams(b), a.hash ? this.hash(a.hash) : this.opcode("emptyHash"), b
        },
        setupFullMustacheParams: function(a, b, c) {
            var d = a.params;
            return this.pushParams(d), this.opcode("pushProgram", b), this.opcode("pushProgram", c), a.hash ? this.hash(a.hash) : this.opcode("emptyHash"), d
        }
    };
    var n = function(a) {
        this.value = a
    };
    m.prototype = {
        nameLookup: function(a, b) {
            return /^[0-9]+$/.test(b) ? a + "[" + b + "]" : m.isValidJavaScriptVariableName(b) ? a + "." + b : a + "['" + b + "']"
        },
        appendToBuffer: function(a) {
            return this.environment.isSimple ? "return " + a + ";" : {
                appendToBuffer: !0,
                content: a,
                toString: function() {
                    return "buffer += " + a + ";"
                }
            }
        },
        initializeBuffer: function() {
            return this.quotedString("")
        },
        namespace: "Handlebars",
        compile: function(b, c, d, e) {
            this.environment = b, this.options = c || {}, a.log(a.logger.DEBUG, this.environment.disassemble() + "\n\n"), this.name = this.environment.name, this.isChild=!!d, this.context = d || {
                programs: [],
                environments: [],
                aliases: {}
            }, this.preamble(), this.stackSlot = 0, this.stackVars = [], this.registers = {
                list: []
            }, this.compileStack = [], this.inlineStack = [], this.compileChildren(b, c);
            var f, g = b.opcodes;
            for (this.i = 0, r = g.length; this.i < r; this.i++)
                f = g[this.i], "DECLARE" === f.opcode ? this[f.name] = f.value : this[f.opcode].apply(this, f.args);
            return this.createFunctionContext(e)
        },
        nextOpcode: function() {
            var a = this.environment.opcodes;
            return a[this.i + 1]
        },
        eat: function() {
            this.i = this.i + 1
        },
        preamble: function() {
            var a = [];
            if (this.isChild)
                a.push("");
            else {
                var b = this.namespace, c = "helpers = this.merge(helpers, " + b + ".helpers);";
                this.environment.usePartial && (c = c + " partials = this.merge(partials, " + b + ".partials);"), this.options.data && (c += " data = data || {};"), a.push(c)
            }
            a.push(this.environment.isSimple ? "" : ", buffer = " + this.initializeBuffer()), this.lastContext = 0, this.source = a
        },
        createFunctionContext: function(b) {
            var c = this.stackVars.concat(this.registers.list);
            if (c.length > 0 && (this.source[1] = this.source[1] + ", " + c.join(", ")), !this.isChild)
                for (var d in this.context.aliases)
                    this.context.aliases.hasOwnProperty(d) && (this.source[1] = this.source[1] + ", " + d + "=" + this.context.aliases[d]);
            this.source[1] && (this.source[1] = "var " + this.source[1].substring(2) + ";"), this.isChild || (this.source[1] += "\n" + this.context.programs.join("\n") + "\n"), this.environment.isSimple || this.source.push("return buffer;");
            for (var e = this.isChild ? ["depth0", "data"] : ["Handlebars", "depth0", "helpers", "partials", "data"], f = 0, g = this.environment.depths.list.length; g > f; f++)
                e.push("depth" + this.environment.depths.list[f]);
            var h = this.mergeSource();
            if (!this.isChild) {
                var i = a.COMPILER_REVISION, j = a.REVISION_CHANGES[i];
                h = "this.compilerInfo = [" + i + ",'" + j + "'];\n" + h
            }
            if (b)
                return e.push(h), Function.apply(this, e);
            var k = "function " + (this.name || "") + "(" + e.join(",") + ") {\n  " + h + "}";
            return a.log(a.logger.DEBUG, k + "\n\n"), k
        },
        mergeSource: function() {
            for (var a, c = "", d = 0, e = this.source.length; e > d; d++) {
                var f = this.source[d];
                f.appendToBuffer ? a = a ? a + "\n    + " + f.content : f.content : (a && (c += "buffer += " + a + ";\n  ", a = b), c += f + "\n  ")
            }
            return c
        },
        blockValue: function() {
            this.context.aliases.blockHelperMissing = "helpers.blockHelperMissing";
            var a = ["depth0"];
            this.setupParams(0, a), this.replaceStack(function(b) {
                return a.splice(1, 0, b), "blockHelperMissing.call(" + a.join(", ") + ")"
            })
        },
        ambiguousBlockValue: function() {
            this.context.aliases.blockHelperMissing = "helpers.blockHelperMissing";
            var a = ["depth0"];
            this.setupParams(0, a);
            var b = this.topStack();
            a.splice(1, 0, b), a[a.length - 1] = "options", this.source.push("if (!" + this.lastHelper + ") { " + b + " = blockHelperMissing.call(" + a.join(", ") + "); }")
        },
        appendContent: function(a) {
            this.source.push(this.appendToBuffer(this.quotedString(a)))
        },
        append: function() {
            this.flushInline();
            var a = this.popStack();
            this.source.push("if(" + a + " || " + a + " === 0) { " + this.appendToBuffer(a) + " }"), this.environment.isSimple && this.source.push("else { " + this.appendToBuffer("''") + " }")
        },
        appendEscaped: function() {
            this.context.aliases.escapeExpression = "this.escapeExpression", this.source.push(this.appendToBuffer("escapeExpression(" + this.popStack() + ")"))
        },
        getContext: function(a) {
            this.lastContext !== a && (this.lastContext = a)
        },
        lookupOnContext: function(a) {
            this.push(this.nameLookup("depth" + this.lastContext, a, "context"))
        },
        pushContext: function() {
            this.pushStackLiteral("depth" + this.lastContext)
        },
        resolvePossibleLambda: function() {
            this.context.aliases.functionType = '"function"', this.replaceStack(function(a) {
                return "typeof " + a + " === functionType ? " + a + ".apply(depth0) : " + a
            })
        },
        lookup: function(a) {
            this.replaceStack(function(b) {
                return b + " == null || " + b + " === false ? " + b + " : " + this.nameLookup(b, a, "context")
            })
        },
        lookupData: function() {
            this.push("data")
        },
        pushStringParam: function(a, b) {
            this.pushStackLiteral("depth" + this.lastContext), this.pushString(b), "string" == typeof a ? this.pushString(a) : this.pushStackLiteral(a)
        },
        emptyHash: function() {
            this.pushStackLiteral("{}"), this.options.stringParams && (this.register("hashTypes", "{}"), this.register("hashContexts", "{}"))
        },
        pushHash: function() {
            this.hash = {
                values: [],
                types: [],
                contexts: []
            }
        },
        popHash: function() {
            var a = this.hash;
            this.hash = b, this.options.stringParams && (this.register("hashContexts", "{" + a.contexts.join(",") + "}"), this.register("hashTypes", "{" + a.types.join(",") + "}")), this.push("{\n    " + a.values.join(",\n    ") + "\n  }")
        },
        pushString: function(a) {
            this.pushStackLiteral(this.quotedString(a))
        },
        push: function(a) {
            return this.inlineStack.push(a), a
        },
        pushLiteral: function(a) {
            this.pushStackLiteral(a)
        },
        pushProgram: function(a) {
            this.pushStackLiteral(null != a ? this.programExpression(a) : null)
        },
        invokeHelper: function(a, b) {
            this.context.aliases.helperMissing = "helpers.helperMissing";
            var c = this.lastHelper = this.setupHelper(a, b, !0), d = this.nameLookup("depth" + this.lastContext, b, "context");
            this.push(c.name + " || " + d), this.replaceStack(function(a) {
                return a + " ? " + a + ".call(" + c.callParams + ") : helperMissing.call(" + c.helperMissingParams + ")"
            })
        },
        invokeKnownHelper: function(a, b) {
            var c = this.setupHelper(a, b);
            this.push(c.name + ".call(" + c.callParams + ")")
        },
        invokeAmbiguous: function(a, b) {
            this.context.aliases.functionType = '"function"', this.pushStackLiteral("{}");
            var c = this.setupHelper(0, a, b), d = this.lastHelper = this.nameLookup("helpers", a, "helper"), e = this.nameLookup("depth" + this.lastContext, a, "context"), f = this.nextStack();
            this.source.push("if (" + f + " = " + d + ") { " + f + " = " + f + ".call(" + c.callParams + "); }"), this.source.push("else { " + f + " = " + e + "; " + f + " = typeof " + f + " === functionType ? " + f + ".apply(depth0) : " + f + "; }")
        },
        invokePartial: function(a) {
            var b = [this.nameLookup("partials", a, "partial"), "'" + a + "'", this.popStack(), "helpers", "partials"];
            this.options.data && b.push("data"), this.context.aliases.self = "this", this.push("self.invokePartial(" + b.join(", ") + ")")
        },
        assignToHash: function(a) {
            var b, c, d = this.popStack();
            this.options.stringParams && (c = this.popStack(), b = this.popStack());
            var e = this.hash;
            b && e.contexts.push("'" + a + "': " + b), c && e.types.push("'" + a + "': " + c), e.values.push("'" + a + "': (" + d + ")")
        },
        compiler: m,
        compileChildren: function(a, b) {
            for (var c, d, e = a.children, f = 0, g = e.length; g > f; f++) {
                c = e[f], d = new this.compiler;
                var h = this.matchExistingProgram(c);
                null == h ? (this.context.programs.push(""), h = this.context.programs.length, c.index = h, c.name = "program" + h, this.context.programs[h] = d.compile(c, b, this.context), this.context.environments[h] = c) : (c.index = h, c.name = "program" + h)
            }
        },
        matchExistingProgram: function(a) {
            for (var b = 0, c = this.context.environments.length; c > b; b++) {
                var d = this.context.environments[b];
                if (d && d.equals(a))
                    return b
            }
        },
        programExpression: function(a) {
            if (this.context.aliases.self = "this", null == a)
                return "self.noop";
            for (var b, c = this.environment.children[a], d = c.depths.list, e = [c.index, c.name, "data"], f = 0, g = d.length; g > f; f++)
                b = d[f], e.push(1 === b ? "depth0" : "depth" + (b - 1));
            return (0 === d.length ? "self.program(" : "self.programWithDepth(") + e.join(", ") + ")"
        },
        register: function(a, b) {
            this.useRegister(a), this.source.push(a + " = " + b + ";")
        },
        useRegister: function(a) {
            this.registers[a] || (this.registers[a]=!0, this.registers.list.push(a))
        },
        pushStackLiteral: function(a) {
            return this.push(new n(a))
        },
        pushStack: function(a) {
            this.flushInline();
            var b = this.incrStack();
            return a && this.source.push(b + " = " + a + ";"), this.compileStack.push(b), b
        },
        replaceStack: function(a) {
            var b, c = "", d = this.isInline();
            if (d) {
                var e = this.popStack(!0);
                if (e instanceof n)
                    b = e.value;
                else {
                    var f = this.stackSlot ? this.topStackName(): this.incrStack();
                    c = "(" + this.push(f) + " = " + e + "),", b = this.topStack()
                }
            } else 
                b = this.topStack();
            var g = a.call(this, b);
            return d ? ((this.inlineStack.length || this.compileStack.length) && this.popStack(), this.push("(" + c + g + ")")) : (/^stack/.test(b) || (b = this.nextStack()), this.source.push(b + " = (" + c + g + ");")), b
        },
        nextStack: function() {
            return this.pushStack()
        },
        incrStack: function() {
            return this.stackSlot++, this.stackSlot > this.stackVars.length && this.stackVars.push("stack" + this.stackSlot), this.topStackName()
        },
        topStackName: function() {
            return "stack" + this.stackSlot
        },
        flushInline: function() {
            var a = this.inlineStack;
            if (a.length) {
                this.inlineStack = [];
                for (var b = 0, c = a.length; c > b; b++) {
                    var d = a[b];
                    d instanceof n ? this.compileStack.push(d) : this.pushStack(d)
                }
            }
        },
        isInline: function() {
            return this.inlineStack.length
        },
        popStack: function(a) {
            var b = this.isInline(), c = (b ? this.inlineStack : this.compileStack).pop();
            return !a && c instanceof n ? c.value : (b || this.stackSlot--, c)
        },
        topStack: function(a) {
            var b = this.isInline() ? this.inlineStack: this.compileStack, c = b[b.length - 1];
            return !a && c instanceof n ? c.value : c
        },
        quotedString: function(a) {
            return '"' + a.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\u2028/g, "\\u2028").replace(/\u2029/g, "\\u2029") + '"'
        },
        setupHelper: function(a, b, c) {
            var d = [];
            this.setupParams(a, d, c);
            var e = this.nameLookup("helpers", b, "helper");
            return {
                params: d,
                name: e,
                callParams: ["depth0"].concat(d).join(", "),
                helperMissingParams: c && ["depth0", this.quotedString(b)].concat(d).join(", ")
            }
        },
        setupParams: function(a, b, c) {
            var d, e, f, g = [], h = [], i = [];
            g.push("hash:" + this.popStack()), e = this.popStack(), f = this.popStack(), (f || e) && (f || (this.context.aliases.self = "this", f = "self.noop"), e || (this.context.aliases.self = "this", e = "self.noop"), g.push("inverse:" + e), g.push("fn:" + f));
            for (var j = 0; a > j; j++)
                d = this.popStack(), b.push(d), this.options.stringParams && (i.push(this.popStack()), h.push(this.popStack()));
            return this.options.stringParams && (g.push("contexts:[" + h.join(",") + "]"), g.push("types:[" + i.join(",") + "]"), g.push("hashContexts:hashContexts"), g.push("hashTypes:hashTypes")), this.options.data && g.push("data:data"), g = "{" + g.join(",") + "}", c ? (this.register("options", g), b.push("options")) : b.push(g), b.join(", ")
        }
    };
    for (var o = "break else new var case finally return void catch for switch while continue function this with default if throw delete in try do instanceof typeof abstract enum int short boolean export interface static byte extends long super char final native synchronized class float package throws const goto private transient debugger implements protected volatile double import public let yield".split(" "), p = m.RESERVED_WORDS = {}, q = 0, r = o.length; r > q; q++)
        p[o[q]]=!0;
    m.isValidJavaScriptVariableName = function(a) {
        return !m.RESERVED_WORDS[a] && /^[a-zA-Z_$][0-9a-zA-Z_$]+$/.test(a)?!0 : !1
    }, a.precompile = function(b, c) {
        if (null == b || "string" != typeof b && b.constructor !== a.AST.ProgramNode)
            throw new a.Exception("You must pass a string or Handlebars AST to Handlebars.precompile. You passed " + b);
        c = c || {}, "data"in c || (c.data=!0);
        var d = a.parse(b), e = (new l).compile(d, c);
        return (new m).compile(e, c)
    }, a.compile = function(c, d) {
        function e() {
            var e = a.parse(c), f = (new l).compile(e, d), g = (new m).compile(f, d, b, !0);
            return a.template(g)
        }
        if (null == c || "string" != typeof c && c.constructor !== a.AST.ProgramNode)
            throw new a.Exception("You must pass a string or Handlebars AST to Handlebars.compile. You passed " + c);
        d = d || {}, "data"in d || (d.data=!0);
        var f;
        return function(a, b) {
            return f || (f = e()), f.call(this, a, b)
        }
    }, a.VM = {
        template: function(b) {
            var c = {
                escapeExpression: a.Utils.escapeExpression,
                invokePartial: a.VM.invokePartial,
                programs: [],
                program: function(b, c, d) {
                    var e = this.programs[b];
                    return d ? e = a.VM.program(b, c, d) : e || (e = this.programs[b] = a.VM.program(b, c)), e
                },
                merge: function(b, c) {
                    var d = b || c;
                    return b && c && (d = {}, a.Utils.extend(d, c), a.Utils.extend(d, b)), d
                },
                programWithDepth: a.VM.programWithDepth,
                noop: a.VM.noop,
                compilerInfo: null
            };
            return function(d, e) {
                e = e || {};
                var f = b.call(c, a, d, e.helpers, e.partials, e.data), g = c.compilerInfo || [], h = g[0] || 1, i = a.COMPILER_REVISION;
                if (h !== i) {
                    if (i > h) {
                        var j = a.REVISION_CHANGES[i], k = a.REVISION_CHANGES[h];
                        throw "Template was precompiled with an older version of Handlebars than the current runtime. Please update your precompiler to a newer version (" + j + ") or downgrade your runtime to an older version (" + k + ")."
                    }
                    throw "Template was precompiled with a newer version of Handlebars than the current runtime. Please update your runtime to a newer version (" + g[1] + ")."
                }
                return f
            }
        },
        programWithDepth: function(a, b, c) {
            var d = Array.prototype.slice.call(arguments, 3), e = function(a, e) {
                return e = e || {}, b.apply(this, [a, e.data || c].concat(d))
            };
            return e.program = a, e.depth = d.length, e
        },
        program: function(a, b, c) {
            var d = function(a, d) {
                return d = d || {}, b(a, d.data || c)
            };
            return d.program = a, d.depth = 0, d
        },
        noop: function() {
            return ""
        },
        invokePartial: function(c, d, e, f, g, h) {
            var i = {
                helpers: f,
                partials: g,
                data: h
            };
            if (c === b)
                throw new a.Exception("The partial " + d + " could not be found");
            if (c instanceof Function)
                return c(e, i);
            if (!a.compile)
                throw new a.Exception("The partial " + d + " could not be compiled when running in runtime-only mode");
            return g[d] = a.compile(c, {
                data: h !== b
            }), g[d](e, i)
        }
    }, a.template = a.VM.template
}(Handlebars), define("handlebars", function(a) {
    return function() {
        var b;
        return b || a.Handlebars
    }
}(this)), define("templates", ["handlebars"], function(a) {
    return this.JST = this.JST || {}, this.JST.auth = a.template(function(a, b, c, d, e) {
        function f(a) {
            var b, c = "";
            return c += "\n\n      <p><b>" + k((b = a.details, b = null == b || b===!1 ? b : b.app_name, typeof b === j ? b.apply(a) : b)) + '</b> would like to access your data on Delicious.</p>\n\n      <div class="app-info">\n        <h4>About <b>' + k((b = a.details, b = null == b || b===!1 ? b : b.app_name, typeof b === j ? b.apply(a) : b)) + "</b>:</h4>\n        <p>" + k((b = a.details, b = null == b || b===!1 ? b : b.description, typeof b === j ? b.apply(a) : b)) + "</p>\n      </div>\n\n    "
        }
        function g() {
            return "\n\n      <p>Sorry, this app has been deleted or disabled by its author.</p>\n\n    "
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var h, i = "", j = "function", k = this.escapeExpression, l = this;
        return i += '<div class="auth-container">\n  <header class="landing-nav item clearfix">\n    <h1 class="pull-left">\n      <span>\n        <div class="logo">\n          <i class="s1"></i>\n          <i class="s2"></i>\n          <i class="s3"></i>\n          <i class="s4"></i>\n        </div>\n        Delicious\n      </span>\n    </h1>\n  </header>\n\n  <div class="auth-dialog item">\n\n    ', h = c["if"].call(b, b.status, {
            hash: {},
            inverse: l.program(3, g, e),
            fn: l.program(1, f, e),
            data: e
        }), (h || 0 === h) && (i += h), i += '\n\n  </div>\n\n  <div class="auth-footer item">\n    <button class="text-btn text-btn-primary auth-confirm">Authorize</button>\n    <button class="text-btn auth-cancel">Refuse</button>\n  </div>\n\n</div>\n'
    }), this.JST["discover/link"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e = "";
            return e += '\n          <a href="/link/', (d = c.md5) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.md5, d = typeof d === p ? d.apply(a) : d), e += q(d) + '"><span><b>', (d = c.retweets) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.retweets, d = typeof d === p ? d.apply(a) : d), e += q(d) + "</b></span></a>\n        "
        }
        function g(a, b) {
            var d, e = "";
            return e += '\n          <a href="/', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === p ? d.apply(a) : d), e += q(d) + '" class="early-bird" title="', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === p ? d.apply(a) : d), e += q(d) + ' was the first user to save this link."><img src="', (d = c.avatar_url) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.avatar_url, d = typeof d === p ? d.apply(a) : d), e += q(d) + '" class="avatar-img"></a>\n        '
        }
        function h(a, b) {
            var d, e = "";
            return e += '\n        <ul class="tags"> ', d = c.each.call(a, a.tags, {
                hash: {},
                inverse: r.noop,
                fn: r.program(6, i, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "</ul>\n      "
        }
        function i(a) {
            var b = "";
            return b += '<li><a href="/tag/' + q(typeof a === p ? a.apply(a) : a) + '">' + q(typeof a === p ? a.apply(a) : a) + "</a></li> "
        }
        function j(a, b) {
            var d, e, f = "";
            return f += '\n        <time datetime="', e = {
                hash: {},
                data: b
            }, f += q((d = c.formatMachineDate0 || a.formatMachineDate0, d ? d.call(a, a.save_at, e) : s.call(a, "formatMachineDate0", a.save_at, e))) + '" class="date" title="', e = {
                hash: {},
                data: b
            }, f += q((d = c.formatDate0Full || a.formatDate0Full, d ? d.call(a, a.save_at, e) : s.call(a, "formatDate0Full", a.save_at, e))) + '">', e = {
                hash: {},
                data: b
            }, f += q((d = c.formatDate0Relative || a.formatDate0Relative, d ? d.call(a, a.save_at, e) : s.call(a, "formatDate0Relative", a.save_at, e))) + "</time>\n        "
        }
        function k(a, b) {
            var d, e, f = "";
            return f += '\n        <time datetime="', e = {
                hash: {},
                data: b
            }, f += q((d = c.formatMachineDate || a.formatMachineDate, d ? d.call(a, a.time_created, e) : s.call(a, "formatMachineDate", a.time_created, e))) + '" class="date" title="', e = {
                hash: {},
                data: b
            }, f += q((d = c.formatDate0Full || a.formatDate0Full, d ? d.call(a, a.save_at, e) : s.call(a, "formatDate0Full", a.save_at, e))) + '">', e = {
                hash: {},
                data: b
            }, f += q((d = c.formatDateRelative || a.formatDateRelative, d ? d.call(a, a.time_created, e) : s.call(a, "formatDateRelative", a.time_created, e))) + "</time>\n      "
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var l, m, n, o = "", p = "function", q = this.escapeExpression, r = this, s = c.helperMissing;
        return o += '<div class="item">\n    <div class="title-wrapper">\n      <a href="', (l = c.url) ? l = l.call(b, {
            hash: {},
            data: e
        }) : (l = b.url, l = typeof l === p ? l.apply(b) : l), o += q(l) + '" class="title">', n = {
            hash: {},
            data: e
        }, o += q((l = c.or || b.or, l ? l.call(b, b.title, b.url, n) : s.call(b, "or", b.title, b.url, n))) + '</a>\n    </div>\n\n    <div class="meta">\n\n      <span class="link-rank">\n        ', m = c["if"].call(b, b.retweets, {
            hash: {},
            inverse: r.noop,
            fn: r.program(1, f, e),
            data: e
        }), (m || 0 === m) && (o += m), o += '\n      </span>\n\n      <div class="link-rank-early-bird">\n        ', m = c["with"].call(b, b.tweet, {
            hash: {},
            inverse: r.noop,
            fn: r.program(3, g, e),
            data: e
        }), (m || 0 === m) && (o += m), o += "\n      </div>\n\n\n      ", m = c["if"].call(b, b.tags, {
            hash: {},
            inverse: r.noop,
            fn: r.program(5, h, e),
            data: e
        }), (m || 0 === m) && (o += m), o += '\n    </div>\n\n    <div class="meta meta-secondary">\n      <div class="actions">\n        <a href="#" class="action-link bookmark-btn">Add Link</a>\n        <div class="share-btn">\n          <div class="dropdown">\n            <a href="#" data-toggle="dropdown" class="action-link dropdown-toggle">Share</a>\n            <ul class="dropdown-menu">\n              <li><a href="https://twitter.com/intent/tweet?text=', (m = c.title) ? m = m.call(b, {
            hash: {},
            data: e
        }) : (m = b.title, m = typeof m === p ? m.apply(b) : m), o += q(m) + "&amp;url=", n = {
            hash: {},
            data: e
        }, o += q((l = c.URLEncode || b.URLEncode, l ? l.call(b, b.url, n) : s.call(b, "URLEncode", b.url, n))) + '&amp;via=delicious" class="share-social noscript">Twitter</a></li>\n              <li><a href="http://www.facebook.com/share.php?u=', n = {
            hash: {},
            data: e
        }, o += q((l = c.URLEncode || b.URLEncode, l ? l.call(b, b.url, n) : s.call(b, "URLEncode", b.url, n))) + '" class="share-social noscript">Facebook</a></li>\n              <li><a href="https://plus.google.com/share?url=', n = {
            hash: {},
            data: e
        }, o += q((l = c.URLEncode || b.URLEncode, l ? l.call(b, b.url, n) : s.call(b, "URLEncode", b.url, n))) + '" class="share-social noscript">Google+</a></li>\n              <li><a href="http://getpocket.com/edit.php?url=', n = {
            hash: {},
            data: e
        }, o += q((l = c.URLEncode || b.URLEncode, l ? l.call(b, b.url, n) : s.call(b, "URLEncode", b.url, n))) + "&amp;title=", n = {
            hash: {},
            data: e
        }, o += q((l = c.URLEncode || b.URLEncode, l ? l.call(b, b.title, n) : s.call(b, "URLEncode", b.title, n))) + '" class="share-social noscript">Pocket</a></li>\n              <li><a href="http://www.instapaper.com/edit?url=', n = {
            hash: {},
            data: e
        }, o += q((l = c.URLEncode || b.URLEncode, l ? l.call(b, b.url, n) : s.call(b, "URLEncode", b.url, n))) + "&amp;title=", n = {
            hash: {},
            data: e
        }, o += q((l = c.URLEncode || b.URLEncode, l ? l.call(b, b.title, n) : s.call(b, "URLEncode", b.title, n))) + '" class="share-social noscript">Instapaper</a></li>\n              <li><a href="mailto:?subject=', n = {
            hash: {},
            data: e
        }, o += q((l = c.URLEncode || b.URLEncode, l ? l.call(b, b.title, n) : s.call(b, "URLEncode", b.title, n))) + "&amp;body=", n = {
            hash: {},
            data: e
        }, o += q((l = c.URLEncode || b.URLEncode, l ? l.call(b, b.url, n) : s.call(b, "URLEncode", b.url, n))) + '" class="share-social noscript">Email</a></li>\n            </ul>\n          </div>\n        </div>\n      </div>\n\n      ', m = c["if"].call(b, b.save_at, {
            hash: {},
            inverse: r.program(10, k, e),
            fn: r.program(8, j, e),
            data: e
        }), (m || 0 === m) && (o += m), o += "\n    </div>\n\n</div>\n<!-- .item -->\n"
    }), this.JST["discover/list"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, "<!-- Delicious_728x90_ATF -->\n<div id='div-gpt-ad-1416871411588-0' style='width:728px; height:90px;'>\n<script type='text/javascript'>\ngoogletag.cmd.push(function() { googletag.display('div-gpt-ad-1416871411588-0'); });\n</script>\n</div>\n<div class=\"link-list\"></div>\n"
    }), this.JST["discover/page"] = a.template(function(a, b, c, d, e) {
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var f, g = "", h = "function", i = this.escapeExpression;
        return g += '<div class="hero-unit">\n  <h2>Discover</h2>\n  Community picked links based on your interests.\n  <a href="/', (f = c.username) ? f = f.call(b, {
            hash: {},
            data: e
        }) : (f = b.username, f = typeof f === h ? f.apply(b) : f), g += i(f) + '/subscriptions">Manage your subscriptions</a> to tweak this list.\n</div>\n<div class="wizard"></div>\n<div class="links"></div>\n', (f = c.loadMore) ? f = f.call(b, {
            hash: {},
            data: e
        }) : (f = b.loadMore, f = typeof f === h ? f.apply(b) : f), g += i(f) + '\n\n<script>\n\njQuery.fn.contentChange = function(callback){\n    var elms = jQuery(this);\n    elms.each(\n      function(i){\n        var elm = jQuery(this);\n        elm.data("lastContents", elm.html());\n        window.watchContentChange = window.watchContentChange ? window.watchContentChange : [];\n        window.watchContentChange.push({"element": elm, "callback": callback});\n      }\n    )\n    return elms;\n  }\n  setInterval(function(){\n    if(window.watchContentChange){\n      for( i in window.watchContentChange){\n        if(window.watchContentChange[i].element.data("lastContents") != window.watchContentChange[i].element.html()){\n          window.watchContentChange[i].callback.apply(window.watchContentChange[i].element);\n          window.watchContentChange[i].element.data("lastContents", window.watchContentChange[i].element.html())\n        };\n      }\n    }\n  },500);\n\n\n    $(\'.links\').contentChange(function(){ \n      initSkimLinks();\n     \n      \n    });\n    \n</script>\n'
    }), this.JST["discover/tags"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e = "";
            return e += '\n  <li><a href="/tag/', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === j ? d.apply(a) : d), e += k(d) + '">', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === j ? d.apply(a) : d), e += k(d) + "</a>\n  ", d = c["if"].call(a, a.bookmark_count, {
                hash: {},
                inverse: l.noop,
                fn: l.program(2, g, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n  </li>\n"
        }
        function g(a, b) {
            var d, e = "";
            return e += '\n    <small class="text-muted" title="', (d = c.bookmark_count) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.bookmark_count, d = typeof d === j ? d.apply(a) : d), e += k(d) + ' links this week">', (d = c.bookmark_count) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.bookmark_count, d = typeof d === j ? d.apply(a) : d), e += k(d) + "</small>\n  "
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var h, i = "", j = "function", k = this.escapeExpression, l = this;
        return i += '<ul class="list-inline">\n  <li><b>Your frequent tags on Delicious</b></li>\n', h = c.each.call(b, b.items, {
            hash: {},
            inverse: l.noop,
            fn: l.program(1, f, e),
            data: e
        }), (h || 0 === h) && (i += h), i += "\n</ul>\n"
    }), this.JST["home/index"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<header class="header">\n      <a href="#" class="logo-delicious">Delicious</a>\n      \n      <div class="header-inner">\n        <div class="nav-utilities">\n          <ul>\n            <li>\n              <a href="#login" class="link-sign-in">Sign In</a>\n            </li>\n            \n            <li>\n              <a href="#" class="link-get">Get the App</a>\n            </li>\n            \n            <li>\n              <a href="#signup" class="link-sign-up">Sign Up</a>\n            </li>\n          </ul>\n        </div><!-- /.nav-utilities -->\n      </div><!-- /.header-inner -->\n    </header><!-- /.header -->\n  \n    <section class="intro">\n      <div class="intro-content">\n        <h1>Never lose a link again.</h1>\n      \n        <p>Delicious is a free and easy tool to save, organize, <br />and discover interesting links on the web.</p>\n      \n        <div class="intro-actions">\n          <a href="#signup">Sign Up</a>\n      \n          <span>or Learn More or <a href="#login">Sign In</a></span>\n        </div><!-- /.intro-actions -->\n      </div><!-- /.intro-content -->\n    </section><!-- /.intro -->\n  \n    <main class="main">\n      <div class="main-head">\n        <p>Popular Tags</p>\n      \n        <ul class="list-tags">\n          \n        </ul><!-- /.list-tags -->\n      </div><!-- /.main-head -->\n      <div class="ad-sponsors-wrapper" style="margin-left:auto;margin-right:auto;">\n\n<div class="ad-sponsors" style="margin-left:auto;margin-right:auto;width:970px;">\n<!-- Delicious_728x90_ATF -->\n<div id=\'div-gpt-ad-1416871411588-0\' style=\'width:728px; height:90px;\'>\n<script type=\'text/javascript\'>\ngoogletag.cmd.push(function() { googletag.display(\'div-gpt-ad-1416871411588-0\'); });\n</script>\n</div>\n      <div class="main-body">        \n        <ul class="trendings">\n\n        </ul><!-- /.trendings -->\n      </div><!-- /.main-body -->\n    </main><!-- /.main -->    \n\n    <div class="landing-wrapper">\n  <div class="landing landing-footer">\n    <ul class="pull-left">\n      <li>&copy;  <span class="mobile-hide">Delicious Science, LLC</span></li>\n      <li class="mobile-hide"><a href="/terms">Terms</a></li>\n      <li class="mobile-hide"><a href="/privacy">Privacy</a></li>\n    </ul>\n    <ul class="pull-right">\n      <li><a href="/apps">Apps</a></li>\n      <li class="mobile-hide"><a href="/help">Help</a></li>\n      <li><a href="http://blog.delicious.com/">Blog</a></li>\n      <li><a href="http://twitter.com/delicious" class="d-icon-twitter twitter"></a></li>\n      <li><a href="http://facebook.com/delicious" class="d-icon-facebook facebook"></a></li>\n      <li><a href="https://plus.google.com/+delicious" class="d-icon-google google"></a></li>\n    </ul>\n  </div>\n</div> \n    <script>\n$(document).ready(function() {\n    if ($.cookie("avid") === null)\n    {\n        if ($.cookie("david") === null)\n        {\n            var avid = randomString(32, \'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\');\n            $.cookie("david", avid, {domain: \'.delicious.com\'});\n        }        \n    }\n    $.ajax({\n        type: "GET",\n        url: "//votingapi.delicious.com/trending",\n        xhrFields: {\n            withCredentials: true\n        },\n        success: function(data) {\n            var links = data.links;\n            update_trending(links);\n            update_tags(data.tags);\n            ga(\'send\', \'event\', \'trending-view\', \'home-page\',\'impression\');\n\n        },\n        error: function(data) {\n            console.log(data.links);\n        }\n    });\n    $(\'.trendings\').delegate(\'.upvote\', \'click\', function() {\n        if (!$(this).hasClass(\'already_voted\'))\n        {\n            vote_link($(this).attr(\'id\'), \'up\');\n        }\n        return false;\n    });\n    $(\'.trendings\').delegate(\'.downvote\', \'click\', function() {\n        if (!$(this).hasClass(\'already_voted\'))\n        {\n            vote_link($(this).attr(\'id\'), \'down\');\n        }\n        return false;\n    });    \n});\n\nfunction update_tags(tags)\n{\n    var popular_tags = "";\n    for (i = 0; i < tags.length; i++) {\n        popular_tags += "<li><a href=\'/tag/"+tags[i]+"\'>" + (tags[i].charAt(0).toUpperCase() + tags[i].substring(1)) + "</a></li>";\n    }\n    $(\'.list-tags\').html("");\n    $(\'.list-tags\').html(popular_tags);\n}\nfunction vote_link(url_md5, type)\n{\n    $.ajax({\n        type: "POST",\n        url: "//votingapi.delicious.com/vote/" + url_md5 + \'/\' + type,\n        xhrFields: {\n            withCredentials: true\n        },\n        success: function(data) {\n            data=JSON.parse(data);\n            if (data.saved)\n            {\n                var class_btn = ".vote-btn-" + url_md5;\n                $(class_btn).addClass(\'already_voted\');\n                $(class_btn).addClass(\'voted-\'+type);\n                var count_vote = "#votes-count-" + url_md5;\n                $(count_vote).html(nFormatter(data.num_votes));\n                ga(\'send\', \'event\', \'trending-vote\', \'home-page\', type);\n            }\n        }\n    });\n}\n\nfunction update_trending(links)\n{\n    var trending = \'\';\n    for (i = 0; i < links.length; i++) {\n        var upvote = "";\n        var downvote = "";\n        trending += \'<li class="trending"><div class = "trending-actions">\';\n        if (links[i].my_vote !== undefined)\n        {\n            if (links[i].my_vote)\n            {\n                upvote = \'<a href = "#" class="upvote voted-up already_voted vote-btn-\' + links[i].url_md5 + \'" id="\' + links[i].url_md5 + \'"></a>\';\n                downvote = \'<a href = "#" class="downvote already_voted vote-btn-\' + links[i].url_md5 + \'" id="\' + links[i].url_md5 + \'"></a></div>\';\n            }\n            else\n            {\n                downvote = \'<a href = "#" class="downvote voted-down already_voted vote-btn-\' + links[i].url_md5 + \'" id="\' + links[i].url_md5 + \'"></a></div>\';\n                upvote = \'<a href = "#" class="upvote already_voted vote-btn-\' + links[i].url_md5 + \'" id="\' + links[i].url_md5 + \'"></a>\';\n            }\n        }\n        else\n        {\n            upvote = \'<a href = "#" class="upvote vote-btn-\' + links[i].url_md5 + \'" id="\' + links[i].url_md5 + \'"></a>\';\n            downvote = \'<a href = "#" class="downvote vote-btn-\' + links[i].url_md5 + \'" id="\' + links[i].url_md5 + \'"></a></div>\';\n        }\n        trending += upvote;\n        trending += \'<span class="votes-count" id="votes-count-\' + links[i].url_md5 + \'">\' + nFormatter(links[i].num_votes) + \'</span>\';\n        trending += downvote;\n        trending += \'<div class ="trending-link"><a href = "\' + links[i].url + \'" class="trending-title">\' + links[i].title + \'</a></div><!--div class="trending-saves"><span id="count_save-\' + links[i].url_md5 + \'">\' + \' &nbsp;</span--></div></li>\';\n    }\n\n    $(\'.trendings\').html("");\n    $(\'.trendings\').html(trending);\n}\nfunction randomString(length, chars) {\n    var result = \'\';\n    for (var i = length; i > 0; --i)\n        result += chars[Math.round(Math.random() * (chars.length - 1))];\n    return result;\n}\nfunction nFormatter(num) {\n    if (num >= 1000000000) {\n        return (num / 1000000000).toFixed(1) + \'G\';\n    }\n    if (num >= 1000000) {\n        return (num / 1000000).toFixed(1) + \'M\';\n    }\n    if (num >= 1000) {\n        return (num / 1000).toFixed(1) + \'K\';\n    }\n    return num;\n}\n    </script>'
    }), this.JST["home/trending"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, "<div class=\"hero-unit\">\n  <h2>Trending</h2>\n  Community picked links based on your interests.  \n</div>\n<div class=\"wizard\"></div>\n<div class=\"links\">\n<main class=\"main\">\n      <div class=\"main-head\">\n        <p>Popular Tags</p>\n      \n        <ul class=\"list-tags\">\n          \n        </ul><!-- /.list-tags -->\n      </div><!-- /.main-head -->\n\n      <div class=\"ad-sponsors-wrapper\" style=\"margin-left:auto;margin-right:auto;\">\n\n<div class=\"ad-sponsors\" style=\"margin-left:auto;margin-right:auto;width:970px;\">\n<!-- Delicious_728x90_ATF -->\n<div id='div-gpt-ad-1416871411588-0' style='width:728px; height:90px;'>\n<script type='text/javascript'>\ngoogletag.cmd.push(function() { googletag.display('div-gpt-ad-1416871411588-0'); });\n</script>\n</div>\n      \n      <div class=\"main-body\">        \n        <ul class=\"trendings\">\n\n        </ul><!-- /.trendings -->\n      </div><!-- /.main-body -->\n    </main><!-- /.main -->\n\n</div>\n\n<script>\n$(document).ready(function() {\n    if ($.cookie(\"avid\") === null)\n    {\n        if ($.cookie(\"david\") === null)\n        {\n            var avid = randomString(32, '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ');\n            $.cookie(\"david\", avid, {domain: '.delicious.com'});\n        } \n    }\n    $.ajax({\n        type: \"GET\",\n        url: \"//votingapi.delicious.com/trending\",\n         xhrFields: {\n            withCredentials: true\n        },\n        success: function(data) {\n            var links = data.links;\n            update_trending(links);\n            update_tags(data.tags);\n            ga('send', 'event', 'trending-view', 'trending-page','impression');\n        },\n        error: function(data) {\n            console.log(data.links);\n        }\n    });\n    $('.trendings').delegate('.upvote', 'click', function() {\n        if (!$(this).hasClass('already_voted'))\n        {\n            vote_link($(this).attr('id'), 'up');\n        }\n        return false;\n    });\n    $('.trendings').delegate('.downvote', 'click', function() {\n        if (!$(this).hasClass('already_voted'))\n        {\n            vote_link($(this).attr('id'), 'down');\n        }\n        return false;\n    });\n});\n\nfunction update_tags(tags)\n{\n    var popular_tags = \"\";\n    for (i = 0; i < tags.length; i++) {\n        popular_tags += \"<li><a href='/tag/\"+tags[i]+\"'>\" + (tags[i].charAt(0).toUpperCase() + tags[i].substring(1)) + \"</a></li>\";\n    }\n    $('.list-tags').html(\"\");\n    $('.list-tags').html(popular_tags);\n}\nfunction vote_link(url_md5, type)\n{\n    $.ajax({\n        type: \"POST\",\n        url: \"//votingapi.delicious.com/vote/\" + url_md5 + '/' + type,\n        xhrFields: {\n            withCredentials: true\n        },\n        success: function(data) {\n            data=JSON.parse(data);\n            if (data.saved)\n            {\n                var class_btn = \".vote-btn-\" + url_md5;\n                console.log(class_btn);\n                $(class_btn).addClass('already_voted');\n                $(class_btn).addClass('voted-'+type);\n                var count_vote = \"#votes-count-\" + url_md5;\n                $(count_vote).html(nFormatter(data.num_votes));\n                ga('send', 'event', 'trending-vote', 'trending-page',type);\n\n            }\n        }\n    });\n}\n\nfunction update_trending(links)\n{\n    var trending = '';\n    for (i = 0; i < links.length; i++) {\n        var upvote = \"\";\n        var downvote = \"\";\n        trending += '<li class=\"trending\"><div class = \"trending-actions\">';\n        if (links[i].my_vote !== undefined)\n        {\n            if (links[i].my_vote)\n            {\n                upvote = '<a href = \"#\" class=\"upvote voted-up already_voted vote-btn-' + links[i].url_md5 + '\" id=\"' + links[i].url_md5 + '\"></a>';\n                downvote = '<a href = \"#\" class=\"downvote already_voted vote-btn-' + links[i].url_md5 + '\" id=\"' + links[i].url_md5 + '\"></a></div>';\n            }\n            else\n            {\n                downvote = '<a href = \"#\" class=\"downvote voted-down already_voted vote-btn-' + links[i].url_md5 + '\" id=\"' + links[i].url_md5 + '\"></a></div>';\n                upvote = '<a href = \"#\" class=\"upvote already_voted vote-btn-' + links[i].url_md5 + '\" id=\"' + links[i].url_md5 + '\"></a>';\n            }\n        }\n        else\n        {\n            upvote = '<a href = \"#\" class=\"upvote vote-btn-' + links[i].url_md5 + '\" id=\"' + links[i].url_md5 + '\"></a>';\n            downvote = '<a href = \"#\" class=\"downvote vote-btn-' + links[i].url_md5 + '\" id=\"' + links[i].url_md5 + '\"></a></div>';\n        }\n        trending += upvote;\n        trending += '<span class=\"votes-count\" id=\"votes-count-' + links[i].url_md5 + '\">' + nFormatter(links[i].num_votes) + '</span>';\n        trending += downvote;\n        trending += '<div class =\"trending-link\"><a href = \"' + links[i].url + '\" class=\"trending-title\">' + links[i].title + '</a></div><div class=\"trending-saves\"><span id=\"count_save-' + links[i].url_md5 + '\">'  + ' &nbsp;</span></div></li>';\n    }\n\n    $('.trendings').html(\"\");\n    $('.trendings').html(trending);\n}\nfunction randomString(length, chars) {\n    var result = '';\n    for (var i = length; i > 0; --i)\n        result += chars[Math.round(Math.random() * (chars.length - 1))];\n    return result;\n}\nfunction nFormatter(num) {\n    if (num >= 1000000000) {\n        return (num / 1000000000).toFixed(1) + 'G';\n    }\n    if (num >= 1000000) {\n        return (num / 1000000).toFixed(1) + 'M';\n    }\n    if (num >= 1000) {\n        return (num / 1000).toFixed(1) + 'K';\n    }\n    return num;\n}\n    </script>\n"
    }), this.JST.html = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="link-nav">\n  <ul class="actions">\n    <li data-page="about"><a href="/about">About</a></li>\n    <li data-page="help"><a href="/help">Help</a></li>\n    <li data-page="apps"><a href="/apps">Apps</a></li>\n    <li data-page="tools"><a href="/tools">Tools</a></li>\n    <li data-page="shortcuts"><a href="/shortcuts">Shortcuts</a></li>\n    <li data-page="rss"><a href="/rss">RSS</a></li>\n    <li data-page="developers"><a href="/developers">Developers</a></li>\n    <li data-page="branding"><a href="/branding">Branding</a></li>\n    <li data-page="terms"><a href="/terms">Terms</a></li>\n    <li data-page="privacy"><a href="/privacy">Privacy</a></li>\n    <li data-page="copyright"><a href="/copyright">Copyright</a></li>\n  </ul>\n</div>\n\n<div class="static-html-wrapper"></div>\n\n'
    }), this.JST["links/compose"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="modal-content-alt"></div>\n'
    }), this.JST["links/date"] = a.template(function(a, b, c, d, e) {
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var f, g, h = "", i = c.helperMissing, j = this.escapeExpression;
        return g = {
            hash: {},
            data: e
        }, h += j((f = c.monthMap || b.monthMap, f ? f.call(b, b, g) : i.call(b, "monthMap", b, g))) + "\n"
    }), this.JST["links/link"] = a.template(function(a, b, c, d, e) {
        function f() {
            return " private"
        }
        function g(a, b) {
            var d, e, f = "";
            return f += '\n    <div class="note">', e = {
                hash: {},
                data: b
            }, f += E((d = c.decode || a.decode, d ? d.call(a, a.note, e) : D.call(a, "decode", a.note, e))) + "</div>\n  "
        }
        function h(a, b) {
            var d, e = "";
            return e += "\n      ", d = c["if"].call(a, a.save_rank, {
                hash: {},
                inverse: G.program(8, j, b),
                fn: G.program(6, i, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n        "
        }
        function i(a, b) {
            var d, e, f, g = "";
            return g += "\n        ", (d = c.owner) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.owner, d = typeof d === F ? d.apply(a) : d), g += E(d) + " is the ", f = {
                hash: {},
                data: b
            }, g += E((d = c.ordinal || a.ordinal, d ? d.call(a, a.save_rank, f) : D.call(a, "ordinal", a.save_rank, f))) + " of ", (e = c.num_saves) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.num_saves, e = typeof e === F ? e.apply(a) : e), g += E(e) + " to save this link"
        }
        function j(a, b) {
            var d, e, f, g = "";
            return (d = c.num_saves) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.num_saves, d = typeof d === F ? d.apply(a) : d), g += E(d) + " ", f = {
                hash: {
                    compare: "1"
                },
                inverse: G.program(11, l, b),
                fn: G.program(9, k, b),
                data: b
            }, d = c.if_gt || a.if_gt, e = d ? d.call(a, a.num_saves, f) : D.call(a, "if_gt", a.num_saves, f), (e || 0 === e) && (g += e), g += " saved this link\n        "
        }
        function k() {
            return "people"
        }
        function l() {
            return "person"
        }
        function m(a, b) {
            var d, e, f, g = "";
            return g += "\n        ", (d = c.num_saves) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.num_saves, d = typeof d === F ? d.apply(a) : d), g += E(d) + " ", f = {
                hash: {
                    compare: "1"
                },
                inverse: G.program(11, l, b),
                fn: G.program(9, k, b),
                data: b
            }, d = c.if_gt || a.if_gt, e = d ? d.call(a, a.num_saves, f) : D.call(a, "if_gt", a.num_saves, f), (e || 0 === e) && (g += e), g += " saved this link\n      "
        }
        function n(a, b) {
            var d, e = "";
            return e += "<span ", d = c["if"].call(a, a.save_rank, {
                hash: {},
                inverse: G.noop,
                fn: G.program(16, o, b),
                data: b
            }), (d || 0 === d) && (e += d), e += ">", (d = c.save_rank) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.save_rank, d = typeof d === F ? d.apply(a) : d), e += E(d) + "</span>"
        }
        function o() {
            return 'class="has-personal"'
        }
        function p(a, b) {
            var d, e = "";
            return e += "\n      ", d = c["with"].call(a, a.first_saver, {
                hash: {},
                inverse: G.noop,
                fn: G.program(19, q, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n    "
        }
        function q(a, b) {
            var d, e = "";
            return e += '\n      <div class="link-rank-early-bird">\n        <a href="/', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === F ? d.apply(a) : d), e += E(d) + '" class="early-bird" title="<b>', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === F ? d.apply(a) : d), e += E(d) + '</b> was the first user to save this link."><img src="', (d = c.avatar_url) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.avatar_url, d = typeof d === F ? d.apply(a) : d), e += E(d) + '" class="avatar-img"></a>\n      </div>\n      '
        }
        function r(a, b, d) {
            var e, f = "";
            return f += '\n      <ul class="tags">\n        ', e = c.each.call(a, a.tags, {
                hash: {},
                inverse: G.noop,
                fn: G.programWithDepth(22, s, b, a, d),
                data: b
            }), (e || 0 === e) && (f += e), f += "\n      </ul>\n    "
        }
        function s(a, b, d, e) {
            var f, g = "";
            return g += "\n          ", f = c["if"].call(a, e.by_user, {
                hash: {},
                inverse: G.program(26, v, b),
                fn: G.programWithDepth(23, t, b, d),
                data: b
            }), (f || 0 === f) && (g += f), g += "\n        "
        }
        function t(a, b, d) {
            var e, f = "";
            return f += '\n          <li class="dropdown">\n            <a href="#" class="dropdown-toggle" data-toggle="dropdown">' + E(typeof a === F ? a.apply(a) : a) + '</a>\n            <ul class="dropdown-menu" role="menu" aria-labelledby="dLabel" data-tag="' + E(typeof a === F ? a.apply(a) : a) + '">\n              ', e = c["if"].call(a, d.filtering, {
                hash: {},
                inverse: G.noop,
                fn: G.program(24, u, b),
                data: b
            }), (e || 0 === e) && (f += e), f += '\n              <li><a href="/' + E((e = d.owner, typeof e === F ? e.apply(a) : e)) + "/" + E(typeof a === F ? a.apply(a) : a) + '" class="tag-by-people"\n              data-tag="' + E(typeof a === F ? a.apply(a) : a) + '">Filter by #<b>' + E(typeof a === F ? a.apply(a) : a) + '</b></a></li>\n              <li><a href="/tag/' + E(typeof a === F ? a.apply(a) : a) + '" class="tag-by-global">#<b>' + E(typeof a === F ? a.apply(a) : a) + "</b> on Delicious</a></li>\n            </ul>\n          </li>\n          "
        }
        function u(a) {
            var b = "";
            return b += '\n                <li><a href="#" class="tag-for-filter" data-tag="' + E(typeof a === F ? a.apply(a) : a) + '">Add #<b>' + E(typeof a === F ? a.apply(a) : a) + "</b> to filter</a></li>\n              "
        }
        function v(a) {
            var b = "";
            return b += '\n          <li><a href="/tag/' + E(typeof a === F ? a.apply(a) : a) + '">' + E(typeof a === F ? a.apply(a) : a) + "</a></li>\n          "
        }
        function w() {
            return '\n        <a href="#" class="action-link bookmark-btn">Add Link</a>\n      '
        }
        function x() {
            return '\n        <a href="#link-tags" class="action-link mobile-tags-toggle">Tags</a>\n      '
        }
        function y() {
            return '\n        <a href="#edit" class="action-link edit">Edit</a>\n        <a href="#delete" class="action-link delete">Delete</a>\n      '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var z, A, B, C = "", D = c.helperMissing, E = this.escapeExpression, F = "function", G = this;
        return C += '<div class="item', z = c["if"].call(b, b["private"], {
            hash: {},
            inverse: G.noop,
            fn: G.program(1, f, e),
            data: e
        }), (z || 0 === z) && (C += z), C += '"  data-md5="', (z = c.md5) ? z = z.call(b, {
            hash: {},
            data: e
        }) : (z = b.md5, z = typeof z === F ? z.apply(b) : z), C += E(z) + '">\n  <div class="title-wrapper">\n    <a href="', (z = c.url) ? z = z.call(b, {
            hash: {},
            data: e
        }) : (z = b.url, z = typeof z === F ? z.apply(b) : z), C += E(z) + '" class="title">', B = {
            hash: {},
            data: e
        }, C += E((z = c.or || b.or, z ? z.call(b, b.title, b.url, B) : D.call(b, "or", b.title, b.url, B))) + '</a>\n    <span class="domain">', (A = c.domain) ? A = A.call(b, {
            hash: {},
            data: e
        }) : (A = b.domain, A = typeof A === F ? A.apply(b) : A), C += E(A) + "</span>\n  </div>\n\n  ", A = c["if"].call(b, b.note, {
            hash: {},
            inverse: G.noop,
            fn: G.program(3, g, e),
            data: e
        }), (A || 0 === A) && (C += A), C += '\n\n  <div class="meta">\n    <a href="/link/', (A = c.md5) ? A = A.call(b, {
            hash: {},
            data: e
        }) : (A = b.md5, A = typeof A === F ? A.apply(b) : A), C += E(A) + '" class="link-rank"\n    title="\n    ', A = c["if"].call(b, b.by_user, {
            hash: {},
            inverse: G.program(13, m, e),
            fn: G.program(5, h, e),
            data: e
        }), (A || 0 === A) && (C += A), C += '">\n      ', A = c["if"].call(b, b.save_rank, {
            hash: {},
            inverse: G.noop,
            fn: G.program(15, n, e),
            data: e
        }), (A || 0 === A) && (C += A), C += "<span>", (A = c.num_saves) ? A = A.call(b, {
            hash: {},
            data: e
        }) : (A = b.num_saves, A = typeof A === F ? A.apply(b) : A), C += E(A) + "</span>\n    </a>\n\n    ", A = c.unless.call(b, b.by_user, {
            hash: {},
            inverse: G.noop,
            fn: G.program(18, p, e),
            data: e
        }), (A || 0 === A) && (C += A), C += "\n\n    ", A = c["if"].call(b, b.tags, {
            hash: {},
            inverse: G.noop,
            fn: G.programWithDepth(21, r, e, b),
            data: e
        }), (A || 0 === A) && (C += A), C += '\n  </div>\n\n  <div class="meta meta-secondary">\n    <div class="actions">\n      ', A = c.unless.call(b, b.editable, {
            hash: {},
            inverse: G.noop,
            fn: G.program(28, w, e),
            data: e
        }), (A || 0 === A) && (C += A), C += "\n      ", A = c["if"].call(b, b.tags, {
            hash: {},
            inverse: G.noop,
            fn: G.program(30, x, e),
            data: e
        }), (A || 0 === A) && (C += A), C += '\n      <div class="share-btn">\n        <div class="dropdown">\n          <a href="#" data-toggle="dropdown" class="action-link dropdown-toggle">Share</a>\n          <ul class="dropdown-menu">\n            <li><a href="https://twitter.com/intent/tweet?text=', (A = c.title) ? A = A.call(b, {
            hash: {},
            data: e
        }) : (A = b.title, A = typeof A === F ? A.apply(b) : A), C += E(A) + "&amp;url=", B = {
            hash: {},
            data: e
        }, C += E((z = c.URLEncode || b.URLEncode, z ? z.call(b, b.url, B) : D.call(b, "URLEncode", b.url, B))) + '&amp;via=delicious" class="share-social noscript">Twitter</a></li>\n            <li><a href="http://www.facebook.com/share.php?u=', B = {
            hash: {},
            data: e
        }, C += E((z = c.URLEncode || b.URLEncode, z ? z.call(b, b.url, B) : D.call(b, "URLEncode", b.url, B))) + '" class="share-social noscript">Facebook</a></li>\n            <li><a href="https://plus.google.com/share?url=', B = {
            hash: {},
            data: e
        }, C += E((z = c.URLEncode || b.URLEncode, z ? z.call(b, b.url, B) : D.call(b, "URLEncode", b.url, B))) + '" class="share-social noscript">Google+</a></li>\n            <li><a href="http://getpocket.com/edit.php?url=', B = {
            hash: {},
            data: e
        }, C += E((z = c.URLEncode || b.URLEncode, z ? z.call(b, b.url, B) : D.call(b, "URLEncode", b.url, B))) + "&amp;title=", B = {
            hash: {},
            data: e
        }, C += E((z = c.URLEncode || b.URLEncode, z ? z.call(b, b.title, B) : D.call(b, "URLEncode", b.title, B))) + '" class="share-social noscript">Pocket</a></li>\n            <li><a href="http://www.instapaper.com/edit?url=', B = {
            hash: {},
            data: e
        }, C += E((z = c.URLEncode || b.URLEncode, z ? z.call(b, b.url, B) : D.call(b, "URLEncode", b.url, B))) + "&amp;title=", B = {
            hash: {},
            data: e
        }, C += E((z = c.URLEncode || b.URLEncode, z ? z.call(b, b.title, B) : D.call(b, "URLEncode", b.title, B))) + '" class="share-social noscript">Instapaper</a></li>\n            <li><a href="mailto:?subject=', B = {
            hash: {},
            data: e
        }, C += E((z = c.URLEncode || b.URLEncode, z ? z.call(b, b.title, B) : D.call(b, "URLEncode", b.title, B))) + "&amp;body=", B = {
            hash: {},
            data: e
        }, C += E((z = c.URLEncode || b.URLEncode, z ? z.call(b, b.url, B) : D.call(b, "URLEncode", b.url, B))) + '" class="share-social noscript">Email</a></li>\n          </ul>\n        </div>\n      </div>\n      ', A = c["if"].call(b, b.editable, {
            hash: {},
            inverse: G.noop,
            fn: G.program(32, y, e),
            data: e
        }), (A || 0 === A) && (C += A), C += '\n    </div>\n\n    <time datetime="', B = {
            hash: {},
            data: e
        }, C += E((z = c.formatMachineDate || b.formatMachineDate, z ? z.call(b, b.time_created, B) : D.call(b, "formatMachineDate", b.time_created, B))) + '" class="date" title="', B = {
            hash: {},
            data: e
        }, C += E((z = c.formatDateFull || b.formatDateFull, z ? z.call(b, b.time_created, B) : D.call(b, "formatDateFull", b.time_created, B))) + '">', B = {
            hash: {},
            data: e
        }, C += E((z = c.formatDate || b.formatDate, z ? z.call(b, b.time_created, B) : D.call(b, "formatDate", b.time_created, B))) + '</time>\n  </div>\n\n  <div class="bulk-edit-indicator">\n    <span class="d-icon-checkmark"></span>\n  </div>\n</div>\n<!-- .item -->\n'
    }), this.JST["links/list"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="link-list"></div>\n<div class="item load-more-link"><a href="#loadmore">Click to load more</a></div>\n'
    }), this.JST["links/note"] = a.template(function(a, b, c, d, e) {
        function f() {
            return ":"
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var g, h = "", i = "function", j = this.escapeExpression, k = this;
        return h += '<a href="/', (g = c.username) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.username, g = typeof g === i ? g.apply(b) : g), h += j(g) + '" data-toggle="tooltip" title="<b>', (g = c.full_name) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.full_name, g = typeof g === i ? g.apply(b) : g), h += j(g) + "</b>", g = c["if"].call(b, b.note, {
            hash: {},
            inverse: k.noop,
            fn: k.program(1, f, e),
            data: e
        }), (g || 0 === g) && (h += g), h += " ", (g = c.note) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.note, g = typeof g === i ? g.apply(b) : g), h += j(g) + '">\n  <img src="', (g = c.avatar_url) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.avatar_url, g = typeof g === i ? g.apply(b) : g), h += j(g) + '" class="avatar-img">\n</a>\n'
    }), this.JST["links/notes"] = a.template(function(a, b, c, d, e) {
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var f, g = "", h = "function", i = this.escapeExpression;
        return g += '<div class="note-list"></div>\n', (f = c.loadMore) ? f = f.call(b, {
            hash: {},
            data: e
        }) : (f = b.loadMore, f = typeof f === h ? f.apply(b) : f), g += i(f) + "\n"
    }), this.JST["links/page"] = a.template(function(a, b, c, d, e) {
        function f() {
            return '<div class="user-info"></div>'
        }
        function g() {
            return '<div class="tag-info"></div>'
        }
        function h(a, b) {
            var d, e = "";
            return e += '\n      <div class="info">\n        <h2 class="name">\n          Results of <span>', (d = c.q) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.q, d = typeof d === t ? d.apply(a) : d), e += u(d) + "</span>\n        </h2>\n      </div>\n      <div class=\"ad-sponsors\">\n<!-- Delicious_728x90_ATF -->\n<div id='div-gpt-ad-1416871411588-0' style='width:728px; height:90px;'>\n<script type='text/javascript'>\ngoogletag.cmd.push(function() { googletag.display('div-gpt-ad-1416871411588-0'); });\n</script>\n</div>\n    "
        }
        function i(a, b) {
            var d, e = "";
            return e += '\n      <ul class="actions user-actions">\n        <li class="action-link-wrapper action-tags-wrapper">\n          <a href="#tags">Tags</a>\n          <div class="tags-edit-dropdown dropdown">\n            <a href="#" data-toggle="dropdown">\n              <i class="d-icon-settings"></i>\n            </a>\n            <ul class="dropdown-menu">\n              <li><a href="#" class="sort-by-alphabet">Sort by alphabet</a></li>\n              <li><a href="#" class="sort-by-count">Sort by count</a></li>\n              ', d = c["if"].call(a, a.editable, {
                hash: {},
                inverse: v.noop,
                fn: v.program(8, j, b),
                data: b
            }), (d || 0 === d) && (e += d), e += '\n            </ul>\n          </div>\n        </li>\n\n        <li class="action-link-wrapper action-tagbundles-wrapper">\n          <a href="#tagbundles"><span class="mobile-hide">Tag </span>Bundles</a>\n          ', d = c["if"].call(a, a.editable, {
                hash: {},
                inverse: v.noop,
                fn: v.program(10, k, b),
                data: b
            }), (d || 0 === d) && (e += d), e += '\n        </li>\n        <li>\n          <a href="#datetable">Date</a>\n        </li>\n        <li>\n          <ul class="action-group filter-action-group">\n            <li class="dropdown">\n              <a class="dropdown-toggle" data-toggle="dropdown" data-target="#" href="#">\n                Extra Filters\n              </a>\n              <ul class="dropdown-menu">\n                <li class="action-link-wrapper action-tagbundles-wrapper">\n                  <a href="#" class="ftag ftag-notag">Untagged</a>\n                </li>\n                ', d = c["if"].call(a, a.editable, {
                hash: {},
                inverse: v.noop,
                fn: v.program(12, l, b),
                data: b
            }), (d || 0 === d) && (e += d), e += '\n              </ul>\n            </li>\n          </ul>\n        </li>\n      </ul>\n\n      <ul class="actions">\n        ', d = c["if"].call(a, a.editable, {
                hash: {},
                inverse: v.noop,
                fn: v.program(14, m, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n      </ul>\n    "
        }
        function j() {
            return '\n                <li><a class="edit-tag" href="#">Edit tags&hellip;</a></li>\n              '
        }
        function k() {
            return '\n            <div class="tags-edit-dropdown dropdown">\n              <a href="#" data-toggle="dropdown">\n                <i class="d-icon-settings"></i>\n              </a>\n              <ul class="dropdown-menu">\n                <li><a href="#edit-tag-bundle">Create new bundle</a></li>\n              </ul>\n            </div>\n          '
        }
        function l() {
            return '\n                <li class="action-link-wrapper action-tagbundles-wrapper">\n                  <a href="#" class="ftag ftag-public">Public links</a>\n                </li>\n                <li class="action-link-wrapper action-tagbundles-wrapper">\n                  <a href="#" class="ftag ftag-private">Private links</a>\n                </li>\n                '
        }
        function m() {
            return '\n          <li>\n            <ul class="action-group bulk-edit-action-group">\n              <li class="dropdown">\n                <a class="dropdown-toggle" data-toggle="dropdown" data-target="#" href="#">\n                  <span class="bulk-edit-selection-count">0</span> Selected <i class="caret"></i>\n                </a>\n                <ul class="dropdown-menu">\n                  <li><a href="#" class="bulk-edit-action-public">Make Public</a></li>\n                  <li><a href="#" class="bulk-edit-action-private">Make Private</a></li>\n                  <li><a href="#" class="bulk-edit-action-remove">Delete Links</a></li>\n                  <li><a href="#" class="bulk-edit-action-add-tag">Add Tags</a></li>\n                  <li><a href="#" class="bulk-edit-action-remove-tag">Remove Tag</a></li>\n                  <li class="divider" role="presentation"></li>\n                  <li><a href="#" class="bulk-edit-action-select-all">Select Visible</a></li>\n                  <li><a href="#" class="bulk-edit-action-select-none">Clear Selection</a></li>\n                </ul>\n              </li>\n            </ul>\n          </li>\n          <li class="pull-right">\n            <a href="#" class="bulk-edit" data-toggle="button">Edit</a>\n          </li>\n        '
        }
        function n(a, b) {
            var d, e = "";
            return e += '\n      <ul class="actions">\n        <li><a href="/tag/', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === t ? d.apply(a) : d), e += u(d) + '/popular" data-tab="hot">Popular</a></li>\n        <li><a href="/tag/', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === t ? d.apply(a) : d), e += u(d) + '/recent" data-tab="recent">Recent</a></li>\n        <li><a href="/tag/', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === t ? d.apply(a) : d), e += u(d) + '/alltime" data-tab="alltime">All Time</a></li>\n      </ul>\n    '
        }
        function o(a, b) {
            var d, e, f = "";
            return f += '\n      <ul class="actions">\n        <li><a href="/search/', (d = c.q) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.q, d = typeof d === t ? d.apply(a) : d), f += u(d) + '/everyone" data-tab="everyone">Everyone</a></li>\n        ', e = {
                hash: {},
                inverse: v.noop,
                fn: v.program(19, p, b),
                data: b
            }, (d = c.ifLoggedIn) ? d = d.call(a, e) : (d = a.ifLoggedIn, d = typeof d === t ? d.apply(a) : d), c.ifLoggedIn || (d = w.call(a, d, e)), (d || 0 === d) && (f += d), f += "\n      </ul>\n    "
        }
        function p(a, b) {
            var d, e = "";
            return e += '\n          <li><a href="/search/', (d = c.q) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.q, d = typeof d === t ? d.apply(a) : d), e += u(d) + '/network" data-tab="network">Network</a></li>\n          <li><a href="/search/', (d = c.q) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.q, d = typeof d === t ? d.apply(a) : d), e += u(d) + '/mine" data-tab="you">My Links</a></li>\n        '
        }
        function q() {
            return '\n    <div class="tagcloud-wrapper">\n      <div class="tagcloud"></div>\n    </div>\n\n    <div class="tagbundles-wrapper">\n      <ul class="tagbundles"></ul>\n    </div>\n\n    <div class="datetable-wrapper">\n      <ul class="datetable"></ul>\n    </div>\n\n    <div class="filter">\n      Current Filter:\n      <span class="filter-cards"></span>\n    </div>\n\n    <div class="relatedtags-wrapper">\n      <ul class="relatedtags"></ul>\n    </div>\n  '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var r, s = "", t = "function", u = this.escapeExpression, v = this, w = c.blockHelperMissing;
        return s += '<div class="item bulk-edit-status">\n  You&rsquo;re bulk editing your links.\n</div>\n\n<div class="link-nav-fixed-wrapper">\n  <div class="hero-unit">\n    ', r = c["if"].call(b, b.is_user, {
            hash: {},
            inverse: v.noop,
            fn: v.program(1, f, e),
            data: e
        }), (r || 0 === r) && (s += r), s += "\n\n    ", r = c["if"].call(b, b.is_tag, {
            hash: {},
            inverse: v.noop,
            fn: v.program(3, g, e),
            data: e
        }), (r || 0 === r) && (s += r), s += "\n\n    ", r = c["if"].call(b, b.is_search, {
            hash: {},
            inverse: v.noop,
            fn: v.program(5, h, e),
            data: e
        }), (r || 0 === r) && (s += r), s += '\n  </div>\n\n  <div class="link-nav">\n    ', r = c["if"].call(b, b.is_user, {
            hash: {},
            inverse: v.noop,
            fn: v.program(7, i, e),
            data: e
        }), (r || 0 === r) && (s += r), s += " \n\n    ", r = c["if"].call(b, b.is_tag, {
            hash: {},
            inverse: v.noop,
            fn: v.program(16, n, e),
            data: e
        }), (r || 0 === r) && (s += r), s += "\n\n    ", r = c["if"].call(b, b.is_search, {
            hash: {},
            inverse: v.noop,
            fn: v.program(18, o, e),
            data: e
        }), (r || 0 === r) && (s += r), s += "\n  </div>\n\n  ", r = c["if"].call(b, b.is_user, {
            hash: {},
            inverse: v.noop,
            fn: v.program(21, q, e),
            data: e
        }), (r || 0 === r) && (s += r), s += '\n</div>\n<!-- .link-nav-fixed-wrapper -->\n\n<div class="links"></div>\n', (r = c.loadMore) ? r = r.call(b, {
            hash: {},
            data: e
        }) : (r = b.loadMore, r = typeof r === t ? r.apply(b) : r), s += u(r) + '\n\n\n\n<script>\njQuery.fn.contentChange = function(callback){\n    var elms = jQuery(this);\n    elms.each(\n      function(i){\n        var elm = jQuery(this);\n        elm.data("lastContents", elm.html());\n        window.watchContentChange = window.watchContentChange ? window.watchContentChange : [];\n        window.watchContentChange.push({"element": elm, "callback": callback});\n      }\n    )\n    return elms;\n  }\n  setInterval(function(){\n    if(window.watchContentChange){\n      for( i in window.watchContentChange){\n        if(window.watchContentChange[i].element.data("lastContents") != window.watchContentChange[i].element.html()){\n          window.watchContentChange[i].callback.apply(window.watchContentChange[i].element);\n          window.watchContentChange[i].element.data("lastContents", window.watchContentChange[i].element.html())\n        };\n      }\n    }\n  },500);\n\n\n    $(\'.links\').contentChange(function(){\n      initSkimLinks();\n      \n    });\n    \n</script>\n'
    }), this.JST["links/related_tags"] = a.template(function(a, b, c, d, e) {
        function f() {
            return "\n    Related tags:\n  "
        }
        function g() {
            return "\n    Tags in bundle:\n  "
        }
        function h(a, b, d) {
            var e, f = "";
            return f += '\n  <li><a href="#" data-tag="', (e = c.name) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.name, e = typeof e === m ? e.apply(a) : e), f += n(e) + '"\n    ', e = c.unless.call(a, d.is_bundle, {
                hash: {},
                inverse: o.program(8, j, b),
                fn: o.program(6, i, b),
                data: b
            }), (e || 0 === e) && (f += e), f += ">", (e = c.name) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.name, e = typeof e === m ? e.apply(a) : e), f += n(e) + "</a></li>\n"
        }
        function i() {
            return '\n      class="filter-tag"\n      '
        }
        function j() {
            return '\n      class="show-tag"\n    '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var k, l = "", m = "function", n = this.escapeExpression, o = this;
        return l += "<li>\n  ", k = c.unless.call(b, b.is_bundle, {
            hash: {},
            inverse: o.program(3, g, e),
            fn: o.program(1, f, e),
            data: e
        }), (k || 0 === k) && (l += k), l += "\n</li>\n", k = c.each.call(b, b.items, {
            hash: {},
            inverse: o.noop,
            fn: o.programWithDepth(5, h, e, b),
            data: e
        }), (k || 0 === k) && (l += k), l += "\n"
    }), this.JST["links/single"] = a.template(function(a, b, c, d, e) {
        function f() {
            return '\n  <div class="single-link"></div>\n\n  <div class="col-md-6">\n    <div class="item single-page-subtitle">\n      Related Links\n    </div>\n    <div class="related-links"></div>\n  </div>\n\n  <div class="col-md-6">\n    <div class="item single-page-subtitle">\n      People\n    </div>\n    <div class="comments"></div>\n    <div class="item notes"></div>\n  </div>\n\n'
        }
        function g() {
            return '\n  <div class="no-link-found">Delicious is processing this link, please try again later.</div>\n'
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var h, i = "", j = this;
        return h = c["if"].call(b, b.url, {
            hash: {},
            inverse: j.program(3, g, e),
            fn: j.program(1, f, e),
            data: e
        }), (h || 0 === h) && (i += h), i += "\n"
    }), this.JST["links/single/comment"] = a.template(function(a, b, c, d, e) {
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var f, g = "", h = "function", i = this.escapeExpression;
        return g += '<div class="item">\n  <p class="title-wrapper">\n    <img src="', (f = c.avatar_url) ? f = f.call(b, {
            hash: {},
            data: e
        }) : (f = b.avatar_url, f = typeof f === h ? f.apply(b) : f), g += i(f) + '" class="avatar-img">\n    <a href="/', (f = c.username) ? f = f.call(b, {
            hash: {},
            data: e
        }) : (f = b.username, f = typeof f === h ? f.apply(b) : f), g += i(f) + '"><b>', (f = c.full_name) ? f = f.call(b, {
            hash: {},
            data: e
        }) : (f = b.full_name, f = typeof f === h ? f.apply(b) : f), g += i(f) + '</b></a>\n  </p>\n\n  <p class="note">', (f = c.note) ? f = f.call(b, {
            hash: {},
            data: e
        }) : (f = b.note, f = typeof f === h ? f.apply(b) : f), g += i(f) + "</p>\n</div>\n"
    }), this.JST["links/single/comments"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="comment-list"></div>\n'
    }), this.JST["links/single/link"] = a.template(function(a, b, c, d, e) {
        function f() {
            var a = "";
            return a
        }
        function g() {
            return "s"
        }
        function h(a, b) {
            var d, e = "";
            return e += '\n      <ul class="tags"> ', d = c.each.call(a, a.tags, {
                hash: {},
                inverse: t.noop,
                fn: t.program(6, i, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "</ul>\n    "
        }
        function i(a) {
            var b = "";
            return b += '<li><a href="/tag/' + s(typeof a === r ? a.apply(a) : a) + '">' + s(typeof a === r ? a.apply(a) : a) + "</a></li> "
        }
        function j() {
            return '\n        <a href="#" class="action-link bookmark-btn">Add Link</a>\n      '
        }
        function k() {
            return '\n        <a href="#link-tags" class="action-link mobile-tags-toggle">Tags</a>\n      '
        }
        function l(a, b) {
            var d, e, f = "";
            return f += '\n  <div class="item">\n    <a href="/">\n      <img src="', (d = c.owner_avatar_url) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.owner_avatar_url, d = typeof d === r ? d.apply(a) : d), f += s(d) + '" class="avatar-img avatar-callout">\n      <b>You</b>\n    </a>\n    saved this link on ', e = {
                hash: {},
                data: b
            }, f += s((d = c.formatDateString || a.formatDateString, d ? d.call(a, (d = a.recommended, null == d || d===!1 ? d : d.previously_saved_time), e) : u.call(a, "formatDateString", (d = a.recommended, null == d || d===!1 ? d : d.previously_saved_time), e))) + '.\n    <a href="#" class="bookmark-btn">Edit</a>\n  </div>\n'
        }
        function m(a, b, d) {
            var e, f, g = "";
            return g += '\n  <div class="item">\n    <a href="/', (e = c.username) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.username, e = typeof e === r ? e.apply(a) : e), g += s(e) + '" class="early-bird">\n      <img src="', (e = c.avatar_url) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.avatar_url, e = typeof e === r ? e.apply(a) : e), g += s(e) + '" class="avatar-img avatar-callout">\n      <b>', (e = c.full_name) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.full_name, e = typeof e === r ? e.apply(a) : e), g += s(e) + "</b>\n    </a>\n    first saved this link on ", f = {
                hash: {},
                data: b
            }, g += s((e = c.formatDateString || d.formatDateString, e ? e.call(a, d.first_saver_date, f) : u.call(a, "formatDateString", d.first_saver_date, f))) + ".\n  </div>\n"
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var n, o, p, q = "", r = "function", s = this.escapeExpression, t = this, u = c.helperMissing;
        return q += '<div class="item item-details">\n  <h2 class="title">\n    <a href="', (n = c.url) ? n = n.call(b, {
            hash: {},
            data: e
        }) : (n = b.url, n = typeof n === r ? n.apply(b) : n), q += s(n) + '" target="_blank">', p = {
            hash: {},
            data: e
        }, q += s((n = c.or || b.or, n ? n.call(b, b.title, b.url, p) : u.call(b, "or", b.title, b.url, p))) + '</a> <span class="domain">', (o = c.domain) ? o = o.call(b, {
            hash: {},
            data: e
        }) : (o = b.domain, o = typeof o === r ? o.apply(b) : o), q += s(o) + '</a>\n  </h2>\n\n  <div class="meta">\n    <span class="link-rank">\n      <span><b>', (o = c.num_saves) ? o = o.call(b, {
            hash: {},
            data: e
        }) : (o = b.num_saves, o = typeof o === r ? o.apply(b) : o), q += s(o) + "</b>\n      Save", p = {
            hash: {
                compare: 1
            },
            inverse: t.program(3, g, e),
            fn: t.program(1, f, e),
            data: e
        }, n = c.if_eq || b.if_eq, o = n ? n.call(b, b.num_saves, p) : u.call(b, "if_eq", b.num_saves, p), (o || 0 === o) && (q += o), q += "</span>\n    </span>\n\n    ", o = c["if"].call(b, b.tags, {
            hash: {},
            inverse: t.noop,
            fn: t.program(5, h, e),
            data: e
        }), (o || 0 === o) && (q += o), q += '\n  </div>\n\n  <div class="meta meta-secondary">\n    <div class="actions">\n      ', o = c.unless.call(b, (n = b.recommended, null == n || n===!1 ? n : n.previously_saved), {
            hash: {},
            inverse: t.noop,
            fn: t.program(8, j, e),
            data: e
        }), (o || 0 === o) && (q += o), q += "\n      ", o = c["if"].call(b, b.tags, {
            hash: {},
            inverse: t.noop,
            fn: t.program(10, k, e),
            data: e
        }), (o || 0 === o) && (q += o), q += '\n    </div>\n    <div class="actions share-btn">\n      <div class="tags-edit-dropdown dropdown">\n        <a href="#" data-toggle="dropdown" class="action-link dropdown-toggle">Share</a>\n        <ul class="dropdown-menu">\n          <li><a href="https://twitter.com/intent/tweet?text=', (o = c.title) ? o = o.call(b, {
            hash: {},
            data: e
        }) : (o = b.title, o = typeof o === r ? o.apply(b) : o), q += s(o) + "&amp;url=", p = {
            hash: {},
            data: e
        }, q += s((n = c.URLEncode || b.URLEncode, n ? n.call(b, b.url, p) : u.call(b, "URLEncode", b.url, p))) + '&amp;via=delicious" class="share-social noscript">Twitter</a></li>\n          <li><a href="http://www.facebook.com/share.php?u=', p = {
            hash: {},
            data: e
        }, q += s((n = c.URLEncode || b.URLEncode, n ? n.call(b, b.url, p) : u.call(b, "URLEncode", b.url, p))) + '" class="share-social noscript">Facebook</a></li>\n          <li><a href="https://plus.google.com/share?url=', p = {
            hash: {},
            data: e
        }, q += s((n = c.URLEncode || b.URLEncode, n ? n.call(b, b.url, p) : u.call(b, "URLEncode", b.url, p))) + '" class="share-social noscript">Google+</a></li>\n          <li><a href="http://getpocket.com/edit.php?url=', p = {
            hash: {},
            data: e
        }, q += s((n = c.URLEncode || b.URLEncode, n ? n.call(b, b.url, p) : u.call(b, "URLEncode", b.url, p))) + "&amp;title=", p = {
            hash: {},
            data: e
        }, q += s((n = c.URLEncode || b.URLEncode, n ? n.call(b, b.title, p) : u.call(b, "URLEncode", b.title, p))) + '" class="share-social noscript">Pocket</a></li>\n          <li><a href="http://www.instapaper.com/edit?url=', p = {
            hash: {},
            data: e
        }, q += s((n = c.URLEncode || b.URLEncode, n ? n.call(b, b.url, p) : u.call(b, "URLEncode", b.url, p))) + "&amp;title=", p = {
            hash: {},
            data: e
        }, q += s((n = c.URLEncode || b.URLEncode, n ? n.call(b, b.title, p) : u.call(b, "URLEncode", b.title, p))) + '" class="share-social noscript">Instapaper</a></li>\n          <li><a href="mailto:?subject=', p = {
            hash: {},
            data: e
        }, q += s((n = c.URLEncode || b.URLEncode, n ? n.call(b, b.title, p) : u.call(b, "URLEncode", b.title, p))) + "&amp;body=", p = {
            hash: {},
            data: e
        }, q += s((n = c.URLEncode || b.URLEncode, n ? n.call(b, b.url, p) : u.call(b, "URLEncode", b.url, p))) + '" class="share-social noscript">Email</a></li>\n        </ul>\n      </div>\n    </div>\n  </div>\n</div>\n\n', o = c["if"].call(b, (n = b.recommended, null == n || n===!1 ? n : n.previously_saved), {
            hash: {},
            inverse: t.noop,
            fn: t.program(12, l, e),
            data: e
        }), (o || 0 === o) && (q += o), q += "\n\n", o = c["with"].call(b, b.first_saver, {
            hash: {},
            inverse: t.noop,
            fn: t.programWithDepth(14, m, e, b),
            data: e
        }), (o || 0 === o) && (q += o), q += "\n"
    }), this.JST["links/single/related"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="link-list"></div>\n'
    }), this.JST["links/single/related_link"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e, f = "";
            return f += '\n    <div class="note">', e = {
                hash: {},
                data: b
            }, f += l((d = c.decode || a.decode, d ? d.call(a, a.note, e) : k.call(a, "decode", a.note, e))) + "</div>\n  "
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var g, h, i, j = "", k = c.helperMissing, l = this.escapeExpression, m = "function", n = this;
        return j += '<div class="item" data-md5="', (g = c.md5) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.md5, g = typeof g === m ? g.apply(b) : g), j += l(g) + '">\n  <div class="title-wrapper">\n    <a href="', (g = c.url) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.url, g = typeof g === m ? g.apply(b) : g), j += l(g) + '" class="title">', i = {
            hash: {},
            data: e
        }, j += l((g = c.or || b.or, g ? g.call(b, b.title, b.url, i) : k.call(b, "or", b.title, b.url, i))) + "</a>\n  </div>\n\n  ", h = c["if"].call(b, b.note, {
            hash: {},
            inverse: n.noop,
            fn: n.program(1, f, e),
            data: e
        }), (h || 0 === h) && (j += h), j += '\n\n  <div class="meta">\n    <a href="/link/', (h = c.md5) ? h = h.call(b, {
            hash: {},
            data: e
        }) : (h = b.md5, h = typeof h === m ? h.apply(b) : h), j += l(h) + '"><b>', (h = c.num_saves) ? h = h.call(b, {
            hash: {},
            data: e
        }) : (h = b.num_saves, h = typeof h === m ? h.apply(b) : h), j += l(h) + '</b> people saved this link</a>\n  </div>\n\n  <div class="meta meta-secondary">\n    <div class="actions">\n      <a href="#" class="action-link bookmark-btn">Add Link</a>\n    </div>\n    <time datetime="', i = {
            hash: {},
            data: e
        }, j += l((g = c.formatMachineDate || b.formatMachineDate, g ? g.call(b, b.time_created, i) : k.call(b, "formatMachineDate", b.time_created, i))) + '" class="date" title="', i = {
            hash: {},
            data: e
        }, j += l((g = c.formatDateFull || b.formatDateFull, g ? g.call(b, b.time_created, i) : k.call(b, "formatDateFull", b.time_created, i))) + '">', i = {
            hash: {},
            data: e
        }, j += l((g = c.formatDate || b.formatDate, g ? g.call(b, b.time_created, i) : k.call(b, "formatDate", b.time_created, i))) + "</time>\n  </div>\n</div>\n<!-- .item -->\n"
    }), this.JST["links/tag"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e = "";
            return e += '\n        <a href="#unfollow" class="follow-toggle follow-toggle-following" data-user="', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === n ? d.apply(a) : d), e += o(d) + '">\n          <span class="following-text">Subscribed</span>\n          <span class="unfollow-text">Ubsubscribe</span>\n        </a>\n        '
        }
        function g(a, b) {
            var d, e = "";
            return e += '\n        <a href="#follow" class="follow-toggle follow-toggle-not-following" data-user="', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === n ? d.apply(a) : d), e += o(d) + '">\n          <span class="follow-text">Subscribe</span>\n        </a>\n      '
        }
        function h() {
            return "link"
        }
        function i() {
            return "links"
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var j, k, l, m = "", n = "function", o = this.escapeExpression, p = this, q = c.helperMissing;
        return m += '<div class="info">\n  <div class="name-overflow-wrapper">\n    <h2 class="name pull-left">\n      Links for #<span>', (j = c.name) ? j = j.call(b, {
            hash: {},
            data: e
        }) : (j = b.name, j = typeof j === n ? j.apply(b) : j), m += o(j) + '</span>\n    </h2>\n\n    <div class="pull-right">\n      ', j = c["if"].call(b, b.subscribed, {
            hash: {},
            inverse: p.program(3, g, e),
            fn: p.program(1, f, e),
            data: e
        }), (j || 0 === j) && (m += j), m += '\n    </div>\n  </div>\n\n  <div class="user-stats">\n    <span><b>', (j = c.bookmark_count) ? j = j.call(b, {
            hash: {},
            data: e
        }) : (j = b.bookmark_count, j = typeof j === n ? j.apply(b) : j), m += o(j) + "</b>\n    ", l = {
            hash: {
                compare: "1"
            },
            inverse: p.program(7, i, e),
            fn: p.program(5, h, e),
            data: e
        }, j = c.if_eq || b.if_eq, k = j ? j.call(b, b.bookmark_count, l) : q.call(b, "if_eq", b.bookmark_count, l), (k || 0 === k) && (m += k), m += " this week</span>\n  </div>\n\n</div>\n\n<!-- Delicious_728x90_ATF -->\n<div id='div-gpt-ad-1416871411588-0' style='width:728px; height:90px;'>\n<script type='text/javascript'>\ngoogletag.cmd.push(function() { googletag.display('div-gpt-ad-1416871411588-0'); });\n</script>\n</div>"
    }), this.JST["links/tagbundles"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e = "";
            return e += "\n  ", d = c.each.call(a, a.items, {
                hash: {},
                inverse: n.noop,
                fn: n.programWithDepth(2, g, b, a),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n  "
        }
        function g(a, b, d) {
            var e, f = "";
            return f += '\n    <li>\n      <a href="#" data-bundle="', (e = c.name) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.name, e = typeof e === l ? e.apply(a) : e), f += m(e) + '">', (e = c.name) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.name, e = typeof e === l ? e.apply(a) : e), f += m(e) + "</a>\n      ", e = c["if"].call(a, d.editable, {
                hash: {},
                inverse: n.noop,
                fn: n.program(3, h, b),
                data: b
            }), (e || 0 === e) && (f += e), f += "\n    </li>\n  "
        }
        function h(a, b) {
            var d, e = "";
            return e += '\n      <a href="#edit" class="edit" data-bundle="', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === l ? d.apply(a) : d), e += m(d) + '"><i class="d-icon-pencil"></i></a>\n      '
        }
        function i() {
            return "\n  No tag bundles.\n"
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var j, k = "", l = "function", m = this.escapeExpression, n = this;
        return j = c["if"].call(b, b.items, {
            hash: {},
            inverse: n.program(5, i, e),
            fn: n.program(1, f, e),
            data: e
        }), (j || 0 === j) && (k += j), k += "\n"
    }), this.JST["links/tagcloud"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e = "";
            return e += '\n  \n  <ul class="tagcloud-core"></ul>\n  <div>\n    ', d = c["if"].call(a, a.has_more, {
                hash: {},
                inverse: m.noop,
                fn: m.program(2, g, b),
                data: b
            }), (d || 0 === d) && (e += d), e += '\n    <label for="filterTagsBy" class="tag-filter-label">Filter tags by</label> <input id="filterTagsBy" class="tag-filter" type="text" value="', (d = c.filter) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.filter, d = typeof d === k ? d.apply(a) : d), e += l(d) + '">\n  </div>\n  '
        }
        function g(a, b) {
            var d, e = "";
            return e += '\n      <a href="#" class="text-btn show-all-tags" data-toggle="tooltip" title="WARNING: This may slow down your browser">Show all ', (d = c.length) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.length, d = typeof d === k ? d.apply(a) : d), e += l(d) + " tags</a>\n    "
        }
        function h() {
            return "\n  No tags.\n"
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var i, j = "", k = "function", l = this.escapeExpression, m = this;
        return i = c["if"].call(b, b.length, {
            hash: {},
            inverse: m.program(4, h, e),
            fn: m.program(1, f, e),
            data: e
        }), (i || 0 === i) && (j += i), j += "\n"
    }), this.JST["links/user"] = a.template(function(a, b, c, d, e) {
        function f() {
            var a = "";
            return a += "\n        \n      "
        }
        function g(a, b) {
            var d, e = "";
            return e += "\n        ", d = c["if"].call(a, a.am_following_them, {
                hash: {},
                inverse: E.program(6, i, b),
                fn: E.program(4, h, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n      "
        }
        function h(a, b) {
            var d, e = "";
            return e += '\n          <a href="#unfollow" class="follow-toggle follow-toggle-following" data-user="', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === C ? d.apply(a) : d), e += D(d) + '">\n            <span class="following-text">Following</span>\n            <span class="unfollow-text">Unfollow</span>\n          </a>\n        '
        }
        function i(a, b) {
            var d, e = "";
            return e += '\n          <a href="#follow" class="follow-toggle follow-toggle-not-following" data-user="', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === C ? d.apply(a) : d), e += D(d) + '">\n            <span class="follow-text">Follow</span>\n          </a>\n        '
        }
        function j(a, b) {
            var d, e = "";
            return e += "\n      ", (d = c.profile_bio) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.profile_bio, d = typeof d === C ? d.apply(a) : d), e += D(d) + "\n    "
        }
        function k(a, b) {
            var d, e, f = "";
            return f += "\n      ", e = {
                hash: {},
                inverse: E.noop,
                fn: E.program(11, l, b),
                data: b
            }, (d = c.ifMe) ? d = d.call(a, e) : (d = a.ifMe, d = typeof d === C ? d.apply(a) : d), c.ifMe || (d = F.call(a, d, e)), (d || 0 === d) && (f += d), f += "\n    "
        }
        function l() {
            return '\n        Tell the Delicious community about yourself on your <a href="/settings">profile settings page</a>.\n      '
        }
        function m(a, b) {
            var d, e, f = "";
            return f += '\n      <a href="', (d = c.profile_url) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.profile_url, d = typeof d === C ? d.apply(a) : d), f += D(d) + '" target="_blank">', e = {
                hash: {},
                data: b
            }, f += D((d = c.profileUrlBeautify || a.profileUrlBeautify, d ? d.call(a, a.profile_url, e) : G.call(a, "profileUrlBeautify", a.profile_url, e))) + "</a>\n    "
        }
        function n(a, b) {
            var d, e, f, g = "";
            return g += "\n          ", f = {
                hash: {
                    compare: 1
                },
                inverse: E.program(18, p, b),
                fn: E.program(16, o, b),
                data: b
            }, d = c.if_eq || a.if_eq, e = d ? d.call(a, a.bookmark_count, f) : G.call(a, "if_eq", a.bookmark_count, f), (e || 0 === e) && (g += e), g += "\n        "
        }
        function o() {
            return "\n            link\n          "
        }
        function p() {
            return "\n            links\n          "
        }
        function q(a, b) {
            var d, e, f, g = "";
            return g += "\n          ", f = {
                hash: {
                    compare: 1
                },
                inverse: E.program(23, s, b),
                fn: E.program(21, r, b),
                data: b
            }, d = c.if_eq || a.if_eq, e = d ? d.call(a, a.public_bookmark_count, f) : G.call(a, "if_eq", a.public_bookmark_count, f), (e || 0 === e) && (g += e), g += "\n        "
        }
        function r() {
            return "\n            public link\n          "
        }
        function s() {
            return "\n            public links\n          "
        }
        function t() {
            return "\n        follower\n      "
        }
        function u() {
            return "\n        followers\n      "
        }
        function v(a, b) {
            var d, e, f, g = "";
            return g += '\n      <a href="/', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === C ? d.apply(a) : d), g += D(d) + '/subscriptions">\n      <span>\n        <b>', f = {
                hash: {},
                data: b
            }, g += D((d = c.or || a.or, d ? d.call(a, a.subscription_count, 0, f) : G.call(a, "or", a.subscription_count, 0, f))) + "</b>\n        ", f = {
                hash: {
                    compare: 1
                },
                inverse: E.program(32, x, b),
                fn: E.program(30, w, b),
                data: b
            }, d = c.if_eq || a.if_eq, e = d ? d.call(a, a.subscription_count, f) : G.call(a, "if_eq", a.subscription_count, f), (e || 0 === e) && (g += e), g += "\n      </span>\n      </a>\n    "
        }
        function w() {
            return "\n          subscription\n          "
        }
        function x() {
            return "\n          subscriptions\n        "
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var y, z, A, B = "", C = "function", D = this.escapeExpression, E = this, F = c.blockHelperMissing, G = c.helperMissing;
        return B += '<img src="', (y = c.avatar_url) ? y = y.call(b, {
            hash: {},
            data: e
        }) : (y = b.avatar_url, y = typeof y === C ? y.apply(b) : y), B += D(y) + '" class="avatar-img">\n<div class="info">\n  <div class="name-overflow-wrapper">\n    <h2 class="name username-wrapper pull-left">\n      <span class="username">', A = {
            hash: {},
            data: e
        }, B += D((y = c.or || b.or, y ? y.call(b, b.full_name, b.display_name, A) : G.call(b, "or", b.full_name, b.display_name, A))) + '</span>\n      <span class="handle">@', (z = c.username) ? z = z.call(b, {
            hash: {},
            data: e
        }) : (z = b.username, z = typeof z === C ? z.apply(b) : z), B += D(z) + '</span>\n    </h2>\n\n    <div class="pull-right">\n      ', A = {
            hash: {},
            inverse: E.program(3, g, e),
            fn: E.program(1, f, e),
            data: e
        }, (z = c.ifMe) ? z = z.call(b, A) : (z = b.ifMe, z = typeof z === C ? z.apply(b) : z), c.ifMe || (z = F.call(b, z, A)), (z || 0 === z) && (B += z), B += '\n    </div>\n  </div>\n\n  <div class="bio">\n    ', z = c["if"].call(b, b.profile_bio, {
            hash: {},
            inverse: E.program(10, k, e),
            fn: E.program(8, j, e),
            data: e
        }), (z || 0 === z) && (B += z), B += '\n  </div>\n\n\n  <div class="site">\n    ', z = c["if"].call(b, b.profile_url, {
            hash: {},
            inverse: E.noop,
            fn: E.program(13, m, e),
            data: e
        }), (z || 0 === z) && (B += z), B += '\n  </div>\n\n\n  <div class="user-stats">\n    <a href="/', (z = c.username) ? z = z.call(b, {
            hash: {},
            data: e
        }) : (z = b.username, z = typeof z === C ? z.apply(b) : z), B += D(z) + '">\n      <span>\n        <b>', A = {
            hash: {},
            data: e
        }, B += D((y = c.or || b.or, y ? y.call(b, b.bookmark_count, b.public_bookmark_count, A) : G.call(b, "or", b.bookmark_count, b.public_bookmark_count, A))) + "</b>\n        ", A = {
            hash: {},
            inverse: E.program(20, q, e),
            fn: E.program(15, n, e),
            data: e
        }, (z = c.ifMe) ? z = z.call(b, A) : (z = b.ifMe, z = typeof z === C ? z.apply(b) : z), c.ifMe || (z = F.call(b, z, A)), (z || 0 === z) && (B += z), B += '\n      </span>\n    </a>\n\n    <a href="/', (z = c.username) ? z = z.call(b, {
            hash: {},
            data: e
        }) : (z = b.username, z = typeof z === C ? z.apply(b) : z), B += D(z) + '/following">\n    <span>\n      <b>', A = {
            hash: {},
            data: e
        }, B += D((y = c.or || b.or, y ? y.call(b, b.following_count, 0, A) : G.call(b, "or", b.following_count, 0, A))) + '</b>\n      following\n    </span>\n    </a>\n\n    <a href="/', (z = c.username) ? z = z.call(b, {
            hash: {},
            data: e
        }) : (z = b.username, z = typeof z === C ? z.apply(b) : z), B += D(z) + '/followers">\n    <span>\n      <b>', A = {
            hash: {},
            data: e
        }, B += D((y = c.or || b.or, y ? y.call(b, b.follower_count, 0, A) : G.call(b, "or", b.follower_count, 0, A))) + "</b>\n      ", A = {
            hash: {
                compare: 1
            },
            inverse: E.program(27, u, e),
            fn: E.program(25, t, e),
            data: e
        }, y = c.if_eq || b.if_eq, z = y ? y.call(b, b.follower_count, A) : G.call(b, "if_eq", b.follower_count, A), (z || 0 === z) && (B += z), B += "\n    </span>\n    </a>\n\n    ", A = {
            hash: {},
            inverse: E.noop,
            fn: E.program(29, v, e),
            data: e
        }, (z = c.ifMe) ? z = z.call(b, A) : (z = b.ifMe, z = typeof z === C ? z.apply(b) : z), c.ifMe || (z = F.call(b, z, A)), (z || 0 === z) && (B += z), B += '\n\n  </div>\n\n  <!-- <div class="joined mobile-hide">Joined on ', A = {
            hash: {},
            data: e
        }, B += D((y = c.formatDateString || b.formatDateString, y ? y.call(b, b.time_created, A) : G.call(b, "formatDateString", b.time_created, A))) + "</div> -->\n</div>\n\n\n<!-- Delicious_728x90_ATF -->\n<div id='div-gpt-ad-1416871411588-0' style='width:728px; height:90px;'>\n<script type='text/javascript'>\ngoogletag.cmd.push(function() { googletag.display('div-gpt-ad-1416871411588-0'); });\n</script>\n</div>"
    }), this.JST["links/welcome"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e = "";
            return e += "\n      ", d = c["if"].call(a, a.firefox, {
                hash: {},
                inverse: v.program(4, h, b),
                fn: v.program(2, g, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n      "
        }
        function g() {
            return '\n        $(".install-tools").load("/html/tools.html #firefox-box")\n        '
        }
        function h(a, b) {
            var d, e = "";
            return e += "\n        ", d = c["if"].call(a, a.chrome, {
                hash: {},
                inverse: v.program(7, j, b),
                fn: v.program(5, i, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n      "
        }
        function i() {
            return '\n          $(".install-tools").load("/html/tools.html #chrome-box")\n          '
        }
        function j() {
            return '\n          $(".install-tools").load("/html/tools.html #bookmarklet-box")\n        '
        }
        function k(a, b) {
            var d, e = "";
            return e += "\n      ", d = c["if"].call(a, a.ipad, {
                hash: {},
                inverse: v.program(12, m, b),
                fn: v.program(10, l, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n    "
        }
        function l() {
            return '\n        $(".install-tools").load("/html/apps.html #ipad-box")\n        '
        }
        function m(a, b) {
            var d, e = "";
            return e += "\n        ", d = c["if"].call(a, a.iphone, {
                hash: {},
                inverse: v.program(15, o, b),
                fn: v.program(13, n, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n      "
        }
        function n() {
            return '\n          $(".install-tools").load("/html/apps.html #iphone-box")\n          '
        }
        function o(a, b) {
            var d, e = "";
            return e += "\n          ", d = c["if"].call(a, a.android, {
                hash: {},
                inverse: v.program(18, q, b),
                fn: v.program(16, p, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n        "
        }
        function p() {
            return '\n            $(".install-tools").load("/html/apps.html #android-box")\n            '
        }
        function q(a, b) {
            var d, e = "";
            return e += "\n            ", d = c["if"].call(a, a.firefoxos, {
                hash: {},
                inverse: v.program(21, s, b),
                fn: v.program(19, r, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n          "
        }
        function r() {
            return '\n              $(".install-tools").load("/html/apps.html #firefoxos-box")\n              '
        }
        function s() {
            return '\n              $(".install-tools").html("<h2>Oops! No tool found.</h2><p>This is typically because you are using a newly updated device, or one we don’t yet have supported for. Stay tuned, the Delicious team is working to bring support for most common devices. You can still <a href=\'#link\'>add links here</a>.</p>")\n            '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var t, u = "", v = this;
        return u += '<div class="item">\n  <p>Welcome to Delicious! It looks like you haven’t saved any links yet. Delicious has tools to help you bookmark from anywhere on any device. Based on your current device, we recommend you install:</p>\n  <p class="install-tools"></p>\n  <script>\n    ', t = c["if"].call(b, b.desktop, {
            hash: {},
            inverse: v.program(9, k, e),
            fn: v.program(1, f, e),
            data: e
        }), (t || 0 === t) && (u += t), u += '\n  </script>\n  <p>Find more <a href="/tools">tools</a> and <a href="/apps">apps</a>, or just start by <a href="#link">adding your first link</a>.</p>\n</div>\n'
    }), this.JST["modals/add_link"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="modal-dialog">\n  <div class="modal-content">\n    <div class="modal-body">\n      <form>\n        <input type="url" name="url" id="url-input" class="form-control" autofocus required spellcheck="false">\n        <label for="url-input">Add a new Link</label>\n      </form>\n    </div>\n    <div class="modal-footer">\n      <div class="pull-left">\n        <a href="/tools" target="_blank">Bookmarklet</a> &middot;\n        <a href="/apps" target="_blank">Apps</a>\n      </div>\n      <div class="pull-right">\n        <button class="text-btn" data-dismiss="modal" aria-hidden="true">Cancel</button>\n        <button class="text-btn text-btn-primary">Add link</button>\n      </div>\n    </div>\n  </div>\n</div>\n'
    }), this.JST["modals/add_tag"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="modal-dialog">\n  <div class="modal-content">\n    <div class="modal-body">\n      Add tags to these links:\n      <div class="tags-input-container"></div>\n    </div>\n    <div class="modal-footer">\n      <div class="pull-right">\n        <button class="text-btn" data-dismiss="modal" aria-hidden="true">Cancel</button>\n        <button class="text-btn text-btn-primary">Add Tags</button>\n      </div>\n    </div>\n  </div>\n</div>\n'
    }), this.JST["modals/app"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e, f, i = "";
            return i += "\n          ", f = {
                hash: {
                    compare: "A"
                },
                inverse: l.program(4, h, b),
                fn: l.program(2, g, b),
                data: b
            }, d = c.if_eq || a.if_eq, e = d ? d.call(a, a.status, f) : m.call(a, "if_eq", a.status, f), (e || 0 === e) && (i += e), i += '\n          <button class="text-btn text-btn-danger delete-app">Delete</button>\n        '
        }
        function g() {
            return '\n          <button class="text-btn text-btn-danger disable-app">Disable</button>\n          '
        }
        function h() {
            return '\n          <button class="text-btn text-btn-primary enable-app">Enable</button>\n          '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var i, j, k = "", l = this, m = c.helperMissing, n = "function", o = this.escapeExpression;
        return k += '<div class="modal-dialog">\n  <div class="modal-content">\n    <div class="modal-header">\n      <h3>Create a Delicious OAuth application</h3>\n    </div>\n    <div class="modal-body">\n      <form>\n        <input type="text" name="app-name" id="app-name" class="form-control" required value="' + o((i = b.details, i = null == i || i===!1 ? i : i.app_name, typeof i === n ? i.apply(b) : i)) + '">\n        <label for="app-name">Name</label>\n\n        <textarea class="form-control" name="app-description" id="app-description" rows="3" maxlength="1000">' + o((i = b.details, i = null == i || i===!1 ? i : i.description, typeof i === n ? i.apply(b) : i)) + '</textarea>\n        <label for="app-description">Description</label>\n\n      </form>\n    </div>\n    <div class="modal-footer">\n      <div class="pull-left">\n        ', j = c["if"].call(b, b.app_key, {
            hash: {},
            inverse: l.noop,
            fn: l.program(1, f, e),
            data: e
        }), (j || 0 === j) && (k += j), k += '\n      </div>\n\n      <div class="pull-right">\n        <button class="text-btn" data-dismiss="modal" aria-hidden="true">Cancel</button>\n        <button class="text-btn text-btn-primary">Save</button>\n      </div>\n    </div>\n  </div>\n</div>\n'
    }), this.JST["modals/compose_link"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="modal-dialog" id="compose-modal">\n  <div class="modal-content">\n  </div>\n</div>\n'
    }), this.JST["modals/create"] = a.template(function(a, b, c, d, e) {
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var f, g = "", h = "function", i = this.escapeExpression;
        return g += '<div class="modal-dialog modal-dialog-signup modal-dialog-narrow">\n  <form id="create-account-form">\n    <div class="modal-content">\n      <div class="modal-body">\n\n        <div class="form-group">\n          <input type="text" name="display-name" id="display-name" value="', (f = c.name) ? f = f.call(b, {
            hash: {},
            data: e
        }) : (f = b.name, f = typeof f === h ? f.apply(b) : f), g += i(f) + '" class="form-control" required autofocus maxlength="20">\n          <label for="display-name">Full name</label>\n        </div>\n\n        <div class="form-group">\n          <input type="email" name="email" id="email" required class="form-control" spellcheck="false" value="', (f = c.email) ? f = f.call(b, {
            hash: {},
            data: e
        }) : (f = b.email, f = typeof f === h ? f.apply(b) : f), g += i(f) + '">\n          <label for="email">Email</label>\n        </div>\n\n        <div class="form-group">\n          <input type="text" name="username" id="username" required class="form-control" maxlength="15" spellcheck="false">\n          <label for="username">Username</label>\n        </div>\n\n        <div class="form-group">\n          <input type="password" name="password" id="password" required class="form-control" pattern=".{8,}">\n          <label for="password">Password (8 characters or more, no spaces)</label>\n        </div>\n\n        \n\n        <div class="terms">\n          By clicking the button, you agree to the <a href="/terms" target="_blank">Terms & Conditions</a>.\n        </div>\n        <div class="error text-danger"></div>\n\n      </div>\n      <div class="modal-footer">\n        <span class="help-inline text-danger"></span>\n        <div class="pull-right">\n          <button type="submit" class="text-btn text-btn-primary join-button">Sign Up for Delicious</button>\n        </div>\n      </div>\n    </div>\n  </form>\n</div>\n'
    }), this.JST["modals/forgot_password"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="modal-dialog modal-dialog-forgot-password modal-dialog-narrow">\n  <div class="modal-content">\n    <div class="modal-body">\n      <form id="forgot-password-form" class="forgot-password-form">\n        <fieldset>\n          <input class="form-control" type="text" name="username" id="username" class="username" required spellcheck="false">\n          <label for="username" class="label-username">Username</label>\n\n          <input type="submit" class="hidden">\n        </fieldset>\n      </form>\n\n      <div class="send-success hidden">\n        <div>\n          <p><strong>Great! There’s only one more step.</strong> <br>You will receive an email from us shortly. Simply click the link to set a new password.</p>\n        </div>\n      </div>\n    </div>\n    <div class="modal-footer">\n      <span class="help-inline text-danger error-msg"></span>\n      <div class="pull-right">\n        <button class="text-btn text-btn-primary submit-button">Submit</button>\n      </div>\n    </div>\n  </div>\n</div>\n'
    }), this.JST["modals/help"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="modal-dialog">\n  <div class="modal-content">\n    <div class="modal-header hidden">\n        <h2>Shortcuts</h2>\n    </div>\n    <div class="modal-body help-modal-body">\n    </div>\n    <div class="modal-footer">\n      <div class="pull-right">\n        <button class="text-btn" data-dismiss="modal" aria-hidden="true">Close</button>\n      </div>\n    </div>\n\n  </div>\n</div>\n<script>\n  $(".help-modal-body").load("/html/shortcuts.html #shortcut-lists")\n</script>\n'
    }), this.JST["modals/login"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="modal-dialog modal-dialog-signin modal-dialog-narrow">\n  <div class="modal-content">\n    <!-- <div class="modal-header">\n      <h3>Sign in</h3>\n    </div> -->\n    <div class="modal-body">\n      <div class="thirdparty-login-wrapper">\n        <button class="text-btn text-btn-primary thirdparty-login thirdparty-login-twitter" data-service="twitter" title="Sign in with Twitter">\n          <i class="d-icon-twitter"></i> <span>Twitter</span>\n        </button>\n\n        <button class="text-btn text-btn-primary thirdparty-login thirdparty-login-facebook" data-service="facebook" title="Sign in with Facebook">\n          <i class="d-icon-facebook"></i> <span>Facebook</span>\n        </button>\n\n        <button class="text-btn text-btn-primary thirdparty-login thirdparty-login-google" data-service="google" title="Sign in with Google+">\n          <i class="d-icon-google"></i> <span>Google</span>\n        </button>\n      </div>\n\n      <hr>\n\n      <form id="login-form" class="login-form">\n        <fieldset>\n          <input class="form-control" type="text" name="username" id="username" class="username" autofocus spellcheck="false">\n          <label for="username" class="label-username">Username</label>\n\n          <input class="form-control" type="password" name="password" id="password" class="password">\n          <label for="password" class="label-password">Password <a href="#forget-password" class="pull-right">Forgot password?</a></label>\n\n          <input type="submit" style="position: absolute; visibility: hidden; width: 0; height: 0;">\n        </fieldset>\n      </form>\n    </div>\n    <div class="modal-footer">\n      <span class="help-inline text-danger"></span>\n      <div class="pull-left">\n        <a href="#signup">Don&rsquo;t have an account?</a>\n      </div>\n      <div class="pull-right">\n        <button class="text-btn text-btn-primary login-button">Sign in</button>\n      </div>\n      <span class="help-inline text-danger"></span>\n    </div>\n  </div>\n</div>\n'
    }), this.JST["modals/popup_login"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="modal-dialog">\n  <div class="modal-content">\n    <div class="modal-body">\n      <a href="/" target="_blank">Sign up or sign in to Delicious to continue.</a>\n    </div>\n  </div>\n</div>\n'
    }), this.JST["modals/remove_tag"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e = "";
            return e += '\n        <ul class="tags remove-tag-list">\n        ', d = c.each.call(a, a, {
                hash: {},
                inverse: n.noop,
                fn: n.program(2, g, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n        </ul>\n        "
        }
        function g(a) {
            var b = "";
            return b += '\n          <li><a href="#">' + m(typeof a === l ? a.apply(a) : a) + "</a></li>\n        "
        }
        function h() {
            return "\n        <p>There&rsquo;s no common tag in your selection.</p>\n      "
        }
        function i() {
            return '\n          <button class="text-btn text-btn-primary">Remove Selected Tags</button>\n        '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var j, k = "", l = "function", m = this.escapeExpression, n = this;
        return k += '<div class="modal-dialog">\n  <div class="modal-content">\n    <div class="modal-body">\n      ', j = c["if"].call(b, b, {
            hash: {},
            inverse: n.program(4, h, e),
            fn: n.program(1, f, e),
            data: e
        }), (j || 0 === j) && (k += j), k += '\n    </div>\n    <div class="modal-footer">\n      <div class="pull-right">\n        <button class="text-btn" data-dismiss="modal" aria-hidden="true">Cancel</button>\n        ', j = c["if"].call(b, b, {
            hash: {},
            inverse: n.noop,
            fn: n.program(6, i, e),
            data: e
        }), (j || 0 === j) && (k += j), k += "\n      </div>\n    </div>\n  </div>\n</div>\n"
    }), this.JST["modals/reset_password"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="modal-dialog modal-dialog-reset-password modal-dialog-narrow">\n  <div class="modal-content">\n    <div class="modal-body">\n      <form id="reset-password-form" class="reset-password-form">\n        <fieldset>\n          <input class="form-control" type="password" name="password" id="password" class="password" required>\n          <label for="username" class="label-password">Set your new password (8 characters or more, no spaces)</label>\n\n          <input type="submit" class="hidden">\n        </fieldset>\n      </form>\n\n      <div class="reset-success hidden">\n        <div>\n          <p>Your password was successfuly reset. You may now <a href="#login">login with your new password</a>.</p>\n        </div>\n      </div>\n    </div>\n    <div class="modal-footer">\n      <span class="help-inline text-danger error-msg"></span>\n      <div class="pull-right">\n        <button class="text-btn text-btn-primary submit-button">Submit</button>\n      </div>\n    </div>\n  </div>\n</div>\n'
    }), this.JST["modals/signup"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="modal-dialog modal-dialog-signup modal-dialog-narrow">\n  <div class="modal-content">\n    <div class="modal-body">\n      <div>\n        <div>Sign up with your&hellip;</div>\n        <button class="text-btn text-btn-primary thirdparty-login thirdparty-login-twitter" data-service="twitter">\n          <i class="d-icon-twitter"></i> Twitter\n        </button>\n\n        <button class="text-btn text-btn-primary thirdparty-login thirdparty-login-facebook" data-service="facebook">\n          <i class="d-icon-facebook"></i> Facebook\n        </button>\n\n        <button class="text-btn text-btn-primary thirdparty-login thirdparty-login-google" data-service="google">\n          <i class="d-icon-google"></i> Google\n        </button>\n      </div>\n    </div>\n    <div class="modal-footer">\n      <div class="pull-left">\n        <a href="#" class="create-account"><b>Sign up with your email address</b></a>\n        <br>Already have a Delicious account? <a href="#" class="switch-signin">Sign in</a>\n      </div>\n    </div>\n  </div>\n</div>\n'
    }), this.JST["modals/tag_editor"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e = "";
            return e += "\n            <option>", (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === i ? d.apply(a) : d), e += j(d) + "</option>\n          "
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var g, h = "", i = "function", j = this.escapeExpression, k = this;
        return h += '<div class="modal-dialog">\n  <div class="modal-content">\n    <div class="modal-body">\n      <div class="tag-editor">\n        Rename\n\n        <select class="form-control inline">\n          ', g = c.each.call(b, b, {
            hash: {},
            inverse: k.noop,
            fn: k.program(1, f, e),
            data: e
        }), (g || 0 === g) && (h += g), h += '\n        </select>\n\n        <br>to\n\n        <input class="form-control new-tag-name contenteditable inline" spellcheck="false">\n\n        <span class="text-success"></span>\n      </div>\n    </div>\n    <div class="modal-footer">\n      <div class="pull-left">\n        <button class="text-btn text-btn-danger delete-tag">Delete</button>\n      </div>\n      <div class="pull-right">\n        <button class="text-btn" data-dismiss="modal" aria-hidden="true">Cancel</button>\n        <button class="text-btn text-btn-primary rename-tag">Submit</button>\n      </div>\n    </div>\n  </div>\n</div>\n'
    }), this.JST["modals/tag_search"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="modal-dialog">\n  <div class="modal-content">\n    <div class="modal-body">\n      <form>\n        <input class="form-control" type="text" name="tag-query" id="tag-query" class="tag-query" autofocus spellcheck="false">\n        <label for="tag-query" class="label-tag-query">Search popular tags (at least 3 characters)</label>\n      </form>\n    </div>\n    <div class="tag-search-result people"></div>\n    <div class="modal-footer">\n      <div class="pull-right">\n        <button class="text-btn" data-dismiss="modal" aria-hidden="true">Done</button>\n      </div>\n    </div>\n  </div>\n</div>\n'
    }), this.JST["modals/tagbundle"] = a.template(function(a, b, c, d, e) {
        function f() {
            return '\n          <button class="text-btn text-btn-danger">Delete</button>\n        '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var g, h = "", i = "function", j = this.escapeExpression, k = this;
        return h += '<div class="modal-dialog modal-dialog-tagbundle">\n  <div class="modal-content">\n    <div class="modal-body">\n      <form>\n        <fieldset>\n          <input class="form-control" type="text" name="name" value="', (g = c.name) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.name, g = typeof g === i ? g.apply(b) : g), h += j(g) + '" id="bundle-name">\n          <label for="bundle-name" class="lable-bundle-name">Bundle Name</label>\n\n          <div class="tags-input-container"></div>\n          <input type="hidden" name="tags" value="', (g = c.tags) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.tags, g = typeof g === i ? g.apply(b) : g), h += j(g) + '" id="bundle-tags">\n          <label for="bundle-tags" class="lable-bundle-tags">Tags</label>\n        </fieldset>\n      </form>\n      <span class="text-danger"></span>\n    </div>\n    <div class="modal-footer">\n      <div class="pull-left">\n        ', g = c["if"].call(b, b.name, {
            hash: {},
            inverse: k.noop,
            fn: k.program(1, f, e),
            data: e
        }), (g || 0 === g) && (h += g), h += '\n      </div>\n      <div class="pull-right">\n        <button class="text-btn" data-dismiss="modal" aria-hidden="true">Cancel</button>\n        <button class="text-btn text-btn-primary">Save Bundle</button>\n      </div>\n    </div>\n  </div>\n</div>\n'
    }), this.JST["network/link"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e = "";
            return e += '\n      <div class="note">', (d = c.note) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.note, d = typeof d === r ? d.apply(a) : d), e += s(d) + "</div>\n    "
        }
        function g(a, b) {
            var d, e, f, g = "";
            return g += "<span ", d = c["if"].call(a, a.save_rank, {
                hash: {},
                inverse: t.noop,
                fn: t.program(4, h, b),
                data: b
            }), (d || 0 === d) && (g += d), g += ' title="', (d = c.owner) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.owner, d = typeof d === r ? d.apply(a) : d), g += s(d) + " was the ", f = {
                hash: {},
                data: b
            }, g += s((d = c.ordinal || a.ordinal, d ? d.call(a, a.save_rank, f) : u.call(a, "ordinal", a.save_rank, f))) + ' user to save to link">', (e = c.save_rank) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.save_rank, e = typeof e === r ? e.apply(a) : e), g += s(e) + "</span>"
        }
        function h() {
            return 'class="has-personal"'
        }
        function i() {
            return "people"
        }
        function j() {
            return "person"
        }
        function k(a, b) {
            var d, e = "";
            return e += '\n        <ul class="tags"> ', d = c.each.call(a, a.tags, {
                hash: {},
                inverse: t.noop,
                fn: t.program(11, l, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "</ul>\n      "
        }
        function l(a) {
            var b = "";
            return b += '<li><a href="/tag/' + s(typeof a === r ? a.apply(a) : a) + '">' + s(typeof a === r ? a.apply(a) : a) + "</a></li> "
        }
        function m() {
            return '\n          <a href="#link-tags" class="mobile-tags-toggle">Tags</a>\n        '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var n, o, p, q = "", r = "function", s = this.escapeExpression, t = this, u = c.helperMissing;
        return q += '<div class="item item-rich">\n  <div class="item-rich-wrapper">\n    <div class="user">\n      <a href="/' + s((n = b.saver, n = null == n || n===!1 ? n : n.username, typeof n === r ? n.apply(b) : n)) + '"><img src="' + s((n = b.saver, n = null == n || n===!1 ? n : n.avatar_url, typeof n === r ? n.apply(b) : n)) + '" class="avatar-img"> <b>' + s((n = b.saver, n = null == n || n===!1 ? n : n.full_name, typeof n === r ? n.apply(b) : n)) + "</b></a>\n    </div>\n\n    ", o = c["if"].call(b, b.note, {
            hash: {},
            inverse: t.noop,
            fn: t.program(1, f, e),
            data: e
        }), (o || 0 === o) && (q += o), q += '\n\n    <div class="title-wrapper">\n      <a href="', (o = c.url) ? o = o.call(b, {
            hash: {},
            data: e
        }) : (o = b.url, o = typeof o === r ? o.apply(b) : o), q += s(o) + '" class="title">', p = {
            hash: {},
            data: e
        }, q += s((n = c.or || b.or, n ? n.call(b, b.title, b.url, p) : u.call(b, "or", b.title, b.url, p))) + '</a>\n      <span class="domain">', (o = c.domain) ? o = o.call(b, {
            hash: {},
            data: e
        }) : (o = b.domain, o = typeof o === r ? o.apply(b) : o), q += s(o) + '</span>\n    </div>\n\n    <div class="meta">\n      <a href="/link/', (o = c.md5) ? o = o.call(b, {
            hash: {},
            data: e
        }) : (o = b.md5, o = typeof o === r ? o.apply(b) : o), q += s(o) + '" class="link-rank">\n        ', o = c["if"].call(b, b.save_rank, {
            hash: {},
            inverse: t.noop,
            fn: t.program(3, g, e),
            data: e
        }), (o || 0 === o) && (q += o), q += '<span title="', (o = c.num_saves) ? o = o.call(b, {
            hash: {},
            data: e
        }) : (o = b.num_saves, o = typeof o === r ? o.apply(b) : o), q += s(o) + " ", p = {
            hash: {
                compare: "1"
            },
            inverse: t.program(8, j, e),
            fn: t.program(6, i, e),
            data: e
        }, n = c.if_gt || b.if_gt, o = n ? n.call(b, b.num_saves, p) : u.call(b, "if_gt", b.num_saves, p), (o || 0 === o) && (q += o), q += ' saved this link">', (o = c.num_saves) ? o = o.call(b, {
            hash: {},
            data: e
        }) : (o = b.num_saves, o = typeof o === r ? o.apply(b) : o), q += s(o) + "</span>\n      </a>\n\n      ", o = c["if"].call(b, b.tags, {
            hash: {},
            inverse: t.noop,
            fn: t.program(10, k, e),
            data: e
        }), (o || 0 === o) && (q += o), q += '\n    </div>\n\n    <div class="meta meta-secondary">\n      <div class="actions">\n        ', o = c["if"].call(b, b.tags, {
            hash: {},
            inverse: t.noop,
            fn: t.program(13, m, e),
            data: e
        }), (o || 0 === o) && (q += o), q += '\n        <a href="#" class="action-link bookmark-btn">Add Link</a>\n      </div>\n\n      <time datetime="', p = {
            hash: {},
            data: e
        }, q += s((n = c.formatMachineDate || b.formatMachineDate, n ? n.call(b, b.time_created, p) : u.call(b, "formatMachineDate", b.time_created, p))) + '" class="date" title="', p = {
            hash: {},
            data: e
        }, q += s((n = c.formatDateFull || b.formatDateFull, n ? n.call(b, b.time_created, p) : u.call(b, "formatDateFull", b.time_created, p))) + '">', p = {
            hash: {},
            data: e
        }, q += s((n = c.formatDateRelative || b.formatDateRelative, n ? n.call(b, b.time_created, p) : u.call(b, "formatDateRelative", b.time_created, p))) + "</time>\n    </div>\n  </div>\n</div>\n<!-- .item -->\n"
    }), this.JST["network/list"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, "<!-- Delicious_728x90_ATF -->\n<div id='div-gpt-ad-1416871411588-0' style='width:728px; height:90px;'>\n<script type='text/javascript'>\ngoogletag.cmd.push(function() { googletag.display('div-gpt-ad-1416871411588-0'); });\n</script>\n</div>\n\n\n\n<div class=\"link-list\"></div>\n"
    }), this.JST["network/page"] = a.template(function(a, b, c, d, e) {
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var f, g = "", h = "function", i = this.escapeExpression;
        return g += '<div class="hero-unit">\n  <h2>My Network</h2>\n  <a href="/settings/friends">Follow your friends</a> or find people you&rsquo;re interested to build an awesome feed.\n</div>\n\n<div class="links"></div>\n', (f = c.loadMore) ? f = f.call(b, {
            hash: {},
            data: e
        }) : (f = b.loadMore, f = typeof f === h ? f.apply(b) : f), g += i(f) + '\n\n\n<script>\njQuery.fn.contentChange = function(callback){\n    var elms = jQuery(this);\n    elms.each(\n      function(i){\n        var elm = jQuery(this);\n        elm.data("lastContents", elm.html());\n        window.watchContentChange = window.watchContentChange ? window.watchContentChange : [];\n        window.watchContentChange.push({"element": elm, "callback": callback});\n      }\n    )\n    return elms;\n  }\n  setInterval(function(){\n    if(window.watchContentChange){\n      for( i in window.watchContentChange){\n        if(window.watchContentChange[i].element.data("lastContents") != window.watchContentChange[i].element.html()){\n          window.watchContentChange[i].callback.apply(window.watchContentChange[i].element);\n          window.watchContentChange[i].element.data("lastContents", window.watchContentChange[i].element.html())\n        };\n      }\n    }\n  },500);\n\n\n    $(\'.links\').contentChange(function(){ \n     initSkimLinks();\n     \n    });\n    \n</script>\n'
    }), this.JST.not_found = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div class="not-found">\n  <div class="hero-unit">\n    <div class="info">\n      <h2>404 Not Found</h2>\n      <div>\n        The page you are looking for could not be found.\n        <a href="/">back to home</a>\n      </div>\n    </div>\n  </div>\n\n  <div class="item">\n    <p>We&rsquo;re sorry, we can&rsquo;t find the link you&rsquo;ve entered! This is usually because you clicked a broken link or mistyped the URL. If you think this is a mistake on our part, try refreshing the page or <a href="mailto:support@delicious.com">send us some feedback</a>.</p>\n  </div>\n</div>\n'
    }), this.JST.notification = a.template(function(a, b, c, d, e) {
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var f, g = "", h = "function", i = this.escapeExpression;
        return g += '<div class="alert" data-notif-name="', (f = c.name) ? f = f.call(b, {
            hash: {},
            data: e
        }) : (f = b.name, f = typeof f === h ? f.apply(b) : f), g += i(f) + '">\n  <button type="button" class="close" data-dismiss="alert">&times;</button>\n  ', (f = c.message) ? f = f.call(b, {
            hash: {},
            data: e
        }) : (f = b.message, f = typeof f === h ? f.apply(b) : f), (f || 0 === f) && (g += f), g += "\n</div>\n"
    }), this.JST.sidebar = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e, f = "";
            return f += 'title="', e = {
                hash: {},
                data: b
            }, f += q((d = c.or || a.or, d ? d.call(a, a.full_name, a.display_name, e) : p.call(a, "or", a.full_name, a.display_name, e))) + '"'
        }
        function g(a, b) {
            var d, e, f = "";
            return f += "\n        ", e = {
                hash: {},
                data: b
            }, f += q((d = c.or || a.or, d ? d.call(a, a.full_name, a.display_name, e) : p.call(a, "or", a.full_name, a.display_name, e))) + "\n      "
        }
        function h() {
            return "\n        Delicious\n      "
        }
        function i() {
            return '\n      <a href="/logout" class="site-nav-sign-out pull-right">Sign out</a>\n    '
        }
        function j() {
            return '\n      <li class="nav-item-home"><a href="/"><span class=d-icon-home></span> My Links</a></li>\n      '
        }
        function k() {
            return '\n      <li class="nav-item-home"><a href="#login"><span class=d-icon-signin></span> Sign in</a></li>\n    '
        }
        function l() {
            return '\n      <li class="nav-item-settings"><a href="/settings"><span class=d-icon-settings></span> Settings</a></li>\n    '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var m, n, o = "", p = c.helperMissing, q = this.escapeExpression, r = this, s = "function", t = c.blockHelperMissing;
        return o += '<aside class="site-nav" tabindex="1">\n  <!--[if lt IE 9]>\n    <p class="browsehappy">You are using an <strong>outdated</strong> browser. Please <a href="http://browsehappy.com/">upgrade your browser</a> to improve your experience.</p>\n  <![endif]-->\n\n\n  <div class="site-nav-profile">\n    <a href="/" class="site-nav-user-link" ', m = c["if"].call(b, b.username, {
            hash: {},
            inverse: r.noop,
            fn: r.program(1, f, e),
            data: e
        }), (m || 0 === m) && (o += m), o += '>\n      <div class="logo site-nav-logo">\n        <i class="s1"></i>\n        <i class="s2"></i>\n        <i class="s3"></i>\n        <i class="s4"></i>\n      </div>\n      ', m = c["if"].call(b, b.username, {
            hash: {},
            inverse: r.program(5, h, e),
            fn: r.program(3, g, e),
            data: e
        }), (m || 0 === m) && (o += m), o += "\n    </a>\n    ", m = c["if"].call(b, b.username, {
            hash: {},
            inverse: r.noop,
            fn: r.program(7, i, e),
            data: e
        }), (m || 0 === m) && (o += m), o += '\n  </div>\n  <!-- .site-nav-profile -->\n\n  <div class="site-nav-search">\n    <form id="location-bar" data-icon="&#xe008;">\n      <input type="text" name="site-search" value="" id="site-search" placeholder="Search&hellip;" spellcheck="false">\n      <ul class="list-group search-bar-tip">\n        <li class="list-group-item help">Search Tips</li>\n        <li class="list-group-item help"><b>#tagname</b><br>Search a specific tag</li>\n        <li class="list-group-item help"><b>@username</b><br>Search a specific user</li>\n        <li class="list-group-item help"><b>keywords</b><br>Search specific keywords</li>\n        <li class="list-group-item help"><b>@username #tagname</b><br>Search tags for a specific user</li>\n        <li class="list-group-item help"><b>@username keywords</b><br>Search keywords for a specific user</li>\n        <li class="list-group-item help"><b>http://</b><br>Search a specific web page on Delicious</li>\n      </ul>\n    </form>\n  </div>\n  <!-- .site-nav-search -->\n\n  <ul class="site-nav-links">\n    ', n = {
            hash: {},
            inverse: r.program(11, k, e),
            fn: r.program(9, j, e),
            data: e
        }, (m = c.ifLoggedIn) ? m = m.call(b, n) : (m = b.ifLoggedIn, m = typeof m === s ? m.apply(b) : m), c.ifLoggedIn || (m = t.call(b, m, n)), (m || 0 === m) && (o += m), o += '\n\n    <li class="nav-item-network">\n      <a href="/network" class="login-check"><span class=d-icon-network></span> Network</a>\n    </li>\n    <li class="nav-item-discover">\n      <a href="/discover" class="login-check"><span class=d-icon-discover></span> Discover</a>\n    </li>\n    <li class="nav-item-trending">\n      <a href="/trending"><span class=d-icon-trending></span> Trending <span class="beta-tag">(beta)</span></a>\n    </li>\n    <li class="nav-item-add"><a href="#link"><span class=d-icon-add></span> Add Link</a></li>\n    ', n = {
            hash: {},
            inverse: r.noop,
            fn: r.program(13, l, e),
            data: e
        }, (m = c.ifLoggedIn) ? m = m.call(b, n) : (m = b.ifLoggedIn, m = typeof m === s ? m.apply(b) : m), c.ifLoggedIn || (m = t.call(b, m, n)), (m || 0 === m) && (o += m), o += '\n  </ul>\n  <!-- .site-nav-links -->\n\n  <div class="site-nav-footer">\n    <ul class="meta-links lowercase">\n      <li><a href="/help">Help</a></li>\n      <li><a href="/apps">Apps</a></li>\n      <li><a href="/tools">Tools</a></li>\n      <li><a href="http://blog.delicious.com">Blog</a></li>\n      \n    </ul>\n     <ul class="meta-links lowercase">\n      <li><a href="/privacy">Privacy Policy</a></li>\n      <li><a href="/terms">Terms</a></li>\n\n    </ul>\n    &copy; Delicious Science, LLC\n  </div>\n  <!-- .site-nav-footer -->\n\n  <div class="site-nav-toggle">\n    <a href="#">\n      <span></span>\n      <span></span>\n      <span></span>\n    </a>\n  </div>\n  <!-- .site-nav-toggle -->\n</aside>\n<!-- .site-nav -->\n\n<script>\n$(document).ready(function() {});\n\n</script>\n\n'
    }), this.JST["user/relations/user_relationship"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e, f, h = "";
            return h += "\n        subscription", f = {
                hash: {
                    compare: 1
                },
                inverse: u.noop,
                fn: u.program(2, g, b),
                data: b
            }, d = c.if_gt || a.if_gt, e = d ? d.call(a, a.subscription_count, f) : v.call(a, "if_gt", a.subscription_count, f), (e || 0 === e) && (h += e), h += "\n          "
        }
        function g() {
            return "s"
        }
        function h(a, b) {
            var d, e = "";
            return e += "\n          ", d = c["if"].call(a, a.is_following, {
                hash: {},
                inverse: u.program(7, j, b),
                fn: u.program(5, i, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n      "
        }
        function i() {
            return "following"
        }
        function j(a, b) {
            var d, e, f, h = "";
            return h += "follower", f = {
                hash: {
                    compare: 1
                },
                inverse: u.noop,
                fn: u.program(2, g, b),
                data: b
            }, d = c.if_gt || a.if_gt, e = d ? d.call(a, a.follower_count, f) : v.call(a, "if_gt", a.follower_count, f), (e || 0 === e) && (h += e), h
        }
        function k(a, b) {
            var d, e, f, h = "";
            return h += "\n        <b>", f = {
                hash: {},
                data: b
            }, h += w((d = c.or || a.or, d ? d.call(a, a.subscription_count, 0, f) : v.call(a, "or", a.subscription_count, 0, f))) + "</b> subscription", f = {
                hash: {
                    compare: 1
                },
                inverse: u.noop,
                fn: u.program(2, g, b),
                data: b
            }, d = c.if_gt || a.if_gt, e = d ? d.call(a, a.subscription_count, f) : v.call(a, "if_gt", a.subscription_count, f), (e || 0 === e) && (h += e), h += "\n        "
        }
        function l(a, b) {
            var d, e = "";
            return e += "\n        ", d = c["if"].call(a, a.is_following, {
                hash: {},
                inverse: u.program(14, n, b),
                fn: u.program(12, m, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n      "
        }
        function m(a, b) {
            var d, e, f = "";
            return f += "\n          <b>", e = {
                hash: {},
                data: b
            }, f += w((d = c.or || a.or, d ? d.call(a, a.following_count, 0, e) : v.call(a, "or", a.following_count, 0, e))) + "</b> following\n          "
        }
        function n(a, b) {
            var d, e, f, h = "";
            return h += "\n          <b>", f = {
                hash: {},
                data: b
            }, h += w((d = c.or || a.or, d ? d.call(a, a.follower_count, 0, f) : v.call(a, "or", a.follower_count, 0, f))) + "</b> follower", f = {
                hash: {
                    compare: 1
                },
                inverse: u.noop,
                fn: u.program(2, g, b),
                data: b
            }, d = c.if_gt || a.if_gt, e = d ? d.call(a, a.follower_count, f) : v.call(a, "if_gt", a.follower_count, f), (e || 0 === e) && (h += e), h += "\n        "
        }
        function o(a, b) {
            var d, e = "";
            return e += '\n      <div class="pull-right mobile-no-float">\n      ', d = c.unless.call(a, a.is_subscription, {
                hash: {},
                inverse: u.program(19, q, b),
                fn: u.program(17, p, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n      </div>\n    "
        }
        function p() {
            return '\n        <a href="/settings/friends">Find more friends on Delicious</a>\n        '
        }
        function q() {
            return '\n        <a href="#search-tag">Add a new subscription</a>\n      '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var r, s, t = "", u = this, v = c.helperMissing, w = this.escapeExpression, x = "function", y = c.blockHelperMissing;
        return t += '<div class="hero-unit">\n  <div class="info">\n    <h2 class="name">\n      <span>', (r = c.full_name) ? r = r.call(b, {
            hash: {},
            data: e
        }) : (r = b.full_name, r = typeof r === x ? r.apply(b) : r), t += w(r) + "</span>&rsquo;s\n      ", r = c["if"].call(b, b.is_subscription, {
            hash: {},
            inverse: u.program(4, h, e),
            fn: u.program(1, f, e),
            data: e
        }), (r || 0 === r) && (t += r), t += '\n    </h2>\n\n    <div class="pull-left mobile-no-float">\n      ', r = c["if"].call(b, b.is_subscription, {
            hash: {},
            inverse: u.program(11, l, e),
            fn: u.program(9, k, e),
            data: e
        }), (r || 0 === r) && (t += r), t += '\n      -\n      <a href="/', (r = c.username) ? r = r.call(b, {
            hash: {},
            data: e
        }) : (r = b.username, r = typeof r === x ? r.apply(b) : r), t += w(r) + '">back to @', (r = c.username) ? r = r.call(b, {
            hash: {},
            data: e
        }) : (r = b.username, r = typeof r === x ? r.apply(b) : r), t += w(r) + "</a>\n    </div>\n\n    ", s = {
            hash: {},
            inverse: u.noop,
            fn: u.program(16, o, e),
            data: e
        }, (r = c.ifMe) ? r = r.call(b, s) : (r = b.ifMe, r = typeof r === x ? r.apply(b) : r), c.ifMe || (r = y.call(b, r, s)), (r || 0 === r) && (t += r), t += '\n  </div>\n</div>\n\n\n<div class="people">\n</div>\n'
    }), this.JST["user/relations/user_relationships"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e, f, j = "";
            return j += '\n  <div class="item user-item">\n    <div class="user-item-wrapper cf', d = c["if"].call(a, a.profile_bio, {
                hash: {},
                inverse: r.noop,
                fn: r.program(2, g, b),
                data: b
            }), (d || 0 === d) && (j += d), j += '">\n      ', d = c["if"].call(a, a.is_me, {
                hash: {},
                inverse: r.program(6, i, b),
                fn: r.program(4, h, b),
                data: b
            }), (d || 0 === d) && (j += d), j += '\n\n      <a href="/', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === p ? d.apply(a) : d), j += q(d) + '" class="name">\n        <img src="', (d = c.avatar_url) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.avatar_url, d = typeof d === p ? d.apply(a) : d), j += q(d) + '" class="avatar-img">\n        <b>', (d = c.full_name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.full_name, d = typeof d === p ? d.apply(a) : d), j += q(d) + '</b>\n        <span class="link-count">\n          <b>', f = {
                hash: {},
                data: b
            }, j += q((d = c.or || a.or, d ? d.call(a, a.public_bookmark_count, 0, f) : s.call(a, "or", a.public_bookmark_count, 0, f))) + "</b> public link", f = {
                hash: {
                    compare: 1
                },
                inverse: r.noop,
                fn: r.program(11, l, b),
                data: b
            }, d = c.if_gt || a.if_gt, e = d ? d.call(a, a.public_bookmark_count, f) : s.call(a, "if_gt", a.public_bookmark_count, f), (e || 0 === e) && (j += e), j += "\n        </span>\n      </a>\n\n      ", e = c["if"].call(a, a.profile_bio, {
                hash: {},
                inverse: r.noop,
                fn: r.program(13, m, b),
                data: b
            }), (e || 0 === e) && (j += e), j += "\n    </div>\n  </div>\n"
        }
        function g() {
            return " has-bio"
        }
        function h() {
            return '\n        <a href="/settings" class="follow-toggle follow-toggle-me pull-right">\n          <span>Edit your profile</span>\n        </a>\n        '
        }
        function i(a, b) {
            var d, e = "";
            return e += "\n        ", d = c["if"].call(a, a.am_following_them, {
                hash: {},
                inverse: r.program(9, k, b),
                fn: r.program(7, j, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n      "
        }
        function j(a, b) {
            var d, e = "";
            return e += '\n          <a href="#unfollow" class="follow-toggle follow-toggle-following pull-right" data-user="', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === p ? d.apply(a) : d), e += q(d) + '">\n            <span class="following-text">Following</span>\n            <span class="unfollow-text">Unfollow</span>\n          </a>\n          '
        }
        function k(a, b) {
            var d, e = "";
            return e += '\n          <a href="#follow" class="follow-toggle follow-toggle-not-following pull-right" data-user="', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === p ? d.apply(a) : d), e += q(d) + '">\n            <span class="follow-text">Follow</span>\n          </a>\n        '
        }
        function l() {
            return "s"
        }
        function m(a, b) {
            var d, e = "";
            return e += '\n        <p class="bio">\n          ', (d = c.profile_bio) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.profile_bio, d = typeof d === p ? d.apply(a) : d), e += q(d) + "\n        </p>\n      "
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var n, o = "", p = "function", q = this.escapeExpression, r = this, s = c.helperMissing;
        return n = c.each.call(b, b, {
            hash: {},
            inverse: r.noop,
            fn: r.program(1, f, e),
            data: e
        }), (n || 0 === n) && (o += n), o += "\n"
    }), this.JST["user/relations/user_subscriptions"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e, f, j = "";
            return j += '\n  <div class="item cf">\n    <div class="subscription-item pull-left">\n      <a href="/tag/', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === l ? d.apply(a) : d), j += m(d) + '" class="name">', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === l ? d.apply(a) : d), j += m(d) + '</a>\n        <span class="link-count">\n          <b>', f = {
                hash: {},
                data: b
            }, j += m((d = c.or || a.or, d ? d.call(a, a.last_week_count, 0, f) : n.call(a, "or", a.last_week_count, 0, f))) + "</b> link", f = {
                hash: {
                    compare: 1
                },
                inverse: o.noop,
                fn: o.program(2, g, b),
                data: b
            }, d = c.if_gt || a.if_gt, e = d ? d.call(a, a.last_week_count, f) : n.call(a, "if_gt", a.last_week_count, f), (e || 0 === e) && (j += e), j += "\n        </span>\n    </div>\n\n    ", e = c["if"].call(a, a.subscribed, {
                hash: {},
                inverse: o.program(6, i, b),
                fn: o.program(4, h, b),
                data: b
            }), (e || 0 === e) && (j += e), j += "\n  </div>\n"
        }
        function g() {
            return "s"
        }
        function h(a, b) {
            var d, e = "";
            return e += '\n      <a href="#unfollow" class="follow-toggle follow-toggle-following pull-right" data-tag="', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === l ? d.apply(a) : d), e += m(d) + '">\n        <span class="following-text">Subscribed</span>\n        <span class="unfollow-text">Unsubscribe</span>\n      </a>\n      '
        }
        function i(a, b) {
            var d, e = "";
            return e += '\n      <a href="#follow" class="follow-toggle follow-toggle-not-following pull-right" data-tag="', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === l ? d.apply(a) : d), e += m(d) + '">\n        <span class="follow-text">Subscribe</span>\n      </a>\n    '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var j, k = "", l = "function", m = this.escapeExpression, n = c.helperMissing, o = this;
        return j = c.each.call(b, b, {
            hash: {},
            inverse: o.noop,
            fn: o.program(1, f, e),
            data: e
        }), (j || 0 === j) && (k += j), k += "\n"
    }), this.JST["user/settings"] = a.template(function(a, b, c, d, e) {
        function f() {
            return "has-avatar"
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var g, h = "", i = this, j = "function", k = this.escapeExpression;
        return h += '<div class="hero-unit">\n  <h2>Settings</h2>\n  Your account settings &amp; personal preferences\n</div>\n\n<div class="link-nav">\n  <ul class="actions">\n    <li class="active"><a href="#profile" data-toggle="tab">Profile</a></li>\n    <li><a href="#account" data-toggle="tab">Account</a></li>\n    <li><a href="#password" data-toggle="tab">Password</a></li>\n    <li><a href="#thirdparty" data-toggle="tab">Social</a></li>\n    <li><a href="#manage" data-toggle="tab">Manage</a></li>\n    <li><a href="#apps" data-toggle="tab">Apps</a></li>\n    <li><a href="#developer" data-toggle="tab">Developer</a></li>\n  </ul>\n</div>\n\n<div class="settings-wrapper">\n  <div class="tab-content">\n    <div class="tab-pane active" id="profile">\n\n      <div class="avatar-wrapper ', g = c.unless.call(b, b.is_default_avatar, {
            hash: {},
            inverse: i.noop,
            fn: i.program(1, f, e),
            data: e
        }), (g || 0 === g) && (h += g), h += '">\n        <img src="', (g = c.avatar_url) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.avatar_url, g = typeof g === j ? g.apply(b) : g), h += k(g) + '" class="avatar-img avatar_display">\n        <form id="photo-form" class="avatar-form">\n          <input type="file" class="hidden" id="avatar_file" name="photo" accept="image/*" />\n          <button class="text-btn avatar-edit" type="button">Change</button>\n          <button class="text-btn avatar-reset" type="button">Reset</button>\n        </form>\n      </div>\n      <h3 class="subtitle">Avatar <small>(Maxium File Size: 2 MB)</small>\n      </h3>\n\n      <form id="profile_form">\n        <fieldset>\n          <legend>Public Profile</legend>\n\n          <div class="form-group">\n            <input type="text" id="display_name" value="', (g = c.full_name) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.full_name, g = typeof g === j ? g.apply(b) : g), h += k(g) + '" class="form-control">\n            <label for="display_name">Name</label>\n          </div>\n\n          <div class="form-group">\n            <textarea id="bio" rows="1" class="form-control" maxlength="140">', (g = c.profile_bio) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.profile_bio, g = typeof g === j ? g.apply(b) : g), h += k(g) + '</textarea>\n            <label for="bio">Bio <span id="bio_chars_remaining" class="pull-right">140</span></label>\n          </div>\n\n          <div class="form-group">\n            <input type="url" id="url" value="', (g = c.profile_url) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.profile_url, g = typeof g === j ? g.apply(b) : g), h += k(g) + '" class="form-control" spellcheck="false">\n            <label for="url">Website</label>\n          </div>\n\n          <div class="form-group">\n            <button class="text-btn text-btn-primary" id="profile_submit">Save</button>\n          </div>\n        </fieldset>\n      </form>\n    </div>\n\n    <div class="tab-pane" id="account">\n      <!-- <h3>Account</h3> -->\n\n      <form id="email_form">\n        <fieldset>\n          <legend>Change your email</legend>\n          <div class="form-group">\n            <input type="email" id="profile_email" value="', (g = c.profile_email) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.profile_email, g = typeof g === j ? g.apply(b) : g), h += k(g) + '" class="form-control" required spellcheck="false">\n            <label for="profile_email">Email</label>\n          </div>\n\n          <div class="form-group">\n            <button class="text-btn text-btn-primary" id="email_submit">Save</button>\n          </div>\n        </fieldset>\n      </form>\n\n      <hr>\n      <button type="button" class="text-btn text-btn-danger" data-toggle="modal" data-target="#deactivate-account">Deactivate Account</button>\n\n      <div class="modal fade danger-zone-wrapper" id="deactivate-account" tabindex="-1" role="dialog" aria-labelledby="myModalLabel" aria-hidden="true">\n        <div class="modal-dialog">\n          <div class="modal-content">\n            <div class="modal-body">\n              <div class="danger-zone">\n                <h3>Deactivate Account</h3>\n\n                <p>Looking for a different flavor? We are always updating the menu, and would love to get your <!--<a href="mailto:feedback@avos.com?subject=Delicious Next feedback from @', (g = c.username) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.username, g = typeof g === j ? g.apply(b) : g), h += k(g) + '">-->feedback  <!--</a>-->. Before you go, please keep in mind the following:</p>\n\n                <ul>\n                  <li>Your account should be removed from delicious within a few minutes, but some content may be viewable on delicious for a few days after deactivation.</li>\n                  <li>We have no control over content indexed by search engines such as Google, Bing, and Blekko.</li>\n                  <li>You are welcome back anytime!</li>\n                </ul>\n\n                <form id="deactivate_form">\n                  <div class="form-group">\n                    <input type="password" id="deactivate_account_password" class="form-control" required>\n                    <label for="deactivate_account_password">Type your password to confirm</label>\n                    <span class="help-block" id="deactivate_error"></span>\n                  </div>\n\n                  <button type="button" class="text-btn text-btn-default" data-dismiss="modal">Cancel</button>\n                  <button type="submit" class="text-btn text-btn-danger" id="deactivate_submit">Yes, deactivate my account</button>\n                </form>\n              </div>\n            </div>\n          </div><!-- /.modal-content -->\n        </div><!-- /.modal-dialog -->\n      </div><!-- /.modal -->\n\n    </div>\n\n    <div class="tab-pane" id="password">\n      <form id="password_form">\n        <fieldset>\n          <legend>Change your password</legend>\n          <div class="form-group">\n            <input type="password" id="original_password" class="form-control" required>\n            <label for="original_password">Current Password</label>\n          </div>\n\n          <div class="form-group">\n            <input type="password" id="new_password" class="form-control"  minlength="8">\n            <label for="new_password">New Password</label>\n          </div>\n\n          <div class="form-group">\n            <input type="password" id="new_password_confirm" class="form-control" minlength="8">\n            <label for="new_password_confirm">Confirm New Password</label>\n          </div>\n\n          <div class="form-group">\n            <button class="text-btn text-btn-primary" id="password_submit">Save</button>\n            <span class="text-danger" id="password_error"></span>\n          </div>\n        </fieldset>\n      </form>\n\n    </div>\n\n    <div class="tab-pane" id="thirdparty">\n      <div class="thirdparty-items"></div>\n    </div>\n\n    <div class="tab-pane" id="manage">\n      <p>Import or export your bookmarks from a file</p>\n      <ul>\n        <li><a href="http://export.delicious.com/settings/bookmarks/import" target="_blank">Import Bookmarks</a></li>\n        <li><a href="http://export.delicious.com/settings/bookmarks/export" target="_blank">Export Bookmarks</a></li>\n      </ul>\n    </div>\n\n    <div class="tab-pane" id="apps">\n      <!-- TODO: copywriting -->\n      <p>List of authorized third-party apps. These apps can access your Delicious account.</p>\n      <div class="apps-container"></div>\n    </div>\n\n    <div class="tab-pane" id="developer">\n      <div class="devapps-container"></div>\n    </div>\n  </div>\n\n  <div class="settings-footer">\n    <hr>\n    <a href="mailto:feedback@delicious.com?subject=Delicious Next feedback from @', (g = c.username) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.username, g = typeof g === j ? g.apply(b) : g), h += k(g) + '">We&rsquo;d love to hear your <b>feedback</b> on the new design!</a>\n    <hr>\n\n    <ul class="meta-links">\n      <li><a href="/help">Help</a></li>\n      <li><a href="/apps">Apps</a></li>\n      <li><a href="/tools">Tools</a></li>\n      <li><a href="http://blog.delicious.com/">Blog</a></li>\n      <li><a href="/logout">Sign out</a></li>\n    </ul>\n    &copy; Delicious Science, LLC\n  </div>\n</div>\n\n\n', (g = c.loadMore) ? g = g.call(b, {
            hash: {},
            data: e
        }) : (g = b.loadMore, g = typeof g === j ? g.apply(b) : g), h += k(g) + "\n"
    }), this.JST["user/settings/app"] = a.template(function(a, b, c, d, e) {
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var f, g = "", h = "function", i = this.escapeExpression;
        return g += '<div class="app">\n  <span class="app-name pull-left">' + i((f = b.details, f = null == f || f===!1 ? f : f.app_name, typeof f === h ? f.apply(b) : f)) + '</span>\n\n  <a class="follow-toggle follow-toggle-following pull-right" href="#revoke">\n    Revoke\n  </a>\n</div>\n\n<hr>\n'
    }), this.JST["user/settings/apps"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<hr>\n\n<div class="app-list"></div>\n'
    }), this.JST["user/settings/devapp"] = a.template(function(a, b, c, d, e) {
        function f() {
            return '\n      <span class="label label-danger">Disabled</span>\n    '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var g, h, i, j = "", k = "function", l = this.escapeExpression, m = this, n = c.helperMissing;
        return j += '<div class="app">\n  <a class="pull-right text-btn text-btn-default" href="#edit">\n    Edit\n  </a>\n\n  <p class="app-name">' + l((g = b.details, g = null == g || g===!1 ? g : g.app_name, typeof g === k ? g.apply(b) : g)) + "\n    ", i = {
            hash: {
                compare: "D"
            },
            inverse: m.noop,
            fn: m.program(1, f, e),
            data: e
        }, g = c.if_eq || b.if_eq, h = g ? g.call(b, b.status, i) : n.call(b, "if_eq", b.status, i), (h || 0 === h) && (j += h), j += '\n  </p>\n\n  <ul class="app-info">\n    <li><p><span class="app-info-key">Description:</span> <span class="app-info-value">' + l((g = b.details, g = null == g || g===!1 ? g : g.description, typeof g === k ? g.apply(b) : g)) + '</span></p></li>\n    <li><p><span class="app-info-key">Client ID:</span> <code class="app-info-value">', (h = c.app_key) ? h = h.call(b, {
            hash: {},
            data: e
        }) : (h = b.app_key, h = typeof h === k ? h.apply(b) : h), j += l(h) + '</code></p></li>\n    <li><p><span class="app-info-key">Client Secret:</span> <code class="app-info-value">', (h = c.app_secret) ? h = h.call(b, {
            hash: {},
            data: e
        }) : (h = b.app_secret, h = typeof h === k ? h.apply(b) : h), j += l(h) + "</code></p></li>\n  </ul>\n\n</div>\n\n<hr>"
    }), this.JST["user/settings/devapps"] = a.template(function(a, b, c, d, e) {
        return this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {}, '<div>\n  <button class="text-btn text-btn-primary create-devapp">Create Your Own App</button>\n  <p><a href="https://github.com/avos/delicious-api/blob/master/api/oauth.md">Learn more</a> about our OAuth 2.0 authentication.</p>\n</div>\n\n<hr>\n\n<div class="app-list"></div>\n'
    }), this.JST["user/settings/friends"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e, f, j = "";
            return j += '\n  <div class="item user-item">\n    <div class="user-item-wrapper cf', d = c["if"].call(a, a.profile_bio, {
                hash: {},
                inverse: s.noop,
                fn: s.program(2, g, b),
                data: b
            }), (d || 0 === d) && (j += d), j += '">\n      ', d = c["if"].call(a, a.is_me, {
                hash: {},
                inverse: s.program(6, i, b),
                fn: s.program(4, h, b),
                data: b
            }), (d || 0 === d) && (j += d), j += '\n\n      <a href="/', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === q ? d.apply(a) : d), j += r(d) + '" class="name">\n        <img src="', (d = c.avatar_url) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.avatar_url, d = typeof d === q ? d.apply(a) : d), j += r(d) + '" class="avatar-img">\n        <b>', (d = c.full_name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.full_name, d = typeof d === q ? d.apply(a) : d), j += r(d) + '</b>\n        <span class="link-count">\n          <b>', f = {
                hash: {},
                data: b
            }, j += r((d = c.or || a.or, d ? d.call(a, a.public_bookmark_count, 0, f) : t.call(a, "or", a.public_bookmark_count, 0, f))) + "</b> public link", f = {
                hash: {
                    compare: 1
                },
                inverse: s.program(13, m, b),
                fn: s.program(11, l, b),
                data: b
            }, d = c.if_eq || a.if_eq, e = d ? d.call(a, a.public_bookmark_count, f) : t.call(a, "if_eq", a.public_bookmark_count, f), (e || 0 === e) && (j += e), j += "\n        </span>\n      </a>\n\n      ", e = c["if"].call(a, a.profile_bio, {
                hash: {},
                inverse: s.noop,
                fn: s.program(15, n, b),
                data: b
            }), (e || 0 === e) && (j += e), j += "\n    </div>\n  </div>\n"
        }
        function g() {
            return " has-bio"
        }
        function h() {
            return '\n        <a href="/settings" class="follow-toggle follow-toggle-me pull-right">\n          <span>Edit your profile</span>\n        </a>\n        '
        }
        function i(a, b) {
            var d, e = "";
            return e += "\n        ", d = c["if"].call(a, a.am_following_them, {
                hash: {},
                inverse: s.program(9, k, b),
                fn: s.program(7, j, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n      "
        }
        function j(a, b) {
            var d, e = "";
            return e += '\n          <a href="#unfollow" class="follow-toggle follow-toggle-following pull-right" data-user="', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === q ? d.apply(a) : d), e += r(d) + '">\n            <span class="following-text">Following</span>\n            <span class="unfollow-text">Unfollow</span>\n          </a>\n          '
        }
        function k(a, b) {
            var d, e = "";
            return e += '\n          <a href="#follow" class="follow-toggle follow-toggle-not-following pull-right" data-user="', (d = c.username) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.username, d = typeof d === q ? d.apply(a) : d), e += r(d) + '">\n            <span class="follow-text">Follow</span>\n          </a>\n        '
        }
        function l() {
            var a = "";
            return a
        }
        function m() {
            return "s"
        }
        function n(a, b) {
            var d, e = "";
            return e += '\n        <p class="bio">\n          ', (d = c.profile_bio) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.profile_bio, d = typeof d === q ? d.apply(a) : d), e += r(d) + "\n        </p>\n      "
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var o, p = "", q = "function", r = this.escapeExpression, s = this, t = c.helperMissing;
        return p += '<div class="hero-unit">\n  <div class="info">\n    <h2 class="name">\n      Your friends on Delicious\n    </h2>\n    Find more friends by <a href="/settings/thirdparty">connecting social networks</a> to Delicious.\n  </div>\n</div>\n\n<div class="people">\n\n', o = c.each.call(b, b.items, {
            hash: {},
            inverse: s.noop,
            fn: s.program(1, f, e),
            data: e
        }), (o || 0 === o) && (p += o), p += "\n\n</div>\n"
    }), this.JST["user/settings/thirdparty"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e = "";
            return e += "\n    ", d = c["with"].call(a, a.twitter, {
                hash: {},
                inverse: x.noop,
                fn: x.program(2, g, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n  "
        }
        function g(a, b) {
            var d, e = "";
            return e += '\n      <button class="text-btn thirdparty-login thirdparty-login-twitter disconnect-button" data-service="twitter">\n        <i class="d-icon-twitter"></i> Connected: ', (d = c.providerDisplayname) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.providerDisplayname, d = typeof d === u ? d.apply(a) : d), e += v(d) + '\n        <small>Click to disconnect</small>\n      </button>\n\n      <form class="thirdparty-options" id="twitter-options" data-service="twitter">\n        <label for="twitter-import-option" class="checkbox">\n          <input id="twitter-import-option" type="checkbox" class="import-control"\n          ', d = c["if"].call(a, a.active, {
                hash: {},
                inverse: x.noop,
                fn: x.program(3, h, b),
                data: b
            }), (d || 0 === d) && (e += d), e += '>\n            Import links I posted on Twitter,\n        </label>\n        <span>tag them with</span>\n\n        <span class="form-control contenteditable inline thirdparty-tag-control" id="twitter-tag-options" contenteditable>\n          ', d = c["if"].call(a, a.tags, {
                hash: {},
                inverse: x.program(7, j, b),
                fn: x.program(5, i, b),
                data: b
            }), (d || 0 === d) && (e += d), e += '\n        </span>\n\n        <span>and</span>\n\n        <select id="twitter-privacy-options" class="form-control inline thirdparty-privacy-control">\n          ', d = c.each.call(a, a.privacy_options, {
                hash: {},
                inverse: x.noop,
                fn: x.programWithDepth(9, k, b, a),
                data: b
            }), (d || 0 === d) && (e += d), e += '\n        </select>\n        <label for="twitter-privacy-options" class="hidden">Default privacy level</label>\n        <span>.</span>\n\n        <button type="submit" class="text-btn text-btn-primary">Save</button>\n      </form>\n    '
        }
        function h() {
            return "checked"
        }
        function i(a, b) {
            var d, e = "";
            return e += "\n            ", (d = c.tags) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.tags, d = typeof d === u ? d.apply(a) : d), e += v(d) + "\n            "
        }
        function j() {
            return "\n            !fromtwitter\n          "
        }
        function k(a, b, d) {
            var e, f, g, h = "";
            return h += '\n            <option value="', (e = c.value) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.value, e = typeof e === u ? e.apply(a) : e), h += v(e) + '" ', g = {
                hash: {},
                data: b
            }, h += v((e = c.privacy_option_check || d.privacy_option_check, e ? e.call(a, d.privateAttri, g) : w.call(a, "privacy_option_check", d.privateAttri, g))) + ">", (f = c.text) ? f = f.call(a, {
                hash: {},
                data: b
            }) : (f = a.text, f = typeof f === u ? f.apply(a) : f), h += v(f) + "</option>\n          "
        }
        function l() {
            return '\n    <button class="text-btn thirdparty-login thirdparty-login-twitter connect-button" data-service="twitter">\n      <i class="d-icon-twitter"></i> Connect to Twitter\n    </button>\n  '
        }
        function m(a, b) {
            var d, e = "";
            return e += "\n    ", d = c["with"].call(a, a.facebook, {
                hash: {},
                inverse: x.noop,
                fn: x.program(14, n, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n    "
        }
        function n(a, b) {
            var d, e = "";
            return e += '\n      <button class="text-btn thirdparty-login thirdparty-login-facebook disconnect-button" data-service="facebook">\n        <i class="d-icon-facebook"></i> Connected: ', (d = c.providerDisplayname) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.providerDisplayname, d = typeof d === u ? d.apply(a) : d), e += v(d) + "\n        <small>Click to disconnect</small>\n      </button>\n    "
        }
        function o() {
            return '\n    <button class="text-btn thirdparty-login thirdparty-login-facebook connect-button" data-service="facebook">\n      <i class="d-icon-facebook"></i> Connect to Facebook\n    </button>\n  '
        }
        function p(a, b) {
            var d, e = "";
            return e += "\n    ", d = c["with"].call(a, a.google, {
                hash: {},
                inverse: x.noop,
                fn: x.program(19, q, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n    "
        }
        function q(a, b) {
            var d, e = "";
            return e += '\n      <button class="text-btn thirdparty-login thirdparty-login-google disconnect-button" data-service="google">\n        <i class="d-icon-google"></i> Connected: ', (d = c.providerUserId) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.providerUserId, d = typeof d === u ? d.apply(a) : d), e += v(d) + "\n        <small>Click to disconnect</small>\n      </button>\n    "
        }
        function r() {
            return '\n    <button class="text-btn thirdparty-login thirdparty-login-google connect-button" data-service="google">\n      <i class="d-icon-google"></i> Connect to Google\n    </button>\n  '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var s, t = "", u = "function", v = this.escapeExpression, w = c.helperMissing, x = this;
        return t += '<div class="thirdparty-item">\n  ', s = c["if"].call(b, b.twitter, {
            hash: {},
            inverse: x.program(11, l, e),
            fn: x.program(1, f, e),
            data: e
        }), (s || 0 === s) && (t += s), t += '\n</div>\n\n<hr>\n\n<div class="thirdparty-item">\n  ', s = c["if"].call(b, b.facebook, {
            hash: {},
            inverse: x.program(16, o, e),
            fn: x.program(13, m, e),
            data: e
        }), (s || 0 === s) && (t += s), t += '\n</div>\n\n<hr>\n\n<div class="thirdparty-item">\n  ', s = c["if"].call(b, b.google, {
            hash: {},
            inverse: x.program(21, r, e),
            fn: x.program(18, p, e),
            data: e
        }), (s || 0 === s) && (t += s), t += "\n</div>\n"
    }), this.JST["widgets/compose_form"] = a.template(function(a, b, c, d, e) {
        function f() {
            return '\n  <div class="item item-welcome">\n    <div class="logo-with-heading">\n      <div class="logo">\n        <i class="s1"></i>\n        <i class="s2"></i>\n        <i class="s3"></i>\n        <i class="s4"></i>\n      </div>\n      <a href="http://delicious.com/" class="logo-text">Delicious</a>\n      <a href="#compose-login" class="pull-right"><small>Already registered?</small></a>\n    </div>\n\n    <p><b>Delicious</b> is the leading bookmarking service to save, organize, and discover interesting links on the web.\n    <a href="http://delicious.com/" target="_top">Learn more</a>.</p>\n    <p>Start to build your personal knowledge base by saving the first link:</p>\n  </div>\n'
        }
        function g(a, b) {
            var d, e, f = "";
            return f += '\n  <div class="item item-status">\n    You saved this link on ', e = {
                hash: {},
                data: b
            }, f += A((d = c.formatDateString || a.formatDateString, d ? d.call(a, (d = a.recommended, null == d || d===!1 ? d : d.previously_saved_time), e) : z.call(a, "formatDateString", (d = a.recommended, null == d || d===!1 ? d : d.previously_saved_time), e))) + ".\n  </div>\n"
        }
        function h(a, b) {
            var d, e = "";
            return e += '\n        <input type="url" id="inputLinkUrl" name="url" value="', (d = c.url) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.url, d = typeof d === B ? d.apply(a) : d), e += A(d) + '" class="form-control" spellcheck="false">\n        <label for="inputLinkUrl">URL</label>\n        '
        }
        function i(a, b) {
            var d, e = "";
            return e += '\n        <input type="hidden" name="url" value="', (d = c.url) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.url, d = typeof d === B ? d.apply(a) : d), e += A(d) + '">\n      '
        }
        function j(a, b) {
            var d, e, f = "";
            return f += '\n        <div class="tag-suggestion">\n          <div class="tag-suggestion-items">\n            ', e = c.each.call(a, (d = a.recommended, null == d || d===!1 ? d : d.suggested_tags), {
                hash: {},
                inverse: C.noop,
                fn: C.program(10, k, b),
                data: b
            }), (e || 0 === e) && (f += e), f += '\n          </div>\n          <span class="subtitle">Suggested Tags\n            <i class="inline-help"\n              data-html="true"\n              data-content="Suggested tags are generated based on your tag cloud and the Delicious community, click the tag above to select."\n              data-placement="bottom"\n              data-trigger="hover"\n              data-container=".inline-help-pop">\n              ?\n            </i>\n          </span>\n        </div>\n      '
        }
        function k(a) {
            var b = "";
            return b += '\n              <span class="tag-suggestion-item" data-tag="' + A(typeof a === B ? a.apply(a) : a) + '">' + A(typeof a === B ? a.apply(a) : a) + "</span>\n            "
        }
        function l() {
            return "on"
        }
        function m() {
            return "private"
        }
        function n() {
            return "publicly visible"
        }
        function o() {
            return "checked"
        }
        function p() {
            return '\n              <i class="d-icon-locked"></i> <span>Private</span>\n              '
        }
        function q() {
            return '\n              <i class="d-icon-locked d-icon-unlocked"></i> <span>Public</span>\n            '
        }
        function r(a, b) {
            var d, e = "";
            return e += "\n        ", d = c.unless.call(a, a.time_created, {
                hash: {},
                inverse: C.noop,
                fn: C.program(25, s, b),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n          "
        }
        function s(a, b) {
            var d, e = "";
            return e += '\n          <label class="checkbox inline checkbox-action checkbox-social checkbox-social-twitter ', d = c["if"].call(a, a.twitter_checked, {
                hash: {},
                inverse: C.noop,
                fn: C.program(12, l, b),
                data: b
            }), (d || 0 === d) && (e += d), e += " ", d = c.unless.call(a, a.twitter, {
                hash: {},
                inverse: C.noop,
                fn: C.program(26, t, b),
                data: b
            }), (d || 0 === d) && (e += d), e += '"\n            data-service="twitter"\n            data-toggle="tooltip"\n            title="Share this link to Twitter"\n            >\n            <input type="checkbox" name="share-twitter" ', d = c["if"].call(a, a.twitter_checked, {
                hash: {},
                inverse: C.noop,
                fn: C.program(18, o, b),
                data: b
            }), (d || 0 === d) && (e += d), e += '>\n              <i class="d-icon-twitter"></i> <span class="mobile-hide">Twitter</span>\n          </label>\n          <label class="checkbox inline checkbox-action checkbox-social checkbox-social-facebook ', d = c["if"].call(a, a.facebook_checked, {
                hash: {},
                inverse: C.noop,
                fn: C.program(12, l, b),
                data: b
            }), (d || 0 === d) && (e += d), e += " ", d = c.unless.call(a, a.facebook, {
                hash: {},
                inverse: C.noop,
                fn: C.program(26, t, b),
                data: b
            }), (d || 0 === d) && (e += d), e += '"\n            data-service="facebook"\n            data-toggle="tooltip"\n            title="Share this link to Facebook"\n            >\n            <input type="checkbox" name="share-facebook" ', d = c["if"].call(a, a.facebook_checked, {
                hash: {},
                inverse: C.noop,
                fn: C.program(18, o, b),
                data: b
            }), (d || 0 === d) && (e += d), e += '>\n              <i class="d-icon-facebook"></i> <span class="mobile-hide">Facebook</span>\n          </label>\n          '
        }
        function t() {
            return "unavailable"
        }
        function u() {
            return '\n      <button class="text-btn text-btn-danger">Delete</button>\n    '
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var v, w, x, y = "", z = c.helperMissing, A = this.escapeExpression, B = "function", C = this;
        return v = c["if"].call(b, b.anonymous, {
            hash: {},
            inverse: C.noop,
            fn: C.program(1, f, e),
            data: e
        }), (v || 0 === v) && (y += v), y += "\n", w = c["if"].call(b, (v = b.recommended, null == v || v===!1 ? v : v.previously_saved), {
            hash: {},
            inverse: C.noop,
            fn: C.program(3, g, e),
            data: e
        }), (w || 0 === w) && (y += w), y += '\n<div class="modal-body">\n  <div class="inline-help-pop"></div>\n  <form>\n    <fieldset>\n      ', w = c["if"].call(b, b.url_edit, {
            hash: {},
            inverse: C.program(7, i, e),
            fn: C.program(5, h, e),
            data: e
        }), (w || 0 === w) && (y += w), y += '\n      <input type="hidden" name="replace" value="', (w = c.replace) ? w = w.call(b, {
            hash: {},
            data: e
        }) : (w = b.replace, w = typeof w === B ? w.apply(b) : w), y += A(w) + '">\n\n      <input class="form-control" type="text" id="inputLinkTitle" name="title" value="', x = {
            hash: {},
            data: e
        }, y += A((v = c.or || b.or, v ? v.call(b, b.title, (v = b.recommended, null == v || v===!1 ? v : v.suggested_title), x) : z.call(b, "or", b.title, (v = b.recommended, null == v || v===!1 ? v : v.suggested_title), x))) + '" autofocus>\n      <label for="inputLinkTitle">Title</label>\n\n      <div class="tags-input-container"></div>\n      <label for="tags">Tags\n        <i class="inline-help"\n          data-html="true"\n          data-content="Tags are an important part of organizing on Delicious. Using tags that are smart and simple benifits both you and the community. <ul><li>Lowercase tags are recommended: <code>webdev</code> instead of <code>WebDev</code></li><li>Spaces are allowed but not recommended: <code>machine-learning</code> instead of <code>Machine Learning</code></li><li>Reuse your tags and the tags of the community as much as possible.</li><li>Use ! at the start of a tag if it\'s unrelated to the link content: <code>!unread</code>, <code>!savedforlater</code>, etc.</li></ul>"\n          data-placement="bottom"\n          data-trigger="hover"\n          data-container=".inline-help-pop">\n          ?\n        </i>\n        <span class="pull-right">Separated by comma</span>\n      </label>\n\n      ', w = c["if"].call(b, (v = b.recommended, null == v || v===!1 ? v : v.suggested_tags), {
            hash: {},
            inverse: C.noop,
            fn: C.program(9, j, e),
            data: e
        }), (w || 0 === w) && (y += w), y += '\n\n      <textarea class="form-control autoresize" rows="1" id="textareaLinkComment" name="note" maxlength="1000">', (w = c.note) ? w = w.call(b, {
            hash: {},
            data: e
        }) : (w = b.note, w = typeof w === B ? w.apply(b) : w), y += A(w) + '</textarea>\n      <label for="textareaLinkComment" class="subtitle">Comment <span class="comment_chars_remaining pull-right">1000</span></label>\n\n      <label class="checkbox inline checkbox-action checkbox-privacy ', w = c.unless.call(b, b["private"], {
            hash: {},
            inverse: C.noop,
            fn: C.program(12, l, e),
            data: e
        }), (w || 0 === w) && (y += w), y += '"\n        data-toggle="tooltip"\n        title="This link is ', w = c["if"].call(b, b["private"], {
            hash: {},
            inverse: C.program(16, n, e),
            fn: C.program(14, m, e),
            data: e
        }), (w || 0 === w) && (y += w), y += '"\n          >\n          <input type="checkbox" name="private" ', w = c["if"].call(b, b["private"], {
            hash: {},
            inverse: C.noop,
            fn: C.program(18, o, e),
            data: e
        }), (w || 0 === w) && (y += w), y += ">\n            ", w = c["if"].call(b, b["private"], {
            hash: {},
            inverse: C.program(22, q, e),
            fn: C.program(20, p, e),
            data: e
        }), (w || 0 === w) && (y += w), y += "\n      </label>\n      ", w = c.unless.call(b, b.anonymous, {
            hash: {},
            inverse: C.noop,
            fn: C.program(24, r, e),
            data: e
        }), (w || 0 === w) && (y += w), y += '\n          <input type="submit" style="position: absolute; visibility: hidden; width: 0; height: 0;">\n    </fieldset>\n  </form>\n</div>\n<div class="modal-footer">\n  <div class="pull-left">\n    ', w = c["if"].call(b, (v = b.recommended, null == v || v===!1 ? v : v.previously_saved), {
            hash: {},
            inverse: C.noop,
            fn: C.program(28, u, e),
            data: e
        }), (w || 0 === w) && (y += w), y += '\n  </div>\n  <div class="pull-right">\n    <button class="text-btn" data-dismiss="modal" aria-hidden="true">Cancel</button>\n    <button class="text-btn text-btn-primary">Save link</button>\n  </div>\n</div>\n'
    }), this.JST["widgets/compose_result"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e, f = "";
            return f += '\n        <div class="mobile-rank">', e = {
                hash: {},
                data: b
            }, f += o((d = c.ordinal || a.ordinal, d ? d.call(a, a.save_rank, e) : n.call(a, "ordinal", a.save_rank, e))) + "</div>\n        You are the <b>", e = {
                hash: {},
                data: b
            }, f += o((d = c.ordinal || a.ordinal, d ? d.call(a, a.save_rank, e) : n.call(a, "ordinal", a.save_rank, e))) + "</b> person to save the link.\n        "
        }
        function g() {
            return "\n        Link saved!\n      "
        }
        function h(a, b) {
            var d, e = "";
            return e += "\n    ", d = c["with"].call(a, a.first_saver, {
                hash: {},
                inverse: q.noop,
                fn: q.programWithDepth(6, i, b, a),
                data: b
            }), (d || 0 === d) && (e += d), e += "\n  "
        }
        function i(a, b, d) {
            var e, f, g = "";
            return g += '\n    <a class="first-saver clearfix" href="/', (e = c.username) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.username, e = typeof e === p ? e.apply(a) : e), g += o(e) + '" target="_blank">\n      <img src="', (e = c.avatar_url) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.avatar_url, e = typeof e === p ? e.apply(a) : e), g += o(e) + '">\n      <div class="name">', (e = c.full_name) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.full_name, e = typeof e === p ? e.apply(a) : e), g += o(e) + "</div>\n      <div>First added this link on ", f = {
                hash: {},
                data: b
            }, g += o((e = c.formatDateString || d.formatDateString, e ? e.call(a, d.first_saver_date, f) : n.call(a, "formatDateString", d.first_saver_date, f))) + "</div>\n    </a>\n    "
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var j, k, l, m = "", n = c.helperMissing, o = this.escapeExpression, p = "function", q = this;
        return m += '<div class="compose-result">\n  <div class="clearfix">\n    <a class="msg" href="/" target="_blank">\n      ', j = c["if"].call(b, b.save_rank, {
            hash: {},
            inverse: q.program(3, g, e),
            fn: q.program(1, f, e),
            data: e
        }), (j || 0 === j) && (m += j), m += "\n    </a>\n  </div>\n\n  ", l = {
            hash: {
                compare: 1
            },
            inverse: q.noop,
            fn: q.program(5, h, e),
            data: e
        }, j = c.if_gt || b.if_gt, k = j ? j.call(b, b.save_rank, l) : n.call(b, "if_gt", b.save_rank, l), (k || 0 === k) && (m += k), m += '\n\n  <div class="related-links clearfix">\n    <h3>Related links on Delicious</h3>\n    \n    <ul></ul>\n  </div>\n\n  <div class="discover-more-link clearfix">\n    <a href="/discover" target="_blank">Discover more links on <b>Delicious</b>.</a>\n  </div>\n</div>\n\n<div class="compose-result-backdrop"></div>\n'
    }), this.JST["widgets/compose_result_related_links"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e, f, g = "";
            return g += '\n  <li>\n    <a href="', (d = c.url) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.url, d = typeof d === i ? d.apply(a) : d), g += j(d) + '" target="_blank">', f = {
                hash: {},
                data: b
            }, g += j((d = c.or || a.or, d ? d.call(a, a.title, a.url, f) : k.call(a, "or", a.title, a.url, f))) + '</a>\n    <!-- <span class="pull-right">', (e = c.num_saves) ? e = e.call(a, {
                hash: {},
                data: b
            }) : (e = a.num_saves, e = typeof e === i ? e.apply(a) : e), g += j(e) + "</span> -->\n  </li>\n"
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var g, h = "", i = "function", j = this.escapeExpression, k = c.helperMissing, l = this;
        return g = c.each.call(b, b.items, {
            hash: {},
            inverse: l.noop,
            fn: l.program(1, f, e),
            data: e
        }), (g || 0 === g) && (h += g), h += "\n"
    }), this.JST["widgets/person_overview"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e = "";
            return e += "\n        <b>", (d = c.bookmark_count) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.bookmark_count, d = typeof d === k ? d.apply(a) : d), e += l(d) + "</b> links\n        "
        }
        function g(a, b) {
            var d, e = "";
            return e += "\n        <b>", (d = c.public_bookmark_count) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.public_bookmark_count, d = typeof d === k ? d.apply(a) : d), e += l(d) + "</b> public links\n      "
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var h, i, j = "", k = "function", l = this.escapeExpression, m = this, n = c.blockHelperMissing;
        return j += '<div class="person-overview cf" data-user="', (h = c.username) ? h = h.call(b, {
            hash: {},
            data: e
        }) : (h = b.username, h = typeof h === k ? h.apply(b) : h), j += l(h) + '">\n  <div class="name">\n    <img src="', (h = c.avatar_url) ? h = h.call(b, {
            hash: {},
            data: e
        }) : (h = b.avatar_url, h = typeof h === k ? h.apply(b) : h), j += l(h) + '" class="avatar-img">\n    <b>', (h = c.full_name) ? h = h.call(b, {
            hash: {},
            data: e
        }) : (h = b.full_name, h = typeof h === k ? h.apply(b) : h), j += l(h) + '</b>\n    <span class="link-count">\n      ', i = {
            hash: {},
            inverse: m.program(3, g, e),
            fn: m.program(1, f, e),
            data: e
        }, (h = c.ifMe) ? h = h.call(b, i) : (h = b.ifMe, h = typeof h === k ? h.apply(b) : h), c.ifMe || (h = n.call(b, h, i)), (h || 0 === h) && (j += h), j += '\n    </span>\n  </div>\n  <p class="bio">', (h = c.profile_bio) ? h = h.call(b, {
            hash: {},
            data: e
        }) : (h = b.profile_bio, h = typeof h === k ? h.apply(b) : h), j += l(h) + '</p>\n  <div class="tags-wrapper">Tags: ', (h = c.tags_html) ? h = h.call(b, {
            hash: {},
            data: e
        }) : (h = b.tags_html, h = typeof h === k ? h.apply(b) : h), (h || 0 === h) && (j += h), j += "</div>\n</div>\n"
    }), this.JST["widgets/tag_input"] = a.template(function(a, b, c, d, e) {
        function f(a, b) {
            var d, e = "";
            return e += '\n  <input type="text" name="', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === k ? d.apply(a) : d), e += l(d) + '" id="', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === k ? d.apply(a) : d), e += l(d) + '" autocomplete="off" value="', (d = c.value) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.value, d = typeof d === k ? d.apply(a) : d), e += l(d) + '" class="form-control tag-input" spellcheck="false">\n  '
        }
        function g(a, b) {
            var d, e = "";
            return e += '\n  <textarea type="text" name="', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === k ? d.apply(a) : d), e += l(d) + '" id="', (d = c.name) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.name, d = typeof d === k ? d.apply(a) : d), e += l(d) + '" autocomplete="off" class="form-control tag-input" spellcheck="false">', (d = c.value) ? d = d.call(a, {
                hash: {},
                data: b
            }) : (d = a.value, d = typeof d === k ? d.apply(a) : d), e += l(d) + "</textarea>\n"
        }
        function h(a) {
            var b, c = "";
            return c += "\n    <li>" + l((b = a.name, typeof b === k ? b.apply(a) : b)) + "</li>\n  "
        }
        this.compilerInfo = [4, ">= 1.0.0"], c = this.merge(c, a.helpers), e = e || {};
        var i, j = "", k = "function", l = this.escapeExpression, m = this;
        return i = c.unless.call(b, b.multiline, {
            hash: {},
            inverse: m.program(3, g, e),
            fn: m.program(1, f, e),
            data: e
        }), (i || 0 === i) && (j += i), j += '\n<ul class="tag-input-candidates">\n  ', i = c.each.call(b, b.tags, {
            hash: {},
            inverse: m.noop,
            fn: m.program(5, h, e),
            data: e
        }), (i || 0 === i) && (j += i), j += "\n</ul>\n"
    }), this.JST
}), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/base/hbsview", ["chaplin", "templates"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.getTemplateFunction = function() {
                return this.template ? this._cachedTemplateFunction || (this._cachedTemplateFunction = c[this.template]) : void 0
            }, d.prototype.startLoading = function() {
                return $(".load-more").addClass("on")
            }, d.prototype.stopLoading = function() {
                return $(".load-more").removeClass("on")
            }, d.prototype.show = function() {
                return this.$el.show()
            }, d.prototype.hide = function() {
                return this.$el.hide()
            }, d
        }(a.View)
    })
}.call(this), function(a) {
    function b(a, b) {
        return function(c) {
            return i(a.call(this, c), b)
        }
    }
    function c(a, b) {
        return function(c) {
            return this.lang().ordinal(a.call(this, c), b)
        }
    }
    function d() {}
    function e(a) {
        g(this, a)
    }
    function f(a) {
        var b = a.years || a.year || a.y || 0, c = a.months || a.month || a.M || 0, d = a.weeks || a.week || a.w || 0, e = a.days || a.day || a.d || 0, f = a.hours || a.hour || a.h || 0, g = a.minutes || a.minute || a.m || 0, h = a.seconds || a.second || a.s || 0, i = a.milliseconds || a.millisecond || a.ms || 0;
        this._input = a, this._milliseconds = i + 1e3 * h + 6e4 * g + 36e5 * f, this._days = e + 7 * d, this._months = c + 12 * b, this._data = {}, this._bubble()
    }
    function g(a, b) {
        for (var c in b)
            b.hasOwnProperty(c) && (a[c] = b[c]);
        return a
    }
    function h(a) {
        return 0 > a ? Math.ceil(a) : Math.floor(a)
    }
    function i(a, b) {
        for (var c = a + ""; c.length < b;)
            c = "0" + c;
        return c
    }
    function j(a, b, c, d) {
        var e, f, g = b._milliseconds, h = b._days, i = b._months;
        g && a._d.setTime( + a._d + g * c), (h || i) && (e = a.minute(), f = a.hour()), h && a.date(a.date() + h * c), i && a.month(a.month() + i * c), g&&!d && H.updateOffset(a), (h || i) && (a.minute(e), a.hour(f))
    }
    function k(a) {
        return "[object Array]" === Object.prototype.toString.call(a)
    }
    function l(a, b) {
        var c, d = Math.min(a.length, b.length), e = Math.abs(a.length - b.length), f = 0;
        for (c = 0; d > c; c++)
            ~~a[c]!==~~b[c] && f++;
        return f + e
    }
    function m(a) {
        return a ? eb[a] || a.toLowerCase().replace(/(.)s$/, "$1") : a
    }
    function n(a, b) {
        return b.abbr = a, L[a] || (L[a] = new d), L[a].set(b), L[a]
    }
    function o(a) {
        if (!a)
            return H.fn._lang;
        if (!L[a] && M)
            try {
                require("./lang/" + a)
        } catch (b) {
            return H.fn._lang
        }
        return L[a]
    }
    function p(a) {
        return a.match(/\[.*\]/) ? a.replace(/^\[|\]$/g, "") : a.replace(/\\/g, "")
    }
    function q(a) {
        var b, c, d = a.match(P);
        for (b = 0, c = d.length; c > b; b++)
            d[b] = ib[d[b]] ? ib[d[b]] : p(d[b]);
        return function(e) {
            var f = "";
            for (b = 0; c > b; b++)
                f += d[b]instanceof Function ? d[b].call(e, a) : d[b];
            return f
        }
    }
    function r(a, b) {
        function c(b) {
            return a.lang().longDateFormat(b) || b
        }
        for (var d = 5; d--&&Q.test(b);)
            b = b.replace(Q, c);
        return fb[b] || (fb[b] = q(b)), fb[b](a)
    }
    function s(a, b) {
        switch (a) {
        case"DDDD":
            return T;
        case"YYYY":
            return U;
        case"YYYYY":
            return V;
        case"S":
        case"SS":
        case"SSS":
        case"DDD":
            return S;
        case"MMM":
        case"MMMM":
        case"dd":
        case"ddd":
        case"dddd":
            return W;
        case"a":
        case"A":
            return o(b._l)._meridiemParse;
        case"X":
            return Z;
        case"Z":
        case"ZZ":
            return X;
        case"T":
            return Y;
        case"MM":
        case"DD":
        case"YY":
        case"HH":
        case"hh":
        case"mm":
        case"ss":
        case"M":
        case"D":
        case"d":
        case"H":
        case"h":
        case"m":
        case"s":
            return R;
        default:
            return new RegExp(a.replace("\\", ""))
        }
    }
    function t(a) {
        var b = (X.exec(a) || [])[0], c = (b + "").match(bb) || ["-", 0, 0], d =+ (60 * c[1])+~~c[2];
        return "+" === c[0]?-d : d
    }
    function u(a, b, c) {
        var d, e = c._a;
        switch (a) {
        case"M":
        case"MM":
            e[1] = null == b ? 0 : ~~b - 1;
            break;
        case"MMM":
        case"MMMM":
            d = o(c._l).monthsParse(b), null != d ? e[1] = d : c._isValid=!1;
            break;
        case"D":
        case"DD":
        case"DDD":
        case"DDDD":
            null != b && (e[2]=~~b);
            break;
        case"YY":
            e[0]=~~b + (~~b > 68 ? 1900 : 2e3);
            break;
        case"YYYY":
        case"YYYYY":
            e[0]=~~b;
            break;
        case"a":
        case"A":
            c._isPm = o(c._l).isPM(b);
            break;
        case"H":
        case"HH":
        case"h":
        case"hh":
            e[3]=~~b;
            break;
        case"m":
        case"mm":
            e[4]=~~b;
            break;
        case"s":
        case"ss":
            e[5]=~~b;
            break;
        case"S":
        case"SS":
        case"SSS":
            e[6]=~~(1e3 * ("0." + b));
            break;
        case"X":
            c._d = new Date(1e3 * parseFloat(b));
            break;
        case"Z":
        case"ZZ":
            c._useUTC=!0, c._tzm = t(b)
        }
        null == b && (c._isValid=!1)
    }
    function v(a) {
        var b, c, d = [];
        if (!a._d) {
            for (b = 0; 7 > b; b++)
                a._a[b] = d[b] = null == a._a[b] ? 2 === b ? 1 : 0 : a._a[b];
            d[3]+=~~((a._tzm || 0) / 60), d[4]+=~~((a._tzm || 0)%60), c = new Date(0), a._useUTC ? (c.setUTCFullYear(d[0], d[1], d[2]), c.setUTCHours(d[3], d[4], d[5], d[6])) : (c.setFullYear(d[0], d[1], d[2]), c.setHours(d[3], d[4], d[5], d[6])), a._d = c
        }
    }
    function w(a) {
        var b, c, d = a._f.match(P), e = a._i;
        for (a._a = [], b = 0; b < d.length; b++)
            c = (s(d[b], a).exec(e) || [])[0], c && (e = e.slice(e.indexOf(c) + c.length)), ib[d[b]] && u(d[b], c, a);
        e && (a._il = e), a._isPm && a._a[3] < 12 && (a._a[3] += 12), a._isPm===!1 && 12 === a._a[3] && (a._a[3] = 0), v(a)
    }
    function x(a) {
        var b, c, d, f, h, i = 99;
        for (f = 0; f < a._f.length; f++)
            b = g({}, a), b._f = a._f[f], w(b), c = new e(b), h = l(b._a, c.toArray()), c._il && (h += c._il.length), i > h && (i = h, d = c);
        g(a, d)
    }
    function y(a) {
        var b, c = a._i, d = $.exec(c);
        if (d) {
            for (a._f = "YYYY-MM-DD" + (d[2] || " "), b = 0; 4 > b; b++)
                if (ab[b][1].exec(c)) {
                    a._f += ab[b][0];
                    break
                }
            X.exec(c) && (a._f += " Z"), w(a)
        } else 
            a._d = new Date(c)
    }
    function z(b) {
        var c = b._i, d = N.exec(c);
        c === a ? b._d = new Date : d ? b._d = new Date( + d[1]) : "string" == typeof c ? y(b) : k(c) ? (b._a = c.slice(0), v(b)) : b._d = new Date(c instanceof Date?+c : c)
    }
    function A(a, b, c, d, e) {
        return e.relativeTime(b || 1, !!c, a, d)
    }
    function B(a, b, c) {
        var d = K(Math.abs(a) / 1e3), e = K(d / 60), f = K(e / 60), g = K(f / 24), h = K(g / 365), i = 45 > d && ["s", d] || 1 === e && ["m"] || 45 > e && ["mm", e] || 1 === f && ["h"] || 22 > f && ["hh", f] || 1 === g && ["d"] || 25 >= g && ["dd", g] || 45 >= g && ["M"] || 345 > g && ["MM", K(g / 30)] || 1 === h && ["y"] || ["yy", h];
        return i[2] = b, i[3] = a > 0, i[4] = c, A.apply({}, i)
    }
    function C(a, b, c) {
        var d, e = c - b, f = c - a.day();
        return f > e && (f -= 7), e - 7 > f && (f += 7), d = H(a).add("d", f), {
            week: Math.ceil(d.dayOfYear() / 7),
            year: d.year()
        }
    }
    function D(a) {
        var b = a._i, c = a._f;
        return null === b || "" === b ? null : ("string" == typeof b && (a._i = b = o().preparse(b)), H.isMoment(b) ? (a = g({}, b), a._d = new Date( + b._d)) : c ? k(c) ? x(a) : w(a) : z(a), new e(a))
    }
    function E(a, b) {
        H.fn[a] = H.fn[a + "s"] = function(a) {
            var c = this._isUTC ? "UTC": "";
            return null != a ? (this._d["set" + c + b](a), H.updateOffset(this), this) : this._d["get" + c + b]()
        }
    }
    function F(a) {
        H.duration.fn[a] = function() {
            return this._data[a]
        }
    }
    function G(a, b) {
        H.duration.fn["as" + a] = function() {
            return + this / b
        }
    }
    for (var H, I, J = "2.1.0", K = Math.round, L = {}, M = "undefined" != typeof module && module.exports, N = /^\/?Date\((\-?\d+)/i, O = /(\-)?(\d*)?\.?(\d+)\:(\d+)\:(\d+)\.?(\d{3})?/, P = /(\[[^\[]*\])|(\\)?(Mo|MM?M?M?|Do|DDDo|DD?D?D?|ddd?d?|do?|w[o|w]?|W[o|W]?|YYYYY|YYYY|YY|gg(ggg?)?|GG(GGG?)?|e|E|a|A|hh?|HH?|mm?|ss?|SS?S?|X|zz?|ZZ?|.)/g, Q = /(\[[^\[]*\])|(\\)?(LT|LL?L?L?|l{1,4})/g, R = /\d\d?/, S = /\d{1,3}/, T = /\d{3}/, U = /\d{1,4}/, V = /[+\-]?\d{1,6}/, W = /[0-9]*['a-z\u00A0-\u05FF\u0700-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]+|[\u0600-\u06FF\/]+(\s*?[\u0600-\u06FF]+){1,2}/i, X = /Z|[\+\-]\d\d:?\d\d/i, Y = /T/i, Z = /[\+\-]?\d+(\.\d{1,3})?/, $ = /^\s*\d{4}-\d\d-\d\d((T| )(\d\d(:\d\d(:\d\d(\.\d\d?\d?)?)?)?)?([\+\-]\d\d:?\d\d)?)?/, _ = "YYYY-MM-DDTHH:mm:ssZ", ab = [["HH:mm:ss.S", /(T| )\d\d:\d\d:\d\d\.\d{1,3}/], ["HH:mm:ss", /(T| )\d\d:\d\d:\d\d/], ["HH:mm", /(T| )\d\d:\d\d/], ["HH", /(T| )\d\d/]], bb = /([\+\-]|\d\d)/gi, cb = "Date|Hours|Minutes|Seconds|Milliseconds".split("|")
        , db = {
        Milliseconds: 1,
        Seconds: 1e3,
        Minutes: 6e4,
        Hours: 36e5,
        Days: 864e5,
        Months: 2592e6,
        Years: 31536e6
    }, eb = {
        ms: "millisecond",
        s: "second",
        m: "minute",
        h: "hour",
        d: "day",
        w: "week",
        M: "month",
        y: "year"
    }, fb = {}, gb = "DDD w W M D d".split(" "), hb = "M D H h m s w W".split(" "), ib = {
        M: function() {
            return this.month() + 1
        },
        MMM: function(a) {
            return this.lang().monthsShort(this, a)
        },
        MMMM: function(a) {
            return this.lang().months(this, a)
        },
        D: function() {
            return this.date()
        },
        DDD: function() {
            return this.dayOfYear()
        },
        d: function() {
            return this.day()
        },
        dd: function(a) {
            return this.lang().weekdaysMin(this, a)
        },
        ddd: function(a) {
            return this.lang().weekdaysShort(this, a)
        },
        dddd: function(a) {
            return this.lang().weekdays(this, a)
        },
        w: function() {
            return this.week()
        },
        W: function() {
            return this.isoWeek()
        },
        YY: function() {
            return i(this.year()%100, 2)
        },
        YYYY: function() {
            return i(this.year(), 4)
        },
        YYYYY: function() {
            return i(this.year(), 5)
        },
        gg: function() {
            return i(this.weekYear()%100, 2)
        },
        gggg: function() {
            return this.weekYear()
        },
        ggggg: function() {
            return i(this.weekYear(), 5)
        },
        GG: function() {
            return i(this.isoWeekYear()%100, 2)
        },
        GGGG: function() {
            return this.isoWeekYear()
        },
        GGGGG: function() {
            return i(this.isoWeekYear(), 5)
        },
        e: function() {
            return this.weekday()
        },
        E: function() {
            return this.isoWeekday()
        },
        a: function() {
            return this.lang().meridiem(this.hours(), this.minutes(), !0)
        },
        A: function() {
            return this.lang().meridiem(this.hours(), this.minutes(), !1)
        },
        H: function() {
            return this.hours()
        },
        h: function() {
            return this.hours()%12 || 12
        },
        m: function() {
            return this.minutes()
        },
        s: function() {
            return this.seconds()
        },
        S: function() {
            return ~~(this.milliseconds() / 100)
        },
        SS: function() {
            return i(~~(this.milliseconds() / 10), 2)
        },
        SSS: function() {
            return i(this.milliseconds(), 3)
        },
        Z: function() {
            var a =- this.zone(), b = "+";
            return 0 > a && (a =- a, b = "-"), b + i(~~(a / 60), 2) + ":" + i(~~a%60, 2)
        },
        ZZ: function() {
            var a =- this.zone(), b = "+";
            return 0 > a && (a =- a, b = "-"), b + i(~~(10 * a / 6), 4)
        },
        z: function() {
            return this.zoneAbbr()
        },
        zz: function() {
            return this.zoneName()
        },
        X: function() {
            return this.unix()
        }
    };
    gb.length;
    )I = gb.pop(), ib[I + "o"] = c(ib[I], I);
    for (; hb.length;)
        I = hb.pop(), ib[I + I] = b(ib[I], 2);
    for (ib.DDDD = b(ib.DDD, 3), d.prototype = {
        set: function(a) {
            var b, c;
            for (c in a)
                b = a[c], "function" == typeof b ? this[c] = b : this["_" + c] = b
        },
        _months: "January_February_March_April_May_June_July_August_September_October_November_December".split("_"),
        months: function(a) {
            return this._months[a.month()]
        },
        _monthsShort: "Jan_Feb_Mar_Apr_May_Jun_Jul_Aug_Sep_Oct_Nov_Dec".split("_"),
        monthsShort: function(a) {
            return this._monthsShort[a.month()]
        },
        monthsParse: function(a) {
            var b, c, d;
            for (this._monthsParse || (this._monthsParse = []), b = 0; 12 > b; b++)
                if (this._monthsParse[b] || (c = H([2e3, b]), d = "^" + this.months(c, "") + "|^" + this.monthsShort(c, ""), this._monthsParse[b] = new RegExp(d.replace(".", ""), "i")), this._monthsParse[b].test(a))
                    return b
        },
        _weekdays: "Sunday_Monday_Tuesday_Wednesday_Thursday_Friday_Saturday".split("_"),
        weekdays: function(a) {
            return this._weekdays[a.day()]
        },
        _weekdaysShort: "Sun_Mon_Tue_Wed_Thu_Fri_Sat".split("_"),
        weekdaysShort: function(a) {
            return this._weekdaysShort[a.day()]
        },
        _weekdaysMin: "Su_Mo_Tu_We_Th_Fr_Sa".split("_"),
        weekdaysMin: function(a) {
            return this._weekdaysMin[a.day()]
        },
        weekdaysParse: function(a) {
            var b, c, d;
            for (this._weekdaysParse || (this._weekdaysParse = []), b = 0; 7 > b; b++)
                if (this._weekdaysParse[b] || (c = H([2e3, 1]).day(b), d = "^" + this.weekdays(c, "") + "|^" + this.weekdaysShort(c, "") + "|^" + this.weekdaysMin(c, ""), this._weekdaysParse[b] = new RegExp(d.replace(".", ""), "i")), this._weekdaysParse[b].test(a))
                    return b
        },
        _longDateFormat: {
            LT: "h:mm A",
            L: "MM/DD/YYYY",
            LL: "MMMM D YYYY",
            LLL: "MMMM D YYYY LT",
            LLLL: "dddd, MMMM D YYYY LT"
        },
        longDateFormat: function(a) {
            var b = this._longDateFormat[a];
            return !b && this._longDateFormat[a.toUpperCase()] && (b = this._longDateFormat[a.toUpperCase()].replace(/MMMM|MM|DD|dddd/g, function(a) {
                return a.slice(1)
            }), this._longDateFormat[a] = b), b
        },
        isPM: function(a) {
            return "p" === (a + "").toLowerCase()[0]
        },
        _meridiemParse: /[ap]\.?m?\.?/i,
        meridiem: function(a, b, c) {
            return a > 11 ? c ? "pm" : "PM" : c ? "am" : "AM"
        },
        _calendar: {
            sameDay: "[Today at] LT",
            nextDay: "[Tomorrow at] LT",
            nextWeek: "dddd [at] LT",
            lastDay: "[Yesterday at] LT",
            lastWeek: "[Last] dddd [at] LT",
            sameElse: "L"
        },
        calendar: function(a, b) {
            var c = this._calendar[a];
            return "function" == typeof c ? c.apply(b) : c
        },
        _relativeTime: {
            future: "in %s",
            past: "%s ago",
            s: "a few seconds",
            m: "a minute",
            mm: "%d minutes",
            h: "an hour",
            hh: "%d hours",
            d: "a day",
            dd: "%d days",
            M: "a month",
            MM: "%d months",
            y: "a year",
            yy: "%d years"
        },
        relativeTime: function(a, b, c, d) {
            var e = this._relativeTime[c];
            return "function" == typeof e ? e(a, b, c, d) : e.replace(/%d/i, a)
        },
        pastFuture: function(a, b) {
            var c = this._relativeTime[a > 0 ? "future": "past"];
            return "function" == typeof c ? c(b) : c.replace(/%s/i, b)
        },
        ordinal: function(a) {
            return this._ordinal.replace("%d", a)
        },
        _ordinal: "%d",
        preparse: function(a) {
            return a
        },
        postformat: function(a) {
            return a
        },
        week: function(a) {
            return C(a, this._week.dow, this._week.doy).week
        },
        _week: {
            dow: 0,
            doy: 6
        }
    }, H = function(a, b, c) {
        return D({
            _i: a,
            _f: b,
            _l: c,
            _isUTC: !1
        })
    }, H.utc = function(a, b, c) {
        return D({
            _useUTC: !0,
            _isUTC: !0,
            _l: c,
            _i: a,
            _f: b
        })
    }, H.unix = function(a) {
        return H(1e3 * a)
    }, H.duration = function(a, b) {
        var c, d, e = H.isDuration(a), g = "number" == typeof a, h = e ? a._input: g ? {}
        : a, i = O.exec(a);
        return g ? b ? h[b] = a : h.milliseconds = a : i && (c = "-" === i[1]?-1 : 1, h = {
            y : 0, d : ~~i[2] * c, h : ~~i[3] * c, m : ~~i[4] * c, s : ~~i[5] * c, ms : ~~i[6] * c
        }), d = new f(h), e && a.hasOwnProperty("_lang") && (d._lang = a._lang), d
    }, H.version = J, H.defaultFormat = _, H.updateOffset = function() {}, H.lang = function(a, b) {
        return a ? (b ? n(a, b) : L[a] || o(a), void(H.duration.fn._lang = H.fn._lang = o(a))) : H.fn._lang._abbr
    }, H.langData = function(a) {
        return a && a._lang && a._lang._abbr && (a = a._lang._abbr), o(a)
    }, H.isMoment = function(a) {
        return a instanceof e
    }, H.isDuration = function(a) {
        return a instanceof f
    }, H.fn = e.prototype = {
        clone: function() {
            return H(this)
        },
        valueOf: function() {
            return + this._d + 6e4 * (this._offset || 0)
        },
        unix: function() {
            return Math.floor( + this / 1e3)
        },
        toString: function() {
            return this.format("ddd MMM DD YYYY HH:mm:ss [GMT]ZZ")
        },
        toDate: function() {
            return this._offset ? new Date( + this) : this._d
        },
        toISOString: function() {
            return r(H(this).utc(), "YYYY-MM-DD[T]HH:mm:ss.SSS[Z]")
        },
        toArray: function() {
            var a = this;
            return [a.year(), a.month(), a.date(), a.hours(), a.minutes(), a.seconds(), a.milliseconds()]
        },
        isValid: function() {
            return null == this._isValid && (this._isValid = this._a?!l(this._a, (this._isUTC ? H.utc(this._a) : H(this._a)).toArray()) : !isNaN(this._d.getTime())), !!this._isValid
        },
        utc: function() {
            return this.zone(0)
        },
        local: function() {
            return this.zone(0), this._isUTC=!1, this
        },
        format: function(a) {
            var b = r(this, a || H.defaultFormat);
            return this.lang().postformat(b)
        },
        add: function(a, b) {
            var c;
            return c = "string" == typeof a ? H.duration( + b, a) : H.duration(a, b), j(this, c, 1), this
        },
        subtract: function(a, b) {
            var c;
            return c = "string" == typeof a ? H.duration( + b, a) : H.duration(a, b), j(this, c, - 1), this
        },
        diff: function(a, b, c) {
            var d, e, f = this._isUTC ? H(a).zone(this._offset || 0): H(a).local(), g = 6e4 * (this.zone() - f.zone());
            return b = m(b), "year" === b || "month" === b ? (d = 432e5 * (this.daysInMonth() + f.daysInMonth()), e = 12 * (this.year() - f.year()) + (this.month() - f.month()), e += (this - H(this).startOf("month") - (f - H(f).startOf("month"))) / d, e -= 6e4 * (this.zone() - H(this).startOf("month").zone() - (f.zone() - H(f).startOf("month").zone())) / d, "year" === b && (e/=12)) : (d = this - f, e = "second" === b ? d / 1e3 : "minute" === b ? d / 6e4 : "hour" === b ? d / 36e5 : "day" === b ? (d - g) / 864e5 : "week" === b ? (d - g) / 6048e5 : d), c ? e : h(e)
        },
        from: function(a, b) {
            return H.duration(this.diff(a)).lang(this.lang()._abbr).humanize(!b)
        },
        fromNow: function(a) {
            return this.from(H(), a)
        },
        calendar: function() {
            var a = this.diff(H().startOf("day"), "days", !0), b =- 6 > a ? "sameElse" : - 1 > a ? "lastWeek" : 0 > a ? "lastDay" : 1 > a ? "sameDay" : 2 > a ? "nextDay" : 7 > a ? "nextWeek" : "sameElse";
            return this.format(this.lang().calendar(b, this))
        },
        isLeapYear: function() {
            var a = this.year();
            return a%4 === 0 && a%100 !== 0 || a%400 === 0
        },
        isDST: function() {
            return this.zone() < this.clone().month(0).zone() || this.zone() < this.clone().month(5).zone()
        },
        day: function(a) {
            var b = this._isUTC ? this._d.getUTCDay(): this._d.getDay();
            return null != a ? "string" == typeof a && (a = this.lang().weekdaysParse(a), "number" != typeof a) ? this : this.add({
                d: a - b
            }) : b
        },
        month: function(a) {
            var b, c = this._isUTC ? "UTC": "";
            return null != a ? "string" == typeof a && (a = this.lang().monthsParse(a), "number" != typeof a) ? this : (b = this.date(), this.date(1), this._d["set" + c + "Month"](a), this.date(Math.min(b, this.daysInMonth())), H.updateOffset(this), this) : this._d["get" + c + "Month"]()
        },
        startOf: function(a) {
            switch (a = m(a)) {
            case"year":
                this.month(0);
            case"month":
                this.date(1);
            case"week":
            case"day":
                this.hours(0);
            case"hour":
                this.minutes(0);
            case"minute":
                this.seconds(0);
            case"second":
                this.milliseconds(0)
            }
            return "week" === a && this.weekday(0), this
        },
        endOf: function(a) {
            return this.startOf(a).add(a, 1).subtract("ms", 1)
        },
        isAfter: function(a, b) {
            return b = "undefined" != typeof b ? b : "millisecond", + this.clone().startOf(b)>+H(a).startOf(b)
        },
        isBefore: function(a, b) {
            return b = "undefined" != typeof b ? b : "millisecond", + this.clone().startOf(b)<+H(a).startOf(b)
        },
        isSame: function(a, b) {
            return b = "undefined" != typeof b ? b : "millisecond", + this.clone().startOf(b) ===+ H(a).startOf(b)
        },
        min: function(a) {
            return a = H.apply(null, arguments), this > a ? this : a
        },
        max: function(a) {
            return a = H.apply(null, arguments), a > this ? this : a
        },
        zone: function(a) {
            var b = this._offset || 0;
            return null == a ? this._isUTC ? b : this._d.getTimezoneOffset() : ("string" == typeof a && (a = t(a)), Math.abs(a) < 16 && (a*=60), this._offset = a, this._isUTC=!0, b !== a && j(this, H.duration(b - a, "m"), 1, !0), this)
        },
        zoneAbbr: function() {
            return this._isUTC ? "UTC" : ""
        },
        zoneName: function() {
            return this._isUTC ? "Coordinated Universal Time" : ""
        },
        daysInMonth: function() {
            return H.utc([this.year(), this.month() + 1, 0]).date()
        },
        dayOfYear: function(a) {
            var b = K((H(this).startOf("day") - H(this).startOf("year")) / 864e5) + 1;
            return null == a ? b : this.add("d", a - b)
        },
        weekYear: function(a) {
            var b = C(this, this.lang()._week.dow, this.lang()._week.doy).year;
            return null == a ? b : this.add("y", a - b)
        },
        isoWeekYear: function(a) {
            var b = C(this, 1, 4).year;
            return null == a ? b : this.add("y", a - b)
        },
        week: function(a) {
            var b = this.lang().week(this);
            return null == a ? b : this.add("d", 7 * (a - b))
        },
        isoWeek: function(a) {
            var b = C(this, 1, 4).week;
            return null == a ? b : this.add("d", 7 * (a - b))
        },
        weekday: function(a) {
            var b = (this._d.getDay() + 7 - this.lang()._week.dow)%7;
            return null == a ? b : this.add("d", a - b)
        },
        isoWeekday: function(a) {
            return null == a ? this.day() || 7 : this.day(this.day()%7 ? a : a - 7)
        },
        lang: function(b) {
            return b === a ? this._lang : (this._lang = o(b), this)
        }
    }, I = 0; I < cb.length; I++)
        E(cb[I].toLowerCase().replace(/s$/, ""), cb[I]);
    E("year", "FullYear"), H.fn.days = H.fn.day, H.fn.months = H.fn.month, H.fn.weeks = H.fn.week, H.fn.isoWeeks = H.fn.isoWeek, H.fn.toJSON = H.fn.toISOString, H.duration.fn = f.prototype = {
        _bubble: function() {
            var a, b, c, d, e = this._milliseconds, f = this._days, g = this._months, i = this._data;
            i.milliseconds = e%1e3, a = h(e / 1e3), i.seconds = a%60, b = h(a / 60), i.minutes = b%60, c = h(b / 60), i.hours = c%24, f += h(c / 24), i.days = f%30, g += h(f / 30), i.months = g%12, d = h(g / 12), i.years = d
        },
        weeks: function() {
            return h(this.days() / 7)
        },
        valueOf: function() {
            return this._milliseconds + 864e5 * this._days + this._months%12 * 2592e6 + 31536e6*~~(this._months / 12)
        },
        humanize: function(a) {
            var b =+ this, c = B(b, !a, this.lang());
            return a && (c = this.lang().pastFuture(b, c)), this.lang().postformat(c)
        },
        add: function(a, b) {
            var c = H.duration(a, b);
            return this._milliseconds += c._milliseconds, this._days += c._days, this._months += c._months, this._bubble(), this
        },
        subtract: function(a, b) {
            var c = H.duration(a, b);
            return this._milliseconds -= c._milliseconds, this._days -= c._days, this._months -= c._months, this._bubble(), this
        },
        get: function(a) {
            return a = m(a), this[a.toLowerCase() + "s"]()
        },
        as: function(a) {
            return a = m(a), this["as" + a.charAt(0).toUpperCase() + a.slice(1) + "s"]()
        },
        lang: H.fn.lang
    };
    for (I in db)
        db.hasOwnProperty(I) && (G(I, db[I]), F(I.toLowerCase()));
    G("Weeks", 6048e5), H.duration.fn.asMonths = function() {
        return ( + this - 31536e6 * this.years()) / 2592e6 + 12 * this.years()
    }, H.lang("en", {
        ordinal: function(a) {
            var b = a%10, c = 1===~~(a%100 / 10) ? "th": 1 === b ? "st": 2 === b ? "nd": 3 === b ? "rd": "th";
            return a + c
        }
    }), M && (module.exports = H), "undefined" == typeof ender && (this.moment = H), "function" == typeof define && define.amd && define("moment", [], function() {
        return H
    })
}.call(this), function(a) {
    function b() {
        try {
            return h in a && a[h]
        } catch (b) {
            return !1
        }
    }
    function c(a) {
        return function() {
            var b = Array.prototype.slice.call(arguments, 0);
            b.unshift(e), j.appendChild(e), e.addBehavior("#default#userData"), e.load(h);
            var c = a.apply(f, b);
            return j.removeChild(e), c
        }
    }
    function d(a) {
        return a.replace(/^d/, "___$&").replace(m, "___")
    }
    var e, f = {}, g = a.document, h = "localStorage", i = "script";
    if (f.disabled=!1, f.set = function() {}, f.get = function() {}, f.remove = function() {}, f.clear = function() {}, f.transact = function(a, b, c) {
        var d = f.get(a);
        null == c && (c = b, b = null), "undefined" == typeof d && (d = b || {}), c(d), f.set(a, d)
    }, f.getAll = function() {}, f.forEach = function() {}, f.serialize = function(a) {
        return JSON.stringify(a)
    }, f.deserialize = function(a) {
        if ("string" != typeof a)
            return void 0;
        try {
            return JSON.parse(a)
        } catch (b) {
            return a || void 0
        }
    }, b())
        e = a[h], f.set = function(a, b) {
            return void 0 === b ? f.remove(a) : (e.setItem(a, f.serialize(b)), b)
        }, f.get = function(a) {
            return f.deserialize(e.getItem(a))
        }, f.remove = function(a) {
            e.removeItem(a)
        }, f.clear = function() {
            e.clear()
        }, f.getAll = function() {
            var a = {};
            return f.forEach(function(b, c) {
                a[b] = c
            }), a
        }, f.forEach = function(a) {
            for (var b = 0; b < e.length; b++) {
                var c = e.key(b);
                a(c, f.get(c))
            }
        };
    else if (g.documentElement.addBehavior) {
        var j, k;
        try {
            k = new ActiveXObject("htmlfile"), k.open(), k.write("<" + i + ">document.w=window</" + i + '><iframe src="/favicon.ico"></iframe>'), k.close(), j = k.w.frames[0].document, e = j.createElement("div")
        } catch (l) {
            e = g.createElement("div"), j = g.body
        }
        var m = new RegExp("[!\"#$%&'()*+,/\\\\:;<=>?@[\\]^`{|}~]", "g");
        f.set = c(function(a, b, c) {
            return b = d(b), void 0 === c ? f.remove(b) : (a.setAttribute(b, f.serialize(c)), a.save(h), c)
        }), f.get = c(function(a, b) {
            return b = d(b), f.deserialize(a.getAttribute(b))
        }), f.remove = c(function(a, b) {
            b = d(b), a.removeAttribute(b), a.save(h)
        }), f.clear = c(function(a) {
            var b = a.XMLDocument.documentElement.attributes;
            a.load(h);
            for (var c, d = 0; c = b[d]; d++)
                a.removeAttribute(c.name);
            a.save(h)
        }), f.getAll = function() {
            var a = {};
            return f.forEach(function(b, c) {
                a[b] = c
            }), a
        }, f.forEach = c(function(a, b) {
            for (var c, d = a.XMLDocument.documentElement.attributes, e = 0; c = d[e]; ++e)
                b(c.name, f.deserialize(a.getAttribute(c.name)))
        })
    }
    try {
        var n = "__storejs__";
        f.set(n, n), f.get(n) != n && (f.disabled=!0), f.remove(n)
    } catch (l) {
        f.disabled=!0
    }
    f.enabled=!f.disabled, "undefined" != typeof module && module.exports && this.module !== module ? module.exports = f : "function" == typeof define && define.amd ? define("store", f) : a.store = f
}(Function("return this")()), function() {
    define("lib/avos", ["underscore", "moment", "store"], function(a, b, c) {
        var d, e, f;
        return e = {
            relativeTime: {
                future: "in %s",
                past: "%s",
                s: "%ds",
                m: "1m",
                mm: "%dm",
                h: "1h",
                hh: "%dh",
                d: "1d",
                dd: "%dd",
                M: "1mo",
                MM: "%dmo",
                y: "1y",
                yy: "%dy"
            },
            calendar: {
                lastDay: "[Yesterday]",
                sameDay: "LT",
                nextDay: "[Tomorrow at] LT",
                lastWeek: "L",
                nextWeek: "dddd [at] LT",
                sameElse: "L"
            }
        }, b.lang("en", a.extend(e, {
            longDateFormat: {
                LT: "h:mm a",
                L: "D/M/YY",
                l: "D/M/YYYY",
                LL: "Do MMMM YYYY",
                ll: "D MMM YYYY",
                LLL: "Do MMMM YYYY LT",
                lll: "D MMM YYYY LT",
                LLLL: "dddd MMMM Do YYYY LT",
                llll: "ddd MMM D YYYY LT"
            }
        })), b.lang("en-us", a.extend(e, {
            longDateFormat: {
                LT: "h:mm a",
                L: "M/D/YY",
                l: "M/D/YYYY",
                LL: "MMMM Do YYYY",
                ll: "MMM D, YYYY",
                LLL: "MMMM Do YYYY LT",
                lll: "MMM D YYYY LT",
                LLLL: "dddd, MMMM Do YYYY LT",
                llll: "ddd, MMM D YYYY LT"
            }
        })), f = navigator.userLanguage || navigator.language, b.lang("en-us" === f.toLowerCase() ? "en-us" : "en"), d = function() {
            return window.testing ? "avosapi-test.delicious.com" : c.get("cn-proxy") ? "cn.api.delicious.com" : "avosapi.delicious.com"
        }, {
            api_host: d,
            api_path: "/api/v1",
            default_avatar: "//delicious-icons.s3.amazonaws.com/default-avatar-2.jpg",
            logging: !0,
            ajax: {
                options: {
                    cache: !1,
                    dataType: "json",
                    type: "GET",
                    timeout: 1e4,
                    headers: {
                        Accept: ""
                    },
                    xhrFields: {
                        withCredentials: !0
                    }
                }
            },
            composer_my_tags_limit: 25,
            webapp_url: "https://delicious.com/delicious.webapp"
        }
    })
}.call(this), function() {
    var a = [].indexOf || function(a) {
        for (var b = 0, c = this.length; c > b; b++)
            if (b in this && this[b] === a)
                return b;
        return - 1
    };
    define("lib/popup", ["lib/avos"], function(b) {
        var c;
        return c = {
            allowedOrigins: ["http://avosapi.delicious.com", "https://avosapi.delicious.com"],
            options: {
                menubar: "no",
                toolbar: "no",
                width: 790,
                height: 360
            }
        }, function(d) {
            var e, f;
            return null == d && (d = {}), f = _.extend({}, c.options, d), delete f.url, delete f.success, delete f.popupWindowName, delete f.crossWindowMessageName, e = function(b) {
                var f;
                return b = b.originalEvent, f = b.origin, a.call(c.allowedOrigins, f) < 0 || b.data.name !== d.crossWindowMessageName ? void 0 : (d.success(b.data.body), $(window).off("message", e))
            }, $(window).on("message", e), window.open("//" + b.api_host() + b.api_path + d.url, d.popupWindowName, $.param(f).replace(/&/g, ","))
        }
    })
}.call(this), function() {
    define("lib/encode", [], function() {
        return function(a) {
            return a = encodeURIComponent(a), a.replace(/\./g, "%2E")
        }
    })
}.call(this), function() {
    define("lib/obfuscate", [], function() {
        return function(a) {
            var b;
            return b = a.split("").reverse().join(""), btoa(b)
        }
    })
}.call(this), function() {
    var a = [].slice;
    define("lib/logger", [], function() {
        var b, c, d, e, f;
        return c = 15, f = (new Date).getTime(), e=!1, b = function(a) {
            var b;
            return null == a && (a = ""), b = c - a.length + 1, 0 > b && (b = 0), new Array(b).join(" ")
        }, d = function() {
            var c, d, g, h, i;
            return g = arguments[0], d = arguments[1], c = 3 <= arguments.length ? a.call(arguments, 2) : [], null == d && (d = "log"), e ? ("log" !== d && "warn" !== d && "error" !== d && (c.unshift(d), d = "log"), "string" != typeof g && (c.unshift(g), g = ""), h = (new Date).getTime(), i = h - f + "ms", console[d](b(i), i + " ■", g), _(c).each(function(a) {
                return "object" == typeof a ? _(a).each(function(a, c) {
                    return console[d](b(c), c, a)
                }) : void 0
            })) : void 0
        }, {
            log: d,
            error: function() {
                var b, c;
                return c = arguments[0], b = 2 <= arguments.length ? a.call(arguments, 1) : [], d.apply(null, [c, "error"].concat(a.call(b)))
            },
            set: function(a) {
                return null == a && (a=!1), a===!0 ? e=!0 : void 0
            },
            warn: function() {
                var b, c;
                return c = arguments[0], b = 2 <= arguments.length ? a.call(arguments, 1) : [], d.apply(null, [c, "warn"].concat(a.call(b)))
            }
        }
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("models/base/model", ["chaplin", "lib/session", "lib/avos", "lib/logger"], function(a, c, d, e) {
        var f, g;
        return c.loginFailCount = 0, c.retryCounter = {}, f = function(f) {
            function h() {
                return g = h.__super__.constructor.apply(this, arguments)
            }
            return b(h, f), h.prototype.xhrPool = [], h.prototype.api = function(b) {
                var f, g, h, i, j, k, l = this;
                return h = _.extend({}, d.ajax.options, b, {
                    dataType: $("html").hasClass("lt-ie10") ? "jsonp": "json",
                    url: "//" + d.api_host() + d.api_path + b.url,
                    success: null,
                    error: null
                }), k = new Date, i = new $.Deferred, j = function(d, f, m) {
                    var n, o;
                    if (e.log("API: Response received", {
                        time: new Date - k,
                        server_time: d.delta_ms
                    }), "invalid_credentials" === (null != d ? d.status : void 0)) {
                        if (null != (o = c.user) ? o.get("username") : void 0)
                            return void a.helpers.redirectTo({
                            url: "/logout"
                        });
                        c.loginFailCount = 0
                    }
                    if ("expired_session" === (null != d ? d.status : void 0)) {
                        if (c.loginFailCount++, e.warn("API Session Expired", {
                            url: h.url,
                            params: h,
                            user: c.user.clone(),
                            failcount: c.loginFailCount
                        }), c.loginFailCount < 4 && c.user.get("password_hash") && c.user.get("username"))
                            return e.warn("Attempting to reAuth with silent login", l), i.notify("Attempting to reAuth with silent login"), c.user.loginSilent().fail(j).done(function() {
                                return g(), c.loginFailCount = 0
                            });
                        e.error("Silent auth was not successful"), a.helpers.redirectTo({
                            url: "/logout"
                        }), c.loginFailCount = 0
                    } else if ("success" === (null != d ? d.status : void 0))
                        return e.log("API Success: " + b.url, {
                            params: h,
                            res: d
                        }), n = _.indexOf(l.xhrPool, m), - 1 !== n && l.xhrPool.splice(n, 1), $.isFunction(b.success) && b.success.apply(b, arguments), i.resolve.apply(i, arguments), void 0;
                    return e.warn("API Error", {
                        res: d,
                        status: f,
                        xhr: m
                    }), _.isFunction(b.error) && b.error.apply(b, arguments), i.reject.apply(i, arguments)
                }, f = function(a, c, d) {
                    return e.error("API Error: " + b.url, {
                        headers: a.getAllResponseHeaders(),
                        status: c,
                        error: d
                    }), "timeout" === d && l.publishEvent("system:timeout"), _.isFunction(b.error) && b.error.apply(b, arguments), i.reject.apply(i, arguments)
                }, g = function() {
                    return e.log("API: Requesting " + b.url, {
                        params: h,
                        options: b
                    }), l.xhrPool.push($.ajax(h).fail(f).done(j))
                }, g(), i
            }, h.prototype.dispose = function() {
                return _.each(this.xhrPool, function(a) {
                    return a.abort()
                }), h.__super__.dispose.apply(this, arguments)
            }, h
        }(a.Model)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("models/base/collection", ["lib/avos", "chaplin", "models/base/model"], function(a, c, d) {
        var e, f;
        return e = function(a) {
            function c() {
                return f = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.xhrPool = [], c.prototype.parse = function(a) {
                return _.isEmpty(a) ? [] : (null != a ? a.pkg : void 0) || []
            }, c.prototype.api = d.prototype.api, c.prototype.initialize = function(a, b) {
                return null == a && (a = []), null == b && (b = {}), this.urlParams = b.urlParams, delete b.urlParams, c.__super__.initialize.call(this, b)
            }, c.prototype.fetch = function(a) {
                return null == a && (a = {}), this.length > 0 && (a.remove=!1), c.__super__.fetch.call(this, a)
            }, c.prototype.sync = function(a, b, c) {
                var d;
                return d = {}, d.url = "function" == typeof this.url ? this.url() : null != b.url ? b.url : void 0, d = _.extend(d, c), this.api(d)
            }, c.prototype._anchor = function() {
                var a, b;
                return this.length < 1?-1 : (b = _.sortBy(this.pluck("time_created"), function(a) {
                    return a
                }), a = _.first(b), (a || 0) - 1)
            }, c.prototype._anchorx = function() {
                var a;
                return this.length < 1 ? null : (a = this.last(), a.get("time_created") + a.get("md5"))
            }, c.prototype._index = function() {
                var a;
                return this.length < 1 ? 0 : (a = _.sortBy(this.pluck("index"), function(a) {
                    return a
                }), _.last(a) || 0)
            }, c.prototype._score = function() {
                var a;
                return this.length < 1 ? null : (a = _.min(this.pluck("score")), a - 1)
            }, c.prototype.dispose = function() {
                return _.each(this.xhrPool, function(a) {
                    return a.abort()
                }), c.__super__.dispose.apply(this, arguments)
            }, c
        }(c.Collection)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("models/tag", ["chaplin", "models/base/model", "lib/session"], function(a, c, d) {
        var e, f;
        return e = function(a) {
            function c() {
                return f = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.normalize = function(a) {
                return "string" == typeof a ? a.trim() : void 0
            }, c.prototype.history = function() {
                var a = this;
                return this.api({
                    data: {
                        tags: this.get("name")
                    },
                    url: "/tags/stats",
                    success: function(b) {
                        var c;
                        return c = b.pkg[0], a.set({
                            bookmark_count: _.last(c.weekly_stats),
                            subscribed: c.subscribed
                        })
                    }
                })
            }, c.quote = function(a, b, c) {
                return null == b && (b = /"/g), null == c && (c = '\\"'), this.normalize(a).replace(b, c)
            }, c.unquote = function(a, b, c) {
                return null == b && (b = /\\"/g), null == c && (c = '"'), this.normalize(a).replace(b, c)
            }, c.prototype.remove = function() {
                var a = this;
                return this.api({
                    data: {
                        tag: this.get("name")
                    },
                    url: "/tags/delete",
                    success: function() {
                        var b;
                        return b = d.user.get("tags"), _.remove(b, {
                            name: a.get("name")
                        }), d.user.set({
                            tags: b
                        }), a.trigger("destroy"), a.publishEvent("tags:managed")
                    }
                })
            }, c.prototype.rename = function(a) {
                var b = this;
                return this.api({
                    data: {
                        old_tag: this.get("name"),
                        new_tag: a
                    },
                    url: "/tags/rename",
                    success: function() {
                        var c, e, f;
                        return c = b.get("name"), b.set({
                            name: a
                        }), f = d.user.get("tags"), e = _.where(f, {
                            name: c
                        }), _.each(e, function(b) {
                            return b.name = a
                        }), d.user.set({
                            tags: f
                        }), b.publishEvent("tags:managed")
                    }
                })
            }, c.prototype.subscribe = function() {
                return this.api({
                    url: "/recommend/subscription/add",
                    data: {
                        category_list: this.get("name")
                    },
                    type: "POST"
                })
            }, c.prototype.unsubscribe = function() {
                return this.api({
                    url: "/recommend/subscription/remove",
                    data: {
                        category_list: this.get("name")
                    },
                    type: "POST"
                })
            }, c
        }(c)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("models/tags", ["models/base/collection", "lib/encode", "chaplin", "lib/session", "models/tag", "store"], function(a, c, d, e, f, g) {
        var h, i;
        return h = function(a) {
            function d() {
                return i = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.model = f, d.prototype.initialize = function(a, b) {
                return null == b && (b = {}), d.__super__.initialize.apply(this, arguments), b.subscribeToFetch ? this.subscribeEvent("search:fetch", function(a) {
                    return this.urlParams.keywords = a.keywords, this.urlParams.tags = a.tags, this.reset([], {
                        silent: !0
                    }), this.fetch()
                }) : void 0
            }, d.prototype.comparator = d.name_comparator, d.prototype.ensure_comparator = function() {
                return this.comparator = "count" === g.get("tags:comparator") ? this.count_comparator : this.name_comparator, this
            }, d.prototype.count_comparator = function(a, b) {
                var c, d;
                return (a.get("num_saves") || 0) === (b.get("num_saves") || 0) ? (c = a.get("name").toLowerCase(), d = b.get("name").toLowerCase(), c > d ? 1 : - 1) : (a.get("num_saves") || 0) > (b.get("num_saves") || 0)?-1 : 1
            }, d.prototype.name_comparator = function(a) {
                return a.get("name").toLowerCase()
            }, d.prototype.sort_by_name = function(a) {
                return this.comparator = this.name_comparator, a && g.set("tags:comparator", "name"), this.sort()
            }, d.prototype.sort_by_count = function(a) {
                return this.comparator = this.count_comparator, a && g.set("tags:comparator", "count"), this.sort()
            }, d.prototype.url = function() {
                var a, b, d, f, g, h, i, j, k, l;
                return this.urlParamDefaults || (this.urlParamDefaults = {
                    type: "public",
                    tags: [],
                    keywords: [],
                    limit: 20,
                    query: null
                }), d = _.extend(this.urlParamDefaults, this.urlParams), i = d.tags, a = d.keywords, j = d.type, b = d.limit, k = d.username, g = d.query, _.isArray(i) && i.length > 0 && (i = _.map(i, function(a) {
                    return /%/.test(a) ? decodeURIComponent(a) : a
                })), f = {
                    tags: _.isArray(i) ? i.join(","): i,
                    keywords: _.isArray(a) ? a.join(","): a,
                    limit: b,
                    anchor: this._anchor(),
                    index: this._index(),
                    query: g
                }, k && (k === (null != (l = e.user) ? l.get("username") : void 0) ? j = "you" : (k = c(k), j = "public/" + k)), h = function() {
                    switch (j) {
                    case"trending":
                        return "/tags/trending";
                    case"subscription":
                        return "/recommend/subscription";
                    case"query":
                        return "/tags/query";
                    default:
                        return "/posts/" + j + "/tags"
                    }
                }(), h + "?" + $.param(f)
            }, d.prototype.parse = function(a) {
                var b, c, e, g, h, i, j, k, l = this;
                return g = d.__super__.parse.call(this, a), g ? (c = g.tags ? (i = _.keys(g.tags), h = _.map(i, function(a) {
                    return g.tags[a]
                }), e = function(a, b, c) {
                    var d;
                    return d = f.normalize(b), _.isEmpty(d) ? a : (a[d] || (a[d] = 0), a[d] += h[c], a)
                }, j = _.reduce(i, e, {}), b = j.length, delete j.length, j = _.map(j, function(a, b) {
                    var c;
                    return {
                        name: b,
                        num_saves: a,
                        owner: null != (c = l.urlParams) ? c.username: void 0
                    }
                }), b ? j.push({
                    name: "length",
                    num_saves: b
                }) : void 0, j) : g.best_tags ? _.map(g.best_tags, function(a) {
                    return {
                        name: f.normalize(a)
                    }
                }) : _.isArray(g) && g.length > 0 && _.isString(g[0]) ? _.map(g, function(a) {
                    return {
                        name: a
                    }
                }) : _.isArray(g) ? _.map(g, function(a) {
                    return a.last_week_count = _.first(_.last(a.weekly_stats, 2)), a
                }) : [], (null != (k = this.urlParams) ? k.tags : void 0) ? _.reject(c, function(a) {
                    return _.include(l.urlParams.tags, a.name)
                }) : c) : []
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/base/modal", ["views/base/hbsview"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.container = "body", c.prototype.autoRender=!0, c.prototype.autoAttach=!0, c.prototype.containerMethod = "append", c.prototype.className = "modal fade", c.prototype.render = function() {
                var a = this;
                return c.__super__.render.apply(this, arguments), this.$el.attr("tabindex", "-1"), this.$el.attr("role", "dialog"), this.$el.attr("aria-labelledby", "delicious-modal"), this.$el.attr("aria-hidden", "true"), this.$el.on("hidden.bs.modal", _.bind(this.dispose, this)), this.$el.on("shown.bs.modal", function() {
                    return a.$el.find("[autofocus]:first").focus()
                })
            }, c.prototype.attach = function() {
                return c.__super__.attach.apply(this, arguments), this.$el.modal("show"), $("body").addClass("modal-on")
            }, c.prototype.dispose = function() {
                return c.__super__.dispose.apply(this, arguments), $("body").removeClass("modal-open modal-on")
            }, c.prototype.close = function() {
                return this.$el.modal("hide")
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/forgot_password", ["chaplin", "lib/session", "views/base/modal"], function(a, c, d) {
        var e, f;
        return e = function(a) {
            function d() {
                return f = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "modals/forgot_password", d.prototype.events = {
                "submit form": "do_submit",
                "click .submit-button": "do_submit"
            }, d.prototype.do_submit = function(a) {
                var b = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.$(".submit-button").prop("disabled", "disabled"), c.user.forgotPassword(this.$("input[name='username']").val()).fail(function(a) {
                    return b.fail(a)
                }).done(function() {
                    return b.success()
                })
            }, d.prototype.fail = function(a) {
                return this.$(".submit-button").prop("disabled", ""), this.show_error("not_found" === a.status ? "That username doesn't exist." : a.error.charAt(0).toUpperCase() + a.error.slice(1))
            }, d.prototype.success = function() {
                return this.$(".send-success").removeClass("hidden"), this.show_error("")
            }, d.prototype.show_error = function(a) {
                return this.$(".error-msg").html(a)
            }, d
        }(d)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/create", ["chaplin", "lib/session", "views/base/modal"], function(a, c, d) {
        var e, f;
        return e = function(d) {
            function e() {
                return f = e.__super__.constructor.apply(this, arguments)
            }
            return b(e, d), e.prototype.template = "modals/create", e.prototype.finished=!1, e.prototype.has_error=!1, e.prototype.initialize = function(a) {
                return this.name = null != a ? a.name : void 0, this.email = null != a ? a.email : void 0
            }, e.prototype.events = {
                "blur input[name='username']": "check_username_availability",
                "submit #create-account-form": "submit_form",
                "change input[type='password']": "validate_password"
            }, e.prototype.getTemplateData = function() {
                return {
                    name: this.name,
                    email: this.email
                }
            }, e.prototype.render = function() {
                return e.__super__.render.apply(this, arguments), this.$el.on("hide.bs.modal", _.bind(this.confirm_close, this))
            }, e.prototype.confirm_close = function() {
                return this.finished ? void 0 : confirm("Are you sure to abort registration?")
            }, e.prototype.attach = function() {
                return e.__super__.attach.apply(this, arguments)
            }, e.prototype.check_username_availability = function() {
                var a, b, d = this;
                return a = this.$("input[name='username']").val(), b = /^[a-zA-Z0-9]+(?:[_][a-zA-Z0-9]+)*$/, this.show_error(""), this.has_error=!1, a.length > 0 ? b.test(a) ? c.user.existsUser(a, function(b) {
                    return b.pkg ? d.show_error("Sorry, the username <b>" + a + "</b> has already been taken. Please choose another username.") : void 0
                }) : (this.has_error=!0, this.show_error("The username contains illegal characters. Only letters, numbers and underscores are allowed.")) : void 0
            }, e.prototype.validate_password = function() {
                var a;
                return a = this.$("input[type='password']").val(), this.show_error(""), this.has_error=!1, a.indexOf(" ") > 0 ? (this.has_error=!0, this.show_error("Sorry, space is not allowed in password.")) : void 0
            }, e.prototype.show_error = function(a) {
                return this.$(".error").html(a)
            }, e.prototype.submit_form = function(b) {
                var d, e, f = this;
                return null != b && "function" == typeof b.preventDefault && b.preventDefault(), this.has_error ? void 0 : (this.$(".join-button").prop("disabled", "disabled"), d = {
                    full_name: this.$("input[name='display-name']").val(),
                    display_name: this.$("input[name='display-name']").val(),
                    email: this.$("input[name='email']").val(),
                    username: this.$("input[name='username']").val(),
                    password: this.$("input[name='password']").val(),
                    recaptcha_challenge_field: this.$("input[name='recaptcha_challenge_field']").val(),
                    recaptcha_response_field: this.$("input[name='recaptcha_response_field']").val()
                }, e = c.user, e.set(d), e.register(d).done(function() {
                    return f.finished=!0, f.$el.modal("hide"), ga("send", "event", "signup", "create", f.name ? "thirdparty" : "direct"), c.user.loginSilent().done(function() {
                        return c.pending_save ? (ga("send", "event", "signup", "from-button"), f.publishEvent("user:login")) : a.helpers.redirectTo({
                            url: "/" + e.get("username")
                        })
                    })
                }).fail(function(a) {
                    return f.$(".join-button").prop("disabled", ""), "invalid_args" === a.status && f.show_error(a.error), "invalid_captcha" === a.status && (f.show_error("Captcha verification failed, please try again."), ga("send", "event", "signup", "captcha-failed")), f.$("input[name='recaptcha_response_field']").val("")
                }))
            }, e
        }(d)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/login", ["chaplin", "lib/session", "views/base/modal", "views/modals/forgot_password", "views/modals/create"], function(a, c, d, e, f) {
        var g, h;
        return g = function(d) {
            function g() {
                return h = g.__super__.constructor.apply(this, arguments)
            }
            return b(g, d), g.prototype.template = "modals/login", g.prototype.initialize = function(a) {
                return g.__super__.initialize.apply(this, arguments), this.redirect_on_login = null != a ? a.redirect_on_login : void 0
            }, g.prototype.events = {
                "click .login-button": "do_login",
                "submit form": "do_login",
                "click a[href='#forget-password']": "do_forgot_password",
                "click .thirdparty-login": "do_thirdparty_login",
                "click a[href='#signup']": "on_signup_clicked"
            }, g.prototype.do_login = function(a) {
                var b, d, e = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), d = this.$("#login-form input[name=username]").val(), b = this.$("#login-form input[name=password]").val(), this.$("button").prop("disabled", "disabled"), c.user.login({
                    username: d,
                    password: b
                }).done(function() {
                    return e.login_success()
                }).fail(function() {
                    return e.login_failed()
                })
            }, g.prototype.do_thirdparty_login = function(a) {
                var b, d = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), b = $(a.currentTarget).attr("data-service"), c.user.loginWithService(b).done(function() {
                    return d.login_success()
                }).fail(function(a) {
                    var b, c;
                    return a.newUser ? (d.close, c = a.displayName, b = a.email, new f({
                        name: c,
                        email: b
                    })) : void 0
                })
            }, g.prototype.login_success = function() {
                return this.close(), c.pending_save ? void this.publishEvent("user:login") : this.redirect_on_login ? a.helpers.redirectTo({
                    url: "/" + c.user.get("username")
                }) : void 0
            }, g.prototype.login_failed = function() {
                return this.$("button").prop("disabled", ""), this.$(".help-inline").text("Incorrect username or password.")
            }, g.prototype.do_forgot_password = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.close(), new e, ga("send", "event", "signin", "forgot-password")
            }, g.prototype.on_signup_clicked = function() {
                return this.close()
            }, g
        }(d)
    })
}.call(this), function() {
    define("helpers/login_deferred", ["chaplin", "views/modals/login"], function(a, b) {
        var c;
        return c = function() {
            function c() {
                this.deferred = new $.Deferred, this.subscribeEvent("login:success", this.resolve)
            }
            return _.extend(c.prototype, a.EventBroker), c.prototype.prompt = function() {
                return new b({
                    redirect_on_login: !1
                }), this.deferred
            }, c.prototype.resolve = function() {
                return this.deferred.resolve(), this.unsubscribeAllEvents()
            }, c
        }()
    })
}.call(this), function() {
    define("helpers/avatar_canvas", [], function() {
        var a, b, c, d, e;
        b = "//delicious-icons.s3.amazonaws.com/default-avatar-2.jpg", e = 200, d = 200;
        try {
            a = $("<canvas width=" + e + " height=" + d + "></canvas>")[0]
        } catch (f) {
            c = f, a = null
        }
        return function(f) {
            var g, h;
            if (!a)
                return b;
            g = _.map(_.first(f.split(" "), 2), function(a) {
                return a.charAt(0)
            }).join("").toUpperCase();
            try {
                return h = a.getContext("2d"), h.clearRect(0, 0, e, d), h.fillStyle = "#EEEEEE", h.fillRect(0, 0, e, d), h.font = "100 96px sans-serif", h.textAlign = "center", h.textBaseline = "middle", h.fillStyle = "#999999", h.fillText(g, e / 2, d / 2), a.toDataURL()
            } catch (i) {
                return c = i, b
            }
        }
    })
}.call(this), function() {
    var a = function(a, b) {
        return function() {
            return a.apply(b, arguments)
        }
    }, b = {}.hasOwnProperty, c = function(a, c) {
        function d() {
            this.constructor = a
        }
        for (var e in c)
            b.call(c, e) && (a[e] = c[e]);
        return d.prototype = c.prototype, a.prototype = new d, a.__super__ = c.prototype, a
    };
    define("models/user", ["lib/popup", "lib/avos", "lib/encode", "lib/obfuscate", "chaplin", "lib/session", "models/base/model", "models/tags", "lib/logger", "store", "helpers/login_deferred", "helpers/avatar_canvas"], function(b, d, e, f, g, h, i, j, k, l, m, n) {
        var o, p;
        return o = function(d) {
            function g() {
                return this.loginSuccess = a(this.loginSuccess, this), p = g.__super__.constructor.apply(this, arguments)
            }
            return c(g, d), g.prototype.defaults = {
                isLoggedIn: !1,
                public_bookmark_count: 0
            }, g.prototype.profile = function(a) {
                var b, c = this;
                return null == a && (a = {}), b = this.get("username"), b ? this.api({
                    url: "/account/public/profile/" + e(b)
                }).done(function(a) {
                    return null !== a.pkg ? c.set(c._userDataFromPackage(a.pkg)) : void 0
                }) : this
            }, g.prototype.clone = function() {
                var a;
                return a = g.__super__.clone.apply(this, arguments), a.set({
                    copy: !0
                }), a
            }, g.prototype.set = function(a) {
                return (null != a ? a.avatar_url : void 0) && /default-avatar/.test(a.avatar_url) && (a.avatar_url = n(a.full_name), a.is_default_avatar=!0), g.__super__.set.apply(this, arguments), this.isMe() && this.get("isLoggedIn")&&!this.get("copy") ? l.set("user", this) : void 0
            }, g.prototype.login = function(a) {
                var b, c, d;
                return d = a.username, c = a.password, b = {
                    success: this.loginSuccess
                }, $("html").hasClass("lt-ie10") ? (d = e(d), c = encodeURIComponent(encodeURIComponent(c)).replace(".", "%252E"), b.url = "/account/webloginpwd/" + d + "/" + c) : _.extend(b, {
                    url: "/account/login",
                    type: "POST",
                    contentType: "application/x-www-form-urlencoded; charset=UTF-8",
                    data: {
                        username: d,
                        password: f(c)
                    }
                }), this.api(b)
            }, g.prototype.loginSilent = function() {
                return this.api({
                    url: "/account/webloginhash/" + e(this.get("username")) + "/" + this.get("password_hash"),
                    type: "POST",
                    success: this.loginSuccess
                })
            }, g.prototype.loginSuccess = function(a) {
                var b;
                return b = this._userDataFromPackage(a.pkg), b.isLoggedIn=!0, this.set(b), this.publishEvent("login:success")
            }, g.prototype.loginWithService = function(a) {
                var c, d, e = this;
                return k.log("Attempting to connect to " + a), c = new $.Deferred, d = {
                    url: "/auth/signin/" + a,
                    crossWindowMessageName: "connect_to_delicious",
                    popupWindowName: "SigninWith" + a,
                    success: function(a) {
                        var b;
                        return a.username && a.password ? (e.set({
                            username: a.username,
                            password_hash: a.password
                        }), e.loginSilent().done(function() {
                            return c.resolve()
                        })) : (a.displayName = null != (b = a.displayName) ? b.replace(/^@/, "") : void 0, c.reject(a))
                    }
                }, ("facebook" === a || "google" === a) && (d = _.extend(d, {
                    width: 1e3,
                    height: 610
                })), b(d), c
            }, g.prototype.logout = function() {
                return l.remove("user"), $.cookie("avid") && $.removeCookie("avid", {
                    domain: ".delicious.com"
                }), this.clear(), this.set("isLoggedIn", !1, {
                    silent: !0
                }), this.publishEvent("login:logout")
            }, g.prototype.isMe = function() {
                var a;
                return (null != (a = h.user) ? a.get("username") : void 0) === this.get("username")
            }, g.prototype.existsUser = function(a, b) {
                return this.api({
                    url: "/account/public/profile/" + a,
                    success: function(a) {
                        return _.isFunction(b) ? b(a) : void 0
                    }
                })
            }, g.prototype.register = function() {
                var a = this;
                return this.api({
                    url: "/account/registerw",
                    data: this.toJSON(),
                    type: "POST",
                    success: function(b) {
                        return k.log("Created new user with pkg", b.pkg), a.set(b.pkg)
                    }
                })
            }, g.prototype.update = function(a) {
                var b = this;
                return this.api({
                    url: "/account/profile/edit",
                    data: a,
                    type: "POST",
                    success: function(a) {
                        return k.log("Successfully updated user's profile", a.pkg), b.set(a.pkg)
                    }
                })
            }, g.prototype.deactivate = function(a) {
                var b = this;
                return this.api({
                    url: "/account/deactivate",
                    data: a,
                    type: "POST",
                    success: function(a) {
                        return "success" === a.status ? b.logout() : void 0
                    }
                })
            }, g.prototype.changePassword = function(a) {
                return a.user_id = this.get("id"), this.api({
                    url: "/account/webchangepassword",
                    type: "POST",
                    data: a,
                    success: function(a) {
                        return k.log("Account password updated", a.pkg)
                    }
                })
            }, g.prototype.forgotPassword = function(a) {
                return this.api({
                    url: "/account/forgot_password/" + a,
                    type: "POST",
                    success: function(a) {
                        return "success" === a.status ? k.log("Password reset email sent. Response: ", a) : void 0
                    }
                })
            }, g.prototype.resetPassword = function(a) {
                return this.api({
                    url: "/account/reset_password",
                    type: "POST",
                    data: a,
                    success: function(a) {
                        return k.log("Account password has been reset", a.pkg)
                    }
                })
            }, g.prototype.resetAvatar = function() {
                var a = this;
                return this.api({
                    url: "/account/profile/remove_image",
                    type: "POST",
                    success: function() {
                        var b;
                        return b = n(a.get("full_name")), a.set({
                            avatar_url: b,
                            is_default_avatar: !0
                        })
                    }
                })
            }, g.prototype.updateAvatar = function(a) {
                var b = this;
                return this.api({
                    url: "/account/profile/edit_image",
                    type: "POST",
                    dataType: "json",
                    data: a,
                    processData: !1,
                    contentType: !1,
                    success: function(a) {
                        return _.delay(function() {
                            return k.log("Successfully uploaded profile avatar", {
                                user: a.pkg
                            }), a.pkg.avatar_url = a.pkg.avatar_url + "?" + (new Date).getTime(), a.pkg.is_default_avatar=!1, b.set(a.pkg)
                        }, 1500)
                    }
                })
            }, g.prototype.validateEmail = function(a, b) {
                return this.api({
                    url: "/account/validate_email/" + a,
                    success: function() {
                        return b({
                            success: !0
                        })
                    },
                    error: function(a) {
                        var c;
                        return c = "invalid_args" === a.status ? "Sorry, that url is invalid." : a.error, b({
                            success: !1,
                            error: c
                        })
                    }
                })
            }, g.prototype._userDataFromPackage = function(a) {
                var b, c, d, e, f;
                return b = a.account, f = _.extend({}, b || a), null != a.follows && (c = _.sortBy(a.follows, function(a) {
                    return a.username.toLowerCase()
                }), f.follows = _.map(c, function(a) {
                    return {
                        name: a.username,
                        username: a.username,
                        full_name: a.full_name
                    }
                })), null != a.user_tags && (d = new j, e = d.parse({
                    pkg: a.user_tags
                }), d.reset(e), f.tags = d.toJSON()), null != a.bundle_names && (f.tag_bundles = _.map(a.bundle_names, function(a) {
                    return {
                        name: a
                    }
                })), a.social && (f.social = a.social), f.password_hash || delete f.password_hash, f
            }, g.prototype.followUser = function(a) {
                var b = this;
                return this.get("isLoggedIn") ? this.api({
                    url: "/account/follow/" + a.get("username"),
                    type: "POST",
                    success: function() {
                        var c;
                        return a.set({
                            am_following_them: !0
                        }), c = _.chain(b.get("follows")).push(a.toJSON()).value(), b.set("follows", c)
                    }
                }) : (new m).prompt().done(_.bind(this.followUser, this, a))
            }, g.prototype.unfollowUser = function(a) {
                var b = this;
                return this.api({
                    url: "/account/unfollow/" + a.get("username"),
                    type: "POST",
                    success: function() {
                        var c;
                        return a.set({
                            am_following_them: !1
                        }), c = _.reject(b.get("follows"), function(b) {
                            return b.username === a.get("username")
                        }), b.set("follows", c)
                    }
                })
            }, g.prototype.getFollowersUrl = function() {
                return "/account/public/followers/" + e(this.get("username"))
            }, g.prototype.getFollowingUrl = function() {
                return "/account/public/follows/" + e(this.get("username"))
            }, g
        }(i)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("models/note", ["models/base/model", "helpers/avatar_canvas"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.set = function(a) {
                return a.avatar_url && /default-avatar/.test(a.avatar_url) && (a.avatar_url = c(a.full_name)), d.__super__.set.apply(this, arguments)
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("models/notes", ["models/base/collection", "models/note"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.model = c, d.prototype.url = function() {
                return _.defaults(this.urlParams, {
                    limit: 1
                }), this.urlParams.md5 ? "/posts/comments/time/" + this.urlParams.md5 + "?" + $.param({
                    limit: this.urlParams.limit,
                    anchor: this._anchor(),
                    exclude_yours: !1,
                    include_empty: !0
                }) : void 0
            }, d.prototype.fetch = function(a) {
                var b = this;
                return this.publishEvent("notes:load-start"), d.__super__.fetch.call(this, a).done(function() {
                    return b.publishEvent("notes:load-finish")
                })
            }, d.prototype.comparator = function(a) {
                return a.get("time_created")
            }, d
        }(a)
    })
}.call(this), function() {
    var a = function(a, b) {
        return function() {
            return a.apply(b, arguments)
        }
    }, b = {}.hasOwnProperty, c = function(a, c) {
        function d() {
            this.constructor = a
        }
        for (var e in c)
            b.call(c, e) && (a[e] = c[e]);
        return d.prototype = c.prototype, a.prototype = new d, a.__super__ = c.prototype, a
    };
    define("models/link", ["lib/logger", "chaplin", "lib/session", "models/base/model", "models/notes", "models/tag", "models/tags", "helpers/avatar_canvas"], function(b, d, e, f, g, h, i, j) {
        var k, l;
        return k = function(d) {
            function f() {
                return this.compose = a(this.compose, this), l = f.__super__.constructor.apply(this, arguments)
            }
            return c(f, d), f.prototype.set = function(a) {
                var b, c, d;
                return a.save_rank && a.num_saves && (a.num_saves = Math.max(a.save_rank, a.num_saves)), b = [a.first_saver, null != (c = a.savers) ? c[0]: void 0, null != (d = a.publishers) ? d[0] : void 0], _.each(b, function(a) {
                    return a ? (a.avatar && (a.avatar_url = a.avatar), a.avatar_url && /default-avatar/.test(a.avatar_url) && (a.full_name || a.text) ? a.avatar_url = j(a.full_name || a.text) : void 0) : void 0
                }), f.__super__.set.call(this, a)
            }, f.prototype.addOrEdit = function(a) {
                var c = this;
                return this.api({
                    data: a,
                    url: "/posts/addoredit",
                    type: "POST",
                    success: function(a) {
                        var d, f, g;
                        return b.log("Successfully saved link"), f = c.get("savers") || [], d = e.user.toJSON(), c.set("savers", [d].concat(f)), a.pkg && ((g = a.pkg[0]).note || (g.note = null), c.set(a.pkg[0])), c.trigger("save")
                    }
                })
            }, f.prototype.bulkEdit = function(a) {
                return this.api({
                    data: a,
                    url: "/tag/edit/all"
                })
            }, f.prototype.compose = function() {
                var a = this;
                return this.api({
                    data: {
                        url: this.get("url")
                    },
                    url: "/posts/compose",
                    success: function(b) {
                        return null == b.pkg || (a.set({
                            recommended: b.pkg
                        }), b.pkg.previously_saved!==!0 || (a.get("title") || a.set({
                            title: b.pkg.suggested_title
                        }), a.get("tags") || a.set({
                            tags: b.pkg.previous_tags
                        }), a.get("note") || a.set({
                            note: b.pkg.note
                        }), a.get("md5"))) ? void 0 : a.set({
                            md5: b.pkg.md5
                        })
                    }
                })
            }, f.prototype.bulk_size = 60, f.prototype.deleteAll = function(a) {
                var b = this;
                return this.api({
                    data: {
                        md5s: a
                    },
                    url: "/links/delete",
                    type: "POST",
                    success: function() {
                        return b.publishEvent("links:deleted", a.length)
                    }
                })
            }, f.prototype.markAs = function(a, b) {
                return this.api({
                    data: {
                        md5s: b
                    },
                    type: "POST",
                    url: "/links/make/" + a
                })
            }, f.prototype.notes = function() {
                return new g([], {
                    urlParams: {
                        md5: this.get("md5")
                    }
                })
            }, f.prototype.tags = function() {
                return _.uniq(_.map(this.get("tags"), function(a) {
                    return h.normalize(a)
                }))
            }, f.prototype.relatedTags = function(a) {
                return new i([], {
                    urlParams: {
                        type: "related",
                        tags: a
                    }
                })
            }, f.prototype.remove = function() {
                var a = this;
                return this.api({
                    data: {
                        md5: this.get("md5")
                    },
                    type: "POST",
                    url: "/posts/delete",
                    success: function() {
                        return a.destroy(), a.publishEvent("links:deleted", 1)
                    }
                })
            }, f
        }(f)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    }, c = [].indexOf || function(a) {
        for (var b = 0, c = this.length; c > b; b++)
            if (b in this && this[b] === a)
                return b;
        return - 1
    };
    define("views/sidebar", ["views/base/hbsview", "models/user", "models/link", "chaplin", "lib/session", "store", "helpers/login_deferred"], function(a, d, e, f, g, h, i) {
        var j, k;
        return j = function(a) {
            function d() {
                return k = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "sidebar", d.prototype.autoRender=!0, d.prototype.autoAttach=!1, d.prototype.container = "#delicious", d.prototype.containerMethod = "prepend", d.prototype.className = "sidebar", d.prototype.last_sidebar_page_update_event = null, d.prototype.events = {
                "submit #location-bar": "goto",
                "click a.site-nav-user-link": "trigger_appcache_update",
                "click ul.site-nav-links a": "sidebar_click_check",
                "click div.site-nav-toggle a": "toggle_mobile_nav",
                "click ul.site-nav-links.on a": "remove_mobile_nav",
                "click a.site-nav-user-link": "remove_mobile_nav",
                "focus #site-search": "on_focus_search",
                "keyup #site-search": "on_search_keyup",
                "keydown #site-search": "on_search_keydown",
                "blur #site-search": "on_blur_search",
                "click ul.search-bar-tip li.suggestion": "on_click_search_suggestion",
                "click a[href='#login']": "track_login"
            }, d.prototype.initialize = function(a) {
                return this.subscribeEvent("sidebar:change", this.update_sidebar), this.subscribeEvent("login:success", this.render), this.subscribeEvent("login:logout", this.render), d.__super__.initialize.call(this, a)
            }, d.prototype.render = function() {
                return d.__super__.render.apply(this, arguments), this.last_sidebar_page_update_event ? this.update_sidebar(this.last_sidebar_page_update_event) : void 0
            }, d.prototype.attach = function() {
                return d.__super__.attach.apply(this, arguments), this.hide()
            }, d.prototype.trigger_appcache_update = function() {
                return window.applicationCache && window.applicationCache.status > 0 ? window.applicationCache.update() : void 0
            }, d.prototype.update_sidebar = function(a) {
                var b;
                switch (a.action) {
                case"show":
                    return this.show();
                case"hide":
                    return this.hide();
                case"page":
                    return this.last_sidebar_page_update_event = a, this.$(".site-nav-links li").removeClass("on"), this.$(".site-nav-search").removeClass("on"), this.$(".site-nav-search form").attr("data-icon", ""), this.$("#site-search").val(""), /^[@#]/.test(a.location) ? (this.$("#site-search").val(a.location), this.$(".site-nav-search").addClass("on"), this.$(".site-nav-search form").attr("data-icon", "")) : /^".*?"$/.test(a.location) ? (b = a.location.length, this.$("#site-search").val(a.location.substring(1, b - 1)), this.$(".site-nav-search").addClass("on"), this.$(".site-nav-search form").attr("data-icon", "")) : this.$(".nav-item-" + a.location).addClass("on")
                }
            }, d.prototype.break_search_tokens = function(a) {
                return a.match(/#?"[^"]*?"|[#@\S]+/g)
            }, d.prototype.goto = function(a) {
                var b, d, h, i, j, k, l, m;
                return null != a && a.preventDefault(), k = this.$("#site-search").val(), /^https?:\/\//.test(k) ? (b = new e({
                    url: k
                }), void b.compose().done(function() {
                    var a;
                    return a = b.get("recommended").md5, a ? f.helpers.redirectTo({
                        url: "/link/" + a
                    }) : void 0
                })) : (l = _.filter(this.break_search_tokens(k), function(a) {
                    return a
                }), m = _.find(l, function(a) {
                    return "@" === a.charAt(0)
                }), j = _.filter(l, function(a) {
                    return "#" === a.charAt(0)
                }), i = _.find(l, function(a) {
                    return /^\[.*\]$/.test(a)
                }), h = _.reject(l, function(a) {
                    return a === m || c.call(j, a) >= 0 || a === i
                }), m && (m = m.substring(1)), g.user && "me" === m && (m = g.user.get("username")), _.isEmpty(j) || (j = _.map(j, function(a) {
                    return encodeURIComponent(a.substring(1).replace(/"/g, ""))
                })), _.isEmpty(i) || (i = i.substring(1, i.length - 1)), _.isEmpty(h) || (h = _.map(h, function(a) {
                    return encodeURIComponent(a)
                })), d = "", m && (d += "/" + m), _.isEmpty(j) ? i && m && (d += "/tag_bundle/" + i) : d += m ? "/" + j.join(",") : "/tag/" + j[0], !(_.isEmpty(h) ||!m&&!_.isEmpty(j) ||!(d += "/search/" + h.join(","))), ga("send", "event", "sidebar", "search", k), _.isEmpty(d) ? void 0 : f.helpers.redirectTo({
                    url: d
                }))
            }, d.prototype.toggle_mobile_nav = function(a) {
                return a.preventDefault(), this.$(".site-nav-links").toggleClass("on")
            }, d.prototype.remove_mobile_nav = function(a) {
                return a.preventDefault(), this.$(".site-nav-links").removeClass("on")
            }, d.prototype.on_focus_search = function(a) {
                return this.on_search_keyup(a), this.$(".site-nav-search").addClass("focus")
            }, d.prototype.on_search_keydown = function(a) {
                var b;
                return b = a.which, 9 === b?!1 : 13 === b && this.$("li.suggestion-noac.selected").length > 0 ? (this.$("li.suggestion-noac.selected").click(), !1) : void 0
            }, d.prototype._update_text_on_cursor = function(a, b) {
                return a.hasClass("suggestion-noac") ? a.hasClass("suggestion-mine") || a.hasClass("suggestion-someone") ? this.$("#site-search").val(b.trim().replace(/(@.*?\s)?(.*)$/, "$1" + $("b", a).text())) : a.hasClass("suggestion-all") || a.hasClass("suggestion-user") ? this.$("#site-search").val($("b", a).text()) : void 0 : this.$("#site-search").val(b.trim().replace(/(.*\s)*(.*)$/, "$1" + a.text()))
            }, d.prototype.on_search_keyup = function(a) {
                var b, c, d, e, f, h, i, j, k, l, m, n, o, p, q, r, s;
                return this.$(".search-bar-tip").show(), n = this.$("#site-search").val(), e = a.which, 40 === e || 9 === e ? (d = this.$(".search-bar-tip li.selected").removeClass("selected").nextAll(".suggestion").first(), d.length > 0 ? d.addClass("selected") : d = this.$(".search-bar-tip li.suggestion").first().addClass("selected"), this._update_text_on_cursor(d, n), void 0) : 38 === e ? (d = this.$(".search-bar-tip li.selected").removeClass("selected").prevAll(".suggestion").first(), d.length > 0 ? d.addClass("selected") : d = this.$(".search-bar-tip li.suggestion").last().addClass("selected"), this._update_text_on_cursor(d, n), void 0) : (27 === e && (this.$(".site-nav-search input").blur(), this.on_blur_search()), this.$(".list-group-item.suggestion").remove(), h = [], /^https?:\/\//.test(n) ? h.push("<li class='list-group-item suggestion suggestion-noac suggestion-all'>Show this link on Delicious</li>") : n && (k = this.break_search_tokens(n), j = _.last(k), "@" === j.charAt(0) ? (c = _.clone(null != (o = g.user) ? o.get("follows") : void 0) || [], (null != (p = g.user) ? p.get("isLoggedIn") : void 0) && c.push({
                    name: g.user.get("username")
                }), b = _.filter(c, function(a) {
                    return 0 === a.name.indexOf(j.substring(1))
                }), h = _.map(b, function(a) {
                    return "<li class='list-group-item suggestion'>@" + _.escape(a.name) + "</li>"
                })) : "#" === j.charAt(0) && (i = null != (q = g.user) ? q.get("tags") : void 0, b = _.filter(i, function(a) {
                    return 0 === a.name.toLowerCase().indexOf(j.substring(1).toLowerCase())
                }), h = _.map(b, function(a) {
                    return "<li class='list-group-item suggestion'>#" + _.escape(a.name) + "</li>"
                })), (null != (r = g.user) ? r.get("isLoggedIn") : void 0) && _.all(k, function(a) {
                    return "@" !== a.charAt(0)
                }) && (h.push("<li class='list-group-item suggestion suggestion-noac suggestion-mine'>Search <b>" + n + "</b> in my links</li>"), h.push("<li class='list-group-item suggestion suggestion-noac suggestion-all'>Search <b>" + n + "</b> in Delicious</li>")), (null != (s = g.user) ? s.get("isLoggedIn") : void 0) && k.indexOf("@" + g.user.get("username"))>-1 ? (f = n.replace("@" + g.user.get("username"), "").trim(), f.length > 0 && h.push("<li class='list-group-item suggestion suggestion-noac suggestion-mine'>Search <b>" + f + "</b> in my links</li>")) : _.any(k, function(a) {
                    return "@" === a.charAt(0)
                }) && (k = this.break_search_tokens(n.trim()), _.every(k, function(a) {
                    return "@" === a.charAt(0)
                }) ? (l = _.find(k, function(a) {
                    return "@" === a.charAt(0)
                }).trim().substring(1), g.user && "me" === l && (l = g.user.get("username")), h.push("<li class='list-group-item suggestion suggestion-noac suggestion-user'>Go to <b>@" + l + "</b>&rsquo;s profile</li>")) : (l = _.find(k, function(a) {
                    return "@" === a.charAt(0)
                }).trim().substring(1), g.user && "me" === l && (m = g.user.get("username")), f = n.replace("@" + l, "").trim(), f.length > 0 && h.push("<li class='list-group-item suggestion suggestion-noac suggestion-someone'>Search <b>" + f + "</b> in @" + m + "&rsquo;s links</li>")))), this.$(".search-bar-tip").prepend(h.join("")), n.length > 0 && n.trim() !== "@" + g.user.get("username") ? this.$(".list-group-item.help").hide() : this.$(".list-group-item.help").show())
            }, d.prototype.on_click_search_suggestion = function(a) {
                var b, c, d, e, f, h;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), b = $(a.currentTarget), c = b.text(), f = this.$("#site-search").val(), b.hasClass("suggestion-mine") ? (f = f.replace(/@.+?\s/, ""), e = null != (h = g.user) ? h.get("username") : void 0, this.$("#site-search").val("@" + e + " " + f), this.$("#location-bar").submit()) : b.hasClass("suggestion-all") || b.hasClass("suggestion-someone") || b.hasClass("suggestion-user") ? this.$("#location-bar").submit() : (d = _.filter(f.split(" "), function(a) {
                    return a
                }), d.length > 0 ? d[d.length - 1] = c : d = [c], this.$("#site-search").val(d.join(" ") + " ")), this.$(".list-group-item.suggestion").remove()
            }, d.prototype.on_blur_search = function() {
                var a = this;
                return _.delay(function() {
                    return a.$(".search-bar-tip").fadeOut(200)
                }, 200), this.$(".site-nav-search").removeClass("focus")
            }, d.prototype.sidebar_click_check = function(a) {
                var b, c;
                return a.preventDefault(), b = $(a.currentTarget), c = b.attr("href"), 0 !== c.indexOf("#") ? (b.hasClass("login-check")&&!g.user.get("isLoggedIn") ? (new i).prompt().done(function() {
                    return f.helpers.redirectTo({
                        url: c
                    })
                }) : f.helpers.redirectTo({
                    url: c
                }, {
                    forceStartup: !0
                }), !1) : void 0
            }, d.prototype.track_login = function() {
                return ga("send", "event", "sidebar", "login")
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/notification", ["views/base/hbsview", "store"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "notification", d.prototype.autoRender=!0, d.prototype.autoAttach=!0, d.prototype.initialize = function(a) {
                return a || (a = []), this.message = a.message, this.id = a.id, this.type = a.type || "news", this.name = a.name, this.parent = a.parent, this.subscribeEvent("page:disposed", this.do_check_dispose), "news" === this.type && this.subscribeEvent("page:loaded", this.attach), d.__super__.initialize.apply(this, arguments)
            }, d.prototype.do_check_dispose = function() {
                return "notification" === this.type ? this.dispose() : void 0
            }, d.prototype.events = {
                "click a": "on_close",
                "closed.bs.alert .alert": "on_close"
            }, d.prototype.getTemplateData = function() {
                return {
                    message: this.message,
                    type: this.type,
                    name: this.name
                }
            }, d.prototype.on_close = function() {
                var a = this;
                return _.delay(function() {
                    return a.id && c.set(a.id, !0), a.dispose()
                }, 1e3), !0
            }, d.prototype.render = function() {
                return d.__super__.render.apply(this, arguments), this.$el.addClass(this.type)
            }, d.prototype.attach = function() {
                return this.parent ? this.parent.prepend(this.$el) : $(".content-wrapper").prepend(this.$el)
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/signup", ["chaplin", "lib/session", "views/base/modal", "views/modals/create", "views/modals/login"], function(a, c, d, e, f) {
        var g, h;
        return g = function(d) {
            function g() {
                return h = g.__super__.constructor.apply(this, arguments)
            }
            return b(g, d), g.prototype.template = "modals/signup", g.prototype.initialize = function() {
                return g.__super__.initialize.apply(this, arguments)
            }, g.prototype.events = {
                "click .create-account": "open_create_dialog",
                "click .thirdparty-login": "do_thirdparty_signup",
                "click .switch-signin": "switch_to_signin"
            }, g.prototype.open_create_dialog = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.close(), new e({
                    name: null != a ? a.name: void 0,
                    email: null != a ? a.email: void 0
                })
            }, g.prototype.switch_to_signin = function(a) {
                return a.preventDefault(), this.close(), new f({
                    redirect_on_login: !0
                })
            }, g.prototype.login_failed = function(a) {
                var b, c;
                return a.newUser ? (c = a.displayName, b = a.email, this.open_create_dialog({
                    name: c,
                    email: b
                })) : void 0
            }, g.prototype.do_thirdparty_signup = function(b) {
                var d, e = this;
                return null != b && "function" == typeof b.preventDefault && b.preventDefault(), d = $(b.currentTarget).attr("data-service"), c.user.loginWithService(d).done(function() {
                    return a.helpers.redirectTo({
                        url: "/" + c.user.get("username")
                    }), e.close()
                }).fail(function(a) {
                    return e.login_failed(a)
                })
            }, g
        }(d)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/help", ["views/base/modal", "lib/session"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "modals/help", d.prototype.initialize = function() {
                return d.__super__.initialize.apply(this, arguments), c.help_modal=!0
            }, d.prototype.dispose = function() {
                return d.__super__.dispose.apply(this, arguments), c.help_modal=!1
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    }, c = [].indexOf || function(a) {
        for (var b = 0, c = this.length; c > b; b++)
            if (b in this && this[b] === a)
                return b;
        return - 1
    };
    define("models/connect", ["chaplin", "lib/session", "models/base/model", "lib/popup"], function(a, d, e, f) {
        var g, h;
        return g = function(a) {
            function e() {
                return h = e.__super__.constructor.apply(this, arguments)
            }
            return b(e, a), e.prototype.defaults = {
                privacy_options: [{
                    value: "ALL_PUBLIC",
                    text: "make 'em all public"
                }, {
                    value: "ALL_PRIVATE",
                    text: "make 'em all private"
                }, {
                    value: "AUTOMATIC",
                    text: "use the same privacy"
                }
                ]
            }, e.prototype.initialize = function() {
                return e.__super__.initialize.apply(this, arguments)
            }, e.prototype.connect = function(a, b) {
                var e, g = this;
                return e = {
                    url: "/connect/to/" + this.get("providerId") + "?active=" + a,
                    crossWindowMessageName: "connect_to_delicious",
                    popupWindowName: "Connect_To_" + this.get("providerId") + "_From_Delicious",
                    success: function(a) {
                        var e, f;
                        return e = d.user.get("social"), f = g.get("providerId").toUpperCase(), c.call(e, f) < 0 && (e.push(g.get("providerId").toUpperCase()), d.user.set({
                            social: e
                        })), _.isFunction(b) ? b(a) : void 0
                    }
                }, "facebook" === this.get("providerId") && (e = _.extend(e, {
                    width: 1e3,
                    height: 610
                })), f(e)
            }, e.prototype.disconnect = function() {
                var a = this;
                return this.api({
                    url: "/connect/delete/" + this.get("providerId"),
                    type: "POST",
                    success: function() {
                        var b, c;
                        return null != (c = a.collection) && c.remove(a), b = d.user.get("social"), b = _.reject(b, function(b) {
                            return b === a.get("providerId").toUpperCase()
                        }), d.user.set({
                            social: b
                        }), a.dispose()
                    }
                })
            }, e.prototype.update = function(a, b) {
                var c = this;
                return this.api({
                    url: "/connect/update/" + this.get("providerId") + "/" + this.get("providerUserId"),
                    type: "POST",
                    data: a,
                    success: function() {
                        return _.isFunction(b) ? b(c) : void 0
                    }
                })
            }, e
        }(e)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    }, c = [].indexOf || function(a) {
        for (var b = 0, c = this.length; c > b; b++)
            if (b in this && this[b] === a)
                return b;
        return - 1
    };
    define("views/widgets/tag_input", ["views/base/hbsview"], function(a) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "widgets/tag_input", d.prototype.autoRender=!0, d.prototype.autoAttach=!0, d.prototype.initialize = function(a) {
                return d.__super__.initialize.call(this, a), this.name = a.name, this.tags = a.tags, this.delimeter = ",", this.value = a.value, this.multiline = a.multiline ||!1, this.on_change = a.on_change || function() {}
            }, d.prototype.events = {
                "keyup .tag-input": "on_input_up",
                "keydown .tag-input": "on_input_down",
                "blur .tag-input": "on_input_blur",
                "click .tag-input-candidates li": "on_click_candidate_item",
                "mouseover .tag-input-candidates li": "on_hover_candidate_item",
                "change .tag-input": "on_input_change"
            }, d.prototype.on_input_change = function(a) {
                return this.on_change(a)
            }, d.prototype.on_hover_candidate_item = function(a) {
                return this.$(".tag-input-candidates li.on").removeClass("selected"), $(a.currentTarget).addClass("selected")
            }, d.prototype.on_click_candidate_item = function(a) {
                var b;
                return b = $(a.currentTarget), this.set_last_tag(b.text())
            }, d.prototype.on_input_blur = function() {
                var a = this;
                return setTimeout(function() {
                    return a.$(".tag-input-candidates li.on").removeClass("on"), a.$(".tag-input-candidates").hide()
                }, 200)
            }, d.prototype.on_input_down = function(a) {
                var b, c, d, e, f;
                return e = a.which, 9 === e && (c = this.$(".tag-input-candidates li.on"), c.length > 1 ? (b=!0, e = a.shiftKey ? 38 : 40) : 1 === c.length && (c.addClass("selected"), e = 13)), 13 === e ? (f = this.$(".tag-input-candidates li.selected").text(), f ? (this.set_last_tag(f), this.update_candidate_list(), !1) : !0) : 40 === e ? (d = this.$(".tag-input-candidates li.selected").removeClass("selected").nextAll(".on").first(), d.length > 0 ? d.addClass("selected") : d = this.$(".tag-input-candidates li.on").first().addClass("selected"), b && this.set_last_tag(d.text(), !0), this.scroll_to_selection(d), !1) : 38 === e ? (d = this.$(".tag-input-candidates li.selected").removeClass("selected").prevAll(".on").first(), d.length > 0 ? d.addClass("selected") : d = this.$(".tag-input-candidates li.on").last().addClass("selected"), b && this.set_last_tag(d.text(), !0), this.scroll_to_selection(d), !1) : this.$(".tag-input-candidates").show()
            }, d.prototype.on_input_up = function(a) {
                var b;
                return b = a.which, 9 === b || 13 === b || 38 === b || 40 === b?!1 : 188 === b ? void this.$(".tag-input-candidates li").removeClass("on") : (this.update_candidate_list(), this.on_change())
            }, d.prototype.update_candidate_list = function() {
                var a, b;
                return b = this.get_current_editing(), a = _.map(this.get_value_array(), function(a) {
                    return a.toLowerCase().trim()
                }).slice(0, - 1), this.$(".tag-input-candidates li").removeClass("selected on"), b ? _.each(this.$(".tag-input-candidates li"), function(d) {
                    var e, f;
                    return e = $(d), f = e.text().toLowerCase(), f.indexOf(b.toLowerCase())>-1 && c.call(a, f) < 0 ? e.addClass("on") : void 0
                }) : void 0
            }, d.prototype.scroll_to_selection = function(a) {
                var b, c, d;
                return b = this.$(".tag-input-candidates"), d = a.offset().top, c = b.offset().top, b.scrollTop(d - c + b.scrollTop())
            }, d.prototype.getTemplateData = function() {
                var a;
                return {
                    tags: this.tags,
                    name: this.name,
                    value: null != (a = this.value) ? a.join(", "): void 0,
                    multiline: this.multiline
                }
            }, d.prototype.get_values = function() {
                return this.$(".tag-input").val()
            }, d.prototype.get_value_array = function() {
                return this.get_values().split(this.delimeter)
            }, d.prototype.set_values = function(a) {
                var b;
                return b = _.isArray(a) ? a.join(this.delimeter + " ") : a, this.$(".tag-input").val(b)
            }, d.prototype.get_current_editing = function() {
                var a;
                return a = this.get_value_array(), a[a.length - 1].trim()
            }, d.prototype.set_last_tag = function(a, b) {
                var c;
                return c = this.get_value_array(), a ? c[c.length - 1] = b ? c[c.length - 1].replace(/\S(.*)$/, "" + a) : c[c.length - 1].replace(/\S(.*)$/, "" + a + ", ") : c = c.slice(0, c.length - 1), this.$(".tag-input").val(c.join(this.delimeter))
            }, d.prototype.append_value = function(a) {
                var b;
                return c.call(this.get_value_array(), a) < 0 && (b = this.get_values(), !b.trim() || /,\s*$/.test(b) || (b += ", ")), this.set_values(b + a + ", ")
            }, d.prototype.remove_value = function(a) {
                var b;
                return b = this.get_value_array(), b = _.reject(b, function(b) {
                    return b.trim() === a
                }), this.set_values(b.join(this.delimeter))
            }, d
        }(a)
    })
}.call(this), function() {
    var a = function(a, b) {
        return function() {
            return a.apply(b, arguments)
        }
    }, b = {}.hasOwnProperty, c = function(a, c) {
        function d() {
            this.constructor = a
        }
        for (var e in c)
            b.call(c, e) && (a[e] = c[e]);
        return d.prototype = c.prototype, a.prototype = new d, a.__super__ = c.prototype, a
    }, d = [].indexOf || function(a) {
        for (var b = 0, c = this.length; c > b; b++)
            if (b in this && this[b] === a)
                return b;
        return - 1
    };
    define("models/links", ["models/base/collection", "chaplin", "moment", "lib/session", "lib/encode", "models/link", "models/tags", "models/tag"], function(b, e, f, g, h, i, j) {
        var k, l;
        return k = function(b) {
            function e() {
                return this.reFetchFromSearch = a(this.reFetchFromSearch, this), l = e.__super__.constructor.apply(this, arguments)
            }
            return c(e, b), e.prototype.model = i, e.prototype.initialize = function(a, b) {
                return e.__super__.initialize.apply(this, arguments), (null != b ? b.subscribeToFetch : void 0) ? this.subscribeEvent("search:fetch", this.reFetchFromSearch) : (null != b ? b.subscribeOnceToFetch : void 0) ? this.subscribeEvent("search:fetch", _.once(this.reFetchFromSearch)) : void 0
            }, e.prototype.reFetchFromSearch = function(a) {
                return this.urlParams = _.extend(this.urlParams, a), this.reset([], {
                    silent: !0
                }), this.fetch()
            }, e.prototype.parse = function(a) {
                var b, c, d, f, g = this;
                return d = e.__super__.parse.call(this, a), f = _.compact(_.uniq(_.pluck(d, "md5"))), b = this.pluck("md5"), c = _.difference(f, b), d = _.filter(d, function(a) {
                    return _.contains(c, a.md5)
                }), _.each(d, function(a) {
                    return a.tags ? void 0 : a.tags = a.best_tags
                }), _.each(d, function(a) {
                    return g.hasLegalImage(a) ? void 0 : a.chosen_image = null
                }), this.urlParams.hasImage && (d = _.filter(d, function(a) {
                    return g.hasLegalImage(a)
                })), d
            }, e.prototype.hasLegalImage = function(a) {
                var b, c;
                return (null != (b = a.chosen_image) ? b.width : void 0) >= 300 || (null != (c = a.chosen_image) ? c.height : void 0) >= 200
            }, e.prototype.time_to_anchorx = function(a) {
                return "" + a + "ffffffffffffffffffffffffffffffff"
            }, e.prototype.url = function() {
                var a, b, c, d, e, f, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w;
                return this.urlParamDefaults || (this.urlParamDefaults = {
                    type: "popular",
                    tags: [],
                    keywords: [],
                    limit: 20,
                    username: null,
                    tagBundle: null,
                    md5: null,
                    hashtag: null,
                    sortBy: null,
                    domain: null,
                    notag: !1,
                    visibility: null
                }), k = _.extend(this.urlParamDefaults, this.urlParams), r = k.type, p = k.tags, f = k.limit, e = k.keywords, s = k.username, o = k.tagBundle, i = k.md5, b = k.hasImage, c = k.hashtag, n = k.sortBy, a = k.domain, j = k.noTag, t = k.visibility, q = k.timeStart, q && (q = this.time_to_anchorx(q), this.urlParams.timeStart = null), _.isArray(p) && p.length > 0 && (p = _.map(p, function(a) {
                    return /%/.test(a) ? decodeURIComponent(a) : a
                })), s && ("network" === r ? s === (null != (u = g.user) ? u.get("username") : void 0) || "me" === s ? r = "network" : (s = h(s), r = "publicnetwork/" + s) : s === (null != (v = g.user) ? v.get("username") : void 0) || "me" === s ? r = "you" : (s = h(s), r = "public/" + s)), m = "/posts/" + r + "/time", l = {
                    tags: _.isArray(p) ? p.join(","): p,
                    keywords: _.isArray(e) ? e.join(","): e,
                    tagsor: !1,
                    limit: f,
                    anchorx: q || this._anchorx(),
                    index: this._index(),
                    inclpriv: 1,
                    bundle_name: decodeURIComponent(o) || null,
                    has_all: !0,
                    exclude_zeen: !0,
                    visibility: t
                }, j && (l.notag = j), s && o && (m = "/tags/bundle/links", s !== (null != (w = g.user) ? w.get("username") : void 0) && (l.username = s)), i && (m = "/posts/md5/" + i), c && (m = "/tags/timeline/" + n, l = {
                    limit: f,
                    tag_name: c,
                    score: this._score()
                }), a && (m = "/posts/" + r + "/domains", l.domain = a), "discover" === r && (m = "/recommend/discover", l = {
                    limit: f,
                    last_score: this._score()
                }), "related" === r && (m = "/posts/related/links", l = {
                    limit: f,
                    md5: i
                }), d = p.length || e.length || s || o || i || a, "everyone" === r&&!d && (r = "recommend", this.length < 1 ? delete l.anchor : l.anchor = this.last().get("md5"), m = "/recommend/links"), m + "?" + $.param(l)
            }, e.prototype.fetch = function(a) {
                var b = this;
                return this.publishEvent("links:load-start"), a || (a = {}), a.success = function() {
                    return b.publishEvent("links:load-finish")
                }, e.__super__.fetch.call(this, a)
            }, e.prototype.related_tags = function() {
                var a;
                return a = _.uniq(_.compact(_.flatten(this.pluck("tags")))), a = _.difference(a, this.urlParams.tags), new j(_.map(a, function(a) {
                    return {
                        name: a
                    }
                }))
            }, e.prototype.add_tag = function(a) {
                var b = this;
                return this.api({
                    url: "/links/edit/tags",
                    type: "POST",
                    data: {
                        tags: a.join(","),
                        md5s: this.map(function(a) {
                            return a.get("md5")
                        }).join(","),
                        action: "add"
                    },
                    success: function() {
                        return b.each(function(b) {
                            var c;
                            return c = b.get("tags"), c = _.union(c, a), b.set({
                                tags: c
                            })
                        })
                    }
                })
            }, e.prototype.remove_tag = function(a) {
                var b = this;
                return this.api({
                    url: "/links/edit/tags",
                    type: "POST",
                    data: {
                        tags: a.join(","),
                        action: "remove",
                        md5s: this.map(function(a) {
                            return a.get("md5")
                        }).join(",")
                    },
                    success: function() {
                        return b.each(function(b) {
                            var c;
                            return c = b.get("tags"), c = _.reject(c, function(b) {
                                return d.call(a, b) >= 0
                            }), b.set({
                                tags: c
                            })
                        })
                    }
                })
            }, e.prototype.to_search_string = function() {
                var a, b;
                return a = [], this.urlParams.username && a.push("@" + this.urlParams.username), b = this.urlParams.tags || [this.urlParams.hashtag], _.isEmpty(b) || _.each(b, function(b) {
                    return d.call(b, " ") >= 0 && (b = '"' + b + '"'), a.push("#" + b)
                }), this.urlParams.tagBundle && a.push("[" + this.urlParams.tagBundle + "]"), this.urlParams.keywords && a.push("" + this.urlParams.keywords), a.join(" ")
            }, e.prototype.to_user_url = function() {
                var a, b, c, d;
                return c = "", c += "/" + this.urlParams.username, _.isEmpty(this.urlParams.tags) ? _.isEmpty(this.urlParams.tagBundle) || (c += "/tag_bundle/" + this.urlParams.tagBundle) : (d = _.map(this.urlParams.tags, function(a) {
                    return encodeURIComponent(a)
                }), c += "/" + d.join(",")), _.isEmpty(this.urlParams.keywords) || (b = encodeURIComponent(this.urlParams.keywords), c += "/search/" + b), this.urlParams.timeStart && (a = f.unix(this.urlParams.timeStart).format("YYYYMM"), c += "/since/" + a), c
            }, e.prototype.getKeywords = function() {
                return _(this.first(5)).map(function(a) {
                    var b;
                    return _.isArray(a.get("tags")) ? a.get("tags") : null != (b = a.get("tags")) ? b.split(",") : void 0
                }).flatten().filter(function(a) {
                    return a
                }).countBy().pairs().sortBy(function(a) {
                    return - a[1]
                }).map(function(a) {
                    return a[0]
                }).first(5).value()
            }, e
        }(b)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/widgets/compose_result", ["views/base/hbsview", "models/links"], function(a, c) {
        var d, e, f, g;
        return d = function(a) {
            function c() {
                return f = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "widgets/compose_result_related_links", c.prototype.autoRender=!0, c.prototype.autoAttach=!0, c
        }(a), e = function(a) {
            function e() {
                return g = e.__super__.constructor.apply(this, arguments)
            }
            return b(e, a), e.prototype.template = "widgets/compose_result", e.prototype.autoRender=!0, e.prototype.autoAttach=!0, e.prototype.container = "#delicious", e.prototype.dispose_timer = null, e.prototype.events = {
                "click div.compose-result-backdrop": "dispose",
                mouseover: "clear_dispose_timer",
                mouseout: "start_dispose_timer",
                "click a.first_saver": "track_first_saver_click"
            }, e.prototype.initialize = function(a) {
                return e.__super__.initialize.apply(this, arguments), this.finish_callback = a.finish_callback || null, this.start_dispose_timer()
            }, e.prototype.attach = function() {
                var a;
                return e.__super__.attach.apply(this, arguments), a = this.$(".compose-result"), a.css("margin-left", - (a.width() / 2)), this.fetch_and_show_related_items()
            }, e.prototype.dispose = function() {
                return e.__super__.dispose.apply(this, arguments), _.isFunction(this.finish_callback) ? this.finish_callback() : void 0
            }, e.prototype.clear_dispose_timer = function() {
                return this.dispose_timer ? (clearTimeout(this.dispose_timer), this.dispose_timer = null) : void 0
            }, e.prototype.start_dispose_timer = function() {
                var a = this;
                return this.dispose_timer ? void 0 : this.dispose_timer = _.delay(function() {
                    return a.dispose()
                }, 6e3)
            }, e.prototype.track_first_saver_click = function() {
                return ga("send", "event", "first-saver", "from-dialog")
            }, e.prototype.fetch_and_show_related_items = function() {
                var a, b, e = this;
                return b = this.model.get("md5"), a = new c([], {
                    urlParams: {
                        limit: 4,
                        md5: b,
                        type: "related"
                    }
                }), a.fetch().done(function() {
                    return e.subview("related-links", new d({
                        collection: a,
                        el: e.$(".related-links ul")
                    })), e.$("div.related-links").slideDown("150")
                })
            }, e
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    }, c = [].indexOf || function(a) {
        for (var b = 0, c = this.length; c > b; b++)
            if (b in this && this[b] === a)
                return b;
        return - 1
    };
    define("views/widgets/compose_form", ["chaplin", "lib/session", "views/base/hbsview", "lib/avos", "store", "models/connect", "views/widgets/tag_input", "helpers/login_deferred", "views/modals/signup", "views/widgets/compose_result", "views/notification"], function(a, d, e, f, g, h, i, j, k, l) {
        var m, n;
        return m = function(a) {
            function e() {
                return n = e.__super__.constructor.apply(this, arguments)
            }
            return b(e, a), e.prototype.template = "widgets/compose_form", e.prototype.autoRender=!0, e.prototype.autoAttach=!0, e.prototype.initialize = function(a) {
                var b, c, f, g, h, i, j;
                return e.__super__.initialize.apply(this, arguments), this.finish_callback = null != a ? a.finish_callback : void 0, this.url_edit = a.url_edit ||!1, this.anonymous = null != (b = d.user)?!b.get("isLoggedIn") : !0, this.show_result = a.show_result ||!1, this.result_shown=!1, (null != (c = this.model.get("recommended")) ? c.previously_saved : void 0) && (this.url_edit=!1, this.model.set({
                    replace: null != (f = this.model.get("recommended")) ? f.previously_saved: void 0,
                    tags: null != (g = this.model.get("recommended")) ? g.previous_tags: void 0,
                    note: null != (h = this.model.get("recommended")) ? h.previously_saved_note: void 0,
                    title: null != (i = this.model.get("recommended")) ? i.previously_saved_title: void 0,
                    "private": null != (j = this.model.get("recommended")) ? j.previously_saved_privacy: void 0
                })), this.anonymous ? (ga("send", "event", "editor", "anonymous"), this.subscribeEvent("user:login", this.do_submit)) : void 0
            }, e.prototype.events = {
                "click a[href='#compose-login']": "login",
                "submit form": "submit_add_edit",
                "click button.text-btn-primary": "submit_add_edit",
                "click button.text-btn-danger": "submit_remove",
                "click span.tag-suggestion-item": "clicking_tag",
                "change #inputLinkTags": "update_tag_selection",
                "keyup #textareaLinkComment": "update_chars_remaining",
                "click .checkbox-privacy": "toggle_checkbox_privacy",
                "click .checkbox-social": "toggle_checkbox_social"
            }, e.prototype.login = function(a) {
                var b = this;
                return this.publishEvent("compose:login-start"), a.preventDefault(), (new j).prompt().done(function() {
                    return b.anonymous=!1, b.render(), b.publishEvent("compose:login-end")
                })
            }, e.prototype.update_chars_remaining = function() {
                var a, b;
                return b = this.$("#textareaLinkComment").val().length, a = 1e3 - b, a >= 0 ? this.$(".comment_chars_remaining").removeClass("error").text(a) : this.$(".comment_chars_remaining").addClass("error").text(a)
            }, e.prototype.submit_add_edit = function(a) {
                var b, c, e, f = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.$("button.text-btn-primary").text("Saving...").prop("disabled", "disabled"), e = _.map(["twitter", "facebook"], function(a) {
                    var b;
                    return b = f.$("input[name='share-" + a + "']").prop("checked"), g.set("compose:share-" + a, b), b ? a : ""
                }), e = _.filter(e, function(a) {
                    return a
                }), e = e.join(","), c = this.$("input[name='private']")[0].checked, g.set("compose:private", c), b = {
                    url: this.$("input[name='url']").val(),
                    description: this.$("input[name='title']").val(),
                    tags: this.$("input[name='tags']").val(),
                    note: this.$("textarea[name='note']").val(),
                    replace: this.$("input[name='replace']").val(),
                    "private": c,
                    share: e
                }, this.anonymous ? (d.pending_save = b, new k, this.$el.addClass("hide"), this.$("button.text-btn-primary").text("Save").prop("disabled", "")) : this.do_submit(b), ga("send", "event", "editor", "submit"), this.$("textarea[name='note']").val() && ga("send", "event", "editor", "submit-with-note"), e && ga("send", "event", "editor", "submit-with-share"), this.$("input[name='tags']").val() && ga("send", "event", "editor", "submit-with-tags"), c ? ga("send", "event", "editor", "submit-with-private") : void 0
            }, e.prototype.do_submit = function(a) {
                var b, c, e = this;
                return a || (a = d.pending_save), a ? (b = null != (c = this.model.get("recommended"))?!c.previously_saved : !0, this.model.addOrEdit(a).done(function() {
                    var a;
                    return a = e.model.clone(), e.show_result && (new l({
                        model: a,
                        finish_callback: e.finish_callback
                    }), e.result_shown=!0), e.publishEvent("compose:submit"), b && e.publishEvent("links:new"), d.pending_save = null
                })) : void 0
            }, e.prototype.submit_remove = function(a) {
                var b = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), confirm("Are you sure to delete this link?") && (this.$("button.text-btn-danger").text("Deleting...").attr("disabled", "disabled"), console.log(this.model), this.model.remove().done(function() {
                    return b.publishEvent("compose:deleted")
                })), ga("send", "event", "editor", "delete")
            }, e.prototype.filter_invalid_tags = function(a) {
                var b = this;
                return _.reject(a, function(a) {
                    var d;
                    return 0 === a.indexOf("!")?!0 : (d = a.toLowerCase().replace(" ", "").replace(":", ""), /^[0-9]*$/.test(d)?!0 : c.call(b.blacklist, d) >= 0?!0 : !1)
                })
            }, e.prototype.getTemplateData = function() {
                var a, b;
                return a = e.__super__.getTemplateData.apply(this, arguments), b = d.user.get("social") || [], a.twitter = c.call(b, "TWITTER") >= 0, a.twitter && (a.twitter_checked = g.get("compose:share-twitter")), a.facebook = c.call(b, "FACEBOOK") >= 0, a.facebook && (a.facebook_checked = g.get("compose:share-facebook")), a.time_created || (a["private"] = g.get("compose:private")), a.url_edit = this.url_edit, a.anonymous = this.anonymous, a
            }, e.prototype.enhanceCommentInput = function() {
                var a, b, c;
                return a = this.$("textarea.autoresize"), c = function() {
                    return a.height("auto"), a.outerHeight(a.prop("scrollHeight") + 1)
                }, b = function() {
                    return setTimeout(c, 1)
                }, a.on("change", c).on("cut", b).on("paste", b).on("drop", b).on("keydown", b), setTimeout(c, 500)
            }, e.prototype.render = function() {
                var a = this;
                return this.$(".checkbox-action").tooltip("destroy"), e.__super__.render.apply(this, arguments), this.enhanceCommentInput(), this.update_chars_remaining(), this.highlight_suggested_tags(), this.tag_input = new i({
                    el: this.$(".tags-input-container"),
                    tags: d.user.get("tags"),
                    name: "tags",
                    value: this.model.get("tags"),
                    on_change: _.bind(this.update_tag_selection, this)
                }), this.subview("taginput", this.tag_input), _.each(this.model.get("tags"), function(b) {
                    var c;
                    return c = b.replace("\\", "\\\\"), a.$("span.tag-suggestion-item[data-tag='" + c + "']").addClass("on")
                }), this.$(".inline-help").popover(), this.$(".checkbox-action").tooltip({
                    container: ".modal-body",
                    animation: !1,
                    html: !0
                }), this.$("input[name='title']").val()&&!this.$("input[name='tags']").val() && (this.$("input[name='title']").prop("autofocus", ""), this.$("input[name='tags']").prop("autofocus", "autofocus")), ga("send", "event", "editor", "open")
            }, e.prototype.clicking_tag = function(a) {
                var b, c;
                return b = $(a.currentTarget), c = b.attr("data-tag"), b.hasClass("on") ? this.tag_input.remove_value(c) : (this.tag_input.append_value(c), ga("send", "event", "editor", "select-tag", c)), this.update_tag_selection()
            }, e.prototype.highlight_suggested_tags = function() {
                var a, b, c = this;
                return a = _.sortBy(_.map(d.user.get("tags"), function(a) {
                    return a.name
                })), _.each(null != (b = this.model.get("recommended")) ? b.suggested_tags : void 0, function(b) {
                    var d;
                    return _.indexOf(a, b, !0) >= 0 ? (d = b.trim().replace("\\", "\\\\"), c.$("span.tag-suggestion-item[data-tag='" + d + "']").addClass("mine")) : void 0
                })
            }, e.prototype.update_tag_selection = function() {
                var a, b = this;
                return this.$("span.tag-suggestion-item").removeClass("on"), a = this.tag_input.get_value_array(), _.each(a, function(a) {
                    var c;
                    return c = a.trim().replace("\\", "\\\\"), b.$("span.tag-suggestion-item[data-tag='" + c + "']").addClass("on")
                })
            }, e.prototype.dispose = function() {
                return this.$(".checkbox-action").tooltip("destroy"), e.__super__.dispose.apply(this, arguments), _.isFunction(this.finish_callback)&&!this.result_shown ? this.finish_callback() : void 0
            }, e.prototype.toggle_checkbox_privacy = function(a) {
                var b, c;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), b = $(a.currentTarget), c = b.children("span"), b.children("i").toggleClass("d-icon-unlocked"), b.children("input").prop("checked", function(a, b) {
                    return !b
                }), "Private" === c.text() ? (c.text("Public"), b.addClass("on"), b.prop("title", "This link is publicly visible").tooltip("fixTitle").tooltip("show")) : (this.$(".checkbox-social").removeClass("on"), this.$(".checkbox-social input").prop("checked", ""), c.text("Private"), b.removeClass("on"), b.prop("title", "This link is private").tooltip("fixTitle").tooltip("show"))
            }, e.prototype.toggle_checkbox_social = function(a) {
                var b, c, d;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), b = $(a.currentTarget), b.hasClass("unavailable") ? (c = b.attr("data-service"), d = new h({
                    providerId: c
                }), d.connect(!1, function(a) {
                    return "success" === a.status ? (ga("send", "event", "editor", "bind", c), b.children("input").prop("checked", "checked"), b.removeClass("unavailable")) : void 0
                }), void 0) : (b.toggleClass("on"), b.children("input").prop("checked", function(a, b) {
                    return !b
                }))
            }, e
        }(e)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/compose_link", ["views/base/modal", "views/widgets/compose_form"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "modals/compose_link", d.prototype.options = null, d.prototype.initialize = function(a) {
                var b = this;
                return d.__super__.initialize.apply(this, arguments), this.options = a, this.subscribeEvent("compose:submit", this.close), this.subscribeEvent("compose:deleted", this.close), this.subscribeEvent("compose:login-start", function() {
                    return b.$el.removeClass("in")
                }), this.subscribeEvent("compose:login-end", function() {
                    return b.$el.addClass("in")
                })
            }, d.prototype.render = function() {
                return d.__super__.render.apply(this, arguments), this.options.el = this.$(".modal-content"), this.subview("form", new c(this.options))
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/add_link", ["chaplin", "lib/session", "views/base/modal", "views/modals/compose_link", "models/link"], function(a, c, d, e, f) {
        var g, h;
        return g = function(a) {
            function d() {
                return h = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "modals/add_link", d.prototype.initialize = function() {
                return d.__super__.initialize.apply(this, arguments), c.add_link_modal=!0
            }, d.prototype.events = {
                "paste #url-input": "check_and_prepend_scheme",
                "click button.text-btn-primary": "submit_dialog",
                "submit form": "submit_dialog"
            }, d.prototype.submit_dialog = function(a) {
                var b, c, d = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), c = this.$("input[type='url']").val(), b = new f({
                    url: c
                }), this.$("button.text-btn-primary").text("Adding...").prop("disabled", "disabled"), b.compose().done(function() {
                    return d.$el.modal("hide"), new e({
                        model: b
                    })
                }).fail(function() {
                    return d.$("button.text-btn-primary").text("Add link").prop("disabled", "")
                }), ga("send", "event", "addlink", "submit")
            }, d.prototype.dispose = function() {
                return delete c.add_link_modal, d.__super__.dispose.apply(this, arguments)
            }, d.prototype.check_and_prepend_scheme = function() {
                var a = this;
                return _.delay(function() {
                    var b;
                    return b = a.$("#url-input").val(), b.trim()&&!/^https?:\/\//.test(b) ? a.$("#url-input").val("http://" + b) : void 0
                }, 250)
            }, d
        }(d)
    })
}.call(this), function(a, b) {
    function c(a, b, c) {
        return a.addEventListener ? void a.addEventListener(b, c, !1) : void a.attachEvent("on" + b, c)
    }
    function d(a) {
        if ("keypress" == a.type) {
            var b = String.fromCharCode(a.which);
            return a.shiftKey || (b = b.toLowerCase()), b
        }
        return y[a.which] ? y[a.which] : z[a.which] ? z[a.which] : String.fromCharCode(a.which).toLowerCase()
    }
    function e(a, b) {
        return a.sort().join(",") === b.sort().join(",")
    }
    function f(a) {
        a = a || {};
        var b, c=!1;
        for (b in E)
            a[b] ? c=!0 : E[b] = 0;
        c || (H=!1)
    }
    function g(a, b, c, d, f, g) {
        var h, i, j = [], k = c.type;
        if (!C[a])
            return [];
        for ("keyup" == k && n(a) && (b = [a]), h = 0; h < C[a].length; ++h)
            if (i = C[a][h], (d ||!i.seq || E[i.seq] == i.level) && k == i.action && ("keypress" == k&&!c.metaKey&&!c.ctrlKey || e(b, i.modifiers))) {
                var l=!d && i.combo == f, m = d && i.seq == d && i.level == g;
                (l || m) && C[a].splice(h, 1), j.push(i)
            }
        return j
    }
    function h(a) {
        var b = [];
        return a.shiftKey && b.push("shift"), a.altKey && b.push("alt"), a.ctrlKey && b.push("ctrl"), a.metaKey && b.push("meta"), b
    }
    function i(a) {
        return a.preventDefault ? void a.preventDefault() : void(a.returnValue=!1)
    }
    function j(a) {
        return a.stopPropagation ? void a.stopPropagation() : void(a.cancelBubble=!0)
    }
    function k(a, b, c, d) {
        J.stopCallback(b, b.target || b.srcElement, c, d) || a(b, c)===!1 && (i(b), j(b))
    }
    function l(a, b, c) {
        var d, e = g(a, b, c), h = {}, i = 0, j=!1;
        for (d = 0; d < e.length; ++d)
            e[d].seq && (i = Math.max(i, e[d].level));
        for (d = 0; d < e.length; ++d)
            if (e[d].seq) {
                if (e[d].level != i)
                    continue;
                    j=!0, h[e[d].seq] = 1, k(e[d].callback, c, e[d].combo, e[d].seq)
            } else 
                j || k(e[d].callback, c, e[d].combo);
        var l = "keypress" == c.type && G;
        c.type == H&&!n(a)&&!l && f(h), G = j && "keydown" == c.type
    }
    function m(a) {
        "number" != typeof a.which && (a.which = a.keyCode);
        var b = d(a);
        if (b)
            return "keyup" == a.type && F === b ? void(F=!1) : void J.handleKey(b, h(a), a)
    }
    function n(a) {
        return "shift" == a || "ctrl" == a || "alt" == a || "meta" == a
    }
    function o() {
        clearTimeout(x), x = setTimeout(f, 1e3)
    }
    function p() {
        if (!w) {
            w = {};
            for (var a in y)
                a > 95 && 112 > a || y.hasOwnProperty(a) && (w[y[a]] = a)
        }
        return w
    }
    function q(a, b, c) {
        return c || (c = p()[a] ? "keydown" : "keypress"), "keypress" == c && b.length && (c = "keydown"), c
    }
    function r(a, b, c, e) {
        function g(b) {
            return function() {
                H = b, ++E[a], o()
            }
        }
        function h(b) {
            k(c, b, a), "keyup" !== e && (F = d(b)), setTimeout(f, 10)
        }
        E[a] = 0;
        for (var i = 0; i < b.length; ++i) {
            var j = i + 1 === b.length, l = j ? h: g(e || t(b[i + 1]).action);
            u(b[i], l, e, a, i)
        }
    }
    function s(a) {
        return "+" === a ? ["+"] : a.split("+")
    }
    function t(a, b) {
        var c, d, e, f = [];
        for (c = s(a), e = 0; e < c.length; ++e)
            d = c[e], B[d] && (d = B[d]), b && "keypress" != b && A[d] && (d = A[d], f.push("shift")), n(d) && f.push(d);
        return b = q(d, f, b), {
            key: d,
            modifiers: f,
            action: b
        }
    }
    function u(a, b, c, d, e) {
        D[a + ":" + c] = b, a = a.replace(/\s+/g, " ");
        var f, h = a.split(" ");
        return h.length > 1 ? void r(a, h, b, c) : (f = t(a, c), C[f.key] = C[f.key] || [], g(f.key, f.modifiers, {
            type: f.action
        }, d, a, e), C[f.key][d ? "unshift": "push"]({
            callback: b,
            modifiers: f.modifiers,
            action: f.action,
            seq: d,
            level: e,
            combo: a
        }), void 0)
    }
    function v(a, b, c) {
        for (var d = 0; d < a.length; ++d)
            u(a[d], b, c)
    }
    for (var w, x, y = {
        8: "backspace",
        9: "tab",
        13: "enter",
        16: "shift",
        17: "ctrl",
        18: "alt",
        20: "capslock",
        27: "esc",
        32: "space",
        33: "pageup",
        34: "pagedown",
        35: "end",
        36: "home",
        37: "left",
        38: "up",
        39: "right",
        40: "down",
        45: "ins",
        46: "del",
        91: "meta",
        93: "meta",
        224: "meta"
    }, z = {
        106: "*",
        107: "+",
        109: "-",
        110: ".",
        111: "/",
        186: ";",
        187: "=",
        188: ",",
        189: "-",
        190: ".",
        191: "/",
        192: "`",
        219: "[",
        220: "\\",
        221: "]",
        222: "'"
    }, A = {
        "~": "`",
        "!": "1",
        "@": "2",
        "#": "3",
        $: "4",
        "%": "5",
        "^": "6",
        "&": "7",
        "*": "8",
        "(": "9",
        ")": "0",
        _: "-",
        "+": "=",
        ":": ";",
        '"': "'",
        "<": ",",
        ">": ".",
        "?": "/",
        "|": "\\"
    }, B = {
        option: "alt",
        command: "meta",
        "return": "enter",
        escape: "esc",
        mod: /Mac|iPod|iPhone|iPad/.test(navigator.platform) ? "meta": "ctrl"
    }, C = {}, D = {}, E = {}, F=!1, G=!1, H=!1, I = 1; 20 > I; ++I)
        y[111 + I] = "f" + I;
    for (I = 0; 9 >= I; ++I)
        y[I + 96] = I;
    c(b, "keypress", m), c(b, "keydown", m), c(b, "keyup", m);
    var J = {
        bind: function(a, b, c) {
            return a = a instanceof Array ? a : [a], v(a, b, c), this
        },
        unbind: function(a, b) {
            return J.bind(a, function() {}, b)
        },
        trigger: function(a, b) {
            return D[a + ":" + b] && D[a + ":" + b]({}, a), this
        },
        reset: function() {
            return C = {}, D = {}, this
        },
        stopCallback: function(a, b) {
            return (" " + b.className + " ").indexOf(" mousetrap ")>-1?!1 : "INPUT" == b.tagName || "SELECT" == b.tagName || "TEXTAREA" == b.tagName || b.isContentEditable
        },
        handleKey: l
    };
    a.Mousetrap = J, "function" == typeof define && define.amd && define("mousetrap", J)
}(window, document), function() {
    define("helpers/ua", [], function() {
        var a;
        return a = {
            isiPhone: function() {
                return navigator.userAgent.match(/iPhone/i) || navigator.userAgent.match(/iPod/i)
            },
            isiPad: function() {
                return navigator.userAgent.match(/iPad/i)
            },
            isiOS: function() {
                return a.isiPhone() || a.isiPad()
            },
            isAndroid: function() {
                return navigator.userAgent.match(/Android/i)
            },
            isSafari: function() {
                return navigator.userAgent.match(/Safari/i)&&!navigator.userAgent.match(/Chrome|Chromium/i)
            },
            isChrome: function() {
                return navigator.userAgent.match(/Chrome|Chromium/i)
            },
            isFirefox: function() {
                return navigator.userAgent.match(/Firefox/i)
            },
            isMobile: function() {
                return navigator.userAgent.match(/Mobile/i) || a.isAndroid()
            },
            isDesktop: function() {
                return !a.isMobile()
            },
            isChromeOS: function() {
                return navigator.userAgent.match(/CrOS/i)
            },
            isFirefoxOS: function() {
                return a.isMobile()&&!a.isAndroid() && a.isFirefox()
            }
        }
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/layout", ["chaplin", "lib/session", "views/sidebar", "views/notification", "models/user", "views/modals/login", "views/modals/signup", "views/modals/help", "views/modals/add_link", "store", "mousetrap", "helpers/login_deferred", "lib/avos", "helpers/ua"], function(a, c, d, e, f, g, h, i, j, k, l, m, n, o) {
        var p, q;
        return p = function(f) {
            function p() {
                return q = p.__super__.constructor.apply(this, arguments)
            }
            return b(p, f), p.prototype.initialize = function() {
                var a = this;
                return p.__super__.initialize.apply(this, arguments), this.sidebar = new d({
                    model: c.user
                }), this.sidebar.attach(), $("html").removeClass("loading"), $(window).scroll(function() {
                    return a.onDocumentScroll()
                }), this.subscribeEvent("links:load-start", function() {
                    return a.loadMoreIsRunning=!0
                }), this.subscribeEvent("links:load-finish", function() {
                    return a.loadMoreIsRunning=!1
                }), this.subscribeEvent("system:timeout", this.show_error), this.subscribeEvent("page:load-start", this.start_loading), this.subscribeEvent("page:load-finish", this.stop_loading), this.setup_ua(), this.setup_shortcuts(), this.checkAndShowFirefoxInstall() && this.checkAndShowAndroidInstall(), this.checkAndAddAppCache()
            }, p.prototype.events = {
                "click a[href='#login']": "login",
                "click a[href='#signup']": "signup",
                "click a[href='#link']": "show_add_link_modal",
                "click a[href='#firefox-webapp']": "install_firefox_webapp",
                "click a[href='#help']": "show_help_modal"
            }, p.prototype.setup_ua = function() {
                return $("html").attr("data-ua", navigator.userAgent)
            }, p.prototype.onDocumentScroll = function(a) {
                var b = this;
                return this._throttledTriggerLoadMoreEvent || (this._throttledTriggerLoadMoreEvent = _.throttle(function() {
                    return b.publishEvent("links:fixed-navbar"), b.loadMoreIsRunning ? void 0 : $(window).scrollTop() >= $(document).height() - $(window).height() - 500 ? b.publishEvent("links:load-trigger") : void 0
                }, 0)), this._throttledTriggerLoadMoreEvent(a)
            }, p.prototype.checkAndShowIphoneApp = function() {
                var a;
                return a = "notif:iphone:1", k.get(a) ? void 0 : new e({
                    message: '<a href="https://itunes.apple.com/us/app/delicious-official-app/id580295142?mt=8">Checkout our new iPhone App</a>.',
                    id: a
                })
            }, p.prototype.checkAndShowFirefoxInstall = function() {
                var a;
                if (o.isDesktop() && o.isFirefox())
                    return a = "notif:firefox:2", k.get(a) ? void 0 : (new e({
                        message: 'Dear Firefox user, you can now activate the new Delicious Firefox integration from our <a href="/tools">tools page</a>.',
                        id: a
                    }), !0)
            }, p.prototype.checkAndShowAndroidInstall = function() {
                var a;
                if (o.isAndroid())
                    return a = "notif:android:1", k.get(a) ? void 0 : new e({
                        message: 'Delicious official Android app is available in <a href="https://play.google.com/store/apps/details?id=com.delicious">Play Store</a>! (<a href="http://blog.delicious.com/2013/10/the-delicious-android-app-is-here/" target="_blank">Learn more</a>)',
                        id: a
                    })
            }, p.prototype.checkAndShowChromeExtensionInstall = function() {
                var a;
                if (o.isChrome()&&!o.isAndroid())
                    return a = "notif:chrome:1", k.get(a) ? void 0 : new e({
                        message: 'Using Chrome? <a href="https://chrome.google.com/webstore/detail/delicious-bookmarks-exten/mnaelnkmidnndgikjbiifihgklnocljd">Delicious official Chrome extension</a> is back!',
                        id: a
                    })
            }, p.prototype.checkAndAddAppCache = function() {
                var a, b;
                return window.applicationCache && (a = function() {
                    return window.applicationCache.swapCache(), _.isEmpty($("[data-notif-name='appcache-update']")) ? new e({
                        name: "appcache-update",
                        message: 'There are some changes to Delicious, <a href="javascript:location.reload();" class="noscript">refresh</a> to update'
                    }) : void 0
                }, null != (b = window.applicationCache) && b.addEventListener("updateready", a), applicationCache.status === applicationCache.UPDATEREADY) ? a() : void 0
            }, p.prototype.promoteNewDiscover = function() {
                var a;
                return a = "notif:discover:1", k.get(a) ? void 0 : new e({
                    id: a,
                    message: 'Never miss a link again! Check out the new <a href="/discover">Discover</a> feed we built with your interests.'
                })
            }, p.prototype.install_firefox_webapp = function(a) {
                var b;
                return null != a && a.preventDefault(), navigator.mozApps ? (b = navigator.mozApps.install(n.webapp_url), b.onsuccess = function() {
                    return alert("Delicious webapp installed.")
                }, b.onerror = function() {
                    return alert("Install failed, error: " + this.error.name)
                }) : void 0
            }, p.prototype.show_error = function() {
                var a;
                return _.isEmpty($("[data-notif-name='timeout']")) ? (a = new e({
                    message: '<span class="text-danger">Error communicating with the server, try <a href="javascript:location.reload();" class="noscript">refreshing page</a>.</span>',
                    type: "notification",
                    name: "timeout"
                }), a.attach()) : void 0
            }, p.prototype.ensure_logged_in = function(a) {
                return function(b) {
                    return c.user.get("isLoggedIn") ? a(b) : void 0
                }
            }, p.prototype.show_help_modal = function(a) {
                return null != a && a.preventDefault(), c.help_modal ? void 0 : new i
            }, p.prototype.setup_shortcuts = function() {
                var b = this;
                return l.bind("?", function() {
                    return b.show_help_modal()
                }), l.bind(["1", "g h"], function() {
                    return b.ensure_logged_in(function() {
                        return a.helpers.redirectTo({
                            url: "/"
                        })
                    })()
                }), l.bind(["2", "g n"], function() {
                    return b.ensure_logged_in(function() {
                        return a.helpers.redirectTo({
                            url: "/network"
                        }, {
                            forceStartup: !0
                        })
                    })()
                }), l.bind(["3", "g d"], function() {
                    return b.ensure_logged_in(function() {
                        return a.helpers.redirectTo({
                            url: "/discover"
                        }, {
                            forceStartup: !0
                        })
                    })()
                }), l.bind(["4", "a"], function() {
                    return b.ensure_logged_in(function() {
                        return b.show_add_link_modal()
                    })()
                }), l.bind("5", function() {
                    return b.ensure_logged_in(function() {
                        return a.helpers.redirectTo({
                            url: "/settings"
                        }, {
                            forceStartup: !0
                        })
                    })()
                }), l.bind(["s", "`"], function() {
                    return $("#site-search").val("").focus(), !1
                }), l.bind("/", function() {
                    return b.ensure_logged_in(function() {
                        var a;
                        return $("#site-search").focus().val("@" + c.user.get("username") + " "), "function" == typeof(a = $("#site-search")[0]).setSelectionRange && a.setSelectionRange(1e3, 1e3), !1
                    })()
                }), l.bind("#", function() {
                    return b.ensure_logged_in(function() {
                        return $("#site-search").focus().val("#"), !1
                    })()
                }), l.bind("@", function() {
                    return b.ensure_logged_in(function() {
                        return $("#site-search").focus().val("@"), !1
                    })()
                }), l.bind("e", function() {
                    return b.ensure_logged_in(function() {
                        return b.publishEvent("links:toggle-edit")
                    })()
                }), l.bind("b", function() {
                    return b.publishEvent("links:toggle-bundles")
                }), l.bind("t", function() {
                    return b.publishEvent("links:toggle-tags")
                }), l.bind("d", function() {
                    return b.publishEvent("links:toggle-datetable")
                })
            }, p.prototype.login = function(a) {
                return a.preventDefault(), new g({
                    redirect_on_login: !0
                })
            }, p.prototype.signup = function(a) {
                return a.preventDefault(), new h
            }, p.prototype.show_add_link_modal = function(a) {
                var b = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), c.add_link_modal ? void 0 : c.user.get("isLoggedIn") ? new j : (new m).prompt().done(function() {
                    return b.show_add_link_modal()
                })
            }, p.prototype.start_loading = function() {
                return $("#site-loading").fadeIn(200)
            }, p.prototype.stop_loading = function() {
                return $("#site-loading").fadeOut(200)
            }, p
        }(a.Layout)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("controllers/base/controller", ["chaplin", "lib/session"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.startLoading = function() {
                return this.publishEvent("page:load-start")
            }, c.prototype.stopLoading = function() {
                return this.publishEvent("page:load-finish")
            }, c
        }(a.Controller)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/base/pageview", ["views/base/hbsview"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.container = "#delicious", c.prototype.autoRender=!0, c.prototype.autoAttach=!0, c.prototype.className = "content-wrapper", c.prototype.containerMethod = "append", c.prototype.showSidebar=!0, c.prototype.attach = function() {
                return c.__super__.attach.apply(this, arguments), this.publishEvent("sidebar:change", {
                    action: this.showSidebar ? "show": "hide"
                }), this.publishEvent("page:loaded", this)
            }, c.prototype.dispose = function() {
                return c.__super__.dispose.apply(this, arguments), this.publishEvent("page:disposed", this)
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/home/index", ["views/base/pageview", "models/tags"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.showSidebar=!1, c.prototype.template = "home/index", c.prototype.className = "homepage", c.prototype.attach = function() {
                return c.__super__.attach.apply(this, arguments), $("body").addClass("body-landing"), $("#parallax").parallax().addClass("loaded")
            }, c.prototype.events = {
                "click a.trending-title": "_track_trending_click"
            }, c.prototype.dispose = function() {
                return c.__super__.dispose.apply(this, arguments), $("body").removeClass("body-landing ready")
            }, c.prototype.on_home_search_submit = function(a) {
                var b;
                return a.preventDefault(), b = this.$("input[name='search-term']").val(), $("#site-search").val(b), $("#location-bar").submit(), ga("send", "event", "landing", "search", b)
            }, c.prototype.update_tags = function() {
                var a, b;
                return a = this.trendingTags.map(function(a) {
                    return "#<a href='/tag/" + a.get("name") + "'>" + a.get("name") + "</a>"
                }), b = _.map(_.first(_.shuffle(this.featuredUsers), 6), function(a) {
                    return "@<a href='/" + a + "'>" + a + "</a>"
                }), this.$(".tips").hide().html(_.union(a, b).join(" ")).slideDown(400)
            }, c.prototype._track_click = function(a) {
                var b;
                return b = $(a.currentTarget).attr("href"), ga("send", "event", "landing", "greeting", b)
            }, c.prototype._track_trending_click = function(a) {
                var b;
                return b = $(a.currentTarget).attr("href"), ga("send", "event", "landing", "trending", b)
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/home/trending", ["views/base/pageview"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "home/trending", c.prototype.initialize = function() {
                return c.__super__.initialize.apply(this, arguments)
            }, c.prototype.getTemplateData = function() {
                var a;
                return a = c.__super__.getTemplateData.apply(this, arguments)
            }, c.prototype.events = {
                "click a.trending-title": "_track_trending_click"
            }, c.prototype._track_trending_click = function(a) {
                var b;
                return b = $(a.currentTarget).attr("href"), ga("send", "event", "trending", "trending", b)
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/reset_password", ["chaplin", "lib/session", "views/base/modal", "views/modals/login", "views/modals/forgot_password"], function(a, c, d, e, f) {
        var g, h;
        return g = function(a) {
            function d() {
                return h = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "modals/reset_password", d.prototype.has_error=!1, d.prototype.initialize = function(a) {
                return this.id = null != a ? a.id : void 0, this.passwd_hash = null != a ? a.passwd_hash : void 0, this.time = null != a ? a.time : void 0
            }, d.prototype.events = {
                "submit form": "do_submit",
                "click .submit-button": "do_submit",
                "click a[href='#forgot_password']": "forgot_password",
                "click a[href='#login']": "login"
            }, d.prototype.validate_password = function() {
                var a;
                return a = this.$("input[type='password']").val(), this.show_error(""), this.has_error=!1, a.length < 8 ? (this.has_error=!0, void this.show_error("Password is too short.")) : a.indexOf(" ") > 0 ? (this.has_error=!0, this.show_error("Sorry, space is not allowed in password.")) : void 0
            }, d.prototype.do_submit = function(a) {
                var b, d, e = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.validate_password(), this.has_error ? void 0 : (d = this.$("input[name='password']").val(), b = {
                    user_id: this.id,
                    timestamp: this.time,
                    old_password_hash: this.passwd_hash,
                    new_password: d
                }, this.$(".submit-button").prop("disabled", "disabled"), c.user.resetPassword(b).fail(function(a) {
                    return e.fail(a)
                }).done(function() {
                    return e.success()
                }))
            }, d.prototype.fail = function(a) {
                return this.$(".submit-button").prop("disabled", ""), this.show_error("expired_session" === a.status ? "Session expired, please <a href='#forgot_password'>try again</a>." : a.error.charAt(0).toUpperCase() + a.error.slice(1))
            }, d.prototype.success = function() {
                return this.$(".reset-success").removeClass("hidden"), this.show_error("")
            }, d.prototype.show_error = function(a) {
                return this.$(".error-msg").html(a)
            }, d.prototype.forgot_password = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.close(), new f
            }, d.prototype.login = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.close(), new e({
                    redirect_on_login: !0
                })
            }, d
        }(d)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/base/html", ["views/base/hbsview"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.autoRender=!0, c.prototype.autoAttach=!0, c.prototype.container = "#delicious", c.prototype.className = "content-wrapper", c.prototype.containerMethod = "append", c.prototype.showSidebar=!0, c.prototype.template = "html", c.prototype.initialize = function(a) {
                return c.__super__.initialize.apply(this, arguments), this.url = a.url
            }, c.prototype.render = function() {
                return c.__super__.render.apply(this, arguments), this.url ? (this.$("li[data-page='" + this.url + "']").addClass("on"), this.$(".static-html-wrapper").load("/html/" + this.url + ".html")) : void 0
            }, c.prototype.attach = function() {
                return c.__super__.attach.apply(this, arguments), this.publishEvent("sidebar:change", {
                    action: this.showSidebar ? "show": "hide"
                })
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/not_found", ["views/base/pageview"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "not_found", c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("controllers/home_controller", ["chaplin", "lib/session", "controllers/base/controller", "models/links", "views/home/index", "views/home/trending", "views/modals/login", "views/modals/signup", "views/modals/reset_password", "views/base/html", "views/notification", "views/not_found"], function(a, c, d, e, f, g, h, i, j, k, l, m) {
        var n, o;
        return n = function(a) {
            function d() {
                return o = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.index = function(a, b, d) {
                var e;
                return e = c.user, d.query.mozsidebar && (c.is_moz_sidebar=!0, ga("send", "event", "firefox-social-api", "sidebar")), (null != e ? e.get("isLoggedIn") : void 0) ? this.redirectTo({
                    url: "/" + e.get("username")
                }) : (this.view = new f, $("title").text("Delicious"))
            }, d.prototype.trending = function() {
                return c.user.get("isLoggedIn") ? (this.view = new g, this.publishEvent("sidebar:change", {
                    action: "page",
                    location: "trending"
                }), this.adjustTitle("Trending")) : void this.redirectTo({
                    url: "/"
                })
            }, d.prototype.logout = function() {
                var a;
                return null != (a = c.user) && a.logout(), this.publishEvent("login:logout"), window.location = "/"
            }, d.prototype.signin = function() {
                var a;
                return (null != (a = c.user) ? a.get("isLoggedIn") : 0) ? this.redirectTo({
                    url: c.user.get("username")
                }) : (this.redirectTo({
                    url: "/"
                }), new h({
                    redirect_on_login: !0
                }))
            }, d.prototype.signup = function() {
                var a;
                return (null != (a = c.user) ? a.get("isLoggedIn") : 0) ? void 0 : (this.redirectTo({
                    url: "/"
                }), new i)
            }, d.prototype.reset_password = function(a) {
                return this.redirectTo({
                    url: "/"
                }), new j({
                    id: a.id,
                    time: a.time,
                    passwd_hash: a.passhash
                })
            }, d.prototype.page = function(a) {
                var b, c;
                return c = {
                    url: a.page
                }, this.view = new k(c), this.publishEvent("sidebar:change", {
                    action: "page"
                }), c.url ? (b = c.url.charAt(0).toUpperCase() + c.url.slice(1), this.adjustTitle(b)) : void 0
            }, d.prototype.validate_email = function(a) {
                var b;
                return b = a.token, c.user.validateEmail(b, function(a) {
                    return new l(a.success ? {
                        type: "notification",
                        message: "Thank you, your email has been validated."
                    } : {
                        type: "notification",
                        message: "" + a.error + " If you are having trouble, please contact <a href='mailto:support@delicious.com'>support@delicious.com</a>"
                    })
                }), this.redirectTo({
                    url: "/"
                })
            }, d.prototype.not_found = function() {
                return this.view = new m
            }, d
        }(d)
    })
}.call(this), function() {
    var a = function(a, b) {
        return function() {
            return a.apply(b, arguments)
        }
    }, b = {}.hasOwnProperty, c = function(a, c) {
        function d() {
            this.constructor = a
        }
        for (var e in c)
            b.call(c, e) && (a[e] = c[e]);
        return d.prototype = c.prototype, a.prototype = new d, a.__super__ = c.prototype, a
    };
    define("models/tag_bundle", ["models/base/model", "chaplin", "lib/session"], function(b, d, e) {
        var f, g;
        return f = function(b) {
            function d() {
                return this.saveResponse = a(this.saveResponse, this), g = d.__super__.constructor.apply(this, arguments)
            }
            return c(d, b), d.prototype["delete"] = function() {
                var a = this;
                return this.api({
                    url: "/tags/bundle/delete",
                    type: "post",
                    data: {
                        bundle_name: this.get("name")
                    },
                    success: function(b) {
                        var c;
                        return "success" === (null != b ? b.status : void 0) ? (a.publishEvent("tagBundle:delete"), null != (c = a.collection) && c.remove(a), a.destroy()) : a.publishEvent("tagBundle:delete:error")
                    }
                })
            }, d.prototype.fetchTags = function(a, b) {
                var c = this;
                return this.api({
                    data: {
                        bundle_owner: a,
                        bundle_name: decodeURIComponent(b)
                    },
                    url: "/tags/bundle/tags",
                    success: function(a) {
                        return c.set({
                            tags: _.map(a.pkg.tags, function(a) {
                                return a.tag_name
                            }),
                            name: b
                        })
                    }
                })
            }, d.prototype.save = function(a) {
                var b;
                return this.newdata = {
                    name: a.name,
                    tags: a.tags
                }, null != a.previous ? (b = {
                    url: "/tags/bundle/edit",
                    type: "post",
                    data: {
                        new_bundle_name: a.name,
                        old_bundle_name: a.previous.get("name"),
                        new_bundle_tags: a.tags,
                        old_bundle_tags: a.previous.get("tags").join(","),
                        bundle_owner: e.user.get("username")
                    },
                    success: this.saveResponse
                }, this.api(b)) : (b = {
                    url: "/tags/bundle/create",
                    type: "post",
                    data: {
                        bundle_name: a.name,
                        tags: a.tags
                    },
                    success: this.saveResponse
                }, this.api(b))
            }, d.prototype.saveResponse = function(a) {
                return "success" === a.status ? (this.set(this.newdata), this.publishEvent("tagBundle:save", this)) : this.publishEvent("tagBundle:save:error", this)
            }, d
        }(b)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("models/tag_bundles", ["models/base/collection", "models/tag_bundle"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.model = c, d.prototype.url = function() {
                var a;
                return this.urlParamDefaults || (this.urlParamDefaults = {
                    username: null
                }), a = _.extend(this.urlParamDefaults, this.urlParams).username, a ? "/tags/bundles/" + a : void 0
            }, d.prototype.parse = function(a) {
                var b, c, e;
                return d.__super__.parse.apply(this, arguments), b = null != (c = a.pkg) && null != (e = c[0]) ? e.bundle_names : void 0, b ? _.map(b, function(a) {
                    return {
                        name: a
                    }
                }) : []
            }, d.prototype.comparator = function(a) {
                var b;
                return null != (b = a.get("name")) ? b.toLowerCase() : void 0
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/add_tag", ["chaplin", "lib/session", "views/base/modal", "views/widgets/tag_input"], function(a, c, d, e) {
        var f, g;
        return f = function(a) {
            function d() {
                return g = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "modals/add_tag", d.prototype.initialize = function() {
                return d.__super__.initialize.apply(this, arguments)
            }, d.prototype.render = function() {
                return d.__super__.render.apply(this, arguments), this.tag_input = new e({
                    el: this.$(".tags-input-container"),
                    tags: c.user.get("tags"),
                    name: "tags"
                }), this.subview("taginput", this.tag_input)
            }, d.prototype.events = {
                "click .text-btn-primary": "on_submit"
            }, d.prototype.on_submit = function() {
                var a, b = this;
                return this.$(".text-btn-primary").prop("disabled", "disabled"), a = this.tag_input.get_value_array(), a = _.map(a, function(a) {
                    return a.trim()
                }), this.collection.add_tag(a).done(function() {
                    return b.close()
                })
            }, d
        }(d)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/remove_tag", ["views/base/modal"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "modals/remove_tag", c.prototype.initialize = function() {
                return c.__super__.initialize.apply(this, arguments), this.tags = _.intersection.apply(null, this.collection.map(function(a) {
                    return a.get("tags")
                }))
            }, c.prototype.getTemplateData = function() {
                return this.tags
            }, c.prototype.events = {
                "click ul.tags li a": "toggle_tag_select",
                "click .text-btn-primary": "on_submit"
            }, c.prototype.toggle_tag_select = function(a) {
                return a.preventDefault(), $(a.currentTarget).toggleClass("selected")
            }, c.prototype.get_selected_tags = function() {
                return _.map(this.$("ul.tags li a.selected"), function(a) {
                    return $(a).text()
                })
            }, c.prototype.on_submit = function() {
                var a, b = this;
                return this.$(".text-btn-primary").prop("disabled", "disabled"), a = this.get_selected_tags(), this.collection.remove_tag(a).done(function() {
                    return b.close()
                })
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/page", ["chaplin", "moment", "views/base/pageview", "models/link", "models/links", "models/tag_bundle", "views/modals/add_tag", "views/modals/remove_tag"], function(a, c, d, e, f, g, h, i) {
        var j, k;
        return j = function(d) {
            function j() {
                return k = j.__super__.constructor.apply(this, arguments)
            }
            return b(j, d), j.prototype.template = "links/page", j.prototype.editable=!1, j.prototype.editing=!1, j.prototype.is_user=!0, j.prototype.is_tag=!1, j.prototype.is_search=!1, j.prototype.previous_date_value = "", j.prototype.initialize = function(a) {
                return j.__super__.initialize.call(this, a), this.editable = a.editable, this.tab = a.tab, this.show_tags = a.show_tags, "tag" === a.type && (this.is_user=!1, this.is_tag=!0), "search" === a.type && (this.is_user=!1, this.is_search=!0), this.subscribeEvent("links:load-finish", this.update_related_tags), this.subscribeEvent("links:fixed-navbar", this.check_and_update_navbar), this.subscribeEvent("links:toggle-edit", this.toggle_bulk_edit), this.subscribeEvent("links:toggle-tags", this.toggle_tagcloud), this.subscribeEvent("links:toggle-bundles", this.toggle_tagbundles), this.subscribeEvent("links:toggle-datetable", this.toggle_datetable)
            }, j.prototype.check_and_update_navbar = function() {
                var a;
                return a = $(".bulk-edit-status").outerHeight(), $(window).scrollTop() > a && $(".content-wrapper").hasClass("bulk-edit-on") && (this.$(".link-nav-fixed-wrapper").addClass("on"), this.$(".bulk-edit-status").css("margin-bottom", $(".link-nav").outerHeight())), $(window).scrollTop() < a && $(".content-wrapper").hasClass("bulk-edit-on") ? (this.$(".link-nav-fixed-wrapper").removeClass("on"), this.$(".bulk-edit-status").css("margin-bottom", 0)) : void 0
            }, j.prototype.getTemplateData = function() {
                var a;
                return a = j.__super__.getTemplateData.apply(this, arguments), a.is_user = this.is_user, a.is_tag = this.is_tag, a.is_search = this.is_search, a.editable = this.editable, a
            }, j.prototype.render = function() {
                return j.__super__.render.apply(this, arguments), this.$(".actions a[data-tab='" + this.tab + "']").parent().addClass("on"), this.show_tags ? this.$("a[href='#tags']").click() : void 0
            }, j.prototype.events = {
                "click a.ftag": "filter_by_ftag",
                "click div.tagcloud a[data-tag]": "filter_by_tag",
                "click ul.relatedtags a.filter-tag": "filter_by_appending_tag",
                "click ul.relatedtags a.show-tag": "filter_by_tag",
                "click a.tag-for-filter": "filter_by_appending_tag",
                "click a.tag-by-people": "filter_by_tag",
                "click ul.tagbundles a[href='#']": "filter_by_tag_bundle",
                "submit form.links-search-form": "filter_by_search",
                "click div.filter a.filter-cancel": "remove_filter",
                "click div.filter a.filter-cancel-all": "remove_filters",
                "click a.bulk-edit": "toggle_bulk_edit",
                "click a[href='#tags']": "toggle_tagcloud",
                "click a[href='#tagbundles']": "toggle_tagbundles",
                "click a[href='#datetable']": "toggle_datetable",
                "click a[data-tab='everyone']": "change_search_type",
                "click a[data-tab='network']": "change_search_type",
                "click a[data-tab='you']": "change_search_type",
                "click a[data-tab='hot']": "sort_tag_link_list",
                "click a[data-tab='recent']": "sort_tag_link_list",
                "click a[data-tab='alltime']": "sort_tag_link_list",
                "click a.bulk-edit-action-public": "bulk_action_public",
                "click a.bulk-edit-action-private": "bulk_action_private",
                "click a.bulk-edit-action-remove": "bulk_action_remove",
                "click a.bulk-edit-action-select-all": "bulk_action_all",
                "click a.bulk-edit-action-select-none": "bulk_action_none",
                "click a.bulk-edit-action-add-tag": "bulk_add_tags",
                "click a.bulk-edit-action-remove-tag": "bulk_remove_tags",
                "click div.bulk-edit-on div.item": "update_selection",
                "click a.sort-by-alphabet": "sort_tags",
                "click a.sort-by-count": "sort_tags",
                "click a.edit-tag": "edit_tag",
                "click a[href='#edit-tag-bundle']": "show_tagbundle_dialog",
                "click div.links.bulk-edit-on ul.tags a": "disable_click_when_bulk_edit",
                "click .datetable a": "filter_by_date"
            }, j.prototype.filter_by_date = function(b) {
                var d, e, f, g;
                return b.preventDefault(), this.$(".datetable a").removeClass("on"), d = $(b.currentTarget), e = d.attr("data-date"), f = this.subview("links").collection, e && e !== this.previous_date_value ? (f.reset(), g = c(e, "YYYYMM").add("M", 1).subtract("d", 1).unix(), f.urlParams.timeStart = g, this.previous_date_value = e, a.mediator.execute("router:changeURL", f.to_user_url()), d.addClass("on"), f.fetch()) : void 0
            }, j.prototype.filter_by_ftag = function(a) {
                var b, c;
                return a.preventDefault(), b = $(a.currentTarget), c = this.subview("links").collection, c.reset(), b.hasClass("ftag-notag") && (c.urlParams.noTag=!b.hasClass("on"), b.toggleClass("on")), b.hasClass("ftag-public") && (b.hasClass("on") ? (c.urlParams.visibility = null, b.removeClass("on")) : (c.urlParams.visibility = "public", b.addClass("on"), this.$(".ftag-private").removeClass("on"))), b.hasClass("ftag-private") && (b.hasClass("on") ? (c.urlParams.visibility = null, b.removeClass("on")) : (c.urlParams.visibility = "private", b.addClass("on"), this.$(".ftag-public").removeClass("on"))), c.fetch()
            }, j.prototype.disable_click_when_bulk_edit = function(a) {
                return a.preventDefault()
            }, j.prototype.update_bundle_tags = function(a) {
                var b, c = this;
                return b = this.subview("relatedtags"), b ? a.fetchTags(this.model.get("username"), a.get("name")).done(function() {
                    var d;
                    return d = _.map(a.get("tags"), function(a) {
                        return {
                            name: a
                        }
                    }), _.isEmpty(d) ? c.$(".relatedtags-wrapper").addClass("is-visible").slideUp(200) : (b.is_bundle=!0, b.collection.reset(d), c.$(".relatedtags-wrapper").addClass("is-visible").slideDown(200).animate({
                        height: c.$(".relatedtags").outerHeight() + 20
                    }, 200))
                }) : void 0
            }, j.prototype.update_related_tags = function() {
                var a, b, c;
                return b = this.subview("relatedtags"), b && (a = this.subview("links").collection, !_.isEmpty(null != a ? a.urlParams.tags : void 0)) ? (c = a.related_tags(), c.isEmpty() ? this.$(".relatedtags-wrapper").removeClass("is-visible").slideUp(200) : (b.is_bundle=!1, b.collection.reset(c.toJSON()), this.$(".relatedtags-wrapper").addClass("is-visible").slideDown(200).animate({
                    height: this.$(".relatedtags").outerHeight() + 20
                }, 200))) : void 0
            }, j.prototype.sort_tag_link_list = function(b) {
                var c, d, e;
                return this.$(".actions li").removeClass("on"), null != b && "function" == typeof b.preventDefault && b.preventDefault(), c = $(b.currentTarget), d = this.subview("links").collection, e = d.urlParams.hashtag || _.first(d.urlParams.tags), this.reset_links(d), "alltime" === c.attr("data-tab") ? (d.urlParams.hashtag = null, d.urlParams.sortBy = null, d.urlParams.type = "everyone", d.urlParams.tags = [e]) : (d.urlParams.sortBy = c.attr("data-tab"), d.urlParams.hashtag = e, d.urlParams.type = null, d.urlParams.tags = []), d.fetch(), c.parent().addClass("on"), a.mediator.execute("router:changeURL", c.attr("href")), !1
            }, j.prototype.change_search_type = function(b) {
                var c, d;
                return this.$(".actions li").removeClass("on"), null != b && "function" == typeof b.preventDefault && b.preventDefault(), c = $(b.currentTarget), d = this.subview("links").collection, this.reset_links(d), d.urlParams.keywords = this.model.get("q"), d.urlParams.type = c.attr("data-tab"), d.fetch(), c.parent().addClass("on"), a.mediator.execute("router:changeURL", c.attr("href")), !1
            }, j.prototype.get_selected_md5s = function() {
                return _.map(this.$(".item[data-md5].bulk-edit-selected"), function(a) {
                    return $(a).attr("data-md5")
                })
            }, j.prototype.get_selected_links = function() {
                var a, b;
                return b = this.get_selected_md5s(), a = this.subview("links").collection, new f(_.map(b, function(b) {
                    return a.findWhere({
                        md5: b
                    })
                }))
            }, j.prototype._bulk_action_mark = function(a, b, c) {
                var d, e, f = this;
                return this.startLoading(), d = _.first(c, b.bulk_size), e = _.rest(c, b.bulk_size), b.markAs(a, _.map(d, function(a) {
                    return a.get("md5")
                }).join(",")).done(function() {
                    return _.each(d, function(b) {
                        return b.set({
                            "private": "private" === a
                        })
                    }), _.isEmpty(e) ? f.stopLoading() : f._bulk_action_mark(a, b, e)
                })
            }, j.prototype.bulk_action_mark = function(a) {
                var b, c, d, f = this;
                return c = this.get_selected_md5s(), _.isEmpty(c) ? void 0 : (d = _.map(c, function(a) {
                    return f.subview("links").collection.findWhere({
                        md5: a
                    })
                }), b = new e, this._bulk_action_mark(a, b, d))
            }, j.prototype.bulk_action_public = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.bulk_action_mark("public"), ga("send", "event", "links", "public")
            }, j.prototype.bulk_action_private = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.bulk_action_mark("private"), ga("send", "event", "links", "private")
            }, j.prototype._bulk_action_remove = function(a, b) {
                var c, d, e = this;
                return this.startLoading(), c = _.first(a, b.bulk_size), d = _.rest(a, b.bulk_size), b.deleteAll(c.join(",")).done(function() {
                    var c;
                    return c = e.subview("links").collection, _.each(a, function(a) {
                        return c.remove(c.findWhere({
                            md5: a
                        }))
                    }), _.isEmpty(d) ? (e.stopLoading(), e.bulk_action_none()) : e._bulk_action_remove(d, b)
                })
            }, j.prototype.bulk_action_remove = function(a) {
                var b, c, d, f;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), d = this.get_selected_md5s(), b = d.length, b > 0 && (f = b > 1 ? "Are you sure you want to delete these " + b + " links?" : "Are you sure you want to delete this link?", confirm(f) && (c = new e, this._bulk_action_remove(d, c))), ga("send", "event", "links", "remove")
            }, j.prototype.bulk_action_all = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.$(".item[data-md5]").addClass("bulk-edit-selected"), this.$(".bulk-edit-selection-count").text(this.$(".item[data-md5].bulk-edit-selected").length)
            }, j.prototype.bulk_action_none = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.$(".item[data-md5]").removeClass("bulk-edit-selected"), this.$(".bulk-edit-selection-count").text(0)
            }, j.prototype.update_selection = function(a) {
                var b, c;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), b = a.currentTarget, $(b).toggleClass("bulk-edit-selected"), c = this.$(".item[data-md5].bulk-edit-selected").length, this.$(".bulk-edit-selection-count").text(c || 0)
            }, j.prototype.toggle_bulk_edit = function(a) {
                var b, c, d;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.editable ? ($(".content-wrapper").toggleClass("bulk-edit-on"), d = this.$(".links"), d.toggleClass("bulk-edit-on"), b = this.$(".bulk-edit"), c = b.text(), b.text("Edit" === c ? "Done" : "Edit"), this.$("ul.tags .dropdown-toggle").toggleClass("disabled"), this.bulk_action_none(), this.$(".hero-unit").stop(!0, !0).animate({
                    height: "toggle",
                    paddingTop: "toggle",
                    paddingBottom: "toggle",
                    opacity: "toggle"
                }, 300), setTimeout(function() {
                    return this.$(".bulk-edit-status").stop(!0, !0).animate({
                        height: "toggle",
                        paddingTop: "toggle",
                        paddingBottom: "toggle",
                        opacity: "toggle"
                    }, 200)
                }, 300), this.$(".relatedtags-wrapper").hasClass("is-visible") && this.$(".relatedtags-wrapper").slideToggle(200), this.$(".tagcloud-wrapper").hasClass("is-visible") && this.$(".tagcloud-wrapper").slideToggle(200), this.$(".tagbundles-wrapper").hasClass("is-visible") && this.$(".tagbundles-wrapper").slideToggle(200), d.hasClass("bulk-edit-on") ? ga("send", "event", "links", "bulk-edit") : void 0) : void 0
            }, j.prototype.remove_filters = function(b) {
                var c;
                return null != b && b.preventDefault(), c = this.subview("links").collection, this.reset_links(c), c.fetch(), this.$(".filter .filter-cards").html(""), this.$(".filter").slideUp(200), this.$(".links-search-form input").val(""), this.$(".relatedtags-wrapper").removeClass("is-visible").slideUp(200), a.mediator.execute("router:changeURL", c.to_user_url()), this.publishEvent("sidebar:change", {
                    action: "page",
                    location: c.to_search_string()
                }), ga("send", "event", "links", "remove-filters")
            }, j.prototype.remove_filter = function(b) {
                var c, d, e, f, g, h = this;
                return b.preventDefault(), c = $(b.currentTarget), g = c.attr("data-filter"), d = c.text(), c.remove(), _.isEmpty(this.$(".filter-card")) ? void this.remove_filters() : (f = this.subview("links").collection, f.reset(), "tagbundle" === g && (f.urlParams.tagBundle = null), "search" === g && (f.urlParams.keywords = []), "tag" === g && (f.urlParams.tags = _.reject(f.urlParams.tags, function(a) {
                    return a === d.substring(1)
                }), e=!0), f.fetch().done(function() {
                    return e ? h.update_related_tags() : void 0
                }), a.mediator.execute("router:changeURL", f.to_user_url()), this.publishEvent("sidebar:change", {
                    action: "page",
                    location: f.to_search_string()
                }), ga("send", "event", "links", "reset-filter"))
            }, j.prototype.show_filter = function(a, b) {
                var c;
                return _.isArray(a) || (a = [a]), c = function(a) {
                    return "<a class='filter-cancel filter-card' href='#' data-filter='" + b + "'>" + _.escape(a) + "</a>"
                }, this.$(".filter .filter-cards").html(_.map(a, c).join(" ")), this.$(".filter").slideDown(200)
            }, j.prototype.append_filter = function(a, b) {
                return this.$(".filter .filter-cards").append(" <a class='filter-cancel filter-card' href='#' data-filter='" + b + "'>" + _.escape(a) + "</a>"), this.$(".filter").slideDown(200)
            }, j.prototype.filter_by_tag = function(b) {
                var c, d, e, f = this;
                return b.preventDefault(), d = b.currentTarget, e = $(d).attr("data-tag"), c = this.subview("links").collection, this.reset_links(c), c.urlParams.tags = [e], c.fetch().done(function() {
                    return f.update_related_tags()
                }), this.show_filter("#" + e, "tag"), this.bulk_action_none(), this.$(".link-nav-fixed-wrapper").hasClass("bulk-edit-on") && $(".links").addClass("bulk-edit-on"), a.mediator.execute("router:changeURL", c.to_user_url()), this.publishEvent("sidebar:change", {
                    action: "page",
                    location: c.to_search_string()
                }), ga("send", "event", "links", "tag")
            }, j.prototype.filter_by_appending_tag = function(b) {
                var c, d, e, f, g = this;
                return b.preventDefault(), e = b.currentTarget, f = $(e).attr("data-tag"), d = this.subview("links").collection, c = d.urlParams.tags || [], _.contains(c, f) ? void 0 : (c.push(f), this.reset_links(d), d.urlParams.tags = c, d.fetch().done(function() {
                    return g.update_related_tags()
                }), this.append_filter("#" + f, "tag"), this.bulk_action_none(), this.$(".link-nav-fixed-wrapper").hasClass("bulk-edit-on") && $(".links").addClass("bulk-edit-on"), a.mediator.execute("router:changeURL", d.to_user_url()), this.publishEvent("sidebar:change", {
                    action: "page",
                    location: d.to_search_string()
                }), _.isEmpty($(e).parents("ul.relatedtags")) ? ga("send", "event", "links", "tag-filter") : ga("send", "event", "links", "related-tags"))
            }, j.prototype.filter_by_tag_bundle = function(b) {
                var c, d, e;
                return b.preventDefault(), d = b.currentTarget, e = $(d).attr("data-bundle"), c = this.subview("links").collection, this.reset_links(c), c.urlParams.tagBundle = e, c.fetch({
                    reset: !0
                }), this.show_filter("[" + e + "]", "tagbundle"), this.bulk_action_none(), this.$(".link-nav-fixed-wrapper").hasClass("bulk-edit-on") && $(".links").addClass("bulk-edit-on"), this.update_bundle_tags(new g({
                    name: e
                })), a.mediator.execute("router:changeURL", c.to_user_url()), this.publishEvent("sidebar:change", {
                    action: "page",
                    location: c.to_search_string()
                }), ga("send", "event", "links", "bundle")
            }, j.prototype.filter_by_search = function(b) {
                var c, d;
                return b.preventDefault(), c = this.$(".links-search-form input").val(), _.isEmpty(c) ? void 0 : (d = this.subview("links").collection, this.reset_links(d), d.urlParams.keywords = c, d.fetch({
                    reset: !0
                }), this.show_filter("“" + c + "”", "search"), a.mediator.execute("router:changeURL", d.to_user_url()), this.publishEvent("sidebar:change", {
                    action: "page",
                    location: d.to_search_string()
                }), ga("send", "event", "links", "search"))
            }, j.prototype.reset_links = function(a) {
                return a.urlParams.tags = [], a.urlParams.keywords = [], a.urlParams.tagBundle = null, a.urlParams.visibility = null, a.urlParams.noTag=!1, a.reset()
            }, j.prototype.toggle_tagcloud = function(a) {
                return null != a && a.preventDefault(), this.$(".tagbundles-wrapper").removeClass("is-visible").slideUp(200), this.$(".datetable-wrapper").removeClass("is-visible").slideUp(200), setTimeout(function() {
                    return this.$(".tagcloud-wrapper").toggleClass("is-visible").slideToggle(200)
                }, 100), this.$(".action-tagbundles-wrapper").removeClass("on"), this.$(".action-tags-wrapper").toggleClass("on")
            }, j.prototype.toggle_tagbundles = function(a) {
                return null != a && a.preventDefault(), this.$(".tagcloud-wrapper").removeClass("is-visible").slideUp(200), this.$(".datetable-wrapper").removeClass("is-visible").slideUp(200), setTimeout(function() {
                    return this.$(".tagbundles-wrapper").toggleClass("is-visible").slideToggle(200)
                }, 100), this.$(".action-tags-wrapper").removeClass("on"), this.$(".action-tagbundles-wrapper").toggleClass("on")
            }, j.prototype.toggle_datetable = function(a) {
                return null != a && a.preventDefault(), this.$(".tagcloud-wrapper").removeClass("is-visible").slideUp(200), this.$(".tagbundles-wrapper").removeClass("is-visible").slideUp(200), setTimeout(function() {
                    return this.$(".datetable-wrapper").toggleClass("is-visible").slideToggle(200)
                }, 100), this.$(".action-tags-wrapper").removeClass("on"), this.$(".action-tagbundles-wrapper").removeClass("on")
            }, j.prototype.bulk_add_tags = function(a) {
                return a.preventDefault(), new h({
                    collection: this.get_selected_links()
                }), ga("send", "event", "links", "bulk-add-tags")
            }, j.prototype.bulk_remove_tags = function(a) {
                return a.preventDefault(), new i({
                    collection: this.get_selected_links()
                }), ga("send", "event", "links", "bulk-remove-tags")
            }, j.prototype.sort_tags = function(a) {
                return this.subview("tagcloud").sort_tags(a)
            }, j.prototype.edit_tag = function(a) {
                return this.subview("tagcloud").edit_tag(a)
            }, j.prototype.show_tagbundle_dialog = function(a) {
                return this.subview("tagbundles").show_tagbundle_dialog(a)
            }, j
        }(d)
    })
}.call(this), function() {
    define("helpers/links_stats_graph", [], function() {
        var a;
        return a = function(a) {
            return ("devicePixelRatio"in window ? window.devicePixelRatio > 1 && a.webkitBackingStorePixelRatio < 2 : void 0) ? window.devicePixelRatio : 1
        }, function(a, b, c) {
            var d, e, f, g, h, i;
            if (c || (c = {}), null != b) {
                try {
                    d = a[0].getContext("2d")
                } catch (j) {
                    return void(e = j)
                }
                return h = a.width() * window.devicePixelRatio, f = a.height() * window.devicePixelRatio, d.canvas.width = h, d.canvas.height = f, d.clearRect(0, 0, h, f), g = Math.max(_.max(b), c.max || 20), i = h / b.length, d.fillStyle = "rgb(238, 238, 238)", _.each(b, function(a, b) {
                    var c, e, h, j;
                    return h = f * (1 - a / g), e = i * b, c = f * (a / g), j = i, d.fillRect(e, h, j, c)
                })
            }
        }
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/user", ["chaplin", "lib/session", "views/base/hbsview", "helpers/links_stats_graph"], function(a, c, d, e) {
        var f, g;
        return f = function(a) {
            function d() {
                return g = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "links/user", d.prototype.autoRender=!0, d.prototype.autoAttach=!0, d.prototype.initialize = function() {
                return d.__super__.initialize.apply(this, arguments), this.subscribeEvent("login:success", this.render), this.listenTo(this.model, "change", this.render)
            }, d.prototype.events = {
                "click a[href=#unfollow]": "unfollow",
                "click a[href=#follow]": "follow"
            }, d.prototype.unfollow = function(a) {
                var b;
                return a.preventDefault(), b = $(a.currentTarget), c.user ? (b.html(b.html() + "..."), c.user.unfollowUser(this.model).done(function() {
                    return b.attr("href", "#follow").html("<span class='follow-text''>Follow</span>").removeClass("follow-toggle-following").addClass("follow-toggle-not-following")
                })) : void 0
            }, d.prototype.follow = function(a) {
                var b;
                return a.preventDefault(), b = $(a.currentTarget), b.html(b.html() + "..."), c.user.followUser(this.model).done(function() {
                    return b.attr("href", "#unfollow").html("<span class='following-text'>Following</span><span class='unfollow-text'>Unfollow</span>").removeClass("follow-toggle-not-following").addClass("follow-toggle-following")
                }), ga("send", "event", "user", "follow", this.model.get("username"))
            }, d.prototype.render = function() {
                var a = this;
                return d.__super__.render.apply(this, arguments), this.model.get("weekly_save_trend") ? setTimeout(function() {
                    return a.$("#links_stats").length > 0 ? e(a.$("#links_stats"), a.model.get("weekly_save_trend"), {
                        max: 20
                    }) : void 0
                }, 1987) : void 0
            }, d
        }(d)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/base/collection_view", ["chaplin", "views/base/hbsview"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.autoRender=!0, d.prototype.getTemplateFunction = c.prototype.getTemplateFunction, d.prototype.defaults = {
                heading: !1,
                headingLevel: 2,
                showHeadingIfEmpty: !0
            }, d.prototype.isMobile = c.prototype.isMobile, d.prototype.isDesktop = c.prototype.isDesktop, d.prototype.isTouch = c.prototype.isTouch, d.prototype.isiPhone = c.prototype.isiPhone, d.prototype.animationDuration = 0, d.prototype.previousLoadCount = 0, d.prototype.initialize = function(a) {
                return this.options = null != a ? a : {}, _.defaults(this.options, this.defaults), d.__super__.initialize.apply(this, arguments)
            }, d.prototype.startLoading = c.prototype.startLoading, d.prototype.stopLoading = c.prototype.stopLoading, d.prototype.loadMore = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.collection.length > this.previousLoadCount ? (this.previousLoadCount = this.collection.length, this.collection.fetch()) : void 0
            }, d
        }(a.CollectionView)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/widgets/person_overview", ["views/base/hbsview", "models/tags"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.autoRender=!0, d.prototype.autoAttach=!1, d.prototype.template = "widgets/person_overview", d.prototype.loaded=!1, d.prototype.tags_html = "...", d.prototype.initialize = function(a) {
                return d.__super__.initialize.apply(this, arguments), this.anchor = a.anchor, this.listenTo(this.model, "change", this.render)
            }, d.prototype.render_more_info = function() {
                var a = this;
                return this.model.profile().done(function() {
                    var b;
                    return a.loaded ? void 0 : (b = new c([], {
                        urlParams: {
                            username: a.model.get("username")
                        }
                    }), b.fetch().done(function() {
                        var c, d;
                        return b.sort_by_count(), c = b.first(3), d = c.map(function(a) {
                            return "<li>" + a.get("name") + "</li>\n"
                        }).join(""), a.tags_html = "<ul class=tags>" + d + "</ul>", a.render(), a.loaded=!0
                    }).fail(function() {
                        return a.tags_html = "", a.render(), a.loaded=!0
                    }))
                })
            }, d.prototype.getTemplateData = function() {
                var a;
                return a = d.__super__.getTemplateData.apply(this, arguments), a.tags_html = this.tags_html, a
            }, d.prototype.render = function() {
                var a;
                return d.__super__.render.apply(this, arguments), $('div.person-overview[data-user="' + this.model.get("username") + '"]').parent().html(this.$el.html()), null != (a = this.anchor.data("bs.popover")) ? a.options.content = this.$el.html() : void 0
            }, d.prototype.attach = function() {
                var a = this;
                return this.anchor.popover({
                    content: this.$el.html(),
                    html: !0,
                    placement: "right",
                    trigger: "hover",
                    delay: 300,
                    container: $("#delicious"),
                    animation: !1
                }), this.anchor.on("show.bs.popover", function() {
                    return a.loaded ? void 0 : a.render_more_info()
                })
            }, d.prototype.dispose = function() {
                return this.anchor.off("show.bs.popover"), this.anchor.popover("destroy"), $(".popover").remove(), d.__super__.dispose.apply(this, arguments)
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/link", ["views/base/hbsview", "views/modals/compose_link", "chaplin", "lib/session", "helpers/login_deferred", "views/widgets/person_overview", "models/user"], function(a, c, d, e, f, g, h) {
        var i, j;
        return i = function(a) {
            function d() {
                return j = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "links/link", d.prototype.className = "link", d.prototype.initialize = function() {
                return d.__super__.initialize.apply(this, arguments), this.listenTo(this.model, "change", this.render), this.listenTo(this.model, "destory", this.dispose)
            }, d.prototype.events = {
                "click a.edit": "show_compose_dialog",
                "click a.bookmark-btn": "show_compose_dialog",
                "click a.mobile-tags-toggle": "show_mobile_tags",
                "dblclick div.item": "show_compose_dialog",
                "click a.title": "track_click",
                "click a.delete": "submit_delete",
                "click a.share-social": "share_popup"
            }, d.prototype.show_compose_dialog = function(a) {
                var b, d = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), e.loading_compose ? void 0 : e.user.get("isLoggedIn") ? (e.loading_compose=!0, this.startLoading(), this.model.set({
                    replace: this.model.get("editable")
                }), b = this.model.get("editable") ? this.model : this.model.clone(), this.model.get("editable") || b.set({
                    tags: [],
                    note: "",
                    time_created: null
                }), b.compose().done(function() {
                    return d.stopLoading(), e.loading_compose=!1, new c({
                        model: b
                    })
                }), this.model.get("editable") ? ga("send", "event", "link", "edit") : ga("send", "event", "link", "pick", this.model.get("url"))) : void(new f).prompt().done(function() {
                    return d.show_compose_dialog()
                })
            }, d.prototype.render = function() {
                var a, b;
                return a = null != (b = this.$(".item")) ? b.hasClass("bulk-edit-selected") : void 0, this.$(".link-rank").tooltip("destroy"), this.subview("first_saver_overview") && this.subview("first_saver_overview").dispose(), d.__super__.render.apply(this, arguments), this.$el.attr("data-md5", this.model.get("md5")), this.$("a.early-bird") && this.create_first_saver_overview(), a && this.$(".item").addClass("bulk-edit-selected"), this.$(".link-rank").tooltip({
                    animation: !1
                })
            }, d.prototype.attach = function() {
                return d.__super__.attach.apply(this, arguments), $(".links").hasClass("bulk-edit-on") ? this.$("ul.tags .dropdown-toggle").addClass("disabled") : void 0
            }, d.prototype.dispose = function() {
                return this.$(".link-rank").tooltip("destroy"), d.__super__.dispose.apply(this, arguments)
            }, d.prototype.create_first_saver_overview = function() {
                var a;
                return a = new g({
                    model: new h(this.model.get("first_saver")),
                    anchor: this.$("a.early-bird")
                }), a.attach(), this.subview("first_saver_overview", a)
            }, d.prototype.show_mobile_tags = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.$(".mobile-tags-toggle").toggleClass("on"), this.$(".tags").toggleClass("on")
            }, d.prototype.track_click = function() {
                return ga("send", "event", "link", "click", this.model.get("url"))
            }, d.prototype.submit_delete = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), confirm("Are you sure to delete this link?") ? (this.$("a.delete").text("Deleting...").attr("disabled", "disabled"), this.model.remove()) : void 0
            }, d.prototype.share_popup = function(a) {
                var b, c, d, e, f, g, h, i, j, k;
                return b = $(a.currentTarget), j = b.attr("href"), /^mailto/.test(j) ? void 0 : (a.preventDefault(), k = 575, d = 400, c = $(window), e = (c.width() - k) / 2, i = (c.height() - d) / 2, j.indexOf("text=")>-1 && (g = j.substring(j.indexOf("text=") + 5, j.indexOf("&url=")), f = encodeURIComponent(j.substring(j.indexOf("text=") + 5, j.indexOf("&url="))), j = j.replace(g, f)), h = "status=1,width=" + k + ",height=" + d + ",top=" + i + ",left=" + e, window.open(j, "Share from Delicious", h))
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/list", ["chaplin", "views/base/collection_view", "views/links/link"], function(a, c, d) {
        var e, f;
        return e = function(a) {
            function c() {
                return f = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "links/list", c.prototype.itemView = d, c.prototype.className = ".links", c.prototype.listSelector = ".link-list", c.prototype.previousLoadCount = 0, c.prototype.editable=!1, c.prototype.by_user=!1, c.prototype.initialize = function(a) {
                return c.__super__.initialize.call(this, a), this.editable = a.editable, this.by_user = a.by_user ||!1, this.owner = a.owner, this.subscribeEvent("links:load-trigger", this.loadMore), this.subscribeEvent("links:load-finish", this.loadFinish), this.subscribeEvent("links:load-start", this.loadStart), this.editable && this.subscribeEvent("links:new", this.refresh), this.subscribeEvent("links:deleted", this.linksRemoved), this.listenTo(this.collection, "reset", this.render)
            }, c.prototype.events = {
                "click a[href='#loadmore']": "loadMore"
            }, c.prototype.refresh = function() {
                return this.collection.reset(), this.collection.fetch()
            }, c.prototype.loadStart = function() {
                return this.startLoading(), this.$(".load-more-link").hide()
            }, c.prototype.renderItem = function(a) {
                var b;
                return b=!_.isEmpty(this.collection.urlParams.tags), a.set({
                    editable: this.editable,
                    by_user: this.by_user,
                    owner: this.owner,
                    filtering: b
                }), c.__super__.renderItem.call(this, a)
            }, c.prototype.loadFinish = function() {
                return 0 === this.collection.length && this.$el.html("<div class='no-link-found'>No links found.</div>"), this.stopLoading(), this.previousLoadCount !== this.collection.length && this.collection.length >= 20 ? this.$(".load-more-link").show() : void 0
            }, c.prototype.linksRemoved = function(a) {
                return this.previousLoadCount -= a
            }, c
        }(c)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/tag_editor", ["views/base/modal", "models/tag"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "modals/tag_editor", c.prototype.events = {
                "click button.rename-tag": "rename_tag",
                "click button.delete-tag": "delete_tag"
            }, c.prototype.getTemplateData = function() {
                var a;
                return a = c.__super__.getTemplateData.apply(this, arguments), _.sortBy(a.items, function(a) {
                    return a.name.toLowerCase()
                })
            }, c.prototype.rename_tag = function(a) {
                var b, c, d, e = this;
                return a.preventDefault(), b = $(a.currentTarget), b.prop("disabled", "disabled"), d = this.collection.find(function(a) {
                    return a.get("name") === this.$("option:selected").text()
                }), c = this.$(".new-tag-name").val().trim(), ga("send", "event", "tagcloud", "rename", "#{tag}, #{new_tag_name}"), c ? d.rename(c).done(function() {
                    var a;
                    return e.$(".text-success").text("successfully."), a = _.find(e.$("option"), function(a) {
                        return $(a).text() === c
                    }), a ? (e.$("option:selected").remove(), $(a).prop("selected", !0)) : e.$("option:selected").text(c), b.prop("disabled", ""), _.delay(function() {
                        return e ? e.$(".text-success").text("") : void 0
                    }, 5e3)
                }) : void 0
            }, c.prototype.delete_tag = function(a) {
                var b, c, d, e = this;
                return a.preventDefault(), b = $(a.currentTarget), b.prop("disabled", "disabled"), c = this.$("option:selected").text(), ga("send", "event", "tagcloud", "delete", c), confirm('Are you sure to remove tag "' + c + '"?') ? (d = this.collection.find(function(a) {
                    return a.get("name") === c
                }), d.remove().done(function() {
                    return e.$("option:selected").remove(), e.$(".text-success").text('"' + c + '" deleted.'), b.prop("disabled", ""), _.delay(function() {
                        return e ? e.$(".text-success").text("") : void 0
                    }, 5e3)
                })) : void 0
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/tagcloud", ["handlebars", "views/base/hbsview", "models/tags", "views/modals/tag_editor"], function(a, c, d, e) {
        var f, g, h, i;
        return f = function(c) {
            function d() {
                return h = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, c), d.prototype.autoRender=!0, d.prototype.autoAttach=!0, d.prototype.initialize = function(a) {
                return d.__super__.initialize.apply(this, arguments), this.parent = a.parent
            }, d.prototype.getTemplateFunction = function() {
                return a.compile("{{tagcloud this}}")
            }, d.prototype.getTemplateData = function() {
                return this.parent.getTagsForDisplay()
            }, d
        }(c), g = function(a) {
            function c() {
                return i = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "links/tagcloud", c.prototype.autoRender=!1, c.prototype.autoAttach=!0, c.prototype.display_limit = 100, c.prototype.show_all=!1, c.prototype.filter = null, c.prototype.initialize = function(a) {
                return c.__super__.initialize.call(this, a), this.editable = a.editable, this.listenTo(this.collection, "sort", this.render), this.listenTo(this.collection, "change", this.render), this.subscribeEvent("tags:managed", this.render)
            }, c.prototype.sort_tags = function(a) {
                var b;
                return a.preventDefault(), b = $(a.currentTarget), b.hasClass("sort-by-alphabet") ? (this.collection.sort_by_name(!0), ga("send", "event", "tagcloud", "sort", "alpha")) : (this.collection.sort_by_count(!0), ga("send", "event", "tagcloud", "sort", "count")), this.collection.sort()
            }, c.prototype.edit_tag = function(a) {
                return a.preventDefault(), ga("send", "event", "tagcloud", "edit"), new e({
                    collection: this.collection
                })
            }, c.prototype.filter_tags = function(a, b) {
                return b = b.toLowerCase(), _.filter(a, function(a) {
                    return 0 === a.name.toLowerCase().indexOf(b)
                })
            }, c.prototype.getTemplateData = function() {
                var a;
                return a = c.__super__.getTemplateData.apply(this, arguments), this.filter ? (a.filter = this.filter, a.has_more=!1) : a.length > this.display_limit&&!this.show_all && (a.has_more=!0), a
            }, c.prototype.getTagsForDisplay = function() {
                var a;
                return a = this.getTemplateData(), this.filter ? a.items = this.filter_tags(a.items, this.filter) : a.length > this.display_limit&&!this.show_all && (a.items = _.first(a.items, this.display_limit)), a
            }, c.prototype.render = function() {
                return c.__super__.render.apply(this, arguments), this.subview("cloud-core", new f({
                    parent: this,
                    el: this.$(".tagcloud-core")
                }))
            }, c.prototype.update_tag_filter = function() {
                var a;
                return a = this.$("input.tag-filter").val(), a !== this.filter ? (this.filter = a, this.subview("cloud-core").render()) : void 0
            }, c.prototype.events = {
                "click a.show-all-tags": "show_all_tags",
                "keyup input.tag-filter": "update_tag_filter"
            }, c.prototype.show_all_tags = function(a) {
                return null != a && a.preventDefault(), this.show_all=!0, this.render()
            }, c
        }(c)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    }, c = [].indexOf || function(a) {
        for (var b = 0, c = this.length; c > b; b++)
            if (b in this && this[b] === a)
                return b;
        return - 1
    };
    define("views/modals/tagbundle", ["chaplin", "lib/session", "views/base/modal", "views/widgets/tag_input"], function(a, d, e, f) {
        var g, h;
        return g = function(a) {
            function e() {
                return h = e.__super__.constructor.apply(this, arguments)
            }
            return b(e, a), e.prototype.template = "modals/tagbundle", e.prototype.bundle_name_blacklist = [], e.prototype.initialize = function(a) {
                var b = this;
                return e.__super__.initialize.apply(this, arguments), this.bundle_name_blacklist = a.bundle_names, this.model.get("name") ? this.bundle_name_blacklist = _.reject(this.bundle_name_blacklist, function(a) {
                    return a === b.model.get("name")
                }) : void 0
            }, e.prototype.events = {
                "click button.text-btn-primary": "save_tagbundle",
                "click button.text-btn-danger": "delete_tagbundle",
                "submit form": "save_tagbundle"
            }, e.prototype.save_tagbundle = function(a) {
                var b, d, e, f = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), b = this.$("input[name='name']").val(), e = this.subview("taginput").get_values(), c.call(this.bundle_name_blacklist, b) >= 0 ? (this.$("span.text-danger").text("A bundle with that name already exists."), void _.delay(function() {
                    return f.el ? f.$("span.text-danger").text("") : void 0
                }, 3e3)) : e.trim() ? (this.$("button.text-btn-primary").text("Saving...").prop("disabled", "disabled"), this.model.get("name") && (d = this.model.clone()), this.model.save({
                    name: b,
                    tags: e,
                    previous: d
                }).done(function() {
                    return f.$el.modal("hide"), d ? f.publishEvent("tagbundles:change") : f.publishEvent("tagbundles:add", f.model)
                }).fail(function(a) {
                    return f.$("button.text-btn-primary").text("Save").prop("disabled", ""), f.$("span.text-danger").text(a.pkg), _.delay(function() {
                        return f.el ? f.$("span.text-danger").text("") : void 0
                    }, 3e3)
                })) : (this.$("span.text-danger").text("Cannot create a bundle without tags."), void _.delay(function() {
                    var a;
                    return f.el && null != (a = f.$("span.text-danger")) ? a.text("") : void 0
                }, 3e3))
            }, e.prototype.delete_tagbundle = function(a) {
                var b = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), ga("send", "event", "tagbundle", "delete"), confirm("Are you sure to remove this bundle?") ? (this.$("button.text-btn-danger").text("Deleting...").attr("disabled", "disabled"), this.model["delete"]().done(function() {
                    return b.$el.modal("hide")
                })) : void 0
            }, e.prototype.render = function() {
                return e.__super__.render.apply(this, arguments), this.subview("taginput", new f({
                    el: this.$(".tags-input-container"),
                    tags: d.user.get("tags"),
                    name: "tags",
                    value: this.model.get("tags"),
                    multiline: !0
                }))
            }, e
        }(e)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/tagbundles", ["views/base/hbsview", "views/modals/tagbundle", "models/tag_bundle", "chaplin", "lib/session"], function(a, c, d, e, f) {
        var g, h;
        return g = function(a) {
            function e() {
                return h = e.__super__.constructor.apply(this, arguments)
            }
            return b(e, a), e.prototype.template = "links/tagbundles", e.prototype.autoRender=!0, e.prototype.autoAttach=!0, e.prototype.events = {
                "click a.edit": "show_tagbundle_dialog"
            }, e.prototype.initialize = function(a) {
                return e.__super__.initialize.call(this, a), this.editable = a.editable, this.listenTo(this.collection, "reset", this.render), this.listenTo(this.collection, "add", this.render), this.listenTo(this.collection, "remove", this.render), this.subscribeEvent("tagbundles:change", this.render), this.subscribeEvent("tagbundles:add", this.on_new_bundle_created)
            }, e.prototype.getTemplateData = function() {
                var a;
                return a = e.__super__.getTemplateData.apply(this, arguments), a.editable = this.editable, a
            }, e.prototype.show_tagbundle_dialog = function(a) {
                var b, e, g, h = this;
                return "function" == typeof a.preventDefault && a.preventDefault(), (b = $(a.currentTarget).attr("data-bundle")) ? (e = this.collection.findWhere({
                    name: b
                }), e ? (ga("send", "event", "tagbundle", "edit"), e.fetchTags(null != (g = f.user) ? g.get("username") : void 0, b).done(function() {
                    return new c({
                        model: e,
                        bundle_names: h.collection.map(function(a) {
                            return a.get("name")
                        })
                    })
                })) : void 0) : (ga("send", "event", "tagbundle", "create"), new c({
                    model: new d,
                    bundle_names: this.collection.map(function(a) {
                        return a.get("name")
                    })
                }))
            }, e.prototype.on_new_bundle_created = function(a) {
                return this.collection.add(a), ga("send", "event", "tagbundle", "created")
            }, e
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/tag", ["views/base/hbsview", "helpers/links_stats_graph"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "links/tag", d.prototype.autoRender=!0, d.prototype.autoAttach=!0, d.prototype.events = {
                "click a[href=#unfollow]": "unsubscribe",
                "click a[href=#follow]": "subscribe"
            }, d.prototype.render = function() {
                var a = this;
                return d.__super__.render.apply(this, arguments), this.model.get("weekly_save_trend") ? setTimeout(function() {
                    return a.$("#links_stats").length > 0 ? c(a.$("#links_stats"), a.model.get("weekly_save_trend"), {
                        max: 750
                    }) : void 0
                }, 1987) : void 0
            }, d.prototype.unsubscribe = function(a) {
                var b;
                return a.preventDefault(), b = $(a.currentTarget), b.html(b.html() + "..."), this.model.unsubscribe().done(function() {
                    return b.attr("href", "#follow").html("<span class='follow-text''>Subscribe</span>").removeClass("follow-toggle-following").addClass("follow-toggle-not-following")
                })
            }, d.prototype.subscribe = function(a) {
                var b;
                return a.preventDefault(), b = $(a.currentTarget), b.html(b.html() + "..."), this.model.subscribe().done(function() {
                    return b.attr("href", "#unfollow").html("<span class='following-text'>Unsubscribed</span><span class='unfollow-text'>Unsubscribe</span>").removeClass("follow-toggle-not-following").addClass("follow-toggle-following")
                })
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/date", ["views/base/hbsview", "moment"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "links/date", d.prototype.autoRender=!0, d.prototype.autoAttach=!0, d.prototype.initialize = function(a) {
                var b;
                return d.__super__.initialize.apply(this, arguments), b = c.unix(a.since), this.range = {
                    since_year: b.year(),
                    since_month: b.month() + 1,
                    now_year: c().year(),
                    now_month: c().month() + 1
                }, this.listenTo(this.collection, "reset", this.clear_on)
            }, d.prototype.getTemplateData = function() {
                return this.range
            }, d.prototype.clear_on = function() {
                return this.$("a").removeClass("on")
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/welcome", ["chaplin", "views/base/hbsview", "helpers/ua"], function(a, c, d) {
        var e, f;
        return e = function(c) {
            function e() {
                return f = e.__super__.constructor.apply(this, arguments)
            }
            return b(e, c), e.prototype.template = "links/welcome", e.prototype.autoRender=!0, e.prototype.autoAttach=!0, e.prototype.initialize = function() {
                return e.__super__.initialize.apply(this, arguments), this.subscribeEvent("links:new", this.on_adding_new_link)
            }, e.prototype.on_adding_new_link = function() {
                return a.helpers.redirectTo({
                    url: "/"
                })
            }, e.prototype.getTemplateData = function() {
                return {
                    android: d.isAndroid(),
                    firefox: d.isDesktop() && d.isFirefox(),
                    chrome: d.isDesktop() && d.isChrome(),
                    ipad: d.isiPad(),
                    iphone: d.isiPhone(),
                    firefoxos: d.isFirefoxOS(),
                    desktop: d.isDesktop()
                }
            }, e
        }(c)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/network/page", ["views/base/pageview"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "network/page", c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/network/link", ["views/base/hbsview", "views/modals/compose_link", "chaplin", "lib/session"], function(a, c, d, e) {
        var f, g;
        return f = function(a) {
            function d() {
                return g = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "network/link", d.prototype.className = "link", d.prototype.events = {
                "dblclick .item": "show_compose_dialog",
                "click a.bookmark-btn": "show_compose_dialog",
                "click a.mobile-tags-toggle": "show_mobile_tags",
                "click a.title": "track_click"
            }, d.prototype.render = function() {
                return d.__super__.render.apply(this, arguments), this.$el.attr("data-md5", this.model.get("md5")), this
            }, d.prototype.getTemplateData = function() {
                var a;
                return a = d.__super__.getTemplateData.apply(this, arguments), a.saver = a.savers[0], a.owner = a.saver.username, a
            }, d.prototype.show_compose_dialog = function(a) {
                var b, d = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), e.loading_compose ? void 0 : (e.loading_compose=!0, this.startLoading(), b = this.model.clone(), b.get("editable") || b.set({
                    tags: [],
                    note: "",
                    time_created: null
                }), b.compose().done(function() {
                    return d.stopLoading(), e.loading_compose=!1, new c({
                        model: b
                    })
                }), ga("send", "event", "network", "save", this.model.get("url")))
            }, d.prototype.show_mobile_tags = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.$(".mobile-tags-toggle").toggleClass("on"), this.$(".tags").toggleClass("on")
            }, d.prototype.track_click = function() {
                return ga("send", "event", "network", "click", this.model.get("url"))
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/network/list", ["views/base/collection_view", "views/network/link", "lib/session"], function(a, c, d) {
        var e, f;
        return e = function(a) {
            function e() {
                return f = e.__super__.constructor.apply(this, arguments)
            }
            return b(e, a), e.prototype.template = "network/list", e.prototype.itemView = c, e.prototype.className = ".links", e.prototype.listSelector = ".link-list", e.prototype.previousLoadCount = 0, e.prototype.initialize = function(a) {
                return e.__super__.initialize.call(this, a), this.subscribeEvent("links:load-trigger", this.loadMore), this.subscribeEvent("links:load-finish", this.loadFinished), this.subscribeEvent("links:load-start", this.startLoading)
            }, e.prototype.loadFinished = function() {
                var a, b, c;
                return this.stopLoading(), 0 === this.collection.length ? this.$el.html("<div class='no-link-found'>There's no activity in your network. Try to <a href='/settings/friends'>find some friends</a> on Delicious.</div>") : !d.is_moz_sidebar && _.isEmpty(this.$(".ad-sponsors iframe")) ? (c = this.collection.getKeywords(), a = "/html/amazon.html?kw=" + c.join(";"), b = $("<iframe/>").attr("src", a).attr("width", 468).attr("height", 68).attr("frameBorder", 0).attr("scrolling", "no"), this.$(".ad-sponsors").append(b)) : void 0
            }, e
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/single", ["views/base/pageview"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "links/single", c.prototype.autoRender=!0, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/single/link", ["views/base/hbsview", "views/modals/compose_link", "chaplin", "lib/session", "helpers/login_deferred", "views/widgets/person_overview", "models/user"], function(a, c, d, e, f, g, h) {
        var i, j;
        return i = function(a) {
            function d() {
                return j = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "links/single/link", d.prototype.autoRender=!0, d.prototype.initialize = function() {
                return d.__super__.initialize.apply(this, arguments), this.listenTo(this.model, "change", this.render)
            }, d.prototype.events = {
                "click a.bookmark-btn": "show_compose_dialog",
                "click a.mobile-tags-toggle": "show_mobile_tags",
                "click a.share-social": "share_popup"
            }, d.prototype.show_compose_dialog = function(a) {
                var b, d = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), e.loading_compose ? void 0 : e.user.get("isLoggedIn") ? (b = this.model.clone(), b.get("editable") || b.set({
                    tags: [],
                    note: "",
                    time_created: null
                }), b.compose().done(function() {
                    return new c({
                        model: b
                    })
                })) : void(new f).prompt().done(function() {
                    return d.show_compose_dialog()
                })
            }, d.prototype.getTemplateData = function() {
                var a, b;
                return a = d.__super__.getTemplateData.apply(this, arguments), (null != (b = a.recommended) ? b.previously_saved : void 0) && (a.owner_avatar_url = e.user.get("avatar_url")), a
            }, d.prototype.render = function() {
                var a;
                return this.subview("person_overview") && this.subview("person_overview").dispose(), d.__super__.render.apply(this, arguments), this.model ? (a = new g({
                    model: new h(this.model.get("first_saver")),
                    anchor: this.$("a.early-bird")
                }), a.attach(), this.subview("person_overview", a)) : void 0
            }, d.prototype.show_mobile_tags = function(a) {
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.$(".mobile-tags-toggle").toggleClass("on"), this.$(".tags").toggleClass("on")
            }, d.prototype.share_popup = function(a) {
                var b, c, d, e, f, g, h, i, j, k;
                return b = $(a.currentTarget), j = b.attr("href"), /^mailto/.test(j) ? void 0 : (a.preventDefault(), k = 575, d = 400, c = $(window), e = (c.width() - k) / 2, i = (c.height() - d) / 2, j.indexOf("text=")>-1 && (g = j.substring(j.indexOf("text=") + 5, j.indexOf("&url=")), f = encodeURIComponent(j.substring(j.indexOf("text=") + 5, j.indexOf("&url="))), j = j.replace(g, f)), h = "status=1,width=" + k + ",height=" + d + ",top=" + i + ",left=" + e, window.open(j, "Share from Delicious", h))
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/single/comments", ["views/base/hbsview", "views/base/collection_view"], function(a, c) {
        var d, e, f, g;
        return d = function(a) {
            function c() {
                return f = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "links/single/comment", c.prototype.autoRender=!0, c.prototype.autoAttach=!0, c
        }(a), e = function(a) {
            function c() {
                return g = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "links/single/comments", c.prototype.itemView = d, c.prototype.className = ".comments", c.prototype.listSelector = ".comment-list", c
        }(c)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/note", ["views/base/hbsview"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "links/note", c.prototype.className = "user-note", c.prototype.attach = function() {
                return c.__super__.attach.apply(this, arguments), this.$("a").tooltip({
                    animation: !1,
                    html: !0
                })
            }, c.prototype.dispose = function() {
                return this.$("a").tooltip("destroy"), c.__super__.dispose.apply(this, arguments)
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/notes", ["views/base/collection_view", "views/links/note"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "links/notes", d.prototype.itemView = c, d.prototype.className = ".notes", d.prototype.listSelector = ".note-list", d.prototype.autoRender=!1, d.prototype.initialize = function() {
                return this.subscribeEvent("notes:load-finish", this.remove_loading)
            }, d.prototype.render = function() {
                return d.__super__.render.apply(this, arguments), 0 === this.collection.length ? this.$el.html("<div class='no-link-found'>We don't have any comments yet.</div>") : void 0
            }, d.prototype.remove_loading = function() {
                return this.$(".load-more").remove()
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/single/related", ["views/base/hbsview", "views/base/collection_view", "lib/session", "helpers/login_deferred", "views/modals/compose_link"], function(a, c, d, e, f) {
        var g, h, i, j;
        return g = function(a) {
            function c() {
                return i = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "links/single/related_link", c.prototype.autoRender=!0, c.prototype.autoAttach=!0, c.prototype.events = {
                "click a.bookmark-btn": "show_compose_dialog"
            }, c.prototype.show_compose_dialog = function(a) {
                var b, c = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), d.loading_compose ? void 0 : d.user.get("isLoggedIn") ? (b = this.model.clone(), b.get("editable") || b.set({
                    tags: [],
                    note: "",
                    time_created: null
                }), b.compose().done(function() {
                    return new f({
                        model: b
                    })
                })) : void(new e).prompt().done(function() {
                    return c.show_compose_dialog()
                })
            }, c
        }(a), h = function(a) {
            function c() {
                return j = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "links/single/related", c.prototype.autoRender=!0, c.prototype.autoAttach=!0, c.prototype.itemView = g, c.prototype.listSelector = ".link-list", c.prototype.initialize = function() {
                return c.__super__.initialize.apply(this, arguments), this.listenTo(this.collection, "reset", this.render)
            }, c
        }(c)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/links/related_tags", ["views/base/hbsview"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "links/related_tags", c.prototype.autoAttach=!0, c.prototype.initialize = function(a) {
                return c.__super__.initialize.apply(this, arguments), this.is_bundle = a.is_bundle, this.listenTo(this.collection, "reset", this.render)
            }, c.prototype.getTemplateData = function() {
                var a;
                return a = c.__super__.getTemplateData.apply(this, arguments), a.is_bundle = this.is_bundle, a
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/discover/page", ["views/base/pageview"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "discover/page", c.prototype.initialize = function() {
                return c.__super__.initialize.apply(this, arguments)
            }, c.prototype.getTemplateData = function() {
                var a;
                return a = c.__super__.getTemplateData.apply(this, arguments)
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/discover/link", ["views/base/hbsview", "models/link", "views/modals/compose_link", "chaplin", "lib/session", "views/widgets/person_overview", "models/user"], function(a, c, d, e, f, g, h) {
        var i, j;
        return i = function(a) {
            function e() {
                return j = e.__super__.constructor.apply(this, arguments)
            }
            return b(e, a), e.prototype.template = "discover/link", e.prototype.className = "link", e.prototype.domain_query = $("<a>"), e.prototype.getTemplateData = function() {
                var a, b;
                return a = e.__super__.getTemplateData.apply(this, arguments), a.tags = null != (b = a.tags) ? b.split(",") : void 0, a.tweet = _.first(a.publishers), a.domain = this.domain_query.attr("href", a.url).prop("hostname"), a
            }, e.prototype.render = function() {
                return e.__super__.render.apply(this, arguments), this.$el.attr("data-md5", this.model.get("md5")), this.$("a.early-bird") ? this.create_first_saver_overview() : void 0
            }, e.prototype.events = {
                "click .bookmark-btn": "show_compose_dialog",
                "click a.title": "track_click",
                "dblclick div.item": "show_compose_dialog",
                "click a.share-social": "share_popup"
            }, e.prototype.show_compose_dialog = function(a) {
                var b, e = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), b = new c({
                    url: this.model.get("url"),
                    title: this.model.get("title")
                }), f.loading_compose ? void 0 : (f.loading_compose=!0, this.startLoading(), b.compose().done(function() {
                    return e.stopLoading(), f.loading_compose=!1, new d({
                        model: b
                    })
                }), ga("send", "event", "discover", "save", this.model.get("url")))
            }, e.prototype.track_click = function() {
                return ga("send", "event", "discover", "click", this.model.get("url"))
            }, e.prototype.create_first_saver_overview = function() {
                var a;
                return a = new g({
                    model: new h({
                        username: this.model.get("publishers")[0].username,
                        full_name: this.model.get("publishers")[0].text
                    }),
                    anchor: this.$("a.early-bird")
                }), a.attach(), this.subview("first_saver_overview", a)
            }, e.prototype.share_popup = function(a) {
                var b, c, d, e, f, g, h, i, j, k;
                return b = $(a.currentTarget), j = b.attr("href"), /^mailto/.test(j) ? void 0 : (a.preventDefault(), k = 575, d = 400, c = $(window), e = (c.width() - k) / 2, i = (c.height() - d) / 2, j.indexOf("text=")>-1 && (g = j.substring(j.indexOf("text=") + 5, j.indexOf("&url=")), f = encodeURIComponent(j.substring(j.indexOf("text=") + 5, j.indexOf("&url="))), j = j.replace(g, f)), h = "status=1,width=" + k + ",height=" + d + ",top=" + i + ",left=" + e, window.open(j, "Share from Delicious", h))
            }, e
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/tag_search", ["chaplin", "lib/session", "views/base/hbsview", "views/base/modal", "models/tag", "models/tags"], function(a, c, d, e, f, g) {
        var h, i, j, k;
        return i = function(a) {
            function c() {
                return j = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.autoRender=!0, c.prototype.template = "user/relations/user_subscriptions", c.prototype.getTemplateData = function() {
                var a;
                return a = c.__super__.getTemplateData.apply(this, arguments), a.items
            }, c
        }(d), h = function(a) {
            function c() {
                return k = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "modals/tag_search", c.prototype.previous_tag = null, c.prototype.previous_fetch = null, c.prototype.dirty=!1, c.prototype.events = {
                "click a[href=#unfollow]": "unsubscribe",
                "click a[href=#follow]": "subscribe",
                "keyup #tag-query": "search_tag"
            }, c.prototype.search_tag = function() {
                var a, b = this;
                return a = this.$("#tag-query").val(), a.trim().length < 3 || a.trim() === this.previous_tag ? void this.$(".tag-search-result").html("") : (this.previous_tag = a.trim(), _.delay(function() {
                    var c;
                    return a === b.$("#tag-query").val().trim() ? (c = new g([], {
                        urlParams: {
                            type: "query",
                            query: a.trim()
                        }
                    }), b.previous_fetch && b.previous_fetch.reject(), b.previous_fetch = c.fetch(), b.previous_fetch.done(function() {
                        var a;
                        return b.previous_fetch = null, a = new i({
                            collection: c
                        }), b.$(".tag-search-result").html(a.$el.html())
                    })) : void 0
                }, 200))
            }, c.prototype.unsubscribe = function(a) {
                var b, c, d, e = this;
                return a.preventDefault(), b = $(a.currentTarget), d = b.attr("data-tag"), b.html(b.html() + "..."), c = new f({
                    name: d
                }), c.unsubscribe().done(function() {
                    return b.attr("href", "#follow").html("<span class='follow-text''>Subscribe</span>").removeClass("follow-toggle-following").addClass("follow-toggle-not-following"), e.dirty=!0
                })
            }, c.prototype.subscribe = function(a) {
                var b, c, d, e = this;
                return a.preventDefault(), b = $(a.currentTarget), b = $(a.currentTarget), d = b.attr("data-tag"), b.html(b.html() + "..."), c = new f({
                    name: d
                }), c.subscribe().done(function() {
                    return b.attr("href", "#unfollow").html("<span class='following-text'>Unsubscribed</span><span class='unfollow-text'>Unsubscribe</span>").removeClass("follow-toggle-not-following").addClass("follow-toggle-following"), e.dirty=!0
                })
            }, c.prototype.dispose = function() {
                return this.dirty && this.publishEvent("subscription:updated"), c.__super__.dispose.apply(this, arguments)
            }, c
        }(e)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/discover/list", ["views/base/collection_view", "views/discover/link", "views/modals/tag_search", "lib/session"], function(a, c, d, e) {
        var f, g;
        return f = function(a) {
            function f() {
                return g = f.__super__.constructor.apply(this, arguments)
            }
            return b(f, a), f.prototype.template = "discover/list", f.prototype.itemView = c, f.prototype.className = ".links", f.prototype.listSelector = ".link-list", f.prototype.previousLoadCount = 0, f.prototype.events = {
                "click a[href=#search-tag]": "show_tag_search"
            }, f.prototype.initialize = function(a) {
                return f.__super__.initialize.call(this, a), this.subscribeEvent("links:load-trigger", this.loadMore), this.subscribeEvent("links:load-finish", this.loadFinished), this.subscribeEvent("subscription:updated", this.reload), this.subscribeEvent("links:load-start", this.startLoading), this.listenTo(this.collection, "reset", this.render)
            }, f.prototype.loadFinished = function() {
                var a, b, c;
                return this.stopLoading(), 0 === this.collection.length ? this.$el.html("<div class='no-link-found'>New to Delicious? Try to <a href='#search-tag'>subscribe some tags</a> to get a personalized feed.</div>") : !e.is_moz_sidebar && _.isEmpty(this.$(".ad-sponsors iframe")) ? (c = this.collection.getKeywords().slice(0, 1), a = "https://rcm-na.amazon-adsystem.com/e/cm?t=delicious03-20&o=1&p=13&l=st1&mode=books&search=" + c.join("+") + "&fc1=000000&lt1=_blank&lc1=3366FF&bg1=FFFFFF&npa=1&f=ifr", b = $("<iframe/>").attr("src", a).attr("width", 468).attr("height", 68).attr("frameBorder", 0).attr("scrolling", "no")) : void 0
            }, f.prototype.reload = function() {
                return this.collection.reset(), this.collection.fetch(), this.startLoading()
            }, f.prototype.show_tag_search = function(a) {
                return a.preventDefault(), new d
            }, f
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("controllers/links_controller", ["chaplin", "lib/session", "moment", "controllers/base/controller", "models/links", "models/user", "models/tag", "models/tags", "models/tag_bundle", "models/tag_bundles", "models/note", "models/notes", "models/base/model", "views/links/page", "views/links/user", "views/links/list", "views/links/tagcloud", "views/links/tagbundles", "views/links/tag", "views/links/date", "views/links/welcome", "views/network/page", "views/network/list", "views/links/single", "views/links/single/link", "views/links/single/comments", "views/links/notes", "views/links/single/related", "views/links/related_tags", "views/discover/page", "views/discover/list", "views/not_found"], function(a, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, A, B, C, D, E, F, G) {
        var H, I;
        return H = function(a) {
            function d() {
                return I = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.person = function(a, b) {
                var d, e, h, l, m, n, t, w, x, y = this;
                return w = a.username, d = w === (null != (x = c.user) ? x.get("username") : void 0), m = a.tag ? decodeURIComponent(a.tag).split(",") : [], n = a.tbd ? decodeURIComponent(a.tbd) : void 0, h = a.q ? decodeURIComponent(a.q.replace(",", " ")) : [], l = a.since ? moment(a.since, "YYYYMM").add("M", 1).subtract("d", 1).unix() : void 0, "me" === w && c.user ? (e = "/" + b.path.replace("me", c.user.get("username")), void this.redirectTo({
                    url : e
                })) : (d ? t = c.user : (t = new g({
                    username: w
                }), this.user = t), this.links = new f([], {
                    urlParams: {
                        username: w,
                        tags: m,
                        tagBundle: n,
                        keywords: h,
                        limit: 20,
                        hasImage: !1,
                        timeStart: l
                    }
                }), this.view = new o({
                    model: t,
                    editable: d
                }), this.view.subview("user", new p({
                    model: t,
                    editable: d,
                    el: this.view.$(".user-info")
                })), this.startLoading(), t.profile().done(function(b) {
                    return y.stopLoading(), null === b.pkg || "D" === t.get("status") ? (y.view.dispose(), void(y.view = new G)) : (y.tags = new i([], {
                        urlParams: {
                            username: w
                        }
                    }), y.tags.fetch().done(function() {
                        return y.tags.ensure_comparator(), y.tags.sort()
                    }), y.tag_bundles = new k([], {
                        urlParams: {
                            username: w
                        }
                    }), y.tag_bundles.fetch({
                        reset: !0
                    }), d && 0 === t.get("bookmark_count") ? y.view.subview("links", new v({
                        el: y.view.$(".links")
                    })) : y.view.subview("links", new q({
                        collection: y.links,
                        el: y.view.$(".links"),
                        editable: d,
                        by_user: !0,
                        owner: w
                    })), y.view.subview("tagcloud", new r({
                        collection: y.tags,
                        el: y.view.$(".tagcloud")
                    })), y.view.subview("tagbundles", new s({
                        collection: y.tag_bundles,
                        el: y.view.$(".tagbundles"),
                        editable: d
                    })), y.view.subview("datepicker", new u({
                        collection: y.links,
                        since: t.get("time_created"),
                        el: y.view.$(".datetable")
                    })), y.view.subview("relatedtags", new D({
                        collection: new i([]),
                        el: y.view.$(".relatedtags")
                    })), y.links.fetch().done(function() {
                        return _.isEmpty(m) || y.view.update_related_tags(), n ? y.view.update_bundle_tags(new j({
                            name: n
                        })) : void 0
                    }), _.each(m, function(a) {
                        return y.view.append_filter("#" + a, "tag")
                    }), n && y.view.append_filter("[" + n + "]", "tagbundle"), _.isEmpty(h) || y.view.append_filter('"' + h + '"', "search"), l ? y.view.$(".datetable a[data-date='" + a.since + "']").addClass("on") : void 0)
                }), this.publishEvent("sidebar:change", {
                    action: "page",
                    location: d && _.all([m, n, h], _.isEmpty) ? "home": this.links.to_search_string()
                }), this.adjustTitle("@" + w))
            }, d.prototype.tag = function(a) {
                var b, c, d = this;
                return c = decodeURIComponent(a.tag), b = a.tab, this.thetag = new h({
                    name: c
                }), this.startLoading(), this.thetag.history().done(function() {
                    return d.stopLoading(), d.links = "alltime" !== b ? new f([], {
                        urlParams: {
                            hashtag: d.thetag.get("name"),
                            sortBy: b,
                            limit: 20
                        }
                    }) : new f([], {
                        urlParams: {
                            type: "everyone",
                            tags: [d.thetag.get("name")],
                            limit: 20
                        }
                    }), d.view = new o({
                        model: d.thetag,
                        type: "tag",
                        tab: b
                    }), d.view.subview("tag", new t({
                        model: d.thetag,
                        el: d.view.$(".tag-info")
                    })), d.view.subview("links", new q({
                        collection: d.links,
                        el: d.view.$(".links")
                    })), d.links.fetch().done(function() {
                        return 0 === d.links.length ? d.redirectTo({
                            url: "/tag/" + c + "/alltime"
                        }) : void 0
                    }), d.adjustTitle(d.links.to_search_string()), d.publishEvent("sidebar:change", {
                        action: "page",
                        location: d.links.to_search_string()
                    })
                })
            }, d.prototype.network = function() {
                return c.user.get("isLoggedIn") ? (this.links = new f([], {
                    urlParams: {
                        type: "network",
                        limit: 20,
                        hasImage: !1
                    }
                }), this.view = new w, this.view.subview("links", new x({
                    collection: this.links,
                    el: this.view.$(".links")
                })), this.links.fetch(), this.adjustTitle("Network"), this.publishEvent("sidebar:change", {
                    action: "page",
                    location: "network"
                })) : void this.redirectTo({
                    url: "/"
                })
            }, d.prototype.single = function(a) {
                var b, d = this;
                return b = a.md5, this.links = new f([], {
                    urlParams: {
                        md5: b
                    }
                }), this.startLoading(), this.links.fetch().done(function() {
                    return d.stopLoading(), d.link = d.links.first(), d.view = new y({
                        model: d.link
                    }), d.link && (d.adjustTitle(d.link.get("title")), d.view.subview("link", new z({
                        model: d.link,
                        el: d.view.$(".single-link")
                    })), c.user.get("isLoggedIn") && d.link.compose(), d.notes = new m([], {
                        urlParams: {
                            md5: b,
                            limit: 20
                        }
                    }), d.related = new f([], {
                        urlParams: {
                            md5: b,
                            limit: 7,
                            type: "related"
                        }
                    }), d.view.subview("notes", new B({
                        collection: d.notes,
                        el: d.view.$(".notes")
                    })), d.view.subview("related", new C({
                        collection: d.related,
                        el: d.view.$(".related-links")
                    })), d.notes.fetch().done(function() {
                        var a;
                        return d.view.subview("notes").render(), a = _(d.notes.filter(function(a) {
                            return a.get("note")
                        })).sortBy(function(a) {
                            return - a.get("note").length
                        }).first(3).value(), d.top_notes = new m(a), d.view.subview("comments", new A({
                            collection: d.top_notes,
                            el: d.view.$(".comments")
                        }))
                    }), d.related.fetch()), d.publishEvent("sidebar:change", {
                        action: "page",
                        location: "link"
                    })
                })
            }, d.prototype.search = function(a) {
                var b, d, e;
                return b = decodeURIComponent(a.q), d = a.tab, (null != (e = c.user) ? e.get("isLoggedIn") : 0) || (d = "everyone"), this.query = new n({
                    q: b
                }), this.links = new f([], {
                    urlParams: {
                        type: d,
                        keywords: b
                    }
                }), this.view = new o({
                    model: this.query,
                    type: "search",
                    tab: d
                }), this.view.subview("links", new q({
                    collection: this.links,
                    el: this.view.$(".links"),
                    editable: "you" === d
                })), this.links.fetch(), this.adjustTitle('"' + b + '"'), this.publishEvent("sidebar:change", {
                    action: "page",
                    location: '"' + b + '"'
                })
            }, d.prototype.discover = function() {
                return c.user.get("isLoggedIn") ? (this.links = new f([], {
                    urlParams: {
                        type: "discover"
                    }
                }), this.view = new E({
                    model: c.user
                }), this.view.subview("links", new F({
                    collection: this.links,
                    el: this.view.$(".links")
                })), this.links.fetch(), this.publishEvent("sidebar:change", {
                    action: "page",
                    location: "discover"
                }), this.adjustTitle("Discover")) : void this.redirectTo({
                    url: "/"
                })
            }, d.prototype.domain = function(a) {
                var b, c;
                return b = a.domain, c = a.tab, this.links = new f([], {
                    urlParams: {
                        type: c,
                        domain: b
                    }
                }), this.view = new o({
                    model: new n({
                        q: "(" + b + ")"
                    }),
                    type: "search",
                    tab: c
                }), this.view.subview("links", new q({
                    collection: this.links,
                    el: this.view.$(".links")
                })), this.links.fetch(), this.adjustTitle("(" + b + ")"), this.publishEvent("sidebar:change", {
                    action: "page",
                    location: '"(' + b + ')"'
                })
            }, d
        }(e)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("models/users", ["models/base/collection", "models/user"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.model = c, d.prototype.comparator = function(a) {
                return a.get("username").toLowerCase()
            }, d.prototype.filterBySocial = function(a) {
                return this.filter(function(b) {
                    return _.contains(b.get("social_types"), a[0].toUpperCase() + a.slice(1))
                })
            }, d.prototype.friends = function() {
                return this.url = "/account/friends", this.fetch()
            }, d.prototype.featured = function() {
                return this.url = "/account/featuredusers", this.fetch()
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/user/relations/user_relationship", ["views/base/pageview", "views/modals/tag_search"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "user/relations/user_relationship", d.prototype.initialize = function(a) {
                return this.type = a.type
            }, d.prototype.events = {
                "click a[href=#search-tag]": "show_tag_search"
            }, d.prototype.getTemplateData = function() {
                var a;
                return a = d.__super__.getTemplateData.apply(this, arguments), a.is_following = "following" === this.type, a.is_subscription = "subscription" === this.type, a
            }, d.prototype.show_tag_search = function(a) {
                return a.preventDefault(), new c
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/user/relations/user_relationships", ["chaplin", "lib/session", "views/base/hbsview"], function(a, c, d) {
        var e, f;
        return e = function(a) {
            function d() {
                return f = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "user/relations/user_relationships", d.prototype.autoRender=!1, d.prototype.initialize = function() {
                return d.__super__.initialize.apply(this, arguments), this.listenTo(this.collection, "reset", this.render)
            }, d.prototype.events = {
                "click a[href=#unfollow]": "unfollow",
                "click a[href=#follow]": "follow"
            }, d.prototype.getTemplateData = function() {
                var a, b;
                return b = c.user.get("follows"), this.collection.each(function(a) {
                    return a.set("am_following_them", !_.isEmpty(_.where(b, {
                        username: a.get("username")
                    }))), a.set("is_me", a.get("username") === c.user.get("username"))
                }), a = d.__super__.getTemplateData.apply(this, arguments), a.items
            }, d.prototype.unfollow = function(a) {
                var b, d, e;
                return a.preventDefault(), b = $(a.currentTarget), e = b.attr("data-user"), c.user ? (b.html(b.html() + "..."), d = this.collection.findWhere({
                    username: e
                }), c.user.unfollowUser(d).done(function() {
                    return b.attr("href", "#follow").html("<span class='follow-text''>Follow</span>").removeClass("follow-toggle-following").addClass("follow-toggle-not-following")
                })) : void 0
            }, d.prototype.follow = function(a) {
                var b, d, e;
                return a.preventDefault(), b = $(a.currentTarget), e = b.attr("data-user"), c.user ? (b.html(b.html() + "..."), d = this.collection.findWhere({
                    username: e
                }), c.user.followUser(d).done(function() {
                    return b.attr("href", "#unfollow").html("<span class='following-text'>Following</span><span class='unfollow-text'>Unfollow</span>").removeClass("follow-toggle-not-following").addClass("follow-toggle-following")
                })) : void 0
            }, d
        }(d)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/user/relations/user_subscriptions", ["chaplin", "lib/session", "views/base/hbsview", "models/tag"], function(a, c, d, e) {
        var f, g;
        return f = function(a) {
            function c() {
                return g = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "user/relations/user_subscriptions", c.prototype.autoRender=!1, c.prototype.initialize = function() {
                return c.__super__.initialize.apply(this, arguments), this.listenTo(this.collection, "reset", this.render), this.subscribeEvent("subscription:updated", this.refresh)
            }, c.prototype.events = {
                "click a[href=#unfollow]": "unsubscribe",
                "click a[href=#follow]": "subscribe"
            }, c.prototype.getTemplateData = function() {
                var a;
                return a = c.__super__.getTemplateData.apply(this, arguments), a.items
            }, c.prototype.refresh = function() {
                return this.collection.fetch({
                    reset: !0
                })
            }, c.prototype.unsubscribe = function(a) {
                var b, c, d;
                return a.preventDefault(), b = $(a.currentTarget), d = b.attr("data-tag"), b.html(b.html() + "..."), c = new e({
                    name: d
                }), c.unsubscribe().done(function() {
                    return b.attr("href", "#follow").html("<span class='follow-text''>Subscribe</span>").removeClass("follow-toggle-following").addClass("follow-toggle-not-following")
                })
            }, c.prototype.subscribe = function(a) {
                var b, c, d;
                return a.preventDefault(), b = $(a.currentTarget), b = $(a.currentTarget), d = b.attr("data-tag"), c = new e({
                    name: d
                }), c.subscribe().done(function() {
                    return b.attr("href", "#unfollow").html("<span class='following-text'>Unsubscribed</span><span class='unfollow-text'>Unsubscribe</span>").removeClass("follow-toggle-not-following").addClass("follow-toggle-following")
                })
            }, c
        }(d)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("models/connects", ["models/base/collection", "models/connect"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.model = c, d.prototype.url = "/connect", d.prototype.parse = function() {
                var a;
                return a = d.__super__.parse.apply(this, arguments), _.flatten(_.values(a))
            }, d.prototype.filterForLinkImporting = function() {
                return this.filter(function(a) {
                    return a.get("active")===!0
                })
            }, d.prototype.findByProvider = function(a) {
                return this.find(function(b) {
                    return b.get("providerId") === a
                })
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("models/app", ["models/base/model"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.idAttribute = "client_id", c.prototype.info = function() {
                var a, b = this;
                return a = {
                    app_keys: this.get("client_id")
                }, this.api({
                    url: "/app/basic_info",
                    type: "GET",
                    data: a,
                    success: function(a) {
                        var c;
                        return !_.isEmpty(a.pkg) && (c = a.pkg[0]) ? b.set(c) : void 0
                    }
                })
            }, c.prototype.auth = function() {
                var a, b = this;
                return a = {
                    client_id: this.get("client_id"),
                    redirect_uri: this.get("redirect_uri")
                }, this.api({
                    url: "/oauth/auth",
                    type: "POST",
                    data: a,
                    success: function(a) {
                        return a.pkg ? b.set(a.pkg) : void 0
                    }
                })
            }, c.prototype.revoke = function() {
                var a;
                return a = {
                    client_id: this.get("app_key")
                }, this.api({
                    url: "/oauth/token/revoke",
                    type: "POST",
                    data: a
                })
            }, c.prototype.save = function(a) {
                var b, c;
                return c = this.get("app_key") ? "/app/update" : "/app/create", b = {
                    details: JSON.stringify(a),
                    app_key: this.get("app_key")
                }, this.api({
                    url: c,
                    type: "POST",
                    data: b
                })
            }, c.prototype.enable = function() {
                return this.api({
                    url: "/app/enable",
                    type: "POST",
                    data: {
                        app_key: this.get("app_key")
                    }
                })
            }, c.prototype.disable = function() {
                return this.api({
                    url: "/app/disable",
                    type: "POST",
                    data: {
                        app_key: this.get("app_key")
                    }
                })
            }, c.prototype["delete"] = function() {
                return this.api({
                    url: "/app/delete",
                    type: "POST",
                    data: {
                        app_key: this.get("app_key")
                    }
                })
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("models/apps", ["models/base/collection", "models/app"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.model = c, d.prototype.initialize = function(a, b) {
                return d.__super__.initialize.apply(this, arguments), this.type = b.type
            }, d.prototype.url = function() {
                switch (this.type) {
                case"authorized":
                    return "/oauth/auth/apps";
                case"developed":
                    return "/app/list"
                }
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/user/settings/thirdparty", ["views/base/hbsview", "models/connect"], function(a, c) {
        var d, e;
        return d = function(a) {
            function d() {
                return e = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "user/settings/thirdparty", d.prototype.autoRender=!0, d.prototype.autoAttach=!0, d.prototype.events = {
                "click .connect-button": "to_connect",
                "change .import-control": "check_import_options",
                "click .disconnect-button": "to_disconnect",
                "submit .thirdparty-options": "save_settings"
            }, d.prototype.to_connect = function(a) {
                var b, d, e = this;
                return d = $(a.currentTarget).attr("data-service"), b = new c({
                    providerId: d
                }), b.connect(!1, function(a) {
                    return "error" !== a.status && "success" === a.status ? e.model.fetch({
                        reset: !0
                    }).done(function() {
                        return e.render()
                    }) : void 0
                })
            }, d.prototype.getTemplateData = function() {
                var a, b, c;
                return {
                    twitter: null != (a = this.model.findWhere({
                        providerId: "twitter"
                    })) ? a.toJSON(): void 0,
                    facebook: null != (b = this.model.findWhere({
                        providerId: "facebook"
                    })) ? b.toJSON() : void 0, google : null != (c = this.model.findWhere({
                        providerId: "google"
                    })) ? c.toJSON() : void 0
                }
            }, d.prototype.render = function() {
                return d.__super__.render.apply(this, arguments), this.check_import_options()
            }, d.prototype.check_import_options = function() {
                return _.each(this.$(".import-control"), function(a) {
                    var b, c;
                    return c = $(a).prop("checked"), b = $(a).parents("form"), $(".thirdparty-privacy-control", b).prop("disabled", c ? "" : "disabled"), $(".thirdparty-tag-control", b).prop("contentEditable", c)
                })
            }, d.prototype.to_disconnect = function(a) {
                var b, c, d = this;
                return c = $(a.currentTarget).attr("data-service"), b = this.model.findWhere({
                    providerId: c
                }), b.disconnect().done(function() {
                    return d.render()
                })
            }, d.prototype.save_settings = function(a) {
                var b, c, d, e, f, g = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), c = $(a.currentTarget), f = c.attr("data-service"), d = {
                    active: c.find(".import-control").prop("checked"),
                    privacy: c.find(".thirdparty-privacy-control option:selected").val(),
                    tags: c.find(".thirdparty-tag-control").text()
                }, e = this.model.findWhere({
                    providerId: f
                }), this.publishEvent("settings:saving"), b = c.find("button[type='submit']"), b.text("Saving..."), b.prop("disabled", "disabled"), e.update(d, function() {
                    return g.publishEvent("settings:saved"), b.text("saved"), _.delay(function() {
                        return b.prop("disabled", ""), b.text("save")
                    }, 2e3)
                })
            }, d
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/user/settings/apps", ["views/base/collection_view", "views/base/hbsview", "models/apps"], function(a, c) {
        var d, e, f, g;
        return e = function(a) {
            function c() {
                return f = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.autoRender=!0, c.prototype.autoAttach=!0, c.prototype.template = "user/settings/app", c.prototype.events = {
                "click a[href=#revoke]": "revoke"
            }, c.prototype.revoke = function(a) {
                var b = this;
                return a.preventDefault(), confirm("Are you sure to revoke access from this app?") ? this.model.revoke().done(function() {
                    return b.model.destroy()
                }) : void 0
            }, c
        }(c), d = function(a) {
            function c() {
                return g = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.autoRender=!0, c.prototype.autoAttach=!0, c.prototype.template = "user/settings/apps", c.prototype.listSelector = ".app-list", c.prototype.itemView = e, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/modals/app", ["views/base/modal", "models/app"], function(a) {
        var c, d;
        return c = function(a) {
            function c() {
                return d = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.template = "modals/app", c.prototype.events = {
                "click button.text-btn-primary": "do_save",
                "click button.disable-app": "do_disable",
                "click button.enable-app": "do_enable",
                "click button.delete-app": "do_delete"
            }, c.prototype.do_save = function(a) {
                var b = this;
                return null != a && a.preventDefault(), this.model.save({
                    app_name: this.$("input[name=app-name]").val(),
                    description: this.$("textarea[name=app-description]").val()
                }).done(function() {
                    return b.publishEvent("apps:updated"), b.close()
                })
            }, c.prototype.do_disable = function(a) {
                var b = this;
                return null != a && a.preventDefault(), confirm("Are you sure to disable this app ?") ? this.model.disable().done(function() {
                    return b.publishEvent("apps:updated"), b.close()
                }) : void 0
            }, c.prototype.do_enable = function(a) {
                var b = this;
                return null != a && a.preventDefault(), this.model.enable().done(function() {
                    return b.publishEvent("apps:updated"), b.close()
                })
            }, c.prototype.do_delete = function(a) {
                var b = this;
                return null != a && a.preventDefault(), confirm("Are you sure to delete this app permanently ?") ? this.model["delete"]().done(function() {
                    return b.publishEvent("apps:updated"), b.close()
                }) : void 0
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/user/settings/devapps", ["views/base/collection_view", "views/base/hbsview", "models/app", "models/apps", "views/modals/app"], function(a, c, d, e, f) {
        var g, h, i, j;
        return h = function(a) {
            function c() {
                return i = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.autoRender=!0, c.prototype.autoAttach=!0, c.prototype.template = "user/settings/devapp", c.prototype.events = {
                "click a[href=#edit]": "edit"
            }, c.prototype.edit = function(a) {
                return a.preventDefault(), new f({
                    model: this.model
                })
            }, c
        }(c), g = function(a) {
            function c() {
                return j = c.__super__.constructor.apply(this, arguments)
            }
            return b(c, a), c.prototype.autoRender=!0, c.prototype.autoAttach=!0, c.prototype.template = "user/settings/devapps", c.prototype.listSelector = ".app-list", c.prototype.itemView = h, c.prototype.initialize = function() {
                return c.__super__.initialize.apply(this, arguments), this.subscribeEvent("apps:updated", this.refresh)
            }, c.prototype.events = {
                "click .create-devapp": "create_app"
            }, c.prototype.create_app = function(a) {
                return a.preventDefault(), new f({
                    model: new d
                })
            }, c.prototype.refresh = function() {
                return this.collection.reset(), this.collection.fetch()
            }, c
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/user/settings", ["views/base/pageview", "chaplin", "lib/session", "models/connects", "models/apps", "views/user/settings/thirdparty", "views/user/settings/apps", "views/user/settings/devapps"], function(a, c, d, e, f, g, h, i) {
        var j, k;
        return j = function(a) {
            function j() {
                return k = j.__super__.constructor.apply(this, arguments)
            }
            return b(j, a), j.prototype.template = "user/settings", j.prototype.maxFileSize = 2e3, j.prototype.initialize = function(a) {
                return j.__super__.initialize.apply(this, arguments), this.tab = a.tab, this.subscribeEvent("settings:saving", this.startLoading), this.subscribeEvent("settings:saved", this.stopLoading)
            }, j.prototype.render = function() {
                var a = this;
                return j.__super__.render.apply(this, arguments), this.update_chars_remaining(), this.conns = new e, this.conns.fetch().done(function() {
                    return a.subview("thirdparty", new g({
                        model: a.conns,
                        el: a.$(".thirdparty-items")
                    }))
                }), this.apps = new f([], {
                    type: "authorized"
                }), this.apps.fetch().done(function() {
                    return a.subview("authorized_apps", new h({
                        collection: a.apps,
                        el: a.$(".apps-container")
                    }))
                }), this.devapps = new f([], {
                    type: "developed"
                }), this.devapps.fetch().done(function() {
                    return a.subview("developer_apps", new i({
                        collection: a.devapps,
                        el: a.$(".devapps-container")
                    }))
                })
            }, j.prototype.attach = function() {
                var a;
                return j.__super__.attach.apply(this, arguments), this.tab && null != (a = this.$("ul.actions a[href='#" + this.tab + "']")) ? a.tab("show") : void 0
            }, j.prototype.events = {
                "submit #profile_form": "submit_profile",
                "click .avatar-edit": "select_avatar",
                "change #avatar_file": "upload_avatar",
                "keyup #bio": "update_chars_remaining",
                "submit #email_form": "submit_email",
                "submit #password_form": "submit_password",
                "submit #deactivate_form": "submit_deactivate",
                "click a[data-toggle]": "update_tab_url",
                "click .avatar-reset": "reset_avatar"
            }, j.prototype.reset_avatar = function(a) {
                var b = this;
                return a.preventDefault(), confirm("Are your sure to remove your avatar?") ? (this.publishEvent("settings:saving"), d.user.resetAvatar().done(function() {
                    return b.$(".avatar_display").attr("src", d.user.get("avatar_url")), b.$(".avatar-wrapper").removeClass("has-avatar"), b.publishEvent("settings:saved")
                })) : void 0
            }, j.prototype.update_tab_url = function(a) {
                var b, d;
                return b = null != (d = $(a.currentTarget).attr("href")) ? d.substring(1) : void 0, b ? c.mediator.execute("router:changeURL", "/settings/" + b) : void 0
            }, j.prototype.update_chars_remaining = function() {
                var a, b;
                return b = this.$("#bio").val().length, a = 140 - b, a >= 0 ? this.$("#bio_chars_remaining").removeClass("error").text(a) : this.$("#bio_chars_remaining").addClass("error").text(a)
            }, j.prototype.submit_profile = function(a) {
                var b, c = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), b = {
                    full_name: this.$("#display_name").val(),
                    profile_bio: this.$("#bio").val(),
                    profile_url: this.$("#url").val()
                }, this.$("#profile_submit").text("Saving...").prop("disabled", "disabled"), this.publishEvent("settings:saving"), d.user.update(b).done(function() {
                    return c.$("#profile_submit").text("Saved"), c.publishEvent("settings:saved"), _.delay(function() {
                        return c.$("#profile_submit").text("Save").prop("disabled", "")
                    }, 2500)
                })
            }, j.prototype.select_avatar = function() {
                return this.$("#avatar_file").click()
            }, j.prototype.upload_avatar = function(a) {
                var b, c = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.$("#avatar_file")[0].files && this.$("#avatar_file")[0].files[0].size / 1e3 > this.maxFileSize ? void alert("Image size too big.") : (b = new FormData, b.append("photo", this.$("#avatar_file")[0].files[0]), this.$(".avatar-edit").text("Uploading").prop("disabled", "disabled"), this.$(".avatar-wrapper").addClass("uploading"), this.publishEvent("settings:saving"), d.user.updateAvatar(b).done(function() {
                    return c.$(".avatar-edit").text("Change").prop("disabled", ""), c.$(".avatar-wrapper").removeClass("uploading"), _.delay(function() {
                        return c.$(".avatar_display").attr("src", d.user.get("avatar_url")), c.$(".avatar-wrapper").addClass("has-avatar"), c.publishEvent("settings:saved")
                    }, 500)
                }))
            }, j.prototype.submit_email = function(a) {
                var b, c = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), b = {
                    profile_email: this.$("#profile_email").val()
                }, this.$("#email_submit").text("Saving...").prop("disabled", "disabled"), this.publishEvent("settings:saving"), d.user.update(b).done(function() {
                    return c.$("#email_submit").text("Saved"), _.delay(function() {
                        return c.$("#email_submit").text("Save").prop("disabled", ""), c.publishEvent("settings:saved")
                    }, 2500)
                })
            }, j.prototype.submit_password = function(a) {
                var b, c = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), this.$("#password_error").text(""), this.$("#new_password").val() !== this.$("#new_password_confirm").val() ? void this.$("#password_error").text("The new password mismatch") : this.$("#new_password").val() === this.$("#original_password").val() ? void this.$("#password_error").text("The new password you entered matches the current password.") : this.$("#new_password").val().length < 8 ? void this.$("#password_error").text("The new password must be at least 8 characters long.") : (b = {
                    old_password: this.$("#original_password").val(),
                    new_password: this.$("#new_password").val()
                }, this.$("#password_submit").text("Saving...").prop("disabled", "disabled"), this.publishEvent("settings:saving"), d.user.changePassword(b).done(function() {
                    return c.$("#password_submit").text("Saved"), c.publishEvent("settings:saved"), _.delay(function() {
                        return c.$("#password_submit").text("Save").prop("disabled", "")
                    }, 2500)
                }).fail(function(a) {
                    return c.$("#password_submit").text("Save").prop("disabled", ""), c.publishEvent("settings:saved"), a ? c.$("#password_error").text(a.error.charAt(0).toUpperCase() + a.error.slice(1) + ".") : void 0
                }), this.$("#password_form input[type='password']").val(""))
            }, j.prototype.submit_deactivate = function(a) {
                var b, c = this;
                return null != a && "function" == typeof a.preventDefault && a.preventDefault(), b = this.$("#deactivate_account_password").val(), this.$("#deactivate_submit").text("Processing").prop("disabled", "disabled"), this.publishEvent("settings:saving"), d.user.deactivate({
                    passwd: b
                }).done(function() {
                    return window.location = "/"
                }).fail(function(a) {
                    return c.$("#deactivate_submit").text("Yes, deactivate my account").prop("disabled", ""), c.publishEvent("settings:saved"), a ? c.$("#deactivate_error").text(a.error.charAt(0).toUpperCase() + a.error.slice(1) + ".") : void 0
                })
            }, j
        }(a)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("views/user/settings/friends", ["chaplin", "lib/session", "views/base/pageview", "models/user", "models/users"], function(a, c, d) {
        var e, f;
        return e = function(a) {
            function d() {
                return f = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.template = "user/settings/friends", d.prototype.autoRender=!1, d.prototype.autoAttach=!0, d.prototype.initialize = function() {
                var a = this;
                return d.__super__.initialize.apply(this, arguments), this.collection.friends().done(function() {
                    return a.collection.length > 0 ? a.render() : a.collection.featured().done(function() {
                        return a.render()
                    })
                })
            }, d.prototype.events = {
                "click a[href=#unfollow]": "unfollow",
                "click a[href=#follow]": "follow"
            }, d.prototype.unfollow = function(a) {
                var b, d, e;
                return a.preventDefault(), b = $(a.currentTarget), e = b.attr("data-user"), c.user ? (b.html(b.html() + "..."), d = this.collection.findWhere({
                    username: e
                }), c.user.unfollowUser(d).done(function() {
                    return b.attr("href", "#follow").html("<span class='follow-text''>Follow</span>").removeClass("follow-toggle-following").addClass("follow-toggle-not-following")
                })) : void 0
            }, d.prototype.follow = function(a) {
                var b, d, e;
                return a.preventDefault(), b = $(a.currentTarget), e = b.attr("data-user"), c.user ? (b.html(b.html() + "..."), d = this.collection.findWhere({
                    username: e
                }), c.user.followUser(d).done(function() {
                    return b.attr("href", "#unfollow").html("<span class='following-text'>Following</span><span class='unfollow-text'>Unfollow</span>").removeClass("follow-toggle-not-following").addClass("follow-toggle-following")
                })) : void 0
            }, d
        }(d)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("controllers/user_controller", ["chaplin", "lib/session", "controllers/base/controller", "models/user", "models/tags", "models/links", "models/users", "views/user/relations/user_relationship", "views/user/relations/user_relationships", "views/user/relations/user_subscriptions", "views/user/settings", "views/user/settings/friends", "views/not_found"], function(a, c, d, e, f, g, h, i, j, k, l, m) {
        var n, o;
        return n = function(a) {
            function d() {
                return o = d.__super__.constructor.apply(this, arguments)
            }
            return b(d, a), d.prototype.following = function(a) {
                var b, c = this;
                return b = a.username, this.user = new e({
                    username: b
                }), this.startLoading(), this.user.profile().done(function() {
                    return c.stopLoading(), c.following = new h, c.following.comparator = function(a) {
                        return - (a.get("time_created") || 0)
                    }, c.following.url = c.user.getFollowingUrl(), c.following.fetch({
                        reset: !0
                    }), c.view = new i({
                        model: c.user,
                        type: "following"
                    }), c.view.subview("people", new j({
                        collection: c.following,
                        el: c.view.$(".people")
                    })), c.adjustTitle("@" + b + " following")
                })
            }, d.prototype.followers = function(a) {
                var b, c = this;
                return b = a.username, this.user = new e({
                    username: b
                }), this.startLoading(), this.user.profile().done(function() {
                    return c.stopLoading(), c.followers = new h, c.followers.comparator = function(a) {
                        return - (a.get("time_created") || 0)
                    }, c.followers.url = c.user.getFollowersUrl(), c.followers.fetch({
                        reset: !0
                    }), c.view = new i({
                        model: c.user,
                        type: "follower"
                    }), c.view.subview("people", new j({
                        collection: c.followers,
                        el: c.view.$(".people")
                    })), c.adjustTitle("@" + b + " followers")
                })
            }, d.prototype.subscriptions = function(a) {
                var b, d, g = this;
                return b = a.username, b !== (null != (d = c.user) ? d.get("username") : void 0) && this.redirectTo({
                    url: "/" + b
                }), this.user = new e({
                    username: b
                }), this.startLoading(), this.user.profile().done(function() {
                    return g.stopLoading(), g.tags = new f([], {
                        urlParams: {
                            type: "subscription"
                        }
                    }), g.tags.fetch({
                        reset: !0
                    }), g.view = new i({
                        model: g.user,
                        type: "subscription"
                    }), g.view.subview("tags", new k({
                        collection: g.tags,
                        el: g.view.$(".people")
                    })), g.adjustTitle("@" + b + " subscriptions")
                })
            }, d.prototype.settings = function(a) {
                return c.user.get("isLoggedIn") || this.redirectTo({
                    url: "/"
                }), this.view = new l({
                    model: c.user.clone(),
                    tab: a.tab
                }), this.publishEvent("sidebar:change", {
                    action: "page",
                    location: "settings"
                }), this.adjustTitle("Settings")
            }, d.prototype.find_friends = function() {
                return c.user.get("isLoggedIn") || this.redirectTo({
                    url: "/"
                }), this.friends = new h, this.view = new m({
                    collection: this.friends
                }), this.publishEvent("sidebar:change", {
                    action: "page"
                }), this.adjustTitle("Find Friends")
            }, d
        }(d)
    })
}.call(this), function() {
    var a = {}.hasOwnProperty, b = function(b, c) {
        function d() {
            this.constructor = b
        }
        for (var e in c)
            a.call(c, e) && (b[e] = c[e]);
        return d.prototype = c.prototype, b.prototype = new d, b.__super__ = c.prototype, b
    };
    define("application", ["chaplin", "views/layout", "models/user", "store", "lib/session", "controllers/home_controller", "controllers/links_controller", "controllers/user_controller"], function(a, c, d, e, f) {
        var g, h;
        return g = function(a) {
            function g() {
                return h = g.__super__.constructor.apply(this, arguments)
            }
            return b(g, a), g.prototype.title = "Delicious", g.prototype.initLayout = function(a) {
                var b, g;
                return b = e.get("user") || null, g = new d(b), f.user = g, g.get("username") && g.loginSilent(), a.openExternalToBlank=!0, this.layout = new c(_.defaults(a, {
                    title: this.title
                }))
            }, g.prototype.initRouter = function() {
                return g.__super__.initRouter.apply(this, arguments), this.subscribeEvent("router:match", function(a) {
                    return ga("send", "pageview", a.path)
                })
            }, g
        }(a.Application)
    })
}.call(this), function() {
    define("routes", [], function() {
        return function(a) {
            return a("", "home#index"), a("trending", "home#trending"), a("logout", "home#logout"), a("validate/:token", "home#validate_email"), a("tag/:tag", "links#tag", {
                params: {
                    tab: "hot"
                }
            }), a("tag/:tag/popular", "links#tag", {
                params: {
                    tab: "hot"
                }
            }), a("tag/:tag/recent", "links#tag", {
                params: {
                    tab: "recent"
                }
            }), a("tag/:tag/alltime", "links#tag", {
                params: {
                    tab: "alltime"
                }
            }), a("search/:q", "links#search", {
                params: {
                    tab: "everyone"
                }
            }), a("search/:q/everyone", "links#search", {
                params: {
                    tab: "everyone"
                }
            }), a("search/:q/network", "links#search", {
                params: {
                    tab: "network"
                }
            }), a("search/:q/mine", "links#search", {
                params: {
                    tab: "you"
                }
            }), a("domain/:domain", "links#domain", {
                params: {
                    tab: "everyone"
                }
            }), a("domain/everyone/:domain", "links#domain", {
                params: {
                    tab: "everyone"
                }
            }), a("domain/network/:domain", "links#domain", {
                params: {
                    tab: "network"
                }
            }), a("domain/you/:domain", "links#domain", {
                params: {
                    tab: "you"
                }
            }), a("link/:md5", "links#single"), a("network", "links#network"), a("discover", "links#discover"), a("settings", "user#settings"), a("settings/pw/:id/:time/:passhash", "home#reset_password"), a("settings/friends", "user#find_friends"), a("settings/:tab", "user#settings"), a("about", "home#page", {
                params: {
                    page: "about"
                }
            }), a("apps", "home#page", {
                params: {
                    page: "apps"
                }
            }), a("tools", "home#page", {
                params: {
                    page: "tools"
                }
            }), a("help", "home#page", {
                params: {
                    page: "help"
                }
            }), a("help/*any", "home#page", {
                params: {
                    page: "help"
                }
            }), a("shortcuts", "home#page", {
                params: {
                    page: "shortcuts"
                }
            }), a("rss", "home#page", {
                params: {
                    page: "rss"
                }
            }), a("developers", "home#page", {
                params: {
                    page: "developers"
                }
            }), a("branding", "home#page", {
                params: {
                    page: "branding"
                }
            }), a("terms", "home#page", {
                params: {
                    page: "terms"
                }
            }), a("terms-update", "home#page", {
                params: {
                    page: "terms-update"
                }
            }), a("privacy", "home#page", {
                params: {
                    page: "privacy"
                }
            }), a("privacy-policy", "home#page", {
                params: {
                    page: "privacy"
                }
            }), a("copyright", "home#page", {
                params: {
                    page: "copyright"
                }
            }), a("upgradebrowser", "home#page", {
                params: {
                    page: "upgradebrowser"
                }
            }), a("signin", "home#signin"), a("join", "home#signup"), a(":username/following", "user#following"), a(":username/followers", "user#followers"), a(":username/subscriptions", "user#subscriptions"), a("post", "bookmarklet#compose_link"), a("save", "bookmarklet#compose_link"), a("auth/authorize", "oauth#auth"), a(":username", "links#person"), a(":username/:tag", "links#person"), a(":username/:tag/search/:q", "links#person"), a(":username/:tag/search/:q/since/:since", "links#person"), a(":username/:tag/since/:since", "links#person"), a(":username/tag_bundle/:tbd", "links#person"), a(":username/tag_bundle/:tbd/since/:since", "links#person"), a(":username/tag_bundle/:tbd/search/:q", "links#person"), a(":username/tag_bundle/:tbd/search/:q/since/:since", "links#person"), a(":username/since/:since", "links#person"), a(":username/search/:q", "links#person"), a(":username/search/:q/since/:since", "links#person"), a("*anything", "home#not_found")
        }
    })
}.call(this), "undefined" == typeof jQuery)
    throw new Error("Bootstrap requires jQuery");
+ function(a) {
    function b() {
        var a = document.createElement("bootstrap"), b = {
            WebkitTransition: "webkitTransitionEnd",
            MozTransition: "transitionend",
            OTransition: "oTransitionEnd otransitionend",
            transition: "transitionend"
        };
        for (var c in b)
            if (void 0 !== a.style[c])
                return {
                    end: b[c]
                }
    }
    a.fn.emulateTransitionEnd = function(b) {
        var c=!1, d = this;
        a(this).one(a.support.transition.end, function() {
            c=!0
        });
        var e = function() {
            c || a(d).trigger(a.support.transition.end)
        };
        return setTimeout(e, b), this
    }, a(function() {
        a.support.transition = b()
    })
}(jQuery), + function(a) {
    var b = '[data-dismiss="alert"]', c = function(c) {
        a(c).on("click", b, this.close)
    };
    c.prototype.close = function(b) {
        function c() {
            f.trigger("closed.bs.alert").remove()
        }
        var d = a(this), e = d.attr("data-target");
        e || (e = d.attr("href"), e = e && e.replace(/.*(?=#[^\s]*$)/, ""));
        var f = a(e);
        b && b.preventDefault(), f.length || (f = d.hasClass("alert") ? d : d.parent()), f.trigger(b = a.Event("close.bs.alert")), b.isDefaultPrevented() || (f.removeClass("in"), a.support.transition && f.hasClass("fade") ? f.one(a.support.transition.end, c).emulateTransitionEnd(150) : c())
    };
    var d = a.fn.alert;
    a.fn.alert = function(b) {
        return this.each(function() {
            var d = a(this), e = d.data("bs.alert");
            e || d.data("bs.alert", e = new c(this)), "string" == typeof b && e[b].call(d)
        })
    }, a.fn.alert.Constructor = c, a.fn.alert.noConflict = function() {
        return a.fn.alert = d, this
    }, a(document).on("click.bs.alert.data-api", b, c.prototype.close)
}(jQuery), + function(a) {
    var b = function(c, d) {
        this.$element = a(c), this.options = a.extend({}, b.DEFAULTS, d)
    };
    b.DEFAULTS = {
        loadingText: "loading..."
    }, b.prototype.setState = function(a) {
        var b = "disabled", c = this.$element, d = c.is("input") ? "val": "html", e = c.data();
        a += "Text", e.resetText || c.data("resetText", c[d]()), c[d](e[a] || this.options[a]), setTimeout(function() {
            "loadingText" == a ? c.addClass(b).attr(b, b) : c.removeClass(b).removeAttr(b)
        }, 0)
    }, b.prototype.toggle = function() {
        var a = this.$element.closest('[data-toggle="buttons"]'), b=!0;
        if (a.length) {
            var c = this.$element.find("input");
            "radio" === c.prop("type") && (c.prop("checked") && this.$element.hasClass("active") ? b=!1 : a.find(".active").removeClass("active")), b && c.prop("checked", !this.$element.hasClass("active")).trigger("change")
        }
        b && this.$element.toggleClass("active")
    };
    var c = a.fn.button;
    a.fn.button = function(c) {
        return this.each(function() {
            var d = a(this), e = d.data("bs.button"), f = "object" == typeof c && c;
            e || d.data("bs.button", e = new b(this, f)), "toggle" == c ? e.toggle() : c && e.setState(c)
        })
    }, a.fn.button.Constructor = b, a.fn.button.noConflict = function() {
        return a.fn.button = c, this
    }, a(document).on("click.bs.button.data-api", "[data-toggle^=button]", function(b) {
        var c = a(b.target);
        c.hasClass("btn") || (c = c.closest(".btn")), c.button("toggle"), b.preventDefault()
    })
}(jQuery), + function(a) {
    var b = function(b, c) {
        this.$element = a(b), this.$indicators = this.$element.find(".carousel-indicators"), this.options = c, this.paused = this.sliding = this.interval = this.$active = this.$items = null, "hover" == this.options.pause && this.$element.on("mouseenter", a.proxy(this.pause, this)).on("mouseleave", a.proxy(this.cycle, this))
    };
    b.DEFAULTS = {
        interval: 5e3,
        pause: "hover",
        wrap: !0
    }, b.prototype.cycle = function(b) {
        return b || (this.paused=!1), this.interval && clearInterval(this.interval), this.options.interval&&!this.paused && (this.interval = setInterval(a.proxy(this.next, this), this.options.interval)), this
    }, b.prototype.getActiveIndex = function() {
        return this.$active = this.$element.find(".item.active"), this.$items = this.$active.parent().children(), this.$items.index(this.$active)
    }, b.prototype.to = function(b) {
        var c = this, d = this.getActiveIndex();
        if (!(b > this.$items.length - 1 || 0 > b))
            return this.sliding ? this.$element.one("slid.bs.carousel", function() {
                c.to(b)
            }) : d == b ? this.pause().cycle() : this.slide(b > d ? "next" : "prev", a(this.$items[b]))
    }, b.prototype.pause = function(b) {
        return b || (this.paused=!0), this.$element.find(".next, .prev").length && a.support.transition.end && (this.$element.trigger(a.support.transition.end), this.cycle(!0)), this.interval = clearInterval(this.interval), this
    }, b.prototype.next = function() {
        return this.sliding ? void 0 : this.slide("next")
    }, b.prototype.prev = function() {
        return this.sliding ? void 0 : this.slide("prev")
    }, b.prototype.slide = function(b, c) {
        var d = this.$element.find(".item.active"), e = c || d[b](), f = this.interval, g = "next" == b ? "left": "right", h = "next" == b ? "first": "last", i = this;
        if (!e.length) {
            if (!this.options.wrap)
                return;
            e = this.$element.find(".item")[h]()
        }
        this.sliding=!0, f && this.pause();
        var j = a.Event("slide.bs.carousel", {
            relatedTarget: e[0],
            direction: g
        });
        if (!e.hasClass("active")) {
            if (this.$indicators.length && (this.$indicators.find(".active").removeClass("active"), this.$element.one("slid.bs.carousel", function() {
                var b = a(i.$indicators.children()[i.getActiveIndex()]);
                b && b.addClass("active")
            })), a.support.transition && this.$element.hasClass("slide")) {
                if (this.$element.trigger(j), j.isDefaultPrevented())
                    return;
                e.addClass(b), e[0].offsetWidth, d.addClass(g), e.addClass(g), d.one(a.support.transition.end, function() {
                    e.removeClass([b, g].join(" ")).addClass("active"), d.removeClass(["active", g].join(" ")), i.sliding=!1, setTimeout(function() {
                        i.$element.trigger("slid.bs.carousel")
                    }, 0)
                }).emulateTransitionEnd(600)
            } else {
                if (this.$element.trigger(j), j.isDefaultPrevented())
                    return;
                d.removeClass("active"), e.addClass("active"), this.sliding=!1, this.$element.trigger("slid.bs.carousel")
            }
            return f && this.cycle(), this
        }
    };
    var c = a.fn.carousel;
    a.fn.carousel = function(c) {
        return this.each(function() {
            var d = a(this), e = d.data("bs.carousel"), f = a.extend({}, b.DEFAULTS, d.data(), "object" == typeof c && c), g = "string" == typeof c ? c: f.slide;
            e || d.data("bs.carousel", e = new b(this, f)), "number" == typeof c ? e.to(c) : g ? e[g]() : f.interval && e.pause().cycle()
        })
    }, a.fn.carousel.Constructor = b, a.fn.carousel.noConflict = function() {
        return a.fn.carousel = c, this
    }, a(document).on("click.bs.carousel.data-api", "[data-slide], [data-slide-to]", function(b) {
        var c, d = a(this), e = a(d.attr("data-target") || (c = d.attr("href")) && c.replace(/.*(?=#[^\s]+$)/, "")), f = a.extend({}, e.data(), d.data()), g = d.attr("data-slide-to");
        g && (f.interval=!1), e.carousel(f), (g = d.attr("data-slide-to")) && e.data("bs.carousel").to(g), b.preventDefault()
    }), a(window).on("load", function() {
        a('[data-ride="carousel"]').each(function() {
            var b = a(this);
            b.carousel(b.data())
        })
    })
}(jQuery), + function(a) {
    var b = function(c, d) {
        this.$element = a(c), this.options = a.extend({}, b.DEFAULTS, d), this.transitioning = null, this.options.parent && (this.$parent = a(this.options.parent)), this.options.toggle && this.toggle()
    };
    b.DEFAULTS = {
        toggle: !0
    }, b.prototype.dimension = function() {
        var a = this.$element.hasClass("width");
        return a ? "width" : "height"
    }, b.prototype.show = function() {
        if (!this.transitioning&&!this.$element.hasClass("in")) {
            var b = a.Event("show.bs.collapse");
            if (this.$element.trigger(b), !b.isDefaultPrevented()) {
                var c = this.$parent && this.$parent.find("> .panel > .in");
                if (c && c.length) {
                    var d = c.data("bs.collapse");
                    if (d && d.transitioning)
                        return;
                    c.collapse("hide"), d || c.data("bs.collapse", null)
                }
                var e = this.dimension();
                this.$element.removeClass("collapse").addClass("collapsing")[e](0), this.transitioning = 1;
                var f = function() {
                    this.$element.removeClass("collapsing").addClass("in")[e]("auto"), this.transitioning = 0, this.$element.trigger("shown.bs.collapse")
                };
                if (!a.support.transition)
                    return f.call(this);
                var g = a.camelCase(["scroll", e].join("-"));
                this.$element.one(a.support.transition.end, a.proxy(f, this)).emulateTransitionEnd(350)[e](this.$element[0][g])
            }
        }
    }, b.prototype.hide = function() {
        if (!this.transitioning && this.$element.hasClass("in")) {
            var b = a.Event("hide.bs.collapse");
            if (this.$element.trigger(b), !b.isDefaultPrevented()) {
                var c = this.dimension();
                this.$element[c](this.$element[c]())[0].offsetHeight, this.$element.addClass("collapsing").removeClass("collapse").removeClass("in"), this.transitioning = 1;
                var d = function() {
                    this.transitioning = 0, this.$element.trigger("hidden.bs.collapse").removeClass("collapsing").addClass("collapse")
                };
                return a.support.transition ? void this.$element[c](0).one(a.support.transition.end, a.proxy(d, this)).emulateTransitionEnd(350) : d.call(this)
            }
        }
    }, b.prototype.toggle = function() {
        this[this.$element.hasClass("in") ? "hide": "show"]()
    };
    var c = a.fn.collapse;
    a.fn.collapse = function(c) {
        return this.each(function() {
            var d = a(this), e = d.data("bs.collapse"), f = a.extend({}, b.DEFAULTS, d.data(), "object" == typeof c && c);
            e || d.data("bs.collapse", e = new b(this, f)), "string" == typeof c && e[c]()
        })
    }, a.fn.collapse.Constructor = b, a.fn.collapse.noConflict = function() {
        return a.fn.collapse = c, this
    }, a(document).on("click.bs.collapse.data-api", "[data-toggle=collapse]", function(b) {
        var c, d = a(this), e = d.attr("data-target") || b.preventDefault() || (c = d.attr("href")) && c.replace(/.*(?=#[^\s]+$)/, ""), f = a(e), g = f.data("bs.collapse"), h = g ? "toggle": d.data(), i = d.attr("data-parent"), j = i && a(i);
        g && g.transitioning || (j && j.find('[data-toggle=collapse][data-parent="' + i + '"]').not(d).addClass("collapsed"), d[f.hasClass("in") ? "addClass": "removeClass"]("collapsed")), f.collapse(h)
    })
}(jQuery), + function(a) {
    function b() {
        a(d).remove(), a(e).each(function(b) {
            var d = c(a(this));
            d.hasClass("open") && (d.trigger(b = a.Event("hide.bs.dropdown")), b.isDefaultPrevented() || d.removeClass("open").trigger("hidden.bs.dropdown"))
        })
    }
    function c(b) {
        var c = b.attr("data-target");
        c || (c = b.attr("href"), c = c && /#/.test(c) && c.replace(/.*(?=#[^\s]*$)/, ""));
        var d = c && a(c);
        return d && d.length ? d : b.parent()
    }
    var d = ".dropdown-backdrop", e = "[data-toggle=dropdown]", f = function(b) {
        a(b).on("click.bs.dropdown", this.toggle)
    };
    f.prototype.toggle = function(d) {
        var e = a(this);
        if (!e.is(".disabled, :disabled")) {
            var f = c(e), g = f.hasClass("open");
            if (b(), !g) {
                if ("ontouchstart"in document.documentElement&&!f.closest(".navbar-nav").length && a('<div class="dropdown-backdrop"/>').insertAfter(a(this)).on("click", b), f.trigger(d = a.Event("show.bs.dropdown")), d.isDefaultPrevented())
                    return;
                f.toggleClass("open").trigger("shown.bs.dropdown"), e.focus()
            }
            return !1
        }
    }, f.prototype.keydown = function(b) {
        if (/(38|40|27)/.test(b.keyCode)) {
            var d = a(this);
            if (b.preventDefault(), b.stopPropagation(), !d.is(".disabled, :disabled")) {
                var f = c(d), g = f.hasClass("open");
                if (!g || g && 27 == b.keyCode)
                    return 27 == b.which && f.find(e).focus(), d.click();
                var h = a("[role=menu] li:not(.divider):visible a", f);
                if (h.length) {
                    var i = h.index(h.filter(":focus"));
                    38 == b.keyCode && i > 0 && i--, 40 == b.keyCode && i < h.length - 1 && i++, ~i || (i = 0), h.eq(i).focus()
                }
            }
        }
    };
    var g = a.fn.dropdown;
    a.fn.dropdown = function(b) {
        return this.each(function() {
            var c = a(this), d = c.data("bs.dropdown");
            d || c.data("bs.dropdown", d = new f(this)), "string" == typeof b && d[b].call(c)
        })
    }, a.fn.dropdown.Constructor = f, a.fn.dropdown.noConflict = function() {
        return a.fn.dropdown = g, this
    }, a(document).on("click.bs.dropdown.data-api", b).on("click.bs.dropdown.data-api", ".dropdown form", function(a) {
        a.stopPropagation()
    }).on("click.bs.dropdown.data-api", e, f.prototype.toggle).on("keydown.bs.dropdown.data-api", e + ", [role=menu]", f.prototype.keydown)
}(jQuery), + function(a) {
    var b = function(b, c) {
        this.options = c, this.$element = a(b), this.$backdrop = this.isShown = null, this.options.remote && this.$element.load(this.options.remote)
    };
    b.DEFAULTS = {
        backdrop: !0,
        keyboard: !0,
        show: !0
    }, b.prototype.toggle = function(a) {
        return this[this.isShown ? "hide": "show"](a)
    }, b.prototype.show = function(b) {
        var c = this, d = a.Event("show.bs.modal", {
            relatedTarget: b
        });
        this.$element.trigger(d), this.isShown || d.isDefaultPrevented() || (this.isShown=!0, this.escape(), this.$element.on("click.dismiss.modal", '[data-dismiss="modal"]', a.proxy(this.hide, this)), this.backdrop(function() {
            var d = a.support.transition && c.$element.hasClass("fade");
            c.$element.parent().length || c.$element.appendTo(document.body), c.$element.show(), d && c.$element[0].offsetWidth, c.$element.addClass("in").attr("aria-hidden", !1), c.enforceFocus();
            var e = a.Event("shown.bs.modal", {
                relatedTarget: b
            });
            d ? c.$element.find(".modal-dialog").one(a.support.transition.end, function() {
                c.$element.focus().trigger(e)
            }).emulateTransitionEnd(300) : c.$element.focus().trigger(e)
        }))
    }, b.prototype.hide = function(b) {
        b && b.preventDefault(), b = a.Event("hide.bs.modal"), this.$element.trigger(b), this.isShown&&!b.isDefaultPrevented() && (this.isShown=!1, this.escape(), a(document).off("focusin.bs.modal"), this.$element.removeClass("in").attr("aria-hidden", !0).off("click.dismiss.modal"), a.support.transition && this.$element.hasClass("fade") ? this.$element.one(a.support.transition.end, a.proxy(this.hideModal, this)).emulateTransitionEnd(300) : this.hideModal())
    }, b.prototype.enforceFocus = function() {
        a(document).off("focusin.bs.modal").on("focusin.bs.modal", a.proxy(function(a) {
            this.$element[0] !== a.target&&!this.$element.has(a.target).length && this.$element.focus()
        }, this))
    }, b.prototype.escape = function() {
        this.isShown && this.options.keyboard ? this.$element.on("keyup.dismiss.bs.modal", a.proxy(function(a) {
            27 == a.which && this.hide()
        }, this)) : this.isShown || this.$element.off("keyup.dismiss.bs.modal")
    }, b.prototype.hideModal = function() {
        var a = this;
        this.$element.hide(), this.backdrop(function() {
            a.removeBackdrop(), a.$element.trigger("hidden.bs.modal")
        })
    }, b.prototype.removeBackdrop = function() {
        this.$backdrop && this.$backdrop.remove(), this.$backdrop = null
    }, b.prototype.backdrop = function(b) {
        var c = this.$element.hasClass("fade") ? "fade": "";
        if (this.isShown && this.options.backdrop) {
            var d = a.support.transition && c;
            if (this.$backdrop = a('<div class="modal-backdrop ' + c + '" />').appendTo(document.body), this.$element.on("click.dismiss.modal", a.proxy(function(a) {
                a.target === a.currentTarget && ("static" == this.options.backdrop ? this.$element[0].focus.call(this.$element[0]) : this.hide.call(this))
            }, this)), d && this.$backdrop[0].offsetWidth, this.$backdrop.addClass("in"), !b)
                return;
            d ? this.$backdrop.one(a.support.transition.end, b).emulateTransitionEnd(150) : b()
        } else 
            !this.isShown && this.$backdrop ? (this.$backdrop.removeClass("in"), a.support.transition && this.$element.hasClass("fade") ? this.$backdrop.one(a.support.transition.end, b).emulateTransitionEnd(150) : b()) : b && b()
    };
    var c = a.fn.modal;
    a.fn.modal = function(c, d) {
        return this.each(function() {
            var e = a(this), f = e.data("bs.modal"), g = a.extend({}, b.DEFAULTS, e.data(), "object" == typeof c && c);
            f || e.data("bs.modal", f = new b(this, g)), "string" == typeof c ? f[c](d) : g.show && f.show(d)
        })
    }, a.fn.modal.Constructor = b, a.fn.modal.noConflict = function() {
        return a.fn.modal = c, this
    }, a(document).on("click.bs.modal.data-api", '[data-toggle="modal"]', function(b) {
        var c = a(this), d = c.attr("href"), e = a(c.attr("data-target") || d && d.replace(/.*(?=#[^\s]+$)/, "")), f = e.data("modal") ? "toggle": a.extend({
            remote: !/#/.test(d) && d
        }, e.data(), c.data());
        b.preventDefault(), e.modal(f, this).one("hide", function() {
            c.is(":visible") && c.focus()
        })
    }), a(document).on("show.bs.modal", ".modal", function() {
        a(document.body).addClass("modal-open")
    }).on("hidden.bs.modal", ".modal", function() {
        a(document.body).removeClass("modal-open")
    })
}(jQuery), + function(a) {
    var b = function(a, b) {
        this.type = this.options = this.enabled = this.timeout = this.hoverState = this.$element = null, this.init("tooltip", a, b)
    };
    b.DEFAULTS = {
        animation: !0,
        placement: "top",
        selector: !1,
        template: '<div class="tooltip"><div class="tooltip-arrow"></div><div class="tooltip-inner"></div></div>',
        trigger: "hover focus",
        title: "",
        delay: 0,
        html: !1,
        container: !1
    }, b.prototype.init = function(b, c, d) {
        this.enabled=!0, this.type = b, this.$element = a(c), this.options = this.getOptions(d);
        for (var e = this.options.trigger.split(" "), f = e.length; f--;) {
            var g = e[f];
            if ("click" == g)
                this.$element.on("click." + this.type, this.options.selector, a.proxy(this.toggle, this));
            else if ("manual" != g) {
                var h = "hover" == g ? "mouseenter": "focus", i = "hover" == g ? "mouseleave": "blur";
                this.$element.on(h + "." + this.type, this.options.selector, a.proxy(this.enter, this)), this.$element.on(i + "." + this.type, this.options.selector, a.proxy(this.leave, this))
            }
        }
        this.options.selector ? this._options = a.extend({}, this.options, {
            trigger: "manual",
            selector: ""
        }) : this.fixTitle()
    }, b.prototype.getDefaults = function() {
        return b.DEFAULTS
    }, b.prototype.getOptions = function(b) {
        return b = a.extend({}, this.getDefaults(), this.$element.data(), b), b.delay && "number" == typeof b.delay && (b.delay = {
            show: b.delay,
            hide: b.delay
        }), b
    }, b.prototype.getDelegateOptions = function() {
        var b = {}, c = this.getDefaults();
        return this._options && a.each(this._options, function(a, d) {
            c[a] != d && (b[a] = d)
        }), b
    }, b.prototype.enter = function(b) {
        var c = b instanceof this.constructor ? b: a(b.currentTarget)[this.type](this.getDelegateOptions()).data("bs." + this.type);
        return clearTimeout(c.timeout), c.hoverState = "in", c.options.delay && c.options.delay.show ? void(c.timeout = setTimeout(function() {
            "in" == c.hoverState && c.show()
        }, c.options.delay.show)) : c.show()
    }, b.prototype.leave = function(b) {
        var c = b instanceof this.constructor ? b: a(b.currentTarget)[this.type](this.getDelegateOptions()).data("bs." + this.type);
        return clearTimeout(c.timeout), c.hoverState = "out", c.options.delay && c.options.delay.hide ? void(c.timeout = setTimeout(function() {
            "out" == c.hoverState && c.hide()
        }, c.options.delay.hide)) : c.hide()
    }, b.prototype.show = function() {
        var b = a.Event("show.bs." + this.type);
        if (this.hasContent() && this.enabled) {
            if (this.$element.trigger(b), b.isDefaultPrevented())
                return;
            var c = this.tip();
            this.setContent(), this.options.animation && c.addClass("fade");
            var d = "function" == typeof this.options.placement ? this.options.placement.call(this, c[0], this.$element[0]): this.options.placement, e = /\s?auto?\s?/i, f = e.test(d);
            f && (d = d.replace(e, "") || "top"), c.detach().css({
                top: 0,
                left: 0,
                display: "block"
            }).addClass(d), this.options.container ? c.appendTo(this.options.container) : c.insertAfter(this.$element);
            var g = this.getPosition(), h = c[0].offsetWidth, i = c[0].offsetHeight;
            if (f) {
                var j = this.$element.parent(), k = d, l = document.documentElement.scrollTop || document.body.scrollTop, m = "body" == this.options.container ? window.innerWidth: j.outerWidth(), n = "body" == this.options.container ? window.innerHeight: j.outerHeight(), o = "body" == this.options.container ? 0: j.offset().left;
                d = "bottom" == d && g.top + g.height + i - l > n ? "top" : "top" == d && g.top - l - i < 0 ? "bottom" : "right" == d && g.right + h > m ? "left" : "left" == d && g.left - h < o ? "right" : d, c.removeClass(k).addClass(d)
            }
            var p = this.getCalculatedOffset(d, g, h, i);
            this.applyPlacement(p, d), this.$element.trigger("shown.bs." + this.type)
        }
    }, b.prototype.applyPlacement = function(a, b) {
        var c, d = this.tip(), e = d[0].offsetWidth, f = d[0].offsetHeight, g = parseInt(d.css("margin-top"), 10), h = parseInt(d.css("margin-left"), 10);
        isNaN(g) && (g = 0), isNaN(h) && (h = 0), a.top = a.top + g, a.left = a.left + h, d.offset(a).addClass("in");
        var i = d[0].offsetWidth, j = d[0].offsetHeight;
        if ("top" == b && j != f && (c=!0, a.top = a.top + f - j), /bottom|top/.test(b)) {
            var k = 0;
            a.left < 0 && (k =- 2 * a.left, a.left = 0, d.offset(a), i = d[0].offsetWidth, j = d[0].offsetHeight), this.replaceArrow(k - e + i, i, "left")
        } else 
            this.replaceArrow(j - f, j, "top");
        c && d.offset(a)
    }, b.prototype.replaceArrow = function(a, b, c) {
        this.arrow().css(c, a ? 50 * (1 - a / b) + "%" : "")
    }, b.prototype.setContent = function() {
        var a = this.tip(), b = this.getTitle();
        a.find(".tooltip-inner")[this.options.html ? "html": "text"](b), a.removeClass("fade in top bottom left right")
    }, b.prototype.hide = function() {
        function b() {
            "in" != c.hoverState && d.detach()
        }
        var c = this, d = this.tip(), e = a.Event("hide.bs." + this.type);
        return this.$element.trigger(e), e.isDefaultPrevented() ? void 0 : (d.removeClass("in"), a.support.transition && this.$tip.hasClass("fade") ? d.one(a.support.transition.end, b).emulateTransitionEnd(150) : b(), this.$element.trigger("hidden.bs." + this.type), this)
    }, b.prototype.fixTitle = function() {
        var a = this.$element;
        (a.attr("title") || "string" != typeof a.attr("data-original-title")) && a.attr("data-original-title", a.attr("title") || "").attr("title", "")
    }, b.prototype.hasContent = function() {
        return this.getTitle()
    }, b.prototype.getPosition = function() {
        var b = this.$element[0];
        return a.extend({}, "function" == typeof b.getBoundingClientRect ? b.getBoundingClientRect() : {
            width: b.offsetWidth,
            height: b.offsetHeight
        }, this.$element.offset())
    }, b.prototype.getCalculatedOffset = function(a, b, c, d) {
        return "bottom" == a ? {
            top: b.top + b.height,
            left: b.left + b.width / 2 - c / 2
        } : "top" == a ? {
            top: b.top - d,
            left: b.left + b.width / 2 - c / 2
        } : "left" == a ? {
            top: b.top + b.height / 2 - d / 2,
            left: b.left - c
        } : {
            top: b.top + b.height / 2 - d / 2,
            left: b.left + b.width
        }
    }, b.prototype.getTitle = function() {
        var a, b = this.$element, c = this.options;
        return a = b.attr("data-original-title") || ("function" == typeof c.title ? c.title.call(b[0]) : c.title)
    }, b.prototype.tip = function() {
        return this.$tip = this.$tip || a(this.options.template)
    }, b.prototype.arrow = function() {
        return this.$arrow = this.$arrow || this.tip().find(".tooltip-arrow")
    }, b.prototype.validate = function() {
        this.$element[0].parentNode || (this.hide(), this.$element = null, this.options = null)
    }, b.prototype.enable = function() {
        this.enabled=!0
    }, b.prototype.disable = function() {
        this.enabled=!1
    }, b.prototype.toggleEnabled = function() {
        this.enabled=!this.enabled
    }, b.prototype.toggle = function(b) {
        var c = b ? a(b.currentTarget)[this.type](this.getDelegateOptions()).data("bs." + this.type): this;
        c.tip().hasClass("in") ? c.leave(c) : c.enter(c)
    }, b.prototype.destroy = function() {
        this.hide().$element.off("." + this.type).removeData("bs." + this.type)
    };
    var c = a.fn.tooltip;
    a.fn.tooltip = function(c) {
        return this.each(function() {
            var d = a(this), e = d.data("bs.tooltip"), f = "object" == typeof c && c;
            e || d.data("bs.tooltip", e = new b(this, f)), "string" == typeof c && e[c]()
        })
    }, a.fn.tooltip.Constructor = b, a.fn.tooltip.noConflict = function() {
        return a.fn.tooltip = c, this
    }
}(jQuery), + function(a) {
    var b = function(a, b) {
        this.init("popover", a, b)
    };
    if (!a.fn.tooltip)
        throw new Error("Popover requires tooltip.js");
    b.DEFAULTS = a.extend({}, a.fn.tooltip.Constructor.DEFAULTS, {
        placement: "right",
        trigger: "click",
        content: "",
        template: '<div class="popover"><div class="arrow"></div><h3 class="popover-title"></h3><div class="popover-content"></div></div>'
    }), b.prototype = a.extend({}, a.fn.tooltip.Constructor.prototype), b.prototype.constructor = b, b.prototype.getDefaults = function() {
        return b.DEFAULTS
    }, b.prototype.setContent = function() {
        var a = this.tip(), b = this.getTitle(), c = this.getContent();
        a.find(".popover-title")[this.options.html ? "html": "text"](b), a.find(".popover-content")[this.options.html ? "html": "text"](c), a.removeClass("fade top bottom left right in"), a.find(".popover-title").html() || a.find(".popover-title").hide()
    }, b.prototype.hasContent = function() {
        return this.getTitle() || this.getContent()
    }, b.prototype.getContent = function() {
        var a = this.$element, b = this.options;
        return a.attr("data-content") || ("function" == typeof b.content ? b.content.call(a[0]) : b.content)
    }, b.prototype.arrow = function() {
        return this.$arrow = this.$arrow || this.tip().find(".arrow")
    }, b.prototype.tip = function() {
        return this.$tip || (this.$tip = a(this.options.template)), this.$tip
    };
    var c = a.fn.popover;
    a.fn.popover = function(c) {
        return this.each(function() {
            var d = a(this), e = d.data("bs.popover"), f = "object" == typeof c && c;
            e || d.data("bs.popover", e = new b(this, f)), "string" == typeof c && e[c]()
        })
    }, a.fn.popover.Constructor = b, a.fn.popover.noConflict = function() {
        return a.fn.popover = c, this
    }
}(jQuery), + function(a) {
    function b(c, d) {
        var e, f = a.proxy(this.process, this);
        this.$element = a(a(c).is("body") ? window : c), this.$body = a("body"), this.$scrollElement = this.$element.on("scroll.bs.scroll-spy.data-api", f), this.options = a.extend({}, b.DEFAULTS, d), this.selector = (this.options.target || (e = a(c).attr("href")) && e.replace(/.*(?=#[^\s]+$)/, "") || "") + " .nav li > a", this.offsets = a([]), this.targets = a([]), this.activeTarget = null, this.refresh(), this.process()
    }
    b.DEFAULTS = {
        offset: 10
    }, b.prototype.refresh = function() {
        var b = this.$element[0] == window ? "offset": "position";
        this.offsets = a([]), this.targets = a([]);
        {
            var c = this;
            this.$body.find(this.selector).map(function() {
                var d = a(this), e = d.data("target") || d.attr("href"), f = /^#\w/.test(e) && a(e);
                return f && f.length && [[f[b]().top + (!a.isWindow(c.$scrollElement.get(0)) && c.$scrollElement.scrollTop()), e]] || null
            }).sort(function(a, b) {
                return a[0] - b[0]
            }).each(function() {
                c.offsets.push(this[0]), c.targets.push(this[1])
            })
        }
    }, b.prototype.process = function() {
        var a, b = this.$scrollElement.scrollTop() + this.options.offset, c = this.$scrollElement[0].scrollHeight || this.$body[0].scrollHeight, d = c - this.$scrollElement.height(), e = this.offsets, f = this.targets, g = this.activeTarget;
        if (b >= d)
            return g != (a = f.last()[0]) && this.activate(a);
        for (a = e.length; a--;)
            g != f[a] && b >= e[a] && (!e[a + 1] || b <= e[a + 1]) && this.activate(f[a])
    }, b.prototype.activate = function(b) {
        this.activeTarget = b, a(this.selector).parents(".active").removeClass("active");
        var c = this.selector + '[data-target="' + b + '"],' + this.selector + '[href="' + b + '"]', d = a(c).parents("li").addClass("active");
        d.parent(".dropdown-menu").length && (d = d.closest("li.dropdown").addClass("active")), d.trigger("activate.bs.scrollspy")
    };
    var c = a.fn.scrollspy;
    a.fn.scrollspy = function(c) {
        return this.each(function() {
            var d = a(this), e = d.data("bs.scrollspy"), f = "object" == typeof c && c;
            e || d.data("bs.scrollspy", e = new b(this, f)), "string" == typeof c && e[c]()
        })
    }, a.fn.scrollspy.Constructor = b, a.fn.scrollspy.noConflict = function() {
        return a.fn.scrollspy = c, this
    }, a(window).on("load", function() {
        a('[data-spy="scroll"]').each(function() {
            var b = a(this);
            b.scrollspy(b.data())
        })
    })
}(jQuery), + function(a) {
    var b = function(b) {
        this.element = a(b)
    };
    b.prototype.show = function() {
        var b = this.element, c = b.closest("ul:not(.dropdown-menu)"), d = b.data("target");
        if (d || (d = b.attr("href"), d = d && d.replace(/.*(?=#[^\s]*$)/, "")), !b.parent("li").hasClass("active")) {
            var e = c.find(".active:last a")[0], f = a.Event("show.bs.tab", {
                relatedTarget: e
            });
            if (b.trigger(f), !f.isDefaultPrevented()) {
                var g = a(d);
                this.activate(b.parent("li"), c), this.activate(g, g.parent(), function() {
                    b.trigger({
                        type: "shown.bs.tab",
                        relatedTarget: e
                    })
                })
            }
        }
    }, b.prototype.activate = function(b, c, d) {
        function e() {
            f.removeClass("active").find("> .dropdown-menu > .active").removeClass("active"), b.addClass("active"), g ? (b[0].offsetWidth, b.addClass("in")) : b.removeClass("fade"), b.parent(".dropdown-menu") && b.closest("li.dropdown").addClass("active"), d && d()
        }
        var f = c.find("> .active"), g = d && a.support.transition && f.hasClass("fade");
        g ? f.one(a.support.transition.end, e).emulateTransitionEnd(150) : e(), f.removeClass("in")
    };
    var c = a.fn.tab;
    a.fn.tab = function(c) {
        return this.each(function() {
            var d = a(this), e = d.data("bs.tab");
            e || d.data("bs.tab", e = new b(this)), "string" == typeof c && e[c]()
        })
    }, a.fn.tab.Constructor = b, a.fn.tab.noConflict = function() {
        return a.fn.tab = c, this
    }, a(document).on("click.bs.tab.data-api", '[data-toggle="tab"], [data-toggle="pill"]', function(b) {
        b.preventDefault(), a(this).tab("show")
    })
}(jQuery), + function(a) {
    var b = function(c, d) {
        this.options = a.extend({}, b.DEFAULTS, d), this.$window = a(window).on("scroll.bs.affix.data-api", a.proxy(this.checkPosition, this)).on("click.bs.affix.data-api", a.proxy(this.checkPositionWithEventLoop, this)), this.$element = a(c), this.affixed = this.unpin = null, this.checkPosition()
    };
    b.RESET = "affix affix-top affix-bottom", b.DEFAULTS = {
        offset: 0
    }, b.prototype.checkPositionWithEventLoop = function() {
        setTimeout(a.proxy(this.checkPosition, this), 1)
    }, b.prototype.checkPosition = function() {
        if (this.$element.is(":visible")) {
            var c = a(document).height(), d = this.$window.scrollTop(), e = this.$element.offset(), f = this.options.offset, g = f.top, h = f.bottom;
            "object" != typeof f && (h = g = f), "function" == typeof g && (g = f.top()), "function" == typeof h && (h = f.bottom());
            var i = null != this.unpin && d + this.unpin <= e.top?!1 : null != h && e.top + this.$element.height() >= c - h ? "bottom" : null != g && g >= d ? "top" : !1;
            this.affixed !== i && (this.unpin && this.$element.css("top", ""), this.affixed = i, this.unpin = "bottom" == i ? e.top - d : null, this.$element.removeClass(b.RESET).addClass("affix" + (i ? "-" + i : "")), "bottom" == i && this.$element.offset({
                top: document.body.offsetHeight - h - this.$element.height()
            }))
        }
    };
    var c = a.fn.affix;
    a.fn.affix = function(c) {
        return this.each(function() {
            var d = a(this), e = d.data("bs.affix"), f = "object" == typeof c && c;
            e || d.data("bs.affix", e = new b(this, f)), "string" == typeof c && e[c]()
        })
    }, a.fn.affix.Constructor = b, a.fn.affix.noConflict = function() {
        return a.fn.affix = c, this
    }, a(window).on("load", function() {
        a('[data-spy="affix"]').each(function() {
            var b = a(this), c = b.data();
            c.offset = c.offset || {}, c.offsetBottom && (c.offset.bottom = c.offsetBottom), c.offsetTop && (c.offset.top = c.offsetTop), b.affix(c)
        })
    })
}(jQuery), define("bootstrap", ["jquery"], function() {}), function() {
    var a = "undefined" != typeof exports ? exports: this, b = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=", c = function() {
        try {
            document.createElement("$")
        } catch (a) {
            return a
        }
    }();
    a.btoa || (a.btoa = function(a) {
        for (var d, e, f = 0, g = b, h = ""; a.charAt(0 | f) || (g = "=", f%1); h += g.charAt(63 & d>>8 - f%1 * 8)) {
            if (e = a.charCodeAt(f += .75), e > 255)
                throw c;
            d = d<<8 | e
        }
        return h
    }), a.atob || (a.atob = function(a) {
        if (a = a.replace(/=+$/, ""), a.length%4 == 1)
            throw c;
        for (var d, e, f = 0, g = 0, h = ""; e = a.charAt(g++); ~e && (d = f%4 ? 64 * d + e : e, f++%4) ? h += String.fromCharCode(255 & d>>( - 2 * f & 6)) : 0)
            e = b.indexOf(e);
        return h
    })
}(), define("base64", function() {}), function(a, b, c) {
    function d(a) {
        return a
    }
    function e(a) {
        return f(decodeURIComponent(a.replace(h, " ")))
    }
    function f(a) {
        return 0 === a.indexOf('"') && (a = a.slice(1, - 1).replace(/\\"/g, '"').replace(/\\\\/g, "\\")), a
    }
    function g(a) {
        return i.json ? JSON.parse(a) : a
    }
    var h = /\+/g, i = a.cookie = function(f, h, j) {
        if (h !== c) {
            if (j = a.extend({}, i.defaults, j), null === h && (j.expires =- 1), "number" == typeof j.expires) {
                var k = j.expires, l = j.expires = new Date;
                l.setDate(l.getDate() + k)
            }
            return h = i.json ? JSON.stringify(h) : String(h), b.cookie = [encodeURIComponent(f), "=", i.raw ? h: encodeURIComponent(h), j.expires ? "; expires=" + j.expires.toUTCString(): "", j.path ? "; path=" + j.path: "", j.domain ? "; domain=" + j.domain: "", j.secure ? "; secure": ""].join("")
        }
        for (var m = i.raw ? d : e, n = b.cookie.split("; "), o = f ? null : {}, p = 0, q = n.length; q > p; p++) {
            var r = n[p].split("="), s = m(r.shift()), t = m(r.join("="));
            if (f && f === s) {
                o = g(t);
                break
            }
            f || (o[s] = g(t))
        }
        return o
    };
    i.defaults = {}, a.removeCookie = function(b, c) {
        return null !== a.cookie(b) ? (a.cookie(b, null, c), !0) : !1
    }
}(jQuery, document), define("$cookie", ["jquery"], function() {}), function() {
    define("helpers/handlebars/common", ["chaplin", "lib/session", "handlebars", "underscore"], function(a, b, c, d) {
        return c.registerHelper("or", function(a, b) {
            return a || b
        }), c.registerHelper("formatDate", function(a) {
            return moment(new Date(1e3 * a)).calendar()
        }), c.registerHelper("formatDateFull", function(a) {
            return moment(new Date(1e3 * a)).format("LT - MMM D, YYYY")
        }), c.registerHelper("formatDateRelative", function(a) {
            return moment(new Date(1e3 * a)).fromNow()
        }), c.registerHelper("formatMachineDate", function(a) {
            return moment(new Date(1e3 * a)).format()
        }), c.registerHelper("formatDateString", function(a) {
            return moment(new Date(1e3 * a)).format("MMM D, YYYY")
        }), c.registerHelper("formatMachineDate0", function(a) {
            return moment(new Date(a)).format()
        }), c.registerHelper("formatDate0", function(a) {
            return moment(new Date(a)).calendar()
        }), c.registerHelper("formatDate0Full", function(a) {
            return moment(new Date(a)).format("LT - MMM D, YYYY")
        }), c.registerHelper("formatDate0Relative", function(a) {
            return moment(new Date(a)).fromNow()
        }), c.registerHelper("tagcloud", function(a) {
            var b, e, f, g, h, i;
            return f = 24, i = 13, e = 1, h = .3, g = d.max(d.map(a.items, function(a) {
                return a.num_saves
            })), b = d.map(a.items, function(a) {
                var b, c;
                return c = a.num_saves / g * (f - i) + i, b = a.num_saves / g * (e - h) + h, "<li><a href='#' data-tag='" + a.name + "' style='font-size: " + c + "px; opacity: " + b + ";' title='" + a.num_saves + " link" + (a.num_saves > 1 ? "s" : "") + " in " + d.escape(a.name) + "'>" + d.escape(a.name) + "</a></li> "
            }).join(" "), new c.SafeString(b)
        }), c.registerHelper("ifMe", function(a) {
            return b.user.get("username") === this.username ? a.fn(this) : a.inverse(this)
        }), c.registerHelper("ifLoggedIn", function(a) {
            return b.user.get("username") ? a.fn(this) : a.inverse(this)
        }), c.registerHelper("ordinal", function(a) {
            var b;
            if (a > 3 && 21 > a)
                b = "" + a + "th";
            else 
                switch (a%10) {
                case 1:
                    b = "" + a + "st";
                    break;
                case 2:
                    b = "" + a + "nd";
                    break;
                case 3:
                    b = "" + a + "rd";
                    break;
                default:
                    b = "" + a + "th"
                }
            return "" + b
        }), c.registerHelper("shortSnippet", function(a, b) {
            var d, e;
            return a ? a.length < b ? new c.SafeString(a) : (d = a.substring(0, b - 1), (null != (e = d.match(/<p>/g)) ? e.length : void 0) !== d.match(/<\/p>/g) && (d += "&hellip;</p>"), new c.SafeString(d)) : ""
        }), c.registerHelper("loadMore", function() {
            return new c.SafeString("<div class='load-more'>\n  <div class='logo load-icon'>\n    <i class='s1'></i>\n    <i class='s2'></i>\n    <i class='s3'></i>\n    <i class='s4'></i>\n  </div>\n</div>")
        }), c.registerHelper("truncateMd5", function(a, b) {
            var d, e;
            if (a.length > b) {
                for (e = a.substr(0, b + 1); e.length && (d = e.substr( - 1), e = e.substr(0, - 1), " " !== d););
                return "" === e && (e = a.substr(0, b)), new c.SafeString(e)
            }
            return a
        }), c.registerHelper("if_eq", function(a, b) {
            return a === b.hash.compare ? b.fn(this) : b.inverse(this)
        }), c.registerHelper("if_gt", function(a, b) {
            return a > b.hash.compare ? b.fn(this) : b.inverse(this)
        }), c.registerHelper("privacy_option_check", function(a) {
            return a === this.value ? "selected" : void 0
        }), c.registerHelper("highlightUrlInTweet", function(a) {
            var b;
            return b = c.Utils.escapeExpression(a), b = b.replace(/(\bhttps?:\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/gi, "<a href='$1' class='tweet-link' target='_blank'><i class='d-icon-external'></i></a>"), new c.SafeString(b)
        }), c.registerHelper("URLEncode", function(a) {
            return encodeURIComponent(a)
        }), c.registerHelper("profileUrlBeautify", function(a) {
            return a.replace(/https?:\/\/(.*?)\/?$/, "$1")
        }), c.registerHelper("decode", function(a) {
            return d.unescape(a)
        }), c.registerHelper("monthMap", function(a) {
            var b, e, f, g;
            return g = a.since_year, f = a.since_month, e = a.now_year, b = a.now_month, new c.SafeString(d.map(d.range(e, g - 1, - 1), function(a) {
                var c, h, i;
                return h = a === g ? f : 1, i = a === e ? b : 12, c = d.map(d.range(h, i + 1), function(b) {
                    return b = 10 > b ? "0" + b : b, "<li><a href='#' data-date='" + a + b + "'>" + b + "</a></li>"
                }).join(""), "<li><span>" + a + "</span> <ul>" + c + "</ul></li>"
            }).join(""))
        })
    })
}.call(this), function() {
    var a = [].indexOf || function(a) {
        for (var b = 0, c = this.length; c > b; b++)
            if (b in this && this[b] === a)
                return b;
        return - 1
    };
    define("lib/shim", [], function() {
        return "function" != typeof String.prototype.trim && (String.prototype.trim = function() {
            return this.replace(/^\s+|\s+$/g, "")
        }), Array.prototype.indexOf ? void 0 : Array.prototype.indexOf = function(b, c) {
            var d;
            for (d = this.length>>>0, c = Number(arguments[1]) || 0, c = 0 > c ? Math.ceil(c) : Math.floor(c), 0 > c && (c += d); d > c;) {
                if (a.call(this, c) >= 0 && this[c] === b)
                    return c;
                c += 1
            }
            return - 1
        }
    })
}.call(this), function(a, b, c, d) {
    function e(b, c) {
        this.element = b, this.$context = a(b).data("api", this), this.$layers = this.$context.find(".layer");
        var d = {
            calibrateX: this.$context.data("calibrate-x") || null,
            calibrateY: this.$context.data("calibrate-y") || null,
            invertX: this.$context.data("invert-x") || null,
            invertY: this.$context.data("invert-y") || null,
            limitX: parseFloat(this.$context.data("limit-x")) || null,
            limitY: parseFloat(this.$context.data("limit-y")) || null,
            scalarX: parseFloat(this.$context.data("scalar-x")) || null,
            scalarY: parseFloat(this.$context.data("scalar-y")) || null,
            frictionX: parseFloat(this.$context.data("friction-x")) || null,
            frictionY: parseFloat(this.$context.data("friction-y")) || null,
            originX: parseFloat(this.$context.data("origin-x")) || null,
            originY: parseFloat(this.$context.data("origin-y")) || null
        };
        for (var e in d)
            null === d[e] && delete d[e];
        a.extend(this, h, c, d), this.calibrationTimer = null, this.calibrationFlag=!0, this.enabled=!1, this.depths = [], this.raf = null, this.bounds = null, this.ex = 0, this.ey = 0, this.ew = 0, this.eh = 0, this.ecx = 0, this.ecy = 0, this.erx = 0, this.ery = 0, this.cx = 0, this.cy = 0, this.ix = 0, this.iy = 0, this.mx = 0, this.my = 0, this.vx = 0, this.vy = 0, this.onMouseMove = this.onMouseMove.bind(this), this.onDeviceOrientation = this.onDeviceOrientation.bind(this), this.onOrientationTimer = this.onOrientationTimer.bind(this), this.onCalibrationTimer = this.onCalibrationTimer.bind(this), this.onAnimationFrame = this.onAnimationFrame.bind(this), this.onWindowResize = this.onWindowResize.bind(this), this.initialise()
    }
    var f = "parallax", g = 30, h = {
        relativeInput: !1,
        clipRelativeInput: !1,
        calibrationThreshold: 100,
        calibrationDelay: 500,
        supportDelay: 500,
        calibrateX: !1,
        calibrateY: !0,
        invertX: !0,
        invertY: !0,
        limitX: !1,
        limitY: !1,
        scalarX: 10,
        scalarY: 10,
        frictionX: .1,
        frictionY: .1,
        originX: .5,
        originY: .5
    };
    e.prototype.transformSupport = function(a) {
        for (var e = c.createElement("div"), f=!1, g = null, h=!1, i = null, j = null, k = 0, l = this.vendors.length; l > k; k++)
            if (null !== this.vendors[k] ? (i = this.vendors[k][0] + "transform", j = this.vendors[k][1] + "Transform") : (i = "transform", j = "transform"), e.style[j] !== d) {
                f=!0;
                break
            }
        switch (a) {
        case"2D":
            h = f;
            break;
        case"3D":
            if (f) {
                var m = c.body || c.createElement("body"), n = c.documentElement, o = n.style.overflow;
                c.body || (n.style.overflow = "hidden", n.appendChild(m), m.style.overflow = "hidden", m.style.background = ""), m.appendChild(e), e.style[j] = "translate3d(1px,1px,1px)", g = b.getComputedStyle(e).getPropertyValue(i), h = g !== d && g.length > 0 && "none" !== g, n.style.overflow = o, m.removeChild(e)
            }
        }
        return h
    }, e.prototype.ww = null, e.prototype.wh = null, e.prototype.wcx = null, e.prototype.wcy = null, e.prototype.wrx = null, e.prototype.wry = null, e.prototype.portrait = null, e.prototype.desktop=!navigator.userAgent.match(/(iPhone|iPod|iPad|Android|BlackBerry|BB10|mobi|tablet|opera mini|nexus 7)/i), e.prototype.vendors = [null, ["-webkit-", "webkit"], ["-moz-", "Moz"], ["-o-", "O"], ["-ms-", "ms"]], e.prototype.motionSupport=!!b.DeviceMotionEvent, e.prototype.orientationSupport=!!b.DeviceOrientationEvent, e.prototype.orientationStatus = 0, e.prototype.transform2DSupport = e.prototype.transformSupport("2D"), e.prototype.transform3DSupport = e.prototype.transformSupport("3D"), e.prototype.propertyCache = {}, e.prototype.initialise = function() {
        "static" === this.$context.css("position") && this.$context.css({
            position: "relative"
        }), this.accelerate(this.$context), this.updateLayers(), this.updateDimensions(), this.enable(), this.queueCalibration(this.calibrationDelay)
    }, e.prototype.updateLayers = function() {
        this.$layers = this.$context.find(".layer"), this.depths = [], this.$layers.css({
            position: "absolute",
            display: "block",
            left: 0,
            top: 0
        }), this.$layers.first().css({
            position: "relative"
        }), this.accelerate(this.$layers), this.$layers.each(a.proxy(function(b, c) {
            this.depths.push(a(c).data("depth") || 0)
        }, this))
    }, e.prototype.updateDimensions = function() {
        this.ww = b.innerWidth, this.wh = b.innerHeight, this.wcx = this.ww * this.originX, this.wcy = this.wh * this.originY, this.wrx = Math.max(this.wcx, this.ww - this.wcx), this.wry = Math.max(this.wcy, this.wh - this.wcy)
    }, e.prototype.updateBounds = function() {
        this.bounds = this.element.getBoundingClientRect(), this.ex = this.bounds.left, this.ey = this.bounds.top, this.ew = this.bounds.width, this.eh = this.bounds.height, this.ecx = this.ew * this.originX, this.ecy = this.eh * this.originY, this.erx = Math.max(this.ecx, this.ew - this.ecx), this.ery = Math.max(this.ecy, this.eh - this.ecy)
    }, e.prototype.queueCalibration = function(a) {
        clearTimeout(this.calibrationTimer), this.calibrationTimer = setTimeout(this.onCalibrationTimer, a)
    }, e.prototype.enable = function() {
        this.enabled || (this.enabled=!0, this.orientationSupport ? (this.portrait = null, b.addEventListener("deviceorientation", this.onDeviceOrientation), setTimeout(this.onOrientationTimer, this.supportDelay)) : (this.cx = 0, this.cy = 0, this.portrait=!1, b.addEventListener("mousemove", this.onMouseMove)), b.addEventListener("resize", this.onWindowResize), this.raf = requestAnimationFrame(this.onAnimationFrame))
    }, e.prototype.disable = function() {
        this.enabled && (this.enabled=!1, this.orientationSupport ? b.removeEventListener("deviceorientation", this.onDeviceOrientation) : b.removeEventListener("mousemove", this.onMouseMove), b.removeEventListener("resize", this.onWindowResize), cancelAnimationFrame(this.raf))
    }, e.prototype.calibrate = function(a, b) {
        this.calibrateX = a === d ? this.calibrateX : a, this.calibrateY = b === d ? this.calibrateY : b
    }, e.prototype.invert = function(a, b) {
        this.invertX = a === d ? this.invertX : a, this.invertY = b === d ? this.invertY : b
    }, e.prototype.friction = function(a, b) {
        this.frictionX = a === d ? this.frictionX : a, this.frictionY = b === d ? this.frictionY : b
    }, e.prototype.scalar = function(a, b) {
        this.scalarX = a === d ? this.scalarX : a, this.scalarY = b === d ? this.scalarY : b
    }, e.prototype.limit = function(a, b) {
        this.limitX = a === d ? this.limitX : a, this.limitY = b === d ? this.limitY : b
    }, e.prototype.origin = function(a, b) {
        this.originX = a === d ? this.originX : a, this.originY = b === d ? this.originY : b
    }, e.prototype.clamp = function(a, b, c) {
        return a = Math.max(a, b), a = Math.min(a, c)
    }, e.prototype.css = function(b, c, e) {
        var f = this.propertyCache[c];
        if (!f)
            for (var g = 0, h = this.vendors.length; h > g; g++)
                if (f = null !== this.vendors[g] ? a.camelCase(this.vendors[g][1] + "-" + c) : c, b.style[f] !== d) {
                    this.propertyCache[c] = f;
                    break
                }
        b.style[f] = e
    }, e.prototype.accelerate = function(a) {
        for (var b = 0, c = a.length; c > b; b++) {
            var d = a[b];
            this.css(d, "transform", "translate3d(0,0,0)"), this.css(d, "transform-style", "preserve-3d"), this.css(d, "backface-visibility", "hidden")
        }
    }, e.prototype.setPosition = function(a, b, c) {
        b += "px", c += "px", this.transform3DSupport ? this.css(a, "transform", "translate3d(" + b + "," + c + ",0)") : this.transform2DSupport ? this.css(a, "transform", "translate(" + b + "," + c + ")") : (a.style.left = b, a.style.top = c)
    }, e.prototype.onOrientationTimer = function() {
        this.orientationSupport && 0 === this.orientationStatus && (this.disable(), this.orientationSupport=!1, this.enable())
    }, e.prototype.onCalibrationTimer = function() {
        this.calibrationFlag=!0
    }, e.prototype.onWindowResize = function() {
        this.updateDimensions()
    }, e.prototype.onAnimationFrame = function() {
        this.updateBounds();
        var a = this.ix - this.cx, b = this.iy - this.cy;
        (Math.abs(a) > this.calibrationThreshold || Math.abs(b) > this.calibrationThreshold) && this.queueCalibration(0), this.portrait ? (this.mx = this.calibrateX ? b : this.iy, this.my = this.calibrateY ? a : this.ix) : (this.mx = this.calibrateX ? a : this.ix, this.my = this.calibrateY ? b : this.iy), this.mx*=this.ew * (this.scalarX / 100), this.my*=this.eh * (this.scalarY / 100), isNaN(parseFloat(this.limitX)) || (this.mx = this.clamp(this.mx, - this.limitX, this.limitX)), isNaN(parseFloat(this.limitY)) || (this.my = this.clamp(this.my, - this.limitY, this.limitY)), this.vx += (this.mx - this.vx) * this.frictionX, this.vy += (this.my - this.vy) * this.frictionY;
        for (var c = 0, d = this.$layers.length; d > c; c++) {
            var e = this.depths[c], f = this.$layers[c], g = this.vx * e * (this.invertX?-1 : 1), h = this.vy * e * (this.invertY?-1 : 1);
            this.setPosition(f, g, h)
        }
        this.raf = requestAnimationFrame(this.onAnimationFrame)
    }, e.prototype.onDeviceOrientation = function(a) {
        if (!this.desktop && null !== a.beta && null !== a.gamma) {
            this.orientationStatus = 1;
            var c = (a.beta || 0) / g, d = (a.gamma || 0) / g, e = b.innerHeight > b.innerWidth;
            this.portrait !== e && (this.portrait = e, this.calibrationFlag=!0), this.calibrationFlag && (this.calibrationFlag=!1, this.cx = c, this.cy = d), this.ix = c, this.iy = d
        }
    }, e.prototype.onMouseMove = function(a) {
        var b = a.clientX, c = a.clientY;
        !this.orientationSupport && this.relativeInput ? (this.clipRelativeInput && (b = Math.max(b, this.ex), b = Math.min(b, this.ex + this.ew), c = Math.max(c, this.ey), c = Math.min(c, this.ey + this.eh)), this.ix = (b - this.ex - this.ecx) / this.erx, this.iy = (c - this.ey - this.ecy) / this.ery) : (this.ix = (b - this.wcx) / this.wrx, this.iy = (c - this.wcy) / this.wry)
    };
    var i = {
        enable: e.prototype.enable,
        disable: e.prototype.disable,
        updateLayers: e.prototype.updateLayers,
        calibrate: e.prototype.calibrate,
        friction: e.prototype.friction,
        invert: e.prototype.invert,
        scalar: e.prototype.scalar,
        limit: e.prototype.limit,
        origin: e.prototype.origin
    };
    a.fn[f] = function(b) {
        var c = arguments;
        return this.each(function() {
            var d = a(this), g = d.data(f);
            g || (g = new e(this, b), d.data(f, g)), i[b] && g[b].apply(g, Array.prototype.slice.call(c, 1))
        })
    }
}(window.jQuery || window.Zepto, window, document), function() {
    for (var a = 0, b = ["ms", "moz", "webkit", "o"], c = 0; c < b.length&&!window.requestAnimationFrame; ++c)
        window.requestAnimationFrame = window[b[c] + "RequestAnimationFrame"], window.cancelAnimationFrame = window[b[c] + "CancelAnimationFrame"] || window[b[c] + "CancelRequestAnimationFrame"];
    window.requestAnimationFrame || (window.requestAnimationFrame = function(b) {
        var c = (new Date).getTime(), d = Math.max(0, 16 - (c - a)), e = window.setTimeout(function() {
            b(c + d)
        }, d);
        return a = c + d, e
    }), window.cancelAnimationFrame || (window.cancelAnimationFrame = function(a) {
        clearTimeout(a)
    })
}(), define("$parallax", ["jquery"], function() {}), function() {
    requirejs.config({
        paths: {
            jquery: "vendor/jquery/jquery",
            underscore: "vendor/lodash/dist/lodash",
            backbone: "vendor/backbone/backbone",
            handlebars: "vendor/handlebars.js/dist/handlebars",
            chaplin: "vendor/chaplin/chaplin",
            bootstrap: "vendor/bootstrap/dist/js/bootstrap",
            store: "vendor/store.js/store",
            base64: "vendor/base64/base64",
            $cookie: "vendor/jquery.cookie/jquery.cookie",
            moment: "vendor/momentjs/moment",
            mousetrap: "vendor/mousetrap/mousetrap",
            $parallax: "vendor/parallax/deploy/jquery.parallax"
        },
        shim: {
            underscore: {
                exports: "_"
            },
            backbone: {
                deps: ["underscore", "jquery"],
                exports: "Backbone"
            },
            bootstrap: {
                deps: ["jquery"]
            },
            handlebars: {
                exports: "Handlebars"
            },
            store: {
                exports: "store"
            },
            $cookie: {
                deps: ["jquery"]
            },
            $parallax: {
                deps: ["jquery"]
            }
        }
    }), define("62fb9a.main", ["application", "routes", "bootstrap", "base64", "$cookie", "helpers/handlebars/common", "moment", "lib/shim", "$parallax"], function(a, b) {
        var c;
        return c = new a({
            routes: b,
            controllerSuffix: "_controller"
        })
    })
}.call(this);
