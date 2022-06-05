(function() {
    function a(a, b, c) {
        c = (c || 0) - 1;
        for (var d = a ? a.length : 0; ++c < d;)
            if (a[c] === b)
                return c;
        return - 1
    }
    function b(b, c) {
        var d = typeof c;
        if (b = b.l, "boolean" == d || null == c)
            return b[c] ? 0 : - 1;
        "number" != d && "string" != d && (d = "object");
        var e = "number" == d ? c: u + c;
        return b = (b = b[d]) && b[e], "object" == d ? b&&-1 < a(b, c) ? 0 : - 1 : b ? 0 : - 1
    }
    function c(a) {
        var b = this.l, c = typeof a;
        if ("boolean" == c || null == a)
            b[a]=!0;
        else {
            "number" != c && "string" != c && (c = "object");
            var d = "number" == c ? a: u + a, b = b[c] || (b[c] = {});
            "object" == c ? (b[d] || (b[d] = [])).push(a) : b[d]=!0
        }
    }
    function d(a) {
        return a.charCodeAt(0)
    }
    function e(a, b) {
        for (var c = a.m, d = b.m, e =- 1, f = c.length; ++e < f;) {
            var g = c[e], h = d[e];
            if (g !== h) {
                if (g > h || "undefined" == typeof g)
                    return 1;
                if (h > g || "undefined" == typeof h)
                    return - 1
            }
        }
        return a.n - b.n
    }
    function f(a) {
        var b =- 1, d = a.length, e = a[0], f = a[d / 2 | 0], g = a[d - 1];
        if (e && "object" == typeof e && f && "object" == typeof f && g && "object" == typeof g)
            return !1;
        for (e = i(), e["false"] = e["null"] = e["true"] = e.undefined=!1, f = i(), f.k = a, f.l = e, f.push = c; ++b < d;)
            f.push(a[b]);
        return f
    }
    function g(a) {
        return "\\" + $[a]
    }
    function h() {
        return q.pop() || []
    }
    function i() {
        return r.pop() || {
            k: null,
            l: null,
            m: null,
            "false": !1,
            n: 0,
            "null": !1,
            number: null,
            object: null,
            push: null,
            string: null,
            "true": !1,
            undefined: !1,
            o: null
        }
    }
    function j(a) {
        return "function" != typeof a.toString && "string" == typeof(a + "")
    }
    function l(a) {
        a.length = 0, q.length < w && q.push(a)
    }
    function m(a) {
        var b = a.l;
        b && m(b), a.k = a.l = a.m = a.object = a.number = a.string = a.o = null, r.length < w && r.push(a)
    }
    function n(a, b, c) {
        b || (b = 0), "undefined" == typeof c && (c = a ? a.length : 0);
        var d =- 1;
        c = c - b || 0;
        for (var e = Array(0 > c ? 0 : c); ++d < c;)
            e[d] = a[b + d];
        return e
    }
    function o(c) {
        function q(a) {
            return a && "object" == typeof a&&!Pc(a) && vc.call(a, "__wrapped__") ? a : new r(a)
        }
        function r(a, b) {
            this.__chain__=!!b, this.__wrapped__ = a
        }
        function w(a) {
            function b() {
                if (d) {
                    var a = n(d);
                    wc.apply(a, arguments)
                }
                if (this instanceof b) {
                    var f = ab(c.prototype), a = c.apply(f, a || arguments);
                    return xb(a) ? a : f
                }
                return c.apply(e, a || arguments)
            }
            var c = a[0], d = a[2], e = a[4];
            return Oc(b, a), b
        }
        function $(a, b, c, d, e) {
            if (c) {
                var f = c(a);
                if ("undefined" != typeof f)
                    return f
            }
            if (!xb(a))
                return a;
            var g = oc.call(a);
            if (!V[g] ||!Nc.nodeClass && j(a))
                return a;
            var i = Lc[g];
            switch (g) {
            case N:
            case O:
                return new i( + a);
            case R:
            case U:
                return new i(a);
            case T:
                return f = i(a.source, C.exec(a)), f.lastIndex = a.lastIndex, f
            }
            if (g = Pc(a), b) {
                var k=!d;
                d || (d = h()), e || (e = h());
                for (var m = d.length; m--;)
                    if (d[m] == a)
                        return e[m];
                f = g ? i(a.length) : {}
            } else 
                f = g ? n(a) : $c({}, a);
            return g && (vc.call(a, "index") && (f.index = a.index), vc.call(a, "input") && (f.input = a.input)), b ? (d.push(a), e.push(f), (g ? Zc : bd)(a, function(a, g) {
                f[g] = $(a, b, c, d, e)
            }), k && (l(d), l(e)), f) : f
        }
        function ab(a) {
            return xb(a) ? Cc(a) : {}
        }
        function bb(a, b, c) {
            if ("function" != typeof a)
                return Wb;
            if ("undefined" == typeof b ||!("prototype"in a))
                return a;
            var d = a.__bindData__;
            if ("undefined" == typeof d && (Nc.funcNames && (d=!a.name), d = d ||!Nc.funcDecomp, !d)) {
                var e = tc.call(a);
                Nc.funcNames || (d=!D.test(e)), d || (d = H.test(e), Oc(a, d))
            }
            if (!1 === d ||!0 !== d && 1 & d[1])
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
            return Ub(a, b)
        }
        function cb(a) {
            function b() {
                var a = i ? g: this;
                if (e) {
                    var o = n(e);
                    wc.apply(o, arguments)
                }
                return (f || k) && (o || (o = n(arguments)), f && wc.apply(o, f), k && o.length < h) ? (d|=16, cb([c, l ? d: - 4 & d, o, null, g, h])) : (o || (o = arguments), j && (c = a[m]), this instanceof b ? (a = ab(c.prototype), o = c.apply(a, o), xb(o) ? o : a) : c.apply(a, o))
            }
            var c = a[0], d = a[1], e = a[2], f = a[3], g = a[4], h = a[5], i = 1 & d, j = 2 & d, k = 4 & d, l = 8 & d, m = c;
            return Oc(b, a), b
        }
        function db(c, d) {
            var e =- 1, g = ob(), h = c ? c.length : 0, i = h >= v && g === a, j = [];
            if (i) {
                var k = f(d);
                k ? (g = b, d = k) : i=!1
            }
            for (; ++e < h;)
                k = c[e], 0 > g(d, k) && j.push(k);
            return i && m(d), j
        }
        function fb(a, b, c, d) {
            d = (d || 0) - 1;
            for (var e = a ? a.length : 0, f = []; ++d < e;) {
                var g = a[d];
                if (g && "object" == typeof g && "number" == typeof g.length && (Pc(g) || sb(g))) {
                    b || (g = fb(g, b, c));
                    var h =- 1, i = g.length, j = f.length;
                    for (f.length += i; ++h < i;)
                        f[j++] = g[h]
                } else 
                    c || f.push(g)
            }
            return f
        }
        function gb(a, b, c, d, e, f) {
            if (c) {
                var g = c(a, b);
                if ("undefined" != typeof g)
                    return !!g
            }
            if (a === b)
                return 0 !== a || 1 / a == 1 / b;
            if (a === a&&!(a && Z[typeof a] || b && Z[typeof b]))
                return !1;
            if (null == a || null == b)
                return a === b;
            var i = oc.call(a), k = oc.call(b);
            if (i == L && (i = S), k == L && (k = S), i != k)
                return !1;
            switch (i) {
            case N:
            case O:
                return + a ==+ b;
            case R:
                return a!=+a ? b!=+b : 0 == a ? 1 / a == 1 / b : a ==+ b;
            case T:
            case U:
                return a == hc(b)
            }
            if (k = i == M, !k) {
                var m = vc.call(a, "__wrapped__"), n = vc.call(b, "__wrapped__");
                if (m || n)
                    return gb(m ? a.__wrapped__ : a, n ? b.__wrapped__ : b, c, d, e, f);
                if (i != S ||!Nc.nodeClass && (j(a) || j(b)))
                    return !1;
                if (i=!Nc.argsObject && sb(a) ? fc : a.constructor, m=!Nc.argsObject && sb(b) ? fc : b.constructor, i != m&&!(wb(i) && i instanceof i && wb(m) && m instanceof m) && "constructor"in a && "constructor"in b)
                    return !1
            }
            for (i=!e, e || (e = h()), f || (f = h()), m = e.length; m--;)
                if (e[m] == a)
                    return f[m] == b;
            var o = 0, g=!0;
            if (e.push(a), f.push(b), k) {
                if (m = a.length, o = b.length, (g = o == m) || d)
                    for (; o--;)
                        if (k = m, n = b[o], d)
                            for (; k--&&!(g = gb(a[k], n, c, d, e, f)););
                        else if (!(g = gb(a[o], n, c, d, e, f)))
                            break
            } else 
                ad(b, function(b, h, i) {
                    return vc.call(i, h) ? (o++, g = vc.call(a, h) && gb(a[h], b, c, d, e, f)) : void 0
                }), g&&!d && ad(a, function(a, b, c) {
                    return vc.call(c, b) ? g =- 1<--o : void 0
                });
            return e.pop(), f.pop(), i && (l(e), l(f)), g
        }
        function hb(a, b, c, d, e) {
            (Pc(b) ? Fb : bd)(b, function(b, f) {
                var g, h, i = b, j = a[f];
                if (b && ((h = Pc(b)) || cd(b))) {
                    for (i = d.length; i--;)
                        if (g = d[i] == b) {
                            j = e[i];
                            break
                        }
                    if (!g) {
                        var k;
                        c && (i = c(j, b), k = "undefined" != typeof i) && (j = i), k || (j = h ? Pc(j) ? j : [] : cd(j) ? j : {}), d.push(b), e.push(j), k || hb(j, b, c, d, e)
                    }
                } else 
                    c && (i = c(j, b), "undefined" == typeof i && (i = b)), "undefined" != typeof i && (j = i);
                a[f] = j
            })
        }
        function ib(a, b) {
            return a + sc(Kc() * (b - a + 1))
        }
        function jb(c, d, e) {
            var g =- 1, i = ob(), j = c ? c.length : 0, k = [], n=!d && j >= v && i === a, o = e || n ? h() : k;
            for (n && (o = f(o), i = b); ++g < j;) {
                var p = c[g], q = e ? e(p, g, c): p;
                (d?!g || o[o.length - 1] !== q : 0 > i(o, q)) && ((e || n) && o.push(q), k.push(p))
            }
            return n ? (l(o.k), m(o)) : e && l(o), k
        }
        function kb(a) {
            return function(b, c, d) {
                var e = {};
                if (c = q.createCallback(c, d, 3), Pc(b)) {
                    d =- 1;
                    for (var f = b.length; ++d < f;) {
                        var g = b[d];
                        a(e, g, c(g, d, b), b)
                    }
                } else 
                    Zc(b, function(b, d, f) {
                        a(e, b, c(b, d, f), f)
                    });
                return e
            }
        }
        function lb(a, b, c, d, e, f) {
            var g = 1 & b, h = 4 & b, i = 16 & b, j = 32 & b;
            if (!(2 & b || wb(a)))
                throw new ic;
            i&&!c.length && (b&=-17, i = c=!1), j&&!d.length && (b&=-33, j = d=!1);
            var k = a && a.__bindData__;
            return k&&!0 !== k ? (k = n(k), k[2] && (k[2] = n(k[2])), k[3] && (k[3] = n(k[3])), !g || 1 & k[1] || (k[4] = e), !g && 1 & k[1] && (b|=8), !h || 4 & k[1] || (k[5] = f), i && wc.apply(k[2] || (k[2] = []), c), j && Ac.apply(k[3] || (k[3] = []), d), k[1]|=b, lb.apply(null, k)) : (1 == b || 17 === b ? w : cb)([a, b, c, d, e, f])
        }
        function mb() {
            Y.h = K, Y.b = Y.c = Y.g = Y.i = "", Y.e = "t", Y.j=!0;
            for (var a, b = 0; a = arguments[b]; b++)
                for (var c in a)
                    Y[c] = a[c];
            b = Y.a, Y.d = /^[^,]+/.exec(b)[0], a = cc, b = "return function(" + b + "){", c = Y;
            var d = "var n,t=" + c.d + ",E=" + c.e + ";if(!t)return E;" + c.i + ";";
            c.b ? (d += "var u=t.length;n=-1;if(" + c.b + "){", Nc.unindexedChars && (d += "if(s(t)){t=t.split('')}"), d += "while(++n<u){" + c.g + ";}}else{") : Nc.nonEnumArgs && (d += "var u=t.length;n=-1;if(u&&p(t)){while(++n<u){n+='';" + c.g + ";}}else{"), Nc.enumPrototypes && (d += "var G=typeof t=='function';"), Nc.enumErrorProps && (d += "var F=t===k||t instanceof Error;");
            var e = [];
            if (Nc.enumPrototypes && e.push('!(G&&n=="prototype")'), Nc.enumErrorProps && e.push('!(F&&(n=="message"||n=="name"))'), c.j && c.f)
                d += "var C=-1,D=B[typeof t]&&v(t),u=D?D.length:0;while(++C<u){n=D[C];", e.length && (d += "if(" + e.join("&&") + "){"), d += c.g + ";", e.length && (d += "}"), d += "}";
            else if (d += "for(n in t){", c.j && e.push("m.call(t, n)"), e.length && (d += "if(" + e.join("&&") + "){"), d += c.g + ";", e.length && (d += "}"), d += "}", Nc.nonEnumShadows) {
                for (d += "if(t!==A){var i=t.constructor,r=t===(i&&i.prototype),f=t===J?I:t===k?j:L.call(t),x=y[f];", k = 0; 7 > k; k++)
                    d += "n='" + c.h[k] + "';if((!(r&&x[n])&&m.call(t,n))", c.j || (d += "||(!x[n]&&t[n]!==A[n])"), d += "){" + c.g + "}";
                d += "}"
            }
            return (c.b || Nc.nonEnumArgs) && (d += "}"), d += c.c + ";return E", a("d,j,k,m,o,p,q,s,v,A,B,y,I,J,L", b + d + "}")(bb, P, kc, vc, t, sb, Pc, zb, Y.f, lc, Z, Mc, U, mc, oc)
        }
        function nb(a) {
            return Vc[a]
        }
        function ob() {
            var b = (b = q.indexOf) === Ob ? a: b;
            return b
        }
        function pb(a) {
            return "function" == typeof a && pc.test(a)
        }
        function qb(a) {
            var b, c;
            return !a || oc.call(a) != S || (b = a.constructor, wb(b)&&!(b instanceof b)) ||!Nc.argsClass && sb(a) ||!Nc.nodeClass && j(a)?!1 : Nc.ownLast ? (ad(a, function(a, b, d) {
                return c = vc.call(d, b), !1
            }), !1 !== c) : (ad(a, function(a, b) {
                c = b
            }), "undefined" == typeof c || vc.call(a, c))
        }
        function rb(a) {
            return Wc[a]
        }
        function sb(a) {
            return a && "object" == typeof a && "number" == typeof a.length && oc.call(a) == L ||!1
        }
        function tb(a, b, c) {
            var d = Rc(a), e = d.length;
            for (b = bb(b, c, 3); e--&&(c = d[e], !1 !== b(a[c], c, a)););
            return a
        }
        function ub(a) {
            var b = [];
            return ad(a, function(a, c) {
                wb(a) && b.push(c)
            }), b.sort()
        }
        function vb(a) {
            for (var b =- 1, c = Rc(a), d = c.length, e = {}; ++b < d;) {
                var f = c[b];
                e[a[f]] = f
            }
            return e
        }
        function wb(a) {
            return "function" == typeof a
        }
        function xb(a) {
            return !(!a ||!Z[typeof a])
        }
        function yb(a) {
            return "number" == typeof a || a && "object" == typeof a && oc.call(a) == R ||!1
        }
        function zb(a) {
            return "string" == typeof a || a && "object" == typeof a && oc.call(a) == U ||!1
        }
        function Ab(a) {
            for (var b =- 1, c = Rc(a), d = c.length, e = _b(d); ++b < d;)
                e[b] = a[c[b]];
            return e
        }
        function Bb(a, b, c) {
            var d =- 1, e = ob(), f = a ? a.length : 0, g=!1;
            return c = (0 > c ? Hc(0, f + c) : c) || 0, Pc(a) ? g =- 1 < e(a, b, c) : "number" == typeof f ? g =- 1 < (zb(a) ? a.indexOf(b, c) : e(a, b, c)) : Zc(a, function(a) {
                return ++d < c ? void 0 : !(g = a === b)
            }), g
        }
        function Cb(a, b, c) {
            var d=!0;
            if (b = q.createCallback(b, c, 3), Pc(a)) {
                c =- 1;
                for (var e = a.length; ++c < e && (d=!!b(a[c], c, a)););
            } else 
                Zc(a, function(a, c, e) {
                    return d=!!b(a, c, e)
                });
            return d
        }
        function Db(a, b, c) {
            var d = [];
            if (b = q.createCallback(b, c, 3), Pc(a)) {
                c =- 1;
                for (var e = a.length; ++c < e;) {
                    var f = a[c];
                    b(f, c, a) && d.push(f)
                }
            } else 
                Zc(a, function(a, c, e) {
                    b(a, c, e) && d.push(a)
                });
            return d
        }
        function Eb(a, b, c) {
            if (b = q.createCallback(b, c, 3), !Pc(a)) {
                var d;
                return Zc(a, function(a, c, e) {
                    return b(a, c, e) ? (d = a, !1) : void 0
                }), d
            }
            c =- 1;
            for (var e = a.length; ++c < e;) {
                var f = a[c];
                if (b(f, c, a))
                    return f
            }
        }
        function Fb(a, b, c) {
            if (b && "undefined" == typeof c && Pc(a)) {
                c =- 1;
                for (var d = a.length; ++c < d&&!1 !== b(a[c], c, a););
            } else 
                Zc(a, b, c);
            return a
        }
        function Gb(a, b, c) {
            var d = a, e = a ? a.length: 0;
            if (b = b && "undefined" == typeof c ? b : bb(b, c, 3), Pc(a))
                for (; e--&&!1 !== b(a[e], e, a););
            else {
                if ("number" != typeof e)
                    var f = Rc(a), e = f.length;
                else 
                    Nc.unindexedChars && zb(a) && (d = a.split(""));
                Zc(a, function(a, c, g) {
                    return c = f ? f[--e] : --e, b(d[c], c, g)
                })
            }
            return a
        }
        function Hb(a, b, c) {
            var d =- 1, e = a ? a.length : 0, f = _b("number" == typeof e ? e : 0);
            if (b = q.createCallback(b, c, 3), Pc(a))
                for (; ++d < e;)
                    f[d] = b(a[d], d, a);
            else 
                Zc(a, function(a, c, e) {
                    f[++d] = b(a, c, e)
                });
            return f
        }
        function Ib(a, b, c) {
            var e =- 1 / 0, f = e;
            if ("function" != typeof b && c && c[b] === a && (b = null), null == b && Pc(a)) {
                c =- 1;
                for (var g = a.length; ++c < g;) {
                    var h = a[c];
                    h > f && (f = h)
                }
            } else 
                b = null == b && zb(a) ? d : q.createCallback(b, c, 3), Zc(a, function(a, c, d) {
                    c = b(a, c, d), c > e && (e = c, f = a)
                });
            return f
        }
        function Jb(a, b, c, d) {
            var e = 3 > arguments.length;
            if (b = q.createCallback(b, d, 4), Pc(a)) {
                var f =- 1, g = a.length;
                for (e && (c = a[++f]); ++f < g;)
                    c = b(c, a[f], f, a)
            } else 
                Zc(a, function(a, d, f) {
                    c = e ? (e=!1, a) : b(c, a, d, f)
                });
            return c
        }
        function Kb(a, b, c, d) {
            var e = 3 > arguments.length;
            return b = q.createCallback(b, d, 4), Gb(a, function(a, d, f) {
                c = e ? (e=!1, a) : b(c, a, d, f)
            }), c
        }
        function Lb(a) {
            var b =- 1, c = a ? a.length : 0, d = _b("number" == typeof c ? c : 0);
            return Fb(a, function(a) {
                var c = ib(0, ++b);
                d[b] = d[c], d[c] = a
            }), d
        }
        function Mb(a, b, c) {
            var d;
            if (b = q.createCallback(b, c, 3), Pc(a)) {
                c =- 1;
                for (var e = a.length; ++c < e&&!(d = b(a[c], c, a)););
            } else 
                Zc(a, function(a, c, e) {
                    return !(d = b(a, c, e))
                });
            return !!d
        }
        function Nb(a, b, c) {
            var d = 0, e = a ? a.length: 0;
            if ("number" != typeof b && null != b) {
                var f =- 1;
                for (b = q.createCallback(b, c, 3); ++f < e && b(a[f], f, a);)
                    d++
            } else if (d = b, null == d || c)
                return a ? a[0] : p;
            return n(a, 0, Ic(Hc(0, d), e))
        }
        function Ob(b, c, d) {
            if ("number" == typeof d) {
                var e = b ? b.length: 0;
                d = 0 > d ? Hc(0, e + d) : d || 0
            } else if (d)
                return d = Qb(b, c), b[d] === c ? d : - 1;
            return a(b, c, d)
        }
        function Pb(a, b, c) {
            if ("number" != typeof b && null != b) {
                var d = 0, e =- 1, f = a ? a.length : 0;
                for (b = q.createCallback(b, c, 3); ++e < f && b(a[e], e, a);)
                    d++
            } else 
                d = null == b || c ? 1 : Hc(0, b);
            return n(a, d)
        }
        function Qb(a, b, c, d) {
            var e = 0, f = a ? a.length: e;
            for (c = c ? q.createCallback(c, d, 1) : Wb, b = c(b); f > e;)
                d = e + f>>>1, c(a[d]) < b ? e = d + 1 : f = d;
            return e
        }
        function Rb(a, b, c, d) {
            return "boolean" != typeof b && null != b && (d = c, c = "function" != typeof b && d && d[b] === a ? null : b, b=!1), null != c && (c = q.createCallback(c, d, 3)), jb(a, b, c)
        }
        function Sb() {
            for (var a = 1 < arguments.length ? arguments : arguments[0], b =- 1, c = a ? Ib(gd(a, "length")) : 0, d = _b(0 > c ? 0 : c); ++b < c;)
                d[b] = gd(a, b);
            return d
        }
        function Tb(a, b) {
            var c =- 1, d = a ? a.length : 0, e = {};
            for (b ||!d || Pc(a[0]) || (b = []); ++c < d;) {
                var f = a[c];
                b ? e[f] = b[c] : f && (e[f[0]] = f[1])
            }
            return e
        }
        function Ub(a, b) {
            return 2 < arguments.length ? lb(a, 17, n(arguments, 2), null, b) : lb(a, 1, null, null, b)
        }
        function Vb(a, b, c) {
            var d, e, f, g, h, i, j, k = 0, l=!1, m=!0;
            if (!wb(a))
                throw new ic;
            if (b = Hc(0, b) || 0, !0 === c)
                var n=!0, m=!1;
            else 
                xb(c) && (n = c.leading, l = "maxWait"in c && (Hc(b, c.maxWait) || 0), m = "trailing"in c ? c.trailing : m);
            var o = function() {
                var c = b - (hd() - g);
                c > 0 ? i = yc(o, c) : (e && rc(e), c = j, e = i = j = p, c && (k = hd(), f = a.apply(h, d), i || e || (d = h = null)))
            }, q = function() {
                i && rc(i), e = i = j = p, (m || l !== b) && (k = hd(), f = a.apply(h, d), i || e || (d = h = null))
            };
            return function() {
                if (d = arguments, g = hd(), h = this, j = m && (i ||!n), !1 === l)
                    var c = n&&!i;
                else {
                    e || n || (k = g);
                    var p = l - (g - k), r = 0 >= p;
                    r ? (e && (e = rc(e)), k = g, f = a.apply(h, d)) : e || (e = yc(q, p))
                }
                return r && i ? i = rc(i) : i || b === l || (i = yc(o, b)), c && (r=!0, f = a.apply(h, d)), !r || i || e || (d = h = null), f
            }
        }
        function Wb(a) {
            return a
        }
        function Xb(a, b, c) {
            var d=!0, e = b && ub(b);
            b && (c || e.length) || (null == c && (c = b), f = r, b = a, a = q, e = ub(b)), !1 === c ? d=!1 : xb(c) && "chain"in c && (d = c.chain);
            var f = a, g = wb(f);
            Fb(e, function(c) {
                var e = a[c] = b[c];
                g && (f.prototype[c] = function() {
                    var b = this.__chain__, c = this.__wrapped__, g = [c];
                    if (wc.apply(g, arguments), g = e.apply(a, g), d || b) {
                        if (c === g && xb(g))
                            return this;
                        g = new f(g), g.__chain__ = b
                    }
                    return g
                })
            })
        }
        function Yb() {}
        function Zb(a) {
            return function(b) {
                return b[a]
            }
        }
        function $b() {
            return this.__wrapped__
        }
        c = c ? eb.defaults(_.Object(), c, eb.pick(_, J)) : _;
        var _b = c.Array, ac = c.Boolean, bc = c.Date, cc = c.Function, dc = c.Math, ec = c.Number, fc = c.Object, gc = c.RegExp, hc = c.String, ic = c.TypeError, jc = [], kc = c.Error.prototype, lc = fc.prototype, mc = hc.prototype, nc = c._, oc = lc.toString, pc = gc("^" + hc(oc).replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/toString| for [^\]]+/g, ".*?") + "$"), qc = dc.ceil, rc = c.clearTimeout, sc = dc.floor, tc = cc.prototype.toString, uc = pb(uc = fc.getPrototypeOf) && uc, vc = lc.hasOwnProperty, wc = jc.push, xc = lc.propertyIsEnumerable, yc = c.setTimeout, zc = jc.splice, Ac = jc.unshift, Bc = function() {
            try {
                var a = {}, b = pb(b = fc.defineProperty) && b, c = b(a, a, a) && b
            } catch (d) {}
            return c
        }(), Cc = pb(Cc = fc.create) && Cc, Dc = pb(Dc = _b.isArray) && Dc, Ec = c.isFinite, Fc = c.isNaN, Gc = pb(Gc = fc.keys) && Gc, Hc = dc.max, Ic = dc.min, Jc = c.parseInt, Kc = dc.random, Lc = {};
        Lc[M] = _b, Lc[N] = ac, Lc[O] = bc, Lc[Q] = cc, Lc[S] = fc, Lc[R] = ec, Lc[T] = gc, Lc[U] = hc;
        var Mc = {};
        Mc[M] = Mc[O] = Mc[R] = {
            constructor: !0,
            toLocaleString: !0,
            toString: !0,
            valueOf: !0
        }, Mc[N] = Mc[U] = {
            constructor: !0,
            toString: !0,
            valueOf: !0
        }, Mc[P] = Mc[Q] = Mc[T] = {
            constructor: !0,
            toString: !0
        }, Mc[S] = {
            constructor: !0
        }, function() {
            for (var a = K.length; a--;) {
                var b, c = K[a];
                for (b in Mc)
                    vc.call(Mc, b)&&!vc.call(Mc[b], c) && (Mc[b][c]=!1)
            }
        }(), r.prototype = q.prototype;
        var Nc = q.support = {};
        !function() {
            var a = function() {
                this.x = 1
            }, b = {
                0: 1,
                length: 1
            }, d = [];
            a.prototype = {
                valueOf: 1,
                y: 1
            };
            for (var e in new a)
                d.push(e);
            for (e in arguments);
            Nc.argsClass = oc.call(arguments) == L, Nc.argsObject = arguments.constructor == fc&&!(arguments instanceof _b), Nc.enumErrorProps = xc.call(kc, "message") || xc.call(kc, "name"), Nc.enumPrototypes = xc.call(a, "prototype"), Nc.funcDecomp=!pb(c.WinRTError) && H.test(o), Nc.funcNames = "string" == typeof cc.name, Nc.nonEnumArgs = 0 != e, Nc.nonEnumShadows=!/valueOf/.test(d), Nc.ownLast = "x" != d[0], Nc.spliceObjects = (jc.splice.call(b, 0, 1), !b[0]), Nc.unindexedChars = "xx" != "x"[0] + fc("x")[0];
            try {
                Nc.nodeClass=!(oc.call(document) == S&&!({
                    toString: 0
                }
                + ""))
            } catch (f) {
                Nc.nodeClass=!0
            }
        }(1), q.templateSettings = {
            escape: /<%-([\s\S]+?)%>/g,
            evaluate: /<%([\s\S]+?)%>/g,
            interpolate: E,
            variable: "",
            imports: {
                _: q
            }
        }, Cc || (ab = function() {
            function a() {}
            return function(b) {
                if (xb(b)) {
                    a.prototype = b;
                    var d = new a;
                    a.prototype = null
                }
                return d || c.Object()
            }
        }());
        var Oc = Bc ? function(a, b) {
            X.value = b, Bc(a, "__bindData__", X)
        }
        : Yb;
        Nc.argsClass || (sb = function(a) {
            return a && "object" == typeof a && "number" == typeof a.length && vc.call(a, "callee")&&!xc.call(a, "callee") ||!1
        });
        var Pc = Dc || function(a) {
            return a && "object" == typeof a && "number" == typeof a.length && oc.call(a) == M ||!1
        }, Qc = mb({
            a: "z",
            e: "[]",
            i: "if(!(B[typeof z]))return E",
            g: "E.push(n)"
        }), Rc = Gc ? function(a) {
            return xb(a) ? Nc.enumPrototypes && "function" == typeof a || Nc.nonEnumArgs && a.length && sb(a) ? Qc(a) : Gc(a) : []
        }
        : Qc, Sc = {
            a: "g,e,K",
            i: "e=e&&typeof K=='undefined'?e:d(e,K,3)",
            b: "typeof u=='number'",
            v: Rc,
            g: "if(e(t[n],n,g)===false)return E"
        }, Tc = {
            a: "z,H,l",
            i: "var a=arguments,b=0,c=typeof l=='number'?2:a.length;while(++b<c){t=a[b];if(t&&B[typeof t]){",
            v: Rc,
            g: "if(typeof E[n]=='undefined')E[n]=t[n]",
            c: "}}"
        }, Uc = {
            i: "if(!B[typeof t])return E;" + Sc.i,
            b: !1
        }, Vc = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        }, Wc = vb(Vc), Xc = gc("(" + Rc(Wc).join("|") + ")", "g"), Yc = gc("[" + Rc(Vc).join("") + "]", "g"), Zc = mb(Sc), $c = mb(Tc, {
            i: Tc.i.replace(";", ";if(c>3&&typeof a[c-2]=='function'){var e=d(a[--c-1],a[c--],2)}else if(c>2&&typeof a[c-1]=='function'){e=a[--c]}"),
            g: "E[n]=e?e(E[n],t[n]):t[n]"
        }), _c = mb(Tc), ad = mb(Sc, Uc, {
            j: !1
        }), bd = mb(Sc, Uc);
        wb(/x/) && (wb = function(a) {
            return "function" == typeof a && oc.call(a) == Q
        });
        var cd = uc ? function(a) {
            if (!a || oc.call(a) != S ||!Nc.argsClass && sb(a))
                return !1;
            var b = a.valueOf, c = pb(b) && (c = uc(b)) && uc(c);
            return c ? a == c || uc(a) == c : qb(a)
        }
        : qb, dd = kb(function(a, b, c) {
            vc.call(a, c) ? a[c]++ : a[c] = 1
        }), ed = kb(function(a, b, c) {
            (vc.call(a, c) ? a[c] : a[c] = []).push(b)
        }), fd = kb(function(a, b, c) {
            a[c] = b
        }), gd = Hb, hd = pb(hd = bc.now) && hd || function() {
            return (new bc).getTime()
        }, id = 8 == Jc(x + "08") ? Jc: function(a, b) {
            return Jc(zb(a) ? a.replace(F, "") : a, b || 0)
        };
        return q.after = function(a, b) {
            if (!wb(b))
                throw new ic;
            return function() {
                return 1>--a ? b.apply(this, arguments) : void 0
            }
        }, q.assign = $c, q.at = function(a) {
            var b = arguments, c =- 1, d = fb(b, !0, !1, 1), b = b[2] && b[2][b[1]] === a ? 1 : d.length, e = _b(b);
            for (Nc.unindexedChars && zb(a) && (a = a.split("")); ++c < b;)
                e[c] = a[d[c]];
            return e
        }, q.bind = Ub, q.bindAll = function(a) {
            for (var b = 1 < arguments.length ? fb(arguments, !0, !1, 1) : ub(a), c =- 1, d = b.length; ++c < d;) {
                var e = b[c];
                a[e] = lb(a[e], 1, null, null, a)
            }
            return a
        }, q.bindKey = function(a, b) {
            return 2 < arguments.length ? lb(b, 19, n(arguments, 2), null, a) : lb(b, 3, null, null, a)
        }, q.chain = function(a) {
            return a = new r(a), a.__chain__=!0, a
        }, q.compact = function(a) {
            for (var b =- 1, c = a ? a.length : 0, d = []; ++b < c;) {
                var e = a[b];
                e && d.push(e)
            }
            return d
        }, q.compose = function() {
            for (var a = arguments, b = a.length; b--;)
                if (!wb(a[b]))
                    throw new ic;
            return function() {
                for (var b = arguments, c = a.length; c--;)
                    b = [a[c].apply(this, b)];
                return b[0]
            }
        }, q.constant = function(a) {
            return function() {
                return a
            }
        }, q.countBy = dd, q.create = function(a, b) {
            var c = ab(a);
            return b ? $c(c, b) : c
        }, q.createCallback = function(a, b, c) {
            var d = typeof a;
            if (null == a || "function" == d)
                return bb(a, b, c);
            if ("object" != d)
                return Zb(a);
            var e = Rc(a), f = e[0], g = a[f];
            return 1 != e.length || g !== g || xb(g) ? function(b) {
                for (var c = e.length, d=!1; c--&&(d = gb(b[e[c]], a[e[c]], null, !0)););
                return d
            } : function(a) {
                return a = a[f], g === a && (0 !== g || 1 / g == 1 / a)
            }
        }, q.curry = function(a, b) {
            return b = "number" == typeof b ? b : + b || a.length, lb(a, 4, null, null, null, b)
        }, q.debounce = Vb, q.defaults = _c, q.defer = function(a) {
            if (!wb(a))
                throw new ic;
            var b = n(arguments, 1);
            return yc(function() {
                a.apply(p, b)
            }, 1)
        }, q.delay = function(a, b) {
            if (!wb(a))
                throw new ic;
            var c = n(arguments, 2);
            return yc(function() {
                a.apply(p, c)
            }, b)
        }, q.difference = function(a) {
            return db(a, fb(arguments, !0, !0, 1))
        }, q.filter = Db, q.flatten = function(a, b, c, d) {
            return "boolean" != typeof b && null != b && (d = c, c = "function" != typeof b && d && d[b] === a ? null : b, b=!1), null != c && (a = Hb(a, c, d)), fb(a, b)
        }, q.forEach = Fb, q.forEachRight = Gb, q.forIn = ad, q.forInRight = function(a, b, c) {
            var d = [];
            ad(a, function(a, b) {
                d.push(b, a)
            });
            var e = d.length;
            for (b = bb(b, c, 3); e--&&!1 !== b(d[e--], d[e], a);
            );
            return a
        }, q.forOwn = bd, q.forOwnRight = tb, q.functions = ub, q.groupBy = ed, q.indexBy = fd, q.initial = function(a, b, c) {
            var d = 0, e = a ? a.length: 0;
            if ("number" != typeof b && null != b) {
                var f = e;
                for (b = q.createCallback(b, c, 3); f--&&b(a[f], f, a);
                )d++
            } else 
                d = null == b || c ? 1 : b || d;
            return n(a, 0, Ic(Hc(0, e - d), e))
        }, q.intersection = function() {
            for (var c = [], d =- 1, e = arguments.length, g = h(), i = ob(), j = i === a, k = h(); ++d < e;) {
                var n = arguments[d];
                (Pc(n) || sb(n)) && (c.push(n), g.push(j && n.length >= v && f(d ? c[d] : k)))
            }
            var j = c[0], o =- 1, p = j ? j.length : 0, q = [];
            a: for (; ++o < p;) {
                var r = g[0], n = j[o];
                if (0 > (r ? b(r, n) : i(k, n))) {
                    for (d = e, (r || k).push(n); --d;)
                        if (r = g[d], 0 > (r ? b(r, n) : i(c[d], n)))
                            continue a;
                    q.push(n)
                }
            }
            for (; e--;)(r = g[e]) 
                && m(r);
            return l(g), l(k), q
        }, q.invert = vb, q.invoke = function(a, b) {
            var c = n(arguments, 2), d =- 1, e = "function" == typeof b, f = a ? a.length : 0, g = _b("number" == typeof f ? f : 0);
            return Fb(a, function(a) {
                g[++d] = (e ? b : a[b]).apply(a, c)
            }), g
        }, q.keys = Rc, q.map = Hb, q.mapValues = function(a, b, c) {
            var d = {};
            return b = q.createCallback(b, c, 3), bd(a, function(a, c, e) {
                d[c] = b(a, c, e)
            }), d
        }, q.max = Ib, q.memoize = function(a, b) {
            if (!wb(a))
                throw new ic;
            var c = function() {
                var d = c.cache, e = b ? b.apply(this, arguments): u + arguments[0];
                return vc.call(d, e) ? d[e] : d[e] = a.apply(this, arguments)
            };
            return c.cache = {}, c
        }, q.merge = function(a) {
            var b = arguments, c = 2;
            if (!xb(a))
                return a;
            if ("number" != typeof b[2] && (c = b.length), c > 3 && "function" == typeof b[c - 2])
                var d = bb(b[--c - 1], b[c--], 2);
            else 
                c > 2 && "function" == typeof b[c - 1] && (d = b[--c]);
            for (var b = n(arguments, 1, c), e =- 1, f = h(), g = h(); ++e < c;)
                hb(a, b[e], d, f, g);
            return l(f), l(g), a
        }, q.min = function(a, b, c) {
            var e = 1 / 0, f = e;
            if ("function" != typeof b && c && c[b] === a && (b = null), null == b && Pc(a)) {
                c =- 1;
                for (var g = a.length; ++c < g;) {
                    var h = a[c];
                    f > h && (f = h)
                }
            } else 
                b = null == b && zb(a) ? d : q.createCallback(b, c, 3), Zc(a, function(a, c, d) {
                    c = b(a, c, d), e > c && (e = c, f = a)
                });
            return f
        }, q.omit = function(a, b, c) {
            var d = {};
            if ("function" != typeof b) {
                var e = [];
                ad(a, function(a, b) {
                    e.push(b)
                });
                for (var e = db(e, fb(arguments, !0, !1, 1)), f =- 1, g = e.length; ++f < g;) {
                    var h = e[f];
                    d[h] = a[h]
                }
            } else 
                b = q.createCallback(b, c, 3), ad(a, function(a, c, e) {
                    b(a, c, e) || (d[c] = a)
                });
            return d
        }, q.once = function(a) {
            var b, c;
            if (!wb(a))
                throw new ic;
            return function() {
                return b ? c : (b=!0, c = a.apply(this, arguments), a = null, c)
            }
        }, q.pairs = function(a) {
            for (var b =- 1, c = Rc(a), d = c.length, e = _b(d); ++b < d;) {
                var f = c[b];
                e[b] = [f, a[f]]
            }
            return e
        }, q.partial = function(a) {
            return lb(a, 16, n(arguments, 1))
        }, q.partialRight = function(a) {
            return lb(a, 32, null, n(arguments, 1))
        }, q.pick = function(a, b, c) {
            var d = {};
            if ("function" != typeof b)
                for (var e =- 1, f = fb(arguments, !0, !1, 1), g = xb(a) ? f.length : 0; ++e < g;) {
                    var h = f[e];
                    h in a && (d[h] = a[h])
                } else 
                    b = q.createCallback(b, c, 3), ad(a, function(a, c, e) {
                        b(a, c, e) && (d[c] = a)
                    });
            return d
        }, q.pluck = gd, q.property = Zb, q.pull = function(a) {
            for (var b = arguments, c = 0, d = b.length, e = a ? a.length : 0; ++c < d;)
                for (var f =- 1, g = b[c]; ++f < e;)
                    a[f] === g && (zc.call(a, f--, 1), e--);
            return a
        }, q.range = function(a, b, c) {
            a =+ a || 0, c = "number" == typeof c ? c : + c || 1, null == b && (b = a, a = 0);
            var d =- 1;
            b = Hc(0, qc((b - a) / (c || 1)));
            for (var e = _b(b); ++d < b;)
                e[d] = a, a += c;
            return e
        }, q.reject = function(a, b, c) {
            return b = q.createCallback(b, c, 3), Db(a, function(a, c, d) {
                return !b(a, c, d)
            })
        }, q.remove = function(a, b, c) {
            var d =- 1, e = a ? a.length : 0, f = [];
            for (b = q.createCallback(b, c, 3); ++d < e;)
                c = a[d], b(c, d, a) && (f.push(c), zc.call(a, d--, 1), e--);
            return f
        }, q.rest = Pb, q.shuffle = Lb, q.sortBy = function(a, b, c) {
            var d =- 1, f = Pc(b), g = a ? a.length : 0, j = _b("number" == typeof g ? g : 0);
            for (f || (b = q.createCallback(b, c, 3)), Fb(a, function(a, c, e) {
                var g = j[++d] = i();
                f ? g.m = Hb(b, function(b) {
                    return a[b]
                }) : (g.m = h())[0] = b(a, c, e), g.n = d, g.o = a
            }), g = j.length, j.sort(e); g--;)
                a = j[g], j[g] = a.o, f || l(a.m), m(a);
            return j
        }, q.tap = function(a, b) {
            return b(a), a
        }, q.throttle = function(a, b, c) {
            var d=!0, e=!0;
            if (!wb(a))
                throw new ic;
            return !1 === c ? d=!1 : xb(c) && (d = "leading"in c ? c.leading : d, e = "trailing"in c ? c.trailing : e), W.leading = d, W.maxWait = b, W.trailing = e, Vb(a, b, W)
        }, q.times = function(a, b, c) {
            a =- 1 < (a =+ a) ? a : 0;
            var d =- 1, e = _b(a);
            for (b = bb(b, c, 1); ++d < a;)
                e[d] = b(d);
            return e
        }, q.toArray = function(a) {
            return a && "number" == typeof a.length ? Nc.unindexedChars && zb(a) ? a.split("") : n(a) : Ab(a)
        }, q.transform = function(a, b, c, d) {
            var e = Pc(a);
            if (null == c)
                if (e)
                    c = [];
                else {
                    var f = a && a.constructor;
                    c = ab(f && f.prototype)
                }
            return b && (b = q.createCallback(b, d, 4), (e ? Zc : bd)(a, function(a, d, e) {
                return b(c, a, d, e)
            })), c
        }, q.union = function() {
            return jb(fb(arguments, !0, !0))
        }, q.uniq = Rb, q.values = Ab, q.where = Db, q.without = function(a) {
            return db(a, n(arguments, 1))
        }, q.wrap = function(a, b) {
            return lb(b, 16, [a])
        }, q.xor = function() {
            for (var a =- 1, b = arguments.length; ++a < b;) {
                var c = arguments[a];
                if (Pc(c) || sb(c))
                    var d = d ? jb(db(d, c).concat(db(c, d))): c
            }
            return d || []
        }, q.zip = Sb, q.zipObject = Tb, q.collect = Hb, q.drop = Pb, q.each = Fb, q.eachRight = Gb, q.extend = $c, q.methods = ub, q.object = Tb, q.select = Db, q.tail = Pb, q.unique = Rb, q.unzip = Sb, Xb(q), q.clone = function(a, b, c, d) {
            return "boolean" != typeof b && null != b && (d = c, c = b, b=!1), $(a, b, "function" == typeof c && bb(c, d, 1))
        }, q.cloneDeep = function(a, b, c) {
            return $(a, !0, "function" == typeof b && bb(b, c, 1))
        }, q.contains = Bb, q.escape = function(a) {
            return null == a ? "" : hc(a).replace(Yc, nb)
        }, q.every = Cb, q.find = Eb, q.findIndex = function(a, b, c) {
            var d =- 1, e = a ? a.length : 0;
            for (b = q.createCallback(b, c, 3); ++d < e;)
                if (b(a[d], d, a))
                    return d;
            return - 1
        }, q.findKey = function(a, b, c) {
            var d;
            return b = q.createCallback(b, c, 3), bd(a, function(a, c, e) {
                return b(a, c, e) ? (d = c, !1) : void 0
            }), d
        }, q.findLast = function(a, b, c) {
            var d;
            return b = q.createCallback(b, c, 3), Gb(a, function(a, c, e) {
                return b(a, c, e) ? (d = a, !1) : void 0
            }), d
        }, q.findLastIndex = function(a, b, c) {
            var d = a ? a.length: 0;
            for (b = q.createCallback(b, c, 3); d--;)
                if (b(a[d], d, a))
                    return d;
            return - 1
        }, q.findLastKey = function(a, b, c) {
            var d;
            return b = q.createCallback(b, c, 3), tb(a, function(a, c, e) {
                return b(a, c, e) ? (d = c, !1) : void 0
            }), d
        }, q.has = function(a, b) {
            return a ? vc.call(a, b) : !1
        }, q.identity = Wb, q.indexOf = Ob, q.isArguments = sb, q.isArray = Pc, q.isBoolean = function(a) {
            return !0 === a ||!1 === a || a && "object" == typeof a && oc.call(a) == N ||!1
        }, q.isDate = function(a) {
            return a && "object" == typeof a && oc.call(a) == O ||!1
        }, q.isElement = function(a) {
            return a && 1 === a.nodeType ||!1
        }, q.isEmpty = function(a) {
            var b=!0;
            if (!a)
                return b;
            var c = oc.call(a), d = a.length;
            return c == M || c == U || (Nc.argsClass ? c == L : sb(a)) || c == S && "number" == typeof d && wb(a.splice)?!d : (bd(a, function() {
                return b=!1
            }), b)
        }, q.isEqual = function(a, b, c, d) {
            return gb(a, b, "function" == typeof c && bb(c, d, 2))
        }, q.isFinite = function(a) {
            return Ec(a)&&!Fc(parseFloat(a))
        }, q.isFunction = wb, q.isNaN = function(a) {
            return yb(a) && a!=+a
        }, q.isNull = function(a) {
            return null === a
        }, q.isNumber = yb, q.isObject = xb, q.isPlainObject = cd, q.isRegExp = function(a) {
            return a && Z[typeof a] && oc.call(a) == T ||!1
        }, q.isString = zb, q.isUndefined = function(a) {
            return "undefined" == typeof a
        }, q.lastIndexOf = function(a, b, c) {
            var d = a ? a.length: 0;
            for ("number" == typeof c && (d = (0 > c ? Hc(0, d + c) : Ic(c, d - 1)) + 1); d--;)
                if (a[d] === b)
                    return d;
            return - 1
        }, q.mixin = Xb, q.noConflict = function() {
            return c._ = nc, this
        }, q.noop = Yb, q.now = hd, q.parseInt = id, q.random = function(a, b, c) {
            var d = null == a, e = null == b;
            return null == c && ("boolean" == typeof a && e ? (c = a, a = 1) : e || "boolean" != typeof b || (c = b, e=!0)), d && e && (b = 1), a =+ a || 0, e ? (b = a, a = 0) : b =+ b || 0, c || a%1 || b%1 ? (c = Kc(), Ic(a + c * (b - a + parseFloat("1e-" + ((c + "").length - 1))), b)) : ib(a, b)
        }, q.reduce = Jb, q.reduceRight = Kb, q.result = function(a, b) {
            if (a) {
                var c = a[b];
                return wb(c) ? a[b]() : c
            }
        }, q.runInContext = o, q.size = function(a) {
            var b = a ? a.length: 0;
            return "number" == typeof b ? b : Rc(a).length
        }, q.some = Mb, q.sortedIndex = Qb, q.template = function(a, b, c) {
            var d = q.templateSettings;
            a = hc(a || ""), c = _c({}, c, d);
            var e, f = _c({}, c.imports, d.imports), d = Rc(f), f = Ab(f), h = 0, i = c.interpolate || G, j = "__p+='", i = gc((c.escape || G).source + "|" + i.source + "|" + (i === E ? B : G).source + "|" + (c.evaluate || G).source + "|$", "g");
            a.replace(i, function(b, c, d, f, i, k) {
                return d || (d = f), j += a.slice(h, k).replace(I, g), c && (j += "'+__e(" + c + ")+'"), i && (e=!0, j += "';" + i + ";\n__p+='"), d && (j += "'+((__t=(" + d + "))==null?'':__t)+'"), h = k + b.length, b
            }), j += "';", i = c = c.variable, i || (c = "obj", j = "with(" + c + "){" + j + "}"), j = (e ? j.replace(y, "") : j).replace(z, "$1").replace(A, "$1;"), j = "function(" + c + "){" + (i ? "" : c + "||(" + c + "={});") + "var __t,__p='',__e=_.escape" + (e ? ",__j=Array.prototype.join;function print(){__p+=__j.call(arguments,'')}" : ";") + j + "return __p}";
            try {
                var k = cc(d, "return " + j).apply(p, f)
            } catch (l) {
                throw l.source = j, l
            }
            return b ? k(b) : (k.source = j, k)
        }, q.unescape = function(a) {
            return null == a ? "" : hc(a).replace(Xc, rb)
        }, q.uniqueId = function(a) {
            var b=++s;
            return hc(null == a ? "" : a) + b
        }, q.all = Cb, q.any = Mb, q.detect = Eb, q.findWhere = Eb, q.foldl = Jb, q.foldr = Kb, q.include = Bb, q.inject = Jb, Xb(function() {
            var a = {};
            return bd(q, function(b, c) {
                q.prototype[c] || (a[c] = b)
            }), a
        }(), !1), q.first = Nb, q.last = function(a, b, c) {
            var d = 0, e = a ? a.length: 0;
            if ("number" != typeof b && null != b) {
                var f = e;
                for (b = q.createCallback(b, c, 3); f--&&b(a[f], f, a);
                )d++
            } else if (d = b, null == d || c)
                return a ? a[e - 1] : p;
            return n(a, Hc(0, e - d))
        }, q.sample = function(a, b, c) {
            return a && "number" != typeof a.length ? a = Ab(a) : Nc.unindexedChars && zb(a) && (a = a.split("")), null == b || c ? a ? a[ib(0, a.length - 1)] : p : (a = Lb(a), a.length = Ic(Hc(0, b), a.length), a)
        }, q.take = Nb, q.head = Nb, bd(q, function(a, b) {
            var c = "sample" !== b;
            q.prototype[b] || (q.prototype[b] = function(b, d) {
                var e = this.__chain__, f = a(this.__wrapped__, b, d);
                return e || null != b && (!d || c && "function" == typeof b) ? new r(f, e) : f
            })
        }), q.VERSION = "2.4.1", q.prototype.chain = function() {
            return this.__chain__=!0, this
        }, q.prototype.toString = function() {
            return hc(this.__wrapped__)
        }, q.prototype.value = $b, q.prototype.valueOf = $b, Zc(["join", "pop", "shift"], function(a) {
            var b = jc[a];
            q.prototype[a] = function() {
                var a = this.__chain__, c = b.apply(this.__wrapped__, arguments);
                return a ? new r(c, a) : c
            }
        }), Zc(["push", "reverse", "sort", "unshift"], function(a) {
            var b = jc[a];
            q.prototype[a] = function() {
                return b.apply(this.__wrapped__, arguments), this
            }
        }), Zc(["concat", "slice", "splice"], function(a) {
            var b = jc[a];
            q.prototype[a] = function() {
                return new r(b.apply(this.__wrapped__, arguments), this.__chain__)
            }
        }), Nc.spliceObjects || Zc(["pop", "shift", "splice"], function(a) {
            var b = jc[a], c = "splice" == a;
            q.prototype[a] = function() {
                var a = this.__chain__, d = this.__wrapped__, e = b.apply(d, arguments);
                return 0 === d.length && delete d[0], a || c ? new r(e, a) : e
            }
        }), q
    }
    var p, q = [], r = [], s = 0, t = {}, u =+ new Date + "", v = 75, w = 40, x = " 	\f ﻿\n\r\u2028\u2029 ᠎             　", y = /\b__p\+='';/g, z = /\b(__p\+=)''\+/g, A = /(__e\(.*?\)|\b__t\))\+'';/g, B = /\$\{([^\\}]*(?:\\.[^\\}]*)*)\}/g, C = /\w*$/, D = /^\s*function[ \n\r\t]+\w/, E = /<%=([\s\S]+?)%>/g, F = RegExp("^[" + x + "]*0+(?=.$)"), G = /($^)/, H = /\bthis\b/, I = /['\n\r\t\u2028\u2029\\]/g, J = "Array Boolean Date Error Function Math Number Object RegExp String _ attachEvent clearTimeout isFinite isNaN parseInt setTimeout".split(" "), K = "constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" "), L = "[object Arguments]", M = "[object Array]", N = "[object Boolean]", O = "[object Date]", P = "[object Error]", Q = "[object Function]", R = "[object Number]", S = "[object Object]", T = "[object RegExp]", U = "[object String]", V = {};
    V[Q]=!1, V[L] = V[M] = V[N] = V[O] = V[R] = V[S] = V[T] = V[U]=!0;
    var W = {
        leading: !1,
        maxWait: 0,
        trailing: !1
    }, X = {
        configurable: !1,
        enumerable: !1,
        value: null,
        writable: !1
    }, Y = {
        a: "",
        b: null,
        c: "",
        d: "",
        e: "",
        v: null,
        g: "",
        h: null,
        support: null,
        i: "",
        j: !1
    }, Z = {
        "boolean": !1,
        "function": !0,
        object: !0,
        number: !1,
        string: !1,
        undefined: !1
    }, $ = {
        "\\": "\\",
        "'": "'",
        "\n": "n",
        "\r": "r",
        "	": "t",
        "\u2028": "u2028",
        "\u2029": "u2029"
    }, _ = Z[typeof window] && window || this, ab = Z[typeof exports] && exports&&!exports.nodeType && exports, bb = Z[typeof module] && module&&!module.nodeType && module, cb = bb && bb.exports === ab && ab, db = Z[typeof global] && global;
    !db || db.global !== db && db.window !== db || (_ = db);
    var eb = o();
    "function" == typeof define && "object" == typeof define.amd && define.amd ? (_._ = eb, define(function() {
        return eb
    })) : ab && bb ? cb ? (bb.exports = eb)._ = eb : ab._ = eb : _._ = eb
}).call(this), function(a) {
    function b(a) {
        return a = decodeURIComponent(a), a = a.replace("+", " ")
    }
    function c(a) {
        var b = /^(?:(?![^:@]+:[^:@\/]*@)([^:\/?#.]+):)?(?:\/\/)?((?:(([^:@]*)(?::([^:@]*))?)?@)?([^:\/?#]*)(?::(\d*))?)(((\/(?:[^?#](?![^?#\/]*\.[^?#\/.]+(?:[?#]|$)))*\/?)?([^?#\/]*))(?:\?([^#]*))?(?:#(.*))?)/, c = ["source", "protocol", "authority", "userInfo", "user", "password", "host", "port", "relative", "path", "directory", "file", "query", "anchor"], d = b.exec(a || ""), e = {};
        return c.forEach(function(a, b) {
            e[a] = d[b] || ""
        }), e
    }
    function d(a) {
        var b, c, d, e, f, g, h = [];
        if ("undefined" == typeof a || null === a || "" === a)
            return h;
        for (0 === a.indexOf("?") && (a = a.substring(1)), c = a.toString().split(/[&;]/), b = 0; b < c.length; b++)
            d = c[b], e = d.split("="), f = e[0], g =- 1 === d.indexOf("=") ? null : null === e[1] ? "" : e[1], h.push([f, g]);
        return h
    }
    function e(a) {
        this.uriParts = c(a), this.queryPairs = d(this.uriParts.query), this.hasAuthorityPrefixUserPref = null
    }
    Array.prototype.forEach || (Array.prototype.forEach = function(a, b) {
        for (var c = 0, d = this.length; d > c; ++c)
            a.call(b || this, this[c], c, this)
    }), ["protocol", "userInfo", "host", "port", "path", "anchor"].forEach(function(a) {
        e.prototype[a] = function(b) {
            return "undefined" != typeof b && (this.uriParts[a] = b), this.uriParts[a]
        }
    }), e.prototype.hasAuthorityPrefix = function(a) {
        return "undefined" != typeof a && (this.hasAuthorityPrefixUserPref = a), null === this.hasAuthorityPrefixUserPref?-1 !== this.uriParts.source.indexOf("//") : this.hasAuthorityPrefixUserPref
    }, e.prototype.query = function(a) {
        var b, c, e = "";
        for ("undefined" != typeof a && (this.queryPairs = d(a)), b = 0; b < this.queryPairs.length; b++)
            c = this.queryPairs[b], e.length > 0 && (e += "&"), null === c[1] ? e += c[0] : (e += c[0], e += "=", c[1] && (e += encodeURIComponent(c[1])));
        return e.length > 0 ? "?" + e : e
    }, e.prototype.getQueryParamValue = function(a) {
        var c, d;
        for (d = 0; d < this.queryPairs.length; d++)
            if (c = this.queryPairs[d], b(a) === b(c[0]))
                return c[1]
    }, e.prototype.getParams = function() {
        var a, b, c, d, e = {};
        for (a = 0; a < this.queryPairs.length; a++)
            b = this.queryPairs[a], c = decodeURIComponent(b[0]), d = decodeURIComponent(b[1]), e[c] = e[c] ? [].concat(e[c]).concat(d) : d;
        return e
    }, e.prototype.getQueryParamValues = function(a) {
        var c, d, e = [];
        for (c = 0; c < this.queryPairs.length; c++)
            d = this.queryPairs[c], b(a) === b(d[0]) && e.push(d[1]);
        return e
    }, e.prototype.deleteQueryParam = function(a, c) {
        var d, e, f, g, h = [];
        for (d = 0; d < this.queryPairs.length; d++)
            e = this.queryPairs[d], f = b(e[0]) === b(a), g = b(e[1]) === b(c), (1 === arguments.length&&!f || 2 === arguments.length&&!f&&!g) && h.push(e);
        return this.queryPairs = h, this
    }, e.prototype.addQueryParam = function(a, b, c) {
        return 3 === arguments.length&&-1 !== c ? (c = Math.min(c, this.queryPairs.length), this.queryPairs.splice(c, 0, [a, b])) : arguments.length > 0 && this.queryPairs.push([a, b]), this
    }, e.prototype.replaceQueryParam = function(a, c, d) {
        var e, f, g =- 1;
        if (3 === arguments.length) {
            for (e = 0; e < this.queryPairs.length; e++)
                if (f = this.queryPairs[e], b(f[0]) === b(a) && decodeURIComponent(f[1]) === b(d)) {
                    g = e;
                    break
                }
            this.deleteQueryParam(a, d).addQueryParam(a, c, g)
        } else {
            for (e = 0; e < this.queryPairs.length; e++)
                if (f = this.queryPairs[e], b(f[0]) === b(a)) {
                    g = e;
                    break
                }
            this.deleteQueryParam(a), this.addQueryParam(a, c, g)
        }
        return this
    }, ["protocol", "hasAuthorityPrefix", "userInfo", "host", "port", "path", "query", "anchor"].forEach(function(a) {
        var b = "set" + a.charAt(0).toUpperCase() + a.slice(1);
        e.prototype[b] = function(b) {
            return this[a](b), this
        }
    }), e.prototype.scheme = function() {
        var a = "";
        return this.protocol() ? (a += this.protocol(), this.protocol().indexOf(":") !== this.protocol().length - 1 && (a += ":"), a += "//") : this.hasAuthorityPrefix() && this.host() && (a += "//"), a
    }, e.prototype.origin = function() {
        var a = this.scheme();
        return this.userInfo() && this.host() && (a += this.userInfo(), this.userInfo().indexOf("@") !== this.userInfo().length - 1 && (a += "@")), this.host() && (a += this.host(), this.port() && (a += ":" + this.port())), a
    }, e.prototype.toString = function() {
        var a = this.origin();
        return this.path() ? a += this.path() : this.host() && (this.query().toString() || this.anchor()) && (a += "/"), this.query().toString() && (0 !== this.query().toString().indexOf("?") && (a += "?"), a += this.query().toString()), this.anchor() && (0 !== this.anchor().indexOf("#") && (a += "#"), a += this.anchor()), a
    }, e.prototype.clone = function() {
        return new e(this.toString())
    }, "undefined" == typeof module ? a.Uri = e : module.exports = e
}(this), Uri.prototype.getQueryParamValue = function(a) {
    var b, c;
    for (c = 0; c < this.queryPairs.length; c++)
        if (b = this.queryPairs[c], decodeURIComponent(a) === decodeURIComponent(b[0]))
            return decodeURIComponent(b[1])
}, Uri.prototype.getQueryParamValues = function(a) {
    var b, c, d = [];
    for (b = 0; b < this.queryPairs.length; b++)
        c = this.queryPairs[b], decodeURIComponent(a) === decodeURIComponent(c[0]) && d.push(decodeURIComponent(c[1]));
    return d
}, function(a, b) {
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
            d = a[g], d.style && (f[g] = kb._data(d, "olddisplay"), c = d.style.display, b ? (f[g] || "none" !== c || (d.style.display = ""), "" === d.style.display && x(d) && (f[g] = kb._data(d, "olddisplay", C(d.nodeName)))) : f[g] || (e = x(d), (c && "none" !== c ||!e) && kb._data(d, "olddisplay", e ? c : kb.css(d, "display"))));
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
    function G(a, c, d, e) {
        function f(i) {
            var j;
            return g[i]=!0, kb.each(a[i] || [], function(a, i) {
                var k = i(c, d, e);
                return "string" != typeof k || h || g[k] ? h?!(j = k) : b : (c.dataTypes.unshift(k), f(k), !1)
            }), j
        }
        var g = {}, h = a === Rc;
        return f(c.dataTypes[0]) ||!g["*"] && f("*")
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
        return g ? (g !== j[0] && j.unshift(g), d[g]) : b
    }
    function J(a, b, c, d) {
        var e, f, g, h, i, j = {}, k = a.dataTypes.slice();
        if (k[1])
            for (g in a.converters)
                j[g.toLowerCase()] = a.converters[g];
        for (f = k.shift(); f;)
            if (a.responseFields[f] && (c[a.responseFields[f]] = b), !i && d && a.dataFilter && (b = a.dataFilter(b, a.dataType)), i = f, f = k.shift())
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
                if (e = "<" === a.charAt(0) && ">" === a.charAt(a.length - 1) && a.length >= 3 ? [null, a, null] : ob.exec(a), !e ||!e[1] && c)
                    return !c || c.jquery ? (c || d).find(a) : this.constructor(c).find(a);
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
        for ("boolean" == typeof h && (k = h, h = arguments[1] || {}, i = 2), "object" == typeof h || kb.isFunction(h) || (h = {}), j === i && (h = this, --i); j > i; i++)
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
            return null == a ? a + "" : "object" == typeof a || "function" == typeof a ? ab[hb.call(a)] || "object" : typeof a
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
            throw Error(a)
        },
        parseHTML: function(a, b, c) {
            if (!a || "string" != typeof a)
                return null;
            "boolean" == typeof b && (c = b, b=!1), b = b || Y;
            var d = pb.exec(a), e=!c && [];
            return d ? [b.createElement(d[1])] : (d = kb.buildFragment([a], b, e), e && kb(e).remove(), kb.merge([], d.childNodes))
        },
        parseJSON: function(c) {
            return a.JSON && a.JSON.parse ? a.JSON.parse(c) : null === c ? c : "string" == typeof c && (c = kb.trim(c), c && qb.test(c.replace(sb, "@").replace(tb, "]").replace(rb, ""))) ? Function("return " + c)() : (kb.error("Invalid JSON: " + c), b)
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
            return d && d.documentElement&&!d.getElementsByTagName("parsererror").length || kb.error("Invalid XML: " + c), d
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
        trim: jb&&!jb.call("﻿ ") ? function(a) {
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
                for (g in z.filter)
                    !(e = rb[g].exec(h)) || j[g]&&!(e = j[g](e)) || (d = e.shift(), f.push({
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
                                s[r] || o[r] || (o[r] = $.call(j));
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
        }, db = "checked|selected|async|autofocus|autoplay|controls|defer|disabled|hidden|ismap|loop|multiple|open|readonly|required|scoped", eb = "[\\x20\\t\\r\\n\\f]", fb = "(?:\\\\.|[\\w-]|[^\\x00-\\xa0])+", gb = fb.replace("w", "w#"), hb = "\\[" + eb + "*(" + fb + ")" + eb + "*(?:([*^$|!~]?=)" + eb + "*(?:(['\"])((?:\\\\.|[^\\\\])*?)\\3|(" + gb + ")|)|)" + eb + "*\\]", ib = ":(" + fb + ")(?:\\(((['\"])((?:\\\\.|[^\\\\])*?)\\3|((?:\\\\.|[^\\\\()[\\]]|" + hb.replace(3, 8) + ")*)|.*)\\)|)", jb = RegExp("^" + eb + "+|((?:^|[^\\\\])(?:\\\\.)*)" + eb + "+$", "g"), lb = RegExp("^" + eb + "*," + eb + "*"), mb = RegExp("^" + eb + "*([>+~]|" + eb + ")" + eb + "*"), nb = RegExp(eb + "*[+~]"), ob = RegExp("=" + eb + "*([^\\]'\"]*)" + eb + "*\\]", "g"), pb = RegExp(ib), qb = RegExp("^" + gb + "$"), rb = {
            ID: RegExp("^#(" + fb + ")"),
            CLASS: RegExp("^\\.(" + fb + ")"),
            TAG: RegExp("^(" + fb.replace("w", "w*") + ")"),
            ATTR: RegExp("^" + hb),
            PSEUDO: RegExp("^" + ib),
            CHILD: RegExp("^:(only|first|last|nth|nth-last)-(child|of-type)(?:\\(" + eb + "*(even|odd|(([+-]|)(\\d*)n|)" + eb + "*(?:([+-]|)" + eb + "*(\\d+)|))" + eb + "*\\)|)", "i"),
            bool: RegExp("^(?:" + db + ")$", "i"),
            needsContext: RegExp("^" + eb + "*[>+~]|:(even|odd|eq|gt|lt|nth|first|last)(?:\\(" + eb + "*((?:-\\d)?\\d*)" + eb + "*\\)|)(?=[^-]|$)", "i")
        }, sb = /^[^{]+\{\s*\[native \w/, tb = /^(?:#([\w-]+)|(\w+)|\.([\w-]+))$/, ub = /^(?:input|select|textarea|button)$/i, vb = /^h\d$/i, wb = /'|\\/g, xb = RegExp("\\\\([\\da-f]{1,6}" + eb + "?|(" + eb + ")|.)", "ig"), yb = function(a, b, c) {
            var d = "0x" + b - 65536;
            return d !== d || c ? b : 0 > d ? String.fromCharCode(d + 65536) : String.fromCharCode(55296 | d>>10, 56320 | 1023 & d)
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
            var c = a ? a.ownerDocument || a: O, d = c.defaultView;
            return c !== G && 9 === c.nodeType && c.documentElement ? (G = c, H = c.documentElement, I=!B(c), d && d.attachEvent && d !== d.top && d.attachEvent("onbeforeunload", function() {
                F()
            }), x.attributes = f(function(a) {
                return a.className = "i", !a.getAttribute("className")
            }), x.getElementsByTagName = f(function(a) {
                return a.appendChild(c.createComment("")), !a.getElementsByTagName("*").length
            }), x.getElementsByClassName = f(function(a) {
                return a.innerHTML = "<div class='a'></div><div class='a i'></div>", a.firstChild.className = "i", 2 === a.getElementsByClassName("i").length
            }), x.getById = f(function(a) {
                return H.appendChild(a).id = N, !c.getElementsByName ||!c.getElementsByName(N).length
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
            }), z.find.TAG = x.getElementsByTagName ? function(a, c) {
                return typeof c.getElementsByTagName !== W ? c.getElementsByTagName(a) : b
            } : function(a, b) {
                var c, d = [], e = 0, f = b.getElementsByTagName(a);
                if ("*" === a) {
                    for (; c = f[e++];)
                        1 === c.nodeType && d.push(c);
                    return d
                }
                return f
            }, z.find.CLASS = x.getElementsByClassName && function(a, c) {
                return typeof c.getElementsByClassName !== W && I ? c.getElementsByClassName(a) : b
            }, K = [], J = [], (x.qsa = sb.test(c.querySelectorAll)) && (f(function(a) {
                a.innerHTML = "<select><option selected=''></option></select>", a.querySelectorAll("[selected]").length || J.push("\\[" + eb + "*(?:value|" + db + ")"), a.querySelectorAll(":checked").length || J.push(":checked")
            }), f(function(a) {
                var b = c.createElement("input");
                b.setAttribute("type", "hidden"), a.appendChild(b).setAttribute("t", ""), a.querySelectorAll("[t^='']").length && J.push("[*^$]=" + eb + "*(?:''|\"\")"), a.querySelectorAll(":enabled").length || J.push(":enabled", ":disabled"), a.querySelectorAll("*,:x"), J.push(",.*:")
            })), (x.matchesSelector = sb.test(L = H.webkitMatchesSelector || H.mozMatchesSelector || H.oMatchesSelector || H.msMatchesSelector)) && f(function(a) {
                x.disconnectedMatch = L.call(a, "div"), L.call(a, "[s!='']:x"), K.push("!=", ib)
            }), J = J.length && RegExp(J.join("|")), K = K.length && RegExp(K.join("|")), M = sb.test(H.contains) || H.compareDocumentPosition ? function(a, b) {
                var c = 9 === a.nodeType ? a.documentElement: a, d = b && b.parentNode;
                return a === d ||!(!d || 1 !== d.nodeType ||!(c.contains ? c.contains(d) : a.compareDocumentPosition && 16 & a.compareDocumentPosition(d)))
            } : function(a, b) {
                if (b)
                    for (; b = b.parentNode;)
                        if (b === a)
                            return !0;
                return !1
            }, V = H.compareDocumentPosition ? function(a, b) {
                if (a === b)
                    return U=!0, 0;
                var d = b.compareDocumentPosition && a.compareDocumentPosition && a.compareDocumentPosition(b);
                return d ? 1 & d ||!x.sortDetached && b.compareDocumentPosition(a) === d ? a === c || M(O, a)?-1 : b === c || M(O, b) ? 1 : E ? cb.call(E, a) - cb.call(E, b) : 0 : 4 & d?-1 : 1 : a.compareDocumentPosition?-1 : 1
            } : function(a, b) {
                var d, e = 0, f = a.parentNode, g = b.parentNode, i = [a], j = [b];
                if (a === b)
                    return U=!0, 0;
                if (!f ||!g)
                    return a === c?-1 : b === c ? 1 : f?-1 : g ? 1 : E ? cb.call(E, a) - cb.call(E, b) : 0;
                if (f === g)
                    return h(a, b);
                for (d = a; d = d.parentNode;)
                    i.unshift(d);
                for (d = b; d = d.parentNode;)
                    j.unshift(d);
                for (; i[e] === j[e];)
                    e++;
                return e ? h(i[e], j[e]) : i[e] === O?-1 : j[e] === O ? 1 : 0
            }, c) : G
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
            throw Error("Syntax error, unrecognized expression: " + a)
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
                    return b || (b = RegExp("(^|" + eb + ")" + a + "(" + eb + "|$)")) && R(a, function(a) {
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
                            return m -= e, m === d || 0 === m%d && m / d >= 0
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
                    for (var d = 0 > c ? c + b : c; b>++d;)
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
        }) || g("type|href|height|width", function(a, c, d) {
            return d ? b : a.getAttribute(c, "type" === c.toLowerCase() ? 1 : 2)
        }), x.attributes && f(function(a) {
            return a.innerHTML = "<input/>", a.firstChild.setAttribute("value", ""), "" === a.firstChild.getAttribute("value")
        }) || g("value", function(a, c, d) {
            return d || "input" !== a.nodeName.toLowerCase() ? b : a.defaultValue
        }), f(function(a) {
            return null == a.getAttribute("disabled")
        }) || g(db, function(a, c, d) {
            var e;
            return d ? b : (e = a.getAttributeNode(c)) && e.specified ? e.value : a[c]===!0 ? c.toLowerCase() : null
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
                            "function" === e ? a.unique && m.has(c) || j.push(c) : c && c.length && "string" !== e && d(c)
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
                return a ? kb.inArray(a, j)>-1 : !(!j ||!j.length)
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
                return !j || f&&!k || (b = b || [], b = [a, b.slice ? b.slice(): b], c ? k.push(b) : l(b)), this
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
                for (b = Array(g), c = Array(g), d = Array(g); g > e; e++)
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
                    for (d = i.attributes; d.length > h; h++)
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
        queue: function(a, c, d) {
            var e;
            return a ? (c = (c || "fx") + "queue", e = kb._data(a, c), d && (!e || kb.isArray(d) ? e = kb._data(a, c, kb.makeArray(d)) : e.push(d)), e || []) : b
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
            return "string" != typeof a && (c = a, a = "fx", d--), d > arguments.length ? kb.queue(this[0], a) : c === b ? this : this.each(function() {
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
                            0 > d.indexOf(" " + e + " ") && (d += e + " ");
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
            return arguments.length ? (e = kb.isFunction(a), this.each(function(c) {
                var f;
                1 === this.nodeType && (f = e ? a.call(this, c, kb(this).val()) : a, null == f ? f = "" : "number" == typeof f ? f += "" : kb.isArray(f) && (f = kb.map(f, function(a) {
                    return null == a ? "" : a + ""
                })), d = kb.valHooks[this.type] || kb.valHooks[this.nodeName.toLowerCase()], d && "set"in d && d.set(this, f, "value") !== b || (this.value = f))
            })) : f ? (d = kb.valHooks[f.type] || kb.valHooks[f.nodeName.toLowerCase()], d && "get"in d && (c = d.get(f, "value")) !== b ? c : (c = f.value, "string" == typeof c ? c.replace(Fb, "") : null == c ? "" : c)) : void 0
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
            return a && 3 !== g && 8 !== g && 2 !== g ? typeof a.getAttribute === W ? kb.prop(a, c, d) : (1 === g && kb.isXMLDoc(a) || (c = c.toLowerCase(), e = kb.attrHooks[c] || (kb.expr.match.bool.test(c) ? Db : Cb)), d === b ? e && "get"in e && null !== (f = e.get(a, c)) ? f : (f = kb.find.attr(a, c), null == f ? b : f) : null !== d ? e && "set"in e && (f = e.set(a, d, c)) !== b ? f : (a.setAttribute(c, d + ""), d) : (kb.removeAttr(a, c), b)) : void 0
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
            return a && 3 !== h && 8 !== h && 2 !== h ? (g = 1 !== h ||!kb.isXMLDoc(a), g && (c = kb.propFix[c] || c, f = kb.propHooks[c]), d !== b ? f && "set"in f && (e = f.set(a, d, c)) !== b ? e : a[c] = d : f && "get"in f && null !== (e = f.get(a, c)) ? e : a[c]) : void 0
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
        set: function(a, c, d) {
            return kb.nodeName(a, "input") ? (a.defaultValue = c, b) : Cb && Cb.set(a, c, d)
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
    }, kb.each(["width", "height"], function(a, c) {
        kb.attrHooks[c] = {
            set: function(a, d) {
                return "" === d ? (a.setAttribute(c, "auto"), d) : b
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
            set: function(a, c) {
                return kb.isArray(c) ? a.checked = kb.inArray(kb(a).val(), c) >= 0 : b
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
                        for (l = kb.event.special[n] || {}, n = (d ? l.delegateType : l.bindType) || n, m = k[n] || [], h = h[2] && RegExp("(^|\\.)" + o.join("\\.(?:.*\\.|)") + "(\\.|$)"), i = f = m.length; f--;)
                            g = m[f], !e && p !== g.origType || c && c.guid !== g.guid || h&&!h.test(g.namespace) || d && d !== g.selector && ("**" !== d ||!g.selector) || (m.splice(f, 1), g.selector && m.delegateCount--, l.remove && l.remove.call(a, g));
                            i&&!m.length && (l.teardown && l.teardown.call(a, o, q.handle)!==!1 || kb.removeEvent(a, n, q.handle), delete k[n])
                    } else 
                        for (n in k)
                            kb.event.remove(a, n + b[j], c, d, !0);
                kb.isEmptyObject(k) && (delete q.handle, kb._removeData(a, "events"))
            }
        },
        trigger: function(c, d, e, f) {
            var g, h, i, j, k, l, m, n = [e || Y], o = ib.call(c, "type") ? c.type: c, p = ib.call(c, "namespace") ? c.namespace.split("."): [];
            if (i = l = e = e || Y, 3 !== e.nodeType && 8 !== e.nodeType&&!Ob.test(o + kb.event.triggered) && (o.indexOf(".") >= 0 && (p = o.split("."), o = p.shift(), p.sort()), h = 0 > o.indexOf(":") && "on" + o, c = c[kb.expando] ? c : new kb.Event(o, "object" == typeof c && c), c.isTrigger = f ? 2 : 3, c.namespace = p.join("."), c.namespace_re = c.namespace ? RegExp("(^|\\.)" + p.join("\\.(?:.*\\.|)") + "(\\.|$)") : null, c.result = b, c.target || (c.target = e), d = null == d ? [c] : kb.makeArray(d, [c]), k = kb.event.special[o] || {}, f ||!k.trigger || k.trigger.apply(e, d)!==!1)) {
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
            return c.length > i && h.push({
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
                return null == a.pageX && null != c.clientX && (e = a.target.ownerDocument || Y, f = e.documentElement, d = e.body, a.pageX = c.clientX + (f && f.scrollLeft || d && d.scrollLeft || 0) - (f && f.clientLeft || d && d.clientLeft || 0), a.pageY = c.clientY + (f && f.scrollTop || d && d.scrollTop || 0) - (f && f.clientTop || d && d.clientTop || 0)), !a.relatedTarget && h && (a.relatedTarget = h === a.target ? c.toElement : h), a.which || g === b || (a.which = 1 & g ? 1 : 2 & g ? 3 : 4 & g ? 2 : 0), a
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
                    return this === k() && this.blur ? (this.blur(), !1) : b
                },
                delegateType: "focusout"
            },
            click: {
                trigger: function() {
                    return kb.nodeName(this, "input") && "checkbox" === this.type && this.click ? (this.click(), !1) : b
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
    }, kb.Event = function(a, c) {
        return this instanceof kb.Event ? (a && a.type ? (this.originalEvent = a, this.type = a.type, this.isDefaultPrevented = a.defaultPrevented || a.returnValue===!1 || a.getPreventDefault && a.getPreventDefault() ? i : j) : this.type = a, c && kb.extend(this, c), this.timeStamp = a && a.timeStamp || kb.now(), this[kb.expando]=!0, b) : new kb.Event(a, c)
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
            return kb.nodeName(this, "form")?!1 : (kb.event.add(this, "click._submit keypress._submit", function(a) {
                var c = a.target, d = kb.nodeName(c, "input") || kb.nodeName(c, "button") ? c.form: b;
                d&&!kb._data(d, "submitBubbles") && (kb.event.add(d, "submit._submit", function(a) {
                    a._submit_bubble=!0
                }), kb._data(d, "submitBubbles", !0))
            }), b)
        },
        postDispatch: function(a) {
            a._submit_bubble && (delete a._submit_bubble, this.parentNode&&!a.isTrigger && kb.event.simulate("submit", this.parentNode, a, !0))
        },
        teardown: function() {
            return kb.nodeName(this, "form")?!1 : (kb.event.remove(this, "._submit"), b)
        }
    }), kb.support.changeBubbles || (kb.event.special.change = {
        setup: function() {
            return Lb.test(this.nodeName) ? (("checkbox" === this.type || "radio" === this.type) && (kb.event.add(this, "propertychange._change", function(a) {
                "checked" === a.originalEvent.propertyName && (this._just_changed=!0)
            }), kb.event.add(this, "click._change", function(a) {
                this._just_changed&&!a.isTrigger && (this._just_changed=!1), kb.event.simulate("change", this, a, !0)
            })), !1) : (kb.event.add(this, "beforeactivate._change", function(a) {
                var b = a.target;
                Lb.test(b.nodeName)&&!kb._data(b, "changeBubbles") && (kb.event.add(b, "change._change", function(a) {
                    !this.parentNode || a.isSimulated || a.isTrigger || kb.event.simulate("change", this.parentNode, a, !0)
                }), kb._data(b, "changeBubbles", !0))
            }), b)
        },
        handle: function(a) {
            var c = a.target;
            return this !== c || a.isSimulated || a.isTrigger || "radio" !== c.type && "checkbox" !== c.type ? a.handleObj.handler.apply(this, arguments) : b
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
        triggerHandler: function(a, c) {
            var d = this[0];
            return d ? kb.event.trigger(a, c, d, !0) : b
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
                    if (11 > c.nodeType && (g ? g.index(c)>-1 : 1 === c.nodeType && kb.find.matchesSelector(c, a))) {
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
    var Ub = "abbr|article|aside|audio|bdi|canvas|data|datalist|details|figcaption|figure|footer|header|hgroup|mark|meter|nav|output|progress|section|summary|time|video", Vb = / jQuery\d+="(?:null|\d+)"/g, Wb = RegExp("<(?:" + Ub + ")[\\s/>]", "i"), Xb = /^\s+/, Yb = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\w:]+)[^>]*)\/>/gi, Zb = /<([\w:]+)/, $b = /<tbody/i, _b = /<|&#?\w+;/, ac = /<(?:script|style|link)/i, bc = /^(?:checkbox|radio)$/i, cc = /checked\s*(?:[^=]|=\s*.checked.)/i, dc = /^$|\/(?:java|ecma)script/i, ec = /^true\/(.*)/, fc = /^\s*<!(?:\[CDATA\[|--)|(?:\]\]|--)>\s*$/g, gc = {
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
                b || 1 !== c.nodeType || kb.cleanData(u(c)), c.parentNode && (b && kb.contains(c.ownerDocument, c) && r(u(c, "script")), c.parentNode.removeChild(c));
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
    var jc, kc, lc, mc = /alpha\([^)]*\)/i, nc = /opacity\s*=\s*([^)]*)/, oc = /^(top|right|bottom|left)$/, pc = /^(none|table(?!-c[ea]).+)/, qc = /^margin/, rc = RegExp("^(" + lb + ")(.*)$", "i"), sc = RegExp("^(" + lb + ")(?!px)[a-z%]+$", "i"), tc = RegExp("^([+-])=(" + lb + ")", "i"), uc = {
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
                if (g = typeof d, "string" === g && (f = tc.exec(d)) && (d = (f[1] + 1) * f[2] + parseFloat(kb.css(a, c)), g = "number"), !(null == d || "number" === g && isNaN(d) || ("number" !== g || kb.cssNumber[i] || (d += "px"), kb.support.clearCloneStyle || "" !== d || 0 !== c.indexOf("background") || (j[c] = "inherit"), h && "set"in h && (d = h.set(a, d, e)) === b)))
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
        return h && ("" !== i || kb.contains(a.ownerDocument, a) || (i = kb.style(a, c)), sc.test(i) && qc.test(c) && (e = j.width, f = j.minWidth, g = j.maxWidth, j.minWidth = j.maxWidth = j.width = i, i = h.width, j.width = e, j.minWidth = f, j.maxWidth = g)), i
    }) : Y.documentElement.currentStyle && (kc = function(a) {
        return a.currentStyle
    }, lc = function(a, c, d) {
        var e, f, g, h = d || kc(a), i = h ? h[c]: b, j = a.style;
        return null == i && j && j[c] && (i = j[c]), sc.test(i)&&!oc.test(c) && (e = j.left, f = a.runtimeStyle, g = f && f.left, g && (f.left = a.currentStyle.left), j.left = "fontSize" === c ? "1em" : i, i = j.pixelLeft + "px", j.left = e, g && (f.left = g)), "" === i ? "auto" : i
    }), kb.each(["height", "width"], function(a, c) {
        kb.cssHooks[c] = {
            get: function(a, d, e) {
                return d ? 0 === a.offsetWidth && pc.test(kb.css(a, "display")) ? kb.swap(a, vc, function() {
                    return B(a, c, e)
                }) : B(a, c, e) : b
            },
            set: function(a, b, d) {
                var e = d && kc(a);
                return z(a, b, d ? A(a, c, d, kb.support.boxSizing && "border-box" === kb.css(a, "boxSizing", !1, e), e) : 0)
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
            get: function(a, c) {
                return c ? kb.swap(a, {
                    display: "inline-block"
                }, lc, [a, "marginRight"]) : b
            }
        }), !kb.support.pixelPosition && kb.fn.position && kb.each(["top", "left"], function(a, c) {
            kb.cssHooks[c] = {
                get: function(a, d) {
                    return d ? (d = lc(a, c), sc.test(d) ? kb(a).position()[c] + "px" : d) : b
                }
            }
        })
    }), kb.expr && kb.expr.filters && (kb.expr.filters.hidden = function(a) {
        return 0 >= a.offsetWidth && 0 >= a.offsetHeight ||!kb.support.reliableHiddenOffsets && "none" === (a.style && a.style.display || kb.css(a, "display"))
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
            if (m.beforeSend && (m.beforeSend.call(n, w, m)===!1 || 2 === u))
                return w.abort();
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
        }), "script") : b
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
                    c.mimeType && i.overrideMimeType && i.overrideMimeType(c.mimeType), c.crossDomain || e["X-Requested-With"] || (e["X-Requested-With"] = "XMLHttpRequest");
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
    var $c, _c, ad = /^(?:toggle|show|hide)$/, bd = RegExp("^(?:([+-])=|)(" + lb + ")([a-z%]*)$", "i"), cd = /queueHooks$/, dd = [Q], ed = {
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
                    f[c].elem !== this || null != a && f[c].queue !== a || (f[c].anim.stop(d), b=!1, f.splice(c, 1));
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
        for ($c = kb.now(); c.length > d; d++)
            a = c[d], a() || c[d] !== a || c.splice(d--, 1);
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
        return g ? (c = g.documentElement, kb.contains(c, f) ? (typeof f.getBoundingClientRect !== W && (e = f.getBoundingClientRect()), d = T(g), {
            top: e.top + (d.pageYOffset || c.scrollTop) - (c.clientTop || 0),
            left: e.left + (d.pageXOffset || c.scrollLeft) - (c.clientLeft || 0)
        }) : e) : void 0
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
                return f === b ? g ? c in g ? g[c] : g.document.documentElement[e] : a[e] : (g ? g.scrollTo(d ? kb(g).scrollLeft() : f, d ? f : kb(g).scrollTop()) : a[e] = f, b)
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
}(window), function(a) {
    function b() {}
    function c(a) {
        g = [a]
    }
    function d(a, b, c) {
        return a && a.apply(b.context || b, c)
    }
    function e(a) {
        return /\?/.test(a) ? "&" : "?"
    }
    function f(f) {
        function n(a) {
            X++||(Y(), S && (A[U] = {
                s: [a]
            }), O && (a = O.apply(f, [a])), d(L, f, [a, v, f]), d(N, f, [f, v]))
        }
        function F(a) {
            X++||(Y(), S && a != w && (A[U] = a), d(M, f, [f, a]), d(N, f, [f, a]))
        }
        f = a.extend({}, C, f);
        var G, H, I, J, K, L = f.success, M = f.error, N = f.complete, O = f.dataFilter, P = f.callbackParameter, Q = f.callback, R = f.cache, S = f.pageCache, T = f.charset, U = f.url, V = f.data, W = f.timeout, X = 0, Y = b;
        return y && y(function(a) {
            a.done(L).fail(M), L = a.resolve, M = a.reject
        }).promise(f), f.abort = function() {
            !X++&&Y()
        }, d(f.beforeSend, f, [f])===!1 || X ? f : (U = U || j, V = V ? "string" == typeof V ? V : a.param(V, f.traditional) : j, U += V ? e(U) + V : j, P && (U += e(U) + encodeURIComponent(P) + "=?"), !R&&!S && (U += e(U) + "_" + (new Date).getTime() + "="), U = U.replace(/=\?(&|$)/, "=" + Q + "$1"), S && (G = A[U]) ? G.s ? n(G.s[0]) : F(G) : (x[Q] = c, I = a(u)[0], I.id = m + B++, T && (I[i] = T), D && D.version() < 11.6 ? (J = a(u)[0]).text = "document.getElementById('" + I.id + "')." + p + "()" : I[h] = h, E && (I.htmlFor = I.id, I.event = o), I[q] = I[p] = I[r] = function(a) {
            if (!I[s] ||!/i/.test(I[s])) {
                try {
                    I[o] && I[o]()
                } catch (b) {}
                a = g, g = 0, a ? n(a[0]) : F(k)
            }
        }, I.src = U, Y = function() {
            K && clearTimeout(K), I[r] = I[q] = I[p] = null, z[t](I), J && z[t](J)
        }, z[l](I, H = z.firstChild), J && z[l](J, H), K = W > 0 && setTimeout(function() {
            F(w)
        }, W)), f)
    }
    var g, h = "async", i = "charset", j = "", k = "error", l = "insertBefore", m = "_jqjsp", n = "on", o = n + "click", p = n + k, q = n + "load", r = n + "readystatechange", s = "readyState", t = "removeChild", u = "<script>", v = "success", w = "timeout", x = window, y = a.Deferred, z = a("head")[0] || document.documentElement, A = {}, B = 0, C = {
        callback: m,
        url: location.href
    }, D = x.opera, E=!!a("<div>").html("<!--[if IE]><i><![endif]-->").find("i").length;
    f.setup = function(b) {
        a.extend(C, b)
    }, a.jsonp = f
}(jQuery), void 0 === jQuery.migrateMute && (jQuery.migrateMute=!0), function(a, b, c) {
    function d(c) {
        var d = b.console;
        f[c] || (f[c]=!0, a.migrateWarnings.push(c), d && d.warn&&!a.migrateMute && (d.warn("JQMIGRATE: " + c), a.migrateTrace && d.trace && d.trace()))
    }
    function e(b, e, f, g) {
        if (Object.defineProperty)
            try {
                return Object.defineProperty(b, e, {
                    configurable: !0,
                    enumerable: !0,
                    get: function() {
                        return d(g), f
                    },
                    set: function(a) {
                        d(g), f = a
                    }
                }), c
        } catch (h) {}
        a._definePropertyBroken=!0, b[e] = f
    }
    var f = {};
    a.migrateWarnings = [], !a.migrateMute && b.console && b.console.log && b.console.log("JQMIGRATE: Logging is active"), a.migrateTrace === c && (a.migrateTrace=!0), a.migrateReset = function() {
        f = {}, a.migrateWarnings.length = 0
    }, "BackCompat" === document.compatMode && d("jQuery is not compatible with Quirks Mode");
    var g = a("<input/>", {
        size: 1
    }).attr("size") && a.attrFn, h = a.attr, i = a.attrHooks.value && a.attrHooks.value.get || function() {
        return null
    }, j = a.attrHooks.value && a.attrHooks.value.set || function() {
        return c
    }, k = /^(?:input|button)$/i, l = /^[238]$/, m = /^(?:autofocus|autoplay|async|checked|controls|defer|disabled|hidden|loop|multiple|open|readonly|required|scoped|selected)$/i, n = /^(?:checked|selected)$/i;
    e(a, "attrFn", g || {}, "jQuery.attrFn is deprecated"), a.attr = function(b, e, f, i) {
        var j = e.toLowerCase(), o = b && b.nodeType;
        return i && (4 > h.length && d("jQuery.fn.attr( props, pass ) is deprecated"), b&&!l.test(o) && (g ? e in g : a.isFunction(a.fn[e]))) ? a(b)[e](f) : ("type" === e && f !== c && k.test(b.nodeName) && b.parentNode && d("Can't change the 'type' of an input or button in IE 6/7/8"), !a.attrHooks[j] && m.test(j) && (a.attrHooks[j] = {
            get: function(b, d) {
                var e, f = a.prop(b, d);
                return f===!0 || "boolean" != typeof f && (e = b.getAttributeNode(d)) && e.nodeValue!==!1 ? d.toLowerCase() : c
            },
            set: function(b, c, d) {
                var e;
                return c===!1 ? a.removeAttr(b, d) : (e = a.propFix[d] || d, e in b && (b[e]=!0), b.setAttribute(d, d.toLowerCase())), d
            }
        }, n.test(j) && d("jQuery.fn.attr('" + j + "') may use property instead of attribute")), h.call(a, b, e, f))
    }, a.attrHooks.value = {
        get: function(a, b) {
            var c = (a.nodeName || "").toLowerCase();
            return "button" === c ? i.apply(this, arguments) : ("input" !== c && "option" !== c && d("jQuery.fn.attr('value') no longer gets properties"), b in a ? a.value : null)
        },
        set: function(a, b) {
            var e = (a.nodeName || "").toLowerCase();
            return "button" === e ? j.apply(this, arguments) : ("input" !== e && "option" !== e && d("jQuery.fn.attr('value', val) no longer sets properties"), a.value = b, c)
        }
    };
    var o, p, q = a.fn.init, r = a.parseJSON, s = /^([^<]*)(<[\w\W]+>)([^>]*)$/;
    a.fn.init = function(b, c, e) {
        var f;
        return b && "string" == typeof b&&!a.isPlainObject(c) && (f = s.exec(a.trim(b))) && f[0] && ("<" !== b.charAt(0) && d("$(html) HTML strings must start with '<' character"), f[3] && d("$(html) HTML text after last tag is ignored"), "#" === f[0].charAt(0) && (d("HTML string cannot start with a '#' character"), a.error("JQMIGRATE: Invalid selector string (XSS)")), c && c.context && (c = c.context), a.parseHTML) ? q.call(this, a.parseHTML(f[2], c, !0), c, e) : q.apply(this, arguments)
    }, a.fn.init.prototype = a.fn, a.parseJSON = function(a) {
        return a || null === a ? r.apply(this, arguments) : (d("jQuery.parseJSON requires a valid JSON string"), null)
    }, a.uaMatch = function(a) {
        a = a.toLowerCase();
        var b = /(chrome)[ \/]([\w.]+)/.exec(a) || /(webkit)[ \/]([\w.]+)/.exec(a) || /(opera)(?:.*version|)[ \/]([\w.]+)/.exec(a) || /(msie) ([\w.]+)/.exec(a) || 0 > a.indexOf("compatible") && /(mozilla)(?:.*? rv:([\w.]+)|)/.exec(a) || [];
        return {
            browser: b[1] || "",
            version: b[2] || "0"
        }
    }, a.browser || (o = a.uaMatch(navigator.userAgent), p = {}, o.browser && (p[o.browser]=!0, p.version = o.version), p.chrome ? p.webkit=!0 : p.webkit && (p.safari=!0), a.browser = p), e(a, "browser", a.browser, "jQuery.browser is deprecated"), a.sub = function() {
        function b(a, c) {
            return new b.fn.init(a, c)
        }
        a.extend(!0, b, this), b.superclass = this, b.fn = b.prototype = this (), b.fn.constructor = b, b.sub = this.sub, b.fn.init = function(d, e) {
            return e && e instanceof a&&!(e instanceof b) && (e = b(e)), a.fn.init.call(this, d, e, c)
        }, b.fn.init.prototype = b.fn;
        var c = b(document);
        return d("jQuery.sub() is deprecated"), b
    }, a.ajaxSetup({
        converters: {
            "text json": a.parseJSON
        }
    });
    var t = a.fn.data;
    a.fn.data = function(b) {
        var e, f, g = this[0];
        return !g || "events" !== b || 1 !== arguments.length || (e = a.data(g, b), f = a._data(g, b), e !== c && e !== f || f === c) ? t.apply(this, arguments) : (d("Use of jQuery.fn.data('events') is deprecated"), f)
    };
    var u = /\/(java|ecma)script/i, v = a.fn.andSelf || a.fn.addBack;
    a.fn.andSelf = function() {
        return d("jQuery.fn.andSelf() replaced by jQuery.fn.addBack()"), v.apply(this, arguments)
    }, a.clean || (a.clean = function(b, e, f, g) {
        e = e || document, e=!e.nodeType && e[0] || e, e = e.ownerDocument || e, d("jQuery.clean() is deprecated");
        var h, i, j, k, l = [];
        if (a.merge(l, a.buildFragment(b, e).childNodes), f)
            for (j = function(a) {
                return !a.type || u.test(a.type) ? g ? g.push(a.parentNode ? a.parentNode.removeChild(a) : a) : f.appendChild(a) : c
            }, h = 0; null != (i = l[h]); h++)
                a.nodeName(i, "script") && j(i) || (f.appendChild(i), i.getElementsByTagName !== c && (k = a.grep(a.merge([], i.getElementsByTagName("script")), j), l.splice.apply(l, [h + 1, 0].concat(k)), h += k.length));
        return l
    });
    var w = a.event.add, x = a.event.remove, y = a.event.trigger, z = a.fn.toggle, A = a.fn.live, B = a.fn.die, C = "ajaxStart|ajaxStop|ajaxSend|ajaxComplete|ajaxError|ajaxSuccess", D = RegExp("\\b(?:" + C + ")\\b"), E = /(?:^|\s)hover(\.\S+|)\b/, F = function(b) {
        return "string" != typeof b || a.event.special.hover ? b : (E.test(b) && d("'hover' pseudo-event is deprecated, use 'mouseenter mouseleave'"), b && b.replace(E, "mouseenter$1 mouseleave$1"))
    };
    a.event.props && "attrChange" !== a.event.props[0] && a.event.props.unshift("attrChange", "attrName", "relatedNode", "srcElement"), a.event.dispatch && e(a.event, "handle", a.event.dispatch, "jQuery.event.handle is undocumented and deprecated"), a.event.add = function(a, b, c, e, f) {
        a !== document && D.test(b) && d("AJAX events should be attached to document: " + b), w.call(this, a, F(b || ""), c, e, f)
    }, a.event.remove = function(a, b, c, d, e) {
        x.call(this, a, F(b) || "", c, d, e)
    }, a.fn.error = function() {
        var a = Array.prototype.slice.call(arguments, 0);
        return d("jQuery.fn.error() is deprecated"), a.splice(0, 0, "error"), arguments.length ? this.bind.apply(this, a) : (this.triggerHandler.apply(this, a), this)
    }, a.fn.toggle = function(b, c) {
        if (!a.isFunction(b) ||!a.isFunction(c))
            return z.apply(this, arguments);
        d("jQuery.fn.toggle(handler, handler...) is deprecated");
        var e = arguments, f = b.guid || a.guid++, g = 0, h = function(c) {
            var d = (a._data(this, "lastToggle" + b.guid) || 0)%g;
            return a._data(this, "lastToggle" + b.guid, d + 1), c.preventDefault(), e[d].apply(this, arguments) ||!1
        };
        for (h.guid = f; e.length > g;)
            e[g++].guid = f;
        return this.click(h)
    }, a.fn.live = function(b, c, e) {
        return d("jQuery.fn.live() is deprecated"), A ? A.apply(this, arguments) : (a(this.context).on(b, this.selector, c, e), this)
    }, a.fn.die = function(b, c) {
        return d("jQuery.fn.die() is deprecated"), B ? B.apply(this, arguments) : (a(this.context).off(b, this.selector || "**", c), this)
    }, a.event.trigger = function(a, b, c, e) {
        return c || D.test(a) || d("Global events are undocumented and deprecated"), y.call(this, a, b, c || document, e)
    }, a.each(C.split("|"), function(b, c) {
        a.event.special[c] = {
            setup: function() {
                var b = this;
                return b !== document && (a.event.add(document, c + "." + a.guid, function() {
                    a.event.trigger(c, null, b, !0)
                }), a._data(this, c, a.guid++)), !1
            },
            teardown: function() {
                return this !== document && a.event.remove(document, c + "." + a._data(this, c)), !1
            }
        }
    })
}(jQuery, window), function(a, b) {
    function c(b, c) {
        var e, f, g, h = b.nodeName.toLowerCase();
        return "area" === h ? (e = b.parentNode, f = e.name, b.href && f && "map" === e.nodeName.toLowerCase() ? (g = a("img[usemap=#" + f + "]")[0], !!g && d(g)) : !1) : (/input|select|textarea|button|object/.test(h)?!b.disabled : "a" === h ? b.href || c : c) && d(b)
    }
    function d(b) {
        return a.expr.filters.visible(b)&&!a(b).parents().addBack().filter(function() {
            return "hidden" === a.css(this, "visibility")
        }).length
    }
    var e = 0, f = /^ui-id-\d+$/;
    a.ui = a.ui || {}, a.extend(a.ui, {
        version: "1.10.3",
        keyCode: {
            BACKSPACE: 8,
            COMMA: 188,
            DELETE: 46,
            DOWN: 40,
            END: 35,
            ENTER: 13,
            ESCAPE: 27,
            HOME: 36,
            LEFT: 37,
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
            SPACE: 32,
            TAB: 9,
            UP: 38
        }
    }), a.fn.extend({
        focus: function(b) {
            return function(c, d) {
                return "number" == typeof c ? this.each(function() {
                    var b = this;
                    setTimeout(function() {
                        a(b).focus(), d && d.call(b)
                    }, c)
                }) : b.apply(this, arguments)
            }
        }(a.fn.focus),
        scrollParent: function() {
            var b;
            return b = a.ui.ie && /(static|relative)/.test(this.css("position")) || /absolute/.test(this.css("position")) ? this.parents().filter(function() {
                return /(relative|absolute|fixed)/.test(a.css(this, "position")) && /(auto|scroll)/.test(a.css(this, "overflow") + a.css(this, "overflow-y") + a.css(this, "overflow-x"))
            }).eq(0) : this.parents().filter(function() {
                return /(auto|scroll)/.test(a.css(this, "overflow") + a.css(this, "overflow-y") + a.css(this, "overflow-x"))
            }).eq(0), /fixed/.test(this.css("position")) ||!b.length ? a(document) : b
        },
        zIndex: function(c) {
            if (c !== b)
                return this.css("zIndex", c);
            if (this.length)
                for (var d, e, f = a(this[0]); f.length && f[0] !== document;) {
                    if (d = f.css("position"), ("absolute" === d || "relative" === d || "fixed" === d) && (e = parseInt(f.css("zIndex"), 10), !isNaN(e) && 0 !== e))
                        return e;
                        f = f.parent()
                }
            return 0
        },
        uniqueId: function() {
            return this.each(function() {
                this.id || (this.id = "ui-id-" + ++e)
            })
        },
        removeUniqueId: function() {
            return this.each(function() {
                f.test(this.id) && a(this).removeAttr("id")
            })
        }
    }), a.extend(a.expr[":"], {
        data: a.expr.createPseudo ? a.expr.createPseudo(function(b) {
            return function(c) {
                return !!a.data(c, b)
            }
        }): function(b, c, d) {
            return !!a.data(b, d[3])
        },
        focusable: function(b) {
            return c(b, !isNaN(a.attr(b, "tabindex")))
        },
        tabbable: function(b) {
            var d = a.attr(b, "tabindex"), e = isNaN(d);
            return (e || d >= 0) && c(b, !e)
        }
    }), a("<a>").outerWidth(1).jquery || a.each(["Width", "Height"], function(c, d) {
        function e(b, c, d, e) {
            return a.each(f, function() {
                c -= parseFloat(a.css(b, "padding" + this)) || 0, d && (c -= parseFloat(a.css(b, "border" + this + "Width")) || 0), e && (c -= parseFloat(a.css(b, "margin" + this)) || 0)
            }), c
        }
        var f = "Width" === d ? ["Left", "Right"]: ["Top", "Bottom"], g = d.toLowerCase(), h = {
            innerWidth: a.fn.innerWidth,
            innerHeight: a.fn.innerHeight,
            outerWidth: a.fn.outerWidth,
            outerHeight: a.fn.outerHeight
        };
        a.fn["inner" + d] = function(c) {
            return c === b ? h["inner" + d].call(this) : this.each(function() {
                a(this).css(g, e(this, c) + "px")
            })
        }, a.fn["outer" + d] = function(b, c) {
            return "number" != typeof b ? h["outer" + d].call(this, b) : this.each(function() {
                a(this).css(g, e(this, b, !0, c) + "px")
            })
        }
    }), a.fn.addBack || (a.fn.addBack = function(a) {
        return this.add(null == a ? this.prevObject : this.prevObject.filter(a))
    }), a("<a>").data("a-b", "a").removeData("a-b").data("a-b") && (a.fn.removeData = function(b) {
        return function(c) {
            return arguments.length ? b.call(this, a.camelCase(c)) : b.call(this)
        }
    }(a.fn.removeData)), a.ui.ie=!!/msie [\w.]+/.exec(navigator.userAgent.toLowerCase()), a.support.selectstart = "onselectstart"in document.createElement("div"), a.fn.extend({
        disableSelection: function() {
            return this.bind((a.support.selectstart ? "selectstart" : "mousedown") + ".ui-disableSelection", function(a) {
                a.preventDefault()
            })
        },
        enableSelection: function() {
            return this.unbind(".ui-disableSelection")
        }
    }), a.extend(a.ui, {
        plugin: {
            add: function(b, c, d) {
                var e, f = a.ui[b].prototype;
                for (e in d)
                    f.plugins[e] = f.plugins[e] || [], f.plugins[e].push([c, d[e]])
            },
            call: function(a, b, c) {
                var d, e = a.plugins[b];
                if (e && a.element[0].parentNode && 11 !== a.element[0].parentNode.nodeType)
                    for (d = 0; e.length > d; d++)
                        a.options[e[d][0]] && e[d][1].apply(a.element, c)
            }
        },
        hasScroll: function(b, c) {
            if ("hidden" === a(b).css("overflow"))
                return !1;
            var d = c && "left" === c ? "scrollLeft": "scrollTop", e=!1;
            return b[d] > 0?!0 : (b[d] = 1, e = b[d] > 0, b[d] = 0, e)
        }
    })
}(jQuery), function(a, b) {
    var c = 0, d = Array.prototype.slice, e = a.cleanData;
    a.cleanData = function(b) {
        for (var c, d = 0; null != (c = b[d]); d++)
            try {
                a(c).triggerHandler("remove")
        } catch (f) {}
        e(b)
    }, a.widget = function(c, d, e) {
        var f, g, h, i, j = {}, k = c.split(".")[0];
        c = c.split(".")[1], f = k + "-" + c, e || (e = d, d = a.Widget), a.expr[":"][f.toLowerCase()] = function(b) {
            return !!a.data(b, f)
        }, a[k] = a[k] || {}, g = a[k][c], h = a[k][c] = function(a, c) {
            return this._createWidget ? (arguments.length && this._createWidget(a, c), b) : new h(a, c)
        }, a.extend(h, g, {
            version: e.version,
            _proto: a.extend({}, e),
            _childConstructors: []
        }), i = new d, i.options = a.widget.extend({}, i.options), a.each(e, function(c, e) {
            return a.isFunction(e) ? (j[c] = function() {
                var a = function() {
                    return d.prototype[c].apply(this, arguments)
                }, b = function(a) {
                    return d.prototype[c].apply(this, a)
                };
                return function() {
                    var c, d = this._super, f = this._superApply;
                    return this._super = a, this._superApply = b, c = e.apply(this, arguments), this._super = d, this._superApply = f, c
                }
            }(), b) : (j[c] = e, b)
        }), h.prototype = a.widget.extend(i, {
            widgetEventPrefix: g ? i.widgetEventPrefix: c
        }, j, {
            constructor: h,
            namespace: k,
            widgetName: c,
            widgetFullName: f
        }), g ? (a.each(g._childConstructors, function(b, c) {
            var d = c.prototype;
            a.widget(d.namespace + "." + d.widgetName, h, c._proto)
        }), delete g._childConstructors) : d._childConstructors.push(h), a.widget.bridge(c, h)
    }, a.widget.extend = function(c) {
        for (var e, f, g = d.call(arguments, 1), h = 0, i = g.length; i > h; h++)
            for (e in g[h])
                f = g[h][e], g[h].hasOwnProperty(e) && f !== b && (c[e] = a.isPlainObject(f) ? a.isPlainObject(c[e]) ? a.widget.extend({}, c[e], f) : a.widget.extend({}, f) : f);
        return c
    }, a.widget.bridge = function(c, e) {
        var f = e.prototype.widgetFullName || c;
        a.fn[c] = function(g) {
            var h = "string" == typeof g, i = d.call(arguments, 1), j = this;
            return g=!h && i.length ? a.widget.extend.apply(null, [g].concat(i)) : g, this.each(h ? function() {
                var d, e = a.data(this, f);
                return e ? a.isFunction(e[g]) && "_" !== g.charAt(0) ? (d = e[g].apply(e, i), d !== e && d !== b ? (j = d && d.jquery ? j.pushStack(d.get()) : d, !1) : b) : a.error("no such method '" + g + "' for " + c + " widget instance") : a.error("cannot call methods on " + c + " prior to initialization; attempted to call method '" + g + "'")
            } : function() {
                var b = a.data(this, f);
                b ? b.option(g || {})._init() : a.data(this, f, new e(g, this))
            }), j
        }
    }, a.Widget = function() {}, a.Widget._childConstructors = [], a.Widget.prototype = {
        widgetName: "widget",
        widgetEventPrefix: "",
        defaultElement: "<div>",
        options: {
            disabled: !1,
            create: null
        },
        _createWidget: function(b, d) {
            d = a(d || this.defaultElement || this)[0], this.element = a(d), this.uuid = c++, this.eventNamespace = "." + this.widgetName + this.uuid, this.options = a.widget.extend({}, this.options, this._getCreateOptions(), b), this.bindings = a(), this.hoverable = a(), this.focusable = a(), d !== this && (a.data(d, this.widgetFullName, this), this._on(!0, this.element, {
                remove: function(a) {
                    a.target === d && this.destroy()
                }
            }), this.document = a(d.style ? d.ownerDocument : d.document || d), this.window = a(this.document[0].defaultView || this.document[0].parentWindow)), this._create(), this._trigger("create", null, this._getCreateEventData()), this._init()
        },
        _getCreateOptions: a.noop,
        _getCreateEventData: a.noop,
        _create: a.noop,
        _init: a.noop,
        destroy: function() {
            this._destroy(), this.element.unbind(this.eventNamespace).removeData(this.widgetName).removeData(this.widgetFullName).removeData(a.camelCase(this.widgetFullName)), this.widget().unbind(this.eventNamespace).removeAttr("aria-disabled").removeClass(this.widgetFullName + "-disabled ui-state-disabled"), this.bindings.unbind(this.eventNamespace), this.hoverable.removeClass("ui-state-hover"), this.focusable.removeClass("ui-state-focus")
        },
        _destroy: a.noop,
        widget: function() {
            return this.element
        },
        option: function(c, d) {
            var e, f, g, h = c;
            if (0 === arguments.length)
                return a.widget.extend({}, this.options);
            if ("string" == typeof c)
                if (h = {}, e = c.split("."), c = e.shift(), e.length) {
                    for (f = h[c] = a.widget.extend({}, this.options[c]), g = 0; e.length - 1 > g; g++)
                        f[e[g]] = f[e[g]] || {}, f = f[e[g]];
                        if (c = e.pop(), d === b)
                            return f[c] === b ? null : f[c];
                            f[c] = d
                } else {
                    if (d === b)
                        return this.options[c] === b ? null : this.options[c];
                        h[c] = d
                }
            return this._setOptions(h), this
        },
        _setOptions: function(a) {
            var b;
            for (b in a)
                this._setOption(b, a[b]);
            return this
        },
        _setOption: function(a, b) {
            return this.options[a] = b, "disabled" === a && (this.widget().toggleClass(this.widgetFullName + "-disabled ui-state-disabled", !!b).attr("aria-disabled", b), this.hoverable.removeClass("ui-state-hover"), this.focusable.removeClass("ui-state-focus")), this
        },
        enable: function() {
            return this._setOption("disabled", !1)
        },
        disable: function() {
            return this._setOption("disabled", !0)
        },
        _on: function(c, d, e) {
            var f, g = this;
            "boolean" != typeof c && (e = d, d = c, c=!1), e ? (d = f = a(d), this.bindings = this.bindings.add(d)) : (e = d, d = this.element, f = this.widget()), a.each(e, function(e, h) {
                function i() {
                    return c || g.options.disabled!==!0&&!a(this).hasClass("ui-state-disabled") ? ("string" == typeof h ? g[h] : h).apply(g, arguments) : b
                }
                "string" != typeof h && (i.guid = h.guid = h.guid || i.guid || a.guid++);
                var j = e.match(/^(\w+)\s*(.*)$/), k = j[1] + g.eventNamespace, l = j[2];
                l ? f.delegate(l, k, i) : d.bind(k, i)
            })
        },
        _off: function(a, b) {
            b = (b || "").split(" ").join(this.eventNamespace + " ") + this.eventNamespace, a.unbind(b).undelegate(b)
        },
        _delay: function(a, b) {
            function c() {
                return ("string" == typeof a ? d[a] : a).apply(d, arguments)
            }
            var d = this;
            return setTimeout(c, b || 0)
        },
        _hoverable: function(b) {
            this.hoverable = this.hoverable.add(b), this._on(b, {
                mouseenter: function(b) {
                    a(b.currentTarget).addClass("ui-state-hover")
                },
                mouseleave: function(b) {
                    a(b.currentTarget).removeClass("ui-state-hover")
                }
            })
        },
        _focusable: function(b) {
            this.focusable = this.focusable.add(b), this._on(b, {
                focusin: function(b) {
                    a(b.currentTarget).addClass("ui-state-focus")
                },
                focusout: function(b) {
                    a(b.currentTarget).removeClass("ui-state-focus")
                }
            })
        },
        _trigger: function(b, c, d) {
            var e, f, g = this.options[b];
            if (d = d || {}, c = a.Event(c), c.type = (b === this.widgetEventPrefix ? b : this.widgetEventPrefix + b).toLowerCase(), c.target = this.element[0], f = c.originalEvent)
                for (e in f)
                    e in c || (c[e] = f[e]);
            return this.element.trigger(c, d), !(a.isFunction(g) && g.apply(this.element[0], [c].concat(d))===!1 || c.isDefaultPrevented())
        }
    }, a.each({
        show: "fadeIn",
        hide: "fadeOut"
    }, function(b, c) {
        a.Widget.prototype["_" + b] = function(d, e, f) {
            "string" == typeof e && (e = {
                effect: e
            });
            var g, h = e ? e===!0 || "number" == typeof e ? c: e.effect || c: b;
            e = e || {}, "number" == typeof e && (e = {
                duration: e
            }), g=!a.isEmptyObject(e), e.complete = f, e.delay && d.delay(e.delay), g && a.effects && a.effects.effect[h] ? d[b](e) : h !== b && d[h] ? d[h](e.duration, e.easing, f) : d.queue(function(c) {
                a(this)[b](), f && f.call(d[0]), c()
            })
        }
    })
}(jQuery), function(a) {
    var b=!1;
    a(document).mouseup(function() {
        b=!1
    }), a.widget("ui.mouse", {
        version: "1.10.3",
        options: {
            cancel: "input,textarea,button,select,option",
            distance: 1,
            delay: 0
        },
        _mouseInit: function() {
            var b = this;
            this.element.bind("mousedown." + this.widgetName, function(a) {
                return b._mouseDown(a)
            }).bind("click." + this.widgetName, function(c) {
                return !0 === a.data(c.target, b.widgetName + ".preventClickEvent") ? (a.removeData(c.target, b.widgetName + ".preventClickEvent"), c.stopImmediatePropagation(), !1) : void 0
            }), this.started=!1
        },
        _mouseDestroy: function() {
            this.element.unbind("." + this.widgetName), this._mouseMoveDelegate && a(document).unbind("mousemove." + this.widgetName, this._mouseMoveDelegate).unbind("mouseup." + this.widgetName, this._mouseUpDelegate)
        },
        _mouseDown: function(c) {
            if (!b) {
                this._mouseStarted && this._mouseUp(c), this._mouseDownEvent = c;
                var d = this, e = 1 === c.which, f = "string" == typeof this.options.cancel && c.target.nodeName ? a(c.target).closest(this.options.cancel).length: !1;
                return e&&!f && this._mouseCapture(c) ? (this.mouseDelayMet=!this.options.delay, this.mouseDelayMet || (this._mouseDelayTimer = setTimeout(function() {
                    d.mouseDelayMet=!0
                }, this.options.delay)), this._mouseDistanceMet(c) && this._mouseDelayMet(c) && (this._mouseStarted = this._mouseStart(c)!==!1, !this._mouseStarted) ? (c.preventDefault(), !0) : (!0 === a.data(c.target, this.widgetName + ".preventClickEvent") && a.removeData(c.target, this.widgetName + ".preventClickEvent"), this._mouseMoveDelegate = function(a) {
                    return d._mouseMove(a)
                }, this._mouseUpDelegate = function(a) {
                    return d._mouseUp(a)
                }, a(document).bind("mousemove." + this.widgetName, this._mouseMoveDelegate).bind("mouseup." + this.widgetName, this._mouseUpDelegate), c.preventDefault(), b=!0, !0)) : !0
            }
        },
        _mouseMove: function(b) {
            return a.ui.ie && (!document.documentMode || 9 > document.documentMode)&&!b.button ? this._mouseUp(b) : this._mouseStarted ? (this._mouseDrag(b), b.preventDefault()) : (this._mouseDistanceMet(b) && this._mouseDelayMet(b) && (this._mouseStarted = this._mouseStart(this._mouseDownEvent, b)!==!1, this._mouseStarted ? this._mouseDrag(b) : this._mouseUp(b)), !this._mouseStarted)
        },
        _mouseUp: function(b) {
            return a(document).unbind("mousemove." + this.widgetName, this._mouseMoveDelegate).unbind("mouseup." + this.widgetName, this._mouseUpDelegate), this._mouseStarted && (this._mouseStarted=!1, b.target === this._mouseDownEvent.target && a.data(b.target, this.widgetName + ".preventClickEvent", !0), this._mouseStop(b)), !1
        },
        _mouseDistanceMet: function(a) {
            return Math.max(Math.abs(this._mouseDownEvent.pageX - a.pageX), Math.abs(this._mouseDownEvent.pageY - a.pageY)) >= this.options.distance
        },
        _mouseDelayMet: function() {
            return this.mouseDelayMet
        },
        _mouseStart: function() {},
        _mouseDrag: function() {},
        _mouseStop: function() {},
        _mouseCapture: function() {
            return !0
        }
    })
}(jQuery), function(a, b) {
    function c(a, b, c) {
        return [parseFloat(a[0]) * (n.test(a[0]) ? b / 100 : 1), parseFloat(a[1]) * (n.test(a[1]) ? c / 100 : 1)]
    }
    function d(b, c) {
        return parseInt(a.css(b, c), 10) || 0
    }
    function e(b) {
        var c = b[0];
        return 9 === c.nodeType ? {
            width: b.width(),
            height: b.height(),
            offset: {
                top: 0,
                left: 0
            }
        } : a.isWindow(c) ? {
            width: b.width(),
            height: b.height(),
            offset: {
                top: b.scrollTop(),
                left: b.scrollLeft()
            }
        } : c.preventDefault ? {
            width: 0,
            height: 0,
            offset: {
                top: c.pageY,
                left: c.pageX
            }
        } : {
            width: b.outerWidth(),
            height: b.outerHeight(),
            offset: b.offset()
        }
    }
    a.ui = a.ui || {};
    var f, g = Math.max, h = Math.abs, i = Math.round, j = /left|center|right/, k = /top|center|bottom/, l = /[\+\-]\d+(\.[\d]+)?%?/, m = /^\w+/, n = /%$/, o = a.fn.position;
    a.position = {
        scrollbarWidth: function() {
            if (f !== b)
                return f;
            var c, d, e = a("<div style='display:block;width:50px;height:50px;overflow:hidden;'><div style='height:100px;width:auto;'></div></div>"), g = e.children()[0];
            return a("body").append(e), c = g.offsetWidth, e.css("overflow", "scroll"), d = g.offsetWidth, c === d && (d = e[0].clientWidth), e.remove(), f = c - d
        },
        getScrollInfo: function(b) {
            var c = b.isWindow ? "": b.element.css("overflow-x"), d = b.isWindow ? "": b.element.css("overflow-y"), e = "scroll" === c || "auto" === c && b.width < b.element[0].scrollWidth, f = "scroll" === d || "auto" === d && b.height < b.element[0].scrollHeight;
            return {
                width: f ? a.position.scrollbarWidth(): 0,
                height: e ? a.position.scrollbarWidth(): 0
            }
        },
        getWithinInfo: function(b) {
            var c = a(b || window), d = a.isWindow(c[0]);
            return {
                element: c,
                isWindow: d,
                offset: c.offset() || {
                    left: 0,
                    top: 0
                },
                scrollLeft: c.scrollLeft(),
                scrollTop: c.scrollTop(),
                width: d ? c.width(): c.outerWidth(),
                height: d ? c.height(): c.outerHeight()
            }
        }
    }, a.fn.position = function(b) {
        if (!b ||!b.of)
            return o.apply(this, arguments);
        b = a.extend({}, b);
        var f, n, p, q, r, s, t = a(b.of), u = a.position.getWithinInfo(b.within), v = a.position.getScrollInfo(u), w = (b.collision || "flip").split(" "), x = {};
        return s = e(t), t[0].preventDefault && (b.at = "left top"), n = s.width, p = s.height, q = s.offset, r = a.extend({}, q), a.each(["my", "at"], function() {
            var a, c, d = (b[this] || "").split(" ");
            1 === d.length && (d = j.test(d[0]) ? d.concat(["center"]) : k.test(d[0]) ? ["center"].concat(d) : ["center", "center"]), d[0] = j.test(d[0]) ? d[0] : "center", d[1] = k.test(d[1]) ? d[1] : "center", a = l.exec(d[0]), c = l.exec(d[1]), x[this] = [a ? a[0]: 0, c ? c[0]: 0], b[this] = [m.exec(d[0])[0], m.exec(d[1])[0]]
        }), 1 === w.length && (w[1] = w[0]), "right" === b.at[0] ? r.left += n : "center" === b.at[0] && (r.left += n / 2), "bottom" === b.at[1] ? r.top += p : "center" === b.at[1] && (r.top += p / 2), f = c(x.at, n, p), r.left += f[0], r.top += f[1], this.each(function() {
            var e, j, k = a(this), l = k.outerWidth(), m = k.outerHeight(), o = d(this, "marginLeft"), s = d(this, "marginTop"), y = l + o + d(this, "marginRight") + v.width, z = m + s + d(this, "marginBottom") + v.height, A = a.extend({}, r), B = c(x.my, k.outerWidth(), k.outerHeight());
            "right" === b.my[0] ? A.left -= l : "center" === b.my[0] && (A.left -= l / 2), "bottom" === b.my[1] ? A.top -= m : "center" === b.my[1] && (A.top -= m / 2), A.left += B[0], A.top += B[1], a.support.offsetFractions || (A.left = i(A.left), A.top = i(A.top)), e = {
                marginLeft: o,
                marginTop: s
            }, a.each(["left", "top"], function(c, d) {
                a.ui.position[w[c]] && a.ui.position[w[c]][d](A, {
                    targetWidth: n,
                    targetHeight: p,
                    elemWidth: l,
                    elemHeight: m,
                    collisionPosition: e,
                    collisionWidth: y,
                    collisionHeight: z,
                    offset: [f[0] + B[0], f[1] + B[1]],
                    my: b.my,
                    at: b.at,
                    within: u,
                    elem: k
                })
            }), b.using && (j = function(a) {
                var c = q.left - A.left, d = c + n - l, e = q.top - A.top, f = e + p - m, i = {
                    target: {
                        element: t,
                        left: q.left,
                        top: q.top,
                        width: n,
                        height: p
                    },
                    element: {
                        element: k,
                        left: A.left,
                        top: A.top,
                        width: l,
                        height: m
                    },
                    horizontal: 0 > d ? "left": c > 0 ? "right": "center",
                    vertical: 0 > f ? "top": e > 0 ? "bottom": "middle"
                };
                l > n && n > h(c + d) && (i.horizontal = "center"), m > p && p > h(e + f) && (i.vertical = "middle"), i.important = g(h(c), h(d)) > g(h(e), h(f)) ? "horizontal" : "vertical", b.using.call(this, a, i)
            }), k.offset(a.extend(A, {
                using: j
            }))
        })
    }, a.ui.position = {
        fit: {
            left: function(a, b) {
                var c, d = b.within, e = d.isWindow ? d.scrollLeft: d.offset.left, f = d.width, h = a.left - b.collisionPosition.marginLeft, i = e - h, j = h + b.collisionWidth - f - e;
                b.collisionWidth > f ? i > 0 && 0 >= j ? (c = a.left + i + b.collisionWidth - f - e, a.left += i - c) : a.left = j > 0 && 0 >= i ? e : i > j ? e + f - b.collisionWidth : e : i > 0 ? a.left += i : j > 0 ? a.left -= j : a.left = g(a.left - h, a.left)
            },
            top: function(a, b) {
                var c, d = b.within, e = d.isWindow ? d.scrollTop: d.offset.top, f = b.within.height, h = a.top - b.collisionPosition.marginTop, i = e - h, j = h + b.collisionHeight - f - e;
                b.collisionHeight > f ? i > 0 && 0 >= j ? (c = a.top + i + b.collisionHeight - f - e, a.top += i - c) : a.top = j > 0 && 0 >= i ? e : i > j ? e + f - b.collisionHeight : e : i > 0 ? a.top += i : j > 0 ? a.top -= j : a.top = g(a.top - h, a.top)
            }
        },
        flip: {
            left: function(a, b) {
                var c, d, e = b.within, f = e.offset.left + e.scrollLeft, g = e.width, i = e.isWindow ? e.scrollLeft: e.offset.left, j = a.left - b.collisionPosition.marginLeft, k = j - i, l = j + b.collisionWidth - g - i, m = "left" === b.my[0]?-b.elemWidth : "right" === b.my[0] ? b.elemWidth : 0, n = "left" === b.at[0] ? b.targetWidth : "right" === b.at[0]?-b.targetWidth : 0, o =- 2 * b.offset[0];
                0 > k ? (c = a.left + m + n + o + b.collisionWidth - g - f, (0 > c || h(k) > c) && (a.left += m + n + o)) : l > 0 && (d = a.left - b.collisionPosition.marginLeft + m + n + o - i, (d > 0 || l > h(d)) && (a.left += m + n + o))
            },
            top: function(a, b) {
                var c, d, e = b.within, f = e.offset.top + e.scrollTop, g = e.height, i = e.isWindow ? e.scrollTop: e.offset.top, j = a.top - b.collisionPosition.marginTop, k = j - i, l = j + b.collisionHeight - g - i, m = "top" === b.my[1], n = m?-b.elemHeight : "bottom" === b.my[1] ? b.elemHeight : 0, o = "top" === b.at[1] ? b.targetHeight : "bottom" === b.at[1]?-b.targetHeight : 0, p =- 2 * b.offset[1];
                0 > k ? (d = a.top + n + o + p + b.collisionHeight - g - f, a.top + n + o + p > k && (0 > d || h(k) > d) && (a.top += n + o + p)) : l > 0 && (c = a.top - b.collisionPosition.marginTop + n + o + p - i, a.top + n + o + p > l && (c > 0 || l > h(c)) && (a.top += n + o + p))
            }
        },
        flipfit: {
            left: function() {
                a.ui.position.flip.left.apply(this, arguments), a.ui.position.fit.left.apply(this, arguments)
            },
            top: function() {
                a.ui.position.flip.top.apply(this, arguments), a.ui.position.fit.top.apply(this, arguments)
            }
        }
    }, function() {
        var b, c, d, e, f, g = document.getElementsByTagName("body")[0], h = document.createElement("div");
        b = document.createElement(g ? "div" : "body"), d = {
            visibility: "hidden",
            width: 0,
            height: 0,
            border: 0,
            margin: 0,
            background: "none"
        }, g && a.extend(d, {
            position: "absolute",
            left: "-1000px",
            top: "-1000px"
        });
        for (f in d)
            b.style[f] = d[f];
        b.appendChild(h), c = g || document.documentElement, c.insertBefore(b, c.firstChild), h.style.cssText = "position: absolute; left: 10.7432222px;", e = a(h).offset().left, a.support.offsetFractions = e > 10 && 11 > e, b.innerHTML = "", c.removeChild(b)
    }()
}(jQuery), function(a) {
    a.widget("ui.draggable", a.ui.mouse, {
        version: "1.10.3",
        widgetEventPrefix: "drag",
        options: {
            addClasses: !0,
            appendTo: "parent",
            axis: !1,
            connectToSortable: !1,
            containment: !1,
            cursor: "auto",
            cursorAt: !1,
            grid: !1,
            handle: !1,
            helper: "original",
            iframeFix: !1,
            opacity: !1,
            refreshPositions: !1,
            revert: !1,
            revertDuration: 500,
            scope: "default",
            scroll: !0,
            scrollSensitivity: 20,
            scrollSpeed: 20,
            snap: !1,
            snapMode: "both",
            snapTolerance: 20,
            stack: !1,
            zIndex: !1,
            drag: null,
            start: null,
            stop: null
        },
        _create: function() {
            "original" !== this.options.helper || /^(?:r|a|f)/.test(this.element.css("position")) || (this.element[0].style.position = "relative"), this.options.addClasses && this.element.addClass("ui-draggable"), this.options.disabled && this.element.addClass("ui-draggable-disabled"), this._mouseInit()
        },
        _destroy: function() {
            this.element.removeClass("ui-draggable ui-draggable-dragging ui-draggable-disabled"), this._mouseDestroy()
        },
        _mouseCapture: function(b) {
            var c = this.options;
            return this.helper || c.disabled || a(b.target).closest(".ui-resizable-handle").length > 0?!1 : (this.handle = this._getHandle(b), this.handle ? (a(c.iframeFix===!0 ? "iframe" : c.iframeFix).each(function() {
                a("<div class='ui-draggable-iframeFix' style='background: #fff;'></div>").css({
                    width: this.offsetWidth + "px",
                    height: this.offsetHeight + "px",
                    position: "absolute",
                    opacity: "0.001",
                    zIndex: 1e3
                }).css(a(this).offset()).appendTo("body")
            }), !0) : !1)
        },
        _mouseStart: function(b) {
            var c = this.options;
            return this.helper = this._createHelper(b), this.helper.addClass("ui-draggable-dragging"), this._cacheHelperProportions(), a.ui.ddmanager && (a.ui.ddmanager.current = this), this._cacheMargins(), this.cssPosition = this.helper.css("position"), this.scrollParent = this.helper.scrollParent(), this.offsetParent = this.helper.offsetParent(), this.offsetParentCssPosition = this.offsetParent.css("position"), this.offset = this.positionAbs = this.element.offset(), this.offset = {
                top: this.offset.top - this.margins.top,
                left: this.offset.left - this.margins.left
            }, this.offset.scroll=!1, a.extend(this.offset, {
                click: {
                    left: b.pageX - this.offset.left,
                    top: b.pageY - this.offset.top
                },
                parent: this._getParentOffset(),
                relative: this._getRelativeOffset()
            }), this.originalPosition = this.position = this._generatePosition(b), this.originalPageX = b.pageX, this.originalPageY = b.pageY, c.cursorAt && this._adjustOffsetFromHelper(c.cursorAt), this._setContainment(), this._trigger("start", b)===!1 ? (this._clear(), !1) : (this._cacheHelperProportions(), a.ui.ddmanager&&!c.dropBehaviour && a.ui.ddmanager.prepareOffsets(this, b), this._mouseDrag(b, !0), a.ui.ddmanager && a.ui.ddmanager.dragStart(this, b), !0)
        },
        _mouseDrag: function(b, c) {
            if ("fixed" === this.offsetParentCssPosition && (this.offset.parent = this._getParentOffset()), this.position = this._generatePosition(b), this.positionAbs = this._convertPositionTo("absolute"), !c) {
                var d = this._uiHash();
                if (this._trigger("drag", b, d)===!1)
                    return this._mouseUp({}), !1;
                this.position = d.position
            }
            return this.options.axis && "y" === this.options.axis || (this.helper[0].style.left = this.position.left + "px"), this.options.axis && "x" === this.options.axis || (this.helper[0].style.top = this.position.top + "px"), a.ui.ddmanager && a.ui.ddmanager.drag(this, b), !1
        },
        _mouseStop: function(b) {
            var c = this, d=!1;
            return a.ui.ddmanager&&!this.options.dropBehaviour && (d = a.ui.ddmanager.drop(this, b)), this.dropped && (d = this.dropped, this.dropped=!1), "original" !== this.options.helper || a.contains(this.element[0].ownerDocument, this.element[0]) ? ("invalid" === this.options.revert&&!d || "valid" === this.options.revert && d || this.options.revert===!0 || a.isFunction(this.options.revert) && this.options.revert.call(this.element, d) ? a(this.helper).animate(this.originalPosition, parseInt(this.options.revertDuration, 10), function() {
                c._trigger("stop", b)!==!1 && c._clear()
            }) : this._trigger("stop", b)!==!1 && this._clear(), !1) : !1
        },
        _mouseUp: function(b) {
            return a("div.ui-draggable-iframeFix").each(function() {
                this.parentNode.removeChild(this)
            }), a.ui.ddmanager && a.ui.ddmanager.dragStop(this, b), a.ui.mouse.prototype._mouseUp.call(this, b)
        },
        cancel: function() {
            return this.helper.is(".ui-draggable-dragging") ? this._mouseUp({}) : this._clear(), this
        },
        _getHandle: function(b) {
            return this.options.handle?!!a(b.target).closest(this.element.find(this.options.handle)).length : !0
        },
        _createHelper: function(b) {
            var c = this.options, d = a.isFunction(c.helper) ? a(c.helper.apply(this.element[0], [b])): "clone" === c.helper ? this.element.clone().removeAttr("id"): this.element;
            return d.parents("body").length || d.appendTo("parent" === c.appendTo ? this.element[0].parentNode : c.appendTo), d[0] === this.element[0] || /(fixed|absolute)/.test(d.css("position")) || d.css("position", "absolute"), d
        },
        _adjustOffsetFromHelper: function(b) {
            "string" == typeof b && (b = b.split(" ")), a.isArray(b) && (b = {
                left: + b[0],
                top: + b[1] || 0
            }), "left"in b && (this.offset.click.left = b.left + this.margins.left), "right"in b && (this.offset.click.left = this.helperProportions.width - b.right + this.margins.left), "top"in b && (this.offset.click.top = b.top + this.margins.top), "bottom"in b && (this.offset.click.top = this.helperProportions.height - b.bottom + this.margins.top)
        },
        _getParentOffset: function() {
            var b = this.offsetParent.offset();
            return "absolute" === this.cssPosition && this.scrollParent[0] !== document && a.contains(this.scrollParent[0], this.offsetParent[0]) && (b.left += this.scrollParent.scrollLeft(), b.top += this.scrollParent.scrollTop()), (this.offsetParent[0] === document.body || this.offsetParent[0].tagName && "html" === this.offsetParent[0].tagName.toLowerCase() && a.ui.ie) && (b = {
                top: 0,
                left: 0
            }), {
                top: b.top + (parseInt(this.offsetParent.css("borderTopWidth"), 10) || 0),
                left: b.left + (parseInt(this.offsetParent.css("borderLeftWidth"), 10) || 0)
            }
        },
        _getRelativeOffset: function() {
            if ("relative" === this.cssPosition) {
                var a = this.element.position();
                return {
                    top: a.top - (parseInt(this.helper.css("top"), 10) || 0) + this.scrollParent.scrollTop(),
                    left: a.left - (parseInt(this.helper.css("left"), 10) || 0) + this.scrollParent.scrollLeft()
                }
            }
            return {
                top: 0,
                left: 0
            }
        },
        _cacheMargins: function() {
            this.margins = {
                left: parseInt(this.element.css("marginLeft"), 10) || 0,
                top: parseInt(this.element.css("marginTop"), 10) || 0,
                right: parseInt(this.element.css("marginRight"), 10) || 0,
                bottom: parseInt(this.element.css("marginBottom"), 10) || 0
            }
        },
        _cacheHelperProportions: function() {
            this.helperProportions = {
                width: this.helper.outerWidth(),
                height: this.helper.outerHeight()
            }
        },
        _setContainment: function() {
            var b, c, d, e = this.options;
            return e.containment ? "window" === e.containment ? void(this.containment = [a(window).scrollLeft() - this.offset.relative.left - this.offset.parent.left, a(window).scrollTop() - this.offset.relative.top - this.offset.parent.top, a(window).scrollLeft() + a(window).width() - this.helperProportions.width - this.margins.left, a(window).scrollTop() + (a(window).height() || document.body.parentNode.scrollHeight) - this.helperProportions.height - this.margins.top]) : "document" === e.containment ? void(this.containment = [0, 0, a(document).width() - this.helperProportions.width - this.margins.left, (a(document).height() || document.body.parentNode.scrollHeight) - this.helperProportions.height - this.margins.top]) : e.containment.constructor === Array ? void(this.containment = e.containment) : ("parent" === e.containment && (e.containment = this.helper[0].parentNode), c = a(e.containment), d = c[0], void(d && (b = "hidden" !== c.css("overflow"), this.containment = [(parseInt(c.css("borderLeftWidth"), 10) || 0) + (parseInt(c.css("paddingLeft"), 10) || 0), (parseInt(c.css("borderTopWidth"), 10) || 0) + (parseInt(c.css("paddingTop"), 10) || 0), (b ? Math.max(d.scrollWidth, d.offsetWidth) : d.offsetWidth) - (parseInt(c.css("borderRightWidth"), 10) || 0) - (parseInt(c.css("paddingRight"), 10) || 0) - this.helperProportions.width - this.margins.left - this.margins.right, (b ? Math.max(d.scrollHeight, d.offsetHeight) : d.offsetHeight) - (parseInt(c.css("borderBottomWidth"), 10) || 0) - (parseInt(c.css("paddingBottom"), 10) || 0) - this.helperProportions.height - this.margins.top - this.margins.bottom], this.relative_container = c))) : void(this.containment = null)
        },
        _convertPositionTo: function(b, c) {
            c || (c = this.position);
            var d = "absolute" === b ? 1: - 1, e = "absolute" !== this.cssPosition || this.scrollParent[0] !== document && a.contains(this.scrollParent[0], this.offsetParent[0]) ? this.scrollParent: this.offsetParent;
            return this.offset.scroll || (this.offset.scroll = {
                top: e.scrollTop(),
                left: e.scrollLeft()
            }), {
                top: c.top + this.offset.relative.top * d + this.offset.parent.top * d - ("fixed" === this.cssPosition?-this.scrollParent.scrollTop() : this.offset.scroll.top) * d,
                left: c.left + this.offset.relative.left * d + this.offset.parent.left * d - ("fixed" === this.cssPosition?-this.scrollParent.scrollLeft() : this.offset.scroll.left) * d
            }
        },
        _generatePosition: function(b) {
            var c, d, e, f, g = this.options, h = "absolute" !== this.cssPosition || this.scrollParent[0] !== document && a.contains(this.scrollParent[0], this.offsetParent[0]) ? this.scrollParent: this.offsetParent, i = b.pageX, j = b.pageY;
            return this.offset.scroll || (this.offset.scroll = {
                top: h.scrollTop(),
                left: h.scrollLeft()
            }), this.originalPosition && (this.containment && (this.relative_container ? (d = this.relative_container.offset(), c = [this.containment[0] + d.left, this.containment[1] + d.top, this.containment[2] + d.left, this.containment[3] + d.top]) : c = this.containment, b.pageX - this.offset.click.left < c[0] && (i = c[0] + this.offset.click.left), b.pageY - this.offset.click.top < c[1] && (j = c[1] + this.offset.click.top), b.pageX - this.offset.click.left > c[2] && (i = c[2] + this.offset.click.left), b.pageY - this.offset.click.top > c[3] && (j = c[3] + this.offset.click.top)), g.grid && (e = g.grid[1] ? this.originalPageY + Math.round((j - this.originalPageY) / g.grid[1]) * g.grid[1] : this.originalPageY, j = c ? e - this.offset.click.top >= c[1] || e - this.offset.click.top > c[3] ? e : e - this.offset.click.top >= c[1] ? e - g.grid[1] : e + g.grid[1] : e, f = g.grid[0] ? this.originalPageX + Math.round((i - this.originalPageX) / g.grid[0]) * g.grid[0] : this.originalPageX, i = c ? f - this.offset.click.left >= c[0] || f - this.offset.click.left > c[2] ? f : f - this.offset.click.left >= c[0] ? f - g.grid[0] : f + g.grid[0] : f)), {
                top: j - this.offset.click.top - this.offset.relative.top - this.offset.parent.top + ("fixed" === this.cssPosition?-this.scrollParent.scrollTop() : this.offset.scroll.top),
                left: i - this.offset.click.left - this.offset.relative.left - this.offset.parent.left + ("fixed" === this.cssPosition?-this.scrollParent.scrollLeft() : this.offset.scroll.left)
            }
        },
        _clear: function() {
            this.helper.removeClass("ui-draggable-dragging"), this.helper[0] === this.element[0] || this.cancelHelperRemoval || this.helper.remove(), this.helper = null, this.cancelHelperRemoval=!1
        },
        _trigger: function(b, c, d) {
            return d = d || this._uiHash(), a.ui.plugin.call(this, b, [c, d]), "drag" === b && (this.positionAbs = this._convertPositionTo("absolute")), a.Widget.prototype._trigger.call(this, b, c, d)
        },
        plugins: {},
        _uiHash: function() {
            return {
                helper: this.helper,
                position: this.position,
                originalPosition: this.originalPosition,
                offset: this.positionAbs
            }
        }
    }), a.ui.plugin.add("draggable", "connectToSortable", {
        start: function(b, c) {
            var d = a(this).data("ui-draggable"), e = d.options, f = a.extend({}, c, {
                item: d.element
            });
            d.sortables = [], a(e.connectToSortable).each(function() {
                var c = a.data(this, "ui-sortable");
                c&&!c.options.disabled && (d.sortables.push({
                    instance: c,
                    shouldRevert: c.options.revert
                }), c.refreshPositions(), c._trigger("activate", b, f))
            })
        },
        stop: function(b, c) {
            var d = a(this).data("ui-draggable"), e = a.extend({}, c, {
                item: d.element
            });
            a.each(d.sortables, function() {
                this.instance.isOver ? (this.instance.isOver = 0, d.cancelHelperRemoval=!0, this.instance.cancelHelperRemoval=!1, this.shouldRevert && (this.instance.options.revert = this.shouldRevert), this.instance._mouseStop(b), this.instance.options.helper = this.instance.options._helper, "original" === d.options.helper && this.instance.currentItem.css({
                    top: "auto",
                    left: "auto"
                })) : (this.instance.cancelHelperRemoval=!1, this.instance._trigger("deactivate", b, e))
            })
        },
        drag: function(b, c) {
            var d = a(this).data("ui-draggable"), e = this;
            a.each(d.sortables, function() {
                var f=!1, g = this;
                this.instance.positionAbs = d.positionAbs, this.instance.helperProportions = d.helperProportions, this.instance.offset.click = d.offset.click, this.instance._intersectsWith(this.instance.containerCache) && (f=!0, a.each(d.sortables, function() {
                    return this.instance.positionAbs = d.positionAbs, this.instance.helperProportions = d.helperProportions, this.instance.offset.click = d.offset.click, this !== g && this.instance._intersectsWith(this.instance.containerCache) && a.contains(g.instance.element[0], this.instance.element[0]) && (f=!1), f
                })), f ? (this.instance.isOver || (this.instance.isOver = 1, this.instance.currentItem = a(e).clone().removeAttr("id").appendTo(this.instance.element).data("ui-sortable-item", !0), this.instance.options._helper = this.instance.options.helper, this.instance.options.helper = function() {
                    return c.helper[0]
                }, b.target = this.instance.currentItem[0], this.instance._mouseCapture(b, !0), this.instance._mouseStart(b, !0, !0), this.instance.offset.click.top = d.offset.click.top, this.instance.offset.click.left = d.offset.click.left, this.instance.offset.parent.left -= d.offset.parent.left - this.instance.offset.parent.left, this.instance.offset.parent.top -= d.offset.parent.top - this.instance.offset.parent.top, d._trigger("toSortable", b), d.dropped = this.instance.element, d.currentItem = d.element, this.instance.fromOutside = d), this.instance.currentItem && this.instance._mouseDrag(b)) : this.instance.isOver && (this.instance.isOver = 0, this.instance.cancelHelperRemoval=!0, this.instance.options.revert=!1, this.instance._trigger("out", b, this.instance._uiHash(this.instance)), this.instance._mouseStop(b, !0), this.instance.options.helper = this.instance.options._helper, this.instance.currentItem.remove(), this.instance.placeholder && this.instance.placeholder.remove(), d._trigger("fromSortable", b), d.dropped=!1)
            })
        }
    }), a.ui.plugin.add("draggable", "cursor", {
        start: function() {
            var b = a("body"), c = a(this).data("ui-draggable").options;
            b.css("cursor") && (c._cursor = b.css("cursor")), b.css("cursor", c.cursor)
        },
        stop: function() {
            var b = a(this).data("ui-draggable").options;
            b._cursor && a("body").css("cursor", b._cursor)
        }
    }), a.ui.plugin.add("draggable", "opacity", {
        start: function(b, c) {
            var d = a(c.helper), e = a(this).data("ui-draggable").options;
            d.css("opacity") && (e._opacity = d.css("opacity")), d.css("opacity", e.opacity)
        },
        stop: function(b, c) {
            var d = a(this).data("ui-draggable").options;
            d._opacity && a(c.helper).css("opacity", d._opacity)
        }
    }), a.ui.plugin.add("draggable", "scroll", {
        start: function() {
            var b = a(this).data("ui-draggable");
            b.scrollParent[0] !== document && "HTML" !== b.scrollParent[0].tagName && (b.overflowOffset = b.scrollParent.offset())
        },
        drag: function(b) {
            var c = a(this).data("ui-draggable"), d = c.options, e=!1;
            c.scrollParent[0] !== document && "HTML" !== c.scrollParent[0].tagName ? (d.axis && "x" === d.axis || (c.overflowOffset.top + c.scrollParent[0].offsetHeight - b.pageY < d.scrollSensitivity ? c.scrollParent[0].scrollTop = e = c.scrollParent[0].scrollTop + d.scrollSpeed : b.pageY - c.overflowOffset.top < d.scrollSensitivity && (c.scrollParent[0].scrollTop = e = c.scrollParent[0].scrollTop - d.scrollSpeed)), d.axis && "y" === d.axis || (c.overflowOffset.left + c.scrollParent[0].offsetWidth - b.pageX < d.scrollSensitivity ? c.scrollParent[0].scrollLeft = e = c.scrollParent[0].scrollLeft + d.scrollSpeed : b.pageX - c.overflowOffset.left < d.scrollSensitivity && (c.scrollParent[0].scrollLeft = e = c.scrollParent[0].scrollLeft - d.scrollSpeed))) : (d.axis && "x" === d.axis || (b.pageY - a(document).scrollTop() < d.scrollSensitivity ? e = a(document).scrollTop(a(document).scrollTop() - d.scrollSpeed) : a(window).height() - (b.pageY - a(document).scrollTop()) < d.scrollSensitivity && (e = a(document).scrollTop(a(document).scrollTop() + d.scrollSpeed))), d.axis && "y" === d.axis || (b.pageX - a(document).scrollLeft() < d.scrollSensitivity ? e = a(document).scrollLeft(a(document).scrollLeft() - d.scrollSpeed) : a(window).width() - (b.pageX - a(document).scrollLeft()) < d.scrollSensitivity && (e = a(document).scrollLeft(a(document).scrollLeft() + d.scrollSpeed)))), e!==!1 && a.ui.ddmanager&&!d.dropBehaviour && a.ui.ddmanager.prepareOffsets(c, b)
        }
    }), a.ui.plugin.add("draggable", "snap", {
        start: function() {
            var b = a(this).data("ui-draggable"), c = b.options;
            b.snapElements = [], a(c.snap.constructor !== String ? c.snap.items || ":data(ui-draggable)" : c.snap).each(function() {
                var c = a(this), d = c.offset();
                this !== b.element[0] && b.snapElements.push({
                    item: this,
                    width: c.outerWidth(),
                    height: c.outerHeight(),
                    top: d.top,
                    left: d.left
                })
            })
        },
        drag: function(b, c) {
            var d, e, f, g, h, i, j, k, l, m, n = a(this).data("ui-draggable"), o = n.options, p = o.snapTolerance, q = c.offset.left, r = q + n.helperProportions.width, s = c.offset.top, t = s + n.helperProportions.height;
            for (l = n.snapElements.length - 1; l >= 0; l--)
                h = n.snapElements[l].left, i = h + n.snapElements[l].width, j = n.snapElements[l].top, k = j + n.snapElements[l].height, h - p > r || q > i + p || j - p > t || s > k + p ||!a.contains(n.snapElements[l].item.ownerDocument, n.snapElements[l].item) ? (n.snapElements[l].snapping && n.options.snap.release && n.options.snap.release.call(n.element, b, a.extend(n._uiHash(), {
                    snapItem: n.snapElements[l].item
                })), n.snapElements[l].snapping=!1) : ("inner" !== o.snapMode && (d = p >= Math.abs(j - t), e = p >= Math.abs(k - s), f = p >= Math.abs(h - r), g = p >= Math.abs(i - q), d && (c.position.top = n._convertPositionTo("relative", {
                    top: j - n.helperProportions.height,
                    left: 0
                }).top - n.margins.top), e && (c.position.top = n._convertPositionTo("relative", {
                    top: k,
                    left: 0
                }).top - n.margins.top), f && (c.position.left = n._convertPositionTo("relative", {
                    top: 0,
                    left: h - n.helperProportions.width
                }).left - n.margins.left), g && (c.position.left = n._convertPositionTo("relative", {
                    top: 0,
                    left: i
                }).left - n.margins.left)), m = d || e || f || g, "outer" !== o.snapMode && (d = p >= Math.abs(j - s), e = p >= Math.abs(k - t), f = p >= Math.abs(h - q), g = p >= Math.abs(i - r), d && (c.position.top = n._convertPositionTo("relative", {
                    top: j,
                    left: 0
                }).top - n.margins.top), e && (c.position.top = n._convertPositionTo("relative", {
                    top: k - n.helperProportions.height,
                    left: 0
                }).top - n.margins.top), f && (c.position.left = n._convertPositionTo("relative", {
                    top: 0,
                    left: h
                }).left - n.margins.left), g && (c.position.left = n._convertPositionTo("relative", {
                    top: 0,
                    left: i - n.helperProportions.width
                }).left - n.margins.left)), !n.snapElements[l].snapping && (d || e || f || g || m) && n.options.snap.snap && n.options.snap.snap.call(n.element, b, a.extend(n._uiHash(), {
                    snapItem: n.snapElements[l].item
                })), n.snapElements[l].snapping = d || e || f || g || m)
        }
    }), a.ui.plugin.add("draggable", "stack", {
        start: function() {
            var b, c = this.data("ui-draggable").options, d = a.makeArray(a(c.stack)).sort(function(b, c) {
                return (parseInt(a(b).css("zIndex"), 10) || 0) - (parseInt(a(c).css("zIndex"), 10) || 0)
            });
            d.length && (b = parseInt(a(d[0]).css("zIndex"), 10) || 0, a(d).each(function(c) {
                a(this).css("zIndex", b + c)
            }), this.css("zIndex", b + d.length))
        }
    }), a.ui.plugin.add("draggable", "zIndex", {
        start: function(b, c) {
            var d = a(c.helper), e = a(this).data("ui-draggable").options;
            d.css("zIndex") && (e._zIndex = d.css("zIndex")), d.css("zIndex", e.zIndex)
        },
        stop: function(b, c) {
            var d = a(this).data("ui-draggable").options;
            d._zIndex && a(c.helper).css("zIndex", d._zIndex)
        }
    })
}(jQuery), function(a) {
    function b(a) {
        return parseInt(a, 10) || 0
    }
    function c(a) {
        return !isNaN(parseInt(a, 10))
    }
    a.widget("ui.resizable", a.ui.mouse, {
        version: "1.10.3",
        widgetEventPrefix: "resize",
        options: {
            alsoResize: !1,
            animate: !1,
            animateDuration: "slow",
            animateEasing: "swing",
            aspectRatio: !1,
            autoHide: !1,
            containment: !1,
            ghost: !1,
            grid: !1,
            handles: "e,s,se",
            helper: !1,
            maxHeight: null,
            maxWidth: null,
            minHeight: 10,
            minWidth: 10,
            zIndex: 90,
            resize: null,
            start: null,
            stop: null
        },
        _create: function() {
            var b, c, d, e, f, g = this, h = this.options;
            if (this.element.addClass("ui-resizable"), a.extend(this, {
                _aspectRatio: !!h.aspectRatio,
                aspectRatio: h.aspectRatio,
                originalElement: this.element,
                _proportionallyResizeElements: [],
                _helper: h.helper || h.ghost || h.animate ? h.helper || "ui-resizable-helper": null
            }), this.element[0].nodeName.match(/canvas|textarea|input|select|button|img/i) && (this.element.wrap(a("<div class='ui-wrapper' style='overflow: hidden;'></div>").css({
                position: this.element.css("position"),
                width: this.element.outerWidth(),
                height: this.element.outerHeight(),
                top: this.element.css("top"),
                left: this.element.css("left")
            })), this.element = this.element.parent().data("ui-resizable", this.element.data("ui-resizable")), this.elementIsWrapper=!0, this.element.css({
                marginLeft: this.originalElement.css("marginLeft"),
                marginTop: this.originalElement.css("marginTop"),
                marginRight: this.originalElement.css("marginRight"),
                marginBottom: this.originalElement.css("marginBottom")
            }), this.originalElement.css({
                marginLeft: 0,
                marginTop: 0,
                marginRight: 0,
                marginBottom: 0
            }), this.originalResizeStyle = this.originalElement.css("resize"), this.originalElement.css("resize", "none"), this._proportionallyResizeElements.push(this.originalElement.css({
                position: "static",
                zoom: 1,
                display: "block"
            })), this.originalElement.css({
                margin: this.originalElement.css("margin")
            }), this._proportionallyResize()), this.handles = h.handles || (a(".ui-resizable-handle", this.element).length ? {
                n: ".ui-resizable-n",
                e: ".ui-resizable-e",
                s: ".ui-resizable-s",
                w: ".ui-resizable-w",
                se: ".ui-resizable-se",
                sw: ".ui-resizable-sw",
                ne: ".ui-resizable-ne",
                nw: ".ui-resizable-nw"
            } : "e,s,se"), this.handles.constructor === String)
                for ("all" === this.handles && (this.handles = "n,e,s,w,se,sw,ne,nw"), b = this.handles.split(","), this.handles = {}, c = 0; b.length > c; c++)
                    d = a.trim(b[c]), f = "ui-resizable-" + d, e = a("<div class='ui-resizable-handle " + f + "'></div>"), e.css({
                        zIndex: h.zIndex
                    }), "se" === d && e.addClass("ui-icon ui-icon-gripsmall-diagonal-se"), this.handles[d] = ".ui-resizable-" + d, this.element.append(e);
            this._renderAxis = function(b) {
                var c, d, e, f;
                b = b || this.element;
                for (c in this.handles)
                    this.handles[c].constructor === String && (this.handles[c] = a(this.handles[c], this.element).show()), this.elementIsWrapper && this.originalElement[0].nodeName.match(/textarea|input|select|button/i) && (d = a(this.handles[c], this.element), f = /sw|ne|nw|se|n|s/.test(c) ? d.outerHeight() : d.outerWidth(), e = ["padding", /ne|nw|n/.test(c) ? "Top": /se|sw|s/.test(c) ? "Bottom": /^e$/.test(c) ? "Right": "Left"].join(""), b.css(e, f), this._proportionallyResize()), a(this.handles[c]).length
            }, this._renderAxis(this.element), this._handles = a(".ui-resizable-handle", this.element).disableSelection(), this._handles.mouseover(function() {
                g.resizing || (this.className && (e = this.className.match(/ui-resizable-(se|sw|ne|nw|n|e|s|w)/i)), g.axis = e && e[1] ? e[1] : "se")
            }), h.autoHide && (this._handles.hide(), a(this.element).addClass("ui-resizable-autohide").mouseenter(function() {
                h.disabled || (a(this).removeClass("ui-resizable-autohide"), g._handles.show())
            }).mouseleave(function() {
                h.disabled || g.resizing || (a(this).addClass("ui-resizable-autohide"), g._handles.hide())
            })), this._mouseInit()
        },
        _destroy: function() {
            this._mouseDestroy();
            var b, c = function(b) {
                a(b).removeClass("ui-resizable ui-resizable-disabled ui-resizable-resizing").removeData("resizable").removeData("ui-resizable").unbind(".resizable").find(".ui-resizable-handle").remove()
            };
            return this.elementIsWrapper && (c(this.element), b = this.element, this.originalElement.css({
                position: b.css("position"),
                width: b.outerWidth(),
                height: b.outerHeight(),
                top: b.css("top"),
                left: b.css("left")
            }).insertAfter(b), b.remove()), this.originalElement.css("resize", this.originalResizeStyle), c(this.originalElement), this
        },
        _mouseCapture: function(b) {
            var c, d, e=!1;
            for (c in this.handles)
                d = a(this.handles[c])[0], (d === b.target || a.contains(d, b.target)) && (e=!0);
            return !this.options.disabled && e
        },
        _mouseStart: function(c) {
            var d, e, f, g = this.options, h = this.element.position(), i = this.element;
            return this.resizing=!0, /absolute/.test(i.css("position")) ? i.css({
                position: "absolute",
                top: i.css("top"),
                left: i.css("left")
            }) : i.is(".ui-draggable") && i.css({
                position: "absolute",
                top: h.top,
                left: h.left
            }), this._renderProxy(), d = b(this.helper.css("left")), e = b(this.helper.css("top")), g.containment && (d += a(g.containment).scrollLeft() || 0, e += a(g.containment).scrollTop() || 0), this.offset = this.helper.offset(), this.position = {
                left: d,
                top: e
            }, this.size = this._helper ? {
                width: i.outerWidth(),
                height: i.outerHeight()
            } : {
                width: i.width(),
                height: i.height()
            }, this.originalSize = this._helper ? {
                width: i.outerWidth(),
                height: i.outerHeight()
            } : {
                width: i.width(),
                height: i.height()
            }, this.originalPosition = {
                left: d,
                top: e
            }, this.sizeDiff = {
                width: i.outerWidth() - i.width(),
                height: i.outerHeight() - i.height()
            }, this.originalMousePosition = {
                left: c.pageX,
                top: c.pageY
            }, this.aspectRatio = "number" == typeof g.aspectRatio ? g.aspectRatio : this.originalSize.width / this.originalSize.height || 1, f = a(".ui-resizable-" + this.axis).css("cursor"), a("body").css("cursor", "auto" === f ? this.axis + "-resize" : f), i.addClass("ui-resizable-resizing"), this._propagate("start", c), !0
        },
        _mouseDrag: function(b) {
            var c, d = this.helper, e = {}, f = this.originalMousePosition, g = this.axis, h = this.position.top, i = this.position.left, j = this.size.width, k = this.size.height, l = b.pageX - f.left || 0, m = b.pageY - f.top || 0, n = this._change[g];
            return n ? (c = n.apply(this, [b, l, m]), this._updateVirtualBoundaries(b.shiftKey), (this._aspectRatio || b.shiftKey) && (c = this._updateRatio(c, b)), c = this._respectSize(c, b), this._updateCache(c), this._propagate("resize", b), this.position.top !== h && (e.top = this.position.top + "px"), this.position.left !== i && (e.left = this.position.left + "px"), this.size.width !== j && (e.width = this.size.width + "px"), this.size.height !== k && (e.height = this.size.height + "px"), d.css(e), !this._helper && this._proportionallyResizeElements.length && this._proportionallyResize(), a.isEmptyObject(e) || this._trigger("resize", b, this.ui()), !1) : !1
        },
        _mouseStop: function(b) {
            this.resizing=!1;
            var c, d, e, f, g, h, i, j = this.options, k = this;
            return this._helper && (c = this._proportionallyResizeElements, d = c.length && /textarea/i.test(c[0].nodeName), e = d && a.ui.hasScroll(c[0], "left") ? 0 : k.sizeDiff.height, f = d ? 0 : k.sizeDiff.width, g = {
                width: k.helper.width() - f,
                height: k.helper.height() - e
            }, h = parseInt(k.element.css("left"), 10) + (k.position.left - k.originalPosition.left) || null, i = parseInt(k.element.css("top"), 10) + (k.position.top - k.originalPosition.top) || null, j.animate || this.element.css(a.extend(g, {
                top: i,
                left: h
            })), k.helper.height(k.size.height), k.helper.width(k.size.width), this._helper&&!j.animate && this._proportionallyResize()), a("body").css("cursor", "auto"), this.element.removeClass("ui-resizable-resizing"), this._propagate("stop", b), this._helper && this.helper.remove(), !1
        },
        _updateVirtualBoundaries: function(a) {
            var b, d, e, f, g, h = this.options;
            g = {
                minWidth: c(h.minWidth) ? h.minWidth: 0,
                maxWidth: c(h.maxWidth) ? h.maxWidth: 1 / 0,
                minHeight: c(h.minHeight) ? h.minHeight: 0,
                maxHeight: c(h.maxHeight) ? h.maxHeight: 1 / 0
            }, (this._aspectRatio || a) && (b = g.minHeight * this.aspectRatio, e = g.minWidth / this.aspectRatio, d = g.maxHeight * this.aspectRatio, f = g.maxWidth / this.aspectRatio, b > g.minWidth && (g.minWidth = b), e > g.minHeight && (g.minHeight = e), g.maxWidth > d && (g.maxWidth = d), g.maxHeight > f && (g.maxHeight = f)), this._vBoundaries = g
        },
        _updateCache: function(a) {
            this.offset = this.helper.offset(), c(a.left) && (this.position.left = a.left), c(a.top) && (this.position.top = a.top), c(a.height) && (this.size.height = a.height), c(a.width) && (this.size.width = a.width)
        },
        _updateRatio: function(a) {
            var b = this.position, d = this.size, e = this.axis;
            return c(a.height) ? a.width = a.height * this.aspectRatio : c(a.width) && (a.height = a.width / this.aspectRatio), "sw" === e && (a.left = b.left + (d.width - a.width), a.top = null), "nw" === e && (a.top = b.top + (d.height - a.height), a.left = b.left + (d.width - a.width)), a
        },
        _respectSize: function(a) {
            var b = this._vBoundaries, d = this.axis, e = c(a.width) && b.maxWidth && b.maxWidth < a.width, f = c(a.height) && b.maxHeight && b.maxHeight < a.height, g = c(a.width) && b.minWidth && b.minWidth > a.width, h = c(a.height) && b.minHeight && b.minHeight > a.height, i = this.originalPosition.left + this.originalSize.width, j = this.position.top + this.size.height, k = /sw|nw|w/.test(d), l = /nw|ne|n/.test(d);
            return g && (a.width = b.minWidth), h && (a.height = b.minHeight), e && (a.width = b.maxWidth), f && (a.height = b.maxHeight), g && k && (a.left = i - b.minWidth), e && k && (a.left = i - b.maxWidth), h && l && (a.top = j - b.minHeight), f && l && (a.top = j - b.maxHeight), a.width || a.height || a.left ||!a.top ? a.width || a.height || a.top ||!a.left || (a.left = null) : a.top = null, a
        },
        _proportionallyResize: function() {
            if (this._proportionallyResizeElements.length) {
                var a, b, c, d, e, f = this.helper || this.element;
                for (a = 0; this._proportionallyResizeElements.length > a; a++) {
                    if (e = this._proportionallyResizeElements[a], !this.borderDif)
                        for (this.borderDif = [], c = [e.css("borderTopWidth"), e.css("borderRightWidth"), e.css("borderBottomWidth"), e.css("borderLeftWidth")], d = [e.css("paddingTop"), e.css("paddingRight"), e.css("paddingBottom"), e.css("paddingLeft")], b = 0; c.length > b; b++)
                            this.borderDif[b] = (parseInt(c[b], 10) || 0) + (parseInt(d[b], 10) || 0);
                    e.css({
                        height: f.height() - this.borderDif[0] - this.borderDif[2] || 0,
                        width: f.width() - this.borderDif[1] - this.borderDif[3] || 0
                    })
                }
            }
        },
        _renderProxy: function() {
            var b = this.element, c = this.options;
            this.elementOffset = b.offset(), this._helper ? (this.helper = this.helper || a("<div style='overflow:hidden;'></div>"), this.helper.addClass(this._helper).css({
                width: this.element.outerWidth() - 1,
                height: this.element.outerHeight() - 1,
                position: "absolute",
                left: this.elementOffset.left + "px",
                top: this.elementOffset.top + "px",
                zIndex: ++c.zIndex
            }), this.helper.appendTo("body").disableSelection()) : this.helper = this.element
        },
        _change: {
            e: function(a, b) {
                return {
                    width: this.originalSize.width + b
                }
            },
            w: function(a, b) {
                var c = this.originalSize, d = this.originalPosition;
                return {
                    left: d.left + b,
                    width: c.width - b
                }
            },
            n: function(a, b, c) {
                var d = this.originalSize, e = this.originalPosition;
                return {
                    top: e.top + c,
                    height: d.height - c
                }
            },
            s: function(a, b, c) {
                return {
                    height: this.originalSize.height + c
                }
            },
            se: function(b, c, d) {
                return a.extend(this._change.s.apply(this, arguments), this._change.e.apply(this, [b, c, d]))
            },
            sw: function(b, c, d) {
                return a.extend(this._change.s.apply(this, arguments), this._change.w.apply(this, [b, c, d]))
            },
            ne: function(b, c, d) {
                return a.extend(this._change.n.apply(this, arguments), this._change.e.apply(this, [b, c, d]))
            },
            nw: function(b, c, d) {
                return a.extend(this._change.n.apply(this, arguments), this._change.w.apply(this, [b, c, d]))
            }
        },
        _propagate: function(b, c) {
            a.ui.plugin.call(this, b, [c, this.ui()]), "resize" !== b && this._trigger(b, c, this.ui())
        },
        plugins: {},
        ui: function() {
            return {
                originalElement: this.originalElement,
                element: this.element,
                helper: this.helper,
                position: this.position,
                size: this.size,
                originalSize: this.originalSize,
                originalPosition: this.originalPosition
            }
        }
    }), a.ui.plugin.add("resizable", "animate", {
        stop: function(b) {
            var c = a(this).data("ui-resizable"), d = c.options, e = c._proportionallyResizeElements, f = e.length && /textarea/i.test(e[0].nodeName), g = f && a.ui.hasScroll(e[0], "left") ? 0: c.sizeDiff.height, h = f ? 0: c.sizeDiff.width, i = {
                width: c.size.width - h,
                height: c.size.height - g
            }, j = parseInt(c.element.css("left"), 10) + (c.position.left - c.originalPosition.left) || null, k = parseInt(c.element.css("top"), 10) + (c.position.top - c.originalPosition.top) || null;
            c.element.animate(a.extend(i, k && j ? {
                top: k,
                left: j
            } : {}), {
                duration: d.animateDuration,
                easing: d.animateEasing,
                step: function() {
                    var d = {
                        width: parseInt(c.element.css("width"), 10),
                        height: parseInt(c.element.css("height"), 10),
                        top: parseInt(c.element.css("top"), 10),
                        left: parseInt(c.element.css("left"), 10)
                    };
                    e && e.length && a(e[0]).css({
                        width: d.width,
                        height: d.height
                    }), c._updateCache(d), c._propagate("resize", b)
                }
            })
        }
    }), a.ui.plugin.add("resizable", "containment", {
        start: function() {
            var c, d, e, f, g, h, i, j = a(this).data("ui-resizable"), k = j.options, l = j.element, m = k.containment, n = m instanceof a ? m.get(0): /parent/.test(m) ? l.parent().get(0): m;
            n && (j.containerElement = a(n), /document/.test(m) || m === document ? (j.containerOffset = {
                left: 0,
                top: 0
            }, j.containerPosition = {
                left: 0,
                top: 0
            }, j.parentData = {
                element: a(document),
                left: 0,
                top: 0,
                width: a(document).width(),
                height: a(document).height() || document.body.parentNode.scrollHeight
            }) : (c = a(n), d = [], a(["Top", "Right", "Left", "Bottom"]).each(function(a, e) {
                d[a] = b(c.css("padding" + e))
            }), j.containerOffset = c.offset(), j.containerPosition = c.position(), j.containerSize = {
                height: c.innerHeight() - d[3],
                width: c.innerWidth() - d[1]
            }, e = j.containerOffset, f = j.containerSize.height, g = j.containerSize.width, h = a.ui.hasScroll(n, "left") ? n.scrollWidth : g, i = a.ui.hasScroll(n) ? n.scrollHeight : f, j.parentData = {
                element: n,
                left: e.left,
                top: e.top,
                width: h,
                height: i
            }))
        },
        resize: function(b) {
            var c, d, e, f, g = a(this).data("ui-resizable"), h = g.options, i = g.containerOffset, j = g.position, k = g._aspectRatio || b.shiftKey, l = {
                top: 0,
                left: 0
            }, m = g.containerElement;
            m[0] !== document && /static/.test(m.css("position")) && (l = i), j.left < (g._helper ? i.left : 0) && (g.size.width = g.size.width + (g._helper ? g.position.left - i.left : g.position.left - l.left), k && (g.size.height = g.size.width / g.aspectRatio), g.position.left = h.helper ? i.left : 0), j.top < (g._helper ? i.top : 0) && (g.size.height = g.size.height + (g._helper ? g.position.top - i.top : g.position.top), k && (g.size.width = g.size.height * g.aspectRatio), g.position.top = g._helper ? i.top : 0), g.offset.left = g.parentData.left + g.position.left, g.offset.top = g.parentData.top + g.position.top, c = Math.abs((g._helper ? g.offset.left - l.left : g.offset.left - l.left) + g.sizeDiff.width), d = Math.abs((g._helper ? g.offset.top - l.top : g.offset.top - i.top) + g.sizeDiff.height), e = g.containerElement.get(0) === g.element.parent().get(0), f = /relative|absolute/.test(g.containerElement.css("position")), e && f && (c -= g.parentData.left), c + g.size.width >= g.parentData.width && (g.size.width = g.parentData.width - c, k && (g.size.height = g.size.width / g.aspectRatio)), d + g.size.height >= g.parentData.height && (g.size.height = g.parentData.height - d, k && (g.size.width = g.size.height * g.aspectRatio))
        },
        stop: function() {
            var b = a(this).data("ui-resizable"), c = b.options, d = b.containerOffset, e = b.containerPosition, f = b.containerElement, g = a(b.helper), h = g.offset(), i = g.outerWidth() - b.sizeDiff.width, j = g.outerHeight() - b.sizeDiff.height;
            b._helper&&!c.animate && /relative/.test(f.css("position")) && a(this).css({
                left: h.left - e.left - d.left,
                width: i,
                height: j
            }), b._helper&&!c.animate && /static/.test(f.css("position")) && a(this).css({
                left: h.left - e.left - d.left,
                width: i,
                height: j
            })
        }
    }), a.ui.plugin.add("resizable", "alsoResize", {
        start: function() {
            var b = a(this).data("ui-resizable"), c = b.options, d = function(b) {
                a(b).each(function() {
                    var b = a(this);
                    b.data("ui-resizable-alsoresize", {
                        width: parseInt(b.width(), 10),
                        height: parseInt(b.height(), 10),
                        left: parseInt(b.css("left"), 10),
                        top: parseInt(b.css("top"), 10)
                    })
                })
            };
            "object" != typeof c.alsoResize || c.alsoResize.parentNode ? d(c.alsoResize) : c.alsoResize.length ? (c.alsoResize = c.alsoResize[0], d(c.alsoResize)) : a.each(c.alsoResize, function(a) {
                d(a)
            })
        },
        resize: function(b, c) {
            var d = a(this).data("ui-resizable"), e = d.options, f = d.originalSize, g = d.originalPosition, h = {
                height: d.size.height - f.height || 0,
                width: d.size.width - f.width || 0,
                top: d.position.top - g.top || 0,
                left: d.position.left - g.left || 0
            }, i = function(b, d) {
                a(b).each(function() {
                    var b = a(this), e = a(this).data("ui-resizable-alsoresize"), f = {}, g = d && d.length ? d: b.parents(c.originalElement[0]).length ? ["width", "height"]: ["width", "height", "top", "left"];
                    a.each(g, function(a, b) {
                        var c = (e[b] || 0) + (h[b] || 0);
                        c && c >= 0 && (f[b] = c || null)
                    }), b.css(f)
                })
            };
            "object" != typeof e.alsoResize || e.alsoResize.nodeType ? i(e.alsoResize) : a.each(e.alsoResize, function(a, b) {
                i(a, b)
            })
        },
        stop: function() {
            a(this).removeData("resizable-alsoresize")
        }
    }), a.ui.plugin.add("resizable", "ghost", {
        start: function() {
            var b = a(this).data("ui-resizable"), c = b.options, d = b.size;
            b.ghost = b.originalElement.clone(), b.ghost.css({
                opacity: .25,
                display: "block",
                position: "relative",
                height: d.height,
                width: d.width,
                margin: 0,
                left: 0,
                top: 0
            }).addClass("ui-resizable-ghost").addClass("string" == typeof c.ghost ? c.ghost : ""), b.ghost.appendTo(b.helper)
        },
        resize: function() {
            var b = a(this).data("ui-resizable");
            b.ghost && b.ghost.css({
                position: "relative",
                height: b.size.height,
                width: b.size.width
            })
        },
        stop: function() {
            var b = a(this).data("ui-resizable");
            b.ghost && b.helper && b.helper.get(0).removeChild(b.ghost.get(0))
        }
    }), a.ui.plugin.add("resizable", "grid", {
        resize: function() {
            var b = a(this).data("ui-resizable"), c = b.options, d = b.size, e = b.originalSize, f = b.originalPosition, g = b.axis, h = "number" == typeof c.grid ? [c.grid, c.grid]: c.grid, i = h[0] || 1, j = h[1] || 1, k = Math.round((d.width - e.width) / i) * i, l = Math.round((d.height - e.height) / j) * j, m = e.width + k, n = e.height + l, o = c.maxWidth && m > c.maxWidth, p = c.maxHeight && n > c.maxHeight, q = c.minWidth && c.minWidth > m, r = c.minHeight && c.minHeight > n;
            c.grid = h, q && (m += i), r && (n += j), o && (m -= i), p && (n -= j), /^(se|s|e)$/.test(g) ? (b.size.width = m, b.size.height = n) : /^(ne)$/.test(g) ? (b.size.width = m, b.size.height = n, b.position.top = f.top - l) : /^(sw)$/.test(g) ? (b.size.width = m, b.size.height = n, b.position.left = f.left - k) : (b.size.width = m, b.size.height = n, b.position.top = f.top - l, b.position.left = f.left - k)
        }
    })
}(jQuery), function(a) {
    var b, c, d, e, f = "ui-button ui-widget ui-state-default ui-corner-all", g = "ui-state-hover ui-state-active ", h = "ui-button-icons-only ui-button-icon-only ui-button-text-icons ui-button-text-icon-primary ui-button-text-icon-secondary ui-button-text-only", i = function() {
        var b = a(this);
        setTimeout(function() {
            b.find(":ui-button").button("refresh")
        }, 1)
    }, j = function(b) {
        var c = b.name, d = b.form, e = a([]);
        return c && (c = c.replace(/'/g, "\\'"), e = d ? a(d).find("[name='" + c + "']") : a("[name='" + c + "']", b.ownerDocument).filter(function() {
            return !this.form
        })), e
    };
    a.widget("ui.button", {
        version: "1.10.3",
        defaultElement: "<button>",
        options: {
            disabled: null,
            text: !0,
            label: null,
            icons: {
                primary: null,
                secondary: null
            }
        },
        _create: function() {
            this.element.closest("form").unbind("reset" + this.eventNamespace).bind("reset" + this.eventNamespace, i), "boolean" != typeof this.options.disabled ? this.options.disabled=!!this.element.prop("disabled") : this.element.prop("disabled", this.options.disabled), this._determineButtonType(), this.hasTitle=!!this.buttonElement.attr("title");
            var g = this, h = this.options, k = "checkbox" === this.type || "radio" === this.type, l = k ? "": "ui-state-active", m = "ui-state-focus";
            null === h.label && (h.label = "input" === this.type ? this.buttonElement.val() : this.buttonElement.html()), this._hoverable(this.buttonElement), this.buttonElement.addClass(f).attr("role", "button").bind("mouseenter" + this.eventNamespace, function() {
                h.disabled || this === b && a(this).addClass("ui-state-active")
            }).bind("mouseleave" + this.eventNamespace, function() {
                h.disabled || a(this).removeClass(l)
            }).bind("click" + this.eventNamespace, function(a) {
                h.disabled && (a.preventDefault(), a.stopImmediatePropagation())
            }), this.element.bind("focus" + this.eventNamespace, function() {
                g.buttonElement.addClass(m)
            }).bind("blur" + this.eventNamespace, function() {
                g.buttonElement.removeClass(m)
            }), k && (this.element.bind("change" + this.eventNamespace, function() {
                e || g.refresh()
            }), this.buttonElement.bind("mousedown" + this.eventNamespace, function(a) {
                h.disabled || (e=!1, c = a.pageX, d = a.pageY)
            }).bind("mouseup" + this.eventNamespace, function(a) {
                h.disabled || (c !== a.pageX || d !== a.pageY) && (e=!0)
            })), "checkbox" === this.type ? this.buttonElement.bind("click" + this.eventNamespace, function() {
                return h.disabled || e?!1 : void 0
            }) : "radio" === this.type ? this.buttonElement.bind("click" + this.eventNamespace, function() {
                if (h.disabled || e)
                    return !1;
                a(this).addClass("ui-state-active"), g.buttonElement.attr("aria-pressed", "true");
                var b = g.element[0];
                j(b).not(b).map(function() {
                    return a(this).button("widget")[0]
                }).removeClass("ui-state-active").attr("aria-pressed", "false")
            }) : (this.buttonElement.bind("mousedown" + this.eventNamespace, function() {
                return h.disabled?!1 : (a(this).addClass("ui-state-active"), b = this, void g.document.one("mouseup", function() {
                    b = null
                }))
            }).bind("mouseup" + this.eventNamespace, function() {
                return h.disabled?!1 : void a(this).removeClass("ui-state-active")
            }).bind("keydown" + this.eventNamespace, function(b) {
                return h.disabled?!1 : void((b.keyCode === a.ui.keyCode.SPACE || b.keyCode === a.ui.keyCode.ENTER) && a(this).addClass("ui-state-active"))
            }).bind("keyup" + this.eventNamespace + " blur" + this.eventNamespace, function() {
                a(this).removeClass("ui-state-active")
            }), this.buttonElement.is("a") && this.buttonElement.keyup(function(b) {
                b.keyCode === a.ui.keyCode.SPACE && a(this).click()
            })), this._setOption("disabled", h.disabled), this._resetButton()
        },
        _determineButtonType: function() {
            var a, b, c;
            this.type = this.element.is("[type=checkbox]") ? "checkbox" : this.element.is("[type=radio]") ? "radio" : this.element.is("input") ? "input" : "button", "checkbox" === this.type || "radio" === this.type ? (a = this.element.parents().last(), b = "label[for='" + this.element.attr("id") + "']", this.buttonElement = a.find(b), this.buttonElement.length || (a = a.length ? a.siblings() : this.element.siblings(), this.buttonElement = a.filter(b), this.buttonElement.length || (this.buttonElement = a.find(b))), this.element.addClass("ui-helper-hidden-accessible"), c = this.element.is(":checked"), c && this.buttonElement.addClass("ui-state-active"), this.buttonElement.prop("aria-pressed", c)) : this.buttonElement = this.element
        },
        widget: function() {
            return this.buttonElement
        },
        _destroy: function() {
            this.element.removeClass("ui-helper-hidden-accessible"), this.buttonElement.removeClass(f + " " + g + " " + h).removeAttr("role").removeAttr("aria-pressed").html(this.buttonElement.find(".ui-button-text").html()), this.hasTitle || this.buttonElement.removeAttr("title")
        },
        _setOption: function(a, b) {
            return this._super(a, b), "disabled" === a ? void(b ? this.element.prop("disabled", !0) : this.element.prop("disabled", !1)) : void this._resetButton()
        },
        refresh: function() {
            var b = this.element.is("input, button") ? this.element.is(":disabled"): this.element.hasClass("ui-button-disabled");
            b !== this.options.disabled && this._setOption("disabled", b), "radio" === this.type ? j(this.element[0]).each(function() {
                a(this).is(":checked") ? a(this).button("widget").addClass("ui-state-active").attr("aria-pressed", "true") : a(this).button("widget").removeClass("ui-state-active").attr("aria-pressed", "false")
            }) : "checkbox" === this.type && (this.element.is(":checked") ? this.buttonElement.addClass("ui-state-active").attr("aria-pressed", "true") : this.buttonElement.removeClass("ui-state-active").attr("aria-pressed", "false"))
        },
        _resetButton: function() {
            if ("input" === this.type)
                return void(this.options.label && this.element.val(this.options.label));
            var b = this.buttonElement.removeClass(h), c = a("<span></span>", this.document[0]).addClass("ui-button-text").html(this.options.label).appendTo(b.empty()).text(), d = this.options.icons, e = d.primary && d.secondary, f = [];
            d.primary || d.secondary ? (this.options.text && f.push("ui-button-text-icon" + (e ? "s" : d.primary ? "-primary" : "-secondary")), d.primary && b.prepend("<span class='ui-button-icon-primary ui-icon " + d.primary + "'></span>"), d.secondary && b.append("<span class='ui-button-icon-secondary ui-icon " + d.secondary + "'></span>"), this.options.text || (f.push(e ? "ui-button-icons-only" : "ui-button-icon-only"), this.hasTitle || b.attr("title", a.trim(c)))) : f.push("ui-button-text-only"), b.addClass(f.join(" "))
        }
    }), a.widget("ui.buttonset", {
        version: "1.10.3",
        options: {
            items: "button, input[type=button], input[type=submit], input[type=reset], input[type=checkbox], input[type=radio], a, :data(ui-button)"
        },
        _create: function() {
            this.element.addClass("ui-buttonset")
        },
        _init: function() {
            this.refresh()
        },
        _setOption: function(a, b) {
            "disabled" === a && this.buttons.button("option", a, b), this._super(a, b)
        },
        refresh: function() {
            var b = "rtl" === this.element.css("direction");
            this.buttons = this.element.find(this.options.items).filter(":ui-button").button("refresh").end().not(":ui-button").button().end().map(function() {
                return a(this).button("widget")[0]
            }).removeClass("ui-corner-all ui-corner-left ui-corner-right").filter(":first").addClass(b ? "ui-corner-right" : "ui-corner-left").end().filter(":last").addClass(b ? "ui-corner-left" : "ui-corner-right").end().end()
        },
        _destroy: function() {
            this.element.removeClass("ui-buttonset"), this.buttons.map(function() {
                return a(this).button("widget")[0]
            }).removeClass("ui-corner-left ui-corner-right").end().button("destroy")
        }
    })
}(jQuery), function(a, b) {
    function c() {
        this._curInst = null, this._keyEvent=!1, this._disabledInputs = [], this._datepickerShowing=!1, this._inDialog=!1, this._mainDivId = "ui-datepicker-div", this._inlineClass = "ui-datepicker-inline", this._appendClass = "ui-datepicker-append", this._triggerClass = "ui-datepicker-trigger", this._dialogClass = "ui-datepicker-dialog", this._disableClass = "ui-datepicker-disabled", this._unselectableClass = "ui-datepicker-unselectable", this._currentClass = "ui-datepicker-current-day", this._dayOverClass = "ui-datepicker-days-cell-over", this.regional = [], this.regional[""] = {
            closeText: "Done",
            prevText: "Prev",
            nextText: "Next",
            currentText: "Today",
            monthNames: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
            monthNamesShort: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            dayNames: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            dayNamesShort: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
            dayNamesMin: ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"],
            weekHeader: "Wk",
            dateFormat: "mm/dd/yy",
            firstDay: 0,
            isRTL: !1,
            showMonthAfterYear: !1,
            yearSuffix: ""
        }, this._defaults = {
            showOn: "focus",
            showAnim: "fadeIn",
            showOptions: {},
            defaultDate: null,
            appendText: "",
            buttonText: "...",
            buttonImage: "",
            buttonImageOnly: !1,
            hideIfNoPrevNext: !1,
            navigationAsDateFormat: !1,
            gotoCurrent: !1,
            changeMonth: !1,
            changeYear: !1,
            yearRange: "c-10:c+10",
            showOtherMonths: !1,
            selectOtherMonths: !1,
            showWeek: !1,
            calculateWeek: this.iso8601Week,
            shortYearCutoff: "+10",
            minDate: null,
            maxDate: null,
            duration: "fast",
            beforeShowDay: null,
            beforeShow: null,
            onSelect: null,
            onChangeMonthYear: null,
            onClose: null,
            numberOfMonths: 1,
            showCurrentAtPos: 0,
            stepMonths: 1,
            stepBigMonths: 12,
            altField: "",
            altFormat: "",
            constrainInput: !0,
            showButtonPanel: !1,
            autoSize: !1,
            disabled: !1
        }, a.extend(this._defaults, this.regional[""]), this.dpDiv = d(a("<div id='" + this._mainDivId + "' class='ui-datepicker ui-widget ui-widget-content ui-helper-clearfix ui-corner-all'></div>"))
    }
    function d(b) {
        var c = "button, .ui-datepicker-prev, .ui-datepicker-next, .ui-datepicker-calendar td a";
        return b.delegate(c, "mouseout", function() {
            a(this).removeClass("ui-state-hover"), - 1 !== this.className.indexOf("ui-datepicker-prev") && a(this).removeClass("ui-datepicker-prev-hover"), - 1 !== this.className.indexOf("ui-datepicker-next") && a(this).removeClass("ui-datepicker-next-hover")
        }).delegate(c, "mouseover", function() {
            a.datepicker._isDisabledDatepicker(f.inline ? b.parent()[0] : f.input[0]) || (a(this).parents(".ui-datepicker-calendar").find("a").removeClass("ui-state-hover"), a(this).addClass("ui-state-hover"), - 1 !== this.className.indexOf("ui-datepicker-prev") && a(this).addClass("ui-datepicker-prev-hover"), - 1 !== this.className.indexOf("ui-datepicker-next") && a(this).addClass("ui-datepicker-next-hover"))
        })
    }
    function e(b, c) {
        a.extend(b, c);
        for (var d in c)
            null == c[d] && (b[d] = c[d]);
        return b
    }
    a.extend(a.ui, {
        datepicker: {
            version: "1.10.3"
        }
    });
    var f, g = "datepicker";
    a.extend(c.prototype, {
        markerClassName: "hasDatepicker",
        maxRows: 4,
        _widgetDatepicker: function() {
            return this.dpDiv
        },
        setDefaults: function(a) {
            return e(this._defaults, a || {}), this
        },
        _attachDatepicker: function(b, c) {
            var d, e, f;
            d = b.nodeName.toLowerCase(), e = "div" === d || "span" === d, b.id || (this.uuid += 1, b.id = "dp" + this.uuid), f = this._newInst(a(b), e), f.settings = a.extend({}, c || {}), "input" === d ? this._connectDatepicker(b, f) : e && this._inlineDatepicker(b, f)
        },
        _newInst: function(b, c) {
            var e = b[0].id.replace(/([^A-Za-z0-9_\-])/g, "\\\\$1");
            return {
                id: e,
                input: b,
                selectedDay: 0,
                selectedMonth: 0,
                selectedYear: 0,
                drawMonth: 0,
                drawYear: 0,
                inline: c,
                dpDiv: c ? d(a("<div class='" + this._inlineClass + " ui-datepicker ui-widget ui-widget-content ui-helper-clearfix ui-corner-all'></div>")): this.dpDiv
            }
        },
        _connectDatepicker: function(b, c) {
            var d = a(b);
            c.append = a([]), c.trigger = a([]), d.hasClass(this.markerClassName) || (this._attachments(d, c), d.addClass(this.markerClassName).keydown(this._doKeyDown).keypress(this._doKeyPress).keyup(this._doKeyUp), this._autoSize(c), a.data(b, g, c), c.settings.disabled && this._disableDatepicker(b))
        },
        _attachments: function(b, c) {
            var d, e, f, g = this._get(c, "appendText"), h = this._get(c, "isRTL");
            c.append && c.append.remove(), g && (c.append = a("<span class='" + this._appendClass + "'>" + g + "</span>"), b[h ? "before": "after"](c.append)), b.unbind("focus", this._showDatepicker), c.trigger && c.trigger.remove(), d = this._get(c, "showOn"), ("focus" === d || "both" === d) && b.focus(this._showDatepicker), ("button" === d || "both" === d) && (e = this._get(c, "buttonText"), f = this._get(c, "buttonImage"), c.trigger = a(this._get(c, "buttonImageOnly") ? a("<img/>").addClass(this._triggerClass).attr({
                src: f,
                alt: e,
                title: e
            }) : a("<button type='button'></button>").addClass(this._triggerClass).html(f ? a("<img/>").attr({
                src: f,
                alt: e,
                title: e
            }) : e)), b[h ? "before": "after"](c.trigger), c.trigger.click(function() {
                return a.datepicker._datepickerShowing && a.datepicker._lastInput === b[0] ? a.datepicker._hideDatepicker() : a.datepicker._datepickerShowing && a.datepicker._lastInput !== b[0] ? (a.datepicker._hideDatepicker(), a.datepicker._showDatepicker(b[0])) : a.datepicker._showDatepicker(b[0]), !1
            }))
        },
        _autoSize: function(a) {
            if (this._get(a, "autoSize")&&!a.inline) {
                var b, c, d, e, f = new Date(2009, 11, 20), g = this._get(a, "dateFormat");
                g.match(/[DM]/) && (b = function(a) {
                    for (c = 0, d = 0, e = 0; a.length > e; e++)
                        a[e].length > c && (c = a[e].length, d = e);
                    return d
                }, f.setMonth(b(this._get(a, g.match(/MM/) ? "monthNames" : "monthNamesShort"))), f.setDate(b(this._get(a, g.match(/DD/) ? "dayNames" : "dayNamesShort")) + 20 - f.getDay())), a.input.attr("size", this._formatDate(a, f).length)
            }
        },
        _inlineDatepicker: function(b, c) {
            var d = a(b);
            d.hasClass(this.markerClassName) || (d.addClass(this.markerClassName).append(c.dpDiv), a.data(b, g, c), this._setDate(c, this._getDefaultDate(c), !0), this._updateDatepicker(c), this._updateAlternate(c), c.settings.disabled && this._disableDatepicker(b), c.dpDiv.css("display", "block"))
        },
        _dialogDatepicker: function(b, c, d, f, h) {
            var i, j, k, l, m, n = this._dialogInst;
            return n || (this.uuid += 1, i = "dp" + this.uuid, this._dialogInput = a("<input type='text' id='" + i + "' style='position: absolute; top: -100px; width: 0px;'/>"), this._dialogInput.keydown(this._doKeyDown), a("body").append(this._dialogInput), n = this._dialogInst = this._newInst(this._dialogInput, !1), n.settings = {}, a.data(this._dialogInput[0], g, n)), e(n.settings, f || {}), c = c && c.constructor === Date ? this._formatDate(n, c) : c, this._dialogInput.val(c), this._pos = h ? h.length ? h : [h.pageX, h.pageY] : null, this._pos || (j = document.documentElement.clientWidth, k = document.documentElement.clientHeight, l = document.documentElement.scrollLeft || document.body.scrollLeft, m = document.documentElement.scrollTop || document.body.scrollTop, this._pos = [j / 2 - 100 + l, k / 2 - 150 + m]), this._dialogInput.css("left", this._pos[0] + 20 + "px").css("top", this._pos[1] + "px"), n.settings.onSelect = d, this._inDialog=!0, this.dpDiv.addClass(this._dialogClass), this._showDatepicker(this._dialogInput[0]), a.blockUI && a.blockUI(this.dpDiv), a.data(this._dialogInput[0], g, n), this
        },
        _destroyDatepicker: function(b) {
            var c, d = a(b), e = a.data(b, g);
            d.hasClass(this.markerClassName) && (c = b.nodeName.toLowerCase(), a.removeData(b, g), "input" === c ? (e.append.remove(), e.trigger.remove(), d.removeClass(this.markerClassName).unbind("focus", this._showDatepicker).unbind("keydown", this._doKeyDown).unbind("keypress", this._doKeyPress).unbind("keyup", this._doKeyUp)) : ("div" === c || "span" === c) && d.removeClass(this.markerClassName).empty())
        },
        _enableDatepicker: function(b) {
            var c, d, e = a(b), f = a.data(b, g);
            e.hasClass(this.markerClassName) && (c = b.nodeName.toLowerCase(), "input" === c ? (b.disabled=!1, f.trigger.filter("button").each(function() {
                this.disabled=!1
            }).end().filter("img").css({
                opacity: "1.0",
                cursor: ""
            })) : ("div" === c || "span" === c) && (d = e.children("." + this._inlineClass), d.children().removeClass("ui-state-disabled"), d.find("select.ui-datepicker-month, select.ui-datepicker-year").prop("disabled", !1)), this._disabledInputs = a.map(this._disabledInputs, function(a) {
                return a === b ? null : a
            }))
        },
        _disableDatepicker: function(b) {
            var c, d, e = a(b), f = a.data(b, g);
            e.hasClass(this.markerClassName) && (c = b.nodeName.toLowerCase(), "input" === c ? (b.disabled=!0, f.trigger.filter("button").each(function() {
                this.disabled=!0
            }).end().filter("img").css({
                opacity: "0.5",
                cursor: "default"
            })) : ("div" === c || "span" === c) && (d = e.children("." + this._inlineClass), d.children().addClass("ui-state-disabled"), d.find("select.ui-datepicker-month, select.ui-datepicker-year").prop("disabled", !0)), this._disabledInputs = a.map(this._disabledInputs, function(a) {
                return a === b ? null : a
            }), this._disabledInputs[this._disabledInputs.length] = b)
        },
        _isDisabledDatepicker: function(a) {
            if (!a)
                return !1;
            for (var b = 0; this._disabledInputs.length > b; b++)
                if (this._disabledInputs[b] === a)
                    return !0;
            return !1
        },
        _getInst: function(b) {
            try {
                return a.data(b, g)
            } catch (c) {
                throw "Missing instance data for this datepicker"
            }
        },
        _optionDatepicker: function(c, d, f) {
            var g, h, i, j, k = this._getInst(c);
            return 2 === arguments.length && "string" == typeof d ? "defaults" === d ? a.extend({}, a.datepicker._defaults) : k ? "all" === d ? a.extend({}, k.settings) : this._get(k, d) : null : (g = d || {}, "string" == typeof d && (g = {}, g[d] = f), k && (this._curInst === k && this._hideDatepicker(), h = this._getDateDatepicker(c, !0), i = this._getMinMaxDate(k, "min"), j = this._getMinMaxDate(k, "max"), e(k.settings, g), null !== i && g.dateFormat !== b && g.minDate === b && (k.settings.minDate = this._formatDate(k, i)), null !== j && g.dateFormat !== b && g.maxDate === b && (k.settings.maxDate = this._formatDate(k, j)), "disabled"in g && (g.disabled ? this._disableDatepicker(c) : this._enableDatepicker(c)), this._attachments(a(c), k), this._autoSize(k), this._setDate(k, h), this._updateAlternate(k), this._updateDatepicker(k)), b)
        },
        _changeDatepicker: function(a, b, c) {
            this._optionDatepicker(a, b, c)
        },
        _refreshDatepicker: function(a) {
            var b = this._getInst(a);
            b && this._updateDatepicker(b)
        },
        _setDateDatepicker: function(a, b) {
            var c = this._getInst(a);
            c && (this._setDate(c, b), this._updateDatepicker(c), this._updateAlternate(c))
        },
        _getDateDatepicker: function(a, b) {
            var c = this._getInst(a);
            return c&&!c.inline && this._setDateFromField(c, b), c ? this._getDate(c) : null
        },
        _doKeyDown: function(b) {
            var c, d, e, f = a.datepicker._getInst(b.target), g=!0, h = f.dpDiv.is(".ui-datepicker-rtl");
            if (f._keyEvent=!0, a.datepicker._datepickerShowing)
                switch (b.keyCode) {
                case 9:
                    a.datepicker._hideDatepicker(), g=!1;
                    break;
                case 13:
                    return e = a("td." + a.datepicker._dayOverClass + ":not(." + a.datepicker._currentClass + ")", f.dpDiv), e[0] && a.datepicker._selectDay(b.target, f.selectedMonth, f.selectedYear, e[0]), c = a.datepicker._get(f, "onSelect"), c ? (d = a.datepicker._formatDate(f), c.apply(f.input ? f.input[0] : null, [d, f])) : a.datepicker._hideDatepicker(), !1;
                case 27:
                    a.datepicker._hideDatepicker();
                    break;
                case 33:
                    a.datepicker._adjustDate(b.target, b.ctrlKey?-a.datepicker._get(f, "stepBigMonths") : - a.datepicker._get(f, "stepMonths"), "M");
                    break;
                case 34:
                    a.datepicker._adjustDate(b.target, b.ctrlKey?+a.datepicker._get(f, "stepBigMonths") : + a.datepicker._get(f, "stepMonths"), "M");
                    break;
                case 35:
                    (b.ctrlKey || b.metaKey) && a.datepicker._clearDate(b.target), g = b.ctrlKey || b.metaKey;
                    break;
                case 36:
                    (b.ctrlKey || b.metaKey) && a.datepicker._gotoToday(b.target), g = b.ctrlKey || b.metaKey;
                    break;
                case 37:
                    (b.ctrlKey || b.metaKey) && a.datepicker._adjustDate(b.target, h ? 1 : - 1, "D"), g = b.ctrlKey || b.metaKey, b.originalEvent.altKey && a.datepicker._adjustDate(b.target, b.ctrlKey?-a.datepicker._get(f, "stepBigMonths") : - a.datepicker._get(f, "stepMonths"), "M");
                    break;
                case 38:
                    (b.ctrlKey || b.metaKey) && a.datepicker._adjustDate(b.target, - 7, "D"), g = b.ctrlKey || b.metaKey;
                    break;
                case 39:
                    (b.ctrlKey || b.metaKey) && a.datepicker._adjustDate(b.target, h?-1 : 1, "D"), g = b.ctrlKey || b.metaKey, b.originalEvent.altKey && a.datepicker._adjustDate(b.target, b.ctrlKey?+a.datepicker._get(f, "stepBigMonths") : + a.datepicker._get(f, "stepMonths"), "M");
                    break;
                case 40:
                    (b.ctrlKey || b.metaKey) && a.datepicker._adjustDate(b.target, 7, "D"), g = b.ctrlKey || b.metaKey;
                    break;
                default:
                    g=!1
                } else 
                    36 === b.keyCode && b.ctrlKey ? a.datepicker._showDatepicker(this) : g=!1;
            g && (b.preventDefault(), b.stopPropagation())
        },
        _doKeyPress: function(c) {
            var d, e, f = a.datepicker._getInst(c.target);
            return a.datepicker._get(f, "constrainInput") ? (d = a.datepicker._possibleChars(a.datepicker._get(f, "dateFormat")), e = String.fromCharCode(null == c.charCode ? c.keyCode : c.charCode), c.ctrlKey || c.metaKey || " " > e ||!d || d.indexOf(e)>-1) : b
        },
        _doKeyUp: function(b) {
            var c, d = a.datepicker._getInst(b.target);
            if (d.input.val() !== d.lastVal)
                try {
                    c = a.datepicker.parseDate(a.datepicker._get(d, "dateFormat"), d.input ? d.input.val() : null, a.datepicker._getFormatConfig(d)), c && (a.datepicker._setDateFromField(d), a.datepicker._updateAlternate(d), a.datepicker._updateDatepicker(d))
            } catch (e) {}
            return !0
        },
        _showDatepicker: function(b) {
            if (b = b.target || b, "input" !== b.nodeName.toLowerCase() && (b = a("input", b.parentNode)[0]), !a.datepicker._isDisabledDatepicker(b) && a.datepicker._lastInput !== b) {
                var c, d, f, g, h, i, j;
                c = a.datepicker._getInst(b), a.datepicker._curInst && a.datepicker._curInst !== c && (a.datepicker._curInst.dpDiv.stop(!0, !0), c && a.datepicker._datepickerShowing && a.datepicker._hideDatepicker(a.datepicker._curInst.input[0])), d = a.datepicker._get(c, "beforeShow"), f = d ? d.apply(b, [b, c]) : {}, f!==!1 && (e(c.settings, f), c.lastVal = null, a.datepicker._lastInput = b, a.datepicker._setDateFromField(c), a.datepicker._inDialog && (b.value = ""), a.datepicker._pos || (a.datepicker._pos = a.datepicker._findPos(b), a.datepicker._pos[1] += b.offsetHeight), g=!1, a(b).parents().each(function() {
                    return g|="fixed" === a(this).css("position"), !g
                }), h = {
                    left: a.datepicker._pos[0],
                    top: a.datepicker._pos[1]
                }, a.datepicker._pos = null, c.dpDiv.empty(), c.dpDiv.css({
                    position: "absolute",
                    display: "block",
                    top: "-1000px"
                }), a.datepicker._updateDatepicker(c), h = a.datepicker._checkOffset(c, h, g), c.dpDiv.css({
                    position: a.datepicker._inDialog && a.blockUI ? "static": g ? "fixed": "absolute",
                    display: "none",
                    left: h.left + "px",
                    top: h.top + "px"
                }), c.inline || (i = a.datepicker._get(c, "showAnim"), j = a.datepicker._get(c, "duration"), c.dpDiv.zIndex(a(b).zIndex() + 1), a.datepicker._datepickerShowing=!0, a.effects && a.effects.effect[i] ? c.dpDiv.show(i, a.datepicker._get(c, "showOptions"), j) : c.dpDiv[i || "show"](i ? j : null), a.datepicker._shouldFocusInput(c) && c.input.focus(), a.datepicker._curInst = c))
            }
        },
        _updateDatepicker: function(b) {
            this.maxRows = 4, f = b, b.dpDiv.empty().append(this._generateHTML(b)), this._attachHandlers(b), b.dpDiv.find("." + this._dayOverClass + " a").mouseover();
            var c, d = this._getNumberOfMonths(b), e = d[1], g = 17;
            b.dpDiv.removeClass("ui-datepicker-multi-2 ui-datepicker-multi-3 ui-datepicker-multi-4").width(""), e > 1 && b.dpDiv.addClass("ui-datepicker-multi-" + e).css("width", g * e + "em"), b.dpDiv[(1 !== d[0] || 1 !== d[1] ? "add" : "remove") + "Class"]("ui-datepicker-multi"), b.dpDiv[(this._get(b, "isRTL") ? "add" : "remove") + "Class"]("ui-datepicker-rtl"), b === a.datepicker._curInst && a.datepicker._datepickerShowing && a.datepicker._shouldFocusInput(b) && b.input.focus(), b.yearshtml && (c = b.yearshtml, setTimeout(function() {
                c === b.yearshtml && b.yearshtml && b.dpDiv.find("select.ui-datepicker-year:first").replaceWith(b.yearshtml), c = b.yearshtml = null
            }, 0))
        },
        _shouldFocusInput: function(a) {
            return a.input && a.input.is(":visible")&&!a.input.is(":disabled")&&!a.input.is(":focus")
        },
        _checkOffset: function(b, c, d) {
            var e = b.dpDiv.outerWidth(), f = b.dpDiv.outerHeight(), g = b.input ? b.input.outerWidth(): 0, h = b.input ? b.input.outerHeight(): 0, i = document.documentElement.clientWidth + (d ? 0 : a(document).scrollLeft()), j = document.documentElement.clientHeight + (d ? 0 : a(document).scrollTop());
            return c.left -= this._get(b, "isRTL") ? e - g : 0, c.left -= d && c.left === b.input.offset().left ? a(document).scrollLeft() : 0, c.top -= d && c.top === b.input.offset().top + h ? a(document).scrollTop() : 0, c.left -= Math.min(c.left, c.left + e > i && i > e ? Math.abs(c.left + e - i) : 0), c.top -= Math.min(c.top, c.top + f > j && j > f ? Math.abs(f + h) : 0), c
        },
        _findPos: function(b) {
            for (var c, d = this._getInst(b), e = this._get(d, "isRTL"); b && ("hidden" === b.type || 1 !== b.nodeType || a.expr.filters.hidden(b));)
                b = b[e ? "previousSibling": "nextSibling"];
            return c = a(b).offset(), [c.left, c.top]
        },
        _hideDatepicker: function(b) {
            var c, d, e, f, h = this._curInst;
            !h || b && h !== a.data(b, g) || this._datepickerShowing && (c = this._get(h, "showAnim"), d = this._get(h, "duration"), e = function() {
                a.datepicker._tidyDialog(h)
            }, a.effects && (a.effects.effect[c] || a.effects[c]) ? h.dpDiv.hide(c, a.datepicker._get(h, "showOptions"), d, e) : h.dpDiv["slideDown" === c ? "slideUp": "fadeIn" === c ? "fadeOut": "hide"](c ? d : null, e), c || e(), this._datepickerShowing=!1, f = this._get(h, "onClose"), f && f.apply(h.input ? h.input[0] : null, [h.input ? h.input.val(): "", h]), this._lastInput = null, this._inDialog && (this._dialogInput.css({
                position: "absolute",
                left: "0",
                top: "-100px"
            }), a.blockUI && (a.unblockUI(), a("body").append(this.dpDiv))), this._inDialog=!1)
        },
        _tidyDialog: function(a) {
            a.dpDiv.removeClass(this._dialogClass).unbind(".ui-datepicker-calendar")
        },
        _checkExternalClick: function(b) {
            if (a.datepicker._curInst) {
                var c = a(b.target), d = a.datepicker._getInst(c[0]);
                (c[0].id !== a.datepicker._mainDivId && 0 === c.parents("#" + a.datepicker._mainDivId).length&&!c.hasClass(a.datepicker.markerClassName)&&!c.closest("." + a.datepicker._triggerClass).length && a.datepicker._datepickerShowing && (!a.datepicker._inDialog ||!a.blockUI) || c.hasClass(a.datepicker.markerClassName) && a.datepicker._curInst !== d) && a.datepicker._hideDatepicker()
            }
        },
        _adjustDate: function(b, c, d) {
            var e = a(b), f = this._getInst(e[0]);
            this._isDisabledDatepicker(e[0]) || (this._adjustInstDate(f, c + ("M" === d ? this._get(f, "showCurrentAtPos") : 0), d), this._updateDatepicker(f))
        },
        _gotoToday: function(b) {
            var c, d = a(b), e = this._getInst(d[0]);
            this._get(e, "gotoCurrent") && e.currentDay ? (e.selectedDay = e.currentDay, e.drawMonth = e.selectedMonth = e.currentMonth, e.drawYear = e.selectedYear = e.currentYear) : (c = new Date, e.selectedDay = c.getDate(), e.drawMonth = e.selectedMonth = c.getMonth(), e.drawYear = e.selectedYear = c.getFullYear()), this._notifyChange(e), this._adjustDate(d)
        },
        _selectMonthYear: function(b, c, d) {
            var e = a(b), f = this._getInst(e[0]);
            f["selected" + ("M" === d ? "Month" : "Year")] = f["draw" + ("M" === d ? "Month" : "Year")] = parseInt(c.options[c.selectedIndex].value, 10), this._notifyChange(f), this._adjustDate(e)
        },
        _selectDay: function(b, c, d, e) {
            var f, g = a(b);
            a(e).hasClass(this._unselectableClass) || this._isDisabledDatepicker(g[0]) || (f = this._getInst(g[0]), f.selectedDay = f.currentDay = a("a", e).html(), f.selectedMonth = f.currentMonth = c, f.selectedYear = f.currentYear = d, this._selectDate(b, this._formatDate(f, f.currentDay, f.currentMonth, f.currentYear)))
        },
        _clearDate: function(b) {
            var c = a(b);
            this._selectDate(c, "")
        },
        _selectDate: function(b, c) {
            var d, e = a(b), f = this._getInst(e[0]);
            c = null != c ? c : this._formatDate(f), f.input && f.input.val(c), this._updateAlternate(f), d = this._get(f, "onSelect"), d ? d.apply(f.input ? f.input[0] : null, [c, f]) : f.input && f.input.trigger("change"), f.inline ? this._updateDatepicker(f) : (this._hideDatepicker(), this._lastInput = f.input[0], "object" != typeof f.input[0] && f.input.focus(), this._lastInput = null)
        },
        _updateAlternate: function(b) {
            var c, d, e, f = this._get(b, "altField");
            f && (c = this._get(b, "altFormat") || this._get(b, "dateFormat"), d = this._getDate(b), e = this.formatDate(c, d, this._getFormatConfig(b)), a(f).each(function() {
                a(this).val(e)
            }))
        },
        noWeekends: function(a) {
            var b = a.getDay();
            return [b > 0 && 6 > b, ""]
        },
        iso8601Week: function(a) {
            var b, c = new Date(a.getTime());
            return c.setDate(c.getDate() + 4 - (c.getDay() || 7)), b = c.getTime(), c.setMonth(0), c.setDate(1), Math.floor(Math.round((b - c) / 864e5) / 7) + 1
        },
        parseDate: function(c, d, e) {
            if (null == c || null == d)
                throw "Invalid arguments";
            if (d = "object" == typeof d ? "" + d : d + "", "" === d)
                return null;
            var f, g, h, i, j = 0, k = (e ? e.shortYearCutoff : null) || this._defaults.shortYearCutoff, l = "string" != typeof k ? k: (new Date).getFullYear()%100 + parseInt(k, 10), m = (e ? e.dayNamesShort : null) || this._defaults.dayNamesShort, n = (e ? e.dayNames : null) || this._defaults.dayNames, o = (e ? e.monthNamesShort : null) || this._defaults.monthNamesShort, p = (e ? e.monthNames : null) || this._defaults.monthNames, q =- 1, r =- 1, s =- 1, t =- 1, u=!1, v = function(a) {
                var b = c.length > f + 1 && c.charAt(f + 1) === a;
                return b && f++, b
            }, w = function(a) {
                var b = v(a), c = "@" === a ? 14: "!" === a ? 20: "y" === a && b ? 4: "o" === a ? 3: 2, e = RegExp("^\\d{1," + c + "}"), f = d.substring(j).match(e);
                if (!f)
                    throw "Missing number at position " + j;
                return j += f[0].length, parseInt(f[0], 10)
            }, x = function(c, e, f) {
                var g =- 1, h = a.map(v(c) ? f : e, function(a, b) {
                    return [[b, a]]
                }).sort(function(a, b) {
                    return - (a[1].length - b[1].length)
                });
                if (a.each(h, function(a, c) {
                    var e = c[1];
                    return d.substr(j, e.length).toLowerCase() === e.toLowerCase() ? (g = c[0], j += e.length, !1) : b
                }), - 1 !== g)
                    return g + 1;
                throw "Unknown name at position " + j
            }, y = function() {
                if (d.charAt(j) !== c.charAt(f))
                    throw "Unexpected literal at position " + j;
                j++
            };
            for (f = 0; c.length > f; f++)
                if (u)
                    "'" !== c.charAt(f) || v("'") ? y() : u=!1;
                else 
                    switch (c.charAt(f)) {
                    case"d":
                        s = w("d");
                        break;
                    case"D":
                        x("D", m, n);
                        break;
                    case"o":
                        t = w("o");
                        break;
                    case"m":
                        r = w("m");
                        break;
                    case"M":
                        r = x("M", o, p);
                        break;
                    case"y":
                        q = w("y");
                        break;
                    case"@":
                        i = new Date(w("@")), q = i.getFullYear(), r = i.getMonth() + 1, s = i.getDate();
                        break;
                    case"!":
                        i = new Date((w("!") - this._ticksTo1970) / 1e4), q = i.getFullYear(), r = i.getMonth() + 1, s = i.getDate();
                        break;
                    case"'":
                        v("'") ? y() : u=!0;
                        break;
                    default:
                        y()
                    }
            if (d.length > j && (h = d.substr(j), !/^\s+/.test(h)))
                throw "Extra/unparsed characters found in date: " + h;
            if ( - 1 === q ? q = (new Date).getFullYear() : 100 > q && (q += (new Date).getFullYear() - (new Date).getFullYear()%100 + (l >= q ? 0 : - 100)), t>-1)
                for (r = 1, s = t; g = this._getDaysInMonth(q, r - 1), !(g >= s);)
                    r++, s -= g;
            if (i = this._daylightSavingAdjust(new Date(q, r - 1, s)), i.getFullYear() !== q || i.getMonth() + 1 !== r || i.getDate() !== s)
                throw "Invalid date";
            return i
        },
        ATOM: "yy-mm-dd",
        COOKIE: "D, dd M yy",
        ISO_8601: "yy-mm-dd",
        RFC_822: "D, d M y",
        RFC_850: "DD, dd-M-y",
        RFC_1036: "D, d M y",
        RFC_1123: "D, d M yy",
        RFC_2822: "D, d M yy",
        RSS: "D, d M y",
        TICKS: "!",
        TIMESTAMP: "@",
        W3C: "yy-mm-dd",
        _ticksTo1970: 864e9 * (718685 + Math.floor(492.5) - Math.floor(19.7) + Math.floor(4.925)),
        formatDate: function(a, b, c) {
            if (!b)
                return "";
            var d, e = (c ? c.dayNamesShort : null) || this._defaults.dayNamesShort, f = (c ? c.dayNames : null) || this._defaults.dayNames, g = (c ? c.monthNamesShort : null) || this._defaults.monthNamesShort, h = (c ? c.monthNames : null) || this._defaults.monthNames, i = function(b) {
                var c = a.length > d + 1 && a.charAt(d + 1) === b;
                return c && d++, c
            }, j = function(a, b, c) {
                var d = "" + b;
                if (i(a))
                    for (; c > d.length;)
                        d = "0" + d;
                return d
            }, k = function(a, b, c, d) {
                return i(a) ? d[b] : c[b]
            }, l = "", m=!1;
            if (b)
                for (d = 0; a.length > d; d++)
                    if (m)
                        "'" !== a.charAt(d) || i("'") ? l += a.charAt(d) : m=!1;
                    else 
                        switch (a.charAt(d)) {
                        case"d":
                            l += j("d", b.getDate(), 2);
                            break;
                        case"D":
                            l += k("D", b.getDay(), e, f);
                            break;
                        case"o":
                            l += j("o", Math.round((new Date(b.getFullYear(), b.getMonth(), b.getDate()).getTime() - new Date(b.getFullYear(), 0, 0).getTime()) / 864e5), 3);
                            break;
                        case"m":
                            l += j("m", b.getMonth() + 1, 2);
                            break;
                        case"M":
                            l += k("M", b.getMonth(), g, h);
                            break;
                        case"y":
                            l += i("y") ? b.getFullYear() : (10 > b.getYear()%100 ? "0" : "") + b.getYear()%100;
                            break;
                        case"@":
                            l += b.getTime();
                            break;
                        case"!":
                            l += 1e4 * b.getTime() + this._ticksTo1970;
                            break;
                        case"'":
                            i("'") ? l += "'" : m=!0;
                            break;
                        default:
                            l += a.charAt(d)
                        }
            return l
        },
        _possibleChars: function(a) {
            var b, c = "", d=!1, e = function(c) {
                var d = a.length > b + 1 && a.charAt(b + 1) === c;
                return d && b++, d
            };
            for (b = 0; a.length > b; b++)
                if (d)
                    "'" !== a.charAt(b) || e("'") ? c += a.charAt(b) : d=!1;
                else 
                    switch (a.charAt(b)) {
                    case"d":
                    case"m":
                    case"y":
                    case"@":
                        c += "0123456789";
                        break;
                    case"D":
                    case"M":
                        return null;
                    case"'":
                        e("'") ? c += "'" : d=!0;
                        break;
                    default:
                        c += a.charAt(b)
                    }
            return c
        },
        _get: function(a, c) {
            return a.settings[c] !== b ? a.settings[c] : this._defaults[c]
        },
        _setDateFromField: function(a, b) {
            if (a.input.val() !== a.lastVal) {
                var c = this._get(a, "dateFormat"), d = a.lastVal = a.input ? a.input.val(): null, e = this._getDefaultDate(a), f = e, g = this._getFormatConfig(a);
                try {
                    f = this.parseDate(c, d, g) || e
                } catch (h) {
                    d = b ? "" : d
                }
                a.selectedDay = f.getDate(), a.drawMonth = a.selectedMonth = f.getMonth(), a.drawYear = a.selectedYear = f.getFullYear(), a.currentDay = d ? f.getDate() : 0, a.currentMonth = d ? f.getMonth() : 0, a.currentYear = d ? f.getFullYear() : 0, this._adjustInstDate(a)
            }
        },
        _getDefaultDate: function(a) {
            return this._restrictMinMax(a, this._determineDate(a, this._get(a, "defaultDate"), new Date))
        },
        _determineDate: function(b, c, d) {
            var e = function(a) {
                var b = new Date;
                return b.setDate(b.getDate() + a), b
            }, f = function(c) {
                try {
                    return a.datepicker.parseDate(a.datepicker._get(b, "dateFormat"), c, a.datepicker._getFormatConfig(b))
                } catch (d) {}
                for (var e = (c.toLowerCase().match(/^c/) ? a.datepicker._getDate(b) : null) || new Date, f = e.getFullYear(), g = e.getMonth(), h = e.getDate(), i = /([+\-]?[0-9]+)\s*(d|D|w|W|m|M|y|Y)?/g, j = i.exec(c); j;) {
                    switch (j[2] || "d") {
                    case"d":
                    case"D":
                        h += parseInt(j[1], 10);
                        break;
                    case"w":
                    case"W":
                        h += 7 * parseInt(j[1], 10);
                        break;
                    case"m":
                    case"M":
                        g += parseInt(j[1], 10), h = Math.min(h, a.datepicker._getDaysInMonth(f, g));
                        break;
                    case"y":
                    case"Y":
                        f += parseInt(j[1], 10), h = Math.min(h, a.datepicker._getDaysInMonth(f, g))
                    }
                    j = i.exec(c)
                }
                return new Date(f, g, h)
            }, g = null == c || "" === c ? d: "string" == typeof c ? f(c): "number" == typeof c ? isNaN(c) ? d: e(c): new Date(c.getTime());
            return g = g && "Invalid Date" == "" + g ? d : g, g && (g.setHours(0), g.setMinutes(0), g.setSeconds(0), g.setMilliseconds(0)), this._daylightSavingAdjust(g)
        },
        _daylightSavingAdjust: function(a) {
            return a ? (a.setHours(a.getHours() > 12 ? a.getHours() + 2 : 0), a) : null
        },
        _setDate: function(a, b, c) {
            var d=!b, e = a.selectedMonth, f = a.selectedYear, g = this._restrictMinMax(a, this._determineDate(a, b, new Date));
            a.selectedDay = a.currentDay = g.getDate(), a.drawMonth = a.selectedMonth = a.currentMonth = g.getMonth(), a.drawYear = a.selectedYear = a.currentYear = g.getFullYear(), e === a.selectedMonth && f === a.selectedYear || c || this._notifyChange(a), this._adjustInstDate(a), a.input && a.input.val(d ? "" : this._formatDate(a))
        },
        _getDate: function(a) {
            var b=!a.currentYear || a.input && "" === a.input.val() ? null : this._daylightSavingAdjust(new Date(a.currentYear, a.currentMonth, a.currentDay));
            return b
        },
        _attachHandlers: function(b) {
            var c = this._get(b, "stepMonths"), d = "#" + b.id.replace(/\\\\/g, "\\");
            b.dpDiv.find("[data-handler]").map(function() {
                var b = {
                    prev: function() {
                        a.datepicker._adjustDate(d, - c, "M")
                    },
                    next: function() {
                        a.datepicker._adjustDate(d, + c, "M")
                    },
                    hide: function() {
                        a.datepicker._hideDatepicker()
                    },
                    today: function() {
                        a.datepicker._gotoToday(d)
                    },
                    selectDay: function() {
                        return a.datepicker._selectDay(d, + this.getAttribute("data-month"), + this.getAttribute("data-year"), this), !1
                    },
                    selectMonth: function() {
                        return a.datepicker._selectMonthYear(d, this, "M"), !1
                    },
                    selectYear: function() {
                        return a.datepicker._selectMonthYear(d, this, "Y"), !1
                    }
                };
                a(this).bind(this.getAttribute("data-event"), b[this.getAttribute("data-handler")])
            })
        },
        _generateHTML: function(a) {
            var b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O = new Date, P = this._daylightSavingAdjust(new Date(O.getFullYear(), O.getMonth(), O.getDate())), Q = this._get(a, "isRTL"), R = this._get(a, "showButtonPanel"), S = this._get(a, "hideIfNoPrevNext"), T = this._get(a, "navigationAsDateFormat"), U = this._getNumberOfMonths(a), V = this._get(a, "showCurrentAtPos"), W = this._get(a, "stepMonths"), X = 1 !== U[0] || 1 !== U[1], Y = this._daylightSavingAdjust(a.currentDay ? new Date(a.currentYear, a.currentMonth, a.currentDay) : new Date(9999, 9, 9)), Z = this._getMinMaxDate(a, "min"), $ = this._getMinMaxDate(a, "max"), _ = a.drawMonth - V, ab = a.drawYear;
            if (0 > _ && (_ += 12, ab--), $)
                for (b = this._daylightSavingAdjust(new Date($.getFullYear(), $.getMonth() - U[0] * U[1] + 1, $.getDate())), b = Z && Z > b ? Z : b; this._daylightSavingAdjust(new Date(ab, _, 1)) > b;)
                    _--, 0 > _ && (_ = 11, ab--);
            for (a.drawMonth = _, a.drawYear = ab, c = this._get(a, "prevText"), c = T ? this.formatDate(c, this._daylightSavingAdjust(new Date(ab, _ - W, 1)), this._getFormatConfig(a)) : c, d = this._canAdjustMonth(a, - 1, ab, _) ? "<a class='ui-datepicker-prev ui-corner-all' data-handler='prev' data-event='click' title='" + c + "'><span class='ui-icon ui-icon-circle-triangle-" + (Q ? "e" : "w") + "'>" + c + "</span></a>" : S ? "" : "<a class='ui-datepicker-prev ui-corner-all ui-state-disabled' title='" + c + "'><span class='ui-icon ui-icon-circle-triangle-" + (Q ? "e" : "w") + "'>" + c + "</span></a>", e = this._get(a, "nextText"), e = T ? this.formatDate(e, this._daylightSavingAdjust(new Date(ab, _ + W, 1)), this._getFormatConfig(a)) : e, f = this._canAdjustMonth(a, 1, ab, _) ? "<a class='ui-datepicker-next ui-corner-all' data-handler='next' data-event='click' title='" + e + "'><span class='ui-icon ui-icon-circle-triangle-" + (Q ? "w" : "e") + "'>" + e + "</span></a>" : S ? "" : "<a class='ui-datepicker-next ui-corner-all ui-state-disabled' title='" + e + "'><span class='ui-icon ui-icon-circle-triangle-" + (Q ? "w" : "e") + "'>" + e + "</span></a>", g = this._get(a, "currentText"), h = this._get(a, "gotoCurrent") && a.currentDay ? Y : P, g = T ? this.formatDate(g, h, this._getFormatConfig(a)) : g, i = a.inline ? "" : "<button type='button' class='ui-datepicker-close ui-state-default ui-priority-primary ui-corner-all' data-handler='hide' data-event='click'>" + this._get(a, "closeText") + "</button>", j = R ? "<div class='ui-datepicker-buttonpane ui-widget-content'>" + (Q ? i : "") + (this._isInRange(a, h) ? "<button type='button' class='ui-datepicker-current ui-state-default ui-priority-secondary ui-corner-all' data-handler='today' data-event='click'>" + g + "</button>" : "") + (Q ? "" : i) + "</div>" : "", k = parseInt(this._get(a, "firstDay"), 10), k = isNaN(k) ? 0 : k, l = this._get(a, "showWeek"), m = this._get(a, "dayNames"), n = this._get(a, "dayNamesMin"), o = this._get(a, "monthNames"), p = this._get(a, "monthNamesShort"), q = this._get(a, "beforeShowDay"), r = this._get(a, "showOtherMonths"), s = this._get(a, "selectOtherMonths"), t = this._getDefaultDate(a), u = "", w = 0; U[0] > w; w++) {
                for (x = "", this.maxRows = 4, y = 0; U[1] > y; y++) {
                    if (z = this._daylightSavingAdjust(new Date(ab, _, a.selectedDay)), A = " ui-corner-all", B = "", X) {
                        if (B += "<div class='ui-datepicker-group", U[1] > 1)
                            switch (y) {
                            case 0:
                                B += " ui-datepicker-group-first", A = " ui-corner-" + (Q ? "right" : "left");
                                break;
                            case U[1] - 1:
                                B += " ui-datepicker-group-last", A = " ui-corner-" + (Q ? "left" : "right");
                                break;
                            default:
                                B += " ui-datepicker-group-middle", A = ""
                            }
                        B += "'>"
                    }
                    for (B += "<div class='ui-datepicker-header ui-widget-header ui-helper-clearfix" + A + "'>" + (/all|left/.test(A) && 0 === w ? Q ? f : d : "") + (/all|right/.test(A) && 0 === w ? Q ? d : f : "") + this._generateMonthYearHeader(a, _, ab, Z, $, w > 0 || y > 0, o, p) + "</div><table class='ui-datepicker-calendar'><thead><tr>", C = l ? "<th class='ui-datepicker-week-col'>" + this._get(a, "weekHeader") + "</th>" : "", v = 0; 7 > v; v++)
                        D = (v + k)%7, C += "<th" + ((v + k + 6)%7 >= 5 ? " class='ui-datepicker-week-end'" : "") + "><span title='" + m[D] + "'>" + n[D] + "</span></th>";
                    for (B += C + "</tr></thead><tbody>", E = this._getDaysInMonth(ab, _), ab === a.selectedYear && _ === a.selectedMonth && (a.selectedDay = Math.min(a.selectedDay, E)), F = (this._getFirstDayOfMonth(ab, _) - k + 7)%7, G = Math.ceil((F + E) / 7), H = X && this.maxRows > G ? this.maxRows : G, this.maxRows = H, I = this._daylightSavingAdjust(new Date(ab, _, 1 - F)), J = 0; H > J; J++) {
                        for (B += "<tr>", K = l ? "<td class='ui-datepicker-week-col'>" + this._get(a, "calculateWeek")(I) + "</td>" : "", v = 0; 7 > v; v++)
                            L = q ? q.apply(a.input ? a.input[0] : null, [I]) : [!0, ""], M = I.getMonth() !== _, N = M&&!s ||!L[0] || Z && Z > I || $ && I > $, K += "<td class='" + ((v + k + 6)%7 >= 5 ? " ui-datepicker-week-end" : "") + (M ? " ui-datepicker-other-month" : "") + (I.getTime() === z.getTime() && _ === a.selectedMonth && a._keyEvent || t.getTime() === I.getTime() && t.getTime() === z.getTime() ? " " + this._dayOverClass : "") + (N ? " " + this._unselectableClass + " ui-state-disabled" : "") + (M&&!r ? "" : " " + L[1] + (I.getTime() === Y.getTime() ? " " + this._currentClass : "") + (I.getTime() === P.getTime() ? " ui-datepicker-today" : "")) + "'" + (M&&!r ||!L[2] ? "" : " title='" + L[2].replace(/'/g, "&#39;") + "'") + (N ? "" : " data-handler='selectDay' data-event='click' data-month='" + I.getMonth() + "' data-year='" + I.getFullYear() + "'") + ">" + (M&&!r ? "&#xa0;" : N ? "<span class='ui-state-default'>" + I.getDate() + "</span>" : "<a class='ui-state-default" + (I.getTime() === P.getTime() ? " ui-state-highlight" : "") + (I.getTime() === Y.getTime() ? " ui-state-active" : "") + (M ? " ui-priority-secondary" : "") + "' href='#'>" + I.getDate() + "</a>") + "</td>", I.setDate(I.getDate() + 1), I = this._daylightSavingAdjust(I);
                        B += K + "</tr>"
                    }
                    _++, _ > 11 && (_ = 0, ab++), B += "</tbody></table>" + (X ? "</div>" + (U[0] > 0 && y === U[1] - 1 ? "<div class='ui-datepicker-row-break'></div>" : "") : ""), x += B
                }
                u += x
            }
            return u += j, a._keyEvent=!1, u
        },
        _generateMonthYearHeader: function(a, b, c, d, e, f, g, h) {
            var i, j, k, l, m, n, o, p, q = this._get(a, "changeMonth"), r = this._get(a, "changeYear"), s = this._get(a, "showMonthAfterYear"), t = "<div class='ui-datepicker-title'>", u = "";
            if (f ||!q)
                u += "<span class='ui-datepicker-month'>" + g[b] + "</span>";
            else {
                for (i = d && d.getFullYear() === c, j = e && e.getFullYear() === c, u += "<select class='ui-datepicker-month' data-handler='selectMonth' data-event='change'>", k = 0; 12 > k; k++)(!i || k >= d.getMonth()
                    ) && (!j || e.getMonth() >= k) && (u += "<option value='" + k + "'" + (k === b ? " selected='selected'" : "") + ">" + h[k] + "</option>");
                u += "</select>"
            }
            if (s || (t += u + (!f && q && r ? "" : "&#xa0;")), !a.yearshtml)
                if (a.yearshtml = "", f ||!r)
                    t += "<span class='ui-datepicker-year'>" + c + "</span>";
                else {
                    for (l = this._get(a, "yearRange").split(":"), m = (new Date).getFullYear(), n = function(a) {
                        var b = a.match(/c[+\-].*/) ? c + parseInt(a.substring(1), 10): a.match(/[+\-].*/) ? m + parseInt(a, 10): parseInt(a, 10);
                        return isNaN(b) ? m : b
                    }, o = n(l[0]), p = Math.max(o, n(l[1] || "")), o = d ? Math.max(o, d.getFullYear()) : o, p = e ? Math.min(p, e.getFullYear()) : p, a.yearshtml += "<select class='ui-datepicker-year' data-handler='selectYear' data-event='change'>"; p >= o; o++)
                        a.yearshtml += "<option value='" + o + "'" + (o === c ? " selected='selected'" : "") + ">" + o + "</option>";
                        a.yearshtml += "</select>", t += a.yearshtml, a.yearshtml = null
                }
            return t += this._get(a, "yearSuffix"), s && (t += (!f && q && r ? "" : "&#xa0;") + u), t += "</div>"
        },
        _adjustInstDate: function(a, b, c) {
            var d = a.drawYear + ("Y" === c ? b : 0), e = a.drawMonth + ("M" === c ? b : 0), f = Math.min(a.selectedDay, this._getDaysInMonth(d, e)) + ("D" === c ? b : 0), g = this._restrictMinMax(a, this._daylightSavingAdjust(new Date(d, e, f)));
            a.selectedDay = g.getDate(), a.drawMonth = a.selectedMonth = g.getMonth(), a.drawYear = a.selectedYear = g.getFullYear(), ("M" === c || "Y" === c) && this._notifyChange(a)
        },
        _restrictMinMax: function(a, b) {
            var c = this._getMinMaxDate(a, "min"), d = this._getMinMaxDate(a, "max"), e = c && c > b ? c: b;
            return d && e > d ? d : e
        },
        _notifyChange: function(a) {
            var b = this._get(a, "onChangeMonthYear");
            b && b.apply(a.input ? a.input[0] : null, [a.selectedYear, a.selectedMonth + 1, a])
        },
        _getNumberOfMonths: function(a) {
            var b = this._get(a, "numberOfMonths");
            return null == b ? [1, 1] : "number" == typeof b ? [1, b] : b
        },
        _getMinMaxDate: function(a, b) {
            return this._determineDate(a, this._get(a, b + "Date"), null)
        },
        _getDaysInMonth: function(a, b) {
            return 32 - this._daylightSavingAdjust(new Date(a, b, 32)).getDate()
        },
        _getFirstDayOfMonth: function(a, b) {
            return new Date(a, b, 1).getDay()
        },
        _canAdjustMonth: function(a, b, c, d) {
            var e = this._getNumberOfMonths(a), f = this._daylightSavingAdjust(new Date(c, d + (0 > b ? b : e[0] * e[1]), 1));
            return 0 > b && f.setDate(this._getDaysInMonth(f.getFullYear(), f.getMonth())), this._isInRange(a, f)
        },
        _isInRange: function(a, b) {
            var c, d, e = this._getMinMaxDate(a, "min"), f = this._getMinMaxDate(a, "max"), g = null, h = null, i = this._get(a, "yearRange");
            return i && (c = i.split(":"), d = (new Date).getFullYear(), g = parseInt(c[0], 10), h = parseInt(c[1], 10), c[0].match(/[+\-].*/) && (g += d), c[1].match(/[+\-].*/) && (h += d)), (!e || b.getTime() >= e.getTime()) && (!f || b.getTime() <= f.getTime()) && (!g || b.getFullYear() >= g) && (!h || h >= b.getFullYear())
        },
        _getFormatConfig: function(a) {
            var b = this._get(a, "shortYearCutoff");
            return b = "string" != typeof b ? b : (new Date).getFullYear()%100 + parseInt(b, 10), {
                shortYearCutoff: b,
                dayNamesShort: this._get(a, "dayNamesShort"),
                dayNames: this._get(a, "dayNames"),
                monthNamesShort: this._get(a, "monthNamesShort"),
                monthNames: this._get(a, "monthNames")
            }
        },
        _formatDate: function(a, b, c, d) {
            b || (a.currentDay = a.selectedDay, a.currentMonth = a.selectedMonth, a.currentYear = a.selectedYear);
            var e = b ? "object" == typeof b ? b: this._daylightSavingAdjust(new Date(d, c, b)): this._daylightSavingAdjust(new Date(a.currentYear, a.currentMonth, a.currentDay));
            return this.formatDate(this._get(a, "dateFormat"), e, this._getFormatConfig(a))
        }
    }), a.fn.datepicker = function(b) {
        if (!this.length)
            return this;
        a.datepicker.initialized || (a(document).mousedown(a.datepicker._checkExternalClick), a.datepicker.initialized=!0), 0 === a("#" + a.datepicker._mainDivId).length && a("body").append(a.datepicker.dpDiv);
        var c = Array.prototype.slice.call(arguments, 1);
        return "string" != typeof b || "isDisabled" !== b && "getDate" !== b && "widget" !== b ? "option" === b && 2 === arguments.length && "string" == typeof arguments[1] ? a.datepicker["_" + b + "Datepicker"].apply(a.datepicker, [this[0]].concat(c)) : this.each(function() {
            "string" == typeof b ? a.datepicker["_" + b + "Datepicker"].apply(a.datepicker, [this].concat(c)) : a.datepicker._attachDatepicker(this, b)
        }) : a.datepicker["_" + b + "Datepicker"].apply(a.datepicker, [this[0]].concat(c))
    }, a.datepicker = new c, a.datepicker.initialized=!1, a.datepicker.uuid = (new Date).getTime(), a.datepicker.version = "1.10.3"
}(jQuery), function(a) {
    function b(b, c) {
        var d = (b.attr("aria-describedby") || "").split(/\s+/);
        d.push(c), b.data("ui-tooltip-id", c).attr("aria-describedby", a.trim(d.join(" ")))
    }
    function c(b) {
        var c = b.data("ui-tooltip-id"), d = (b.attr("aria-describedby") || "").split(/\s+/), e = a.inArray(c, d);
        - 1 !== e && d.splice(e, 1), b.removeData("ui-tooltip-id"), d = a.trim(d.join(" ")), d ? b.attr("aria-describedby", d) : b.removeAttr("aria-describedby")
    }
    var d = 0;
    a.widget("ui.tooltip", {
        version: "1.10.3",
        options: {
            content: function() {
                var b = a(this).attr("title") || "";
                return a("<a>").text(b).html()
            },
            hide: !0,
            items: "[title]:not([disabled])",
            position: {
                my: "left top+15",
                at: "left bottom",
                collision: "flipfit flip"
            },
            show: !0,
            tooltipClass: null,
            track: !1,
            close: null,
            open: null
        },
        _create: function() {
            this._on({
                mouseover: "open",
                focusin: "open"
            }), this.tooltips = {}, this.parents = {}, this.options.disabled && this._disable()
        },
        _setOption: function(b, c) {
            var d = this;
            return "disabled" === b ? (this[c ? "_disable": "_enable"](), void(this.options[b] = c)) : (this._super(b, c), void("content" === b && a.each(this.tooltips, function(a, b) {
                d._updateContent(b)
            })))
        },
        _disable: function() {
            var b = this;
            a.each(this.tooltips, function(c, d) {
                var e = a.Event("blur");
                e.target = e.currentTarget = d[0], b.close(e, !0)
            }), this.element.find(this.options.items).addBack().each(function() {
                var b = a(this);
                b.is("[title]") && b.data("ui-tooltip-title", b.attr("title")).attr("title", "")
            })
        },
        _enable: function() {
            this.element.find(this.options.items).addBack().each(function() {
                var b = a(this);
                b.data("ui-tooltip-title") && b.attr("title", b.data("ui-tooltip-title"))
            })
        },
        open: function(b) {
            var c = this, d = a(b ? b.target : this.element).closest(this.options.items);
            d.length&&!d.data("ui-tooltip-id") && (d.attr("title") && d.data("ui-tooltip-title", d.attr("title")), d.data("ui-tooltip-open", !0), b && "mouseover" === b.type && d.parents().each(function() {
                var b, d = a(this);
                d.data("ui-tooltip-open") && (b = a.Event("blur"), b.target = b.currentTarget = this, c.close(b, !0)), d.attr("title") && (d.uniqueId(), c.parents[this.id] = {
                    element: this,
                    title: d.attr("title")
                }, d.attr("title", ""))
            }), this._updateContent(d, b))
        },
        _updateContent: function(a, b) {
            var c, d = this.options.content, e = this, f = b ? b.type: null;
            return "string" == typeof d ? this._open(b, a, d) : (c = d.call(a[0], function(c) {
                a.data("ui-tooltip-open") && e._delay(function() {
                    b && (b.type = f), this._open(b, a, c)
                })
            }), void(c && this._open(b, a, c)))
        },
        _open: function(c, d, e) {
            function f(a) {
                j.of = a, g.is(":hidden") || g.position(j)
            }
            var g, h, i, j = a.extend({}, this.options.position);
            if (e) {
                if (g = this._find(d), g.length)
                    return void g.find(".ui-tooltip-content").html(e);
                d.is("[title]") && (c && "mouseover" === c.type ? d.attr("title", "") : d.removeAttr("title")), g = this._tooltip(d), b(d, g.attr("id")), g.find(".ui-tooltip-content").html(e), this.options.track && c && /^mouse/.test(c.type) ? (this._on(this.document, {
                    mousemove: f
                }), f(c)) : g.position(a.extend({
                    of: d
                }, this.options.position)), g.hide(), this._show(g, this.options.show), this.options.show && this.options.show.delay && (i = this.delayedShow = setInterval(function() {
                    g.is(":visible") && (f(j.of), clearInterval(i))
                }, a.fx.interval)), this._trigger("open", c, {
                    tooltip: g
                }), h = {
                    keyup: function(b) {
                        if (b.keyCode === a.ui.keyCode.ESCAPE) {
                            var c = a.Event(b);
                            c.currentTarget = d[0], this.close(c, !0)
                        }
                    },
                    remove: function() {
                        this._removeTooltip(g)
                    }
                }, c && "mouseover" !== c.type || (h.mouseleave = "close"), c && "focusin" !== c.type || (h.focusout = "close"), this._on(!0, d, h)
            }
        },
        close: function(b) {
            var d = this, e = a(b ? b.currentTarget : this.element), f = this._find(e);
            this.closing || (clearInterval(this.delayedShow), e.data("ui-tooltip-title") && e.attr("title", e.data("ui-tooltip-title")), c(e), f.stop(!0), this._hide(f, this.options.hide, function() {
                d._removeTooltip(a(this))
            }), e.removeData("ui-tooltip-open"), this._off(e, "mouseleave focusout keyup"), e[0] !== this.element[0] && this._off(e, "remove"), this._off(this.document, "mousemove"), b && "mouseleave" === b.type && a.each(this.parents, function(b, c) {
                a(c.element).attr("title", c.title), delete d.parents[b]
            }), this.closing=!0, this._trigger("close", b, {
                tooltip: f
            }), this.closing=!1)
        },
        _tooltip: function(b) {
            var c = "ui-tooltip-" + d++, e = a("<div>").attr({
                id: c,
                role: "tooltip"
            }).addClass("ui-tooltip ui-widget ui-corner-all ui-widget-content " + (this.options.tooltipClass || ""));
            return a("<div>").addClass("ui-tooltip-content").appendTo(e), e.appendTo(this.document[0].body), this.tooltips[c] = b, e
        },
        _find: function(b) {
            var c = b.data("ui-tooltip-id");
            return c ? a("#" + c) : a()
        },
        _removeTooltip: function(a) {
            a.remove(), delete this.tooltips[a.attr("id")]
        },
        _destroy: function() {
            var b = this;
            a.each(this.tooltips, function(c, d) {
                var e = a.Event("blur");
                e.target = e.currentTarget = d[0], b.close(e, !0), a("#" + c).remove(), d.data("ui-tooltip-title") && (d.attr("title", d.data("ui-tooltip-title")), d.removeData("ui-tooltip-title"))
            })
        }
    })
}(jQuery), function(a) {
    "function" == typeof define && define.amd ? define(["../datepicker"], a) : a(jQuery.datepicker)
}(function(a) {
    return a.regional.de = {
        closeText: "Schließen",
        prevText: "&#x3C;Zurück",
        nextText: "Vor&#x3E;",
        currentText: "Heute",
        monthNames: ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"],
        monthNamesShort: ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"],
        dayNames: ["Sonntag", "Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag"],
        dayNamesShort: ["So", "Mo", "Di", "Mi", "Do", "Fr", "Sa"],
        dayNamesMin: ["So", "Mo", "Di", "Mi", "Do", "Fr", "Sa"],
        weekHeader: "KW",
        dateFormat: "dd.mm.yy",
        firstDay: 1,
        isRTL: !1,
        showMonthAfterYear: !1,
        yearSuffix: ""
    }, a.setDefaults(a.regional.de), a.regional.de
}), function(a) {
    "function" == typeof define && define.amd ? define(["../datepicker"], a) : a(jQuery.datepicker)
}(function(a) {
    return a.regional.es = {
        closeText: "Cerrar",
        prevText: "&#x3C;Ant",
        nextText: "Sig&#x3E;",
        currentText: "Hoy",
        monthNames: ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
        monthNamesShort: ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"],
        dayNames: ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"],
        dayNamesShort: ["dom", "lun", "mar", "mié", "jue", "vie", "sáb"],
        dayNamesMin: ["D", "L", "M", "X", "J", "V", "S"],
        weekHeader: "Sm",
        dateFormat: "dd/mm/yy",
        firstDay: 1,
        isRTL: !1,
        showMonthAfterYear: !1,
        yearSuffix: ""
    }, a.setDefaults(a.regional.es), a.regional.es
}), function(a) {
    "function" == typeof define && define.amd ? define(["../datepicker"], a) : a(jQuery.datepicker)
}(function(a) {
    return a.regional["pt-BR"] = {
        closeText: "Fechar",
        prevText: "&#x3C;Anterior",
        nextText: "Próximo&#x3E;",
        currentText: "Hoje",
        monthNames: ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"],
        monthNamesShort: ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"],
        dayNames: ["Domingo", "Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado"],
        dayNamesShort: ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"],
        dayNamesMin: ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"],
        weekHeader: "Sm",
        dateFormat: "dd/mm/yy",
        firstDay: 0,
        isRTL: !1,
        showMonthAfterYear: !1,
        yearSuffix: ""
    }, a.setDefaults(a.regional["pt-BR"]), a.regional["pt-BR"]
}), function(a, b, c) {
    function d(a) {
        var b = {}, d = /^jQuery\d+$/;
        return c.each(a.attributes, function(a, c) {
            c.specified&&!d.test(c.name) && (b[c.name] = c.value)
        }), b
    }
    function e(a, b) {
        var d = this, e = c(d);
        if (d.value == e.attr("placeholder") && e.hasClass("placeholder"))
            if (e.data("placeholder-password")) {
                if (e = e.hide().next().show().attr("id", e.removeAttr("id").data("placeholder-id")), a===!0)
                    return e[0].value = b;
                    e.focus()
            } else 
                d.value = "", e.removeClass("placeholder"), d == g() && d.select()
    }
    function f() {
        var a, b = this, f = c(b), g = this.id;
        if ("" == b.value) {
            if ("password" == b.type) {
                if (!f.data("placeholder-textinput")) {
                    try {
                        a = f.clone().attr({
                            type: "text"
                        })
                    } catch (h) {
                        a = c("<input>").attr(c.extend(d(this), {
                            type: "text"
                        }))
                    }
                    a.removeAttr("name").data({
                        "placeholder-password": f,
                        "placeholder-id": g
                    }).bind("focus.placeholder", e), f.data({
                        "placeholder-textinput": a,
                        "placeholder-id": g
                    }).before(a)
                }
                f = f.removeAttr("id").hide().prev().attr("id", g).show()
            }
            f.addClass("placeholder"), f[0].value = f.attr("placeholder")
        } else 
            f.removeClass("placeholder")
    }
    function g() {
        try {
            return b.activeElement
        } catch (a) {}
    }
    var h, i, j = "placeholder"in b.createElement("input"), k = "placeholder"in b.createElement("textarea"), l = c.fn, m = c.valHooks, n = c.propHooks;
    j && k ? (i = l.placeholder = function() {
        return this
    }, i.input = i.textarea=!0) : (i = l.placeholder = function() {
        var a = this;
        return a.filter((j ? "textarea" : ":input") + "[placeholder]").not(".placeholder").bind({
            "focus.placeholder": e,
            "blur.placeholder": f
        }).data("placeholder-enabled", !0).trigger("blur.placeholder"), a
    }, i.input = j, i.textarea = k, h = {
        get: function(a) {
            var b = c(a), d = b.data("placeholder-password");
            return d ? d[0].value : b.data("placeholder-enabled") && b.hasClass("placeholder") ? "" : a.value
        },
        set: function(a, b) {
            var d = c(a), h = d.data("placeholder-password");
            return h ? h[0].value = b : d.data("placeholder-enabled") ? ("" == b ? (a.value = b, a != g() && f.call(a)) : d.hasClass("placeholder") ? e.call(a, !0, b) || (a.value = b) : a.value = b, d) : a.value = b
        }
    }, j || (m.input = h, n.value = h), k || (m.textarea = h, n.value = h), c(function() {
        c(b).delegate("form", "submit.placeholder", function() {
            var a = c(".placeholder", this).each(e);
            setTimeout(function() {
                a.each(f)
            }, 10)
        })
    }), c(a).bind("beforeunload.placeholder", function() {
        c(".placeholder").each(function() {
            this.value = ""
        })
    }))
}(this, document, jQuery);
var Handlebars = {};
!function(a, b) {
    a.VERSION = "1.0.0-rc.4", a.COMPILER_REVISION = 3, a.REVISION_CHANGES = {
        1: "<= 1.0.rc.2",
        2: "== 1.0.0-rc.3",
        3: ">= 1.0.0-rc.4"
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
        throw new Error("Could not find property '" + a + "'")
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
    }, a.registerHelper("each", function(b, c) {
        var d, e = c.fn, f = c.inverse, g = 0, h = "";
        if (c.data && (d = a.createFrame(c.data)), b && "object" == typeof b)
            if (b instanceof Array)
                for (var i = b.length; i > g; g++)
                    d && (d.index = g), h += e(b[g], {
                        data: d
                    });
            else 
                for (var j in b)
                    b.hasOwnProperty(j) && (d && (d.key = j), h += e(b[j], {
                        data: d
                    }), g++);
        return 0 === g && (h = f(this)), h
    }), a.registerHelper("if", function(b, e) {
        var f = c.call(b);
        return f === d && (b = b.call(this)), !b || a.Utils.isEmpty(b) ? e.inverse(this) : e.fn(this)
    }), a.registerHelper("unless", function(b, c) {
        return a.helpers["if"].call(this, b, {
            fn: c.inverse,
            inverse: c.fn
        })
    }), a.registerHelper("with", function(b, c) {
        return a.Utils.isEmpty(b) ? void 0 : c.fn(b)
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
                OPEN_PARTIAL: 24,
                partialName: 25,
                params: 26,
                hash: 27,
                DATA: 28,
                param: 29,
                STRING: 30,
                INTEGER: 31,
                BOOLEAN: 32,
                hashSegments: 33,
                hashSegment: 34,
                ID: 35,
                EQUALS: 36,
                PARTIAL_NAME: 37,
                pathSegments: 38,
                SEP: 39,
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
                24: "OPEN_PARTIAL",
                28: "DATA",
                30: "STRING",
                31: "INTEGER",
                32: "BOOLEAN",
                35: "ID",
                36: "EQUALS",
                37: "PARTIAL_NAME",
                39: "SEP"
            },
            productions_: [0, [3, 2], [4, 2], [4, 3], [4, 2], [4, 1], [4, 1], [4, 0], [7, 1], [7, 2], [8, 3], [8, 3], [8, 1], [8, 1], [8, 1], [8, 1], [11, 3], [9, 3], [10, 3], [12, 3], [12, 3], [13, 3], [13, 4], [6, 2], [17, 3], [17, 2], [17, 2], [17, 1], [17, 1], [26, 2], [26, 1], [29, 1], [29, 1], [29, 1], [29, 1], [29, 1], [27, 1], [33, 2], [33, 1], [34, 3], [34, 3], [34, 3], [34, 3], [34, 3], [25, 1], [21, 1], [38, 3], [38, 1]],
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
                    this.$ = new d.MustacheNode(f[g - 1][0], f[g - 1][1]);
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
                    this.$ = [[new d.DataNode(f[g])], null];
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
                    this.$ = new d.DataNode(f[g]);
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
                    this.$ = [f[g - 2], new d.DataNode(f[g])];
                    break;
                case 44:
                    this.$ = new d.PartialNameNode(f[g]);
                    break;
                case 45:
                    this.$ = new d.IdNode(f[g]);
                    break;
                case 46:
                    f[g - 2].push(f[g]), this.$ = f[g - 2];
                    break;
                case 47:
                    this.$ = [f[g]]
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
                24: [1, 16]
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
                24: [1, 16]
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
                24: [1, 16]
            }, {
                17: 23,
                18: [1, 22],
                21: 24,
                28: [1, 25],
                35: [1, 27],
                38: 26
            }, {
                5: [2, 8],
                14: [2, 8],
                15: [2, 8],
                16: [2, 8],
                19: [2, 8],
                20: [2, 8],
                22: [2, 8],
                23: [2, 8],
                24: [2, 8]
            }, {
                4: 28,
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
                24: [1, 16]
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
                24: [1, 16]
            }, {
                5: [2, 12],
                14: [2, 12],
                15: [2, 12],
                16: [2, 12],
                19: [2, 12],
                20: [2, 12],
                22: [2, 12],
                23: [2, 12],
                24: [2, 12]
            }, {
                5: [2, 13],
                14: [2, 13],
                15: [2, 13],
                16: [2, 13],
                19: [2, 13],
                20: [2, 13],
                22: [2, 13],
                23: [2, 13],
                24: [2, 13]
            }, {
                5: [2, 14],
                14: [2, 14],
                15: [2, 14],
                16: [2, 14],
                19: [2, 14],
                20: [2, 14],
                22: [2, 14],
                23: [2, 14],
                24: [2, 14]
            }, {
                5: [2, 15],
                14: [2, 15],
                15: [2, 15],
                16: [2, 15],
                19: [2, 15],
                20: [2, 15],
                22: [2, 15],
                23: [2, 15],
                24: [2, 15]
            }, {
                17: 30,
                21: 24,
                28: [1, 25],
                35: [1, 27],
                38: 26
            }, {
                17: 31,
                21: 24,
                28: [1, 25],
                35: [1, 27],
                38: 26
            }, {
                17: 32,
                21: 24,
                28: [1, 25],
                35: [1, 27],
                38: 26
            }, {
                25: 33,
                37: [1, 34]
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
                24: [1, 16]
            }, {
                17: 23,
                21: 24,
                28: [1, 25],
                35: [1, 27],
                38: 26
            }, {
                5: [2, 4],
                7: 35,
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
                24: [1, 16]
            }, {
                5: [2, 9],
                14: [2, 9],
                15: [2, 9],
                16: [2, 9],
                19: [2, 9],
                20: [2, 9],
                22: [2, 9],
                23: [2, 9],
                24: [2, 9]
            }, {
                5: [2, 23],
                14: [2, 23],
                15: [2, 23],
                16: [2, 23],
                19: [2, 23],
                20: [2, 23],
                22: [2, 23],
                23: [2, 23],
                24: [2, 23]
            }, {
                18: [1, 36]
            }, {
                18: [2, 27],
                21: 41,
                26: 37,
                27: 38,
                28: [1, 45],
                29: 39,
                30: [1, 42],
                31: [1, 43],
                32: [1, 44],
                33: 40,
                34: 46,
                35: [1, 47],
                38: 26
            }, {
                18: [2, 28]
            }, {
                18: [2, 45],
                28: [2, 45],
                30: [2, 45],
                31: [2, 45],
                32: [2, 45],
                35: [2, 45],
                39: [1, 48]
            }, {
                18: [2, 47],
                28: [2, 47],
                30: [2, 47],
                31: [2, 47],
                32: [2, 47],
                35: [2, 47],
                39: [2, 47]
            }, {
                10: 49,
                20: [1, 50]
            }, {
                10: 51,
                20: [1, 50]
            }, {
                18: [1, 52]
            }, {
                18: [1, 53]
            }, {
                18: [1, 54]
            }, {
                18: [1, 55],
                21: 56,
                35: [1, 27],
                38: 26
            }, {
                18: [2, 44],
                35: [2, 44]
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
                24: [1, 16]
            }, {
                14: [2, 17],
                15: [2, 17],
                16: [2, 17],
                19: [2, 17],
                20: [2, 17],
                22: [2, 17],
                23: [2, 17],
                24: [2, 17]
            }, {
                18: [2, 25],
                21: 41,
                27: 57,
                28: [1, 45],
                29: 58,
                30: [1, 42],
                31: [1, 43],
                32: [1, 44],
                33: 40,
                34: 46,
                35: [1, 47],
                38: 26
            }, {
                18: [2, 26]
            }, {
                18: [2, 30],
                28: [2, 30],
                30: [2, 30],
                31: [2, 30],
                32: [2, 30],
                35: [2, 30]
            }, {
                18: [2, 36],
                34: 59,
                35: [1, 60]
            }, {
                18: [2, 31],
                28: [2, 31],
                30: [2, 31],
                31: [2, 31],
                32: [2, 31],
                35: [2, 31]
            }, {
                18: [2, 32],
                28: [2, 32],
                30: [2, 32],
                31: [2, 32],
                32: [2, 32],
                35: [2, 32]
            }, {
                18: [2, 33],
                28: [2, 33],
                30: [2, 33],
                31: [2, 33],
                32: [2, 33],
                35: [2, 33]
            }, {
                18: [2, 34],
                28: [2, 34],
                30: [2, 34],
                31: [2, 34],
                32: [2, 34],
                35: [2, 34]
            }, {
                18: [2, 35],
                28: [2, 35],
                30: [2, 35],
                31: [2, 35],
                32: [2, 35],
                35: [2, 35]
            }, {
                18: [2, 38],
                35: [2, 38]
            }, {
                18: [2, 47],
                28: [2, 47],
                30: [2, 47],
                31: [2, 47],
                32: [2, 47],
                35: [2, 47],
                36: [1, 61],
                39: [2, 47]
            }, {
                35: [1, 62]
            }, {
                5: [2, 10],
                14: [2, 10],
                15: [2, 10],
                16: [2, 10],
                19: [2, 10],
                20: [2, 10],
                22: [2, 10],
                23: [2, 10],
                24: [2, 10]
            }, {
                21: 63,
                35: [1, 27],
                38: 26
            }, {
                5: [2, 11],
                14: [2, 11],
                15: [2, 11],
                16: [2, 11],
                19: [2, 11],
                20: [2, 11],
                22: [2, 11],
                23: [2, 11],
                24: [2, 11]
            }, {
                14: [2, 16],
                15: [2, 16],
                16: [2, 16],
                19: [2, 16],
                20: [2, 16],
                22: [2, 16],
                23: [2, 16],
                24: [2, 16]
            }, {
                5: [2, 19],
                14: [2, 19],
                15: [2, 19],
                16: [2, 19],
                19: [2, 19],
                20: [2, 19],
                22: [2, 19],
                23: [2, 19],
                24: [2, 19]
            }, {
                5: [2, 20],
                14: [2, 20],
                15: [2, 20],
                16: [2, 20],
                19: [2, 20],
                20: [2, 20],
                22: [2, 20],
                23: [2, 20],
                24: [2, 20]
            }, {
                5: [2, 21],
                14: [2, 21],
                15: [2, 21],
                16: [2, 21],
                19: [2, 21],
                20: [2, 21],
                22: [2, 21],
                23: [2, 21],
                24: [2, 21]
            }, {
                18: [1, 64]
            }, {
                18: [2, 24]
            }, {
                18: [2, 29],
                28: [2, 29],
                30: [2, 29],
                31: [2, 29],
                32: [2, 29],
                35: [2, 29]
            }, {
                18: [2, 37],
                35: [2, 37]
            }, {
                36: [1, 61]
            }, {
                21: 65,
                28: [1, 69],
                30: [1, 66],
                31: [1, 67],
                32: [1, 68],
                35: [1, 27],
                38: 26
            }, {
                18: [2, 46],
                28: [2, 46],
                30: [2, 46],
                31: [2, 46],
                32: [2, 46],
                35: [2, 46],
                39: [2, 46]
            }, {
                18: [1, 70]
            }, {
                5: [2, 22],
                14: [2, 22],
                15: [2, 22],
                16: [2, 22],
                19: [2, 22],
                20: [2, 22],
                22: [2, 22],
                23: [2, 22],
                24: [2, 22]
            }, {
                18: [2, 39],
                35: [2, 39]
            }, {
                18: [2, 40],
                35: [2, 40]
            }, {
                18: [2, 41],
                35: [2, 41]
            }, {
                18: [2, 42],
                35: [2, 42]
            }, {
                18: [2, 43],
                35: [2, 43]
            }, {
                5: [2, 18],
                14: [2, 18],
                15: [2, 18],
                16: [2, 18],
                19: [2, 18],
                20: [2, 18],
                22: [2, 18],
                23: [2, 18],
                24: [2, 18]
            }
            ],
            defaultActions: {
                17: [2, 1],
                25: [2, 28],
                38: [2, 26],
                57: [2, 24]
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
                    return this.begin("par"), 24;
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
                    return 23;
                case 12:
                    this.popState(), this.begin("com");
                    break;
                case 13:
                    return b.yytext = b.yytext.substr(3, b.yyleng - 5), this.popState(), 15;
                case 14:
                    return 22;
                case 15:
                    return 36;
                case 16:
                    return 35;
                case 17:
                    return 35;
                case 18:
                    return 39;
                case 19:
                    break;
                case 20:
                    return this.popState(), 18;
                case 21:
                    return this.popState(), 18;
                case 22:
                    return b.yytext = b.yytext.substr(1, b.yyleng - 2).replace(/\\"/g, '"'), 30;
                case 23:
                    return b.yytext = b.yytext.substr(1, b.yyleng - 2).replace(/\\'/g, "'"), 30;
                case 24:
                    return b.yytext = b.yytext.substr(1), 28;
                case 25:
                    return 32;
                case 26:
                    return 32;
                case 27:
                    return 31;
                case 28:
                    return 35;
                case 29:
                    return b.yytext = b.yytext.substr(1, b.yyleng - 2), 35;
                case 30:
                    return "INVALID";
                case 31:
                    break;
                case 32:
                    return this.popState(), 37;
                case 33:
                    return 5
                }
            }, a.rules = [/^(?:\\\\(?=(\{\{)))/, /^(?:[^\x00]*?(?=(\{\{)))/, /^(?:[^\x00]+)/, /^(?:[^\x00]{2,}?(?=(\{\{|$)))/, /^(?:[\s\S]*?--\}\})/, /^(?:\{\{>)/, /^(?:\{\{#)/, /^(?:\{\{\/)/, /^(?:\{\{\^)/, /^(?:\{\{\s*else\b)/, /^(?:\{\{\{)/, /^(?:\{\{&)/, /^(?:\{\{!--)/, /^(?:\{\{![\s\S]*?\}\})/, /^(?:\{\{)/, /^(?:=)/, /^(?:\.(?=[}/ ]))/, /^(?:\.\.)/, /^(?:[\/.])/, /^(?:\s+)/, /^(?:\}\}\})/, /^(?:\}\})/, /^(?:"(\\["]|[^"])*")/, /^(?:'(\\[']|[^'])*')/, /^(?:@[a-zA-Z]+)/, /^(?:true(?=[}\s]))/, /^(?:false(?=[}\s]))/, /^(?:-?[0-9]+(?=[}\s]))/, /^(?:[a-zA-Z0-9_$:\-]+(?=[=}\s\/.]))/, /^(?:\[[^\]]*\])/, /^(?:.)/, /^(?:\s+)/, /^(?:[a-zA-Z0-9_$\-\/]+)/, /^(?:$)/], a.conditions = {
                mu: {
                    rules: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33],
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
                par: {
                    rules: [31, 32],
                    inclusive: !1
                },
                INITIAL: {
                    rules: [0, 1, 2, 33],
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
        this.type = "ID", this.original = b.join(".");
        for (var c = [], d = 0, e = 0, f = b.length; f > e; e++) {
            var g = b[e];
            if (".." === g || "." === g || "this" === g) {
                if (c.length > 0)
                    throw new a.Exception("Invalid path: " + this.original);
                ".." === g ? d++ : this.isScoped=!0
            } else 
                c.push(g)
        }
        this.parts = c, this.string = c.join("."), this.depth = d, this.isSimple = 1 === b.length&&!this.isScoped && 0 === d, this.stringModeValue = this.string
    }, a.AST.PartialNameNode = function(a) {
        this.type = "PARTIAL_NAME", this.name = a
    }, a.AST.DataNode = function(a) {
        this.type = "DATA", this.id = a
    }, a.AST.StringNode = function(a) {
        this.type = "STRING", this.string = a, this.stringModeValue = a
    }, a.AST.IntegerNode = function(a) {
        this.type = "INTEGER", this.integer = a, this.stringModeValue = Number(a)
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
        DATA: function(a) {
            this.options.data=!0, this.opcode("lookupData", a.id)
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
                var b = this.namespace, c = "helpers = helpers || " + b + ".helpers;";
                this.environment.usePartial && (c = c + " partials = partials || " + b + ".partials;"), this.options.data && (c += " data = data || {};"), a.push(c)
            }
            a.push(this.environment.isSimple ? "" : ", buffer = " + this.initializeBuffer()), this.lastContext = 0, this.source = a
        },
        createFunctionContext: function(b) {
            var c = this.stackVars.concat(this.registers.list);
            if (c.length > 0 && (this.source[1] = this.source[1] + ", " + c.join(", ")), !this.isChild)
                for (var d in this.context.aliases)
                    this.source[1] = this.source[1] + ", " + d + "=" + this.context.aliases[d];
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
        lookupData: function(a) {
            this.push(this.nameLookup("data", a, "data"))
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
            var c = this.lastHelper = this.setupHelper(a, b, !0);
            this.push(c.name), this.replaceStack(function(a) {
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
            if (a.compile)
                return g[d] = a.compile(c, {
                    data: h !== b
                }), g[d](e, i);
            throw new a.Exception("The partial " + d + " could not be compiled when running in runtime-only mode")
        }
    }, a.template = a.VM.template
}(Handlebars), !function() {
    function a(a, b) {
        if (!b || "function" == typeof b)
            return a;
        for (var c in b)
            a[c] = b[c];
        return a
    }
    function b(a, c) {
        for (var d in c)
            d in a ? b(a[d], c[d]) : a[d] = c[d];
        return a
    }
    function c(a, b, c) {
        var d, e = 0, f = a.length, g = void 0 === f || "[object Array]" !== Object.prototype.toString.apply(a) || "function" == typeof a;
        if (c)
            if (g) {
                for (d in a)
                    if (b.apply(a[d], c)===!1)
                        break
            } else 
                for (; f > e && b.apply(a[e++], c)!==!1;);
                else if (g) {
            for (d in a)
                if (b.call(a[d], d, a[d])===!1)
                    break
        } else 
            for (; f > e && b.call(a[e], e, a[e++])!==!1;);
        return a
    }
    function d(a) {
        return "string" == typeof a ? a.replace(/[&<>"'\/]/g, function(a) {
            return O[a]
        }) : a
    }
    function e(a) {
        var b = function(a) {
            if (window.XMLHttpRequest)
                return a(null, new XMLHttpRequest);
            if (window.ActiveXObject)
                try {
                    return a(null, new ActiveXObject("Msxml2.XMLHTTP"))
            } catch (b) {
                return a(null, new ActiveXObject("Microsoft.XMLHTTP"))
            }
            return a(new Error)
        }, c = function(a) {
            if ("string" == typeof a)
                return a;
            var b = [];
            for (var c in a)
                a.hasOwnProperty(c) && b.push(encodeURIComponent(c) + "=" + encodeURIComponent(a[c]));
            return b.join("&")
        }, d = function(a) {
            a = a.replace(/\r\n/g, "\n");
            for (var b = "", c = 0; c < a.length; c++) {
                var d = a.charCodeAt(c);
                128 > d ? b += String.fromCharCode(d) : d > 127 && 2048 > d ? (b += String.fromCharCode(192 | d>>6), b += String.fromCharCode(128 | 63 & d)) : (b += String.fromCharCode(224 | d>>12), b += String.fromCharCode(128 | 63 & d>>6), b += String.fromCharCode(128 | 63 & d))
            }
            return b
        }, e = function(a) {
            var b = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
            a = d(a);
            var c, e, f, g, h, i, j, k = "", l = 0;
            do 
                c = a.charCodeAt(l++), e = a.charCodeAt(l++), f = a.charCodeAt(l++), g = c>>2, h = (3 & c)<<4 | e>>4, i = (15 & e)<<2 | f>>6, j = 63 & f, isNaN(e) ? i = j = 64 : isNaN(f) && (j = 64), k += b.charAt(g) + b.charAt(h) + b.charAt(i) + b.charAt(j), c = e = f = "", g = h = i = j = "";
            while (l < a.length);
            return k
        }, f = function() {
            for (var a = arguments[0], b = 1; b < arguments.length; b++) {
                var c = arguments[b];
                for (var d in c)
                    c.hasOwnProperty(d) && (a[d] = c[d])
            }
            return a
        }, g = function(a, d, e, h) {
            "function" == typeof e && (h = e, e = {}), e.cache = e.cache ||!1, e.data = e.data || {}, e.headers = e.headers || {}, e.jsonp = e.jsonp ||!1, e.async = void 0 === e.async?!0 : e.async;
            var i, j = f({
                accept: "*/*",
                "content-type": "application/x-www-form-urlencoded;charset=UTF-8"
            }, g.headers, e.headers);
            if (i = "application/json" === j["content-type"] ? JSON.stringify(e.data) : c(e.data), "GET" === a) {
                var k = [];
                if (i && (k.push(i), i = null), e.cache || k.push("_=" + (new Date).getTime()), e.jsonp && (k.push("callback=" + e.jsonp), k.push("jsonp=" + e.jsonp)), k = k.join("&"), k.length > 1 && (d += d.indexOf("?")>-1 ? "&" + k : "?" + k), e.jsonp) {
                    var l = document.getElementsByTagName("head")[0], m = document.createElement("script");
                    return m.type = "text/javascript", m.src = d, void l.appendChild(m)
                }
            }
            b(function(b, c) {
                if (b)
                    return h(b);
                c.open(a, d, e.async);
                for (var f in j)
                    j.hasOwnProperty(f) && c.setRequestHeader(f, j[f]);
                c.onreadystatechange = function() {
                    if (4 === c.readyState) {
                        var a = c.responseText || "";
                        if (!h)
                            return;
                        h(c.status, {
                            text: function() {
                                return a
                            },
                            json: function() {
                                return JSON.parse(a)
                            }
                        })
                    }
                }, c.send(i)
            })
        }, h = {
            authBasic: function(a, b) {
                g.headers.Authorization = "Basic " + e(a + ":" + b)
            },
            connect: function(a, b, c) {
                return g("CONNECT", a, b, c)
            },
            del: function(a, b, c) {
                return g("DELETE", a, b, c)
            },
            get: function(a, b, c) {
                return g("GET", a, b, c)
            },
            head: function(a, b, c) {
                return g("HEAD", a, b, c)
            },
            headers: function(a) {
                g.headers = a || {}
            },
            isAllowed: function(a, b, c) {
                this.options(a, function(a, d) {
                    c( - 1 !== d.text().indexOf(b))
                })
            },
            options: function(a, b, c) {
                return g("OPTIONS", a, b, c)
            },
            patch: function(a, b, c) {
                return g("PATCH", a, b, c)
            },
            post: function(a, b, c) {
                return g("POST", a, b, c)
            },
            put: function(a, b, c) {
                return g("PUT", a, b, c)
            },
            trace: function(a, b, c) {
                return g("TRACE", a, b, c)
            }
        }, i = a.type ? a.type.toLowerCase(): "get";
        h[i](a.url, a, function(b, c) {
            200 === b || 0 === b && c.text() ? a.success(c.json(), b, null) : a.error(c.text(), b, null)
        })
    }
    function f(a, b) {
        "function" == typeof a && (b = a, a = {}), a = a || {}, R.extend(N, a), delete N.fixLng, "string" == typeof N.ns && (N.ns = {
            namespaces: [N.ns],
            defaultNs: N.ns
        }), "string" == typeof N.fallbackNS && (N.fallbackNS = [N.fallbackNS]), ("string" == typeof N.fallbackLng || "boolean" == typeof N.fallbackLng) && (N.fallbackLng = [N.fallbackLng]), N.interpolationPrefixEscaped = R.regexEscape(N.interpolationPrefix), N.interpolationSuffixEscaped = R.regexEscape(N.interpolationSuffix), N.lng || (N.lng = R.detectLanguage()), J = R.toLanguages(N.lng), D = J[0], R.log("currentLng set to: " + D), N.useCookie && R.cookie.read(N.cookieName) !== D && R.cookie.create(N.cookieName, D, N.cookieExpirationTime, N.cookieDomain), N.detectLngFromLocalStorage && "undefined" != typeof document && window.localstorage && window.localStorage.setItem("i18next_lng", D);
        var c = x;
        a.fixLng && (c = function(a, b) {
            return b = b || {}, b.lng = b.lng || c.lng, x(a, b)
        }, c.lng = D), S.setCurrentLng(D), F && N.setJqueryExt && q();
        var d;
        if (F && F.Deferred && (d = F.Deferred()), !N.resStore) {
            var e = R.toLanguages(N.lng);
            "string" == typeof N.preload && (N.preload = [N.preload]);
            for (var f = 0, g = N.preload.length; g > f; f++)
                for (var h = R.toLanguages(N.preload[f]), i = 0, j = h.length; j > i; i++)
                    e.indexOf(h[i]) < 0 && e.push(h[i]);
            return G.sync.load(e, N, function(a, e) {
                H = e, K=!0, b && b(c), d && d.resolve(c)
            }), d ? d.promise() : void 0
        }
        return H = N.resStore, K=!0, b && b(c), d && d.resolve(c), d ? d.promise() : void 0
    }
    function g(a, b) {
        "string" == typeof a && (a = [a]);
        for (var c = 0, d = a.length; d > c; c++)
            N.preload.indexOf(a[c]) < 0 && N.preload.push(a[c]);
        return f(b)
    }
    function h(a, b, c, d) {
        "string" != typeof b ? (c = b, b = N.ns.defaultNs) : N.ns.namespaces.indexOf(b) < 0 && N.ns.namespaces.push(b), H[a] = H[a] || {}, H[a][b] = H[a][b] || {}, d ? R.deepExtend(H[a][b], c) : R.extend(H[a][b], c)
    }
    function i(a, b) {
        "string" != typeof b && (b = N.ns.defaultNs), H[a] = H[a] || {}, H[a][b] = {}
    }
    function j(a, b, c, d) {
        "string" != typeof b ? (resource = b, b = N.ns.defaultNs) : N.ns.namespaces.indexOf(b) < 0 && N.ns.namespaces.push(b), H[a] = H[a] || {}, H[a][b] = H[a][b] || {};
        for (var e = c.split(N.keyseparator), f = 0, g = H[N.lng][b]; e[f];)
            f == e.length - 1 ? g[e[f]] = d : (null == g[e[f]] && (g[e[f]] = {}), g = g[e[f]]), f++
    }
    function k(a, b, c) {
        "string" != typeof b ? (resource = b, b = N.ns.defaultNs) : N.ns.namespaces.indexOf(b) < 0 && N.ns.namespaces.push(b);
        for (var d in c)
            "string" == typeof c[d] && j(a, b, d, c[d])
    }
    function l(a) {
        N.ns.defaultNs = a
    }
    function m(a, b) {
        n([a], b)
    }
    function n(a, b) {
        var c = {
            dynamicLoad: N.dynamicLoad,
            resGetPath: N.resGetPath,
            getAsync: N.getAsync,
            customLoad: N.customLoad,
            ns: {
                namespaces: a,
                defaultNs: ""
            }
        }, d = R.toLanguages(N.lng);
        "string" == typeof N.preload && (N.preload = [N.preload]);
        for (var e = 0, f = N.preload.length; f > e; e++)
            for (var g = R.toLanguages(N.preload[e]), h = 0, i = g.length; i > h; h++)
                d.indexOf(g[h]) < 0 && d.push(g[h]);
        for (var j = [], k = 0, l = d.length; l > k; k++) {
            var m=!1, n = H[d[k]];
            if (n)
                for (var o = 0, p = a.length; p > o; o++)
                    n[a[o]] || (m=!0);
            else 
                m=!0;
            m && j.push(d[k])
        }
        j.length ? G.sync._fetch(j, c, function(c, d) {
            var e = a.length * j.length;
            R.each(a, function(a, c) {
                N.ns.namespaces.indexOf(c) < 0 && N.ns.namespaces.push(c), R.each(j, function(a, f) {
                    H[f] = H[f] || {}, H[f][c] = d[f][c], e--, 0 === e && b && (N.useLocalStorage && G.sync._storeLocal(H), b())
                })
            })
        }) : b && b()
    }
    function o(a, b, c) {
        return "function" == typeof b ? (c = b, b = {}) : b || (b = {}), b.lng = a, f(b, c)
    }
    function p() {
        return D
    }
    function q() {
        function a(a, b, c) {
            if (0 !== b.length) {
                var d = "text";
                if (0 === b.indexOf("[")) {
                    var e = b.split("]");
                    b = e[1], d = e[0].substr(1, e[0].length - 1)
                }
                b.indexOf(";") === b.length - 1 && (b = b.substr(0, b.length - 2));
                var f;
                if ("html" === d)
                    f = N.defaultValueFromContent ? F.extend({
                        defaultValue: a.html()
                    }, c) : c, a.html(F.t(b, f));
                else if ("text" === d)
                    f = N.defaultValueFromContent ? F.extend({
                        defaultValue: a.text()
                    }, c) : c, a.text(F.t(b, f));
                else if ("prepend" === d)
                    f = N.defaultValueFromContent ? F.extend({
                        defaultValue: a.html()
                    }, c) : c, a.prepend(F.t(b, f));
                else if ("append" === d)
                    f = N.defaultValueFromContent ? F.extend({
                        defaultValue: a.html()
                    }, c) : c, a.append(F.t(b, f));
                else if (0 === d.indexOf("data-")) {
                    var g = d.substr("data-".length);
                    f = N.defaultValueFromContent ? F.extend({
                        defaultValue: a.data(g)
                    }, c) : c;
                    var h = F.t(b, f);
                    a.data(g, h), a.attr(d, h)
                } else 
                    f = N.defaultValueFromContent ? F.extend({
                        defaultValue: a.attr(d)
                    }, c) : c, a.attr(d, F.t(b, f))
            }
        }
        function b(b, c) {
            var d = b.attr(N.selectorAttr);
            if (d || "undefined" == typeof d || d===!1 || (d = b.text() || b.val()), d) {
                var e = b, f = b.data("i18n-target");
                if (f && (e = b.find(f) || b), c || N.useDataAttrOptions!==!0 || (c = b.data("i18n-options")), c = c || {}, d.indexOf(";") >= 0) {
                    var g = d.split(";");
                    F.each(g, function(b, d) {
                        "" !== d && a(e, d, c)
                    })
                } else 
                    a(e, d, c);
                N.useDataAttrOptions===!0 && b.data("i18n-options", c)
            }
        }
        F.t = F.t || x, F.fn.i18n = function(a) {
            return this.each(function() {
                b(F(this), a);
                var c = F(this).find("[" + N.selectorAttr + "]");
                c.each(function() {
                    b(F(this), a)
                })
            })
        }
    }
    function r(a, b, c, d) {
        if (!a)
            return a;
        if (d = d || b, a.indexOf(d.interpolationPrefix || N.interpolationPrefix) < 0)
            return a;
        var e = d.interpolationPrefix ? R.regexEscape(d.interpolationPrefix): N.interpolationPrefixEscaped, f = d.interpolationSuffix ? R.regexEscape(d.interpolationSuffix): N.interpolationSuffixEscaped, g = "HTML" + f, h = b.replace && "object" == typeof b.replace ? b.replace: b;
        return R.each(h, function(b, h) {
            var i = c ? c + N.keyseparator + b: b;
            "object" == typeof h && null !== h ? a = r(a, h, i, d) : d.escapeInterpolation || N.escapeInterpolation ? (a = a.replace(new RegExp([e, i, g].join(""), "g"), R.regexReplacementEscape(h)), a = a.replace(new RegExp([e, i, f].join(""), "g"), R.regexReplacementEscape(R.escape(h)))) : a = a.replace(new RegExp([e, i, f].join(""), "g"), R.regexReplacementEscape(h))
        }), a
    }
    function s(a, b) {
        var c = ",", d = "{", e = "}", f = R.extend({}, b);
        for (delete f.postProcess; - 1 != a.indexOf(N.reusePrefix) && (I++, !(I > N.maxRecursion));) {
            var g = a.lastIndexOf(N.reusePrefix), h = a.indexOf(N.reuseSuffix, g) + N.reuseSuffix.length, i = a.substring(g, h), j = i.replace(N.reusePrefix, "").replace(N.reuseSuffix, "");
            if (g >= h)
                return R.error("there is an missing closing in following translation value", a), "";
            if ( - 1 != j.indexOf(c)) {
                var k = j.indexOf(c);
                if ( - 1 != j.indexOf(d, k)&&-1 != j.indexOf(e, k)) {
                    var l = j.indexOf(d, k), m = j.indexOf(e, l) + e.length;
                    try {
                        f = R.extend(f, JSON.parse(j.substring(l, m))), j = j.substring(0, k)
                    } catch (n) {}
                }
            }
            var o = A(j, f);
            a = a.replace(i, R.regexReplacementEscape(o))
        }
        return a
    }
    function t(a) {
        return a.context && ("string" == typeof a.context || "number" == typeof a.context)
    }
    function u(a, b) {
        return void 0 !== a.count && "string" != typeof a.count && S.needsPlural(b, a.count)
    }
    function v(a) {
        return void 0 !== a.indefinite_article && "string" != typeof a.indefinite_article && a.indefinite_article
    }
    function w(a, b) {
        b = b || {};
        var c = y(a, b), d = B(a, b);
        return void 0 !== d || d === c
    }
    function x(a, b) {
        return b = b || {}, K ? (I = 0, A.apply(null, arguments)) : (R.log("i18next not finished initialization. you might have called t function before loading resources finished."), b.defaultValue || "")
    }
    function y(a, b) {
        return void 0 !== b.defaultValue ? b.defaultValue : a
    }
    function z() {
        for (var a = [], b = 1; b < arguments.length; b++)
            a.push(arguments[b]);
        return {
            postProcess: "sprintf",
            sprintf: a
        }
    }
    function A(a, b) {
        if (b && "object" != typeof b ? "sprintf" === N.shortcutFunction ? b = z.apply(null, arguments) : "defaultValue" === N.shortcutFunction && (b = {
            defaultValue: b
        }) : b = b || {}, void 0 === a || null === a || "" === a)
            return "";
        "string" == typeof a && (a = [a]);
        var c = a[0];
        if (a.length > 1)
            for (var d = 0; d < a.length && (c = a[d], !w(c, b)); d++);
        var e, f = y(c, b), g = B(c, b), h = b.lng ? R.toLanguages(b.lng, b.fallbackLng): J, i = b.ns || N.ns.defaultNs;
        c.indexOf(N.nsseparator)>-1 && (e = c.split(N.nsseparator), i = e[0], c = e[1]), void 0 === g && N.sendMissing && "function" == typeof N.missingKeyHandler && (b.lng ? N.missingKeyHandler(h[0], i, c, f, h) : N.missingKeyHandler(N.lng, i, c, f, h));
        var j = b.postProcess || N.postProcess;
        void 0 !== g && j && T[j] && (g = T[j](g, c, b));
        var k = f;
        if (f.indexOf(N.nsseparator)>-1 && (e = f.split(N.nsseparator), k = e[1]), k === c && N.parseMissingKey && (f = N.parseMissingKey(f)), void 0 === g && (f = r(f, b), f = s(f, b), j && T[j])) {
            var l = y(c, b);
            g = T[j](l, c, b)
        }
        return void 0 !== g ? g : f
    }
    function B(a, b) {
        b = b || {};
        var c, d, e = y(a, b), f = J;
        if (!H)
            return e;
        if ("cimode" === f[0].toLowerCase())
            return e;
        if (b.lng && (f = R.toLanguages(b.lng, b.fallbackLng), !H[f[0]])) {
            var g = N.getAsync;
            N.getAsync=!1, G.sync.load(f, N, function(a, b) {
                R.extend(H, b), N.getAsync = g
            })
        }
        var h = b.ns || N.ns.defaultNs;
        if (a.indexOf(N.nsseparator)>-1) {
            var i = a.split(N.nsseparator);
            h = i[0], a = i[1]
        }
        if (t(b)) {
            c = R.extend({}, b), delete c.context, c.defaultValue = N.contextNotFound;
            var j = h + N.nsseparator + a + "_" + b.context;
            if (d = x(j, c), d != N.contextNotFound)
                return r(d, {
                    context: b.context
                })
        }
        if (u(b, f[0])) {
            c = R.extend({}, b), delete c.count, c.defaultValue = N.pluralNotFound;
            var k = h + N.nsseparator + a + N.pluralSuffix, l = S.get(f[0], b.count);
            if (l >= 0 ? k = k + "_" + l : 1 === l && (k = h + N.nsseparator + a), d = x(k, c), d != N.pluralNotFound)
                return r(d, {
                    count: b.count,
                    interpolationPrefix: b.interpolationPrefix,
                    interpolationSuffix: b.interpolationSuffix
                })
        }
        if (v(b)) {
            var m = R.extend({}, b);
            delete m.indefinite_article, m.defaultValue = N.indefiniteNotFound;
            var n = h + N.nsseparator + a + (b.count&&!u(b, f[0]) ||!b.count ? N.indefiniteSuffix : "");
            if (d = x(n, m), d != N.indefiniteNotFound)
                return d
        }
        for (var o, p = a.split(N.keyseparator), q = 0, w = f.length; w > q && void 0 === o; q++) {
            for (var z = f[q], C = 0, D = H[z] && H[z][h]; p[C];)
                D = D && D[p[C]], C++;
            if (void 0 !== D) {
                var E = Object.prototype.toString.apply(D);
                if ("string" == typeof D)
                    D = r(D, b), D = s(D, b);
                else if ("[object Array]" !== E || N.returnObjectTrees || b.returnObjectTrees) {
                    if (null === D && N.fallbackOnNull===!0)
                        D = void 0;
                    else if (null !== D)
                        if (N.returnObjectTrees || b.returnObjectTrees) {
                            if ("[object Number]" !== E && "[object Function]" !== E && "[object RegExp]" !== E) {
                                var F = "[object Array]" === E ? []: {};
                                R.each(D, function(c) {
                                    F[c] = A(h + N.nsseparator + a + N.keyseparator + c, b)
                                }), D = F
                            }
                        } else 
                            N.objectTreeKeyHandler && "function" == typeof N.objectTreeKeyHandler ? D = N.objectTreeKeyHandler(a, D, z, h, b) : (D = "key '" + h + ":" + a + " (" + z + ")' returned an object instead of string.", R.log(D))
                        } else 
                            D = D.join("\n"), D = r(D, b), D = s(D, b);
                "string" == typeof D && "" === D.trim() && N.fallbackOnEmpty===!0 && (D = void 0), o = D
            }
        }
        if (void 0 === o&&!b.isFallbackLookup && (N.fallbackToDefaultNS===!0 || N.fallbackNS && N.fallbackNS.length > 0)
            ) {
            if (b.isFallbackLookup=!0, N.fallbackNS.length) {
                for (var I = 0, K = N.fallbackNS.length; K > I; I++)
                    if (o = B(N.fallbackNS[I] + N.nsseparator + a, b)) {
                        var L = o.indexOf(N.nsseparator)>-1 ? o.split(N.nsseparator)[1]: o, M = e.indexOf(N.nsseparator)>-1 ? e.split(N.nsseparator)[1]: e;
                        if (L !== M)
                            break
                    }
            } else 
                o = B(a, b);
            b.isFallbackLookup=!1
        }
        return o
    }
    function C() {
        var a, b = [];
        if ("undefined" != typeof window && (!function() {
            for (var a = window.location.search.substring(1), c = a.split("&"), d = 0; d < c.length; d++) {
                var e = c[d].indexOf("=");
                if (e > 0) {
                    var f = c[d].substring(0, e), g = c[d].substring(e + 1);
                    b[f] = g
                }
            }
        }(), b[N.detectLngQS] && (a = b[N.detectLngQS])), !a && "undefined" != typeof document && N.useCookie) {
            var c = R.cookie.read(N.cookieName);
            c && (a = c)
        }
        return !a && "undefined" != typeof document && window.localstorage && N.detectLngFromLocalStorage && (a = window.localStorage.getItem("i18next_lng")), a || "undefined" == typeof navigator || (a = navigator.language ? navigator.language : navigator.userLanguage), a || (a = N.fallbackLng[0]), a
    }
    Array.prototype.indexOf || (Array.prototype.indexOf = function(a) {
        "use strict";
        if (null == this)
            throw new TypeError;
        var b = Object(this), c = b.length>>>0;
        if (0 === c)
            return - 1;
        var d = 0;
        if (arguments.length > 0 && (d = Number(arguments[1]), d != d ? d = 0 : 0 != d && 1 / 0 != d && d!=-1 / 0 && (d = (d > 0||-1) * Math.floor(Math.abs(d)))), d >= c)
            return - 1;
        for (var e = d >= 0 ? d : Math.max(c - Math.abs(d), 0); c > e; e++)
            if (e in b && b[e] === a)
                return e;
        return - 1
    }), Array.prototype.lastIndexOf || (Array.prototype.lastIndexOf = function(a) {
        "use strict";
        if (null == this)
            throw new TypeError;
        var b = Object(this), c = b.length>>>0;
        if (0 === c)
            return - 1;
        var d = c;
        arguments.length > 1 && (d = Number(arguments[1]), d != d ? d = 0 : 0 != d && d != 1 / 0 && d!=-(1 / 0) && (d = (d > 0||-1) * Math.floor(Math.abs(d))));
        for (var e = d >= 0 ? Math.min(d, c - 1) : c - Math.abs(d); e >= 0; e--)
            if (e in b && b[e] === a)
                return e;
        return - 1
    }), "function" != typeof String.prototype.trim && (String.prototype.trim = function() {
        return this.replace(/^\s+|\s+$/g, "")
    });
    var D, E = this, F = E.jQuery || E.Zepto, G = {}, H = {}, I = 0, J = [], K=!1, L = {};
    if ("undefined" != typeof module && module.exports) {
        if (!F)
            try {
                F = require("jquery")
            } catch (M) {}
        F && (F.i18n = F.i18n || G), module.exports = G
    } else 
        F && (F.i18n = F.i18n || G), E.i18n = E.i18n || G;
    L = {
        load: function(a, b, c) {
            b.useLocalStorage ? L._loadLocal(a, b, function(d, e) {
                for (var f = [], g = 0, h = a.length; h > g; g++)
                    e[a[g]] || f.push(a[g]);
                f.length > 0 ? L._fetch(f, b, function(a, b) {
                    R.extend(e, b), L._storeLocal(b), c(null, e)
                }) : c(null, e)
            }) : L._fetch(a, b, function(a, b) {
                c(null, b)
            })
        },
        _loadLocal: function(a, b, c) {
            var d = {}, e = (new Date).getTime();
            if (window.localStorage) {
                var f = a.length;
                R.each(a, function(a, g) {
                    var h = window.localStorage.getItem("res_" + g);
                    h && (h = JSON.parse(h), h.i18nStamp && h.i18nStamp + b.localStorageExpirationTime > e && (d[g] = h)), f--, 0 === f && c(null, d)
                })
            }
        },
        _storeLocal: function(a) {
            if (window.localStorage)
                for (var b in a)
                    a[b].i18nStamp = (new Date).getTime(), window.localStorage.setItem("res_" + b, JSON.stringify(a[b]))
        },
        _fetch: function(a, b, c) {
            var d = b.ns, e = {};
            if (b.dynamicLoad) {
                var f = function(a, b) {
                    c(null, b)
                };
                if ("function" == typeof b.customLoad)
                    b.customLoad(a, d.namespaces, b, f);
                else {
                    var g = r(b.resGetPath, {
                        lng: a.join("+"),
                        ns: d.namespaces.join("+")
                    });
                    R.ajax({
                        url: g,
                        success: function(a) {
                            R.log("loaded: " + g), f(null, a)
                        },
                        error: function(a, b, c) {
                            R.log("failed loading: " + g), f("failed loading resource.json error: " + c)
                        },
                        dataType: "json",
                        async: b.getAsync
                    })
                }
            } else {
                var h, i = d.namespaces.length * a.length;
                R.each(d.namespaces, function(d, f) {
                    R.each(a, function(a, d) {
                        var g = function(a, b) {
                            a && (h = h || [], h.push(a)), e[d] = e[d] || {}, e[d][f] = b, i--, 0 === i && c(h, e)
                        };
                        "function" == typeof b.customLoad ? b.customLoad(d, f, b, g) : L._fetchOne(d, f, b, g)
                    })
                })
            }
        },
        _fetchOne: function(a, b, c, d) {
            var e = r(c.resGetPath, {
                lng: a,
                ns: b
            });
            R.ajax({
                url: e,
                success: function(a) {
                    R.log("loaded: " + e), d(null, a)
                },
                error: function(a, b, c) {
                    if (b && 200 == b || a && a.status && 200 == a.status)
                        R.error("There is a typo in: " + e);
                    else if (b && 404 == b || a && a.status && 404 == a.status)
                        R.log("Does not exist: " + e);
                    else {
                        var f = b ? b: a && a.status ? a.status: null;
                        R.log(f + " when loading " + e)
                    }
                    d(c, {})
                },
                dataType: "json",
                async: c.getAsync
            })
        },
        postMissing: function(a, b, c, d, e) {
            var f = {};
            f[c] = d;
            var g = [];
            if ("fallback" === N.sendMissingTo && N.fallbackLng[0]!==!1)
                for (var h = 0; h < N.fallbackLng.length; h++)
                    g.push({
                        lng: N.fallbackLng[h],
                        url: r(N.resPostPath, {
                            lng: N.fallbackLng[h],
                            ns: b
                        })
                    });
            else if ("current" === N.sendMissingTo || "fallback" === N.sendMissingTo && N.fallbackLng[0]===!1)
                g.push({
                    lng: a,
                    url: r(N.resPostPath, {
                        lng: a,
                        ns: b
                    })
                });
            else if ("all" === N.sendMissingTo)
                for (var h = 0, i = e.length; i > h; h++)
                    g.push({
                        lng: e[h],
                        url: r(N.resPostPath, {
                            lng: e[h],
                            ns: b
                        })
                    });
            for (var j = 0, k = g.length; k > j; j++) {
                var l = g[j];
                R.ajax({
                    url: l.url,
                    type: N.sendType,
                    data: f,
                    success: function() {
                        R.log("posted missing key '" + c + "' to: " + l.url);
                        for (var a = c.split("."), e = 0, f = H[l.lng][b]; a[e];)
                            f = f[a[e]] = e === a.length - 1 ? d : f[a[e]] || {}, e++
                    },
                    error: function() {
                        R.log("failed posting missing key '" + c + "' to: " + l.url)
                    },
                    dataType: "json",
                    async: N.postAsync
                })
            }
        }
    };
    var N = {
        lng: void 0,
        load: "all",
        preload: [],
        lowerCaseLng: !1,
        returnObjectTrees: !1,
        fallbackLng: ["dev"],
        fallbackNS: [],
        detectLngQS: "setLng",
        detectLngFromLocalStorage: !1,
        ns: "translation",
        fallbackOnNull: !0,
        fallbackOnEmpty: !1,
        fallbackToDefaultNS: !1,
        nsseparator: ":",
        keyseparator: ".",
        selectorAttr: "data-i18n",
        debug: !1,
        resGetPath: "locales/__lng__/__ns__.json",
        resPostPath: "locales/add/__lng__/__ns__",
        getAsync: !0,
        postAsync: !0,
        resStore: void 0,
        useLocalStorage: !1,
        localStorageExpirationTime: 6048e5,
        dynamicLoad: !1,
        sendMissing: !1,
        sendMissingTo: "fallback",
        sendType: "POST",
        interpolationPrefix: "__",
        interpolationSuffix: "__",
        reusePrefix: "$t(",
        reuseSuffix: ")",
        pluralSuffix: "_plural",
        pluralNotFound: ["plural_not_found",
        Math.random()].join(""),
        contextNotFound: ["context_not_found",
        Math.random()].join(""),
        escapeInterpolation: !1,
        indefiniteSuffix: "_indefinite",
        indefiniteNotFound: ["indefinite_not_found",
        Math.random()].join(""),
        setJqueryExt: !0,
        defaultValueFromContent: !0,
        useDataAttrOptions: !1,
        cookieExpirationTime: void 0,
        useCookie: !0,
        cookieName: "i18next",
        cookieDomain: void 0,
        objectTreeKeyHandler: void 0,
        postProcess: void 0,
        parseMissingKey: void 0,
        missingKeyHandler: L.postMissing,
        shortcutFunction: "sprintf"
    }, O = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
        "/": "&#x2F;"
    }, P = {
        create: function(a, b, c, d) {
            var e;
            if (c) {
                var f = new Date;
                f.setTime(f.getTime() + 6e4 * c), e = "; expires=" + f.toGMTString()
            } else 
                e = "";
            d = d ? "domain=" + d + ";" : "", document.cookie = a + "=" + b + e + ";" + d + "path=/"
        },
        read: function(a) {
            for (var b = a + "=", c = document.cookie.split(";"), d = 0; d < c.length; d++) {
                for (var e = c[d]; " " == e.charAt(0);)
                    e = e.substring(1, e.length);
                if (0 === e.indexOf(b))
                    return e.substring(b.length, e.length)
            }
            return null
        },
        remove: function(a) {
            this.create(a, "", - 1)
        }
    }, Q = {
        create: function() {},
        read: function() {
            return null
        },
        remove: function() {}
    }, R = {
        extend: F ? F.extend: a,
        deepExtend: b,
        each: F ? F.each: c,
        ajax: F ? F.ajax: "undefined" != typeof document ? e: function() {},
        cookie: "undefined" != typeof document ? P: Q,
        detectLanguage: C,
        escape: d,
        log: function(a) {
            N.debug && "undefined" != typeof console && console.log(a)
        },
        error: function(a) {
            "undefined" != typeof console && console.error(a)
        },
        getCountyIndexOfLng: function(a) {
            var b = 0;
            return ("nb-NO" === a || "nn-NO" === a) && (b = 1), b
        },
        toLanguages: function(a) {
            var b = this.log, c = [], d = N.lngWhitelist ||!1, e = function(a) {
                !d || d.indexOf(a)>-1 ? c.push(a) : b("rejecting non-whitelisted language: " + a)
            };
            if ("string" == typeof a && a.indexOf("-")>-1) {
                var f = a.split("-");
                a = N.lowerCaseLng ? f[0].toLowerCase() + "-" + f[1].toLowerCase() : f[0].toLowerCase() + "-" + f[1].toUpperCase(), "unspecific" !== N.load && e(a), "current" !== N.load && e(f[this.getCountyIndexOfLng(a)])
            } else 
                e(a);
            for (var g = 0; g < N.fallbackLng.length; g++) 
                - 1 === c.indexOf(N.fallbackLng[g]) && N.fallbackLng[g] && c.push(N.fallbackLng[g]);
            return c
        },
        regexEscape: function(a) {
            return a.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, "\\$&")
        },
        regexReplacementEscape: function(a) {
            return "string" == typeof a ? a.replace(/\$/g, "$$$$") : a
        }
    };
    R.applyReplacement = r;
    var S = {
        rules: {
            ach: {
                name: "Acholi",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            af: {
                name: "Afrikaans",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ak: {
                name: "Akan",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            am: {
                name: "Amharic",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            an: {
                name: "Aragonese",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ar: {
                name: "Arabic",
                numbers: [0, 1, 2, 3, 11, 100],
                plurals: function(a) {
                    return Number(0 === a ? 0 : 1 == a ? 1 : 2 == a ? 2 : a%100 >= 3 && 10 >= a%100 ? 3 : a%100 >= 11 ? 4 : 5)
                }
            },
            arn: {
                name: "Mapudungun",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            ast: {
                name: "Asturian",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ay: {
                name: "Aymará",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            az: {
                name: "Azerbaijani",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            be: {
                name: "Belarusian",
                numbers: [1, 2, 5],
                plurals: function(a) {
                    return Number(1 == a%10 && 11 != a%100 ? 0 : a%10 >= 2 && 4 >= a%10 && (10 > a%100 || a%100 >= 20) ? 1 : 2)
                }
            },
            bg: {
                name: "Bulgarian",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            bn: {
                name: "Bengali",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            bo: {
                name: "Tibetan",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            br: {
                name: "Breton",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            bs: {
                name: "Bosnian",
                numbers: [1, 2, 5],
                plurals: function(a) {
                    return Number(1 == a%10 && 11 != a%100 ? 0 : a%10 >= 2 && 4 >= a%10 && (10 > a%100 || a%100 >= 20) ? 1 : 2)
                }
            },
            ca: {
                name: "Catalan",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            cgg: {
                name: "Chiga",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            cs: {
                name: "Czech",
                numbers: [1, 2, 5],
                plurals: function(a) {
                    return Number(1 == a ? 0 : a >= 2 && 4 >= a ? 1 : 2)
                }
            },
            csb: {
                name: "Kashubian",
                numbers: [1, 2, 5],
                plurals: function(a) {
                    return Number(1 == a ? 0 : a%10 >= 2 && 4 >= a%10 && (10 > a%100 || a%100 >= 20) ? 1 : 2)
                }
            },
            cy: {
                name: "Welsh",
                numbers: [1, 2, 3, 8],
                plurals: function(a) {
                    return Number(1 == a ? 0 : 2 == a ? 1 : 8 != a && 11 != a ? 2 : 3)
                }
            },
            da: {
                name: "Danish",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            de: {
                name: "German",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            dz: {
                name: "Dzongkha",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            el: {
                name: "Greek",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            en: {
                name: "English",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            eo: {
                name: "Esperanto",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            es: {
                name: "Spanish",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            es_ar: {
                name: "Argentinean Spanish",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            et: {
                name: "Estonian",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            eu: {
                name: "Basque",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            fa: {
                name: "Persian",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            fi: {
                name: "Finnish",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            fil: {
                name: "Filipino",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            fo: {
                name: "Faroese",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            fr: {
                name: "French",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a >= 2)
                }
            },
            fur: {
                name: "Friulian",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            fy: {
                name: "Frisian",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ga: {
                name: "Irish",
                numbers: [1, 2, 3, 7, 11],
                plurals: function(a) {
                    return Number(1 == a ? 0 : 2 == a ? 1 : 7 > a ? 2 : 11 > a ? 3 : 4)
                }
            },
            gd: {
                name: "Scottish Gaelic",
                numbers: [1, 2, 3, 20],
                plurals: function(a) {
                    return Number(1 == a || 11 == a ? 0 : 2 == a || 12 == a ? 1 : a > 2 && 20 > a ? 2 : 3)
                }
            },
            gl: {
                name: "Galician",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            gu: {
                name: "Gujarati",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            gun: {
                name: "Gun",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            ha: {
                name: "Hausa",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            he: {
                name: "Hebrew",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            hi: {
                name: "Hindi",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            hr: {
                name: "Croatian",
                numbers: [1, 2, 5],
                plurals: function(a) {
                    return Number(1 == a%10 && 11 != a%100 ? 0 : a%10 >= 2 && 4 >= a%10 && (10 > a%100 || a%100 >= 20) ? 1 : 2)
                }
            },
            hu: {
                name: "Hungarian",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            hy: {
                name: "Armenian",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ia: {
                name: "Interlingua",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            id: {
                name: "Indonesian",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            is: {
                name: "Icelandic",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a%10 || 11 == a%100)
                }
            },
            it: {
                name: "Italian",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ja: {
                name: "Japanese",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            jbo: {
                name: "Lojban",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            jv: {
                name: "Javanese",
                numbers: [0, 1],
                plurals: function(a) {
                    return Number(0 !== a)
                }
            },
            ka: {
                name: "Georgian",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            kk: {
                name: "Kazakh",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            km: {
                name: "Khmer",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            kn: {
                name: "Kannada",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ko: {
                name: "Korean",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            ku: {
                name: "Kurdish",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            kw: {
                name: "Cornish",
                numbers: [1, 2, 3, 4],
                plurals: function(a) {
                    return Number(1 == a ? 0 : 2 == a ? 1 : 3 == a ? 2 : 3)
                }
            },
            ky: {
                name: "Kyrgyz",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            lb: {
                name: "Letzeburgesch",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ln: {
                name: "Lingala",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            lo: {
                name: "Lao",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            lt: {
                name: "Lithuanian",
                numbers: [1, 2, 10],
                plurals: function(a) {
                    return Number(1 == a%10 && 11 != a%100 ? 0 : a%10 >= 2 && (10 > a%100 || a%100 >= 20) ? 1 : 2)
                }
            },
            lv: {
                name: "Latvian",
                numbers: [1, 2, 0],
                plurals: function(a) {
                    return Number(1 == a%10 && 11 != a%100 ? 0 : 0 !== a ? 1 : 2)
                }
            },
            mai: {
                name: "Maithili",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            mfe: {
                name: "Mauritian Creole",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            mg: {
                name: "Malagasy",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            mi: {
                name: "Maori",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            mk: {
                name: "Macedonian",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 == a || 1 == a%10 ? 0 : 1)
                }
            },
            ml: {
                name: "Malayalam",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            mn: {
                name: "Mongolian",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            mnk: {
                name: "Mandinka",
                numbers: [0, 1, 2],
                plurals: function(a) {
                    return Number(1 == a ? 1 : 2)
                }
            },
            mr: {
                name: "Marathi",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ms: {
                name: "Malay",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            mt: {
                name: "Maltese",
                numbers: [1, 2, 11, 20],
                plurals: function(a) {
                    return Number(1 == a ? 0 : 0 === a || a%100 > 1 && 11 > a%100 ? 1 : a%100 > 10 && 20 > a%100 ? 2 : 3)
                }
            },
            nah: {
                name: "Nahuatl",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            nap: {
                name: "Neapolitan",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            nb: {
                name: "Norwegian Bokmal",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ne: {
                name: "Nepali",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            nl: {
                name: "Dutch",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            nn: {
                name: "Norwegian Nynorsk",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            no: {
                name: "Norwegian",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            nso: {
                name: "Northern Sotho",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            oc: {
                name: "Occitan",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            or: {
                name: "Oriya",
                numbers: [2, 1],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            pa: {
                name: "Punjabi",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            pap: {
                name: "Papiamento",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            pl: {
                name: "Polish",
                numbers: [1, 2, 5],
                plurals: function(a) {
                    return Number(1 == a ? 0 : a%10 >= 2 && 4 >= a%10 && (10 > a%100 || a%100 >= 20) ? 1 : 2)
                }
            },
            pms: {
                name: "Piemontese",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ps: {
                name: "Pashto",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            pt: {
                name: "Portuguese",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            pt_br: {
                name: "Brazilian Portuguese",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            rm: {
                name: "Romansh",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ro: {
                name: "Romanian",
                numbers: [1, 2, 20],
                plurals: function(a) {
                    return Number(1 == a ? 0 : 0 === a || a%100 > 0 && 20 > a%100 ? 1 : 2)
                }
            },
            ru: {
                name: "Russian",
                numbers: [1, 2, 5],
                plurals: function(a) {
                    return Number(1 == a%10 && 11 != a%100 ? 0 : a%10 >= 2 && 4 >= a%10 && (10 > a%100 || a%100 >= 20) ? 1 : 2)
                }
            },
            sah: {
                name: "Yakut",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            sco: {
                name: "Scots",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            se: {
                name: "Northern Sami",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            si: {
                name: "Sinhala",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            sk: {
                name: "Slovak",
                numbers: [1, 2, 5],
                plurals: function(a) {
                    return Number(1 == a ? 0 : a >= 2 && 4 >= a ? 1 : 2)
                }
            },
            sl: {
                name: "Slovenian",
                numbers: [5, 1, 2, 3],
                plurals: function(a) {
                    return Number(1 == a%100 ? 1 : 2 == a%100 ? 2 : 3 == a%100 || 4 == a%100 ? 3 : 0)
                }
            },
            so: {
                name: "Somali",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            son: {
                name: "Songhay",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            sq: {
                name: "Albanian",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            sr: {
                name: "Serbian",
                numbers: [1, 2, 5],
                plurals: function(a) {
                    return Number(1 == a%10 && 11 != a%100 ? 0 : a%10 >= 2 && 4 >= a%10 && (10 > a%100 || a%100 >= 20) ? 1 : 2)
                }
            },
            su: {
                name: "Sundanese",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            sv: {
                name: "Swedish",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            sw: {
                name: "Swahili",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            ta: {
                name: "Tamil",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            te: {
                name: "Telugu",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            tg: {
                name: "Tajik",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            th: {
                name: "Thai",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            ti: {
                name: "Tigrinya",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            tk: {
                name: "Turkmen",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            tr: {
                name: "Turkish",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            tt: {
                name: "Tatar",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            ug: {
                name: "Uyghur",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            uk: {
                name: "Ukrainian",
                numbers: [1, 2, 5],
                plurals: function(a) {
                    return Number(1 == a%10 && 11 != a%100 ? 0 : a%10 >= 2 && 4 >= a%10 && (10 > a%100 || a%100 >= 20) ? 1 : 2)
                }
            },
            ur: {
                name: "Urdu",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            uz: {
                name: "Uzbek",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            vi: {
                name: "Vietnamese",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            wa: {
                name: "Walloon",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(a > 1)
                }
            },
            wo: {
                name: "Wolof",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            },
            yo: {
                name: "Yoruba",
                numbers: [1, 2],
                plurals: function(a) {
                    return Number(1 != a)
                }
            },
            zh: {
                name: "Chinese",
                numbers: [1],
                plurals: function() {
                    return 0
                }
            }
        },
        addRule: function(a, b) {
            S.rules[a] = b
        },
        setCurrentLng: function(a) {
            if (!S.currentRule || S.currentRule.lng !== a) {
                var b = a.split("-");
                S.currentRule = {
                    lng: a,
                    rule: S.rules[b[0]]
                }
            }
        },
        needsPlural: function(a, b) {
            var c, d = a.split("-");
            return c = S.currentRule && S.currentRule.lng === a ? S.currentRule.rule : S.rules[d[R.getCountyIndexOfLng(a)]], c && c.numbers.length <= 1?!1 : 1 !== this.get(a, b)
        },
        get: function(a, b) {
            function c(b, c) {
                var d;
                if (d = S.currentRule && S.currentRule.lng === a ? S.currentRule.rule : S.rules[b]) {
                    var e;
                    e = d.plurals(d.noAbs ? c : Math.abs(c));
                    var f = d.numbers[e];
                    return 2 === d.numbers.length && 1 === d.numbers[0] && (2 === f ? f =- 1 : 1 === f && (f = 1)), f
                }
                return 1 === c ? "1" : "-1"
            }
            var d = a.split("-");
            return c(d[R.getCountyIndexOfLng(a)], b)
        }
    }, T = {}, U = function(a, b) {
        T[a] = b
    }, V = function() {
        function a(a) {
            return Object.prototype.toString.call(a).slice(8, - 1).toLowerCase()
        }
        function b(a, b) {
            for (var c = []; b > 0; c[--b] = a);
            return c.join("")
        }
        var c = function() {
            return c.cache.hasOwnProperty(arguments[0]) || (c.cache[arguments[0]] = c.parse(arguments[0])), c.format.call(null, c.cache[arguments[0]], arguments)
        };
        return c.format = function(c, d) {
            var e, f, g, h, i, j, k, l = 1, m = c.length, n = "", o = [];
            for (f = 0; m > f; f++)
                if (n = a(c[f]), "string" === n)
                    o.push(c[f]);
                else if ("array" === n) {
                    if (h = c[f], h[2])
                        for (e = d[l], g = 0; g < h[2].length; g++) {
                            if (!e.hasOwnProperty(h[2][g]))
                                throw V('[sprintf] property "%s" does not exist', h[2][g]);
                                e = e[h[2][g]]
                        } else 
                            e = h[1] ? d[h[1]] : d[l++];
                            if (/[^s]/.test(h[8]) && "number" != a(e))
                                throw V("[sprintf] expecting number but found %s", a(e));
                                switch (h[8]) {
                                case"b":
                                    e = e.toString(2);
                                    break;
                                case"c":
                                    e = String.fromCharCode(e);
                                    break;
                                case"d":
                                    e = parseInt(e, 10);
                                    break;
                                case"e":
                                    e = h[7] ? e.toExponential(h[7]) : e.toExponential();
                                    break;
                                case"f":
                                    e = h[7] ? parseFloat(e).toFixed(h[7]) : parseFloat(e);
                                    break;
                                case"o":
                                    e = e.toString(8);
                                    break;
                                case"s":
                                    e = (e = String(e)) && h[7] ? e.substring(0, h[7]) : e;
                                    break;
                                case"u":
                                    e = Math.abs(e);
                                    break;
                                case"x":
                                    e = e.toString(16);
                                    break;
                                case"X":
                                    e = e.toString(16).toUpperCase()
                                }
                                e = /[def]/.test(h[8]) && h[3] && e >= 0 ? "+" + e : e, j = h[4] ? "0" == h[4] ? "0" : h[4].charAt(1) : " ", k = h[6] - String(e).length, i = h[6] ? b(j, k) : "", o.push(h[5] ? e + i : i + e)
                            }
            return o.join("")
        }, c.cache = {}, c.parse = function(a) {
            for (var b = a, c = [], d = [], e = 0; b;) {
                if (null !== (c = /^[^\x25]+/.exec(b)))
                    d.push(c[0]);
                else if (null !== (c = /^\x25{2}/.exec(b)))
                    d.push("%");
                else {
                    if (null === (c = /^\x25(?:([1-9]\d*)\$|\(([^\)]+)\))?(\+)?(0|'[^$])?(-)?(\d+)?(?:\.(\d+))?([b-fosuxX])/.exec(b)))
                        throw "[sprintf] huh?";
                    if (c[2]) {
                        e|=1;
                        var f = [], g = c[2], h = [];
                        if (null === (h = /^([a-z_][a-z_\d]*)/i.exec(g)))
                            throw "[sprintf] huh?";
                        for (f.push(h[1]); "" !== (g = g.substring(h[0].length));)
                            if (null !== (h = /^\.([a-z_][a-z_\d]*)/i.exec(g)))
                                f.push(h[1]);
                            else {
                                if (null === (h = /^\[(\d+)\]/.exec(g)))
                                    throw "[sprintf] huh?";
                                    f.push(h[1])
                                }
                        c[2] = f
                    } else 
                        e|=2;
                    if (3 === e)
                        throw "[sprintf] mixing positional and named placeholders is not (yet) supported";
                    d.push(c)
                }
                b = b.substring(c[0].length)
            }
            return d
        }, c
    }(), W = function(a, b) {
        return b.unshift(a), V.apply(null, b)
    };
    U("sprintf", function(a, b, c) {
        return c.sprintf ? "[object Array]" === Object.prototype.toString.apply(c.sprintf) ? W(a, c.sprintf) : "object" == typeof c.sprintf ? V(a, c.sprintf) : a : a
    }), G.init = f, G.setLng = o, G.preload = g, G.addResourceBundle = h, G.addResource = j, G.addResources = k, G.removeResourceBundle = i, G.loadNamespace = m, G.loadNamespaces = n, G.setDefaultNamespace = l, G.t = x, G.translate = x, G.exists = w, G.detectLanguage = R.detectLanguage, G.pluralExtensions = S, G.sync = L, G.functions = R, G.lng = p, G.addPostProcessor = U, G.options = N
}(), function(e) {
    function J() {}
    function j(a, b, c) {
        if (a&&!b)
            var b = j(), c = b.e, e = b.h, a = /^(?:[\w0-9]+\:)?\/\//.test(a) ? 0 === a.indexOf("/") ? e + a: a: e + "//" + b.g + (0 === a.indexOf("/") ? a : 0 === a.indexOf("?") ? c + a : 0 === a.indexOf("#") ? c + b.f + a : c.replace(/[^\/]+$/g, "") + a);
        else 
            a = b ? a : d.href, (!p || c) && (a = a.replace(/^[^#]*/, "") || "#", a = d.protocol + "//" + d.host + k.basepath + a.replace(RegExp("^#[/]?(?:" + k.type + ")?"), ""));
        O.href = a;
        var a = /(?:([\w0-9]+:))?(?:\/\/(?:[^@]*@)?([^\/:\?#]+)(?::([0-9]+))?)?([^\?#]*)(?:(\?[^#]+)|\?)?(?:(#.*))?/.exec(O.href), b = a[2] + (a[3] ? ":" + a[3] : ""), c = a[4] || "/", e = a[5] || "", f = "#" === a[6] ? "": a[6] || "", g = c + e + f, h = c.replace(RegExp("^" + k.basepath, "i"), k.type) + e;
        return {
            a: a[1] + "//" + b + g,
            h: a[1],
            g: b,
            i: a[2],
            k: a[3] || "",
            e: c,
            f: e,
            b: f,
            c: g,
            j: h,
            d: h + f
        }
    }
    function Y(a) {
        var b = "";
        if (D)
            b += D.getItem(E);
        else {
            var c = m.cookie.split(E + "=");
            1 < c.length && (b += c.pop().split(";").shift() || "null")
        }
        try {
            q = a.parse(b) || {}
        } catch (e) {
            q = {}
        }
        u(v + "unload", function() {
            if (D)
                D.setItem(E, a.stringify(q));
            else {
                var b = {};
                (b[d.href] = g.state) && (m.cookie = E + "=" + a.stringify(b))
            }
        }, n)
    }
    function w(a, b, c, d) {
        var c = c || {
            set: J
        }, f=!c.set, g=!c.get, h = {
            configurable: i,
            set: function() {
                f = 1
            },
            get: function() {
                g = 1
            }
        };
        try {
            B(a, b, h), a[b] = a[b], B(a, b, c)
        } catch (j) {}
        if (!f ||!g)
            if (a.__defineGetter__ && (a.__defineGetter__(b, h.get), a.__defineSetter__(b, h.set), a[b] = a[b], c.get && a.__defineGetter__(b, c.get), c.set && a.__defineSetter__(b, c.set)), f && g || a !== e) {
                if (!f ||!g)
                    try {
                        try {
                            var k = F.create(a);
                            B(F.getPrototypeOf(k) === a ? k : a, b, c);
                            for (var m in a)
                                "function" == typeof a[m] && (k[m] = a[m].bind(a));
                                try {
                                    d.call(k, k, a)
                                } catch (o) {}
                                a = k
                            } catch (p) {
                                B(a.constructor.prototype, b, c)
                            }
                        } catch (q) {
                            return n
                        }
        } else {
            try {
                var r = a[b];
                a[b] = l
            } catch (s) {}
            if ("execScript"in e)
                e.execScript("Public " + b, "VBScript");
            else 
                try {
                    B(a, b, {
                        value: J
                    })
                } catch (t) {}
                a[b] = r
        }
        return a
    }
    function $(a, b, c) {
        return c = c || {}, a = a === K ? d : a, c.set = c.set || function(c) {
            a[b] = c
        }, c.get = c.get || function() {
            return a[b]
        }, c
    }
    function G(a, b) {
        var c = ("" + ("string" == typeof a ? a : a.type)).replace(/^on/, ""), d = y[c];
        if (d) {
            if (b = "string" == typeof a ? b : a, b.target == l)
                for (var f = ["target", "currentTarget", "srcElement", "type"]; a = f.pop();)
                    b = w(b, a, {
                        get: "type" === a ? function() {
                            return c
                        }
                        : function() {
                            return e
                        }
                    });
            (("popstate" === c ? e.onpopstate : e.onhashchange) || J).call(e, b);
            for (var f = 0, g = d.length; g > f; f++)
                d[f].call(e, b);
            return i
        }
        return aa(a, b)
    }
    function P() {
        var a = m.createEvent ? m.createEvent("Event"): m.createEventObject();
        a.initEvent ? a.initEvent("popstate", n, n) : a.type = "popstate", a.state = g.state, G(a)
    }
    function r(a, b, c, e) {
        p || (b = j(b), b.c !== j().c && (z = e, c ? d.replace("#" + b.d) : d.hash = b.d)), !H && a && (q[d.href] = a), C = n
    }
    function L(a) {
        if (z) {
            Q !== d.href && P();
            var a = a || e.event, b = j(z, i), c = j();
            a.oldURL || (a.oldURL = b.a, a.newURL = c.a), b.b !== c.b && G(a)
        }
        z = d.href
    }
    function R(a) {
        setTimeout(function() {
            u("popstate", function(a) {
                Q = d.href, H || (a = w(a, "state", {
                    get: function() {
                        return g.state
                    }
                })), G(a)
            }, n)
        }, 0), !p && a !== i && g.location && (S(g.location.hash), C && (C = n, P()))
    }
    function ba(a) {
        var a = a || e.event, b = a.target || a.srcElement, c = "defaultPrevented"in a ? a.defaultPrevented: a.returnValue === n;
        b && "A" === b.nodeName&&!c && (c = j(), b = j(b.getAttribute("href", 2)), c.a.split("#").shift() === b.a.split("#").shift() && (c.b !== b.b && (g.location.hash = b.b), S(b.b), a.preventDefault ? a.preventDefault() : a.returnValue = n))
    }
    function S(a) {
        var b = m.getElementById(a = (a || "").replace(/^#/, ""));
        b && b.id === a && "A" === b.nodeName && (a = b.getBoundingClientRect(), e.scrollTo(I.scrollLeft || 0, a.top + (I.scrollTop || 0) - (I.clientTop || 0)))
    }
    function ca() {
        function a(a) {
            var b, d = [], f = "VBHistoryClass" + (new Date).getTime() + c++, g = ["Class " + f];
            for (b in a)
                if (a.hasOwnProperty(b)) {
                    var h = a[b];
                    h && (h.get || h.set) ? (h.get && g.push("Public " + ("_" === b ? "Default " : "") + "Property Get [" + b + "]", "Call VBCVal([(accessors)].[" + b + "].get.call(me),[" + b + "])", "End Property"), h.set && g.push("Public Property Let [" + b + "](val)", h = "Call [(accessors)].[" + b + "].set.call(me,val)\nEnd Property", "Public Property Set [" + b + "](val)", h)) : (d.push(b), g.push("Public [" + b + "]"))
                }
            for (g.push("Private [(accessors)]", "Private Sub Class_Initialize()", "Set [(accessors)]=" + f + "FactoryJS()", "End Sub", "End Class", "Function " + f + "Factory()", "Set " + f + "Factory=New " + f, "End Function"), e.execScript(g.join("\n"), "VBScript"), e[f + "FactoryJS"] = function() {
                return a
            }, f = e[f + "Factory"](), g = 0; g < d.length; g++)
                f[d[g]] = a[d[g]];
            return f
        }
        function b(a) {
            var b = /[\\"\u0000-\u001f\u007f-\u009f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g, c = {
                "\b": "\\b",
                "	": "\\t",
                "\n": "\\n",
                "\f": "\\f",
                "\r": "\\r",
                '"': '\\"',
                "\\": "\\\\"
            };
            return b.test(a) ? '"' + a.replace(b, function(a) {
                return a in c ? c[a] : "\\u" + ("0000" + a.charCodeAt(0).toString(16)).slice( - 4)
            }) + '"' : '"' + a + '"'
        }
        var c = e.eval && eval("/*@cc_on 1;@*/");
        if (c&&!(7<+(/msie (\d+)/i.exec(navigator.userAgent) || [, 8])[1])) {
            var f = w, k = j().a, h = m.createElement("iframe");
            h.src = "javascript:true;", h = I.firstChild.appendChild(h).contentWindow, e.execScript("Public history\nFunction VBCVal(o,r) If IsObject(o) Then Set r=o Else r=o End If End Function", "VBScript"), s = {
                _: {
                    get: K.toString
                }
            }, g = {
                back: o.back,
                forward: o.forward,
                go: o.go,
                emulate: l,
                _: {
                    get: function() {
                        return "[object History]"
                    }
                }
            }, A = {
                parse: function(a) {
                    try {
                        return new Function("", "return " + a)()
                    } catch (b) {
                        return l
                    }
                },
                stringify: function(a) {
                    var c = (typeof a).charCodeAt(2);
                    if (114 === c)
                        a = b(a);
                    else if (109 === c)
                        a = isFinite(a) ? "" + a : "null";
                    else if (111 === c || 108 === c)
                        a = "" + a;
                    else if (106 === c)
                        if (a) {
                            var d = (c = "[object Array]" === F.prototype.toString.call(a)) ? "[": "{";
                            if (c)
                                for (var e = 0; e < a.length; e++)
                                    d += (0 == e ? "" : ",") + A.stringify(a[e]);
                            else 
                                for (e in a)
                                    a.hasOwnProperty(e) && (d += (1 == d.length ? "" : ",") + b(e) + ":" + A.stringify(a[e]));
                                    a = d + (c ? "]" : "}")
                        } else 
                            a = "null";
                    else 
                        a = "void 0";
                    return a
                }
            }, r = function(a, b, c, e, f) {
                var g = h.document, b = j(b);
                C = n, b.c !== j().c || f ? (z = e, c ? h.lfirst ? (history.back(), r(a, b.a, 0, e, 1)) : d.replace("#" + b.d) : (b.a != k || f) && (h.lfirst || (h.lfirst = 1, r(a, k, 0, e, 1)), g.open(), g.write('<script>lfirst=1;parent.location.hash="' + b.d.replace(/"/g, '\\"') + '";</script>'), g.close()), !f && a && (q[d.href] = a)) : a && (q[d.href] = a)
            }, w = function(b, c, d) {
                return f.apply(this, arguments) || (b === s ? s[c] = d : b === g ? (g[c] = d, "state" === c && (s = a(s), e.history = g = a(g))) : b[c] = d.get && d.get()), b
            }, setInterval(function() {
                var a = j().a;
                if (a != k) {
                    var b = m.createEventObject();
                    b.oldURL = k, b.newURL = k = a, b.type = "hashchange", L(b)
                }
            }, 100), e.JSON = A
        }
    }
    var i=!0, l = null, n=!1, m = e.document, I = m.documentElement, D = e.sessionStorage, F = e.Object, A = e.JSON, d = e.location, o = e.history, g = o, M = o.pushState, T = o.replaceState, p=!!M, H = "state"in o, B = F.defineProperty, s = w({}, "t") ? {} : m.createElement("a"), v = "", N = e.addEventListener ? "addEventListener" : (v = "on") && "attachEvent", U = e.removeEventListener ? "removeEventListener" : "detachEvent", V = e.dispatchEvent ? "dispatchEvent" : "fireEvent", u = e[N], W = e[U], aa = e[V], k = {
        basepath: "/",
        redirect: 0,
        type: "/"
    }, E = "__historyAPI__", O = m.createElement("a"), z = d.href, Q = "", C = n, q = {}, y = {}, da = {
        onhashchange: l,
        onpopstate: l
    }, X = {
        redirect: function(a, b) {
            if (k.basepath = b = b == l ? k.basepath : b, k.type = a = a == l ? k.type : a, e.top == e.self) {
                var c = j(l, n, i).c, f = d.pathname + d.search;
                p ? (f = f.replace(/([^\/])$/, "$1/"), c != b && RegExp("^" + b + "$", "i").test(f) && d.replace(c)) : f != b && (f = f.replace(/([^\/])\?/, "$1/?"), RegExp("^" + b, "i").test(f) && d.replace(b + "#" + f.replace(RegExp("^" + b, "i"), a) + d.hash))
            }
        },
        pushState: function(a, b, c) {
            M && M.apply(o, arguments), r(a, c)
        },
        replaceState: function(a, b, c) {
            delete q[d.href], T && T.apply(o, arguments), r(a, c, i)
        },
        location: {
            set: function(a) {
                e.location = a
            },
            get: function() {
                return p ? d : s
            }
        },
        state: {
            get: function() {
                return q[d.href] || l
            }
        }
    }, K = {
        assign: function(a) {
            0 === ("" + a).indexOf("#") ? r(l, a) : d.assign(a)
        },
        reload: function() {
            d.reload()
        },
        replace: function(a) {
            0 === ("" + a).indexOf("#") ? r(l, a, i) : d.replace(a)
        },
        toString: function() {
            return this.href
        },
        href: {
            get: function() {
                return j().a
            }
        },
        protocol: l,
        host: l,
        hostname: l,
        port: l,
        pathname: {
            get: function() {
                return j().e
            }
        },
        search: {
            get: function() {
                return j().f
            }
        },
        hash: {
            set: function(a) {
                r(l, ("" + a).replace(/^(#|)/, "#"), n, z)
            },
            get: function() {
                return j().b
            }
        }
    };
    (function() {
        var a = m.getElementsByTagName("script"), a = (a[a.length - 1] || {}).src || "";
        ( - 1 !== a.indexOf("?") ? a.split("?").pop() : "").replace(/(\w+)(?:=([^&]*))?/g, function(a, b, c) {
            k[b] = (c || ("basepath" === b ? "/" : "")).replace(/^(0|false)$/, "")
        }), ca(), u(v + "hashchange", L, n);
        var b = [K, s, da, e, X, g];
        H && delete X.state;
        for (var c = 0; c < b.length; c += 2)
            for (var d in b[c])
                if (b[c].hasOwnProperty(d))
                    if ("function" == typeof b[c][d])
                        b[c + 1][d] = b[c][d];
                    else {
                        if (a = $(b[c], d, b[c][d]), !w(b[c + 1], d, a, function(a, d) {
                            d === g && (e.history = g = b[c + 1] = a)
                        }))
                            return W(v + "hashchange", L, n), n;
                            b[c + 1] === e && (y[d] = y[d.substr(2)] = [])
                    }
        return k.redirect && g.redirect(), !H && A && Y(A), p || m[N](v + "click", ba, n), "complete" === m.readyState ? R(i) : (!p && j().c !== k.basepath && (C = i), u(v + "load", R, n)), i
    })() && (g.emulate=!p, e[N] = function(a, b, c) {
        a in y ? y[a].push(b) : 3 < arguments.length ? u(a, b, c, arguments[3]) : u(a, b, c)
    }, e[U] = function(a, b, c) {
        var d = y[a];
        if (d) {
            for (a = d.length; --a;)
                if (d[a] === b) {
                    d.splice(a, 1);
                    break
                }
        } else 
            W(a, b, c)
    }, e[V] = G)
}(window), !function(a, b) {
    "use strict";
    function c(a, b) {
        var c, d;
        b = b || {}, a = "raven" + a.substr(0, 1).toUpperCase() + a.substr(1), document.createEvent ? (c = document.createEvent("HTMLEvents"), c.initEvent(a, !0, !0)) : (c = document.createEventObject(), c.eventType = a);
        for (d in b)
            j(b, d) && (c[d] = b[d]);
        if (document.createEvent)
            document.dispatchEvent(c);
        else 
            try {
                document.fireEvent("on" + c.eventType.toLowerCase(), c)
        } catch (e) {}
    }
    function d(a) {
        this.name = "RavenConfigError", this.message = a
    }
    function e(a) {
        var b = O.exec(a), c = {}, e = 7;
        try {
            for (; e--;)
                c[N[e]] = b[e] || ""
        } catch (f) {
            throw new d("Invalid DSN: " + a)
        }
        if (c.pass)
            throw new d("Do not specify your private key in the DSN!");
        return c
    }
    function f(a) {
        return "undefined" == typeof a
    }
    function g(a) {
        return "function" == typeof a
    }
    function h(a) {
        return "string" == typeof a
    }
    function i(a) {
        for (var b in a)
            return !1;
        return !0
    }
    function j(a, b) {
        return Object.prototype.hasOwnProperty.call(a, b)
    }
    function k(a, b) {
        var c, d;
        if (f(a.length))
            for (c in a)
                j(a, c) && b.call(null, c, a[c]);
        else if (d = a.length)
            for (c = 0; d > c; c++)
                b.call(null, c, a[c])
    }
    function l() {
        I = "?sentry_version=4&sentry_client=raven-js/" + M.VERSION + "&sentry_key=" + G
    }
    function m(a, b) {
        var d = [];
        a.stack && a.stack.length && k(a.stack, function(a, b) {
            var c = n(b);
            c && d.push(c)
        }), c("handle", {
            stackInfo: a,
            options: b
        }), p(a.name, a.message, a.url, a.lineno, d, b)
    }
    function n(a) {
        if (a.url) {
            var b, c = {
                filename: a.url,
                lineno: a.line,
                colno: a.column,
                "function": a.func || "?"
            }, d = o(a);
            if (d) {
                var e = ["pre_context", "context_line", "post_context"];
                for (b = 3; b--;)
                    c[e[b]] = d[b]
            }
            return c.in_app=!(!L.includePaths.test(c.filename) || /(Raven|TraceKit)\./.test(c["function"]) || /raven\.(min\.)?js$/.test(c.filename)), c
        }
    }
    function o(a) {
        if (a.context && L.fetchContext) {
            for (var b = a.context, c=~~(b.length / 2), d = b.length, e=!1; d--;)
                if (b[d].length > 300) {
                    e=!0;
                    break
                }
            if (e) {
                if (f(a.column))
                    return;
                return [[], b[c].substr(a.column, 50), []]
            }
            return [b.slice(0, c), b[c], b.slice(c + 1)]
        }
    }
    function p(a, b, c, d, e, f) {
        var g, h;
        b += "", ("Error" !== a || b) && (L.ignoreErrors.test(b) || (e && e.length ? (c = e[0].filename || c, e.reverse(), g = {
            frames: e
        }) : c && (g = {
            frames: [{
                filename: c,
                lineno: d,
                in_app: !0
            }
            ]
        }), b = r(b, 100), L.ignoreUrls && L.ignoreUrls.test(c) || (!L.whitelistUrls || L.whitelistUrls.test(c)) && (h = d ? b + " at " + d : b, t(q({
            exception: {
                type: a,
                value: b
            },
            stacktrace: g,
            culprit: c,
            message: h
        }, f)))))
    }
    function q(a, b) {
        return b ? (k(b, function(b, c) {
            a[b] = c
        }), a) : a
    }
    function r(a, b) {
        return a.length <= b ? a : a.substr(0, b) + "…"
    }
    function s() {
        var a = {
            url: document.location.href,
            headers: {
                "User-Agent": navigator.userAgent
            }
        };
        return document.referrer && (a.headers.Referer = document.referrer), a
    }
    function t(a) {
        v() && (a = q({
            project: H,
            logger: L.logger,
            site: L.site,
            platform: "javascript",
            request: s()
        }, a), a.tags = q(L.tags, a.tags), a.extra = q(L.extra, a.extra), i(a.tags) && delete a.tags, i(a.extra) && delete a.extra, F && (a.user = F), g(L.dataCallback) && (a = L.dataCallback(a)), (!g(L.shouldSendCallback) || L.shouldSendCallback(a)) && (D = a.event_id || (a.event_id = x()), u(a)))
    }
    function u(a) {
        var b = new Image, d = E + I + "&sentry_data=" + encodeURIComponent(JSON.stringify(a));
        b.onload = function() {
            c("success", {
                data: a,
                src: d
            })
        }, b.onerror = b.onabort = function() {
            c("failure", {
                data: a,
                src: d
            })
        }, b.src = d
    }
    function v() {
        return K ? E?!0 : (a.console && console.error && console.error("Error: Raven has not been configured."), !1) : !1
    }
    function w(a) {
        for (var b, c = [], d = 0, e = a.length; e > d; d++)
            b = a[d], h(b) ? c.push(b.replace(/([.*+?^=!:${}()|\[\]\/\\])/g, "\\$1")) : b && b.source && c.push(b.source);
        return new RegExp(c.join("|"), "i")
    }
    function x() {
        return "xxxxxxxxxxxx4xxxyxxxxxxxxxxxxxxx".replace(/[xy]/g, function(a) {
            var b = 16 * Math.random() | 0, c = "x" == a ? b: 3 & b | 8;
            return c.toString(16)
        })
    }
    function y() {
        var b = a.RavenConfig;
        b && M.config(b.dsn, b.config).install()
    }
    var z = {
        remoteFetching: !1,
        collectWindowErrors: !0,
        linesOfContext: 7
    }, A = [].slice, B = "?";
    z.wrap = function(a) {
        function b() {
            try {
                return a.apply(this, arguments)
            } catch (b) {
                throw z.report(b), b
            }
        }
        return b
    }, z.report = function() {
        function c(a) {
            h(), o.push(a)
        }
        function d(a) {
            for (var b = o.length - 1; b >= 0; --b)
                o[b] === a && o.splice(b, 1)
        }
        function e() {
            i(), o = []
        }
        function f(a, b) {
            var c = null;
            if (!b || z.collectWindowErrors) {
                for (var d in o)
                    if (j(o, d))
                        try {
                            o[d].apply(null, [a].concat(A.call(arguments, 2)))
                } catch (e) {
                    c = e
                }
                if (c)
                    throw c
            }
        }
        function g(a, b, c, d, e) {
            var g = null;
            if (r)
                z.computeStackTrace.augmentStackTraceWithInitialElement(r, b, c, a), k();
            else if (e)
                g = z.computeStackTrace(e), f(g, !0);
            else {
                var h = {
                    url: b,
                    line: c,
                    column: d
                };
                h.func = z.computeStackTrace.guessFunctionName(h.url, h.line), h.context = z.computeStackTrace.gatherContext(h.url, h.line), g = {
                    message: a,
                    url: document.location.href,
                    stack: [h]
                }, f(g, !0)
            }
            return m ? m.apply(this, arguments) : !1
        }
        function h() {
            n || (m = a.onerror, a.onerror = g, n=!0)
        }
        function i() {
            n && (a.onerror = m, n=!1, m = b)
        }
        function k() {
            var a = r, b = p;
            p = null, r = null, q = null, f.apply(null, [a, !1].concat(b))
        }
        function l(b, c) {
            var d = A.call(arguments, 1);
            if (r) {
                if (q === b)
                    return;
                k()
            }
            var e = z.computeStackTrace(b);
            if (r = e, q = b, p = d, a.setTimeout(function() {
                q === b && k()
            }, e.incomplete ? 2e3 : 0), c!==!1)
                throw b
        }
        var m, n, o = [], p = null, q = null, r = null;
        return l.subscribe = c, l.unsubscribe = d, l.uninstall = e, l
    }(), z.computeStackTrace = function() {
        function b(b) {
            if (!z.remoteFetching)
                return "";
            try {
                var c = function() {
                    try {
                        return new a.XMLHttpRequest
                    } catch (b) {
                        return new a.ActiveXObject("Microsoft.XMLHTTP")
                    }
                }, d = c();
                return d.open("GET", b, !1), d.send(""), d.responseText
            } catch (e) {
                return ""
            }
        }
        function c(a) {
            if (!h(a))
                return [];
            if (!j(v, a)) {
                var c = "";
                - 1 !== a.indexOf(document.domain) && (c = b(a)), v[a] = c ? c.split("\n") : []
            }
            return v[a]
        }
        function d(a, b) {
            var d, e = /function ([^(]*)\(([^)]*)\)/, g = /['"]?([0-9A-Za-z$_]+)['"]?\s*[:=]\s*(function|eval|new Function)/, h = "", i = 10, j = c(a);
            if (!j.length)
                return B;
            for (var k = 0; i > k; ++k)
                if (h = j[b - k] + h, !f(h)) {
                    if (d = g.exec(h))
                        return d[1];
                        if (d = e.exec(h))
                            return d[1]
                }
            return B
        }
        function e(a, b) {
            var d = c(a);
            if (!d.length)
                return null;
            var e = [], g = Math.floor(z.linesOfContext / 2), h = g + z.linesOfContext%2, i = Math.max(0, b - g - 1), j = Math.min(d.length, b + h - 1);
            b -= 1;
            for (var k = i; j > k; ++k)
                f(d[k]) || e.push(d[k]);
            return e.length > 0 ? e : null
        }
        function g(a) {
            return a.replace(/[\-\[\]{}()*+?.,\\\^$|#]/g, "\\$&")
        }
        function i(a) {
            return g(a).replace("<", "(?:<|&lt;)").replace(">", "(?:>|&gt;)").replace("&", "(?:&|&amp;)").replace('"', '(?:"|&quot;)').replace(/\s+/g, "\\s+")
        }
        function k(a, b) {
            for (var d, e, f = 0, g = b.length; g > f; ++f)
                if ((d = c(b[f])).length && (d = d.join("\n"), e = a.exec(d)))
                    return {
                        url: b[f],
                        line: d.substring(0, e.index).split("\n").length,
                        column: e.index - d.lastIndexOf("\n", e.index) - 1
                    };
            return null
        }
        function l(a, b, d) {
            var e, f = c(b), h = new RegExp("\\b" + g(a) + "\\b");
            return d -= 1, f && f.length > d && (e = h.exec(f[d])) ? e.index : null
        }
        function m(b) {
            for (var c, d, e, f, h = [a.location.href], j = document.getElementsByTagName("script"), l = "" + b, m = /^function(?:\s+([\w$]+))?\s*\(([\w\s,]*)\)\s*\{\s*(\S[\s\S]*\S)\s*\}\s*$/, n = /^function on([\w$]+)\s*\(event\)\s*\{\s*(\S[\s\S]*\S)\s*\}\s*$/, o = 0; o < j.length; ++o) {
                var p = j[o];
                p.src && h.push(p.src)
            }
            if (e = m.exec(l)) {
                var q = e[1] ? "\\s+" + e[1]: "", r = e[2].split(",").join("\\s*,\\s*");
                c = g(e[3]).replace(/;$/, ";?"), d = new RegExp("function" + q + "\\s*\\(\\s*" + r + "\\s*\\)\\s*{\\s*" + c + "\\s*}")
            } else 
                d = new RegExp(g(l).replace(/\s+/g, "\\s+"));
            if (f = k(d, h))
                return f;
            if (e = n.exec(l)) {
                var s = e[1];
                if (c = i(e[2]), d = new RegExp("on" + s + "=[\\'\"]\\s*" + c + "\\s*[\\'\"]", "i"), f = k(d, h[0]))
                    return f;
                if (d = new RegExp(c), f = k(d, h))
                    return f
            }
            return null
        }
        function n(a) {
            if (!a.stack)
                return null;
            for (var b, c, g = /^\s*at (?:((?:\[object object\])?\S+(?: \[as \S+\])?) )?\(?((?:file|https?|chrome-extension):.*?):(\d+)(?::(\d+))?\)?\s*$/i, h = /^\s*(\S*)(?:\((.*?)\))?@((?:file|https?).*?):(\d+)(?::(\d+))?\s*$/i, i = a.stack.split("\n"), j = [], k = /^(.*) is undefined$/.exec(a.message), m = 0, n = i.length; n > m; ++m) {
                if (b = h.exec(i[m]))
                    c = {
                        url: b[3],
                        func: b[1] || B,
                        args: b[2] ? b[2].split(","): "",
                        line: + b[4],
                        column: b[5]?+b[5]: null
                    };
                else {
                    if (!(b = g.exec(i[m])))
                        continue;
                    c = {
                        url: b[2],
                        func: b[1] || B,
                        line: + b[3],
                        column: b[4]?+b[4]: null
                    }
                }
                !c.func && c.line && (c.func = d(c.url, c.line)), c.line && (c.context = e(c.url, c.line)), j.push(c)
            }
            return j.length ? (j[0].line&&!j[0].column && k ? j[0].column = l(k[1], j[0].url, j[0].line) : j[0].column || f(a.columnNumber) || (j[0].column = a.columnNumber + 1), {
                name: a.name,
                message: a.message,
                url: document.location.href,
                stack: j
            }) : null
        }
        function o(a) {
            for (var b, c = a.stacktrace, f = / line (\d+), column (\d+) in (?:<anonymous function: ([^>]+)>|([^\)]+))\((.*)\) in (.*):\s*$/i, g = c.split("\n"), h = [], i = 0, j = g.length; j > i; i += 2)
                if (b = f.exec(g[i])) {
                    var k = {
                        line: + b[1],
                        column: + b[2],
                        func: b[3] || b[4],
                        args: b[5] ? b[5].split(","): [],
                        url: b[6]
                    };
                    if (!k.func && k.line && (k.func = d(k.url, k.line)), k.line)
                        try {
                            k.context = e(k.url, k.line)
                        } catch (l) {}
                        k.context || (k.context = [g[i + 1]]), h.push(k)
            }
            return h.length ? {
                name: a.name,
                message: a.message,
                url: document.location.href,
                stack: h
            } : null
        }
        function p(b) {
            var f = b.message.split("\n");
            if (f.length < 4)
                return null;
            var g, h, l, m, n = /^\s*Line (\d+) of linked script ((?:file|https?)\S+)(?:: in function (\S+))?\s*$/i, o = /^\s*Line (\d+) of inline#(\d+) script in ((?:file|https?)\S+)(?:: in function (\S+))?\s*$/i, p = /^\s*Line (\d+) of function script\s*$/i, q = [], r = document.getElementsByTagName("script"), s = [];
            for (h in r)
                j(r, h)&&!r[h].src && s.push(r[h]);
            for (h = 2, l = f.length; l > h; h += 2) {
                var t = null;
                if (g = n.exec(f[h]))
                    t = {
                        url: g[2],
                        func: g[3],
                        line: + g[1]
                    };
                else if (g = o.exec(f[h])) {
                    t = {
                        url: g[3],
                        func: g[4]
                    };
                    var u =+ g[1], v = s[g[2] - 1];
                    if (v && (m = c(t.url))) {
                        m = m.join("\n");
                        var w = m.indexOf(v.innerText);
                        w >= 0 && (t.line = u + m.substring(0, w).split("\n").length)
                    }
                } else if (g = p.exec(f[h])) {
                    var x = a.location.href.replace(/#.*$/, ""), y = g[1], z = new RegExp(i(f[h + 1]));
                    m = k(z, [x]), t = {
                        url: x,
                        line: m ? m.line: y,
                        func: ""
                    }
                }
                if (t) {
                    t.func || (t.func = d(t.url, t.line));
                    var A = e(t.url, t.line), B = A ? A[Math.floor(A.length / 2)]: null;
                    t.context = A && B.replace(/^\s*/, "") === f[h + 1].replace(/^\s*/, "") ? A : [f[h + 1]], q.push(t)
                }
            }
            return q.length ? {
                name: b.name,
                message: f[0],
                url: document.location.href,
                stack: q
            } : null
        }
        function q(a, b, c, f) {
            var g = {
                url: b,
                line: c
            };
            if (g.url && g.line) {
                a.incomplete=!1, g.func || (g.func = d(g.url, g.line)), g.context || (g.context = e(g.url, g.line));
                var h = / '([^']+)' /.exec(f);
                if (h && (g.column = l(h[1], g.url, g.line)), a.stack.length > 0 && a.stack[0].url === g.url) {
                    if (a.stack[0].line === g.line)
                        return !1;
                    if (!a.stack[0].line && a.stack[0].func === g.func)
                        return a.stack[0].line = g.line, a.stack[0].context = g.context, !1
                }
                return a.stack.unshift(g), a.partial=!0, !0
            }
            return a.incomplete=!0, !1
        }
        function r(a, b) {
            for (var c, e, f, g = /function\s+([_$a-zA-Z\xA0-\uFFFF][_$a-zA-Z0-9\xA0-\uFFFF]*)?\s*\(/i, h = [], i = {}, j=!1, k = r.caller; k&&!j; k = k.caller)
                if (k !== s && k !== z.report) {
                    if (e = {
                        url: null,
                        func: B,
                        line: null,
                        column: null
                    }, k.name ? e.func = k.name : (c = g.exec(k.toString())) && (e.func = c[1]), f = m(k)) {
                        e.url = f.url, e.line = f.line, e.func === B && (e.func = d(e.url, e.line));
                        var n = / '([^']+)' /.exec(a.message || a.description);
                        n && (e.column = l(n[1], f.url, f.line))
                    }
                    i["" + k] ? j=!0 : i["" + k]=!0, h.push(e)
                }
            b && h.splice(0, b);
            var o = {
                name: a.name,
                message: a.message,
                url: document.location.href,
                stack: h
            };
            return q(o, a.sourceURL || a.fileName, a.line || a.lineNumber, a.message || a.description), o
        }
        function s(a, b) {
            var c = null;
            b = null == b ? 0 : + b;
            try {
                if (c = o(a))
                    return c
                } catch (d) {
                if (u)
                    throw d
            }
            try {
                if (c = n(a))
                    return c
            } catch (d) {
                if (u)
                    throw d
            }
            try {
                if (c = p(a))
                    return c
            } catch (d) {
                if (u)
                    throw d
            }
            try {
                if (c = r(a, b + 1))
                    return c
            } catch (d) {
                if (u)
                    throw d
            }
            return {}
        }
        function t(a) {
            a = (null == a ? 0 : + a) + 1;
            try {
                throw new Error
            } catch (b) {
                return s(b, a + 1)
            }
        }
        var u=!1, v = {};
        return s.augmentStackTraceWithInitialElement = q, s.guessFunctionName = d, s.gatherContext = e, s.ofCaller = t, s
    }();
    var C, D, E, F, G, H, I, J = a.Raven, K=!(!a.JSON ||!a.JSON.stringify), L = {
        logger: "javascript",
        ignoreErrors: [],
        ignoreUrls: [],
        whitelistUrls: [],
        includePaths: [],
        collectWindowErrors: !0,
        tags: {},
        extra: {}
    }, M = {
        VERSION: "1.1.15",
        noConflict: function() {
            return a.Raven = J, M
        },
        config: function(a, b) {
            if (!a)
                return M;
            var c = e(a), d = c.path.lastIndexOf("/"), f = c.path.substr(1, d);
            return b && k(b, function(a, b) {
                L[a] = b
            }), L.ignoreErrors.push("Script error."), L.ignoreErrors.push("Script error"), L.ignoreErrors = w(L.ignoreErrors), L.ignoreUrls = L.ignoreUrls.length ? w(L.ignoreUrls) : !1, L.whitelistUrls = L.whitelistUrls.length ? w(L.whitelistUrls) : !1, L.includePaths = w(L.includePaths), G = c.user, H = c.path.substr(d + 1), E = "//" + c.host + (c.port ? ":" + c.port : "") + "/" + f + "api/" + H + "/store/", c.protocol && (E = c.protocol + ":" + E), L.fetchContext && (z.remoteFetching=!0), L.linesOfContext && (z.linesOfContext = L.linesOfContext), z.collectWindowErrors=!!L.collectWindowErrors, l(), M
        },
        install: function() {
            return v() && z.report.subscribe(m), M
        },
        context: function(a, c, d) {
            return g(a) && (d = c || [], c = a, a = b), M.wrap(a, c).apply(this, d)
        },
        wrap: function(a, c) {
            function d() {
                for (var b = [], d = arguments.length, e=!a || a && a.deep!==!1; d--;)
                    b[d] = e ? M.wrap(a, arguments[d]) : arguments[d];
                try {
                    return c.apply(this, b)
                } catch (f) {
                    throw M.captureException(f, a), f
                }
            }
            if (f(c)&&!g(a))
                return a;
            if (g(a) && (c = a, a = b), !g(c))
                return c;
            if (c.__raven__)
                return c;
            for (var e in c)
                j(c, e) && (d[e] = c[e]);
            return d.__raven__=!0, d.__inner__ = c, d
        },
        uninstall: function() {
            return z.report.uninstall(), M
        },
        captureException: function(a, b) {
            if (!(a instanceof Error))
                return M.captureMessage(a, b);
            C = a;
            try {
                z.report(a, b)
            } catch (c) {
                if (a !== c)
                    throw c
            }
            return M
        },
        captureMessage: function(a, b) {
            return t(q({
                message: a + ""
            }, b)), M
        },
        setUserContext: function(a) {
            return F = a, M
        },
        setExtraContext: function(a) {
            return L.extra = a || {}, M
        },
        setTagsContext: function(a) {
            return L.tags = a || {}, M
        },
        lastException: function() {
            return C
        },
        lastEventId: function() {
            return D
        }
    };
    M.setUser = M.setUserContext;
    var N = "source protocol user pass host port path".split(" "), O = /^(?:(\w+):)?\/\/(\w+)(:\w+)?@([\w\.-]+)(?::(\d+))?(\/.*)/;
    d.prototype = new Error, d.prototype.constructor = d, y(), a.Raven = M, "function" == typeof define && define.amd && define("raven", [], function() {
        return M
    })
}(this), function(a, b, c) {
    "use strict";
    function d(a) {
        return function(c, d, e) {
            var f = d._callback || d;
            return d = b.wrap(d), d._callback = f, a.call(this, c, d, e)
        }
    }
    if (c)
        for (var e = [c.Events, c, c.Model.prototype, c.Collection.prototype, c.View.prototype, c.Router.prototype, c.History.prototype], f = 0, g = e.length; g > f; f++) {
            var h = e[f];
            h.on = d(h.on), h.bind = h.on
        }
}(this, Raven, window.Backbone), function(a, b, c) {
    "use strict";
    if (c) {
        var d = c.event.add;
        c.event.add = function(a, e, f, g, h) {
            var i;
            return f && f.handler ? (i = f.handler, f.handler = b.wrap(f.handler)) : (i = f, f = b.wrap(f)), f.guid = i.guid ? i.guid : i.guid = c.guid++, d.call(this, a, e, f, g, h)
        };
        var e = c.fn.ready;
        c.fn.ready = function(a) {
            return e.call(this, b.wrap(a))
        };
        var f = c.ajax;
        c.ajax = function(a, d) {
            var e, g = ["complete", "error", "success"];
            for ("object" == typeof a && (d = a, a = void 0), d = d || {}; e = g.pop();)
                c.isFunction(d[e]) && (d[e] = b.wrap(d[e]));
            try {
                return f.call(this, a, d)
            } catch (h) {
                throw b.captureException(h), h
            }
        }
    }
}(this, Raven, window.jQuery), function(a, b) {
    "use strict";
    var c = function(c) {
        var d = a[c];
        a[c] = function() {
            var a = [].slice.call(arguments), c = a[0];
            return "function" == typeof c && (a[0] = b.wrap(c)), d.apply ? d.apply(this, a) : d(a[0], a[1])
        }
    };
    c("setTimeout"), c("setInterval")
}(this, Raven), function(a) {
    function b(a, b) {
        return function(c) {
            return i(a.call(this, c), b)
        }
    }
    function c(a) {
        return function(b) {
            return this.lang().ordinal(a.call(this, b))
        }
    }
    function d() {}
    function e(a) {
        g(this, a)
    }
    function f(a) {
        var b = this._data = {}, c = a.years || a.year || a.y || 0, d = a.months || a.month || a.M || 0, e = a.weeks || a.week || a.w || 0, f = a.days || a.day || a.d || 0, g = a.hours || a.hour || a.h || 0, i = a.minutes || a.minute || a.m || 0, j = a.seconds || a.second || a.s || 0, k = a.milliseconds || a.millisecond || a.ms || 0;
        this._milliseconds = k + 1e3 * j + 6e4 * i + 36e5 * g, this._days = f + 7 * e, this._months = d + 12 * c, b.milliseconds = k%1e3, j += h(k / 1e3), b.seconds = j%60, i += h(j / 60), b.minutes = i%60, g += h(i / 60), b.hours = g%24, f += h(g / 24), f += 7 * e, b.days = f%30, d += h(f / 30), b.months = d%12, c += h(d / 12), b.years = c
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
    function j(a, b, c) {
        var d, e = b._milliseconds, f = b._days, g = b._months;
        e && a._d.setTime( + a + e * c), f && a.date(a.date() + f * c), g && (d = a.date(), a.date(1).month(a.month() + g * c).date(Math.min(d, a.daysInMonth())))
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
    function m(a, b) {
        return b.abbr = a, J[a] || (J[a] = new d), J[a].set(b), J[a]
    }
    function n(a) {
        return a ? (!J[a] && K && require("./lang/" + a), J[a]) : F.fn._lang
    }
    function o(a) {
        return a.match(/\[.*\]/) ? a.replace(/^\[|\]$/g, "") : a.replace(/\\/g, "")
    }
    function p(a) {
        var b, c, d = a.match(M);
        for (b = 0, c = d.length; c > b; b++)
            d[b] = eb[d[b]] ? eb[d[b]] : o(d[b]);
        return function(e) {
            var f = "";
            for (b = 0; c > b; b++)
                f += "function" == typeof d[b].call ? d[b].call(e, a) : d[b];
            return f
        }
    }
    function q(a, b) {
        function c(b) {
            return a.lang().longDateFormat(b) || b
        }
        for (var d = 5; d--&&N.test(b);)
            b = b.replace(N, c);
        return bb[b] || (bb[b] = p(b)), bb[b](a)
    }
    function r(a) {
        switch (a) {
        case"DDDD":
            return Q;
        case"YYYY":
            return R;
        case"YYYYY":
            return S;
        case"S":
        case"SS":
        case"SSS":
        case"DDD":
            return P;
        case"MMM":
        case"MMMM":
        case"dd":
        case"ddd":
        case"dddd":
        case"a":
        case"A":
            return T;
        case"X":
            return W;
        case"Z":
        case"ZZ":
            return U;
        case"T":
            return V;
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
            return O;
        default:
            return new RegExp(a.replace("\\", ""))
        }
    }
    function s(a, b, c) {
        var d, e = c._a;
        switch (a) {
        case"M":
        case"MM":
            e[1] = null == b ? 0 : ~~b - 1;
            break;
        case"MMM":
        case"MMMM":
            d = n(c._l).monthsParse(b), null != d ? e[1] = d : c._isValid=!1;
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
            c._isPm = "pm" === (b + "").toLowerCase();
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
            c._useUTC=!0, d = (b + "").match($), d && d[1] && (c._tzh=~~d[1]), d && d[2] && (c._tzm=~~d[2]), d && "+" === d[0] && (c._tzh =- c._tzh, c._tzm =- c._tzm)
        }
        null == b && (c._isValid=!1)
    }
    function t(a) {
        var b, c, d = [];
        if (!a._d) {
            for (b = 0; 7 > b; b++)
                a._a[b] = d[b] = null == a._a[b] ? 2 === b ? 1 : 0 : a._a[b];
            d[3] += a._tzh || 0, d[4] += a._tzm || 0, c = new Date(0), a._useUTC ? (c.setUTCFullYear(d[0], d[1], d[2]), c.setUTCHours(d[3], d[4], d[5], d[6])) : (c.setFullYear(d[0], d[1], d[2]), c.setHours(d[3], d[4], d[5], d[6])), a._d = c
        }
    }
    function u(a) {
        var b, c, d = a._f.match(M), e = a._i;
        for (a._a = [], b = 0; b < d.length; b++)
            c = (r(d[b]).exec(e) || [])[0], c && (e = e.slice(e.indexOf(c) + c.length)), eb[d[b]] && s(d[b], c, a);
        a._isPm && a._a[3] < 12 && (a._a[3] += 12), a._isPm===!1 && 12 === a._a[3] && (a._a[3] = 0), t(a)
    }
    function v(a) {
        for (var b, c, d, f, h = 99; a._f.length;) {
            if (b = g({}, a), b._f = a._f.pop(), u(b), c = new e(b), c.isValid()) {
                d = c;
                break
            }
            f = l(b._a, c.toArray()), h > f && (h = f, d = c)
        }
        g(a, d)
    }
    function w(a) {
        var b, c = a._i;
        if (X.exec(c)) {
            for (a._f = "YYYY-MM-DDT", b = 0; 4 > b; b++)
                if (Z[b][1].exec(c)) {
                    a._f += Z[b][0];
                    break
                }
            U.exec(c) && (a._f += " Z"), u(a)
        } else 
            a._d = new Date(c)
    }
    function x(b) {
        var c = b._i, d = L.exec(c);
        c === a ? b._d = new Date : d ? b._d = new Date( + d[1]) : "string" == typeof c ? w(b) : k(c) ? (b._a = c.slice(0), t(b)) : b._d = new Date(c instanceof Date?+c : c)
    }
    function y(a, b, c, d, e) {
        return e.relativeTime(b || 1, !!c, a, d)
    }
    function z(a, b, c) {
        var d = I(Math.abs(a) / 1e3), e = I(d / 60), f = I(e / 60), g = I(f / 24), h = I(g / 365), i = 45 > d && ["s", d] || 1 === e && ["m"] || 45 > e && ["mm", e] || 1 === f && ["h"] || 22 > f && ["hh", f] || 1 === g && ["d"] || 25 >= g && ["dd", g] || 45 >= g && ["M"] || 345 > g && ["MM", I(g / 30)] || 1 === h && ["y"] || ["yy", h];
        return i[2] = b, i[3] = a > 0, i[4] = c, y.apply({}, i)
    }
    function A(a, b, c) {
        var d = c - b, e = c - a.day();
        return e > d && (e -= 7), d - 7 > e && (e += 7), Math.ceil(F(a).add("d", e).dayOfYear() / 7)
    }
    function B(a) {
        var b = a._i, c = a._f;
        return null === b || "" === b ? null : ("string" == typeof b && (a._i = b = n().preparse(b)), F.isMoment(b) ? (a = g({}, b), a._d = new Date( + b._d)) : c ? k(c) ? v(a) : u(a) : x(a), new e(a))
    }
    function C(a, b) {
        F.fn[a] = F.fn[a + "s"] = function(a) {
            var c = this._isUTC ? "UTC": "";
            return null != a ? (this._d["set" + c + b](a), this) : this._d["get" + c + b]()
        }
    }
    function D(a) {
        F.duration.fn[a] = function() {
            return this._data[a]
        }
    }
    function E(a, b) {
        F.duration.fn["as" + a] = function() {
            return + this / b
        }
    }
    if (!this.moment) {
        for (var F, G, H = "2.0.0", I = Math.round, J = {}, K = "undefined" != typeof module && module.exports, L = /^\/?Date\((\-?\d+)/i, M = /(\[[^\[]*\])|(\\)?(Mo|MM?M?M?|Do|DDDo|DD?D?D?|ddd?d?|do?|w[o|w]?|W[o|W]?|YYYYY|YYYY|YY|a|A|hh?|HH?|mm?|ss?|SS?S?|X|zz?|ZZ?|.)/g, N = /(\[[^\[]*\])|(\\)?(LT|LL?L?L?|l{1,4})/g, O = /\d\d?/, P = /\d{1,3}/, Q = /\d{3}/, R = /\d{1,4}/, S = /[+\-]?\d{1,6}/, T = /[0-9]*[a-z\u00A0-\u05FF\u0700-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]+|[\u0600-\u06FF]+\s*?[\u0600-\u06FF]+/i, U = /Z|[\+\-]\d\d:?\d\d/i, V = /T/i, W = /[\+\-]?\d+(\.\d{1,3})?/, X = /^\s*\d{4}-\d\d-\d\d((T| )(\d\d(:\d\d(:\d\d(\.\d\d?\d?)?)?)?)?([\+\-]\d\d:?\d\d)?)?/, Y = "YYYY-MM-DDTHH:mm:ssZ", Z = [["HH:mm:ss.S", /(T| )\d\d:\d\d:\d\d\.\d{1,3}/], ["HH:mm:ss", /(T| )\d\d:\d\d:\d\d/], ["HH:mm", /(T| )\d\d:\d\d/], ["HH", /(T| )\d\d/]], $ = /([\+\-]|\d\d)/gi, _ = "Month|Date|Hours|Minutes|Seconds|Milliseconds".split("|")
            , ab = {
            Milliseconds: 1,
            Seconds: 1e3,
            Minutes: 6e4,
            Hours: 36e5,
            Days: 864e5,
            Months: 2592e6,
            Years: 31536e6
        }, bb = {}, cb = "DDD w W M D d".split(" "), db = "M D H h m s w W".split(" "), eb = {
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
            X: function() {
                return this.unix()
            }
        };
        cb.length;
        )G = cb.pop(), eb[G + "o"] = c(eb[G]);
        for (; db.length;)
            G = db.pop(), eb[G + G] = b(eb[G], 2);
        for (eb.DDDD = b(eb.DDD, 3), d.prototype = {
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
                    if (this._monthsParse[b] || (c = F([2e3, b]), d = "^" + this.months(c, "") + "|^" + this.monthsShort(c, ""), this._monthsParse[b] = new RegExp(d.replace(".", ""), "i")), this._monthsParse[b].test(a))
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
            meridiem: function(a, b, c) {
                return a > 11 ? c ? "pm" : "PM" : c ? "am" : "AM"
            },
            _calendar: {
                sameDay: "[Today at] LT",
                nextDay: "[Tomorrow at] LT",
                nextWeek: "dddd [at] LT",
                lastDay: "[Yesterday at] LT",
                lastWeek: "[last] dddd [at] LT",
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
                return A(a, this._week.dow, this._week.doy)
            },
            _week: {
                dow: 0,
                doy: 6
            }
        }, F = function(a, b, c) {
            return B({
                _i: a,
                _f: b,
                _l: c,
                _isUTC: !1
            })
        }, F.utc = function(a, b, c) {
            return B({
                _useUTC: !0,
                _isUTC: !0,
                _l: c,
                _i: a,
                _f: b
            })
        }, F.unix = function(a) {
            return F(1e3 * a)
        }, F.duration = function(a, b) {
            var c, d = F.isDuration(a), e = "number" == typeof a, g = d ? a._data: e ? {}
            : a;
            return e && (b ? g[b] = a : g.milliseconds = a), c = new f(g), d && a.hasOwnProperty("_lang") && (c._lang = a._lang), c
        }, F.version = H, F.defaultFormat = Y, F.lang = function(a, b) {
            return a ? (b ? m(a, b) : J[a] || n(a), void(F.duration.fn._lang = F.fn._lang = n(a))) : F.fn._lang._abbr
        }, F.langData = function(a) {
            return a && a._lang && a._lang._abbr && (a = a._lang._abbr), n(a)
        }, F.isMoment = function(a) {
            return a instanceof e
        }, F.isDuration = function(a) {
            return a instanceof f
        }, F.fn = e.prototype = {
            clone: function() {
                return F(this)
            },
            valueOf: function() {
                return + this._d
            },
            unix: function() {
                return Math.floor( + this._d / 1e3)
            },
            toString: function() {
                return this.format("ddd MMM DD YYYY HH:mm:ss [GMT]ZZ")
            },
            toDate: function() {
                return this._d
            },
            toJSON: function() {
                return F.utc(this).format("YYYY-MM-DD[T]HH:mm:ss.SSS[Z]")
            },
            toArray: function() {
                var a = this;
                return [a.year(), a.month(), a.date(), a.hours(), a.minutes(), a.seconds(), a.milliseconds()]
            },
            isValid: function() {
                return null == this._isValid && (this._isValid = this._a?!l(this._a, (this._isUTC ? F.utc(this._a) : F(this._a)).toArray()) : !isNaN(this._d.getTime())), !!this._isValid
            },
            utc: function() {
                return this._isUTC=!0, this
            },
            local: function() {
                return this._isUTC=!1, this
            },
            format: function(a) {
                var b = q(this, a || F.defaultFormat);
                return this.lang().postformat(b)
            },
            add: function(a, b) {
                var c;
                return c = "string" == typeof a ? F.duration( + b, a) : F.duration(a, b), j(this, c, 1), this
            },
            subtract: function(a, b) {
                var c;
                return c = "string" == typeof a ? F.duration( + b, a) : F.duration(a, b), j(this, c, - 1), this
            },
            diff: function(a, b, c) {
                var d, e, f = this._isUTC ? F(a).utc(): F(a).local(), g = 6e4 * (this.zone() - f.zone());
                return b && (b = b.replace(/s$/, "")), "year" === b || "month" === b ? (d = 432e5 * (this.daysInMonth() + f.daysInMonth()), e = 12 * (this.year() - f.year()) + (this.month() - f.month()), e += (this - F(this).startOf("month") - (f - F(f).startOf("month"))) / d, "year" === b && (e/=12)) : (d = this - f - g, e = "second" === b ? d / 1e3 : "minute" === b ? d / 6e4 : "hour" === b ? d / 36e5 : "day" === b ? d / 864e5 : "week" === b ? d / 6048e5 : d), c ? e : h(e)
            },
            from: function(a, b) {
                return F.duration(this.diff(a)).lang(this.lang()._abbr).humanize(!b)
            },
            fromNow: function(a) {
                return this.from(F(), a)
            },
            calendar: function() {
                var a = this.diff(F().startOf("day"), "days", !0), b =- 6 > a ? "sameElse" : - 1 > a ? "lastWeek" : 0 > a ? "lastDay" : 1 > a ? "sameDay" : 2 > a ? "nextDay" : 7 > a ? "nextWeek" : "sameElse";
                return this.format(this.lang().calendar(b, this))
            },
            isLeapYear: function() {
                var a = this.year();
                return a%4 === 0 && a%100 !== 0 || a%400 === 0
            },
            isDST: function() {
                return this.zone() < F([this.year()]).zone() || this.zone() < F([this.year(), 5]).zone()
            },
            day: function(a) {
                var b = this._isUTC ? this._d.getUTCDay(): this._d.getDay();
                return null == a ? b : this.add({
                    d: a - b
                })
            },
            startOf: function(a) {
                switch (a = a.replace(/s$/, "")) {
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
                return "week" === a && this.day(0), this
            },
            endOf: function(a) {
                return this.startOf(a).add(a.replace(/s?$/, "s"), 1).subtract("ms", 1)
            },
            isAfter: function(a, b) {
                return b = "undefined" != typeof b ? b : "millisecond", + this.clone().startOf(b)>+F(a).startOf(b)
            },
            isBefore: function(a, b) {
                return b = "undefined" != typeof b ? b : "millisecond", + this.clone().startOf(b)<+F(a).startOf(b)
            },
            isSame: function(a, b) {
                return b = "undefined" != typeof b ? b : "millisecond", + this.clone().startOf(b) ===+ F(a).startOf(b)
            },
            zone: function() {
                return this._isUTC ? 0 : this._d.getTimezoneOffset()
            },
            daysInMonth: function() {
                return F.utc([this.year(), this.month() + 1, 0]).date()
            },
            dayOfYear: function(a) {
                var b = I((F(this).startOf("day") - F(this).startOf("year")) / 864e5) + 1;
                return null == a ? b : this.add("d", a - b)
            },
            isoWeek: function(a) {
                var b = A(this, 1, 4);
                return null == a ? b : this.add("d", 7 * (a - b))
            },
            week: function(a) {
                var b = this.lang().week(this);
                return null == a ? b : this.add("d", 7 * (a - b))
            },
            lang: function(b) {
                return b === a ? this._lang : (this._lang = n(b), this)
            }
        }, G = 0; G < _.length; G++)
            C(_[G].toLowerCase().replace(/s$/, ""), _[G]);
        C("year", "FullYear"), F.fn.days = F.fn.day, F.fn.weeks = F.fn.week, F.fn.isoWeeks = F.fn.isoWeek, F.duration.fn = f.prototype = {
            weeks: function() {
                return h(this.days() / 7)
            },
            valueOf: function() {
                return this._milliseconds + 864e5 * this._days + 2592e6 * this._months
            },
            humanize: function(a) {
                var b =+ this, c = z(b, !a, this.lang());
                return a && (c = this.lang().pastFuture(b, c)), this.lang().postformat(c)
            },
            lang: F.fn.lang
        };
        for (G in ab)
            ab.hasOwnProperty(G) && (E(G, ab[G]), D(G.toLowerCase()));
        E("Weeks", 6048e5), F.lang("en", {
            ordinal: function(a) {
                var b = a%10, c = 1===~~(a%100 / 10) ? "th": 1 === b ? "st": 2 === b ? "nd": 3 === b ? "rd": "th";
                return a + c
            }
        }), K && (module.exports = F), "undefined" == typeof ender && (this.moment = F), "function" == typeof define && define.amd && define("moment", [], function() {
            return F
        })
    }
}.call(this), function(a) {
    var b = {
        handle: {},
        timeout: {}
    };
    b.registerHandler = function(a, c) {
        b.handle[a] = b.handle[a] || [], b.handle[a].push(c)
    }, b.origin = "0.1.25", b.host = "pingback.issuu.com", b.timeout.general = 1500, b.timeout.documentRead = 2e3, b.timeout.pageReadIdle = 6e5, b.timeout.pageReadDelays = [2e3, 3e3, 4e3, 6e3, 8e3, 12e3], b.maxNumberOfEvents = 20, b.platform = function(a, c) {
        function d(c) {
            var d = 0 === a.location.protocol.indexOf("https") ? "https": "http";
            $.ajax({
                type: "POST",
                url: d + "://" + b.host + "/ping",
                data: c,
                crossDomain: !0,
                xhrFields: {
                    withCredentials: !0
                }
            })
        }
        function e(a) {
            console.log(a)
        }
        function f(b) {
            b = "string" == typeof b ? b : JSON.stringify(b), 0 === a.location.protocol.indexOf("file") ? e(b) : d(b)
        }
        function g() {
            var a = /site\.model\.username=([^;$]+)/, b = c.cookie.match(a);
            return b && b.length >= 2 ? b[1] : null
        }
        function h() {
            return a.innerWidth ? a.innerWidth : a.document.documentElement && a.document.documentElement.clientWidth ? a.document.documentElement.clientWidth : a.document.body ? a.document.body.clientWidth : 0
        }
        function i() {
            return a.innerHeight ? a.innerHeight : a.document.documentElement && a.document.documentElement.clientHeight ? a.document.documentElement.clientHeight : a.document.body ? a.document.body.clientHeight : 0
        }
        function j() {
            return a.location.href.toString()
        }
        function k() {
            return a.document.referrer || null
        }
        function l() {
            return - 1 !== (a.document.referrer || "").indexOf("issuu.com") ? "internal" : "external"
        }
        function m() {
            return ""
        }
        function n() {}
        var o = function() {
            return console && console.error ? console.error : n
        }(), p = function() {
            return console && console.log ? console.log : n
        }();
        return {
            global: a,
            post: f,
            getUsername: g,
            getWidth: h,
            getHeight: i,
            getLocation: j,
            getReferrer: k,
            getSource: l,
            getOriginPrefix: m,
            error: o,
            log: p,
            setTimeout: function(b, c) {
                return a.setTimeout(b, c)
            },
            clearTimeout: function(b) {
                return a.clearTimeout(b)
            },
            setInterval: function(b, c) {
                return a.setInterval(b, c)
            },
            clearInterval: function(b) {
                return a.clearInterval(b)
            }
        }
    }(window, document), b.clipEventsHandler = function() {
        function a(a) {
            var c = {
                type: "clipping_layer",
                action: a.action
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.timeoutPingback()
        }
        function c(a) {
            var c = {
                type: "clipping_action",
                clippingId: a.clippingId,
                action: a.action,
                on_page: a.onPage,
                creator: a.creator,
                url: a.url,
                page: a.page,
                service: a.service
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.timeoutPingback()
        }
        function d(a) {
            var c = {
                type: "clipping_comment",
                clippingId: a.clippingId,
                commentId: a.commentId,
                action: a.action,
                on_page: a.onPage,
                creator: a.creator,
                url: a.url
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.timeoutPingback()
        }
        function e() {}
        !function() {
            b.registerHandler("clippingLayer", a), b.registerHandler("clippingAction", c), b.registerHandler("clippingComment", d), b.registerHandler("flush", e)
        }()
    }(), b.documentEventsHandler = function() {
        function c() {
            v = b.timeout.pageReadDelays.slice(0), t = u = (new Date).getTime(), d(), e()
        }
        function d() {
            w = b.platform.setTimeout(function() {
                w = a, l(), d()
            }, g())
        }
        function e() {
            x = b.platform.setTimeout(function() {
                x = a, w && (b.platform.clearTimeout(w), w = a, l())
            }, b.timeout.pageReadIdle)
        }
        function f() {
            x && (b.platform.clearTimeout(x), x = a), w && (b.platform.clearTimeout(w), w = a, l()), s && (b.platform.clearTimeout(s), s = a)
        }
        function g() {
            var a = v.shift();
            if (isNaN(a)) {
                var c = b.timeout.pageReadDelays.length;
                return b.timeout.pageReadDelays[c - 1] + .2 * ((new Date).getTime() - t)
            }
            return a
        }
        function h() {
            var a = {
                type: "document_impression"
            };
            b.readerSignalHandler.updateSignal(a), b.readerSignalHandler.timeoutPingback()
        }
        function i() {
            for (; y.length > 0;)
                b.readerSignalHandler.updateSignal(y.shift());
            var a = {
                type: "document_read"
            };
            b.readerSignalHandler.updateSignal(a), b.readerSignalHandler.executePingback()
        }
        function j(a) {
            for (var c in a.pages) {
                var d = {
                    type: "page_impression",
                    page: a.pages[c]
                };
                b.readerSignalHandler.updateSignal(d)
            }
            b.readerSignalHandler.timeoutPingback()
        }
        function k(a) {
            for (var c in a.pages) {
                var d = {
                    type: "page_read",
                    page: a.pages[c]
                };
                s ? y.push(d) : b.readerSignalHandler.updateSignal(d)
            }
            b.readerSignalHandler.timeoutPingback()
        }
        function l() {
            var a = (new Date).getTime(), c = b.readerSignalHandler.getCurrentPages();
            for (var d in c) {
                var e = {
                    type: "page_read_time",
                    time_increment: Math.max(0, Math.floor((a - u) / c.length)),
                    page: c[d]
                };
                s ? y.push(e) : b.readerSignalHandler.updateSignal(e)
            }
            u = a, b.readerSignalHandler.timeoutPingback()
        }
        function m(d) {
            f(), y = [], b.readerSignalHandler.initializeModel(d), h(d), j(d), s = b.platform.setTimeout(function() {
                s = a, i(d)
            }, b.timeout.documentRead), k(d), c()
        }
        function n() {
            r(), b.readerSignalHandler.clearModel()
        }
        function o(d) {
            x && (b.platform.clearTimeout(x), x = a), w && (b.platform.clearTimeout(w), w = a, l()), s && (b.platform.clearTimeout(s), s = a, i()), b.readerSignalHandler.updateModel(d), j(d), k(d), c()
        }
        function p() {
            x ? (b.platform.clearTimeout(x), x = a) : (u = (new Date).getTime(), d()), e()
        }
        function q(a) {
            b.readerSignalHandler.updateModel(a)
        }
        function r() {
            f(), y = [], b.readerSignalHandler.executePingback()
        }
        var s, t, u, v, w, x, y = [];
        !function() {
            b.registerHandler("documentLoad", m), b.registerHandler("documentUnLoad", n), b.registerHandler("pageChange", o), b.registerHandler("zoom", p), b.registerHandler("resize", q), b.registerHandler("flush", r)
        }()
    }(), b.readerSignalHandler = function() {
        function c() {
            return {
                version: "1.3.1",
                username: b.platform.getUsername(),
                location: b.platform.getLocation(),
                referrer: b.platform.getReferrer(),
                source: b.platform.getSource(),
                contexts: []
            }
        }
        function d() {
            if (p) {
                var a=!1, d = c();
                b.forEach(["version", "username", "location", "referrer", "source"], function(b) {
                    a = a || p.data[b] !== d[b]
                }), a && m()
            }
        }
        function e() {
            return b.platform.getWidth() + "x" + b.platform.getHeight()
        }
        function f() {
            var a = p.data.contexts.length;
            if (0 === a)
                return !0;
            var c=!1;
            return b.forEach(["doc_id", "doc_creator", "doc_name", "ad_id", "ad_token", "embed_id", "display_size"], function(b) {
                c = c || p.data.contexts[a - 1][b] !== o[b]
            }), c = c || p.data.contexts[a - 1].pages.toString() !== o.pages.toString()
        }
        function g(a) {
            o = {
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase(),
                pages: a.pages,
                pageNumber: a.pageNumber,
                ad_id: a.adpageId || null,
                ad_token: a.adpageToken || null,
                embed_id: a.embedId || null,
                display_size: a.displaySize || e()
            }
        }
        function h() {
            m(), o = a
        }
        function i(a) {
            o ? (o.pages = a.pages || o.pages, o.display_size = a.displaySize || o.display_size) : b.requestHandler.error(new Error("reader pingback handler - update model - no model."))
        }
        function j(a) {
            if (!o)
                return void b.requestHandler.error(new Error("reader pingback handler - update signal - no model."));
            if (d(), p = p || b.signalWrapper.getSignalWrapper("reader"), p.data = p.data || c(), f() && p.data.contexts.push({
                doc_id: o.doc_id,
                doc_creator: o.doc_creator,
                doc_name: o.doc_name,
                pages: o.pages,
                ad_id: o.ad_id,
                ad_token: o.ad_token,
                embed_id: o.embed_id,
                display_size: o.display_size,
                display_state: null,
                events: []
            }), a) {
                var e = p.data.contexts.length;
                e > 0 ? p.data.contexts[e - 1].events.push(a) : b.requestHandler.error(new Error("reader pingback handler context length invalid."))
            }
        }
        function k() {
            return o ? o.pages : void b.requestHandler.error(new Error("reader pingback handler - get current pages - no model."))
        }
        function l() {
            if (p) {
                n();
                for (var a = 0, c = p.data.contexts.length - 1; c >= 0; c--)
                    a += p.data.contexts[c].events.length;
                a >= b.maxNumberOfEvents ? m() : q = b.platform.setTimeout(m, b.timeout.general)
            }
        }
        function m() {
            n(), p && (b.platform.post(p), p = a)
        }
        function n() {
            q && (b.platform.clearTimeout(q), q = a)
        }
        var o, p, q;
        return {
            initializeModel: g,
            clearModel: h,
            updateModel: i,
            updateSignal: j,
            getCurrentPages: k,
            timeoutPingback: l,
            executePingback: m
        }
    }(), b.linkEventsHandler = function() {
        function a(a) {
            var c = {
                type: "link_click",
                page: a.pageNumber,
                url: a.url,
                link_position: a.linkPosition
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.executePingback()
        }
        function c(a) {
            var c = {
                type: "watermark_click",
                url: a.url
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.executePingback()
        }
        function d(a) {
            var c = {
                type: "ad_cta_impression",
                cta_type: a.type,
                page: a.pageNumber
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.timeoutPingback()
        }
        function e(a) {
            var c = {
                type: "ad_cta_click",
                cta_type: a.type,
                page: a.pageNumber
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.executePingback()
        }
        function f() {}
        !function() {
            b.registerHandler("linkClick", a), b.registerHandler("watermarkClick", c), b.registerHandler("ctaLoad", d), b.registerHandler("ctaClick", e), b.registerHandler("flush", f)
        }()
    }(), b.readerSignalReference = {
        documentLoad: {
            revisionId: "string",
            publicationId: "string",
            publicationName: "string",
            ownerUsername: "string",
            pages: "arrayOfNumbers",
            pageNumber: "number",
            "?adpageId": "string",
            "?adpageToken": "string",
            "?embedId": "string"
        },
        documentUnLoad: "null",
        pageChange: {
            pages: "arrayOfNumbers",
            pageNumber: "number"
        },
        zoom: "null",
        resize: {
            displaySize: "string"
        },
        linkClick: {
            url: "string",
            pageNumber: "number",
            linkPosition: "string"
        },
        watermarkClick: {
            url: "string"
        },
        ctaLoad: {
            type: "string",
            pageNumber: "number"
        },
        ctaClick: {
            type: "string",
            pageNumber: "number"
        },
        adpageLoad: {
            revisionId: "string",
            publicationId: "string",
            ownerUsername: "string",
            publicationName: "string",
            pageNumber: "number",
            adpageId: "string",
            absolutePosition: "number",
            relativePosition: "number"
        },
        adpageClick: {
            revisionId: "string",
            publicationId: "string",
            ownerUsername: "string",
            publicationName: "string",
            adpageId: "string",
            adpageToken: "string",
            absolutePosition: "number",
            relativePosition: "number"
        },
        relatedLoad: {
            revisionId: "string",
            publicationId: "string",
            ownerUsername: "string",
            publicationName: "string",
            pageNumber: "number",
            absolutePosition: "number",
            relativePosition: "number"
        },
        relatedClick: {
            revisionId: "string",
            publicationId: "string",
            ownerUsername: "string",
            publicationName: "string",
            absolutePosition: "number",
            relativePosition: "number"
        },
        archiveLoad: {
            revisionId: "string",
            publicationId: "string",
            ownerUsername: "string",
            publicationName: "string",
            pageNumber: "number",
            absolutePosition: "number",
            relativePosition: "number"
        },
        archiveClick: {
            revisionId: "string",
            publicationId: "string",
            ownerUsername: "string",
            publicationName: "string",
            absolutePosition: "number",
            relativePosition: "number"
        },
        clippingLayer: {
            action: "string"
        },
        clippingComment: {
            onPage: "number",
            action: "string",
            clippingId: "string",
            commentId: "number",
            "?url": "string",
            creator: "string"
        },
        clippingAction: {
            clippingId: "string",
            action: "string",
            onPage: "number",
            creator: "string",
            "?url": "string",
            "?page": "number",
            "?service": "string"
        }
    }, b.sidebarEventsHandler = function() {
        function a(a) {
            var c = {
                type: "sidebar_ad_impression",
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase(),
                page: a.pageNumber,
                ad_id: a.adpageId,
                absolute_position: a.absolutePosition,
                relative_position: a.relativePosition
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.timeoutPingback()
        }
        function c(a) {
            var c = {
                type: "sidebar_ad_click",
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase(),
                ad_id: a.adpageId,
                token: a.adpageToken,
                absolute_position: a.absolutePosition,
                relative_position: a.relativePosition
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.executePingback()
        }
        function d(a) {
            var c = {
                type: "sidebar_related_impression",
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase(),
                page: a.pageNumber,
                absolute_position: a.absolutePosition,
                relative_position: a.relativePosition
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.timeoutPingback()
        }
        function e(a) {
            var c = {
                type: "sidebar_related_click",
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase(),
                absolute_position: a.absolutePosition,
                relative_position: a.relativePosition
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.executePingback()
        }
        function f(a) {
            var c = {
                type: "sidebar_archive_impression",
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase(),
                page: a.pageNumber,
                absolute_position: a.absolutePosition,
                relative_position: a.relativePosition
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.timeoutPingback()
        }
        function g(a) {
            var c = {
                type: "sidebar_archive_click",
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase(),
                absolute_position: a.absolutePosition,
                relative_position: a.relativePosition
            };
            b.readerSignalHandler.updateSignal(c), b.readerSignalHandler.executePingback()
        }
        function h() {}
        !function() {
            b.registerHandler("adpageLoad", a), b.registerHandler("adpageClick", c), b.registerHandler("relatedLoad", d), b.registerHandler("relatedClick", e), b.registerHandler("archiveLoad", f), b.registerHandler("archiveClick", g), b.registerHandler("flush", h)
        }()
    }(), b.websiteSignalHandler = function() {
        function c(a) {
            return {
                version: "2.1.0",
                username: a ? null: b.platform.getUsername(),
                location: b.platform.getLocation(),
                referrer: b.platform.getReferrer(),
                contexts: []
            }
        }
        function d() {
            if (C) {
                var a=!1, d = c();
                b.forEach(["version", "username", "location", "referrer"], function(b) {
                    a = a || C.data[b] !== d[b]
                }), a && i()
            }
        }
        function e() {
            return b.platform.getWidth() + "x" + b.platform.getHeight()
        }
        function f() {
            var a = C.data.contexts.length;
            return 0 === a?!0 : C.data.contexts[a - 1].browser_size !== e()?!0 : !1
        }
        function g(a) {
            if (d(), C = C || b.signalWrapper.getSignalWrapper("website"), C.data = C.data || c(), f() && C.data.contexts.push({
                browser_size: e(),
                events: []
            }), a) {
                var g = C.data.contexts.length;
                C.data.contexts[g - 1].events.push(a)
            }
        }
        function h() {
            D && (b.platform.clearTimeout(D), D = a);
            for (var c = 0, d = C.data.contexts.length - 1; d >= 0; d--)
                c += C.data.contexts[d].events.length;
            c >= b.maxNumberOfEvents ? i() : D = b.platform.setTimeout(i, b.timeout.general)
        }
        function i() {
            D && (b.platform.clearTimeout(D), D = a), C && (b.platform.post(C), C = a)
        }
        function j(a) {
            var d = b.signalWrapper.getSignalWrapper("website");
            d.data = c(!0), d.data.contexts.push({
                browser_size: e(),
                events: [a]
            }), b.platform.post(d)
        }
        function k() {
            E = {};
            var a = {
                type: "webpage_load"
            };
            g(a), h()
        }
        function l(a) {
            var b = {
                type: "webpage_continuation",
                index: a.index
            };
            g(b), h()
        }
        function m(a) {
            var b = {
                type: "document_share",
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase(),
                service: a.service
            };
            g(b), i()
        }
        function n(a) {
            var c = {
                type: "document_download",
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase()
            };
            a.anonymous && b.platform.getUsername() ? j(c) : (g(c), h())
        }
        function o(a) {
            return E[a] = E[a] || {}, E[a]
        }
        function p(a) {
            if (a.section) {
                var b = o(a.section), c = a.publicationId;
                if (b[c])
                    return;
                b[c]=!0
            }
            var d = {
                type: "document_impression",
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase(),
                ad_id: a.adpageId || null,
                page: a.pageNumber,
                stream_origin: a.streamOrigin || null,
                stream_ranking: a.streamRanking
            };
            g(d), h()
        }
        function q(a) {
            var b = {
                type: "document_click",
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase(),
                ad_id: a.adpageId || null,
                ad_token: a.adpageToken || null,
                stream_origin: a.streamOrigin || null,
                stream_ranking: a.streamRanking,
                click_coordinates: a.clickCoordinates || null
            };
            g(b), i()
        }
        function r(a) {
            var b = {
                type: "infobox_impression",
                id: a.infoboxId,
                stream_origin: a.streamOrigin || null,
                stream_ranking: a.streamRanking
            };
            g(b), h()
        }
        function s(a) {
            var b = {
                type: "infobox_click",
                id: a.infoboxId,
                link: a.link,
                stream_origin: a.streamOrigin || null,
                stream_ranking: a.streamRanking,
                click_coordinates: a.clickCoordinates || null
            };
            g(b), i()
        }
        function t(a) {
            var b = {
                type: "curated_impression",
                source: a.source,
                link: a.link,
                ranking: a.ranking
            };
            g(b), h()
        }
        function u(a) {
            var b = {
                type: "curated_click",
                source: a.source,
                link: a.link,
                ranking: a.ranking
            };
            g(b), i()
        }
        function v(a) {
            var b = {
                type: "embed_impression",
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase(),
                embed_id: a.embedId
            };
            g(b), i()
        }
        function w(a) {
            var b = {
                type: "embed_click",
                doc_id: a.revisionId + "-" + a.publicationId,
                doc_creator: a.ownerUsername.toLowerCase(),
                doc_name: a.publicationName.toLowerCase(),
                embed_id: a.embedId
            };
            g(b), i()
        }
        function x(a) {
            var b = {
                type: "stream_ad_request",
                advertiser: a.advertiser,
                placement: a.placement,
                stream_origin: a.streamOrigin,
                ranking: a.ranking
            };
            g(b), h()
        }
        function y(a) {
            var b = {
                type: "stream_ad_inserted",
                advertiser: a.advertiser,
                placement: a.placement,
                stream_origin: a.streamOrigin,
                ranking: a.ranking,
                timespan: a.timespan
            };
            g(b), h()
        }
        function z(a) {
            var b = {
                type: "stream_ad_not_available",
                advertiser: a.advertiser,
                placement: a.placement,
                stream_origin: a.streamOrigin,
                ranking: a.ranking,
                timespan: a.timespan
            };
            g(b), h()
        }
        function A(a) {
            var b = {
                type: "stream_ad_impression",
                advertiser: a.advertiser,
                placement: a.placement,
                stream_origin: a.streamOrigin,
                ranking: a.ranking
            };
            g(b), h()
        }
        function B() {
            i()
        }
        var C, D, E = {};
        !function() {
            b.registerHandler("webpageLoad", k), b.registerHandler("streamScrolling", l), b.registerHandler("documentShare", m), b.registerHandler("documentDownload", n), b.registerHandler("streamDocumentLoad", p), b.registerHandler("streamDocumentClick", q), b.registerHandler("infoboxLoad", r), b.registerHandler("infoboxClick", s), b.registerHandler("curatedLoad", t), b.registerHandler("curatedClick", u), b.registerHandler("embedLoad", v), b.registerHandler("embedClick", w), b.registerHandler("streamAdRequest", x), b.registerHandler("streamAdInserted", y), b.registerHandler("streamAdNotAvailable", z), b.registerHandler("streamAdImpression", A), b.registerHandler("flush", B)
        }()
    }(), b.websiteSignalReference = {
        webpageLoad: "null",
        streamScrolling: {
            index: "number"
        },
        documentShare: {
            revisionId: "string",
            publicationId: "string",
            publicationName: "string",
            ownerUsername: "string",
            service: "string"
        },
        documentDownload: {
            anonymous: "boolean",
            revisionId: "string",
            publicationId: "string",
            publicationName: "string",
            ownerUsername: "string"
        },
        streamDocumentLoad: {
            revisionId: "string",
            publicationId: "string",
            publicationName: "string",
            ownerUsername: "string",
            "?section": "string",
            "?adpageId": "string",
            pageNumber: "number",
            "?streamOrigin": "array",
            streamRanking: "number"
        },
        streamDocumentClick: {
            revisionId: "string",
            publicationId: "string",
            publicationName: "string",
            ownerUsername: "string",
            "?adpageId": "string",
            "?adpageToken": "string",
            "?streamOrigin": "array",
            streamRanking: "number",
            "?clickCoordinates": "string"
        },
        infoboxLoad: {
            infoboxId: "string",
            "?streamOrigin": "array",
            streamRanking: "number"
        },
        infoboxClick: {
            infoboxId: "string",
            link: "string",
            "?streamOrigin": "array",
            streamRanking: "number",
            "?clickCoordinates": "string"
        },
        curatedLoad: {
            source: "string",
            link: "string",
            ranking: "number"
        },
        curatedClick: {
            source: "string",
            link: "string",
            ranking: "number"
        },
        embedLoad: {
            revisionId: "string",
            publicationId: "string",
            publicationName: "string",
            ownerUsername: "string",
            embedId: "string"
        },
        embedClick: {
            revisionId: "string",
            publicationId: "string",
            publicationName: "string",
            ownerUsername: "string",
            embedId: "string"
        },
        streamAdRequest: {
            advertiser: "string",
            placement: "string",
            "?streamOrigin": "array",
            ranking: "number"
        },
        streamAdInserted: {
            advertiser: "string",
            placement: "string",
            "?streamOrigin": "array",
            ranking: "number",
            timespan: "number"
        },
        streamAdNotAvailable: {
            advertiser: "string",
            placement: "string",
            "?streamOrigin": "array",
            ranking: "number",
            timespan: "number"
        },
        streamAdImpression: {
            advertiser: "string",
            placement: "string",
            "?streamOrigin": "array",
            ranking: "number"
        }
    }, b.signalIgnoreHandler = function() {
        function a(a) {
            var c = b.signalWrapper.getSignalWrapper("ignore");
            c.data = a, b.platform.post(c)
        }
        function c() {}
        !function() {
            b.registerHandler("ignore", a), b.registerHandler("flush", c)
        }()
    }(), b.signalWrapper = function() {
        function a(a) {
            return {
                version: "2.0.0",
                origin: b.platform.getOriginPrefix() + "build" + b.origin,
                type: a
            }
        }
        return {
            getSignalWrapper: a
        }
    }(), b.monitorSignalHandler = function() {
        function a(a) {
            var c = a.type, d = b.signalWrapper.getSignalWrapper("monitor");
            d.data = b.extend(a, {
                version: "1.0.0",
                type: c
            }), b.platform.post(d)
        }
        !function() {
            b.registerHandler("monitor", a)
        }()
    }(), function(a) {
        function b(a) {
            return "function" == typeof a && j.test(a)
        }
        function c(a, b, c) {
            var d = l(a);
            if (b && d)
                if (Array.prototype.forEach)
                    for (var e =- 1, f = a.length, g = "object" == typeof c ? c : a; ++e < f && b.call(g, a[e], e, a)!==!1;);
                else 
                    Array.prototype.forEach.call(a, b, c);
            return a
        }
        function d(a, b, c) {
            var d = m(a);
            if (b && d) {
                var e = "object" == typeof c ? c: a;
                for (var f in a)
                    if (a.hasOwnProperty(f) && b.call(e, a[f], f, a)===!1)
                        break
            }
            return a
        }
        function e(a, b) {
            return d(b, function(b, c) {
                a[c] = b
            }), a
        }
        function f(a) {
            if (m(a) && arguments.length > 1)
                for (var b = 1; b < arguments.length; b++)
                    e(a, arguments[b]);
            return a
        }
        var g = Object.prototype.toString, h = g.call([]), i = String(g);
        i = i.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), i = i.replace(/toString| for [^\]]+/g, ".*?");
        var j = new RegExp("^" + i + "$"), k = b(k = Array.isArray) && k, l = k || function(a) {
            return a && "object" == typeof a && "number" == typeof a.length && g.call(a) === h ||!1
        }, m = function(a) {
            return a && "object" == typeof a&&!l(a)
        };
        a.isObject = m, a.isArray = l, a.forEach = c, a.forOwn = d, a.extend = f
    }(b), b.requestHandler = function() {
        function a(a, c) {
            var d=!1;
            if (b.handle[a])
                for (var e = b.handle[a].length - 1; e >= 0; e--)
                    b.handle[a][e](c), d=!0;
            return d
        }
        function c(c) {
            a("ignore", c);
            var d = /^pingback\.issuu\.com$/gi;
            if (!d.test(b.host)) {
                var e = b.platform.global.mocha && b.platform.global.chai;
                e || b.platform.error("string" == typeof c ? c : JSON.stringify(c))
            }
        }
        function d(a) {
            var d = b.platform.global.Raven;
            if (d && d.captureException)
                d.captureException(a);
            else {
                var e = a instanceof Error ? a.message: a;
                c(e)
            }
        }
        function e(d) {
            var e, f;
            if ("string" == typeof d)
                e = d;
            else {
                if ("object" != typeof d ||!d)
                    return void c({
                    message: "request MUST be a string or an object, got: " + JSON.stringify(d)
                });
                e = d.event, f = d.data
            }
            var g = b.requestValidator.validate(e, f);
            g ? c({
                message: g
            }) : a(e, f) || c({
                message: "Unknown action: " + e + ", please use this library according to the reference."
            })
        }
        return {
            handle: e,
            error: d
        }
    }(), b.requestValidator = function() {
        function c(a) {
            return function(c) {
                if (b.isArray(c)) {
                    var d=!0;
                    return b.forEach(c, function(b) {
                        return d = d && g(a, b)
                    }), d
                }
                return !1
            }
        }
        function d(a) {
            return /^\?(\w+)$/.test(a)
        }
        function e(b, c) {
            var d = /^[\?]?(\w+)$/.exec(c);
            return d && d[1] ? b[d[1]] : a
        }
        function f(c, d) {
            switch (c) {
            case"array":
                return b.isArray(d);
            case"null":
            case"undefined":
                return null === d || d === a;
            case"arrayOfNumbers":
                return j(d);
            default:
                return typeof d === c
            }
        }
        function g(a, c, h) {
            var i;
            return d(h) && f("null", c)?!0 : f("string", a) ? f(a, c) : "function" == typeof a?!!a(c) : b.isArray(a) ? (i=!1, b.forEach(a, function(a) {
                return i = i || a === c, !i
            }), i) : b.isObject(a) && b.isObject(c) ? (i=!0, b.forOwn(a, function(a, b) {
                var d = e(c, b);
                return i = i && g(a, d, b)
            }), i) : !1
        }
        function h(a, c) {
            if (i = i || b.extend({}, b.websiteSignalReference, b.readerSignalReference), i.hasOwnProperty(a)) {
                var d = e(i, a);
                if (!g(d, c, a))
                    return {
                        action: a,
                        expect: d,
                        got: c
                    }
            }
        }
        var i, j = c("number");
        return {
            validate: h
        }
    }(), b.init = function() {
        function a() {
            for (; 0 !== b.platform.global._tracq.length;)
                b.requestHandler.handle(b.platform.global._tracq.shift())
        }
        b.platform.global._tracq = b.platform.global._tracq || [], b.platform.global._tracq.push = function() {
            return Array.prototype.push.apply(this, arguments), a(), 0
        }, a()
    }, b.init()
}();
