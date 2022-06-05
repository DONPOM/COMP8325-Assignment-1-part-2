(function() {
    var ga = this;
    function na(c) {
        return void 0 !== c
    }
    function va(c) {
        c = c.split(".");
        for (var a = ga, b; b = c.shift();)
            if (null != a[b])
                a = a[b];
            else 
                return null;
        return a
    }
    function wa(c) {
        var a = typeof c;
        if ("object" == a)
            if (c) {
                if (c instanceof Array)
                    return "array";
                    if (c instanceof Object)
                        return a;
                        var b = Object.prototype.toString.call(c);
                        if ("[object Window]" == b)
                            return "object";
                            if ("[object Array]" == b || "number" == typeof c.length && "undefined" != typeof c.splice && "undefined" != typeof c.propertyIsEnumerable&&!c.propertyIsEnumerable("splice"))
                                return "array";
                                if ("[object Function]" == b || "undefined" != typeof c.call && "undefined" != typeof c.propertyIsEnumerable&&!c.propertyIsEnumerable("call"))
                                    return "function"
            } else 
                return "null";
                else if ("function" == a && "undefined" == typeof c.call)
    return "object";
return a
}
var xa = Date.now || function() {
    return + new Date
};
function ya(c, a) {
    var b = c.split("."), l = ga;
    b[0]in l ||!l.execScript || l.execScript("var " + b[0]);
    for (var e; b.length && (e = b.shift());)
        !b.length && na(a) ? l[e] = a : l[e] ? l = l[e] : l = l[e] = {}
};
var za, Ea, Fa, Ga, Ha, Ia, Ja, Ka, La, Ma, Na, Oa, Pa, Qa, Ra, Sa, Ta, Ua, Va, Ya, Za, $a, ab, bb, cb, db, eb, fb, gb, hb, kb, lb, mb, nb, ob, pb, qb, rb, sb, tb, ub, vb, wb, xb, yb, zb, Ab, Bb, Cb, Db, Hb, Ib, Jb, Kb, Lb, Mb, Nb, Ob, Pb, Qb, Rb, Sb, Tb, Ub, Vb, Wb, Xb, Yb = /^[6-9]$/, Zb = {
    ii: 0,
    Ue: 1,
    hi: 2,
    Jb: 3
}, $b = {
    EMPTY: 0,
    Mg: 1,
    ve: 2
}, ac = {
    ng: 1,
    og: 2,
    Dj: 3,
    mg: 4,
    pg: 5,
    Lj: 6,
    zj: 7,
    nb: 8
}, bc = {
    DONT_CARE: 1,
    lg: 2,
    qg: 3
}, cc = {
    Te: 0,
    ji: 1,
    Jb: 2
}, dc = [23, 24], g = {
    lh: 0,
    wj: 114,
    $: 115,
    Sa: 116,
    S: 117,
    Qa: 494,
    F: 118,
    na: 119,
    Z: 374,
    ca: 120,
    ha: 121,
    re: 122,
    T: 123,
    ua: 124,
    rb: 125,
    Gh: 230,
    Ij: 553,
    W: 126,
    ia: 127,
    P: 128,
    ue: 343,
    Ja: 129,
    vj: 231,
    ma: 130,
    hh: 131,
    xj: 237,
    yj: 570,
    Fj: 132,
    Db: 134,
    za: 189,
    xh: 246,
    Hj: 264,
    Gj: 256,
    ub: 133,
    Ng: 184,
    yh: 419,
    Mj: 503,
    Nj: 578,
    Oj: 579,
    Pj: 505,
    Sj: 516,
    Qj: 509,
    Rj: 512,
    oa: 173,
    Aj: 568,
    Bj: 569,
    Rd: 135,
    lc: 136,
    nc: 137,
    Vd: 138,
    Fa: 139,
    ih: 140,
    Sd: 141,
    Td: 142,
    Ud: 240,
    jh: 143,
    kh: 144,
    yb: 347,
    cd: 191,
    Hb: 150,
    Ia: 145,
    Ye: 146,
    Na: 147,
    Tj: 148,
    Jj: 245,
    De: 155,
    Ta: 149,
    kb: 154,
    je: 311,
    mc: 153,
    RENDERER: 152,
    Ha: 156,
    zb: 151,
    Tc: 158,
    te: 294,
    Og: 157,
    qa: 160,
    Cj: 328,
    Uj: 580,
    Cb: 159
}, ec = {
    Ee: 161,
    Fe: 162
};
function fc(c) {
    return {
        nd: function() {
            return c.nd()
        },
        Y: function() {
            return c.Y()
        },
        U: function() {
            return c.U()
        }
    }
};
(function() {
    function c(a) {
        for (var b = [], c = 0, e; e = a[c++];)
            b.push(e.api || {
                a: e.gd,
                b: e.K,
                c: e.wa,
                d: e.getType,
                e: e.Xc,
                f: e.ri,
                g: e.qi,
                i: e.cc,
                j: e.D,
                k: e.Lc,
                l: e.pi
            });
        return b
    }
    za = function(a) {
        var b = {};
        if (a)
            for (var c = 0; c < a.length; ++c)
                b[a[c]]=!0;
        return b
    };
    Ea = function(a) {
        var b = c(a.U());
        return a.api || {
            a: a.Y,
            b: function() {
                return b
            },
            c: a.nd
        }
    };
    Fa = function(a) {
        return a ? (a = a.toLowerCase(), "zh-tw" == a || "zh-cn" == a || "ja" == a || "ko" == a) : !1
    };
    Ga = function() {
        return (new Date).getTime()
    };
    Ha = function(a) {
        return "string" == typeof a
    };
    Ia = function(a) {
        return "number" ==
        typeof a
    }
})();
function gc() {
    return {
        getInstance: function() {
            return {
                clientName: "hp",
                pb: "hp",
                Jc: "google.com",
                we: "",
                Ya: "en",
                Od: "",
                Qd: "",
                jb: "",
                Ab: 0,
                jg: "",
                ic: "",
                ye: !1,
                se: "",
                Qc: "",
                Pc: 0,
                bj: null,
                ze: !1,
                Vi: !1,
                Pd: !1,
                hb: za([19, 5, 0]),
                Li: !1,
                Qf: !0,
                Hh: 10,
                ig: !0,
                Nd: !0,
                Ii: !1,
                Rf: !1,
                Jh: !1,
                rd: !1,
                Ti: !1,
                Qi: !1,
                Di: !0,
                Yi: "en",
                od: !0,
                cf: !1,
                lf: 500,
                Lb: !1,
                Ih: !0,
                Pi: !0,
                fb: !1,
                gb: "",
                oe: "//www.google.com/textinputassistant",
                pe: "",
                vg: 7,
                Mi: !1,
                Ni: !1,
                kf: !1,
                Uf: !0,
                Vf: !1,
                vd: !1,
                ff: !1,
                ef: !1,
                eb: 1,
                ud: !0,
                Ua: !1,
                Yb: !1,
                Kd: !1,
                Jf: 10,
                Fc: !1,
                Gi: 0,
                Oi: !1,
                Kf: !0,
                df: !1,
                Ra: document.body,
                Mf: !0,
                Yd: null,
                $a: {},
                Ji: {},
                Ui: 0,
                Gf: !1,
                Sf: !0,
                ta: !1,
                Ki: !1,
                Xi: null,
                Lf: !1,
                Yg: null,
                Zg: null,
                fe: !1,
                Wf: !0,
                Ef: !1,
                aj: 1,
                Ei: 1,
                spellcheck: !1,
                Ld: !1,
                Yf: "Search",
                gc: "I'm  Feeling Lucky",
                Zf: "",
                Xf: "Learn more",
                ne: "Remove",
                le: "This search was removed from your Web History",
                Ri: "",
                Hi: "Did you mean:",
                ug: "",
                Wi: "",
                Zi: "Search by voice",
                Ff: !1,
                Hf: null,
                Wb: 0,
                If: 0,
                fc: "",
                kc: "",
                Si: !1,
                Pa: "absolute",
                Tf: !1,
                ie: !1,
                Zb: null,
                Cd: !0,
                $i: 0,
                wb: [0, 0, 0],
                sg: null,
                mf: null,
                he: [0],
                Zd: 0,
                tg: 1,
                Rb: "",
                cg: "",
                $f: "",
                Oc: null,
                Qg: "",
                Pg: "",
                Fi: 1,
                Md: {},
                Af: !0
            }
        }
    }
};
var hc = /<\/?(?:b|em)>/gi, jc = {
    gh: 8,
    nb: 9,
    Ce: 13,
    Yc: 27,
    Kj: 32,
    bh: 37,
    fh: 38,
    eh: 39,
    $g: 40,
    DELETE: 46,
    Ej: 190
};
var Q = function() {
    function c(a, f, b) {
        k[a] = b;
        d[a] = [f]
    }
    function a(a, f, c) {
        var v = m[a];
        v ? v != b && (m[a] = b) : m[a] = c;
        (v = d[a]) ? v.push(f) : d[a] = [f];
        h[f] = c
    }
    var b = Ja, l = 0, e = {}, k = {}, h = {}, m = {}, d = {}, p = 1E4;
    return {
        va: function() {
            return l++
        },
        G: function() {
            return p++
        },
        $e: c,
        oc: function(a, f) {
            var b = p++;
            c(a, b, f);
            return b
        },
        register: a,
        uj: function(b, f) {
            var d = p++;
            a(b, d, f);
            return d
        },
        Xg: function() {
            return d
        },
        getInstance: function(a, f) {
            var d = e[a];
            return d ? d : (d = k[a]) ? e[a] = d() : f ? (d = h[f]) ? d() : null : (d = m[a]) && d != b ? d() : null
        }
    }
}();
function kc(c, a, b, l, e, k) {
    function h() {
        if (r) {
            for (var a = 0, f; f = s[a++];)
                f.N && f.N();
            r=!1
        }
    }
    function m(a) {
        for (var b in a) {
            var v = b, c = a[v];
            if (c != f.Ee)
                if (z[v]) {
                    for (var p = n[v] || [], e = 0, u = void 0; e < c.length; ++e)(u = d(v, c[e])
                        ) && p.push(u);
                        n[v] = p
                } else (c = d(v, c)
                    ) && (w[v] = c)
        }
    }
    function d(a, f) {
        var d;
        if (f && f instanceof Object)
            d = f;
        else if (d = t.getInstance(a, f), !d)
            return null;
        if (d.Ma) {
            var b = d.Ma();
            if (b)
                for (var v = 0, c, p, w; c = b[v++];) {
                    w=!1;
                    p = c.getType();
                    if (z[p]) {
                        if (w = x[p]) {
                            w.push(c);
                            continue
                        }
                        w=!0
                    }
                    x[p] = w ? [c] : c
                }
        }
        y.push([d, a]);
        s.push(d);
        return d
    }
    function p(a) {
        for (var f = g.lh, d = 0, b; b = y[d++];)
            b[0] == a && (f = b[1]);
        return f
    }
    function q(a, f) {
        var d = Ka(a.getType(), u), b = Ka(f.getType(), u);
        return 0 > d ? 1 : 0 > b?-1 : d - b
    }
    var f = ec, z = za([g.Cb, g.De, g.Ta, g.mc, g.kb, g.je, g.RENDERER, g.Ha, g.re, g.zb, g.Tc, g.te, g.qa]), v = [g.Ia, g.S, g.F, g.na, g.Z, g.W, g.$, g.Sa, g.ca, g.Na, g.ha, g.ub, g.T, g.ua, g.rb, g.ia, g.P, g.ue, g.Ja], u = [g.ia, g.Ta, g.Db, g.T, g.ha, g.W, g.F, g.$, g.P, g.qa, g.oa, g.na, g.Sa, g.RENDERER, g.mc, g.Ja, g.ca, g.Z, g.ua, g.Tc, g.De, g.hh, g.ma, g.Na, g.Sd, g.Td, g.nc, g.Ud, g.jh, g.Vd, g.kh,
    g.Fa, g.ih, g.Rd, g.lc], w = {}, n = {}, x = {}, y = [], s = [], r=!1, t = Q, G = {
        activate: function(a) {
            h();
            for (var f = 0, d; d = s[f++];)
                d.activate && d.activate(a);
            r=!0
        },
        N: h,
        isActive: function() {
            return r
        },
        get: function(a, f) {
            var d = w[a];
            if (d)
                return d.A ? d.A(p(f)) : {}
        },
        M: function(a, f) {
            var d = n[a];
            if (d) {
                for (var b = [], v = p(f), c = 0, w; w = d[c++];)
                    b.push(w.A ? w.A(v) : {});
                return b
            }
            return []
        },
        la: function() {
            return c
        },
        Ca: function() {
            return e
        },
        G: function(a, f) {
            var d = n[g.Cb];
            if (d)
                for (var b = 0, v; v = d[b++];)
                    if (v.B() == a)
                        return v.A ? v.A(p(f)) : {};
            return null
        }
    };
    (function() {
        if (k.Af) {
            var p =
            t.Xg(), e, u, h, n;
            for (n in p) {
                var y = n;
                e = p[y];
                u = z[y];
                if (h = a[y]) {
                    if (h != f.Ee && u && h.length) {
                        u = y;
                        h = h.slice(0);
                        for (var y = [], M = {}, R = 0, S = void 0, Z = void 0; Z = h[R++];)
                            Z instanceof Object && (S = Z.B(), M[S] || (y.push(Z), M[S] = 1), h.splice(--R, 1));
                        R = za(h);
                        R[f.Fe] && (R = za(h.concat(e)), delete R[f.Fe]);
                        for (S in R)
                            M[S] || y.push(parseInt(S, 10));
                        a[u] = y
                    }
                } else 
                    a[y] = u ? e : e[0]
            }
        }
        m(a);
        for (p = 0; n = v[p++];)
            a[n] || (u = d(n, void 0)) && (w[n] = u);
        m(x);
        s.sort(q);
        for (p = 0; n = s[p++];)
            n.O && n.O(b, l);
        c.Nc(l, b.Za());
        l.mh();
        for (p = 0; n = s[p++];)
            n.C && n.C(G);
        for (p =
        0; n = s[p++];)
            n.I && n.I(k);
        for (p = 0; n = s[p++];)
            n.activate && n.activate(k);
        r=!0
    })();
    return G
};
function lc(c, a, b) {
    function l() {
        return c
    }
    function e() {
        return u
    }
    function k() {
        return w
    }
    function h() {
        return a
    }
    function m() {
        return b || ""
    }
    function d(a, f) {
        z(a, f)
    }
    function p(a, f) {
        z(a, f, !0)
    }
    function q() {
        r || (t = G=!0)
    }
    function f() {
        J=!0
    }
    function z(a, f, d) {
        r || (t=!0, n[a] = f, d && (x[a] = f))
    }
    var v = Na(), u, w, n = {}, x = {}, y, s, r=!1, t=!1, G=!1, H=!1, J=!1, P = {
        getId: function() {
            return v
        },
        He: function() {
            var a = parseInt(v, 36);
            return isNaN(a)?-1 : a
        },
        Y: l,
        Pe: e,
        ja: k,
        Ga: h,
        D: function() {
            return n
        },
        Qe: function() {
            return y
        },
        oc: m,
        getTimestamp: function() {
            return s
        },
        zd: function() {
            return {
                Y: l,
                Pe: e,
                ja: k,
                Ga: h,
                oc: m,
                setParameter: d,
                $c: p,
                va: q,
                G: f
            }
        },
        setParameter: d,
        $c: p,
        va: q,
        G: f,
        uh: function() {
            return G
        },
        sh: function() {
            t = H=!0
        },
        dg: function(f, d, v) {
            return !t && c == f && a.equals(d) && b == v
        },
        Ie: function() {
            return H
        },
        Re: function() {
            return J
        },
        rh: function() {
            r || (s = Ga(), "cp"in x || p("cp", a.getPosition()), z("gs_id", v), y = La(x) + ":" + c, t = r=!0)
        }
    };
    u = c.toLowerCase();
    w = Ma(u);
    return P
};
function mc(c, a, b, l, e, k) {
    function h() {
        return !!a&&!!a[0]
    }
    var m, d=!0, p, q = {
        da: function() {
            return c
        },
        Y: function() {
            return c.Y()
        },
        Mb: function() {
            return h() ? a[0] : null
        },
        U: function() {
            return a
        },
        ga: h,
        D: function() {
            return b
        },
        Ae: function() {
            return l
        },
        fd: function() {
            return e
        },
        $h: function() {
            return k
        },
        getType: function() {
            return d
        },
        Be: function() {
            p || (p = fc(q));
            return p
        },
        nd: function() {
            return m
        }
    };
    a ? a.length && 33 == a[0].getType() && (e = d=!1) : a = [];
    b ? m = b.oi("t") : b = nc;
    return q
};
function oc(c, a, b, l, e, k) {
    function h(a) {
        if (e)
            for (var d = 0, f; f = a[d++];)
                if ( - 1 != Ka(f, e))
                    return !0;
        return !1
    }
    var m=!1, d = {
        gd: function() {
            return c
        },
        K: function() {
            return a
        },
        wa: function() {
            return b
        },
        getType: function() {
            return l
        },
        Lc: function() {
            return k.getString("za")
        },
        pi: function() {
            return k.getString("zb")
        },
        Xc: function() {
            return e || []
        },
        ri: function(a) {
            return !!e && h([a])
        },
        qi: h,
        D: function() {
            return k
        },
        cc: function() {
            return m
        }
    };
    switch (l) {
    case 0:
    case 32:
    case 38:
    case 39:
    case 400:
    case 407:
    case 35:
    case 33:
    case 41:
    case 34:
    case 44:
    case 45:
    case 40:
    case 46:
    case 56:
    case 30:
        m =
        !0
    }
    k || (k = nc);
    return d
};
(function() {
    var c = /\s/g, a = /\u3000/g, b = /^\s/, l = /\s+/, e = /\s+/g, k = /^\s+|\s+$/g, h = /^\s+$/, m = /<[^>]*>/g, d = /&nbsp;/g, p = /&#x3000;/g, q = [/&/g, /&amp;/g, /</g, /&lt;/g, />/g, /&gt;/g, /"/g, /&quot;/g, /'/g, /&#39;/g, /{/g, /&#123;/g], f = document.getElementsByTagName("head")[0], z = 0;
    Pa = function(a, f) {
        function d() {
            return f
        }
        void 0 === f && (f = a);
        return {
            getPosition: d,
            ge: function() {
                return a
            },
            ti: d,
            G: function() {
                return a < f
            },
            equals: function(d) {
                return d && a == d.ge() && f == d.ti()
            }
        }
    };
    Oa = function(a, f, d, b) {
        if (null == f || "" === f) {
            if (!b)
                return;
            f =
            ""
        }
        d.push(a + "=" + encodeURIComponent(String(f)))
    };
    La = function(a) {
        var f = [], d;
        for (d in a)
            Oa(d, a[d], f);
        return f.join("&")
    };
    Qa = function(a) {
        return !!a&&!h.test(a)
    };
    Ra = function(a) {
        for (var f = q.length, d = 0; d < f; d += 2)
            a = a.replace(q[d], q[d + 1].source);
        return a
    };
    Sa = function(a) {
        for (var f = q.length, b = 0; b < f; b += 2)
            a = a.replace(q[b + 1], q[b].source);
        a = a.replace(d, " ");
        return a.replace(p, "\u3000")
    };
    Ta = function(a) {
        return a.replace(hc, "")
    };
    Ua = function(a) {
        return a.replace(m, "")
    };
    Va = function(f) {
        return f && ( - 1 < f.indexOf(" ") || l.test(f)) ?
        (f = f.replace(a, "&#x3000;"), f.replace(c, "&nbsp;")) : f
    };
    Ma = function(a, f) {
        return a && ( - 1 < a.indexOf(" ") || l.test(a)) ? (a = a.replace(e, " "), a.replace(f ? k : b, "")) : a
    };
    Ya = function(a, f, d) {
        d && (a = a.toLowerCase(), f = f.toLowerCase());
        return f.length <= a.length && a.substring(0, f.length) == f
    };
    Za = function(a, f) {
        return a || f?!!a&&!!f && a.toLowerCase() == f.toLowerCase() : !0
    };
    $a = function(a) {
        window.clearTimeout(a)
    };
    Ja = function() {};
    ab = function() {
        return f
    };
    Na = function() {
        return (z++).toString(36)
    };
    bb = function(a) {
        return Yb.test(a)
    };
    Ka =
    function(a, f) {
        if (f.indexOf)
            return f.indexOf(a);
        for (var d = 0, b = f.length; d < b; ++d)
            if (f[d] === a)
                return d;
        return - 1
    };
    cb = function(a, f) {
        return a.V() - f.V()
    };
    db = function(a, f) {
        return f.V() - a.V()
    };
    eb = function(a) {
        var f = {}, d;
        for (d in a)
            f[d] = a[d];
        return f
    }
})();
function pc(c) {
    return {
        contains: function(a) {
            return a in c
        },
        G: function(a) {
            return !!c[a]
        },
        xe: function(a) {
            return c[a] || 0
        },
        getString: function(a) {
            return c[a] || ""
        },
        oi: function(a) {
            return c[a] || null
        }
    }
}
var nc = pc({});
(function() {
    function c(a, b, c) {
        a = document.createElement(a);
        b && (a.className = b);
        c && (a.id = c);
        return a
    }
    function a(a) {
        return c("div", a)
    }
    function b(a, b) {
        var c = a.getElementsByTagName("input");
        if (c)
            for (var f = 0, e; e = c[f++];)
                if (e.name == b && "submit" != e.type.toLowerCase())
                    return e;
        return null
    }
    function l(a) {
        a && (a.preventDefault && a.preventDefault(), a.returnValue=!1);
        return !1
    }
    function e(a) {
        return a ? a.ownerDocument || a.document : window.document
    }
    function k(a) {
        return a ? (a = e(a), a.defaultView || a.parentWindow) : window
    }
    var h =
    void 0 != document.documentElement.style.opacity, m = {
        rtl: "right",
        ltr: "left"
    };
    mb = function(a, b) {
        try {
            if (a.setSelectionRange)
                a.setSelectionRange(b, b);
            else if (a.createTextRange) {
                var c = a.createTextRange();
                c.collapse(!0);
                c.moveStart("character", b);
                c.select()
            }
        } catch (f) {}
    };
    nb = function(a) {
        try {
            var b, c;
            if ("selectionStart"in a)
                b = a.selectionStart, c = a.selectionEnd;
            else {
                var f = a.createTextRange(), h = e(a).selection.createRange();
                f.inRange(h) && (f.setEndPoint("EndToStart", h), b = f.text.length, f.setEndPoint("EndToEnd", h), c =
                f.text.length)
            }
            if (void 0 !== b)
                return Pa(b, c)
        } catch (v) {}
        return null
    };
    ob = function(a, b) {
        for (var c = 0, f = 0; a && (!b || a != b);) {
            c += a.offsetTop;
            f += a.offsetLeft;
            try {
                a = a.offsetParent
            } catch (e) {
                a = null
            }
        }
        return {
            jd: c,
            ka: f
        }
    };
    pb = function(a) {
        try {
            return e(a).activeElement == a
        } catch (b) {}
        return !1
    };
    qb = function(a) {
        return 38 == a || 40 == a
    };
    hb = c;
    rb = function() {
        var a = c("table");
        a.cellPadding = a.cellSpacing = 0;
        a.style.width = "100%";
        return a
    };
    sb = a;
    tb = function(b, c) {
        var e = a(b), f = e.style;
        f.background = "transparent";
        f.color = "#000";
        f.padding = 0;
        f.position = "absolute";
        c && (f.zIndex = c);
        f.whiteSpace = "pre";
        return e
    };
    ub = function(a, b) {
        a.innerHTML != b && (b && (fb ? b = Va(b) : gb && (b = '<pre style="font:inherit;margin:0">' + b + "</pre>")), a.innerHTML = b)
    };
    vb = function(a, b, c) {
        var f = a.style;
        "INPUT" != a.nodeName && (c += 1);
        f.left = f.right = "";
        f[b] = c + "px"
    };
    wb = function(a) {
        return "rtl" == a ? "right" : "left"
    };
    xb = function(a, b) {
        a.dir != b && (a.dir = b, a.style.textAlign = m[b])
    };
    yb = function(a, e, h) {
        if (b(a, e))
            return null;
        var f = c("input");
        f.type = "hidden";
        f.name = e;
        h && (f.value = h);
        return a.appendChild(f)
    };
    zb = b;
    Ab = function(a) {
        var b = document.createEvent("KeyboardEvent");
        b.initKeyEvent && (b.initKeyEvent("keypress", !0, !0, null, !1, !1, !0, !1, 27, 0), a.dispatchEvent(b))
    };
    Bb = l;
    Cb = function(a) {
        if (a = a || window.event)
            a.stopPropagation && a.stopPropagation(), a.cancelBubble = a.cancel=!0;
        return l(a)
    };
    Db = function(a, b) {
        b.parentNode.insertBefore(a, b.nextSibling)
    };
    Hb = function(a) {
        a = a.insertCell( - 1);
        var b = hb("a");
        b.href = "#ifl";
        b.className = "gssb_j gss_ifl";
        a.appendChild(b);
        return b
    };
    Ib = function(a, b) {
        var c = k(a);
        return (c = c.getComputedStyle ?
        c.getComputedStyle(a, "") : a.currentStyle) ? c[b] : null
    };
    Jb = function(a) {
        var b = a || window;
        a = b.document;
        var c = b.innerWidth, b = b.innerHeight;
        if (!c) {
            var f = a.documentElement;
            f && (c = f.clientWidth, b = f.clientHeight);
            c || (c = a.body.clientWidth, b = a.body.clientHeight)
        }
        return {
            Ke: c,
            Je: b
        }
    };
    Kb = function(a) {
        return (a || window).document.documentElement.clientWidth
    };
    Lb = function(a) {
        a = a.style;
        a.border = "none";
        a.padding = kb || lb ? "0 1px" : "0";
        a.margin = "0";
        a.height = "auto";
        a.width = "100%"
    };
    Mb = function(a) {
        return (h ? "opacity" : "filter") + ":" +
        (h ? a + "" : (fb ? "progid:DXImageTransform.Microsoft.Alpha(" : "alpha(") + "opacity=" + Math.floor(100 * a) + ")") + ";"
    };
    Nb = function(a) {
        var b = {};
        if (a)
            for (var c = 0, f; f = a[c++];)
                b[f.kd()] = f;
        return b
    };
    Ob = e;
    Pb = k;
    Qb = function(a) {
        kb && (a.tabIndex = 0)
    }
})();
Q.$e(g.cd, 192, function() {
    function c(a) {
        Ha(a) && (a = l(a));
        var b = "";
        if (a) {
            for (var c = a.length, d = 0, e = 0, h = 0; c--;)
                for (e<<=8, e|=a[h++], d += 8; 6 <= d;)
                    var k = e>>d - 6 & 63, b = b + "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".charAt(k), d = d - 6;
            d && (k = e<<8>>d + 8 - 6 & 63, b += "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".charAt(k))
        }
        return b
    }
    function a(a) {
        var b = [];
        if (a)
            for (var c = 0, e = 0, h = 0; h < a.length; ++h) {
                var k = a.charCodeAt(h);
                if (32 > k || 127 < k ||!d[k - 32])
                    return [];
                    c<<=6;
                    c|=d[k - 32] - 1;
                    e += 6;
                    8 <= e &&
                    (b.push(c>>e - 8 & 255), e -= 8)
            }
        return b
    }
    function b(a, b) {
        var c = {};
        c.H = Array(4);
        c.buffer = Array(4);
        c.ni = Array(4);
        c.padding = Array(64);
        c.padding[0] = 128;
        for (var d = 1; 64 > d; ++d)
            c.padding[d] = 0;
        e(c);
        var d = Array(64), w;
        64 < b.length ? (e(c), h(c, b), w = m(c)) : w = b;
        for (var n = 0; n < w.length; ++n)
            d[n] = w[n]^92;
        for (n = w.length; 64 > n; ++n)
            d[n] = 92;
        e(c);
        for (n = 0; 64 > n; ++n)
            c.buffer[n] = d[n]^106;
        k(c, c.buffer);
        c.total = 64;
        h(c, l(a));
        w = m(c);
        e(c);
        k(c, d);
        c.total = 64;
        h(c, w);
        return m(c)
    }
    function l(a) {
        for (var b = [], c = 0, d = 0; d < a.length; ++d) {
            var e = a.charCodeAt(d);
            128 > e ? b[c++] = e : (2048 > e ? b[c++] = e>>6 | 192 : (b[c++] = e>>12 | 224, b[c++] = e>>6 & 63 | 128), b[c++] = e & 63 | 128)
        }
        return b
    }
    function e(a) {
        a.H[0] = 1732584193;
        a.H[1] = 4023233417;
        a.H[2] = 2562383102;
        a.H[3] = 271733878;
        a.Kb = a.total = 0
    }
    function k(a, b) {
        for (var c = a.ni, d = 0; 64 > d; d += 4)
            c[d / 4] = b[d] | b[d + 1]<<8 | b[d + 2]<<16 | b[d + 3]<<24;
        for (var e = a.H[0], d = a.H[1], h = a.H[2], k = a.H[3], l, m, r, t = 0; 64 > t; ++t)
            16 > t ? (l = k^d & (h^k), m = t) : 32 > t ? (l = h^k & (d^h), m = 5 * t + 1 & 15) : 48 > t ? (l = d^h^k, m = 3 * t + 5 & 15) : (l = h^(d|~k), m = 7 * t & 15), r = k, k = h, h = d, e = e + l + q[t] + c[m] & 4294967295, l = p[t],
            d = d + ((e<<l | e>>>32 - l) & 4294967295) & 4294967295, e = r;
        a.H[0] = a.H[0] + e & 4294967295;
        a.H[1] = a.H[1] + d & 4294967295;
        a.H[2] = a.H[2] + h & 4294967295;
        a.H[3] = a.H[3] + k & 4294967295
    }
    function h(a, b, c) {
        c || (c = b.length);
        a.total += c;
        for (var d = 0; d < c; ++d)
            a.buffer[a.Kb++] = b[d], 64 == a.Kb && (k(a, a.buffer), a.Kb = 0)
    }
    function m(a) {
        var b = Array(16), c = 8 * a.total, d = a.Kb;
        h(a, a.padding, 56 > d ? 56 - d : 64 - (d - 56));
        for (var e = 56; 64 > e; ++e)
            a.buffer[e] = c & 255, c>>>=8;
        k(a, a.buffer);
        for (e = d = 0; 4 > e; ++e)
            for (c = 0; 32 > c; c += 8)
                b[d++] = a.H[e]>>c & 255;
        return b
    }
    var d = [0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 0, 0, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 0, 0, 0, 0, 64, 0, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 0, 0, 0, 0, 0], p = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21], q = [3614090360, 3905402710, 606105819, 3250441966, 4118548399, 1200080426, 2821735955, 4249261313,
    1770035416, 2336552879, 4294925233, 2304563134, 1804603682, 4254626195, 2792965006, 1236535329, 4129170786, 3225465664, 643717713, 3921069994, 3593408605, 38016083, 3634488961, 3889429448, 568446438, 3275163606, 4107603335, 1163531501, 2850285829, 4243563512, 1735328473, 2368359562, 4294588738, 2272392833, 1839030562, 4259657740, 2763975236, 1272893353, 4139469664, 3200236656, 681279174, 3936430074, 3572445317, 76029189, 3654602809, 3873151461, 530742520, 3299628645, 4096336452, 1126891415, 2878612391, 4237533241, 1700485571, 2399980690, 4293915773,
    2240044497, 1873313359, 4264355552, 2734768916, 1309151649, 4149444226, 3174756917, 718787259, 3951481745];
    return {
        getType: function() {
            return g.cd
        },
        B: function() {
            return 192
        },
        A: function() {
            return {
                Lh: c,
                Kh: a,
                Mh: b
            }
        }
    }
});
Q.$e(g.Hb, 95, function() {
    function c(a, b) {
        b = Ra(Ta(b));
        a = Ra(Ma(a, !0));
        if (Ya(b, a))
            return a + "<b>" + b.substr(a.length) + "</b>";
        for (var c = "", e = [], k = b.length - 1, h = 0, m =- 1, d; d = b.charAt(h); ++h)
            " " == d || "\t" == d ? c.length && (e.push({
                Ze: c,
                s: m,
                e: h + 1
            }), c = "", m =- 1) : (c += d, - 1 == m ? m = h : h == k && e.push({
                Ze: c,
                s: m,
                e: h + 1
            }));
        c = a.split(/\s+/);
        h = {};
        for (k = 0; m = c[k++];)
            h[m] = 1;
        d =- 1;
        for (var c = [], p = e.length - 1, k = 0; m = e[k]; ++k)
            h[m.Ze] ? (m =- 1 == d, k == p ? c.push({
                s: m ? k: d,
                e: k
            }) : m && (d = k)) : - 1 < d && (c.push({
                s: d,
                e: k - 1
            }), d =- 1);
        if (!c.length)
            return "<b>" + b + "</b>";
        k = "";
        for (h = m = 0; d = c[h]; ++h)(p = e[d.s].s) 
            && (k += "<b>" + b.substring(m, p - 1) + "</b> "), m = e[d.e].e, k += b.substring(p, m);
        m < b.length && (k += "<b>" + b.substring(m) + "</b> ");
        return k
    }
    return {
        getType: function() {
            return g.Hb
        },
        B: function() {
            return 95
        },
        A: function() {
            return {
                bold: c
            }
        }
    }
});
Q.register(g.Ye, 12, function() {
    function c(c) {
        c = a(c, f, b);
        c = a(c, z, l);
        return a(c, u, e)
    }
    function a(a, b, c) {
        for (var f, d = "", e = 0; null != (f = b.exec(a));)
            e < f.index && (d += a.substring(e, f.index)), d += c(f[0]), e = b.lastIndex;
        if (!d)
            return a;
        e < a.length && (d += a.substring(e));
        return d
    }
    function b(a) {
        return String.fromCharCode(a.charCodeAt(0) - 65248)
    }
    function l(a) {
        var b = a.charCodeAt(0);
        return 1 == a.length ? h.charAt(b - 65377) : 65438 == a.charCodeAt(1) ? m.charAt(b - 65395) : d.charAt(b - 65418)
    }
    function e(a) {
        var b = a.charCodeAt(0);
        return 12443 ==
        a.charCodeAt(1) ? p.charAt(b - 12454) : q.charAt(b - 12495)
    }
    function k(a) {
        return eval('"\\u30' + a.split(",").join("\\u30") + '"')
    }
    var h = k("02,0C,0D,01,FB,F2,A1,A3,A5,A7,A9,E3,E5,E7,C3,FC,A2,A4,A6,A8,AA,AB,AD,AF,B1,B3,B5,B7,B9,BB,BD,BF,C1,C4,C6,C8,CA,CB,CC,CD,CE,CF,D2,D5,D8,DB,DE,DF,E0,E1,E2,E4,E6,E8,E9,EA,EB,EC,ED,EF,F3,9B,9C"), m = k("F4__,AC,AE,B0,B2,B4,B6,B8,BA,BC,BE,C0,C2,C5,C7,C9_____,D0,D3,D6,D9,DC"), d = k("D1,D4,D7,DA,DD"), p = k("F4____,AC_,AE_,B0_,B2_,B4_,B6_,B8_,BA_,BC_,BE_,C0_,C2__,C5_,C7_,C9______,D0__,D3__,D6__,D9__,DC"),
    q = k("D1__,D4__,D7__,DA__,DD"), f = /[\uFF01-\uFF5E]/g, z = RegExp("([\uff73\uff76-\uff84\uff8a-\uff8e]\uff9e)|([\uff8a-\uff8e]\uff9f)|([\uff61-\uff9f])", "g"), v = "([" + k("A6,AB,AD,AF,B1,B3,B5,B7,B9,BB,BD,BF,C1,C4,C6,C8,CF,D2,D5,D8,DB") + "]\u309b)|([" + k("CF,D2,D5,D8,DB") + "]\u309c)", u = new RegExp(v, "g");
    return {
        getType: function() {
            return g.Ye
        },
        B: function() {
            return 12
        },
        A: function() {
            return {
                G: c
            }
        }
    }
});
function qc(c, a, b, l, e) {
    var k = Rb ? "-moz-": lb ? "-ms-": kb ? "-o-": Sb ? "-webkit-": "", h = ".gstl_" + l, m = new RegExp("(\\.(" + e.join("|") + ")\\b)"), d = [];
    return {
        addRule: function(c, e) {
            if (a) {
                if (b) {
                    for (var f = c.split(","), k = [], l = 0, u; u = f[l++];)
                        u = m.test(u) ? u.replace(m, h + "$1") : h + " " + u, k.push(u);
                    c = k.join(",")
                }
                d.push(c, "{", e, "}")
            }
        },
        mh: function() {
            if (a && d.length) {
                a=!1;
                var b = hb("style");
                b.setAttribute("type", "text/css");
                (c || ab()).appendChild(b);
                var e = d.join("");
                d = null;
                b.styleSheet ? b.styleSheet.cssText = e : b.appendChild(document.createTextNode(e))
            }
        },
        prefix: function(a, b) {
            var c = a + (b || "");
            k && (c += b ? a + k + b : k + a);
            return c
        }
    }
};
Q.register(g.Na, 10, function() {
    function c(a) {
        var c = 0;
        a && (h || b(), l(), a in m ? c = m[a] : (ub(h, Ra(a)), m[a] = c = h.offsetWidth, ub(h, "")));
        return c
    }
    function a() {
        h || b();
        l();
        d || (ub(h, "|"), d = h.offsetHeight);
        return d
    }
    function b() {
        h = tb(e.fc);
        h.style.visibility = "hidden";
        k.appendChild(h)
    }
    function l() {
        var a = Ga();
        if (!q || q + 3E3 < a)
            q = a, a = Ib(h, "fontSize"), p && a == p || (m = {}, d = null, p = a)
    }
    var e, k, h, m, d, p, q;
    return {
        O: function(a) {
            k = a.ke() || document.body
        },
        I: function(a) {
            e = a
        },
        getType: function() {
            return g.Na
        },
        B: function() {
            return 10
        },
        A: function() {
            return {
                getWidth: c,
                getHeight: a
            }
        }
    }
});
function rc(c) {
    var a;
    (function() {
        function b(a) {
            return c[a] || l
        }
        function l() {}
        c || (c = {});
        a = {
            Rc: b("a"),
            search: b("b"),
            Gb: b("c"),
            redirect: b("d"),
            Fb: b("e"),
            tb: b("f"),
            zc: b("g"),
            Ac: b("h"),
            vc: b("i"),
            ed: b("j"),
            xb: b("k"),
            Wc: b("l"),
            yc: b("m"),
            Xe: b("n"),
            Gc: b("o"),
            Hc: b("p"),
            Eb: b("q"),
            Nc: b("r"),
            Ve: b("s"),
            We: b("t"),
            xc: b("u"),
            Ic: b("w"),
            qc: b("x"),
            wc: b("y"),
            uc: b("z"),
            tc: b("aa"),
            Bc: b("ab"),
            Zc: b("ac")
        }
    })();
    return {
        Rc: function() {
            return a.Rc()
        },
        search: function(b, c) {
            a.search(b, c)
        },
        Gb: function(b) {
            a.Gb(b)
        },
        redirect: function(b) {
            a.redirect(b)
        },
        Fb: function(b) {
            return a.Fb(b)
        },
        tb: function(b) {
            a.tb(b)
        },
        zc: function(b) {
            a.zc(b)
        },
        Ac: function(b) {
            a.Ac(b)
        },
        vc: function(b) {
            a.vc(b)
        },
        ed: function(b, c) {
            a.ed(b, c)
        },
        xb: function(b, c) {
            a.xb(b, c)
        },
        Wc: function() {
            a.Wc()
        },
        yc: function(b) {
            a.yc(b)
        },
        Xe: function(b) {
            a.Xe(b)
        },
        Gc: function() {
            a.Gc()
        },
        Hc: function() {
            a.Hc()
        },
        Eb: function(b) {
            a.Eb(b)
        },
        Nc: function(b, c) {
            a.Nc(b, c)
        },
        Ve: function(b) {
            a.Ve(b)
        },
        We: function() {
            a.We()
        },
        xc: function() {
            a.xc()
        },
        wc: function() {
            a.wc()
        },
        Ic: function(b) {
            a.Ic(b)
        },
        qc: function() {
            a.qc()
        },
        uc: function() {
            a.uc()
        },
        tc: function() {
            a.tc()
        },
        Bc: function() {
            a.Bc()
        },
        Zc: function(b, c) {
            return a.Zc(b, c)
        }
    }
};
Q.register(g.Ta, 6, function() {
    function c(a, b, c, d) {
        var h = a.getId(), k = a.Y();
        w.ye || e();
        b = q + f + z + "?" + (v ? v + "&" : "") + (b ? b + "&" : "");
        var l = Oa;
        a = [];
        l("q", k, a, !0);
        w.ze || l("callback", "google.sbox.p" + p, a);
        if (u) {
            for (var k = "", y = 4 + Math.floor(32 * Math.random()), O = 0, D; O < y; ++O)
                D = .3 > Math.random() ? 48 + Math.floor(10 * Math.random()) : (.5 < Math.random() ? 65 : 97) + Math.floor(26 * Math.random()), k += String.fromCharCode(D);
            l("gs_gbg", k, a)
        }
        l = hb("script");
        l.src = b + a.join("&");
        l.charset = "utf-8";
        n[h] = l;
        x = w.ye ? d : c;
        m.appendChild(l);
        return !0
    }
    function a() {
        return 0
    }
    function b() {
        return 0
    }
    function l(a) {
        var b = n[a];
        b && (m.removeChild(b), delete n[a])
    }
    function e() {
        for (var a in n)
            m.removeChild(n[a]);
        n = {};
        x = null
    }
    function k(a) {
        x && x(a)
    }
    function h(a) {
        a || (a = Ja);
        var b = window.google;
        w.ze ? b.ac.h = a : b.sbox["p" + p] = a
    }
    var m = ab(), d, p, q, f, z, v, u, w, n = {}, x, y = {
        C: function(a) {
            d = a.get(g.ia, y);
            p = a.Ca().getId()
        },
        activate: function(a) {
            w = a;
            0 == a.Pc && (a = d.Yh(), q = a.protocol, f = a.host, z = a.Qc, v = a.Zh, u = "https:" == document.location.protocol, h(k), (new Image).src = q + f + "/generate_204")
        },
        getType: function() {
            return g.Ta
        },
        B: function() {
            return 6
        },
        A: function() {
            return {
                sendRequest: c,
                zh: l,
                mb: Ja,
                Se: a,
                Le: b
            }
        },
        N: function() {
            h(null);
            e()
        }
    };
    return y
});
Q.register(g.Ia, 1, function() {
    function c(a) {
        if (!m)
            return !0;
        for (var b=!1, c=!1, d = 0, h; d < a.length; ++d)
            if (h = a.charAt(d), !l.test(h) && (e.test(h) ? c=!0 : b=!0, c && b))
                return !0;
        return !1
    }
    function a(a, b, c) {
        if (!m)
            return !0;
        var d = k.test(c), e = h.test(b);
        return "ltr" == a ? d || e || l.test(c) || l.test(b) : !d ||!e
    }
    function b(a) {
        var b = d;
        m && (e.test(a) ? b = "ltr" : l.test(a) || (b = "rtl"));
        return b
    }
    var l = RegExp("^[\x00- !-@[-`{-\u00bf\u00d7\u00f7\u02b9-\u02ff\u2000-\u2bff]*$"), e = RegExp("^[\x00- !-@[-`{-\u00bf\u00d7\u00f7\u02b9-\u02ff\u2000-\u2bff]*(?:\\d[\x00- !-@[-`{-\u00bf\u00d7\u00f7\u02b9-\u02ff\u2000-\u2bff]*$|[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02b8\u0300-\u0590\u0800-\u1fff\u2c00-\ufb1c\ufdfe-\ufe6f\ufefd-\uffff])"),
    k = RegExp("^[\x00- !-@[-`{-\u00bf\u00d7\u00f7\u02b9-\u02ff\u2000-\u2bff]*(?:\\d|[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02b8\u0300-\u0590\u0800-\u1fff\u2c00-\ufb1c\ufdfe-\ufe6f\ufefd-\uffff])"), h = RegExp("(?:\\d|[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02b8\u0300-\u0590\u0800-\u1fff\u2c00-\ufb1c\ufdfe-\ufe6f\ufefd-\uffff])[\x00- !-@[-`{-\u00bf\u00d7\u00f7\u02b9-\u02ff\u2000-\u2bff]*$"), m = e.test("x"), d;
    return {
        O: function(a) {
            d = a.Da()
        },
        getType: function() {
            return g.Ia
        },
        B: function() {
            return 1
        },
        A: function() {
            return {
                xg: c,
                kg: a,
                $b: b
            }
        }
    }
});
Q.register(g.S, 2, function() {
    function c(a, b, c, d, f) {
        var e = q(a);
        e || (e = {}, u.push({
            element: a,
            di: e
        }));
        var h = e[b];
        h || (h = e[b] = [], e = a.bi ? window : Pb(a), e = p(b, e, h), Ha(b) ? a.addEventListener ? a.addEventListener(b, e, !1) : a["on" + b] = e : a[b] = e);
        h.push({
            gi: !!f,
            ld: !1,
            priority: d || 0,
            process: c
        });
        h.sort(z);
        c.ci = b
    }
    function a(a, b) {
        var c = q(a);
        if (c && (c = c[b.ci]))
            for (var d = 0, f; f = c[d++];)
                if (f.process == b) {
                    f.ld=!0;
                    break
                }
    }
    function b(a, b, f, d) {
        c(w, a, b, f, d)
    }
    function l(b) {
        a(w, b)
    }
    function e(a, b) {
        var c = b || {}, f = w[a];
        f && f(c, c.Ka)
    }
    function k(a,
    b, c) {
        a.addEventListener ? a.addEventListener(b, c, !1) : a.attachEvent("on" + b, c)
    }
    function h(a, b, c) {
        a.removeEventListener ? a.removeEventListener(b, c, !1) : a.detachEvent("on" + b, c)
    }
    function m(a) {
        v ? (n || (n = [], k(window, "message", d)), n.push(a), a = window.location.href, window.postMessage("sbox.df", /HTTPS?:\/\//i.test(a) ? a : "*")) : window.setTimeout(a, 0)
    }
    function d(a) {
        n && a && a.source == window && "sbox.df" == a.data && n.length && (n.shift()(), n && n.length && window.postMessage("sbox.df", window.location.href))
    }
    function p(a, b, c) {
        return function(d,
        e) {
            if (c.length) {
                var h;
                if (!(h = d)) {
                    h = {};
                    var k = b.event;
                    k && (k.keyCode && (h.keyCode = k.keyCode), h.fi=!0)
                }
                h.Ka = e || a;
                for (var k = h, m, l, n = 0, v; v = c[n++];)
                    v.ld ? l=!0 : m || (v.gi ? f(v, k) : m = v.process(k));
                if (l)
                    for (n = 0; v = c[n];)
                        v.ld ? c.splice(n, 1) : ++n;
                if (h.Bb)
                    return delete h.Bb, h.fi && (h = b.event || h), Cb(h), h.returnValue=!1
            }
        }
    }
    function q(a) {
        for (var b = 0, c; b < u.length; ++b)
            if (c = u[b], c.element == a)
                return c.di;
        return null
    }
    function f(a, b) {
        m(function() {
            a.process(b)
        })
    }
    function z(a, b) {
        return b.priority - a.priority
    }
    var v = window.postMessage &&
    !(lb || Tb || kb), u = [], w = {
        bi: 1
    }, n;
    return {
        getType: function() {
            return g.S
        },
        B: function() {
            return 2
        },
        A: function() {
            return {
                R: c,
                Sc: a,
                Oa: b,
                va: l,
                J: e,
                listen: k,
                G: h,
                defer: m
            }
        },
        N: function() {
            n = null
        }
    }
});
Q.register(g.Qa, 495, function() {
    function c(a) {
        m[a.getId()]=!0
    }
    function a(a) {
        var b = a.da(), c = b.getId();
        c in m && (b = b.getTimestamp(), b = Ga() - b, p += b, ++d, a.D().getString("e")&&++q, delete m[c])
    }
    function b() {
        var a = 0, b;
        for (b in m)
            a++;
        return a
    }
    function l() {
        return d
    }
    function e() {
        return p
    }
    function k(a) {
        var b=!1;
        a && (b = a.D().getString("e"));
        a = 0;
        b ? (a|=1, 1 < q && (a|=2)) : 0 < q && (a|=2);
        return a
    }
    function h() {
        m = {};
        q = p = d = 0
    }
    var m, d, p, q;
    return {
        activate: function() {
            h()
        },
        getType: function() {
            return g.Qa
        },
        B: function() {
            return 495
        },
        A: function() {
            return {
                Dh: c,
                Wh: a,
                Nh: b,
                Ph: l,
                Qh: e,
                Oh: k,
                reset: h
            }
        }
    }
});
Q.register(g.Z, 375, function() {
    function c(a) {
        e[a]=!0;
        k = a
    }
    function a() {
        var a = [], b;
        for (b in e)
            a.push(parseInt(b, 10));
        return a
    }
    function b() {
        return k
    }
    function l() {
        e = {};
        k = null
    }
    var e, k;
    return {
        activate: function() {
            l()
        },
        getType: function() {
            return g.Z
        },
        B: function() {
            return 375
        },
        A: function() {
            return {
                add: c,
                Rh: a,
                Sg: b,
                reset: l
            }
        }
    }
});
Q.register(g.ca, 9, function() {
    function c(a) {
        var b = q.X(), c;
        c = [];
        c[27] = 23;
        c[0] = l(n.clientName);
        c[28] = l(n.pb);
        c[1] = void 0 == a ? "" : a + "";
        c[26] = f.Rh().join("j");
        a = "";
        v.Mc() ? a = "o" : u.Ba() && (a = u.ce() + "");
        c[2] = a;
        a = "";
        var h = u.U();
        if (h) {
            for (var r, F = 0, O = 0, D; D = h[O++];) {
                var M = D;
                D = M.getType() + "";
                M = M.Xc();
                M.length && (D += "i" + M.join("i"));
                D != r && (1 < F && (a += "l" + F), a += (r ? "j" : "") + D, F = 0, r = D);
                ++F
            }
            1 < F && (a += "l" + F)
        }
        c[3] = a;
        c[4] = e(q.Bf());
        c[5] = e(q.Cf());
        c[6] = x;
        c[7] = Ga() - y;
        c[18] = e(q.Df());
        c[8] = p.Eg();
        if (r = p.yg())
            c[25] = r.Hg ? "1" + (n.ig ?
            "a" : "") + (n.Nd ? "c" : "") : "", c[10] = r.Gg;
        c[11] = p.Kc();
        c[12] = p.Bg();
        if (r = p.Ag())
            c[9] = r.Jg, c[22] = r.Ig, c[17] = r.Kg;
        c[13] = p.Dg();
        c[14] = p.Cg();
        c[15] = p.Fg();
        c[16] = p.zg();
        c[30] = z.Nh();
        c[31] = z.Ph();
        c[32] = z.Qh();
        c[19] = l(n.ic);
        r = v.fa();
        r = z.Oh(r);
        c[20] = 0 == r ? "" : r + "";
        for (r = 0; a = w[r++];)
            h = a.wa(), m[h] && (c[h] = void 0 == c[h] ? l(a.getValue()) : "");
        c = c.join(".").replace(k, "");
        d && s ? (r = b + c, a = d.Kh(s), r = d.Mh(r, a), r = r.slice(0, 8), r = d.Lh(r)) : r = "";
        return {
            oq: b,
            gs_l: c + "." + r
        }
    }
    function a() {
        y = Ga();
        ++x;
        q.pa();
        f.reset();
        p.pa();
        for (var a = 0, b; b =
        w[a++];)
            b.reset()
    }
    function b(a) {
        s = a
    }
    function l(a) {
        return a ? a.replace(h, "-") : ""
    }
    function e(a) {
        return Math.max(a - y, 0)
    }
    var k = /\.+$/, h = /\./g, m = za(dc), d, p, q, f, z, v, u, w, n, x =- 1, y, s, r = {
        C: function(a) {
            d = a.get(g.cd, r);
            p = a.get(g.T, r);
            q = a.get(g.F, r);
            f = a.get(g.Z, r);
            z = a.get(g.Qa, r);
            v = a.get(g.W, r);
            u = a.get(g.P, r);
            w = a.M(g.je, r);
            Nb(a.M(g.RENDERER, r))
        },
        I: function(a) {
            s = a.jg
        },
        activate: function(b) {
            n = b;
            a()
        },
        getType: function() {
            return g.ca
        },
        B: function() {
            return 9
        },
        A: function() {
            return {
                D: c,
                reset: a,
                Lg: b
            }
        }
    };
    return r
});
Q.register(g.ha, 11, function() {
    function c(a, b) {
        if (v) {
            for (var c=!1, f = 0, e; e = v[f++];)
                2 == e.Tb(a, b) && (c=!0);
            if (c)
                return 
        }
        if (Qa(a) || r.ta || d && d.ta())
            bb(b) ? s&&!y && (y = yb(s, "btnI", "1")) : y && (s.removeChild(y), y = null), h(b), x.search(a, b), k(), p.J(14, {
                query: a
            })
    }
    function a(a) {
        h();
        x.Gb(a);
        k()
    }
    function b(a) {
        h();
        x.redirect(a);
        k()
    }
    function l(a) {
        h(1);
        x.Eb(a);
        k()
    }
    function e(a) {
        return x.Fb(a)
    }
    function k() {
        q.Nb();
        q.Rg();
        z.reset();
        w ? w.clear() : u.clear();
        f.X() != f.ea() && f.Nf();
        n && n.clear()
    }
    function h(a) {
        m && r.fe && m.Ec(a)
    }
    var m,
    d, p, q, f, z, v, u, w, n, x, y, s, r, t = {
        O: function(a) {
            s = a.ke()
        },
        C: function(a) {
            m = a.get(g.yb, t);
            d = a.get(g.ma, t);
            p = a.get(g.S, t);
            q = a.get(g.T, t);
            f = a.get(g.F, t);
            z = a.get(g.ca, t);
            u = a.get(g.P, t);
            w = a.get(g.ue, t);
            n = a.get(g.Fa, t);
            x = a.la();
            v = a.M(g.te, t)
        },
        activate: function(a) {
            r = a
        },
        getType: function() {
            return g.ha
        },
        B: function() {
            return 11
        },
        A: function() {
            return {
                search: c,
                Gb: a,
                redirect: b,
                Eb: l,
                Fb: e
            }
        }
    };
    return t
});
Q.register(g.ua, 14, function() {
    function c(a) {
        return (a[e.Jb] || {}).j
    }
    function a(a) {
        return a[e.Te]
    }
    function b(a, b) {
        var c = a[e.Te], q = a[e.ji];
        b || (b = lc(c, Pa(c.length)));
        var w = {}, n = a[e.Jb];
        if (n)
            for (var x in n) {
                var y = n[x];
                x in p && (y = p[x].parse(y));
                w[x] = y
            }
        var n = b, s=!1, r=!1;
        x=!1;
        for (var y = 0, t; t = q[y++];)
            if (33 == (t[k.Ue] || 0) ? r=!0 : s=!0, r && s) {
                x=!0;
                break
            }
        s = 0;
        r = [];
        for (y = 0; t = q[y++];) {
            var G = t[k.Ue] || 0;
            if (h[G] && (!x || 33 != G)) {
                var H;
                H = t[k.ii];
                d && (H = m.bold(c.toLowerCase(), Ua(Sa(H))));
                r.push(oc(H, Ua(Sa(H)), s++, G, t[k.hi] || [],
                l(t)))
            }
        }
        return mc(n, r, pc(w), !1, !0, !1)
    }
    function l(a) {
        return (a = a[k.Jb]) ? pc(a) : nc
    }
    var e = cc, k = Zb, h, m, d, p = {}, q = {
        C: function(a) {
            m = a.get(g.Hb, q);
            if (a = a.M(g.Tc, q))
                for (var b = 0, c; c = a[b++];)
                    p[c.tj()] = c
        },
        activate: function(a) {
            h = a.hb;
            d = a.Fc
        },
        getType: function() {
            return g.ua
        },
        B: function() {
            return 14
        },
        A: function() {
            return {
                oh: c,
                G: a,
                jc: b
            }
        }
    };
    return q
});
Q.register(g.rb, 15, function() {
    function c(b) {
        var c = a(b);
        if (c) {
            if (h)
                for (var e = 0, q; q = h[e++];)
                    b = q.edit(b);
            m.Vh(b);
            e = b;
            q = e.da().Y();
            var w = e.U();
            if (d.isEnabled())
                if (w.length) {
                    var n = 0 == e.getType();
                    d.Ub(q, w, n) && k.Wh(e)
                } else 
                    d.clear();
            l.J(3, {
                input: q,
                Of: w
            })
        }
        p.ed(b, c);
        return c
    }
    function a(a) {
        var b = e.ea(), c = m.fa(), b = b.toLowerCase(), d = a.Y().toLowerCase();
        b == d ? c=!0 : (b = Ma(b), a = (d = a.da()) ? d.ja() : Ma(a.Y().toLowerCase()), c = c ? c.da().ja() : "", c = 0 == b.indexOf(a) ? 0 == b.indexOf(c) ? a.length >= c.length : !0 : !1);
        return c
    }
    function b(a,
    b) {
        return a.V() - b.V()
    }
    var l, e, k, h, m, d, p, q = {
        C: function(a) {
            l = a.get(g.S, q);
            e = a.get(g.F, q);
            k = a.get(g.Qa, q);
            h = a.M(g.re, q);
            m = a.get(g.W, q);
            d = a.get(g.P, q);
            p = a.la();
            h.sort(b)
        },
        getType: function() {
            return g.rb
        },
        B: function() {
            return 15
        },
        A: function() {
            return {
                process: c,
                ob: a
            }
        }
    };
    return q
});
Q.register(g.T, 13, function() {
    function c(a, b) {
        if (!(!X || $ || O && O.Ai())) {
            a.$c("ds", ea.Od);
            a.$c("pq", ja);
            a.rh();
            var c=!0, d = a.He();
            d > I && (I = d);
            ++A;
            S.Dh(a);
            var d = Ga(), e;
            for (e in da) {
                var f = da[e].getTimestamp();
                2500 < d - f && t(e)
            }
            V && (e = F.get(a)) && ((c = K || a.uh()) && ea.Sf && a.sh(), T.process(e), e.Ae()&&++N, E = null);
            c && (E = a, B&&!b || r())
        }
    }
    function a() {
        return 10 <= ka || 3 <= D.Le()?!0 : !1
    }
    function b() {
        Y = I
    }
    function l() {
        return I <= Y
    }
    function e() {
        E = null
    }
    function k() {
        return A
    }
    function h() {
        return {
            Hg: V,
            Gg: V ? F.qh(): 0
        }
    }
    function m() {
        return V ?
        F.Kc() : 0
    }
    function d() {
        return N
    }
    function p() {
        return {
            Jg: la,
            Ig: fa,
            Kg: pa
        }
    }
    function q() {
        return sa
    }
    function f() {
        return ma
    }
    function z(a) {
        a = Z.jc(a);
        return T.ob(a)
    }
    function v() {
        return ha
    }
    function u() {
        for (var a = [], b = 0, c, d = 0; d <= J; ++d)
            c = ia[d], 0 == c ? b++ : (b = 1 == b ? "0j" : 1 < b ? d + "-" : "", a.push(b + c), b = 0);
        return a.join("j")
    }
    function w() {
        V && F.clearAll()
    }
    function n(a) {
        V && F.ph(a)
    }
    function x() {
        V && F.pa();
        ha = ma = sa = pa = fa = la = N = ka = A = 0;
        ia = [];
        for (var a = 0; a <= J; ++a)
            ia[a] = 0
    }
    function y(a) {
        ja = a
    }
    function s(a) {
        return function(b) {
            G(b, a)
        }
    }
    function r() {
        null != B && ($a(B), B = null);
        if (!(2 < D.Le()) && E) {
            var a = [], b = E.D();
            if (b)
                for (var c in b)
                    Oa(c, b[c], a);
            C.Wc();
            a = a.join("&");
            a = D.sendRequest(E, a, s(E), G);
            E.Ie() || (++la, a ? (a = E, da[a.getId()] = a, ++ka) : ++fa);
            E = null;
            a = 100;
            b = (ka - 2) / 2;
            for (c = 1; c++<=b;)
                a*=2;
            a < W && (a = W);
            B = window.setTimeout(r, a)
        }
    }
    function t(a) {
        D.zh(a);
        delete da[a];
        ka&&--ka
    }
    function G(a, b) {
        if (X) {
            if (!b) {
                var c = Z.oh(a);
                b = da[c];
                if (!b)
                    return 
            }
            if (!b.Ie()) {
                c = Z.jc(a, b);
                if (aa)
                    var d = M.ea(), c = aa.Bi(c, d);
                V && F.put(c);
                b.He() <= Y || (++pa, T.process(c)||++sa, d = b,
                W = c.D().xe("d"), d && (t(d.getId()), d = d.getTimestamp(), d = Ga() - d, ha += d, ma = Math.max(d, ma), ++ia[d > P ? J: H[Math.floor(d / 100)]]));
                c && (c = c.D().getString("q")) && R.Lg(c)
            }
        }
    }
    var H = [0, 1, 2, 3, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8], J = H[H.length - 1] + 1, P = 100 * H.length - 1, F, O, D, M, R, S, Z, T, aa, U, C, X=!1, E, I =- 1, da, A, ka, N, la, fa, pa, sa, ma, ha, ia, W, B, K, $, Y, V, ea, ja, ba = {
        C: function(a) {
            F = a.get(g.ub, ba);
            O = a.get(g.ma, ba);
            a.get(g.S, ba);
            M = a.get(g.F, ba);
            R = a.get(g.ca, ba);
            S = a.get(g.Qa, ba);
            Z = a.get(g.ua, ba);
            T = a.get(g.rb, ba);
            aa = a.get(g.Gh, ba);
            a.get(g.W,
            ba);
            U = a.get(g.ia, ba);
            a.get(g.P, ba);
            C = a.la()
        },
        activate: function(a) {
            D = U.nh();
            ea = a;
            X=!0;
            da = {};
            W = 0;
            K = a.Rf;
            $ = a.Pd;
            Y =- 1;
            V = ea.Qf&&!!F;
            ja = a.Qd
        },
        getType: function() {
            return g.T
        },
        B: function() {
            return 13
        },
        A: function() {
            return {
                qd: c,
                Mc: a,
                Nb: b,
                Wd: l,
                Rg: e,
                Eg: k,
                yg: h,
                Kc: m,
                Bg: d,
                Ag: p,
                Dg: q,
                Cg: f,
                ob: z,
                Fg: v,
                zg: u,
                mb: w,
                wh: n,
                pa: x,
                Xd: y
            }
        },
        N: function() {
            X=!1;
            null != B && ($a(B), B = null);
            da = E = null;
            b()
        }
    };
    return ba
});
Q.register(g.W, 5, function() {
    function c() {
        return e.Mc()
    }
    function a(a) {
        k = a
    }
    function b() {
        return k
    }
    function l() {
        k = null
    }
    var e, k, h = {
        C: function(a) {
            e = a.get(g.T, h)
        },
        activate: function() {
            k = null
        },
        getType: function() {
            return g.W
        },
        B: function() {
            return 5
        },
        A: function() {
            return {
                Mc: c,
                Vh: a,
                fa: b,
                G: l
            }
        }
    };
    return h
});
Q.register(g.ia, 16, function() {
    function c() {
        return e
    }
    function a() {
        return k
    }
    function b() {
        e && e.mb()
    }
    var l = {}, e, k, h = {
        C: function(a) {
            a = a.M(g.Ta, h);
            for (var b = 0, c; c = a[b++];)
                l[c.Se()] = c
        },
        activate: function(a) {
            var b = "https:" == document.location.protocol, c = Oa, h = [];
            c("client", a.clientName, h);
            c("hl", a.Ya, h);
            c("gl", a.we, h);
            c("sugexp", a.ic, h);
            c("gs_rn", 23, h);
            c("gs_ri", a.pb, h);
            a.Ab && c("authuser", a.Ab, h);
            k = {
                protocol: "http" + (b ? "s" : "") + "://",
                host: a.se || "clients1." + a.Jc,
                Qc: a.Qc || "/complete/search",
                Zh: h.length ? h.join("&"):
                ""
            };
            e && e.Se() == a.Pc || (e = l[a.Pc])
        },
        getType: function() {
            return g.ia
        },
        B: function() {
            return 16
        },
        A: function(e) {
            return {
                nh: e == g.T ? c: Ja,
                Yh: a,
                Xh: b
            }
        }
    };
    return h
});
Q.register(g.$, 7, function() {
    function c(a) {
        d.qb(a)
    }
    function a() {
        return p
    }
    function b(a) {
        if (a in q) {
            if (f) {
                if (a == f.Dc())
                    return;
                k();
                f.ee()
            }
            f = q[a];
            d.setPanel(f)
        }
    }
    function l() {
        return p ? d.getHeight() : 0
    }
    function e() {
        p || (d.show(h()), p=!0)
    }
    function k() {
        p && (d.hide(), p=!1)
    }
    function h() {
        var a = eb(m);
        f.de(a);
        return a
    }
    var m = {
        Sh: !1,
        hc: "left",
        Ge: !0,
        ra: null,
        marginWidth: 0
    }, d, p, q = {}, f, z = {
        C: function(a) {
            d = a.get(g.Sa, z);
            if (a = a.M(g.kb, z))
                for (var b = 0, c; c = a[b++];)
                    q[c.Dc()] = c
        },
        activate: function() {
            p=!1
        },
        getType: function() {
            return g.$
        },
        B: function() {
            return 7
        },
        A: function() {
            return {
                isVisible: a,
                setPanel: b,
                getHeight: l,
                show: e,
                hide: k,
                qb: c
            }
        },
        N: function() {
            k()
        }
    };
    return z
});
Q.register(g.F, 3, function() {
    function c() {
        var a = {};
        V.J(13, a);
        !a.cancel && ta.od && V.defer(L.Wa);
        ua.wc()
    }
    function a() {
        V.J(12);
        ua.xc()
    }
    function b() {
        fa("rtl")
    }
    function l() {
        fa("ltr")
    }
    function e() {
        L.hf()
    }
    function k(a) {
        L.ga() ? L.gf() : L.Va(a)
    }
    function h() {
        if (0 == ta.eb)
            return !1;
        if (4 == ta.eb)
            return ua.Bc(), !1;
        var a = pa();
        if (a)
            switch (ta.eb) {
            case 1:
                if (sa(a, !0))
                    return ja.add(B.nb), !0;
                    break;
                case 3:
                    return L.Ea(a)
            }
        return !1
    }
    function m() {
        ta.ff ? da(5) : (L.isVisible() ? L.Wa() : w(), G())
    }
    function d(a) {
        ca && a.ge() == ca.length && (Aa &&
        Aa.clear(), ta.ef && da(2), ua.vc(ca))
    }
    function p(a) {
        $ && 0 == a.getPosition() && $.rg()
    }
    function q(a, b, c, d) {
        ta.df&&!a && L.pd(!0);
        ta.cf&&!L.isVisible() && "mousedown" == c && L.Va(b);
        var e;
        Xa && Xa.dg(a, b, c) ? e = Xa : Xa = e = lc(a, b, c);
        var f = b=!1;
        if (a != ca || "onremovechip" == c)
            Ya(c, "key") ? ja.add(B.ng) : "paste" == c && ja.add(B.og), b=!0, ia(a), V.J(1, {
                Ka: c,
                ra: Ca
            }), ua.tb(a), f = Ga(), ib || (ib = f), Eb = f, Qa(a) && (d=!0), f=!0;
        a = W.DONT_CARE;
        var h = e.zd(), k = oa.fa();
        if (qa)
            for (var l = 0, n; n = qa[l++];)
                n = n.Tb(h, k), n > a && (a = n);
        switch (a) {
        case W.lg:
            d=!0;
            break;
        case W.qg:
            d=!1
        }
        d ? (b && L.jf(), jb && e.setParameter("gs_is", 1), ua.yc(jb), ea.qd(e), Xa = null) : f && (L.clear(), ea.Nb());
        V.J(2, {
            Ka: c
        })
    }
    function f(a) {
        (jb = a) && ja.add(B.mg)
    }
    function z(a) {
        Fb != a && ((Fb = a) ? ua.uc() : ua.tc())
    }
    function v(a) {
        ma(a)
    }
    function u() {
        K.focus()
    }
    function w() {
        K.blur()
    }
    function n() {
        return K.Pb()
    }
    function x(a, b, c) {
        Ya(a, ca, !0) && (a = ca + a.substr(ca.length));
        c = c || Pa(a.length);
        q(a, c, "", b);
        ma(a, !0)
    }
    function y(a) {
        x(a, !0);
        Gb = Ga();
        ja.add(B.pg)
    }
    function s() {
        q(ca, D(), "onremovechip")
    }
    function r(a) {
        ia(a);
        K.refresh();
        V.J(4, {
            ra: Ca,
            input: a
        })
    }
    function t() {
        K.select()
    }
    function G() {
        ca != Da && ia(Da);
        V.J(5, {
            input: Da,
            Of: L.U(),
            ra: Ca
        });
        K.refresh();
        ua.Ac(Da)
    }
    function H() {
        Da = ca
    }
    function J() {
        return K.Fd()
    }
    function P() {
        return Da
    }
    function F() {
        return ca
    }
    function O() {
        return Ca
    }
    function D() {
        return K.Ga()
    }
    function M() {
        return K.Xb()
    }
    function R() {
        return K.getHeight()
    }
    function S() {
        return K.getWidth()
    }
    function Z() {
        return K.Gd()
    }
    function T() {
        return ib
    }
    function aa() {
        return Eb
    }
    function U() {
        return Gb
    }
    function C() {
        return 0 != ic
    }
    function X() {
        if (Wa) {
            if (ta.Lb)
                return !0;
            for (var a = 0, b; b = Ba[a++];)
                if (b.isEnabled())
                    return !0
        }
        return !1
    }
    function E(a) {
        if (a == ca)
            return !0;
        var b = ca.length;
        return a.substr(0, b) == ca ? Y.kg(Ca, ca, a.substr(b)) : !1
    }
    function I() {
        K.Sb()
    }
    function da(a) {
        ba.search(ca, a)
    }
    function A(a) {
        ca && (ia(""), K.clear(), V.J(1), L.clear(), ua.tb(ca));
        a && ua.qc()
    }
    function ka() {
        Gb = Eb = ib = 0
    }
    function N(a) {
        K.Jd(a)
    }
    function la() {
        var a = pa();
        a && sa(a)
    }
    function fa(a) {
        var b = D().getPosition();
        Ca == a ? L.ga() && b == ca.length && (L.Ba() ? ta.Ua && (a = L.Aa(), ba.search(a.K(), 6)) : ta.ud && h()) : $ && 0 == b && $.rg()
    }
    function pa() {
        if (L.ga()) {
            var a = L.Ba() ? L.Aa(): L.Mb();
            if (a.cc())
                return a
        }
        return null
    }
    function sa(a, b) {
        var c = a.K();
        return Za(Da, c)?!1 : (H(), b ? x(c, !0) : r(c), !0)
    }
    function ma(a, b) {
        ca = a || "";
        ha();
        K.refresh();
        b || (V.J(4, {
            ra: Ca,
            input: ca
        }), ua.zc(ca))
    }
    function ha() {
        var a = Y.$b(ca);
        a != Ca && (K.Vb(a), Ca = a)
    }
    function ia(a) {
        ca = Da = a || "";
        ha()
    }
    var W = bc, B = ac, K, $, Y, V, ea, ja, ba, qa, oa, L, Aa, Wa, Ba, ua, Da, ca, Ca, ic, ib, Eb, Gb, jb, Fb, Xa, ta, ra = {
        C: function(a) {
            K = a.get(g.na, ra);
            $ = a.get(g.ma, ra);
            Y = a.get(g.Ia, ra);
            V = a.get(g.S, ra);
            ea = a.get(g.T,
            ra);
            ja = a.get(g.Z, ra);
            ba = a.get(g.ha, ra);
            qa = a.M(g.Ha, ra);
            oa = a.get(g.W, ra);
            L = a.get(g.P, ra);
            Aa = a.get(g.Fa, ra);
            Wa = a.get(g.oa, ra);
            Ba = a.M(g.qa, ra);
            ua = a.la();
            ic = a.Ca().Ob()
        },
        I: function(a) {
            ta = a;
            qa.sort(cb);
            ca = Da = K.nf() || ""
        },
        activate: function(a) {
            ta = a;
            Fb = jb=!1;
            ha()
        },
        getType: function() {
            return g.F
        },
        B: function() {
            return 3
        },
        A: function() {
            return {
                Hd: c,
                sf: a,
                uf: b,
                vf: l,
                wf: e,
                qf: k,
                Ea: h,
                rf: m,
                pf: d,
                of: p,
                tf: q,
                yf: f,
                Id: z,
                cb: v,
                Ed: u,
                dc: w,
                Pf: n,
                yd: x,
                G: y,
                va: s,
                Qb: r,
                xd: t,
                Bd: G,
                Nf: H,
                Fd: J,
                X: P,
                ea: F,
                $b: O,
                Ga: D,
                Xb: M,
                getHeight: R,
                getWidth: S,
                Gd: Z,
                Bf: T,
                Cf: aa,
                Df: U,
                xf: C,
                ec: X,
                oc: E,
                Sb: I,
                search: da,
                clear: A,
                pa: ka,
                Jd: N,
                wd: la
            }
        }
    };
    return ra
});
Q.register(g.P, 17, function() {
    function c(a) {
        a.ra = qa;
        a.marginWidth = ba;
        var b = oa.mf;
        b || (b = "rtl" == qa ? "right" : "left");
        a.hc = b
    }
    function a(a, c, d) {
        var e=!1;
        a = ma && ma.xi(c);
        G();
        (W = c) && c.length && (e = c[0].K(), da.xg(e) && (e = N.X()), qa = da.$b(e), d ? ($ = E.Mg, e = I.fg(c, qa), c = c[0].D().getString("a"), c = Sa(c), ba = la.getWidth(c)) : ($ = E.ve, e = I.render(T(), qa), ba = 0), a && (K = ma.wi(), b(ma.vi())), e ? r() : G());
        return e
    }
    function b(a) {
        X();
        if (B != a) {
            var b = B;
            B = a;
            C(b)
        }
    }
    function l() {
        if (x())
            if (Y) {
                var a = B;
                B == W.length - 1 ? K = B = null : null == B ? B = 0 : ++B;
                K =
                B;
                U(a, l)
            } else 
                r()
    }
    function e() {
        if (x())
            if (Y) {
                var a = B;
                W && 0 != B ? null == B ? B = W.length - 1 : --B : K = B = null;
                K = B;
                U(a, e)
            } else 
                r()
    }
    function k(a) {
        var b = a ? 4: 3;
        y() ? (a = w(), I.ib(a) || N.search(b), b = N.X(), ia.xb(b, a)) : N.search(b)
    }
    function h(a) {
        return I.Ea(a)
    }
    function m(a) {
        K = B = a;
        a = W[a];
        var b = N.X();
        ia.xb(b, a)
    }
    function d() {
        return Y
    }
    function p() {
        return V
    }
    function q(a) {
        V&&!a && G();
        V = a
    }
    function f() {
        return $
    }
    function z() {
        return W
    }
    function v() {
        return x() ? W[0] : null
    }
    function u() {
        return B
    }
    function w() {
        return y() ? W[K] : null
    }
    function n() {
        return K
    }
    function x() {
        return !(!W ||!W.length)
    }
    function y() {
        return null != K
    }
    function s() {
        Y&&!ea && (ea = window.setTimeout(G, oa.lf))
    }
    function r() {
        Y || (A.setPanel(17), A.show(), Y=!0, ia.Gc())
    }
    function t() {
        Y && (ea && ($a(ea), ea = null), A.hide(), Y=!1, ia.Hc())
    }
    function G() {
        t();
        W = null;
        $ = E.EMPTY;
        null != B && I.vb(B);
        K = B = null;
        I.clear()
    }
    function H() {
        ka.Nb();
        t()
    }
    function J() {
        null != B && I.vb(B);
        K = B = null
    }
    function P() {
        X();
        ja = window.setTimeout(J, 0)
    }
    function F() {
        X()
    }
    function O(a) {
        if (x())
            r();
        else {
            var b = N.X();
            if (b) {
                a = a || N.Ga();
                b = lc(b, a);
                if (pa) {
                    a =
                    b.zd();
                    for (var c = sa.fa(), d = 0, e; e = pa[d++];)
                        e.Tb(a, c)
                    }
                ka.qd(b)
            }
        }
    }
    function D() {
        return I.L()
    }
    function M() {
        return I.lb()
    }
    function R() {
        Y=!1
    }
    function S() {
        I.La()
    }
    function Z() {
        return 17
    }
    function T() {
        if (x() && $ == E.ve) {
            for (var a = [], b = [], c = 0, d; (d = fa[c++])&&!d.getMessage(N.X(), W, b););
            (c = b ? b.length : 0) && (c -= aa(b, a, 0));
            for (d = 0; d < W.length; ++d)
                a.push(W[d]);
            c && (c -= aa(b, a, 1));
            oa.kf && a.push(1);
            c && aa(b, a, 2);
            oa.vd && a.push(2);
            ha && ha.ui(a);
            return a
        }
        return null
    }
    function aa(a, b, c) {
        for (var d = 0, e = 0, f; e < a.length; ++e)(f = a[e]) 
            &&
            f.position == c && (b.push(f), ++d);
        return d
    }
    function U(a, b) {
        if (null == B || I.xa(B))
            if (C(a), null == B)
                N.Bd();
            else {
                var c = I.Cc(W[B]);
                N.cb(c);
                ia.Ic(c)
            } else 
                I.vb(a), b()
    }
    function C(a) {
        X();
        null != a && I.vb(a);
        null != B && I.highlight(B)
    }
    function X() {
        ja && ($a(ja), ja = null)
    }
    var E = $b, I, da, A, ka, N, la, fa, pa, sa, ma, ha, ia, W, B, K, $, Y, V, ea, ja, ba, qa, oa, L = {
        C: function(a) {
            I = a.get(g.Ja, L);
            da = a.get(g.Ia, L);
            A = a.get(g.$, L);
            ka = a.get(g.T, L);
            N = a.get(g.F, L);
            la = a.get(g.Na, L);
            fa = a.M(g.mc, L);
            pa = a.M(g.Ha, L);
            sa = a.get(g.W, L);
            ma = a.get(g.Ng, L);
            ha = a.get(g.Og,
            L);
            ia = a.la()
        },
        I: function() {
            pa.sort(cb);
            fa.sort(db)
        },
        activate: function(a) {
            oa = a;
            K = B = null;
            $ = E.EMPTY;
            Y=!1;
            V=!0;
            qa = "";
            ba = 0
        },
        getType: function() {
            return g.P
        },
        B: function() {
            return 17
        },
        A: function() {
            return {
                Ub: a,
                hg: b,
                gf: l,
                hf: e,
                ib: k,
                Ea: h,
                eg: m,
                isVisible: d,
                isEnabled: p,
                pd: q,
                zf: f,
                U: z,
                Mb: v,
                wg: u,
                Aa: w,
                ce: n,
                ga: x,
                Ba: y,
                jf: s,
                show: r,
                hide: t,
                clear: G,
                Wa: H,
                $d: J,
                gg: P,
                G: F,
                Va: O
            }
        },
        Ma: function() {
            var a = {
                de: c,
                L: D,
                lb: M,
                ee: R,
                La: S,
                Dc: Z
            };
            return [{
                O: Ja,
                C: Ja,
                I: Ja,
                activate: Ja,
                getType: function() {
                    return g.kb
                },
                B: function() {
                    return 17
                },
                A: function() {
                    return a
                },
                Ma: Ja,
                N: Ja
            }
            ]
        },
        N: function() {
            ea && ($a(ea), ea = null);
            W = null;
            t()
        }
    };
    return L
});
Q.register(g.Sa, 8, function() {
    function c(a) {
        a != t && (t = a, a = a.L(), G ? a != G && s.replaceChild(a, G) : s.appendChild(a), G = a)
    }
    function a() {
        r || (r = s ? Math.max(s.offsetHeight, 0) : 0);
        return r
    }
    function b(a) {
        s.className = a.Sh ? "gssb_e gsdd_a" : "gssb_e";
        var b = a.ra || O;
        w != b && (w = b, xb(u, b));
        b = a.marginWidth;
        if (y != b) {
            var c = x.style;
            b ? (n.hasChildNodes() || n.appendChild(x), c.width = b + "px", Rb && (c.paddingLeft = "1px")) : (n.hasChildNodes() && n.removeChild(x), c.paddingLeft = "");
            y = b
        }
        M = a.Ge;
        R = a.hc;
        d(H, !0);
        d(F, !0);
        f.J(16);
        e()
    }
    function l() {
        r = 0;
        d(H,
        !1);
        d(F, !1);
        var a = O;
        w != a && (w = a, xb(u, a));
        f.J(11)
    }
    function e() {
        r = 0;
        h();
        if (F) {
            var b = z.he[0], c = F.style;
            "relative" != z.Pa && (c.top = u.style.top, c.left = u.offsetLeft + n.offsetWidth + "px");
            b = a() + b;
            F.style.height = Math.max(b, 0) + "px";
            m(F, s.offsetWidth)
        }
        t && t.La()
    }
    function k(a) {
        if (J)
            P != a && J.replaceChild(a, P);
        else {
            var b = u.insertRow( - 1);
            b.style.height = "0";
            b.insertCell( - 1);
            J = b.insertCell( - 1);
            p.isVisible() || (d(s, !1), d(u, !0), e());
            H = s;
            J.appendChild(a)
        }
        P = a
    }
    function h() {
        var a, b, c;
        a = (b = t && t.lb()) ? b.offsetWidth : q.getWidth();
        (c = D) ? Ha(c) && (c = null) : y ||!M ? (s.style.width = "", u.style.width = "") : (s.style.width = "100%", c = a + z.wb[2], m(u, c));
        if ("relative" != z.Pa) {
            var d = q.Xb();
            b && (d.ka = ob(b).ka);
            b = z.wb;
            var e = b[1];
            b = b[0];
            b = d.jd + q.getHeight() + b;
            "right" == R ? (c = Pb(u), a = Kb(c) - (d.ka - e + a), c = void 0) : (d = d.ka + e, "center" == R && c && (d += (a - c) / 2), c = d, a = void 0);
            e = {
                ka: 0,
                jd: 0
            };
            "absolute" == z.Pa && z.Ra && z.Ra != document.body && (e = ob(z.Ra));
            d = u.style;
            d.top = b - e.jd + "px";
            d.left = d.right = "";
            void 0 != c ? d.left = c - e.ka + "px" : d.right = a + e.ka + "px"
        }
        fb && (d.zoom = "normal", d.zoom =
        1)
    }
    function m(a, b) {
        Ia(b) ? 0 < b && (a.style.width = b + "px") : a.style.width = b
    }
    function d(a, b) {
        a && (a.style.display = b ? "" : "none")
    }
    var p, q, f, z, v, u, w, n, x, y, s, r, t, G, H, J, P, F, O, D, M=!0, R, S = {
        O: function(a, b) {
            O = a.Da();
            b.addRule(".gssb_c", "border:0;position:absolute;z-index:989");
            b.addRule(".gssb_e", "border:1px solid #ccc;border-top-color:#d9d9d9;" + b.prefix("box-shadow:0 2px 4px rgba(0,0,0,0.2);") + "cursor:default");
            b.addRule(".gssb_f", "visibility:hidden;white-space:nowrap");
            b.addRule(".gssb_k", "border:0;display:block;position:absolute;top:0;z-index:988");
            b.addRule(".gsdd_a", "border:none!important")
        },
        C: function(a) {
            p = a.get(g.$, S);
            q = a.get(g.F, S);
            f = a.get(g.S, S);
            v = a.Ca().getId()
        },
        I: function(a) {
            z = a;
            u = rb();
            u.className = "gstl_" + v + " gssb_c";
            d(u, !1);
            H = u;
            var b = u.insertRow( - 1);
            n = b.insertCell( - 1);
            n.className = "gssb_f";
            x = sb();
            s = b.insertCell( - 1);
            s.className = "gssb_e";
            s.style.width = "100%";
            z.ie && (F = hb("iframe", "gstl_" + v + " gssb_k"), d(F, !1), (z.Ra || document.body).appendChild(F));
            if (D = z.sg)
                Ia(D) && (D += z.wb[2]), m(u, D);
            h();
            (a.Ra || document.body).appendChild(u);
            f.Oa(8, e)
        },
        activate: function(a) {
            z =
            a;
            u.style.position = a.Pa
        },
        getType: function() {
            return g.Sa
        },
        B: function() {
            return 8
        },
        A: function() {
            return {
                setPanel: c,
                getHeight: a,
                qb: k,
                show: b,
                hide: l,
                La: e
            }
        }
    };
    return S
});
Q.register(g.na, 4, function() {
    function c(a, b) {
        pa && (pa=!1, E.Sc(A, F), E.Sc(A, O));
        b || (b = a);
        A.parentNode.replaceChild(a, A);
        b.appendChild(A);
        fa && la.Kf && (lb || Rb ? E.defer(function() {
            A.focus();
            mb(A, ha.getPosition())
        }) : A.focus());
        D()
    }
    function a() {
        return $
    }
    function b(a) {
        var b = "rtl" == a == ("rtl" == qa);
        A.dir = a;
        if (Y) {
            I.Vb(a);
            var c = B.parentNode;
            c.removeChild(Y);
            b ? Db(Y, B) : c.insertBefore(Y, B)
        }
        $ && ($.dir = a, c = $.parentNode, c.removeChild($), b ? c.insertBefore($, B) : Db($, B));
        0 != ka && (a = wb(a), vb(A, a, 0))
    }
    function l() {
        return ha
    }
    function e() {
        return ob(K)
    }
    function k() {
        var a = K ? K.offsetHeight: 0;
        L > a && (a = L);
        return a
    }
    function h() {
        return Aa ? Aa : K ? K.offsetWidth : 0
    }
    function m() {
        var a = A.offsetWidth;
        la.rd && (a -= A.offsetHeight);
        return a
    }
    function d() {
        return A.value
    }
    function p(a) {
        (la.Ff ? A : B || Wa || A).style.background = a || "transparent"
    }
    function q() {
        W=!0
    }
    function f() {
        A.select();
        T()
    }
    function z() {
        Ub && (A.value = "");
        A.value = C.ea();
        Ub && (A.value = A.value);
        y()
    }
    function v() {
        if (!fa)
            try {
                A.focus(), fa=!0, y()
        } catch (a) {}
    }
    function u() {
        fa && (A.blur(), fa=!1)
    }
    function w() {
        return fa
    }
    function n() {
        A.value =
        ""
    }
    function x() {
        var a = oa.get("gs_id");
        if (a)
            $ = oa.get("gs_ttc"), B = oa.get("gs_tti"), C.ec() && I && (V = I.L(), Y = V.parentNode);
        else {
            a = rb();
            a.id = oa.getId("gs_id");
            a.className = "gstl_" + N + " " + (la.Rb || A.className);
            var b = a.insertRow( - 1), d = a.style, e = A.style;
            d.width = Aa ? Aa + "px" : e.width;
            d.height = L ? L + "px" : e.height;
            d.padding = "0";
            Lb(A);
            A.className = la.fc;
            ba && ($ = b.insertCell( - 1), $.id = oa.getId("gs_ttc"), $.style.whiteSpace = "nowrap");
            B = b.insertCell( - 1);
            B.id = oa.getId("gs_tti");
            B.className = "gsib_a";
            C.ec() && I && (V = I.L(), Y = b.insertCell( - 1),
            Y.className = "gsib_b", Y.appendChild(V));
            c(a, B)
        }
        Vb && Sb && (A.style.height = "1.25em", A.style.marginTop = "-0.0625em");
        s(a);
        K = a
    }
    function y() {
        if (fa) {
            var a = A.value.length;
            ha = Pa(a);
            mb(A, a)
        }
    }
    function s(a) {
        E.R(a, "mouseup", function() {
            A.focus()
        })
    }
    function r() {
        function a(c) {
            E.R(A, c, P, 10, b)
        }
        E.R(A, "keydown", G);
        (kb || la.Ef) && E.R(A, "keypress", J);
        E.R(A, "select", T, 10);
        var b=!1;
        a("mousedown");
        a("keyup");
        a("keypress");
        b=!0;
        a("mouseup");
        a("keydown");
        a("focus");
        a("blur");
        a("cut");
        a("paste");
        a("input");
        E.R(A, "compositionstart",
        t);
        E.R(A, "compositionend", t)
    }
    function t(a) {
        a = a.type;
        "compositionstart" == a ? C.Id(!0) : "compositionend" == a && C.Id(!1)
    }
    function G(a) {
        var b = a.keyCode;
        ia = b;
        var c = (Sb || Rb) && qb(b) && X.ga(), d = b == U.Ce, e = b == U.Yc;
        ea=!1;
        b == U.nb && (ea = C.Ea());
        d && ((b = X.Aa()) && H(b) ? X.ib(a.shiftKey) : E.defer(function() {
            X.ib(a.shiftKey)
        }));
        if (c || d || e || ea)
            a.Bb=!0
    }
    function H(a) {
        return (a = da[a.getType()].yi) && a()
    }
    function J(a) {
        var b = a.keyCode, c = b == U.Yc, d = b == U.nb && ea;
        if (b == U.Ce || c || d)
            a.Bb=!0
    }
    function P(a) {
        if (!ja) {
            var b = a.Ka;
            if (!(b.indexOf("key") ||
            a.ctrlKey || a.altKey || a.shiftKey || a.metaKey))
                n: if (a = a.keyCode, "keypress" != b) {
                var c = qb(a), d;
                if ("keydown" == b) {
                    if (C.yf(229 == a), c)
                        break n
                } else if (d = a != ia, ia =- 1, !c || d)
                    break n;
                    switch (a) {
                    case U.Yc:
                        C.rf();
                        break;
                    case U.bh:
                        C.uf();
                        break;
                    case U.eh:
                        C.vf();
                        break;
                    case U.fh:
                        C.wf();
                        break;
                    case U.$g:
                        C.qf(ha);
                        break;
                    case U.DELETE:
                        C.pf(ha);
                        break;
                    case U.gh:
                        C.of(ha)
                    }
                }
            T();
            C.tf(A.value, ha, b)
        }
    }
    function F() {
        fa=!0;
        C.sf()
    }
    function O() {
        fa=!1;
        C.Hd()
    }
    function D() {
        pa || (pa=!0, E.R(A, "focus", F, 99), E.R(A, "blur", O, 99))
    }
    function M() {
        ma ||
        (ma = window.setInterval(S, la.Jf || 50))
    }
    function R() {
        ma && ($a(ma), ma = null)
    }
    function S() {
        P({
            Ka: "polling"
        })
    }
    function Z() {
        Rb && Ab(A)
    }
    function T() {
        if (fa) {
            var a = nb(A);
            a && (ha = a)
        }
    }
    function aa() {
        var a;
        E.listen(window, "pagehide", function() {
            ja=!0;
            a = A.value
        });
        E.listen(window, "pageshow", function(b) {
            ja=!1;
            b.persisted && C.Qb(a)
        })
    }
    var U = jc, C, X, E, I, da, A, ka, N, la, fa, pa=!1, sa, ma, ha = Pa(0), ia =- 1, W=!1, B, K, $, Y, V, ea, ja, ba, qa, oa, L, Aa, Wa, Ba = {
        O: function(a, b) {
            oa = a;
            A = a.Dd();
            qa = a.Da();
            a.Za() || (b.addRule(".gsib_a", "width:100%;padding:4px 6px 0"),
            b.addRule(".gsib_a,.gsib_b", "vertical-align:top"))
        },
        C: function(a) {
            C = a.get(g.F, Ba);
            E = a.get(g.S, Ba);
            X = a.get(g.P, Ba);
            I = a.get(g.oa, Ba);
            da = Nb(a.M(g.RENDERER, Ba));
            a = a.Ca();
            ka = a.Ob();
            N = a.getId()
        },
        I: function(a) {
            la = a;
            L = a.Wb;
            Aa = a.If;
            fa = pb(A);
            T();
            lb && E.R(A, "beforedeactivate", function(a) {
                W && (W=!1, a.Bb=!0)
            }, 10);
            Rb && aa();
            K = A;
            ba=!!a.$a[g.ma];
            (C.xf() || C.ec() || ba || a.Gf) && x();
            a.Kd && (E.R(A, "blur", R, 10), E.R(A, "focus", M, 10), sa=!0);
            E.Oa(8, Z);
            r();
            D()
        },
        activate: function(a) {
            la = a;
            var b = a.Hf;
            b && (Wa = oa.Xa(b));
            A.setAttribute("autocomplete",
            "off");
            A.setAttribute("spellcheck", a.spellcheck);
            A.style.outline = a.Ld ? "" : "none";
            sa && M()
        },
        getType: function() {
            return g.na
        },
        B: function() {
            return 4
        },
        A: function() {
            return {
                G: c,
                Fd: a,
                Vb: b,
                Ga: l,
                Xb: e,
                getHeight: k,
                getWidth: h,
                Gd: m,
                nf: d,
                Jd: p,
                Sb: q,
                select: f,
                refresh: z,
                focus: v,
                blur: u,
                Pb: w,
                clear: n
            }
        },
        N: function() {
            sa && R();
            la.od && E.Sc(A, C.Hd)
        }
    };
    return Ba
});
Q.register(g.Ja, 18, function() {
    function c(a, b) {
        if (!T)
            return !1;
        S = b;
        x();
        for (var c=!1, d = 0, e; e = a[d++];)
            z(e) && (c=!0);
        return c
    }
    function a(a) {
        var b = t[a.getType()];
        return b && b.Eh ? b.Eh(a) : !1
    }
    function b(a) {
        return t[a.getType()].Vc(null, a, G)
    }
    function l(a) {
        var b = t[a.getType()];
        if (b && b.Cc) {
            var c = r.X();
            return b.Cc(a, c)
        }
        return a.K()
    }
    function e(a, b) {
        if (!T)
            return !1;
        S = b;
        x();
        for (var c=!1, d = 0, e; e = a[d++];)
            if (1 == e)
                if (U)
                    aa.appendChild(U);
                else {
                    e = u();
                    var f = e.style;
                    f.textAlign = "center";
                    f.whiteSpace = "nowrap";
                    e.dir = Z;
                    f = sb();
                    f.style.position = "relative";
                    C = sb();
                    C.className = "gssb_g";
                    J.vd && (C.style.paddingBottom = "1px");
                    v(J.Yf, C, 13);
                    J.Uf ? v(J.gc, C, 8) : J.Vf && v(J.Zf, C, 14);
                    f.appendChild(C);
                    e.appendChild(f);
                    U = e.parentNode
                } else 
                    2 == e ? X ? aa.appendChild(X) : (e = u(), f = e.style, f.padding = "1px 4px 2px 0", f.fontSize = "11px", f.textAlign = "right", f = hb("a"), f.id = "gssb_b", f.href = "http://www.google.com/support/websearch/bin/answer.py?hl=" + J.Ya + "&answer=106230", f.innerHTML = J.Xf, e.appendChild(f), X = e.parentNode) : 3 == e ? (e = D.pop()) ? aa.appendChild(e) : (e =
                    T.insertRow( - 1), e.Fh=!0, e = e.insertCell( - 1), f = hb("div", "gssb_l"), e.appendChild(f)) : z(e) && (c=!0);
        return c
    }
    function k(a) {
        w(a, E);
        var b = y.U();
        b && s.J(9, {
            index: a,
            zi: b[a],
            template: M[a]
        })
    }
    function h(a) {
        w(a, "");
        s.J(10)
    }
    function m() {
        for (var a, b, c; c = F.pop();)
            a = c.getType(), (b = P[a]) || (b = P[a] = []), b.push(c), a = c.L(), a.parentNode.removeChild(a);
        for (; a = aa.firstChild;)
            a = aa.removeChild(a), a.Fh ? D.push(a) : a != U && a != X && O.push(a);
        M = []
    }
    function d(a) {
        return (a = M[a]) ? a.xa() : !1
    }
    function p() {
        x()
    }
    function q() {
        return T
    }
    function f() {
        return J.Cd ||
        Z == S ? R : null
    }
    function z(a) {
        var b = a.getType(), c = t[b];
        if (!c)
            return !1;
        var d = (b = P[b]) && b.pop();
        d || (d = c.Uc(G));
        c.render(a, d);
        F.push(d);
        var e = d.L(), b = u();
        b.className = "gssb_a " + J.kc;
        b.appendChild(e);
        if (void 0 !== a.wa) {
            M.push(d);
            var d = S, f = a.wa();
            J.Wf && (e.onmouseover = function() {
                y.hg(f)
            }, e.onmouseout = function() {
                y.gg()
            });
            e.onclick = function(b) {
                r.dc();
                a.cc() && r.cb(a.K());
                y.$d();
                y.eg(f);
                b = b || Pb(e).event;
                c.ya(b, a, G)
            }
        } else 
            d = Z;
        xb(b, d);
        return !0
    }
    function v(a, b, c) {
        var d = hb("input");
        d.type = "button";
        d.value = Sa(a);
        d.onclick =
        function() {
            G.search(r.ea(), c)
        };
        var e;
        if (J.Tf) {
            a = "lsb";
            e = hb("span");
            var f = hb("span");
            e.className = "ds";
            f.className = "lsbb";
            e.appendChild(f);
            f.appendChild(d)
        } else 
            a = "gssb_h", e = d;
        d.className = a;
        b.appendChild(e)
    }
    function u() {
        var a = O.pop();
        if (a)
            return aa.appendChild(a), a.firstChild;
        a = T.insertRow( - 1);
        a = a.insertCell( - 1);
        a.className = J.kc;
        a.onmousedown = n;
        return a
    }
    function w(a, b) {
        var c = M[a];
        c && c.xa() && (c.L().parentNode.parentNode.className = b)
    }
    function n(a) {
        a = a || Pb(T).event;
        a.stopPropagation ? a.stopPropagation() :
        kb || lb && r.Sb();
        return !1
    }
    function x() {
        if (C) {
            var a = J.Zd ? J.Zd: r.getWidth() - 3;
            0 < a && (C.style.width = a + "px")
        }
    }
    var y, s, r, t, G, H, J, P = {}, F = [], O = [], D = [], M = [], R, S, Z, T, aa, U, C, X, E, I = {
        O: function(a, b) {
            H = a;
            Z = a.Da();
            b.addRule(".gssb_a", "padding:0 7px");
            b.addRule(".gssb_a,.gssb_a td", "white-space:nowrap;overflow:hidden;line-height:22px");
            b.addRule("#gssb_b", "font-size:11px;color:#36c;text-decoration:none");
            b.addRule("#gssb_b:hover", "font-size:11px;color:#36c;text-decoration:underline");
            b.addRule(".gssb_g", "text-align:center;padding:8px 0 7px;position:relative");
            b.addRule(".gssb_h", "font-size:15px;height:28px;margin:0.2em" + (Sb ? ";-webkit-appearance:button" : ""));
            b.addRule(".gssb_i", "background:#eee");
            b.addRule(".gss_ifl", "visibility:hidden;padding-left:5px");
            b.addRule(".gssb_i .gss_ifl", "visibility:visible");
            b.addRule("a.gssb_j", "font-size:13px;color:#36c;text-decoration:none;line-height:100%");
            b.addRule("a.gssb_j:hover", "text-decoration:underline");
            b.addRule(".gssb_l", "height:1px;background-color:#e5e5e5");
            b.addRule(".gssb_m", "color:#000;background:#fff")
        },
        C: function(a) {
            y = a.get(g.P, I);
            s = a.get(g.S, I);
            r = a.get(g.F, I);
            G = a.get(g.ha, I);
            t = Nb(a.M(g.RENDERER, I))
        },
        I: function(a) {
            J = a;
            T = rb();
            a = hb("tbody");
            T.appendChild(a);
            aa = T.getElementsByTagName("tbody")[0]
        },
        activate: function(a) {
            J = a;
            var b = a.Zb;
            b && (R = H.Xa(b));
            T.className = a.cg || "gssb_m";
            E = a.$f || "gssb_i"
        },
        getType: function() {
            return g.Ja
        },
        B: function() {
            return 18
        },
        A: function() {
            return {
                fg: c,
                Cc: l,
                ib: b,
                Ea: a,
                render: e,
                highlight: k,
                vb: h,
                clear: m,
                xa: d,
                La: p,
                L: q,
                lb: f
            }
        }
    };
    return I
});
Q.register(g.yb, 346, function() {
    function c(c) {
        c = b.D(c);
        for (var e in h)
            e in c || (c[e] = h[e]);
        a(l + La(c))
    }
    function a(a) {
        var b = new Image, c = k;
        b.onerror = b.onload = b.onabort = function() {
            try {
                delete e[c]
            } catch (a) {}
        };
        e[k] = b;
        b.src = a;
        k++
    }
    var b, l, e = [], k = 0, h, m = {
        C: function(a) {
            b = a.get(g.ca, m)
        },
        activate: function(a) {
            l = "//" + (a.Yg || "www." + a.Jc) + "/gen_204?";
            h = a.Zg || {}
        },
        getType: function() {
            return g.yb
        },
        B: function() {
            return 346
        },
        A: function() {
            return {
                Ec: c
            }
        }
    };
    return m
});
Q.register(g.ub, 21, function() {
    function c(a) {
        m(a);
        var b = a.da();
        if ((!b ||!b.Re()) && f)
            for (b = 0; b < f.length; ++b)
                f[b].update(a)
    }
    function a(a) {
        var b = q[a.Qe()] || null;
        if (b)++z;
        else if (f&&!a.Re())
            for (var c = 0; c < f.length; ++c)
                if (b = f[c].get(a)) {
                    m(b);
                    ++v;
                    break
                }
        return b ? mc(a, b.U(), b.D(), b.Ae(), b.fd(), b.$h()) : null
    }
    function b() {
        return z
    }
    function l() {
        return v
    }
    function e() {
        v = z = 0
    }
    function k(a) {
        var b, c, e, f;
        for (f in q)
            for (b = q[f], b = b.U(), e = 0; c = b[e++];)
                if (c.getType() == a) {
                    delete q[f];
                    break
                }
        d()
    }
    function h() {
        q = {};
        d()
    }
    function m(a) {
        a &&
        a.fd() && (q[a.da().Qe()] = a)
    }
    function d() {
        if (f)
            for (var a = 0; a < f.length; ++a)
                f[a].reset()
    }
    function p(a, b) {
        return b.V() - a.V()
    }
    var q = {}, f, z, v, u = {
        C: function(a) {
            f = a.M(g.zb, u);
            f.sort(p)
        },
        activate: function() {
            e()
        },
        getType: function() {
            return g.ub
        },
        B: function() {
            return 21
        },
        A: function() {
            return {
                put: c,
                get: a,
                qh: b,
                Kc: l,
                pa: e,
                ph: k,
                clearAll: h
            }
        }
    };
    return u
});
Q.register(g.Cb, 190, function() {
    function c() {
        q && d.md(m)
    }
    function a() {
        q && d.Ib(m)
    }
    function b() {
        q && p.md(m)
    }
    function l() {
        q && p.Ib(m)
    }
    var e, k, h, m, d, p, q=!1, f = {
        O: function(a, b) {
            function c(a) {
                return "box-shadow:" + a + "-moz-box-shadow:" + a + "-webkit-box-shadow:" + a
            }
            h = a;
            b.addRule(".gsfe_a", "border:1px solid #b9b9b9;border-top-color:#a0a0a0;" + c("inset 0px 1px 2px rgba(0,0,0,0.1);"));
            b.addRule(".gsfe_b", "border:1px solid #4d90fe;outline:none;" + c("inset 0px 1px 2px rgba(0,0,0,0.3);"))
        },
        C: function(a) {
            e = a.get(g.S, f);
            k = a.get(g.F,
            f)
        },
        I: function(f) {
            var k = f.Oc;
            if (m = k ? h.Xa(k) : null)
                e.Oa(12, b), e.Oa(13, l), e.R(m, "mouseover", c), e.R(m, "mouseout", a), d = sc(f.Qg || "gsfe_a"), p = sc(f.Pg || "gsfe_b")
        },
        activate: function() {
            q=!0;
            m && k.Pf() && p.md(m)
        },
        getType: function() {
            return g.Cb
        },
        B: function() {
            return 190
        },
        N: function() {
            q=!1;
            m && (d.Ib(m), p.Ib(m))
        }
    };
    return f
});
function sc(c) {
    var a = new RegExp("(?:^|\\s+)" + c + "(?:$|\\s+)");
    return {
        md: function(b) {
            b&&!a.test(b.className) && (b.className += " " + c)
        },
        Ib: function(b) {
            b && (b.className = b.className.replace(a, " "))
        }
    }
};
Q.register(g.RENDERER, 33, function() {
    function c(a) {
        q = a.tg;
        f = a.ne;
        z = a.le;
        v = a.Yb ? a.gc : ""
    }
    function a(a) {
        return tc(k, h, m, d, p, a, q, z)
    }
    function b(a, b) {
        b.render(a.gd(), a.K(), a.wa(), f, v)
    }
    function l(a, b, c) {
        c.search(b.K(), 1)
    }
    function e() {
        return 35
    }
    var k, h, m, d, p, q, f, z, v, u = {
        O: function(a, b) {
            b.addRule("a.gspqs_a", "padding:0 3px 0 8px");
            b.addRule(".gspqs_b", "color:#666;line-height:22px")
        },
        C: function(a) {
            m = a.get(g.T, u);
            d = a.get(g.F, u);
            h = a.get(g.za, u);
            k = a.get(g.ia, u);
            p = a.get(g.P, u)
        },
        I: c,
        activate: c,
        getType: function() {
            return g.RENDERER
        },
        B: function() {
            return 33
        },
        A: function() {
            return {
                Uc: a,
                render: b,
                ya: l,
                Vc: Ja,
                kd: e
            }
        }
    };
    return u
});
function tc(c, a, b, l, e, k, h, m) {
    function d(b) {
        r=!0;
        a.qe(x, p);
        return Cb(b)
    }
    function p() {
        r && (b.wh(35), c.Xh(), q.onmouseover = q.onmouseout = q.onclick = null, f.style.display = "none", z.style.display = "", e.ce() == y && l.Bd(), e.wg() == y && (e.$d(), l.Ed()), s=!1)
    }
    var q, f, z, v, u, w, n, x, y, s=!0, r=!1;
    (function() {
        q = sb();
        q.className = "gsq_a";
        var a = rb();
        q.appendChild(a);
        f = a.insertRow( - 1);
        var b = f.insertCell( - 1);
        v = hb("span");
        v.style.color = "#52188c";
        b.appendChild(v);
        if (0 != h) {
            w = hb("a");
            w.href = "#ps";
            w.className = "gspqs_a gssb_j";
            var c = f.insertCell( - 1);
            c.appendChild(w);
            (2 == h ? c : b).style.width = "100%";
            z = a.insertRow( - 1);
            n = z.insertCell( - 1);
            n.className = "gspqs_b";
            n.innerHTML = m;
            n.colSpan = "2"
        }
    })();
    return {
        L: function() {
            return q
        },
        getType: function() {
            return 35
        },
        xa: function() {
            return s
        },
        render: function(a, b, c, e, m) {
            r=!1;
            s=!0;
            x = b;
            y = c;
            f.style.display = "";
            v.innerHTML = a;
            0 != h && (z.style.display = "none", w.innerHTML = e, w.onclick = d);
            m&&!u && (u = Hb(f), u.onclick = function(a) {
                l.dc();
                l.cb(x);
                k.search(x, 9);
                return Cb(a)
            });
            m ? (u.innerHTML = m + " &raquo;", u.style.display = "") : u && (u.style.display =
            "none")
        }
    }
};
Q.register(g.za, 188, function() {
    function c() {
        var a = {};
        e && (a.tok = l);
        return a
    }
    function a(a, c) {
        b.ki(a, c)
    }
    var b, l, e, k = {
        C: function(a) {
            b = a.get(g.Db, k)
        },
        activate: function(a) {
            l = a.jb;
            var c = "https:" == document.location.protocol;
            a=!!a.hb[35];
            e=!!(b && l && c && a)
        },
        getType: function() {
            return g.za
        },
        B: function() {
            return 188
        },
        A: function() {
            return {
                mi: c,
                qe: a
            }
        }
    };
    return k
});
Q.register(g.Db, 186, function() {
    function c(a, c) {
        p[a] = c;
        var q = [];
        Oa("delq", a, q);
        Oa("client", m, q);
        Oa("callback", "google.sbox.d" + l, q);
        var u = e;
        Oa("tok", k, q);
        h && Oa("authuser", h, q);
        d = hb("script");
        d.src = u + q.join("&");
        b.appendChild(d)
    }
    function a(a) {
        d && (b.removeChild(d), d = null);
        a = a[0];
        var c = p[a];
        c && (delete p[a], c())
    }
    var b = ab(), l, e, k, h, m, d, p = {}, q = {
        C: function(a) {
            a.get(g.za, q);
            l = a.Ca().getId()
        },
        I: function() {
            window.google.sbox["d" + l] = a
        },
        activate: function(a) {
            e = "https://" + (a.se || "clients1." + a.Jc) + "/complete/deleteitems?";
            k = a.jb;
            h = a.Ab;
            m = a.clientName
        },
        getType: function() {
            return g.Db
        },
        B: function() {
            return 186
        },
        A: function() {
            return {
                ki: c
            }
        },
        N: function() {
            d && (b.removeChild(d), d = null)
        }
    };
    return q
});
Q.register(g.Ha, 187, function() {
    function c(a) {
        var c = b.mi(), h;
        for (h in c)
            a.setParameter(h, c[h]);
        return 1
    }
    function a() {
        return 12
    }
    var b, l = {
        C: function(a) {
            b = a.get(g.za, l)
        },
        getType: function() {
            return g.Ha
        },
        B: function() {
            return 187
        },
        A: function() {
            return {
                Tb: c,
                V: a
            }
        }
    };
    return l
});
Q.register(g.zb, 98, function() {
    function c() {
        return 3
    }
    function a(a) {
        if (e) {
            var b = a.da(), c = a.U();
            if (c.length) {
                var d = b.ja();
                n:
                for (var b = Number.MAX_VALUE, l, m = 0; l = c[m++];) {
                    if (!k[l.getType()]) {
                        b =- 1;
                        break n
                    }
                    l = l.K();
                    b = Math.min(l.length, b)
                }
                if ( - 1 != b) {
                    var n = c[0].K();
                    if (Ya(n, d, !0))
                        for (m = d.length + 1; m <= b;) {
                            d = null;
                            for (l = 0; n = c[l++];) {
                                n = n.K();
                                if (m > n.length)
                                    return;
                                    n = n.substr(0, m);
                                    if (!d)
                                        d = n;
                                    else if (d != n)
                                        return 
                            }
                            h[d] = a;
                            ++m
                        }
                }
            }
        }
    }
    function b(a) {
        if (e) {
            var b = h[a.ja()];
            if (b) {
                var c = a.Pe(), k = a.ja();
                b.da().ja();
                for (var l = b.D(), p =
                d ||!l.xe("k"), n = [], x, y, s = b.U(), r = 0, t; t = s[r++];)
                    y = t.K(), x = p ? m.bold(c, y) : Ra(y), n.push(oc(x, y, t.wa(), t.getType(), t.Xc(), t.D()));
                delete h[k];
                return mc(a, n, l, !0, b.fd(), !1)
            }
        }
        return null
    }
    function l() {
        h = {}
    }
    var e=!0, k, h = {}, m, d, p = {
        C: function(a) {
            m = a.get(g.Hb, p)
        },
        I: function() {
            k = za([0])
        },
        activate: function(a) {
            d = a.Fc;
            e = a.Nd
        },
        getType: function() {
            return g.zb
        },
        B: function() {
            return 98
        },
        A: function() {
            return {
                V: c,
                update: a,
                get: b,
                reset: l
            }
        },
        N: function() {
            e=!1
        }
    };
    return p
});
Q.register(g.RENDERER, 31, function() {
    function c() {
        return uc()
    }
    function a(a, b) {
        var c = a.D(), l = c.getString("a"), c = c.getString("b"), d = a.K();
        b.render(l, c, d)
    }
    function b(a, b, c) {
        c.search(b.K(), 1)
    }
    function l() {
        return 33
    }
    return {
        O: function(a, b) {
            b.addRule(".gspr_a", "padding-right:1px")
        },
        getType: function() {
            return g.RENDERER
        },
        B: function() {
            return 31
        },
        A: function() {
            return {
                Uc: c,
                render: a,
                ya: b,
                Vc: Ja,
                kd: l
            }
        }
    }
});
function uc() {
    var c;
    c = sb();
    c.className = "gspr_a";
    return {
        getType: function() {
            return 33
        },
        L: function() {
            return c
        },
        xa: function() {
            return !0
        },
        render: function(a, b) {
            c.innerHTML = b
        }
    }
};
Q.register(g.RENDERER, 20, function() {
    function c(a) {
        return vc(e, a)
    }
    function a(a, b) {
        b.render(a.gd(), a.K(), k)
    }
    function b(a, b, c) {
        c.search(b.K(), 1)
    }
    function l() {
        return 0
    }
    var e, k, h = {
        O: function(a, b) {
            b.addRule(".gsq_a", "padding:0")
        },
        C: function(a) {
            e = a.get(g.F, h)
        },
        activate: function(a) {
            k = a.Yb ? a.gc : ""
        },
        getType: function() {
            return g.RENDERER
        },
        B: function() {
            return 20
        },
        A: function() {
            return {
                Uc: c,
                render: a,
                ya: b,
                Vc: Ja,
                kd: l
            }
        }
    };
    return h
});
function vc(c, a) {
    var b, l, e, k, h;
    (function() {
        b = sb();
        b.className = "gsq_a";
        var a = rb();
        b.appendChild(a);
        l = a.insertRow( - 1);
        a = l.insertCell( - 1);
        a.style.width = "100%";
        e = hb("span");
        a.appendChild(e)
    })();
    return {
        L: function() {
            return b
        },
        getType: function() {
            return 0
        },
        xa: function() {
            return !0
        },
        render: function(b, d, p) {
            e.innerHTML = b;
            h = d;
            p&&!k && (k = Hb(l), k.onclick = function(b) {
                c.dc();
                c.cb(h);
                a.search(h, 9);
                return Cb(b)
            });
            p ? (k.innerHTML = p + " &raquo;", k.style.display = "") : k && (k.style.display = "none")
        }
    }
};
Q.register(g.qa, 78, function() {
    function c() {
        return w
    }
    function a() {
        return 78
    }
    function b() {
        return 3
    }
    function l() {
        return r
    }
    function e() {
        return {
            tooltip: u
        }
    }
    function k(a) {
        if (!G)
            a = document.createElement("script"), a.src = "//www.google.com/textinputassistant/" + s + "/" + y + "_tia.js", document.body.appendChild(a), G=!0, f.add(3);
        else if (n.onclick)
            n.onclick(a)
    }
    function h() {
        z.Wa()
    }
    function m() {
        v.Ah()
    }
    function d(a) {
        v.Bh(78, a)
    }
    function p(a) {
        v.Ch(78, a)
    }
    function q(a) {
        r.className = "gsok_a gsst_e " + a
    }
    var f, z, v, u, w, n, x, y, s, r,
    t, G, H = {
        O: function(a, b) {
            t = a;
            a.Za() || (b.addRule(".gsok_a", "background:url(data:image/gif;base64,R0lGODlhEwALAKECAAAAABISEv///////yH5BAEKAAIALAAAAAATAAsAAAIdDI6pZ+suQJyy0ocV3bbm33EcCArmiUYk1qxAUAAAOw==) no-repeat center;display:inline-block;height:11px;line-height:0;width:19px"), b.addRule(".gsok_a img", "border:none;visibility:hidden"))
        },
        C: function(a) {
            f = a.get(g.Z, H);
            z = a.get(g.P, H);
            v = a.get(g.oa, H)
        },
        I: function(a) {
            w=!!a.fb;
            x = a.oe;
            y = a.gb;
            s = a.vg;
            u = a.ug;
            (r = t.get("gs_ok")) ? n = r.firstChild : (n = hb("img"), n.src =
            x + "/tia.png", r = hb("span", "gsok_a gsst_e"), r.id = t.getId("gs_ok"), r.appendChild(n));
            n.ds = h;
            n.hd = m;
            n.sc = q;
            n.sd = d;
            n.td = p;
            n.setAttribute("tia_field_name", t.Dd().name);
            n.setAttribute("tia_disable_swap", !0)
        },
        activate: function(a) {
            a.Lb && (w=!!a.fb);
            n.setAttribute("tia_property", a.pe)
        },
        getType: function() {
            return g.qa
        },
        B: function() {
            return 78
        },
        A: function() {
            return {
                isEnabled: c,
                Ne: a,
                V: b,
                L: l,
                Me: e,
                ya: k
            }
        }
    };
    return H
});
Q.register(g.oa, 174, function() {
    function c() {
        return 174
    }
    function a(a) {
        T != a && (P.dir = T = a, k())
    }
    function b() {
        return P
    }
    function l(a) {
        (a = O[a]) && a.style && (a.style.display = "")
    }
    function e(a) {
        (a = O[a]) && a.style && (a.style.display = "none")
    }
    function k() {
        D && (O[D].className = "gsst_a", s.hide(), D = null)
    }
    function h(a, b) {
        D = a;
        var c = O[a];
        c.className = "gsst_a gsst_g";
        var d = M.lastChild;
        d != b && (d == R ? M.appendChild(b) : M.replaceChild(b, d));
        s.setPanel(174);
        s.show();
        c = c.clientWidth;
        R.style.width = c + "px";
        R.style.left = "rtl" == T ? "0" : M.clientWidth -
        c + "px"
    }
    function m(a, b) {
        D == a ? k() : h(a, b)
    }
    function d(a) {
        a.hc = "rtl" == T ? "left" : "right";
        a.Ge=!1
    }
    function p() {
        return M
    }
    function q() {
        return J.Cd || Z == T ? aa : null
    }
    function f() {
        k()
    }
    function z() {
        return 174
    }
    function v(a, b) {
        return b.V() - a.V()
    }
    function u() {
        S != D && k()
    }
    function w() {
        for (var a, b = 0, c; c = G[b++];)
            if (c.isEnabled()) {
                a=!0;
                var d = hb("a", "gsst_a");
                y(d, c);
                d.appendChild(c.L());
                P.appendChild(d)
            }
        P.style.display = a ? "" : "none"
    }
    function n() {
        S = null
    }
    function x() {
        O = {};
        for (var a = 0, b; b = G[a++];)
            if (b.isEnabled()) {
                var c = b.Ne(),
                d = b.L().parentNode;
                d.onclick = b.ya;
                d.onmouseover = function() {
                    S = c
                };
                d.onmouseout = n;
                O[c] = d;
                b.Me && (b = b.Me(), b.Ci && e(c), (b = b.tooltip)&&!H.Zc(d, b) && (d.title = b))
            }
    }
    function y(a, b) {
        a.href = "javascript:void(0)";
        Qb(a);
        a.onkeydown = function(a) {
            a = a || window.event;
            var c = a.keyCode;
            if (13 == c || 32 == c)
                b.ya(a), t.Ed(), Cb(a)
        }
    }
    var s, r, t, G, H, J, P, F, O = {}, D, M, R, S, Z, T, aa, U, C = {
        O: function(a, b) {
            F = a;
            Z = a.Da();
            a.Za() || (b.addRule(".gsst_a", "display:inline-block"), b.addRule(".gsst_a", "cursor:pointer;padding:0 4px"), b.addRule(".gsst_a:hover",
            "text-decoration:none!important"), b.addRule(".gsst_b", "font-size:16px;padding:0 2px;position:relative;" + b.prefix("user-select:none;") + "white-space:nowrap"), b.addRule(".gsst_e", Mb(.55)), b.addRule(".gsst_a:hover .gsst_e,.gsst_a:focus .gsst_e", Mb(.72)), b.addRule(".gsst_a:active .gsst_e", Mb(1)), b.addRule(".gsst_f", "background:white;text-align:left"), b.addRule(".gsst_g", "background-color:white;border:1px solid #ccc;border-top-color:#d9d9d9;" + b.prefix("box-shadow:0 2px 4px rgba(0,0,0,0.2);") + "margin:-1px -3px;padding:0 6px"),
            b.addRule(".gsst_h", "background-color:white;height:1px;margin-bottom:-1px;position:relative;top:-1px"))
        },
        C: function(a) {
            s = a.get(g.$, C);
            r = a.get(g.S, C);
            t = a.get(g.F, C);
            G = a.M(g.qa, C);
            H = a.la()
        },
        I: function(a) {
            U = a.Lb;
            G.sort(v);
            P = F.get("gs_st");
            if (!P) {
                P = sb("gsst_b");
                P.id = F.getId("gs_st");
                if (a = a.Wb)
                    P.style.lineHeight = a + "px";
                w()
            }
            x()
        },
        activate: function(a) {
            J = a;
            (a = a.Zb) && (aa = F.Xa(a));
            if (U) {
                a = 0;
                for (var b; b = G[a++];) {
                    var c=!!O[b.Ne()];
                    if (b.isEnabled() != c) {
                        for (; P.hasChildNodes();)
                            P.removeChild(P.lastChild);
                        w();
                        x();
                        break
                    }
                }
            }
            R = sb("gsst_h");
            M = sb("gsst_f");
            M.dir = "ltr";
            M.appendChild(R);
            r.Oa(13, u)
        },
        getType: function() {
            return g.oa
        },
        B: c,
        A: function() {
            return {
                Vb: a,
                L: b,
                va: l,
                G: e,
                Ah: k,
                Bh: h,
                Ch: m
            }
        },
        Ma: function() {
            var a = {
                de: d,
                L: p,
                lb: q,
                ee: f,
                La: Ja,
                Dc: z
            };
            return [{
                O: Ja,
                C: Ja,
                I: Ja,
                activate: Ja,
                getType: function() {
                    return g.kb
                },
                B: c,
                A: function() {
                    return a
                },
                Ma: Ja,
                N: Ja
            }
            ]
        }
    };
    return C
});
Wb = function() {
    function c(b) {
        return 0 <= a.indexOf(b)
    }
    var a = window.navigator.userAgent, b = {};
    window.opera ? b[2]=!0 : c("MSIE") || c("Trident") ? b[0]=!0 : c("WebKit") ? (b[5]=!0, c("Chrome") ? b[3]=!0 : c("Android") ? b[7]=!0 : c("Safari") && (b[4]=!0), c("iPad") && (b[6]=!0)) : c("Gecko") && (b[1]=!0);
    return b
};
function wc() {
    function c(a) {
        return {
            api: a,
            install: a.a,
            activate: a.b,
            N: a.c,
            hj: a.d,
            cj: a.e,
            X: a.f,
            ea: a.g,
            ga: a.h,
            Ba: a.i,
            Oe: a.j,
            Uh: a.k,
            mj: a.l,
            lj: a.m,
            Th: a.n,
            mb: a.o,
            fj: a.p,
            yd: a.q,
            gj: a.r,
            dj: a.s,
            ob: a.t,
            xd: a.u,
            focus: a.v,
            blur: a.w,
            Wd: a.x,
            fa: a.y,
            Qb: a.z,
            nj: a.aa,
            pa: a.ab,
            search: a.ad,
            oj: a.ae,
            rj: a.af,
            Va: a.ag,
            Aa: a.ah,
            sj: a.ai,
            wd: a.al,
            isActive: a.am,
            Wg: a.an,
            ta: a.ao,
            Ug: a.ap,
            ij: a.aq,
            Ob: a.ar,
            getId: a.as,
            ej: a.at,
            Ub: a.au,
            qj: a.av,
            Pb: a.aw,
            Xd: a.ax,
            qb: a.ay,
            Vg: a.az,
            Lc: a.ba,
            pj: a.bb,
            kj: a.bc,
            Ec: a.bd,
            jj: a.be,
            Tg: a.bf
        }
    }
    return {
        getInstance: function(a,
        b, l, e) {
            try {
                var k = window.google.sbox(a, b, l, e);
                return c(k)
            } catch (h) {
                return null
            }
        },
        translate: function(a) {
            return c(a.api || a)
        }
    }
};
window.google || (window.google = {});
window.google.sbox = function(c, a, b, l) {
    function e() {
        t.N()
    }
    function k(a) {
        O.Qb(a || "")
    }
    function h() {
        return da
    }
    function m() {
        return I
    }
    function d() {
        return O.ea()
    }
    function p() {
        return C.Aa()
    }
    function q() {
        P.J(8)
    }
    function f(a) {
        return R.D(a)
    }
    function z() {
        return ka||!!H && H.ta()
    }
    function v() {
        return M.Sg()
    }
    function u(a) {
        a = a.Yd || ab();
        a = Pb(a);
        void 0 == a.nextSearchboxId && (a.nextSearchboxId = 50);
        return a.nextSearchboxId++
    }
    function w() {
        if (c)
            for (var a = c; a = a.parentNode;) {
                var b = a.dir;
                if (b)
                    return b
            }
        return "ltr"
    }
    function n(a) {
        a =
        eb(a);
        a.hb[35] || (a.jb = "");
        var b = a.gb;
        b ? a.gb = b.toLowerCase() : a.fb=!1;
        a.Ua&&!a.Yb && (a.Ua=!1);
        Xb || (a.rd=!1);
        return a
    }
    function x(a, b) {
        var c = b.exec(a);
        return c && c[1] ? parseInt(c[1], 10) || 0 : 0
    }
    function y() {
        var a = Pb(c), b = Jb(a);
        P.listen(a, "resize", function() {
            var c = Jb(a);
            if (c.Ke != b.Ke || c.Je != b.Je)
                b = c, q()
        })
    }
    function s(a) {
        var b = a.$a, c = b[g.Rd], d = b[g.nc], e = b[g.Ud], f = b[g.Vd], h = b[g.Fa], e = d || f || e;
        b[g.lc] || h || c || e ? (a.$a[g.lc]=!0, a.$a[g.Sd]=!0, e ? (a = Fa(a.Ya), !d || Rb && (Vb || a) || lb && a ? (da = 3, b[g.nc]=!1, b[g.Td]=!1) : da = 2) : da =
        1) : da = 0
    }
    var r, t, G, H, J, P, F, O, D, M, R, S, Z, T, aa, U, C, X, E, I, da, A=!1, ka, N = {
        a: function(b) {
            if (!A) {
                b = n(b);
                I = null == l ? u(b) : l;
                var d = Ob(c), e = w(), f=!!d.getElementById("gs_id" + I), h = ["gssb_c", "gssb_k"];
                b.Rb && h.push(b.Rb);
                h = qc(b.Yd, b.Mf, b.Lf, I, h);
                s(b);
                ka = b.ta;
                t = kc(r, b.Md || {}, {
                    Za: function() {
                        return f
                    },
                    get: function(a) {
                        return d.getElementById(a + I)
                    },
                    Xa: function(a) {
                        return d.getElementById(a)
                    },
                    ke: function() {
                        return a
                    },
                    Da: function() {
                        return e
                    },
                    getId: function(a) {
                        return a + I
                    },
                    Dd: function() {
                        return c
                    }
                }, h, N, b);
                G = t.get(g.yb, N);
                H = t.get(g.ma,
                N);
                J = t.get(g.$, N);
                P = t.get(g.S, N);
                F = t.get(g.T, N);
                O = t.get(g.F, N);
                D = t.get(g.na, N);
                M = t.get(g.Z, N);
                R = t.get(g.ca, N);
                S = t.get(g.za, N);
                Z = t.get(g.xh, N);
                T = t.get(g.ua, N);
                aa = t.get(g.yh, N);
                U = t.get(g.W, N);
                C = t.get(g.P, N);
                X = t.get(g.Fa, N);
                E = t.get(g.ha, N);
                y();
                A=!0
            }
        },
        b: function(a) {
            e();
            a = n(a);
            s(a);
            ka = a.ta;
            t.activate(a)
        },
        c: e,
        d: function() {
            return a
        },
        e: function(a, b) {
            return yb(a, b)
        },
        f: function() {
            return O.X()
        },
        g: d,
        h: function() {
            return C.ga()
        },
        i: function() {
            return C.Ba()
        },
        j: f,
        k: function(a, b) {
            a || (a = R.D(b));
            return La(a)
        },
        l: function() {
            return C.isVisible()
        },
        m: function() {
            return C.zf()
        },
        n: function(a, b) {
            P.listen(a, "click", function(a) {
                E.search(d(), b);
                return Bb(a)
            })
        },
        o: function() {
            F.mb()
        },
        p: function() {
            C.Wa()
        },
        q: function(a) {
            O.yd(a || "")
        },
        r: function() {
            return J.getHeight()
        },
        s: function() {
            O.clear()
        },
        t: function(a) {
            return F.ob(a)
        },
        u: function() {
            O.xd()
        },
        v: function() {
            D.focus()
        },
        w: function() {
            D.blur()
        },
        x: function() {
            return F.Wd()
        },
        y: function() {
            var a = U.fa();
            return a ? Ea(a.Be()) : null
        },
        z: k,
        aa: function(a) {
            a = T.jc(a);
            return Ea(a.Be())
        },
        ab: function() {
            R.reset()
        },
        ad: function(a, b) {
            E.search(a,
            b)
        },
        ae: function() {
            X && X.refresh()
        },
        af: function(a) {
            C.pd(a)
        },
        ag: function() {
            C.Va()
        },
        ah: p,
        ai: q,
        al: function() {
            O.wd()
        },
        am: function() {
            return t && t.isActive()
        },
        an: function(a) {
            H && H.Wg(a)
        },
        ao: z,
        ap: function() {
            return z() && H ? H.Ug() : ""
        },
        aq: function(a, b) {
            return zb(a, b)
        },
        ar: h,
        as: m,
        at: function() {
            X && X.clear()
        },
        au: function(a, b) {
            k(a);
            C.isEnabled() && C.Ub(a, b, !1)
        },
        av: function(a) {
            P.J(15, {
                query: a
            })
        },
        aw: function() {
            return D.Pb()
        },
        ax: function(a) {
            F.Xd(a)
        },
        ay: function(a) {
            J.qb(a)
        },
        az: function(a) {
            return !!Z && Z.Vg(a)
        },
        ba: function() {
            var a,
            b = U.fa();
            if (b) {
                var c = b.Mb();
                c && ((a = c.Lc()) || (a = b.D().getString("o")))
            }
            return a || ""
        },
        bb: function(a, b) {
            return S ? (S.qe(a, b), !0) : !1
        },
        bc: function(a, b) {
            switch (a) {
            case "oq":
            case "gs_l":
                return f(b)[a] || null;
            case "gs_ssp":
                var c;
                n:
                {
                    if ((c = p()) && 46 == c.getType() && (c = c.D().getString("g")))
                        break n;
                    c = null
                }
                return c;
            default:
                return null
            }
        },
        bd: function(a) {
            G && G.Ec(a)
        },
        be: v,
        bf: function(a) {
            return 6 == v()&&!!aa && aa.Tg(a)
        },
        getId: m,
        Ob: h
    };
    r = rc(b);
    (function() {
        var a = window.navigator.userAgent, b = r.Rc(), c = x(a, /Version\/(\d+)/);
        c ||
        (c = x(a, /(?:Android|Chrome|Firefox|Opera|MSIE)[\s\/](\d+)/));
        c || (c = x(a, /Trident[^)]*rv:(\d+)/));
        a = c;
        gb = (lb = b[0]) && 8 >= a;
        fb = lb && 7 >= a;
        Rb = b[1];
        kb = b[2];
        Sb = b[5];
        Tb = b[4];
        Xb = b[3];
        Ub = b[7]
    })();
    Vb = /Mac/.test(navigator && (navigator.platform || navigator.appVersion) || "");
    return N
};
var xc;
(function() {
    function c() {
        return !0
    }
    function a(a) {
        var b = 0 <= a.indexOf("?") ? "&": "?", c = h.Oe();
        return a + b + h.Uh(c)
    }
    var b = /MSIE\s+(\S+)/, l = /Version\/(\S+)/, e = /\/(movie|show)s?($|[?#/])/i, k = /\/results\?(.*&)?search_type=(movies|shows)($|[&#])/i, h, m, d, p, q, f, z = {
        a: function() {
            return d
        },
        b: function(a, b) {
            c(h.Oe(b)) && m.submit()
        },
        d: function(b) {
            f(a(b))
        },
        e: a,
        r: function(a) {
            a.addRule(".gsfi", "font-size:16px");
            a.addRule(".gsfs", "font-size:16px");
            a.addRule("a.gssb_j", "font-size:12px;color:#03c");
            a.addRule(".gssb_a,.gssb_a td", "line-height:20px");
            a.addRule(".gssb_a", "padding:0 6px");
            a.addRule(".gssb_c", "z-index:2000000007");
            a.addRule(".gssb_i td", "background:#eee");
            a.addRule(".gssb_k", "z-index:2000000006");
            a.addRule(".gssb_l", "margin:2px 0");
            a.addRule(".gsib_a", "padding:0 4px");
            a.addRule(".gsok_a", "padding:0");
            a.addRule(".gsok_a img", "display:block");
            a.addRule(".gsfe_b", ["border:1px solid #1c62b9;", a.prefix("box-shadow:inset 0 1px 2px rgba(0,0,0,0.3);"), "outline:none;"].join(""))
        }
    };
    (function() {
        function a(b) {
            return (b = c.match(b)) ? parseInt(b[1],
            10) : NaN
        }
        var c = window.navigator.userAgent;
        d = Wb();
        d[2] ? q = a(l) : d[0] && (q = a(b));
        p = 0 <= c.indexOf("Windows")
    })();
    xc = function(a, b, l, n, x, y, s) {
        m = a;
        f = y;
        s && (c = s);
        y = wc();
        s = gc().getInstance();
        var r = "youtube";
        s.clientName = r;
        s.pb = r;
        s.Od = "yt";
        s.Ya = n.REQUEST_LANGUAGE;
        s.we = n.REQUEST_DOMAIN;
        s.Ih=!1;
        s.eb = 0;
        s.ud=!1;
        s.Ua=!1;
        s.Ld=!1;
        s.Fc=!0;
        s.fc = "gsfi";
        s.kc = "gsfs";
        s.Jh=!0;
        r = window.location.href;
        r = e.test(r) || k.test(r);
        s.Pd = r;
        s.fb = n.HAS_ON_SCREEN_KEYBOARD;
        s.gb = n.REQUEST_LANGUAGE;
        s.oe = "//www.gstatic.com/inputtools/images";
        s.pe =
        "youtube";
        s.fe=!0;
        s.Pa = "fixed";
        n.PQ && (s.Qd = n.PQ);
        s.jb = n.PSUGGEST_TOKEN;
        s.Ab = n.SESSION_INDEX;
        s.le = x.SUGGESTION_DISMISSED_LABEL;
        s.ne = x.SUGGESTION_DISMISS_LABEL;
        s.hb = za([0, 33, 35]);
        s.Oc = "masthead-search-terms";
        s.Wb = 30;
        s.Md = {};
        d[2] || (s.ie=!0);
        x = d[2];
        r = d[4];
        - 1 != window.location.href.indexOf("/watch?") && (x || p && r) && (r = "youtube-reduced", s.clientName = r, s.pb = r, s.Hh = 4);
        s.Zb = s.Oc;
        x =- 3;
        p && d[0] && 8 == q && (x =- 5);
        s.wb = [x, 0, 0];
        x = [0];
        d[0] && 8 == q && (x[0] =- 1);
        s.he = x;
        Fa(n.REQUEST_LANGUAGE) && (s.Kd=!0);
        if (n = n.SUGG_EXP_ID || n.EXPERIMENT_STR)
            s.ic =
            n;
        h ? h.activate(s) : (h = y.getInstance(b, a, z), h.install(s), l && (h.Th(l, 12), l.onclick = null))
    };
    return z
})();
function yc(c, a, b) {
    c.push(encodeURIComponent(a) + "=" + encodeURIComponent(b))
}
function zc(c) {
    var a = c.type;
    if (!na(a))
        return null;
    switch (a.toLowerCase()) {
    case "checkbox":
    case "radio":
        return c.checked ? c.value : null;
    case "select-one":
        return a = c.selectedIndex, 0 <= a ? c.options[a].value : null;
    case "select-multiple":
        for (var a = [], b, l = 0; b = c.options[l]; l++)
            b.selected && a.push(b.value);
        return a.length ? a : null;
    default:
        return na(c.value) ? c.value : null
    }
};
var Ac;
n: {
    var Bc = ga.navigator;
    if (Bc) {
        var Cc = Bc.userAgent;
        if (Cc) {
            Ac = Cc;
            break n
        }
    }
    Ac = ""
};
var Dc =- 1 != Ac.indexOf("Opera")||-1 != Ac.indexOf("OPR"), Ec =- 1 != Ac.indexOf("Trident")||-1 != Ac.indexOf("MSIE"), Fc =- 1 != Ac.indexOf("Gecko")&&-1 == Ac.toLowerCase().indexOf("webkit")&&!( - 1 != Ac.indexOf("Trident")||-1 != Ac.indexOf("MSIE")), Gc =- 1 != Ac.toLowerCase().indexOf("webkit");
(function() {
    var c = "", a;
    if (Dc && ga.opera)
        return c = ga.opera.version, "function" == wa(c) ? c() : c;
    Fc ? a = /rv\:([^\);]+)(\)|;)/ : Ec ? a = /\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/ : Gc && (a = /WebKit\/(\S+)/);
    a && (c = (c = a.exec(Ac)) ? c[1] : "");
    return Ec && (a = (a = ga.document) ? a.documentMode : void 0, a > parseFloat(c)) ? String(a) : c
})();
var Hc = /^(?:([^:/?#.]+):)?(?:\/\/(?:([^/?#]*)@)?([^/#?]*?)(?::([0-9]+))?(?=[/#?]|$))?([^?#]+)?(?:\?([^#]*))?(?:#(.*))?$/;
function Ic(c) {
    if (Jc) {
        Jc=!1;
        var a = ga.location;
        if (a) {
            var b = a.href;
            if (b && (b = Kc(b)) && b != a.hostname)
                throw Jc=!0, Error();
        }
    }
    return c.match(Hc)
}
var Jc = Gc;
function Kc(c) {
    return (c = Ic(c)[3] || null) ? decodeURI(c) : c
}
function Lc(c, a, b) {
    if ("array" == wa(a))
        for (var l = 0; l < a.length; l++)
            Lc(c, String(a[l]), b);
    else 
        null != a && b.push("&", c, "" === a ? "" : "=", encodeURIComponent(String(a)))
};
function Mc(c) {
    this.G = c
}
var Nc = /\s*;\s*/;
Mc.prototype.isEnabled = function() {
    return navigator.cookieEnabled
};
Mc.prototype.set = function(c, a, b, l, e, k) {
    if (/[;=\s]/.test(c))
        throw Error('Invalid cookie name "' + c + '"');
    if (/[;\r\n]/.test(a))
        throw Error('Invalid cookie value "' + a + '"');
    na(b) || (b =- 1);
    e = e ? ";domain=" + e : "";
    l = l ? ";path=" + l : "";
    k = k ? ";secure" : "";
    b = 0 > b ? "" : 0 == b ? ";expires=" + (new Date(1970, 1, 1)).toUTCString() : ";expires=" + (new Date(xa() + 1E3 * b)).toUTCString();
    this.G.cookie = c + "=" + a + e + l + b + k
};
Mc.prototype.get = function(c, a) {
    for (var b = c + "=", l = (this.G.cookie || "").split(Nc), e = 0, k; k = l[e]; e++) {
        if (0 == k.lastIndexOf(b, 0))
            return k.substr(b.length);
        if (k == c)
            return ""
    }
    return a
};
Mc.prototype.remove = function(c, a, b) {
    var l = na(this.get(c));
    this.set(c, "", 0, a, b);
    return l
};
Mc.prototype.clear = function() {
    for (var c = (this.G.cookie || "").split(Nc), a = [], b = [], l, e, k = 0; e = c[k]; k++)
        l = e.indexOf("="), - 1 == l ? (a.push(""), b.push(e)) : (a.push(e.substring(0, l)), b.push(e.substring(l + 1)));
    for (c = a.length - 1; 0 <= c; c--)
        this.remove(a[c])
};
var Oc = new Mc(document);
Oc.va = 3950;
var Pc = window.yt && window.yt.config_ || {};
ya("yt.config_", Pc);
ya("yt.tokens_", window.yt && window.yt.tokens_ || {});
ya("yt.msgs_", window.yt && window.yt.msgs_ || {});
function Qc(c) {
    return c in Pc ? Pc[c] : void 0
};
function Rc(c, a) {
    var b = Qc("EVENT_ID");
    if (b && (a.ei = b, b = Kc(c), b == Kc(window.location.href) ||!b && 0 == c.lastIndexOf("/", 0))) {
        var l = Ic(c), b = l[5], e = l[6], l = l[7], k = "";
        b && (k += b);
        e && (k += "?" + e);
        l && (k += "#" + l);
        b = k;
        e = b.indexOf("#");
        if (b = 0 > e ? b : b.substr(0, e)) {
            for (l = e = 0; l < b.length; ++l)
                e = 31 * e + b.charCodeAt(l), e%=4294967296;
            var b = "s_tempdata-" + e, h;
            if (a) {
                e = [];
                for (h in a)
                    Lc(h, a[h], e);
                e[0] = "";
                h = e.join("")
            } else 
                h = "";
            b = "" + b;
            Oc.set(b, h, 5, "/", "youtube.com")
        }
    }
}
function Sc(c, a) {
    Rc(c, a ? {
        feature: a
    } : {});
    var b = va("yt.window.navigate");
    try {
        b(c)
    } catch (l) {
        window.location = c
    }
}
function Tc(c) {
    for (var a = document.getElementById("masthead-search"), b = [], l = a.elements, e, k = 0; e = l[k]; k++)
        if (e.form == a&&!e.disabled && "fieldset" != e.tagName.toLowerCase()) {
            var h = e.name;
            switch (e.type.toLowerCase()) {
            case "file":
            case "submit":
            case "reset":
            case "button":
                break;
            case "select-multiple":
                e = zc(e);
                if (null != e)
                    for (var m, d = 0; m = e[d]; d++)
                        yc(b, h, m);
                        break;
                    default:
                        m = zc(e), null != m && yc(b, h, m)
                    }
        }
    l = a.getElementsByTagName("input");
    for (k = 0; e = l[k]; k++)
        e.form == a && "image" == e.type.toLowerCase() && (h = e.name, yc(b, h, e.value),
        yc(b, h + ".x", "0"), yc(b, h + ".y", "0"));
    b = b.join("&").replace(/%20/g, "+");
    a = a.action + "?" + b;
    Rc(a, c);
    c=!!Qc("SPF_SEARCH_BOX");
    if (!va("ytspf.enabled") ||!c)
        return !0;
    c = va("yt.window.navigate");
    try {
        c(a)
    } catch (p) {
        return !0
    }
    return !1
};
ya("searchbox.yt.install", xc);
ya("yt.www.masthead.searchbox.init", function() {
    var c = Qc("SBOX_SETTINGS");
    if (c) {
        var a = document.getElementById("masthead-search"), b = a.search_query, l = document.getElementById("search-btn"), e = Qc("SBOX_LABELS");
        a && c && e && window.setTimeout(function() {
            va("searchbox.yt.install")(a, b, l, c, e, Sc, Tc)
        }, 100)
    }
});
})();

