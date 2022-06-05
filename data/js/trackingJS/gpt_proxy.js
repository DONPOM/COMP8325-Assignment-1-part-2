// Copyright 2011 Google Inc. All Rights Reserved.
(function() {
    var k, l = this, m = function(a, b) {
        for (var c = a.split("."), d = b || l, e; e = c.shift();)
            if (null != d[e])
                d = d[e];
            else 
                return null;
        return d
    }, aa = function(a) {
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
                                if ("[object Function]" ==
                                c || "undefined" != typeof a.call && "undefined" != typeof a.propertyIsEnumerable&&!a.propertyIsEnumerable("call"))
                                    return "function"
            } else 
                return "null";
                else if ("function" == b && "undefined" == typeof a.call)
    return "object";
return b
}, p = function(a) {
    return "array" == aa(a)
}, ba = function(a) {
    var b = aa(a);
    return "array" == b || "object" == b && "number" == typeof a.length
}, q = function(a) {
    return "string" == typeof a
}, r = function(a) {
    return "function" == aa(a)
}, ca = function(a) {
    var b = typeof a;
    return "object" == b && null != a || "function" == b
}, da = function(a,
b, c) {
    return a.call.apply(a.bind, arguments)
}, ea = function(a, b, c) {
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
}, s = function(a, b, c) {
    s = Function.prototype.bind&&-1 != Function.prototype.bind.toString().indexOf("native code") ? da : ea;
    return s.apply(null, arguments)
}, fa = function(a, b) {
    var c = Array.prototype.slice.call(arguments,
    1);
    return function() {
        var b = c.slice();
        b.push.apply(b, arguments);
        return a.apply(this, b)
    }
}, t = function(a, b) {
    function c() {}
    c.prototype = b.prototype;
    a.s = b.prototype;
    a.prototype = new c;
    a.Q = function(a, c, f) {
        return b.prototype[c].apply(a, Array.prototype.slice.call(arguments, 2))
    }
};
Function.prototype.bind = Function.prototype.bind || function(a, b) {
    if (1 < arguments.length) {
        var c = Array.prototype.slice.call(arguments, 1);
        c.unshift(this, a);
        return s.apply(null, c)
    }
    return s(this, a)
};
var ga = String.prototype.trim ? function(a) {
    return a.trim()
}
: function(a) {
    return a.replace(/^[\s\xa0]+|[\s\xa0]+$/g, "")
}, oa = function(a) {
    if (!ha.test(a))
        return a;
    - 1 != a.indexOf("&") && (a = a.replace(ia, "&amp;"));
    - 1 != a.indexOf("<") && (a = a.replace(ja, "&lt;"));
    - 1 != a.indexOf(">") && (a = a.replace(ka, "&gt;"));
    - 1 != a.indexOf('"') && (a = a.replace(la, "&quot;"));
    - 1 != a.indexOf("'") && (a = a.replace(ma, "&#39;"));
    - 1 != a.indexOf("\x00") && (a = a.replace(na, "&#0;"));
    return a
}, ia = /&/g, ja = /</g, ka = />/g, la = /"/g, ma = /'/g, na = /\x00/g, ha = /[\x00&<>"']/,
pa = function(a, b) {
    return a < b?-1 : a > b ? 1 : 0
}, qa = 2147483648 * Math.random() | 0;
var u = Array.prototype, ra = u.indexOf ? function(a, b, c) {
    return u.indexOf.call(a, b, c)
}
: function(a, b, c) {
    c = null == c ? 0 : 0 > c ? Math.max(0, a.length + c) : c;
    if (q(a))
        return q(b) && 1 == b.length ? a.indexOf(b, c) : - 1;
    for (; c < a.length; c++)
        if (c in a && a[c] === b)
            return c;
    return - 1
}, sa = u.forEach ? function(a, b, c) {
    u.forEach.call(a, b, c)
}
: function(a, b, c) {
    for (var d = a.length, e = q(a) ? a.split("") : a, f = 0; f < d; f++)
        f in e && b.call(c, e[f], f, a)
}, ua = function(a) {
    var b = ta;
    n: {
        for (var c = b.length, d = q(b) ? b.split("") : b, e = 0; e < c; e++)
            if (e in d && a.call(void 0, d[e],
            e, b)) {
                a = e;
                break n
            }
        a =- 1
    }
    return 0 > a ? null : q(b) ? b.charAt(a) : b[a]
}, va = function(a) {
    return u.concat.apply(u, arguments)
}, wa = function(a) {
    var b = a.length;
    if (0 < b) {
        for (var c = Array(b), d = 0; d < b; d++)
            c[d] = a[d];
        return c
    }
    return []
};
var xa = function(a, b, c) {
    for (var d in a)
        b.call(c, a[d], d, a)
}, ya = "constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" "), za = function(a, b) {
    for (var c, d, e = 1; e < arguments.length; e++) {
        d = arguments[e];
        for (c in d)
            a[c] = d[c];
        for (var f = 0; f < ya.length; f++)
            c = ya[f], Object.prototype.hasOwnProperty.call(d, c) && (a[c] = d[c])
    }
}, Aa = function(a) {
    var b = arguments.length;
    if (1 == b && p(arguments[0]))
        return Aa.apply(null, arguments[0]);
    for (var c = {}, d = 0; d < b; d++)
        c[arguments[d]]=!0;
    return c
};
Aa("area base br col command embed hr img input keygen link meta param source track wbr".split(" "));
Aa("action", "cite", "data", "formaction", "href", "manifest", "poster", "src");
Aa("link", "script", "style");
var Ba = "StopIteration"in l ? l.StopIteration: Error("StopIteration"), Ca = function() {};
Ca.prototype.next = function() {
    throw Ba;
};
Ca.prototype.q = function() {
    return this
};
var w = function(a, b) {
    this.d = {};
    this.a = [];
    this.e = this.b = 0;
    var c = arguments.length;
    if (1 < c) {
        if (c%2)
            throw Error("Uneven number of arguments");
        for (var d = 0; d < c; d += 2)
            v(this, arguments[d], arguments[d + 1])
    } else if (a) {
        var e;
        if (a instanceof w)
            e = a.u(), d = a.t();
        else {
            var c = [], f = 0;
            for (e in a)
                c[f++] = e;
            e = c;
            c = [];
            f = 0;
            for (d in a)
                c[f++] = a[d];
            d = c
        }
        for (c = 0; c < e.length; c++)
            v(this, e[c], d[c])
    }
};
w.prototype.t = function() {
    Da(this);
    for (var a = [], b = 0; b < this.a.length; b++)
        a.push(this.d[this.a[b]]);
    return a
};
w.prototype.u = function() {
    Da(this);
    return this.a.concat()
};
var Da = function(a) {
    if (a.b != a.a.length) {
        for (var b = 0, c = 0; b < a.a.length;) {
            var d = a.a[b];
            x(a.d, d) && (a.a[c++] = d);
            b++
        }
        a.a.length = c
    }
    if (a.b != a.a.length) {
        for (var e = {}, c = b = 0; b < a.a.length;)
            d = a.a[b], x(e, d) || (a.a[c++] = d, e[d] = 1), b++;
        a.a.length = c
    }
}, y = function(a, b) {
    return x(a.d, b) ? a.d[b] : void 0
}, v = function(a, b, c) {
    x(a.d, b) || (a.b++, a.a.push(b), a.e++);
    a.d[b] = c
};
w.prototype.forEach = function(a, b) {
    for (var c = this.u(), d = 0; d < c.length; d++) {
        var e = c[d];
        a.call(b, y(this, e), e, this)
    }
};
w.prototype.clone = function() {
    return new w(this)
};
w.prototype.q = function(a) {
    Da(this);
    var b = 0, c = this.a, d = this.d, e = this.e, f = this, g = new Ca;
    g.next = function() {
        for (; ;) {
            if (e != f.e)
                throw Error("The map has changed since the iterator was created");
            if (b >= c.length)
                throw Ba;
            var g = c[b++];
            return a ? g : d[g]
        }
    };
    return g
};
var x = function(a, b) {
    return Object.prototype.hasOwnProperty.call(a, b)
};
var z;
n: {
    var Ea = l.navigator;
    if (Ea) {
        var Fa = Ea.userAgent;
        if (Fa) {
            z = Fa;
            break n
        }
    }
    z = ""
};
var Ga =- 1 != z.indexOf("Opera")||-1 != z.indexOf("OPR"), A =- 1 != z.indexOf("Trident")||-1 != z.indexOf("MSIE"), B =- 1 != z.indexOf("Gecko")&&-1 == z.toLowerCase().indexOf("webkit")&&!( - 1 != z.indexOf("Trident")||-1 != z.indexOf("MSIE")), Ha =- 1 != z.toLowerCase().indexOf("webkit"), Ia = function() {
    var a = l.document;
    return a ? a.documentMode : void 0
}, Ja = function() {
    var a = "", b;
    if (Ga && l.opera)
        return a = l.opera.version, r(a) ? a() : a;
    B ? b = /rv\:([^\);]+)(\)|;)/ : A ? b = /\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/ : Ha && (b = /WebKit\/(\S+)/);
    b && (a = (a = b.exec(z)) ?
    a[1] : "");
    return A && (b = Ia(), b > parseFloat(a)) ? String(b) : a
}(), Ka = {}, C = function(a) {
    var b;
    if (!(b = Ka[a])) {
        b = 0;
        for (var c = ga(String(Ja)).split("."), d = ga(String(a)).split("."), e = Math.max(c.length, d.length), f = 0; 0 == b && f < e; f++) {
            var g = c[f] || "", h = d[f] || "", n = RegExp("(\\d*)(\\D*)", "g"), Kb = RegExp("(\\d*)(\\D*)", "g");
            do {
                var N = n.exec(g) || ["", "", ""], O = Kb.exec(h) || ["", "", ""];
                if (0 == N[0].length && 0 == O[0].length)
                    break;
                b = pa(0 == N[1].length ? 0 : parseInt(N[1], 10), 0 == O[1].length ? 0 : parseInt(O[1], 10)) || pa(0 == N[2].length, 0 == O[2].length) ||
                pa(N[2], O[2])
            }
            while (0 == b)
            }
        b = Ka[a] = 0 <= b
    }
    return b
}, La = l.document, Ma = La && A ? Ia() || ("CSS1Compat" == La.compatMode ? parseInt(Ja, 10) : 5) : void 0;
var Na=!A || A && 9 <= Ma;
!B&&!A || A && A && 9 <= Ma || B && C("1.9.1");
A && C("9");
var Pa = function(a, b) {
    xa(b, function(b, d) {
        "style" == d ? a.style.cssText = b : "class" == d ? a.className = b : "for" == d ? a.htmlFor = b : d in Oa ? a.setAttribute(Oa[d], b) : 0 == d.lastIndexOf("aria-", 0) || 0 == d.lastIndexOf("data-", 0) ? a.setAttribute(d, b) : a[d] = b
    })
}, Oa = {
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
}, Ra = function(a, b, c) {
    var d = arguments, e = document,
    f = d[0], g = d[1];
    if (!Na && g && (g.name || g.type)) {
        f = ["<", f];
        g.name && f.push(' name="', oa(g.name), '"');
        if (g.type) {
            f.push(' type="', oa(g.type), '"');
            var h = {};
            za(h, g);
            delete h.type;
            g = h
        }
        f.push(">");
        f = f.join("")
    }
    f = e.createElement(f);
    g && (q(g) ? f.className = g : p(g) ? f.className = g.join(" ") : Pa(f, g));
    2 < d.length && Qa(e, f, d);
    return f
}, Qa = function(a, b, c) {
    function d(c) {
        c && b.appendChild(q(c) ? a.createTextNode(c) : c)
    }
    for (var e = 2; e < c.length; e++) {
        var f = c[e];
        !ba(f) || ca(f) && 0 < f.nodeType ? d(f) : sa(Sa(f) ? wa(f) : f, d)
    }
}, Sa = function(a) {
    if (a &&
    "number" == typeof a.length) {
        if (ca(a))
            return "function" == typeof a.item || "string" == typeof a.item;
        if (r(a))
            return "function" == typeof a.item
    }
    return !1
};
var D = function() {
    this.q = this.q;
    this.a = this.a
};
D.prototype.q=!1;
D.prototype.w = function() {
    this.q || (this.q=!0, this.d())
};
D.prototype.d = function() {
    if (this.a)
        for (; this.a.length;)
            this.a.shift()()
};
var Ta = function(a) {
    a && "function" == typeof a.w && a.w()
};
var Ua = function(a) {
    Ua[" "](a);
    return a
};
Ua[" "] = function() {};
var Va=!A || A && 9 <= Ma, Wa = A&&!C("9");
!Ha || C("528");
B && C("1.9b") || A && C("8") || Ga && C("9.5") || Ha && C("528");
B&&!C("8") || A && C("9");
var E = function(a, b) {
    this.type = a;
    this.a = this.d = b;
    this.J=!0
};
E.prototype.w = function() {};
E.prototype.e = function() {
    this.J=!1
};
var F = function(a, b) {
    E.call(this, a ? a.type : "");
    this.b = this.state = this.a = this.d = null;
    if (a) {
        this.type = a.type;
        this.d = a.target || a.srcElement;
        this.a = b;
        var c = a.relatedTarget;
        if (c && B)
            try {
                Ua(c.nodeName)
            } catch (d) {}
        this.state = a.state;
        this.b = a;
        a.defaultPrevented && this.e()
    }
};
t(F, E);
F.prototype.e = function() {
    F.s.e.call(this);
    var a = this.b;
    if (a.preventDefault)
        a.preventDefault();
    else if (a.returnValue=!1, Wa)
        try {
            if (a.ctrlKey || 112 <= a.keyCode && 123 >= a.keyCode)
                a.keyCode =- 1
    } catch (b) {}
};
var Xa = "closure_listenable_" + (1E6 * Math.random() | 0), G = function(a) {
    return !(!a ||!a[Xa])
}, Ya = 0;
var Za = function(a, b, c, d, e) {
    this.k = a;
    this.a = null;
    this.src = b;
    this.type = c;
    this.A=!!d;
    this.C = e;
    this.K=++Ya;
    this.o = this.B=!1
}, $a = function(a) {
    a.o=!0;
    a.k = null;
    a.a = null;
    a.src = null;
    a.C = null
};
var ab = function(a) {
    this.src = a;
    this.a = {};
    this.d = 0
}, cb = function(a, b, c, d, e, f) {
    var g = b.toString();
    b = a.a[g];
    b || (b = a.a[g] = [], a.d++);
    var h = bb(b, c, e, f);
    - 1 < h ? (a = b[h], d || (a.B=!1)) : (a = new Za(c, a.src, g, !!e, f), a.B = d, b.push(a));
    return a
}, db = function(a, b) {
    var c = b.type;
    if (!(c in a.a))
        return !1;
    var d = a.a[c], e = ra(d, b), f;
    (f = 0 <= e) && u.splice.call(d, e, 1);
    f && ($a(b), 0 == a.a[c].length && (delete a.a[c], a.d--));
    return f
}, eb = function(a, b, c, d, e) {
    a = a.a[b.toString()];
    b =- 1;
    a && (b = bb(a, c, d, e));
    return - 1 < b ? a[b] : null
}, bb = function(a, b, c,
d) {
    for (var e = 0; e < a.length; ++e) {
        var f = a[e];
        if (!f.o && f.k == b && f.A==!!c && f.C == d)
            return e
    }
    return - 1
};
var fb = "closure_lm_" + (1E6 * Math.random() | 0), gb = {}, hb = 0, ib = function(a, b, c, d, e) {
    if (p(b)) {
        for (var f = 0; f < b.length; f++)
            ib(a, b[f], c, d, e);
        return null
    }
    c = jb(c);
    return G(a) ? a.e(b, c, d, e) : kb(a, b, c, !1, d, e)
}, kb = function(a, b, c, d, e, f) {
    if (!b)
        throw Error("Invalid event type");
    var g=!!e, h = H(a);
    h || (a[fb] = h = new ab(a));
    c = cb(h, b, c, d, e, f);
    if (c.a)
        return c;
    d = lb();
    c.a = d;
    d.src = a;
    d.k = c;
    a.addEventListener ? a.addEventListener(b.toString(), d, g) : a.attachEvent(mb(b.toString()), d);
    hb++;
    return c
}, lb = function() {
    var a = nb, b = Va ? function(c) {
        return a.call(b.src,
        b.k, c)
    }
    : function(c) {
        c = a.call(b.src, b.k, c);
        if (!c)
            return c
    };
    return b
}, ob = function(a, b, c, d, e) {
    if (p(b)) {
        for (var f = 0; f < b.length; f++)
            ob(a, b[f], c, d, e);
        return null
    }
    c = jb(c);
    return G(a) ? cb(a.g, String(b), c, !0, d, e) : kb(a, b, c, !0, d, e)
}, pb = function(a, b, c, d, e) {
    if (p(b))
        for (var f = 0; f < b.length; f++)
            pb(a, b[f], c, d, e);
    else 
        c = jb(c), G(a) ? a.I(b, c, d, e) : a && (a = H(a)) && (b = eb(a, b, c, !!d, e)) && I(b)
}, I = function(a) {
    if ("number" == typeof a ||!a || a.o)
        return !1;
    var b = a.src;
    if (G(b))
        return db(b.g, a);
    var c = a.type, d = a.a;
    b.removeEventListener ? b.removeEventListener(c,
    d, a.A) : b.detachEvent && b.detachEvent(mb(c), d);
    hb--;
    (c = H(b)) ? (db(c, a), 0 == c.d && (c.src = null, b[fb] = null)) : $a(a);
    return !0
}, mb = function(a) {
    return a in gb ? gb[a] : gb[a] = "on" + a
}, rb = function(a, b, c, d) {
    var e = 1;
    if (a = H(a))
        if (b = a.a[b.toString()])
            for (b = b.concat(), a = 0; a < b.length; a++) {
                var f = b[a];
                f && f.A == c&&!f.o && (e&=!1 !== qb(f, d))
            }
    return Boolean(e)
}, qb = function(a, b) {
    var c = a.k, d = a.C || a.src;
    a.B && I(a);
    return c.call(d, b)
}, nb = function(a, b) {
    if (a.o)
        return !0;
    if (!Va) {
        var c = b || m("window.event"), d = new F(c, this), e=!0;
        if (!(0 > c.keyCode ||
        void 0 != c.returnValue)) {
            n:
            {
                var f=!1;
                if (0 == c.keyCode)
                    try {
                        c.keyCode =- 1;
                        break n
                } catch (g) {
                    f=!0
                }
                if (f || void 0 == c.returnValue)
                    c.returnValue=!0
            }
            c = [];
            for (f = d.a; f; f = f.parentNode)
                c.push(f);
            for (var f = a.type, h = c.length - 1; 0 <= h; h--)
                d.a = c[h], e&=rb(c[h], f, !0, d);
            for (h = 0; h < c.length; h++)
                d.a = c[h], e&=rb(c[h], f, !1, d)
            }
        return e
    }
    return qb(a, new F(b, this))
}, H = function(a) {
    a = a[fb];
    return a instanceof ab ? a : null
}, sb = "__closure_events_fn_" + (1E9 * Math.random()>>>0), jb = function(a) {
    if (r(a))
        return a;
    a[sb] || (a[sb] = function(b) {
        return a.handleEvent(b)
    });
    return a[sb]
};
var J = function(a) {
    D.call(this);
    this.i = a;
    this.f = {}
};
t(J, D);
var tb = [];
J.prototype.e = function(a, b, c, d) {
    p(b) || (b && (tb[0] = b.toString()), b = tb);
    for (var e = 0; e < b.length; e++) {
        var f = ib(a, b[e], c || this.handleEvent, d ||!1, this.i || this);
        if (!f)
            break;
        this.f[f.K] = f
    }
    return this
};
J.prototype.I = function(a, b, c, d, e) {
    if (p(b))
        for (var f = 0; f < b.length; f++)
            this.I(a, b[f], c, d, e);
    else 
        c = c || this.handleEvent, e = e || this.i || this, c = jb(c), d=!!d, b = G(a) ? eb(a.g, String(b), c, d, e) : a ? (a = H(a)) ? eb(a, b, c, d, e) : null : null, b && (I(b), delete this.f[b.K]);
    return this
};
J.prototype.d = function() {
    J.s.d.call(this);
    xa(this.f, I);
    this.f = {}
};
J.prototype.handleEvent = function() {
    throw Error("EventHandler.handleEvent not implemented");
};
var ub = /^(?:([^:/?#.]+):)?(?:\/\/(?:([^/?#]*)@)?([^/#?]*?)(?::([0-9]+))?(?=[/#?]|$))?([^?#]+)?(?:\?([^#]*))?(?:#(.*))?$/, wb = function(a) {
    if (vb) {
        vb=!1;
        var b = l.location;
        if (b) {
            var c = b.href;
            if (c && (c = (c = wb(c)[3] || null) ? decodeURI(c) : c) && c != b.hostname)
                throw vb=!0, Error();
        }
    }
    return a.match(ub)
}, vb = Ha;
var K = function(a, b) {
    var c;
    a instanceof K ? (this.l = void 0 !== b ? b : a.l, xb(this, a.j), this.G = a.G, this.p = a.p, yb(this, a.F), this.D = a.D, zb(this, a.a.clone()), this.v = a.v) : a && (c = wb(String(a))) ? (this.l=!!b, xb(this, c[1] || "", !0), this.G = L(c[2] || ""), this.p = L(c[3] || "", !0), yb(this, c[4]), this.D = L(c[5] || "", !0), zb(this, c[6] || "", !0), this.v = L(c[7] || "")) : (this.l=!!b, this.a = new M(null, 0, this.l))
};
k = K.prototype;
k.j = "";
k.G = "";
k.p = "";
k.F = null;
k.D = "";
k.v = "";
k.l=!1;
k.toString = function() {
    var a = [], b = this.j;
    b && a.push(P(b, Ab, !0), ":");
    if (b = this.p) {
        a.push("//");
        var c = this.G;
        c && a.push(P(c, Ab, !0), "@");
        a.push(encodeURIComponent(String(b)).replace(/%25([0-9a-fA-F]{2})/g, "%$1"));
        b = this.F;
        null != b && a.push(":", String(b))
    }
    if (b = this.D)
        this.p && "/" != b.charAt(0) && a.push("/"), a.push(P(b, "/" == b.charAt(0) ? Bb : Cb, !0));
    (b = this.a.toString()) && a.push("?", b);
    (b = this.v) && a.push("#", P(b, Db));
    return a.join("")
};
k.clone = function() {
    return new K(this)
};
var xb = function(a, b, c) {
    a.j = c ? L(b, !0) : b;
    a.j && (a.j = a.j.replace(/:$/, ""))
}, yb = function(a, b) {
    if (b) {
        b = Number(b);
        if (isNaN(b) || 0 > b)
            throw Error("Bad port number " + b);
        a.F = b
    } else 
        a.F = null
}, zb = function(a, b, c) {
    b instanceof M ? (a.a = b, Eb(a.a, a.l)) : (c || (b = P(b, Fb)), a.a = new M(b, 0, a.l))
}, Gb = function(a) {
    return a.v
}, L = function(a, b) {
    return a ? b ? decodeURI(a) : decodeURIComponent(a) : ""
}, P = function(a, b, c) {
    return q(a) ? (a = encodeURI(a).replace(b, Hb), c && (a = a.replace(/%25([0-9a-fA-F]{2})/g, "%$1")), a) : null
}, Hb = function(a) {
    a = a.charCodeAt(0);
    return "%" + (a>>4 & 15).toString(16) + (a & 15).toString(16)
}, Ab = /[#\/\?@]/g, Cb = /[\#\?:]/g, Bb = /[\#\?]/g, Fb = /[\#\?@]/g, Db = /#/g, M = function(a, b, c) {
    this.d = a || null;
    this.e=!!c
}, R = function(a) {
    if (!a.a && (a.a = new w, a.b = 0, a.d))
        for (var b = a.d.split("&"), c = 0; c < b.length; c++) {
            var d = b[c].indexOf("="), e = null, f = null;
            0 <= d ? (e = b[c].substring(0, d), f = b[c].substring(d + 1)) : e = b[c];
            e = decodeURIComponent(e.replace(/\+/g, " "));
            e = Q(a, e);
            d = a;
            f = f ? decodeURIComponent(f.replace(/\+/g, " ")) : "";
            R(d);
            d.d = null;
            var e = Q(d, e), g = y(d.a, e);
            g || v(d.a, e,
            g = []);
            g.push(f);
            d.b++
        }
};
M.prototype.a = null;
M.prototype.b = null;
var Ib = function(a, b) {
    R(a);
    b = Q(a, b);
    if (x(a.a.d, b)) {
        a.d = null;
        a.b -= y(a.a, b).length;
        var c = a.a;
        x(c.d, b) && (delete c.d[b], c.b--, c.e++, c.a.length > 2 * c.b && Da(c))
    }
}, Jb = function(a, b) {
    R(a);
    b = Q(a, b);
    return x(a.a.d, b)
};
M.prototype.u = function() {
    R(this);
    for (var a = this.a.t(), b = this.a.u(), c = [], d = 0; d < b.length; d++)
        for (var e = a[d], f = 0; f < e.length; f++)
            c.push(b[d]);
    return c
};
M.prototype.t = function(a) {
    R(this);
    var b = [];
    if (q(a))
        Jb(this, a) && (b = va(b, y(this.a, Q(this, a))));
    else {
        a = this.a.t();
        for (var c = 0; c < a.length; c++)
            b = va(b, a[c])
    }
    return b
};
M.prototype.toString = function() {
    if (this.d)
        return this.d;
    if (!this.a)
        return "";
    for (var a = [], b = this.a.u(), c = 0; c < b.length; c++)
        for (var d = b[c], e = encodeURIComponent(String(d)), d = this.t(d), f = 0; f < d.length; f++) {
            var g = e;
            "" !== d[f] && (g += "=" + encodeURIComponent(String(d[f])));
            a.push(g)
        }
    return this.d = a.join("&")
};
M.prototype.clone = function() {
    var a = new M;
    a.d = this.d;
    this.a && (a.a = this.a.clone(), a.b = this.b);
    return a
};
var Q = function(a, b) {
    var c = String(b);
    a.e && (c = c.toLowerCase());
    return c
}, Eb = function(a, b) {
    b&&!a.e && (R(a), a.d = null, a.a.forEach(function(a, b) {
        var e = b.toLowerCase();
        b != e && (Ib(this, b), Ib(this, e), 0 < a.length && (this.d = null, v(this.a, Q(this, e), wa(a)), this.b += a.length))
    }, a));
    a.e = b
};
var S = function() {
    D.call(this);
    this.g = new ab(this);
    this.O = this;
    this.h = null
};
t(S, D);
S.prototype[Xa]=!0;
S.prototype.addEventListener = function(a, b, c, d) {
    ib(this, a, b, c, d)
};
S.prototype.removeEventListener = function(a, b, c, d) {
    pb(this, a, b, c, d)
};
var Mb = function(a, b) {
    var c, d = a.h;
    if (d)
        for (c = []; d; d = d.h)
            c.push(d);
    var d = a.O, e = b, f = e.type || e;
    if (q(e))
        e = new E(e, d);
    else if (e instanceof E)
        e.d = e.d || d;
    else {
        var g = e, e = new E(f, d);
        za(e, g)
    }
    var g=!0, h;
    if (c)
        for (var n = c.length - 1; 0 <= n; n--)
            h = e.a = c[n], g = Lb(h, f, !0, e) && g;
    h = e.a = d;
    g = Lb(h, f, !0, e) && g;
    g = Lb(h, f, !1, e) && g;
    if (c)
        for (n = 0; n < c.length; n++)
            h = e.a = c[n], g = Lb(h, f, !1, e) && g
};
S.prototype.d = function() {
    S.s.d.call(this);
    if (this.g) {
        var a = this.g, b = 0, c;
        for (c in a.a) {
            for (var d = a.a[c], e = 0; e < d.length; e++)
                ++b, $a(d[e]);
            delete a.a[c];
            a.d--
        }
    }
    this.h = null
};
S.prototype.e = function(a, b, c, d) {
    return cb(this.g, String(a), b, !1, c, d)
};
S.prototype.I = function(a, b, c, d) {
    var e;
    e = this.g;
    a = String(a).toString();
    if (a in e.a) {
        var f = e.a[a];
        b = bb(f, b, c, d);
        - 1 < b ? ($a(f[b]), u.splice.call(f, b, 1), 0 == f.length && (delete e.a[a], e.d--), e=!0) : e=!1
    } else 
        e=!1;
    return e
};
var Lb = function(a, b, c, d) {
    b = a.g.a[String(b)];
    if (!b)
        return !0;
    b = b.concat();
    for (var e=!0, f = 0; f < b.length; ++f) {
        var g = b[f];
        if (g&&!g.o && g.A == c) {
            var h = g.k, n = g.C || g.src;
            g.B && db(a.g, g);
            e=!1 !== h.call(n, d) && e
        }
    }
    return e && 0 != d.J
};
var Nb = function(a) {
    a = String(a);
    if (/^\s*$/.test(a) ? 0 : /^[\],:{}\s\u2028\u2029]*$/.test(a.replace(/\\["\\\/bfnrtu]/g, "@").replace(/"[^"\\\n\r\u2028\u2029\x00-\x08\x0a-\x1f]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, "]").replace(/(?:^|:|,)(?:[\s\u2028\u2029]*\[)+/g, "")))
        try {
            return eval("(" + a + ")")
    } catch (b) {}
    throw Error("Invalid JSON string: " + a);
}, Ob = function() {}, Qb = function(a, b, c) {
    switch (typeof b) {
    case "string":
        Pb(b, c);
        break;
    case "number":
        c.push(isFinite(b)&&!isNaN(b) ? b : "null");
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
        if (p(b)) {
            var d = b.length;
            c.push("[");
            for (var e = "", f = 0; f < d; f++)
                c.push(e), Qb(a, b[f], c), e = ",";
            c.push("]");
            break
        }
        c.push("{");
        d = "";
        for (e in b)
            Object.prototype.hasOwnProperty.call(b, e) && (f = b[e], "function" != typeof f && (c.push(d), Pb(e, c), c.push(":"), Qb(a, f, c), d = ","));
        c.push("}");
        break;
    case "function":
        break;
    default:
        throw Error("Unknown type: " + typeof b);
    }
}, Rb = {
    '"': '\\"',
    "\\": "\\\\",
    "/": "\\/",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\x0B": "\\u000b"
}, Sb = /\uffff/.test("\uffff") ? /[\\\"\x00-\x1f\x7f-\uffff]/g: /[\\\"\x00-\x1f\x7f-\xff]/g, Pb = function(a, b) {
    b.push('"', a.replace(Sb, function(a) {
        if (a in Rb)
            return Rb[a];
        var b = a.charCodeAt(0), e = "\\u";
        16 > b ? e += "000" : 256 > b ? e += "00" : 4096 > b && (e += "0");
        return Rb[a] = e + b.toString(16)
    }), '"')
};
var T = function(a) {
    S.call(this);
    this.i = a || "goog_" + qa++;
    this.f = []
};
t(T, S);
T.prototype.N=!1;
var Ub = function(a) {
    for (a.N=!0; 0 != a.f.length;) {
        var b = a.f.shift();
        Tb(a, b.name, b.type, b.data)
    }
};
T.prototype.send = function(a, b, c) {
    this.N ? Tb(this, a, b, c) : this.f.push({
        name: a,
        type: b,
        data: c
    })
};
var U = function(a, b, c, d, e) {
    E.call(this, a);
    this.r = b;
    this.H = c;
    this.M = d;
    this.L = e
};
t(U, E);
U.prototype.toString = function() {
    return ""
};
var Vb = function(a, b) {
    if (!r(a))
        if (a && "function" == typeof a.handleEvent)
            a = s(a.handleEvent, a);
        else 
            throw Error("Invalid listener argument");
    return 2147483647 < b?-1 : l.setTimeout(a, b || 0)
};
var ta = ["://secure-...imrworldwide.com/", "://cdn.imrworldwide.com/", "://aksecure.imrworldwide.com/", "www.google.com/pagead/sul", "www.youtube.com/gen_204\\?a=sul"], Wb = /\bocr\b/, Xb = 0, Yb = {}, Zb = function(a) {
    return /^[\s\xa0]*$/.test(null == a ? "" : String(a))?!1 : Gb(new K(a)).match(Wb)?!0 : null != ua(function(b) {
        return null != a.match(b)
    })
}, $b = function(a) {
    if (a) {
        var b = Ra("iframe", {
            src: 'javascript:"data:text/html,<body><img src=\\"' + a + '\\"></body>"',
            style: "display:none"
        });
        a = (9 == b.nodeType ? b : b.ownerDocument || b.document).body;
        var c, d = Vb(function() {
            I(c);
            b && b.parentNode && b.parentNode.removeChild(b)
        }, 15E3);
        c = ob(b, ["load", "error"], function() {
            Vb(function() {
                l.clearTimeout(d);
                b && b.parentNode && b.parentNode.removeChild(b)
            }, 5E3)
        });
        a.appendChild(b)
    }
}, ac = function(a) {
    if (a) {
        var b = new Image, c = "" + Xb++;
        Yb[c] = b;
        b.onload = b.onerror = function() {
            delete Yb[c]
        };
        b.src = a
    }
};
var bc = "ad.doubleclick.net bid.g.doubleclick.net corp.google.com ggpht.com google.co.uk google.com googleads.g.doubleclick.net googleads4.g.doubleclick.net googleadservices.com googlesyndication.com googleusercontent.com gstatic.com prod.google.com pubads.g.doubleclick.net s0.2mdn.net static.doubleclick.net static.doubleclick.net surveys.g.doubleclick.net youtube.com ytimg.com".split(" ");
var cc = function(a) {
    var b;
    if (b = "https:" == window.location.protocol)
        b = (new RegExp("^https?://([a-z0-9-]{1,63}\\.)*(" + bc.join("|").replace(/\./g, ".") + ")(:[0-9]+)?([/?#]|$)", "i")).test(a);
    b && (a = new K(a), xb(a, "https"), a = a.toString());
    a && (Zb(a) ? $b(a) : ac(a))
};
var V = function() {
    this.d = .05 > Math.random();
    this.a = Math.floor(4503599627370496 * Math.random())
};
V.getInstance = function() {
    return V.a ? V.a : V.a = new V
};
var fc = function() {
    var a = V.getInstance();
    if (a.d) {
        var b = {
            lid: 11
        }, b = dc(a, b), c = new K("http://pagead2.googlesyndication.com/pagead/gen_204");
        xa(b, function(a, b) {
            var f = c.a, g = b, h = null != a ? "boolean" == typeof a ? a ? "t": "f": "" + a: "";
            R(f);
            f.d = null;
            g = Q(f, g);
            Jb(f, g) && (f.b -= y(f.a, g).length);
            v(f.a, g, [h]);
            f.b++
        }, a);
        a = ec();
        xb(c, a.j);
        cc(c.toString())
    }
}, dc = function(a, b) {
    b.id = "";
    var c = ec();
    b.c = a.a;
    b.domain = c.p;
    return b
}, ec = function() {
    var a = window, b = document;
    return new K(a.parent == a ? a.location.href : b.referrer)
};
var W = function(a, b) {
    T.call(this, b);
    this.n = a;
    this.b = null;
    this.m = new J(this);
    this.m.e(window, "message", this.P)
};
t(W, T);
var gc = function(a) {
    if (null == a ||!q(a) || 0 != a.lastIndexOf("ima://", 0))
        return null;
    a = a.substr(6);
    try {
        return Nb(a)
    } catch (b) {
        return null
    }
}, Tb = function(a, b, c, d) {
    null != a.b && null != a.b.postMessage && a.b.postMessage(hc(a, b, c, d), "*");
    null != a.b && null == a.b.postMessage && fc()
};
W.prototype.d = function() {
    this.m.w();
    W.s.d.call(this)
};
W.prototype.P = function(a) {
    a = a.b;
    var b = gc(a.data);
    if (null != b) {
        if (null == this.b)
            this.b = a.source;
        else if (this.b != a.source)
            return;
        var c = b.channel;
        null != c && c == this.n && (c = b.sid, null != c && ("*" != this.i && c != this.i || Mb(this, new U(b.name, b.type, b.data || {}, b.sid, a.origin))))
    }
};
var hc = function(a, b, c, d) {
    var e = {};
    e.name = b;
    e.type = c;
    null != d && (e.data = d);
    e.sid = a.i;
    e.channel = a.n;
    a = [];
    Qb(new Ob, e, a);
    return "ima://" + a.join("")
};
var X = function(a, b) {
    S.call(this);
    this.n = a;
    this.i = b;
    this.b = {};
    this.f = new J(this);
    this.f.e(window, "message", this.m)
};
t(X, S);
X.prototype.send = function(a) {
    var b = a.b;
    this.b.hasOwnProperty(b) && this.b[b].send(a.type, a.r, a.H)
};
var jc = function(a, b, c, d) {
    a.b.hasOwnProperty(b) || (c = new W(b, c), a.f.e(c, a.n, function(a) {
        Mb(this, new ic(a.type, a.r, a.H, a.M, a.L, b))
    }), c.b = d, Ub(c), a.b[b] = c)
};
X.prototype.d = function() {
    this.f.w();
    for (var a in this.b)
        Ta(this.b[a]);
    X.s.d.call(this)
};
X.prototype.m = function(a) {
    a = a.b;
    var b = gc(a.data);
    if (null != b) {
        var c = b.channel;
        if (this.i&&!this.b.hasOwnProperty(c)) {
            var d = b.sid;
            jc(this, c, d, a.source);
            Mb(this, new ic(b.name, b.type, b.data || {}, d, a.origin, c))
        }
    }
};
var ic = function(a, b, c, d, e, f) {
    U.call(this, a, b, c, d, e);
    this.b = f
};
t(ic, U);
var lc = function() {
    var a = m("google.ima.gptProxyInstance", window);
    if (null != a)
        return a;
    J.call(this);
    this.h = new X("gpt", !0);
    a = fa(Ta, this.h);
    this.a || (this.a = []);
    this.a.push(a);
    this.e(this.h, "gpt", this.m);
    this.b = null;
    kc() || window.top === window || (this.b = new X("gpt", !1), a = fa(Ta, this.b), this.a || (this.a = []), this.a.push(a), this.e(this.b, "gpt", this.n))
};
t(lc, J);
var kc = function() {
    return !!m("googletag.cmd", window)
}, mc = function() {
    var a = m("googletag.console", window);
    return null != a ? a : null
};
lc.prototype.m = function(a) {
    var b = a.L, c = wb("//imasdk.googleapis.com"), b = wb(b);
    if (c[3] == b[3] && c[4] == b[4])
        if (null != this.b)
            jc(this.b, a.b, a.M, window.parent), null != this.b && this.b.send(a);
        else if (c = a.H, null != c && void 0 !== c.scope) {
            var b = c.scope, c = c.args, d;
            if ("proxy" == b)
                c = a.r, "isGptPresent" == c ? d = kc() : "isConsolePresent" == c && (d = null != mc());
            else if (kc())
                if ("pubads" == b || "companionAds" == b) {
                    d = a.r;
                    var e, f = window.googletag;
                    if (null != f && null != f[b] && (f = f[b](), null != f && (d = f[d], null != d)))
                        try {
                            e = d.apply(f, c)
                        } catch (g) {}
                        d =
                        e
                    } else if ("console" == b) {
                        if (f = a.r, e = mc(), null != e && (f = e[f], null != f))
                            try {
                                f.apply(e, c)
                            } catch (h) {}
                        } else if (null === b) {
                            e = a.r;
                            d = window;
                            if (0 <= ra(["googleGetCompanionAdSlots", "googleSetCompanionAdContents"], e) && (e = d[e], null != e))
                                try {
                                    f = e.apply(d, c)
                                } catch (n) {}
                                d = f
                            }
                            void 0 !== d && (a.H.returnValue = d, this.h.send(a))
                    }
            };
lc.prototype.n = function(a) {
    this.h.send(a)
};
var nc = window, oc = m("google.ima.gptProxyInstance", nc);
if (null == oc) {
    var pc = oc = new lc, Y = ["google", "ima", "gptProxyInstance"], Z = nc || l;
    Y[0]in Z ||!Z.execScript || Z.execScript("var " + Y[0]);
    for (var $; Y.length && ($ = Y.shift());)
        Y.length || void 0 === pc ? Z[$] ? Z = Z[$] : Z = Z[$] = {} : Z[$] = pc
};
})();

