var _yt_www = {};
(function(m) {
    var window = this;
    var pa, qa;
    m.aa = function(a) {
        return function() {
            return m.ba[a].apply(this, arguments)
        }
    };
    m.ca = function(a) {
        return void 0 !== a
    };
    m.l = function(a, b, c) {
        a = a.split(".");
        c = c || m.da;
        a[0]in c ||!c.execScript || c.execScript("var " + a[0]);
        for (var d; a.length && (d = a.shift());)
            !a.length && m.ca(b) ? c[d] = b : c[d] ? c = c[d] : c = c[d] = {}
    };
    m.n = function(a, b) {
        for (var c = a.split("."), d = b || m.da, e; e = c.shift();)
            if (null != d[e])
                d = d[e];
            else 
                return null;
        return d
    };
    m.ea = function() {};
    m.fa = function(a) {
        a.getInstance = function() {
            return a.hb ? a.hb : a.hb = new a
        }
    };
    m.ga = function(a) {
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
};
m.ha = function(a) {
    return "array" == m.ga(a)
};
m.ia = function(a) {
    var b = m.ga(a);
    return "array" == b || "object" == b && "number" == typeof a.length
};
m.ja = function(a) {
    return "string" == typeof a
};
m.ka = function(a) {
    return "number" == typeof a
};
m.ma = function(a) {
    return "function" == m.ga(a)
};
m.na = function(a) {
    var b = typeof a;
    return "object" == b && null != a || "function" == b
};
m.oa = function(a) {
    return a[pa] || (a[pa]=++qa)
};
var ra = function(a, b, c) {
    return a.call.apply(a.bind, arguments)
};
var sa = function(a, b, c) {
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
};
m.p = function(a, b, c) {
    m.p = Function.prototype.bind&&-1 != Function.prototype.bind.toString().indexOf("native code") ? ra : sa;
    return m.p.apply(null, arguments)
};
m.q = function(a, b) {
    var c = Array.prototype.slice.call(arguments, 1);
    return function() {
        var b = c.slice();
        b.push.apply(b, arguments);
        return a.apply(this, b)
    }
};
m.s = function(a, b) {
    function c() {}
    c.prototype = b.prototype;
    a.K = b.prototype;
    a.prototype = new c;
    a.prototype.constructor = a;
    a.base = function(a, c, f) {
        return b.prototype[c].apply(a, Array.prototype.slice.call(arguments, 2))
    }
};
m.ta = function(a) {
    m.ta[" "](a);
    return a
};
m.ua = function(a) {
    if (Error.captureStackTrace)
        Error.captureStackTrace(this, m.ua);
    else {
        var b = Error().stack;
        b && (this.stack = b)
    }
    a && (this.message = String(a))
};
m.va = function(a) {
    return (0, window.encodeURIComponent)(String(a))
};
m.wa = function(a) {
    if (!xa.test(a))
        return a;
    - 1 != a.indexOf("&") && (a = a.replace(ya, "&amp;"));
    - 1 != a.indexOf("<") && (a = a.replace(za, "&lt;"));
    - 1 != a.indexOf(">") && (a = a.replace(Aa, "&gt;"));
    - 1 != a.indexOf('"') && (a = a.replace(Ba, "&quot;"));
    - 1 != a.indexOf("'") && (a = a.replace(Ca, "&#39;"));
    - 1 != a.indexOf("\x00") && (a = a.replace(Da, "&#0;"));
    return a
};
m.Ea = function(a, b) {
    for (var c = 0, d = (0, m.Fa)(String(a)).split("."), e = (0, m.Fa)(String(b)).split("."), f = Math.max(d.length, e.length), h = 0; 0 == c && h < f; h++) {
        var k = d[h] || "", r = e[h] || "", w = RegExp("(\\d*)(\\D*)", "g"), B = RegExp("(\\d*)(\\D*)", "g");
        do {
            var Q = w.exec(k) || ["", "", ""], Y = B.exec(r) || ["", "", ""];
            if (0 == Q[0].length && 0 == Y[0].length)
                break;
            c = Ga(0 == Q[1].length ? 0 : (0, window.parseInt)(Q[1], 10), 0 == Y[1].length ? 0 : (0, window.parseInt)(Y[1], 10)) || Ga(0 == Q[2].length, 0 == Y[2].length) || Ga(Q[2], Y[2])
        }
        while (0 == c)
        }
    return c
};
var Ga = function(a, b) {
    return a < b?-1 : a > b ? 1 : 0
};
m.Ha = function(a) {
    return String(a).replace(/\-([a-z])/g, function(a, c) {
        return c.toUpperCase()
    })
};
m.Ia = function(a) {
    var b = m.ja(void 0) ? "undefined".replace(/([-()\[\]{}+?*.$\^|,:#<!\\])/g, "\\$1").replace(/\x08/g, "\\x08"): "\\s";
    return a.replace(new RegExp("(^" + (b ? "|[" + b + "]+" : "") + ")([a-z])", "g"), function(a, b, e) {
        return b + e.toUpperCase()
    })
};
m.Ja = function(a, b, c) {
    b = m.Ka(a, b, c);
    return 0 > b ? null : m.ja(a) ? a.charAt(b) : a[b]
};
m.Ka = function(a, b, c) {
    for (var d = a.length, e = m.ja(a) ? a.split("") : a, f = 0; f < d; f++)
        if (f in e && b.call(c, e[f], f, a))
            return f;
    return - 1
};
m.La = function(a, b) {
    return 0 <= (0, m.Na)(a, b)
};
m.Oa = function(a, b) {
    var c = (0, m.Na)(a, b), d;
    (d = 0 <= c) && m.Pa(a, c);
    return d
};
m.Pa = function(a, b) {
    m.Qa.splice.call(a, b, 1)
};
m.Ra = function(a) {
    var b = a.length;
    if (0 < b) {
        for (var c = Array(b), d = 0; d < b; d++)
            c[d] = a[d];
        return c
    }
    return []
};
m.Sa = function(a, b, c) {
    return 2 >= arguments.length ? m.Qa.slice.call(a, b) : m.Qa.slice.call(a, b, c)
};
m.t = function() {
    this.lb = this.lb;
    this.xa = this.xa
};
m.Ta = function(a, b) {
    m.Ua(a, m.q(m.Va, b))
};
m.Ua = function(a, b) {
    a.xa || (a.xa = []);
    a.xa.push(m.ca(void 0) ? (0, m.p)(b, void 0) : b)
};
m.Va = function(a) {
    a && "function" == typeof a.dispose && a.dispose()
};
m.Wa = function() {
    m.t.call(this);
    this.ya = [];
    this.Ob = {}
};
m.Xa = function(a) {
    Ya(m.Za, arguments)
};
m.u = function(a, b) {
    return a in m.Za ? m.Za[a] : b
};
m.v = function(a, b) {
    m.ma(a) && (a = m.$a(a));
    return window.setTimeout(a, b)
};
m.ab = function(a, b) {
    m.ma(a) && (a = m.$a(a));
    return window.setInterval(a, b)
};
m.bb = function(a) {
    window.clearTimeout(a)
};
m.cb = function(a) {
    window.clearInterval(a)
};
m.$a = function(a) {
    return a && window.yterr ? function() {
        try {
            return a.apply(this, arguments)
        } catch (b) {
            throw m.db(b), b;
        }
    } : a
};
m.db = function(a, b) {
    var c = m.n("yt.www.errors.log");
    c ? c(a, b) : (c = m.u("ERRORS") || [], c.push([a, b]), m.Xa("ERRORS", c))
};
m.eb = function(a) {
    Ya(m.fb, arguments)
};
m.gb = function(a) {
    return a in m.fb
};
m.hb = function(a, b, c) {
    var d = b || {};
    if (a = a in m.fb ? m.fb[a] : c)
        for (var e in d)
            a = a.replace(new RegExp("\\$" + e, "gi"), function() {
                return d[e]
            });
    return a
};
var Ya = function(a, b) {
    if (1 < b.length) {
        var c = b[0];
        a[c] = b[1]
    } else {
        var d = b[0];
        for (c in d)
            a[c] = d[c]
    }
};
m.x = function(a, b, c) {
    var d = ib();
    if (d) {
        var e = d.subscribe(a, function() {
            if (!m.jb || m.jb != e) {
                var d = arguments, h = function() {
                    kb[e] && b.apply(c || window, d)
                };
                try {
                    lb[a] ? h() : m.v(h, 0)
                } catch (k) {
                    m.db(k)
                }
            }
        }, c);
        kb[e]=!0;
        mb[a] || (mb[a] = []);
        mb[a].push(e);
        return e
    }
    return 0
};
m.nb = function(a) {
    var b = ib();
    b && ("number" == typeof a ? a = [a] : "string" == typeof a && (a = [(0, window.parseInt)(a, 10)]), (0, m.y)(a, function(a) {
        b.unsubscribeByKey(a);
        delete kb[a]
    }))
};
m.z = function(a, b) {
    var c = ib();
    return c ? c.publish.apply(c, arguments) : !1
};
m.ob = function(a, b) {
    lb[a]=!0;
    var c = ib(), c = c ? c.publish.apply(c, arguments): !1;
    lb[a]=!1;
    return c
};
var ib = function() {
    return m.n("yt.pubsub.instance_")
};
m.pb = function(a) {
    if (a.classList)
        return a.classList;
    a = a.className;
    return m.ja(a) && a.match(/\S+/g) || []
};
m.A = function(a, b) {
    return a.classList ? a.classList.contains(b) : m.La(m.pb(a), b)
};
m.C = function(a, b) {
    a.classList ? a.classList.add(b) : m.A(a, b) || (a.className += 0 < a.className.length ? " " + b : b)
};
m.D = function(a, b) {
    a.classList ? a.classList.remove(b) : m.A(a, b) && (a.className = (0, m.qb)(m.pb(a), function(a) {
        return a != b
    }).join(" "))
};
m.rb = function(a, b, c) {
    c ? m.C(a, b) : m.D(a, b)
};
m.sb = function(a, b, c) {
    return Math.min(Math.max(a, b), c)
};
m.tb = function(a, b, c) {
    for (var d in a)
        b.call(c, a[d], d, a)
};
m.ub = function(a) {
    var b = [], c = 0, d;
    for (d in a)
        b[c++] = d;
    return b
};
m.vb = function(a, b, c) {
    for (var d in a)
        if (b.call(c, a[d], d, a))
            return d
};
m.wb = function(a, b) {
    b in a && delete a[b]
};
m.xb = function(a, b, c) {
    if (b in a)
        throw Error('The object already contains the key "' + b + '"');
    a[b] = c
};
m.yb = function(a) {
    var b = {}, c;
    for (c in a)
        b[c] = a[c];
    return b
};
m.zb = function(a, b) {
    for (var c, d, e = 1; e < arguments.length; e++) {
        d = arguments[e];
        for (c in d)
            a[c] = d[c];
        for (var f = 0; f < Ab.length; f++)
            c = Ab[f], Object.prototype.hasOwnProperty.call(d, c) && (a[c] = d[c])
    }
};
m.Bb = function(a) {
    var b = arguments.length;
    if (1 == b && m.ha(arguments[0]))
        return m.Bb.apply(null, arguments[0]);
    for (var c = {}, d = 0; d < b; d++)
        c[arguments[d]]=!0;
    return c
};
var Cb = function(a) {
    return - 1 != m.Db.indexOf(a)
};
var Eb = function() {
    var a = m.da.document;
    return a ? a.documentMode : void 0
};
m.Fb = function(a) {
    return Gb[a] || (Gb[a] = 0 <= m.Ea(Hb, a))
};
m.Ib = function(a) {
    return m.E && m.Jb >= a
};
m.F = function(a) {
    return m.ja(a) ? window.document.getElementById(a) : a
};
m.Kb = function(a) {
    return m.Lb(window.document, a)
};
m.Lb = function(a, b) {
    return m.ja(b) ? a.getElementById(b) : b
};
m.G = function(a, b) {
    var c = b || window.document, d = null;
    c.querySelectorAll && c.querySelector ? d = c.querySelector("." + a) : d = m.Mb("*", a, b)[0];
    return d || null
};
m.Mb = function(a, b, c) {
    var d = window.document;
    c = c || d;
    a = a && "*" != a ? a.toUpperCase() : "";
    if (c.querySelectorAll && c.querySelector && (a || b))
        return c.querySelectorAll(a + (b ? "." + b : ""));
    if (b && c.getElementsByClassName) {
        c = c.getElementsByClassName(b);
        if (a) {
            for (var d = {}, e = 0, f = 0, h; h = c[f]; f++)
                a == h.nodeName && (d[e++] = h);
            d.length = e;
            return d
        }
        return c
    }
    c = c.getElementsByTagName(a || "*");
    if (b) {
        d = {};
        for (f = e = 0; h = c[f]; f++)
            a = h.className, "function" == typeof a.split && m.La(a.split(/\s+/), b) && (d[e++] = h);
        d.length = e;
        return d
    }
    return c
};
m.Nb = function(a, b) {
    m.tb(b, function(b, d) {
        "style" == d ? a.style.cssText = b : "class" == d ? a.className = b : "for" == d ? a.htmlFor = b : d in Ob ? a.setAttribute(Ob[d], b) : 0 == d.lastIndexOf("aria-", 0) || 0 == d.lastIndexOf("data-", 0) ? a.setAttribute(d, b) : a[d] = b
    })
};
m.Pb = function(a, b, c) {
    return m.Qb(window.document, arguments)
};
m.Qb = function(a, b) {
    var c = b[0], d = b[1];
    if (!Rb && d && (d.name || d.type)) {
        c = ["<", c];
        d.name && c.push(' name="', m.wa(d.name), '"');
        if (d.type) {
            c.push(' type="', m.wa(d.type), '"');
            var e = {};
            m.zb(e, d);
            delete e.type;
            d = e
        }
        c.push(">");
        c = c.join("")
    }
    c = a.createElement(c);
    d && (m.ja(d) ? c.className = d : m.ha(d) ? c.className = d.join(" ") : m.Nb(c, d));
    2 < b.length && m.Sb(a, c, b, 2);
    return c
};
m.Sb = function(a, b, c, d) {
    function e(c) {
        c && b.appendChild(m.ja(c) ? a.createTextNode(c) : c)
    }
    for (; d < c.length; d++) {
        var f = c[d];
        !m.ia(f) || m.na(f) && 0 < f.nodeType ? e(f) : (0, m.y)(Tb(f) ? m.Ra(f) : f, e)
    }
};
m.Ub = function(a) {
    for (var b; b = a.firstChild;)
        a.removeChild(b)
};
var Wb = function(a) {
    if (!a)
        return null;
    if (a.firstChild)
        return a.firstChild;
    for (; a&&!a.nextSibling;)
        a = a.parentNode;
    return a ? a.nextSibling : null
};
var Xb = function(a) {
    if (!a)
        return null;
    if (!a.previousSibling)
        return a.parentNode;
    for (a = a.previousSibling; a && a.lastChild;)
        a = a.lastChild;
    return a
};
m.Yb = function(a) {
    return 9 == a.nodeType ? a : a.ownerDocument || a.document
};
var Tb = function(a) {
    if (a && "number" == typeof a.length) {
        if (m.na(a))
            return "function" == typeof a.item || "string" == typeof a.item;
        if (m.ma(a))
            return "function" == typeof a.item
    }
    return !1
};
m.Zb = function(a, b, c, d) {
    c || (a = a.parentNode);
    c = null == d;
    for (var e = 0; a && (c || e <= d);) {
        if (b(a))
            return a;
        a = a.parentNode;
        e++
    }
    return null
};
m.$b = function(a) {
    return function() {
        return a
    }
};
m.ac = function(a) {
    var b=!1, c;
    return function() {
        b || (c = a(), b=!0);
        return c
    }
};
m.bc = function() {};
m.cc = function(a) {
    if (a instanceof m.bc)
        return a;
    if ("function" == typeof a.fb)
        return a.fb(!1);
    if (m.ia(a)) {
        var b = 0, c = new m.bc;
        c.next = function() {
            for (; ;) {
                if (b >= a.length)
                    throw m.dc;
                if (b in a)
                    return a[b++];
                b++
            }
        };
        return c
    }
    throw Error("Not implemented");
};
m.ec = function(a, b, c) {
    if (m.ia(a))
        try {
            (0, m.y)(a, b, c)
    } catch (d) {
        if (d !== m.dc)
            throw d;
    } else {
        a = m.cc(a);
        try {
            for (; ;)
                b.call(c, a.next(), void 0, a)
            } catch (e) {
            if (e !== m.dc)
                throw e;
        }
    }
};
m.fc = function(a) {
    if (m.ia(a))
        return m.Ra(a);
    a = m.cc(a);
    var b = [];
    m.ec(a, function(a) {
        b.push(a)
    });
    return b
};
m.gc = function(a, b, c, d, e, f, h) {
    var k = "";
    a && (k += a + ":");
    c && (k += "//", b && (k += b + "@"), k += c, d && (k += ":" + d));
    e && (k += e);
    f && (k += "?" + f);
    h && (k += "#" + h);
    return k
};
m.hc = function(a) {
    if (ic) {
        ic=!1;
        var b = m.da.location;
        if (b) {
            var c = b.href;
            if (c && (c = m.jc(c)) && c != b.hostname)
                throw ic=!0, Error();
        }
    }
    return a.match(kc)
};
m.lc = function(a) {
    return a ? (0, window.decodeURI)(a) : a
};
m.jc = function(a) {
    return m.lc(m.hc(a)[3] || null)
};
m.mc = function(a) {
    a = m.hc(a);
    return m.gc(null, null, null, null, a[5], a[6], a[7])
};
m.nc = function(a) {
    if (a[1]) {
        var b = a[0], c = b.indexOf("#");
        0 <= c && (a.push(b.substr(c)), a[0] = b = b.substr(0, c));
        c = b.indexOf("?");
        0 > c ? a[1] = "?" : c == b.length - 1 && (a[1] = void 0)
    }
    return a.join("")
};
m.oc = function(a, b, c) {
    if (m.ha(b))
        for (var d = 0; d < b.length; d++)
            m.oc(a, String(b[d]), c);
    else 
        null != b && c.push("&", a, "" === b ? "" : "=", m.va(b))
};
var pc = function(a, b) {
    for (var c in b)
        m.oc(c, b[c], a);
    return a
};
m.qc = function(a) {
    a = pc([], a);
    a[0] = "";
    return a.join("")
};
m.rc = function(a, b) {
    return m.nc(pc([a], b))
};
m.sc = function(a, b, c) {
    a && (c ? a && (a = m.Pb("iframe", {
        src: 'javascript:"data:text/html,<body><img src=\\"' + a + '\\"></body>"',
        style: "display:none"
    }), m.Yb(a).body.appendChild(a)) : tc(a, b))
};
var tc = function(a, b) {
    var c = new window.Image, d = "" + uc++;
    vc[d] = c;
    c.onload = c.onerror = function() {
        b && vc[d] && b();
        delete vc[d]
    };
    c.src = a;
    c = eval("null")
};
m.wc = function(a, b, c) {
    xc(c)[a] = b || (0, m.H)();
    b || yc(a)
};
var zc = function(a, b) {
    Ac(void 0).yt_sts = a;
    m.wc("_start", b, void 0)
};
var Bc = function(a, b, c) {
    var d = Cc(c).span;
    c = xc(c);
    "number" == typeof b ? d[a] = b : b in c && (d[a] = (0, m.H)() - c[b])
};
var Dc = function(a) {
    var b = xc(a), c = Cc(a).span, d = Ac(a), e = m.n("yt.timing.reportbuilder_");
    if (e) {
        if (e = e(b, c, d, a))
            Ec(e), Fc(a), Gc();
        return window.NaN
    }
    e = {
        v: 2,
        s: "youtube",
        action: m.u("TIMING_ACTION")
    };
    if (Hc.now && Hc.timing) {
        var f = Hc.timing.navigationStart + Hc.now(), f = Math.round((0, m.H)() - f);
        d.yt_hrd = f
    }
    var f = m.u("TIMING_INFO") || {}, h;
    for (h in f)
        d[h] = f[h];
    h = d.srt;
    delete d.srt;
    var k;
    h || 0 === h || (k = Hc.timing || {}, h = Math.max(0, k.responseStart - k.navigationStart), (0, window.isNaN)(h) && d.pt && (h = d.pt));
    if (h || 0 === h)
        d.srt = h;
    d.h5jse && (f = window.location.protocol + m.n("ytplayer.config.assets.js"), (f = Hc.getEntriesByName ? Hc.getEntriesByName(f)[0] : null) ? d.h5jse = Math.round(d.h5jse - f.responseEnd) : delete d.h5jse);
    b.aft || (b.aft = b.pbr && b.pbs > b.pbr ? b.pbr : b.pbs ? b.pbs : b.vr ? b.vr : b.ol);
    f = b._start;
    if (!Ic) {
        k || (k = Hc.timing || {});
        var r;
        t:
        if (r = k, r.msFirstPaint)
            r = Math.max(0, r.msFirstPaint);
        else {
            var w = window.chrome;
            if (w && (w = w.loadTimes, m.ma(w))) {
                var w = w(), B = 1E3 * Math.min(w.requestTime || window.Infinity, w.startLoadTime || window.Infinity), B = window.Infinity ===
                B ? 0: r.navigationStart - B;
                r = Math.max(0, Math.round(1E3 * w.firstPaintTime + B) || 0);
                break t
            }
            r = 0
        }
        0 < r && r > f && (b.fpt = r);
        r = k.redirectEnd - k.redirectStart;
        0 < r && (c.rtime_ = r);
        r = k.domainLookupEnd - k.domainLookupStart;
        0 < r && (c.dns_ = r);
        r = k.connectEnd - k.connectStart;
        0 < r && (c.tcp_ = r);
        r = k.connectEnd - k.secureConnectionStart;
        k.secureConnectionStart && 0 < r && (c.stcp_ = r);
        r = k.responseStart - k.requestStart;
        0 < r && (c.req_ = r);
        r = k.responseEnd - k.responseStart;
        0 < r && (c.rcv_ = r);
        Ic=!0
    }
    for (var Q in d)
        "_" != Q.charAt(0) && (e[Q] = d[Q]);
    d = {};
    Q = [];
    for (var Y in b)
        "_" !=
        Y.charAt(0) && (r = Math.max(Math.round(b[Y] - f), 0), d[Y] = r, Q.push(Y + "." + r));
    e.rt = Q.join(",");
    b = {};
    Y = [];
    for (var la in c)
        "_" != la.charAt(0) && (b[la] = c[la], Y.push(la + "." + c[la]));
    e.it = Y.join(",");
    (c = m.n("ytdebug.logTiming")) && c(e, d, b);
    Fc(a);
    Gc();
    m.u("EXP_DEFER_CSI_PING") ? (Jc(), Kc = e, Lc = m.v(Jc, 0)) : Ec(e);
    return d.aft + (h || 0)
};
var Mc = function(a, b) {
    var c = m.u("TIMING_ACTION"), d = xc(b);
    if (c && d._start)
        if (c = m.u("TIMING_WAIT", []), !a && c.length) {
            for (var e=!0, f = 0, h = c.length; f < h; ++f)
                if (!(c[f]in d)) {
                    e=!1;
                    break
                }
                if (e)
                    return Dc(b)
        } else if (d.ol || d.aft)
            return Dc(b);
    return window.NaN
};
var Nc = function(a) {
    if (m.u("TIMING_WAIT"))
        a = Mc(!1, a);
    else 
        t: {
        var b = m.u("TIMING_ACTION"), c=!!m.u("TIMING_WFF"), d = xc(a);
        if (b && d._start) {
            if (c && d.vr ||!c && (d.ol || d.aft)) {
                a = Dc(a);
                break t
            }
            a && Dc(a)
        }
        a = window.NaN
    }
    return a
};
var Ec = function(a) {
    m.u("EXP_DEFER_CSI_PING") && (m.bb(Lc), Kc = null);
    var b = "https:" == window.location.protocol ? "https://gg.google.com/csi": "http://csi.gstatic.com/csi", c = "", d;
    for (d in a)
        c += "&" + d + "=" + a[d];
    m.sc(b + "?" + c.substring(1))
};
var Jc = function(a) {
    Kc && (a && (Kc.yt_fss = a), Ec(Kc))
};
var xc = function(a) {
    return Cc(a).tick
};
var Ac = function(a) {
    return Cc(a).info
};
var Cc = function(a) {
    return m.n("ytcsi." + (a || "") + "data_") || Fc(a)
};
var Fc = function(a) {
    var b = {
        tick: {},
        span: {},
        info: {}
    };
    m.l("ytcsi." + (a || "") + "data_", b, void 0);
    return b
};
m.Oc = function() {
    return !!m.n("yt.scheduler.instance")
};
m.Pc = function() {
    return m.Oc() && m.A(window.document.body, "exp-js-scheduler-more")
};
m.Qc = function(a, b) {
    void 0 === b && (b = window.NaN);
    var c = m.n("yt.scheduler.instance.addJob");
    return c ? ((0, window.isNaN)(b) && (b = 0), c(a, 0, b)) : (0, window.isNaN)(b) ? (a(), window.NaN) : m.v(a, b || 0)
};
m.Rc = function(a, b) {
    return m.Qc(a, b)
};
m.Sc = function(a) {
    var b = m.n("yt.scheduler.instance.cancelJob");
    b ? b(a) : m.bb(a)
};
m.Tc = function(a) {
    for (var b = 0, c = a.length; b < c; b++)
        m.Sc(a[b])
};
m.Uc = function(a, b, c, d, e) {
    this.name = a;
    this.deps = b || [];
    this.page = c || "";
    this.D = d ? m.$a(d) : null;
    this.C = e ? m.$a(e) : null;
    this.ya = [];
    this.j = this.k = 0
};
m.Vc = function(a) {
    m.Pc() ? (m.Sc(a.k), a.k = m.Rc((0, m.p)(a.init, a))) : a.init()
};
m.Wc = function(a) {
    a.name in Xc && Yc(a.name);
    Xc[a.name] = {
        reqs: [],
        disable: (0, m.p)(a.disable, a)
    };
    (0, m.y)(a.deps, function(b) {
        if (!(b in Xc))
            throw Error("Module " + b + " required by " + a.name);
        Xc[b].reqs.push(a.name)
    });
    a.enable()
};
var Yc = function(a) {
    if (a in Xc) {
        var b = Xc[a];
        (0, m.y)(b.reqs, function(a) {
            Yc(a)
        });
        try {
            b.disable()
        } catch (c) {}
        delete Xc[a]
    }
};
m.Zc = function(a) {
    return a && "readyState"in a ? a.readyState : 0
};
var $c = function() {
    if (m.u("SCHEDULE_AJAX") && window.XMLHttpRequest&&!ad) {
        bd = m.u("SCHEDULER_BUFFER_LENGTH");
        var a = window.XMLHttpRequest;
        cd = a.prototype.abort;
        dd = a.prototype.open;
        ad = a.prototype.send;
        a.prototype.abort = ed;
        a.prototype.open = fd;
        a.prototype.send = gd
    }
};
var ed = function() {
    hd(!0);
    cd.call(this)
};
var fd = function() {
    this.url = arguments[1];
    dd.apply(this, arguments)
};
var gd = function() {
    var a = this, b = arguments;
    if (a.url && a.url.match(id))
        ad.apply(a, b);
    else {
        if ("onloadend"in a)
            a.addEventListener("loadend", function() {
                4 == m.Zc(a) && hd(!0)
            }, !1);
        else {
            var c = a.onreadystatechange;
            a.onreadystatechange = function() {
                4 == m.Zc(a) && hd(!0);
                c && c.apply(this, arguments)
            }
        }
        jd.push(function() {
            ad.apply(a, b)
        });
        hd()
    }
};
var hd = function(a) {
    a && (kd=!1, m.bb(ld));
    if (!kd && jd.length)
        if (bd&&!md())
            m.bb(nd), nd = m.v(hd, 100);
        else {
            kd=!0;
            a = jd.shift();
            try {
                a()
            } catch (b) {
                hd(!0)
            }
            ld = m.v(m.q(hd, !0), 5E3)
        }
};
var md = function() {
    var a = bd, b = m.n("yt.player.getPlayerByElement");
    if (!b)
        return !0;
    var b = b("player-api"), c = m.u("ADS_DATA");
    return b && (!c ||!c.show_instream) && b.getVideoLoadedFraction && b.getDuration && b.getCurrentTime && b.getPlayerState ? (c = b.getVideoLoadedFraction() * b.getDuration() - b.getCurrentTime(), 0 >= b.getPlayerState() || c > a || .3 < b.getVideoLoadedFraction()) : !0
};
m.I = function(a, b) {
    return a ? a.dataset ? a.dataset[m.od(b)] : a.getAttribute("data-" + b) : null
};
m.pd = function(a, b) {
    return a ? a.dataset ? m.od(b)in a.dataset : a.hasAttribute?!!a.hasAttribute("data-" + b) : !!a.getAttribute("data-" + b) : !1
};
m.od = function(a) {
    return qd[a] || (qd[a] = String(a).replace(/\-([a-z])/g, function(a, c) {
        return c.toUpperCase()
    }))
};
m.rd = function(a, b) {
    var c = m.sd(a);
    window.spf.script.load(a, c, b)
};
m.td = function(a, b) {
    var c = m.sd(a);
    window.spf.script.ignore(c, b)
};
m.ud = function(a) {
    a && (m.ja(a) ? m.rd(a) : m.ha(a) ? (0, m.y)(a, function(a) {
        m.ud(a)
    }) : m.tb(a, function(a, c) {
        m.rd(c, function() {
            m.ud(a)
        })
    }))
};
m.sd = function(a) {
    var b = "";
    if (a) {
        var c = a.indexOf("jsbin/"), d = a.lastIndexOf(".js"), e = c + 6;
        - 1 < c&&-1 < d && d > e && (b = a.substring(e, d), b = b.replace(vd, ""), b = b.replace(wd, ""), b = b.replace("debug-", ""), b = b.replace("tracing-", ""))
    }
    return b
};
var xd = function(a) {
    a = m.yd(a);
    return m.zd(a) || (null === a?!1 : "google" == a[1]?!0 : "google" == a[2] ? "au" == a[0] && "com" == a[1]?!0 : "uk" == a[0] && "co" == a[1]?!0 : !1 : !1)
};
m.zd = function(a) {
    return null === a?!1 : "com" == a[0] && a[1].match(/^youtube(?:-nocookie)?$/)?!0 : !1
};
m.yd = function(a) {
    a = (0, m.Ad)(a);
    return null === a ? null : a.split(".").reverse()
};
m.Bd = function(a, b) {
    m.sc("/gen_204?" + a, b)
};
var Cd = function(a) {
    a = a || {};
    this.url = a.url || "";
    this.urlV8 = a.url_v8 || "";
    this.urlV9As2 = a.url_v9as2 || "";
    this.args = a.args || m.yb(Dd);
    this.assets = a.assets || {};
    this.attrs = a.attrs || m.yb(Ed);
    this.params = a.params || m.yb(Fd);
    this.minVersion = a.min_version || "8.0.0";
    this.fallback = a.fallback || null;
    this.fallbackMessage = a.fallbackMessage || null;
    this.html5=!!a.html5;
    this.disable = a.disable || {};
    this.loaded=!!a.loaded;
    this.messages = a.messages || {}
};
m.Gd = function(a) {
    a instanceof Cd || (a = new Cd(a));
    return a
};
var Hd = function() {
    return null != m.Id
};
m.Jd = function() {
    return m.Id ? m.Id.invoke() : null
};
m.Kd = function(a) {
    var b = a.cloneNode(!1);
    "TR" == b.tagName || "SELECT" == b.tagName ? (0, m.y)(a.childNodes, function(a) {
        b.appendChild(m.Kd(a))
    }) : b.innerHTML = a.innerHTML;
    return b
};
m.Ld = function(a) {
    if (a = a || window.event) {
        for (var b in a)
            b in Md || (this[b] = a[b]);
        this.scale = a.scale;
        this.rotation = a.rotation;
        this.wa = a;
        (b = a.target || a.srcElement) && 3 == b.nodeType && (b = b.parentNode);
        this.target = b;
        if (b = a.relatedTarget)
            try {
                b = b.nodeName ? b : null
        } catch (c) {
            b = null
        } else 
            "mouseover" == this.type ? b = a.fromElement : "mouseout" == this.type && (b = a.toElement);
        this.relatedTarget = b;
        this.clientX = void 0 != a.clientX ? a.clientX : a.pageX;
        this.clientY = void 0 != a.clientY ? a.clientY : a.pageY;
        this.keyCode = a.keyCode ? a.keyCode :
        a.which;
        this.charCode = a.charCode || ("keypress" == this.type ? this.keyCode : 0);
        this.altKey = a.altKey;
        this.ctrlKey = a.ctrlKey;
        this.shiftKey = a.shiftKey;
        "MozMousePixelScroll" == this.type ? (this.wheelDeltaX = a.axis == a.HORIZONTAL_AXIS ? a.detail : 0, this.wheelDeltaY = a.axis == a.HORIZONTAL_AXIS ? 0 : a.detail) : window.opera ? (this.wheelDeltaX = 0, this.wheelDeltaY = a.detail) : 0 == a.wheelDelta%120 ? "WebkitTransform"in window.document.documentElement.style ? window.chrome && 0 == window.navigator.platform.indexOf("Mac") ? (this.wheelDeltaX = a.wheelDeltaX /
        - 30, this.wheelDeltaY = a.wheelDeltaY/-30) : (this.wheelDeltaX = a.wheelDeltaX/-1.2, this.wheelDeltaY = a.wheelDeltaY/-1.2) : (this.wheelDeltaX = 0, this.wheelDeltaY = a.wheelDelta/-1.6) : (this.wheelDeltaX = a.wheelDeltaX/-3, this.wheelDeltaY = a.wheelDeltaY/-3);
        this.Gk = a.pageX;
        this.Hk = a.pageY
    }
};
m.Nd = function(a, b, c, d) {
    return m.vb(m.Od, function(e) {
        return e[0] == a && e[1] == b && e[2] == c && e[4]==!!d
    })
};
m.J = function(a, b, c, d) {
    if (!a ||!a.addEventListener&&!a.attachEvent)
        return "";
    d=!!d;
    var e = m.Nd(a, b, c, d);
    if (e)
        return e;
    var e=++Pd.count + "", f=!("mouseenter" != b && "mouseleave" != b ||!a.addEventListener || "onmouseenter"in window.document), h;
    h = f ? function(d) {
        d = new m.Ld(d);
        if (!m.Zb(d.relatedTarget, function(b) {
            return b == a
        }, !0))
            return d.currentTarget = a, d.type = b, c.call(a, d)
    } : function(b) {
        b = new m.Ld(b);
        b.currentTarget = a;
        return c.call(a, b)
    };
    h = m.$a(h);
    m.Od[e] = [a, b, c, h, d];
    a.addEventListener ? "mouseenter" == b && f ? a.addEventListener("mouseover",
    h, d) : "mouseleave" == b && f ? a.addEventListener("mouseout", h, d) : "mousewheel" == b && "MozBoxSizing"in window.document.documentElement.style ? a.addEventListener("MozMousePixelScroll", h, d) : a.addEventListener(b, h, d) : a.attachEvent("on" + b, h);
    return e
};
m.Qd = function(a) {
    this.g = a
};
m.Rd = function(a) {
    a = (a.g.cookie || "").split(m.Sd);
    for (var b = [], c = [], d, e, f = 0; e = a[f]; f++)
        d = e.indexOf("="), - 1 == d ? (b.push(""), c.push(e)) : (b.push(e.substring(0, d)), c.push(e.substring(d + 1)));
    return {
        keys: b,
        values: c
    }
};
m.Td = function(a, b, c) {
    m.Ud.set("" + a, b, c, "/", "youtube.com")
};
m.Vd = function(a, b) {
    return m.Ud.get("" + a, b)
};
m.Wd = function(a, b) {
    a=!!a;
    m.l("_lactCookie", a, window);
    if (null == m.n("_lact", window)) {
        if (a && b) {
            var c = m.Vd("ACTIVITY", "-1");
            m.l("_lact", (0, window.parseInt)(c, 10), window)
        } else 
            m.l("_lact", - 1, window), Xd();
        m.J(window.document, "keypress", Xd);
        m.J(window.document, "mousedown", Xd);
        m.J(window.document, "mouseup", Xd)
    }
};
var Xd = function() {
    var a = m.n("_lact", window);
    null == a && (m.Wd(), a = m.n("_lact", window));
    var b = (0, m.H)();
    m.l("_lact", b, window);
    m.n("_lactCookie", window) && 1E3 <= b - a && m.Td("ACTIVITY", "" + b)
};
m.Yd = function() {
    var a = m.n("_lact", window);
    return null == a?-1 : Math.max((0, m.H)() - a, 0)
};
m.Zd = function(a, b, c) {
    var d = m.jc(a);
    if (d == m.jc(window.location.href) ||!d && 0 == a.lastIndexOf("/", 0))
        if (a = m.mc(a), d = a.indexOf("#"), a = 0 > d ? a : a.substr(0, d))
            c ? (c = (0, window.parseInt)(c, 10), (0, window.isFinite)(c) && 0 < c && $d(a, b, c)) : $d(a, b)
};
var $d = function(a, b, c) {
    for (var d = 0, e = 0; e < a.length; ++e)
        d = 31 * d + a.charCodeAt(e), d%=4294967296;
    a = "s_tempdata-" + d;
    c = c || 5;
    b = b ? m.qc(b) : "";
    m.Td(a, b, c)
};
m.ae = function(a, b, c) {
    c = c || "";
    window.location = m.rc(a, b || {}) + c
};
m.be = function(a, b) {
    var c = window.ytspf || {};
    b && m.Zd(a, b);
    c.enabled ? window.spf.navigate(a) : m.ae(a)
};
m.ce = function(a, b, c) {
    var d = m.u("EVENT_ID");
    d && (b || (b = {}), b.ei || (b.ei = d));
    b && m.Zd(a, b);
    if (c)
        return !1;
    m.be(a);
    return !0
};
var de = function(a) {
    m.da.setTimeout(function() {
        throw a;
    }, 0)
};
var ee = function() {
    var a = m.da.MessageChannel;
    "undefined" === typeof a && "undefined" !== typeof window && window.postMessage && window.addEventListener && (a = function() {
        var a = window.document.createElement("iframe");
        a.style.display = "none";
        a.src = "";
        window.document.documentElement.appendChild(a);
        var b = a.contentWindow, a = b.document;
        a.open();
        a.write("");
        a.close();
        var c = "callImmediate" + Math.random(), d = "file:" == b.location.protocol ? "*": b.location.protocol + "//" + b.location.host, a = (0, m.p)(function(a) {
            if (("*" == d || a.origin ==
            d) && a.data == c)
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
    if ("undefined" !== typeof a&&!Cb("Trident")&&!Cb("MSIE")) {
        var b = new a, c = {}, d = c;
        b.port1.onmessage = function() {
            if (m.ca(c.next)) {
                c = c.next;
                var a = c.Mp;
                c.Mp = null;
                a()
            }
        };
        return function(a) {
            d.next = {
                Mp: a
            };
            d = d.next;
            b.port2.postMessage(0)
        }
    }
    return "undefined" !== typeof window.document && "onreadystatechange"in window.document.createElement("script") ? function(a) {
        var b =
        window.document.createElement("script");
        b.onreadystatechange = function() {
            b.onreadystatechange = null;
            b.parentNode.removeChild(b);
            b = null;
            a();
            a = null
        };
        window.document.documentElement.appendChild(b)
    } : function(a) {
        m.da.setTimeout(a, 0)
    }
};
var fe = function(a, b) {
    ge || he();
    ie || (ge(), ie=!0);
    je.push(new ke(a, b))
};
var he = function() {
    if (m.da.Promise && m.da.Promise.resolve) {
        var a = m.da.Promise.resolve();
        ge = function() {
            a.then(le)
        }
    } else 
        ge = function() {
            var a = le;
            !m.ma(m.da.setImmediate) || m.da.Window && m.da.Window.prototype.setImmediate == m.da.setImmediate ? (me || (me = ee()), me(a)) : m.da.setImmediate(a)
        }
};
var le = function() {
    for (; je.length;) {
        var a = je;
        je = [];
        for (var b = 0; b < a.length; b++) {
            var c = a[b];
            try {
                c.se.call(c.scope)
            } catch (d) {
                de(d)
            }
        }
    }
    ie=!1
};
var ke = function(a, b) {
    this.se = a;
    this.scope = b
};
m.ne = function(a) {
    a.prototype.then = a.prototype.then;
    a.prototype.$goog_Thenable=!0
};
m.oe = function(a) {
    if (!a)
        return !1;
    try {
        return !!a.$goog_Thenable
    } catch (b) {
        return !1
    }
};
m.pe = function(a, b) {
    this.j = 0;
    this.F = void 0;
    this.g = this.k = null;
    this.C = this.D=!1;
    try {
        var c = this;
        a.call(b, function(a) {
            qe(c, 2, a)
        }, function(a) {
            qe(c, 3, a)
        })
    } catch (d) {
        qe(this, 3, d)
    }
};
m.re = function(a) {
    return new m.pe(function(b) {
        b(a)
    })
};
var se = function(a, b) {
    if (0 == a.j)
        if (a.k) {
            var c = a.k;
            if (c.g) {
                for (var d = 0, e =- 1, f = 0, h; h = c.g[f]; f++)
                    if (h = h.Gf)
                        if (d++, h == a && (e = f), 0 <= e && 1 < d)
                            break;
                            0 <= e && (0 == c.j && 1 == d ? se(c, b) : (d = c.g.splice(e, 1)[0], te(c, d, 3, b)))
                        }
        } else 
            qe(a, 3, b)
    };
var ue = function(a, b) {
    a.g && a.g.length || 2 != a.j && 3 != a.j || ve(a);
    a.g || (a.g = []);
    a.g.push(b)
};
m.we = function(a, b, c, d) {
    var e = {
        Gf: null,
        In: null,
        Jn: null
    };
    e.Gf = new m.pe(function(a, h) {
        e.In = b ? function(c) {
            try {
                var e = b.call(d, c);
                a(e)
            } catch (w) {
                h(w)
            }
        } : a;
        e.Jn = c ? function(b) {
            try {
                var e = c.call(d, b);
                !m.ca(e) && b instanceof xe ? h(b) : a(e)
            } catch (w) {
                h(w)
            }
        } : h
    });
    e.Gf.k = a;
    ue(a, e);
    return e.Gf
};
var qe = function(a, b, c) {
    if (0 == a.j) {
        if (a == c)
            b = 3, c = new TypeError("Promise cannot resolve to itself");
        else {
            if (m.oe(c)) {
                a.j = 1;
                c.then(a.H, a.L, a);
                return 
            }
            if (m.na(c))
                try {
                    var d = c.then;
                    if (m.ma(d)) {
                        ye(a, c, d);
                        return 
                    }
            } catch (e) {
                b = 3, c = e
            }
        }
        a.F = c;
        a.j = b;
        ve(a);
        3 != b || c instanceof xe || ze(a, c)
    }
};
var ye = function(a, b, c) {
    function d(b) {
        f || (f=!0, a.L(b))
    }
    function e(b) {
        f || (f=!0, a.H(b))
    }
    a.j = 1;
    var f=!1;
    try {
        c.call(b, e, d)
    } catch (h) {
        d(h)
    }
};
var ve = function(a) {
    a.D || (a.D=!0, fe(a.J, a))
};
var te = function(a, b, c, d) {
    if (2 == c)
        b.In(d);
    else {
        if (b.Gf)
            for (; a && a.C; a = a.k)
                a.C=!1;
        b.Jn(d)
    }
};
var ze = function(a, b) {
    a.C=!0;
    fe(function() {
        a.C && Ae.call(null, b)
    })
};
var xe = function(a) {
    m.ua.call(this, a)
};
var Be = function(a) {
    this.k = a;
    this.j = null;
    var b = this;
    a.then(function(a) {
        b.j = a
    })
};
var Ce = function(a, b) {
    var c = b || {}, d = De().then(function(b) {
        return b(a, {
            apiKey: c.lf || c.apiKey,
            environment: c.mf || c.environment,
            helpCenterPath: c.KA || c.helpCenterPath,
            locale: c.locale || c.locale,
            productData: c.LA || c.productData,
            receiverUri: c.MA || c.receiverUri,
            theme: c.theme || c.theme,
            window: c.window || c.window
        })
    });
    return new Be(d)
};
var De = function() {
    if (Ee)
        return Ee;
    var a = m.n("help.service.Lazy.create");
    return a ? Ee = m.re(a) : Ee = (new m.pe(function(a, c) {
        var d = window.document.createElement("script");
        d.async=!0;
        d.src = "https://www.gstatic.com/inproduct_help/service/lazy.min.js";
        d.onload = d.onreadystatechange = function() {
            d.readyState && "loaded" != d.readyState && "complete" != d.readyState || a(null)
        };
        d.onerror = c;
        (window.document.head || window.document.getElementsByTagName("head")[0]).appendChild(d)
    })).then(function() {
        var a = m.n("help.service.Lazy.create");
        if (!a)
            throw Error("Failed to load help.service.Lazy.create from https://www.gstatic.com/inproduct_help/service/lazy.min.js");
        return a
    })
};
var Fe = function(a, b, c) {
    a.k.then(function(a) {
        a[b].apply(a, c)
    })
};
var Ge = function(a) {
    a = a.match(/[\d]+/g);
    if (!a)
        return "";
    a.length = 3;
    return a.join(".")
};
var He = function(a) {
    return (a = a.exec(m.Db)) ? a[1] : ""
};
var Ie = function() {
    this.k = this.j = this.g = 0;
    this.C = "";
    var a = m.n("window.navigator.plugins"), b = m.n("window.navigator.mimeTypes"), a = a && a["Shockwave Flash"], b = b && b["application/x-shockwave-flash"], b = a && b && b.enabledPlugin && a.description || "";
    if (a = b) {
        var c = a.indexOf("Shockwave Flash");
        0 <= c && (a = a.substr(c + 15));
        for (var c = a.split(" "), d = "", a = "", e = 0, f = c.length; e < f; e++)
            if (d)
                if (a)
                    break;
                else 
                    a = c[e];
        else 
            d = c[e];
        d = d.split(".");
        c = (0, window.parseInt)(d[0], 10) || 0;
        d = (0, window.parseInt)(d[1], 10) || 0;
        e = 0;
        if ("r" == a.charAt(0) ||
        "d" == a.charAt(0))
            e = (0, window.parseInt)(a.substr(1), 10) || 0;
        a = [c, d, e]
    } else 
        a = [0, 0, 0];
    this.C = b;
    b = a;
    this.g = b[0];
    this.j = b[1];
    this.k = b[2];
    if (0 >= this.g) {
        var h, k, r, w;
        if (m.Je)
            try {
                h = new window.ActiveXObject("ShockwaveFlash.ShockwaveFlash")
            } catch (B) {
            h = null
        } else 
            r = window.document.body, w = window.document.createElement("object"), w.setAttribute("type", "application/x-shockwave-flash"), h = r.appendChild(w);
        if (h && "GetVariable"in h)
            try {
                k = h.GetVariable("$version")
            } catch (Q) {
            k = ""
        }
        r && w && r.removeChild(w);
        (h = k || "") ? (h = h.split(" ")[1].split(","),
        h = [(0, window.parseInt)(h[0], 10) || 0, (0, window.parseInt)(h[1], 10) || 0, (0, window.parseInt)(h[2], 10) || 0]) : h = [0, 0, 0];
        this.g = h[0];
        this.j = h[1];
        this.k = h[2]
    }
};
var Ke = function(a, b, c, d) {
    b = "string" == typeof b ? b.split(".") : [b, c, d];
    b[0] = (0, window.parseInt)(b[0], 10) || 0;
    b[1] = (0, window.parseInt)(b[1], 10) || 0;
    b[2] = (0, window.parseInt)(b[2], 10) || 0;
    return a.g > b[0] || a.g == b[0] && a.j > b[1] || a.g == b[0] && a.j == b[1] && a.k >= b[2]
};
var Le = function(a) {
    return - 1 < a.C.indexOf("Gnash")&&-1 == a.C.indexOf("AVM2") || 9 == a.g && 1 == a.j || 9 == a.g && 0 == a.j && 1 == a.k?!1 : 9 <= a.g
};
var Me = function(a) {
    return m.Ne?!Ke(a, 11, 2) : m.Oe?!Ke(a, 11, 3) : !Le(a)
};
m.Pe = function() {
    return m.Qe ? "Webkit" : m.Re ? "Moz" : m.E ? "ms" : m.Se ? "O" : null
};
m.Te = function() {
    return m.Qe ? "-webkit" : m.Re ? "-moz" : m.E ? "-ms" : m.Se ? "-o" : null
};
m.Ve = function(a, b, c, d) {
    this.top = a;
    this.right = b;
    this.bottom = c;
    this.left = d
};
m.We = function(a, b) {
    return a && b ? b instanceof m.Ve ? b.left >= a.left && b.right <= a.right && b.top >= a.top && b.bottom <= a.bottom : b.x >= a.left && b.x <= a.right && b.y >= a.top && b.y <= a.bottom : !1
};
m.Xe = function(a, b) {
    var c = m.Ha(b);
    if (void 0 === a.style[c]) {
        var d = m.Pe() + m.Ia(c);
        if (void 0 !== a.style[d])
            return d
    }
    return c
};
m.Ye = function(a, b) {
    "number" == typeof a && (a = (b ? Math.round(a) : a) + "px");
    return a
};
m.Ze = function(a, b) {
    a.style.display = b ? "" : "none"
};
m.$e = function() {
    this.g = "";
    this.j = af
};
var bf = function(a) {
    return a instanceof m.$e && a.constructor === m.$e && a.j === af ? a.g : "type_error:Const"
};
var cf = function() {
    this.g = "";
    this.j = df
};
var ef = function(a) {
    var b = new cf;
    b.g = a;
    return b
};
var ff = function() {
    this.g = gf
};
var hf = function() {
    this.g = jf
};
var kf = function() {
    this.g = "";
    this.k = lf;
    this.j = null
};
m.mf = function(a) {
    return a instanceof kf && a.constructor === kf && a.k === lf ? a.g : "type_error:SafeHtml"
};
m.nf = function(a, b) {
    if (!of.test(a))
        throw Error("Invalid tag name <" + a + ">.");
    if (a.toLowerCase()in pf)
        throw Error("Tag name <" + a + "> is not allowed for SafeHtml.");
    var c = null, d = "<" + a;
    if (b)
        for (var e in b) {
            if (!of.test(e))
                throw Error('Invalid attribute name "' + e + '".');
                var f = b[e];
                if (null != f) {
                    var h, k = a;
                    h = e;
                    if (f instanceof m.$e)
                        f = bf(f);
                    else if ("style" == h.toLowerCase()) {
                        if (!m.na(f))
                            throw Error('The "style" attribute requires goog.html.SafeStyle or map of style properties, ' + typeof f + " given: " + f);
                            if (!(f instanceof
                            cf)) {
                                var k = "", r = void 0;
                                for (r in f) {
                                    if (!/^[-_a-zA-Z0-9]+$/.test(r))
                                        throw Error("Name allows only [-_a-zA-Z0-9], got: " + r);
                                        var w = f[r];
                                        null != w && (w instanceof m.$e ? w = bf(w) : qf.test(w) || (w = "zClosurez"), k += r + ":" + w + ";")
                                    }
                                    f = k ? ef(k) : rf
                            }
                            f = f instanceof cf && f.constructor === cf && f.j === df ? f.g : "type_error:SafeStyle"
                        } else {
                            if (/^on/i.test(h))
                                throw Error('Attribute "' + h + '" requires goog.string.Const value, "' + f + '" given.');
                                if (h.toLowerCase()in sf)
                                    if (f instanceof hf)
                                        f = f instanceof hf && f.constructor === hf && f.g === jf ? "" : "type_error:TrustedResourceUrl";
                                    else if (f instanceof ff)
                                        f = f instanceof ff && f.constructor === ff && f.g === gf ? "" : "type_error:SafeUrl";
                                    else 
                                        throw Error('Attribute "' + h + '" on tag "' + k + '" requires goog.html.SafeUrl or goog.string.Const value, "' + f + '" given.');
                        }
                        f.$e && (f = f.Ze());
                        h = h + '="' + m.wa(String(f)) + '"';
                        d = d + (" " + h)
                    }
            }
    e = void 0;
    m.ca(e) ? m.ha(e) || (e = [e]) : e = [];
    !0 === tf[a.toLowerCase()] ? d += ">" : (c = uf(e), d += ">" + m.mf(c) + "</" + a + ">", c = c.$f());
    (e = b && b.dir) && (/^(ltr|rtl|auto)$/i.test(e) ? c = 0 : c = null);
    return vf(d, c)
};
var uf = function(a) {
    function b(a) {
        if (m.ha(a))(0, m.y)(a, b);
        else {
            if (!(a instanceof kf)) {
                var f = null;
                a.Ik && (f = a.$f());
                a = vf(m.wa(a.$e ? a.Ze() : String(a)), f)
            }
            d += m.mf(a);
            a = a.$f();
            0 == c ? c = a : 0 != a && c != a && (c = null)
        }
    }
    var c = 0, d = "";
    (0, m.y)(arguments, b);
    return vf(d, c)
};
var vf = function(a, b) {
    var c = new kf;
    c.g = a;
    c.j = b;
    return c
};
m.wf = function(a, b) {
    (a = m.F(a)) && a.style && (m.Ze(a, b), m.rb(a, "hid", !b))
};
m.K = function(a) {
    (0, m.y)(arguments, function(a) {
        m.wf(a, !0)
    })
};
var xf = function(a, b) {
    var c = m.u("FEEDBACK_LOCALE_LANGUAGE"), d = m.yf, e = m.u("FEEDBACK_LOCALE_EXTRAS", {});
    a ? m.zb(d, a) : m.zb(d, e);
    try {
        var f, h = m.n("yt.player.getPlayerByElement");
        (f = h ? h("player-api") : null) && f.pauseVideo && f.pauseVideo();
        var k = Ie.getInstance();
        d.flashVersion = k.getVersion().join(".");
        f && (d.playback_id = f.getVideoData().cpn)
    } catch (r) {}
    b && m.zb(d, {
        trackingParam: b
    });
    return {
        helpCenterPath: "/youtube",
        locale: c,
        productData: d
    }
};
var zf = function() {
    var a = m.u("SESSION_INDEX"), b = m.u("FEEDBACK_BUCKET_ID"), c = {
        abuseLink: "https://support.google.com/youtube/bin/answer.py?answer=140536",
        customZIndex: "2000000005"
    };
    a && (c.authuser = a + "");
    b && (c.bucket = b);
    return c
};
m.Af = function(a, b) {
    try {
        var c = (a || "59") + "", d = xf(b), e = zf();
        Ce(c, d).C(e);
        return !1
    } catch (f) {
        return !0
    }
};
m.Bf = function(a, b, c, d, e, f, h, k) {
    b = (b || "59") + "";
    d = xf(d, h);
    a = {
        context: c,
        anchor: a,
        enableSendFeedback: f?!1: !0,
        defaultHelpArticleId: k
    };
    m.zb(a, zf());
    try {
        var r = Ce(b, d);
        if (e&&!Cf)
            try {
                r.g(a), Cf=!0
        } catch (w) {}
        r.D(a);
        return !1
    } catch (B) {
        return !0
    }
};
m.Df = function(a, b, c, d) {
    return m.Bf(void 0, d, b, c, void 0, void 0, void 0, a)
};
m.Ef = function() {
    var a = m.Vd("PREF");
    if (a)
        for (var a = (0, window.unescape)(a).split("&"), b = 0; b < a.length; b++) {
            var c = a[b].split("="), d = c[0];
            (c = c[1]) && (m.Ff[d] = c.toString())
        }
};
var Gf = function(a) {
    if (/^f([1-9][0-9]*)$/.test(a))
        throw "ExpectedRegexMatch: " + a;
};
var Hf = function(a) {
    if (!/^\w+$/.test(a))
        throw "ExpectedRegexMismatch: " + a;
};
m.If = function(a) {
    a = void 0 !== m.Ff[a] ? m.Ff[a].toString() : null;
    return null != a && /^[A-Fa-f0-9]+$/.test(a) ? (0, window.parseInt)(a, 16) : null
};
m.Jf = function(a, b) {
    return !!((m.If("f" + (Math.floor(b / 31) + 1)) || 0) & 1<<b%31)
};
var Kf = function(a, b, c) {
    if (b) {
        a = m.ja(a) ? m.Kb(a) : a;
        c = m.Gd(c);
        var d = m.yb(c.attrs);
        d.tabindex = 0;
        var e = m.yb(c.params);
        e.flashvars = m.qc(c.args);
        if (m.Je) {
            d.classid = "clsid:D27CDB6E-AE6D-11cf-96B8-444553540000";
            e.movie = b;
            b = window.document.createElement("object");
            for (var f in d)
                b.setAttribute(f, d[f]);
            for (f in e)
                d = window.document.createElement("param"), d.setAttribute("name", f), d.setAttribute("value", e[f]), b.appendChild(d)
            } else {
            d.type = "application/x-shockwave-flash";
            d.src = b;
            b = window.document.createElement("embed");
            b.setAttribute("name", d.id);
            for (f in d)
                b.setAttribute(f, d[f]);
            for (f in e)
                b.setAttribute(f, e[f])
            }
        e = window.document.createElement("div");
        e.appendChild(b);
        a.innerHTML = e.innerHTML
    }
};
m.Lf = function(a, b) {
    a = m.ja(a) ? m.Kb(a) : a;
    b = m.Gd(b);
    if (window != window.top) {
        var c = null;
        window.document.referrer && (c = window.document.referrer.substring(0, 128));
        b.args.framer = c
    }
    c = Mf();
    Ke(c, b.minVersion) ? (c = Nf(b, c), Kf(a, c, b)) : Of(a, b, c)
};
var Pf = function(a, b, c) {
    if (a && a.attrs && a.attrs.id) {
        a = m.Gd(a);
        var d=!!b, e = m.F(a.attrs.id), f = e ? e.parentNode : null;
        if (e && f) {
            if (window != window.top) {
                var h = null;
                if (window.document.referrer) {
                    var k = window.document.referrer.substring(0, 128);
                    xd(k) || (h = k)
                } else 
                    h = "unknown";
                h && (d=!0, a.args.framer = h)
            }
            h = Mf();
            if (Ke(h, a.minVersion)) {
                var k = Nf(a, h), r = "";
                - 1 < window.navigator.userAgent.indexOf("Sony/COM2") || (r = e.getAttribute("src") || e.movie);
                (r != k || d) && Kf(f, k, a);
                Me(h) && Qf()
            } else 
                Of(f, a, h);
            c && c()
        } else 
            m.v(function() {
                Pf(a,
                b, c)
            }, 50)
    }
};
var Of = function(a, b, c) {
    0 == c.g && b.fallback ? b.fallback() : 0 == c.g && b.fallbackMessage ? b.fallbackMessage() : a.innerHTML = '<div id="flash-upgrade">' + m.hb("FLASH_UPGRADE", void 0, 'You need to upgrade your Adobe Flash Player to watchthis video. <br> <a href="http://get.adobe.com/flashplayer/">Download it from Adobe.</a>') + "</div>"
};
var Nf = function(a, b) {
    var c;
    (c = Le(b) && a.url) || (c = ( - 1 < window.navigator.userAgent.indexOf("Sony/COM2")&&!Ke(b, 9, 1, 58)?!1 : !0) && a.urlV9As2);
    return c || a.urlV8 || a.url
};
var Mf = function() {
    var a = Ie.getInstance(), b = m.Ef.getInstance();
    b.set("fv", a.getVersion().join("."));
    b.save();
    return a
};
var Qf = function() {
    var a = m.F("flash10-promo-div"), b = m.Jf(m.Ef.getInstance(), 107);
    a&&!b && m.K(a)
};
m.Rf = function(a, b) {
    var c = a.match(Sf);
    window.spf.style.load(a, c ? c[1] : "", b)
};
var Tf = function() {
    if (2.2 == Uf)
        return !0;
    var a;
    a = m.n("yt.player.utils.videoElement_");
    a || (a = window.document.createElement("video"), m.l("yt.player.utils.videoElement_", a, void 0));
    try {
        return !(!a ||!a.canPlayType ||!a.canPlayType('video/mp4; codecs="avc1.42001E, mp4a.40.2"')&&!a.canPlayType('video/webm; codecs="vp8.0, vorbis"'))
    } catch (b) {
        return !1
    }
};
var Vf = function() {};
m.Wf = function() {};
m.Xf = function(a) {
    this.g = a
};
m.Yf = function() {
    var a = null;
    try {
        a = window.localStorage || null
    } catch (b) {}
    this.g = a
};
var Zf = function() {
    var a = null;
    try {
        a = window.sessionStorage || null
    } catch (b) {}
    this.g = a
};
m.$f = function(a) {
    a = String(a);
    if (/^\s*$/.test(a) ? 0 : /^[\],:{}\s\u2028\u2029]*$/.test(a.replace(/\\["\\\/bfnrtu]/g, "@").replace(/"[^"\\\n\r\u2028\u2029\x00-\x08\x0a-\x1f]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, "]").replace(/(?:^|:|,)(?:[\s\u2028\u2029]*\[)+/g, "")))
        try {
            return eval("(" + a + ")")
    } catch (b) {}
    throw Error("Invalid JSON string: " + a);
};
m.ag = function(a) {
    return m.bg(new m.cg(void 0), a)
};
m.cg = function(a) {
    this.g = a
};
m.bg = function(a, b) {
    var c = [];
    dg(a, b, c);
    return c.join("")
};
var dg = function(a, b, c) {
    switch (typeof b) {
    case "string":
        eg(b, c);
        break;
    case "number":
        c.push((0, window.isFinite)(b)&&!(0, window.isNaN)(b) ? b : "null");
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
        if (m.ha(b)) {
            var d = b.length;
            c.push("[");
            for (var e = "", f = 0; f < d; f++)
                c.push(e), e = b[f], dg(a, a.g ? a.g.call(b, String(f), e) : e, c), e = ",";
            c.push("]");
            break
        }
        c.push("{");
        d = "";
        for (f in b)
            Object.prototype.hasOwnProperty.call(b, f) && (e = b[f], "function" != typeof e &&
            (c.push(d), eg(f, c), c.push(":"), dg(a, a.g ? a.g.call(b, f, e) : e, c), d = ","));
        c.push("}");
        break;
    case "function":
        break;
    default:
        throw Error("Unknown type: " + typeof b);
    }
};
var eg = function(a, b) {
    b.push('"', a.replace(fg, function(a) {
        if (a in gg)
            return gg[a];
        var b = a.charCodeAt(0), e = "\\u";
        16 > b ? e += "000" : 256 > b ? e += "00" : 4096 > b && (e += "0");
        return gg[a] = e + b.toString(16)
    }), '"')
};
m.hg = function(a) {
    this.g = a
};
var ig = function(a) {
    this.g = a
};
var jg = function(a) {
    this.data = a
};
var kg = function(a) {
    return !m.ca(a) || a instanceof jg ? a : new jg(a)
};
var lg = function(a) {
    this.g = a
};
m.mg = function(a) {
    var b = a.creation;
    a = a.expiration;
    return !!a && a < (0, m.H)()||!!b && b > (0, m.H)()
};
m.ng = function(a) {
    this.g = a
};
m.og = function(a, b, c) {
    var d = c && 0 < c ? c: 0;
    c = d ? (0, m.H)() + 1E3 * d : 0;
    if ((d = d ? m.pg : m.qg) && window.JSON) {
        m.ja(b) || (b = JSON.stringify(b, void 0));
        try {
            d.set(a, b, c)
        } catch (e) {
            d.remove(a)
        }
    }
};
m.rg = function(a) {
    if (!m.qg&&!m.pg ||!window.JSON)
        return null;
    var b;
    try {
        b = m.qg.get(a)
    } catch (c) {}
    if (!m.ja(b))
        try {
            b = m.pg.get(a)
    } catch (d) {}
    if (!m.ja(b))
        return null;
    try {
        b = JSON.parse(b, void 0)
    } catch (e) {}
    return b
};
var sg = function() {
    var a = {
        volume: 100,
        muted: !1
    }, b = m.rg("yt-player-volume") || {};
    a.volume = (0, window.isNaN)(b.volume) ? 100 : m.sb(b.volume, 0, 100);
    a.muted = void 0 == b.muted?!1 : b.muted;
    return a
};
var tg = function(a, b) {
    m.t.call(this);
    this.k = this.tf = a;
    this.J = b;
    this.F=!1;
    this.g = {};
    this.U = this.L = null;
    this.S = new m.Wa;
    m.Ta(this, this.S);
    this.C = {};
    this.D = this.X = this.j = this.Ca = this.fa = null;
    this.O=!1;
    this.ja = this.H = null;
    this.ba = {};
    this.Ga = ["onReady"];
    this.qa = null;
    this.Ba = 0;
    this.T = {};
    ug(this);
    this.we("onVolumeChange", (0, m.p)(this.Yp, this));
    this.we("onError", (0, m.p)(this.Xp, this));
    this.we("onTabOrderChange", (0, m.p)(this.Wp, this));
    this.we("WATCH_LATER_VIDEO_ADDED", (0, m.p)(this.Zp, this));
    this.we("WATCH_LATER_VIDEO_REMOVED",
    (0, m.p)(this.$p, this))
};
var vg = function(a, b) {
    a.Ca = b;
    a.fa = b.clone();
    a.j = a.fa.attrs.id || a.j;
    "video-player" == a.j && (a.j = a.J, a.fa.attrs.id = a.J);
    a.k.id == a.j && (a.j = a.j + "-player", a.fa.attrs.id = a.j);
    var c = a.fa.args, d;
    if (!(d = a.fa.args.eurl)) {
        d = window.document.location.toString();
        var e=!!d&&-1 != d.search(wg), f =- 1 != d.indexOf("/embed/");
        e != f && m.db(Error("isIframeEmbed(" + d + ") behavior is not consistent"));
        d = f ? window.document.referrer && window.document.referrer.substring(0, 128) || "unknown" : d
    }
    c.eurl = d;
    a.fa.args.enablejsapi = "1";
    a.fa.args.playerapiid =
    a.J;
    a.X || (a.X = xg(a, a.fa.args.jsapicallback || "onYouTubePlayerReady"));
    a.fa.args.jsapicallback = null;
    if (c = a.fa.attrs.width)
        a.k.style.width = m.Ye(Number(c) || c, !0);
    if (c = a.fa.attrs.height)
        a.k.style.height = m.Ye(Number(c) || c, !0);
    a.k.style.overflow = "hidden"
};
var yg = function(a) {
    a.fa.loaded || (a.fa.loaded=!0, "0" != a.fa.args.autoplay ? a.g.loadVideoByPlayerVars(a.fa.args) : a.g.cueVideoByPlayerVars(a.fa.args))
};
var zg = function(a) {
    if (!m.ca(a.fa.disable.flash)) {
        var b = a.fa.disable, c;
        c = Ke(Mf(), a.fa.minVersion);
        b.flash=!c
    }
    return !a.fa.disable.flash
};
var Ag = function(a) {
    var b = Bg(a);
    b && b.stopVideo && b.stopVideo();
    if (zg(a)) {
        var c = a.fa;
        b && b.getUpdatedConfigurationData && (b = m.Gd(b.getUpdatedConfigurationData()), b.args.video_id == c.args.video_id && (c = b));
        c.args.autoplay = 1;
        c.args.html5_unavailable = "1";
        vg(a, c);
        Cg(a, "flash")
    }
};
var Cg = function(a, b) {
    if (!a.ha()) {
        if (!b) {
            var c;
            if (!(c=!a.fa.html5 && zg(a))) {
                if (!m.ca(a.fa.disable.html5)) {
                    if (c = Tf())
                        c = Dg(a) || a.fa.assets.js;
                    a.fa.disable.html5=!c;
                    c || (a.fa.args.html5_unavailable = "1")
                }
                c=!!a.fa.disable.html5
            }
            b = c ? zg(a) ? "flash" : "unsupported" : "html5"
        }("flash" == b ? a.Ky : "html5" == b ? a.Ly : a.My).call(a)
    }
};
var Dg = function(a) {
    var b=!0, c = Bg(a);
    c && a.fa && (a = a.fa, b = m.I(c, "version") == a.assets.js);
    return b&&!!m.n("yt.player.Application.create")
};
var Bg = function(a) {
    var b = m.F(a.j);
    !b && a.k && a.k.querySelector && (b = a.k.querySelector("#" + a.j));
    return b
};
var Eg = function(a, b, c) {
    var d = b[c];
    return function() {
        try {
            return a.qa = null, d.apply(b, arguments)
        } catch (e) {
            "Bad NPObject as private data!" != e.message && (e.message += " (" + c + ")", a.qa = e, m.db(e, "WARNING"))
        }
    }
};
var ug = function(a) {
    a.F=!1;
    if (a.U)
        for (var b in a.C)
            a.U(b, a.C[b]);
    for (var c in a.T)
        m.bb((0, window.parseInt)(c, 10));
    a.T = {};
    a.L = null;
    a.U = null;
    for (var d in a.g)
        a.g[d] = null;
    a.g.addEventListener = (0, m.p)(a.we, a);
    a.g.removeEventListener = (0, m.p)(a.zt, a);
    a.g.destroy = (0, m.p)(a.dispose, a);
    a.g.getLastError = (0, m.p)(a.wt, a);
    a.g.getPlayerType = (0, m.p)(a.xt, a);
    a.g.getCurrentVideoConfig = (0, m.p)(a.vt, a);
    a.g.loadNewVideoConfig = (0, m.p)(a.nj, a);
    a.g.isReady = (0, m.p)(a.At, a)
};
var xg = function(a, b) {
    var c = b;
    if ("string" == typeof b) {
        if (a.ba[b])
            return a.ba[b];
        c = function() {
            var a = m.n(b);
            a && a.apply(m.da, arguments)
        };
        a.ba[b] = c
    }
    return c ? c : null
};
var Fg = function(a, b) {
    var c = "ytPlayer" + b + a.J;
    a.C[b] = c;
    m.da[c] = function(c) {
        var e = m.v(function() {
            a.ha() || (a.S.publish(b, c), m.wb(a.T, e.toString()))
        }, 0);
        m.xb(a.T, e.toString(), !0)
    };
    return c
};
var Gg = function(a) {
    return (a = Bg(a)) ? "div" == a.tagName.toLowerCase() ? "html5" : "flash" : null
};
var Hg = function(a) {
    m.wc("dcp");
    a.cancel();
    ug(a);
    a.D = null;
    a.fa && (a.fa.loaded=!1);
    var b = Bg(a);
    "html5" == Gg(a) ? a.ja = b : b && b.destroy && b.destroy();
    m.Ub(a.tf)
};
m.Ig = function(a, b) {
    a = m.ja(a) ? m.Kb(a) : a;
    b = m.Gd(b);
    var c = Jg + "_" + m.oa(a), d = Kg[c];
    if (d)
        return d.nj(b), d.g;
    d = new tg(a, c);
    Kg[c] = d;
    m.z("player-added", d.g);
    m.Ua(d, m.q(aaa, d));
    m.v(function() {
        d.nj(b)
    }, 0);
    return d.g
};
var Lg = function() {
    for (var a in Kg) {
        var b = Kg[a];
        b && b.cancel()
    }
};
m.Mg = function(a) {
    if (a = m.F(a))
        a = Jg + "_" + m.oa(a), (a = Kg[a]) && a.dispose()
};
var aaa = function(a) {
    Kg[a.getId()] = null
};
m.Ng = function(a) {
    a = m.F(a);
    if (!a)
        return null;
    var b = Jg + "_" + m.oa(a), c = Kg[b];
    c || (c = new tg(a, b), Kg[b] = c);
    return c.g
};
var baa = function() {
    m.l("yt.abuse.botguardInitialized", Hd, void 0);
    m.l("yt.abuse.invokeBotguard", m.Jd, void 0);
    m.l("yt.player.exports.navigate", m.ce, void 0);
    m.l("yt.player.embed", m.Ig, void 0);
    m.l("yt.player.destroy", m.Mg, void 0);
    m.l("yt.player.cancelAll", Lg, void 0);
    m.l("yt.player.getPlayerByElement", m.Ng, void 0);
    m.l("yt.player.exports.feedbackStart", m.Af, void 0);
    m.l("yt.player.exports.feedbackShowArticle", m.Df, void 0);
    m.l("yt.util.activity.init", m.Wd, void 0);
    m.l("yt.util.activity.getTimeSinceActive",
    m.Yd, void 0);
    m.l("yt.util.activity.setTimestamp", Xd, void 0);
    var a = m.n("ytplayer.config");
    Og(a);
    m.n("ytspf.enabled") && (Pg = m.ab(Qg, 5E3), (a = m.Rg()) ? (a.addEventListener("onReady", Qg), a.addEventListener("onStateChange", Qg)) : Sg("unable to init PP monitor"))
};
var Og = function(a) {
    Lg();
    if (a = a || null) {
        var b = m.Rg(), c=!0;
        b && b.getVideoData && (c = b.getVideoData(), c = c.video_id != a.args.video_id || c.list != a.args.list);
        c ? (Tg = "", m.Ig("player-api", a)) : b.playVideo();
        a = m.Gd(a);
        a.loaded=!0
    }
    Qg();
    m.l("ytplayer.config", a, void 0)
};
m.Rg = function() {
    var a = m.n("yt.player.getPlayerByElement");
    return a ? a("player-api") : null
};
var Qg = function() {
    var a = m.F("player"), b = m.Rg();
    if (a && b) {
        var a = a && m.A(a, "off-screen"), c = 1 == (b && b.isReady() ? b.getPlayerState() : void 0), d = "watch" == m.u("PAGE_NAME");
        c&&!d && (m.n("ytplayer.REFACTOR") ? (Sg("PP playing off watch (paused)"), b.pauseVideo()) : Sg("PP playing off watch"));
        a && d ? Sg("PP off-screen on watch") : a || d || Sg("PP !off-screen off watch")
    }
};
var Sg = function(a) {
    var b = m.u("PAGE_NAME");
    b && (b = [b, Tg, m.n("_spf_state.nav-counter")], a += "(" + b.join(",") + ")", m.l("yt.www.persistentplayer.issue", a, void 0), m.db(Error(a), "WARNING"))
};
var Ug = function() {
    m.wc("ol");
    Vg("init");
    var a = m.u("PAGE_NAME");
    a && Vg("init-" + a);
    Nc()
};
var Vg = function(a) {
    m.Oc() ? Wg.push(m.Qc(m.q(m.ob, a), 0)) : m.z(a)
};
var Yg = function() {
    m.Tc(Wg);
    Wg.length = 0;
    var a = m.u("PAGE_NAME");
    a && m.ob("dispose-" + a);
    m.ob("dispose")
};
var caa = function() {
    Ug()
};
var daa = function() {
    m.u("TIMING_REPORT_ON_UNLOAD") && Mc(!0);
    m.u("EXP_DEFER_CSI_PING") && Jc("u");
    Yg()
};
var eaa = function(a, b, c, d, e) {
    for (var f = window.document.getElementsByTagName("script"), h=!1, k = 0, r = f.length; k < r; k++)
        if (0 < f[k].src.indexOf("/debug-")) {
            h=!0;
            break
        }
    h && (e || (e = Error(), e.message = a, e.fileName = b, e.lineNumber = c, (0, window.isNaN)(d) || (e.columnNumber = d)), m.db(e))
};
var Zg = function() {
    $g = 1;
    ah = 0;
    m.u("TIMING_REPORT_ON_UNLOAD") && Mc(!0);
    m.u("EXP_DEFER_CSI_PING") && Jc("n");
    Fc(void 0);
    Gc();
    m.u("CSI_TRACK_ON_BEGIN") && zc("nb");
    yc("mark_navigation_requested");
    bh(faa);
    m.ob("navigate")
};
var ch = function(a) {
    a = a.detail.part || a.detail.partial;
    m.wc("nc" + ah);
    ++ah;
    var b = 1 == $g;
    $g = 2;
    b ? (bh(gaa), dh()) : bh(haa);
    (a = a.data && a.data.swfcfg) && eh(a)
};
var fh = function() {};
var gh = function(a) {
    a = a.detail.response;
    var b = 1 == $g;
    $g = 3;
    b && (bh(iaa), dh());
    (a = a.data && a.data.swfcfg) && eh(a)
};
var hh = function(a) {
    m.wc("nd");
    a = a.detail.response;
    m.ih = a.cacheKey;
    a = a.timing;
    var b = window._spf_state, b = b && b["nav-counter"] || 0;
    Ac(void 0).yt_nav = b;
    zc("ne", a.responseStart);
    b = Math.max(0, a.responseStart - a.navigationStart);
    Ac(void 0).srt = b;
    b = a.spfCached ? "hot" : "warm";
    Ac(void 0).yt_lt = b;
    b=+!!a.spfPrefetched;
    Ac(void 0).yt_pft = b;
    Bc("req", a.responseStart - a.fetchStart);
    bh(jh);
    window.document.getElementById("content").style.height = "";
    Ug();
    $g = 0
};
var kh = function(a) {
    var b = a.detail.url;
    if (a = a.detail.err)
        a.message && (a.message += " (url: " + b + ")"), m.db(a)
};
var lh = function(a) {
    a = m.qc({
        spfreload: "1",
        url: a.detail.url,
        reason: a.detail.reason
    });
    m.Bd(a)
};
var jaa = function() {
    kaa()
};
var laa = function() {};
var dh = function() {
    window.scroll(0, 0);
    var a = window.document.getElementById("content");
    - 1 < a.className.indexOf("spf-animate") && (a.style.height = a.clientHeight + "px");
    Yg();
    for (var b in m.Za)
        delete m.Za[b];
    m.l("ytplayer.config", null, void 0);
    if ((a = m.Rg()) && a.stopVideo) {
        if (a.stopVideo(), a = a.getLastError())
            Tg = a, (a = m.F("movie_player")) && a.stopVideo ? (a.stopVideo(), Tg = "recovered") : Tg = "missing2"
    } else 
        Tg = "missing";
    Qg()
};
var eh = function(a) {
    "cfg"in xc(void 0) || m.wc("cfg");
    Og(a)
};
var bh = function(a) {
    function b() {
        var b = a[0], e = a[1], f = a[2];
        c.className = "";
        var h = c.style;
        h.transitionDuration = h.webkitTransitionDuration = b + "ms";
        h.width = e + "%";
        m.bb(mh);
        mh = m.v(function() {
            c.className = f
        }, b)
    }
    var c = window.document.getElementById("progress");
    c || (c = window.document.createElement("div"), c.id = "progress", c.innerHTML = "<dt><dd>", window.document.body.appendChild(c));
    m.Pc() ? (m.Sc(nh), nh = m.Qc(b, 0)) : (m.bb(mh), mh = m.v(b, 0))
};
var kaa = function() {
    var a = jh[0] + 50;
    m.bb(mh);
    mh = m.v(function() {
        var a = window.document.getElementById("progress");
        a && a.parentNode.removeChild(a)
    }, a)
};
var oh = function() {
    var a=!1;
    640 > (window.innerWidth || window.document.documentElement.clientWidth) && (a=!0);
    m.rb(m.F("player"), "watch-tiny", a);
    m.rb(window.document.body, "watch-while-skinny", a)
};
var ph = function() {
    m.Uc.call(this, "www/base");
    this.g = 0
};
var qh = function(a) {
    (a = a.detail.name) && Yc(a)
};
m.ba = [];
m.rh = m.rh || {};
m.da = this;
pa = "closure_uid_" + (1E9 * Math.random()>>>0);
qa = 0;
m.H = Date.now || function() {
    return + new Date
};
Function.prototype.bind = Function.prototype.bind || function(a, b) {
    if (1 < arguments.length) {
        var c = Array.prototype.slice.call(arguments, 1);
        c.unshift(this, a);
        return m.p.apply(null, c)
    }
    return (0, m.p)(this, a)
};
m.ta[" "] = m.ea;
m.s(m.ua, Error);
m.ua.prototype.name = "CustomError";
var ya, za, Aa, Ba, Ca, Da, xa;
m.Fa = String.prototype.trim ? function(a) {
    return a.trim()
} : function(a) {
    return a.replace(/^[\s\xa0]+|[\s\xa0]+$/g, "")
};
ya = /&/g;
za = /</g;
Aa = />/g;
Ba = /"/g;
Ca = /'/g;
Da = /\x00/g;
xa = /[\x00&<>"']/;
m.maa = 2147483648 * Math.random() | 0;
m.Qa = Array.prototype;
m.Na = m.Qa.indexOf ? function(a, b, c) {
    return m.Qa.indexOf.call(a, b, c)
} : function(a, b, c) {
    c = null == c ? 0 : 0 > c ? Math.max(0, a.length + c) : c;
    if (m.ja(a))
        return m.ja(b) && 1 == b.length ? a.indexOf(b, c) : - 1;
    for (; c < a.length; c++)
        if (c in a && a[c] === b)
            return c;
    return - 1
};
m.naa = m.Qa.lastIndexOf ? function(a, b, c) {
    return m.Qa.lastIndexOf.call(a, b, null == c ? a.length - 1 : c)
} : function(a, b, c) {
    c = null == c ? a.length - 1 : c;
    0 > c && (c = Math.max(0, a.length + c));
    if (m.ja(a))
        return m.ja(b) && 1 == b.length ? a.lastIndexOf(b, c) : - 1;
    for (; 0 <= c; c--)
        if (c in a && a[c] === b)
            return c;
    return - 1
};
m.y = m.Qa.forEach ? function(a, b, c) {
    m.Qa.forEach.call(a, b, c)
} : function(a, b, c) {
    for (var d = a.length, e = m.ja(a) ? a.split("") : a, f = 0; f < d; f++)
        f in e && b.call(c, e[f], f, a)
};
m.qb = m.Qa.filter ? function(a, b, c) {
    return m.Qa.filter.call(a, b, c)
} : function(a, b, c) {
    for (var d = a.length, e = [], f = 0, h = m.ja(a) ? a.split("") : a, k = 0; k < d; k++)
        if (k in h) {
            var r = h[k];
            b.call(c, r, k, a) && (e[f++] = r)
        }
    return e
};
m.sh = m.Qa.map ? function(a, b, c) {
    return m.Qa.map.call(a, b, c)
} : function(a, b, c) {
    for (var d = a.length, e = Array(d), f = m.ja(a) ? a.split("") : a, h = 0; h < d; h++)
        h in f && (e[h] = b.call(c, f[h], h, a));
    return e
};
m.th = m.Qa.reduce ? function(a, b, c, d) {
    d && (b = (0, m.p)(b, d));
    return m.Qa.reduce.call(a, b, c)
} : function(a, b, c, d) {
    var e = c;
    (0, m.y)(a, function(c, h) {
        e = b.call(d, e, c, h, a)
    });
    return e
};
m.uh = m.Qa.some ? function(a, b, c) {
    return m.Qa.some.call(a, b, c)
} : function(a, b, c) {
    for (var d = a.length, e = m.ja(a) ? a.split("") : a, f = 0; f < d; f++)
        if (f in e && b.call(c, e[f], f, a))
            return !0;
    return !1
};
m.vh = m.Qa.every ? function(a, b, c) {
    return m.Qa.every.call(a, b, c)
} : function(a, b, c) {
    for (var d = a.length, e = m.ja(a) ? a.split("") : a, f = 0; f < d; f++)
        if (f in e&&!b.call(c, e[f], f, a))
            return !1;
    return !0
};
m.t.prototype.lb=!1;
m.t.prototype.ha = function() {
    return this.lb
};
m.t.prototype.dispose = function() {
    this.lb || (this.lb=!0, this.N())
};
m.t.prototype.N = function() {
    if (this.xa)
        for (; this.xa.length;)
            this.xa.shift()()
};
m.s(m.Wa, m.t);
m.g = m.Wa.prototype;
m.g.ap = 1;
m.g.Qh = 0;
m.g.subscribe = function(a, b, c) {
    var d = this.Ob[a];
    d || (d = this.Ob[a] = []);
    var e = this.ap;
    this.ya[e] = a;
    this.ya[e + 1] = b;
    this.ya[e + 2] = c;
    this.ap = e + 3;
    d.push(e);
    return e
};
m.g.unsubscribe = function(a, b, c) {
    if (a = this.Ob[a]) {
        var d = this.ya;
        if (a = m.Ja(a, function(a) {
            return d[a + 1] == b && d[a + 2] == c
        }))
            return this.Ib(a)
    }
    return !1
};
m.g.Ib = function(a) {
    if (0 != this.Qh)
        return this.g || (this.g = []), this.g.push(a), !1;
    var b = this.ya[a];
    if (b) {
        var c = this.Ob[b];
        c && m.Oa(c, a);
        delete this.ya[a];
        delete this.ya[a + 1];
        delete this.ya[a + 2]
    }
    return !!b
};
m.g.publish = function(a, b) {
    var c = this.Ob[a];
    if (c) {
        this.Qh++;
        for (var d = m.Sa(arguments, 1), e = 0, f = c.length; e < f; e++) {
            var h = c[e];
            this.ya[h + 1].apply(this.ya[h + 2], d)
        }
        this.Qh--;
        if (this.g && 0 == this.Qh)
            for (; c = this.g.pop();)
                this.Ib(c);
        return 0 != e
    }
    return !1
};
m.g.clear = function(a) {
    if (a) {
        var b = this.Ob[a];
        b && ((0, m.y)(b, this.Ib, this), delete this.Ob[a])
    } else 
        this.ya.length = 0, this.Ob = {}
};
m.g.ua = m.aa(3);
m.g.N = function() {
    m.Wa.K.N.call(this);
    delete this.ya;
    delete this.Ob;
    delete this.g
};
m.Za = window.yt && window.yt.config_ || {};
m.l("yt.config_", m.Za, void 0);
m.l("yt.tokens_", window.yt && window.yt.tokens_ || {}, void 0);
m.fb = window.yt && window.yt.msgs_ || {};
m.l("yt.msgs_", m.fb, void 0);
m.Je = "Microsoft Internet Explorer" == window.navigator.appName;
var oaa = m.n("yt.pubsub.instance_") || new m.Wa;
m.Wa.prototype.subscribe = m.Wa.prototype.subscribe;
m.Wa.prototype.unsubscribeByKey = m.Wa.prototype.Ib;
m.Wa.prototype.publish = m.Wa.prototype.publish;
m.Wa.prototype.clear = m.Wa.prototype.clear;
m.l("yt.pubsub.instance_", oaa, void 0);
var kb = m.n("yt.pubsub.subscribedKeys_") || {};
m.l("yt.pubsub.subscribedKeys_", kb, void 0);
var mb = m.n("yt.pubsub.topicToKeys_") || {};
m.l("yt.pubsub.topicToKeys_", mb, void 0);
var lb = m.n("yt.pubsub.isSynchronous_") || {};
m.l("yt.pubsub.isSynchronous_", lb, void 0);
m.jb = m.n("yt.pubsub.skipSubId_") || null;
m.l("yt.pubsub.skipSubId_", m.jb, void 0);
var Ab = "constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");
t: {
    var wh = m.da.navigator;
    if (wh) {
        var xh = wh.userAgent;
        if (xh) {
            m.Db = xh;
            break t
        }
    }
    m.Db = ""
};
var yh, Hb, Gb, zh;
m.Se = Cb("Opera") || Cb("OPR");
m.E = Cb("Trident") || Cb("MSIE");
m.Re = Cb("Gecko")&&-1 == m.Db.toLowerCase().indexOf("webkit")&&!(Cb("Trident") || Cb("MSIE"));
m.Qe =- 1 != m.Db.toLowerCase().indexOf("webkit");
m.Qe && Cb("Mobile");
m.Oe = Cb("Macintosh");
m.Ne = Cb("Windows");
m.paa = Cb("Linux");
yh = m.da.navigator || null;
m.qaa=!!yh&&-1 != (yh.appVersion || "").indexOf("X11");
m.raa = Cb("Android");
m.saa = Cb("iPhone")&&!Cb("iPod")&&!Cb("iPad");
m.taa = Cb("iPad");
Hb = function() {
    var a = "", b;
    if (m.Se && m.da.opera)
        return a = m.da.opera.version, m.ma(a) ? a() : a;
    m.Re ? b = /rv\:([^\);]+)(\)|;)/ : m.E ? b = /\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/ : m.Qe && (b = /WebKit\/(\S+)/);
    b && (a = (a = b.exec(m.Db)) ? a[1] : "");
    return m.E && (b = Eb(), b > (0, window.parseFloat)(a)) ? String(b) : a
}();
Gb = {};
zh = m.da.document;
m.Jb = zh && m.E ? Eb() || ("CSS1Compat" == zh.compatMode ? (0, window.parseInt)(Hb, 10) : 5) : void 0;
var Rb;
Rb=!m.E || m.Ib(9);
m.uaa=!m.Re&&!m.E || m.E && m.Ib(9) || m.Re && m.Fb("1.9.1");
m.Ah = m.E&&!m.Fb("9");
m.vaa = m.E || m.Se || m.Qe;
m.waa = m.E&&!m.Ib(9);
var Ob = {
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
};
m.Bh = m.$b(!1);
m.Ch = m.$b(!0);
m.dc = "StopIteration"in m.da ? m.da.StopIteration : Error("StopIteration");
m.bc.prototype.next = function() {
    throw m.dc;
};
m.bc.prototype.fb = function() {
    return this
};
var kc = /^(?:([^:/?#.]+):)?(?:\/\/(?:([^/?#]*)@)?([^/#?]*?)(?::([0-9]+))?(?=[/#?]|$))?([^?#]+)?(?:\?([^#]*))?(?:#(.*))?$/, ic = m.Qe;
var wg = /^https?:\/\/([A-Za-z0-9-]{1,63}\.)*(corp\.google\.com|borg\.google\.com|prod\.google\.com|video\.google\.com|youtube\.com|youtube\.googleapis\.com|youtube-nocookie\.com|youtubeeducation\.com)(:[0-9]+)?\/embed\//;
var vc = {}, uc = 0, xaa = m.n("yt.net.ping.workerUrl_") || null;
m.l("yt.net.ping.workerUrl_", xaa, void 0);
var Ic, Kc, Lc;
var Hc = window.performance || window.mozPerformance || window.msPerformance || window.webkitPerformance || {}, Gc = (0, m.p)(Hc.clearResourceTimings || Hc.webkitClearResourceTimings || Hc.mozClearResourceTimings || Hc.msClearResourceTimings || Hc.oClearResourceTimings || m.ea, Hc), yc = Hc.mark ? function(a) {
    Hc.mark(a)
}
: m.ea;
m.g = m.Uc.prototype;
m.g.lx = function() {
    2 == this.j || this.init()
};
m.g.enable = function() {
    this.j = 1;
    (0, m.y)("string" == typeof this.page ? [this.page] : this.page, function(a) {
        a && (this.subscribe("init-" + a, this.lx, this), this.subscribe("dispose-" + a, this.dispose, this), m.u("PAGE_NAME") == a && m.Vc(this))
    }, this)
};
m.g.init = function() {
    m.Pc() && m.Sc(this.k);
    this.j = 2;
    this.D && this.D()
};
m.g.dispose = function() {
    this.j = 3;
    m.Sc(this.k);
    this.C && this.C()
};
m.g.disable = function() {
    this.j = 4;
    this.clear();
    this.dispose()
};
m.g.subscribe = function(a, b, c) {
    a = m.x(a, b, c);
    this.ya.push(a);
    return a
};
m.g.clear = function() {
    m.nb(this.ya);
    this.ya = []
};
var Xc = m.n("yt.modules.registry_") || {};
m.l("yt.modules.registry_", Xc, void 0);
m.Dh = null;
"undefined" != typeof window.XMLHttpRequest ? m.Dh = function() {
    return new window.XMLHttpRequest
} : "undefined" != typeof window.ActiveXObject && (m.Dh = function() {
    return new window.ActiveXObject("Microsoft.XMLHTTP")
});
var ad, bd, cd, dd, kd, nd = 0, ld = 0, jd = [], id = /.+\.googlevideo\.com|c\.youtube\.com|\/videoplayback\?|spf\=navigate/g;
var qd = {};
var vd = /\.vflset|-vfl[a-zA-Z0-9_+=-]+/, wd = /-[a-zA-Z]{2,3}_[a-zA-Z]{2,3}(?=(\/|$))/;
m.Ad = m.jc;
var Dd = {
    enablejsapi: 1
}, Ed = {}, Fd = {
    allowscriptaccess: "always",
    allowfullscreen: "true",
    bgcolor: "#000000"
};
Cd.prototype.clone = function() {
    var a = new Cd, b;
    for (b in this) {
        var c = this[b];
        "object" == m.ga(c) ? a[b] = m.yb(c) : a[b] = c
    }
    return a
};
m.Id = null;
m.yaa = m.Qe ? "webkit" : m.Re ? "moz" : m.E ? "ms" : m.Se ? "o" : "";
m.Eh = m.n("yt.dom.getNextId_");
if (!m.Eh) {
    m.Eh = function() {
        return ++zaa
    };
    m.l("yt.dom.getNextId_", m.Eh, void 0);
    var zaa = 0
};
m.g = m.Ld.prototype;
m.g.wa = null;
m.g.type = "";
m.g.target = null;
m.g.relatedTarget = null;
m.g.currentTarget = null;
m.g.data = null;
m.g.source = null;
m.g.state = null;
m.g.keyCode = 0;
m.g.charCode = 0;
m.g.altKey=!1;
m.g.ctrlKey=!1;
m.g.shiftKey=!1;
m.g.clientX = 0;
m.g.clientY = 0;
m.g.wheelDeltaX = 0;
m.g.wheelDeltaY = 0;
m.g.rotation = 0;
m.g.scale = 1;
m.g.changedTouches = null;
m.g.preventDefault = m.aa(4);
m.g.stopPropagation = m.aa(5);
m.g.stopImmediatePropagation = m.aa(6);
var Md = {
    stopImmediatePropagation: 1,
    stopPropagation: 1,
    preventMouseEvent: 1,
    preventManipulation: 1,
    preventDefault: 1,
    layerX: 1,
    layerY: 1,
    scale: 1,
    rotation: 1
};
m.Od = m.n("yt.events.listeners_") || {};
m.l("yt.events.listeners_", m.Od, void 0);
var Pd = m.n("yt.events.counter_") || {
    count: 0
};
m.l("yt.events.counter_", Pd, void 0);
m.Sd = /\s*;\s*/;
m.g = m.Qd.prototype;
m.g.isEnabled = m.aa(7);
m.g.set = function(a, b, c, d, e, f) {
    if (/[;=\s]/.test(a))
        throw Error('Invalid cookie name "' + a + '"');
    if (/[;\r\n]/.test(b))
        throw Error('Invalid cookie value "' + b + '"');
    m.ca(c) || (c =- 1);
    e = e ? ";domain=" + e : "";
    d = d ? ";path=" + d : "";
    f = f ? ";secure" : "";
    c = 0 > c ? "" : 0 == c ? ";expires=" + (new Date(1970, 1, 1)).toUTCString() : ";expires=" + (new Date((0, m.H)() + 1E3 * c)).toUTCString();
    this.g.cookie = a + "=" + b + e + d + c + f
};
m.g.get = function(a, b) {
    for (var c = a + "=", d = (this.g.cookie || "").split(m.Sd), e = 0, f; f = d[e]; e++) {
        if (0 == f.lastIndexOf(c, 0))
            return f.substr(c.length);
        if (f == a)
            return ""
    }
    return b
};
m.g.remove = function(a, b, c) {
    var d = m.ca(this.get(a));
    this.set(a, "", 0, b, c);
    return d
};
m.g.Sa = m.aa(8);
m.g.za = m.aa(9);
m.g.isEmpty = m.aa(10);
m.g.ua = m.aa(2);
m.g.rf = m.aa(11);
m.g.clear = function() {
    for (var a = m.Rd(this).keys, b = a.length - 1; 0 <= b; b--)
        this.remove(a[b])
};
m.Ud = new m.Qd(window.document);
m.Ud.j = 3950;
var me;
var ge, ie=!1, je = [];
m.pe.prototype.then = function(a, b, c) {
    return m.we(this, m.ma(a) ? a : null, m.ma(b) ? b : null, c)
};
m.ne(m.pe);
m.pe.prototype.cancel = function(a) {
    0 == this.j && fe(function() {
        var b = new xe(a);
        se(this, b)
    }, this)
};
m.pe.prototype.H = function(a) {
    this.j = 0;
    qe(this, 2, a)
};
m.pe.prototype.L = function(a) {
    this.j = 0;
    qe(this, 3, a)
};
m.pe.prototype.J = function() {
    for (; this.g && this.g.length;) {
        var a = this.g;
        this.g = [];
        for (var b = 0; b < a.length; b++)
            te(this, a[b], this.j, this.F)
    }
    this.D=!1
};
var Ae = de;
m.s(xe, m.ua);
xe.prototype.name = "cancel";
var Ee = null;
Be.prototype.C = function(a) {
    Fe(this, "startFeedback", arguments)
};
Be.prototype.D = function(a) {
    Fe(this, "startHelp", arguments)
};
Be.prototype.g = function(a) {
    Fe(this, "loadChatSupport", arguments)
};
var Fh=!1, Gh = "";
if (window.navigator.plugins && window.navigator.plugins.length) {
    var Hh = window.navigator.plugins["Shockwave Flash"];
    Hh && (Fh=!0, Hh.description && (Gh = Ge(Hh.description)));
    window.navigator.plugins["Shockwave Flash 2.0"] && (Fh=!0, Gh = "2.0.0.11")
} else if (window.navigator.mimeTypes && window.navigator.mimeTypes.length) {
    var Ih = window.navigator.mimeTypes["application/x-shockwave-flash"];
    (Fh = Ih && Ih.enabledPlugin) && (Gh = Ge(Ih.enabledPlugin.description))
} else 
    try {
        var Jh = new window.ActiveXObject("ShockwaveFlash.ShockwaveFlash.7"), Fh =
        !0, Gh = Ge(Jh.GetVariable("$version"))
} catch (Aaa) {
    try {
        Jh = new window.ActiveXObject("ShockwaveFlash.ShockwaveFlash.6"), Fh=!0, Gh = "6.0.21"
    } catch (Baa) {
        try {
            Jh = new window.ActiveXObject("ShockwaveFlash.ShockwaveFlash"), Fh=!0, Gh = Ge(Jh.GetVariable("$version"))
        } catch (Caa) {}
    }
}
m.Daa = Fh;
m.Kh = Gh;
var Lh, Mh, Nh, Oh, Ph, Qh, Rh, Eaa, Faa;
Rh = Qh = Ph = Oh = Nh = Mh = Lh=!1;
var Sh = m.Db;
Sh && ( - 1 != Sh.indexOf("Firefox") ? Lh=!0 : - 1 != Sh.indexOf("Camino") ? Mh=!0 : - 1 != Sh.indexOf("iPad") ? Oh=!0 : - 1 != Sh.indexOf("iPhone")||-1 != Sh.indexOf("iPod") ? Nh=!0 : - 1 != Sh.indexOf("Chrome") ? Qh=!0 : - 1 != Sh.indexOf("Android") ? Ph=!0 : - 1 != Sh.indexOf("Safari") && (Rh=!0));
m.Th = Lh;
Eaa = Mh;
m.Uh = Nh;
m.Vh = Oh;
Faa = Ph;
m.Wh = Qh;
m.Xh = Rh;
m.Gaa = function() {
    if (m.Th)
        return He(/Firefox\/([0-9.]+)/);
    if (m.E || m.Se)
        return Hb;
    if (m.Wh)
        return He(/Chrome\/([0-9.]+)/);
    if (m.Xh)
        return He(/Version\/([0-9.]+)/);
    if (m.Uh || m.Vh) {
        var a;
        if (a = /Version\/(\S+).*Mobile\/(\S+)/.exec(m.Db))
            return a[1] + "." + a[2]
    } else {
        if (Faa)
            return (a = He(/Android\s+([0-9.]+)/)) ? a : He(/Version\/([0-9.]+)/);
        if (Eaa)
            return He(/Camino\/([0-9.]+)/)
    }
    return ""
}();
m.fa(Ie);
Ie.prototype.getVersion = function() {
    return [this.g, this.j, this.k]
};
m.g = m.Ve.prototype;
m.g.clone = function() {
    return new m.Ve(this.top, this.right, this.bottom, this.left)
};
m.g.contains = function(a) {
    return m.We(this, a)
};
m.g.ceil = m.aa(12);
m.g.floor = function() {
    this.top = Math.floor(this.top);
    this.right = Math.floor(this.right);
    this.bottom = Math.floor(this.bottom);
    this.left = Math.floor(this.left);
    return this
};
m.g.round = function() {
    this.top = Math.round(this.top);
    this.right = Math.round(this.right);
    this.bottom = Math.round(this.bottom);
    this.left = Math.round(this.left);
    return this
};
m.g.scale = function(a, b) {
    var c = m.ka(b) ? b: a;
    this.left*=a;
    this.right*=a;
    this.top*=c;
    this.bottom*=c;
    return this
};
var tf = m.Bb("area base br col command embed hr img input keygen link meta param source track wbr".split(" "));
m.Haa = RegExp("^[^\u0591-\u07ff\u200f\ufb1d-\ufdff\ufe70-\ufefc]*[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02b8\u0300-\u0590\u0800-\u1fff\u200e\u2c00-\ufb1c\ufe00-\ufe6f\ufefd-\uffff]");
m.Iaa = RegExp("^[^A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02b8\u0300-\u0590\u0800-\u1fff\u200e\u2c00-\ufb1c\ufe00-\ufe6f\ufefd-\uffff]*[\u0591-\u07ff\u200f\ufb1d-\ufdff\ufe70-\ufefc]");
m.$e.prototype.$e=!0;
m.$e.prototype.Ze = function() {
    return this.g
};
m.$e.prototype.toString = function() {
    return "Const{" + this.g + "}"
};
var af = {};
cf.prototype.$e=!0;
var df = {};
cf.prototype.Ze = function() {
    return this.g
};
var rf = ef(""), qf = /^[-.%_!# a-zA-Z0-9]+$/;
ff.prototype.$e=!0;
ff.prototype.Ze = function() {
    return ""
};
ff.prototype.Ik=!0;
ff.prototype.$f = function() {
    return 1
};
var gf = {};
hf.prototype.$e=!0;
hf.prototype.Ze = function() {
    return ""
};
hf.prototype.Ik=!0;
hf.prototype.$f = function() {
    return 1
};
var jf = {};
var of, sf, pf, lf;
kf.prototype.Ik=!0;
kf.prototype.$f = function() {
    return this.j
};
kf.prototype.$e=!0;
kf.prototype.Ze = function() {
    return this.g
};
of = /^[a-zA-Z0-9-]+$/;
sf = m.Bb("action", "cite", "data", "formaction", "href", "manifest", "poster", "src");
pf = m.Bb("embed", "iframe", "link", "script", "style", "template");
lf = {};
m.Yh = vf("", 0);
m.Zh = m.ac(function() {
    if (m.E)
        return m.Fb("10.0");
    var a = window.document.createElement("div"), b = m.Te(), c = {
        transition: "opacity 1s linear"
    };
    b && (c[b + "-transition"] = "opacity 1s linear");
    b = m.nf("div", {
        style: c
    });
    a.innerHTML = m.mf(b);
    a = a.firstChild;
    b = a.style[m.Ha("transition")];
    return "" != ("undefined" !== typeof b ? b : a.style[m.Xe(a, "transition")] || "")
});
var Cf;
m.yf = {};
Cf=!1;
m.fa(m.Ef);
m.Ff = m.n("yt.prefs.UserPrefs.prefs_") || {};
m.l("yt.prefs.UserPrefs.prefs_", m.Ff, void 0);
m.g = m.Ef.prototype;
m.g.get = function(a, b) {
    Hf(a);
    Gf(a);
    var c = void 0 !== m.Ff[a] ? m.Ff[a].toString() : null;
    return null != c ? c : b ? b : ""
};
m.g.set = function(a, b) {
    Hf(a);
    Gf(a);
    if (null == b)
        throw "ExpectedNotNull";
    m.Ff[a] = b.toString()
};
m.g.remove = function(a) {
    Hf(a);
    Gf(a);
    delete m.Ff[a]
};
m.g.save = function() {
    var a;
    a = [];
    for (var b in m.Ff)
        a.push(b + "=" + (0, window.escape)(m.Ff[b]));
    a = a.join("&");
    m.Td("PREF", a, 63072E3)
};
m.g.clear = function() {
    m.Ff = {}
};
var Sf = /cssbin\/(?:debug-)?([a-zA-Z0-9_-]+?)(?:-2x|-web|-rtl|-vfl|.css)/;
var Uf;
var $h = m.Db, $h = $h.toLowerCase();
if ( - 1 != $h.indexOf("android")) {
    var ai = $h.match(/android\D*(\d\.\d)[^\;|\)]*[\;\)]/);
    if (ai)
        Uf = Number(ai[1]);
    else {
        var bi = {
            cupcake: 1.5,
            donut: 1.6,
            eclair: 2,
            froyo: 2.2,
            gingerbread: 2.3,
            honeycomb: 3,
            "ice cream sandwich": 4,
            jellybean: 4.1
        }, ci = $h.match("(" + m.ub(bi).join("|") + ")");
        Uf = ci ? bi[ci[0]] : 0
    }
} else 
    Uf = void 0;
m.s(m.Wf, Vf);
m.Wf.prototype.ua = m.aa(1);
m.Wf.prototype.clear = function() {
    var a = m.fc(this.fb(!0)), b = this;
    (0, m.y)(a, function(a) {
        b.remove(a)
    })
};
m.s(m.Xf, m.Wf);
m.g = m.Xf.prototype;
m.g.isAvailable = function() {
    if (!this.g)
        return !1;
    try {
        return this.g.setItem("__sak", "1"), this.g.removeItem("__sak"), !0
    } catch (a) {
        return !1
    }
};
m.g.set = function(a, b) {
    try {
        this.g.setItem(a, b)
    } catch (c) {
        if (0 == this.g.length)
            throw "Storage mechanism: Storage disabled";
        throw "Storage mechanism: Quota exceeded";
    }
};
m.g.get = function(a) {
    a = this.g.getItem(a);
    if (!m.ja(a) && null !== a)
        throw "Storage mechanism: Invalid value was encountered";
    return a
};
m.g.remove = function(a) {
    this.g.removeItem(a)
};
m.g.ua = m.aa(0);
m.g.fb = function(a) {
    var b = 0, c = this.g, d = new m.bc;
    d.next = function() {
        if (b >= c.length)
            throw m.dc;
        var d;
        d = c.key(b++);
        if (a)
            return d;
        d = c.getItem(d);
        if (!m.ja(d))
            throw "Storage mechanism: Invalid value was encountered";
        return d
    };
    return d
};
m.g.clear = function() {
    this.g.clear()
};
m.g.key = function(a) {
    return this.g.key(a)
};
m.s(m.Yf, m.Xf);
m.s(Zf, m.Xf);
var gg = {
    '"': '\\"',
    "\\": "\\\\",
    "/": "\\/",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\x0B": "\\u000b"
}, fg = /\uffff/.test("\uffff") ? /[\\\"\x00-\x1f\x7f-\uffff]/g: /[\\\"\x00-\x1f\x7f-\xff]/g;
m.hg.prototype.set = function(a, b) {
    m.ca(b) ? this.g.set(a, m.ag(b)) : this.g.remove(a)
};
m.hg.prototype.get = function(a) {
    var b;
    try {
        b = this.g.get(a)
    } catch (c) {
        return 
    }
    if (null !== b)
        try {
            return m.$f(b)
    } catch (d) {
        throw "Storage: Invalid value was encountered";
    }
};
m.hg.prototype.remove = function(a) {
    this.g.remove(a)
};
m.s(ig, m.hg);
ig.prototype.set = function(a, b) {
    ig.K.set.call(this, a, kg(b))
};
ig.prototype.j = function(a) {
    a = ig.K.get.call(this, a);
    if (!m.ca(a) || a instanceof Object)
        return a;
    throw "Storage: Invalid value was encountered";
};
ig.prototype.get = function(a) {
    if (a = this.j(a)) {
        if (a = a.data, !m.ca(a))
            throw "Storage: Invalid value was encountered";
    } else 
        a = void 0;
    return a
};
m.s(lg, ig);
lg.prototype.set = function(a, b, c) {
    if (b = kg(b)) {
        if (c) {
            if (c < (0, m.H)()) {
                lg.prototype.remove.call(this, a);
                return 
            }
            b.expiration = c
        }
        b.creation = (0, m.H)()
    }
    lg.K.set.call(this, a, b)
};
lg.prototype.j = function(a, b) {
    var c = lg.K.j.call(this, a);
    if (c)
        if (!b && m.mg(c))
            lg.prototype.remove.call(this, a);
        else 
            return c
};
m.s(m.ng, lg);
var di, ei;
di = new m.Yf;
m.pg = di.isAvailable() ? new m.ng(di) : null;
ei = new Zf;
m.qg = ei.isAvailable() ? new m.ng(ei) : null;
m.s(tg, m.t);
m.g = tg.prototype;
m.g.getId = function() {
    return this.J
};
m.g.nj = function(a, b) {
    this.ha() || (vg(this, a), Cg(this, b), this.F && yg(this))
};
m.g.vt = function() {
    return this.Ca
};
m.g.Ly = function() {
    if (!this.O) {
        var a = Dg(this);
        a && "html5" == Gg(this) ? (this.D = "html5", this.F || this.Je()) : (Hg(this), this.D = "html5", a && this.ja ? (this.tf.appendChild(this.ja), this.Je()) : (this.fa.loaded=!0, this.H = (0, m.p)(function() {
            var a = this.tf, c = this.fa.clone();
            m.n("yt.player.Application.create")(a, c);
            this.Je()
        }, this), this.O=!0, a ? this.H() : (m.rd(this.fa.assets.js, this.H), m.Rf(this.fa.assets.css))))
    }
};
m.g.Ky = function() {
    var a = this.fa.clone();
    a.attrs.width = a.attrs.width || "100%";
    a.attrs.height = a.attrs.height || "100%";
    "flash" == Gg(this) ? (this.D = "flash", this.F || Pf(a, !1, (0, m.p)(this.Je, this))) : (Hg(this), this.D = "flash", this.fa.loaded=!0, m.Lf(this.tf, a), this.Je())
};
m.g.Je = function() {
    var a = Bg(this), b=!1;
    try {
        a && a.getApiInterface && a.getApiInterface() && (b=!0)
    } catch (c) {}
    if (b)
        if (this.O=!1, a.isNotServable && a.isNotServable())
            Ag(this);
        else {
            ug(this);
            this.F=!0;
            a = Bg(this);
            a.addEventListener && (this.L = Eg(this, a, "addEventListener"));
            a.removeEventListener && (this.U = Eg(this, a, "removeEventListener"));
            for (var b = a.getApiInterface(), b = b.concat(a.getInternalApiInterface()), d = 0; d < b.length; d++) {
                var e = b[d];
                this.g[e] || (this.g[e] = Eg(this, a, e))
            }
            for (var f in this.C)
                this.L(f, this.C[f]);
                yg(this);
                this.X && this.X(this.g);
                this.S.publish("onReady", this.g)
        } else 
            this.Ba = m.v((0, m.p)(this.Je, this), 50)
    };
m.g.At = function() {
    return this.F
};
m.g.we = function(a, b) {
    if (!this.ha()) {
        var c = xg(this, b);
        if (c) {
            if (!m.La(this.Ga, a)&&!this.C[a]) {
                var d = Fg(this, a);
                this.L && this.L(a, d)
            }
            this.S.subscribe(a, c);
            "onReady" == a && this.F && m.v(m.q(c, this.g), 0)
        }
    }
};
m.g.zt = function(a, b) {
    if (!this.ha()) {
        var c = xg(this, b);
        c && this.S.unsubscribe(a, c)
    }
};
m.g.Wp = function(a) {
    a = a ? Xb : Wb;
    for (var b = a(window.document.activeElement); b && (1 != b.nodeType || (b.focus(), b != window.document.activeElement));)
        b = a(b)
};
m.g.Yp = function(a) {
    var b = {};
    b.volume = (0, window.isNaN)(a.volume) ? sg().volume : m.sb(a.volume, 0, 100);
    b.muted = void 0 == a.muted ? sg().muted : a.muted;
    m.og("yt-player-volume", b, 2592E3)
};
m.g.Xp = function(a) {
    5 == a && Ag(this)
};
m.g.Zp = function(a) {
    m.z("WATCH_LATER_VIDEO_ADDED", a)
};
m.g.$p = function(a) {
    m.z("WATCH_LATER_VIDEO_REMOVED", a)
};
m.g.My = function() {
    Hg(this);
    this.D = "unsupported";
    var a = 'Adobe Flash Player or an HTML5 supported browser is required for video playback. <br> <a href="http://get.adobe.com/flashplayer/">Get the latest Flash Player</a> <br> <a href="/html5">Learn more about upgrading to an HTML5 browser</a>', b = window.navigator.userAgent.match(/Version\/(\d).*Safari/);
    b && 5 <= (0, window.parseInt)(b[1], 10) && (a = 'Adobe Flash Player or QuickTime is required for video playback. <br> <a href="http://get.adobe.com/flashplayer/"> Get the latest Flash Player</a> <br> <a href="http://www.apple.com/quicktime/download/">Get the latest version of QuickTime</a>');
    b = this.fa.messages.player_fallback || a;
    a = m.F("player-unavailable");
    m.F("unavailable-submessage") && a && (m.F("unavailable-submessage").innerHTML = b, (b = m.G("icon", a)) && m.pd(b, "icon") && (b.src = m.I(b, "icon")), m.D(a, "hid"), m.C(m.F("player"), "off-screen-trigger"))
};
m.g.xt = function() {
    return this.D || Gg(this)
};
m.g.wt = function() {
    return this.qa
};
m.g.cancel = function() {
    this.H && m.td(this.fa.assets.js, this.H);
    m.bb(this.Ba);
    this.O=!1
};
m.g.N = function() {
    Hg(this);
    this.ba = null;
    for (var a in this.C)
        m.da[this.C[a]] = null;
    this.g = null;
    delete this.tf;
    delete this.k;
    this.fa && (this.Ca = this.fa = this.fa.fallback = null);
    tg.K.N.call(this)
};
var Kg = {}, Jg = "player_uid_" + (1E9 * Math.random()>>>0);
var Jaa = m.n("yt.abuse.botguardInitialized") || Hd;
m.l("yt.abuse.botguardInitialized", Jaa, void 0);
var Kaa = m.n("yt.abuse.invokeBotguard") || m.Jd;
m.l("yt.abuse.invokeBotguard", Kaa, void 0);
var Laa = m.n("yt.player.exports.navigate") || m.ce;
m.l("yt.player.exports.navigate", Laa, void 0);
var Maa = m.n("yt.player.embed") || m.Ig;
m.l("yt.player.embed", Maa, void 0);
var Naa = m.n("yt.player.destroy") || m.Mg;
m.l("yt.player.destroy", Naa, void 0);
var Oaa = m.n("yt.player.cancelAll") || Lg;
m.l("yt.player.cancelAll", Oaa, void 0);
var Paa = m.n("yt.player.getPlayerByElement") || m.Ng;
m.l("yt.player.getPlayerByElement", Paa, void 0);
var Qaa = m.n("yt.player.exports.feedbackStart") || m.Af;
m.l("yt.player.exports.feedbackStart", Qaa, void 0);
var Raa = m.n("yt.player.exports.feedbackShowArticle") || m.Df;
m.l("yt.player.exports.feedbackShowArticle", Raa, void 0);
var Saa = m.n("yt.util.activity.init") || m.Wd;
m.l("yt.util.activity.init", Saa, void 0);
var Taa = m.n("yt.util.activity.getTimeSinceActive") || m.Yd;
m.l("yt.util.activity.getTimeSinceActive", Taa, void 0);
var Uaa = m.n("yt.util.activity.setTimestamp") || Xd;
m.l("yt.util.activity.setTimestamp", Uaa, void 0);
var Pg, Tg = "";
var Wg = [];
var $g, mh, nh, ah, faa, iaa, gaa, haa, jh;
ah = 0;
faa = [900, 60, "waiting"];
iaa = [500, 99, "waiting"];
gaa = [300, 60, "waiting"];
haa = [400, 99, "waiting"];
jh = [300, 101, "done"];
var fi = [];
window.yt = window.yt || {};
m.l("yt.setConfig", m.Xa, void 0);
m.l("yt.getConfig", m.u, void 0);
m.l("yt.hasMsg", m.gb, void 0);
m.l("yt.pubsub.publish", m.z, void 0);
m.l("yt.pubsub.subscribe", m.x, void 0);
m.l("yt.setMsg", m.eb, void 0);
m.l("yt.setGoogMsg", function(a) {
    Ya(m.fb, arguments)
}, void 0);
m.l("yt.setAjaxToken", function() {}, void 0);
m.l("reportTimingMaps", function(a, b, c, d) {
    if (a)
        for (var e in a)
            m.wc(e, a[e], d);
    if (c)
        for (var f in c)
            Bc(f, c[f], d);
    if (b)
        for (var h in b)
            a = h, c = b[h], Ac(d)[a] = c;
    return Nc(d)
}, void 0);
m.l("yt.timing.reportAdToVideo", function(a, b, c) {
    c = {
        v: 2,
        s: "youtube",
        action: c ? "watch,ad2video_html5": "watch,ad2video"
    };
    a.aft = a.pbs = a.astv || a.actv;
    var d = [], e;
    for (e in a)
        "_" != e.charAt(0) && d.push(e + "." + a[e]);
    c.rt = d.join(",");
    for (var f in b)
        "_" != f.charAt(0) && (c[f] = b[f]);
    a = m.u("TIMING_INFO") || {};
    "e"in a && (c.e = a.e);
    "ei"in a && (c.ei = a.ei);
    Ec(c)
}, void 0);
m.s(ph, m.Uc);
ph.prototype.enable = function() {
    window.onload = caa;
    window.onunload = daa;
    window.onerror = eaa;
    var a = window.ytspf || {};
    a.enabled ? (window.addEventListener && (window.addEventListener("spfrequest", Zg), window.addEventListener("spfpartprocess", ch), window.addEventListener("spfpartdone", fh), window.addEventListener("spfprocess", gh), window.addEventListener("spfdone", hh), window.addEventListener("spferror", kh), window.addEventListener("spfreload", lh), window.addEventListener("spfjsbeforeunload", qh)), a.config = a.config ||
    {}, window.ytdepmap ? window.spf.script.ready("spf", function() {
        a.enabled = window.spf.init(a.config)
    }) : a.enabled = window.spf.init(a.config), this.subscribe("init", jaa), this.subscribe("dispose", laa)) : window.spf.dispose();
    this.subscribe("init", this.init, this);
    this.subscribe("dispose", this.dispose, this)
};
ph.prototype.init = function() {
    ph.K.init.call(this);
    (window.ytspf || {}).enabled || window.spf.dispose();
    $c();
    var a = window.ytPageFrameLoaded ||!1, b, c;
    !a && m.u("PAGEFRAME_JS") ? (b = m.u("PAGEFRAME_JS"), c = function() {
        m.n("ytbin.www.pageframe.setup")();
        window.ytPageFrameLoaded=!0
    }) : a && (a = m.n("yt.www.masthead.loadSearchbox")) && a();
    var a = m.u("JS_COMMON_MODULE"), d = m.u("JS_PAGE_MODULES");
    if (window.ytdepmap)
        d || (d = [a]), a = m.u("JS_DELAY_LOAD", void 0), 0 < a ? (m.bb(this.g), this.g = m.v(function() {
            b && m.rd(b, c);
            window.spf.script.require(d)
        },
        a)) : (b && m.rd(b, c), window.spf.script.require(d));
    else {
        var e = {};
        e[a] = d;
        a = m.u("JS_DELAY_LOAD", void 0);
        0 < a ? (m.bb(this.g), this.g = m.v(function() {
            b && m.rd(b, c);
            m.ud(e)
        }, a)) : (b && m.rd(b, c), m.ud(e))
    }
    m.u("SKYWATCH_ENABLED") && (fi.push(m.x("page-resize", oh)), oh());
    baa();
    var a=!!m.n("yt.reactLoaded"), f = m.u("REACT_JS");
    !a && f && m.rd(f, function() {
        m.l("yt.reactLoaded", !0, void 0)
    })
};
ph.prototype.dispose = function() {
    m.bb(this.g);
    var a = m.n("ytbin.www.pageframe.cancelSetup");
    a && a();
    m.nb(fi);
    fi.length = 0;
    m.cb(Pg);
    if (a = m.Rg())
        a.removeEventListener("onReady", Qg), a.removeEventListener("onStateChange", Qg);
    ph.K.dispose.call(this)
};
ph.prototype.disable = function() {
    ph.K.disable.call(this);
    window.removeEventListener && (window.removeEventListener("spfrequest", Zg), window.removeEventListener("spfpartprocess", ch), window.removeEventListener("spfpartdone", fh), window.removeEventListener("spfprocess", gh), window.removeEventListener("spfdone", hh), window.removeEventListener("spferror", kh), window.removeEventListener("spfreload", lh), window.removeEventListener("spfjsbeforeunload", qh));
    window.onload = null;
    window.onunload = null;
    window.onerror = null
};
m.Wc(new ph);
})(_yt_www);

