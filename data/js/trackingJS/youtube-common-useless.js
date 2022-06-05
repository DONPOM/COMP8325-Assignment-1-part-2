(function(m) {
    var window = this;
    var Vaa = function(a, b) {
        var c = [];
        m.ec(b, function(a) {
            var b;
            try {
                b = m.ng.prototype.j.call(this, a, !0)
            } catch (f) {
                if ("Storage: Invalid value was encountered" == f)
                    return;
                throw f;
            }
            m.ca(b) ? m.mg(b) && c.push(a) : c.push(a)
        }, a);
        return c
    };
    var Waa = function(a, b) {
        var c = Vaa(a, b);
        (0, m.y)(c, function(a) {
            m.ng.prototype.remove.call(this, a)
        }, a)
    };
    var gi = function(a) {
        if (window.document.body && window.document.documentElement) {
            var b = window.document.body.scrollTop + window.document.documentElement.scrollTop;
            a.Gk = a.clientX + (window.document.body.scrollLeft + window.document.documentElement.scrollLeft);
            a.Hk = a.clientY + b
        }
    };
    m.hi = function(a, b) {
        return m.ba[a] = b
    };
    var ii = function() {
        var a = m.pg;
        Waa(a, a.g.fb(!0))
    };
    m.ji = function(a, b) {
        var c = "f" + (Math.floor(a / 31) + 1), d = 1<<a%31, e = m.If(c) || 0, e = b ? e | d: e&~d;
        0 == e ? delete m.Ff[c] : (d = e.toString(16), m.Ff[c] = d.toString())
    };
    var ki = function(a) {
        m.we(a, null, m.ea, void 0)
    };
    var li = function(a) {
        m.ca(a.Hk) || gi(a);
        return a.Hk
    };
    var mi = function(a) {
        m.ca(a.Gk) || gi(a);
        return a.Gk
    };
    var Xaa = function(a, b) {
        var c = a.subscribe("ROOT_MENU_REMOVED", function(a) {
            b.apply(void 0, arguments);
            this.Ib(c)
        }, a)
    };
    m.ni = function() {};
    var Yaa = function(a, b) {
        m.l(a, b, void 0)
    };
    var oi = function(a) {
        return "CSS1Compat" == a.compatMode
    };
    m.pi = function(a, b) {
        this.width = a;
        this.height = b
    };
    m.qi = function(a, b, c, d) {
        if (!b&&!c)
            return null;
        var e = b ? b.toUpperCase(): null;
        return m.Zb(a, function(a) {
            return (!e || a.nodeName == e) && (!c || m.ja(a.className) && m.La(a.className.split(/\s+/), c))
        }, !0, d)
    };
    m.ri = function(a) {
        return a && a.parentNode ? a.parentNode.removeChild(a) : null
    };
    var si = function(a) {
        return !m.Qe && oi(a) ? a.documentElement : a.body || a.documentElement
    };
    var ti = function(a) {
        a = a.document;
        a = oi(a) ? a.documentElement : a.body;
        return new m.pi(a.clientWidth, a.clientHeight)
    };
    m.ui = function(a, b) {
        this.x = m.ca(a) ? a : 0;
        this.y = m.ca(b) ? b : 0
    };
    m.L = function(a, b, c) {
        return m.qi(a, null, b, c)
    };
    m.vi = function(a, b) {
        if (a.contains && 1 == b.nodeType)
            return a == b || a.contains(b);
        if ("undefined" != typeof a.compareDocumentPosition)
            return a == b || Boolean(a.compareDocumentPosition(b) & 16);
        for (; b && a != b;)
            b = b.parentNode;
        return b == a
    };
    m.wi = function(a) {
        return m.uaa && void 0 != a.children ? a.children : (0, m.qb)(a.childNodes, function(a) {
            return 1 == a.nodeType
        })
    };
    m.xi = function(a) {
        var b, c = a.parentNode;
        if (c && 11 != c.nodeType) {
            if (a.removeNode)
                return a.removeNode(!1);
            for (; b = a.firstChild;)
                c.insertBefore(b, a);
            return m.ri(a)
        }
    };
    m.yi = function(a, b) {
        b.parentNode && b.parentNode.insertBefore(a, b)
    };
    m.zi = function(a, b) {
        m.Sb(m.Yb(a), a, arguments, 1)
    };
    m.Ai = function(a) {
        var b = si(a);
        a = a.parentWindow || a.defaultView;
        return m.E && m.Fb("10") && a.pageYOffset != b.scrollTop ? new m.ui(b.scrollLeft, b.scrollTop) : new m.ui(a.pageXOffset || b.scrollLeft, a.pageYOffset || b.scrollTop)
    };
    m.Bi = function(a) {
        return ti(a || window)
    };
    m.M = function(a, b) {
        var c = b || window.document;
        return c.querySelectorAll && c.querySelector ? c.querySelectorAll("." + a) : m.Mb("*", a, b)
    };
    m.Ci = function(a, b) {
        var c = m.Yb(a);
        return c.defaultView && c.defaultView.getComputedStyle && (c = c.defaultView.getComputedStyle(a, null)) ? c[b] || c.getPropertyValue(b) || "" : ""
    };
    m.Di = function(a) {
        this.g = a || m.da.document || window.document
    };
    m.Ei = function(a) {
        return oi(a.g)
    };
    m.Fi = function(a) {
        a = a.g;
        return a.parentWindow || a.defaultView
    };
    m.Gi = function(a) {
        return m.Ai(a.g)
    };
    m.Hi = function(a, b) {
        return m.Ci(a, b) || (a.currentStyle ? a.currentStyle[b] : null) || a.style && a.style[b]
    };
    m.Ii = function(a) {
        return a ? new m.Di(m.Yb(a)) : Ji || (Ji = new m.Di)
    };
    m.Ki = function(a) {
        var b = [], c = 0, d;
        for (d in a)
            b[c++] = a[d];
        return b
    };
    var Li = function(a, b) {
        for (var c = 1; c < arguments.length; c++) {
            var d = arguments[c];
            if (m.ia(d)) {
                var e = a.length || 0, f = d.length || 0;
                a.length = e + f;
                for (var h = 0; h < f; h++)
                    a[e + h] = d[h]
            } else 
                a.push(d)
        }
    };
    m.Mi = function(a) {
        return (0, window.decodeURIComponent)(a.replace(/\+/g, " "))
    };
    var Ni = function(a, b) {
        if (/^\d+px?$/.test(b))
            return (0, window.parseInt)(b, 10);
        var c = a.style.left, d = a.runtimeStyle.left;
        a.runtimeStyle.left = a.currentStyle.left;
        a.style.left = b;
        var e = a.style.pixelLeft;
        a.style.left = c;
        a.runtimeStyle.left = d;
        return e
    };
    var Oi = function(a) {
        if (m.E&&!m.Ib(8))
            return a.offsetParent;
        var b = m.Yb(a), c = m.Hi(a, "position"), d = "fixed" == c || "absolute" == c;
        for (a = a.parentNode; a && a != b; a = a.parentNode)
            if (c = m.Hi(a, "position"), d = d && "static" == c && a != b.documentElement && a != b.body, !d && (a.scrollWidth > a.clientWidth || a.scrollHeight > a.clientHeight || "fixed" == c || "absolute" == c || "relative" == c))
                return a;
        return null
    };
    var Pi = function(a) {
        var b;
        try {
            b = a.getBoundingClientRect()
        } catch (c) {
            return {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0
            }
        }
        m.E && a.ownerDocument.body && (a = a.ownerDocument, b.left -= a.documentElement.clientLeft + a.body.clientLeft, b.top -= a.documentElement.clientTop + a.body.clientTop);
        return b
    };
    m.Qi = function(a) {
        return m.Hi(a, "position")
    };
    m.Ri = function(a) {
        if ("function" == typeof a.Sa)
            return a.Sa();
        if ("function" != typeof a.za) {
            if (m.ia(a) || m.ja(a)) {
                var b = [];
                a = a.length;
                for (var c = 0; c < a; c++)
                    b.push(c);
                return b
            }
            return m.ub(a)
        }
    };
    var Si = function(a) {
        if ("function" == typeof a.za)
            return a.za();
        if (m.ja(a))
            return a.split("");
        if (m.ia(a)) {
            for (var b = [], c = a.length, d = 0; d < c; d++)
                b.push(a[d]);
            return b
        }
        return m.Ki(a)
    };
    var Ti = function(a, b) {
        return Object.prototype.hasOwnProperty.call(a, b)
    };
    var Zaa = function(a, b) {
        return a === b
    };
    var Ui = function(a, b) {
        if ("none" == (a.currentStyle ? a.currentStyle[b + "Style"] : null))
            return 0;
        var c = a.currentStyle ? a.currentStyle[b + "Width"]: null;
        return c in Vi ? Vi[c] : Ni(a, c)
    };
    var Wi = function(a, b) {
        var c = a.currentStyle ? a.currentStyle[b]: null;
        return c ? Ni(a, c) : 0
    };
    m.Xi = function(a) {
        var b, c = m.Yb(a), d = m.Hi(a, "position"), e = m.Re && c.getBoxObjectFor&&!a.getBoundingClientRect && "absolute" == d && (b = c.getBoxObjectFor(a)) && (0 > b.screenX || 0 > b.screenY), f = new m.ui(0, 0), h;
        b = c ? m.Yb(c) : window.document;
        h=!m.E || m.Ib(9) || m.Ei(m.Ii(b)) ? b.documentElement : b.body;
        if (a == h)
            return f;
        if (a.getBoundingClientRect)
            b = Pi(a), a = m.Gi(m.Ii(c)), f.x = b.left + a.x, f.y = b.top + a.y;
        else if (c.getBoxObjectFor&&!e)
            b = c.getBoxObjectFor(a), a = c.getBoxObjectFor(h), f.x = b.screenX - a.screenX, f.y = b.screenY - a.screenY;
        else {
            b =
            a;
            do {
                f.x += b.offsetLeft;
                f.y += b.offsetTop;
                b != a && (f.x += b.clientLeft || 0, f.y += b.clientTop || 0);
                if (m.Qe && "fixed" == m.Qi(b)) {
                    f.x += c.body.scrollLeft;
                    f.y += c.body.scrollTop;
                    break
                }
                b = b.offsetParent
            }
            while (b && b != a);
            if (m.Se || m.Qe && "absolute" == d)
                f.y -= c.body.offsetTop;
            for (b = a; (b = Oi(b)) && b != c.body && b != h;)
                f.x -= b.scrollLeft, m.Se && "TR" == b.tagName || (f.y -= b.scrollTop)
        }
        return f
    };
    m.Yi = function(a) {
        "?" == a.charAt(0) && (a = a.substr(1));
        a = a.split("&");
        for (var b = {}, c = 0, d = a.length; c < d; c++) {
            var e = a[c].split("=");
            if (1 == e.length && e[0] || 2 == e.length) {
                var f = m.Mi(e[0] || ""), e = m.Mi(e[1] || "");
                f in b ? m.ha(b[f]) ? Li(b[f], e) : b[f] = [b[f], e] : b[f] = e
            }
        }
        return b
    };
    var $aa = function(a) {
        a = a.charCodeAt(0);
        return "%" + (a>>4 & 15).toString(16) + (a & 15).toString(16)
    };
    var Zi = function(a, b) {
        return a ? b ? (0, window.decodeURI)(a) : (0, window.decodeURIComponent)(a) : ""
    };
    m.$i = function(a, b, c) {
        if ("function" == typeof a.forEach)
            a.forEach(b, c);
        else if (m.ia(a) || m.ja(a))(0, m.y)(a, b, c);
        else 
            for (var d = m.Ri(a), e = Si(a), f = e.length, h = 0; h < f; h++)
                b.call(c, e[h], d && d[h], a)
    };
    m.aj = function(a, b) {
        this.j = {};
        this.g = [];
        this.Kb = this.k = 0;
        var c = arguments.length;
        if (1 < c) {
            if (c%2)
                throw Error("Uneven number of arguments");
            for (var d = 0; d < c; d += 2)
                this.set(arguments[d], arguments[d + 1])
        } else 
            a && m.bj(this, a)
    };
    m.cj = function(a, b) {
        return Ti(a.j, b)
    };
    var dj = function(a) {
        if (a.k != a.g.length) {
            for (var b = 0, c = 0; b < a.g.length;) {
                var d = a.g[b];
                Ti(a.j, d) && (a.g[c++] = d);
                b++
            }
            a.g.length = c
        }
        if (a.k != a.g.length) {
            for (var e = {}, c = b = 0; b < a.g.length;)
                d = a.g[b], Ti(e, d) || (a.g[c++] = d, e[d] = 1), b++;
            a.g.length = c
        }
    };
    m.bj = function(a, b) {
        var c, d;
        b instanceof m.aj ? (c = b.Sa(), d = b.za()) : (c = m.ub(b), d = m.Ki(b));
        for (var e = 0; e < c.length; e++)
            a.set(c[e], d[e])
    };
    m.ej = function(a) {
        dj(a);
        for (var b = {}, c = 0; c < a.g.length; c++) {
            var d = a.g[c];
            b[d] = a.j[d]
        }
        return b
    };
    m.fj = function(a) {
        return m.Qa.concat.apply(m.Qa, arguments)
    };
    m.gj = function(a) {
        return (a = m.F(a))?!("none" == a.style.display || m.A(a, "hid")) : !1
    };
    m.hj = function(a) {
        if (m.E&&!m.Ib(9)) {
            var b = Ui(a, "borderLeft"), c = Ui(a, "borderRight"), d = Ui(a, "borderTop");
            a = Ui(a, "borderBottom");
            return new m.Ve(d, c, a, b)
        }
        b = m.Ci(a, "borderLeftWidth");
        c = m.Ci(a, "borderRightWidth");
        d = m.Ci(a, "borderTopWidth");
        a = m.Ci(a, "borderBottomWidth");
        return new m.Ve((0, window.parseFloat)(d), (0, window.parseFloat)(c), (0, window.parseFloat)(a), (0, window.parseFloat)(b))
    };
    m.ij = function(a, b) {
        if (m.E) {
            var c = Wi(a, b + "Left"), d = Wi(a, b + "Right"), e = Wi(a, b + "Top"), f = Wi(a, b + "Bottom");
            return new m.Ve(e, d, f, c)
        }
        c = m.Ci(a, b + "Left");
        d = m.Ci(a, b + "Right");
        e = m.Ci(a, b + "Top");
        f = m.Ci(a, b + "Bottom");
        return new m.Ve((0, window.parseFloat)(e), (0, window.parseFloat)(d), (0, window.parseFloat)(f), (0, window.parseFloat)(c))
    };
    m.jj = function(a) {
        return "rtl" == m.Hi(a, "direction")
    };
    var aba = function(a) {
        var b = a.offsetWidth, c = a.offsetHeight, d = m.Qe&&!b&&!c;
        return m.ca(b)&&!d ||!a.getBoundingClientRect ? new m.pi(b, c) : (a = Pi(a), new m.pi(a.right - a.left, a.bottom - a.top))
    };
    var kj = function(a, b) {
        if ("none" != m.Hi(b, "display"))
            return a(b);
        var c = b.style, d = c.display, e = c.visibility, f = c.position;
        c.visibility = "hidden";
        c.position = "absolute";
        c.display = "inline";
        var h = a(b);
        c.display = d;
        c.position = f;
        c.visibility = e;
        return h
    };
    var lj = function(a) {
        var b;
        if (a.getBoundingClientRect)
            b = Pi(a), b = new m.ui(b.left, b.top);
        else {
            b = m.Gi(m.Ii(a));
            var c = m.Xi(a);
            b = new m.ui(c.x - b.x, c.y - b.y)
        }
        if (m.Re&&!m.Fb(12)) {
            i:
            {
                c = m.Ha("transform");
                if (void 0 === a.style[c] && (c = m.Pe() + m.Ia(c), void 0 !== a.style[c])) {
                    c = m.Te() + "-transform";
                    break i
                }
                c = "transform"
            }
            a = (a = m.Hi(a, c) || m.Hi(a, "transform")) ? (a = a.match(bba)) ? new m.ui((0, window.parseFloat)(a[1]), (0, window.parseFloat)(a[2])) : new m.ui(0, 0) : new m.ui(0, 0);
            a = new m.ui(b.x + a.x, b.y + a.y)
        } else 
            a = b;
        return a
    };
    m.mj = function(a, b) {
        if (b && a in b)
            return a;
        var c = m.Pe();
        return c ? (c = c.toLowerCase(), c += m.Ia(a), !m.ca(b) || c in b ? c : null) : null
    };
    m.nj = function(a) {
        return 0 <= m.Ea(m.Gaa, a)
    };
    m.oj = function(a, b) {
        b || (b = {});
        var c = window, d = "undefined" != typeof a.href ? a.href: String(a), e = b.target || a.target, f = [], h;
        for (h in b)
            switch (h) {
            case "width":
            case "height":
            case "top":
            case "left":
                f.push(h + "=" + b[h]);
                break;
            case "target":
            case "noreferrer":
                break;
            default:
                f.push(h + "=" + (b[h] ? 1 : 0))
            }
        f = f.join(",");
        if (b.noreferrer) {
            if (c = c.open("", e, f))
                m.E&&-1 != d.indexOf(";") && (d = "'" + d.replace(/'/g, "%27") + "'"), c.opener = null, d = m.wa(d), c.document.write('<META HTTP-EQUIV="refresh" content="0; url=' + d + '">'), c.document.close()
        } else 
            c =
            c.open(d, e, f);
        return c
    };
    var pj = function(a, b, c, d) {
        var e = a || window.document;
        return m.J(e, b, function(a) {
            var b = m.Zb(a.target, function(a) {
                return a === e || d(a)
            }, !0);
            b && b !== e&&!b.disabled && (a.currentTarget = b, c.call(b, a))
        })
    };
    var cba = function(a, b) {
        if (m.u("EVENTS_TRACKER_INSTALLED")) {
            var c = qj[a];
            if (!c) {
                var d = window._gaq._getAsyncTracker("eventsPageTracker");
                if (!d)
                    return;
                window._gaq.push(function() {
                    c = d._createEventTracker(a);
                    qj[a] = c
                })
            }
            window._gaq.push(function() {
                c._trackEvent(b, void 0, void 0)
            })
        }
    };
    m.rj = function(a, b, c) {
        this.g = a || null;
        this.j=!!c
    };
    var sj = function(a) {
        if (!a.Ka && (a.Ka = new m.aj, a.yb = 0, a.g))
            for (var b = a.g.split("&"), c = 0; c < b.length; c++) {
                var d = b[c].indexOf("="), e = null, f = null;
                0 <= d ? (e = b[c].substring(0, d), f = b[c].substring(d + 1)) : e = b[c];
                e = m.Mi(e);
                e = tj(a, e);
                a.add(e, f ? m.Mi(f) : "")
            }
    };
    var uj = function(a, b) {
        sj(a);
        b = tj(a, b);
        return m.cj(a.Ka, b)
    };
    var vj = function(a, b, c) {
        a.remove(b);
        0 < c.length && (a.g = null, a.Ka.set(tj(a, b), m.Ra(c)), a.yb += c.length)
    };
    var tj = function(a, b) {
        var c = String(b);
        a.j && (c = c.toLowerCase());
        return c
    };
    var dba = function(a, b) {
        b&&!a.j && (sj(a), a.g = null, a.Ka.forEach(function(a, b) {
            var e = b.toLowerCase();
            b != e && (this.remove(b), vj(this, e, a))
        }, a));
        a.j = b
    };
    var wj = function(a, b, c) {
        return m.ja(a) ? (a = (0, window.encodeURI)(a).replace(b, $aa), c && (a = a.replace(/%25([0-9a-fA-F]{2})/g, "%$1")), a) : null
    };
    var xj = function(a) {
        a = a.tabIndex;
        return m.ka(a) && 0 <= a && 32768 > a
    };
    var yj = function(a) {
        a = a.getAttributeNode("tabindex");
        return null != a && a.specified
    };
    m.zj = function(a, b) {
        for (; a && 1 != a.nodeType;)
            a = b ? a.nextSibling : a.previousSibling;
        return a
    };
    m.Aj = function(a) {
        var b = window.document, c = b.createElement("div");
        m.E ? (c.innerHTML = "<br>" + a, c.removeChild(c.firstChild)) : c.innerHTML = a;
        if (1 == c.childNodes.length)
            return c.removeChild(c.firstChild);
        for (a = b.createDocumentFragment(); c.firstChild;)
            a.appendChild(c.firstChild);
        return a
    };
    m.Bj = function(a) {
        return a ? a.parentWindow || a.defaultView : window
    };
    var eba = function(a) {
        return m.Mb("IMG", null, a)
    };
    var Cj = function(a, b) {
        for (var c in a)
            if (a[c] == b)
                return !0;
        return !1
    };
    var Dj = function(a) {
        var b = 0, c;
        for (c in a)
            b++;
        return b
    };
    m.Ej = function(a, b) {
        return a > b ? 1 : a < b?-1 : 0
    };
    var Fj = function(a, b, c, d) {
        return m.Qa.splice.apply(a, m.Sa(arguments, 1))
    };
    m.Gj = function() {
        return Math.floor(2147483648 * Math.random()).toString(36) + Math.abs(Math.floor(2147483648 * Math.random())^(0, m.H)()).toString(36)
    };
    var fba = function() {
        var a = m.F("movie_player");
        if (a && a.currentAdInformation) {
            var b = {};
            try {
                b = a.currentAdInformation()
            } catch (c) {}
            a = b;
            a.adIds && (m.yf.ad_ids = a.adIds.join());
            a.adSystems && (m.yf.ad_systems = a.adSystems.join())
        }
    };
    var gba = function(a) {
        a.preventDefault();
        var b = m.I(a.currentTarget, "ghelp-anchor") || a.currentTarget, c = m.I(a.currentTarget, "ghelp-tracking-param") || "", b = window.document.getElementById(b), d = a.currentTarget;
        a = m.u("GOOGLE_HELP_CONTEXT");
        var e = m.u("GOOGLE_HELP_PRODUCT_ID"), f=!!m.I(d, "ypc-load-chat-support"), h=!!m.I(d, "ypc-hide-feedback-link"), k = m.I(d, "ypc-product-data-content-type"), r=!!m.I(d, "ypc-product-data-has-ypc-offer"), w=!!m.I(d, "ypc-product-data-active-entitlement"), B = m.I(d, "ypc-product-data-order-id"),
        d=!!m.I(d, "ypc-product-data-c2c");
        m.Bf(b, e, a, {
            "content-type": k,
            has_ypc_offer: r,
            active_entitlement: w,
            order_id: B,
            c2c: d
        }, f, h, c)
    };
    var Hj = function(a) {
        if (a in Ij)
            return Ij[a];
        var b = m.mj(a, window.document.body.style);
        return Ij[a] = b
    };
    var hba = function(a) {
        if (a = m.F(a))
            m.gj(a) ? (m.Ze(a, !1), m.C(a, "hid")) : (m.Ze(a, !0), m.D(a, "hid"))
    };
    m.Jj = function(a) {
        return kj(aba, a)
    };
    var Kj = function(a) {
        if (1 == a.nodeType)
            return lj(a);
        var b = m.ma(a.Tn), c = a;
        a.targetTouches && a.targetTouches.length ? c = a.targetTouches[0] : b && a.wa.targetTouches && a.wa.targetTouches.length && (c = a.wa.targetTouches[0]);
        return new m.ui(c.clientX, c.clientY)
    };
    m.Lj = function(a, b, c, d) {
        this.left = a;
        this.top = b;
        this.width = c;
        this.height = d
    };
    var Mj = function(a) {
        var b = typeof a;
        return "object" == b && a || "function" == b ? "o" + m.oa(a) : b.substr(0, 1) + a
    };
    m.Nj = function(a, b) {
        var c = b || {};
        c.target = c.target || a.target || "YouTube";
        c.width = c.width || 600;
        c.height = c.height || 600;
        c = m.oj(a, c);
        if (!c)
            return null;
        c.opener || (c.opener = window);
        c.focus();
        return c
    };
    m.N = function(a) {
        a && ("string" == typeof a && (a = [a]), (0, m.y)(a, function(a) {
            if (a in m.Od) {
                var c = m.Od[a], d = c[0], e = c[1], f = c[3], c = c[4];
                d.removeEventListener ? d.removeEventListener(e, f, c) : d.detachEvent && d.detachEvent("on" + e, f);
                delete m.Od[a]
            }
        }))
    };
    m.O = function(a, b, c, d) {
        return pj(a, b, c, function(a) {
            return m.A(a, d)
        })
    };
    m.Oj = function(a, b, c) {
        a = m.Mb(a, b, c);
        return a.length ? a[0] : null
    };
    m.Pj = function(a, b, c) {
        cba(a, b || "null");
        m.Bd("a=" + a + (b ? "&" + b : "").replace(/\//g, "&"), c)
    };
    var Qj = function(a) {
        return a && "status"in a ? a.status : - 1
    };
    m.Rj = function(a, b) {
        var c;
        a instanceof m.Rj ? (this.ge = m.ca(b) ? b : a.ge, Sj(this, a.xd), this.fe = a.fe, m.Tj(this, a.Tb), Uj(this, a.ee), m.Vj(this, a.Mc), m.Wj(this, a.g.clone()), m.Xj(this, a.ig)) : a && (c = m.hc(String(a))) ? (this.ge=!!b, Sj(this, c[1] || "", !0), this.fe = Zi(c[2] || ""), m.Tj(this, c[3] || "", !0), Uj(this, c[4]), m.Vj(this, c[5] || "", !0), m.Wj(this, c[6] || "", !0), m.Xj(this, c[7] || "", !0)) : (this.ge=!!b, this.g = new m.rj(null, 0, this.ge))
    };
    var Sj = function(a, b, c) {
        a.xd = c ? Zi(b, !0) : b;
        a.xd && (a.xd = a.xd.replace(/:$/, ""))
    };
    m.Tj = function(a, b, c) {
        a.Tb = c ? Zi(b, !0) : b
    };
    var Uj = function(a, b) {
        if (b) {
            b = Number(b);
            if ((0, window.isNaN)(b) || 0 > b)
                throw Error("Bad port number " + b);
            a.ee = b
        } else 
            a.ee = null
    };
    m.Vj = function(a, b, c) {
        a.Mc = c ? Zi(b, !0) : b;
        return a
    };
    m.Wj = function(a, b, c) {
        b instanceof m.rj ? (a.g = b, dba(a.g, a.ge)) : (c || (b = wj(b, iba)), a.g = new m.rj(b, 0, a.ge));
        return a
    };
    m.Yj = function(a, b, c) {
        a.g.set(b, c);
        return a
    };
    m.Zj = function(a, b, c) {
        m.ha(c) || (c = [String(c)]);
        vj(a.g, b, c)
    };
    m.ak = function(a, b) {
        return a.g.get(b)
    };
    m.Xj = function(a, b, c) {
        a.ig = c ? Zi(b) : b;
        return a
    };
    var bk = function(a) {
        m.Yj(a, "zx", m.Gj());
        return a
    };
    var ck = function(a, b, c) {
        Math.max(b.length - (c || 0), 0);
        for (c = c || 0; c < b.length; c += 2)
            m.oc(b[c], b[c + 1], a);
        return a
    };
    var jba = function(a, b) {
        if ("function" == typeof a.every)
            return a.every(b, void 0);
        if (m.ia(a) || m.ja(a))
            return (0, m.vh)(a, b, void 0);
        for (var c = m.Ri(a), d = Si(a), e = d.length, f = 0; f < e; f++)
            if (!b.call(void 0, d[f], c && c[f], a))
                return !1;
        return !0
    };
    m.dk = function(a, b) {
        return "function" == typeof a.contains ? a.contains(b) : "function" == typeof a.rf ? a.rf(b) : m.ia(a) || m.ja(a) ? m.La(a, b) : Cj(a, b)
    };
    var ek = function(a) {
        return "function" == typeof a.ua ? a.ua() : m.ia(a) || m.ja(a) ? a.length : Dj(a)
    };
    m.fk = function(a, b, c) {
        if (!(a.nodeName in kba))
            if (3 == a.nodeType)
                c ? b.push(String(a.nodeValue).replace(/(\r\n|\r|\n)/g, "")) : b.push(a.nodeValue);
            else if (a.nodeName in gk)
                b.push(gk[a.nodeName]);
            else 
                for (a = a.firstChild; a;)
                    m.fk(a, b, c), a = a.nextSibling
    };
    var hk = function(a, b, c, d) {
        if (null != a)
            for (a = a.firstChild; a;) {
                if (b(a) && (c.push(a), d) || hk(a, b, c, d))
                    return !0;
                    a = a.nextSibling
            }
        return !1
    };
    m.ik = function(a) {
        return m.na(a) && 1 == a.nodeType
    };
    m.jk = function(a) {
        return void 0 != a.firstElementChild ? a.firstElementChild : m.zj(a.firstChild, !0)
    };
    m.kk = function(a) {
        for (var b in a)
            return !1;
        return !0
    };
    var lk = function(a, b) {
        return new m.ui(a.x - b.x, a.y - b.y)
    };
    m.mk = function(a) {
        m.Je ? window[a] = void 0 : delete window[a]
    };
    var lba = function(a, b) {
        var c = m.u("I18N_PLURAL_RULES") || function(a) {
            return 1 == a ? "one" : "other"
        };
        return (c = a["case" + b] || a[c(b)]) ? c.replace("#", b.toString()) : b + ""
    };
    var mba = function(a, b) {
        return a === b
    };
    m.nk = function(a, b, c) {
        for (var d = m.ja(a) ? a.split("") : a, e = a.length - 1; 0 <= e; e--)
            if (e in d && b.call(c, d[e], e, a))
                return e;
        return - 1
    };
    m.ok = function(a) {
        return null == a ? "" : String(a)
    };
    m.pk = function(a, b) {
        return Array(b + 1).join(a)
    };
    var nba = function(a) {
        return a.replace(/&([^;]+);/g, function(a, c) {
            switch (c) {
            case "amp":
                return "&";
            case "lt":
                return "<";
            case "gt":
                return ">";
            case "quot":
                return '"';
            default:
                if ("#" == c.charAt(0)) {
                    var d = Number("0" + c.substr(1));
                    if (!(0, window.isNaN)(d))
                        return String.fromCharCode(d)
                    }
                return a
            }
        })
    };
    var oba = function(a) {
        var b = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"'
        }, c;
        c = m.da.document.createElement("div");
        return a.replace(pba, function(a, e) {
            var f = b[a];
            if (f)
                return f;
            if ("#" == e.charAt(0)) {
                var h = Number("0" + e.substr(1));
                (0, window.isNaN)(h) || (f = String.fromCharCode(h))
            }
            f || (c.innerHTML = a + " ", f = c.firstChild.nodeValue.slice(0, - 1));
            return b[a] = f
        })
    };
    var qk = function(a) {
        return a.replace(/^[\s\xa0]+/, "")
    };
    m.rk = function(a) {
        return /^[\s\xa0]*$/.test(a)
    };
    var sk = function(a) {
        a = String(a.substr(0, 3)).toLowerCase();
        return 0 == ("<tr" < a?-1 : "<tr" == a ? 0 : 1)
    };
    var tk = function(a) {
        m.qg && m.qg.remove(a);
        m.pg && m.pg.remove(a)
    };
    m.uk = function(a) {
        return eval("(" + a + ")")
    };
    var qba = function() {
        if (!(m.E ? m.nj("7") && 0 <= m.Ea(m.Kh, "9") : m.Th ? m.nj("3.6") : m.Xh ? m.nj("5") : m.Wh)) {
            var a = m.M("reportbug");
            (0, m.y)(a, function(a) {
                m.wf(a, !1)
            })
        }
    };
    var rba = function(a) {
        if (!(0, m.Zh)())
            return null;
        var b = Hj("transitionProperty"), b = m.Ci(a, b), c = Hj("transitionDuration");
        a = m.Ci(a, c);
        if (!b ||!a)
            return null;
        var d = {}, e = b.split(",");
        (0, m.y)(a.split(","), function(a, b) {
            var c = e[b].trim();
            if (c) {
                var r;
                r =- 1 < a.indexOf("ms") ? (0, window.parseInt)(a, 10) : Math.round(1E3 * (0, window.parseFloat)(a));
                d[c] = r
            }
        });
        return d
    };
    var vk = function(a, b, c) {
        (a = m.F(a)) && a.style && (b = Hj(b)) && (a.style[b] = c)
    };
    m.wk = function(a) {
        (0, m.y)(arguments, hba)
    };
    m.P = function(a) {
        (0, m.y)(arguments, function(a) {
            m.wf(a, !1)
        })
    };
    var xk = function(a) {
        a = m.F(a);
        if (!a)
            return null;
        var b = 0, c = 0;
        if (a.offsetParent) {
            do 
                b += a.offsetLeft, c += a.offsetTop;
            while (a = a.offsetParent)
            }
        return new m.ui(b, c)
    };
    m.yk = function(a, b) {
        var c = a.style;
        "opacity"in c ? c.opacity = b : "MozOpacity"in c ? c.MozOpacity = b : "filter"in c && (c.filter = "" === b ? "" : "alpha(opacity=" + 100 * b + ")")
    };
    m.zk = function(a) {
        var b = m.Xi(a);
        a = m.Jj(a);
        return new m.Lj(b.x, b.y, a.width, a.height)
    };
    var sba = function(a) {
        if (!a.getBoundingClientRect)
            return null;
        a = kj(Pi, a);
        return new m.pi(a.right - a.left, a.bottom - a.top)
    };
    m.Ak = function(a, b, c) {
        if (b instanceof m.pi)
            c = b.height, b = b.width;
        else if (void 0 == c)
            throw Error("missing height argument");
        a.style.width = m.Ye(b, !0);
        a.style.height = m.Ye(c, !0)
    };
    var tba = function(a, b) {
        var c = Kj(a), d = Kj(b);
        return new m.ui(c.x - d.x, c.y - d.y)
    };
    var Bk = function(a) {
        return m.Xi(a).x
    };
    var Ck = function(a) {
        for (var b = new m.Ve(0, window.Infinity, window.Infinity, 0), c = m.Ii(a), d = c.g.body, e = c.g.documentElement, f = si(c.g); a = Oi(a);)
            if (!(m.E && 0 == a.clientWidth || m.Qe && 0 == a.clientHeight && a == d) && a != d && a != e && "visible" != m.Hi(a, "overflow")) {
                var h = m.Xi(a), k;
                k = a;
                if (m.Re&&!m.Fb("1.9")) {
                    var r = (0, window.parseFloat)(m.Ci(k, "borderLeftWidth"));
                    if (m.jj(k))
                        var w = k.offsetWidth - k.clientWidth - r - (0, window.parseFloat)(m.Ci(k, "borderRightWidth")), r = r + w;
                        k = new m.ui(r, (0, window.parseFloat)(m.Ci(k, "borderTopWidth")))
                    } else 
                        k =
                        new m.ui(k.clientLeft, k.clientTop);
                        h.x += k.x;
                        h.y += k.y;
                        b.top = Math.max(b.top, h.y);
                        b.right = Math.min(b.right, h.x + a.clientWidth);
                        b.bottom = Math.min(b.bottom, h.y + a.clientHeight);
                        b.left = Math.max(b.left, h.x)
            }
        d = f.scrollLeft;
        f = f.scrollTop;
        b.left = Math.max(b.left, d);
        b.top = Math.max(b.top, f);
        c = m.Bi(m.Fi(c));
        b.right = Math.min(b.right, d + c.width);
        b.bottom = Math.min(b.bottom, f + c.height);
        return 0 <= b.top && 0 <= b.left && b.bottom > b.top && b.right > b.left ? b : null
    };
    m.Dk = function(a, b, c) {
        var d, e = m.Re && (m.Oe || m.qaa) && m.Fb("1.9");
        b instanceof m.ui ? (d = b.x, b = b.y) : (d = b, b = c);
        a.style.left = m.Ye(d, e);
        a.style.top = m.Ye(b, e)
    };
    m.Ek = function(a, b, c) {
        if (m.ja(b))(b = m.Xe(a, b)) && (a.style[b] = c);
        else 
            for (var d in b) {
                c = a;
                var e = b[d], f = m.Xe(c, d);
                f && (c.style[f] = e)
            }
        };
    var uba = function(a, b) {
        return a.left <= b.right && b.left <= a.right && a.top <= b.bottom && b.top <= a.bottom
    };
    m.Fk = function(a) {
        return new m.pe(function(b, c) {
            var d = a.length, e = [];
            if (d)
                for (var f = function(a, c) {
                    d--;
                    e[a] = c;
                    0 == d && b(e)
                }, h = function(a) {
                    c(a)
                }, k = 0, r; r = a[k]; k++)
                    r.then(m.q(f, k), h);
            else 
                b(e)
        })
    };
    var Gk = function() {
        return new m.pe(function(a, b) {
            b(void 0)
        })
    };
    var Hk = function() {};
    m.Ik = function(a) {
        this.g = new m.aj;
        if (a) {
            a = Si(a);
            for (var b = a.length, c = 0; c < b; c++)
                this.add(a[c])
        }
    };
    var vba = function(a, b) {
        var c = ek(b);
        if (a.ua() > c)
            return !1;
        !(b instanceof m.Ik) && 5 < c && (b = new m.Ik(b));
        return jba(a, function(a) {
            return m.dk(b, a)
        })
    };
    var Jk = function(a, b) {
        return !m.Nj(a, b)
    };
    m.Kk = function(a) {
        m.Ud.remove("" + a, "/", "youtube.com")
    };
    m.Lk = function(a, b) {
        if (window.document.createEvent) {
            var c = window.document.createEvent("HTMLEvents");
            c.initEvent(b, !0, !0);
            a.dispatchEvent(c)
        } else 
            c = window.document.createEventObject(), a.fireEvent("on" + b, c)
    };
    m.Mk = function(a) {
        a = a || window.event;
        a.returnValue=!1;
        a.preventDefault && a.preventDefault()
    };
    m.Nk = function(a) {
        a = a || window.event;
        a = a.target || a.srcElement;
        3 == a.nodeType && (a = a.parentNode);
        return a
    };
    m.Ok = function(a) {
        for (var b in m.Od)
            m.Od[b][0] == a && m.N(b)
    };
    m.Pk = function(a, b, c) {
        (a = m.Nd(a, b, c, !1)) && m.N(a)
    };
    m.Qk = function(a, b, c, d) {
        return pj(a, "click", b, function(a) {
            return a.nodeName.toLowerCase() === c.toLowerCase() && (!d || m.A(a, d))
        })
    };
    m.Rk = function(a, b, c) {
        var d;
        return d = m.J(a, b, function() {
            m.N(d);
            c.apply(a, arguments)
        }, void 0)
    };
    var wba = function() {
        m.rb(window.document.body, "hide-players", !1);
        var a = m.M("preserve-players");
        (0, m.y)(a, function(a) {
            m.D(a, "preserve-players")
        })
    };
    var Sk = function(a) {
        m.rb(window.document.body, "hide-players", !0);
        a && m.rb(a, "preserve-players", !0)
    };
    m.Tk = function() {
        var a = window.document, b;
        (0, m.uh)(["fullscreenElement", "fullScreenElement"], function(c) {
            c in a ? b = a[c] : (c = m.yaa + c.charAt(0).toUpperCase() + c.substr(1), b = c in a ? a[c] : void 0);
            return !!b
        });
        return b
    };
    m.Uk = function(a) {
        a = qk(a);
        if (sk(a))
            return a = m.Aj("<table><tbody>" + a + "</tbody></table>"), m.Oj("tr", null, a);
        var b = window.document.createElement("div");
        b.innerHTML = a;
        return m.jk(b)
    };
    m.Vk = function(a) {
        a = qk(a);
        var b = sk(a);
        b && (a = "<table>" + a + "</table>");
        a = m.Aj(a);
        var c = window.document.createDocumentFragment();
        if (b)
            return b = m.Mb("tr", null, a), (0, m.y)(b, function(a) {
                c.appendChild(a)
            }), c;
        c.appendChild(a);
        return c
    };
    m.Wk = function(a, b) {
        "disabled"in a && (a.disabled=!b);
        1 == a.nodeType && m.rb(a, "disabled", !b);
        if (a.hasChildNodes())
            for (var c = 0, d; d = a.childNodes[c]; ++c)
                m.Wk(d, b)
    };
    var Xk = function(a, b) {
        a = m.F(a);
        b = m.F(b);
        return !!m.Zb(a, function(a) {
            return a === b
        }, !0, void 0)
    };
    m.Yk = function(a) {
        a = m.Kd(m.F(a));
        a.removeAttribute("id");
        return a
    };
    m.Zk = function(a) {
        var b = a.__yt_uid_key;
        b || (b = (0, m.Eh)(), a.__yt_uid_key = b);
        return b
    };
    var $k = function(a, b, c, d, e, f) {
        var h = {};
        b && (h.v = b);
        c && (h.list = c);
        d && (h.url = d);
        a = {
            name: a,
            locale: e,
            feature: f
        };
        for (var k in h)
            a[k] = h[k];
        h = m.rc("/sharing_services", a);
        m.sc(h)
    };
    var xba = function() {
        var a = window.ytspf, a = a ? a.enabled: !1, b = {
            a: "navigation",
            enabled: a
        };
        a && m.tb(window._spf_state || {}, function(a, d) {
            switch (d) {
            case "nav-init":
            case "nav-counter":
            case "cache-counter":
                b[d] = a;
                break;
            case "cache-storage":
                b[d] = Dj(a);
                break;
            case "config":
                m.tb(a, function(a, c) {
                    - 1 != c.indexOf("callback") || (b["config." + c] = a)
                })
            }
        });
        m.Bd(m.qc(b))
    };
    m.al = function(a, b, c) {
        m.Pj(a, b, c)
    };
    m.bl = function(a, b) {
        var c = a.split("#", 2);
        a = c[0];
        var c = 1 < c.length ? "#" + c[1]: "", d = a.split("?", 2);
        a = d[0];
        var d = m.Yi(d[1] || ""), e;
        for (e in b)
            d[e] = b[e];
        return m.rc(a, d) + c
    };
    m.cl = function(a) {
        return - 1 != a.indexOf("?") ? (a = (a || "").split("#")[0], a = a.split("?", 2), m.Yi(1 < a.length ? a[1] : a[0])) : {}
    };
    m.dl = function(a) {
        var b = [];
        m.tb(a, function(a, d) {
            var e = m.va(d), f;
            m.ha(a) ? f = a : f = [a];
            (0, m.y)(f, function(a) {
                "" == a ? b.push(e) : b.push(e + "=" + m.va(a))
            })
        });
        return b.join("&")
    };
    m.el = function(a, b) {
        a && (a.dataset ? delete a.dataset[m.od(b)] : a.removeAttribute("data-" + b))
    };
    m.fl = function(a, b, c) {
        a && (a.dataset ? a.dataset[m.od(b)] = c : a.setAttribute("data-" + b, c))
    };
    m.gl = function(a) {
        return new m.Uc(a.name, a.deps, a.page, a.init, a.dispose)
    };
    m.hl = function(a) {
        return m.rk(m.ok(a))?!1 : - 1 != a.search(yba)||-1 != a.search(zba)
    };
    var Aba = function(a, b, c, d) {
        var e = new m.Rj(null, void 0);
        a && Sj(e, a);
        b && m.Tj(e, b);
        c && Uj(e, c);
        d && m.Vj(e, d);
        return e
    };
    m.il = function(a) {
        return a instanceof m.Rj ? a.clone() : new m.Rj(a, void 0)
    };
    m.jl = function(a, b) {
        return m.nc(2 == arguments.length ? ck([a], arguments[1], 0) : ck([a], arguments, 1))
    };
    m.kl = function(a) {
        a = m.hc(a);
        return m.gc(a[1], a[2], a[3], a[4])
    };
    m.ll = function(a) {
        var b;
        b = b || 0;
        return function() {
            return a.apply(this, Array.prototype.slice.call(arguments, 0, b))
        }
    };
    m.ml = function(a) {
        if (m.Ah && "innerText"in a)
            a = a.innerText.replace(/(\r\n|\r|\n)/g, "\n");
        else {
            var b = [];
            m.fk(a, b, !0);
            a = b.join("")
        }
        a = a.replace(/ \xAD /g, " ").replace(/\xAD/g, "");
        a = a.replace(/\u200B/g, "");
        m.Ah || (a = a.replace(/ +/g, " "));
        " " != a && (a = a.replace(/^\s*/, ""));
        return a
    };
    var nl = function(a) {
        var b;
        (b = "A" == a.tagName || "INPUT" == a.tagName || "TEXTAREA" == a.tagName || "SELECT" == a.tagName || "BUTTON" == a.tagName?!a.disabled && (!yj(a) || xj(a)) : yj(a) && xj(a)) && m.E ? (a = m.ma(a.getBoundingClientRect) ? a.getBoundingClientRect() : {
            height: a.offsetHeight,
            width: a.offsetWidth
        }, a = null != a && 0 < a.height && 0 < a.width) : a = b;
        return a
    };
    var ol = function(a, b) {
        var c = [];
        return hk(a, b, c, !0) ? c[0] : void 0
    };
    m.pl = function(a, b) {
        if ("textContent"in a)
            a.textContent = b;
        else if (3 == a.nodeType)
            a.data = b;
        else if (a.firstChild && 3 == a.firstChild.nodeType) {
            for (; a.lastChild != a.firstChild;)
                a.removeChild(a.lastChild);
            a.firstChild.data = b
        } else {
            m.Ub(a);
            var c = m.Yb(a);
            a.appendChild(c.createTextNode(String(b)))
        }
    };
    m.ql = function(a) {
        return a.contentDocument || a.contentWindow.document
    };
    m.rl = function(a) {
        var b;
        if (m.vaa&&!(m.E && m.Fb("9")&&!m.Fb("10") && m.da.SVGElement && a instanceof m.da.SVGElement) && (b = a.parentElement))
            return b;
        b = a.parentNode;
        return m.ik(b) ? b : null
    };
    m.sl = function(a) {
        return void 0 != a.previousElementSibling ? a.previousElementSibling : m.zj(a.previousSibling, !1)
    };
    m.tl = function(a) {
        return void 0 != a.nextElementSibling ? a.nextElementSibling : m.zj(a.nextSibling, !0)
    };
    m.ul = function(a, b) {
        var c = b.parentNode;
        c && c.replaceChild(a, b)
    };
    m.vl = function(a, b, c) {
        a.insertBefore(b, a.childNodes[c] || null)
    };
    m.wl = function(a, b) {
        b.parentNode && b.parentNode.insertBefore(a, b.nextSibling)
    };
    m.xl = function(a) {
        return window.document.createElement(a)
    };
    m.yl = function(a) {
        var b = {}, c;
        for (c in a)
            b[a[c]] = c;
        return b
    };
    var zl = function(a) {
        var b = m.ga(a);
        if ("object" == b || "array" == b) {
            if (a.clone)
                return a.clone();
            var b = "array" == b ? []: {}, c;
            for (c in a)
                b[c] = zl(a[c]);
            return b
        }
        return a
    };
    var Al = function(a, b) {
        return b in a ? a[b] : void 0
    };
    m.Bl = function(a, b, c) {
        var d = {}, e;
        for (e in a)
            b.call(c, a[e], e, a) && (d[e] = a[e]);
        return d
    };
    var Cl = function(a, b, c) {
        return a + c * (b - a)
    };
    m.Dl = function(a, b, c) {
        m.D(a, b);
        m.C(a, c)
    };
    m.El = function(a, b) {
        var c=!m.A(a, b);
        m.rb(a, b, c);
        return c
    };
    m.Fl = function(a, b, c) {
        m.A(a, b) && (m.D(a, b), m.C(a, c))
    };
    m.Gl = function(a, b) {
        a.classList ? (0, m.y)(b, function(b) {
            m.D(a, b)
        }) : a.className = (0, m.qb)(m.pb(a), function(a) {
            return !m.La(b, a)
        }).join(" ")
    };
    m.Hl = function(a, b) {
        if (a.classList)(0, m.y)(b, function(b) {
            m.C(a, b)
        });
        else {
            var c = {};
            (0, m.y)(m.pb(a), function(a) {
                c[a]=!0
            });
            (0, m.y)(b, function(a) {
                c[a]=!0
            });
            a.className = "";
            for (var d in c)
                a.className += 0 < a.className.length ? " " + d : d
        }
    };
    var Bba = function(a, b, c) {
        m.jb = a;
        var d = m.z.apply(null, Array.prototype.slice.call(arguments, 1));
        m.jb = null;
        return d
    };
    m.Il = function(a, b, c) {
        var d = m.x(a, function(a) {
            b.apply(c, arguments);
            m.nb(d)
        }, c)
    };
    m.Jl = function(a, b) {
        return lba(a in m.fb ? m.fb[a] : {}, b)
    };
    m.Kl = function() {
        return m.u("XSRF_TOKEN")
    };
    var Ll = function(a) {
        for (var b = [], c = 0; c < arguments.length; c++) {
            var d = arguments[c];
            if (m.ha(d))
                for (var e = 0; e < d.length; e += 8192)
                    for (var f = m.Sa(d, e, e + 8192), f = Ll.apply(null, f), h = 0; h < f.length; h++)
                        b.push(f[h]);
            else 
                b.push(d)
        }
        return b
    };
    m.Ml = function(a, b, c) {
        if (!m.ia(a) ||!m.ia(b) || a.length != b.length)
            return !1;
        var d = a.length;
        c = c || mba;
        for (var e = 0; e < d; e++)
            if (!c(a[e], b[e]))
                return !1;
        return !0
    };
    m.Nl = function(a, b) {
        a.sort(b || m.Ej)
    };
    var Ol = function(a, b) {
        var c = m.Ka(a, b, void 0);
        0 <= c && m.Pa(a, c)
    };
    m.Pl = function(a, b) {
        m.La(a, b) || a.push(b)
    };
    m.Ql = function(a) {
        if (!m.ha(a))
            for (var b = a.length - 1; 0 <= b; b--)
                delete a[b];
        a.length = 0
    };
    m.Rl = function(a) {
        return 0 == a.length
    };
    var Sl = function(a, b) {
        var c = m.nk(a, b, void 0);
        return 0 > c ? null : m.ja(a) ? a.charAt(c) : a[c]
    };
    m.Tl = function(a, b) {
        var c = m.ca(void 0) ? a.toFixed(void 0): String(a), d = c.indexOf(".");
        - 1 == d && (d = c.length);
        return m.pk("0", Math.max(0, b - d)) + c
    };
    m.Ul = function(a) {
        return - 1 != a.indexOf("&") ? "document"in m.da ? oba(a) : nba(a) : a
    };
    m.Vl = function(a, b) {
        for (var c = a.split("%s"), d = "", e = Array.prototype.slice.call(arguments, 1); e.length && 1 < c.length;)
            d += c.shift() + e.shift();
        return d + c.join("%s")
    };
    m.Wl = function(a, b) {
        var c = a.length - b.length;
        return 0 <= c && a.indexOf(b, c) == c
    };
    m.Xl = function(a, b) {
        try {
            return m.ta(a[b]), !0
        } catch (c) {}
        return !1
    };
    m.Yl = function(a, b) {
        b && (a = a.replace(/\{\$([^}]+)}/g, function(a, d) {
            return d in b ? b[d] : a
        }));
        return a
    };
    m.Zl = function(a, b) {
        for (var c in b)
            a[c] = b[c]
    };
    m.$l = function(a, b) {
        this.type = a;
        this.currentTarget = this.target = b;
        this.defaultPrevented = this.qe=!1;
        this.zn=!0
    };
    m.am = function(a, b) {
        m.$l.call(this, a ? a.type : "");
        this.relatedTarget = this.currentTarget = this.target = null;
        this.charCode = this.keyCode = this.button = this.screenY = this.screenX = this.clientY = this.clientX = this.offsetY = this.offsetX = 0;
        this.metaKey = this.shiftKey = this.altKey = this.ctrlKey=!1;
        this.wa = this.state = null;
        a && this.init(a, b)
    };
    var bm = function(a) {
        return !(!a ||!a[cm])
    };
    var Cba = function(a, b, c, d, e) {
        this.listener = a;
        this.proxy = null;
        this.src = b;
        this.type = c;
        this.yh=!!d;
        this.ic = e;
        this.key=++Dba;
        this.removed = this.zh=!1
    };
    var dm = function(a) {
        a.removed=!0;
        a.listener = null;
        a.proxy = null;
        a.src = null;
        a.ic = null
    };
    var em = function(a) {
        this.src = a;
        this.g = {};
        this.j = 0
    };
    var fm = function(a, b) {
        var c = b.type;
        if (!(c in a.g))
            return !1;
        var d = m.Oa(a.g[c], b);
        d && (dm(b), 0 == a.g[c].length && (delete a.g[c], a.j--));
        return d
    };
    var gm = function(a, b, c, d, e) {
        a = a.g[b.toString()];
        b =- 1;
        a && (b = hm(a, c, d, e));
        return - 1 < b ? a[b] : null
    };
    var hm = function(a, b, c, d) {
        for (var e = 0; e < a.length; ++e) {
            var f = a[e];
            if (!f.removed && f.listener == b && f.yh==!!c && f.ic == d)
                return e
        }
        return - 1
    };
    m.im = function(a, b, c, d, e) {
        if (m.ha(b)) {
            for (var f = 0; f < b.length; f++)
                m.im(a, b[f], c, d, e);
            return null
        }
        c = jm(c);
        return bm(a) ? a.listen(b, c, d, e) : km(a, b, c, !1, d, e)
    };
    var km = function(a, b, c, d, e, f) {
        if (!b)
            throw Error("Invalid event type");
        var h=!!e, k = lm(a);
        k || (a[mm] = k = new em(a));
        c = k.add(b, c, d, e, f);
        if (c.proxy)
            return c;
        d = Eba();
        c.proxy = d;
        d.src = a;
        d.listener = c;
        a.addEventListener ? a.addEventListener(b.toString(), d, h) : a.attachEvent(nm(b.toString()), d);
        om++;
        return c
    };
    var Eba = function() {
        var a = Fba, b = pm ? function(c) {
            return a.call(b.src, b.listener, c)
        }
        : function(c) {
            c = a.call(b.src, b.listener, c);
            if (!c)
                return c
        };
        return b
    };
    m.qm = function(a, b, c, d, e) {
        if (m.ha(b)) {
            for (var f = 0; f < b.length; f++)
                m.qm(a, b[f], c, d, e);
            return null
        }
        c = jm(c);
        return bm(a) ? a.Ic.add(String(b), c, !0, d, e) : km(a, b, c, !0, d, e)
    };
    m.rm = function(a, b, c, d, e) {
        if (m.ha(b))
            for (var f = 0; f < b.length; f++)
                m.rm(a, b[f], c, d, e);
        else 
            c = jm(c), bm(a) ? a.Da(b, c, d, e) : a && (a = lm(a)) && (b = gm(a, b, c, !!d, e)) && m.sm(b)
    };
    m.sm = function(a) {
        if (m.ka(a) ||!a || a.removed)
            return !1;
        var b = a.src;
        if (bm(b))
            return fm(b.Ic, a);
        var c = a.type, d = a.proxy;
        b.removeEventListener ? b.removeEventListener(c, d, a.yh) : b.detachEvent && b.detachEvent(nm(c), d);
        om--;
        (c = lm(b)) ? (fm(c, a), 0 == c.j && (c.src = null, b[mm] = null)) : dm(a);
        return !0
    };
    m.tm = function(a, b, c, d, e) {
        c = jm(c);
        d=!!d;
        return bm(a) ? gm(a.Ic, String(b), c, d, e) : a ? (a = lm(a)) ? gm(a, b, c, d, e) : null : null
    };
    var nm = function(a) {
        return a in um ? um[a] : um[a] = "on" + a
    };
    var vm = function(a, b, c, d) {
        var e = 1;
        if (a = lm(a))
            if (b = a.g[b.toString()])
                for (b = b.concat(), a = 0; a < b.length; a++) {
                    var f = b[a];
                    f && f.yh == c&&!f.removed && (e&=!1 !== wm(f, d))
                }
        return Boolean(e)
    };
    var wm = function(a, b) {
        var c = a.listener, d = a.ic || a.src;
        a.zh && m.sm(a);
        return c.call(d, b)
    };
    var Fba = function(a, b) {
        if (a.removed)
            return !0;
        if (!pm) {
            var c = b || m.n("window.event"), d = new m.am(c, this), e=!0;
            if (!(0 > c.keyCode || void 0 != c.returnValue)) {
                t:
                {
                    var f=!1;
                    if (0 == c.keyCode)
                        try {
                            c.keyCode =- 1;
                            break t
                    } catch (h) {
                        f=!0
                    }
                    if (f || void 0 == c.returnValue)
                        c.returnValue=!0
                }
                c = [];
                for (f = d.currentTarget; f; f = f.parentNode)
                    c.push(f);
                for (var f = a.type, k = c.length - 1; !d.qe && 0 <= k; k--)
                    d.currentTarget = c[k], e&=vm(c[k], f, !0, d);
                for (k = 0; !d.qe && k < c.length; k++)
                    d.currentTarget = c[k], e&=vm(c[k], f, !1, d)
                }
            return e
        }
        return wm(a, new m.am(b,
        this))
    };
    var lm = function(a) {
        a = a[mm];
        return a instanceof em ? a : null
    };
    var jm = function(a) {
        if (m.ma(a))
            return a;
        a[xm] || (a[xm] = function(b) {
            return a.handleEvent(b)
        });
        return a[xm]
    };
    m.ym = function() {
        m.t.call(this);
        this.Ic = new em(this);
        this.vf = this;
        this.Ca = null
    };
    var zm = function(a, b, c, d) {
        b = a.Ic.g[String(b)];
        if (!b)
            return !0;
        b = b.concat();
        for (var e=!0, f = 0; f < b.length; ++f) {
            var h = b[f];
            if (h&&!h.removed && h.yh == c) {
                var k = h.listener, r = h.ic || h.src;
                h.zh && fm(a.Ic, h);
                e=!1 !== k.call(r, d) && e
            }
        }
        return e && 0 != d.zn
    };
    m.Am = function(a, b) {
        m.ym.call(this);
        this.g = a || 1;
        this.j = b || m.da;
        this.k = (0, m.p)(this.Kr, this);
        this.C = (0, m.H)()
    };
    var Bm = function(a, b) {
        a.g = b;
        a.Ub && a.enabled ? (a.stop(), a.start()) : a.Ub && a.stop()
    };
    m.Cm = function(a, b, c) {
        if (m.ma(a))
            c && (a = (0, m.p)(a, c));
        else if (a && "function" == typeof a.handleEvent)
            a = (0, m.p)(a.handleEvent, a);
        else 
            throw Error("Invalid listener argument");
        return 2147483647 < b?-1 : m.da.setTimeout(a, b || 0)
    };
    var Dm = function(a, b, c) {
        m.t.call(this);
        this.j = a;
        this.g = b;
        this.aa = c;
        this.Xa = (0, m.p)(this.Wt, this)
    };
    var Em = function(a) {
        a.We = m.Cm(a.Xa, a.g);
        a.j.call(a.aa)
    };
    m.Fm = function(a) {
        if (!Gm || a)
            Gm = m.Bi();
        return Gm
    };
    var Hm = function(a) {
        if (!Im || a)
            Im = m.Ai(window.document);
        return Im
    };
    var Gba = function() {
        m.bb(Jm);
        Jm = m.v(function() {
            var a = m.Fm(!0);
            m.z("page-resize", a)
        }, 200)
    };
    var Hba = function() {
        var a = Hm();
        m.ob("page-scroll", a);
        Km = 0
    };
    var Iba = function() {
        var a = Hm(!0);
        m.Pc()&&!Km ? Km = m.Qc(Hba, 0) : m.z("page-scroll", a)
    };
    var Jba = function() {
        Lm = Mm();
        Nm()
    };
    var Kba = function() {
        Lm = Mm();
        Om()
    };
    var Mm = function() {
        var a = Hm(!0), b = m.Fm();
        return new m.Ve(a.y, a.x + b.width - 1, a.y + b.height - 1, a.x)
    };
    var Pm = function(a) {
        var b = m.Zk(a), c = Qm[b];
        if (c)
            return c;
        c = m.J(a, "scroll", Lba);
        return c = Qm[b] = {
            Fa: a,
            gz: c,
            of: null
        }
    };
    var Mba = function() {
        m.tb(Qm, function(a, b) {
            m.N(Qm[b].gz);
            delete Qm[b]
        })
    };
    var Lba = function(a) {
        Nm(a.target)
    };
    var Rm = function(a, b) {
        var c;
        b ? c = a : c = m.rl(a);
        return c ? (c = m.L(c, "yt-viewport")) ? Pm(c) : null : null
    };
    var Sm = function(a, b) {
        if (a.of&&!b)
            return a.of;
        var c = Tm(a.Fa), d = Rm(a.Fa);
        d && (d = Sm(d, b), c = Um(c, d));
        return a.of = c
    };
    var Tm = function(a) {
        var b = m.Xi(a);
        a = new m.pi(a.offsetWidth, a.offsetHeight);
        return new m.Ve(b.y, b.x + a.width - 1, b.y + a.height - 1, b.x)
    };
    var Um = function(a) {
        var b = [], c = [], d = [], e = [];
        (0, m.y)(arguments, function(a) {
            b.push(a.top);
            c.push(a.right);
            d.push(a.bottom);
            e.push(a.left)
        });
        var f = Math.max.apply(Math, b), h = Math.min.apply(Math, c), k = Math.min.apply(Math, d), r = Math.max.apply(Math, e);
        return f > k || r > h ? new m.Ve(0, 0, 0, 0) : new m.Ve(f, h, k, r)
    };
    var Vm = function(a, b) {
        var c = Lm, d = Tm(a);
        if (b.transform) {
            var e = b.transform;
            m.na(e) ? (d.top -= e.top, d.right += e.right, d.bottom += e.bottom, d.left -= e.left) : (d.top -= e, d.right += void 0, d.bottom += void 0, d.left -= window.NaN)
        }
        var f;
        b.complete ? f = m.We : f = uba;
        if (!f.call(m.Ve, c, d))
            return !1;
        e = Rm(a);
        if (!e)
            return !0;
        Sm(e);
        c = Um(c, e.of);
        return f.call(m.Ve, c, d)
    };
    var Wm = function(a, b, c) {
        var d, e = m.Zk(a) + "";
        b.hasOwnProperty(e) ? d = b[e] : b[e] = d = Vm(a, c);
        if (Boolean(Xm[e]) != d)
            return d ? "visible" : "hidden"
    };
    var Nba = function(a, b) {
        m.tb(Ym, function(c) {
            if (!b || m.vi(b, c.Fa)) {
                var d = Wm(c.Fa, a, c.options);
                d && d == c.type && m.v(m.q(c.se, c.Fa, c.type), 0)
            }
        })
    };
    var Oba = function(a, b) {
        m.tb(Zm, function(c) {
            if (!b || m.vi(c.Fa, b) || m.vi(b, c.Fa)) {
                var d = c.filter(c.Fa);
                if (d && d.length) {
                    var e = [];
                    (0, m.y)(d, function(b) {
                        var d = Wm(b, a, c.options);
                        d && d == c.type && e.push(b)
                    });
                    e.length && m.v(m.q(c.se, e, c.type), 0)
                }
            }
        })
    };
    var Om = function(a) {
        var b = {};
        Nba(b, a);
        Oba(b, a);
        m.zb(Xm, b)
    };
    var Pba = function(a, b, c) {
        return m.vb(Ym, function(d) {
            return d.Fa == a && d.type == b && d.se == c
        })
    };
    var Qba = function(a, b, c, d) {
        return m.vb(Zm, function(e) {
            return e.Fa == a && e.type == b && e.se == c && e.className == d
        })
    };
    var Rba = function(a, b) {
        var c = m.M("yt-viewport", b);
        (0, m.y)(c, a)
    };
    var $m = function(a) {
        Rba(function(a) {
            Pm(a)
        }, a);
        m.ik(a) && Rm(a, !0)
    };
    var Sba = function(a, b) {
        m.tb(Qm, function(c) {
            b&&!m.vi(b, c.Fa) || b == c.Fa || a(c)
        })
    };
    var Tba = function(a) {
        var b = Sm(a, !0);
        a = a.of;
        return !(a == b || a && b && a.top == b.top && a.right == b.right && a.bottom == b.bottom && a.left == b.left)
    };
    var Nm = function(a) {
        if (an) {
            var b;
            if (a)
                for (b = Rm(a, !0); b && Tba(b);)
                    b = Rm(a);
            Sba(function(a) {
                delete a.of
            }, b ? b.Fa : a);
            Om(a)
        }
    };
    var Uba = function() {
        var a = window.ytbuffer;
        if (a) {
            if (a = a.bufferedClick) {
                window.document.removeEventListener ? window.document.removeEventListener("click", window.ytbuffer.handleClick, !1) : window.document.detachEvent("onclick", window.ytbuffer.handleClick);
                var b = m.M("yt-is-buffered");
                (0, m.y)(b, function(a) {
                    m.D(a, "yt-is-buffered")
                });
                m.Lk(a.target || a.srcElement, "click");
                m.al("buffer_click", m.qc({
                    page_name: m.u("PAGE_NAME")
                }))
            }
            m.mk("ytbuffer")
        }
    };
    m.bn = function(a, b) {
        this.F = [];
        this.T = a;
        this.O = b || null;
        this.C = this.g=!1;
        this.k = void 0;
        this.S = this.U = this.L=!1;
        this.H = 0;
        this.j = null;
        this.D = 0
    };
    var cn = function(a, b, c) {
        a.g=!0;
        a.k = c;
        a.C=!b;
        dn(a)
    };
    var en = function(a) {
        if (a.g) {
            if (!a.S)
                throw new fn;
            a.S=!1
        }
    };
    m.gn = function(a, b, c, d) {
        a.F.push([b, c, d]);
        a.g && dn(a);
        return a
    };
    m.hn = function(a, b) {
        b instanceof m.bn ? a.addCallback((0, m.p)(b.J, b)) : a.addCallback(function() {
            return b
        })
    };
    var jn = function(a) {
        return (0, m.uh)(a.F, function(a) {
            return m.ma(a[1])
        })
    };
    var dn = function(a) {
        if (a.H && a.g && jn(a)) {
            var b = a.H, c = kn[b];
            c && (m.da.clearTimeout(c.ta), delete kn[b]);
            a.H = 0
        }
        a.j && (a.j.D--, delete a.j);
        for (var b = a.k, d = c=!1; a.F.length&&!a.L;) {
            var e = a.F.shift(), f = e[0], h = e[1], e = e[2];
            if (f = a.C ? h : f)
                try {
                    var k = f.call(e || a.O, b);
                    m.ca(k) && (a.C = a.C && (k == b || k instanceof Error), a.k = b = k);
                    m.oe(b) && (d=!0, a.L=!0)
                } catch (r) {
                b = r, a.C=!0, jn(a) || (c=!0)
            }
        }
        a.k = b;
        d && (k = (0, m.p)(a.Wk, a, !0), d = (0, m.p)(a.Wk, a, !1), b instanceof m.bn ? (m.gn(b, k, d), b.U=!0) : b.then(k, d));
        c && (b = new ln(b), kn[b.ta] = b, a.H = b.ta)
    };
    var fn = function() {
        m.ua.call(this)
    };
    var mn = function() {
        m.ua.call(this)
    };
    var ln = function(a) {
        this.ta = m.da.setTimeout((0, m.p)(this.j, this), 0);
        this.g = a
    };
    m.nn = function(a, b) {
        var c = b || {}, d = c.document || window.document, e = m.xl("SCRIPT"), f = {
            $o: e,
            nb: void 0
        }, h = new m.bn(Vba, f), k = null, r = null != c.timeout ? c.timeout: 5E3;
        0 < r && (k = window.setTimeout(function() {
            on(e, !0);
            h.De(new pn(1, "Timeout reached for loading script " + a))
        }, r), f.nb = k);
        e.onload = e.onreadystatechange = function() {
            e.readyState && "loaded" != e.readyState && "complete" != e.readyState || (on(e, c.Vt ||!1, k), h.Ia(null))
        };
        e.onerror = function() {
            on(e, !0, k);
            h.De(new pn(0, "Error while loading script " + a))
        };
        m.Nb(e, {
            type: "text/javascript",
            charset: "UTF-8",
            src: a
        });
        Wba(d).appendChild(e);
        return h
    };
    var Wba = function(a) {
        var b = a.getElementsByTagName("HEAD");
        return !b || m.Rl(b) ? a.documentElement : b[0]
    };
    var Vba = function() {
        if (this && this.$o) {
            var a = this.$o;
            a && "SCRIPT" == a.tagName && on(a, !0, this.nb)
        }
    };
    var on = function(a, b, c) {
        null != c && m.da.clearTimeout(c);
        a.onload = m.ea;
        a.onerror = m.ea;
        a.onreadystatechange = m.ea;
        b && window.setTimeout(function() {
            m.ri(a)
        }, 0)
    };
    var pn = function(a, b) {
        var c = "Jsloader error (code #" + a + ")";
        b && (c += ": " + b);
        m.ua.call(this, c)
    };
    var qn = function() {
        var a = (m.Ud.get("GuidedHelpResume") || "").split(":"), b = a[2];
        return {
            pe: (0, window.parseInt)(a[0], 10) || 0,
            oe: a[1] || "",
            source: (Cj(Xba, b) ? b : void 0) || "CUSTOM"
        }
    };
    var rn = function(a, b) {
        return Yba({
            theme: b.theme,
            document: b.document,
            mf: b.mf || "https://clients6.google.com",
            lf: b.lf,
            yg: b.yg,
            locale: b.locale
        }).then(function() {
            var c = b.oe, d = b.bz, e = b.fz, f = b.source || "CUSTOM";
            return m.n("help.guide.loadFlow")(a, c, d, e, f)
        })
    };
    var sn = function(a, b, c, d, e, f, h, k, r, w, B, Q) {
        ki(rn(a, {
            lf: Q || "",
            fz: r ||!1,
            document: w || void 0,
            mf: B || void 0,
            yg: c,
            locale: d,
            theme: e || void 0,
            source: "CUSTOM",
            oe: k || void 0
        }))
    };
    var tn = function() {
        var a = m.il(m.Bj().top.location), b = m.ak(a, "ghstartsource") || "";
        return {
            pe: (0, window.parseInt)(m.ak(a, "ghstartflowid"), 10) || 0,
            oe: m.ak(a, "ghstartstepid") || "",
            source: {
                helpcenter: "HELPCENTER",
                email: "EMAIL",
                chat: "CHAT",
                forum: "FORUM",
                marketing: "MARKETING",
                onebox: "ONEBOX",
                testing: "TESTING"
            }
            [b] || "LINK"
        }
    };
    var Yba = function(a) {
        var b;
        window.guidedhelp = window.guidedhelp || {};
        window.guidedhelp.loaded ? b = m.re() : (un || (un = m.nn("https://ssl.gstatic.com/inproduct_help/guidedhelp/guide_inproduct.js").then()), b = un);
        return b.then(function() {
            var b = a.yg, d = a.locale, e = a.theme, f = a.document, h = a.mf, k = a.lf;
            m.n("help.guide.init")(b, d, e, f, h, k, Zba)
        })
    };
    m.vn = function() {
        var a = m.n("help.guide.dismissFlow");
        a && a()
    };
    var $ba = function() {};
    var wn = function(a, b, c) {
        return a[b] = a[b] || c
    };
    var aca = function(a) {
        for (var b = 0; b < this.length; b++)
            if (this[b] === a)
                return b;
        return - 1
    };
    var bca = function(a) {
        a = a.sort();
        for (var b = [], c = void 0, d = 0; d < a.length; d++) {
            var e = a[d];
            e != c && b.push(e);
            c = e
        }
        return b
    };
    var xn = function() {
        var a;
        if ((a = Object.create) && cca.test(a))
            a = a(null);
        else {
            a = {};
            for (var b in a)
                a[b] = void 0
        }
        return a
    };
    var yn = function() {
        var a = dca.href, b;
        if (zn.dpo)
            b = zn.h;
        else {
            b = zn.h;
            var c = RegExp("([#].*&|[#])jsh=([^&#]*)", "g"), d = RegExp("([?#].*&|[?#])jsh=([^&#]*)", "g");
            if (a = a && (c.exec(a) || d.exec(a)))
                try {
                    b = (0, window.decodeURIComponent)(a[2])
                } catch (e) {}
        }
        return b
    };
    var eca = function(a) {
        var b = wn(zn, "PQ", []);
        zn.PQ = [];
        var c = b.length;
        if (0 === c)
            a();
        else 
            for (var d = 0, e = function() {
                ++d === c && a()
            }, f = 0; f < c; f++)
                b[f](e)
    };
    var An = function(a) {
        return wn(wn(zn, "H", xn()), a, xn())
    };
    var Bn = function(a, b, c) {
        b && 0 < b.length && (b = Cn(b), c && 0 < c.length && (b += "___" + Cn(c)), 28 < b.length && (b = b.substr(0, 28) + (b.length - 28)), c = b, b = wn(fca, "_p", xn()), wn(b, c, xn())[a] = (new Date).getTime(), b = Dn.r, "function" === typeof b ? b(a, "_p", c) : b.push([a, "_p", c]))
    };
    var Cn = function(a) {
        return a.join("__").replace(/\./g, "_").replace(/\-/g, "_").replace(/\,/g, "_")
    };
    var En = function(a) {
        throw Error("Bad hint" + (a ? ": " + a : ""));
    };
    var gca = function(a, b, c, d) {
        var e = a.split(";"), f = Fn[e.shift()], h = null;
        f && (h = f(e, b, c, d));
        if (b = h)
            b = h, c = b.match(hca), d = b.match(ica), b=!!d && 1 === d.length && jca.test(b)&&!!c && 1 === c.length;
        b || En(a);
        return h
    };
    var kca = function(a, b, c, d) {
        function e(a) {
            return (0, window.encodeURIComponent)(a).replace(/%2C/g, ",")
        }
        a = lca(a);
        mca.test(c) || En("invalid_callback");
        b = Gn(b);
        d = d && d.length ? Gn(d) : null;
        return [(0, window.encodeURIComponent)(a.vz).replace(/%2C/g, ",").replace(/%2F/g, "/"), "/k=", e(a.version), "/m=", e(b), d ? "/exm=" + e(d): "", "/rt=j/sv=1/d=1/ed=1", a.yp ? "/am=" + e(a.yp): "", a.zp ? "/rs=" + e(a.zp): "", "/cb=", e(c)].join("")
    };
    var lca = function(a) {
        "/" !== a.charAt(0) && En("relative path");
        for (var b = a.substring(1).split("/"), c = []; b.length;) {
            a = b.shift();
            if (!a.length || 0 == a.indexOf("."))
                En("empty/relative directory");
            else if (0 < a.indexOf("=")) {
                b.unshift(a);
                break
            }
            c.push(a)
        }
        a = {};
        for (var d = 0, e = b.length; d < e; ++d) {
            var f = b[d].split("="), h = (0, window.decodeURIComponent)(f[0]), k = (0, window.decodeURIComponent)(f[1]);
            2 == f.length && h && k && (a[h] = a[h] || k)
        }
        b = "/" + c.join("/");
        nca.test(b) || En("invalid_prefix");
        c = Hn(a, "k", !0);
        d = Hn(a, "am");
        a = Hn(a, "rs");
        return {
            vz: b,
            version: c,
            yp: d,
            zp: a
        }
    };
    var Gn = function(a) {
        for (var b = [], c = 0, d = a.length; c < d; ++c) {
            var e = a[c].replace(/\./g, "_").replace(/-/g, "_");
            oca.test(e) && b.push(e)
        }
        return b.join(",")
    };
    var Hn = function(a, b, c) {
        a = a[b];
        !a && c && En("missing: " + b);
        if (a) {
            if (pca.test(a))
                return a;
            En("invalid: " + b)
        }
        return null
    };
    var qca = function() {
        var a = yn();
        if (!a)
            throw Error("Bad hint");
        return a
    };
    var In = function(a, b) {
        for (var c = [], d = 0; d < a.length; ++d) {
            var e = a[d];
            e && 0 > aca.call(b, e) && c.push(e)
        }
        return c
    };
    var rca = function(a) {
        "loading" != Jn.readyState ? Kn(a) : Jn.write("<" + Ln + ' src="' + (0, window.encodeURI)(a) + '"></' + Ln + ">")
    };
    var Kn = function(a) {
        var b = Jn.createElement(Ln);
        b.setAttribute("src", a);
        b.async = "true";
        (a = Jn.getElementsByTagName(Ln)[0]) ? a.parentNode.insertBefore(b, a) : (Jn.head || Jn.body || Jn.documentElement).appendChild(b)
    };
    var sca = function(a, b) {
        var c = b && b._c;
        if (c)
            for (var d = 0; d < Mn.length; d++) {
                var e = Mn[d][0], f = Mn[d][1];
                f && Object.prototype.hasOwnProperty.call(c, e) && f(c[e], a, b)
            }
    };
    var tca = function(a, b) {
        Nn(function() {
            var c;
            c = b === yn() ? wn(On, "_", xn()) : xn();
            c = wn(An(b), "_", c);
            a(c)
        })
    };
    var Pn = function(a, b) {
        var c = b || {};
        "function" == typeof b && (c = {}, c.callback = b);
        sca(a, c);
        var d = a ? a.split(":"): [], e = c.h || qca(), f = wn(zn, "ah", xn());
        if (f["::"] && d.length) {
            for (var h = [], k = null; k = d.shift();) {
                var r = k.split("."), r = f[k] || f[r[1] && "ns:" + r[0] || ""] || e, w = h.length && h[h.length - 1] || null, B = w;
                w && w.hint == r || (B = {
                    hint: r,
                    features: []
                }, h.push(B));
                B.features.push(k)
            }
            var Q = h.length;
            if (1 < Q) {
                var Y = c.callback;
                Y && (c.callback = function() {
                    0==--Q && Y()
                })
            }
            for (; d = h.shift();)
                Qn(d.features, c, d.hint)
        } else 
            Qn(d || [], c, e)
    };
    var Qn = function(a, b, c) {
        function d(a, b) {
            if (w)
                return 0;
            Rn.clearTimeout(r);
            Q.push.apply(Q, la);
            var d = ((On || {}).config || {}).update;
            d ? d(f) : f && wn(zn, "cu", []).push(f);
            if (b) {
                Bn("me0", a, Y);
                try {
                    tca(b, c)
                } finally {
                    Bn("me1", a, Y)
                }
            }
            return 1
        }
        a = bca(a) || [];
        var e = b.callback, f = b.config, h = b.timeout, k = b.ontimeout, r = null, w=!1;
        if (h&&!k ||!h && k)
            throw "Timeout requires both the timeout parameter and ontimeout parameter to be set";
        var B = wn(An(c), "r", []).sort(), Q = wn(An(c), "L", []).sort(), Y = [].concat(B);
        0 < h && (r = Rn.setTimeout(function() {
            w =
            !0;
            k()
        }, h));
        var la = In(a, Q);
        if (la.length) {
            var la = In(a, B), Ma = wn(zn, "CP", []), Vb = Ma.length;
            Ma[Vb] = function(a) {
                function b() {
                    var a = Ma[Vb + 1];
                    a && a()
                }
                function c(b) {
                    Ma[Vb] = null;
                    d(la, a) && eca(function() {
                        e && e();
                        b()
                    })
                }
                if (!a)
                    return 0;
                Bn("ml1", la, Y);
                0 < Vb && Ma[Vb - 1] ? Ma[Vb] = function() {
                    c(b)
                } : c(b)
            };
            if (la.length) {
                var Xg = "loaded_" + zn.I++;
                On[Xg] = function(a) {
                    Ma[Vb](a);
                    On[Xg] = null
                };
                a = gca(c, la, "gapi." + Xg, B);
                B.push.apply(B, la);
                Bn("ml0", la, Y);
                b.sync || Rn.___gapisync ? rca(a) : Kn(a)
            } else 
                Ma[Vb]($ba)
        } else 
            d(la) && e && e()
    };
    var Nn = function(a) {
        if (zn.hee && 0 < zn.hel)
            try {
                return a()
        } catch (b) {
            zn.hel--, Pn("debug_error", function() {
                try {
                    window.___jsl.hefn(b)
                } catch (a) {
                    throw b;
                }
            })
        } else 
            return a()
    };
    m.Sn = function(a, b) {
        var c = m.ma(b) ? {
            callback: b
        }
        : b || {};
        c._c && c._c.jsl && c._c.jsl.h || m.zb(c, {
            _c: {
                jsl: {
                    h: m.u("GAPI_HINT_PARAMS")
                }
            }
        });
        if (m.u("GAPI_HINT_OVERRIDE")) {
            var d = m.cl(window.document.location.href).gapi_jsh;
            d && m.zb(c, {
                _c: {
                    jsl: {
                        h: d
                    }
                }
            })
        }
        Pn(a, c)
    };
    m.Tn = function(a, b, c) {
        var d = m.u("GAPI_HINT_PARAMS"), e = m.u("LOGGED_IN"), f = m.u("SESSION_INDEX"), h = m.u("DELEGATED_SESSION_ID"), k = {
            lang: m.u("GAPI_LOCALE"),
            "googleapis.config": {
                signedIn: e
            },
            iframes: {
                ":socialhost:": m.u("GAPI_HOST")
            }
        };
        b && c && (k.iframes[b] = {
            url: c
        });
        e && (f && (k["googleapis.config"].sessionIndex = f), h && (k["googleapis.config"].sessionDelegate = h));
        return {
            callback: a,
            config: k,
            _c: {
                jsl: {
                    h: d
                }
            }
        }
    };
    m.Un = function() {
        m.Sn("client", function() {
            m.tb(m.u("GUIDED_HELP_FLOWS"), function(a, b) {
                m.Vn(b, !0)
            })
        })
    };
    m.Vn = function(a, b) {
        if (!b ||!uca(a)) {
            var c = m.u("GUIDED_HELP_LOCALE") || "en_US", d = "https://clients6.google.com";
            "test" == m.u("GUIDED_HELP_ENVIRONMENT") && (d = "https://content-googleapis-test.sandbox.google.com");
            window.guidedhelp = window.guidedhelp || {};
            window.guidedhelp.onError = function() {};
            sn(a, 0, "/youtube", c, "youtube2", 0, 0, null, null, null, d, "AIzaSyBsoPoGbMyCfUNm_M-Hb-th-xAkxSutDvM");
            vca(a)
        }
    };
    m.Wn = function(a, b) {
        var c = m.u("GUIDED_HELP_FLOWS");
        c || (c = {});
        c[a] = b;
        m.Xa("GUIDED_HELP_FLOWS", c)
    };
    var vca = function(a) {
        var b = Xn(a);
        b && m.Td(b, "true", 7776E3);
        m.z("DISMISSED", a)
    };
    var uca = function(a) {
        return (a = Xn(a))?!!m.Vd(a) : !1
    };
    var Xn = function(a) {
        var b = m.u("GUIDED_HELP_FLOWS");
        return b && b[a]
    };
    var Yn = function(a, b) {
        this.version = a;
        this.args = b
    };
    var wca = function(a) {
        if (!a.Kb) {
            var b = {};
            a.call(b);
            a.Kb = b.version
        }
        return a.Kb
    };
    var Zn = function(a, b) {
        function c() {
            a.apply(this, b.args)
        }
        if (!b.args ||!b.version)
            throw Error("yt.pubsub2.Data.deserialize(): serializedData is incomplete.");
        var d;
        try {
            d = wca(a)
        } catch (e) {}
        if (!d || b.version != d)
            throw Error("yt.pubsub2.Data.deserialize(): serializedData version is incompatible.");
        c.prototype = a.prototype;
        try {
            return new c
        } catch (f) {
            throw f.message = "yt.pubsub2.Data.deserialize(): " + f.message, f;
        }
    };
    var $n = function(a, b) {
        this.g = a;
        this.Vh = b
    };
    var ao = function(a, b, c, d, e, f) {
        Yn.call(this, 1, arguments);
        this.j = a;
        this.g = b;
        this.D = c || null;
        this.C = d || null;
        this.k = e || null;
        this.source = f || null
    };
    var bo = function(a, b, c, d, e, f, h) {
        Yn.call(this, 1, arguments);
        this.j = a;
        this.Ui = b;
        this.g = c;
        this.D = d || null;
        this.C = e || null;
        this.k = f || null;
        this.source = h || null
    };
    var co = function() {
        this.j = null
    };
    var eo = function(a) {
        this.message = a ? String(a) : "Site PubSub mechanism is not supported."
    };
    var fo = function() {
        this.j = null;
        if (!go())
            throw new eo;
        this.g = null;
        this.g = (0, m.p)(this.k, this);
        window.addEventListener && window.addEventListener("storage", this.g, !1)
    };
    var go = function() {
        return !("undefined" == typeof window.localStorage || m.E&&!m.Fb("9"))
    };
    var ho = function(a) {
        if (m.ia(a))
            return xca(a);
        if (m.na(a)&&!m.ma(a)&&!(m.na(a) && 0 < a.nodeType))
            return yca(a);
        try {
            return m.da.JSON.stringify(a), a
        } catch (b) {}
    };
    var yca = function(a) {
        var b = {};
        m.tb(a, function(a, d) {
            b[d] = ho(a)
        });
        return b
    };
    var xca = function(a) {
        var b = [];
        (0, m.y)(a, function(a, d) {
            b[d] = ho(a)
        });
        return b
    };
    var io = function() {
        this.j = go() ? new fo : null;
        this.g = null;
        this.Kb = m.u("PAGE_CL") || 0;
        if (this.j) {
            var a = this.j, b = this.iu;
            this && (b = (0, m.p)(b, this));
            a.j = b;
            this.g = new m.Wa
        }
    };
    var zca = function() {
        (0, m.y)(jo, function(a) {
            io.getInstance().subscribe(a, m.q(Aca, a))
        })
    };
    var Bca = function() {
        (0, m.y)(jo, function(a) {
            if (!ko[a]) {
                var b = m.x(a, m.q(Cca, a));
                b && (ko[a] = b)
            }
        })
    };
    var Cca = function(a, b) {
        var c = io.getInstance();
        c.publish.apply(c, arguments);
        lo("pubsub", a)
    };
    var Aca = function(a, b) {
        var c = ko[a];
        if (c) {
            var d = Array.prototype.slice.call(arguments);
            d.unshift(c);
            Bba.apply(null, d)
        } else 
            m.z.apply(null, arguments);
        lo("sitepubsub", a)
    };
    var lo = function(a, b) {
        var c = {
            category: "proxy",
            action: a,
            page: m.u("PAGE_NAME", "")
        };
        b && m.zb(c, {
            topic: b
        });
        c = m.qc(c);
        m.Pj("pigeon", c, void 0)
    };
    m.mo = function(a) {
        var b = new m.aj;
        no(a, b, Dca);
        return b
    };
    m.oo = function(a) {
        var b = [];
        no(a, b, Eca);
        return b.join("&")
    };
    var no = function(a, b, c) {
        for (var d = a.elements, e, f = 0; e = d[f]; f++)
            if (e.form == a&&!e.disabled && "fieldset" != e.tagName.toLowerCase()) {
                var h = e.name;
                switch (e.type.toLowerCase()) {
                case "file":
                case "submit":
                case "reset":
                case "button":
                    break;
                case "select-multiple":
                    e = m.po(e);
                    if (null != e)
                        for (var k, r = 0; k = e[r]; r++)
                            c(b, h, k);
                            break;
                        default:
                            k = m.po(e), null != k && c(b, h, k)
                        }
            }
        d = a.getElementsByTagName("input");
        for (f = 0; e = d[f]; f++)
            e.form == a && "image" == e.type.toLowerCase() && (h = e.name, c(b, h, e.value), c(b, h + ".x", "0"), c(b, h + ".y", "0"))
        };
    var Dca = function(a, b, c) {
        var d = a.get(b);
        d || (d = [], a.set(b, d));
        d.push(c)
    };
    var Eca = function(a, b, c) {
        a.push((0, window.encodeURIComponent)(b) + "=" + (0, window.encodeURIComponent)(c))
    };
    m.po = function(a) {
        var b = a.type;
        if (!m.ca(b))
            return null;
        switch (b.toLowerCase()) {
        case "checkbox":
        case "radio":
            return a.checked ? a.value : null;
        case "select-one":
            return b = a.selectedIndex, 0 <= b ? a.options[b].value : null;
        case "select-multiple":
            for (var b = [], c, d = 0; c = a.options[d]; d++)
                c.selected && b.push(c.value);
            return b.length ? b : null;
        default:
            return m.ca(a.value) ? a.value : null
        }
    };
    var Fca = function(a, b, c, d, e, f, h) {
        function k() {
            4 == m.Zc(r) && b && m.$a(b)(r)
        }
        var r = m.Dh && (0, m.Dh)();
        if (!("open"in r))
            return null;
        "onloadend"in r ? r.addEventListener("loadend", k, !1) : r.onreadystatechange = k;
        c = (c || "GET").toUpperCase();
        d = d || "";
        r.open(c, a, !0);
        f && (r.responseType = f);
        h && (r.withCredentials=!0);
        f = "POST" == c;
        if (e = Gca(a, e))
            for (var w in e)
                r.setRequestHeader(w, e[w]), "content-type" == w.toLowerCase() && (f=!1);
        f && r.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
        r.send(d);
        return r
    };
    var Gca = function(a, b) {
        b = b || {};
        for (var c in qo) {
            var d = m.u(qo[c]), e;
            if (e = d) {
                e = a;
                var f = void 0;
                f = window.location.href;
                var h = m.hc(e)[1] || null, k = (0, m.Ad)(e);
                h && k ? (e = m.hc(e), f = m.hc(f), e = e[3] == f[3] && e[1] == f[1] && e[4] == f[4]) : e = k ? (0, m.Ad)(f) == k && (Number(m.hc(f)[4] || null) || null) == (Number(m.hc(e)[4] || null) || null) : !0;
                e || (e = c, f = m.u("CORS_HEADER_WHITELIST") || {}, e = (h = (0, m.Ad)(a)) ? (f = f[h]) ? m.La(f, e) : !1 : !0)
            }
            e && (b[c] = d)
        }
        return b
    };
    m.ro = function(a, b, c) {
        c.method = "POST";
        c.V || (c.V = {});
        c.V[m.u("XSRF_FIELD_NAME")] = b;
        return m.R(a, c)
    };
    var Hca = function(a, b) {
        var c = m.u("XSRF_FIELD_NAME"), d;
        b.headers && (d = b.headers["Content-Type"]);
        return !b.to && (!(0, m.Ad)(a) || (0, m.Ad)(a) == window.document.location.hostname) && "POST" == b.method && (!d || "application/x-www-form-urlencoded" == d)&&!(b.V && b.V[c])
    };
    m.R = function(a, b) {
        var c = b.format || "JSON";
        b.HA && (a = window.document.location.protocol + "//" + window.document.location.hostname + (window.document.location.port ? ":" + window.document.location.port : "") + a);
        var d = m.u("XSRF_FIELD_NAME"), e = m.u("XSRF_TOKEN"), f = b.Z;
        f && (f[d] && delete f[d], a = m.bl(a, f));
        var h = b.Rb || "", f = b.V;
        Hca(a, b) && (f || (f = {}), f[d] = e);
        f && m.ja(h) && (d = m.Yi(h), m.zb(d, f), h = m.qc(d));
        var k=!1, r, w = Fca(a, function(a) {
            if (!k) {
                k=!0;
                r && m.bb(r);
                var d;
                t:
                switch (Qj(a)) {
                case 200:
                case 201:
                case 202:
                case 203:
                case 204:
                case 205:
                case 206:
                case 304:
                    d =
                    !0;
                    break t;
                default:
                    d=!1
                }
                var e = null;
                if (d || 400 <= a.status && 500 > a.status)
                    e = Ica(c, a);
                if (d)
                    t: {
                    switch (c) {
                    case "XML":
                        d = 0 == (0, window.parseInt)(e && e.return_code, 10);
                        break t;
                    case "RAW":
                        d=!0;
                        break t
                    }
                    d=!!e
                }
                var e = e || {}, f = b.context || m.da;
                d ? b.R && b.R.call(f, a, e) : b.onError && b.onError.call(f, a, e);
                b.ka && b.ka.call(f, a, e)
            }
        }, b.method, h, b.headers, b.responseType, b.withCredentials);
        b.Zb && 0 < b.timeout && (r = m.v(function() {
            k || (k=!0, w.abort(), m.bb(r), b.Zb.call(b.context || m.da, w))
        }, b.timeout));
        return w
    };
    m.so = function(a, b) {
        var c = b || {};
        c.method = a.method.toUpperCase();
        if ("POST" == c.method)
            c.Rb = m.oo(a);
        else {
            var d = m.ej(m.mo(a)), e = c.Z || {};
            m.zb(e, d);
            c.Z = e
        }
        return m.R(a.action, c)
    };
    var Ica = function(a, b) {
        var c = null;
        switch (a) {
        case "JSON":
            var d = b.responseText, e = b.getResponseHeader("Content-Type") || "";
            d && 0 <= e.indexOf("json") && (c = m.uk(d));
            break;
        case "XML":
            if (d = (d = b.responseXML) ? to(d) : null)
                c = {}, (0, m.y)(d.getElementsByTagName("*"), function(a) {
                    c[a.tagName] = uo(a)
                })
        }
        return c
    };
    var to = function(a) {
        return a ? (a = ("responseXML"in a ? a.responseXML : a).getElementsByTagName("root")) && 0 < a.length ? a[0] : null : null
    };
    var vo = function(a, b) {
        if (!a)
            return null;
        var c = a.getElementsByTagName(b);
        return c && 0 < c.length ? uo(c[0]) : null
    };
    var uo = function(a) {
        var b = "";
        (0, m.y)(a.childNodes, function(a) {
            b += a.nodeValue
        });
        return b
    };
    var wo = function(a) {
        return a ? 24 == a.length && "UC" == a.slice(0, 2) ? a.substr(2) : 22 == a.length ? a : null : null
    };
    m.xo = function(a, b) {
        if (!b) {
            var c = m.u("CONVERSION_CONFIG_DICT");
            if (!c)
                return;
            b = c.uid || null;
            if (!b)
                return 
        }
        if ("subscribe" == a || "unsubscribe" == a) {
            if (m.ja(b)) {
                var d = wo(b);
                d && (d = {
                    label: "followon_" + a,
                    foc_id: d,
                    r: Math.round(1E4 * Math.random())
                }, (d = m.rc("//googleads.g.doubleclick.net/pagead/viewthroughconversion/962985656/", d)) && m.sc(d))
            }
        } else 
            t: {
            c = m.u("CONVERSION_CONFIG_DICT");
            if (m.ja(b)) {
                var e = wo(b);
                if (!c || c.uid != e)
                    if (c = Jca[e], !c || c.uid != e)
                        break t
            }
            if (a && c && c.baseUrl && c.uid) {
                var f = c.rmktEnabled, e = c.focEnabled &&
                (!c.isAd || "view" != a);
                if (f || e) {
                    var h = {};
                    if (f) {
                        f = {
                            utuid: c.uid,
                            type: a,
                            client_name: "html5"
                        };
                        "cvisit" == a && (f.type = "cview");
                        c.vid && (f.utvid = c.vid);
                        c.eventLabel && (f.el = c.eventLabel);
                        c.playerStyle && (f.ps = c.playerStyle);
                        c.feature && (f.feature = c.feature);
                        c.ppe && (f.ppe = c.ppe);
                        c.subscribed && (f.subscribed = c.subscribed);
                        c.engaged && (f.engaged = c.engaged);
                        var k = [];
                        for (d in f)
                            k.push((0, window.encodeURIComponent)(d) + "=" + (0, window.encodeURIComponent)(f[d]));
                            d = k.join(";");
                            h.data = d
                    }
                    e && (h.label = "followon_" + a, h.foc_id = c.uid,
                    h.r = Math.round(1E4 * Math.random()));
                    if ("unsubscribe" == a || "dislike" == a)
                        h.r = Math.round(1E4 * Math.random());
                        d = m.rc(c.baseUrl, h)
                    } else 
                        d = null
            } else 
                d = null;
                d && m.sc(d)
        }
    };
    m.yo = function(a, b) {
        var c = m.u("CONVERSION_CONFIG_DICT");
        c && c.vid == b && m.xo(a, c.uid)
    };
    m.zo = function(a) {
        if (a.altKey&&!a.ctrlKey || a.metaKey || 112 <= a.keyCode && 123 >= a.keyCode)
            return !1;
        switch (a.keyCode) {
        case 18:
        case 20:
        case 93:
        case 17:
        case 40:
        case 35:
        case 27:
        case 36:
        case 45:
        case 37:
        case 224:
        case 91:
        case 144:
        case 12:
        case 34:
        case 33:
        case 19:
        case 255:
        case 44:
        case 39:
        case 145:
        case 16:
        case 38:
        case 224:
        case 92:
            return !1;
        case 0:
            return !m.Re;
        default:
            return 166 > a.keyCode || 183 < a.keyCode
        }
    };
    var Kca = function(a, b, c, d, e) {
        if (!(m.E || m.Qe && m.Fb("525")))
            return !0;
        if (m.Oe && e)
            return Ao(a);
        if (e&&!d)
            return !1;
        m.ka(b) && (b = Bo(b));
        if (!c && (17 == b || 18 == b || m.Oe && 91 == b))
            return !1;
        if (m.Qe && d && c)
            switch (a) {
            case 220:
            case 219:
            case 221:
            case 192:
            case 186:
            case 189:
            case 187:
            case 188:
            case 190:
            case 191:
            case 192:
            case 222:
                return !1
            }
        if (m.E && d && b == a)
            return !1;
        switch (a) {
        case 13:
            return !0;
        case 27:
            return !m.Qe
        }
        return Ao(a)
    };
    var Ao = function(a) {
        if (48 <= a && 57 >= a || 96 <= a && 106 >= a || 65 <= a && 90 >= a || m.Qe && 0 == a)
            return !0;
        switch (a) {
        case 32:
        case 63:
        case 107:
        case 109:
        case 110:
        case 111:
        case 186:
        case 59:
        case 189:
        case 187:
        case 61:
        case 188:
        case 190:
        case 191:
        case 192:
        case 222:
        case 219:
        case 220:
        case 221:
            return !0;
        default:
            return !1
        }
    };
    var Bo = function(a) {
        if (m.Re)
            a = Lca(a);
        else if (m.Oe && m.Qe)
            t: switch (a) {
        case 93:
            a = 91;
            break t
        }
        return a
    };
    var Lca = function(a) {
        switch (a) {
        case 61:
            return 187;
        case 59:
            return 186;
        case 173:
            return 189;
        case 224:
            return 91;
        case 0:
            return 224;
        default:
            return a
        }
    };
    m.Co = function(a, b, c, d, e, f, h, k, r) {
        var w, B;
        if (w = c.offsetParent) {
            var Q = "HTML" == w.tagName || "BODY" == w.tagName;
            Q && "static" == m.Qi(w) || (B = m.Xi(w), Q || (Q = (Q = m.jj(w)) && m.Re?-w.scrollLeft : !Q || m.E && m.Fb("8") || "visible" == m.Hi(w, "overflowX") ? w.scrollLeft : w.scrollWidth - w.clientWidth - w.scrollLeft, B = lk(B, new m.ui(Q, w.scrollTop))))
        }
        w = B || new m.ui;
        B = m.zk(a);
        if (Q = Ck(a)) {
            var Y = new m.Lj(Q.left, Q.top, Q.right - Q.left, Q.bottom - Q.top), Q = Math.max(B.left, Y.left), la = Math.min(B.left + B.width, Y.left + Y.width);
            if (Q <= la) {
                var Ma = Math.max(B.top,
                Y.top), Y = Math.min(B.top + B.height, Y.top + Y.height);
                Ma <= Y && (B.left = Q, B.top = Ma, B.width = la - Q, B.height = Y - Ma)
            }
        }
        Q = m.Ii(a);
        Ma = m.Ii(c);
        if (Q.g != Ma.g) {
            var la = Q.g.body, Ma = m.Fi(Ma), Y = new m.ui(0, 0), Vb = m.Bj(m.Yb(la)), Xg = la;
            do {
                var vy = Vb == Ma ? m.Xi(Xg): lj(Xg);
                Y.x += vy.x;
                Y.y += vy.y
            }
            while (Vb && Vb != Ma && (Xg = Vb.frameElement) && (Vb = Vb.parent));
            la = lk(Y, m.Xi(la));
            !m.E || m.Ib(9) || m.Ei(Q) || (la = lk(la, m.Gi(Q)));
            B.left += la.x;
            B.top += la.y
        }
        a = Do(a, b);
        b = new m.ui(a & 2 ? B.left + B.width : B.left, a & 1 ? B.top + B.height : B.top);
        b = lk(b, w);
        e && (b.x += (a & 2?-1 :
        1) * e.x, b.y += (a & 1?-1 : 1) * e.y);
        var Ue;
        if (h)
            if (r)
                Ue = r;
            else if (Ue = Ck(c))
                Ue.top -= w.y, Ue.right -= w.x, Ue.bottom -= w.y, Ue.left -= w.x;
        return Eo(b, c, d, f, Ue, h, k)
    };
    var Eo = function(a, b, c, d, e, f, h) {
        a = a.clone();
        var k = Do(b, c);
        c = m.Jj(b);
        h = h ? h.clone() : c.clone();
        a = a.clone();
        h = h.clone();
        var r = 0;
        if (d || 0 != k)
            k & 2 ? a.x -= h.width + (d ? d.right : 0) : d && (a.x += d.left), k & 1 ? a.y -= h.height + (d ? d.bottom : 0) : d && (a.y += d.top);
        f && (e ? (d = a, k = h, r = 0, 65 == (f & 65) && (d.x < e.left || d.x >= e.right) && (f&=-2), 132 == (f & 132) && (d.y < e.top || d.y >= e.bottom) && (f&=-5), d.x < e.left && f & 1 && (d.x = e.left, r|=1), d.x < e.left && d.x + k.width > e.right && f & 16 && (k.width = Math.max(k.width - (d.x + k.width - e.right), 0), r|=4), d.x + k.width > e.right && f &
        1 && (d.x = Math.max(e.right - k.width, e.left), r|=1), f & 2 && (r = r | (d.x < e.left ? 16 : 0) | (d.x + k.width > e.right ? 32 : 0)), d.y < e.top && f & 4 && (d.y = e.top, r|=2), d.y <= e.top && d.y + k.height < e.bottom && f & 32 && (k.height = Math.max(k.height - (e.top - d.y), 0), d.y = e.top, r|=8), d.y >= e.top && d.y + k.height > e.bottom && f & 32 && (k.height = Math.max(k.height - (d.y + k.height - e.bottom), 0), r|=8), d.y + k.height > e.bottom && f & 4 && (d.y = Math.max(e.bottom - k.height, e.top), r|=2), f & 8 && (r = r | (d.y < e.top ? 64 : 0) | (d.y + k.height > e.bottom ? 128 : 0)), e = r) : e = 256, r = e);
        f = new m.Lj(0, 0, 0,
        0);
        f.left = a.x;
        f.top = a.y;
        f.width = h.width;
        f.height = h.height;
        e = r;
        if (e & 496)
            return e;
        m.Dk(b, new m.ui(f.left, f.top));
        h = new m.pi(f.width, f.height);
        c == h || c && h && c.width == h.width && c.height == h.height || (c = h, h = m.Yb(b), f = m.Ei(m.Ii(h)), !m.E || m.Fb("10") || f && m.Fb("8") ? (b = b.style, m.Re ? b.MozBoxSizing = "border-box" : m.Qe ? b.WebkitBoxSizing = "border-box" : b.boxSizing = "border-box", b.width = Math.max(c.width, 0) + "px", b.height = Math.max(c.height, 0) + "px") : (h = b.style, f ? (f = m.ij(b, "padding"), b = m.hj(b), h.pixelWidth = c.width - b.left - f.left -
        f.right - b.right, h.pixelHeight = c.height - b.top - f.top - f.bottom - b.bottom) : (h.pixelWidth = c.width, h.pixelHeight = c.height)));
        return e
    };
    var Do = function(a, b) {
        return (b & 4 && m.jj(a) ? b^2 : b)&-5
    };
    var Fo = function(a, b, c) {
        b in Go || (Go[b] = new m.Wa);
        Go[b].subscribe(a, c)
    };
    var Ho = function(a, b, c) {
        if (b in Go) {
            var d = Go[b];
            d.unsubscribe(a, c);
            0 >= d.ua() && (d.dispose(), delete Go[b])
        }
    };
    var Io = function(a, b, c) {
        var d;
        switch (a) {
        case "mouseover":
        case "mouseout":
            d = 3;
            break;
        case "mouseenter":
        case "mouseleave":
            d = 9
        }
        return m.Zb(c, function(a) {
            return m.A(a, b)
        }, !0, d)
    };
    var Jo = function(a) {
        var b = "mouseover" == a.type && "mouseenter"in Go || "mouseout" == a.type && "mouseleave"in Go, c = a.type in Go || b;
        if ("HTML" != a.target.tagName && c) {
            if (b) {
                var b = "mouseover" == a.type ? "mouseenter": "mouseleave", c = Go[b], d;
                for (d in c.Ob) {
                    var e = Io(b, d, a.target);
                    e&&!m.Zb(a.relatedTarget, function(a) {
                        return a == e
                    }, !0) && c.publish(d, e, b, a)
                }
            }
            if (b = Go[a.type])
                for (d in b.Ob)(e = Io(a.type, d, a.target)
                    ) && b.publish(d, e, a.type, a)
        }
    };
    m.Ko = function() {
        this.L = {};
        this.J = []
    };
    m.S = function(a, b) {
        return Lo(a) + (b ? "-" + b : "")
    };
    m.T = function(a, b, c, d) {
        d = m.S(a, d);
        var e = (0, m.p)(c, a);
        Fo(d, b, e);
        a.L[c] = e
    };
    m.Mo = function(a, b, c, d) {
        Ho(m.S(a, d), b, a.L[c]);
        delete a.L[c]
    };
    var Mca = function(a, b) {
        m.fl(a, "tooltip-text", b)
    };
    m.No = function(a, b, c) {
        return m.G(m.S(a, b), c)
    };
    var Lo = function(a) {
        return "yt-uix" + (a.va ? "-" + a.va : "")
    };
    m.Oo = function() {
        m.Ko.call(this);
        this.j = [];
        this.g = {}
    };
    var Nca = function(a, b, c, d, e) {
        var f = m.gj(c), h = 9 == d.keyCode;
        h || 32 == d.keyCode || 13 == d.keyCode ? (d = m.Po(a, c)) ? (b = m.jk(d), "a" == b.tagName.toLowerCase() ? window.location = b.href : m.Lk(b, "click")) : h && m.Qo(a, b) : f ? 27 == d.keyCode ? (m.Po(a, c), m.Qo(a, b)) : e(b, c, d) : (a = m.A(b, m.S(a, "reverse")) ? 38 : 40, d.keyCode == a && (m.Lk(b, "click"), d.preventDefault()))
    };
    m.Po = function(a, b) {
        var c = m.S(a, "menu-item-highlight"), d = m.G(c, b);
        d && m.D(d, c);
        return d
    };
    m.Ro = function(a, b, c) {
        m.C(c, m.S(a, "menu-item-highlight"));
        var d = c.getAttribute("id");
        d || (d = m.S(a, "item-id-" + m.oa(c)), c.setAttribute("id", d));
        b.setAttribute("aria-activedescendant", d)
    };
    var So = function(a, b) {
        var c = m.Oj("span", "yt-uix-button-icon-checkbox", b), c=!(c && m.gj(c));
        To(0, b, c);
        return c
    };
    var To = function(a, b, c) {
        a = m.Oj("span", "yt-uix-button-icon-wrapper", b);
        if (!a && c) {
            var d = m.Pb("span", {
                "class": "yt-uix-button-icon-wrapper yt-uix-button-icon-checkbox"
            }), e = m.Pb("div", {
                "class": "yt-uix-button-icon-dropdown-checked"
            });
            d.appendChild(e);
            m.vl(b, d, 0)
        }
        m.wf(a, c)
    };
    var Uo = function(a, b, c, d) {
        var e = b.length;
        a = (0, m.Na)(b, a);
        if ( - 1 == a)
            if (38 == d.keyCode)
                a = e - c;
            else {
                if (37 == d.keyCode || 38 == d.keyCode || 40 == d.keyCode)
                    a = 0
            } else 
                39 == d.keyCode ? (a%c == c - 1 && (a -= c), a += 1) : 37 == d.keyCode ? (0 == a%c && (a += c), --a) : 38 == d.keyCode ? (a < c && (a += e), a -= c) : 40 == d.keyCode && (a >= e - c && (a -= e), a += c);
        return a
    };
    m.Vo = function(a, b) {
        var c = Wo(a, b), d = m.Xo(a, c);
        d && d != b ? (m.Qo(a, d), m.v((0, m.p)(a.Rp, a, b), 1)) : m.gj(c) ? m.Qo(a, b) : a.Rp(b)
    };
    var Yo = function(a, b) {
        var c = b.iframeMask;
        c || (c = window.document.createElement("iframe"), c.src = 'javascript:""', c.className = m.S(a, "menu-mask"), b.iframeMask = c);
        return c
    };
    var Zo = function(a, b, c, d) {
        var e = m.L(b, m.S(a, "group")), f=!!a.getData(b, "button-menu-ignore-group"), e = e&&!f ? e : b, f = 5, h = 4, k = m.zk(b);
        if (m.A(b, m.S(a, "reverse"))) {
            f = 4;
            h = 5;
            k = k.top + "px";
            try {
                c.style.maxHeight = k
            } catch (r) {}
        }
        m.A(b, "flip") && (m.A(b, m.S(a, "reverse")) ? (f = 6, h = 7) : (f = 7, h = 6));
        var w;
        a.getData(b, "button-has-sibling-menu") ? w = Oi(e) : a.getData(b, "button-menu-root-container") && (w = $o(a, b));
        m.E&&!m.Fb("8") && (w = null);
        var B;
        w && (B = m.zk(w), B = new m.Ve( - B.top, B.left, B.top, - B.left));
        w = new m.ui(0, 1);
        m.A(b, m.S(a, "center-menu")) &&
        (w.x -= Math.round((m.Jj(c).width - m.Jj(b).width) / 2));
        d && (w.y += m.Ai(window.document).y);
        if (a = Yo(a, b))
            b = m.Jj(c), a.style.width = b.width + "px", a.style.height = b.height + "px", m.Co(e, f, a, h, w, B, 197), d && vk(a, "position", "fixed");
        m.Co(e, f, c, h, w, B, 197)
    };
    var $o = function(a, b) {
        if (a.getData(b, "button-menu-root-container")) {
            var c = a.getData(b, "button-menu-root-container");
            return m.L(b, c)
        }
        return window.document.body
    };
    m.Qo = function(a, b) {
        if (b) {
            var c = Wo(a, b);
            if (c) {
                a.Ye = null;
                b.setAttribute("aria-pressed", "false");
                b.setAttribute("aria-expanded", "false");
                b.removeAttribute("aria-activedescendant");
                m.P(c);
                a.hc(b, "button-menu-action", !1);
                var d = Yo(a, b), e = m.Zk(c).toString();
                delete a.g[e];
                m.v(function() {
                    d && d.parentNode && d.parentNode.removeChild(d);
                    c.originalParentNode && (c.parentNode.removeChild(c), c.originalParentNode.appendChild(c), c.originalParentNode = null, c.activeButtonNode = null)
                }, 1)
            }
            var e = m.L(b, m.S(a, "group")), f = [m.S(a,
            "active")];
            e && f.push(m.S(a, "group-active"));
            m.Gl(b, f);
            m.z("yt-uix-button-menu-hide", b);
            m.N(a.j);
            a.j.length = 0
        }
    };
    m.ap = function(a, b) {
        return m.G(m.S(a, "content"), b)
    };
    m.bp = function(a, b) {
        return Wo(a, b)
    };
    m.cp = function(a, b) {
        return m.Xo(a, b)
    };
    m.Xo = function(a, b) {
        return m.L(b.activeButtonNode || b.parentNode, m.S(a))
    };
    var Oca = function(a, b, c) {
        var d = m.S(a, "menu-item-selected");
        a = m.M(d, b);
        (0, m.y)(a, function(a) {
            m.D(a, d)
        });
        m.C(c.parentNode, d)
    };
    var Wo = function(a, b) {
        if (!b.widgetMenu) {
            var c = a.getData(b, "button-menu-id"), c = c && m.F(c), d = m.S(a, "menu");
            c ? m.Hl(c, [d, m.S(a, "menu-external")]) : c = m.G(d, b);
            b.widgetMenu = c
        }
        return b.widgetMenu
    };
    var dp = function(a, b) {
        if (a.getData(b, "button-toggle")) {
            var c = m.L(b, m.S(a, "group")), d = m.S(a, "toggled"), e = m.A(b, d);
            if (c && a.getData(c, "button-toggle-group")) {
                var f = a.getData(c, "button-toggle-group"), c = m.M(m.S(a), c);
                (0, m.y)(c, function(a) {
                    a != b || "optional" == f && e ? (m.D(a, d), a.removeAttribute("aria-pressed")) : (m.C(b, d), a.setAttribute("aria-pressed", "true"))
                })
            } else 
                e ? b.removeAttribute("aria-pressed") : b.setAttribute("aria-pressed", "true"), m.El(b, d)
        }
    };
    m.ep = function(a) {
        a.Ye && m.Qo(a, a.Ye)
    };
    var fp = function() {
        m.Ko.call(this)
    };
    var gp = function(a, b, c) {
        a = a.value;
        c && (c = a, a = window.document.createElement("div"), a.innerHTML = c, a = m.ml(a));
        return b ? (0, window.unescape)((0, window.encodeURIComponent)(a)).length : a.length
    };
    var hp = function() {
        m.Ko.call(this)
    };
    var ip = function(a, b, c) {
        var d = c || b, e = m.S(a, "card");
        c = m.jp(a, d);
        var f = m.F(m.S(a, "card") + m.Zk(d));
        if (f)
            return a = m.G(m.S(a, "card-body"), f), m.vi(a, c) || (m.ri(c), a.appendChild(c)), f;
        f = window.document.createElement("div");
        f.id = m.S(a, "card") + m.Zk(d);
        f.className = e;
        (d = a.getData(d, "card-class")) && m.Hl(f, d.split(/\s+/));
        d = window.document.createElement("div");
        d.className = m.S(a, "card-border");
        b = a.getData(b, "orientation") || "horizontal";
        e = window.document.createElement("div");
        e.className = "yt-uix-card-border-arrow yt-uix-card-border-arrow-" +
        b;
        var h = window.document.createElement("div");
        h.className = m.S(a, "card-body");
        a = window.document.createElement("div");
        a.className = "yt-uix-card-body-arrow yt-uix-card-body-arrow-" + b;
        m.ri(c);
        h.appendChild(c);
        d.appendChild(a);
        d.appendChild(h);
        f.appendChild(e);
        f.appendChild(d);
        window.document.body.appendChild(f);
        return f
    };
    var kp = function(a, b, c) {
        var d = a.getData(b, "orientation") || "horizontal", e = a.getData(b, "position"), f=!!a.getData(b, "force-position"), h = a.getData(b, "position-fixed"), d = "horizontal" == d, k = "bottomright" == e || "bottomleft" == e, r = "topright" == e || "bottomright" == e, w, B;
        r && k ? (B = 7, w = 4) : r&&!k ? (B = 6, w = 5) : !r && k ? (B = 5, w = 6) : (B = 4, w = 7);
        var Q = m.jj(window.document.body), e = m.jj(b);
        Q != e && (B^=2);
        var Y;
        d ? (e = b.offsetHeight / 2 - 12, Y = new m.ui( - 12, b.offsetHeight + 6)) : (e = b.offsetWidth / 2 - 6, Y = new m.ui(b.offsetWidth + 6, - 12));
        var la = m.Jj(c),
        e = Math.min(e, (d ? la.height : la.width) - 24 - 6);
        6 > e && (e = 6, d ? Y.y += 12 - b.offsetHeight / 2 : Y.x += 12 - b.offsetWidth / 2);
        var Ma = null;
        f || (Ma = 10);
        la = m.S(a, "card-flip");
        a = m.S(a, "card-reverse");
        m.rb(c, la, r);
        m.rb(c, a, k);
        Ma = m.Co(b, B, c, w, Y, null, Ma);
        !f && Ma && (Ma & 48 && (r=!r, B^=2, w^=2), Ma & 192 && (k=!k, B^=1, w^=1), m.rb(c, la, r), m.rb(c, a, k), m.Co(b, B, c, w, Y));
        h && (b = (0, window.parseInt)(c.style.top, 10), f = m.Ai(window.document).y, vk(c, "position", "fixed"), vk(c, "top", b - f + "px"));
        Q && (c.style.right = "", b = m.zk(c), b.left = b.left || (0, window.parseInt)(c.style.left,
        10), f = m.Bi(), c.style.left = "", c.style.right = f.width - b.left - b.width + "px");
        b = m.G("yt-uix-card-body-arrow", c);
        f = m.G("yt-uix-card-border-arrow", c);
        d = d ? k ? "top" : "bottom" : !Q && r || Q&&!r ? "left" : "right";
        b.setAttribute("style", "");
        f.setAttribute("style", "");
        b.style[d] = e + "px";
        f.style[d] = e + "px";
        k = m.G("yt-uix-card-arrow", c);
        r = m.G("yt-uix-card-arrow-background", c);
        k && r && (c = "right" == d ? m.Jj(c).width - e - 13 : e + 11, e = c / Math.sqrt(2), k.style.left = c + "px", k.style.marginLeft = "1px", r.style.marginLeft =- e + "px", r.style.marginTop =
        e + "px")
    };
    m.lp = function(a) {
        a.k && a.hide(a.k)
    };
    m.jp = function(a, b) {
        var c = b.cardContentNode;
        if (!c) {
            var d = m.S(a, "content"), e = m.S(a, "card-content");
            (c = (c = a.getData(b, "card-id")) ? m.F(c) : m.G(d, b)) || (c = window.document.createElement("div"));
            m.Dl(c, d, e);
            b.cardContentNode = c
        }
        return c
    };
    var mp = function(a) {
        var b = a.cardMask;
        b || (b = window.document.createElement("iframe"), b.src = 'javascript:""', m.Hl(b, ["yt-uix-card-iframe-mask"]), a.cardMask = b);
        b.style.position = a.style.position;
        b.style.top = a.style.top;
        b.style.left = a.offsetLeft + "px";
        b.style.height = a.clientHeight + "px";
        b.style.width = a.clientWidth + "px";
        window.document.body.appendChild(b)
    };
    m.np = function() {
        m.Ko.call(this);
        this.g = {};
        this.j = {}
    };
    var op = function() {
        m.Ko.call(this)
    };
    var pp = function() {
        m.Ko.call(this)
    };
    var qp = function(a, b) {
        var c = a.la(b);
        c && (c && (nl(c) || m.Nb(c, {
            tabIndex: "0"
        }), c.focus()), m.El(c, m.S(a, "collapsed")), rp(a, c), a.hc(c, "expander-action"))
    };
    var sp = function(a, b) {
        var c = a.la(b);
        c && (m.C(c, m.S(a, "collapsed")), rp(a, c), a.hc(c, "expander-action"))
    };
    var rp = function(a, b) {
        var c=!m.A(b, m.S(a, "collapsed"));
        m.z("yt-uix-expander-toggle", b, c);
        m.z("yt-dom-content-change", b)
    };
    m.tp = function() {
        m.Ko.call(this)
    };
    var up = function(a) {
        return a.options[Math.max(a.selectedIndex, 0)]
    };
    var Pca = function() {
        m.vp()
    };
    m.wp = function(a, b) {
        a.checked = b;
        m.tp.getInstance().tc(a)
    };
    m.xp = function() {
        var a = m.tp.getInstance(), b = m.M(m.S(a, "checkbox"));
        (0, m.y)(b, a.tc, a)
    };
    var yp = function() {
        var a = m.tp.getInstance(), b = m.M(m.S(a, "radio"));
        (0, m.y)(b, a.ep, a)
    };
    m.vp = function() {
        var a = m.tp.getInstance(), b = m.M(m.S(a, "select-element"));
        (0, m.y)(b, a.Jb, a)
    };
    m.zp = function() {
        m.xp();
        yp();
        m.vp()
    };
    m.Ap = function(a, b) {
        m.C(a, "yt-uix-form-error");
        var c = m.G("yt-uix-form-error-message", a);
        c ? c.innerHTML = b : (c = m.Pb("span", "yt-uix-form-error-message", window.document.createTextNode(String(b))), a.appendChild(c))
    };
    m.Bp = function(a) {
        m.D(a, "yt-uix-form-error");
        a = m.G("yt-uix-form-error-message", a);
        m.ri(a)
    };
    m.Cp = function() {
        m.Ko.call(this)
    };
    m.Dp = function(a, b) {
        return a.dataset ? b in a.dataset ? a.dataset[b] : null : a.getAttribute("data-" + String(b).replace(/([A-Z])/g, "-$1").toLowerCase())
    };
    var Ep = function() {
        m.Ko.call(this)
    };
    var Fp = function(a, b, c) {
        b && c && (m.C(c, m.S(a)), a = b.id, a || (a = "kbd-nav-" + Math.floor(1E6 * Math.random() + 1), b.id = a), b = a, c.dataset ? c.dataset.kbdNavMoveOut = b : c.setAttribute("data-" + "kbdNavMoveOut".replace(/([A-Z])/g, "-$1").toLowerCase(), b))
    };
    var Gp = function(a, b) {
        if (b) {
            var c = m.qi(b, "LI");
            c && (m.C(c, m.S(a, "highlight")), Hp = m.J(b, "blur", (0, m.p)(function(a) {
                m.D(a, m.S(this, "highlight"));
                m.N(Hp)
            }, a, c)))
        }
    };
    var Ip = function(a) {
        if ("UL" != a.tagName.toUpperCase())
            return [];
        a = (0, m.qb)(m.wi(a), function(a) {
            return "LI" == a.tagName.toUpperCase()
        });
        return (0, m.qb)((0, m.sh)(a, function(a) {
            return m.gj(a) ? ol(a, function(a) {
                return m.ik(a) ? nl(a) : !1
            }) : !1
        }), function(a) {
            return !!a
        })
    };
    var Jp = function() {
        m.Ko.call(this)
    };
    var Kp = function(a, b, c) {
        a = m.M(m.S(a, "main-button"));
        var d = m.Oo.getInstance();
        (0, m.y)(a, function(a) {
            if (m.I(a, "uix-livereminder-video-id") == b)
                switch (c) {
                case "disable":
                    m.Wk(a, !1);
                    break;
                case "cancel":
                    m.Wk(a, !0);
                    break;
                case "set":
                    m.C(a, "yt-uix-button-livereminder-set");
                    m.pl(m.ap(d, a), m.I(a, "reminder-set-text"));
                    m.Wk(a, !0);
                    break;
                case "removed":
                    m.D(a, "yt-uix-button-livereminder-set"), m.pl(m.ap(d, a), m.I(a, "set-reminder-text")), m.Wk(a, !0)
                }
        })
    };
    var Lp = function() {
        m.Rl(Mp) && (Mp.push(m.x("page-resize", Np)), Mp.push(m.x("page-scroll", Np)))
    };
    m.Op = function(a) {
        m.Rl(Mp) && Lp();
        m.zb(Pp, a);
        Qp()
    };
    var Qca = function() {
        (0, m.y)(["load-more-auto"], function(a) {
            m.wb(Pp, a)
        })
    };
    var Rp = function() {
        var a = m.Fm();
        return Hm().y + a.height
    };
    var Np = function() {
        var a = Rp(), b = Sp;
        if (!b || a >= b)
            Qp(), Sp = a
    };
    var Qp = function() {
        var a = Rp();
        Rca(function(b, c) {
            if (a >= xk(b).y) {
                var d = Pp[c];
                d&&-1 == (0, m.Na)(Tp, b) && (d.call(m.da, b), Tp.push(b))
            }
        })
    };
    var Rca = function(a) {
        var b = m.M("scrolldetect");
        (0, m.y)(b, function(b) {
            var d = m.I(b, "scrolldetect-callback");
            d && a.call(m.da, b, d)
        })
    };
    m.Up = function() {
        m.Ko.call(this)
    };
    var Sca = function(a, b, c) {
        m.C(b, m.S(a, "loading"));
        b.disabled=!0;
        m.P(m.G("load-more-text", b));
        m.K(m.G("load-more-loading", b));
        a.removeClass(b, "error");
        m.R(c, {
            context: a,
            R: (0, m.p)(a.yy, a, b),
            onError: (0, m.p)(a.xk, a, b),
            Zb: (0, m.p)(a.xk, a, b),
            ka: function() {
                m.P(m.G("load-more-loading", b));
                m.K(m.G("load-more-text", b));
                b.disabled=!1;
                m.D(b, m.S(this, "loading"))
            }
        })
    };
    m.Vp = function() {
        m.Ko.call(this);
        this.j = this.g = null;
        this.k = {};
        this.D = {};
        this.C = null
    };
    m.Wp = function(a) {
        return (a = m.Xp(a)) ? Yp(m.Vp.getInstance(), a) : null
    };
    m.Xp = function(a) {
        var b = m.Vp.getInstance();
        if (m.A(a, m.S(b)))
            return a;
        var c = b.la(a);
        return c ? c : m.L(a, m.S(b, "content")) == b.g ? b.j : null
    };
    var Zp = function(a, b, c) {
        var d = $p(a, b);
        d && m.Ak(d, m.Jj(c));
        if (c == a.g) {
            var e = 5, f = 4;
            m.A(b, m.S(a, "reversed")) && (e^=1, f^=1);
            m.A(b, m.S(a, "flipped")) && (e^=2, f^=2);
            a = new m.ui(0, 1);
            d && m.Co(b, e, d, f, a, null, 65);
            m.Co(b, e, c, f, a, null, 65)
        }
    };
    var aq = function(a, b, c) {
        bq(a, b)&&!c ? cq(a, b) : (Tca(a, b), !a.g || Xk(b, a.g) ? a.Mj(b) : Xaa(a.C, (0, m.p)(a.Mj, a, b)))
    };
    var Tca = function(a, b) {
        if (b) {
            var c = m.L(b, m.S(a, "content"));
            c && (c = m.M(m.S(a), c), (0, m.y)(c, function(a) {
                !Xk(a, b) && bq(this, a) && m.dq(this, a)
            }, a))
        }
    };
    var cq = function(a, b) {
        if (b) {
            var c = [];
            c.push(b);
            var d = Yp(a, b);
            d && (d = m.M(m.S(a), d), d = m.Ra(d), c = c.concat(d), (0, m.y)(c, function(a) {
                bq(this, a) && m.dq(this, a)
            }, a))
        }
    };
    m.dq = function(a, b) {
        if (b) {
            var c = Yp(a, b);
            m.D(eq(a, b), m.S(a, "trigger-selected"));
            m.C(c, m.S(a, "content-hidden"));
            var d = Yp(a, b);
            d && m.Nb(d, {
                "aria-expanded": "false"
            });
            (d = $p(a, b)) && d.parentNode && m.ri(d);
            c && c == a.g && (a.j.appendChild(c), a.g = null, a.j = null, a.C.publish("ROOT_MENU_REMOVED"));
            m.z("yt-uix-menu-hide", b);
            c = m.oa(b).toString();
            m.N(a.k[c]);
            delete a.k[c]
        }
    };
    var Uca = function(a, b) {
        var c = Yp(a, b);
        if (c) {
            (0, m.y)(c.children, function(a) {
                "LI" == a.tagName && m.Nb(a, {
                    role: "menuitem"
                })
            });
            m.Nb(c, {
                "aria-expanded": "true"
            });
            var d = c.id;
            d || (d = "aria-menu-id-" + m.oa(c), c.id = d);
            (c = eq(a, b)) && m.Nb(c, {
                "aria-controls": d
            })
        }
    };
    var Vca = function(a, b, c) {
        var d = Yp(a, b);
        d && m.A(b, m.S(a, "checked")) && (a = m.qi(c, "LI")) && (a = m.G("yt-ui-menu-item-checked-hid", a)) && (d = m.M("yt-ui-menu-item-checked", d), (0, m.y)(d, function(a) {
            m.Fl(a, "yt-ui-menu-item-checked", "yt-ui-menu-item-checked-hid")
        }), m.Fl(a, "yt-ui-menu-item-checked-hid", "yt-ui-menu-item-checked"))
    };
    var bq = function(a, b) {
        var c = Yp(a, b);
        return c?!m.A(c, m.S(a, "content-hidden")) : !1
    };
    var fq = function(a, b) {
        var c = m.Mb("UL", null, b);
        (0, m.y)(c, function(a) {
            a.tabIndex = "0";
            var b = Ep.getInstance();
            m.Hl(a, [m.S(b), m.S(b, "list")])
        })
    };
    var Yp = function(a, b) {
        var c = m.I(b, "menu-content-id");
        return c && (c = m.F(c)) ? (m.Hl(c, [m.S(a, "content"), m.S(a, "content-external")]), c) : b == a.j ? a.g : m.G(m.S(a, "content"), b)
    };
    var $p = function(a, b) {
        var c = m.oa(b).toString(), d = a.D[c];
        if (!d) {
            d = window.document.createElement("IFRAME");
            d.src = 'javascript:""';
            var e = [m.S(a, "mask")];
            (0, m.y)(m.pb(b), function(a) {
                e.push(a + "-mask")
            });
            m.Hl(d, e);
            a.D[c] = d
        }
        return d || null
    };
    var eq = function(a, b) {
        return m.G(m.S(a, "trigger"), b)
    };
    var gq = function(a, b) {
        return Xk(b, a.g) || Xk(b, a.j)
    };
    m.hq = function(a, b, c, d) {
        this.ob = a;
        this.F=!1;
        this.k = new m.Wa;
        this.H = m.O(this.ob, "click", (0, m.p)(this.er, this), "yt-dialog-dismiss");
        this.setState("content");
        this.L = b;
        this.S = c;
        this.J = d;
        this.g = this.D = this.C = null
    };
    m.iq = function(a, b, c, d) {
        a.ha() || a.k.subscribe((d ? "pre-" : "post-") + b, c)
    };
    var Wca = function() {
        var a = m.M("yt-dialog");
        return (0, m.uh)(a, function(a) {
            return m.gj(a)
        })
    };
    var Xca = function(a) {
        var b = m.Mb("iframe", null, a.ob);
        (0, m.y)(b, function(a) {
            var b = m.I(a, "onload");
            b && (b = m.n(b)) && m.J(a, "load", b);
            if (b = m.I(a, "src"))
                a.src = b
        }, a);
        return m.Ra(b)
    };
    var Yca = function(a) {
        (0, m.y)(window.document.getElementsByTagName("iframe"), function(b) {
            - 1 == (0, m.Na)(a, b) && m.C(b, "iframe-hid")
        })
    };
    var Zca = function() {
        var a = m.M("iframe-hid");
        (0, m.y)(a, function(a) {
            m.D(a, "iframe-hid")
        })
    };
    m.jq = function(a, b) {
        if (!a.ha()) {
            a.k.publish("pre-all");
            a.k.publish("pre-" + b);
            m.P(a.ob);
            m.lp(m.np.getInstance());
            m.lp(m.Cp.getInstance());
            Wca() || (m.P(a.j), m.D(window.document.body, "yt-dialog-active"), wba(), Zca());
            a.C && (m.N(a.C), a.C = null);
            a.D && (m.N(a.D), a.D = null);
            var c = a.ob;
            if (c) {
                var d = m.I(c, "player-ready-pubsub-key");
                d && (m.nb(d), m.el(c, "player-ready-pubsub-key"))
            }
            a.k.publish("post-all");
            m.z("yt-ui-dialog-hide-complete", a);
            "cancel" == b && m.z("yt-ui-dialog-cancelled", a);
            a.k && a.k.publish("post-" + b);
            a.g =
            null
        }
    };
    m.kq = function() {
        m.Ko.call(this)
    };
    var lq = function(a) {
        a.j || (a.j = m.x("yt-uix-overlay-hide", $ca));
        a.lc && m.iq(a.lc, "all", function() {
            var a = m.kq.getInstance();
            a.Xe = null;
            a.lc.dispose();
            a.lc = null
        })
    };
    var mq = function(a) {
        if (a.lc) {
            var b = a.Xe;
            m.jq(a.lc, "overlayhide");
            a.hc(b, "overlay-hidden");
            a.Xe = null;
            a.g && (m.N(a.g), a.g = null);
            a.lc = null
        }
    };
    var ada = function(a, b) {
        var c;
        if (a)
            if (c = m.G("yt-dialog", a)) {
                var d = m.F("body-container");
                d && (d.appendChild(c), a.overlayContentNode = c, c.overlayParentNode = a)
            } else 
                c = a.overlayContentNode;
        else 
            b && (c = m.L(b, "yt-dialog"));
        return c
    };
    m.nq = function() {
        var a = m.kq.getInstance();
        if (a.Xe)
            a = m.G("yt-dialog-fg-content", a.Xe.overlayContentNode);
        else 
            t: {
            if (a = m.M("yt-dialog-fg-content"))
                for (var b = 0; b < a.length; b++) {
                    var c = m.L(a[b], "yt-dialog");
                    if (m.gj(c)) {
                        a = a[b];
                        break t
                    }
                }
                a = null
        }
        return a
    };
    var $ca = function() {
        mq(m.kq.getInstance())
    };
    var oq = function() {
        m.Ko.call(this)
    };
    var bda = function(a, b, c, d, e, f) {
        m.R("/playlist_ajax?action_playlist_vote=1", {
            method: "POST",
            V: {
                session_token: f,
                list: b,
                vote: c
            },
            R: function(a, f) {
                if (200 == f.code) {
                    var r = {
                        id: f.browse_list_id,
                        url: f.url,
                        title: f.title,
                        IA: f.playlist_type
                    };
                    if ("like" == c) {
                        m.C(d, e);
                        var w = m.I(d, "unlike-label") || "", B = m.I(d, "unlike-tooltip") || "", Q = "yt-uix-playlistlike-liked"
                    } else 
                        m.D(d, e), w = m.I(d, "like-label") || "", B = m.I(d, "like-tooltip") || "", Q = "yt-uix-playlistlike-unliked";
                    var Y = m.G("yt-uix-button-content", d);
                    Y && (Y.innerHTML = m.wa(w));
                    m.fl(d, "tooltip-text", B);
                    r.id && m.z(Q, r);
                    f.show_feed_privacy_dialog && m.z("SHOW-FEED-PRIVACY-LIKE-PLAYLIST-DIALOG", b)
                }
            },
            context: a
        })
    };
    var pq = function() {
        m.Ko.call(this)
    };
    var qq = function() {
        m.Ko.call(this)
    };
    m.rq = function() {
        m.Ko.call(this);
        this.g = {}
    };
    m.sq = function(a) {
        a = m.I(a, "sessionlink") || "";
        return m.Yi(a)
    };
    var tq = function() {
        m.Ko.call(this)
    };
    var uq = function(a) {
        var b;
        if (b = m.I(a, "sessionlink-target") || a.href || "") {
            var c = m.sq(a);
            (a = (0, window.parseInt)(m.I(a, "sessionlink-lifetime") || "", 10)) ? m.Zd(b, c, a) : m.Zd(b, c)
        }
    };
    m.vq = function() {
        m.Ko.call(this)
    };
    var wq = function(a, b, c) {
        var d = a.la(b);
        if (!m.A(d, m.S(a, "is-moving"))) {
            b = m.No(a, "list", d);
            var e = m.No(a, "body", d), f = m.Ra(m.M(m.S(a, "item"), d));
            if (f) {
                var h = f[0];
                if ("forward" == c) {
                    var k = xq(d, e, f);
                    c = m.tl(k);
                    if (!c)
                        return;
                    k = cda(a, c, e, f);
                    h = c
                } else if ("back" == c) {
                    k = yq(d, e, f);
                    c = m.sl(k);
                    if (!c)
                        return;
                    k = dda(a, c, e, f);
                    h = c
                }
                m.C(d, m.S(a, "is-moving"));
                m.jj(d) ? b.style.right = ((0, window.parseInt)(b.style.right, 10) || 0) - k + "px" : b.style.left = ((0, window.parseInt)(b.style.left, 10) || 0) + k + "px";
                var r = m.v((0, m.p)(a.bp, a, d, h), 510);
                m.Rk(b,
                zq, (0, m.p)(function(a) {
                    m.bb(r);
                    this.bp(d, a)
                }, a, h))
            }
        }
    };
    var yq = function(a, b, c) {
        return m.jj(a) ? Aq(a, b, c) : Bq(a, b, c)
    };
    var xq = function(a, b, c) {
        return m.jj(a) ? Bq(a, b, c) : Aq(a, b, c)
    };
    var Bq = function(a, b, c) {
        function d(a) {
            return Bk(a) > e - 1
        }
        var e = Bk(b);
        return m.jj(a) ? Sl(c, d) : m.Ja(c, d)
    };
    var Aq = function(a, b, c) {
        function d(a) {
            a = Bk(a) + a.offsetWidth;
            return e > a - 1
        }
        var e = Bk(b) + b.offsetWidth;
        return m.jj(a) ? m.Ja(c, d) : Sl(c, d)
    };
    var dda = function(a, b, c, d) {
        var e, f = a.la(b);
        e = Bk(c);
        var h = e + c.offsetWidth, k = Bk(b);
        b = k + b.offsetWidth;
        var r = d[0];
        d = Bk(r);
        r = d + r.offsetWidth;
        m.jj(f) ? (c = a.Ek(f, c) - k, e = h - r) : (c = a.Fk(f, c) - b, e -= d);
        m.rb(f, m.S(a, "at-tail"), !1);
        if (Math.abs(c) + 1 < Math.abs(e))
            return c;
        m.rb(f, m.S(a, "at-head"), !0);
        return e
    };
    var cda = function(a, b, c, d) {
        var e, f = a.la(b);
        e = Bk(c);
        var h = e + c.offsetWidth, k = Bk(b);
        b = k + b.offsetWidth;
        var r = d[d.length - 1];
        d = Bk(r);
        r = d + r.offsetWidth;
        m.jj(f) ? (c = a.Fk(f, c) - b, e -= d) : (c = a.Ek(f, c) - k, e = h - r);
        m.rb(f, m.S(a, "at-head"), !1);
        if (Math.abs(c) + 1 < Math.abs(e))
            return c;
        m.rb(f, m.S(a, "at-tail"), !0);
        return e
    };
    var Cq = function(a, b) {
        var c = m.No(a, "body", b), d = m.Ra(m.M(m.S(a, "item"), b)), e;
        d && d.length ? (e = yq(b, c, d), c = xq(b, c, d), e = e == d[0], d = c == d[d.length - 1]) : e = d=!0;
        m.rb(b, m.S(a, "at-head"), e);
        m.rb(b, m.S(a, "at-tail"), d);
        if (a.getData(b, "disable-slider-buttons")) {
            if (c = m.No(a, "prev", b))
                c.disabled = e;
            if (e = m.No(a, "next", b))
                e.disabled = d
        }
    };
    m.Dq = function(a, b) {
        var c = Eq();
        c && c.publish.call(c, a.toString(), a, b)
    };
    var Fq = function(a, b) {
        var c = Eq();
        if (!c)
            return 0;
        var d = c.subscribe(a.toString(), function(c, f) {
            if (!window.yt.pubsub2.skipSubKey_ || window.yt.pubsub2.skipSubKey_ != d) {
                var h = function() {
                    if (Gq[d])
                        try {
                            if (f && a instanceof $n && a != c)
                                try {
                                    f = Zn(a.Vh, f)
                                } catch (h) {
                                    throw h.message = "yt.pubsub2 cross-binary conversion error for " + a.toString() + ": " + h.message, h;
                                }
                                b.call(window, f)
                    } catch (r) {
                        m.db(r)
                    }
                };
                Hq[a.toString()] ? m.Oc() ? m.Qc(h, void 0) : m.v(h, 0) : h()
            }
        });
        Gq[d]=!0;
        Iq[a.toString()] || (Iq[a.toString()] = []);
        Iq[a.toString()].push(d);
        return d
    };
    var Jq = function(a) {
        var b = Eq();
        b && (m.ka(a) && (a = [a]), (0, m.y)(a, function(a) {
            b.unsubscribeByKey(a);
            delete Gq[a]
        }))
    };
    var Eq = function() {
        return m.n("yt.pubsub2.instance_")
    };
    var Kq = function(a, b) {
        var c = window.document.location.protocol + "//" + window.document.domain + "/post_login";
        b && (c = m.jl(c, "mode", b));
        c = m.jl("/signin?context=popup", "next", c);
        c = m.jl(c, "feature", "sub_button");
        if (c = window.open(c, "loginPopup", "width=375,height=440,resizable=yes,scrollbars=yes", !0)) {
            var d = m.x("LOGGED_IN", function(b) {
                m.nb(m.u("LOGGED_IN_PUBSUB_KEY"));
                m.Xa("LOGGED_IN", !0);
                a(b)
            });
            m.Xa("LOGGED_IN_PUBSUB_KEY", d);
            c.moveTo((window.screen.width - 375) / 2, (window.screen.height - 440) / 2)
        }
    };
    var Lq = function() {
        var a = m.u("PLAYER_CONFIG");
        return a && a.args && void 0 !== a.args.authuser?!0 : !(!m.u("SESSION_INDEX")&&!m.u("LOGGED_IN"))
    };
    m.Mq = function() {
        m.Ko.call(this);
        this.g = {}
    };
    m.Nq = function(a, b, c) {
        m.fl(b, "tooltip-text", c);
        a = a.getData(b, "content-id");
        if (a = m.F(a))
            a.innerHTML = c
    };
    var Oq = function(a, b) {
        return a.getData(b, "tooltip-text") || b.title
    };
    var eda = function(a, b) {
        if (b) {
            var c = Oq(a, b);
            if (c) {
                var d = m.F(Pq(a, b));
                if (!d) {
                    d = window.document.createElement("div");
                    d.id = Pq(a, b);
                    d.className = m.S(a, "tip");
                    var e = window.document.createElement("div");
                    e.className = m.S(a, "tip-body");
                    var f = window.document.createElement("div");
                    f.className = m.S(a, "tip-arrow");
                    var h = window.document.createElement("div");
                    h.className = m.S(a, "tip-content");
                    var k = fda(a, b), r = Pq(a, b, "content");
                    h.id = r;
                    m.fl(b, "content-id", r);
                    e.appendChild(h);
                    k && d.appendChild(k);
                    d.appendChild(e);
                    d.appendChild(f);
                    (m.Tk() || window.document.body).appendChild(d);
                    m.Nq(a, b, c);
                    (c = (0, window.parseInt)(a.getData(b, "tooltip-max-width"), 10)) && e.offsetWidth > c && (e.style.width = c + "px", m.C(h, m.S(a, "normal-wrap")));
                    h = m.A(b, m.S(a, "reverse"));
                    Qq(a, b, d, e, k, h) || Qq(a, b, d, e, k, !h);
                    var w = m.S(a, "tip-visible");
                    m.v(function() {
                        m.C(d, w)
                    }, 0)
                }
            }
        }
    };
    var Qq = function(a, b, c, d, e, f) {
        m.rb(c, m.S(a, "tip-reverse"), f);
        var h = 0;
        f && (h = 1);
        var k = m.Jj(b);
        f = new m.ui((k.width - 10) / 2, f ? k.height : 0);
        var r = m.Xi(b);
        Eo(new m.ui(r.x + f.x, r.y + f.y), c, h);
        var r = m.Bi(), w = Kj(c);
        c = m.Jj(d);
        var B = Math.floor(c.width / 2), h=!!(r.height < w.y + k.height), k=!!(w.y < k.height);
        f=!!(w.x < B);
        r=!!(r.width < w.x + B);
        w = (c.width + 3)/-2 - - 5;
        a = a.getData(b, "force-tooltip-direction");
        if ("left" == a || f)
            w =- 5;
        else if ("right" == a || r)
            w = 20 - c.width - 3;
        a = Math.floor(w) + "px";
        d.style.left = a;
        e && (e.style.left = a, e.style.height =
        c.height + "px", e.style.width = c.width + "px");
        return !(h || k)
    };
    var Pq = function(a, b, c) {
        a = m.S(a) + m.Zk(b);
        c && (a += "-" + c);
        return a
    };
    var fda = function(a, b) {
        var c = null;
        m.Ne && m.A(b, m.S(a, "masked")) && ((c = m.F("yt-uix-tooltip-shared-mask")) ? (c.parentNode.removeChild(c), m.K(c)) : (c = window.document.createElement("iframe"), c.src = 'javascript:""', c.id = "yt-uix-tooltip-shared-mask", c.className = m.S(a, "tip-mask")));
        return c
    };
    var gda = function(a) {
        var b = m.F("yt-uix-tooltip-shared-mask"), c = b && m.Zb(b, function(b) {
            return b == a
        }, !1, 2);
        b && c && (b.parentNode.removeChild(b), m.P(b), window.document.body.appendChild(b))
    };
    m.Rq = function() {
        m.Ko.call(this)
    };
    m.Sq = function(a, b) {
        return !!a.getData(b, "is-subscribed")
    };
    m.Tq = function(a, b) {
        return !!a.getData(b, "ypc-enabled")
    };
    var hda = function(a, b) {
        var c = a.getData(b, Uq.pz), d = m.Sq(a, b), c = "-" + c, e = Vq.lz + c;
        m.rb(b, Vq.kz + c, !d);
        m.rb(b, e, d);
        a.getData(b, Uq.nz)&&!a.getData(b, Uq.mz) && (c = m.S(m.Mq.getInstance()), m.rb(b, c, !d), b.title = d ? "" : a.getData(b, Uq.oz));
        d ? m.v(function() {
            m.C(b, Vq.lp)
        }, 1E3) : m.D(b, Vq.lp)
    };
    var ida = function(a, b) {
        var c = m.M(m.S(a));
        return (0, m.qb)(c, function(a) {
            return b == this.Cg(a)
        }, a)
    };
    var jda = function(a, b) {
        var c = (0, m.p)(function(a) {
            a.discoverable_subscriptions && m.Xa("SUBSCRIBE_EMBED_DISCOVERABLE_SUBSCRIPTIONS", a.discoverable_subscriptions);
            this.pk(b)
        }, a);
        Kq(c, "subscribe")
    };
    var Wq = function() {
        m.Ko.call(this)
    };
    m.Xq = function() {
        m.Ko.call(this)
    };
    var Yq = function(a, b) {
        var c = a.getData(b, "uix-tabs-target-id"), c = m.F(c), d = a.getData(b, "uix-tabs-delay-load-url");
        d && c && (d = kda(d)) && (a.removeData(b, "uix-tabs-delay-load-url"), c.innerHTML = d, m.z("yt-uix-tabs-delayed-content-loaded", c.id));
        return c
    };
    var kda = function(a) {
        var b;
        m.R(a, {
            R: function(a) {
                b = a.responseText
            }
        });
        return b
    };
    var Zq = function() {
        m.Ko.call(this)
    };
    m.$q = function(a) {
        a = a.getInstance();
        var b = m.S(a);
        b in ar || (a.register(), a.ca("yt-uix-init-" + b, a.init), a.ca("yt-uix-dispose-" + b, a.dispose), ar[b] = a)
    };
    m.br = function(a) {
        a = a.getInstance();
        var b = m.S(a);
        b in ar && (a.unregister(), delete ar[b])
    };
    var cr = function(a) {
        a = Lo(a.getInstance());
        a in ar && m.ob("yt-uix-dispose-" + a)
    };
    m.dr = function(a) {
        a.init = function() {
            m.$q(a)
        };
        a.dispose = function() {
            cr(a)
        }
    };
    var er = function(a) {
        fr("add_to_watch_later_list", a)
    };
    var gr = function(a) {
        fr("delete_from_watch_later_list", a)
    };
    m.hr = function(a) {
        fr("delete_from_playlist", a)
    };
    var fr = function(a, b) {
        m.R("/playlist_video_ajax?action_" + a + "=1", {
            method: "POST",
            Z: {
                feature: b.Bm || null,
                authuser: b.ok || null,
                pageid: b.pageId || null
            },
            V: {
                video_ids: b.videoIds || null,
                source_playlist_id: b.sourcePlaylistId || null,
                full_list_id: b.ak || null,
                delete_from_playlists: b.Yw || null,
                add_to_playlists: b.Yc || null,
                plid: m.u("PLAYBACK_ID") || null
            },
            context: b.context,
            onError: b.onError,
            R: b.R,
            ka: b.ka
        })
    };
    m.ir = function(a) {
        m.t.call(this);
        this.aa = a;
        this.g = {}
    };
    m.jr = function(a, b, c, d, e, f) {
        m.ha(c) || (c && (kr[0] = c.toString()), c = kr);
        for (var h = 0; h < c.length; h++) {
            var k = m.im(b, c[h], d || a.handleEvent, e ||!1, f || a.aa || a);
            if (!k)
                break;
            a.g[k.key] = k
        }
        return a
    };
    m.lr = function(a, b, c, d, e, f) {
        if (m.ha(c))
            for (var h = 0; h < c.length; h++)
                m.lr(a, b, c[h], d, e, f);
        else {
            b = m.qm(b, c, d || a.handleEvent, e, f || a.aa || a);
            if (!b)
                return a;
            a.g[b.key] = b
        }
        return a
    };
    m.mr = function() {};
    m.nr = function(a) {
        m.ym.call(this);
        this.g = a || m.Ii();
        this.ta = null;
        this.Va=!1;
        this.j = null;
        this.S = void 0;
        this.L = this.D = this.H = null;
        this.Yb=!1
    };
    m.or = function(a) {
        a.S || (a.S = new m.ir(a));
        return a.S
    };
    var pr = function(a, b) {
        if (a == b)
            throw Error("Unable to set parent component");
        var c;
        if (c = b && a.H && a.ta) {
            c = a.H;
            var d = a.ta;
            c = c.L && d ? Al(c.L, d) || null : null
        }
        if (c && a.H != b)
            throw Error("Unable to set parent component");
        a.H = b;
        m.nr.K.bi.call(a, b)
    };
    m.qr = function(a, b, c) {
        if (b.Va&&!a.Va)
            throw Error("Component already rendered");
        if (0 > c || c > (a.D ? a.D.length : 0))
            throw Error("Child component index out of bounds");
        a.L && a.D || (a.L = {}, a.D = []);
        if (b.H == a) {
            var d = b.getId();
            a.L[d] = b;
            m.Oa(a.D, b)
        } else 
            m.xb(a.L, b.getId(), b);
        pr(b, a);
        Fj(a.D, c, 0, b);
        b.Va && a.Va && b.H == a ? (a = a.Sg(), a.insertBefore(b.M(), a.childNodes[c] || null)) : a.Va&&!b.Va && b.j && b.j.parentNode && 1 == b.j.parentNode.nodeType && b.W()
    };
    var rr = function(a, b) {
        a.D && (0, m.y)(a.D, b, void 0)
    };
    m.sr = function(a) {
        m.nr.call(this, a);
        this.ja = [];
        this.qa = []
    };
    m.tr = function(a, b, c, d, e, f) {
        e = e || a;
        f = f || a.M();
        b = m.O(f, b, (0, m.p)(d, e), c);
        a.ja.push(b)
    };
    m.ur = function(a, b) {
        var c = [];
        a.init = m.q(lda, a, b, c);
        a.dispose = m.q(mda, c)
    };
    var lda = function(a, b, c) {
        0 < c.length || (b = m.M(b), (0, m.y)(b, function(b) {
            var e = new a;
            e.oa(b);
            c.push(e)
        }))
    };
    var mda = function(a) {
        (0, m.y)(a, function(a) {
            a.dispose()
        });
        a.length = 0
    };
    m.vr = function(a) {
        m.sr.call(this);
        this.F = a;
        this.ba = 0;
        this.O = this.T = this.J = this.C = this.k = this.X = null
    };
    var wr = function(a) {
        var b = a.k.value.replace(/^\s+|\s+$/, "").length;
        return 0 < b && (!a.ba || b <= a.ba)
    };
    var xr = function(a, b) {
        var c = m.ap(m.Oo.getInstance(), a.C);
        m.pl(c, (0, m.Fa)(m.ml(b)));
        c = m.G("yt-uix-button-menu-item-selected", a.X);
        m.D(c, "yt-uix-button-menu-item-selected");
        m.C(b, "yt-uix-button-menu-item-selected");
        c = m.I(b, "privacy-state-of-menu");
        m.fl(a.C, "privacy-state", c)
    };
    var yr = function(a, b) {
        m.Wk(a.J, b);
        m.Wk(a.T, b);
        m.Wk(a.k, b);
        m.Wk(a.C, b)
    };
    var zr = function(a) {
        a.k.value = "";
        var b = a.P("title-input-container");
        m.Bp(b);
        m.Wk(a.C, !0);
        (b = m.G("yt-uix-button-menu-item-selected", a.X)) && a.O && b != a.O && xr(a, a.O);
        m.Wk(a.J, !1);
        m.Wk(a.T, !0);
        m.Wk(a.k, !0)
    };
    m.Ar = function(a) {
        m.sr.call(this);
        this.F = a.sourcePlaylistId || "";
        this.k = a.videoIds || "";
        this.T = a.Bm || "";
        this.O = "";
        this.J=!!a.CA;
        this.U=!!a.Su;
        this.C = null
    };
    m.Br = function(a, b) {
        a.k != b && (a.k = b, Cr(a))
    };
    var Cr = function(a, b) {
        if (a.k || a.F) {
            var c;
            a.J && (c = Dr(a));
            var d = {
                video_ids: a.k,
                src_playlist_id: a.F
            }, e = {
                hide_watch_later: a.U
            };
            a.T && (e.feature = a.T);
            m.R("/addto_ajax?action_get_dropdown=1", {
                format: "XML",
                method: "POST",
                Z: e,
                V: d,
                R: m.q(a.Yv, c, b),
                context: a
            })
        }
    };
    var nda = function(a, b) {
        var c = a.Ja("addto-playlist-item");
        (0, m.y)(c, function(a) {
            var c = m.I(a, "full-list-id");
            - 1 < (0, m.Na)(b.re, c) ? m.C(a, "to-be-removed") : - 1 < (0, m.Na)(b.Yc, c) && m.C(a, "to-be-added")
        })
    };
    var Er = function(a) {
        var b = m.Ra(a.Ja("contains-all-selected-videos")), c = m.Ra(a.Ja("contains-some-selected-videos")), d = m.Ra(a.Ja("to-be-added")), e = m.Ra(a.Ja("to-be-removed")), f = Boolean(d.length + e.length), b = m.fj(b, c, d), b = (0, m.qb)(b, function(a) {
            return - 1 == (0, m.Na)(e, a)
        }), b = (0, m.sh)(b, function(a) {
            return {
                id: m.I(a, "full-list-id"),
                name: m.I(a, "item-name"),
                url: m.I(a, "url")
            }
        });
        m.z("addto-menu-changed", {
            videoIds: a.k,
            selectedPlaylists: b,
            hasUnsavedChanges: f
        })
    };
    var Dr = function(a) {
        var b = [], c = [], d = a.Ja("addto-playlist-item");
        (0, m.y)(d, function(a) {
            m.A(a, "to-be-removed") && b.push(Fr(a));
            m.A(a, "to-be-added") && c.push(Fr(a))
        }, a);
        return {
            re: b,
            Yc: c
        }
    };
    var Fr = function(a) {
        var b = m.I(a, "full-list-id"), c = m.I(a, "url");
        a = m.I(a, "item-name");
        return {
            id: b,
            url: c,
            title: a
        }
    };
    var Gr = function(a, b, c) {
        var d = Dr(a);
        1 > d.re.length + d.Yc.length || (1 < d.re.length + d.Yc.length ? oda(a, d.re, d.Yc, b, c) : d.re.length ? pda(a, d.re[0], b, c) : d.Yc.length && qda(a, d.Yc[0], b, c), a = a.Ja("addto-playlist-item"), (0, m.y)(a, function(a) {
            var b = m.I(a, "full-list-id");
            ( - 1 < (0, m.Na)(d.re, b)||-1 < (0, m.Na)(d.Yc, b)) && m.C(a, "loading")
        }))
    };
    var oda = function(a, b, c, d, e) {
        b = (0, m.sh)(b, Hr);
        c = (0, m.sh)(c, Hr);
        a = {
            videoIds: a.k,
            Yw: b,
            Yc: c,
            R: (0, m.p)(a.nx, a, d),
            onError: (0, m.p)(a.ck, a, e),
            context: a
        };
        fr("update_playlists_videos", a)
    };
    var qda = function(a, b, c, d) {
        b = {
            ak: Hr(b),
            R: (0, m.p)(a.$j, a, c, !0, b),
            onError: (0, m.p)(a.ck, a, d),
            context: a
        };
        a.k ? (b.videoIds = a.k, fr("add_to_playlist", b)) : a.F && (b.sourcePlaylistId = a.F, fr("copy_from_playlist", b))
    };
    var pda = function(a, b, c, d) {
        m.z("yt-uix-videoactionmenu-hide");
        b = {
            ak: Hr(b),
            R: (0, m.p)(a.$j, a, c, !1, b),
            onError: (0, m.p)(a.ck, a, d),
            context: a
        };
        a.k ? (b.videoIds = a.k, m.hr(b)) : a.F && (b.sourcePlaylistId = a.F, fr("uncopy_from_playlist", b))
    };
    var Ir = function(a, b) {
        var c, d = a.P("create-playlist-item"), e = a.P("create-playlist-panel");
        switch (b) {
        case "create-playlist-item":
            c = "addto-menu-show-create-playlist-panel";
            m.D(e, "active-panel");
            m.C(d, "active-panel");
            break;
        case "create-playlist-form":
            c = "addto-menu-show-create-playlist-panel";
            m.D(d, "active-panel");
            m.C(e, "active-panel");
            break;
        default:
            throw "Invalid panel id";
        }
        m.z(c, a.M())
    };
    var Hr = function(a) {
        return a.id
    };
    var Jr = function() {
        m.Ko.call(this)
    };
    m.Kr = function() {
        Lr || (Lr = new m.pe(function(a) {
            try {
                m.Sn("client", m.q(rda, a))
            } catch (b) {
                m.db(b)
            }
        }))
    };
    var rda = function(a) {
        var b = m.n("gapi.config.update");
        b("googleapis.config/auth/useFirstPartyAuth", !0);
        var c = m.u("APIARY_HOST");
        m.rk(m.ok(c)) || b("googleapis.config/root", window.location.protocol + "//" + c);
        c = m.u("APIARY_HOST_FIRSTPARTY");
        m.rk(m.ok(c)) || b("googleapis.config/root-1p", window.location.protocol + "//" + c);
        b("googleapis.config/sessionIndex", m.u("SESSION_INDEX"));
        m.n("gapi.client.setApiKey")(m.u("INNERTUBE_API_KEY"));
        Mr=!0;
        a()
    };
    m.Nr = function() {
        var a = {
            client: {
                hl: m.u("INNERTUBE_CONTEXT_HL"),
                gl: m.u("INNERTUBE_CONTEXT_GL"),
                clientName: "WEB",
                clientVersion: m.u("INNERTUBE_CONTEXT_CLIENT_VERSION")
            }
        };
        m.u("DELEGATED_SESSION_ID") && (a.user = {
            onBehalfOfUser: m.u("DELEGATED_SESSION_ID")
        });
        return a
    };
    var sda = function(a, b, c) {
        Lr || m.Kr();
        var d = {
            path: "/youtubei/" + m.u("INNERTUBE_API_VERSION") + "/" + a,
            headers: {
                "X-Goog-Visitor-Id": m.u("VISITOR_DATA")
            },
            method: "POST",
            body: m.ag(b)
        };
        Lr.then(function() {
            m.n("gapi.client.request")(d).execute(c || m.ea)
        })
    };
    m.Or = function(a, b, c) {
        var d, e=!1;
        0 < c.timeout && (d = m.v(function() {
            e || (e=!0, c.Zb && c.Zb())
        }, c.timeout));
        sda(a, b, function(a) {
            if (!e)
                if (e=!0, d && m.bb(d), a)
                    c.R && c.R(a);
                else if (c.onError)
                    c.onError()
        })
    };
    var tda = function() {
        m.n("yt.dom.VisibilityMonitor.delegateByClass")(null, m.n("yt.dom.VisibilityMonitor.States.VISIBLE"), uda, "vve-check");
        m.n("yt.dom.VisibilityMonitor.refresh")()
    };
    var uda = function(a) {
        (0, m.y)(a, vda);
        Pr ? (m.bb(Qr), Qr = m.v(Rr, 5E3)) : (Sr = m.x("navigate", Rr), Pr=!0, Rr())
    };
    var vda = function(a) {
        m.D(a, "vve-check");
        Tr.push(a)
    };
    var Rr = function() {
        if (0 != Tr.length && Mr) {
            var a = [];
            (0, m.y)(Tr, function(b) {
                (b = m.I(b, "visibility-tracking")) && a.push(b)
            });
            if (0 != a.length) {
                var b = {
                    context: m.Nr()
                };
                b.interactions = [{
                    visibilityUpdate: {
                        requestTrackingParams: a,
                        csn: m.u("EVENT_ID")
                    }
                }
                ];
                m.Or("log_interaction", b, {});
                Tr.length = 0
            }
        }
    };
    var Ur = function() {
        this.k = []
    };
    var Vr = function(a, b, c, d, e) {
        a.k.push(m.O(b, c, (0, m.p)(d, a), e))
    };
    var Wr = function(a) {
        if (!a)
            return !1;
        var b = a.redirect_url;
        if (!b)
            try {
                var c = to(a), b = vo(c, "redirect_url")
        } catch (d) {
            b = null
        }
        if (!b)
            return !1;
        m.ae(b);
        return !0
    };
    var Xr = function(a, b, c, d, e, f, h) {
        a = {
            format: b,
            method: "POST",
            onError: h || (0, m.p)(a.di, a),
            ka: m.ea,
            R: f || m.ea,
            V: e || {},
            Z: d || {}
        };
        m.R(c, a)
    };
    m.Yr = function() {
        this.k = [];
        this.F = this.D = "";
        this.H = null;
        this.L = this.C=!1;
        this.J = null;
        this.O = this.S = ""
    };
    var Zr = function(a, b) {
        switch (b) {
        case "content":
            a.ia.setState("content");
            break;
        case "loading":
            a.ia.setState("loading");
            break;
        case "working":
            a.ia.setState("working")
        }
    };
    var $r = function(a, b) {
        if (a.C) {
            var c = b || a.O;
            c && (a.F ? m.Dl(a.g, a.F, c) : m.C(a.g, c), a.F = c)
        }
    };
    var as = function() {
        this.k = [];
        this.F = this.C = null;
        this.D = this.qb=!1
    };
    var bs = function(a) {
        var b = m.oa(a), c = cs[b];
        c || (c = new a, cs[b] = c);
        return c
    };
    var ds = function(a) {
        a = m.oa(a);
        cs[a] && (cs[a].dispose(), delete cs[a])
    };
    var es = function() {
        as.call(this);
        this.j = this.g = null;
        this.L=!1
    };
    var fs = function(a, b) {
        var c;
        c = bs(es);
        b && (c.g = b);
        c.init(m.u("CREATE_CHANNEL_CSS_URL"), m.u("CREATE_CHANNEL_JS_URL"), !a, m.F("body-container"), "create-channel-lightbox")
    };
    m.gs = function(a, b, c) {
        m.hs("yt-alert-error", a, b, c)
    };
    m.hs = function(a, b, c, d) {
        if (d) {
            c.removeAttribute("id");
            m.Gl(c, wda);
            m.C(c, a);
            a = m.G("yt-alert-message", c);
            var e = m.G("yt-alert-content", c);
            (a || e).innerHTML = b;
            d.appendChild(c);
            m.K(c);
            return c
        }
        return null
    };
    var is = function(a, b) {
        this.ia = new m.hq(a, !0);
        this.j = b;
        this.k = this.D = this.C = this.F = ""
    };
    var js = function() {
        ks.push(m.x("SHOW-FEED-PRIVACY-COMMENT-DIALOG", xda));
        ks.push(m.x("SHOW-FEED-PRIVACY-FAVORITE-DIALOG", yda));
        ks.push(m.x("SHOW-FEED-PRIVACY-LIKE-DIALOG", zda));
        ks.push(m.x("SHOW-FEED-PRIVACY-ADD-TO-PLAYLIST-DIALOG", Ada));
        ks.push(m.x("SHOW-FEED-PRIVACY-LIKE-PLAYLIST-DIALOG", Bda));
        ks.push(m.x("SHOW-FEED-PRIVACY-SUBSCRIBE-DIALOG", Cda))
    };
    var ls = function(a, b, c, d, e) {
        var f = m.F("feed-privacy-lb");
        f && (ms = new is(f, a), ms.F = b || "", ms.k = c || "", ms.C = d || "", ms.D = e || "", a = ms, m.Rf(m.u("FEED_PRIVACY_CSS_URL")), ns(a, null, {
            channel_id: a.C,
            setting_type: a.j,
            time_created: a.k,
            playlist_id: a.D,
            video_id: a.F
        }))
    };
    var Cda = function(a) {
        ls("SUBSCRIPTIONS", void 0, void 0, a)
    };
    var xda = function(a, b) {
        ls("COMMENTS", a, b)
    };
    var zda = function(a) {
        ls("LIKES", a)
    };
    var yda = function(a) {
        ls("FAVORITES", a)
    };
    var Ada = function(a, b) {
        ls("PLAYLISTS", a, void 0, void 0, b)
    };
    var Bda = function(a) {
        ls("LIKE_PLAYLISTS", void 0, void 0, void 0, a)
    };
    var ns = function(a, b, c) {
        b = b || {};
        c = c || {};
        c.session_token = m.Kl();
        a = {
            method: "POST",
            format: "XML",
            Z: b,
            V: c,
            R: (0, m.p)(a.H, a),
            onError: (0, m.p)(a.L, a)
        };
        m.R("/feed_privacy_ajax", a)
    };
    var os = function() {
        m.Yr.call(this);
        m.Rf(m.u("IDENTITY_PROMPT_CSS_URL"))
    };
    var ps = function() {
        var a = m.F("body-container");
        qs.push(m.O(a, "mousedown", rs, "link-gplus-lightbox"));
        qs.push(m.O(a, "click", Dda, "link-gplus-lightbox"));
        ss.push(m.x("LINK-GPLUS-LOADER-DISMISS", Eda));
        ss.push(m.x("LINK-GPLUS-LOADER-CANCEL", Fda));
        ss.push(m.x("LINK-GPLUS-LOADER-GOTO-CONTENT-STATE", Gda));
        ss.push(m.x("LINK-GPLUS-LOADER-GOTO-WORKING-STATE", Hda));
        ss.push(m.x("LINK-GPLUS-LOADER-SET-WAIT-CURSOR", ts));
        ss.push(m.x("LINK-GPLUS-LOADER-SHOW-DIALOG", Ida));
        m.u("SHOW_LINK_GPLUS_LIGHTBOX") && (rs(), us())
    };
    var Eda = function() {
        vs=!0;
        m.jq(ws, "cancel")
    };
    var Fda = function() {
        m.jq(ws, "cancel")
    };
    var Gda = function() {
        ws.setState("content")
    };
    var Hda = function() {
        ws.setState("working")
    };
    var ts = function(a) {
        a ? window.document.body.style.cursor = "wait" : "wait" == window.document.body.style.cursor && (window.document.body.style.cursor = "default")
    };
    var rs = function() {
        if (!m.F("link-gplus-css")) {
            m.rd(m.u("LINK_GPLUS_JS_URL"), Jda);
            var a = m.u("LINK_GPLUS_CSS_URL"), b = m.Mb("head", void 0, void 0)[0], a = m.Pb("link", {
                id: "link-gplus-css",
                rel: "stylesheet",
                href: a
            });
            m.vl(b, a, 0)
        }
    };
    var Jda = function() {
        xs=!0;
        ys && xs && zs()
    };
    var Dda = function(a) {
        var b = m.L(a.target, "link-gplus-lightbox");
        As = m.A(b, "ignore-opt-out");
        Bs = m.I(b, "upsell");
        a.preventDefault();
        Cs()
    };
    var us = function(a) {
        if (a)
            As=!0, Bs = a;
        else if (As=!1, Bs = "signin", (a = m.u("ID_MERGE_FEATURE_TYPE")) && (Bs = a), "channel" == a || "settings" == a)
            As=!0;
        Cs()
    };
    var Cs = function() {
        if (!ws) {
            var a = m.F("link-gplus-lb");
            if (!a)
                return;
            ws = new m.hq(a, !0)
        }
        vs ? "upload" == Bs && m.ae("/upload") : (ys=!0, ts(!0), xs && zs())
    };
    var zs = function() {
        var a = "";
        if ("upload" == Bs)
            a = "/upload";
        else if ("settings" == Bs)
            a = "/account";
        else if ("fans" == Bs)
            a = "/audience";
        else if ("active_signin" == Bs || "channel" == Bs || "comment" == Bs || "plus_page" == Bs)
            a = m.u("LINK_GPLUS_NEXT_URL");
        m.n("yt.www.account.LinkGplusDialog.fetchAndShow")(Bs, a, As)
    };
    var Ida = function() {
        ws.setState("content");
        ws.show();
        var a = m.G("yt-dialog-fg", m.F("link-gplus-lb")), b = xk(a);
        a.style.position = "fixed";
        a.style.top = "95px";
        b.x && (a.style.left = b.x + "px")
    };
    m.Ds = function() {
        m.u("SERVER_SIDE_AD_REQUEST") && m.R("//googleads.g.doubleclick.net/pagead/id", {
            format: "RAW",
            method: "GET",
            R: Kda
        }) && (Es = m.v(m.Ds, 18E5))
    };
    var Kda = function(a) {
        if (a = a.responseText) {
            a = (a = a.match(/{"id": "(.*)"}/)) && a[1] ? a[1] : "";
            var b;
            t:
            {
                try {
                    b = window.top.location.href
                } catch (c) {
                    b = 2;
                    break t
                }
                b = null != b ? b == window.document.location.href ? 0 : 1 : 2
            }
            b = {
                bid: a,
                dt: Lda,
                frm: b
            };
            b.u_tz =- (new Date).getTimezoneOffset();
            var d;
            a = window;
            try {
                d = a.history.length
            } catch (e) {
                d = 0
            }
            b.u_his = d;
            b.u_java = window.navigator.javaEnabled();
            window.screen && (b.u_h = window.screen.height, b.u_w = window.screen.width, b.u_ah = window.screen.availHeight, b.u_aw = window.screen.availWidth, b.u_cd =
            window.screen.colorDepth);
            window.navigator.plugins && (b.u_nplug = window.navigator.plugins.length);
            window.navigator.mimeTypes && (b.u_nmime = window.navigator.mimeTypes.length);
            m.Daa ? (b.flash = m.Kh, b.ca_type = "flash") : b.ca_type = "image";
            m.ro("/ad_data_204", m.Kl(), {
                V: b
            })
        }
    };
    var Mda = function(a) {
        Fs("keyboard");
        27 == a.keyCode && window.document.activeElement && window.document.activeElement.blur()
    };
    var Gs = function() {
        Fs("mouse")
    };
    var Fs = function(a) {
        Hs !== a && (Hs = a, m.N(Is), Is.length = 0, "keyboard" == Hs ? (Js(!0), Is = [m.J(window, "click", Gs), m.J(window, "mousemove", Gs)]) : "mouse" == Hs && (Js(!1), Is = [m.J(window, "keydown", Mda)]))
    };
    var Js = function(a) {
        m.rb(window.document.documentElement, "no-focus-outline", !a)
    };
    var Ks = function() {
        var a = m.u("GUIDE_SELECTED_ITEM"), b = m.n("yt.www.guide.selectGuideItem");
        b && b(a)
    };
    var Nda = function() {
        (0, m.y)(Ls, function(a) {
            a && a.abort()
        });
        Ls.length = 0;
        Ms.length = 0;
        m.nb(Ns);
        Ns.length = 0
    };
    m.Os = function(a, b, c, d, e) {
        e || (a = Oda(a));
        a.length && (c = c || {}, c.frags = a.join(","), c = m.rc("/watch_fragments_ajax", c), e = m.qc({
            session_token: m.Kl(),
            client_url: window.location.href
        }), Ls.push(window.spf.load(c, {
            method: "POST",
            postData: e,
            onDone: function() {
                if (m.La(a, "guide")) {
                    var c = m.n("yt.www.guide.setup");
                    c && c(b);
                    Ks();
                    m.z("appbar-guide-delay-load")
                }
                m.z("yt-www-pageFrameCssNotifications-load");
                d && d()
            }
        })), Ms = m.fj(Ms, a))
    };
    var Ps = function() {
        var a = m.cl(window.location.href);
        a.tr = "nonwatch";
        m.Os(["guide"], !1, a)
    };
    var Qs = function(a) {
        var b = m.n("yt.www.watchtimeline.setupTimeline"), c = {};
        a && (c = {
            v: a
        });
        m.Os(["timeline"], !1, c, b, !0)
    };
    var Oda = function(a) {
        return (0, m.qb)(a, function(a) {
            return !m.La(Ms, a)
        })
    };
    var Rs = function() {
        return m.n("gapi.iframes.getContext")()
    };
    var Ss = function(a, b) {
        this.j = {};
        this.k = a;
        this.g = b;
        var c = (0, m.p)(this.C, this), d = this.g;
        Rs().addOnConnectHandler("ytsubscribe", c, ["ytapi"], d)
    };
    var Ts = function() {
        this.g = null;
        this.j = []
    };
    var Us = function(a) {
        Vs = m.I(a, "video-ids")
    };
    var Pda = function(a) {
        var b = m.bl("/addto_ajax", {
            action_redirect_to_signin_with_add: 1,
            video_ids: Vs,
            next_url: window.document.location
        }), c = m.xl("form");
        c.action = b;
        c.method = "POST";
        b = m.xl("input");
        b.type = "hidden";
        b.name = "session_token";
        b.value = m.Kl();
        c.appendChild(b);
        window.document.body.appendChild(c);
        c.submit();
        a.preventDefault()
    };
    var Ws = function(a) {
        m.Fl(a, "addto-watch-later-button", "addto-watch-later-button-loading");
        m.Nb(a, {
            "aria-pressed": "true"
        });
        var b = m.I(a, "video-ids");
        er({
            videoIds: b,
            R: function(c, d) {
                var e = d.list_id;
                Xs(e, b, a);
                m.z("playlist-addto", b, e)
            },
            onError: function(c, d) {
                6 == d.return_code ? Xs(d.list_id, b, a) : Ys(a, d)
            }
        })
    };
    var Zs = function(a) {
        m.Fl(a, "addto-watch-later-button-success", "addto-watch-later-button-loading");
        var b = m.I(a, "video-ids");
        gr({
            videoIds: b,
            R: function() {
                m.Fl(a, "addto-watch-later-button-loading", "addto-watch-later-button");
                var b = m.hb("ADDTO_WATCH_LATER");
                m.Nq(m.Mq.getInstance(), a, b);
                m.z("WATCH_LATER_VIDEO_REMOVED")
            },
            onError: function(b, d) {
                Ys(a, d)
            }
        })
    };
    var $s = function(a) {
        var b = m.I(a, "video-ids");
        gr({
            videoIds: b,
            R: function(b, d) {
                m.z("WATCH_LATER_VIDEO_REMOVED", a, d.result.video_count)
            },
            onError: function(b, d) {
                Ys(a, d)
            }
        })
    };
    var Xs = function(a, b, c) {
        m.Fl(c, "addto-watch-later-button-loading", "addto-watch-later-button-success");
        var d = m.hb("ADDTO_WATCH_LATER_ADDED");
        m.Nq(m.Mq.getInstance(), c, d);
        m.z("WATCH_LATER_VIDEO_ADDED", a, b.split(","))
    };
    var Ys = function(a, b) {
        m.Fl(a, "addto-watch-later-button-loading", "addto-watch-later-button-error");
        var c = b.error_message || m.hb("ADDTO_WATCH_LATER_ERROR");
        m.Nq(m.Mq.getInstance(), a, c)
    };
    var Qda = function() {
        var a = m.u("WATCH_LATER_TEMPLATE");
        if (a) {
            var b = m.M("yt-uix-simple-thumb-wrap");
            (0, m.y)(b, function(b) {
                if (!m.G("addto-watch-later-button", b)) {
                    var d = m.I(b, "vid");
                    d && (d = a.replace("__VIDEO_ID__", d), m.zi(b, m.Aj(d)))
                }
            })
        }
    };
    var Rda = function(a, b) {
        Lq() ? at(a, "action_set_reminder") : b ? m.ae(b) : bt(a)
    };
    var Sda = function(a) {
        at(a, "action_remove_reminder")
    };
    var bt = function(a) {
        m.z("live-event-reminder-failed", a)
    };
    var at = function(a, b) {
        var c = {
            vid: a
        };
        c[b] = 1;
        c = m.rc("/live_events_reminders_ajax", c);
        m.R(c, {
            method: "POST",
            R: function(c, e) {
                e && e.success ? "action_set_reminder" == b ? m.z("live-event-reminder-set", a) : m.z("live-event-reminder-removed", a) : bt(a)
            },
            Sq: function() {
                bt(a)
            },
            onError: function() {
                bt(a)
            }
        })
    };
    var ct = function(a) {
        this.k = 1E3 / a;
        this.j = null;
        this.g = []
    };
    var dt = function(a, b) {
        m.Pa(a.g, b);
        a.g.length || (m.cb(a.j), delete a.j)
    };
    var et = function(a, b, c, d, e, f, h, k) {
        this.g = a;
        this.D = b;
        this.k = c;
        this.F = d;
        this.C = e;
        this.H = f;
        this.j = h;
        this.L = k
    };
    var ft = function(a, b) {
        if (0 == b)
            return a.g;
        if (1 == b)
            return a.j;
        var c = Cl(a.g, a.k, b), d = Cl(a.k, a.C, b), e = Cl(a.C, a.j, b), c = Cl(c, d, b), d = Cl(d, e, b);
        return Cl(c, d, b)
    };
    var Tda = function(a, b) {
        var c = (b - a.g) / (a.j - a.g);
        if (0 >= c)
            return 0;
        if (1 <= c)
            return 1;
        for (var d = 0, e = 1, f = 0; 8 > f; f++) {
            var h = ft(a, c), k = (ft(a, c + 1E-6) - h) / 1E-6;
            if (1E-6 > Math.abs(h - b))
                return c;
            if (1E-6 > Math.abs(k))
                break;
            else 
                h < b ? d = c : e = c, c -= (h - b) / k
        }
        for (f = 0; 1E-6 < Math.abs(h - b) && 8 > f; f++)
            h < b ? (d = c, c = (c + e) / 2) : (e = c, c = (c + d) / 2), h = ft(a, c);
        return c
    };
    var gt = function(a, b) {
        var c;
        c = Tda(a, b);
        if (0 == c)
            c = a.D;
        else if (1 == c)
            c = a.L;
        else {
            var d = Cl(a.D, a.F, c), e = Cl(a.F, a.H, c), f = Cl(a.H, a.L, c), d = Cl(d, e, c), e = Cl(e, f, c);
            c = Cl(d, e, c)
        }
        return c
    };
    var ht = function(a, b) {
        this.g = new et(0, 0, a.x, a.y, b.x, b.y, 1, 1)
    };
    var Uda = function(a) {
        return a
    };
    var Vda = function(a) {
        return gt(Wda.g, a)
    };
    var Xda = function(a) {
        return gt(Yda.g, a)
    };
    var Zda = function(a) {
        return gt($da.g, a)
    };
    var aea = function(a) {
        return gt(bea.g, a)
    };
    var cea = function(a) {
        switch (a) {
        case "linear":
            return Uda;
        case "ease-in":
            return Xda;
        case "ease-out":
            return Zda;
        case "ease-in-out":
            return aea;
        default:
            return Vda
        }
    };
    var it = function(a, b) {
        var c = b || {};
        this.Fa = a;
        this.duration = c.duration || .25;
        this.Kg = c.ka || null;
        this.Df = c.Df || "ease";
        this.g(c);
        c.DA || this.play()
    };
    var jt = function() {
        if (!m.ca(kt)) {
            var a = window.document.createElement("div");
            m.ca(a.style.MozTransition) ? kt = "Moz" : m.ca(a.style.WebkitTransition) ? kt = "Webkit" : m.ca(a.style.g) ? kt = "O" : kt = null
        }
        return kt
    };
    var lt = function(a, b) {
        it.call(this, a, b)
    };
    var mt = function(a, b) {
        it.call(this, a, b)
    };
    var nt = function(a, b) {
        var c = 1E3 * a.duration;
        a.C = Math.min(a.C + (b - a.D), c);
        a.D = b;
        var d = a.L(a.C / c);
        a.F(d);
        if (c = a.C >= c)
            a.ka(), a.Kg && m.v((0, m.p)(a.Kg, m.da, a), 0);
        return c
    };
    var ot = function(a, b) {
        it.call(this, a, b)
    };
    var pt = function(a, b) {
        it.call(this, a, b)
    };
    m.qt = function(a, b) {
        it.call(this, a, b)
    };
    m.rt = function(a, b, c, d) {
        d = d || {};
        d.start = b;
        d.end = c;
        jt() ? new pt(a, d) : new ot(a, d)
    };
    m.st = function(a, b) {
        m.rt(a, 1, 0, b)
    };
    m.tt = function(a) {
        var b = m.G("email-on-upload", a), c = m.G("is-upload-only", a);
        a = m.G("updates-frequency-menu", a);
        var d=!1, e=!a || m.A(a, "hidden");
        e || "occasionally" != m.I(a, "frequency") || (d=!0);
        var f;
        "INPUT" == c.tagName ? f = c.checked : f = m.ut(c);
        return {
            "email-on-upload": b.checked&&!d,
            "receive-no-updates": e?!1: !b.checked,
            "is-upload-only": f
        }
    };
    m.ut = function(a) {
        return "uploads_only" == m.I(a, "preference")
    };
    var vt = function(a) {
        a = m.L(a.currentTarget, "overlay-confirmation-content");
        var b = m.G("updates-frequency-menu", a);
        b && (b.disabled=!m.G("email-on-upload", a).checked)
    };
    var dea = function(a) {
        var b = a.currentTarget;
        a = m.I(b, "frequency");
        var c = m.Oo.getInstance(), b = m.L(b, m.S(c, "menu")), b = m.cp(m.Oo.getInstance(), b);
        m.I(b, "frequency") != a && m.fl(b, "frequency", a)
    };
    var eea = function(a) {
        a = m.rl(a);
        return m.G("yt-dialog", a)
    };
    var fea = function(a, b) {
        wt || (wt = eea(a));
        xt(!0);
        m.kq.getInstance().show(wt);
        var c = {};
        c.session_token = m.Kl();
        c.c = b;
        m.R("/subscription_ajax?action_get_subscription_preferences_overlay=1", {
            method: "POST",
            V: c,
            R: function(a, b) {
                var c = b.content_html;
                xt(!1);
                m.G("subscription-preferences-overlay-content", wt).innerHTML = c;
                var c = m.nq(), h = m.G("overlay-confirmation-save-button", c);
                m.N(zt);
                zt = [m.J(h, "click", gea)];
                zt.push(m.O(c, "click", vt, "email-on-upload"));
                zt.push(m.O(c, "keypressed", vt, "email-on-upload"));
                zt.push(m.O(window.document.body,
                "click", dea, "updates-frequency-choice"))
            },
            onError: function() {
                m.kq.getInstance().hide()
            }
        })
    };
    var xt = function(a) {
        var b = wt, c = m.G("subscription-preferences-overlay-loading", b), b = m.G("subscription-preferences-overlay-content", b);
        m.wf(c, a);
        m.wf(b, !a)
    };
    var gea = function(a) {
        var b = m.L(a.currentTarget, "yt-dialog-fg");
        if (b) {
            a = m.I(a.currentTarget, "channel-external-id");
            var c = m.tt(b);
            m.z("subscription-prefs", a, c["email-on-upload"], c["receive-no-updates"], c["is-upload-only"], function() {
                m.wp(m.G("email-on-upload", b), null["email-on-upload"]);
                m.wp(m.G("is-upload-only", b), null["is-upload-only"])
            });
            m.kq.getInstance().hide()
        }
    };
    m.At = function(a, b) {
        Yn.call(this, 1, arguments);
        this.g = a;
        this.offerId = b || null
    };
    var Bt = function(a) {
        Yn.call(this, 1, arguments);
        this.Ia = a
    };
    var Ct = function(a, b, c) {
        Yn.call(this, 1, arguments);
        this.k = a;
        this.j = b;
        this.g = c
    };
    var Dt = function(a, b, c, d) {
        Yn.call(this, 1, arguments);
        this.k = a;
        this.g = b;
        this.C = c || null;
        this.j = d || null
    };
    var Et = function(a, b) {
        Yn.call(this, 1, arguments);
        this.j = a;
        this.g = b || null
    };
    var Ft = function(a) {
        Yn.call(this, 1, arguments);
        this.g = a
    };
    var Gt = function(a) {
        a.g ? Ht ? m.Dq(It, a) : m.Dq(Jt, new Bt(function() {
            m.Dq(Kt, new Ft(a.g))
        })) : hea(a.j, a.D, a.C, a.k, a.source)
    };
    var Lt = function(a) {
        a.g ? Ht ? m.Dq(Mt, a) : m.Dq(Jt, new Bt(function() {
            m.Dq(iea, new Ft(a.g))
        })) : jea(a.j, a.Ui, a.D, a.C, a.k, a.source)
    };
    var Nt = function(a) {
        if (a.length) {
            var b = Fj(a, 0, 40);
            m.z("subscription-batch-subscribe-loading", b);
            Ot("subscription-subscribe-loading", b);
            var c = {};
            c.session_token = m.Kl();
            c.a = b.join(",");
            var d = function() {
                m.z("subscription-batch-subscribe-loaded", b);
                Ot("subscription-subscribe-loaded", b)
            };
            m.R("/subscription_ajax?action_create_subscription_to_all=1", {
                method: "POST",
                V: c,
                R: function(c, f) {
                    d();
                    var h = f.response, k = h.id;
                    if (m.ha(k) && k.length == b.length) {
                        var r = {}, w = h.channel_info_map;
                        (0, m.y)(k, function(a, c) {
                            var d = b[c],
                            e = w[d];
                            r[d] = a;
                            m.z("subscription-subscribe-success", d, a, null, e)
                        });
                        m.z("subscription-batch-subscribe-success", r);
                        (h = h.guide_notification_html_content) && m.z("guide-add-subscription", h);
                        a.length && Nt(a)
                    }
                },
                onError: function() {
                    d();
                    m.z("subscription-batch-subscribe-failure", b);
                    Ot("subscription-subscribe-failure", b)
                }
            })
        }
    };
    var Pt = function(a) {
        if (a.length) {
            var b = Fj(a, 0, 40);
            m.z("subscription-batch-unsubscribe-loading", b);
            Ot("subscription-unsubscirbe-loading", b);
            var c = {};
            c.session_token = m.Kl();
            c.c = b.join(",");
            var d = function() {
                m.z("subscription-batch-unsubscribe-loaded", b);
                Ot("subscription-unsubscribe-loaded", b)
            };
            m.R("/subscription_ajax?action_remove_subscriptions=1", {
                method: "POST",
                V: c,
                R: function() {
                    d();
                    m.z("subscription-batch-unsubscribe-success", b);
                    Ot("subscription-unsubscribe-success", b);
                    a.length && Pt(a)
                },
                onError: function() {
                    d();
                    m.z("subscription-batch-unsubscribe-failure", b);
                    Ot("subscription-unsubscribe-failure", b)
                }
            })
        }
    };
    var kea = function(a, b, c) {
        Qt(a, b, null, null, c)
    };
    var lea = function(a, b, c) {
        Qt(a, null, null, b, c)
    };
    var mea = function(a, b, c) {
        Rt(a, b, null, c)
    };
    var nea = function(a, b, c) {
        Rt(a, null, b, c)
    };
    var oea = function(a, b, c, d, e) {
        Qt(a, b, c, d, e)
    };
    var pea = function(a) {
        var b = a.k.subscriptionElement || null, c = a.g.subscriptionId, d = a.g.channelInfo;
        c && m.z("subscription-subscribe-success", a.j, c, b, d)
    };
    var qea = function(a) {
        var b = a.g;
        m.tb(a.k, function(a, d) {
            m.z("subscription-subscribe-success", d, a, null, b[d])
        });
        m.z("guide-add-subscription", a.j)
    };
    var rea = function(a) {
        m.z("subscription-unsubscribe-success", a.j.itemId, a.j.subscriptionElement || null);
        a.g && (m.z("subscription-batch-unsubscribe-success", a.g), Ot("subscription-unsubscribe-success", a.g), Ot("subscription-enable-ypc", a.g))
    };
    var hea = function(a, b, c, d, e) {
        m.z("subscription-subscribe-loading", a);
        var f = {};
        f.c = a;
        d && (f.eurl = d);
        e && (f.source = e);
        d = {};
        d.session_token = m.Kl();
        (e = m.u("PLAYBACK_ID")) && (d.plid = e);
        c && St("/subscription_ajax?action_create_subscription_to_channel=1", f, c);
        m.R("/subscription_ajax?action_create_subscription_to_channel=1", {
            method: "POST",
            Z: f,
            V: d,
            R: function(c, d) {
                var e = d.response;
                m.z("subscription-subscribe-success", a, e.id, b, e.channel_info);
                e.show_feed_privacy_dialog && m.z("SHOW-FEED-PRIVACY-SUBSCRIBE-DIALOG", a);
                (e = e.guide_notification_html_content) && m.z("guide-add-subscription", e)
            },
            onError: function() {
                m.z("subscription-subscribe-failure", a)
            },
            ka: function() {
                m.z("subscription-subscribe-loaded", a)
            }
        })
    };
    var jea = function(a, b, c, d, e, f) {
        m.z("subscription-unsubscirbe-loading", a);
        var h = {};
        e && (h.eurl = e);
        f && (h.source = f);
        e = {};
        e.session_token = m.Kl();
        e.c = a;
        e.s = b;
        (b = m.u("PLAYBACK_ID")) && (e.plid = b);
        d && St("/subscription_ajax?action_remove_subscriptions=1", {}, d);
        m.R("/subscription_ajax?action_remove_subscriptions=1", {
            method: "POST",
            Z: h,
            V: e,
            R: function() {
                m.z("subscription-unsubscribe-success", a, c)
            },
            onError: function() {
                m.z("subscription-unsubscribe-failure", a)
            },
            ka: function() {
                m.z("subscription-unsubscribe-loaded", a)
            }
        })
    };
    var Qt = function(a, b, c, d, e) {
        if (null !== b || null !== c || null !== d) {
            var f = {};
            f.session_token = m.Kl();
            a && (f.channel_id = a);
            null === b || (f.email_on_upload = b);
            null === c || (f.receive_no_updates = c);
            null === d || (f.uploads_only = d);
            m.R("/subscription_ajax?action_update_subscription_preferences=1", {
                method: "POST",
                V: f,
                onError: function() {
                    e && e()
                }
            })
        }
    };
    var Rt = function(a, b, c, d) {
        if (a.length && (null !== b || null !== c)) {
            var e = Fj(a, 0, 40);
            m.z("subscription-batch-prefs-loading", e);
            var f = {};
            f.session_token = m.Kl();
            f.s = e.join(",");
            null !== b && (f.email_on_upload = b, f.receive_no_updates=!b);
            null === c || (f.uploads_only = c);
            var h = function() {
                m.z("subscription-batch-prefs-loaded", e)
            };
            m.R("/subscription_ajax?action_update_subscription_preferences_batch=1", {
                method: "POST",
                V: f,
                R: function() {
                    h();
                    m.z("subscription-batch-prefs-success", e);
                    a.length && Rt(a, b, c, d)
                },
                onError: function() {
                    h();
                    d && d();
                    m.z("subscription-batch-prefs-failure", e)
                }
            })
        }
    };
    var Ot = function(a, b) {
        (0, m.y)(b, function(b) {
            m.z(a, b)
        })
    };
    var St = function(a, b, c) {
        a = m.bl(a, b);
        c = m.Yi(c);
        m.Zd(a, c)
    };
    m.Tt = function() {
        this.g = [];
        this.j = []
    };
    m.Ut = function(a) {
        m.Rl(a.g) && (a.g = a.j, a.g.reverse(), a.j = []);
        return a.g.pop()
    };
    var Vt = function(a) {
        Wt();
        Xt.push(a);
        Yt(Xt)
    };
    var Zt = function(a) {
        var b = m.n("yt.mdx.remote.debug.handlers_");
        m.Oa(b || [], a)
    };
    var $t = function(a, b) {
        Wt();
        var c = Xt, d = sea(a, String(b));
        m.Rl(c) ? tea(d) : (Yt(c), (0, m.y)(c, function(a) {
            a(d)
        }))
    };
    var Wt = function() {
        Xt || (Xt = m.n("yt.mdx.remote.debug.handlers_") || [], m.l("yt.mdx.remote.debug.handlers_", Xt, void 0))
    };
    var tea = function(a) {
        var b = (au + 1)%50;
        au = b;
        bu[b] = a;
        cu || (cu = 49 == b)
    };
    var Yt = function(a) {
        var b = bu;
        if (b[0]) {
            var c = au, d = cu ? c: - 1;
            do {
                var d = (d + 1)%50, e = b[d];
                (0, m.y)(a, function(a) {
                    a(e)
                })
            }
            while (d != c);
            bu = Array(50);
            au =- 1;
            cu=!1
        }
    };
    var sea = function(a, b) {
        var c = ((0, m.H)() - uea) / 1E3;
        c.toFixed && (c = c.toFixed(3));
        var d = [];
        d.push("[", c + "s", "] ");
        d.push("[", "yt.mdx.remote", "] ");
        d.push(a + ": " + b, "\n");
        return d.join("")
    };
    var du = function(a) {
        a = a || {};
        this.name = a.name || "";
        this.id = a.id || a.screenId || "";
        this.token = a.token || a.loungeToken || "";
        this.uuid = a.uuid || a.dialId || ""
    };
    var eu = function(a, b) {
        return !!b && (a.id == b || a.uuid == b)
    };
    var fu = function(a, b) {
        return a || b?!a!=!b?!1 : a.id == b.id : !0
    };
    var gu = function(a, b) {
        return a || b?!a!=!b?!1 : a.id == b.id && a.token == b.token && a.name == b.name && a.uuid == b.uuid : !0
    };
    var hu = function(a) {
        return {
            name: a.name,
            screenId: a.id,
            loungeToken: a.token,
            dialId: a.uuid
        }
    };
    var vea = function(a) {
        return new du(a)
    };
    var iu = function(a) {
        return m.ha(a) ? (0, m.sh)(a, vea) : []
    };
    var ju = function(a) {
        return a ? '{name:"' + a.name + '",id:' + a.id.substr(0, 6) + "..,token:" + (a.token ? ".." + a.token.slice( - 6) : "-") + ",uuid:" + (a.uuid ? ".." + a.uuid.slice( - 6) : "-") + "}" : "null"
    };
    var ku = function(a) {
        return m.ha(a) ? "[" + (0, m.sh)(a, ju).join(",") + "]" : "null"
    };
    var wea = function(a) {
        var b = m.rg("yt-remote-cast-last-extension");
        b ? "none" == b ? a(null) : a(b) : lu(0, a)
    };
    var lu = function(a, b) {
        a == mu.length ? (m.og("yt-remote-cast-last-extension", "none"), b(null)) : xea(mu[a], function(c) {
            c ? (c = mu[a], m.og("yt-remote-cast-last-extension", c), b(c)) : lu(a + 1, b)
        })
    };
    var nu = function(a) {
        return "chrome-extension://" + a + "/cast_sender.js"
    };
    var xea = function(a, b) {
        var c = new window.XMLHttpRequest;
        c.onreadystatechange = function() {
            4 == c.readyState && 200 == c.status && b(!0)
        };
        c.onerror = function() {
            b(!1)
        };
        try {
            c.open("GET", nu(a), !0), c.send()
        } catch (d) {
            b(!1)
        }
    };
    var yea = function(a) {
        window.__onGCastApiAvailable = a;
        wea(function(b) {
            if (b) {
                $t("bootstrap", "Found cast extension: " + b);
                m.l("chrome.cast.extensionId", b, void 0);
                var c = window.document.createElement("script");
                c.src = nu(b);
                c.onerror = function() {
                    ou();
                    tk("yt-remote-cast-last-extension");
                    a(!1, "Extension JS failed to load.")
                };
                (window.document.head || window.document.documentElement).appendChild(c)
            } else 
                $t("bootstrap", "No cast extension found"), a(!1, "No cast extension found")
        })
    };
    var ou = function() {
        window.__onGCastApiAvailable && delete window.__onGCastApiAvailable
    };
    var pu = function(a) {
        this.scheme = "https";
        this.port = this.k = "";
        this.g = "/api/lounge";
        this.j=!0;
        a = a || window.document.location.href;
        var b = Number(m.hc(a)[4] || null) || null || "";
        b && (this.port = ":" + b);
        this.k = m.jc(a) || "";
        a = m.Db;
        0 <= a.search("MSIE") && (a = a.match(/MSIE ([\d.]+)/)[1], 0 > m.Ea(a, "10.0") && (this.j=!1))
    };
    var qu = function(a, b, c, d) {
        var e = a.g;
        if (m.ca(d) ? d : a.j)
            e = a.scheme + "://" + a.k + a.port + a.g;
        return m.rc(e + b, c || {})
    };
    var ru = function(a) {
        a && (this.id = a.id || "", this.name = a.name || "", this.activityId = a.activityId || "", this.status = a.status || "UNKNOWN")
    };
    var su = function(a) {
        return {
            id: a.id,
            name: a.name,
            activityId: a.activityId,
            status: a.status
        }
    };
    var tu = function(a) {
        a = a || [];
        return "[" + (0, m.sh)(a, function(a) {
            return a ? a.toString() : "null"
        }).join(",") + "]"
    };
    var uu = function() {
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(a) {
            var b = 16 * Math.random() | 0;
            return ("x" == a ? b : b & 3 | 8).toString(16)
        })
    };
    var zea = function(a, b) {
        return m.Ja(a, function(a) {
            return a.key == b
        })
    };
    var Aea = function(a) {
        return (0, m.sh)(a, function(a) {
            return {
                key: a.id,
                name: a.name
            }
        })
    };
    var vu = function(a) {
        return (0, m.sh)(a, function(a) {
            return su(a)
        })
    };
    var wu = function(a) {
        return (0, m.sh)(a, function(a) {
            return new ru(a)
        })
    };
    var Bea = function(a, b) {
        return a || b ? a && b ? a.id == b.id && a.name == b.name : !1 : !0
    };
    var xu = function(a, b) {
        return m.Ja(a, function(a) {
            return a.id == b
        })
    };
    var yu = function(a, b) {
        return m.Ja(a, function(a) {
            return fu(a, b)
        })
    };
    var zu = function(a, b) {
        return m.Ja(a, function(a) {
            return eu(a, b)
        })
    };
    var Au = function() {
        m.t.call(this);
        this.C = new m.Wa;
        m.Ta(this, this.C)
    };
    var Bu = function(a) {
        Au.call(this);
        this.L = a;
        this.screens = []
    };
    var Cu = function(a, b) {
        var c = a.get(b.uuid) || a.get(b.id);
        if (c) {
            var d = c.name;
            c.id = b.id || c.id;
            c.name = b.name;
            c.token = b.token;
            c.uuid = b.uuid || c.uuid;
            return c.name != d
        }
        a.screens.push(b);
        return !0
    };
    var Cea = function(a, b) {
        var c = a.screens.length != b.length;
        a.screens = (0, m.qb)(a.screens, function(a) {
            return !!yu(b, a)
        });
        for (var d = 0, e = b.length; d < e; d++)
            c = Cu(a, b[d]) || c;
        return c
    };
    var Du = function(a, b) {
        var c = a.screens.length;
        a.screens = (0, m.qb)(a.screens, function(a) {
            return !fu(a, b)
        });
        return a.screens.length < c
    };
    var Eu = function(a, b) {
        this.pf = a;
        this.te = b + "::"
    };
    m.Fu = function(a) {
        var b = new m.Yf;
        return b.isAvailable() ? a ? new Eu(b, a) : b : null
    };
    var Gu = function(a) {
        a && (this.id = a.id || a.name, this.name = a.name, this.app = a.app, this.type = a.type || "REMOTE_CONTROL", this.tA = a.user || "", this.avatar = a.userAvatarUri || "", this.theme = a.theme || "u")
    };
    var Hu = function() {
        var a = Iu(), b = Ju();
        m.La(a, b);
        if (m.Ku()) {
            var c = a, d;
            d = 0;
            for (var e = c.length, f; d < e;) {
                var h = d + e>>1, k;
                k = m.Ej(b, c[h]);
                0 < k ? d = h + 1 : (e = h, f=!k)
            }
            d = f ? d : ~d;
            0 > d && Fj(c, - (d + 1), 0, b)
        }
        a = Dea(a);
        if (m.Rl(a))
            try {
                m.Kk("remote_sid")
        } catch (r) {} else 
            try {
                m.Td("remote_sid", a.join(","))
        } catch (w) {}
    };
    var Iu = function() {
        var a = m.rg("yt-remote-connected-devices") || [];
        m.Nl(a);
        return a
    };
    var Dea = function(a) {
        if (m.Rl(a))
            return [];
        var b = a[0].indexOf("#"), c =- 1 == b ? a[0] : a[0].substring(0, b);
        return (0, m.sh)(a, function(a, b) {
            return 0 == b ? a : a.substring(c.length)
        })
    };
    var Lu = function(a) {
        m.og("yt-remote-connected-devices", a, 86400)
    };
    var Ju = function() {
        if (Mu)
            return Mu;
        var a = m.rg("yt-remote-device-id");
        a || (a = uu(), m.og("yt-remote-device-id", a, 31536E3));
        for (var b = Iu(), c = 1, d = a; m.La(b, d);)
            c++, d = a + "#" + c;
        return Mu = d
    };
    var Nu = function() {
        return m.rg("yt-remote-session-browser-channel")
    };
    m.Ku = function() {
        return m.rg("yt-remote-session-screen-id")
    };
    var Ou = function(a) {
        5 < a.length && (a = a.slice(a.length - 5));
        var b = (0, m.sh)(Pu(), function(a) {
            return a.loungeToken
        }), c = (0, m.sh)(a, function(a) {
            return a.loungeToken
        });
        (0, m.vh)(c, function(a) {
            return !m.La(b, a)
        }) && Qu();
        m.og("yt-remote-local-screens", a, 31536E3)
    };
    var Pu = function() {
        return m.rg("yt-remote-local-screens") || []
    };
    var Ru = function(a) {
        m.og("yt-remote-load-account-screens", a, 31536E3)
    };
    var Qu = function() {
        m.og("yt-remote-lounge-token-expiration", !0, 86400)
    };
    var Su = function() {
        return !m.rg("yt-remote-lounge-token-expiration")
    };
    var Tu = function(a) {
        m.og("yt-remote-online-screens", a, 60)
    };
    var Uu = function() {
        return m.rg("yt-remote-online-screens") || []
    };
    var Vu = function(a) {
        m.og("yt-remote-online-dial-devices", a, 30)
    };
    var Wu = function() {
        return m.rg("yt-remote-online-dial-devices") || []
    };
    var Eea = function(a, b) {
        m.og("yt-remote-session-browser-channel", a);
        m.og("yt-remote-session-screen-id", b);
        var c = Iu(), d = Ju();
        m.La(c, d) || c.push(d);
        Lu(c);
        Hu()
    };
    var Xu = function(a) {
        a || (tk("yt-remote-session-screen-id"), tk("yt-remote-session-video-id"));
        Hu();
        a = Iu();
        m.Oa(a, Ju());
        Lu(a)
    };
    var Yu = function() {
        if (!Zu) {
            var a = m.Fu();
            a && (Zu = new m.hg(a))
        }
        return Zu?!!Zu.get("yt-remote-use-staging-server") : !1
    };
    var $u = function(a) {
        Bu.call(this, "AccountScreenService");
        this.g = a;
        this.j = window.NaN;
        this.screens = Fea();
        this.info("Initializing with " + ku(this.screens))
    };
    var Gea = function(a, b) {
        return a.length != b.length?!1 : m.Ml(a, b, function(a, b) {
            return a || b?!a!=!b?!1 : a.id == b.id && a.name == b.name : !0
        })
    };
    var Fea = function() {
        var a = iu(Pu()), b = iu(Uu());
        return (0, m.qb)(b, function(b) {
            return !zu(a, b.id)
        })
    };
    var av = function(a, b) {
        var c = qu(a.g, "/screens");
        return b ? c + "/" + b : c
    };
    var bv = function(a, b, c, d) {
        Au.call(this);
        this.L = a;
        this.H = b;
        this.D = c;
        this.F = d;
        this.k = 0;
        this.g = null;
        this.j = window.NaN
    };
    var cv = function(a) {
        Bu.call(this, "LocalScreenService");
        this.j = a;
        this.g = window.NaN;
        dv(this);
        this.info("Initializing with " + ku(this.screens))
    };
    var ev = function(a) {
        if (a.screens.length) {
            var b = (0, m.sh)(a.screens, function(a) {
                return a.id
            }), c = qu(a.j, "/pairing/get_lounge_token_batch");
            a.j.sendRequest("POST", c, {
                screen_ids: b.join(",")
            }, (0, m.p)(a.jx, a), (0, m.p)(a.ix, a))
        }
    };
    var dv = function(a) {
        var b = iu(Pu()), b = (0, m.qb)(b, function(a) {
            return !a.uuid
        });
        return Cea(a, b)
    };
    var fv = function(a, b) {
        Ou((0, m.sh)(a.screens, hu));
        b && Qu()
    };
    var gv = function(a, b) {
        Au.call(this);
        this.F = b;
        this.g = hv(this);
        this.H = a;
        this.k = this.D = window.NaN;
        this.j = null;
        this.L = (0, m.p)(this.Tq, this);
        iv("Initialized with " + m.ag(this.g))
    };
    var jv = function(a, b, c) {
        var d = qu(a.H, "/pairing/get_screen_availability");
        a.H.sendRequest("POST", d, {
            lounge_token: b.token
        }, (0, m.p)(function(a) {
            a = a.screens || [];
            for (var d = 0, h = a.length; d < h; ++d)
                if (a[d].loungeToken == b.token) {
                    c("online" == a[d].status);
                    return 
                }
            c(!1)
        }, a), (0, m.p)(function() {
            c(!1)
        }, a))
    };
    var kv = function(a, b) {
        var c;
        t: if (Dj(b) != Dj(a.g))
            c=!1;
        else {
            c = m.ub(b);
            for (var d = 0, e = c.length; d < e; ++d)
                if (!a.g[c[d]]) {
                    c=!1;
                    break t
                }
            c=!0
        }
        c || (iv("Updated online screens: " + m.ag(a.g)), a.g = b, a.publish("screenChange"));
        Hea(a)
    };
    var lv = function(a, b) {
        (0, window.isNaN)(a.k) || m.bb(a.k);
        var c = (0, m.p)(a.Jj, a), d;
        d = 0;
        b && (d = 2E3 + 8E3 * Math.random());
        d = 0 < a.D && a.D < (0, m.H)() ? 2E4 + d : 1E4 + d;
        a.k = m.v(c, d)
    };
    var iv = function(a) {
        $t("OnlineScreenService", a)
    };
    var mv = function(a) {
        var b = {};
        (0, m.y)(a.F(), function(a) {
            a.token ? b[a.token] = a.id : this.Ea("Requesting availability of screen w/o lounge token.")
        });
        return b
    };
    var hv = function(a) {
        var b = m.rg("yt-remote-online-screen-ids") || "", b = b ? b.split(","): [], c = {};
        a = a.F();
        for (var d = 0, e = a.length; d < e; ++d) {
            var f = a[d].id;
            c[f] = m.La(b, f)
        }
        return c
    };
    var Hea = function(a) {
        var b = m.ub(m.Bl(a.g, function(a) {
            return a
        }));
        m.Nl(b);
        b.length ? m.og("yt-remote-online-screen-ids", b.join(","), 60) : tk("yt-remote-online-screen-ids");
        a = (0, m.qb)(a.F(), function(a) {
            return !!this.g[a.id]
        }, a);
        Tu((0, m.sh)(a, hu))
    };
    var nv = function(a, b) {
        Bu.call(this, "ScreenService");
        this.D = a;
        this.k = this.g = this.j = null;
        this.F = [];
        this.H = {};
        Iea(this, b)
    };
    var ov = function(a, b, c, d, e, f) {
        a.info("getAutomaticScreenByIds " + c + " / " + b);
        c || (c = a.H[b]);
        var h = a.ab();
        if (h = (c ? zu(h, c) : null) || zu(h, b)) {
            h.uuid = b;
            var k = pv(a, h);
            jv(a.k, k, function(a) {
                e(a ? k : null)
            })
        } else 
            c ? Jea(a, c, (0, m.p)(function(a) {
                var f = pv(this, new du({
                    name: d,
                    screenId: c,
                    loungeToken: a,
                    dialId: b || ""
                }));
                jv(this.k, f, function(a) {
                    e(a ? f : null)
                })
            }, a), f) : e(null)
    };
    var Kea = function(a, b, c, d, e, f) {
        a.info("getDialScreenByPairingCode " + b + " / " + c);
        var h = new bv(a.D, b, c, d);
        h.subscribe("pairingComplete", (0, m.p)(function(a) {
            m.Va(h);
            e(pv(this, a))
        }, a));
        h.subscribe("pairingFailed", function(a) {
            m.Va(h);
            f(a)
        });
        h.start();
        return (0, m.p)(h.stop, h)
    };
    var qv = function(a, b) {
        for (var c = 0, d = a.screens.length; c < d; ++c)
            if (a.screens[c].name == b)
                return a.screens[c];
        return null
    };
    var Jea = function(a, b, c, d) {
        a.info("requestLoungeToken_ for " + b);
        var e = {
            V: {
                screen_ids: b
            },
            method: "POST",
            context: a,
            R: function(a, e) {
                var k = e && e.screens || [];
                k[0] && k[0].screenId == b ? c(k[0].loungeToken) : d(Error("Missing lounge token in token response"))
            },
            onError: function() {
                d(Error("Request screen lounge token failed"))
            }
        };
        m.R(qu(a.D, "/pairing/get_lounge_token_batch"), e)
    };
    var rv = function(a) {
        a.g ? a.screens = m.fj(a.g.ab(), (0, m.qb)(a.j.ab(), function(a) {
            return !this.g.contains(a)
        }, a)) : a.screens = a.j.ab();
        for (var b = m.yl(a.H), c = 0, d = a.screens.length; c < d; ++c) {
            var e = a.screens[c];
            e.uuid = b[e.id] || ""
        }
        a.info("Updated manual screens: " + ku(a.screens))
    };
    var Iea = function(a, b) {
        sv(a);
        a.j = new cv(a.D);
        a.j.subscribe("screenChange", (0, m.p)(a.Ef, a));
        b && (a.g = new $u(a.D), a.g.subscribe("screenChange", (0, m.p)(a.Ef, a)));
        rv(a);
        a.F = iu(m.rg("yt-remote-automatic-screen-cache") || []);
        sv(a);
        a.info("Initializing automatic screens: " + ku(a.F));
        a.k = new gv(a.D, (0, m.p)(a.ab, a, !0));
        a.k.subscribe("screenChange", (0, m.p)(function() {
            this.publish("onlineScreenChange")
        }, a))
    };
    var pv = function(a, b) {
        var c = a.get(b.id);
        c ? (c.uuid = b.uuid, b = c) : ((c = zu(a.F, b.uuid)) ? (c.id = b.id, c.token = b.token, b = c) : a.F.push(b), m.og("yt-remote-automatic-screen-cache", (0, m.sh)(a.F, hu)));
        sv(a);
        a.H[b.uuid] = b.id;
        m.og("yt-remote-device-id-map", a.H, 31536E3);
        return b
    };
    var sv = function(a) {
        a.H = m.rg("yt-remote-device-id-map") || {}
    };
    var tv = function(a, b, c) {
        Au.call(this);
        this.U = c;
        this.S = a;
        this.j = b;
        this.k = null
    };
    var uv = function(a, b) {
        a.k = b;
        a.publish("sessionScreen", a.k)
    };
    var vv = function(a, b) {
        tv.call(this, a, b, "CastSession");
        this.g = null;
        this.F = 0;
        this.D = null;
        this.L = (0, m.p)(this.Ot, this);
        this.H = (0, m.p)(this.Nt, this);
        this.F = m.v((0, m.p)(function() {
            wv(this, null)
        }, this), 12E4)
    };
    var xv = function(a) {
        var b = a.D.videoId || a.D.videoIds[a.D.index];
        b && yv(a, "flingVideo", {
            videoId: b,
            currentTime: a.D.currentTime || 0
        });
        a.D = null
    };
    var yv = function(a, b, c) {
        a.info("sendYoutubeMessage_: " + b + " " + m.ag(c));
        var d = {};
        d.type = b;
        c && (d.data = c);
        a.g ? a.g.sendMessage("urn:x-cast:com.google.youtube.mdx", d, m.ea, (0, m.p)(function() {
            this.warn("Failed to send message: " + b + ".")
        }, a)) : a.warn("Sending yt message without session: " + m.ag(d))
    };
    var wv = function(a, b) {
        m.bb(a.F);
        b ? (a.info("onConnectedScreenId_: Received screenId: " + b), a.k && a.k.id == b || ov(a.S, a.j.label, b, a.j.friendlyName, (0, m.p)(function(a) {
            a ? uv(this, a) : this.ub(Error("Unable to fetch screen."))
        }, a), (0, m.p)(a.ub, a))) : a.ub(Error("Waiting for session status timed out."))
    };
    var zv = function(a, b) {
        tv.call(this, a, b, "DialSession");
        this.F = this.J = null;
        this.O = "";
        this.D = null;
        this.L = m.ea;
        this.H = window.NaN;
        this.T = (0, m.p)(this.Js, this);
        this.g = m.ea
    };
    var Av = function(a) {
        a.g = Kea(a.S, a.O, a.j.label, a.j.friendlyName, (0, m.p)(function(a) {
            this.g = m.ea;
            uv(this, a)
        }, a), (0, m.p)(function(a) {
            this.g = m.ea;
            this.ub(a)
        }, a))
    };
    var Bv = function(a) {
        var b = {};
        b.pairingCode = a.O;
        if (a.D) {
            var c = a.D.index || 0, d = a.D.currentTime || 0;
            b.v = a.D.videoId || a.D.videoIds[c];
            b.t = d
        }
        Yu() && (b.env_useStageMdx = 1);
        return m.qc(b)
    };
    var Cv = function(a, b) {
        tv.call(this, a, b, "ManualSession");
        this.g = m.v((0, m.p)(this.Ke, this, null), 150)
    };
    var Dv = function(a) {
        Au.call(this);
        this.j = a;
        this.g = null;
        this.F=!1;
        this.k = [];
        this.D = (0, m.p)(this.jr, this)
    };
    var Ev = function(a, b) {
        return b ? m.Ja(a.k, function(a) {
            return eu(b, a.label)
        }, a) : null
    };
    var Fv = function(a) {
        $t("Controller", a)
    };
    var Gv = function(a) {
        window.chrome && window.chrome.cast && window.chrome.cast.logMessage && window.chrome.cast.logMessage(a)
    };
    var Hv = function(a) {
        return a.F||!!a.k.length||!!a.g
    };
    var Iv = function(a, b, c) {
        m.Va(a.g);
        (a.g = b) ? (c ? a.publish("yt-remote-cast2-receiver-resumed", b.j) : a.publish("yt-remote-cast2-receiver-selected", b.j), b.subscribe("sessionScreen", (0, m.p)(a.Kl, a, b)), b.k ? a.publish("yt-remote-cast2-session-change", b.k) : c && a.g.Ke(null)) : a.publish("yt-remote-cast2-session-change", null)
    };
    var Jv = function(a) {
        var b = a.j.xm(), c = a.g && a.g.j;
        a = (0, m.sh)(b, function(a) {
            c && eu(a, c.label) && (c = null);
            var b = a.uuid ? a.uuid: a.id, f = Ev(this, a);
            f ? (f.label = b, f.friendlyName = a.name) : (f = new window.chrome.cast.Receiver(b, a.name), f.receiverType = window.chrome.cast.ReceiverType.CUSTOM);
            return f
        }, a);
        c && (c.receiverType != window.chrome.cast.ReceiverType.CUSTOM && (c = new window.chrome.cast.Receiver(c.label, c.friendlyName), c.receiverType = window.chrome.cast.ReceiverType.CUSTOM), a.push(c));
        return a
    };
    var Lea = function(a, b) {
        m.Wh ? Mea(a) && (Lv(!0), window.chrome && window.chrome.cast && window.chrome.cast.isAvailable ? Mv(b) : yea(function(a, d) {
            a ? Mv(b) : (Nv("Failed to load cast API: " + d), Ov(!1), Lv(!1), tk("yt-remote-cast-available"), tk("yt-remote-cast-receiver"), Pv(), b(!1))
        })) : Kv("Cannot initialize because not running Chrome")
    };
    var Pv = function() {
        Kv("dispose");
        ou();
        var a = Qv();
        a && a.dispose();
        Rv = null;
        m.l("yt.mdx.remote.cloudview.instance_", null, void 0);
        Sv(!1);
        m.nb(Tv);
        Tv.length = 0
    };
    var Uv = function() {
        return !!m.rg("yt-remote-cast-installed")
    };
    var Vv = function() {
        var a = m.rg("yt-remote-cast-receiver");
        return a ? a.friendlyName : null
    };
    var Wv = function() {
        Kv("clearCurrentReciever");
        tk("yt-remote-cast-receiver")
    };
    var Nea = function() {
        return Uv() ? Qv() ? Rv.getCastSession() : (Nv("getCastSelector: Cast is not initialized."), null) : (Nv("getCastSelector: Cast API is not installed!"), null)
    };
    var Xv = function() {
        Uv() ? Qv() ? Yv() ? (Kv("Requesting cast selector."), Rv.requestSession()) : (Kv("Wait for cast API to be ready to request the session."), Tv.push(m.x("yt-remote-cast2-api-ready", Xv))) : Nv("requestCastSelector: Cast is not initialized.") : Nv("requestCastSelector: Cast API is not installed!")
    };
    var Oea = function(a) {
        Yv() ? Qv().setLaunchParams(a) : Nv("setLaunchParams called before ready.")
    };
    var Zv = function(a, b) {
        Yv() ? Qv().setConnectedScreenStatus(a, b) : Nv("setConnectedScreenStatus called before ready.")
    };
    var Pea = function(a) {
        Rv.init(!1, a)
    };
    var Mea = function(a) {
        var b=!1;
        if (!Rv) {
            var c = m.n("yt.mdx.remote.cloudview.instance_");
            c || (c = new Dv(a), c.subscribe("yt-remote-cast2-availability-change", function(a) {
                m.og("yt-remote-cast-available", a);
                m.z("yt-remote-cast2-availability-change", a)
            }), c.subscribe("yt-remote-cast2-receiver-selected", function(a) {
                Kv("onReceiverSelected: " + a.friendlyName);
                m.og("yt-remote-cast-receiver", a);
                m.z("yt-remote-cast2-receiver-selected", a)
            }), c.subscribe("yt-remote-cast2-receiver-resumed", function(a) {
                Kv("onReceiverResumed: " +
                a.friendlyName);
                m.og("yt-remote-cast-receiver", a)
            }), c.subscribe("yt-remote-cast2-session-change", function(a) {
                Kv("onSessionChange: " + ju(a));
                a || tk("yt-remote-cast-receiver");
                m.z("yt-remote-cast2-session-change", a)
            }), m.l("yt.mdx.remote.cloudview.instance_", c, void 0), b=!0);
            Rv = c
        }
        Kv("cloudview.createSingleton_: " + b);
        return b
    };
    var Qv = function() {
        Rv || (Rv = m.n("yt.mdx.remote.cloudview.instance_"));
        return Rv
    };
    var Mv = function(a) {
        Ov(!0);
        Lv(!1);
        Pea(function(b) {
            b ? (Sv(!0), m.z("yt-remote-cast2-api-ready")) : (Nv("Failed to initialize cast API."), Ov(!1), tk("yt-remote-cast-available"), tk("yt-remote-cast-receiver"), Pv());
            a(b)
        })
    };
    var Kv = function(a) {
        $t("cloudview", a)
    };
    var Nv = function(a) {
        $t("cloudview", a)
    };
    var Ov = function(a) {
        Kv("setCastInstalled_ " + a);
        m.og("yt-remote-cast-installed", a)
    };
    var Yv = function() {
        return !!m.n("yt.mdx.remote.cloudview.apiReady_")
    };
    var Sv = function(a) {
        Kv("setApiReady_ " + a);
        m.l("yt.mdx.remote.cloudview.apiReady_", a, void 0)
    };
    var Lv = function(a) {
        m.l("yt.mdx.remote.cloudview.initializing_", a, void 0)
    };
    var $v = function(a) {
        aw(this, a)
    };
    var bw = function(a, b) {
        if (a.j)
            throw Error(b + " is not allowed in V3.");
    };
    var cw = function(a) {
        a.C =- 1;
        a.k=!1;
        a.F = null;
        a.g =- 1;
        a.D = null;
        a.H = 0;
        a.L = (0, m.H)()
    };
    var aw = function(a, b) {
        a.videoIds = [];
        a.j = "";
        dw(a);
        b && (a.videoIds = b.videoIds, a.index = b.index, a.j = b.listId, a.videoId = b.videoId, a.g = b.playerState, a.D = b.errorReason, a.C = b.volume, a.k = b.muted, a.F = b.trackData, a.H = b.playerTime, a.L = b.playerTimeAt)
    };
    var dw = function(a) {
        a.index =- 1;
        a.videoId = "";
        cw(a)
    };
    var ew = function(a) {
        return a.j ? a.videoId : a.videoIds[a.index]
    };
    var fw = function(a, b) {
        a.H = b;
        a.L = (0, m.H)()
    };
    var gw = function(a) {
        switch (a.g) {
        case 1:
            return ((0, m.H)() - a.L) / 1E3 + a.H;
        case - 1E3:
            return 0
        }
        return a.H
    };
    var hw = function(a, b, c) {
        bw(a, "setPlaylist");
        c = c || ew(a);
        m.Ml(a.videoIds, b) && c == ew(a) || (a.videoIds = m.Ra(b), a.setVideoId(c))
    };
    var iw = function(a) {
        var b = {};
        b.videoIds = m.Ra(a.videoIds);
        b.index = a.index;
        b.listId = a.j;
        b.videoId = a.videoId;
        b.playerState = a.g;
        b.errorReason = a.D;
        b.volume = a.C;
        b.muted = a.k;
        b.trackData = zl(a.F);
        b.playerTime = a.H;
        b.playerTimeAt = a.L;
        return b
    };
    var jw = function(a, b) {
        Au.call(this);
        this.k = 0;
        this.D = a;
        this.L = [];
        this.H = new m.Tt;
        this.F = window.NaN;
        this.j = this.g = null;
        this.O = (0, m.p)(this.Pq, this);
        this.J = (0, m.p)(this.yf, this);
        this.S = (0, m.p)(this.Oq, this);
        var c = 0;
        a ? (c = a.getProxyState(), 3 != c && (a.subscribe("proxyStateChange", this.Qi, this), kw(this))) : c = 3;
        0 != c && (b ? this.Qi(c) : m.v((0, m.p)(function() {
            this.Qi(c)
        }, this), 0));
        lw(this, Nea())
    };
    var mw = function(a) {
        return new $v(a.D.getPlayerContextData())
    };
    var kw = function(a) {
        (0, m.y)(["remotePlayerChange", "remoteQueueChange"], function(a) {
            this.L.push(this.D.subscribe(a, m.q(this.Cx, a), this))
        }, a)
    };
    var nw = function(a) {
        (0, m.y)(a.L, function(a) {
            this.D.unsubscribeByKey(a)
        }, a);
        a.L.length = 0
    };
    var ow = function(a, b) {
        50 > a.H.ua() && a.H.j.push(b)
    };
    var pw = function(a, b, c) {
        var d = mw(a);
        fw(d, c);
        - 1E3 != d.g && (d.g = b);
        qw(a, d)
    };
    var rw = function(a, b, c) {
        a.D.sendMessage(b, c)
    };
    var qw = function(a, b) {
        nw(a);
        a.D.setPlayerContextData(iw(b));
        kw(a)
    };
    var sw = function(a) {
        m.bb(a.F);
        a.F = m.v((0, m.p)(function() {
            this.publish("remotePlayerChange");
            this.F = window.NaN
        }, a), 2E3)
    };
    var lw = function(a, b) {
        a.j && (a.j.removeUpdateListener(a.O), a.j.removeMediaListener(a.J), a.yf(null));
        a.j = b;
        a.j && ($t("CP", "Setting cast session: " + a.j.sessionId), a.j.addUpdateListener(a.O), a.j.addMediaListener(a.J), a.j.media.length && a.yf(a.j.media[0]))
    };
    var tw = function(a) {
        var b = a.g.customData;
        if (a.g.media) {
            var c = a.g.media, d = mw(a);
            c.contentId != d.videoId && $t("CP", "Cast changing video to: " + c.contentId);
            var e = c.customData;
            d.index = e.currentIndex;
            d.j = e.listId;
            d.videoId = c.contentId;
            d.g = b.playerState;
            fw(d, a.g.getEstimatedTime());
            qw(a, d)
        } else 
            $t("CP", "No cast media video. Ignoring state update.")
    };
    var Qea = function(a, b) {
        this.action = a;
        this.params = b || null
    };
    var Rea = function() {
        if (!("cast"in window))
            return !1;
        var a = window.cast || {};
        return "ActivityStatus"in a && "Api"in a && "LaunchRequest"in a && "Receiver"in a
    };
    var uw = function(a) {
        $t("CAST", a)
    };
    var vw = function(a) {
        var b = ww();
        b && b.logMessage && b.logMessage(a)
    };
    var Sea = function(a) {
        if (a.source == window && a.data && "CastApi" == a.data.source && "Hello" == a.data.event)
            for (; xw.length;)
                xw.shift()()
    };
    var yw = function() {
        if (!m.n("yt.mdx.remote.castv2_")&&!zw && (m.Rl(Aw) && Li(Aw, Wu()), Rea())) {
            var a = ww();
            a ? (a.removeReceiverListener("YouTube", Bw), a.addReceiverListener("YouTube", Bw), uw("API initialized in the other binary")) : (a = new window.cast.Api, Cw(a), a.addReceiverListener("YouTube", Bw), a.setReloadTabRequestHandler && a.setReloadTabRequestHandler(function() {
                m.v(function() {
                    window.location.reload(!0)
                }, 1E3)
            }), Vt(vw), uw("API initialized"));
            zw=!0
        }
    };
    var Tea = function() {
        var a = ww();
        a && (uw("API disposed"), Zt(vw), a.setReloadTabRequestHandler && a.setReloadTabRequestHandler(m.ea), a.removeReceiverListener("YouTube", Bw), Cw(null));
        zw=!1;
        xw = null;
        m.Pk(window, "message", Sea)
    };
    var Dw = function(a) {
        var b = m.Ka(Aw, function(b) {
            return b.id == a.id
        });
        0 <= b && (Aw[b] = su(a))
    };
    var Bw = function(a) {
        a.length && uw("Updating receivers: " + m.ag(a));
        Uea(a);
        m.z("yt-remote-cast-device-list-update");
        (0, m.y)(Ew(), function(a) {
            Vea(a.id)
        });
        (0, m.y)(a, function(a) {
            if (a.isTabProjected) {
                var c = Fw(a.id);
                uw("Detected device: " + c.id + " is tab projected. Firing DEVICE_TAB_PROJECTED event.");
                m.v(function() {
                    m.z("yt-remote-cast-device-tab-projected", c.id)
                }, 1E3)
            }
        })
    };
    var Gw = function(a, b) {
        uw("Updating " + a + " activity status: " + m.ag(b));
        var c = Fw(a);
        c ? (b.activityId && (c.activityId = b.activityId), c.status = "running" == b.status ? "RUNNING" : "stopped" == b.status ? "STOPPED" : "error" == b.status ? "ERROR" : "UNKNOWN", "RUNNING" != c.status && (c.activityId = ""), Dw(c), m.z("yt-remote-cast-device-status-update", c)) : uw("Device not found")
    };
    var Ew = function() {
        yw();
        return wu(Aw)
    };
    var Uea = function(a) {
        a = (0, m.sh)(a, function(a) {
            var c = {
                id: a.id,
                name: m.Ul(a.name)
            };
            if (a = Fw(a.id))
                c.activityId = a.activityId, c.status = a.status;
            return c
        });
        m.Ql(Aw);
        Li(Aw, a)
    };
    var Fw = function(a) {
        var b = Ew();
        return m.Ja(b, function(b) {
            return b.id == a
        }) || null
    };
    var Vea = function(a) {
        var b = Fw(a), c = ww();
        c && b && b.activityId && c.getActivityStatus(b.activityId, function(b) {
            "error" == b.status && (b.status = "stopped");
            Gw(a, b)
        })
    };
    var Hw = function(a) {
        yw();
        var b = Fw(a), c = ww();
        c && b && b.activityId ? (uw("Stopping cast activity"), c.stopActivity(b.activityId, m.q(Gw, a))) : uw("Dropping cast activity stop")
    };
    var ww = function() {
        return m.n("yt.mdx.remote.castapi.api_")
    };
    var Cw = function(a) {
        m.l("yt.mdx.remote.castapi.api_", a, void 0)
    };
    var Iw = function() {
        this.g = (0, m.H)()
    };
    var Jw = function(a, b) {
        this.j = new m.cg(a);
        this.g = b ? m.uk : m.$f
    };
    var Kw = function() {};
    var Lw = function(a) {
        var b;
        (b = a.g) || (b = {}, Mw(a) && (b[0]=!0, b[1]=!0), b = a.g = b);
        return b
    };
    var Nw = function() {};
    var Ow = function(a) {
        return (a = Mw(a)) ? new window.ActiveXObject(a) : new window.XMLHttpRequest
    };
    var Mw = function(a) {
        if (!a.j && "undefined" == typeof window.XMLHttpRequest && "undefined" != typeof window.ActiveXObject) {
            for (var b = ["MSXML2.XMLHTTP.6.0", "MSXML2.XMLHTTP.3.0", "MSXML2.XMLHTTP", "Microsoft.XMLHTTP"], c = 0; c < b.length; c++) {
                var d = b[c];
                try {
                    return new window.ActiveXObject(d), a.j = d
                } catch (e) {}
            }
            throw Error("Could not create ActiveXObject. ActiveX might be disabled, or MSXML might not be installed");
        }
        return a.j
    };
    var Pw = function(a, b, c, d, e) {
        this.g = a;
        this.C = c;
        this.H = d;
        this.F = e || 1;
        this.nb = 45E3;
        this.k = new m.ir(this);
        this.j = new m.Am;
        Bm(this.j, 250)
    };
    var Qw = function(a, b, c) {
        a.He = 1;
        a.jd = bk(b.clone());
        a.Kd = c;
        a.D=!0;
        Rw(a, null)
    };
    var Sw = function(a, b, c, d, e) {
        a.He = 1;
        a.jd = bk(b.clone());
        a.Kd = null;
        a.D = c;
        e && (a.Ql=!1);
        Rw(a, d)
    };
    var Rw = function(a, b) {
        a.Lf = (0, m.H)();
        Tw(a);
        a.rd = a.jd.clone();
        m.Zj(a.rd, "t", a.F);
        a.Tf = 0;
        a.Ra = a.g.mj(a.g.Jf() ? b : null);
        0 < a.cj && (a.Vg = new Dm((0, m.p)(a.$l, a, a.Ra), a.cj));
        a.k.listen(a.Ra, "readystatechange", a.st);
        var c = a.Id ? m.yb(a.Id): {};
        a.Kd ? (a.oh = "POST", c["Content-Type"] = "application/x-www-form-urlencoded", a.Ra.send(a.rd, a.oh, a.Kd, c)) : (a.oh = "GET", a.Ql&&!m.Qe && (c.Connection = "close"), a.Ra.send(a.rd, a.oh, null, c));
        a.g.k(1)
    };
    var Uw = function(a, b, c) {
        for (var d=!0; !a.Nd && a.Tf < c.length;) {
            var e = Wea(a, c);
            if (e == Vw) {
                4 == b && (a.hd = 4, Ww(), d=!1);
                break
            } else if (e == Xw) {
                a.hd = 4;
                Ww();
                d=!1;
                break
            } else 
                Yw(a, e)
        }
        4 == b && 0 == c.length && (a.hd = 1, Ww(), d=!1);
        a.dc = a.dc && d;
        d || (Zw(a), $w(a))
    };
    var Wea = function(a, b) {
        var c = a.Tf, d = b.indexOf("\n", c);
        if ( - 1 == d)
            return Vw;
        c = Number(b.substring(c, d));
        if ((0, window.isNaN)(c))
            return Xw;
        d += 1;
        if (d + c > b.length)
            return Vw;
        var e = b.substr(d, c);
        a.Tf = d + c;
        return e
    };
    var ax = function(a, b) {
        a.Lf = (0, m.H)();
        Tw(a);
        var c = b ? window.location.hostname: "";
        a.rd = a.jd.clone();
        m.Yj(a.rd, "DOMAIN", c);
        m.Yj(a.rd, "t", a.F);
        try {
            a.gc = new window.ActiveXObject("htmlfile")
        } catch (d) {
            Zw(a);
            a.hd = 7;
            Ww();
            $w(a);
            return 
        }
        var e = "<html><body>";
        b && (e += '<script>document.domain="' + c + '"\x3c/script>');
        e += "</body></html>";
        a.gc.open();
        a.gc.write(e);
        a.gc.close();
        a.gc.parentWindow.m = (0, m.p)(a.Bt, a);
        a.gc.parentWindow.d = (0, m.p)(a.cm, a, !0);
        a.gc.parentWindow.rpcClose = (0, m.p)(a.cm, a, !1);
        c = a.gc.createElement("div");
        a.gc.parentWindow.document.body.appendChild(c);
        c.innerHTML = '<iframe src="' + a.rd + '"></iframe>';
        a.g.k(1)
    };
    var Tw = function(a) {
        a.Gj = (0, m.H)() + a.nb;
        bx(a, a.nb)
    };
    var bx = function(a, b) {
        if (null != a.bf)
            throw Error("WatchDog timer not null");
        a.bf = cx((0, m.p)(a.yz, a), b)
    };
    var dx = function(a) {
        a.bf && (m.da.clearTimeout(a.bf), a.bf = null)
    };
    var $w = function(a) {
        a.g.Dn() || a.Nd || a.g.Qg(a)
    };
    var Zw = function(a) {
        dx(a);
        m.Va(a.Vg);
        a.Vg = null;
        a.j.stop();
        a.k.removeAll();
        if (a.Ra) {
            var b = a.Ra;
            a.Ra = null;
            b.abort();
            b.dispose()
        }
        a.gc && (a.gc = null)
    };
    var Yw = function(a, b) {
        try {
            a.g.Am(a, b), a.g.k(4)
        } catch (c) {}
    };
    var ex = function(a, b, c, d, e) {
        if (0 == d)
            c(!1);
        else {
            var f = e || 0;
            d--;
            fx(a, b, function(e) {
                e ? c(!0) : m.da.setTimeout(function() {
                    ex(a, b, c, d, f)
                }, f)
            })
        }
    };
    var fx = function(a, b, c) {
        var d = new window.Image;
        d.onload = function() {
            try {
                gx(d), c(!0)
            } catch (a) {}
        };
        d.onerror = function() {
            try {
                gx(d), c(!1)
            } catch (a) {}
        };
        d.onabort = function() {
            try {
                gx(d), c(!1)
            } catch (a) {}
        };
        d.ontimeout = function() {
            try {
                gx(d), c(!1)
            } catch (a) {}
        };
        m.da.setTimeout(function() {
            if (d.ontimeout)
                d.ontimeout()
        }, b);
        d.src = a
    };
    var gx = function(a) {
        a.onload = null;
        a.onerror = null;
        a.onabort = null;
        a.ontimeout = null
    };
    var hx = function(a) {
        this.g = a;
        this.j = new Jw(null, !0)
    };
    var ix = function(a) {
        var b = jx(a.g, a.Kf, "/mail/images/cleardot.gif");
        bk(b);
        ex(b.toString(), 5E3, (0, m.p)(a.su, a), 3, 2E3);
        a.k(1)
    };
    var kx = function(a) {
        var b = a.g.S;
        if (null != b)
            Ww(), b ? (Ww(), lx(a.g, a, !1)) : (Ww(), lx(a.g, a, !0));
        else if (a.Mb = new Pw(a, 0, void 0, void 0, void 0), a.Mb.Id = a.hj, b = a.g, b = jx(b, b.Jf() ? a.Sf : null, a.Qj), Ww(), !m.E || m.Ib(10))
            m.Zj(b, "TYPE", "xmlhttp"), Sw(a.Mb, b, !1, a.Sf, !1);
        else {
            m.Zj(b, "TYPE", "html");
            var c = a.Mb;
            a = Boolean(a.Sf);
            c.He = 3;
            c.jd = bk(b.clone());
            ax(c, a)
        }
    };
    m.mx = function(a) {
        m.ym.call(this);
        this.headers = new m.aj;
        this.T = a || null;
        this.j=!1;
        this.O = this.g = null;
        this.ba = this.L = "";
        this.C = 0;
        this.D = "";
        this.k = this.X = this.H = this.U=!1;
        this.F = 0;
        this.J = null;
        this.ja = "";
        this.S = this.qa=!1
    };
    var Xea = function(a) {
        return m.E && m.Fb(9) && m.ka(a.timeout) && m.ca(a.ontimeout)
    };
    var Yea = function(a) {
        return "content-type" == a.toLowerCase()
    };
    var nx = function(a, b) {
        a.j=!1;
        a.g && (a.k=!0, a.g.abort(), a.k=!1);
        a.D = b;
        a.C = 5;
        ox(a);
        px(a)
    };
    var ox = function(a) {
        a.U || (a.U=!0, a.dispatchEvent("complete"), a.dispatchEvent("error"))
    };
    var qx = function(a) {
        if (a.j && "undefined" != typeof m.rh)
            if (a.O[1] && 4 == rx(a) && 2 == a.getStatus())
                sx(a, "Local request error detected and ignored");
            else if (a.H && 4 == rx(a))
                m.Cm(a.al, 0, a);
            else if (a.dispatchEvent("readystatechange"), 4 == rx(a)) {
                sx(a, "Request complete");
                a.j=!1;
                try {
                    if (a.ve())
                        a.dispatchEvent("complete"), a.dispatchEvent("success");
                    else {
                        a.C = 6;
                        var b;
                        try {
                            b = 2 < rx(a) ? a.g.statusText : ""
                        } catch (c) {
                            b = ""
                        }
                        a.D = b + " [" + a.getStatus() + "]";
                        ox(a)
                    }
                } finally {
                    px(a)
                }
        }
    };
    var px = function(a, b) {
        if (a.g) {
            tx(a);
            var c = a.g, d = a.O[0] ? m.ea: null;
            a.g = null;
            a.O = null;
            b || a.dispatchEvent("ready");
            try {
                c.onreadystatechange = d
            } catch (e) {}
        }
    };
    var tx = function(a) {
        a.g && a.S && (a.g.ontimeout = null);
        m.ka(a.J) && (m.da.clearTimeout(a.J), a.J = null)
    };
    var rx = function(a) {
        return a.g ? a.g.readyState : 0
    };
    m.ux = function(a) {
        try {
            return a.g ? a.g.responseText : ""
        } catch (b) {
            return ""
        }
    };
    var sx = function(a, b) {
        return b + " [" + a.ba + " " + a.L + " " + a.getStatus() + "]"
    };
    var vx = function(a, b, c) {
        this.L = a || null;
        this.g = 1;
        this.j = [];
        this.D = [];
        this.F = new Jw(null, !0);
        this.J = b || null;
        this.S = null != c ? c : null
    };
    var Zea = function(a, b) {
        this.mapId = a;
        this.map = b;
        this.context = null
    };
    var wx = function(a) {
        m.$l.call(this, "statevent", a)
    };
    var xx = function(a, b) {
        m.$l.call(this, "timingevent", a);
        this.size = b
    };
    var yx = function(a) {
        m.$l.call(this, "serverreachability", a)
    };
    var zx = function(a) {
        a.md && (a.md.abort(), a.md = null);
        a.Na && (a.Na.cancel(), a.Na = null);
        a.Wc && (m.da.clearTimeout(a.Wc), a.Wc = null);
        Ax(a);
        a.tb && (a.tb.cancel(), a.tb = null);
        a.wd && (m.da.clearTimeout(a.wd), a.wd = null)
    };
    var Bx = function(a, b) {
        if (0 == a.g)
            throw Error("Invalid operation: sending map when state is closed");
        a.j.push(new Zea(a.mu++, b));
        2 != a.g && 3 != a.g || Cx(a)
    };
    var Dx = function(a) {
        var b = 0;
        a.Na && b++;
        a.tb && b++;
        return b
    };
    var Cx = function(a) {
        a.tb || a.wd || (a.wd = cx((0, m.p)(a.Qm, a), 0), a.If = 0)
    };
    var Ex = function(a, b) {
        if (1 == a.g) {
            if (!b) {
                a.Hf = Math.floor(1E5 * Math.random());
                var c = a.Hf++, d = new Pw(a, 0, "", c, void 0);
                d.Id = a.Rg;
                var e = Fx(a), f = a.Ug.clone();
                m.Yj(f, "RID", c);
                a.L && m.Yj(f, "CVER", a.L);
                Gx(a, f);
                Qw(d, f, e);
                a.tb = d;
                a.g = 2
            }
        } else 
            3 == a.g && (b ? Hx(a, b) : 0 != a.j.length && (a.tb || Hx(a)))
    };
    var Hx = function(a, b) {
        var c, d;
        b ? 6 < a.Od ? (a.j = a.D.concat(a.j), a.D.length = 0, c = a.Hf - 1, d = Fx(a)) : (c = b.H, d = b.Kd) : (c = a.Hf++, d = Fx(a));
        var e = a.Ug.clone();
        m.Yj(e, "SID", a.C);
        m.Yj(e, "RID", c);
        m.Yj(e, "AID", a.Be);
        Gx(a, e);
        c = new Pw(a, 0, a.C, c, a.If + 1);
        c.Id = a.Rg;
        c.setTimeout(Math.round(1E4) + Math.round(1E4 * Math.random()));
        a.tb = c;
        Qw(c, e, d)
    };
    var Gx = function(a, b) {
        if (a.aa) {
            var c = a.aa.Xo(a);
            c && m.tb(c, function(a, c) {
                m.Yj(b, c, a)
            })
        }
    };
    var Fx = function(a) {
        var b = Math.min(a.j.length, 1E3), c = ["count=" + b], d;
        6 < a.Od && 0 < b ? (d = a.j[0].mapId, c.push("ofs=" + d)) : d = 0;
        for (var e = 0; e < b; e++) {
            var f = a.j[e].mapId, h = a.j[e].map, f = 6 >= a.Od ? e: f - d;
            try {
                m.$i(h, function(a, b) {
                    c.push("req" + f + "_" + b + "=" + (0, window.encodeURIComponent)(a))
                })
            } catch (k) {
                c.push("req" + f + "_type=" + (0, window.encodeURIComponent)("_badmap"))
            }
        }
        a.D = a.D.concat(a.j.splice(0, b));
        return c.join("&")
    };
    var Ix = function(a) {
        a.Na || a.Wc || (a.H = 1, a.Wc = cx((0, m.p)(a.vo, a), 0), a.Ue = 0)
    };
    var Jx = function(a) {
        if (a.Na || a.Wc || 3 <= a.Ue)
            return !1;
        a.H++;
        a.Wc = cx((0, m.p)(a.vo, a), Kx(a, a.Ue));
        a.Ue++;
        return !0
    };
    var lx = function(a, b, c) {
        a.mh = c;
        a.$b = b.Tc;
        a.ez(1, 0);
        a.Ug = jx(a, null, a.bj);
        Cx(a)
    };
    var Ax = function(a) {
        null != a.$d && (m.da.clearTimeout(a.$d), a.$d = null)
    };
    var Kx = function(a, b) {
        var c = 5E3 + Math.floor(1E4 * Math.random());
        a.isActive() || (c*=2);
        return c * b
    };
    var Lx = function(a, b) {
        if (2 == b || 9 == b) {
            var c = null;
            a.aa && (c = null);
            var d = (0, m.p)(a.Zy, a);
            c || (c = new m.Rj("//www.google.com/images/cleardot.gif"), bk(c));
            fx(c.toString(), 1E4, d)
        } else 
            Ww();
        Mx(a, b)
    };
    var Mx = function(a, b) {
        a.g = 0;
        a.aa && a.aa.Te(a, b);
        Nx(a);
        zx(a)
    };
    var Nx = function(a) {
        a.g = 0;
        a.$b =- 1;
        if (a.aa)
            if (0 == a.D.length && 0 == a.j.length)
                a.aa.ed(a);
            else {
                var b = m.Ra(a.D), c = m.Ra(a.j);
                a.D.length = 0;
                a.j.length = 0;
                a.aa.ed(a, b, c)
            }
    };
    var jx = function(a, b, c) {
        var d = m.il(c);
        if ("" != d.Tb)
            b && m.Tj(d, b + "." + d.Tb), Uj(d, d.ee);
        else 
            var e = window.location, d = Aba(e.protocol, b ? b + "." + e.hostname : e.hostname, e.port, c);
        a.kg && m.tb(a.kg, function(a, b) {
            m.Yj(d, b, a)
        });
        m.Yj(d, "VER", a.Od);
        Gx(a, d);
        return d
    };
    var cx = function(a, b) {
        if (!m.ma(a))
            throw Error("Fn must not be null and must be a function");
        return m.da.setTimeout(function() {
            a()
        }, b)
    };
    var Ww = function() {
        Ox.dispatchEvent(new wx(Ox))
    };
    var Px = function() {};
    var Qx = function(a, b) {
        m.Am.call(this);
        if (m.ma(a))
            b && (a = (0, m.p)(a, b));
        else if (a && m.ma(a.handleEvent))
            a = (0, m.p)(a.handleEvent, a);
        else 
            throw Error("Invalid listener argument");
        this.H = a;
        m.im(this, "tick", (0, m.p)(this.F, this));
        this.stop();
        Bm(this, 5E3 + 2E4 * Math.random())
    };
    var Rx = function(a, b) {
        this.O = a;
        this.C = b;
        this.k = new m.Wa;
        this.j = new Qx(this.vq, this);
        this.g = null;
        this.S=!1;
        this.F = null;
        this.J = "";
        this.L = this.D = 0;
        this.H = []
    };
    var $ea = function(a) {
        return {
            firstTestResults: [""],
            secondTestResults: !a.g.mh,
            sessionId: a.g.C,
            arrayId: a.g.Be
        }
    };
    var afa = function(a, b) {
        (a.C.loungeIdToken = b) || a.j.stop()
    };
    var Sx = function(a, b, c) {
        Au.call(this);
        this.U = a;
        this.Aa = [];
        this.Aa.push(m.J(window, "beforeunload", (0, m.p)(this.et, this)));
        this.g = [];
        this.$ = new $v;
        3 == c["mdx-version"] && (this.$.j = "RQ" + b.token);
        this.L = b.id;
        this.aa = bfa(this, c);
        this.aa.subscribe("handlerOpened", this.kt, this);
        this.aa.subscribe("handlerClosed", this.ft, this);
        this.aa.subscribe("handlerError", this.gt, this);
        this.$.j ? this.aa.subscribe("handlerMessage", this.ht, this) : this.aa.subscribe("handlerMessage", this.jt, this);
        afa(this.aa, b.token);
        this.subscribe("remoteQueueChange",
        function() {
            var a = this.$.videoId;
            m.Ku() && m.og("yt-remote-session-video-id", a)
        }, this)
    };
    var bfa = function(a, b) {
        return new Rx(qu(a.U, "/bc", void 0, !1), b)
    };
    var Tx = function(a, b) {
        a.publish("proxyStateChange", b)
    };
    var cfa = function(a) {
        a.eg = m.v((0, m.p)(function() {
            this.ma("Connecting timeout");
            this.D(1)
        }, a), 2E4)
    };
    var Ux = function(a) {
        m.bb(a.eg);
        a.eg = window.NaN
    };
    var Vx = function(a) {
        m.bb(a.Zg);
        a.Zg = window.NaN
    };
    var Wx = function(a) {
        Xx(a);
        a.mk = m.v((0, m.p)(function() {
            this.j("getNowPlaying")
        }, a), 2E4)
    };
    var Xx = function(a) {
        m.bb(a.mk);
        a.mk = window.NaN
    };
    var Yx = function(a) {
        var b = a.aa;
        return !!b.g && 3 == b.g.getState() && (0, window.isNaN)(a.eg)
    };
    var Zx = function(a, b) {
        b && (Ux(a), Vx(a));
        b == Yx(a) ? b && (Tx(a, 1), a.j("getSubtitlesTrack")) : b ? (a.H() && aw(a.$), Tx(a, 1), a.j("getNowPlaying")) : a.D(1)
    };
    var dfa = function(a, b) {
        var c = b.params.videoId;
        delete b.params.videoId;
        c == a.$.videoId && (m.kk(b.params) ? a.$.F = null : a.$.F = b.params, a.publish("remotePlayerChange"))
    };
    var $x = function(a, b) {
        var c = b.params.videoId || b.params.video_id, d = (0, window.parseInt)(b.params.currentIndex, 10);
        a.$.j = b.params.listId;
        var e = a.$, f = e.videoId;
        e.videoId = c;
        e.index = d;
        c != f && cw(e);
        a.publish("remoteQueueChange")
    };
    var efa = function(a, b) {
        b.params = b.params || {};
        $x(a, b);
        ay(a, b)
    };
    var ay = function(a, b) {
        var c = (0, window.parseInt)(b.params.currentTime || b.params.current_time, 10);
        fw(a.$, (0, window.isNaN)(c) ? 0 : c);
        c = (0, window.parseInt)(b.params.state, 10);
        c = (0, window.isNaN)(c)?-1 : c;
        - 1 == c&&-1E3 == a.$.g && (c =- 1E3);
        a.$.g = c;
        var d = null;
        - 1E3 == c && (d = a.$.D || "unknown", m.ca(b.params.currentError) && (d = m.$f(b.params.currentError).reason || d));
        a.$.D = d;
        1 == a.$.g ? Wx(a) : Xx(a);
        a.publish("remotePlayerChange")
    };
    var ffa = function(a, b) {
        var c = "true" == b.params.muted;
        a.$.C = (0, window.parseInt)(b.params.volume, 10);
        a.$.k = c;
        a.publish("remotePlayerChange")
    };
    var gfa = function(a, b) {
        switch (b.action) {
        case "loungeStatus":
            var c = m.$f(b.params.devices);
            a.g = (0, m.sh)(c, function(a) {
                return new Gu(a)
            });
            break;
        case "loungeScreenDisconnected":
            Ol(a.g, function(a) {
                return "LOUNGE_SCREEN" == a.type
            });
            break;
        case "remoteConnected":
            var d = new Gu(m.$f(b.params.device));
            m.Ja(a.g, function(a) {
                return a.equals(d)
            }) || m.Pl(a.g, d);
            break;
        case "remoteDisconnected":
            d = new Gu(m.$f(b.params.device)), Ol(a.g, function(a) {
                return a.equals(d)
            })
        }
    };
    var hfa = function(a, b) {
        var c=!1;
        if ("loungeStatus" == b.action)
            c=!!m.Ja(a.g, function(a) {
                return "LOUNGE_SCREEN" == a.type
            });
        else if ("loungeScreenConnected" == b.action)
            c=!0;
        else if ("loungeScreenDisconnected" == b.action)
            c=!1;
        else 
            return;
        if (!(0, window.isNaN)(a.Zg))
            if (c)
                Vx(a);
            else 
                return;
        c == Yx(a) ? c && Tx(a, 1) : c ? (Ux(a), a.H() && aw(a.$), Tx(a, 1), a.j("getNowPlaying")) : a.D(1)
    };
    var by = function(a) {
        Au.call(this);
        this.k = a;
        this.ec = cy();
        this.ma("Initializing local screens: " + ku(this.ec));
        this.ld = dy();
        this.ma("Initializing account screens: " + ku(this.ld));
        this.Xi = null;
        this.g = [];
        this.j = [];
        ey(this, Ew() || []);
        this.ma("Initializing DIAL devices: " + tu(this.j));
        a = iu(Uu());
        fy(this, a);
        this.ma("Initializing online screens: " + ku(this.g));
        this.D = (0, m.H)() + 3E5;
        ifa(this)
    };
    var dy = function() {
        var a = cy(), b = iu(Uu());
        return (0, m.qb)(b, function(b) {
            return !yu(a, b)
        })
    };
    var cy = function() {
        var a = iu(Pu());
        return (0, m.qb)(a, function(a) {
            return !a.uuid
        })
    };
    var ifa = function(a) {
        m.x("yt-remote-cast-device-list-update", function() {
            var a = Ew();
            ey(this, a || [])
        }, a);
        m.x("yt-remote-cast-device-status-update", a.ay, a);
        a.Wh();
        var b = (0, m.H)() > a.D ? 2E4: 1E4;
        m.ab((0, m.p)(a.Wh, a), b)
    };
    var gy = function(a) {
        fy(a, a.g, !0);
        ey(a, Ew() || []);
        a.publish("managedScreenChange", hy(a))
    };
    var jfa = function(a, b) {
        var c = hy(a);
        return (0, m.qb)(b, function(a) {
            return a.uuid ? (a = xu(this.j, a.uuid), !!a && "RUNNING" == a.status) : !!yu(c, a)
        }, a)
    };
    var ey = function(a, b) {
        var c=!1;
        (0, m.y)(b, function(a) {
            var b = zu(this.ec, a.id);
            b && b.name != a.name && (this.ma("Renaming screen id " + b.id + " from " + b.name + " to " + a.name), b.name = a.name, c=!0)
        }, a);
        c && (a.ma("Renaming due to DIAL."), iy(a));
        Vu(vu(b));
        var d=!m.Ml(a.j, b, Bea);
        d && a.ma("Updating DIAL devices: " + tu(a.j) + " to " + tu(b));
        a.j = b;
        fy(a, a.g);
        d && a.publish("onlineReceiverChange")
    };
    var fy = function(a, b, c) {
        var d = jfa(a, b), e=!m.Ml(a.g, d, gu);
        if (e || c)
            m.Rl(b) || Tu((0, m.sh)(d, hu));
        e && (a.ma("Updating online screens: " + ku(a.g) + " -> " + ku(d)), a.g = d, a.publish("onlineReceiverChange"))
    };
    var jy = function(a, b) {
        var c = [], d = {};
        (0, m.y)(b, function(a) {
            a.token && (d[a.token] = a, c.push(a.token))
        });
        var e = {
            method: "POST",
            V: {
                lounge_token: c.join(",")
            },
            context: a,
            R: function(a, b) {
                var c = [];
                (0, m.y)(b.screens || [], function(a) {
                    "online" == a.status && c.push(d[a.loungeToken])
                });
                var e = this.Xi ? ky(this, this.Xi): null;
                e&&!yu(c, e) && c.push(e);
                fy(this, c, !0)
            }
        };
        m.R(qu(a.k, "/pairing/get_screen_availability"), e)
    };
    var kfa = function(a) {
        var b = hy(a), c = (0, m.sh)(b, function(a) {
            return a.id
        });
        m.Rl(c) || (a.ma("Updating lounge tokens for: " + m.ag(c)), m.R(qu(a.k, "/pairing/get_lounge_token_batch"), {
            V: {
                screen_ids: c.join(",")
            },
            method: "POST",
            context: a,
            R: function(a, c) {
                lfa(this, c.screens || []);
                this.ec = (0, m.qb)(this.ec, function(a) {
                    return !!a.token
                });
                iy(this);
                jy(this, b)
            }
        }))
    };
    var lfa = function(a, b) {
        (0, m.y)(m.fj(a.ec, a.ld), function(a) {
            var d = m.Ja(b, function(b) {
                return a.id == b.screenId
            });
            d && (a.token = d.loungeToken)
        })
    };
    var iy = function(a) {
        var b = cy();
        m.Ml(a.ec, b, gu) || (a.ma("Saving local screens: " + ku(b) + " to " + ku(a.ec)), Ou((0, m.sh)(a.ec, hu)), gy(a))
    };
    var mfa = function(a, b, c) {
        var d = m.Ka(b, function(a) {
            return fu(c, a)
        }), e = 0 > d;
        0 > d ? b.push(c) : b[d] = c;
        yu(a.g, c) || a.g.push(c);
        return e
    };
    var ly = function(a, b, c) {
        var d = my, e = "";
        ny(d);
        if (xu(d.j, a)) {
            if (!e) {
                var f = e = uu();
                yw();
                var h = Fw(a), k = ww();
                if (k && h) {
                    var r = new window.cast.Receiver(h.id, h.name), r = new window.cast.LaunchRequest("YouTube", r);
                    r.parameters = "pairingCode=" + f;
                    r.description = new window.cast.LaunchDescription;
                    r.description.text = window.document.title;
                    b && (r.parameters += "&v=" + b, c && (r.parameters += "&t=" + Math.round(c)), r.description.url = "http://i.ytimg.com/vi/" + b + "/default.jpg");
                    "UNKNOWN" != h.status && (h.status = "UNKNOWN", Dw(h), m.z("yt-remote-cast-device-status-update",
                    h));
                    uw("Sending a cast launch request with params: " + r.parameters);
                    k.launch(r, m.q(Gw, a))
                } else 
                    uw("No cast API or no cast device. Dropping cast launch.")
                }
            d.Mh = e;
            d.gg = m.v((0, m.p)(d.Rm, d, a, 0, e), oy[0])
        } else 
            d.ma("No DIAL device with id: " + a)
    };
    var ny = function(a) {
        m.bb(a.gg);
        a.gg = window.NaN;
        a.Mh = ""
    };
    var ky = function(a, b) {
        var c = zu(hy(a), b);
        a.ma("Found screen: " + ju(c) + " with key: " + b);
        return c
    };
    var nfa = function(a) {
        var b = my, c = zu(b.g, a);
        b.ma("Found online screen: " + ju(c) + " with key: " + a);
        return c
    };
    var py = function(a) {
        var b = my, c = xu(b.j, a);
        if (!c) {
            var d = zu(b.ec, a);
            d && (c = xu(b.j, d.uuid))
        }
        b.ma("Found DIAL: " + (c ? c.toString() : "null") + " with key: " + a);
        return c
    };
    var hy = function(a) {
        return m.fj(a.ld, (0, m.qb)(a.ec, function(a) {
            return !yu(this.ld, a)
        }, a))
    };
    var qy = function(a) {
        m.Rl(a.ld) && m.Rl(dy()) || (a.ma("Clear cached account screens."), a.ld = [], a.Wh(), gy(a))
    };
    var ry = function(a) {
        Bu.call(this, "ScreenServiceProxy");
        this.hb = a;
        this.g = [];
        this.g.push(this.hb.$_s("screenChange", (0, m.p)(this.aw, this)));
        this.g.push(this.hb.$_s("onlineScreenChange", (0, m.p)(this.$v, this)))
    };
    var sy = function(a) {
        var b=!!m.u("MDX_ENABLE_CASTV2"), c=!!m.u("MDX_ENABLE_QUEUE");
        b ? m.l("yt.mdx.remote.castv2_", !0, void 0) : yw();
        m.pg && ii();
        Hu();
        ty || (ty = new pu, Yu() && (ty.g = "/api/loungedev"));
        my || b || (my = new by(ty), my.subscribe("screenPair", ofa), my.subscribe("managedScreenChange", pfa), my.subscribe("onlineReceiverChange", function() {
            m.z("yt-remote-receiver-availability-change")
        }));
        uy || (uy = m.n("yt.mdx.remote.deferredProxies_") || [], m.l("yt.mdx.remote.deferredProxies_", uy, void 0));
        qfa(c);
        c = wy();
        if (b&&!c) {
            var d =
            new nv(ty, xy(a));
            m.l("yt.mdx.remote.screenService_", d, void 0);
            c = wy();
            Lea(d, function(a) {
                a ? yy() && Zv(yy(), "YouTube TV") : d.subscribe("onlineScreenChange", function() {
                    m.z("yt-remote-receiver-availability-change")
                })
            })
        }
        if (a&&!m.n("yt.mdx.remote.initialized_")) {
            m.l("yt.mdx.remote.initialized_", !0, void 0);
            zy("Initializing: " + m.ag(a));
            Ay.push(m.x("yt-remote-cast2-availability-change", function() {
                m.z("yt-remote-receiver-availability-change")
            }));
            Ay.push(m.x("yt-remote-cast2-receiver-selected", function() {
                By(null);
                m.z("yt-remote-auto-connect", "cast-selector-receiver")
            }));
            Ay.push(m.x("yt-remote-cast2-session-change", rfa));
            Ay.push(m.x("yt-remote-connection-change", function(a) {
                a ? Zv(yy(), "YouTube TV") : Cy() || (Zv(null, null), Wv())
            }));
            var e = Dy();
            a.isAuto && (e.id += "#dial");
            e.name = a.device;
            e.app = a.app;
            zy(" -- with channel params: " + m.ag(e));
            sfa(e);
            b && c.start(xy(a));
            yy() || tfa();
            my && (a.loadAccountScreens || yy() ? my.lw() : qy(my))
        }
    };
    var Ey = function() {
        m.nb(Ay);
        Ay.length = 0;
        m.Va(Fy);
        Fy = null;
        uy && ((0, m.y)(uy, function(a) {
            a(null)
        }), uy.length = 0, uy = null, m.l("yt.mdx.remote.deferredProxies_", null, void 0));
        my && (m.Va(my), my = null);
        ty = null;
        Tea()
    };
    var Gy = function() {
        if (Hy() && Uv()) {
            var a = [];
            if (m.rg("yt-remote-cast-available") || m.n("yt.mdx.remote.cloudview.castButtonShown_") || Iy())
                a.push({
                    key: "cast-selector-receiver",
                    name: ufa()
                }), m.l("yt.mdx.remote.cloudview.castButtonShown_", !0, void 0);
            return a
        }
        if (m.n("yt.mdx.remote.cloudview.initializing_"))
            return [];
        var b = [], b = Jy() ? wy().hb.$_gos(): iu(Uu());
        (a = Ky()) && Iy() && (yu(b, a) || b.push(a));
        Jy() || (a = wu(Wu()), a = (0, m.qb)(a, function(a) {
            return !zu(b, a.id)
        }), b = m.fj(b, a));
        return Aea(b)
    };
    m.Ly = function() {
        if (Hy() && Uv()) {
            var a = Vv();
            return a ? {
                key: "cast-selector-receiver",
                name: a
            } : null
        }
        var a = Gy(), b = My(), c = Ky();
        c || (c = Cy());
        return m.Ja(a, function(a) {
            return c && eu(c, a.key) || b && (a = py(a.key)) && a.id == b?!0 : !1
        })
    };
    var ufa = function() {
        if (Hy() && Uv())
            return Vv();
        var a = Ky();
        return a ? a.name : null
    };
    var Ky = function() {
        var a = yy();
        if (!a)
            return null;
        if (!my) {
            var b = wy().ab();
            return zu(b, a)
        }
        return ky(my, a)
    };
    var rfa = function(a) {
        zy("remote.onCastSessionChange_: " + ju(a));
        if (a) {
            var b = Ky();
            b && b.id == a.id ? Zv(b.id, "YouTube TV") : (b && Ny(), Oy(a, 1))
        } else 
            Ny()
    };
    var vfa = function(a, b) {
        zy("Connecting to: " + m.ag(a));
        if ("cast-selector-receiver" == a.key)
            By(b || null), Oea(b || null);
        else {
            Ny();
            By(b || null);
            var c = null;
            my ? c = nfa(a.key) : (c = wy().ab(), c = zu(c, a.key));
            if (c)
                Oy(c, 1);
            else {
                if (my && (c = py(a.key))) {
                    wfa(c);
                    return 
                }
                m.v(function() {
                    Py(null)
                }, 0)
            }
        }
    };
    var Ny = function() {
        my && ny(my);
        t: {
            var a = Iy();
            if (a && (a = a.getOtherConnectedRemoteId())) {
                zy("Do not stop DIAL due to " + a);
                Qy("");
                break t
            }(a = My()) ? (zy("Stopping DIAL: " + a), Hw(a), Qy("")) : (a = Ky()) && a.uuid && (zy("Stopping DIAL: " + a.uuid), Hw(a.uuid))
        }
        Yv() ? Qv().stopSession() : Nv("stopSession called before API ready.");
        (a = Iy()) ? a.disconnect(1) : (m.ob("yt-remote-before-disconnect", 1), m.ob("yt-remote-connection-change", !1));
        Py(null)
    };
    m.Ry = function() {
        var a = Iy();
        return a && 3 != a.getProxyState() ? new jw(a, void 0) : null
    };
    var zy = function(a) {
        $t("remote", a)
    };
    var Hy = function() {
        return !!m.n("yt.mdx.remote.castv2_")
    };
    var Jy = function() {
        return m.n("yt.mdx.remote.screenService_")
    };
    var wy = function() {
        if (!Fy) {
            var a = Jy();
            a && (Fy = new ry(a))
        }
        return Fy
    };
    var xy = function(a) {
        var b=!!m.rg("yt-remote-load-account-screens");
        if (a) {
            var c=!!a.isSignedIn;
            c ? (b|=!!a.loadAccountScreens, Ru(b)) : Ru(!1);
            return c && b
        }
        return b
    };
    var yy = function() {
        return m.n("yt.mdx.remote.currentScreenId_")
    };
    var Sy = function(a) {
        m.l("yt.mdx.remote.currentScreenId_", a, void 0);
        if (my) {
            var b = my;
            b.D = (0, m.H)() + 3E5;
            if ((b.Xi = a) && (a = ky(b, a))&&!yu(b.g, a)) {
                var c = m.Ra(b.g);
                c.push(a);
                fy(b, c, !0)
            }
        }
    };
    var My = function() {
        return m.n("yt.mdx.remote.currentDialId_")
    };
    var Qy = function(a) {
        m.l("yt.mdx.remote.currentDialId_", a, void 0)
    };
    var Ty = function() {
        return m.n("yt.mdx.remote.connectData_")
    };
    var By = function(a) {
        m.l("yt.mdx.remote.connectData_", a, void 0)
    };
    var Iy = function() {
        return m.n("yt.mdx.remote.connection_")
    };
    var Py = function(a) {
        var b = Iy();
        By(null);
        a ? m.ni(!Iy()) : (Sy(""), Qy(""));
        m.l("yt.mdx.remote.connection_", a, void 0);
        uy && ((0, m.y)(uy, function(b) {
            b(a)
        }), uy.length = 0);
        b&&!a ? m.ob("yt-remote-connection-change", !1) : !b && a && m.z("yt-remote-connection-change", !0)
    };
    var Cy = function() {
        var a = m.Ku();
        if (!a)
            return null;
        if (Jy()) {
            var b = wy().ab();
            return zu(b, a)
        }
        return my ? ky(my, a) : null
    };
    var Oy = function(a, b) {
        m.ni(!yy());
        Sy(a.id);
        var c = new Sx(ty, a, Dy());
        c.connect(b, Ty());
        c.subscribe("beforeDisconnect", function(a) {
            m.ob("yt-remote-before-disconnect", a)
        });
        c.subscribe("beforeDispose", function() {
            Iy() && (Iy(), Py(null))
        });
        Py(c)
    };
    var wfa = function(a) {
        My();
        zy("Connecting to: " + (a ? a.toString() : "null"));
        Qy(a.id);
        var b = Ty();
        b ? ly(a.id, b.videoIds[b.index], b.currentTime) : ly(a.id)
    };
    var tfa = function() {
        var a = Cy();
        a ? (zy("Resume connection to: " + ju(a)), Oy(a, 0)) : (Xu(), Wv(), zy("Skipping connecting because no session screen found."))
    };
    var ofa = function(a) {
        zy("Paired with: " + ju(a));
        a ? Oy(a, 1) : Py(null)
    };
    var pfa = function() {
        var a = yy();
        a&&!Ky() && (zy("Dropping current screen with id: " + a), Ny());
        Cy() || Xu()
    };
    var qfa = function(a) {
        var b = Dy();
        if (m.kk(b)) {
            var b = Ju(), c = m.rg("yt-remote-session-name") || "", d = m.rg("yt-remote-session-app") || "", b = {
                device: "REMOTE_CONTROL",
                id: b,
                name: c,
                app: d
            };
            a && (b["mdx-version"] = 3);
            m.l("yt.mdx.remote.channelParams_", b, void 0)
        }
    };
    var Dy = function() {
        return m.n("yt.mdx.remote.channelParams_") || {}
    };
    var sfa = function(a) {
        a ? (m.og("yt-remote-session-app", a.app), m.og("yt-remote-session-name", a.name)) : (tk("yt-remote-session-app"), tk("yt-remote-session-name"));
        m.l("yt.mdx.remote.channelParams_", a, void 0)
    };
    var Uy = function(a, b) {
        this.type = a;
        this.videoIds = b || []
    };
    m.Vy = function() {
        sy();
        Wy.push(m.x("yt-remote-connection-change", xfa));
        var a = m.Ry();
        a && Xy(a)
    };
    m.Yy = function() {
        m.Va(Zy);
        Zy = null;
        m.nb(Wy);
        Wy.length = 0;
        Ey()
    };
    m.$y = function(a, b, c) {
        var d = "";
        m.ja(a) && (d = a, a = [d]);
        var e = new Uy(0, a);
        Zy ? az(function() {
            d ? Zy.Wn(d) : Zy.fp(a)
        }, e, b, c) : bz(e, {
            action_add_to_watch_queue: 1
        }, a, b, c)
    };
    m.cz = function(a, b, c) {
        m.ja(a) && (a = [a]);
        var d = new Uy(1, a);
        Zy ? az(function() {
            (0, m.y)(a, function(a) {
                Zy.Xn(a)
            })
        }, d, b, c) : bz(d, {
            action_remove_from_watch_queue: 1
        }, a, b, c)
    };
    m.dz = function(a, b, c) {
        var d = new Uy(2);
        Zy ? az(function() {
            Zy.Lp(a)
        }, d, b, c) : c && m.v(function() {
            c("Not implemented")
        }, 0)
    };
    m.ez = function(a, b) {
        var c = new Uy(2);
        Zy ? az(function() {}, c, a, b) : b && m.v(function() {
            b("Not implemented")
        }, 0)
    };
    var Xy = function(a) {
        Zy = a;
        Zy.subscribe("remoteQueueChange", function() {
            m.z("queue-change", new Uy(2))
        })
    };
    var xfa = function() {
        var a = m.Ry();
        m.Va(Zy);
        Zy = null;
        a ? Xy(a) : m.z("queue-change", new Uy(2))
    };
    var az = function(a, b, c, d) {
        Zy && 1 == Zy.getState() ? (a.call(m.da), c && m.v(function() {
            c()
        }, 0), m.ca(b) && m.z("queue-change", b)) : d && m.v(function() {
            d()
        }, 0)
    };
    var bz = function(a, b, c, d, e) {
        m.R("/watch_queue_ajax", {
            method: "POST",
            Z: b,
            V: {
                session_token: m.Kl(),
                list: "WQ",
                video_ids: c.join(",")
            },
            R: function() {
                d && d();
                m.z("queue-change", a)
            },
            onError: function() {
                e && e()
            }
        })
    };
    m.fz = function() {
        if (!gz) {
            var a = m.F("watch-queue");
            if (!a)
                return [];
            gz = m.G("watch-queue-items-list", a)
        }
        var b = [], a = m.wi(gz);
        (0, m.y)(a, function(a) {
            (a = m.I(a, "video-id")) && b.push(a)
        });
        return b
    };
    var hz = function(a) {
        return "tv-queue" == m.I(a, "style") ? "addto-tv-queue-button" : "addto-watch-queue-button"
    };
    var iz = function(a) {
        var b = hz(a);
        m.Fl(a, b, "addto-watch-queue-button-loading");
        var c = m.I(a, "video-ids"), d = m.I(a, "list-id"), e = Oq(m.Mq.getInstance(), a);
        d ? m.dz(d, function() {
            jz(a)
        }, function(c) {
            kz(a, b, e, c)
        }) : m.$y(c, function() {
            jz(a)
        }, function(c) {
            kz(a, b, e, c)
        })
    };
    var lz = function(a) {
        var b = m.I(a, "action");
        yfa[b](a)
    };
    var mz = function(a) {
        m.Fl(a, "addto-watch-queue-button-success", "addto-watch-queue-button-loading");
        var b = m.I(a, "video-ids"), c = m.I(a, "list-id"), d = Oq(m.Mq.getInstance(), a);
        c ? m.ez(function() {
            nz(a)
        }, function(b) {
            kz(a, "addto-watch-queue-button-success", d, b)
        }) : m.cz(b, function() {
            nz(a)
        }, function(b) {
            kz(a, "addto-watch-queue-button-success", d, b)
        })
    };
    var jz = function(a) {
        m.Fl(a, "addto-watch-queue-button-loading", "addto-watch-queue-button-success");
        var b = m.hb("ADDTO_WATCH_QUEUE_ADDED");
        m.Nq(m.Mq.getInstance(), a, b);
        m.I(a, "list-id") ? m.z("watch-queue-addto-playlist-added") : m.z("watch-queue-addto-video-added")
    };
    var nz = function(a) {
        var b = hz(a);
        m.Fl(a, "addto-watch-queue-button-loading", b);
        b = "addto-watch-queue-button" == b ? m.hb("ADDTO_WATCH_QUEUE") : m.hb("ADDTO_TV_QUEUE");
        m.Nq(m.Mq.getInstance(), a, b);
        m.I(a, "list-id") ? m.z("watch-queue-addto-playlist-removed") : m.z("watch-queue-addto-video-removed")
    };
    var kz = function(a, b, c, d) {
        m.Fl(a, "addto-watch-queue-button-loading", "addto-watch-queue-button-error");
        d = d || m.hb("ADDTO_WATCH_QUEUE_ERROR");
        m.Nq(m.Mq.getInstance(), a, d);
        m.v(function() {
            m.Fl(a, "addto-watch-queue-button-error", b);
            m.Nq(m.Mq.getInstance(), a, c)
        }, 5E3)
    };
    var oz = function() {
        var a = m.fz();
        if (!m.Ml(pz, a)) {
            pz = a;
            var b = {};
            (0, m.y)(pz, function(a) {
                b[a]=!0
            });
            a = m.M("addto-queue-button");
            (0, m.y)(a, function(a) {
                var d = m.I(a, "video-ids");
                if (d && m.ja(d)) {
                    var e = hz(a);
                    b[d] ? (m.Fl(a, e, "addto-watch-queue-button-success"), d = m.hb("ADDTO_WATCH_QUEUE_ADDED")) : (m.Fl(a, "addto-watch-queue-button-success", e), d = "addto-watch-queue-button" == e ? m.hb("ADDTO_WATCH_QUEUE") : m.hb("ADDTO_TV_QUEUE"));
                    m.Nq(m.Mq.getInstance(), a, d)
                }
            })
        }
    };
    var qz = function(a, b, c) {
        this.j = a;
        rz && (this.D = b);
        this.L = c || window;
        this.k = this.L.location;
        this.J = this.k.href.split("#")[0];
        this.H = (0, m.p)(this.S, this)
    };
    var sz = function(a) {
        a = a.k.href;
        var b = a.indexOf("#");
        return 0 > b ? "" : a.substring(b + 1)
    };
    var tz = function(a, b) {
        var c = a.J + "#" + b, d = a.k.href;
        d != c && d + "#" != c && (a.k.href = c)
    };
    var uz = function(a, b) {
        var c = a.D.contentWindow.document, d = "#" + m.va(b);
        (c.body ? c.body.innerHTML : "") != d && (d = ["<title>", m.wa(window.document.title || ""), "</title><body>", d, "</body>"], c.open("text/html"), c.write(d.join("")), c.close())
    };
    var vz = function(a, b) {
        this.k = a;
        this.C = b || window;
        this.j = this.C.location;
        this.H = (0, m.p)(this.L, this)
    };
    var zfa = function() {
        var a = wz("state");
        a.setEnabled.call(a, !0, !0)
    };
    var Afa = function(a) {
        var b = wz();
        b.replace.call(b, a, null, !0)
    };
    var wz = function(a) {
        a = a || "hash";
        var b = m.n("yt.history.instance_");
        b || ("state" == a ? (b = new vz(xz), vz.prototype.setEnabled = vz.prototype.setEnabled, vz.prototype.add = vz.prototype.add, vz.prototype.replace = vz.prototype.replace) : (b = new qz(xz, m.F("legacy-history-iframe")), qz.prototype.setEnabled = qz.prototype.setEnabled, qz.prototype.add = qz.prototype.add, qz.prototype.replace = qz.prototype.add), yz = b, m.l("yt.history.instance_", yz, void 0));
        return b
    };
    var xz = function(a, b) {
        m.z("navigate", a, b)
    };
    var zz = function(a, b, c) {
        if (Lq())
            Az(function() {
                m.n("yt.www.ypc.checkout.showYpcOverlay")(a, b, c)
            });
        else {
            var d = Bz({
                ypc_it: a,
                ypc_ii: b,
                ypc_ft: c
            });
            m.ae(d)
        }
    };
    var Bfa = function(a, b) {
        if (Lq())
            Az(function() {
                m.n("yt.www.ypc.checkout.showYpcRedeemGiftOverlay")(a, b)
            });
        else {
            var c = Bz({
                ypc_grc: a,
                ypc_ft: b
            });
            m.ae(c)
        }
    };
    var Cfa = function(a) {
        Cz || ((0, m.y)(m.u("YPC_LOADER_CALLBACKS"), function(a) {
            (a = m.n(a)) && a()
        }), Cz=!0);
        a && a()
    };
    var Bz = function(a) {
        a = m.bl(window.location.href, a);
        var b = m.u("YPC_SIGNIN_URL"), c = m.cl(b)["continue"], c = m.bl(c, {
            next: a
        });
        return m.bl(b, {
            "continue": c
        })
    };
    var Dfa = function(a) {
        var b = a.g;
        a = b.itemId;
        var c = b.itemType, b = b.flowType;
        Dz("paid-subscribe-button-click", {
            itemType: c,
            itemId: a
        });
        zz(c, a, b)
    };
    var Efa = function(a) {
        var b, c = Lq();
        if (b = c ? m.G("ypc-checkout-button", a.g) : m.G("ypc-signin-button", a.g)) {
            a = m.I(b, "ypc-item-type");
            var d = m.I(b, "ypc-item-id");
            b = m.I(b, "ypc-flow-type");
            c ? Dz("container-button-click", {
                itemId: d,
                itemType: a
            }) : Dz("signin-button-click");
            zz(a, d, b)
        }
    };
    var Az = function(a) {
        if (!Ez.length) {
            var b = m.u("YPC_LOADER_CSS");
            Ez.push(new m.pe(function(a) {
                m.Rf(b, a)
            }));
            var c = m.u("YPC_LOADER_JS");
            Ez.push(new m.pe(function(a) {
                m.rd(c, a)
            }));
            var d = m.u("YPC_LOADER_CONFIGS");
            Ez.push(new m.pe(function(a) {
                m.R(d, {
                    R: function(b, c) {
                        m.Xa(c.configs);
                        m.eb(c.messages);
                        a()
                    }
                })
            }))
        }
        m.Fk(Ez).then(function() {
            Cfa(a)
        })
    };
    var Ffa = function(a) {
        Az(a.Ia)
    };
    var Fz = function(a) {
        var b = a.currentTarget;
        if (b) {
            a = m.I(b, "ypc-item-type");
            var c = m.I(b, "ypc-item-id"), b = m.I(b, "ypc-flow-type");
            Lq() ? Dz("purchase-button-click", {
                itemId: c,
                itemType: a
            }) : Dz("signin-button-click");
            zz(a, c, b)
        }
    };
    var Gz = function(a) {
        var b = m.cl(window.location.href);
        Hz && (m.tb(a, function(a) {
            m.wb(b, a)
        }), a = m.rc(window.location.href.split("?", 2)[0], b), zfa(), Afa(a))
    };
    var Dz = function(a, b) {
        var c = {};
        m.zb(c, {
            label: a,
            pageName: m.u("PAGE_NAME")
        });
        b && m.zb(c, b);
        c = m.qc(c);
        m.Pj("ypc-checkout", c, void 0)
    };
    var Gfa = function() {
        (0, m.y)(m.u("ERRORS") || [], function(a) {
            Iz.apply(null, a)
        });
        m.Xa("ERRORS", [])
    };
    var Iz = function(a, b) {
        if (a && window && window.yterr&&!(5 <= Jz)) {
            var c = a.stacktrace, d = a.columnNumber;
            var e = a, f = m.n("window.location.href");
            if (m.ja(e))
                a = {
                    message: e,
                    name: "Unknown error",
                    lineNumber: "Not available",
                    fileName: f,
                    stack: "Not available"
                };
            else {
                var h, k, r=!1;
                try {
                    h = e.lineNumber || e.line || "Not available"
                } catch (w) {
                    h = "Not available", r=!0
                }
                try {
                    k = e.fileName || e.filename || e.sourceURL || m.da.$googDebugFname || f
                } catch (B) {
                    k = "Not available", r=!0
                }
                a=!r && e.lineNumber && e.fileName && e.stack && e.message && e.name ? e : {
                    message: e.message ||
                    "Not available",
                    name: e.name || "UnknownError",
                    lineNumber: h,
                    fileName: k,
                    stack: e.stack || "Not available"
                }
            }
            c = c || a.stack;
            e = a.lineNumber.toString();
            (0, window.isNaN)(e) || (0, window.isNaN)(d) || (e = e + ":" + d);
            Kz[a.message] || (d = {
                Z: {
                    a: "logerror",
                    t: "jserror",
                    type: a.name,
                    msg: a.message.substr(0, 1E3),
                    line: e,
                    level: b || "ERROR"
                },
                V: {
                    url: window.location.href,
                    file: a.fileName
                },
                method: "POST"
            }, c && (d.V.stack = c), m.R("/gen_204", d), Kz[a.message]=!0, Jz++)
        }
    };
    var Hfa = function(a) {
        if (!Ifa[a.target.tagName]&&!a.defaultPrevented)
            if (63 == a.keyCode)
                Lz();
            else if (47 == a.keyCode) {
                var b;
                (b = Mz) || (Mz = m.Kb("masthead-search-term"), b=!!Mz);
                b && (Mz.focus(), Nz && m.gj(Nz.ob) && Lz());
                m.Mk(a)
            }
    };
    var Lz = function() {
        var a;
        if (!(a = Nz))
            if (a = m.F("yt-keyboard-shortcuts-dialog")) {
                Nz = new m.hq(a, !1, !0, !0);
                Oz = m.Kb("yt-keyboard-shortcuts");
                if (a = m.F("yt-keyboard-shortcuts-close"))
                    a = m.J(a, "click", Lz), Pz.push(a);
                    a=!(!Nz ||!Oz)
            } else 
                a=!1;
        a && (m.gj(Nz.ob) ? m.jq(Nz, "close") : (Nz.show(), Oz.focus()))
    };
    m.Qz = function() {
        if (Rz&&!Sz) {
            Sz=!0;
            Tz = 0;
            Uz = window.document.querySelectorAll('[rel="spf-prefetch"]');
            for (var a = 0, b = Vz; a < b; a++)
                if (Rz && Tz < Vz) {
                    if (!Uz || a >= Uz.length)
                        break;
                        var c = Uz[a];
                        if (c && c.href) {
                            if (c && c.href) {
                                var d = m.sq(c);
                                m.Wz(c.href, void 0, d, void 0, void 0)
                            }
                            Tz++
                        }
                    }
            }
    };
    var Jfa = function() {
        Xz && (Yz = m.v(function() {
            (0, window.spf.style.prefetch)(m.u("PREFETCH_CSS_RESOURCES"));
            (0, window.spf.script.prefetch)(m.u("PREFETCH_JS_RESOURCES"))
        }, 5E3))
    };
    m.Wz = function(a, b, c, d, e) {
        var f = window.ytspf || {};
        window.spf && f.enabled && a && (c && m.Zd(a, c), window.spf.prefetch(a, {
            onDone: d,
            onError: e,
            onPartDone: m.q(Kfa, b)
        }))
    };
    var Kfa = function(a, b) {
        if (!1 !== a && (void 0 !== a || Zz && $z < aA) && b && b.part) {
            var c = b.part;
            if (c = c.data && c.data.swfcfg) {
                var d = m.Rg(), e = c.args;
                if (d && e) {
                    var f = function() {
                        d.preloadVideoByPlayerVars(e)
                    };
                    bA && bA();
                    d.addEventListener("onReady", f);
                    bA = function() {
                        d.removeEventListener("onReady", f);
                        bA = null
                    };
                    $z++
                }
            }
        }
    };
    var cA = function() {
        dA = [m.n("yt.dom.VisibilityMonitor.delegateByClass")(null, m.n("yt.dom.VisibilityMonitor.States.VISIBLE"), eA, "yt-thumb"), m.n("yt.dom.VisibilityMonitor.delegateByClass")(null, m.n("yt.dom.VisibilityMonitor.States.VISIBLE"), eA, "yt-uix-simple-thumb-wrap")];
        m.n("yt.dom.VisibilityMonitor.refresh")();
        m.wc("tdl")
    };
    var eA = function(a) {
        for (var b = 0, c = a.length; b < c; b++) {
            var d = m.Mb("img", null, a[b])[0];
            if (d) {
                var e = m.I(d, "thumb");
                e && (d.src = e, m.el(d, "thumb"))
            }
        }
    };
    var fA = function() {
        m.Uc.call(this, "www/common", ["www/base"]);
        this.g = []
    };
    m.gA = function(a, b) {
        this.j = a;
        this.k = b;
        this.g = a
    };
    m.hA = function(a, b, c, d, e) {
        this.g = a;
        this.k = b;
        this.C = c;
        d && (this.wl = d);
        e && (this.j = e)
    };
    m.iA = function(a) {
        null == a.Uh && (a.Uh = window.setInterval((0, m.p)(a.Xk, a), a.g))
    };
    m.jA = function(a) {
        window.clearInterval(a.Uh);
        a.Uh = null
    };
    m.kA = function(a, b, c) {
        m.t.call(this);
        this.k = a;
        this.aa = c;
        this.g = b || window;
        this.Xa = (0, m.p)(this.j, this)
    };
    var lA = function(a) {
        a = a.g;
        return a.requestAnimationFrame || a.webkitRequestAnimationFrame || a.mozRequestAnimationFrame || a.oRequestAnimationFrame || a.msRequestAnimationFrame || null
    };
    var mA = function(a) {
        a = a.g;
        return a.cancelRequestAnimationFrame || a.webkitCancelRequestAnimationFrame || a.mozCancelRequestAnimationFrame || a.oCancelRequestAnimationFrame || a.msCancelRequestAnimationFrame || null
    };
    var nA = function() {
        m.ym.call(this);
        this.g = 0;
        this.C = this.startTime = null
    };
    m.oA = function(a, b) {
        b ? a.setAttribute("role", b) : a.removeAttribute("role")
    };
    m.pA = function(a, b, c) {
        m.ha(c) && (c = c.join(" "));
        var d = "aria-" + b;
        "" === c || void 0 == c ? (qA || (qA = {
            atomic: !1,
            autocomplete: "none",
            dropeffect: "none",
            haspopup: !1,
            live: "off",
            multiline: !1,
            multiselectable: !1,
            orientation: "vertical",
            readonly: !1,
            relevant: "additions text",
            required: !1,
            sort: "none",
            busy: !1,
            disabled: !1,
            hidden: !1,
            invalid: "false"
        }), c = qA, b in c ? a.setAttribute(d, c[b]) : a.removeAttribute(d)) : a.setAttribute(d, c)
    };
    m.rA = function(a, b) {
        return a.Ha("iframe", {
            frameborder: 0,
            style: "border:0;vertical-align:bottom;" + (b || ""),
            src: 'javascript:""'
        })
    };
    m.sA = function(a, b, c) {
        this.j = a;
        (a = b || null) || (a = Lfa(this.j));
        a = "(" + a.join("|") + ")";
        a = m.Vl("__%s__", a);
        this.k = new RegExp(a, "g");
        this.g = c || {}
    };
    m.tA = function(a, b, c) {
        a = m.uA(m.F(a));
        return new m.sA(a, b, c)
    };
    m.uA = function(a) {
        a = a.innerHTML;
        a = a.replace(/^\s*(\x3c!--\s*)?/, "");
        return a = a.replace(/(\s*--\x3e)?\s*$/, "")
    };
    var Lfa = function(a) {
        var b = [], c = {};
        a.replace(Mfa, function(a, e) {
            e in c || (c[e]=!0, b.push(e))
        });
        return b
    };
    m.vA = function(a, b, c) {
        m.t.call(this);
        this.g = a;
        this.k = b || 0;
        this.aa = c;
        this.Xa = (0, m.p)(this.j, this)
    };
    var wA = function(a) {
        a = m.oa(a);
        delete xA[a];
        m.kk(xA) && yA && yA.stop()
    };
    var zA = function() {
        yA || (yA = new m.vA(function() {
            Nfa()
        }, 20));
        var a = yA;
        a.isActive() || a.start()
    };
    var Nfa = function() {
        var a = (0, m.H)();
        m.tb(xA, function(b) {
            AA(b, a)
        });
        m.kk(xA) || zA()
    };
    var BA = function(a, b, c, d) {
        nA.call(this);
        if (!m.ha(a) ||!m.ha(b))
            throw Error("Start and end parameters must be arrays");
        if (a.length != b.length)
            throw Error("Start and end points must be the same length");
        this.k = a;
        this.H = b;
        this.duration = c;
        this.F = d;
        this.j = []
    };
    var AA = function(a, b) {
        a.Hc = (b - a.startTime) / (a.C - a.startTime);
        1 <= a.Hc && (a.Hc = 1);
        a.Rl = 1E3 / (b - a.Wi);
        a.Wi = b;
        CA(a, a.Hc);
        1 == a.Hc ? (a.g = 0, wA(a), a.ka(), a.Sb()) : 1 == a.g && a.Hj()
    };
    var CA = function(a, b) {
        m.ma(a.F) && (b = a.F(b));
        a.j = Array(a.k.length);
        for (var c = 0; c < a.k.length; c++)
            a.j[c] = (a.H[c] - a.k[c]) * b + a.k[c]
    };
    var DA = function(a, b) {
        m.$l.call(this, a);
        this.x = b.j[0];
        this.y = b.j[1];
        this.duration = b.duration;
        this.fps = b.Rl;
        this.state = b.g
    };
    var EA = function(a, b, c, d, e) {
        BA.call(this, b, c, d, e);
        this.element = a
    };
    m.FA = function(a, b, c, d, e) {
        if (2 != b.length || 2 != c.length)
            throw Error("Start and end points must be 2D");
        EA.apply(this, arguments)
    };
    var GA = function(a, b, c, d, e) {
        m.ka(b) && (b = [b]);
        m.ka(c) && (c = [c]);
        EA.call(this, a, b, c, d, e);
        if (1 != b.length || 1 != c.length)
            throw Error("Start and end points must be 1D");
        this.D =- 1
    };
    m.HA = function(a, b, c) {
        GA.call(this, a, 1, 0, b, c)
    };
    m.IA = function(a, b, c) {
        GA.call(this, a, 0, 1, b, c)
    };
    m.JA = function(a, b) {
        m.ym.call(this);
        a && m.KA(this, a, b)
    };
    m.KA = function(a, b, c) {
        a.ti && LA(a);
        a.nf = b;
        a.ri = m.im(a.nf, "keypress", a, c);
        a.Dk = m.im(a.nf, "keydown", a.hz, c, a);
        a.ti = m.im(a.nf, "keyup", a.iz, c, a)
    };
    var LA = function(a) {
        a.ri && (m.sm(a.ri), m.sm(a.Dk), m.sm(a.ti), a.ri = null, a.Dk = null, a.ti = null);
        a.nf = null;
        a.Xb =- 1;
        a.Dd =- 1
    };
    var MA = function(a, b, c, d) {
        m.am.call(this, d);
        this.type = "key";
        this.keyCode = a;
        this.charCode = b
    };
    m.NA = function() {};
    var OA = function(a, b) {
        this.Yd = a instanceof m.ui ? a : new m.ui(a, b)
    };
    var Ofa = function(a) {
        Yaa("yt.net.apiloader.onApiLoaded_" + a, function() {
            Pfa(a)
        })
    };
    var Pfa = function(a) {
        PA[a]=!0;
        (0, m.y)(QA[a], function(a) {
            a.call()
        });
        delete QA[a]
    };
    m.RA = function(a, b, c, d) {
        var e = Qfa[a];
        e && (PA[a] ? b.call() : (QA[a] || (QA[a] = []), QA[a].push(b), SA[a] || (Ofa(a), b = {
            callback: "yt.net.apiloader.onApiLoaded_" + a
        }, c && (b.language = c), d && (b.v = d), e = m.rc(e, b), c = window.document.createElement("script"), c.src = e, window.document.body.appendChild(c), SA[a]=!0)))
    };
    m.TA = function(a, b) {
        switch (b) {
        case 1:
            return 0 != a%4 || 0 == a%100 && 0 != a%400 ? 28 : 29;
        case 5:
        case 8:
        case 10:
        case 3:
            return 30
        }
        return 31
    };
    m.UA = function(a, b, c) {
        m.ka(a) ? (this.g = VA(a, b || 0, c || 1), WA(this, c || 1)) : m.na(a) ? (this.g = VA(a.getFullYear(), a.getMonth(), a.getDate()), WA(this, a.getDate())) : (this.g = new Date((0, m.H)()), this.g.setHours(0), this.g.setMinutes(0), this.g.setSeconds(0), this.g.setMilliseconds(0))
    };
    var VA = function(a, b, c) {
        b = new Date(a, b, c);
        0 <= a && 100 > a && b.setFullYear(b.getFullYear() - 1900);
        return b
    };
    m.XA = function(a) {
        a = a.getTimezoneOffset();
        if (0 == a)
            a = "Z";
        else {
            var b = Math.abs(a) / 60, c = Math.floor(b), b = 60 * (b - c);
            a = (0 < a ? "-" : "+") + m.Tl(c, 2) + ":" + m.Tl(b, 2)
        }
        return a
    };
    var WA = function(a, b) {
        a.getDate() != b && a.g.setUTCHours(a.g.getUTCHours() + (a.getDate() < b ? 1 : - 1))
    };
    var YA = function() {};
    var ZA = function(a) {
        if ("number" == typeof a) {
            var b = new YA;
            b.g = a;
            var c;
            c = a;
            if (0 == c)
                c = "Etc/GMT";
            else {
                var d = ["Etc/GMT", 0 > c ? "-": "+"];
                c = Math.abs(c);
                d.push(Math.floor(c / 60)%100);
                c%=60;
                0 != c && d.push(":", m.Tl(c, 2));
                c = d.join("")
            }
            b.C = c;
            0 == a ? a = "UTC" : (c = ["UTC", 0 > a ? "+": "-"], a = Math.abs(a), c.push(Math.floor(a / 60)%100), a%=60, 0 != a && c.push(":", a), a = c.join(""));
            b.k = [a, a];
            b.j = [];
            return b
        }
        b = new YA;
        b.C = a.id;
        b.g =- a.std_offset;
        b.k = a.names;
        b.j = a.transitions;
        return b
    };
    var $A = function(a, b) {
        for (var c = Date.UTC(b.getUTCFullYear(), b.getUTCMonth(), b.getUTCDate(), b.getUTCHours(), b.getUTCMinutes()) / 36E5, d = 0; d < a.j.length && c >= a.j[d];)
            d += 2;
        return 0 == d ? 0 : a.j[d - 1]
    };
    m.aB = function(a, b) {
        this.j = [];
        this.g = b || m.bB;
        "number" == typeof a ? cB(this, a) : dB(this, a)
    };
    var dB = function(a, b) {
        for (; b;)
            for (var c = 0; c < eB.length; ++c) {
                var d = b.match(eB[c]);
                if (d) {
                    d = d[0];
                    b = b.substring(d.length);
                    0 == c && ("''" == d ? d = "'" : (d = d.substring(1, d.length - 1), d = d.replace(/\'\'/, "'")));
                    a.j.push({
                        text: d,
                        type: c
                    });
                    break
                }
            }
    };
    var cB = function(a, b) {
        var c;
        if (4 > b)
            c = a.g.Re[b];
        else if (8 > b)
            c = a.g.Sh[b - 4];
        else if (12 > b)
            c = a.g.Cn[b - 8], c = c.replace("{1}", a.g.Re[b - 8]), c = c.replace("{0}", a.g.Sh[b - 8]);
        else {
            cB(a, 10);
            return 
        }
        dB(a, c)
    };
    var fB = function(a, b) {
        var c;
        c = String(b);
        var d = a.g || m.bB;
        if (void 0 !== d.Vj) {
            for (var e = [], f = 0; f < c.length; f++) {
                var h = c.charCodeAt(f);
                e.push(48 <= h && 57 >= h ? String.fromCharCode(d.Vj + h - 48) : c.charAt(f))
            }
            c = e.join("")
        }
        return c
    };
    var gB = function(a) {
        if (!(a.getHours && a.getSeconds && a.getMinutes))
            throw Error("The date to format has no time (probably a goog.date.Date). Use Date or goog.date.DateTime, or use a pattern without time fields.");
    };
    var Rfa = function(a, b, c, d, e, f) {
        var h = b.length;
        switch (b.charAt(0)) {
        case "G":
            return c = 0 < d.getFullYear() ? 1 : 0, 4 <= h ? a.g.Zs[c] : a.g.Yk[c];
        case "y":
            return c = d.getFullYear(), 0 > c && (c =- c), 2 == h && (c%=100), fB(a, m.Tl(c, h));
        case "M":
            t:
            switch (c = d.getMonth(), h) {
            case 5:
                a = a.g.$s[c];
                break t;
            case 4:
                a = a.g.Ji[c];
                break t;
            case 3:
                a = a.g.Ki[c];
                break t;
            default:
                a = fB(a, m.Tl(c + 1, h))
            }
            return a;
        case "k":
            return gB(e), fB(a, m.Tl(e.getHours() || 24, h));
        case "S":
            return fB(a, (e.getTime()%1E3 / 1E3).toFixed(Math.min(3, h)).substr(2) + (3 < h ? m.Tl(0,
            h - 3) : ""));
        case "E":
            return c = d.getDay(), 4 <= h ? a.g.Ni[c] : a.g.Li[c];
        case "a":
            return gB(e), h = e.getHours(), a.g.Ii[12 <= h && 24 > h ? 1: 0];
        case "h":
            return gB(e), fB(a, m.Tl(e.getHours()%12 || 12, h));
        case "K":
            return gB(e), fB(a, m.Tl(e.getHours()%12, h));
        case "H":
            return gB(e), fB(a, m.Tl(e.getHours(), h));
        case "c":
            t:
            switch (c = d.getDay(), h) {
            case 5:
                a = a.g.ct[c];
                break t;
            case 4:
                a = a.g.Wl[c];
                break t;
            case 3:
                a = a.g.fj[c];
                break t;
            default:
                a = fB(a, m.Tl(c, 1))
            }
            return a;
        case "L":
            t:
            switch (c = d.getMonth(), h) {
            case 5:
                a = a.g.bt[c];
                break t;
            case 4:
                a =
                a.g.xe[c];
                break t;
            case 3:
                a = a.g.Mi[c];
                break t;
            default:
                a = fB(a, m.Tl(c + 1, h))
            }
            return a;
        case "Q":
            return c = Math.floor(d.getMonth() / 3), 4 > h ? a.g.$k[c] : a.g.Zk[c];
        case "d":
            return fB(a, m.Tl(d.getDate(), h));
        case "m":
            return gB(e), fB(a, m.Tl(e.getMinutes(), h));
        case "s":
            return gB(e), fB(a, m.Tl(e.getSeconds(), h));
        case "v":
            return a = f || ZA(c.getTimezoneOffset()), a.C;
        case "w":
            return c = new Date(e.getFullYear(), e.getMonth(), e.getDate()), b = a.g.ah || 0, c = c.valueOf() + 864E5 * (((a.g.bh || 3) - b + 7)%7 - ((c.getDay() + 6)%7 - b + 7)%7), c = Math.floor(Math.round((c -
            (new Date((new Date(c)).getFullYear(), 0, 1)).valueOf()) / 864E5) / 7) + 1, fB(a, m.Tl(c, h));
        case "z":
            return a = f || ZA(c.getTimezoneOffset()), 4 > h ? a.k[0 < $A(a, c) ? 2: 0] : a.k[0 < $A(a, c) ? 3: 1];
        case "Z":
            return b = f || ZA(c.getTimezoneOffset()), 4 > h ? (a =- (b.g - $A(b, c)), h = [0 > a ? "-": "+"], a = Math.abs(a), h.push(m.Tl(Math.floor(a / 60)%100, 2), m.Tl(a%60, 2)), a = h.join("")) : (h = b.g - $A(b, c), c = ["GMT"], c.push(0 >= h ? "+" : "-"), h = Math.abs(h), c.push(m.Tl(Math.floor(h / 60)%100, 2), ":", m.Tl(h%60, 2)), h = c.join(""), a = fB(a, h)), a;
        default:
            return ""
        }
    };
    m.hB = function(a) {
        var b = m.u("LOCAL_DATE_TIME_CONFIG");
        a = m.M("localized-date", a);
        (0, m.y)(a, function(a) {
            var d = m.I(a, "timestamp"), e = m.I(a, "format");
            a.innerHTML = m.iB(b, e, (0, window.parseInt)(d, 10))
        })
    };
    m.iB = function(a, b, c) {
        var d = "";
        "longdate" == b ? d = a.formatLongDate : "weekdayshorttime" == b ? d = a.formatWeekdayShortTime : "longdateonly" == b ? d = a.formatLongDateOnly : "shortdate" == b && (d = a.formatShortDate);
        m.bB = m.jB(a);
        return (new m.aB(d)).format(new Date(1E3 * c))
    };
    m.jB = function(a) {
        return {
            Ji: a.months,
            xe: a.months,
            Ki: a.shortMonths,
            Mi: a.shortMonths,
            Re: a.dateFormats,
            Ni: a.weekdays,
            Wl: a.shortMonths,
            Li: a.shortWeekdays,
            fj: a.shortWeekdays,
            gj: a.weekendRange,
            ah: a.firstDayOfWeek,
            bh: a.firstWeekCutoffDay,
            Ii: a.amPms
        }
    };
    m.kB = function() {
        nA.call(this);
        this.j = []
    };
    m.lB = function() {
        m.kB.call(this);
        this.k = 0
    };
    m.mB = function(a) {
        this.Yh = a || window;
        this.Aa = []
    };
    m.nB = function(a, b) {
        this.j = a;
        this.g = b
    };
    var Sfa = function(a, b, c) {
        this.k = a;
        this.j = b.name || null;
        this.g = {};
        for (a = 0; a < c.length; a++)
            b = c[a], this.g[b.j] = b
    };
    m.oB = function(a) {
        a = m.Ki(a.g);
        m.Nl(a, function(a, c) {
            return a.j - c.j
        });
        return a
    };
    m.pB = function(a) {
        return new a.k
    };
    var Tfa = function(a, b, c) {
        this.D = a;
        this.j = b;
        this.L = c.name;
        this.J=!!c.zA;
        this.k=!!c.Y;
        this.g = c.A;
        this.C = c.type;
        this.H=!1;
        switch (this.g) {
        case 3:
        case 4:
        case 6:
        case 16:
        case 18:
        case 2:
        case 1:
            this.H=!0
        }
        this.F = c.defaultValue
    };
    m.qB = function(a) {
        return 11 == a.g || 10 == a.g
    };
    m.U = function() {
        this.j = {};
        this.k = this.B().g;
        this.g = null
    };
    var rB = function(a, b) {
        for (var c = m.oB(a.B()), d = 0; d < c.length; d++) {
            var e = c[d], f = e.j;
            if (m.sB(b, f)) {
                a.g && delete a.g[e.j];
                var h = m.qB(e);
                if (e.k)
                    for (var e = m.tB(b, f), k = 0; k < e.length; k++)
                        m.uB(a, f, h ? e[k].clone() : e[k]);
                else 
                    e = vB(b, f), h ? (h = vB(a, f)) ? rB(h, e) : m.V(a, f, e.clone()) : m.V(a, f, e)
                }
        }
    };
    m.sB = function(a, b) {
        return null != a.j[b]
    };
    var vB = function(a, b) {
        var c = a.j[b];
        return null == c ? null : c
    };
    m.W = function(a, b, c) {
        var d = vB(a, b);
        return a.k[b].k ? d[c || 0] : d
    };
    m.wB = function(a, b) {
        var c;
        if (m.sB(a, b))
            c = m.W(a, b, void 0);
        else 
            t: {
            c = a.k[b];
            if (void 0 === c.F) {
                var d = c.C;
                if (d === Boolean)
                    c.F=!1;
                else if (d === Number)
                    c.F = 0;
                else if (d === String)
                    c.F = c.H ? "0" : "";
                else {
                    c = new d;
                    break t
                }
            }
            c = c.F
        }
        return c
    };
    m.tB = function(a, b) {
        return vB(a, b) || []
    };
    m.V = function(a, b, c) {
        a.j[b] = c;
        a.g && (a.g[b] = c)
    };
    m.uB = function(a, b, c) {
        a.j[b] || (a.j[b] = []);
        a.j[b].push(c);
        a.g && delete a.g[b]
    };
    m.xB = function(a, b) {
        delete a.j[b];
        a.g && delete a.g[b]
    };
    m.X = function(a, b) {
        var c = [], d = b[0], e;
        for (e in b)
            0 != e && c.push(new Tfa(a, e, b[e]));
        return new Sfa(a, d, c)
    };
    m.yB = function(a) {
        m.zB();
        for (var b = AB, c = [], d = 0; d < a.length;) {
            var e = b[a.charAt(d++)], f = d < a.length ? b[a.charAt(d)]: 0;
            ++d;
            var h = d < a.length ? b[a.charAt(d)]: 64;
            ++d;
            var k = d < a.length ? b[a.charAt(d)]: 64;
            ++d;
            if (null == e || null == f || null == h || null == k)
                throw Error();
            c.push(e<<2 | f>>4);
            64 != h && (c.push(f<<4 & 240 | h>>2), 64 != k && c.push(h<<6 & 192 | k))
        }
        return c
    };
    m.zB = function() {
        if (!m.BB) {
            m.BB = {};
            m.CB = {};
            AB = {};
            for (var a = 0; 65 > a; a++)
                m.BB[a] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".charAt(a), m.CB[a] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.".charAt(a), AB[m.CB[a]] = a, 62 <= a && (AB["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".charAt(a)] = a)
        }
    };
    var DB = function(a, b) {
        this.j = a | 0;
        this.g = b | 0
    };
    m.EB = function(a) {
        if ( - 128 <= a && 128 > a) {
            var b = FB[a];
            if (b)
                return b
        }
        b = new DB(a | 0, 0 > a?-1 : 0);
        - 128 <= a && 128 > a && (FB[a] = b);
        return b
    };
    m.GB = function(a) {
        return (0, window.isNaN)(a) ||!(0, window.isFinite)(a) ? m.HB : a<=-IB ? JB : a + 1 >= IB ? Ufa : 0 > a ? m.KB(m.GB( - a)) : new DB(a%4294967296 | 0, a / 4294967296 | 0)
    };
    m.LB = function(a, b) {
        return new DB(a, b)
    };
    var MB = function(a) {
        return 0 <= a.j ? a.j : 4294967296 + a.j
    };
    var NB = function(a) {
        return 0 == a.g && 0 == a.j
    };
    m.OB = function(a, b) {
        if (a.equals(b))
            return 0;
        var c = 0 > a.g, d = 0 > b.g;
        return c&&!d?-1 : !c && d ? 1 : 0 > PB(a, b).g?-1 : 1
    };
    m.KB = function(a) {
        return a.equals(JB) ? JB : m.LB(~a.j, ~a.g).add(QB)
    };
    var PB = function(a, b) {
        return a.add(m.KB(b))
    };
    m.RB = function(a, b) {
        if (NB(a) || NB(b))
            return m.HB;
        if (a.equals(JB))
            return 1 == (b.j & 1) ? JB : m.HB;
        if (b.equals(JB))
            return 1 == (a.j & 1) ? JB : m.HB;
        if (0 > a.g)
            return 0 > b.g ? m.RB(m.KB(a), m.KB(b)) : m.KB(m.RB(m.KB(a), b));
        if (0 > b.g)
            return m.KB(m.RB(a, m.KB(b)));
        if (0 > m.OB(a, SB) && 0 > m.OB(b, SB))
            return m.GB((4294967296 * a.g + MB(a)) * (4294967296 * b.g + MB(b)));
        var c = a.g>>>16, d = a.g & 65535, e = a.j>>>16, f = a.j & 65535, h = b.g>>>16, k = b.g & 65535, r = b.j>>>16, w = b.j & 65535, B, Q, Y, la;
        la = 0 + f * w;
        Y = 0 + (la>>>16);
        Y += e * w;
        Q = 0 + (Y>>>16);
        Y = (Y & 65535) + f * r;
        Q += Y>>>16;
        Y&=
        65535;
        Q += d * w;
        B = 0 + (Q>>>16);
        Q = (Q & 65535) + e * r;
        B += Q>>>16;
        Q&=65535;
        Q += f * k;
        B += Q>>>16;
        Q&=65535;
        B = B + (c * w + d * r + e * k + f * h) & 65535;
        return m.LB(Y<<16 | la & 65535, B<<16 | Q)
    };
    var TB = function(a, b) {
        if (NB(b))
            throw Error("division by zero");
        if (NB(a))
            return m.HB;
        if (a.equals(JB)) {
            if (b.equals(QB) || b.equals(m.UB))
                return JB;
            if (b.equals(JB))
                return QB;
            var c;
            c = 1;
            if (0 == c)
                c = a;
            else {
                var d = a.g;
                c = 32 > c ? m.LB(a.j>>>c | d<<32 - c, d>>c) : m.LB(d>>c - 32, 0 <= d ? 0 : - 1)
            }
            c = m.VB(TB(c, b), 1);
            if (c.equals(m.HB))
                return 0 > b.g ? QB : m.UB;
            d = PB(a, m.RB(b, c));
            return c.add(TB(d, b))
        }
        if (b.equals(JB))
            return m.HB;
        if (0 > a.g)
            return 0 > b.g ? TB(m.KB(a), m.KB(b)) : m.KB(TB(m.KB(a), b));
        if (0 > b.g)
            return m.KB(TB(a, m.KB(b)));
        for (var e = m.HB,
        d = a; 0 <= m.OB(d, b);) {
            c = Math.max(1, Math.floor((4294967296 * d.g + MB(d)) / (4294967296 * b.g + MB(b))));
            for (var f = Math.ceil(Math.log(c) / Math.LN2), f = 48 >= f ? 1 : Math.pow(2, f - 48), h = m.GB(c), k = m.RB(h, b); 0 > k.g || 0 < m.OB(k, d);)
                c -= f, h = m.GB(c), k = m.RB(h, b);
            NB(h) && (h = QB);
            e = e.add(h);
            d = PB(d, k)
        }
        return e
    };
    m.WB = function(a, b) {
        return m.LB(a.j^b.j, a.g^b.g)
    };
    m.VB = function(a, b) {
        b&=63;
        if (0 == b)
            return a;
        var c = a.j;
        return 32 > b ? m.LB(c<<b, a.g<<b | c>>>32 - b) : m.LB(0, c<<b - 32)
    };
    m.XB = function(a, b) {
        b&=63;
        if (0 == b)
            return a;
        var c = a.g;
        return 32 > b ? m.LB(a.j>>>b | c<<32 - b, c>>>b) : 32 == b ? m.LB(c, 0) : m.LB(c>>>b - 32, 0)
    };
    m.YB = function() {};
    m.ZB = function(a, b, c) {
        b = m.pB(b);
        a.F(b, c);
        return b
    };
    m.$B = function() {
        this.g = [];
        this.H = {
            value: 0,
            length: 0
        };
        this.L = {
            value: m.HB,
            length: 0
        };
        this.k = new window.DataView(new window.ArrayBuffer(8))
    };
    var aC = function(a, b) {
        for (var c = a.L, d = m.GB(0), e = 0; e < b.length; e++) {
            var f = m.VB(m.EB(b[e] & 127), 7 * e), d = m.LB(d.j | f.j, d.g | f.g);
            if (0 == (b[e] & 128))
                break
        }
        c.value = d;
        c.length = e + 1;
        return c
    };
    var bC = function(a, b) {
        for (var c = a.H, d = 0, e = 0; e < b.length && (d|=(b[e] & 127)<<7 * e, 0 != (b[e] & 128)); e++);
        c.value = d;
        c.length = e + 1;
        return c
    };
    var cC = function(a) {
        for (var b = 0, c = 0; c < a.length; c++)
            b|=a[c]<<8 * c;
        return b
    };
    var dC = function(a) {
        var b = "";
        a = new window.Uint16Array(a);
        for (var c = 0; c < a.length; c += 65536)
            var d = Math.min(65536, a.length - c), b = b + String.fromCharCode.apply(null, a.subarray(c, c + d));
        return b
    };
    m.eC = function() {
        fC && fC.abort && fC.abort();
        m.gC && m.ri(m.gC)
    };
    m.hC = function(a, b, c, d) {
        var e = m.Ja(b.media_template_data, function(a) {
            return !!a.imageUrl
        });
        e && (a = {
            video_id: e.videoId,
            ad_type: a,
            headline: m.Ul(b.line1),
            image_url: e.imageUrl,
            description1: m.Ul(b.line2),
            description2: m.Ul(b.line3),
            channel_title: e.channelName,
            visible_url: m.Ul(b.visible_url)
        }, m.iC = m.Ul(b.url), jC = m.ak(new m.Rj(m.iC), "adurl") || "", m.rk(m.ok(b.creative_view_url)) || (m.kC = m.Ul(b.creative_view_url)), m.rk(m.ok(b.engaged_view_url)) || (lC.part2viewed = m.Ul(b.engaged_view_url)), m.rk(m.ok(b.videoplaytime_25_url)) ||
        (lC.videoplaytime25 = m.Ul(b.videoplaytime_25_url)), m.rk(m.ok(b.videoplaytime_50_url)) || (lC.videoplaytime50 = m.Ul(b.videoplaytime_50_url)), m.rk(m.ok(b.videoplaytime_75_url)) || (lC.videoplaytime75 = m.Ul(b.videoplaytime_75_url)), m.rk(m.ok(b.videoplaytime_100_url)) || (lC.videoplaytime100 = m.Ul(b.videoplaytime_100_url)), fC = m.R("/pyv?" + m.qc(a), {
            format: "XML",
            R: function(a, b) {
                c && b.html_content && (m.F(c).innerHTML = b.html_content);
                d && d(a, b)
            }
        }))
    };
    m.mC = function(a) {
        var b = "feed", c = "feed-pyv-container", d = m.F("medium-shelf-pyv-container"), e = m.F("shelf-pyv-container");
        m.F("feed-pyv-container") ? (b = "feed", c = "feed-pyv-container") : d ? (b = "medium_shelf", c = "medium-shelf-pyv-container") : e && (b = "shelf", c = "shelf-pyv-container");
        var f = m.F(c);
        if (!f || 0 == a.length)
            if ("feed" == b || "shelf" == b || "medium_shelf" == b) {
                m.ri(f);
                return 
            }
        m.hC(b, a[0], f, function() {
            (0, m.y)(Vfa, function(a) {
                if (a = window.document.getElementById(a))
                    a.setAttribute("href", m.iC), m.J(a, "click", Wfa)
            });
            m.z("yt-dom-content-change",
            f);
            var a = m.kC;
            m.sc(a, void 0, m.hl(a))
        })
    };
    var Wfa = function() {
        var a = {};
        a.adpings = m.dl(lC);
        m.Zd(jC, a)
    };
    m.nC = function() {
        m.Ko.call(this)
    };
    m.oC = function() {
        if (!pC) {
            var a = new Xfa;
            qC(a, ["_setSampleRate", "50"]);
            qC(a, ["_trackPageview", window.document.location.pathname]);
            qC(a, ["_trackPageLoadTime"]);
            pC = a
        }
    };
    var Xfa = function() {
        this.g = "t__" + m.oa(this) + ".";
        m.ca(window._gaq) || (window._gaq = [], m.rd("https://ssl.google-analytics.com/ga.js"));
        qC(this, ["_setAccount", "UA-22977505-2"])
    };
    var qC = function(a, b) {
        b[0] = a.g + b[0];
        window._gaq.push(b)
    };
    m.rC = function(a, b, c) {
        this.L = b || sC.fq;
        this.lb = c || 0;
        this.O = 40;
        this.g = 1;
        this.X = 0;
        this.C = 3;
        this.T = this.j = 0;
        this.ja=!1;
        this.S = this.D = "";
        this.k = "-";
        this.F = "";
        this.J = 1;
        this.xa = 3;
        this.U = this.ba=!1;
        this.H = 0;
        if ("number" == typeof a)
            switch (a) {
            case 1:
                tC(this, sC.DECIMAL_PATTERN);
                break;
            case 2:
                tC(this, sC.hq);
                break;
            case 3:
                tC(this, sC.gq);
                break;
            case 4:
                a = sC.dq;
                b = ["0"];
                c = uC[this.L][0] & 7;
                if (0 < c) {
                    b.push(".");
                    for (var d = 0; d < c; d++)
                        b.push("0")
                    }
                    a = a.replace(/0.00/g, b.join(""));
                    tC(this, a);
                    break;
                case 5:
                    vC(this, 1);
                    break;
                case 6:
                    vC(this,
                    2);
                    break;
                default:
                    throw Error("Unsupported pattern type.");
            } else 
                tC(this, a)
        };
    var tC = function(a, b) {
        b.replace(/ /g, "\u00a0");
        var c = [0];
        a.D = wC(a, b, c);
        for (var d = c[0], e =- 1, f = 0, h = 0, k = 0, r =- 1, w = b.length, B=!0; c[0] < w && B; c[0]++)
            switch (b.charAt(c[0])) {
            case "#":
                0 < h ? k++ : f++;
                0 <= r && 0 > e && r++;
                break;
            case "0":
                if (0 < k)
                    throw Error('Unexpected "0" in pattern "' + b + '"');
                    h++;
                    0 <= r && 0 > e && r++;
                    break;
                case ",":
                    r = 0;
                    break;
                case ".":
                    if (0 <= e)
                        throw Error('Multiple decimal separators in pattern "' + b + '"');
                        e = f + h + k;
                        break;
                    case "E":
                        if (a.U)
                            throw Error('Multiple exponential symbols in pattern "' + b + '"');
                            a.U=!0;
                            a.T = 0;
                            c[0] +
                            1 < w && "+" == b.charAt(c[0] + 1) && (c[0]++, a.ja=!0);
                            for (; c[0] + 1 < w && "0" == b.charAt(c[0] + 1);)
                                c[0]++, a.T++;
                                if (1 > f + h || 1 > a.T)
                                    throw Error('Malformed exponential pattern "' + b + '"');
                                    B=!1;
                                    break;
                                default:
                                    c[0]--, B=!1
            }
        0 == h && 0 < f && 0 <= e && (h = e, 0 == h && h++, k = f - h, f = h - 1, h = 1);
        if (0 > e && 0 < k || 0 <= e && (e < f || e > f + h) || 0 == r)
            throw Error('Malformed pattern "' + b + '"');
        k = f + h + k;
        a.C = 0 <= e ? k - e : 0;
        0 <= e && (a.j = f + h - e, 0 > a.j && (a.j = 0));
        a.g = (0 <= e ? e : k) - f;
        a.U && (a.O = f + a.g, 0 == a.C && 0 == a.g && (a.g = 1));
        a.xa = Math.max(0, r);
        a.ba = 0 == e || e == k;
        d = c[0] - d;
        a.S = wC(a, b, c);
        c[0] <
        b.length && ";" == b.charAt(c[0]) ? (c[0]++, a.k = wC(a, b, c), c[0] += d, a.F = wC(a, b, c)) : (a.k = a.D + a.k, a.F += a.S)
    };
    var vC = function(a, b) {
        a.H = b;
        tC(a, sC.DECIMAL_PATTERN);
        a.j = 0;
        a.C = 2;
        if (0 < a.j)
            throw Error("Can't combine significant digits and minimum fraction digits");
        a.X = 2
    };
    var xC = function(a, b) {
        var c = Math.pow(10, a.C), d = 0 >= a.X ? Math.round(b * c): Math.round(Yfa(b * c, a.X, a.C)), e;
        (0, window.isFinite)(d) ? (e = Math.floor(d / c), c = Math.floor(d - e * c)) : (e = b, c = 0);
        return {
            intValue: e,
            mw: c
        }
    };
    var yC = function(a, b, c, d) {
        if (a.j > a.C)
            throw Error("Min value must be less than max value");
        b = xC(a, b);
        var e = Math.pow(10, a.C), f = b.intValue, h = b.mw, k = 0 < a.j || 0 < h ||!1;
        b = a.j;
        k && (b = a.j);
        for (var r = "", w = f; 1E20 < w;)
            r = "0" + r, w = Math.round(w / 10);
        var r = w + r, B = sC.DECIMAL_SEP, Q = sC.GROUP_SEP, w = sC.Xj.charCodeAt(0), Y = r.length;
        if (0 < f || 0 < c) {
            for (f = Y; f < c; f++)
                d.push(String.fromCharCode(w));
            for (f = 0; f < Y; f++)
                d.push(String.fromCharCode(w + 1 * r.charAt(f))), 1 < Y - f && 0 < a.xa && 1 == (Y - f)%a.xa && d.push(Q)
        } else 
            k || d.push(String.fromCharCode(w));
        (a.ba || k) && d.push(B);
        a = "" + (h + e);
        for (c = a.length; "0" == a.charAt(c - 1) && c > b + 1;)
            c--;
        for (f = 1; f < c; f++)
            d.push(String.fromCharCode(w + 1 * a.charAt(f)))
    };
    var zC = function(a, b, c) {
        c.push(sC.Um);
        0 > b ? (b =- b, c.push(sC.Fy)) : a.ja && c.push(sC.Gy);
        b = "" + b;
        for (var d = sC.Xj, e = b.length; e < a.T; e++)
            c.push(d);
        c.push(b)
    };
    var AC = function(a) {
        a = a.charCodeAt(0);
        if (48 <= a && 58 > a)
            return a - 48;
        var b = sC.Xj.charCodeAt(0);
        return b <= a && a < b + 10 ? a - b : - 1
    };
    var wC = function(a, b, c) {
        for (var d = "", e=!1, f = b.length; c[0] < f; c[0]++) {
            var h = b.charAt(c[0]);
            if ("'" == h)
                c[0] + 1 < f && "'" == b.charAt(c[0] + 1) ? (c[0]++, d += "'") : e=!e;
            else if (e)
                d += h;
            else 
                switch (h) {
                case "#":
                case "0":
                case ",":
                case ".":
                case ";":
                    return d;
                case "\u00a4":
                    if (c[0] + 1 < f && "\u00a4" == b.charAt(c[0] + 1))
                        c[0]++, d += a.L;
                    else 
                        switch (a.lb) {
                        case 0:
                            d += uC[a.L][1];
                            break;
                        case 2:
                            var h = a.L, k = uC[h], d = d + (h == k[1] ? h : h + " " + k[1]);
                            break;
                        case 1:
                            d += uC[a.L][2]
                        }
                        break;
                    case "%":
                        if (1 != a.J)
                            throw Error("Too many percent/permill");
                            a.J = 100;
                            d +=
                            sC.Km;
                            break;
                        case "\u2030":
                            if (1 != a.J)
                                throw Error("Too many percent/permill");
                                a.J = 1E3;
                                d += sC.Lm;
                                break;
                            default:
                                d += h
                }
        }
        return d
    };
    var BC = function(a, b) {
        var c = 1 == a.H ? CC.my: CC.ly;
        if (3 > b)
            return DC;
        b = Math.min(14, b);
        c = c[Math.pow(10, b)];
        if (!c)
            return DC;
        c = c.other;
        return c && "0" != c ? (c = /([^0]*)(0+)(.*)/.exec(c)) ? {
            prefix: c[1],
            Bl: c[3],
            Zi: b - (c[2].length - 1)
        } : DC : DC
    };
    var EC = function(a) {
        for (var b = 0; 1 <= (a/=10);)
            b++;
        return b
    };
    var Yfa = function(a, b, c) {
        if (!a)
            return a;
        b = b - EC(a) - 1;
        if (b<-c)
            return c = Math.pow(10, c), Math.round(a / c) * c;
        c = Math.pow(10, b);
        return Math.round(a * c) / c
    };
    m.FC = function(a, b) {
        this.da = a;
        m.O(this.da, "mouseover", (0, m.p)(this.g, this), "helpable-preview");
        if (b) {
            var c = m.G("overlay-checkbox", a), d = m.G("trueview-checkbox", a), e = m.G("standard-instream-checkbox", a);
            m.wp(c, b.overlay);
            m.wp(d, b.trueview_instream);
            e && m.wp(e, b.standard_instream)
        }
    };
    var GC = function(a, b, c) {
        m.t.call(this);
        this.g = a;
        this.j = new m.Am(50);
        this.D = new m.ir(this);
        this.C = new m.ui;
        this.F = m.zk(a);
        this.H = b || 0;
        if (b) {
            if (a = this.F.clone(), b = this.H) {
                var d = Math.min(b, .25 * a.height);
                a.top += d;
                a.height -= 2 * d;
                b = Math.min(b, .25 * a.width);
                a.top += b;
                a.height -= 2 * b
            }
        } else 
            a = this.F;
        this.k = a;
        this.Qc(!!c)
    };
    var HC = function(a, b, c) {
        var d = 0;
        a < b ? d =- 8 : a > b + c && (d = 8);
        return d
    };
    m.IC = function() {
        m.Ko.call(this);
        this.F = [];
        var a = m.xl("div");
        window.document.body.appendChild(a);
        var b = "auto" == m.Ci(a, "pointerEvents");
        m.ri(a);
        this.O = b
    };
    var JC = function(a, b) {
        if (a.j) {
            var c = mi(b) + a.D.x, d = li(b) + a.D.y;
            m.Dk(a.j, c, d)
        }
    };
    var KC = function(a, b) {
        var c = m.Xi(b), d = new m.ui(mi(a), li(a)), e = c.x - d.x, c = c.y - d.y;
        return Math.sqrt(e * e + c * c)
    };
    m.LC = function(a) {
        a.o = m.u("CREATOR_CONTEXT", "U");
        return a
    };
    m.MC = function(a, b) {
        if ((0, m.Zh)()) {
            var c = rba(a), d = 0, e = m.Te() + "-transform";
            c && (c.transform || c[e]) && (d = c.transform || c[e]);
            var f = m.v(function() {
                m.N(h);
                b.call(a)
            }, d + 100), h = m.J(a, zq, function(c) {
                c.target == a && "transform" == c.propertyName && (m.N(h), m.bb(f), b.call(a))
            })
        } else 
            m.v(function() {
                b.call(a)
            }, 0)
    };
    m.NC = function() {
        m.Ko.call(this)
    };
    m.OC = function(a, b, c) {
        var d = m.S(a, "front-side"), e = m.S(a, "back-side"), f = m.S(a, "flipping");
        m.C(b, f);
        m.MC(b, function() {
            m.D(b, f)
        });
        c ? m.Dl(b, d, e) : m.Dl(b, e, d)
    };
    m.PC = function() {
        var a = m.u("PAGE_NAME");
        return a && "watch" != a ? (m.db(Error("getMoviePlayer called on " + a), "WARNING"), null) : m.Rg()
    };
    var QC = function(a, b, c, d, e) {
        m.t.call(this);
        this.da = a;
        this.Ri=!1;
        this.g = [];
        this.C = null;
        a = m.kl(window.document.location.href) + "/share_ajax";
        this.C = m.R(a, {
            format: "JSON",
            Z: {
                action_get_email: 1,
                video_id: c,
                list: d,
                channel_id: e
            },
            R: function(a, c) {
                this.da.innerHTML = c.email_html;
                this.Qc();
                this.focus();
                var d = c.sharing_binary_url;
                d && Zfa(this, d, c.contacts, b)
            },
            context: this
        })
    };
    var Zfa = function(a, b, c, d) {
        m.rd(b, (0, m.p)(function() {
            if (!this.ha()) {
                var a = m.n("yt.sharing.ContactTools");
                a && a.createContactTools(this.k, c, d)
            }
        }, a))
    };
    var RC = function(a) {
        a.Ri=!1;
        a.k.value = "";
        a.j.value = "";
        a.D.innerHTML = "";
        m.P(a.Ti);
        m.K(a.Db)
    };
    var SC = function(a, b, c) {
        m.t.call(this);
        this.da = a;
        this.g = [];
        this.D = null;
        a = m.kl(window.document.location.href) + "/share_ajax";
        this.D = m.R(a, {
            format: "JSON",
            Z: {
                action_get_embed: 1,
                video_id: b,
                list: c
            },
            R: function(a, b) {
                this.da.innerHTML = b.embed_html;
                this.Xd = b.iframe_url;
                this.Ds = b.iframe_code;
                this.Ij = b.alternate_embed_urls;
                this.Qc();
                var c = m.Ef.getInstance();
                this.Wd && (this.Wd.checked=!m.Jf(0, 2));
                this.Ud && (this.Ud.checked=!m.Jf(0, 137));
                this.Vd && (this.Vd.checked=!m.Jf(0, 138));
                this.kh.checked = m.Jf(0, 44);
                t: {
                    c = c.get("ems");
                    "custom" == c && m.K(m.F("share-embed-customize"));
                    for (var h = 0; h < this.Ld.length; h++) {
                        var k = this.Ld[h];
                        if (k.value == c) {
                            k.selected=!0;
                            m.tp.getInstance().Jb(this.ze);
                            break t
                        }
                    }
                    this.Ld[0].selected=!0;
                    m.tp.getInstance().Jb(this.ze)
                }
                TC(this);
                this.Ge()
            },
            context: this
        })
    };
    var $fa = function(a) {
        a.ze = m.F("embed-layout-options");
        a.Ld = m.Mb("option", null, a.ze);
        var b = (0, window.parseInt)(m.I(a.Ld[0], "width"), 10), c = (0, window.parseInt)(m.I(a.Ld[0], "height"), 10);
        a.F = b / c;
        a.g.push(m.J(a.ze, "change", (0, m.p)(function() {
            var a = up(this.ze), a = a ? a.value || a.text: null, b = m.Ef.getInstance();
            b.set("ems", a);
            b.save();
            TC(this);
            b = m.F("share-embed-customize");
            "custom" == a ? m.K(b) : (m.P(b), this.focus())
        }, a)));
        b = m.F("share-embed-customize");
        a.C = m.G("share-embed-size-custom-width", b);
        a.k = m.G("share-embed-size-custom-height",
        b);
        a.g.push(m.J(a.C, "change", (0, m.p)(a.As, a)));
        a.g.push(m.J(a.k, "change", (0, m.p)(a.zs, a)))
    };
    var aga = function(a) {
        var b = {}, c = m.M("share-embed-option", a.da);
        (0, m.y)(c, function(a) {
            b[a.name] = a
        });
        a.Wd = b["show-related"];
        a.Wd && a.g.push(m.J(a.Wd, "click", (0, m.p)(a.Tv, a)));
        a.Ud = b["show-controls"];
        a.Ud && a.g.push(m.J(a.Ud, "click", (0, m.p)(a.Rv, a)));
        a.Vd = b["show-info"];
        a.Vd && a.g.push(m.J(a.Vd, "click", (0, m.p)(a.Sv, a)));
        a.kh = b["delayed-cookies"];
        a.g.push(m.J(a.kh, "click", (0, m.p)(a.Qv, a)))
    };
    var TC = function(a) {
        var b, c, d;
        d = a.Ds;
        var e = a.Xd;
        a.kh.checked && (e = e.replace("youtube.com", "youtube-nocookie.com"));
        e = e.split("//");
        e[0] = "";
        var e = e.join("//"), f = {};
        a.Wd&&!a.Wd.checked && (f.rel = 0);
        a.Ud&&!a.Ud.checked && (f.controls = 0);
        a.Vd&&!a.Vd.checked && (f.showinfo = 0);
        var e = m.bl(e, f), f = (0, window.parseInt)(m.I(a.Ld[0], "width"), 10), h = (0, window.parseInt)(m.I(a.Ld[0], "height"), 10);
        (c = up(a.ze)) ? "custom" == c.value ? (b = a.Xf, c = a.Wf) : (b = (0, window.parseInt)(m.I(c, "width"), 10), c = (0, window.parseInt)(m.I(c, "height"),
        10)) : c = b = 0;
        if (!b || 200 > b)
            b = f, c = h;
        e = d = d.replace(/__url__/g, m.wa(e));
        d = d.replace(/__width__/g, b + "");
        d = d.replace(/__height__/g, c + "");
        e = e.replace(/__width__/g, f + "");
        e = e.replace(/__height__/g, h + "");
        m.F("video-preview").innerHTML = e;
        d != a.j.value && (a.j.value = d)
    };
    var UC = function(a) {
        m.ym.call(this);
        this.j = {};
        this.g = {};
        this.aa = new m.ir(this);
        this.k = a
    };
    var bga = function() {
        this.j = m.F(m.F("gif-suggestion-menu-entry-template"));
        this.g = m.tA(this.j, ["start_time_ms", "end_time_ms", "display_value"])
    };
    m.VC = function(a, b, c) {
        m.ym.call(this);
        this.target = a;
        this.k = b || a;
        this.C = c || new m.Lj(window.NaN, window.NaN, window.NaN, window.NaN);
        this.j = m.Yb(a);
        this.g = new m.ir(this);
        m.Ta(this, this.g);
        m.im(this.k, ["touchstart", "mousedown"], this.Si, !1, this)
    };
    m.WC = function(a, b) {
        a.C = b || new m.Lj(window.NaN, window.NaN, window.NaN, window.NaN)
    };
    var XC = function(a) {
        var b = a.type;
        "touchstart" == b || "touchmove" == b ? a.init(a.wa.targetTouches[0], a.currentTarget) : "touchend" != b && "touchcancel" != b || a.init(a.wa.changedTouches[0], a.currentTarget)
    };
    var YC = function(a, b, c) {
        var d = m.Gi(m.Ii(a.j));
        b += d.x - a.D.x;
        c += d.y - a.D.y;
        a.D = d;
        a.Oe += b;
        a.deltaY += c;
        b = ZC(a, a.Oe);
        a = $C(a, a.deltaY);
        return new m.ui(b, a)
    };
    var aD = function(a, b, c, d) {
        a.target.style.left = c + "px";
        a.target.style.top = d + "px";
        a.dispatchEvent(new bD("drag", a, b.clientX, b.clientY, 0, c, d))
    };
    var ZC = function(a, b) {
        var c = a.C, d = (0, window.isNaN)(c.left) ? null: c.left, c = (0, window.isNaN)(c.width) ? 0: c.width;
        return Math.min(null != d ? d + c : window.Infinity, Math.max(null != d ? d : - window.Infinity, b))
    };
    var $C = function(a, b) {
        var c = a.C, d = (0, window.isNaN)(c.top) ? null: c.top, c = (0, window.isNaN)(c.height) ? 0: c.height;
        return Math.min(null != d ? d + c : window.Infinity, Math.max(null != d ? d : - window.Infinity, b))
    };
    var bD = function(a, b, c, d, e, f, h) {
        m.$l.call(this, a);
        this.clientX = c;
        this.clientY = d;
        this.left = m.ca(f) ? f : b.Oe;
        this.top = m.ca(h) ? h : b.deltaY
    };
    var cD = function(a) {
        var b;
        m.La([0, 1, 2, 3], b) || (b = 1);
        var c = Math.pow(10, 3 - b);
        a = Math.ceil(a / c) * c;
        var d = Math.floor(a / 36E5);
        a -= 36E5 * d;
        var e = Math.floor(a / 6E4);
        a -= 6E4 * e;
        var c = Math.floor(a / 1E3), f = d ? d + ":": "", d = (d && 10 > e ? "0" + e : e) + ":", e = 10 > c ? "0" + c: c, h = "";
        b && (h = m.Tl(a - 1E3 * c, 3), h = "." + h.substring(0, b));
        return f + d + e + h
    };
    var dD = function(a, b, c) {
        this.J = m.F(a);
        this.media = b;
        this.H = c;
        this.C = []
    };
    var eD = function(a, b, c, d) {
        b = m.J(b, c, (0, m.p)(d, a));
        a.C.push(b)
    };
    m.fD = function(a, b, c, d, e, f) {
        dD.call(this, a, b, c);
        this.g = {};
        this.g.Oa = this.M("left-trimmer");
        this.g.jl = this.M("start-readout");
        this.g.Vq = this.M("start-grayout");
        this.g.Ya = this.M("right-trimmer");
        this.g.il = this.M("end-readout");
        this.g.Uq = this.M("end-grayout");
        this.j = tba(this.M("left-trimmer"), this.M("clip-trimmer")).y;
        this.width = this.M("framestrip").offsetWidth;
        this.fps = d || 15;
        this.F = e ? Math.min(e, b.length_ms) : 1E3 / this.fps;
        this.D = f ? Math.min(f, b.length_ms) : b.length_ms;
        this.Oa = new m.VC(this.g.Oa, this.M("knurling-area",
        this.g.Oa));
        this.Ya = new m.VC(this.g.Ya, this.M("knurling-area", this.g.Ya));
        eD(this, this.Oa, "drag", this.nl);
        eD(this, this.Ya, "drag", this.nl);
        eD(this, this.Oa, "end", this.pl);
        eD(this, this.Ya, "end", this.pl);
        eD(this, this.M("nudge-left", this.g.Oa), "click", this.Pg);
        eD(this, this.M("nudge-right", this.g.Oa), "click", this.Pg);
        eD(this, this.M("nudge-left", this.g.Ya), "click", this.Pg);
        eD(this, this.M("nudge-right", this.g.Ya), "click", this.Pg);
        this.L = this.M("clip-trimmer");
        this.k = m.im(this.L, ["touchstart", "mousedown"],
        this.Wq, !0, this);
        a = eba(this.M("framestrip"));
        (0, m.y)(a, function(a) {
            a.src = b.framestrip_url
        });
        gD(this);
        hD(this, !0)
    };
    var iD = function(a, b, c) {
        var d, e, f = 1E3 / a.fps;
        c ? (d = Math.max(0, a.media.end_ms - a.D), e = a.media.end_ms - a.F) : (d = a.media.start_ms + a.F, e = Math.min(a.media.start_ms + a.D, a.media.length_ms));
        b < d ? b = d : b > e && (b = e);
        b = Math.round(a.fps * b / 1E3);
        b = Math.round(b / a.fps * 1E3);
        !c && b + f / 2 > a.media.length_ms && (b = a.media.length_ms);
        a.media[c ? "start_ms": "end_ms"] = b
    };
    var gD = function(a) {
        var b = new OA(Math.round(a.media.start_ms / a.media.length_ms * a.width), a.j), c = new OA(Math.round(a.media.end_ms / a.media.length_ms * a.width), a.j);
        b.Ua(a.g.Oa, 2);
        c.Ua(a.g.Ya, 0)
    };
    var hD = function(a, b) {
        var c = a.media.duration_scale || 1, d = Math.round(c * a.media.start_ms), c = Math.round(c * a.media.end_ms);
        m.pl(m.jk(a.g.jl), cD(d));
        m.pl(m.jk(a.g.il), cD(c));
        for (var e = [a.g.Oa, a.g.Ya], d = [a.g.jl, a.g.il], f = Array(2), c = Array(2), h = 0; 2 > h; h++) {
            var k = e[h];
            f[h] = m.jk(d[h]).offsetWidth;
            c[h] = new OA(k.offsetLeft + k.offsetWidth / 2 - f[h] / 2, k.offsetTop + k.clientHeight + 5)
        }
        e = c[1].Yd.x - (c[0].Yd.x + f[0]);
        4 > e && (c[0].Yd.x -= (4 - e) / 2, c[1].Yd.x += (4 - e) / 2);
        c[0].Ua(d[0], 0);
        c[1].Ua(d[1], 0);
        d = a.g.Oa.offsetWidth;
        m.WC(a.Oa, new m.Lj(0 -
        d, a.j, a.g.Ya.offsetLeft, 0));
        m.WC(a.Ya, new m.Lj(a.g.Oa.offsetLeft + d, a.j, a.width - a.g.Oa.offsetLeft - d, 0));
        jD(a);
        a.H && a.H(a.media, b ? "start_ms" : "end_ms", !0, void 0)
    };
    var jD = function(a) {
        var b = a.g.Ya.offsetWidth, c = a.g.Oa.offsetLeft + 3;
        a.g.Vq.style.width = 0 < c ? c + "px" : "0";
        b = a.width - a.g.Ya.offsetLeft - b + 3;
        a.g.Uq.style.width = 0 < b ? b + "px" : "0"
    };
    var kD = function() {};
    var cga = function(a) {
        kD = a
    };
    var lD = function(a, b) {
        var c = ["/gen_204?a=editor_ajax&err=", b, "&url=", (0, window.escape)(a)].join("");
        m.sc(c)
    };
    m.mD = function(a, b, c, d, e) {
        var f = {};
        e && (f.feature = e);
        m.tb({}, function(a, b) {
            f[b] = a
        });
        f["action_" + a] = 1;
        c["action_" + a] = 1;
        c[m.u("XSRF_FIELD_NAME")] = m.Kl();
        a = {
            format: "JSON",
            headers: {
                "Content-Type": "application/json"
            },
            method: "POST",
            to: !0,
            Rb: m.ag(c),
            Z: f,
            R: function(a, c) {
                b(c)
            },
            onError: function(a) {
                var b = Qj(a);
                a = a.responseText || m.hb(503 == b ? "AJAX_REQUEST_503" : "AJAX_REQUEST_FAILED");
                kD(a);
                lD("/editor_ajax", b);
                d && d()
            }
        };
        m.R("/editor_ajax", a)
    };
    var nD = function(a) {
        var b = [], c = {
            vc: (0, m.sh)(a, oD),
            ac: (0, m.sh)(b, oD)
        };
        0 == a.length && 1 == b.length && (c.ac[0].title = b[0].title || "", c.ac[0].artist = b[0].artist || "");
        return c
    };
    var oD = function(a) {
        var b = {
            type: a.type,
            id: a.id,
            start_ms: a.start_ms,
            end_ms: a.end_ms,
            length_ms: a.length_ms,
            effects: dga(a.effects)
        };
        a.image_type && (b.image_type = a.image_type);
        a.position && (b.position = a.position);
        return b
    };
    var dga = function(a) {
        if (!a)
            return [];
        var b = [];
        (0, m.y)(a, function(a) {
            a.enabled && b.push({
                id: a.id,
                parameters: m.yb(a.parameters)
            })
        });
        return b
    };
    m.pD = function(a, b, c) {
        m.t.call(this);
        this.da = a;
        this.D = {};
        this.g = [];
        this.S = null;
        a = m.kl(window.document.location.href) + "/share_ajax";
        c = {
            action_get_gif: 1,
            video_id: c
        };
        cga((0, m.p)(this.Ff, this));
        this.J = new UC;
        this.S = m.R(a, {
            format: "JSON",
            Z: c,
            R: function(a, b) {
                this.da.innerHTML = b.inner_html;
                this.Qc();
                this.Ma = b.media;
                this.ut = m.Ja(this.Ma.effects, qD("TOP"));
                this.tt = m.Ja(this.Ma.effects, qD("BOTTOM"));
                var c = m.G("share-panel-gif-trimmer", void 0);
                this.Ah = new m.fD(c, this.Ma, (0, m.p)(this.qu, this));
                c = m.PC();
                c = Math.round(1E3 *
                c.getCurrentTime());
                c = [c, c + 3E3];
                c = m.rD(this, c, !0);
                this.Md.value = cD(c[0]);
                this.Ig.value = cD(c[1]);
                sD(this, c[0], c[1]);
                this.lj=!0;
                tD(this);
                this.Jg && (c = (0, m.p)(this.ou, this), m.mD("suggest_loops", c, {
                    video_id: this.Ma.id
                }))
            },
            context: this
        })
    };
    var qD = function(a) {
        return function(b) {
            return "BANNER" == b.id && b.parameters.v_align == a
        }
    };
    var uD = function(a, b) {
        var c = m.PC();
        1 == c.getPlayerState() && c.pauseVideo();
        if (Math.round(1E3 * c.getCurrentTime()) != b) {
            var d = b / 1E3;
            b >= a.Ma.length_ms - 1E3 && (d = a.Ma.length_ms / 1E3 - 1);
            c.seekTo(d, !0)
        }
    };
    var sD = function(a, b, c) {
        if (a.Ah) {
            var d = a.Ah;
            d.media.start_ms = b;
            d.media.end_ms = c;
            gD(a.Ah);
            jD(a.Ah)
        }
    };
    m.vD = function(a, b, c) {
        a.Md.value = cD(b[0]);
        a.Ig.value = cD(b[1]);
        sD(a, b[0], b[1]);
        c && tD(a)
    };
    m.rD = function(a, b, c) {
        500 > b[1] - b[0] ? c ? b[1] = b[0] + 500 : b[0] = b[1] - 500 : 6E3 < b[1] - b[0] && (c ? b[1] = b[0] + 6E3 : b[0] = b[1] - 6E3);
        0 > b[0] && (b[1] -= b[0], b[0] = 0);
        b[1] > a.Ma.length_ms && (c = b[1] - b[0], b[1] = a.Ma.length_ms, b[0] = Math.max(0, b[1] - c));
        return b
    };
    var ega = function(a) {
        var b = 0, c = a.split(".");
        if (2 < c.length)
            return null;
        2 == c.length && (b += 100 * (0, window.parseInt)(c[1].match(/\d+/g), 10));
        if (a = c[0].match(/\d+/g)) {
            if (3 < a.length)
                return null;
            a.reverse();
            for (c = 1E3; a.length && 36E5 >= c;)
                b += c * a.shift(), c*=60
        } else if (2 > c.length)
            return null;
        return b
    };
    var wD = function(a, b, c, d, e) {
        var f = {};
        m.tb(e || {}, function(a, b) {
            f[b] = a
        });
        f["action_" + a] = 1;
        c["action_" + a] = 1;
        c[m.u("XSRF_FIELD_NAME")] = m.u("XSRF_TOKEN");
        a = {
            format: "JSON",
            headers: {
                "Content-Type": "application/json"
            },
            method: "POST",
            to: !0,
            Rb: m.ag(c),
            Z: f,
            R: function(a, c) {
                b(c)
            },
            onError: function(a) {
                var b = Qj(a);
                a = a.responseText || m.hb(503 == b ? "EDITOR_AJAX_REQUEST_503" : "EDITOR_AJAX_REQUEST_FAILED");
                lD("/share_gif_ajax", b);
                d && d(a)
            }
        };
        m.R("/share_gif_ajax", a)
    };
    var tD = function(a) {
        var b = a.F ? (0, m.Fa)(a.F.value): "", c = a.L ? (0, m.Fa)(a.L.value): "";
        !a.lj || a.D.start_ms == a.Ma.start_ms && a.D.end_ms == a.Ma.end_ms && a.D.top_text == b && a.D.bottom_text == c || (a.ba && m.da.clearTimeout(a.ba), a.D = {
            start_ms: a.Ma.start_ms,
            end_ms: a.Ma.end_ms,
            top_text: b,
            bottom_text: c
        }, a.ba = m.Cm((0, m.p)(a.hn, a), 120))
    };
    var xD = function(a, b, c) {
        m.C(a.C, "loading");
        a.O.src = "";
        a.O.src = b;
        (b = m.ja(b) ? b : b.src) && (a.J.j["gif-preview"] = {
            src: b,
            tl: m.ca(void 0) ? void 0: null
        });
        m.sm(a.ja);
        a.ja = m.im(a.J, "complete", c);
        a.J.start()
    };
    m.yD = function(a, b, c, d, e, f, h, k, r, w, B, Q, Y, la) {
        m.t.call(this);
        this.da = a;
        this.C = b || null;
        this.T = c || null;
        this.qa=!1;
        this.Yb = e || null;
        this.La=!!f;
        this.Ba=!!h;
        this.Ca = r || null;
        this.Ed = w || null;
        this.$a = B || null;
        this.Qa = Q || null;
        this.Qb = k || null;
        this.kb = Y || null;
        this.vf = la || null;
        this.ja = null;
        this.g = [];
        fga(this, d)
    };
    var gga = function(a) {
        var b = ["h", "m", "s"], c = m.Ra(b);
        c.reverse();
        var d = {};
        a = a.toLowerCase().match(/\d+\s*[hms]?/g) || [];
        a = (0, m.qb)(a, function(a) {
            var b = (a.match(/[hms]/) || [""])[0];
            return b ? (d[b] = (0, window.parseInt)(a.match(/\d+/)[0], 10), !1) : !0
        });
        for (a.reverse(); a.length && c.length;) {
            var e = c.shift();
            e in d || (d[e] = (0, window.parseInt)(a.shift(), 10))
        }
        if (a.length || 59 < d.s || 59 < d.m || 9 < d.h)
            return null;
        var f = "";
        (0, m.y)(b, function(a) {
            d[a] && (f += d[a] + a)
        });
        return f || null
    };
    var fga = function(a, b) {
        var c = m.kl(window.document.location.href) + "/share_ajax";
        a.ja = m.R(c, {
            format: "JSON",
            Z: {
                action_get_share_box: 1,
                video_id: a.C,
                list: a.T,
                feature: a.Yb,
                share_at: !a.Ba,
                caption_text: a.Ca,
                title: a.Ed,
                image_url: a.$a,
                t: a.Qa,
                referer: a.Qb,
                max_count: a.kb,
                index: a.vf
            },
            R: function(a, c) {
                this.da.innerHTML = c.share_html;
                this.am = c.url_short;
                this.gx = c.lang;
                this.Zj = null;
                "session_index"in c && (this.Zj = c.session_index);
                this.Vn = null;
                "delegated_session_id"in c && (this.Vn = c.delegated_session_id);
                this.Qc();
                b && b();
                this.Ge()
            },
            context: a
        })
    };
    var zD = function(a, b) {
        a.F && b != a.F && m.P(a.F);
        a.H && b != a.H && m.P(a.H);
        a.D && b != a.D && m.P(a.D);
        a.L && b != a.L && m.P(a.L)
    };
    var AD = function(a) {
        m.F("share-with-playlist").checked=!0;
        BD(a)
    };
    var BD = function(a) {
        if (a.j) {
            var b = m.F("share-with-playlist").checked;
            a.J.disabled = b;
            a.j.disabled = b;
            a.Ga && m.wf(a.Ga, !b)
        }
    };
    var CD = function(a, b, c) {
        var d = {
            action_get_share_urls: 1,
            video_id: a.C,
            caption_text: a.Ca
        };
        b && (d.list = a.T);
        c && (d.use_first_video=!0);
        m.R("share_ajax", {
            format: "JSON",
            Z: d,
            R: function(a, b) {
                this.am = b.url_short;
                this.zf();
                m.F("share-services-container").innerHTML = b.share_services_html
            },
            context: a
        })
    };
    m.DD = function(a, b, c) {
        a&&!ED && (ED=!0, FD = b, c && (GD.hide = function(a) {
            HD(a, c)
        }), ID.push(m.O(window.document.documentElement, "click", hga, "dismiss-menu-choice")), ID.push(m.O(window.document.documentElement, "click", iga, "undo-replace-action")), ID.push(m.O(m.F("body-container"), "click", jga, "action-never-show-in-feed")), ID.push(m.O(a, "click", kga, "feed-promo-dismissal")), ID.push(m.O(a, "click", lga, "feed-item-dismissal-activate-question")), JD.push(m.x("subscription-subscribe-success", mga), m.x("subscription-unsubscribe-success",
        nga)))
    };
    m.KD = function() {
        ED=!1;
        GD.hide = HD;
        m.N(ID);
        ID.length = 0;
        m.nb(JD);
        JD.length = 0
    };
    var hga = function(a) {
        var b = m.I(a.currentTarget, "action");
        GD[b](a.currentTarget)
    };
    var kga = function(a) {
        a = m.L(a.currentTarget, "feed-promo");
        var b = (0, window.parseInt)(m.I(a, "cookie-id"), 10), c = m.Ef.getInstance();
        m.ji(b, !0);
        c.save();
        "i18n-local-languages-feed-promo" == a.id && (a = {
            gl: m.I(a, "promo-gl")
        }, a = m.qc(a), m.Pj("i18n-promo-close", a, void 0))
    };
    var lga = function(a) {
        a = m.L(a.currentTarget, "feed-item-dismissal-notices");
        a = m.G("swappable-shelf", a);
        m.wk(a);
        m.z("yt-dom-content-change", a)
    };
    var mga = function(a, b) {
        LD(a, function(a) {
            if (a = m.G("feed-item-action-menu", a))
                m.C(a, "subscribed"), (a = m.G("unsubscribe", a)) && m.fl(a, "subscription-id", b)
        })
    };
    var nga = function(a, b) {
        if (!FD) {
            if (b && m.A(b, "dismiss-menu-choice")) {
                var c = m.L(b, "feed-item-container"), d = m.L(b, "feed-item-dismissable");
                MD(d);
                c = m.G("feed-item-dismissal-unsubscribe", c);
                m.K(c)
            }
            LD(a, function(a) {
                a = m.G("feed-item-dismissable", a);
                ND(a)
            })
        }
    };
    var iga = function(a) {
        var b = a.currentTarget;
        OD(b);
        a = m.L(b, "yt-lockup");
        var b = m.L(b, "feed-item-container"), c, d;
        if (a)
            c = m.G("yt-lockup-notifications-container", a), d = m.G("yt-lockup-dismissable", a);
        else if (b)
            c = m.G("feed-item-dismissal-notices", b), d = m.G("feed-item-dismissable", b);
        else 
            return;
        c && m.P(c);
        d && (c = d, m.K(c), m.rt(c, 0, 1, {
            duration: .3
        }));
        m.z("yt-dom-content-change", a || b)
    };
    var OD = function(a) {
        (a = m.I(a, "feedback-token")) && m.R("/feed_change_ajax?action_give_feedback=1", {
            method: "POST",
            V: {
                session_token: m.Kl(),
                feedback_tokens: [a],
                wait_for_response: 1
            }
        })
    };
    var HD = function(a, b) {
        var c = m.I(a, "dismissal-token"), d = m.I(a, "feedback-token"), e=!m.A(a, "no-notification"), f = m.pd(a, "redirect-url"), h = {
            session_token: m.Kl()
        }, k = "";
        c ? b ? b(c) : (h.dismissal_token = c, m.u("DISMISS_THROUGH_IT") && (h.it = 1), k = "action_dismiss_item") : d && (k = "action_give_feedback", h.feedback_tokens = [d], h.wait_for_response = f ? 0 : 1);
        k && m.R("/feed_change_ajax?" + k + "=1", {
            method: "POST",
            V: h,
            R: function() {
                f && (window.location.href = m.I(a, "redirect-url"))
            }
        });
        f || (d = m.L(a, "feed-item-container"), MD(d), c = m.G("feed-item-dismissal-hide",
        d), d = m.G("feed-item-dismissable", d), e && m.K(c), ND(d))
    };
    var jga = function() {
        HD(PD)
    };
    var QD = function(a) {
        if (!m.I(a, "ypc-enabled"))
            return null;
        var b = m.I(a, "ypc-item-type"), c = m.I(a, "ypc-item-id");
        a = m.I(a, "ypc-offers-url");
        return {
            itemType: b,
            itemId: c,
            offersUrl: a
        }
    };
    var RD = function(a, b) {
        var c = m.I(a, "channel-key");
        LD(c, b)
    };
    var LD = function(a, b) {
        var c = m.M("feed-item-container");
        (0, m.y)(c, function(c) {
            m.I(c, "channel-key") == a && b(c)
        })
    };
    var MD = function(a) {
        a = m.M("feed-item-dismissal", a);
        m.P.apply(null, a)
    };
    var ND = function(a) {
        m.st(a, {
            duration: .3,
            ka: function() {
                m.P(a);
                m.z("yt-dom-content-change", a)
            }
        })
    };
    m.SD = function(a) {
        var b = m.F("yt-comments-dftpop");
        b && m.R("/watch_actions_ajax", {
            method: "POST",
            format: "JSON",
            Z: {
                action_display_plus_one_promo: "1"
            },
            V: {
                session_token: m.Kl()
            },
            R: function(c, d) {
                if (d.show_promo)
                    t: {
                    if (m.ik(a))
                        a.appendChild(b);
                    else {
                        var e = m.F("yt-comments-dftpop-content");
                        if (!e)
                            break t;
                            vk(e, "left", a.x - 25 + "px");
                            vk(e, "top", a.y + 10 + "px")
                    }
                    m.C(b, "show")
                } else 
                    m.ri(b)
            }
        })
    };
    var oga = function(a) {
        var b = m.n("yt.www.watch.player.seekTo");
        b && b(a)
    };
    var pga = function() {
        m.z("yt-www-comments-sharebox-open")
    };
    var qga = function() {
        var a = m.F("distiller-spinner");
        a && m.P(a)
    };
    var TD = function() {
        var a = m.F("distiller-alert");
        a && m.K(a)
    };
    var rga = function(a, b, c) {
        m.R("/comment_voting", {
            format: "XML",
            method: "POST",
            Z: {
                a: 1,
                id: c,
                video_id: a,
                old_vote: 0
            },
            V: {
                session_token: b
            }
        })
    };
    var UD = function(a, b) {
        if (a)
            if ("#" == a) {
                var c = m.n(b);
                c && c("comment")
            } else 
                m.ae(a);
        else 
            TD()
    };
    var sga = function(a) {
        var b = a.channel_id, c = a.create_channel_url, d = a.query, e = a.id_merge_url, f = a.owner_id, h = a.page_size, k = a.privacy_setting, r = a.reauth, w = a.signin_url, B = a.video_id;
        a = a.viewer_id;
        var Q=!!m.u("DISTILLER_CONFIG"), Y=!B, la = null;
        m.ca(w) ? la = r && Q ? TD : m.q(m.ae, w, null, null) : m.ca(c) ? la = m.q(UD, c, "yt.www.account.CreateChannelLoader.show") : m.ca(e) && (la = m.q(UD, e, "yt.www.account.LinkGplusLoader.show"));
        b = {
            first_party_property: "YOUTUBE",
            href: window.top.location,
            onfirsttimeplusonepromo: m.SD,
            onthumbsup: m.q(rga,
            B, m.Kl()),
            ontimestampclicked: oga,
            onshareboxopen: pga,
            onready: qga,
            owner_id: f,
            query: d,
            stream_id: b,
            substream_id: Y ? "channel": B,
            view_type: "FILTERED",
            width: VD()
        };
        la && (b.onupgradeaccount = la);
        h && (b.page_size = h);
        k && (b.youtube_video_acl = k);
        a && (b.viewer_id = a);
        Y || (b.onallcommentsclicked = m.q(m.ae, "/all_comments", {
            v: B
        }));
        m.n("gapi.comments.render")("comments-test-iframe", b);
        WD = m.x("page-resize", m.XD)
    };
    m.XD = function() {
        var a = m.F("comments-test-iframe"), b = VD();
        a && b && (a = m.Mb("iframe", null, a), a.length && (a[0].style.maxWidth = b + "px"))
    };
    var VD = function() {
        var a = m.F("comments-test-iframe");
        return (a = a && m.rl(a)) ? Math.floor(sba(a).width) : 0
    };
    m.YD = function() {
        var a = m.u("DISTILLER_CONFIG"), b = m.F("comments-test-iframe");
        if (!a ||!b)
            return !1;
        var c = "signin_url"in a, b = m.Tn(m.q(sga, a), "comments", c ? ":socialhost:/:im_prefix::session_prefix:_/widget/render/comments?usegapi=1" : ":socialhost:/:im_prefix::session_prefix:wm/4/_/widget/render/comments?usegapi=1");
        c && (b.config["googleapis.config"].signedIn=!1);
        (a = a.host_override) && (b.config.iframes[":socialhost:"] = a);
        500 > window.location.search.length + window.location.hash.length && (b.config.iframes.comments.params =
        {
            location: ["search", "hash"]
        });
        m.Sn("comments:iframes", b);
        return !0
    };
    m.ZD = function() {
        var a = m.n("gapi.comments.dispose");
        a && a();
        WD && (m.nb(WD), WD = null)
    };
    m.$D = function() {
        (aE = m.F("feed")) || (aE = m.F("browse-items-primary"));
        m.bE.push(m.x("page-resize", cE));
        cE();
        m.bE.push(m.x("yt-uix-expander-toggle", tga))
    };
    var cE = function() {
        var a = m.M("multirow-shelf", aE);
        (0, m.y)(a, function(a) {
            var c = 0, d = null, e = m.Mb("li", "yt-shelf-grid-item", a);
            (0, m.y)(e, function(a) {
                m.gj(a) && (a = Kj(a).x, (a <= d ||!d) && c++, d = a)
            });
            a = m.G("yt-uix-expander-head", a);
            m.rb(a, "hidden-expander", 2 >= c)
        })
    };
    var tga = function(a) {
        m.vi(aE, a) && m.z("yt-dom-content-change", aE)
    };
    m.dE = function(a, b, c) {
        sy({
            device: "Desktop",
            app: "youtube-desktop",
            isSignedIn: m.u("LOGGED_IN"),
            loadAccountScreens: b
        });
        eE = a;
        eE.addEventListener("onReady", fE);
        eE.addEventListener("onRemoteReceiverSelected", gE);
        hE.push(m.x("yt-remote-receiver-availability-change", iE));
        hE.push(m.x("yt-remote-auto-connect", jE));
        c && hE.push(m.x("yt-remote-cast-device-tab-projected", jE))
    };
    m.kE = function() {
        m.nb(hE);
        hE.length = 0;
        eE && (eE.removeEventListener("onRemoteReceiverSelected", gE), eE.removeEventListener("onReady", fE), eE = null);
        Ey()
    };
    var fE = function() {
        iE();
        if (m.Ly()) {
            var a = eE;
            "html5" != a.getPlayerType() && a.loadNewVideoConfig(a.getCurrentVideoConfig(), "html5")
        }
    };
    var gE = function(a) {
        "cast-selector-receiver" == a ? Xv() : jE(a)
    };
    var jE = function(a) {
        var b = Gy();
        if (a = zea(b, a)) {
            var c = eE, d = c.getVideoData().video_id, e = c.getVideoData().list, f = c.getCurrentTime();
            vfa(a, {
                videoIds: [d],
                listId: e,
                videoId: d,
                index: 0,
                currentTime: f
            });
            "html5" != c.getPlayerType() ? c.loadNewVideoConfig(c.getCurrentVideoConfig(), "html5") : c.updateRemoteReceivers && c.updateRemoteReceivers(b, a)
        }
    };
    var iE = function() {
        var a = eE;
        a && a.updateRemoteReceivers && a.updateRemoteReceivers(Gy(), m.Ly())
    };
    m.lE = function(a) {
        mE = a;
        mE.addEventListener("SUBSCRIBE", uga);
        mE.addEventListener("UNSUBSCRIBE", nE);
        oE.push(m.x("subscription-subscribe-success", vga));
        oE.push(m.x("subscription-unsubscribe-success", wga))
    };
    var pE = function(a) {
        return {
            externalChannelId: a.externalChannelId,
            Cp: a.offersUrl,
            source: a.source,
            Ui: a.subscriptionId
        }
    };
    var uga = function(a) {
        qE(pE(a))
    };
    var qE = function(a) {
        Lq() ? (m.Dq(rE, new ao(a.externalChannelId, a.Cp ? {
            itemType: "U",
            itemId: a.externalChannelId,
            offersUrl: a.Cp
        } : null)), m.Bd(m.qc({
            event: "subscribe",
            source: a.source
        }))) : xga(a)
    };
    var xga = function(a) {
        Kq(function(b) {
            b.subscription_ajax && qE(a)
        }, null)
    };
    var nE = function(a) {
        a = pE(a);
        m.Dq(sE, new bo(a.externalChannelId, a.Ui, null));
        m.Bd(m.qc({
            event: "unsubscribe",
            source: a.source
        }))
    };
    var vga = function(a, b) {
        mE && mE.channelSubscribed(a, b)
    };
    var wga = function(a) {
        mE && mE.channelUnsubscribed(a)
    };
    m.tE = function() {
        mE && (mE.removeEventListener("SUBSCRIBE", qE), mE.removeEventListener("UNSUBSCRIBE", nE));
        mE = null;
        m.nb(oE)
    };
    m.uE = function(a) {
        a = m.M("yt-creator-tip-list", a);
        (0, m.y)(a, function(a) {
            m.Qk(a, m.vE, "a")
        })
    };
    m.wE = function(a, b) {
        var c = m.I(a, "name"), d = m.I(a, "video-id"), e = m.I(a, "location"), f = m.I(a, "ui-type"), h = {};
        h[m.u("XSRF_FIELD_NAME")] = m.Kl();
        c = {
            new_state: b,
            suggestion: c,
            location: e,
            ui_type: f
        };
        d ? (c.action_update_video_suggestion = 1, c.video_id = d, "EVENT_ELIGIBLE" == b && (c.variation = m.I(a, "variation")), h.experiment_version = m.I(a, "experiment-version"), h.offset_timestamp = m.I(a, "offset-timestamp"), h.event_timestamp = m.I(a, "event-timestamp")) : c.action_update_channel_suggestion = 1;
        m.R("/creator_suggestions_ajax", {
            method: "POST",
            V: h,
            Z: c,
            ka: void 0,
            Zb: void 0,
            timeout: 1E3
        })
    };
    m.vE = function(a) {
        (a = m.L(a.currentTarget, "yt-creator-tip-item-container")) && m.wE(a, "EVENT_CLICKED")
    };
    m.Ve.prototype.ceil = m.hi(12, function() {
        this.top = Math.ceil(this.top);
        this.right = Math.ceil(this.right);
        this.bottom = Math.ceil(this.bottom);
        this.left = Math.ceil(this.left);
        return this
    });
    m.Qd.prototype.rf = m.hi(11, function(a) {
        for (var b = m.Rd(this).values, c = 0; c < b.length; c++)
            if (b[c] == a)
                return !0;
        return !1
    });
    m.Qd.prototype.isEmpty = m.hi(10, function() {
        return !this.g.cookie
    });
    m.Qd.prototype.za = m.hi(9, function() {
        return m.Rd(this).values
    });
    m.Qd.prototype.Sa = m.hi(8, function() {
        return m.Rd(this).keys
    });
    m.Ld.prototype.stopImmediatePropagation = m.hi(6, function() {
        this.wa.cancelBubble=!0;
        this.wa.stopImmediatePropagation && this.wa.stopImmediatePropagation()
    });
    m.Ld.prototype.stopPropagation = m.hi(5, function() {
        this.wa.cancelBubble=!0;
        this.wa.stopPropagation && this.wa.stopPropagation()
    });
    m.Ld.prototype.preventDefault = m.hi(4, function() {
        this.wa.returnValue=!1;
        this.wa.preventDefault && this.wa.preventDefault()
    });
    m.Wa.prototype.ua = m.hi(3, function(a) {
        if (a) {
            var b = this.Ob[a];
            return b ? b.length : 0
        }
        a = 0;
        for (b in this.Ob)
            a += this.ua(b);
        return a
    });
    m.Qd.prototype.ua = m.hi(2, function() {
        return this.g.cookie ? (this.g.cookie || "").split(m.Sd).length : 0
    });
    m.Wf.prototype.ua = m.hi(1, function() {
        var a = 0;
        m.ec(this.fb(!0), function() {
            a++
        });
        return a
    });
    m.Xf.prototype.ua = m.hi(0, function() {
        return this.g.length
    });
    m.g = m.pi.prototype;
    m.g.clone = function() {
        return new m.pi(this.width, this.height)
    };
    m.g.isEmpty = function() {
        return !(this.width * this.height)
    };
    m.g.ceil = function() {
        this.width = Math.ceil(this.width);
        this.height = Math.ceil(this.height);
        return this
    };
    m.g.floor = function() {
        this.width = Math.floor(this.width);
        this.height = Math.floor(this.height);
        return this
    };
    m.g.round = function() {
        this.width = Math.round(this.width);
        this.height = Math.round(this.height);
        return this
    };
    m.g.scale = function(a, b) {
        var c = m.ka(b) ? b: a;
        this.width*=a;
        this.height*=c;
        return this
    };
    m.g = m.ui.prototype;
    m.g.clone = function() {
        return new m.ui(this.x, this.y)
    };
    m.g.ceil = function() {
        this.x = Math.ceil(this.x);
        this.y = Math.ceil(this.y);
        return this
    };
    m.g.floor = function() {
        this.x = Math.floor(this.x);
        this.y = Math.floor(this.y);
        return this
    };
    m.g.round = function() {
        this.x = Math.round(this.x);
        this.y = Math.round(this.y);
        return this
    };
    m.g.scale = function(a, b) {
        var c = m.ka(b) ? b: a;
        this.x*=a;
        this.y*=c;
        return this
    };
    m.g = m.Di.prototype;
    m.g.M = function(a) {
        return m.ja(a) ? this.g.getElementById(a) : a
    };
    m.g.Ja = function(a, b) {
        return m.M(a, b || this.g)
    };
    m.g.P = function(a, b) {
        return m.G(a, b || this.g)
    };
    m.g.Ha = function(a, b, c) {
        return m.Qb(this.g, arguments)
    };
    m.g.createElement = function(a) {
        return this.g.createElement(a)
    };
    m.g.appendChild = function(a, b) {
        a.appendChild(b)
    };
    m.g.append = m.zi;
    m.g.Dh = m.ri;
    m.g.oy = m.xi;
    m.g.getChildren = m.wi;
    m.g.contains = m.vi;
    m.g.ov = m.L;
    var Ji, bba = /matrix\([0-9\.\-]+, [0-9\.\-]+, [0-9\.\-]+, [0-9\.\-]+, ([0-9\.\-]+)p?x?, ([0-9\.\-]+)p?x?\)/, Vi = {
        thin: 2,
        medium: 4,
        thick: 6
    }, qj = {};
    m.g = m.aj.prototype;
    m.g.ua = function() {
        return this.k
    };
    m.g.za = function() {
        dj(this);
        for (var a = [], b = 0; b < this.g.length; b++)
            a.push(this.j[this.g[b]]);
        return a
    };
    m.g.Sa = function() {
        dj(this);
        return this.g.concat()
    };
    m.g.rf = function(a) {
        for (var b = 0; b < this.g.length; b++) {
            var c = this.g[b];
            if (Ti(this.j, c) && this.j[c] == a)
                return !0
        }
        return !1
    };
    m.g.equals = function(a, b) {
        if (this === a)
            return !0;
        if (this.k != a.ua())
            return !1;
        var c = b || Zaa;
        dj(this);
        for (var d, e = 0; d = this.g[e]; e++)
            if (!c(this.get(d), a.get(d)))
                return !1;
        return !0
    };
    m.g.isEmpty = function() {
        return 0 == this.k
    };
    m.g.clear = function() {
        this.j = {};
        this.Kb = this.k = this.g.length = 0
    };
    m.g.remove = function(a) {
        return Ti(this.j, a) ? (delete this.j[a], this.k--, this.Kb++, this.g.length > 2 * this.k && dj(this), !0) : !1
    };
    m.g.get = function(a, b) {
        return Ti(this.j, a) ? this.j[a] : b
    };
    m.g.set = function(a, b) {
        Ti(this.j, a) || (this.k++, this.g.push(a), this.Kb++);
        this.j[a] = b
    };
    m.g.forEach = function(a, b) {
        for (var c = this.Sa(), d = 0; d < c.length; d++) {
            var e = c[d], f = this.get(e);
            a.call(b, f, e, this)
        }
    };
    m.g.clone = function() {
        return new m.aj(this)
    };
    m.g.fb = function(a) {
        dj(this);
        var b = 0, c = this.g, d = this.j, e = this.Kb, f = this, h = new m.bc;
        h.next = function() {
            for (; ;) {
                if (e != f.Kb)
                    throw Error("The map has changed since the iterator was created");
                if (b >= c.length)
                    throw m.dc;
                var h = c[b++];
                return a ? h : d[h]
            }
        };
        return h
    };
    var Ij = {}, yba = /^https?:\/\/(www\.google\.com\/pagead\/sul|www\.youtube\.com\/gen_204\?a=sul)/, zba = /^https?.*#ocr$|^https?:\/\/(secure\-..\.imrworldwide\.com\/|cdn\.imrworldwide\.com\/|aksecure\.imrworldwide\.com\/)/;
    m.g = m.rj.prototype;
    m.g.Ka = null;
    m.g.yb = null;
    m.g.ua = function() {
        sj(this);
        return this.yb
    };
    m.g.add = function(a, b) {
        sj(this);
        this.g = null;
        a = tj(this, a);
        var c = this.Ka.get(a);
        c || this.Ka.set(a, c = []);
        c.push(b);
        this.yb++;
        return this
    };
    m.g.remove = function(a) {
        sj(this);
        a = tj(this, a);
        return m.cj(this.Ka, a) ? (this.g = null, this.yb -= this.Ka.get(a).length, this.Ka.remove(a)) : !1
    };
    m.g.clear = function() {
        this.Ka = this.g = null;
        this.yb = 0
    };
    m.g.isEmpty = function() {
        sj(this);
        return 0 == this.yb
    };
    m.g.rf = function(a) {
        var b = this.za();
        return m.La(b, a)
    };
    m.g.Sa = function() {
        sj(this);
        for (var a = this.Ka.za(), b = this.Ka.Sa(), c = [], d = 0; d < b.length; d++)
            for (var e = a[d], f = 0; f < e.length; f++)
                c.push(b[d]);
        return c
    };
    m.g.za = function(a) {
        sj(this);
        var b = [];
        if (m.ja(a))
            uj(this, a) && (b = m.fj(b, this.Ka.get(tj(this, a))));
        else {
            a = this.Ka.za();
            for (var c = 0; c < a.length; c++)
                b = m.fj(b, a[c])
        }
        return b
    };
    m.g.set = function(a, b) {
        sj(this);
        this.g = null;
        a = tj(this, a);
        uj(this, a) && (this.yb -= this.Ka.get(a).length);
        this.Ka.set(a, [b]);
        this.yb++;
        return this
    };
    m.g.get = function(a, b) {
        var c = a ? this.za(a): [];
        return 0 < c.length ? String(c[0]) : b
    };
    m.g.toString = function() {
        if (this.g)
            return this.g;
        if (!this.Ka)
            return "";
        for (var a = [], b = this.Ka.Sa(), c = 0; c < b.length; c++)
            for (var d = b[c], e = m.va(d), d = this.za(d), f = 0; f < d.length; f++) {
                var h = e;
                "" !== d[f] && (h += "=" + m.va(d[f]));
                a.push(h)
            }
        return this.g = a.join("&")
    };
    m.g.clone = function() {
        var a = new m.rj;
        a.g = this.g;
        this.Ka && (a.Ka = this.Ka.clone(), a.yb = this.yb);
        return a
    };
    m.g.extend = m.aa(13);
    var yga = /#/g, iba = /[\#\?@]/g, zga = /[\#\?]/g, Aga = /[\#\?:]/g, xE = /[#\/\?@]/g, gk = {
        IMG: " ",
        BR: "\n"
    }, kba = {
        SCRIPT: 1,
        STYLE: 1,
        HEAD: 1,
        IFRAME: 1,
        OBJECT: 1
    }, pba = /&([^;\s<&]+);?/g, yE = [], zE = [];
    m.g = m.Lj.prototype;
    m.g.clone = function() {
        return new m.Lj(this.left, this.top, this.width, this.height)
    };
    m.g.contains = function(a) {
        return a instanceof m.Lj ? this.left <= a.left && this.left + this.width >= a.left + a.width && this.top <= a.top && this.top + this.height >= a.top + a.height : a.x >= this.left && a.x <= this.left + this.width && a.y >= this.top && a.y <= this.top + this.height
    };
    m.g.ceil = function() {
        this.left = Math.ceil(this.left);
        this.top = Math.ceil(this.top);
        this.width = Math.ceil(this.width);
        this.height = Math.ceil(this.height);
        return this
    };
    m.g.floor = function() {
        this.left = Math.floor(this.left);
        this.top = Math.floor(this.top);
        this.width = Math.floor(this.width);
        this.height = Math.floor(this.height);
        return this
    };
    m.g.round = function() {
        this.left = Math.round(this.left);
        this.top = Math.round(this.top);
        this.width = Math.round(this.width);
        this.height = Math.round(this.height);
        return this
    };
    m.g.scale = function(a, b) {
        var c = m.ka(b) ? b: a;
        this.left*=a;
        this.width*=a;
        this.top*=c;
        this.height*=c;
        return this
    };
    var Xba = {
        AC: "GHELP_SEARCH",
        zC: "GHELP_AUTOCOMPLETE",
        BC: "GHELP_SUGGEST",
        yC: "GHELP_ARTICLE",
        eD: "LINK",
        HC: "HELPCENTER",
        kC: "EMAIL",
        FB: "CHAT",
        tC: "FORUM",
        fD: "MARKETING",
        ED: "ONEBOX",
        CUSTOM: "CUSTOM",
        sB: "AUTO",
        NE: "TESTING"
    };
    m.g = m.Rj.prototype;
    m.g.xd = "";
    m.g.fe = "";
    m.g.Tb = "";
    m.g.ee = null;
    m.g.Mc = "";
    m.g.ig = "";
    m.g.ge=!1;
    m.g.toString = function() {
        var a = [], b = this.xd;
        b && a.push(wj(b, xE, !0), ":");
        if (b = this.Tb) {
            a.push("//");
            var c = this.fe;
            c && a.push(wj(c, xE, !0), "@");
            a.push(m.va(b).replace(/%25([0-9a-fA-F]{2})/g, "%$1"));
            b = this.ee;
            null != b && a.push(":", String(b))
        }
        if (b = this.Mc)
            this.Tb && "/" != b.charAt(0) && a.push("/"), a.push(wj(b, "/" == b.charAt(0) ? zga : Aga, !0));
        (b = this.g.toString()) && a.push("?", b);
        (b = this.ig) && a.push("#", wj(b, yga));
        return a.join("")
    };
    m.g.resolve = function(a) {
        var b = this.clone(), c=!!a.xd;
        c ? Sj(b, a.xd) : c=!!a.fe;
        c ? b.fe = a.fe : c=!!a.Tb;
        c ? m.Tj(b, a.Tb) : c = null != a.ee;
        var d = a.Mc;
        if (c)
            Uj(b, a.ee);
        else if (c=!!a.Mc) {
            if ("/" != d.charAt(0))
                if (this.Tb&&!this.Mc)
                    d = "/" + d;
                else {
                    var e = b.Mc.lastIndexOf("/");
                    - 1 != e && (d = b.Mc.substr(0, e + 1) + d)
                }
            e = d;
            if (".." == e || "." == e)
                d = "";
            else if ( - 1 != e.indexOf("./")||-1 != e.indexOf("/.")) {
                for (var d = 0 == e.lastIndexOf("/", 0), e = e.split("/"), f = [], h = 0; h < e.length;) {
                    var k = e[h++];
                    "." == k ? d && h == e.length && f.push("") : ".." == k ? ((1 < f.length || 1 ==
                    f.length && "" != f[0]) && f.pop(), d && h == e.length && f.push("")) : (f.push(k), d=!0)
                }
                d = f.join("/")
            } else 
                d = e
        }
        c ? m.Vj(b, d) : c = "" !== a.g.toString();
        c ? m.Wj(b, Zi(a.g.toString())) : c=!!a.ig;
        c && m.Xj(b, a.ig);
        return b
    };
    m.g.clone = function() {
        return new m.Rj(this)
    };
    var Im, Gm;
    m.g = m.Ik.prototype;
    m.g.ua = function() {
        return this.g.ua()
    };
    m.g.add = function(a) {
        this.g.set(Mj(a), a)
    };
    m.g.removeAll = function(a) {
        a = Si(a);
        for (var b = a.length, c = 0; c < b; c++)
            this.remove(a[c])
    };
    m.g.remove = function(a) {
        return this.g.remove(Mj(a))
    };
    m.g.clear = function() {
        this.g.clear()
    };
    m.g.isEmpty = function() {
        return this.g.isEmpty()
    };
    m.g.contains = function(a) {
        return m.cj(this.g, Mj(a))
    };
    m.g.za = function() {
        return this.g.za()
    };
    m.g.clone = function() {
        return new m.Ik(this)
    };
    m.g.equals = function(a) {
        return this.ua() == ek(a) && vba(this, a)
    };
    m.g.fb = function() {
        return this.g.fb(!1)
    };
    m.Yl = function(a, b) {
        var c = b || {}, d = m.gb(a) ? m.fb[a]: a;
        if (d)
            for (var e in c)
                var f = ("" + c[e]).replace(/\$/g, "$$$$"), d = d.replace(new RegExp("\\{\\$" + e + "\\}", "gi"), f), d = d.replace(new RegExp("\\$" + e, "gi"), f);
        return d
    };
    m.$l.prototype.N = function() {};
    m.$l.prototype.dispose = function() {};
    m.$l.prototype.stopPropagation = function() {
        this.qe=!0
    };
    m.$l.prototype.preventDefault = function() {
        this.defaultPrevented=!0;
        this.zn=!1
    };
    var Bga=!m.E || m.Ib(9), pm=!m.E || m.Ib(9), Cga = m.E&&!m.Fb("9");
    !m.Qe || m.Fb("528");
    m.Re && m.Fb("1.9b") || m.E && m.Fb("8") || m.Se && m.Fb("9.5") || m.Qe && m.Fb("528");
    m.Re&&!m.Fb("8") || m.E && m.Fb("9");
    var zq;
    m.AE = m.E ? "focusin" : "DOMFocusIn";
    m.Dga = m.E ? "focusout" : "DOMFocusOut";
    zq = m.Qe ? "webkitTransitionEnd" : m.Se ? "otransitionend" : "transitionend";
    m.s(m.am, m.$l);
    var Ega = [1, 4, 2];
    m.g = m.am.prototype;
    m.g.init = function(a, b) {
        var c = this.type = a.type;
        this.target = a.target || a.srcElement;
        this.currentTarget = b;
        var d = a.relatedTarget;
        d ? m.Re && (m.Xl(d, "nodeName") || (d = null)) : "mouseover" == c ? d = a.fromElement : "mouseout" == c && (d = a.toElement);
        this.relatedTarget = d;
        this.offsetX = m.Qe || void 0 !== a.offsetX ? a.offsetX : a.layerX;
        this.offsetY = m.Qe || void 0 !== a.offsetY ? a.offsetY : a.layerY;
        this.clientX = void 0 !== a.clientX ? a.clientX : a.pageX;
        this.clientY = void 0 !== a.clientY ? a.clientY : a.pageY;
        this.screenX = a.screenX || 0;
        this.screenY = a.screenY ||
        0;
        this.button = a.button;
        this.keyCode = a.keyCode || 0;
        this.charCode = a.charCode || ("keypress" == c ? a.keyCode : 0);
        this.ctrlKey = a.ctrlKey;
        this.altKey = a.altKey;
        this.shiftKey = a.shiftKey;
        this.metaKey = a.metaKey;
        this.state = a.state;
        this.wa = a;
        a.defaultPrevented && this.preventDefault()
    };
    m.g.stopPropagation = function() {
        m.am.K.stopPropagation.call(this);
        this.wa.stopPropagation ? this.wa.stopPropagation() : this.wa.cancelBubble=!0
    };
    m.g.preventDefault = function() {
        m.am.K.preventDefault.call(this);
        var a = this.wa;
        if (a.preventDefault)
            a.preventDefault();
        else if (a.returnValue=!1, Cga)
            try {
                if (a.ctrlKey || 112 <= a.keyCode && 123 >= a.keyCode)
                    a.keyCode =- 1
        } catch (b) {}
    };
    m.g.Tn = function() {
        return this.wa
    };
    m.g.N = function() {};
    var cm = "closure_listenable_" + (1E6 * Math.random() | 0), Dba = 0;
    em.prototype.add = function(a, b, c, d, e) {
        var f = a.toString();
        a = this.g[f];
        a || (a = this.g[f] = [], this.j++);
        var h = hm(a, b, d, e);
        - 1 < h ? (b = a[h], c || (b.zh=!1)) : (b = new Cba(b, this.src, f, !!d, e), b.zh = c, a.push(b));
        return b
    };
    em.prototype.remove = function(a, b, c, d) {
        a = a.toString();
        if (!(a in this.g))
            return !1;
        var e = this.g[a];
        b = hm(e, b, c, d);
        return - 1 < b ? (dm(e[b]), m.Pa(e, b), 0 == e.length && (delete this.g[a], this.j--), !0) : !1
    };
    em.prototype.removeAll = function(a) {
        a = a && a.toString();
        var b = 0, c;
        for (c in this.g)
            if (!a || c == a) {
                for (var d = this.g[c], e = 0; e < d.length; e++)
                    ++b, dm(d[e]);
                    delete this.g[c];
                    this.j--
            }
        return b
    };
    var mm = "closure_lm_" + (1E6 * Math.random() | 0), um = {}, om = 0, xm = "__closure_events_fn_" + (1E9 * Math.random()>>>0);
    m.s(m.ym, m.t);
    m.ym.prototype[cm]=!0;
    m.g = m.ym.prototype;
    m.g.bi = function(a) {
        this.Ca = a
    };
    m.g.addEventListener = function(a, b, c, d) {
        m.im(this, a, b, c, d)
    };
    m.g.removeEventListener = function(a, b, c, d) {
        m.rm(this, a, b, c, d)
    };
    m.g.dispatchEvent = function(a) {
        var b, c = this.Ca;
        if (c) {
            b = [];
            for (var d = 1; c; c = c.Ca)
                b.push(c), ++d
        }
        c = this.vf;
        d = a.type || a;
        if (m.ja(a))
            a = new m.$l(a, c);
        else if (a instanceof m.$l)
            a.target = a.target || c;
        else {
            var e = a;
            a = new m.$l(d, c);
            m.zb(a, e)
        }
        var e=!0, f;
        if (b)
            for (var h = b.length - 1; !a.qe && 0 <= h; h--)
                f = a.currentTarget = b[h], e = zm(f, d, !0, a) && e;
        a.qe || (f = a.currentTarget = c, e = zm(f, d, !0, a) && e, a.qe || (e = zm(f, d, !1, a) && e));
        if (b)
            for (h = 0; !a.qe && h < b.length; h++)
                f = a.currentTarget = b[h], e = zm(f, d, !1, a) && e;
        return e
    };
    m.g.N = function() {
        m.ym.K.N.call(this);
        this.removeAllListeners();
        this.Ca = null
    };
    m.g.listen = function(a, b, c, d) {
        return this.Ic.add(String(a), b, !1, c, d)
    };
    m.g.Da = function(a, b, c, d) {
        return this.Ic.remove(String(a), b, c, d)
    };
    m.g.removeAllListeners = function(a) {
        return this.Ic ? this.Ic.removeAll(a) : 0
    };
    m.s(m.Am, m.ym);
    m.g = m.Am.prototype;
    m.g.enabled=!1;
    m.g.Ub = null;
    m.g.Kr = function() {
        if (this.enabled) {
            var a = (0, m.H)() - this.C;
            0 < a && a < .8 * this.g ? this.Ub = this.j.setTimeout(this.k, this.g - a) : (this.Ub && (this.j.clearTimeout(this.Ub), this.Ub = null), this.dispatchEvent("tick"), this.enabled && (this.Ub = this.j.setTimeout(this.k, this.g), this.C = (0, m.H)()))
        }
    };
    m.g.start = function() {
        this.enabled=!0;
        this.Ub || (this.Ub = this.j.setTimeout(this.k, this.g), this.C = (0, m.H)())
    };
    m.g.stop = function() {
        this.enabled=!1;
        this.Ub && (this.j.clearTimeout(this.Ub), this.Ub = null)
    };
    m.g.N = function() {
        m.Am.K.N.call(this);
        this.stop();
        delete this.j
    };
    m.s(Dm, m.t);
    m.g = Dm.prototype;
    m.g.wi=!1;
    m.g.Kk = 0;
    m.g.We = null;
    m.g.pn = function() {
        this.We || this.Kk ? this.wi=!0 : Em(this)
    };
    m.g.stop = function() {
        this.We && (m.da.clearTimeout(this.We), this.We = null, this.wi=!1)
    };
    m.g.pause = function() {
        this.Kk++
    };
    m.g.N = function() {
        Dm.K.N.call(this);
        this.stop()
    };
    m.g.Wt = function() {
        this.We = null;
        this.wi&&!this.Kk && (this.wi=!1, Em(this))
    };
    var BE = [], Jm = 0, Km = 0;
    var Qm, Ym, Zm, Xm, Lm, CE, DE = 0, an=!1;
    m.l("yt.dom.VisibilityMonitor.refresh", Nm, void 0);
    m.l("yt.dom.VisibilityMonitor.isVisible", function(a, b) {
        if (!an)
            throw Error("yt.dom.VisibilityMonitor is not initialized.");
        return Vm(a, b || {})
    }, void 0);
    m.l("yt.dom.VisibilityMonitor.listen", function(a, b, c, d) {
        if (!an)
            return "";
        var e = Pba(a, b, c);
        if (e)
            return e;
        $m(a);
        e=++DE + "";
        Ym[e] = {
            Fa: a,
            type: b,
            se: c,
            options: d || {
                transform: void 0,
                complete: void 0
            }
        };
        return e
    }, void 0);
    m.l("yt.dom.VisibilityMonitor.delegateByClass", function(a, b, c, d, e) {
        if (!an)
            return "";
        a = a || window.document;
        var f = Qba(a, b, c, d);
        if (f)
            return f;
        $m(a);
        f=++DE + "";
        Zm[f] = {
            Fa: a,
            type: b,
            se: c,
            filter: function(a) {
                return m.M(d, a)
            },
            className: d,
            options: e || {
                transform: void 0,
                complete: void 0
            }
        };
        return f
    }, void 0);
    m.l("yt.dom.VisibilityMonitor.unlistenByKey", function(a) {
        an && (delete Ym[a], delete Zm[a])
    }, void 0);
    m.l("yt.dom.VisibilityMonitor.States.VISIBLE", "visible", void 0);
    m.l("yt.dom.VisibilityMonitor.States.HIDDEN", "hidden", void 0);
    m.l("help.common.helpapiservice.Environment.PROD", "https://clients6.google.com", void 0);
    m.l("help.common.helpapiservice.Environment.STAGING", "https://content-googleapis-staging.sandbox.google.com", void 0);
    m.l("help.common.helpapiservice.Environment.TEST", "https://content-googleapis-test.sandbox.google.com", void 0); /*
     Portions of this code are from MochiKit, received by
     The Closure Authors under the MIT license. All other code is Copyright
     2005-2009 The Closure Authors. All Rights Reserved.
    */
    m.g = m.bn.prototype;
    m.g.cancel = function(a) {
        if (this.g)
            this.k instanceof m.bn && this.k.cancel();
        else {
            if (this.j) {
                var b = this.j;
                delete this.j;
                a ? b.cancel(a) : (b.D--, 0 >= b.D && b.cancel())
            }
            this.T ? this.T.call(this.O, this) : this.S=!0;
            this.g || this.De(new mn)
        }
    };
    m.g.Wk = function(a, b) {
        this.L=!1;
        cn(this, a, b)
    };
    m.g.Ia = function(a) {
        en(this);
        cn(this, !0, a)
    };
    m.g.De = function(a) {
        en(this);
        cn(this, !1, a)
    };
    m.g.addCallback = function(a, b) {
        return m.gn(this, a, null, b)
    };
    m.g.then = function(a, b, c) {
        var d, e, f = new m.pe(function(a, b) {
            d = a;
            e = b
        });
        m.gn(this, d, function(a) {
            a instanceof mn ? f.cancel() : e(a)
        });
        return f.then(a, b, c)
    };
    m.ne(m.bn);
    m.bn.prototype.J = function(a) {
        var b = new m.bn;
        m.gn(this, b.Ia, b.De, b);
        a && (b.j = this, this.D++);
        return b
    };
    m.s(fn, m.ua);
    fn.prototype.message = "Deferred has already fired";
    fn.prototype.name = "AlreadyCalledError";
    m.s(mn, m.ua);
    mn.prototype.message = "Deferred was canceled";
    mn.prototype.name = "CanceledError";
    ln.prototype.j = function() {
        delete kn[this.ta];
        throw this.g;
    };
    var kn = {};
    m.s(pn, m.ua);
    var Zba = {}, un = null;
    m.l("hgb.loadFlow", sn, void 0);
    m.l("hgb.resumeCookiedFlow", function(a, b, c, d, e, f, h, k, r) {
        if (!qn().pe)
            return !1;
        a = r || "";
        h = h || void 0;
        k = k || void 0;
        d = d || void 0;
        e = qn();
        e.pe ? (m.Ud.remove("GuidedHelpResume", "/", ""), b = rn(e.pe, {
            lf: a,
            document: h,
            mf: k,
            yg: b,
            locale: c,
            bz: !1,
            oe: e.oe || void 0,
            theme: d,
            source: e.source
        })) : b = Gk();
        ki(b);
        return !0
    }, void 0);
    m.l("hgb.startFlowFromUrl", function(a, b, c, d, e, f, h, k, r) {
        if (!tn().pe)
            return !1;
        a = r || "";
        k = k || void 0;
        d = d || void 0;
        e = tn();
        b = e.pe ? rn(e.pe, {
            lf: a,
            document: h,
            mf: k,
            yg: b,
            locale: c,
            source: e.source,
            oe: e.oe,
            theme: d
        }) : Gk();
        ki(b);
        return !0
    }, void 0);
    var Rn = window, Jn = window.document, dca = Rn.location, cca = /\[native code\]/, On = wn(Rn, "gapi", {});
    var zn;
    zn = wn(Rn, "___jsl", xn());
    wn(zn, "I", 0);
    wn(zn, "hel", 10);
    var Dn = wn(zn, "perf", xn());
    wn(Dn, "g", xn());
    var fca = wn(Dn, "i", xn());
    wn(Dn, "r", []);
    xn();
    xn();
    var Fn = xn(), Mn = [];
    Mn.push(["jsl", function(a) {
        for (var b in a)
            if (Object.prototype.hasOwnProperty.call(a, b)) {
                var c = a[b];
                "object" == typeof c ? zn[b] = wn(zn, b, []).concat(c) : wn(zn, b, c)
            }
        if (b = a.u)
            a = wn(zn, "us", []), a.push(b), (b = /^https:(.*)$/.exec(b)) && a.push("http:" + b[1])
    }
    ]);
    var nca = /^(\/[a-zA-Z0-9_\-]+)+$/, pca = /^[a-zA-Z0-9\-_\.,!]+$/, mca = /^gapi\.loaded_[0-9]+$/, oca = /^[a-zA-Z0-9,._-]+$/, jca = /^https?:\/\/[a-z0-9_.-]+\.google\.com(:\d+)?\/[a-zA-Z0-9_.,!=\-\/]+$/, ica = /\/cb=/g, hca = /\/\//g;
    Fn.m = function(a, b, c, d) {
        (a = a[0]) || En("missing_hint");
        return "https://apis.google.com" + kca(a, b, c, d)
    };
    var Ln = (0, window.decodeURI)("%73cript");
    On.load = function(a, b) {
        return Nn(function() {
            return Pn(a, b)
        })
    };
    $n.prototype.toString = function() {
        return this.g
    };
    m.s(ao, Yn);
    m.s(bo, Yn);
    var rE = new $n("subscription-subscribe", ao), It = new $n("subscription-subscribe-external", ao), sE = new $n("subscription-unsubscribe", bo), Mt = new $n("subscription-external-unsubscribe", bo);
    var jo = "subscription-subscribe-success subscription-unsubscribe-success subscription-subscribe-loading subscription-unsubscirbe-loading subscription-subscribe-loaded subscription-unsubscribe-loaded subscription-subscribe-failure subscription-unsubscribe-failure subscription-enable-ypc subscription-disable-ypc guide-add-subscription".split(" ");
    co.prototype.dispose = m.ea;
    co.prototype.onMessage = function(a) {
        this.j && this.j(a)
    };
    m.s(eo, Error);
    eo.prototype.name = "MechanismUnsupportedError";
    m.s(fo, co);
    fo.prototype.postMessage = function(a) {
        a = JSON.stringify(a);
        window.localStorage.setItem("sitepubsub", a);
        window.localStorage.removeItem("sitepubsub")
    };
    fo.prototype.dispose = function() {
        window.removeEventListener && window.removeEventListener("storage", this.g, !1);
        this.g = null
    };
    fo.prototype.k = function(a) {
        if ("sitepubsub" == a.key && null !== a.newValue)
            try {
                var b = JSON.parse(a.newValue);
                this.onMessage(b)
        } catch (c) {}
    };
    m.fa(io);
    m.g = io.prototype;
    m.g.subscribe = function(a, b, c) {
        return this.g ? this.g.subscribe(a, b, c) : 0
    };
    m.g.publish = function(a, b) {
        if (this.j) {
            var c = this.kx.apply(this, arguments);
            this.j.postMessage(c)
        }
    };
    m.g.dispose = function() {
        this.j && this.j.dispose();
        this.g && this.g.dispose();
        delete io.hb
    };
    m.g.iu = function(a) {
        this.g && a.version == this.Kb && this.g.publish.apply(this.g, a.arguments)
    };
    m.g.kx = function(a, b) {
        return {
            arguments: ho(Array.prototype.slice.call(arguments)),
            version: this.Kb
        }
    };
    var ko = {};
    var qo = {
        "X-YouTube-Page-CL": "PAGE_CL",
        "X-YouTube-Page-Timestamp": "PAGE_BUILD_TIMESTAMP",
        "X-YouTube-Variants-Checksum": "VARIANTS_CHECKSUM"
    };
    var Jca = {};
    var EE = [], FE = [];
    var Go = {}, Fga = "ontouchstart"in window.document;
    m.J(window.document, "blur", Jo, !0);
    m.J(window.document, "change", Jo, !0);
    m.J(window.document, "click", Jo);
    m.J(window.document, "focus", Jo, !0);
    m.J(window.document, "mouseover", Jo);
    m.J(window.document, "mouseout", Jo);
    m.J(window.document, "mousedown", Jo);
    m.J(window.document, "keydown", Jo);
    m.J(window.document, "keyup", Jo);
    m.J(window.document, "keypress", Jo);
    m.J(window.document, "cut", Jo);
    m.J(window.document, "paste", Jo);
    Fga && (m.J(window.document, "touchstart", Jo), m.J(window.document, "touchend", Jo), m.J(window.document, "touchcancel", Jo));
    m.g = m.Ko.prototype;
    m.g.la = function(a) {
        return m.L(a, m.S(this))
    };
    m.g.unregister = function() {
        m.nb(this.J);
        this.J.length = 0
    };
    m.g.init = m.ea;
    m.g.dispose = m.ea;
    m.g.ca = function(a, b, c) {
        a = m.x(a, b, c || this);
        this.J.push(a)
    };
    m.g.hc = function(a, b, c) {
        var d = this.getData(a, b);
        if (d && (d = m.n(d))) {
            var e = m.Sa(arguments, 2);
            Fj(e, 0, 0, a);
            d.apply(null, e)
        }
    };
    m.g.getData = function(a, b) {
        return m.I(a, b)
    };
    m.g.removeData = function(a, b) {
        m.el(a, b)
    };
    m.s(m.Oo, m.Ko);
    m.fa(m.Oo);
    m.g = m.Oo.prototype;
    m.g.va = "button";
    m.g.Ye = null;
    m.g.register = function() {
        m.T(this, "click", this.fn);
        m.T(this, "keydown", this.en);
        m.T(this, "keypress", this.gn);
        this.ca("page-scroll", this.Yy)
    };
    m.g.unregister = function() {
        m.Mo(this, "click", this.fn);
        m.Mo(this, "keydown", this.en);
        m.Mo(this, "keypress", this.gn);
        m.ep(this);
        this.g = {};
        m.Oo.K.unregister.call(this)
    };
    m.g.fn = function(a) {
        a&&!a.disabled && (dp(this, a), this.click(a))
    };
    m.g.en = function(a, b, c) {
        if (!(c.altKey || c.ctrlKey || c.shiftKey) && (b = Wo(this, a))) {
            var d = function(a) {
                var b = "";
                a.tagName && (b = a.tagName.toLowerCase());
                return "ul" == b || "table" == b
            }, e;
            d(b) ? e = b : e = ol(b, d);
            if (e) {
                e = e.tagName.toLowerCase();
                var f;
                "ul" == e ? f = this.Fz : "table" == e && (f = this.Ez);
                f && Nca(this, a, b, c, (0, m.p)(f, this))
            }
        }
    };
    m.g.Yy = function() {
        var a = this.g;
        if (0 != Dj(a))
            for (var b in a) {
                var c = a[b], d = m.Xo(this, c);
                if (void 0 == d || void 0 == c)
                    break;
                    Zo(this, d, c, !0)
            }
    };
    m.g.gn = function(a, b, c) {
        c.altKey || c.ctrlKey || c.shiftKey || (a = Wo(this, a), m.gj(a) && c.preventDefault())
    };
    m.g.Ez = function(a, b, c) {
        var d = m.Po(this, b);
        b = m.Oj("table", null, b);
        var e = m.Oj("tr", null, b), e = m.Mb("td", null, e).length;
        b = m.Mb("td", null, b);
        d = Uo(d, b, e, c);
        - 1 != d && (m.Ro(this, a, b[d]), c.preventDefault())
    };
    m.g.Fz = function(a, b, c) {
        if (40 == c.keyCode || 38 == c.keyCode) {
            var d = m.Po(this, b);
            b = m.qb(m.Mb("li", null, b), m.gj);
            d = Uo(d, b, 1, c);
            m.Ro(this, a, b[d]);
            c.preventDefault()
        }
    };
    m.g.Rp = function(a) {
        if (a) {
            var b = Wo(this, a);
            if (b) {
                a.setAttribute("aria-pressed", "true");
                a.setAttribute("aria-expanded", "true");
                b.originalParentNode = b.parentNode;
                b.activeButtonNode = a;
                b.parentNode.removeChild(b);
                var c;
                this.getData(a, "button-has-sibling-menu") ? c = a.parentNode : c = $o(this, a);
                c.appendChild(b);
                b.style.minWidth = a.offsetWidth - 2 + "px";
                var d = Yo(this, a);
                d && c.appendChild(d);
                (c=!!this.getData(a, "button-menu-fixed")) && (this.g[m.Zk(a).toString()] = b);
                Zo(this, a, b, c);
                m.ob("yt-uix-button-menu-before-show",
                a, b);
                m.K(b);
                this.hc(a, "button-menu-action", !0);
                m.C(a, m.S(this, "active"));
                b = (0, m.p)(this.vm, this, a, !1);
                c = (0, m.p)(this.vm, this, a, !0);
                d = (0, m.p)(this.du, this, a, void 0);
                this.Ye && Wo(this, this.Ye) == Wo(this, a) || m.ep(this);
                m.z("yt-uix-button-menu-show", a);
                m.N(this.j);
                this.j = [m.J(window.document, "click", c), m.J(window.document, "contextmenu", b), m.J(window, "resize", d)];
                this.Ye = a
            }
        }
    };
    m.g.du = function(a, b) {
        var c = Wo(this, a);
        if (c) {
            b && (c.innerHTML = b);
            var d=!!this.getData(a, "button-menu-fixed");
            Zo(this, a, c, d)
        }
    };
    m.g.vm = function(a, b, c) {
        c = m.Nk(c);
        var d = m.L(c, m.S(this));
        if (d) {
            var d = Wo(this, d), e = Wo(this, a);
            if (d == e)
                return 
        }
        var d = m.L(c, m.S(this, "menu")), e = d == Wo(this, a), f = m.A(c, m.S(this, "menu-item")), h = m.A(c, m.S(this, "menu-close"));
        if (!d || e && (f || h))
            m.Qo(this, a), d && b && this.getData(a, "button-menu-indicate-selected") && ((a = m.G(m.S(this, "content"), a)) && m.pl(a, m.ml(c)), Oca(this, d, c))
    };
    m.g.isToggled = function(a) {
        return m.A(a, m.S(this, "toggled"))
    };
    m.g.click = function(a) {
        Wo(this, a) && (m.Vo(this, a), a.focus());
        this.hc(a, "button-action")
    };
    m.s(fp, m.Ko);
    m.fa(fp);
    fp.prototype.va = "char-counter";
    fp.prototype.register = function() {
        m.T(this, "keydown", this.g, "input");
        m.T(this, "paste", this.g, "input");
        m.T(this, "cut", this.g, "input")
    };
    fp.prototype.unregister = function() {
        m.Mo(this, "keydown", this.g, "input");
        m.Mo(this, "paste", this.g, "input");
        m.Mo(this, "cut", this.g, "input")
    };
    fp.prototype.g = function(a) {
        var b = this.la(a);
        if (b) {
            var c = "true" == this.getData(b, "count-char-by-size"), d = (0, window.parseInt)(this.getData(b, "char-limit"), 10);
            (0, window.isNaN)(d) || 0 >= d || m.v((0, m.p)(function() {
                var e = "true" == this.getData(b, "use-plaintext-length"), f = (0, window.parseInt)(a.getAttribute("maxlength"), 10);
                if (!(0, window.isNaN)(f)) {
                    var h = gp(a, c, e);
                    if (c) {
                        if (h > f) {
                            var k = a.value, r = k.length, w = 0, f = h - f, h = "", B = 0;
                            do 
                                h += k[r - w], B = (0, window.unescape)((0, window.encodeURIComponent)(h)).length, w++;
                            while (B < f);
                            a.value = a.value.substring(0, r - w)
                        }
                    } else 
                        h > f && (a.value = a.value.substring(0, f))
                }
                k = (0, window.parseInt)(this.getData(b, "warn-at-chars-remaining"), 10);
                (0, window.isNaN)(k) && (k = 0);
                e = d - gp(a, c, e);
                m.rb(b, m.S(this, "maxed-out"), e < k);
                "true" == this.getData(b, "maxed-out-as-positive") && (e = Math.abs(e));
                m.G(m.S(this, "remaining"), b).innerHTML = e
            }, this), 0)
        }
    };
    m.s(hp, m.Ko);
    m.g = hp.prototype;
    m.g.la = function(a) {
        var b = m.Ko.prototype.la.call(this, a);
        return b ? b : a
    };
    m.g.register = function() {
        this.ca("yt-uix-kbd-nav-move-out-done", this.hide)
    };
    m.g.getData = function(a, b) {
        var c = hp.K.getData.call(this, a, b);
        return c ? c : (c = hp.K.getData.call(this, a, "card-config")) && (c = m.n(c)) && c[b] ? c[b] : null
    };
    m.g.show = function(a) {
        var b = this.la(a);
        if (b) {
            m.C(b, m.S(this, "active"));
            var c = ip(this, a, b);
            if (c) {
                c.cardTargetNode = a;
                c.cardRootNode = b;
                kp(this, a, c);
                var d = m.S(this, "card-visible"), e = this.getData(a, "card-delegate-show") && this.getData(b, "card-action");
                this.hc(b, "card-action", a);
                this.k = a;
                m.P(c);
                m.v((0, m.p)(function() {
                    e || (m.K(c), m.z("yt-uix-card-show", b, a, c));
                    mp(c);
                    m.C(c, d);
                    m.z("yt-uix-kbd-nav-move-in-to", c)
                }, this), 10)
            }
        }
    };
    m.g.hide = function(a) {
        if (a = this.la(a)) {
            var b = m.F(m.S(this, "card") + m.Zk(a));
            b && (m.D(a, m.S(this, "active")), m.D(b, m.S(this, "card-visible")), m.P(b), this.k = null, b.cardTargetNode = null, b.cardRootNode = null, b.cardMask && (m.ri(b.cardMask), b.cardMask = null))
        }
    };
    m.g.Ft = function(a, b) {
        var c = this.la(a);
        if (c) {
            if (b) {
                var d = m.jp(this, c);
                if (!d)
                    return;
                d.innerHTML = b
            }
            m.A(c, m.S(this, "active")) && (c = ip(this, a, c), kp(this, a, c), m.K(c), mp(c))
        }
    };
    hp.prototype.isActive = function(a) {
        return (a = this.la(a)) ? m.A(a, m.S(this, "active")) : !1
    };
    m.s(m.np, hp);
    m.fa(m.np);
    m.g = m.np.prototype;
    m.g.va = "clickcard";
    m.g.register = function() {
        m.np.K.register.call(this);
        m.T(this, "click", this.mm, "target");
        m.T(this, "click", this.lm, "close")
    };
    m.g.unregister = function() {
        m.np.K.unregister.call(this);
        m.Mo(this, "click", this.mm, "target");
        m.Mo(this, "click", this.lm, "close");
        for (var a in this.g)
            m.N(this.g[a]);
        this.g = {};
        for (a in this.j)
            m.N(this.j[a]);
        this.j = {}
    };
    m.g.mm = function(a, b, c) {
        b = m.qi(c.target, "button");
        b && b.disabled || (a = (b = this.getData(a, "card-target")) ? m.F(b) : a, b = this.la(a), this.getData(b, "disabled") || (m.A(b, m.S(this, "active")) ? (this.hide(a), m.D(b, m.S(this, "active"))) : (this.show(a), m.C(b, m.S(this, "active")))))
    };
    m.g.show = function(a) {
        m.np.K.show.call(this, a);
        var b = this.la(a);
        if (!m.I(b, "click-outside-persists")) {
            var c = m.oa(a);
            if (this.g[c])
                return;
            var b = m.J(window.document, "click", (0, m.p)(this.fm, this, a)), d = m.J(window, "blur", (0, m.p)(this.fm, this, a));
            this.g[c] = [b, d]
        }
        a = m.J(window, "resize", (0, m.p)(this.Ft, this, a, void 0));
        this.j[c] = a
    };
    m.g.hide = function(a) {
        m.np.K.hide.call(this, a);
        a = m.oa(a);
        var b = this.g[a];
        b && (m.N(b), this.g[a] = null);
        if (b = this.j[a])
            m.N(b), this.j[a] = null
    };
    m.g.fm = function(a, b) {
        m.L(b.target, Lo(this) + "-card") || this.hide(a)
    };
    m.g.lm = function(a) {
        (a = m.L(a, m.S(this, "card"))) && this.hide(a.cardTargetNode)
    };
    m.s(op, m.Ko);
    m.fa(op);
    op.prototype.va = "close";
    op.prototype.register = function() {
        m.T(this, "click", this.g)
    };
    op.prototype.unregister = function() {
        m.Mo(this, "click", this.g)
    };
    op.prototype.g = function(a) {
        var b, c = this.getData(a, "close-parent-class"), d = this.getData(a, "close-parent-id");
        d ? b = m.F(d) : c && (b = m.L(a, c));
        b && (m.P(b), c = this.getData(a, "close-focus-target-id")) && (c = m.F(c)) && (d = m.Oo.getInstance(), d.isToggled(c) && dp(d, c), c.focus());
        this.hc(a, "close-callback", b)
    };
    m.s(pp, m.Ko);
    m.fa(pp);
    m.g = pp.prototype;
    m.g.va = "expander";
    m.g.register = function() {
        m.T(this, "click", this.Dp, "head");
        m.T(this, "keypress", this.Ep, "head")
    };
    m.g.unregister = function() {
        m.Mo(this, "click", this.Dp, "head");
        m.Mo(this, "keypress", this.Ep, "head")
    };
    m.g.Dp = function(a) {
        qp(this, a)
    };
    m.g.Ep = function(a, b, c) {
        c && 13 == c.keyCode && qp(this, a)
    };
    m.s(m.tp, m.Ko);
    m.fa(m.tp);
    m.g = m.tp.prototype;
    m.g.va = "form-input";
    m.g.register = function() {
        m.E&&!m.Fb(9) && (m.T(this, "click", this.tc, "checkbox"), m.T(this, "keypressed", this.tc, "checkbox"), m.T(this, "click", this.gi, "radio"), m.T(this, "keypressed", this.gi, "radio"));
        m.E&&!m.Fb(10) && m.T(this, "click", this.Bo, "placeholder");
        m.T(this, "change", this.tc, "checkbox");
        m.T(this, "blur", this.zo, "select-element");
        m.T(this, "change", this.Jb, "select-element");
        m.T(this, "keyup", this.Jb, "select-element");
        m.T(this, "focus", this.Ao, "select-element");
        m.T(this, "keyup", this.Xc, "text");
        m.T(this,
        "keyup", this.Xc, "textarea");
        m.T(this, "keyup", this.Xc, "bidi");
        m.T(this, "click", this.az, "reset")
    };
    m.g.unregister = function() {
        m.E&&!m.Fb(9) && (m.Mo(this, "click", this.tc, "checkbox"), m.Mo(this, "keypressed", this.tc, "checkbox"), m.Mo(this, "click", this.gi, "radio"), m.Mo(this, "keypressed", this.gi, "radio"));
        m.E&&!m.Fb(10) && m.Mo(this, "click", this.Bo, "placeholder");
        m.Mo(this, "change", this.tc, "checkbox");
        m.Mo(this, "blur", this.zo, "select-element");
        m.Mo(this, "change", this.Jb, "select-element");
        m.Mo(this, "keyup", this.Jb, "select-element");
        m.Mo(this, "focus", this.Ao, "select-element");
        m.Mo(this, "keyup", this.Xc, "text");
        m.Mo(this, "keyup", this.Xc, "textarea");
        m.Mo(this, "keyup", this.Xc, "bidi");
        m.tp.K.unregister.call(this)
    };
    m.g.tc = function(a) {
        var b = m.L(a, m.S(this, "checkbox-container"));
        a.checked && m.A(b, "partial") && (a.checked=!1, a.j=!1, m.D(b, "partial"));
        m.rb(b, "checked", a.checked)
    };
    m.g.ep = function(a) {
        var b = m.L(a, m.S(this, "radio-container"));
        b && m.rb(b, "checked", a.checked)
    };
    m.g.gi = function() {
        yp()
    };
    m.g.Xc = function(a) {
        var b = a.value;
        m.Iaa.test(b) ? a.dir = "rtl" : m.Haa.test(b) ? a.dir = "ltr" : a.removeAttribute("dir");
        m.E&&!m.Fb(10) && (b = m.L(a, m.S(this, "container"))) && m.rb(b, m.S(this, "non-empty"), !!a.value)
    };
    m.g.Bo = function(a) {
        a = m.L(a, m.S(this, "container"));
        (a = m.No(this, "text", a) || m.No(this, "textarea", a)) && a.focus()
    };
    m.g.Ao = function(a) {
        var b = m.L(a, m.S(this, "select"));
        m.C(b, "focused");
        this.Jb(a)
    };
    m.g.zo = function(a) {
        var b = m.L(a, m.S(this, "select"));
        m.D(b, "focused");
        this.Jb(a)
    };
    m.g.Jb = function(a) {
        var b = m.L(a, m.S(this, "select")), c = m.G(m.S(this, "select-value"), b), d = up(a);
        d && ("" != c.innerHTML && d.innerHTML != c.innerHTML && this.hc(a, "onchange-callback"), c.innerHTML = d.innerHTML);
        m.rb(b, m.S(this, "select-disabled"), a.disabled)
    };
    m.g.az = function() {
        m.zp()
    };
    m.s(m.Cp, hp);
    m.fa(m.Cp);
    m.g = m.Cp.prototype;
    m.g.va = "hovercard";
    m.g.register = function() {
        m.T(this, "mouseenter", this.mp, "target");
        m.T(this, "mouseleave", this.pp, "target");
        m.T(this, "mouseenter", this.np, "card");
        m.T(this, "mouseleave", this.qp, "card")
    };
    m.g.unregister = function() {
        m.Mo(this, "mouseenter", this.mp, "target");
        m.Mo(this, "mouseleave", this.pp, "target");
        m.Mo(this, "mouseenter", this.np, "card");
        m.Mo(this, "mouseleave", this.qp, "card")
    };
    m.g.mp = function(a) {
        if (GE != a) {
            GE && (this.hide(GE), GE = null);
            var b = (0, m.p)(this.show, this, a), c = (0, window.parseInt)(this.getData(a, "delay-show"), 10), b = m.v(b, - 1 < c ? c : 200);
            m.fl(a, "card-timer", b.toString());
            GE = a;
            a.alt && (m.fl(a, "card-alt", a.alt), a.alt = "");
            a.title && (m.fl(a, "card-title", a.title), a.title = "")
        }
    };
    m.g.pp = function(a) {
        var b = (0, window.parseInt)(this.getData(a, "card-timer"), 10);
        m.bb(b);
        this.la(a).isCardHidable=!0;
        b = (0, window.parseInt)(this.getData(a, "delay-hide"), 10);
        b =- 1 < b ? b : 200;
        m.v((0, m.p)(this.cz, this, a), b);
        if (b = this.getData(a, "card-alt"))
            a.alt = b;
        (b = this.getData(a, "card-title")) && (a.title = b)
    };
    m.g.cz = function(a) {
        this.la(a).isCardHidable && (this.hide(a), GE = null)
    };
    m.g.np = function(a) {
        a && (a.cardRootNode.isCardHidable=!1)
    };
    m.g.qp = function(a) {
        a && this.hide(a.cardTargetNode)
    };
    var GE = null;
    var Hp;
    m.s(Ep, m.Ko);
    m.fa(Ep);
    m.g = Ep.prototype;
    m.g.va = "kbd-nav";
    m.g.register = function() {
        m.T(this, "keydown", this.Wo);
        this.ca("yt-uix-kbd-nav-move-in", this.Uo);
        this.ca("yt-uix-kbd-nav-move-in-to", this.Uy);
        this.ca("yt-uix-kbd-move-next", this.Vo);
        this.ca("yt-uix-kbd-nav-move-to", this.oi)
    };
    m.g.unregister = function() {
        m.Mo(this, "keydown", this.Wo);
        m.N(Hp)
    };
    m.g.Wo = function(a, b, c) {
        var d = c.keyCode;
        if (a = m.L(a, m.S(this)))
            switch (d) {
            case 13:
            case 32:
                this.Uo(a);
                break;
            case 27:
                c.preventDefault();
                c.stopImmediatePropagation();
                t:
                {
                    for (c = m.Dp(a, "kbdNavMoveOut"); !c;) {
                        c = m.L(a.parentElement, m.S(this));
                        if (!c)
                            break t;
                            c = m.Dp(c, "kbdNavMoveOut")
                    }
                    c = m.F(c);
                    this.oi(c);
                    m.z("yt-uix-kbd-nav-move-out-done", c)
                }
                break;
            case 40:
            case 38:
                if ((b = c.target) && m.A(a, m.S(this, "list")))
                    switch (d) {
                    case 40:
                        this.Vo(b, a);
                        break;
                    case 38:
                        d = window.document.activeElement == a, a = Ip(a), b = a.indexOf(b), 0 > b&&!d ||
                        (b = d ? a.length - 1 : (a.length + b - 1)%a.length, a[b].focus(), Gp(this, a[b]))
                    }
                    c.preventDefault()
                }
    };
    m.g.Uo = function(a) {
        var b = m.Dp(a, "kbdNavMoveIn"), b = m.F(b);
        Fp(this, a, b);
        this.oi(b)
    };
    m.g.Uy = function(a) {
        var b;
        t: {
            var c = window.document;
            try {
                b = c && c.activeElement;
                break t
            } catch (d) {}
            b = null
        }
        Fp(this, b, a);
        this.oi(a)
    };
    m.g.oi = function(a) {
        if (a) {
            var b = ol(a, function(a) {
                return m.ik(a) ? nl(a) : !1
            });
            b ? b.focus() : (0 > a.tabIndex && (a.tabIndex = 0), a.focus())
        }
    };
    m.g.Vo = function(a, b) {
        var c = window.document.activeElement == b, d = Ip(b), e = d.indexOf(a);
        0 > e&&!c || (c = c ? 0 : (e + 1)%d.length, d[c].focus(), Gp(this, d[c]))
    };
    m.s(Jp, m.Ko);
    m.fa(Jp);
    m.g = Jp.prototype;
    m.g.va = "livereminder";
    m.g.register = function() {
        m.T(this, "click", this.xo, "main-button");
        m.T(this, "click", this.wo, "menu-item-gcal");
        this.ca("live-event-reminder-set", this.Vy);
        this.ca("live-event-reminder-removed", this.Xy);
        this.ca("live-event-reminder-failed", this.Wy)
    };
    m.g.unregister = function() {
        m.Mo(this, "click", this.xo, "main-button");
        m.Mo(this, "click", this.wo, "menu-item-gcal");
        Jp.K.unregister.call(this)
    };
    m.g.xo = function(a) {
        var b = m.I(a, "uix-livereminder-video-id");
        if (b) {
            var c = m.I(a, "href");
            Kp(this, b, "disable");
            m.A(a, "yt-uix-button-livereminder-set") ? m.z("live-event-reminder-remove-request", b) : m.z("live-event-reminder-set-request", b, c)
        }
    };
    m.g.Vy = function(a) {
        Kp(this, a, "set")
    };
    m.g.Xy = function(a) {
        Kp(this, a, "removed")
    };
    m.g.Wy = function(a) {
        Kp(this, a, "cancel")
    };
    m.g.wo = function(a) {
        a = m.I(a, "gcal-url");
        Jk(a, {
            width: 855,
            height: 750
        })
    };
    var Sp = 0, Mp = [], Pp = {}, Tp = [];
    m.s(m.Up, m.Ko);
    m.fa(m.Up);
    m.g = m.Up.prototype;
    m.g.va = "load-more";
    m.g.register = function() {
        m.T(this, "click", this.loadMore)
    };
    m.g.unregister = function() {
        m.Mo(this, "click", this.loadMore);
        m.Up.K.unregister.call(this);
        Qca()
    };
    m.g.init = function() {
        var a = {};
        a["load-more-auto"] = (0, m.p)(this.loadMore, this);
        m.Op(a)
    };
    m.g.addClass = function(a, b) {
        m.C(a, m.S(this, b))
    };
    m.g.removeClass = function(a, b) {
        m.D(a, m.S(this, b))
    };
    m.g.loadMore = function(a) {
        if (!a.disabled) {
            var b = this.getData(a, "uix-load-more-href") || "";
            Sca(this, a, b)
        }
    };
    m.g.yy = function(a, b, c) {
        var d = c.content_html;
        b = m.F(this.getData(a, "uix-load-more-target-id"));
        d && b ? (this.getData(a, "uix-load-more-replace-content") && m.Ub(b), d = m.Vk(d), this.getData(a, "uix-load-more-insert-before-content") ? m.vl(b, d, 0) : m.zi(b, d), m.z("yt-dom-content-change", a.parentElement), c = m.Vk(c.load_more_widget_html || ""), m.ul(c, a), m.z("yt-uix-load-more-success", b)) : this.xk(a)
    };
    m.g.xk = function(a) {
        this.addClass(a, "error")
    };
    m.s(m.Vp, m.Ko);
    m.fa(m.Vp);
    m.g = m.Vp.prototype;
    m.g.va = "menu";
    m.g.register = function() {
        m.T(this, "click", this.kl);
        m.T(this, "mouseenter", this.xx);
        this.ca("page-scroll", this.yx);
        this.ca("yt-uix-kbd-nav-move-out-done", function(a) {
            a = this.la(a);
            cq(this, a)
        });
        this.C = new m.Wa
    };
    m.g.unregister = function() {
        m.Mo(this, "click", this.kl);
        this.j = this.g = null;
        m.N(Ll(m.Ki(this.k)));
        this.k = {};
        m.tb(this.D, function(a) {
            m.ri(a)
        }, this);
        this.D = {};
        m.Va(this.C);
        this.C = null;
        m.Vp.K.unregister.call(this)
    };
    m.g.kl = function(a, b, c) {
        a && (b = eq(this, a), Xk(c.target, b) && aq(this, a))
    };
    m.g.xx = function(a, b, c) {
        a && m.A(a, m.S(this, "hover")) && (b = eq(this, a), Xk(c.target, b) && aq(this, a, !0))
    };
    m.g.yx = function() {
        this.g && this.j && Zp(this, this.j, this.g)
    };
    m.g.Mj = function(a) {
        if (a) {
            var b = Yp(this, a);
            if (b) {
                m.ob("yt-uix-menu-before-show", a, b);
                if (this.g)
                    Xk(a, this.g) || cq(this, this.j);
                else {
                    this.j = a;
                    this.g = b;
                    m.A(a, m.S(this, "sibling-content")) || (m.ri(b), window.document.body.appendChild(b));
                    var c = eq(this, a).offsetWidth - 2;
                    b.style.minWidth = c + "px"
                }(c = $p(this, a)) && m.wl(c, b);
                m.D(b, m.S(this, "content-hidden"));
                Zp(this, a, b);
                m.C(eq(this, a), m.S(this, "trigger-selected"));
                m.z("yt-uix-menu-show", a);
                fq(0, b);
                Uca(this, a);
                m.z("yt-uix-kbd-nav-move-in-to", b);
                var d = (0, m.p)(this.ns,
                this, a), e = (0, m.p)(this.ls, this, a), c = m.oa(a).toString();
                this.k[c] = [m.J(b, "click", e), m.J(window.document, "click", d)];
                m.A(a, m.S(this, "hover")) && (a = (0, m.p)(this.ms, this, a), this.k[c].push(m.J(window.document, "mousemove", a)))
            }
        }
    };
    m.g.ms = function(a, b) {
        var c = m.Nk(b);
        if (c) {
            var d = eq(this, a);
            Xk(c, d) || gq(this, c) || m.dq(this, a)
        }
    };
    m.g.ns = function(a, b) {
        var c = m.Nk(b);
        if (c) {
            if (gq(this, c)) {
                var d = m.L(c, m.S(this, "content")), e = m.qi(c, "LI");
                e && d && m.vi(d, e) && m.ob("yt-uix-menu-item-clicked", c);
                if (!m.L(c, m.S(this, "close-on-select")))
                    return 
            }
            cq(this, a)
        }
    };
    m.g.ls = function(a, b) {
        var c = m.Nk(b);
        c && Vca(this, a, c)
    };
    var Gga = {
        LOADING: "loading",
        MB: "content",
        NF: "working"
    };
    m.g = m.hq.prototype;
    m.g.setState = function(a) {
        var b = m.G("yt-dialog-fg-content", this.ob), c = [];
        m.tb(Gga, function(a) {
            c.push("yt-dialog-show-" + a)
        });
        m.Gl(b, c);
        m.C(b, "yt-dialog-show-" + a)
    };
    m.g.show = function(a) {
        if (!this.ha()) {
            window.document.activeElement && window.document.activeElement != window.document.body && window.document.activeElement.blur && window.document.activeElement.blur();
            if (!this.J) {
                this.j || (this.j = m.F("yt-dialog-bg"), this.j || (this.j = m.xl("div"), this.j.id = "yt-dialog-bg", this.j.className = "yt-dialog-bg", window.document.body.appendChild(this.j)));
                var b;
                t:
                {
                    var c = window, d = c.document;
                    b = 0;
                    if (d) {
                        b = d.body;
                        var e = d.documentElement;
                        if (!e ||!b) {
                            b = 0;
                            break t
                        }
                        c = ti(c).height;
                        if (oi(d) && e.scrollHeight)
                            b =
                            e.scrollHeight != c ? e.scrollHeight : e.offsetHeight;
                        else {
                            var d = e.scrollHeight, f = e.offsetHeight;
                            e.clientHeight != f && (d = b.scrollHeight, f = b.offsetHeight);
                            b = d > c ? d > f ? d : f : d < f ? d : f
                        }
                    }
                }
                this.j.style.height = b + "px";
                m.K(this.j)
            }
            Sk(this.ob);
            b = Xca(this);
            Yca(b);
            this.L || (this.C = m.J(window.document, "keydown", (0, m.p)(this.Ir, this)));
            b = this.ob;
            e = m.x("player-added", this.Hr, this);
            m.fl(b, "player-ready-pubsub-key", e);
            this.S && (this.D = m.J(window.document, "click", (0, m.p)(this.Jr, this)));
            m.K(this.ob);
            m.C(window.document.body, "yt-dialog-active");
            m.ep(m.Oo.getInstance());
            m.lp(m.np.getInstance());
            m.lp(m.Cp.getInstance());
            this.g = a
        }
    };
    m.g.Hr = function() {
        Sk(this.ob)
    };
    m.g.er = function(a) {
        a = a.currentTarget;
        a.disabled || (a = m.I(a, "action") || "", m.jq(this, a))
    };
    m.g.setTitle = function(a) {
        m.pl(m.G("yt-dialog-title", this.ob), a)
    };
    m.g.getData = function() {
        return this.g
    };
    m.g.Ir = function(a) {
        m.v((0, m.p)(function() {
            27 == a.keyCode && m.jq(this, "cancel")
        }, this), 0)
    };
    m.g.Jr = function(a) {
        "yt-dialog-base" == a.target.className && m.jq(this, "cancel")
    };
    m.g.ha = function() {
        return this.F
    };
    m.g.dispose = function() {
        m.gj(this.ob) && m.jq(this, "dispose");
        m.N(this.H);
        this.k.dispose();
        this.k = null;
        this.F=!0
    };
    m.l("yt.ui.Dialog", m.hq, void 0);
    m.s(m.kq, m.Ko);
    m.fa(m.kq);
    m.g = m.kq.prototype;
    m.g.lc = null;
    m.g.Xe = null;
    m.g.va = "overlay";
    m.g.register = function() {
        m.T(this, "click", this.uj, "target");
        m.T(this, "click", this.hide, "close");
        lq(this)
    };
    m.g.unregister = function() {
        m.kq.K.unregister.call(this);
        m.Mo(this, "click", this.uj, "target");
        m.Mo(this, "click", this.hide, "close");
        this.j && (m.nb(this.j), this.j = null);
        this.g && (m.N(this.g), this.g = null)
    };
    m.g.uj = function(a) {
        if (!this.lc ||!m.gj(this.lc.ob)) {
            var b = this.la(a);
            a = ada(b, a);
            b || (b = a ? a.overlayParentNode : null);
            if (b && a) {
                var c=!!this.getData(b, "disable-shortcuts") ||!1;
                this.lc = new m.hq(a, c);
                this.Xe = b;
                var d = m.G("yt-dialog-fg", a);
                if (d) {
                    var e = this.getData(b, "overlay-class") || "", f = this.getData(b, "overlay-style") || "default", h = this.getData(b, "overlay-shape") || "default", e = e ? e.split(" "): [];
                    e.push(m.S(this, f));
                    e.push(m.S(this, h));
                    m.Hl(d, e)
                }
                this.lc.show();
                m.z("yt-uix-kbd-nav-move-to", a);
                lq(this);
                c || (c = (0, m.p)(function(a) {
                    m.A(a.target,
                    "yt-dialog-base") && mq(this)
                }, this), a = m.G("yt-dialog-base", a), this.g = m.J(a, "click", c));
                this.hc(b, "overlay-shown")
            }
        }
    };
    m.g.hide = function() {
        m.z("yt-uix-overlay-hide")
    };
    m.g.show = function(a) {
        this.uj(a)
    };
    m.s(oq, m.Ko);
    m.fa(oq);
    oq.prototype.va = "playlistlike";
    oq.prototype.register = function() {
        m.T(this, "click", this.g)
    };
    oq.prototype.unregister = function() {
        m.Mo(this, "click", this.g)
    };
    oq.prototype.g = function(a) {
        var b = this.getData(a, "toggle-class"), b = (0, m.Fa)(b), c = this.getData(a, "playlist-id"), d = this.getData(a, "token");
        if (a && b && c && d) {
            var e = m.A(a, b) ? "remove_like": "like";
            bda(this, c, e, a, b, d)
        }
    };
    m.s(pq, m.Ko);
    m.fa(pq);
    pq.prototype.va = "post-anchor";
    pq.prototype.register = function() {
        m.T(this, "click", this.g)
    };
    pq.prototype.unregister = function() {
        m.Mo(this, "click", this.g)
    };
    pq.prototype.g = function(a, b, c) {
        b = this.getData(a, "post-data") || "";
        b = m.Yi(b);
        var d = window.document.createElement("form");
        d.setAttribute("method", "POST");
        d.setAttribute("action", a.href);
        a = window.document.createElement("input");
        a.setAttribute("type", "hidden");
        a.setAttribute("name", "session_token");
        a.setAttribute("value", m.u("XSRF_TOKEN"));
        d.appendChild(a);
        for (var e in b)
            a = window.document.createElement("input"), a.setAttribute("type", "hidden"), a.setAttribute("name", e), a.setAttribute("value", b[e]), d.appendChild(a);
        window.document.body.appendChild(d);
        c.preventDefault();
        d.submit()
    };
    m.s(qq, m.Ko);
    m.fa(qq);
    qq.prototype.va = "redirect-link";
    qq.prototype.register = function() {
        m.T(this, "click", this.g)
    };
    qq.prototype.unregister = function() {
        m.Mo(this, "click", this.g)
    };
    qq.prototype.g = function(a) {
        if (!m.I(a, "redirect-href-updated")) {
            m.fl(a, "redirect-href-updated", "true");
            var b = m.u("XSRF_REDIRECT_TOKEN");
            if (b) {
                var c = {};
                c.q = a.href;
                c.redir_token = b;
                a.href = m.rc("/redirect", c)
            }
        }
    };
    m.s(m.rq, m.Ko);
    m.fa(m.rq);
    m.g = m.rq.prototype;
    m.g.va = "scroller";
    m.g.register = function() {
        m.T(this, "mouseenter", this.qn);
        m.T(this, "mouseleave", this.Ph)
    };
    m.g.unregister = function() {
        m.Mo(this, "mouseenter", this.qn);
        m.Mo(this, "mouseleave", this.Ph);
        for (var a in this.g)
            this.Ph(this.g[a]);
        this.g = {};
        m.rq.K.unregister.call(this)
    };
    m.g.dispose = function() {
        for (var a in this.g)
            this.Ph(this.g[a]);
        this.g = {}
    };
    m.g.qn = function(a) {
        var b = m.J(a, "mousewheel", (0, m.p)(this.iw, this, a));
        m.fl(a, "scroller-mousewheel-listener", b);
        b = m.J(a, "scroll", (0, m.p)(this.jw, this, a));
        m.fl(a, "scroller-scroll-listener", b);
        a && (b = m.oa(a).toString(), this.g[b] = a)
    };
    m.g.Ph = function(a) {
        var b = this.getData(a, "scroller-mousewheel-listener") || "";
        m.fl(a, "scroller-mousewheel-listener", "");
        var c = this.getData(a, "scroller-scroll-listener") || "";
        m.fl(a, "scroller-scroll-listener", "");
        m.N(b);
        m.N(c);
        m.fl(a, "scroller-scroll-listener", "");
        a && (a = m.oa(a).toString(), delete this.g[a])
    };
    m.g.iw = function(a, b) {
        b.preventDefault();
        b.wheelDeltaY && (a.scrollTop = a.scrollTop + b.wheelDeltaY);
        m.z("yt-dom-content-change", a)
    };
    m.g.jw = function(a) {
        this.hc(a, "scroll-action")
    };
    m.s(tq, m.Ko);
    m.fa(tq);
    tq.prototype.va = "sessionlink";
    tq.prototype.register = function() {
        m.T(this, "mousedown", this.g);
        m.T(this, "click", this.g)
    };
    tq.prototype.unregister = function() {
        m.Mo(this, "mousedown", this.g);
        m.Mo(this, "click", this.g)
    };
    tq.prototype.g = function(a) {
        uq(a)
    };
    m.s(m.vq, m.Ko);
    m.fa(m.vq);
    m.g = m.vq.prototype;
    m.g.va = "slider";
    m.g.register = function() {
        m.T(this, "click", this.mn, "prev");
        m.T(this, "click", this.ln, "next");
        m.T(this, "keyup", this.nn, "item");
        this.g = m.J(window, "resize", (0, m.p)(this.ew, this))
    };
    m.g.unregister = function() {
        m.Mo(this, "click", this.mn, "prev");
        m.Mo(this, "click", this.ln, "next");
        m.Mo(this, "click", this.nn, "item");
        m.N(this.g);
        m.vq.K.unregister.call(this)
    };
    m.g.sc = function() {
        var a = m.M(m.S(this));
        (0, m.y)(a, function(a) {
            Cq(this, a)
        }, this)
    };
    m.g.ew = function() {
        m.bb(this.j);
        this.j = m.v((0, m.p)(this.sc, this), 200)
    };
    m.g.ln = function(a) {
        wq(this, a, "forward")
    };
    m.g.mn = function(a) {
        wq(this, a, "back")
    };
    m.g.nn = function(a) {
        (a = this.la(a)) && m.z("yt-dom-content-change", a)
    };
    m.g.Ek = function(a, b) {
        return Bk(b)
    };
    m.g.Fk = function(a, b) {
        return Bk(b) + b.offsetWidth
    };
    m.g.bp = function(a, b) {
        a && (m.D(a, m.S(this, "is-moving")), Cq(this, a), m.z("yt-uix-slider-slide-shown", b), m.z("yt-dom-content-change", a))
    };
    var Hga = m.n("yt.pubsub2.instance_") || new m.Wa;
    m.Wa.prototype.subscribe = m.Wa.prototype.subscribe;
    m.Wa.prototype.unsubscribeByKey = m.Wa.prototype.Ib;
    m.Wa.prototype.publish = m.Wa.prototype.publish;
    m.Wa.prototype.clear = m.Wa.prototype.clear;
    m.l("yt.pubsub2.instance_", Hga, void 0);
    var Gq = m.n("yt.pubsub2.subscribedKeys_") || {};
    m.l("yt.pubsub2.subscribedKeys_", Gq, void 0);
    var Iq = m.n("yt.pubsub2.topicToKeys_") || {};
    m.l("yt.pubsub2.topicToKeys_", Iq, void 0);
    var Hq = m.n("yt.pubsub2.isAsync_") || {};
    m.l("yt.pubsub2.isAsync_", Hq, void 0);
    m.l("yt.pubsub2.skipSubKey_", null, void 0);
    m.l("yt.pubsub.publish", m.z, void 0);
    m.s(m.Mq, m.Ko);
    m.fa(m.Mq);
    m.g = m.Mq.prototype;
    m.g.va = "tooltip";
    m.g.Th = 0;
    m.g.register = function() {
        m.T(this, "mouseover", this.sk);
        m.T(this, "mouseout", this.df);
        m.T(this, "click", this.df);
        m.T(this, "touchstart", this.Eo);
        m.T(this, "touchend", this.ii);
        m.T(this, "touchcancel", this.ii)
    };
    m.g.unregister = function() {
        m.Mo(this, "mouseover", this.sk);
        m.Mo(this, "mouseout", this.df);
        m.Mo(this, "click", this.df);
        m.Mo(this, "touchstart", this.Eo);
        m.Mo(this, "touchend", this.ii);
        m.Mo(this, "touchcancel", this.ii);
        this.dispose();
        m.Mq.K.unregister.call(this)
    };
    m.g.dispose = function() {
        for (var a in this.g)
            this.df(this.g[a]);
        this.g = {}
    };
    m.g.sk = function(a) {
        if (!(this.Th && 1E3 > (0, m.H)() - this.Th)) {
            var b = (0, window.parseInt)(this.getData(a, "tooltip-hide-timer"), 10);
            b && (this.removeData(a, "tooltip-hide-timer"), m.bb(b));
            var b = (0, m.p)(function() {
                eda(this, a);
                this.removeData(a, "tooltip-show-timer")
            }, this), c = (0, window.parseInt)(this.getData(a, "tooltip-show-delay"), 10) || 0, b = m.v(b, c);
            m.fl(a, "tooltip-show-timer", b.toString());
            a.title && (Mca(a, Oq(this, a)), a.title = "");
            b = m.oa(a).toString();
            this.g[b] = a
        }
    };
    m.g.df = function(a) {
        var b = (0, window.parseInt)(this.getData(a, "tooltip-show-timer"), 10);
        b && (m.bb(b), this.removeData(a, "tooltip-show-timer"));
        b = (0, m.p)(function() {
            if (a) {
                var b = m.F(Pq(this, a));
                b && (gda(b), m.ri(b), this.removeData(a, "content-id"))
            }
            this.removeData(a, "tooltip-hide-timer")
        }, this);
        b = m.v(b, 50);
        m.fl(a, "tooltip-hide-timer", b.toString());
        (b = this.getData(a, "tooltip-text")) && (a.title = b);
        b = m.oa(a).toString();
        delete this.g[b]
    };
    m.g.Eo = function(a, b, c) {
        this.Th = 0;
        a = Io(b, m.S(this), c.changedTouches[0].target);
        this.sk(a)
    };
    m.g.ii = function(a, b, c) {
        this.Th = (0, m.H)();
        a = Io(b, m.S(this), c.changedTouches[0].target);
        this.df(a)
    };
    m.s(m.Rq, m.Ko);
    m.fa(m.Rq);
    m.Rq.prototype.va = "subscription-button";
    m.Rq.prototype.register = function() {
        m.T(this, "click", this.pk);
        this.ca("subscription-subscribe-loading", this.So);
        this.ca("subscription-subscribe-loaded", this.To);
        this.ca("subscription-unsubscirbe-loading", this.So);
        this.ca("subscription-unsubscribe-loaded", this.To);
        this.ca("subscription-subscribe-success", this.Ry);
        this.ca("subscription-unsubscribe-success", this.Sy);
        this.ca("subscription-enable-ypc", this.Qy);
        this.ca("subscription-disable-ypc", this.Py)
    };
    m.Rq.prototype.unregister = function() {
        m.Mo(this, "click", this.pk);
        m.Rq.K.unregister.call(this)
    };
    var Vq = {
        lp: "hover-enabled",
        kz: "yt-uix-button-subscribe",
        lz: "yt-uix-button-subscribed",
        yB: "ypc-enabled",
        Cz: "yt-uix-button-subscription-container",
        Dz: "yt-subscription-button-disabled-mask-container"
    }, Uq = {
        EB: "channel-external-id",
        mz: "subscriber-count-show-when-subscribed",
        nz: "subscriber-count-tooltip",
        oz: "subscriber-count-title",
        IC: "href",
        Hp: "is-subscribed",
        HD: "parent-url",
        tE: "sessionlink",
        pz: "style-type",
        Ip: "subscription-id",
        LE: "target",
        zz: "ypc-enabled",
        wp: "ypc-offers-url"
    };
    m.g = m.Rq.prototype;
    m.g.pk = function(a) {
        var b = this.getData(a, "href"), c = Lq();
        if (b)
            a = this.getData(a, "target") || "_self", window.open(b, a);
        else if (c) {
            var b = this.Cg(a), c = this.getData(a, "sessionlink"), d;
            if (m.Tq(this, a)) {
                d = this.getData(a, "ypc-item-type");
                var e = this.getData(a, "ypc-item-id"), f = this.getData(a, Uq.wp);
                d = {
                    itemType: d,
                    itemId: e,
                    offersUrl: f,
                    subscriptionElement: a
                }
            } else 
                d = null;
            e = this.getData(a, "parent-url");
            m.Sq(this, a) ? (f = this.getData(a, "subscription-id"), m.Dq(sE, new bo(b, f, d, a, c, e))) : m.Dq(rE, new ao(b, d, a, c, e))
        } else 
            jda(this,
            a)
    };
    m.g.So = function(a) {
        this.qf(a, this.xi, !0)
    };
    m.g.To = function(a) {
        this.qf(a, this.xi, !1)
    };
    m.g.Ry = function(a, b) {
        this.qf(a, this.Ap, !0, b)
    };
    m.g.Sy = function(a) {
        this.qf(a, this.Ap, !1)
    };
    m.g.Qy = function(a) {
        this.qf(a, this.xz)
    };
    m.g.Py = function(a) {
        this.qf(a, this.wz)
    };
    m.g.Ap = function(a, b, c) {
        b ? (m.fl(a, Uq.Hp, "true"), c && m.fl(a, Uq.Ip, c)) : (this.removeData(a, Uq.Hp), this.removeData(a, Uq.Ip));
        hda(this, a)
    };
    m.g.Cg = function(a) {
        return this.getData(a, "channel-external-id")
    };
    m.g.xi = function(a, b) {
        var c = m.L(a, Vq.Cz);
        m.rb(c, Vq.Dz, b);
        a.setAttribute("aria-busy", b ? "true" : "false");
        a.disabled = b
    };
    m.g.xz = function(a) {
        var b=!!this.getData(a, "ypc-item-type"), c=!!this.getData(a, "ypc-item-id"), d=!!this.getData(a, Uq.wp);
        !m.Tq(this, a) && b && c && d && (m.C(a, "ypc-enabled"), m.fl(a, Uq.zz, "true"))
    };
    m.g.wz = function(a) {
        m.Tq(this, a) && (m.D(a, "ypc-enabled"), this.removeData(a, "ypc-enabled"))
    };
    m.g.Lk = function(a, b, c) {
        var d = m.Sa(arguments, 2);
        (0, m.y)(a, function(a) {
            b.apply(this, m.fj(a, d))
        }, this)
    };
    m.g.qf = function(a, b, c) {
        var d = ida(this, a), d = m.fj([d], m.Sa(arguments, 1));
        this.Lk.apply(this, d)
    };
    m.s(Wq, m.Ko);
    m.fa(Wq);
    m.g = Wq.prototype;
    m.g.va = "subscription-preferences-button";
    m.g.register = function() {
        m.T(this, "click", this.Fo)
    };
    m.g.unregister = function() {
        m.Mo(this, "click", this.Fo);
        Wq.K.unregister.call(this)
    };
    m.g.Fo = function(a) {
        var b = this.Cg(a);
        m.z("subscription-show-pref-overlay", a, b)
    };
    m.g.Cg = function(a) {
        return this.getData(a, "channel-external-id")
    };
    m.s(m.Xq, m.Ko);
    m.fa(m.Xq);
    m.Xq.prototype.va = "tabs";
    m.Xq.prototype.register = function() {
        m.T(this, "click", this.g, "tab")
    };
    m.Xq.prototype.unregister = function() {
        m.Mo(this, "click", this.g, "tab");
        m.Xq.K.unregister.call(this)
    };
    m.Xq.prototype.g = function(a) {
        var b = this.la(a), c = m.S(this, "selected"), d = this.getData(b, "uix-tabs-selected-extra-class");
        if (b = m.G(c, b)) {
            var e = Yq(this, b);
            m.D(b, c);
            d && m.D(b, d);
            m.P(e)
        }
        b = Yq(this, a);
        m.C(a, c);
        d && m.C(a, d);
        m.K(b);
        m.z("yt-uix-tabs-after-shown", a, b)
    };
    m.s(Zq, m.Ko);
    m.fa(Zq);
    Zq.prototype.va = "tile";
    Zq.prototype.register = function() {
        m.T(this, "click", this.g)
    };
    Zq.prototype.unregister = function() {
        m.Mo(this, "click", this.g)
    };
    Zq.prototype.g = function(a, b, c) {
        m.qi(c.target, "a") || m.qi(c.target, "button") ||!(a = m.G(m.S(this, "link"), a)) || (m.E&&!m.Fb(9) ? a.click() : (m.A(a, "yt-uix-sessionlink") && uq(a), m.A(a, "spf-link") ? m.be(a.href) : m.ae(a.href)))
    };
    var ar = window.yt && window.yt.uix && window.yt.uix.widgets_ || {};
    m.l("yt.uix.widgets_", ar, void 0);
    m.s(m.ir, m.t);
    var kr = [];
    m.g = m.ir.prototype;
    m.g.listen = function(a, b, c, d) {
        return m.jr(this, a, b, c, d)
    };
    m.g.ij = m.aa(14);
    m.g.Da = function(a, b, c, d, e) {
        if (m.ha(b))
            for (var f = 0; f < b.length; f++)
                this.Da(a, b[f], c, d, e);
        else if (a = m.tm(a, b, c || this.handleEvent, d, e || this.aa || this))
            m.sm(a), delete this.g[a.key];
        return this
    };
    m.g.removeAll = function() {
        m.tb(this.g, m.sm);
        this.g = {}
    };
    m.g.N = function() {
        m.ir.K.N.call(this);
        this.removeAll()
    };
    m.g.handleEvent = function() {
        throw Error("EventHandler.handleEvent not implemented");
    };
    m.fa(m.mr);
    m.mr.prototype.g = 0;
    m.s(m.nr, m.ym);
    m.g = m.nr.prototype;
    m.g.Zv = m.mr.getInstance();
    m.g.getId = function() {
        return this.ta || (this.ta = ":" + (this.Zv.g++).toString(36))
    };
    m.g.M = function() {
        return this.j
    };
    m.g.Ja = function(a) {
        return this.j ? this.g.Ja(a, this.j) : []
    };
    m.g.P = function(a) {
        return this.j ? this.g.P(a, this.j) : null
    };
    m.g.bi = function(a) {
        if (this.H && this.H != a)
            throw Error("Method not supported");
        m.nr.K.bi.call(this, a)
    };
    m.g.rb = function() {
        this.j = this.g.createElement("div")
    };
    m.g.render = function(a) {
        if (this.Va)
            throw Error("Component already rendered");
        this.j || this.rb();
        a ? a.insertBefore(this.j, null) : this.g.g.body.appendChild(this.j);
        this.H&&!this.H.Va || this.W()
    };
    m.g.oa = function(a) {
        if (this.Va)
            throw Error("Component already rendered");
        if (a && this.Sj(a)) {
            this.Yb=!0;
            var b = m.Yb(a);
            this.g && this.g.g == b || (this.g = m.Ii(a));
            this.bb(a);
            this.W()
        } else 
            throw Error("Invalid element to decorate");
    };
    m.g.Sj = function() {
        return !0
    };
    m.g.bb = function(a) {
        this.j = a
    };
    m.g.W = function() {
        this.Va=!0;
        rr(this, function(a) {
            !a.Va && a.M() && a.W()
        })
    };
    m.g.na = function() {
        rr(this, function(a) {
            a.Va && a.na()
        });
        this.S && this.S.removeAll();
        this.Va=!1
    };
    m.g.N = function() {
        this.Va && this.na();
        this.S && (this.S.dispose(), delete this.S);
        rr(this, function(a) {
            a.dispose()
        });
        !this.Yb && this.j && m.ri(this.j);
        this.H = this.j = this.L = this.D = null;
        m.nr.K.N.call(this)
    };
    m.g.Sg = function() {
        return this.j
    };
    m.g.removeChild = function(a, b) {
        if (a) {
            var c = m.ja(a) ? a: a.getId();
            a = this.L && c ? Al(this.L, c) || null : null;
            c && a && (m.wb(this.L, c), m.Oa(this.D, a), b && (a.na(), a.j && m.ri(a.j)), pr(a, null))
        }
        if (!a)
            throw Error("Child is not in parent component");
        return a
    };
    m.s(m.sr, m.nr);
    m.sr.prototype.na = function() {
        m.y(this.ja, m.N);
        m.nb(this.qa);
        m.sr.K.na.call(this)
    };
    m.sr.prototype.ca = function(a, b, c) {
        a = m.x(a, (0, m.p)(b, c || this));
        this.qa.push(a)
    };
    m.s(m.vr, m.sr);
    m.g = m.vr.prototype;
    m.g.W = function() {
        m.vr.K.W.call(this);
        this.U = this.M();
        this.ba = (0, window.parseInt)(m.I(this.U, "max-title-length"), 10) || 0;
        this.J = this.P("create-button");
        this.T = this.P("cancel-button");
        this.C = this.P("privacy-button");
        var a = m.or(this);
        this.k = this.P("title-input");
        a.listen(this.k, "keyup", this.Em);
        a.listen(this.k, "paste", this.Em);
        a.listen(this.U, "reset", this.Ju);
        a.listen(this.U, "submit", this.Ku);
        this.X = this.P("create-playlist-widget-privacy-menu");
        this.O = this.P("yt-uix-button-menu-item-selected");
        m.tr(this,
        "click", "privacy-option", this.Lu)
    };
    m.g.na = function() {
        zr(this);
        this.O = this.T = this.J = this.X = this.C = this.k = null;
        m.vr.K.na.call(this)
    };
    m.g.Em = function() {
        m.Wk(this.J, wr(this))
    };
    m.g.Lu = function(a) {
        (a = a.currentTarget) && xr(this, a);
        a = m.I(a, "value");
        this.P("privacy-value-input").value = a
    };
    m.g.Ku = function(a) {
        a.preventDefault();
        m.so(this.U, {
            context: this,
            R: this.ty,
            onError: this.sy
        });
        yr(this, !1)
    };
    m.g.ty = function(a, b) {
        zr(this);
        this.F && m.ma(this.F.cg) && this.F.cg({
            playlistId: b.response.playlistId,
            playlistName: b.response.playlistName,
            fo: b.response.playlistEditorUrl
        });
        m.z("yt-uix-videoactionmenu-hide")
    };
    m.g.sy = function(a, b) {
        if (b && b.errors && b.errors.length) {
            var c = this.P("title-input-container");
            m.Ap(c, b.errors[0]);
            yr(this, !0)
        }
    };
    m.g.Ju = function() {
        zr(this);
        this.F && m.ma(this.F.Nh) && this.F.Nh()
    };
    m.s(m.Ar, m.sr);
    m.g = m.Ar.prototype;
    m.g.W = function() {
        m.Ar.K.W.call(this);
        m.tr(this, "click", "addto-playlist-item", this.jy);
        m.tr(this, "click", "create-playlist-item", this.iy);
        m.tr(this, "click", "cancel-button", this.hy);
        m.tr(this, "keyup", "addto-search-box", this.ky);
        m.tr(this, "keydown", "playlists", this.uo);
        m.tr(this, "keydown", "create-playlist-item", this.uo);
        Cr(this)
    };
    m.g.uo = function(a) {
        13 != a.keyCode && 9 != a.keyCode && m.zo(a) && (a = this.P("addto-search-box")) && a.focus()
    };
    m.g.ky = function(a) {
        var b = a.currentTarget.value.toLowerCase(), c = this.Ja("addto-playlist-item");
        a=!1;
        for (var d = 0; d < c.length; d++) {
            var e=!1, f = m.G("playlist-name", c[d]);
            0 != f.textContent.toLowerCase().indexOf(b) ? e=!0 : (a=!0, f.innerHTML = "<strong>" + f.textContent.substring(0, b.length) + "</strong>" + f.textContent.substring(b.length));
            m.rb(c[d], "hid", e)
        }
        d = a;
        c = m.hb("ADDTO_CREATE_NEW_PLAYLIST");
        this.O = "";
        d || (c = m.hb("ADDTO_CREATE_PLAYLIST_DYNAMIC_TITLE", {
            dynamic_title_placeholder: '<strong>"' + b + '"</strong>'
        }), this.O =
        b);
        if (b = this.P("create-playlist-item"))
            b.innerHTML = c;
        b = this.P("addto-playlist-panel");
        b = m.Mb("UL", null, b);
        a ? (b[0].focus(), m.z("yt-uix-kbd-move-next", b[0], b[0])) : (a = this.P("create-playlist-item")) && a.focus()
    };
    m.g.N = function() {
        m.Ar.K.N.call(this);
        this.C.dispose();
        this.C = null
    };
    m.g.save = function(a, b) {
        Gr(this, a, b)
    };
    m.g.Yv = function(a, b, c, d) {
        this.C && (this.C.dispose(), this.C = null);
        c = d.html_content || "";
        this.M().innerHTML = c;
        a && nda(this, a);
        (a = this.P("addto-menu")) && fq(m.Vp.getInstance(), a);
        (a = this.P("addto-search-box")) && a.focus();
        Er(this);
        b && b()
    };
    m.g.jy = function(a) {
        a = a.currentTarget;
        !m.A(a, "loading") && m.I(a, "full-list-id") && (m.A(a, "contains-all-selected-videos") ? m.El(a, "to-be-removed") : m.El(a, "to-be-added"), Er(this), this.J || Gr(this))
    };
    m.g.nx = function(a) {
        Cr(this);
        a && a()
    };
    m.g.$j = function(a, b, c) {
        m.z("yt-uix-videoactionmenu-hide");
        if (c) {
            var d = "addto-menu-video-added";
            b || (d = "addto-menu-video-removed");
            m.z(d, c)
        }
        this.k && b && (b = this.k.split(",")[0]) && c.id && m.z("playlist-addto", b, c.id);
        Cr(this);
        a && a()
    };
    m.g.ck = function(a) {
        (0, m.y)(this.Ja("loading"), function(a) {
            m.D(a, "loading")
        });
        a && a()
    };
    m.g.iy = function() {
        if (!this.C) {
            var a = this.P("create-playlist-widget-form");
            a && (this.C = new m.vr({
                cg: (0, m.p)(this.iv, this)
            }), this.C.oa(a))
        }
        this.C && (this.J || (a = this.k, this.C.P("video-ids-input").value = a, a = this.F, this.C.P("source-playlist-id-input").value = a), Ir(this, "create-playlist-form"), this.P("title-input").value = this.O, a = this.P("create-button"), m.Wk(a, wr(this.C)), a.focus())
    };
    m.g.iv = function(a) {
        this.J ? Cr(this, (0, m.p)(function() {
            var b = this.Ja("addto-playlist-item");
            if (b = m.Ja(b, function(b) {
                return (m.I(b, "full-list-id") || "") == a.playlistId
            }))
                m.El(b, "to-be-added"), Er(this)
        }, this)) : this.$j(m.ea, !0, {
            id: a.playlistId,
            url: a.fo,
            title: a.playlistName
        })
    };
    m.g.hy = function(a) {
        a.preventDefault();
        Ir(this, "create-playlist-item")
    };
    m.s(Jr, m.Ko);
    m.fa(Jr);
    m.g = Jr.prototype;
    m.g.va = "videoactionmenu";
    m.g.register = function() {
        this.ca("addto-menu-show-create-playlist-panel", function() {
            m.rb(m.F("yt-uix-videoactionmenu-menu"), "create-playlist-item", !0)
        });
        this.ca("yt-uix-videoactionmenu-hide", this.$y);
        m.T(this, "click", this.yo, "button")
    };
    m.g.unregister = function() {
        m.Mo(this, "click", this.yo);
        Jr.K.unregister.call(this)
    };
    m.g.dispose = function() {
        var a = m.F("hidden-component-template-wrapper"), b = m.F("yt-uix-videoactionmenu-menu");
        a && b && a.appendChild(b)
    };
    m.g.yo = function(a) {
        var b = this.la(a), b = this.getData(b, "video-id") || "";
        this.g ? m.Br(this.g, b) : (this.g = new m.Ar({
            videoIds: b
        }), a && (b = m.Wp(a), b = m.G("add-to-widget", b), this.g.oa(b), a = m.Xp(a), m.Vp.getInstance().Mj(a)))
    };
    m.g.$y = function() {
        var a = m.F("yt-uix-videoactionmenu-menu");
        a && (a = m.Xp(a), m.dq(m.Vp.getInstance(), a))
    };
    var Mr=!1, Lr = null;
    var Sr, Tr = [], Qr = 0, HE = 0, Pr=!1;
    m.g = Ur.prototype;
    m.g.co = function() {};
    m.g.dispose = function() {
        this.k && (m.N(this.k), this.k = []);
        this.co()
    };
    m.g.listen = function(a, b, c, d) {
        this.k.push(m.J(a, b, (0, m.p)(c, d || this)))
    };
    m.g.di = function() {};
    m.g.Hs = function(a, b, c) {
        (c = m.$f(c.responseText)) && (a && Wr(c) || b && b.call(this, c))
    };
    m.s(m.Yr, Ur);
    m.g = m.Yr.prototype;
    m.g.om = function() {};
    m.g.close = function(a) {
        this.L=!1;
        m.jq(this.ia, a || "close");
        this.dispose()
    };
    m.g.create = function(a, b, c, d) {
        this.L && (b && (this.H = b), c && (this.J = c), a&&!this.C ? this.Zw({}, d) : this.gk())
    };
    m.g.open = function(a, b, c, d, e, f, h, k, r) {
        this.D = a;
        this.S = b;
        if (this.ga = m.F(this.D + "-lb")) {
            (a = m.Rg()) && a.pauseVideo && a.pauseVideo();
            this.C ? this.C && $r(this) : this.ia = new m.hq(this.ga, k);
            Zr(this, "loading");
            try {
                this.ia.setTitle("")
            } catch (w) {}
            this.ia.show();
            this.L=!0
        }
        d && this.create(e, f, h, r)
    };
    m.g.hasClass = m.aa(15);
    m.g.toggleClass = m.aa(16);
    m.g.Zw = function(a, b, c, d, e) {
        arguments.length && Zr(this, e || "loading");
        var f = a || {};
        this.H && (f.feature = this.H);
        this.J && (f.next = this.J);
        var h = b || {};
        h.session_token = m.Kl();
        Xr(this, "XML", this.S, f, h, (0, m.p)(this.Ix, this, c || null), d)
    };
    m.g.gk = function(a) {
        a && (m.F(this.D + "-dialog").innerHTML = a);
        this.g = m.G("yt-dialog-fg", this.ga);
        a = m.G("yt-pd-params", this.ga);
        this.O = m.I(a, "start-page") || "";
        Vr(this, this.g, "click", this.Qt, "yt-pd-close");
        Vr(this, this.g, "click", this.Pt, "yt-pd-setclass");
        Vr(this, this.g, "click", this.Rt, "yt-pd-setpage");
        this.om();
        Zr(this, "content");
        this.C=!0;
        $r(this)
    };
    m.g.Qt = function() {
        this.close("cancel")
    };
    m.g.Pt = function(a) {
        a = m.L(a.target, "yt-pd-setclass");
        var b = m.I(a, "off");
        b && m.rb(this.g, b, !1);
        (a = m.I(a, "on")) && m.rb(this.g, a, !0)
    };
    m.g.Rt = function(a) {
        a = m.L(a.target, "yt-pd-setpage");
        (a = m.I(a, "state-container-id")) && $r(this, a)
    };
    m.g.di = function(a) {
        m.Yr.K.di.call(this, a);
        this.close()
    };
    m.g.Ix = function(a, b) {
        var c = to(b);
        if (c) {
            var d = vo(c, "not_eligible"), e = vo(c, "error_message");
            d || e ? this.di(b) : Wr(b) || (c = vo(c, "html_content") || void 0, a ? a(b, (0, m.p)(this.gk, this, c)) : this.gk(c))
        }
    };
    m.s(as, Ur);
    m.g = as.prototype;
    m.g.lo = function() {};
    m.g.ek = function() {};
    m.g.init = function(a, b, c, d, e) {
        this.C = a || null;
        this.F = b || null;
        c ? d && e && (Vr(this, d, "mousedown", this.$n, e), Vr(this, d, "click", this.Zn, e)) : (this.$n(), this.Zn())
    };
    m.g.kj = function(a, b) {
        var c = Array.prototype.slice.call(arguments, 1);
        (m.ja(a) ? m.n(a + ".init") : a.init).apply(this, c)
    };
    m.g.$n = function(a) {
        this.qb || (this.C && m.Rf(this.C), this.F && m.rd(this.F, (0, m.p)(function() {
            (this.qb=!0, this.D) && this.ek(a)
        }, this)))
    };
    m.g.Zn = function(a) {
        this.lo(a);
        this.D=!0;
        this.qb && this.D && this.ek(a)
    };
    var cs = {};
    m.s(es, as);
    es.prototype.lo = function(a) {
        if (!this.g)
            if (a) {
                if (a = m.I(a.currentTarget, "upsell"), "settings" == a || "upload" == a || "playlist" == a || "guide" == a || "comment" == a || "message" == a || "captions" == a)
                    this.g = a
            } else 
                this.g = "default"
    };
    es.prototype.ek = function() {
        switch (this.g) {
        case "settings":
            this.j = "/profile";
            break;
        case "guide":
            this.j = m.u("CREATE_CHANNEL_NEXT_URL_GUIDE");
            break;
        case "upload":
            this.j = m.u("CREATE_CHANNEL_NEXT_URL_UPLOAD");
            break;
        default:
            this.j = window.document.location.href
        }
        m.u("CREATE_CHANNEL_NEXT_URL") && (this.j = m.u("CREATE_CHANNEL_NEXT_URL"));
        if (m.u("CREATE_CHANNEL_USERNAME_MODE"))
            this.kj("yt.www.account.CreateChannelDialog", this.H, this.g, this.j);
        else if (!this.L) {
            this.L=!0;
            var a = this.J, b = {
                session_token: m.Kl()
            };
            b.session_token =
            m.Kl();
            a = (0, m.p)(this.Hs, this, !1, a || null);
            Xr(this, "JSON", "/create_channel_ajax", {
                action_get_type: 1
            }, b, a)
        }
    };
    es.prototype.H = function() {
        var a = m.u("CREATE_CHANNEL_NEXT_URL");
        a && ("/" == a ? m.ae(a) : window.history.back())
    };
    es.prototype.J = function(a) {
        this.L=!1;
        switch (a.type) {
        case 4:
        case 6:
        case 7:
            bs(m.Yr).open("create-channel-identity", "/create_channel_ajax", "create_channel_ajax", !0, !0, this.g, this.j);
            break;
        case 2:
            this.kj("yt.www.account.CreateGplusDialog", this.H, this.g, this.j);
            break;
        case 8:
            m.ae("/oops");
        case 5:
            m.ae("/create_channel?action_blocked_misc=1");
        default:
            this.kj("yt.www.account.CreateChannelDialog", this.H, this.g, this.j)
        }
    };
    var wda = m.Ki({
        dA: "yt-alert-success",
        ERROR: "yt-alert-error",
        CF: "yt-alert-warn",
        oB: "yt-alert-announce",
        INFO: "yt-alert-info",
        BE: "yt-alert-status",
        XD: "yt-alert-promo"
    });
    var ks = [], IE = [], ms = null;
    is.prototype.H = function(a) {
        a = to(a);
        var b = vo(a, "invalid_request"), c = vo(a, "missing_setting_type"), d = vo(a, "already_seen_dialog");
        if (!(b || c || d))
            if (b = m.F("feed-privacy-dialog"), c = m.Rg(), d = vo(a, "success_message")) {
                var e = m.F("alerts");
                a = vo(a, "alert_template");
                a = m.Uk(a);
                m.hs("yt-alert-success", d, a, e);
                m.Ok(b);
                window.scroll(0, 0);
                m.jq(this.ia, "cancel");
                c && c.playVideo && c.playVideo()
            } else 
                c && c.pauseVideo && c.pauseVideo(), b.innerHTML = vo(a, "html_content"), IE.push(m.O(b, "click", (0, m.p)(this.g, this, !1), "make-activity-public-button")),
                IE.push(m.O(b, "click", (0, m.p)(this.g, this, !0), "make-activity-private-button")), m.z("yt-dom-content-change", b), this.ia.setState("content"), this.ia.show()
    };
    is.prototype.L = function() {
        m.jq(this.ia, "cancel")
    };
    is.prototype.g = function(a) {
        var b = {};
        b[a ? "action_make_private": "action_make_public"] = "1";
        a = {
            setting_type: this.j
        };
        this.ia.setState("working");
        ns(this, b, a)
    };
    m.s(os, m.Yr);
    os.prototype.om = function() {
        this.j || (this.j = []);
        this.j.push(m.O(this.g, "click", (0, m.p)(this.U, this), "identity-prompt-account-list-item"));
        this.j.push(m.O(this.g, "click", (0, m.p)(this.T, this), "specialized-identity-prompt-account-item"))
    };
    os.prototype.co = function() {
        this.j && (m.N(this.j), this.j = [])
    };
    os.prototype.U = function(a) {
        var b = m.F("identity-prompt-confirm-button");
        b ? b.disabled=!1 : (b = m.F("identity-prompt-form"), a = m.Mb("input", void 0, a.currentTarget), b && a && 1 == a.length && (a[0].checked=!0, b.submit()))
    };
    os.prototype.T = function(a) {
        a = m.I(a.currentTarget, "switch-url");
        m.F("dont_ask_again").checked && (a += "&dont_ask_again=on");
        m.ae(a)
    };
    var ys, ws, As, Bs, qs = [], xs=!1, vs=!1, ss = [];
    var Lda = (new Date).getTime();
    var Es = 0;
    var Hs, Is = [];
    var Ls = [], Ms = [], Ns = [];
    var Iga = "http://www.youtube.com https://www.youtube.com https://plus.google.com https://plus.googleapis.com https://plus.sandbox.google.com https://plusone.google.com https://plusone.sandbox.google.com https://apis.google.com https://apis.sandbox.google.com".split(" "), Jga = [It, Mt], JE = "subscription-subscribe-success subscription-unsubscribe-success subscription-subscribe-loading subscription-unsubscirbe-loading subscription-subscribe-loaded subscription-unsubscribe-loaded subscription-subscribe-failure subscription-unsubscribe-failure subscription-enable-ypc subscription-disable-ypc".split(" ");
    Ss.prototype.dispose = function() {
        Rs().removeOnConnectHandler("ytsubscribe")
    };
    Ss.prototype.C = function(a, b) {
        var c = b.id;
        this.j[c] = a;
        var d = {
            iframe: a,
            role: "yt"
        };
        Rs().connectIframes(d);
        c = (0, m.p)(this.D, this, c);
        a.registerWasClosed(c, this.g);
        a.register("msg-youtube-pubsub", this.k, this.g)
    };
    Ss.prototype.D = function(a) {
        delete this.j[a]
    };
    Ss.prototype.send = function(a, b) {
        m.tb(this.j, function(c) {
            c.send(a, b, void 0, this.g)
        }, this)
    };
    m.fa(Ts);
    m.g = Ts.prototype;
    m.g.init = function() {
        if (m.u("UNIVERSAL_HOVERCARDS")) {
            var a = (0, m.p)(this.xA, this), b = m.u("LOGGED_IN"), c = m.u("SESSION_INDEX"), d = m.u("DELEGATED_SESSION_ID"), e = {
                callback: a,
                "googleapis.config": {
                    signedIn: b
                },
                iframes: {
                    card: {
                        url: m.u("GAPI_HOST") + "/:session_prefix:_/hovercard/internalcard?p=36&hl=" + m.u("GAPI_LOCALE")
                    }
                }
            };
            b && (c && (e["googleapis.config"].sessionIndex = c), d && (e["googleapis.config"].sessionDelegate = d));
            m.Sn("card:gapi.iframes", {
                callback: a,
                config: e
            })
        }
    };
    m.g.dispose = function() {
        this.g && (this.g.dispose(), this.g = null);
        m.nb(this.j);
        this.j.length = 0;
        var a = m.n("gapi.card.unwatch");
        a && a()
    };
    m.g.xA = function() {
        var a = m.n("gapi.config.update");
        if (a) {
            var b = (m.lc(m.hc(window.location.href)[5] || null) || "/").split("/");
            a("card/source", "youtube" + (b[1] ? "." + b[1] : ""));
            a("card/hoverDelay", 450);
            a("card/loadDelay", 250);
            a("card/closeDelay", 200);
            a("card/usegapi", 1);
            a("card", {
                p: 36
            })
        }(a = m.n("gapi.card.watch")) && a();
        a = m.n("gapi.iframes.makeWhiteListIframesFilter")(Iga);
        b = (0, m.p)(this.ku, this);
        this.g = new Ss(b, a);
        for (a = 0; a < JE.length; a++)
            b = JE[a], this.j.push(m.x(b, m.q(this.lu, b), this))
    };
    m.g.ku = function(a) {
        var b = a.eventType;
        if ("pubsub" == b) {
            var c = a.pubsub;
            c && m.z.apply(null, c)
        } else if ("pubsub2" == b) {
            var b = m.Ja(Jga, function(b) {
                return b.toString() == a.topicString
            }), d = a.serializedData;
            if (b && (!b.Vh || d)) {
                if (b.Vh)
                    try {
                        c = Zn(b.Vh, d)
                    } catch (e) {
                    return 
                }
                m.Dq(b, c)
            }
        }
    };
    m.g.lu = function(a, b) {
        if (this.g) {
            var c = ho(arguments);
            this.g.send("cmd-youtube-pubsub", c)
        }
    };
    var KE = [], Vs = "";
    var LE = [];
    var Kga = new ct(24);
    ct.prototype.C = function() {
        for (var a = (0, m.H)(), b = this.g.length - 1; 0 <= b; b--)
            nt(this.g[b], a) && dt(this, b)
    };
    ct.prototype.add = function(a) {
        this.g.push(a);
        this.j || (this.j = m.ab((0, m.p)(this.C, this), this.k))
    };
    ct.prototype.remove = function(a) {
        a = (0, m.Na)(this.g, a);
        0 <= a && dt(this, a)
    };
    et.prototype.clone = function() {
        return new et(this.g, this.D, this.k, this.F, this.C, this.H, this.j, this.L)
    };
    et.prototype.equals = function(a) {
        return this.g == a.g && this.D == a.D && this.k == a.k && this.F == a.F && this.C == a.C && this.H == a.H && this.j == a.j && this.L == a.L
    };
    var Wda = new ht({
        x: .25,
        y: .1
    }, {
        x: .25,
        y: 1
    }), Yda = new ht({
        x: .42,
        y: 0
    }, {
        x: 1,
        y: 1
    }), $da = new ht({
        x: 0,
        y: 0
    }, {
        x: .58,
        y: 1
    }), bea = new ht({
        x: .42,
        y: 0
    }, {
        x: .58,
        y: 1
    });
    it.prototype.g = function() {};
    var kt;
    m.s(lt, it);
    lt.prototype.setProperty = function(a, b) {
        a = jt() + a;
        this.Fa.style[a] = b
    };
    lt.prototype.play = function() {
        this.Fa.style.opacity = this.j;
        m.v((0, m.p)(function() {
            this.setProperty("TransitionTimingFunction", this.Df);
            this.setProperty("TransitionDuration", this.duration + "s");
            this.setProperty("TransitionProperty", "opacity");
            m.Rk(this.Fa, m.Qe ? "webkitTransitionEnd" : m.Se ? "oTransitionEnd" : "transitionend", (0, m.p)(function() {
                this.setProperty("TransitionTimingFunction", "");
                this.setProperty("TransitionDuration", "");
                this.setProperty("TransitionProperty", "");
                this.Kg && this.Kg(this)
            }, this));
            this.Fa.style.opacity =
            this.k
        }, this), 20)
    };
    m.s(mt, it);
    mt.prototype.g = function(a) {
        this.C = 0;
        this.Bf = a.loop || Kga;
        this.L = cea(this.Df)
    };
    mt.prototype.play = function() {
        this.D = (0, m.H)();
        nt(this, this.D);
        this.Bf.add(this)
    };
    mt.prototype.pause = function() {
        this.Bf.remove(this)
    };
    mt.prototype.ka = function() {};
    m.s(ot, mt);
    ot.prototype.g = function(a) {
        var b = (0, window.parseFloat)(a.start), c = (0, window.parseFloat)(a.end);
        this.j = (0, window.isNaN)(b) ? 1 : b;
        this.k = (0, window.isNaN)(c) ? 0 : c;
        this.H=!m.ca(this.Fa.style.opacity);
        ot.K.g.call(this, a)
    };
    ot.prototype.F = function(a) {
        a = this.j - (this.j - this.k) * a;
        this.H ? this.Fa.style.filter = "alpha(opacity=" + Math.floor(100 * a) + ")" : this.Fa.style.opacity = a
    };
    ot.prototype.ka = function() {
        this.H && 1 == this.k && (this.Fa.style.filter = "")
    };
    m.s(pt, lt);
    pt.prototype.g = function(a) {
        var b = (0, window.parseFloat)(a.start), c = (0, window.parseFloat)(a.end);
        this.j = (0, window.isNaN)(b) ? 1 : b;
        this.k = (0, window.isNaN)(c) ? 0 : c;
        pt.K.g.call(this, a)
    };
    m.s(m.qt, mt);
    m.qt.prototype.g = function(a) {
        m.qt.K.g.call(this, a);
        this.j = a.start;
        this.k = a.end
    };
    m.qt.prototype.F = function(a) {
        this.Fa.scrollTop = (this.k - this.j) * a + this.j
    };
    var wt, ME = [], zt = [];
    var Jt, Lga, Mga, iea, Nga, Kt;
    m.s(m.At, Yn);
    m.s(Bt, Yn);
    m.s(Ct, Yn);
    m.s(Dt, Yn);
    m.s(Et, Yn);
    m.s(Ft, Yn);
    m.NE = new $n("ypc-init-purchase-for-container", m.At);
    Jt = new $n("ypc-delayedloader-load", Bt);
    Lga = new $n("ypc-guide-sync-success", Ct);
    Mga = new $n("ypc-purchase-success", Dt);
    iea = new $n("ypc-subscription-cancel", Ft);
    Nga = new $n("ypc-subscription-cancel-success", Et);
    Kt = new $n("ypc-init-subscription", Ft);
    var Ht=!1, OE = [], PE = [];
    m.g = m.Tt.prototype;
    m.g.ua = function() {
        return this.g.length + this.j.length
    };
    m.g.isEmpty = function() {
        return m.Rl(this.g) && m.Rl(this.j)
    };
    m.g.clear = function() {
        this.g = [];
        this.j = []
    };
    m.g.contains = function(a) {
        return m.La(this.g, a) || m.La(this.j, a)
    };
    m.g.remove = function(a) {
        var b = (0, m.naa)(this.g, a);
        if (0 > b)
            return m.Oa(this.j, a);
        m.Pa(this.g, b);
        return !0
    };
    m.g.za = function() {
        for (var a = [], b = this.g.length - 1; 0 <= b; --b)
            a.push(this.g[b]);
        for (var c = this.j.length, b = 0; b < c; ++b)
            a.push(this.j[b]);
        return a
    };
    var uea = (0, m.H)(), Xt = null, bu = Array(50), au =- 1, cu=!1;
    var mu = ["boadgeojelhgndaghljhdicfkmllpafd", "dliochdbjfkdbacpmhlcpmleaejidimm", "hfaagokkkhdbgiakmmlclaapfelnkoah", "fmfcbgogabcbclcofgocippekhfcmgfj", "enhhojjnijigcajfphajepfemndkmdlo"];
    pu.prototype.sendRequest = function(a, b, c, d, e, f, h) {
        a = {
            format: f ? "RAW": "JSON",
            method: a,
            context: this,
            timeout: 5E3,
            withCredentials: !!h,
            R: m.q(this.D, d, !f),
            onError: m.q(this.C, e),
            Zb: m.q(this.F, e)
        };
        c && (a.V = c, a.headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        });
        return m.R(b, a)
    };
    pu.prototype.D = function(a, b, c, d) {
        b ? a(d) : a({
            text: c.responseText
        })
    };
    pu.prototype.C = function(a, b) {
        a(Error("Request error: " + b.status))
    };
    pu.prototype.F = function(a) {
        a(Error("request timed out"))
    };
    ru.prototype.id = "";
    ru.prototype.name = "";
    ru.prototype.activityId = "";
    ru.prototype.status = "UNKNOWN";
    ru.prototype.toString = function() {
        return "{id:" + this.id + ",name:" + this.name + ",activityId:" + this.activityId + ",status:" + this.status + "}"
    };
    m.s(Au, m.t);
    Au.prototype.subscribe = function(a, b, c) {
        return this.ha() ? 0 : this.C.subscribe(a, b, c)
    };
    Au.prototype.unsubscribe = function(a, b, c) {
        return this.ha()?!1 : this.C.unsubscribe(a, b, c)
    };
    Au.prototype.Ib = function(a) {
        return this.ha()?!1 : this.C.Ib(a)
    };
    Au.prototype.publish = function(a, b) {
        return this.ha()?!1 : this.C.publish.apply(this.C, arguments)
    };
    m.s(Bu, Au);
    m.g = Bu.prototype;
    m.g.ab = function() {
        return this.screens
    };
    m.g.contains = function(a) {
        return !!yu(this.screens, a)
    };
    m.g.get = function(a) {
        return a ? zu(this.screens, a) : null
    };
    m.g.info = function(a) {
        $t(this.L, a)
    };
    m.g.warn = function(a) {
        $t(this.L, a)
    };
    m.s(Eu, m.Wf);
    m.g = Eu.prototype;
    m.g.pf = null;
    m.g.te = "";
    m.g.set = function(a, b) {
        this.pf.set(this.te + a, b)
    };
    m.g.get = function(a) {
        return this.pf.get(this.te + a)
    };
    m.g.remove = function(a) {
        this.pf.remove(this.te + a)
    };
    m.g.fb = function(a) {
        var b = this.pf.fb(!0), c = this, d = new m.bc;
        d.next = function() {
            for (var d = b.next(); d.substr(0, c.te.length) != c.te;)
                d = b.next();
            return a ? d.substr(c.te.length) : c.pf.get(d)
        };
        return d
    };
    Gu.prototype.id = "";
    Gu.prototype.name = "";
    m.g = Gu.prototype;
    m.g.app = "";
    m.g.type = "REMOTE_CONTROL";
    m.g.tA = "";
    m.g.avatar = "";
    m.g.theme = "u";
    m.g.equals = function(a) {
        return a ? this.id == a.id : !1
    };
    var Zu, Mu = "";
    m.s($u, Bu);
    m.g = $u.prototype;
    m.g.start = function() {
        this.Sn()
    };
    m.g.add = function(a, b, c) {
        if (this.contains(a))
            this.sd(a, a.name, b, c);
        else {
            var d = av(this, a.id);
            this.g.sendRequest("POST", d, {
                screen_name: a.name
            }, (0, m.p)(this.bw, this, a, b, c), (0, m.p)(this.xh, this, "add", c), !0, !0)
        }
    };
    m.g.remove = function(a, b, c) {
        if (this.contains(a)) {
            var d = av(this, a.id);
            this.g.sendRequest("DELETE", d, null, (0, m.p)(this.gw, this, a, b, c), (0, m.p)(this.xh, this, "remove", c), !0, !0)
        } else 
            a = "Trying to remove non-account screen: " + a.name, this.warn(a), c(Error(a))
    };
    m.g.sd = function(a, b, c, d) {
        if (this.contains(a)) {
            var e = av(this, a.id);
            this.g.sendRequest("PUT", e, {
                screen_name: b
            }, (0, m.p)(this.hw, this, a, b, c, d), (0, m.p)(this.xh, this, "update", d), !0, !0)
        } else 
            a = "Trying to update non-account screen: " + a.name, this.warn(a), d(Error(a))
    };
    m.g.N = function() {
        m.bb(this.j);
        $u.K.N.call(this)
    };
    m.g.Sn = function() {
        if (!this.ha()) {
            m.bb(this.j);
            var a = av(this);
            this.g.sendRequest("GET", a, null, (0, m.p)(this.au, this), m.p(this.xh, this, "load", m.ea), !1, !0)
        }
    };
    m.g.au = function(a) {
        if (!this.ha())
            if (a = iu(m.Ki(a)), Gea(this.screens, a))
                for (var b = 0, c = this.screens.length; b < c; ++b) {
                    var d = this.screens[b], e = zu(a, d.id);
                    e && e.token && (d.token = e.token)
                } else 
                    this.info("Updated account screens: " + ku(a)), this.screens = a, this.publish("screenChange"), a = this.ab().length ? 12E4 : 6E4, this.j = m.v((0, m.p)(this.Sn, this), a)
    };
    m.g.bw = function(a, b) {
        this.ha() || (Cu(this, a), this.publish("screenChange"), b(a))
    };
    m.g.gw = function(a, b) {
        this.ha() || (Du(this, a), this.publish("screenChange"), b(a))
    };
    m.g.hw = function(a, b, c) {
        this.ha() || (a = this.get(a.id), a.name = b, c(a), this.publish("screenChange"))
    };
    m.g.xh = function(a, b, c) {
        this.ha() || (a = "Failed to " + a + " account screen: " + c, this.warn(a), b(Error(a)))
    };
    m.s(bv, Au);
    var QE = [2E3, 2E3, 1E3, 1E3, 1E3, 2E3, 2E3, 5E3, 5E3, 1E4];
    m.g = bv.prototype;
    m.g.start = function() {
        !this.g && (0, window.isNaN)(this.j) && this.Pl()
    };
    m.g.stop = function() {
        this.g && (this.g.abort(), this.g = null);
        (0, window.isNaN)(this.j) || (m.bb(this.j), this.j = window.NaN)
    };
    m.g.N = function() {
        this.stop();
        bv.K.N.call(this)
    };
    m.g.Pl = function() {
        this.j = window.NaN;
        this.g = m.R(qu(this.L, "/pairing/get_screen"), {
            method: "POST",
            V: {
                pairing_code: this.H
            },
            timeout: 5E3,
            R: (0, m.p)(this.Fs, this),
            onError: (0, m.p)(this.Es, this),
            Zb: (0, m.p)(this.Gs, this)
        })
    };
    m.g.Fs = function(a, b) {
        this.g = null;
        var c = b.screen || {};
        c.dialId = this.D;
        c.name = this.F;
        this.publish("pairingComplete", new du(c))
    };
    m.g.Es = function(a) {
        this.g = null;
        a.status && 404 == a.status ? this.k >= QE.length ? this.publish("pairingFailed", Error("DIAL polling timed out")) : (a = QE[this.k], this.j = m.v((0, m.p)(this.Pl, this), a), this.k++) : this.publish("pairingFailed", Error("Server error " + a.status))
    };
    m.g.Gs = function() {
        this.g = null;
        this.publish("pairingFailed", Error("Server not responding"))
    };
    m.s(cv, Bu);
    m.g = cv.prototype;
    m.g.start = function() {
        dv(this) && this.publish("screenChange");
        Su() && ev(this);
        m.bb(this.g);
        this.g = m.v((0, m.p)(this.start, this), 1E4)
    };
    m.g.add = function(a, b) {
        dv(this);
        Cu(this, a);
        fv(this, !1);
        this.publish("screenChange");
        b(a);
        a.token || ev(this)
    };
    m.g.remove = function(a, b) {
        var c = dv(this);
        Du(this, a) && (fv(this, !1), c=!0);
        b(a);
        c && this.publish("screenChange")
    };
    m.g.sd = function(a, b, c, d) {
        var e = dv(this), f = this.get(a.id);
        f ? (f.name != b && (f.name = b, fv(this, !1), e=!0), c(a)) : d(Error("no such local screen."));
        e && this.publish("screenChange")
    };
    m.g.N = function() {
        m.bb(this.g);
        cv.K.N.call(this)
    };
    m.g.jx = function(a) {
        dv(this);
        var b = this.screens.length;
        a = a && a.screens || [];
        for (var c = 0, d = a.length; c < d; ++c) {
            var e = a[c], f = this.get(e.screenId);
            f && (f.token = e.loungeToken, --b)
        }
        fv(this, !b);
        b && this.warn("Missed " + b + " lounge tokens.")
    };
    m.g.ix = function(a) {
        this.warn("Requesting lounge tokens failed: " + a)
    };
    m.s(gv, Au);
    m.g = gv.prototype;
    m.g.start = function() {
        var a = (0, window.parseInt)(m.rg("yt-remote-fast-check-period") || "0", 10);
        (this.D = (0, m.H)() - 144E5 < a ? 0 : a) ? lv(this, !1) : (this.D = (0, m.H)() + 3E5, m.og("yt-remote-fast-check-period", this.D), this.Jj());
        m.J(window, "storage", this.L)
    };
    m.g.isEmpty = function() {
        return m.kk(this.g)
    };
    m.g.update = function() {
        iv("Updating availability on schedule.");
        var a = this.F(), b = m.Bl(this.g, function(b, d) {
            return b&&!!zu(a, d)
        }, this);
        kv(this, b)
    };
    m.g.N = function() {
        m.bb(this.k);
        this.k = window.NaN;
        this.j && (this.j.abort(), this.j = null);
        m.Pk(window, "storage", this.L);
        gv.K.N.call(this)
    };
    m.g.Jj = function() {
        m.bb(this.k);
        this.k = window.NaN;
        this.j && this.j.abort();
        var a = mv(this);
        if (Dj(a)) {
            var b = qu(this.H, "/pairing/get_screen_availability"), c = {
                lounge_token: m.ub(a).join(",")
            };
            this.j = this.H.sendRequest("POST", b, c, (0, m.p)(this.tv, this, a), (0, m.p)(this.sv, this))
        } else 
            kv(this, {}), lv(this, !1)
    };
    m.g.tv = function(a, b) {
        this.j = null;
        var c = m.ub(mv(this));
        if (m.Ml(c, m.ub(a))) {
            for (var c = b.screens || [], d = {}, e = 0, f = c.length; e < f; ++e)
                d[a[c[e].loungeToken]] = "online" == c[e].status;
            kv(this, d);
            lv(this, !1)
        } else 
            this.Ea("Changing Screen set during request."), this.Jj()
    };
    m.g.sv = function(a) {
        this.Ea("Screen availability failed: " + a);
        this.j = null;
        lv(this, !1)
    };
    m.g.Tq = function(a) {
        "yt-remote-online-screen-ids" == a.key && (this.g = hv(this), lv(this, !0))
    };
    m.g.Ea = function(a) {
        $t("OnlineScreenService", a)
    };
    m.s(nv, Bu);
    m.g = nv.prototype;
    m.g.start = function(a) {
        a ? (this.g || (this.g = new $u(this.D), this.g.subscribe("screenChange", (0, m.p)(this.Ef, this))), this.g.start()) : this.g && (m.Va(this.g), this.g = null, rv(this));
        this.j.start();
        this.k.start();
        this.screens.length && (this.publish("screenChange"), this.k.isEmpty() || this.publish("onlineScreenChange"))
    };
    m.g.add = function(a, b, c) {
        this.g ? this.g.add(a, (0, m.p)(function(a) {
            this.Ef();
            this.j.add(a, m.ea, m.ea);
            b(a)
        }, this), c) : this.j.add(a, b, c)
    };
    m.g.remove = function(a, b, c) {
        this.g && this.g.contains(a) ? (this.g.remove(a, (0, m.p)(function(a) {
            this.Ef();
            b(a)
        }, this), c), this.j.remove(a, m.ea, m.ea)) : this.j.remove(a, b, c);
        this.k.update()
    };
    m.g.sd = function(a, b, c, d) {
        this.g && this.g.contains(a) ? (this.g.sd(a, b, c, d), this.j.sd(a, b, m.ea, m.ea)) : this.j.contains(a) ? this.j.sd(a, b, c, d) : (a = "Updating name of unknown screen: " + a.name, this.warn(a), d(Error(a)))
    };
    m.g.ab = function(a) {
        return a ? this.screens : m.fj(this.screens, (0, m.qb)(this.F, function(a) {
            return !this.contains(a)
        }, this))
    };
    m.g.xm = function() {
        return (0, m.qb)(this.ab(!0), function(a) {
            return !!this.k.g[a.id]
        }, this)
    };
    m.g.ho = function(a, b) {
        for (var c = 2, d = b(a, c); qv(this, d);) {
            c++;
            if (20 < c)
                return a;
            d = b(a, c)
        }
        return d
    };
    m.g.uA = function(a, b, c, d) {
        m.R(qu(this.D, "/pairing/get_screen"), {
            method: "POST",
            V: {
                pairing_code: a
            },
            timeout: 5E3,
            R: (0, m.p)(function(a, d) {
                var h = new du(d.screen || {});
                if (!h.name || qv(this, h.name))
                    h.name = this.ho(h.name, b);
                c(pv(this, h))
            }, this),
            onError: (0, m.p)(function(a) {
                d(Error("pairing request failed: " + a.status))
            }, this),
            Zb: (0, m.p)(function() {
                d(Error("pairing request timed out."))
            }, this)
        })
    };
    m.g.N = function() {
        m.Va(this.g);
        m.Va(this.j);
        m.Va(this.k);
        nv.K.N.call(this)
    };
    m.g.Ef = function() {
        rv(this);
        this.publish("screenChange");
        this.k.update()
    };
    nv.prototype.dispose = nv.prototype.dispose;
    m.s(tv, Au);
    m.g = tv.prototype;
    m.g.ub = function(a) {
        this.ha() || (a && this.warn("" + a), this.k = null, this.publish("sessionScreen", null))
    };
    m.g.info = function(a) {
        $t(this.U, a)
    };
    m.g.warn = function(a) {
        $t(this.U, a)
    };
    m.g.En = function() {
        return null
    };
    m.g.aj = function(a) {
        var b = this.j;
        a ? (b.displayStatus = new window.chrome.cast.ReceiverDisplayStatus(a, []), b.displayStatus.showStop=!0) : b.displayStatus = null;
        window.chrome.cast.setReceiverDisplayStatus(b, (0, m.p)(function() {
            this.info("Updated receiver status for " + b.friendlyName + ": " + a)
        }, this), (0, m.p)(function() {
            this.warn("Failed to update receiver status for: " + b.friendlyName)
        }, this))
    };
    m.g.N = function() {
        this.aj("");
        tv.K.N.call(this)
    };
    m.s(vv, tv);
    m.g = vv.prototype;
    m.g.xj = function(a) {
        if (this.g) {
            if (this.g == a)
                return;
            this.warn("Overriding cast sesison with new session object");
            this.g.removeUpdateListener(this.L);
            this.g.removeMessageListener("urn:x-cast:com.google.youtube.mdx", this.H)
        }
        this.g = a;
        this.g.addUpdateListener(this.L);
        this.g.addMessageListener("urn:x-cast:com.google.youtube.mdx", this.H);
        this.D && xv(this);
        yv(this, "getMdxSessionStatus")
    };
    m.g.Ke = function(a) {
        this.info("launchWithParams: " + m.ag(a));
        this.D = a;
        this.g && xv(this)
    };
    m.g.stop = function() {
        this.g ? this.g.stop((0, m.p)(function() {
            this.ub()
        }, this), (0, m.p)(function() {
            this.ub(Error("Failed to stop receiver app."))
        }, this)) : this.ub(Error("Stopping cast device witout session."))
    };
    m.g.aj = m.ea;
    m.g.N = function() {
        this.info("disposeInternal");
        m.bb(this.F);
        this.F = 0;
        this.g && (this.g.removeUpdateListener(this.L), this.g.removeMessageListener("urn:x-cast:com.google.youtube.mdx", this.H));
        this.g = null;
        vv.K.N.call(this)
    };
    m.g.Nt = function(a, b) {
        if (!this.ha())
            if (b) {
                var c = m.uk(b);
                if (c) {
                    var d = "" + c.type, c = c.data || {};
                    this.info("onYoutubeMessage_: " + d + " " + m.ag(c));
                    switch (d) {
                    case "mdxSessionStatus":
                        wv(this, c.screenId);
                        break;
                    default:
                        this.warn("Unknown youtube message: " + d)
                    }
                } else 
                    this.warn("Unable to parse message.")
            } else 
                this.warn("No data in message.")
        };
    m.g.En = function() {
        return this.g
    };
    m.g.Ot = function(a) {
        this.ha() || a || (this.warn("Cast session died."), this.ub())
    };
    m.s(zv, tv);
    m.g = zv.prototype;
    m.g.xj = function(a) {
        this.F = a;
        this.F.addUpdateListener(this.T)
    };
    m.g.Ke = function(a) {
        this.D = a;
        this.L()
    };
    m.g.stop = function() {
        this.g();
        this.g = m.ea;
        m.bb(this.H);
        this.F ? this.F.stop((0, m.p)(this.ub, this, null), (0, m.p)(this.ub, this, "Failed to stop DIAL device.")) : this.ub()
    };
    m.g.N = function() {
        this.g();
        this.g = m.ea;
        m.bb(this.H);
        this.F && this.F.removeUpdateListener(this.T);
        this.F = null;
        zv.K.N.call(this)
    };
    m.g.Js = function(a) {
        this.ha() || a || (this.warn("DIAL session died."), this.g(), this.g = m.ea, this.ub())
    };
    m.g.$i = function(a) {
        this.O = uu();
        if (this.D) {
            var b = new window.chrome.cast.DialLaunchResponse(!0, Bv(this));
            a(b);
            Av(this)
        } else 
            this.L = (0, m.p)(function() {
                m.bb(this.H);
                this.L = m.ea;
                this.H = window.NaN;
                var b = new window.chrome.cast.DialLaunchResponse(!0, Bv(this));
                a(b);
                Av(this)
            }, this), this.H = m.v((0, m.p)(function() {
                this.L()
            }, this), 100)
    };
    m.g.Ur = function(a, b) {
        ov(this.S, this.J.receiver.label, a, this.j.friendlyName, (0, m.p)(function(a) {
            a && a.token ? (uv(this, a), b(new window.chrome.cast.DialLaunchResponse(!1))) : this.$i(b)
        }, this), (0, m.p)(function(a) {
            this.warn("Failed to get DIAL screen: " + a);
            this.$i(b)
        }, this))
    };
    m.s(Cv, tv);
    Cv.prototype.stop = function() {
        this.ub()
    };
    Cv.prototype.xj = m.ea;
    Cv.prototype.Ke = function() {
        m.bb(this.g);
        this.g = window.NaN;
        var a = zu(this.S.ab(), this.j.label);
        a ? uv(this, a) : this.ub(Error("No such screen"))
    };
    Cv.prototype.N = function() {
        m.bb(this.g);
        this.g = window.NaN;
        Cv.K.N.call(this)
    };
    m.s(Dv, Au);
    m.g = Dv.prototype;
    m.g.init = function(a, b) {
        window.chrome.cast.timeout.requestSession = 3E4;
        var c = new window.chrome.cast.SessionRequest("233637DE");
        c.dialRequest = new window.chrome.cast.DialRequest("YouTube");
        var d = window.chrome.cast.AutoJoinPolicy.TAB_AND_ORIGIN_SCOPED, e = a ? window.chrome.cast.DefaultActionPolicy.CAST_THIS_TAB: window.chrome.cast.DefaultActionPolicy.CREATE_SESSION, c = new window.chrome.cast.ApiConfig(c, (0, m.p)(this.Jm, this), (0, m.p)(this.Ru, this), d, e);
        c.customDialLaunchCallback = (0, m.p)(this.Qu, this);
        window.chrome.cast.initialize(c,
        (0, m.p)(function() {
            this.ha() || (window.chrome.cast.addReceiverActionListener(this.D), Vt(Gv), this.j.subscribe("onlineScreenChange", (0, m.p)(this.vl, this)), this.k = Jv(this), window.chrome.cast.setCustomReceivers(this.k, m.ea, (0, m.p)(function(a) {
                this.Ea("Failed to set initial custom receivers: " + m.ag(a))
            }, this)), this.publish("yt-remote-cast2-availability-change", Hv(this)), b(!0))
        }, this), function(a) {
            this.Ea("Failed to initialize API: " + m.ag(a));
            b(!1)
        })
    };
    m.g.oA = function(a, b) {
        Fv("Setting connected screen ID: " + a + " -> " + b);
        if (this.g) {
            var c = this.g.k;
            if (!a || c && c.id != a)
                Fv("Unsetting old screen status: " + this.g.j.friendlyName), m.Va(this.g), this.g = null
        }
        if (a && b) {
            if (!this.g) {
                c = zu(this.j.ab(), a);
                if (!c) {
                    Fv("setConnectedScreenStatus: Unknown screen.");
                    return 
                }
                var d = Ev(this, c);
                d || (Fv("setConnectedScreenStatus: Connected receiver not custom..."), d = new window.chrome.cast.Receiver(c.uuid ? c.uuid : c.id, c.name), d.receiverType = window.chrome.cast.ReceiverType.CUSTOM, this.k.push(d),
                window.chrome.cast.setCustomReceivers(this.k, m.ea, (0, m.p)(function(a) {
                    this.Ea("Failed to set initial custom receivers: " + m.ag(a))
                }, this)));
                Fv("setConnectedScreenStatus: new active receiver: " + d.friendlyName);
                Iv(this, new Cv(this.j, d), !0)
            }
            this.g.aj(b)
        } else 
            Fv("setConnectedScreenStatus: no screen.")
    };
    m.g.pA = function(a) {
        this.ha() ? this.Ea("Setting connection data on disposed cast v2") : this.g ? this.g.Ke(a) : this.Ea("Setting connection data without a session")
    };
    m.g.stopSession = function() {
        this.ha() ? this.Ea("Stopping session on disposed cast v2") : this.g ? (this.g.stop(), m.Va(this.g), this.g = null) : Fv("Stopping non-existing session")
    };
    m.g.requestSession = function() {
        window.chrome.cast.requestSession((0, m.p)(this.Jm, this), (0, m.p)(this.Bz, this))
    };
    m.g.N = function() {
        this.j.unsubscribe("onlineScreenChange", (0, m.p)(this.vl, this));
        window.chrome && window.chrome.cast && window.chrome.cast.removeReceiverActionListener(this.D);
        Zt(Gv);
        m.Va(this.g);
        Dv.K.N.call(this)
    };
    m.g.Ea = function(a) {
        $t("Controller", a)
    };
    m.g.Kl = function(a, b) {
        this.g == a && (b || Iv(this, null), this.publish("yt-remote-cast2-session-change", b))
    };
    m.g.jr = function(a, b) {
        if (!this.ha())
            if (a)
                switch (Fv("onReceiverAction_ " + a.label + " / " + a.friendlyName + "-- " + b), b) {
                case window.chrome.cast.ReceiverAction.CAST:
                    if (this.g)
                        if (this.g.j.label != a.label)
                            Fv("onReceiverAction_: Stopping active receiver: " + this.g.j.friendlyName), this.g.stop();
                        else {
                            Fv("onReceiverAction_: Casting to active receiver.");
                            this.g.k && this.publish("yt-remote-cast2-session-change", this.g.k);
                            break
                        }
                        switch (a.receiverType) {
                        case window.chrome.cast.ReceiverType.CUSTOM:
                            Iv(this, new Cv(this.j, a));
                            break;
                        case window.chrome.cast.ReceiverType.DIAL:
                            Iv(this, new zv(this.j, a));
                            break;
                        case window.chrome.cast.ReceiverType.CAST:
                            Iv(this, new vv(this.j, a));
                            break;
                        default:
                            this.Ea("Unknown receiver type: " + a.receiverType);
                            return 
                        }
                        break;
                    case window.chrome.cast.ReceiverAction.STOP:
                        this.g && this.g.j.label == a.label ? this.g.stop() : this.Ea("Stopping receiver w/o session: " + a.friendlyName)
                } else 
                    this.Ea("onReceiverAction_ called without receiver.")
    };
    m.g.Qu = function(a) {
        if (this.ha())
            return window.Promise.reject(Error("disposed"));
        var b = a.receiver;
        b.receiverType != window.chrome.cast.ReceiverType.DIAL && (this.Ea("Not DIAL receiver: " + b.friendlyName), b.receiverType = window.chrome.cast.ReceiverType.DIAL);
        var c = this.g ? this.g.j: null;
        if (!c || c.label != b.label)
            return this.Ea("Receiving DIAL launch request for non-clicked DIAL receiver: " + b.friendlyName), window.Promise.reject(Error("illegal DIAL launch"));
        if (c && c.label == b.label && c.receiverType != window.chrome.cast.ReceiverType.DIAL) {
            if (this.g.k)
                return Fv("Reselecting dial screen."),
                this.publish("yt-remote-cast2-session-change", this.g.k), window.Promise.resolve(new window.chrome.cast.DialLaunchResponse(!1));
            this.Ea('Changing CAST intent from "' + c.receiverType + '" to "dial" for ' + b.friendlyName);
            Iv(this, new zv(this.j, b))
        }
        b = this.g;
        b.J = a;
        return b.J.appState == window.chrome.cast.DialAppState.RUNNING ? new window.Promise((0, m.p)(b.Ur, b, (b.J.extraData || {}).screenId || null)) : new window.Promise((0, m.p)(b.$i, b))
    };
    m.g.Jm = function(a) {
        if (!this.ha()) {
            Fv("New cast session ID: " + a.sessionId);
            var b = a.receiver;
            if (b.receiverType != window.chrome.cast.ReceiverType.CUSTOM) {
                if (!this.g)
                    if (b.receiverType == window.chrome.cast.ReceiverType.CAST)
                        Fv("Got resumed cast session before resumed mdx connection."), Iv(this, new vv(this.j, b), !0);
                    else {
                        this.Ea("Got non-cast session without previous mdx receiver event, or mdx resume.");
                        return 
                    }
                var c = this.g.j, d = zu(this.j.ab(), c.label);
                d && eu(d, b.label) && c.receiverType != window.chrome.cast.ReceiverType.CAST &&
                b.receiverType == window.chrome.cast.ReceiverType.CAST && (Fv("onSessionEstablished_: manual to cast session change " + b.friendlyName), m.Va(this.g), this.g = new vv(this.j, b), this.g.subscribe("sessionScreen", (0, m.p)(this.Kl, this, this.g)), this.g.Ke(null));
                this.g.xj(a)
            }
        }
    };
    m.g.vA = function() {
        return this.g ? this.g.En() : null
    };
    m.g.Bz = function(a) {
        this.ha() || (this.Ea("Failed to estabilish a session: " + m.ag(a)), a.code != window.chrome.cast.ErrorCode.CANCEL && Iv(this, null))
    };
    m.g.Ru = function(a) {
        Fv("Receiver availability updated: " + a);
        if (!this.ha()) {
            var b = Hv(this);
            this.F = a == window.chrome.cast.ReceiverAvailability.AVAILABLE;
            Hv(this) != b && this.publish("yt-remote-cast2-availability-change", Hv(this))
        }
    };
    m.g.vl = function() {
        if (!this.ha()) {
            var a = Hv(this);
            this.k = Jv(this);
            Fv("Updating custom receivers: " + m.ag(this.k));
            window.chrome.cast.setCustomReceivers(this.k, m.ea, (0, m.p)(function() {
                this.Ea("Failed to set custom receivers.")
            }, this));
            var b = Hv(this);
            b != a && this.publish("yt-remote-cast2-availability-change", b)
        }
    };
    Dv.prototype.setLaunchParams = Dv.prototype.pA;
    Dv.prototype.setConnectedScreenStatus = Dv.prototype.oA;
    Dv.prototype.stopSession = Dv.prototype.stopSession;
    Dv.prototype.getCastSession = Dv.prototype.vA;
    Dv.prototype.requestSession = Dv.prototype.requestSession;
    Dv.prototype.init = Dv.prototype.init;
    Dv.prototype.dispose = Dv.prototype.dispose;
    var Rv = null, Tv = [];
    $v.prototype.setVideoId = function(a) {
        bw(this, "setVideoId");
        var b = this.index;
        this.index = (0, m.Na)(this.videoIds, a);
        b != this.index && cw(this);
        return - 1 != b
    };
    $v.prototype.add = function(a) {
        bw(this, "add");
        return a&&!m.La(this.videoIds, a) ? (this.videoIds.push(a), !0) : !1
    };
    $v.prototype.remove = function(a) {
        bw(this, "remove");
        var b = ew(this);
        return m.Oa(this.videoIds, a) ? (this.index = (0, m.Na)(this.videoIds, b), !0) : !1
    };
    $v.prototype.clone = function() {
        return new $v(iw(this))
    };
    m.s(jw, Au);
    m.g = jw.prototype;
    m.g.getState = function() {
        return this.k
    };
    m.g.play = function() {
        1 == this.getState() ? (this.g ? this.g.play(null, m.ea, (0, m.p)(function() {
            this.Ea("Failed to play video with cast v2 channel.");
            rw(this, "play")
        }, this)) : rw(this, "play"), pw(this, 1, gw(mw(this))), sw(this)) : ow(this, this.play)
    };
    m.g.pause = function() {
        1 == this.getState() ? (this.g ? this.g.pause(null, m.ea, (0, m.p)(function() {
            this.Ea("Failed to pause video with cast v2 channel.");
            rw(this, "pause")
        }, this)) : rw(this, "pause"), pw(this, 2, gw(mw(this))), sw(this)) : ow(this, this.pause)
    };
    m.g.stop = function() {
        if (1 == this.getState()) {
            this.g ? this.g.stop(null, m.ea, (0, m.p)(function() {
                this.Ea("Failed to stop video with cast v2 channel.");
                rw(this, "stopVideo")
            }, this)) : rw(this, "stopVideo");
            var a = mw(this);
            dw(a);
            qw(this, a);
            sw(this)
        } else 
            ow(this, this.stop)
    };
    m.g.Wn = function(a) {
        if (1 == this.getState()) {
            rw(this, "addVideo", {
                videoId: a
            });
            var b = mw(this);
            b.j || (b.add(a), qw(this, b))
        } else 
            ow(this, m.q(this.Wn, a))
    };
    m.g.Lp = function(a) {
        1 == this.getState() ? rw(this, "addVideos", {
            listId: a
        }) : ow(this, m.q(this.Lp, a))
    };
    m.g.fp = function(a) {
        m.Rl(a) ? this.Ea("Ignore add videos request due to empty list") : 1 == this.getState() ? rw(this, "addVideos", {
            videoIds: a.join(",")
        }) : ow(this, m.q(this.fp, a))
    };
    m.g.Xn = function(a) {
        if (1 == this.getState()) {
            rw(this, "removeVideo", {
                videoId: a
            });
            var b = mw(this);
            b.j || (b.remove(a), qw(this, b))
        } else 
            ow(this, m.q(this.Xn, a))
    };
    m.g.dispose = function() {
        if (3 != this.k) {
            var a = this.k;
            this.k = 3;
            this.publish("proxyStateChange", a, this.k)
        }
        jw.K.dispose.call(this)
    };
    m.g.N = function() {
        m.bb(this.F);
        this.F = window.NaN;
        nw(this);
        this.D = null;
        this.H.clear();
        lw(this, null);
        jw.K.N.call(this)
    };
    m.g.Qi = function(a) {
        if ((a != this.k || 2 == a) && 3 != this.k && 0 != a) {
            var b = this.k;
            this.k = a;
            this.publish("proxyStateChange", b, a);
            if (1 == a)
                for (; !this.H.isEmpty();)
                    m.Ut(this.H).apply(this);
            else 
                3 == a && this.dispose()
        }
    };
    m.g.Cx = function(a) {
        ("remotePlayerChange" != a || (0, window.isNaN)(this.F)) && this.publish(a)
    };
    m.g.Pq = function(a) {
        if (!a)
            this.yf(null), lw(this, null);
        else if (this.j.receiver.volume) {
            a = this.j.receiver.volume;
            var b = mw(this);
            if (b.C != a.level || b.k != a.muted)
                $t("CP", "Cast volume update: " + a.level + (a.muted ? " muted" : "")), b.C = Math.round(100 * a.level || 0), b.k=!!a.muted, qw(this, b), sw(this)
        }
    };
    m.g.yf = function(a) {
        $t("CP", "Cast media: "+!!a);
        this.g && this.g.removeUpdateListener(this.S);
        if (this.g = a)
            this.g.addUpdateListener(this.S), tw(this), sw(this)
    };
    m.g.Oq = function(a) {
        a ? (tw(this), sw(this)) : this.yf(null)
    };
    m.g.Ea = function(a) {
        $t("CP", a)
    };
    var zw=!1, xw = null, Aw = m.n("yt.mdx.remote.castapi.devices_") || [];
    m.l("yt.mdx.remote.castapi.devices_", Aw, void 0);
    new Iw;
    Iw.prototype.set = function(a) {
        this.g = a
    };
    Iw.prototype.get = function() {
        return this.g
    };
    Jw.prototype.stringify = function(a) {
        return m.bg(this.j, a)
    };
    Jw.prototype.parse = function(a) {
        return this.g(a)
    };
    Kw.prototype.g = null;
    var RE;
    m.s(Nw, Kw);
    RE = new Nw;
    m.g = Pw.prototype;
    m.g.Id = null;
    m.g.dc=!1;
    m.g.bf = null;
    m.g.Gj = null;
    m.g.Lf = null;
    m.g.He = null;
    m.g.jd = null;
    m.g.rd = null;
    m.g.Kd = null;
    m.g.Ra = null;
    m.g.Tf = 0;
    m.g.gc = null;
    m.g.oh = null;
    m.g.hd = null;
    m.g.Cf =- 1;
    m.g.Ql=!0;
    m.g.Nd=!1;
    m.g.cj = 0;
    m.g.Vg = null;
    var Xw = {}, Vw = {};
    m.g = Pw.prototype;
    m.g.setTimeout = function(a) {
        this.nb = a
    };
    m.g.st = function(a) {
        a = a.target;
        var b = this.Vg;
        b && 3 == rx(a) ? b.pn() : this.$l(a)
    };
    m.g.$l = function(a) {
        try {
            if (a == this.Ra)
                t: {
                var b = rx(this.Ra), c = this.Ra.C, d = this.Ra.getStatus();
                if (m.E&&!m.Ib(10) || m.Qe&&!m.Fb("420+")) {
                    if (4 > b)
                        break t
                } else if (3 > b || 3 == b&&!m.Se&&!m.ux(this.Ra))
                    break t;
                    this.Nd || 4 != b || 7 == c || (8 == c || 0 >= d ? this.g.k(3) : this.g.k(2));
                    dx(this);
                    var e = this.Ra.getStatus();
                    this.Cf = e;
                    var f = m.ux(this.Ra);
                    (this.dc = 200 == e) ? (4 == b && Zw(this), this.D ? (Uw(this, b, f), m.Se && this.dc && 3 == b && (this.k.listen(this.j, "tick", this.dr), this.j.start())) : Yw(this, f), this.dc&&!this.Nd && (4 == b ? this.g.Qg(this) :
                    (this.dc=!1, Tw(this)))) : (400 == e && 0 < f.indexOf("Unknown SID") ? this.hd = 3 : this.hd = 0, Ww(), Zw(this), $w(this))
                }
        } catch (h) {
            this.Ra && m.ux(this.Ra)
        } finally {}
    };
    m.g.dr = function() {
        var a = rx(this.Ra), b = m.ux(this.Ra);
        this.Tf < b.length && (dx(this), Uw(this, a, b), this.dc && 4 != a && Tw(this))
    };
    m.g.Bt = function(a) {
        cx((0, m.p)(this.nA, this, a), 0)
    };
    m.g.nA = function(a) {
        this.Nd || (dx(this), Yw(this, a), Tw(this))
    };
    m.g.cm = function(a) {
        cx((0, m.p)(this.mA, this, a), 0)
    };
    m.g.mA = function(a) {
        this.Nd || (Zw(this), this.dc = a, this.g.Qg(this), this.g.k(4))
    };
    m.g.cancel = function() {
        this.Nd=!0;
        Zw(this)
    };
    m.g.yz = function() {
        this.bf = null;
        var a = (0, m.H)();
        0 <= a - this.Gj ? (2 != this.He && this.g.k(3), Zw(this), this.hd = 2, Ww(), $w(this)) : bx(this, this.Gj - a)
    };
    m.g = hx.prototype;
    m.g.hj = null;
    m.g.Mb = null;
    m.g.uh=!1;
    m.g.bm = null;
    m.g.sh = null;
    m.g.wj = null;
    m.g.Qj = null;
    m.g.Vb = null;
    m.g.Tc =- 1;
    m.g.Sf = null;
    m.g.Kf = null;
    m.g.connect = function(a) {
        this.Qj = a;
        a = jx(this.g, null, this.Qj);
        Ww();
        this.bm = (0, m.H)();
        var b = this.g.J;
        null != b ? (this.Sf = b[0], (this.Kf = b[1]) ? (this.Vb = 1, ix(this)) : (this.Vb = 2, kx(this))) : (m.Zj(a, "MODE", "init"), this.Mb = new Pw(this, 0, void 0, void 0, void 0), this.Mb.Id = this.hj, Sw(this.Mb, a, !1, null, !0), this.Vb = 0)
    };
    m.g.su = function(a) {
        if (a)
            this.Vb = 2, kx(this);
        else {
            Ww();
            var b = this.g;
            b.$b = b.md.Tc;
            Lx(b, 9)
        }
        a && this.k(2)
    };
    m.g.mj = function(a) {
        return this.g.mj(a)
    };
    m.g.abort = function() {
        this.Mb && (this.Mb.cancel(), this.Mb = null);
        this.Tc =- 1
    };
    m.g.Dn = function() {
        return !1
    };
    m.g.Am = function(a, b) {
        this.Tc = a.Cf;
        if (0 == this.Vb)
            if (b) {
                try {
                    var c = this.j.parse(b)
                } catch (d) {
                    c = this.g;
                    c.$b = this.Tc;
                    Lx(c, 2);
                    return 
                }
                this.Sf = c[0];
                this.Kf = c[1]
        } else 
            c = this.g, c.$b = this.Tc, Lx(c, 2);
        else if (2 == this.Vb)
            if (this.uh)
                Ww(), this.wj = (0, m.H)();
            else if ("11111" == b) {
                if (Ww(), this.uh=!0, this.sh = (0, m.H)(), c = this.sh - this.bm, !m.E || m.Ib(10) || 500 > c)
                    this.Tc = 200, this.Mb.cancel(), Ww(), lx(this.g, this, !0)
            } else 
                Ww(), this.sh = this.wj = (0, m.H)(), this.uh=!1
        };
    m.g.Qg = function() {
        this.Tc = this.Mb.Cf;
        if (this.Mb.dc)
            0 == this.Vb ? this.Kf ? (this.Vb = 1, ix(this)) : (this.Vb = 2, kx(this)) : 2 == this.Vb && (a=!1, (a=!m.E || m.Ib(10) ? this.uh : 200 > this.wj - this.sh?!1 : !0) ? (Ww(), lx(this.g, this, !0)) : (Ww(), lx(this.g, this, !1)));
        else {
            0 == this.Vb ? Ww() : 2 == this.Vb && Ww();
            var a = this.g;
            a.$b = this.Tc;
            Lx(a, 2)
        }
    };
    m.g.Jf = function() {
        return this.g.Jf()
    };
    hx.prototype.isActive = function() {
        return this.g.isActive()
    };
    hx.prototype.k = function(a) {
        this.g.k(a)
    };
    m.s(m.mx, m.ym);
    var Oga = /^https?$/i, Pga = ["POST", "PUT"];
    m.g = m.mx.prototype;
    m.g.Nm = m.aa(17);
    m.g.send = function(a, b, c, d) {
        if (this.g)
            throw Error("[goog.net.XhrIo] Object is active with another request=" + this.L + "; newUri=" + a);
        b = b ? b.toUpperCase() : "GET";
        this.L = a;
        this.D = "";
        this.C = 0;
        this.ba = b;
        this.U=!1;
        this.j=!0;
        this.g = this.T ? Ow(this.T) : Ow(RE);
        this.O = this.T ? Lw(this.T) : Lw(RE);
        this.g.onreadystatechange = (0, m.p)(this.al, this);
        try {
            Hk(sx(this, "Opening Xhr")), this.X=!0, this.g.open(b, String(a), !0), this.X=!1
        } catch (e) {
            Hk(sx(this, "Error opening Xhr: " + e.message));
            nx(this, e);
            return 
        }
        a = c || "";
        var f = this.headers.clone();
        d && m.$i(d, function(a, b) {
            f.set(b, a)
        });
        d = m.Ja(f.Sa(), Yea);
        c = m.da.FormData && a instanceof m.da.FormData;
        !m.La(Pga, b) || d || c || f.set("Content-Type", "application/x-www-form-urlencoded;charset=utf-8");
        f.forEach(function(a, b) {
            this.g.setRequestHeader(b, a)
        }, this);
        this.ja && (this.g.responseType = this.ja);
        "withCredentials"in this.g && (this.g.withCredentials = this.qa);
        try {
            tx(this), 0 < this.F && (this.S = Xea(this.g), Hk(sx(this, "Will abort after " + this.F + "ms if incomplete, xhr2 " + this.S)), this.S ? (this.g.timeout = this.F, this.g.ontimeout =
            (0, m.p)(this.nb, this)) : this.J = m.Cm(this.nb, this.F, this)), Hk(sx(this, "Sending request")), this.H=!0, this.g.send(a), this.H=!1
        } catch (h) {
            Hk(sx(this, "Send error: " + h.message)), nx(this, h)
        }
    };
    m.g.nb = function() {
        "undefined" != typeof m.rh && this.g && (this.D = "Timed out after " + this.F + "ms, aborting", this.C = 8, sx(this, this.D), this.dispatchEvent("timeout"), this.abort(8))
    };
    m.g.abort = function(a) {
        this.g && this.j && (sx(this, "Aborting"), this.j=!1, this.k=!0, this.g.abort(), this.k=!1, this.C = a || 7, this.dispatchEvent("complete"), this.dispatchEvent("abort"), px(this))
    };
    m.g.N = function() {
        this.g && (this.j && (this.j=!1, this.k=!0, this.g.abort(), this.k=!1), px(this, !0));
        m.mx.K.N.call(this)
    };
    m.g.al = function() {
        this.ha() || (this.X || this.H || this.k ? qx(this) : this.Xw())
    };
    m.g.Xw = function() {
        qx(this)
    };
    m.mx.prototype.isActive = function() {
        return !!this.g
    };
    m.mx.prototype.ve = function() {
        var a = this.getStatus(), b;
        t: switch (a) {
        case 200:
        case 201:
        case 202:
        case 204:
        case 206:
        case 304:
        case 1223:
            b=!0;
            break t;
        default:
            b=!1
        }
        if (!b) {
            if (a = 0 === a)
                a = m.hc(String(this.L))[1] || null, !a && window.self.location && (a = window.self.location.protocol, a = a.substr(0, a.length - 1)), a=!Oga.test(a ? a.toLowerCase() : "");
            b = a
        }
        return b
    };
    m.mx.prototype.getStatus = function() {
        try {
            return 2 < rx(this) ? this.g.status : - 1
        } catch (a) {
            return - 1
        }
    };
    m.g = vx.prototype;
    m.g.Rg = null;
    m.g.kg = null;
    m.g.tb = null;
    m.g.Na = null;
    m.g.bj = null;
    m.g.Ug = null;
    m.g.Zl = null;
    m.g.nh = null;
    m.g.Hf = 0;
    m.g.mu = 0;
    m.g.aa = null;
    m.g.wd = null;
    m.g.Wc = null;
    m.g.$d = null;
    m.g.md = null;
    m.g.mh = null;
    m.g.Be =- 1;
    m.g.pm =- 1;
    m.g.$b =- 1;
    m.g.If = 0;
    m.g.Ue = 0;
    m.g.Od = 8;
    var Ox = new m.ym;
    m.s(wx, m.$l);
    m.s(xx, m.$l);
    m.s(yx, m.$l);
    m.g = vx.prototype;
    m.g.connect = function(a, b, c, d, e) {
        Ww();
        this.bj = b;
        this.kg = c || {};
        d && m.ca(e) && (this.kg.OSID = d, this.kg.OAID = e);
        this.md = new hx(this);
        this.md.hj = this.Rg;
        this.md.j = this.F;
        this.md.connect(a)
    };
    m.g.disconnect = function() {
        zx(this);
        if (3 == this.g) {
            var a = this.Hf++, b = this.Ug.clone();
            m.Yj(b, "SID", this.C);
            m.Yj(b, "RID", a);
            m.Yj(b, "TYPE", "terminate");
            Gx(this, b);
            a = new Pw(this, 0, this.C, a, void 0);
            a.He = 2;
            a.jd = bk(b.clone());
            (new window.Image).src = a.jd;
            a.Lf = (0, m.H)();
            Tw(a)
        }
        Nx(this)
    };
    m.g.Dn = function() {
        return 0 == this.g
    };
    m.g.getState = function() {
        return this.g
    };
    m.g.Qm = function(a) {
        this.wd = null;
        Ex(this, a)
    };
    m.g.vo = function() {
        this.Wc = null;
        this.Na = new Pw(this, 0, this.C, "rpc", this.H);
        this.Na.Id = this.Rg;
        this.Na.cj = 0;
        var a = this.Zl.clone();
        m.Yj(a, "RID", "rpc");
        m.Yj(a, "SID", this.C);
        m.Yj(a, "CI", this.mh ? "0" : "1");
        m.Yj(a, "AID", this.Be);
        Gx(this, a);
        if (!m.E || m.Ib(10))
            m.Yj(a, "TYPE", "xmlhttp"), Sw(this.Na, a, !0, this.nh, !1);
        else {
            m.Yj(a, "TYPE", "html");
            var b = this.Na, c = Boolean(this.nh);
            b.He = 3;
            b.jd = bk(a.clone());
            ax(b, c)
        }
    };
    m.g.Am = function(a, b) {
        if (0 != this.g && (this.Na == a || this.tb == a))
            if (this.$b = a.Cf, this.tb == a && 3 == this.g)
                if (7 < this.Od) {
                    var c;
                    try {
                        c = this.F.parse(b)
                    } catch (d) {
                        c = null
                    }
                    if (m.ha(c) && 3 == c.length)
                        if (0 == c[0])
                            t: {
                                if (!this.Wc) {
                                    if (this.Na)
                                        if (this.Na.Lf + 3E3 < this.tb.Lf)
                                            Ax(this), this.Na.cancel(), this.Na = null;
                                        else 
                                            break t;
                                            Jx(this);
                                            Ww()
                                }
                            } else 
                                this.pm = c[1], 0 < this.pm - this.Be && 37500 > c[2] && this.mh && 0 == this.Ue&&!this.$d && (this.$d = cx((0, m.p)(this.St, this), 6E3));
            else 
                Lx(this, 11)
            } else 
                "y2f%" != b && Lx(this, 11);
        else if (this.Na == a && Ax(this),
        !m.rk(b)) {
            c = this.F.parse(b);
            m.ha(c);
            for (var e = 0; e < c.length; e++) {
                var f = c[e];
                this.Be = f[0];
                f = f[1];
                2 == this.g ? "c" == f[0] ? (this.C = f[1], this.nh = f[2], f = f[3], null != f ? this.Od = f : this.Od = 6, this.g = 3, this.aa && this.aa.Fe(this), this.Zl = jx(this, this.Jf() ? this.nh : null, this.bj), Ix(this)) : "stop" == f[0] && Lx(this, 7) : 3 == this.g && ("stop" == f[0] ? Lx(this, 7) : "noop" != f[0] && this.aa && this.aa.Ee(this, f), this.Ue = 0)
            }
        }
    };
    m.g.St = function() {
        null != this.$d && (this.$d = null, this.Na.cancel(), this.Na = null, Jx(this), Ww())
    };
    m.g.Qg = function(a) {
        var b;
        if (this.Na == a)
            Ax(this), this.Na = null, b = 2;
        else if (this.tb == a)
            this.tb = null, b = 1;
        else 
            return;
        this.$b = a.Cf;
        if (0 != this.g)
            if (a.dc)
                1 == b ? ((0, m.H)(), Ox.dispatchEvent(new xx(Ox, a.Kd ? a.Kd.length : 0)), Cx(this), this.D.length = 0) : Ix(this);
            else {
                var c = a.hd, d;
                if (!(d = 3 == c || 7 == c || 0 == c && 0 < this.$b)) {
                    if (d = 1 == b)
                        this.tb || this.wd || 1 == this.g || 2 <= this.If ? d=!1 : (this.wd = cx((0, m.p)(this.Qm, this, a), Kx(this, this.If)), this.If++, d=!0);
                        d=!(d || 2 == b && Jx(this))
                    }
                    if (d)
                        switch (c) {
                        case 1:
                            Lx(this, 5);
                            break;
                        case 4:
                            Lx(this,
                            10);
                            break;
                        case 3:
                            Lx(this, 6);
                            break;
                        case 7:
                            Lx(this, 12);
                            break;
                        default:
                            Lx(this, 2)
                        }
                    }
        };
    m.g.ez = function(a) {
        if (!m.La(arguments, this.g))
            throw Error("Unexpected channel state: " + this.g);
        };
    m.g.Zy = function(a) {
        a ? Ww() : (Ww(), Mx(this, 8))
    };
    m.g.mj = function(a) {
        if (a)
            throw Error("Can't create secondary domain capable XhrIo object.");
        a = new m.mx;
        a.qa=!1;
        return a
    };
    vx.prototype.isActive = function() {
        return !!this.aa && this.aa.isActive(this)
    };
    vx.prototype.k = function() {
        Ox.dispatchEvent(new yx(Ox))
    };
    vx.prototype.Jf = function() {
        return !(!m.E || m.Ib(10))
    };
    m.g = Px.prototype;
    m.g.Fe = function() {};
    m.g.Ee = function() {};
    m.g.Te = function() {};
    m.g.ed = function() {};
    m.g.Xo = function() {
        return {}
    };
    Px.prototype.isActive = function() {
        return !0
    };
    m.s(Qx, m.Am);
    Qx.prototype.D = 0;
    Qx.prototype.F = function() {
        if (500 < this.g) {
            var a = this.g;
            24E4 > 2 * a && (a*=2);
            Bm(this, a)
        }
        this.H()
    };
    Qx.prototype.start = function() {
        Qx.K.start.call(this);
        this.D = (0, m.H)() + this.g
    };
    Qx.prototype.stop = function() {
        this.D = 0;
        Qx.K.stop.call(this)
    };
    m.s(Rx, Px);
    m.g = Rx.prototype;
    m.g.subscribe = function(a, b, c) {
        return this.k.subscribe(a, b, c)
    };
    m.g.unsubscribe = function(a, b, c) {
        return this.k.unsubscribe(a, b, c)
    };
    m.g.Ib = function(a) {
        return this.k.Ib(a)
    };
    m.g.publish = function(a, b) {
        return this.k.publish.apply(this.k, arguments)
    };
    m.g.dispose = function() {
        this.S || (this.S=!0, this.k.clear(), this.disconnect(), m.Va(this.k))
    };
    m.g.ha = function() {
        return this.S
    };
    m.g.connect = function(a, b, c) {
        if (!this.g || 2 != this.g.getState()) {
            this.J = "";
            this.j.stop();
            this.F = a || null;
            this.D = b || 0;
            a = this.O + "/test";
            b = this.O + "/bind";
            var d = new vx("1", c ? c.firstTestResults : null, c ? c.secondTestResults : null), e = this.g;
            e && (e.aa = null);
            d.aa = this;
            this.g = d;
            e ? (3 != e.getState() && 0 == Dx(e) || e.getState(), this.g.connect(a, b, this.C, e.C, e.Be)) : c ? this.g.connect(a, b, this.C, c.sessionId, c.arrayId) : this.g.connect(a, b, this.C)
        }
    };
    m.g.disconnect = function(a) {
        this.L = a || 0;
        this.j.stop();
        this.g && (3 == this.g.getState() && Ex(this.g), this.g.disconnect());
        this.L = 0
    };
    m.g.sendMessage = function(a, b) {
        var c = {
            _sc: a
        };
        b && m.zb(c, b);
        this.j.enabled || 2 == (this.g ? this.g.getState() : 0) ? this.H.push(c) : this.g && 3 == this.g.getState() && Bx(this.g, c)
    };
    m.g.Fe = function() {
        var a = this.j;
        a.stop();
        Bm(a, 5E3 + 2E4 * Math.random());
        this.F = null;
        this.D = 0;
        if (this.H.length) {
            a = this.H;
            this.H = [];
            for (var b = 0, c = a.length; b < c; ++b)
                Bx(this.g, a[b])
        }
        this.publish("handlerOpened")
    };
    m.g.Te = function(a, b) {
        var c = 2 == b && 401 == this.g.$b;
        if (4 != b&&!c) {
            if (6 == b || 410 == this.g.$b)
                c = this.j, c.stop(), Bm(c, 500);
            this.j.start()
        }
        this.publish("handlerError", b)
    };
    m.g.ed = function(a, b, c) {
        if (!this.j.enabled)
            this.publish("handlerClosed");
        else if (c)
            for (a = 0, b = c.length; a < b; ++a)
                this.H.push(c[a].map)
    };
    m.g.Xo = function() {
        var a = {
            v: 2
        };
        this.J && (a.gsessionid = this.J);
        0 != this.D && (a.ui = "" + this.D);
        0 != this.L && (a.ui = "" + this.L);
        this.F && m.zb(a, this.F);
        return a
    };
    m.g.Ee = function(a, b) {
        if ("S" == b[0])
            this.J = b[1];
        else if ("gracefulReconnect" == b[0]) {
            var c = this.j;
            c.stop();
            Bm(c, 500);
            this.j.start();
            this.g.disconnect()
        } else 
            this.publish("handlerMessage", new Qea(b[0], b[1]))
    };
    m.g.getDeviceId = function() {
        return this.C.id
    };
    m.g.vq = function() {
        this.j.stop();
        0 != Dx(this.g) ? this.j.start() : this.connect(this.F, this.D)
    };
    m.s(Sx, Au);
    m.g = Sx.prototype;
    m.g.eg = window.NaN;
    m.g.lk=!1;
    m.g.Zg = window.NaN;
    m.g.mk = window.NaN;
    m.g.Mg = window.NaN;
    m.g.connect = function(a, b) {
        if (b) {
            if (this.$.j) {
                var c = b.listId, d = b.videoId, e = b.index, f = b.currentTime || 0;
                5 >= f && (f = 0);
                h = {
                    videoId: d,
                    currentTime: f
                };
                c && (h.listId = c);
                m.ca(e) && (h.currentIndex = e);
                c && (this.$.j = c);
                this.$.videoId = d;
                this.$.index = e || 0
            } else {
                var d = b.videoIds[b.index], f = b.currentTime || 0;
                5 >= f && (f = 0);
                var h = {
                    videoIds: d,
                    videoId: d,
                    currentTime: f
                };
                this.$.videoIds = [d];
                this.$.index = 0
            }
            this.$.state = 3;
            fw(this.$, f);
            this.ma("Connecting with setPlaylist and params: " + m.ag(h));
            this.aa.connect({
                method: "setPlaylist",
                params: m.ag(h)
            }, a, Nu())
        } else 
            this.ma("Connecting without params"), this.aa.connect({}, a, Nu());
        cfa(this)
    };
    m.g.dispose = function() {
        this.ha() || (this.publish("beforeDispose"), Tx(this, 3));
        Sx.K.dispose.call(this)
    };
    m.g.N = function() {
        Ux(this);
        Xx(this);
        Vx(this);
        m.bb(this.Mg);
        this.Mg = window.NaN;
        this.k = null;
        m.N(this.Aa);
        this.Aa.length = 0;
        this.aa.dispose();
        Sx.K.N.call(this);
        this.g = this.$ = this.aa = null
    };
    m.g.ma = function(a) {
        $t("conn", a)
    };
    m.g.et = function() {
        this.D(2)
    };
    m.g.kt = function() {
        this.ma("Channel opened");
        this.lk && (this.lk=!1, Vx(this), this.Zg = m.v((0, m.p)(function() {
            this.ma("Timing out waiting for a screen.");
            this.D(1)
        }, this), 15E3));
        Eea($ea(this.aa), this.L)
    };
    m.g.ft = function() {
        this.ma("Channel closed");
        (0, window.isNaN)(this.eg) ? Xu(!0) : Xu();
        this.dispose()
    };
    m.g.gt = function(a) {
        Xu();
        (0, window.isNaN)(this.F()) ? (this.ma("Channel error: " + a + " without reconnection"), this.dispose()) : (this.lk=!0, this.ma("Channel error: " + a + " with reconnection in " + this.F() + " ms"), Tx(this, 2))
    };
    m.g.ht = function(a) {
        a.params ? this.ma("Received: action=" + a.action + ", params=" + m.ag(a.params)) : this.ma("Received: action=" + a.action + " {}");
        switch (a.action) {
        case "loungeStatus":
            a = m.$f(a.params.devices);
            this.g = (0, m.sh)(a, function(a) {
                return new Gu(a)
            });
            a=!!m.Ja(this.g, function(a) {
                return "LOUNGE_SCREEN" == a.type
            });
            Zx(this, a);
            break;
        case "loungeScreenConnected":
            Zx(this, !0);
            break;
        case "loungeScreenDisconnected":
            Ol(this.g, function(a) {
                return "LOUNGE_SCREEN" == a.type
            });
            Zx(this, !1);
            break;
        case "remoteConnected":
            var b =
            new Gu(m.$f(a.params.device));
            m.Ja(this.g, function(a) {
                return a.equals(b)
            }) || m.Pl(this.g, b);
            break;
        case "remoteDisconnected":
            b = new Gu(m.$f(a.params.device));
            Ol(this.g, function(a) {
                return a.equals(b)
            });
            break;
        case "gracefulDisconnect":
            break;
        case "playlistModified":
            $x(this, a);
            break;
        case "nowPlaying":
            efa(this, a);
            break;
        case "onStateChange":
            ay(this, a);
            break;
        case "onVolumeChanged":
            ffa(this, a);
            break;
        case "onSubtitlesTrackChanged":
            dfa(this, a);
            break;
        default:
            this.ma("Unrecognized action: " + a.action)
        }
    };
    m.g.jt = function(a) {
        a.params ? this.ma("Received: action=" + a.action + ", params=" + m.ag(a.params)) : this.ma("Received: action=" + a.action);
        gfa(this, a);
        hfa(this, a);
        if (Yx(this)) {
            var b = this.$.clone(), c=!1, d, e, f, h, k, r, w;
            a.params && (d = a.params.videoId || a.params.video_id, e = a.params.videoIds || a.params.video_ids, f = a.params.state, h = a.params.currentTime || a.params.current_time, k = a.params.volume, r = a.params.muted, m.ca(a.params.currentError) && (w = m.$f(a.params.currentError)));
            if ("onSubtitlesTrackChanged" == a.action)
                d == ew(this.$) &&
                (delete a.params.videoId, m.kk(a.params) ? this.$.F = null : this.$.F = a.params, this.publish("remotePlayerChange"));
            else if (ew(this.$) || "onStateChange" != a.action)
                "playlistModified" != a.action && "nowPlayingPlaylist" != a.action || e ? (d || "nowPlaying" != a.action && "nowPlayingPlaylist" != a.action ? d || (d = ew(this.$)) : this.$.setVideoId(""), e && (e = e.split(","), hw(this.$, e, d))) : hw(this.$, []), this.$.add(d) && this.j("getPlaylist"), d && this.$.setVideoId(d), b.index == this.$.index && m.Ml(b.videoIds, this.$.videoIds) || this.publish("remoteQueueChange"),
                m.ca(f) && (b = (0, window.parseInt)(f, 10), b = (0, window.isNaN)(b)?-1 : b, - 1 == b&&-1E3 == this.$.g && (b =- 1E3), 0 == b && "0" == h && (b =- 1), c = c || b != this.$.g, this.$.g = b, d = null, - 1E3 == b && (d = this.$.D || "unknown", w && (d = w.reason || d)), c = c || this.$.D != d, this.$.D = d, 1 == this.$.g ? Wx(this) : Xx(this)), "onError" != a.action||-1 != this.$.g&&-1E3 != this.$.g || (a = m.$f(a.params.errors) || [], 1 == a.length && "PLAYER_ERROR" == a[0].error && a[0].videoId == ew(this.$) && (this.$.g =- 1E3, this.$.D = a[0].reason || "unknown", c=!0)), h && (b = (0, window.parseInt)(h, 10), fw(this.$,
            (0, window.isNaN)(b) ? 0 : b), c=!0), m.ca(k) && (b = (0, window.parseInt)(k, 10), (0, window.isNaN)(b) || (c = c || this.$.C != b, this.$.C = b), m.ca(r) && (r = "true" == r, c = c || this.$.k != r, this.$.k = r)), c && this.publish("remotePlayerChange")
        }
    };
    m.g.cr = function() {
        if (this.k) {
            var a = this.k;
            this.k = null;
            this.$.videoId != a && this.j("getNowPlaying")
        }
    };
    Sx.prototype.subscribe = Sx.prototype.subscribe;
    Sx.prototype.unsubscribeByKey = Sx.prototype.Ib;
    Sx.prototype.O = function() {
        var a = 3;
        this.ha() || (a = 0, (0, window.isNaN)(this.F()) ? Yx(this) && (a = 1) : a = 2);
        return a
    };
    Sx.prototype.getProxyState = Sx.prototype.O;
    Sx.prototype.D = function(a) {
        this.ma("Disconnecting with " + a);
        Ux(this);
        this.publish("beforeDisconnect", a);
        1 == a && Xu();
        this.aa.disconnect(a);
        this.dispose()
    };
    Sx.prototype.disconnect = Sx.prototype.D;
    Sx.prototype.S = function() {
        var a = this.$;
        if (this.k) {
            var b = a = this.$.clone(), c = this.k, d = a.index, e = b.videoId;
            b.videoId = c;
            b.index = d;
            c != e && cw(b)
        }
        return iw(a)
    };
    Sx.prototype.getPlayerContextData = Sx.prototype.S;
    Sx.prototype.T = function(a) {
        var b = new $v(a);
        b.videoId && b.videoId != this.$.videoId && (this.k = b.videoId, m.bb(this.Mg), this.Mg = m.v((0, m.p)(this.cr, this), 5E3));
        var c = [];
        this.$.j == b.j && this.$.videoId == b.videoId && this.$.index == b.index && m.Ml(this.$.videoIds, b.videoIds) || c.push("remoteQueueChange");
        this.$.g == b.g && this.$.C == b.C && this.$.k == b.k && gw(this.$) == gw(b) && m.ag(this.$.F) == m.ag(b.F) || c.push("remotePlayerChange");
        aw(this.$, a);
        (0, m.y)(c, function(a) {
            this.publish(a)
        }, this)
    };
    Sx.prototype.setPlayerContextData = Sx.prototype.T;
    Sx.prototype.J = function() {
        return this.aa.C.loungeIdToken
    };
    Sx.prototype.getLoungeToken = Sx.prototype.J;
    Sx.prototype.H = function() {
        var a = this.aa.getDeviceId(), b = m.Ja(this.g, function(b) {
            return "REMOTE_CONTROL" == b.type && b.id != a
        });
        return b ? b.id : ""
    };
    Sx.prototype.getOtherConnectedRemoteId = Sx.prototype.H;
    Sx.prototype.F = function() {
        var a = this.aa;
        return a.j.enabled ? a.j.D - (0, m.H)() : window.NaN
    };
    Sx.prototype.getReconnectTimeout = Sx.prototype.F;
    Sx.prototype.X = function() {
        if (!(0, window.isNaN)(this.F())) {
            var a = this.aa.j;
            a.enabled && (a.stop(), a.start(), a.F())
        }
    };
    Sx.prototype.reconnect = Sx.prototype.X;
    Sx.prototype.j = function(a, b) {
        b ? this.ma("Sending: action=" + a + ", params=" + m.ag(b)) : this.ma("Sending: action=" + a);
        this.aa.sendMessage(a, b)
    };
    Sx.prototype.sendMessage = Sx.prototype.j;
    m.s(by, Au);
    var oy = [2E3, 2E3, 1E3, 1E3, 1E3, 2E3, 2E3, 5E3, 5E3, 1E4];
    m.g = by.prototype;
    m.g.gg = window.NaN;
    m.g.Mh = "";
    m.g.ma = function(a) {
        $t("RM", a)
    };
    m.g.Ea = function(a) {
        $t("RM", a)
    };
    m.g.publish = function(a, b) {
        if (this.ha())
            return !1;
        this.ma("Firing " + a);
        return this.C.publish.apply(this.C, arguments)
    };
    m.g.Wh = function() {
        var a = Ew() || [];
        m.Rl(a) || ey(this, a);
        a = hy(this);
        m.Rl(a) || ((0, m.uh)(a, function(a) {
            return !yu(this.ld, a)
        }, this) && Su() ? kfa(this) : jy(this, a))
    };
    m.g.ay = function(a) {
        var b = xu(this.j, a.id);
        b && (this.ma("Updating DIAL device: " + b.id + "(" + b.name + ") from status: " + b.status + " to status: " + a.status + " and from activityId: " + b.activityId + " to activityId: " + a.activityId), b.activityId = a.activityId, b.status = a.status, Vu(vu(this.j)));
        fy(this, this.g)
    };
    m.g.ho = function(a, b) {
        for (var c = hy(this), c = (0, m.sh)(c, function(a) {
            return a.name
        }), d = a, e = 2; m.La(c, d);)
            d = b.call(m.da, e), e++;
        return d
    };
    m.g.Rm = function(a, b, c) {
        var d=!1;
        b >= oy.length && (this.ma("Pairing DIAL device " + a + " with " + c + " timed out."), d=!0);
        var e = xu(this.j, a);
        if (!e)
            this.ma("Pairing DIAL device " + a + " with " + c + " failed: no device for " + a), d=!0;
        else if ("ERROR" == e.status || "STOPPED" == e.status)
            this.ma("Pairing DIAL device " + a + " with " + c + " failed: launch error on " + a), d=!0;
        d ? (ny(this), this.publish("screenPair", null)) : m.R(qu(this.k, "/pairing/get_screen"), {
            method: "POST",
            V: {
                pairing_code: c
            },
            context: this,
            R: function(a, b) {
                if (c == this.Mh) {
                    ny(this);
                    var d = new du(b.screen);
                    d.name = e.name;
                    d.uuid = e.id;
                    this.ma("Pairing " + c + " succeeded.");
                    var r = mfa(this, this.ec, d);
                    this.ma("Paired with " + (r ? "a new" : "an old") + " local screen:" + ju(d));
                    iy(this);
                    this.publish("screenPair", d)
                }
            },
            onError: function() {
                c == this.Mh && (this.ma("Polling pairing code: " + c), m.bb(this.gg), this.gg = m.v((0, m.p)(this.Rm, this, a, b + 1, c), oy[b]))
            }
        })
    };
    m.g.lw = function() {
        var a = qu(this.k, "/screens");
        m.R(a, {
            method: "GET",
            context: this,
            withCredentials: !0,
            R: function(a, c) {
                this.ld = (0, m.sh)(m.Ki(c), function(a) {
                    return new du(a)
                });
                this.ma("Load account screens successfully.");
                this.Wh();
                gy(this);
                this.publish("accountScreenLoad")
            },
            onError: function() {
                this.Ea("Load account screens failed.");
                qy(this)
            }
        })
    };
    m.s(ry, Bu);
    m.g = ry.prototype;
    m.g.ab = function(a) {
        return this.hb.$_gs(a)
    };
    m.g.contains = function(a) {
        return !!this.hb.$_c(a)
    };
    m.g.get = function(a) {
        return this.hb.$_g(a)
    };
    m.g.start = function(a) {
        this.hb.$_st(a)
    };
    m.g.add = function(a, b, c) {
        this.hb.$_a(a, b, c)
    };
    m.g.remove = function(a, b, c) {
        this.hb.$_r(a, b, c)
    };
    m.g.sd = function(a, b, c, d) {
        this.hb.$_un(a, b, c, d)
    };
    m.g.N = function() {
        for (var a = 0, b = this.g.length; a < b; ++a)
            this.hb.$_ubk(this.g[a]);
        this.g.length = 0;
        this.hb = null;
        ry.K.N.call(this)
    };
    m.g.aw = function() {
        this.publish("screenChange")
    };
    m.g.$v = function() {
        this.publish("onlineScreenChange")
    };
    nv.prototype.$_st = nv.prototype.start;
    nv.prototype.$_gspc = nv.prototype.uA;
    nv.prototype.$_c = nv.prototype.contains;
    nv.prototype.$_g = nv.prototype.get;
    nv.prototype.$_a = nv.prototype.add;
    nv.prototype.$_un = nv.prototype.sd;
    nv.prototype.$_r = nv.prototype.remove;
    nv.prototype.$_gs = nv.prototype.ab;
    nv.prototype.$_gos = nv.prototype.xm;
    nv.prototype.$_s = nv.prototype.subscribe;
    nv.prototype.$_ubk = nv.prototype.Ib;
    var ty = null, uy = null, Fy = null, my = null, Ay = [];
    var Zy, Wy = [];
    var gz;
    var yfa = {
        "play-next": function(a) {
            var b = m.I(a, "list-id");
            a = m.I(a, "video-ids");
            b ? m.z("watch-queue-addto-playlist-play-next", b, a) : m.z("watch-queue-addto-video-play-next", a)
        },
        "play-now": function(a) {
            var b = m.I(a, "list-id");
            a = m.I(a, "video-ids");
            b ? m.z("watch-queue-addto-playlist-play-now", b, a) : m.z("watch-queue-addto-video-play-now", a)
        }
    }, SE = [], pz = [];
    var yz, TE = m.E && 8 <= window.document.documentMode || m.Re && m.Fb("1.9.2") || m.Qe && m.Fb("532.1"), rz = m.E&&!TE;
    qz.prototype.setEnabled = function(a, b) {
        this.C && (m.N(this.C), delete this.C);
        this.F && (m.cb(this.F), delete this.F);
        if (a) {
            this.g = sz(this);
            if (rz) {
                var c = this.D.contentWindow.document.body;
                c && c.innerHTML || uz(this, this.g)
            }
            b || this.j(this.g);
            TE ? this.C = m.J(this.L, "hashchange", this.H) : this.F = m.ab(this.H, 200)
        }
    };
    qz.prototype.S = function() {
        if (rz) {
            var a;
            a = (a = this.D.contentWindow.document.body) ? m.Mi(a.innerHTML.substring(1)) : "";
            a != this.g ? (this.g = a, tz(this, a), this.j(a)) : (a = sz(this), a != this.g && (this.g = a, uz(this, a), this.j(a)))
        } else 
            a = sz(this), a != this.g && (this.g = a, this.j(a))
    };
    qz.prototype.add = function(a, b, c) {
        this.g = "" + a;
        rz && uz(this, a);
        tz(this, a);
        c || this.j(this.g)
    };
    var Hz=!!window.history.pushState && (!m.Qe || m.Qe && m.Fb("534.11"));
    vz.prototype.setEnabled = function(a, b) {
        this.D && (m.N(this.D), delete this.D);
        this.F && (m.cb(this.F), delete this.F);
        a && Hz && (this.g = this.j.href, b || this.k(this.g), this.D = m.J(this.C, "popstate", this.H))
    };
    vz.prototype.L = function(a) {
        var b = this.j.href;
        if ((a = a.state) || b != this.g)
            this.g = b, this.k(b, a)
    };
    vz.prototype.add = function(a, b, c) {
        if (a || b)
            a = a || this.j.href, this.C.history.pushState(b, "", a), this.g = a, c || this.k(a, b)
    };
    vz.prototype.replace = function(a, b, c) {
        if (a || b)
            a = a || this.j.href, this.C.history.replaceState(b, "", a), this.g = a, c || this.k(a, b)
    };
    var Qga = {
        YC: "ypc_ii",
        ZC: "ypc_it",
        Nz: "ypc_ft"
    }, Rga = {
        CC: "ypc_grc",
        Nz: "ypc_ft"
    }, UE = [], VE = [], Ez = [], Cz=!1;
    var Kz = {}, Jz = 0;
    var Nz, Oz, Pz, Mz, Ifa = {
        INPUT: !0,
        SELECT: !0,
        TEXTAREA: !0
    };
    var Rz, Tz, Vz, Zz, $z, aA, Xz, Sz, Uz, Yz, bA, WE = [];
    var dA = [];
    var XE = [];
    m.l("yt.www.errors.log", Iz, void 0);
    m.l("yt.style.show", m.K, void 0);
    m.l("yt.style.hide", m.P, void 0);
    m.l("yt.style.toggle", m.wk, void 0);
    m.l("yt.net.ping.send", m.sc, void 0);
    m.l("yt.tracking.track", m.al, void 0);
    m.l("yt.tracking.share", $k, void 0);
    m.l("yt.uix.FormInput.selectOnChangeActionIE", function(a) {
        m.tp.getInstance().Jb(a)
    }, void 0);
    m.l("yt.window.popup", Jk, void 0);
    m.l("yt.window.navigate", m.be, void 0);
    m.l("yt.www.ypc.delayedloader.loadOffers", zz, void 0);
    m.l("yt.www.account.CreateChannelLoader.show", function(a) {
        fs(!0, a)
    }, void 0);
    m.l("yt.www.account.FeedPrivacyDialog.init", js, void 0);
    m.l("yt.www.account.LinkGplusLoader.init", ps, void 0);
    m.l("yt.www.account.LinkGplusLoader.show", function(a) {
        ps();
        rs();
        us(a)
    }, void 0);
    m.l("yt.www.feedback.start", m.Af, void 0);
    m.l("yt.www.feedback.startHelp", m.Bf, void 0);
    m.l("yt.www.feedback.showArticle", m.Df, void 0);
    m.s(fA, m.Uc);
    fA.prototype.enable = function() {
        fA.K.enable.call(this);
        for (var a = fA.g.length, b = 0; b < a; b++)
            m.$q(fA.g[b]);
        this.subscribe("init", Pca);
        this.subscribe("init", this.init, this);
        this.subscribe("dispose", this.dispose, this);
        m.Vc(this)
    };
    fA.prototype.init = function() {
        fA.K.init.call(this);
        Gfa();
        BE.push(m.J(window, "resize", Gba));
        var a = new Dm(Iba, 200);
        BE.push(m.J(window, "scroll", (0, m.p)(a.pn, a)));
        m.C(window.document.body, "page-loaded");
        var a = m.Ef.getInstance(), b = 1 < window.devicePixelRatio;
        m.Jf(0, 119) != b && (m.ji(119, b), a.save());
        if (a = m.G("first-focus"))
            nl(a) || (a.tabIndex = "0"), nl(a) && a.focus();
        Fs("keyboard");
        Pz = [m.J(window.document.body, "keypress", Hfa)];
        an || (Ym = {}, Zm = {}, Qm = {}, Xm = {}, CE = [], Lm = Mm(), CE.push(m.x("page-resize", Jba)), CE.push(m.x("page-scroll",
        Kba)), CE.push(m.x("yt-dom-content-change", Nm)), an=!0);
        m.Oc() ? this.g.push(m.Qc(cA, void 0)) : cA();
        m.vp();
        a = fA.g.length;
        for (b = 0; b < a; b++) {
            var c = Lo(fA.g[b].getInstance());
            c in ar && m.ob("yt-uix-init-" + c)
        }
        Lp();
        m.pg && ii();
        m.u("CREATE_CHANNEL_LIGHTBOX") && fs();
        m.u("FEED_PRIVACY_LIGHTBOX_ENABLED") && js();
        m.u("SHOW_IDENTITY_PROMPT_LIGHTBOX") && (a = bs(os), b = m.u("IDENTITY_PROMPT_PAGEID"), a.open("identity-prompt", "/identity_prompt_ajax", "identity_prompt_ajax", !0, !0, void 0, m.u("IDENTITY_PROMPT_NEXT_URL", window.document.location.href),
        !0, b ? {
            pageid: b
        } : void 0));
        (m.u("SHOW_LINK_GPLUS_LIGHTBOX") || m.u("LINK_GPLUS_LIGHTBOX_ENABLED")) && ps();
        "watch" != m.u("PAGE_NAME") && m.Ds();
        yE.push(m.x("player-ad-start", fba));
        zE.push(m.O(window.document.body, "click", gba, "yt-google-help-link"));
        qba();
        a=!!m.G("guide-module-loading");
        window.spf && window.spf.load && a && (m.u("GUIDE_DELAY_LOAD") || Ps(), Ns.push(m.x("appbar-show-guide", Ps)));
        (a = m.F("watch-timeline-mole-container")) && 0 == m.wi(a).length && "watch" != m.u("PAGE_NAME") && Qs();
        Ns.push(m.x("watch-timeline-refresh",
        Qs));
        Ks();
        Fo("addto-watch-later-button", "click", Ws);
        Fo("addto-watch-later-button-success", "click", Zs);
        Fo("addto-watch-later-button-remove", "click", $s);
        Fo("addto-watch-later-button-sign-in", "click", Us);
        KE.push(m.O(m.F("shared-addto-watch-later-login"), "click", Pda, "sign-in-link"));
        m.v(Qda, 1E3);
        m.u("WATCH_LATER_VIDEO_ID_TO_LOAD") && er({
            videoIds: m.u("WATCH_LATER_VIDEO_ID_TO_LOAD")
        });
        m.Vy();
        Fo("addto-watch-queue-button", "click", iz);
        Fo("addto-tv-queue-button", "click", iz);
        Fo("addto-watch-queue-button-success",
        "click", mz);
        Fo("addto-watch-queue-menu-choice", "click", lz);
        SE.push(m.x("watch-queue-update", oz));
        oz();
        (window.ytspf || {}).enabled ? (Rz=!!m.u("PREFETCH_LINKS"), Tz = 0, Vz = m.u("PREFETCH_LINKS_MAX") || 0, Zz=!!m.u("PREBUFFER_LINKS"), $z = 0, aA = m.u("PREBUFFER_MAX") || 0) : Zz = Rz=!1;
        Xz=!!window.spf&&!(!m.u("PREFETCH_CSS_RESOURCES")&&!m.u("PREFETCH_JS_RESOURCES"));
        "watch" != m.u("PAGE_NAME") && (m.Qz(), Jfa());
        Ht=!1;
        PE.push(Fq(rE, Gt), Fq(sE, Lt));
        Ht || (PE.push(Fq(It, Gt), Fq(Mt, Lt)), OE.push(m.x("subscription-batch-subscribe", Nt)),
        OE.push(m.x("subscription-batch-unsubscribe", Pt)), OE.push(m.x("subscription-pref-email", kea)), OE.push(m.x("subscription-pref-uploads", lea)), OE.push(m.x("subscription-prefs", oea)), OE.push(m.x("subscription-batch-pref-email", mea)), OE.push(m.x("subscription-batch-pref-uploads", nea)), PE.push(Fq(Mga, pea), Fq(Nga, rea), Fq(Lga, qea)), FE.push(m.x("subscription-subscribe-success", m.q(m.xo, "subscribe")), m.x("subscription-unsubscribe-success", m.q(m.xo, "unsubscribe")), m.x("player-subscribe", m.q(m.xo, "subscribe")),
        m.x("player-unsubscribe", m.q(m.xo, "unsubscribe"))));
        ME.push(m.x("subscription-show-pref-overlay", fea));
        LE.push(m.x("live-event-reminder-set-request", Rda));
        LE.push(m.x("live-event-reminder-remove-request", Sda));
        m.u("RESOLUTION_TRACKING_ENABLED") && (a = "CSS1Compat" == window.document.compatMode ? window.document.documentElement : window.document.body, a = {
            a: "resolution",
            width: window.screen.width,
            height: window.screen.height,
            depth: window.screen.colorDepth,
            win_width: a.clientWidth,
            win_height: a.clientHeight
        }, window.devicePixelRatio &&
        (a.pixel_ratio = window.devicePixelRatio), m.Bd(m.qc(a)));
        m.u("MEMORY_TRACKING_ENABLED") && window.performance && window.performance.memory && (a = window.performance.memory, b = window._spf_state, m.Bd(m.qc({
            a: "memory",
            navCounter: b ? b["nav-counter"]: - 1,
            heapLimit: a.jsHeapSizeLimit,
            usedHeapSize: a.usedJSHeapSize,
            totalHeapSize: a.totalJSHeapSize
        })));
        m.u("NAVIGATION_TRACKING_ENABLED") && xba();
        m.u("ADBLOCK_TRACKING_ENABLED") && "undefined" != typeof window.History && m.Bd(m.qc({
            a: "adblock",
            active: void 0 == window.History.prototype.pushState &&
            void 0 != window.History.prototype.replaceState
        }));
        a = m.q(m.yo, "add_to_playlist");
        EE.push(m.x("WATCH_LATER_VIDEO_ADDED", a), m.x("playlist-favorite", a), m.x("playlist-addto", a));
        m.A(window.document.body, "visibility-logging-enabled") && (Pr=!1, m.Kr(), HE = m.v(tda, "watch" == m.u("PAGE_NAME") ? 3E3 : 1E3));
        for (a = 0; a < window.document.forms.length; a++) {
            c=!1;
            for (b = 0; b < XE.length; b++)
                window.document.forms[a].name == XE[b] && (c=!0);
            b = window.document.forms[a];
            if ("post" == b.method.toLowerCase() && 0 == c) {
                for (var c=!1, d = 0; d < b.elements.length; d++)
                    b.elements[d].name ==
                    m.u("XSRF_FIELD_NAME") && (c=!0);
                c || (c = void 0, c = m.u("XSRF_TOKEN"), d = window.document.createElement("input"), d.setAttribute("name", m.u("XSRF_FIELD_NAME")), d.setAttribute("type", "hidden"), d.setAttribute("value", c), b.appendChild(d))
            }
        }
        m.u("YPC_LOADER_ENABLED") && (b = m.cl(window.location.href), c = b.ypc_it, d = b.ypc_ii, a = b.ypc_ft || "D", "channel" == m.u("PAGE_NAME") && "fan_fund"in b && (a = "T", c = "U", d = m.u("CHANNEL_ID")), "RG" == a ? (b = b.ypc_grc, Gz(Rga), Bfa(b, a)) : d && c && (Gz(Qga), zz(c, d, a)), m.G("ypc-delayedloader-target") && Az(),
        VE.push(Fq(Jt, Ffa)), VE.push(Fq(m.NE, Efa), Fq(Kt, Dfa)), UE.push(m.O(window.document.documentElement, "click", Fz, "ypc-checkout-button"), m.O(window.document.documentElement, "click", Fz, "ypc-signin-button")));
        m.u("SITEPUBSUB_ENABLED") && (zca(), Bca(), lo("init"));
        Ts.getInstance().init();
        if (b = m.u("GUIDED_HELP_PARAMS"))
            a = b.flowId, b = b.cookieName, a && b && (m.Wn(a, b), m.Un());
        try {
            if (window.ytbuffer && window.ytbuffer.queue) {
                for (var e = window.ytbuffer.queue, a = 0; 5 > a && a < e.length; a++) {
                    var f = e[a], h = f.target || f.srcElement;
                    m.Bd(m.qc({
                        a: "buffer_events",
                        event_index: a,
                        clientX: f.clientX,
                        clientY: f.clientY,
                        localName: h.localName,
                        className: h.className,
                        id: h.id,
                        page_name: m.u("PAGE_NAME")
                    }))
                }
                window.document.removeEventListener ? window.document.removeEventListener("click", window.ytbuffer.enqueueEvent, !1) : window.document.detachEvent("onclick", window.ytbuffer.enqueueEvent);
                m.mk("ytbuffer")
            }
        } catch (k) {}
        Uba()
    };
    fA.prototype.dispose = function() {
        fA.K.dispose.call(this);
        m.Tc(this.g);
        this.g.length = 0;
        m.D(window.document.body, "page-loaded");
        m.bb(Jm);
        Jm = 0;
        m.N(BE);
        BE.length = 0;
        Im = Gm = null;
        m.Sc(Km);
        Km = 0;
        m.N(Is);
        Is.length = 0;
        Hs = null;
        for (var a = dA, b = a.length; - 1 < b; b--)
            m.n("yt.dom.VisibilityMonitor.unlistenByKey")(a[b]);
        dA.length = 0;
        Nz && Nz.dispose();
        Oz = Nz = null;
        Pz && (m.N(Pz), Pz.length = 0);
        Mz = null;
        an && (m.nb(CE), Mba(), Ym = {}, Zm = {}, Qm = {}, Xm = {}, CE.length = 0, Lm = null, an=!1);
        Sp = 0;
        m.nb(Mp);
        Mp.length = 0;
        Tp.length = 0;
        Pp = {};
        m.pg && ii();
        a = fA.g.length;
        for (b = 0; b < a; b++)
            cr(fA.g[b]);
        ds(es);
        m.nb(ks);
        ks = [];
        m.N(IE);
        IE = [];
        ds(os);
        m.nb(ss);
        ss.length = 0;
        m.N(qs);
        qs.length = 0;
        ys=!1;
        m.bb(Es);
        m.nb(yE);
        m.N(zE);
        yE.length = 0;
        zE.length = 0;
        m.yf = {};
        Nda();
        Ho("addto-watch-later-button", "click", Ws);
        Ho("addto-watch-later-button-success", "click", Zs);
        Ho("addto-watch-later-button-remove", "click", $s);
        Ho("addto-watch-later-button-sign-in", "click", Us);
        m.N(KE);
        KE = [];
        m.nb(SE);
        SE.length = 0;
        pz = [];
        Ho("addto-watch-queue-button", "click", iz);
        Ho("addto-tv-queue-button", "click", iz);
        Ho("addto-watch-queue-button-success",
        "click", mz);
        Ho("addto-watch-queue-menu-choice", "click", lz);
        m.Yy();
        m.nb(EE);
        EE = [];
        Sz = Xz = Zz = Rz=!1;
        aA = $z = Vz = Tz = 0;
        Uz = null;
        m.bb(void 0);
        m.bb(Yz);
        m.N(WE);
        WE.length = 0;
        bA && bA();
        m.nb(LE);
        LE.length = 0;
        m.nb(OE);
        OE.length = 0;
        Jq(PE);
        PE.length = 0;
        Ht=!1;
        m.nb(FE);
        FE.length = 0;
        m.nb(ME);
        ME.length = 0;
        m.N(zt);
        zt.length = 0;
        wt && (m.ri(wt), wt = null);
        m.A(window.document.body, "visibility-logging-enabled") && (m.bb(Qr), m.bb(HE), Sr && m.nb(Sr), Pr=!1);
        Jq(VE);
        VE = [];
        Ez = [];
        m.N(UE);
        UE = [];
        m.u("SITEPUBSUB_ENABLED") && (m.nb(m.Ki(ko)), ko = {}, io.getInstance().dispose(),
        lo("dispose"));
        Ts.getInstance().dispose();
        m.vn()
    };
    fA.prototype.disable = function() {
        fA.K.disable.call(this);
        for (var a = fA.g.length, b = 0; b < a; b++)
            m.br(fA.g[b])
    };
    fA.g = [m.Oo, fp, m.np, op, pp, m.tp, m.Cp, Ep, Jp, m.Up, m.Vp, m.kq, oq, pq, qq, m.rq, tq, m.vq, m.Rq, Wq, m.Xq, Zq, m.Mq, Jr];
    m.Wc(new fA);
    m.gA.prototype.getValue = function() {
        return this.g
    };
    m.g = m.hA.prototype;
    m.g.Uh = null;
    m.g.wl = "JSON";
    m.g.start = function() {
        m.iA(this)
    };
    m.g.stop = function() {
        m.jA(this)
    };
    m.g.Xk = function() {
        var a = {};
        this.j && (a.session_token = this.j);
        m.R(this.k(), {
            format: this.wl,
            method: "POST",
            V: a,
            ka: (0, m.p)(function(a) {
                this.C(a)
            }, this)
        })
    };
    m.s(m.kA, m.t);
    m.g = m.kA.prototype;
    m.g.ta = null;
    m.g.Fj=!1;
    m.g.start = function() {
        this.stop();
        this.Fj=!1;
        var a = lA(this), b = mA(this);
        a&&!b && this.g.mozRequestAnimationFrame ? (this.ta = m.im(this.g, "MozBeforePaint", this.Xa), this.g.mozRequestAnimationFrame(null), this.Fj=!0) : this.ta = a && b ? a.call(this.g, this.Xa) : this.g.setTimeout(m.ll(this.Xa), 20)
    };
    m.g.stop = function() {
        if (this.isActive()) {
            var a = lA(this), b = mA(this);
            a&&!b && this.g.mozRequestAnimationFrame ? m.sm(this.ta) : a && b ? b.call(this.g, this.ta) : this.g.clearTimeout(this.ta)
        }
        this.ta = null
    };
    m.g.Jc = m.aa(19);
    m.kA.prototype.isActive = function() {
        return null != this.ta
    };
    m.kA.prototype.j = function() {
        this.Fj && this.ta && m.sm(this.ta);
        this.ta = null;
        this.k.call(this.aa, (0, m.H)())
    };
    m.kA.prototype.N = function() {
        this.stop();
        m.kA.K.N.call(this)
    };
    m.s(nA, m.ym);
    m.g = nA.prototype;
    m.g.cc = function() {
        this.eb("begin")
    };
    m.g.Sb = function() {
        this.eb("end")
    };
    m.g.ka = function() {
        this.eb("finish")
    };
    m.g.onStop = function() {
        this.eb("stop")
    };
    m.g.eb = function(a) {
        this.dispatchEvent(a)
    };
    var qA;
    var Mfa = /__([a-z]+(?:_[a-z]+)*)__/g;
    m.sA.prototype.render = function(a, b, c) {
        var d = (0, m.p)(function(d, f) {
            b && (f = b(f));
            return c ? a[f] || this.g[f] || "" : m.wa(a[f] || this.g[f] || "")
        }, this);
        return this.j.replace(this.k, d)
    };
    m.s(m.vA, m.t);
    m.g = m.vA.prototype;
    m.g.ta = 0;
    m.g.N = function() {
        m.vA.K.N.call(this);
        this.stop();
        delete this.g;
        delete this.aa
    };
    m.g.start = function(a) {
        this.stop();
        this.ta = m.Cm(this.Xa, m.ca(a) ? a : this.k)
    };
    m.g.stop = function() {
        this.isActive() && m.da.clearTimeout(this.ta);
        this.ta = 0
    };
    m.g.Jc = m.aa(18);
    m.vA.prototype.isActive = function() {
        return 0 != this.ta
    };
    m.vA.prototype.j = function() {
        this.ta = 0;
        this.g && this.g.call(this.aa)
    };
    var xA = {}, yA = null;
    m.s(BA, nA);
    m.g = BA.prototype;
    m.g.Rl = 0;
    m.g.Hc = 0;
    m.g.Wi = null;
    m.g.play = function(a) {
        if (a || 0 == this.g)
            this.Hc = 0, this.j = this.k;
        else if (1 == this.g)
            return !1;
        wA(this);
        this.startTime = a = (0, m.H)();
        - 1 == this.g && (this.startTime -= this.duration * this.Hc);
        this.C = this.startTime + this.duration;
        this.Wi = this.startTime;
        this.Hc || this.cc();
        this.eb("play");
        - 1 == this.g && this.eb("resume");
        this.g = 1;
        var b = m.oa(this);
        b in xA || (xA[b] = this);
        zA();
        AA(this, a);
        return !0
    };
    m.g.stop = function(a) {
        wA(this);
        this.g = 0;
        a && (this.Hc = 1);
        CA(this, this.Hc);
        this.onStop();
        this.Sb()
    };
    m.g.pause = function() {
        1 == this.g && (wA(this), this.g =- 1, this.eb("pause"))
    };
    m.g.N = function() {
        0 == this.g || this.stop(!1);
        this.eb("destroy");
        BA.K.N.call(this)
    };
    m.g.destroy = function() {
        this.dispose()
    };
    m.g.Hj = function() {
        this.eb("animate")
    };
    m.g.eb = function(a) {
        this.dispatchEvent(new DA(a, this))
    };
    m.s(DA, m.$l);
    m.s(EA, BA);
    EA.prototype.tg = m.ea;
    EA.prototype.Hj = function() {
        this.tg();
        EA.K.Hj.call(this)
    };
    EA.prototype.Sb = function() {
        this.tg();
        EA.K.Sb.call(this)
    };
    EA.prototype.cc = function() {
        this.tg();
        EA.K.cc.call(this)
    };
    m.s(m.FA, EA);
    m.FA.prototype.tg = function() {
        this.element.style.left = Math.round(this.j[0]) + "px";
        this.element.style.top = Math.round(this.j[1]) + "px"
    };
    m.s(GA, EA);
    var Sga = 1 / 1024;
    m.g = GA.prototype;
    m.g.tg = function() {
        var a = this.j[0];
        Math.abs(a - this.D) >= Sga && (m.yk(this.element, a), this.D = a)
    };
    m.g.cc = function() {
        this.D =- 1;
        GA.K.cc.call(this)
    };
    m.g.Sb = function() {
        this.D =- 1;
        GA.K.Sb.call(this)
    };
    m.g.show = function() {
        this.element.style.display = ""
    };
    m.g.hide = function() {
        this.element.style.display = "none"
    };
    m.s(m.HA, GA);
    m.HA.prototype.cc = function() {
        this.show();
        m.HA.K.cc.call(this)
    };
    m.HA.prototype.Sb = function() {
        this.hide();
        m.HA.K.Sb.call(this)
    };
    m.s(m.IA, GA);
    m.IA.prototype.cc = function() {
        this.show();
        m.IA.K.cc.call(this)
    };
    m.s(m.JA, m.ym);
    m.g = m.JA.prototype;
    m.g.nf = null;
    m.g.ri = null;
    m.g.Dk = null;
    m.g.ti = null;
    m.g.Xb =- 1;
    m.g.Dd =- 1;
    m.g.Ck=!1;
    var YE = {
        3: 13,
        12: 144,
        63232: 38,
        63233: 40,
        63234: 37,
        63235: 39,
        63236: 112,
        63237: 113,
        63238: 114,
        63239: 115,
        63240: 116,
        63241: 117,
        63242: 118,
        63243: 119,
        63244: 120,
        63245: 121,
        63246: 122,
        63247: 123,
        63248: 44,
        63272: 46,
        63273: 36,
        63275: 35,
        63276: 33,
        63277: 34,
        63289: 144,
        63302: 45
    }, ZE = {
        Up: 38,
        Down: 40,
        Left: 37,
        Right: 39,
        Enter: 13,
        F1: 112,
        F2: 113,
        F3: 114,
        F4: 115,
        F5: 116,
        F6: 117,
        F7: 118,
        F8: 119,
        F9: 120,
        F10: 121,
        F11: 122,
        F12: 123,
        "U+007F": 46,
        Home: 36,
        End: 35,
        PageUp: 33,
        PageDown: 34,
        Insert: 45
    }, Tga = m.E || m.Qe && m.Fb("525"), $E = m.Oe && m.Re;
    m.g = m.JA.prototype;
    m.g.hz = function(a) {
        m.Qe && (17 == this.Xb&&!a.ctrlKey || 18 == this.Xb&&!a.altKey || m.Oe && 91 == this.Xb&&!a.metaKey) && (this.Dd = this.Xb =- 1);
        - 1 == this.Xb && (a.ctrlKey && 17 != a.keyCode ? this.Xb = 17 : a.altKey && 18 != a.keyCode ? this.Xb = 18 : a.metaKey && 91 != a.keyCode && (this.Xb = 91));
        Tga&&!Kca(a.keyCode, this.Xb, a.shiftKey, a.ctrlKey, a.altKey) ? this.handleEvent(a) : (this.Dd = Bo(a.keyCode), $E && (this.Ck = a.altKey))
    };
    m.g.iz = function(a) {
        this.Dd = this.Xb =- 1;
        this.Ck = a.altKey
    };
    m.g.handleEvent = function(a) {
        var b = a.wa, c, d, e = b.altKey;
        m.E && "keypress" == a.type ? (c = this.Dd, d = 13 != c && 27 != c ? b.keyCode : 0) : m.Qe && "keypress" == a.type ? (c = this.Dd, d = 0 <= b.charCode && 63232 > b.charCode && Ao(c) ? b.charCode : 0) : m.Se ? (c = this.Dd, d = Ao(c) ? b.keyCode : 0) : (c = b.keyCode || this.Dd, d = b.charCode || 0, $E && (e = this.Ck), m.Oe && 63 == d && 224 == c && (c = 191));
        var f = c = Bo(c), h = b.keyIdentifier;
        c ? 63232 <= c && c in YE ? f = YE[c] : 25 == c && a.shiftKey && (f = 9) : h && h in ZE && (f = ZE[h]);
        this.Xb = f;
        a = new MA(f, d, 0, b);
        a.altKey = e;
        this.dispatchEvent(a)
    };
    m.g.M = function() {
        return this.nf
    };
    m.g.N = function() {
        m.JA.K.N.call(this);
        LA(this)
    };
    m.s(MA, m.am);
    m.NA.prototype.Ua = function() {};
    m.s(OA, m.NA);
    OA.prototype.Ua = function(a, b, c, d) {
        Eo(this.Yd, a, b, c, null, null, d)
    };
    var Qfa = {
        GOOGLE_MAPS_API: "//maps.google.com/maps/api/js?sensor=false",
        GOOGLE_LANGUAGE_API_VIRTUAL_KEYBOARD: "//www.google.com/jsapi?key=youtube-internal-vk",
        GOOGLE_LANGUAGE_API_INPUT_TOOLS: "//www.google.com/jsapi?key=youtube-internal-it",
        GOOGLE_PLUS_ONE: "//apis.google.com/js/platform.js",
        GOOGLE_JSAPI: "//www.google.com/jsapi"
    }, SA = {}, PA = {}, QA = {};
    m.bB = {
        Yk: ["BC", "AD"],
        Zs: ["Before Christ", "Anno Domini"],
        $s: "JFMAMJJASOND".split(""),
        bt: "JFMAMJJASOND".split(""),
        Ji: "January February March April May June July August September October November December".split(" "),
        xe: "January February March April May June July August September October November December".split(" "),
        Ki: "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(" "),
        Mi: "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(" "),
        Ni: "Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),
        Wl: "Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),
        Li: "Sun Mon Tue Wed Thu Fri Sat".split(" "),
        fj: "Sun Mon Tue Wed Thu Fri Sat".split(" "),
        g: "SMTWTFS".split(""),
        ct: "SMTWTFS".split(""),
        $k: ["Q1", "Q2", "Q3", "Q4"],
        Zk: ["1st quarter", "2nd quarter", "3rd quarter", "4th quarter"],
        Ii: ["AM", "PM"],
        Re: ["EEEE, MMMM d, y", "MMMM d, y", "MMM d, y", "M/d/yy"],
        Sh: ["h:mm:ss a zzzz", "h:mm:ss a z", "h:mm:ss a", "h:mm a"],
        Cn: ["{1} 'at' {0}", "{1} 'at' {0}", "{1}, {0}", "{1}, {0}"],
        ah: 6,
        gj: [5, 6],
        bh: 5
    };
    m.g = m.UA.prototype;
    m.g.kd = m.bB.ah;
    m.g.Qf = m.bB.bh;
    m.g.clone = function() {
        var a = new m.UA(this.g);
        a.kd = this.kd;
        a.Qf = this.Qf;
        return a
    };
    m.g.getFullYear = function() {
        return this.g.getFullYear()
    };
    m.g.getMonth = function() {
        return this.g.getMonth()
    };
    m.g.getDate = function() {
        return this.g.getDate()
    };
    m.g.getTime = function() {
        return this.g.getTime()
    };
    m.g.getDay = function() {
        return this.g.getDay()
    };
    m.g.getUTCFullYear = function() {
        return this.g.getUTCFullYear()
    };
    m.g.getUTCMonth = function() {
        return this.g.getUTCMonth()
    };
    m.g.getUTCDate = function() {
        return this.g.getUTCDate()
    };
    m.g.getUTCHours = function() {
        return this.g.getUTCHours()
    };
    m.g.getUTCMinutes = function() {
        return this.g.getUTCMinutes()
    };
    m.g.getTimezoneOffset = function() {
        return this.g.getTimezoneOffset()
    };
    m.g.set = function(a) {
        this.g = new Date(a.getFullYear(), a.getMonth(), a.getDate())
    };
    m.g.setFullYear = function(a) {
        this.g.setFullYear(a)
    };
    m.g.setMonth = function(a) {
        this.g.setMonth(a)
    };
    m.g.setDate = function(a) {
        this.g.setDate(a)
    };
    m.g.setTime = m.aa(20);
    m.g.add = function(a) {
        if (a.F || a.D) {
            var b = this.getMonth() + a.D + 12 * a.F, c = this.getFullYear() + Math.floor(b / 12), b = b%12;
            0 > b && (b += 12);
            var d = Math.min(m.TA(c, b), this.getDate());
            this.setDate(1);
            this.setFullYear(c);
            this.setMonth(b);
            this.setDate(d)
        }
        a.g && (a = new Date((new Date(this.getFullYear(), this.getMonth(), this.getDate(), 12)).getTime() + 864E5 * a.g), this.setDate(1), this.setFullYear(a.getFullYear()), this.setMonth(a.getMonth()), this.setDate(a.getDate()), WA(this, a.getDate()))
    };
    m.g.Ci = function(a, b) {
        return [this.getFullYear(), m.Tl(this.getMonth() + 1, 2), m.Tl(this.getDate(), 2)].join(a ? "-" : "") + (b ? m.XA(this) : "")
    };
    m.g.equals = function(a) {
        return !(!a || this.getFullYear() != a.getFullYear() || this.getMonth() != a.getMonth() || this.getDate() != a.getDate())
    };
    m.g.toString = function() {
        return this.Ci()
    };
    m.g.valueOf = function() {
        return this.g.valueOf()
    };
    var eB = [/^\'(?:[^\']|\'\')*\'/, /^(?:G+|y+|M+|k+|S+|E+|a+|h+|K+|H+|c+|L+|Q+|d+|m+|s+|v+|w+|z+|Z+)/, /^[^\'GyMkSEahKHcLQdmsvwzZ]+/];
    m.aB.prototype.format = function(a, b) {
        if (!a)
            throw Error("The date to format must be non-null.");
        var c = b ? 6E4 * (a.getTimezoneOffset() - (b.g - $A(b, a))): 0, d = c ? new Date(a.getTime() + c): a, e = d;
        b && d.getTimezoneOffset() != a.getTimezoneOffset() && (e = new Date(a.getTime() + (c + (0 < c?-864E5 : 864E5))));
        for (var c = [], f = 0; f < this.j.length; ++f) {
            var h = this.j[f].text;
            1 == this.j[f].type ? c.push(Rfa(this, h, a, d, e, b)) : c.push(h)
        }
        return c.join("")
    };
    m.s(m.kB, nA);
    m.kB.prototype.add = function(a) {
        m.La(this.j, a) || (this.j.push(a), m.im(a, "finish", this.D, !1, this))
    };
    m.kB.prototype.remove = function(a) {
        m.Oa(this.j, a) && m.rm(a, "finish", this.D, !1, this)
    };
    m.kB.prototype.N = function() {
        (0, m.y)(this.j, function(a) {
            a.dispose()
        });
        this.j.length = 0;
        m.kB.K.N.call(this)
    };
    m.s(m.lB, m.kB);
    m.lB.prototype.play = function(a) {
        if (0 == this.j.length)
            return !1;
        if (a || 0 == this.g)
            this.k < this.j.length && 0 != this.j[this.k].g && this.j[this.k].stop(!1), this.k = 0, this.cc();
        else if (1 == this.g)
            return !1;
        this.eb("play");
        - 1 == this.g && this.eb("resume");
        this.startTime = (0, m.H)();
        this.C = null;
        this.g = 1;
        this.j[this.k].play(a);
        return !0
    };
    m.lB.prototype.pause = function() {
        1 == this.g && (this.j[this.k].pause(), this.g =- 1, this.eb("pause"))
    };
    m.lB.prototype.stop = function(a) {
        this.g = 0;
        this.C = (0, m.H)();
        if (a)
            for (a = this.k; a < this.j.length; ++a) {
                var b = this.j[a];
                0 == b.g && b.play();
                0 == b.g || b.stop(!0)
            } else 
                this.k < this.j.length && this.j[this.k].stop(!1);
        this.onStop();
        this.Sb()
    };
    m.lB.prototype.D = function() {
        1 == this.g && (this.k++, this.k < this.j.length ? this.j[this.k].play() : (this.C = (0, m.H)(), this.g = 0, this.ka(), this.Sb()))
    };
    m.g = m.mB.prototype;
    m.g.Yh = null;
    m.g.Aa = null;
    m.g.listen = function(a, b, c, d) {
        c = (0, m.p)(c, d || this.Yh);
        a = m.J(a, b, c);
        this.Aa.push(a);
        return a
    };
    m.g.Da = function(a) {
        m.N(a);
        m.Oa(this.Aa, a)
    };
    m.g.removeAll = function() {
        m.N(this.Aa);
        this.Aa = []
    };
    m.nB.prototype.write = function(a, b, c, d) {
        var e = m.Kl();
        a = {
            action_write_promo: 1,
            method: a,
            subtype: this.j
        };
        this.g && (a.encrypted_video_id = this.g);
        e = {
            session_token: e
        };
        c && (e.json_data = c);
        m.R(d || "/promo_ajax", {
            method: "POST",
            Z: a,
            V: e,
            R: function() {
                if (b && (b.dm && m.P(m.F(b.dm)), b.Ct && m.K(m.F(b.Ct)), b.R && b.R(), b.redirectUrl))
                    return m.ae(b.redirectUrl)
            },
            onError: function() {
                b && b.Dt && m.K(m.F(b.Dt));
                if (b && b.onError)
                    b.onError()
            }
        })
    };
    m.nB.prototype.getStatus = function(a, b) {
        m.R("/promo_ajax", {
            method: "POST",
            Z: {
                action_get_promo_status: 1,
                subtype: this.j
            },
            V: {
                session_token: m.Kl()
            },
            R: function(b, d) {
                a(d.show_promo)
            },
            onError: b || void 0
        })
    };
    m.g = m.U.prototype;
    m.g.B = function() {
        var a = this.constructor;
        return a.Gz || (a.Gz = m.X(a, a.NA))
    };
    m.g.has = m.aa(21);
    m.g.get = function(a, b) {
        a.D.B();
        this.B();
        return m.W(this, a.j, b)
    };
    m.g.set = function(a, b) {
        a.D.B();
        this.B();
        m.V(this, a.j, b)
    };
    m.g.add = function(a, b) {
        a.D.B();
        this.B();
        m.uB(this, a.j, b)
    };
    m.g.clear = function(a) {
        a.D.B();
        this.B();
        m.xB(this, a.j)
    };
    m.g.equals = function(a) {
        if (!a || this.constructor != a.constructor)
            return !1;
        for (var b = m.oB(this.B()), c = 0; c < b.length; c++) {
            var d = b[c], e = d.j;
            if (m.sB(this, e) != m.sB(a, e))
                return !1;
            if (m.sB(this, e)) {
                var f = m.qB(d), h = vB(this, e), e = vB(a, e);
                if (d.k) {
                    if (h.length != e.length)
                        return !1;
                    for (d = 0; d < h.length; d++) {
                        var k = h[d], r = e[d];
                        if (f?!k.equals(r) : k != r)
                            return !1
                    }
                } else if (f?!h.equals(e) : h != e)
                    return !1
            }
        }
        return !0
    };
    m.g.clone = function() {
        var a = new this.constructor;
        a != this && (a.j = {}, a.g && (a.g = {}), rB(a, this));
        return a
    };
    var AB;
    m.BB = null;
    m.CB = null;
    AB = null;
    m.Uga = m.Re || m.Qe || m.Se || "function" == typeof m.da.atob;
    var FB, IB, QB, Ufa, JB, SB;
    FB = {};
    IB = 4294967296 * 4294967296 / 2;
    m.HB = m.EB(0);
    QB = m.EB(1);
    m.UB = m.EB( - 1);
    Ufa = m.LB( - 1, 2147483647);
    JB = m.LB(0, - 2147483648);
    SB = m.EB(16777216);
    DB.prototype.toString = function(a) {
        a = a || 10;
        if (2 > a || 36 < a)
            throw Error("radix out of range: " + a);
        if (NB(this))
            return "0";
        if (0 > this.g) {
            if (this.equals(JB)) {
                var b = m.GB(a), c = TB(this, b), b = PB(m.RB(c, b), this);
                return c.toString(a) + b.j.toString(a)
            }
            return "-" + m.KB(this).toString(a)
        }
        for (var c = m.GB(Math.pow(a, 6)), b = this, d = ""; ;) {
            var e = TB(b, c), f = PB(b, m.RB(e, c)).j.toString(a), b = e;
            if (NB(b))
                return f + d;
            for (; 6 > f.length;)
                f = "0" + f;
            d = "" + f + d
        }
    };
    DB.prototype.equals = function(a) {
        return this.g == a.g && this.j == a.j
    };
    DB.prototype.add = function(a) {
        var b = this.g>>>16, c = this.g & 65535, d = this.j>>>16, e = a.g>>>16, f = a.g & 65535, h = a.j>>>16, k;
        k = 0 + ((this.j & 65535) + (a.j & 65535));
        a = 0 + (k>>>16);
        a += d + h;
        d = 0 + (a>>>16);
        d += c + f;
        c = 0 + (d>>>16);
        c = c + (b + e) & 65535;
        return m.LB((a & 65535)<<16 | k & 65535, c<<16 | d & 65535)
    };
    m.YB.prototype.D = m.aa(23);
    m.YB.prototype.C = function(a, b) {
        if (m.qB(a))
            return b instanceof m.U ? b : m.ZB(this, a.C.B(), b);
        if (14 == a.g ||!a.H)
            return b;
        var c = a.C;
        if (c === String) {
            if (m.ka(b))
                return String(b)
        } else if (c === Number && m.ja(b) && ("Infinity" === b || "-Infinity" === b || "NaN" === b || /^-?[0-9]+$/.test(b)))
            return Number(b);
        return b
    };
    m.s(m.$B, m.YB);
    m.$B.prototype.j = m.aa(24);
    m.$B.prototype.D = m.aa(22);
    m.$B.prototype.F = function(a, b) {
        if (null == b)
            return b;
        for (var c = a.B(), d = 0; d < b.length;) {
            var e = bC(this, b.subarray(d)), f = e.value, h = f>>3, f = f & 7, d = d + e.length;
            if (e = c.g[(0, window.parseInt)(h, 10)] || null)
                if (e.J)
                    for (h = bC(this, b.subarray(d)), f = h.value, d += h.length; 0 < f && d < b.length;) {
                        h = this.C(e, b.subarray(d));
                        if (!h)
                            throw Error("Expected " + e.g);
                            a.add(e, h.value);
                            d += h.length;
                            f -= h.length
                    } else {
                        f = this.C(e, b.subarray(d));
                        if (!f)
                            throw Error("Expected " + e.g);
                            d += f.length;
                            e.k ? a.add(e, f.value) : a.set(e, f.value)
                        } else {
                e = d;
                d = b.subarray(d);
                h = 0;
                switch (f) {
                case 0:
                    h = aC(this, d).length;
                    break;
                case 1:
                    h = 8;
                    break;
                case 2:
                    d = aC(this, d);
                    h = d.length + d.value.j;
                    break;
                case 5:
                    h = 4
                }
                d = e + h
            }
        }
    };
    m.$B.prototype.C = function(a, b) {
        var c = null, d = a.g, e = aC(this, b), f = e.length;
        switch (d) {
        case 17:
            c = e.value.j;
            c = c>>>1^-(c & 1);
            break;
        case 18:
            c = e.value;
            c = m.WB(m.XB(c, 1), m.KB(m.LB(c.j & QB.j, c.g & QB.g))).toString();
            break;
        case 8:
            c = e.value.equals(QB);
            break;
        case 3:
        case 4:
            c = e.value.toString();
            break;
        case 14:
        case 5:
        case 13:
            c = e.value.j;
            break;
        case 6:
        case 16:
            c = cC(b.subarray(0, 8)).toString();
            f = 8;
            break;
        case 1:
            c = b.subarray(0, 8);
            for (f = 0; 8 > f; f++)
                this.k.setUint8(f, c[f]);
            c = this.k.getFloat64(0, !0);
            f = 8;
            break;
        case 9:
            c = b.subarray(e.length,
            e.length + e.value.j);
            c = dC(c);
            c = (0, window.decodeURIComponent)((0, window.escape)(c));
            f = e.length + e.value.j;
            break;
        case 12:
            c = b.subarray(e.length, e.length + e.value.j);
            c = dC(c);
            f = e.length + e.value.j;
            break;
        case 10:
            for (var f = c = m.pB(a.C.B()), e = b, d = f.B(), h = 0; ;) {
                var k = bC(this, e), r = k.value, k = k.length, w = r>>3;
                if (4 == (r & 7))
                    break;
                h += k;
                r = {
                    value: void 0,
                    length: 0
                };
                (w = d.g[(0, window.parseInt)(w, 10)] || null) && (r = this.C(w, e.subarray(k))) && null !== r.value && (w.k ? f.add(w, r.value) : f.set(w, r.value));
                h += r.length;
                if (e.length < k + r.length)
                    break;
                e = e.subarray(k + r.length)
            }
            f = h;
            e = aC(this, b.subarray(f));
            f += e.length;
            break;
        case 11:
            f = e.length + e.value.j;
            e = b.subarray(e.length, f);
            c = m.pB(a.C.B());
            this.F(c, e);
            break;
        case 7:
        case 15:
            c = cC(b.subarray(0, 4));
            f = 4;
            break;
        case 2:
            c = b.subarray(0, 4);
            for (f = 0; 4 > f; f++)
                this.k.setUint8(f, c[f]);
            c = this.k.getFloat32(0, !0);
            f = 4
        }
        return {
            value: c,
            length: f
        }
    };
    m.Vga = new m.sA('<div class="ads-mute-survey"><span class="ads-mute-check"></span><b>__mute_gone__</b> __mute_question__<div class="ads-mute-undo">__mute_undo__</div></div>');
    var fC, Vfa, jC, lC;
    Vfa = "pyv-feed-item-headline-dest-url pyv-feed-item-thumb-dest-url pyv-shelf-headline-dest-url pyv-shelf-thumb-dest-url pyv-medium-shelf-headline-dest-url pyv-medium-shelf-thumb-dest-url".split(" ");
    m.iC = "";
    jC = "";
    lC = {};
    m.s(m.nC, m.vq);
    m.fa(m.nC);
    m.nC.prototype.va = "shelfslider";
    m.nC.prototype.Ek = function(a) {
        a = m.jj(a) ? m.No(this, "next", a) : m.No(this, "prev", a);
        return Bk(a) + a.offsetWidth
    };
    m.nC.prototype.Fk = function(a) {
        a = m.jj(a) ? m.No(this, "prev", a) : m.No(this, "next", a);
        return Bk(a)
    };
    var pC;
    var aF = {
        my: {
            1E3: {
                other: "0K"
            },
            1E4: {
                other: "00K"
            },
            1E5: {
                other: "000K"
            },
            1E6: {
                other: "0M"
            },
            1E7: {
                other: "00M"
            },
            1E8: {
                other: "000M"
            },
            1E9: {
                other: "0B"
            },
            1E10: {
                other: "00B"
            },
            1E11: {
                other: "000B"
            },
            1E12: {
                other: "0T"
            },
            1E13: {
                other: "00T"
            },
            1E14: {
                other: "000T"
            }
        },
        ly: {
            1E3: {
                other: "0 thousand"
            },
            1E4: {
                other: "00 thousand"
            },
            1E5: {
                other: "000 thousand"
            },
            1E6: {
                other: "0 million"
            },
            1E7: {
                other: "00 million"
            },
            1E8: {
                other: "000 million"
            },
            1E9: {
                other: "0 billion"
            },
            1E10: {
                other: "00 billion"
            },
            1E11: {
                other: "000 billion"
            },
            1E12: {
                other: "0 trillion"
            },
            1E13: {
                other: "00 trillion"
            },
            1E14: {
                other: "000 trillion"
            }
        }
    }, CC = aF, CC = aF;
    var uC = {
        AED: [2, "dh", "\u062f.\u0625.", "DH"],
        ALL: [0, "Lek", "Lek"],
        AUD: [2, "$", "AU$"],
        BDT: [2, "\u09f3", "Tk"],
        BGN: [2, "lev", "lev"],
        BRL: [2, "R$", "R$"],
        CAD: [2, "$", "C$"],
        CDF: [2, "FrCD", "CDF"],
        CHF: [2, "CHF", "CHF"],
        CLP: [0, "$", "CL$"],
        CNY: [2, "\u00a5", "RMB\u00a5"],
        COP: [0, "$", "COL$"],
        CRC: [0, "\u20a1", "CR\u20a1"],
        CZK: [50, "K\u010d", "K\u010d"],
        DKK: [18, "kr", "kr"],
        DOP: [2, "$", "RD$"],
        EGP: [2, "\u00a3", "LE"],
        ETB: [2, "Birr", "Birr"],
        EUR: [2, "\u20ac", "\u20ac"],
        GBP: [2, "\u00a3", "GB\u00a3"],
        HKD: [2, "$", "HK$"],
        HRK: [2, "kn", "kn"],
        HUF: [0, "Ft",
        "Ft"],
        IDR: [0, "Rp", "Rp"],
        ILS: [2, "\u20aa", "IL\u20aa"],
        INR: [2, "\u20b9", "Rs"],
        IRR: [0, "Rial", "IRR"],
        ISK: [0, "kr", "kr"],
        JMD: [2, "$", "JA$"],
        JPY: [0, "\u00a5", "JP\u00a5"],
        KRW: [0, "\u20a9", "KR\u20a9"],
        LKR: [2, "Rs", "SLRs"],
        LTL: [2, "Lt", "Lt"],
        LVL: [2, "Ls", "Ls"],
        MNT: [0, "\u20ae", "MN\u20ae"],
        MXN: [2, "$", "Mex$"],
        MYR: [2, "RM", "RM"],
        NOK: [50, "kr", "NOkr"],
        PAB: [2, "B/.", "B/."],
        PEN: [2, "S/.", "S/."],
        PHP: [2, "\u20b1", "Php"],
        PKR: [0, "Rs", "PKRs."],
        PLN: [50, "z\u0142", "z\u0142"],
        RON: [2, "RON", "RON"],
        RSD: [0, "din", "RSD"],
        RUB: [50, "\u0440\u0443\u0431.",
        "\u0440\u0443\u0431."],
        SAR: [2, "Rial", "Rial"],
        SEK: [2, "kr", "kr"],
        SGD: [2, "$", "S$"],
        THB: [2, "\u0e3f", "THB"],
        TRY: [2, "TL", "YTL"],
        TWD: [2, "NT$", "NT$"],
        TZS: [0, "TSh", "TSh"],
        UAH: [2, "\u20b4", "UAH"],
        USD: [2, "$", "US$"],
        UYU: [2, "$", "$U"],
        VND: [0, "\u20ab", "VN\u20ab"],
        YER: [0, "Rial", "Rial"],
        ZAR: [2, "R", "ZAR"]
    };
    var bF = {
        DECIMAL_SEP: ".",
        GROUP_SEP: ",",
        Km: "%",
        Xj: "0",
        Gy: "+",
        Fy: "-",
        Um: "E",
        Lm: "\u2030",
        Yi: "\u221e",
        Rr: "NaN",
        DECIMAL_PATTERN: "#,##0.###",
        hq: "#E0",
        gq: "#,##0%",
        dq: "\u00a4#,##0.00",
        fq: "USD"
    }, sC = bF, sC = bF;
    m.rC.prototype.parse = function(a, b) {
        var c = b || [0];
        if (0 != this.H)
            throw Error("Parsing of compact numbers is unimplemented");
        var d = window.NaN;
        a = a.replace(/ /g, "\u00a0");
        var e = a.indexOf(this.D, c[0]) == c[0], f = a.indexOf(this.k, c[0]) == c[0];
        e && f && (this.D.length > this.k.length ? f=!1 : this.D.length < this.k.length && (e=!1));
        e ? c[0] += this.D.length : f && (c[0] += this.k.length);
        if (a.indexOf(sC.Yi, c[0]) == c[0])
            c[0] += sC.Yi.length, d = window.Infinity;
        else {
            var d = a, h=!1, k=!1, r=!1, w = 1, B = sC.DECIMAL_SEP, Q = sC.GROUP_SEP, Y = sC.Um;
            if (0 != this.H)
                throw Error("Parsing of compact style numbers is not implemented");
            for (var la = ""; c[0] < d.length; c[0]++) {
                var Ma = d.charAt(c[0]), Vb = AC(Ma);
                if (0 <= Vb && 9 >= Vb)
                    la += Vb, r=!0;
                else if (Ma == B.charAt(0)) {
                    if (h || k)
                        break;
                    la += ".";
                    h=!0
                } else if (Ma == Q.charAt(0) && ("\u00a0" != Q.charAt(0) || c[0] + 1 < d.length && 0 <= AC(d.charAt(c[0] + 1)))) {
                    if (h || k)
                        break
                } else if (Ma == Y.charAt(0)) {
                    if (k)
                        break;
                    la += "E";
                    k=!0
                } else if ("+" == Ma || "-" == Ma)
                    la += Ma;
                else if (Ma == sC.Km.charAt(0)) {
                    if (1 != w)
                        break;
                    w = 100;
                    if (r) {
                        c[0]++;
                        break
                    }
                } else if (Ma == sC.Lm.charAt(0)) {
                    if (1 != w)
                        break;
                    w = 1E3;
                    if (r) {
                        c[0]++;
                        break
                    }
                } else 
                    break
            }
            d = (0, window.parseFloat)(la) /
            w
        }
        if (e) {
            if (a.indexOf(this.S, c[0]) != c[0])
                return window.NaN;
            c[0] += this.S.length
        } else if (f) {
            if (a.indexOf(this.F, c[0]) != c[0])
                return window.NaN;
            c[0] += this.F.length
        }
        return f?-d : d
    };
    m.rC.prototype.format = function(a) {
        if ((0, window.isNaN)(a))
            return sC.Rr;
        var b = [], c;
        var d = a, e = a;
        0 == this.H ? c = DC : (d = Math.abs(d), e = Math.abs(e), c = BC(this, 1 >= d ? 0 : EC(d)).Zi, xC(this, e / Math.pow(10, c)), d = xC(this, d / Math.pow(10, c)), c = BC(this, c + EC(d.intValue)));
        a/=Math.pow(10, c.Zi);
        b.push(c.prefix);
        d = 0 > a || 0 == a && 0 > 1 / a;
        b.push(d ? this.k : this.D);
        if ((0, window.isFinite)(a))
            if (a = a * (d?-1 : 1) * this.J, this.U)
                if (0 == a)
                    yC(this, a, this.g, b), zC(this, 0, b);
                else {
                    e = Math.log(a) / Math.log(10);
                    e = Math.floor(e + 2E-15);
                    a/=Math.pow(10, e);
                    var f =
                    this.g;
                    if (1 < this.O && this.O > this.g) {
                        for (; 0 != e%this.O;)
                            a*=10, e--;
                            f = 1
                    } else 
                        1 > this.g ? (e++, a/=10) : (e -= this.g - 1, a*=Math.pow(10, this.g - 1));
                        yC(this, a, f, b);
                        zC(this, e, b)
                } else 
                    yC(this, a, this.g, b);
        else 
            b.push(sC.Yi);
        b.push(d ? this.F : this.S);
        b.push(c.Bl);
        return b.join("")
    };
    var DC = {
        prefix: "",
        Bl: "",
        Zi: 0
    };
    m.FC.prototype.g = function(a) {
        a = m.I(a.target, "preview-id");
        for (var b = m.M("ad-format-preview", this.da), c = 0; c < b.length; c++)
            b[c].id === a ? m.K(b[c]) : m.P(b[c])
    };
    m.s(GC, m.t);
    GC.prototype.Qc = function(a) {
        a || this.D.listen(m.Yb(this.g), "mousemove", this.L);
        this.D.listen(this.j, "tick", this.J)
    };
    GC.prototype.J = function() {
        this.g.scrollTop += this.C.y;
        this.g.scrollLeft += this.C.x
    };
    GC.prototype.L = function(a) {
        var b = HC(a.clientX, this.k.left, this.k.width);
        a = HC(a.clientY, this.k.top, this.k.height);
        this.C.x = b;
        this.C.y = a;
        (b=!b&&!a) || (b=!1);
        b ? this.j.stop() : this.j.enabled || this.j.start()
    };
    GC.prototype.N = function() {
        GC.K.N.call(this);
        this.D.dispose();
        this.j.dispose()
    };
    m.s(m.IC, m.Ko);
    m.fa(m.IC);
    m.dr(m.IC);
    m.g = m.IC.prototype;
    m.g.va = "dragdrop";
    m.g.register = function() {
        m.T(this, "mousedown", this.qo, "drag-handle");
        m.T(this, "mouseover", this.po, "draggable-item");
        m.T(this, "mouseover", this.oo, "container");
        var a = window.document.body, b = m.J(a, "mouseup", (0, m.p)(this.fi, this));
        this.F.push(b);
        b = m.J(a, "mousemove", (0, m.p)(this.$x, this));
        this.F.push(b);
        b = m.J(a, "mousedown", (0, m.p)(function(a) {
            3 == a.which && this.fi()
        }, this));
        this.F.push(b);
        a = m.J(a, "mouseout", (0, m.p)(function(a) {
            (a = a.relatedTarget || a.toElement) && "HTML" != a.nodeName || this.fi()
        }, this));
        this.F.push(a)
    };
    m.g.unregister = function() {
        this.fi();
        m.Mo(this, "mousedown", this.qo, "drag-handle");
        m.Mo(this, "mouseover", this.po, "draggable-item");
        m.Mo(this, "mouseover", this.oo, "container");
        (0, m.y)(this.F, function(a) {
            m.N(a)
        })
    };
    m.g.qo = function(a, b, c) {
        if (3 != c.which) {
            this.g = m.L(a, m.S(this, "draggable-item"));
            "TR" == this.g.tagName ? (a = m.Pb("table"), b = m.Pb("tbody"), a.appendChild(b), b.appendChild(this.g.cloneNode(!0)), this.j = a) : this.j = this.g.cloneNode(!0);
            this.j.style.position = "absolute";
            this.S = new GC(si(window.document), 30);
            a = window.document.body;
            b = m.Jj(this.g);
            if (this.O) {
                var d = Kj(this.g);
                this.D = lk(d, new m.ui(c.clientX, c.clientY))
            } else 
                "rtl" == window.document.body.getAttribute("dir") ? this.D = new m.ui( - b.width, 0) : this.D = new m.ui(0,
                0), m.C(this.j, m.S(this, "no-pointer-events")), m.C(a, m.S(this, "show-move-cursor"));
            JC(this, c);
            var d = m.ij(this.g, "padding"), e = d.top + d.bottom;
            this.j.style.width = m.Ye(b.width - (d.left + d.right), !0);
            this.j.style.height = m.Ye(b.height - e, !0);
            m.C(this.g, m.S(this, "dragged-item"));
            m.C(this.j, m.S(this, "cursor-follower"));
            m.D(this.j, m.S(this, "draggable-item"));
            a.appendChild(this.j);
            this.C=!0;
            c.preventDefault()
        }
    };
    m.g.fi = function() {
        if (this.C) {
            this.C=!1;
            m.D(window.document.body, m.S(this, "show-move-cursor"));
            m.D(this.g, m.S(this, "dragged-item"));
            var a = this.la(this.g);
            m.ri(this.j);
            this.D = this.j = null;
            this.S.dispose();
            this.S = null;
            a && this.g && m.z("yt-uix-dragdrop-notification-dragend", a, this.g);
            this.g = null
        }
    };
    m.g.po = function(a) {
        this.C&&!m.A(a, m.S(this, "cursor-follower")) && (this.H = null, this.k = a)
    };
    m.g.oo = function(a) {
        !this.C || m.A(a, m.S(this, "cursor-follower")) || m.M(m.S(this, "draggable-item"), a).length || (this.k = null, this.H = a)
    };
    m.g.$x = function(a) {
        if (this.C && (JC(this, a), (this.k || this.H) && this.g))
            if (this.H)
                this.H.appendChild(this.g);
            else {
                var b = m.tl(this.k) || this.k, c = m.sl(this.k) || this.k, d = KC(a, c);
                a = b == c ? b.offsetHeight - d : KC(a, b);
                d > a ? this.g != b && m.wl(this.g, this.k) : d < a && this.g != c && m.yi(this.g, this.k)
            }
    };
    m.s(m.NC, m.Ko);
    m.fa(m.NC);
    m.NC.prototype.va = "flipwidget";
    m.NC.prototype.register = function() {
        m.T(this, "click", this.g, "flip")
    };
    m.NC.prototype.unregister = function() {
        m.Mo(this, "click", this.g, "flip")
    };
    m.NC.prototype.g = function(a) {
        a = this.la(a);
        m.A(a, m.S(this, "front-side")) ? m.OC(this, a, !0) : m.OC(this, a, !1)
    };
    m.s(QC, m.t);
    m.g = QC.prototype;
    m.g.N = function() {
        m.N(this.g);
        this.g.length = 0;
        this.C && this.C.abort();
        QC.K.N.call(this)
    };
    m.g.Qc = function() {
        this.Db = this.da.getElementsByTagName("form")[0];
        this.g.push(m.J(this.Db, "submit", (0, m.p)(this.ir, this)));
        this.Ti = m.G("share-email-success", this.da);
        this.F = m.G("share-email-remail", this.Ti);
        this.g.push(m.J(this.F, "click", (0, m.p)(function() {
            RC(this);
            this.focus()
        }, this)));
        this.k = m.G("share-email-recipients", this.da);
        this.j = m.G("share-email-note", this.da);
        this.D = m.G("share-email-preview-note", this.da);
        this.g.push(m.J(this.j, "keyup", (0, m.p)(this.gr, this)))
    };
    m.g.Ge = function() {
        this.Db && (this.Ri && RC(this), this.focus())
    };
    m.g.focus = function() {
        this.k.focus()
    };
    m.g.gr = function() {
        var a = this.j.value, a = a.substring(0, 300), a = m.wa(a), a = a.replace(/\n/g, "<br>");
        this.D.innerHTML = a
    };
    m.g.ir = function(a) {
        a.preventDefault();
        var b = m.Mb("button", null, this.Db)[0];
        b.disabled=!0;
        var c = m.G("share-email-captcha", this.da), d = m.G("share-email-error", this.da), e = m.G("yt-alert-content", d);
        a = m.kl(window.document.location.href) + m.mc(this.Db.action);
        m.R(a, {
            format: "JSON",
            method: "POST",
            Rb: m.oo(this.Db),
            R: function() {
                this.Ri=!0;
                m.K(this.Ti);
                m.P(this.Db);
                m.P(d);
                m.P(c)
            },
            onError: function(a, b) {
                b.captcha_html && (c.innerHTML = b.captcha_html, m.K(c));
                b.errors && (e.innerHTML = b.errors.join("<br>"), m.K(d))
            },
            ka: function() {
                b.disabled =
                !1
            },
            context: this
        })
    };
    m.s(SC, m.t);
    m.g = SC.prototype;
    m.g.Xf = 0;
    m.g.Wf = 0;
    m.g.N = function() {
        m.N(this.g);
        this.g.length = 0;
        this.D && this.D.abort();
        SC.K.N.call(this)
    };
    m.g.Qc = function() {
        this.j = m.G("share-embed-code", this.da);
        this.g.push(m.J(this.j, "keydown", (0, m.p)(this.Xt, this)));
        $fa(this);
        this.g.push(m.J(m.F("embed-with-playlist"), "click", (0, m.p)(this.$t, this)));
        this.g.push(m.J(m.F("embed-with-playlist-current"), "click", (0, m.p)(this.Yt, this)));
        this.g.push(m.J(m.F("embed-with-playlist-first"), "click", (0, m.p)(this.Zt, this)));
        aga(this)
    };
    m.g.Xt = function(a) {
        (a.ctrlKey || a.metaKey) && 67 == a.keyCode && m.Pj("embedCodeCopied", void 0, void 0)
    };
    m.g.As = function() {
        this.Xf = (0, window.parseInt)(this.C.value, 10);
        this.Wf = Math.round(this.Xf / this.F) || 0;
        this.k.value = this.Wf + "";
        TC(this)
    };
    m.g.zs = function() {
        this.Wf = (0, window.parseInt)(this.k.value, 10);
        this.Xf = Math.round(this.Wf * this.F) || 0;
        this.C.value = this.Xf + "";
        TC(this)
    };
    m.g.Ge = function() {
        this.focus()
    };
    m.g.focus = function() {
        this.j && (this.j.focus(), this.j.select())
    };
    m.g.Tv = function() {
        var a = this.Wd.checked, b = m.Ef.getInstance();
        m.ji(2, !a);
        b.save();
        TC(this)
    };
    m.g.Rv = function() {
        var a = this.Ud.checked, b = m.Ef.getInstance();
        m.ji(137, !a);
        b.save();
        TC(this)
    };
    m.g.Sv = function() {
        var a = this.Vd.checked, b = m.Ef.getInstance();
        m.ji(138, !a);
        b.save();
        TC(this)
    };
    m.g.Qv = function() {
        var a = this.kh.checked, b = m.Ef.getInstance();
        m.ji(44, a);
        b.save();
        TC(this)
    };
    m.g.$t = function(a) {
        a.target.checked ? (a = m.qi(m.F("embed-with-playlist-first"), "li"), a = m.A(a, "yt-uix-button-menu-item-selected") ? "first" : "default") : a = "nolist";
        this.Xd = this.Ij[a] || this.Xd;
        TC(this)
    };
    m.g.Yt = function() {
        this.Xd = this.Ij["default"] || this.Xd;
        TC(this)
    };
    m.g.Zt = function() {
        this.Xd = this.Ij.first || this.Xd;
        TC(this)
    };
    m.s(UC, m.ym);
    var cF = [m.E&&!m.Fb("11") ? "readystatechange": "load", "abort", "error"];
    UC.prototype.start = function() {
        var a = this.j;
        (0, m.y)(m.ub(a), function(b) {
            var c = a[b];
            if (c && (delete a[b], !this.ha())) {
                var d;
                this.k ? d = m.Ii(this.k).Ha("img") : d = new window.Image;
                c.tl && (d.crossOrigin = c.tl);
                this.aa.listen(d, cF, this.C);
                this.g[b] = d;
                d.id = b;
                d.src = c.src
            }
        }, this)
    };
    UC.prototype.C = function(a) {
        var b = a.currentTarget;
        if (b) {
            if ("readystatechange" == a.type)
                if ("complete" == b.readyState)
                    a.type = "load";
                else 
                    return;
            "undefined" == typeof b.naturalWidth && ("load" == a.type ? (b.naturalWidth = b.width, b.naturalHeight = b.height) : (b.naturalWidth = 0, b.naturalHeight = 0));
            this.dispatchEvent({
                type: a.type,
                target: b
            });
            !this.ha() && (a = b.id, delete this.j[a], b = this.g[a]) && (delete this.g[a], this.aa.Da(b, cF, this.C), m.kk(this.g) && m.kk(this.j) && this.dispatchEvent("complete"))
        }
    };
    UC.prototype.N = function() {
        delete this.j;
        delete this.g;
        m.Va(this.aa);
        UC.K.N.call(this)
    };
    m.s(m.VC, m.ym);
    var dF = m.E || m.Re && m.Fb("1.9.3");
    m.g = m.VC.prototype;
    m.g.clientX = 0;
    m.g.clientY = 0;
    m.g.screenX = 0;
    m.g.screenY = 0;
    m.g.Nl = 0;
    m.g.Ol = 0;
    m.g.Oe = 0;
    m.g.deltaY = 0;
    m.g.Gb=!0;
    m.g.Sd=!1;
    m.g.setEnabled = function(a) {
        this.Gb = a
    };
    m.g.N = function() {
        m.VC.K.N.call(this);
        m.rm(this.k, ["touchstart", "mousedown"], this.Si, !1, this);
        this.g.removeAll();
        dF && this.j.releaseCapture();
        this.k = this.target = null
    };
    m.g.Si = function(a) {
        var b = "mousedown" == a.type;
        if (!this.Gb || this.Sd || b && (!(Bga ? 0 == a.wa.button : "click" == a.type || a.wa.button & Ega[0]) || m.Qe && m.Oe && a.ctrlKey))
            this.dispatchEvent("earlycancel");
        else if (XC(a), this.dispatchEvent(new bD("start", this, a.clientX, a.clientY))) {
            this.Sd=!0;
            a.preventDefault();
            var b = this.j, c = b.documentElement, d=!dF;
            this.g.listen(b, ["touchmove", "mousemove"], this.ys, d);
            this.g.listen(b, ["touchend", "mouseup"], this.fh, d);
            dF ? (c.setCapture(!1), this.g.listen(c, "losecapture", this.fh)) : this.g.listen(m.Bj(b),
            "blur", this.fh);
            this.F && this.g.listen(this.F, "scroll", this.xs, d);
            this.clientX = this.Nl = a.clientX;
            this.clientY = this.Ol = a.clientY;
            this.screenX = a.screenX;
            this.screenY = a.screenY;
            this.Oe = this.target.offsetLeft;
            this.deltaY = this.target.offsetTop;
            this.D = m.Gi(m.Ii(this.j));
            (0, m.H)()
        }
    };
    m.g.fh = function(a) {
        this.g.removeAll();
        dF && this.j.releaseCapture();
        if (this.Sd) {
            XC(a);
            this.Sd=!1;
            var b = ZC(this, this.Oe), c = $C(this, this.deltaY);
            this.dispatchEvent(new bD("end", this, a.clientX, a.clientY, 0, b, c))
        } else 
            this.dispatchEvent("earlycancel")
    };
    m.g.ys = function(a) {
        if (this.Gb) {
            XC(a);
            var b = 1 * (a.clientX - this.clientX), c = a.clientY - this.clientY;
            this.clientX = a.clientX;
            this.clientY = a.clientY;
            this.screenX = a.screenX;
            this.screenY = a.screenY;
            if (!this.Sd) {
                var d = this.Nl - this.clientX, e = this.Ol - this.clientY;
                if (0 < d * d + e * e)
                    if (this.dispatchEvent(new bD("start", this, a.clientX, a.clientY)))
                        this.Sd=!0;
                    else {
                        this.ha() || this.fh(a);
                        return 
                    }
            }
            c = YC(this, b, c);
            b = c.x;
            c = c.y;
            this.Sd && this.dispatchEvent(new bD("beforedrag", this, a.clientX, a.clientY, 0, b, c)) && (aD(this, a, b, c), a.preventDefault())
        }
    };
    m.g.xs = function(a) {
        var b = YC(this, 0, 0);
        a.clientX = this.clientX;
        a.clientY = this.clientY;
        aD(this, a, b.x, b.y)
    };
    m.s(bD, m.$l);
    dD.prototype.dispose = function() {
        m.y(this.C, m.N);
        this.N()
    };
    dD.prototype.N = function() {};
    dD.prototype.M = function(a, b) {
        return m.G(a, b || this.J)
    };
    m.s(m.fD, dD);
    m.g = m.fD.prototype;
    m.g.N = function() {
        this.Oa && this.Oa.dispose();
        this.Ya && this.Ya.dispose();
        this.Ya = this.Oa = null;
        this.k && (m.sm(this.k), this.k = null)
    };
    m.g.Pg = function(a) {
        var b = m.L(a.target, "trimmer"), b = m.A(b, "left-trimmer");
        a = m.qi(a.target, "BUTTON");
        a = (m.A(a, "nudge-left")?-1E3 : 1E3) / this.fps;
        iD(this, this.media[b ? "start_ms": "end_ms"] + a, b);
        gD(this);
        hD(this, b)
    };
    m.g.nl = function(a) {
        var b = a.target;
        a = b == this.Oa;
        b = b.target;
        iD(this, this.media.length_ms * (b.offsetLeft + (a ? b.offsetWidth : 0)) / this.width, a);
        hD(this, a)
    };
    m.g.pl = function(a) {
        gD(this);
        hD(this, a.target == this.Oa)
    };
    m.g.Wq = function(a) {
        if (!m.L(a.target, "trimmer")) {
            var b = this.media.length_ms * (a.clientX - xk(this.L).x) / this.width, c = Math.abs(this.media.start_ms - b) <= Math.abs(this.media.end_ms - b);
            iD(this, b, c);
            gD(this);
            hD(this, c);
            (c ? this.Oa : this.Ya).Si(a);
            a.stopPropagation()
        }
    };
    m.s(m.pD, m.t);
    m.g = m.pD.prototype;
    m.g.N = function() {
        m.N(this.g);
        this.g.length = 0;
        this.S && this.S.abort();
        m.pD.K.N.call(this)
    };
    m.g.Ff = function(a) {
        this.k && (m.Ub(this.k.parentNode), this.k = null);
        var b = m.Kb("share-panel-gif-alert-container"), c = m.Kb("share-panel-gif-alert-template").cloneNode(!0);
        this.k = m.hs("yt-alert-error", a, c, b)
    };
    m.g.Qc = function() {
        this.Md = m.Kb("gif-start-at");
        this.Ig = m.Kb("gif-end-at");
        this.F = m.Kb("gif-top-text");
        this.L = m.Kb("gif-bottom-text");
        this.La = m.G("share-panel-gif-create", void 0);
        m.G("loading-gif-preview", void 0);
        this.C = m.G("gif-preview-container", void 0);
        this.O = m.G("animated-gif-preview-img", void 0);
        this.j = m.G("share-panel-gif-result", void 0);
        this.T = m.G("loading-gif-result", void 0);
        if (this.Jg=!!m.F("gif-suggestion-menu-entry-template"))
            this.Ga = new bga, this.X = m.F("suggest-menu-content"), this.Ba = m.G("gif-suggestion-loading-spinner"),
            this.qa = m.G("share-panel-gif-loop-suggest-button");
        this.g.push(m.J(this.La, "click", (0, m.p)(this.Dr, this)));
        var a = (0, m.p)(this.Er, this);
        this.g.push(m.J(this.Md, "change", a));
        this.g.push(m.J(this.Ig, "change", a));
        var a = [this.F, this.L], b = (0, m.p)(this.Fr, this);
        (0, m.y)(a, function(a) {
            this.g.push(m.J(a, "change", b));
            this.g.push(m.J(a, "keyup", b))
        }, this);
        a = (0, m.p)(this.Tg, this);
        this.g.push(m.J(this.Md, "focus", a));
        this.g.push(m.J(this.Ig, "focus", a));
        this.Jg && this.g.push(m.O(this.X, "click", (0, m.p)(this.Gr, this),
        "gif-loop-suggestion-menu-item"))
    };
    m.g.jg = m.aa(26);
    m.g.Tg = function() {
        this.gp=!1
    };
    m.g.qu = function(a, b) {
        if (this.lj) {
            this.Tg();
            var c = m.rD(this, [a.start_ms, a.end_ms], "start_ms" == b);
            uD(this, c["start_ms" == b ? 0: 1]);
            m.vD(this, c, !0)
        }
    };
    m.g.ou = function(a) {
        a = a.intervals;
        null != a && ((0, m.y)(a, function(a) {
            a = {
                start_time_ms: a[0].toString(),
                end_time_ms: a[1].toString(),
                display_value: cD(a[0]) + " - " + cD(a[1])
            };
            var c = this.Ga.g.render(a, void 0), c = qk(c);
            (a = sk(c)) && (c = "<table><tbody>" + c + "</tbody></table>");
            c = m.Aj(c);
            a && (c = m.jk(m.jk(c)));
            this.X.appendChild(c)
        }, this), this.qa.disabled=!1);
        m.D(this.Ba, "loading")
    };
    m.g.Gr = function(a) {
        this.Tg();
        var b = a.currentTarget;
        a = (0, window.parseInt)(m.I(b, "start-time-ms"), 10);
        b = (0, window.parseInt)(m.I(b, "end-time-ms"), 10);
        m.vD(this, [a, b], !0)
    };
    m.g.Er = function(a) {
        this.Tg();
        var b = [this.Ma.start_ms, this.Ma.end_ms], c = a.target == this.Md ? 0: 1, d = ega(a.target.value);
        null != d ? (b[c] = d, b = m.rD(this, b, a.target == this.Md), uD(this, b[a.target == this.Md ? 0: 1]), m.vD(this, b, !0)) : a.target.value = cD(b[c])
    };
    m.g.Fr = function(a) {
        var b;
        b = a.target == this.F ? this.ut : this.tt;
        a = (0, m.Fa)(a.target.value);
        b.enabled = 0 < a.length;
        b.parameters.label = a;
        tD(this)
    };
    m.g.Dr = function() {
        m.P(m.G("share-panel-gif-options", void 0));
        m.P(m.Kb("share-panel-gif-trimmer-container"));
        m.P(this.j);
        m.P(this.C);
        m.K(this.T);
        var a = nD([this.Ma]), b = {
            length_ms: this.Ma.end_ms - this.Ma.start_ms
        };
        wD("create_gif", (0, m.p)(this.tw, this), a, (0, m.p)(this.Ff, this), b)
    };
    m.g.mt = function() {
        m.Ub(this.j);
        m.K(m.G("share-panel-gif-options", void 0));
        m.K(m.Kb("share-panel-gif-trimmer-container"));
        this.hn()
    };
    m.g.tw = function(a) {
        var b = m.Uk(a.gif_html);
        m.Ub(this.j);
        this.j.appendChild(b);
        this.H = m.G("share-panel-gif-url", this.j);
        this.U = m.G("share-panel-gif-iframe", this.j);
        this.Ca = m.G("share-panel-gif-new", this.j);
        this.g.push(m.J(this.H, "click", (0, m.p)(this.qt, this)));
        this.g.push(m.J(this.U, "click", (0, m.p)(this.ot, this)));
        this.g.push(m.J(this.Ca, "click", (0, m.p)(this.mt, this)));
        this.H.select();
        xD(this, a.url, (0, m.p)(this.nt, this))
    };
    m.g.nt = function(a) {
        this.vn(a);
        m.P(this.T);
        m.K(this.j);
        m.K(this.C);
        this.H.select()
    };
    m.g.qt = function() {
        this.H.select()
    };
    m.g.ot = function() {
        this.U.select()
    };
    m.g.vn = function(a) {
        m.D(this.C, "loading");
        a = a.currentTarget;
        a = Math.round(320 * a.naturalHeight / a.naturalWidth);
        m.Ek(this.C, "width", "320px");
        m.Ek(this.C, "height", a + "px")
    };
    m.g.hn = function() {
        this.ci += 1;
        var a = this.ci + "";
        m.C(this.C, "loading");
        var b = nD([this.Ma]);
        b.qid = a;
        a = {
            length_ms: this.Ma.end_ms - this.Ma.start_ms
        };
        wD("preview_gif", (0, m.p)(this.Ax, this), b, (0, m.p)(this.Ff, this), a)
    };
    m.g.Ax = function(a) {
        a.qid == this.ci && xD(this, a.thumb_url, (0, m.p)(this.vn, this))
    };
    m.g.lj=!1;
    m.g.ci =- 1;
    m.g.gp=!0;
    m.g.Jg=!1;
    m.s(m.yD, m.t);
    m.g = m.yD.prototype;
    m.g.N = function() {
        m.N(this.g);
        this.g.length = 0;
        m.bb(0);
        this.ja && this.ja.abort();
        m.yD.K.N.call(this)
    };
    m.g.Qc = function() {
        var a = m.G("share-panel-show-url-options");
        this.g.push(m.J(a, "click", (0, m.p)(this.yr, this)));
        a = m.G("share-panel-show-more");
        this.g.push(m.J(a, "click", (0, m.p)(this.sr, this)));
        a = m.G("share-panel-services", this.da);
        this.g.push(m.J(a, "click", (0, m.p)(this.ur, this)));
        a = m.G("share-panel-embed", this.da);
        this.g.push(m.J(a, "click", (0, m.p)(this.pr, this)));
        a = m.G("share-panel-email", this.da);
        this.g.push(m.J(a, "click", (0, m.p)(this.or, this)));
        a = m.G("share-panel-gif", this.da);
        this.g.push(m.J(a, "click",
        (0, m.p)(this.qr, this)));
        (a = m.G("share-panel-hangout", this.da)) && this.g.push(m.J(a, "click", (0, m.p)(this.rr, this)));
        this.k = m.G("share-panel-url", this.da);
        this.g.push(m.J(this.k, "click", (0, m.p)(this.zr, this)));
        this.g.push(m.J(this.k, "focus", (0, m.p)(function() {
            m.C(this.k, "focused")
        }, this)));
        this.g.push(m.J(this.k, "blur", (0, m.p)(function() {
            m.D(this.k, "focused")
        }, this)));
        this.Ba || (this.Ga = m.G("share-panel-start-at-container", this.da), this.J = m.G("share-panel-start-at", this.da), this.j = m.G("share-panel-start-at-time",
        this.da), this.g.push(m.J(this.j, "change", (0, m.p)(this.nr, this))), this.g.push(m.J(this.j, "click", (0, m.p)(this.xr, this))), this.g.push(m.J(this.j, "focus", (0, m.p)(function() {
            m.C(this.j, "focused")
        }, this))), this.g.push(m.J(this.j, "blur", (0, m.p)(function() {
            m.D(this.j, "focused")
        }, this))));
        this.S = m.G("share-panel-url-options", this.da);
        this.g.push(m.J(this.S, "click", (0, m.p)(this.zf, this)));
        this.U = m.G("share-panel-services", this.da);
        a = m.G("share-panel-show-more", this.da);
        this.g.push(m.J(a, "click", (0, m.p)(this.wr,
        this)));
        this.L = m.G("share-panel-services-container", this.da);
        this.g.push(m.J(m.F("share-with-playlist"), "click", (0, m.p)(this.Cr, this)));
        this.g.push(m.J(m.F("share-with-playlist-current"), "click", (0, m.p)(this.Ar, this)));
        this.g.push(m.J(m.F("share-with-playlist-first"), "click", (0, m.p)(this.Br, this)))
    };
    m.g.Ge = function() {
        this.zf();
        this.La ||!this.k || m.A(this.k, "focused") || this.k.select()
    };
    m.g.jg = m.aa(25);
    m.g.zf = function() {
        if (this.k&&!m.A(this.k, "focused")) {
            var a = this.am, b = {}, c = this.j&&!this.J.disabled && this.J.checked && gga(this.j.value);
            c && (b.t = c);
            a = m.rc(a, b);
            this.k.value != a && (this.k.value = a)
        }
    };
    m.g.nr = function() {
        this.j && (this.qa=!0, this.J.checked=!0, this.zf())
    };
    m.g.xr = function() {
        this.j && (this.J.checked=!0, this.j.value.match(/[1-9]/) || (this.j.value = ""))
    };
    m.g.zr = function() {
        this.k.select();
        $k("URL-CLICK", this.C + "")
    };
    m.g.or = function() {
        var a = pp.getInstance();
        sp(a, this.S);
        sp(a, this.U);
        zD(this, this.F);
        this.F || (this.F = m.G("share-panel-email-container", this.da));
        m.K(this.F);
        !m.I(this.F, "disabled") && m.gj(this.F) && (this.X ? this.X.Ge() : (this.X = new QC(this.F, this.Zj, this.C, this.T, null), m.Ta(this, this.X)));
        $k("EMAIL", this.C + "")
    };
    m.g.qr = function() {
        var a = pp.getInstance();
        sp(a, this.S);
        sp(a, this.U);
        zD(this, this.D);
        this.D || (this.D = m.G("share-panel-gif-container", this.da));
        m.K(this.D);
        m.I(this.D, "disabled") ||!m.gj(this.D) || this.O || ((a = m.PC()) && a.pauseVideo && a.getPlayerState && 1 == a.getPlayerState() && a.pauseVideo(), this.O = new m.pD(this.D, 0, this.C), m.Ta(this, this.O));
        $k("GIF", this.C + "")
    };
    m.g.rr = function() {
        var a = m.PC();
        a && a.pauseVideo && a.pauseVideo();
        var a = m.rc("https://talkgadget.google.com/hangouts", {
            hl: this.gx,
            authuser: this.Zj,
            eid: this.Vn,
            gid: "youtube",
            gd: this.C,
            hs: 5
        }), b = window.screen.height, c = Math.min(.9 * window.screen.width, 1E3), b = Math.min(.9 * b, 800);
        $k("HANGOUT", this.C + "");
        Jk(a, {
            width: c,
            height: b
        })
    };
    m.g.pr = function() {
        var a = pp.getInstance();
        sp(a, this.S);
        sp(a, this.U);
        zD(this, this.H);
        this.H || (this.H = m.G("share-panel-embed-container", this.da));
        m.K(this.H);
        !m.I(this.H, "disabled") && m.gj(this.H) && (this.ba ? this.ba.Ge() : (this.ba = new SC(this.H, this.C, this.T), m.Ta(this, this.ba)));
        $k("EMBED", this.C + "")
    };
    m.g.yr = function() {
        sp(pp.getInstance(), this.U);
        zD(this, this.L)
    };
    m.g.sr = function() {
        sp(pp.getInstance(), this.S);
        zD(this, this.L)
    };
    m.g.wr = function() {
        zD(this, this.L)
    };
    m.g.ur = function() {
        zD(this, this.L);
        this.L && m.K(this.L)
    };
    m.g.Cr = function(a) {
        BD(this);
        var b=!1;
        a.target.checked && (b = m.qi(m.F("share-with-playlist-first"), "li"), b = m.A(b, "yt-uix-button-menu-item-selected"));
        CD(this, a.target.checked, b)
    };
    m.g.Ar = function() {
        AD(this);
        CD(this, !0)
    };
    m.g.Br = function() {
        AD(this);
        CD(this, !0, !0)
    };
    var FD, PD, ED=!1, ID = [], JD = [], GD = {
        hide: HD,
        "hide-badge-enclosing-action": function(a) {
            OD(a);
            m.G("yt-uix-menu-show-badge-enclosing-action", m.sl(a));
            a = m.Xp(a);
            var b = m.L(a, "feed-item-container");
            b && ((a = m.G("badge-action-container", b)) && m.P(a), m.z("yt-dom-content-change", b))
        },
        "show-badge-enclosing-action": function(a) {
            OD(a);
            var b = m.G("yt-uix-menu-show-badge-enclosing-action", m.sl(a));
            a = m.Xp(a);
            var c = m.L(a, "feed-item-container");
            c && ((a = m.G("badge-action-container", c)) && m.K(a), a && b && (m.Ub(a), m.zi(a, b)), m.z("yt-dom-content-change",
            c))
        },
        "replace-enclosing-action": function(a) {
            OD(a);
            var b = m.G("yt-uix-menu-replace-enclosing-action-notification", m.sl(a)), c, d;
            c = m.Xp(a);
            a = m.L(c, "yt-lockup");
            var e = m.L(c, "feed-item-container");
            if (a)
                c = m.G("yt-lockup-notifications-container", a), d = m.G("yt-lockup-dismissable", a);
            else if (e)
                c = m.G("feed-item-dismissal-notices", e), d = m.G("feed-item-dismissable", e);
            else 
                return;
            c && m.K(c);
            c && b && (m.Ub(c), m.zi(c, b));
            d && ND(d);
            m.z("yt-dom-content-change", a || e)
        },
        "never-show": function(a) {
            PD = a;
            var b = m.L(a, "feed-item-action-menu");
            if (a = m.I(a, "sub-action"))
                a = m.S(m.kq.getInstance(), a), b = m.G(a, b), m.kq.getInstance().show(b)
        },
        uploads: function(a) {
            var b = m.I(a, "channel-key");
            if (b) {
                var c = So(m.Oo.getInstance(), a.parentElement);
                m.z("subscription-pref-uploads", b, c);
                a = m.L(a, "feed-item-container");
                MD(a);
                b = m.qi(a, "li");
                b = c ? m.G("feed-item-dismissal-uploads", b) : m.G("feed-item-dismissal-all-activity", b);
                m.K(b);
                RD(a, function(a) {
                    a = m.G("feed-item-action-menu", a);
                    a = m.G("uploads", a).parentElement;
                    To(m.Oo.getInstance(), a, c)
                })
            }
        },
        subscribe: function(a) {
            var b =
            m.I(a, "sessionlink"), c = m.L(a, "feed-item-container"), c = m.I(c, "channel-key"), d = QD(a);
            m.Dq(rE, new ao(c, d, a, b))
        },
        unsubscribe: function(a) {
            var b = m.I(a, "sessionlink"), c = QD(a), d = m.I(a, "channel-key");
            d && m.Dq(sE, new bo(d, null, c, a, b))
        },
        "email-on-upload": function(a) {
            var b = m.I(a, "channel-key");
            if (b) {
                var c = So(m.Oo.getInstance(), a.parentElement);
                m.z("subscription-pref-email", b, c);
                a = m.L(a, "feed-item-container");
                MD(a);
                b = c ? m.G("feed-item-dismissal-email-on-upload", a) : m.G("feed-item-dismissal-no-email", a);
                m.K(b);
                RD(a,
                function(a) {
                    a = m.G("feed-item-action-menu", a);
                    a = m.G("email-on-upload", a).parentElement;
                    To(m.Oo.getInstance(), a, c)
                })
            }
        }
    };
    m.l("yt.www.comments.PlusOnePromo.dismiss", function() {
        var a = m.F("yt-comments-dftpop");
        a && (m.ri(a), m.R("/watch_actions_ajax", {
            method: "POST",
            format: "JSON",
            Z: {
                action_dismiss_plus_one_promo: "1"
            },
            V: {
                session_token: m.Kl()
            }
        }))
    }, void 0);
    var WD;
    var aE;
    m.bE = [];
    var eE = null, hE = [];
    var mE = null, oE = [];
})(_yt_www);

