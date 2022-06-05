/* JS */
gapi.loaded_1(function(_) {
    var window = this;
    var rs, ss, ts, vs, ws;
    _.ps = function(a) {
        return "rtl" == _.$r(a, "direction")
    };
    _.qs = function(a, b, c, d) {
        this.top = a;
        this.right = b;
        this.bottom = c;
        this.left = d
    };
    _.h = _.qs.prototype;
    _.h.Rb = function() {
        return this.right - this.left
    };
    _.h.Jb = function() {
        return this.bottom - this.top
    };
    _.h.clone = function() {
        return new _.qs(this.top, this.right, this.bottom, this.left)
    };
    _.h.contains = function(a) {
        return this && a ? "undefined" != typeof _.qs && a instanceof _.qs ? a.left >= this.left && a.right <= this.right && a.top >= this.top && a.bottom <= this.bottom : a.x >= this.left && a.x <= this.right && a.y >= this.top && a.y <= this.bottom : !1
    };
    _.h.ceil = function() {
        this.top = Math.ceil(this.top);
        this.right = Math.ceil(this.right);
        this.bottom = Math.ceil(this.bottom);
        this.left = Math.ceil(this.left);
        return this
    };
    _.h.floor = function() {
        this.top = Math.floor(this.top);
        this.right = Math.floor(this.right);
        this.bottom = Math.floor(this.bottom);
        this.left = Math.floor(this.left);
        return this
    };
    _.h.round = function() {
        this.top = Math.round(this.top);
        this.right = Math.round(this.right);
        this.bottom = Math.round(this.bottom);
        this.left = Math.round(this.left);
        return this
    };
    _.h.translate = function(a, b) {
        a instanceof _.Hr ? (this.left += a.x, this.right += a.x, this.top += a.y, this.bottom += a.y) : (this.left += a, this.right += a, _.Vc(b) && (this.top += b, this.bottom += b));
        return this
    };
    rs = function(a, b) {
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
    ss = function(a, b) {
        var c = _.Zr(a, b);
        return c ? rs(a, c) : 0
    };
    ts = {
        thin: 2,
        medium: 4,
        thick: 6
    };
    _.us = function(a, b) {
        if (_.M) {
            var c = ss(a, b + "Left"), d = ss(a, b + "Right"), e = ss(a, b + "Top"), f = ss(a, b + "Bottom");
            return new _.qs(e, d, f, c)
        }
        c = _.Yr(a, b + "Left");
        d = _.Yr(a, b + "Right");
        e = _.Yr(a, b + "Top");
        f = _.Yr(a, b + "Bottom");
        return new _.qs((0, window.parseFloat)(e), (0, window.parseFloat)(d), (0, window.parseFloat)(f), (0, window.parseFloat)(c))
    };
    vs = function(a, b) {
        if ("none" == _.Zr(a, b + "Style"))
            return 0;
        var c = _.Zr(a, b + "Width");
        return c in ts ? ts[c] : rs(a, c)
    };
    ws = function(a, b) {
        return new _.Hr(a.x - b.x, a.y - b.y)
    };
    _.xs = function(a, b, c, d) {
        this.left = a;
        this.top = b;
        this.width = c;
        this.height = d
    };
    _.xs.prototype.clone = function() {
        return new _.xs(this.left, this.top, this.width, this.height)
    };
    _.ys = function(a) {
        return new _.qs(a.top, a.left + a.width, a.top + a.height, a.left)
    };
    _.h = _.xs.prototype;
    _.h.contains = function(a) {
        return "undefined" != typeof _.xs && a instanceof _.xs ? this.left <= a.left && this.left + this.width >= a.left + a.width && this.top <= a.top && this.top + this.height >= a.top + a.height : a.x >= this.left && a.x <= this.left + this.width && a.y >= this.top && a.y <= this.top + this.height
    };
    _.h.ceil = function() {
        this.left = Math.ceil(this.left);
        this.top = Math.ceil(this.top);
        this.width = Math.ceil(this.width);
        this.height = Math.ceil(this.height);
        return this
    };
    _.h.floor = function() {
        this.left = Math.floor(this.left);
        this.top = Math.floor(this.top);
        this.width = Math.floor(this.width);
        this.height = Math.floor(this.height);
        return this
    };
    _.h.round = function() {
        this.left = Math.round(this.left);
        this.top = Math.round(this.top);
        this.width = Math.round(this.width);
        this.height = Math.round(this.height);
        return this
    };
    _.h.translate = function(a, b) {
        a instanceof _.Hr ? (this.left += a.x, this.top += a.y) : (this.left += a, _.Vc(b) && (this.top += b));
        return this
    };
    _.zs = function(a) {
        return _.us(a, "padding")
    };
    _.As = function(a) {
        if (_.M&&!_.je(9)) {
            var b = vs(a, "borderLeft"), c = vs(a, "borderRight"), d = vs(a, "borderTop");
            a = vs(a, "borderBottom");
            return new _.qs(d, c, a, b)
        }
        b = _.Yr(a, "borderLeftWidth");
        c = _.Yr(a, "borderRightWidth");
        d = _.Yr(a, "borderTopWidth");
        a = _.Yr(a, "borderBottomWidth");
        return new _.qs((0, window.parseFloat)(d), (0, window.parseFloat)(c), (0, window.parseFloat)(a), (0, window.parseFloat)(b))
    };
    _.Bs = function(a, b) {
        return (b & 4 && _.ps(a) ? b^2 : b)&-5
    };
    _.Cs = function(a, b, c) {
        var d, e = _.Td && (_.Md || _.$d) && _.he("1.9");
        b instanceof _.Hr ? (d = b.x, b = b.y) : (d = b, b = c);
        a.style.left = _.hs(d, e);
        a.style.top = _.hs(b, e)
    };
    _.Ds = function(a) {
        for (var b = new _.qs(0, window.Infinity, window.Infinity, 0), c = _.Or(a), d = c.A.body, e = c.A.documentElement, f = _.Fr(c.A); a = _.cs(a);)
            if (!(_.M && 0 == a.clientWidth || _.Ud && 0 == a.clientHeight && a == d) && a != d && a != e && "visible" != _.$r(a, "overflow")) {
                var g = _.ds(a), k;
                k = a;
                if (_.Td&&!_.he("1.9")) {
                    var l = (0, window.parseFloat)(_.Yr(k, "borderLeftWidth"));
                    if (_.ps(k))
                        var m = k.offsetWidth - k.clientWidth - l - (0, window.parseFloat)(_.Yr(k, "borderRightWidth")), l = l + m;
                        k = new _.Hr(l, (0, window.parseFloat)(_.Yr(k, "borderTopWidth")))
                    } else 
                        k =
                        new _.Hr(k.clientLeft, k.clientTop);
                        g.x += k.x;
                        g.y += k.y;
                        b.top = Math.max(b.top, g.y);
                        b.right = Math.min(b.right, g.x + a.clientWidth);
                        b.bottom = Math.min(b.bottom, g.y + a.clientHeight);
                        b.left = Math.max(b.left, g.x)
            }
        d = f.scrollLeft;
        f = f.scrollTop;
        b.left = Math.max(b.left, d);
        b.top = Math.max(b.top, f);
        c = _.Mr(c.Ja());
        b.right = Math.min(b.right, d + c.width);
        b.bottom = Math.min(b.bottom, f + c.height);
        return 0 <= b.top && 0 <= b.left && b.bottom > b.top && b.right > b.left ? b : null
    };
    _.Es = function(a) {
        var b = _.ds(a);
        a = _.ls(a);
        return new _.xs(b.x, b.y, a.width, a.height)
    };
    _.Fs = function(a) {
        if (1 == a.nodeType)
            return _.es(a);
        var b = _.Uc(a.pE), c = a;
        a.targetTouches && a.targetTouches.length ? c = a.targetTouches[0] : b && a.A.targetTouches && a.A.targetTouches.length && (c = a.A.targetTouches[0]);
        return new _.Hr(c.clientX, c.clientY)
    };
    _.Gs = function(a, b, c, d, e, f, g, k, l) {
        var m, n;
        if (m = c.offsetParent) {
            var p = "HTML" == m.tagName || "BODY" == m.tagName;
            p && "static" == _.as(m) || (n = _.ds(m), p || (p = (p = _.ps(m)) && _.Td?-m.scrollLeft : !p || _.M && _.he("8") || "visible" == _.$r(m, "overflowX") ? m.scrollLeft : m.scrollWidth - m.clientWidth - m.scrollLeft, n = ws(n, new _.Hr(p, m.scrollTop))))
        }
        m = n || new _.Hr;
        n = _.Es(a);
        if (p = _.Ds(a)) {
            var v = new _.xs(p.left, p.top, p.right - p.left, p.bottom - p.top), p = Math.max(n.left, v.left), x = Math.min(n.left + n.width, v.left + v.width);
            if (p <= x) {
                var t = Math.max(n.top,
                v.top), v = Math.min(n.top + n.height, v.top + v.height);
                t <= v && (n.left = p, n.top = t, n.width = x - p, n.height = v - t)
            }
        }
        p = _.Or(a);
        t = _.Or(c);
        p.A != t.A && (x = p.A.body, t = _.fs(x, t.Ja()), t = ws(t, _.ds(x)), !_.M || _.je(9) || _.Lr(p) || (t = ws(t, _.Kr(p))), n.left += t.x, n.top += t.y);
        a = _.Bs(a, b);
        b = new _.Hr(a & 2 ? n.left + n.width : n.left, a & 1 ? n.top + n.height : n.top);
        b = ws(b, m);
        e && (b.x += (a & 2?-1 : 1) * e.x, b.y += (a & 1?-1 : 1) * e.y);
        var w;
        if (g)
            if (l)
                w = l;
            else if (w = _.Ds(c))
                w.top -= m.y, w.right -= m.x, w.bottom -= m.y, w.left -= m.x;
        e = w;
        l = b.clone();
        w = _.Bs(c, d);
        d = _.ls(c);
        a = k ?
        k.clone() : d.clone();
        k = l;
        l = a;
        k = k.clone();
        l = l.clone();
        a = 0;
        if (f || 0 != w)
            w & 2 ? k.x -= l.width + (f ? f.right : 0) : f && (k.x += f.left), w & 1 ? k.y -= l.height + (f ? f.bottom : 0) : f && (k.y += f.top);
        g && (e ? (f = k, w = l, a = 0, 65 == (g & 65) && (f.x < e.left || f.x >= e.right) && (g&=-2), 132 == (g & 132) && (f.y < e.top || f.y >= e.bottom) && (g&=-5), f.x < e.left && g & 1 && (f.x = e.left, a|=1), f.x < e.left && f.x + w.width > e.right && g & 16 && (w.width = Math.max(w.width - (f.x + w.width - e.right), 0), a|=4), f.x + w.width > e.right && g & 1 && (f.x = Math.max(e.right - w.width, e.left), a|=1), g & 2 && (a = a | (f.x < e.left ?
        16 : 0) | (f.x + w.width > e.right ? 32 : 0)), f.y < e.top && g & 4 && (f.y = e.top, a|=2), f.y <= e.top && f.y + w.height < e.bottom && g & 32 && (w.height = Math.max(w.height - (e.top - f.y), 0), f.y = e.top, a|=8), f.y >= e.top && f.y + w.height > e.bottom && g & 32 && (w.height = Math.max(w.height - (f.y + w.height - e.bottom), 0), a|=8), f.y + w.height > e.bottom && g & 4 && (f.y = Math.max(e.bottom - w.height, e.top), a|=2), g & 8 && (a = a | (f.y < e.top ? 64 : 0) | (f.y + w.height > e.bottom ? 128 : 0)), g = a) : g = 256, a = g);
        f = new _.xs(0, 0, 0, 0);
        f.left = k.x;
        f.top = k.y;
        f.width = l.width;
        f.height = l.height;
        g = a;
        g & 496 ||
        (_.Cs(c, new _.Hr(f.left, f.top)), a = new _.Gr(f.width, f.height), d == a || d && a && d.width == a.width && d.height == a.height || (f = a, d = _.Lr(_.Or(_.wg(c))), !_.M || _.he("10") || d && _.he("8") ? (c = c.style, _.Td ? c.MozBoxSizing = "border-box" : _.Ud ? c.WebkitBoxSizing = "border-box" : c.boxSizing = "border-box", c.width = Math.max(f.width, 0) + "px", c.height = Math.max(f.height, 0) + "px") : (k = c.style, d ? (d = _.zs(c), c = _.As(c), k.pixelWidth = f.width - c.left - d.left - d.right - c.right, k.pixelHeight = f.height - c.top - d.top - d.bottom - c.bottom) : (k.pixelWidth = f.width,
        k.pixelHeight = f.height))));
        return g
    };
    _.Hs = function(a, b, c) {
        c || (a = a.parentNode);
        for (c = 0; a;) {
            if (b(a))
                return a;
            a = a.parentNode;
            c++
        }
        return null
    };
    _.Is = function(a, b) {
        a.style.display = b ? "" : "none"
    };
    _.Js = function(a, b) {
        var c = a.style;
        "opacity"in c ? c.opacity = b : "MozOpacity"in c ? c.MozOpacity = b : "filter"in c && (c.filter = "" === b ? "" : "alpha(opacity=" + 100 * b + ")")
    };
    _.Ks = function(a, b) {
        b ? a.tabIndex = 0 : (a.tabIndex =- 1, a.removeAttribute("tabIndex"))
    };
    _.Ls = function(a, b) {
        var c = {}, d;
        for (d in a)
            b.call(void 0, a[d], d, a) && (c[d] = a[d]);
        return c
    };

    _.Ms = function(a) {
        a = a.className;
        return _.ja(a) && a.match(/\S+/g) || []
    };
    _.Ns = function() {
        var a = window, b = a.document, c = 0;
        if (b) {
            var c = b.body, d = b.documentElement;
            if (!d ||!c)
                return 0;
            a = _.Jr(a).height;
            if (_.Er(b) && d.scrollHeight)
                c = d.scrollHeight != a ? d.scrollHeight : d.offsetHeight;
            else {
                var b = d.scrollHeight, e = d.offsetHeight;
                d.clientHeight != e && (b = c.scrollHeight, e = c.offsetHeight);
                c = b > a ? b > e ? b : e : b < e ? b : e
            }
        }
        return c
    };
    _.Os = function(a) {
        this.S = a;
        this.A = a.kb()
    };
    _.Os.prototype.wa = function() {
        Ps(this)
    };
    _.Os.prototype.onBeforeParentOpen = _.Os.prototype.wa;
    var Ps = function(a) {
        var b = a.S.Kb();
        if (a.A.anchorBox && b && b.Fa())
            b = _.ds(b.Fa()), a.A.anchorBox.left += b.x, a.A.anchorBox.top += b.y;
        else {
            b = a.A.anchor;
            if ("_default" != b && "_iframe" != b) {
                var c = _.Sr(b);
                if (c)
                    a.A.anchorBox = _.Es(c);
                else {
                    _.Ib("Anchor not found in DOM: " + b + '. Falling back to "_default".');
                    a.A.anchor = "_default";
                    return 
                }
            }
            "_iframe" == b && (b = _.Mr(), a.A.anchorBox = new _.xs(0, 0, b.width, b.height))
        }
        a.A.anchor = ""
    };
    _.Qs = function(a) {
        _.Os.call(this, a)
    };
    _.u(_.Qs, _.Os);
    _.Qs.prototype.open = function() {
        var a = this.A, b = window.document.createElement("ins");
        window.document.getElementById(a.container).appendChild(b);
        b.style.display = "block";
        _.Q(b, a.containerCss);
        this.S.Ud(b);
        this.S.Gd(b)
    };
    _.Qs.prototype.kc = function() {
        window.document.getElementById(this.S.id).style.height = this.S.height + "px"
    };
    _.Qs.prototype.close = function() {
        _.Ag(this.S.Pa());
        _.Ag(this.V);
        this.V = null
    };
    _.Ts = function(a) {
        if (a.V)
            return a.V;
        var b = a.A;
        !b.anchorBox && b.anchor && Ps(a);
        var c = a.S.Kb();
        if ("_default" == b.anchor && c) {
            var d = _.Es(_.Sr(c.Pa()));
            b.anchorBox = d
        }
        if (!b.anchorBox)
            return _.Ib("No anchor box defined."), null;
        b = new _.Hr(b.anchorBox.left, b.anchorBox.top);
        c && (c = _.fs(c.Fa(), window), a.R = new _.Hr, a.R.x = c.x, a.R.y = c.y, b.x += c.x, b.y += c.y, _.Rs(b));
        a.da = b;
        c = _.Ss(a, !0);
        a.V = window.document.createElement("ins");
        a.V.style.cssText = c;
        window.document.body.appendChild(a.V);
        return a.V
    };
    _.Ss = function(a, b) {
        var c = a.A;
        return "position: absolute !important;background-color: transparent !important;left: " + a.da.x + "px !important;top: " + a.da.y + "px !important;width: " + c.anchorBox.width + "px !important;height: " + c.anchorBox.height + "px !important;z-index: -10000 !important;display: " + (b ? "block" : "none") + " !important;"
    };
    _.Us = function(a, b) {
        var c = 0, d = 0;
        if (b.pageX || b.pageY)
            c = b.pageX, d = b.pageY;
        else if (b.clientX || b.clientY) {
            var c = b.target ? b.target: b.srcElement, e;
            c.ownerDocument && c.ownerDocument.parentWindow ? e = c.ownerDocument.parentWindow : e = window;
            d = c = 0;
            _.M ? (c = e.document.documentElement.scrollLeft, d = e.document.documentElement.scrollTop) : (c = e.pageXOffset, d = e.pageYOffset);
            c = b.clientX + c;
            d = b.clientY + d
        }
        e = new _.Hr(c, d);
        return (c = _.ys(_.Es(a))) && c.contains(e)
    };
    _.Rs = function(a) {
        var b = window, c = window.document.body, d = _.ds(c), b = c.currentStyle || b.getComputedStyle(c, "");
        b.position && "static" != b.position && (a.x -= d.x, a.y -= d.y)
    };
    _.Vs = function(a) {
        var b = a.S.Kb() && a.S.Kb().Pa(), b = b && b.style.zIndex ? (0, window.parseInt)(b.style.zIndex, 10) + 1: 0;
        return Math.min(2147483647, Math.max(2E9, a.A.zIndex || b))
    };
    var Ws, Xs, Ys;
    Ws = {
        "bottom-center": 1,
        "bottom-end": 7,
        "bottom-left": 1,
        "bottom-right": 3,
        "bottom-start": 5,
        "left-bottom": 1,
        "left-center": 0,
        "left-top": 0,
        "right-bottom": 3,
        "right-center": 2,
        "right-top": 2,
        "top-center": 0,
        "top-end": 6,
        "top-left": 0,
        "top-right": 2,
        "top-start": 4
    };
    Xs = {
        "bottom-center": !0,
        "top-center": !0
    };
    Ys = {
        "left-center": !0,
        "right-center": !0
    };
    _.Zs = function(a, b, c, d, e) {
        e = e || {
            x: 0,
            y: 0
        };
        if (Xs[b]) {
            var f = _.ls(a).width / 2;
            e.x = "top-right" == d || "bottom-right" == d ? e.x + f : e.x - f
        }
        Xs[d] && (f = _.ls(c).width / 2, e.x += f);
        Ys[b] && (f = _.ls(a).height / 2, e.y = "right-bottom" == d || "left-bottom" == d ? e.y + f : e.y - f);
        Ys[d] && (e.y += _.ls(c).height / 2);
        _.Gs(c, Ws[d], a, Ws[b], new _.Hr(e.x, e.y));
        d = _.Hs(a, function(a) {
            if (a == window.document)
                return !1;
            a = _.as(a);
            return !!a && "static" != a
        });
        c = b = 0;
        d && (c = _.ds(d), b =- c.x, c =- c.y);
        a = a.style;
        (0, window.parseInt)(a.left, 10) < b && (a.left = b + "px");
        (0, window.parseInt)(a.top, 10) < c && (a.top = c + "px")
    };
    _.$s = function(a) {
        _.Os.call(this, a.S);
        this.G = a;
        this.B = null
    };
    _.u(_.$s, _.Qs);
    _.$s.prototype.wa = function() {
        this.G.wa()
    };
    _.$s.prototype.onBeforeParentOpen = _.$s.prototype.wa;
    _.$s.prototype.open = function() {
        this.G.open();
        if (this.A.closeClickDetection || this.A.hideClickDetection)
            this.B = _.N(window.document, ["click", "touchstart"], (0, _.s)(this.C, this), !1)
    };
    _.$s.prototype.open = _.$s.prototype.open;
    _.$s.prototype.kc = function(a) {
        this.G.kc(a)
    };
    _.$s.prototype.onready = _.$s.prototype.kc;
    _.$s.prototype.D = function(a) {
        this.G.onRenderStart && this.G.onRenderStart(a)
    };
    _.$s.prototype.onRenderStart = _.$s.prototype.D;
    _.$s.prototype.close = function() {
        if (this.A.closeClickDetection || this.A.hideClickDetection)
            _.af(this.B), this.B = null;
        this.G.close()
    };
    _.$s.prototype.close = _.$s.prototype.close;
    _.$s.prototype.C = function(a) {
        _.Us(this.S.Pa(), a) || (this.A.hideClickDetection && this.G.show ? this.G.show(!1) : this.G.close())
    };

    _.QC = function(a) {
        _.Os.call(this, a.S);
        this.G = a;
        this.B = null
    };
    _.u(_.QC, _.Qs);
    _.QC.prototype.open = function() {
        this.A.targetPos = this.A.targetPos || "top-start";
        this.A.anchorPos = this.A.anchorPos || "bottom-start";
        var a = _.Ts(this), b = this.S.Pa();
        b ? (b.style.visibility = "hidden", b.style.position = "absolute", a.parentNode.appendChild(b)) : this.G.open()
    };
    _.QC.prototype.open = _.QC.prototype.open;
    _.QC.prototype.kc = function() {
        if (this.A.closeClickDetection) {
            var a = this, b = function(b) {
                _.Us(a.S.Pa(), b) || (a.S.close(), a.B = null)
            };
            window.document.A ? (window.document.A("click", b), this.B = function() {
                window.document.removeEventListener("click", b, !1)
            }) : window.document.attachEvent && (window.document.attachEvent("onclick", b), this.B = function() {
                window.document.detachEvent("onclick", b)
            })
        }
        var c = this.S.Pa();
        window.document.getElementById(this.S.id).style.height = this.S.height + "px";
        var d = _.Ts(this);
        d && _.Zs(c, this.A.targetPos, d, this.A.anchorPos, {
            x: this.A.leftOffset || 0,
            y: this.A.topOffset || 0
        });
        c.style.visibility = "visible"
    };
    _.QC.prototype.onready = _.QC.prototype.kc;
    _.QC.prototype.close = function() {
        this.G.close();
        this.B && this.B()
    };
    _.QC.prototype.close = _.QC.prototype.close;
    var SC;
    _.RC = function(a) {
        _.Os.call(this, a)
    };
    _.u(_.RC, _.Qs);
    SC = ["transition", "WebkitTransition", "MozTransition", "OTranstion", "msTransition"];
    _.TC = function(a, b) {
        a.B = window.document.createElement("div");
        var c = {
            position: "absolute",
            top: "-10000px",
            zIndex: _.Vs(a)
        };
        a.A.width && (c.width = a.A.width + "px");
        for (var d in c)
            a.B.style[d] = c[d];
        (b || window.document.body).appendChild(a.B)
    };
    _.RC.prototype.open = function(a) {
        this.S.Nc("updateContainer", (0, _.s)(this.C, this));
        _.TC(this, a);
        this.S.Gd(this.B);
        this.S.Ud(this.B)
    };
    _.RC.prototype.open = _.RC.prototype.open;
    _.RC.prototype.kc = function() {
        var a = UC(this);
        if (a) {
            var b = window.document.getElementById(this.S.ca());
            b.style.height = a.height + "px";
            b.style.width = a.width + "px";
            this.S.width = a.width;
            this.S.height = a.height;
            b.style.boxShadow = "0 4px 16px rgba(0, 0, 0, 0.3)"
        }
    };
    _.RC.prototype.onready = _.RC.prototype.kc;
    _.RC.prototype.close = function() {
        this.B.parentNode && this.B.parentNode.removeChild(this.B)
    };
    _.RC.prototype.close = _.RC.prototype.close;
    _.RC.prototype.C = function(a, b, c) {
        var d = this.S.Pa();
        d && (a ? (VC(this, b, c), d.style.opacity = 0, d.style.display = "", window.setTimeout((0, _.s)(function() {
            WC(d, !0);
            d.style.opacity = 1
        }, this), 0)) : (d.style.display = "none", WC(d, !1), d.style.opacity = 0))
    };
    var WC = function(a, b) {
        for (var c = 0; c < SC.length; c++)
            a.style[SC[c]] = b ? "opacity .13s linear" : ""
    }, VC = function(a, b, c) {
        var d = a.S.Kb();
        b += 10;
        c += 10;
        if (_.K.ta && (d && (d = _.fs(d.Pa(), window), b += d.x, c += d.y), d = UC(a))) {
            var e = _.Mr(window), f = _.Ir(window.document);
            e.width && (b = Math.min(b, e.width + f.x - d.width - 8), c + d.height > e.height + f.y - 8 && (c -= 20 + d.height, c = Math.max(c, Math.min(f.y + 1, _.Ns() - d.height))))
        }
        a = a.S.Pa();
        a.style.left = b + "px";
        a.style.top = c + "px"
    }, UC = function(a) {
        return a.S.width && a.S.height ? {
            width: a.S.width,
            height: a.S.height
        } : (a = a.S.Fa()) && a.offsetWidth && a.offsetHeight ? {
            width: a.offsetWidth,
            height: a.offsetHeight
        } : null
    };
    _.XC = function(a) {
        _.Os.call(this, a)
    };
    _.u(_.XC, _.Qs);
    _.XC.prototype.C = null;
    _.XC.prototype.D = null;
    _.XC.prototype.B = null;
    var YC = {
        end: "start",
        left: "right",
        right: "left",
        start: "end"
    }, ZC = {
        top: "bottom",
        bottom: "top"
    };
    _.XC.prototype.open = function() {
        var a = window.document.createElement("div");
        _.Q(a, {
            top: "-10000px",
            position: "absolute",
            zIndex: _.Vs(this)
        });
        _.Ts(this).parentNode.appendChild(a);
        this.S.Ud(a);
        this.S.Gd(a)
    };
    _.XC.prototype.open = _.XC.prototype.open;
    _.XC.prototype.kc = function() {
        var a = window.document.getElementById(this.S.id);
        a.style.height = this.S.height + "px";
        a.style.width = this.S.width + "px";
        a.style.boxShadow = "0 4px 16px rgba(0, 0, 0, 0.3)";
        var b = this.S.Pa();
        b.style.lineHeight = 0;
        var c = _.Ts(this), d = this.A.targetPos || "top-start", e = this.A.anchorPos || "bottom-start", f = this.A.leftOffset || 0, g = this.A.topOffset || 0;
        _.Zs(b, d, c, e, {
            x: f,
            y: g
        });
        var k = _.Mr(window), l = _.Ir(window.document), m = b.offsetLeft < l.x || b.offsetLeft + b.offsetWidth > k.width + l.x, k = b.offsetTop <
        l.y || b.offsetTop + b.offsetHeight > k.height + l.y, d = $C(d, m, k), e = $C(e, m, k);
        _.Zs(b, d, c, e, {
            x: f * (m?-1 : 1),
            y: g * (k?-1 : 1)
        });
        b.style.visibility = "visible";
        this.C = _.N(window.document, "mouseover", (0, _.s)(function(b) {
            b.target === a && this.B && (window.clearTimeout(this.B), this.B = null)
        }, this));
        this.D = _.N(window.document, "mouseout", (0, _.s)(function(b) {
            b.target === a && (this.B = window.setTimeout((0, _.s)(this.S.close, this.S), 1E3))
        }, this))
    };
    _.XC.prototype.onready = _.XC.prototype.kc;
    var $C = function(a, b, c) {
        a = a.split("-");
        for (var d = 0; 2 > d; d++)
            b && YC[a[d]] && (a[d] = YC[a[d]]), c && ZC[a[d]] && (a[d] = ZC[a[d]]);
        return a.join("-")
    };
    _.XC.prototype.close = function() {
        this.C && (_.af(this.C), _.af(this.D), this.D = this.C = null);
        this.B && (window.clearTimeout(this.B), this.B = null);
        _.XC.J.close.call(this)
    };
    _.XC.prototype.close = _.XC.prototype.close;

    var pN, qN, rN;
    for (_.oN = function(a) {
        return {
            kb: function() {
                return a
            },
            Kb: function() {
                return a.openerIframe
            }
        }
    }, pN = function(a) {
        (new _.Os(_.oN(a))).wa()
    }, qN = "bubble circlepicker float hover hover-menu slide-menu".split(" "), rN = 0; rN < qN.length; rN++)
        _.Mm[qN[rN]] = pN;

    _.Pm.hover = function(a) {
        var b = new _.RC(_.oN(a));
        _.TC(b, a.where);
        a.where = b.B;
        a.onClose = function() {
            b.close()
        };
        a.onRestyle = function(a) {
            if (a.updateContainer) {
                var d = a.updateContainer;
                b.C(d.visible, d.x, d.y)
            }
            a.width && (b.S.width = a.width);
            a.height && (b.S.height = a.height)
        };
        a.onCreate = function(c) {
            b.S = c;
            c.Kb = function() {
                return a.openerIframe
            };
            c.register("_ready", (0, _.s)(b.kc, b), _.Am);
            c.updateContainer = function(a, c, f) {
                b.C(a, c, f)
            }
        }
    };

    var gH = function(a, b) {
        this.A = a;
        this.yg = b
    };
    gH.prototype.bn = function() {
        this.A.oc(this.yg)
    };
    _.hH = function(a) {
        this.w = this.ib = a;
        this.n = null;
        this.slf = 0;
        this.ssh=!1;
        this.sen=!0;
        this.shl = this.itm = null
    };
    _.hH.prototype.sm = function(a, b) {
        this.ib.Ni(new gH(a, this));
        this.n = b
    };
    _.hH.prototype.sh = function() {
        this.ib.show()
    };
    _.hH.prototype.hi = function() {
        this.ib.rb()
    };
    _.hH.prototype.cl = function() {
        this.ib.close()
    };
    _.hH.prototype.en = function() {
        this.ib.enable()
    };
    _.hH.prototype.di = function() {
        this.ib.disable()
    };
    _.hH.prototype.hl = function(a) {
        this.ib.JE(a)
    };
    _.hH.prototype.vr = function(a, b) {
        this.ib.nF(a, b)
    };

    _.pm.G("ytsubscribe", function(a, b) {
        var c = _.pm.B;
        c && _.pm.M({
            role: "ytsubscribe",
            iframe: c,
            data: b
        }, {
            role: "ytrelay",
            iframe: a
        })
    }, void 0, _.Am);
    var DN = function(a) {
        a.A.os(a.yg)
    }, EN = function(a) {
        this.yg = a;
        this.Vc = 0;
        this.Gk=!1;
        this.Wu=!0;
        this.ig = null
    };
    EN.prototype.Lf = function() {
        return this.yg
    };
    var FN = function(a) {
        return 5 == a.Vc || 4 == a.Vc
    };
    EN.prototype.isEnabled = function() {
        return this.Wu
    };
    EN.prototype.Na = function(a) {
        this.Wu = a
    };
    var GN = function(a, b, c) {
        this.A = a || {};
        this.B = b || 0;
        this.D = c || 0;
        a = {};
        b = (0, _.s)(this.pv, this);
        a.fc = b;
        b = (0, _.s)(this.zr, this);
        a.rc = b;
        b = (0, _.s)(this.ds, this);
        a.sc = b;
        b = (0, _.s)(this.jk, this);
        a.hc = b;
        b = (0, _.s)(this.xj, this);
        a.cc = b;
        b = (0, _.s)(this.Bw, this);
        a.os = b;
        b = (0, _.s)(this.Aw, this);
        a.or = b;
        b = (0, _.s)(this.yw, this);
        a.oh = b;
        b = (0, _.s)(this.ww, this);
        a.oc = b;
        b = (0, _.s)(this.xw, this);
        a.oe = b;
        b = (0, _.s)(this.zw, this);
        a.oi = b;
        this.C = a
    };
    _.h = GN.prototype;
    _.h.zr = function(a, b, c) {
        try {
            a = a + (null != b ? "_" + b : ""), c.sm(this.C, a), this.A[a] = new EN(c)
        } catch (d) {
            return !1
        }
        return !0
    };
    _.h.pv = function(a, b) {
        var c = this.A[a + (null != b ? "_" + b : "")];
        return c ? c.Lf() : null
    };
    _.h.ds = function(a) {
        var b = HN(this, a);
        if (b && (2 == b.Vc || 3 == b.Vc) && b.isEnabled()&&!b.Gk) {
            try {
                a.sh()
            } catch (c) {}
            b.Gk=!0
        }
    };
    _.h.jk = function(a) {
        var b = HN(this, a);
        if (b && (2 == b.Vc || 3 == b.Vc || FN(b)) && b.Gk) {
            try {
                a.hi()
            } catch (c) {}
            b.Gk=!1
        }
    };
    _.h.xj = function(a) {
        var b = HN(this, a);
        if (b && 5 != b.Vc) {
            try {
                this.jk(a), a.cl()
            } catch (c) {}
            IN(this, b)
        }
    };
    _.h.Bw = function(a) {
        (a = HN(this, a)) && 0 == a.Vc && (JN(this, a), a.Vc = 1)
    };
    var JN = function(a, b) {
        if (a.B) {
            var c = (0, window.setTimeout)((0, _.s)(function() {
                FN(b) || KN(this, b)
            }, a), a.B);
            b.ig = c
        } else 
            KN(a, b)
    }, KN = function(a, b) {
        var c = a.D - a.B;
        0 < c && (c = (0, window.setTimeout)((0, _.s)(function() {
            FN(b) || (b.Vc = 4, this.xj(b.Lf()))
        }, a), c), b.ig = c)
    }, LN = function(a) {
        null != a.ig && ((0, window.clearTimeout)(a.ig), a.ig = null)
    };
    _.h = GN.prototype;
    _.h.Aw = function(a) {
        (a = HN(this, a))&&!FN(a) && 1 == a.Vc && (LN(a), a.Vc = 3)
    };
    _.h.yw = function(a) {
        (a = HN(this, a))&&!FN(a) && (a.Gk=!1)
    };
    _.h.ww = function(a) {
        var b = HN(this, a);
        if (b&&!FN(b)) {
            try {
                this.jk(a)
            } catch (c) {}
            IN(this, b)
        }
    };
    _.h.xw = function(a) {
        (a = HN(this, a))&&!FN(a) && (a.Vc = 4, this.xj(a.Lf()))
    };
    _.h.zw = function(a, b) {
        var c = HN(this, a);
        c&&!FN(c) && 2 <= b && 4 >= b&&!FN(c) && (LN(c), c.Vc = 2)
    };
    var IN = function(a, b) {
        LN(b);
        b.Vc = 5;
        var c = a.A, d;
        for (d in c)
            c[d] == b && delete c[d]
    }, HN = function(a, b) {
        return a.A[b.n]
    }, MN = function() {
        GN.call(this)
    };
    _.u(MN, GN);
    var NN = function(a) {
        this.A = a
    };
    _.h = NN.prototype;
    _.h.zr = function(a, b, c) {
        return this.A.rc(a, b, c)
    };
    _.h.pv = function(a, b) {
        return this.A.fc(a, b)
    };
    _.h.ds = function(a) {
        this.A.sc(a)
    };
    _.h.jk = function(a) {
        this.A.hc(a)
    };
    _.h.xj = function(a) {
        this.A.cc(a)
    };
    _.h.Bw = function(a) {
        this.A.os(a)
    };
    _.h.Aw = function(a, b) {
        this.A.or(a, b)
    };
    _.h.yw = function(a) {
        this.A.oh(a)
    };
    _.h.ww = function(a) {
        this.A.oc(a)
    };
    _.h.xw = function(a, b, c, d, e, f) {
        this.A.oe(a, b, c, d, e, f)
    };
    _.h.zw = function(a, b, c, d) {
        this.A.oi(a, b, c, d)
    };
    var ON = function() {
        return window.___jsl.man
    }, PN = function(a) {
        if (window.___jsl.man)
            a(window.___jsl.man);
        else {
            var b = function() {
                var b = new MN;
                window.___jsl.man = b;
                a(b)
            };
            if (window.gbar) {
                var c = function() {
                    if (window.gbar.wg) {
                        var c = new NN(window.gbar.wg);
                        window.___jsl.man = c;
                        a(c)
                    } else 
                        b()
                };
                window.gbar.wg ? c() : window.gbar.qm(c)
            } else 
                b()
        }
    };
    var QN = _.xd("contactid", "cdu", "cmp", "email", "hl", "n", "m", "p", "src", "userid", "sp", "ytid");
    _.xd("hl", "origin", "ri", "src", "userid");
    var RN = _.xd("nm", "s", "pr", "v");
    _.Jd(RN, QN);
    var SN = /(?:^|\s)g-(?:hovercard|profile)(?:$|\s)/, TN = {
        loadHovercardDelay: 250,
        loadDelay: 150,
        hoverDelay: 500,
        closeDelay: 500
    }, UN = 0, VN = function(a) {
        return window.document.body == a ? "body" : a.__cardid || null
    }, WN = function(a) {
        var b = VN(a);
        b || (b = a.__cardid = UN++);
        return b
    }, XN = function(a) {
        var b = a.className || "getAttribute"in a && a.getAttribute("class");
        return b && SN.test(b) || "getAttribute"in a && a.getAttribute("oid") && 36 == _.E("card/p")?!0 : "G:HOVERCARD" == a.tagName.toUpperCase()
    }, YN = function(a, b) {
        var c = {};
        _.Jd(c, TN, _.E("iframes/card") ||
        {}, _.E("card") || {});
        for (var d = [], e = a; e; e = e.parentNode) {
            var f = VN(e);
            f && b[f] && d.push(b[f])
        }(0, _.dd)(d.reverse(), function(a) {
            _.Jd(c, a)
        });
        d = "G:HOVERCARD" == a.tagName.toUpperCase() ? "" : "data-";
        e = a.attributes;
        for (f = 0; f < e.length; f++)
            _.Wc(e[f].name, d) && (c[e[f].name.substring(d.length)] = e[f].value);
        "getAttribute"in a && a.getAttribute("oid") && 36 == _.E("card/p") && (c.ytid = a.getAttribute("oid"));
        !c.userid && "A" == a.tagName.toUpperCase() && a.pathname && (d = a.pathname.match(/^\/?(\d+)$/), /\.google\.com$/.test(a.hostname) && d && (c.userid = d[1]));
        c.hl || (c.hl = _.E("lang") || _.H.Fb().hl || void 0);
        c.m = c.entity;
        c.src = c.source;
        delete c.entity;
        delete c.source;
        return c
    }, ZN = function(a, b) {
        var c = b[a];
        _.Vc(c) || (c = TN[a]);
        return 0 > c ? 0 : c
    };
    var $N = function(a) {
        this.B = a;
        this.yg = new _.hH(this);
        this.C=!1;
        this.D = 0
    };
    _.h = $N.prototype;
    _.h.Lf = function() {
        return this.yg
    };
    _.h.load = function(a) {
        DN(this.vb);
        a = _.Ls(a, function(a, b) {
            return QN[b] && null != a
        });
        a.origin = window.location.protocol + "//" + window.location.host;
        var b = this, c = this.B, d = {
            _event: function(a) {
                if (!(a.timestamp < c.ia)) {
                    "sgcp_ams" == a.event ? (c.Y=!0, c.R=!1) : c.Y && "mouseover" == a.event ? c.R=!0 : c.Y && "mouseout" == a.event ? c.R=!1 : "sgcp_amh" == a.event && (c.Y=!1, c.R || aO(c));
                    if ("mouseover" == a.event || "sgcp_ams" == a.event)
                        window.clearTimeout(c.G), c.G = null;
                    if (a.cpid) {
                        for (var b = window.document.getElementById(a.cpid); b && "BODY" != b.parentNode.tagName;)
                            b =
                            b.parentNode;
                        c.ka = b
                    }
                    a.fromCard && "mouseout" == a.event && aO(c)
                }
            },
            _ready: (0, _.s)(this.kE, this),
            version: function(a) {
                bO(c, c.F, {
                    type: "circles_changed",
                    version: a.v
                })
            },
            loaded: function(a) {
                a.ri == b.D && b.G()
            },
            rendered: function() {
                var a = b.B.M, c = _.Kr(_.Or((window || _.r || window).document));
                cO(b, !0, a.x + c.x, a.y + c.y);
                a = b.B;
                bO(a, a.F, {
                    type: "show",
                    frame: b.A
                })
            },
            renderfailed: function() {
                cO(b, !1, 0, 0)
            },
            disposed: function() {
                b.A.close()
            },
            cardAction: function(a) {
                dO(c, a)
            }
        }, e = ":card";
        !_.E("iframes/card/url") && _.E("iframes/hovercard/url") &&
        (e = ":hovercard");
        a = _.em(_.fm(_.gm(_.Zm(_.$m(new _.dm({
            disableMultiLevelParentRelay: !0,
            hover: !0
        }), d), _.Am), a), "hover"), e);
        _.E("card/relayOpenTop") && (_.hm(a, - 1), _.cn(new _.cm(a.A), "_default"));
        _.pm.open(a.value(), (0, _.s)(function(a) {
            this.A = a
        }, this))
    };
    _.h.kE = function() {
        this.C=!0;
        var a = this.vb;
        a.A.or(a.yg, {});
        a = this.B;
        a.A && a.Sq(a.A)
    };
    _.h.Ni = function(a) {
        this.vb = a
    };
    _.h.JE = function(a) {
        this.A.send("getHealthc", void 0, a, _.Am)
    };
    _.h.nF = function(a, b) {
        this.A.send("getVarc", a, b, _.Am)
    };
    var cO = function(a, b, c, d) {
        a.A.updateContainer ? a.A.updateContainer(b, c, d) : a.A.Ji({
            updateContainer: {
                visible: b,
                x: c,
                y: d
            }
        })
    };
    _.h = $N.prototype;
    _.h.show = function() {
        cO(this, !0, 0, - 1E4);
        this.A.send("render", void 0, void 0, _.Am)
    };
    _.h.rb = function() {
        this.A.send("hide", void 0, void 0, _.Am);
        cO(this, !1, 0, 0);
        var a = this.B;
        bO(a, a.F, {
            type: "hide"
        });
        a.F = null;
        a.va = null
    };
    _.h.close = function() {
        this.A.send("dispose", void 0, void 0, _.Am)
    };
    _.h.enable = function() {};
    _.h.disable = function() {};
    var eO = function() {
        this.Ia = 0;
        this.Sa = [];
        this.V = {};
        this.ya = {};
        this.N = {};
        this.ia = this.G = this.B = this.K = this.va = this.F = this.D = this.A = this.C = null;
        this.M = {
            x: 0,
            y: 0
        };
        this.R = this.Y=!1;
        this.ka = null;
        this.U = new _.mf;
        _.qe("gapi.load")("gapi.iframes.style.slide-menu")
    }, fO = 0, gO = function(a, b, c, d) {
        var e = WN(b), e = a.N[e] || (a.N[e] = {});
        e[c] || (e[c] = d = (0, _.s)(d, a), b.addEventListener ? b.addEventListener(c, d, "focus" == c || "blur" == c) : ("focus" == c ? c = "focusin" : "blur" == c && (c = "focusout"), b.attachEvent("on" + c, d)))
    }, hO = function(a, b, c) {
        (a =
        a.N[WN(b)]) && a[c] && (b.addEventListener ? b.removeEventListener(c, a[c], "focus" == c || "blur" == c) : b.detachEvent("focus" == c ? "onfocusin" : "blur" == c ? "onfocusout" : "on" + c, a[c]), delete a[c])
    }, iO = function(a, b) {
        var c = a.N[b.id];
        if (c)
            for (var d in c)
                c.hasOwnProperty(d) && hO(a, b, d)
    };
    eO.prototype.watch = function(a, b, c) {
        if (a = a || window.document.body) {
            this.Ia++;
            var d = WN(a);
            b && (this.V[d] = b);
            c && (this.ya[d] = c);
            gO(this, a, "mouseover", this.jv);
            gO(this, a, "mouseout", this.zq);
            gO(this, a, "mousedown", this.iv);
            gO(this, a, "focus", this.jv);
            gO(this, a, "blur", this.zq);
            gO(this, window.document.body, "mouseout", this.zq);
            gO(this, window.document.body, "mousedown", this.iv)
        } else 
            window.setTimeout((0, _.s)(this.watch, this), 100)
    };
    eO.prototype.unwatch = function(a) {
        if (a = a || window.document.body)
            if (aO(this, 0), a != window.document.body ? iO(this, a) : hO(this, window.document.body, "mouseover"), a = WN(a), delete this.V[a], delete this.ya[a], !(0<--this.Ia)) {
                iO(this, window.document.body);
                var b = this.C;
                this.da();
                this.C = null;
                window.setTimeout(function() {
                    var a = ON();
                    a && b && a.xj(b.Lf())
                }, 100)
            }
    };
    eO.prototype.Ob = function(a) {
        this.Sa.push(a)
    };
    eO.prototype.hh = function(a) {
        _.se(this.Sa, a)
    };
    var bO = function(a, b, c) {
        for (var d = []; b;) {
            var e = VN(b);
            e && a.V[e] && d.push(a.V[e]);
            b = b.parentNode
        }
        _.jd(d, a.Sa);
        (0, _.dd)(d, function(a) {
            a(c)
        })
    };
    _.h = eO.prototype;
    _.h.jv = function(a) {
        this.ia = (0, _.sa)();
        var b = a.target || a.srcElement;
        if (b && "IFRAME" != b.tagName) {
            for (; b&&!XN(b);)
                if (b = b.parentNode, 1 != b.nodeType)
                    return;
            if (b)
                if (this.F == b || this.A == b)
                    this.G && (window.clearTimeout(this.G), this.G = null);
                else {
                    this.A = b;
                    gO(this, b, "mousemove", this.ZF);
                    "focus" == a.type || "focusin" == a.type ? (a = _.Fs(b), this.M.x = a.x, this.M.y = a.y + b.offsetHeight) : (this.M.x = a.clientX, this.M.y = a.clientY);
                    this.K = (0, _.sa)();
                    a = this.D = YN(b, this.ya);
                    var c = ZN("hoverDelay", a);
                    this.C ? this.C.C && (window.clearTimeout(this.B),
                    this.B = window.setTimeout((0, _.s)(this.Sq, this, b), c - ZN("loadDelay", a))) : (window.clearTimeout(this.B), this.B = window.setTimeout((0, _.s)(this.nH, this), c - ZN("loadHovercardDelay", a)))
                }
            }
    };
    _.h.zq = function(a) {
        this.ia = (0, _.sa)();
        if ("blur" != a.type || a.target == this.F || a.target == this.A) {
            if (a = a.relatedTarget || a.toElement) {
                if ("IFRAME" == a.tagName)
                    return;
                if (this.ka)
                    for (; a && "BODY" != a.tagName;) {
                        if (a == this.ka)
                            return;
                            a = a.parentNode
                    }
            }
            aO(this)
        }
    };
    _.h.iv = function() {
        aO(this, 0)
    };
    _.h.ZF = function(a) {
        this.M.x = a.clientX;
        this.M.y = a.clientY
    };
    _.h.nH = function() {
        this.B && (window.clearTimeout(this.B), this.B = null);
        if (this.A && (bO(this, this.A, {
            type: "hover",
            config: this.D
        }), !this.C)) {
            var a = this.C = new $N(this);
            PN((0, _.s)(function(b) {
                b.zr("card", fO++, a.Lf()) && a.load(this.D)
            }, this))
        }
    };
    _.h.Sq = function(a) {
        this.B && (window.clearTimeout(this.B), this.B = null);
        if (this.A == a) {
            var b = ZN("hoverDelay", this.D) - ZN("loadDelay", this.D) - (0, _.sa)() + this.K;
            if (0 < b)
                this.B = window.setTimeout((0, _.s)(this.Sq, this, a), b);
            else {
                bO(this, a, {
                    type: "hover",
                    config: this.D
                });
                var b = this.C, c = this.D;
                a = (0, _.s)(this.Rz, this, a);
                b.C && (b.G = a, c.ri=++b.D, b.A.send("loadData", c, void 0, _.Am))
            }
        }
    };
    _.h.Rz = function(a) {
        if (this.A === a && this.C && this.C.C && this.K) {
            var b = ZN("hoverDelay", this.D) - (0, _.sa)() + this.K;
            0 < b ? window.setTimeout((0, _.s)(this.Rz, this, a), b) : (this.da(), this.F = this.A, this.va = this.D, hO(this, this.A, "mousemove"), this.K = this.D = this.A = null, ON().ds(this.C.Lf()))
        }
    };
    var aO = function(a, b) {
        a.A && hO(a, a.A, "mousemove");
        a.A = null;
        a.D = null;
        a.K = null;
        a.B && (window.clearTimeout(a.B), a.B = null);
        !a.G && a.F && (a.G = window.setTimeout((0, _.s)(a.da, a), _.Vc(b) ? b : ZN("closeDelay", a.va)))
    };
    eO.prototype.da = function() {
        this.G && (window.clearTimeout(this.G), this.G = null);
        this.F && ON().jk(this.C.Lf())
    };
    var dO = function(a, b) {
        bO(a, null, b);
        a.pa && a.pa.send("cardAction", b, void 0, a.wa)
    };
    eO.prototype.Ro = function(a, b, c) {
        var d = {};
        d.frame = a;
        d.filter = b || _.zm;
        d.nc = c || _.Dd;
        this.U.set(_.Sc(a), d);
        a.register("cardAction", (0, _.s)(function(a) {
            dO(this, a);
            d.nc(a)
        }, this), d.filter)
    };
    eO.prototype.So = function(a) {
        this.wa = a || _.zm;
        _.pm.D((0, _.s)(function(a) {
            this.pa = a;
            this.pa.register("cardAction", (0, _.s)(this.rl, this), this.wa)
        }, this), void 0, this.wa)
    };
    eO.prototype.rl = function(a) {
        this.U.isEmpty() || (0, _.dd)(this.U.xc(), function(b) {
            b.frame.send("cardAction", a, void 0, b.filter)
        });
        this.C && this.C.A.send("cardAction", a, void 0, _.Am)
    };
    _.jO = function() {
        var a = {}, b = new eO;
        a.watch = function(a, d, e) {
            b.watch(a, d, e)
        };
        a.unwatch = function(a) {
            b.unwatch(a)
        };
        a.Ob = function(a) {
            b.Ob(a)
        };
        a.hh = function(a) {
            b.hh(a)
        };
        a.Ro = function(a, d, e) {
            b.Ro(a, d, e)
        };
        a.So = function(a) {
            b.So(a)
        };
        a.rl = function(a) {
            b.rl(a)
        };
        a.Z = function(a, b) {
            b.origin = window.location.protocol + "//" + window.location.host;
            var e = _.pm.Xf({
                url: ":card",
                where: window.document.getElementById(a),
                queryParams: b,
                messageHandlers: {
                    _ready: function() {
                        e.send("loadData", b, void 0, _.Am)
                    },
                    loaded: function() {
                        e.send("render", void 0, void 0, _.Am)
                    }
                },
                messageHandlersFilter: _.Am
            })
        };
        return a
    }();
    _.B("gapi.card.watch", _.jO.watch);
    _.B("gapi.card.unwatch", _.jO.unwatch);
    _.B("gapi.card.addCallback", _.jO.Ob);
    _.B("gapi.card.removeCallback", _.jO.hh);
    _.B("gapi.card.render", _.jO.Z);
    _.B("gapi.card.connectChild", _.jO.Ro);
    _.B("gapi.card.connectOpener", _.jO.So);
    _.B("gapi.card.broadcast", _.jO.rl);

    _.C("gapi.iframes.create", _.am);
    _.Im.prototype.Gy = _.L(14, function(a, b) {
        this.register("_g_wasRestyled", a, b)
    });
    _.Hm.prototype.pa = _.L(13, function(a) {
        this.Bn("onRestyleSelfFilter", a)
    });
    _.Hm.prototype.wa = _.L(12, function(a) {
        this.Bn("onCloseSelfFilter", a)
    });
    _.Im.prototype.ping = _.L(11, function(a, b) {
        return _.Jm(this, "_g_ping", b, a)
    });
    _.Hm.prototype.U = _.L(10, function() {
        return this.B
    });
    _.Hm.prototype.Bn = _.L(9, function(a, b) {
        this.R[a] = b
    });
    _.C("gapi.iframes.registerStyle", function(a, b) {
        _.Pm[a] = b
    });
    _.C("gapi.iframes.registerBeforeOpenStyle", function(a, b) {
        _.Mm[a] = b
    });
    _.C("gapi.iframes.getStyle", _.Om);
    _.C("gapi.iframes.getBeforeOpenStyle", function(a) {
        return _.Mm[a]
    });
    _.C("gapi.iframes.registerIframesApi", _.Dm);
    _.C("gapi.iframes.registerIframesApiHandler", _.Em);
    _.C("gapi.iframes.getContext", _.Gm);
    _.C("gapi.iframes.SAME_ORIGIN_IFRAMES_FILTER", _.zm);
    _.C("gapi.iframes.CROSS_ORIGIN_IFRAMES_FILTER", _.Am);
    _.C("gapi.iframes.makeWhiteListIframesFilter", _.Bm);
    _.C("gapi.iframes.Context", _.Hm);
    _.C("gapi.iframes.Context.prototype.isDisposed", _.Hm.prototype.ob);
    _.C("gapi.iframes.Context.prototype.getWindow", _.Hm.prototype.Ja);
    _.C("gapi.iframes.Context.prototype.getFrameName", _.Hm.prototype.Ic);
    _.C("gapi.iframes.Context.prototype.getGlobalParam", _.Hm.prototype.Qp);
    _.C("gapi.iframes.Context.prototype.setGlobalParam", _.Hm.prototype.Bn);
    _.C("gapi.iframes.Context.prototype.open", _.Hm.prototype.open);
    _.C("gapi.iframes.Context.prototype.openChild", _.Hm.prototype.Xf);
    _.C("gapi.iframes.Context.prototype.getParentIframe", _.Hm.prototype.U);
    _.C("gapi.iframes.Context.prototype.closeSelf", _.Hm.prototype.F);
    _.C("gapi.iframes.Context.prototype.restyleSelf", _.Hm.prototype.Y);
    _.C("gapi.iframes.Context.prototype.setCloseSelfFilter", _.Hm.prototype.wa);
    _.C("gapi.iframes.Context.prototype.setRestyleSelfFilter", _.Hm.prototype.pa);
    _.C("gapi.iframes.Iframe", _.Im);
    _.C("gapi.iframes.Iframe.prototype.isDisposed", _.Im.prototype.ob);
    _.C("gapi.iframes.Iframe.prototype.getContext", _.Im.prototype.getContext);
    _.C("gapi.iframes.Iframe.prototype.getFrameName", _.Im.prototype.Ic);
    _.C("gapi.iframes.Iframe.prototype.getId", _.Im.prototype.ca);
    _.C("gapi.iframes.Iframe.prototype.register", _.Im.prototype.register);
    _.C("gapi.iframes.Iframe.prototype.unregister", _.Im.prototype.uh);
    _.C("gapi.iframes.Iframe.prototype.send", _.Im.prototype.send);
    _.C("gapi.iframes.Iframe.prototype.applyIframesApi", _.Im.prototype.nj);
    _.C("gapi.iframes.Iframe.prototype.getIframeEl", _.Im.prototype.Fa);
    _.C("gapi.iframes.Iframe.prototype.getSiteEl", _.Im.prototype.Pa);
    _.C("gapi.iframes.Iframe.prototype.setSiteEl", _.Im.prototype.Ud);
    _.C("gapi.iframes.Iframe.prototype.getWindow", _.Im.prototype.Ja);
    _.C("gapi.iframes.Iframe.prototype.getOrigin", _.Im.prototype.Ba);
    _.C("gapi.iframes.Iframe.prototype.close", _.Im.prototype.close);
    _.C("gapi.iframes.Iframe.prototype.restyle", _.Im.prototype.Ji);
    _.C("gapi.iframes.Iframe.prototype.restyleDone", _.Im.prototype.Ki);
    _.C("gapi.iframes.Iframe.prototype.registerWasRestyled", _.Im.prototype.Gy);
    _.C("gapi.iframes.Iframe.prototype.registerWasClosed", _.Im.prototype.gh);
    _.C("gapi.iframes.Iframe.prototype.getParam", _.Im.prototype.Qj);
    _.C("gapi.iframes.Iframe.prototype.setParam", _.Im.prototype.wb);
    _.C("gapi.iframes.Iframe.prototype.ping", _.Im.prototype.ping);

    _.Hm.prototype.ka = _.L(15, function(a, b) {
        var c = _.Ha(this.N, a, []);
        if (b)
            for (var d = 0, e=!1; !e && d < c.length; d++)
                c[d].Kc === b && (e=!0, c.splice(d, 1));
        else 
            c.splice(0, c.length)
    });
    _.C("gapi.iframes.Context.prototype.addOnConnectHandler", _.Hm.prototype.G);
    _.C("gapi.iframes.Context.prototype.removeOnConnectHandler", _.Hm.prototype.ka);
    _.C("gapi.iframes.Context.prototype.addOnOpenerHandler", _.Hm.prototype.D);
    _.C("gapi.iframes.Context.prototype.connectIframes", _.Hm.prototype.M);

});
// Google Inc.

