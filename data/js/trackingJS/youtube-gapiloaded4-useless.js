/* JS */
gapi.loaded_3(function(_) {
    var window = this;
    var aD = function(a, b, c, d, e) {
        this.width = a;
        this.height = b;
        this.A = c;
        this.B = d;
        this.opacity = e
    }, bD = function(a, b, c, d) {
        return new aD(void 0 == b ? a.width : b, void 0 == c ? a.height : c, a.A, a.B, void 0 == d ? a.opacity : d)
    };
    aD.prototype.equals = function(a) {
        return this.width == a.width && this.height == a.height && this.A == a.A && this.B == a.B && this.opacity == a.opacity
    };
    aD.prototype.interpolate = function(a, b) {
        var c = 1 - b;
        return new aD(Math.round(this.width * c + a.width * b), Math.round(this.height * c + a.height * b), Math.round(this.A * c + a.A * b), Math.round(this.B * c + a.B * b), this.opacity * c + a.opacity * b)
    };
    _.cD = function(a) {
        _.Os.call(this, a);
        this.$ = new _.gt(this)
    };
    _.u(_.cD, _.Qs);
    _.eD = function(a) {
        var b = Number(a.F(200, "widgetWidth", "width")), c = Number(a.F(100, "widgetHeight", "height")), d = a.F("auto", "corner"), e = a.F({}, "menuCss"), f = a.F(!0, "hideOnClick");
        a.ec=!!a.F(!0, "hideOnLeave");
        a.mb=!!a.F((0, _.dv)(), "useCss3Transition");
        a.hd=!!a.F(!0, "allowOffset");
        var g = window.document.createElement("div");
        a.B = g;
        var k = _.Ts(a);
        k.parentNode.appendChild(g);
        var l = window.document.createElement("div");
        a.N = l;
        g.appendChild(l);
        e.position = "absolute";
        e.width = b + "px";
        e.height = "0";
        e.border = e.border || "1px solid #aaa";
        e.background = e.background || "#fff";
        e.zIndex = _.Vs(a);
        e.overflow = "hidden";
        e.boxShadow = e.MozBoxShadow = e.webkitBoxShadow = e.boxShadow || "0 2px 2px rgba(0,0,0,0.3)";
        _.Q(g, e);
        _.Q(l, {
            left: "-1000px",
            top: "-1000px",
            position: "absolute",
            width: b + "px",
            height: c + "px"
        });
        a.kd = g.offsetWidth - b;
        a.ld = g.offsetHeight;
        e = "auto" == d ? ["top", "start"] : d.split("-");
        k = _.ps(k);
        a.Sa = "bottom" == e[0];
        a.ka = "right" == e[1] || "left" != e[1] && k;
        a.C = dD(a, b, c, "auto" == d);
        a.cb = _.r.setTimeout((0, _.s)(a.jd, a), 500);
        a.$.P(g, "mouseover", a.QG);
        _.it(a.$, g,
        "mouseout", a.PG, !1, a);
        f && _.r.setTimeout((0, _.s)(function() {
            this.U || this.$.P(window.document, "click", this.IH)
        }, a), 0)
    };
    _.cD.prototype.open = function() {
        _.eD(this);
        this.S.Ud(this.B);
        this.S.Ob("resize", (0, _.s)(this.resize, this));
        this.S.Nc("showMenu", (0, _.s)(this.Sz, this));
        var a = this.S.methods;
        a.setHideOnLeave = (0, _.s)(this.Yw, this);
        a.displayStateCallback && (this.td = a.displayStateCallback, delete a.displayStateCallback);
        this.S.Gd(this.N, {
            height: "100%"
        });
        _.fD(this, this.C)
    };
    _.cD.prototype.open = _.cD.prototype.open;
    _.cD.prototype.kc = function(a) {
        this.U || (a = a || {}, this.N.style.left = "0", this.N.style.top = "0", this.D && (_.Ag(this.D), this.D = null), this.cb && (_.r.clearTimeout(this.cb), this.cb = null), gD(this, (0, window.parseInt)(a.width, 10) || (0, window.parseInt)(this.S.width, 10) || this.C.width, (0, window.parseInt)(a.height, 10) || (0, window.parseInt)(this.S.height, 10) || this.C.height))
    };
    _.cD.prototype.onready = _.cD.prototype.kc;
    _.cD.prototype.resize = function(a) {
        this.U || gD(this, (0, window.isNaN)( + a.width) ? this.C.width : + a.width, (0, window.isNaN)( + a.height) ? this.C.height : + a.height)
    };
    _.cD.prototype.close = function() {
        this.Ia ? hD(this) : (this.pa=!0, iD(this))
    };
    _.cD.prototype.close = _.cD.prototype.close;
    _.cD.prototype.F = function(a, b) {
        for (var c, d = 1, e = arguments.length; d < e && (c = this.A[arguments[d]], void 0 === c); ++d);
        return void 0 !== c ? c && "object" == typeof c ? (d = {}, _.Ur(d, c), d) : c : a
    };
    var dD = function(a, b, c, d) {
        var e = _.Ts(a), f = _.Ds(e.offsetParent);
        f.right -= a.kd;
        f.bottom -= a.ld;
        if (d) {
            var g = e.offsetLeft;
            d = g + b < f.right;
            g = g + e.offsetWidth - b >= f.left;
            a.ka = a.ka?!d || g : !d && g;
            g = e.offsetTop;
            d = g + c < f.bottom;
            g = g + e.offsetHeight - c >= f.top;
            a.Sa = a.Sa?!d || g : !d && g
        }
        d = jD(a.ka, e.offsetLeft, e.offsetWidth, b, f.left, f.right);
        a = jD(a.Sa, e.offsetTop, e.offsetHeight, c, f.top, f.bottom);
        return new aD(b, c, d, a, 1)
    }, jD = function(a, b, c, d, e, f) {
        return a ? (a = b + c, Math.min(f - a, Math.max(e + d - a, 0))) : Math.max(e - b, Math.min(f - b - d, 0))
    },
    gD = function(a, b, c) {
        a.S.Fa().style.width = b + "px";
        a.S.Fa().style.height = c + "px";
        a.N.style.width = b + "px";
        a.N.style.height = c + "px";
        b = a.hd ? dD(a, b, c, !1) : bD(a.va, b, c);
        a.C = b;
        a.M || kD(a, b)
    };
    _.h = _.cD.prototype;
    _.h.Yw = function(a) {
        this.ec=!!a
    };
    _.h.QG = function() {
        !this.M && this.K && (_.r.clearTimeout(this.K), this.K = null)
    };
    _.h.PG = function(a) {
        if (!(this.M || this.K ||!this.ec || a.relatedTarget && _.Fg(this.B, a.relatedTarget))) {
            var b = _.Es(this.B);
            a = a.clientY > b.top && a.clientY < b.top + b.height ? 0 : 300;
            this.K = _.r.setTimeout((0, _.s)(this.ub, this), a)
        }
    };
    _.h.IH = function(a) {
        _.Us(this.B, a) || this.ub()
    };
    _.h.Sz = function(a) {
        this.show(!!a || void 0 === a)
    };
    _.h.show = function(a) {
        a?!this.U && this.Ia && (this.M = this.Ia=!1, _.fD(this, this.C)) : this.ub()
    };
    _.fD = function(a, b) {
        lD(a, bD(b, void 0, 0, 0));
        _.r.setTimeout((0, _.s)(function() {
            this.M || (this.C = b, this.mb && _.cv(this.B, "width 350ms cubic-bezier(.23, .50, .32, 1),height 350ms cubic-bezier(.23, .50, .32, 1),left 350ms cubic-bezier(.23, .50, .32, 1),top 350ms cubic-bezier(.23, .50, .32, 1),opacity 350ms cubic-bezier(.23, .50, .32, 1)"), kD(this, b), this.ia(!0, !1), this.ya && (_.r.clearTimeout(this.ya), this.pa=!1), this.ya = _.r.setTimeout((0, _.s)(this.ia, this, !0, !0), 350))
        }, a), 0)
    };
    _.cD.prototype.ub = function() {
        this.M || iD(this)
    };
    var iD = function(a) {
        a.M || (a.ia(!1, !1), kD(a, bD(a.va, void 0, 0, 0)), a.M=!0, a.ya && _.r.clearTimeout(a.ya), a.ya = _.r.setTimeout((0, _.s)(function() {
            this.Ia=!0;
            this.pa ? (hD(this), this.pa=!1) : (this.mb && _.cv(this.B, ""), this.B.visibility = "hidden", this.B.left = "-1000px", this.B.top = "-1000px");
            this.ia(!1, !0)
        }, a), 350))
    }, hD = function(a) {
        a.U=!0;
        _.ue(a.$);
        a.$ = null;
        a.K && (_.r.clearTimeout(a.K), a.K = null);
        a.Y && (_.r.clearTimeout(a.Y), a.Y = null);
        _.Ag(a.B);
        a.B = null;
        _.Ag(_.Ts(a));
        a.N = null;
        a.D = null
    };
    _.cD.prototype.ia = function(a, b) {
        var c = this.td;
        c && _.r.setTimeout(function() {
            c(a, b)
        }, 0)
    };
    _.cD.prototype.jd = function() {
        this.D = window.document.createElement("div");
        this.D.style.width = this.D.style.height = "100%";
        this.D.style.position = "absolute";
        this.D.style.background = "url(//ssl.gstatic.com/ui/v1/activityindicator/loading_gs.gif) no-repeat " + (this.C.width - 19) / 2 + "px " + (this.C.height - 19) / 2 + "px";
        this.B.appendChild(this.D)
    };
    var kD = function(a, b) {
        if (!a.M)
            if (a.mb)
                lD(a, b);
            else {
                var c = (0, _.sa)() - 20 + 350;
                a.Y && _.r.clearTimeout(a.Y);
                a.Oc(a.va, b, c)
            }
    }, lD = function(a, b) {
        var c = a.B, d = _.Ts(a);
        c.style.left = a.ka ? d.offsetLeft + d.offsetWidth + b.A - b.width + "px" : d.offsetLeft + b.A + "px";
        c.style.width = b.width + "px";
        c.style.top = a.Sa ? d.offsetTop + d.offsetHeight + b.B - b.height + "px" : d.offsetTop + b.B + "px";
        c.style.height = b.height + "px";
        _.Js(c, b.opacity);
        a.va = b
    };
    _.cD.prototype.Oc = function(a, b, c) {
        this.Y = null;
        if (!this.U&&!a.equals(b)) {
            var d = (0, _.sa)();
            d >= c ? lD(this, b) : (d = Math.min(1 - (c - d) / 350, 1), lD(this, a.interpolate(b, 1 - Math.pow(1 - d, 3))), this.Y = _.r.setTimeout((0, _.s)(this.Oc, this, a, b, c), 20))
        }
    };

    _.Pm["slide-menu"] = function(a) {
        var b = new _.cD(_.oN(a)), c = new _.$s(b);
        b.open = function() {};
        _.eD(b);
        a.attributes = {
            height: "100%"
        };
        a.where = b.N;
        a.onClose = function() {
            b.close()
        };
        a.onRestyle = function(a) {
            if (a) {
                var c=!1;
                a.hasOwnProperty("showMenu") && (b.Sz(a.showMenu), c=!0);
                a.hasOwnProperty("setHideOnLeave") && (b.Yw(a.setHideOnLeave), c=!0);
                c || b.resize(a)
            }
        };
        a.onCreate = function(a) {
            a.Kb = function() {
                return this.Qj("openerIframe")
            };
            c.S = b.S = a;
            a.Ud(b.B);
            a.register("_ready", (0, _.s)(b.kc, b), _.Am);
            b.td = function(b, c) {
                !a.ob() && a.Ki([b, c])
            };
            _.fD(b, b.C);
            c.open()
        }
    };

});
// Google Inc.

