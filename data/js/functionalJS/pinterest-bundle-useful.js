function printStackTrace(a) {
    a = a || {
        guess: !0
    };
    var b = a.e || null, c=!!a.guess, d = new printStackTrace.implementation, e = d.run(b);
    return c ? d.guessAnonymousFunctions(e) : e
}
function FastClick(a, b) {
    "use strict";
    function c(a, b) {
        return function() {
            return a.apply(b, arguments)
        }
    }
    var d;
    if (b = b || {}, this.trackingClick=!1, this.trackingClickStart = 0, this.targetElement = null, this.touchStartX = 0, this.touchStartY = 0, this.lastTouchIdentifier = 0, this.touchBoundary = b.touchBoundary || 10, this.layer = a, this.tapDelay = b.tapDelay || 200, !FastClick.notNeeded(a)) {
        for (var e = ["onMouse", "onClick", "onTouchStart", "onTouchMove", "onTouchEnd", "onTouchCancel"], f = this, g = 0, h = e.length; g < h; g++)
            f[e[g]] = c(f[e[g]], f);
        deviceIsAndroid && (a.addEventListener("mouseover", this.onMouse, !0), a.addEventListener("mousedown", this.onMouse, !0), a.addEventListener("mouseup", this.onMouse, !0)), a.addEventListener("click", this.onClick, !0), a.addEventListener("touchstart", this.onTouchStart, !1), a.addEventListener("touchmove", this.onTouchMove, !1), a.addEventListener("touchend", this.onTouchEnd, !1), a.addEventListener("touchcancel", this.onTouchCancel, !1), Event.prototype.stopImmediatePropagation || (a.removeEventListener = function(b, c, d) {
            var e = Node.prototype.removeEventListener;
            "click" === b ? e.call(a, b, c.hijacked || c, d) : e.call(a, b, c, d)
        }, a.addEventListener = function(b, c, d) {
            var e = Node.prototype.addEventListener;
            "click" === b ? e.call(a, b, c.hijacked || (c.hijacked = function(a) {
                a.propagationStopped || c(a)
            }), d) : e.call(a, b, c, d)
        }), "function" === typeof a.onclick && (d = a.onclick, a.addEventListener("click", function(a) {
            d(a)
        }, !1), a.onclick = null)
    }
}
window.Modernizr = function(a, b, c) {
    function d(a) {
        r.cssText = a
    }
    function e(a, b) {
        return typeof a === b
    }
    function f(a, b) {
        return !!~("" + a).indexOf(b)
    }
    function g(a, b) {
        for (var d in a) {
            var e = a[d];
            if (!f(e, "-") && r[e] !== c)
                return "pfx" == b ? e : !0
        }
        return !1
    }
    function h(a, b, d) {
        for (var f in a) {
            var g = b[a[f]];
            if (g !== c)
                return d===!1 ? a[f] : e(g, "function") ? g.bind(d || b) : g
        }
        return !1
    }
    function i(a, b, c) {
        var d = a.charAt(0).toUpperCase() + a.slice(1), f = (a + " " + u.join(d + " ") + d).split(" ");
        return e(b, "string") || e(b, "undefined") ? g(f, b) : (f = (a + " " + v.join(d + " ") + d).split(" "), h(f, b, c))
    }
    var j, k, l, m = "2.7.1", n = {}, o = b.documentElement, p = "modernizr", q = b.createElement(p), r = q.style, s = ({}.toString, " -webkit- -moz- -o- -ms- ".split(" ")), t = "Webkit Moz O ms", u = t.split(" "), v = t.toLowerCase().split(" "), w = {}, x = [], y = x.slice, z = function(a, c, d, e) {
        var f, g, h, i, j = b.createElement("div"), k = b.body, l = k || b.createElement("body");
        if (parseInt(d, 10))
            for (; d--;)
                h = b.createElement("div"), h.id = e ? e[d] : p + (d + 1), j.appendChild(h);
        return f = ["&#173;", '<style id="s', p, '">', a, "</style>"].join(""), j.id = p, (k ? j : l).innerHTML += f, l.appendChild(j), k || (l.style.background = "", l.style.overflow = "hidden", i = o.style.overflow, o.style.overflow = "hidden", o.appendChild(l)), g = c(j, a), k ? j.parentNode.removeChild(j) : (l.parentNode.removeChild(l), o.style.overflow = i), !!g
    }, A = {}.hasOwnProperty;
    l = e(A, "undefined") || e(A.call, "undefined") ? function(a, b) {
        return b in a && e(a.constructor.prototype[b], "undefined")
    } : function(a, b) {
        return A.call(a, b)
    }, Function.prototype.bind || (Function.prototype.bind = function(a) {
        var b = this;
        if ("function" != typeof b)
            throw new TypeError;
        var c = y.call(arguments, 1), d = function() {
            if (this instanceof d) {
                var e = function() {};
                e.prototype = b.prototype;
                var f = new e, g = b.apply(f, c.concat(y.call(arguments)));
                return Object(g) === g ? g : f
            }
            return b.apply(a, c.concat(y.call(arguments)))
        };
        return d
    }), w.touch = function() {
        var c;
        return "ontouchstart"in a || a.DocumentTouch && b instanceof DocumentTouch ? c=!0 : z(["@media (", s.join("touch-enabled),("), p, ")", "{#modernizr{top:9px;position:absolute}}"].join(""), function(a) {
            c = 9 === a.offsetTop
        }), c
    }, w.csstransforms3d = function() {
        var a=!!i("perspective");
        return a && "webkitPerspective"in o.style && z("@media (transform-3d),(-webkit-transform-3d){#modernizr{left:9px;position:absolute;height:3px;}}", function(b) {
            a = 9 === b.offsetLeft && 3 === b.offsetHeight
        }), a
    };
    for (var B in w)
        l(w, B) && (k = B.toLowerCase(), n[k] = w[B](), x.push((n[k] ? "" : "no-") + k));
    return n.addTest = function(a, b) {
        if ("object" == typeof a)
            for (var d in a)
                l(a, d) && n.addTest(d, a[d]);
        else {
            if (a = a.toLowerCase(), n[a] !== c)
                return n;
            b = "function" == typeof b ? b() : b, "undefined" !== typeof enableClasses && enableClasses && (o.className += " " + (b ? "" : "no-") + a), n[a] = b
        }
        return n
    }, d(""), q = j = null, n._version = m, n._prefixes = s, n._domPrefixes = v, n._cssomPrefixes = u, n.testProp = function(a) {
        return g([a])
    }, n.testAllProps = i, n.testStyles = z, n
}(this, this.document), function(a, b, c) {
    function d(a) {
        return "[object Function]" == q.call(a)
    }
    function e(a) {
        return "string" == typeof a
    }
    function f() {}
    function g(a) {
        return !a || "loaded" == a || "complete" == a || "uninitialized" == a
    }
    function h() {
        var a = r.shift();
        s = 1, a ? a.t ? o(function() {
            ("c" == a.t ? m.injectCss : m.injectJs)(a.s, 0, a.a, a.x, a.e, 1)
        }, 0) : (a(), h()) : s = 0
    }
    function i(a, c, d, e, f, i, j) {
        function k(b) {
            if (!n && g(l.readyState) && (t.r = n = 1, !s && h(), l.onload = l.onreadystatechange = null, b)) {
                "img" != a && o(function() {
                    v.removeChild(l)
                }, 50);
                for (var d in A[c])
                    A[c].hasOwnProperty(d) && A[c][d].onload()
            }
        }
        var j = j || m.errorTimeout, l = b.createElement(a), n = 0, q = 0, t = {
            t: d,
            s: c,
            e: f,
            a: i,
            x: j
        };
        1 === A[c] && (q = 1, A[c] = []), "object" == a ? l.data = c : (l.src = c, l.type = a), l.width = l.height = "0", l.onerror = l.onload = l.onreadystatechange = function() {
            k.call(this, q)
        }, r.splice(e, 0, t), "img" != a && (q || 2 === A[c] ? (v.insertBefore(l, u ? null : p), o(k, j)) : A[c].push(l))
    }
    function j(a, b, c, d, f) {
        return s = 0, b = b || "j", e(a) ? i("c" == b ? x : w, a, b, this.i++, c, d, f) : (r.splice(this.i++, 0, a), 1 == r.length && h()), this
    }
    function k() {
        var a = m;
        return a.loader = {
            load: j,
            i: 0
        }, a
    }
    var l, m, n = b.documentElement, o = a.setTimeout, p = b.getElementsByTagName("script")[0], q = {}.toString, r = [], s = 0, t = "MozAppearance"in n.style, u = t&&!!b.createRange().compareNode, v = u ? n: p.parentNode, n = a.opera && "[object Opera]" == q.call(a.opera), n=!!b.attachEvent&&!n, w = t ? "object" : n ? "script" : "img", x = n ? "script" : w, y = Array.isArray || function(a) {
        return "[object Array]" == q.call(a)
    }, z = [], A = {}, B = {
        timeout: function(a, b) {
            return b.length && (a.timeout = b[0]), a
        }
    };
    m = function(a) {
        function b(a) {
            var b, c, d, a = a.split("!"), e = z.length, f = a.pop(), g = a.length, f = {
                url: f,
                origUrl: f,
                prefixes: a
            };
            for (c = 0; c < g; c++)
                d = a[c].split("="), (b = B[d.shift()]) && (f = b(f, d));
            for (c = 0; c < e; c++)
                f = z[c](f);
            return f
        }
        function g(a, e, f, g, h) {
            var i = b(a), j = i.autoCallback;
            i.url.split(".").pop().split("?").shift(), i.bypass || (e && (e = d(e) ? e : e[a] || e[g] || e[a.split("/").pop().split("?")[0]]), i.instead ? i.instead(a, e, f, g, h) : (A[i.url] ? i.noexec=!0 : A[i.url] = 1, f.load(i.url, i.forceCSS ||!i.forceJS && "css" == i.url.split(".").pop().split("?").shift() ? "c" : c, i.noexec, i.attrs, i.timeout), (d(e) || d(j)) && f.load(function() {
                k(), e && e(i.origUrl, h, g), j && j(i.origUrl, h, g), A[i.url] = 2
            })))
        }
        function h(a, b) {
            function c(a, c) {
                if (a) {
                    if (e(a))
                        c || (l = function() {
                            var a = [].slice.call(arguments);
                            m.apply(this, a), n()
                        }), g(a, l, b, 0, j);
                    else if (Object(a) === a)
                        for (i in h = function() {
                            var b, c = 0;
                            for (b in a)
                                a.hasOwnProperty(b) && c++;
                                return c
                            }(), a)
                                a.hasOwnProperty(i) && (!c&&!--h && (d(l) ? l = function() {
                                    var a = [].slice.call(arguments);
                                    m.apply(this, a), n()
                                } : l[i] = function(a) {
                                    return function() {
                                        var b = [].slice.call(arguments);
                                        a && a.apply(this, b), n()
                                    }
                                }(m[i])), g(a[i], l, b, i, j))
                } else 
                    !c && n()
            }
            var h, i, j=!!a.test, k = a.load || a.both, l = a.callback || f, m = l, n = a.complete || f;
            c(j ? a.yep : a.nope, !!k), k && c(k)
        }
        var i, j, l = this.yepnope.loader;
        if (e(a))
            g(a, 0, l, 0);
        else if (y(a))
            for (i = 0; i < a.length; i++)
                j = a[i], e(j) ? g(j, 0, l, 0) : y(j) ? m(j) : Object(j) === j && h(j, l);
        else 
            Object(a) === a && h(a, l)
    }, m.addPrefix = function(a, b) {
        B[a] = b
    }, m.addFilter = function(a) {
        z.push(a)
    }, m.errorTimeout = 1e4, null == b.readyState && b.addEventListener && (b.readyState = "loading", b.addEventListener("DOMContentLoaded", l = function() {
        b.removeEventListener("DOMContentLoaded", l, 0), b.readyState = "complete"
    }, 0)), a.yepnope = k(), a.yepnope.executeStack = h, a.yepnope.injectJs = function(a, c, d, e, i, j) {
        var k, l, n = b.createElement("script"), e = e || m.errorTimeout;
        n.src = a;
        for (l in d)
            n.setAttribute(l, d[l]);
        c = j ? h : c || f, n.onreadystatechange = n.onload = function() {
            !k && g(n.readyState) && (k = 1, c(), n.onload = n.onreadystatechange = null)
        }, o(function() {
            k || (k = 1, c(1))
        }, e), i ? n.onload() : p.parentNode.insertBefore(n, p)
    }, a.yepnope.injectCss = function(a, c, d, e, g, i) {
        var j, e = b.createElement("link"), c = i ? h: c || f;
        e.href = a, e.rel = "stylesheet", e.type = "text/css";
        for (j in d)
            e.setAttribute(j, d[j]);
        g || (p.parentNode.insertBefore(e, p), o(c, 0))
    }
}(this, document), Modernizr.load = function() {
    yepnope.apply(window, [].slice.call(arguments, 0))
}, Modernizr.addTest("fileinput", function() {
    var a = document.createElement("input");
    return a.type = "file", !a.disabled
});
var goog = {
    require: function() {},
    provide: function(a) {
        for (var b = window, c = a.split(/\./), d = 0; d < c.length; d++) {
            var e = c[d];
            void 0 === b[e] && (b[e] = {}), b = b[e]
        }
    }
};
goog.i18n = {}, goog.i18n.pluralRules = {}, goog.i18n.pluralRules.Keyword = {
    ZERO: "zero",
    ONE: "one",
    TWO: "two",
    FEW: "few",
    MANY: "many",
    OTHER: "other"
}, goog.i18n.pluralRules.defaultSelect_ = function() {
    return goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.decimals_ = function(a) {
    var b = a + "", c = b.indexOf(".");
    return - 1 == c ? 0 : b.length - c - 1
}, goog.i18n.pluralRules.get_vf_ = function(a, b) {
    var c = 3;
    if (void 0 === b)
        var d = Math.min(goog.i18n.pluralRules.decimals_(a), c);
    else 
        var d = b;
    var e = Math.pow(10, d), f = (a * e | 0)%e;
    return {
        v: d,
        f: f
    }
}, goog.i18n.pluralRules.get_wt_ = function(a, b) {
    if (0 === b)
        return {
            w: 0,
            t: 0
        };
    for (; b%10 === 0;)
        b/=10, a--;
    return {
        w: a,
        t: b
    }
}, goog.i18n.pluralRules.gaSelect_ = function(a) {
    return 1 == a ? goog.i18n.pluralRules.Keyword.ONE : 2 == a ? goog.i18n.pluralRules.Keyword.TWO : a >= 3 && a <= 6 ? goog.i18n.pluralRules.Keyword.FEW : a >= 7 && a <= 10 ? goog.i18n.pluralRules.Keyword.MANY : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.roSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b);
    return 1 == c && 0 == d.v ? goog.i18n.pluralRules.Keyword.ONE : 0 != d.v || 0 == a || 1 != a && a%100 >= 1 && a%100 <= 19 ? goog.i18n.pluralRules.Keyword.FEW : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.frSelect_ = function(a) {
    var b = 0 | a;
    return 0 == b || 1 == b ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.enSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b);
    return 1 == c && 0 == d.v ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.mtSelect_ = function(a) {
    return 1 == a ? goog.i18n.pluralRules.Keyword.ONE : 0 == a || a%100 >= 2 && a%100 <= 10 ? goog.i18n.pluralRules.Keyword.FEW : a%100 >= 11 && a%100 <= 19 ? goog.i18n.pluralRules.Keyword.MANY : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.daSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b), e = goog.i18n.pluralRules.get_wt_(d.v, d.f);
    return 1 == a || 0 != e.t && (0 == c || 1 == c) ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.cySelect_ = function(a) {
    return 0 == a ? goog.i18n.pluralRules.Keyword.ZERO : 1 == a ? goog.i18n.pluralRules.Keyword.ONE : 2 == a ? goog.i18n.pluralRules.Keyword.TWO : 3 == a ? goog.i18n.pluralRules.Keyword.FEW : 6 == a ? goog.i18n.pluralRules.Keyword.MANY : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.brSelect_ = function(a) {
    return a%10 == 1 && a%100 != 11 && a%100 != 71 && a%100 != 91 ? goog.i18n.pluralRules.Keyword.ONE : a%10 == 2 && a%100 != 12 && a%100 != 72 && a%100 != 92 ? goog.i18n.pluralRules.Keyword.TWO : (a%10 >= 3 && a%10 <= 4 || a%10 == 9) && (a%100 < 10 || a%100 > 19) && (a%100 < 70 || a%100 > 79) && (a%100 < 90 || a%100 > 99) ? goog.i18n.pluralRules.Keyword.FEW : 0 != a && a%1e6 == 0 ? goog.i18n.pluralRules.Keyword.MANY : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.esSelect_ = function(a) {
    return 1 == a ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.siSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b);
    return 0 == a || 1 == a || 0 == c && 1 == d.f ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.slSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b);
    return 0 == d.v && c%100 == 1 ? goog.i18n.pluralRules.Keyword.ONE : 0 == d.v && c%100 == 2 ? goog.i18n.pluralRules.Keyword.TWO : 0 == d.v && c%100 >= 3 && c%100 <= 4 || 0 != d.v ? goog.i18n.pluralRules.Keyword.FEW : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.tzmSelect_ = function(a) {
    return a >= 0 && a <= 1 || a >= 11 && a <= 99 ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.srSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b);
    return 0 == d.v && c%10 == 1 && c%100 != 11 || d.f%10 == 1 && d.f%100 != 11 ? goog.i18n.pluralRules.Keyword.ONE : 0 == d.v && c%10 >= 2 && c%10 <= 4 && (c%100 < 12 || c%100 > 14) || d.f%10 >= 2 && d.f%10 <= 4 && (d.f%100 < 12 || d.f%100 > 14) ? goog.i18n.pluralRules.Keyword.FEW : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.hiSelect_ = function(a) {
    var b = 0 | a;
    return 0 == b || 1 == a ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.mkSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b);
    return 0 == d.v && c%10 == 1 || d.f%10 == 1 ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.arSelect_ = function(a) {
    return 0 == a ? goog.i18n.pluralRules.Keyword.ZERO : 1 == a ? goog.i18n.pluralRules.Keyword.ONE : 2 == a ? goog.i18n.pluralRules.Keyword.TWO : a%100 >= 3 && a%100 <= 10 ? goog.i18n.pluralRules.Keyword.FEW : a%100 >= 11 && a%100 <= 99 ? goog.i18n.pluralRules.Keyword.MANY : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.iuSelect_ = function(a) {
    return 1 == a ? goog.i18n.pluralRules.Keyword.ONE : 2 == a ? goog.i18n.pluralRules.Keyword.TWO : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.csSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b);
    return 1 == c && 0 == d.v ? goog.i18n.pluralRules.Keyword.ONE : c >= 2 && c <= 4 && 0 == d.v ? goog.i18n.pluralRules.Keyword.FEW : 0 != d.v ? goog.i18n.pluralRules.Keyword.MANY : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.pt_PTSelect_ = function(a, b) {
    var c = goog.i18n.pluralRules.get_vf_(a, b);
    return 1 == a && 0 == c.v ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.beSelect_ = function(a) {
    return a%10 == 1 && a%100 != 11 ? goog.i18n.pluralRules.Keyword.ONE : a%10 >= 2 && a%10 <= 4 && (a%100 < 12 || a%100 > 14) ? goog.i18n.pluralRules.Keyword.FEW : a%10 == 0 || a%10 >= 5 && a%10 <= 9 || a%100 >= 11 && a%100 <= 14 ? goog.i18n.pluralRules.Keyword.MANY : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.akSelect_ = function(a) {
    return a >= 0 && a <= 1 ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.ptSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b), e = goog.i18n.pluralRules.get_wt_(d.v, d.f);
    return 1 == c && 0 == d.v || 0 == c && 1 == e.t ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.plSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b);
    return 1 == c && 0 == d.v ? goog.i18n.pluralRules.Keyword.ONE : 0 == d.v && c%10 >= 2 && c%10 <= 4 && (c%100 < 12 || c%100 > 14) ? goog.i18n.pluralRules.Keyword.FEW : 0 == d.v && 1 != c && c%10 >= 0 && c%10 <= 1 || 0 == d.v && c%10 >= 5 && c%10 <= 9 || 0 == d.v && c%100 >= 12 && c%100 <= 14 ? goog.i18n.pluralRules.Keyword.MANY : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.ruSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b);
    return 0 == d.v && c%10 == 1 && c%100 != 11 ? goog.i18n.pluralRules.Keyword.ONE : 0 == d.v && c%10 >= 2 && c%10 <= 4 && (c%100 < 12 || c%100 > 14) ? goog.i18n.pluralRules.Keyword.FEW : 0 == d.v && c%10 == 0 || 0 == d.v && c%10 >= 5 && c%10 <= 9 || 0 == d.v && c%100 >= 11 && c%100 <= 14 ? goog.i18n.pluralRules.Keyword.MANY : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.gvSelect_ = function(a) {
    return a%10 == 1 ? goog.i18n.pluralRules.Keyword.ONE : a%10 == 2 ? goog.i18n.pluralRules.Keyword.TWO : a%100 == 0 || a%100 == 20 || a%100 == 40 || a%100 == 60 ? goog.i18n.pluralRules.Keyword.FEW : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.lagSelect_ = function(a) {
    var b = 0 | a;
    return 0 == a ? goog.i18n.pluralRules.Keyword.ZERO : 0 != b && 1 != b || 0 == a ? goog.i18n.pluralRules.Keyword.OTHER : goog.i18n.pluralRules.Keyword.ONE
}, goog.i18n.pluralRules.shiSelect_ = function(a) {
    var b = 0 | a;
    return 0 == b || 1 == a ? goog.i18n.pluralRules.Keyword.ONE : a >= 2 && a <= 10 ? goog.i18n.pluralRules.Keyword.FEW : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.heSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b);
    return 1 == c && 0 == d.v ? goog.i18n.pluralRules.Keyword.ONE : 2 == c && 0 == d.v ? goog.i18n.pluralRules.Keyword.TWO : 0 == d.v && (a < 0 || a > 10) && a%10 == 0 ? goog.i18n.pluralRules.Keyword.MANY : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.isSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b), e = goog.i18n.pluralRules.get_wt_(d.v, d.f);
    return 0 == e.t && c%10 == 1 && c%100 != 11 || 0 != e.t ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.ltSelect_ = function(a, b) {
    var c = goog.i18n.pluralRules.get_vf_(a, b);
    return a%10 == 1 && (a%100 < 11 || a%100 > 19) ? goog.i18n.pluralRules.Keyword.ONE : a%10 >= 2 && a%10 <= 9 && (a%100 < 11 || a%100 > 19) ? goog.i18n.pluralRules.Keyword.FEW : 0 != c.f ? goog.i18n.pluralRules.Keyword.MANY : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.gdSelect_ = function(a) {
    return 1 == a || 11 == a ? goog.i18n.pluralRules.Keyword.ONE : 2 == a || 12 == a ? goog.i18n.pluralRules.Keyword.TWO : a >= 3 && a <= 10 || a >= 13 && a <= 19 ? goog.i18n.pluralRules.Keyword.FEW : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.filSelect_ = function(a, b) {
    var c = 0 | a, d = goog.i18n.pluralRules.get_vf_(a, b);
    return c >= 0 && c <= 1 && 0 == d.v ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.lvSelect_ = function(a, b) {
    var c = goog.i18n.pluralRules.get_vf_(a, b);
    return a%10 == 0 || a%100 >= 11 && a%100 <= 19 || 2 == c.v && c.f%100 >= 11 && c.f%100 <= 19 ? goog.i18n.pluralRules.Keyword.ZERO : a%10 == 1 && a%100 != 11 || 2 == c.v && c.f%10 == 1 && c.f%100 != 11 || 2 != c.v && c.f%10 == 1 ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.kshSelect_ = function(a) {
    return 0 == a ? goog.i18n.pluralRules.Keyword.ZERO : 1 == a ? goog.i18n.pluralRules.Keyword.ONE : goog.i18n.pluralRules.Keyword.OTHER
}, goog.i18n.pluralRules.select = function(a, b, c) {
    switch (c = c.replace("-", "_")) {
    case"af":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"am":
        return goog.i18n.pluralRules.hiSelect_(a, b);
    case"ar":
        return goog.i18n.pluralRules.arSelect_(a, b);
    case"az":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"bg":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"bn":
        return goog.i18n.pluralRules.hiSelect_(a, b);
    case"br":
        return goog.i18n.pluralRules.brSelect_(a, b);
    case"ca":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"chr":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"cs":
        return goog.i18n.pluralRules.csSelect_(a, b);
    case"cy":
        return goog.i18n.pluralRules.cySelect_(a, b);
    case"da":
        return goog.i18n.pluralRules.daSelect_(a, b);
    case"de":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"de_AT":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"de_CH":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"el":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"en":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"en_AU":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"en_GB":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"en_IE":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"en_IN":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"en_ISO":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"en_SG":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"en_IN":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"en_US":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"en_ZA":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"es":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"es_419":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"es_ES":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"et":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"eu":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"fa":
        return goog.i18n.pluralRules.hiSelect_(a, b);
    case"fi":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"fil":
        return goog.i18n.pluralRules.filSelect_(a, b);
    case"fr":
        return goog.i18n.pluralRules.frSelect_(a, b);
    case"fr_CA":
        return goog.i18n.pluralRules.frSelect_(a, b);
    case"gl":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"gsw":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"gu":
        return goog.i18n.pluralRules.hiSelect_(a, b);
    case"haw":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"he":
        return goog.i18n.pluralRules.heSelect_(a, b);
    case"hi":
        return goog.i18n.pluralRules.hiSelect_(a, b);
    case"hr":
        return goog.i18n.pluralRules.srSelect_(a, b);
    case"hu":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"hy":
        return goog.i18n.pluralRules.frSelect_(a, b);
    case"id":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"in":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"is":
        return goog.i18n.pluralRules.isSelect_(a, b);
    case"it":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"iw":
        return goog.i18n.pluralRules.heSelect_(a, b);
    case"ja":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"ka":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"kk":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"km":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"kn":
        return goog.i18n.pluralRules.hiSelect_(a, b);
    case"ko":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"ky":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"ln":
        return goog.i18n.pluralRules.akSelect_(a, b);
    case"lo":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"lt":
        return goog.i18n.pluralRules.ltSelect_(a, b);
    case"lv":
        return goog.i18n.pluralRules.lvSelect_(a, b);
    case"mk":
        return goog.i18n.pluralRules.mkSelect_(a, b);
    case"ml":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"mn":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"mo":
        return goog.i18n.pluralRules.roSelect_(a, b);
    case"mr":
        return goog.i18n.pluralRules.hiSelect_(a, b);
    case"ms":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"mt":
        return goog.i18n.pluralRules.mtSelect_(a, b);
    case"my":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"nb":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"ne":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"nl":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"no":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"no_NO":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"or":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"pa":
        return goog.i18n.pluralRules.akSelect_(a, b);
    case"pl":
        return goog.i18n.pluralRules.plSelect_(a, b);
    case"pt":
        return goog.i18n.pluralRules.ptSelect_(a, b);
    case"pt_BR":
        return goog.i18n.pluralRules.ptSelect_(a, b);
    case"pt_PT":
        return goog.i18n.pluralRules.pt_PTSelect_(a, b);
    case"ro":
        return goog.i18n.pluralRules.roSelect_(a, b);
    case"ru":
        return goog.i18n.pluralRules.ruSelect_(a, b);
    case"sh":
        return goog.i18n.pluralRules.srSelect_(a, b);
    case"si":
        return goog.i18n.pluralRules.siSelect_(a, b);
    case"sk":
        return goog.i18n.pluralRules.csSelect_(a, b);
    case"sl":
        return goog.i18n.pluralRules.slSelect_(a, b);
    case"sq":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"sr":
        return goog.i18n.pluralRules.srSelect_(a, b);
    case"sv":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"sw":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"ta":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"te":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"th":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"tl":
        return goog.i18n.pluralRules.filSelect_(a, b);
    case"tr":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"uk":
        return goog.i18n.pluralRules.ruSelect_(a, b);
    case"ur":
        return goog.i18n.pluralRules.enSelect_(a, b);
    case"uz":
        return goog.i18n.pluralRules.esSelect_(a, b);
    case"vi":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"zh":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"zh_CN":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"zh_HK":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"zh_TW":
        return goog.i18n.pluralRules.defaultSelect_(a, b);
    case"zu":
        return goog.i18n.pluralRules.hiSelect_(a, b);
    default:
        return goog.i18n.pluralRules.enSelect_(a, b)
    }
}, function(a, b) {
    "object" === typeof module && "object" === typeof module.exports ? module.exports = a.document ? b(a, !0) : function(a) {
        if (!a.document)
            throw new Error("jQuery requires a window with a document");
        return b(a)
    } : b(a)
}("undefined" !== typeof window ? window : this, function(a, b) {
    function c(a) {
        var b = a.length, c = eb.type(a);
        return "function" === c || eb.isWindow(a)?!1 : 1 === a.nodeType && b?!0 : "array" === c || 0 === b || "number" === typeof b && b > 0 && b - 1 in a
    }
    function d(a, b, c) {
        if (eb.isFunction(b))
            return eb.grep(a, function(a, d) {
                return !!b.call(a, d, a) !== c
            });
        if (b.nodeType)
            return eb.grep(a, function(a) {
                return a === b !== c
            });
        if ("string" === typeof b) {
            if (mb.test(b))
                return eb.filter(b, a, c);
            b = eb.filter(b, a)
        }
        return eb.grep(a, function(a) {
            return eb.inArray(a, b) >= 0 !== c
        })
    }
    function e(a, b) {
        do 
            a = a[b];
        while (a && 1 !== a.nodeType);
        return a
    }
    function f(a) {
        var b = ub[a] = {};
        return eb.each(a.match(tb) || [], function(a, c) {
            b[c]=!0
        }), b
    }
    function g() {
        ob.addEventListener ? (ob.removeEventListener("DOMContentLoaded", h, !1), a.removeEventListener("load", h, !1)) : (ob.detachEvent("onreadystatechange", h), a.detachEvent("onload", h))
    }
    function h() {
        (ob.addEventListener || "load" === event.type || "complete" === ob.readyState) && (g(), eb.ready())
    }
    function i(a, b, c) {
        if (void 0 === c && 1 === a.nodeType) {
            var d = "data-" + b.replace(zb, "-$1").toLowerCase();
            if (c = a.getAttribute(d), "string" === typeof c) {
                try {
                    c = "true" === c?!0 : "false" === c?!1 : "null" === c ? null : + c + "" === c?+c : yb.test(c) ? eb.parseJSON(c) : c
                } catch (e) {}
                eb.data(a, b, c)
            } else 
                c = void 0
        }
        return c
    }
    function j(a) {
        var b;
        for (b in a)
            if (("data" !== b ||!eb.isEmptyObject(a[b])) && "toJSON" !== b)
                return !1;
        return !0
    }
    function k(a, b, c, d) {
        if (eb.acceptData(a)) {
            var e, f, g = eb.expando, h = a.nodeType, i = h ? eb.cache: a, j = h ? a[g]: a[g] && g;
            if (j && i[j] && (d || i[j].data) || void 0 !== c || "string" !== typeof b)
                return j || (j = h ? a[g] = W.pop() || eb.guid++ : g), i[j] || (i[j] = h ? {} : {
                    toJSON: eb.noop
                }), ("object" === typeof b || "function" === typeof b) && (d ? i[j] = eb.extend(i[j], b) : i[j].data = eb.extend(i[j].data, b)), f = i[j], d || (f.data || (f.data = {}), f = f.data), void 0 !== c && (f[eb.camelCase(b)] = c), "string" === typeof b ? (e = f[b], null == e && (e = f[eb.camelCase(b)])) : e = f, e
        }
    }
    function l(a, b, c) {
        if (eb.acceptData(a)) {
            var d, e, f = a.nodeType, g = f ? eb.cache: a, h = f ? a[eb.expando]: eb.expando;
            if (g[h]) {
                if (b && (d = c ? g[h] : g[h].data)) {
                    eb.isArray(b) ? b = b.concat(eb.map(b, eb.camelCase)) : b in d ? b = [b] : (b = eb.camelCase(b), b = b in d ? [b] : b.split(" ")), e = b.length;
                    for (; e--;)
                        delete d[b[e]];
                    if (c?!j(d) : !eb.isEmptyObject(d)
                        )return 
                }(c || (delete g[h].data, j(g[h]))) && (f ? eb.cleanData([a], !0) : cb.deleteExpando || g != g.window ? delete g[h] : g[h] = null)
            }
        }
    }
    function m() {
        return !0
    }
    function n() {
        return !1
    }
    function o() {
        try {
            return ob.activeElement
        } catch (a) {}
    }
    function p(a) {
        var b = Kb.split("|"), c = a.createDocumentFragment();
        if (c.createElement)
            for (; b.length;)
                c.createElement(b.pop());
        return c
    }
    function q(a, b) {
        var c, d, e = 0, f = typeof a.getElementsByTagName !== xb ? a.getElementsByTagName(b || "*"): typeof a.querySelectorAll !== xb ? a.querySelectorAll(b || "*"): void 0;
        if (!f)
            for (f = [], c = a.childNodes || a; null != (d = c[e]); e++)
                !b || eb.nodeName(d, b) ? f.push(d) : eb.merge(f, q(d, b));
        return void 0 === b || b && eb.nodeName(a, b) ? eb.merge([a], f) : f
    }
    function r(a) {
        Eb.test(a.type) && (a.defaultChecked = a.checked)
    }
    function s(a, b) {
        return eb.nodeName(a, "table") && eb.nodeName(11 !== b.nodeType ? b : b.firstChild, "tr") ? a.getElementsByTagName("tbody")[0] || a.appendChild(a.ownerDocument.createElement("tbody")) : a
    }
    function t(a) {
        return a.type = (null !== eb.find.attr(a, "type")) + "/" + a.type, a
    }
    function u(a) {
        var b = Vb.exec(a.type);
        return b ? a.type = b[1] : a.removeAttribute("type"), a
    }
    function v(a, b) {
        for (var c, d = 0; null != (c = a[d]); d++)
            eb._data(c, "globalEval", !b || eb._data(b[d], "globalEval"))
    }
    function w(a, b) {
        if (1 === b.nodeType && eb.hasData(a)) {
            var c, d, e, f = eb._data(a), g = eb._data(b, f), h = f.events;
            if (h) {
                delete g.handle, g.events = {};
                for (c in h)
                    for (d = 0, e = h[c].length; d < e; d++)
                        eb.event.add(b, c, h[c][d])
                    }
            g.data && (g.data = eb.extend({}, g.data))
        }
    }
    function x(a, b) {
        var c, d, e;
        if (1 === b.nodeType) {
            if (c = b.nodeName.toLowerCase(), !cb.noCloneEvent && b[eb.expando]) {
                e = eb._data(b);
                for (d in e.events)
                    eb.removeEvent(b, d, e.handle);
                b.removeAttribute(eb.expando)
            }
            "script" === c && b.text !== a.text ? (t(b).text = a.text, u(b)) : "object" === c ? (b.parentNode && (b.outerHTML = a.outerHTML), cb.html5Clone && a.innerHTML&&!eb.trim(b.innerHTML) && (b.innerHTML = a.innerHTML)) : "input" === c && Eb.test(a.type) ? (b.defaultChecked = b.checked = a.checked, b.value !== a.value && (b.value = a.value)) : "option" === c ? b.defaultSelected = b.selected = a.defaultSelected : ("input" === c || "textarea" === c) && (b.defaultValue = a.defaultValue)
        }
    }
    function y(b, c) {
        var d, e = eb(c.createElement(b)).appendTo(c.body), f = a.getDefaultComputedStyle && (d = a.getDefaultComputedStyle(e[0])) ? d.display: eb.css(e[0], "display");
        return e.detach(), f
    }
    function z(a) {
        var b = ob, c = _b[a];
        return c || (c = y(a, b), "none" !== c && c || ($b = ($b || eb("<iframe frameborder='0' width='0' height='0'/>")).appendTo(b.documentElement), b = ($b[0].contentWindow || $b[0].contentDocument).document, b.write(), b.close(), c = y(a, b), $b.detach()), _b[a] = c), c
    }
    function A(a, b) {
        return {
            get: function() {
                var c = a();
                if (null != c)
                    return c ? void delete this.get : (this.get = b).apply(this, arguments)
            }
        }
    }
    function B(a, b) {
        if (b in a)
            return b;
        for (var c = b.charAt(0).toUpperCase() + b.slice(1), d = b, e = mc.length; e--;)
            if (b = mc[e] + c, b in a)
                return b;
        return d
    }
    function C(a, b) {
        for (var c, d, e, f = [], g = 0, h = a.length; g < h; g++)
            d = a[g], d.style && (f[g] = eb._data(d, "olddisplay"), c = d.style.display, b ? (f[g] || "none" !== c || (d.style.display = ""), "" === d.style.display && Cb(d) && (f[g] = eb._data(d, "olddisplay", z(d.nodeName)))) : (e = Cb(d), (c && "none" !== c ||!e) && eb._data(d, "olddisplay", e ? c : eb.css(d, "display"))));
        for (g = 0; g < h; g++)
            d = a[g], d.style && (b && "none" !== d.style.display && "" !== d.style.display || (d.style.display = b ? f[g] || "" : "none"));
        return a
    }
    function D(a, b, c) {
        var d = ic.exec(b);
        return d ? Math.max(0, d[1] - (c || 0)) + (d[2] || "px") : b
    }
    function E(a, b, c, d, e) {
        for (var f = c === (d ? "border" : "content") ? 4 : "width" === b ? 1 : 0, g = 0; f < 4; f += 2)
            "margin" === c && (g += eb.css(a, c + Bb[f], !0, e)), d ? ("content" === c && (g -= eb.css(a, "padding" + Bb[f], !0, e)), "margin" !== c && (g -= eb.css(a, "border" + Bb[f] + "Width", !0, e))) : (g += eb.css(a, "padding" + Bb[f], !0, e), "padding" !== c && (g += eb.css(a, "border" + Bb[f] + "Width", !0, e)));
        return g
    }
    function F(a, b, c) {
        var d=!0, e = "width" === b ? a.offsetWidth : a.offsetHeight, f = ac(a), g = cb.boxSizing && "border-box" === eb.css(a, "boxSizing", !1, f);
        if (e <= 0 || null == e) {
            if (e = bc(a, b, f), (e < 0 || null == e) && (e = a.style[b]), dc.test(e))
                return e;
            d = g && (cb.boxSizingReliable() || e === a.style[b]), e = parseFloat(e) || 0
        }
        return e + E(a, b, c || (g ? "border" : "content"), d, f) + "px"
    }
    function G(a, b, c, d, e) {
        return new G.prototype.init(a, b, c, d, e)
    }
    function H() {
        return setTimeout(function() {
            nc = void 0
        }), nc = eb.now()
    }
    function I(a, b) {
        var c, d = {
            height: a
        }, e = 0;
        for (b = b ? 1 : 0; e < 4; e += 2 - b)
            c = Bb[e], d["margin" + c] = d["padding" + c] = a;
        return b && (d.opacity = d.width = a), d
    }
    function J(a, b, c) {
        for (var d, e = (tc[b] || []).concat(tc["*"]), f = 0, g = e.length; f < g; f++)
            if (d = e[f].call(c, b, a))
                return d
    }
    function K(a, b, c) {
        var d, e, f, g, h, i, j, k, l = this, m = {}, n = a.style, o = a.nodeType && Cb(a), p = eb._data(a, "fxshow");
        c.queue || (h = eb._queueHooks(a, "fx"), null == h.unqueued && (h.unqueued = 0, i = h.empty.fire, h.empty.fire = function() {
            h.unqueued || i()
        }), h.unqueued++, l.always(function() {
            l.always(function() {
                h.unqueued--, eb.queue(a, "fx").length || h.empty.fire()
            })
        })), 1 === a.nodeType && ("height"in b || "width"in b) && (c.overflow = [n.overflow, n.overflowX, n.overflowY], j = eb.css(a, "display"), k = "none" === j ? eb._data(a, "olddisplay") || z(a.nodeName) : j, "inline" === k && "none" === eb.css(a, "float") && (cb.inlineBlockNeedsLayout && "inline" !== z(a.nodeName) ? n.zoom = 1 : n.display = "inline-block")), c.overflow && (n.overflow = "hidden", cb.shrinkWrapBlocks() || l.always(function() {
            n.overflow = c.overflow[0], n.overflowX = c.overflow[1], n.overflowY = c.overflow[2]
        }));
        for (d in b)
            if (e = b[d], pc.exec(e)) {
                if (delete b[d], f = f || "toggle" === e, e === (o ? "hide" : "show")) {
                    if ("show" !== e ||!p || void 0 === p[d])
                        continue;
                        o=!0
                }
                m[d] = p && p[d] || eb.style(a, d)
            } else 
                j = void 0;
        if (eb.isEmptyObject(m))
            "inline" === ("none" === j ? z(a.nodeName) : j) && (n.display = j);
        else {
            p ? "hidden"in p && (o = p.hidden) : p = eb._data(a, "fxshow", {}), f && (p.hidden=!o), o ? eb(a).show() : l.done(function() {
                eb(a).hide()
            }), l.done(function() {
                var b;
                eb._removeData(a, "fxshow");
                for (b in m)
                    eb.style(a, b, m[b])
            });
            for (d in m)
                g = J(o ? p[d] : 0, d, l), d in p || (p[d] = g.start, o && (g.end = g.start, g.start = "width" === d || "height" === d ? 1 : 0))
        }
    }
    function L(a, b) {
        var c, d, e, f, g;
        for (c in a)
            if (d = eb.camelCase(c), e = b[d], f = a[c], eb.isArray(f) && (e = f[1], f = a[c] = f[0]), c !== d && (a[d] = f, delete a[c]), g = eb.cssHooks[d], g && "expand"in g) {
                f = g.expand(f), delete a[d];
                for (c in f)
                    c in a || (a[c] = f[c], b[c] = e)
            } else 
                b[d] = e
    }
    function M(a, b, c) {
        var d, e, f = 0, g = sc.length, h = eb.Deferred().always(function() {
            delete i.elem
        }), i = function() {
            if (e)
                return !1;
            for (var b = nc || H(), c = Math.max(0, j.startTime + j.duration - b), d = c / j.duration || 0, f = 1 - d, g = 0, i = j.tweens.length; g < i; g++)
                j.tweens[g].run(f);
            return h.notifyWith(a, [j, f, c]), f < 1 && i ? c : (h.resolveWith(a, [j]), !1)
        }, j = h.promise({
            elem: a,
            props: eb.extend({}, b),
            opts: eb.extend(!0, {
                specialEasing: {}
            }, c),
            originalProperties: b,
            originalOptions: c,
            startTime: nc || H(),
            duration: c.duration,
            tweens: [],
            createTween: function(b, c) {
                var d = eb.Tween(a, j.opts, b, c, j.opts.specialEasing[b] || j.opts.easing);
                return j.tweens.push(d), d
            },
            stop: function(b) {
                var c = 0, d = b ? j.tweens.length: 0;
                if (e)
                    return this;
                for (e=!0; c < d; c++)
                    j.tweens[c].run(1);
                return b ? h.resolveWith(a, [j, b]) : h.rejectWith(a, [j, b]), this
            }
        }), k = j.props;
        for (L(k, j.opts.specialEasing);
        f < g;
        f++)if (d = sc[f].call(j, a, k, j.opts))
            return d;
        return eb.map(k, J, j), eb.isFunction(j.opts.start) && j.opts.start.call(a, j), eb.fx.timer(eb.extend(i, {
            elem: a,
            anim: j,
            queue: j.opts.queue
        })), j.progress(j.opts.progress).done(j.opts.done, j.opts.complete).fail(j.opts.fail).always(j.opts.always)
    }
    function N(a) {
        return function(b, c) {
            "string" !== typeof b && (c = b, b = "*");
            var d, e = 0, f = b.toLowerCase().match(tb) || [];
            if (eb.isFunction(c))
                for (; d = f[e++];)
                    "+" === d.charAt(0) ? (d = d.slice(1) || "*", (a[d] = a[d] || []).unshift(c)) : (a[d] = a[d] || []).push(c)
        }
    }
    function O(a, b, c, d) {
        function e(h) {
            var i;
            return f[h]=!0, eb.each(a[h] || [], function(a, h) {
                var j = h(b, c, d);
                return "string" !== typeof j || g || f[j] ? g?!(i = j) : void 0 : (b.dataTypes.unshift(j), e(j), !1)
            }), i
        }
        var f = {}, g = a === Rc;
        return e(b.dataTypes[0]) ||!f["*"] && e("*")
    }
    function P(a, b) {
        var c, d, e = eb.ajaxSettings.flatOptions || {};
        for (d in b)
            void 0 !== b[d] && ((e[d] ? a : c || (c = {}))[d] = b[d]);
        return c && eb.extend(!0, a, c), a
    }
    function Q(a, b, c) {
        for (var d, e, f, g, h = a.contents, i = a.dataTypes; "*" === i[0];)
            i.shift(), void 0 === e && (e = a.mimeType || b.getResponseHeader("Content-Type"));
        if (e)
            for (g in h)
                if (h[g] && h[g].test(e)) {
                    i.unshift(g);
                    break
                }
        if (i[0]in c)
            f = i[0];
        else {
            for (g in c) {
                if (!i[0] || a.converters[g + " " + i[0]]) {
                    f = g;
                    break
                }
                d || (d = g)
            }
            f = f || d
        }
        return f ? (f !== i[0] && i.unshift(f), c[f]) : void 0
    }
    function R(a, b, c, d) {
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
    function S(a, b, c, d) {
        var e;
        if (eb.isArray(b))
            eb.each(b, function(b, e) {
                c || Vc.test(a) ? d(a, e) : S(a + "[" + ("object" === typeof e ? b : "") + "]", e, c, d)
            });
        else if (c || "object" !== eb.type(b))
            d(a, b);
        else 
            for (e in b)
                S(a + "[" + e + "]", b[e], c, d)
            }
    function T() {
        try {
            return new a.XMLHttpRequest
        } catch (b) {}
    }
    function U() {
        try {
            return new a.ActiveXObject("Microsoft.XMLHTTP")
        } catch (b) {}
    }
    function V(a) {
        return eb.isWindow(a) ? a : 9 === a.nodeType ? a.defaultView || a.parentWindow : !1
    }
    var W = [], X = W.slice, Y = W.concat, Z = W.push, $ = W.indexOf, _ = {}, ab = _.toString, bb = _.hasOwnProperty, cb = {}, db = "1.11.1", eb = function(a, b) {
        return new eb.fn.init(a, b)
    }, fb = /^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g, gb = /^-ms-/, hb = /-([\da-z])/gi, ib = function(a, b) {
        return b.toUpperCase()
    };
    eb.fn = eb.prototype = {
        jquery: db,
        constructor: eb,
        selector: "",
        length: 0,
        toArray: function() {
            return X.call(this)
        },
        get: function(a) {
            return null != a ? a < 0 ? this[a + this.length] : this[a] : X.call(this)
        },
        pushStack: function(a) {
            var b = eb.merge(this.constructor(), a);
            return b.prevObject = this, b.context = this.context, b
        },
        each: function(a, b) {
            return eb.each(this, a, b)
        },
        map: function(a) {
            return this.pushStack(eb.map(this, function(b, c) {
                return a.call(b, c, b)
            }))
        },
        slice: function() {
            return this.pushStack(X.apply(this, arguments))
        },
        first: function() {
            return this.eq(0)
        },
        last: function() {
            return this.eq( - 1)
        },
        eq: function(a) {
            var b = this.length, c =+ a + (a < 0 ? b : 0);
            return this.pushStack(c >= 0 && c < b ? [this[c]] : [])
        },
        end: function() {
            return this.prevObject || this.constructor(null)
        },
        push: Z,
        sort: W.sort,
        splice: W.splice
    }, eb.extend = eb.fn.extend = function() {
        var a, b, c, d, e, f, g = arguments[0] || {}, h = 1, i = arguments.length, j=!1;
        for ("boolean" === typeof g && (j = g, g = arguments[h] || {}, h++), "object" === typeof g || eb.isFunction(g) || (g = {}), h === i && (g = this, h--); h < i; h++)
            if (null != (e = arguments[h]))
                for (d in e)
                    a = g[d], c = e[d], g !== c && (j && c && (eb.isPlainObject(c) || (b = eb.isArray(c))) ? (b ? (b=!1, f = a && eb.isArray(a) ? a : []) : f = a && eb.isPlainObject(a) ? a : {}, g[d] = eb.extend(j, f, c)) : void 0 !== c && (g[d] = c));
        return g
    }, eb.extend({
        expando: "jQuery" + (db + Math.random()).replace(/\D/g, ""),
        isReady: !0,
        error: function(a) {
            throw new Error(a)
        },
        noop: function() {},
        isFunction: function(a) {
            return "function" === eb.type(a)
        },
        isArray: Array.isArray || function(a) {
            return "array" === eb.type(a)
        },
        isWindow: function(a) {
            return null != a && a == a.window
        },
        isNumeric: function(a) {
            return !eb.isArray(a) && a - parseFloat(a) >= 0
        },
        isEmptyObject: function(a) {
            var b;
            for (b in a)
                return !1;
            return !0
        },
        isPlainObject: function(a) {
            var b;
            if (!a || "object" !== eb.type(a) || a.nodeType || eb.isWindow(a))
                return !1;
            try {
                if (a.constructor&&!bb.call(a, "constructor")&&!bb.call(a.constructor.prototype, "isPrototypeOf"))
                    return !1
            } catch (c) {
                return !1
            }
            if (cb.ownLast)
                for (b in a)
                    return bb.call(a, b);
            for (b in a);
            return void 0 === b || bb.call(a, b)
        },
        type: function(a) {
            return null == a ? a + "" : "object" === typeof a || "function" === typeof a ? _[ab.call(a)] || "object" : typeof a
        },
        globalEval: function(b) {
            b && eb.trim(b) && (a.execScript || function(b) {
                a.eval.call(a, b)
            })(b)
        },
        camelCase: function(a) {
            return a.replace(gb, "ms-").replace(hb, ib)
        },
        nodeName: function(a, b) {
            return a.nodeName && a.nodeName.toLowerCase() === b.toLowerCase()
        },
        each: function(a, b, d) {
            var e, f = 0, g = a.length, h = c(a);
            if (d) {
                if (h)
                    for (; f < g && (e = b.apply(a[f], d), e!==!1); f++);
                else 
                    for (f in a)
                        if (e = b.apply(a[f], d), e===!1)
                            break
            } else if (h)
                for (; f < g && (e = b.call(a[f], f, a[f]), e!==!1); f++);
            else 
                for (f in a)
                    if (e = b.call(a[f], f, a[f]), e===!1)
                        break;
            return a
        },
        trim: function(a) {
            return null == a ? "" : (a + "").replace(fb, "")
        },
        makeArray: function(a, b) {
            var d = b || [];
            return null != a && (c(Object(a)) ? eb.merge(d, "string" === typeof a ? [a] : a) : Z.call(d, a)), d
        },
        inArray: function(a, b, c) {
            var d;
            if (b) {
                if ($)
                    return $.call(b, a, c);
                for (d = b.length, c = c ? c < 0 ? Math.max(0, d + c) : c : 0; c < d; c++)
                    if (c in b && b[c] === a)
                        return c
            }
            return - 1
        },
        merge: function(a, b) {
            for (var c =+ b.length, d = 0, e = a.length; d < c;)
                a[e++] = b[d++];
            if (c !== c)
                for (; void 0 !== b[d];)
                    a[e++] = b[d++];
            return a.length = e, a
        },
        grep: function(a, b, c) {
            for (var d, e = [], f = 0, g = a.length, h=!c; f < g; f++)
                d=!b(a[f], f), d !== h && e.push(a[f]);
            return e
        },
        map: function(a, b, d) {
            var e, f = 0, g = a.length, h = c(a), i = [];
            if (h)
                for (; f < g; f++)
                    e = b(a[f], f, d), null != e && i.push(e);
            else 
                for (f in a)
                    e = b(a[f], f, d), null != e && i.push(e);
            return Y.apply([], i)
        },
        guid: 1,
        proxy: function(a, b) {
            var c, d, e;
            return "string" === typeof b && (e = a[b], b = a, a = e), eb.isFunction(a) ? (c = X.call(arguments, 2), d = function() {
                return a.apply(b || this, c.concat(X.call(arguments)))
            }, d.guid = a.guid = a.guid || eb.guid++, d) : void 0
        },
        now: function() {
            return + new Date
        },
        support: cb
    }), eb.each("Boolean Number String Function Array Date RegExp Object Error".split(" "), function(a, b) {
        _["[object " + b + "]"] = b.toLowerCase()
    });
    var jb = function(a) {
        function b(a, b, c, d) {
            var e, f, g, h, i, j, l, n, o, p;
            if ((b ? b.ownerDocument || b : O) !== G && F(b), b = b || G, c = c || [], !a || "string" !== typeof a)
                return c;
            if (1 !== (h = b.nodeType) && 9 !== h)
                return [];
            if (I&&!d) {
                if (e = sb.exec(a))
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
                            return _.apply(c, b.getElementsByTagName(a)), c;
                            if ((g = e[3]) && v.getElementsByClassName && b.getElementsByClassName)
                                return _.apply(c, b.getElementsByClassName(g)), c
                    }
                if (v.qsa && (!J ||!J.test(a))) {
                    if (n = l = N, o = b, p = 9 === h && a, 1 === h && "object" !== b.nodeName.toLowerCase()) {
                        for (j = z(a), (l = b.getAttribute("id")) ? n = l.replace(ub, "\\$&") : b.setAttribute("id", n), n = "[id='" + n + "'] ", i = j.length; i--;)
                            j[i] = n + m(j[i]);
                        o = tb.test(a) && k(b.parentNode) || b, p = j.join(",")
                    }
                    if (p)
                        try {
                            return _.apply(c, o.querySelectorAll(p)), c
                    } catch (q) {} finally {
                        l || b.removeAttribute("id")
                    }
                }
            }
            return B(a.replace(ib, "$1"), b, c, d)
        }
        function c() {
            function a(c, d) {
                return b.push(c + " ") > w.cacheLength && delete a[b.shift()], a[c + " "] = d
            }
            var b = [];
            return a
        }
        function d(a) {
            return a[N]=!0, a
        }
        function e(a) {
            var b = G.createElement("div");
            try {
                return !!a(b)
            } catch (c) {
                return !1
            } finally {
                b.parentNode && b.parentNode.removeChild(b), b = null
            }
        }
        function f(a, b) {
            for (var c = a.split("|"), d = a.length; d--;)
                w.attrHandle[c[d]] = b
        }
        function g(a, b) {
            var c = b && a, d = c && 1 === a.nodeType && 1 === b.nodeType && (~b.sourceIndex || W) - (~a.sourceIndex || W);
            if (d)
                return d;
            if (c)
                for (; c = c.nextSibling;)
                    if (c === b)
                        return - 1;
            return a ? 1 : - 1
        }
        function h(a) {
            return function(b) {
                var c = b.nodeName.toLowerCase();
                return "input" === c && b.type === a
            }
        }
        function i(a) {
            return function(b) {
                var c = b.nodeName.toLowerCase();
                return ("input" === c || "button" === c) && b.type === a
            }
        }
        function j(a) {
            return d(function(b) {
                return b =+ b, d(function(c, d) {
                    for (var e, f = a([], c.length, b), g = f.length; g--;)
                        c[e = f[g]] && (c[e]=!(d[e] = c[e]))
                })
            })
        }
        function k(a) {
            return a && typeof a.getElementsByTagName !== V && a
        }
        function l() {}
        function m(a) {
            for (var b = 0, c = a.length, d = ""; b < c; b++)
                d += a[b].value;
            return d
        }
        function n(a, b, c) {
            var d = b.dir, e = c && "parentNode" === d, f = Q++;
            return b.first ? function(b, c, f) {
                for (; b = b[d];)
                    if (1 === b.nodeType || e)
                        return a(b, c, f)
            } : function(b, c, g) {
                var h, i, j = [P, f];
                if (g) {
                    for (; b = b[d];)
                        if ((1 === b.nodeType || e) && a(b, c, g))
                            return !0
                } else 
                    for (; b = b[d];)
                        if (1 === b.nodeType || e) {
                            if (i = b[N] || (b[N] = {}), (h = i[d]) && h[0] === P && h[1] === f)
                                return j[2] = h[2];
                                if (i[d] = j, j[2] = a(b, c, g))
                                    return !0
                        }
            }
        }
        function o(a) {
            return a.length > 1 ? function(b, c, d) {
                for (var e = a.length; e--;)
                    if (!a[e](b, c, d))
                        return !1;
                return !0
            } : a[0]
        }
        function p(a, c, d) {
            for (var e = 0, f = c.length; e < f; e++)
                b(a, c[e], d);
            return d
        }
        function q(a, b, c, d, e) {
            for (var f, g = [], h = 0, i = a.length, j = null != b; h < i; h++)(f = a[h]) 
                && (!c || c(f, d, e)) && (g.push(f), j && b.push(h));
            return g
        }
        function r(a, b, c, e, f, g) {
            return e&&!e[N] && (e = r(e)), f&&!f[N] && (f = r(f, g)), d(function(d, g, h, i) {
                var j, k, l, m = [], n = [], o = g.length, r = d || p(b || "*", h.nodeType ? [h] : h, []), s=!a ||!d && b ? r : q(r, m, a, h, i), t = c ? f || (d ? a : o || e) ? [] : g : s;
                if (c && c(s, t, h, i), e)
                    for (j = q(t, n), e(j, [], h, i), k = j.length; k--;)(l = j[k]) 
                        && (t[n[k]]=!(s[n[k]] = l));
                if (d) {
                    if (f || a) {
                        if (f) {
                            for (j = [], k = t.length; k--;)(l = t[k]) 
                                && j.push(s[k] = l);
                            f(null, t = [], j, i)
                        }
                        for (k = t.length; k--;)(l = t[k]) 
                            && (j = f ? bb.call(d, l) : m[k])>-1 && (d[j]=!(g[j] = l))
                        }
                } else 
                    t = q(t === g ? t.splice(o, t.length) : t), f ? f(null, g, t, i) : _.apply(g, t)
            })
        }
        function s(a) {
            for (var b, c, d, e = a.length, f = w.relative[a[0].type], g = f || w.relative[" "], h = f ? 1 : 0, i = n(function(a) {
                return a === b
            }, g, !0), j = n(function(a) {
                return bb.call(b, a)>-1
            }, g, !0), k = [function(a, c, d) {
                return !f && (d || c !== C) || ((b = c).nodeType ? i(a, c, d) : j(a, c, d))
            }
            ]; h < e; h++)
                if (c = w.relative[a[h].type])
                    k = [n(o(k), c)];
                else {
                    if (c = w.filter[a[h].type].apply(null, a[h].matches), c[N]) {
                        for (d=++h; d < e&&!w.relative[a[d].type]; d++);
                        return r(h > 1 && o(k), h > 1 && m(a.slice(0, h - 1).concat({
                            value: " " === a[h - 2].type ? "*": ""
                        })).replace(ib, "$1"), c, h < d && s(a.slice(h, d)), d < e && s(a = a.slice(d)), d < e && m(a))
                    }
                    k.push(c)
                }
            return o(k)
        }
        function t(a, c) {
            var e = c.length > 0, f = a.length > 0, g = function(d, g, h, i, j) {
                var k, l, m, n = 0, o = "0", p = d && [], r = [], s = C, t = d || f && w.find.TAG("*", j), u = P += null == s ? 1: Math.random() || .1, v = t.length;
                for (j && (C = g !== G && g); o !== v && null != (k = t[o]); o++) {
                    if (f && k) {
                        for (l = 0; m = a[l++];)
                            if (m(k, g, h)) {
                                i.push(k);
                                break
                            }
                        j && (P = u)
                    }
                    e && ((k=!m && k) && n--, d && p.push(k))
                }
                if (n += o, e && o !== n) {
                    for (l = 0; m = c[l++];)
                        m(p, r, g, h);
                    if (d) {
                        if (n > 0)
                            for (; o--;)
                                p[o] || r[o] || (r[o] = Z.call(i));
                        r = q(r)
                    }
                    _.apply(i, r), j&&!d && r.length > 0 && n + c.length > 1 && b.uniqueSort(i)
                }
                return j && (P = u, C = s), p
            };
            return e ? d(g) : g
        }
        var u, v, w, x, y, z, A, B, C, D, E, F, G, H, I, J, K, L, M, N = "sizzle" +- new Date, O = a.document, P = 0, Q = 0, R = c(), S = c(), T = c(), U = function(a, b) {
            return a === b && (E=!0), 0
        }, V = "undefined", W = 1<<31, X = {}.hasOwnProperty, Y = [], Z = Y.pop, $ = Y.push, _ = Y.push, ab = Y.slice, bb = Y.indexOf || function(a) {
            for (var b = 0, c = this.length; b < c; b++)
                if (this[b] === a)
                    return b;
            return - 1
        }, cb = "checked|selected|async|autofocus|autoplay|controls|defer|disabled|hidden|ismap|loop|multiple|open|readonly|required|scoped", db = "[\\x20\\t\\r\\n\\f]", eb = "(?:\\\\.|[\\w-]|[^\\x00-\\xa0])+", fb = eb.replace("w", "w#"), gb = "\\[" + db + "*(" + eb + ")(?:" + db + "*([*^$|!~]?=)" + db + "*(?:'((?:\\\\.|[^\\\\'])*)'|\"((?:\\\\.|[^\\\\\"])*)\"|(" + fb + "))|)" + db + "*\\]", hb = ":(" + eb + ")(?:\\((('((?:\\\\.|[^\\\\'])*)'|\"((?:\\\\.|[^\\\\\"])*)\")|((?:\\\\.|[^\\\\()[\\]]|" + gb + ")*)|.*)\\)|)", ib = new RegExp("^" + db + "+|((?:^|[^\\\\])(?:\\\\.)*)" + db + "+$", "g"), jb = new RegExp("^" + db + "*," + db + "*"), kb = new RegExp("^" + db + "*([>+~]|" + db + ")" + db + "*"), lb = new RegExp("=" + db + "*([^\\]'\"]*?)" + db + "*\\]", "g"), mb = new RegExp(hb), nb = new RegExp("^" + fb + "$"), ob = {
            ID: new RegExp("^#(" + eb + ")"),
            CLASS: new RegExp("^\\.(" + eb + ")"),
            TAG: new RegExp("^(" + eb.replace("w", "w*") + ")"),
            ATTR: new RegExp("^" + gb),
            PSEUDO: new RegExp("^" + hb),
            CHILD: new RegExp("^:(only|first|last|nth|nth-last)-(child|of-type)(?:\\(" + db + "*(even|odd|(([+-]|)(\\d*)n|)" + db + "*(?:([+-]|)" + db + "*(\\d+)|))" + db + "*\\)|)", "i"),
            bool: new RegExp("^(?:" + cb + ")$", "i"),
            needsContext: new RegExp("^" + db + "*[>+~]|:(even|odd|eq|gt|lt|nth|first|last)(?:\\(" + db + "*((?:-\\d)?\\d*)" + db + "*\\)|)(?=[^-]|$)", "i")
        }, pb = /^(?:input|select|textarea|button)$/i, qb = /^h\d$/i, rb = /^[^{]+\{\s*\[native \w/, sb = /^(?:#([\w-]+)|(\w+)|\.([\w-]+))$/, tb = /[+~]/, ub = /'|\\/g, vb = new RegExp("\\\\([\\da-f]{1,6}" + db + "?|(" + db + ")|.)", "ig"), wb = function(a, b, c) {
            var d = "0x" + b - 65536;
            return d !== d || c ? b : d < 0 ? String.fromCharCode(d + 65536) : String.fromCharCode(d>>10 | 55296, 1023 & d | 56320)
        };
        try {
            _.apply(Y = ab.call(O.childNodes), O.childNodes), Y[O.childNodes.length].nodeType
        } catch (xb) {
            _ = {
                apply: Y.length ? function(a, b) {
                    $.apply(a, ab.call(b))
                }
                : function(a, b) {
                    for (var c = a.length, d = 0; a[c++] = b[d++];);
                    a.length = c - 1
                }
            }
        }
        v = b.support = {}, y = b.isXML = function(a) {
            var b = a && (a.ownerDocument || a).documentElement;
            return b ? "HTML" !== b.nodeName : !1
        }, F = b.setDocument = function(a) {
            var b, c = a ? a.ownerDocument || a: O, d = c.defaultView;
            return c !== G && 9 === c.nodeType && c.documentElement ? (G = c, H = c.documentElement, I=!y(c), d && d !== d.top && (d.addEventListener ? d.addEventListener("unload", function() {
                F()
            }, !1) : d.attachEvent && d.attachEvent("onunload", function() {
                F()
            })), v.attributes = e(function(a) {
                return a.className = "i", !a.getAttribute("className")
            }), v.getElementsByTagName = e(function(a) {
                return a.appendChild(c.createComment("")), !a.getElementsByTagName("*").length
            }), v.getElementsByClassName = rb.test(c.getElementsByClassName) && e(function(a) {
                return a.innerHTML = "<div class='a'></div><div class='a i'></div>", a.firstChild.className = "i", 2 === a.getElementsByClassName("i").length
            }), v.getById = e(function(a) {
                return H.appendChild(a).id = N, !c.getElementsByName ||!c.getElementsByName(N).length
            }), v.getById ? (w.find.ID = function(a, b) {
                if (typeof b.getElementById !== V && I) {
                    var c = b.getElementById(a);
                    return c && c.parentNode ? [c] : []
                }
            }, w.filter.ID = function(a) {
                var b = a.replace(vb, wb);
                return function(a) {
                    return a.getAttribute("id") === b
                }
            }) : (delete w.find.ID, w.filter.ID = function(a) {
                var b = a.replace(vb, wb);
                return function(a) {
                    var c = typeof a.getAttributeNode !== V && a.getAttributeNode("id");
                    return c && c.value === b
                }
            }), w.find.TAG = v.getElementsByTagName ? function(a, b) {
                return typeof b.getElementsByTagName !== V ? b.getElementsByTagName(a) : void 0
            } : function(a, b) {
                var c, d = [], e = 0, f = b.getElementsByTagName(a);
                if ("*" === a) {
                    for (; c = f[e++];)
                        1 === c.nodeType && d.push(c);
                    return d
                }
                return f
            }, w.find.CLASS = v.getElementsByClassName && function(a, b) {
                return typeof b.getElementsByClassName !== V && I ? b.getElementsByClassName(a) : void 0
            }, K = [], J = [], (v.qsa = rb.test(c.querySelectorAll)) && (e(function(a) {
                a.innerHTML = "<select msallowclip=''><option selected=''></option></select>", a.querySelectorAll("[msallowclip^='']").length && J.push("[*^$]=" + db + "*(?:''|\"\")"), a.querySelectorAll("[selected]").length || J.push("\\[" + db + "*(?:value|" + cb + ")"), a.querySelectorAll(":checked").length || J.push(":checked")
            }), e(function(a) {
                var b = c.createElement("input");
                b.setAttribute("type", "hidden"), a.appendChild(b).setAttribute("name", "D"), a.querySelectorAll("[name=d]").length && J.push("name" + db + "*[*^$|!~]?="), a.querySelectorAll(":enabled").length || J.push(":enabled", ":disabled"), a.querySelectorAll("*,:x"), J.push(",.*:")
            })), (v.matchesSelector = rb.test(L = H.matches || H.webkitMatchesSelector || H.mozMatchesSelector || H.oMatchesSelector || H.msMatchesSelector)) && e(function(a) {
                v.disconnectedMatch = L.call(a, "div"), L.call(a, "[s!='']:x"), K.push("!=", hb)
            }), J = J.length && new RegExp(J.join("|")), K = K.length && new RegExp(K.join("|")), b = rb.test(H.compareDocumentPosition), M = b || rb.test(H.contains) ? function(a, b) {
                var c = 9 === a.nodeType ? a.documentElement: a, d = b && b.parentNode;
                return a === d ||!(!d || 1 !== d.nodeType ||!(c.contains ? c.contains(d) : a.compareDocumentPosition && 16 & a.compareDocumentPosition(d)))
            } : function(a, b) {
                if (b)
                    for (; b = b.parentNode;)
                        if (b === a)
                            return !0;
                return !1
            }, U = b ? function(a, b) {
                if (a === b)
                    return E=!0, 0;
                var d=!a.compareDocumentPosition-!b.compareDocumentPosition;
                return d ? d : (d = (a.ownerDocument || a) === (b.ownerDocument || b) ? a.compareDocumentPosition(b) : 1, 1 & d ||!v.sortDetached && b.compareDocumentPosition(a) === d ? a === c || a.ownerDocument === O && M(O, a)?-1 : b === c || b.ownerDocument === O && M(O, b) ? 1 : D ? bb.call(D, a) - bb.call(D, b) : 0 : 4 & d?-1 : 1)
            } : function(a, b) {
                if (a === b)
                    return E=!0, 0;
                var d, e = 0, f = a.parentNode, h = b.parentNode, i = [a], j = [b];
                if (!f ||!h)
                    return a === c?-1 : b === c ? 1 : f?-1 : h ? 1 : D ? bb.call(D, a) - bb.call(D, b) : 0;
                if (f === h)
                    return g(a, b);
                for (d = a; d = d.parentNode;)
                    i.unshift(d);
                for (d = b; d = d.parentNode;)
                    j.unshift(d);
                for (; i[e] === j[e];)
                    e++;
                return e ? g(i[e], j[e]) : i[e] === O?-1 : j[e] === O ? 1 : 0
            }, c) : G
        }, b.matches = function(a, c) {
            return b(a, null, null, c)
        }, b.matchesSelector = function(a, c) {
            if ((a.ownerDocument || a) !== G && F(a), c = c.replace(lb, "='$1']"), v.matchesSelector && I && (!K ||!K.test(c)) && (!J ||!J.test(c)))
                try {
                    var d = L.call(a, c);
                    if (d || v.disconnectedMatch || a.document && 11 !== a.document.nodeType)
                        return d
            } catch (e) {}
            return b(c, G, null, [a]).length > 0
        }, b.contains = function(a, b) {
            return (a.ownerDocument || a) !== G && F(a), M(a, b)
        }, b.attr = function(a, b) {
            (a.ownerDocument || a) !== G && F(a);
            var c = w.attrHandle[b.toLowerCase()], d = c && X.call(w.attrHandle, b.toLowerCase()) ? c(a, b, !I): void 0;
            return void 0 !== d ? d : v.attributes ||!I ? a.getAttribute(b) : (d = a.getAttributeNode(b)) && d.specified ? d.value : null
        }, b.error = function(a) {
            throw new Error("Syntax error, unrecognized expression: " + a)
        }, b.uniqueSort = function(a) {
            var b, c = [], d = 0, e = 0;
            if (E=!v.detectDuplicates, D=!v.sortStable && a.slice(0), a.sort(U), E) {
                for (; b = a[e++];)
                    b === a[e] && (d = c.push(e));
                for (; d--;)
                    a.splice(c[d], 1)
            }
            return D = null, a
        }, x = b.getText = function(a) {
            var b, c = "", d = 0, e = a.nodeType;
            if (e) {
                if (1 === e || 9 === e || 11 === e) {
                    if ("string" === typeof a.textContent)
                        return a.textContent;
                    for (a = a.firstChild; a; a = a.nextSibling)
                        c += x(a)
                    } else if (3 === e || 4 === e)
                    return a.nodeValue
            } else 
                for (; b = a[d++];)
                    c += x(b);
            return c
        }, w = b.selectors = {
            cacheLength: 50,
            createPseudo: d,
            match: ob,
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
                    return a[1] = a[1].replace(vb, wb), a[3] = (a[3] || a[4] || a[5] || "").replace(vb, wb), "~=" === a[2] && (a[3] = " " + a[3] + " "), a.slice(0, 4)
                },
                CHILD: function(a) {
                    return a[1] = a[1].toLowerCase(), "nth" === a[1].slice(0, 3) ? (a[3] || b.error(a[0]), a[4] =+ (a[4] ? a[5] + (a[6] || 1) : 2 * ("even" === a[3] || "odd" === a[3])), a[5] =+ (a[7] + a[8] || "odd" === a[3])) : a[3] && b.error(a[0]), a
                },
                PSEUDO: function(a) {
                    var b, c=!a[6] && a[2];
                    return ob.CHILD.test(a[0]) ? null : (a[3] ? a[2] = a[4] || a[5] || "" : c && mb.test(c) && (b = z(c, !0)) && (b = c.indexOf(")", c.length - b) - c.length) && (a[0] = a[0].slice(0, b), a[2] = c.slice(0, b)), a.slice(0, 3))
                }
            },
            filter: {
                TAG: function(a) {
                    var b = a.replace(vb, wb).toLowerCase();
                    return "*" === a ? function() {
                        return !0
                    } : function(a) {
                        return a.nodeName && a.nodeName.toLowerCase() === b
                    }
                },
                CLASS: function(a) {
                    var b = R[a + " "];
                    return b || (b = new RegExp("(^|" + db + ")" + a + "(" + db + "|$)")) && R(a, function(a) {
                        return b.test("string" === typeof a.className && a.className || typeof a.getAttribute !== V && a.getAttribute("class") || "")
                    })
                },
                ATTR: function(a, c, d) {
                    return function(e) {
                        var f = b.attr(e, a);
                        return null == f ? "!=" === c : c ? (f += "", "=" === c ? f === d : "!=" === c ? f !== d : "^=" === c ? d && 0 === f.indexOf(d) : "*=" === c ? d && f.indexOf(d)>-1 : "$=" === c ? d && f.slice( - d.length) === d : "~=" === c ? (" " + f + " ").indexOf(d)>-1 : "|=" === c ? f === d || f.slice(0, d.length + 1) === d + "-" : !1) : !0
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
                PSEUDO: function(a, c) {
                    var e, f = w.pseudos[a] || w.setFilters[a.toLowerCase()] || b.error("unsupported pseudo: " + a);
                    return f[N] ? f(c) : f.length > 1 ? (e = [a, a, "", c], w.setFilters.hasOwnProperty(a.toLowerCase()) ? d(function(a, b) {
                        for (var d, e = f(a, c), g = e.length; g--;)
                            d = bb.call(a, e[g]), a[d]=!(b[d] = e[g])
                    }) : function(a) {
                        return f(a, 0, e)
                    }) : f
                }
            },
            pseudos: {
                not: d(function(a) {
                    var b = [], c = [], e = A(a.replace(ib, "$1"));
                    return e[N] ? d(function(a, b, c, d) {
                        for (var f, g = e(a, null, d, []), h = a.length; h--;)(f = g[h]) 
                            && (a[h]=!(b[h] = f))
                    }) : function(a, d, f) {
                        return b[0] = a, e(b, null, f, c), !c.pop()
                    }
                }),
                has: d(function(a) {
                    return function(c) {
                        return b(a, c).length > 0
                    }
                }),
                contains: d(function(a) {
                    return function(b) {
                        return (b.textContent || b.innerText || x(b)).indexOf(a)>-1
                    }
                }),
                lang: d(function(a) {
                    return nb.test(a || "") || b.error("unsupported lang: " + a), a = a.replace(vb, wb).toLowerCase(), function(b) {
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
                        if (a.nodeType < 6)
                            return !1;
                    return !0
                },
                parent: function(a) {
                    return !w.pseudos.empty(a)
                },
                header: function(a) {
                    return qb.test(a.nodeName)
                },
                input: function(a) {
                    return pb.test(a.nodeName)
                },
                button: function(a) {
                    var b = a.nodeName.toLowerCase();
                    return "input" === b && "button" === a.type || "button" === b
                },
                text: function(a) {
                    var b;
                    return "input" === a.nodeName.toLowerCase() && "text" === a.type && (null == (b = a.getAttribute("type")) || "text" === b.toLowerCase())
                },
                first: j(function() {
                    return [0]
                }),
                last: j(function(a, b) {
                    return [b - 1]
                }),
                eq: j(function(a, b, c) {
                    return [c < 0 ? c + b: c]
                }),
                even: j(function(a, b) {
                    for (var c = 0; c < b; c += 2)
                        a.push(c);
                    return a
                }),
                odd: j(function(a, b) {
                    for (var c = 1; c < b; c += 2)
                        a.push(c);
                    return a
                }),
                lt: j(function(a, b, c) {
                    for (var d = c < 0 ? c + b : c; --d >= 0;)
                        a.push(d);
                    return a
                }),
                gt: j(function(a, b, c) {
                    for (var d = c < 0 ? c + b : c; ++d < b;)
                        a.push(d);
                    return a
                })
            }
        }, w.pseudos.nth = w.pseudos.eq;
        for (u in{
            radio: !0,
            checkbox: !0,
            file: !0,
            password: !0,
            image: !0
        })
            w.pseudos[u] = h(u);
        for (u in{
            submit: !0,
            reset: !0
        })
            w.pseudos[u] = i(u);
        return l.prototype = w.filters = w.pseudos, w.setFilters = new l, z = b.tokenize = function(a, c) {
            var d, e, f, g, h, i, j, k = S[a + " "];
            if (k)
                return c ? 0 : k.slice(0);
            for (h = a, i = [], j = w.preFilter; h;) {
                (!d || (e = jb.exec(h))) && (e && (h = h.slice(e[0].length) || h), i.push(f = [])), d=!1, (e = kb.exec(h)) && (d = e.shift(), f.push({
                    value: d,
                    type: e[0].replace(ib, " ")
                }), h = h.slice(d.length));
                for (g in w.filter)
                    !(e = ob[g].exec(h)) || j[g]&&!(e = j[g](e)) || (d = e.shift(), f.push({
                        value: d,
                        type: g,
                        matches: e
                    }), h = h.slice(d.length));
                if (!d)
                    break
            }
            return c ? h.length : h ? b.error(a) : S(a, i).slice(0)
        }, A = b.compile = function(a, b) {
            var c, d = [], e = [], f = T[a + " "];
            if (!f) {
                for (b || (b = z(a)), c = b.length; c--;)
                    f = s(b[c]), f[N] ? d.push(f) : e.push(f);
                f = T(a, t(e, d)), f.selector = a
            }
            return f
        }, B = b.select = function(a, b, c, d) {
            var e, f, g, h, i, j = "function" === typeof a && a, l=!d && z(a = j.selector || a);
            if (c = c || [], 1 === l.length) {
                if (f = l[0] = l[0].slice(0), f.length > 2 && "ID" === (g = f[0]).type && v.getById && 9 === b.nodeType && I && w.relative[f[1].type]) {
                    if (b = (w.find.ID(g.matches[0].replace(vb, wb), b) || [])[0], !b)
                        return c;
                    j && (b = b.parentNode), a = a.slice(f.shift().value.length)
                }
                for (e = ob.needsContext.test(a) ? 0 : f.length; e--&&(g = f[e], !w.relative[h = g.type]);)
                    if ((i = w.find[h]) && (d = i(g.matches[0].replace(vb, wb), tb.test(f[0].type) && k(b.parentNode) || b))) {
                        if (f.splice(e, 1), a = d.length && m(f), !a)
                            return _.apply(c, d), c;
                            break
                    }
            }
            return (j || A(a, l))(d, b, !I, c, tb.test(a) && k(b.parentNode) || b), c
        }, v.sortStable = N.split("").sort(U).join("") === N, v.detectDuplicates=!!E, F(), v.sortDetached = e(function(a) {
            return 1 & a.compareDocumentPosition(G.createElement("div"))
        }), e(function(a) {
            return a.innerHTML = "<a href='#'></a>", "#" === a.firstChild.getAttribute("href")
        }) || f("type|href|height|width", function(a, b, c) {
            return c ? void 0 : a.getAttribute(b, "type" === b.toLowerCase() ? 1 : 2)
        }), v.attributes && e(function(a) {
            return a.innerHTML = "<input/>", a.firstChild.setAttribute("value", ""), "" === a.firstChild.getAttribute("value")
        }) || f("value", function(a, b, c) {
            return c || "input" !== a.nodeName.toLowerCase() ? void 0 : a.defaultValue
        }), e(function(a) {
            return null == a.getAttribute("disabled")
        }) || f(cb, function(a, b, c) {
            var d;
            return c ? void 0 : a[b]===!0 ? b.toLowerCase() : (d = a.getAttributeNode(b)) && d.specified ? d.value : null
        }), b
    }(a);
    eb.find = jb, eb.expr = jb.selectors, eb.expr[":"] = eb.expr.pseudos, eb.unique = jb.uniqueSort, eb.text = jb.getText, eb.isXMLDoc = jb.isXML, eb.contains = jb.contains;
    var kb = eb.expr.match.needsContext, lb = /^<(\w+)\s*\/?>(?:<\/\1>|)$/, mb = /^.[^:#\[\.,]*$/;
    eb.filter = function(a, b, c) {
        var d = b[0];
        return c && (a = ":not(" + a + ")"), 1 === b.length && 1 === d.nodeType ? eb.find.matchesSelector(d, a) ? [d] : [] : eb.find.matches(a, eb.grep(b, function(a) {
            return 1 === a.nodeType
        }))
    }, eb.fn.extend({
        find: function(a) {
            var b, c = [], d = this, e = d.length;
            if ("string" !== typeof a)
                return this.pushStack(eb(a).filter(function() {
                    for (b = 0; b < e; b++)
                        if (eb.contains(d[b], this))
                            return !0
                        }));
            for (b = 0; b < e; b++)
                eb.find(a, d[b], c);
            return c = this.pushStack(e > 1 ? eb.unique(c) : c), c.selector = this.selector ? this.selector + " " + a : a, c
        },
        filter: function(a) {
            return this.pushStack(d(this, a || [], !1))
        },
        not: function(a) {
            return this.pushStack(d(this, a || [], !0))
        },
        is: function(a) {
            return !!d(this, "string" === typeof a && kb.test(a) ? eb(a) : a || [], !1).length
        }
    });
    var nb, ob = a.document, pb = /^(?:\s*(<[\w\W]+>)[^>]*|#([\w-]*))$/, qb = eb.fn.init = function(a, b) {
        var c, d;
        if (!a)
            return this;
        if ("string" === typeof a) {
            if (c = "<" === a.charAt(0) && ">" === a.charAt(a.length - 1) && a.length >= 3 ? [null, a, null] : pb.exec(a), !c ||!c[1] && b)
                return !b || b.jquery ? (b || nb).find(a) : this.constructor(b).find(a);
            if (c[1]) {
                if (b = b instanceof eb ? b[0] : b, eb.merge(this, eb.parseHTML(c[1], b && b.nodeType ? b.ownerDocument || b : ob, !0)), lb.test(c[1]) && eb.isPlainObject(b))
                    for (c in b)
                        eb.isFunction(this[c]) ? this[c](b[c]) : this.attr(c, b[c]);
                return this
            }
            if (d = ob.getElementById(c[2]), d && d.parentNode) {
                if (d.id !== c[2])
                    return nb.find(a);
                this.length = 1, this[0] = d
            }
            return this.context = ob, this.selector = a, this
        }
        return a.nodeType ? (this.context = this[0] = a, this.length = 1, this) : eb.isFunction(a) ? "undefined" !== typeof nb.ready ? nb.ready(a) : a(eb) : (void 0 !== a.selector && (this.selector = a.selector, this.context = a.context), eb.makeArray(a, this))
    };
    qb.prototype = eb.fn, nb = eb(ob);
    var rb = /^(?:parents|prev(?:Until|All))/, sb = {
        children: !0,
        contents: !0,
        next: !0,
        prev: !0
    };
    eb.extend({
        dir: function(a, b, c) {
            for (var d = [], e = a[b]; e && 9 !== e.nodeType && (void 0 === c || 1 !== e.nodeType ||!eb(e).is(c));
            )1 === e.nodeType && d.push(e), e = e[b];
            return d
        },
        sibling: function(a, b) {
            for (var c = []; a; a = a.nextSibling)
                1 === a.nodeType && a !== b && c.push(a);
            return c
        }
    }), eb.fn.extend({
        has: function(a) {
            var b, c = eb(a, this), d = c.length;
            return this.filter(function() {
                for (b = 0; b < d; b++)
                    if (eb.contains(this, c[b]))
                        return !0
            })
        },
        closest: function(a, b) {
            for (var c, d = 0, e = this.length, f = [], g = kb.test(a) || "string" !== typeof a ? eb(a, b || this.context) : 0; d < e; d++)
                for (c = this[d]; c && c !== b; c = c.parentNode)
                    if (c.nodeType < 11 && (g ? g.index(c)>-1 : 1 === c.nodeType && eb.find.matchesSelector(c, a))) {
                        f.push(c);
                        break
                    }
            return this.pushStack(f.length > 1 ? eb.unique(f) : f)
        },
        index: function(a) {
            return a ? "string" === typeof a ? eb.inArray(this[0], eb(a)) : eb.inArray(a.jquery ? a[0] : a, this) : this[0] && this[0].parentNode ? this.first().prevAll().length : - 1
        },
        add: function(a, b) {
            return this.pushStack(eb.unique(eb.merge(this.get(), eb(a, b))))
        },
        addBack: function(a) {
            return this.add(null == a ? this.prevObject : this.prevObject.filter(a))
        }
    }), eb.each({
        parent: function(a) {
            var b = a.parentNode;
            return b && 11 !== b.nodeType ? b : null
        },
        parents: function(a) {
            return eb.dir(a, "parentNode")
        },
        parentsUntil: function(a, b, c) {
            return eb.dir(a, "parentNode", c)
        },
        next: function(a) {
            return e(a, "nextSibling")
        },
        prev: function(a) {
            return e(a, "previousSibling")
        },
        nextAll: function(a) {
            return eb.dir(a, "nextSibling")
        },
        prevAll: function(a) {
            return eb.dir(a, "previousSibling")
        },
        nextUntil: function(a, b, c) {
            return eb.dir(a, "nextSibling", c)
        },
        prevUntil: function(a, b, c) {
            return eb.dir(a, "previousSibling", c)
        },
        siblings: function(a) {
            return eb.sibling((a.parentNode || {}).firstChild, a)
        },
        children: function(a) {
            return eb.sibling(a.firstChild)
        },
        contents: function(a) {
            return eb.nodeName(a, "iframe") ? a.contentDocument || a.contentWindow.document : eb.merge([], a.childNodes)
        }
    }, function(a, b) {
        eb.fn[a] = function(c, d) {
            var e = eb.map(this, b, c);
            return "Until" !== a.slice( - 5) && (d = c), d && "string" === typeof d && (e = eb.filter(d, e)), this.length > 1 && (sb[a] || (e = eb.unique(e)), rb.test(a) && (e = e.reverse())), this.pushStack(e)
        }
    });
    var tb = /\S+/g, ub = {};
    eb.Callbacks = function(a) {
        a = "string" === typeof a ? ub[a] || f(a) : eb.extend({}, a);
        var b, c, d, e, g, h, i = [], j=!a.once && [], k = function(f) {
            for (c = a.memory && f, d=!0, g = h || 0, h = 0, e = i.length, b=!0; i && g < e; g++)
                if (i[g].apply(f[0], f[1])===!1 && a.stopOnFalse) {
                    c=!1;
                    break
                }
            b=!1, i && (j ? j.length && k(j.shift()) : c ? i = [] : l.disable())
        }, l = {
            add: function() {
                if (i) {
                    var d = i.length;
                    !function f(b) {
                        eb.each(b, function(b, c) {
                            var d = eb.type(c);
                            "function" === d ? a.unique && l.has(c) || i.push(c) : c && c.length && "string" !== d && f(c)
                        })
                    }(arguments), b ? e = i.length : c && (h = d, k(c))
                }
                return this
            },
            remove: function() {
                return i && eb.each(arguments, function(a, c) {
                    for (var d; (d = eb.inArray(c, i, d))>-1;)
                        i.splice(d, 1), b && (d <= e && e--, d <= g && g--)
                }), this
            },
            has: function(a) {
                return a ? eb.inArray(a, i)>-1 : !(!i ||!i.length)
            },
            empty: function() {
                return i = [], e = 0, this
            },
            disable: function() {
                return i = j = c = void 0, this
            },
            disabled: function() {
                return !i
            },
            lock: function() {
                return j = void 0, c || l.disable(), this
            },
            locked: function() {
                return !j
            },
            fireWith: function(a, c) {
                return !i || d&&!j || (c = c || [], c = [a, c.slice ? c.slice(): c], b ? j.push(c) : k(c)), this
            },
            fire: function() {
                return l.fireWith(this, arguments), this
            },
            fired: function() {
                return !!d
            }
        };
        return l
    }, eb.extend({
        Deferred: function(a) {
            var b = [["resolve", "done", eb.Callbacks("once memory"), "resolved"], ["reject", "fail", eb.Callbacks("once memory"), "rejected"], ["notify", "progress", eb.Callbacks("memory")]], c = "pending", d = {
                state: function() {
                    return c
                },
                always: function() {
                    return e.done(arguments).fail(arguments), this
                },
                then: function() {
                    var a = arguments;
                    return eb.Deferred(function(c) {
                        eb.each(b, function(b, f) {
                            var g = eb.isFunction(a[b]) && a[b];
                            e[f[1]](function() {
                                var a = g && g.apply(this, arguments);
                                a && eb.isFunction(a.promise) ? a.promise().done(c.resolve).fail(c.reject).progress(c.notify) : c[f[0] + "With"](this === d ? c.promise() : this, g ? [a] : arguments)
                            })
                        }), a = null
                    }).promise()
                },
                promise: function(a) {
                    return null != a ? eb.extend(a, d) : d
                }
            }, e = {};
            return d.pipe = d.then, eb.each(b, function(a, f) {
                var g = f[2], h = f[3];
                d[f[1]] = g.add, h && g.add(function() {
                    c = h
                }, b[1^a][2].disable, b[2][2].lock), e[f[0]] = function() {
                    return e[f[0] + "With"](this === e ? d : this, arguments), this
                }, e[f[0] + "With"] = g.fireWith
            }), d.promise(e), a && a.call(e, e), e
        },
        when: function(a) {
            var b, c, d, e = 0, f = X.call(arguments), g = f.length, h = 1 !== g || a && eb.isFunction(a.promise) ? g: 0, i = 1 === h ? a: eb.Deferred(), j = function(a, c, d) {
                return function(e) {
                    c[a] = this, d[a] = arguments.length > 1 ? X.call(arguments) : e, d === b ? i.notifyWith(c, d) : --h || i.resolveWith(c, d)
                }
            };
            if (g > 1)
                for (b = new Array(g), c = new Array(g), d = new Array(g); e < g; e++)
                    f[e] && eb.isFunction(f[e].promise) ? f[e].promise().done(j(e, d, f)).fail(i.reject).progress(j(e, c, b)) : --h;
            return h || i.resolveWith(d, f), i.promise()
        }
    });
    var vb;
    eb.fn.ready = function(a) {
        return eb.ready.promise().done(a), this
    }, eb.extend({
        isReady: !1,
        readyWait: 1,
        holdReady: function(a) {
            a ? eb.readyWait++ : eb.ready(!0)
        },
        ready: function(a) {
            if (a===!0?!--eb.readyWait : !eb.isReady) {
                if (!ob.body)
                    return setTimeout(eb.ready);
                eb.isReady=!0, a!==!0&&--eb.readyWait > 0 || (vb.resolveWith(ob, [eb]), eb.fn.triggerHandler && (eb(ob).triggerHandler("ready"), eb(ob).off("ready")))
            }
        }
    }), eb.ready.promise = function(b) {
        if (!vb)
            if (vb = eb.Deferred(), "complete" === ob.readyState)
                setTimeout(eb.ready);
            else if (ob.addEventListener)
                ob.addEventListener("DOMContentLoaded", h, !1), a.addEventListener("load", h, !1);
            else {
                ob.attachEvent("onreadystatechange", h), a.attachEvent("onload", h);
                var c=!1;
                try {
                    c = null == a.frameElement && ob.documentElement
                } catch (d) {}
                c && c.doScroll&&!function e() {
                    if (!eb.isReady) {
                        try {
                            c.doScroll("left")
                        } catch (a) {
                            return setTimeout(e, 50)
                        }
                        g(), eb.ready()
                    }
                }()
            }
        return vb.promise(b)
    };
    var wb, xb = "undefined";
    for (wb in eb(cb))
        break;
    cb.ownLast = "0" !== wb, cb.inlineBlockNeedsLayout=!1, eb(function() {
        var a, b, c, d;
        c = ob.getElementsByTagName("body")[0], c && c.style && (b = ob.createElement("div"), d = ob.createElement("div"), d.style.cssText = "position:absolute;border:0;width:0;height:0;top:0;left:-9999px", c.appendChild(d).appendChild(b), typeof b.style.zoom !== xb && (b.style.cssText = "display:inline;margin:0;border:0;padding:1px;width:1px;zoom:1", cb.inlineBlockNeedsLayout = a = 3 === b.offsetWidth, a && (c.style.zoom = 1)), c.removeChild(d))
    }), function() {
        var a = ob.createElement("div");
        if (null == cb.deleteExpando) {
            cb.deleteExpando=!0;
            try {
                delete a.test
            } catch (b) {
                cb.deleteExpando=!1
            }
        }
        a = null
    }(), eb.acceptData = function(a) {
        var b = eb.noData[(a.nodeName + " ").toLowerCase()], c =+ a.nodeType || 1;
        return 1 !== c && 9 !== c?!1 : !b || b!==!0 && a.getAttribute("classid") === b
    };
    var yb = /^(?:\{[\w\W]*\}|\[[\w\W]*\])$/, zb = /([A-Z])/g;
    eb.extend({
        cache: {},
        noData: {
            "applet ": !0,
            "embed ": !0,
            "object ": "clsid:D27CDB6E-AE6D-11cf-96B8-444553540000"
        },
        hasData: function(a) {
            return a = a.nodeType ? eb.cache[a[eb.expando]] : a[eb.expando], !!a&&!j(a)
        },
        data: function(a, b, c) {
            return k(a, b, c)
        },
        removeData: function(a, b) {
            return l(a, b)
        },
        _data: function(a, b, c) {
            return k(a, b, c, !0)
        },
        _removeData: function(a, b) {
            return l(a, b, !0)
        }
    }), eb.fn.extend({
        data: function(a, b) {
            var c, d, e, f = this[0], g = f && f.attributes;
            if (void 0 === a) {
                if (this.length && (e = eb.data(f), 1 === f.nodeType&&!eb._data(f, "parsedAttrs"))) {
                    for (c = g.length; c--;)
                        g[c] && (d = g[c].name, 0 === d.indexOf("data-") && (d = eb.camelCase(d.slice(5)), i(f, d, e[d])));
                    eb._data(f, "parsedAttrs", !0)
                }
                return e
            }
            return "object" === typeof a ? this.each(function() {
                eb.data(this, a)
            }) : arguments.length > 1 ? this.each(function() {
                eb.data(this, a, b)
            }) : f ? i(f, a, eb.data(f, a)) : void 0
        },
        removeData: function(a) {
            return this.each(function() {
                eb.removeData(this, a)
            })
        }
    }), eb.extend({
        queue: function(a, b, c) {
            var d;
            return a ? (b = (b || "fx") + "queue", d = eb._data(a, b), c && (!d || eb.isArray(c) ? d = eb._data(a, b, eb.makeArray(c)) : d.push(c)), d || []) : void 0
        },
        dequeue: function(a, b) {
            b = b || "fx";
            var c = eb.queue(a, b), d = c.length, e = c.shift(), f = eb._queueHooks(a, b), g = function() {
                eb.dequeue(a, b)
            };
            "inprogress" === e && (e = c.shift(), d--), e && ("fx" === b && c.unshift("inprogress"), delete f.stop, e.call(a, g, f)), !d && f && f.empty.fire()
        },
        _queueHooks: function(a, b) {
            var c = b + "queueHooks";
            return eb._data(a, c) || eb._data(a, c, {
                empty: eb.Callbacks("once memory").add(function() {
                    eb._removeData(a, b + "queue"), eb._removeData(a, c)
                })
            })
        }
    }), eb.fn.extend({
        queue: function(a, b) {
            var c = 2;
            return "string" !== typeof a && (b = a, a = "fx", c--), arguments.length < c ? eb.queue(this[0], a) : void 0 === b ? this : this.each(function() {
                var c = eb.queue(this, a, b);
                eb._queueHooks(this, a), "fx" === a && "inprogress" !== c[0] && eb.dequeue(this, a)
            })
        },
        dequeue: function(a) {
            return this.each(function() {
                eb.dequeue(this, a)
            })
        },
        clearQueue: function(a) {
            return this.queue(a || "fx", [])
        },
        promise: function(a, b) {
            var c, d = 1, e = eb.Deferred(), f = this, g = this.length, h = function() {
                --d || e.resolveWith(f, [f])
            };
            for ("string" !== typeof a && (b = a, a = void 0), a = a || "fx"; g--;)
                c = eb._data(f[g], a + "queueHooks"), c && c.empty && (d++, c.empty.add(h));
            return h(), e.promise(b)
        }
    });
    var Ab = /[+-]?(?:\d*\.|)\d+(?:[eE][+-]?\d+|)/.source, Bb = ["Top", "Right", "Bottom", "Left"], Cb = function(a, b) {
        return a = b || a, "none" === eb.css(a, "display") ||!eb.contains(a.ownerDocument, a)
    }, Db = eb.access = function(a, b, c, d, e, f, g) {
        var h = 0, i = a.length, j = null == c;
        if ("object" === eb.type(c)) {
            e=!0;
            for (h in c)
                eb.access(a, b, h, c[h], !0, f, g)
        } else if (void 0 !== d && (e=!0, eb.isFunction(d) || (g=!0), j && (g ? (b.call(a, d), b = null) : (j = b, b = function(a, b, c) {
            return j.call(eb(a), c)
        })), b))for (; h < i; h++)
            b(a[h], c, g ? d : d.call(a[h], h, b(a[h], c)));
        return e ? a : j ? b.call(a) : i ? b(a[0], c) : f
    }, Eb = /^(?:checkbox|radio)$/i;
    !function() {
        var a = ob.createElement("input"), b = ob.createElement("div"), c = ob.createDocumentFragment();
        if (b.innerHTML = "  <link/><table></table><a href='/a'>a</a><input type='checkbox'/>", cb.leadingWhitespace = 3 === b.firstChild.nodeType, cb.tbody=!b.getElementsByTagName("tbody").length, cb.htmlSerialize=!!b.getElementsByTagName("link").length, cb.html5Clone = "<:nav></:nav>" !== ob.createElement("nav").cloneNode(!0).outerHTML, a.type = "checkbox", a.checked=!0, c.appendChild(a), cb.appendChecked = a.checked, b.innerHTML = "<textarea>x</textarea>", cb.noCloneChecked=!!b.cloneNode(!0).lastChild.defaultValue, c.appendChild(b), b.innerHTML = "<input type='radio' checked='checked' name='t'/>", cb.checkClone = b.cloneNode(!0).cloneNode(!0).lastChild.checked, cb.noCloneEvent=!0, b.attachEvent && (b.attachEvent("onclick", function() {
            cb.noCloneEvent=!1
        }), b.cloneNode(!0).click()), null == cb.deleteExpando) {
            cb.deleteExpando=!0;
            try {
                delete b.test
            } catch (d) {
                cb.deleteExpando=!1
            }
        }
    }(), function() {
        var b, c, d = ob.createElement("div");
        for (b in{
            submit: !0,
            change: !0,
            focusin: !0
        })
            c = "on" + b, (cb[b + "Bubbles"] = c in a) || (d.setAttribute(c, "t"), cb[b + "Bubbles"] = d.attributes[c].expando===!1);
        d = null
    }();
    var Fb = /^(?:input|select|textarea)$/i, Gb = /^key/, Hb = /^(?:mouse|pointer|contextmenu)|click/, Ib = /^(?:focusinfocus|focusoutblur)$/, Jb = /^([^.]*)(?:\.(.+)|)$/;
    eb.event = {
        global: {},
        add: function(a, b, c, d, e) {
            var f, g, h, i, j, k, l, m, n, o, p, q = eb._data(a);
            if (q) {
                for (c.handler && (i = c, c = i.handler, e = i.selector), c.guid || (c.guid = eb.guid++), (g = q.events) || (g = q.events = {}), (k = q.handle) || (k = q.handle = function(a) {
                    return typeof eb === xb || a && eb.event.triggered === a.type ? void 0 : eb.event.dispatch.apply(k.elem, arguments)
                }, k.elem = a), b = (b || "").match(tb) || [""], h = b.length; h--;)
                    f = Jb.exec(b[h]) || [], n = p = f[1], o = (f[2] || "").split(".").sort(), n && (j = eb.event.special[n] || {}, n = (e ? j.delegateType : j.bindType) || n, j = eb.event.special[n] || {}, l = eb.extend({
                        type: n,
                        origType: p,
                        data: d,
                        handler: c,
                        guid: c.guid,
                        selector: e,
                        needsContext: e && eb.expr.match.needsContext.test(e),
                        namespace: o.join(".")
                    }, i), (m = g[n]) || (m = g[n] = [], m.delegateCount = 0, j.setup && j.setup.call(a, d, o, k)!==!1 || (a.addEventListener ? a.addEventListener(n, k, !1) : a.attachEvent && a.attachEvent("on" + n, k))), j.add && (j.add.call(a, l), l.handler.guid || (l.handler.guid = c.guid)), e ? m.splice(m.delegateCount++, 0, l) : m.push(l), eb.event.global[n]=!0);
                a = null
            }
        },
        remove: function(a, b, c, d, e) {
            var f, g, h, i, j, k, l, m, n, o, p, q = eb.hasData(a) && eb._data(a);
            if (q && (k = q.events)) {
                for (b = (b || "").match(tb) || [""], j = b.length; j--;)
                    if (h = Jb.exec(b[j]) || [], n = p = h[1], o = (h[2] || "").split(".").sort(), n) {
                        for (l = eb.event.special[n] || {}, n = (d ? l.delegateType : l.bindType) || n, m = k[n] || [], h = h[2] && new RegExp("(^|\\.)" + o.join("\\.(?:.*\\.|)") + "(\\.|$)"), i = f = m.length; f--;)
                            g = m[f], !e && p !== g.origType || c && c.guid !== g.guid || h&&!h.test(g.namespace) || d && d !== g.selector && ("**" !== d ||!g.selector) || (m.splice(f, 1), g.selector && m.delegateCount--, l.remove && l.remove.call(a, g));
                            i&&!m.length && (l.teardown && l.teardown.call(a, o, q.handle)!==!1 || eb.removeEvent(a, n, q.handle), delete k[n])
                    } else 
                        for (n in k)
                            eb.event.remove(a, n + b[j], c, d, !0);
                eb.isEmptyObject(k) && (delete q.handle, eb._removeData(a, "events"))
            }
        },
        trigger: function(b, c, d, e) {
            var f, g, h, i, j, k, l, m = [d || ob], n = bb.call(b, "type") ? b.type: b, o = bb.call(b, "namespace") ? b.namespace.split("."): [];
            if (h = k = d = d || ob, 3 !== d.nodeType && 8 !== d.nodeType&&!Ib.test(n + eb.event.triggered) && (n.indexOf(".") >= 0 && (o = n.split("."), n = o.shift(), o.sort()), g = n.indexOf(":") < 0 && "on" + n, b = b[eb.expando] ? b : new eb.Event(n, "object" === typeof b && b), b.isTrigger = e ? 2 : 3, b.namespace = o.join("."), b.namespace_re = b.namespace ? new RegExp("(^|\\.)" + o.join("\\.(?:.*\\.|)") + "(\\.|$)") : null, b.result = void 0, b.target || (b.target = d), c = null == c ? [b] : eb.makeArray(c, [b]), j = eb.event.special[n] || {}, e ||!j.trigger || j.trigger.apply(d, c)!==!1)
                ) {
                if (!e&&!j.noBubble&&!eb.isWindow(d)) {
                    for (i = j.delegateType || n, Ib.test(i + n) || (h = h.parentNode); h; h = h.parentNode)
                        m.push(h), k = h;
                    k === (d.ownerDocument || ob) && m.push(k.defaultView || k.parentWindow || a)
                }
                for (l = 0; (h = m[l++])&&!b.isPropagationStopped();)
                    b.type = l > 1 ? i : j.bindType || n, f = (eb._data(h, "events") || {})[b.type] && eb._data(h, "handle"), f && f.apply(h, c), f = g && h[g], f && f.apply && eb.acceptData(h) && (b.result = f.apply(h, c), b.result===!1 && b.preventDefault());
                if (b.type = n, !e&&!b.isDefaultPrevented() && (!j._default || j._default.apply(m.pop(), c)===!1) && eb.acceptData(d) && g && d[n]&&!eb.isWindow(d)) {
                    k = d[g], k && (d[g] = null), eb.event.triggered = n;
                    try {
                        d[n]()
                    } catch (p) {}
                    eb.event.triggered = void 0, k && (d[g] = k)
                }
                return b.result
            }
        },
        dispatch: function(a) {
            a = eb.event.fix(a);
            var b, c, d, e, f, g = [], h = X.call(arguments), i = (eb._data(this, "events") || {})[a.type] || [], j = eb.event.special[a.type] || {};
            if (h[0] = a, a.delegateTarget = this, !j.preDispatch || j.preDispatch.call(this, a)!==!1) {
                for (g = eb.event.handlers.call(this, a, i), b = 0; (e = g[b++])&&!a.isPropagationStopped();)
                    for (a.currentTarget = e.elem, f = 0; (d = e.handlers[f++])&&!a.isImmediatePropagationStopped();)(!a.namespace_re || a.namespace_re.test(d.namespace)
                        ) && (a.handleObj = d, a.data = d.data, c = ((eb.event.special[d.origType] || {}).handle || d.handler).apply(e.elem, h), void 0 !== c && (a.result = c)===!1 && (a.preventDefault(), a.stopPropagation()));
                return j.postDispatch && j.postDispatch.call(this, a), a.result
            }
        },
        handlers: function(a, b) {
            var c, d, e, f, g = [], h = b.delegateCount, i = a.target;
            if (h && i.nodeType && (!a.button || "click" !== a.type))
                for (; i != this; i = i.parentNode || this)
                    if (1 === i.nodeType && (i.disabled!==!0 || "click" !== a.type)) {
                        for (e = [], f = 0; f < h; f++)
                            d = b[f], c = d.selector + " ", void 0 === e[c] && (e[c] = d.needsContext ? eb(c, this).index(i) >= 0 : eb.find(c, this, null, [i]).length), e[c] && e.push(d);
                            e.length && g.push({
                                elem: i,
                                handlers: e
                            })
                    }
            return h < b.length && g.push({
                elem: this,
                handlers: b.slice(h)
            }), g
        },
        fix: function(a) {
            if (a[eb.expando])
                return a;
            var b, c, d, e = a.type, f = a, g = this.fixHooks[e];
            for (g || (this.fixHooks[e] = g = Hb.test(e) ? this.mouseHooks : Gb.test(e) ? this.keyHooks : {}), d = g.props ? this.props.concat(g.props) : this.props, a = new eb.Event(f), b = d.length; b--;)
                c = d[b], a[c] = f[c];
            return a.target || (a.target = f.srcElement || ob), 3 === a.target.nodeType && (a.target = a.target.parentNode), a.metaKey=!!a.metaKey, g.filter ? g.filter(a, f) : a
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
            filter: function(a, b) {
                var c, d, e, f = b.button, g = b.fromElement;
                return null == a.pageX && null != b.clientX && (d = a.target.ownerDocument || ob, e = d.documentElement, c = d.body, a.pageX = b.clientX + (e && e.scrollLeft || c && c.scrollLeft || 0) - (e && e.clientLeft || c && c.clientLeft || 0), a.pageY = b.clientY + (e && e.scrollTop || c && c.scrollTop || 0) - (e && e.clientTop || c && c.clientTop || 0)), !a.relatedTarget && g && (a.relatedTarget = g === a.target ? b.toElement : g), a.which || void 0 === f || (a.which = 1 & f ? 1 : 2 & f ? 3 : 4 & f ? 2 : 0), a
            }
        },
        special: {
            load: {
                noBubble: !0
            },
            focus: {
                trigger: function() {
                    if (this !== o() && this.focus)
                        try {
                            return this.focus(), !1
                    } catch (a) {}
                },
                delegateType: "focusin"
            },
            blur: {
                trigger: function() {
                    return this === o() && this.blur ? (this.blur(), !1) : void 0
                },
                delegateType: "focusout"
            },
            click: {
                trigger: function() {
                    return eb.nodeName(this, "input") && "checkbox" === this.type && this.click ? (this.click(), !1) : void 0
                },
                _default: function(a) {
                    return eb.nodeName(a.target, "a")
                }
            },
            beforeunload: {
                postDispatch: function(a) {
                    void 0 !== a.result && a.originalEvent && (a.originalEvent.returnValue = a.result)
                }
            }
        },
        simulate: function(a, b, c, d) {
            var e = eb.extend(new eb.Event, c, {
                type: a,
                isSimulated: !0,
                originalEvent: {}
            });
            d ? eb.event.trigger(e, null, b) : eb.event.dispatch.call(b, e), e.isDefaultPrevented() && c.preventDefault()
        }
    }, eb.removeEvent = ob.removeEventListener ? function(a, b, c) {
        a.removeEventListener && a.removeEventListener(b, c, !1)
    } : function(a, b, c) {
        var d = "on" + b;
        a.detachEvent && (typeof a[d] === xb && (a[d] = null), a.detachEvent(d, c))
    }, eb.Event = function(a, b) {
        return this instanceof eb.Event ? (a && a.type ? (this.originalEvent = a, this.type = a.type, this.isDefaultPrevented = a.defaultPrevented || void 0 === a.defaultPrevented && a.returnValue===!1 ? m : n) : this.type = a, b && eb.extend(this, b), this.timeStamp = a && a.timeStamp || eb.now(), void(this[eb.expando]=!0)) : new eb.Event(a, b)
    }, eb.Event.prototype = {
        isDefaultPrevented: n,
        isPropagationStopped: n,
        isImmediatePropagationStopped: n,
        preventDefault: function() {
            var a = this.originalEvent;
            this.isDefaultPrevented = m, a && (a.preventDefault ? a.preventDefault() : a.returnValue=!1)
        },
        stopPropagation: function() {
            var a = this.originalEvent;
            this.isPropagationStopped = m, a && (a.stopPropagation && a.stopPropagation(), a.cancelBubble=!0)
        },
        stopImmediatePropagation: function() {
            var a = this.originalEvent;
            this.isImmediatePropagationStopped = m, a && a.stopImmediatePropagation && a.stopImmediatePropagation(), this.stopPropagation()
        }
    }, eb.each({
        mouseenter: "mouseover",
        mouseleave: "mouseout",
        pointerenter: "pointerover",
        pointerleave: "pointerout"
    }, function(a, b) {
        eb.event.special[a] = {
            delegateType: b,
            bindType: b,
            handle: function(a) {
                var c, d = this, e = a.relatedTarget, f = a.handleObj;
                return (!e || e !== d&&!eb.contains(d, e)) && (a.type = f.origType, c = f.handler.apply(this, arguments), a.type = b), c
            }
        }
    }), cb.submitBubbles || (eb.event.special.submit = {
        setup: function() {
            return eb.nodeName(this, "form")?!1 : void eb.event.add(this, "click._submit keypress._submit", function(a) {
                var b = a.target, c = eb.nodeName(b, "input") || eb.nodeName(b, "button") ? b.form: void 0;
                c&&!eb._data(c, "submitBubbles") && (eb.event.add(c, "submit._submit", function(a) {
                    a._submit_bubble=!0
                }), eb._data(c, "submitBubbles", !0))
            })
        },
        postDispatch: function(a) {
            a._submit_bubble && (delete a._submit_bubble, this.parentNode&&!a.isTrigger && eb.event.simulate("submit", this.parentNode, a, !0))
        },
        teardown: function() {
            return eb.nodeName(this, "form")?!1 : void eb.event.remove(this, "._submit")
        }
    }), cb.changeBubbles || (eb.event.special.change = {
        setup: function() {
            return Fb.test(this.nodeName) ? (("checkbox" === this.type || "radio" === this.type) && (eb.event.add(this, "propertychange._change", function(a) {
                "checked" === a.originalEvent.propertyName && (this._just_changed=!0)
            }), eb.event.add(this, "click._change", function(a) {
                this._just_changed&&!a.isTrigger && (this._just_changed=!1), eb.event.simulate("change", this, a, !0)
            })), !1) : void eb.event.add(this, "beforeactivate._change", function(a) {
                var b = a.target;
                Fb.test(b.nodeName)&&!eb._data(b, "changeBubbles") && (eb.event.add(b, "change._change", function(a) {
                    !this.parentNode || a.isSimulated || a.isTrigger || eb.event.simulate("change", this.parentNode, a, !0)
                }), eb._data(b, "changeBubbles", !0))
            })
        },
        handle: function(a) {
            var b = a.target;
            return this !== b || a.isSimulated || a.isTrigger || "radio" !== b.type && "checkbox" !== b.type ? a.handleObj.handler.apply(this, arguments) : void 0
        },
        teardown: function() {
            return eb.event.remove(this, "._change"), !Fb.test(this.nodeName)
        }
    }), cb.focusinBubbles || eb.each({
        focus: "focusin",
        blur: "focusout"
    }, function(a, b) {
        var c = function(a) {
            eb.event.simulate(b, a.target, eb.event.fix(a), !0)
        };
        eb.event.special[b] = {
            setup: function() {
                var d = this.ownerDocument || this, e = eb._data(d, b);
                e || d.addEventListener(a, c, !0), eb._data(d, b, (e || 0) + 1)
            },
            teardown: function() {
                var d = this.ownerDocument || this, e = eb._data(d, b) - 1;
                e ? eb._data(d, b, e) : (d.removeEventListener(a, c, !0), eb._removeData(d, b))
            }
        }
    }), eb.fn.extend({
        on: function(a, b, c, d, e) {
            var f, g;
            if ("object" === typeof a) {
                "string" !== typeof b && (c = c || b, b = void 0);
                for (f in a)
                    this.on(f, b, c, a[f], e);
                return this
            }
            if (null == c && null == d ? (d = b, c = b = void 0) : null == d && ("string" === typeof b ? (d = c, c = void 0) : (d = c, c = b, b = void 0)), d===!1)
                d = n;
            else if (!d)
                return this;
            return 1 === e && (g = d, d = function(a) {
                return eb().off(a), g.apply(this, arguments)
            }, d.guid = g.guid || (g.guid = eb.guid++)), this.each(function() {
                eb.event.add(this, a, d, c, b)
            })
        },
        one: function(a, b, c, d) {
            return this.on(a, b, c, d, 1)
        },
        off: function(a, b, c) {
            var d, e;
            if (a && a.preventDefault && a.handleObj)
                return d = a.handleObj, eb(a.delegateTarget).off(d.namespace ? d.origType + "." + d.namespace : d.origType, d.selector, d.handler), this;
            if ("object" === typeof a) {
                for (e in a)
                    this.off(e, b, a[e]);
                return this
            }
            return (b===!1 || "function" === typeof b) && (c = b, b = void 0), c===!1 && (c = n), this.each(function() {
                eb.event.remove(this, a, c, b)
            })
        },
        trigger: function(a, b) {
            return this.each(function() {
                eb.event.trigger(a, b, this)
            })
        },
        triggerHandler: function(a, b) {
            var c = this[0];
            return c ? eb.event.trigger(a, b, c, !0) : void 0
        }
    });
    var Kb = "abbr|article|aside|audio|bdi|canvas|data|datalist|details|figcaption|figure|footer|header|hgroup|mark|meter|nav|output|progress|section|summary|time|video", Lb = / jQuery\d+="(?:null|\d+)"/g, Mb = new RegExp("<(?:" + Kb + ")[\\s/>]", "i"), Nb = /^\s+/, Ob = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\w:]+)[^>]*)\/>/gi, Pb = /<([\w:]+)/, Qb = /<tbody/i, Rb = /<|&#?\w+;/, Sb = /<(?:script|style|link)/i, Tb = /checked\s*(?:[^=]|=\s*.checked.)/i, Ub = /^$|\/(?:java|ecma)script/i, Vb = /^true\/(.*)/, Wb = /^\s*<!(?:\[CDATA\[|--)|(?:\]\]|--)>\s*$/g, Xb = {
        option: [1, "<select multiple='multiple'>", "</select>"],
        legend: [1, "<fieldset>", "</fieldset>"],
        area: [1, "<map>", "</map>"],
        param: [1, "<object>", "</object>"],
        thead: [1, "<table>", "</table>"],
        tr: [2, "<table><tbody>", "</tbody></table>"],
        col: [2, "<table><tbody></tbody><colgroup>", "</colgroup></table>"],
        td: [3, "<table><tbody><tr>", "</tr></tbody></table>"],
        _default: cb.htmlSerialize ? [0, "", ""]: [1, "X<div>", "</div>"]
    }, Yb = p(ob), Zb = Yb.appendChild(ob.createElement("div"));
    Xb.optgroup = Xb.option, Xb.tbody = Xb.tfoot = Xb.colgroup = Xb.caption = Xb.thead, Xb.th = Xb.td, eb.extend({
        clone: function(a, b, c) {
            var d, e, f, g, h, i = eb.contains(a.ownerDocument, a);
            if (cb.html5Clone || eb.isXMLDoc(a) ||!Mb.test("<" + a.nodeName + ">") ? f = a.cloneNode(!0) : (Zb.innerHTML = a.outerHTML, Zb.removeChild(f = Zb.firstChild)), (!cb.noCloneEvent ||!cb.noCloneChecked) && (1 === a.nodeType || 11 === a.nodeType)&&!eb.isXMLDoc(a))
                for (d = q(f), h = q(a), g = 0; null != (e = h[g]); ++g)
                    d[g] && x(e, d[g]);
            if (b)
                if (c)
                    for (h = h || q(a), d = d || q(f), g = 0; null != (e = h[g]); g++)
                        w(e, d[g]);
                else 
                    w(a, f);
            return d = q(f, "script"), d.length > 0 && v(d, !i && q(a, "script")), d = h = e = null, f
        },
        buildFragment: function(a, b, c, d) {
            for (var e, f, g, h, i, j, k, l = a.length, m = p(b), n = [], o = 0; o < l; o++)
                if (f = a[o], f || 0 === f)
                    if ("object" === eb.type(f))
                        eb.merge(n, f.nodeType ? [f] : f);
                    else if (Rb.test(f)) {
                        for (h = h || m.appendChild(b.createElement("div")), i = (Pb.exec(f) || ["", ""])[1].toLowerCase(), k = Xb[i] || Xb._default, h.innerHTML = k[1] + f.replace(Ob, "<$1></$2>") + k[2], e = k[0]; e--;)
                            h = h.lastChild;
                            if (!cb.leadingWhitespace && Nb.test(f) && n.push(b.createTextNode(Nb.exec(f)[0])), !cb.tbody)
                                for (f = "table" !== i || Qb.test(f) ? "<table>" !== k[1] || Qb.test(f) ? 0 : h : h.firstChild, e = f && f.childNodes.length; e--;)
                                    eb.nodeName(j = f.childNodes[e], "tbody")&&!j.childNodes.length && f.removeChild(j);
                                    for (eb.merge(n, h.childNodes), h.textContent = ""; h.firstChild;)
                                        h.removeChild(h.firstChild);
                                        h = m.lastChild
                    } else 
                        n.push(b.createTextNode(f));
            for (h && m.removeChild(h), cb.appendChecked || eb.grep(q(n, "input"), r), o = 0; f = n[o++];)
                if ((!d||-1 === eb.inArray(f, d)) && (g = eb.contains(f.ownerDocument, f), h = q(m.appendChild(f), "script"), g && v(h), c))
                    for (e = 0; f = h[e++];)
                        Ub.test(f.type || "") && c.push(f);
            return h = null, m
        },
        cleanData: function(a, b) {
            for (var c, d, e, f, g = 0, h = eb.expando, i = eb.cache, j = cb.deleteExpando, k = eb.event.special; null != (c = a[g]); g++)
                if ((b || eb.acceptData(c)) && (e = c[h], f = e && i[e])) {
                    if (f.events)
                        for (d in f.events)
                            k[d] ? eb.event.remove(c, d) : eb.removeEvent(c, d, f.handle);
                            i[e] && (delete i[e], j ? delete c[h] : typeof c.removeAttribute !== xb ? c.removeAttribute(h) : c[h] = null, W.push(e))
                }
        }
    }), eb.fn.extend({
        text: function(a) {
            return Db(this, function(a) {
                return void 0 === a ? eb.text(this) : this.empty().append((this[0] && this[0].ownerDocument || ob).createTextNode(a))
            }, null, a, arguments.length)
        },
        append: function() {
            return this.domManip(arguments, function(a) {
                if (1 === this.nodeType || 11 === this.nodeType || 9 === this.nodeType) {
                    var b = s(this, a);
                    b.appendChild(a)
                }
            })
        },
        prepend: function() {
            return this.domManip(arguments, function(a) {
                if (1 === this.nodeType || 11 === this.nodeType || 9 === this.nodeType) {
                    var b = s(this, a);
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
            for (var c, d = a ? eb.filter(a, this) : this, e = 0; null != (c = d[e]); e++)
                b || 1 !== c.nodeType || eb.cleanData(q(c)), c.parentNode && (b && eb.contains(c.ownerDocument, c) && v(q(c, "script")), c.parentNode.removeChild(c));
            return this
        },
        empty: function() {
            for (var a, b = 0; null != (a = this[b]); b++) {
                for (1 === a.nodeType && eb.cleanData(q(a, !1)); a.firstChild;)
                    a.removeChild(a.firstChild);
                a.options && eb.nodeName(a, "select") && (a.options.length = 0)
            }
            return this
        },
        clone: function(a, b) {
            return a = null == a?!1 : a, b = null == b ? a : b, this.map(function() {
                return eb.clone(this, a, b)
            })
        },
        html: function(a) {
            return Db(this, function(a) {
                var b = this[0] || {}, c = 0, d = this.length;
                if (void 0 === a)
                    return 1 === b.nodeType ? b.innerHTML.replace(Lb, "") : void 0;
                if ("string" === typeof a&&!Sb.test(a) && (cb.htmlSerialize ||!Mb.test(a)) && (cb.leadingWhitespace ||!Nb.test(a))&&!Xb[(Pb.exec(a) || ["", ""])[1].toLowerCase()]) {
                    a = a.replace(Ob, "<$1></$2>");
                    try {
                        for (; c < d; c++)
                            b = this[c] || {}, 1 === b.nodeType && (eb.cleanData(q(b, !1)), b.innerHTML = a);
                        b = 0
                    } catch (e) {}
                }
                b && this.empty().append(a)
            }, null, a, arguments.length)
        },
        replaceWith: function() {
            var a = arguments[0];
            return this.domManip(arguments, function(b) {
                a = this.parentNode, eb.cleanData(q(this)), a && a.replaceChild(b, this)
            }), a && (a.length || a.nodeType) ? this : this.remove()
        },
        detach: function(a) {
            return this.remove(a, !0)
        },
        domManip: function(a, b) {
            a = Y.apply([], a);
            var c, d, e, f, g, h, i = 0, j = this.length, k = this, l = j - 1, m = a[0], n = eb.isFunction(m);
            if (n || j > 1 && "string" === typeof m&&!cb.checkClone && Tb.test(m))
                return this.each(function(c) {
                    var d = k.eq(c);
                    n && (a[0] = m.call(this, c, d.html())), d.domManip(a, b)
                });
            if (j && (h = eb.buildFragment(a, this[0].ownerDocument, !1, this), c = h.firstChild, 1 === h.childNodes.length && (h = c), c)) {
                for (f = eb.map(q(h, "script"), t), e = f.length; i < j; i++)
                    d = h, i !== l && (d = eb.clone(d, !0, !0), e && eb.merge(f, q(d, "script"))), b.call(this[i], d, i);
                if (e)
                    for (g = f[f.length - 1].ownerDocument, eb.map(f, u), i = 0; i < e; i++)
                        d = f[i], Ub.test(d.type || "")&&!eb._data(d, "globalEval") && eb.contains(g, d) && (d.src ? eb._evalUrl && eb._evalUrl(d.src) : eb.globalEval((d.text || d.textContent || d.innerHTML || "").replace(Wb, "")));
                h = c = null
            }
            return this
        }
    }), eb.each({
        appendTo: "append",
        prependTo: "prepend",
        insertBefore: "before",
        insertAfter: "after",
        replaceAll: "replaceWith"
    }, function(a, b) {
        eb.fn[a] = function(a) {
            for (var c, d = 0, e = [], f = eb(a), g = f.length - 1; d <= g; d++)
                c = d === g ? this : this.clone(!0), eb(f[d])[b](c), Z.apply(e, c.get());
            return this.pushStack(e)
        }
    });
    var $b, _b = {};
    !function() {
        var a;
        cb.shrinkWrapBlocks = function() {
            if (null != a)
                return a;
            a=!1;
            var b, c, d;
            return c = ob.getElementsByTagName("body")[0], c && c.style ? (b = ob.createElement("div"), d = ob.createElement("div"), d.style.cssText = "position:absolute;border:0;width:0;height:0;top:0;left:-9999px", c.appendChild(d).appendChild(b), typeof b.style.zoom !== xb && (b.style.cssText = "-webkit-box-sizing:content-box;-moz-box-sizing:content-box;box-sizing:content-box;display:block;margin:0;border:0;padding:1px;width:1px;zoom:1", b.appendChild(ob.createElement("div")).style.width = "5px", a = 3 !== b.offsetWidth), c.removeChild(d), a) : void 0
        }
    }();
    var ac, bc, cc = /^margin/, dc = new RegExp("^(" + Ab + ")(?!px)[a-z%]+$", "i"), ec = /^(top|right|bottom|left)$/;
    a.getComputedStyle ? (ac = function(a) {
        return a.ownerDocument.defaultView.getComputedStyle(a, null)
    }, bc = function(a, b, c) {
        var d, e, f, g, h = a.style;
        return c = c || ac(a), g = c ? c.getPropertyValue(b) || c[b] : void 0, c && ("" !== g || eb.contains(a.ownerDocument, a) || (g = eb.style(a, b)), dc.test(g) && cc.test(b) && (d = h.width, e = h.minWidth, f = h.maxWidth, h.minWidth = h.maxWidth = h.width = g, g = c.width, h.width = d, h.minWidth = e, h.maxWidth = f)), void 0 === g ? g : g + ""
    }) : ob.documentElement.currentStyle && (ac = function(a) {
        return a.currentStyle
    }, bc = function(a, b, c) {
        var d, e, f, g, h = a.style;
        return c = c || ac(a), g = c ? c[b] : void 0, null == g && h && h[b] && (g = h[b]), dc.test(g)&&!ec.test(b) && (d = h.left, e = a.runtimeStyle, f = e && e.left, f && (e.left = a.currentStyle.left), h.left = "fontSize" === b ? "1em" : g, g = h.pixelLeft + "px", h.left = d, f && (e.left = f)), void 0 === g ? g : g + "" || "auto"
    }), function() {
        function b() {
            var b, c, d, e;
            c = ob.getElementsByTagName("body")[0], c && c.style && (b = ob.createElement("div"), d = ob.createElement("div"), d.style.cssText = "position:absolute;border:0;width:0;height:0;top:0;left:-9999px", c.appendChild(d).appendChild(b), b.style.cssText = "-webkit-box-sizing:border-box;-moz-box-sizing:border-box;box-sizing:border-box;display:block;margin-top:1%;top:1%;border:1px;padding:1px;width:4px;position:absolute", f = g=!1, i=!0, a.getComputedStyle && (f = "1%" !== (a.getComputedStyle(b, null) || {}).top, g = "4px" === (a.getComputedStyle(b, null) || {
                width: "4px"
            }).width, e = b.appendChild(ob.createElement("div")), e.style.cssText = b.style.cssText = "-webkit-box-sizing:content-box;-moz-box-sizing:content-box;box-sizing:content-box;display:block;margin:0;border:0;padding:0", e.style.marginRight = e.style.width = "0", b.style.width = "1px", i=!parseFloat((a.getComputedStyle(e, null) || {}).marginRight)), b.innerHTML = "<table><tr><td></td><td>t</td></tr></table>", e = b.getElementsByTagName("td"), e[0].style.cssText = "margin:0;border:0;padding:0;display:none", h = 0 === e[0].offsetHeight, h && (e[0].style.display = "", e[1].style.display = "none", h = 0 === e[0].offsetHeight), c.removeChild(d))
        }
        var c, d, e, f, g, h, i;
        c = ob.createElement("div"), c.innerHTML = "  <link/><table></table><a href='/a'>a</a><input type='checkbox'/>", e = c.getElementsByTagName("a")[0], d = e && e.style, d && (d.cssText = "float:left;opacity:.5", cb.opacity = "0.5" === d.opacity, cb.cssFloat=!!d.cssFloat, c.style.backgroundClip = "content-box", c.cloneNode(!0).style.backgroundClip = "", cb.clearCloneStyle = "content-box" === c.style.backgroundClip, cb.boxSizing = "" === d.boxSizing || "" === d.MozBoxSizing || "" === d.WebkitBoxSizing, eb.extend(cb, {
            reliableHiddenOffsets: function() {
                return null == h && b(), h
            },
            boxSizingReliable: function() {
                return null == g && b(), g
            },
            pixelPosition: function() {
                return null == f && b(), f
            },
            reliableMarginRight: function() {
                return null == i && b(), i
            }
        }))
    }(), eb.swap = function(a, b, c, d) {
        var e, f, g = {};
        for (f in b)
            g[f] = a.style[f], a.style[f] = b[f];
        e = c.apply(a, d || []);
        for (f in b)
            a.style[f] = g[f];
        return e
    };
    var fc = /alpha\([^)]*\)/i, gc = /opacity\s*=\s*([^)]*)/, hc = /^(none|table(?!-c[ea]).+)/, ic = new RegExp("^(" + Ab + ")(.*)$", "i"), jc = new RegExp("^([+-])=(" + Ab + ")", "i"), kc = {
        position: "absolute",
        visibility: "hidden",
        display: "block"
    }, lc = {
        letterSpacing: "0",
        fontWeight: "400"
    }, mc = ["Webkit", "O", "Moz", "ms"];
    eb.extend({
        cssHooks: {
            opacity: {
                get: function(a, b) {
                    if (b) {
                        var c = bc(a, "opacity");
                        return "" === c ? "1" : c
                    }
                }
            }
        },
        cssNumber: {
            columnCount: !0,
            fillOpacity: !0,
            flexGrow: !0,
            flexShrink: !0,
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
            "float": cb.cssFloat ? "cssFloat": "styleFloat"
        },
        style: function(a, b, c, d) {
            if (a && 3 !== a.nodeType && 8 !== a.nodeType && a.style) {
                var e, f, g, h = eb.camelCase(b), i = a.style;
                if (b = eb.cssProps[h] || (eb.cssProps[h] = B(i, h)), g = eb.cssHooks[b] || eb.cssHooks[h], void 0 === c)
                    return g && "get"in g && void 0 !== (e = g.get(a, !1, d)) ? e : i[b];
                if (f = typeof c, "string" === f && (e = jc.exec(c)) && (c = (e[1] + 1) * e[2] + parseFloat(eb.css(a, b)), f = "number"), null != c && c === c && ("number" !== f || eb.cssNumber[h] || (c += "px"), cb.clearCloneStyle || "" !== c || 0 !== b.indexOf("background") || (i[b] = "inherit"), !g ||!("set"in g) || void 0 !== (c = g.set(a, c, d))
                    ))try {
                    i[b] = c
                } catch (j) {}
            }
        },
        css: function(a, b, c, d) {
            var e, f, g, h = eb.camelCase(b);
            return b = eb.cssProps[h] || (eb.cssProps[h] = B(a.style, h)), g = eb.cssHooks[b] || eb.cssHooks[h], g && "get"in g && (f = g.get(a, !0, c)), void 0 === f && (f = bc(a, b, d)), "normal" === f && b in lc && (f = lc[b]), "" === c || c ? (e = parseFloat(f), c===!0 || eb.isNumeric(e) ? e || 0 : f) : f
        }
    }), eb.each(["height", "width"], function(a, b) {
        eb.cssHooks[b] = {
            get: function(a, c, d) {
                return c ? hc.test(eb.css(a, "display")) && 0 === a.offsetWidth ? eb.swap(a, kc, function() {
                    return F(a, b, d)
                }) : F(a, b, d) : void 0
            },
            set: function(a, c, d) {
                var e = d && ac(a);
                return D(a, c, d ? E(a, b, d, cb.boxSizing && "border-box" === eb.css(a, "boxSizing", !1, e), e) : 0)
            }
        }
    }), cb.opacity || (eb.cssHooks.opacity = {
        get: function(a, b) {
            return gc.test((b && a.currentStyle ? a.currentStyle.filter : a.style.filter) || "") ? .01 * parseFloat(RegExp.$1) + "" : b ? "1" : ""
        },
        set: function(a, b) {
            var c = a.style, d = a.currentStyle, e = eb.isNumeric(b) ? "alpha(opacity=" + 100 * b + ")": "", f = d && d.filter || c.filter || "";
            c.zoom = 1, (b >= 1 || "" === b) && "" === eb.trim(f.replace(fc, "")) && c.removeAttribute && (c.removeAttribute("filter"), "" === b || d&&!d.filter) || (c.filter = fc.test(f) ? f.replace(fc, e) : f + " " + e)
        }
    }), eb.cssHooks.marginRight = A(cb.reliableMarginRight, function(a, b) {
        return b ? eb.swap(a, {
            display: "inline-block"
        }, bc, [a, "marginRight"]) : void 0
    }), eb.each({
        margin: "",
        padding: "",
        border: "Width"
    }, function(a, b) {
        eb.cssHooks[a + b] = {
            expand: function(c) {
                for (var d = 0, e = {}, f = "string" === typeof c ? c.split(" ") : [c]; d < 4; d++)
                    e[a + Bb[d] + b] = f[d] || f[d - 2] || f[0];
                return e
            }
        }, cc.test(a) || (eb.cssHooks[a + b].set = D)
    }), eb.fn.extend({
        css: function(a, b) {
            return Db(this, function(a, b, c) {
                var d, e, f = {}, g = 0;
                if (eb.isArray(b)) {
                    for (d = ac(a), e = b.length; g < e; g++)
                        f[b[g]] = eb.css(a, b[g], !1, d);
                    return f
                }
                return void 0 !== c ? eb.style(a, b, c) : eb.css(a, b)
            }, a, b, arguments.length > 1)
        },
        show: function() {
            return C(this, !0)
        },
        hide: function() {
            return C(this)
        },
        toggle: function(a) {
            return "boolean" === typeof a ? a ? this.show() : this.hide() : this.each(function() {
                Cb(this) ? eb(this).show() : eb(this).hide()
            })
        }
    }), eb.Tween = G, G.prototype = {
        constructor: G,
        init: function(a, b, c, d, e, f) {
            this.elem = a, this.prop = c, this.easing = e || "swing", this.options = b, this.start = this.now = this.cur(), this.end = d, this.unit = f || (eb.cssNumber[c] ? "" : "px")
        },
        cur: function() {
            var a = G.propHooks[this.prop];
            return a && a.get ? a.get(this) : G.propHooks._default.get(this)
        },
        run: function(a) {
            var b, c = G.propHooks[this.prop];
            return this.pos = b = this.options.duration ? eb.easing[this.easing](a, this.options.duration * a, 0, 1, this.options.duration) : a, this.now = (this.end - this.start) * b + this.start, this.options.step && this.options.step.call(this.elem, this.now, this), c && c.set ? c.set(this) : G.propHooks._default.set(this), this
        }
    }, G.prototype.init.prototype = G.prototype, G.propHooks = {
        _default: {
            get: function(a) {
                var b;
                return null == a.elem[a.prop] || a.elem.style && null != a.elem.style[a.prop] ? (b = eb.css(a.elem, a.prop, ""), b && "auto" !== b ? b : 0) : a.elem[a.prop]
            },
            set: function(a) {
                eb.fx.step[a.prop] ? eb.fx.step[a.prop](a) : a.elem.style && (null != a.elem.style[eb.cssProps[a.prop]] || eb.cssHooks[a.prop]) ? eb.style(a.elem, a.prop, a.now + a.unit) : a.elem[a.prop] = a.now
            }
        }
    }, G.propHooks.scrollTop = G.propHooks.scrollLeft = {
        set: function(a) {
            a.elem.nodeType && a.elem.parentNode && (a.elem[a.prop] = a.now)
        }
    }, eb.easing = {
        linear: function(a) {
            return a
        },
        swing: function(a) {
            return .5 - Math.cos(a * Math.PI) / 2
        }
    }, eb.fx = G.prototype.init, eb.fx.step = {};
    var nc, oc, pc = /^(?:toggle|show|hide)$/, qc = new RegExp("^(?:([+-])=|)(" + Ab + ")([a-z%]*)$", "i"), rc = /queueHooks$/, sc = [K], tc = {
        "*": [function(a, b) {
            var c = this.createTween(a, b), d = c.cur(), e = qc.exec(b), f = e && e[3] || (eb.cssNumber[a] ? "" : "px"), g = (eb.cssNumber[a] || "px" !== f&&+d) && qc.exec(eb.css(c.elem, a)), h = 1, i = 20;
            if (g && g[3] !== f) {
                f = f || g[3], e = e || [], g =+ d || 1;
                do 
                    h = h || ".5", g/=h, eb.style(c.elem, a, g + f);
                while (h !== (h = c.cur() / d) && 1 !== h&&--i)
                }
            return e && (g = c.start =+ g||+d || 0, c.unit = f, c.end = e[1] ? g + (e[1] + 1) * e[2] : + e[2]), c
        }
        ]
    };
    eb.Animation = eb.extend(M, {
        tweener: function(a, b) {
            eb.isFunction(a) ? (b = a, a = ["*"]): a = a.split(" ");
            for (var c,
            d = 0,
            e = a.length;
            d < e;
            d++)c = a[d],
            tc[c] = tc[c] || [],
            tc[c].unshift(b)
        }, prefilter : function(a, b) {
            b ? sc.unshift(a) : sc.push(a)
        }
    }), eb.speed = function(a, b, c) {
        var d = a && "object" === typeof a ? eb.extend({}, a): {
            complete: c ||!c && b || eb.isFunction(a) && a,
            duration: a,
            easing: c && b || b&&!eb.isFunction(b) && b
        };
        return d.duration = eb.fx.off ? 0 : "number" === typeof d.duration ? d.duration : d.duration in eb.fx.speeds ? eb.fx.speeds[d.duration] : eb.fx.speeds._default, (null == d.queue || d.queue===!0) && (d.queue = "fx"), d.old = d.complete, d.complete = function() {
            eb.isFunction(d.old) && d.old.call(this), d.queue && eb.dequeue(this, d.queue)
        }, d
    }, eb.fn.extend({
        fadeTo: function(a, b, c, d) {
            return this.filter(Cb).css("opacity", 0).show().end().animate({
                opacity: b
            }, a, c, d)
        },
        animate: function(a, b, c, d) {
            var e = eb.isEmptyObject(a), f = eb.speed(b, c, d), g = function() {
                var b = M(this, eb.extend({}, a), f);
                (e || eb._data(this, "finish")) && b.stop(!0)
            };
            return g.finish = g, e || f.queue===!1 ? this.each(g) : this.queue(f.queue, g)
        },
        stop: function(a, b, c) {
            var d = function(a) {
                var b = a.stop;
                delete a.stop, b(c)
            };
            return "string" !== typeof a && (c = b, b = a, a = void 0), b && a!==!1 && this.queue(a || "fx", []), this.each(function() {
                var b=!0, e = null != a && a + "queueHooks", f = eb.timers, g = eb._data(this);
                if (e)
                    g[e] && g[e].stop && d(g[e]);
                else 
                    for (e in g)
                        g[e] && g[e].stop && rc.test(e) && d(g[e]);
                for (e = f.length; e--;)
                    f[e].elem !== this || null != a && f[e].queue !== a || (f[e].anim.stop(c), b=!1, f.splice(e, 1));
                (b ||!c) && eb.dequeue(this, a)
            })
        },
        finish: function(a) {
            return a!==!1 && (a = a || "fx"), this.each(function() {
                var b, c = eb._data(this), d = c[a + "queue"], e = c[a + "queueHooks"], f = eb.timers, g = d ? d.length: 0;
                for (c.finish=!0, eb.queue(this, a, []), e && e.stop && e.stop.call(this, !0), b = f.length; b--;)
                    f[b].elem === this && f[b].queue === a && (f[b].anim.stop(!0), f.splice(b, 1));
                for (b = 0; b < g; b++)
                    d[b] && d[b].finish && d[b].finish.call(this);
                delete c.finish
            })
        }
    }), eb.each(["toggle", "show", "hide"], function(a, b) {
        var c = eb.fn[b];
        eb.fn[b] = function(a, d, e) {
            return null == a || "boolean" === typeof a ? c.apply(this, arguments) : this.animate(I(b, !0), a, d, e)
        }
    }), eb.each({
        slideDown: I("show"),
        slideUp: I("hide"),
        slideToggle: I("toggle"),
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
        eb.fn[a] = function(a, c, d) {
            return this.animate(b, a, c, d)
        }
    }), eb.timers = [], eb.fx.tick = function() {
        var a, b = eb.timers, c = 0;
        for (nc = eb.now(); c < b.length; c++)
            a = b[c], a() || b[c] !== a || b.splice(c--, 1);
        b.length || eb.fx.stop(), nc = void 0
    }, eb.fx.timer = function(a) {
        eb.timers.push(a), a() ? eb.fx.start() : eb.timers.pop()
    }, eb.fx.interval = 13, eb.fx.start = function() {
        oc || (oc = setInterval(eb.fx.tick, eb.fx.interval))
    }, eb.fx.stop = function() {
        clearInterval(oc), oc = null
    }, eb.fx.speeds = {
        slow: 600,
        fast: 200,
        _default: 400
    }, eb.fn.delay = function(a, b) {
        return a = eb.fx ? eb.fx.speeds[a] || a : a, b = b || "fx", this.queue(b, function(b, c) {
            var d = setTimeout(b, a);
            c.stop = function() {
                clearTimeout(d)
            }
        })
    }, function() {
        var a, b, c, d, e;
        b = ob.createElement("div"), b.setAttribute("className", "t"), b.innerHTML = "  <link/><table></table><a href='/a'>a</a><input type='checkbox'/>", d = b.getElementsByTagName("a")[0], c = ob.createElement("select"), e = c.appendChild(ob.createElement("option")), a = b.getElementsByTagName("input")[0], d.style.cssText = "top:1px", cb.getSetAttribute = "t" !== b.className, cb.style = /top/.test(d.getAttribute("style")), cb.hrefNormalized = "/a" === d.getAttribute("href"), cb.checkOn=!!a.value, cb.optSelected = e.selected, cb.enctype=!!ob.createElement("form").enctype, c.disabled=!0, cb.optDisabled=!e.disabled, a = ob.createElement("input"), a.setAttribute("value", ""), cb.input = "" === a.getAttribute("value"), a.value = "t", a.setAttribute("type", "radio"), cb.radioValue = "t" === a.value
    }();
    var uc = /\r/g;
    eb.fn.extend({
        val: function(a) {
            var b, c, d, e = this[0];
            {
                if (arguments.length)
                    return d = eb.isFunction(a), this.each(function(c) {
                        var e;
                        1 === this.nodeType && (e = d ? a.call(this, c, eb(this).val()) : a, null == e ? e = "" : "number" === typeof e ? e += "" : eb.isArray(e) && (e = eb.map(e, function(a) {
                            return null == a ? "" : a + ""
                        })), b = eb.valHooks[this.type] || eb.valHooks[this.nodeName.toLowerCase()], b && "set"in b && void 0 !== b.set(this, e, "value") || (this.value = e))
                    });
                if (e)
                    return b = eb.valHooks[e.type] || eb.valHooks[e.nodeName.toLowerCase()], b && "get"in b && void 0 !== (c = b.get(e, "value")) ? c : (c = e.value, "string" === typeof c ? c.replace(uc, "") : null == c ? "" : c)
            }
        }
    }), eb.extend({
        valHooks: {
            option: {
                get: function(a) {
                    var b = eb.find.attr(a, "value");
                    return null != b ? b : eb.trim(eb.text(a))
                }
            },
            select: {
                get: function(a) {
                    for (var b, c, d = a.options, e = a.selectedIndex, f = "select-one" === a.type || e < 0, g = f ? null : [], h = f ? e + 1 : d.length, i = e < 0 ? h : f ? e : 0; i < h; i++)
                        if (c = d[i], (c.selected || i === e) && (cb.optDisabled?!c.disabled : null === c.getAttribute("disabled")) && (!c.parentNode.disabled ||!eb.nodeName(c.parentNode, "optgroup"))
                            ) {
                        if (b = eb(c).val(), f)
                            return b;
                        g.push(b)
                    }
                    return g
                },
                set: function(a, b) {
                    for (var c, d, e = a.options, f = eb.makeArray(b), g = e.length; g--;)
                        if (d = e[g], eb.inArray(eb.valHooks.option.get(d), f) >= 0)
                            try {
                                d.selected = c=!0
                    } catch (h) {
                        d.scrollHeight
                    } else 
                        d.selected=!1;
                    return c || (a.selectedIndex =- 1), e
                }
            }
        }
    }), eb.each(["radio", "checkbox"], function() {
        eb.valHooks[this] = {
            set: function(a, b) {
                return eb.isArray(b) ? a.checked = eb.inArray(eb(a).val(), b) >= 0 : void 0
            }
        }, cb.checkOn || (eb.valHooks[this].get = function(a) {
            return null === a.getAttribute("value") ? "on" : a.value
        })
    });
    var vc, wc, xc = eb.expr.attrHandle, yc = /^(?:checked|selected)$/i, zc = cb.getSetAttribute, Ac = cb.input;
    eb.fn.extend({
        attr: function(a, b) {
            return Db(this, eb.attr, a, b, arguments.length > 1)
        },
        removeAttr: function(a) {
            return this.each(function() {
                eb.removeAttr(this, a)
            })
        }
    }), eb.extend({
        attr: function(a, b, c) {
            var d, e, f = a.nodeType;
            if (a && 3 !== f && 8 !== f && 2 !== f)
                return typeof a.getAttribute === xb ? eb.prop(a, b, c) : (1 === f && eb.isXMLDoc(a) || (b = b.toLowerCase(), d = eb.attrHooks[b] || (eb.expr.match.bool.test(b) ? wc : vc)), void 0 === c ? d && "get"in d && null !== (e = d.get(a, b)) ? e : (e = eb.find.attr(a, b), null == e ? void 0 : e) : null !== c ? d && "set"in d && void 0 !== (e = d.set(a, c, b)) ? e : (a.setAttribute(b, c + ""), c) : void eb.removeAttr(a, b))
        },
        removeAttr: function(a, b) {
            var c, d, e = 0, f = b && b.match(tb);
            if (f && 1 === a.nodeType)
                for (; c = f[e++];)
                    d = eb.propFix[c] || c, eb.expr.match.bool.test(c) ? Ac && zc ||!yc.test(c) ? a[d]=!1 : a[eb.camelCase("default-" + c)] = a[d]=!1 : eb.attr(a, c, ""), a.removeAttribute(zc ? c : d)
        },
        attrHooks: {
            type: {
                set: function(a, b) {
                    if (!cb.radioValue && "radio" === b && eb.nodeName(a, "input")) {
                        var c = a.value;
                        return a.setAttribute("type", b), c && (a.value = c), b
                    }
                }
            }
        }
    }), wc = {
        set: function(a, b, c) {
            return b===!1 ? eb.removeAttr(a, c) : Ac && zc ||!yc.test(c) ? a.setAttribute(!zc && eb.propFix[c] || c, c) : a[eb.camelCase("default-" + c)] = a[c]=!0, c
        }
    }, eb.each(eb.expr.match.bool.source.match(/\w+/g), function(a, b) {
        var c = xc[b] || eb.find.attr;
        xc[b] = Ac && zc ||!yc.test(b) ? function(a, b, d) {
            var e, f;
            return d || (f = xc[b], xc[b] = e, e = null != c(a, b, d) ? b.toLowerCase() : null, xc[b] = f), e
        } : function(a, b, c) {
            return c ? void 0 : a[eb.camelCase("default-" + b)] ? b.toLowerCase() : null
        }
    }), Ac && zc || (eb.attrHooks.value = {
        set: function(a, b, c) {
            return eb.nodeName(a, "input") ? void(a.defaultValue = b) : vc && vc.set(a, b, c)
        }
    }), zc || (vc = {
        set: function(a, b, c) {
            var d = a.getAttributeNode(c);
            return d || a.setAttributeNode(d = a.ownerDocument.createAttribute(c)), d.value = b += "", "value" === c || b === a.getAttribute(c) ? b : void 0
        }
    }, xc.id = xc.name = xc.coords = function(a, b, c) {
        var d;
        return c ? void 0 : (d = a.getAttributeNode(b)) && "" !== d.value ? d.value : null
    }, eb.valHooks.button = {
        get: function(a, b) {
            var c = a.getAttributeNode(b);
            return c && c.specified ? c.value : void 0
        },
        set: vc.set
    }, eb.attrHooks.contenteditable = {
        set: function(a, b, c) {
            vc.set(a, "" === b?!1 : b, c)
        }
    }, eb.each(["width", "height"], function(a, b) {
        eb.attrHooks[b] = {
            set: function(a, c) {
                return "" === c ? (a.setAttribute(b, "auto"), c) : void 0
            }
        }
    })), cb.style || (eb.attrHooks.style = {
        get: function(a) {
            return a.style.cssText || void 0
        },
        set: function(a, b) {
            return a.style.cssText = b + ""
        }
    });
    var Bc = /^(?:input|select|textarea|button|object)$/i, Cc = /^(?:a|area)$/i;
    eb.fn.extend({
        prop: function(a, b) {
            return Db(this, eb.prop, a, b, arguments.length > 1)
        },
        removeProp: function(a) {
            return a = eb.propFix[a] || a, this.each(function() {
                try {
                    this[a] = void 0, delete this[a]
                } catch (b) {}
            })
        }
    }), eb.extend({
        propFix: {
            "for": "htmlFor",
            "class": "className"
        },
        prop: function(a, b, c) {
            var d, e, f, g = a.nodeType;
            if (a && 3 !== g && 8 !== g && 2 !== g)
                return f = 1 !== g ||!eb.isXMLDoc(a), f && (b = eb.propFix[b] || b, e = eb.propHooks[b]), void 0 !== c ? e && "set"in e && void 0 !== (d = e.set(a, c, b)) ? d : a[b] = c : e && "get"in e && null !== (d = e.get(a, b)) ? d : a[b]
        },
        propHooks: {
            tabIndex: {
                get: function(a) {
                    var b = eb.find.attr(a, "tabindex");
                    return b ? parseInt(b, 10) : Bc.test(a.nodeName) || Cc.test(a.nodeName) && a.href ? 0 : - 1
                }
            }
        }
    }), cb.hrefNormalized || eb.each(["href", "src"], function(a, b) {
        eb.propHooks[b] = {
            get: function(a) {
                return a.getAttribute(b, 4)
            }
        }
    }), cb.optSelected || (eb.propHooks.selected = {
        get: function(a) {
            var b = a.parentNode;
            return b && (b.selectedIndex, b.parentNode && b.parentNode.selectedIndex), null
        }
    }), eb.each(["tabIndex", "readOnly", "maxLength", "cellSpacing", "cellPadding", "rowSpan", "colSpan", "useMap", "frameBorder", "contentEditable"], function() {
        eb.propFix[this.toLowerCase()] = this
    }), cb.enctype || (eb.propFix.enctype = "encoding");
    var Dc = /[\t\r\n\f]/g;
    eb.fn.extend({
        addClass: function(a) {
            var b, c, d, e, f, g, h = 0, i = this.length, j = "string" === typeof a && a;
            if (eb.isFunction(a))
                return this.each(function(b) {
                    eb(this).addClass(a.call(this, b, this.className))
                });
            if (j)
                for (b = (a || "").match(tb) || []; h < i; h++)
                    if (c = this[h], d = 1 === c.nodeType && (c.className ? (" " + c.className + " ").replace(Dc, " ") : " ")) {
                        for (f = 0; e = b[f++];)
                            d.indexOf(" " + e + " ") < 0 && (d += e + " ");
                            g = eb.trim(d), c.className !== g && (c.className = g)
                    }
            return this
        },
        removeClass: function(a) {
            var b, c, d, e, f, g, h = 0, i = this.length, j = 0 === arguments.length || "string" === typeof a && a;
            if (eb.isFunction(a))
                return this.each(function(b) {
                    eb(this).removeClass(a.call(this, b, this.className))
                });
            if (j)
                for (b = (a || "").match(tb) || []; h < i; h++)
                    if (c = this[h], d = 1 === c.nodeType && (c.className ? (" " + c.className + " ").replace(Dc, " ") : "")) {
                        for (f = 0; e = b[f++];)
                            for (; d.indexOf(" " + e + " ") >= 0;)
                                d = d.replace(" " + e + " ", " ");
                                g = a ? eb.trim(d) : "", c.className !== g && (c.className = g)
                    }
            return this
        },
        toggleClass: function(a, b) {
            var c = typeof a;
            return "boolean" === typeof b && "string" === c ? b ? this.addClass(a) : this.removeClass(a) : this.each(eb.isFunction(a) ? function(c) {
                eb(this).toggleClass(a.call(this, c, this.className, b), b)
            } : function() {
                if ("string" === c)
                    for (var b, d = 0, e = eb(this), f = a.match(tb) || []; b = f[d++];)
                        e.hasClass(b) ? e.removeClass(b) : e.addClass(b);
                else (c === xb || "boolean" === c) 
                    && (this.className && eb._data(this, "__className__", this.className), this.className = this.className || a===!1 ? "" : eb._data(this, "__className__") || "")
            })
        },
        hasClass: function(a) {
            for (var b = " " + a + " ", c = 0, d = this.length; c < d; c++)
                if (1 === this[c].nodeType && (" " + this[c].className + " ").replace(Dc, " ").indexOf(b) >= 0)
                    return !0;
            return !1
        }
    }), eb.each("blur focus focusin focusout load resize scroll unload click dblclick mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave change select submit keydown keypress keyup error contextmenu".split(" "), function(a, b) {
        eb.fn[b] = function(a, c) {
            return arguments.length > 0 ? this.on(b, null, a, c) : this.trigger(b)
        }
    }), eb.fn.extend({
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
    var Ec = eb.now(), Fc = /\?/, Gc = /(,)|(\[|{)|(}|])|"(?:[^"\\\r\n]|\\["\\\/bfnrt]|\\u[\da-fA-F]{4})*"\s*:?|true|false|null|-?(?!0\d)\d+(?:\.\d+|)(?:[eE][+-]?\d+|)/g;
    eb.parseJSON = function(b) {
        if (a.JSON && a.JSON.parse)
            return a.JSON.parse(b + "");
        var c, d = null, e = eb.trim(b + "");
        return e&&!eb.trim(e.replace(Gc, function(a, b, e, f) {
            return c && b && (d = 0), 0 === d ? a : (c = e || b, d+=!f-!e, "")
        })) ? Function("return " + e)() : eb.error("Invalid JSON: " + b)
    }, eb.parseXML = function(b) {
        var c, d;
        if (!b || "string" !== typeof b)
            return null;
        try {
            a.DOMParser ? (d = new DOMParser, c = d.parseFromString(b, "text/xml")) : (c = new ActiveXObject("Microsoft.XMLDOM"), c.async = "false", c.loadXML(b))
        } catch (e) {
            c = void 0
        }
        return c && c.documentElement&&!c.getElementsByTagName("parsererror").length || eb.error("Invalid XML: " + b), c
    };
    var Hc, Ic, Jc = /#.*$/, Kc = /([?&])_=[^&]*/, Lc = /^(.*?):[ \t]*([^\r\n]*)\r?$/gm, Mc = /^(?:about|app|app-storage|.+-extension|file|res|widget):$/, Nc = /^(?:GET|HEAD)$/, Oc = /^\/\//, Pc = /^([\w.+-]+:)(?:\/\/(?:[^\/?#]*@|)([^\/?#:]*)(?::(\d+)|)|)/, Qc = {}, Rc = {}, Sc = "*/".concat("*");
    try {
        Ic = location.href
    } catch (Tc) {
        Ic = ob.createElement("a"), Ic.href = "", Ic = Ic.href
    }
    Hc = Pc.exec(Ic.toLowerCase()) || [], eb.extend({
        active: 0,
        lastModified: {},
        etag: {},
        ajaxSettings: {
            url: Ic,
            type: "GET",
            isLocal: Mc.test(Hc[1]),
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
                "text json": eb.parseJSON,
                "text xml": eb.parseXML
            },
            flatOptions: {
                url: !0,
                context: !0
            }
        },
        ajaxSetup: function(a, b) {
            return b ? P(P(a, eb.ajaxSettings), b) : P(eb.ajaxSettings, a)
        },
        ajaxPrefilter: N(Qc),
        ajaxTransport: N(Rc),
        ajax: function(a, b) {
            function c(a, b, c, d) {
                var e, k, r, s, u, w = b;
                2 !== t && (t = 2, h && clearTimeout(h), j = void 0, g = d || "", v.readyState = a > 0 ? 4 : 0, e = a >= 200 && a < 300 || 304 === a, c && (s = Q(l, v, c)), s = R(l, s, v, e), e ? (l.ifModified && (u = v.getResponseHeader("Last-Modified"), u && (eb.lastModified[f] = u), u = v.getResponseHeader("etag"), u && (eb.etag[f] = u)), 204 === a || "HEAD" === l.type ? w = "nocontent" : 304 === a ? w = "notmodified" : (w = s.state, k = s.data, r = s.error, e=!r)) : (r = w, (a ||!w) && (w = "error", a < 0 && (a = 0))), v.status = a, v.statusText = (b || w) + "", e ? o.resolveWith(m, [k, w, v]) : o.rejectWith(m, [v, w, r]), v.statusCode(q), q = void 0, i && n.trigger(e ? "ajaxSuccess" : "ajaxError", [v, l, e ? k: r]), p.fireWith(m, [v, w]), i && (n.trigger("ajaxComplete", [v, l]), --eb.active || eb.event.trigger("ajaxStop")))
            }
            "object" === typeof a && (b = a, a = void 0), b = b || {};
            var d, e, f, g, h, i, j, k, l = eb.ajaxSetup({}, b), m = l.context || l, n = l.context && (m.nodeType || m.jquery) ? eb(m): eb.event, o = eb.Deferred(), p = eb.Callbacks("once memory"), q = l.statusCode || {}, r = {}, s = {}, t = 0, u = "canceled", v = {
                readyState: 0,
                getResponseHeader: function(a) {
                    var b;
                    if (2 === t) {
                        if (!k)
                            for (k = {}; b = Lc.exec(g);)
                                k[b[1].toLowerCase()] = b[2];
                        b = k[a.toLowerCase()]
                    }
                    return null == b ? null : b
                },
                getAllResponseHeaders: function() {
                    return 2 === t ? g : null
                },
                setRequestHeader: function(a, b) {
                    var c = a.toLowerCase();
                    return t || (a = s[c] = s[c] || a, r[a] = b), this
                },
                overrideMimeType: function(a) {
                    return t || (l.mimeType = a), this
                },
                statusCode: function(a) {
                    var b;
                    if (a)
                        if (t < 2)
                            for (b in a)
                                q[b] = [q[b], a[b]];
                        else 
                            v.always(a[v.status]);
                    return this
                },
                abort: function(a) {
                    var b = a || u;
                    return j && j.abort(b), c(0, b), this
                }
            };
            if (o.promise(v).complete = p.add, v.success = v.done, v.error = v.fail, l.url = ((a || l.url || Ic) + "").replace(Jc, "").replace(Oc, Hc[1] + "//"), l.type = b.method || b.type || l.method || l.type, l.dataTypes = eb.trim(l.dataType || "*").toLowerCase().match(tb) || [""], null == l.crossDomain && (d = Pc.exec(l.url.toLowerCase()), l.crossDomain=!(!d || d[1] === Hc[1] && d[2] === Hc[2] && (d[3] || ("http:" === d[1] ? "80" : "443")) === (Hc[3] || ("http:" === Hc[1] ? "80" : "443")))), l.data && l.processData && "string" !== typeof l.data && (l.data = eb.param(l.data, l.traditional)), O(Qc, l, b, v), 2 === t)
                return v;
            i = l.global, i && 0 === eb.active++&&eb.event.trigger("ajaxStart"), l.type = l.type.toUpperCase(), l.hasContent=!Nc.test(l.type), f = l.url, l.hasContent || (l.data && (f = l.url += (Fc.test(f) ? "&" : "?") + l.data, delete l.data), l.cache===!1 && (l.url = Kc.test(f) ? f.replace(Kc, "$1_=" + Ec++) : f + (Fc.test(f) ? "&" : "?") + "_=" + Ec++)), l.ifModified && (eb.lastModified[f] && v.setRequestHeader("If-Modified-Since", eb.lastModified[f]), eb.etag[f] && v.setRequestHeader("If-None-Match", eb.etag[f])), (l.data && l.hasContent && l.contentType!==!1 || b.contentType) && v.setRequestHeader("Content-Type", l.contentType), v.setRequestHeader("Accept", l.dataTypes[0] && l.accepts[l.dataTypes[0]] ? l.accepts[l.dataTypes[0]] + ("*" !== l.dataTypes[0] ? ", " + Sc + "; q=0.01" : "") : l.accepts["*"]);
            for (e in l.headers)
                v.setRequestHeader(e, l.headers[e]);
            if (l.beforeSend && (l.beforeSend.call(m, v, l)===!1 || 2 === t))
                return v.abort();
            u = "abort";
            for (e in{
                success: 1,
                error: 1,
                complete: 1
            })
                v[e](l[e]);
            if (j = O(Rc, l, b, v)) {
                v.readyState = 1, i && n.trigger("ajaxSend", [v, l]), l.async && l.timeout > 0 && (h = setTimeout(function() {
                    v.abort("timeout")
                }, l.timeout));
                try {
                    t = 1, j.send(r, c)
                } catch (w) {
                    if (!(t < 2))
                        throw w;
                    c( - 1, w)
                }
            } else 
                c( - 1, "No Transport");
            return v
        },
        getJSON: function(a, b, c) {
            return eb.get(a, b, c, "json")
        },
        getScript: function(a, b) {
            return eb.get(a, void 0, b, "script")
        }
    }), eb.each(["get", "post"], function(a, b) {
        eb[b] = function(a, c, d, e) {
            return eb.isFunction(c) && (e = e || d, d = c, c = void 0), eb.ajax({
                url: a,
                type: b,
                dataType: e,
                data: c,
                success: d
            })
        }
    }), eb.each(["ajaxStart", "ajaxStop", "ajaxComplete", "ajaxError", "ajaxSuccess", "ajaxSend"], function(a, b) {
        eb.fn[b] = function(a) {
            return this.on(b, a)
        }
    }), eb._evalUrl = function(a) {
        return eb.ajax({
            url: a,
            type: "GET",
            dataType: "script",
            async: !1,
            global: !1,
            "throws": !0
        })
    }, eb.fn.extend({
        wrapAll: function(a) {
            if (eb.isFunction(a))
                return this.each(function(b) {
                    eb(this).wrapAll(a.call(this, b))
                });
            if (this[0]) {
                var b = eb(a, this[0].ownerDocument).eq(0).clone(!0);
                this[0].parentNode && b.insertBefore(this[0]), b.map(function() {
                    for (var a = this; a.firstChild && 1 === a.firstChild.nodeType;)
                        a = a.firstChild;
                    return a
                }).append(this)
            }
            return this
        },
        wrapInner: function(a) {
            return this.each(eb.isFunction(a) ? function(b) {
                eb(this).wrapInner(a.call(this, b))
            } : function() {
                var b = eb(this), c = b.contents();
                c.length ? c.wrapAll(a) : b.append(a)
            })
        },
        wrap: function(a) {
            var b = eb.isFunction(a);
            return this.each(function(c) {
                eb(this).wrapAll(b ? a.call(this, c) : a)
            })
        },
        unwrap: function() {
            return this.parent().each(function() {
                eb.nodeName(this, "body") || eb(this).replaceWith(this.childNodes)
            }).end()
        }
    }), eb.expr.filters.hidden = function(a) {
        return a.offsetWidth <= 0 && a.offsetHeight <= 0 ||!cb.reliableHiddenOffsets() && "none" === (a.style && a.style.display || eb.css(a, "display"))
    }, eb.expr.filters.visible = function(a) {
        return !eb.expr.filters.hidden(a)
    };
    var Uc = /%20/g, Vc = /\[\]$/, Wc = /\r?\n/g, Xc = /^(?:submit|button|image|reset|file)$/i, Yc = /^(?:input|select|textarea|keygen)/i;
    eb.param = function(a, b) {
        var c, d = [], e = function(a, b) {
            b = eb.isFunction(b) ? b() : null == b ? "" : b, d[d.length] = encodeURIComponent(a) + "=" + encodeURIComponent(b)
        };
        if (void 0 === b && (b = eb.ajaxSettings && eb.ajaxSettings.traditional), eb.isArray(a) || a.jquery&&!eb.isPlainObject(a)
            )eb.each(a, function() {
            e(this.name, this.value)
        });
        else 
            for (c in a)
                S(c, a[c], b, e);
        return d.join("&").replace(Uc, "+")
    }, eb.fn.extend({
        serialize: function() {
            return eb.param(this.serializeArray())
        },
        serializeArray: function() {
            return this.map(function() {
                var a = eb.prop(this, "elements");
                return a ? eb.makeArray(a) : this
            }).filter(function() {
                var a = this.type;
                return this.name&&!eb(this).is(":disabled") && Yc.test(this.nodeName)&&!Xc.test(a) && (this.checked ||!Eb.test(a))
            }).map(function(a, b) {
                var c = eb(this).val();
                return null == c ? null : eb.isArray(c) ? eb.map(c, function(a) {
                    return {
                        name: b.name,
                        value: a.replace(Wc, "\r\n")
                    }
                }) : {
                    name: b.name,
                    value: c.replace(Wc, "\r\n")
                }
            }).get()
        }
    }), eb.ajaxSettings.xhr = void 0 !== a.ActiveXObject ? function() {
        return !this.isLocal && /^(get|post|head|put|delete|options)$/i.test(this.type) && T() || U()
    } : T;
    var Zc = 0, $c = {}, _c = eb.ajaxSettings.xhr();
    a.ActiveXObject && eb(a).on("unload", function() {
        for (var a in $c)
            $c[a](void 0, !0)
    }), cb.cors=!!_c && "withCredentials"in _c, _c = cb.ajax=!!_c, _c && eb.ajaxTransport(function(a) {
        if (!a.crossDomain || cb.cors) {
            var b;
            return {
                send: function(c, d) {
                    var e, f = a.xhr(), g=++Zc;
                    if (f.open(a.type, a.url, a.async, a.username, a.password), a.xhrFields)
                        for (e in a.xhrFields)
                            f[e] = a.xhrFields[e];
                    a.mimeType && f.overrideMimeType && f.overrideMimeType(a.mimeType), a.crossDomain || c["X-Requested-With"] || (c["X-Requested-With"] = "XMLHttpRequest");
                    for (e in c)
                        void 0 !== c[e] && f.setRequestHeader(e, c[e] + "");
                    f.send(a.hasContent && a.data || null), b = function(c, e) {
                        var h, i, j;
                        if (b && (e || 4 === f.readyState))
                            if (delete $c[g], b = void 0, f.onreadystatechange = eb.noop, e)
                                4 !== f.readyState && f.abort();
                            else {
                                j = {}, h = f.status, "string" === typeof f.responseText && (j.text = f.responseText);
                                try {
                                    i = f.statusText
                                } catch (k) {
                                    i = ""
                                }
                                h ||!a.isLocal || a.crossDomain ? 1223 === h && (h = 204) : h = j.text ? 200 : 404
                            }
                        j && d(h, i, j, f.getAllResponseHeaders())
                    }, a.async ? 4 === f.readyState ? setTimeout(b) : f.onreadystatechange = $c[g] = b : b()
                },
                abort: function() {
                    b && b(void 0, !0)
                }
            }
        }
    }), eb.ajaxSetup({
        accepts: {
            script: "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript"
        },
        contents: {
            script: /(?:java|ecma)script/
        },
        converters: {
            "text script": function(a) {
                return eb.globalEval(a), a
            }
        }
    }), eb.ajaxPrefilter("script", function(a) {
        void 0 === a.cache && (a.cache=!1), a.crossDomain && (a.type = "GET", a.global=!1)
    }), eb.ajaxTransport("script", function(a) {
        if (a.crossDomain) {
            var b, c = ob.head || eb("head")[0] || ob.documentElement;
            return {
                send: function(d, e) {
                    b = ob.createElement("script"), b.async=!0, a.scriptCharset && (b.charset = a.scriptCharset), b.src = a.url, b.onload = b.onreadystatechange = function(a, c) {
                        (c ||!b.readyState || /loaded|complete/.test(b.readyState)) && (b.onload = b.onreadystatechange = null, b.parentNode && b.parentNode.removeChild(b), b = null, c || e(200, "success"))
                    }, c.insertBefore(b, c.firstChild)
                },
                abort: function() {
                    b && b.onload(void 0, !0)
                }
            }
        }
    });
    var ad = [], bd = /(=)\?(?=&|$)|\?\?/;
    eb.ajaxSetup({
        jsonp: "callback",
        jsonpCallback: function() {
            var a = ad.pop() || eb.expando + "_" + Ec++;
            return this[a]=!0, a
        }
    }), eb.ajaxPrefilter("json jsonp", function(b, c, d) {
        var e, f, g, h = b.jsonp!==!1 && (bd.test(b.url) ? "url" : "string" === typeof b.data&&!(b.contentType || "").indexOf("application/x-www-form-urlencoded") && bd.test(b.data) && "data");
        return h || "jsonp" === b.dataTypes[0] ? (e = b.jsonpCallback = eb.isFunction(b.jsonpCallback) ? b.jsonpCallback() : b.jsonpCallback, h ? b[h] = b[h].replace(bd, "$1" + e) : b.jsonp!==!1 && (b.url += (Fc.test(b.url) ? "&" : "?") + b.jsonp + "=" + e), b.converters["script json"] = function() {
            return g || eb.error(e + " was not called"), g[0]
        }, b.dataTypes[0] = "json", f = a[e], a[e] = function() {
            g = arguments
        }, d.always(function() {
            a[e] = f, b[e] && (b.jsonpCallback = c.jsonpCallback, ad.push(e)), g && eb.isFunction(f) && f(g[0]), g = f = void 0
        }), "script") : void 0
    }), eb.parseHTML = function(a, b, c) {
        if (!a || "string" !== typeof a)
            return null;
        "boolean" === typeof b && (c = b, b=!1), b = b || ob;
        var d = lb.exec(a), e=!c && [];
        return d ? [b.createElement(d[1])] : (d = eb.buildFragment([a], b, e), e && e.length && eb(e).remove(), eb.merge([], d.childNodes))
    };
    var cd = eb.fn.load;
    eb.fn.load = function(a, b, c) {
        if ("string" !== typeof a && cd)
            return cd.apply(this, arguments);
        var d, e, f, g = this, h = a.indexOf(" ");
        return h >= 0 && (d = eb.trim(a.slice(h, a.length)), a = a.slice(0, h)), eb.isFunction(b) ? (c = b, b = void 0) : b && "object" === typeof b && (f = "POST"), g.length > 0 && eb.ajax({
            url: a,
            type: f,
            dataType: "html",
            data: b
        }).done(function(a) {
            e = arguments, g.html(d ? eb("<div>").append(eb.parseHTML(a)).find(d) : a)
        }).complete(c && function(a, b) {
            g.each(c, e || [a.responseText, b, a])
        }), this
    }, eb.expr.filters.animated = function(a) {
        return eb.grep(eb.timers, function(b) {
            return a === b.elem
        }).length
    };
    var dd = a.document.documentElement;
    eb.offset = {
        setOffset: function(a, b, c) {
            var d, e, f, g, h, i, j, k = eb.css(a, "position"), l = eb(a), m = {};
            "static" === k && (a.style.position = "relative"), h = l.offset(), f = eb.css(a, "top"), i = eb.css(a, "left"), j = ("absolute" === k || "fixed" === k) && eb.inArray("auto", [f, i])>-1, j ? (d = l.position(), g = d.top, e = d.left) : (g = parseFloat(f) || 0, e = parseFloat(i) || 0), eb.isFunction(b) && (b = b.call(a, c, h)), null != b.top && (m.top = b.top - h.top + g), null != b.left && (m.left = b.left - h.left + e), "using"in b ? b.using.call(a, m) : l.css(m)
        }
    }, eb.fn.extend({
        offset: function(a) {
            if (arguments.length)
                return void 0 === a ? this : this.each(function(b) {
                eb.offset.setOffset(this, a, b)
            });
            var b, c, d = {
                top: 0,
                left: 0
            }, e = this[0], f = e && e.ownerDocument;
            if (f)
                return b = f.documentElement, eb.contains(b, e) ? (typeof e.getBoundingClientRect !== xb && (d = e.getBoundingClientRect()), c = V(f), {
                    top: d.top + (c.pageYOffset || b.scrollTop) - (b.clientTop || 0),
                    left: d.left + (c.pageXOffset || b.scrollLeft) - (b.clientLeft || 0)
                }) : d
        },
        position: function() {
            if (this[0]) {
                var a, b, c = {
                    top: 0,
                    left: 0
                }, d = this[0];
                return "fixed" === eb.css(d, "position") ? b = d.getBoundingClientRect() : (a = this.offsetParent(), b = this.offset(), eb.nodeName(a[0], "html") || (c = a.offset()), c.top += eb.css(a[0], "borderTopWidth", !0), c.left += eb.css(a[0], "borderLeftWidth", !0)), {
                    top: b.top - c.top - eb.css(d, "marginTop", !0),
                    left: b.left - c.left - eb.css(d, "marginLeft", !0)
                }
            }
        },
        offsetParent: function() {
            return this.map(function() {
                for (var a = this.offsetParent || dd; a&&!eb.nodeName(a, "html") && "static" === eb.css(a, "position");)
                    a = a.offsetParent;
                return a || dd
            })
        }
    }), eb.each({
        scrollLeft: "pageXOffset",
        scrollTop: "pageYOffset"
    }, function(a, b) {
        var c = /Y/.test(b);
        eb.fn[a] = function(d) {
            return Db(this, function(a, d, e) {
                var f = V(a);
                return void 0 === e ? f ? b in f ? f[b] : f.document.documentElement[d] : a[d] : void(f ? f.scrollTo(c ? eb(f).scrollLeft() : e, c ? e : eb(f).scrollTop()) : a[d] = e)
            }, a, d, arguments.length, null)
        }
    }), eb.each(["top", "left"], function(a, b) {
        eb.cssHooks[b] = A(cb.pixelPosition, function(a, c) {
            return c ? (c = bc(a, b), dc.test(c) ? eb(a).position()[b] + "px" : c) : void 0
        })
    }), eb.each({
        Height: "height",
        Width: "width"
    }, function(a, b) {
        eb.each({
            padding: "inner" + a,
            content: b,
            "": "outer" + a
        }, function(c, d) {
            eb.fn[d] = function(d, e) {
                var f = arguments.length && (c || "boolean" !== typeof d), g = c || (d===!0 || e===!0 ? "margin" : "border");
                return Db(this, function(b, c, d) {
                    var e;
                    return eb.isWindow(b) ? b.document.documentElement["client" + a] : 9 === b.nodeType ? (e = b.documentElement, Math.max(b.body["scroll" + a], e["scroll" + a], b.body["offset" + a], e["offset" + a], e["client" + a])) : void 0 === d ? eb.css(b, c, g) : eb.style(b, c, d, g)
                }, b, f ? d : void 0, f, null)
            }
        })
    }), eb.fn.size = function() {
        return this.length
    }, eb.fn.andSelf = eb.fn.addBack, "function" === typeof define && define.amd && define("jquery", [], function() {
        return eb
    });
    var ed = a.jQuery, fd = a.$;
    return eb.noConflict = function(b) {
        return a.$ === eb && (a.$ = fd), b && a.jQuery === eb && (a.jQuery = ed), eb
    }, typeof b === xb && (a.jQuery = a.$ = eb), eb
}), function() {
    var a = this, b = a._, c = {}, d = Array.prototype, e = Object.prototype, f = Function.prototype, g = d.push, h = d.slice, i = d.concat, j = e.toString, k = e.hasOwnProperty, l = d.forEach, m = d.map, n = d.reduce, o = d.reduceRight, p = d.filter, q = d.every, r = d.some, s = d.indexOf, t = d.lastIndexOf, u = Array.isArray, v = Object.keys, w = f.bind, x = function(a) {
        return a instanceof x ? a : this instanceof x ? void(this._wrapped = a) : new x(a)
    };
    "undefined" !== typeof exports ? ("undefined" !== typeof module && module.exports && (exports = module.exports = x), exports._ = x) : a._ = x, x.VERSION = "1.5.2";
    var y = x.each = x.forEach = function(a, b, d) {
        if (null != a)
            if (l && a.forEach === l)
                a.forEach(b, d);
            else if (a.length ===+ a.length) {
                for (var e = 0, f = a.length; e < f; e++)
                    if (b.call(d, a[e], e, a) === c)
                        return 
            } else 
                for (var g = x.keys(a), e = 0, f = g.length; e < f; e++)
                    if (b.call(d, a[g[e]], g[e], a) === c)
                        return 
    };
    x.map = x.collect = function(a, b, c) {
        var d = [];
        return null == a ? d : m && a.map === m ? a.map(b, c) : (y(a, function(a, e, f) {
            d.push(b.call(c, a, e, f))
        }), d)
    };
    var z = "Reduce of empty array with no initial value";
    x.reduce = x.foldl = x.inject = function(a, b, c, d) {
        var e = arguments.length > 2;
        if (null == a && (a = []), n && a.reduce === n)
            return d && (b = x.bind(b, d)), e ? a.reduce(b, c) : a.reduce(b);
        if (y(a, function(a, f, g) {
            e ? c = b.call(d, c, a, f, g) : (c = a, e=!0)
        }), !e)
            throw new TypeError(z);
        return c
    }, x.reduceRight = x.foldr = function(a, b, c, d) {
        var e = arguments.length > 2;
        if (null == a && (a = []), o && a.reduceRight === o)
            return d && (b = x.bind(b, d)), e ? a.reduceRight(b, c) : a.reduceRight(b);
        var f = a.length;
        if (f!==+f) {
            var g = x.keys(a);
            f = g.length
        }
        if (y(a, function(h, i, j) {
            i = g ? g[--f] : --f, e ? c = b.call(d, c, a[i], i, j) : (c = a[i], e=!0)
        }), !e)
            throw new TypeError(z);
        return c
    }, x.find = x.detect = function(a, b, c) {
        var d;
        return A(a, function(a, e, f) {
            return b.call(c, a, e, f) ? (d = a, !0) : void 0
        }), d
    }, x.filter = x.select = function(a, b, c) {
        var d = [];
        return null == a ? d : p && a.filter === p ? a.filter(b, c) : (y(a, function(a, e, f) {
            b.call(c, a, e, f) && d.push(a)
        }), d)
    }, x.reject = function(a, b, c) {
        return x.filter(a, function(a, d, e) {
            return !b.call(c, a, d, e)
        }, c)
    }, x.every = x.all = function(a, b, d) {
        b || (b = x.identity);
        var e=!0;
        return null == a ? e : q && a.every === q ? a.every(b, d) : (y(a, function(a, f, g) {
            return (e = e && b.call(d, a, f, g)) ? void 0 : c
        }), !!e)
    };
    var A = x.some = x.any = function(a, b, d) {
        b || (b = x.identity);
        var e=!1;
        return null == a ? e : r && a.some === r ? a.some(b, d) : (y(a, function(a, f, g) {
            return e || (e = b.call(d, a, f, g)) ? c : void 0
        }), !!e)
    };
    x.contains = x.include = function(a, b) {
        return null == a?!1 : s && a.indexOf === s?-1 != a.indexOf(b) : A(a, function(a) {
            return a === b
        })
    }, x.invoke = function(a, b) {
        var c = h.call(arguments, 2), d = x.isFunction(b);
        return x.map(a, function(a) {
            return (d ? b : a[b]).apply(a, c)
        })
    }, x.pluck = function(a, b) {
        return x.map(a, function(a) {
            return a[b]
        })
    }, x.where = function(a, b, c) {
        return x.isEmpty(b) ? c ? void 0 : [] : x[c ? "find": "filter"](a, function(a) {
            for (var c in b)
                if (b[c] !== a[c])
                    return !1;
            return !0
        })
    }, x.findWhere = function(a, b) {
        return x.where(a, b, !0)
    }, x.max = function(a, b, c) {
        if (!b && x.isArray(a) && a[0] ===+ a[0] && a.length < 65535)
            return Math.max.apply(Math, a);
        if (!b && x.isEmpty(a))
            return - 1 / 0;
        var d = {
            computed: - 1 / 0,
            value: - 1 / 0
        };
        return y(a, function(a, e, f) {
            var g = b ? b.call(c, a, e, f): a;
            g > d.computed && (d = {
                value: a,
                computed: g
            })
        }), d.value
    }, x.min = function(a, b, c) {
        if (!b && x.isArray(a) && a[0] ===+ a[0] && a.length < 65535)
            return Math.min.apply(Math, a);
        if (!b && x.isEmpty(a))
            return 1 / 0;
        var d = {
            computed: 1 / 0,
            value: 1 / 0
        };
        return y(a, function(a, e, f) {
            var g = b ? b.call(c, a, e, f): a;
            g < d.computed && (d = {
                value: a,
                computed: g
            })
        }), d.value
    }, x.shuffle = function(a) {
        var b, c = 0, d = [];
        return y(a, function(a) {
            b = x.random(c++), d[c - 1] = d[b], d[b] = a
        }), d
    }, x.sample = function(a, b, c) {
        return arguments.length < 2 || c ? a[x.random(a.length - 1)] : x.shuffle(a).slice(0, Math.max(0, b))
    };
    var B = function(a) {
        return x.isFunction(a) ? a : function(b) {
            return b[a]
        }
    };
    x.sortBy = function(a, b, c) {
        var d = B(b);
        return x.pluck(x.map(a, function(a, b, e) {
            return {
                value: a,
                index: b,
                criteria: d.call(c, a, b, e)
            }
        }).sort(function(a, b) {
            var c = a.criteria, d = b.criteria;
            if (c !== d) {
                if (c > d || void 0 === c)
                    return 1;
                if (c < d || void 0 === d)
                    return - 1
            }
            return a.index - b.index
        }), "value")
    };
    var C = function(a) {
        return function(b, c, d) {
            var e = {}, f = null == c ? x.identity: B(c);
            return y(b, function(c, g) {
                var h = f.call(d, c, g, b);
                a(e, h, c)
            }), e
        }
    };
    x.groupBy = C(function(a, b, c) {
        (x.has(a, b) ? a[b] : a[b] = []).push(c)
    }), x.indexBy = C(function(a, b, c) {
        a[b] = c
    }), x.countBy = C(function(a, b) {
        x.has(a, b) ? a[b]++ : a[b] = 1
    }), x.sortedIndex = function(a, b, c, d) {
        c = null == c ? x.identity : B(c);
        for (var e = c.call(d, b), f = 0, g = a.length; f < g;) {
            var h = f + g>>>1;
            c.call(d, a[h]) < e ? f = h + 1 : g = h
        }
        return f
    }, x.toArray = function(a) {
        return a ? x.isArray(a) ? h.call(a) : a.length ===+ a.length ? x.map(a, x.identity) : x.values(a) : []
    }, x.size = function(a) {
        return null == a ? 0 : a.length ===+ a.length ? a.length : x.keys(a).length
    }, x.first = x.head = x.take = function(a, b, c) {
        return null == a ? void 0 : null == b || c ? a[0] : h.call(a, 0, b)
    }, x.initial = function(a, b, c) {
        return h.call(a, 0, a.length - (null == b || c ? 1 : b))
    }, x.last = function(a, b, c) {
        return null == a ? void 0 : null == b || c ? a[a.length - 1] : h.call(a, Math.max(a.length - b, 0))
    }, x.rest = x.tail = x.drop = function(a, b, c) {
        return h.call(a, null == b || c ? 1 : b)
    }, x.compact = function(a) {
        return x.filter(a, x.identity)
    };
    var D = function(a, b, c) {
        return b && x.every(a, x.isArray) ? i.apply(c, a) : (y(a, function(a) {
            x.isArray(a) || x.isArguments(a) ? b ? g.apply(c, a) : D(a, b, c) : c.push(a)
        }), c)
    };
    x.flatten = function(a, b) {
        return D(a, b, [])
    }, x.without = function(a) {
        return x.difference(a, h.call(arguments, 1))
    }, x.uniq = x.unique = function(a, b, c, d) {
        x.isFunction(b) && (d = c, c = b, b=!1);
        var e = c ? x.map(a, c, d): a, f = [], g = [];
        return y(e, function(c, d) {
            (b ? d && g[g.length - 1] === c : x.contains(g, c)) || (g.push(c), f.push(a[d]))
        }), f
    }, x.union = function() {
        return x.uniq(x.flatten(arguments, !0))
    }, x.intersection = function(a) {
        var b = h.call(arguments, 1);
        return x.filter(x.uniq(a), function(a) {
            return x.every(b, function(b) {
                return x.indexOf(b, a) >= 0
            })
        })
    }, x.difference = function(a) {
        var b = i.apply(d, h.call(arguments, 1));
        return x.filter(a, function(a) {
            return !x.contains(b, a)
        })
    }, x.zip = function() {
        for (var a = x.max(x.pluck(arguments, "length").concat(0)), b = new Array(a), c = 0; c < a; c++)
            b[c] = x.pluck(arguments, "" + c);
        return b
    }, x.object = function(a, b) {
        if (null == a)
            return {};
        for (var c = {}, d = 0, e = a.length; d < e; d++)
            b ? c[a[d]] = b[d] : c[a[d][0]] = a[d][1];
        return c
    }, x.indexOf = function(a, b, c) {
        if (null == a)
            return - 1;
        var d = 0, e = a.length;
        if (c) {
            if ("number" != typeof c)
                return d = x.sortedIndex(a, b), a[d] === b ? d : - 1;
            d = c < 0 ? Math.max(0, e + c) : c
        }
        if (s && a.indexOf === s)
            return a.indexOf(b, c);
        for (; d < e; d++)
            if (a[d] === b)
                return d;
        return - 1
    }, x.lastIndexOf = function(a, b, c) {
        if (null == a)
            return - 1;
        var d = null != c;
        if (t && a.lastIndexOf === t)
            return d ? a.lastIndexOf(b, c) : a.lastIndexOf(b);
        for (var e = d ? c : a.length; e--;)
            if (a[e] === b)
                return e;
        return - 1
    }, x.range = function(a, b, c) {
        arguments.length <= 1 && (b = a || 0, a = 0), c = arguments[2] || 1;
        for (var d = Math.max(Math.ceil((b - a) / c), 0), e = 0, f = new Array(d); e < d;)
            f[e++] = a, a += c;
        return f
    };
    var E = function() {};
    x.bind = function(a, b) {
        var c, d;
        if (w && a.bind === w)
            return w.apply(a, h.call(arguments, 1));
        if (!x.isFunction(a))
            throw new TypeError;
        return c = h.call(arguments, 2), d = function() {
            if (!(this instanceof d))
                return a.apply(b, c.concat(h.call(arguments)));
            E.prototype = a.prototype;
            var e = new E;
            E.prototype = null;
            var f = a.apply(e, c.concat(h.call(arguments)));
            return Object(f) === f ? f : e
        }
    }, x.partial = function(a) {
        var b = h.call(arguments, 1);
        return function() {
            return a.apply(this, b.concat(h.call(arguments)))
        }
    }, x.bindAll = function(a) {
        var b = h.call(arguments, 1);
        if (0 === b.length)
            throw new Error("bindAll must be passed function names");
        return y(b, function(b) {
            a[b] = x.bind(a[b], a)
        }), a
    }, x.memoize = function(a, b) {
        var c = {};
        return b || (b = x.identity), function() {
            var d = b.apply(this, arguments);
            return x.has(c, d) ? c[d] : c[d] = a.apply(this, arguments)
        }
    }, x.delay = function(a, b) {
        var c = h.call(arguments, 2);
        return setTimeout(function() {
            return a.apply(null, c)
        }, b)
    }, x.defer = function(a) {
        return x.delay.apply(x, [a, 1].concat(h.call(arguments, 1)))
    }, x.throttle = function(a, b, c) {
        var d, e, f, g = null, h = 0;
        c || (c = {});
        var i = function() {
            h = c.leading===!1 ? 0 : new Date, g = null, f = a.apply(d, e)
        };
        return function() {
            var j = new Date;
            h || c.leading!==!1 || (h = j);
            var k = b - (j - h);
            return d = this, e = arguments, k <= 0 ? (clearTimeout(g), g = null, h = j, f = a.apply(d, e)) : g || c.trailing===!1 || (g = setTimeout(i, k)), f
        }
    }, x.debounce = function(a, b, c) {
        var d, e, f, g, h;
        return function() {
            f = this, e = arguments, g = new Date;
            var i = function() {
                var j = new Date - g;
                j < b ? d = setTimeout(i, b - j) : (d = null, c || (h = a.apply(f, e)))
            }, j = c&&!d;
            return d || (d = setTimeout(i, b)), j && (h = a.apply(f, e)), h
        }
    }, x.once = function(a) {
        var b, c=!1;
        return function() {
            return c ? b : (c=!0, b = a.apply(this, arguments), a = null, b)
        }
    }, x.wrap = function(a, b) {
        return function() {
            var c = [a];
            return g.apply(c, arguments), b.apply(this, c)
        }
    }, x.compose = function() {
        var a = arguments;
        return function() {
            for (var b = arguments, c = a.length - 1; c >= 0; c--)
                b = [a[c].apply(this, b)];
            return b[0]
        }
    }, x.after = function(a, b) {
        return function() {
            return --a < 1 ? b.apply(this, arguments) : void 0
        }
    }, x.keys = v || function(a) {
        if (a !== Object(a))
            throw new TypeError("Invalid object");
        var b = [];
        for (var c in a)
            x.has(a, c) && b.push(c);
        return b
    }, x.values = function(a) {
        for (var b = x.keys(a), c = b.length, d = new Array(c), e = 0; e < c; e++)
            d[e] = a[b[e]];
        return d
    }, x.pairs = function(a) {
        for (var b = x.keys(a), c = b.length, d = new Array(c), e = 0; e < c; e++)
            d[e] = [b[e], a[b[e]]];
        return d
    }, x.invert = function(a) {
        for (var b = {}, c = x.keys(a), d = 0, e = c.length; d < e; d++)
            b[a[c[d]]] = c[d];
        return b
    }, x.functions = x.methods = function(a) {
        var b = [];
        for (var c in a)
            x.isFunction(a[c]) && b.push(c);
        return b.sort()
    }, x.extend = function(a) {
        return y(h.call(arguments, 1), function(b) {
            if (b)
                for (var c in b)
                    a[c] = b[c]
        }), a
    }, x.pick = function(a) {
        var b = {}, c = i.apply(d, h.call(arguments, 1));
        return y(c, function(c) {
            c in a && (b[c] = a[c])
        }), b
    }, x.omit = function(a) {
        var b = {}, c = i.apply(d, h.call(arguments, 1));
        for (var e in a)
            x.contains(c, e) || (b[e] = a[e]);
        return b
    }, x.defaults = function(a) {
        return y(h.call(arguments, 1), function(b) {
            if (b)
                for (var c in b)
                    void 0 === a[c] && (a[c] = b[c])
        }), a
    }, x.clone = function(a) {
        return x.isObject(a) ? x.isArray(a) ? a.slice() : x.extend({}, a) : a
    }, x.tap = function(a, b) {
        return b(a), a
    };
    var F = function(a, b, c, d) {
        if (a === b)
            return 0 !== a || 1 / a == 1 / b;
        if (null == a || null == b)
            return a === b;
        a instanceof x && (a = a._wrapped), b instanceof x && (b = b._wrapped);
        var e = j.call(a);
        if (e != j.call(b))
            return !1;
        switch (e) {
        case"[object String]":
            return a == String(b);
        case"[object Number]":
            return a!=+a ? b!=+b : 0 == a ? 1 / a == 1 / b : a ==+ b;
        case"[object Date]":
        case"[object Boolean]":
            return + a ==+ b;
        case"[object RegExp]":
            return a.source == b.source && a.global == b.global && a.multiline == b.multiline && a.ignoreCase == b.ignoreCase
        }
        if ("object" != typeof a || "object" != typeof b)
            return !1;
        for (var f = c.length; f--;)
            if (c[f] == a)
                return d[f] == b;
        var g = a.constructor, h = b.constructor;
        if (g !== h&&!(x.isFunction(g) && g instanceof g && x.isFunction(h) && h instanceof h))
            return !1;
        c.push(a), d.push(b);
        var i = 0, k=!0;
        if ("[object Array]" == e) {
            if (i = a.length, k = i == b.length)
                for (; i--&&(k = F(a[i], b[i], c, d)););
        } else {
            for (var l in a)
                if (x.has(a, l) && (i++, !(k = x.has(b, l) && F(a[l], b[l], c, d))))
                    break;
            if (k) {
                for (l in b)
                    if (x.has(b, l)&&!i--)
                        break;
                k=!i
            }
        }
        return c.pop(), d.pop(), k
    };
    x.isEqual = function(a, b) {
        return F(a, b, [], [])
    }, x.isEmpty = function(a) {
        if (null == a)
            return !0;
        if (x.isArray(a) || x.isString(a))
            return 0 === a.length;
        for (var b in a)
            if (x.has(a, b))
                return !1;
        return !0
    }, x.isElement = function(a) {
        return !(!a || 1 !== a.nodeType)
    }, x.isArray = u || function(a) {
        return "[object Array]" == j.call(a)
    }, x.isObject = function(a) {
        return a === Object(a)
    }, y(["Arguments", "Function", "String", "Number", "Date", "RegExp"], function(a) {
        x["is" + a] = function(b) {
            return j.call(b) == "[object " + a + "]"
        }
    }), x.isArguments(arguments) || (x.isArguments = function(a) {
        return !(!a ||!x.has(a, "callee"))
    }), "function" !== typeof/./ && (x.isFunction = function(a) {
        return "function" === typeof a
    }), x.isFinite = function(a) {
        return isFinite(a)&&!isNaN(parseFloat(a))
    }, x.isNaN = function(a) {
        return x.isNumber(a) && a!=+a
    }, x.isBoolean = function(a) {
        return a===!0 || a===!1 || "[object Boolean]" == j.call(a)
    }, x.isNull = function(a) {
        return null === a
    }, x.isUndefined = function(a) {
        return void 0 === a
    }, x.has = function(a, b) {
        return k.call(a, b)
    }, x.noConflict = function() {
        return a._ = b, this
    }, x.identity = function(a) {
        return a
    }, x.times = function(a, b, c) {
        for (var d = Array(Math.max(0, a)), e = 0; e < a; e++)
            d[e] = b.call(c, e);
        return d
    }, x.random = function(a, b) {
        return null == b && (b = a, a = 0), a + Math.floor(Math.random() * (b - a + 1))
    };
    var G = {
        escape: {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;"
        }
    };
    G.unescape = x.invert(G.escape);
    var H = {
        escape: new RegExp("[" + x.keys(G.escape).join("") + "]", "g"),
        unescape: new RegExp("(" + x.keys(G.unescape).join("|") + ")", "g")
    };
    x.each(["escape", "unescape"], function(a) {
        x[a] = function(b) {
            return null == b ? "" : ("" + b).replace(H[a], function(b) {
                return G[a][b]
            })
        }
    }), x.result = function(a, b) {
        if (null == a)
            return void 0;
        var c = a[b];
        return x.isFunction(c) ? c.call(a) : c
    }, x.mixin = function(a) {
        y(x.functions(a), function(b) {
            var c = x[b] = a[b];
            x.prototype[b] = function() {
                var a = [this._wrapped];
                return g.apply(a, arguments), M.call(this, c.apply(x, a))
            }
        })
    };
    var I = 0;
    x.uniqueId = function(a) {
        var b=++I + "";
        return a ? a + b : b
    }, x.templateSettings = {
        evaluate: /<%([\s\S]+?)%>/g,
        interpolate: /<%=([\s\S]+?)%>/g,
        escape: /<%-([\s\S]+?)%>/g
    };
    var J = /(.)^/, K = {
        "'": "'",
        "\\": "\\",
        "\r": "r",
        "\n": "n",
        "	": "t",
        "\u2028": "u2028",
        "\u2029": "u2029"
    }, L = /\\|'|\r|\n|\t|\u2028|\u2029/g;
    x.template = function(a, b, c) {
        var d;
        c = x.defaults({}, c, x.templateSettings);
        var e = new RegExp([(c.escape || J).source, (c.interpolate || J).source, (c.evaluate || J).source].join("|") + "|$", "g"), f = 0, g = "__p+='";
        a.replace(e, function(b, c, d, e, h) {
            return g += a.slice(f, h).replace(L, function(a) {
                return "\\" + K[a]
            }), c && (g += "'+\n((__t=(" + c + "))==null?'':_.escape(__t))+\n'"), d && (g += "'+\n((__t=(" + d + "))==null?'':__t)+\n'"), e && (g += "';\n" + e + "\n__p+='"), f = h + b.length, b
        }), g += "';\n", c.variable || (g = "with(obj||{}){\n" + g + "}\n"), g = "var __t,__p='',__j=Array.prototype.join,print=function(){__p+=__j.call(arguments,'');};\n" + g + "return __p;\n";
        try {
            d = new Function(c.variable || "obj", "_", g)
        } catch (h) {
            throw h.source = g, h
        }
        if (b)
            return d(b, x);
        var i = function(a) {
            return d.call(this, a, x)
        };
        return i.source = "function(" + (c.variable || "obj") + "){\n" + g + "}", i
    }, x.chain = function(a) {
        return x(a).chain()
    };
    var M = function(a) {
        return this._chain ? x(a).chain() : a
    };
    x.mixin(x), y(["pop", "push", "reverse", "shift", "sort", "splice", "unshift"], function(a) {
        var b = d[a];
        x.prototype[a] = function() {
            var c = this._wrapped;
            return b.apply(c, arguments), "shift" != a && "splice" != a || 0 !== c.length || delete c[0], M.call(this, c)
        }
    }), y(["concat", "join", "slice"], function(a) {
        var b = d[a];
        x.prototype[a] = function() {
            return M.call(this, b.apply(this._wrapped, arguments))
        }
    }), x.extend(x.prototype, {
        chain: function() {
            return this._chain=!0, this
        },
        value: function() {
            return this._wrapped
        }
    })
}.call(this), function() {
    {
        var a, b = this, c = b.Backbone, d = [], e = (d.push, d.slice);
        d.splice
    }
    a = "undefined" !== typeof exports ? exports : b.Backbone = {}, a.VERSION = "1.1.0";
    var f = b._;
    f || "undefined" === typeof require || (f = require("underscore")), a.$ = b.jQuery || b.Zepto || b.ender || b.$, a.noConflict = function() {
        return b.Backbone = c, this
    }, a.emulateHTTP=!1, a.emulateJSON=!1;
    var g = a.Events = {
        on: function(a, b, c) {
            if (!i(this, "on", a, [b, c]) ||!b)
                return this;
            this._events || (this._events = {});
            var d = this._events[a] || (this._events[a] = []);
            return d.push({
                callback: b,
                context: c,
                ctx: c || this
            }), this
        },
        once: function(a, b, c) {
            if (!i(this, "once", a, [b, c]) ||!b)
                return this;
            var d = this, e = f.once(function() {
                d.off(a, e), b.apply(this, arguments)
            });
            return e._callback = b, this.on(a, e, c)
        },
        off: function(a, b, c) {
            var d, e, g, h, j, k, l, m;
            if (!this._events ||!i(this, "off", a, [b, c]))
                return this;
            if (!a&&!b&&!c)
                return this._events = {}, this;
            for (h = a ? [a] : f.keys(this._events), j = 0, k = h.length; j < k; j++)
                if (a = h[j], g = this._events[a]) {
                    if (this._events[a] = d = [], b || c)
                        for (l = 0, m = g.length; l < m; l++)
                            e = g[l], (b && b !== e.callback && b !== e.callback._callback || c && c !== e.context) && d.push(e);
                            d.length || delete this._events[a]
                }
            return this
        },
        trigger: function(a) {
            if (!this._events)
                return this;
            var b = e.call(arguments, 1);
            if (!i(this, "trigger", a, b))
                return this;
            var c = this._events[a], d = this._events.all;
            return c && j(c, b), d && j(d, arguments), this
        },
        stopListening: function(a, b, c) {
            var d = this._listeningTo;
            if (!d)
                return this;
            var e=!b&&!c;
            c || "object" !== typeof b || (c = this), a && ((d = {})[a._listenId] = a);
            for (var g in d)
                a = d[g], a.off(b, c, this), (e || f.isEmpty(a._events)) && delete this._listeningTo[g];
            return this
        }
    }, h = /\s+/, i = function(a, b, c, d) {
        if (!c)
            return !0;
        if ("object" === typeof c) {
            for (var e in c)
                a[b].apply(a, [e, c[e]].concat(d));
            return !1
        }
        if (h.test(c)) {
            for (var f = c.split(h), g = 0, i = f.length; g < i; g++)
                a[b].apply(a, [f[g]].concat(d));
            return !1
        }
        return !0
    }, j = function(a, b) {
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
                .callback.apply(c.ctx, b)
        }
    }, k = {
        listenTo: "on",
        listenToOnce: "once"
    };
    f.each(k, function(a, b) {
        g[b] = function(b, c, d) {
            var e = this._listeningTo || (this._listeningTo = {}), g = b._listenId || (b._listenId = f.uniqueId("l"));
            return e[g] = b, d || "object" !== typeof c || (d = this), b[a](c, d, this), this
        }
    }), g.bind = g.on, g.unbind = g.off, f.extend(a, g);
    var l = a.Model = function(a, b) {
        var c = a || {};
        b || (b = {}), this.cid = f.uniqueId("c"), this.attributes = {}, b.collection && (this.collection = b.collection), b.parse && (c = this.parse(c, b) || {}), c = f.defaults({}, c, f.result(this, "defaults")), this.set(c, b), this.changed = {}, this.initialize.apply(this, arguments)
    };
    f.extend(l.prototype, g, {
        changed: null,
        validationError: null,
        idAttribute: "id",
        initialize: function() {},
        toJSON: function() {
            return f.clone(this.attributes)
        },
        sync: function() {
            return a.sync.apply(this, arguments)
        },
        get: function(a) {
            return this.attributes[a]
        },
        escape: function(a) {
            return f.escape(this.get(a))
        },
        has: function(a) {
            return null != this.get(a)
        },
        set: function(a, b, c) {
            var d, e, g, h, i, j, k, l;
            if (null == a)
                return this;
            if ("object" === typeof a ? (e = a, c = b) : (e = {})[a] = b, c || (c = {}), !this._validate(e, c))
                return !1;
            g = c.unset, i = c.silent, h = [], j = this._changing, this._changing=!0, j || (this._previousAttributes = f.clone(this.attributes), this.changed = {}), l = this.attributes, k = this._previousAttributes, this.idAttribute in e && (this.id = e[this.idAttribute]);
            for (d in e)
                b = e[d], f.isEqual(l[d], b) || h.push(d), f.isEqual(k[d], b) ? delete this.changed[d] : this.changed[d] = b, g ? delete l[d] : l[d] = b;
            if (!i) {
                h.length && (this._pending=!0);
                for (var m = 0, n = h.length; m < n; m++)
                    this.trigger("change:" + h[m], this, l[h[m]], c)
            }
            if (j)
                return this;
            if (!i)
                for (; this._pending;)
                    this._pending=!1, this.trigger("change", this, c);
            return this._pending=!1, this._changing=!1, this
        },
        unset: function(a, b) {
            return this.set(a, void 0, f.extend({}, b, {
                unset : !0
            }))
        },
        clear: function(a) {
            var b = {};
            for (var c in this.attributes)
                b[c] = void 0;
            return this.set(b, f.extend({}, a, {
                unset: !0
            }))
        },
        hasChanged: function(a) {
            return null == a?!f.isEmpty(this.changed) : f.has(this.changed, a)
        },
        changedAttributes: function(a) {
            if (!a)
                return this.hasChanged() ? f.clone(this.changed) : !1;
            var b, c=!1, d = this._changing ? this._previousAttributes : this.attributes;
            for (var e in a)
                f.isEqual(d[e], b = a[e]) || ((c || (c = {}))[e] = b);
            return c
        },
        previous: function(a) {
            return null != a && this._previousAttributes ? this._previousAttributes[a] : null
        },
        previousAttributes: function() {
            return f.clone(this._previousAttributes)
        },
        fetch: function(a) {
            a = a ? f.clone(a) : {}, void 0 === a.parse && (a.parse=!0);
            var b = this, c = a.success;
            return a.success = function(d) {
                return b.set(b.parse(d, a), a) ? (c && c(b, d, a), void b.trigger("sync", b, d, a)) : !1
            }, K(this, a), this.sync("read", this, a)
        },
        save: function(a, b, c) {
            var d, e, g, h = this.attributes;
            if (null == a || "object" === typeof a ? (d = a, c = b) : (d = {})[a] = b, c = f.extend({
                validate: !0
            }, c), d&&!c.wait) {
                if (!this.set(d, c))
                    return !1
            } else if (!this._validate(d, c))
                return !1;
            d && c.wait && (this.attributes = f.extend({}, h, d)), void 0 === c.parse && (c.parse=!0);
            var i = this, j = c.success;
            return c.success = function(a) {
                i.attributes = h;
                var b = i.parse(a, c);
                return c.wait && (b = f.extend(d || {}, b)), f.isObject(b)&&!i.set(b, c)?!1 : (j && j(i, a, c), void i.trigger("sync", i, a, c))
            }, K(this, c), e = this.isNew() ? "create" : c.patch ? "patch" : "update", "patch" === e && (c.attrs = d), g = this.sync(e, this, c), d && c.wait && (this.attributes = h), g
        },
        destroy: function(a) {
            a = a ? f.clone(a) : {};
            var b = this, c = a.success, d = function() {
                b.trigger("destroy", b, b.collection, a)
            };
            if (a.success = function(e) {
                (a.wait || b.isNew()) && d(), c && c(b, e, a), b.isNew() || b.trigger("sync", b, e, a)
            }, this.isNew())
                return a.success(), !1;
            K(this, a);
            var e = this.sync("delete", this, a);
            return a.wait || d(), e
        },
        url: function() {
            var a = f.result(this, "urlRoot") || f.result(this.collection, "url") || J();
            return this.isNew() ? a : a + ("/" === a.charAt(a.length - 1) ? "" : "/") + encodeURIComponent(this.id)
        },
        parse: function(a) {
            return a
        },
        clone: function() {
            return new this.constructor(this.attributes)
        },
        isNew: function() {
            return null == this.id
        },
        isValid: function(a) {
            return this._validate({}, f.extend(a || {}, {
                validate: !0
            }))
        },
        _validate: function(a, b) {
            if (!b.validate ||!this.validate)
                return !0;
            a = f.extend({}, this.attributes, a);
            var c = this.validationError = this.validate(a, b) || null;
            return c ? (this.trigger("invalid", this, c, f.extend(b, {
                validationError: c
            })), !1) : !0
        }
    });
    var m = ["keys", "values", "pairs", "invert", "pick", "omit"];
    f.each(m, function(a) {
        l.prototype[a] = function() {
            var b = e.call(arguments);
            return b.unshift(this.attributes), f[a].apply(f, b)
        }
    });
    var n = a.Collection = function(a, b) {
        b || (b = {}), b.model && (this.model = b.model), void 0 !== b.comparator && (this.comparator = b.comparator), this._reset(), this.initialize.apply(this, arguments), a && this.reset(a, f.extend({
            silent: !0
        }, b))
    }, o = {
        add: !0,
        remove: !0,
        merge: !0
    }, p = {
        add: !0,
        remove: !1
    };
    f.extend(n.prototype, g, {
        model: l,
        initialize: function() {},
        toJSON: function(a) {
            return this.map(function(b) {
                return b.toJSON(a)
            })
        },
        sync: function() {
            return a.sync.apply(this, arguments)
        },
        add: function(a, b) {
            return this.set(a, f.extend({
                merge: !1
            }, b, p))
        },
        remove: function(a, b) {
            var c=!f.isArray(a);
            a = c ? [a] : f.clone(a), b || (b = {});
            var d, e, g, h;
            for (d = 0, e = a.length; d < e; d++)
                h = a[d] = this.get(a[d]), h && (delete this._byId[h.id], delete this._byId[h.cid], g = this.indexOf(h), this.models.splice(g, 1), this.length--, b.silent || (b.index = g, h.trigger("remove", h, this, b)), this._removeReference(h));
            return c ? a[0] : a
        },
        set: function(a, b) {
            b = f.defaults({}, b, o), b.parse && (a = this.parse(a, b));
            var c=!f.isArray(a);
            a = c ? a ? [a] : [] : f.clone(a);
            var d, e, g, h, i, j, k, m = b.at, n = this.model, p = this.comparator && null == m && b.sort!==!1, q = f.isString(this.comparator) ? this.comparator: null, r = [], s = [], t = {}, u = b.add, v = b.merge, w = b.remove, x=!p && u && w ? [] : !1;
            for (d = 0, e = a.length; d < e; d++) {
                if (i = a[d], g = i instanceof l ? h = i : i[n.prototype.idAttribute], j = this.get(g))
                    w && (t[j.cid]=!0), v && (i = i === h ? h.attributes : i, b.parse && (i = j.parse(i, b)), j.set(i, b), p&&!k && j.hasChanged(q) && (k=!0)), a[d] = j;
                else if (u) {
                    if (h = a[d] = this._prepareModel(i, b), !h)
                        continue;
                    r.push(h), h.on("all", this._onModelEvent, this), this._byId[h.cid] = h, null != h.id && (this._byId[h.id] = h)
                }
                x && x.push(j || h)
            }
            if (w) {
                for (d = 0, e = this.length; d < e; ++d)
                    t[(h = this.models[d]).cid] || s.push(h);
                s.length && this.remove(s, b)
            }
            if (r.length || x && x.length)
                if (p && (k=!0), this.length += r.length, null != m)
                    for (d = 0, e = r.length; d < e; d++)
                        this.models.splice(m + d, 0, r[d]);
                else {
                    x && (this.models.length = 0);
                    var y = x || r;
                    for (d = 0, e = y.length; d < e; d++)
                        this.models.push(y[d])
                }
            if (k && this.sort({
                silent: !0
            }), !b.silent) {
                for (d = 0, e = r.length; d < e; d++)(h = r[d])
                    .trigger("add", h, this, b);
                (k || x && x.length) && this.trigger("sort", this, b)
            }
            return c ? a[0] : a
        },
        reset: function(a, b) {
            b || (b = {});
            for (var c = 0, d = this.models.length; c < d; c++)
                this._removeReference(this.models[c]);
            return b.previousModels = this.models, this._reset(), a = this.add(a, f.extend({
                silent: !0
            }, b)), b.silent || this.trigger("reset", this, b), a
        },
        push: function(a, b) {
            return this.add(a, f.extend({
                at: this.length
            }, b))
        },
        pop: function(a) {
            var b = this.at(this.length - 1);
            return this.remove(b, a), b
        },
        unshift: function(a, b) {
            return this.add(a, f.extend({
                at: 0
            }, b))
        },
        shift: function(a) {
            var b = this.at(0);
            return this.remove(b, a), b
        },
        slice: function() {
            return e.apply(this.models, arguments)
        },
        get: function(a) {
            return null == a ? void 0 : this._byId[a.id] || this._byId[a.cid] || this._byId[a]
        },
        at: function(a) {
            return this.models[a]
        },
        where: function(a, b) {
            return f.isEmpty(a) ? b ? void 0 : [] : this[b ? "find": "filter"](function(b) {
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
            return a || (a = {}), f.isString(this.comparator) || 1 === this.comparator.length ? this.models = this.sortBy(this.comparator, this) : this.models.sort(f.bind(this.comparator, this)), a.silent || this.trigger("sort", this, a), this
        },
        pluck: function(a) {
            return f.invoke(this.models, "get", a)
        },
        fetch: function(a) {
            a = a ? f.clone(a) : {}, void 0 === a.parse && (a.parse=!0);
            var b = a.success, c = this;
            return a.success = function(d) {
                var e = a.reset ? "reset": "set";
                c[e](d, a), b && b(c, d, a), c.trigger("sync", c, d, a)
            }, K(this, a), this.sync("read", this, a)
        },
        create: function(a, b) {
            if (b = b ? f.clone(b) : {}, !(a = this._prepareModel(a, b)))
                return !1;
            b.wait || this.add(a, b);
            var c = this, d = b.success;
            return b.success = function(a, b, e) {
                e.wait && c.add(a, e), d && d(a, b, e)
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
            if (a instanceof l)
                return a.collection || (a.collection = this), a;
            b = b ? f.clone(b) : {}, b.collection = this;
            var c = new this.model(a, b);
            return c.validationError ? (this.trigger("invalid", this, c.validationError, b), !1) : c
        },
        _removeReference: function(a) {
            this === a.collection && delete a.collection, a.off("all", this._onModelEvent, this)
        },
        _onModelEvent: function(a, b, c, d) {
            ("add" !== a && "remove" !== a || c === this) && ("destroy" === a && this.remove(b, d), b && a === "change:" + b.idAttribute && (delete this._byId[b.previous(b.idAttribute)], null != b.id && (this._byId[b.id] = b)), this.trigger.apply(this, arguments))
        }
    });
    var q = ["forEach", "each", "map", "collect", "reduce", "foldl", "inject", "reduceRight", "foldr", "find", "detect", "filter", "select", "reject", "every", "all", "some", "any", "include", "contains", "invoke", "max", "min", "toArray", "size", "first", "head", "take", "initial", "rest", "tail", "drop", "last", "without", "difference", "indexOf", "shuffle", "lastIndexOf", "isEmpty", "chain"];
    f.each(q, function(a) {
        n.prototype[a] = function() {
            var b = e.call(arguments);
            return b.unshift(this.models), f[a].apply(f, b)
        }
    });
    var r = ["groupBy", "countBy", "sortBy"];
    f.each(r, function(a) {
        n.prototype[a] = function(b, c) {
            var d = f.isFunction(b) ? b: function(a) {
                return a.get(b)
            };
            return f[a](this.models, d, c)
        }
    });
    var s = a.View = function(a) {
        this.cid = f.uniqueId("view"), a || (a = {}), f.extend(this, f.pick(a, u)), this._ensureElement(), this.initialize.apply(this, arguments), this.delegateEvents()
    }, t = /^(\S+)\s*(.*)$/, u = ["model", "collection", "el", "id", "attributes", "className", "tagName", "events"];
    f.extend(s.prototype, g, {
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
        setElement: function(b, c) {
            return this.$el && this.undelegateEvents(), this.$el = b instanceof a.$ ? b : a.$(b), this.el = this.$el[0], c!==!1 && this.delegateEvents(), this
        },
        delegateEvents: function(a) {
            if (!a&&!(a = f.result(this, "events")))
                return this;
            this.undelegateEvents();
            for (var b in a) {
                var c = a[b];
                if (f.isFunction(c) || (c = this[a[b]]), c) {
                    var d = b.match(t), e = d[1], g = d[2];
                    c = f.bind(c, this), e += ".delegateEvents" + this.cid, "" === g ? this.$el.on(e, c) : this.$el.on(e, g, c)
                }
            }
            return this
        },
        undelegateEvents: function() {
            return this.$el.off(".delegateEvents" + this.cid), this
        },
        _ensureElement: function() {
            if (this.el)
                this.setElement(f.result(this, "el"), !1);
            else {
                var b = f.extend({}, f.result(this, "attributes"));
                this.id && (b.id = f.result(this, "id")), this.className && (b["class"] = f.result(this, "className"));
                var c = a.$("<" + f.result(this, "tagName") + ">").attr(b);
                this.setElement(c, !1)
            }
        }
    }), a.sync = function(b, c, d) {
        var e = w[b];
        f.defaults(d || (d = {}), {
            emulateHTTP: a.emulateHTTP,
            emulateJSON: a.emulateJSON
        });
        var g = {
            type: e,
            dataType: "json"
        };
        if (d.url || (g.url = f.result(c, "url") || J()), null != d.data ||!c || "create" !== b && "update" !== b && "patch" !== b || (g.contentType = "application/json", g.data = JSON.stringify(d.attrs || c.toJSON(d))), d.emulateJSON && (g.contentType = "application/x-www-form-urlencoded", g.data = g.data ? {
            model: g.data
        } : {}), d.emulateHTTP && ("PUT" === e || "DELETE" === e || "PATCH" === e)) {
            g.type = "POST", d.emulateJSON && (g.data._method = e);
            var h = d.beforeSend;
            d.beforeSend = function(a) {
                return a.setRequestHeader("X-HTTP-Method-Override", e), h ? h.apply(this, arguments) : void 0
            }
        }
        "GET" === g.type || d.emulateJSON || (g.processData=!1), "PATCH" === g.type && v && (g.xhr = function() {
            return new ActiveXObject("Microsoft.XMLHTTP")
        });
        var i = d.xhr = a.ajax(f.extend(g, d));
        return c.trigger("request", c, i, d), i
    };
    var v = "undefined" !== typeof window&&!!window.ActiveXObject&&!(window.XMLHttpRequest && (new XMLHttpRequest).dispatchEvent), w = {
        create: "POST",
        update: "PUT",
        patch: "PATCH",
        "delete": "DELETE",
        read: "GET"
    };
    a.ajax = function() {
        return a.$.ajax.apply(a.$, arguments)
    };
    var x = a.Router = function(a) {
        a || (a = {}), a.routes && (this.routes = a.routes), this._bindRoutes(), this.initialize.apply(this, arguments)
    }, y = /\((.*?)\)/g, z = /(\(\?)?:\w+/g, A = /\*\w+/g, B = /[\-{}\[\]+?.,\\\^$|#\s]/g;
    f.extend(x.prototype, g, {
        initialize: function() {},
        route: function(b, c, d) {
            f.isRegExp(b) || (b = this._routeToRegExp(b)), f.isFunction(c) && (d = c, c = ""), d || (d = this[c]);
            var e = this;
            return a.history.route(b, function(f) {
                var g = e._extractParameters(b, f);
                d && d.apply(e, g), e.trigger.apply(e, ["route:" + c].concat(g)), e.trigger("route", c, g), a.history.trigger("route", e, c, g)
            }), this
        },
        navigate: function(b, c) {
            return a.history.navigate(b, c), this
        },
        _bindRoutes: function() {
            if (this.routes) {
                this.routes = f.result(this, "routes");
                for (var a, b = f.keys(this.routes); null != (a = b.pop());)
                    this.route(a, this.routes[a])
            }
        },
        _routeToRegExp: function(a) {
            return a = a.replace(B, "\\$&").replace(y, "(?:$1)?").replace(z, function(a, b) {
                return b ? a : "([^/]+)"
            }).replace(A, "(.*?)"), new RegExp("^" + a + "$")
        },
        _extractParameters: function(a, b) {
            var c = a.exec(b).slice(1);
            return f.map(c, function(a) {
                return a ? decodeURIComponent(a) : null
            })
        }
    });
    var C = a.History = function() {
        this.handlers = [], f.bindAll(this, "checkUrl"), "undefined" !== typeof window && (this.location = window.location, this.history = window.history)
    }, D = /^[#\/]|\s+$/g, E = /^\/+|\/+$/g, F = /msie [\w.]+/, G = /\/$/, H = /[?#].*$/;
    C.started=!1, f.extend(C.prototype, g, {
        interval: 50,
        getHash: function(a) {
            var b = (a || this).location.href.match(/#(.*)$/);
            return b ? b[1] : ""
        },
        getFragment: function(a, b) {
            if (null == a)
                if (this._hasPushState ||!this._wantsHashChange || b) {
                    a = this.location.pathname;
                    var c = this.root.replace(G, "");
                    a.indexOf(c) || (a = a.slice(c.length))
                } else 
                    a = this.getHash();
            return a.replace(D, "")
        },
        start: function(b) {
            if (C.started)
                throw new Error("Backbone.history has already been started");
            C.started=!0, this.options = f.extend({
                root: "/"
            }, this.options, b), this.root = this.options.root, this._wantsHashChange = this.options.hashChange!==!1, this._wantsPushState=!!this.options.pushState, this._hasPushState=!!(this.options.pushState && this.history && this.history.pushState);
            var c = this.getFragment(), d = document.documentMode, e = F.exec(navigator.userAgent.toLowerCase()) && (!d || d <= 7);
            this.root = ("/" + this.root + "/").replace(E, "/"), e && this._wantsHashChange && (this.iframe = a.$('<iframe src="javascript:0" tabindex="-1" />').hide().appendTo("body")[0].contentWindow, this.navigate(c)), this._hasPushState ? a.$(window).on("popstate", this.checkUrl) : this._wantsHashChange && "onhashchange"in window&&!e ? a.$(window).on("hashchange", this.checkUrl) : this._wantsHashChange && (this._checkUrlInterval = setInterval(this.checkUrl, this.interval)), this.fragment = c;
            var g = this.location, h = g.pathname.replace(/[^\/]$/, "$&/") === this.root;
            if (this._wantsHashChange && this._wantsPushState) {
                if (!this._hasPushState&&!h)
                    return this.fragment = this.getFragment(null, !0), this.location.replace(this.root + this.location.search + "#" + this.fragment), !0;
                this._hasPushState && h && g.hash && (this.fragment = this.getHash().replace(D, ""), this.history.replaceState({}, document.title, this.root + this.fragment + g.search))
            }
            return this.options.silent ? void 0 : this.loadUrl()
        },
        stop: function() {
            a.$(window).off("popstate", this.checkUrl).off("hashchange", this.checkUrl), clearInterval(this._checkUrlInterval), C.started=!1
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
            return a = this.fragment = this.getFragment(a), f.any(this.handlers, function(b) {
                return b.route.test(a) ? (b.callback(a), !0) : void 0
            })
        },
        navigate: function(a, b) {
            if (!C.started)
                return !1;
            b && b!==!0 || (b = {
                trigger: !!b
            });
            var c = this.root + (a = this.getFragment(a || ""));
            if (a = a.replace(H, ""), this.fragment !== a) {
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
    }), a.history = new C;
    var I = function(a, b) {
        var c, d = this;
        c = a && f.has(a, "constructor") ? a.constructor : function() {
            return d.apply(this, arguments)
        }, f.extend(c, d, b);
        var e = function() {
            this.constructor = c
        };
        return e.prototype = d.prototype, c.prototype = new e, a && f.extend(c.prototype, a), c.__super__ = d.prototype, c
    };
    l.extend = n.extend = x.extend = s.extend = C.extend = I;
    var J = function() {
        throw new Error('A "url" property or function must be specified')
    }, K = function(a, b) {
        var c = b.error;
        b.error = function(d) {
            c && c(a, d, b), a.trigger("error", a, d, b)
        }
    }
}.call(this), "undefined" !== typeof module && module.exports && (module.exports = printStackTrace), printStackTrace.implementation = function() {}, printStackTrace.implementation.prototype = {
    run: function(a, b) {
        return a = a || this.createException(), b = b || this.mode(a), "other" === b ? this.other(arguments.callee) : this[b](a)
    },
    createException: function() {
        try {
            this.undef()
        } catch (a) {
            return a
        }
    },
    mode: function(a) {
        return a.arguments && a.stack ? "chrome" : a.stack && a.sourceURL ? "safari" : a.stack && a.number ? "ie" : "string" === typeof a.message && "undefined" !== typeof window && window.opera ? a.stacktrace ? a.message.indexOf("\n")>-1 && a.message.split("\n").length > a.stacktrace.split("\n").length ? "opera9" : a.stack ? a.stacktrace.indexOf("called from line") < 0 ? "opera10b" : "opera11" : "opera10a" : "opera9" : a.stack ? "firefox" : "other"
    },
    instrumentFunction: function(a, b, c) {
        a = a || window;
        var d = a[b];
        a[b] = function() {
            return c.call(this, printStackTrace().slice(4)), a[b]._instrumented.apply(this, arguments)
        }, a[b]._instrumented = d
    },
    deinstrumentFunction: function(a, b) {
        a[b].constructor === Function && a[b]._instrumented && a[b]._instrumented.constructor === Function && (a[b] = a[b]._instrumented)
    },
    chrome: function(a) {
        var b = (a.stack + "\n").replace(/^\S[^\(]+?[\n$]/gm, "").replace(/^\s+(at eval )?at\s+/gm, "").replace(/^([^\(]+?)([\n$])/gm, "{anonymous}()@$1$2").replace(/^Object.<anonymous>\s*\(([^\)]+)\)/gm, "{anonymous}()@$1").split("\n");
        return b.pop(), b
    },
    safari: function(a) {
        return a.stack.replace(/\[native code\]\n/m, "").replace(/^(?=\w+Error\:).*$\n/m, "").replace(/^@/gm, "{anonymous}()@").split("\n")
    },
    ie: function(a) {
        var b = /^.*at (\w+) \(([^\)]+)\)$/gm;
        return a.stack.replace(/at Anonymous function /gm, "{anonymous}()@").replace(/^(?=\w+Error\:).*$\n/m, "").replace(b, "$1@$2").split("\n")
    },
    firefox: function(a) {
        return a.stack.replace(/(?:\n@:0)?\s+$/m, "").replace(/^[\(@]/gm, "{anonymous}()@").split("\n")
    },
    opera11: function(a) {
        for (var b = "{anonymous}", c = /^.*line (\d+), column (\d+)(?: in (.+))? in (\S+):$/, d = a.stacktrace.split("\n"), e = [], f = 0, g = d.length; f < g; f += 2) {
            var h = c.exec(d[f]);
            if (h) {
                var i = h[4] + ":" + h[1] + ":" + h[2], j = h[3] || "global code";
                j = j.replace(/<anonymous function: (\S+)>/, "$1").replace(/<anonymous function>/, b), e.push(j + "@" + i + " -- " + d[f + 1].replace(/^\s+/, ""))
            }
        }
        return e
    },
    opera10b: function(a) {
        for (var b = /^(.*)@(.+):(\d+)$/, c = a.stacktrace.split("\n"), d = [], e = 0, f = c.length; e < f; e++) {
            var g = b.exec(c[e]);
            if (g) {
                var h = g[1] ? g[1] + "()": "global code";
                d.push(h + "@" + g[2] + ":" + g[3])
            }
        }
        return d
    },
    opera10a: function(a) {
        for (var b = "{anonymous}", c = /Line (\d+).*script (?:in )?(\S+)(?:: In function (\S+))?$/i, d = a.stacktrace.split("\n"), e = [], f = 0, g = d.length; f < g; f += 2) {
            var h = c.exec(d[f]);
            if (h) {
                var i = h[3] || b;
                e.push(i + "()@" + h[2] + ":" + h[1] + " -- " + d[f + 1].replace(/^\s+/, ""))
            }
        }
        return e
    },
    opera9: function(a) {
        for (var b = "{anonymous}", c = /Line (\d+).*script (?:in )?(\S+)/i, d = a.message.split("\n"), e = [], f = 2, g = d.length; f < g; f += 2) {
            var h = c.exec(d[f]);
            h && e.push(b + "()@" + h[2] + ":" + h[1] + " -- " + d[f + 1].replace(/^\s+/, ""))
        }
        return e
    },
    other: function(a) {
        for (var b, c, d = "{anonymous}", e = /function\s*([\w\-$]+)?\s*\(/i, f = [], g = 10; a && a.arguments && f.length < g;)
            b = e.test(a.toString()) ? RegExp.$1 || d : d, c = Array.prototype.slice.call(a.arguments || []), f[f.length] = b + "(" + this.stringifyArguments(c) + ")", a = a.caller;
        return f
    },
    stringifyArguments: function(a) {
        for (var b = [], c = Array.prototype.slice, d = 0; d < a.length; ++d) {
            var e = a[d];
            void 0 === e ? b[d] = "undefined" : null === e ? b[d] = "null" : e.constructor && (e.constructor === Array ? b[d] = e.length < 3 ? "[" + this.stringifyArguments(e) + "]" : "[" + this.stringifyArguments(c.call(e, 0, 1)) + "..." + this.stringifyArguments(c.call(e, - 1)) + "]" : e.constructor === Object ? b[d] = "#object" : e.constructor === Function ? b[d] = "#function" : e.constructor === String ? b[d] = '"' + e + '"' : e.constructor === Number && (b[d] = e))
        }
        return b.join(",")
    },
    sourceCache: {},
    ajax: function(a) {
        var b = this.createXMLHTTPObject();
        if (b)
            try {
                return b.open("GET", a, !1), b.send(null), b.responseText
        } catch (c) {}
        return ""
    },
    createXMLHTTPObject: function() {
        for (var a, b = [function() {
            return new XMLHttpRequest
        }, function() {
            return new ActiveXObject("Msxml2.XMLHTTP")
        }, function() {
            return new ActiveXObject("Msxml3.XMLHTTP")
        }, function() {
            return new ActiveXObject("Microsoft.XMLHTTP")
        }
        ], c = 0; c < b.length; c++)
            try {
                return a = b[c](), this.createXMLHTTPObject = b[c], a
        } catch (d) {}
    },
    isSameDomain: function(a) {
        return "undefined" !== typeof location&&-1 !== a.indexOf(location.hostname)
    },
    getSource: function(a) {
        return a in this.sourceCache || (this.sourceCache[a] = this.ajax(a).split("\n")), this.sourceCache[a]
    },
    guessAnonymousFunctions: function(a) {
        for (var b = 0; b < a.length; ++b) {
            var c = /\{anonymous\}\(.*\)@(.*)/, d = /^(.*?)(?::(\d+))(?::(\d+))?(?: -- .+)?$/, e = a[b], f = c.exec(e);
            if (f) {
                var g = d.exec(f[1]);
                if (g) {
                    var h = g[1], i = g[2], j = g[3] || 0;
                    if (h && this.isSameDomain(h) && i) {
                        var k = this.guessAnonymousFunction(h, i, j);
                        a[b] = e.replace("{anonymous}", k)
                    }
                }
            }
        }
        return a
    },
    guessAnonymousFunction: function(a, b) {
        var c;
        try {
            c = this.findFunctionName(this.getSource(a), b)
        } catch (d) {
            c = "getSource failed with url: " + a + ", exception: " + d.toString()
        }
        return c
    },
    findFunctionName: function(a, b) {
        for (var c, d, e, f = /function\s+([^(]*?)\s*\(([^)]*)\)/, g = /['"]?([$_A-Za-z][$_A-Za-z0-9]*)['"]?\s*[:=]\s*function\b/, h = /['"]?([$_A-Za-z][$_A-Za-z0-9]*)['"]?\s*[:=]\s*(?:eval|new Function)\b/, i = "", j = Math.min(b, 20), k = 0; k < j; ++k)
            if (c = a[b - k - 1], e = c.indexOf("//"), e >= 0 && (c = c.substr(0, e)), c) {
                if (i = c + i, d = g.exec(i), d && d[1])
                    return d[1];
                    if (d = f.exec(i), d && d[1])
                        return d[1];
                        if (d = h.exec(i), d && d[1])
                            return d[1]
            }
        return "(?)"
    }
}, Cookie = {
    get: function(a) {
        return tmp = document.cookie.match(new RegExp("\\b" + a + "=.+?($|;)", "g")), tmp && tmp[0] ? unescape(tmp[0].substring(a.length + 1, tmp[0].length).replace(";", "")) || null : null
    },
    set: function(a, b, c, d, e) {
        return cookie = [a + "=" + escape(b), "path=" + (d && "" != d ? d : "/")], c = Cookie.hoursToExpireDate(c), c && cookie.push("expires=" + c), e && cookie.push("secure"), document.cookie = cookie.join("; ")
    },
    unset: function(a, b) {
        b = b && "string" == typeof b ? b : "", Cookie.get(a) && Cookie.set(a, "", - 100, b)
    },
    hoursToExpireDate: function(a) {
        return isNaN(parseInt(a)) ? "" : (now = new Date, now.setTime(now.getTime() + 60 * parseInt(a) * 60 * 1e3), now.toGMTString())
    },
    test: function() {
        return Cookie.set("b49f729efde9b2578ea9f00563d06e57", "true"), "true" == Cookie.get("b49f729efde9b2578ea9f00563d06e57") ? (Cookie.unset("b49f729efde9b2578ea9f00563d06e57"), !0) : !1
    },
    dump: function() {
        "undefined" != typeof console && console.log(document.cookie.split(";"))
    }
}, function() {
    !function(a) {
        return a(window.jQuery)
    }(function(a) {
        var b, c, d, e, f, g;
        return f = function() {
            function b(a) {
                this.$inputor = a
            }
            return b.prototype.css_attr = ["overflowY", "height", "width", "paddingTop", "paddingLeft", "paddingRight", "paddingBottom", "marginTop", "marginLeft", "marginRight", "marginBottom", "fontFamily", "borderStyle", "borderWidth", "wordWrap", "fontSize", "lineHeight", "overflowX", "text-align"], b.prototype.copy_inputor_css = function() {
                var b, c = this;
                return b = {
                    position: "absolute",
                    left: - 9999,
                    top: 0,
                    zIndex: - 2e4,
                    "white-space": "pre-wrap"
                }, a.each(this.css_attr, function(a, d) {
                    return b[d] = c.$inputor.css(d)
                }), b
            }, b.prototype.create = function(b) {
                return this.$mirror = a("<div></div>"), this.$mirror.css(this.copy_inputor_css()), this.$mirror.html(b), this.$inputor.after(this.$mirror), this
            }, b.prototype.get_flag_rect = function() {
                var a, b, c;
                return a = this.$mirror.find("span#flag"), b = a.position(), c = {
                    left: b.left,
                    top: b.top,
                    bottom: a.height() + b.top
                }, this.$mirror.remove(), c
            }, b
        }(), e = {
            DOWN: 40,
            UP: 38,
            ESC: 27,
            TAB: 9,
            ENTER: 13
        }, c = {
            data_refactor: function(b) {
                return a.isArray(b) ? a.map(b, function(b) {
                    return a.isPlainObject(b) || (b = {
                        name: b
                    }), b
                }) : b
            },
            matcher: function(a, b) {
                var c, d, e;
                return e = new RegExp(a + "([A-Za-z0-9_+-]*)$|" + a + "([^\\x00-\\xff]*)$", "gi"), c = e.exec(b), d = null, c && (d = c[2] ? c[2] : c[1]), d
            },
            filter: function(b, c, d) {
                return a.map(c, function(c) {
                    var e;
                    return e = a.isPlainObject(c) ? c[d] : c, e.toLowerCase().indexOf(b) >= 0 ? c : void 0
                })
            },
            remote_filter: function(b, c, d) {
                return a.ajax(c, {
                    data: b,
                    success: function(a) {
                        return d(a)
                    }
                })
            },
            sorter: function(a, b, c) {
                var d, e, f, g, h;
                if (!a)
                    return b.sort(function(a, b) {
                        return a[c].toLowerCase() > b[c].toLowerCase()
                    });
                for (e = [], g = 0, h = b.length; g < h; g++)
                    d = b[g], f = d[c], d.order = f.toLowerCase().indexOf(a), e.push(d);
                return e.sort(function(a, b) {
                    return a.order - b.order
                })
            },
            tpl_eval: function(a, b) {
                var c;
                try {
                    return c = a.replace(/\$\{([^\}]*)\}/g, function(a, c) {
                        return b[c]
                    })
                } catch (d) {
                    return ""
                }
            },
            highlighter: function(a, b) {
                return b ? a.replace(new RegExp(">\\s*(\\w*)(" + b.replace("+", "\\+") + ")(\\w*)\\s*<", "ig"), function(a, b, c, d) {
                    return "> " + b + "<strong>" + c + "</strong>" + d + " <"
                }) : a
            },
            selector: function(a) {
                return a.length > 0 ? this.replace_str(a.data("value") || "") : void 0
            }
        }, b = function() {
            function b(b) {
                this.settings = {}, this.common_settings = {}, this.pos = 0, this.flags = null, this.current_flag = null, this.query = null, this.$inputor = a(b), this.mirror = new f(this.$inputor), this.common_settings = a.extend({}, a.fn.atwho["default"]), this.view = new g(this, this.$el), this.listen()
            }
            return b.prototype.listen = function() {
                var a = this;
                return this.$inputor.on("keyup.atwho", function(b) {
                    return a.on_keyup(b)
                }).on("keydown.atwho", function(b) {
                    return a.on_keydown(b)
                }).on("scroll.atwho", function() {
                    return a.view.hide()
                }).on("blur.atwho", function() {
                    return a.view.hide(a.get_opt("display_timeout"))
                })
            }, b.prototype.reg = function(b, c) {
                var d, e;
                return d = {}, d = a.isPlainObject(b) ? this.common_settings = a.extend({}, this.common_settings, b) : this.settings[b] = this.settings[b] ? a.extend({}, this.settings[b], c) : a.extend({}, c), e = d.data, d.data = this.callbacks("data_refactor").call(this, e), this
            }, b.prototype.trigger = function(a, b) {
                return b || (b = []), b.push(this), this.$inputor.trigger("" + a + ".atwho", b)
            }, b.prototype.data = function() {
                return this.get_opt("data")
            }, b.prototype.callbacks = function(a) {
                var b;
                return (b = this.get_opt("callbacks", {})[a]) || (b = this.common_settings.callbacks[a]), b
            }, b.prototype.get_opt = function(a, b) {
                var c;
                try {
                    return this.current_flag && (c = this.settings[this.current_flag][a]), void 0 === c && (c = this.common_settings[a]), c = void 0 === c ? b : c
                } catch (d) {
                    return c = void 0 === b ? null : b
                }
            }, b.prototype.rect = function() {
                var b, c, d, e, f, g, h, i, j, k;
                return b = this.$inputor, document.selection ? (c = document.selection.createRange(), j = c.boundingLeft + b.scrollLeft(), k = c.boundingTop + a(window).scrollTop() + b.scrollTop(), e = k + c.boundingHeight, {
                    top: k - 2,
                    left: j - 2,
                    bottom: e - 2
                }) : (f = function(a) {
                    return a.replace(/</g, "&lt").replace(/>/g, "&gt").replace(/`/g, "&#96").replace(/"/g, "&quot").replace(/\r\n|\r|\n/g, "<br />")
                }, i = b.val().slice(0, this.pos - 1), g = "<span>" + f(i) + "</span>", g += "<span id='flag'>?</span>", h = b.offset(), d = this.mirror.create(g).get_flag_rect(), j = h.left + d.left - b.scrollLeft(), k = h.top - b.scrollTop(), e = k + d.bottom, k += d.top, {
                    top: k,
                    left: j,
                    bottom: e + 2
                })
            }, b.prototype.catch_query = function() {
                var b, c, d, e, f, g, h = this;
                return c = this.$inputor.val(), b = this.$inputor.caretPos(), g = c.slice(0, b), e = null, a.each(this.settings, function(a) {
                    return e = h.callbacks("matcher").call(h, a, g), null != e ? (h.current_flag = a, !1) : void 0
                }), "string" === typeof e && e.length <= 20 ? (f = b - e.length, d = f + e.length, this.pos = f, e = {
                    text: e.toLowerCase(),
                    head_pos: f,
                    end_pos: d
                }, this.trigger("matched", [this.current_flag, e.text])) : this.view.hide(), this.query = e
            }, b.prototype.replace_str = function(a) {
                var b, c, d, e, f;
                return b = this.$inputor, d = b.val(), c = this.get_opt("display_flag") ? 0 : this.current_flag.length, e = d.slice(0, (this.query.head_pos || 0) - c), f = "" + e + a + " " + d.slice(this.query.end_pos || 0), b.val(f), b.caretPos(e.length + a.length + 1), b.change()
            }, b.prototype.on_keyup = function(b) {
                switch (b.keyCode) {
                case e.ESC:
                    b.preventDefault(), this.view.hide();
                    break;
                case e.DOWN:
                case e.UP:
                    a.noop();
                    break;
                default:
                    this.look_up()
                }
                return b.stopPropagation()
            }, b.prototype.on_keydown = function(b) {
                if (this.view.visible()) {
                    switch (b.keyCode) {
                    case e.ESC:
                        b.preventDefault(), this.view.hide();
                        break;
                    case e.UP:
                        b.preventDefault(), this.view.prev();
                        break;
                    case e.DOWN:
                        b.preventDefault(), this.view.next();
                        break;
                    case e.TAB:
                    case e.ENTER:
                        if (!this.view.visible())
                            return;
                        b.preventDefault(), this.view.choose();
                        break;
                    default:
                        a.noop()
                    }
                    return b.stopPropagation()
                }
            }, b.prototype.render_view = function(a) {
                var b;
                return b = this.get_opt("search_key"), a = this.callbacks("sorter").call(this, this.query.text, a, b), a = a.splice(0, this.get_opt("limit")), this.view.render(a)
            }, b.prototype.remote_call = function(b, c) {
                var d, e;
                return d = {
                    q: c.text,
                    limit: this.get_opt("limit")
                }, e = function(a) {
                    return this.reg(this.current_flag, {
                        data: a
                    }), this.render_view(this.data())
                }, e = a.proxy(e, this), this.callbacks("remote_filter").call(this, d, b, e)
            }, b.prototype.look_up = function() {
                var b, c, d;
                return (c = this.catch_query()) ? (b = this.data(), d = this.get_opt("search_key"), "string" === typeof b ? this.remote_call(b, c) : (b = this.callbacks("filter").call(this, c.text, b, d)) ? this.render_view(b) : this.view.hide(), a.noop()) : !1
            }, b
        }(), g = function() {
            function b(b) {
                this.controller = b, this.id = this.controller.get_opt("view_id", "at-view"), this.timeout_id = null, this.$el = a("#" + this.id), this.create_view()
            }
            return b.prototype.create_view = function() {
                var b, c, d = this;
                if (!this.exist())
                    return c = "<div id='" + this.id + "' class='at-view'><ul id='" + this.id + "-ul'></ul></div>", a("body").append(c), this.$el = a("#" + this.id), b = this.$el.find("ul"), b.on("mouseenter.view", "li", function(c) {
                        return b.find(".cur").removeClass("cur"), a(c.currentTarget).addClass("cur")
                    }).on("click", function(a) {
                        return a.stopPropagation(), a.preventDefault(), d.$el.data("_view").choose()
                    })
            }, b.prototype.exist = function() {
                return a("#" + this.id).length > 0
            }, b.prototype.visible = function() {
                return this.$el.is(":visible")
            }, b.prototype.choose = function() {
                var a;
                return a = this.$el.find(".cur"), this.controller.callbacks("selector").call(this.controller, a), this.controller.trigger("choose", [a]), this.hide()
            }, b.prototype.reposition = function() {
                var b, c;
                return c = this.controller.rect(), c.bottom + this.$el.height() - a(window).scrollTop() > a(window).height() && (c.bottom = c.top - this.$el.height()), b = {
                    left: c.left,
                    top: c.bottom
                }, this.$el.offset(b), this.controller.trigger("reposition", [b])
            }, b.prototype.next = function() {
                var b, c;
                return b = this.$el.find(".cur").removeClass("cur"), c = b.next(), c.length || (c = a(this.$el.find("li")[0])), c.addClass("cur")
            }, b.prototype.prev = function() {
                var a, b;
                return a = this.$el.find(".cur").removeClass("cur"), b = a.prev(), b.length || (b = this.$el.find("li").last()), b.addClass("cur")
            }, b.prototype.show = function() {
                return this.visible() || this.$el.show(), this.reposition()
            }, b.prototype.hide = function(a) {
                var b, c = this;
                return isNaN(a) ? this.visible() ? this.$el.hide() : void 0 : (b = function() {
                    return c.hide()
                }, clearTimeout(this.timeout_id), this.timeout_id = setTimeout(b, a))
            }, b.prototype.clear = function() {
                return this.$el.find("ul").empty()
            }, b.prototype.render = function(b) {
                var c, e, f = this;
                return a.isArray(b) ? b.length <= 0 ? (this.hide(), !0) : (this.clear(), this.$el.data("_view", this), c = this.$el.find("ul"), e = this.controller.get_opt("tpl", d), a.each(b, function(b, d) {
                    var g, h;
                    return h = f.controller.callbacks("tpl_eval").call(f.controller, e, d), g = a(f.controller.callbacks("highlighter").call(f.controller, h, f.controller.query.text)), g.data("info", d), c.append(g)
                }), this.show(), c.find("li:eq(0)").addClass("cur")) : !1
            }, b
        }(), d = "<li data-value='${name}'>${name}</li>", a.fn.atwho = function(c, d) {
            return this.filter("textarea, input").each(function() {
                var e, f;
                return e = a(this), f = e.data("atwho"), f || e.data("atwho", f = new b(this)), f.reg(c, d)
            })
        }, a.fn.atwho.Controller = b, a.fn.atwho.View = g, a.fn.atwho.Mirror = f, a.fn.atwho["default"] = {
            data: null,
            search_key: "name",
            callbacks: c,
            limit: 5,
            display_flag: !0,
            display_timeout: 300,
            tpl: d
        }
    })
}.call(this), function() {
    !function(a) {
        return a(window.jQuery)
    }(function(a) {
        var b, c;
        return b = function(a) {
            var b, c, d, e, f, g, h, i;
            return document.selection ? (g = document.selection.createRange(), f = 0, g && g.parentElement() === a && (e = a.value.replace(/\r\n/g, "\n"), d = e.length, i = a.createTextRange(), i.moveToBookmark(g.getBookmark()), c = a.createTextRange(), c.collapse(!1), i.compareEndPoints("StartToEnd", c)>-1 ? h = b = d : (h =- i.moveStart("character", - d), b =- i.moveEnd("character", - d)))) : h = a.selectionStart, h
        }, c = function(a, b) {
            var c;
            return document.selection ? (c = a.createTextRange(), c.move("character", b), c.select()) : a.setSelectionRange(b, b)
        }, a.fn.caretPos = function(a) {
            var d;
            return d = this[0], d.focus(), a ? c(d, a) : b(d)
        }
    })
}.call(this);
var window_lib = window.jQuery ? jQuery: Zepto;
!function(a) {
    a.fn.passStrength = function(b, c, d, e) {
        return this.each(function() {
            var f = a(this);
            a(f).unbind().bind("keyup blur", function() {
                if (null !== a(this).val()) {
                    var f = a.fn.teststrength(a(this).val(), e, d);
                    f[2] ? (b.attr("disabled", "disabled"), b.addClass("disabled")) : (b.removeAttr("disabled"), b.removeClass("disabled")), c.html(f[0]), c.removeClass(e.badPassStyle), c.removeClass(e.goodPassStyle), c.removeClass(e.strongPassStyle), c.addClass(f[1])
                }
            })
        })
    }, a.fn.teststrength = function(b, c, d) {
        var e = 0;
        return b.length < 6 ? [c.shortPassStr, c.badPassStyle, 1] : c.userid.length > 0 && b.toLowerCase() == c.userid.toLowerCase() ? [c.samePasswordStr, c.badPassStyle, 1] : a.inArray(b.toLowerCase(), d)>-1 ? [c.blackPassStr, c.badPassStyle, 1] : (e += 2 * b.length, e += 3 * a.fn.uniqueCharacters(b), b.match(/(.*[0-9].*[0-9].*[0-9])/) && (e += 15), b.match(/(.*[!,@,#,$,%,\^,&,*,?,_,~].*[!,@,#,$,%,\^,&,*,?,_,~])/) && (e += 15), b.match(/([a-z].*[A-Z])|([A-Z].*[a-z])/) && (e += 10), b.match(/([a-zA-Z])/) && b.match(/([0-9])/) && (e += 15), b.match(/([!,@,#,$,%,\^,&,*,?,_,~])/) && b.match(/([0-9])/) && (e += 15), b.match(/([!,@,#,$,%,\^,&,*,?,_,~])/) && b.match(/([a-zA-Z])/) && (e += 15), (b.match(/^[a-z]+$/) || b.match(/^[A-Z]+$/) || b.match(/^\d+$/)) && (e -= 10), e < 34 ? [c.badPassStr, c.badPassStyle, 0] : e < 68 ? [c.goodPassStr, c.goodPassStyle, 0] : [c.strongPassStr, c.strongPassStyle, 0])
    }
}(window_lib), $.fn.uniqueCharacters = function(a) {
    for (var b = {}, c = 0, d = 0; d < a.length; d++)
        a.charAt(d)in b || (b[a.charAt(d)]=!0, c++);
    return c
}, window.location.origin || (window.location.origin = window.location.protocol + "//" + window.location.hostname + (window.location.port ? ":" + window.location.port : "")), function(a, b) {
    function c(a, b, c) {
        var d = a + ":", e = document.createElement("test"), f = e.style;
        return f.cssText = c ? d + b : d + ["-webkit-", "-moz-", "-ms-", "-o-", ""].join(b + ";" + d) + b + ";", - 1 !== f[a].indexOf(b)
    }
    function d(a) {
        return parseInt(a, 10) || 0
    }
    var e = {
        classes: {
            plugin: "fixedsticky",
            active: "fixedsticky-on",
            inactive: "fixedsticky-off",
            clone: "fixedsticky-dummy",
            withoutFixedFixed: "fixedsticky-withoutfixedfixed"
        },
        keys: {
            offset: "fixedStickyOffset",
            position: "fixedStickyPosition"
        },
        tests: {
            sticky: c("position", "sticky"),
            fixed: c("position", "fixed", !0)
        },
        getScrollTop: function() {
            var b = "pageYOffset", c = "scrollTop";
            return a ? b in a ? a[b] : a.document.documentElement[c] : a.document.body[c]
        },
        bypass: function() {
            return e.tests.sticky&&!e.optOut ||!e.tests.fixed || a.FixedFixed&&!b(a.document.documentElement).hasClass("fixed-supported")
        },
        update: function(a) {
            function c() {
                var a = m + h;
                return l < a && a + k <= s + t
            }
            function f() {
                return l + (k || 0) > m + p - i && m + p - i >= s + (k || 0)
            }
            if (a.offsetWidth) {
                var g, h, i, j = b(a), k = j.outerHeight(), l = j.data(e.keys.offset), m = e.getScrollTop(), n = j.is("." + e.classes.active), o = function(a) {
                    j[a ? "addClass": "removeClass"](e.classes.active)[a ? "removeClass": "addClass"](e.classes.inactive)
                }, p = b(window).height(), q = j.data(e.keys.position), r = j.parent(), s = r.offset().top, t = r.outerHeight();
                l || (l = j.offset().top, j.data(e.keys.offset, l), j.after(b("<div>").addClass(e.classes.clone).height(k))), q || (g = "auto" !== j.css("top") || "auto" !== j.css("bottom"), g || j.css("position", "fixed"), q = {
                    top: "auto" !== j.css("top"),
                    bottom: "auto" !== j.css("bottom")
                }, g || j.css("position", ""), j.data(e.keys.position, q)), h = d(j.css("top")), i = d(j.css("bottom")), q.top && c() || q.bottom && f() ? n || o(!0) : n && o(!1)
            }
        },
        destroy: function(c) {
            var d = b(c);
            if (!e.bypass())
                return b(a).unbind(".fixedsticky"), d.each(function() {
                    b(this).removeData([e.keys.offset, e.keys.position]).removeClass(e.classes.active).removeClass(e.classes.inactive).next("." + e.classes.clone).remove()
                })
        },
        init: function(c) {
            var d = b(c);
            if (!e.bypass())
                return d.each(function() {
                    var c = this;
                    b(a).bind("scroll.fixedsticky", function() {
                        e.update(c)
                    }), e.update(this), b(a).bind("resize.fixedsticky", function() {
                        d.is("." + e.classes.active) && e.update(c)
                    })
                })
        }
    };
    a.FixedSticky = e, b.fn.fixedsticky = function(a) {
        if ("function" === typeof e[a])
            return e[a].call(e, this);
        if ("object" !== typeof a && a)
            throw new Error("Method `" + a + "` does not exist on jQuery.fixedsticky");
        return e.init.call(e, this)
    }, a.FixedFixed || b(a.document.documentElement).addClass(e.classes.withoutFixedFixed)
}(this, jQuery), FixedSticky.tests.sticky=!1;
var deviceIsAndroid = navigator.userAgent.indexOf("Android") > 0, deviceIsIOS = /iP(ad|hone|od)/.test(navigator.userAgent), deviceIsIOS4 = deviceIsIOS && /OS 4_\d(_\d)?/.test(navigator.userAgent), deviceIsIOSWithBadTarget = deviceIsIOS && /OS ([6-9]|\d{2})_\d/.test(navigator.userAgent), deviceIsBlackBerry10 = navigator.userAgent.indexOf("BB10") > 0;
FastClick.prototype.needsClick = function(a) {
    "use strict";
    switch (a.nodeName.toLowerCase()) {
    case"button":
    case"select":
    case"textarea":
        if (a.disabled)
            return !0;
        break;
    case"input":
        if (deviceIsIOS && "file" === a.type || a.disabled)
            return !0;
        break;
    case"label":
    case"video":
        return !0
    }
    return /\bneedsclick\b/.test(a.className)
}, FastClick.prototype.needsFocus = function(a) {
    "use strict";
    switch (a.nodeName.toLowerCase()) {
    case"textarea":
        return !0;
    case"select":
        return !deviceIsAndroid;
    case"input":
        switch (a.type) {
        case"button":
        case"checkbox":
        case"file":
        case"image":
        case"radio":
        case"submit":
            return !1
        }
        return !a.disabled&&!a.readOnly;
    default:
        return /\bneedsfocus\b/.test(a.className)
    }
}, FastClick.prototype.sendClick = function(a, b) {
    "use strict";
    var c, d;
    document.activeElement && document.activeElement !== a && document.activeElement.blur(), d = b.changedTouches[0], c = document.createEvent("MouseEvents"), c.initMouseEvent(this.determineEventType(a), !0, !0, window, 1, d.screenX, d.screenY, d.clientX, d.clientY, !1, !1, !1, !1, 0, null), c.forwardedTouchEvent=!0, a.dispatchEvent(c)
}, FastClick.prototype.determineEventType = function(a) {
    "use strict";
    return deviceIsAndroid && "select" === a.tagName.toLowerCase() ? "mousedown" : "click"
}, FastClick.prototype.focus = function(a) {
    "use strict";
    var b;
    deviceIsIOS && a.setSelectionRange && 0 !== a.type.indexOf("date") && "time" !== a.type ? (b = a.value.length, a.setSelectionRange(b, b)) : a.focus()
}, FastClick.prototype.updateScrollParent = function(a) {
    "use strict";
    var b, c;
    if (b = a.fastClickScrollParent, !b ||!b.contains(a)) {
        c = a;
        do {
            if (c.scrollHeight > c.offsetHeight) {
                b = c, a.fastClickScrollParent = c;
                break
            }
            c = c.parentElement
        }
        while (c)
        }
    b && (b.fastClickLastScrollTop = b.scrollTop)
}, FastClick.prototype.getTargetElementFromEventTarget = function(a) {
    "use strict";
    return a.nodeType === Node.TEXT_NODE ? a.parentNode : a
}, FastClick.prototype.onTouchStart = function(a) {
    "use strict";
    var b, c, d;
    if (a.targetTouches.length > 1)
        return !0;
    if (b = this.getTargetElementFromEventTarget(a.target), c = a.targetTouches[0], deviceIsIOS) {
        if (d = window.getSelection(), d.rangeCount&&!d.isCollapsed)
            return !0;
        if (!deviceIsIOS4) {
            if (c.identifier === this.lastTouchIdentifier)
                return a.preventDefault(), !1;
            this.lastTouchIdentifier = c.identifier, this.updateScrollParent(b)
        }
    }
    return this.trackingClick=!0, this.trackingClickStart = a.timeStamp, this.targetElement = b, this.touchStartX = c.pageX, this.touchStartY = c.pageY, a.timeStamp - this.lastClickTime < this.tapDelay && a.preventDefault(), !0
}, FastClick.prototype.touchHasMoved = function(a) {
    "use strict";
    var b = a.changedTouches[0], c = this.touchBoundary;
    return Math.abs(b.pageX - this.touchStartX) > c || Math.abs(b.pageY - this.touchStartY) > c?!0 : !1
}, FastClick.prototype.onTouchMove = function(a) {
    "use strict";
    return this.trackingClick ? ((this.targetElement !== this.getTargetElementFromEventTarget(a.target) || this.touchHasMoved(a)) && (this.trackingClick=!1, this.targetElement = null), !0) : !0
}, FastClick.prototype.findControl = function(a) {
    "use strict";
    return void 0 !== a.control ? a.control : a.htmlFor ? document.getElementById(a.htmlFor) : a.querySelector("button, input:not([type=hidden]), keygen, meter, output, progress, select, textarea")
}, FastClick.prototype.onTouchEnd = function(a) {
    "use strict";
    var b, c, d, e, f, g = this.targetElement;
    if (!this.trackingClick)
        return !0;
    if (a.timeStamp - this.lastClickTime < this.tapDelay)
        return this.cancelNextClick=!0, !0;
    if (this.cancelNextClick=!1, this.lastClickTime = a.timeStamp, c = this.trackingClickStart, this.trackingClick=!1, this.trackingClickStart = 0, deviceIsIOSWithBadTarget && (f = a.changedTouches[0], g = document.elementFromPoint(f.pageX - window.pageXOffset, f.pageY - window.pageYOffset) || g, g.fastClickScrollParent = this.targetElement.fastClickScrollParent), d = g.tagName.toLowerCase(), "label" === d) {
        if (b = this.findControl(g)) {
            if (this.focus(g), deviceIsAndroid)
                return !1;
            g = b
        }
    } else if (this.needsFocus(g))
        return a.timeStamp - c > 100 || deviceIsIOS && window.top !== window && "input" === d ? (this.targetElement = null, !1) : (this.focus(g), this.sendClick(g, a), deviceIsIOS && "select" === d || (this.targetElement = null, a.preventDefault()), !1);
    return deviceIsIOS&&!deviceIsIOS4 && (e = g.fastClickScrollParent, e && e.fastClickLastScrollTop !== e.scrollTop)?!0 : (this.needsClick(g) || (a.preventDefault(), this.sendClick(g, a)), !1)
}, FastClick.prototype.onTouchCancel = function() {
    "use strict";
    this.trackingClick=!1, this.targetElement = null
}, FastClick.prototype.onMouse = function(a) {
    "use strict";
    return this.targetElement ? a.forwardedTouchEvent?!0 : a.cancelable && (!this.needsClick(this.targetElement) || this.cancelNextClick) ? (a.stopImmediatePropagation ? a.stopImmediatePropagation() : a.propagationStopped=!0, a.stopPropagation(), a.preventDefault(), !1) : !0 : !0
}, FastClick.prototype.onClick = function(a) {
    "use strict";
    var b;
    return this.trackingClick ? (this.targetElement = null, this.trackingClick=!1, !0) : "submit" === a.target.type && 0 === a.detail?!0 : (b = this.onMouse(a), b || (this.targetElement = null), b)
}, FastClick.prototype.destroy = function() {
    "use strict";
    var a = this.layer;
    deviceIsAndroid && (a.removeEventListener("mouseover", this.onMouse, !0), a.removeEventListener("mousedown", this.onMouse, !0), a.removeEventListener("mouseup", this.onMouse, !0)), a.removeEventListener("click", this.onClick, !0), a.removeEventListener("touchstart", this.onTouchStart, !1), a.removeEventListener("touchmove", this.onTouchMove, !1), a.removeEventListener("touchend", this.onTouchEnd, !1), a.removeEventListener("touchcancel", this.onTouchCancel, !1)
}, FastClick.notNeeded = function(a) {
    "use strict";
    var b, c, d;
    if ("undefined" === typeof window.ontouchstart)
        return !0;
    if (c =+ (/Chrome\/([0-9]+)/.exec(navigator.userAgent) || [, 0])[1]) {
        if (!deviceIsAndroid)
            return !0;
        if (b = document.querySelector("meta[name=viewport]")) {
            if ( - 1 !== b.content.indexOf("user-scalable=no"))
                return !0;
            if (c > 31 && document.documentElement.scrollWidth <= window.outerWidth)
                return !0
        }
    }
    if (deviceIsBlackBerry10 && (d = navigator.userAgent.match(/Version\/([0-9]*)\.([0-9]*)/), d[1] >= 10 && d[2] >= 3 && (b = document.querySelector("meta[name=viewport]")))) {
        if ( - 1 !== b.content.indexOf("user-scalable=no"))
            return !0;
        if (document.documentElement.scrollWidth <= window.outerWidth)
            return !0
    }
    return "none" === a.style.msTouchAction?!0 : !1
}, FastClick.attach = function(a, b) {
    "use strict";
    return new FastClick(a, b)
}, "function" == typeof define && "object" == typeof define.amd && define.amd ? define(function() {
    "use strict";
    return FastClick
}) : "undefined" !== typeof module && module.exports ? (module.exports = FastClick.attach, module.exports.FastClick = FastClick) : window.FastClick = FastClick, function() {
    var a, b, c, d;
    !function() {
        var e = {}, f = {};
        a = function(a, b, c) {
            e[a] = {
                deps: b,
                callback: c
            }
        }, d = c = b = function(a) {
            function c(b) {
                if ("." !== b.charAt(0))
                    return b;
                for (var c = b.split("/"), d = a.split("/").slice(0, - 1), e = 0, f = c.length; e < f; e++) {
                    var g = c[e];
                    if (".." === g)
                        d.pop();
                    else {
                        if ("." === g)
                            continue;
                        d.push(g)
                    }
                }
                return d.join("/")
            }
            if (d._eak_seen = e, f[a])
                return f[a];
            if (f[a] = {}, !e[a])
                throw new Error("Could not find module " + a);
            for (var g, h = e[a], i = h.deps, j = h.callback, k = [], l = 0, m = i.length; l < m; l++)
                k.push("exports" === i[l] ? g = {} : b(c(i[l])));
            var n = j.apply(this, k);
            return f[a] = g || n
        }
    }(), a("promise/all", ["./utils", "exports"], function(a, b) {
        "use strict";
        function c(a) {
            var b = this;
            if (!d(a))
                throw new TypeError("You must pass an array to all.");
            return new b(function(b, c) {
                function d(a) {
                    return function(b) {
                        f(a, b)
                    }
                }
                function f(a, c) {
                    h[a] = c, 0===--i && b(h)
                }
                var g, h = [], i = a.length;
                0 === i && b([]);
                for (var j = 0; j < a.length; j++)
                    g = a[j], g && e(g.then) ? g.then(d(j), c) : f(j, g)
            })
        }
        var d = a.isArray, e = a.isFunction;
        b.all = c
    }), a("promise/asap", ["exports"], function(a) {
        "use strict";
        function b() {
            return function() {
                process.nextTick(e)
            }
        }
        function c() {
            var a = 0, b = new i(e), c = document.createTextNode("");
            return b.observe(c, {
                characterData: !0
            }), function() {
                c.data = a=++a%2
            }
        }
        function d() {
            return function() {
                j.setTimeout(e, 1)
            }
        }
        function e() {
            for (var a = 0; a < k.length; a++) {
                var b = k[a], c = b[0], d = b[1];
                c(d)
            }
            k = []
        }
        function f(a, b) {
            var c = k.push([a, b]);
            1 === c && g()
        }
        var g, h = "undefined" !== typeof window ? window: {}, i = h.MutationObserver || h.WebKitMutationObserver, j = "undefined" !== typeof global ? global: void 0 === this ? window : this, k = [];
        g = "undefined" !== typeof process && "[object process]" === {}.toString.call(process) ? b() : i ? c() : d(), a.asap = f
    }), a("promise/config", ["exports"], function(a) {
        "use strict";
        function b(a, b) {
            return 2 !== arguments.length ? c[a] : void(c[a] = b)
        }
        var c = {
            instrument: !1
        };
        a.config = c, a.configure = b
    }), a("promise/polyfill", ["./promise", "./utils", "exports"], function(a, b, c) {
        "use strict";
        function d() {
            var a;
            a = "undefined" !== typeof global ? global : "undefined" !== typeof window && window.document ? window : self;
            var b = "Promise"in a && "resolve"in a.Promise && "reject"in a.Promise && "all"in a.Promise && "race"in a.Promise && function() {
                var b;
                return new a.Promise(function(a) {
                    b = a
                }), f(b)
            }();
            b || (a.Promise = e)
        }
        var e = a.Promise, f = b.isFunction;
        c.polyfill = d
    }), a("promise/promise", ["./config", "./utils", "./all", "./race", "./resolve", "./reject", "./asap", "exports"], function(a, b, c, d, e, f, g, h) {
        "use strict";
        function i(a) {
            if (!v(a))
                throw new TypeError("You must pass a resolver function as the first argument to the promise constructor");
            if (!(this instanceof i))
                throw new TypeError("Failed to construct 'Promise': Please use the 'new' operator, this object constructor cannot be called as a function.");
            this._subscribers = [], j(a, this)
        }
        function j(a, b) {
            function c(a) {
                o(b, a)
            }
            function d(a) {
                q(b, a)
            }
            try {
                a(c, d)
            } catch (e) {
                d(e)
            }
        }
        function k(a, b, c, d) {
            var e, f, g, h, i = v(c);
            if (i)
                try {
                    e = c(d), g=!0
            } catch (j) {
                h=!0, f = j
            } else 
                e = d, g=!0;
            n(b, e) || (i && g ? o(b, e) : h ? q(b, f) : a === D ? o(b, e) : a === E && q(b, e))
        }
        function l(a, b, c, d) {
            var e = a._subscribers, f = e.length;
            e[f] = b, e[f + D] = c, e[f + E] = d
        }
        function m(a, b) {
            for (var c, d, e = a._subscribers, f = a._detail, g = 0; g < e.length; g += 3)
                c = e[g], d = e[g + b], k(b, c, d, f);
            a._subscribers = null
        }
        function n(a, b) {
            var c, d = null;
            try {
                if (a === b)
                    throw new TypeError("A promises callback cannot return that same promise.");
                if (u(b) && (d = b.then, v(d)))
                    return d.call(b, function(d) {
                        return c?!0 : (c=!0, void(b !== d ? o(a, d) : p(a, d)))
                    }, function(b) {
                        return c?!0 : (c=!0, void q(a, b))
                    }), !0
            } catch (e) {
                return c?!0 : (q(a, e), !0)
            }
            return !1
        }
        function o(a, b) {
            a === b ? p(a, b) : n(a, b) || p(a, b)
        }
        function p(a, b) {
            a._state === B && (a._state = C, a._detail = b, t.async(r, a))
        }
        function q(a, b) {
            a._state === B && (a._state = C, a._detail = b, t.async(s, a))
        }
        function r(a) {
            m(a, a._state = D)
        }
        function s(a) {
            m(a, a._state = E)
        }
        var t = a.config, u = (a.configure, b.objectOrFunction), v = b.isFunction, w = (b.now, c.all), x = d.race, y = e.resolve, z = f.reject, A = g.asap;
        t.async = A;
        var B = void 0, C = 0, D = 1, E = 2;
        i.prototype = {
            constructor: i,
            _state: void 0,
            _detail: void 0,
            _subscribers: void 0,
            then: function(a,
            b) {
                var c = this,
                d = new this.constructor(function() {});
                if (this._state) {
                    var e = arguments;
                    t.async(function() {
                        k(c._state, d, e[c._state - 1], c._detail)
                    })
                } else 
                    l(this, d, a, b);
                return d
            }, "catch" : function(a) {
                return this.then(null, a)
            }
        }, i.all = w, i.race = x, i.resolve = y, i.reject = z, h.Promise = i
    }), a("promise/race", ["./utils", "exports"], function(a, b) {
        "use strict";
        function c(a) {
            var b = this;
            if (!d(a))
                throw new TypeError("You must pass an array to race.");
            return new b(function(b, c) {
                for (var d, e = 0; e < a.length; e++)
                    d = a[e], d && "function" === typeof d.then ? d.then(b, c) : b(d)
            })
        }
        var d = a.isArray;
        b.race = c
    }), a("promise/reject", ["exports"], function(a) {
        "use strict";
        function b(a) {
            var b = this;
            return new b(function(b, c) {
                c(a)
            })
        }
        a.reject = b
    }), a("promise/resolve", ["exports"], function(a) {
        "use strict";
        function b(a) {
            if (a && "object" === typeof a && a.constructor === this)
                return a;
            var b = this;
            return new b(function(b) {
                b(a)
            })
        }
        a.resolve = b
    }), a("promise/utils", ["exports"], function(a) {
        "use strict";
        function b(a) {
            return c(a) || "object" === typeof a && null !== a
        }
        function c(a) {
            return "function" === typeof a
        }
        function d(a) {
            return "[object Array]" === Object.prototype.toString.call(a)
        }
        var e = Date.now || function() {
            return (new Date).getTime()
        };
        a.objectOrFunction = b, a.isFunction = c, a.isArray = d, a.now = e
    }), b("promise/polyfill").polyfill()
}(), function(a, b) {
    "function" === typeof define && define.amd ? define(b) : "object" === typeof exports ? module.exports = b() : a.returnExports = b()
}(this, function() {
    function a(a) {
        return a =+ a, a !== a ? a = 0 : 0 !== a && a !== 1 / 0 && a!==-(1 / 0) && (a = (a > 0||-1) * Math.floor(Math.abs(a))), a
    }
    function b(a) {
        var b = typeof a;
        return null === a || "undefined" === b || "boolean" === b || "number" === b || "string" === b
    }
    function c(a) {
        var c, d, e;
        if (b(a))
            return a;
        if (d = a.valueOf, p(d) && (c = d.call(a), b(c)))
            return c;
        if (e = a.toString, p(e) && (c = e.call(a), b(c)))
            return c;
        throw new TypeError
    }
    function d() {}
    var e, f = Array.prototype, g = Object.prototype, h = Function.prototype, i = String.prototype, j = Number.prototype, k = f.slice, l = f.splice, m = (f.push, f.unshift), n = h.call, o = g.toString, p = function(a) {
        return "[object Function]" === g.toString.call(a)
    }, q = function(a) {
        return "[object RegExp]" === g.toString.call(a)
    }, r = function(a) {
        return "[object Array]" === o.call(a)
    }, s = function(a) {
        return "[object String]" === o.call(a)
    }, t = function(a) {
        var b = o.call(a), c = "[object Arguments]" === b;
        return c || (c=!r(a) && null !== a && "object" === typeof a && "number" === typeof a.length && a.length >= 0 && p(a.callee)), c
    }, u = Object.defineProperty && function() {
        try {
            return Object.defineProperty({}, "x", {}), !0
        } catch (a) {
            return !1
        }
    }();
    e = u ? function(a, b, c, d) {
        !d && b in a || Object.defineProperty(a, b, {
            configurable: !0,
            enumerable: !1,
            writable: !0,
            value: c
        })
    } : function(a, b, c, d) {
        !d && b in a || (a[b] = c)
    };
    var v = function(a, b, c) {
        for (var d in b)
            g.hasOwnProperty.call(b, d) && e(a, d, b[d], c)
    }, w = function(a) {
        if (null == a)
            throw new TypeError("can't convert " + a + " to object");
        return Object(a)
    }, x = function(a) {
        return a>>>0
    };
    v(h, {
        bind: function(a) {
            var b = this;
            if (!p(b))
                throw new TypeError("Function.prototype.bind called on incompatible " + b);
            for (var c = k.call(arguments, 1), e = function() {
                if (this instanceof i) {
                    var d = b.apply(this, c.concat(k.call(arguments)));
                    return Object(d) === d ? d : this
                }
                return b.apply(a, c.concat(k.call(arguments)))
            }, f = Math.max(0, b.length - c.length), g = [], h = 0; h < f; h++)
                g.push("$" + h);
            var i = Function("binder", "return function (" + g.join(",") + "){return binder.apply(this,arguments)}")(e);
            return b.prototype && (d.prototype = b.prototype, i.prototype = new d, d.prototype = null), i
        }
    });
    var y, z, A, B, C, D = n.bind(g.hasOwnProperty);
    (C = D(g, "__defineGetter__")) && (y = n.bind(g.__defineGetter__), z = n.bind(g.__defineSetter__), A = n.bind(g.__lookupGetter__), B = n.bind(g.__lookupSetter__));
    var E = function() {
        var a = [1, 2], b = a.splice();
        return 2 === a.length && r(b) && 0 === b.length
    }();
    v(f, {
        splice: function() {
            return 0 === arguments.length ? [] : l.apply(this, arguments)
        }
    }, E);
    var F = function() {
        var a = {};
        return f.splice.call(a, 0, 0, 1), 1 === a.length
    }();
    v(f, {
        splice: function(b, c) {
            if (0 === arguments.length)
                return [];
            var d = arguments;
            return this.length = Math.max(a(this.length), 0), arguments.length > 0 && "number" !== typeof c && (d = k.call(arguments), d.length < 2 ? d.push(this.length - b) : d[1] = a(c)), l.apply(this, d)
        }
    }, !F);
    var G = 1 !== [].unshift(0);
    v(f, {
        unshift: function() {
            return m.apply(this, arguments), this.length
        }
    }, G), v(Array, {
        isArray: r
    });
    var H = Object("a"), I = "a" !== H[0] ||!(0 in H), J = function(a) {
        var b=!0, c=!0;
        return a && (a.call("foo", function(a, c, d) {
            "object" !== typeof d && (b=!1)
        }), a.call([1], function() {
            "use strict";
            c = "string" === typeof this
        }, "x")), !!a && b && c
    };
    v(f, {
        forEach: function(a) {
            var b = w(this), c = I && s(this) ? this.split(""): b, d = arguments[1], e =- 1, f = c.length>>>0;
            if (!p(a))
                throw new TypeError;
            for (; ++e < f;)
                e in c && a.call(d, c[e], e, b)
        }
    }, !J(f.forEach)), v(f, {
        map: function(a) {
            var b = w(this), c = I && s(this) ? this.split(""): b, d = c.length>>>0, e = Array(d), f = arguments[1];
            if (!p(a))
                throw new TypeError(a + " is not a function");
            for (var g = 0; g < d; g++)
                g in c && (e[g] = a.call(f, c[g], g, b));
            return e
        }
    }, !J(f.map)), v(f, {
        filter: function(a) {
            var b, c = w(this), d = I && s(this) ? this.split(""): c, e = d.length>>>0, f = [], g = arguments[1];
            if (!p(a))
                throw new TypeError(a + " is not a function");
            for (var h = 0; h < e; h++)
                h in d && (b = d[h], a.call(g, b, h, c) && f.push(b));
            return f
        }
    }, !J(f.filter)), v(f, {
        every: function(a) {
            var b = w(this), c = I && s(this) ? this.split(""): b, d = c.length>>>0, e = arguments[1];
            if (!p(a))
                throw new TypeError(a + " is not a function");
            for (var f = 0; f < d; f++)
                if (f in c&&!a.call(e, c[f], f, b))
                    return !1;
            return !0
        }
    }, !J(f.every)), v(f, {
        some: function(a) {
            var b = w(this), c = I && s(this) ? this.split(""): b, d = c.length>>>0, e = arguments[1];
            if (!p(a))
                throw new TypeError(a + " is not a function");
            for (var f = 0; f < d; f++)
                if (f in c && a.call(e, c[f], f, b))
                    return !0;
            return !1
        }
    }, !J(f.some));
    var K=!1;
    f.reduce && (K = "object" === typeof f.reduce.call("es5", function(a, b, c, d) {
        return d
    })), v(f, {
        reduce: function(a) {
            var b = w(this), c = I && s(this) ? this.split(""): b, d = c.length>>>0;
            if (!p(a))
                throw new TypeError(a + " is not a function");
            if (!d && 1 === arguments.length)
                throw new TypeError("reduce of empty array with no initial value");
            var e, f = 0;
            if (arguments.length >= 2)
                e = arguments[1];
            else 
                for (; ;) {
                    if (f in c) {
                        e = c[f++];
                        break
                    }
                    if (++f >= d)
                        throw new TypeError("reduce of empty array with no initial value")
                }
            for (; f < d; f++)
                f in c && (e = a.call(void 0, e, c[f], f, b));
            return e
        }
    }, !K);
    var L=!1;
    f.reduceRight && (L = "object" === typeof f.reduceRight.call("es5", function(a, b, c, d) {
        return d
    })), v(f, {
        reduceRight: function(a) {
            var b = w(this), c = I && s(this) ? this.split(""): b, d = c.length>>>0;
            if (!p(a))
                throw new TypeError(a + " is not a function");
            if (!d && 1 === arguments.length)
                throw new TypeError("reduceRight of empty array with no initial value");
            var e, f = d - 1;
            if (arguments.length >= 2)
                e = arguments[1];
            else 
                for (; ;) {
                    if (f in c) {
                        e = c[f--];
                        break
                    }
                    if (--f < 0)
                        throw new TypeError("reduceRight of empty array with no initial value")
                }
            if (f < 0)
                return e;
            do 
                f in c && (e = a.call(void 0, e, c[f], f, b));
            while (f--);
            return e
        }
    }, !L);
    var M = Array.prototype.indexOf&&-1 !== [0, 1].indexOf(1, 2);
    v(f, {
        indexOf: function(b) {
            var c = I && s(this) ? this.split(""): w(this), d = c.length>>>0;
            if (!d)
                return - 1;
            var e = 0;
            for (arguments.length > 1 && (e = a(arguments[1])), e = e >= 0 ? e : Math.max(0, d + e); e < d; e++)
                if (e in c && c[e] === b)
                    return e;
            return - 1
        }
    }, M);
    var N = Array.prototype.lastIndexOf&&-1 !== [0, 1].lastIndexOf(0, - 3);
    v(f, {
        lastIndexOf: function(b) {
            var c = I && s(this) ? this.split(""): w(this), d = c.length>>>0;
            if (!d)
                return - 1;
            var e = d - 1;
            for (arguments.length > 1 && (e = Math.min(e, a(arguments[1]))), e = e >= 0 ? e : d - Math.abs(e); e >= 0; e--)
                if (e in c && b === c[e])
                    return e;
            return - 1
        }
    }, N);
    var O=!{
        toString: null
    }.propertyIsEnumerable("toString"), P = function() {}.propertyIsEnumerable("prototype"), Q = ["toString", "toLocaleString", "valueOf", "hasOwnProperty", "isPrototypeOf", "propertyIsEnumerable", "constructor"], R = Q.length;
    v(Object, {
        keys: function(a) {
            var b = p(a), c = t(a), d = null !== a && "object" === typeof a, e = d && s(a);
            if (!d&&!b&&!c)
                throw new TypeError("Object.keys called on a non-object");
            var f = [], g = P && b;
            if (e || c)
                for (var h = 0; h < a.length; ++h)
                    f.push(String(h));
            else 
                for (var i in a)
                    g && "prototype" === i ||!D(a, i) || f.push(String(i));
            if (O)
                for (var j = a.constructor, k = j && j.prototype === a, l = 0; l < R; l++) {
                    var m = Q[l];
                    k && "constructor" === m ||!D(a, m) || f.push(m)
                }
            return f
        }
    });
    var S = Object.keys && function() {
        return 2 === Object.keys(arguments).length
    }(1, 2), T = Object.keys;
    v(Object, {
        keys: function(a) {
            return T(t(a) ? f.slice.call(a) : a)
        }
    }, !S);
    var U =- 621987552e5, V = "-000001", W = Date.prototype.toISOString&&-1 === new Date(U).toISOString().indexOf(V);
    v(Date.prototype, {
        toISOString: function() {
            var a, b, c, d, e;
            if (!isFinite(this))
                throw new RangeError("Date.prototype.toISOString called on non-finite value.");
            for (d = this.getUTCFullYear(), e = this.getUTCMonth(), d += Math.floor(e / 12), e = (e%12 + 12)%12, a = [e + 1, this.getUTCDate(), this.getUTCHours(), this.getUTCMinutes(), this.getUTCSeconds()], d = (d < 0 ? "-" : d > 9999 ? "+" : "") + ("00000" + Math.abs(d)).slice(0 <= d && d <= 9999?-4 : - 6), b = a.length; b--;)
                c = a[b], c < 10 && (a[b] = "0" + c);
            return d + "-" + a.slice(0, 2).join("-") + "T" + a.slice(2).join(":") + "." + ("000" + this.getUTCMilliseconds()).slice( - 3) + "Z"
        }
    }, W);
    var X=!1;
    try {
        X = Date.prototype.toJSON && null === new Date(0 / 0).toJSON()&&-1 !== new Date(U).toJSON().indexOf(V) && Date.prototype.toJSON.call({
            toISOString: function() {
                return !0
            }
        })
    } catch (Y) {}
    X || (Date.prototype.toJSON = function() {
        var a, b = Object(this), d = c(b);
        if ("number" === typeof d&&!isFinite(d))
            return null;
        if (a = b.toISOString, "function" !== typeof a)
            throw new TypeError("toISOString property is not callable");
        return a.call(b)
    });
    var Z = 1e15 === Date.parse("+033658-09-27T01:46:40.000Z"), $=!isNaN(Date.parse("2012-04-04T24:00:00.500Z")) ||!isNaN(Date.parse("2012-11-31T23:59:59.000Z")), _ = isNaN(Date.parse("2000-01-01T00:00:00.000Z"));
    (!Date.parse || _ || $ ||!Z) && (Date = function(a) {
        function b(c, d, e, f, g, h, i) {
            var j = arguments.length;
            if (this instanceof a) {
                var k = 1 === j && String(c) === c ? new a(b.parse(c)): j >= 7 ? new a(c, d, e, f, g, h, i): j >= 6 ? new a(c, d, e, f, g, h): j >= 5 ? new a(c, d, e, f, g): j >= 4 ? new a(c, d, e, f): j >= 3 ? new a(c, d, e): j >= 2 ? new a(c, d): j >= 1 ? new a(c): new a;
                return k.constructor = b, k
            }
            return a.apply(this, arguments)
        }
        function c(a, b) {
            var c = b > 1 ? 1: 0;
            return f[b] + Math.floor((a - 1969 + c) / 4) - Math.floor((a - 1901 + c) / 100) + Math.floor((a - 1601 + c) / 400) + 365 * (a - 1970)
        }
        function d(b) {
            return Number(new a(1970, 0, 1, 0, 0, 0, b))
        }
        var e = new RegExp("^(\\d{4}|[+-]\\d{6})(?:-(\\d{2})(?:-(\\d{2})(?:T(\\d{2}):(\\d{2})(?::(\\d{2})(?:(\\.\\d{1,}))?)?(Z|(?:([-+])(\\d{2}):(\\d{2})))?)?)?)?$"), f = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365];
        for (var g in a)
            b[g] = a[g];
        return b.now = a.now, b.UTC = a.UTC, b.prototype = a.prototype, b.prototype.constructor = b, b.parse = function(b) {
            var f = e.exec(b);
            if (f) {
                var g, h = Number(f[1]), i = Number(f[2] || 1) - 1, j = Number(f[3] || 1) - 1, k = Number(f[4] || 0), l = Number(f[5] || 0), m = Number(f[6] || 0), n = Math.floor(1e3 * Number(f[7] || 0)), o = Boolean(f[4]&&!f[8]), p = "-" === f[9] ? 1: - 1, q = Number(f[10] || 0), r = Number(f[11] || 0);
                return k < (l > 0 || m > 0 || n > 0 ? 24 : 25) && l < 60 && m < 60 && n < 1e3 && i>-1 && i < 12 && q < 24 && r < 60 && j>-1 && j < c(h, i + 1) - c(h, i) && (g = 60 * (24 * (c(h, i) + j) + k + q * p), g = 1e3 * (60 * (g + l + r * p) + m) + n, o && (g = d(g)), - 864e13 <= g && g <= 864e13) ? g : 0 / 0
            }
            return a.parse.apply(this, arguments)
        }, b
    }(Date)), Date.now || (Date.now = function() {
        return (new Date).getTime()
    });
    var ab = j.toFixed && ("0.000" !== 8e-5.toFixed(3) || "1" !== .9.toFixed(0) || "1.25" !== 1.255.toFixed(2) || "1000000000000000128" !== 0xde0b6b3a7640080.toFixed(0)), bb = {
        base: 1e7,
        size: 6,
        data: [0, 0, 0, 0, 0, 0],
        multiply: function(a, b) {
            for (var c =- 1; ++c < bb.size;)
                b += a * bb.data[c], bb.data[c] = b%bb.base, b = Math.floor(b / bb.base)
        },
        divide: function(a) {
            for (var b = bb.size, c = 0; --b >= 0;)
                c += bb.data[b], bb.data[b] = Math.floor(c / a), c = c%a * bb.base
        },
        numToString: function() {
            for (var a = bb.size, b = ""; --a >= 0;)
                if ("" !== b || 0 === a || 0 !== bb.data[a]) {
                    var c = String(bb.data[a]);
                    "" === b ? b = c : b += "0000000".slice(0, 7 - c.length) + c
                }
            return b
        },
        pow: function nb(a, b, c) {
            return 0 === b ? c : b%2 === 1 ? nb(a, b - 1, c * a) : nb(a * a, b / 2, c)
        },
        log: function(a) {
            for (var b = 0; a >= 4096;)
                b += 12, a/=4096;
            for (; a >= 2;)
                b += 1, a/=2;
            return b
        }
    };
    v(j, {
        toFixed: function(a) {
            var b, c, d, e, f, g, h, i;
            if (b = Number(a), b = b !== b ? 0 : Math.floor(b), b < 0 || b > 20)
                throw new RangeError("Number.toFixed called with invalid number of decimals");
            if (c = Number(this), c !== c)
                return "NaN";
            if (c<=-1e21 || c >= 1e21)
                return String(c);
            if (d = "", c < 0 && (d = "-", c =- c), e = "0", c > 1e-21)
                if (f = bb.log(c * bb.pow(2, 69, 1)) - 69, g = f < 0 ? c * bb.pow(2, - f, 1) : c / bb.pow(2, f, 1), g*=4503599627370496, f = 52 - f, f > 0) {
                    for (bb.multiply(0, g), h = b; h >= 7;)
                        bb.multiply(1e7, 0), h -= 7;
                        for (bb.multiply(bb.pow(10, h, 1), 0), h = f - 1; h >= 23;)
                            bb.divide(1<<23), h -= 23;
                            bb.divide(1<<h), bb.multiply(1, 1), bb.divide(2), e = bb.numToString()
                } else 
                    bb.multiply(0, g), bb.multiply(1<<-f, 0), e = bb.numToString() + "0.00000000000000000000".slice(2, 2 + b);
            return b > 0 ? (i = e.length, e = i <= b ? d + "0.0000000000000000000".slice(0, b - i + 2) + e : d + e.slice(0, i - b) + "." + e.slice(i - b)) : e = d + e, e
        }
    }, ab);
    var cb = i.split;
    2 !== "ab".split(/(?:ab)*/).length || 4 !== ".".split(/(.?)(.?)/).length || "t" === "tesst".split(/(s)*/)[1] || 4 !== "test".split(/(?:)/, - 1).length || "".split(/.?/).length || ".".split(/()()/).length > 1?!function() {
        var a = void 0 === /()??/.exec("")[1];
        i.split = function(b, c) {
            var d = this;
            if (void 0 === b && 0 === c)
                return [];
            if ("[object RegExp]" !== o.call(b))
                return cb.call(this, b, c);
            var e, g, h, i, j = [], k = (b.ignoreCase ? "i" : "") + (b.multiline ? "m" : "") + (b.extended ? "x" : "") + (b.sticky ? "y" : ""), l = 0;
            for (b = new RegExp(b.source, k + "g"), d += "", a || (e = new RegExp("^" + b.source + "$(?!\\s)", k)), c = void 0 === c?-1>>>0 : x(c);
            (g = b.exec(d)) && (h = g.index + g[0].length, !(h > l && (j.push(d.slice(l, g.index)), !a && g.length > 1 && g[0].replace(e, function() {
                for (var a = 1; a < arguments.length - 2; a++)
                    void 0 === arguments[a] && (g[a] = void 0)
            }), g.length > 1 && g.index < d.length && f.push.apply(j, g.slice(1)), i = g[0].length, l = h, j.length >= c)));
            )b.lastIndex === g.index && b.lastIndex++;
            return l === d.length ? (i ||!b.test("")) && j.push("") : j.push(d.slice(l)), j.length > c ? j.slice(0, c) : j
        }
    }() : "0".split(void 0, 0).length && (i.split = function(a, b) {
        return void 0 === a && 0 === b ? [] : cb.call(this, a, b)
    });
    var db = i.replace, eb = function() {
        var a = [];
        return "x".replace(/x(.)?/g, function(b, c) {
            a.push(c)
        }), 1 === a.length && "undefined" === typeof a[0]
    }();
    eb || (i.replace = function(a, b) {
        var c = p(b), d = q(a) && /\)[*?]/.test(a.source);
        if (c && d) {
            var e = function(c) {
                var d = arguments.length, e = a.lastIndex;
                a.lastIndex = 0;
                var f = a.exec(c);
                return a.lastIndex = e, f.push(arguments[d - 2], arguments[d - 1]), b.apply(this, f)
            };
            return db.call(this, a, e)
        }
        return db.call(this, a, b)
    });
    var fb = i.substr, gb = "".substr && "b" !== "0b".substr( - 1);
    v(i, {
        substr: function(a, b) {
            return fb.call(this, a < 0 && (a = this.length + a) < 0 ? 0 : a, b)
        }
    }, gb);
    var hb = "	\n\f\r   ᠎             　\u2028\u2029﻿", ib = "​", jb = "[" + hb + "]", kb = new RegExp("^" + jb + jb + "*"), lb = new RegExp(jb + jb + "*$"), mb = i.trim && (hb.trim() ||!ib.trim());
    v(i, {
        trim: function() {
            if (void 0 === this || null === this)
                throw new TypeError("can't convert " + this + " to object");
            return String(this).replace(kb, "").replace(lb, "")
        }
    }, mb), (8 !== parseInt(hb + "08") || 22 !== parseInt(hb + "0x16")) && (parseInt = function(a) {
        var b = /^0[xX]/;
        return function(c, d) {
            return c = String(c).trim(), Number(d) || (d = b.test(c) ? 16 : 10), a(c, d)
        }
    }(parseInt))
}), function(a, b) {
    "function" === typeof define && define.amd ? define(b) : "object" === typeof exports ? module.exports = b() : a.returnExports = b()
}(this, function() {
    function a(a) {
        try {
            return a.sentinel = 0, 0 === Object.getOwnPropertyDescriptor(a, "sentinel").value
        } catch (b) {}
    }
    function b(a) {
        try {
            return Object.defineProperty(a, "sentinel", {}), "sentinel"in a
        } catch (b) {}
    }
    var c, d, e, f, g = Function.prototype.call, h = Object.prototype, i = g.bind(h.hasOwnProperty), j = i(h, "__defineGetter__");
    if (j && (c = g.bind(h.__defineGetter__), d = g.bind(h.__defineSetter__), e = g.bind(h.__lookupGetter__), f = g.bind(h.__lookupSetter__)), Object.getPrototypeOf || (Object.getPrototypeOf = function(a) {
        var b = a.__proto__;
        return b || null === b ? b : a.constructor ? a.constructor.prototype : h
    }), Object.defineProperty) {
        var k = a({}), l = "undefined" === typeof document || a(document.createElement("div"));
        if (!l ||!k)
            var m = Object.getOwnPropertyDescriptor
    }
    if (!Object.getOwnPropertyDescriptor || m) {
        var n = "Object.getOwnPropertyDescriptor called on a non-object: ";
        Object.getOwnPropertyDescriptor = function(a, b) {
            if ("object" !== typeof a && "function" !== typeof a || null === a)
                throw new TypeError(n + a);
            if (m)
                try {
                    return m.call(Object, a, b)
            } catch (c) {}
            if (i(a, b)) {
                var d = {
                    enumerable: !0,
                    configurable: !0
                };
                if (j) {
                    var g = a.__proto__, k = a !== h;
                    k && (a.__proto__ = h);
                    var l = e(a, b), o = f(a, b);
                    if (k && (a.__proto__ = g), l || o)
                        return l && (d.get = l), o && (d.set = o), d
                }
                return d.value = a[b], d.writable=!0, d
            }
        }
    }
    if (Object.getOwnPropertyNames || (Object.getOwnPropertyNames = function(a) {
        return Object.keys(a)
    }), !Object.create) {
        var o, p=!({
            __proto__: null
        }
        instanceof Object);
        o = p || "undefined" === typeof document ? function() {
            return {
                __proto__: null
            }
        } : function() {
            function a() {}
            var b = document.createElement("iframe"), c = document.body || document.documentElement;
            b.style.display = "none", c.appendChild(b), b.src = "javascript:";
            var d = b.contentWindow.Object.prototype;
            return c.removeChild(b), b = null, delete d.constructor, delete d.hasOwnProperty, delete d.propertyIsEnumerable, delete d.isPrototypeOf, delete d.toLocaleString, delete d.toString, delete d.valueOf, d.__proto__ = null, a.prototype = d, o = function() {
                return new a
            }, new a
        }, Object.create = function(a, b) {
            function c() {}
            var d;
            if (null === a)
                d = o();
            else {
                if ("object" !== typeof a && "function" !== typeof a)
                    throw new TypeError("Object prototype may only be an Object or null");
                c.prototype = a, d = new c, d.__proto__ = a
            }
            return void 0 !== b && Object.defineProperties(d, b), d
        }
    }
    if (Object.defineProperty) {
        var q = b({}), r = "undefined" === typeof document || b(document.createElement("div"));
        if (!q ||!r)
            var s = Object.defineProperty, t = Object.defineProperties
    }
    if (!Object.defineProperty || s) {
        var u = "Property description must be an object: ", v = "Object.defineProperty called on non-object: ", w = "getters & setters can not be defined on this javascript engine";
        Object.defineProperty = function(a, b, g) {
            if ("object" !== typeof a && "function" !== typeof a || null === a)
                throw new TypeError(v + a);
            if ("object" !== typeof g && "function" !== typeof g || null === g)
                throw new TypeError(u + g);
            if (s)
                try {
                    return s.call(Object, a, b, g)
            } catch (k) {}
            if (i(g, "value"))
                if (j && (e(a, b) || f(a, b))) {
                    var l = a.__proto__;
                    a.__proto__ = h, delete a[b], a[b] = g.value, a.__proto__ = l
                } else 
                    a[b] = g.value;
            else {
                if (!j)
                    throw new TypeError(w);
                i(g, "get") && c(a, b, g.get), i(g, "set") && d(a, b, g.set)
            }
            return a
        }
    }(!Object.defineProperties || t) && (Object.defineProperties = function(a, b) {
        if (t)
            try {
                return t.call(Object, a, b)
        } catch (c) {}
        for (var d in b)
            i(b, d) && "__proto__" !== d && Object.defineProperty(a, d, b[d]);
        return a
    }), Object.seal || (Object.seal = function(a) {
        return a
    }), Object.freeze || (Object.freeze = function(a) {
        return a
    });
    try {
        Object.freeze(function() {})
    } catch (x) {
        Object.freeze = function(a) {
            return function(b) {
                return "function" === typeof b ? b : a(b)
            }
        }(Object.freeze)
    }
    Object.preventExtensions || (Object.preventExtensions = function(a) {
        return a
    }), Object.isSealed || (Object.isSealed = function() {
        return !1
    }), Object.isFrozen || (Object.isFrozen = function() {
        return !1
    }), Object.isExtensible || (Object.isExtensible = function(a) {
        if (Object(a) !== a)
            throw new TypeError;
        for (var b = ""; i(a, b);)
            b += "?";
        a[b]=!0;
        var c = i(a, b);
        return delete a[b], c
    })
}), function() {
    var a = {};
    !function() {
        function b(a, c, d) {
            var e = function() {};
            e.prototype = a.prototype;
            var f = new e, g = /xyz/.test(function() {}) ? /\bparent\b/: /.*/;
            d = d || {};
            for (var h in d) {
                var i = d[h], j = f[h];
                f[h] = "function" == typeof j && "function" == typeof i && g.test(i) ? function(a, b) {
                    return function() {
                        var c = this.parent;
                        this.parent = b;
                        var d = a.apply(this, arguments);
                        return this.parent = c, d
                    }
                }(i, j) : i
            }
            f.typename = c;
            var k = function() {
                f.init && f.init.apply(this, arguments)
            };
            return k.prototype = f, k.prototype.constructor = k, k.extend = function(a, c) {
                return "object" == typeof a && (c = a, a = "anonymous"), b(k, a, c)
            }, k
        }
        a.object = b(Object, "Object", {})
    }(), function() {
        var b = Array.prototype, c = Object.prototype, d = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#39;",
            "<": "&lt;",
            ">": "&gt;"
        }, e = /[&"'<>]/g, f = function(a) {
            return d[a]
        }, g = a.lib = {};
        g.withPrettyErrors = function(a, b, c) {
            try {
                return c()
            } catch (d) {
                if (d.Update || (d = new g.TemplateError(d)), d.Update(a), !b) {
                    var e = d;
                    d = new Error(e.message), d.name = e.name
                }
                throw d
            }
        }, g.TemplateError = function(a, b, c) {
            var d = this;
            return a instanceof Error ? (d = a, a = a.name + ": " + a.message) : Error.captureStackTrace && Error.captureStackTrace(d), d.name = "Template render error", d.message = a, d.lineno = b, d.colno = c, d.firstUpdate=!0, d.Update = function(a) {
                var b = "(" + (a || "unknown path") + ")";
                return this.firstUpdate && (this.lineno && this.colno ? b += " [Line " + this.lineno + ", Column " + this.colno + "]" : this.lineno && (b += " [Line " + this.lineno + "]")), b += "\n ", this.firstUpdate && (b += " "), this.message = b + (this.message || ""), this.firstUpdate=!1, this
            }, d
        }, g.TemplateError.prototype = Error.prototype, g.escape = function(a) {
            return a.replace(e, f)
        }, g.isFunction = function(a) {
            return "[object Function]" == c.toString.call(a)
        }, g.isArray = Array.isArray || function(a) {
            return "[object Array]" == c.toString.call(a)
        }, g.isString = function(a) {
            return "[object String]" == c.toString.call(a)
        }, g.isObject = function(a) {
            return "[object Object]" == c.toString.call(a)
        }, g.groupBy = function(a, b) {
            for (var c = {}, d = g.isFunction(b) ? b : function(a) {
                return a[b]
            }, e = 0; e < a.length; e++) {
                var f = a[e], h = d(f, e);
                (c[h] || (c[h] = [])).push(f)
            }
            return c
        }, g.toArray = function(a) {
            return Array.prototype.slice.call(a)
        }, g.without = function(a) {
            var b = [];
            if (!a)
                return b;
            for (var c =- 1, d = a.length, e = g.toArray(arguments).slice(1); ++c < d;) 
                - 1 === g.indexOf(e, a[c]) && b.push(a[c]);
            return b
        }, g.extend = function(a, b) {
            for (var c in b)
                a[c] = b[c];
            return a
        }, g.repeat = function(a, b) {
            for (var c = "", d = 0; d < b; d++)
                c += a;
            return c
        }, g.each = function(a, c, d) {
            if (null != a)
                if (b.each && a.each == b.each)
                    a.forEach(c, d);
                else if (a.length ===+ a.length)
                    for (var e = 0, f = a.length; e < f; e++)
                        c.call(d, a[e], e, a)
        }, g.map = function(a, c) {
            var d = [];
            if (null == a)
                return d;
            if (b.map && a.map === b.map)
                return a.map(c);
            for (var e = 0; e < a.length; e++)
                d[d.length] = c(a[e], e);
            return a.length ===+ a.length && (d.length = a.length), d
        }, g.asyncIter = function(a, b, c) {
            function d() {
                e++, e < a.length ? b(a[e], e, d, c) : c()
            }
            var e =- 1;
            d()
        }, g.asyncFor = function(a, b, c) {
            function d() {
                h++;
                var g = e[h];
                h < f ? b(g, a[g], h, f, d) : c()
            }
            var e = g.keys(a), f = e.length, h =- 1;
            d()
        }, g.indexOf = Array.prototype.indexOf ? function(a, b, c) {
            return Array.prototype.indexOf.call(a, b, c)
        } : function(a, b, c) {
            var d = this.length>>>0;
            for (c =+ c || 0, 1 / 0 === Math.abs(c) && (c = 0), c < 0 && (c += d, c < 0 && (c = 0)); c < d; c++)
                if (a[c] === b)
                    return c;
            return - 1
        }, Array.prototype.map || (Array.prototype.map = function() {
            throw new Error("map is unimplemented for this js engine")
        }), g.keys = function(a) {
            if (Object.prototype.keys)
                return a.keys();
            var b = [];
            for (var c in a)
                a.hasOwnProperty(c) && b.push(c);
            return b
        }
    }(), function() {
        function b(a, b, c) {
            return function() {
                var f, g = e(arguments), h = d(arguments);
                if (g > a.length) {
                    f = Array.prototype.slice.call(arguments, 0, a.length);
                    for (var i = Array.prototype.slice.call(arguments, f.length, g), j = 0; j < i.length; j++)
                        j < b.length && (h[b[j]] = i[j]);
                    f.push(h)
                } else if (g < a.length) {
                    f = Array.prototype.slice.call(arguments, 0, g);
                    for (var j = g; j < a.length; j++) {
                        var k = a[j];
                        f.push(h[k]), delete h[k]
                    }
                    f.push(h)
                } else 
                    f = arguments;
                return c.apply(this, f)
            }
        }
        function c(a) {
            return a.__keywords=!0, a
        }
        function d(a) {
            var b = a.length;
            if (b) {
                var c = a[b - 1];
                if (c && c.hasOwnProperty("__keywords"))
                    return c
            }
            return {}
        }
        function e(a) {
            var b = a.length;
            if (0 === b)
                return 0;
            var c = a[b - 1];
            return c && c.hasOwnProperty("__keywords") ? b - 1 : b
        }
        function f(a) {
            return "string" != typeof a ? a : void(this.val = a)
        }
        function g(a, b) {
            return a instanceof f ? new f(b) : b.toString()
        }
        function h(a) {
            var b = typeof a;
            return "string" === b ? new f(a) : "function" !== b ? a : function() {
                var b = a.apply(this, arguments);
                return "string" === typeof b ? new f(b) : b
            }
        }
        function i(a, b) {
            return a = void 0 !== a && null !== a ? a : "", b && "string" === typeof a && (a = p.escape(a)), a
        }
        function j(a, b) {
            return a = a || {}, "function" === typeof a[b] ? function() {
                return a[b].apply(a, arguments)
            } : a[b]
        }
        function k(a, b, c) {
            if (!a)
                throw new Error("Unable to call `" + b + "`, which is undefined or falsey");
            if ("function" !== typeof a)
                throw new Error("Unable to call `" + b + "`, which is not a function");
            return a.apply(this, c)
        }
        function l(a, b, c) {
            var d = b.lookup(c);
            return void 0 !== d && null !== d ? d : a.lookup(c)
        }
        function m(a, b, c) {
            return a.lineno ? a : new p.TemplateError(a, b, c)
        }
        function n(a, b, c, d) {
            if (p.isArray(a)) {
                var e = a.length;
                p.asyncIter(a, function(a, d, f) {
                    switch (b) {
                    case 1:
                        c(a, d, e, f);
                        break;
                    case 2:
                        c(a[0], a[1], d, e, f);
                        break;
                    case 3:
                        c(a[0], a[1], a[2], d, e, f);
                        break;
                    default:
                        a.push(d, f), c.apply(this, a)
                    }
                }, d)
            } else 
                p.asyncFor(a, function(a, b, d, e, f) {
                    c(a, b, d, e, f)
                }, d)
        }
        function o(a, b, c, d) {
            function e(a, b) {
                h++, g[a] = b, h == f && d(null, g.join(""))
            }
            var f, g, h = 0;
            if (p.isArray(a))
                if (f = a.length, g = new Array(f), 0 == f)
                    d(null, "");
                else 
                    for (var i = 0; i < a.length; i++) {
                        var j = a[i];
                        switch (b) {
                        case 1:
                            c(j, i, f, e);
                            break;
                        case 2:
                            c(j[0], j[1], i, f, e);
                            break;
                        case 3:
                            c(j[0], j[1], j[2], i, f, e);
                            break;
                        default:
                            j.push(i, e), c.apply(this, j)
                        }
                    } else {
                var k = p.keys(a);
                if (f = k.length, g = new Array(f), 0 == f)
                    d(null, "");
                else 
                    for (var i = 0; i < k.length; i++) {
                        var l = k[i];
                        c(l, a[l], i, f, e)
                    }
                }
                }
        var p = a.lib, q = a.object, r = q.extend({
            init: function(a) {
                this.variables = {}, this.parent = a
            },
            set: function(a, b, c) {
                var d = a.split("."), e = this.variables, f = this;
                if (c) {
                    if (f = this.resolve(d[0]))
                        return void f.set(a, b);
                    f = this
                }
                for (var g = 0; g < d.length - 1; g++) {
                    var h = d[g];
                    e[h] || (e[h] = {}), e = e[h]
                }
                e[d[d.length - 1]] = b
            },
            get: function(a) {
                var b = this.variables[a];
                return void 0 !== b && null !== b ? b : null
            },
            lookup: function(a) {
                var b = this.parent, c = this.variables[a];
                return void 0 !== c && null !== c ? c : b && b.lookup(a)
            },
            resolve: function(a) {
                var b = this.parent, c = this.variables[a];
                return null != c ? this : b && b.resolve(a)
            },
            push: function() {
                return new r(this)
            },
            pop: function() {
                return this.parent
            }
        });
        f.prototype = Object.create(String.prototype), f.prototype.valueOf = function() {
            return this.val
        }, f.prototype.toString = function() {
            return this.val
        }, a.runtime = {
            Frame: r,
            makeMacro: b,
            makeKeywordArgs: c,
            numArgs: e,
            suppressValue: i,
            memberLookup: j,
            contextOrFrameLookup: l,
            callWrap: k,
            handleError: m,
            isArray: p.isArray,
            keys: p.keys,
            SafeString: f,
            copySafeness: g,
            markSafe: h,
            asyncEach: n,
            asyncAll: o
        }
    }(), function() {
        var b = a.object, c = a.lib, d = b.extend({
            on: function(a, b) {
                this.listeners = this.listeners || {}, this.listeners[a] = this.listeners[a] || [], this.listeners[a].push(b)
            },
            emit: function(a) {
                var b = Array.prototype.slice.call(arguments, 1);
                this.listeners && this.listeners[a] && c.each(this.listeners[a], function(a) {
                    a.apply(null, b)
                })
            }
        });
        a.loader = d
    }(), function() {
        var b = a.loader, c = b.extend({
            init: function(a, b) {
                this.precompiled = window.nunjucksPrecompiled || {}, this.baseURL = a || "", this.neverUpdate = b
            },
            getSource: function(a) {
                if (this.precompiled[a])
                    return {
                        src: {
                            type: "code",
                            obj: this.precompiled[a]
                        },
                        path: a
                    };
                var b = this.fetch(this.baseURL + "/" + a);
                return b ? {
                    src: b,
                    path: a,
                    noCache: !this.neverUpdate
                } : null
            },
            fetch: function(a) {
                var b, c, d=!0;
                return window.XMLHttpRequest ? b = new XMLHttpRequest : window.ActiveXObject && (b = new ActiveXObject("Microsoft.XMLHTTP")), b.onreadystatechange = function() {
                    4 !== b.readyState || 0 !== b.status && 200 !== b.status ||!d || (d=!1, c = b.responseText)
                }, a += ( - 1 === a.indexOf("?") ? "?" : "&") + "s=" + (new Date).getTime(), b.open("GET", a, !1), b.send(), c
            }
        });
        a["web-loaders"] = {
            WebLoader: c
        }
    }(), function() {
        a.loaders = "undefined" === typeof window || window !== this ? a["node-loaders"] : a["web-loaders"]
    }(), function() {
        var b = a.lib, c = a.runtime, d = {
            abs: function(a) {
                return Math.abs(a)
            },
            batch: function(a, b, c) {
                for (var d = [], e = [], f = 0; f < a.length; f++)
                    f%b === 0 && e.length && (d.push(e), e = []), e.push(a[f]);
                if (e.length) {
                    if (c)
                        for (var f = e.length; f < b; f++)
                            e.push(c);
                    d.push(e)
                }
                return d
            },
            capitalize: function(a) {
                var b = a.toLowerCase();
                return c.copySafeness(a, b.charAt(0).toUpperCase() + b.slice(1))
            },
            center: function(a, d) {
                if (d = d || 80, a.length >= d)
                    return a;
                var e = d - a.length, f = b.repeat(" ", e / 2 - e%2), g = b.repeat(" ", e / 2);
                return c.copySafeness(a, f + a + g)
            },
            "default": function(a, b) {
                return a ? a : b
            },
            dictsort: function(a, c, d) {
                if (!b.isObject(a))
                    throw new b.TemplateError("dictsort filter: val must be an object");
                var e = [];
                for (var f in a)
                    e.push([f, a[f]]);
                var g;
                if (void 0 === d || "key" === d)
                    g = 0;
                else {
                    if ("value" !== d)
                        throw new b.TemplateError("dictsort filter: You can only sort by either key or value");
                    g = 1
                }
                return e.sort(function(a, d) {
                    var e = a[g], f = d[g];
                    return c || (b.isString(e) && (e = e.toUpperCase()), b.isString(f) && (f = f.toUpperCase())), e > f ? 1 : e == f ? 0 : - 1
                }), e
            },
            escape: function(a) {
                return "string" == typeof a || a instanceof c.SafeString ? b.escape(a) : a
            },
            safe: function(a) {
                return c.markSafe(a)
            },
            first: function(a) {
                return a[0]
            },
            groupby: function(a, c) {
                return b.groupBy(a, c)
            },
            indent: function(a, d, e) {
                d = d || 4;
                for (var f = "", g = a.split("\n"), h = b.repeat(" ", d), i = 0; i < g.length; i++)
                    f += 0 != i || e ? h + g[i] + "\n" : g[i] + "\n";
                return c.copySafeness(a, f)
            },
            join: function(a, c, d) {
                return c = c || "", d && (a = b.map(a, function(a) {
                    return a[d]
                })), a.join(c)
            },
            last: function(a) {
                return a[a.length - 1]
            },
            length: function(a) {
                return void 0 !== a ? a.length : 0
            },
            list: function(a) {
                if (b.isString(a))
                    return a.split("");
                if (b.isObject(a)) {
                    var c = [];
                    if (Object.keys)
                        c = Object.keys(a);
                    else 
                        for (var d in a)
                            c.push(d);
                    return b.map(c, function(b) {
                        return {
                            key: b,
                            value: a[b]
                        }
                    })
                }
                throw new b.TemplateError("list filter: type not iterable")
            },
            lower: function(a) {
                return a.toLowerCase()
            },
            random: function(a) {
                return a[Math.floor(Math.random() * a.length)]
            },
            replace: function(a, b, d, e) {
                if (b instanceof RegExp)
                    return a.replace(b, d);
                var f = a, g = f, h = 1;
                for (f = f.replace(b, d); g != f&&!(h >= e);)
                    g = f, f = f.replace(b, d), h++;
                return c.copySafeness(a, f)
            },
            reverse: function(a) {
                var e;
                return e = b.isString(a) ? d.list(a) : b.map(a, function(a) {
                    return a
                }), e.reverse(), b.isString(a) ? c.copySafeness(a, e.join("")) : e
            },
            round: function(a, b, c) {
                b = b || 0;
                var d, e = Math.pow(10, b);
                return d = "ceil" == c ? Math.ceil : "floor" == c ? Math.floor : Math.round, d(a * e) / e
            },
            slice: function(a, b, c) {
                for (var d = Math.floor(a.length / b), e = a.length%b, f = 0, g = [], h = 0; h < b; h++) {
                    var i = f + h * d;
                    h < e && f++;
                    var j = f + (h + 1) * d, k = a.slice(i, j);
                    c && h >= e && k.push(c), g.push(k)
                }
                return g
            },
            sort: function(a, c, d, e) {
                return a = b.map(a, function(a) {
                    return a
                }), a.sort(function(a, f) {
                    var g, h;
                    return e ? (g = a[e], h = f[e]) : (g = a, h = f), !d && b.isString(g) && b.isString(h) && (g = g.toLowerCase(), h = h.toLowerCase()), g < h ? c ? 1 : - 1 : g > h ? c?-1 : 1 : 0
                }), a
            },
            string: function(a) {
                return c.copySafeness(a, a)
            },
            title: function(a) {
                for (var b = a.split(" "), e = 0; e < b.length; e++)
                    b[e] = d.capitalize(b[e]);
                return c.copySafeness(a, b.join(" "))
            },
            trim: function(a) {
                return c.copySafeness(a, a.replace(/^\s*|\s*$/g, ""))
            },
            truncate: function(a, b, d, e) {
                var f = a;
                if (b = b || 255, a.length <= b)
                    return a;
                if (d)
                    a = a.substring(0, b);
                else {
                    var g = a.lastIndexOf(" ", b);
                    - 1 === g && (g = b), a = a.substring(0, g)
                }
                return a += void 0 !== e && null !== e ? e : "...", c.copySafeness(f, a)
            },
            upper: function(a) {
                return a.toUpperCase()
            },
            urlencode: function(a) {
                var c = encodeURIComponent;
                if (b.isString(a))
                    return c(a);
                var d;
                if (b.isArray(a))
                    d = a.map(function(a) {
                        return c(a[0]) + "=" + c(a[1])
                    });
                else {
                    d = [];
                    for (var e in a)
                        a.hasOwnProperty(e) && d.push(c(e) + "=" + c(a[e]))
                }
                return d.join("&")
            },
            urlize: function(a, b, c) {
                isNaN(b) && (b = 1 / 0);
                var d = c===!0 ? ' rel="nofollow"': "", e = /^(?:\(|<|&lt;)?(.*?)(?:\.|,|\)|\n|&gt;)?$/, f = /^[\w.!#$%&'*+\-\/=?\^`{|}~]+@[a-z\d\-]+(\.[a-z\d\-]+)+$/i, g = /^https?:\/\/.*$/, h = /^www\./, i = /\.(?:org|net|com)(?:\:|\/|$)/, j = a.split(/\s+/).filter(function(a) {
                    return a && a.length
                }).map(function(a) {
                    var c = a.match(e), j = c && c[1] || a;
                    return g.test(j) ? '<a href="' + j + '"' + d + ">" + j.substr(0, b) + "</a>" : h.test(j) ? '<a href="http://' + j + '"' + d + ">" + j.substr(0, b) + "</a>" : f.test(j) ? '<a href="mailto:' + j + '">' + j + "</a>" : i.test(j) ? '<a href="http://' + j + '"' + d + ">" + j.substr(0, b) + "</a>" : a
                });
                return j.join(" ")
            },
            wordcount: function(a) {
                var b = a ? a.match(/\w+/g): null;
                return b ? b.length : null
            },
            "float": function(a, b) {
                var c = parseFloat(a);
                return isNaN(c) ? b : c
            },
            "int": function(a, b) {
                var c = parseInt(a, 10);
                return isNaN(c) ? b : c
            }
        };
        d.d = d["default"], d.e = d.escape, a.filters = d
    }(), function() {
        function b(a) {
            var b =- 1;
            return this.current = null, {
                reset: function() {
                    b =- 1, this.current = null
                },
                next: function() {
                    return b++, b >= a.length && (b = 0), this.current = a[b], this.current
                }
            }
        }
        function c(a) {
            a = a || ",";
            var b=!0;
            return function() {
                var c = b ? "": a;
                return b=!1, c
            }
        }
        var d = {
            range: function(a, b, c) {
                b ? c || (c = 1) : (b = a, a = 0, c = 1);
                for (var d = [], e = a; e < b; e += c)
                    d.push(e);
                return d
            },
            cycler: function() {
                return b(Array.prototype.slice.call(arguments))
            },
            joiner: function(a) {
                return c(a)
            }
        };
        a.globals = d
    }(), function() {
        var b = a.path, c = a.lib, d = a.object, e = (a.lexer, a.compiler), f = a.filters, g = a.loaders, h = a.runtime, i = a.globals, j = h.Frame, k = d.extend({
            init: function(a, b) {
                b = b || {}, this.dev=!!b.dev, this.lexerTags = b.tags, this.autoesc=!!b.autoescape, this.loaders = a ? c.isArray(a) ? a : [a] : g.FileSystemLoader ? [new g.FileSystemLoader("views")] : [new g.WebLoader("/views")], this.initCache(), this.filters = {}, this.asyncFilters = [], this.extensions = {}, this.extensionsList = [];
                for (var d in f)
                    this.addFilter(d, f[d])
            },
            initCache: function() {
                var a = {};
                c.each(this.loaders, function(b) {
                    "function" === typeof b.on && b.on("update", function(b) {
                        a[b] = null
                    })
                }), this.cache = a
            },
            addExtension: function(a, b) {
                b._name = a, this.extensions[a] = b, this.extensionsList.push(b)
            },
            getExtension: function(a) {
                return this.extensions[a]
            },
            addGlobal: function(a, b) {
                i[a] = b
            },
            addFilter: function(a, b, c) {
                var d = b;
                c && this.asyncFilters.push(a), this.filters[a] = d
            },
            getFilter: function(a) {
                if (!this.filters[a])
                    throw new Error("filter not found: " + a);
                return this.filters[a]
            },
            getTemplate: function(a, b, d) {
                if (a && a.raw && (a = a.raw), c.isFunction(b) && (d = b, b=!1), "string" !== typeof a)
                    throw new Error("template names must be a string: " + a);
                var e = this.cache[a];
                if (!e) {
                    var f;
                    return c.asyncIter(this.loaders, function(b, c, d, e) {
                        function f(a) {
                            a ? e(a) : d()
                        }
                        b.async ? b.getSource(a, function(a, b) {
                            if (a)
                                throw a;
                            f(b)
                        }) : f(b.getSource(a))
                    }, function(c) {
                        if (c) {
                            var e = new m(c.src, this, c.path, b);
                            c.noCache || (this.cache[a] = e), d ? d(null, e) : f = e
                        } else {
                            var g = new Error("template not found: " + a);
                            if (!d)
                                throw g;
                            d(g)
                        }
                    }.bind(this)), f
                }
                return b && e.compile(), d ? void d(null, e) : e
            },
            express: function(a) {
                function c(a, c) {
                    if (this.name = a, this.path = a, this.defaultEngine = c.defaultEngine, this.ext = b.extname(a), !this.ext&&!this.defaultEngine)
                        throw new Error("No default engine was specified and no extension was provided.");
                    this.ext || (this.name += this.ext = ("." !== this.defaultEngine[0] ? "." : "") + this.defaultEngine)
                }
                var d = this;
                c.prototype.render = function(a, b) {
                    d.render(this.name, a, b)
                }, a.set("view", c)
            },
            render: function(a, b, d) {
                c.isFunction(b) && (d = b, b = null);
                var e = null;
                return this.getTemplate(a, function(a, c) {
                    if (a && d)
                        d(a);
                    else {
                        if (a)
                            throw a;
                        c.render(b, d || function(a, b) {
                            if (a)
                                throw a;
                            e = b
                        })
                    }
                }), e
            },
            renderString: function(a, b, c) {
                var d = new m(a, this);
                return d.render(b, c)
            }
        }), l = d.extend({
            init: function(a, b) {
                this.ctx = a, this.blocks = {}, this.exported = [];
                for (var c in b)
                    this.addBlock(c, b[c])
            },
            lookup: function(a) {
                return a in i&&!(a in this.ctx) ? i[a] : this.ctx[a]
            },
            setVariable: function(a, b) {
                this.ctx[a] = b
            },
            getVariables: function() {
                return this.ctx
            },
            addBlock: function(a, b) {
                this.blocks[a] = this.blocks[a] || [], this.blocks[a].push(b)
            },
            getBlock: function(a) {
                if (!this.blocks[a])
                    throw new Error('unknown block "' + a + '"');
                return this.blocks[a][0]
            },
            getSuper: function(a, b, d, e, f, g) {
                var h = c.indexOf(this.blocks[b] || [], d), i = this.blocks[b][h + 1], j = this;
                if ( - 1 == h ||!i)
                    throw new Error('no super block available for "' + b + '"');
                i(a, j, e, f, g)
            },
            addExport: function(a) {
                this.exported.push(a)
            },
            getExported: function() {
                for (var a = {}, b = 0; b < this.exported.length; b++) {
                    var c = this.exported[b];
                    a[c] = this.ctx[c]
                }
                return a
            }
        }), m = d.extend({
            init: function(a, b, d, e) {
                if (this.env = b || new k, c.isObject(a))
                    switch (a.type) {
                    case"code":
                        this.tmplProps = a.obj;
                        break;
                    case"string":
                        this.tmplStr = a.obj
                    } else {
                    if (!c.isString(a))
                        throw new Error("src must be a string or an object describing the source");
                    this.tmplStr = a
                }
                this.path = d, e ? c.withPrettyErrors(this.path, this.env.dev, this._compile.bind(this)) : this.compiled=!1
            },
            render: function(a, b, d) {
                return "function" === typeof a ? (d = a, a = {}) : "function" === typeof b && (d = b, b = null), c.withPrettyErrors(this.path, this.env.dev, function() {
                    this.compile();
                    var c = new l(a || {}, this.blocks), e = null;
                    return this.rootRenderFunc(this.env, c, b || new j, h, d || function(a, b) {
                        if (a)
                            throw a;
                        e = b
                    }), e
                }.bind(this))
            },
            getExported: function(a) {
                this.compile();
                var b = new l({}, this.blocks);
                this.rootRenderFunc(this.env, b, new j, h, function() {
                    a(null, b.getExported())
                })
            },
            compile: function() {
                this.compiled || this._compile()
            },
            _compile: function() {
                var a;
                if (this.tmplProps)
                    a = this.tmplProps;
                else {
                    var b = e.compile(this.tmplStr, this.env.asyncFilters, this.env.extensionsList, this.path, this.env.lexerTags), c = new Function(b);
                    a = c()
                }
                this.blocks = this._getBlocks(a), this.rootRenderFunc = a.root, this.compiled=!0
            },
            _getBlocks: function(a) {
                var b = {};
                for (var c in a)
                    "b_" == c.slice(0, 2) && (b[c.slice(2)] = a[c]);
                return b
            }
        });
        a.environment = {
            Environment: k,
            Template: m
        }
    }();
    var b, c = a.lib, d = a.environment, e = a.compiler, f = a.parser, g = a.lexer, h = a.runtime, i = a.loader, j = a.loaders, k = a.precompile;
    b = {}, b.Environment = d.Environment, b.Template = d.Template, b.Loader = i, b.FileSystemLoader = j.FileSystemLoader, b.WebLoader = j.WebLoader, b.compiler = e, b.parser = f, b.lexer = g, b.runtime = h;
    var l;
    b.configure = function(a, b) {
        b = b || {}, c.isObject(a) && (b = a, a = null);
        var e = "watch"in b?!b.watch : !1, f = j.FileSystemLoader || j.WebLoader;
        return l = new d.Environment(new f(a, e), b), b && b.express && l.express(b.express), l
    }, b.compile = function(a, c, d, e) {
        return l || b.configure(), new b.Template(a, c, d, e)
    }, b.render = function(a, c, d) {
        return l || b.configure(), l.render(a, c, d)
    }, b.renderString = function(a, c, d) {
        return l || b.configure(), l.renderString(a, c, d)
    }, k && (b.precompile = k.precompile, b.precompileString = k.precompileString), b.require = function(b) {
        return a[b]
    }, "function" === typeof define && define.amd ? define(function() {
        return b
    }) : (window.nunjucks = b, "undefined" !== typeof module && (module.exports = b))
}(), function() {
    var a = nunjucks.require("runtime"), b = nunjucks.require("lib"), c = a.contextOrFrameLookup;
    a.contextOrFrameLookup = function(a, b, d) {
        var e = c.apply(this, arguments);
        if (void 0 === e)
            switch (d) {
            case"True":
                return !0;
            case"False":
                return !1;
            case"None":
                return null;
            case"none":
                return null
            }
        return e
    };
    var d = a.memberLookup, e = {
        pop: function(a) {
            if (void 0 === a)
                return this.pop();
            if (a >= this.length || a < 0)
                throw new Error("KeyError");
            return this.splice(a, 1)
        },
        remove: function(a) {
            for (var b = 0; b < this.length; b++)
                if (this[b] == a)
                    return this.splice(b, 1);
            throw new Error("ValueError")
        },
        count: function(a) {
            for (var b = 0, c = 0; c < this.length; c++)
                this[c] == a && b++;
            return b
        },
        index: function(a) {
            var b;
            if ( - 1 == (b = this.indexOf(a)))
                throw new Error("ValueError");
            return b
        },
        find: function(a) {
            return this.indexOf(a)
        },
        insert: function(a, b) {
            return this.splice(a, 0, b)
        },
        append: function(a) {
            return this.push(a)
        }
    }, f = {
        values: function() {
            var a = [];
            for (var b in this)
                a.push(this[b]);
            return a
        },
        keys: function() {
            var a = [];
            for (var b in this)
                a.push(b);
            return a
        },
        get: function(a, b) {
            var c = this[a];
            return void 0 === c && (c = b), c
        },
        has_key: function(a) {
            return this.hasOwnProperty(a)
        },
        copy: function() {
            var a = _.extend({}, this);
            return a
        },
        pop: function(a, b) {
            var c = this[a];
            if (void 0 === c && void 0 !== b)
                c = b;
            else {
                if (void 0 === c)
                    throw new Error("KeyError");
                delete this[a]
            }
            return c
        },
        popitem: function() {
            for (var a in this) {
                var b = this[a];
                return delete this[a], [a, b]
            }
            throw new Error("KeyError")
        },
        setdefault: function(a, b) {
            return a in this ? this[a] : (void 0 === b && (b = null), this[a] = b)
        },
        update: function(a) {
            for (var b in a)
                this[b] = a[b];
            return null
        }
    };
    f.iteritems = f.items, f.itervalues = f.values, f.iterkeys = f.keys, a.memberLookup = function(a, c) {
        return a = a || {}, b.isArray(a) && e.hasOwnProperty(c) ? function() {
            return e[c].apply(a, arguments)
        } : b.isObject(a) && f.hasOwnProperty(c) ? function() {
            return f[c].apply(a, arguments)
        } : d.apply(this, arguments)
    }
}();
var qq = qq || {}, qq = function(a) {
    "use strict";
    return {
        hide: function() {
            return a.style.display = "none", this
        },
        attach: function(b, c) {
            return a.addEventListener ? a.addEventListener(b, c, !1) : a.attachEvent && a.attachEvent("on" + b, c), function() {
                qq(a).detach(b, c)
            }
        },
        detach: function(b, c) {
            return a.removeEventListener ? a.removeEventListener(b, c, !1) : a.attachEvent && a.detachEvent("on" + b, c), this
        },
        contains: function(b) {
            return a == b?!0 : a.contains ? a.contains(b) : !!(8 & b.compareDocumentPosition(a))
        },
        insertBefore: function(b) {
            return b.parentNode.insertBefore(a, b), this
        },
        remove: function() {
            return a.parentNode.removeChild(a), this
        },
        css: function(b) {
            return null != b.opacity && "string" != typeof a.style.opacity && "undefined" != typeof a.filters && (b.filter = "alpha(opacity=" + Math.round(100 * b.opacity) + ")"), qq.extend(a.style, b), this
        },
        hasClass: function(b) {
            var c = new RegExp("(^| )" + b + "( |$)");
            return c.test(a.className)
        },
        addClass: function(b) {
            return qq(a).hasClass(b) || (a.className += " " + b), this
        },
        removeClass: function(b) {
            var c = new RegExp("(^| )" + b + "( |$)");
            return a.className = a.className.replace(c, " ").replace(/^\s+|\s+$/g, ""), this
        },
        getByClass: function(b) {
            if (a.querySelectorAll)
                return a.querySelectorAll("." + b);
            for (var c = [], d = a.getElementsByTagName("*"), e = d.length, f = 0; f < e; f++)
                qq(d[f]).hasClass(b) && c.push(d[f]);
            return c
        },
        children: function() {
            for (var b = [], c = a.firstChild; c;)
                1 == c.nodeType && b.push(c), c = c.nextSibling;
            return b
        },
        setText: function(b) {
            return a.innerText = b, a.textContent = b, this
        },
        clearText: function() {
            return qq(a).setText("")
        }
    }
};
qq.log = function(a, b) {
    window.console && (b && "info" !== b ? window.console[b] ? window.console[b](a) : window.console.log("<" + b + "> " + a) : window.console.log(a))
}, qq.isObject = function(a) {
    "use strict";
    return null !== a && a && "object" === typeof a && a.constructor === Object
}, qq.extend = function(a, b, c) {
    "use strict";
    var d;
    for (d in b)
        b.hasOwnProperty(d) && (c && qq.isObject(b[d]) ? (void 0 === a[d] && (a[d] = {}), qq.extend(a[d], b[d], !0)) : a[d] = b[d])
}, qq.indexOf = function(a, b, c) {
    if (a.indexOf)
        return a.indexOf(b, c);
    c = c || 0;
    var d = a.length;
    for (c < 0 && (c += d); c < d; c++)
        if (c in a && a[c] === b)
            return c;
    return - 1
}, qq.getUniqueId = function() {
    var a = 0;
    return function() {
        return a++
    }
}(), qq.ie = function() {
    return - 1 != navigator.userAgent.indexOf("MSIE")
}, qq.ie10 = function() {
    return - 1 != navigator.userAgent.indexOf("MSIE 10")
}, qq.safari = function() {
    return void 0 != navigator.vendor&&-1 != navigator.vendor.indexOf("Apple")
}, qq.chrome = function() {
    return void 0 != navigator.vendor&&-1 != navigator.vendor.indexOf("Google")
}, qq.firefox = function() {
    return - 1 != navigator.userAgent.indexOf("Mozilla") && void 0 != navigator.vendor && "" == navigator.vendor
}, qq.windows = function() {
    return "Win32" == navigator.platform
}, qq.preventDefault = function(a) {
    a.preventDefault ? a.preventDefault() : a.returnValue=!1
}, qq.toElement = function() {
    var a = document.createElement("div");
    return function(b) {
        a.innerHTML = b;
        var c = a.firstChild;
        return a.removeChild(c), c
    }
}(), qq.obj2url = function(a, b, c) {
    var d = [], e = "&", f = function(a, c) {
        var e = b ? /\[\]$/.test(b) ? b: b + "[" + c + "]": c;
        "undefined" != e && "undefined" != c && d.push("object" === typeof a ? qq.obj2url(a, e, !0) : "[object Function]" === Object.prototype.toString.call(a) ? encodeURIComponent(e) + "=" + encodeURIComponent(a()) : encodeURIComponent(e) + "=" + encodeURIComponent(a))
    };
    if (!c && b)
        e = /\?/.test(b) ? /\?$/.test(b) ? "" : "&" : "?", d.push(b), d.push(qq.obj2url(a));
    else if ("[object Array]" === Object.prototype.toString.call(a) && "undefined" != typeof a)
        for (var g = 0, h = a.length; g < h; ++g)
            f(a[g], g);
    else if ("undefined" != typeof a && null !== a && "object" === typeof a)
        for (var g in a)
            f(a[g], g);
    else 
        d.push(encodeURIComponent(b) + "=" + encodeURIComponent(a));
    return b ? d.join(e) : d.join(e).replace(/^&/, "").replace(/%20/g, "+")
}, qq.DisposeSupport = {
    _disposers: [],
    dispose: function() {
        for (var a; a = this._disposers.shift();)
            a()
    },
    addDisposer: function(a) {
        this._disposers.push(a)
    },
    _attach: function() {
        this.addDisposer(qq(arguments[0]).attach.apply(this, Array.prototype.slice.call(arguments, 1)))
    }
}, qq.UploadButton = function(a) {
    this._options = {
        element: null,
        multiple: !1,
        acceptFiles: null,
        name: "file",
        onChange: function() {},
        hoverClass: "qq-upload-button-hover",
        focusClass: "qq-upload-button-focus"
    }, qq.extend(this._options, a), qq.extend(this, qq.DisposeSupport), this._element = this._options.element, qq(this._element).css({
        position: "relative",
        overflow: "hidden",
        direction: "ltr"
    }), this._input = this._createInput()
}, qq.UploadButton.prototype = {
    getInput: function() {
        return this._input
    },
    reset: function() {
        this._input.parentNode && qq(this._input).remove(), qq(this._element).removeClass(this._options.focusClass), this._input = this._createInput()
    },
    _createInput: function() {
        var a = document.createElement("input");
        this._options.multiple && a.setAttribute("multiple", "multiple"), this._options.acceptFiles && a.setAttribute("accept", this._options.acceptFiles), a.setAttribute("type", "file"), a.setAttribute("name", this._options.name), qq(a).css({
            position: "absolute",
            right: 0,
            top: 0,
            fontFamily: "Arial",
            fontSize: "118px",
            margin: 0,
            padding: 0,
            cursor: "pointer",
            opacity: 0
        }), this._element.appendChild(a);
        var b = this;
        return this._attach(a, "change", function() {
            b._options.onChange(a)
        }), this._attach(a, "mouseover", function() {
            qq(b._element).addClass(b._options.hoverClass)
        }), this._attach(a, "mouseout", function() {
            qq(b._element).removeClass(b._options.hoverClass)
        }), this._attach(a, "focus", function() {
            qq(b._element).addClass(b._options.focusClass)
        }), this._attach(a, "blur", function() {
            qq(b._element).removeClass(b._options.focusClass)
        }), window.attachEvent && a.setAttribute("tabIndex", "-1"), a
    }
}, qq.FineUploaderBasic = function(a) {
    this._options = {
        debug: !1,
        button: null,
        multiple: !0,
        maxConnections: 3,
        disableCancelForFormUploads: !1,
        autoUpload: !0,
        request: {
            endpoint: "/server/upload",
            params: {},
            customHeaders: {},
            forceMultipart: !1,
            inputName: "qqfile"
        },
        validation: {
            allowedExtensions: [],
            sizeLimit: 0,
            minSizeLimit: 0,
            stopOnFirstInvalidFile: !0
        },
        callbacks: {
            onSubmit: function() {},
            onComplete: function() {},
            onCancel: function() {},
            onUpload: function() {},
            onProgress: function() {},
            onError: function() {},
            onAutoRetry: function() {},
            onManualRetry: function() {},
            onValidate: function() {}
        },
        messages: {
            typeError: "{file} has an invalid extension. Valid extension(s): {extensions}.",
            sizeError: "{file} is too large, maximum file size is {sizeLimit}.",
            minSizeError: "{file} is too small, minimum file size is {minSizeLimit}.",
            emptyError: "{file} is empty, please select files again without it.",
            noFilesError: "No files to upload.",
            onLeave: "The files are being uploaded, if you leave now the upload will be cancelled."
        },
        retry: {
            enableAuto: !1,
            maxAutoAttempts: 3,
            autoAttemptDelay: 5,
            preventRetryResponseProperty: "preventRetry"
        }
    }, qq.extend(this._options, a, !0), this._wrapCallbacks(), qq.extend(this, qq.DisposeSupport), this._filesInProgress = 0, this._storedFileIds = [], this._autoRetries = [], this._retryTimeouts = [], this._preventRetries = [], this._handler = this._createUploadHandler(), this._options.button && (this._button = this._createUploadButton(this._options.button)), this._preventLeaveInProgress()
}, qq.FineUploaderBasic.prototype = {
    log: function(a, b) {
        !this._options.debug || b && "info" !== b ? b && "info" !== b && qq.log("[FineUploader] " + a, b) : qq.log("[FineUploader] " + a)
    },
    setParams: function(a) {
        this._options.request.params = a
    },
    getInProgress: function() {
        return this._filesInProgress
    },
    uploadStoredFiles: function() {
        "use strict";
        for (; this._storedFileIds.length;)
            this._filesInProgress++, this._handler.upload(this._storedFileIds.shift(), this._options.request.params)
    },
    clearStoredFiles: function() {
        this._storedFileIds = []
    },
    retry: function(a) {
        return this._onBeforeManualRetry(a) ? (this._handler.retry(a), !0) : !1
    },
    cancel: function(a) {
        this._handler.cancel(a)
    },
    reset: function() {
        this.log("Resetting uploader..."), this._handler.reset(), this._filesInProgress = 0, this._storedFileIds = [], this._autoRetries = [], this._retryTimeouts = [], this._preventRetries = [], this._button.reset()
    },
    _createUploadButton: function(a) {
        var b = this, c = new qq.UploadButton({
            element: a,
            multiple: this._options.multiple && qq.UploadHandlerXhr.isSupported(),
            acceptFiles: this._options.validation.acceptFiles,
            onChange: function(a) {
                b._onInputChange(a)
            }
        });
        return this.addDisposer(function() {
            c.dispose()
        }), c
    },
    _createUploadHandler: function() {
        var a, b = this;
        a = qq.UploadHandlerXhr.isSupported() ? "UploadHandlerXhr" : "UploadHandlerForm";
        var c = new qq[a]({
            debug: this._options.debug,
            endpoint: this._options.request.endpoint,
            forceMultipart: this._options.request.forceMultipart,
            maxConnections: this._options.maxConnections,
            customHeaders: this._options.request.customHeaders,
            inputName: this._options.request.inputName,
            demoMode: this._options.demoMode,
            log: this.log,
            onProgress: function(a, c, d, e) {
                b._onProgress(a, c, d, e), b._options.callbacks.onProgress(a, c, d, e)
            },
            onComplete: function(a, c, d, e) {
                b._onComplete(a, c, d, e), b._options.callbacks.onComplete(a, c, d)
            },
            onCancel: function(a, c) {
                b._onCancel(a, c), b._options.callbacks.onCancel(a, c)
            },
            onUpload: function(a, c, d) {
                b._onUpload(a, c, d), b._options.callbacks.onUpload(a, c, d)
            },
            onAutoRetry: function(a, c, d, e) {
                return b._preventRetries[a] = d[b._options.retry.preventRetryResponseProperty], b._shouldAutoRetry(a, c, d) ? (b._maybeParseAndSendUploadError(a, c, d, e), b._options.callbacks.onAutoRetry(a, c, b._autoRetries[a] + 1), b._onBeforeAutoRetry(a, c), b._retryTimeouts[a] = setTimeout(function() {
                    b._onAutoRetry(a, c, d)
                }, 1e3 * b._options.retry.autoAttemptDelay), !0) : !1
            }
        });
        return c
    },
    _preventLeaveInProgress: function() {
        var a = this;
        this._attach(window, "beforeunload", function(b) {
            if (a._filesInProgress) {
                var b = b || window.event;
                return b.returnValue = a._options.messages.onLeave, a._options.messages.onLeave
            }
        })
    },
    _onSubmit: function() {
        this._options.autoUpload && this._filesInProgress++
    },
    _onProgress: function() {},
    _onComplete: function(a, b, c, d) {
        this._filesInProgress--, this._maybeParseAndSendUploadError(a, b, c, d)
    },
    _onCancel: function(a) {
        clearTimeout(this._retryTimeouts[a]);
        var b = qq.indexOf(this._storedFileIds, a);
        this._options.autoUpload || b < 0 ? this._filesInProgress-- : this._options.autoUpload || this._storedFileIds.splice(b, 1)
    },
    _onUpload: function() {},
    _onInputChange: function(a) {
        this._handler instanceof qq.UploadHandlerXhr ? this._uploadFileList(a.files) : this._validateFile(a) && this._uploadFile(a), this._button.reset()
    },
    _onBeforeAutoRetry: function(a, b) {
        this.log("Waiting " + this._options.retry.autoAttemptDelay + " seconds before retrying " + b + "...")
    },
    _onAutoRetry: function(a, b) {
        this.log("Retrying " + b + "..."), this._autoRetries[a]++, this._handler.retry(a)
    },
    _shouldAutoRetry: function(a) {
        return !this._preventRetries[a] && this._options.retry.enableAuto ? (void 0 === this._autoRetries[a] && (this._autoRetries[a] = 0), this._autoRetries[a] < this._options.retry.maxAutoAttempts) : !1
    },
    _onBeforeManualRetry: function(a) {
        if (this._preventRetries[a])
            return this.log("Retries are forbidden for id " + a, "warn"), !1;
        if (this._handler.isValid(a)) {
            var b = this._handler.getName(a);
            return this._options.callbacks.onManualRetry(a, b)===!1?!1 : (this.log("Retrying upload for '" + b + "' (id: " + a + ")..."), this._filesInProgress++, !0)
        }
        return this.log("'" + a + "' is not a valid file ID", "error"), !1
    },
    _maybeParseAndSendUploadError: function(a, b, c, d) {
        if (!c.success)
            if (d && 200 !== d.status&&!c.error)
                this._options.callbacks.onError(a, b, "XHR returned response code " + d.status);
            else {
                var e = c.error ? c.error: "Upload failure reason unknown";
                this._options.callbacks.onError(a, b, e)
            }
    },
    _uploadFileList: function(a) {
        var b, c, d;
        if (b = this._getValidationDescriptors(a), b.length > 1 && (d = this._options.callbacks.onValidate(b)===!1), !d)
            if (a.length > 0) {
                for (c = 0; c < a.length; c++)
                    if (this._validateFile(a[c]))
                        this._uploadFile(a[c]);
                    else if (this._options.validation.stopOnFirstInvalidFile)
                        return 
            } else 
                this._error("noFilesError", "")
    },
    _uploadFile: function(a) {
        var b = this._handler.add(a), c = this._handler.getName(b);
        this._options.callbacks.onSubmit(b, c)!==!1 && (this._onSubmit(b, c), this._options.autoUpload ? this._handler.upload(b, this._options.request.params) : this._storeFileForLater(b))
    },
    _storeFileForLater: function(a) {
        this._storedFileIds.push(a)
    },
    _validateFile: function(a) {
        var b, c, d;
        return b = this._getValidationDescriptor(a), c = b.name, d = b.size, this._options.callbacks.onValidate([b])===!1?!1 : this._isAllowedExtension(c) ? 0 === d ? (this._error("emptyError", c), !1) : d && this._options.validation.sizeLimit && d > this._options.validation.sizeLimit ? (this._error("sizeError", c), !1) : d && d < this._options.validation.minSizeLimit ? (this._error("minSizeError", c), !1) : !0 : (this._error("typeError", c), !1)
    },
    _error: function(a, b) {
        function c(a, b) {
            d = d.replace(a, b)
        }
        var d = this._options.messages[a], e = this._options.validation.allowedExtensions.join(", ");
        return c("{file}", this._formatFileName(b)), c("{extensions}", e), c("{sizeLimit}", this._formatSize(this._options.validation.sizeLimit)), c("{minSizeLimit}", this._formatSize(this._options.validation.minSizeLimit)), this._options.callbacks.onError(null, b, d), d
    },
    _formatFileName: function(a) {
        return a.length > 33 && (a = a.slice(0, 19) + "..." + a.slice( - 13)), a
    },
    _isAllowedExtension: function(a) {
        var b =- 1 !== a.indexOf(".") ? a.replace(/.*[.]/, "").toLowerCase() : "", c = this._options.validation.allowedExtensions;
        if (!c.length)
            return !0;
        for (var d = 0; d < c.length; d++)
            if (c[d].toLowerCase() == b)
                return !0;
        return !1
    },
    _formatSize: function(a) {
        var b =- 1;
        do 
            a/=1024, b++;
        while (a > 99);
        return Math.max(a, .1).toFixed(1) + ["kB", "MB", "GB", "TB", "PB", "EB"][b]
    },
    _wrapCallbacks: function() {
        var a, b;
        a = this, b = function(b, c, d) {
            try {
                return c.apply(a, d)
            } catch (e) {
                a.log("Caught exception in '" + b + "' callback - " + e, "error")
            }
        };
        for (var c in this._options.callbacks)
            !function() {
                var d = a._options.callbacks[c];
                a._options.callbacks[c] = function() {
                    return b(c, d, arguments)
                }
            }()
    },
    _parseFileName: function(a) {
        var b;
        return b = a.value ? a.value.replace(/.*(\/|\\)/, "") : null !== a.fileName && void 0 !== a.fileName ? a.fileName : a.name
    },
    _parseFileSize: function(a) {
        var b;
        return a.value || (b = null !== a.fileSize && void 0 !== a.fileSize ? a.fileSize : a.size), b
    },
    _getValidationDescriptor: function(a) {
        var b, c, d;
        return d = {}, b = this._parseFileName(a), c = this._parseFileSize(a), d.name = b, c && (d.size = c), d
    },
    _getValidationDescriptors: function(a) {
        var b, c;
        for (c = [], b = 0; b < a.length; b++)
            c.push(a[b]);
        return c
    }
}, qq.FineUploader = function(a) {
    qq.FineUploaderBasic.apply(this, arguments), qq.extend(this._options, {
        element: null,
        listElement: null,
        dragAndDrop: {
            extraDropzones: [],
            hideDropzones: !0,
            disableDefaultDropzone: !1
        },
        text: {
            uploadButton: "Upload a file",
            cancelButton: "Cancel",
            retryButton: "Retry",
            failUpload: "Upload failed",
            dragZone: "Drop files here to upload",
            formatProgress: "{percent}% of {total_size}",
            waitingForResponse: "Processing..."
        },
        template: '<div class="qq-uploader">' + (this._options.dragAndDrop && this._options.dragAndDrop.disableDefaultDropzone ? "" : '<div class="qq-upload-drop-area"><span>{dragZoneText}</span></div>') + (this._options.button ? "" : '<div class="qq-upload-button"><div>{uploadButtonText}</div></div>') + (this._options.listElement ? "" : '<ul class="qq-upload-list"></ul>') + "</div>",
        fileTemplate: '<li><div class="qq-progress-bar"></div><span class="qq-upload-spinner"></span><span class="qq-upload-finished"></span><span class="qq-upload-file"></span><span class="qq-upload-size"></span><a class="qq-upload-cancel" href="#">{cancelButtonText}</a><a class="qq-upload-retry" href="#">{retryButtonText}</a><span class="qq-upload-status-text">{statusText}</span></li>',
        classes: {
            button: "qq-upload-button",
            drop: "qq-upload-drop-area",
            dropActive: "qq-upload-drop-area-active",
            dropDisabled: "qq-upload-drop-area-disabled",
            list: "qq-upload-list",
            progressBar: "qq-progress-bar",
            file: "qq-upload-file",
            spinner: "qq-upload-spinner",
            finished: "qq-upload-finished",
            retrying: "qq-upload-retrying",
            retryable: "qq-upload-retryable",
            size: "qq-upload-size",
            cancel: "qq-upload-cancel",
            retry: "qq-upload-retry",
            statusText: "qq-upload-status-text",
            success: "qq-upload-success",
            fail: "qq-upload-fail",
            successIcon: null,
            failIcon: null
        },
        failedUploadTextDisplay: {
            mode: "default",
            maxChars: 50,
            responseProperty: "error",
            enableTooltip: !0
        },
        messages: {
            tooManyFilesError: "You may only drop one file"
        },
        retry: {
            showAutoRetryNote: !0,
            autoRetryNote: "Retrying {retryNum}/{maxAuto}...",
            showButton: !1
        },
        showMessage: function(a) {
            alert(a)
        }
    }, !0), qq.extend(this._options, a, !0), this._wrapCallbacks(), this._options.template = this._options.template.replace(/\{dragZoneText\}/g, this._options.text.dragZone), this._options.template = this._options.template.replace(/\{uploadButtonText\}/g, this._options.text.uploadButton), this._options.fileTemplate = this._options.fileTemplate.replace(/\{cancelButtonText\}/g, this._options.text.cancelButton), this._options.fileTemplate = this._options.fileTemplate.replace(/\{retryButtonText\}/g, this._options.text.retryButton), this._options.fileTemplate = this._options.fileTemplate.replace(/\{statusText\}/g, ""), this._element = this._options.element, this._element.innerHTML = this._options.template, this._listElement = this._options.listElement || this._find(this._element, "list"), this._classes = this._options.classes, this._button || (this._button = this._createUploadButton(this._find(this._element, "button"))), this._bindCancelAndRetryEvents(), this._setupDragDrop()
}, qq.extend(qq.FineUploader.prototype, qq.FineUploaderBasic.prototype), qq.extend(qq.FineUploader.prototype, {
    clearStoredFiles: function() {
        qq.FineUploaderBasic.prototype.clearStoredFiles.apply(this, arguments), this._listElement.innerHTML = ""
    },
    addExtraDropzone: function(a) {
        this._setupExtraDropzone(a)
    },
    removeExtraDropzone: function(a) {
        var b = this._options.dragAndDrop.extraDropzones;
        for (var c in b)
            if (b[c] === a)
                return this._options.dragAndDrop.extraDropzones.splice(c, 1)
        },
    getItemByFileId: function(a) {
        for (var b = this._listElement.firstChild; b;) {
            if (b.qqFileId == a)
                return b;
            b = b.nextSibling
        }
    },
    reset: function() {
        qq.FineUploaderBasic.prototype.reset.apply(this, arguments), this._element.innerHTML = this._options.template, this._listElement = this._options.listElement || this._find(this._element, "list"), this._options.button || (this._button = this._createUploadButton(this._find(this._element, "button"))), this._bindCancelAndRetryEvents(), this._setupDragDrop()
    },
    _leaving_document_out: function(a) {
        return (qq.chrome() || qq.safari() && qq.windows()) && 0 == a.clientX && 0 == a.clientY || qq.firefox()&&!a.relatedTarget
    },
    _storeFileForLater: function(a) {
        qq.FineUploaderBasic.prototype._storeFileForLater.apply(this, arguments);
        var b = this.getItemByFileId(a);
        qq(this._find(b, "spinner")).hide()
    },
    _find: function(a, b) {
        var c = qq(a).getByClass(this._options.classes[b])[0];
        if (!c)
            throw new Error("element not found " + b);
        return c
    },
    _setupExtraDropzone: function(a) {
        this._options.dragAndDrop.extraDropzones.push(a), this._setupDropzone(a)
    },
    _setupDropzone: function(a) {
        var b = this, c = new qq.UploadDropZone({
            element: a,
            onEnter: function(c) {
                qq(a).addClass(b._classes.dropActive), c.stopPropagation()
            },
            onLeave: function() {},
            onLeaveNotDescendants: function() {
                qq(a).removeClass(b._classes.dropActive)
            },
            onDrop: function(c) {
                b._options.dragAndDrop.hideDropzones && qq(a).hide(), qq(a).removeClass(b._classes.dropActive), c.dataTransfer.files.length > 1&&!b._options.multiple ? b._error("tooManyFilesError", "") : b._uploadFileList(c.dataTransfer.files)
            }
        });
        this.addDisposer(function() {
            c.dispose()
        }), this._options.dragAndDrop.hideDropzones && qq(a).hide()
    },
    _setupDragDrop: function() {
        var a, b;
        a = this, this._options.dragAndDrop.disableDefaultDropzone || (b = this._find(this._element, "drop"), this._options.dragAndDrop.extraDropzones.push(b));
        var c, d = this._options.dragAndDrop.extraDropzones;
        for (c = 0; c < d.length; c++)
            this._setupDropzone(d[c]);
        this._options.dragAndDrop.disableDefaultDropzone || qq.ie()&&!qq.ie10() || this._attach(document, "dragenter", function() {
            if (!qq(b).hasClass(a._classes.dropDisabled))
                for (b.style.display = "block", c = 0; c < d.length; c++)
                    d[c].style.display = "block"
        }), this._attach(document, "dragleave", function(b) {
            if (a._options.dragAndDrop.hideDropzones && qq.FineUploader.prototype._leaving_document_out(b))
                for (c = 0; c < d.length; c++)
                    qq(d[c]).hide()
        }), qq(document).attach("drop", function(b) {
            if (a._options.dragAndDrop.hideDropzones)
                for (c = 0; c < d.length; c++)
                    qq(d[c]).hide();
            b.preventDefault()
        })
    },
    _onSubmit: function(a, b) {
        qq.FineUploaderBasic.prototype._onSubmit.apply(this, arguments), this._addToList(a, b)
    },
    _onProgress: function(a, b, c, d) {
        qq.FineUploaderBasic.prototype._onProgress.apply(this, arguments);
        var e, f, g, h, i, j;
        e = this.getItemByFileId(a), f = this._find(e, "progressBar"), h = Math.round(c / d * 100), c === d ? (i = this._find(e, "cancel"), qq(i).hide(), qq(f).hide(), qq(this._find(e, "statusText")).setText(this._options.text.waitingForResponse), g = this._formatSize(d)) : (g = this._formatProgress(c, d), qq(f).css({
            display: "block"
        })), qq(f).css({
            width: h + "%"
        }), j = this._find(e, "size"), qq(j).css({
            display: "inline"
        }), qq(j).setText(g)
    },
    _onComplete: function(a, b, c) {
        qq.FineUploaderBasic.prototype._onComplete.apply(this, arguments);
        var d = this.getItemByFileId(a);
        qq(this._find(d, "statusText")).clearText(), qq(d).removeClass(this._classes.retrying), qq(this._find(d, "progressBar")).hide(), (!this._options.disableCancelForFormUploads || qq.UploadHandlerXhr.isSupported()) && qq(this._find(d, "cancel")).hide(), qq(this._find(d, "spinner")).hide(), c.success ? (qq(d).addClass(this._classes.success), this._classes.successIcon && (this._find(d, "finished").style.display = "inline-block", qq(d).addClass(this._classes.successIcon))) : (qq(d).addClass(this._classes.fail), this._classes.failIcon && (this._find(d, "finished").style.display = "inline-block", qq(d).addClass(this._classes.failIcon)), this._options.retry.showButton&&!this._preventRetries[a] && qq(d).addClass(this._classes.retryable), this._controlFailureTextDisplay(d, c))
    },
    _onUpload: function(a) {
        qq.FineUploaderBasic.prototype._onUpload.apply(this, arguments);
        var b = this.getItemByFileId(a);
        this._showSpinner(b)
    },
    _onBeforeAutoRetry: function(a) {
        var b, c, d, e, f, g;
        qq.FineUploaderBasic.prototype._onBeforeAutoRetry.apply(this, arguments), b = this.getItemByFileId(a), c = this._find(b, "progressBar"), this._showCancelLink(b), c.style.width = 0, qq(c).hide(), this._options.retry.showAutoRetryNote && (d = this._find(b, "statusText"), e = this._autoRetries[a] + 1, f = this._options.retry.maxAutoAttempts, g = this._options.retry.autoRetryNote.replace(/\{retryNum\}/g, e), g = g.replace(/\{maxAuto\}/g, f), qq(d).setText(g), 1 === e && qq(b).addClass(this._classes.retrying))
    },
    _onBeforeManualRetry: function(a) {
        if (qq.FineUploaderBasic.prototype._onBeforeManualRetry.apply(this, arguments)) {
            var b = this.getItemByFileId(a);
            return this._find(b, "progressBar").style.width = 0, qq(b).removeClass(this._classes.fail), this._showSpinner(b), this._showCancelLink(b), !0
        }
        return !1
    },
    _addToList: function(a, b) {
        var c = qq.toElement(this._options.fileTemplate);
        if (this._options.disableCancelForFormUploads&&!qq.UploadHandlerXhr.isSupported()) {
            var d = this._find(c, "cancel");
            qq(d).remove()
        }
        c.qqFileId = a;
        var e = this._find(c, "file");
        qq(e).setText(this._formatFileName(b)), qq(this._find(c, "size")).hide(), this._options.multiple || this._clearList(), this._listElement.appendChild(c)
    },
    _clearList: function() {
        this._listElement.innerHTML = "", this.clearStoredFiles()
    },
    _bindCancelAndRetryEvents: function() {
        var a = this, b = this._listElement;
        this._attach(b, "click", function(b) {
            b = b || window.event;
            var c = b.target || b.srcElement;
            if (qq(c).hasClass(a._classes.cancel) || qq(c).hasClass(a._classes.retry)) {
                qq.preventDefault(b);
                for (var d = c.parentNode; void 0 == d.qqFileId;)
                    d = c = c.parentNode;
                qq(c).hasClass(a._classes.cancel) ? (a.cancel(d.qqFileId), qq(d).remove()) : (qq(d).removeClass(a._classes.retryable), a.retry(d.qqFileId))
            }
        })
    },
    _formatProgress: function(a, b) {
        function c(a, b) {
            d = d.replace(a, b)
        }
        var d = this._options.text.formatProgress;
        return c("{percent}", Math.round(a / b * 100)), c("{total_size}", this._formatSize(b)), d
    },
    _controlFailureTextDisplay: function(a, b) {
        var c, d, e, f, g;
        c = this._options.failedUploadTextDisplay.mode, d = this._options.failedUploadTextDisplay.maxChars, e = this._options.failedUploadTextDisplay.responseProperty, "custom" === c ? (f = b[e], f ? f.length > d && (g = f.substring(0, d) + "...") : (f = this._options.text.failUpload, this.log("'" + e + "' is not a valid property on the server response.", "warn")), qq(this._find(a, "statusText")).setText(g || f), this._options.failedUploadTextDisplay.enableTooltip && this._showTooltip(a, f)) : "default" === c ? qq(this._find(a, "statusText")).setText(this._options.text.failUpload) : "none" !== c && this.log("failedUploadTextDisplay.mode value of '" + c + "' is not valid", "warn")
    },
    _showTooltip: function(a, b) {
        a.title = b
    },
    _showSpinner: function(a) {
        var b = this._find(a, "spinner");
        b.style.display = "inline-block"
    },
    _showCancelLink: function(a) {
        if (!this._options.disableCancelForFormUploads || qq.UploadHandlerXhr.isSupported()) {
            var b = this._find(a, "cancel");
            b.style.display = "inline"
        }
    },
    _error: function() {
        var a = qq.FineUploaderBasic.prototype._error.apply(this, arguments);
        this._options.showMessage(a)
    }
}), qq.UploadDropZone = function(a) {
    this._options = {
        element: null,
        onEnter: function() {},
        onLeave: function() {},
        onLeaveNotDescendants: function() {},
        onDrop: function() {}
    }, qq.extend(this._options, a), qq.extend(this, qq.DisposeSupport), this._element = this._options.element, this._disableDropOutside(), this._attachEvents()
}, qq.UploadDropZone.prototype = {
    _dragover_should_be_canceled: function() {
        return qq.safari() || qq.firefox() && qq.windows()
    },
    _disableDropOutside: function() {
        qq.UploadDropZone.dropOutsideDisabled || (this._dragover_should_be_canceled ? qq(document).attach("dragover", function(a) {
            a.preventDefault()
        }) : qq(document).attach("dragover", function(a) {
            a.dataTransfer && (a.dataTransfer.dropEffect = "none", a.preventDefault())
        }), qq.UploadDropZone.dropOutsideDisabled=!0)
    },
    _attachEvents: function() {
        var a = this;
        a._attach(a._element, "dragover", function(b) {
            if (a._isValidFileDrag(b)) {
                var c = qq.ie() ? null: b.dataTransfer.effectAllowed;
                b.dataTransfer.dropEffect = "move" == c || "linkMove" == c ? "move" : "copy", b.stopPropagation(), b.preventDefault()
            }
        }), a._attach(a._element, "dragenter", function(b) {
            a._isValidFileDrag(b) && a._options.onEnter(b)
        }), a._attach(a._element, "dragleave", function(b) {
            if (a._isValidFileDrag(b)) {
                a._options.onLeave(b);
                var c = document.elementFromPoint(b.clientX, b.clientY);
                qq(this).contains(c) || a._options.onLeaveNotDescendants(b)
            }
        }), a._attach(a._element, "drop", function(b) {
            a._isValidFileDrag(b) && (b.preventDefault(), a._options.onDrop(b))
        })
    },
    _isValidFileDrag: function(a) {
        if (qq.ie()&&!qq.ie10())
            return !1;
        var b = a.dataTransfer, c = qq.safari(), d = qq.ie10()?!0 : "none" != b.effectAllowed;
        return b && d && (b.files ||!c && b.types.contains && b.types.contains("Files"))
    }
}, qq.UploadHandlerAbstract = function(a) {
    this._options = {
        debug: !1,
        endpoint: "/upload.php",
        maxConnections: 999,
        log: function() {},
        onProgress: function() {},
        onComplete: function() {},
        onCancel: function() {},
        onUpload: function() {},
        onAutoRetry: function() {}
    }, qq.extend(this._options, a), this._queue = [], this._params = [], this.log = this._options.log
}, qq.UploadHandlerAbstract.prototype = {
    add: function() {},
    upload: function(a, b) {
        var c = this._queue.push(a), d = {};
        qq.extend(d, b), this._params[a] = d, c <= this._options.maxConnections && this._upload(a, this._params[a])
    },
    retry: function(a) {
        var b = qq.indexOf(this._queue, a);
        b >= 0 ? this._upload(a, this._params[a]) : this.upload(a, this._params[a])
    },
    cancel: function(a) {
        this.log("Cancelling " + a), this._cancel(a), this._dequeue(a)
    },
    cancelAll: function() {
        for (var a = 0; a < this._queue.length; a++)
            this._cancel(this._queue[a]);
        this._queue = []
    },
    getName: function() {},
    getSize: function() {},
    getQueue: function() {
        return this._queue
    },
    reset: function() {
        this.log("Resetting upload handler"), this._queue = [], this._params = []
    },
    _upload: function() {},
    _cancel: function() {},
    _dequeue: function(a) {
        var b = qq.indexOf(this._queue, a);
        this._queue.splice(b, 1);
        var c = this._options.maxConnections;
        if (this._queue.length >= c && b < c) {
            var d = this._queue[c - 1];
            this._upload(d, this._params[d])
        }
    },
    isValid: function() {}
}, qq.UploadHandlerForm = function() {
    qq.UploadHandlerAbstract.apply(this, arguments), this._inputs = {}, this._detach_load_events = {}
}, qq.extend(qq.UploadHandlerForm.prototype, qq.UploadHandlerAbstract.prototype), qq.extend(qq.UploadHandlerForm.prototype, {
    add: function(a) {
        a.setAttribute("name", this._options.inputName);
        var b = "qq-upload-handler-iframe" + qq.getUniqueId();
        return this._inputs[b] = a, a.parentNode && qq(a).remove(), b
    },
    getName: function(a) {
        return this._inputs[a].value.replace(/.*(\/|\\)/, "")
    },
    isValid: function(a) {
        return void 0 !== this._inputs[a]
    },
    reset: function() {
        qq.UploadHandlerAbstract.prototype.reset.apply(this, arguments), this._inputs = {}, this._detach_load_events = {}
    },
    _cancel: function(a) {
        this._options.onCancel(a, this.getName(a)), delete this._inputs[a], delete this._detach_load_events[a];
        var b = document.getElementById(a);
        b && (b.setAttribute("src", "javascript:false;"), qq(b).remove())
    },
    _upload: function(a, b) {
        this._options.onUpload(a, this.getName(a), !1);
        var c = this._inputs[a];
        if (!c)
            throw new Error("file with passed id was not added, or already uploaded or cancelled");
        var d = this.getName(a);
        b[this._options.inputName] = d;
        var e = this._createIframe(a), f = this._createForm(e, b);
        f.appendChild(c);
        var g = this;
        return this._attachLoadEvent(e, function() {
            g.log("iframe loaded");
            var b = g._getIframeContentJSON(e);
            setTimeout(function() {
                g._detach_load_events[a](), delete g._detach_load_events[a], qq(e).remove()
            }, 1), (b.success ||!g._options.onAutoRetry(a, d, b)) && (g._options.onComplete(a, d, b), g._dequeue(a))
        }), this.log("Sending upload request for " + a), f.submit(), qq(f).remove(), a
    },
    _attachLoadEvent: function(a, b) {
        var c = this;
        this._detach_load_events[a.id] = qq(a).attach("load", function() {
            if (c.log("Received response for " + a.id), a.parentNode) {
                try {
                    if (a.contentDocument && a.contentDocument.body && "false" == a.contentDocument.body.innerHTML)
                        return 
                } catch (d) {
                    c.log("Error when attempting to access iframe during handling of upload response (" + d + ")", "error")
                }
                b()
            }
        })
    },
    _getIframeContentJSON: function(iframe) {
        try {
            var doc = iframe.contentDocument ? iframe.contentDocument: iframe.contentWindow.document, response, innerHTML = doc.body.innerHTML;
            this.log("converting iframe's innerHTML to JSON"), this.log("innerHTML = " + innerHTML), innerHTML && innerHTML.match(/^<pre/i) && (innerHTML = doc.body.firstChild.firstChild.nodeValue), response = eval("(" + innerHTML + ")")
        } catch (error) {
            this.log("Error when attempting to parse form upload response (" + error + ")", "error"), response = {
                success: !1
            }
        }
        return response
    },
    _createIframe: function(a) {
        var b = qq.toElement('<iframe src="javascript:false;" name="' + a + '" />');
        return b.setAttribute("id", a), b.style.display = "none", document.body.appendChild(b), b
    },
    _createForm: function(a, b) {
        var c = this._options.demoMode ? "GET": "POST", d = qq.toElement('<form method="' + c + '" enctype="multipart/form-data"></form>'), e = qq.obj2url(b, this._options.endpoint);
        return d.setAttribute("action", e), d.setAttribute("target", a.name), d.style.display = "none", document.body.appendChild(d), d
    }
}), qq.UploadHandlerXhr = function() {
    qq.UploadHandlerAbstract.apply(this, arguments), this._files = [], this._xhrs = [], this._loaded = []
}, qq.UploadHandlerXhr.isSupported = function() {
    var a = document.createElement("input");
    return a.type = "file", "multiple"in a && "undefined" != typeof File && "undefined" != typeof FormData && "undefined" != typeof(new XMLHttpRequest).upload
}, qq.extend(qq.UploadHandlerXhr.prototype, qq.UploadHandlerAbstract.prototype), qq.extend(qq.UploadHandlerXhr.prototype, {
    add: function(a) {
        if (!(a instanceof File))
            throw new Error("Passed obj in not a File (in qq.UploadHandlerXhr)");
        return this._files.push(a) - 1
    },
    getName: function(a) {
        var b = this._files[a];
        return null !== b.fileName && void 0 !== b.fileName ? b.fileName : b.name
    },
    getSize: function(a) {
        var b = this._files[a];
        return null != b.fileSize ? b.fileSize : b.size
    },
    getLoaded: function(a) {
        return this._loaded[a] || 0
    },
    isValid: function(a) {
        return void 0 !== this._files[a]
    },
    reset: function() {
        qq.UploadHandlerAbstract.prototype.reset.apply(this, arguments), this._files = [], this._xhrs = [], this._loaded = []
    },
    _upload: function(a, b) {
        this._options.onUpload(a, this.getName(a), !0);
        {
            var c = this._files[a], d = this.getName(a);
            this.getSize(a)
        }
        this._loaded[a] = 0;
        var e = this._xhrs[a] = new XMLHttpRequest, f = this;
        e.upload.onprogress = function(b) {
            b.lengthComputable && (f._loaded[a] = b.loaded, f._options.onProgress(a, d, b.loaded, b.total))
        }, e.onreadystatechange = function() {
            4 == e.readyState && f._onComplete(a, e)
        }, b = b || {}, b[this._options.inputName] = d;
        var g = qq.obj2url(b, this._options.endpoint), h = this._options.demoMode ? "GET": "POST";
        if (e.open(h, g, !0), e.setRequestHeader("X-Requested-With", "XMLHttpRequest"), e.setRequestHeader("X-File-Name", encodeURIComponent(d)), e.setRequestHeader("Cache-Control", "no-cache"), this._options.forceMultipart) {
            var i = new FormData;
            i.append(this._options.inputName, c), c = i
        } else 
            e.setRequestHeader("Content-Type", "application/octet-stream"), e.setRequestHeader("X-Mime-Type", c.type);
        for (key in this._options.customHeaders)
            e.setRequestHeader(key, this._options.customHeaders[key]);
        this.log("Sending upload request for " + a), e.send(c)
    },
    _onComplete: function(id, xhr) {
        "use strict";
        if (this._files[id]) {
            var name = this.getName(id), size = this.getSize(id), response;
            this._options.onProgress(id, name, size, size), this.log("xhr - server response received for " + id), this.log("responseText = " + xhr.responseText);
            try {
                response = "function" === typeof JSON.parse ? JSON.parse(xhr.responseText) : eval("(" + xhr.responseText + ")")
            } catch (error) {
                this.log("Error when attempting to parse xhr response text (" + error + ")", "error"), response = {}
            }(200 === xhr.status && response.success ||!this._options.onAutoRetry(id, name, response, xhr)) && (this._options.onComplete(id, name, response, xhr), this._xhrs[id] = null, this._dequeue(id))
        }
    },
    _cancel: function(a) {
        this._options.onCancel(a, this.getName(a)), this._files[a] = null, this._xhrs[a] && (this._xhrs[a].abort(), this._xhrs[a] = null)
    }
}), function() {
    var a = qq.UploadHandlerForm.prototype._createForm;
    qq.UploadHandlerForm.prototype._createForm = function() {
        var b = a.apply(this, arguments), c = $('<input type="hidden" name="csrfmiddlewaretoken" value="' + P.csrf.getCSRFToken() + '">');
        return b.appendChild(c.get(0)), b
    }
}(), function(a) {
    function b(a) {
        if (a in l.style)
            return a;
        var b = ["Moz", "Webkit", "O", "ms"], c = a.charAt(0).toUpperCase() + a.substr(1);
        if (a in l.style)
            return a;
        for (var d = 0; d < b.length; ++d) {
            var e = b[d] + c;
            if (e in l.style)
                return e
        }
    }
    function c() {
        return l.style[m.transform] = "", l.style[m.transform] = "rotateY(90deg)", "" !== l.style[m.transform]
    }
    function d(a) {
        return "string" === typeof a && this.parse(a), this
    }
    function e(a, b, c) {
        b===!0 ? a.queue(c) : b ? a.queue(b, c) : c()
    }
    function f(b) {
        var c = [];
        return a.each(b, function(b) {
            b = a.camelCase(b), b = a.transit.propertyMap[b] || a.cssProps[b] || b, b = i(b), m[b] && (b = i(m[b])), - 1 === a.inArray(b, c) && c.push(b)
        }), c
    }
    function g(b, c, d, e) {
        var g = f(b);
        a.cssEase[d] && (d = a.cssEase[d]);
        var h = "" + k(c) + " " + d;
        parseInt(e, 10) > 0 && (h += " " + k(e));
        var i = [];
        return a.each(g, function(a, b) {
            i.push(b + " " + h)
        }), i.join(", ")
    }
    function h(b, c) {
        c || (a.cssNumber[b]=!0), a.transit.propertyMap[b] = m.transform, a.cssHooks[b] = {
            get: function(c) {
                var d = a(c).css("transit:transform");
                return d.get(b)
            },
            set: function(c, d) {
                var e = a(c).css("transit:transform");
                e.setFromString(b, d), a(c).css({
                    "transit:transform": e
                })
            }
        }
    }
    function i(a) {
        return a.replace(/([A-Z])/g, function(a) {
            return "-" + a.toLowerCase()
        })
    }
    function j(a, b) {
        return "string" !== typeof a || a.match(/^[\-0-9\.]+$/) ? "" + a + b : a
    }
    function k(b) {
        var c = b;
        return "string" !== typeof c || c.match(/^[\-0-9\.]+/) || (c = a.fx.speeds[c] || a.fx.speeds._default), j(c, "ms")
    }
    a.transit = {
        version: "0.9.9",
        propertyMap: {
            marginLeft: "margin",
            marginRight: "margin",
            marginBottom: "margin",
            marginTop: "margin",
            paddingLeft: "padding",
            paddingRight: "padding",
            paddingBottom: "padding",
            paddingTop: "padding"
        },
        enabled: !0,
        useTransitionEnd: !1
    };
    var l = document.createElement("div"), m = {}, n = navigator.userAgent.toLowerCase().indexOf("chrome")>-1;
    m.transition = b("transition"), m.transitionDelay = b("transitionDelay"), m.transform = b("transform"), m.transformOrigin = b("transformOrigin"), m.filter = b("Filter"), m.transform3d = c();
    var o = {
        transition: "transitionEnd",
        MozTransition: "transitionend",
        OTransition: "oTransitionEnd",
        WebkitTransition: "webkitTransitionEnd",
        msTransition: "MSTransitionEnd"
    }, p = m.transitionEnd = o[m.transition] || null;
    for (var q in m)
        m.hasOwnProperty(q) && "undefined" === typeof a.support[q] && (a.support[q] = m[q]);
    l = null, a.cssEase = {
        _default: "ease",
        "in": "ease-in",
        out: "ease-out",
        "in-out": "ease-in-out",
        snap: "cubic-bezier(0,1,.5,1)",
        easeOutCubic: "cubic-bezier(.215,.61,.355,1)",
        easeInOutCubic: "cubic-bezier(.645,.045,.355,1)",
        easeInCirc: "cubic-bezier(.6,.04,.98,.335)",
        easeOutCirc: "cubic-bezier(.075,.82,.165,1)",
        easeInOutCirc: "cubic-bezier(.785,.135,.15,.86)",
        easeInExpo: "cubic-bezier(.95,.05,.795,.035)",
        easeOutExpo: "cubic-bezier(.19,1,.22,1)",
        easeInOutExpo: "cubic-bezier(1,0,0,1)",
        easeInQuad: "cubic-bezier(.55,.085,.68,.53)",
        easeOutQuad: "cubic-bezier(.25,.46,.45,.94)",
        easeInOutQuad: "cubic-bezier(.455,.03,.515,.955)",
        easeInQuart: "cubic-bezier(.895,.03,.685,.22)",
        easeOutQuart: "cubic-bezier(.165,.84,.44,1)",
        easeInOutQuart: "cubic-bezier(.77,0,.175,1)",
        easeInQuint: "cubic-bezier(.755,.05,.855,.06)",
        easeOutQuint: "cubic-bezier(.23,1,.32,1)",
        easeInOutQuint: "cubic-bezier(.86,0,.07,1)",
        easeInSine: "cubic-bezier(.47,0,.745,.715)",
        easeOutSine: "cubic-bezier(.39,.575,.565,1)",
        easeInOutSine: "cubic-bezier(.445,.05,.55,.95)",
        easeInBack: "cubic-bezier(.6,-.28,.735,.045)",
        easeOutBack: "cubic-bezier(.175, .885,.32,1.275)",
        easeInOutBack: "cubic-bezier(.68,-.55,.265,1.55)"
    }, a.cssHooks["transit:transform"] = {
        get: function(b) {
            return a(b).data("transform") || new d
        },
        set: function(b, c) {
            var e = c;
            e instanceof d || (e = new d(e)), b.style[m.transform] = "WebkitTransform" !== m.transform || n ? e.toString() : e.toString(!0), a(b).data("transform", e)
        }
    }, a.cssHooks.transform = {
        set: a.cssHooks["transit:transform"].set
    }, a.cssHooks.filter = {
        get: function(a) {
            return a.style[m.filter]
        },
        set: function(a, b) {
            a.style[m.filter] = b
        }
    }, a.fn.jquery < "1.8" && (a.cssHooks.transformOrigin = {
        get: function(a) {
            return a.style[m.transformOrigin]
        },
        set: function(a, b) {
            a.style[m.transformOrigin] = b
        }
    }, a.cssHooks.transition = {
        get: function(a) {
            return a.style[m.transition]
        },
        set: function(a, b) {
            a.style[m.transition] = b
        }
    }), h("scale"), h("translate"), h("rotate"), h("rotateX"), h("rotateY"), h("rotate3d"), h("perspective"), h("skewX"), h("skewY"), h("x", !0), h("y", !0), d.prototype = {
        setFromString: function(a, b) {
            var c = "string" === typeof b ? b.split(","): b.constructor === Array ? b: [b];
            c.unshift(a), d.prototype.set.apply(this, c)
        },
        set: function(a) {
            var b = Array.prototype.slice.apply(arguments, [1]);
            this.setter[a] ? this.setter[a].apply(this, b) : this[a] = b.join(",")
        },
        get: function(a) {
            return this.getter[a] ? this.getter[a].apply(this) : this[a] || 0
        },
        setter: {
            rotate: function(a) {
                this.rotate = j(a, "deg")
            },
            rotateX: function(a) {
                this.rotateX = j(a, "deg")
            },
            rotateY: function(a) {
                this.rotateY = j(a, "deg")
            },
            scale: function(a, b) {
                void 0 === b && (b = a), this.scale = a + "," + b
            },
            skewX: function(a) {
                this.skewX = j(a, "deg")
            },
            skewY: function(a) {
                this.skewY = j(a, "deg")
            },
            perspective: function(a) {
                this.perspective = j(a, "px")
            },
            x: function(a) {
                this.set("translate", a, null)
            },
            y: function(a) {
                this.set("translate", null, a)
            },
            translate: function(a, b) {
                void 0 === this._translateX && (this._translateX = 0), void 0 === this._translateY && (this._translateY = 0), null !== a && void 0 !== a && (this._translateX = j(a, "px")), null !== b && void 0 !== b && (this._translateY = j(b, "px")), this.translate = this._translateX + "," + this._translateY
            }
        },
        getter: {
            x: function() {
                return this._translateX || 0
            },
            y: function() {
                return this._translateY || 0
            },
            scale: function() {
                var a = (this.scale || "1,1").split(",");
                return a[0] && (a[0] = parseFloat(a[0])), a[1] && (a[1] = parseFloat(a[1])), a[0] === a[1] ? a[0] : a
            },
            rotate3d: function() {
                for (var a = (this.rotate3d || "0,0,0,0deg").split(","), b = 0; b <= 3; ++b)
                    a[b] && (a[b] = parseFloat(a[b]));
                return a[3] && (a[3] = j(a[3], "deg")), a
            }
        },
        parse: function(a) {
            var b = this;
            a.replace(/([a-zA-Z0-9]+)\((.*?)\)/g, function(a, c, d) {
                b.setFromString(c, d)
            })
        },
        toString: function(a) {
            var b = [];
            for (var c in this)
                if (this.hasOwnProperty(c)) {
                    if (!m.transform3d && ("rotateX" === c || "rotateY" === c || "perspective" === c || "transformOrigin" === c))
                        continue;
                        "_" !== c[0] && b.push(a && "scale" === c ? c + "3d(" + this[c] + ",1)" : a && "translate" === c ? c + "3d(" + this[c] + ",0)" : c + "(" + this[c] + ")")
                }
            return b.join(" ")
        }
    }, a.fn.transition = a.fn.transit = function(b, c, d, f) {
        var h = this, i = 0, j=!0, l = jQuery.extend(!0, {}, b);
        "function" === typeof c && (f = c, c = void 0), "object" === typeof c && (d = c.easing, i = c.delay || 0, j = c.queue ||!0, f = c.complete, c = c.duration), "function" === typeof d && (f = d, d = void 0), "undefined" !== typeof l.easing && (d = l.easing, delete l.easing), "undefined" !== typeof l.duration && (c = l.duration, delete l.duration), "undefined" !== typeof l.complete && (f = l.complete, delete l.complete), "undefined" !== typeof l.queue && (j = l.queue, delete l.queue), "undefined" !== typeof l.delay && (i = l.delay, delete l.delay), "undefined" === typeof c && (c = a.fx.speeds._default), "undefined" === typeof d && (d = a.cssEase._default), c = k(c);
        var n = g(l, c, d, i), o = a.transit.enabled && m.transition, q = o ? parseInt(c, 10) + parseInt(i, 10): 0;
        if (0 === q) {
            var r = function(a) {
                h.css(l), f && f.apply(h), a && a()
            };
            return e(h, j, r), h
        }
        var s = {}, t = function(c) {
            var d=!1, e = function() {
                d && h.unbind(p, e), q > 0 && h.each(function() {
                    this.style[m.transition] = s[this] || null
                }), "function" === typeof f && f.apply(h), "function" === typeof c && c()
            };
            q > 0 && p && a.transit.useTransitionEnd ? (d=!0, h.bind(p, e)) : window.setTimeout(e, q), h.each(function() {
                q > 0 && (this.style[m.transition] = n), a(this).css(b)
            })
        }, u = function(a) {
            this.offsetWidth = this.offsetWidth, t(a)
        };
        return e(h, j, u), this
    }, a.transit.getTransitionValue = g
}(jQuery), function() {
    $.fn.grid = function(b) {
        var e;
        if ("string" === typeof arguments[0]) {
            var f = arguments;
            this.each(function() {
                var a = c[d(this)];
                a && (e = a[b].apply(a, Array.prototype.slice.call(f, 1)))
            })
        } else 
            this.each(function() {
                b.container = this, c[d(this)] = new a(b)
            });
        return e || this
    };
    var a = function(a) {
        if (this.options = {
            $container: $(a.container),
            children: a.children,
            appendedChildren: [],
            gridSize: a.gridSize,
            gridGutter: a.gridGutter,
            setChildPosition: a.setChildPosition,
            unsetChildPosition: a.unsetChildPosition,
            setChildProperties: a.setChildProperties,
            unsetChildProperties: a.unsetChildProperties,
            maxRows: a.maxRows
        }, this.options.children.length && this.reflowContent(), a.onInitialized) {
            var b = this;
            _.defer(function() {
                a.onInitialized.apply(b)
            })
        }
    };
    a.prototype.reflowContent = function(a) {
        if (!a) {
            var b=!(!this._columns ||!this._columns.length);
            this._createColumns();
            for (var c, d = 0; d < this.options.children.length; d++)
                c = this.options.children[d], b && this._unpositionChild(c, d), this._positionChild(c, d)
        }
        if (this.options.appendedChildren.length) {
            for (var d = 0; d < this.options.appendedChildren.length; d++) {
                var c = this.options.appendedChildren[d];
                this.options.children.push(c), this._positionChild(c, this.options.children.length)
            }
            this.options.appendedChildren = []
        }
        var e = this.getContainerHeight();
        this.options.$container.css({
            height: e
        })
    }, a.prototype.getContainerHeight = function() {
        var a = Math.max.apply(null, _.map(this._columns, function(a) {
            return a.height
        }));
        return a
    }, a.prototype.getFilledHeight = function() {
        var a = Math.min.apply(null, _.map(this._columns, function(a) {
            return a.height
        }));
        return a
    }, a.prototype._createColumns = function() {
        this.options.$container.css({
            width: "auto"
        });
        var a = this._computeColumnCount();
        this.options.$container.css({
            width: this._computeGridWidth(a)
        }), this._columns = [], this._currentColumn = 0;
        for (var c = 0; c < a; c++)
            this._columns[c] = new b(c, this)
    }, a.prototype._computeColumnCount = function() {
        var a = this.options.$container.width(), b = Math.floor(a / (this.options.gridSize + this.options.gridGutter));
        return b
    }, a.prototype._computeGridWidth = function(a) {
        return a * (this.options.gridSize + this.options.gridGutter)
    }, a.prototype.appendChild = function(a, b) {
        if (b)
            this.options.appendedChildren.push(a);
        else {
            this._columns || this._createColumns(), this.options.children.push(a), this._positionChild(a, this.options.children.length);
            var c = this.getContainerHeight();
            this.options.$container.css({
                height: c
            })
        }
    }, a.prototype._positionChild = function(a, b) {
        if (this._columns && this._columns.length && (!this.options.maxRows || this._columns[this._currentColumn].filledRows !== this.options.maxRows)) {
            var c, d = [[]], e = 0, f = this._columns.length, g = this._columns[this._currentColumn].height, h = $.data(a, "grid-columns") || 1, i = $.data(a, "grid-rows") || 1;
            for (c = this._currentColumn; c < f + this._currentColumn; c++) {
                var j = c%f;
                0 === j && (d[e] = []), this._columns[j].height == g ? (d[e].length < h && d[e].push(this._columns[j]), d[e].length == h && (e++, d[e] = [])) : (d[e] = [this._columns[j]], g = this._columns[j].height)
            }
            var k = d[0];
            for (c = 1; c < d.length; c++)
                d[c].length === h && d[c][0].height < k[0].height && (k = d[c]);
            if (k.length == h&&!(this.options.maxRows && k[0].filledRows + i > this.options.maxRows)) {
                for (var l = 0; l < k.length; l++)
                    k[l].addChild(a, 0 === l, b);
                this._setChildProperties(a, b), this._recalculateSmallestColumn()
            }
        }
    }, a.prototype._unpositionChild = function(a, b) {
        this.unsetChildPosition(a, b), this._unsetChildProperties(a, b)
    }, a.prototype._setChildProperties = function(a, b) {
        var c = {
            width: this._getDimensionFromSize($.data(a, "grid-columns")) || 1,
            height: this._getDimensionFromSize($.data(a, "grid-rows")) || 1
        };
        this.options.setChildProperties ? this.options.setChildProperties(a, c, b) : $(a).css(c)
    }, a.prototype._unsetChildProperties = function(a, b) {
        this.options.unsetChildProperties ? this.options.unsetChildProperties(a, b) : $(a).css({
            width: "",
            height: ""
        })
    }, a.prototype.setChildPosition = function(a, b, c) {
        this.options.setChildPosition ? this.options.setChildPosition(a, b, c) : a.css(b)
    }, a.prototype.unsetChildPosition = function(a, b) {
        this.options.unsetChildPosition ? this.options.unsetChildPosition(a, b) : a.css({
            left: "",
            top: ""
        })
    }, a.prototype._getDimensionFromSize = function(a) {
        return a * this.options.gridSize + (a - 1) * this.options.gridGutter
    }, a.prototype._recalculateSmallestColumn = function() {
        var a = this;
        this._currentColumn = _.min(_.range(this._columns.length), function(b) {
            return a._columns[b].height
        })
    }, a.prototype.setOptions = function(a) {
        _.each(a, function(a, b) {
            this.options[b] = a
        }, this)
    };
    var b = function(a, b) {
        this.id = a, this.children = [], this.height = 0, this.filledRows = 0, this.grid = b
    };
    b.prototype.addChild = function(a, b, c) {
        var d = this.grid.options;
        if (this.children.push(a), b) {
            var e = {
                left: this.id * (d.gridSize + d.gridGutter),
                top: this.height
            };
            this.grid.setChildPosition(a, e, c)
        }
        var f = $.data(a, "grid-rows") || 1;
        this.filledRows = this.filledRows + f, this.height = this.filledRows * (d.gridSize + d.gridGutter)
    };
    var c = {}, d = function(a) {
        var b = $(a), d = b.attr("grid-id");
        return void 0 === d && (d = _.size(c), b.attr("grid-id", d)), d
    }
}(), function(a, b) {
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
    a.ui = a.ui || {}, a.ui.version || (a.extend(a.ui, {
        version: "1.10.1",
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
        _focus: a.fn.focus,
        focus: function(b, c) {
            return "number" === typeof b ? this.each(function() {
                var d = this;
                setTimeout(function() {
                    a(d).focus(), c && c.call(d)
                }, b)
            }) : this._focus.apply(this, arguments)
        },
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
            return "number" !== typeof b ? h["outer" + d].call(this, b) : this.each(function() {
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
                    for (d = 0; d < e.length; d++)
                        a.options[e[d][0]] && e[d][1].apply(a.element, c)
            }
        },
        hasScroll: function(b, c) {
            if ("hidden" === a(b).css("overflow"))
                return !1;
            var d = c && "left" === c ? "scrollLeft": "scrollTop", e=!1;
            return b[d] > 0?!0 : (b[d] = 1, e = b[d] > 0, b[d] = 0, e)
        }
    }))
}(jQuery), function(a, b) {
    var c = 0, d = Array.prototype.slice, e = a.cleanData;
    a.cleanData = function(b) {
        for (var c, d = 0; null != (c = b[d]); d++)
            try {
                a(c).triggerHandler("remove")
        } catch (f) {}
        e(b)
    }, a.widget = function(b, c, d) {
        var e, f, g, h, i = {}, j = b.split(".")[0];
        b = b.split(".")[1], e = j + "-" + b, d || (d = c, c = a.Widget), a.expr[":"][e.toLowerCase()] = function(b) {
            return !!a.data(b, e)
        }, a[j] = a[j] || {}, f = a[j][b], g = a[j][b] = function(a, b) {
            return this._createWidget ? void(arguments.length && this._createWidget(a, b)) : new g(a, b)
        }, a.extend(g, f, {
            version: d.version,
            _proto: a.extend({}, d),
            _childConstructors: []
        }), h = new c, h.options = a.widget.extend({}, h.options), a.each(d, function(b, d) {
            return a.isFunction(d) ? void(i[b] = function() {
                var a = function() {
                    return c.prototype[b].apply(this, arguments)
                }, e = function(a) {
                    return c.prototype[b].apply(this, a)
                };
                return function() {
                    var b, c = this._super, f = this._superApply;
                    return this._super = a, this._superApply = e, b = d.apply(this, arguments), this._super = c, this._superApply = f, b
                }
            }()) : void(i[b] = d)
        }), g.prototype = a.widget.extend(h, {
            widgetEventPrefix: f ? h.widgetEventPrefix: b
        }, i, {
            constructor: g,
            namespace: j,
            widgetName: b,
            widgetFullName: e
        }), f ? (a.each(f._childConstructors, function(b, c) {
            var d = c.prototype;
            a.widget(d.namespace + "." + d.widgetName, g, c._proto)
        }), delete f._childConstructors) : c._childConstructors.push(g), a.widget.bridge(b, g)
    }, a.widget.extend = function(c) {
        for (var e, f, g = d.call(arguments, 1), h = 0, i = g.length; h < i; h++)
            for (e in g[h])
                f = g[h][e], g[h].hasOwnProperty(e) && f !== b && (c[e] = a.isPlainObject(f) ? a.isPlainObject(c[e]) ? a.widget.extend({}, c[e], f) : a.widget.extend({}, f) : f);
        return c
    }, a.widget.bridge = function(c, e) {
        var f = e.prototype.widgetFullName || c;
        a.fn[c] = function(g) {
            var h = "string" === typeof g, i = d.call(arguments, 1), j = this;
            return g=!h && i.length ? a.widget.extend.apply(null, [g].concat(i)) : g, this.each(h ? function() {
                var d, e = a.data(this, f);
                return e ? a.isFunction(e[g]) && "_" !== g.charAt(0) ? (d = e[g].apply(e, i), d !== e && d !== b ? (j = d && d.jquery ? j.pushStack(d.get()) : d, !1) : void 0) : a.error("no such method '" + g + "' for " + c + " widget instance") : a.error("cannot call methods on " + c + " prior to initialization; attempted to call method '" + g + "'")
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
            if ("string" === typeof c)
                if (h = {}, e = c.split("."), c = e.shift(), e.length) {
                    for (f = h[c] = a.widget.extend({}, this.options[c]), g = 0; g < e.length - 1; g++)
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
        _on: function(b, c, d) {
            var e, f = this;
            "boolean" !== typeof b && (d = c, c = b, b=!1), d ? (c = e = a(c), this.bindings = this.bindings.add(c)) : (d = c, c = this.element, e = this.widget()), a.each(d, function(d, g) {
                function h() {
                    return b || f.options.disabled!==!0&&!a(this).hasClass("ui-state-disabled") ? ("string" === typeof g ? f[g] : g).apply(f, arguments) : void 0
                }
                "string" !== typeof g && (h.guid = g.guid = g.guid || h.guid || a.guid++);
                var i = d.match(/^(\w+)\s*(.*)$/), j = i[1] + f.eventNamespace, k = i[2];
                k ? e.delegate(k, j, h) : c.bind(j, h)
            })
        },
        _off: function(a, b) {
            b = (b || "").split(" ").join(this.eventNamespace + " ") + this.eventNamespace, a.unbind(b).undelegate(b)
        },
        _delay: function(a, b) {
            function c() {
                return ("string" === typeof a ? d[a] : a).apply(d, arguments)
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
            "string" === typeof e && (e = {
                effect: e
            });
            var g, h = e ? e===!0 || "number" === typeof e ? c: e.effect || c: b;
            e = e || {}, "number" === typeof e && (e = {
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
        version: "1.10.1",
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
                var d = this, e = 1 === c.which, f = "string" === typeof this.options.cancel && c.target.nodeName ? a(c.target).closest(this.options.cancel).length: !1;
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
            return a.ui.ie && (!document.documentMode || document.documentMode < 9)&&!b.button ? this._mouseUp(b) : this._mouseStarted ? (this._mouseDrag(b), b.preventDefault()) : (this._mouseDistanceMet(b) && this._mouseDelayMet(b) && (this._mouseStarted = this._mouseStart(this._mouseDownEvent, b)!==!1, this._mouseStarted ? this._mouseDrag(b) : this._mouseUp(b)), !this._mouseStarted)
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
}(jQuery), function(a) {
    function b(a, b, c) {
        return a > b && a < b + c
    }
    a.widget("ui.sortable", a.ui.mouse, {
        version: "1.10.1",
        widgetEventPrefix: "sort",
        ready: !1,
        options: {
            appendTo: "parent",
            axis: !1,
            connectWith: !1,
            containment: !1,
            cursor: "auto",
            cursorAt: !1,
            dropOnEmpty: !0,
            forcePlaceholderSize: !1,
            forceHelperSize: !1,
            grid: !1,
            handle: !1,
            helper: "original",
            items: "> *",
            opacity: !1,
            placeholder: !1,
            revert: !1,
            scroll: !0,
            scrollSensitivity: 20,
            scrollSpeed: 20,
            scope: "default",
            tolerance: "intersect",
            zIndex: 1e3,
            activate: null,
            beforeStop: null,
            change: null,
            deactivate: null,
            out: null,
            over: null,
            receive: null,
            remove: null,
            sort: null,
            start: null,
            stop: null,
            update: null
        },
        _create: function() {
            var a = this.options;
            this.containerCache = {}, this.element.addClass("ui-sortable"), this.refresh(), this.floating = this.items.length ? "x" === a.axis || /left|right/.test(this.items[0].item.css("float")) || /inline|table-cell/.test(this.items[0].item.css("display")) : !1, this.offset = this.element.offset(), this._mouseInit(), this.ready=!0
        },
        _destroy: function() {
            this.element.removeClass("ui-sortable ui-sortable-disabled"), this._mouseDestroy();
            for (var a = this.items.length - 1; a >= 0; a--)
                this.items[a].item.removeData(this.widgetName + "-item");
            return this
        },
        _setOption: function(b, c) {
            "disabled" === b ? (this.options[b] = c, this.widget().toggleClass("ui-sortable-disabled", !!c)) : a.Widget.prototype._setOption.apply(this, arguments)
        },
        _mouseCapture: function(b, c) {
            var d = null, e=!1, f = this;
            return this.reverting?!1 : this.options.disabled || "static" === this.options.type?!1 : (this._refreshItems(b), a(b.target).parents().each(function() {
                return a.data(this, f.widgetName + "-item") === f ? (d = a(this), !1) : void 0
            }), a.data(b.target, f.widgetName + "-item") === f && (d = a(b.target)), d && (!this.options.handle || c || (a(this.options.handle, d).find("*").addBack().each(function() {
                this === b.target && (e=!0)
            }), e)) ? (this.currentItem = d, this._removeCurrentsFromItems(), !0) : !1)
        },
        _mouseStart: function(b, c, d) {
            var e, f = this.options;
            if (this.currentContainer = this, this.refreshPositions(), this.helper = this._createHelper(b), this._cacheHelperProportions(), this._cacheMargins(), this.scrollParent = this.helper.scrollParent(), this.offset = this.currentItem.offset(), this.offset = {
                top: this.offset.top - this.margins.top,
                left: this.offset.left - this.margins.left
            }, a.extend(this.offset, {
                click: {
                    left: b.pageX - this.offset.left,
                    top: b.pageY - this.offset.top
                },
                parent: this._getParentOffset(),
                relative: this._getRelativeOffset()
            }), this.helper.css("position", "absolute"), this.cssPosition = this.helper.css("position"), this.originalPosition = this._generatePosition(b), this.originalPageX = b.pageX, this.originalPageY = b.pageY, f.cursorAt && this._adjustOffsetFromHelper(f.cursorAt), this.domPosition = {
                prev: this.currentItem.prev()[0],
                parent: this.currentItem.parent()[0]
            }, this.helper[0] !== this.currentItem[0] && this.currentItem.hide(), this._createPlaceholder(), f.containment && this._setContainment(), f.cursor && (a("body").css("cursor") && (this._storedCursor = a("body").css("cursor")), a("body").css("cursor", f.cursor)), f.opacity && (this.helper.css("opacity") && (this._storedOpacity = this.helper.css("opacity")), this.helper.css("opacity", f.opacity)), f.zIndex && (this.helper.css("zIndex") && (this._storedZIndex = this.helper.css("zIndex")), this.helper.css("zIndex", f.zIndex)), this.scrollParent[0] !== document && "HTML" !== this.scrollParent[0].tagName && (this.overflowOffset = this.scrollParent.offset()), this._trigger("start", b, this._uiHash()), this._preserveHelperProportions || this._cacheHelperProportions(), !d)
                for (e = this.containers.length - 1; e >= 0; e--)
                    this.containers[e]._trigger("activate", b, this._uiHash(this));
            return a.ui.ddmanager && (a.ui.ddmanager.current = this), a.ui.ddmanager&&!f.dropBehaviour && a.ui.ddmanager.prepareOffsets(this, b), this.dragging=!0, this.helper.addClass("ui-sortable-helper"), this._mouseDrag(b), !0
        },
        _mouseDrag: function(b) {
            var c, d, e, f, g = this.options, h=!1;
            for (this.position = this._generatePosition(b), this.positionAbs = this._convertPositionTo("absolute"), this.lastPositionAbs || (this.lastPositionAbs = this.positionAbs), this.options.scroll && (this.scrollParent[0] !== document && "HTML" !== this.scrollParent[0].tagName ? (this.overflowOffset.top + this.scrollParent[0].offsetHeight - b.pageY < g.scrollSensitivity ? this.scrollParent[0].scrollTop = h = this.scrollParent[0].scrollTop + g.scrollSpeed : b.pageY - this.overflowOffset.top < g.scrollSensitivity && (this.scrollParent[0].scrollTop = h = this.scrollParent[0].scrollTop - g.scrollSpeed), this.overflowOffset.left + this.scrollParent[0].offsetWidth - b.pageX < g.scrollSensitivity ? this.scrollParent[0].scrollLeft = h = this.scrollParent[0].scrollLeft + g.scrollSpeed : b.pageX - this.overflowOffset.left < g.scrollSensitivity && (this.scrollParent[0].scrollLeft = h = this.scrollParent[0].scrollLeft - g.scrollSpeed)) : (b.pageY - a(document).scrollTop() < g.scrollSensitivity ? h = a(document).scrollTop(a(document).scrollTop() - g.scrollSpeed) : a(window).height() - (b.pageY - a(document).scrollTop()) < g.scrollSensitivity && (h = a(document).scrollTop(a(document).scrollTop() + g.scrollSpeed)), b.pageX - a(document).scrollLeft() < g.scrollSensitivity ? h = a(document).scrollLeft(a(document).scrollLeft() - g.scrollSpeed) : a(window).width() - (b.pageX - a(document).scrollLeft()) < g.scrollSensitivity && (h = a(document).scrollLeft(a(document).scrollLeft() + g.scrollSpeed))), h!==!1 && a.ui.ddmanager&&!g.dropBehaviour && a.ui.ddmanager.prepareOffsets(this, b)), this.positionAbs = this._convertPositionTo("absolute"), this.options.axis && "y" === this.options.axis || (this.helper[0].style.left = this.position.left + "px"), this.options.axis && "x" === this.options.axis || (this.helper[0].style.top = this.position.top + "px"), c = this.items.length - 1; c >= 0; c--)
                if (d = this.items[c], e = d.item[0], f = this._intersectsWithPointer(d), f && d.instance === this.currentContainer && e !== this.currentItem[0] && this.placeholder[1 === f ? "next": "prev"]()[0] !== e&&!a.contains(this.placeholder[0], e) && ("semi-dynamic" === this.options.type?!a.contains(this.element[0], e) : !0)) {
                    if (this.direction = 1 === f ? "down" : "up", "pointer" !== this.options.tolerance&&!this._intersectsWithSides(d))
                        break;
                        this._rearrange(b, d), this._trigger("change", b, this._uiHash());
                        break
                }
            return this._contactContainers(b), a.ui.ddmanager && a.ui.ddmanager.drag(this, b), this._trigger("sort", b, this._uiHash()), this.lastPositionAbs = this.positionAbs, !1
        },
        _mouseStop: function(b, c) {
            if (b) {
                if (a.ui.ddmanager&&!this.options.dropBehaviour && a.ui.ddmanager.drop(this, b), this.options.revert) {
                    var d = this, e = this.placeholder.offset();
                    this.reverting=!0, a(this.helper).animate({
                        left: e.left - this.offset.parent.left - this.margins.left + (this.offsetParent[0] === document.body ? 0 : this.offsetParent[0].scrollLeft),
                        top: e.top - this.offset.parent.top - this.margins.top + (this.offsetParent[0] === document.body ? 0 : this.offsetParent[0].scrollTop)
                    }, parseInt(this.options.revert, 10) || 500, function() {
                        d._clear(b)
                    })
                } else 
                    this._clear(b, c);
                return !1
            }
        },
        cancel: function() {
            if (this.dragging) {
                this._mouseUp({
                    target: null
                }), "original" === this.options.helper ? this.currentItem.css(this._storedCSS).removeClass("ui-sortable-helper") : this.currentItem.show();
                for (var b = this.containers.length - 1; b >= 0; b--)
                    this.containers[b]._trigger("deactivate", null, this._uiHash(this)), this.containers[b].containerCache.over && (this.containers[b]._trigger("out", null, this._uiHash(this)), this.containers[b].containerCache.over = 0)
            }
            return this.placeholder && (this.placeholder[0].parentNode && this.placeholder[0].parentNode.removeChild(this.placeholder[0]), "original" !== this.options.helper && this.helper && this.helper[0].parentNode && this.helper.remove(), a.extend(this, {
                helper: null,
                dragging: !1,
                reverting: !1,
                _noFinalSort: null
            }), this.domPosition.prev ? a(this.domPosition.prev).after(this.currentItem) : a(this.domPosition.parent).prepend(this.currentItem)), this
        },
        serialize: function(b) {
            var c = this._getItemsAsjQuery(b && b.connected), d = [];
            return b = b || {}, a(c).each(function() {
                var c = (a(b.item || this).attr(b.attribute || "id") || "").match(b.expression || /(.+)[\-=_](.+)/);
                c && d.push((b.key || c[1] + "[]") + "=" + (b.key && b.expression ? c[1] : c[2]))
            }), !d.length && b.key && d.push(b.key + "="), d.join("&")
        },
        toArray: function(b) {
            var c = this._getItemsAsjQuery(b && b.connected), d = [];
            return b = b || {}, c.each(function() {
                d.push(a(b.item || this).attr(b.attribute || "id") || "")
            }), d
        },
        _intersectsWith: function(a) {
            var b = this.positionAbs.left, c = b + this.helperProportions.width, d = this.positionAbs.top, e = d + this.helperProportions.height, f = a.left, g = f + a.width, h = a.top, i = h + a.height, j = this.offset.click.top, k = this.offset.click.left, l = d + j > h && d + j < i && b + k > f && b + k < g;
            return "pointer" === this.options.tolerance || this.options.forcePointerForContainers || "pointer" !== this.options.tolerance && this.helperProportions[this.floating ? "width": "height"] > a[this.floating ? "width": "height"] ? l : f < b + this.helperProportions.width / 2 && c - this.helperProportions.width / 2 < g && h < d + this.helperProportions.height / 2 && e - this.helperProportions.height / 2 < i
        },
        _intersectsWithPointer: function(a) {
            var c = "x" === this.options.axis || b(this.positionAbs.top + this.offset.click.top, a.top, a.height), d = "y" === this.options.axis || b(this.positionAbs.left + this.offset.click.left, a.left, a.width), e = c && d, f = this._getDragVerticalDirection(), g = this._getDragHorizontalDirection();
            return e ? this.floating ? g && "right" === g || "down" === f ? 2 : 1 : f && ("down" === f ? 2 : 1) : !1
        },
        _intersectsWithSides: function(a) {
            var c = b(this.positionAbs.top + this.offset.click.top, a.top + a.height / 2, a.height), d = b(this.positionAbs.left + this.offset.click.left, a.left + a.width / 2, a.width), e = this._getDragVerticalDirection(), f = this._getDragHorizontalDirection();
            return this.floating && f ? "right" === f && d || "left" === f&&!d : e && ("down" === e && c || "up" === e&&!c)
        },
        _getDragVerticalDirection: function() {
            var a = this.positionAbs.top - this.lastPositionAbs.top;
            return 0 !== a && (a > 0 ? "down" : "up")
        },
        _getDragHorizontalDirection: function() {
            var a = this.positionAbs.left - this.lastPositionAbs.left;
            return 0 !== a && (a > 0 ? "right" : "left")
        },
        refresh: function(a) {
            return this._refreshItems(a), this.refreshPositions(), this
        },
        _connectWith: function() {
            var a = this.options;
            return a.connectWith.constructor === String ? [a.connectWith] : a.connectWith
        },
        _getItemsAsjQuery: function(b) {
            var c, d, e, f, g = [], h = [], i = this._connectWith();
            if (i && b)
                for (c = i.length - 1; c >= 0; c--)
                    for (e = a(i[c]), d = e.length - 1; d >= 0; d--)
                        f = a.data(e[d], this.widgetFullName), f && f !== this&&!f.options.disabled && h.push([a.isFunction(f.options.items) ? f.options.items.call(f.element): a(f.options.items, f.element).not(".ui-sortable-helper").not(".ui-sortable-placeholder"), f]);
            for (h.push([a.isFunction(this.options.items) ? this.options.items.call(this.element, null, {
                options: this.options,
                item: this.currentItem
            }): a(this.options.items, this.element).not(".ui-sortable-helper").not(".ui-sortable-placeholder"), this]), c = h.length - 1; c >= 0; c--)
                h[c][0].each(function() {
                    g.push(this)
                });
            return a(g)
        },
        _removeCurrentsFromItems: function() {
            var b = this.currentItem.find(":data(" + this.widgetName + "-item)");
            this.items = a.grep(this.items, function(a) {
                for (var c = 0; c < b.length; c++)
                    if (b[c] === a.item[0])
                        return !1;
                return !0
            })
        },
        _refreshItems: function(b) {
            this.items = [], this.containers = [this];
            var c, d, e, f, g, h, i, j, k = this.items, l = [[a.isFunction(this.options.items) ? this.options.items.call(this.element[0], b, {
                item: this.currentItem
            }): a(this.options.items, this.element), this]], m = this._connectWith();
            if (m && this.ready)
                for (c = m.length - 1; c >= 0; c--)
                    for (e = a(m[c]), d = e.length - 1; d >= 0; d--)
                        f = a.data(e[d], this.widgetFullName), f && f !== this&&!f.options.disabled && (l.push([a.isFunction(f.options.items) ? f.options.items.call(f.element[0], b, {
                            item: this.currentItem
                        }): a(f.options.items, f.element), f]), this.containers.push(f));
            for (c = l.length - 1; c >= 0; c--)
                for (g = l[c][1], h = l[c][0], d = 0, j = h.length; d < j; d++)
                    i = a(h[d]), i.data(this.widgetName + "-item", g), k.push({
                        item: i,
                        instance: g,
                        width: 0,
                        height: 0,
                        left: 0,
                        top: 0
                    })
        },
        refreshPositions: function(b) {
            this.offsetParent && this.helper && (this.offset.parent = this._getParentOffset());
            var c, d, e, f;
            for (c = this.items.length - 1; c >= 0; c--)
                d = this.items[c], d.instance !== this.currentContainer && this.currentContainer && d.item[0] !== this.currentItem[0] || (e = this.options.toleranceElement ? a(this.options.toleranceElement, d.item) : d.item, b || (d.width = e.outerWidth(), d.height = e.outerHeight()), f = e.offset(), d.left = f.left, d.top = f.top);
            if (this.options.custom && this.options.custom.refreshContainers)
                this.options.custom.refreshContainers.call(this);
            else 
                for (c = this.containers.length - 1; c >= 0; c--)
                    f = this.containers[c].element.offset(), this.containers[c].containerCache.left = f.left, this.containers[c].containerCache.top = f.top, this.containers[c].containerCache.width = this.containers[c].element.outerWidth(), this.containers[c].containerCache.height = this.containers[c].element.outerHeight();
            return this
        },
        _createPlaceholder: function(b) {
            b = b || this;
            var c, d = b.options;
            d.placeholder && d.placeholder.constructor !== String || (c = d.placeholder, d.placeholder = {
                element: function() {
                    var d = a(document.createElement(b.currentItem[0].nodeName)).addClass(c || b.currentItem[0].className + " ui-sortable-placeholder").removeClass("ui-sortable-helper")[0];
                    return c || (d.style.visibility = "hidden"), d
                },
                update: function(a, e) {
                    (!c || d.forcePlaceholderSize) && (e.height() || e.height(b.currentItem.innerHeight() - parseInt(b.currentItem.css("paddingTop") || 0, 10) - parseInt(b.currentItem.css("paddingBottom") || 0, 10)), e.width() || e.width(b.currentItem.innerWidth() - parseInt(b.currentItem.css("paddingLeft") || 0, 10) - parseInt(b.currentItem.css("paddingRight") || 0, 10)))
                }
            }), b.placeholder = a(d.placeholder.element.call(b.element, b.currentItem)), b.currentItem.after(b.placeholder), d.placeholder.update(b, b.placeholder)
        },
        _contactContainers: function(b) {
            var c, d, e, f, g, h, i, j, k, l = null, m = null;
            for (c = this.containers.length - 1; c >= 0; c--)
                if (!a.contains(this.currentItem[0], this.containers[c].element[0]))
                    if (this._intersectsWith(this.containers[c].containerCache)) {
                        if (l && a.contains(this.containers[c].element[0], l.element[0]))
                            continue;
                            l = this.containers[c], m = c
                    } else 
                        this.containers[c].containerCache.over && (this.containers[c]._trigger("out", b, this._uiHash(this)), this.containers[c].containerCache.over = 0);
            if (l)
                if (1 === this.containers.length)
                    this.containers[m]._trigger("over", b, this._uiHash(this)), this.containers[m].containerCache.over = 1;
                else {
                    for (e = 1e4, f = null, g = this.containers[m].floating ? "left" : "top", h = this.containers[m].floating ? "width" : "height", i = this.positionAbs[g] + this.offset.click[g], d = this.items.length - 1; d >= 0; d--)
                        a.contains(this.containers[m].element[0], this.items[d].item[0]) && this.items[d].item[0] !== this.currentItem[0] && (j = this.items[d].item.offset()[g], k=!1, Math.abs(j - i) > Math.abs(j + this.items[d][h] - i) && (k=!0, j += this.items[d][h]), Math.abs(j - i) < e && (e = Math.abs(j - i), f = this.items[d], this.direction = k ? "up" : "down"));
                        if (!f&&!this.options.dropOnEmpty)
                            return;
                            this.currentContainer = this.containers[m], f ? this._rearrange(b, f, null, !0) : this._rearrange(b, null, this.containers[m].element, !0), this._trigger("change", b, this._uiHash()), this.containers[m]._trigger("change", b, this._uiHash(this)), this.options.placeholder.update(this.currentContainer, this.placeholder), this.containers[m]._trigger("over", b, this._uiHash(this)), this.containers[m].containerCache.over = 1
                }
        },
        _createHelper: function(b) {
            var c = this.options, d = a.isFunction(c.helper) ? a(c.helper.apply(this.element[0], [b, this.currentItem])): "clone" === c.helper ? this.currentItem.clone(): this.currentItem;
            return d.parents("body").length || a("parent" !== c.appendTo ? c.appendTo : this.currentItem[0].parentNode)[0].appendChild(d[0]), d[0] === this.currentItem[0] && (this._storedCSS = {
                width: this.currentItem[0].style.width,
                height: this.currentItem[0].style.height,
                position: this.currentItem.css("position"),
                top: this.currentItem.css("top"),
                left: this.currentItem.css("left")
            }), (!d[0].style.width || c.forceHelperSize) && d.width(this.currentItem.width()), (!d[0].style.height || c.forceHelperSize) && d.height(this.currentItem.height()), d
        },
        _adjustOffsetFromHelper: function(b) {
            "string" === typeof b && (b = b.split(" ")), a.isArray(b) && (b = {
                left: + b[0],
                top: + b[1] || 0
            }), "left"in b && (this.offset.click.left = b.left + this.margins.left), "right"in b && (this.offset.click.left = this.helperProportions.width - b.right + this.margins.left), "top"in b && (this.offset.click.top = b.top + this.margins.top), "bottom"in b && (this.offset.click.top = this.helperProportions.height - b.bottom + this.margins.top)
        },
        _getParentOffset: function() {
            this.offsetParent = this.helper.offsetParent();
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
                var a = this.currentItem.position();
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
                left: parseInt(this.currentItem.css("marginLeft"), 10) || 0,
                top: parseInt(this.currentItem.css("marginTop"), 10) || 0
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
            "parent" === e.containment && (e.containment = this.helper[0].parentNode), ("document" === e.containment || "window" === e.containment) && (this.containment = [0 - this.offset.relative.left - this.offset.parent.left, 0 - this.offset.relative.top - this.offset.parent.top, a("document" === e.containment ? document : window).width() - this.helperProportions.width - this.margins.left, (a("document" === e.containment ? document : window).height() || document.body.parentNode.scrollHeight) - this.helperProportions.height - this.margins.top]), /^(document|window|parent)$/.test(e.containment) || (b = a(e.containment)[0], c = a(e.containment).offset(), d = "hidden" !== a(b).css("overflow"), this.containment = [c.left + (parseInt(a(b).css("borderLeftWidth"), 10) || 0) + (parseInt(a(b).css("paddingLeft"), 10) || 0) - this.margins.left, c.top + (parseInt(a(b).css("borderTopWidth"), 10) || 0) + (parseInt(a(b).css("paddingTop"), 10) || 0) - this.margins.top, c.left + (d ? Math.max(b.scrollWidth, b.offsetWidth) : b.offsetWidth) - (parseInt(a(b).css("borderLeftWidth"), 10) || 0) - (parseInt(a(b).css("paddingRight"), 10) || 0) - this.helperProportions.width - this.margins.left, c.top + (d ? Math.max(b.scrollHeight, b.offsetHeight) : b.offsetHeight) - (parseInt(a(b).css("borderTopWidth"), 10) || 0) - (parseInt(a(b).css("paddingBottom"), 10) || 0) - this.helperProportions.height - this.margins.top])
        },
        _convertPositionTo: function(b, c) {
            c || (c = this.position);
            var d = "absolute" === b ? 1: - 1, e = "absolute" !== this.cssPosition || this.scrollParent[0] !== document && a.contains(this.scrollParent[0], this.offsetParent[0]) ? this.scrollParent: this.offsetParent, f = /(html|body)/i.test(e[0].tagName);
            return {
                top: c.top + this.offset.relative.top * d + this.offset.parent.top * d - ("fixed" === this.cssPosition?-this.scrollParent.scrollTop() : f ? 0 : e.scrollTop()) * d,
                left: c.left + this.offset.relative.left * d + this.offset.parent.left * d - ("fixed" === this.cssPosition?-this.scrollParent.scrollLeft(): f ? 0: e.scrollLeft()) * d
            }
        },
        _generatePosition: function(b) {
            var c, d, e = this.options, f = b.pageX, g = b.pageY, h = "absolute" !== this.cssPosition || this.scrollParent[0] !== document && a.contains(this.scrollParent[0], this.offsetParent[0]) ? this.scrollParent: this.offsetParent, i = /(html|body)/i.test(h[0].tagName);
            return "relative" !== this.cssPosition || this.scrollParent[0] !== document && this.scrollParent[0] !== this.offsetParent[0] || (this.offset.relative = this._getRelativeOffset()), this.originalPosition && (this.containment && (b.pageX - this.offset.click.left < this.containment[0] && (f = this.containment[0] + this.offset.click.left), b.pageY - this.offset.click.top < this.containment[1] && (g = this.containment[1] + this.offset.click.top), b.pageX - this.offset.click.left > this.containment[2] && (f = this.containment[2] + this.offset.click.left), b.pageY - this.offset.click.top > this.containment[3] && (g = this.containment[3] + this.offset.click.top)), e.grid && (c = this.originalPageY + Math.round((g - this.originalPageY) / e.grid[1]) * e.grid[1], g = this.containment ? c - this.offset.click.top >= this.containment[1] && c - this.offset.click.top <= this.containment[3] ? c : c - this.offset.click.top >= this.containment[1] ? c - e.grid[1] : c + e.grid[1] : c, d = this.originalPageX + Math.round((f - this.originalPageX) / e.grid[0]) * e.grid[0], f = this.containment ? d - this.offset.click.left >= this.containment[0] && d - this.offset.click.left <= this.containment[2] ? d : d - this.offset.click.left >= this.containment[0] ? d - e.grid[0] : d + e.grid[0] : d)), {
                top: g - this.offset.click.top - this.offset.relative.top - this.offset.parent.top + ("fixed" === this.cssPosition?-this.scrollParent.scrollTop() : i ? 0 : h.scrollTop()),
                left: f - this.offset.click.left - this.offset.relative.left - this.offset.parent.left + ("fixed" === this.cssPosition?-this.scrollParent.scrollLeft(): i ? 0: h.scrollLeft())
            }
        },
        _rearrange: function(a, b, c, d) {
            c ? c[0].appendChild(this.placeholder[0]) : b.item[0].parentNode.insertBefore(this.placeholder[0], "down" === this.direction ? b.item[0] : b.item[0].nextSibling), this.counter = this.counter?++this.counter : 1;
            var e = this.counter;
            this._delay(function() {
                e === this.counter && this.refreshPositions(!d)
            })
        },
        _clear: function(b, c) {
            this.reverting=!1;
            var d, e = [];
            if (!this._noFinalSort && this.currentItem.parent().length && this.placeholder.before(this.currentItem), this._noFinalSort = null, this.helper[0] === this.currentItem[0]) {
                for (d in this._storedCSS)("auto" === this._storedCSS[d] || "static" === this._storedCSS[d]) 
                    && (this._storedCSS[d] = "");
                this.currentItem.css(this._storedCSS).removeClass("ui-sortable-helper")
            } else 
                this.currentItem.show();
            for (this.fromOutside&&!c && e.push(function(a) {
                this._trigger("receive", a, this._uiHash(this.fromOutside))
            }), !this.fromOutside && this.domPosition.prev === this.currentItem.prev().not(".ui-sortable-helper")[0] && this.domPosition.parent === this.currentItem.parent()[0] || c || e.push(function(a) {
                this._trigger("update", a, this._uiHash())
            }), this !== this.currentContainer && (c || (e.push(function(a) {
                this._trigger("remove", a, this._uiHash())
            }), e.push(function(a) {
                return function(b) {
                    a._trigger("receive", b, this._uiHash(this))
                }
            }.call(this, this.currentContainer)), e.push(function(a) {
                return function(b) {
                    a._trigger("update", b, this._uiHash(this))
                }
            }.call(this, this.currentContainer)))), d = this.containers.length - 1; d >= 0; d--)
                c || e.push(function(a) {
                    return function(b) {
                        a._trigger("deactivate", b, this._uiHash(this))
                    }
                }.call(this, this.containers[d])), this.containers[d].containerCache.over && (e.push(function(a) {
                    return function(b) {
                        a._trigger("out", b, this._uiHash(this))
                    }
                }.call(this, this.containers[d])), this.containers[d].containerCache.over = 0);
            if (this._storedCursor && a("body").css("cursor", this._storedCursor), this._storedOpacity && this.helper.css("opacity", this._storedOpacity), this._storedZIndex && this.helper.css("zIndex", "auto" === this._storedZIndex ? "" : this._storedZIndex), this.dragging=!1, this.cancelHelperRemoval) {
                if (!c) {
                    for (this._trigger("beforeStop", b, this._uiHash()), d = 0; d < e.length; d++)
                        e[d].call(this, b);
                    this._trigger("stop", b, this._uiHash())
                }
                return this.fromOutside=!1, !1
            }
            if (c || this._trigger("beforeStop", b, this._uiHash()), this.placeholder[0].parentNode.removeChild(this.placeholder[0]), this.helper[0] !== this.currentItem[0] && this.helper.remove(), this.helper = null, !c) {
                for (d = 0; d < e.length; d++)
                    e[d].call(this, b);
                this._trigger("stop", b, this._uiHash())
            }
            return this.fromOutside=!1, !0
        },
        _trigger: function() {
            a.Widget.prototype._trigger.apply(this, arguments)===!1 && this.cancel()
        },
        _uiHash: function(b) {
            var c = b || this;
            return {
                helper: c.helper,
                placeholder: c.placeholder || a([]),
                position: c.position,
                originalPosition: c.originalPosition,
                offset: c.positionAbs,
                item: c.currentItem,
                sender: b ? b.element: null
            }
        }
    })
}(jQuery), function(a, b) {
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
        version: "1.10.4",
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
                return "number" === typeof c ? this.each(function() {
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
            return "number" !== typeof b ? h["outer" + d].call(this, b) : this.each(function() {
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
                    for (d = 0; d < e.length; d++)
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
    }, a.widget = function(b, c, d) {
        var e, f, g, h, i = {}, j = b.split(".")[0];
        b = b.split(".")[1], e = j + "-" + b, d || (d = c, c = a.Widget), a.expr[":"][e.toLowerCase()] = function(b) {
            return !!a.data(b, e)
        }, a[j] = a[j] || {}, f = a[j][b], g = a[j][b] = function(a, b) {
            return this._createWidget ? void(arguments.length && this._createWidget(a, b)) : new g(a, b)
        }, a.extend(g, f, {
            version: d.version,
            _proto: a.extend({}, d),
            _childConstructors: []
        }), h = new c, h.options = a.widget.extend({}, h.options), a.each(d, function(b, d) {
            return a.isFunction(d) ? void(i[b] = function() {
                var a = function() {
                    return c.prototype[b].apply(this, arguments)
                }, e = function(a) {
                    return c.prototype[b].apply(this, a)
                };
                return function() {
                    var b, c = this._super, f = this._superApply;
                    return this._super = a, this._superApply = e, b = d.apply(this, arguments), this._super = c, this._superApply = f, b
                }
            }()) : void(i[b] = d)
        }), g.prototype = a.widget.extend(h, {
            widgetEventPrefix: f ? h.widgetEventPrefix || b: b
        }, i, {
            constructor: g,
            namespace: j,
            widgetName: b,
            widgetFullName: e
        }), f ? (a.each(f._childConstructors, function(b, c) {
            var d = c.prototype;
            a.widget(d.namespace + "." + d.widgetName, g, c._proto)
        }), delete f._childConstructors) : c._childConstructors.push(g), a.widget.bridge(b, g)
    }, a.widget.extend = function(c) {
        for (var e, f, g = d.call(arguments, 1), h = 0, i = g.length; h < i; h++)
            for (e in g[h])
                f = g[h][e], g[h].hasOwnProperty(e) && f !== b && (c[e] = a.isPlainObject(f) ? a.isPlainObject(c[e]) ? a.widget.extend({}, c[e], f) : a.widget.extend({}, f) : f);
        return c
    }, a.widget.bridge = function(c, e) {
        var f = e.prototype.widgetFullName || c;
        a.fn[c] = function(g) {
            var h = "string" === typeof g, i = d.call(arguments, 1), j = this;
            return g=!h && i.length ? a.widget.extend.apply(null, [g].concat(i)) : g, this.each(h ? function() {
                var d, e = a.data(this, f);
                return e ? a.isFunction(e[g]) && "_" !== g.charAt(0) ? (d = e[g].apply(e, i), d !== e && d !== b ? (j = d && d.jquery ? j.pushStack(d.get()) : d, !1) : void 0) : a.error("no such method '" + g + "' for " + c + " widget instance") : a.error("cannot call methods on " + c + " prior to initialization; attempted to call method '" + g + "'")
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
            if ("string" === typeof c)
                if (h = {}, e = c.split("."), c = e.shift(), e.length) {
                    for (f = h[c] = a.widget.extend({}, this.options[c]), g = 0; g < e.length - 1; g++)
                        f[e[g]] = f[e[g]] || {}, f = f[e[g]];
                        if (c = e.pop(), 1 === arguments.length)
                            return f[c] === b ? null : f[c];
                            f[c] = d
                } else {
                    if (1 === arguments.length)
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
        _on: function(b, c, d) {
            var e, f = this;
            "boolean" !== typeof b && (d = c, c = b, b=!1), d ? (c = e = a(c), this.bindings = this.bindings.add(c)) : (d = c, c = this.element, e = this.widget()), a.each(d, function(d, g) {
                function h() {
                    return b || f.options.disabled!==!0&&!a(this).hasClass("ui-state-disabled") ? ("string" === typeof g ? f[g] : g).apply(f, arguments) : void 0
                }
                "string" !== typeof g && (h.guid = g.guid = g.guid || h.guid || a.guid++);
                var i = d.match(/^(\w+)\s*(.*)$/), j = i[1] + f.eventNamespace, k = i[2];
                k ? e.delegate(k, j, h) : c.bind(j, h)
            })
        },
        _off: function(a, b) {
            b = (b || "").split(" ").join(this.eventNamespace + " ") + this.eventNamespace, a.unbind(b).undelegate(b)
        },
        _delay: function(a, b) {
            function c() {
                return ("string" === typeof a ? d[a] : a).apply(d, arguments)
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
            "string" === typeof e && (e = {
                effect: e
            });
            var g, h = e ? e===!0 || "number" === typeof e ? c: e.effect || c: b;
            e = e || {}, "number" === typeof e && (e = {
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
        version: "1.10.4",
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
                var d = this, e = 1 === c.which, f = "string" === typeof this.options.cancel && c.target.nodeName ? a(c.target).closest(this.options.cancel).length: !1;
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
            return a.ui.ie && (!document.documentMode || document.documentMode < 9)&&!b.button ? this._mouseUp(b) : this._mouseStarted ? (this._mouseDrag(b), b.preventDefault()) : (this._mouseDistanceMet(b) && this._mouseDelayMet(b) && (this._mouseStarted = this._mouseStart(this._mouseDownEvent, b)!==!1, this._mouseStarted ? this._mouseDrag(b) : this._mouseUp(b)), !this._mouseStarted)
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
}(jQuery), function(a) {
    a.widget("ui.draggable", a.ui.mouse, {
        version: "1.10.4",
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
            "string" === typeof b && (b = b.split(" ")), a.isArray(b) && (b = {
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
                h = n.snapElements[l].left, i = h + n.snapElements[l].width, j = n.snapElements[l].top, k = j + n.snapElements[l].height, r < h - p || q > i + p || t < j - p || s > k + p ||!a.contains(n.snapElements[l].item.ownerDocument, n.snapElements[l].item) ? (n.snapElements[l].snapping && n.options.snap.release && n.options.snap.release.call(n.element, b, a.extend(n._uiHash(), {
                    snapItem: n.snapElements[l].item
                })), n.snapElements[l].snapping=!1) : ("inner" !== o.snapMode && (d = Math.abs(j - t) <= p, e = Math.abs(k - s) <= p, f = Math.abs(h - r) <= p, g = Math.abs(i - q) <= p, d && (c.position.top = n._convertPositionTo("relative", {
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
                }).left - n.margins.left)), m = d || e || f || g, "outer" !== o.snapMode && (d = Math.abs(j - s) <= p, e = Math.abs(k - t) <= p, f = Math.abs(h - q) <= p, g = Math.abs(i - r) <= p, d && (c.position.top = n._convertPositionTo("relative", {
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
    function b(a, b, c) {
        return a > b && a < b + c
    }
    a.widget("ui.droppable", {
        version: "1.10.4",
        widgetEventPrefix: "drop",
        options: {
            accept: "*",
            activeClass: !1,
            addClasses: !0,
            greedy: !1,
            hoverClass: !1,
            scope: "default",
            tolerance: "intersect",
            activate: null,
            deactivate: null,
            drop: null,
            out: null,
            over: null
        },
        _create: function() {
            var b, c = this.options, d = c.accept;
            this.isover=!1, this.isout=!0, this.accept = a.isFunction(d) ? d : function(a) {
                return a.is(d)
            }, this.proportions = function() {
                return arguments.length ? void(b = arguments[0]) : b ? b : b = {
                    width: this.element[0].offsetWidth,
                    height: this.element[0].offsetHeight
                }
            }, a.ui.ddmanager.droppables[c.scope] = a.ui.ddmanager.droppables[c.scope] || [], a.ui.ddmanager.droppables[c.scope].push(this), c.addClasses && this.element.addClass("ui-droppable")
        },
        _destroy: function() {
            for (var b = 0, c = a.ui.ddmanager.droppables[this.options.scope]; b < c.length; b++)
                c[b] === this && c.splice(b, 1);
            this.element.removeClass("ui-droppable ui-droppable-disabled")
        },
        _setOption: function(b, c) {
            "accept" === b && (this.accept = a.isFunction(c) ? c : function(a) {
                return a.is(c)
            }), a.Widget.prototype._setOption.apply(this, arguments)
        },
        _activate: function(b) {
            var c = a.ui.ddmanager.current;
            this.options.activeClass && this.element.addClass(this.options.activeClass), c && this._trigger("activate", b, this.ui(c))
        },
        _deactivate: function(b) {
            var c = a.ui.ddmanager.current;
            this.options.activeClass && this.element.removeClass(this.options.activeClass), c && this._trigger("deactivate", b, this.ui(c))
        },
        _over: function(b) {
            var c = a.ui.ddmanager.current;
            c && (c.currentItem || c.element)[0] !== this.element[0] && this.accept.call(this.element[0], c.currentItem || c.element) && (this.options.hoverClass && this.element.addClass(this.options.hoverClass), this._trigger("over", b, this.ui(c)))
        },
        _out: function(b) {
            var c = a.ui.ddmanager.current;
            c && (c.currentItem || c.element)[0] !== this.element[0] && this.accept.call(this.element[0], c.currentItem || c.element) && (this.options.hoverClass && this.element.removeClass(this.options.hoverClass), this._trigger("out", b, this.ui(c)))
        },
        _drop: function(b, c) {
            var d = c || a.ui.ddmanager.current, e=!1;
            return d && (d.currentItem || d.element)[0] !== this.element[0] ? (this.element.find(":data(ui-droppable)").not(".ui-draggable-dragging").each(function() {
                var b = a.data(this, "ui-droppable");
                return b.options.greedy&&!b.options.disabled && b.options.scope === d.options.scope && b.accept.call(b.element[0], d.currentItem || d.element) && a.ui.intersect(d, a.extend(b, {
                    offset: b.element.offset()
                }), b.options.tolerance) ? (e=!0, !1) : void 0
            }), e?!1 : this.accept.call(this.element[0], d.currentItem || d.element) ? (this.options.activeClass && this.element.removeClass(this.options.activeClass), this.options.hoverClass && this.element.removeClass(this.options.hoverClass), this._trigger("drop", b, this.ui(d)), this.element) : !1) : !1
        },
        ui: function(a) {
            return {
                draggable: a.currentItem || a.element,
                helper: a.helper,
                position: a.position,
                offset: a.positionAbs
            }
        }
    }), a.ui.intersect = function(a, c, d) {
        if (!c.offset)
            return !1;
        var e, f, g = (a.positionAbs || a.position.absolute).left, h = (a.positionAbs || a.position.absolute).top, i = g + a.helperProportions.width, j = h + a.helperProportions.height, k = c.offset.left, l = c.offset.top, m = k + c.proportions().width, n = l + c.proportions().height;
        switch (d) {
        case"fit":
            return k <= g && i <= m && l <= h && j <= n;
        case"intersect":
            return k < g + a.helperProportions.width / 2 && i - a.helperProportions.width / 2 < m && l < h + a.helperProportions.height / 2 && j - a.helperProportions.height / 2 < n;
        case"pointer":
            return e = (a.positionAbs || a.position.absolute).left + (a.clickOffset || a.offset.click).left, f = (a.positionAbs || a.position.absolute).top + (a.clickOffset || a.offset.click).top, b(f, l, c.proportions().height) && b(e, k, c.proportions().width);
        case"touch":
            return (h >= l && h <= n || j >= l && j <= n || h < l && j > n) && (g >= k && g <= m || i >= k && i <= m || g < k && i > m);
        default:
            return !1
        }
    }, a.ui.ddmanager = {
        current: null,
        droppables: {
            "default": []
        },
        prepareOffsets: function(b, c) {
            var d, e, f = a.ui.ddmanager.droppables[b.options.scope] || [], g = c ? c.type: null, h = (b.currentItem || b.element).find(":data(ui-droppable)").addBack();
            a: for (d = 0; d < f.length; d++)
                if (!(f[d].options.disabled || b&&!f[d].accept.call(f[d].element[0], b.currentItem || b.element))) {
                    for (e = 0; e < h.length; e++)
                        if (h[e] === f[d].element[0]) {
                            f[d].proportions().height = 0;
                            continue a
                        }
                        f[d].visible = "none" !== f[d].element.css("display"), f[d].visible && ("mousedown" === g && f[d]._activate.call(f[d], c), f[d].offset = f[d].element.offset(), f[d].proportions({
                            width: f[d].element[0].offsetWidth,
                            height: f[d].element[0].offsetHeight
                        }))
                }
        },
        drop: function(b, c) {
            var d=!1;
            return a.each((a.ui.ddmanager.droppables[b.options.scope] || []).slice(), function() {
                this.options && (!this.options.disabled && this.visible && a.ui.intersect(b, this, this.options.tolerance) && (d = this._drop.call(this, c) || d), !this.options.disabled && this.visible && this.accept.call(this.element[0], b.currentItem || b.element) && (this.isout=!0, this.isover=!1, this._deactivate.call(this, c)))
            }), d
        },
        dragStart: function(b, c) {
            b.element.parentsUntil("body").bind("scroll.droppable", function() {
                b.options.refreshPositions || a.ui.ddmanager.prepareOffsets(b, c)
            })
        },
        drag: function(b, c) {
            b.options.refreshPositions && a.ui.ddmanager.prepareOffsets(b, c), a.each(a.ui.ddmanager.droppables[b.options.scope] || [], function() {
                if (!this.options.disabled&&!this.greedyChild && this.visible) {
                    var d, e, f, g = a.ui.intersect(b, this, this.options.tolerance), h=!g && this.isover ? "isout" : g&&!this.isover ? "isover" : null;
                    h && (this.options.greedy && (e = this.options.scope, f = this.element.parents(":data(ui-droppable)").filter(function() {
                        return a.data(this, "ui-droppable").options.scope === e
                    }), f.length && (d = a.data(f[0], "ui-droppable"), d.greedyChild = "isover" === h)), d && "isover" === h && (d.isover=!1, d.isout=!0, d._out.call(d, c)), this[h]=!0, this["isout" === h ? "isover": "isout"]=!1, this["isover" === h ? "_over": "_out"].call(this, c), d && "isout" === h && (d.isout=!1, d.isover=!0, d._over.call(d, c)))
                }
            })
        },
        dragStop: function(b, c) {
            b.element.parentsUntil("body").unbind("scroll.droppable"), b.options.refreshPositions || a.ui.ddmanager.prepareOffsets(b, c)
        }
    }
}(jQuery), function(a) {
    var b = document.createElement("input"), c = "oninput"in b && (!("documentMode"in document) || document.documentMode > 9), d = function(a) {
        return "INPUT" === a.nodeName && ("text" === a.type || "password" === a.type)
    }, e = null, f = null, g = null, h = {
        get: function() {
            return g.get.call(this)
        },
        set: function(a) {
            f = a, g.set.call(this, a)
        }
    }, i = function(a) {
        e = a, f = a.value, g = Object.getOwnPropertyDescriptor(a.constructor.prototype, "value"), Object.defineProperty(e, "value", h), e.attachEvent("onpropertychange", k)
    }, j = function() {
        e && (delete e.value, e.detachEvent("onpropertychange", k), e = null, f = null, g = null)
    }, k = function(b) {
        if ("value" === b.propertyName) {
            var c = b.srcElement.value;
            c !== f && (f = c, a(e).trigger("textchange"))
        }
    };
    c ? a(document).on("input", function(b) {
        "TEXTAREA" !== b.target.nodeName && a(b.target).trigger("textchange")
    }) : a(document).on("focusin", function(a) {
        d(a.target) && (j(), i(a.target))
    }).on("focusout", function() {
        j()
    }).on("selectionchange keyup keydown", function() {
        e && e.value !== f && (f = e.value, a(e).trigger("textchange"))
    })
}(jQuery), function(a) {
    a.color = {}, a.color.make = function(b, c, d, e) {
        var f = {};
        return f.r = b || 0, f.g = c || 0, f.b = d || 0, f.a = null != e ? e : 1, f.add = function(a, b) {
            for (var c = 0; c < a.length; ++c)
                f[a.charAt(c)] += b;
            return f.normalize()
        }, f.scale = function(a, b) {
            for (var c = 0; c < a.length; ++c)
                f[a.charAt(c)]*=b;
            return f.normalize()
        }, f.toString = function() {
            return f.a >= 1 ? "rgb(" + [f.r, f.g, f.b].join(",") + ")" : "rgba(" + [f.r, f.g, f.b, f.a].join(",") + ")"
        }, f.normalize = function() {
            function a(a, b, c) {
                return b < a ? a : b > c ? c : b
            }
            return f.r = a(0, parseInt(f.r), 255), f.g = a(0, parseInt(f.g), 255), f.b = a(0, parseInt(f.b), 255), f.a = a(0, f.a, 1), f
        }, f.clone = function() {
            return a.color.make(f.r, f.b, f.g, f.a)
        }, f.normalize()
    }, a.color.extract = function(b, c) {
        var d;
        do {
            if (d = b.css(c).toLowerCase(), "" != d && "transparent" != d)
                break;
            b = b.parent()
        }
        while (!a.nodeName(b.get(0), "body"));
        return "rgba(0, 0, 0, 0)" == d && (d = "transparent"), a.color.parse(d)
    }, a.color.parse = function(c) {
        var d, e = a.color.make;
        if (d = /rgb\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*\)/.exec(c))
            return e(parseInt(d[1], 10), parseInt(d[2], 10), parseInt(d[3], 10));
        if (d = /rgba\(\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]{1,3})\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)/.exec(c))
            return e(parseInt(d[1], 10), parseInt(d[2], 10), parseInt(d[3], 10), parseFloat(d[4]));
        if (d = /rgb\(\s*([0-9]+(?:\.[0-9]+)?)\%\s*,\s*([0-9]+(?:\.[0-9]+)?)\%\s*,\s*([0-9]+(?:\.[0-9]+)?)\%\s*\)/.exec(c))
            return e(2.55 * parseFloat(d[1]), 2.55 * parseFloat(d[2]), 2.55 * parseFloat(d[3]));
        if (d = /rgba\(\s*([0-9]+(?:\.[0-9]+)?)\%\s*,\s*([0-9]+(?:\.[0-9]+)?)\%\s*,\s*([0-9]+(?:\.[0-9]+)?)\%\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)/.exec(c))
            return e(2.55 * parseFloat(d[1]), 2.55 * parseFloat(d[2]), 2.55 * parseFloat(d[3]), parseFloat(d[4]));
        if (d = /#([a-fA-F0-9]{2})([a-fA-F0-9]{2})([a-fA-F0-9]{2})/.exec(c))
            return e(parseInt(d[1], 16), parseInt(d[2], 16), parseInt(d[3], 16));
        if (d = /#([a-fA-F0-9])([a-fA-F0-9])([a-fA-F0-9])/.exec(c))
            return e(parseInt(d[1] + d[1], 16), parseInt(d[2] + d[2], 16), parseInt(d[3] + d[3], 16));
        var f = a.trim(c).toLowerCase();
        return "transparent" == f ? e(255, 255, 255, 0) : (d = b[f] || [0, 0, 0], e(d[0], d[1], d[2]))
    };
    var b = {
        aqua: [0, 255, 255],
        azure: [240, 255, 255],
        beige: [245, 245, 220],
        black: [0, 0, 0],
        blue: [0, 0, 255],
        brown: [165, 42, 42],
        cyan: [0, 255, 255],
        darkblue: [0, 0, 139],
        darkcyan: [0, 139, 139],
        darkgrey: [169, 169, 169],
        darkgreen: [0, 100, 0],
        darkkhaki: [189, 183, 107],
        darkmagenta: [139, 0, 139],
        darkolivegreen: [85, 107, 47],
        darkorange: [255, 140, 0],
        darkorchid: [153, 50, 204],
        darkred: [139, 0, 0],
        darksalmon: [233, 150, 122],
        darkviolet: [148, 0, 211],
        fuchsia: [255, 0, 255],
        gold: [255, 215, 0],
        green: [0, 128, 0],
        indigo: [75, 0, 130],
        khaki: [240, 230, 140],
        lightblue: [173, 216, 230],
        lightcyan: [224, 255, 255],
        lightgreen: [144, 238, 144],
        lightgrey: [211, 211, 211],
        lightpink: [255, 182, 193],
        lightyellow: [255, 255, 224],
        lime: [0, 255, 0],
        magenta: [255, 0, 255],
        maroon: [128, 0, 0],
        navy: [0, 0, 128],
        olive: [128, 128, 0],
        orange: [255, 165, 0],
        pink: [255, 192, 203],
        purple: [128, 0, 128],
        violet: [128, 0, 128],
        red: [255, 0, 0],
        silver: [192, 192, 192],
        white: [255, 255, 255],
        yellow: [255, 255, 0]
    }
}(jQuery), function(a) {
    function b(b, d, e, f) {
        function g(a, b) {
            b = [ub].concat(b);
            for (var c = 0; c < a.length; ++c)
                a[c].apply(this, b)
        }
        function h() {
            for (var b = 0; b < f.length; ++b) {
                var c = f[b];
                c.init(ub), c.options && a.extend(!0, gb, c.options)
            }
        }
        function j(b) {
            var c;
            for (a.extend(!0, gb, b), null == gb.xaxis.color && (gb.xaxis.color = gb.grid.color), null == gb.yaxis.color && (gb.yaxis.color = gb.grid.color), null == gb.xaxis.tickColor && (gb.xaxis.tickColor = gb.grid.tickColor), null == gb.yaxis.tickColor && (gb.yaxis.tickColor = gb.grid.tickColor), null == gb.grid.borderColor && (gb.grid.borderColor = gb.grid.color), null == gb.grid.tickColor && (gb.grid.tickColor = a.color.parse(gb.grid.color).scale("a", .22).toString()), c = 0; c < Math.max(1, gb.xaxes.length); ++c)
                gb.xaxes[c] = a.extend(!0, {}, gb.xaxis, gb.xaxes[c]);
            for (c = 0; c < Math.max(1, gb.yaxes.length); ++c)
                gb.yaxes[c] = a.extend(!0, {}, gb.yaxis, gb.yaxes[c]);
            for (gb.xaxis.noTicks && null == gb.xaxis.ticks && (gb.xaxis.ticks = gb.xaxis.noTicks), gb.yaxis.noTicks && null == gb.yaxis.ticks && (gb.yaxis.ticks = gb.yaxis.noTicks), gb.x2axis && (gb.xaxes[1] = a.extend(!0, {}, gb.xaxis, gb.x2axis), gb.xaxes[1].position = "top"), gb.y2axis && (gb.yaxes[1] = a.extend(!0, {}, gb.yaxis, gb.y2axis), gb.yaxes[1].position = "right"), gb.grid.coloredAreas && (gb.grid.markings = gb.grid.coloredAreas), gb.grid.coloredAreasColor && (gb.grid.markingsColor = gb.grid.coloredAreasColor), gb.lines && a.extend(!0, gb.series.lines, gb.lines), gb.points && a.extend(!0, gb.series.points, gb.points), gb.bars && a.extend(!0, gb.series.bars, gb.bars), null != gb.shadowSize && (gb.series.shadowSize = gb.shadowSize), c = 0; c < gb.xaxes.length; ++c)
                q(mb, c + 1).options = gb.xaxes[c];
            for (c = 0; c < gb.yaxes.length; ++c)
                q(nb, c + 1).options = gb.yaxes[c];
            for (var d in tb)
                gb.hooks[d] && gb.hooks[d].length && (tb[d] = tb[d].concat(gb.hooks[d]));
            g(tb.processOptions, [gb])
        }
        function k(a) {
            fb = l(a), r(), s()
        }
        function l(b) {
            for (var c = [], d = 0; d < b.length; ++d) {
                var e = a.extend(!0, {}, gb.series);
                null != b[d].data ? (e.data = b[d].data, delete b[d].data, a.extend(!0, e, b[d]), b[d].data = e.data) : e.data = b[d], c.push(e)
            }
            return c
        }
        function m(a, b) {
            var c = a[b + "axis"];
            return "object" == typeof c && (c = c.n), "number" != typeof c && (c = 1), c
        }
        function n() {
            return a.grep(mb.concat(nb), function(a) {
                return a
            })
        }
        function o(a) {
            var b, c, d = {};
            for (b = 0; b < mb.length; ++b)
                c = mb[b], c && c.used && (d["x" + c.n] = c.c2p(a.left));
            for (b = 0; b < nb.length; ++b)
                c = nb[b], c && c.used && (d["y" + c.n] = c.c2p(a.top));
            return void 0 !== d.x1 && (d.x = d.x1), void 0 !== d.y1 && (d.y = d.y1), d
        }
        function p(a) {
            var b, c, d, e = {};
            for (b = 0; b < mb.length; ++b)
                if (c = mb[b], c && c.used && (d = "x" + c.n, null == a[d] && 1 == c.n && (d = "x"), null != a[d])) {
                    e.left = c.p2c(a[d]);
                    break
                }
            for (b = 0; b < nb.length; ++b)
                if (c = nb[b], c && c.used && (d = "y" + c.n, null == a[d] && 1 == c.n && (d = "y"), null != a[d])) {
                    e.top = c.p2c(a[d]);
                    break
                }
            return e
        }
        function q(b, c) {
            return b[c - 1] || (b[c - 1] = {
                n: c,
                direction: b == mb ? "x": "y",
                options: a.extend(!0, {}, b == mb ? gb.xaxis : gb.yaxis)
            }), b[c - 1]
        }
        function r() {
            var b, c = fb.length, d = [], e = [];
            for (b = 0; b < fb.length; ++b) {
                var f = fb[b].color;
                null != f && (--c, "number" == typeof f ? e.push(f) : d.push(a.color.parse(fb[b].color)))
            }
            for (b = 0; b < e.length; ++b)
                c = Math.max(c, e[b] + 1);
            var g = [], h = 0;
            for (b = 0; g.length < c;) {
                var i;
                i = gb.colors.length == b ? a.color.make(100, 100, 100) : a.color.parse(gb.colors[b]);
                var j = h%2 == 1?-1 : 1;
                i.scale("rgb", 1 + j * Math.ceil(h / 2) * .2), g.push(i), ++b, b >= gb.colors.length && (b = 0, ++h)
            }
            var k, l = 0;
            for (b = 0; b < fb.length; ++b) {
                if (k = fb[b], null == k.color ? (k.color = g[l].toString(), ++l) : "number" == typeof k.color && (k.color = g[k.color].toString()), null == k.lines.show) {
                    var n, o=!0;
                    for (n in k)
                        if (k[n] && k[n].show) {
                            o=!1;
                            break
                        }
                    o && (k.lines.show=!0)
                }
                k.xaxis = q(mb, m(k, "x")), k.yaxis = q(nb, m(k, "y"))
            }
        }
        function s() {
            function b(a, b, c) {
                b < a.datamin && b!=-q && (a.datamin = b), c > a.datamax && c != q && (a.datamax = c)
            }
            var c, d, e, f, h, i, j, k, l, m, o = Number.POSITIVE_INFINITY, p = Number.NEGATIVE_INFINITY, q = Number.MAX_VALUE;
            for (a.each(n(), function(a, b) {
                b.datamin = o, b.datamax = p, b.used=!1
            }), c = 0; c < fb.length; ++c)
                h = fb[c], h.datapoints = {
                    points: []
                }, g(tb.processRawData, [h, h.data, h.datapoints]);
            for (c = 0; c < fb.length; ++c) {
                h = fb[c];
                var r = h.data, s = h.datapoints.format;
                if (s || (s = [], s.push({
                    x: !0,
                    number: !0,
                    required: !0
                }), s.push({
                    y: !0,
                    number: !0,
                    required: !0
                }), (h.bars.show || h.lines.show && h.lines.fill) && (s.push({
                    y: !0,
                    number: !0,
                    required: !1,
                    defaultValue: 0
                }), h.bars.horizontal && (delete s[s.length - 1].y, s[s.length - 1].x=!0)), h.datapoints.format = s), null == h.datapoints.pointsize)
                    for (h.datapoints.pointsize = s.length, j = h.datapoints.pointsize, i = h.datapoints.points, insertSteps = h.lines.show && h.lines.steps, h.xaxis.used = h.yaxis.used=!0, d = e = 0; d < r.length; ++d, e += j) {
                        m = r[d];
                        var t = null == m;
                        if (!t)
                            for (f = 0; f < j; ++f)
                                k = m[f], l = s[f], l && (l.number && null != k && (k =+ k, isNaN(k) ? k = null : 1 / 0 == k ? k = q : k==-1 / 0 && (k =- q)), null == k && (l.required && (t=!0), null != l.defaultValue && (k = l.defaultValue))), i[e + f] = k;
                                if (t)
                                    for (f = 0; f < j; ++f)
                                        k = i[e + f], null != k && (l = s[f], l.x && b(h.xaxis, k, k), l.y && b(h.yaxis, k, k)), i[e + f] = null;
                                else if (insertSteps && e > 0 && null != i[e - j] && i[e - j] != i[e] && i[e - j + 1] != i[e + 1]) {
                                    for (f = 0; f < j; ++f)
                                        i[e + j + f] = i[e + f];
                                        i[e + 1] = i[e - j + 1], e += j
                                }
                    }
            }
            for (c = 0; c < fb.length; ++c)
                h = fb[c], g(tb.processDatapoints, [h, h.datapoints]);
            for (c = 0; c < fb.length; ++c) {
                h = fb[c], i = h.datapoints.points, j = h.datapoints.pointsize;
                var u = o, v = o, w = p, x = p;
                for (d = 0; d < i.length; d += j)
                    if (null != i[d])
                        for (f = 0; f < j; ++f)
                            k = i[d + f], l = s[f], l && k != q && k!=-q && (l.x && (k < u && (u = k), k > w && (w = k)), l.y && (k < v && (v = k), k > x && (x = k)));
                if (h.bars.show) {
                    var y = "left" == h.bars.align ? 0: - h.bars.barWidth / 2;
                    h.bars.horizontal ? (v += y, x += y + h.bars.barWidth) : (u += y, w += y + h.bars.barWidth)
                }
                b(h.xaxis, u, w), b(h.yaxis, v, x)
            }
            a.each(n(), function(a, b) {
                b.datamin == o && (b.datamin = null), b.datamax == p && (b.datamax = null)
            })
        }
        function t(c, d) {
            var e = document.createElement("canvas");
            return e.className = d, e.width = pb, e.height = qb, c || a(e).css({
                position: "absolute",
                left: 0,
                top: 0
            }), a(e).appendTo(b), e.getContext || (e = window.G_vmlCanvasManager.initElement(e)), e.getContext("2d").save(), e
        }
        function u() {
            if (pb = b.width(), qb = b.height(), pb <= 0 || qb <= 0)
                throw "Invalid dimensions for plot, width = " + pb + ", height = " + qb
        }
        function v(a) {
            a.width != pb && (a.width = pb), a.height != qb && (a.height = qb);
            var b = a.getContext("2d");
            b.restore(), b.save()
        }
        function w() {
            var c, d = b.children("canvas.base"), e = b.children("canvas.overlay");
            0 == d.length || 0 == e ? (b.html(""), b.css({
                padding: 0
            }), "static" == b.css("position") && b.css("position", "relative"), u(), hb = t(!1, "base"), ib = t(!1, "overlay"), c=!1) : (hb = d.get(0), ib = e.get(0), c=!0), kb = hb.getContext("2d"), lb = ib.getContext("2d"), jb = a([ib, hb]), c && (b.data("plot").shutdown(), ub.resize(), lb.clearRect(0, 0, pb, qb), jb.unbind(), b.children().not([hb, ib]).remove()), b.data("plot", ub)
        }
        function x() {
            gb.grid.hoverable && (jb.mousemove(V), jb.mouseleave(W)), gb.grid.clickable && jb.click(X), g(tb.bindEvents, [jb])
        }
        function y() {
            wb && clearTimeout(wb), jb.unbind("mousemove", V), jb.unbind("mouseleave", W), jb.unbind("click", X), g(tb.shutdown, [jb])
        }
        function z(a) {
            function b(a) {
                return a
            }
            var c, d, e = a.options.transform || b, f = a.options.inverseTransform;
            "x" == a.direction ? (c = a.scale = rb / Math.abs(e(a.max) - e(a.min)), d = Math.min(e(a.max), e(a.min))) : (c = a.scale = sb / Math.abs(e(a.max) - e(a.min)), c =- c, d = Math.max(e(a.max), e(a.min))), a.p2c = e == b ? function(a) {
                return (a - d) * c
            } : function(a) {
                return (e(a) - d) * c
            }, a.c2p = f ? function(a) {
                return f(d + a / c)
            } : function(a) {
                return d + a / c
            }
        }
        function A(c) {
            function d(d, e) {
                return a('<div style="position:absolute;top:-10000px;' + e + 'font-size:smaller"><div class="' + c.direction + "Axis " + c.direction + c.n + 'Axis">' + d.join("") + "</div></div>").appendTo(b)
            }
            var e, f, g, h = c.options, i = c.ticks || [], j = [], k = h.labelWidth, l = h.labelHeight;
            if ("x" == c.direction) {
                if (null == k && (k = Math.floor(pb / (i.length > 0 ? i.length : 1))), null == l) {
                    for (j = [], e = 0; e < i.length; ++e)
                        f = i[e].label, f && j.push('<div class="tickLabel" style="float:left;width:' + k + 'px">' + f + "</div>");
                    j.length > 0 && (j.push('<div style="clear:left"></div>'), g = d(j, "width:10000px;"), l = g.height(), g.remove())
                }
            } else if (null == k || null == l) {
                for (e = 0; e < i.length; ++e)
                    f = i[e].label, f && j.push('<div class="tickLabel">' + f + "</div>");
                j.length > 0 && (g = d(j, ""), null == k && (k = g.children().width()), null == l && (l = g.find("div.tickLabel").height()), g.remove())
            }
            null == k && (k = 0), null == l && (l = 0), c.labelWidth = k, c.labelHeight = l
        }
        function B(b) {
            var c = b.labelWidth, d = b.labelHeight, e = b.options.position, f = b.options.tickLength, g = gb.grid.axisMargin, h = gb.grid.labelMargin, i = "x" == b.direction ? mb: nb, j = a.grep(i, function(a) {
                return a && a.options.position == e && a.reserveSpace
            });
            a.inArray(b, j) == j.length - 1 && (g = 0), null == f && (f = "full");
            var k = a.grep(i, function(a) {
                return a && a.reserveSpace
            }), l = 0 == a.inArray(b, k);
            l || "full" != f || (f = 5), isNaN( + f) || (h+=+f), "x" == b.direction ? (d += h, "bottom" == e ? (ob.bottom += d - 30, b.box = {
                top: qb - ob.bottom,
                height: d
            }) : b.box = {
                top: ob.top + g,
                height: d
            }) : (c += h, "left" == e ? b.box = {
                left: ob.left + g,
                width: c
            } : (ob.right += c + g, b.box = {
                left: pb - ob.right,
                width: c
            })), b.position = e, b.tickLength = f, b.box.padding = h, b.innermost = l
        }
        function C(a) {
            "x" == a.direction ? (a.box.left = ob.left, a.box.width = rb) : (a.box.top = ob.top, a.box.height = sb)
        }
        function D() {
            var b, c = n();
            if (a.each(c, function(a, b) {
                b.show = b.options.show, null == b.show && (b.show = b.used), b.reserveSpace = b.show || b.options.reserveSpace, E(b)
            }), allocatedAxes = a.grep(c, function(a) {
                return a.reserveSpace
            }), ob.left = ob.right = ob.top = ob.bottom = 0, gb.grid.show) {
                for (a.each(allocatedAxes, function(a, b) {
                    F(b), G(b), H(b, b.ticks), A(b)
                }), b = allocatedAxes.length - 1; b >= 0; --b)
                    B(allocatedAxes[b]);
                var d = gb.grid.minBorderMargin;
                if (null == d)
                    for (d = 0, b = 0; b < fb.length; ++b)
                        d = Math.max(d, fb[b].points.radius + fb[b].points.lineWidth / 2);
                for (var e in ob)
                    ob[e] += gb.grid.borderWidth, ob[e] = Math.max(d, ob[e])
            }
            rb = pb - ob.left - ob.right, sb = qb - ob.bottom - ob.top, a.each(c, function(a, b) {
                z(b)
            }), gb.grid.show && (a.each(allocatedAxes, function(a, b) {
                C(b)
            }), M()), T()
        }
        function E(a) {
            var b = a.options, c =+ (null != b.min ? b.min : a.datamin), d =+ (null != b.max ? b.max : a.datamax), e = d - c;
            if (0 == e) {
                var f = 0 == d ? 1: .01;
                null == b.min && (c -= f), (null == b.max || null != b.min) && (d += f)
            } else {
                var g = b.autoscaleMargin;
                null != g && (null == b.min && (c -= e * g, c < 0 && null != a.datamin && a.datamin >= 0 && (c = 0)), null == b.max && (d += e * g, d > 0 && null != a.datamax && a.datamax <= 0 && (d = 0)))
            }
            a.min = c, a.max = d
        }
        function F(b) {
            var d, e = b.options;
            d = "number" == typeof e.ticks && e.ticks > 0 ? e.ticks : .3 * Math.sqrt("x" == b.direction ? pb : qb);
            var f, g, h, i, j, k, l, m = (b.max - b.min) / d;
            if ("time" == e.mode) {
                var n = {
                    second: 1e3,
                    minute: 6e4,
                    hour: 36e5,
                    day: 864e5,
                    month: 2592e6,
                    year: 525949.2 * 60 * 1e3
                }, o = [[1, "second"], [2, "second"], [5, "second"], [10, "second"], [30, "second"], [1, "minute"], [2, "minute"], [5, "minute"], [10, "minute"], [30, "minute"], [1, "hour"], [2, "hour"], [4, "hour"], [8, "hour"], [12, "hour"], [1, "day"], [2, "day"], [3, "day"], [.25, "month"], [.5, "month"], [1, "month"], [2, "month"], [3, "month"], [6, "month"], [1, "year"]], p = 0;
                null != e.minTickSize && (p = "number" == typeof e.tickSize ? e.tickSize : e.minTickSize[0] * n[e.minTickSize[1]]);
                for (var j = 0; j < o.length - 1&&!(m < (o[j][0] * n[o[j][1]] + o[j + 1][0] * n[o[j + 1][1]]) / 2 && o[j][0] * n[o[j][1]] >= p); ++j);
                f = o[j][0], h = o[j][1], "year" == h && (k = Math.pow(10, Math.floor(Math.log(m / n.year) / Math.LN10)), l = m / n.year / k, f = l < 1.5 ? 1 : l < 3 ? 2 : l < 7.5 ? 5 : 10, f*=k), b.tickSize = e.tickSize || [f, h], g = function(a) {
                    var b = [], d = a.tickSize[0], e = a.tickSize[1], f = new Date(a.min), g = d * n[e];
                    "second" == e && f.setUTCSeconds(c(f.getUTCSeconds(), d)), "minute" == e && f.setUTCMinutes(c(f.getUTCMinutes(), d)), "hour" == e && f.setUTCHours(c(f.getUTCHours(), d)), "month" == e && f.setUTCMonth(c(f.getUTCMonth(), d)), "year" == e && f.setUTCFullYear(c(f.getUTCFullYear(), d)), f.setUTCMilliseconds(0), g >= n.minute && f.setUTCSeconds(0), g >= n.hour && f.setUTCMinutes(0), g >= n.day && f.setUTCHours(0), g >= 4 * n.day && f.setUTCDate(1), g >= n.year && f.setUTCMonth(0);
                    var h, i = 0, j = Number.NaN;
                    do 
                        if (h = j, j = f.getTime(), b.push(j), "month" == e)
                            if (d < 1) {
                                f.setUTCDate(1);
                                var k = f.getTime();
                                f.setUTCMonth(f.getUTCMonth() + 1);
                                var l = f.getTime();
                                f.setTime(j + i * n.hour + (l - k) * d), i = f.getUTCHours(), f.setUTCHours(0)
                            } else 
                                f.setUTCMonth(f.getUTCMonth() + d);
                        else 
                            "year" == e ? f.setUTCFullYear(f.getUTCFullYear() + d) : f.setTime(j + g);
                    while (j < a.max && j != h);
                    return b
                }, i = function(b, c) {
                    var d = new Date(b);
                    if (null != e.timeformat)
                        return a.plot.formatDate(d, e.timeformat, e.monthNames);
                    var f = c.tickSize[0] * n[c.tickSize[1]], g = c.max - c.min, h = e.twelveHourClock ? " %p": "";
                    return fmt = f < n.minute ? "%h:%M:%S" + h : f < n.day ? g < 2 * n.day ? "%h:%M" + h : "%b %d %h:%M" + h : f < n.month ? "%b %d" : f < n.year ? g < n.year ? "%b" : "%b %y" : "%y", a.plot.formatDate(d, fmt, e.monthNames)
                }
            } else {
                var q = e.tickDecimals, r =- Math.floor(Math.log(m) / Math.LN10);
                null != q && r > q && (r = q), k = Math.pow(10, - r), l = m / k, l < 1.5 ? f = 1 : l < 3 ? (f = 2, l > 2.25 && (null == q || r + 1 <= q) && (f = 2.5, ++r)) : f = l < 7.5 ? 5 : 10, f*=k, null != e.minTickSize && f < e.minTickSize && (f = e.minTickSize), b.tickDecimals = Math.max(0, null != q ? q : r), b.tickSize = e.tickSize || f, g = function(a) {
                    var b, d = [], e = c(a.min, a.tickSize), f = 0, g = Number.NaN;
                    do 
                        b = g, g = e + f * a.tickSize, d.push(g), ++f;
                    while (g < a.max && g != b);
                    return d
                }, i = function(a, b) {
                    return a.toFixed(b.tickDecimals)
                }
            }
            if (null != e.alignTicksWithAxis) {
                var s = ("x" == b.direction ? mb : nb)[e.alignTicksWithAxis - 1];
                if (s && s.used && s != b) {
                    var t = g(b);
                    if (t.length > 0 && (null == e.min && (b.min = Math.min(b.min, t[0])), null == e.max && t.length > 1 && (b.max = Math.max(b.max, t[t.length - 1]))), g = function(a) {
                        var b, c, d = [];
                        for (c = 0; c < s.ticks.length; ++c)
                            b = (s.ticks[c].v - s.min) / (s.max - s.min), b = a.min + b * (a.max - a.min), d.push(b);
                        return d
                    }, "time" != b.mode && null == e.tickDecimals) {
                        var u = Math.max(0, - Math.floor(Math.log(m) / Math.LN10) + 1), v = g(b);
                        v.length > 1 && /\..*0$/.test((v[1] - v[0]).toFixed(u)) || (b.tickDecimals = u)
                    }
                }
            }
            b.tickGenerator = g, b.tickFormatter = a.isFunction(e.tickFormatter) ? function(a, b, c) {
                return "" + e.tickFormatter(a, b, c)
            } : i
        }
        function G(b) {
            var c = b.options.ticks, d = [];
            null == c || "number" == typeof c && c > 0 ? d = b.tickGenerator(b) : c && (d = a.isFunction(c) ? c({
                min: b.min,
                max: b.max
            }) : c);
            var e, f, g=!0, h = 0;
            for (b.ticks = [], e = 0; e < d.length; ++e) {
                var i = null, j = null, k = d[e];
                "object" == typeof k ? (f =+ k[0], k.length > 1 && (i = k[1])) : f =+ k, null == i && (i = b.tickFormatter(f, b, g), j = b.tickFormatter(h, b, g), i === j && (i = "")), isNaN(f) || (b.ticks.push({
                    v: f,
                    label: i
                }), h = f, i && (g=!1))
            }
        }
        function H(a, b) {
            a.options.autoscaleMargin && b.length > 0 && (null == a.options.min && (a.min = Math.min(a.min, b[0].v)), null == a.options.max && b.length > 1 && (a.max = Math.max(a.max, b[b.length - 1].v)))
        }
        function I() {
            kb.clearRect(0, 0, pb, qb);
            var a = gb.grid;
            a.show && a.backgroundColor && K(), a.show&&!a.aboveData && L();
            for (var b = 0; b < fb.length; ++b)
                g(tb.drawSeries, [kb, fb[b]]), N(fb[b]);
            g(tb.draw, [kb]), a.show && a.aboveData && L()
        }
        function J(a, b) {
            var c, d, e, f, g = n();
            for (i = 0; i < g.length; ++i)
                if (c = g[i], c.direction == b && (f = b + c.n + "axis", a[f] || 1 != c.n || (f = b + "axis"), a[f])) {
                    d = a[f].from, e = a[f].to;
                    break
                }
            if (a[f] || (c = "x" == b ? mb[0] : nb[0], d = a[b + "1"], e = a[b + "2"]), null != d && null != e && d > e) {
                var h = d;
                d = e, e = h
            }
            return {
                from: d,
                to: e,
                axis: c
            }
        }
        function K() {
            kb.save(), kb.translate(ob.left, ob.top), kb.fillStyle = eb(gb.grid.backgroundColor, sb, 0, "rgba(255, 255, 255, 0)"), kb.fillRect(0, 0, rb, sb), kb.restore()
        }
        function L() {
            var b;
            kb.save(), kb.translate(ob.left, ob.top);
            var c = gb.grid.markings;
            if (c) {
                if (a.isFunction(c)) {
                    var d = ub.getAxes();
                    d.xmin = d.xaxis.min, d.xmax = d.xaxis.max, d.ymin = d.yaxis.min, d.ymax = d.yaxis.max, c = c(d)
                }
                for (b = 0; b < c.length; ++b) {
                    var e = c[b], f = J(e, "x"), g = J(e, "y");
                    null == f.from && (f.from = f.axis.min), null == f.to && (f.to = f.axis.max), null == g.from && (g.from = g.axis.min), null == g.to && (g.to = g.axis.max), f.to < f.axis.min || f.from > f.axis.max || g.to < g.axis.min || g.from > g.axis.max || (f.from = Math.max(f.from, f.axis.min), f.to = Math.min(f.to, f.axis.max), g.from = Math.max(g.from, g.axis.min), g.to = Math.min(g.to, g.axis.max), (f.from != f.to || g.from != g.to) && (f.from = f.axis.p2c(f.from), f.to = f.axis.p2c(f.to), g.from = g.axis.p2c(g.from), g.to = g.axis.p2c(g.to), f.from == f.to || g.from == g.to ? (kb.beginPath(), kb.strokeStyle = e.color || gb.grid.markingsColor, kb.lineWidth = e.lineWidth || gb.grid.markingsLineWidth, kb.moveTo(f.from, g.from), kb.lineTo(f.to, g.to), kb.stroke()) : (kb.fillStyle = e.color || gb.grid.markingsColor, kb.fillRect(f.from, g.to, f.to - f.from, g.from - g.to))))
                }
            }
            for (var d = n(), h = gb.grid.borderWidth, i = 0; i < d.length; ++i) {
                var j, k, l, m, o = d[i], p = o.box, q = o.tickLength;
                if (o.show && 0 != o.ticks.length) {
                    for (kb.strokeStyle = o.options.tickColor || a.color.parse(o.options.color).scale("a", .22).toString(), kb.lineWidth = 1, "x" == o.direction ? (j = 0, k = "full" == q ? "top" == o.position ? 0 : sb : p.top - ob.top + ("top" == o.position ? p.height : 0)) : (k = 0, j = "full" == q ? "left" == o.position ? 0 : rb : p.left - ob.left + ("left" == o.position ? p.width : 0)), o.innermost || (kb.beginPath(), l = m = 0, "x" == o.direction ? l = rb : m = sb, 1 == kb.lineWidth && (j = Math.floor(j) + .5, k = Math.floor(k) + .5), kb.moveTo(j, k), kb.lineTo(j + l, k + m), kb.stroke()), kb.beginPath(), b = 0; b < o.ticks.length; ++b) {
                        var r = o.ticks[b].v;
                        if (l = m = 0, !(r < o.min || r > o.max || "full" == q && h > 0 && (r == o.min || r == o.max))) {
                            if ("x" == o.direction)
                                j = o.p2c(r), m = "full" == q?-sb : q, "top" == o.position && (m =- m);
                            else {
                                if (k = o.p2c(r), k < 10)
                                    continue;
                                l = "full" == q?-rb : q, "left" == o.position && (l =- l)
                            }
                            1 == kb.lineWidth && ("x" == o.direction ? j = Math.floor(j) + .5 : k = Math.floor(k) + .5), kb.moveTo(j, k), kb.lineTo(j + l, k + m)
                        }
                    }
                    kb.stroke()
                }
            }
            h && (kb.lineWidth = h, kb.strokeStyle = gb.grid.borderColor, kb.strokeRect( - h / 2, - h / 2, rb + h, sb + h)), kb.restore()
        }
        function M() {
            b.find(".tickLabels").remove();
            for (var a = ['<div class="tickLabels" style="font-size:smaller">'], c = n(), d = 0; d < c.length; ++d) {
                var e = c[d], f = e.box;
                if (e.show) {
                    a.push('<div class="' + e.direction + "Axis " + e.direction + e.n + 'Axis" style="color:' + e.options.color + '">');
                    for (var g = 0; g < e.ticks.length; ++g) {
                        var h = e.ticks[g];
                        if (!(!h.label || h.v < e.min || h.v > e.max)) {
                            var i, j = {};
                            if ("x" == e.direction)
                                i = "center", j.left = Math.round(ob.left + e.p2c(h.v) - e.labelWidth / 2), "bottom" == e.position ? j.top = f.top + f.padding - 45 : j.bottom = qb - (f.top + f.height - f.padding);
                            else {
                                if (j.top = Math.round(ob.top + e.p2c(h.v) - e.labelHeight / 2), j.top < 15)
                                    continue;
                                "left" == e.position ? (j.right = pb - (f.left + f.width - f.padding) - 10, i = "right") : (j.left = f.left + f.padding, i = "left")
                            }
                            j.width = e.labelWidth;
                            var k = ["position:absolute", "text-align:" + i];
                            for (var l in j)
                                k.push(l + ":" + j[l] + "px");
                            a.push('<div class="tickLabel" style="' + k.join(";") + '">' + h.label + "</div>")
                        }
                    }
                    a.push("</div>")
                }
            }
            a.push("</div>"), b.append(a.join(""))
        }
        function N(a) {
            a.lines.show && O(a), a.bars.show && R(a), a.points.show && P(a)
        }
        function O(a) {
            function b(a, b, c, d, e) {
                var f = a.points, g = a.pointsize, h = null, i = null;
                kb.beginPath();
                for (var j = g; j < f.length; j += g) {
                    var k = f[j - g], l = f[j - g + 1], m = f[j], n = f[j + 1];
                    if (null != k && null != m) {
                        if (l <= n && l < e.min) {
                            if (n < e.min)
                                continue;
                            k = (e.min - l) / (n - l) * (m - k) + k, l = e.min
                        } else if (n <= l && n < e.min) {
                            if (l < e.min)
                                continue;
                            m = (e.min - l) / (n - l) * (m - k) + k, n = e.min
                        }
                        if (l >= n && l > e.max) {
                            if (n > e.max)
                                continue;
                            k = (e.max - l) / (n - l) * (m - k) + k, l = e.max
                        } else if (n >= l && n > e.max) {
                            if (l > e.max)
                                continue;
                            m = (e.max - l) / (n - l) * (m - k) + k, n = e.max
                        }
                        if (k <= m && k < d.min) {
                            if (m < d.min)
                                continue;
                            l = (d.min - k) / (m - k) * (n - l) + l, k = d.min
                        } else if (m <= k && m < d.min) {
                            if (k < d.min)
                                continue;
                            n = (d.min - k) / (m - k) * (n - l) + l, m = d.min
                        }
                        if (k >= m && k > d.max) {
                            if (m > d.max)
                                continue;
                            l = (d.max - k) / (m - k) * (n - l) + l, k = d.max
                        } else if (m >= k && m > d.max) {
                            if (k > d.max)
                                continue;
                            n = (d.max - k) / (m - k) * (n - l) + l, m = d.max
                        }(k != h || l != i) && kb.moveTo(d.p2c(k) + b, e.p2c(l) + c), h = m, i = n, kb.lineTo(d.p2c(m) + b, e.p2c(n) + c)
                    }
                }
                kb.stroke()
            }
            function c(a, b, c) {
                for (var d = a.points, e = a.pointsize, f = Math.min(Math.max(0, c.min), c.max), g = 0, h=!1, i = 1, j = 0, k = 0; ;) {
                    if (e > 0 && g > d.length + e)
                        break;
                    g += e;
                    var l = d[g - e], m = d[g - e + i], n = d[g], o = d[g + i];
                    if (h) {
                        if (e > 0 && null != l && null == n) {
                            k = g, e =- e, i = 2;
                            continue
                        }
                        if (e < 0 && g == j + e) {
                            kb.fill(), h=!1, e =- e, i = 1, g = j = k + e;
                            continue
                        }
                    }
                    if (null != l && null != n) {
                        if (l <= n && l < b.min) {
                            if (n < b.min)
                                continue;
                            m = (b.min - l) / (n - l) * (o - m) + m, l = b.min
                        } else if (n <= l && n < b.min) {
                            if (l < b.min)
                                continue;
                            o = (b.min - l) / (n - l) * (o - m) + m, n = b.min
                        }
                        if (l >= n && l > b.max) {
                            if (n > b.max)
                                continue;
                            m = (b.max - l) / (n - l) * (o - m) + m, l = b.max
                        } else if (n >= l && n > b.max) {
                            if (l > b.max)
                                continue;
                            o = (b.max - l) / (n - l) * (o - m) + m, n = b.max
                        }
                        if (h || (kb.beginPath(), kb.moveTo(b.p2c(l), c.p2c(f)), h=!0), m >= c.max && o >= c.max)
                            kb.lineTo(b.p2c(l), c.p2c(c.max)), kb.lineTo(b.p2c(n), c.p2c(c.max));
                        else if (m <= c.min && o <= c.min)
                            kb.lineTo(b.p2c(l), c.p2c(c.min)), kb.lineTo(b.p2c(n), c.p2c(c.min));
                        else {
                            var p = l, q = n;
                            m <= o && m < c.min && o >= c.min ? (l = (c.min - m) / (o - m) * (n - l) + l, m = c.min) : o <= m && o < c.min && m >= c.min && (n = (c.min - m) / (o - m) * (n - l) + l, o = c.min), m >= o && m > c.max && o <= c.max ? (l = (c.max - m) / (o - m) * (n - l) + l, m = c.max) : o >= m && o > c.max && m <= c.max && (n = (c.max - m) / (o - m) * (n - l) + l, o = c.max), l != p && kb.lineTo(b.p2c(p), c.p2c(m)), kb.lineTo(b.p2c(l), c.p2c(m)), kb.lineTo(b.p2c(n), c.p2c(o)), n != q && (kb.lineTo(b.p2c(n), c.p2c(o)), kb.lineTo(b.p2c(q), c.p2c(o)))
                        }
                    }
                }
            }
            kb.save(), kb.translate(ob.left, ob.top), kb.lineJoin = "round";
            var d = a.lines.lineWidth, e = a.shadowSize;
            if (d > 0 && e > 0) {
                kb.lineWidth = e, kb.strokeStyle = "rgba(0,0,0,0.1)";
                var f = Math.PI / 18;
                b(a.datapoints, Math.sin(f) * (d / 2 + e / 2), Math.cos(f) * (d / 2 + e / 2), a.xaxis, a.yaxis), kb.lineWidth = e / 2, b(a.datapoints, Math.sin(f) * (d / 2 + e / 4), Math.cos(f) * (d / 2 + e / 4), a.xaxis, a.yaxis)
            }
            kb.lineWidth = d, kb.strokeStyle = a.color;
            var g = S(a.lines, a.color, 0, sb);
            g && (kb.fillStyle = g, c(a.datapoints, a.xaxis, a.yaxis)), d > 0 && b(a.datapoints, 0, 0, a.xaxis, a.yaxis), kb.restore()
        }
        function P(a) {
            function b(a, b, c, d, e, f, g, h) {
                for (var i = a.points, j = a.pointsize, k = 0; k < i.length; k += j) {
                    var l = i[k], m = i[k + 1];
                    null == l || l < f.min || l > f.max || m < g.min || m > g.max || (kb.strokeStyle = "rgb(249, 249, 249)", kb.beginPath(), l = f.p2c(l), m = g.p2c(m) + d, "circle" == h ? kb.arc(l, m, b, 0, e ? Math.PI : 2 * Math.PI, !1) : h(kb, l, m, b, e), kb.closePath(), c && (kb.fillStyle = c, kb.fill()), kb.stroke())
                }
            }
            kb.save(), kb.translate(ob.left, ob.top);
            var c = a.points.lineWidth, d = a.shadowSize, e = a.points.radius, f = a.points.symbol;
            if (c > 0 && d > 0) {
                var g = d / 2;
                kb.lineWidth = g, kb.strokeStyle = "rgba(0,0,0,0.1)", b(a.datapoints, e, null, g + g / 2, !0, a.xaxis, a.yaxis, f), kb.strokeStyle = "rgba(0,0,0,0.2)", b(a.datapoints, e, null, g / 2, !0, a.xaxis, a.yaxis, f)
            }
            kb.lineWidth = c, kb.strokeStyle = a.color, b(a.datapoints, e, S(a.points, a.color), 0, !1, a.xaxis, a.yaxis, f), kb.restore()
        }
        function Q(a, b, c, d, e, f, g, h, i, j, k, l) {
            var m, n, o, p, q, r, s, t, u;
            k ? (t = r = s=!0, q=!1, m = c, n = a, p = b + d, o = b + e, n < m && (u = n, n = m, m = u, q=!0, r=!1)) : (q = r = s=!0, t=!1, m = a + d, n = a + e, o = c, p = b, p < o && (u = p, p = o, o = u, t=!0, s=!1)), n < h.min || m > h.max || p < i.min || o > i.max || (m < h.min && (m = h.min, q=!1), n > h.max && (n = h.max, r=!1), o < i.min && (o = i.min, t=!1), p > i.max && (p = i.max, s=!1), m = h.p2c(m), o = i.p2c(o), n = h.p2c(n), p = i.p2c(p), g && (j.beginPath(), j.moveTo(m, o), j.lineTo(m, p), j.lineTo(n, p), j.lineTo(n, o), j.fillStyle = g(o, p), j.fill()), l > 0 && (q || r || s || t) && (j.beginPath(), j.moveTo(m, o + f), q ? j.lineTo(m, p + f) : j.moveTo(m, p + f), s ? j.lineTo(n, p + f) : j.moveTo(n, p + f), r ? j.lineTo(n, o + f) : j.moveTo(n, o + f), t ? j.lineTo(m, o + f) : j.moveTo(m, o + f), j.stroke()))
        }
        function R(a) {
            function b(b, c, d, e, f, g, h) {
                for (var i = b.points, j = b.pointsize, k = 0; k < i.length; k += j)
                    null != i[k] && Q(i[k], i[k + 1], i[k + 2], c, d, e, f, g, h, kb, a.bars.horizontal, a.bars.lineWidth)
            }
            kb.save(), kb.translate(ob.left, ob.top), kb.lineWidth = a.bars.lineWidth, kb.strokeStyle = a.color;
            var c = "left" == a.bars.align ? 0: - a.bars.barWidth / 2, d = a.bars.fill ? function(b, c) {
                return S(a.bars, a.color, b, c)
            }
            : null;
            b(a.datapoints, c, c + a.bars.barWidth, 0, d, a.xaxis, a.yaxis), kb.restore()
        }
        function S(b, c, d, e) {
            var f = b.fill;
            if (!f)
                return null;
            if (b.fillColor)
                return eb(b.fillColor, d, e, c);
            var g = a.color.parse(c);
            return g.a = "number" == typeof f ? f : .4, g.normalize(), g.toString()
        }
        function T() {
            if (b.find(".legend").remove(), gb.legend.show) {
                for (var c, d, e = [], f=!1, g = gb.legend.labelFormatter, h = 0; h < fb.length; ++h)
                    c = fb[h], d = c.label, d && (h%gb.legend.noColumns == 0 && (f && e.push("</tr>"), e.push("<tr>"), f=!0), g && (d = g(d, c)), e.push('<td class="legendColorBox" data-index="' + h + '"><div data-index="' + h + '" style="border:1px solid ' + gb.legend.labelBoxBorderColor + ';padding:1px"><div data-index="' + h + '" style="background:' + c.color + ";border:5px solid " + c.color + '">&#x2713;</div></div></td><td class="legendLabel" data-index="' + h + '" style="color:' + c.color + '">' + d + "</td>"));
                if (f && e.push("</tr>"), 0 != e.length) {
                    var i = '<table style="font-size:smaller;color:' + gb.grid.color + '">' + e.join("") + "</table>";
                    if (null != gb.legend.container)
                        a(gb.legend.container).html(i);
                    else {
                        var j = "", k = gb.legend.position, l = gb.legend.margin;
                        null == l[0] && (l = [l, l]), "n" == k.charAt(0) ? j += "top:" + (l[1] + ob.top) + "px;" : "s" == k.charAt(0) && (j += "bottom:" + (l[1] + ob.bottom) + "px;"), "e" == k.charAt(1) ? j += "right:0px;" : "w" == k.charAt(1) && (j += "left:" + (l[0] + ob.left) + "px;");
                        var m = a('<div class="legend">' + i.replace('style="', 'style="position:absolute;' + j + ";") + "</div>").appendTo(b);
                        if (0 != gb.legend.backgroundOpacity) {
                            var n = gb.legend.backgroundColor;
                            null == n && (n = gb.grid.backgroundColor, n = n && "string" == typeof n ? a.color.parse(n) : a.color.extract(m, "background-color"), n.a = 1, n = n.toString());
                            var o = m.children();
                            a('<div style="position:absolute;width:' + o.width() + "px;height:" + o.height() + "px;" + j + "background-color:" + n + ';"> </div>').prependTo(m).css("opacity", gb.legend.backgroundOpacity)
                        }
                    }
                }
            }
        }
        function U(a, b, c) {
            var d, e, f = gb.grid.mouseActiveRadius, g = f * f + 1, h = null;
            for (d = fb.length - 1; d >= 0; --d)
                if (c(fb[d])) {
                    var i = fb[d], j = i.xaxis, k = i.yaxis, l = i.datapoints.points, m = i.datapoints.pointsize, n = j.c2p(a), o = k.c2p(b), p = f / j.scale, q = f / k.scale;
                    if (j.options.inverseTransform && (p = Number.MAX_VALUE), k.options.inverseTransform && (q = Number.MAX_VALUE), i.lines.show || i.points.show)
                        for (e = 0; e < l.length; e += m) {
                            var r = l[e], s = l[e + 1];
                            if (null != r&&!(r - n > p || r - n<-p || s - o > q || s - o<-q)) {
                                var t = Math.abs(j.p2c(r) - a), u = Math.abs(k.p2c(s) - b), v = t * t + u * u;
                                v < g && (g = v, h = [d, e / m])
                            }
                        }
                        if (i.bars.show&&!h) {
                            var w = "left" == i.bars.align ? 0: - i.bars.barWidth / 2, x = w + i.bars.barWidth;
                            for (e = 0; e < l.length; e += m) {
                                var r = l[e], s = l[e + 1], y = l[e + 2];
                                null != r && (fb[d].bars.horizontal ? n <= Math.max(y, r) && n >= Math.min(y, r) && o >= s + w && o <= s + x : n >= r + w && n <= r + x && o >= Math.min(y, s) && o <= Math.max(y, s)) && (h = [d, e / m])
                            }
                        }
                    }
            return h ? (d = h[0], e = h[1], m = fb[d].datapoints.pointsize, {
                datapoint: fb[d].datapoints.points.slice(e * m, (e + 1) * m),
                dataIndex: e,
                series: fb[d],
                seriesIndex: d
            }) : null
        }
        function V(a) {
            gb.grid.hoverable && Y("plothover", a, function(a) {
                return 0 != a.hoverable
            })
        }
        function W(a) {
            gb.grid.hoverable && Y("plothover", a, function() {
                return !1
            })
        }
        function X(a) {
            Y("plotclick", a, function(a) {
                return 0 != a.clickable
            })
        }
        function Y(a, c, d) {
            var e = jb.offset(), f = c.pageX - e.left - ob.left, g = c.pageY - e.top - ob.top, h = o({
                left: f,
                top: g
            });
            h.pageX = c.pageX, h.pageY = c.pageY;
            var i = U(f, g, d);
            if (i && (i.pageX = parseInt(i.series.xaxis.p2c(i.datapoint[0]) + e.left + ob.left), i.pageY = parseInt(i.series.yaxis.p2c(i.datapoint[1]) + e.top + ob.top)), gb.grid.autoHighlight) {
                for (var j = 0; j < vb.length; ++j) {
                    var k = vb[j];
                    k.auto != a || i && k.series == i.series && k.point[0] == i.datapoint[0] && k.point[1] == i.datapoint[1] || ab(k.series, k.point)
                }
                i && _(i.series, i.datapoint, a)
            }
            b.trigger(a, [h, i])
        }
        function Z() {
            wb || (wb = setTimeout($, 30))
        }
        function $() {
            wb = null, lb.save(), lb.clearRect(0, 0, pb, qb), lb.translate(ob.left, ob.top);
            var a, b;
            for (a = 0; a < vb.length; ++a)
                b = vb[a], b.series.bars.show ? db(b.series, b.point) : cb(b.series, b.point);
            lb.restore(), g(tb.drawOverlay, [lb])
        }
        function _(a, b, c) {
            if ("number" == typeof a && (a = fb[a]), "number" == typeof b) {
                var d = a.datapoints.pointsize;
                b = a.datapoints.points.slice(d * b, d * (b + 1))
            }
            var e = bb(a, b);
            - 1 == e ? (vb.push({
                series: a,
                point: b,
                auto: c
            }), Z()) : c || (vb[e].auto=!1)
        }
        function ab(a, b) {
            null == a && null == b && (vb = [], Z()), "number" == typeof a && (a = fb[a]), "number" == typeof b && (b = a.data[b]);
            var c = bb(a, b);
            - 1 != c && (vb.splice(c, 1), Z())
        }
        function bb(a, b) {
            for (var c = 0; c < vb.length; ++c) {
                var d = vb[c];
                if (d.series == a && d.point[0] == b[0] && d.point[1] == b[1])
                    return c
            }
            return - 1
        }
        function cb(b, c) {
            var d = c[0], e = c[1], f = b.xaxis, g = b.yaxis;
            if (!(d < f.min || d > f.max || e < g.min || e > g.max)) {
                var h = b.points.radius + b.points.lineWidth / 2;
                lb.lineWidth = h, lb.strokeStyle = a.color.parse(b.color).scale("a", .5).toString();
                var i = 1.5 * h, d = f.p2c(d), e = g.p2c(e);
                lb.beginPath(), "circle" == b.points.symbol ? lb.arc(d, e, i, 0, 2 * Math.PI, !1) : b.points.symbol(lb, d, e, i, !1), lb.closePath(), lb.fillStyle = "rgb(255, 255, 255)", lb.fill(), lb.stroke()
            }
        }
        function db(b, c) {
            lb.lineWidth = b.bars.lineWidth, lb.strokeStyle = a.color.parse(b.color).scale("a", .5).toString();
            var d = a.color.parse(b.color).scale("a", .5).toString(), e = "left" == b.bars.align ? 0: - b.bars.barWidth / 2;
            Q(c[0], c[1], c[2] || 0, e, e + b.bars.barWidth, 0, function() {
                return d
            }, b.xaxis, b.yaxis, lb, b.bars.horizontal, b.bars.lineWidth)
        }
        function eb(b, c, d, e) {
            if ("string" == typeof b)
                return b;
            for (var f = kb.createLinearGradient(0, d, 0, c), g = 0, h = b.colors.length; g < h; ++g) {
                var i = b.colors[g];
                if ("string" != typeof i) {
                    var j = a.color.parse(e);
                    null != i.brightness && (j = j.scale("rgb", i.brightness)), null != i.opacity && (j.a*=i.opacity), i = j.toString()
                }
                f.addColorStop(g / (h - 1), i)
            }
            return f
        }
        var fb = [], gb = {
            colors: ["#edc240", "#afd8f8", "#cb4b4b", "#4da74d", "#9440ed"],
            legend: {
                show: !0,
                noColumns: 1,
                labelFormatter: null,
                labelBoxBorderColor: "#ccc",
                container: null,
                position: "ne",
                margin: 5,
                backgroundColor: null,
                backgroundOpacity: .85
            },
            xaxis: {
                show: null,
                position: "bottom",
                mode: null,
                color: null,
                tickColor: null,
                transform: null,
                inverseTransform: null,
                min: null,
                max: null,
                autoscaleMargin: null,
                ticks: null,
                tickFormatter: null,
                labelWidth: null,
                labelHeight: null,
                reserveSpace: null,
                tickLength: null,
                alignTicksWithAxis: null,
                tickDecimals: null,
                tickSize: null,
                minTickSize: null,
                monthNames: null,
                timeformat: null,
                twelveHourClock: !1
            },
            yaxis: {
                autoscaleMargin: .02,
                position: "left"
            },
            xaxes: [],
            yaxes: [],
            series: {
                points: {
                    show: !1,
                    radius: 3,
                    lineWidth: 2,
                    fill: !0,
                    fillColor: "#ffffff",
                    symbol: "circle"
                },
                lines: {
                    lineWidth: 2,
                    fill: !1,
                    fillColor: null,
                    steps: !1
                },
                bars: {
                    show: !1,
                    lineWidth: 2,
                    barWidth: 1,
                    fill: !0,
                    fillColor: null,
                    align: "left",
                    horizontal: !1
                },
                shadowSize: 3
            },
            grid: {
                show: !0,
                aboveData: !1,
                color: "#545454",
                backgroundColor: null,
                borderColor: null,
                tickColor: null,
                labelMargin: 5,
                axisMargin: 8,
                borderWidth: 2,
                minBorderMargin: null,
                markings: null,
                markingsColor: "#f4f4f4",
                markingsLineWidth: 2,
                clickable: !1,
                hoverable: !1,
                autoHighlight: !0,
                mouseActiveRadius: 10
            },
            hooks: {}
        }, hb = null, ib = null, jb = null, kb = null, lb = null, mb = [], nb = [], ob = {
            left: 0,
            right: 0,
            top: 0,
            bottom: 0
        }, pb = 0, qb = 0, rb = 0, sb = 0, tb = {
            processOptions: [],
            processRawData: [],
            processDatapoints: [],
            drawSeries: [],
            draw: [],
            bindEvents: [],
            drawOverlay: [],
            shutdown: []
        }, ub = this;
        ub.setData = k, ub.setupGrid = D, ub.draw = I, ub.getPlaceholder = function() {
            return b
        }, ub.getCanvas = function() {
            return hb
        }, ub.getPlotOffset = function() {
            return ob
        }, ub.width = function() {
            return rb
        }, ub.height = function() {
            return sb
        }, ub.offset = function() {
            var a = jb.offset();
            return a.left += ob.left, a.top += ob.top, a
        }, ub.getData = function() {
            return fb
        }, ub.getAxes = function() {
            var b = {};
            return a.each(mb.concat(nb), function(a, c) {
                c && (b[c.direction + (1 != c.n ? c.n : "") + "axis"] = c)
            }), b
        }, ub.getXAxes = function() {
            return mb
        }, ub.getYAxes = function() {
            return nb
        }, ub.c2p = o, ub.p2c = p, ub.getOptions = function() {
            return gb
        }, ub.highlight = _, ub.unhighlight = ab, ub.triggerRedrawOverlay = Z, ub.pointOffset = function(a) {
            return {
                left: parseInt(mb[m(a, "x") - 1].p2c( + a.x) + ob.left),
                top: parseInt(nb[m(a, "y") - 1].p2c( + a.y) + ob.top)
            }
        }, ub.shutdown = y, ub.resize = function() {
            u(), v(hb), v(ib)
        }, ub.hooks = tb, h(ub), j(e), w(), k(d), D(), I(), x();
        var vb = [], wb = null
    }
    function c(a, b) {
        return b * Math.floor(a / b)
    }
    a.plot = function(c, d, e) {
        var f = new b(a(c), d, e, a.plot.plugins);
        return f
    }, a.plot.version = "0.7", a.plot.plugins = [], a.plot.formatDate = function(a, b, c) {
        var d = function(a) {
            return a = "" + a, 1 == a.length ? "0" + a : a
        }, e = [], f=!1, g=!1, h = a.getUTCHours(), i = h < 12;
        null == c && (c = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]), - 1 != b.search(/%p|%P/) && (h > 12 ? h -= 12 : 0 == h && (h = 12));
        for (var j = 0; j < b.length; ++j) {
            var k = b.charAt(j);
            if (f) {
                switch (k) {
                case"h":
                    k = "" + h;
                    break;
                case"H":
                    k = d(h);
                    break;
                case"M":
                    k = d(a.getUTCMinutes());
                    break;
                case"S":
                    k = d(a.getUTCSeconds());
                    break;
                case"d":
                    k = "" + a.getUTCDate();
                    break;
                case"m":
                    k = "" + (a.getUTCMonth() + 1);
                    break;
                case"y":
                    k = "" + a.getUTCFullYear();
                    break;
                case"b":
                    k = "" + c[a.getUTCMonth()];
                    break;
                case"p":
                    k = i ? "am" : "pm";
                    break;
                case"P":
                    k = i ? "AM" : "PM";
                    break;
                case"0":
                    k = "", g=!0
                }
                k && g && (k = d(k), g=!1), e.push(k), g || (f=!1)
            } else 
                "%" == k ? f=!0 : e.push(k)
        }
        return e.join("")
    }
}(jQuery), $(function() {
    function a(a) {
        function b() {
            var b = [a];
            return null != a.getOptions().multihighlight.linkedPlots && $.each(a.getOptions().multihighlight.linkedPlots, function(a, c) {
                b.push(c)
            }), b
        }
        function c(a, c) {
            $.each(b(), function(a, b) {
                if (null!=!b.getOptions().multihighlight.mode) {
                    b.unhighlight(), b.getPlaceholder().trigger("unmultihighlighted"), - 1 !== b.getOptions().grid.markings[1].xaxis.from && (b.getOptions().grid.markings[1].xaxis = {
                        from: - 1,
                        to: - 1
                    }, b.draw());
                    var d = "x" == b.getOptions().multihighlight.mode ? c.x: c.y, e = "x" == b.getOptions().multihighlight.mode ? 0: 1, f = new Array;
                    $.each(b.getData(), function(a, c) {
                        var g;
                        for (g = 0; g < c.data.length&&!(null == c.data[g] || c.data[g][e] > d); g++);
                        if (0 != g && null != c.data[g]) {
                            var h = g - 1;
                            d - c.data[g - 1][e] > Math.abs(d - c.data[g][e]) && (h = g), b.highlight(a, h), f.push({
                                serieIndex: a,
                                dataIndex: h
                            });
                            var i = b.getOptions().grid.markings[1].xaxis.from;
                            i !== c.data[h][e] && (b.getOptions().grid.markings[1].xaxis = {
                                from: c.data[h][e],
                                to: c.data[h][e]
                            }, b.draw())
                        }
                    }), f.length > 0 ? b.getPlaceholder().trigger("multihighlighted", [c, f]) : b.getPlaceholder().trigger("unmultihighlighted")
                }
            })
        }
        function d() {
            $.each(b(), function(a, b) {
                null!=!b.getOptions().multihighlight.mode && (b.unhighlight(), b.getPlaceholder().trigger("unmultihighlighted"), - 1 !== b.getOptions().grid.markings[1].xaxis.from && (b.getOptions().grid.markings[1].xaxis = {
                    from: - 1,
                    to: - 1
                }, b.draw()))
            })
        }
        a.hooks.bindEvents.push(function(a) {
            a.getOptions().multihighlight.mode && (a.getPlaceholder().bind("plothover", c), a.getPlaceholder().bind("mouseout", d))
        }), a.hooks.shutdown.push(function(a) {
            a.getPlaceholder().unbind("plothover", c), a.getPlaceholder().unbind("mouseout", d)
        })
    }
    var b = {
        multihighlight: {
            mode: null,
            linkedPlots: null
        }
    };
    $.plot.plugins.push({
        init: a,
        options: b,
        name: "multihighlight",
        version: "1.0"
    })
}), function(a) {
    var b = function() {
        var b = {
            years: "datepickerViewYears",
            months: "datepickerViewMonths",
            days: "datepickerViewDays"
        }, c = {
            wrapper: '<div class="datepicker"><div class="datepickerContainer"><table width="100%" cellspacing="0" cellpadding="0"><tbody><tr></tr></tbody></table></div></div>',
            head: ["<td>", '<table cellspacing="0" cellpadding="0" width="100%">', "<thead>", "<tr>", '<th class="datepickerGoPrev"><a href="#"><span><%=prev%></span></a></th>', '<th colspan="5" class="datepickerMonth"><a href="#"><span></span></a></th>', '<th class="datepickerGoNext"><a href="#"><span><%=next%></span></a></th>', "</tr>", '<tr class="datepickerDoW">', "<th><span><%=day1%></span></th>", "<th><span><%=day2%></span></th>", "<th><span><%=day3%></span></th>", "<th><span><%=day4%></span></th>", "<th><span><%=day5%></span></th>", "<th><span><%=day6%></span></th>", "<th><span><%=day7%></span></th>", "</tr>", "</thead>", "</table></td>"],
            space: '<td class="datepickerSpace"><div></div></td>',
            days: ['<tbody class="datepickerDays">', "<tr>", '<td class="datepickerFirstDay <%=weeks[0].days[0].classname%>"><a href="#"><span><%=weeks[0].days[0].text%></span></a></td>', '<td class="<%=weeks[0].days[1].classname%>"><a href="#"><span><%=weeks[0].days[1].text%></span></a></td>', '<td class="<%=weeks[0].days[2].classname%>"><a href="#"><span><%=weeks[0].days[2].text%></span></a></td>', '<td class="<%=weeks[0].days[3].classname%>"><a href="#"><span><%=weeks[0].days[3].text%></span></a></td>', '<td class="<%=weeks[0].days[4].classname%>"><a href="#"><span><%=weeks[0].days[4].text%></span></a></td>', '<td class="<%=weeks[0].days[5].classname%>"><a href="#"><span><%=weeks[0].days[5].text%></span></a></td>', '<td class="datepickerLastDay <%=weeks[0].days[6].classname%>"><a href="#"><span><%=weeks[0].days[6].text%></span></a></td>', "</tr>", "<tr>", '<td class="datepickerFirstDay <%=weeks[1].days[0].classname%>"><a href="#"><span><%=weeks[1].days[0].text%></span></a></td>', '<td class="<%=weeks[1].days[1].classname%>"><a href="#"><span><%=weeks[1].days[1].text%></span></a></td>', '<td class="<%=weeks[1].days[2].classname%>"><a href="#"><span><%=weeks[1].days[2].text%></span></a></td>', '<td class="<%=weeks[1].days[3].classname%>"><a href="#"><span><%=weeks[1].days[3].text%></span></a></td>', '<td class="<%=weeks[1].days[4].classname%>"><a href="#"><span><%=weeks[1].days[4].text%></span></a></td>', '<td class="<%=weeks[1].days[5].classname%>"><a href="#"><span><%=weeks[1].days[5].text%></span></a></td>', '<td class="datepickerLastDay <%=weeks[1].days[6].classname%>"><a href="#"><span><%=weeks[1].days[6].text%></span></a></td>', "</tr>", "<tr>", '<td class="datepickerFirstDay <%=weeks[2].days[0].classname%>"><a href="#"><span><%=weeks[2].days[0].text%></span></a></td>', '<td class="<%=weeks[2].days[1].classname%>"><a href="#"><span><%=weeks[2].days[1].text%></span></a></td>', '<td class="<%=weeks[2].days[2].classname%>"><a href="#"><span><%=weeks[2].days[2].text%></span></a></td>', '<td class="<%=weeks[2].days[3].classname%>"><a href="#"><span><%=weeks[2].days[3].text%></span></a></td>', '<td class="<%=weeks[2].days[4].classname%>"><a href="#"><span><%=weeks[2].days[4].text%></span></a></td>', '<td class="<%=weeks[2].days[5].classname%>"><a href="#"><span><%=weeks[2].days[5].text%></span></a></td>', '<td class="datepickerLastDay <%=weeks[2].days[6].classname%>"><a href="#"><span><%=weeks[2].days[6].text%></span></a></td>', "</tr>", "<tr>", '<td class="datepickerFirstDay <%=weeks[3].days[0].classname%>"><a href="#"><span><%=weeks[3].days[0].text%></span></a></td>', '<td class="<%=weeks[3].days[1].classname%>"><a href="#"><span><%=weeks[3].days[1].text%></span></a></td>', '<td class="<%=weeks[3].days[2].classname%>"><a href="#"><span><%=weeks[3].days[2].text%></span></a></td>', '<td class="<%=weeks[3].days[3].classname%>"><a href="#"><span><%=weeks[3].days[3].text%></span></a></td>', '<td class="<%=weeks[3].days[4].classname%>"><a href="#"><span><%=weeks[3].days[4].text%></span></a></td>', '<td class="<%=weeks[3].days[5].classname%>"><a href="#"><span><%=weeks[3].days[5].text%></span></a></td>', '<td class="datepickerLastDay <%=weeks[3].days[6].classname%>"><a href="#"><span><%=weeks[3].days[6].text%></span></a></td>', "</tr>", "<tr>", '<td class="datepickerFirstDay <%=weeks[4].days[0].classname%>"><a href="#"><span><%=weeks[4].days[0].text%></span></a></td>', '<td class="<%=weeks[4].days[1].classname%>"><a href="#"><span><%=weeks[4].days[1].text%></span></a></td>', '<td class="<%=weeks[4].days[2].classname%>"><a href="#"><span><%=weeks[4].days[2].text%></span></a></td>', '<td class="<%=weeks[4].days[3].classname%>"><a href="#"><span><%=weeks[4].days[3].text%></span></a></td>', '<td class="<%=weeks[4].days[4].classname%>"><a href="#"><span><%=weeks[4].days[4].text%></span></a></td>', '<td class="<%=weeks[4].days[5].classname%>"><a href="#"><span><%=weeks[4].days[5].text%></span></a></td>', '<td class="datepickerLastDay <%=weeks[4].days[6].classname%>"><a href="#"><span><%=weeks[4].days[6].text%></span></a></td>', "</tr>", "<tr>", '<td class="datepickerFirstDay <%=weeks[5].days[0].classname%>"><a href="#"><span><%=weeks[5].days[0].text%></span></a></td>', '<td class="<%=weeks[5].days[1].classname%>"><a href="#"><span><%=weeks[5].days[1].text%></span></a></td>', '<td class="<%=weeks[5].days[2].classname%>"><a href="#"><span><%=weeks[5].days[2].text%></span></a></td>', '<td class="<%=weeks[5].days[3].classname%>"><a href="#"><span><%=weeks[5].days[3].text%></span></a></td>', '<td class="<%=weeks[5].days[4].classname%>"><a href="#"><span><%=weeks[5].days[4].text%></span></a></td>', '<td class="<%=weeks[5].days[5].classname%>"><a href="#"><span><%=weeks[5].days[5].text%></span></a></td>', '<td class="datepickerLastDay <%=weeks[5].days[6].classname%>"><a href="#"><span><%=weeks[5].days[6].text%></span></a></td>', "</tr>", "</tbody>"],
            months: ['<tbody class="<%=className%>">', "<tr>", '<td colspan="2"><a href="#"><span><%=data[0]%></span></a></td>', '<td colspan="2"><a href="#"><span><%=data[1]%></span></a></td>', '<td colspan="2"><a href="#"><span><%=data[2]%></span></a></td>', '<td colspan="2"><a href="#"><span><%=data[3]%></span></a></td>', "</tr>", "<tr>", '<td colspan="2"><a href="#"><span><%=data[4]%></span></a></td>', '<td colspan="2"><a href="#"><span><%=data[5]%></span></a></td>', '<td colspan="2"><a href="#"><span><%=data[6]%></span></a></td>', '<td colspan="2"><a href="#"><span><%=data[7]%></span></a></td>', "</tr>", "<tr>", '<td colspan="2"><a href="#"><span><%=data[8]%></span></a></td>', '<td colspan="2"><a href="#"><span><%=data[9]%></span></a></td>', '<td colspan="2"><a href="#"><span><%=data[10]%></span></a></td>', '<td colspan="2"><a href="#"><span><%=data[11]%></span></a></td>', "</tr>", "</tbody>"]
        }, d = {
            flat: !1,
            starts: 1,
            prev: "&#9664;",
            next: "&#9654;",
            lastSel: !1,
            mode: "single",
            view: "days",
            calendars: 1,
            format: "Y-m-d",
            position: "bottom",
            eventName: "click",
            onRender: function() {
                return {}
            },
            onChange: function() {
                return !0
            },
            onShow: function() {
                return !0
            },
            onBeforeShow: function() {
                return !0
            },
            onHide: function() {
                return !0
            },
            showWeeks: !1,
            locale: {
                days: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                daysShort: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                daysMin: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                months: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
                monthsShort: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                weekMin: "wk"
            }
        }, e = function(b) {
            var d, e, f, h, i, j, k, l, m, n = a(b).data("datepicker"), o = a(b), p = Math.floor(n.calendars / 2), q = 0;
            o.find("td>table tbody").remove();
            for (var r = 0; r < n.calendars; r++) {
                switch (d = new Date(n.current), d.addMonths( - p + r), m = o.find("table").eq(r + 1), m[0].className) {
                case"datepickerViewDays":
                    f = g(d, "B, Y");
                    break;
                case"datepickerViewMonths":
                    f = d.getFullYear();
                    break;
                case"datepickerViewYears":
                    f = d.getFullYear() - 6 + " - " + (d.getFullYear() + 5)
                }
                m.find("thead tr:first th:eq(1) span").text(f), f = d.getFullYear() - 6, e = {
                    data: [],
                    className: "datepickerYears"
                };
                for (var s = 0; s < 12; s++)
                    e.data.push(f + s);
                l = tmpl(c.months.join(""), e), d.setDate(1), e = {
                    weeks: [],
                    test: 10
                }, h = d.getMonth();
                var t, f = (d.getDay() - n.starts)%7;
                for (d.addDays( - (f + (f < 0 ? 7 : 0))), i =- 1, q = 0; q < 42;) {
                    j = parseInt(q / 7, 10), k = q%7, e.weeks[j] || (i = d.getWeekNumber(), e.weeks[j] = {
                        week: i,
                        days: []
                    }), e.weeks[j].days[k] = {
                        text: d.getDate(),
                        classname: []
                    }, t = e.weeks[j].days[k], h != d.getMonth() && t.classname.push("datepickerNotInMonth"), 0 == d.getDay() && t.classname.push("datepickerSunday"), 6 == d.getDay() && t.classname.push("datepickerSaturday");
                    var u = n.onRender(d), v = d.valueOf();
                    d.addDays(1), u.selected || n.date == v || a.inArray(v, n.date)>-1 || "range" == n.mode && v >= n.date[0] && v <= n.date[1] ? (t.classname.push("datepickerSelected"), n.date[0] ? v == n.date[0] ? t.classname.push(d.valueOf() < n.date[n.date.length - 1] ? "datepickerSelectedFirst" : "datepickerSelectedSingle") : d.valueOf() > n.date[n.date.length - 1] && t.classname.push("datepickerSelectedLast") : t.classname.push("datepickerSelectedSingle")) : t.classname.push("datepickerNotSelected"), u.disabled && t.classname.push("datepickerDisabled"), u.className && t.classname.push(u.className), t.classname = t.classname.join(" "), q++
                }
                l = tmpl(c.days.join(""), e) + l, e = {
                    data: n.locale.monthsShort,
                    className: "datepickerMonths"
                }, l = tmpl(c.months.join(""), e) + l, m.append(l)
            }
        }, f = function(a, b) {
            if (a.constructor == Date)
                return new Date(a);
            for (var c, d, e, f, g, h = a.split(/\W+/), i = b.split(/\W+/), j = new Date, k = 0; k < h.length; k++)
                switch (i[k]) {
                case"d":
                case"e":
                    c = parseInt(h[k], 10);
                    break;
                case"m":
                    d = parseInt(h[k], 10) - 1;
                    break;
                case"Y":
                case"y":
                    e = parseInt(h[k], 10), e += e > 100 ? 0 : e < 29 ? 2e3 : 1900;
                    break;
                case"H":
                case"I":
                case"k":
                case"l":
                    f = parseInt(h[k], 10);
                    break;
                case"P":
                case"p":
                    /pm/i.test(h[k]) && f < 12 ? f += 12 : /am/i.test(h[k]) && f >= 12 && (f -= 12);
                    break;
                case"M":
                    g = parseInt(h[k], 10)
                }
            return new Date(void 0 === e ? j.getFullYear() : e, void 0 === d ? j.getMonth() : d, void 0 === c ? j.getDate() : c, void 0 === f ? j.getHours() : f, void 0 === g ? j.getMinutes() : g, 0)
        }, g = function(a, b) {
            var c = a.getMonth(), d = a.getDate(), e = a.getFullYear(), f = (a.getWeekNumber(), a.getDay()), g = a.getHours(), h = g >= 12, i = h ? g - 12: g, j = a.getDayOfYear();
            0 == i && (i = 12);
            for (var k, l = a.getMinutes(), m = a.getSeconds(), n = b.split(""), o = 0; o < n.length; o++) {
                switch (k = n[o], n[o]) {
                case"a":
                    k = a.getDayName();
                    break;
                case"A":
                    k = a.getDayName(!0);
                    break;
                case"b":
                    k = a.getMonthName();
                    break;
                case"B":
                    k = a.getMonthName(!0);
                    break;
                case"C":
                    k = 1 + Math.floor(e / 100);
                    break;
                case"d":
                    k = d < 10 ? "0" + d : d;
                    break;
                case"e":
                    k = d;
                    break;
                case"H":
                    k = g < 10 ? "0" + g : g;
                    break;
                case"I":
                    k = i < 10 ? "0" + i : i;
                    break;
                case"j":
                    k = j < 100 ? j < 10 ? "00" + j : "0" + j : j;
                    break;
                case"k":
                    k = g;
                    break;
                case"l":
                    k = i;
                    break;
                case"m":
                    k = c < 9 ? "0" + (1 + c) : 1 + c;
                    break;
                case"M":
                    k = l < 10 ? "0" + l : l;
                    break;
                case"p":
                case"P":
                    k = h ? "PM" : "AM";
                    break;
                case"s":
                    k = Math.floor(a.getTime() / 1e3);
                    break;
                case"S":
                    k = m < 10 ? "0" + m : m;
                    break;
                case"u":
                    k = f + 1;
                    break;
                case"w":
                    k = f;
                    break;
                case"y":
                    k = ("" + e).substr(2, 2);
                    break;
                case"Y":
                    k = e
                }
                n[o] = k
            }
            return n.join("")
        }, h = function(a) {
            Date.prototype.tempDate || (Date.prototype.tempDate = null, Date.prototype.months = a.months, Date.prototype.monthsShort = a.monthsShort, Date.prototype.days = a.days, Date.prototype.daysShort = a.daysShort, Date.prototype.getMonthName = function(a) {
                return this[a ? "months": "monthsShort"][this.getUTCMonth()]
            }, Date.prototype.getDayName = function(a) {
                return this[a ? "days": "daysShort"][this.getUTCDay()]
            }, Date.prototype.addDays = function(a) {
                this.setDate(this.getDate() + a), this.tempDate = this.getDate()
            }, Date.prototype.addMonths = function(a) {
                null == this.tempDate && (this.tempDate = this.getDate()), this.setDate(1), this.setMonth(this.getMonth() + a), this.setDate(Math.min(this.tempDate, this.getMaxDays()))
            }, Date.prototype.addYears = function(a) {
                null == this.tempDate && (this.tempDate = this.getDate()), this.setDate(1), this.setFullYear(this.getFullYear() + a), this.setDate(Math.min(this.tempDate, this.getMaxDays()))
            }, Date.prototype.getMaxDays = function() {
                var a, b = new Date(Date.parse(this)), c = 28;
                for (a = b.getMonth(), c = 28; b.getMonth() == a;)
                    c++, b.setDate(c);
                return c - 1
            }, Date.prototype.getFirstDay = function() {
                var a = new Date(Date.parse(this));
                return a.setDate(1), a.getDay()
            }, Date.prototype.getWeekNumber = function() {
                var a = new Date(this);
                a.setDate(a.getDate() - (a.getDay() + 6)%7 + 3);
                var b = a.valueOf();
                return a.setMonth(0), a.setDate(4), Math.round((b - a.valueOf()) / 6048e5) + 1
            }, Date.prototype.getDayOfYear = function() {
                var a = new Date(this.getFullYear(), this.getMonth(), this.getDate(), 0, 0, 0), b = new Date(this.getFullYear(), 0, 0, 0, 0, 0), c = a - b;
                return Math.floor(c / 24 * 60 * 60 * 1e3)
            })
        }, i = function(a) {
            return new Date(a.getFullYear(), a.getMonth(), 1)
        }, j = function(a) {
            return new Date(a.getFullYear(), a.getMonth() + 1, 0)
        }, k = function(a) {
            return new Date(a.getFullYear(), 0, 1)
        }, l = function(a) {
            return new Date(a.getFullYear() + 1, 0, 0)
        }, m = function(b) {
            a(b.target).is("span") && (b.target = b.target.parentNode);
            var c = a(b.target);
            if (c.is("a")) {
                if (b.target.blur(), c.hasClass("datepickerDisabled"))
                    return !1;
                var d = a(this).data("datepicker"), f = c.parent(), g = f.parent().parent().parent(), h = a("table", this).index(g.get(0)) - 1, m = new Date(d.current), o=!1, p=!1;
                if (f.is("th"))
                    if (f.hasClass("datepickerWeek") && "range" == d.mode&&!f.next().hasClass("datepickerDisabled")) {
                        var q = parseInt(f.next().text(), 10);
                        m.addMonths(h - Math.floor(d.calendars / 2)), f.next().hasClass("datepickerNotInMonth") && m.addMonths(q > 15?-1 : 1), m.setDate(q), d.date[0] = m.setHours(0, 0, 0, 0).valueOf(), m.setHours(23, 59, 59, 0), m.addDays(6), d.date[1] = m.valueOf(), p=!0, o=!0, d.lastSel=!1
                    } else {
                        if (f.hasClass("datepickerMonth"))
                            return !1;
                            if (f.parent().parent().is("thead")) {
                                var r = new Date(d.current), s = f.hasClass("datepickerGoPrev");
                                switch (g.get(0).className) {
                                case"datepickerViewDays":
                                    r.addMonths(s?-1 : 1), r = s ? j(r) : i(r);
                                    break;
                                case"datepickerViewMonths":
                                    r.addYears(s?-1 : 1), r = s ? l(r) : k(r);
                                    break;
                                case"datepickerViewYears":
                                    r.addYears(s?-12 : 12), r = s ? l(r) : k(r)
                                }
                                d.onRender(r).disabled!==!0 && (d.current = r), p=!0
                            }
                        } else if (f.is("td")&&!f.hasClass("datepickerDisabled")) {
                    switch (g.get(0).className) {
                    case"datepickerViewMonths":
                        d.current.setMonth(g.find("tbody.datepickerMonths td").index(f)), d.current.setFullYear(parseInt(g.find("thead th.datepickerMonth span").text(), 10)), d.current.addMonths(Math.floor(d.calendars / 2) - h), g.get(0).className = "datepickerViewDays";
                        break;
                    case"datepickerViewYears":
                        d.current.setFullYear(parseInt(c.text(), 10)), g.get(0).className = "datepickerViewMonths";
                        break;
                    default:
                        var q = parseInt(c.text(), 10);
                        switch (m.addMonths(h - Math.floor(d.calendars / 2)), f.hasClass("datepickerNotInMonth") && m.addMonths(q > 15?-1 : 1), m.setDate(q), d.mode) {
                        case"multiple":
                            q = m.setHours(0, 0, 0, 0).valueOf(), a.inArray(q, d.date)>-1 ? a.each(d.date, function(a, b) {
                                return b == q ? (d.date.splice(a, 1), !1) : void 0
                            }) : d.date.push(q);
                            break;
                        case"range":
                            d.lastSel || (d.date[0] = m.setHours(0, 0, 0, 0).valueOf()), q = m.setHours(23, 59, 59, 0).valueOf(), q < d.date[0] ? (d.date[1] = d.date[0] + 86399e3, d.date[0] = q - 86399e3) : d.date[1] = q, d.lastSel=!d.lastSel;
                            break;
                        default:
                            d.date = m.valueOf()
                        }
                    }
                    p=!0, o=!0
                }
                p && e(this), o && d.onChange.apply(this, n(d))
            }
            return !1
        }, n = function(b) {
            var c;
            return "single" == b.mode ? (c = new Date(b.date), [g(c, b.format), c, b.el]) : (c = [[], [], b.el], a.each(b.date, function(a, d) {
                var e = new Date(d);
                c[0].push(g(e, b.format)), c[1].push(e)
            }), c)
        }, o = function() {
            var a = "CSS1Compat" == document.compatMode;
            return {
                l: window.pageXOffset || (a ? document.documentElement.scrollLeft : document.body.scrollLeft),
                t: window.pageYOffset || (a ? document.documentElement.scrollTop : document.body.scrollTop),
                w: window.innerWidth || (a ? document.documentElement.clientWidth : document.body.clientWidth),
                h: window.innerHeight || (a ? document.documentElement.clientHeight : document.body.clientHeight)
            }
        }, p = function(a, b, c) {
            if (a == b)
                return !0;
            if (a.contains)
                return a.contains(b);
            if (a.compareDocumentPosition)
                return !!(16 & a.compareDocumentPosition(b));
            for (var d = b.parentNode; d && d != c;) {
                if (d == a)
                    return !0;
                d = d.parentNode
            }
            return !1
        }, q = function() {
            var b = a("#" + a(this).data("datepickerId"));
            if (!b.is(":visible")) {
                var c = b.get(0);
                e(c);
                var d = b.data("datepicker");
                d.onBeforeShow.apply(this, [b.get(0)]);
                {
                    var f = a(this).offset(), g = o(), h = f.top, i = f.left;
                    a.curCSS(c, "display")
                }
                switch (b.css({
                    visibility: "hidden",
                    display: "block"
                }), d.position) {
                case"top":
                    h -= c.offsetHeight;
                    break;
                case"left":
                    i -= c.offsetWidth;
                    break;
                case"right":
                    i += this.offsetWidth;
                    break;
                case"bottom":
                    h += this.offsetHeight
                }
                h + c.offsetHeight > g.t + g.h && (h = f.top - c.offsetHeight), h < g.t && (h = f.top + this.offsetHeight + c.offsetHeight), i + c.offsetWidth > g.l + g.w && (i = f.left - c.offsetWidth), i < g.l && (i = f.left + this.offsetWidth), b.css({
                    visibility: "visible",
                    display: "block",
                    top: h + "px",
                    left: i + "px"
                }), 0 != d.onShow.apply(this, [b.get(0)]) && b.show(), a(document).bind("mousedown", {
                    cal: b,
                    trigger: this
                }, r)
            }
            return !1
        }, r = function(b) {
            b.target == b.data.trigger || p(b.data.cal.get(0), b.target, b.data.cal.get(0)) || (0 != b.data.cal.data("datepicker").onHide.apply(this, [b.data.cal.get(0)]) && b.data.cal.hide(), a(document).unbind("mousedown", r))
        };
        return {
            init: function(g) {
                return g = a.extend({}, d, g || {}), h(g.locale), g.calendars = Math.max(1, parseInt(g.calendars, 10) || 1), g.mode = /single|multiple|range/.test(g.mode) ? g.mode : "single", this.each(function() {
                    if (!a(this).data("datepicker")) {
                        if (g.el = this, g.date.constructor == String && (g.date = f(g.date, g.format), g.date.setHours(0, 0, 0, 0)), "single" != g.mode)
                            if (g.date.constructor != Array)
                                g.date = [g.date.valueOf()], "range" == g.mode && g.date.push(new Date(g.date[0]).setHours(23, 59, 59, 0).valueOf());
                            else {
                                for (var d = 0; d < g.date.length; d++)
                                    g.date[d] = f(g.date[d], g.format).setHours(0, 0, 0, 0).valueOf();
                                    "range" == g.mode && (g.date[1] = new Date(g.date[1]).setHours(23, 59, 59, 0).valueOf())
                                } else 
                                    g.date = g.date.valueOf();
                        g.current = g.current ? f(g.current, g.format) : new Date, g.current.setDate(1), g.current.setHours(0, 0, 0, 0);
                        var h, i = "datepicker_" + parseInt(1e3 * Math.random());
                        g.id = i, a(this).data("datepickerId", g.id);
                        var j = a(c.wrapper).attr("id", i).bind("click", m).data("datepicker", g);
                        g.className && j.addClass(g.className);
                        for (var k = "", d = 0; d < g.calendars; d++)
                            h = g.starts, d > 0 && (k += c.space), k += tmpl(c.head.join(""), {
                                week: g.locale.weekMin,
                                prev: g.prev,
                                next: g.next,
                                day1: g.locale.daysMin[h++%7],
                                day2: g.locale.daysMin[h++%7],
                                day3: g.locale.daysMin[h++%7],
                                day4: g.locale.daysMin[h++%7],
                                day5: g.locale.daysMin[h++%7],
                                day6: g.locale.daysMin[h++%7],
                                day7: g.locale.daysMin[h++%7]
                            });
                        j.find("tr:first").append(k).find("table").addClass(b[g.view]), e(j.get(0)), g.flat ? j.appendTo(this).show() : (j.appendTo(document.body), a(this).bind(g.eventName, q))
                    }
                })
            },
            showPicker: function() {
                return this.each(function() {
                    a(this).data("datepickerId") && q.apply(this)
                })
            },
            hidePicker: function() {
                return this.each(function() {
                    a(this).data("datepickerId") && a("#" + a(this).data("datepickerId")).hide()
                })
            },
            setDate: function(b, c) {
                return this.each(function() {
                    if (a(this).data("datepickerId")) {
                        var d = a("#" + a(this).data("datepickerId")), g = d.data("datepicker");
                        if (g.date = b, g.date.constructor == String && (g.date = f(g.date, g.format), g.date.setHours(0, 0, 0, 0)), "single" != g.mode)
                            if (g.date.constructor != Array)
                                g.date = [g.date.valueOf()], "range" == g.mode && g.date.push(new Date(g.date[0]).setHours(23, 59, 59, 0).valueOf());
                            else {
                                for (var h = 0; h < g.date.length; h++)
                                    g.date[h] = f(g.date[h], g.format).setHours(0, 0, 0, 0).valueOf();
                                    "range" == g.mode && (g.date[1] = new Date(g.date[1]).setHours(23, 59, 59, 0).valueOf())
                                } else 
                                    g.date = g.date.valueOf();
                        c && (g.current = new Date("single" != g.mode ? g.date[0] : g.date)), e(d.get(0))
                    }
                })
            },
            getDate: function(b) {
                return this.size() > 0 ? n(a("#" + a(this).data("datepickerId")).data("datepicker"))[b ? 0: 1] : void 0
            },
            clear: function() {
                return this.each(function() {
                    if (a(this).data("datepickerId")) {
                        var b = a("#" + a(this).data("datepickerId")), c = b.data("datepicker");
                        c.date = "single" != c.mode ? [] : null, e(b.get(0))
                    }
                })
            }
        }
    }();
    a.fn.extend({
        DatePicker: b.init,
        DatePickerHide: b.hidePicker,
        DatePickerShow: b.showPicker,
        DatePickerSetDate: b.setDate,
        DatePickerGetDate: b.getDate,
        DatePickerClear: b.clear
    })
}(jQuery), function() {
    var a = {};
    this.tmpl = function b(c, d) {
        var e = /\W/.test(c) ? new Function("obj", "var p=[],print=function(){p.push.apply(p,arguments);};with(obj){p.push('" + c.replace(/[\r\t\n]/g, " ").split("<%").join("	").replace(/((^|%>)[^\t]*)'/g, "$1\r").replace(/\t=(.*?)%>/g, "',$1,'").split("	").join("');").split("%>").join("p.push('").split("\r").join("\\'") + "');}return p.join('');"): a[c] = a[c] || b(document.getElementById(c).innerHTML);
        return d ? e(d) : e
    }
}(), function(a) {
    var b = window.EYE = function() {
        var b = {
            init: []
        };
        return {
            init: function() {
                a.each(b.init, function(a, b) {
                    b.call()
                })
            },
            extend: function(a) {
                for (var b in a)
                    void 0 != a[b] && (this[b] = a[b])
            },
            register: function(a, c) {
                b[c] || (b[c] = []), b[c].push(a)
            }
        }
    }();
    a(b.init)
}(jQuery), function(a) {
    EYE.extend({
        getPosition: function(a, b) {
            var c = 0, d = 0, e = a.style, f=!1;
            if (b && "none" == jQuery.curCSS(a, "display")) {
                var g = e.visibility, h = e.position;
                f=!0, e.visibility = "hidden", e.display = "block", e.position = "absolute"
            }
            var i = a;
            if (i.getBoundingClientRect) {
                var j = i.getBoundingClientRect();
                c = j.left + Math.max(document.documentElement.scrollLeft, document.body.scrollLeft) - 2, d = j.top + Math.max(document.documentElement.scrollTop, document.body.scrollTop) - 2
            } else {
                if (c = i.offsetLeft, d = i.offsetTop, i = i.offsetParent, a != i)
                    for (; i;)
                        c += i.offsetLeft, d += i.offsetTop, i = i.offsetParent;
                for (i = a.parentNode; i && "BODY" != i.tagName.toUpperCase() && "HTML" != i.tagName.toUpperCase();)
                    "inline" != jQuery.curCSS(i, "display") && (c -= i.scrollLeft, d -= i.scrollTop), i = i.parentNode
            }
            return 1 == f && (e.display = "none", e.position = h, e.visibility = g), {
                x: c,
                y: d
            }
        },
        getSize: function(a) {
            var b = parseInt(jQuery.curCSS(a, "width"), 10), c = parseInt(jQuery.curCSS(a, "height"), 10), d = 0, e = 0;
            if ("none" != jQuery.curCSS(a, "display"))
                d = a.offsetWidth, e = a.offsetHeight;
            else {
                var f = a.style, g = f.visibility, h = f.position;
                f.visibility = "hidden", f.display = "block", f.position = "absolute", d = a.offsetWidth, e = a.offsetHeight, f.display = "none", f.position = h, f.visibility = g
            }
            return {
                w: b,
                h: c,
                wb: d,
                hb: e
            }
        },
        getClient: function(a) {
            var b, c;
            if (a)
                c = a.clientWidth, b = a.clientHeight;
            else {
                var d = document.documentElement;
                c = window.innerWidth || self.innerWidth || d && d.clientWidth || document.body.clientWidth, b = window.innerHeight || self.innerHeight || d && d.clientHeight || document.body.clientHeight
            }
            return {
                w: c,
                h: b
            }
        },
        getScroll: function(a) {
            var b = 0, c = 0, d = 0, e = 0, f = 0, g = 0;
            return a && "body" != a.nodeName.toLowerCase() ? (b = a.scrollTop, c = a.scrollLeft, d = a.scrollWidth, e = a.scrollHeight) : (document.documentElement ? (b = document.documentElement.scrollTop, c = document.documentElement.scrollLeft, d = document.documentElement.scrollWidth, e = document.documentElement.scrollHeight) : document.body && (b = document.body.scrollTop, c = document.body.scrollLeft, d = document.body.scrollWidth, e = document.body.scrollHeight), "undefined" != typeof pageYOffset && (b = pageYOffset, c = pageXOffset), f = self.innerWidth || document.documentElement.clientWidth || document.body.clientWidth || 0, g = self.innerHeight || document.documentElement.clientHeight || document.body.clientHeight || 0), {
                t: b,
                l: c,
                w: d,
                h: e,
                iw: f,
                ih: g
            }
        },
        getMargins: function(a, b) {
            var c = jQuery.curCSS(a, "marginTop") || "", d = jQuery.curCSS(a, "marginRight") || "", e = jQuery.curCSS(a, "marginBottom") || "", f = jQuery.curCSS(a, "marginLeft") || "";
            return b ? {
                t: parseInt(c, 10) || 0,
                r: parseInt(d, 10) || 0,
                b: parseInt(e, 10) || 0,
                l: parseInt(f, 10)
            } : {
                t: c,
                r: d,
                b: e,
                l: f
            }
        },
        getPadding: function(a, b) {
            var c = jQuery.curCSS(a, "paddingTop") || "", d = jQuery.curCSS(a, "paddingRight") || "", e = jQuery.curCSS(a, "paddingBottom") || "", f = jQuery.curCSS(a, "paddingLeft") || "";
            return b ? {
                t: parseInt(c, 10) || 0,
                r: parseInt(d, 10) || 0,
                b: parseInt(e, 10) || 0,
                l: parseInt(f, 10)
            } : {
                t: c,
                r: d,
                b: e,
                l: f
            }
        },
        getBorder: function(a, b) {
            var c = jQuery.curCSS(a, "borderTopWidth") || "", d = jQuery.curCSS(a, "borderRightWidth") || "", e = jQuery.curCSS(a, "borderBottomWidth") || "", f = jQuery.curCSS(a, "borderLeftWidth") || "";
            return b ? {
                t: parseInt(c, 10) || 0,
                r: parseInt(d, 10) || 0,
                b: parseInt(e, 10) || 0,
                l: parseInt(f, 10) || 0
            } : {
                t: c,
                r: d,
                b: e,
                l: f
            }
        },
        traverseDOM: function(a, b) {
            for (b(a)
                , a = a.firstChild;
            a;
            )EYE.traverseDOM(a, b), a = a.nextSibling
        },
        getInnerWidth: function(a, b) {
            var c = a.offsetWidth;
            return b ? Math.max(a.scrollWidth, c) - c + a.clientWidth : a.clientWidth
        },
        getInnerHeight: function(a, b) {
            var c = a.offsetHeight;
            return b ? Math.max(a.scrollHeight, c) - c + a.clientHeight : a.clientHeight
        },
        getExtraWidth: function(b) {
            return a.boxModel ? (parseInt(a.curCSS(b, "paddingLeft")) || 0) + (parseInt(a.curCSS(b, "paddingRight")) || 0) + (parseInt(a.curCSS(b, "borderLeftWidth")) || 0) + (parseInt(a.curCSS(b, "borderRightWidth")) || 0) : 0
        },
        getExtraHeight: function(b) {
            return a.boxModel ? (parseInt(a.curCSS(b, "paddingTop")) || 0) + (parseInt(a.curCSS(b, "paddingBottom")) || 0) + (parseInt(a.curCSS(b, "borderTopWidth")) || 0) + (parseInt(a.curCSS(b, "borderBottomWidth")) || 0) : 0
        },
        isChildOf: function(b, c, d) {
            if (b == c)
                return !0;
            if (!c ||!c.nodeType || 1 != c.nodeType)
                return !1;
            if (b.contains&&!a.browser.safari)
                return b.contains(c);
            if (b.compareDocumentPosition)
                return !!(16 & b.compareDocumentPosition(c));
            for (var e = c.parentNode; e && e != d;) {
                if (e == b)
                    return !0;
                e = e.parentNode
            }
            return !1
        },
        centerEl: function(b, c) {
            var d = EYE.getScroll(), e = EYE.getSize(b);
            c && "vertically" != c || a(b).css({
                top: d.t + (Math.min(d.h, d.ih) - e.hb) / 2 + "px"
            }), c && "horizontally" != c || a(b).css({
                left: d.l + (Math.min(d.w, d.iw) - e.wb) / 2 + "px"
            })
        }
    }), a.easing.easeout || (a.easing.easeout = function(a, b, c, d, e) {
        return - d * ((b = b / e - 1) * b * b * b - 1) + c
    })
}(jQuery), function(a) {
    var b = function() {
        var b = window.location.hash.replace("#", ""), d = a("ul.navigationTabs a").bind("click", c).filter('a[rel="' + b + '"]');
        0 == d.size() && (d = a("ul.navigationTabs a:first")), c.apply(d.get(0)), a("#date").DatePicker({
            flat: !0,
            date: "2008-07-31",
            current: "2008-07-31",
            calendars: 1,
            starts: 1,
            view: "years"
        });
        var e = new Date;
        e.addDays( - 10);
        var f = new Date;
        f.addDays( - 5), f.setHours(0, 0, 0, 0), a("#date2").DatePicker({
            flat: !0,
            date: ["2008-07-31", "2008-07-28"],
            current: "2008-07-31",
            format: "Y-m-d",
            calendars: 1,
            mode: "multiple",
            onRender: function(a) {
                return {
                    disabled: a.valueOf() < e.valueOf(),
                    className: a.valueOf() == f.valueOf() ? "datepickerSpecial": !1
                }
            },
            onChange: function() {},
            starts: 0
        }), a("#clearSelection").bind("click", function() {
            return a("#date3").DatePickerClear(), !1
        }), a("#date3").DatePicker({
            flat: !0,
            date: ["2009-12-28", "2010-01-23"],
            current: "2010-01-01",
            calendars: 3,
            mode: "range",
            starts: 1
        }), a(".inputDate").DatePicker({
            format: "m/d/Y",
            date: a("#inputDate").val(),
            current: a("#inputDate").val(),
            starts: 1,
            position: "right",
            onBeforeShow: function() {
                a("#inputDate").DatePickerSetDate(a("#inputDate").val(), !0)
            },
            onChange: function(b) {
                a("#inputDate").val(b), a("#closeOnSelect input").is(":checked") && a("#inputDate").DatePickerHide()
            }
        });
        var g = new Date;
        g.addDays( - 4);
        var h = new Date;
        a("#widgetCalendar").DatePicker({
            flat: !0,
            format: "d B, Y",
            date: [new Date(g), new Date(h)],
            calendars: 3,
            mode: "range",
            starts: 1,
            onChange: function(b) {
                a("#widgetField span").get(0).innerHTML = b.join(" &divide; ")
            }
        });
        var i=!1;
        a("#widgetField>a").bind("click", function() {
            return a("#widgetCalendar").stop().animate({
                height: i ? 0: a("#widgetCalendar div.datepicker").get(0).offsetHeight
            }, 500), i=!i, !1
        }), a("#widgetCalendar div.datepicker").css("position", "absolute")
    }, c = function() {
        var b = a("ul.navigationTabs a").removeClass("active").index(this);
        a(this).addClass("active").blur(), a("div.tab").hide().eq(b).show()
    };
    EYE.register(b, "init")
}(jQuery);
