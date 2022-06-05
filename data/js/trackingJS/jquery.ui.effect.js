(function(c, w) {
    c.effects = {
        effect: {}
    };
    (function(c, h) {
        function b(f, a, c) {
            var b = s[a.type] || {};
            if (null == f)
                return c ||!a.def ? null : a.def;
            f = b.floor?~~f : parseFloat(f);
            return isNaN(f) ? a.def : b.mod ? (f + b.mod)%b.mod : 0 > f ? 0 : b.max < f ? b.max : f
        }
        function a(f) {
            var a = g(), b = a._rgba = [];
            f = f.toLowerCase();
            p(l, function(c, d) {
                var e, g = d.re.exec(f);
                e = g && d.parse(g);
                g = d.space || "rgba";
                if (e)
                    return e = a[g](e), a[m[g].cache] = e[m[g].cache], b = a._rgba = e._rgba, !1
            });
            return b.length ? ("0,0,0,0" === b.join() && c.extend(b, u.transparent), a) : u[f]
        }
        function d(f,
        a, b) {
            b = (b + 1)%1;
            return 1 > 6 * b ? f + (a - f) * b * 6 : 1 > 2 * b ? a : 2 > 3 * b ? f + (a - f) * (2 / 3 - b) * 6 : f
        }
        var e = /^([\-+])=\s*(\d+\.?\d*)/, l = [{
            re: /rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*(?:,\s*(\d?(?:\.\d+)?)\s*)?\)/,
            parse: function(f) {
                return [f[1], f[2], f[3], f[4]]
            }
        }, {
            re: /rgba?\(\s*(\d+(?:\.\d+)?)\%\s*,\s*(\d+(?:\.\d+)?)\%\s*,\s*(\d+(?:\.\d+)?)\%\s*(?:,\s*(\d?(?:\.\d+)?)\s*)?\)/,
            parse: function(f) {
                return [2.55 * f[1], 2.55 * f[2], 2.55 * f[3], f[4]]
            }
        }, {
            re: /#([a-f0-9]{2})([a-f0-9]{2})([a-f0-9]{2})/,
            parse: function(f) {
                return [parseInt(f[1],
                16), parseInt(f[2], 16), parseInt(f[3], 16)]
            }
        }, {
            re: /#([a-f0-9])([a-f0-9])([a-f0-9])/,
            parse: function(f) {
                return [parseInt(f[1] + f[1], 16), parseInt(f[2] + f[2], 16), parseInt(f[3] + f[3], 16)]
            }
        }, {
            re: /hsla?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\%\s*,\s*(\d+(?:\.\d+)?)\%\s*(?:,\s*(\d?(?:\.\d+)?)\s*)?\)/,
            space: "hsla",
            parse: function(f) {
                return [f[1], f[2] / 100, f[3] / 100, f[4]]
            }
        }
        ], g = c.Color = function(f, a, b, d) {
            return new c.Color.fn.parse(f, a, b, d)
        }, m = {
            rgba: {
                props: {
                    red: {
                        idx: 0,
                        type: "byte"
                    },
                    green: {
                        idx: 1,
                        type: "byte"
                    },
                    blue: {
                        idx: 2,
                        type: "byte"
                    }
                }
            },
            hsla: {
                props: {
                    hue: {
                        idx: 0,
                        type: "degrees"
                    },
                    saturation: {
                        idx: 1,
                        type: "percent"
                    },
                    lightness: {
                        idx: 2,
                        type: "percent"
                    }
                }
            }
        }, s = {
            "byte": {
                floor: !0,
                max: 255
            },
            percent: {
                max: 1
            },
            degrees: {
                mod: 360,
                floor: !0
            }
        }, x = g.support = {}, r = c("\x3cp\x3e")[0], u, p = c.each;
        r.style.cssText = "background-color:rgba(1,1,1,.5)";
        x.rgba =- 1 < r.style.backgroundColor.indexOf("rgba");
        p(m, function(f, a) {
            a.cache = "_" + f;
            a.props.alpha = {
                idx: 3,
                type: "percent",
                def: 1
            }
        });
        g.fn = c.extend(g.prototype, {
            parse: function(f, d, e, n) {
                if (f === h)
                    return this._rgba = [null,
                    null, null, null], this;
                if (f.jquery || f.nodeType)
                    f = c(f).css(d), d = h;
                var k = this, t = c.type(f), l = this._rgba = [];
                d !== h && (f = [f, d, e, n], t = "array");
                if ("string" === t)
                    return this.parse(a(f) || u._default);
                if ("array" === t)
                    return p(m.rgba.props, function(a, c) {
                        l[c.idx] = b(f[c.idx], c)
                    }), this;
                if ("object" === t)
                    return f instanceof g ? p(m, function(a, b) {
                        f[b.cache] && (k[b.cache] = f[b.cache].slice())
                    }) : p(m, function(a, d) {
                        var e = d.cache;
                        p(d.props, function(a, c) {
                            if (!k[e] && d.to) {
                                if ("alpha" === a || null == f[a])
                                    return;
                                    k[e] = d.to(k._rgba)
                            }
                            k[e][c.idx] =
                            b(f[a], c, !0)
                        });
                        k[e] && 0 > c.inArray(null, k[e].slice(0, 3)) && (k[e][3] = 1, d.from && (k._rgba = d.from(k[e])))
                    }), this
            },
            is: function(a) {
                var b = g(a), c=!0, d = this;
                p(m, function(a, f) {
                    var e, g = b[f.cache];
                    g && (e = d[f.cache] || f.to && f.to(d._rgba) || [], p(f.props, function(a, f) {
                        if (null != g[f.idx])
                            return c = g[f.idx] === e[f.idx]
                    }));
                    return c
                });
                return c
            },
            _space: function() {
                var a = [], b = this;
                p(m, function(c, d) {
                    b[d.cache] && a.push(c)
                });
                return a.pop()
            },
            transition: function(a, c) {
                var d = g(a), e = d._space(), k = m[e], h = 0 === this.alpha() ? g("transparent"):
                this, l = h[k.cache] || k.to(h._rgba), v = l.slice(), d = d[k.cache];
                p(k.props, function(a, f) {
                    var e = f.idx, n = l[e], g = d[e], k = s[f.type] || {};
                    null !== g && (null === n ? v[e] = g : (k.mod && (g - n > k.mod / 2 ? n += k.mod : n - g > k.mod / 2 && (n -= k.mod)), v[e] = b((g - n) * c + n, f)))
                });
                return this[e](v)
            },
            blend: function(a) {
                if (1 === this._rgba[3])
                    return this;
                var b = this._rgba.slice(), d = b.pop(), e = g(a)._rgba;
                return g(c.map(b, function(a, f) {
                    return (1 - d) * e[f] + d * a
                }))
            },
            toRgbaString: function() {
                var a = "rgba(", b = c.map(this._rgba, function(a, b) {
                    return null == a ? 2 < b ? 1 : 0 : a
                });
                1 === b[3] && (b.pop(), a = "rgb(");
                return a + b.join() + ")"
            },
            toHslaString: function() {
                var a = "hsla(", b = c.map(this.hsla(), function(a, b) {
                    null == a && (a = 2 < b ? 1 : 0);
                    b && 3 > b && (a = Math.round(100 * a) + "%");
                    return a
                });
                1 === b[3] && (b.pop(), a = "hsl(");
                return a + b.join() + ")"
            },
            toHexString: function(a) {
                var b = this._rgba.slice(), d = b.pop();
                a && b.push(~~(255 * d));
                return "#" + c.map(b, function(a) {
                    a = (a || 0).toString(16);
                    return 1 === a.length ? "0" + a : a
                }).join("")
            },
            toString: function() {
                return 0 === this._rgba[3] ? "transparent" : this.toRgbaString()
            }
        });
        g.fn.parse.prototype =
        g.fn;
        m.hsla.to = function(a) {
            if (null == a[0] || null == a[1] || null == a[2])
                return [null, null, null, a[3]];
            var b = a[0] / 255, d = a[1] / 255, c = a[2] / 255;
            a = a[3];
            var e = Math.max(b, d, c), g = Math.min(b, d, c), h = e - g, l = e + g, m = 0.5 * l, l = 0 === h ? 0: 0.5 >= m ? h / l: h / (2 - l);
            return [Math.round(g === e ? 0 : b === e ? 60 * (d - c) / h + 360 : d === e ? 60 * (c - b) / h + 120 : 60 * (b - d) / h + 240)%360, l, m, null == a ? 1: a]
        };
        m.hsla.from = function(a) {
            if (null == a[0] || null == a[1] || null == a[2])
                return [null, null, null, a[3]];
            var b = a[0] / 360, c = a[1], e = a[2];
            a = a[3];
            c = 0.5 >= e ? e * (1 + c) : e + c - e * c;
            e = 2 * e - c;
            return [Math.round(255 *
            d(e, c, b + 1 / 3)), Math.round(255 * d(e, c, b)), Math.round(255 * d(e, c, b - 1 / 3)), a]
        };
        p(m, function(a, d) {
            var l = d.props, n = d.cache, k = d.to, m = d.from;
            g.fn[a] = function(a) {
                k&&!this[n] && (this[n] = k(this._rgba));
                if (a === h)
                    return this[n].slice();
                var d, e = c.type(a), f = "array" === e || "object" === e ? a: arguments, s = this[n].slice();
                p(l, function(a, d) {
                    var c = f["object" === e ? a: d.idx];
                    null == c && (c = s[d.idx]);
                    s[d.idx] = b(c, d)
                });
                return m ? (d = g(m(s)), d[n] = s, d) : g(s)
            };
            p(l, function(b, d) {
                g.fn[b] || (g.fn[b] = function(g) {
                    var h = c.type(g), l = "alpha" === b ? this._hsla ?
                    "hsla": "rgba": a, k = this[l](), m = k[d.idx];
                    if ("undefined" === h)
                        return m;
                    "function" === h && (g = g.call(this, m), h = c.type(g));
                    if (null == g && d.empty)
                        return this;
                    "string" === h && (h = e.exec(g)) && (g = m + parseFloat(h[2]) * ("+" === h[1] ? 1 : - 1));
                    k[d.idx] = g;
                    return this[l](k)
                })
            })
        });
        g.hook = function(b) {
            b = b.split(" ");
            p(b, function(b, d) {
                c.cssHooks[d] = {
                    set: function(b, e) {
                        var f, h = "";
                        if ("transparent" !== e && ("string" !== c.type(e) || (f = a(e)))) {
                            e = g(f || e);
                            if (!x.rgba && 1 !== e._rgba[3]) {
                                for (f = "backgroundColor" === d ? b.parentNode : b; ("" === h || "transparent" ===
                                h) && f && f.style;)
                                    try {
                                        h = c.css(f, "backgroundColor"), f = f.parentNode
                                } catch (l) {}
                                e = e.blend(h && "transparent" !== h ? h : "_default")
                            }
                            e = e.toRgbaString()
                        }
                        try {
                            b.style[d] = e
                        } catch (m) {}
                    }
                };
                c.fx.step[d] = function(a) {
                    a.colorInit || (a.start = g(a.elem, d), a.end = g(a.end), a.colorInit=!0);
                    c.cssHooks[d].set(a.elem, a.start.transition(a.end, a.pos))
                }
            })
        };
        g.hook("backgroundColor borderBottomColor borderLeftColor borderRightColor borderTopColor color columnRuleColor outlineColor textDecorationColor textEmphasisColor");
        c.cssHooks.borderColor =
        {
            expand: function(a) {
                var b = {};
                p(["Top", "Right", "Bottom", "Left"], function(d, e) {
                    b["border" + e + "Color"] = a
                });
                return b
            }
        };
        u = c.Color.names = {
            aqua: "#00ffff",
            black: "#000000",
            blue: "#0000ff",
            fuchsia: "#ff00ff",
            gray: "#808080",
            green: "#008000",
            lime: "#00ff00",
            maroon: "#800000",
            navy: "#000080",
            olive: "#808000",
            purple: "#800080",
            red: "#ff0000",
            silver: "#c0c0c0",
            teal: "#008080",
            white: "#ffffff",
            yellow: "#ffff00",
            transparent: [null, null, null, 0],
            _default: "#ffffff"
        }
    })(jimdoGen002);
    (function() {
        function q(a) {
            var b, e = a.ownerDocument.defaultView ?
            a.ownerDocument.defaultView.getComputedStyle(a, null): a.currentStyle, h = {};
            if (e && e.length && e[0] && e[e[0]])
                for (a = e.length; a--;)
                    b = e[a], "string" === typeof e[b] && (h[c.camelCase(b)] = e[b]);
            else 
                for (b in e)
                    "string" === typeof e[b] && (h[b] = e[b]);
            return h
        }
        var h = ["add", "remove", "toggle"], b = {
            border: 1,
            borderBottom: 1,
            borderColor: 1,
            borderLeft: 1,
            borderRight: 1,
            borderTop: 1,
            borderWidth: 1,
            margin: 1,
            padding: 1
        };
        c.each(["borderLeftStyle", "borderRightStyle", "borderBottomStyle", "borderTopStyle"], function(a, b) {
            c.fx.step[b] = function(a) {
                if ("none" !==
                a.end&&!a.setAttr || 1 === a.pos&&!a.setAttr)
                    jimdoGen002.style(a.elem, b, a.end), a.setAttr=!0
            }
        });
        c.fn.addBack || (c.fn.addBack = function(a) {
            return this.add(null == a ? this.prevObject : this.prevObject.filter(a))
        });
        c.effects.animateClass = function(a, d, e, l) {
            var g = c.speed(d, e, l);
            return this.queue(function() {
                var d = c(this), e = d.attr("class") || "", l, r = g.children ? d.find("*").addBack(): d, r = r.map(function() {
                    return {
                        el: c(this),
                        start: q(this)
                    }
                });
                l = function() {
                    c.each(h, function(b, c) {
                        if (a[c])
                            d[c + "Class"](a[c])
                    })
                };
                l();
                r = r.map(function() {
                    this.end =
                    q(this.el[0]);
                    var a = this.start, d = this.end, e = {}, g, h;
                    for (g in d)
                        h = d[g], a[g] === h || b[g] ||!c.fx.step[g] && isNaN(parseFloat(h)) || (e[g] = h);
                    this.diff = e;
                    return this
                });
                d.attr("class", e);
                r = r.map(function() {
                    var a = this, b = c.Deferred(), d = c.extend({}, g, {
                        queue: !1,
                        complete: function() {
                            b.resolve(a)
                        }
                    });
                    this.el.animate(this.diff, d);
                    return b.promise()
                });
                c.when.apply(c, r.get()).done(function() {
                    l();
                    c.each(arguments, function() {
                        var a = this.el;
                        c.each(this.diff, function(b) {
                            a.css(b, "")
                        })
                    });
                    g.complete.call(d[0])
                })
            })
        };
        c.fn.extend({
            addClass: function(a) {
                return function(b,
                e, h, g) {
                    return e ? c.effects.animateClass.call(this, {
                        add: b
                    }, e, h, g) : a.apply(this, arguments)
                }
            }(c.fn.addClass),
            removeClass: function(a) {
                return function(b, e, h, g) {
                    return 1 < arguments.length ? c.effects.animateClass.call(this, {
                        remove: b
                    }, e, h, g) : a.apply(this, arguments)
                }
            }(c.fn.removeClass),
            toggleClass: function(a) {
                return function(b, e, h, g, m) {
                    return "boolean" === typeof e || e === w ? h ? c.effects.animateClass.call(this, e ? {
                        add: b
                    } : {
                        remove: b
                    }, h, g, m) : a.apply(this, arguments) : c.effects.animateClass.call(this, {
                        toggle: b
                    }, e, h, g)
                }
            }(c.fn.toggleClass),
            switchClass: function(a, b, e, h, g) {
                return c.effects.animateClass.call(this, {
                    add: b,
                    remove: a
                }, e, h, g)
            }
        })
    })();
    (function() {
        function q(b, a, d, e) {
            c.isPlainObject(b) && (a = b, b = b.effect);
            b = {
                effect: b
            };
            null == a && (a = {});
            c.isFunction(a) && (e = a, d = null, a = {});
            if ("number" === typeof a || c.fx.speeds[a])
                e = d, d = a, a = {};
            c.isFunction(d) && (e = d, d = null);
            a && c.extend(b, a);
            d = d || a.duration;
            b.duration = c.fx.off ? 0 : "number" === typeof d ? d : d in c.fx.speeds ? c.fx.speeds[d] : c.fx.speeds._default;
            b.complete = e || a.complete;
            return b
        }
        function h(b) {
            return !b ||
            "number" === typeof b || c.fx.speeds[b] || "string" === typeof b&&!c.effects.effect[b] || c.isFunction(b) || "object" === typeof b&&!b.effect?!0 : !1
        }
        c.extend(c.effects, {
            version: "1.10.3",
            save: function(b, a) {
                for (var d = 0; d < a.length; d++)
                    null !== a[d] && b.data("ui-effects-" + a[d], b[0].style[a[d]])
            },
            restore: function(b, a) {
                var d, c;
                for (c = 0; c < a.length; c++)
                    null !== a[c] && (d = b.data("ui-effects-" + a[c]), d === w && (d = ""), b.css(a[c], d))
            },
            setMode: function(b, a) {
                "toggle" === a && (a = b.is(":hidden") ? "show" : "hide");
                return a
            },
            getBaseline: function(b,
            a) {
                var c, e;
                switch (b[0]) {
                case "top":
                    c = 0;
                    break;
                case "middle":
                    c = 0.5;
                    break;
                case "bottom":
                    c = 1;
                    break;
                default:
                    c = b[0] / a.height
                }
                switch (b[1]) {
                case "left":
                    e = 0;
                    break;
                case "center":
                    e = 0.5;
                    break;
                case "right":
                    e = 1;
                    break;
                default:
                    e = b[1] / a.width
                }
                return {
                    x: e,
                    y: c
                }
            },
            createWrapper: function(b) {
                if (b.parent().is(".ui-effects-wrapper"))
                    return b.parent();
                var a = {
                    width: b.outerWidth(!0),
                    height: b.outerHeight(!0),
                    "float": b.css("float")
                }, d = c("\x3cdiv\x3e\x3c/div\x3e").addClass("ui-effects-wrapper").css({
                    fontSize: "100%",
                    background: "transparent",
                    border: "none",
                    margin: 0,
                    padding: 0
                }), e = {
                    width: b.width(),
                    height: b.height()
                }, h = document.activeElement;
                try {
                    h.id
                } catch (g) {
                    h = document.body
                }
                b.wrap(d);
                (b[0] === h || c.contains(b[0], h)) && c(h).focus();
                d = b.parent();
                "static" === b.css("position") ? (d.css({
                    position: "relative"
                }), b.css({
                    position: "relative"
                })) : (c.extend(a, {
                    position: b.css("position"),
                    zIndex: b.css("z-index")
                }), c.each(["top", "left", "bottom", "right"], function(c, d) {
                    a[d] = b.css(d);
                    isNaN(parseInt(a[d], 10)) && (a[d] = "auto")
                }), b.css({
                    position: "relative",
                    top: 0,
                    left: 0,
                    right: "auto",
                    bottom: "auto"
                }));
                b.css(e);
                return d.css(a).show()
            },
            removeWrapper: function(b) {
                var a = document.activeElement;
                b.parent().is(".ui-effects-wrapper") && (b.parent().replaceWith(b), (b[0] === a || c.contains(b[0], a)) && c(a).focus());
                return b
            },
            setTransition: function(b, a, d, e) {
                e = e || {};
                c.each(a, function(a, c) {
                    var h = b.cssUnit(c);
                    0 < h[0] && (e[c] = h[0] * d + h[1])
                });
                return e
            }
        });
        c.fn.extend({
            effect: function() {
                function b(b) {
                    function d() {
                        c.isFunction(q) && q.call(e[0]);
                        c.isFunction(b) && b()
                    }
                    var e = c(this), q = a.complete, r = a.mode;
                    (e.is(":hidden") ? "hide" === r : "show" === r) ? (e[r](), d()) : h.call(e[0], a, d)
                }
                var a = q.apply(this, arguments), d = a.mode, e = a.queue, h = c.effects.effect[a.effect];
                return c.fx.off ||!h ? d ? this[d](a.duration, a.complete) : this.each(function() {
                    a.complete && a.complete.call(this)
                }) : !1 === e ? this.each(b) : this.queue(e || "fx", b)
            },
            show: function(b) {
                return function(a) {
                    if (h(a))
                        return b.apply(this, arguments);
                    var c = q.apply(this, arguments);
                    c.mode = "show";
                    return this.effect.call(this, c)
                }
            }(c.fn.show),
            hide: function(b) {
                return function(a) {
                    if (h(a))
                        return b.apply(this,
                        arguments);
                    var c = q.apply(this, arguments);
                    c.mode = "hide";
                    return this.effect.call(this, c)
                }
            }(c.fn.hide),
            toggle: function(b) {
                return function(a) {
                    if (h(a) || "boolean" === typeof a)
                        return b.apply(this, arguments);
                    var c = q.apply(this, arguments);
                    c.mode = "toggle";
                    return this.effect.call(this, c)
                }
            }(c.fn.toggle),
            cssUnit: function(b) {
                var a = this.css(b), d = [];
                c.each(["em", "px", "%", "pt"], function(b, c) {
                    0 < a.indexOf(c) && (d = [parseFloat(a), c])
                });
                return d
            }
        })
    })();
    (function() {
        var q = {};
        c.each(["Quad", "Cubic", "Quart", "Quint", "Expo"],
        function(c, b) {
            q[b] = function(a) {
                return Math.pow(a, c + 2)
            }
        });
        c.extend(q, {
            Sine: function(c) {
                return 1 - Math.cos(c * Math.PI / 2)
            },
            Circ: function(c) {
                return 1 - Math.sqrt(1 - c * c)
            },
            Elastic: function(c) {
                return 0 === c || 1 === c ? c : - Math.pow(2, 8 * (c - 1)) * Math.sin((80 * (c - 1) - 7.5) * Math.PI / 15)
            },
            Back: function(c) {
                return c * c * (3 * c - 2)
            },
            Bounce: function(c) {
                for (var b, a = 4; c < ((b = Math.pow(2, --a)) - 1) / 11;);
                return 1 / Math.pow(4, 3 - a) - 7.5625 * Math.pow((3 * b - 2) / 22 - c, 2)
            }
        });
        c.each(q, function(h, b) {
            c.easing["easeIn" + h] = b;
            c.easing["easeOut" + h] = function(a) {
                return 1 -
                b(1 - a)
            };
            c.easing["easeInOut" + h] = function(a) {
                return 0.5 > a ? b(2 * a) / 2 : 1 - b( - 2 * a + 2) / 2
            }
        })
    })()
})(jimdoGen002);
