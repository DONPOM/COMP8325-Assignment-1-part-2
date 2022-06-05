function FastClick(e, t) {
    "use strict";
    function n(e, t) {
        return function() {
            return e.apply(t, arguments)
        }
    }
    var i;
    if (t = t || {}, this.trackingClick=!1, this.trackingClickStart = 0, this.targetElement = null, this.touchStartX = 0, this.touchStartY = 0, this.lastTouchIdentifier = 0, this.touchBoundary = t.touchBoundary || 10, this.layer = e, this.tapDelay = t.tapDelay || 200, !FastClick.notNeeded(e)) {
        for (var r = ["onMouse", "onClick", "onTouchStart", "onTouchMove", "onTouchEnd", "onTouchCancel"], o = this, a = 0, s = r.length; s > a; a++)
            o[r[a]] = n(o[r[a]], o);
        deviceIsAndroid && (e.addEventListener("mouseover", this.onMouse, !0), e.addEventListener("mousedown", this.onMouse, !0), e.addEventListener("mouseup", this.onMouse, !0)), e.addEventListener("click", this.onClick, !0), e.addEventListener("touchstart", this.onTouchStart, !1), e.addEventListener("touchmove", this.onTouchMove, !1), e.addEventListener("touchend", this.onTouchEnd, !1), e.addEventListener("touchcancel", this.onTouchCancel, !1), Event.prototype.stopImmediatePropagation || (e.removeEventListener = function(t, n, i) {
            var r = Node.prototype.removeEventListener;
            "click" === t ? r.call(e, t, n.hijacked || n, i) : r.call(e, t, n, i)
        }, e.addEventListener = function(t, n, i) {
            var r = Node.prototype.addEventListener;
            "click" === t ? r.call(e, t, n.hijacked || (n.hijacked = function(e) {
                e.propagationStopped || n(e)
            }), i) : r.call(e, t, n, i)
        }), "function" == typeof e.onclick && (i = e.onclick, e.addEventListener("click", function(e) {
            i(e)
        }, !1), e.onclick = null)
    }
}
!function(e, t) {
    "object" == typeof module && "object" == typeof module.exports ? module.exports = e.document ? t(e, !0) : function(e) {
        if (!e.document)
            throw new Error("jQuery requires a window with a document");
        return t(e)
    } : t(e)
}("undefined" != typeof window ? window : this, function(e, t) {
    function n(e) {
        var t = e.length, n = K.type(e);
        return "function" === n || K.isWindow(e)?!1 : 1 === e.nodeType && t?!0 : "array" === n || 0 === t || "number" == typeof t && t > 0 && t - 1 in e
    }
    function i(e, t, n) {
        if (K.isFunction(t))
            return K.grep(e, function(e, i) {
                return !!t.call(e, i, e) !== n
            });
        if (t.nodeType)
            return K.grep(e, function(e) {
                return e === t !== n
            });
        if ("string" == typeof t) {
            if (st.test(t))
                return K.filter(t, e, n);
            t = K.filter(t, e)
        }
        return K.grep(e, function(e) {
            return X.call(t, e) >= 0 !== n
        })
    }
    function r(e, t) {
        for (; (e = e[t]) && 1 !== e.nodeType;);
        return e
    }
    function o(e) {
        var t = pt[e] = {};
        return K.each(e.match(ht) || [], function(e, n) {
            t[n]=!0
        }), t
    }
    function a() {
        G.removeEventListener("DOMContentLoaded", a, !1), e.removeEventListener("load", a, !1), K.ready()
    }
    function s() {
        Object.defineProperty(this.cache = {}, 0, {
            get: function() {
                return {}
            }
        }), this.expando = K.expando + Math.random()
    }
    function l(e, t, n) {
        var i;
        if (void 0 === n && 1 === e.nodeType)
            if (i = "data-" + t.replace(_t, "-$1").toLowerCase(), n = e.getAttribute(i), "string" == typeof n) {
                try {
                    n = "true" === n?!0 : "false" === n?!1 : "null" === n ? null : + n + "" === n?+n : bt.test(n) ? K.parseJSON(n) : n
                } catch (r) {}
                yt.set(e, t, n)
            } else 
                n = void 0;
        return n
    }
    function c() {
        return !0
    }
    function u() {
        return !1
    }
    function d() {
        try {
            return G.activeElement
        } catch (e) {}
    }
    function f(e, t) {
        return K.nodeName(e, "table") && K.nodeName(11 !== t.nodeType ? t : t.firstChild, "tr") ? e.getElementsByTagName("tbody")[0] || e.appendChild(e.ownerDocument.createElement("tbody")) : e
    }
    function h(e) {
        return e.type = (null !== e.getAttribute("type")) + "/" + e.type, e
    }
    function p(e) {
        var t = Ot.exec(e.type);
        return t ? e.type = t[1] : e.removeAttribute("type"), e
    }
    function g(e, t) {
        for (var n = 0, i = e.length; i > n; n++)
            vt.set(e[n], "globalEval", !t || vt.get(t[n], "globalEval"))
    }
    function m(e, t) {
        var n, i, r, o, a, s, l, c;
        if (1 === t.nodeType) {
            if (vt.hasData(e) && (o = vt.access(e), a = vt.set(t, o), c = o.events)) {
                delete a.handle, a.events = {};
                for (r in c)
                    for (n = 0, i = c[r].length; i > n; n++)
                        K.event.add(t, r, c[r][n])
                    }
            yt.hasData(e) && (s = yt.access(e), l = K.extend({}, s), yt.set(t, l))
        }
    }
    function v(e, t) {
        var n = e.getElementsByTagName ? e.getElementsByTagName(t || "*"): e.querySelectorAll ? e.querySelectorAll(t || "*"): [];
        return void 0 === t || t && K.nodeName(e, t) ? K.merge([e], n) : n
    }
    function y(e, t) {
        var n = t.nodeName.toLowerCase();
        "input" === n && kt.test(e.type) ? t.checked = e.checked : ("input" === n || "textarea" === n) && (t.defaultValue = e.defaultValue)
    }
    function b(t, n) {
        var i, r = K(n.createElement(t)).appendTo(n.body), o = e.getDefaultComputedStyle && (i = e.getDefaultComputedStyle(r[0])) ? i.display: K.css(r[0], "display");
        return r.detach(), o
    }
    function _(e) {
        var t = G, n = Pt[e];
        return n || (n = b(e, t), "none" !== n && n || (Ht = (Ht || K("<iframe frameborder='0' width='0' height='0'/>")).appendTo(t.documentElement), t = Ht[0].contentDocument, t.write(), t.close(), n = b(e, t), Ht.detach()), Pt[e] = n), n
    }
    function w(e, t, n) {
        var i, r, o, a, s = e.style;
        return n = n || Bt(e), n && (a = n.getPropertyValue(t) || n[t]), n && ("" !== a || K.contains(e.ownerDocument, e) || (a = K.style(e, t)), $t.test(a) && zt.test(t) && (i = s.width, r = s.minWidth, o = s.maxWidth, s.minWidth = s.maxWidth = s.width = a, a = n.width, s.width = i, s.minWidth = r, s.maxWidth = o)), void 0 !== a ? a + "" : a
    }
    function x(e, t) {
        return {
            get: function() {
                return e() ? void delete this.get : (this.get = t).apply(this, arguments)
            }
        }
    }
    function C(e, t) {
        if (t in e)
            return t;
        for (var n = t[0].toUpperCase() + t.slice(1), i = t, r = Yt.length; r--;)
            if (t = Yt[r] + n, t in e)
                return t;
        return i
    }
    function k(e, t, n) {
        var i = Ut.exec(t);
        return i ? Math.max(0, i[1] - (n || 0)) + (i[2] || "px") : t
    }
    function S(e, t, n, i, r) {
        for (var o = n === (i ? "border" : "content") ? 4 : "width" === t ? 1 : 0, a = 0; 4 > o; o += 2)
            "margin" === n && (a += K.css(e, n + xt[o], !0, r)), i ? ("content" === n && (a -= K.css(e, "padding" + xt[o], !0, r)), "margin" !== n && (a -= K.css(e, "border" + xt[o] + "Width", !0, r))) : (a += K.css(e, "padding" + xt[o], !0, r), "padding" !== n && (a += K.css(e, "border" + xt[o] + "Width", !0, r)));
        return a
    }
    function T(e, t, n) {
        var i=!0, r = "width" === t ? e.offsetWidth : e.offsetHeight, o = Bt(e), a = "border-box" === K.css(e, "boxSizing", !1, o);
        if (0 >= r || null == r) {
            if (r = w(e, t, o), (0 > r || null == r) && (r = e.style[t]), $t.test(r))
                return r;
            i = a && (Z.boxSizingReliable() || r === e.style[t]), r = parseFloat(r) || 0
        }
        return r + S(e, t, n || (a ? "border" : "content"), i, o) + "px"
    }
    function F(e, t) {
        for (var n, i, r, o = [], a = 0, s = e.length; s > a; a++)
            i = e[a], i.style && (o[a] = vt.get(i, "olddisplay"), n = i.style.display, t ? (o[a] || "none" !== n || (i.style.display = ""), "" === i.style.display && Ct(i) && (o[a] = vt.access(i, "olddisplay", _(i.nodeName)))) : (r = Ct(i), "none" === n && r || vt.set(i, "olddisplay", r ? n : K.css(i, "display"))));
        for (a = 0; s > a; a++)
            i = e[a], i.style && (t && "none" !== i.style.display && "" !== i.style.display || (i.style.display = t ? o[a] || "" : "none"));
        return e
    }
    function A(e, t, n, i, r) {
        return new A.prototype.init(e, t, n, i, r)
    }
    function E() {
        return setTimeout(function() {
            Zt = void 0
        }), Zt = K.now()
    }
    function j(e, t) {
        var n, i = 0, r = {
            height: e
        };
        for (t = t ? 1 : 0; 4 > i; i += 2 - t)
            n = xt[i], r["margin" + n] = r["padding" + n] = e;
        return t && (r.opacity = r.width = e), r
    }
    function D(e, t, n) {
        for (var i, r = (nn[t] || []).concat(nn["*"]), o = 0, a = r.length; a > o; o++)
            if (i = r[o].call(n, t, e))
                return i
    }
    function N(e, t, n) {
        var i, r, o, a, s, l, c, u, d = this, f = {}, h = e.style, p = e.nodeType && Ct(e), g = vt.get(e, "fxshow");
        n.queue || (s = K._queueHooks(e, "fx"), null == s.unqueued && (s.unqueued = 0, l = s.empty.fire, s.empty.fire = function() {
            s.unqueued || l()
        }), s.unqueued++, d.always(function() {
            d.always(function() {
                s.unqueued--, K.queue(e, "fx").length || s.empty.fire()
            })
        })), 1 === e.nodeType && ("height"in t || "width"in t) && (n.overflow = [h.overflow, h.overflowX, h.overflowY], c = K.css(e, "display"), u = "none" === c ? vt.get(e, "olddisplay") || _(e.nodeName) : c, "inline" === u && "none" === K.css(e, "float") && (h.display = "inline-block")), n.overflow && (h.overflow = "hidden", d.always(function() {
            h.overflow = n.overflow[0], h.overflowX = n.overflow[1], h.overflowY = n.overflow[2]
        }));
        for (i in t)
            if (r = t[i], Jt.exec(r)) {
                if (delete t[i], o = o || "toggle" === r, r === (p ? "hide" : "show")) {
                    if ("show" !== r ||!g || void 0 === g[i])
                        continue;
                        p=!0
                }
                f[i] = g && g[i] || K.style(e, i)
            } else 
                c = void 0;
        if (K.isEmptyObject(f))
            "inline" === ("none" === c ? _(e.nodeName) : c) && (h.display = c);
        else {
            g ? "hidden"in g && (p = g.hidden) : g = vt.access(e, "fxshow", {}), o && (g.hidden=!p), p ? K(e).show() : d.done(function() {
                K(e).hide()
            }), d.done(function() {
                var t;
                vt.remove(e, "fxshow");
                for (t in f)
                    K.style(e, t, f[t])
            });
            for (i in f)
                a = D(p ? g[i] : 0, i, d), i in g || (g[i] = a.start, p && (a.end = a.start, a.start = "width" === i || "height" === i ? 1 : 0))
        }
    }
    function q(e, t) {
        var n, i, r, o, a;
        for (n in e)
            if (i = K.camelCase(n), r = t[i], o = e[n], K.isArray(o) && (r = o[1], o = e[n] = o[0]), n !== i && (e[i] = o, delete e[n]), a = K.cssHooks[i], a && "expand"in a) {
                o = a.expand(o), delete e[i];
                for (n in o)
                    n in e || (e[n] = o[n], t[n] = r)
            } else 
                t[i] = r
    }
    function L(e, t, n) {
        var i, r, o = 0, a = tn.length, s = K.Deferred().always(function() {
            delete l.elem
        }), l = function() {
            if (r)
                return !1;
            for (var t = Zt || E(), n = Math.max(0, c.startTime + c.duration - t), i = n / c.duration || 0, o = 1 - i, a = 0, l = c.tweens.length; l > a; a++)
                c.tweens[a].run(o);
            return s.notifyWith(e, [c, o, n]), 1 > o && l ? n : (s.resolveWith(e, [c]), !1)
        }, c = s.promise({
            elem: e,
            props: K.extend({}, t),
            opts: K.extend(!0, {
                specialEasing: {}
            }, n),
            originalProperties: t,
            originalOptions: n,
            startTime: Zt || E(),
            duration: n.duration,
            tweens: [],
            createTween: function(t, n) {
                var i = K.Tween(e, c.opts, t, n, c.opts.specialEasing[t] || c.opts.easing);
                return c.tweens.push(i), i
            },
            stop: function(t) {
                var n = 0, i = t ? c.tweens.length: 0;
                if (r)
                    return this;
                for (r=!0; i > n; n++)
                    c.tweens[n].run(1);
                return t ? s.resolveWith(e, [c, t]) : s.rejectWith(e, [c, t]), this
            }
        }), u = c.props;
        for (q(u, c.opts.specialEasing);
        a > o;
        o++)if (i = tn[o].call(c, e, u, c.opts))
            return i;
        return K.map(u, D, c), K.isFunction(c.opts.start) && c.opts.start.call(e, c), K.fx.timer(K.extend(l, {
            elem: e,
            anim: c,
            queue: c.opts.queue
        })), c.progress(c.opts.progress).done(c.opts.done, c.opts.complete).fail(c.opts.fail).always(c.opts.always)
    }
    function I(e) {
        return function(t, n) {
            "string" != typeof t && (n = t, t = "*");
            var i, r = 0, o = t.toLowerCase().match(ht) || [];
            if (K.isFunction(n))
                for (; i = o[r++];)
                    "+" === i[0] ? (i = i.slice(1) || "*", (e[i] = e[i] || []).unshift(n)) : (e[i] = e[i] || []).push(n)
        }
    }
    function O(e, t, n, i) {
        function r(s) {
            var l;
            return o[s]=!0, K.each(e[s] || [], function(e, s) {
                var c = s(t, n, i);
                return "string" != typeof c || a || o[c] ? a?!(l = c) : void 0 : (t.dataTypes.unshift(c), r(c), !1)
            }), l
        }
        var o = {}, a = e === xn;
        return r(t.dataTypes[0]) ||!o["*"] && r("*")
    }
    function M(e, t) {
        var n, i, r = K.ajaxSettings.flatOptions || {};
        for (n in t)
            void 0 !== t[n] && ((r[n] ? e : i || (i = {}))[n] = t[n]);
        return i && K.extend(!0, e, i), e
    }
    function R(e, t, n) {
        for (var i, r, o, a, s = e.contents, l = e.dataTypes; "*" === l[0];)
            l.shift(), void 0 === i && (i = e.mimeType || t.getResponseHeader("Content-Type"));
        if (i)
            for (r in s)
                if (s[r] && s[r].test(i)) {
                    l.unshift(r);
                    break
                }
        if (l[0]in n)
            o = l[0];
        else {
            for (r in n) {
                if (!l[0] || e.converters[r + " " + l[0]]) {
                    o = r;
                    break
                }
                a || (a = r)
            }
            o = o || a
        }
        return o ? (o !== l[0] && l.unshift(o), n[o]) : void 0
    }
    function H(e, t, n, i) {
        var r, o, a, s, l, c = {}, u = e.dataTypes.slice();
        if (u[1])
            for (a in e.converters)
                c[a.toLowerCase()] = e.converters[a];
        for (o = u.shift(); o;)
            if (e.responseFields[o] && (n[e.responseFields[o]] = t), !l && i && e.dataFilter && (t = e.dataFilter(t, e.dataType)), l = o, o = u.shift())
                if ("*" === o)
                    o = l;
                else if ("*" !== l && l !== o) {
                    if (a = c[l + " " + o] || c["* " + o], !a)
                        for (r in c)
                            if (s = r.split(" "), s[1] === o && (a = c[l + " " + s[0]] || c["* " + s[0]])) {
                                a===!0 ? a = c[r] : c[r]!==!0 && (o = s[0], u.unshift(s[1]));
                                break
                            }
                            if (a!==!0)
                                if (a && e["throws"])
                                    t = a(t);
                                else 
                                    try {
                                        t = a(t)
                                    } catch (d) {
                                        return {
                                            state: "parsererror",
                                            error: a ? d: "No conversion from " + l + " to " + o
                                        }
                                    }
                                }
        return {
            state: "success",
            data: t
        }
    }
    function P(e, t, n, i) {
        var r;
        if (K.isArray(t))
            K.each(t, function(t, r) {
                n || Tn.test(e) ? i(e, r) : P(e + "[" + ("object" == typeof r ? t : "") + "]", r, n, i)
            });
        else if (n || "object" !== K.type(t))
            i(e, t);
        else 
            for (r in t)
                P(e + "[" + r + "]", t[r], n, i)
            }
    function z(e) {
        return K.isWindow(e) ? e : 9 === e.nodeType && e.defaultView
    }
    var $ = [], B = $.slice, W = $.concat, U = $.push, X = $.indexOf, Q = {}, V = Q.toString, Y = Q.hasOwnProperty, Z = {}, G = e.document, J = "2.1.1", K = function(e, t) {
        return new K.fn.init(e, t)
    }, et = /^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g, tt = /^-ms-/, nt = /-([\da-z])/gi, it = function(e, t) {
        return t.toUpperCase()
    };
    K.fn = K.prototype = {
        jquery: J,
        constructor: K,
        selector: "",
        length: 0,
        toArray: function() {
            return B.call(this)
        },
        get: function(e) {
            return null != e ? 0 > e ? this[e + this.length] : this[e] : B.call(this)
        },
        pushStack: function(e) {
            var t = K.merge(this.constructor(), e);
            return t.prevObject = this, t.context = this.context, t
        },
        each: function(e, t) {
            return K.each(this, e, t)
        },
        map: function(e) {
            return this.pushStack(K.map(this, function(t, n) {
                return e.call(t, n, t)
            }))
        },
        slice: function() {
            return this.pushStack(B.apply(this, arguments))
        },
        first: function() {
            return this.eq(0)
        },
        last: function() {
            return this.eq( - 1)
        },
        eq: function(e) {
            var t = this.length, n =+ e + (0 > e ? t : 0);
            return this.pushStack(n >= 0 && t > n ? [this[n]] : [])
        },
        end: function() {
            return this.prevObject || this.constructor(null)
        },
        push: U,
        sort: $.sort,
        splice: $.splice
    }, K.extend = K.fn.extend = function() {
        var e, t, n, i, r, o, a = arguments[0] || {}, s = 1, l = arguments.length, c=!1;
        for ("boolean" == typeof a && (c = a, a = arguments[s] || {}, s++), "object" == typeof a || K.isFunction(a) || (a = {}), s === l && (a = this, s--); l > s; s++)
            if (null != (e = arguments[s]))
                for (t in e)
                    n = a[t], i = e[t], a !== i && (c && i && (K.isPlainObject(i) || (r = K.isArray(i))) ? (r ? (r=!1, o = n && K.isArray(n) ? n : []) : o = n && K.isPlainObject(n) ? n : {}, a[t] = K.extend(c, o, i)) : void 0 !== i && (a[t] = i));
        return a
    }, K.extend({
        expando: "jQuery" + (J + Math.random()).replace(/\D/g, ""),
        isReady: !0,
        error: function(e) {
            throw new Error(e)
        },
        noop: function() {},
        isFunction: function(e) {
            return "function" === K.type(e)
        },
        isArray: Array.isArray,
        isWindow: function(e) {
            return null != e && e === e.window
        },
        isNumeric: function(e) {
            return !K.isArray(e) && e - parseFloat(e) >= 0
        },
        isPlainObject: function(e) {
            return "object" !== K.type(e) || e.nodeType || K.isWindow(e)?!1 : e.constructor&&!Y.call(e.constructor.prototype, "isPrototypeOf")?!1 : !0
        },
        isEmptyObject: function(e) {
            var t;
            for (t in e)
                return !1;
            return !0
        },
        type: function(e) {
            return null == e ? e + "" : "object" == typeof e || "function" == typeof e ? Q[V.call(e)] || "object" : typeof e
        },
        globalEval: function(e) {
            var t, n = eval;
            e = K.trim(e), e && (1 === e.indexOf("use strict") ? (t = G.createElement("script"), t.text = e, G.head.appendChild(t).parentNode.removeChild(t)) : n(e))
        },
        camelCase: function(e) {
            return e.replace(tt, "ms-").replace(nt, it)
        },
        nodeName: function(e, t) {
            return e.nodeName && e.nodeName.toLowerCase() === t.toLowerCase()
        },
        each: function(e, t, i) {
            var r, o = 0, a = e.length, s = n(e);
            if (i) {
                if (s)
                    for (; a > o && (r = t.apply(e[o], i), r!==!1); o++);
                else 
                    for (o in e)
                        if (r = t.apply(e[o], i), r===!1)
                            break
            } else if (s)
                for (; a > o && (r = t.call(e[o], o, e[o]), r!==!1); o++);
            else 
                for (o in e)
                    if (r = t.call(e[o], o, e[o]), r===!1)
                        break;
            return e
        },
        trim: function(e) {
            return null == e ? "" : (e + "").replace(et, "")
        },
        makeArray: function(e, t) {
            var i = t || [];
            return null != e && (n(Object(e)) ? K.merge(i, "string" == typeof e ? [e] : e) : U.call(i, e)), i
        },
        inArray: function(e, t, n) {
            return null == t?-1 : X.call(t, e, n)
        },
        merge: function(e, t) {
            for (var n =+ t.length, i = 0, r = e.length; n > i; i++)
                e[r++] = t[i];
            return e.length = r, e
        },
        grep: function(e, t, n) {
            for (var i, r = [], o = 0, a = e.length, s=!n; a > o; o++)
                i=!t(e[o], o), i !== s && r.push(e[o]);
            return r
        },
        map: function(e, t, i) {
            var r, o = 0, a = e.length, s = n(e), l = [];
            if (s)
                for (; a > o; o++)
                    r = t(e[o], o, i), null != r && l.push(r);
            else 
                for (o in e)
                    r = t(e[o], o, i), null != r && l.push(r);
            return W.apply([], l)
        },
        guid: 1,
        proxy: function(e, t) {
            var n, i, r;
            return "string" == typeof t && (n = e[t], t = e, e = n), K.isFunction(e) ? (i = B.call(arguments, 2), r = function() {
                return e.apply(t || this, i.concat(B.call(arguments)))
            }, r.guid = e.guid = e.guid || K.guid++, r) : void 0
        },
        now: Date.now,
        support: Z
    }), K.each("Boolean Number String Function Array Date RegExp Object Error".split(" "), function(e, t) {
        Q["[object " + t + "]"] = t.toLowerCase()
    });
    var rt = function(e) {
        function t(e, t, n, i) {
            var r, o, a, s, l, c, d, h, p, g;
            if ((t ? t.ownerDocument || t : P) !== N && D(t), t = t || N, n = n || [], !e || "string" != typeof e)
                return n;
            if (1 !== (s = t.nodeType) && 9 !== s)
                return [];
            if (L&&!i) {
                if (r = yt.exec(e))
                    if (a = r[1]) {
                        if (9 === s) {
                            if (o = t.getElementById(a), !o ||!o.parentNode)
                                return n;
                                if (o.id === a)
                                    return n.push(o), n
                        } else if (t.ownerDocument && (o = t.ownerDocument.getElementById(a)) && R(t, o) && o.id === a)
                            return n.push(o), n
                    } else {
                        if (r[2])
                            return K.apply(n, t.getElementsByTagName(e)), n;
                            if ((a = r[3]) && w.getElementsByClassName && t.getElementsByClassName)
                                return K.apply(n, t.getElementsByClassName(a)), n
                    }
                if (w.qsa && (!I ||!I.test(e))) {
                    if (h = d = H, p = t, g = 9 === s && e, 1 === s && "object" !== t.nodeName.toLowerCase()) {
                        for (c = S(e), (d = t.getAttribute("id")) ? h = d.replace(_t, "\\$&") : t.setAttribute("id", h), h = "[id='" + h + "'] ", l = c.length; l--;)
                            c[l] = h + f(c[l]);
                        p = bt.test(e) && u(t.parentNode) || t, g = c.join(",")
                    }
                    if (g)
                        try {
                            return K.apply(n, p.querySelectorAll(g)), n
                    } catch (m) {} finally {
                        d || t.removeAttribute("id")
                    }
                }
            }
            return F(e.replace(lt, "$1"), t, n, i)
        }
        function n() {
            function e(n, i) {
                return t.push(n + " ") > x.cacheLength && delete e[t.shift()], e[n + " "] = i
            }
            var t = [];
            return e
        }
        function i(e) {
            return e[H]=!0, e
        }
        function r(e) {
            var t = N.createElement("div");
            try {
                return !!e(t)
            } catch (n) {
                return !1
            } finally {
                t.parentNode && t.parentNode.removeChild(t), t = null
            }
        }
        function o(e, t) {
            for (var n = e.split("|"), i = e.length; i--;)
                x.attrHandle[n[i]] = t
        }
        function a(e, t) {
            var n = t && e, i = n && 1 === e.nodeType && 1 === t.nodeType && (~t.sourceIndex || V) - (~e.sourceIndex || V);
            if (i)
                return i;
            if (n)
                for (; n = n.nextSibling;)
                    if (n === t)
                        return - 1;
            return e ? 1 : - 1
        }
        function s(e) {
            return function(t) {
                var n = t.nodeName.toLowerCase();
                return "input" === n && t.type === e
            }
        }
        function l(e) {
            return function(t) {
                var n = t.nodeName.toLowerCase();
                return ("input" === n || "button" === n) && t.type === e
            }
        }
        function c(e) {
            return i(function(t) {
                return t =+ t, i(function(n, i) {
                    for (var r, o = e([], n.length, t), a = o.length; a--;)
                        n[r = o[a]] && (n[r]=!(i[r] = n[r]))
                })
            })
        }
        function u(e) {
            return e && typeof e.getElementsByTagName !== Q && e
        }
        function d() {}
        function f(e) {
            for (var t = 0, n = e.length, i = ""; n > t; t++)
                i += e[t].value;
            return i
        }
        function h(e, t, n) {
            var i = t.dir, r = n && "parentNode" === i, o = $++;
            return t.first ? function(t, n, o) {
                for (; t = t[i];)
                    if (1 === t.nodeType || r)
                        return e(t, n, o)
            } : function(t, n, a) {
                var s, l, c = [z, o];
                if (a) {
                    for (; t = t[i];)
                        if ((1 === t.nodeType || r) && e(t, n, a))
                            return !0
                } else 
                    for (; t = t[i];)
                        if (1 === t.nodeType || r) {
                            if (l = t[H] || (t[H] = {}), (s = l[i]) && s[0] === z && s[1] === o)
                                return c[2] = s[2];
                                if (l[i] = c, c[2] = e(t, n, a))
                                    return !0
                        }
            }
        }
        function p(e) {
            return e.length > 1 ? function(t, n, i) {
                for (var r = e.length; r--;)
                    if (!e[r](t, n, i))
                        return !1;
                return !0
            } : e[0]
        }
        function g(e, n, i) {
            for (var r = 0, o = n.length; o > r; r++)
                t(e, n[r], i);
            return i
        }
        function m(e, t, n, i, r) {
            for (var o, a = [], s = 0, l = e.length, c = null != t; l > s; s++)(o = e[s]) 
                && (!n || n(o, i, r)) && (a.push(o), c && t.push(s));
            return a
        }
        function v(e, t, n, r, o, a) {
            return r&&!r[H] && (r = v(r)), o&&!o[H] && (o = v(o, a)), i(function(i, a, s, l) {
                var c, u, d, f = [], h = [], p = a.length, v = i || g(t || "*", s.nodeType ? [s] : s, []), y=!e ||!i && t ? v : m(v, f, e, s, l), b = n ? o || (i ? e : p || r) ? [] : a : y;
                if (n && n(y, b, s, l), r)
                    for (c = m(b, h), r(c, [], s, l), u = c.length; u--;)(d = c[u]) 
                        && (b[h[u]]=!(y[h[u]] = d));
                if (i) {
                    if (o || e) {
                        if (o) {
                            for (c = [], u = b.length; u--;)(d = b[u]) 
                                && c.push(y[u] = d);
                            o(null, b = [], c, l)
                        }
                        for (u = b.length; u--;)(d = b[u]) 
                            && (c = o ? tt.call(i, d) : f[u])>-1 && (i[c]=!(a[c] = d))
                        }
                } else 
                    b = m(b === a ? b.splice(p, b.length) : b), o ? o(null, a, b, l) : K.apply(a, b)
            })
        }
        function y(e) {
            for (var t, n, i, r = e.length, o = x.relative[e[0].type], a = o || x.relative[" "], s = o ? 1 : 0, l = h(function(e) {
                return e === t
            }, a, !0), c = h(function(e) {
                return tt.call(t, e)>-1
            }, a, !0), u = [function(e, n, i) {
                return !o && (i || n !== A) || ((t = n).nodeType ? l(e, n, i) : c(e, n, i))
            }
            ]; r > s; s++)
                if (n = x.relative[e[s].type])
                    u = [h(p(u), n)];
                else {
                    if (n = x.filter[e[s].type].apply(null, e[s].matches), n[H]) {
                        for (i=++s; r > i&&!x.relative[e[i].type]; i++);
                        return v(s > 1 && p(u), s > 1 && f(e.slice(0, s - 1).concat({
                            value: " " === e[s - 2].type ? "*": ""
                        })).replace(lt, "$1"), n, i > s && y(e.slice(s, i)), r > i && y(e = e.slice(i)), r > i && f(e))
                    }
                    u.push(n)
                }
            return p(u)
        }
        function b(e, n) {
            var r = n.length > 0, o = e.length > 0, a = function(i, a, s, l, c) {
                var u, d, f, h = 0, p = "0", g = i && [], v = [], y = A, b = i || o && x.find.TAG("*", c), _ = z += null == y ? 1: Math.random() || .1, w = b.length;
                for (c && (A = a !== N && a); p !== w && null != (u = b[p]); p++) {
                    if (o && u) {
                        for (d = 0; f = e[d++];)
                            if (f(u, a, s)) {
                                l.push(u);
                                break
                            }
                        c && (z = _)
                    }
                    r && ((u=!f && u) && h--, i && g.push(u))
                }
                if (h += p, r && p !== h) {
                    for (d = 0; f = n[d++];)
                        f(g, v, a, s);
                    if (i) {
                        if (h > 0)
                            for (; p--;)
                                g[p] || v[p] || (v[p] = G.call(l));
                        v = m(v)
                    }
                    K.apply(l, v), c&&!i && v.length > 0 && h + n.length > 1 && t.uniqueSort(l)
                }
                return c && (z = _, A = y), g
            };
            return r ? i(a) : a
        }
        var _, w, x, C, k, S, T, F, A, E, j, D, N, q, L, I, O, M, R, H = "sizzle" +- new Date, P = e.document, z = 0, $ = 0, B = n(), W = n(), U = n(), X = function(e, t) {
            return e === t && (j=!0), 0
        }, Q = "undefined", V = 1<<31, Y = {}.hasOwnProperty, Z = [], G = Z.pop, J = Z.push, K = Z.push, et = Z.slice, tt = Z.indexOf || function(e) {
            for (var t = 0, n = this.length; n > t; t++)
                if (this[t] === e)
                    return t;
            return - 1
        }, nt = "checked|selected|async|autofocus|autoplay|controls|defer|disabled|hidden|ismap|loop|multiple|open|readonly|required|scoped", it = "[\\x20\\t\\r\\n\\f]", rt = "(?:\\\\.|[\\w-]|[^\\x00-\\xa0])+", ot = rt.replace("w", "w#"), at = "\\[" + it + "*(" + rt + ")(?:" + it + "*([*^$|!~]?=)" + it + "*(?:'((?:\\\\.|[^\\\\'])*)'|\"((?:\\\\.|[^\\\\\"])*)\"|(" + ot + "))|)" + it + "*\\]", st = ":(" + rt + ")(?:\\((('((?:\\\\.|[^\\\\'])*)'|\"((?:\\\\.|[^\\\\\"])*)\")|((?:\\\\.|[^\\\\()[\\]]|" + at + ")*)|.*)\\)|)", lt = new RegExp("^" + it + "+|((?:^|[^\\\\])(?:\\\\.)*)" + it + "+$", "g"), ct = new RegExp("^" + it + "*," + it + "*"), ut = new RegExp("^" + it + "*([>+~]|" + it + ")" + it + "*"), dt = new RegExp("=" + it + "*([^\\]'\"]*?)" + it + "*\\]", "g"), ft = new RegExp(st), ht = new RegExp("^" + ot + "$"), pt = {
            ID: new RegExp("^#(" + rt + ")"),
            CLASS: new RegExp("^\\.(" + rt + ")"),
            TAG: new RegExp("^(" + rt.replace("w", "w*") + ")"),
            ATTR: new RegExp("^" + at),
            PSEUDO: new RegExp("^" + st),
            CHILD: new RegExp("^:(only|first|last|nth|nth-last)-(child|of-type)(?:\\(" + it + "*(even|odd|(([+-]|)(\\d*)n|)" + it + "*(?:([+-]|)" + it + "*(\\d+)|))" + it + "*\\)|)", "i"),
            bool: new RegExp("^(?:" + nt + ")$", "i"),
            needsContext: new RegExp("^" + it + "*[>+~]|:(even|odd|eq|gt|lt|nth|first|last)(?:\\(" + it + "*((?:-\\d)?\\d*)" + it + "*\\)|)(?=[^-]|$)", "i")
        }, gt = /^(?:input|select|textarea|button)$/i, mt = /^h\d$/i, vt = /^[^{]+\{\s*\[native \w/, yt = /^(?:#([\w-]+)|(\w+)|\.([\w-]+))$/, bt = /[+~]/, _t = /'|\\/g, wt = new RegExp("\\\\([\\da-f]{1,6}" + it + "?|(" + it + ")|.)", "ig"), xt = function(e, t, n) {
            var i = "0x" + t - 65536;
            return i !== i || n ? t : 0 > i ? String.fromCharCode(i + 65536) : String.fromCharCode(i>>10 | 55296, 1023 & i | 56320)
        };
        try {
            K.apply(Z = et.call(P.childNodes), P.childNodes), Z[P.childNodes.length].nodeType
        } catch (Ct) {
            K = {
                apply: Z.length ? function(e, t) {
                    J.apply(e, et.call(t))
                }
                : function(e, t) {
                    for (var n = e.length, i = 0; e[n++] = t[i++];);
                    e.length = n - 1
                }
            }
        }
        w = t.support = {}, k = t.isXML = function(e) {
            var t = e && (e.ownerDocument || e).documentElement;
            return t ? "HTML" !== t.nodeName : !1
        }, D = t.setDocument = function(e) {
            var t, n = e ? e.ownerDocument || e: P, i = n.defaultView;
            return n !== N && 9 === n.nodeType && n.documentElement ? (N = n, q = n.documentElement, L=!k(n), i && i !== i.top && (i.addEventListener ? i.addEventListener("unload", function() {
                D()
            }, !1) : i.attachEvent && i.attachEvent("onunload", function() {
                D()
            })), w.attributes = r(function(e) {
                return e.className = "i", !e.getAttribute("className")
            }), w.getElementsByTagName = r(function(e) {
                return e.appendChild(n.createComment("")), !e.getElementsByTagName("*").length
            }), w.getElementsByClassName = vt.test(n.getElementsByClassName) && r(function(e) {
                return e.innerHTML = "<div class='a'></div><div class='a i'></div>", e.firstChild.className = "i", 2 === e.getElementsByClassName("i").length
            }), w.getById = r(function(e) {
                return q.appendChild(e).id = H, !n.getElementsByName ||!n.getElementsByName(H).length
            }), w.getById ? (x.find.ID = function(e, t) {
                if (typeof t.getElementById !== Q && L) {
                    var n = t.getElementById(e);
                    return n && n.parentNode ? [n] : []
                }
            }, x.filter.ID = function(e) {
                var t = e.replace(wt, xt);
                return function(e) {
                    return e.getAttribute("id") === t
                }
            }) : (delete x.find.ID, x.filter.ID = function(e) {
                var t = e.replace(wt, xt);
                return function(e) {
                    var n = typeof e.getAttributeNode !== Q && e.getAttributeNode("id");
                    return n && n.value === t
                }
            }), x.find.TAG = w.getElementsByTagName ? function(e, t) {
                return typeof t.getElementsByTagName !== Q ? t.getElementsByTagName(e) : void 0
            } : function(e, t) {
                var n, i = [], r = 0, o = t.getElementsByTagName(e);
                if ("*" === e) {
                    for (; n = o[r++];)
                        1 === n.nodeType && i.push(n);
                    return i
                }
                return o
            }, x.find.CLASS = w.getElementsByClassName && function(e, t) {
                return typeof t.getElementsByClassName !== Q && L ? t.getElementsByClassName(e) : void 0
            }, O = [], I = [], (w.qsa = vt.test(n.querySelectorAll)) && (r(function(e) {
                e.innerHTML = "<select msallowclip=''><option selected=''></option></select>", e.querySelectorAll("[msallowclip^='']").length && I.push("[*^$]=" + it + "*(?:''|\"\")"), e.querySelectorAll("[selected]").length || I.push("\\[" + it + "*(?:value|" + nt + ")"), e.querySelectorAll(":checked").length || I.push(":checked")
            }), r(function(e) {
                var t = n.createElement("input");
                t.setAttribute("type", "hidden"), e.appendChild(t).setAttribute("name", "D"), e.querySelectorAll("[name=d]").length && I.push("name" + it + "*[*^$|!~]?="), e.querySelectorAll(":enabled").length || I.push(":enabled", ":disabled"), e.querySelectorAll("*,:x"), I.push(",.*:")
            })), (w.matchesSelector = vt.test(M = q.matches || q.webkitMatchesSelector || q.mozMatchesSelector || q.oMatchesSelector || q.msMatchesSelector)) && r(function(e) {
                w.disconnectedMatch = M.call(e, "div"), M.call(e, "[s!='']:x"), O.push("!=", st)
            }), I = I.length && new RegExp(I.join("|")), O = O.length && new RegExp(O.join("|")), t = vt.test(q.compareDocumentPosition), R = t || vt.test(q.contains) ? function(e, t) {
                var n = 9 === e.nodeType ? e.documentElement: e, i = t && t.parentNode;
                return e === i ||!(!i || 1 !== i.nodeType ||!(n.contains ? n.contains(i) : e.compareDocumentPosition && 16 & e.compareDocumentPosition(i)))
            } : function(e, t) {
                if (t)
                    for (; t = t.parentNode;)
                        if (t === e)
                            return !0;
                return !1
            }, X = t ? function(e, t) {
                if (e === t)
                    return j=!0, 0;
                var i=!e.compareDocumentPosition-!t.compareDocumentPosition;
                return i ? i : (i = (e.ownerDocument || e) === (t.ownerDocument || t) ? e.compareDocumentPosition(t) : 1, 1 & i ||!w.sortDetached && t.compareDocumentPosition(e) === i ? e === n || e.ownerDocument === P && R(P, e)?-1 : t === n || t.ownerDocument === P && R(P, t) ? 1 : E ? tt.call(E, e) - tt.call(E, t) : 0 : 4 & i?-1 : 1)
            } : function(e, t) {
                if (e === t)
                    return j=!0, 0;
                var i, r = 0, o = e.parentNode, s = t.parentNode, l = [e], c = [t];
                if (!o ||!s)
                    return e === n?-1 : t === n ? 1 : o?-1 : s ? 1 : E ? tt.call(E, e) - tt.call(E, t) : 0;
                if (o === s)
                    return a(e, t);
                for (i = e; i = i.parentNode;)
                    l.unshift(i);
                for (i = t; i = i.parentNode;)
                    c.unshift(i);
                for (; l[r] === c[r];)
                    r++;
                return r ? a(l[r], c[r]) : l[r] === P?-1 : c[r] === P ? 1 : 0
            }, n) : N
        }, t.matches = function(e, n) {
            return t(e, null, null, n)
        }, t.matchesSelector = function(e, n) {
            if ((e.ownerDocument || e) !== N && D(e), n = n.replace(dt, "='$1']"), !(!w.matchesSelector ||!L || O && O.test(n) || I && I.test(n)))
                try {
                    var i = M.call(e, n);
                    if (i || w.disconnectedMatch || e.document && 11 !== e.document.nodeType)
                        return i
            } catch (r) {}
            return t(n, N, null, [e]).length > 0
        }, t.contains = function(e, t) {
            return (e.ownerDocument || e) !== N && D(e), R(e, t)
        }, t.attr = function(e, t) {
            (e.ownerDocument || e) !== N && D(e);
            var n = x.attrHandle[t.toLowerCase()], i = n && Y.call(x.attrHandle, t.toLowerCase()) ? n(e, t, !L): void 0;
            return void 0 !== i ? i : w.attributes ||!L ? e.getAttribute(t) : (i = e.getAttributeNode(t)) && i.specified ? i.value : null
        }, t.error = function(e) {
            throw new Error("Syntax error, unrecognized expression: " + e)
        }, t.uniqueSort = function(e) {
            var t, n = [], i = 0, r = 0;
            if (j=!w.detectDuplicates, E=!w.sortStable && e.slice(0), e.sort(X), j) {
                for (; t = e[r++];)
                    t === e[r] && (i = n.push(r));
                for (; i--;)
                    e.splice(n[i], 1)
            }
            return E = null, e
        }, C = t.getText = function(e) {
            var t, n = "", i = 0, r = e.nodeType;
            if (r) {
                if (1 === r || 9 === r || 11 === r) {
                    if ("string" == typeof e.textContent)
                        return e.textContent;
                    for (e = e.firstChild; e; e = e.nextSibling)
                        n += C(e)
                    } else if (3 === r || 4 === r)
                    return e.nodeValue
            } else 
                for (; t = e[i++];)
                    n += C(t);
            return n
        }, x = t.selectors = {
            cacheLength: 50,
            createPseudo: i,
            match: pt,
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
                ATTR: function(e) {
                    return e[1] = e[1].replace(wt, xt), e[3] = (e[3] || e[4] || e[5] || "").replace(wt, xt), "~=" === e[2] && (e[3] = " " + e[3] + " "), e.slice(0, 4)
                },
                CHILD: function(e) {
                    return e[1] = e[1].toLowerCase(), "nth" === e[1].slice(0, 3) ? (e[3] || t.error(e[0]), e[4] =+ (e[4] ? e[5] + (e[6] || 1) : 2 * ("even" === e[3] || "odd" === e[3])), e[5] =+ (e[7] + e[8] || "odd" === e[3])) : e[3] && t.error(e[0]), e
                },
                PSEUDO: function(e) {
                    var t, n=!e[6] && e[2];
                    return pt.CHILD.test(e[0]) ? null : (e[3] ? e[2] = e[4] || e[5] || "" : n && ft.test(n) && (t = S(n, !0)) && (t = n.indexOf(")", n.length - t) - n.length) && (e[0] = e[0].slice(0, t), e[2] = n.slice(0, t)), e.slice(0, 3))
                }
            },
            filter: {
                TAG: function(e) {
                    var t = e.replace(wt, xt).toLowerCase();
                    return "*" === e ? function() {
                        return !0
                    } : function(e) {
                        return e.nodeName && e.nodeName.toLowerCase() === t
                    }
                },
                CLASS: function(e) {
                    var t = B[e + " "];
                    return t || (t = new RegExp("(^|" + it + ")" + e + "(" + it + "|$)")) && B(e, function(e) {
                        return t.test("string" == typeof e.className && e.className || typeof e.getAttribute !== Q && e.getAttribute("class") || "")
                    })
                },
                ATTR: function(e, n, i) {
                    return function(r) {
                        var o = t.attr(r, e);
                        return null == o ? "!=" === n : n ? (o += "", "=" === n ? o === i : "!=" === n ? o !== i : "^=" === n ? i && 0 === o.indexOf(i) : "*=" === n ? i && o.indexOf(i)>-1 : "$=" === n ? i && o.slice( - i.length) === i : "~=" === n ? (" " + o + " ").indexOf(i)>-1 : "|=" === n ? o === i || o.slice(0, i.length + 1) === i + "-" : !1) : !0
                    }
                },
                CHILD: function(e, t, n, i, r) {
                    var o = "nth" !== e.slice(0, 3), a = "last" !== e.slice( - 4), s = "of-type" === t;
                    return 1 === i && 0 === r ? function(e) {
                        return !!e.parentNode
                    } : function(t, n, l) {
                        var c, u, d, f, h, p, g = o !== a ? "nextSibling": "previousSibling", m = t.parentNode, v = s && t.nodeName.toLowerCase(), y=!l&&!s;
                        if (m) {
                            if (o) {
                                for (; g;) {
                                    for (d = t; d = d[g];)
                                        if (s ? d.nodeName.toLowerCase() === v : 1 === d.nodeType)
                                            return !1;
                                    p = g = "only" === e&&!p && "nextSibling"
                                }
                                return !0
                            }
                            if (p = [a ? m.firstChild: m.lastChild], a && y) {
                                for (u = m[H] || (m[H] = {}), c = u[e] || [], h = c[0] === z && c[1], f = c[0] === z && c[2], d = h && m.childNodes[h]; d=++h && d && d[g] || (f = h = 0) 
                                    || p.pop();
                                )if (1 === d.nodeType&&++f && d === t) {
                                    u[e] = [z, h, f];
                                    break
                                }
                            } else if (y && (c = (t[H] || (t[H] = {}))[e]) && c[0] === z)
                                f = c[1];
                            else 
                                for (; (d=++h && d && d[g] || (f = h = 0) || p.pop()) 
                                    && ((s ? d.nodeName.toLowerCase() !== v : 1 !== d.nodeType)||!++f || (y && ((d[H] || (d[H] = {}))[e] = [z, f]), d !== t));
                            );
                            return f -= r, f === i || f%i === 0 && f / i >= 0
                        }
                    }
                },
                PSEUDO: function(e, n) {
                    var r, o = x.pseudos[e] || x.setFilters[e.toLowerCase()] || t.error("unsupported pseudo: " + e);
                    return o[H] ? o(n) : o.length > 1 ? (r = [e, e, "", n], x.setFilters.hasOwnProperty(e.toLowerCase()) ? i(function(e, t) {
                        for (var i, r = o(e, n), a = r.length; a--;)
                            i = tt.call(e, r[a]), e[i]=!(t[i] = r[a])
                    }) : function(e) {
                        return o(e, 0, r)
                    }) : o
                }
            },
            pseudos: {
                not: i(function(e) {
                    var t = [], n = [], r = T(e.replace(lt, "$1"));
                    return r[H] ? i(function(e, t, n, i) {
                        for (var o, a = r(e, null, i, []), s = e.length; s--;)(o = a[s]) 
                            && (e[s]=!(t[s] = o))
                    }) : function(e, i, o) {
                        return t[0] = e, r(t, null, o, n), !n.pop()
                    }
                }),
                has: i(function(e) {
                    return function(n) {
                        return t(e, n).length > 0
                    }
                }),
                contains: i(function(e) {
                    return function(t) {
                        return (t.textContent || t.innerText || C(t)).indexOf(e)>-1
                    }
                }),
                lang: i(function(e) {
                    return ht.test(e || "") || t.error("unsupported lang: " + e), e = e.replace(wt, xt).toLowerCase(), function(t) {
                        var n;
                        do 
                            if (n = L ? t.lang : t.getAttribute("xml:lang") || t.getAttribute("lang"))
                                return n = n.toLowerCase(), n === e || 0 === n.indexOf(e + "-");
                        while ((t = t.parentNode) && 1 === t.nodeType);
                        return !1
                    }
                }),
                target: function(t) {
                    var n = e.location && e.location.hash;
                    return n && n.slice(1) === t.id
                },
                root: function(e) {
                    return e === q
                },
                focus: function(e) {
                    return e === N.activeElement && (!N.hasFocus || N.hasFocus())&&!!(e.type || e.href||~e.tabIndex)
                },
                enabled: function(e) {
                    return e.disabled===!1
                },
                disabled: function(e) {
                    return e.disabled===!0
                },
                checked: function(e) {
                    var t = e.nodeName.toLowerCase();
                    return "input" === t&&!!e.checked || "option" === t&&!!e.selected
                },
                selected: function(e) {
                    return e.parentNode && e.parentNode.selectedIndex, e.selected===!0
                },
                empty: function(e) {
                    for (e = e.firstChild; e; e = e.nextSibling)
                        if (e.nodeType < 6)
                            return !1;
                    return !0
                },
                parent: function(e) {
                    return !x.pseudos.empty(e)
                },
                header: function(e) {
                    return mt.test(e.nodeName)
                },
                input: function(e) {
                    return gt.test(e.nodeName)
                },
                button: function(e) {
                    var t = e.nodeName.toLowerCase();
                    return "input" === t && "button" === e.type || "button" === t
                },
                text: function(e) {
                    var t;
                    return "input" === e.nodeName.toLowerCase() && "text" === e.type && (null == (t = e.getAttribute("type")) || "text" === t.toLowerCase())
                },
                first: c(function() {
                    return [0]
                }),
                last: c(function(e, t) {
                    return [t - 1]
                }),
                eq: c(function(e, t, n) {
                    return [0 > n ? n + t: n]
                }),
                even: c(function(e, t) {
                    for (var n = 0; t > n; n += 2)
                        e.push(n);
                    return e
                }),
                odd: c(function(e, t) {
                    for (var n = 1; t > n; n += 2)
                        e.push(n);
                    return e
                }),
                lt: c(function(e, t, n) {
                    for (var i = 0 > n ? n + t : n; --i >= 0;)
                        e.push(i);
                    return e
                }),
                gt: c(function(e, t, n) {
                    for (var i = 0 > n ? n + t : n; ++i < t;)
                        e.push(i);
                    return e
                })
            }
        }, x.pseudos.nth = x.pseudos.eq;
        for (_ in{
            radio: !0,
            checkbox: !0,
            file: !0,
            password: !0,
            image: !0
        })
            x.pseudos[_] = s(_);
        for (_ in{
            submit: !0,
            reset: !0
        })
            x.pseudos[_] = l(_);
        return d.prototype = x.filters = x.pseudos, x.setFilters = new d, S = t.tokenize = function(e, n) {
            var i, r, o, a, s, l, c, u = W[e + " "];
            if (u)
                return n ? 0 : u.slice(0);
            for (s = e, l = [], c = x.preFilter; s;) {
                (!i || (r = ct.exec(s))) && (r && (s = s.slice(r[0].length) || s), l.push(o = [])), i=!1, (r = ut.exec(s)) && (i = r.shift(), o.push({
                    value: i,
                    type: r[0].replace(lt, " ")
                }), s = s.slice(i.length));
                for (a in x.filter)
                    !(r = pt[a].exec(s)) || c[a]&&!(r = c[a](r)) || (i = r.shift(), o.push({
                        value: i,
                        type: a,
                        matches: r
                    }), s = s.slice(i.length));
                if (!i)
                    break
            }
            return n ? s.length : s ? t.error(e) : W(e, l).slice(0)
        }, T = t.compile = function(e, t) {
            var n, i = [], r = [], o = U[e + " "];
            if (!o) {
                for (t || (t = S(e)), n = t.length; n--;)
                    o = y(t[n]), o[H] ? i.push(o) : r.push(o);
                o = U(e, b(r, i)), o.selector = e
            }
            return o
        }, F = t.select = function(e, t, n, i) {
            var r, o, a, s, l, c = "function" == typeof e && e, d=!i && S(e = c.selector || e);
            if (n = n || [], 1 === d.length) {
                if (o = d[0] = d[0].slice(0), o.length > 2 && "ID" === (a = o[0]).type && w.getById && 9 === t.nodeType && L && x.relative[o[1].type]) {
                    if (t = (x.find.ID(a.matches[0].replace(wt, xt), t) || [])[0], !t)
                        return n;
                    c && (t = t.parentNode), e = e.slice(o.shift().value.length)
                }
                for (r = pt.needsContext.test(e) ? 0 : o.length; r--&&(a = o[r], !x.relative[s = a.type]);)
                    if ((l = x.find[s]) && (i = l(a.matches[0].replace(wt, xt), bt.test(o[0].type) && u(t.parentNode) || t))) {
                        if (o.splice(r, 1), e = i.length && f(o), !e)
                            return K.apply(n, i), n;
                            break
                    }
            }
            return (c || T(e, d))(i, t, !L, n, bt.test(e) && u(t.parentNode) || t), n
        }, w.sortStable = H.split("").sort(X).join("") === H, w.detectDuplicates=!!j, D(), w.sortDetached = r(function(e) {
            return 1 & e.compareDocumentPosition(N.createElement("div"))
        }), r(function(e) {
            return e.innerHTML = "<a href='#'></a>", "#" === e.firstChild.getAttribute("href")
        }) || o("type|href|height|width", function(e, t, n) {
            return n ? void 0 : e.getAttribute(t, "type" === t.toLowerCase() ? 1 : 2)
        }), w.attributes && r(function(e) {
            return e.innerHTML = "<input/>", e.firstChild.setAttribute("value", ""), "" === e.firstChild.getAttribute("value")
        }) || o("value", function(e, t, n) {
            return n || "input" !== e.nodeName.toLowerCase() ? void 0 : e.defaultValue
        }), r(function(e) {
            return null == e.getAttribute("disabled")
        }) || o(nt, function(e, t, n) {
            var i;
            return n ? void 0 : e[t]===!0 ? t.toLowerCase() : (i = e.getAttributeNode(t)) && i.specified ? i.value : null
        }), t
    }(e);
    K.find = rt, K.expr = rt.selectors, K.expr[":"] = K.expr.pseudos, K.unique = rt.uniqueSort, K.text = rt.getText, K.isXMLDoc = rt.isXML, K.contains = rt.contains;
    var ot = K.expr.match.needsContext, at = /^<(\w+)\s*\/?>(?:<\/\1>|)$/, st = /^.[^:#\[\.,]*$/;
    K.filter = function(e, t, n) {
        var i = t[0];
        return n && (e = ":not(" + e + ")"), 1 === t.length && 1 === i.nodeType ? K.find.matchesSelector(i, e) ? [i] : [] : K.find.matches(e, K.grep(t, function(e) {
            return 1 === e.nodeType
        }))
    }, K.fn.extend({
        find: function(e) {
            var t, n = this.length, i = [], r = this;
            if ("string" != typeof e)
                return this.pushStack(K(e).filter(function() {
                    for (t = 0; n > t; t++)
                        if (K.contains(r[t], this))
                            return !0
                        }));
            for (t = 0; n > t; t++)
                K.find(e, r[t], i);
            return i = this.pushStack(n > 1 ? K.unique(i) : i), i.selector = this.selector ? this.selector + " " + e : e, i
        },
        filter: function(e) {
            return this.pushStack(i(this, e || [], !1))
        },
        not: function(e) {
            return this.pushStack(i(this, e || [], !0))
        },
        is: function(e) {
            return !!i(this, "string" == typeof e && ot.test(e) ? K(e) : e || [], !1).length
        }
    });
    var lt, ct = /^(?:\s*(<[\w\W]+>)[^>]*|#([\w-]*))$/, ut = K.fn.init = function(e, t) {
        var n, i;
        if (!e)
            return this;
        if ("string" == typeof e) {
            if (n = "<" === e[0] && ">" === e[e.length - 1] && e.length >= 3 ? [null, e, null] : ct.exec(e), !n ||!n[1] && t)
                return !t || t.jquery ? (t || lt).find(e) : this.constructor(t).find(e);
            if (n[1]) {
                if (t = t instanceof K ? t[0] : t, K.merge(this, K.parseHTML(n[1], t && t.nodeType ? t.ownerDocument || t : G, !0)), at.test(n[1]) && K.isPlainObject(t))
                    for (n in t)
                        K.isFunction(this[n]) ? this[n](t[n]) : this.attr(n, t[n]);
                return this
            }
            return i = G.getElementById(n[2]), i && i.parentNode && (this.length = 1, this[0] = i), this.context = G, this.selector = e, this
        }
        return e.nodeType ? (this.context = this[0] = e, this.length = 1, this) : K.isFunction(e) ? "undefined" != typeof lt.ready ? lt.ready(e) : e(K) : (void 0 !== e.selector && (this.selector = e.selector, this.context = e.context), K.makeArray(e, this))
    };
    ut.prototype = K.fn, lt = K(G);
    var dt = /^(?:parents|prev(?:Until|All))/, ft = {
        children: !0,
        contents: !0,
        next: !0,
        prev: !0
    };
    K.extend({
        dir: function(e, t, n) {
            for (var i = [], r = void 0 !== n; (e = e[t]) && 9 !== e.nodeType;)
                if (1 === e.nodeType) {
                    if (r && K(e).is(n))
                        break;
                        i.push(e)
                }
            return i
        },
        sibling: function(e, t) {
            for (var n = []; e; e = e.nextSibling)
                1 === e.nodeType && e !== t && n.push(e);
            return n
        }
    }), K.fn.extend({
        has: function(e) {
            var t = K(e, this), n = t.length;
            return this.filter(function() {
                for (var e = 0; n > e; e++)
                    if (K.contains(this, t[e]))
                        return !0
            })
        },
        closest: function(e, t) {
            for (var n, i = 0, r = this.length, o = [], a = ot.test(e) || "string" != typeof e ? K(e, t || this.context) : 0; r > i; i++)
                for (n = this[i]; n && n !== t; n = n.parentNode)
                    if (n.nodeType < 11 && (a ? a.index(n)>-1 : 1 === n.nodeType && K.find.matchesSelector(n, e))) {
                        o.push(n);
                        break
                    }
            return this.pushStack(o.length > 1 ? K.unique(o) : o)
        },
        index: function(e) {
            return e ? "string" == typeof e ? X.call(K(e), this[0]) : X.call(this, e.jquery ? e[0] : e) : this[0] && this[0].parentNode ? this.first().prevAll().length : - 1
        },
        add: function(e, t) {
            return this.pushStack(K.unique(K.merge(this.get(), K(e, t))))
        },
        addBack: function(e) {
            return this.add(null == e ? this.prevObject : this.prevObject.filter(e))
        }
    }), K.each({
        parent: function(e) {
            var t = e.parentNode;
            return t && 11 !== t.nodeType ? t : null
        },
        parents: function(e) {
            return K.dir(e, "parentNode")
        },
        parentsUntil: function(e, t, n) {
            return K.dir(e, "parentNode", n)
        },
        next: function(e) {
            return r(e, "nextSibling")
        },
        prev: function(e) {
            return r(e, "previousSibling")
        },
        nextAll: function(e) {
            return K.dir(e, "nextSibling")
        },
        prevAll: function(e) {
            return K.dir(e, "previousSibling")
        },
        nextUntil: function(e, t, n) {
            return K.dir(e, "nextSibling", n)
        },
        prevUntil: function(e, t, n) {
            return K.dir(e, "previousSibling", n)
        },
        siblings: function(e) {
            return K.sibling((e.parentNode || {}).firstChild, e)
        },
        children: function(e) {
            return K.sibling(e.firstChild)
        },
        contents: function(e) {
            return e.contentDocument || K.merge([], e.childNodes)
        }
    }, function(e, t) {
        K.fn[e] = function(n, i) {
            var r = K.map(this, t, n);
            return "Until" !== e.slice( - 5) && (i = n), i && "string" == typeof i && (r = K.filter(i, r)), this.length > 1 && (ft[e] || K.unique(r), dt.test(e) && r.reverse()), this.pushStack(r)
        }
    });
    var ht = /\S+/g, pt = {};
    K.Callbacks = function(e) {
        e = "string" == typeof e ? pt[e] || o(e) : K.extend({}, e);
        var t, n, i, r, a, s, l = [], c=!e.once && [], u = function(o) {
            for (t = e.memory && o, n=!0, s = r || 0, r = 0, a = l.length, i=!0; l && a > s; s++)
                if (l[s].apply(o[0], o[1])===!1 && e.stopOnFalse) {
                    t=!1;
                    break
                }
            i=!1, l && (c ? c.length && u(c.shift()) : t ? l = [] : d.disable())
        }, d = {
            add: function() {
                if (l) {
                    var n = l.length;
                    !function o(t) {
                        K.each(t, function(t, n) {
                            var i = K.type(n);
                            "function" === i ? e.unique && d.has(n) || l.push(n) : n && n.length && "string" !== i && o(n)
                        })
                    }(arguments), i ? a = l.length : t && (r = n, u(t))
                }
                return this
            },
            remove: function() {
                return l && K.each(arguments, function(e, t) {
                    for (var n; (n = K.inArray(t, l, n))>-1;)
                        l.splice(n, 1), i && (a >= n && a--, s >= n && s--)
                }), this
            },
            has: function(e) {
                return e ? K.inArray(e, l)>-1 : !(!l ||!l.length)
            },
            empty: function() {
                return l = [], a = 0, this
            },
            disable: function() {
                return l = c = t = void 0, this
            },
            disabled: function() {
                return !l
            },
            lock: function() {
                return c = void 0, t || d.disable(), this
            },
            locked: function() {
                return !c
            },
            fireWith: function(e, t) {
                return !l || n&&!c || (t = t || [], t = [e, t.slice ? t.slice(): t], i ? c.push(t) : u(t)), this
            },
            fire: function() {
                return d.fireWith(this, arguments), this
            },
            fired: function() {
                return !!n
            }
        };
        return d
    }, K.extend({
        Deferred: function(e) {
            var t = [["resolve", "done", K.Callbacks("once memory"), "resolved"], ["reject", "fail", K.Callbacks("once memory"), "rejected"], ["notify", "progress", K.Callbacks("memory")]], n = "pending", i = {
                state: function() {
                    return n
                },
                always: function() {
                    return r.done(arguments).fail(arguments), this
                },
                then: function() {
                    var e = arguments;
                    return K.Deferred(function(n) {
                        K.each(t, function(t, o) {
                            var a = K.isFunction(e[t]) && e[t];
                            r[o[1]](function() {
                                var e = a && a.apply(this, arguments);
                                e && K.isFunction(e.promise) ? e.promise().done(n.resolve).fail(n.reject).progress(n.notify) : n[o[0] + "With"](this === i ? n.promise() : this, a ? [e] : arguments)
                            })
                        }), e = null
                    }).promise()
                },
                promise: function(e) {
                    return null != e ? K.extend(e, i) : i
                }
            }, r = {};
            return i.pipe = i.then, K.each(t, function(e, o) {
                var a = o[2], s = o[3];
                i[o[1]] = a.add, s && a.add(function() {
                    n = s
                }, t[1^e][2].disable, t[2][2].lock), r[o[0]] = function() {
                    return r[o[0] + "With"](this === r ? i : this, arguments), this
                }, r[o[0] + "With"] = a.fireWith
            }), i.promise(r), e && e.call(r, r), r
        },
        when: function(e) {
            var t, n, i, r = 0, o = B.call(arguments), a = o.length, s = 1 !== a || e && K.isFunction(e.promise) ? a: 0, l = 1 === s ? e: K.Deferred(), c = function(e, n, i) {
                return function(r) {
                    n[e] = this, i[e] = arguments.length > 1 ? B.call(arguments) : r, i === t ? l.notifyWith(n, i) : --s || l.resolveWith(n, i)
                }
            };
            if (a > 1)
                for (t = new Array(a), n = new Array(a), i = new Array(a); a > r; r++)
                    o[r] && K.isFunction(o[r].promise) ? o[r].promise().done(c(r, i, o)).fail(l.reject).progress(c(r, n, t)) : --s;
            return s || l.resolveWith(i, o), l.promise()
        }
    });
    var gt;
    K.fn.ready = function(e) {
        return K.ready.promise().done(e), this
    }, K.extend({
        isReady: !1,
        readyWait: 1,
        holdReady: function(e) {
            e ? K.readyWait++ : K.ready(!0)
        },
        ready: function(e) {
            (e===!0?--K.readyWait : K.isReady) || (K.isReady=!0, e!==!0&&--K.readyWait > 0 || (gt.resolveWith(G, [K]), K.fn.triggerHandler && (K(G).triggerHandler("ready"), K(G).off("ready"))))
        }
    }), K.ready.promise = function(t) {
        return gt || (gt = K.Deferred(), "complete" === G.readyState ? setTimeout(K.ready) : (G.addEventListener("DOMContentLoaded", a, !1), e.addEventListener("load", a, !1))), gt.promise(t)
    }, K.ready.promise();
    var mt = K.access = function(e, t, n, i, r, o, a) {
        var s = 0, l = e.length, c = null == n;
        if ("object" === K.type(n)) {
            r=!0;
            for (s in n)
                K.access(e, t, s, n[s], !0, o, a)
        } else if (void 0 !== i && (r=!0, K.isFunction(i) || (a=!0), c && (a ? (t.call(e, i), t = null) : (c = t, t = function(e, t, n) {
            return c.call(K(e), n)
        })), t))for (; l > s; s++)
            t(e[s], n, a ? i : i.call(e[s], s, t(e[s], n)));
        return r ? e : c ? t.call(e) : l ? t(e[0], n) : o
    };
    K.acceptData = function(e) {
        return 1 === e.nodeType || 9 === e.nodeType||!+e.nodeType
    }, s.uid = 1, s.accepts = K.acceptData, s.prototype = {
        key: function(e) {
            if (!s.accepts(e))
                return 0;
            var t = {}, n = e[this.expando];
            if (!n) {
                n = s.uid++;
                try {
                    t[this.expando] = {
                        value: n
                    }, Object.defineProperties(e, t)
                } catch (i) {
                    t[this.expando] = n, K.extend(e, t)
                }
            }
            return this.cache[n] || (this.cache[n] = {}), n
        },
        set: function(e, t, n) {
            var i, r = this.key(e), o = this.cache[r];
            if ("string" == typeof t)
                o[t] = n;
            else if (K.isEmptyObject(o))
                K.extend(this.cache[r], t);
            else 
                for (i in t)
                    o[i] = t[i];
            return o
        },
        get: function(e, t) {
            var n = this.cache[this.key(e)];
            return void 0 === t ? n : n[t]
        },
        access: function(e, t, n) {
            var i;
            return void 0 === t || t && "string" == typeof t && void 0 === n ? (i = this.get(e, t), void 0 !== i ? i : this.get(e, K.camelCase(t))) : (this.set(e, t, n), void 0 !== n ? n : t)
        },
        remove: function(e, t) {
            var n, i, r, o = this.key(e), a = this.cache[o];
            if (void 0 === t)
                this.cache[o] = {};
            else {
                K.isArray(t) ? i = t.concat(t.map(K.camelCase)) : (r = K.camelCase(t), t in a ? i = [t, r] : (i = r, i = i in a ? [i] : i.match(ht) || [])), n = i.length;
                for (; n--;)
                    delete a[i[n]]
            }
        },
        hasData: function(e) {
            return !K.isEmptyObject(this.cache[e[this.expando]] || {})
        },
        discard: function(e) {
            e[this.expando] && delete this.cache[e[this.expando]]
        }
    };
    var vt = new s, yt = new s, bt = /^(?:\{[\w\W]*\}|\[[\w\W]*\])$/, _t = /([A-Z])/g;
    K.extend({
        hasData: function(e) {
            return yt.hasData(e) || vt.hasData(e)
        },
        data: function(e, t, n) {
            return yt.access(e, t, n)
        },
        removeData: function(e, t) {
            yt.remove(e, t)
        },
        _data: function(e, t, n) {
            return vt.access(e, t, n)
        },
        _removeData: function(e, t) {
            vt.remove(e, t)
        }
    }), K.fn.extend({
        data: function(e, t) {
            var n, i, r, o = this[0], a = o && o.attributes;
            if (void 0 === e) {
                if (this.length && (r = yt.get(o), 1 === o.nodeType&&!vt.get(o, "hasDataAttrs"))) {
                    for (n = a.length; n--;)
                        a[n] && (i = a[n].name, 0 === i.indexOf("data-") && (i = K.camelCase(i.slice(5)), l(o, i, r[i])));
                    vt.set(o, "hasDataAttrs", !0)
                }
                return r
            }
            return "object" == typeof e ? this.each(function() {
                yt.set(this, e)
            }) : mt(this, function(t) {
                var n, i = K.camelCase(e);
                if (o && void 0 === t) {
                    if (n = yt.get(o, e), void 0 !== n)
                        return n;
                    if (n = yt.get(o, i), void 0 !== n)
                        return n;
                    if (n = l(o, i, void 0), void 0 !== n)
                        return n
                } else 
                    this.each(function() {
                        var n = yt.get(this, i);
                        yt.set(this, i, t), - 1 !== e.indexOf("-") && void 0 !== n && yt.set(this, e, t)
                    })
            }, null, t, arguments.length > 1, null, !0)
        },
        removeData: function(e) {
            return this.each(function() {
                yt.remove(this, e)
            })
        }
    }), K.extend({
        queue: function(e, t, n) {
            var i;
            return e ? (t = (t || "fx") + "queue", i = vt.get(e, t), n && (!i || K.isArray(n) ? i = vt.access(e, t, K.makeArray(n)) : i.push(n)), i || []) : void 0
        },
        dequeue: function(e, t) {
            t = t || "fx";
            var n = K.queue(e, t), i = n.length, r = n.shift(), o = K._queueHooks(e, t), a = function() {
                K.dequeue(e, t)
            };
            "inprogress" === r && (r = n.shift(), i--), r && ("fx" === t && n.unshift("inprogress"), delete o.stop, r.call(e, a, o)), !i && o && o.empty.fire()
        },
        _queueHooks: function(e, t) {
            var n = t + "queueHooks";
            return vt.get(e, n) || vt.access(e, n, {
                empty: K.Callbacks("once memory").add(function() {
                    vt.remove(e, [t + "queue", n])
                })
            })
        }
    }), K.fn.extend({
        queue: function(e, t) {
            var n = 2;
            return "string" != typeof e && (t = e, e = "fx", n--), arguments.length < n ? K.queue(this[0], e) : void 0 === t ? this : this.each(function() {
                var n = K.queue(this, e, t);
                K._queueHooks(this, e), "fx" === e && "inprogress" !== n[0] && K.dequeue(this, e)
            })
        },
        dequeue: function(e) {
            return this.each(function() {
                K.dequeue(this, e)
            })
        },
        clearQueue: function(e) {
            return this.queue(e || "fx", [])
        },
        promise: function(e, t) {
            var n, i = 1, r = K.Deferred(), o = this, a = this.length, s = function() {
                --i || r.resolveWith(o, [o])
            };
            for ("string" != typeof e && (t = e, e = void 0), e = e || "fx"; a--;)
                n = vt.get(o[a], e + "queueHooks"), n && n.empty && (i++, n.empty.add(s));
            return s(), r.promise(t)
        }
    });
    var wt = /[+-]?(?:\d*\.|)\d+(?:[eE][+-]?\d+|)/.source, xt = ["Top", "Right", "Bottom", "Left"], Ct = function(e, t) {
        return e = t || e, "none" === K.css(e, "display") ||!K.contains(e.ownerDocument, e)
    }, kt = /^(?:checkbox|radio)$/i;
    !function() {
        var e = G.createDocumentFragment(), t = e.appendChild(G.createElement("div")), n = G.createElement("input");
        n.setAttribute("type", "radio"), n.setAttribute("checked", "checked"), n.setAttribute("name", "t"), t.appendChild(n), Z.checkClone = t.cloneNode(!0).cloneNode(!0).lastChild.checked, t.innerHTML = "<textarea>x</textarea>", Z.noCloneChecked=!!t.cloneNode(!0).lastChild.defaultValue
    }();
    var St = "undefined";
    Z.focusinBubbles = "onfocusin"in e;
    var Tt = /^key/, Ft = /^(?:mouse|pointer|contextmenu)|click/, At = /^(?:focusinfocus|focusoutblur)$/, Et = /^([^.]*)(?:\.(.+)|)$/;
    K.event = {
        global: {},
        add: function(e, t, n, i, r) {
            var o, a, s, l, c, u, d, f, h, p, g, m = vt.get(e);
            if (m)
                for (n.handler && (o = n, n = o.handler, r = o.selector), n.guid || (n.guid = K.guid++), (l = m.events) || (l = m.events = {}), (a = m.handle) || (a = m.handle = function(t) {
                    return typeof K !== St && K.event.triggered !== t.type ? K.event.dispatch.apply(e, arguments) : void 0
                }), t = (t || "").match(ht) || [""], c = t.length; c--;)
                    s = Et.exec(t[c]) || [], h = g = s[1], p = (s[2] || "").split(".").sort(), h && (d = K.event.special[h] || {}, h = (r ? d.delegateType : d.bindType) || h, d = K.event.special[h] || {}, u = K.extend({
                        type: h,
                        origType: g,
                        data: i,
                        handler: n,
                        guid: n.guid,
                        selector: r,
                        needsContext: r && K.expr.match.needsContext.test(r),
                        namespace: p.join(".")
                    }, o), (f = l[h]) || (f = l[h] = [], f.delegateCount = 0, d.setup && d.setup.call(e, i, p, a)!==!1 || e.addEventListener && e.addEventListener(h, a, !1)), d.add && (d.add.call(e, u), u.handler.guid || (u.handler.guid = n.guid)), r ? f.splice(f.delegateCount++, 0, u) : f.push(u), K.event.global[h]=!0)
        },
        remove: function(e, t, n, i, r) {
            var o, a, s, l, c, u, d, f, h, p, g, m = vt.hasData(e) && vt.get(e);
            if (m && (l = m.events)) {
                for (t = (t || "").match(ht) || [""], c = t.length; c--;)
                    if (s = Et.exec(t[c]) || [], h = g = s[1], p = (s[2] || "").split(".").sort(), h) {
                        for (d = K.event.special[h] || {}, h = (i ? d.delegateType : d.bindType) || h, f = l[h] || [], s = s[2] && new RegExp("(^|\\.)" + p.join("\\.(?:.*\\.|)") + "(\\.|$)"), a = o = f.length; o--;)
                            u = f[o], !r && g !== u.origType || n && n.guid !== u.guid || s&&!s.test(u.namespace) || i && i !== u.selector && ("**" !== i ||!u.selector) || (f.splice(o, 1), u.selector && f.delegateCount--, d.remove && d.remove.call(e, u));
                            a&&!f.length && (d.teardown && d.teardown.call(e, p, m.handle)!==!1 || K.removeEvent(e, h, m.handle), delete l[h])
                    } else 
                        for (h in l)
                            K.event.remove(e, h + t[c], n, i, !0);
                K.isEmptyObject(l) && (delete m.handle, vt.remove(e, "events"))
            }
        },
        trigger: function(t, n, i, r) {
            var o, a, s, l, c, u, d, f = [i || G], h = Y.call(t, "type") ? t.type: t, p = Y.call(t, "namespace") ? t.namespace.split("."): [];
            if (a = s = i = i || G, 3 !== i.nodeType && 8 !== i.nodeType&&!At.test(h + K.event.triggered) && (h.indexOf(".") >= 0 && (p = h.split("."), h = p.shift(), p.sort()), c = h.indexOf(":") < 0 && "on" + h, t = t[K.expando] ? t : new K.Event(h, "object" == typeof t && t), t.isTrigger = r ? 2 : 3, t.namespace = p.join("."), t.namespace_re = t.namespace ? new RegExp("(^|\\.)" + p.join("\\.(?:.*\\.|)") + "(\\.|$)") : null, t.result = void 0, t.target || (t.target = i), n = null == n ? [t] : K.makeArray(n, [t]), d = K.event.special[h] || {}, r ||!d.trigger || d.trigger.apply(i, n)!==!1)
                ) {
                if (!r&&!d.noBubble&&!K.isWindow(i)) {
                    for (l = d.delegateType || h, At.test(l + h) || (a = a.parentNode); a; a = a.parentNode)
                        f.push(a), s = a;
                    s === (i.ownerDocument || G) && f.push(s.defaultView || s.parentWindow || e)
                }
                for (o = 0; (a = f[o++])&&!t.isPropagationStopped();)
                    t.type = o > 1 ? l : d.bindType || h, u = (vt.get(a, "events") || {})[t.type] && vt.get(a, "handle"), u && u.apply(a, n), u = c && a[c], u && u.apply && K.acceptData(a) && (t.result = u.apply(a, n), t.result===!1 && t.preventDefault());
                return t.type = h, r || t.isDefaultPrevented() || d._default && d._default.apply(f.pop(), n)!==!1 ||!K.acceptData(i) || c && K.isFunction(i[h])&&!K.isWindow(i) && (s = i[c], s && (i[c] = null), K.event.triggered = h, i[h](), K.event.triggered = void 0, s && (i[c] = s)), t.result
            }
        },
        dispatch: function(e) {
            e = K.event.fix(e);
            var t, n, i, r, o, a = [], s = B.call(arguments), l = (vt.get(this, "events") || {})[e.type] || [], c = K.event.special[e.type] || {};
            if (s[0] = e, e.delegateTarget = this, !c.preDispatch || c.preDispatch.call(this, e)!==!1) {
                for (a = K.event.handlers.call(this, e, l), t = 0; (r = a[t++])&&!e.isPropagationStopped();)
                    for (e.currentTarget = r.elem, n = 0; (o = r.handlers[n++])&&!e.isImmediatePropagationStopped();)(!e.namespace_re || e.namespace_re.test(o.namespace)
                        ) && (e.handleObj = o, e.data = o.data, i = ((K.event.special[o.origType] || {}).handle || o.handler).apply(r.elem, s), void 0 !== i && (e.result = i)===!1 && (e.preventDefault(), e.stopPropagation()));
                return c.postDispatch && c.postDispatch.call(this, e), e.result
            }
        },
        handlers: function(e, t) {
            var n, i, r, o, a = [], s = t.delegateCount, l = e.target;
            if (s && l.nodeType && (!e.button || "click" !== e.type))
                for (; l !== this; l = l.parentNode || this)
                    if (l.disabled!==!0 || "click" !== e.type) {
                        for (i = [], n = 0; s > n; n++)
                            o = t[n], r = o.selector + " ", void 0 === i[r] && (i[r] = o.needsContext ? K(r, this).index(l) >= 0 : K.find(r, this, null, [l]).length), i[r] && i.push(o);
                            i.length && a.push({
                                elem: l,
                                handlers: i
                            })
                    }
            return s < t.length && a.push({
                elem: this,
                handlers: t.slice(s)
            }), a
        },
        props: "altKey bubbles cancelable ctrlKey currentTarget eventPhase metaKey relatedTarget shiftKey target timeStamp view which".split(" "),
        fixHooks: {},
        keyHooks: {
            props: "char charCode key keyCode".split(" "),
            filter: function(e, t) {
                return null == e.which && (e.which = null != t.charCode ? t.charCode : t.keyCode), e
            }
        },
        mouseHooks: {
            props: "button buttons clientX clientY offsetX offsetY pageX pageY screenX screenY toElement".split(" "),
            filter: function(e, t) {
                var n, i, r, o = t.button;
                return null == e.pageX && null != t.clientX && (n = e.target.ownerDocument || G, i = n.documentElement, r = n.body, e.pageX = t.clientX + (i && i.scrollLeft || r && r.scrollLeft || 0) - (i && i.clientLeft || r && r.clientLeft || 0), e.pageY = t.clientY + (i && i.scrollTop || r && r.scrollTop || 0) - (i && i.clientTop || r && r.clientTop || 0)), e.which || void 0 === o || (e.which = 1 & o ? 1 : 2 & o ? 3 : 4 & o ? 2 : 0), e
            }
        },
        fix: function(e) {
            if (e[K.expando])
                return e;
            var t, n, i, r = e.type, o = e, a = this.fixHooks[r];
            for (a || (this.fixHooks[r] = a = Ft.test(r) ? this.mouseHooks : Tt.test(r) ? this.keyHooks : {}), i = a.props ? this.props.concat(a.props) : this.props, e = new K.Event(o), t = i.length; t--;)
                n = i[t], e[n] = o[n];
            return e.target || (e.target = G), 3 === e.target.nodeType && (e.target = e.target.parentNode), a.filter ? a.filter(e, o) : e
        },
        special: {
            load: {
                noBubble: !0
            },
            focus: {
                trigger: function() {
                    return this !== d() && this.focus ? (this.focus(), !1) : void 0
                },
                delegateType: "focusin"
            },
            blur: {
                trigger: function() {
                    return this === d() && this.blur ? (this.blur(), !1) : void 0
                },
                delegateType: "focusout"
            },
            click: {
                trigger: function() {
                    return "checkbox" === this.type && this.click && K.nodeName(this, "input") ? (this.click(), !1) : void 0
                },
                _default: function(e) {
                    return K.nodeName(e.target, "a")
                }
            },
            beforeunload: {
                postDispatch: function(e) {
                    void 0 !== e.result && e.originalEvent && (e.originalEvent.returnValue = e.result)
                }
            }
        },
        simulate: function(e, t, n, i) {
            var r = K.extend(new K.Event, n, {
                type: e,
                isSimulated: !0,
                originalEvent: {}
            });
            i ? K.event.trigger(r, null, t) : K.event.dispatch.call(t, r), r.isDefaultPrevented() && n.preventDefault()
        }
    }, K.removeEvent = function(e, t, n) {
        e.removeEventListener && e.removeEventListener(t, n, !1)
    }, K.Event = function(e, t) {
        return this instanceof K.Event ? (e && e.type ? (this.originalEvent = e, this.type = e.type, this.isDefaultPrevented = e.defaultPrevented || void 0 === e.defaultPrevented && e.returnValue===!1 ? c : u) : this.type = e, t && K.extend(this, t), this.timeStamp = e && e.timeStamp || K.now(), void(this[K.expando]=!0)) : new K.Event(e, t)
    }, K.Event.prototype = {
        isDefaultPrevented: u,
        isPropagationStopped: u,
        isImmediatePropagationStopped: u,
        preventDefault: function() {
            var e = this.originalEvent;
            this.isDefaultPrevented = c, e && e.preventDefault && e.preventDefault()
        },
        stopPropagation: function() {
            var e = this.originalEvent;
            this.isPropagationStopped = c, e && e.stopPropagation && e.stopPropagation()
        },
        stopImmediatePropagation: function() {
            var e = this.originalEvent;
            this.isImmediatePropagationStopped = c, e && e.stopImmediatePropagation && e.stopImmediatePropagation(), this.stopPropagation()
        }
    }, K.each({
        mouseenter: "mouseover",
        mouseleave: "mouseout",
        pointerenter: "pointerover",
        pointerleave: "pointerout"
    }, function(e, t) {
        K.event.special[e] = {
            delegateType: t,
            bindType: t,
            handle: function(e) {
                var n, i = this, r = e.relatedTarget, o = e.handleObj;
                return (!r || r !== i&&!K.contains(i, r)) && (e.type = o.origType, n = o.handler.apply(this, arguments), e.type = t), n
            }
        }
    }), Z.focusinBubbles || K.each({
        focus: "focusin",
        blur: "focusout"
    }, function(e, t) {
        var n = function(e) {
            K.event.simulate(t, e.target, K.event.fix(e), !0)
        };
        K.event.special[t] = {
            setup: function() {
                var i = this.ownerDocument || this, r = vt.access(i, t);
                r || i.addEventListener(e, n, !0), vt.access(i, t, (r || 0) + 1)
            },
            teardown: function() {
                var i = this.ownerDocument || this, r = vt.access(i, t) - 1;
                r ? vt.access(i, t, r) : (i.removeEventListener(e, n, !0), vt.remove(i, t))
            }
        }
    }), K.fn.extend({
        on: function(e, t, n, i, r) {
            var o, a;
            if ("object" == typeof e) {
                "string" != typeof t && (n = n || t, t = void 0);
                for (a in e)
                    this.on(a, t, n, e[a], r);
                return this
            }
            if (null == n && null == i ? (i = t, n = t = void 0) : null == i && ("string" == typeof t ? (i = n, n = void 0) : (i = n, n = t, t = void 0)), i===!1)
                i = u;
            else if (!i)
                return this;
            return 1 === r && (o = i, i = function(e) {
                return K().off(e), o.apply(this, arguments)
            }, i.guid = o.guid || (o.guid = K.guid++)), this.each(function() {
                K.event.add(this, e, i, n, t)
            })
        },
        one: function(e, t, n, i) {
            return this.on(e, t, n, i, 1)
        },
        off: function(e, t, n) {
            var i, r;
            if (e && e.preventDefault && e.handleObj)
                return i = e.handleObj, K(e.delegateTarget).off(i.namespace ? i.origType + "." + i.namespace : i.origType, i.selector, i.handler), this;
            if ("object" == typeof e) {
                for (r in e)
                    this.off(r, t, e[r]);
                return this
            }
            return (t===!1 || "function" == typeof t) && (n = t, t = void 0), n===!1 && (n = u), this.each(function() {
                K.event.remove(this, e, n, t)
            })
        },
        trigger: function(e, t) {
            return this.each(function() {
                K.event.trigger(e, t, this)
            })
        },
        triggerHandler: function(e, t) {
            var n = this[0];
            return n ? K.event.trigger(e, t, n, !0) : void 0
        }
    });
    var jt = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\w:]+)[^>]*)\/>/gi, Dt = /<([\w:]+)/, Nt = /<|&#?\w+;/, qt = /<(?:script|style|link)/i, Lt = /checked\s*(?:[^=]|=\s*.checked.)/i, It = /^$|\/(?:java|ecma)script/i, Ot = /^true\/(.*)/, Mt = /^\s*<!(?:\[CDATA\[|--)|(?:\]\]|--)>\s*$/g, Rt = {
        option: [1, "<select multiple='multiple'>", "</select>"],
        thead: [1, "<table>", "</table>"],
        col: [2, "<table><colgroup>", "</colgroup></table>"],
        tr: [2, "<table><tbody>", "</tbody></table>"],
        td: [3, "<table><tbody><tr>", "</tr></tbody></table>"],
        _default: [0, "", ""]
    };
    Rt.optgroup = Rt.option, Rt.tbody = Rt.tfoot = Rt.colgroup = Rt.caption = Rt.thead, Rt.th = Rt.td, K.extend({
        clone: function(e, t, n) {
            var i, r, o, a, s = e.cloneNode(!0), l = K.contains(e.ownerDocument, e);
            if (!(Z.noCloneChecked || 1 !== e.nodeType && 11 !== e.nodeType || K.isXMLDoc(e)))
                for (a = v(s), o = v(e), i = 0, r = o.length; r > i; i++)
                    y(o[i], a[i]);
            if (t)
                if (n)
                    for (o = o || v(e), a = a || v(s), i = 0, r = o.length; r > i; i++)
                        m(o[i], a[i]);
                else 
                    m(e, s);
            return a = v(s, "script"), a.length > 0 && g(a, !l && v(e, "script")), s
        },
        buildFragment: function(e, t, n, i) {
            for (var r, o, a, s, l, c, u = t.createDocumentFragment(), d = [], f = 0, h = e.length; h > f; f++)
                if (r = e[f], r || 0 === r)
                    if ("object" === K.type(r))
                        K.merge(d, r.nodeType ? [r] : r);
                    else if (Nt.test(r)) {
                        for (o = o || u.appendChild(t.createElement("div")), a = (Dt.exec(r) || ["", ""])[1].toLowerCase(), s = Rt[a] || Rt._default, o.innerHTML = s[1] + r.replace(jt, "<$1></$2>") + s[2], c = s[0]; c--;)
                            o = o.lastChild;
                            K.merge(d, o.childNodes), o = u.firstChild, o.textContent = ""
                    } else 
                        d.push(t.createTextNode(r));
            for (u.textContent = "", f = 0; r = d[f++];)
                if ((!i||-1 === K.inArray(r, i)) && (l = K.contains(r.ownerDocument, r), o = v(u.appendChild(r), "script"), l && g(o), n))
                    for (c = 0; r = o[c++];)
                        It.test(r.type || "") && n.push(r);
            return u
        },
        cleanData: function(e) {
            for (var t, n, i, r, o = K.event.special, a = 0; void 0 !== (n = e[a]); a++) {
                if (K.acceptData(n) && (r = n[vt.expando], r && (t = vt.cache[r]))) {
                    if (t.events)
                        for (i in t.events)
                            o[i] ? K.event.remove(n, i) : K.removeEvent(n, i, t.handle);
                    vt.cache[r] && delete vt.cache[r]
                }
                delete yt.cache[n[yt.expando]]
            }
        }
    }), K.fn.extend({
        text: function(e) {
            return mt(this, function(e) {
                return void 0 === e ? K.text(this) : this.empty().each(function() {
                    (1 === this.nodeType || 11 === this.nodeType || 9 === this.nodeType) && (this.textContent = e)
                })
            }, null, e, arguments.length)
        },
        append: function() {
            return this.domManip(arguments, function(e) {
                if (1 === this.nodeType || 11 === this.nodeType || 9 === this.nodeType) {
                    var t = f(this, e);
                    t.appendChild(e)
                }
            })
        },
        prepend: function() {
            return this.domManip(arguments, function(e) {
                if (1 === this.nodeType || 11 === this.nodeType || 9 === this.nodeType) {
                    var t = f(this, e);
                    t.insertBefore(e, t.firstChild)
                }
            })
        },
        before: function() {
            return this.domManip(arguments, function(e) {
                this.parentNode && this.parentNode.insertBefore(e, this)
            })
        },
        after: function() {
            return this.domManip(arguments, function(e) {
                this.parentNode && this.parentNode.insertBefore(e, this.nextSibling)
            })
        },
        remove: function(e, t) {
            for (var n, i = e ? K.filter(e, this) : this, r = 0; null != (n = i[r]); r++)
                t || 1 !== n.nodeType || K.cleanData(v(n)), n.parentNode && (t && K.contains(n.ownerDocument, n) && g(v(n, "script")), n.parentNode.removeChild(n));
            return this
        },
        empty: function() {
            for (var e, t = 0; null != (e = this[t]); t++)
                1 === e.nodeType && (K.cleanData(v(e, !1)), e.textContent = "");
            return this
        },
        clone: function(e, t) {
            return e = null == e?!1 : e, t = null == t ? e : t, this.map(function() {
                return K.clone(this, e, t)
            })
        },
        html: function(e) {
            return mt(this, function(e) {
                var t = this[0] || {}, n = 0, i = this.length;
                if (void 0 === e && 1 === t.nodeType)
                    return t.innerHTML;
                if ("string" == typeof e&&!qt.test(e)&&!Rt[(Dt.exec(e) || ["", ""])[1].toLowerCase()]) {
                    e = e.replace(jt, "<$1></$2>");
                    try {
                        for (; i > n; n++)
                            t = this[n] || {}, 1 === t.nodeType && (K.cleanData(v(t, !1)), t.innerHTML = e);
                        t = 0
                    } catch (r) {}
                }
                t && this.empty().append(e)
            }, null, e, arguments.length)
        },
        replaceWith: function() {
            var e = arguments[0];
            return this.domManip(arguments, function(t) {
                e = this.parentNode, K.cleanData(v(this)), e && e.replaceChild(t, this)
            }), e && (e.length || e.nodeType) ? this : this.remove()
        },
        detach: function(e) {
            return this.remove(e, !0)
        },
        domManip: function(e, t) {
            e = W.apply([], e);
            var n, i, r, o, a, s, l = 0, c = this.length, u = this, d = c - 1, f = e[0], g = K.isFunction(f);
            if (g || c > 1 && "string" == typeof f&&!Z.checkClone && Lt.test(f))
                return this.each(function(n) {
                    var i = u.eq(n);
                    g && (e[0] = f.call(this, n, i.html())), i.domManip(e, t)
                });
            if (c && (n = K.buildFragment(e, this[0].ownerDocument, !1, this), i = n.firstChild, 1 === n.childNodes.length && (n = i), i)) {
                for (r = K.map(v(n, "script"), h), o = r.length; c > l; l++)
                    a = n, l !== d && (a = K.clone(a, !0, !0), o && K.merge(r, v(a, "script"))), t.call(this[l], a, l);
                if (o)
                    for (s = r[r.length - 1].ownerDocument, K.map(r, p), l = 0; o > l; l++)
                        a = r[l], It.test(a.type || "")&&!vt.access(a, "globalEval") && K.contains(s, a) && (a.src ? K._evalUrl && K._evalUrl(a.src) : K.globalEval(a.textContent.replace(Mt, "")))
            }
            return this
        }
    }), K.each({
        appendTo: "append",
        prependTo: "prepend",
        insertBefore: "before",
        insertAfter: "after",
        replaceAll: "replaceWith"
    }, function(e, t) {
        K.fn[e] = function(e) {
            for (var n, i = [], r = K(e), o = r.length - 1, a = 0; o >= a; a++)
                n = a === o ? this : this.clone(!0), K(r[a])[t](n), U.apply(i, n.get());
            return this.pushStack(i)
        }
    });
    var Ht, Pt = {}, zt = /^margin/, $t = new RegExp("^(" + wt + ")(?!px)[a-z%]+$", "i"), Bt = function(e) {
        return e.ownerDocument.defaultView.getComputedStyle(e, null)
    };
    !function() {
        function t() {
            a.style.cssText = "-webkit-box-sizing:border-box;-moz-box-sizing:border-box;box-sizing:border-box;display:block;margin-top:1%;top:1%;border:1px;padding:1px;width:4px;position:absolute", a.innerHTML = "", r.appendChild(o);
            var t = e.getComputedStyle(a, null);
            n = "1%" !== t.top, i = "4px" === t.width, r.removeChild(o)
        }
        var n, i, r = G.documentElement, o = G.createElement("div"), a = G.createElement("div");
        a.style && (a.style.backgroundClip = "content-box", a.cloneNode(!0).style.backgroundClip = "", Z.clearCloneStyle = "content-box" === a.style.backgroundClip, o.style.cssText = "border:0;width:0;height:0;top:0;left:-9999px;margin-top:1px;position:absolute", o.appendChild(a), e.getComputedStyle && K.extend(Z, {
            pixelPosition: function() {
                return t(), n
            },
            boxSizingReliable: function() {
                return null == i && t(), i
            },
            reliableMarginRight: function() {
                var t, n = a.appendChild(G.createElement("div"));
                return n.style.cssText = a.style.cssText = "-webkit-box-sizing:content-box;-moz-box-sizing:content-box;box-sizing:content-box;display:block;margin:0;border:0;padding:0", n.style.marginRight = n.style.width = "0", a.style.width = "1px", r.appendChild(o), t=!parseFloat(e.getComputedStyle(n, null).marginRight), r.removeChild(o), t
            }
        }))
    }(), K.swap = function(e, t, n, i) {
        var r, o, a = {};
        for (o in t)
            a[o] = e.style[o], e.style[o] = t[o];
        r = n.apply(e, i || []);
        for (o in t)
            e.style[o] = a[o];
        return r
    };
    var Wt = /^(none|table(?!-c[ea]).+)/, Ut = new RegExp("^(" + wt + ")(.*)$", "i"), Xt = new RegExp("^([+-])=(" + wt + ")", "i"), Qt = {
        position: "absolute",
        visibility: "hidden",
        display: "block"
    }, Vt = {
        letterSpacing: "0",
        fontWeight: "400"
    }, Yt = ["Webkit", "O", "Moz", "ms"];
    K.extend({
        cssHooks: {
            opacity: {
                get: function(e, t) {
                    if (t) {
                        var n = w(e, "opacity");
                        return "" === n ? "1" : n
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
            "float": "cssFloat"
        },
        style: function(e, t, n, i) {
            if (e && 3 !== e.nodeType && 8 !== e.nodeType && e.style) {
                var r, o, a, s = K.camelCase(t), l = e.style;
                return t = K.cssProps[s] || (K.cssProps[s] = C(l, s)), a = K.cssHooks[t] || K.cssHooks[s], void 0 === n ? a && "get"in a && void 0 !== (r = a.get(e, !1, i)) ? r : l[t] : (o = typeof n, "string" === o && (r = Xt.exec(n)) && (n = (r[1] + 1) * r[2] + parseFloat(K.css(e, t)), o = "number"), void(null != n && n === n && ("number" !== o || K.cssNumber[s] || (n += "px"), Z.clearCloneStyle || "" !== n || 0 !== t.indexOf("background") || (l[t] = "inherit"), a && "set"in a && void 0 === (n = a.set(e, n, i)) || (l[t] = n))))
            }
        },
        css: function(e, t, n, i) {
            var r, o, a, s = K.camelCase(t);
            return t = K.cssProps[s] || (K.cssProps[s] = C(e.style, s)), a = K.cssHooks[t] || K.cssHooks[s], a && "get"in a && (r = a.get(e, !0, n)), void 0 === r && (r = w(e, t, i)), "normal" === r && t in Vt && (r = Vt[t]), "" === n || n ? (o = parseFloat(r), n===!0 || K.isNumeric(o) ? o || 0 : r) : r
        }
    }), K.each(["height", "width"], function(e, t) {
        K.cssHooks[t] = {
            get: function(e, n, i) {
                return n ? Wt.test(K.css(e, "display")) && 0 === e.offsetWidth ? K.swap(e, Qt, function() {
                    return T(e, t, i)
                }) : T(e, t, i) : void 0
            },
            set: function(e, n, i) {
                var r = i && Bt(e);
                return k(e, n, i ? S(e, t, i, "border-box" === K.css(e, "boxSizing", !1, r), r) : 0)
            }
        }
    }), K.cssHooks.marginRight = x(Z.reliableMarginRight, function(e, t) {
        return t ? K.swap(e, {
            display: "inline-block"
        }, w, [e, "marginRight"]) : void 0
    }), K.each({
        margin: "",
        padding: "",
        border: "Width"
    }, function(e, t) {
        K.cssHooks[e + t] = {
            expand: function(n) {
                for (var i = 0, r = {}, o = "string" == typeof n ? n.split(" ") : [n]; 4 > i; i++)
                    r[e + xt[i] + t] = o[i] || o[i - 2] || o[0];
                return r
            }
        }, zt.test(e) || (K.cssHooks[e + t].set = k)
    }), K.fn.extend({
        css: function(e, t) {
            return mt(this, function(e, t, n) {
                var i, r, o = {}, a = 0;
                if (K.isArray(t)) {
                    for (i = Bt(e), r = t.length; r > a; a++)
                        o[t[a]] = K.css(e, t[a], !1, i);
                    return o
                }
                return void 0 !== n ? K.style(e, t, n) : K.css(e, t)
            }, e, t, arguments.length > 1)
        },
        show: function() {
            return F(this, !0)
        },
        hide: function() {
            return F(this)
        },
        toggle: function(e) {
            return "boolean" == typeof e ? e ? this.show() : this.hide() : this.each(function() {
                Ct(this) ? K(this).show() : K(this).hide()
            })
        }
    }), K.Tween = A, A.prototype = {
        constructor: A,
        init: function(e, t, n, i, r, o) {
            this.elem = e, this.prop = n, this.easing = r || "swing", this.options = t, this.start = this.now = this.cur(), this.end = i, this.unit = o || (K.cssNumber[n] ? "" : "px")
        },
        cur: function() {
            var e = A.propHooks[this.prop];
            return e && e.get ? e.get(this) : A.propHooks._default.get(this)
        },
        run: function(e) {
            var t, n = A.propHooks[this.prop];
            return this.pos = t = this.options.duration ? K.easing[this.easing](e, this.options.duration * e, 0, 1, this.options.duration) : e, this.now = (this.end - this.start) * t + this.start, this.options.step && this.options.step.call(this.elem, this.now, this), n && n.set ? n.set(this) : A.propHooks._default.set(this), this
        }
    }, A.prototype.init.prototype = A.prototype, A.propHooks = {
        _default: {
            get: function(e) {
                var t;
                return null == e.elem[e.prop] || e.elem.style && null != e.elem.style[e.prop] ? (t = K.css(e.elem, e.prop, ""), t && "auto" !== t ? t : 0) : e.elem[e.prop]
            },
            set: function(e) {
                K.fx.step[e.prop] ? K.fx.step[e.prop](e) : e.elem.style && (null != e.elem.style[K.cssProps[e.prop]] || K.cssHooks[e.prop]) ? K.style(e.elem, e.prop, e.now + e.unit) : e.elem[e.prop] = e.now
            }
        }
    }, A.propHooks.scrollTop = A.propHooks.scrollLeft = {
        set: function(e) {
            e.elem.nodeType && e.elem.parentNode && (e.elem[e.prop] = e.now)
        }
    }, K.easing = {
        linear: function(e) {
            return e
        },
        swing: function(e) {
            return .5 - Math.cos(e * Math.PI) / 2
        }
    }, K.fx = A.prototype.init, K.fx.step = {};
    var Zt, Gt, Jt = /^(?:toggle|show|hide)$/, Kt = new RegExp("^(?:([+-])=|)(" + wt + ")([a-z%]*)$", "i"), en = /queueHooks$/, tn = [N], nn = {
        "*": [function(e, t) {
            var n = this.createTween(e, t), i = n.cur(), r = Kt.exec(t), o = r && r[3] || (K.cssNumber[e] ? "" : "px"), a = (K.cssNumber[e] || "px" !== o&&+i) && Kt.exec(K.css(n.elem, e)), s = 1, l = 20;
            if (a && a[3] !== o) {
                o = o || a[3], r = r || [], a =+ i || 1;
                do 
                    s = s || ".5", a/=s, K.style(n.elem, e, a + o);
                while (s !== (s = n.cur() / i) && 1 !== s&&--l)
                }
            return r && (a = n.start =+ a||+i || 0, n.unit = o, n.end = r[1] ? a + (r[1] + 1) * r[2] : + r[2]), n
        }
        ]
    };
    K.Animation = K.extend(L, {
        tweener: function(e, t) {
            K.isFunction(e) ? (t = e, e = ["*"]): e = e.split(" ");
            for (var n,
            i = 0,
            r = e.length;
            r > i;
            i++)n = e[i],
            nn[n] = nn[n] || [],
            nn[n].unshift(t)
        }, prefilter : function(e, t) {
            t ? tn.unshift(e) : tn.push(e)
        }
    }), K.speed = function(e, t, n) {
        var i = e && "object" == typeof e ? K.extend({}, e): {
            complete: n ||!n && t || K.isFunction(e) && e,
            duration: e,
            easing: n && t || t&&!K.isFunction(t) && t
        };
        return i.duration = K.fx.off ? 0 : "number" == typeof i.duration ? i.duration : i.duration in K.fx.speeds ? K.fx.speeds[i.duration] : K.fx.speeds._default, (null == i.queue || i.queue===!0) && (i.queue = "fx"), i.old = i.complete, i.complete = function() {
            K.isFunction(i.old) && i.old.call(this), i.queue && K.dequeue(this, i.queue)
        }, i
    }, K.fn.extend({
        fadeTo: function(e, t, n, i) {
            return this.filter(Ct).css("opacity", 0).show().end().animate({
                opacity: t
            }, e, n, i)
        },
        animate: function(e, t, n, i) {
            var r = K.isEmptyObject(e), o = K.speed(t, n, i), a = function() {
                var t = L(this, K.extend({}, e), o);
                (r || vt.get(this, "finish")) && t.stop(!0)
            };
            return a.finish = a, r || o.queue===!1 ? this.each(a) : this.queue(o.queue, a)
        },
        stop: function(e, t, n) {
            var i = function(e) {
                var t = e.stop;
                delete e.stop, t(n)
            };
            return "string" != typeof e && (n = t, t = e, e = void 0), t && e!==!1 && this.queue(e || "fx", []), this.each(function() {
                var t=!0, r = null != e && e + "queueHooks", o = K.timers, a = vt.get(this);
                if (r)
                    a[r] && a[r].stop && i(a[r]);
                else 
                    for (r in a)
                        a[r] && a[r].stop && en.test(r) && i(a[r]);
                for (r = o.length; r--;)
                    o[r].elem !== this || null != e && o[r].queue !== e || (o[r].anim.stop(n), t=!1, o.splice(r, 1));
                (t ||!n) && K.dequeue(this, e)
            })
        },
        finish: function(e) {
            return e!==!1 && (e = e || "fx"), this.each(function() {
                var t, n = vt.get(this), i = n[e + "queue"], r = n[e + "queueHooks"], o = K.timers, a = i ? i.length: 0;
                for (n.finish=!0, K.queue(this, e, []), r && r.stop && r.stop.call(this, !0), t = o.length; t--;)
                    o[t].elem === this && o[t].queue === e && (o[t].anim.stop(!0), o.splice(t, 1));
                for (t = 0; a > t; t++)
                    i[t] && i[t].finish && i[t].finish.call(this);
                delete n.finish
            })
        }
    }), K.each(["toggle", "show", "hide"], function(e, t) {
        var n = K.fn[t];
        K.fn[t] = function(e, i, r) {
            return null == e || "boolean" == typeof e ? n.apply(this, arguments) : this.animate(j(t, !0), e, i, r)
        }
    }), K.each({
        slideDown: j("show"),
        slideUp: j("hide"),
        slideToggle: j("toggle"),
        fadeIn: {
            opacity: "show"
        },
        fadeOut: {
            opacity: "hide"
        },
        fadeToggle: {
            opacity: "toggle"
        }
    }, function(e, t) {
        K.fn[e] = function(e, n, i) {
            return this.animate(t, e, n, i)
        }
    }), K.timers = [], K.fx.tick = function() {
        var e, t = 0, n = K.timers;
        for (Zt = K.now(); t < n.length; t++)
            e = n[t], e() || n[t] !== e || n.splice(t--, 1);
        n.length || K.fx.stop(), Zt = void 0
    }, K.fx.timer = function(e) {
        K.timers.push(e), e() ? K.fx.start() : K.timers.pop()
    }, K.fx.interval = 13, K.fx.start = function() {
        Gt || (Gt = setInterval(K.fx.tick, K.fx.interval))
    }, K.fx.stop = function() {
        clearInterval(Gt), Gt = null
    }, K.fx.speeds = {
        slow: 600,
        fast: 200,
        _default: 400
    }, K.fn.delay = function(e, t) {
        return e = K.fx ? K.fx.speeds[e] || e : e, t = t || "fx", this.queue(t, function(t, n) {
            var i = setTimeout(t, e);
            n.stop = function() {
                clearTimeout(i)
            }
        })
    }, function() {
        var e = G.createElement("input"), t = G.createElement("select"), n = t.appendChild(G.createElement("option"));
        e.type = "checkbox", Z.checkOn = "" !== e.value, Z.optSelected = n.selected, t.disabled=!0, Z.optDisabled=!n.disabled, e = G.createElement("input"), e.value = "t", e.type = "radio", Z.radioValue = "t" === e.value
    }();
    var rn, on, an = K.expr.attrHandle;
    K.fn.extend({
        attr: function(e, t) {
            return mt(this, K.attr, e, t, arguments.length > 1)
        },
        removeAttr: function(e) {
            return this.each(function() {
                K.removeAttr(this, e)
            })
        }
    }), K.extend({
        attr: function(e, t, n) {
            var i, r, o = e.nodeType;
            return e && 3 !== o && 8 !== o && 2 !== o ? typeof e.getAttribute === St ? K.prop(e, t, n) : (1 === o && K.isXMLDoc(e) || (t = t.toLowerCase(), i = K.attrHooks[t] || (K.expr.match.bool.test(t) ? on : rn)), void 0 === n ? i && "get"in i && null !== (r = i.get(e, t)) ? r : (r = K.find.attr(e, t), null == r ? void 0 : r) : null !== n ? i && "set"in i && void 0 !== (r = i.set(e, n, t)) ? r : (e.setAttribute(t, n + ""), n) : void K.removeAttr(e, t)) : void 0
        },
        removeAttr: function(e, t) {
            var n, i, r = 0, o = t && t.match(ht);
            if (o && 1 === e.nodeType)
                for (; n = o[r++];)
                    i = K.propFix[n] || n, K.expr.match.bool.test(n) && (e[i]=!1), e.removeAttribute(n)
        },
        attrHooks: {
            type: {
                set: function(e, t) {
                    if (!Z.radioValue && "radio" === t && K.nodeName(e, "input")) {
                        var n = e.value;
                        return e.setAttribute("type", t), n && (e.value = n), t
                    }
                }
            }
        }
    }), on = {
        set: function(e, t, n) {
            return t===!1 ? K.removeAttr(e, n) : e.setAttribute(n, n), n
        }
    }, K.each(K.expr.match.bool.source.match(/\w+/g), function(e, t) {
        var n = an[t] || K.find.attr;
        an[t] = function(e, t, i) {
            var r, o;
            return i || (o = an[t], an[t] = r, r = null != n(e, t, i) ? t.toLowerCase() : null, an[t] = o), r
        }
    });
    var sn = /^(?:input|select|textarea|button)$/i;
    K.fn.extend({
        prop: function(e, t) {
            return mt(this, K.prop, e, t, arguments.length > 1)
        },
        removeProp: function(e) {
            return this.each(function() {
                delete this[K.propFix[e] || e]
            })
        }
    }), K.extend({
        propFix: {
            "for": "htmlFor",
            "class": "className"
        },
        prop: function(e, t, n) {
            var i, r, o, a = e.nodeType;
            return e && 3 !== a && 8 !== a && 2 !== a ? (o = 1 !== a ||!K.isXMLDoc(e), o && (t = K.propFix[t] || t, r = K.propHooks[t]), void 0 !== n ? r && "set"in r && void 0 !== (i = r.set(e, n, t)) ? i : e[t] = n : r && "get"in r && null !== (i = r.get(e, t)) ? i : e[t]) : void 0
        },
        propHooks: {
            tabIndex: {
                get: function(e) {
                    return e.hasAttribute("tabindex") || sn.test(e.nodeName) || e.href ? e.tabIndex : - 1
                }
            }
        }
    }), Z.optSelected || (K.propHooks.selected = {
        get: function(e) {
            var t = e.parentNode;
            return t && t.parentNode && t.parentNode.selectedIndex, null
        }
    }), K.each(["tabIndex", "readOnly", "maxLength", "cellSpacing", "cellPadding", "rowSpan", "colSpan", "useMap", "frameBorder", "contentEditable"], function() {
        K.propFix[this.toLowerCase()] = this
    });
    var ln = /[\t\r\n\f]/g;
    K.fn.extend({
        addClass: function(e) {
            var t, n, i, r, o, a, s = "string" == typeof e && e, l = 0, c = this.length;
            if (K.isFunction(e))
                return this.each(function(t) {
                    K(this).addClass(e.call(this, t, this.className))
                });
            if (s)
                for (t = (e || "").match(ht) || []; c > l; l++)
                    if (n = this[l], i = 1 === n.nodeType && (n.className ? (" " + n.className + " ").replace(ln, " ") : " ")) {
                        for (o = 0; r = t[o++];)
                            i.indexOf(" " + r + " ") < 0 && (i += r + " ");
                            a = K.trim(i), n.className !== a && (n.className = a)
                    }
            return this
        },
        removeClass: function(e) {
            var t, n, i, r, o, a, s = 0 === arguments.length || "string" == typeof e && e, l = 0, c = this.length;
            if (K.isFunction(e))
                return this.each(function(t) {
                    K(this).removeClass(e.call(this, t, this.className))
                });
            if (s)
                for (t = (e || "").match(ht) || []; c > l; l++)
                    if (n = this[l], i = 1 === n.nodeType && (n.className ? (" " + n.className + " ").replace(ln, " ") : "")) {
                        for (o = 0; r = t[o++];)
                            for (; i.indexOf(" " + r + " ") >= 0;)
                                i = i.replace(" " + r + " ", " ");
                                a = e ? K.trim(i) : "", n.className !== a && (n.className = a)
                    }
            return this
        },
        toggleClass: function(e, t) {
            var n = typeof e;
            return "boolean" == typeof t && "string" === n ? t ? this.addClass(e) : this.removeClass(e) : this.each(K.isFunction(e) ? function(n) {
                K(this).toggleClass(e.call(this, n, this.className, t), t)
            } : function() {
                if ("string" === n)
                    for (var t, i = 0, r = K(this), o = e.match(ht) || []; t = o[i++];)
                        r.hasClass(t) ? r.removeClass(t) : r.addClass(t);
                else (n === St || "boolean" === n) 
                    && (this.className && vt.set(this, "__className__", this.className), this.className = this.className || e===!1 ? "" : vt.get(this, "__className__") || "")
            })
        },
        hasClass: function(e) {
            for (var t = " " + e + " ", n = 0, i = this.length; i > n; n++)
                if (1 === this[n].nodeType && (" " + this[n].className + " ").replace(ln, " ").indexOf(t) >= 0)
                    return !0;
            return !1
        }
    });
    var cn = /\r/g;
    K.fn.extend({
        val: function(e) {
            var t, n, i, r = this[0];
            return arguments.length ? (i = K.isFunction(e), this.each(function(n) {
                var r;
                1 === this.nodeType && (r = i ? e.call(this, n, K(this).val()) : e, null == r ? r = "" : "number" == typeof r ? r += "" : K.isArray(r) && (r = K.map(r, function(e) {
                    return null == e ? "" : e + ""
                })), t = K.valHooks[this.type] || K.valHooks[this.nodeName.toLowerCase()], t && "set"in t && void 0 !== t.set(this, r, "value") || (this.value = r))
            })) : r ? (t = K.valHooks[r.type] || K.valHooks[r.nodeName.toLowerCase()], t && "get"in t && void 0 !== (n = t.get(r, "value")) ? n : (n = r.value, "string" == typeof n ? n.replace(cn, "") : null == n ? "" : n)) : void 0
        }
    }), K.extend({
        valHooks: {
            option: {
                get: function(e) {
                    var t = K.find.attr(e, "value");
                    return null != t ? t : K.trim(K.text(e))
                }
            },
            select: {
                get: function(e) {
                    for (var t, n, i = e.options, r = e.selectedIndex, o = "select-one" === e.type || 0 > r, a = o ? null : [], s = o ? r + 1 : i.length, l = 0 > r ? s : o ? r : 0; s > l; l++)
                        if (n = i[l], !(!n.selected && l !== r || (Z.optDisabled ? n.disabled : null !== n.getAttribute("disabled")) || n.parentNode.disabled && K.nodeName(n.parentNode, "optgroup"))) {
                            if (t = K(n).val(), o)
                                return t;
                                a.push(t)
                        }
                    return a
                },
                set: function(e, t) {
                    for (var n, i, r = e.options, o = K.makeArray(t), a = r.length; a--;)
                        i = r[a], (i.selected = K.inArray(i.value, o) >= 0) && (n=!0);
                    return n || (e.selectedIndex =- 1), o
                }
            }
        }
    }), K.each(["radio", "checkbox"], function() {
        K.valHooks[this] = {
            set: function(e, t) {
                return K.isArray(t) ? e.checked = K.inArray(K(e).val(), t) >= 0 : void 0
            }
        }, Z.checkOn || (K.valHooks[this].get = function(e) {
            return null === e.getAttribute("value") ? "on" : e.value
        })
    }), K.each("blur focus focusin focusout load resize scroll unload click dblclick mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave change select submit keydown keypress keyup error contextmenu".split(" "), function(e, t) {
        K.fn[t] = function(e, n) {
            return arguments.length > 0 ? this.on(t, null, e, n) : this.trigger(t)
        }
    }), K.fn.extend({
        hover: function(e, t) {
            return this.mouseenter(e).mouseleave(t || e)
        },
        bind: function(e, t, n) {
            return this.on(e, null, t, n)
        },
        unbind: function(e, t) {
            return this.off(e, null, t)
        },
        delegate: function(e, t, n, i) {
            return this.on(t, e, n, i)
        },
        undelegate: function(e, t, n) {
            return 1 === arguments.length ? this.off(e, "**") : this.off(t, e || "**", n)
        }
    });
    var un = K.now(), dn = /\?/;
    K.parseJSON = function(e) {
        return JSON.parse(e + "")
    }, K.parseXML = function(e) {
        var t, n;
        if (!e || "string" != typeof e)
            return null;
        try {
            n = new DOMParser, t = n.parseFromString(e, "text/xml")
        } catch (i) {
            t = void 0
        }
        return (!t || t.getElementsByTagName("parsererror").length) && K.error("Invalid XML: " + e), t
    };
    var fn, hn, pn = /#.*$/, gn = /([?&])_=[^&]*/, mn = /^(.*?):[ \t]*([^\r\n]*)$/gm, vn = /^(?:about|app|app-storage|.+-extension|file|res|widget):$/, yn = /^(?:GET|HEAD)$/, bn = /^\/\//, _n = /^([\w.+-]+:)(?:\/\/(?:[^\/?#]*@|)([^\/?#:]*)(?::(\d+)|)|)/, wn = {}, xn = {}, Cn = "*/".concat("*");
    try {
        hn = location.href
    } catch (kn) {
        hn = G.createElement("a"), hn.href = "", hn = hn.href
    }
    fn = _n.exec(hn.toLowerCase()) || [], K.extend({
        active: 0,
        lastModified: {},
        etag: {},
        ajaxSettings: {
            url: hn,
            type: "GET",
            isLocal: vn.test(fn[1]),
            global: !0,
            processData: !0,
            async: !0,
            contentType: "application/x-www-form-urlencoded; charset=UTF-8",
            accepts: {
                "*": Cn,
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
                "text json": K.parseJSON,
                "text xml": K.parseXML
            },
            flatOptions: {
                url: !0,
                context: !0
            }
        },
        ajaxSetup: function(e, t) {
            return t ? M(M(e, K.ajaxSettings), t) : M(K.ajaxSettings, e)
        },
        ajaxPrefilter: I(wn),
        ajaxTransport: I(xn),
        ajax: function(e, t) {
            function n(e, t, n, a) {
                var l, u, v, y, _, x = t;
                2 !== b && (b = 2, s && clearTimeout(s), i = void 0, o = a || "", w.readyState = e > 0 ? 4 : 0, l = e >= 200 && 300 > e || 304 === e, n && (y = R(d, w, n)), y = H(d, y, w, l), l ? (d.ifModified && (_ = w.getResponseHeader("Last-Modified"), _ && (K.lastModified[r] = _), _ = w.getResponseHeader("etag"), _ && (K.etag[r] = _)), 204 === e || "HEAD" === d.type ? x = "nocontent" : 304 === e ? x = "notmodified" : (x = y.state, u = y.data, v = y.error, l=!v)) : (v = x, (e ||!x) && (x = "error", 0 > e && (e = 0))), w.status = e, w.statusText = (t || x) + "", l ? p.resolveWith(f, [u, x, w]) : p.rejectWith(f, [w, x, v]), w.statusCode(m), m = void 0, c && h.trigger(l ? "ajaxSuccess" : "ajaxError", [w, d, l ? u: v]), g.fireWith(f, [w, x]), c && (h.trigger("ajaxComplete", [w, d]), --K.active || K.event.trigger("ajaxStop")))
            }
            "object" == typeof e && (t = e, e = void 0), t = t || {};
            var i, r, o, a, s, l, c, u, d = K.ajaxSetup({}, t), f = d.context || d, h = d.context && (f.nodeType || f.jquery) ? K(f): K.event, p = K.Deferred(), g = K.Callbacks("once memory"), m = d.statusCode || {}, v = {}, y = {}, b = 0, _ = "canceled", w = {
                readyState: 0,
                getResponseHeader: function(e) {
                    var t;
                    if (2 === b) {
                        if (!a)
                            for (a = {}; t = mn.exec(o);)
                                a[t[1].toLowerCase()] = t[2];
                        t = a[e.toLowerCase()]
                    }
                    return null == t ? null : t
                },
                getAllResponseHeaders: function() {
                    return 2 === b ? o : null
                },
                setRequestHeader: function(e, t) {
                    var n = e.toLowerCase();
                    return b || (e = y[n] = y[n] || e, v[e] = t), this
                },
                overrideMimeType: function(e) {
                    return b || (d.mimeType = e), this
                },
                statusCode: function(e) {
                    var t;
                    if (e)
                        if (2 > b)
                            for (t in e)
                                m[t] = [m[t], e[t]];
                        else 
                            w.always(e[w.status]);
                    return this
                },
                abort: function(e) {
                    var t = e || _;
                    return i && i.abort(t), n(0, t), this
                }
            };
            if (p.promise(w).complete = g.add, w.success = w.done, w.error = w.fail, d.url = ((e || d.url || hn) + "").replace(pn, "").replace(bn, fn[1] + "//"), d.type = t.method || t.type || d.method || d.type, d.dataTypes = K.trim(d.dataType || "*").toLowerCase().match(ht) || [""], null == d.crossDomain && (l = _n.exec(d.url.toLowerCase()), d.crossDomain=!(!l || l[1] === fn[1] && l[2] === fn[2] && (l[3] || ("http:" === l[1] ? "80" : "443")) === (fn[3] || ("http:" === fn[1] ? "80" : "443")))), d.data && d.processData && "string" != typeof d.data && (d.data = K.param(d.data, d.traditional)), O(wn, d, t, w), 2 === b)
                return w;
            c = d.global, c && 0 === K.active++&&K.event.trigger("ajaxStart"), d.type = d.type.toUpperCase(), d.hasContent=!yn.test(d.type), r = d.url, d.hasContent || (d.data && (r = d.url += (dn.test(r) ? "&" : "?") + d.data, delete d.data), d.cache===!1 && (d.url = gn.test(r) ? r.replace(gn, "$1_=" + un++) : r + (dn.test(r) ? "&" : "?") + "_=" + un++)), d.ifModified && (K.lastModified[r] && w.setRequestHeader("If-Modified-Since", K.lastModified[r]), K.etag[r] && w.setRequestHeader("If-None-Match", K.etag[r])), (d.data && d.hasContent && d.contentType!==!1 || t.contentType) && w.setRequestHeader("Content-Type", d.contentType), w.setRequestHeader("Accept", d.dataTypes[0] && d.accepts[d.dataTypes[0]] ? d.accepts[d.dataTypes[0]] + ("*" !== d.dataTypes[0] ? ", " + Cn + "; q=0.01" : "") : d.accepts["*"]);
            for (u in d.headers)
                w.setRequestHeader(u, d.headers[u]);
            if (d.beforeSend && (d.beforeSend.call(f, w, d)===!1 || 2 === b))
                return w.abort();
            _ = "abort";
            for (u in{
                success: 1,
                error: 1,
                complete: 1
            })
                w[u](d[u]);
            if (i = O(xn, d, t, w)) {
                w.readyState = 1, c && h.trigger("ajaxSend", [w, d]), d.async && d.timeout > 0 && (s = setTimeout(function() {
                    w.abort("timeout")
                }, d.timeout));
                try {
                    b = 1, i.send(v, n)
                } catch (x) {
                    if (!(2 > b))
                        throw x;
                    n( - 1, x)
                }
            } else 
                n( - 1, "No Transport");
            return w
        },
        getJSON: function(e, t, n) {
            return K.get(e, t, n, "json")
        },
        getScript: function(e, t) {
            return K.get(e, void 0, t, "script")
        }
    }), K.each(["get", "post"], function(e, t) {
        K[t] = function(e, n, i, r) {
            return K.isFunction(n) && (r = r || i, i = n, n = void 0), K.ajax({
                url: e,
                type: t,
                dataType: r,
                data: n,
                success: i
            })
        }
    }), K.each(["ajaxStart", "ajaxStop", "ajaxComplete", "ajaxError", "ajaxSuccess", "ajaxSend"], function(e, t) {
        K.fn[t] = function(e) {
            return this.on(t, e)
        }
    }), K._evalUrl = function(e) {
        return K.ajax({
            url: e,
            type: "GET",
            dataType: "script",
            async: !1,
            global: !1,
            "throws": !0
        })
    }, K.fn.extend({
        wrapAll: function(e) {
            var t;
            return K.isFunction(e) ? this.each(function(t) {
                K(this).wrapAll(e.call(this, t))
            }) : (this[0] && (t = K(e, this[0].ownerDocument).eq(0).clone(!0), this[0].parentNode && t.insertBefore(this[0]), t.map(function() {
                for (var e = this; e.firstElementChild;)
                    e = e.firstElementChild;
                return e
            }).append(this)), this)
        },
        wrapInner: function(e) {
            return this.each(K.isFunction(e) ? function(t) {
                K(this).wrapInner(e.call(this, t))
            } : function() {
                var t = K(this), n = t.contents();
                n.length ? n.wrapAll(e) : t.append(e)
            })
        },
        wrap: function(e) {
            var t = K.isFunction(e);
            return this.each(function(n) {
                K(this).wrapAll(t ? e.call(this, n) : e)
            })
        },
        unwrap: function() {
            return this.parent().each(function() {
                K.nodeName(this, "body") || K(this).replaceWith(this.childNodes)
            }).end()
        }
    }), K.expr.filters.hidden = function(e) {
        return e.offsetWidth <= 0 && e.offsetHeight <= 0
    }, K.expr.filters.visible = function(e) {
        return !K.expr.filters.hidden(e)
    };
    var Sn = /%20/g, Tn = /\[\]$/, Fn = /\r?\n/g, An = /^(?:submit|button|image|reset|file)$/i, En = /^(?:input|select|textarea|keygen)/i;
    K.param = function(e, t) {
        var n, i = [], r = function(e, t) {
            t = K.isFunction(t) ? t() : null == t ? "" : t, i[i.length] = encodeURIComponent(e) + "=" + encodeURIComponent(t)
        };
        if (void 0 === t && (t = K.ajaxSettings && K.ajaxSettings.traditional), K.isArray(e) || e.jquery&&!K.isPlainObject(e)
            )K.each(e, function() {
            r(this.name, this.value)
        });
        else 
            for (n in e)
                P(n, e[n], t, r);
        return i.join("&").replace(Sn, "+")
    }, K.fn.extend({
        serialize: function() {
            return K.param(this.serializeArray())
        },
        serializeArray: function() {
            return this.map(function() {
                var e = K.prop(this, "elements");
                return e ? K.makeArray(e) : this
            }).filter(function() {
                var e = this.type;
                return this.name&&!K(this).is(":disabled") && En.test(this.nodeName)&&!An.test(e) && (this.checked ||!kt.test(e))
            }).map(function(e, t) {
                var n = K(this).val();
                return null == n ? null : K.isArray(n) ? K.map(n, function(e) {
                    return {
                        name: t.name,
                        value: e.replace(Fn, "\r\n")
                    }
                }) : {
                    name: t.name,
                    value: n.replace(Fn, "\r\n")
                }
            }).get()
        }
    }), K.ajaxSettings.xhr = function() {
        try {
            return new XMLHttpRequest
        } catch (e) {}
    };
    var jn = 0, Dn = {}, Nn = {
        0: 200,
        1223: 204
    }, qn = K.ajaxSettings.xhr();
    e.ActiveXObject && K(e).on("unload", function() {
        for (var e in Dn)
            Dn[e]()
    }), Z.cors=!!qn && "withCredentials"in qn, Z.ajax = qn=!!qn, K.ajaxTransport(function(e) {
        var t;
        return Z.cors || qn&&!e.crossDomain ? {
            send: function(n, i) {
                var r, o = e.xhr(), a=++jn;
                if (o.open(e.type, e.url, e.async, e.username, e.password), e.xhrFields)
                    for (r in e.xhrFields)
                        o[r] = e.xhrFields[r];
                e.mimeType && o.overrideMimeType && o.overrideMimeType(e.mimeType), e.crossDomain || n["X-Requested-With"] || (n["X-Requested-With"] = "XMLHttpRequest");
                for (r in n)
                    o.setRequestHeader(r, n[r]);
                t = function(e) {
                    return function() {
                        t && (delete Dn[a], t = o.onload = o.onerror = null, "abort" === e ? o.abort() : "error" === e ? i(o.status, o.statusText) : i(Nn[o.status] || o.status, o.statusText, "string" == typeof o.responseText ? {
                            text: o.responseText
                        } : void 0, o.getAllResponseHeaders()))
                    }
                }, o.onload = t(), o.onerror = t("error"), t = Dn[a] = t("abort");
                try {
                    o.send(e.hasContent && e.data || null)
                } catch (s) {
                    if (t)
                        throw s
                }
            },
            abort: function() {
                t && t()
            }
        } : void 0
    }), K.ajaxSetup({
        accepts: {
            script: "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript"
        },
        contents: {
            script: /(?:java|ecma)script/
        },
        converters: {
            "text script": function(e) {
                return K.globalEval(e), e
            }
        }
    }), K.ajaxPrefilter("script", function(e) {
        void 0 === e.cache && (e.cache=!1), e.crossDomain && (e.type = "GET")
    }), K.ajaxTransport("script", function(e) {
        if (e.crossDomain) {
            var t, n;
            return {
                send: function(i, r) {
                    t = K("<script>").prop({
                        async: !0,
                        charset: e.scriptCharset,
                        src: e.url
                    }).on("load error", n = function(e) {
                        t.remove(), n = null, e && r("error" === e.type ? 404 : 200, e.type)
                    }), G.head.appendChild(t[0])
                },
                abort: function() {
                    n && n()
                }
            }
        }
    });
    var Ln = [], In = /(=)\?(?=&|$)|\?\?/;
    K.ajaxSetup({
        jsonp: "callback",
        jsonpCallback: function() {
            var e = Ln.pop() || K.expando + "_" + un++;
            return this[e]=!0, e
        }
    }), K.ajaxPrefilter("json jsonp", function(t, n, i) {
        var r, o, a, s = t.jsonp!==!1 && (In.test(t.url) ? "url" : "string" == typeof t.data&&!(t.contentType || "").indexOf("application/x-www-form-urlencoded") && In.test(t.data) && "data");
        return s || "jsonp" === t.dataTypes[0] ? (r = t.jsonpCallback = K.isFunction(t.jsonpCallback) ? t.jsonpCallback() : t.jsonpCallback, s ? t[s] = t[s].replace(In, "$1" + r) : t.jsonp!==!1 && (t.url += (dn.test(t.url) ? "&" : "?") + t.jsonp + "=" + r), t.converters["script json"] = function() {
            return a || K.error(r + " was not called"), a[0]
        }, t.dataTypes[0] = "json", o = e[r], e[r] = function() {
            a = arguments
        }, i.always(function() {
            e[r] = o, t[r] && (t.jsonpCallback = n.jsonpCallback, Ln.push(r)), a && K.isFunction(o) && o(a[0]), a = o = void 0
        }), "script") : void 0
    }), K.parseHTML = function(e, t, n) {
        if (!e || "string" != typeof e)
            return null;
        "boolean" == typeof t && (n = t, t=!1), t = t || G;
        var i = at.exec(e), r=!n && [];
        return i ? [t.createElement(i[1])] : (i = K.buildFragment([e], t, r), r && r.length && K(r).remove(), K.merge([], i.childNodes))
    };
    var On = K.fn.load;
    K.fn.load = function(e, t, n) {
        if ("string" != typeof e && On)
            return On.apply(this, arguments);
        var i, r, o, a = this, s = e.indexOf(" ");
        return s >= 0 && (i = K.trim(e.slice(s)), e = e.slice(0, s)), K.isFunction(t) ? (n = t, t = void 0) : t && "object" == typeof t && (r = "POST"), a.length > 0 && K.ajax({
            url: e,
            type: r,
            dataType: "html",
            data: t
        }).done(function(e) {
            o = arguments, a.html(i ? K("<div>").append(K.parseHTML(e)).find(i) : e)
        }).complete(n && function(e, t) {
            a.each(n, o || [e.responseText, t, e])
        }), this
    }, K.expr.filters.animated = function(e) {
        return K.grep(K.timers, function(t) {
            return e === t.elem
        }).length
    };
    var Mn = e.document.documentElement;
    K.offset = {
        setOffset: function(e, t, n) {
            var i, r, o, a, s, l, c, u = K.css(e, "position"), d = K(e), f = {};
            "static" === u && (e.style.position = "relative"), s = d.offset(), o = K.css(e, "top"), l = K.css(e, "left"), c = ("absolute" === u || "fixed" === u) && (o + l).indexOf("auto")>-1, c ? (i = d.position(), a = i.top, r = i.left) : (a = parseFloat(o) || 0, r = parseFloat(l) || 0), K.isFunction(t) && (t = t.call(e, n, s)), null != t.top && (f.top = t.top - s.top + a), null != t.left && (f.left = t.left - s.left + r), "using"in t ? t.using.call(e, f) : d.css(f)
        }
    }, K.fn.extend({
        offset: function(e) {
            if (arguments.length)
                return void 0 === e ? this : this.each(function(t) {
                K.offset.setOffset(this, e, t)
            });
            var t, n, i = this[0], r = {
                top: 0,
                left: 0
            }, o = i && i.ownerDocument;
            return o ? (t = o.documentElement, K.contains(t, i) ? (typeof i.getBoundingClientRect !== St && (r = i.getBoundingClientRect()), n = z(o), {
                top: r.top + n.pageYOffset - t.clientTop,
                left: r.left + n.pageXOffset - t.clientLeft
            }) : r) : void 0
        },
        position: function() {
            if (this[0]) {
                var e, t, n = this[0], i = {
                    top: 0,
                    left: 0
                };
                return "fixed" === K.css(n, "position") ? t = n.getBoundingClientRect() : (e = this.offsetParent(), t = this.offset(), K.nodeName(e[0], "html") || (i = e.offset()), i.top += K.css(e[0], "borderTopWidth", !0), i.left += K.css(e[0], "borderLeftWidth", !0)), {
                    top: t.top - i.top - K.css(n, "marginTop", !0),
                    left: t.left - i.left - K.css(n, "marginLeft", !0)
                }
            }
        },
        offsetParent: function() {
            return this.map(function() {
                for (var e = this.offsetParent || Mn; e&&!K.nodeName(e, "html") && "static" === K.css(e, "position");)
                    e = e.offsetParent;
                return e || Mn
            })
        }
    }), K.each({
        scrollLeft: "pageXOffset",
        scrollTop: "pageYOffset"
    }, function(t, n) {
        var i = "pageYOffset" === n;
        K.fn[t] = function(r) {
            return mt(this, function(t, r, o) {
                var a = z(t);
                return void 0 === o ? a ? a[n] : t[r] : void(a ? a.scrollTo(i ? e.pageXOffset : o, i ? o : e.pageYOffset) : t[r] = o)
            }, t, r, arguments.length, null)
        }
    }), K.each(["top", "left"], function(e, t) {
        K.cssHooks[t] = x(Z.pixelPosition, function(e, n) {
            return n ? (n = w(e, t), $t.test(n) ? K(e).position()[t] + "px" : n) : void 0
        })
    }), K.each({
        Height: "height",
        Width: "width"
    }, function(e, t) {
        K.each({
            padding: "inner" + e,
            content: t,
            "": "outer" + e
        }, function(n, i) {
            K.fn[i] = function(i, r) {
                var o = arguments.length && (n || "boolean" != typeof i), a = n || (i===!0 || r===!0 ? "margin" : "border");
                return mt(this, function(t, n, i) {
                    var r;
                    return K.isWindow(t) ? t.document.documentElement["client" + e] : 9 === t.nodeType ? (r = t.documentElement, Math.max(t.body["scroll" + e], r["scroll" + e], t.body["offset" + e], r["offset" + e], r["client" + e])) : void 0 === i ? K.css(t, n, a) : K.style(t, n, i, a)
                }, t, o ? i : void 0, o, null)
            }
        })
    }), K.fn.size = function() {
        return this.length
    }, K.fn.andSelf = K.fn.addBack, "function" == typeof define && define.amd && define("jquery", [], function() {
        return K
    });
    var Rn = e.jQuery, Hn = e.$;
    return K.noConflict = function(t) {
        return e.$ === K && (e.$ = Hn), t && e.jQuery === K && (e.jQuery = Rn), K
    }, typeof t === St && (e.jQuery = e.$ = K), K
}), function(e, t, n) {
    function i(e) {
        var t = {}, i = /^jQuery\d+$/;
        return n.each(e.attributes, function(e, n) {
            n.specified&&!i.test(n.name) && (t[n.name] = n.value)
        }), t
    }
    function r(e, t) {
        var i = this, r = n(i);
        if (i.value == r.attr("placeholder") && r.hasClass("placeholder"))
            if (r.data("placeholder-password")) {
                if (r = r.hide().next().show().attr("id", r.removeAttr("id").data("placeholder-id")), e===!0)
                    return r[0].value = t;
                    r.focus()
            } else 
                i.value = "", r.removeClass("placeholder"), i == a() && i.select()
    }
    function o() {
        var e, t = this, o = n(t), a = this.id;
        if ("" == t.value) {
            if ("password" == t.type) {
                if (!o.data("placeholder-textinput")) {
                    try {
                        e = o.clone().attr({
                            type: "text"
                        })
                    } catch (s) {
                        e = n("<input>").attr(n.extend(i(this), {
                            type: "text"
                        }))
                    }
                    e.removeAttr("name").data({
                        "placeholder-password": o,
                        "placeholder-id": a
                    }).bind("focus.placeholder", r), o.data({
                        "placeholder-textinput": e,
                        "placeholder-id": a
                    }).before(e)
                }
                o = o.removeAttr("id").hide().prev().attr("id", a).show()
            }
            o.addClass("placeholder"), o[0].value = o.attr("placeholder")
        } else 
            o.removeClass("placeholder")
    }
    function a() {
        try {
            return t.activeElement
        } catch (e) {}
    }
    var s, l, c = "[object OperaMini]" == Object.prototype.toString.call(e.operamini), u = "placeholder"in t.createElement("input")&&!c, d = "placeholder"in t.createElement("textarea")&&!c, f = n.fn, h = n.valHooks, p = n.propHooks;
    u && d ? (l = f.placeholder = function() {
        return this
    }, l.input = l.textarea=!0) : (l = f.placeholder = function() {
        var e = this;
        return e.filter((u ? "textarea" : ":input") + "[placeholder]").not(".placeholder").bind({
            "focus.placeholder": r,
            "blur.placeholder": o
        }).data("placeholder-enabled", !0).trigger("blur.placeholder"), e
    }, l.input = u, l.textarea = d, s = {
        get: function(e) {
            var t = n(e), i = t.data("placeholder-password");
            return i ? i[0].value : t.data("placeholder-enabled") && t.hasClass("placeholder") ? "" : e.value
        },
        set: function(e, t) {
            var i = n(e), s = i.data("placeholder-password");
            return s ? s[0].value = t : i.data("placeholder-enabled") ? ("" == t ? (e.value = t, e != a() && o.call(e)) : i.hasClass("placeholder") ? r.call(e, !0, t) || (e.value = t) : e.value = t, i) : e.value = t
        }
    }, u || (h.input = s, p.value = s), d || (h.textarea = s, p.value = s), n(function() {
        n(t).delegate("form", "submit.placeholder", function() {
            var e = n(".placeholder", this).each(r);
            setTimeout(function() {
                e.each(o)
            }, 10)
        })
    }), n(e).bind("beforeunload.placeholder", function() {
        n(".placeholder").each(function() {
            this.value = ""
        })
    }))
}(this, document, jQuery), function(e) {
    "function" == typeof define && define.amd ? define(["jquery"], e) : e("object" == typeof exports ? require("jquery") : jQuery)
}(function(e) {
    function t(e) {
        return s.raw ? e : encodeURIComponent(e)
    }
    function n(e) {
        return s.raw ? e : decodeURIComponent(e)
    }
    function i(e) {
        return t(s.json ? JSON.stringify(e) : String(e))
    }
    function r(e) {
        0 === e.indexOf('"') && (e = e.slice(1, - 1).replace(/\\"/g, '"').replace(/\\\\/g, "\\"));
        try {
            return e = decodeURIComponent(e.replace(a, " ")), s.json ? JSON.parse(e) : e
        } catch (t) {}
    }
    function o(t, n) {
        var i = s.raw ? t: r(t);
        return e.isFunction(n) ? n(i) : i
    }
    var a = /\+/g, s = e.cookie = function(r, a, l) {
        if (void 0 !== a&&!e.isFunction(a)
            ) {
            if (l = e.extend({}, s.defaults, l), "number" == typeof l.expires) {
                var c = l.expires, u = l.expires = new Date;
                u.setTime( + u + 864e5 * c)
            }
            return document.cookie = [t(r), "=", i(a), l.expires ? "; expires=" + l.expires.toUTCString(): "", l.path ? "; path=" + l.path: "", l.domain ? "; domain=" + l.domain: "", l.secure ? "; secure": ""].join("")
        }
        for (var d = r ? void 0 : {}, f = document.cookie ? document.cookie.split("; ") : [], h = 0, p = f.length; p > h; h++) {
            var g = f[h].split("="), m = n(g.shift()), v = g.join("=");
            if (r && r === m) {
                d = o(v, a);
                break
            }
            r || void 0 === (v = o(v)) || (d[m] = v)
        }
        return d
    };
    s.defaults = {}, e.removeCookie = function(t, n) {
        return void 0 === e.cookie(t)?!1 : (e.cookie(t, "", e.extend({}, n, {
            expires: - 1
        })), !e.cookie(t))
    }
});
var deviceIsAndroid = navigator.userAgent.indexOf("Android") > 0, deviceIsIOS = /iP(ad|hone|od)/.test(navigator.userAgent), deviceIsIOS4 = deviceIsIOS && /OS 4_\d(_\d)?/.test(navigator.userAgent), deviceIsIOSWithBadTarget = deviceIsIOS && /OS ([6-9]|\d{2})_\d/.test(navigator.userAgent);
FastClick.prototype.needsClick = function(e) {
    "use strict";
    switch (e.nodeName.toLowerCase()) {
    case"button":
    case"select":
    case"textarea":
        if (e.disabled)
            return !0;
        break;
    case"input":
        if (deviceIsIOS && "file" === e.type || e.disabled)
            return !0;
        break;
    case"label":
    case"video":
        return !0
    }
    return /\bneedsclick\b/.test(e.className)
}, FastClick.prototype.needsFocus = function(e) {
    "use strict";
    switch (e.nodeName.toLowerCase()) {
    case"textarea":
        return !0;
    case"select":
        return !deviceIsAndroid;
    case"input":
        switch (e.type) {
        case"button":
        case"checkbox":
        case"file":
        case"image":
        case"radio":
        case"submit":
            return !1
        }
        return !e.disabled&&!e.readOnly;
    default:
        return /\bneedsfocus\b/.test(e.className)
    }
}, FastClick.prototype.sendClick = function(e, t) {
    "use strict";
    var n, i;
    document.activeElement && document.activeElement !== e && document.activeElement.blur(), i = t.changedTouches[0], n = document.createEvent("MouseEvents"), n.initMouseEvent(this.determineEventType(e), !0, !0, window, 1, i.screenX, i.screenY, i.clientX, i.clientY, !1, !1, !1, !1, 0, null), n.forwardedTouchEvent=!0, e.dispatchEvent(n)
}, FastClick.prototype.determineEventType = function(e) {
    "use strict";
    return deviceIsAndroid && "select" === e.tagName.toLowerCase() ? "mousedown" : "click"
}, FastClick.prototype.focus = function(e) {
    "use strict";
    var t;
    deviceIsIOS && e.setSelectionRange && 0 !== e.type.indexOf("date") && "time" !== e.type ? (t = e.value.length, e.setSelectionRange(t, t)) : e.focus()
}, FastClick.prototype.updateScrollParent = function(e) {
    "use strict";
    var t, n;
    if (t = e.fastClickScrollParent, !t ||!t.contains(e)) {
        n = e;
        do {
            if (n.scrollHeight > n.offsetHeight) {
                t = n, e.fastClickScrollParent = n;
                break
            }
            n = n.parentElement
        }
        while (n)
        }
    t && (t.fastClickLastScrollTop = t.scrollTop)
}, FastClick.prototype.getTargetElementFromEventTarget = function(e) {
    "use strict";
    return e.nodeType === Node.TEXT_NODE ? e.parentNode : e
}, FastClick.prototype.onTouchStart = function(e) {
    "use strict";
    var t, n, i;
    if (e.targetTouches.length > 1)
        return !0;
    if (t = this.getTargetElementFromEventTarget(e.target), n = e.targetTouches[0], deviceIsIOS) {
        if (i = window.getSelection(), i.rangeCount&&!i.isCollapsed)
            return !0;
        if (!deviceIsIOS4) {
            if (n.identifier === this.lastTouchIdentifier)
                return e.preventDefault(), !1;
            this.lastTouchIdentifier = n.identifier, this.updateScrollParent(t)
        }
    }
    return this.trackingClick=!0, this.trackingClickStart = e.timeStamp, this.targetElement = t, this.touchStartX = n.pageX, this.touchStartY = n.pageY, e.timeStamp - this.lastClickTime < this.tapDelay && e.preventDefault(), !0
}, FastClick.prototype.touchHasMoved = function(e) {
    "use strict";
    var t = e.changedTouches[0], n = this.touchBoundary;
    return Math.abs(t.pageX - this.touchStartX) > n || Math.abs(t.pageY - this.touchStartY) > n?!0 : !1
}, FastClick.prototype.onTouchMove = function(e) {
    "use strict";
    return this.trackingClick ? ((this.targetElement !== this.getTargetElementFromEventTarget(e.target) || this.touchHasMoved(e)) && (this.trackingClick=!1, this.targetElement = null), !0) : !0
}, FastClick.prototype.findControl = function(e) {
    "use strict";
    return void 0 !== e.control ? e.control : e.htmlFor ? document.getElementById(e.htmlFor) : e.querySelector("button, input:not([type=hidden]), keygen, meter, output, progress, select, textarea")
}, FastClick.prototype.onTouchEnd = function(e) {
    "use strict";
    var t, n, i, r, o, a = this.targetElement;
    if (!this.trackingClick)
        return !0;
    if (e.timeStamp - this.lastClickTime < this.tapDelay)
        return this.cancelNextClick=!0, !0;
    if (this.cancelNextClick=!1, this.lastClickTime = e.timeStamp, n = this.trackingClickStart, this.trackingClick=!1, this.trackingClickStart = 0, deviceIsIOSWithBadTarget && (o = e.changedTouches[0], a = document.elementFromPoint(o.pageX - window.pageXOffset, o.pageY - window.pageYOffset) || a, a.fastClickScrollParent = this.targetElement.fastClickScrollParent), i = a.tagName.toLowerCase(), "label" === i) {
        if (t = this.findControl(a)) {
            if (this.focus(a), deviceIsAndroid)
                return !1;
            a = t
        }
    } else if (this.needsFocus(a))
        return e.timeStamp - n > 100 || deviceIsIOS && window.top !== window && "input" === i ? (this.targetElement = null, !1) : (this.focus(a), this.sendClick(a, e), deviceIsIOS && "select" === i || (this.targetElement = null, e.preventDefault()), !1);
    return deviceIsIOS&&!deviceIsIOS4 && (r = a.fastClickScrollParent, r && r.fastClickLastScrollTop !== r.scrollTop)?!0 : (this.needsClick(a) || (e.preventDefault(), this.sendClick(a, e)), !1)
}, FastClick.prototype.onTouchCancel = function() {
    "use strict";
    this.trackingClick=!1, this.targetElement = null
}, FastClick.prototype.onMouse = function(e) {
    "use strict";
    return this.targetElement ? e.forwardedTouchEvent?!0 : e.cancelable && (!this.needsClick(this.targetElement) || this.cancelNextClick) ? (e.stopImmediatePropagation ? e.stopImmediatePropagation() : e.propagationStopped=!0, e.stopPropagation(), e.preventDefault(), !1) : !0 : !0
}, FastClick.prototype.onClick = function(e) {
    "use strict";
    var t;
    return this.trackingClick ? (this.targetElement = null, this.trackingClick=!1, !0) : "submit" === e.target.type && 0 === e.detail?!0 : (t = this.onMouse(e), t || (this.targetElement = null), t)
}, FastClick.prototype.destroy = function() {
    "use strict";
    var e = this.layer;
    deviceIsAndroid && (e.removeEventListener("mouseover", this.onMouse, !0), e.removeEventListener("mousedown", this.onMouse, !0), e.removeEventListener("mouseup", this.onMouse, !0)), e.removeEventListener("click", this.onClick, !0), e.removeEventListener("touchstart", this.onTouchStart, !1), e.removeEventListener("touchmove", this.onTouchMove, !1), e.removeEventListener("touchend", this.onTouchEnd, !1), e.removeEventListener("touchcancel", this.onTouchCancel, !1)
}, FastClick.notNeeded = function(e) {
    "use strict";
    var t, n;
    if ("undefined" == typeof window.ontouchstart)
        return !0;
    if (n =+ (/Chrome\/([0-9]+)/.exec(navigator.userAgent) || [, 0])[1]) {
        if (!deviceIsAndroid)
            return !0;
        if (t = document.querySelector("meta[name=viewport]")) {
            if ( - 1 !== t.content.indexOf("user-scalable=no"))
                return !0;
            if (n > 31 && document.documentElement.scrollWidth <= window.outerWidth)
                return !0
        }
    }
    return "none" === e.style.msTouchAction?!0 : !1
}, FastClick.attach = function(e, t) {
    "use strict";
    return new FastClick(e, t)
}, "function" == typeof define && "object" == typeof define.amd && define.amd ? define(function() {
    "use strict";
    return FastClick
}) : "undefined" != typeof module && module.exports ? (module.exports = FastClick.attach, module.exports.FastClick = FastClick) : window.FastClick = FastClick, !function(e, t, n, i) {
    "use strict";
    function r(e) {
        return ("string" == typeof e || e instanceof String) && (e = e.replace(/^['\\/"]+|(;\s?})+|['\\/"]+$/g, "")), e
    }
    var o = function(t) {
        for (var n = t.length, i = e("head"); n--;)
            0 === i.has("." + t[n]).length && i.append('<meta class="' + t[n] + '" />')
    };
    o(["foundation-mq-small", "foundation-mq-medium", "foundation-mq-large", "foundation-mq-xlarge", "foundation-mq-xxlarge", "foundation-data-attribute-namespace"]), e(function() {
        "undefined" != typeof FastClick && "undefined" != typeof n.body && FastClick.attach(n.body)
    });
    var a = function(t, i) {
        if ("string" == typeof t) {
            if (i) {
                var r;
                if (i.jquery) {
                    if (r = i[0], !r)
                        return i
                } else 
                    r = i;
                return e(r.querySelectorAll(t))
            }
            return e(n.querySelectorAll(t))
        }
        return e(t, i)
    }, s = function(e) {
        var t = [];
        return e || t.push("data"), this.namespace.length > 0 && t.push(this.namespace), t.push(this.name), t.join("-")
    }, l = function(e) {
        for (var t = e.split("-"), n = t.length, i = []; n--;)
            0 !== n ? i.push(t[n]) : this.namespace.length > 0 ? i.push(this.namespace, t[n]) : i.push(t[n]);
        return i.reverse().join("-")
    }, c = function(t, n) {
        var i = this, r=!a(this).data(this.attr_name(!0));
        return a(this.scope).is("[" + this.attr_name() + "]") ? (a(this.scope).data(this.attr_name(!0) + "-init", e.extend({}, this.settings, n || t, this.data_options(a(this.scope)))), r && this.events(this.scope)) : a("[" + this.attr_name() + "]", this.scope).each(function() {
            var r=!a(this).data(i.attr_name(!0) + "-init");
            a(this).data(i.attr_name(!0) + "-init", e.extend({}, i.settings, n || t, i.data_options(a(this)))), r && i.events(this)
        }), "string" == typeof t ? this[t].call(this, n) : void 0
    }, u = function(e, t) {
        function n() {
            t(e[0])
        }
        function i() {
            if (this.one("load", n), /MSIE (\d+\.\d+);/.test(navigator.userAgent)) {
                var e = this.attr("src"), t = e.match(/\?/) ? "&": "?";
                t += "random=" + (new Date).getTime(), this.attr("src", e + t)
            }
        }
        return e.attr("src") ? void(e[0].complete || 4 === e[0].readyState ? n() : i.call(e)) : void n()
    };
    t.matchMedia = t.matchMedia || function(e) {
        var t, n = e.documentElement, i = n.firstElementChild || n.firstChild, r = e.createElement("body"), o = e.createElement("div");
        return o.id = "mq-test-1", o.style.cssText = "position:absolute;top:-100em", r.style.background = "none", r.appendChild(o), function(e) {
            return o.innerHTML = '&shy;<style media="' + e + '"> #mq-test-1 { width: 42px; }</style>', n.insertBefore(r, i), t = 42 === o.offsetWidth, n.removeChild(r), {
                matches: t,
                media: e
            }
        }
    }(n), function() {
        function e() {
            n && (o(e), s && jQuery.fx.tick())
        }
        for (var n, i = 0, r = ["webkit", "moz"], o = t.requestAnimationFrame, a = t.cancelAnimationFrame, s = "undefined" != typeof jQuery.fx; i < r.length&&!o; i++)
            o = t[r[i] + "RequestAnimationFrame"], a = a || t[r[i] + "CancelAnimationFrame"] || t[r[i] + "CancelRequestAnimationFrame"];
        o ? (t.requestAnimationFrame = o, t.cancelAnimationFrame = a, s && (jQuery.fx.timer = function(t) {
            t() && jQuery.timers.push(t)&&!n && (n=!0, e())
        }, jQuery.fx.stop = function() {
            n=!1
        })) : (t.requestAnimationFrame = function(e) {
            var n = (new Date).getTime(), r = Math.max(0, 16 - (n - i)), o = t.setTimeout(function() {
                e(n + r)
            }, r);
            return i = n + r, o
        }, t.cancelAnimationFrame = function(e) {
            clearTimeout(e)
        })
    }(jQuery), t.Foundation = {
        name: "Foundation",
        version: "5.4.7",
        media_queries: {
            small: a(".foundation-mq-small").css("font-family").replace(/^[\/\\'"]+|(;\s?})+|[\/\\'"]+$/g, ""),
            medium: a(".foundation-mq-medium").css("font-family").replace(/^[\/\\'"]+|(;\s?})+|[\/\\'"]+$/g, ""),
            large: a(".foundation-mq-large").css("font-family").replace(/^[\/\\'"]+|(;\s?})+|[\/\\'"]+$/g, ""),
            xlarge: a(".foundation-mq-xlarge").css("font-family").replace(/^[\/\\'"]+|(;\s?})+|[\/\\'"]+$/g, ""),
            xxlarge: a(".foundation-mq-xxlarge").css("font-family").replace(/^[\/\\'"]+|(;\s?})+|[\/\\'"]+$/g, "")
        },
        stylesheet: e("<style></style>").appendTo("head")[0].sheet,
        global: {
            namespace: i
        },
        init: function(e, n, i, r, o) {
            var s = [e, i, r, o], l = [];
            if (this.rtl = /rtl/i.test(a("html").attr("dir")), this.scope = e || this.scope, this.set_namespace(), n && "string" == typeof n&&!/reflow/i.test(n))
                this.libs.hasOwnProperty(n) && l.push(this.init_lib(n, s));
            else 
                for (var c in this.libs)
                    l.push(this.init_lib(c, n));
            return a(t).load(function() {
                a(t).trigger("resize.fndtn.clearing").trigger("resize.fndtn.dropdown").trigger("resize.fndtn.equalizer").trigger("resize.fndtn.interchange").trigger("resize.fndtn.joyride").trigger("resize.fndtn.magellan").trigger("resize.fndtn.topbar").trigger("resize.fndtn.slider")
            }), e
        },
        init_lib: function(t, n) {
            return this.libs.hasOwnProperty(t) ? (this.patch(this.libs[t]), n && n.hasOwnProperty(t) ? ("undefined" != typeof this.libs[t].settings ? e.extend(!0, this.libs[t].settings, n[t]) : "undefined" != typeof this.libs[t].defaults && e.extend(!0, this.libs[t].defaults, n[t]), this.libs[t].init.apply(this.libs[t], [this.scope, n[t]])) : (n = n instanceof Array ? n : new Array(n), this.libs[t].init.apply(this.libs[t], n))) : function() {}
        },
        patch: function(e) {
            e.scope = this.scope, e.namespace = this.global.namespace, e.rtl = this.rtl, e.data_options = this.utils.data_options, e.attr_name = s, e.add_namespace = l, e.bindings = c, e.S = this.utils.S
        },
        inherit: function(e, t) {
            for (var n = t.split(" "), i = n.length; i--;)
                this.utils.hasOwnProperty(n[i]) && (e[n[i]] = this.utils[n[i]])
        },
        set_namespace: function() {
            var t = this.global.namespace === i ? e(".foundation-data-attribute-namespace").css("font-family"): this.global.namespace;
            this.global.namespace = t === i || /false/i.test(t) ? "" : t
        },
        libs: {},
        utils: {
            S: a,
            throttle: function(e, t) {
                var n = null;
                return function() {
                    var i = this, r = arguments;
                    null == n && (n = setTimeout(function() {
                        e.apply(i, r), n = null
                    }, t))
                }
            },
            debounce: function(e, t, n) {
                var i, r;
                return function() {
                    var o = this, a = arguments, s = function() {
                        i = null, n || (r = e.apply(o, a))
                    }, l = n&&!i;
                    return clearTimeout(i), i = setTimeout(s, t), l && (r = e.apply(o, a)), r
                }
            },
            data_options: function(t, n) {
                function i(e) {
                    return !isNaN(e - 0) && null !== e && "" !== e && e!==!1 && e!==!0
                }
                function r(t) {
                    return "string" == typeof t ? e.trim(t) : t
                }
                n = n || "options";
                var o, a, s, l = {}, c = function(e) {
                    var t = Foundation.global.namespace;
                    return e.data(t.length > 0 ? t + "-" + n : n)
                }, u = c(t);
                if ("object" == typeof u)
                    return u;
                for (s = (u || ":").split(";"), o = s.length; o--;)
                    a = s[o].split(":"), a = [a[0], a.slice(1).join(":")], /true/i.test(a[1]) && (a[1]=!0), /false/i.test(a[1]) && (a[1]=!1), i(a[1]) && (a[1] =- 1 === a[1].indexOf(".") ? parseInt(a[1], 10) : parseFloat(a[1])), 2 === a.length && a[0].length > 0 && (l[r(a[0])] = r(a[1]));
                return l
            },
            register_media: function(t, n) {
                Foundation.media_queries[t] === i && (e("head").append('<meta class="' + n + '"/>'), Foundation.media_queries[t] = r(e("." + n).css("font-family")))
            },
            add_custom_rule: function(e, t) {
                if (t === i && Foundation.stylesheet)
                    Foundation.stylesheet.insertRule(e, Foundation.stylesheet.cssRules.length);
                else {
                    var n = Foundation.media_queries[t];
                    n !== i && Foundation.stylesheet.insertRule("@media " + Foundation.media_queries[t] + "{ " + e + " }")
                }
            },
            image_loaded: function(e, t) {
                var n = this, i = e.length;
                0 === i && t(e), e.each(function() {
                    u(n.S(this), function() {
                        i -= 1, 0 === i && t(e)
                    })
                })
            },
            random_str: function() {
                return this.fidx || (this.fidx = 0), this.prefix = this.prefix || [this.name || "F", ( + new Date).toString(36)].join("-"), this.prefix + (this.fidx++).toString(36)
            }
        }
    }, e.fn.foundation = function() {
        var e = Array.prototype.slice.call(arguments, 0);
        return this.each(function() {
            return Foundation.init.apply(Foundation, [this].concat(e)), this
        })
    }
}(jQuery, window, window.document), function(e, t, n) {
    "use strict";
    Foundation.libs.abide = {
        name: "abide",
        version: "5.4.7",
        settings: {
            live_validate: !0,
            focus_on_invalid: !0,
            error_labels: !0,
            error_class: "error",
            timeout: 1e3,
            patterns: {
                alpha: /^[a-zA-Z]+$/,
                alpha_numeric: /^[a-zA-Z0-9]+$/,
                integer: /^[-+]?\d+$/,
                number: /^[-+]?\d*(?:[\.\,]\d+)?$/,
                card: /^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})$/,
                cvv: /^([0-9]){3,4}$/,
                email: /^[a-zA-Z0-9.!#$%&'*+\/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$/,
                url: /^(https?|ftp|file|ssh):\/\/(((([a-zA-Z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:)*@)?(((\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5])\.(\d|[1-9]\d|1\d\d|2[0-4]\d|25[0-5]))|((([a-zA-Z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-zA-Z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-zA-Z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-zA-Z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.)+(([a-zA-Z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-zA-Z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-zA-Z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-zA-Z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.?)(:\d*)?)(\/((([a-zA-Z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)+(\/(([a-zA-Z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)*)*)?)?(\?((([a-zA-Z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)|[\uE000-\uF8FF]|\/|\?)*)?(\#((([a-zA-Z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(%[\da-f]{2})|[!\$&'\(\)\*\+,;=]|:|@)|\/|\?)*)?$/,
                domain: /^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}$/,
                datetime: /^([0-2][0-9]{3})\-([0-1][0-9])\-([0-3][0-9])T([0-5][0-9])\:([0-5][0-9])\:([0-5][0-9])(Z|([\-\+]([0-1][0-9])\:00))$/,
                date: /(?:19|20)[0-9]{2}-(?:(?:0[1-9]|1[0-2])-(?:0[1-9]|1[0-9]|2[0-9])|(?:(?!02)(?:0[1-9]|1[0-2])-(?:30))|(?:(?:0[13578]|1[02])-31))$/,
                time: /^(0[0-9]|1[0-9]|2[0-3])(:[0-5][0-9]){2}$/,
                dateISO: /^\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}$/,
                month_day_year: /^(0[1-9]|1[012])[- \/.](0[1-9]|[12][0-9]|3[01])[- \/.]\d{4}$/,
                day_month_year: /^(0[1-9]|[12][0-9]|3[01])[- \/.](0[1-9]|1[012])[- \/.]\d{4}$/,
                color: /^#?([a-fA-F0-9]{6}|[a-fA-F0-9]{3})$/
            },
            validators: {
                equalTo: function(e) {
                    var t = n.getElementById(e.getAttribute(this.add_namespace("data-equalto"))).value, i = e.value, r = t === i;
                    return r
                }
            }
        },
        timer: null,
        init: function(e, t, n) {
            this.bindings(t, n)
        },
        events: function(t) {
            var n = this, i = n.S(t).attr("novalidate", "novalidate"), r = i.data(this.attr_name(!0) + "-init") || {};
            this.invalid_attr = this.add_namespace("data-invalid"), i.off(".abide").on("submit.fndtn.abide validate.fndtn.abide", function(e) {
                var t = /ajax/i.test(n.S(this).attr(n.attr_name()));
                return n.validate(n.S(this).find("input, textarea, select").get(), e, t)
            }).on("reset", function() {
                return n.reset(e(this))
            }).find("input, textarea, select").off(".abide").on("blur.fndtn.abide change.fndtn.abide", function(e) {
                n.validate([this], e)
            }).on("keydown.fndtn.abide", function(e) {
                r.live_validate===!0 && (clearTimeout(n.timer), n.timer = setTimeout(function() {
                    n.validate([this], e)
                }.bind(this), r.timeout))
            })
        },
        reset: function(t) {
            t.removeAttr(this.invalid_attr), e(this.invalid_attr, t).removeAttr(this.invalid_attr), e("." + this.settings.error_class, t).not("small").removeClass(this.settings.error_class)
        },
        validate: function(e, t, n) {
            for (var i = this.parse_patterns(e), r = i.length, o = this.S(e[0]).closest("form"), a = /submit/.test(t.type), s = 0; r > s; s++)
                if (!i[s] && (a || n))
                    return this.settings.focus_on_invalid && e[s].focus(), o.trigger("invalid"), this.S(e[s]).closest("form").attr(this.invalid_attr, ""), !1;
            return (a || n) && o.trigger("valid"), o.removeAttr(this.invalid_attr), n?!1 : !0
        },
        parse_patterns: function(e) {
            for (var t = e.length, n = []; t--;)
                n.push(this.pattern(e[t]));
            return this.check_validation_and_apply_styles(n)
        },
        pattern: function(e) {
            var t = e.getAttribute("type"), n = "string" == typeof e.getAttribute("required"), i = e.getAttribute("pattern") || "";
            return this.settings.patterns.hasOwnProperty(i) && i.length > 0 ? [e, this.settings.patterns[i], n] : i.length > 0 ? [e, new RegExp(i), n] : this.settings.patterns.hasOwnProperty(t) ? [e, this.settings.patterns[t], n] : (i = /.*/, [e, i, n])
        },
        check_validation_and_apply_styles: function(t) {
            var n = t.length, i = [], r = this.S(t[0][0]).closest("[data-" + this.attr_name(!0) + "]");
            for (r.data(this.attr_name(!0) + "-init") || {}; n--;) {
                var o, a, s = t[n][0], l = t[n][2], c = s.value.trim(), u = this.S(s).parent(), d = s.getAttribute(this.add_namespace("data-abide-validator")), f = "radio" === s.type, h = "checkbox" === s.type, p = this.S('label[for="' + s.getAttribute("id") + '"]'), g = l ? s.value.length > 0: !0, m = [];
                if (s.getAttribute(this.add_namespace("data-equalto")) && (d = "equalTo"), o = u.is("label") ? u.parent() : u, d && (a = this.settings.validators[d].apply(this, [s, l, o]), m.push(a)), f && l)
                    m.push(this.valid_radio(s, l));
                else if (h && l)
                    m.push(this.valid_checkbox(s, l));
                else if (m.push(t[n][1].test(c) && g ||!l && s.value.length < 1 || e(s).attr("disabled")?!0 : !1), m = [m.every(function(e) {
                    return e
                })], m[0])
                    this.S(s).removeAttr(this.invalid_attr), s.setAttribute("aria-invalid", "false"), s.removeAttribute("aria-describedby"), o.removeClass(this.settings.error_class), p.length > 0 && this.settings.error_labels && p.removeClass(this.settings.error_class).removeAttr("role"), e(s).triggerHandler("valid");
                else {
                    this.S(s).attr(this.invalid_attr, ""), s.setAttribute("aria-invalid", "true");
                    var v = o.find("small." + this.settings.error_class, "span." + this.settings.error_class), y = v.length > 0 ? v[0].id: "";
                    y.length > 0 && s.setAttribute("aria-describedby", y), o.addClass(this.settings.error_class), p.length > 0 && this.settings.error_labels && p.addClass(this.settings.error_class).attr("role", "alert"), e(s).triggerHandler("invalid")
                }
                i.push(m[0])
            }
            return i = [i.every(function(e) {
                return e
            })]
        },
        valid_checkbox: function(e, t) {
            var e = this.S(e), n = e.is(":checked") ||!t;
            return n ? e.removeAttr(this.invalid_attr).parent().removeClass(this.settings.error_class) : e.attr(this.invalid_attr, "").parent().addClass(this.settings.error_class), n
        },
        valid_radio: function(e) {
            for (var t = e.getAttribute("name"), n = this.S(e).closest("[data-" + this.attr_name(!0) + "]").find("[name='" + t + "']"), i = n.length, r=!1, o = 0; i > o; o++)
                n[o].checked && (r=!0);
            for (var o = 0; i > o; o++)
                r ? this.S(n[o]).removeAttr(this.invalid_attr).parent().removeClass(this.settings.error_class) : this.S(n[o]).attr(this.invalid_attr, "").parent().addClass(this.settings.error_class);
            return r
        },
        valid_equal: function(e, t, i) {
            var r = n.getElementById(e.getAttribute(this.add_namespace("data-equalto"))).value, o = e.value, a = r === o;
            return a ? (this.S(e).removeAttr(this.invalid_attr), i.removeClass(this.settings.error_class), label.length > 0 && settings.error_labels && label.removeClass(this.settings.error_class)) : (this.S(e).attr(this.invalid_attr, ""), i.addClass(this.settings.error_class), label.length > 0 && settings.error_labels && label.addClass(this.settings.error_class)), a
        },
        valid_oneof: function(e, t, n, i) {
            var e = this.S(e), r = this.S("[" + this.add_namespace("data-oneof") + "]"), o = r.filter(":checked").length > 0;
            if (o ? e.removeAttr(this.invalid_attr).parent().removeClass(this.settings.error_class) : e.attr(this.invalid_attr, "").parent().addClass(this.settings.error_class), !i) {
                var a = this;
                r.each(function() {
                    a.valid_oneof.call(a, this, null, null, !0)
                })
            }
            return o
        }
    }
}(jQuery, window, window.document), function(e, t) {
    "use strict";
    Foundation.libs.dropdown = {
        name: "dropdown",
        version: "5.4.7",
        settings: {
            active_class: "open",
            disabled_class: "disabled",
            mega_class: "mega",
            align: "bottom",
            is_hover: !1,
            opened: function() {},
            closed: function() {}
        },
        init: function(e, t, n) {
            Foundation.inherit(this, "throttle"), this.bindings(t, n)
        },
        events: function() {
            var n = this, i = n.S;
            i(this.scope).off(".dropdown").on("click.fndtn.dropdown", "[" + this.attr_name() + "]", function(t) {
                var r = i(this).data(n.attr_name(!0) + "-init") || n.settings;
                (!r.is_hover || Modernizr.touch) && (t.preventDefault(), n.toggle(e(this)))
            }).on("mouseenter.fndtn.dropdown", "[" + this.attr_name() + "], [" + this.attr_name() + "-content]", function(e) {
                var t, r, o = i(this);
                clearTimeout(n.timeout), o.data(n.data_attr()) ? (t = i("#" + o.data(n.data_attr())), r = o) : (t = o, r = i("[" + n.attr_name() + "='" + t.attr("id") + "']"));
                var a = r.data(n.attr_name(!0) + "-init") || n.settings;
                i(e.target).data(n.data_attr()) && a.is_hover && n.closeall.call(n), a.is_hover && n.open.apply(n, [t, r])
            }).on("mouseleave.fndtn.dropdown", "[" + this.attr_name() + "], [" + this.attr_name() + "-content]", function() {
                var e = i(this);
                n.timeout = setTimeout(function() {
                    if (e.data(n.data_attr())) {
                        var t = e.data(n.data_attr(!0) + "-init") || n.settings;
                        t.is_hover && n.close.call(n, i("#" + e.data(n.data_attr())))
                    } else {
                        var r = i("[" + n.attr_name() + '="' + i(this).attr("id") + '"]'), t = r.data(n.attr_name(!0) + "-init") || n.settings;
                        t.is_hover && n.close.call(n, e)
                    }
                }.bind(this), 150)
            }).on("click.fndtn.dropdown", function(t) {
                var r = i(t.target).closest("[" + n.attr_name() + "-content]");
                return i(t.target).closest("[" + n.attr_name() + "]").length > 0 ? void 0 : !i(t.target).data("revealId") && r.length > 0 && (i(t.target).is("[" + n.attr_name() + "-content]") || e.contains(r.first()[0], t.target)) ? void t.stopPropagation() : void n.close.call(n, i("[" + n.attr_name() + "-content]"))
            }).on("opened.fndtn.dropdown", "[" + n.attr_name() + "-content]", function() {
                n.settings.opened.call(this)
            }).on("closed.fndtn.dropdown", "[" + n.attr_name() + "-content]", function() {
                n.settings.closed.call(this)
            }), i(t).off(".dropdown").on("resize.fndtn.dropdown", n.throttle(function() {
                n.resize.call(n)
            }, 50)), this.resize()
        },
        close: function(t) {
            var n = this;
            t.each(function() {
                var i = e("[" + n.attr_name() + "=" + t[0].id + "]") || e("aria-controls=" + t[0].id + "]");
                i.attr("aria-expanded", "false"), n.S(this).hasClass(n.settings.active_class) && (n.S(this).css(Foundation.rtl ? "right" : "left", "-99999px").attr("aria-hidden", "true").removeClass(n.settings.active_class).prev("[" + n.attr_name() + "]").removeClass(n.settings.active_class).removeData("target"), n.S(this).trigger("closed").trigger("closed.fndtn.dropdown", [t]))
            })
        },
        closeall: function() {
            var t = this;
            e.each(t.S("[" + this.attr_name() + "-content]"), function() {
                t.close.call(t, t.S(this))
            })
        },
        open: function(e, t) {
            this.css(e.addClass(this.settings.active_class), t), e.prev("[" + this.attr_name() + "]").addClass(this.settings.active_class), e.data("target", t.get(0)).trigger("opened").trigger("opened.fndtn.dropdown", [e, t]), e.attr("aria-hidden", "false"), t.attr("aria-expanded", "true"), e.focus()
        },
        data_attr: function() {
            return this.namespace.length > 0 ? this.namespace + "-" + this.name : this.name
        },
        toggle: function(e) {
            if (!e.hasClass(this.settings.disabled_class)) {
                var t = this.S("#" + e.data(this.data_attr()));
                0 !== t.length && (this.close.call(this, this.S("[" + this.attr_name() + "-content]").not(t)), t.hasClass(this.settings.active_class) ? (this.close.call(this, t), t.data("target") !== e.get(0) && this.open.call(this, t, e)) : this.open.call(this, t, e))
            }
        },
        resize: function() {
            var e = this.S("[" + this.attr_name() + "-content].open"), t = this.S("[" + this.attr_name() + "='" + e.attr("id") + "']");
            e.length && t.length && this.css(e, t)
        },
        css: function(e, t) {
            var n = Math.max((t.width() - e.width()) / 2, 8), i = t.data(this.attr_name(!0) + "-init") || this.settings;
            if (this.clear_idx(), this.small()) {
                var r = this.dirs.bottom.call(e, t, i);
                e.attr("style", "").removeClass("drop-left drop-right drop-top").css({
                    position: "absolute",
                    width: "95%",
                    "max-width": "none",
                    top: r.top
                }), e.css(Foundation.rtl ? "right" : "left", n)
            } else 
                this.style(e, t, i);
            return e
        },
        style: function(t, n, i) {
            var r = e.extend({
                position: "absolute"
            }, this.dirs[i.align].call(t, n, i));
            t.attr("style", "").css(r)
        },
        dirs: {
            _base: function(e) {
                var t = this.offsetParent(), n = t.offset(), i = e.offset();
                return i.top -= n.top, i.left -= n.left, i
            },
            top: function(e, t) {
                var n = Foundation.libs.dropdown, i = n.dirs._base.call(this, e);
                return this.addClass("drop-top"), (e.outerWidth() < this.outerWidth() || n.small() || this.hasClass(t.mega_menu)) && n.adjust_pip(this, e, t, i), Foundation.rtl ? {
                    left: i.left - this.outerWidth() + e.outerWidth(),
                    top: i.top - this.outerHeight()
                } : {
                    left: i.left,
                    top: i.top - this.outerHeight()
                }
            },
            bottom: function(e, t) {
                var n = Foundation.libs.dropdown, i = n.dirs._base.call(this, e);
                return (e.outerWidth() < this.outerWidth() || n.small() || this.hasClass(t.mega_menu)) && n.adjust_pip(this, e, t, i), n.rtl ? {
                    left: i.left - this.outerWidth() + e.outerWidth(),
                    top: i.top + e.outerHeight()
                } : {
                    left: i.left,
                    top: i.top + e.outerHeight()
                }
            },
            left: function(e) {
                var t = Foundation.libs.dropdown.dirs._base.call(this, e);
                return this.addClass("drop-left"), {
                    left: t.left - this.outerWidth(),
                    top: t.top
                }
            },
            right: function(e) {
                var t = Foundation.libs.dropdown.dirs._base.call(this, e);
                return this.addClass("drop-right"), {
                    left: t.left + e.outerWidth(),
                    top: t.top
                }
            }
        },
        adjust_pip: function(e, t, n, i) {
            var r = Foundation.stylesheet, o = 8;
            e.hasClass(n.mega_class) ? o = i.left + t.outerWidth() / 2 - 8 : this.small() && (o += i.left - 8), this.rule_idx = r.cssRules.length;
            var a = ".f-dropdown.open:before", s = ".f-dropdown.open:after", l = "left: " + o + "px;", c = "left: " + (o - 1) + "px;";
            r.insertRule ? (r.insertRule([a, "{", l, "}"].join(" "), this.rule_idx), r.insertRule([s, "{", c, "}"].join(" "), this.rule_idx + 1)) : (r.addRule(a, l, this.rule_idx), r.addRule(s, c, this.rule_idx + 1))
        },
        clear_idx: function() {
            var e = Foundation.stylesheet;
            "undefined" != typeof this.rule_idx && (e.deleteRule(this.rule_idx), e.deleteRule(this.rule_idx), delete this.rule_idx)
        },
        small: function() {
            return matchMedia(Foundation.media_queries.small).matches&&!matchMedia(Foundation.media_queries.medium).matches
        },
        off: function() {
            this.S(this.scope).off(".fndtn.dropdown"), this.S("html, body").off(".fndtn.dropdown"), this.S(t).off(".fndtn.dropdown"), this.S("[data-dropdown-content]").off(".fndtn.dropdown")
        },
        reflow: function() {}
    }
}(jQuery, window, window.document), function(e, t) {
    "use strict";
    Foundation.libs.equalizer = {
        name: "equalizer",
        version: "5.4.7",
        settings: {
            use_tallest: !0,
            before_height_change: e.noop,
            after_height_change: e.noop,
            equalize_on_stack: !1
        },
        init: function(e, t, n) {
            Foundation.inherit(this, "image_loaded"), this.bindings(t, n), this.reflow()
        },
        events: function() {
            this.S(t).off(".equalizer").on("resize.fndtn.equalizer", function() {
                this.reflow()
            }.bind(this))
        },
        equalize: function(t) {
            var n=!1, i = t.find("[" + this.attr_name() + "-watch]:visible"), r = t.data(this.attr_name(!0) + "-init");
            if (0 !== i.length) {
                var o = i.first().offset().top;
                if (r.before_height_change(), t.trigger("before-height-change").trigger("before-height-change.fndth.equalizer"), i.height("inherit"), i.each(function() {
                    var t = e(this);
                    t.offset().top !== o && (n=!0)
                }), r.equalize_on_stack!==!1 ||!n) {
                    var a = i.map(function() {
                        return e(this).outerHeight(!1)
                    }).get();
                    if (r.use_tallest) {
                        var s = Math.max.apply(null, a);
                        i.css("height", s)
                    } else {
                        var l = Math.min.apply(null, a);
                        i.css("height", l)
                    }
                    r.after_height_change(), t.trigger("after-height-change").trigger("after-height-change.fndtn.equalizer")
                }
            }
        },
        reflow: function() {
            var t = this;
            this.S("[" + this.attr_name() + "]", this.scope).each(function() {
                var n = e(this);
                t.image_loaded(t.S("img", this), function() {
                    t.equalize(n)
                })
            })
        }
    }
}(jQuery, window, window.document), function(e, t) {
    "use strict";
    Foundation.libs["magellan-expedition"] = {
        name: "magellan-expedition",
        version: "5.4.7",
        settings: {
            active_class: "active",
            threshold: 0,
            destination_threshold: 20,
            throttle_delay: 30,
            fixed_top: 0
        },
        init: function(e, t, n) {
            Foundation.inherit(this, "throttle"), this.bindings(t, n)
        },
        events: function() {
            var n = this, i = n.S, r = n.settings;
            n.set_expedition_position(), i(n.scope).off(".magellan").on("click.fndtn.magellan", "[" + n.add_namespace("data-magellan-arrival") + '] a[href^="#"]', function(t) {
                t.preventDefault();
                var i = e(this).closest("[" + n.attr_name() + "]"), r = i.data("magellan-expedition-init"), o = this.hash.split("#").join(""), a = e("a[name='" + o + "']");
                0 === a.length && (a = e("#" + o));
                var s = a.offset().top - r.destination_threshold + 1;
                s -= i.outerHeight(), e("html, body").stop().animate({
                    scrollTop: s
                }, 700, "swing", function() {
                    history.pushState ? history.pushState(null, null, "#" + o) : location.hash = "#" + o
                })
            }).on("scroll.fndtn.magellan", n.throttle(this.check_for_arrivals.bind(this), r.throttle_delay)), e(t).on("resize.fndtn.magellan", n.throttle(this.set_expedition_position.bind(this), r.throttle_delay))
        },
        check_for_arrivals: function() {
            var e = this;
            e.update_arrivals(), e.update_expedition_positions()
        },
        set_expedition_position: function() {
            var t = this;
            e("[" + this.attr_name() + "=fixed]", t.scope).each(function() {
                var n, i, r = e(this), o = r.data("magellan-expedition-init"), a = r.attr("styles");
                r.attr("style", ""), n = r.offset().top + o.threshold, i = parseInt(r.data("magellan-fixed-top")), isNaN(i) || (t.settings.fixed_top = i), r.data(t.data_attr("magellan-top-offset"), n), r.attr("style", a)
            })
        },
        update_expedition_positions: function() {
            var n = this, i = e(t).scrollTop();
            e("[" + this.attr_name() + "=fixed]", n.scope).each(function() {
                var t = e(this), r = t.data("magellan-expedition-init"), o = t.attr("style"), a = t.data("magellan-top-offset");
                if (i + n.settings.fixed_top >= a) {
                    var s = t.prev("[" + n.add_namespace("data-magellan-expedition-clone") + "]");
                    0 === s.length && (s = t.clone(), s.removeAttr(n.attr_name()), s.attr(n.add_namespace("data-magellan-expedition-clone"), ""), t.before(s)), t.css({
                        position: "fixed",
                        top: r.fixed_top
                    }).addClass("fixed")
                } else 
                    t.prev("[" + n.add_namespace("data-magellan-expedition-clone") + "]").remove(), t.attr("style", o).css("position", "").css("top", "").removeClass("fixed")
            })
        },
        update_arrivals: function() {
            var n = this, i = e(t).scrollTop();
            e("[" + this.attr_name() + "]", n.scope).each(function() {
                var t = e(this), r = t.data(n.attr_name(!0) + "-init"), o = n.offsets(t, i), a = t.find("[" + n.add_namespace("data-magellan-arrival") + "]"), s=!1;
                o.each(function(e, i) {
                    if (i.viewport_offset >= i.top_offset) {
                        var o = t.find("[" + n.add_namespace("data-magellan-arrival") + "]");
                        return o.not(i.arrival).removeClass(r.active_class), i.arrival.addClass(r.active_class), s=!0, !0
                    }
                }), s || a.removeClass(r.active_class)
            })
        },
        offsets: function(t, n) {
            var i = this, r = t.data(i.attr_name(!0) + "-init"), o = n;
            return t.find("[" + i.add_namespace("data-magellan-arrival") + "]").map(function() {
                var n = e(this).data(i.data_attr("magellan-arrival")), a = e("[" + i.add_namespace("data-magellan-destination") + "=" + n + "]");
                if (a.length > 0) {
                    var s = Math.floor(a.offset().top - r.destination_threshold - t.outerHeight());
                    return {
                        destination: a,
                        arrival: e(this),
                        top_offset: s,
                        viewport_offset: o
                    }
                }
            }).sort(function(e, t) {
                return e.top_offset < t.top_offset?-1 : e.top_offset > t.top_offset ? 1 : 0
            })
        },
        data_attr: function(e) {
            return this.namespace.length > 0 ? this.namespace + "-" + e : e
        },
        off: function() {
            this.S(this.scope).off(".magellan"), this.S(t).off(".magellan")
        },
        reflow: function() {
            var t = this;
            e("[" + t.add_namespace("data-magellan-expedition-clone") + "]", t.scope).remove()
        }
    }
}(jQuery, window, window.document), function(e, t, n, i) {
    "use strict";
    Foundation.libs.tab = {
        name: "tab",
        version: "5.4.7",
        settings: {
            active_class: "active",
            callback: function() {},
            deep_linking: !1,
            scroll_to_content: !0,
            is_hover: !1
        },
        default_tab_hashes: [],
        init: function(e, t, n) {
            var i = this, r = this.S;
            this.bindings(t, n), this.handle_location_hash_change(), r("[" + this.attr_name() + "] > .active > a", this.scope).each(function() {
                i.default_tab_hashes.push(this.hash)
            })
        },
        events: function() {
            var e = this, n = this.S, i = function(t) {
                var i = n(this).closest("[" + e.attr_name() + "]").data(e.attr_name(!0) + "-init");
                (!i.is_hover || Modernizr.touch) && (t.preventDefault(), t.stopPropagation(), e.toggle_active_tab(n(this).parent()))
            };
            n(this.scope).off(".tab").on("focus.fndtn.tab", "[" + this.attr_name() + "] > * > a", i).on("click.fndtn.tab", "[" + this.attr_name() + "] > * > a", i).on("mouseenter.fndtn.tab", "[" + this.attr_name() + "] > * > a", function() {
                var t = n(this).closest("[" + e.attr_name() + "]").data(e.attr_name(!0) + "-init");
                t.is_hover && e.toggle_active_tab(n(this).parent())
            }), n(t).on("hashchange.fndtn.tab", function(t) {
                t.preventDefault(), e.handle_location_hash_change()
            })
        },
        handle_location_hash_change: function() {
            var t = this, n = this.S;
            n("[" + this.attr_name() + "]", this.scope).each(function() {
                var r = n(this).data(t.attr_name(!0) + "-init");
                if (r.deep_linking) {
                    var o;
                    if (o = r.scroll_to_content ? t.scope.location.hash : t.scope.location.hash.replace("fndtn-", ""), "" != o) {
                        var a = n(o);
                        if (a.hasClass("content") && a.parent().hasClass("tabs-content"))
                            t.toggle_active_tab(e("[" + t.attr_name() + "] > * > a[href=" + o + "]").parent());
                        else {
                            var s = a.closest(".content").attr("id");
                            s != i && t.toggle_active_tab(e("[" + t.attr_name() + "] > * > a[href=#" + s + "]").parent(), o)
                        }
                    } else 
                        for (var l = 0; l < t.default_tab_hashes.length; l++)
                            t.toggle_active_tab(e("[" + t.attr_name() + "] > * > a[href=" + t.default_tab_hashes[l] + "]").parent())
                }
            })
        },
        toggle_active_tab: function(r, o) {
            var a = this.S, s = r.closest("[" + this.attr_name() + "]"), l = r.find("a"), c = r.children("a").first(), u = "#" + c.attr("href").split("#")[1], d = a(u), f = r.siblings(), h = s.data(this.attr_name(!0) + "-init"), p = function(t) {
                var i, r = e(this), o = e(this).parents("li").prev().children('[role="tab"]'), a = e(this).parents("li").next().children('[role="tab"]');
                switch (t.keyCode) {
                case 37:
                    i = o;
                    break;
                case 39:
                    i = a;
                    break;
                default:
                    i=!1
                }
                i.length && (r.attr({
                    tabindex: "-1",
                    "aria-selected": null
                }), i.attr({
                    tabindex: "0",
                    "aria-selected": !0
                }).focus()), e('[role="tabpanel"]').attr("aria-hidden", "true"), e("#" + e(n.activeElement).attr("href").substring(1)).attr("aria-hidden", null)
            };
            a(this).data(this.data_attr("tab-content")) && (u = "#" + a(this).data(this.data_attr("tab-content")).split("#")[1], d = a(u)), h.deep_linking && (h.scroll_to_content ? (t.location.hash = o || u, o == i || o == u ? r.parent()[0].scrollIntoView() : a(u)[0].scrollIntoView()) : t.location.hash = o != i ? "fndtn-" + o.replace("#", "") : "fndtn-" + u.replace("#", "")), r.addClass(h.active_class).triggerHandler("opened"), l.attr({
                "aria-selected": "true",
                tabindex: 0
            }), f.removeClass(h.active_class), f.find("a").attr({
                "aria-selected": "false",
                tabindex: - 1
            }), d.siblings().removeClass(h.active_class).attr({
                "aria-hidden": "true",
                tabindex: - 1
            }), d.addClass(h.active_class).attr("aria-hidden", "false").removeAttr("tabindex"), h.callback(r), d.triggerHandler("toggled", [r]), s.triggerHandler("toggled", [d]), l.off("keydown").on("keydown", p)
        },
        data_attr: function(e) {
            return this.namespace.length > 0 ? this.namespace + "-" + e : e
        },
        off: function() {},
        reflow: function() {}
    }
}(jQuery, window, window.document), function(e, t, n) {
    "use strict";
    Foundation.libs.topbar = {
        name: "topbar",
        version: "5.4.7",
        settings: {
            index: 0,
            sticky_class: "sticky",
            custom_back_text: !0,
            back_text: "Back",
            mobile_show_parent_link: !0,
            is_hover: !0,
            scrolltop: !0,
            sticky_on: "all"
        },
        init: function(t, n, i) {
            Foundation.inherit(this, "add_custom_rule register_media throttle");
            var r = this;
            r.register_media("topbar", "foundation-mq-topbar"), this.bindings(n, i), r.S("[" + this.attr_name() + "]", this.scope).each(function() {
                var t = e(this), n = t.data(r.attr_name(!0) + "-init");
                r.S("section, .top-bar-section", this), t.data("index", 0);
                var i = t.parent();
                i.hasClass("fixed") || r.is_sticky(t, i, n) ? (r.settings.sticky_class = n.sticky_class, r.settings.sticky_topbar = t, t.data("height", i.outerHeight()), t.data("stickyoffset", i.offset().top)) : t.data("height", t.outerHeight()), n.assembled || r.assemble(t), n.is_hover ? r.S(".has-dropdown", t).addClass("not-click") : r.S(".has-dropdown", t).removeClass("not-click"), r.add_custom_rule(".f-topbar-fixed { padding-top: " + t.data("height") + "px }"), i.hasClass("fixed") && r.S("body").addClass("f-topbar-fixed")
            })
        },
        is_sticky: function(e, t, n) {
            var i = t.hasClass(n.sticky_class);
            return i && "all" === n.sticky_on?!0 : i && this.small() && "small" === n.sticky_on ? matchMedia(Foundation.media_queries.small).matches&&!matchMedia(Foundation.media_queries.medium).matches&&!matchMedia(Foundation.media_queries.large).matches : i && this.medium() && "medium" === n.sticky_on ? matchMedia(Foundation.media_queries.small).matches && matchMedia(Foundation.media_queries.medium).matches&&!matchMedia(Foundation.media_queries.large).matches : i && this.large() && "large" === n.sticky_on ? matchMedia(Foundation.media_queries.small).matches && matchMedia(Foundation.media_queries.medium).matches && matchMedia(Foundation.media_queries.large).matches : !1
        },
        toggle: function(n) {
            var i, r = this;
            i = n ? r.S(n).closest("[" + this.attr_name() + "]") : r.S("[" + this.attr_name() + "]");
            var o = i.data(this.attr_name(!0) + "-init"), a = r.S("section, .top-bar-section", i);
            r.breakpoint() && (r.rtl ? (a.css({
                right: "0%"
            }), e(">.name", a).css({
                right: "100%"
            })) : (a.css({
                left: "0%"
            }), e(">.name", a).css({
                left: "100%"
            })), r.S("li.moved", a).removeClass("moved"), i.data("index", 0), i.toggleClass("expanded").css("height", "")), o.scrolltop ? i.hasClass("expanded") ? i.parent().hasClass("fixed") && (o.scrolltop ? (i.parent().removeClass("fixed"), i.addClass("fixed"), r.S("body").removeClass("f-topbar-fixed"), t.scrollTo(0, 0)) : i.parent().removeClass("expanded")) : i.hasClass("fixed") && (i.parent().addClass("fixed"), i.removeClass("fixed"), r.S("body").addClass("f-topbar-fixed")) : (r.is_sticky(i, i.parent(), o) && i.parent().addClass("fixed"), i.parent().hasClass("fixed") && (i.hasClass("expanded") ? (i.addClass("fixed"), i.parent().addClass("expanded"), r.S("body").addClass("f-topbar-fixed")) : (i.removeClass("fixed"), i.parent().removeClass("expanded"), r.update_sticky_positioning())))
        },
        timer: null,
        events: function() {
            var n = this, i = this.S;
            i(this.scope).off(".topbar").on("click.fndtn.topbar", "[" + this.attr_name() + "] .toggle-topbar", function(e) {
                e.preventDefault(), n.toggle(this)
            }).on("click.fndtn.topbar", '.top-bar .top-bar-section li a[href^="#"],[' + this.attr_name() + '] .top-bar-section li a[href^="#"]', function() {
                var t = e(this).closest("li");
                !n.breakpoint() || t.hasClass("back") || t.hasClass("has-dropdown") || n.toggle()
            }).on("click.fndtn.topbar", "[" + this.attr_name() + "] li.has-dropdown", function(t) {
                var r = i(this), o = i(t.target), a = r.closest("[" + n.attr_name() + "]"), s = a.data(n.attr_name(!0) + "-init");
                return o.data("revealId") ? void n.toggle() : void(n.breakpoint() || (!s.is_hover || Modernizr.touch) && (t.stopImmediatePropagation(), r.hasClass("hover") ? (r.removeClass("hover").find("li").removeClass("hover"), r.parents("li.hover").removeClass("hover")) : (r.addClass("hover"), e(r).siblings().removeClass("hover"), "A" === o[0].nodeName && o.parent().hasClass("has-dropdown") && t.preventDefault())))
            }).on("click.fndtn.topbar", "[" + this.attr_name() + "] .has-dropdown>a", function(e) {
                if (n.breakpoint()) {
                    e.preventDefault();
                    var t = i(this), r = t.closest("[" + n.attr_name() + "]"), o = r.find("section, .top-bar-section"), a = (t.next(".dropdown").outerHeight(), t.closest("li"));
                    r.data("index", r.data("index") + 1), a.addClass("moved"), n.rtl ? (o.css({
                        right: - (100 * r.data("index")) + "%"
                    }), o.find(">.name").css({
                        right: 100 * r.data("index") + "%"
                    })) : (o.css({
                        left: - (100 * r.data("index")) + "%"
                    }), o.find(">.name").css({
                        left: 100 * r.data("index") + "%"
                    })), r.css("height", t.siblings("ul").outerHeight(!0) + r.data("height"))
                }
            }), i(t).off(".topbar").on("resize.fndtn.topbar", n.throttle(function() {
                n.resize.call(n)
            }, 50)).trigger("resize").trigger("resize.fndtn.topbar").load(function() {
                i(this).trigger("resize.fndtn.topbar")
            }), i("body").off(".topbar").on("click.fndtn.topbar", function(e) {
                var t = i(e.target).closest("li").closest("li.hover");
                t.length > 0 || i("[" + n.attr_name() + "] li.hover").removeClass("hover")
            }), i(this.scope).on("click.fndtn.topbar", "[" + this.attr_name() + "] .has-dropdown .back", function(e) {
                e.preventDefault();
                var t = i(this), r = t.closest("[" + n.attr_name() + "]"), o = r.find("section, .top-bar-section"), a = (r.data(n.attr_name(!0) + "-init"), t.closest("li.moved")), s = a.parent();
                r.data("index", r.data("index") - 1), n.rtl ? (o.css({
                    right: - (100 * r.data("index")) + "%"
                }), o.find(">.name").css({
                    right: 100 * r.data("index") + "%"
                })) : (o.css({
                    left: - (100 * r.data("index")) + "%"
                }), o.find(">.name").css({
                    left: 100 * r.data("index") + "%"
                })), 0 === r.data("index") ? r.css("height", "") : r.css("height", s.outerHeight(!0) + r.data("height")), setTimeout(function() {
                    a.removeClass("moved")
                }, 300)
            }), i(this.scope).find(".dropdown a").focus(function() {
                e(this).parents(".has-dropdown").addClass("hover")
            }).blur(function() {
                e(this).parents(".has-dropdown").removeClass("hover")
            })
        },
        resize: function() {
            var e = this;
            e.S("[" + this.attr_name() + "]").each(function() {
                var t, i = e.S(this), r = i.data(e.attr_name(!0) + "-init"), o = i.parent("." + e.settings.sticky_class);
                if (!e.breakpoint()) {
                    var a = i.hasClass("expanded");
                    i.css("height", "").removeClass("expanded").find("li").removeClass("hover"), a && e.toggle(i)
                }
                e.is_sticky(i, o, r) && (o.hasClass("fixed") ? (o.removeClass("fixed"), t = o.offset().top, e.S(n.body).hasClass("f-topbar-fixed") && (t -= i.data("height")), i.data("stickyoffset", t), o.addClass("fixed")) : (t = o.offset().top, i.data("stickyoffset", t)))
            })
        },
        breakpoint: function() {
            return !matchMedia(Foundation.media_queries.topbar).matches
        },
        small: function() {
            return matchMedia(Foundation.media_queries.small).matches
        },
        medium: function() {
            return matchMedia(Foundation.media_queries.medium).matches
        },
        large: function() {
            return matchMedia(Foundation.media_queries.large).matches
        },
        assemble: function(t) {
            var n = this, i = t.data(this.attr_name(!0) + "-init"), r = n.S("section, .top-bar-section", t);
            r.detach(), n.S(".has-dropdown>a", r).each(function() {
                var t, r = n.S(this), o = r.siblings(".dropdown"), a = r.attr("href");
                o.find(".title.back").length || (t = e(1 == i.mobile_show_parent_link && a ? '<li class="title back js-generated"><h5><a href="javascript:void(0)"></a></h5></li><li class="parent-link show-for-small"><a class="parent-link js-generated" href="' + a + '">' + r.html() + "</a></li>" : '<li class="title back js-generated"><h5><a href="javascript:void(0)"></a></h5>'), e("h5>a", t).html(1 == i.custom_back_text ? i.back_text : "&laquo; " + r.html()), o.prepend(t))
            }), r.appendTo(t), this.sticky(), this.assembled(t)
        },
        assembled: function(t) {
            t.data(this.attr_name(!0), e.extend({}, t.data(this.attr_name(!0)), {
                assembled: !0
            }))
        },
        height: function(t) {
            var n = 0, i = this;
            return e("> li", t).each(function() {
                n += i.S(this).outerHeight(!0)
            }), n
        },
        sticky: function() {
            var e = this;
            this.S(t).on("scroll", function() {
                e.update_sticky_positioning()
            })
        },
        update_sticky_positioning: function() {
            var e = "." + this.settings.sticky_class, n = this.S(t), i = this;
            if (i.settings.sticky_topbar && i.is_sticky(this.settings.sticky_topbar, this.settings.sticky_topbar.parent(), this.settings)) {
                var r = this.settings.sticky_topbar.data("stickyoffset");
                i.S(e).hasClass("expanded") || (n.scrollTop() > r ? i.S(e).hasClass("fixed") || (i.S(e).addClass("fixed"), i.S("body").addClass("f-topbar-fixed")) : n.scrollTop() <= r && i.S(e).hasClass("fixed") && (i.S(e).removeClass("fixed"), i.S("body").removeClass("f-topbar-fixed")))
            }
        },
        off: function() {
            this.S(this.scope).off(".fndtn.topbar"), this.S(t).off(".fndtn.topbar")
        },
        reflow: function() {}
    }
}(jQuery, window, window.document), !function(e, t, n, i) {
    var r = e(t);
    e.fn.lazyload = function(o) {
        function a() {
            var t = 0;
            l.each(function() {
                var n = e(this);
                if (!c.skip_invisible || n.is(":visible"))
                    if (e.abovethetop(this, c) || e.leftofbegin(this, c));
                else if (e.belowthefold(this, c) || e.rightoffold(this, c)) {
                    if (++t > c.failure_limit)
                        return !1
                } else 
                    n.trigger("appear"), t = 0
            })
        }
        var s, l = this, c = {
            threshold: 0,
            failure_limit: 0,
            event: "scroll",
            effect: "show",
            container: t,
            data_attribute: "original",
            skip_invisible: !0,
            appear: null,
            load: null,
            placeholder: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsQAAA7EAZUrDhsAAAANSURBVBhXYzh8+PB/AAffA0nNPuCLAAAAAElFTkSuQmCC"
        };
        return o && (i !== o.failurelimit && (o.failure_limit = o.failurelimit, delete o.failurelimit), i !== o.effectspeed && (o.effect_speed = o.effectspeed, delete o.effectspeed), e.extend(c, o)), s = c.container === i || c.container === t ? r : e(c.container), 0 === c.event.indexOf("scroll") && s.bind(c.event, function() {
            return a()
        }), this.each(function() {
            var t = this, n = e(t);
            t.loaded=!1, (n.attr("src") === i || n.attr("src")===!1) && n.attr("src", c.placeholder), n.one("appear", function() {
                if (!this.loaded) {
                    if (c.appear) {
                        var i = l.length;
                        c.appear.call(t, i, c)
                    }
                    e("<img />").bind("load", function() {
                        var i = n.data(c.data_attribute);
                        n.hide(), n.is("img") ? n.attr("src", i) : n.css("background-image", "url('" + i + "')"), n[c.effect](c.effect_speed), t.loaded=!0;
                        var r = e.grep(l, function(e) {
                            return !e.loaded
                        });
                        if (l = e(r), c.load) {
                            var o = l.length;
                            c.load.call(t, o, c)
                        }
                    }).attr("src", n.data(c.data_attribute))
                }
            }), 0 !== c.event.indexOf("scroll") && n.bind(c.event, function() {
                t.loaded || n.trigger("appear")
            })
        }), r.bind("resize", function() {
            a()
        }), /iphone|ipod|ipad.*os 5/gi.test(navigator.appVersion) && r.bind("pageshow", function(t) {
            t.originalEvent && t.originalEvent.persisted && l.each(function() {
                e(this).trigger("appear")
            })
        }), e(n).ready(function() {
            a()
        }), this
    }, e.belowthefold = function(n, o) {
        var a;
        return a = o.container === i || o.container === t ? (t.innerHeight ? t.innerHeight : r.height()) + r.scrollTop() : e(o.container).offset().top + e(o.container).height(), a <= e(n).offset().top - o.threshold
    }, e.rightoffold = function(n, o) {
        var a;
        return a = o.container === i || o.container === t ? r.width() + r.scrollLeft() : e(o.container).offset().left + e(o.container).width(), a <= e(n).offset().left - o.threshold
    }, e.abovethetop = function(n, o) {
        var a;
        return a = o.container === i || o.container === t ? r.scrollTop() : e(o.container).offset().top, a >= e(n).offset().top + o.threshold + e(n).height()
    }, e.leftofbegin = function(n, o) {
        var a;
        return a = o.container === i || o.container === t ? r.scrollLeft() : e(o.container).offset().left, a >= e(n).offset().left + o.threshold + e(n).width()
    }, e.inviewport = function(t, n) {
        return !(e.rightoffold(t, n) || e.leftofbegin(t, n) || e.belowthefold(t, n) || e.abovethetop(t, n))
    }, e.extend(e.expr[":"], {
        "below-the-fold": function(t) {
            return e.belowthefold(t, {
                threshold: 0
            })
        },
        "above-the-top": function(t) {
            return !e.belowthefold(t, {
                threshold: 0
            })
        },
        "right-of-screen": function(t) {
            return e.rightoffold(t, {
                threshold: 0
            })
        },
        "left-of-screen": function(t) {
            return !e.rightoffold(t, {
                threshold: 0
            })
        },
        "in-viewport": function(t) {
            return e.inviewport(t, {
                threshold: 0
            })
        },
        "above-the-fold": function(t) {
            return !e.belowthefold(t, {
                threshold: 0
            })
        },
        "right-of-fold": function(t) {
            return e.rightoffold(t, {
                threshold: 0
            })
        },
        "left-of-fold": function(t) {
            return !e.rightoffold(t, {
                threshold: 0
            })
        }
    })
}(jQuery, window, document), slideshare_object.SS_LI_FillAd = function(e) {
    var t = function(e) {
        var t, n = e.split("?"), i = n[0], r = [], o = n[1];
        if ("undefined" != typeof o) {
            r = o.split("&");
            for (var a = r.length - 1; a >= 0; a -= 1)
                t = r[a].split("=")[0], /\S+@\S+\.\S+/.test(r[a]) && r.splice(a, 1);
            i = i + "?" + r.join("&")
        }
        return i
    }, n = function() {
        var t = "<iframe ";
        return t += ' width="' + (o > 1 ? o : "100%") + '"', t += ' height="' + (r > 1 ? r : "100%") + '"', t += ' marginheight="0" marginwidth="0" topmargin="0" leftmargin="0" allowtransparency="true" frameborder="0" scrolling="no" ', t += e.delayedAd ? 'source="' + p + '"></iframe>' : 'src="' + p + '"></iframe>'
    }, i = t(location.href), r = e.height || 0, o = e.width || 0, a = (e.tile || "", e.zone || "", e.dart_code || ""), s = e.custom_params || "", l = escape(";dc_ref=" + i), c = e.width + "x" + e.height, u = document.getElementById(e.appendTo), d = "&_x=" + escape(s._x || "") + a + l, f = "&_rx=" + escape(";adsense=t") + escape(s._rx || "") + a;
    f += l, slideshare_object.slideshow && "14748437" === slideshare_object.slideshow.id && (f += escape(";adTest=true")), f += escape(";$DCOPT;ord=$ORD");
    var h = i, p = "http://www.linkedin.com/csp/dtag?sz=" + c;
    p += "&dc_ref=" + encodeURIComponent(h), p += "&ti=" + e.tile, p += "&z=" + e.zone, p += "&p=5", a && (p += d), p += f, u ? u.innerHTML += n() : document.write(n())
}, slideshare_object.delayedLIAd = function(e) {
    e.delayedAd=!0, slideshare_object.SS_LI_FillAd(e)
}, slideshare_object.loadDelayedLIAd = function(e, t) {
    var n = e.clone(!0);
    e.replaceWith(n), n.attr("src", n.attr("source")), window.setTimeout(function() {
        t && t.removeClass("hide")
    }, 1e3)
}, slideshare_object.processAdQueue = function() {
    adQ = slideshare_object._adQueue;
    var e = function(e) {
        window.setTimeout(function() {
            e.removeClass("hide")
        }, 1e3)
    };
    for (var t in adQ)
        if (adQ.hasOwnProperty(t)) {
            var n = adQ[t];
            n.lazyLoad ? slideshare_object.delayedLIAd(n) : (slideshare_object.SS_LI_FillAd(n), e($("#" + n.appendTo).next(".j-cta_remove_ad_fancy")))
        }
}, function(e, t) {
    e.facebook_activity_queue = [], e.setup_fbconnect_interaction = function() {
        function n(t) {
            return e.user.is_opengraph_user && e.user.opengraph_permissions[t] && e.user.fb_access_token
        }
        function i(t) {
            var n = t.ppt_url || location_without_params(), i = {};
            switch (t.name) {
            case"view":
            case"favorite":
                i.presentation = n;
                break;
            case"download":
                var r = getUrlVar("s_title"), o = getUrlVar("user_login");
                i.presentation = "http://www.slideshare.net/" + o + "/" + r;
                break;
            case"comment":
                i.presentation = n, i.message = t.message;
                break;
            case"follow":
                i.profile = t.facebook_user_id;
                break;
            default:
                error("Unknown action: " + t.name)
            }
            return i.access_token = e.user.fb_access_token, i
        }
        function r(t) {
            return "follow" === t ? "/me/og.follows" : "/me/" + e.fb_app_name + ":" + t
        }
        function o(e) {
            var t = getUrlVar("fb_source");
            getUrlVar("code") || t ? t ? slideshare_object.ga("fb_opengraph", "publish_action", "from_facebook_" + e + "_" + t) : slideshare_object.ga("fb_opengraph", "publish_action", "from_facebook_" + e) : slideshare_object.ga("fb_opengraph", "publish_action", e)
        }
        function a(e, n) {
            var i = escape(window.location.href), r = window.fromSource, o = window.fromType || getUrlVar("from");
            switch (!window.location.pathname.startsWith("/upload") || 1 !== e.status && 2 !== e.status && 3 !== e.status || (_gaq.push(["_trackEvent", "Upload", "fb_login_successful"]), i = i.replace("loggedout_", "loggedin_loggedout_")), e.status) {
            case 1:
                _gaq.push(["_trackEvent", "Signup", "fb_login", "fb_signup_from_" + n]), window.location.replace("undefined" != typeof r ? "/fbconnect/landingpage?from=" + r : "/fbconnect/landingpage?from=" + i);
                break;
            case 2:
                window.location.replace("undefined" != typeof r ? "/fbconnect/landingpage?from=" + r : "/fbconnect/landingpage?from=" + i);
                break;
            case 3:
                _gaq.push(["_trackEvent", "Login", "fb_login", "fb_login_from_" + n]);
                var a = decodeURIComponent(r);
                if (void 0 !== getUrlVars().from_logout && "undefined" == typeof o)window.location.replace("/");
                else {
                    if (window.location.pathname.startsWith("/upload"))
                        return window.location.replace(window.location.href.replace("loggedout_", "loggedin_loggedout_")), !1;
                    r && r.length > 1 && isInternalRedirect(r) && window.location.href !== a ? window.location.href = a : window.location.reload()
                }
                break;
            case 4:
                t("#login_modal").modal("hide"), t("#page-error span").html("Login unsuccessful. Please try again."), t("#page-error").show();
                break;
            case 5:
                t("#login_modal").modal("hide"), t("#page-error span").html("Login unsuccessful. Your account has been suspended as it was found to be in violation of SlideShare's Terms of Service and/or Community Guidelines. Please read the <a href='http://help.slideshare.com/entries/22330620-Why-was-my-slideshare-account-suspended-' target='_blank' class='blue_link_bold'>Suspended Users FAQ</a> to resolve this issue."), t("#page-error").show();
                break;
            default:
                window.location.reload()
            }
        }
        function s(e) {
            switch (t("#j-fb-topnav-indicator").hide(), t("#j-fb-login-indicator").hide(), e.status) {
            case h.PROFILE_LINKED:
                window.location.reload();
                break;
            case h.USER_ALREADY_EXISTS:
                t("#page-error").append("We could not connect your SlideShare account to Facebook. This Facebook login is already being used with another SlideShare account.").show();
                break;
            case h.NO_PUBLISH_PERMISSIONS:
                t("#page-error").append("We could not connect your SlideShare account to Facebook. Sharing on SlideShare is better when you allow us to post to Facebook.").show(), t("#fb-login") && t("#fb-login").html("<strong>Connect</strong>").removeClass("disabled");
                break;
            default:
                t("#page-error").append("There was an error.").show()
            }
        }
        function l(e) {
            e.status === h.USER_ALREADY_EXISTS ? t("#page-error").append("This Facebook login is already being used with another SlideShare account. Try again using another account.").show() : window.location.reload()
        }
        function c() {
            window.location.reload()
        }
        function u() {
            if (window.localStorage)
                try {
                    var e = "fbErrorDismissals", t = Number(window.localStorage[e]) || 0;
                    window.localStorage[e] = String(t + 1), window.localStorage.fbErrorLastDismissal = (new Date).getTime()
            } catch (n) {}
        }
        function d() {
            if (!window.localStorage)
                return !0;
            try {
                var e = "fbErrorDismissals", t = Number(window.localStorage[e]) || 0;
                if (0 === t)
                    return !1;
                if (t >= p.MAX_NUMBER_OF_DISMISSALS)
                    return !0;
                var n = window.localStorage.fbErrorLastDismissal || (new Date).getTime(), i = (new Date).getTime(), r = Math.ceil((i - n) / 864e5);
                return r <= p.DAYS_UNTIL_NEXT_NOTICE
            } catch (o) {
                return !0
            }
        }
        var f = {
            INVALID_ACCESS_TOKEN: 190,
            REQUIRES_EXTENDED_PERMISSIONS: 200
        }, h = {
            PROFILE_LINKED: 1,
            NOT_LOGGED_IN: 2,
            USER_ALREADY_EXISTS: 3,
            NO_PUBLISH_PERMISSIONS: 4
        }, p = {
            DAYS_UNTIL_NEXT_NOTICE: 3,
            MAX_NUMBER_OF_DISMISSALS: 2
        };
        slideshare_object.user.loggedin || t("body").on("click", "#fb-login-modalbox", function() {
            window.fromSource = window.fromSource || getUrlVar("from_source");
            var e = "fancybox";
            t("#j-fb-modalbox-indicator").show();
            var n = slideshare_object.fbconnect.basic_perms;
            "undefined" != typeof thisLink && is_abbr(thisLink) && (n = slideshare_object.fbconnect.download_perms), slideshare_object.unsubscribeFBStatusChange(), window.FB && FB.login(function(n) {
                n && n.authResponse ? "undefined" != typeof thisLink && is_abbr(thisLink) ? slideshare_object.fbConnect("/fbconnect/create_account_or_login?from_page=download", {
                    access_token: n.authResponse.accessToken,
                    fb_user_id: n.authResponse.userID,
                    login_source: window.loginSource
                }, function(t) {
                    a(t, e)
                }) : slideshare_object.fbConnect("/fbconnect/create_account_or_login", {
                    access_token: n.authResponse.accessToken,
                    fb_user_id: n.authResponse.userID,
                    login_source: window.loginSource
                }, function(t) {
                    a(t, e)
                }) : t("#j-fb-modalbox-indicator").hide()
            }, n)
        }), e.pushPendingFBactivities = function() {
            for (var e = slideshare_object.facebook_activity_queue; e.length;) {
                var t = e.pop();
                slideshare_object.push_activity_to_facebook(t.action, t.callback)
            }
        }, e.post_activity_to_facebook = function(t, i) {
            return n(t.name) ? window.FB ? void e.push_activity_to_facebook(t, i) : void e.facebook_activity_queue.push({
                action : t, callback : i
            }) : void i()
        }, e.push_activity_to_facebook = function(e, n) {
            var a = r(e.name);
            "view" === e.name && (t(".fb-msg").addClass("hide"), t(".fb-activity-ss, .fb-activity-ss .progress").removeClass("hide")), FB.api(a, "post", i(e), function(e) {
                e && e.error && (e.error.code === f.REQUIRES_EXTENDED_PERMISSIONS ? t.post("/fbconnect/remove_publish_perms") : e.error.code === f.INVALID_ACCESS_TOKEN && t.post("/fbconnect/invalidate_access_token"), slideshare_object.ga("fb_opengraph_error", e.error.type + "-" + e.error.code, e.error.message, slideshare_object.user.id)), n(e)
            }), o(e.name)
        }, e.remove_activity_from_facebook = function(e, n) {
            t(".fb-msg").addClass("hide"), t(".fb-activity-ss .progress").removeClass("hide"), FB.api("/" + e, "delete", {
                access_token: slideshare_object.user.fb_access_token
            }, n)
        }, t(document).ready(function() {
            if (_gaq.push(["_setAccount", "UA-2330466-1"]), _gaq.push(["_setDomainName", ".slideshare.net"]), _gaq.push(["_addIgnoredRef", "slideshare.net"]), t(document).bind("fbinitialized", function() {
                t("#fb-login").removeClass("disabled")
            }), t("#j-delink-fb").on("click", function(e) {
                e.preventDefault(), t.post("/fbconnect/delink_facebook_profile", {}, c, "json")
            }), (slideshare_object.isHomePage() || slideshare_object.isSlideViewPage())&&!slideshare_object.is_mobile && slideshare_object.user && slideshare_object.user.is_opengraph_user&&!slideshare_object.user.fb_access_token&&!d()) {
                var e = "Your facebook connection has expired. <a href='#' id='fb-reconnect'>Click here to connect and continue sharing.</a>", n = t("#page-notice");
                n.removeClass("alert").addClass("notification").find("span").html(e), n.find(".closeThis").addClass("fb-notification-close"), n.slideDown("slow"), slideshare_object.ga("fb_opengraph", "reconnect-notice", "displayed"), t(".fb-notification-close").click(function() {
                    u(), slideshare_object.ga("fb_opengraph", "reconnect-notice", "cancelled")
                })
            }
            t("#fb-reconnect").click(function(e) {
                if (e.preventDefault(), slideshare_object.user.loggedin && slideshare_object.user.is_opengraph_user&&!slideshare_object.user.fb_access_token) {
                    var n = slideshare_object.fbconnect.download_perms;
                    n.return_scopes=!0, slideshare_object.unsubscribeFBStatusChange(), window.FB && FB.login(function(e) {
                        e && e.authResponse && t.post("/fbconnect/link_facebook_profile", {
                            access_token: e.authResponse.accessToken,
                            fb_user_id: e.authResponse.userID,
                            granted_perms: e.authResponse.grantedScopes
                        }, function(e) {
                            l(e)
                        }, "json")
                    }, n), slideshare_object.ga("fb_opengraph", "reconnect-notice", "connected")
                }
            }), t("#fb-login").click(function(e) {
                if (!t(this).hasClass("disabled")) {
                    var n = "login";
                    e.preventDefault(), window.fromSource = getUrlVar("from_source") || encodeURIComponent("/"), t("#j-fb-login-indicator").show(), t(this).hasClass("fb-login-new") || t(this).html("Connecting...").addClass("disabled");
                    var i = slideshare_object.fbconnect.basic_perms;
                    slideshare_object.user.loggedin && (i = slideshare_object.fbconnect.download_perms), i.return_scopes=!0, slideshare_object.unsubscribeFBStatusChange(), window.FB && FB.login(function(e) {
                        e && e.authResponse ? slideshare_object.user.loggedin ? slideshare_object.user.is_fbuser && slideshare_object.user.is_opengraph_user && slideshare_object.user.fb_access_token || t.post("/fbconnect/link_facebook_profile", {
                            access_token: e.authResponse.accessToken,
                            fb_user_id: e.authResponse.userID,
                            granted_perms: e.authResponse.grantedScopes
                        }, function(e) {
                            s(e)
                        }, "json") : t.post("/fbconnect/fb_login", {}, function() {
                            slideshare_object.fbConnect("/fbconnect/create_account_or_login", {
                                access_token: e.authResponse.accessToken,
                                fb_user_id: e.authResponse.userID,
                                login_source: window.loginSource,
                                granted_perms: e.authResponse.grantedScopes
                            }, function(e) {
                                a(e, n)
                            })
                        }, "json") : t("#j-fb-login-indicator").hide()
                    }, i)
                }
            })
        })
    }, e.setup_opengraph_interaction = function() {
        t(".player").bind("fb-view-published", function(n, i) {
            t(".fb-activity-ss .progress").addClass("hide"), e.slideshow.view_action_state = "published", t(".fb-removed").addClass("hide"), t(".fb-viewed").removeClass("hide"), e.slideshow.view_action_id = i, "true" !== cookie("og_notified") && t(".fb-viewed").popover({
                placement: "top",
                html: !0,
                trigger: "manual"
            }).popover("show")
        }), t(".player").bind("fb-view-failed", function(e, n) {
            slideshare_object.ga("fb_opengraph", "view-publish-failed", n.id, n.message), slideshare_object.slideshow.view_action_state = "failed", t(".fb-activity-ss .progress").addClass("hide")
        }), t(".player").bind("fb-favorite-failed", function(e, t) {
            slideshare_object.ga("fb_opengraph", "favorite-publish-failed", t.id || t.type, t.message)
        }), t("#j-popover-okay").on("click", function(e) {
            e.preventDefault(), cookie("og_notified", !0, {
                path: "/",
                expires: 365
            }), t(".fb-viewed").popover("hide")
        }), t(".fb-viewed").on("click", function(n) {
            n.preventDefault(), "published" != e.slideshow.view_action_state || null === e.slideshow.view_action_id && void 0 === e.slideshow.view_action_id || e.remove_activity_from_facebook(e.slideshow.view_action_id, function(e) {
                e&&!e.error && (t(".fb-activity-ss .progress").addClass("hide"), t(".player").trigger("fb-view-removed"))
            })
        }), t(".player").bind("fb-view-removed", function() {
            t(".fb-activity-ss .progress").addClass("hide"), e.slideshow.view_action_state = "removed", e.slideshow.view_action_id = null, t(".fb-viewed").addClass("hide"), t(".fb-removed").removeClass("hide"), slideshare_object.ga("fb_opengraph", "fb-view-removed")
        }), t(".fb-removed").on("click", function(n) {
            n.preventDefault(), e.post_activity_to_facebook({
                name: "view"
            }, function(e) {
                void 0 !== e && e.id && t(".player").trigger("fb-view-published", e.id)
            })
        })
    }
}(slideshare_object, jQuery);
var mobile_util = {};
mobile_util.portraitWidth = function() {
    return Math.min(window.innerHeight, window.innerWidth)
}, mobile_util.portraitHeight = function() {
    return Math.max(window.innerHeight, window.innerWidth)
}, mobile_util.portraitMode = function() {
    return window.innerWidth < window.innerHeight
}, mobile_util.landscapeMode = function() {
    return !this.portraitMode()
}, mobile_util.deviceAspectRatio = function() {
    return this.portraitWidth() / this.portraitHeight()
}, mobile_util.isIOS = function() {
    return /(iPad|iPhone|iPod)/i.test(navigator.userAgent)
}, mobile_util.isIOSWebView = function() {
    return this.isIOS&&!/Safari/i.test(navigator.userAgent)
}, mobile_util.isIOS8 = function() {
    return this.isIOS() && /OS 8/.test(navigator.userAgent)
}, mobile_util.isIPad = function() {
    return this.isIOS() && .75 == this.deviceAspectRatio()
}, mobile_util.isAndroid = function() {
    return /Android/.test(navigator.userAgent)
}, mobile_util.isAndroidWebView = function() {
    return this.isAndroid() && /Version\/(\d+\.)+(\d+)/.test(navigator.userAgent)
}, mobile_util.isAndroidTablet = function() {
    return !/Mobile/.test(navigator.userAgent) && this.isAndroid() && this.portraitWidth() >= 600
}, mobile_util.isAndroidPhone = function() {
    return this.isAndroid()&&!this.isAndroidTablet()
}, mobile_util.isMobile = function() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
}, mobile_util.hasCookie = function(e) {
    return document.cookie.indexOf(e) >= 0
}, mobile_util.setCookie = function(e, t, n) {
    t = t || 0, n = n || 7;
    var i = new Date;
    i.setDate(i.getDate() + n), document.cookie = e + "=" + t + "; expires=" + i + "; path=/"
}, mobile_util.getCookie = function(e) {
    var t = new RegExp(e + "=.[^;]*"), n = document.cookie.match(t);
    return n ? n[0].split("=")[1] : !1
}, mobile_util.lazyloadImages = function(e, t) {
    t = t || 0;
    var n = function() {
        $(e).lazyload({
            threshold: t
        })
    };
    "undefined" != typeof slideshare_mobile_object && slideshare_mobile_object.windowLoaded ? n() : $(window).load(function() {
        n()
    })
}, mobile_util.trackGAEvent = function(e, t, n) {
    var i = window._gaq || [];
    i.push(["_trackEvent", e, t, n])
}, mobile_util.addSpinnerTo = function(e, t, n) {
    if ("undefined" != typeof e) {
        n = n || "#FFFFFF";
        var i = {
            S: "fa-lg",
            M: "fa-2x",
            L: "fa-3x",
            XL: "fa-4x"
        };
        spinnerSize = i[t.toUpperCase()] || i.M;
        var r = $(e), o = $('<i class="fa fa-spinner fa-spin"></i>');
        o.prependTo(r).addClass(spinnerSize).css({
            color: n,
            position: "absolute",
            "z-index": 1e3,
            top: "50%",
            transform: "translate(-50%, -50%)",
            left: "50%"
        })
    }
}, mobile_util.removeSpinnersFrom = function(e) {
    "undefined" != typeof e && $(e).find(".fa-spinner").remove()
}, function(e) {
    var t, n, i, r, o = 1, a = this, s=!1, l = "postMessage", c = "addEventListener", u = a[l]&&!window.opera;
    e[l] = function(t, n, i) {
        n && (t = "string" == typeof t ? t : e.param(t), i = i || parent, u ? i[l](t, n.replace(/([^:]+:\/\/[^\/]+).*/, "$1")) : n && (i.location = n.replace(/#.*$/, "") + "#" + + new Date + o++ + "&" + t))
    }, e.receiveMessage = r = function(o, l, d) {
        u ? (o && (i && r(), i = function(t) {
            return "string" == typeof l && t.origin !== l || e.isFunction(l) && l(t.origin) === s ? s : void o(t)
        }), a[c] ? a[o ? c: "removeEventListener"]("message", i, s) : a[o ? "attachEvent": "detachEvent"]("onmessage", i)) : (t && clearInterval(t), t = null, o && (d = "number" == typeof l ? l : "number" == typeof d ? d : 100, t = setInterval(function() {
            var e = document.location.hash, t = /^#?\d+&/;
            e !== n && t.test(e) && (n = e, o({
                data: e.replace(t, "")
            }))
        }, d)))
    }
}(jQuery), function(e, t) {
    e.extend = function(n, i, r) {
        for (var o in i)
            n[o] = i[o];
        return e.dev && (n.__noSuchMethod__ = function(e, t) {
            error(e, " : no such method exists", t)
        }), t.isFunction(r) && r(), n
    }, e.extend(window, {
        log: e.dev && window.console ? function() {
            try {
                console.log.apply(console, arguments)
            } catch (e) {}
        }
        : function() {},
        error: e.dev && window.console ? function() {
            try {
                console.error.apply(console, arguments)
            } catch (e) {}
        }
        : function() {},
        dir: e.dev && window.console ? function(e) {
            try {
                console.dir(e)
            } catch (t) {}
        }
        : function() {},
        info: e.dev && window.console ? function(e) {
            try {
                console.info(e)
            } catch (t) {}
        }
        : function() {},
        getUrlVars: function(e) {
            e = e ? e : window.location.href;
            for (var t, n = [], i = e.slice(e.indexOf("?") + 1).split("&"), r = 0; r < i.length; r++)
                t = i[r].split("="), n.push(t[0]), n[t[0]] = t[1];
            return n
        },
        getUrlVar: function(e, t) {
            return this.getUrlVars(t)[e]
        },
        cookie: function(e, t, n) {
            if ("undefined" == typeof t) {
                var i = null;
                if (document.cookie && "" !== document.cookie)
                    for (var r = document.cookie.split(";"), o = 0; o < r.length; o++) {
                        var a = jQuery.trim(r[o]);
                        if (a.substring(0, e.length + 1) == e + "=") {
                            i = decodeURIComponent(a.substring(e.length + 1));
                            break
                        }
                    }
                return i
            }
            n = n || {}, null === t && (t = "", n.expires =- 1);
            var s = "";
            if (n.expires && ("number" == typeof n.expires || n.expires.toUTCString)) {
                var l;
                "number" == typeof n.expires ? (l = new Date, l.setTime(l.getTime() + 24 * n.expires * 60 * 60 * 1e3)) : l = n.expires, s = "; expires=" + l.toUTCString()
            }
            var c = n.path ? "; path=" + n.path: "", u = n.domain ? "; domain=" + n.domain: "", d = n.secure ? "; secure": "";
            document.cookie = [e, "=", encodeURIComponent(t), s, c, u, d].join("")
        },
        isBrowserMSIE: function() {
            if ("Microsoft Internet Explorer" == navigator.appName) {
                var e = navigator.userAgent, t = new RegExp("MSIE ([0-9]{1,}[.0-9]{0,})");
                if (null != t.exec(e))
                    return !0
            }
            return !1
        }
    }), e.extend(e, {
        ga: function(e, t, n, i, r) {
            r = r ||!0, window._gaq.push(["_trackEvent", e, t, n, i, r])
        },
        isHomePage: function() {
            return "/" == window.location.pathname
        },
        setLanguageSelector: function() {
            try {
                var e = window.location.href.match(/:\/\/([^.]*)/)[1];
                t(".j-languages-selector .j-" + e).addClass("active")
            } catch (n) {
                t(".j-languages-selector .j-www").addClass("active")
            }
        },
        addAfterLoginEvent: function(t) {
            var n = JSON.stringify(t);
            e.cookie("after_login_action", n, {
                path: "/"
            })
        },
        add_signin_link: function(e) {
            t(e).each(function() {
                var e = "/login";
                if (e += "/" === document.location.pathname ? "?from_source=" + encodeURIComponent(document.location.href) + "&login_source=homepage.popup.like&from=favorite" : "?from_source=" + encodeURIComponent(document.location.href), t(this).hasClass("void_fancybox"))
                    t(this).hasClass("mobile") ? t(this).attr("href", "/mobile" + e) : t(this).attr("href", e);
                else {
                    var n = {
                        href: e,
                        "data-target": "#login_modal"
                    };
                    t(this).attr(n)
                }
            })
        },
        add_login_source: function(e, n) {
            t(e).each(function() {
                var e = t(this).attr("href") + "&login_source=" + n;
                t(this).attr("href", e)
            })
        },
        bind_favorites: function(e) {
            var n = ".j-favorited, .j-favorite";
            t(e).delegate(n, "click", n, slideshare_object.favorites_handler)
        },
        favorites_handler: function(e) {
            var n = t(this);
            if (slideshare_object.user.loggedin) {
                e.preventDefault(), e.stopPropagation();
                var i=!n.hasClass("favorited"), r = n.data("ss-url"), o = i ? slideshare_object.favorites.create_url : slideshare_object.favorites.delete_url, a = {
                    element: t(this),
                    url: o,
                    selectors: e.data,
                    event_category: "",
                    is_favorited: i,
                    ss_url: r,
                    data: {
                        slideshow_id: n.attr("data-ssid"),
                        user_id: slideshare_object.user.id,
                        source: "slideviewer",
                        response_type: "json"
                    }
                };
                slideshare_object.favorites_call_handler(a)
            } else {
                var s = {
                    event: "like",
                    data: {
                        slideshow_id: n.attr("data-ssid")
                    }
                };
                slideshare_object.addAfterLoginEvent(s)
            }
        },
        toggleLikeButton: function(e, n) {
            t(n).toggleClass("favorited")
        },
        favorites_call_handler: function(e) {
            var n = e.is_favorited ? "favorite_click": "unfavorite_click";
            window._gaq.push(["_trackEvent", e.event_category, n]);
            var i = (e.is_favorited ? "unfavorited" : "favorited", function() {
                var n = e.element.closest(".iso_slideshow");
                n && n.length > 0 ? t.each(n.find(e.selectors), slideshare_object.toggleLikeButton) : t.each(e.element.parent().find(e.selectors), slideshare_object.toggleLikeButton)
            });
            i(), t.ajax({
                url: e.url,
                data: e.data,
                dataType: "json",
                type: "POST",
                success: function(t) {
                    if (t.success) {
                        e.is_favorited && (slideshare_object.post_activity_to_facebook && slideshare_object.post_activity_to_facebook({
                            name: "favorite",
                            ppt_url: e.ss_url
                        }, function() {}), slideshare_object.isHomePage() && slideshare_object.prompt_connect_li_and_ss());
                        var n = e.is_favorited ? "favorite_click_success": "unfavorite_click_success";
                        window._gaq.push(["_trackEvent", e.event_category, n])
                    } else 
                        i()
                },
                error: function() {
                    i()
                }
            })
        },
        favoritesStatusUpdate: function() {
            var e = [];
            favoriteElements = t(".j-favorite"), favoriteElements.each(function() {
                e.push(t(this).attr("data-ssid"))
            }), t.ajax({
                type: "GET",
                url: slideshare_object.favorites && slideshare_object.favorites.user_favorites ? slideshare_object.favorites.user_favorites: "/favorite/get_favorites",
                data: "slideshow_id=" + e.join(","),
                dataType: "json",
                success: function(e) {
                    if (e.favorited.length > 0) {
                        var n = "[data-ssid=" + e.favorited.join("],[data-ssid=") + "]";
                        favoriteElements.filter(n).each(function() {
                            t.each(t(this).parent().children(".favorite-cta, .j-favorite.action-btn, .j-favorited.action-btn"), function(e, n) {
                                t(n).addClass("favorited")
                            })
                        })
                    }
                }
            })
        },
        lazyloadThumbnails: function() {
            var e = function() {
                t(".j-lazy-thumb").lazyload({
                    threshold: 100,
                    container: t("body .wrapper")
                }).removeClass("j-lazy-thumb")
            };
            slideshare_object.windowLoaded && e(), t(window).load(function() {
                slideshare_object.windowLoaded=!0, e()
            })
        },
        autosuggestTop: function() {
            var e = t("#nav-search form"), n = t("#nav-search-query"), i = t("#nav-search .search-suggestions");
            t("body").on("submit", "#nav-search form", function() {
                n.blur(), i.hide()
            }), t("body").on("focus", "#nav-search-query", function() {
                i.is(":empty") || i.show()
            }), t("body").on("click", "#nav-search .search-suggestions li", function() {
                n.val(t(this).text()), i.hide(), e.submit()
            }), t("body").on("focusout", "#nav-search-query", function() {
                setTimeout(function() {
                    i.hide()
                }, 100)
            }), t("body").on("mousemove, mouseover", "#nav-search .search-suggestions li", function() {
                i.children("li").removeClass("hovered"), t(this).addClass("hovered")
            }).on("mouseout", "#nav-search .search-suggestions", function() {
                i.children("li").removeClass("hovered")
            });
            var r = function() {
                var e = 0;
                return function(t, n) {
                    clearTimeout(e), e = setTimeout(t, n)
                }
            }();
            t("body").on("keyup", "#nav-search-query", function(e) {
                r(function() {
                    if (38 === parseInt(e.keyCode, 10) || 40 === parseInt(e.keyCode, 10)) {
                        var n =- 1;
                        40 === parseInt(e.keyCode, 10) && (n = 1);
                        var r = i.children("li.suggestion"), o = t(".search-suggestions li.suggestion.hovered"), a = (r.index(o) + n)%r.length, s = i.children("li.suggestion").eq(a);
                        return r.removeClass("hovered"), s.addClass("hovered"), void t("#nav-search-query").val(s.text())
                    }
                    if (37 !== parseInt(e.keyCode, 10) && 39 !== parseInt(e.keyCode, 10)) {
                        var l = t("#nav-search-query").val(), c = l.replace(/[\(\)\[\]\!\#\~'"*&]/gi, "").replace(/\s{2,}/g, " ").trim(c).replace(/ /g, "?");
                        if (0 === c.length)
                            return void i.empty().hide();
                        if ( - 1 === t.inArray("site_search", slideshare_object.feature_flag) && (c += "*", c += "&rows=5&wt=json&sort=frequency%20desc", c += "&fq=%2Bresults%3A%5B10%20TO%20*%5D&json.wrf=?", c.length > 2)) {
                            var u = "//autosuggest.slideshare.net/?q=" + c;
                            t.getJSON(u, function(e) {
                                i.empty(), e.response.docs.length > 0 ? (l = t.trim(l).replace(/[^\w\s]/g, "").replace(/\s{2,}/g, " "), t.each(e.response.docs, function(t, n) {
                                    var r = n.query;
                                    r = r.replace(l, ""), i.append(r == n.query ? "<li class='suggestion'><a>" + r + "</a></li>" : "<li class='suggestion'><a><b>" + l + "</b>" + r + "</a></li>"), t !== e.response.docs.length - 1 && i.append("<li class='divider'></li>")
                                }), i.show()) : i.hide()
                            })
                        }
                    }
                }, 50)
            })
        },
        crossDomainMessageReceivers: [],
        initCrossDomainMessaging: function() {
            t.isFunction(t.receiveMessage) && t.receiveMessage(t.proxy(function(e) {
                t(this.crossDomainMessageReceivers).each(function(t, n) {
                    n(e.data)
                })
            }, this), function() {
                var e = "https";
                return e + "://" + t(location).attr("hostname")
            }())
        },
        registerMessageReceiver: function(e) {
            this.crossDomainMessageReceivers.push(e)
        }
    }, function() {
        console.log("$$ extended")
    }), e.initCrossDomainMessaging()
}(slideshare_object, jQuery);
