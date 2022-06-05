(function(e, t) {
    function H(e) {
        var t = e.length, n = w.type(e);
        return w.isWindow(e)?!1 : e.nodeType === 1 && t?!0 : n === "array" || n !== "function" && (t === 0 || typeof t == "number" && t > 0 && t - 1 in e)
    }
    function j(e) {
        var t = B[e] = {};
        w.each(e.match(S) || [], function(e, n) {
            t[n]=!0
        });
        return t
    }
    function q(e, n, r, i) {
        if (!w.acceptData(e))
            return;
        var s, o, u = w.expando, a = e.nodeType, f = a ? w.cache: e, l = a ? e[u]: e[u] && u;
        if ((!l ||!f[l] ||!i&&!f[l].data) && r === t && typeof n == "string")
            return;
        l || (a ? l = e[u] = c.pop() || w.guid++ : l = u);
        f[l] || (f[l] = a ? {} : {
            toJSON: w.noop
        });
        if (typeof n == "object" || typeof n == "function")
            i ? f[l] = w.extend(f[l], n) : f[l].data = w.extend(f[l].data, n);
        o = f[l];
        if (!i) {
            o.data || (o.data = {});
            o = o.data
        }
        r !== t && (o[w.camelCase(n)] = r);
        if (typeof n == "string") {
            s = o[n];
            s == null && (s = o[w.camelCase(n)])
        } else 
            s = o;
        return s
    }
    function R(e, t, n) {
        if (!w.acceptData(e))
            return;
        var r, i, s = e.nodeType, o = s ? w.cache: e, u = s ? e[w.expando]: w.expando;
        if (!o[u])
            return;
        if (t) {
            r = n ? o[u] : o[u].data;
            if (r) {
                if (!w.isArray(t))
                    if (t in r)
                        t = [t];
                    else {
                        t = w.camelCase(t);
                        t in r ? t = [t] : t = t.split(" ")
                    } else 
                        t = t.concat(w.map(t, w.camelCase));
                i = t.length;
                while (i--)
                    delete r[t[i]];
                if (n?!z(r) : !w.isEmptyObject(r)
                    )return 
            }
        }
        if (!n) {
            delete o[u].data;
            if (!z(o[u]))
                return 
        }
        s ? w.cleanData([e], !0) : w.support.deleteExpando || o != o.window ? delete o[u] : o[u] = null
    }
    function U(e, n, r) {
        if (r === t && e.nodeType === 1) {
            var i = "data-" + n.replace(I, "-$1").toLowerCase();
            r = e.getAttribute(i);
            if (typeof r == "string") {
                try {
                    r = r === "true"?!0 : r === "false"?!1 : r === "null" ? null : + r + "" === r?+r : F.test(r) ? w.parseJSON(r) : r
                } catch (s) {}
                w.data(e, n, r)
            } else 
                r = t
        }
        return r
    }
    function z(e) {
        var t;
        for (t in e) {
            if (t === "data" && w.isEmptyObject(e[t]))
                continue;
            if (t !== "toJSON")
                return !1
        }
        return !0
    }
    function it() {
        return !0
    }
    function st() {
        return !1
    }
    function ot() {
        try {
            return o.activeElement
        } catch (e) {}
    }
    function ct(e, t) {
        do 
            e = e[t];
        while (e && e.nodeType !== 1);
        return e
    }
    function ht(e, t, n) {
        if (w.isFunction(t))
            return w.grep(e, function(e, r) {
                return !!t.call(e, r, e) !== n
            });
        if (t.nodeType)
            return w.grep(e, function(e) {
                return e === t !== n
            });
        if (typeof t == "string") {
            if (ut.test(t))
                return w.filter(t, e, n);
            t = w.filter(t, e)
        }
        return w.grep(e, function(e) {
            return w.inArray(e, t) >= 0 !== n
        })
    }
    function pt(e) {
        var t = dt.split("|"), n = e.createDocumentFragment();
        if (n.createElement)
            while (t.length)
                n.createElement(t.pop());
        return n
    }
    function Mt(e, t) {
        return w.nodeName(e, "table") && w.nodeName(t.nodeType === 1 ? t : t.firstChild, "tr") ? e.getElementsByTagName("tbody")[0] || e.appendChild(e.ownerDocument.createElement("tbody")) : e
    }
    function _t(e) {
        e.type = (w.find.attr(e, "type") !== null) + "/" + e.type;
        return e
    }
    function Dt(e) {
        var t = Ct.exec(e.type);
        t ? e.type = t[1] : e.removeAttribute("type");
        return e
    }
    function Pt(e, t) {
        var n, r = 0;
        for (; (n = e[r]) != null; r++)
            w._data(n, "globalEval", !t || w._data(t[r], "globalEval"))
    }
    function Ht(e, t) {
        if (t.nodeType !== 1 ||!w.hasData(e))
            return;
        var n, r, i, s = w._data(e), o = w._data(t, s), u = s.events;
        if (u) {
            delete o.handle;
            o.events = {};
            for (n in u)
                for (r = 0, i = u[n].length; r < i; r++)
                    w.event.add(t, n, u[n][r])
        }
        o.data && (o.data = w.extend({}, o.data))
    }
    function Bt(e, t) {
        var n, r, i;
        if (t.nodeType !== 1)
            return;
        n = t.nodeName.toLowerCase();
        if (!w.support.noCloneEvent && t[w.expando]) {
            i = w._data(t);
            for (r in i.events)
                w.removeEvent(t, r, i.handle);
            t.removeAttribute(w.expando)
        }
        if (n === "script" && t.text !== e.text) {
            _t(t).text = e.text;
            Dt(t)
        } else if (n === "object") {
            t.parentNode && (t.outerHTML = e.outerHTML);
            w.support.html5Clone && e.innerHTML&&!w.trim(t.innerHTML) && (t.innerHTML = e.innerHTML)
        } else if (n === "input" && xt.test(e.type)) {
            t.defaultChecked = t.checked = e.checked;
            t.value !== e.value && (t.value = e.value)
        } else if (n === "option")
            t.defaultSelected = t.selected = e.defaultSelected;
        else if (n === "input" || n === "textarea")
            t.defaultValue = e.defaultValue
    }
    function jt(e, n) {
        var r, s, o = 0, u = typeof e.getElementsByTagName !== i ? e.getElementsByTagName(n || "*"): typeof e.querySelectorAll !== i ? e.querySelectorAll(n || "*"): t;
        if (!u)
            for (u = [], r = e.childNodes || e; (s = r[o]) != null; o++)
                !n || w.nodeName(s, n) ? u.push(s) : w.merge(u, jt(s, n));
        return n === t || n && w.nodeName(e, n) ? w.merge([e], u) : u
    }
    function Ft(e) {
        xt.test(e.type) && (e.defaultChecked = e.checked)
    }
    function tn(e, t) {
        if (t in e)
            return t;
        var n = t.charAt(0).toUpperCase() + t.slice(1), r = t, i = en.length;
        while (i--) {
            t = en[i] + n;
            if (t in e)
                return t
        }
        return r
    }
    function nn(e, t) {
        e = t || e;
        return w.css(e, "display") === "none" ||!w.contains(e.ownerDocument, e)
    }
    function rn(e, t) {
        var n, r, i, s = [], o = 0, u = e.length;
        for (; o < u; o++) {
            r = e[o];
            if (!r.style)
                continue;
            s[o] = w._data(r, "olddisplay");
            n = r.style.display;
            if (t) {
                !s[o] && n === "none" && (r.style.display = "");
                r.style.display === "" && nn(r) && (s[o] = w._data(r, "olddisplay", an(r.nodeName)))
            } else if (!s[o]) {
                i = nn(r);
                (n && n !== "none" ||!i) && w._data(r, "olddisplay", i ? n : w.css(r, "display"))
            }
        }
        for (o = 0; o < u; o++) {
            r = e[o];
            if (!r.style)
                continue;
            if (!t || r.style.display === "none" || r.style.display === "")
                r.style.display = t ? s[o] || "" : "none"
        }
        return e
    }
    function sn(e, t, n) {
        var r = $t.exec(t);
        return r ? Math.max(0, r[1] - (n || 0)) + (r[2] || "px") : t
    }
    function on(e, t, n, r, i) {
        var s = n === (r ? "border" : "content") ? 4: t === "width" ? 1: 0, o = 0;
        for (; s < 4; s += 2) {
            n === "margin" && (o += w.css(e, n + Zt[s], !0, i));
            if (r) {
                n === "content" && (o -= w.css(e, "padding" + Zt[s], !0, i));
                n !== "margin" && (o -= w.css(e, "border" + Zt[s] + "Width", !0, i))
            } else {
                o += w.css(e, "padding" + Zt[s], !0, i);
                n !== "padding" && (o += w.css(e, "border" + Zt[s] + "Width", !0, i))
            }
        }
        return o
    }
    function un(e, t, n) {
        var r=!0, i = t === "width" ? e.offsetWidth : e.offsetHeight, s = qt(e), o = w.support.boxSizing && w.css(e, "boxSizing", !1, s) === "border-box";
        if (i <= 0 || i == null) {
            i = Rt(e, t, s);
            if (i < 0 || i == null)
                i = e.style[t];
            if (Jt.test(i))
                return i;
            r = o && (w.support.boxSizingReliable || i === e.style[t]);
            i = parseFloat(i) || 0
        }
        return i + on(e, t, n || (o ? "border" : "content"), r, s) + "px"
    }
    function an(e) {
        var t = o, n = Qt[e];
        if (!n) {
            n = fn(e, t);
            if (n === "none" ||!n) {
                It = (It || w("<iframe frameborder='0' width='0' height='0'/>").css("cssText", "display:block !important")).appendTo(t.documentElement);
                t = (It[0].contentWindow || It[0].contentDocument).document;
                t.write("<!doctype html><html><body>");
                t.close();
                n = fn(e, t);
                It.detach()
            }
            Qt[e] = n
        }
        return n
    }
    function fn(e, t) {
        var n = w(t.createElement(e)).appendTo(t.body), r = w.css(n[0], "display");
        n.remove();
        return r
    }
    function vn(e, t, n, r) {
        var i;
        if (w.isArray(t))
            w.each(t, function(t, i) {
                n || cn.test(e) ? r(e, i) : vn(e + "[" + (typeof i == "object" ? t : "") + "]", i, n, r)
            });
        else if (!n && w.type(t) === "object")
            for (i in t)
                vn(e + "[" + i + "]", t[i], n, r);
        else 
            r(e, t)
    }
    function _n(e) {
        return function(t, n) {
            if (typeof t != "string") {
                n = t;
                t = "*"
            }
            var r, i = 0, s = t.toLowerCase().match(S) || [];
            if (w.isFunction(n))
                while (r = s[i++])
                    if (r[0] === "+") {
                        r = r.slice(1) || "*";
                        (e[r] = e[r] || []).unshift(n)
                    } else (e[r] = e[r] || [])
                        .push(n)
        }
    }
    function Dn(e, t, n, r) {
        function o(u) {
            var a;
            i[u]=!0;
            w.each(e[u] || [], function(e, u) {
                var f = u(t, n, r);
                if (typeof f == "string"&&!s&&!i[f]) {
                    t.dataTypes.unshift(f);
                    o(f);
                    return !1
                }
                if (s)
                    return !(a = f)
            });
            return a
        }
        var i = {}, s = e === An;
        return o(t.dataTypes[0]) ||!i["*"] && o("*")
    }
    function Pn(e, n) {
        var r, i, s = w.ajaxSettings.flatOptions || {};
        for (i in n)
            n[i] !== t && ((s[i] ? e : r || (r = {}))[i] = n[i]);
        r && w.extend(!0, e, r);
        return e
    }
    function Hn(e, n, r) {
        var i, s, o, u, a = e.contents, f = e.dataTypes;
        while (f[0] === "*") {
            f.shift();
            s === t && (s = e.mimeType || n.getResponseHeader("Content-Type"))
        }
        if (s)
            for (u in a)
                if (a[u] && a[u].test(s)) {
                    f.unshift(u);
                    break
                }
        if (f[0]in r)
            o = f[0];
        else {
            for (u in r) {
                if (!f[0] || e.converters[u + " " + f[0]]) {
                    o = u;
                    break
                }
                i || (i = u)
            }
            o = o || i
        }
        if (o) {
            o !== f[0] && f.unshift(o);
            return r[o]
        }
    }
    function Bn(e, t, n, r) {
        var i, s, o, u, a, f = {}, l = e.dataTypes.slice();
        if (l[1])
            for (o in e.converters)
                f[o.toLowerCase()] = e.converters[o];
        s = l.shift();
        while (s) {
            e.responseFields[s] && (n[e.responseFields[s]] = t);
            !a && r && e.dataFilter && (t = e.dataFilter(t, e.dataType));
            a = s;
            s = l.shift();
            if (s)
                if (s === "*")
                    s = a;
                else if (a !== "*" && a !== s) {
                    o = f[a + " " + s] || f["* " + s];
                    if (!o)
                        for (i in f) {
                            u = i.split(" ");
                            if (u[1] === s) {
                                o = f[a + " " + u[0]] || f["* " + u[0]];
                                if (o) {
                                    if (o===!0)
                                        o = f[i];
                                    else if (f[i]!==!0) {
                                        s = u[0];
                                        l.unshift(u[1])
                                    }
                                    break
                                }
                            }
                        }
                        if (o!==!0)
                            if (o && e["throws"])
                                t = o(t);
                            else 
                                try {
                                    t = o(t)
                                } catch (c) {
                                    return {
                                        state: "parsererror",
                                        error: o ? c: "No conversion from " + a + " to " + s
                                    }
                                }
            }
        }
        return {
            state: "success",
            data: t
        }
    }
    function zn() {
        try {
            return new e.XMLHttpRequest
        } catch (t) {}
    }
    function Wn() {
        try {
            return new e.ActiveXObject("Microsoft.XMLHTTP")
        } catch (t) {}
    }
    function Yn() {
        setTimeout(function() {
            Xn = t
        });
        return Xn = w.now()
    }
    function Zn(e, t, n) {
        var r, i = (Gn[t] || []).concat(Gn["*"]), s = 0, o = i.length;
        for (; s < o; s++)
            if (r = i[s].call(n, t, e))
                return r
    }
    function er(e, t, n) {
        var r, i, s = 0, o = Qn.length, u = w.Deferred().always(function() {
            delete a.elem
        }), a = function() {
            if (i)
                return !1;
            var t = Xn || Yn(), n = Math.max(0, f.startTime + f.duration - t), r = n / f.duration || 0, s = 1 - r, o = 0, a = f.tweens.length;
            for (; o < a; o++)
                f.tweens[o].run(s);
            u.notifyWith(e, [f, s, n]);
            if (s < 1 && a)
                return n;
            u.resolveWith(e, [f]);
            return !1
        }, f = u.promise({
            elem: e,
            props: w.extend({}, t),
            opts: w.extend(!0, {
                specialEasing: {}
            }, n),
            originalProperties: t,
            originalOptions: n,
            startTime: Xn || Yn(),
            duration: n.duration,
            tweens: [],
            createTween: function(t, n) {
                var r = w.Tween(e, f.opts, t, n, f.opts.specialEasing[t] || f.opts.easing);
                f.tweens.push(r);
                return r
            },
            stop: function(t) {
                var n = 0, r = t ? f.tweens.length: 0;
                if (i)
                    return this;
                i=!0;
                for (; n < r; n++)
                    f.tweens[n].run(1);
                t ? u.resolveWith(e, [f, t]) : u.rejectWith(e, [f, t]);
                return this
            }
        }), l = f.props;
        tr(l, f.opts.specialEasing);
        for (; s < o; s++) {
            r = Qn[s].call(f, e, l, f.opts);
            if (r)
                return r
        }
        w.map(l, Zn, f);
        w.isFunction(f.opts.start) && f.opts.start.call(e, f);
        w.fx.timer(w.extend(a, {
            elem: e,
            anim: f,
            queue: f.opts.queue
        }));
        return f.progress(f.opts.progress).done(f.opts.done, f.opts.complete).fail(f.opts.fail).always(f.opts.always)
    }
    function tr(e, t) {
        var n, r, i, s, o;
        for (n in e) {
            r = w.camelCase(n);
            i = t[r];
            s = e[n];
            if (w.isArray(s)) {
                i = s[1];
                s = e[n] = s[0]
            }
            if (n !== r) {
                e[r] = s;
                delete e[n]
            }
            o = w.cssHooks[r];
            if (o && "expand"in o) {
                s = o.expand(s);
                delete e[r];
                for (n in s)
                    if (!(n in e)) {
                        e[n] = s[n];
                        t[n] = i
                    }
            } else 
                t[r] = i
        }
    }
    function nr(e, t, n) {
        var r, i, s, o, u, a, f = this, l = {}, c = e.style, h = e.nodeType && nn(e), p = w._data(e, "fxshow");
        if (!n.queue) {
            u = w._queueHooks(e, "fx");
            if (u.unqueued == null) {
                u.unqueued = 0;
                a = u.empty.fire;
                u.empty.fire = function() {
                    u.unqueued || a()
                }
            }
            u.unqueued++;
            f.always(function() {
                f.always(function() {
                    u.unqueued--;
                    w.queue(e, "fx").length || u.empty.fire()
                })
            })
        }
        if (e.nodeType === 1 && ("height"in t || "width"in t)) {
            n.overflow = [c.overflow, c.overflowX, c.overflowY];
            w.css(e, "display") === "inline" && w.css(e, "float") === "none" && (!w.support.inlineBlockNeedsLayout || an(e.nodeName) === "inline" ? c.display = "inline-block" : c.zoom = 1)
        }
        if (n.overflow) {
            c.overflow = "hidden";
            w.support.shrinkWrapBlocks || f.always(function() {
                c.overflow = n.overflow[0];
                c.overflowX = n.overflow[1];
                c.overflowY = n.overflow[2]
            })
        }
        for (r in t) {
            i = t[r];
            if ($n.exec(i)) {
                delete t[r];
                s = s || i === "toggle";
                if (i === (h ? "hide" : "show"))
                    continue;
                l[r] = p && p[r] || w.style(e, r)
            }
        }
        if (!w.isEmptyObject(l)) {
            p ? "hidden"in p && (h = p.hidden) : p = w._data(e, "fxshow", {});
            s && (p.hidden=!h);
            h ? w(e).show() : f.done(function() {
                w(e).hide()
            });
            f.done(function() {
                var t;
                w._removeData(e, "fxshow");
                for (t in l)
                    w.style(e, t, l[t])
            });
            for (r in l) {
                o = Zn(h ? p[r] : 0, r, f);
                if (!(r in p)) {
                    p[r] = o.start;
                    if (h) {
                        o.end = o.start;
                        o.start = r === "width" || r === "height" ? 1 : 0
                    }
                }
            }
        }
    }
    function rr(e, t, n, r, i) {
        return new rr.prototype.init(e, t, n, r, i)
    }
    function ir(e, t) {
        var n, r = {
            height: e
        }, i = 0;
        t = t ? 1 : 0;
        for (; i < 4; i += 2 - t) {
            n = Zt[i];
            r["margin" + n] = r["padding" + n] = e
        }
        t && (r.opacity = r.width = e);
        return r
    }
    function sr(e) {
        return w.isWindow(e) ? e : e.nodeType === 9 ? e.defaultView || e.parentWindow : !1
    }
    var n, r, i = typeof t, s = e.location, o = e.document, u = o.documentElement, a = e.jQuery, f = e.$, l = {}, c = [], h = "1.10.2", p = c.concat, d = c.push, v = c.slice, m = c.indexOf, g = l.toString, y = l.hasOwnProperty, b = h.trim, w = function(e, t) {
        return new w.fn.init(e, t, r)
    }, E = /[+-]?(?:\d*\.|)\d+(?:[eE][+-]?\d+|)/.source, S = /\S+/g, x = /^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g, T = /^(?:\s*(<[\w\W]+>)[^>]*|#([\w-]*))$/, N = /^<(\w+)\s*\/?>(?:<\/\1>|)$/, C = /^[\],:{}\s]*$/, k = /(?:^|:|,)(?:\s*\[)+/g, L = /\\(?:["\\\/bfnrt]|u[\da-fA-F]{4})/g, A = /"[^"\\\r\n]*"|true|false|null|-?(?:\d+\.|)\d+(?:[eE][+-]?\d+|)/g, O = /^-ms-/, M = /-([\da-z])/gi, _ = function(e, t) {
        return t.toUpperCase()
    }, D = function(e) {
        if (o.addEventListener || e.type === "load" || o.readyState === "complete") {
            P();
            w.ready()
        }
    }, P = function() {
        if (o.addEventListener) {
            o.removeEventListener("DOMContentLoaded", D, !1);
            e.removeEventListener("load", D, !1)
        } else {
            o.detachEvent("onreadystatechange", D);
            e.detachEvent("onload", D)
        }
    };
    w.fn = w.prototype = {
        jquery: h,
        constructor: w,
        init: function(e, n, r) {
            var i, s;
            if (!e)
                return this;
            if (typeof e == "string") {
                e.charAt(0) === "<" && e.charAt(e.length - 1) === ">" && e.length >= 3 ? i = [null, e, null] : i = T.exec(e);
                if (i && (i[1] ||!n)) {
                    if (i[1]) {
                        n = n instanceof w ? n[0] : n;
                        w.merge(this, w.parseHTML(i[1], n && n.nodeType ? n.ownerDocument || n : o, !0));
                        if (N.test(i[1]) && w.isPlainObject(n))
                            for (i in n)
                                w.isFunction(this[i]) ? this[i](n[i]) : this.attr(i, n[i]);
                        return this
                    }
                    s = o.getElementById(i[2]);
                    if (s && s.parentNode) {
                        if (s.id !== i[2])
                            return r.find(e);
                        this.length = 1;
                        this[0] = s
                    }
                    this.context = o;
                    this.selector = e;
                    return this
                }
                return !n || n.jquery ? (n || r).find(e) : this.constructor(n).find(e)
            }
            if (e.nodeType) {
                this.context = this[0] = e;
                this.length = 1;
                return this
            }
            if (w.isFunction(e))
                return r.ready(e);
            if (e.selector !== t) {
                this.selector = e.selector;
                this.context = e.context
            }
            return w.makeArray(e, this)
        },
        selector: "",
        length: 0,
        toArray: function() {
            return v.call(this)
        },
        get: function(e) {
            return e == null ? this.toArray() : e < 0 ? this[this.length + e] : this[e]
        },
        pushStack: function(e) {
            var t = w.merge(this.constructor(), e);
            t.prevObject = this;
            t.context = this.context;
            return t
        },
        each: function(e, t) {
            return w.each(this, e, t)
        },
        ready: function(e) {
            w.ready.promise().done(e);
            return this
        },
        slice: function() {
            return this.pushStack(v.apply(this, arguments))
        },
        first: function() {
            return this.eq(0)
        },
        last: function() {
            return this.eq( - 1)
        },
        eq: function(e) {
            var t = this.length, n =+ e + (e < 0 ? t : 0);
            return this.pushStack(n >= 0 && n < t ? [this[n]] : [])
        },
        map: function(e) {
            return this.pushStack(w.map(this, function(t, n) {
                return e.call(t, n, t)
            }))
        },
        end: function() {
            return this.prevObject || this.constructor(null)
        },
        push: d,
        sort: [].sort,
        splice: [].splice
    };
    w.fn.init.prototype = w.fn;
    w.extend = w.fn.extend = function() {
        var e, n, r, i, s, o, u = arguments[0] || {}, a = 1, f = arguments.length, l=!1;
        if (typeof u == "boolean") {
            l = u;
            u = arguments[1] || {};
            a = 2
        }
        typeof u != "object"&&!w.isFunction(u) && (u = {});
        if (f === a) {
            u = this;
            --a
        }
        for (; a < f; a++)
            if ((s = arguments[a]) != null)
                for (i in s) {
                    e = u[i];
                    r = s[i];
                    if (u === r)
                        continue;
                        if (l && r && (w.isPlainObject(r) || (n = w.isArray(r)))) {
                            if (n) {
                                n=!1;
                                o = e && w.isArray(e) ? e : []
                            } else 
                                o = e && w.isPlainObject(e) ? e : {};
                                u[i] = w.extend(l, o, r)
                            } else 
                                r !== t && (u[i] = r)
                    }
        return u
    };
    w.extend({
        expando: "jQuery" + (h + Math.random()).replace(/\D/g, ""),
        noConflict: function(t) {
            e.$ === w && (e.$ = f);
            t && e.jQuery === w && (e.jQuery = a);
            return w
        },
        isReady: !1,
        readyWait: 1,
        holdReady: function(e) {
            e ? w.readyWait++ : w.ready(!0)
        },
        ready: function(e) {
            if (e===!0?--w.readyWait : w.isReady)
                return;
            if (!o.body)
                return setTimeout(w.ready);
            w.isReady=!0;
            if (e!==!0&&--w.readyWait > 0)
                return;
            n.resolveWith(o, [w]);
            w.fn.trigger && w(o).trigger("ready").off("ready")
        },
        isFunction: function(e) {
            return w.type(e) === "function"
        },
        isArray: Array.isArray || function(e) {
            return w.type(e) === "array"
        },
        isWindow: function(e) {
            return e != null && e == e.window
        },
        isNumeric: function(e) {
            return !isNaN(parseFloat(e)) && isFinite(e)
        },
        type: function(e) {
            return e == null ? String(e) : typeof e == "object" || typeof e == "function" ? l[g.call(e)] || "object" : typeof e
        },
        isPlainObject: function(e) {
            var n;
            if (!e || w.type(e) !== "object" || e.nodeType || w.isWindow(e))
                return !1;
            try {
                if (e.constructor&&!y.call(e, "constructor")&&!y.call(e.constructor.prototype, "isPrototypeOf"))
                    return !1
                    } catch (r) {
                return !1
            }
            if (w.support.ownLast)
                for (n in e)
                    return y.call(e, n);
            for (n in e);
            return n === t || y.call(e, n)
        },
        isEmptyObject: function(e) {
            var t;
            for (t in e)
                return !1;
            return !0
        },
        error: function(e) {
            throw new Error(e)
        },
        parseHTML: function(e, t, n) {
            if (!e || typeof e != "string")
                return null;
            if (typeof t == "boolean") {
                n = t;
                t=!1
            }
            t = t || o;
            var r = N.exec(e), i=!n && [];
            if (r)
                return [t.createElement(r[1])];
            r = w.buildFragment([e], t, i);
            i && w(i).remove();
            return w.merge([], r.childNodes)
        },
        parseJSON: function(t) {
            if (e.JSON && e.JSON.parse)
                return e.JSON.parse(t);
            if (t === null)
                return t;
            if (typeof t == "string") {
                t = w.trim(t);
                if (t && C.test(t.replace(L, "@").replace(A, "]").replace(k, "")))
                    return (new Function("return " + t))()
            }
            w.error("Invalid JSON: " + t)
        },
        parseXML: function(n) {
            var r, i;
            if (!n || typeof n != "string")
                return null;
            try {
                if (e.DOMParser) {
                    i = new DOMParser;
                    r = i.parseFromString(n, "text/xml")
                } else {
                    r = new ActiveXObject("Microsoft.XMLDOM");
                    r.async = "false";
                    r.loadXML(n)
                }
            } catch (s) {
                r = t
            }(!r ||!r.documentElement || r.getElementsByTagName("parsererror").length) && w.error("Invalid XML: " + n);
            return r
        },
        noop: function() {},
        globalEval: function(t) {
            t && w.trim(t) && (e.execScript || function(t) {
                e.eval.call(e, t)
            })(t)
        },
        camelCase: function(e) {
            return e.replace(O, "ms-").replace(M, _)
        },
        nodeName: function(e, t) {
            return e.nodeName && e.nodeName.toLowerCase() === t.toLowerCase()
        },
        each: function(e, t, n) {
            var r, i = 0, s = e.length, o = H(e);
            if (n)
                if (o)
                    for (; i < s; i++) {
                        r = t.apply(e[i], n);
                        if (r===!1)
                            break
                    } else 
                        for (i in e) {
                            r = t.apply(e[i], n);
                            if (r===!1)
                                break
                        } else if (o)
                for (; i < s; i++) {
                    r = t.call(e[i], i, e[i]);
                    if (r===!1)
                        break
                } else 
                    for (i in e) {
                        r = t.call(e[i], i, e[i]);
                        if (r===!1)
                            break
                    }
            return e
        },
        trim: b&&!b.call("﻿ ") ? function(e) {
            return e == null ? "" : b.call(e)
        }
        : function(e) {
            return e == null ? "" : (e + "").replace(x, "")
        },
        makeArray: function(e, t) {
            var n = t || [];
            e != null && (H(Object(e)) ? w.merge(n, typeof e == "string" ? [e] : e) : d.call(n, e));
            return n
        },
        inArray: function(e, t, n) {
            var r;
            if (t) {
                if (m)
                    return m.call(t, e, n);
                r = t.length;
                n = n ? n < 0 ? Math.max(0, r + n) : n : 0;
                for (; n < r; n++)
                    if (n in t && t[n] === e)
                        return n
            }
            return - 1
        },
        merge: function(e, n) {
            var r = n.length, i = e.length, s = 0;
            if (typeof r == "number")
                for (; s < r; s++)
                    e[i++] = n[s];
            else 
                while (n[s] !== t)
                    e[i++] = n[s++];
            e.length = i;
            return e
        },
        grep: function(e, t, n) {
            var r, i = [], s = 0, o = e.length;
            n=!!n;
            for (; s < o; s++) {
                r=!!t(e[s], s);
                n !== r && i.push(e[s])
            }
            return i
        },
        map: function(e, t, n) {
            var r, i = 0, s = e.length, o = H(e), u = [];
            if (o)
                for (; i < s; i++) {
                    r = t(e[i], i, n);
                    r != null && (u[u.length] = r)
                } else 
                    for (i in e) {
                        r = t(e[i], i, n);
                        r != null && (u[u.length] = r)
                    }
            return p.apply([], u)
        },
        guid: 1,
        proxy: function(e, n) {
            var r, i, s;
            if (typeof n == "string") {
                s = e[n];
                n = e;
                e = s
            }
            if (!w.isFunction(e))
                return t;
            r = v.call(arguments, 2);
            i = function() {
                return e.apply(n || this, r.concat(v.call(arguments)))
            };
            i.guid = e.guid = e.guid || w.guid++;
            return i
        },
        access: function(e, n, r, i, s, o, u) {
            var a = 0, f = e.length, l = r == null;
            if (w.type(r) === "object") {
                s=!0;
                for (a in r)
                    w.access(e, n, a, r[a], !0, o, u)
            } else if (i !== t) {
                s=!0;
                w.isFunction(i) || (u=!0);
                if (l)
                    if (u) {
                        n.call(e, i);
                        n = null
                    } else {
                        l = n;
                        n = function(e, t, n) {
                            return l.call(w(e), n)
                        }
                    }
                if (n)
                    for (; a < f; a++)
                        n(e[a], r, u ? i : i.call(e[a], a, n(e[a], r)))
            }
            return s ? e : l ? n.call(e) : f ? n(e[0], r) : o
        },
        now: function() {
            return (new Date).getTime()
        },
        swap: function(e, t, n, r) {
            var i, s, o = {};
            for (s in t) {
                o[s] = e.style[s];
                e.style[s] = t[s]
            }
            i = n.apply(e, r || []);
            for (s in t)
                e.style[s] = o[s];
            return i
        }
    });
    w.ready.promise = function(t) {
        if (!n) {
            n = w.Deferred();
            if (o.readyState === "complete")
                setTimeout(w.ready);
            else if (o.addEventListener) {
                o.addEventListener("DOMContentLoaded", D, !1);
                e.addEventListener("load", D, !1)
            } else {
                o.attachEvent("onreadystatechange", D);
                e.attachEvent("onload", D);
                var r=!1;
                try {
                    r = e.frameElement == null && o.documentElement
                } catch (i) {}
                r && r.doScroll && function s() {
                    if (!w.isReady) {
                        try {
                            r.doScroll("left")
                        } catch (e) {
                            return setTimeout(s, 50)
                        }
                        P();
                        w.ready()
                    }
                }()
            }
        }
        return n.promise(t)
    };
    w.each("Boolean Number String Function Array Date RegExp Object Error".split(" "), function(e, t) {
        l["[object " + t + "]"] = t.toLowerCase()
    });
    r = w(o);
    (function(e, t) {
        function ot(e, t, n, i) {
            var s, o, u, a, f, l, p, m, g, w;
            (t ? t.ownerDocument || t : E) !== h && c(t);
            t = t || h;
            n = n || [];
            if (!e || typeof e != "string")
                return n;
            if ((a = t.nodeType) !== 1 && a !== 9)
                return [];
            if (d&&!i) {
                if (s = Z.exec(e))
                    if (u = s[1]) {
                        if (a === 9) {
                            o = t.getElementById(u);
                            if (!o ||!o.parentNode)
                                return n;
                                if (o.id === u) {
                                    n.push(o);
                                    return n
                                }
                        } else if (t.ownerDocument && (o = t.ownerDocument.getElementById(u)) && y(t, o) && o.id === u) {
                            n.push(o);
                            return n
                        }
                    } else {
                        if (s[2]) {
                            H.apply(n, t.getElementsByTagName(e));
                            return n
                        }
                        if ((u = s[3]) && r.getElementsByClassName && t.getElementsByClassName) {
                            H.apply(n, t.getElementsByClassName(u));
                            return n
                        }
                    }
                if (r.qsa && (!v ||!v.test(e))) {
                    m = p = b;
                    g = t;
                    w = a === 9 && e;
                    if (a === 1 && t.nodeName.toLowerCase() !== "object") {
                        l = mt(e);
                        (p = t.getAttribute("id")) ? m = p.replace(nt, "\\$&") : t.setAttribute("id", m);
                        m = "[id='" + m + "'] ";
                        f = l.length;
                        while (f--)
                            l[f] = m + gt(l[f]);
                        g = $.test(e) && t.parentNode || t;
                        w = l.join(",")
                    }
                    if (w)
                        try {
                            H.apply(n, g.querySelectorAll(w));
                            return n
                    } catch (S) {} finally {
                        p || t.removeAttribute("id")
                    }
                }
            }
            return Nt(e.replace(W, "$1"), t, n, i)
        }
        function ut() {
            function t(n, r) {
                e.push(n += " ") > s.cacheLength && delete t[e.shift()];
                return t[n] = r
            }
            var e = [];
            return t
        }
        function at(e) {
            e[b]=!0;
            return e
        }
        function ft(e) {
            var t = h.createElement("div");
            try {
                return !!e(t)
            } catch (n) {
                return !1
            } finally {
                t.parentNode && t.parentNode.removeChild(t);
                t = null
            }
        }
        function lt(e, t) {
            var n = e.split("|"), r = e.length;
            while (r--)
                s.attrHandle[n[r]] = t
        }
        function ct(e, t) {
            var n = t && e, r = n && e.nodeType === 1 && t.nodeType === 1 && (~t.sourceIndex || O) - (~e.sourceIndex || O);
            if (r)
                return r;
            if (n)
                while (n = n.nextSibling)
                    if (n === t)
                        return - 1;
            return e ? 1 : - 1
        }
        function ht(e) {
            return function(t) {
                var n = t.nodeName.toLowerCase();
                return n === "input" && t.type === e
            }
        }
        function pt(e) {
            return function(t) {
                var n = t.nodeName.toLowerCase();
                return (n === "input" || n === "button") && t.type === e
            }
        }
        function dt(e) {
            return at(function(t) {
                t =+ t;
                return at(function(n, r) {
                    var i, s = e([], n.length, t), o = s.length;
                    while (o--)
                        n[i = s[o]] && (n[i]=!(r[i] = n[i]))
                })
            })
        }
        function vt() {}
        function mt(e, t) {
            var n, r, i, o, u, a, f, l = N[e + " "];
            if (l)
                return t ? 0 : l.slice(0);
            u = e;
            a = [];
            f = s.preFilter;
            while (u) {
                if (!n || (r = X.exec(u))) {
                    r && (u = u.slice(r[0].length) || u);
                    a.push(i = [])
                }
                n=!1;
                if (r = V.exec(u)) {
                    n = r.shift();
                    i.push({
                        value: n,
                        type: r[0].replace(W, " ")
                    });
                    u = u.slice(n.length)
                }
                for (o in s.filter)
                    if ((r = G[o].exec(u)) && (!f[o] || (r = f[o](r)))) {
                        n = r.shift();
                        i.push({
                            value: n,
                            type: o,
                            matches: r
                        });
                        u = u.slice(n.length)
                    }
                if (!n)
                    break
            }
            return t ? u.length : u ? ot.error(e) : N(e, a).slice(0)
        }
        function gt(e) {
            var t = 0, n = e.length, r = "";
            for (; t < n; t++)
                r += e[t].value;
            return r
        }
        function yt(e, t, n) {
            var r = t.dir, s = n && r === "parentNode", o = x++;
            return t.first ? function(t, n, i) {
                while (t = t[r])
                    if (t.nodeType === 1 || s)
                        return e(t, n, i)
            } : function(t, n, u) {
                var a, f, l, c = S + " " + o;
                if (u) {
                    while (t = t[r])
                        if (t.nodeType === 1 || s)
                            if (e(t, n, u))
                                return !0
                } else 
                    while (t = t[r])
                        if (t.nodeType === 1 || s) {
                            l = t[b] || (t[b] = {});
                            if ((f = l[r]) && f[0] === c) {
                                if ((a = f[1])===!0 || a === i)
                                    return a===!0
                            } else {
                                f = l[r] = [c];
                                f[1] = e(t, n, u) || i;
                                if (f[1]===!0)
                                    return !0
                            }
                        }
            }
        }
        function bt(e) {
            return e.length > 1 ? function(t, n, r) {
                var i = e.length;
                while (i--)
                    if (!e[i](t, n, r))
                        return !1;
                return !0
            } : e[0]
        }
        function wt(e, t, n, r, i) {
            var s, o = [], u = 0, a = e.length, f = t != null;
            for (; u < a; u++)
                if (s = e[u])
                    if (!n || n(s, r, i)) {
                        o.push(s);
                        f && t.push(u)
                    }
            return o
        }
        function Et(e, t, n, r, i, s) {
            r&&!r[b] && (r = Et(r));
            i&&!i[b] && (i = Et(i, s));
            return at(function(s, o, u, a) {
                var f, l, c, h = [], p = [], d = o.length, v = s || Tt(t || "*", u.nodeType ? [u] : u, []), m = e && (s ||!t) ? wt(v, h, e, u, a): v, g = n ? i || (s ? e : d || r) ? []: o: m;
                n && n(m, g, u, a);
                if (r) {
                    f = wt(g, p);
                    r(f, [], u, a);
                    l = f.length;
                    while (l--)
                        if (c = f[l])
                            g[p[l]]=!(m[p[l]] = c)
                }
                if (s) {
                    if (i || e) {
                        if (i) {
                            f = [];
                            l = g.length;
                            while (l--)(c = g[l]) && f.push(m[l] = c);
                            i(null, g = [], f, a)
                        }
                        l = g.length;
                        while (l--)(c = g[l]) && (f = i ? j.call(s, c) : h[l])>-1 && (s[f]=!(o[f] = c))
                        }
                } else {
                    g = wt(g === o ? g.splice(d, g.length) : g);
                    i ? i(null, o, g, a) : H.apply(o, g)
                }
            })
        }
        function St(e) {
            var t, n, r, i = e.length, o = s.relative[e[0].type], u = o || s.relative[" "], a = o ? 1: 0, l = yt(function(e) {
                return e === t
            }, u, !0), c = yt(function(e) {
                return j.call(t, e)>-1
            }, u, !0), h = [function(e, n, r) {
                return !o && (r || n !== f) || ((t = n).nodeType ? l(e, n, r) : c(e, n, r))
            }
            ];
            for (; a < i; a++)
                if (n = s.relative[e[a].type])
                    h = [yt(bt(h), n)];
                else {
                    n = s.filter[e[a].type].apply(null, e[a].matches);
                    if (n[b]) {
                        r=++a;
                        for (; r < i; r++)
                            if (s.relative[e[r].type])
                                break;
                                return Et(a > 1 && bt(h), a > 1 && gt(e.slice(0, a - 1).concat({
                                    value: e[a - 2].type === " " ? "*": ""
                                })).replace(W, "$1"), n, a < r && St(e.slice(a, r)), r < i && St(e = e.slice(r)), r < i && gt(e))
                            }
                            h.push(n)
                }
            return bt(h)
        }
        function xt(e, t) {
            var n = 0, r = t.length > 0, o = e.length > 0, u = function(u, a, l, c, p) {
                var d, v, m, g = [], y = 0, b = "0", w = u && [], E = p != null, x = f, T = u || o && s.find.TAG("*", p && a.parentNode || a), N = S += x == null ? 1: Math.random() || .1;
                if (E) {
                    f = a !== h && a;
                    i = n
                }
                for (; (d = T[b]) != null; b++) {
                    if (o && d) {
                        v = 0;
                        while (m = e[v++])
                            if (m(d, a, l)) {
                                c.push(d);
                                break
                            }
                        if (E) {
                            S = N;
                            i=++n
                        }
                    }
                    if (r) {
                        (d=!m && d) && y--;
                        u && w.push(d)
                    }
                }
                y += b;
                if (r && b !== y) {
                    v = 0;
                    while (m = t[v++])
                        m(w, g, a, l);
                    if (u) {
                        if (y > 0)
                            while (b--)!w[b]&&!g[b] && (g[b] = D.call(c));
                        g = wt(g)
                    }
                    H.apply(c, g);
                    E&&!u && g.length > 0 && y + t.length > 1 && ot.uniqueSort(c)
                }
                if (E) {
                    S = N;
                    f = x
                }
                return w
            };
            return r ? at(u) : u
        }
        function Tt(e, t, n) {
            var r = 0, i = t.length;
            for (; r < i; r++)
                ot(e, t[r], n);
            return n
        }
        function Nt(e, t, n, i) {
            var o, u, f, l, c, h = mt(e);
            if (!i && h.length === 1) {
                u = h[0] = h[0].slice(0);
                if (u.length > 2 && (f = u[0]).type === "ID" && r.getById && t.nodeType === 9 && d && s.relative[u[1].type]) {
                    t = (s.find.ID(f.matches[0].replace(rt, it), t) || [])[0];
                    if (!t)
                        return n;
                    e = e.slice(u.shift().value.length)
                }
                o = G.needsContext.test(e) ? 0 : u.length;
                while (o--) {
                    f = u[o];
                    if (s.relative[l = f.type])
                        break;
                    if (c = s.find[l])
                        if (i = c(f.matches[0].replace(rt, it), $.test(u[0].type) && t.parentNode || t)) {
                            u.splice(o, 1);
                            e = i.length && gt(u);
                            if (!e) {
                                H.apply(n, i);
                                return n
                            }
                            break
                        }
                }
            }
            a(e, h)(i, t, !d, n, $.test(e));
            return n
        }
        var n, r, i, s, o, u, a, f, l, c, h, p, d, v, m, g, y, b = "sizzle" + - (new Date), E = e.document, S = 0, x = 0, T = ut(), N = ut(), C = ut(), k=!1, L = function(e, t) {
            if (e === t) {
                k=!0;
                return 0
            }
            return 0
        }, A = typeof t, O = 1<<31, M = {}.hasOwnProperty, _ = [], D = _.pop, P = _.push, H = _.push, B = _.slice, j = _.indexOf || function(e) {
            var t = 0, n = this.length;
            for (; t < n; t++)
                if (this[t] === e)
                    return t;
            return - 1
        }, F = "checked|selected|async|autofocus|autoplay|controls|defer|disabled|hidden|ismap|loop|multiple|open|readonly|required|scoped", I = "[\\x20\\t\\r\\n\\f]", q = "(?:\\\\.|[\\w-]|[^\\x00-\\xa0])+", R = q.replace("w", "w#"), U = "\\[" + I + "*(" + q + ")" + I + "*(?:([*^$|!~]?=)" + I + "*(?:(['\"])((?:\\\\.|[^\\\\])*?)\\3|(" + R + ")|)|)" + I + "*\\]", z = ":(" + q + ")(?:\\(((['\"])((?:\\\\.|[^\\\\])*?)\\3|((?:\\\\.|[^\\\\()[\\]]|" + U.replace(3, 8) + ")*)|.*)\\)|)", W = new RegExp("^" + I + "+|((?:^|[^\\\\])(?:\\\\.)*)" + I + "+$", "g"), X = new RegExp("^" + I + "*," + I + "*"), V = new RegExp("^" + I + "*([>+~]|" + I + ")" + I + "*"), $ = new RegExp(I + "*[+~]"), J = new RegExp("=" + I + "*([^\\]'\"]*)" + I + "*\\]", "g"), K = new RegExp(z), Q = new RegExp("^" + R + "$"), G = {
            ID: new RegExp("^#(" + q + ")"),
            CLASS: new RegExp("^\\.(" + q + ")"),
            TAG: new RegExp("^(" + q.replace("w", "w*") + ")"),
            ATTR: new RegExp("^" + U),
            PSEUDO: new RegExp("^" + z),
            CHILD: new RegExp("^:(only|first|last|nth|nth-last)-(child|of-type)(?:\\(" + I + "*(even|odd|(([+-]|)(\\d*)n|)" + I + "*(?:([+-]|)" + I + "*(\\d+)|))" + I + "*\\)|)", "i"),
            bool: new RegExp("^(?:" + F + ")$", "i"),
            needsContext: new RegExp("^" + I + "*[>+~]|:(even|odd|eq|gt|lt|nth|first|last)(?:\\(" + I + "*((?:-\\d)?\\d*)" + I + "*\\)|)(?=[^-]|$)", "i")
        }, Y = /^[^{]+\{\s*\[native \w/, Z = /^(?:#([\w-]+)|(\w+)|\.([\w-]+))$/, et = /^(?:input|select|textarea|button)$/i, tt = /^h\d$/i, nt = /'|\\/g, rt = new RegExp("\\\\([\\da-f]{1,6}" + I + "?|(" + I + ")|.)", "ig"), it = function(e, t, n) {
            var r = "0x" + t - 65536;
            return r !== r || n ? t : r < 0 ? String.fromCharCode(r + 65536) : String.fromCharCode(r>>10 | 55296, r & 1023 | 56320)
        };
        try {
            H.apply(_ = B.call(E.childNodes), E.childNodes);
            _[E.childNodes.length].nodeType
        } catch (st) {
            H = {
                apply: _.length ? function(e, t) {
                    P.apply(e, B.call(t))
                }
                : function(e, t) {
                    var n = e.length, r = 0;
                    while (e[n++] = t[r++]);
                    e.length = n - 1
                }
            }
        }
        u = ot.isXML = function(e) {
            var t = e && (e.ownerDocument || e).documentElement;
            return t ? t.nodeName !== "HTML" : !1
        };
        r = ot.support = {};
        c = ot.setDocument = function(e) {
            var t = e ? e.ownerDocument || e: E, n = t.defaultView;
            if (t === h || t.nodeType !== 9 ||!t.documentElement)
                return h;
            h = t;
            p = t.documentElement;
            d=!u(t);
            n && n.attachEvent && n !== n.top && n.attachEvent("onbeforeunload", function() {
                c()
            });
            r.attributes = ft(function(e) {
                e.className = "i";
                return !e.getAttribute("className")
            });
            r.getElementsByTagName = ft(function(e) {
                e.appendChild(t.createComment(""));
                return !e.getElementsByTagName("*").length
            });
            r.getElementsByClassName = ft(function(e) {
                e.innerHTML = "<div class='a'></div><div class='a i'></div>";
                e.firstChild.className = "i";
                return e.getElementsByClassName("i").length === 2
            });
            r.getById = ft(function(e) {
                p.appendChild(e).id = b;
                return !t.getElementsByName ||!t.getElementsByName(b).length
            });
            if (r.getById) {
                s.find.ID = function(e, t) {
                    if (typeof t.getElementById !== A && d) {
                        var n = t.getElementById(e);
                        return n && n.parentNode ? [n] : []
                    }
                };
                s.filter.ID = function(e) {
                    var t = e.replace(rt, it);
                    return function(e) {
                        return e.getAttribute("id") === t
                    }
                }
            } else {
                delete s.find.ID;
                s.filter.ID = function(e) {
                    var t = e.replace(rt, it);
                    return function(e) {
                        var n = typeof e.getAttributeNode !== A && e.getAttributeNode("id");
                        return n && n.value === t
                    }
                }
            }
            s.find.TAG = r.getElementsByTagName ? function(e, t) {
                if (typeof t.getElementsByTagName !== A)
                    return t.getElementsByTagName(e)
            } : function(e, t) {
                var n, r = [], i = 0, s = t.getElementsByTagName(e);
                if (e === "*") {
                    while (n = s[i++])
                        n.nodeType === 1 && r.push(n);
                    return r
                }
                return s
            };
            s.find.CLASS = r.getElementsByClassName && function(e, t) {
                if (typeof t.getElementsByClassName !== A && d)
                    return t.getElementsByClassName(e)
            };
            m = [];
            v = [];
            if (r.qsa = Y.test(t.querySelectorAll)) {
                ft(function(e) {
                    e.innerHTML = "<select><option selected=''></option></select>";
                    e.querySelectorAll("[selected]").length || v.push("\\[" + I + "*(?:value|" + F + ")");
                    e.querySelectorAll(":checked").length || v.push(":checked")
                });
                ft(function(e) {
                    var n = t.createElement("input");
                    n.setAttribute("type", "hidden");
                    e.appendChild(n).setAttribute("t", "");
                    e.querySelectorAll("[t^='']").length && v.push("[*^$]=" + I + "*(?:''|\"\")");
                    e.querySelectorAll(":enabled").length || v.push(":enabled", ":disabled");
                    e.querySelectorAll("*,:x");
                    v.push(",.*:")
                })
            }(r.matchesSelector = Y.test(g = p.webkitMatchesSelector || p.mozMatchesSelector || p.oMatchesSelector || p.msMatchesSelector)) && ft(function(e) {
                r.disconnectedMatch = g.call(e, "div");
                g.call(e, "[s!='']:x");
                m.push("!=", z)
            });
            v = v.length && new RegExp(v.join("|"));
            m = m.length && new RegExp(m.join("|"));
            y = Y.test(p.contains) || p.compareDocumentPosition ? function(e, t) {
                var n = e.nodeType === 9 ? e.documentElement: e, r = t && t.parentNode;
                return e === r||!!r && r.nodeType === 1&&!!(n.contains ? n.contains(r) : e.compareDocumentPosition && e.compareDocumentPosition(r) & 16)
            } : function(e, t) {
                if (t)
                    while (t = t.parentNode)
                        if (t === e)
                            return !0;
                return !1
            };
            L = p.compareDocumentPosition ? function(e, n) {
                if (e === n) {
                    k=!0;
                    return 0
                }
                var i = n.compareDocumentPosition && e.compareDocumentPosition && e.compareDocumentPosition(n);
                if (i)
                    return i & 1 ||!r.sortDetached && n.compareDocumentPosition(e) === i ? e === t || y(E, e)?-1 : n === t || y(E, n) ? 1 : l ? j.call(l, e) - j.call(l, n) : 0 : i & 4?-1 : 1;
                return e.compareDocumentPosition?-1 : 1
            } : function(e, n) {
                var r, i = 0, s = e.parentNode, o = n.parentNode, u = [e], a = [n];
                if (e === n) {
                    k=!0;
                    return 0
                }
                if (!s ||!o)
                    return e === t?-1 : n === t ? 1 : s?-1 : o ? 1 : l ? j.call(l, e) - j.call(l, n) : 0;
                if (s === o)
                    return ct(e, n);
                r = e;
                while (r = r.parentNode)
                    u.unshift(r);
                r = n;
                while (r = r.parentNode)
                    a.unshift(r);
                while (u[i] === a[i])
                    i++;
                return i ? ct(u[i], a[i]) : u[i] === E?-1 : a[i] === E ? 1 : 0
            };
            return t
        };
        ot.matches = function(e, t) {
            return ot(e, null, null, t)
        };
        ot.matchesSelector = function(e, t) {
            (e.ownerDocument || e) !== h && c(e);
            t = t.replace(J, "='$1']");
            if (r.matchesSelector && d && (!m ||!m.test(t)) && (!v ||!v.test(t)))
                try {
                    var n = g.call(e, t);
                    if (n || r.disconnectedMatch || e.document && e.document.nodeType !== 11)
                        return n
            } catch (i) {}
            return ot(t, h, null, [e]).length > 0
        };
        ot.contains = function(e, t) {
            (e.ownerDocument || e) !== h && c(e);
            return y(e, t)
        };
        ot.attr = function(e, n) {
            (e.ownerDocument || e) !== h && c(e);
            var i = s.attrHandle[n.toLowerCase()], o = i && M.call(s.attrHandle, n.toLowerCase()) ? i(e, n, !d): t;
            return o === t ? r.attributes ||!d ? e.getAttribute(n) : (o = e.getAttributeNode(n)) && o.specified ? o.value : null : o
        };
        ot.error = function(e) {
            throw new Error("Syntax error, unrecognized expression: " + e)
        };
        ot.uniqueSort = function(e) {
            var t, n = [], i = 0, s = 0;
            k=!r.detectDuplicates;
            l=!r.sortStable && e.slice(0);
            e.sort(L);
            if (k) {
                while (t = e[s++])
                    t === e[s] && (i = n.push(s));
                while (i--)
                    e.splice(n[i], 1)
            }
            return e
        };
        o = ot.getText = function(e) {
            var t, n = "", r = 0, i = e.nodeType;
            if (!i)
                for (; t = e[r]; r++)
                    n += o(t);
            else if (i === 1 || i === 9 || i === 11) {
                if (typeof e.textContent == "string")
                    return e.textContent;
                for (e = e.firstChild; e; e = e.nextSibling)
                    n += o(e)
            } else if (i === 3 || i === 4)
                return e.nodeValue;
            return n
        };
        s = ot.selectors = {
            cacheLength: 50,
            createPseudo: at,
            match: G,
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
                    e[1] = e[1].replace(rt, it);
                    e[3] = (e[4] || e[5] || "").replace(rt, it);
                    e[2] === "~=" && (e[3] = " " + e[3] + " ");
                    return e.slice(0, 4)
                },
                CHILD: function(e) {
                    e[1] = e[1].toLowerCase();
                    if (e[1].slice(0, 3) === "nth") {
                        e[3] || ot.error(e[0]);
                        e[4] =+ (e[4] ? e[5] + (e[6] || 1) : 2 * (e[3] === "even" || e[3] === "odd"));
                        e[5] =+ (e[7] + e[8] || e[3] === "odd")
                    } else 
                        e[3] && ot.error(e[0]);
                    return e
                },
                PSEUDO: function(e) {
                    var n, r=!e[5] && e[2];
                    if (G.CHILD.test(e[0]))
                        return null;
                    if (e[3] && e[4] !== t)
                        e[2] = e[4];
                    else if (r && K.test(r) && (n = mt(r, !0)) && (n = r.indexOf(")", r.length - n) - r.length)) {
                        e[0] = e[0].slice(0, n);
                        e[2] = r.slice(0, n)
                    }
                    return e.slice(0, 3)
                }
            },
            filter: {
                TAG: function(e) {
                    var t = e.replace(rt, it).toLowerCase();
                    return e === "*" ? function() {
                        return !0
                    } : function(e) {
                        return e.nodeName && e.nodeName.toLowerCase() === t
                    }
                },
                CLASS: function(e) {
                    var t = T[e + " "];
                    return t || (t = new RegExp("(^|" + I + ")" + e + "(" + I + "|$)")) && T(e, function(e) {
                        return t.test(typeof e.className == "string" && e.className || typeof e.getAttribute !== A && e.getAttribute("class") || "")
                    })
                },
                ATTR: function(e, t, n) {
                    return function(r) {
                        var i = ot.attr(r, e);
                        if (i == null)
                            return t === "!=";
                        if (!t)
                            return !0;
                        i += "";
                        return t === "=" ? i === n : t === "!=" ? i !== n : t === "^=" ? n && i.indexOf(n) === 0 : t === "*=" ? n && i.indexOf(n)>-1 : t === "$=" ? n && i.slice( - n.length) === n : t === "~=" ? (" " + i + " ").indexOf(n)>-1 : t === "|=" ? i === n || i.slice(0, n.length + 1) === n + "-" : !1
                    }
                },
                CHILD: function(e, t, n, r, i) {
                    var s = e.slice(0, 3) !== "nth", o = e.slice( - 4) !== "last", u = t === "of-type";
                    return r === 1 && i === 0 ? function(e) {
                        return !!e.parentNode
                    } : function(t, n, a) {
                        var f, l, c, h, p, d, v = s !== o ? "nextSibling": "previousSibling", m = t.parentNode, g = u && t.nodeName.toLowerCase(), y=!a&&!u;
                        if (m) {
                            if (s) {
                                while (v) {
                                    c = t;
                                    while (c = c[v])
                                        if (u ? c.nodeName.toLowerCase() === g : c.nodeType === 1)
                                            return !1;
                                    d = v = e === "only"&&!d && "nextSibling"
                                }
                                return !0
                            }
                            d = [o ? m.firstChild: m.lastChild];
                            if (o && y) {
                                l = m[b] || (m[b] = {});
                                f = l[e] || [];
                                p = f[0] === S && f[1];
                                h = f[0] === S && f[2];
                                c = p && m.childNodes[p];
                                while (c=++p && c && c[v] || (h = p = 0) || d.pop()
                                    )if (c.nodeType === 1&&++h && c === t) {
                                    l[e] = [S, p, h];
                                    break
                                }
                            } else if (y && (f = (t[b] || (t[b] = {}))[e]) && f[0] === S)
                                h = f[1];
                            else 
                                while (c=++p && c && c[v] || (h = p = 0) || d.pop()
                                    )if ((u ? c.nodeName.toLowerCase() === g : c.nodeType === 1)&&++h) {
                                y && ((c[b] || (c[b] = {}))[e] = [S, h]);
                                if (c === t)
                                    break
                            }
                            h -= i;
                            return h === r || h%r === 0 && h / r >= 0
                        }
                    }
                },
                PSEUDO: function(e, t) {
                    var n, r = s.pseudos[e] || s.setFilters[e.toLowerCase()] || ot.error("unsupported pseudo: " + e);
                    if (r[b])
                        return r(t);
                    if (r.length > 1) {
                        n = [e, e, "", t];
                        return s.setFilters.hasOwnProperty(e.toLowerCase()) ? at(function(e, n) {
                            var i, s = r(e, t), o = s.length;
                            while (o--) {
                                i = j.call(e, s[o]);
                                e[i]=!(n[i] = s[o])
                            }
                        }) : function(e) {
                            return r(e, 0, n)
                        }
                    }
                    return r
                }
            },
            pseudos: {
                not: at(function(e) {
                    var t = [], n = [], r = a(e.replace(W, "$1"));
                    return r[b] ? at(function(e, t, n, i) {
                        var s, o = r(e, null, i, []), u = e.length;
                        while (u--)
                            if (s = o[u])
                                e[u]=!(t[u] = s)
                    }) : function(e, i, s) {
                        t[0] = e;
                        r(t, null, s, n);
                        return !n.pop()
                    }
                }),
                has: at(function(e) {
                    return function(t) {
                        return ot(e, t).length > 0
                    }
                }),
                contains: at(function(e) {
                    return function(t) {
                        return (t.textContent || t.innerText || o(t)).indexOf(e)>-1
                    }
                }),
                lang: at(function(e) {
                    Q.test(e || "") || ot.error("unsupported lang: " + e);
                    e = e.replace(rt, it).toLowerCase();
                    return function(t) {
                        var n;
                        do 
                            if (n = d ? t.lang : t.getAttribute("xml:lang") || t.getAttribute("lang")) {
                                n = n.toLowerCase();
                                return n === e || n.indexOf(e + "-") === 0
                            }
                        while ((t = t.parentNode) && t.nodeType === 1);
                        return !1
                    }
                }),
                target: function(t) {
                    var n = e.location && e.location.hash;
                    return n && n.slice(1) === t.id
                },
                root: function(e) {
                    return e === p
                },
                focus: function(e) {
                    return e === h.activeElement && (!h.hasFocus || h.hasFocus())&&!!(e.type || e.href||~e.tabIndex)
                },
                enabled: function(e) {
                    return e.disabled===!1
                },
                disabled: function(e) {
                    return e.disabled===!0
                },
                checked: function(e) {
                    var t = e.nodeName.toLowerCase();
                    return t === "input"&&!!e.checked || t === "option"&&!!e.selected
                },
                selected: function(e) {
                    e.parentNode && e.parentNode.selectedIndex;
                    return e.selected===!0
                },
                empty: function(e) {
                    for (e = e.firstChild; e; e = e.nextSibling)
                        if (e.nodeName > "@" || e.nodeType === 3 || e.nodeType === 4)
                            return !1;
                    return !0
                },
                parent: function(e) {
                    return !s.pseudos.empty(e)
                },
                header: function(e) {
                    return tt.test(e.nodeName)
                },
                input: function(e) {
                    return et.test(e.nodeName)
                },
                button: function(e) {
                    var t = e.nodeName.toLowerCase();
                    return t === "input" && e.type === "button" || t === "button"
                },
                text: function(e) {
                    var t;
                    return e.nodeName.toLowerCase() === "input" && e.type === "text" && ((t = e.getAttribute("type")) == null || t.toLowerCase() === e.type)
                },
                first: dt(function() {
                    return [0]
                }),
                last: dt(function(e, t) {
                    return [t - 1]
                }),
                eq: dt(function(e, t, n) {
                    return [n < 0 ? n + t: n]
                }),
                even: dt(function(e, t) {
                    var n = 0;
                    for (; n < t; n += 2)
                        e.push(n);
                    return e
                }),
                odd: dt(function(e, t) {
                    var n = 1;
                    for (; n < t; n += 2)
                        e.push(n);
                    return e
                }),
                lt: dt(function(e, t, n) {
                    var r = n < 0 ? n + t: n;
                    for (; --r >= 0;)
                        e.push(r);
                    return e
                }),
                gt: dt(function(e, t, n) {
                    var r = n < 0 ? n + t: n;
                    for (; ++r < t;)
                        e.push(r);
                    return e
                })
            }
        };
        s.pseudos.nth = s.pseudos.eq;
        for (n in{
            radio: !0,
            checkbox: !0,
            file: !0,
            password: !0,
            image: !0
        })
            s.pseudos[n] = ht(n);
        for (n in{
            submit: !0,
            reset: !0
        })
            s.pseudos[n] = pt(n);
        vt.prototype = s.filters = s.pseudos;
        s.setFilters = new vt;
        a = ot.compile = function(e, t) {
            var n, r = [], i = [], s = C[e + " "];
            if (!s) {
                t || (t = mt(e));
                n = t.length;
                while (n--) {
                    s = St(t[n]);
                    s[b] ? r.push(s) : i.push(s)
                }
                s = C(e, xt(i, r))
            }
            return s
        };
        r.sortStable = b.split("").sort(L).join("") === b;
        r.detectDuplicates = k;
        c();
        r.sortDetached = ft(function(e) {
            return e.compareDocumentPosition(h.createElement("div")) & 1
        });
        ft(function(e) {
            e.innerHTML = "<a href='#'></a>";
            return e.firstChild.getAttribute("href") === "#"
        }) || lt("type|href|height|width", function(e, t, n) {
            if (!n)
                return e.getAttribute(t, t.toLowerCase() === "type" ? 1 : 2)
        });
        (!r.attributes ||!ft(function(e) {
            e.innerHTML = "<input/>";
            e.firstChild.setAttribute("value", "");
            return e.firstChild.getAttribute("value") === ""
        })) && lt("value", function(e, t, n) {
            if (!n && e.nodeName.toLowerCase() === "input")
                return e.defaultValue
        });
        ft(function(e) {
            return e.getAttribute("disabled") == null
        }) || lt(F, function(e, t, n) {
            var r;
            if (!n)
                return (r = e.getAttributeNode(t)) && r.specified ? r.value : e[t]===!0 ? t.toLowerCase() : null
        });
        w.find = ot;
        w.expr = ot.selectors;
        w.expr[":"] = w.expr.pseudos;
        w.unique = ot.uniqueSort;
        w.text = ot.getText;
        w.isXMLDoc = ot.isXML;
        w.contains = ot.contains
    })(e);
    var B = {};
    w.Callbacks = function(e) {
        e = typeof e == "string" ? B[e] || j(e) : w.extend({}, e);
        var n, r, i, s, o, u, a = [], f=!e.once && [], l = function(t) {
            r = e.memory && t;
            i=!0;
            o = u || 0;
            u = 0;
            s = a.length;
            n=!0;
            for (; a && o < s; o++)
                if (a[o].apply(t[0], t[1])===!1 && e.stopOnFalse) {
                    r=!1;
                    break
                }
            n=!1;
            a && (f ? f.length && l(f.shift()) : r ? a = [] : c.disable())
        }, c = {
            add: function() {
                if (a) {
                    var t = a.length;
                    (function i(t) {
                        w.each(t, function(t, n) {
                            var r = w.type(n);
                            r === "function" ? (!e.unique ||!c.has(n)) && a.push(n) : n && n.length && r !== "string" && i(n)
                        })
                    })(arguments);
                    if (n)
                        s = a.length;
                    else if (r) {
                        u = t;
                        l(r)
                    }
                }
                return this
            },
            remove: function() {
                a && w.each(arguments, function(e, t) {
                    var r;
                    while ((r = w.inArray(t, a, r))>-1) {
                        a.splice(r, 1);
                        if (n) {
                            r <= s && s--;
                            r <= o && o--
                        }
                    }
                });
                return this
            },
            has: function(e) {
                return e ? w.inArray(e, a)>-1 : !!a&&!!a.length
            },
            empty: function() {
                a = [];
                s = 0;
                return this
            },
            disable: function() {
                a = f = r = t;
                return this
            },
            disabled: function() {
                return !a
            },
            lock: function() {
                f = t;
                r || c.disable();
                return this
            },
            locked: function() {
                return !f
            },
            fireWith: function(e, t) {
                if (a && (!i || f)) {
                    t = t || [];
                    t = [e, t.slice ? t.slice(): t];
                    n ? f.push(t) : l(t)
                }
                return this
            },
            fire: function() {
                c.fireWith(this, arguments);
                return this
            },
            fired: function() {
                return !!i
            }
        };
        return c
    };
    w.extend({
        Deferred: function(e) {
            var t = [["resolve", "done", w.Callbacks("once memory"), "resolved"], ["reject", "fail", w.Callbacks("once memory"), "rejected"], ["notify", "progress", w.Callbacks("memory")]], n = "pending", r = {
                state: function() {
                    return n
                },
                always: function() {
                    i.done(arguments).fail(arguments);
                    return this
                },
                then: function() {
                    var e = arguments;
                    return w.Deferred(function(n) {
                        w.each(t, function(t, s) {
                            var o = s[0], u = w.isFunction(e[t]) && e[t];
                            i[s[1]](function() {
                                var e = u && u.apply(this, arguments);
                                e && w.isFunction(e.promise) ? e.promise().done(n.resolve).fail(n.reject).progress(n.notify) : n[o + "With"](this === r ? n.promise() : this, u ? [e] : arguments)
                            })
                        });
                        e = null
                    }).promise()
                },
                promise: function(e) {
                    return e != null ? w.extend(e, r) : r
                }
            }, i = {};
            r.pipe = r.then;
            w.each(t, function(e, s) {
                var o = s[2], u = s[3];
                r[s[1]] = o.add;
                u && o.add(function() {
                    n = u
                }, t[e^1][2].disable, t[2][2].lock);
                i[s[0]] = function() {
                    i[s[0] + "With"](this === i ? r : this, arguments);
                    return this
                };
                i[s[0] + "With"] = o.fireWith
            });
            r.promise(i);
            e && e.call(i, i);
            return i
        },
        when: function(e) {
            var t = 0, n = v.call(arguments), r = n.length, i = r !== 1 || e && w.isFunction(e.promise) ? r: 0, s = i === 1 ? e: w.Deferred(), o = function(e, t, n) {
                return function(r) {
                    t[e] = this;
                    n[e] = arguments.length > 1 ? v.call(arguments) : r;
                    n === u ? s.notifyWith(t, n) : --i || s.resolveWith(t, n)
                }
            }, u, a, f;
            if (r > 1) {
                u = new Array(r);
                a = new Array(r);
                f = new Array(r);
                for (; t < r; t++)
                    n[t] && w.isFunction(n[t].promise) ? n[t].promise().done(o(t, f, n)).fail(s.reject).progress(o(t, a, u)) : --i
            }
            i || s.resolveWith(f, n);
            return s.promise()
        }
    });
    w.support = function(t) {
        var n, r, s, u, a, f, l, c, h, p = o.createElement("div");
        p.setAttribute("className", "t");
        p.innerHTML = "  <link/><table></table><a href='/a'>a</a><input type='checkbox'/>";
        n = p.getElementsByTagName("*") || [];
        r = p.getElementsByTagName("a")[0];
        if (!r ||!r.style ||!n.length)
            return t;
        u = o.createElement("select");
        f = u.appendChild(o.createElement("option"));
        s = p.getElementsByTagName("input")[0];
        r.style.cssText = "top:1px;float:left;opacity:.5";
        t.getSetAttribute = p.className !== "t";
        t.leadingWhitespace = p.firstChild.nodeType === 3;
        t.tbody=!p.getElementsByTagName("tbody").length;
        t.htmlSerialize=!!p.getElementsByTagName("link").length;
        t.style = /top/.test(r.getAttribute("style"));
        t.hrefNormalized = r.getAttribute("href") === "/a";
        t.opacity = /^0.5/.test(r.style.opacity);
        t.cssFloat=!!r.style.cssFloat;
        t.checkOn=!!s.value;
        t.optSelected = f.selected;
        t.enctype=!!o.createElement("form").enctype;
        t.html5Clone = o.createElement("nav").cloneNode(!0).outerHTML !== "<:nav></:nav>";
        t.inlineBlockNeedsLayout=!1;
        t.shrinkWrapBlocks=!1;
        t.pixelPosition=!1;
        t.deleteExpando=!0;
        t.noCloneEvent=!0;
        t.reliableMarginRight=!0;
        t.boxSizingReliable=!0;
        s.checked=!0;
        t.noCloneChecked = s.cloneNode(!0).checked;
        u.disabled=!0;
        t.optDisabled=!f.disabled;
        try {
            delete p.test
        } catch (d) {
            t.deleteExpando=!1
        }
        s = o.createElement("input");
        s.setAttribute("value", "");
        t.input = s.getAttribute("value") === "";
        s.value = "t";
        s.setAttribute("type", "radio");
        t.radioValue = s.value === "t";
        s.setAttribute("checked", "t");
        s.setAttribute("name", "t");
        a = o.createDocumentFragment();
        a.appendChild(s);
        t.appendChecked = s.checked;
        t.checkClone = a.cloneNode(!0).cloneNode(!0).lastChild.checked;
        if (p.attachEvent) {
            p.attachEvent("onclick", function() {
                t.noCloneEvent=!1
            });
            p.cloneNode(!0).click()
        }
        for (h in{
            submit: !0,
            change: !0,
            focusin: !0
        }) {
            p.setAttribute(l = "on" + h, "t");
            t[h + "Bubbles"] = l in e || p.attributes[l].expando===!1
        }
        p.style.backgroundClip = "content-box";
        p.cloneNode(!0).style.backgroundClip = "";
        t.clearCloneStyle = p.style.backgroundClip === "content-box";
        for (h in w(t))
            break;
        t.ownLast = h !== "0";
        w(function() {
            var n, r, s, u = "padding:0;margin:0;border:0;display:block;box-sizing:content-box;-moz-box-sizing:content-box;-webkit-box-sizing:content-box;", a = o.getElementsByTagName("body")[0];
            if (!a)
                return;
            n = o.createElement("div");
            n.style.cssText = "border:0;width:0;height:0;position:absolute;top:0;left:-9999px;margin-top:1px";
            a.appendChild(n).appendChild(p);
            p.innerHTML = "<table><tr><td></td><td>t</td></tr></table>";
            s = p.getElementsByTagName("td");
            s[0].style.cssText = "padding:0;margin:0;border:0;display:none";
            c = s[0].offsetHeight === 0;
            s[0].style.display = "";
            s[1].style.display = "none";
            t.reliableHiddenOffsets = c && s[0].offsetHeight === 0;
            p.innerHTML = "";
            p.style.cssText = "box-sizing:border-box;-moz-box-sizing:border-box;-webkit-box-sizing:border-box;padding:1px;border:1px;display:block;width:4px;margin-top:1%;position:absolute;top:1%;";
            w.swap(a, a.style.zoom != null ? {
                zoom: 1
            } : {}, function() {
                t.boxSizing = p.offsetWidth === 4
            });
            if (e.getComputedStyle) {
                t.pixelPosition = (e.getComputedStyle(p, null) || {}).top !== "1%";
                t.boxSizingReliable = (e.getComputedStyle(p, null) || {
                    width: "4px"
                }).width === "4px";
                r = p.appendChild(o.createElement("div"));
                r.style.cssText = p.style.cssText = u;
                r.style.marginRight = r.style.width = "0";
                p.style.width = "1px";
                t.reliableMarginRight=!parseFloat((e.getComputedStyle(r, null) || {}).marginRight)
            }
            if (typeof p.style.zoom !== i) {
                p.innerHTML = "";
                p.style.cssText = u + "width:1px;padding:1px;display:inline;zoom:1";
                t.inlineBlockNeedsLayout = p.offsetWidth === 3;
                p.style.display = "block";
                p.innerHTML = "<div></div>";
                p.firstChild.style.width = "5px";
                t.shrinkWrapBlocks = p.offsetWidth !== 3;
                t.inlineBlockNeedsLayout && (a.style.zoom = 1)
            }
            a.removeChild(n);
            n = p = s = r = null
        });
        n = u = a = f = r = s = null;
        return t
    }({});
    var F = /(?:\{[\s\S]*\}|\[[\s\S]*\])$/, I = /([A-Z])/g;
    w.extend({
        cache: {},
        noData: {
            applet: !0,
            embed: !0,
            object: "clsid:D27CDB6E-AE6D-11cf-96B8-444553540000"
        },
        hasData: function(e) {
            e = e.nodeType ? w.cache[e[w.expando]] : e[w.expando];
            return !!e&&!z(e)
        },
        data: function(e, t, n) {
            return q(e, t, n)
        },
        removeData: function(e, t) {
            return R(e, t)
        },
        _data: function(e, t, n) {
            return q(e, t, n, !0)
        },
        _removeData: function(e, t) {
            return R(e, t, !0)
        },
        acceptData: function(e) {
            if (e.nodeType && e.nodeType !== 1 && e.nodeType !== 9)
                return !1;
            var t = e.nodeName && w.noData[e.nodeName.toLowerCase()];
            return !t || t!==!0 && e.getAttribute("classid") === t
        }
    });
    w.fn.extend({
        data: function(e, n) {
            var r, i, s = null, o = 0, u = this[0];
            if (e === t) {
                if (this.length) {
                    s = w.data(u);
                    if (u.nodeType === 1&&!w._data(u, "parsedAttrs")) {
                        r = u.attributes;
                        for (; o < r.length; o++) {
                            i = r[o].name;
                            if (i.indexOf("data-") === 0) {
                                i = w.camelCase(i.slice(5));
                                U(u, i, s[i])
                            }
                        }
                        w._data(u, "parsedAttrs", !0)
                    }
                }
                return s
            }
            return typeof e == "object" ? this.each(function() {
                w.data(this, e)
            }) : arguments.length > 1 ? this.each(function() {
                w.data(this, e, n)
            }) : u ? U(u, e, w.data(u, e)) : null
        },
        removeData: function(e) {
            return this.each(function() {
                w.removeData(this, e)
            })
        }
    });
    w.extend({
        queue: function(e, t, n) {
            var r;
            if (e) {
                t = (t || "fx") + "queue";
                r = w._data(e, t);
                n && (!r || w.isArray(n) ? r = w._data(e, t, w.makeArray(n)) : r.push(n));
                return r || []
            }
        },
        dequeue: function(e, t) {
            t = t || "fx";
            var n = w.queue(e, t), r = n.length, i = n.shift(), s = w._queueHooks(e, t), o = function() {
                w.dequeue(e, t)
            };
            if (i === "inprogress") {
                i = n.shift();
                r--
            }
            if (i) {
                t === "fx" && n.unshift("inprogress");
                delete s.stop;
                i.call(e, o, s)
            }
            !r && s && s.empty.fire()
        },
        _queueHooks: function(e, t) {
            var n = t + "queueHooks";
            return w._data(e, n) || w._data(e, n, {
                empty: w.Callbacks("once memory").add(function() {
                    w._removeData(e, t + "queue");
                    w._removeData(e, n)
                })
            })
        }
    });
    w.fn.extend({
        queue: function(e, n) {
            var r = 2;
            if (typeof e != "string") {
                n = e;
                e = "fx";
                r--
            }
            return arguments.length < r ? w.queue(this[0], e) : n === t ? this : this.each(function() {
                var t = w.queue(this, e, n);
                w._queueHooks(this, e);
                e === "fx" && t[0] !== "inprogress" && w.dequeue(this, e)
            })
        },
        dequeue: function(e) {
            return this.each(function() {
                w.dequeue(this, e)
            })
        },
        delay: function(e, t) {
            e = w.fx ? w.fx.speeds[e] || e : e;
            t = t || "fx";
            return this.queue(t, function(t, n) {
                var r = setTimeout(t, e);
                n.stop = function() {
                    clearTimeout(r)
                }
            })
        },
        clearQueue: function(e) {
            return this.queue(e || "fx", [])
        },
        promise: function(e, n) {
            var r, i = 1, s = w.Deferred(), o = this, u = this.length, a = function() {
                --i || s.resolveWith(o, [o])
            };
            if (typeof e != "string") {
                n = e;
                e = t
            }
            e = e || "fx";
            while (u--) {
                r = w._data(o[u], e + "queueHooks");
                if (r && r.empty) {
                    i++;
                    r.empty.add(a)
                }
            }
            a();
            return s.promise(n)
        }
    });
    var W, X, V = /[\t\r\n\f]/g, $ = /\r/g, J = /^(?:input|select|textarea|button|object)$/i, K = /^(?:a|area)$/i, Q = /^(?:checked|selected)$/i, G = w.support.getSetAttribute, Y = w.support.input;
    w.fn.extend({
        attr: function(e, t) {
            return w.access(this, w.attr, e, t, arguments.length > 1)
        },
        removeAttr: function(e) {
            return this.each(function() {
                w.removeAttr(this, e)
            })
        },
        prop: function(e, t) {
            return w.access(this, w.prop, e, t, arguments.length > 1)
        },
        removeProp: function(e) {
            e = w.propFix[e] || e;
            return this.each(function() {
                try {
                    this[e] = t;
                    delete this[e]
                } catch (n) {}
            })
        },
        addClass: function(e) {
            var t, n, r, i, s, o = 0, u = this.length, a = typeof e == "string" && e;
            if (w.isFunction(e))
                return this.each(function(t) {
                    w(this).addClass(e.call(this, t, this.className))
                });
            if (a) {
                t = (e || "").match(S) || [];
                for (; o < u; o++) {
                    n = this[o];
                    r = n.nodeType === 1 && (n.className ? (" " + n.className + " ").replace(V, " ") : " ");
                    if (r) {
                        s = 0;
                        while (i = t[s++])
                            r.indexOf(" " + i + " ") < 0 && (r += i + " ");
                        n.className = w.trim(r)
                    }
                }
            }
            return this
        },
        removeClass: function(e) {
            var t, n, r, i, s, o = 0, u = this.length, a = arguments.length === 0 || typeof e == "string" && e;
            if (w.isFunction(e))
                return this.each(function(t) {
                    w(this).removeClass(e.call(this, t, this.className))
                });
            if (a) {
                t = (e || "").match(S) || [];
                for (; o < u; o++) {
                    n = this[o];
                    r = n.nodeType === 1 && (n.className ? (" " + n.className + " ").replace(V, " ") : "");
                    if (r) {
                        s = 0;
                        while (i = t[s++])
                            while (r.indexOf(" " + i + " ") >= 0)
                                r = r.replace(" " + i + " ", " ");
                        n.className = e ? w.trim(r) : ""
                    }
                }
            }
            return this
        },
        toggleClass: function(e, t) {
            var n = typeof e;
            return typeof t == "boolean" && n === "string" ? t ? this.addClass(e) : this.removeClass(e) : w.isFunction(e) ? this.each(function(n) {
                w(this).toggleClass(e.call(this, n, this.className, t), t)
            }) : this.each(function() {
                if (n === "string") {
                    var t, r = 0, s = w(this), o = e.match(S) || [];
                    while (t = o[r++])
                        s.hasClass(t) ? s.removeClass(t) : s.addClass(t)
                } else if (n === i || n === "boolean") {
                    this.className && w._data(this, "__className__", this.className);
                    this.className = this.className || e===!1 ? "" : w._data(this, "__className__") || ""
                }
            })
        },
        hasClass: function(e) {
            var t = " " + e + " ", n = 0, r = this.length;
            for (; n < r; n++)
                if (this[n].nodeType === 1 && (" " + this[n].className + " ").replace(V, " ").indexOf(t) >= 0)
                    return !0;
            return !1
        },
        val: function(e) {
            var n, r, i, s = this[0];
            if (!arguments.length) {
                if (s) {
                    r = w.valHooks[s.type] || w.valHooks[s.nodeName.toLowerCase()];
                    if (r && "get"in r && (n = r.get(s, "value")) !== t)
                        return n;
                    n = s.value;
                    return typeof n == "string" ? n.replace($, "") : n == null ? "" : n
                }
                return 
            }
            i = w.isFunction(e);
            return this.each(function(n) {
                var s;
                if (this.nodeType !== 1)
                    return;
                i ? s = e.call(this, n, w(this).val()) : s = e;
                s == null ? s = "" : typeof s == "number" ? s += "" : w.isArray(s) && (s = w.map(s, function(e) {
                    return e == null ? "" : e + ""
                }));
                r = w.valHooks[this.type] || w.valHooks[this.nodeName.toLowerCase()];
                if (!r ||!("set"in r) || r.set(this, s, "value") === t)
                    this.value = s
            })
        }
    });
    w.extend({
        valHooks: {
            option: {
                get: function(e) {
                    var t = w.find.attr(e, "value");
                    return t != null ? t : e.text
                }
            },
            select: {
                get: function(e) {
                    var t, n, r = e.options, i = e.selectedIndex, s = e.type === "select-one" || i < 0, o = s ? null: [], u = s ? i + 1: r.length, a = i < 0 ? u: s ? i: 0;
                    for (; a < u; a++) {
                        n = r[a];
                        if ((n.selected || a === i) && (w.support.optDisabled?!n.disabled : n.getAttribute("disabled") === null) && (!n.parentNode.disabled ||!w.nodeName(n.parentNode, "optgroup"))
                            ) {
                            t = w(n).val();
                            if (s)
                                return t;
                            o.push(t)
                        }
                    }
                    return o
                },
                set: function(e, t) {
                    var n, r, i = e.options, s = w.makeArray(t), o = i.length;
                    while (o--) {
                        r = i[o];
                        if (r.selected = w.inArray(w(r).val(), s) >= 0)
                            n=!0
                    }
                    n || (e.selectedIndex =- 1);
                    return s
                }
            }
        },
        attr: function(e, n, r) {
            var s, o, u = e.nodeType;
            if (!e || u === 3 || u === 8 || u === 2)
                return;
            if (typeof e.getAttribute === i)
                return w.prop(e, n, r);
            if (u !== 1 ||!w.isXMLDoc(e)) {
                n = n.toLowerCase();
                s = w.attrHooks[n] || (w.expr.match.bool.test(n) ? X : W)
            }
            if (r === t) {
                if (s && "get"in s && (o = s.get(e, n)) !== null)
                    return o;
                o = w.find.attr(e, n);
                return o == null ? t : o
            }
            if (r !== null) {
                if (s && "set"in s && (o = s.set(e, r, n)) !== t)
                    return o;
                e.setAttribute(n, r + "");
                return r
            }
            w.removeAttr(e, n)
        },
        removeAttr: function(e, t) {
            var n, r, i = 0, s = t && t.match(S);
            if (s && e.nodeType === 1)
                while (n = s[i++]) {
                    r = w.propFix[n] || n;
                    w.expr.match.bool.test(n) ? Y && G ||!Q.test(n) ? e[r]=!1 : e[w.camelCase("default-" + n)] = e[r]=!1 : w.attr(e, n, "");
                    e.removeAttribute(G ? n : r)
                }
        },
        attrHooks: {
            type: {
                set: function(e, t) {
                    if (!w.support.radioValue && t === "radio" && w.nodeName(e, "input")) {
                        var n = e.value;
                        e.setAttribute("type", t);
                        n && (e.value = n);
                        return t
                    }
                }
            }
        },
        propFix: {
            "for": "htmlFor",
            "class": "className"
        },
        prop: function(e, n, r) {
            var i, s, o, u = e.nodeType;
            if (!e || u === 3 || u === 8 || u === 2)
                return;
            o = u !== 1 ||!w.isXMLDoc(e);
            if (o) {
                n = w.propFix[n] || n;
                s = w.propHooks[n]
            }
            return r !== t ? s && "set"in s && (i = s.set(e, r, n)) !== t ? i : e[n] = r : s && "get"in s && (i = s.get(e, n)) !== null ? i : e[n]
        },
        propHooks: {
            tabIndex: {
                get: function(e) {
                    var t = w.find.attr(e, "tabindex");
                    return t ? parseInt(t, 10) : J.test(e.nodeName) || K.test(e.nodeName) && e.href ? 0 : - 1
                }
            }
        }
    });
    X = {
        set: function(e, t, n) {
            t===!1 ? w.removeAttr(e, n) : Y && G ||!Q.test(n) ? e.setAttribute(!G && w.propFix[n] || n, n) : e[w.camelCase("default-" + n)] = e[n]=!0;
            return n
        }
    };
    w.each(w.expr.match.bool.source.match(/\w+/g), function(e, n) {
        var r = w.expr.attrHandle[n] || w.find.attr;
        w.expr.attrHandle[n] = Y && G ||!Q.test(n) ? function(e, n, i) {
            var s = w.expr.attrHandle[n], o = i ? t: (w.expr.attrHandle[n] = t) != r(e, n, i) ? n.toLowerCase(): null;
            w.expr.attrHandle[n] = s;
            return o
        } : function(e, n, r) {
            return r ? t : e[w.camelCase("default-" + n)] ? n.toLowerCase() : null
        }
    });
    if (!Y ||!G)
        w.attrHooks.value = {
            set: function(e, t, n) {
                if (!w.nodeName(e, "input"))
                    return W && W.set(e, t, n);
                    e.defaultValue = t
                }
            };
    if (!G) {
        W = {
            set: function(e, n, r) {
                var i = e.getAttributeNode(r);
                i || e.setAttributeNode(i = e.ownerDocument.createAttribute(r));
                i.value = n += "";
                return r === "value" || n === e.getAttribute(r) ? n : t
            }
        };
        w.expr.attrHandle.id = w.expr.attrHandle.name = w.expr.attrHandle.coords = function(e, n, r) {
            var i;
            return r ? t : (i = e.getAttributeNode(n)) && i.value !== "" ? i.value : null
        };
        w.valHooks.button = {
            get: function(e, n) {
                var r = e.getAttributeNode(n);
                return r && r.specified ? r.value : t
            },
            set: W.set
        };
        w.attrHooks.contenteditable = {
            set: function(e, t, n) {
                W.set(e, t === ""?!1 : t, n)
            }
        };
        w.each(["width", "height"], function(e, t) {
            w.attrHooks[t] = {
                set: function(e, n) {
                    if (n === "") {
                        e.setAttribute(t, "auto");
                        return n
                    }
                }
            }
        })
    }
    w.support.hrefNormalized || w.each(["href", "src"], function(e, t) {
        w.propHooks[t] = {
            get: function(e) {
                return e.getAttribute(t, 4)
            }
        }
    });
    w.support.style || (w.attrHooks.style = {
        get: function(e) {
            return e.style.cssText || t
        },
        set: function(e, t) {
            return e.style.cssText = t + ""
        }
    });
    w.support.optSelected || (w.propHooks.selected = {
        get: function(e) {
            var t = e.parentNode;
            if (t) {
                t.selectedIndex;
                t.parentNode && t.parentNode.selectedIndex
            }
            return null
        }
    });
    w.each(["tabIndex", "readOnly", "maxLength", "cellSpacing", "cellPadding", "rowSpan", "colSpan", "useMap", "frameBorder", "contentEditable"], function() {
        w.propFix[this.toLowerCase()] = this
    });
    w.support.enctype || (w.propFix.enctype = "encoding");
    w.each(["radio", "checkbox"], function() {
        w.valHooks[this] = {
            set: function(e, t) {
                if (w.isArray(t))
                    return e.checked = w.inArray(w(e).val(), t) >= 0
            }
        };
        w.support.checkOn || (w.valHooks[this].get = function(e) {
            return e.getAttribute("value") === null ? "on" : e.value
        })
    });
    var Z = /^(?:input|select|textarea)$/i, et = /^key/, tt = /^(?:mouse|contextmenu)|click/, nt = /^(?:focusinfocus|focusoutblur)$/, rt = /^([^.]*)(?:\.(.+)|)$/;
    w.event = {
        global: {},
        add: function(e, n, r, s, o) {
            var u, a, f, l, c, h, p, d, v, m, g, y = w._data(e);
            if (!y)
                return;
            if (r.handler) {
                l = r;
                r = l.handler;
                o = l.selector
            }
            r.guid || (r.guid = w.guid++);
            (a = y.events) || (a = y.events = {});
            if (!(h = y.handle)) {
                h = y.handle = function(e) {
                    return typeof w === i||!!e && w.event.triggered === e.type ? t : w.event.dispatch.apply(h.elem, arguments)
                };
                h.elem = e
            }
            n = (n || "").match(S) || [""];
            f = n.length;
            while (f--) {
                u = rt.exec(n[f]) || [];
                v = g = u[1];
                m = (u[2] || "").split(".").sort();
                if (!v)
                    continue;
                c = w.event.special[v] || {};
                v = (o ? c.delegateType : c.bindType) || v;
                c = w.event.special[v] || {};
                p = w.extend({
                    type: v,
                    origType: g,
                    data: s,
                    handler: r,
                    guid: r.guid,
                    selector: o,
                    needsContext: o && w.expr.match.needsContext.test(o),
                    namespace: m.join(".")
                }, l);
                if (!(d = a[v])) {
                    d = a[v] = [];
                    d.delegateCount = 0;
                    if (!c.setup || c.setup.call(e, s, m, h)===!1)
                        e.addEventListener ? e.addEventListener(v, h, !1) : e.attachEvent && e.attachEvent("on" + v, h)
                    }
                if (c.add) {
                    c.add.call(e, p);
                    p.handler.guid || (p.handler.guid = r.guid)
                }
                o ? d.splice(d.delegateCount++, 0, p) : d.push(p);
                w.event.global[v]=!0
            }
            e = null
        },
        remove: function(e, t, n, r, i) {
            var s, o, u, a, f, l, c, h, p, d, v, m = w.hasData(e) && w._data(e);
            if (!m ||!(l = m.events))
                return;
            t = (t || "").match(S) || [""];
            f = t.length;
            while (f--) {
                u = rt.exec(t[f]) || [];
                p = v = u[1];
                d = (u[2] || "").split(".").sort();
                if (!p) {
                    for (p in l)
                        w.event.remove(e, p + t[f], n, r, !0);
                    continue
                }
                c = w.event.special[p] || {};
                p = (r ? c.delegateType : c.bindType) || p;
                h = l[p] || [];
                u = u[2] && new RegExp("(^|\\.)" + d.join("\\.(?:.*\\.|)") + "(\\.|$)");
                a = s = h.length;
                while (s--) {
                    o = h[s];
                    if ((i || v === o.origType) && (!n || n.guid === o.guid) && (!u || u.test(o.namespace)) && (!r || r === o.selector || r === "**" && o.selector)) {
                        h.splice(s, 1);
                        o.selector && h.delegateCount--;
                        c.remove && c.remove.call(e, o)
                    }
                }
                if (a&&!h.length) {
                    (!c.teardown || c.teardown.call(e, d, m.handle)===!1) && w.removeEvent(e, p, m.handle);
                    delete l[p]
                }
            }
            if (w.isEmptyObject(l)) {
                delete m.handle;
                w._removeData(e, "events")
            }
        },
        trigger: function(n, r, i, s) {
            var u, a, f, l, c, h, p, d = [i || o], v = y.call(n, "type") ? n.type: n, m = y.call(n, "namespace") ? n.namespace.split("."): [];
            f = h = i = i || o;
            if (i.nodeType === 3 || i.nodeType === 8)
                return;
            if (nt.test(v + w.event.triggered))
                return;
            if (v.indexOf(".") >= 0) {
                m = v.split(".");
                v = m.shift();
                m.sort()
            }
            a = v.indexOf(":") < 0 && "on" + v;
            n = n[w.expando] ? n : new w.Event(v, typeof n == "object" && n);
            n.isTrigger = s ? 2 : 3;
            n.namespace = m.join(".");
            n.namespace_re = n.namespace ? new RegExp("(^|\\.)" + m.join("\\.(?:.*\\.|)") + "(\\.|$)") : null;
            n.result = t;
            n.target || (n.target = i);
            r = r == null ? [n] : w.makeArray(r, [n]);
            c = w.event.special[v] || {};
            if (!s && c.trigger && c.trigger.apply(i, r)===!1)
                return;
            if (!s&&!c.noBubble&&!w.isWindow(i)) {
                l = c.delegateType || v;
                nt.test(l + v) || (f = f.parentNode);
                for (; f; f = f.parentNode) {
                    d.push(f);
                    h = f
                }
                h === (i.ownerDocument || o) && d.push(h.defaultView || h.parentWindow || e)
            }
            p = 0;
            while ((f = d[p++])&&!n.isPropagationStopped()) {
                n.type = p > 1 ? l : c.bindType || v;
                u = (w._data(f, "events") || {})[n.type] && w._data(f, "handle");
                u && u.apply(f, r);
                u = a && f[a];
                u && w.acceptData(f) && u.apply && u.apply(f, r)===!1 && n.preventDefault()
            }
            n.type = v;
            if (!s&&!n.isDefaultPrevented() && (!c._default || c._default.apply(d.pop(), r)===!1) && w.acceptData(i) && a && i[v]&&!w.isWindow(i)) {
                h = i[a];
                h && (i[a] = null);
                w.event.triggered = v;
                try {
                    i[v]()
                } catch (g) {}
                w.event.triggered = t;
                h && (i[a] = h)
            }
            return n.result
        },
        dispatch: function(e) {
            e = w.event.fix(e);
            var n, r, i, s, o, u = [], a = v.call(arguments), f = (w._data(this, "events") || {})[e.type] || [], l = w.event.special[e.type] || {};
            a[0] = e;
            e.delegateTarget = this;
            if (l.preDispatch && l.preDispatch.call(this, e)===!1)
                return;
            u = w.event.handlers.call(this, e, f);
            n = 0;
            while ((s = u[n++])&&!e.isPropagationStopped()) {
                e.currentTarget = s.elem;
                o = 0;
                while ((i = s.handlers[o++])&&!e.isImmediatePropagationStopped())
                    if (!e.namespace_re || e.namespace_re.test(i.namespace)) {
                        e.handleObj = i;
                        e.data = i.data;
                        r = ((w.event.special[i.origType] || {}).handle || i.handler).apply(s.elem, a);
                        if (r !== t && (e.result = r)===!1) {
                            e.preventDefault();
                            e.stopPropagation()
                        }
                    }
                }
            l.postDispatch && l.postDispatch.call(this, e);
            return e.result
        },
        handlers: function(e, n) {
            var r, i, s, o, u = [], a = n.delegateCount, f = e.target;
            if (a && f.nodeType && (!e.button || e.type !== "click"))
                for (; f != this; f = f.parentNode || this)
                    if (f.nodeType === 1 && (f.disabled!==!0 || e.type !== "click")) {
                        s = [];
                        for (o = 0; o < a; o++) {
                            i = n[o];
                            r = i.selector + " ";
                            s[r] === t && (s[r] = i.needsContext ? w(r, this).index(f) >= 0 : w.find(r, this, null, [f]).length);
                            s[r] && s.push(i)
                        }
                        s.length && u.push({
                            elem: f,
                            handlers: s
                        })
                    }
            a < n.length && u.push({
                elem: this,
                handlers: n.slice(a)
            });
            return u
        },
        fix: function(e) {
            if (e[w.expando])
                return e;
            var t, n, r, i = e.type, s = e, u = this.fixHooks[i];
            u || (this.fixHooks[i] = u = tt.test(i) ? this.mouseHooks : et.test(i) ? this.keyHooks : {});
            r = u.props ? this.props.concat(u.props) : this.props;
            e = new w.Event(s);
            t = r.length;
            while (t--) {
                n = r[t];
                e[n] = s[n]
            }
            e.target || (e.target = s.srcElement || o);
            e.target.nodeType === 3 && (e.target = e.target.parentNode);
            e.metaKey=!!e.metaKey;
            return u.filter ? u.filter(e, s) : e
        },
        props: "altKey bubbles cancelable ctrlKey currentTarget eventPhase metaKey relatedTarget shiftKey target timeStamp view which".split(" "),
        fixHooks: {},
        keyHooks: {
            props: "char charCode key keyCode".split(" "),
            filter: function(e, t) {
                e.which == null && (e.which = t.charCode != null ? t.charCode : t.keyCode);
                return e
            }
        },
        mouseHooks: {
            props: "button buttons clientX clientY fromElement offsetX offsetY pageX pageY screenX screenY toElement".split(" "),
            filter: function(e, n) {
                var r, i, s, u = n.button, a = n.fromElement;
                if (e.pageX == null && n.clientX != null) {
                    i = e.target.ownerDocument || o;
                    s = i.documentElement;
                    r = i.body;
                    e.pageX = n.clientX + (s && s.scrollLeft || r && r.scrollLeft || 0) - (s && s.clientLeft || r && r.clientLeft || 0);
                    e.pageY = n.clientY + (s && s.scrollTop || r && r.scrollTop || 0) - (s && s.clientTop || r && r.clientTop || 0)
                }
                !e.relatedTarget && a && (e.relatedTarget = a === e.target ? n.toElement : a);
                !e.which && u !== t && (e.which = u & 1 ? 1 : u & 2 ? 3 : u & 4 ? 2 : 0);
                return e
            }
        },
        special: {
            load: {
                noBubble: !0
            },
            focus: {
                trigger: function() {
                    if (this !== ot() && this.focus)
                        try {
                            this.focus();
                            return !1
                    } catch (e) {}
                },
                delegateType: "focusin"
            },
            blur: {
                trigger: function() {
                    if (this === ot() && this.blur) {
                        this.blur();
                        return !1
                    }
                },
                delegateType: "focusout"
            },
            click: {
                trigger: function() {
                    if (w.nodeName(this, "input") && this.type === "checkbox" && this.click) {
                        this.click();
                        return !1
                    }
                },
                _default: function(e) {
                    return w.nodeName(e.target, "a")
                }
            },
            beforeunload: {
                postDispatch: function(e) {
                    e.result !== t && (e.originalEvent.returnValue = e.result)
                }
            }
        },
        simulate: function(e, t, n, r) {
            var i = w.extend(new w.Event, n, {
                type: e,
                isSimulated: !0,
                originalEvent: {}
            });
            r ? w.event.trigger(i, null, t) : w.event.dispatch.call(t, i);
            i.isDefaultPrevented() && n.preventDefault()
        }
    };
    w.removeEvent = o.removeEventListener ? function(e, t, n) {
        e.removeEventListener && e.removeEventListener(t, n, !1)
    } : function(e, t, n) {
        var r = "on" + t;
        if (e.detachEvent) {
            typeof e[r] === i && (e[r] = null);
            e.detachEvent(r, n)
        }
    };
    w.Event = function(e, t) {
        if (!(this instanceof w.Event))
            return new w.Event(e, t);
        if (e && e.type) {
            this.originalEvent = e;
            this.type = e.type;
            this.isDefaultPrevented = e.defaultPrevented || e.returnValue===!1 || e.getPreventDefault && e.getPreventDefault() ? it : st
        } else 
            this.type = e;
        t && w.extend(this, t);
        this.timeStamp = e && e.timeStamp || w.now();
        this[w.expando]=!0
    };
    w.Event.prototype = {
        isDefaultPrevented: st,
        isPropagationStopped: st,
        isImmediatePropagationStopped: st,
        preventDefault: function() {
            var e = this.originalEvent;
            this.isDefaultPrevented = it;
            if (!e)
                return;
            e.preventDefault ? e.preventDefault() : e.returnValue=!1
        },
        stopPropagation: function() {
            var e = this.originalEvent;
            this.isPropagationStopped = it;
            if (!e)
                return;
            e.stopPropagation && e.stopPropagation();
            e.cancelBubble=!0
        },
        stopImmediatePropagation: function() {
            this.isImmediatePropagationStopped = it;
            this.stopPropagation()
        }
    };
    w.each({
        mouseenter: "mouseover",
        mouseleave: "mouseout"
    }, function(e, t) {
        w.event.special[e] = {
            delegateType: t,
            bindType: t,
            handle: function(e) {
                var n, r = this, i = e.relatedTarget, s = e.handleObj;
                if (!i || i !== r&&!w.contains(r, i)) {
                    e.type = s.origType;
                    n = s.handler.apply(this, arguments);
                    e.type = t
                }
                return n
            }
        }
    });
    w.support.submitBubbles || (w.event.special.submit = {
        setup: function() {
            if (w.nodeName(this, "form"))
                return !1;
            w.event.add(this, "click._submit keypress._submit", function(e) {
                var n = e.target, r = w.nodeName(n, "input") || w.nodeName(n, "button") ? n.form: t;
                if (r&&!w._data(r, "submitBubbles")) {
                    w.event.add(r, "submit._submit", function(e) {
                        e._submit_bubble=!0
                    });
                    w._data(r, "submitBubbles", !0)
                }
            })
        },
        postDispatch: function(e) {
            if (e._submit_bubble) {
                delete e._submit_bubble;
                this.parentNode&&!e.isTrigger && w.event.simulate("submit", this.parentNode, e, !0)
            }
        },
        teardown: function() {
            if (w.nodeName(this, "form"))
                return !1;
            w.event.remove(this, "._submit")
        }
    });
    w.support.changeBubbles || (w.event.special.change = {
        setup: function() {
            if (Z.test(this.nodeName)) {
                if (this.type === "checkbox" || this.type === "radio") {
                    w.event.add(this, "propertychange._change", function(e) {
                        e.originalEvent.propertyName === "checked" && (this._just_changed=!0)
                    });
                    w.event.add(this, "click._change", function(e) {
                        this._just_changed&&!e.isTrigger && (this._just_changed=!1);
                        w.event.simulate("change", this, e, !0)
                    })
                }
                return !1
            }
            w.event.add(this, "beforeactivate._change", function(e) {
                var t = e.target;
                if (Z.test(t.nodeName)&&!w._data(t, "changeBubbles")) {
                    w.event.add(t, "change._change", function(e) {
                        this.parentNode&&!e.isSimulated&&!e.isTrigger && w.event.simulate("change", this.parentNode, e, !0)
                    });
                    w._data(t, "changeBubbles", !0)
                }
            })
        },
        handle: function(e) {
            var t = e.target;
            if (this !== t || e.isSimulated || e.isTrigger || t.type !== "radio" && t.type !== "checkbox")
                return e.handleObj.handler.apply(this, arguments)
        },
        teardown: function() {
            w.event.remove(this, "._change");
            return !Z.test(this.nodeName)
        }
    });
    w.support.focusinBubbles || w.each({
        focus: "focusin",
        blur: "focusout"
    }, function(e, t) {
        var n = 0, r = function(e) {
            w.event.simulate(t, e.target, w.event.fix(e), !0)
        };
        w.event.special[t] = {
            setup: function() {
                n++===0 && o.addEventListener(e, r, !0)
            },
            teardown: function() {
                --n === 0 && o.removeEventListener(e, r, !0)
            }
        }
    });
    w.fn.extend({
        on: function(e, n, r, i, s) {
            var o, u;
            if (typeof e == "object") {
                if (typeof n != "string") {
                    r = r || n;
                    n = t
                }
                for (o in e)
                    this.on(o, n, r, e[o], s);
                return this
            }
            if (r == null && i == null) {
                i = n;
                r = n = t
            } else if (i == null)
                if (typeof n == "string") {
                    i = r;
                    r = t
                } else {
                    i = r;
                    r = n;
                    n = t
                }
            if (i===!1)
                i = st;
            else if (!i)
                return this;
            if (s === 1) {
                u = i;
                i = function(e) {
                    w().off(e);
                    return u.apply(this, arguments)
                };
                i.guid = u.guid || (u.guid = w.guid++)
            }
            return this.each(function() {
                w.event.add(this, e, i, r, n)
            })
        },
        one: function(e, t, n, r) {
            return this.on(e, t, n, r, 1)
        },
        off: function(e, n, r) {
            var i, s;
            if (e && e.preventDefault && e.handleObj) {
                i = e.handleObj;
                w(e.delegateTarget).off(i.namespace ? i.origType + "." + i.namespace : i.origType, i.selector, i.handler);
                return this
            }
            if (typeof e == "object") {
                for (s in e)
                    this.off(s, n, e[s]);
                return this
            }
            if (n===!1 || typeof n == "function") {
                r = n;
                n = t
            }
            r===!1 && (r = st);
            return this.each(function() {
                w.event.remove(this, e, r, n)
            })
        },
        trigger: function(e, t) {
            return this.each(function() {
                w.event.trigger(e, t, this)
            })
        },
        triggerHandler: function(e, t) {
            var n = this[0];
            if (n)
                return w.event.trigger(e, t, n, !0)
        }
    });
    var ut = /^.[^:#\[\.,]*$/, at = /^(?:parents|prev(?:Until|All))/, ft = w.expr.match.needsContext, lt = {
        children: !0,
        contents: !0,
        next: !0,
        prev: !0
    };
    w.fn.extend({
        find: function(e) {
            var t, n = [], r = this, i = r.length;
            if (typeof e != "string")
                return this.pushStack(w(e).filter(function() {
                    for (t = 0; t < i; t++)
                        if (w.contains(r[t], this))
                            return !0
                        }));
            for (t = 0; t < i; t++)
                w.find(e, r[t], n);
            n = this.pushStack(i > 1 ? w.unique(n) : n);
            n.selector = this.selector ? this.selector + " " + e : e;
            return n
        },
        has: function(e) {
            var t, n = w(e, this), r = n.length;
            return this.filter(function() {
                for (t = 0; t < r; t++)
                    if (w.contains(this, n[t]))
                        return !0
            })
        },
        not: function(e) {
            return this.pushStack(ht(this, e || [], !0))
        },
        filter: function(e) {
            return this.pushStack(ht(this, e || [], !1))
        },
        is: function(e) {
            return !!ht(this, typeof e == "string" && ft.test(e) ? w(e) : e || [], !1).length
        },
        closest: function(e, t) {
            var n, r = 0, i = this.length, s = [], o = ft.test(e) || typeof e != "string" ? w(e, t || this.context): 0;
            for (; r < i; r++)
                for (n = this[r]; n && n !== t; n = n.parentNode)
                    if (n.nodeType < 11 && (o ? o.index(n)>-1 : n.nodeType === 1 && w.find.matchesSelector(n, e))) {
                        n = s.push(n);
                        break
                    }
            return this.pushStack(s.length > 1 ? w.unique(s) : s)
        },
        index: function(e) {
            return e ? typeof e == "string" ? w.inArray(this[0], w(e)) : w.inArray(e.jquery ? e[0] : e, this) : this[0] && this[0].parentNode ? this.first().prevAll().length : - 1
        },
        add: function(e, t) {
            var n = typeof e == "string" ? w(e, t): w.makeArray(e && e.nodeType ? [e] : e), r = w.merge(this.get(), n);
            return this.pushStack(w.unique(r))
        },
        addBack: function(e) {
            return this.add(e == null ? this.prevObject : this.prevObject.filter(e))
        }
    });
    w.each({
        parent: function(e) {
            var t = e.parentNode;
            return t && t.nodeType !== 11 ? t : null
        },
        parents: function(e) {
            return w.dir(e, "parentNode")
        },
        parentsUntil: function(e, t, n) {
            return w.dir(e, "parentNode", n)
        },
        next: function(e) {
            return ct(e, "nextSibling")
        },
        prev: function(e) {
            return ct(e, "previousSibling")
        },
        nextAll: function(e) {
            return w.dir(e, "nextSibling")
        },
        prevAll: function(e) {
            return w.dir(e, "previousSibling")
        },
        nextUntil: function(e, t, n) {
            return w.dir(e, "nextSibling", n)
        },
        prevUntil: function(e, t, n) {
            return w.dir(e, "previousSibling", n)
        },
        siblings: function(e) {
            return w.sibling((e.parentNode || {}).firstChild, e)
        },
        children: function(e) {
            return w.sibling(e.firstChild)
        },
        contents: function(e) {
            return w.nodeName(e, "iframe") ? e.contentDocument || e.contentWindow.document : w.merge([], e.childNodes)
        }
    }, function(e, t) {
        w.fn[e] = function(n, r) {
            var i = w.map(this, t, n);
            e.slice( - 5) !== "Until" && (r = n);
            r && typeof r == "string" && (i = w.filter(r, i));
            if (this.length > 1) {
                lt[e] || (i = w.unique(i));
                at.test(e) && (i = i.reverse())
            }
            return this.pushStack(i)
        }
    });
    w.extend({
        filter: function(e, t, n) {
            var r = t[0];
            n && (e = ":not(" + e + ")");
            return t.length === 1 && r.nodeType === 1 ? w.find.matchesSelector(r, e) ? [r] : [] : w.find.matches(e, w.grep(t, function(e) {
                return e.nodeType === 1
            }))
        },
        dir: function(e, n, r) {
            var i = [], s = e[n];
            while (s && s.nodeType !== 9 && (r === t || s.nodeType !== 1 ||!w(s).is(r))) {
                s.nodeType === 1 && i.push(s);
                s = s[n]
            }
            return i
        },
        sibling: function(e, t) {
            var n = [];
            for (; e; e = e.nextSibling)
                e.nodeType === 1 && e !== t && n.push(e);
            return n
        }
    });
    var dt = "abbr|article|aside|audio|bdi|canvas|data|datalist|details|figcaption|figure|footer|header|hgroup|mark|meter|nav|output|progress|section|summary|time|video", vt = / jQuery\d+="(?:null|\d+)"/g, mt = new RegExp("<(?:" + dt + ")[\\s/>]", "i"), gt = /^\s+/, yt = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\w:]+)[^>]*)\/>/gi, bt = /<([\w:]+)/, wt = /<tbody/i, Et = /<|&#?\w+;/, St = /<(?:script|style|link)/i, xt = /^(?:checkbox|radio)$/i, Tt = /checked\s*(?:[^=]|=\s*.checked.)/i, Nt = /^$|\/(?:java|ecma)script/i, Ct = /^true\/(.*)/, kt = /^\s*<!(?:\[CDATA\[|--)|(?:\]\]|--)>\s*$/g, Lt = {
        option: [1, "<select multiple='multiple'>", "</select>"],
        legend: [1, "<fieldset>", "</fieldset>"],
        area: [1, "<map>", "</map>"],
        param: [1, "<object>", "</object>"],
        thead: [1, "<table>", "</table>"],
        tr: [2, "<table><tbody>", "</tbody></table>"],
        col: [2, "<table><tbody></tbody><colgroup>", "</colgroup></table>"],
        td: [3, "<table><tbody><tr>", "</tr></tbody></table>"],
        _default: w.support.htmlSerialize ? [0, "", ""]: [1, "X<div>", "</div>"]
    }, At = pt(o), Ot = At.appendChild(o.createElement("div"));
    Lt.optgroup = Lt.option;
    Lt.tbody = Lt.tfoot = Lt.colgroup = Lt.caption = Lt.thead;
    Lt.th = Lt.td;
    w.fn.extend({
        text: function(e) {
            return w.access(this, function(e) {
                return e === t ? w.text(this) : this.empty().append((this[0] && this[0].ownerDocument || o).createTextNode(e))
            }, null, e, arguments.length)
        },
        append: function() {
            return this.domManip(arguments, function(e) {
                if (this.nodeType === 1 || this.nodeType === 11 || this.nodeType === 9) {
                    var t = Mt(this, e);
                    t.appendChild(e)
                }
            })
        },
        prepend: function() {
            return this.domManip(arguments, function(e) {
                if (this.nodeType === 1 || this.nodeType === 11 || this.nodeType === 9) {
                    var t = Mt(this, e);
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
            var n, r = e ? w.filter(e, this): this, i = 0;
            for (; (n = r[i]) != null; i++) {
                !t && n.nodeType === 1 && w.cleanData(jt(n));
                if (n.parentNode) {
                    t && w.contains(n.ownerDocument, n) && Pt(jt(n, "script"));
                    n.parentNode.removeChild(n)
                }
            }
            return this
        },
        empty: function() {
            var e, t = 0;
            for (; (e = this[t]) != null; t++) {
                e.nodeType === 1 && w.cleanData(jt(e, !1));
                while (e.firstChild)
                    e.removeChild(e.firstChild);
                e.options && w.nodeName(e, "select") && (e.options.length = 0)
            }
            return this
        },
        clone: function(e, t) {
            e = e == null?!1 : e;
            t = t == null ? e : t;
            return this.map(function() {
                return w.clone(this, e, t)
            })
        },
        html: function(e) {
            return w.access(this, function(e) {
                var n = this[0] || {}, r = 0, i = this.length;
                if (e === t)
                    return n.nodeType === 1 ? n.innerHTML.replace(vt, "") : t;
                if (typeof e == "string"&&!St.test(e) && (w.support.htmlSerialize ||!mt.test(e)) && (w.support.leadingWhitespace ||!gt.test(e))&&!Lt[(bt.exec(e) || ["", ""])[1].toLowerCase()]) {
                    e = e.replace(yt, "<$1></$2>");
                    try {
                        for (; r < i; r++) {
                            n = this[r] || {};
                            if (n.nodeType === 1) {
                                w.cleanData(jt(n, !1));
                                n.innerHTML = e
                            }
                        }
                        n = 0
                    } catch (s) {}
                }
                n && this.empty().append(e)
            }, null, e, arguments.length)
        },
        replaceWith: function() {
            var e = w.map(this, function(e) {
                return [e.nextSibling, e.parentNode]
            }), t = 0;
            this.domManip(arguments, function(n) {
                var r = e[t++], i = e[t++];
                if (i) {
                    r && r.parentNode !== i && (r = this.nextSibling);
                    w(this).remove();
                    i.insertBefore(n, r)
                }
            }, !0);
            return t ? this : this.remove()
        },
        detach: function(e) {
            return this.remove(e, !0)
        },
        domManip: function(e, t, n) {
            e = p.apply([], e);
            var r, i, s, o, u, a, f = 0, l = this.length, c = this, h = l - 1, d = e[0], v = w.isFunction(d);
            if (v ||!(l <= 1 || typeof d != "string" || w.support.checkClone ||!Tt.test(d)))
                return this.each(function(r) {
                    var i = c.eq(r);
                    v && (e[0] = d.call(this, r, i.html()));
                    i.domManip(e, t, n)
                });
            if (l) {
                a = w.buildFragment(e, this[0].ownerDocument, !1, !n && this);
                r = a.firstChild;
                a.childNodes.length === 1 && (a = r);
                if (r) {
                    o = w.map(jt(a, "script"), _t);
                    s = o.length;
                    for (; f < l; f++) {
                        i = a;
                        if (f !== h) {
                            i = w.clone(i, !0, !0);
                            s && w.merge(o, jt(i, "script"))
                        }
                        t.call(this[f], i, f)
                    }
                    if (s) {
                        u = o[o.length - 1].ownerDocument;
                        w.map(o, Dt);
                        for (f = 0; f < s; f++) {
                            i = o[f];
                            Nt.test(i.type || "")&&!w._data(i, "globalEval") && w.contains(u, i) && (i.src ? w._evalUrl(i.src) : w.globalEval((i.text || i.textContent || i.innerHTML || "").replace(kt, "")))
                        }
                    }
                    a = r = null
                }
            }
            return this
        }
    });
    w.each({
        appendTo: "append",
        prependTo: "prepend",
        insertBefore: "before",
        insertAfter: "after",
        replaceAll: "replaceWith"
    }, function(e, t) {
        w.fn[e] = function(e) {
            var n, r = 0, i = [], s = w(e), o = s.length - 1;
            for (; r <= o; r++) {
                n = r === o ? this : this.clone(!0);
                w(s[r])[t](n);
                d.apply(i, n.get())
            }
            return this.pushStack(i)
        }
    });
    w.extend({
        clone: function(e, t, n) {
            var r, i, s, o, u, a = w.contains(e.ownerDocument, e);
            if (w.support.html5Clone || w.isXMLDoc(e) ||!mt.test("<" + e.nodeName + ">"))
                s = e.cloneNode(!0);
            else {
                Ot.innerHTML = e.outerHTML;
                Ot.removeChild(s = Ot.firstChild)
            }
            if ((!w.support.noCloneEvent ||!w.support.noCloneChecked) && (e.nodeType === 1 || e.nodeType === 11)&&!w.isXMLDoc(e)) {
                r = jt(s);
                u = jt(e);
                for (o = 0; (i = u[o]) != null; ++o)
                    r[o] && Bt(i, r[o])
            }
            if (t)
                if (n) {
                    u = u || jt(e);
                    r = r || jt(s);
                    for (o = 0; (i = u[o]) != null; o++)
                        Ht(i, r[o])
                } else 
                    Ht(e, s);
            r = jt(s, "script");
            r.length > 0 && Pt(r, !a && jt(e, "script"));
            r = u = i = null;
            return s
        },
        buildFragment: function(e, t, n, r) {
            var i, s, o, u, a, f, l, c = e.length, h = pt(t), p = [], d = 0;
            for (; d < c; d++) {
                s = e[d];
                if (s || s === 0)
                    if (w.type(s) === "object")
                        w.merge(p, s.nodeType ? [s] : s);
                    else if (!Et.test(s))
                        p.push(t.createTextNode(s));
                    else {
                        u = u || h.appendChild(t.createElement("div"));
                        a = (bt.exec(s) || ["", ""])[1].toLowerCase();
                        l = Lt[a] || Lt._default;
                        u.innerHTML = l[1] + s.replace(yt, "<$1></$2>") + l[2];
                        i = l[0];
                        while (i--)
                            u = u.lastChild;
                            !w.support.leadingWhitespace && gt.test(s) && p.push(t.createTextNode(gt.exec(s)[0]));
                            if (!w.support.tbody) {
                                s = a === "table"&&!wt.test(s) ? u.firstChild : l[1] === "<table>"&&!wt.test(s) ? u : 0;
                                i = s && s.childNodes.length;
                                while (i--)
                                    w.nodeName(f = s.childNodes[i], "tbody")&&!f.childNodes.length && s.removeChild(f)
                                }
                                w.merge(p, u.childNodes);
                                u.textContent = "";
                                while (u.firstChild)
                                    u.removeChild(u.firstChild);
                                    u = h.lastChild
                        }
                }
            u && h.removeChild(u);
            w.support.appendChecked || w.grep(jt(p, "input"), Ft);
            d = 0;
            while (s = p[d++]) {
                if (r && w.inArray(s, r)!==-1)
                    continue;
                o = w.contains(s.ownerDocument, s);
                u = jt(h.appendChild(s), "script");
                o && Pt(u);
                if (n) {
                    i = 0;
                    while (s = u[i++])
                        Nt.test(s.type || "") && n.push(s)
                    }
            }
            u = null;
            return h
        },
        cleanData: function(e, t) {
            var n, r, s, o, u = 0, a = w.expando, f = w.cache, l = w.support.deleteExpando, h = w.event.special;
            for (; (n = e[u]) != null; u++)
                if (t || w.acceptData(n)) {
                    s = n[a];
                    o = s && f[s];
                    if (o) {
                        if (o.events)
                            for (r in o.events)
                                h[r] ? w.event.remove(n, r) : w.removeEvent(n, r, o.handle);
                                if (f[s]) {
                                    delete f[s];
                                    l ? delete n[a] : typeof n.removeAttribute !== i ? n.removeAttribute(a) : n[a] = null;
                                    c.push(s)
                                }
                            }
                }
            },
        _evalUrl: function(e) {
            return w.ajax({
                url: e,
                type: "GET",
                dataType: "script",
                async: !1,
                global: !1,
                "throws": !0
            })
        }
    });
    w.fn.extend({
        wrapAll: function(e) {
            if (w.isFunction(e))
                return this.each(function(t) {
                    w(this).wrapAll(e.call(this, t))
                });
            if (this[0]) {
                var t = w(e, this[0].ownerDocument).eq(0).clone(!0);
                this[0].parentNode && t.insertBefore(this[0]);
                t.map(function() {
                    var e = this;
                    while (e.firstChild && e.firstChild.nodeType === 1)
                        e = e.firstChild;
                    return e
                }).append(this)
            }
            return this
        },
        wrapInner: function(e) {
            return w.isFunction(e) ? this.each(function(t) {
                w(this).wrapInner(e.call(this, t))
            }) : this.each(function() {
                var t = w(this), n = t.contents();
                n.length ? n.wrapAll(e) : t.append(e)
            })
        },
        wrap: function(e) {
            var t = w.isFunction(e);
            return this.each(function(n) {
                w(this).wrapAll(t ? e.call(this, n) : e)
            })
        },
        unwrap: function() {
            return this.parent().each(function() {
                w.nodeName(this, "body") || w(this).replaceWith(this.childNodes)
            }).end()
        }
    });
    var It, qt, Rt, Ut = /alpha\([^)]*\)/i, zt = /opacity\s*=\s*([^)]*)/, Wt = /^(top|right|bottom|left)$/, Xt = /^(none|table(?!-c[ea]).+)/, Vt = /^margin/, $t = new RegExp("^(" + E + ")(.*)$", "i"), Jt = new RegExp("^(" + E + ")(?!px)[a-z%]+$", "i"), Kt = new RegExp("^([+-])=(" + E + ")", "i"), Qt = {
        BODY: "block"
    }, Gt = {
        position: "absolute",
        visibility: "hidden",
        display: "block"
    }, Yt = {
        letterSpacing: 0,
        fontWeight: 400
    }, Zt = ["Top", "Right", "Bottom", "Left"], en = ["Webkit", "O", "Moz", "ms"];
    w.fn.extend({
        css: function(e, n) {
            return w.access(this, function(e, n, r) {
                var i, s, o = {}, u = 0;
                if (w.isArray(n)) {
                    s = qt(e);
                    i = n.length;
                    for (; u < i; u++)
                        o[n[u]] = w.css(e, n[u], !1, s);
                    return o
                }
                return r !== t ? w.style(e, n, r) : w.css(e, n)
            }, e, n, arguments.length > 1)
        },
        show: function() {
            return rn(this, !0)
        },
        hide: function() {
            return rn(this)
        },
        toggle: function(e) {
            return typeof e == "boolean" ? e ? this.show() : this.hide() : this.each(function() {
                nn(this) ? w(this).show() : w(this).hide()
            })
        }
    });
    w.extend({
        cssHooks: {
            opacity: {
                get: function(e, t) {
                    if (t) {
                        var n = Rt(e, "opacity");
                        return n === "" ? "1" : n
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
            "float": w.support.cssFloat ? "cssFloat": "styleFloat"
        },
        style: function(e, n, r, i) {
            if (!e || e.nodeType === 3 || e.nodeType === 8 ||!e.style)
                return;
            var s, o, u, a = w.camelCase(n), f = e.style;
            n = w.cssProps[a] || (w.cssProps[a] = tn(f, a));
            u = w.cssHooks[n] || w.cssHooks[a];
            if (r === t)
                return u && "get"in u && (s = u.get(e, !1, i)) !== t ? s : f[n];
            o = typeof r;
            if (o === "string" && (s = Kt.exec(r))) {
                r = (s[1] + 1) * s[2] + parseFloat(w.css(e, n));
                o = "number"
            }
            if (r == null || o === "number" && isNaN(r))
                return;
            o === "number"&&!w.cssNumber[a] && (r += "px");
            !w.support.clearCloneStyle && r === "" && n.indexOf("background") === 0 && (f[n] = "inherit");
            if (!u ||!("set"in u) || (r = u.set(e, r, i)) !== t)
                try {
                    f[n] = r
            } catch (l) {}
        },
        css: function(e, n, r, i) {
            var s, o, u, a = w.camelCase(n);
            n = w.cssProps[a] || (w.cssProps[a] = tn(e.style, a));
            u = w.cssHooks[n] || w.cssHooks[a];
            u && "get"in u && (o = u.get(e, !0, r));
            o === t && (o = Rt(e, n, i));
            o === "normal" && n in Yt && (o = Yt[n]);
            if (r === "" || r) {
                s = parseFloat(o);
                return r===!0 || w.isNumeric(s) ? s || 0 : o
            }
            return o
        }
    });
    if (e.getComputedStyle) {
        qt = function(t) {
            return e.getComputedStyle(t, null)
        };
        Rt = function(e, n, r) {
            var i, s, o, u = r || qt(e), a = u ? u.getPropertyValue(n) || u[n]: t, f = e.style;
            if (u) {
                a === ""&&!w.contains(e.ownerDocument, e) && (a = w.style(e, n));
                if (Jt.test(a) && Vt.test(n)) {
                    i = f.width;
                    s = f.minWidth;
                    o = f.maxWidth;
                    f.minWidth = f.maxWidth = f.width = a;
                    a = u.width;
                    f.width = i;
                    f.minWidth = s;
                    f.maxWidth = o
                }
            }
            return a
        }
    } else if (o.documentElement.currentStyle) {
        qt = function(e) {
            return e.currentStyle
        };
        Rt = function(e, n, r) {
            var i, s, o, u = r || qt(e), a = u ? u[n]: t, f = e.style;
            a == null && f && f[n] && (a = f[n]);
            if (Jt.test(a)&&!Wt.test(n)) {
                i = f.left;
                s = e.runtimeStyle;
                o = s && s.left;
                o && (s.left = e.currentStyle.left);
                f.left = n === "fontSize" ? "1em" : a;
                a = f.pixelLeft + "px";
                f.left = i;
                o && (s.left = o)
            }
            return a === "" ? "auto" : a
        }
    }
    w.each(["height", "width"], function(e, t) {
        w.cssHooks[t] = {
            get: function(e, n, r) {
                if (n)
                    return e.offsetWidth === 0 && Xt.test(w.css(e, "display")) ? w.swap(e, Gt, function() {
                        return un(e, t, r)
                    }) : un(e, t, r)
            },
            set: function(e, n, r) {
                var i = r && qt(e);
                return sn(e, n, r ? on(e, t, r, w.support.boxSizing && w.css(e, "boxSizing", !1, i) === "border-box", i) : 0)
            }
        }
    });
    w.support.opacity || (w.cssHooks.opacity = {
        get: function(e, t) {
            return zt.test((t && e.currentStyle ? e.currentStyle.filter : e.style.filter) || "") ? .01 * parseFloat(RegExp.$1) + "" : t ? "1" : ""
        },
        set: function(e, t) {
            var n = e.style, r = e.currentStyle, i = w.isNumeric(t) ? "alpha(opacity=" + t * 100 + ")": "", s = r && r.filter || n.filter || "";
            n.zoom = 1;
            if ((t >= 1 || t === "") && w.trim(s.replace(Ut, "")) === "" && n.removeAttribute) {
                n.removeAttribute("filter");
                if (t === "" || r&&!r.filter)
                    return 
            }
            n.filter = Ut.test(s) ? s.replace(Ut, i) : s + " " + i
        }
    });
    w(function() {
        w.support.reliableMarginRight || (w.cssHooks.marginRight = {
            get: function(e, t) {
                if (t)
                    return w.swap(e, {
                        display: "inline-block"
                    }, Rt, [e, "marginRight"])
            }
        });
        !w.support.pixelPosition && w.fn.position && w.each(["top", "left"], function(e, t) {
            w.cssHooks[t] = {
                get: function(e, n) {
                    if (n) {
                        n = Rt(e, t);
                        return Jt.test(n) ? w(e).position()[t] + "px" : n
                    }
                }
            }
        })
    });
    if (w.expr && w.expr.filters) {
        w.expr.filters.hidden = function(e) {
            return e.offsetWidth <= 0 && e.offsetHeight <= 0 ||!w.support.reliableHiddenOffsets && (e.style && e.style.display || w.css(e, "display")) === "none"
        };
        w.expr.filters.visible = function(e) {
            return !w.expr.filters.hidden(e)
        }
    }
    w.each({
        margin: "",
        padding: "",
        border: "Width"
    }, function(e, t) {
        w.cssHooks[e + t] = {
            expand: function(n) {
                var r = 0, i = {}, s = typeof n == "string" ? n.split(" "): [n];
                for (; r < 4; r++)
                    i[e + Zt[r] + t] = s[r] || s[r - 2] || s[0];
                return i
            }
        };
        Vt.test(e) || (w.cssHooks[e + t].set = sn)
    });
    var ln = /%20/g, cn = /\[\]$/, hn = /\r?\n/g, pn = /^(?:submit|button|image|reset|file)$/i, dn = /^(?:input|select|textarea|keygen)/i;
    w.fn.extend({
        serialize: function() {
            return w.param(this.serializeArray())
        },
        serializeArray: function() {
            return this.map(function() {
                var e = w.prop(this, "elements");
                return e ? w.makeArray(e) : this
            }).filter(function() {
                var e = this.type;
                return this.name&&!w(this).is(":disabled") && dn.test(this.nodeName)&&!pn.test(e) && (this.checked ||!xt.test(e))
            }).map(function(e, t) {
                var n = w(this).val();
                return n == null ? null : w.isArray(n) ? w.map(n, function(e) {
                    return {
                        name: t.name,
                        value: e.replace(hn, "\r\n")
                    }
                }) : {
                    name: t.name,
                    value: n.replace(hn, "\r\n")
                }
            }).get()
        }
    });
    w.param = function(e, n) {
        var r, i = [], s = function(e, t) {
            t = w.isFunction(t) ? t() : t == null ? "" : t;
            i[i.length] = encodeURIComponent(e) + "=" + encodeURIComponent(t)
        };
        n === t && (n = w.ajaxSettings && w.ajaxSettings.traditional);
        if (w.isArray(e) || e.jquery&&!w.isPlainObject(e))
            w.each(e, function() {
                s(this.name, this.value)
            });
        else 
            for (r in e)
                vn(r, e[r], n, s);
        return i.join("&").replace(ln, "+")
    };
    w.each("blur focus focusin focusout load resize scroll unload click dblclick mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave change select submit keydown keypress keyup error contextmenu".split(" "), function(e, t) {
        w.fn[t] = function(e, n) {
            return arguments.length > 0 ? this.on(t, null, e, n) : this.trigger(t)
        }
    });
    w.fn.extend({
        hover: function(e, t) {
            return this.mouseenter(e).mouseleave(t || e)
        },
        bind: function(e, t, n) {
            return this.on(e, null, t, n)
        },
        unbind: function(e, t) {
            return this.off(e, null, t)
        },
        delegate: function(e, t, n, r) {
            return this.on(t, e, n, r)
        },
        undelegate: function(e, t, n) {
            return arguments.length === 1 ? this.off(e, "**") : this.off(t, e || "**", n)
        }
    });
    var mn, gn, yn = w.now(), bn = /\?/, wn = /#.*$/, En = /([?&])_=[^&]*/, Sn = /^(.*?):[ \t]*([^\r\n]*)\r?$/mg, xn = /^(?:about|app|app-storage|.+-extension|file|res|widget):$/, Tn = /^(?:GET|HEAD)$/, Nn = /^\/\//, Cn = /^([\w.+-]+:)(?:\/\/([^\/?#:]*)(?::(\d+)|)|)/, kn = w.fn.load, Ln = {}, An = {}, On = "*/".concat("*");
    try {
        gn = s.href
    } catch (Mn) {
        gn = o.createElement("a");
        gn.href = "";
        gn = gn.href
    }
    mn = Cn.exec(gn.toLowerCase()) || [];
    w.fn.load = function(e, n, r) {
        if (typeof e != "string" && kn)
            return kn.apply(this, arguments);
        var i, s, o, u = this, a = e.indexOf(" ");
        if (a >= 0) {
            i = e.slice(a, e.length);
            e = e.slice(0, a)
        }
        if (w.isFunction(n)) {
            r = n;
            n = t
        } else 
            n && typeof n == "object" && (o = "POST");
        u.length > 0 && w.ajax({
            url: e,
            type: o,
            dataType: "html",
            data: n
        }).done(function(e) {
            s = arguments;
            u.html(i ? w("<div>").append(w.parseHTML(e)).find(i) : e)
        }).complete(r && function(e, t) {
            u.each(r, s || [e.responseText, t, e])
        });
        return this
    };
    w.each(["ajaxStart", "ajaxStop", "ajaxComplete", "ajaxError", "ajaxSuccess", "ajaxSend"], function(e, t) {
        w.fn[t] = function(e) {
            return this.on(t, e)
        }
    });
    w.extend({
        active: 0,
        lastModified: {},
        etag: {},
        ajaxSettings: {
            url: gn,
            type: "GET",
            isLocal: xn.test(mn[1]),
            global: !0,
            processData: !0,
            async: !0,
            contentType: "application/x-www-form-urlencoded; charset=UTF-8",
            accepts: {
                "*": On,
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
                "text json": w.parseJSON,
                "text xml": w.parseXML
            },
            flatOptions: {
                url: !0,
                context: !0
            }
        },
        ajaxSetup: function(e, t) {
            return t ? Pn(Pn(e, w.ajaxSettings), t) : Pn(w.ajaxSettings, e)
        },
        ajaxPrefilter: _n(Ln),
        ajaxTransport: _n(An),
        ajax: function(e, n) {
            function N(e, n, r, i) {
                var l, g, y, E, S, T = n;
                if (b === 2)
                    return;
                b = 2;
                u && clearTimeout(u);
                f = t;
                o = i || "";
                x.readyState = e > 0 ? 4 : 0;
                l = e >= 200 && e < 300 || e === 304;
                r && (E = Hn(c, x, r));
                E = Bn(c, E, x, l);
                if (l) {
                    if (c.ifModified) {
                        S = x.getResponseHeader("Last-Modified");
                        S && (w.lastModified[s] = S);
                        S = x.getResponseHeader("etag");
                        S && (w.etag[s] = S)
                    }
                    if (e === 204 || c.type === "HEAD")
                        T = "nocontent";
                    else if (e === 304)
                        T = "notmodified";
                    else {
                        T = E.state;
                        g = E.data;
                        y = E.error;
                        l=!y
                    }
                } else {
                    y = T;
                    if (e ||!T) {
                        T = "error";
                        e < 0 && (e = 0)
                    }
                }
                x.status = e;
                x.statusText = (n || T) + "";
                l ? d.resolveWith(h, [g, T, x]) : d.rejectWith(h, [x, T, y]);
                x.statusCode(m);
                m = t;
                a && p.trigger(l ? "ajaxSuccess" : "ajaxError", [x, c, l ? g: y]);
                v.fireWith(h, [x, T]);
                if (a) {
                    p.trigger("ajaxComplete", [x, c]);
                    --w.active || w.event.trigger("ajaxStop")
                }
            }
            if (typeof e == "object") {
                n = e;
                e = t
            }
            n = n || {};
            var r, i, s, o, u, a, f, l, c = w.ajaxSetup({}, n), h = c.context || c, p = c.context && (h.nodeType || h.jquery) ? w(h): w.event, d = w.Deferred(), v = w.Callbacks("once memory"), m = c.statusCode || {}, g = {}, y = {}, b = 0, E = "canceled", x = {
                readyState: 0,
                getResponseHeader: function(e) {
                    var t;
                    if (b === 2) {
                        if (!l) {
                            l = {};
                            while (t = Sn.exec(o))
                                l[t[1].toLowerCase()] = t[2]
                        }
                        t = l[e.toLowerCase()]
                    }
                    return t == null ? null : t
                },
                getAllResponseHeaders: function() {
                    return b === 2 ? o : null
                },
                setRequestHeader: function(e, t) {
                    var n = e.toLowerCase();
                    if (!b) {
                        e = y[n] = y[n] || e;
                        g[e] = t
                    }
                    return this
                },
                overrideMimeType: function(e) {
                    b || (c.mimeType = e);
                    return this
                },
                statusCode: function(e) {
                    var t;
                    if (e)
                        if (b < 2)
                            for (t in e)
                                m[t] = [m[t], e[t]];
                        else 
                            x.always(e[x.status]);
                    return this
                },
                abort: function(e) {
                    var t = e || E;
                    f && f.abort(t);
                    N(0, t);
                    return this
                }
            };
            d.promise(x).complete = v.add;
            x.success = x.done;
            x.error = x.fail;
            c.url = ((e || c.url || gn) + "").replace(wn, "").replace(Nn, mn[1] + "//");
            c.type = n.method || n.type || c.method || c.type;
            c.dataTypes = w.trim(c.dataType || "*").toLowerCase().match(S) || [""];
            if (c.crossDomain == null) {
                r = Cn.exec(c.url.toLowerCase());
                c.crossDomain=!(!r || r[1] === mn[1] && r[2] === mn[2] && (r[3] || (r[1] === "http:" ? "80" : "443")) === (mn[3] || (mn[1] === "http:" ? "80" : "443")))
            }
            c.data && c.processData && typeof c.data != "string" && (c.data = w.param(c.data, c.traditional));
            Dn(Ln, c, n, x);
            if (b === 2)
                return x;
            a = c.global;
            a && w.active++===0 && w.event.trigger("ajaxStart");
            c.type = c.type.toUpperCase();
            c.hasContent=!Tn.test(c.type);
            s = c.url;
            if (!c.hasContent) {
                if (c.data) {
                    s = c.url += (bn.test(s) ? "&" : "?") + c.data;
                    delete c.data
                }
                c.cache===!1 && (c.url = En.test(s) ? s.replace(En, "$1_=" + yn++) : s + (bn.test(s) ? "&" : "?") + "_=" + yn++)
            }
            if (c.ifModified) {
                w.lastModified[s] && x.setRequestHeader("If-Modified-Since", w.lastModified[s]);
                w.etag[s] && x.setRequestHeader("If-None-Match", w.etag[s])
            }(c.data && c.hasContent && c.contentType!==!1 || n.contentType) && x.setRequestHeader("Content-Type", c.contentType);
            x.setRequestHeader("Accept", c.dataTypes[0] && c.accepts[c.dataTypes[0]] ? c.accepts[c.dataTypes[0]] + (c.dataTypes[0] !== "*" ? ", " + On + "; q=0.01" : "") : c.accepts["*"]);
            for (i in c.headers)
                x.setRequestHeader(i, c.headers[i]);
            if (!c.beforeSend || c.beforeSend.call(h, x, c)!==!1 && b !== 2) {
                E = "abort";
                for (i in{
                    success: 1,
                    error: 1,
                    complete: 1
                })
                    x[i](c[i]);
                f = Dn(An, c, n, x);
                if (!f)
                    N( - 1, "No Transport");
                else {
                    x.readyState = 1;
                    a && p.trigger("ajaxSend", [x, c]);
                    c.async && c.timeout > 0 && (u = setTimeout(function() {
                        x.abort("timeout")
                    }, c.timeout));
                    try {
                        b = 1;
                        f.send(g, N)
                    } catch (T) {
                        if (!(b < 2))
                            throw T;
                        N( - 1, T)
                    }
                }
                return x
            }
            return x.abort()
        },
        getJSON: function(e, t, n) {
            return w.get(e, t, n, "json")
        },
        getScript: function(e, n) {
            return w.get(e, t, n, "script")
        }
    });
    w.each(["get", "post"], function(e, n) {
        w[n] = function(e, r, i, s) {
            if (w.isFunction(r)) {
                s = s || i;
                i = r;
                r = t
            }
            return w.ajax({
                url: e,
                type: n,
                dataType: s,
                data: r,
                success: i
            })
        }
    });
    w.ajaxSetup({
        accepts: {
            script: "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript"
        },
        contents: {
            script: /(?:java|ecma)script/
        },
        converters: {
            "text script": function(e) {
                w.globalEval(e);
                return e
            }
        }
    });
    w.ajaxPrefilter("script", function(e) {
        e.cache === t && (e.cache=!1);
        if (e.crossDomain) {
            e.type = "GET";
            e.global=!1
        }
    });
    w.ajaxTransport("script", function(e) {
        if (e.crossDomain) {
            var n, r = o.head || w("head")[0] || o.documentElement;
            return {
                send: function(t, i) {
                    n = o.createElement("script");
                    n.async=!0;
                    e.scriptCharset && (n.charset = e.scriptCharset);
                    n.src = e.url;
                    n.onload = n.onreadystatechange = function(e, t) {
                        if (t ||!n.readyState || /loaded|complete/.test(n.readyState)) {
                            n.onload = n.onreadystatechange = null;
                            n.parentNode && n.parentNode.removeChild(n);
                            n = null;
                            t || i(200, "success")
                        }
                    };
                    r.insertBefore(n, r.firstChild)
                },
                abort: function() {
                    n && n.onload(t, !0)
                }
            }
        }
    });
    var jn = [], Fn = /(=)\?(?=&|$)|\?\?/;
    w.ajaxSetup({
        jsonp: "callback",
        jsonpCallback: function() {
            var e = jn.pop() || w.expando + "_" + yn++;
            this[e]=!0;
            return e
        }
    });
    w.ajaxPrefilter("json jsonp", function(n, r, i) {
        var s, o, u, a = n.jsonp!==!1 && (Fn.test(n.url) ? "url" : typeof n.data == "string"&&!(n.contentType || "").indexOf("application/x-www-form-urlencoded") && Fn.test(n.data) && "data");
        if (a || n.dataTypes[0] === "jsonp") {
            s = n.jsonpCallback = w.isFunction(n.jsonpCallback) ? n.jsonpCallback() : n.jsonpCallback;
            a ? n[a] = n[a].replace(Fn, "$1" + s) : n.jsonp!==!1 && (n.url += (bn.test(n.url) ? "&" : "?") + n.jsonp + "=" + s);
            n.converters["script json"] = function() {
                u || w.error(s + " was not called");
                return u[0]
            };
            n.dataTypes[0] = "json";
            o = e[s];
            e[s] = function() {
                u = arguments
            };
            i.always(function() {
                e[s] = o;
                if (n[s]) {
                    n.jsonpCallback = r.jsonpCallback;
                    jn.push(s)
                }
                u && w.isFunction(o) && o(u[0]);
                u = o = t
            });
            return "script"
        }
    });
    var In, qn, Rn = 0, Un = e.ActiveXObject && function() {
        var e;
        for (e in In)
            In[e](t, !0)
    };
    w.ajaxSettings.xhr = e.ActiveXObject ? function() {
        return !this.isLocal && zn() || Wn()
    } : zn;
    qn = w.ajaxSettings.xhr();
    w.support.cors=!!qn && "withCredentials"in qn;
    qn = w.support.ajax=!!qn;
    qn && w.ajaxTransport(function(n) {
        if (!n.crossDomain || w.support.cors) {
            var r;
            return {
                send: function(i, s) {
                    var o, u, a = n.xhr();
                    n.username ? a.open(n.type, n.url, n.async, n.username, n.password) : a.open(n.type, n.url, n.async);
                    if (n.xhrFields)
                        for (u in n.xhrFields)
                            a[u] = n.xhrFields[u];
                    n.mimeType && a.overrideMimeType && a.overrideMimeType(n.mimeType);
                    !n.crossDomain&&!i["X-Requested-With"] && (i["X-Requested-With"] = "XMLHttpRequest");
                    try {
                        for (u in i)
                            a.setRequestHeader(u, i[u])
                    } catch (f) {}
                    a.send(n.hasContent && n.data || null);
                    r = function(e, i) {
                        var u, f, l, c;
                        try {
                            if (r && (i || a.readyState === 4)) {
                                r = t;
                                if (o) {
                                    a.onreadystatechange = w.noop;
                                    Un && delete In[o]
                                }
                                if (i)
                                    a.readyState !== 4 && a.abort();
                                else {
                                    c = {};
                                    u = a.status;
                                    f = a.getAllResponseHeaders();
                                    typeof a.responseText == "string" && (c.text = a.responseText);
                                    try {
                                        l = a.statusText
                                    } catch (h) {
                                        l = ""
                                    }
                                    !u && n.isLocal&&!n.crossDomain ? u = c.text ? 200 : 404 : u === 1223 && (u = 204)
                                }
                            }
                        } catch (p) {
                            i || s( - 1, p)
                        }
                        c && s(u, l, c, f)
                    };
                    if (!n.async)
                        r();
                    else if (a.readyState === 4)
                        setTimeout(r);
                    else {
                        o=++Rn;
                        if (Un) {
                            if (!In) {
                                In = {};
                                w(e).unload(Un)
                            }
                            In[o] = r
                        }
                        a.onreadystatechange = r
                    }
                },
                abort: function() {
                    r && r(t, !0)
                }
            }
        }
    });
    var Xn, Vn, $n = /^(?:toggle|show|hide)$/, Jn = new RegExp("^(?:([+-])=|)(" + E + ")([a-z%]*)$", "i"), Kn = /queueHooks$/, Qn = [nr], Gn = {
        "*": [function(e, t) {
            var n = this.createTween(e, t), r = n.cur(), i = Jn.exec(t), s = i && i[3] || (w.cssNumber[e] ? "" : "px"), o = (w.cssNumber[e] || s !== "px"&&+r) && Jn.exec(w.css(n.elem, e)), u = 1, a = 20;
            if (o && o[3] !== s) {
                s = s || o[3];
                i = i || [];
                o =+ r || 1;
                do {
                    u = u || ".5";
                    o/=u;
                    w.style(n.elem, e, o + s)
                }
                while (u !== (u = n.cur() / r) && u !== 1&&--a)
                }
            if (i) {
                o = n.start =+ o||+r || 0;
                n.unit = s;
                n.end = i[1] ? o + (i[1] + 1) * i[2] : + i[2]
            }
            return n
        }
        ]
    };
    w.Animation = w.extend(er, {
        tweener: function(e, t) {
            if (w.isFunction(e)) {
                t = e;
                e = ["*"]
            } else 
                e = e.split(" ");
            var n, r = 0, i = e.length;
            for (; r < i; r++) {
                n = e[r];
                Gn[n] = Gn[n] || [];
                Gn[n].unshift(t)
            }
        }, prefilter : function(e, t) {
            t ? Qn.unshift(e) : Qn.push(e)
        }
    });
    w.Tween = rr;
    rr.prototype = {
        constructor: rr,
        init: function(e, t, n, r, i, s) {
            this.elem = e;
            this.prop = n;
            this.easing = i || "swing";
            this.options = t;
            this.start = this.now = this.cur();
            this.end = r;
            this.unit = s || (w.cssNumber[n] ? "" : "px")
        },
        cur: function() {
            var e = rr.propHooks[this.prop];
            return e && e.get ? e.get(this) : rr.propHooks._default.get(this)
        },
        run: function(e) {
            var t, n = rr.propHooks[this.prop];
            this.options.duration ? this.pos = t = w.easing[this.easing](e, this.options.duration * e, 0, 1, this.options.duration) : this.pos = t = e;
            this.now = (this.end - this.start) * t + this.start;
            this.options.step && this.options.step.call(this.elem, this.now, this);
            n && n.set ? n.set(this) : rr.propHooks._default.set(this);
            return this
        }
    };
    rr.prototype.init.prototype = rr.prototype;
    rr.propHooks = {
        _default: {
            get: function(e) {
                var t;
                if (e.elem[e.prop] == null||!!e.elem.style && e.elem.style[e.prop] != null) {
                    t = w.css(e.elem, e.prop, "");
                    return !t || t === "auto" ? 0 : t
                }
                return e.elem[e.prop]
            },
            set: function(e) {
                w.fx.step[e.prop] ? w.fx.step[e.prop](e) : e.elem.style && (e.elem.style[w.cssProps[e.prop]] != null || w.cssHooks[e.prop]) ? w.style(e.elem, e.prop, e.now + e.unit) : e.elem[e.prop] = e.now
            }
        }
    };
    rr.propHooks.scrollTop = rr.propHooks.scrollLeft = {
        set: function(e) {
            e.elem.nodeType && e.elem.parentNode && (e.elem[e.prop] = e.now)
        }
    };
    w.each(["toggle", "show", "hide"], function(e, t) {
        var n = w.fn[t];
        w.fn[t] = function(e, r, i) {
            return e == null || typeof e == "boolean" ? n.apply(this, arguments) : this.animate(ir(t, !0), e, r, i)
        }
    });
    w.fn.extend({
        fadeTo: function(e, t, n, r) {
            return this.filter(nn).css("opacity", 0).show().end().animate({
                opacity: t
            }, e, n, r)
        },
        animate: function(e, t, n, r) {
            var i = w.isEmptyObject(e), s = w.speed(t, n, r), o = function() {
                var t = er(this, w.extend({}, e), s);
                (i || w._data(this, "finish")) && t.stop(!0)
            };
            o.finish = o;
            return i || s.queue===!1 ? this.each(o) : this.queue(s.queue, o)
        },
        stop: function(e, n, r) {
            var i = function(e) {
                var t = e.stop;
                delete e.stop;
                t(r)
            };
            if (typeof e != "string") {
                r = n;
                n = e;
                e = t
            }
            n && e!==!1 && this.queue(e || "fx", []);
            return this.each(function() {
                var t=!0, n = e != null && e + "queueHooks", s = w.timers, o = w._data(this);
                if (n)
                    o[n] && o[n].stop && i(o[n]);
                else 
                    for (n in o)
                        o[n] && o[n].stop && Kn.test(n) && i(o[n]);
                for (n = s.length; n--;)
                    if (s[n].elem === this && (e == null || s[n].queue === e)) {
                        s[n].anim.stop(r);
                        t=!1;
                        s.splice(n, 1)
                    }(t ||!r) && w.dequeue(this, e)
            })
        },
        finish: function(e) {
            e!==!1 && (e = e || "fx");
            return this.each(function() {
                var t, n = w._data(this), r = n[e + "queue"], i = n[e + "queueHooks"], s = w.timers, o = r ? r.length: 0;
                n.finish=!0;
                w.queue(this, e, []);
                i && i.stop && i.stop.call(this, !0);
                for (t = s.length; t--;)
                    if (s[t].elem === this && s[t].queue === e) {
                        s[t].anim.stop(!0);
                        s.splice(t, 1)
                    }
                for (t = 0; t < o; t++)
                    r[t] && r[t].finish && r[t].finish.call(this);
                delete n.finish
            })
        }
    });
    w.each({
        slideDown: ir("show"),
        slideUp: ir("hide"),
        slideToggle: ir("toggle"),
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
        w.fn[e] = function(e, n, r) {
            return this.animate(t, e, n, r)
        }
    });
    w.speed = function(e, t, n) {
        var r = e && typeof e == "object" ? w.extend({}, e): {
            complete: n ||!n && t || w.isFunction(e) && e,
            duration: e,
            easing: n && t || t&&!w.isFunction(t) && t
        };
        r.duration = w.fx.off ? 0 : typeof r.duration == "number" ? r.duration : r.duration in w.fx.speeds ? w.fx.speeds[r.duration] : w.fx.speeds._default;
        if (r.queue == null || r.queue===!0)
            r.queue = "fx";
        r.old = r.complete;
        r.complete = function() {
            w.isFunction(r.old) && r.old.call(this);
            r.queue && w.dequeue(this, r.queue)
        };
        return r
    };
    w.easing = {
        linear: function(e) {
            return e
        },
        swing: function(e) {
            return .5 - Math.cos(e * Math.PI) / 2
        }
    };
    w.timers = [];
    w.fx = rr.prototype.init;
    w.fx.tick = function() {
        var e, n = w.timers, r = 0;
        Xn = w.now();
        for (; r < n.length; r++) {
            e = n[r];
            !e() && n[r] === e && n.splice(r--, 1)
        }
        n.length || w.fx.stop();
        Xn = t
    };
    w.fx.timer = function(e) {
        e() && w.timers.push(e) && w.fx.start()
    };
    w.fx.interval = 13;
    w.fx.start = function() {
        Vn || (Vn = setInterval(w.fx.tick, w.fx.interval))
    };
    w.fx.stop = function() {
        clearInterval(Vn);
        Vn = null
    };
    w.fx.speeds = {
        slow: 600,
        fast: 200,
        _default: 400
    };
    w.fx.step = {};
    w.expr && w.expr.filters && (w.expr.filters.animated = function(e) {
        return w.grep(w.timers, function(t) {
            return e === t.elem
        }).length
    });
    w.fn.offset = function(e) {
        if (arguments.length)
            return e === t ? this : this.each(function(t) {
                w.offset.setOffset(this, e, t)
            });
        var n, r, s = {
            top: 0,
            left: 0
        }, o = this[0], u = o && o.ownerDocument;
        if (!u)
            return;
        n = u.documentElement;
        if (!w.contains(n, o))
            return s;
        typeof o.getBoundingClientRect !== i && (s = o.getBoundingClientRect());
        r = sr(u);
        return {
            top: s.top + (r.pageYOffset || n.scrollTop) - (n.clientTop || 0),
            left: s.left + (r.pageXOffset || n.scrollLeft) - (n.clientLeft || 0)
        }
    };
    w.offset = {
        setOffset: function(e, t, n) {
            var r = w.css(e, "position");
            r === "static" && (e.style.position = "relative");
            var i = w(e), s = i.offset(), o = w.css(e, "top"), u = w.css(e, "left"), a = (r === "absolute" || r === "fixed") && w.inArray("auto", [o, u])>-1, f = {}, l = {}, c, h;
            if (a) {
                l = i.position();
                c = l.top;
                h = l.left
            } else {
                c = parseFloat(o) || 0;
                h = parseFloat(u) || 0
            }
            w.isFunction(t) && (t = t.call(e, n, s));
            t.top != null && (f.top = t.top - s.top + c);
            t.left != null && (f.left = t.left - s.left + h);
            "using"in t ? t.using.call(e, f) : i.css(f)
        }
    };
    w.fn.extend({
        position: function() {
            if (!this[0])
                return;
            var e, t, n = {
                top: 0,
                left: 0
            }, r = this[0];
            if (w.css(r, "position") === "fixed")
                t = r.getBoundingClientRect();
            else {
                e = this.offsetParent();
                t = this.offset();
                w.nodeName(e[0], "html") || (n = e.offset());
                n.top += w.css(e[0], "borderTopWidth", !0);
                n.left += w.css(e[0], "borderLeftWidth", !0)
            }
            return {
                top: t.top - n.top - w.css(r, "marginTop", !0),
                left: t.left - n.left - w.css(r, "marginLeft", !0)
            }
        },
        offsetParent: function() {
            return this.map(function() {
                var e = this.offsetParent || u;
                while (e&&!w.nodeName(e, "html") && w.css(e, "position") === "static")
                    e = e.offsetParent;
                return e || u
            })
        }
    });
    w.each({
        scrollLeft: "pageXOffset",
        scrollTop: "pageYOffset"
    }, function(e, n) {
        var r = /Y/.test(n);
        w.fn[e] = function(i) {
            return w.access(this, function(e, i, s) {
                var o = sr(e);
                if (s === t)
                    return o ? n in o ? o[n] : o.document.documentElement[i] : e[i];
                o ? o.scrollTo(r ? w(o).scrollLeft() : s, r ? s : w(o).scrollTop()) : e[i] = s
            }, e, i, arguments.length, null)
        }
    });
    w.each({
        Height: "height",
        Width: "width"
    }, function(e, n) {
        w.each({
            padding: "inner" + e,
            content: n,
            "": "outer" + e
        }, function(r, i) {
            w.fn[i] = function(i, s) {
                var o = arguments.length && (r || typeof i != "boolean"), u = r || (i===!0 || s===!0 ? "margin" : "border");
                return w.access(this, function(n, r, i) {
                    var s;
                    if (w.isWindow(n))
                        return n.document.documentElement["client" + e];
                    if (n.nodeType === 9) {
                        s = n.documentElement;
                        return Math.max(n.body["scroll" + e], s["scroll" + e], n.body["offset" + e], s["offset" + e], s["client" + e])
                    }
                    return i === t ? w.css(n, r, u) : w.style(n, r, i, u)
                }, n, o ? i : t, o, null)
            }
        })
    });
    w.fn.size = function() {
        return this.length
    };
    w.fn.andSelf = w.fn.addBack;
    if (typeof module == "object" && module && typeof module.exports == "object")
        module.exports = w;
    else {
        e.jQuery = e.$ = w;
        typeof define == "function" && define.amd && define("jquery", [], function() {
            return w
        })
    }
})(window);
(function(e, t) {
    "$:nomunge";
    var n = e.jQuery || e.Cowboy || (e.Cowboy = {}), r;
    n.getObject = r = function(n, r, i) {
        typeof n == "string" && (n = n.split("."));
        if (typeof r != "boolean") {
            i = r;
            r = t
        }
        i = i || e;
        var s;
        while (i && n.length) {
            s = n.shift();
            i[s] === t && r && (i[s] = {});
            i = i[s]
        }
        return i
    };
    n.setObject = function(e, n, i) {
        var s = e.split("."), o = s.pop(), u = r(s, !0, i);
        return u && typeof u == "object" && o ? u[o] = n : t
    };
    n.exists = function(e, n) {
        return r(e, n) !== t
    }
})(window);
jQuery.cookie = function(e, t, n) {
    if (arguments.length > 1 && (t === null || typeof t != "object")) {
        n = jQuery.extend({}, n);
        t === null && (n.expires =- 1);
        if (typeof n.expires == "number") {
            var r = n.expires, i = n.expires = new Date;
            i.setDate(i.getDate() + r)
        }
        return document.cookie = [encodeURIComponent(e), "=", n.raw ? String(t): encodeURIComponent(String(t)), n.expires ? "; expires=" + n.expires.toUTCString(): "", n.path ? "; path=" + n.path: "", n.domain ? "; domain=" + n.domain: "", n.secure ? "; secure": ""].join("")
    }
    n = t || {};
    var s, o = n.raw ? function(e) {
        return e
    }
    : decodeURIComponent;
    return (s = (new RegExp("(?:^|; )" + encodeURIComponent(e) + "=([^;]*)")).exec(document.cookie)) ? o(s[1]) : null
};
(function(e) {
    e.fn.hoverIntent = function(t, n, r) {
        var i = {
            interval: 100,
            sensitivity: 7,
            timeout: 0
        };
        typeof t == "object" ? i = e.extend(i, t) : e.isFunction(n) ? i = e.extend(i, {
            over: t,
            out: n,
            selector: r
        }) : i = e.extend(i, {
            over: t,
            out: t,
            selector: n
        });
        var s, o, u, a, f = function(e) {
            s = e.pageX;
            o = e.pageY
        }, l = function(t, n) {
            n.hoverIntent_t = clearTimeout(n.hoverIntent_t);
            if (Math.abs(u - s) + Math.abs(a - o) < i.sensitivity) {
                e(n).off("mousemove.hoverIntent", f);
                n.hoverIntent_s = 1;
                return i.over.apply(n, [t])
            }
            u = s;
            a = o;
            n.hoverIntent_t = setTimeout(function() {
                l(t, n)
            }, i.interval)
        }, c = function(e, t) {
            t.hoverIntent_t = clearTimeout(t.hoverIntent_t);
            t.hoverIntent_s = 0;
            return i.out.apply(t, [e])
        }, h = function(t) {
            var n = jQuery.extend({}, t), r = this;
            r.hoverIntent_t && (r.hoverIntent_t = clearTimeout(r.hoverIntent_t));
            if (t.type == "mouseenter") {
                u = n.pageX;
                a = n.pageY;
                e(r).on("mousemove.hoverIntent", f);
                r.hoverIntent_s != 1 && (r.hoverIntent_t = setTimeout(function() {
                    l(n, r)
                }, i.interval))
            } else {
                e(r).off("mousemove.hoverIntent", f);
                r.hoverIntent_s == 1 && (r.hoverIntent_t = setTimeout(function() {
                    c(n, r)
                }, i.timeout))
            }
        };
        return this.on({
            "mouseenter.hoverIntent": h,
            "mouseleave.hoverIntent": h
        }, i.selector)
    }
})(jQuery);
(function(e, t) {
    "$:nomunge";
    var n = e.jQuery || e.Cowboy || (e.Cowboy = {}), r;
    n.throttle = r = function(e, r, i, s) {
        function a() {
            function l() {
                u =+ (new Date);
                i.apply(n, f)
            }
            function c() {
                o = t
            }
            var n = this, a =+ (new Date) - u, f = arguments;
            s&&!o && l();
            o && clearTimeout(o);
            s === t && a > e ? l() : r!==!0 && (o = setTimeout(s ? c : l, s === t ? e - a : e))
        }
        var o, u = 0;
        if (typeof r != "boolean") {
            s = i;
            i = r;
            r = t
        }
        n.guid && (a.guid = i.guid = i.guid || n.guid++);
        return a
    };
    n.debounce = function(e, n, i) {
        return i === t ? r(e, n, !1) : r(e, i, n!==!1)
    }
})(window);
(function(e, t, n) {
    function f(e) {
        var t = {}, r = /^jQuery\d+$/;
        n.each(e.attributes, function(e, n) {
            n.specified&&!r.test(n.name) && (t[n.name] = n.value)
        });
        return t
    }
    function l(e, r) {
        var i = this, s = n(i);
        if (i.value == s.attr("placeholder") && s.hasClass("placeholder"))
            if (s.data("placeholder-password")) {
                s = s.hide().next().show().attr("id", s.removeAttr("id").data("placeholder-id"));
                if (e===!0)
                    return s[0].value = r;
                    s.focus()
            } else {
                i.value = "";
                s.removeClass("placeholder");
                i == t.activeElement && i.select()
            }
    }
    function c() {
        var e, t = this, r = n(t), i = r, s = this.id;
        if (t.value == "") {
            if (t.type == "password") {
                if (!r.data("placeholder-textinput")) {
                    try {
                        e = r.clone().attr({
                            type: "text"
                        })
                    } catch (o) {
                        e = n("<input>").attr(n.extend(f(this), {
                            type: "text"
                        }))
                    }
                    e.removeAttr("name").data({
                        "placeholder-password": !0,
                        "placeholder-id": s
                    }).bind("focus.placeholder", l);
                    r.data({
                        "placeholder-textinput": e,
                        "placeholder-id": s
                    }).before(e)
                }
                r = r.removeAttr("id").hide().prev().attr("id", s).show()
            }
            r.addClass("placeholder");
            r[0].value = r.attr("placeholder")
        } else 
            r.removeClass("placeholder")
    }
    var r = "placeholder"in t.createElement("input"), i = "placeholder"in t.createElement("textarea"), s = n.fn, o = n.valHooks, u, a;
    if (r && i) {
        a = s.placeholder = function() {
            return this
        };
        a.input = a.textarea=!0
    } else {
        a = s.placeholder = function() {
            var e = this;
            e.filter((r ? "textarea" : ":input") + "[placeholder]").not(".placeholder").bind({
                "focus.placeholder": l,
                "blur.placeholder": c
            }).data("placeholder-enabled", !0).trigger("blur.placeholder");
            return e
        };
        a.input = r;
        a.textarea = i;
        u = {
            get: function(e) {
                var t = n(e);
                return t.data("placeholder-enabled") && t.hasClass("placeholder") ? "" : e.value
            },
            set: function(e, r) {
                var i = n(e);
                if (!i.data("placeholder-enabled"))
                    return e.value = r;
                if (r == "") {
                    e.value = r;
                    e != t.activeElement && c.call(e)
                } else 
                    i.hasClass("placeholder") ? l.call(e, !0, r) || (e.value = r) : e.value = r;
                return i
            }
        };
        r || (o.input = u);
        i || (o.textarea = u);
        n(function() {
            n(t).delegate("form", "submit.placeholder", function() {
                var e = n(".placeholder", this).each(l);
                setTimeout(function() {
                    e.each(c)
                }, 10)
            })
        });
        n(e).bind("beforeunload.placeholder", function() {
            n(".placeholder").each(function() {
                this.value = ""
            })
        })
    }
})(this, document, jQuery);
$(window).load(function() {
    $("input, textarea").filter(function() {
        return $(this).closest("#register").length == 0
    }).placeholder();
    Modernizr.input.placeholder || $("#register input").each(function() {
        if ($(this).val() == "" && $(this).attr("placeholder") != "") {
            $(this).val($(this).attr("placeholder"));
            $(this).focus(function() {
                if ($(this).val() == $(this).attr("placeholder")) {
                    $(this).val("");
                    $(this).removeClass("placeholder")
                }
            });
            $(this).blur(function() {
                if ($(this).val() == "") {
                    $(this).val($(this).attr("placeholder"));
                    $(this).addClass("placeholder")
                }
            })
        }
    })
});
(function(e) {
    "$:nomunge";
    var t, n, r = 1, i, s = this, o=!1, u = "postMessage", a = "addEventListener", f, l = s[u]&&!e("html").hasClass("is_opera");
    e[u] = function(t, n, i) {
        if (!n)
            return;
        t = typeof t == "string" ? t : e.param(t);
        i = i || parent;
        l ? i[u](t, n.replace(/([^:]+:\/\/[^\/]+).*/, "$1")) : n && (i.location = n.replace(/#.*$/, "") + "#" + + (new Date) + r++ + "&" + t)
    };
    e.receiveMessage = f = function(r, u, c) {
        if (l) {
            if (r) {
                i && f();
                i = function(t) {
                    if (typeof u == "string" && t.origin !== u || e.isFunction(u) && u(t.origin) === o)
                        return o;
                    r(t)
                }
            }
            s[a] ? s[r ? a: "removeEventListener"]("message", i, o) : s[r ? "attachEvent": "detachEvent"]("onmessage", i)
        } else {
            t && clearInterval(t);
            t = null;
            if (r) {
                c = typeof u == "number" ? u : typeof c == "number" ? c : 100;
                t = setInterval(function() {
                    var e = document.location.hash, t = /^#?\d+&/;
                    if (e !== n && t.test(e)) {
                        n = e;
                        r({
                            data: e.replace(t, "")
                        })
                    }
                }, c)
            }
        }
    }
})(jQuery);
window.Modernizr = function(e, t, n) {
    function A(e) {
        f.cssText = e
    }
    function O(e, t) {
        return A(p.join(e + ";") + (t || ""))
    }
    function M(e, t) {
        return typeof e === t
    }
    function _(e, t) {
        return !!~("" + e).indexOf(t)
    }
    function D(e, t) {
        for (var r in e) {
            var i = e[r];
            if (!_(i, "-") && f[i] !== n)
                return t == "pfx" ? i : !0
        }
        return !1
    }
    function P(e, t, r) {
        for (var i in e) {
            var s = t[e[i]];
            if (s !== n)
                return r===!1 ? e[i] : M(s, "function") ? s.bind(r || t) : s
        }
        return !1
    }
    function H(e, t, n) {
        var r = e.charAt(0).toUpperCase() + e.slice(1), i = (e + " " + v.join(r + " ") + r).split(" ");
        if (M(t, "string") || M(t, "undefined"))
            return D(i, t);
        i = (e + " " + m.join(r + " ") + r).split(" ");
        return P(i, t, n)
    }
    function B() {
        i.input = function(n) {
            for (var r = 0, i = n.length; r < i; r++)
                w[n[r]] = n[r]in l;
            w.list && (w.list=!!t.createElement("datalist")&&!!e.HTMLDataListElement);
            return w
        }("autocomplete autofocus list placeholder max min multiple pattern required step".split(" "));
        i.inputtypes = function(e) {
            for (var r = 0, i, s, u, a = e.length; r < a; r++) {
                l.setAttribute("type", s = e[r]);
                i = l.type !== "text";
                if (i) {
                    l.value = c;
                    l.style.cssText = "position:absolute;visibility:hidden;";
                    if (/^range$/.test(s) && l.style.WebkitAppearance !== n) {
                        o.appendChild(l);
                        u = t.defaultView;
                        i = u.getComputedStyle && u.getComputedStyle(l, null).WebkitAppearance !== "textfield" && l.offsetHeight !== 0;
                        o.removeChild(l)
                    } else 
                        /^(search | tel)$ / .test(s) || (/^(url|email)$/.test(s) ? i = l.checkValidity && l.checkValidity()===!1 : i = l.value != c)
                }
                b[e[r]]=!!i
            }
            return b
        }("search tel url email datetime date month week time datetime-local number range color".split(" "))
    }
    var r = "2.6.1", i = {}, s=!0, o = t.documentElement, u = "modernizr", a = t.createElement(u), f = a.style, l = t.createElement("input"), c = ":)", h = {}.toString, p = " -webkit- -moz- -o- -ms- ".split(" "), d = "Webkit Moz O ms", v = d.split(" "), m = d.toLowerCase().split(" "), g = {
        svg: "http://www.w3.org/2000/svg"
    }, y = {}, b = {}, w = {}, E = [], S = E.slice, x, T = function(e, n, r, i) {
        var s, a, f, l = t.createElement("div"), c = t.body, h = c ? c: t.createElement("body");
        if (parseInt(r, 10))
            while (r--) {
                f = t.createElement("div");
                f.id = i ? i[r] : u + (r + 1);
                l.appendChild(f)
            }
        s = ["&#173;", '<style id="s', u, '">', e, "</style>"].join("");
        l.id = u;
        (c ? l : h).innerHTML += s;
        h.appendChild(l);
        if (!c) {
            h.style.background = "";
            o.appendChild(h)
        }
        a = n(l, e);
        c ? l.parentNode.removeChild(l) : h.parentNode.removeChild(h);
        return !!a
    }, N = function(t) {
        var n = e.matchMedia || e.msMatchMedia;
        if (n)
            return n(t).matches;
        var r;
        T("@media " + t + " { #" + u + " { position: absolute; } }", function(t) {
            r = (e.getComputedStyle ? getComputedStyle(t, null) : t.currentStyle)["position"] == "absolute"
        });
        return r
    }, C = function() {
        function r(r, i) {
            i = i || t.createElement(e[r] || "div");
            r = "on" + r;
            var s = r in i;
            if (!s) {
                i.setAttribute || (i = t.createElement("div"));
                if (i.setAttribute && i.removeAttribute) {
                    i.setAttribute(r, "");
                    s = M(i[r], "function");
                    M(i[r], "undefined") || (i[r] = n);
                    i.removeAttribute(r)
                }
            }
            i = null;
            return s
        }
        var e = {
            select: "input",
            change: "input",
            submit: "form",
            reset: "form",
            error: "img",
            load: "img",
            abort: "img"
        };
        return r
    }(), k = {}.hasOwnProperty, L;
    !M(k, "undefined")&&!M(k.call, "undefined") ? L = function(e, t) {
        return k.call(e, t)
    } : L = function(e, t) {
        return t in e && M(e.constructor.prototype[t], "undefined")
    };
    Function.prototype.bind || (Function.prototype.bind = function(t) {
        var n = this;
        if (typeof n != "function")
            throw new TypeError;
        var r = S.call(arguments, 1), i = function() {
            if (this instanceof i) {
                var e = function() {};
                e.prototype = n.prototype;
                var s = new e, o = n.apply(s, r.concat(S.call(arguments)));
                return Object(o) === o ? o : s
            }
            return n.apply(t, r.concat(S.call(arguments)))
        };
        return i
    });
    y.flexbox = function() {
        return H("flexWrap")
    };
    y.flexboxlegacy = function() {
        return H("boxDirection")
    };
    y.canvas = function() {
        var e = t.createElement("canvas");
        return !!e.getContext&&!!e.getContext("2d")
    };
    y.canvastext = function() {
        return !!i.canvas&&!!M(t.createElement("canvas").getContext("2d").fillText, "function")
    };
    y.webgl = function() {
        return !!e.WebGLRenderingContext
    };
    y.touch = function() {
        var n;
        "ontouchstart"in e || e.DocumentTouch && t instanceof DocumentTouch ? n=!0 : T(["@media (", p.join("touch-enabled),("), u, ")", "{#modernizr{top:9px;position:absolute}}"].join(""), function(e) {
            n = e.offsetTop === 9
        });
        return n
    };
    y.geolocation = function() {
        return "geolocation"in navigator
    };
    y.postmessage = function() {
        return !!e.postMessage
    };
    y.websqldatabase = function() {
        return !!e.openDatabase
    };
    y.indexedDB = function() {
        return !!H("indexedDB", e)
    };
    y.hashchange = function() {
        return C("hashchange", e) && (t.documentMode === n || t.documentMode > 7)
    };
    y.history = function() {
        return !!e.history&&!!history.pushState
    };
    y.draganddrop = function() {
        var e = t.createElement("div");
        return "draggable"in e || "ondragstart"in e && "ondrop"in e
    };
    y.websockets = function() {
        return "WebSocket"in e || "MozWebSocket"in e
    };
    y.rgba = function() {
        A("background-color:rgba(150,255,150,.5)");
        return _(f.backgroundColor, "rgba")
    };
    y.hsla = function() {
        A("background-color:hsla(120,40%,100%,.5)");
        return _(f.backgroundColor, "rgba") || _(f.backgroundColor, "hsla")
    };
    y.multiplebgs = function() {
        A("background:url(https://),url(https://),red url(https://)");
        return /(url\s*\(.*?){3}/.test(f.background)
    };
    y.backgroundsize = function() {
        return H("backgroundSize")
    };
    y.borderimage = function() {
        return H("borderImage")
    };
    y.borderradius = function() {
        return H("borderRadius")
    };
    y.boxshadow = function() {
        return H("boxShadow")
    };
    y.textshadow = function() {
        return t.createElement("div").style.textShadow === ""
    };
    y.opacity = function() {
        O("opacity:.55");
        return /^0.55$/.test(f.opacity)
    };
    y.cssanimations = function() {
        return H("animationName")
    };
    y.csscolumns = function() {
        return H("columnCount")
    };
    y.cssgradients = function() {
        var e = "background-image:", t = "gradient(linear,left top,right bottom,from(#9f9),to(white));", n = "linear-gradient(left top,#9f9, white);";
        A((e + "-webkit- ".split(" ").join(t + e) + p.join(n + e)).slice(0, - e.length));
        return _(f.backgroundImage, "gradient")
    };
    y.cssreflections = function() {
        return H("boxReflect")
    };
    y.csstransforms = function() {
        return !!H("transform")
    };
    y.csstransforms3d = function() {
        var e=!!H("perspective");
        e && "webkitPerspective"in o.style && T("@media (transform-3d),(-webkit-transform-3d){#modernizr{left:9px;position:absolute;height:3px;}}", function(t, n) {
            e = t.offsetLeft === 9 && t.offsetHeight === 3
        });
        return e
    };
    y.csstransitions = function() {
        return H("transition")
    };
    y.fontface = function() {
        var e;
        T('@font-face {font-family:"font";src:url("https://")}', function(n, r) {
            var i = t.getElementById("smodernizr"), s = i.sheet || i.styleSheet, o = s ? s.cssRules && s.cssRules[0] ? s.cssRules[0].cssText: s.cssText || "": "";
            e = /src/i.test(o) && o.indexOf(r.split(" ")[0]) === 0
        });
        return e
    };
    y.generatedcontent = function() {
        var e;
        T(['#modernizr:after{content:"', c, '";visibility:hidden}'].join(""), function(t) {
            e = t.offsetHeight >= 1
        });
        return e
    };
    y.video = function() {
        var e = t.createElement("video"), n=!1;
        try {
            if (n=!!e.canPlayType) {
                n = new Boolean(n);
                n.ogg = e.canPlayType('video/ogg; codecs="theora"').replace(/^no$/, "");
                n.h264 = e.canPlayType('video/mp4; codecs="avc1.42E01E"').replace(/^no$/, "");
                n.webm = e.canPlayType('video/webm; codecs="vp8, vorbis"').replace(/^no$/, "")
            }
        } catch (r) {}
        return n
    };
    y.audio = function() {
        var e = t.createElement("audio"), n=!1;
        try {
            if (n=!!e.canPlayType) {
                n = new Boolean(n);
                n.ogg = e.canPlayType('audio/ogg; codecs="vorbis"').replace(/^no$/, "");
                n.mp3 = e.canPlayType("audio/mpeg;").replace(/^no$/, "");
                n.wav = e.canPlayType('audio/wav; codecs="1"').replace(/^no$/, "");
                n.m4a = (e.canPlayType("audio/x-m4a;") || e.canPlayType("audio/aac;")).replace(/^no$/, "")
            }
        } catch (r) {}
        return n
    };
    y.localstorage = function() {
        try {
            localStorage.setItem(u, u);
            localStorage.removeItem(u);
            return !0
        } catch (e) {
            return !1
        }
    };
    y.sessionstorage = function() {
        try {
            sessionStorage.setItem(u, u);
            sessionStorage.removeItem(u);
            return !0
        } catch (e) {
            return !1
        }
    };
    y.webworkers = function() {
        return !!e.Worker
    };
    y.applicationcache = function() {
        return !!e.applicationCache
    };
    y.svg = function() {
        return !!t.createElementNS&&!!t.createElementNS(g.svg, "svg").createSVGRect
    };
    y.inlinesvg = function() {
        var e = t.createElement("div");
        e.innerHTML = "<svg/>";
        return (e.firstChild && e.firstChild.namespaceURI) == g.svg
    };
    y.smil = function() {
        return !!t.createElementNS && /SVGAnimate/.test(h.call(t.createElementNS(g.svg, "animate")))
    };
    y.svgclippaths = function() {
        return !!t.createElementNS && /SVGClipPath/.test(h.call(t.createElementNS(g.svg, "clipPath")))
    };
    for (var j in y)
        if (L(y, j)) {
            x = j.toLowerCase();
            i[x] = y[j]();
            E.push((i[x] ? "" : "no-") + x)
        }
    i.input || B();
    i.addTest = function(e, t) {
        if (typeof e == "object")
            for (var r in e)
                L(e, r) && i.addTest(r, e[r]);
        else {
            e = e.toLowerCase();
            if (i[e] !== n)
                return i;
            t = typeof t == "function" ? t() : t;
            s && (o.className += " " + (t ? "" : "no-") + e);
            i[e] = t
        }
        return i
    };
    A("");
    a = l = null;
    (function(e, t) {
        function l(e, t) {
            var n = e.createElement("p"), r = e.getElementsByTagName("head")[0] || e.documentElement;
            n.innerHTML = "x<style>" + t + "</style>";
            return r.insertBefore(n.lastChild, r.firstChild)
        }
        function c() {
            var e = g.elements;
            return typeof e == "string" ? e.split(" ") : e
        }
        function h(e) {
            var t = a[e[o]];
            if (!t) {
                t = {};
                u++;
                e[o] = u;
                a[u] = t
            }
            return t
        }
        function p(e, n, s) {
            n || (n = t);
            if (f)
                return n.createElement(e);
            s || (s = h(n));
            var o;
            s.cache[e] ? o = s.cache[e].cloneNode() : i.test(e) ? o = (s.cache[e] = s.createElem(e)).cloneNode() : o = s.createElem(e);
            return o.canHaveChildren&&!r.test(e) ? s.frag.appendChild(o) : o
        }
        function d(e, n) {
            e || (e = t);
            if (f)
                return e.createDocumentFragment();
            n = n || h(e);
            var r = n.frag.cloneNode(), i = 0, s = c(), o = s.length;
            for (; i < o; i++)
                r.createElement(s[i]);
            return r
        }
        function v(e, t) {
            if (!t.cache) {
                t.cache = {};
                t.createElem = e.createElement;
                t.createFrag = e.createDocumentFragment;
                t.frag = t.createFrag()
            }
            e.createElement = function(n) {
                return g.shivMethods ? p(n, e, t) : t.createElem(n)
            };
            e.createDocumentFragment = Function("h,f", "return function(){var n=f.cloneNode(),c=n.createElement;h.shivMethods&&(" + c().join().replace(/\w+/g, function(e) {
                t.createElem(e);
                t.frag.createElement(e);
                return 'c("' + e + '")'
            }) + ");return n}")(g, t.frag)
        }
        function m(e) {
            e || (e = t);
            var n = h(e);
            g.shivCSS&&!s&&!n.hasCSS && (n.hasCSS=!!l(e, "article,aside,figcaption,figure,footer,header,hgroup,nav,section{display:block}mark{background:#FF0;color:#000}"));
            f || v(e, n);
            return e
        }
        var n = e.html5 || {}, r = /^<|^(?:button|map|select|textarea|object|iframe|option|optgroup)$/i, i = /^<|^(?:a|b|button|code|div|fieldset|form|h1|h2|h3|h4|h5|h6|i|iframe|img|input|label|li|link|ol|option|p|param|q|script|select|span|strong|style|table|tbody|td|textarea|tfoot|th|thead|tr|ul)$/i, s, o = "_html5shiv", u = 0, a = {}, f;
        (function() {
            try {
                var e = t.createElement("a");
                e.innerHTML = "<xyz></xyz>";
                s = "hidden"in e;
                f = e.childNodes.length == 1 || function() {
                    t.createElement("a");
                    var e = t.createDocumentFragment();
                    return typeof e.cloneNode == "undefined" || typeof e.createDocumentFragment == "undefined" || typeof e.createElement == "undefined"
                }()
            } catch (n) {
                s=!0;
                f=!0
            }
        })();
        var g = {
            elements: n.elements || "abbr article aside audio bdi canvas data datalist details figcaption figure footer header hgroup mark meter nav output progress section summary time video",
            shivCSS: n.shivCSS!==!1,
            supportsUnknownElements: f,
            shivMethods: n.shivMethods!==!1,
            type: "default",
            shivDocument: m,
            createElement: p,
            createDocumentFragment: d
        };
        e.html5 = g;
        m(t)
    })(this, t);
    i._version = r;
    i._prefixes = p;
    i._domPrefixes = m;
    i._cssomPrefixes = v;
    i.mq = N;
    i.hasEvent = C;
    i.testProp = function(e) {
        return D([e])
    };
    i.testAllProps = H;
    i.testStyles = T;
    i.prefixed = function(e, t, n) {
        return t ? H(e, t, n) : H(e, "pfx")
    };
    o.className = o.className.replace(/(^|\s)no-js(\s|$)/, "$1$2") + (s ? " js " + E.join(" ") : "");
    return i
}(this, this.document);
window.JSON || (window.JSON = {});
(function() {
    "use strict";
    function f(e) {
        return e < 10 ? "0" + e : e
    }
    function quote(e) {
        escapable.lastIndex = 0;
        return escapable.test(e) ? '"' + e.replace(escapable, function(e) {
            var t = meta[e];
            return typeof t == "string" ? t : "\\u" + ("0000" + e.charCodeAt(0).toString(16)).slice( - 4)
        }) + '"' : '"' + e + '"'
    }
    function str(e, t) {
        var n, r, i, s, o = gap, u, a = t[e];
        a && typeof a == "object" && typeof a.toJSON == "function" && (a = a.toJSON(e));
        typeof rep == "function" && (a = rep.call(t, e, a));
        switch (typeof a) {
        case"string":
            return quote(a);
        case"number":
            return isFinite(a) ? String(a) : "null";
        case"boolean":
        case"null":
            return String(a);
        case"object":
            if (!a)
                return "null";
            gap += indent;
            u = [];
            if (Object.prototype.toString.apply(a) === "[object Array]") {
                s = a.length;
                for (n = 0; n < s; n += 1)
                    u[n] = str(n, a) || "null";
                i = u.length === 0 ? "[]" : gap ? "[\n" + gap + u.join(",\n" + gap) + "\n" + o + "]" : "[" + u.join(",") + "]";
                gap = o;
                return i
            }
            if (rep && typeof rep == "object") {
                s = rep.length;
                for (n = 0; n < s; n += 1) {
                    r = rep[n];
                    if (typeof r == "string") {
                        i = str(r, a);
                        i && u.push(quote(r) + (gap ? ": " : ":") + i)
                    }
                }
            } else 
                for (r in a)
                    if (Object.hasOwnProperty.call(a, r)) {
                        i = str(r, a);
                        i && u.push(quote(r) + (gap ? ": " : ":") + i)
                    }
            i = u.length === 0 ? "{}" : gap ? "{\n" + gap + u.join(",\n" + gap) + "\n" + o + "}" : "{" + u.join(",") + "}";
            gap = o;
            return i
        }
    }
    if (typeof Date.prototype.toJSON != "function") {
        Date.prototype.toJSON = function(e) {
            return isFinite(this.valueOf()) ? this.getUTCFullYear() + "-" + f(this.getUTCMonth() + 1) + "-" + f(this.getUTCDate()) + "T" + f(this.getUTCHours()) + ":" + f(this.getUTCMinutes()) + ":" + f(this.getUTCSeconds()) + "Z" : null
        };
        String.prototype.toJSON = Number.prototype.toJSON = Boolean.prototype.toJSON = function(e) {
            return this.valueOf()
        }
    }
    var JSON = window.JSON, cx = /[\u0000\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g, escapable = /[\\\"\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0604\u070f\u17b4\u17b5\u200c-\u200f\u2028-\u202f\u2060-\u206f\ufeff\ufff0-\uffff]/g, gap, indent, meta = {
        "\b": "\\b",
        "	": "\\t",
        "\n": "\\n",
        "\f": "\\f",
        "\r": "\\r",
        '"': '\\"',
        "\\": "\\\\"
    }, rep;
    typeof JSON.stringify != "function" && (JSON.stringify = function(e, t, n) {
        var r;
        gap = "";
        indent = "";
        if (typeof n == "number")
            for (r = 0; r < n; r += 1)
                indent += " ";
        else 
            typeof n == "string" && (indent = n);
        rep = t;
        if (!t || typeof t == "function" || typeof t == "object" && typeof t.length == "number")
            return str("", {
                "": e
            });
        throw new Error("JSON.stringify")
    });
    typeof JSON.parse != "function" && (JSON.parse = function(text, reviver) {
        function walk(e, t) {
            var n, r, i = e[t];
            if (i && typeof i == "object")
                for (n in i)
                    if (Object.hasOwnProperty.call(i, n)) {
                        r = walk(i, n);
                        r !== undefined ? i[n] = r : delete i[n]
                    }
            return reviver.call(e, t, i)
        }
        var j;
        text = String(text);
        cx.lastIndex = 0;
        cx.test(text) && (text = text.replace(cx, function(e) {
            return "\\u" + ("0000" + e.charCodeAt(0).toString(16)).slice( - 4)
        }));
        if (/^[\],:{}\s]*$/.test(text.replace(/\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, "@").replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, "]").replace(/(?:^|:|,)(?:\s*\[)+/g, ""))) {
            j = eval("(" + text + ")");
            return typeof reviver == "function" ? walk({
                "": j
            }, "") : j
        }
        throw new SyntaxError("JSON.parse")
    })
})();
(function() {
    var e = typeof window != "undefined" ? window: exports, t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=", n = function() {
        try {
            document.createElement("$")
        } catch (e) {
            return e
        }
    }();
    e.btoa || (e.btoa = function(e) {
        for (var r, i, s = 0, o = t, u = ""; e.charAt(s | 0) || (o = "=", s%1); u += o.charAt(63 & r>>8 - s%1 * 8)) {
            i = e.charCodeAt(s += .75);
            if (i > 255)
                throw n;
            r = r<<8 | i
        }
        return u
    });
    e.atob || (e.atob = function(e) {
        e = e.replace(/=+$/, "");
        if (e.length%4 == 1)
            throw n;
        for (var r = 0, i, s, o = 0, u = ""; s = e.charAt(o++); ~s && (i = r%4 ? i * 64 + s : s, r++%4) ? u += String.fromCharCode(255 & i>>( - 2 * r & 6)) : 0)
            s = t.indexOf(s);
        return u
    })
})();
$.getObject("Pb", !0).Object = Object.create === undefined ? function() {
    function e() {}
    return function(t) {
        e.prototype = t;
        return new e
    }
}() : function(e) {
    return Object.create(e)
};
$.getObject("Pb", !0).subclass = function(e, t, n, r) {
    var i = t.prototype, s = Pb.Object(i);
    e.prototype = s;
    s.constructor = e;
    e.superclass = i;
    t != Object && i.constructor == Object.prototype.constructor && (i.constructor = t);
    n && $.extend(s, n);
    r && $.extend(e, r);
    return e
};
$.getObject("Pb", !0).DataObject = function() {
    function e() {
        this.data = {}
    }
    var t = {
        data: null,
        init: function(e) {
            if (e) {
                var t;
                for (t in e)
                    e.hasOwnProperty(t) && this.add(t, e[t])
            }
        },
        add: function(e, t) {
            this.data[e] = t
        },
        remove: function(e) {
            delete this.data[e]
        },
        get: function(e) {
            return e in this.data ? this.data[e] : null
        },
        has: function(e) {
            return e in this.data
        },
        addFromChildWindow: function(e, t) {
            var n;
            $.isPlainObject(t) && $("html").hasClass("is_ie") ? n = JSON.parse(JSON.stringify(t)) : n = t;
            this.data[e] = n
        }
    };
    e.prototype = t;
    return e
}();
$.getObject("Pb", !0).Data = new Pb.DataObject;
$.getObject("Pb", !0).DataCookie = function() {
    return {
        init: function() {},
        set: function(e, t) {
            Pb.Utilities.cookie(e, JSON.stringify(t))
        },
        remove: function(e) {
            Pb.Utilities.cookie(e, null)
        },
        get: function(e) {
            var t = Pb.Utilities.cookie(e);
            return t ? JSON.parse(t) : null
        },
        has: function(e) {
            var t = Pb.Utilities.cookie(e);
            return e in data
        }
    }
}();
$.getObject("Pb", !0).Ajax = function() {
    function e(e) {
        this.init(e)
    }
    function t(t) {
        var n = new e(t);
        return n.execute()
    }
    e.prototype = {
        messageHandler: null,
        showProgressModal: !1,
        suppressErrorPushdown: !1,
        progressTitle: "",
        defaults: {
            dataType: "json",
            cache: !1,
            type: "post",
            data: {}
        },
        init: function(e) {
            var t, n, r, i, s = this;
            e.messageHandler && (this.messageHandler = function() {
                return e.messageHandler.apply(this, arguments)
            });
            if (e.showProgressModal) {
                this.showProgressModal=!0;
                e.progressTitle && (this.progressTitle = e.progressTitle)
            }
            e.suppressErrorPushdown && (this.suppressErrorPushdown=!0);
            r = this.successHandler;
            e.success ? t = function() {
                if (r.apply(s, arguments)) {
                    var t = arguments;
                    t[0] && t[0].hasOwnProperty("header") && t[0].hasOwnProperty("body") && (t[0] = t[0].body);
                    e.success.apply(this, t)
                }
            } : t = r;
            i = this.errorHandler;
            e.error ? n = function() {
                i.apply(s, arguments) && e.error.apply(this, arguments)
            } : n = i;
            this.isCrossDomain(e.url) && (this.defaults.dataType = "jsonp");
            this.settings = {};
            $.extend(!0, this.settings, this.defaults, e);
            if (this.settings.dataType === "jsonp") {
                this.settings.data.jsonp = 1;
                this.settings.jsonp || (this.settings.jsonp = "jsoncallback")
            } else 
                this.settings.data.json = 1;
            this.settings.success = $.proxy(t, this);
            this.settings.error = $.proxy(n, this)
        },
        execute: function() {
            var e = $("#token").val();
            e && (this.settings.data.hash = $("#token").val());
            this.showProgressModal && this.addProgressModal();
            return $.ajax(this.settings)
        },
        errorHandler: function(e, t, n) {
            if (e.readyState === 4 && t !== "abort") {
                Pb.Component.Element.Message.remove();
                Pb.Component.Element.Message.showDefaultErrorMessage();
                this.showProgressModal && this.hideProgressModal()
            }
            return !0
        },
        successHandler: function(e) {
            e.hash !== null && e.hash && $("#token").val(e.hash);
            var t = {
                debug: {
                    messages: [],
                    stopwatche: []
                }
            }, n = this;
            if (e.hasOwnProperty("header")) {
                t = e.header;
                t.hasOwnProperty("debug") && Pb.Debug && Pb.Debug.add(t.debug);
                e.hasOwnProperty("body") && (e = e.body)
            }
            if (this.showProgressModal) {
                Pb.Modal.ModalProgressBar.updateProgressIndicator("#ajaxProgressModal", 100);
                setTimeout(function() {
                    n.handleMessages(e);
                    n.hideProgressModal()
                }, 200)
            } else 
                this.handleMessages(e);
            return !0
        },
        isCrossDomain: function(e) {
            var t = e.match(/:\/\/(.[^/]+)/);
            return t !== null && t.length > 1 && location.host !== t[1]?!0 : !1
        },
        handleMessages: function(e) {
            var t = e.messages, n = e.form, r=!1, i=!1, s, o, u, a, f = this, l=!1;
            t && $.each(t, function() {
                if (this.memo !== "") {
                    f.messageHandler && (i = f.messageHandler(this));
                    if (!i) {
                        if (n)
                            s = "form[name='" + n + "'] [name='" + this.memo + "'], " + "form[id='" + n + "'] [name='" + this.memo + "']";
                        else {
                            s = "form [name='" + this.memo + "'], form input#" + this.memo;
                            o = "#" + this.memo
                        }
                        u = $(s + ", " + o);
                        a = u.closest(".control-group");
                        if (u.length > 0 && a.length > 0) {
                            a.addClass("error");
                            a.find(".help-inline, .help-block").html(this.message + this.details);
                            r=!0
                        } else 
                            Pb.Component.Element.Message.show(this)
                        }
                } else {
                    f.messageHandler && (i = f.messageHandler(this));
                    if (!i) {
                        Pb.Component.Element.Message.show(this);
                        l=!0
                    }
                }
            });
            r&&!l&&!f.suppressErrorPushdown && Pb.Component.Element.Message.showDefaultErrorMessage()
        },
        addProgressModal: function() {
            var e = $("#ajaxProgressModal");
            if (!e.length) {
                e = $('<div class="modal hide fade no-footer" id="ajaxProgressModal" data-backdrop="static" data-keyboard="true"><div class="modal-header"><a class="close" data-dismiss="modal"><i class="icon-remove"></i></a><h2 class="title">' + this.progressTitle + "</h2></div>" + '<div class="modal-body"></div></div>');
                $(document.body).append(e)
            } else 
                $("#ajaxProgressModal").find(".title").html(this.progressTitle);
            Pb.Modal.ModalProgressBar.addProgressIndicator("#ajaxProgressModal");
            e.modal("show")
        },
        hideProgressModal: function() {
            $("#ajaxProgressModal").modal("hide");
            Pb.Modal.ModalProgressBar.removeProgressIndicator("#ajaxProgressModal")
        }
    };
    return {
        ajax: t
    }
}();
$.getObject("Pb").InitEventQueue = function() {
    function p() {
        throw new Error("Event was added to InitEventQueue after domready or load was called")
    }
    function d(t, n, r) {
        r = r || e;
        var i = t.length - 1;
        while (i >= 0 && t[i][1] > r)
            i--;
        t.splice(i + 1, 0, [n, r])
    }
    function v(e, t) {
        d(o, e, t)
    }
    function m(e, t) {
        d(u, e, t)
    }
    function g(e) {
        return $.inArray(e, a)===-1?!1 : !0
    }
    function y(e, t) {
        var n = 0;
        while (n < t.length)
            for (; n < t.length; n++) {
                var r = (new Date).getTime();
                t[n][0]();
                var i = (new Date).getTime() - r;
                typeof console != "undefined" && i > 20 && console.log(e + " pri " + n + " completed in " + i + "ms")
            }
        t = null
    }
    function b(e, t) {
        if (typeof console != "undefined") {
            var n = e + " (" + window.location + ")";
            window.jsStartTime !== undefined && (n += " " + (t.getTime() - window.jsStartTime.getTime() + "ms"));
            console.log(n)
        }
    }
    var e = 50, t = 1, n = 5, r = 25, i = 40, s = 51, o = [], u = [], a = [], f=!1, l=!1, c, h;
    $(document).ready(function() {
        if (!f) {
            f=!0;
            c = new Date;
            b("DomReady START", c);
            y("DomReady", o);
            o = null;
            Pb.InitEventQueue.addToDomReady = p;
            a.push("DomReady");
            var e = new Date;
            b("DomReady END (" + (e.getTime() - c.getTime()) + "ms)", e)
        }
    });
    $(window).load(function() {
        if (!l) {
            l=!0;
            h = new Date;
            b("PageLoad START", h);
            y("PageLoad", u);
            u = null;
            Pb.InitEventQueue.addToPageLoad = p;
            a.push("PageLoad");
            var e = new Date;
            b("PageLoad END (" + (e.getTime() - h.getTime()) + "ms)", e)
        }
    });
    return {
        addToDomReady: v,
        add: v,
        addToPageLoad: m,
        hasEventOccurred: g,
        DEFAULT_PRIORITY: e,
        TOP_PRIORITY: t,
        HIGH_PRIORITY: n,
        MEDIUM_PRIORITY: r,
        ACTION_EVENT_PRIORITY: i,
        LOW_PRIORITY: s
    }
}();
$.getObject("Pb.Data").Shared = function() {
    function E(e, t, n, s) {
        var o = null, u, a;
        typeof s == "undefined" && (s=!0);
        if ($.isNumeric(n)) {
            var f = T(e);
            if (!$.isArray(f)) {
                f = [];
                if (e === i && Pb.Data.Albums) {
                    u = Pb.Data.Albums.getCurrentAlbum();
                    u && u.addItems(t)
                } else {
                    a = x(e);
                    a ? a.addItems(t) : Pb.Data.add(e, f)
                }
            }
            f[n] = t;
            o = f
        } else {
            if (e === r && Pb.Data.Albums)
                Pb.Data.Albums.addAlbum(t);
            else if (e === i && Pb.Data.Albums) {
                u = Pb.Data.Albums.getCurrentAlbum();
                u && u.addItems(t)
            } else {
                a = x(e);
                a ? a.addItems(t) : Pb.Data.add(e, t)
            }
            o = t
        }
        s && _(e, o);
        return o
    }
    function S(e, t) {
        Pb.Data.add(e + "Collection", t)
    }
    function x(e) {
        return Pb.Data.get(e + "Collection")
    }
    function T(e, t) {
        var n, s;
        if (e === r && Pb.Data.Albums) {
            n = Pb.Data.Albums.getCurrentAlbumData();
            if (!n)
                return null;
            n = n[0]
        } else if (e === i && Pb.Data.Albums)
            n = Pb.Data.Albums.getCurrentMediaData();
        else {
            s = x(e);
            s ? n = s.getLinkableCollectionItemsData() : n = Pb.Data.get(e)
        }
        var o = null;
        $.isNumeric(t) ? o = n[t] : o = n;
        return o
    }
    function N(e) {
        var t = null, n;
        if (e === i && Pb.Data.Albums)
            t = Pb.Data.Albums.getCurrentPageData();
        else {
            n = x(e);
            n && (t = n.getLinkableCollectionPageData())
        }
        return t
    }
    function C(n, r, i) {
        $(n).data(e, r);
        $.isNumeric(i) && $(n).attr("data-" + t, i)
    }
    function k(e) {
        var t = L(e);
        return T(A(t), O(t))
    }
    function L(t) {
        return $(t).closest("[data-" + e + "]")
    }
    function A(t) {
        var n = $(t).data(e);
        return n
    }
    function O(e) {
        var n = $(e).attr("data-" + t);
        return n
    }
    function M(e, t) {
        if (e === null || typeof e == "undefined")
            return null;
        typeof t == "undefined" && (t = T(e));
        t !== null && (t = _(e, t));
        return t
    }
    function _(e, t) {
        $.isArray(t) ? $.each(t, function(t, n) {
            _(e, n)
        }) : $.observable(t).setProperty("pbdkey", e);
        return t
    }
    var e = "pbdkey", t = "pbdoffset", n = "shared.globalAlbumSet", r = "shared.album", i = "shared.albumSet", s = "shared.media", o = "shared.mediaSet", u = "shared.searchSet", a = "shared.recentSet", f = "shared.categorySet", l = "shared.likeSet", c = "shared.activityObjectSet", h = "shared.mediaselector", p = "shared.mediaselect.album", d = "shared.story", v = "shared.storyset", m = "shared.storyset.modal", g = "shared.browse.storyset", y = "shared.subAlbumSet", b = "shared.searchAlbumSet", w = "shared.slideshowSet";
    ALBUM_STORY_SET = "shared.albumStorySet";
    return {
        PBD_ATTR_KEY: e,
        PBD_ATTR_OFFSET: t,
        GLOBAL_ALBUM_SET: n,
        ALBUM: r,
        ALBUM_SET: i,
        MEDIA: s,
        MEDIA_SET: o,
        SEARCH_SET: u,
        RECENT_SET: a,
        CATEGORY_SET: f,
        LIKE_SET: l,
        ACTIVITY_OBJECT_SET: c,
        MEDIA_SELECTOR: h,
        MEDIA_SELECTOR_ALBUM: p,
        STORY: d,
        STORY_SET: v,
        STORY_SET_MODAL: m,
        BROWSE_STORY_SET: g,
        ALBUM_SUBALBUM_SET: y,
        ALBUM_SEARCH_SET: b,
        SLIDESHOW_SET: w,
        ALBUM_STORY_SET: ALBUM_STORY_SET,
        put: E,
        get: T,
        getPaginated: N,
        getData: k,
        sync: M,
        attach: C,
        closest: L,
        key: A,
        offset: O,
        addCollection: S,
        getCollection: x
    }
}();
$.getObject("Pb", !0).Utilities = function() {
    function t(e, t, n) {
        if (arguments.length >= 2) {
            n = $.extend({
                domain: ".photobucket.com",
                expires: 1,
                path: "/"
            }, n);
            return $.cookie(e, t, n)
        }
        return $.cookie(e)
    }
    function n(t) {
        t && (e = null);
        if (!e) {
            e = new QueryParameters;
            var n = window.location.search.substr(1).split("&");
            if (n !== "")
                for (var r = 0; r < n.length; ++r) {
                    var i = n[r].split("=");
                    i.length === 2 && (e[i[0]] = decodeURIComponent(i[1].replace(/\+/g, " ")))
                }
            }
        return e
    }
    var e;
    QueryParameters = function() {};
    QueryParameters.prototype.toQueryString = function() {
        var e = "", t = 0;
        for (var n in this)
            if (typeof this[n] != "function") {
                t === 0 && (e += "?");
                t >= 1 && (e += "&");
                e += n;
                e += "=";
                e += escape(this[n]);
                t++
            }
        return e
    };
    QueryParameters.prototype.clone = function() {
        var e = new QueryParameters;
        for (var t in this)
            this.hasOwnProperty(t) && (e[t] = this[t]);
        return e
    };
    return {
        cookie: t,
        getQueryParameters: n
    }
}();
$.getObject("Pb.Media", !0).MediaIdentifierFactory = function() {
    function e() {}
    var t = {
        getIdentifier: function(e) {
            var t;
            e instanceof Pb.Path.PBPath ? t = this.fromPBPath(e) : e.path ? t = this.fromPBPath(new Pb.Path.PBPath(e.path)) : t = null;
            return t
        },
        fromPBPath: function(e) {
            var t = null;
            if (e.filename != null) {
                var n = "path:";
                n += e.location;
                n += e.filename;
                t = btoa(n)
            }
            return t
        }
    };
    $.extend(e.prototype, t);
    return new e
}();
$.getObject("Pb.Path", !0).PBPath = function() {
    function e(e) {
        this.pathString = decodeURIComponent(e);
        this.parse(this.pathString)
    }
    var t = {
        pathFormat: /^\/?([^\/]+)\/([^\/]+)\/([^\/]+)\/(.*\/)?([^\/]+)$/,
        parse: function(e) {
            var t = this.pathFormat.exec(e);
            if (t == null) {
                this.albumType = null;
                this.vee = null;
                this.ownername = null;
                this.location = null;
                this.filename = null
            } else {
                this.albumType = t[1];
                this.vee = t[2];
                this.ownername = t[3];
                this.location = t[4] === undefined ? "" : t[4];
                this.filename = t[5]
            }
        }
    };
    $.extend(e.prototype, t);
    return e
}();
(function(e, t, n) {
    "use strict";
    function H(e, t) {
        if (t && t.onError && t.onError(e) === p)
            return;
        this.name = "JsRender Error";
        this.message = e || "JsRender error"
    }
    function B(e, t) {
        var n;
        e = e || {};
        for (n in t)
            e[n] = t[n];
        return e
    }
    function j(e, t, n) {
        if (!P.rTag || arguments.length) {
            a = e ? e.charAt(0) : a;
            f = e ? e.charAt(1) : f;
            l = t ? t.charAt(0) : l;
            c = t ? t.charAt(1) : c;
            h = n || h;
            e = "\\" + a + "(\\" + h + ")?\\" + f;
            t = "\\" + l + "\\" + c;
            o = "(?:(?:(\\w+(?=[\\/\\s\\" + l + "]))|(?:(\\w+)?(:)|(>)|!--((?:[^-]|-(?!-))*)--|(\\*)))" + "\\s*((?:[^\\" + l + "]|\\" + l + "(?!\\" + c + "))*?)";
            P.rTag = o + ")";
            o = new RegExp(e + o + "(\\/)?|(?:\\/(\\w+)))" + t, "g");
            u = new RegExp("<.*>|([^\\\\]|^)[{}]|" + e + ".*" + t)
        }
        return [a, f, l, c, h]
    }
    function F(e) {
        var t = this, n=!e || e === "root";
        while (n && t.parent.parent || t && t.type !== e)
            t = t.parent;
        return t
    }
    function I(e) {
        var t, r = this, i = r.ctx || {}, s = r.tmpl.helpers || {};
        e = (i[e] !== n ? i : s[e] !== n ? s : it[e] !== n ? it : {})[e];
        if (e && typeof e == "function") {
            t = function() {
                var t, n = arguments;
                for (t in n)
                    typeof n[t] == "function" && (n[t] = n[t].call(r.data));
                return e.apply(r, n)
            };
            t.depends = e.depends
        }
        return t || e
    }
    function q(e, t, n, r, i, s) {
        if (e || i) {
            var o, u=!n.markup && n, a = {
                tagName: e + ":",
                tagCtx: r
            }, f = r.args = M.call(arguments, 5);
            r.view = t;
            r.bind=!!u||!!i;
            if (u) {
                u.tag = a;
                a.linkCtx = u;
                r.ctx = tt(r.ctx, u.view.ctx)
            }
            a.ctx = r.ctx || {};
            delete r.ctx;
            if (e) {
                o = t.tmpl.converters;
                o = o && o[e] || rt[e];
                if (o || K("Unknown converter: {{" + e + ":"))
                    s = o.apply(a, f)
                }
            if (i) {
                i = t.tmpl.bnds[i - 1];
                u.paths = i;
                t._.tag = a;
                s = t._.onRender(s, t, d)
            }
        }
        return s
    }
    function R(e, t, r, i, s, o) {
        var u, a, f, l, c, h, d = r.markup && r, v=!d && r, m = t._, g = d || t.tmpl, y = g.tags, b = g.templates, w = y && y[e] || st[e], E = s.args = arguments.length > 6 ? M.call(arguments, 6) : [], S = s.props || {};
        s.view = t;
        s.ctx = tt(s.ctx, t.ctx);
        f = s.ctx || {};
        delete s.ctx;
        if (!w) {
            K("Unknown tag: {{" + e + "}}");
            return ""
        }
        d = S.tmpl;
        i = i && g.tmpls[i - 1];
        d = d || i || w.template || n;
        d = "" + d === d ? b && b[d] || nt[d] || nt(d) : d;
        if (e === "else") {
            c = t._.tag;
            l = c._elses = c._elses || [];
            l.push(d);
            s.isElse = l.length;
            a = c.render
        }
        if (w.init) {
            h = f.tags = t.ctx && tt(f.tags, t.ctx.tags) || {};
            c = c || v.tag;
            if (c)
                h[e] = c;
            else {
                c = h[e] = new w.init(s, v, f);
                c.tmpl = d;
                v && (c.attr = v.attr = v.attr || w.attr || "")
            }
            f.tag = c
        } else 
            c = c || {
                render: w.render,
                renderContent: J,
                tmpl: d,
                tagName: e
            };
        c.tagCtx = s;
        c.ctx = f;
        if (v) {
            v.tag = c;
            c.linkCtx = v
        }
        c._is = "tag";
        c._done = s.isElse ? c._done : p;
        d = d || c.tmpl;
        l = c._elses;
        m.tag = c;
        if (a = a || c.render)
            u = a.apply(c, E);
        u = u !== n ? u : d ? c.renderContent(s.data !== n ? s.data : t.data, n, t) : "";
        return o ? m.onRender(u, t, o) : u
    }
    function U(e, t, r, i, s, o, u) {
        var a, f, l = t === "array", c = {
            key: 0,
            useKey: l ? 0: 1,
            id: "" + N++,
            onRender: u,
            bnd: {}
        }, h = {
            data: i,
            tmpl: s,
            views: l ? []: {},
            parent: r,
            ctx: e,
            type: t,
            get: F,
            _hlp: I,
            _: c
        };
        if (r) {
            a = r.views;
            f = r._;
            f.useKey ? a[c.key = "_" + f.useKey++] = h : a.splice(c.key = h.index = o !== n ? o : a.length, 0, h);
            h.ctx = e || r.ctx
        }
        return h
    }
    function z(e) {
        var t, n, r, i, s;
        for (t in D) {
            i = D[t];
            if ((s = i.compile) && (n = e[t + "s"]))
                for (r in n)
                    n[r] = s(r, n[r], e, t, i)
        }
    }
    function W(e, t, n) {
        var r, i;
        if (typeof t == "function")
            t = {
                tagName: e,
                render: t,
                depends: t.depends
            };
        else {
            t.tagName = e;
            if (i = t.template)
                t.template = "" + i === i ? nt[i] || nt(i) : i;
            if (t.init !== p) {
                r = t.init = t.init || function(e) {};
                r.prototype = t;
                (r.prototype = t).constructor = r
            }
        }
        t.renderContent = J;
        t.attr = "html";
        n && (t._parentTmpl = n);
        return t
    }
    function X(e, r, i, s, o, a) {
        function f(n) {
            if ("" + n === n || n.nodeType > 0) {
                try {
                    c = n.nodeType > 0 ? n : !u.test(n) && t && t(n)[0]
                } catch (r) {}
                if (c) {
                    n = nt[c.getAttribute(A)];
                    if (!n) {
                        e = e || "_" + T++;
                        c.setAttribute(A, e);
                        nt[e] = n = X(e, c.innerHTML, i, s, o, a)
                    }
                }
                return n
            }
        }
        var l, c;
        r = r || "";
        l = f(r);
        a = a || (r.markup ? r : {});
        a.tmplName = e;
        i && (a._parentTmpl = i);
        !l && r.markup && (l = f(r.markup)) && l.fn && (l.debug !== r.debug || l.allowCode !== r.allowCode) && (l = l.markup);
        if (l !== n) {
            e&&!i && (_[e] = function() {
                return r.render.apply(r, arguments)
            });
            if (l.fn || r.fn)
                l.fn && (e && e !== l.name ? r = tt(a, l) : r = l);
            else {
                r = V(l, a);
                G(l, r)
            }
            z(a);
            return r
        }
    }
    function V(e, t) {
        var n, r = ut.wrapMap || {}, s = B({
            markup: e,
            tmpls: [],
            links: {},
            bnds: [],
            render: J
        }, t);
        if (!t.htmlTag) {
            n = S.exec(e);
            s.htmlTag = n ? n[1].toLowerCase() : ""
        }
        n = r[s.htmlTag];
        if (n && n !== r.div) {
            s.markup = i.trim(s.markup);
            s._elCnt = d
        }
        return s
    }
    function $(e, t) {
        function r(s, o, u) {
            var a, f, l, c, h, p, d, v;
            if (s && "" + s !== s&&!s.nodeType&&!s.markup) {
                for (c in s)
                    r(c, s[c], o);
                return P
            }
            d = u ? u[i] = u[i] || {} : r;
            if (o === n) {
                o = s;
                s = n
            }
            f = t.compile;
            if (a = ot.onBeforeStoreItem)
                f = a(d, s, o, f) || f;
            if (!s)
                o = f(n, o);
            else if ("" + s === s)
                if (o === null)
                    delete d[s];
                else {
                    d[s] = f ? o = f(s, o, u, e, t) : o;
                    if (e === "template" && (h = d[s].templates))
                        for (v in D) {
                            v += "s";
                            if (l = o[v])
                                for (p in h) {
                                    p = h[p];
                                    p[v] = tt(p[v], l)
                                }
                            }
                    }
            o && (o._is = e);
            (a = ot.onStoreItem) && a(d, s, o, f);
            return o
        }
        var i = e + "s";
        P[i] = r;
        D[e] = t
    }
    function J(e, t, r, s, o, u) {
        var a, f, l, c, h, v, m, g, y, b, w, E, S = this, x = S, T = S.attr === n || S.attr === "html", N = "";
        if (s === d) {
            y = d;
            s = 0
        }
        if (w = S._is === "tag") {
            b = S.tagCtx;
            x = b.isElse ? S._elses[b.isElse - 1] : S.tmpl;
            t = tt(t, S.ctx);
            g = b.props || {};
            if (g.link === p) {
                t = t || {};
                t.link = p
            }
            r = r || b.view
        } else 
            x = S.jquery && (S[0] || K('Unknown template: "' + S.selector + '"')) || S;
        if (x) {
            if (r) {
                u = u || r._.onRender;
                m = r.ctx;
                if (e === r) {
                    e = r.data;
                    o = d
                }
            }
            t = tt(t, m);
            x.fn || (x = nt[x] || nt(x));
            if (x) {
                u = (t && t.link) !== p && T && u;
                E = u;
                if (u === d) {
                    E = n;
                    u = r._.onRender
                }
                if (i.isArray(e)&&!o) {
                    c = y ? r : s !== n && r || U(t, "array", r, e, x, s, u);
                    for (a = 0, f = e.length; a < f; a++) {
                        l = e[a];
                        h = U(t, "item", c, l, x, (s || 0) + a, u);
                        v = x.fn(l, h, i.views);
                        N += c._.onRender ? c._.onRender(v, h) : v
                    }
                } else {
                    c = y ? r : U(t, "data", r, e, x, s, u);
                    N += x.fn(e, c, P)
                }
                return E ? E(N, c) : N
            }
        }
        return ""
    }
    function K(e) {
        if (ut.debugMode)
            throw new P.sub.Error(e)
    }
    function Q(e) {
        K("Syntax error\n" + e)
    }
    function G(e, t, n) {
        function r(t) {
            t -= l;
            t && h.push(e.substr(l, t).replace(g, "\\n"))
        }
        function i(t) {
            t && Q('Unmatched or missing tag: "{{/' + t + '}}" in template:\n' + e)
        }
        function s(t, s, o, f, v, m, g, b, S, x, T, N) {
            if (m) {
                v = ":";
                f = "html"
            }
            var C, k = [], L = "", A = "", O = "", M=!x&&!v&&!g&&!n;
            o = o || v;
            r(N);
            l = N + t.length;
            if (b)
                a && h.push(["*", "\n" + S.replace(y, "$1") + "\n"]);
            else if (o) {
                if (o === "else") {
                    E.test(S) && Q('for "{{else if expr}}" use "{{else expr}}"');
                    p[6] = e.substring(p[6], N);
                    p = c.pop();
                    h = p[3];
                    M = d
                }
                S && (L = Z(S, k).replace(w, function(e, t, n) {
                    t ? O += n + "," : A += n + ",";
                    return ""
                }));
                A = A.slice(0, - 1);
                L = L.slice(0, - 1);
                u = [o, f || "", L, M && [], "{" + (A ? "props:{" + A + "}," : "") + 'params:"' + S + '"' + (O ? ",ctx:{" + O.slice(0, - 1) + "}" : "") + "},", s && k || 0];
                h.push(u);
                if (M) {
                    c.push(p);
                    p = u;
                    p[6] = l
                }
            } else if (T) {
                C = p[0];
                i(T !== C && C && C !== "else");
                p[6] = e.substring(p[6], N);
                p = c.pop()
            }
            i(!p && T);
            h = p[3]
        }
        var u, a = t && t.allowCode, f = [], l = 0, c = [], h = f, p = [, , , f];
        e = e.replace(b, "\\$1");
        i(c[0] && c[0][3].pop()[0]);
        e.replace(o, s);
        r(e.length);
        (l = f[f.length - 1]) && i("" + l !== l&&+l[6] === l[6] && l[0]);
        return Y(f, t)
    }
    function Y(e, t) {
        var n, r, i, s, o, u, a, f, l, c, h, p, v, m, g, y, b, w, E = "", S = {}, x = e.length;
        if (t) {
            if (y = t.allowCode)
                S.allowCode = d;
            t.debug && (S.debug = d);
            p = t.bnds;
            m = t.tmpls
        }
        for (n = 0; n < x; n++) {
            r = e[n];
            E += "\nret+=";
            if ("" + r === r)
                E += '"' + r + '";';
            else {
                f = r[0];
                if (f === "*")
                    E = (n ? E.slice(0, - 5) : "") + r[1];
                else {
                    l = r[1];
                    c = r[2];
                    b = r[3];
                    h = r[4];
                    v = r[5];
                    w = r[6];
                    if (b) {
                        g = V(w, S);
                        Y(b, g);
                        m.push(g)
                    }
                    if (v) {
                        p.push(v);
                        v = p.length
                    }
                    a = a || h.indexOf("view")>-1;
                    E += f === ":" ? l === "html"&&!v ? (s = d, "h(" + c + ");") : l || v ? (u = d, 'c("' + l + '",view,this,' + h + v + "," + c + ");") : (o = d, "(v=" + c + ')!=u?v:"";') : (i = d, 't("' + f + '",view,this,' + (b ? m.length : '""') + "," + h + v + (c ? "," : "") + c) + ");"
                }
            }
        }
        E = O + (o ? "v," : "") + (i ? "t=j._tag," : "") + (u ? "c=j._cnvt," : "") + (s ? "h=j.converters.html," : "") + 'ret=""; try{\n' + (S.debug ? "debugger;" : "") + E + "\nreturn ret;\n\n}catch(e){return j._err(e);}";
        try {
            E = new Function("data, view, j, u", E)
        } catch (T) {
            Q("Compiled template code:\n\n" + E, T)
        }
        t && (t.fn = E);
        return E
    }
    function Z(e, t) {
        function n(n, a, f, l, c, h, m, g, y, b, w, E, S, x, T, N) {
            function C(e, n, r, i, s, o, u) {
                if (n) {
                    t.push(l);
                    if (n !== ".") {
                        var a, f = (r ? 'view._hlp("' + r + '")' : i ? "view" : "data") + (u ? (s ? "." + s : r ? "" : i ? "" : "." + n) + (o || "") : (u = r ? "" : i ? s || "" : n, ""));
                        a = u ? "." + u : "";
                        f += a;
                        f = f.slice(0, 9) === "view.data" ? f.slice(5) : f;
                        return f
                    }
                }
                return e
            }
            c = c || "";
            f = f || a || w;
            l = l || g;
            y = y || T || "";
            if (!h)
                return u ? (u=!E, u ? n : '"') : o ? (o=!S, o ? n : '"') : (f ? (s++, f) : "") + (N ? s ? "" : r ? (r = p, "\b") : "," : m ? (s && Q(e), r = d, "\b" + l + ":") : l ? l.split("^").join(".").replace(v, C) + (y ? (i[++s] = d, y) : c) : c ? c : x ? (i[s--] = p, x) + (y ? (i[++s] = d, y) : "") : b ? (i[s] || Q(e), ",") : a ? "" : (u = E, o = S, '"'));
            Q(e)
        }
        var r, i = {}, s = 0, o = p, u = p;
        t.expr = e.replace(y, "$1");
        return (e + " ").replace(m, n)
    }
    function et(e) {
        return C[e] || (C[e] = "&#" + e.charCodeAt(0) + ";")
    }
    function tt(e, t) {
        return e && e !== t ? t ? B(B({}, t), e) : e : t && B({}, t)
    }
    if (t && t.views || e.jsviews)
        return;
    var r = "v1.0pre", i, s, o, u, a = "{", f = "{", l = "}", c = "}", h = "^", p=!1, d=!0, v = /^(?:null|true|false|\d[\d.]*|([\w$]+|\.|~([\w$]+)|#(view|([\w$]+))?)([\w$.^]*?)(?:[.[^]([\w$]+)\]?)?)$/g, m = /(\()(?=|\s*\()|(?:([([])\s*)?(?:([#~]?[\w$.^]+)?\s*((\+\+|--)|\+|-|&&|\|\||===|!==|==|!=|<=|>=|[<>%*!:?\/]|(=))\s*|([#~]?[\w$.^]+)([([])?)|(,\s*)|(\(?)\\?(?:(')|("))|(?:\s*([)\]])([([]?))|(\s+)/g, g = /\r?\n/g, y = /\\(['"])/g, b = /([\\'"])/g, w = /\x08(~)?([^\x08]+)\x08/g, E = /^if\s/, S = /<(\w+)[>\s]/, x = /<(\w+)[^>\/]*>[^>]*$/, T = 0, N = 0, C = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;"
    }, k = /[<"'&]/g, L = /[\x00<>"'&]/g, A = "data-jsv-tmpl", O = "var j=j||" + (t ? "jQuery." : "js") + "views,", M = [].slice, _ = {}, D = {
        template: {
            compile: X
        },
        tag: {
            compile: W
        },
        helper: {},
        converter: {}
    }, P = {
        jsviews: r,
        render: _,
        View: U,
        settings: {
            delimiters: j,
            debugMode: d
        },
        sub: {
            Error: H,
            tmplFn: G,
            parse: Z,
            extend: B,
            error: K
        },
        _cnvt: q,
        _tag: R,
        _err: function(e) {
            return ut.debugMode ? "Error: " + (e.message || e) + ". " : ""
        }
    };
    (H.prototype = new Error).constructor = H;
    for (s in D)
        $(s, D[s]);
    var nt = P.templates, rt = P.converters, it = P.helpers, st = P.tags, ot = P.sub, ut = P.settings;
    if (t) {
        i = t;
        i.render = _;
        i.views = P;
        i.templates = nt = P.templates;
        i.fn.renderContent = J;
        i.fn.extendCtx = tt;
        i.fn.View = U
    } else {
        i = e.jsviews = P;
        i.isArray = Array && Array.isArray || function(e) {
            return Object.prototype.toString.call(e) === "[object Array]"
        }
    }
    st({
        "if": function(e) {
            var t = this;
            return t._done || arguments.length&&!e ? "" : (t._done=!0, t.renderContent(t.tagCtx.view))
        },
        "else": function() {},
        "for": function() {
            var e, t, r, s = this, o = s.tagCtx, u = "", a = arguments, f = 0, l = a.length;
            if (!l)
                return o.done ? "" : (o.done = d, s.renderContent(o.view));
            for (e = 0; e < l; e++) {
                t = a[e];
                r = t === n;
                if (!!r)
                    return "";
                f += i.isArray(t) ? t.length : 1;
                u += s.renderContent(t)
            }
            o.done = f;
            return u
        },
        "*": function(e) {
            return e
        }
    });
    rt({
        html: function(e) {
            return e != n ? String(e).replace(L, et) : ""
        },
        attr: function(e) {
            return e != n ? String(e).replace(k, et) : ""
        },
        url: function(e) {
            return e != n ? encodeURI(String(e)) : ""
        }
    });
    j()
})(window, window.jQuery);
(function(e, t, n) {
    "use strict";
    function m(e) {
        return o(e) ? new y(e) : new g(e)
    }
    function g(e) {
        this._data = e;
        return this
    }
    function y(e) {
        this._data = e;
        return this
    }
    function b(e, t, n) {
        if (e && t) {
            var r = t.split(".");
            while (e && r.length > n)
                e = e[r.shift()];
            return n ? [e, r.join(".")] : e
        }
    }
    function w(e) {
        if (typeof e != "number")
            throw "Invalid index."
    }
    function E(e, t) {
        e = o(e) ? e : [e];
        var n, r, s = t, u = s, a = e.length, f = [];
        for (n = 0; n < a; n++) {
            r = e[n];
            if (h(r)) {
                i.apply(f, [f.length, 1].concat(E(r.call(t, t), t)));
                continue
            }
            if ("" + r !== r) {
                t = u = r;
                continue
            }
            u !== s && f.push(s = u);
            f.push(r)
        }
        return f
    }
    function S(e, t) {
        var n = e.data;
        if (n.prop === "*" || n.prop === t.path) {
            typeof t.oldValue === a && T(t.oldValue, n.path, n.cb);
            typeof t.value === a && x(t.value, n.path, n.cb, n.root);
            n.cb.call(n.root, e, t)
        }
    }
    function x() {
        function e(e, n) {
            A = m[u];
            if (g) {
                if (A) {
                    t(m).off(e, S);
                    delete v[A.obId]
                }
            } else {
                if (n === "*" && (N = A)) {
                    N = N && N.events;
                    N = N && N.propertyChange;
                    w = N && N.length;
                    while (w--) {
                        T = N[w].data;
                        if (T.cb === y && T.prop !== n) {
                            t(m).off(e + "." + T.prop, S);
                            delete v[A.obId]
                        }
                    }
                }
                t(m).on(e, null, {
                    root: H,
                    path: n,
                    prop: f,
                    cb: y
                }, S);
                if (L) {
                    A = m[u];
                    L[A.obId = A.obId || p++] = m
                }
            }
        }
        var r, o, a, f, c, m, g, y, b, w, T, N, C, k, L, A, O, M = 1, _ = l, D = s.apply([], arguments), P = D.pop(), H = D[0], B = "" + H !== H ? D.shift(): n, j = D.length;
        H = B;
        if (h(P))
            y = P;
        else {
            if (P===!0)
                g = P;
            else if (P) {
                H = P;
                M = 0
            }
            if (h(D[j - 1])) {
                y = D.pop();
                j--
            }
        }
        if (h(D[j - 1])) {
            C = y;
            y = D.pop();
            j--
        }
        _ += g ? y ? "." + y.cbNs : "" : "." + (y.cbNs = y.cbNs || "obs" + d++);
        b = y && y.cbNs;
        g && j === 0 && B && t(B).off(l, S);
        L = y ? M ? y._bnd = y._bnd || {} : y._bnd : n;
        O = 0;
        for (r = 0; r < j; r++) {
            a = D[r];
            if ("" + a !== a) {
                B = a;
                M && (H = B);
                continue
            }
            o = a.split("^");
            if (o[1]) {
                O = o[0].split(".").length;
                a = o.join(".");
                O = a.split(".").length - O
            }
            if (C && (k = C(a, B))) {
                j += k.length - 1;
                i.apply(D, [r--, 1].concat(k));
                continue
            }
            m = B;
            o = a.split(".");
            while (m && typeof m == "object" && (f = o.shift())) {
                if (o.length < O + 1&&!m.nodeType) {
                    if (!g && (N = m[u])) {
                        N = N.events;
                        N = N && N.propertyChange;
                        w = N && N.length;
                        while (w--) {
                            T = N[w].data;
                            if (T && T.cb === y && (T.prop === f || T.prop === "*"))
                                break
                        }
                        if (w>-1) {
                            m = m && m[f];
                            continue
                        }
                    }
                    c = m[f];
                    f === "*" ? h(m) ? (c = h(m) && m.depends) && x(m.depends, y, g || H) : e(_, f) : (c = h(c) && c.depends) ? x(m, E(c, m), y, g || H) : e(_ + "." + f, o.join("."))
                }
                m = m && m[f]
            }
        }
        return {
            cbNs: b,
            bnd: L
        }
    }
    function T() {
        [].push.call(arguments, !0);
        return x.apply(this, arguments)
    }
    if (!t)
        throw "requires jQuery or JsRender";
    if (t.observable)
        return;
    var r = "v1.0pre", i = [].splice, s = [].concat, o = t.isArray, u = t.expando, a = "object", f = "propertyChange", l = f + ".observe", c = "arrayChange", h = t.isFunction, p = 1, d = 1, v;
    t.event.special.propertyChange = {
        remove: function(e) {
            (e = e.data) && (e = e.cb) && (v = e._bnd)
        }
    };
    t.observable = m;
    m.Object = g;
    m.Array = y;
    m.observe = x;
    m.unobserve = T;
    g.prototype = {
        _data: null,
        data: function() {
            return this._data
        },
        observe: function(e, t) {
            return x(this._data, e, t)
        },
        unobserve: function(e, t) {
            return T(this._data, e, t)
        },
        setProperty: function(e, t) {
            var n, r, i, s = this;
            if (o(e)) {
                r = e.length;
                while (r--) {
                    i = e[r];
                    s.setProperty(i.name, i.value)
                }
            } else if ("" + e !== e)
                for (r in e)
                    s.setProperty(r, e[r]);
            else (n = b(s._data, e, 1)
                ) && e !== u && s._setProperty(n[0], n[1], t);
            return s
        },
        _setProperty: function(e, t, n) {
            var r, i, s = e[t];
            if (h(s) && s.set) {
                i = s;
                r = i.set;
                s = s.call(e)
            }
            if (s != n)
                if (!(s instanceof Date) || s > n || s < n) {
                    if (r) {
                        r.call(e, n);
                        n = i.call(e)
                    } else 
                        e[t] = n;
                        this._trigger(e, {
                            path: t,
                            value: n,
                            oldValue: s
                        })
                }
            },
        _trigger: function(e, n) {
            t(e).triggerHandler(f, n)
        }
    };
    y.prototype = {
        _data: null,
        data: function() {
            return this._data
        },
        insert: function(e, t) {
            w(e);
            if (arguments.length > 1) {
                t = o(t) ? t : [t];
                t.length && this._insert(e, t)
            }
            return this
        },
        _insert: function(e, t) {
            i.apply(this._data, [e, 0].concat(t));
            this._trigger({
                change: "insert",
                index: e,
                items: t
            })
        },
        remove: function(e, t) {
            w(e);
            t = t === n || t === null ? 1 : t;
            if (t && e>-1) {
                var r = this._data.slice(e, e + t);
                t = r.length;
                t && this._remove(e, t, r)
            }
            return this
        },
        _remove: function(e, t, n) {
            this._data.splice(e, t);
            this._trigger({
                change: "remove",
                index: e,
                items: n
            })
        },
        move: function(e, t, r) {
            w(e);
            w(t);
            r = r === n || r === null ? 1 : r;
            if (r) {
                var i = this._data.slice(e, e + r);
                this._move(e, t, r, i)
            }
            return this
        },
        _move: function(e, t, n, r) {
            data.splice(e, n);
            i.apply(this._data, [t, 0].concat(r));
            this._trigger({
                change: "move",
                oldIndex: e,
                index: t,
                items: r
            })
        },
        refresh: function(e) {
            var t = this._data.slice(0);
            this._refresh(t, e);
            return this
        },
        _refresh: function(e, t) {
            i.apply(this._data, [0, this._data.length].concat(t));
            this._trigger({
                change: "refresh",
                oldItems: e
            })
        },
        _trigger: function(e) {
            t([this._data]).triggerHandler(c, e)
        }
    }
})(window, window.jQuery || window.jsviews);
(function(e, t, n) {
    "use strict";
    function rt(e) {
        var r, i, s, o, u, a, f, l, c, h, p, v, m, g = e.target, b, w, E = g._jsvBnd;
        if (E) {
            v = E.slice(1).split("&");
            m = v.length;
            while (m--) {
                p = Z[v[m]];
                o = p.linkCtx;
                h = o.view;
                if (E = p.to) {
                    c = t(g);
                    b = h._hlp(D);
                    w = h._hlp(P);
                    s = at(g);
                    r = q[s];
                    u = T(s) ? s(g) : r ? c[r]() : c.attr(s);
                    if ((!b ||!(i = b.call(h, e) === y)) && u !== n) {
                        f = E[1];
                        E = E[0];
                        l = E[0];
                        E = E[2] || E[1];
                        if (T(f))
                            a = f;
                        else {
                            a = h.tmpl.converters;
                            a = a && a[f] || d.converters[f]
                        }
                        a && (u = a.call(o.tag, u));
                        if (u !== n && l) {
                            try {
                                C(l).setProperty(E, u)
                            } catch (S) {
                                W(S)
                            }
                            w && w.call(o, e)
                        }
                    }
                    i && e.stopImmediatePropagation()
                }
            }
        }
    }
    function it(e, r, i) {
        var s, o, u, a, f, l, c, h, p, v, m = this, g = m.tag || {}, S = m.data, x = m.elem, N = x.parentNode, C = N, k = t(x), L = m.view, A = L._hlp(D);
        if ((!A ||!r || A.call(m, e, r) !== y) && (!r || e.data.prop === r.path)) {
            v = L.linkCtx;
            L.linkCtx = m;
            r && (m.eventArgs = r);
            if (r || m._initVal) {
                delete m._initVal;
                a = i.call(m, S, L, d);
                s = m.attr || at(x, b);
                if (l = m.tag)
                    l.parentElem = N;
                l = l || {};
                c = l.ctx;
                T(a) && W(m.expr + ": missing parens");
                if (s === "none")
                    return;
                if (s === "visible") {
                    s = "css-display";
                    a = a && a !== "false" ? ot(x) : "none"
                }
                if (f = s.lastIndexOf("css-", 0) === 0 && s.substr(4))(u = t.style(x, f) !== a) && t.style(x, f, a);
                else {
                    if (s === "value") {
                        if (x.type === "radio") {
                            if (x.value !== "" + a) {
                                ct(m, m.data, m.elem);
                                return 
                            }
                            a = s = "checked"
                        }
                        if (x.type === E) {
                            s = "checked";
                            a = a && a !== "false" ? s : n
                        }
                    }
                    o = q[s];
                    if (o) {
                        if (l || (u = k[o]() !== a)) {
                            if (s === "html") {
                                if (yt(x, b)) {
                                    g.nodes && t(g.nodes(b)).remove();
                                    h = x;
                                    p = x.nextSibling
                                } else {
                                    k.empty();
                                    C = x
                                }
                                l.onBeforeLink && l.onBeforeLink();
                                L.link(S, C, h, p, a);
                                Et(l)
                            } else 
                                s === "text"&&!x.children[0] ? x.textContent !== n ? x.textContent = a : x.innerText = a : k[o](a);
                            x.nodeName.toLowerCase() === "input" && k.blur()
                        }
                    } else if (u = k.attr(s) != a)
                        s === "type" ? k.prop(s, a === n || a === "" ? w : a) : k.attr(s, a === n || a === "" ? w : a)
                    }
                r && u && (A = L._hlp(P)) && A.call(m, e, r)
            }
            ct(m, S, x);
            v ? L.linkCtx = v : delete L.linkCtx
        }
    }
    function st(e, t) {
        var n = this, r = n._hlp(D), i = n._hlp(P);
        if (!r || r.call(this, e, t) !== y) {
            if (t) {
                var s = t.change, o = t.index, u = t.items;
                switch (s) {
                case"insert":
                    n.addViews(o, u);
                    break;
                case"remove":
                    n.removeViews(o, u.length);
                    break;
                case"move":
                    n.refresh();
                    break;
                case"refresh":
                    n.refresh()
                }
            }
            i && i.call(this, e, t)
        }
    }
    function ot(t) {
        var n, r, i = e.getComputedStyle, s = (t.currentStyle || i.call(e, t, "")).display;
        if (s === "none"&&!(s = Q[r = t.nodeName])) {
            n = p.createElement(r);
            p.body.appendChild(n);
            s = (i ? i.call(e, n, "") : n.currentStyle).display;
            Q[r] = s;
            p.body.removeChild(n)
        }
        return s
    }
    function ut(e) {
        var r, i = e.data, s = e._onArrayChange;
        if (!e._.useKey) {
            if (s) {
                t([s[1]]).off(M, s[0]);
                e._onArrayChange = n
            }
            if (i) {
                r = function() {
                    e.data !== n && st.apply(e, arguments)
                };
                t([i]).on(M, r);
                e._onArrayChange = [r, i]
            }
        }
    }
    function at(e, t) {
        var n = d.merge[e.nodeName.toLowerCase()];
        return n ? t ? n.to.toAttr : n.from.fromAttr : t ? "html" : ""
    }
    function ft(e, r, i, s, o, u, a) {
        var f, l, c, h, p, d, v, m, g = e.parentElem, w = e._prv, E = e._nxt, S = e._elCnt;
        if (w && w.parentNode !== g)
            return y;
        if (a) {
            m = e.nodes();
            if (S && w !== E) {
                v = yt(w);
                w && w.removeAttribute(A);
                if (v) {
                    f = v.length;
                    while (f--) {
                        d = v[f];
                        d.open && (Y[d.id]._prv = v.tokens)
                    }
                }
            }
            e.removeViews(n, n, b);
            c = E;
            w = S ? w.previousSibling : w;
            t(m).remove()
        } else {
            if (r) {
                h = s[r - 1];
                if (!h)
                    return y;
                w = h._nxt
            }
            if (S) {
                c = w;
                w = c ? c.previousSibling : h && g.lastChild
            } else 
                c = w.nextSibling
        }
        l = i.render(o, u, e, a || r, e._.useKey && a, b);
        p = e.tag || {};
        p.onBeforeLink && p.onBeforeLink();
        e.link(o, g, w, c, l, a);
        Et(p)
    }
    function lt(e, t, n) {
        var r, i = "_#", s = "_/", o = "_`";
        if (n) {
            Z[r = G++] = t._.tag;
            r = "^" + r
        } else 
            Y[r = t._.id] = t;
        return i + r + o + e + s + r + o
    }
    function ct(e, t, r) {
        var i, s, o, u, a = [], f = e._bndId || "" + G++;
        delete e._bndId;
        if (i = e.tag) {
            a = i.depends;
            a = T(a) ? i.depends() : a;
            o = i.onChange
        }
        o = o || e._cvtBk;
        if (!e._depends || "" + e._depends != "" + a) {
            e._depends && k(t, e._depends, e._handler, b);
            s = k(t, e.paths, a, e._handler, e._filter);
            s.tgt = r;
            s.linkCtx = e;
            r._jsvBnd = r._jsvBnd || "";
            r._jsvBnd += "&" + f;
            e._depends = a;
            e.view._.bnd[f] = f;
            Z[f] = s;
            if (o !== n) {
                u = e.paths[0].split("^").join(".");
                s.to = [e._filter(u) || [e.data, u], o]
            }
        }
    }
    function ht(e, n, r, i, s, o, u) {
        return t.link(this, e, n, r, i, s, o, u)
    }
    function pt(e, r, i, u, a, f, l, c) {
        if (e && r) {
            r = r.jquery ? r : t(r);
            if (!s) {
                s = p.body;
                t(s).on(_, rt)
            }
            var h, d, v, m = lt, g = u && u.target === "replace", w = r.length;
            while (w--) {
                v = r[w];
                if ("" + e === e)
                    vt(e, v, o(v), i, u);
                else {
                    a = a || o(v);
                    if (e.markup !== n) {
                        if (a.link === y) {
                            u = u || {};
                            u.link = m = y
                        }
                        g && (d = v.parentNode);
                        h = e.render(i, u, a, n, n, m);
                        if (d) {
                            f = v.previousSibling;
                            l = v.nextSibling;
                            t.cleanData([v], b);
                            d.removeChild(v);
                            v = d
                        } else {
                            f = l = n;
                            t(v).empty()
                        }
                    } else if (e !== b)
                        break;
                    a.link(i, v, f, l, h)
                }
            }
        }
        return r
    }
    function dt(e, r, s, u, a, f) {
        function l(e, t, n, r, i, s, o, u, a, f, l, c) {
            var h = "";
            u = u || a || "";
            r = r || n || f || c;
            if (r) {
                it = rt;
                W = pt.shift();
                rt = J[W];
                if (it && (ct || ot)) {
                    ct += ot;
                    ot = "";
                    rt || (h = (f || "") + j + "@" + ct + B + (l || ""));
                    ct = rt ? ct + "-" : ""
                }
            }
            if (rt)
                if (s) {
                    ot += s;
                    if (u) {
                        t += u + " " + A + '="' + ot + '"';
                        ot = ""
                    }
                } else 
                    t = (f || c || "") + u;
            else 
                t = s ? t + h + i + j + s + B + o + u : h || e;
            if (u) {
                pt.unshift(W);
                W = u.slice(1);
                it = rt = J[W];
                ct && rt && (ct += "+")
            }
            return t
        }
        function c(e, t) {
            var n, s, o, u;
            if (e) {
                b = e.length;
                if (e.tokens.charAt(3) === "@") {
                    t = S.previousSibling;
                    S.parentNode.removeChild(S)
                }
                while (b--) {
                    N = e[b];
                    if (N.opBnd || N.clBnd)
                        ht.push([S, N]);
                    else {
                        x = Y[u = N.id];
                        if (n = N.path) {
                            y = n.length - 1;
                            while (s = n.charAt(y--))
                                if (s === "+")
                                    if (n.charAt(y) === "-") {
                                        y--;
                                        t = t.previousSibling
                                    } else 
                                        t = t.parentNode;
                                else 
                                    t = t.lastChild
                        }
                        if (!x.link) {
                            x.parentElem = t || S && S.parentNode || r;
                            if (t) {
                                O = "|" + u;
                                nt = t.getAttribute(A) || "";
                                nt.indexOf(O) < 0 && t.setAttribute(A, O + nt);
                                x._prv = 0
                            }
                            g(x, i);
                            x._.onRender = lt;
                            ut(x)
                        }
                        o = x.parentElem;
                        if (N.open) {
                            x.parent === ft && (M = x);
                            x._elCnt = N.elCnt;
                            if (t)
                                t._dfr = "#" + u + (t._dfr || "");
                            else {
                                if (!x._prv) {
                                    o = x.parentElem;
                                    if (x._prv === 0) {
                                        O = o.getAttribute(A);
                                        (O = gt(O, "|" + u)) ? o.setAttribute(A, O) : o.removeAttribute(A)
                                    }
                                    o._dfr = gt(o._dfr, "#" + u)
                                }
                                x._prv = S
                            }
                        } else {
                            if (t)
                                t._dfr = "/" + u + (t._dfr || "");
                            else {
                                x._nxt || (o._dfr = gt(o._dfr, "/" + u));
                                x._nxt = S
                            }
                            h = x.linkCtx;
                            dt && dt.call(h, x)
                        }
                    }
                }
            }
        }
        var h, d, v, m, y, b, E, S, x, T, N, C, k, O, M, _, D, P, F, q, R, U, z, W, Q, G, et, nt, rt, it, st, ot, at, ft = this, ct = "", ht = [], pt = [], dt = ft._hlp(H);
        r = "" + r === r ? t(r)[0] : r.jquery ? r[0] : r || ft.parentElem || p.body;
        W = r.tagName.toLowerCase();
        rt=!!J[W];
        s = s && bt(s, rt);
        u = u && bt(u, rt) || w;
        if (a) {
            G = p.createElement("div");
            Q = G;
            at = ot = "";
            st = r.namespaceURI === "http://www.w3.org/2000/svg" ? "svg" : (z = X.exec(a)) && z[1] || "";
            if (rt) {
                F = u ? u.previousSibling : r.lastChild;
                while (F&&!(T = yt(F)))
                    F = F.previousSibling;
                if (F) {
                    m = T.length;
                    while (m--) {
                        x = T[m].id;
                        if (Y[x].parent === (f ? ft.parent : ft)) {
                            D = x;
                            break
                        }
                    }
                }
                F = u;
                while (F&&!(P = yt(F)))
                    F = F.nextSibling;
                if (nt = P ? P.tokens : r._dfr) {
                    O = D ? "/" + D : "#" + ft._.id;
                    y = nt.indexOf(O);
                    if (y + 1) {
                        y += O.length;
                        P || (r._dfr = nt.slice(y));
                        at = ot = nt.slice(0, y)
                    }
                }
            }
            a = a.replace(tt, l);
            V.appendChild(G);
            st = K[st] || K.div;
            q = st[0];
            Q.innerHTML = st[1] + a + st[2];
            while (q--)
                Q = Q.lastChild;
            V.removeChild(G);
            R = p.createDocumentFragment();
            while (U = Q.firstChild) {
                !_ && U.nodeType === 1 && (_ = U);
                R.appendChild(U)
            }
            r.insertBefore(R, u)
        }
        E = $ ? r.querySelectorAll(I) : t(I, r).get();
        m = E.length;
        if (s) {
            k = $ ? s.querySelectorAll(I) : t(I, s).get();
            s = k.length ? k[k.length - 1] : s
        }
        for (v = 0; v < m; v++) {
            S = E[v];
            if (s)
                s = S !== s && s;
            else {
                if (u && S === u)
                    break;
                if (S.parentNode) {
                    c(yt(S));
                    S.getAttribute(L) && ht.push([S])
                }
            }
        }
        u && u.type === "jsv" && u.parentNode.removeChild(u);
        s && s.type === "jsv" && s.parentNode.removeChild(s);
        if (rt && ct + ot) {
            S = u;
            c(yt(ct + ot), !S && r);
            if (u) {
                nt = u.getAttribute(A);
                nt && u.setAttribute(A, ot + nt.slice(nt.indexOf(at) + at.length));
                M && (M._nxt = u)
            }
        }
        M && M._elCnt && M.parentElem === r && (M._nxt = u);
        if (rt && ft._prv&&!ft._prv.parentNode) {
            T = yt(ft._prv);
            m = T.length;
            _ ? _.setAttribute(A, T.tokens) : u.setAttribute(A, et + u.getAttribute(A));
            while (m--) {
                N = T[m];
                if (x = Y[N.id])
                    N.open ? x._prv = _ : N.close && (x._nxt = _ || ft._nxt)
                }
        }
        m = ht.length;
        for (v = 0; v < m; v++) {
            S = ht[v];
            C = S[1];
            S = S[0];
            if (C)
                if (C.opBnd) {
                    d = Z[C.id];
                    d.parentElem = S.parentNode;
                    d._prv = S;
                    d._elCnt = C.elCnt;
                    d && d.onBeforeLink && d.onBeforeLink()
                } else {
                    d = Z[C.id];
                    x = d.tagCtx.view;
                    d._nxt = S;
                    Et(d);
                    vt(n, d._prv, x, x.data || e, C.id)
                } else {
                x = o(S);
                vt(S.getAttribute(L), S, x, x.data || e)
            }
        }
    }
    function vt(e, t, r, i, s) {
        var o, h, p, m, g, y, E, S, x, T, N;
        if (s) {
            T = Z[s];
            e = f + T.tagName + " " + T.tagCtx.params + l
        }
        if (e && t) {
            o = r.tmpl.links;
            e = wt(e, t);
            u.lastIndex = 0;
            while (h = u.exec(e)) {
                N = u.lastIndex;
                p = s ? "html" : h[1];
                E = h[3];
                g = h[10];
                m = n;
                x = {
                    data: i,
                    elem: t,
                    view: r,
                    attr: p,
                    _initVal: !s&&!h[2]
                };
                if (h[6]) {
                    if (!p && (m = /^.*:([\w$]*)$/.exec(g))) {
                        m = m[1];
                        if (m !== n) {
                            y =- m.length - 1;
                            E = E.slice(0, y - 1) + l;
                            g = g.slice(0, y)
                        }
                    }
                    m === w && (m = n)
                }
                if (s) {
                    x.tag = T;
                    x._bndId = s
                }
                x.expr = p + E;
                S = o[E] = o[E] || d.sub.tmplFn(a + E + c, n, b);
                v.parse(g, S.paths = x.paths = []);
                !p && m !== n && (x._cvtBk = m);
                mt(x, S);
                u.lastIndex = N
            }
        }
    }
    function mt(e, t) {
        function n(n, r) {
            it.call(e, n, r, t)
        }
        e._filter = At(e);
        e._handler = n;
        n()
    }
    function gt(e, t) {
        e = e || "";
        var n = e.indexOf(t);
        return n + 1 ? e.slice(0, n) + e.slice(n + t.length) : e
    }
    function yt(e, t) {
        function n(e, t, n, i, s, o, u, a, f) {
            r.push({
                elCnt: !t,
                id: a,
                open: n,
                close: i,
                opBnd: s,
                clBnd: o,
                prnt: u,
                path: f
            })
        }
        var r = [], i = "" + e === e ? e: e.tagName === "SCRIPT" ? e.type: e.nodeType === 1 && e.getAttribute(A);
        if (i) {
            r.tokens = i;
            i.replace(nt, n);
            return t ? r.pop().opBnd : r
        }
    }
    function bt(e, t) {
        var n = e;
        while (t && n && n.nodeType !== 1)
            n = n.previousSibling;
        if (n)
            if (n.nodeType !== 1) {
                n = p.createElement("SCRIPT");
                n.type = "jsv";
                e.parentNode.insertBefore(n, e)
            } else 
                !yt(n)&&!n.getAttribute(L) && n.setAttribute(L, "");
        return n
    }
    function wt(e, n) {
        e = t.trim(e);
        return e.slice( - 1) !== l ? e = f + ":" + e + (at(n) ? ":" : "") + l : e
    }
    function Et(e) {
        if (e) {
            e.contents = St;
            e.nodes = xt;
            e.onAfterLink && e.onAfterLink()
        }
    }
    function St(e, n) {
        var r, i = t(this.nodes());
        if (i[0]) {
            r = e ? i.filter(e) : i;
            i = n && e ? r.add(i.find(e)) : r
        }
        return i
    }
    function xt(e, t, n) {
        var r, i = this._elCnt, s=!t && i, o = [];
        t = t || this._prv;
        n = n || this._nxt;
        r = s ? t === this._nxt ? this.parentElem.lastSibling : t : t.nextSibling;
        while (r && (!n || r !== n)) {
            (e || i ||!yt(r)) && o.push(r);
            r = r.nextSibling
        }
        return o
    }
    function Tt(e) {
        var t, r, i, s, o, u, a, f = [];
        for (t = 0; r = e[t]; t++)
            f.push(r);
        for (t = 0; r = f[t]; t++)
            if (r.parentNode) {
                if (o = r._jsvBnd) {
                    o = o.slice(1).split("&");
                    r._jsvBnd = "";
                    u = o.length;
                    while (u--)
                        Ct(o[u])
                    }
                    if ((i = yt(r)) && i.length) {
                        i = i[i.length - 1];
                        i.opBnd || i.clBnd ? r.parentNode.removeChild(r) : (i.open || i.prnt) && (i = Y[i.id]) && i.data !== n && i.parent.removeViews(i._.key, n, b)
                    } else if (a = r._jsvVws) {
                        a = a.slice(1).split("&");
                        u = a.length;
                        while (u--)(s = Y[a[u]]) && s.parent.removeViews(s._.key, n, b)
                        }
                }
            }
    function Nt(e, n, r) {
        var i, s = 0, o = e.length;
        while (o-->0) {
            i = e[o];
            if (!(n && n !== i[0] || r && r !== i[1])) {
                x(i[0]) ? t([i[0]]).off(M, i[2]) : $unobserve(i[0], i[2]);
                e.splice(o, 1);
                s++
            }
        }
        return s
    }
    function Ct(e) {
        var n, r, i, s = Z[e];
        if (s) {
            for (n in s.bnd) {
                t(s.bnd[n]).off(O + "." + s.cbNs);
                delete s.bnd[n]
            }
            r = s.linkCtx;
            i = r.tag || {};
            i.onDispose && i.onDispose();
            delete r.view._.bnd[e];
            delete Z[e]
        }
    }
    function kt(e, r) {
        if (!arguments.length) {
            if (s) {
                t(s).off(_, rt);
                s = n
            }
            e = b;
            S.removeViews();
            Tt(p.body.getElementsByTagName("*"))
        } else if (r) {
            r = r.jquery ? r : t(r);
            e === b ? t.each(r, function() {
                var e;
                while ((e = o(this, b)) && e.parent)
                    e.parent.removeViews(e._.key, n, b);
                Tt(this.getElementsByTagName("*"))
            }) : e === n && Tt(r)
        }
        return r
    }
    function Lt(e, t) {
        return kt(this, e, t)
    }
    function At(e) {
        return function(t, n) {
            var r, i = e.view.ctx;
            if (t) {
                if (t.charAt(0) === "~") {
                    if (t.slice(0, 4) === "~tag") {
                        if (t.charAt(4) === ".") {
                            r = t.slice(5).split(".");
                            i = i.tag
                        } else if (t.slice(4, 6) === "s.") {
                            r = t.slice(6).split(".");
                            i = i.tags[r.shift()]
                        }
                        if (r)
                            return i ? [i, r.join("."), n] : []
                    }
                    t = t.slice(1).split(".");
                    return [e.view._hlp(t.shift()), t.join("."), n]
                }
                if (t.charAt(0) === "#")
                    return t === "#data" ? [] : [e.view, t.replace(et, ""), n]
            }
        }
    }
    function Ot(e) {
        return e.type === E ? e.checked : t(e).val()
    }
    function Mt(e) {
        if ("" + e === e) {
            var t = e.split("[");
            e = N[t.shift()];
            while (e && t.length)
                e = e.tmpls[t.shift().slice(0, - 1)]
        }
        return e
    }
    if (!t)
        throw "requires jQuery";
    if (!t.views)
        throw "requires JsRender";
    if (t.link)
        return;
    var r = "v1.0pre", i, s, o, u, a, f, l, c, h, p = e.document, d = t.views, v = d.sub, m = d.settings, g = v.extend, y=!1, b=!0, w = null, E = "checkbox", S = d.View(n, "top"), x = t.isArray, T = t.isFunction, N = d.templates, C = t.observable, k = C.observe, L = "data-link", A = "data-jsv", O = "propertyChange", M = "arrayChange", _ = "change.jsv", D = "onBeforeChange", P = "onAfterChange", H = "onAfterCreate", B = '"></script>', j = '<script type="jsv', F = "script,[" + A + "]", I = F + ",[" + L + "]", q = {
        value: "val",
        input: "val",
        html: "html",
        text: "text"
    }, R = {
        from: {
            fromAttr: "value"
        },
        to: {
            toAttr: "value"
        }
    }, U = t.cleanData, z = d.settings.delimiters, W = v.error, X = /<(?!script)(\w+)[>\s]/, V = p.createDocumentFragment(), $ = p.querySelector, J = {
        ol: 1,
        ul: 1,
        table: 1,
        tbody: 1,
        thead: 1,
        tfoot: 1,
        tr: 1,
        colgroup: 1,
        dl: 1,
        select: 1,
        optgroup: 1
    }, K = d.settings.wrapMap = {
        option: [1, "<select multiple='multiple'>", "</select>"],
        legend: [1, "<fieldset>", "</fieldset>"],
        thead: [1, "<table>", "</table>"],
        tr: [2, "<table><tbody>", "</tbody></table>"],
        td: [3, "<table><tbody><tr>", "</tr></tbody></table>"],
        col: [2, "<table><tbody></tbody><colgroup>", "</colgroup></table>"],
        area: [1, "<map>", "</map>"],
        svg: [1, "<svg>", "</svg>"],
        div: [1, "x<div>", "</div>"]
    }, Q = {}, G = 1, Y = {
        0: S
    }, Z = {}, et = /^#(view\.?)?/, tt = /(^|(\/>)|(<\/\w+>)|>|)(\s*)_([#\/]\^?\d+)_`(\s*)(<\w+(?=[\s\/>]))?|\s*(?:(<\w+(?=[\s\/>]))|(<\/\w+>)(\s*)|(\/>)\s*)/g, nt = /(jsv)?(?:(#)|(\/)|(#\^)|(\/\^)|(\|))(\d+)([-+@\d]+)?/g;
    K.optgroup = K.option;
    K.tbody = K.tfoot = K.colgroup = K.caption = K.thead;
    K.th = K.td;
    v.onStoreItem = function(e, n, r) {
        if (r && e === N) {
            r.link = ht;
            r.unlink = Lt;
            if (n) {
                t.link[n] = function() {
                    return ht.apply(r, arguments)
                };
                t.unlink[n] = function() {
                    return Lt.apply(r, arguments)
                }
            }
        }
    };
    (d.settings.delimiters = function() {
        var e = z.apply(d, arguments);
        a = e[0];
        f = e[1];
        l = e[2];
        c = e[3];
        h = e[4];
        u = new RegExp("(?:^|\\s*)([\\w-]*)(\\" + h + ")?(\\" + f + d.rTag + "\\" + l + ")", "g");
        return this
    })();
    i = {
        refresh: function(e) {
            var t = this, n = t.parent;
            t.tmpl = Mt(t.tmpl);
            if (n) {
                ft(t, t.index, t.tmpl, n.views, t.data, e, b);
                ut(t)
            }
            return t
        },
        addViews: function(e, t, n) {
            var r, i, s = this, o = t.length, u = s.views;
            if (!s._.useKey && o && (n = Mt(n || s.tmpl))) {
                i = u.length + o;
                if (ft(s, e, n, u, t, s.ctx) !== y)
                    for (r = e + o; r < i; r++)
                        C(u[r]).setProperty("index", r)
            }
            return s
        },
        removeViews: function(e, r, i) {
            function s(e) {
                var r, s, o, a, f, l, h, p, d, v, m, g, y, w, E = 0, S = c[e];
                if (S) {
                    i || (v = S.nodes());
                    S.removeViews(n, n, b);
                    S.data = n;
                    r = S._.id;
                    a = S._prv;
                    f = S._nxt;
                    o = S.parentElem;
                    l = a === f;
                    i || t(v).remove();
                    if (S._elCnt) {
                        if (a) {
                            p = yt(a);
                            h = "";
                            for (g = 0, y = p.length; g < y; g++) {
                                m = p[g];
                                if (m.id === r)
                                    break;
                                if (!l) {
                                    u = Y[m.id];
                                    if (m.open) {
                                        if (!g&&!f) {
                                            o.setAttribute(A, "|" + m.id + (o.getAttribute(A) || ""));
                                            f = 0
                                        }
                                        u._prv&&!f && (h += "#" + u._.id);
                                        u._prv = f
                                    } else if (m.close) {
                                        u._nxt&&!f && (h += "/" + u._.id);
                                        u._nxt = f
                                    }
                                }
                                E += m.id.length + 1
                            }
                            o._dfr = h + (o._dfr || "");
                            if (f) {
                                h = f && f.getAttribute(A);
                                h && f.setAttribute(A, p.tokens.slice(0, E) + h.slice(h.indexOf("/" + r) + r.length + 1))
                            } else if (S._prev === 0) {
                                h = o.getAttribute(A);
                                (token = gt(h, "|" + r)) ? o.setAttribute(A, token) : o.removeAttribute(A)
                            }
                        } else 
                            o._dfr = gt(o._dfr, "#" + r);
                        f || (o._dfr = gt(o._dfr, "/" + r))
                    } else 
                        try {
                            o.removeChild(a);
                            o.removeChild(f)
                        } catch (x) {}
                    ut(S);
                    for (s in S._.bnd)
                        Ct(s, i);
                    delete Y[r]
                }
            }
            var o, u, a, f = this, l=!f._.useKey, c = f.views;
            l && (a = c.length);
            if (e === n)
                if (l) {
                    o = a;
                    while (o--)
                        s(o);
                        f.views = []
                } else {
                    for (u in c)
                        s(u);
                        f.views = {}
                } else {
                if (r === n)
                    if (l)
                        r = 1;
                    else {
                        s(e);
                        delete c[e]
                    }
                if (l && r) {
                    o = e + r;
                    while (o-->e)
                        s(o);
                    c.splice(e, r);
                    if (a = c.length)
                        while (e < a)
                            C(c[e]).setProperty("index", e++)
                        }
            }
            return this
        },
        nodes: xt,
        contents: St,
        link: dt
    };
    g(d, {
        linkAttr: L,
        merge: {
            input: {
                from: {
                    fromAttr: Ot
                },
                to: {
                    toAttr: "value"
                }
            },
            textarea: R,
            select: R,
            optgroup: {
                from: {
                    fromAttr: "label"
                },
                to: {
                    toAttr: "label"
                }
            }
        }
    });
    m.debugMode && (window._jsv = {
        views: Y,
        bindings: Z
    });
    g(t, {
        view: o = function(e, r) {
            var i, s, o, u = 0, a = p.body, f = r === b;
            if (e && e !== a && S._.useKey > 1) {
                e = "" + e === e ? t(e)[0] : e.jquery ? e[0] : e;
                if (e) {
                    if (f)
                        return (i = $ ? e.querySelector(F) : t(F, e)[0]) && (i = yt(i)) && Y[i[0].id] || n;
                    while (e) {
                        if (s = yt(e)) {
                            o = s.length;
                            while (o--) {
                                i = s[o];
                                if (u <= 0 && i.open) {
                                    i = Y[i.id];
                                    return i && r ? i.get(r) : i
                                }
                                u += i.close ? 1 : i.open?-1 : 0
                            }
                        }
                        e = e.previousSibling || e.parentNode
                    }
                }
            }
            return f ? n : S
        },
        link: pt,
        unlink: kt,
        cleanData: function(e) {
            if (e.length) {
                Tt(e);
                U.call(t, e)
            }
        }
    });
    g(t.fn, {
        link: function(e, t, n, r, i, s, o) {
            return pt(e, this, t, n, r, i, s, o)
        },
        unlink: function(e, t) {
            return kt(e, this, t)
        },
        view: function(e) {
            return o(this[0], e)
        }
    });
    g(S, {
        tmpl: {
            links: {}
        }
    });
    g(S, i);
    S._.onRender = lt
})(window, window.jQuery);
$.getObject("Pb", !0).JSTemplates = function() {
    function e(e) {
        e.indexOf("#")===-1 && (e = "#" + e);
        return e
    }
    function t(t) {
        var n = e(t);
        typeof templates == "undefined" && (window.templates = {});
        if (templates.hasOwnProperty(n))
            return templates[n];
        if ($(n).length) {
            templates[n] || (templates[n] = $.templates(n));
            return templates[n]
        }
        return null
    }
    function n(e, n, r) {
        var i = t(e);
        i && i.link(n, r)
    }
    function r(e, n) {
        var r = t(e);
        return r ? r.render(n) : null
    }
    function i() {
        if (typeof templates != "undefined") {
            var t;
            for (t in templates)
                if (templates.hasOwnProperty(t)) {
                    t = e(t);
                    $.views.templates[t] = templates[t]
                }
        }
    }
    function s() {
        var e = {
            getPicUrl: function(e) {
                return Pb.Data.get("picUrl") + e
            },
            getIsMobile: function(e) {
                return Pb.Data.get("isMobile")
            },
            getFields: function(e) {
                var t, n, r = [], i = 0;
                for (t in e)
                    if (e.hasOwnProperty(t)) {
                        n = e[t];
                        r.push({
                            key: t,
                            value: n,
                            index: i
                        });
                        i++
                    }
                return r
            },
            truncateString: function(e, t) {
                return e.length > t ? e.substring(0, t - 3) + "..." : e
            }
        };
        $.views.helpers(e)
    }
    s();
    return {
        link: n,
        render: r,
        setupTemplates: i
    }
}();
!function(e) {
    "use strict";
    var t = '[data-dismiss="alert"]', n = function(n) {
        e(n).on("click", t, this.close)
    };
    n.prototype.close = function(t) {
        function s() {
            i.trigger("closed").remove()
        }
        var n = e(this), r = n.attr("data-target"), i;
        if (!r) {
            r = n.attr("href");
            r = r && r.replace(/.*(?=#[^\s]*$)/, "")
        }
        i = e(r);
        t && t.preventDefault();
        i.length || (i = n.hasClass("alert") ? n : n.closest(".alert"));
        i.trigger(t = e.Event("close"));
        if (t.isDefaultPrevented())
            return;
        i.removeClass("in");
        e.support.transition && i.hasClass("fade") ? i.on(e.support.transition.end, s) : s()
    };
    e.fn.alert = function(t) {
        return this.each(function() {
            var r = e(this), i = r.data("alert");
            i || r.data("alert", i = new n(this));
            r.addClass("in");
            typeof t == "string" && i[t].call(r)
        })
    };
    e.fn.alert.Constructor = n;
    e(function() {
        e("body").on("click.alert.data-api", t, n.prototype.close)
    })
}(window.jQuery);
!function(e) {
    "use strict";
    function r() {
        e(t).parent().removeClass("open")
    }
    var t = '[data-toggle="dropdown"]', n = function(t) {
        var n = e(t).on("click.dropdown.data-api", this.toggle);
        e("html").on("click.dropdown.data-api", function() {
            n.parent().removeClass("open")
        })
    };
    n.prototype = {
        constructor: n,
        toggle: function(t) {
            var n = e(this), i = n.attr("data-target"), s, o;
            if (!i) {
                i = n.attr("href");
                i = i && i.replace(/.*(?=#[^\s]*$)/, "")
            }
            s = e(i);
            s.length || (s = n.parent());
            o = s.hasClass("open");
            r();
            !o && s.toggleClass("open");
            return !1
        }
    };
    e.fn.dropdown = function(t) {
        return this.each(function() {
            var r = e(this), i = r.data("dropdown");
            i || r.data("dropdown", i = new n(this));
            typeof t == "string" && i[t].call(r)
        })
    };
    e.fn.dropdown.Constructor = n;
    e(function() {
        e("html").on("click.dropdown.data-api", r);
        e("body").on("click.dropdown.data-api", t, n.prototype.toggle)
    })
}(window.jQuery);
!function(e) {
    "use strict";
    function n() {
        var t = this, n = setTimeout(function() {
            t.$element.off(e.support.transition.end);
            r.call(t)
        }, 500);
        this.$element.one(e.support.transition.end, function() {
            clearTimeout(n);
            r.call(t)
        })
    }
    function r(e) {
        this.$element.hide().trigger("hidden");
        i.call(this)
    }
    function i(t) {
        var n = this, r = this.$element.hasClass("fade") ? "fade": "";
        if (this.isShown && this.options.backdrop) {
            var i = e.support.transition && r;
            this.$backdrop = e('<div class="modal-backdrop ' + r + '" />').appendTo(document.body);
            this.options.backdrop != "static" && this.$backdrop.click(e.proxy(this.hide, this));
            i && this.$backdrop[0].offsetWidth;
            this.$backdrop.addClass("in");
            i ? this.$backdrop.one(e.support.transition.end, t) : t()
        } else if (!this.isShown && this.$backdrop) {
            this.$backdrop.removeClass("in");
            e.support.transition && this.$element.hasClass("fade") ? this.$backdrop.one(e.support.transition.end, e.proxy(s, this)) : s.call(this)
        } else 
            t && t()
    }
    function s() {
        this.$backdrop.remove();
        this.$backdrop = null
    }
    function o() {
        var t = this;
        this.isShown && this.options.keyboard ? e(document).on("keyup.dismiss.modal", function(e) {
            e.which == 27 && t.hide()
        }) : this.isShown || e(document).off("keyup.dismiss.modal")
    }
    function u() {
        var t = this.$element.width(), n = this.$element.height(), r = e(window).width(), i = e(window).height(), s, o;
        n > i ? s = 20 : s = Math.floor((i - n) / 2);
        t > r ? o = 20 : o = Math.floor((r - t) / 2);
        this.$element.css("top", e(window).scrollTop() + s + "px").css("left", e(window).scrollLeft() + o + "px")
    }
    var t = function(t, n) {
        this.options = n;
        this.$element = e(t).delegate('[data-dismiss="modal"]', "click.dismiss.modal", e.proxy(this.hide, this))
    };
    t.prototype = {
        constructor: t,
        toggle: function() {
            return this[this.isShown ? "hide": "show"]()
        },
        show: function() {
            var t = this;
            if (this.isShown)
                return;
            e("body").addClass("modal-open");
            this.isShown=!0;
            this.$element.trigger("show");
            o.call(this);
            i.call(this, function() {
                var n = e.support.transition && t.$element.hasClass("fade");
                !t.$element.parent().length && t.$element.appendTo(document.body);
                t.$element.show();
                u.call(t);
                n && t.$element[0].offsetWidth;
                t.$element.addClass("in");
                t.$element.on("shown", function() {
                    u.call(t);
                    e(window).on("resize.modal", e.proxy(u, t))
                });
                n ? t.$element.one(e.support.transition.end, function() {
                    t.$element.trigger("shown")
                }) : t.$element.trigger("shown")
            })
        },
        hide: function(t) {
            t && t.preventDefault();
            if (!this.isShown)
                return;
            var i = this;
            this.isShown=!1;
            e("body").removeClass("modal-open");
            o.call(this);
            e(window).off("resize.modal");
            this.$element.trigger("hide").removeClass("in");
            e.support.transition && this.$element.hasClass("fade") ? n.call(this) : r.call(this)
        }
    };
    e.fn.modal = function(n) {
        return this.each(function() {
            var r = e(this), i = r.data("modal"), s = e.extend({}, e.fn.modal.defaults, r.data(), typeof n == "object" && n);
            i || r.data("modal", i = new t(this, s));
            typeof n == "string" ? i[n]() : s.show && i.show()
        })
    };
    e.fn.modal.defaults = {
        backdrop: !0,
        keyboard: !0,
        show: !0
    };
    e.fn.modal.Constructor = t;
    e(function() {
        e("body").on("click.modal.data-api", '[data-toggle="modal"]', function(t) {
            var n = e(this), r, i = e(n.attr("data-target") || (r = n.attr("href")) && r.replace(/.*(?=#[^\s]+$)/, "")), s = i.data("modal") ? "toggle": e.extend({}, i.data(), n.data());
            t.preventDefault();
            i.modal(s)
        })
    })
}(window.jQuery);
!function(e) {
    "use strict";
    var t = function(e, t) {
        this.init("tooltip", e, t)
    };
    t.prototype = {
        constructor: t,
        init: function(t, n, r) {
            var i, s;
            this.type = t;
            this.$element = e(n);
            this.options = this.getOptions(r);
            this.enabled=!0;
            if (this.options.trigger != "manual") {
                i = this.options.trigger == "hover" ? "mouseenter" : "focus";
                s = this.options.trigger == "hover" ? "mouseleave" : "blur";
                this.$element.on(i, this.options.selector, e.proxy(this.enter, this));
                this.$element.on(s, this.options.selector, e.proxy(this.leave, this))
            }
            this.options.selector ? this._options = e.extend({}, this.options, {
                trigger: "manual",
                selector: ""
            }) : this.fixTitle()
        },
        getOptions: function(t) {
            t = e.extend({}, e.fn[this.type].defaults, t, this.$element.data());
            t.delay && typeof t.delay == "number" && (t.delay = {
                show: t.delay,
                hide: t.delay
            });
            return t
        },
        enter: function(t) {
            var n = e(t.currentTarget)[this.type](this._options).data(this.type);
            if (!n.options.delay ||!n.options.delay.show)
                n.show();
            else {
                n.hoverState = "in";
                setTimeout(function() {
                    n.hoverState == "in" && n.show()
                }, n.options.delay.show)
            }
        },
        leave: function(t) {
            var n = e(t.currentTarget)[this.type](this._options).data(this.type);
            if (!n.options.delay ||!n.options.delay.hide)
                n.hide();
            else {
                n.hoverState = "out";
                setTimeout(function() {
                    n.hoverState == "out" && n.hide()
                }, n.options.delay.hide)
            }
        },
        show: function() {
            var e, t, n, r, i, s, o;
            if (this.hasContent() && this.enabled) {
                e = this.tip();
                this.setContent();
                this.options.animation && e.addClass("fade");
                s = typeof this.options.placement == "function" ? this.options.placement.call(this, e[0], this.$element[0]) : this.options.placement;
                t = /in/.test(s);
                e.remove().css({
                    top: 0,
                    left: 0,
                    display: "block"
                }).appendTo(t ? this.$element : document.body);
                n = this.getPosition(t);
                r = e[0].offsetWidth;
                i = e[0].offsetHeight;
                switch (t ? s.split(" ")[1] : s) {
                case"bottom":
                    o = {
                        top: n.top + n.height,
                        left: n.left + n.width / 2 - r / 2
                    };
                    break;
                case"top":
                    o = {
                        top: n.top - i,
                        left: n.left + n.width / 2 - r / 2
                    };
                    break;
                case"left":
                    o = {
                        top: n.top + n.height / 2 - i / 2,
                        left: n.left - r
                    };
                    break;
                case"right":
                    o = {
                        top: n.top + n.height / 2 - i / 2,
                        left: n.left + n.width
                    }
                }
                e.css(o).addClass(s).addClass("in")
            }
        },
        setContent: function() {
            var e = this.tip();
            e.find(".tooltip-inner").html(this.getTitle());
            e.removeClass("fade in top bottom left right")
        },
        hide: function() {
            function r() {
                var t = setTimeout(function() {
                    n.off(e.support.transition.end).remove()
                }, 500);
                n.one(e.support.transition.end, function() {
                    clearTimeout(t);
                    n.remove()
                })
            }
            var t = this, n = this.tip();
            n.removeClass("in");
            e.support.transition && this.$tip.hasClass("fade") ? r() : n.remove()
        },
        fixTitle: function() {
            var e = this.$element;
            (e.attr("title") || typeof e.attr("data-original-title") != "string") && e.attr("data-original-title", e.attr("title") || "").removeAttr("title")
        },
        hasContent: function() {
            return this.getTitle()
        },
        getPosition: function(t) {
            return e.extend({}, t ? {
                top: 0,
                left: 0
            } : this.$element.offset(), {
                width: this.$element[0].offsetWidth,
                height: this.$element[0].offsetHeight
            })
        },
        getTitle: function() {
            var e, t = this.$element, n = this.options;
            e = t.attr("data-original-title") || (typeof n.title == "function" ? n.title.call(t[0]) : n.title);
            e = e.toString().replace(/(^\s*|\s*$)/, "");
            return e
        },
        tip: function() {
            return this.$tip = this.$tip || e(this.options.template)
        },
        validate: function() {
            if (!this.$element[0].parentNode) {
                this.hide();
                this.$element = null;
                this.options = null
            }
        },
        enable: function() {
            this.enabled=!0
        },
        disable: function() {
            this.enabled=!1
        },
        toggleEnabled: function() {
            this.enabled=!this.enabled
        },
        toggle: function() {
            this[this.tip().hasClass("in") ? "hide": "show"]()
        }
    };
    e.fn.tooltip = function(n) {
        return this.each(function() {
            var r = e(this), i = r.data("tooltip"), s = typeof n == "object" && n;
            i || r.data("tooltip", i = new t(this, s));
            typeof n == "string" && i[n]()
        })
    };
    e.fn.tooltip.Constructor = t;
    e.fn.tooltip.defaults = {
        animation: !0,
        delay: 0,
        selector: !1,
        placement: "top",
        trigger: "hover",
        title: "",
        template: '<div class="tooltip"><div class="tooltip-arrow"></div><div class="tooltip-inner"></div></div>'
    }
}(window.jQuery);
!function(e) {
    "use strict";
    var t = function(e, t) {
        this.init("popover", e, t)
    };
    t.prototype = e.extend({}, e.fn.tooltip.Constructor.prototype, {
        constructor: t,
        setContent: function() {
            var t = this.tip(), n = this.getTitle(), r = this.getContent();
            t.find(".popover-title")[e.type(n) == "object" ? "append": "html"](n);
            t.find(".popover-content > *")[e.type(r) == "object" ? "append": "html"](r);
            t.removeClass("fade top bottom left right in")
        },
        hasContent: function() {
            return this.getTitle() || this.getContent()
        },
        getContent: function() {
            var e, t = this.$element, n = this.options;
            e = t.attr("data-content") || (typeof n.content == "function" ? n.content.call(t[0]) : n.content);
            e = e.toString().replace(/(^\s*|\s*$)/, "");
            return e
        },
        tip: function() {
            this.$tip || (this.$tip = e(this.options.template));
            return this.$tip
        }
    });
    e.fn.popover = function(n) {
        return this.each(function() {
            var r = e(this), i = r.data("popover"), s = typeof n == "object" && n;
            i || r.data("popover", i = new t(this, s));
            typeof n == "string" && i[n]()
        })
    };
    e.fn.popover.Constructor = t;
    e.fn.popover.defaults = e.extend({}, e.fn.tooltip.defaults, {
        placement: "right",
        content: "",
        template: '<div class="popover"><div class="arrow"></div><div class="popover-inner"><h3 class="popover-title"></h3><div class="popover-content"><p></p></div></div></div>'
    })
}(window.jQuery);
!function(e) {
    "use strict";
    var t = function(t) {
        this.element = e(t)
    };
    t.prototype = {
        constructor: t,
        show: function() {
            var t = this.element, n = t.closest("ul:not(.dropdown-menu)"), r = t.attr("data-target"), i, s;
            if (!r) {
                r = t.attr("href");
                r = r && r.replace(/.*(?=#[^\s]*$)/, "")
            }
            if (t.parent("li").hasClass("active"))
                return;
            i = n.find(".active a").last()[0];
            t.trigger({
                type: "show",
                relatedTarget: i
            });
            s = e(r);
            this.activate(t.parent("li"), n);
            this.activate(s, s.parent(), function() {
                t.trigger({
                    type: "shown",
                    relatedTarget: i
                })
            })
        },
        activate: function(t, n, r) {
            function o() {
                i.removeClass("active").find("> .dropdown-menu > .active").removeClass("active");
                t.addClass("active");
                if (s) {
                    t[0].offsetWidth;
                    t.addClass("in")
                } else 
                    t.removeClass("fade");
                t.parent(".dropdown-menu") && t.closest("li.dropdown").addClass("active");
                r && r()
            }
            var i = n.find("> .active"), s = r && e.support.transition && i.hasClass("fade");
            s ? i.one(e.support.transition.end, o) : o();
            i.removeClass("in")
        }
    };
    e.fn.tab = function(n) {
        return this.each(function() {
            var r = e(this), i = r.data("tab");
            i || r.data("tab", i = new t(this));
            typeof n == "string" && i[n]()
        })
    };
    e.fn.tab.Constructor = t;
    e(function() {
        e("body").on("click.tab.data-api", '[data-toggle="tab"], [data-toggle="pill"]', function(t) {
            t.preventDefault();
            e(this).tab("show")
        })
    })
}(window.jQuery);
!function(e) {
    "use strict";
    e(function() {
        e.support.transition = function() {
            var e = function() {
                var e = document.createElement("bootstrap"), t = {
                    WebkitTransition: "webkitTransitionEnd",
                    MozTransition: "transitionend",
                    OTransition: "oTransitionEnd otransitionend",
                    transition: "transitionend"
                }, n;
                for (n in t)
                    if (e.style[n] !== undefined)
                        return t[n]
            }();
            return e && {
                end: e
            }
        }()
    })
}(window.jQuery);
!function(e) {
    "use strict";
    var t = function(t, n) {
        this.$element = e(t);
        this.$indicators = this.$element.find(".carousel-indicators");
        this.options = n;
        this.options.pause == "hover" && this.$element.on("mouseenter", e.proxy(this.pause, this)).on("mouseleave", e.proxy(this.cycle, this))
    };
    t.prototype = {
        cycle: function(t) {
            t || (this.paused=!1);
            this.interval && clearInterval(this.interval);
            this.options.interval&&!this.paused && (this.interval = setInterval(e.proxy(this.next, this), this.options.interval));
            return this
        },
        getActiveIndex: function() {
            this.$active = this.$element.find(".item.active");
            this.$items = this.$active.parent().children();
            return this.$items.index(this.$active)
        },
        to: function(t) {
            var n = this.getActiveIndex(), r = this;
            if (t > this.$items.length - 1 || t < 0)
                return;
            return this.sliding ? this.$element.one("slid", function() {
                r.to(t)
            }) : n == t ? this.pause().cycle() : this.slide(t > n ? "next" : "prev", e(this.$items[t]))
        },
        pause: function(t) {
            t || (this.paused=!0);
            if (this.$element.find(".next, .prev").length && e.support.transition.end) {
                this.$element.trigger(e.support.transition.end);
                this.cycle(!0)
            }
            clearInterval(this.interval);
            this.interval = null;
            return this
        },
        next: function() {
            if (this.sliding)
                return;
            return this.slide("next")
        },
        prev: function() {
            if (this.sliding)
                return;
            return this.slide("prev")
        },
        slide: function(t, n) {
            var r = this.$element.find(".item.active"), i = n || r[t](), s = this.interval, o = t == "next" ? "left": "right", u = t == "next" ? "first": "last", a = this, f;
            this.sliding=!0;
            s && this.pause();
            i = i.length ? i : this.$element.find(".item")[u]();
            f = e.Event("slide", {
                relatedTarget: i[0],
                direction: o
            });
            if (i.hasClass("active"))
                return;
            if (this.$indicators.length) {
                this.$indicators.find(".active").removeClass("active");
                this.$element.one("slid", function() {
                    var t = e(a.$indicators.children()[a.getActiveIndex()]);
                    t && t.addClass("active")
                })
            }
            if (e.support.transition && this.$element.hasClass("slide")) {
                this.$element.trigger(f);
                if (f.isDefaultPrevented())
                    return;
                i.addClass(t);
                i[0].offsetWidth;
                r.addClass(o);
                i.addClass(o);
                this.$element.one(e.support.transition.end, function() {
                    i.removeClass([t, o].join(" ")).addClass("active");
                    r.removeClass(["active", o].join(" "));
                    a.sliding=!1;
                    setTimeout(function() {
                        a.$element.trigger("slid")
                    }, 0)
                })
            } else {
                this.$element.trigger(f);
                if (f.isDefaultPrevented())
                    return;
                r.removeClass("active");
                i.addClass("active");
                this.sliding=!1;
                this.$element.trigger("slid")
            }
            s && this.cycle();
            return this
        }
    };
    var n = e.fn.carousel;
    e.fn.carousel = function(n) {
        return this.each(function() {
            var r = e(this), i = r.data("carousel"), s = e.extend({}, e.fn.carousel.defaults, typeof n == "object" && n), o = typeof n == "string" ? n: s.slide;
            i || r.data("carousel", i = new t(this, s));
            typeof n == "number" ? i.to(n) : o ? i[o]() : s.interval && i.pause().cycle()
        })
    };
    e.fn.carousel.defaults = {
        interval: 5e3,
        pause: "hover"
    };
    e.fn.carousel.Constructor = t;
    e.fn.carousel.noConflict = function() {
        e.fn.carousel = n;
        return this
    };
    e(document).on("click.carousel.data-api", "[data-slide], [data-slide-to]", function(t) {
        var n = e(this), r, i = e(n.attr("data-target") || (r = n.attr("href")) && r.replace(/.*(?=#[^\s]+$)/, "")), s = e.extend({}, i.data(), n.data()), o;
        i.carousel(s);
        (o = n.attr("data-slide-to")) && i.data("carousel").pause().to(o).cycle();
        t.preventDefault()
    })
}(window.jQuery);
(function() {
    var e = {
        actionEventData: null,
        elementListener: null,
        init: function() {
            Pb.Data.add("actionEvents", []);
            this.actionEventData = Pb.Data.get("actionEvents");
            $(document).on("ActionEventsDispatcher::ondomready", $.proxy(this.onDomReady, this));
            $(document).on("ActionEventsDispatcher::registerevent", $.proxy(this.registerEvent, this));
            $(document).on("ActionEventsDispatcher::bindevents", $.proxy(this.bindEvents, this))
        },
        onDomReady: function() {
            this.actionEventData = Pb.Data.get("actionEvents");
            this.elementListener = $("body")
        },
        bindEvents: function() {
            for (var e = 0; e < this.actionEventData.length; e++) {
                var t = this.actionEventData[e];
                $(this.elementListener).on(t.eventType, t.eventSelector, $.proxy(function(e) {
                    e.preventDefault();
                    var t = this.findEvent($(e.currentTarget));
                    if (t) {
                        var n = Pb.Data.Shared.getData(e.currentTarget), r = $.Event(t, {
                            pbEventTarget: e.currentTarget,
                            pbData: n
                        });
                        $(document).trigger(r)
                    }
                }, this))
            }
        },
        findEvent: function(e) {
            for (var t = 0; t < this.actionEventData.length; t++) {
                var n = this.actionEventData[t], r = n.eventSelector.split(" "), i;
                $.isArray(r) ? i = r[r.length - 1] : i = n.eventSelector;
                if (i.indexOf(".")!==-1) {
                    var s = i.split(".")[1];
                    if (e.hasClass(s))
                        return n.event
                } else if (i.indexOf("#")!==-1) {
                    var o = i.split("#")[1];
                    if (e.attr("id") === o)
                        return n.event
                }
            }
            return null
        },
        registerEvent: function(e) {
            typeof e.pbEvent == "object" && this.actionEventData.push(e.pbEvent)
        }
    };
    e.init()
})();
Pb.InitEventQueue.addToDomReady(function() {
    $(document).trigger($.Event("ActionEventsDispatcher::ondomready"))
}, Pb.InitEventQueue.ACTION_EVENT_PRIORITY);
Pb.InitEventQueue.addToDomReady(function() {
    $(document).trigger($.Event("ActionEventsDispatcher::bindevents"))
}, Pb.InitEventQueue.LOW_PRIORITY + 1);
(function() {
    function e(e) {
        if (e.origin !== "http://b.photobucket.com")
            return;
        if (e.data) {
            var t = e.data.split("p~b"), n, r, i = {}, s, o;
            for (r = 0; r < t.length; r++) {
                s = t[r].split(/\=(.+)/);
                i[s[0]] = s[1]
            }
            if (i.type === "Banner" || i.type === "NavBar") {
                i.hasOwnProperty("color") && $(".bannerAd").css("background-color", i.color);
                if (i.hasOwnProperty("width") && i.hasOwnProperty("height")) {
                    o = $(".bannerAdIframe");
                    o.data("originalstyle") || o.data("originalstyle", o.attr("style"));
                    o.width(parseInt(i.width, 10) + "px").height(parseInt(i.height, 10) + "px")
                }
            } else if (i.type === "Skin") {
                $("html").addClass("sponsored");
                $(document.body).addClass("sponsored");
                if ($("div.homeContainer").length) {
                    $("div.homeContainer").removeClass("boxShadow");
                    $("div.contentWrapper").addClass("boxShadow")
                }
                $(document.body).css("background-image", "url('" + i.skin + "')").css("background-color", i.color);
                i.href && i.href !== "" ? $(document.body).click(function(e) {
                    e.originalEvent && (e.originalEvent.target.id === "content" || e.originalEvent.target.className === "basicContainer") && window.open(i.href)
                }) : $(document.body).css("cursor", "default");
                typeof console != "undefined" && console.log("triggering event");
                $(document).trigger("skinned.pbad");
                $(".sponsorSkinClick").attr("onclick", "window.open('" + i.href + "')")
            } else if (i.type === "Billboard") {
                $(".billboardAd").removeClass("hide");
                $(".bannerAd").hide()
            } else if (i.type === "300x600") {
                var u = i.pos === "inf" ? $(".mrecIframe").last(): $(".mrecIframe").first();
                u.css("height", "600px")
            }
            if ($(document.body).hasClass("sponsored")) {
                var a = $("div.navbar:first"), f = $("ad.bannerAd:first"), n = 0;
                if (a.length) {
                    n = a.height();
                    f.length && (n += f.height());
                    $(document.body).css("background-position", "center " + n + "px")
                }
            }
        }
    }
    window.addEventListener ? window.addEventListener("message", e, !1) : window.attachEvent && window.attachEvent("onmessage", e)
})();
$.getObject("Pb").Form = function() {
    function e() {
        var e = $(this), t = e.find("input[name=hash]");
        if (!t.length) {
            t = $('<input type="hidden" value="" name="hash" />');
            e.append(t)
        }
        t.val($("#token").val());
        return !0
    }
    function t(t) {
        t = $(t);
        t.on("submit", e)
    }
    return {
        addHash: t
    }
}();
$.getObject("Pb.Component.Element", !0).Message = function() {
    function u() {
        var e;
        $(".global.alert").alert();
        e = v($(".global.alert").data("alertclass"));
        $(".global.alert").delay(e).fadeOut("slow", function() {
            $(this).remove()
        })
    }
    function a(e) {
        var t = [], n;
        if ($.isArray(e)) {
            $.each(e, function(e, r) {
                n = d(r);
                t.push(n)
            });
            p(t)
        } else {
            n = d(e);
            p(n)
        }
    }
    function f() {
        a(s)
    }
    function l() {
        a(o)
    }
    function c(r, i) {
        var s;
        switch (i.status) {
        case e:
            s = "error";
            break;
        case t:
            s = "warning";
            break;
        case n:
            s = "warning";
            break;
        default:
            s = ""
        }
        $(r).closest(".control-group").addClass(s).find(".help-inline, .help-block").html(i.message + i.details)
    }
    function h(e) {
        $(e).closest(".control-group").removeClass("error warning").find(".help-inline, .help-block").html("")
    }
    function p(e) {
        Pb.JSTemplates.link("#alertTemplate", "#messageContainer", e);
        setTimeout(function() {
            $(".global.alert").alert()
        }, 200);
        e.displayTime >= 0 && $(".global.alert").delay(e.displayTime).fadeOut("slow", function() {
            $(this).remove()
        })
    }
    function d(e) {
        var n = "success", r = "thumbs-up", s = i, o;
        if (!e.status) {
            n = "error";
            r = "warning-sign"
        } else if (e.status === t) {
            n = "info";
            r = "flag";
            s = 15e3
        }
        e.displayTime ? s = e.displayTime : s = v(n);
        o = {
            messageClass: n,
            messageIcon: r,
            message: e.message,
            details: e.details,
            displayTime: s
        };
        return o
    }
    function v(e) {
        return e === "info" ? 15e3 : i
    }
    function m(e) {
        e || (e = $("form"));
        $(".help-inline, .help-block", e).html("");
        $(".error", e).removeClass("error");
        $(".global.alert").find(".close").click()
    }
    function g(e, t, n, r) {
        if (typeof JSON != "undefined") {
            var i = {
                status: e,
                message: t,
                details: n,
                memo: r
            }, s = JSON.stringify(i);
            Pb.Utilities.cookie("pb_message", s, {
                expires: 1
            })
        }
    }
    var e = 0, t = 1, n = 2, r = 3, i = 5e3, s = {
        status: e,
        message: "Hmmm. Something didn't click.  Want to give it another shot? Try again now.",
        details: ""
    }, o = {
        status: r,
        message: "Success! That click paid off.",
        details: ""
    };
    return {
        globalAlert: u,
        show: a,
        showDefaultErrorMessage: f,
        showDefaultSuccessMessage: l,
        showInlineMessage: c,
        removeInlineMessage: h,
        addMessageToCookie: g,
        remove: m,
        STATUS_ERROR: e,
        STATUS_NOTICE: t,
        STATUS_WARNING: n,
        STATUS_SUCCESS: r,
        defaultErrorMessage: s
    }
}();
$.getObject("Pb.Modal", !0).ModalActionBase = function() {
    function e() {}
    var t = {
        eventName: null,
        modalId: null,
        modalName: null,
        submitButtonId: null,
        submitMethod: "POST",
        submitUrl: null,
        hideOnFinish: !0,
        shownEvent: null,
        hideEvent: null,
        modalData: [],
        showProgress: !1,
        messageHandler: null,
        refreshAdsOnHide: !1,
        embed: !1,
        init: function(e) {
            this.eventName = e.eventName;
            this.modalId = e.modalId;
            this.modalName = e.modalName;
            this.submitButtonId = e.submitButtonId;
            this.submitUrl = e.submitUrl;
            this.shownEvent = e.shownEvent;
            this.hideEvent = e.hideEvent;
            this.embedId = "#" + this.modalName + "Embed";
            e.hideOnFinish && (this.hideOnFinish = e.hideOnFinish);
            e.submitMethod && (this.submitMethod = e.submitMethod);
            e.showProgress && (this.showProgress = e.showProgress);
            e.messageHandler && (this.messageHandler = e.messageHandler);
            e.refreshAdsOnHide && (this.refreshAdsOnHide = e.refreshAdsOnHide);
            if (this.modalName !== null) {
                this.eventName && $(document).on(this.eventName, $.proxy(this.onShow, this));
                Pb.InitEventQueue.add($.proxy(this.bindModalEvents, this), Pb.InitEventQueue.LOW_PRIORITY)
            }
        },
        bindModalEvents: function() {
            var e = $(this.modalId), t = $(this.embedId);
            if (e.length || t.length) {
                this.submitButtonId && $(this.submitButtonId).closest("form").on("submit", $.proxy(this.onSubmit, this));
                $(this.modalId).on("hide", $.proxy(this.onHide, this))
            }
        },
        buildSubmitData: function() {
            return {}
        },
        onSubmit: function(e) {
            e.preventDefault();
            if ($(e.currentTarget).hasClass("disabled"))
                return;
            this.submit();
            var t = $(this.modalId), n = ($(window).width() - t.width()) / 2;
            t.css("left", Math.floor(n) + "px")
        },
        submit: function() {
            var e = this.buildSubmitData(), t = this.modalId;
            if (e) {
                var n = {
                    type: this.submitMethod,
                    url: this.submitUrl,
                    data: e,
                    success: $.proxy(this.onResponse, this),
                    error: $.proxy(this.onAjaxError, this)
                };
                this.messageHandler && (n.messageHandler = this.messageHandler);
                Pb.Ajax.ajax(n)
            }
            if (this.showProgress) {
                this.embed && (t = this.embedId);
                Pb.Modal.ModalProgressBar.addProgressIndicator(t)
            }
        },
        handleResponse: function(e) {
            var t = this.modalId;
            this.embed && (t = this.embedId);
            if (e.status) {
                this.onSuccess(e);
                if (this.hideOnFinish) {
                    $(this.modalId).modal("hide");
                    this.embed && Pb.Modal.ModalProgressBar.removeProgressIndicator(t)
                } else 
                    Pb.Modal.ModalProgressBar.removeProgressIndicator(t)
            } else {
                Pb.Modal.ModalProgressBar.removeProgressIndicator(t);
                this.onFail(e)
            }
        },
        onResponse: function(e) {
            var t = this.modalId;
            this.embed && (t = this.embedId);
            if (this.showProgress) {
                Pb.Modal.ModalProgressBar.updateProgressIndicator(t, 100);
                setTimeout($.proxy(function() {
                    this.handleResponse(e)
                }, this), 200)
            } else 
                this.handleResponse(e)
        },
        onSuccess: function(e) {},
        onFail: function(e) {},
        onAjaxError: function() {
            var e = this.modalId;
            this.embed && (e = this.embedId);
            Pb.Modal.ModalProgressBar.removeProgressIndicator(e);
            this.onFail()
        },
        onShow: function(e) {
            var t = this.modalId;
            this.embed && (t = this.embedId);
            Pb.Modal.ModalProgressBar.removeProgressIndicator(t);
            if (!this.embed) {
                var n = $(this.modalId);
                n.modal("show")
            } else 
                $(this.embedId).show();
            if (this.shownEvent) {
                var r = {
                    type: this.shownEvent
                }, i;
                for (i in this.modalData)
                    this.modalData.hasOwnProperty(i) && (r[i] = this.modalData[i]);
                $(document).trigger(r)
            }
        },
        onHide: function() {
            this.refreshAdsOnHide && Pb.Component.Ad !== undefined && Pb.Component.Ad.AdRefresh.reloadAds();
            if (this.hideEvent) {
                var e = $.Event(this.hideEvent, this.modalData);
                $(document).trigger(e)
            }
        }
    };
    $.extend(e.prototype, t);
    return e
}();
$.getObject("Pb.Modal").ModalProgressBar = function() {
    function e(e) {
        if (!$(e).hasClass("inProgressModal")) {
            $(e).addClass("inProgressModal");
            $(e).find(".modal-body").length ? $(e).find(".modal-body").after('<div class="progress progress-info progress-striped active"><div class="bar" style="width: 40%;"></div></div>') : $(e).prepend('<div class="progress progress-info progress-striped active"><div class="bar" style="width: 40%;"></div></div>')
        }
    }
    function t(e, t) {
        var n = $(e);
        n.hasClass("inProgressModal") && n.find(".progress .bar").css("width", t + "%")
    }
    function n(e) {
        var t = $(e);
        t.hasClass("inProgressModal") && t.removeClass("inProgressModal").find(".progress").remove()
    }
    return {
        addProgressIndicator: e,
        updateProgressIndicator: t,
        removeProgressIndicator: n
    }
}();
$.getObject("Pb.Component.Element", !0).Modal = function() {
    function e() {
        e.superclass.constructor.call(this)
    }
    var t = {
        templateLinked: !1,
        modalTemplateData: [],
        prepareDataForTemplate: function(e) {
            return e
        },
        linkTemplate: function(e) {
            var t = Pb.Data.get(this.modalName), n, r, i, s = t, o = "#modalTemplate";
            s ? s.embed=!1 : s = {
                embed: !1
            };
            e && (s = $.extend(t, e));
            s = this.prepareDataForTemplate(s);
            r = this.modalName + "Wrapper";
            i = this.modalName + "Embed";
            if (!this.templateLinked) {
                if (!this.embed) {
                    this.modalTemplateData = [s];
                    this.modalTemplateData[0].embed=!1;
                    n = $("<div />");
                    n.attr("id", r);
                    $(document.body).append(n);
                    Pb.JSTemplates.link(o, "#" + r, this.modalTemplateData);
                    this.templateLinked=!0;
                    this.onTemplateLinked()
                }
            } else 
                $.observable(this.modalTemplateData).refresh(s);
            if (this.embed) {
                this.modalTemplateData = [s];
                this.modalTemplateData[0].embed=!0;
                o = s.templateName;
                Pb.JSTemplates.link(o, "#" + i, this.modalTemplateData);
                this.onTemplateLinked()
            }
        },
        updateHeader: function(e) {
            $(this.modalId).find(".modal-header h2").html(e)
        },
        onShow: function(t) {
            t.embed ? this.embed = t.embed : this.embed=!1;
            this.linkTemplate(t.modalData);
            t.headerTitle && this.updateHeader(t.headerTitle);
            this.bindModalEvents();
            e.superclass.onShow.call(this, t)
        },
        onTemplateLinked: function() {}
    };
    Pb.subclass(e, Pb.Modal.ModalActionBase, t);
    return e
}();
$.getObject("Pb.RulesEngine", !0).PromptUpdaterObject = function() {
    function n() {
        $(document).on("Pb.RulesEngine.PromptUpdater::onShown", $.proxy(this.onShown, this))
    }
    var e = "prompt", t = "/action/rulesEngine/update", r = {
        init: function(e) {
            this.name = e
        },
        onShown: function(n) {
            var r = {
                name: n.name,
                type: e
            }, i = {
                url: t,
                type: "POST",
                data: r,
                error: function() {}
            };
            n.success && $.isFunction(n.success) && (i.success = n.success);
            Pb.Ajax.ajax(i)
        }
    };
    $.extend(n.prototype, r);
    return n
}();
$.getObject("Pb", !0).PromptUpdater = new Pb.RulesEngine.PromptUpdaterObject;
$.getObject("Pb", !0).Errors = {
    ACCOUNT_OVER_SIZE_LIMIT: 193
};
(function() {
    function e(e, t) {
        return t.tabs[e].id === t.activeTab?!0 : !1
    }
    function t() {
        return e(this.index, this.parent.parent.data) ? "active" : ""
    }
    function n() {
        return this.index === 0?!1 : e(this.index, this.parent.parent.data)?!1 : e(this.index - 1, this.parent.parent.data)?!1 : !0
    }
    function r() {
        $.views.helpers({
            getTabClass: t,
            showTabDivider: n
        })
    }
    Pb.InitEventQueue.add(r)
})();
$.getObject("Pb", !0).InputValidator = function() {
    function e(e) {
        this.init(e)
    }
    $.extend(e.prototype, {
        regex: {
            username: /^[a-zA-Z0-9_][-a-zA-Z0-9_]+$/,
            reg_username: /^[\w]+$/,
            password: /^[a-zA-Z1-9_-][a-zA-Z0-9_-]+$/,
            postalcode: /^[\w|-]{5,11}$/,
            email: /^((([a-z]|\d|[!#\$%&'\*\+\-\/=\?\^_`{\|}~]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])+(\.([a-z]|\d|[!#\$%&'\*\+\-\/=\?\^_`{\|}~]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])+)*)|((\x22)((((\x20|\x09)*(\x0d\x0a))?(\x20|\x09)+)?(([\x01-\x08\x0b\x0c\x0e-\x1f\x7f]|\x21|[\x23-\x5b]|[\x5d-\x7e]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(\\([\x01-\x09\x0b\x0c\x0d-\x7f]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]))))*(((\x20|\x09)*(\x0d\x0a))?(\x20|\x09)+)?(\x22)))@((([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|\d|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.)+(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])|(([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])([a-z]|\d|-|\.|_|~|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])*([a-z]|[\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF])))\.?$/i,
            hashtag: /^[a-zA-Z1-9][a-zA-Z0-9]+$/
        },
        messages: {
            required: "",
            email: "Please enter a valid email address.",
            password: "Password must contain at least 6 characters: a-z, A-Z, 0-9, - and _ only.  No spaces, please.",
            username: "Username must be 3-50 characters long. It can contain a-z, A-Z, 0-9, -, _ and must begin with either a letter, number or underscore. No spaces, please.",
            passwordMatch: "Passwords must match.",
            postalcode: "Zip codes must be at least 5 characters long. They can only contain the letters a-z, A-Z, 0-9, and -. No spaces, please.",
            whitespace: "Photobucket usernames do not have spaces in them, please enter another username with no spaces.",
            hashtag: "Hashtags must be 4-50 characters long.  It can contain a-z, A-Z, 0-9 and must begin with either a letter, or a number.  No spaces, please."
        },
        init: function(e) {},
        validateRequired: function(e, t, n) {
            $.isFunction(t) || (t = function() {
                return !1
            });
            $.each(e, $.proxy(function(e, r) {
                this.validRequiredField(r) || t(r, this, n)
            }, this))
        },
        validRequiredField: function(e) {
            if (e === undefined)
                return !1;
            if (e.attr("type") === "radio" || e.attr("type") === "checkbox") {
                if (e.filter(":checked").length === 0)
                    return !1
            } else if (e.val().length === 0)
                return !1;
            return !0
        },
        validUsername: function(e) {
            return e.length === 0?!1 : e.length < 3 || e.length > 50 ||!this.regex.username.test(e)?!1 : !0
        },
        validRegUsername: function(e) {
            return e.length === 0?!1 : e.length < 3 || e.length > 50 ||!this.regex.reg_username.test(e)?!1 : !0
        },
        validPassword: function(e) {
            return e.length === 0?!1 : e.length < 6 || e.length > 50 ||!this.regex.password.test(e)?!1 : !0
        },
        validEmail: function(e) {
            return e.length === 0?!1 : this.regex.email.test(e)?!0 : !1
        },
        stringEquals: function(e, t) {
            return e === t?!0 : !1
        },
        validPostalCode: function(e) {
            return e.length === 0?!1 : this.regex.postalcode.test(e)?!0 : !1
        },
        validHashtag: function(e) {
            return e.length === 0?!1 : e.length < 4 || e.length > 50 ||!this.regex.hashtag.test(e)?!1 : !0
        }
    });
    return e
}();
$.getObject("Pb.Component.Common.InsetBox", !0).InsetBox = function() {
    function e() {
        this.init()
    }
    var t = {
        init: function() {
            $(document).on("click", ".insetBoxAction", $.proxy(this.toggleShow, this))
        },
        toggleShow: function(e) {
            var t = $(e.currentTarget).data("action"), n = $(e.currentTarget).data("target"), r = $(n).find(".contentText"), i = r.data("height");
            t === "expand" && $(n).addClass("expanded");
            r.animate({
                height: i
            }, 500, function() {
                t !== "expand" && $(n).removeClass("expanded")
            })
        }
    };
    $.extend(e.prototype, t);
    return e
}();
Pb.InitEventQueue.addToDomReady(function() {
    var e = new Pb.Component.Common.InsetBox.InsetBox
});
(function() {
    function e() {
        if (Pb.Data.get("trackUnsupportedBrowser") && $("html").hasClass("is_ie")) {
            var e = $(document.body).attr("id").replace(/page$/, "");
            if ($("html").hasClass("is_ie7") || $("html").hasClass("is_ie6") || $("html").hasClass("is_ie5")) {
                Pb.Track.trn(e, (Pb.Data.get("isLoggedIn") ? "loggedin" : "loggedout") + " ie user pageview");
                Pb.Data.get("showUnsupportedBrowserMessage") && Pb.Component.Element.Message.show({
                    status: Pb.Component.Element.Message.STATUS_NOTICE,
                    message: 'Photobucket does not support this version of Internet Explorer. Please upgrade your browser to a newer version of Internet Explorer, or switch to another browser (i.e. Chrome, Firefox, etc.). For more information, please <a href="http://photobucket.zendesk.com/entries/21105791-supported-browsers-and-operating-systems" target="_blank">click here.</a>',
                    displayTime: 1e4
                })
            } else if (document.documentMode && document.documentMode <= 7) {
                Pb.Track.trn(e, (Pb.Data.get("isLoggedIn") ? "loggedin" : "loggedout") + " ie" + document.documentMode + "compatmodeuser pageview");
                Pb.Data.get("showUnsupportedBrowserMessage") && Pb.Component.Element.Message.show({
                    status: Pb.Component.Element.Message.STATUS_NOTICE,
                    message: 'Your Internet Explorer browser is running in compatibility mode which will cause display problems on the site. To learn how to turn this mode off, please <a href="http://photobucket.zendesk.com/entries/21886602-ie-8-9-in-compatibility-mode" target="_blank">click here.</a>',
                    displayTime: 1e4
                })
            }
        }
    }
    Pb.InitEventQueue.add(e)
})();
$.getObject("Pb.Component", !0).Debug = function() {
    function e(e) {
        this.init(e)
    }
    var t = {
        data: [],
        startTime: 0,
        totalTime: 0,
        linked: !1,
        init: function(e) {
            this.add(e)
        },
        link: function() {
            Pb.JSTemplates.link("#stopwatchesTemplate", "#stopwatches", this.data);
            Pb.JSTemplates.link("#debugMessagesTemplate", "#debug-messages", this.data);
            var e = this;
            $.views.helpers({
                truncate: function(e) {
                    return e == null ? "" : e.substr(0, 27) + (e.length > 27 ? "..." : "")
                },
                barLeft: function(t) {
                    return Math.round((t.start - e.startTime) / e.totalTime * 1024)
                },
                barWidth: function(t) {
                    return t.marker ? 10 : Math.max(Math.round(t.elapsed / e.totalTime * 1024), 1)
                },
                marker: function(t) {
                    return (t.start - e.startTime).toFixed(3)
                },
                round: function(e) {
                    return e.toFixed(3)
                },
                indent: function(e) {
                    var t = e.level;
                    return (new Array(isNaN(t) ? 1 : ++t)).join("&nbsp;")
                },
                messageStyle: function(e) {
                    return e.indexOf("ERR")!==-1 || e.indexOf("ERROR")!==-1 ? "error" : e.indexOf("WARN")!==-1 ? "warn" : ""
                }
            });
            this.linked=!0
        },
        add: function(e) {
            this.linked || this.link();
            this.startTime = e.stopwatch.start;
            this.totalTime = e.stopwatch.elapsed;
            $.observable(this.data).insert(this.data.length, e)
        }
    };
    e.prototype = t;
    return e
}();
$.getObject("Pb.Component.Element", !0).Lightbox = function() {
    function r(t) {
        $.extend(e, t);
        n = $("#" + e.lightboxId);
        e.data !== undefined && $.each(e.data, function(e, t) {
            n.data(e, t)
        });
        n.height(e.initialHeight).css("position", e.position).css("max-height", e.maxHeight);
        i();
        return this
    }
    function i() {
        n.on("show", s)
    }
    function s(t) {
        if (e.contentUrl.length > 0) {
            u();
            e.cache || n.find(".modal-body").empty();
            var r = {
                url: e.contentUrl,
                data: e.ajaxData,
                cache: !e.cache,
                success: o
            };
            $.ajax(r)
        }
    }
    function o(e) {
        n.find(".modal-body").html(e);
        f()
    }
    function u() {
        n.find(".progress").show()
    }
    function a(e) {
        n.find(".progress .bar").css("width", e)
    }
    function f() {
        n.find(".progress").hide()
    }
    function l(e, r, i) {
        var s = n.find(".alert");
        s.attr("class", "alert");
        s.addClass(t[e]);
        s.find(".alert-heading").text(r);
        s.find(".alert-body").text(i);
        s.alert()
    }
    function c() {
        var e = n.find(".alert");
        e.hide()
    }
    var e = {
        lightboxId: "pb-lightbox",
        cache: !0,
        contentUrl: "",
        ajaxData: {},
        initialHeight: "auto",
        position: "fixed",
        maxHeight: 500
    }, t = {
        error: "alert-error",
        warning: "alert-block",
        success: "alert-success",
        info: "alert-info"
    }, n = null;
    return {
        init: r,
        showProgress: u,
        updateProgress: a,
        hideProgress: f,
        alertTypes: t,
        showAlertMessage: l,
        hideAlertMessage: c
    }
}();
$.getObject("Pb", !0).TrackObject = function() {
    function e() {
        this.init()
    }
    var t = {
        trackUrl: "/action/track/log",
        trackEvent: "Pb::track",
        trackObj: {},
        init: function() {
            Pb.Data.add("trackEvent", this.trackEvent);
            $(document).on(this.trackEvent, $.proxy(function(e) {
                this.trn(e.pbPage, this.sanitizeTrackString(e.pbTrackString))
            }, this))
        },
        tr: function(e) {
            if (typeof e != "string")
                return !0;
            var t = e.split(","), n, r;
            for (n = 0; n < t.length; n++) {
                e = this.sanitizeTrackString(t[n]);
                r = this.trackObj[e];
                r = typeof r == "number" ? r + 1 : 1;
                this.trackObj[e] = r;
                Pb.DataCookie.set("pbtr", this.trackObj)
            }
            return !0
        },
        trl: function(e) {
            if (typeof e != "string")
                return;
            var t = e.split(","), n, r, i;
            for (n = 0; n < t.length; n++) {
                e = this.sanitizeTrackString(t[n]);
                i = this.trackObj.legacy;
                typeof i == "undefined" && (i = this.trackObj.legacy = {});
                r = i[e];
                r = typeof r == "number" ? r + 1 : 1;
                i[e] = r;
                Pb.DataCookie.set("pbtr", this.trackObj)
            }
        },
        trn: function(e, t) {
            var n = {
                page: e,
                key: this.sanitizeTrackString(t)
            };
            this.doTrackNow(n)
        },
        trw: function(e, t) {
            var n = {
                key: this.sanitizeTrackString(e)
            };
            return this.doTrackNow(n, t)
        },
        trf: function(e) {
            var t = Pb.Data.get("featureTrackingKey"), n = e.split(","), r;
            this.tr(e);
            for (r = 0; r < n.length; r++)
                t && this.tr(n[r] + "_" + t)
        },
        googleEvent: function(e, t, n, r) {
            typeof _gaq != "undefined" && _gaq.push($.merge(["_trackEvent"], [e, t, n, r, !0]))
        },
        sanitizeTrackString: function(e) {
            e = e.toLowerCase().substring(0, 128).replace(/[ -]/g, "_");
            e = e.replace(/['"\/|\\!\?\.,]/g, "");
            e = e.replace(/[&]/g, "and");
            return e
        },
        doTrackNow: function(e, t) {
            var n = {
                url: this.trackUrl,
                data: e,
                success: function(e) {
                    t && t(e)
                },
                error: function(e) {
                    t && t(e)
                },
                messageHandler: function() {
                    return !0
                }
            };
            t && (n.async=!1);
            return Pb.Ajax.ajax(n)
        }
    };
    e.prototype = t;
    return e
}();
$.getObject("Pb", !0).Track = new Pb.TrackObject;
$.getObject("Pb", !0).PageTracker = function() {
    function e(e) {
        var n = e.pb_pg, r = e.val ? e.val: null;
        if (typeof _gaq != "undefined") {
            _gaq.push(["_trackPageview"]);
            t(n, r)
        }
    }
    function t(e, t) {
        if (typeof _gaq != "undefined") {
            Pb.Data.get("isLoggedIn") ? e = "loggedin:" + e : e = "loggedout:" + e;
            Pb.Track.googleEvent("pbtng", "page view", e, t)
        }
    }
    function n(e, t, n, i, s) {
        Pb.Data.get("isLoggedIn") ? i = "loggedin:" + i : i = "loggedout:" + i;
        if (typeof _gaq != "undefined") {
            r(e, t);
            _gaq.push(["_trackPageview", n]);
            _gaq.push($.merge(["_trackEvent"], [s, "page view", i, null, !0]))
        }
    }
    function r(e, t) {
        var n, r;
        _gaq.push(function() {
            r = _gat._getTrackerByName();
            n = r._getAccount()
        });
        if (n !== e) {
            _gaq.push(["_setAccount", e]);
            _gaq.push(["_setDomainName", t])
        }
    }
    function i(e, t, n, i, s, o) {
        if (typeof _gaq != "undefined") {
            r(e, t);
            _gaq.push($.merge(["_trackEvent"], [n, i, s, o, !0]))
        }
    }
    function s() {
        var e = new Image(1, 1), t;
        e.onerror = e.onload = function() {
            e.onerror = e.onload = null
        };
        t = ["//secure-us.imrworldwide.com/cgi-bin/m?ci=us-904896h&cg=0&cc=1&si=", escape(location.href), "&rp=", escape(document.referrer), "&ts=compact&c0=usergen,1&rnd=", (new Date).getTime()].join("");
        e.src = t
    }
    function o(e) {
        Pb.Component.Tracking.Comscore.trackPage(e)
    }
    function u() {
        Pb.Data.get("isLoggedIn") && Pb.Track.tr("logged_in_page");
        Pb.Track.tr("page_view");
        Pb.Track.trl("page_view")
    }
    function a(t) {
        o(t);
        e(t);
        s();
        u()
    }
    function f(t) {
        o(t);
        e(t.pb_pg);
        s();
        u()
    }
    return {
        trackPageView: a,
        trackContentRefreshPageView: f,
        trackGoogleAnalyticsEventPageView: t,
        addGoogleAnalyticsPartnerPageView: n,
        addGoogleAnalyticsPartnerEvent: i
    }
}();
$.getObject("Pb.Component.Tracking", !0).Comscore = function() {
    function e(e) {
        typeof COMSCORE != "undefined" && COMSCORE.beacon({
            c1: 2,
            c2: 16955838,
            c3: "",
            c4: location.href,
            c5: "",
            c6: "",
            c15: ""
        })
    }
    return {
        trackPage: e
    }
}();
$.getObject("Pb.Component.Ad", !0).AdDebugger = function() {
    function t() {
        $(".ad[data-debug]").each(function(e, t) {
            params = $(t).attr("data-debug");
            params = params.split(",");
            $(t).css("position") !== "fixed" && $(t).css("position", "relative");
            content = params.join("</div><div>");
            $(t).append('<div class="adDebug"><div>' + content + "</div></div>")
        });
        typeof _comscore != "undefined" && e.push("comscore");
        typeof LOTCC != "undefined" && e.push("lotame");
        if (e.length > 0) {
            $("#footer").css("position", "relative");
            content = e.join("</div><div>");
            $("#footer").append('<div class="adDebug"><div>Beacons:</div><div>' + content + "</div></div>")
        }
    }
    var e = [];
    return {
        display: t
    }
}();
$.getObject("Pb.Component.Ad", !0).AdRefresh = function() {
    function m(t) {
        var n = (new Date).getTime();
        a++;
        typeof t == "object" && (t = undefined);
        if (a%3 === 0 || n - c > v) {
            a = 0;
            c = n;
            E();
            S(s, s, t);
            S(i, i, t);
            S(e, e, t);
            S(o, o, t);
            S(u, u, t)
        }
    }
    function g(t) {
        var n = (new Date).getTime();
        typeof t == "object" && (t = undefined);
        if (n - c > v) {
            a = 0;
            c = n;
            E();
            S(s, s, t);
            S(i, i, t);
            S(e, e, t);
            S(o, o, t);
            S(u, u, t)
        }
    }
    function y() {
        function e() {
            if (d || w()) {
                h++;
                if (h >= 10) {
                    h = 0;
                    m()
                }
            }
        }
        $(window).on("mousemove click mouseup mousedown keydown keypress keyup submit change mouseenter scroll resize dblclick", function() {
            h = 0
        });
        $(window).on("blur", function() {
            d=!1
        });
        $(window).on("focus", function() {
            d=!0
        });
        p = setInterval(e, 1e3)
    }
    function b() {
        clearInterval(p)
    }
    function w() {
        return typeof document.hasFocus == "undefined" ? document.visibilityState === "visible" : document.hasFocus()
    }
    function E() {
        var e = $(".ad.lazy");
        e.find(">iframe").attr("src", "");
        e.attr("data-loaded", !1)
    }
    function S(e, t, n, r) {
        e === "pixlrSkin" ? x(e, t, n, r) : T(e, t, n)
    }
    function x(e, t, n, r) {
        var i = $("." + e), s = $(i).attr("data-debug");
        s = $.parseJSON(s);
        var o = {};
        if (typeof s == "undefined" || s === null)
            return;
        s && $.map(s, function(e, t) {
            var n = e.split("=");
            return n[0] === "area" ? o.area = n[1] : n[0] === "username" ? o.username = n[1] : n[0] === "ownername" ? o.ownername = n[1] : n[0] === "search_kw" ? o.search_kw = n[1] : n[0] === "ptype" ? o.ptype = n[1] : null
        });
        if (r)
            for (var u in r)
                r.hasOwnProperty(u) && (o[u] = r[u]);
        var a = "adType=" + e + "&adArea=" + o.area + "&username=" + o.username + "&ownername=" + o.ownername + "&searchKeyword=" + o.search_kw + "&ptype=" + o.ptype;
        t !== undefined && (a += "&target=" + t);
        arguments.length === 3 && n !== undefined && (a += "&" + n);
        var f = "/ad/?" + a, l;
        if (e === "pixlrSkin") {
            l = $("<iframe/>");
            l.addClass("adRefreshFrame");
            l.addClass(e + "Frame");
            l.addClass("hide");
            l.attr("id", t + "Frame");
            l.attr("src", f);
            $("body").append(l)
        } else {
            l = $("." + t).find("iframe:first");
            l.attr("src", f)
        }
    }
    function T(e, t, n) {
        var r = $("." + e + ":not(." + e + ".lazy)"), i = "";
        arguments.length === 3 && n !== undefined && (i = n);
        switch (t) {
        case"bannerAd2":
        case"mrec2":
            N(r, t, i);
            break;
        case"bannerAd":
            r.find(".bannerAdIframe").height("90").width("728px");
            C(t, i);
            break;
        default:
            C(t, i)
        }
    }
    function N(e, t, n) {
        var r, i, s = e.find("iframe");
        if (s.length === 0)
            return;
        r = s.attr("src");
        t === "bannerAd2" ? r = r.replace(/size%3D([A-Za-z0-9_%\+]+?)%2F/g, "size%3DBANNER%2F") : t === "mrec2" && (r = r.replace(/size%3D([A-Za-z0-9_%\+]+?)%2F/g, "size%3DRECTANGLE%2F"));
        i = L(r, n, s.data("type"), s.data("width"));
        var o = $("<iframe/>");
        o.attr("frameborder", "no");
        o.attr("scrolling", "no");
        o.attr("src", i);
        o.attr("marginheight", 0);
        o.attr("marginwidth", 0);
        s.data("originalstyle") ? o.attr("style", s.data("originalstyle")) : o.attr("style", s.attr("style"));
        s.data("type") && o.data("type", s.data("type"));
        s.data("width") && o.data("width", s.data("width"));
        $("." + t).empty();
        $("." + t).attr("data-debug", $(e).attr("data-debug"));
        $("." + t).append(o)
    }
    function C(e, t) {
        var n, r, s = $("." + e + ":not(.mrec2):not(.bannerAd2):not(.lazy)").find("iframe"), o = s.clone();
        if (s.length) {
            n = s.attr("src");
            r = L(n, t, s.data("type"), s.data("width"));
            e === i && o.attr("style", "width: 300px; height: 250px;");
            o.attr("src", r);
            var u = $("." + e).attr("data-debug");
            $("." + e).attr("data-debug", k(u, t));
            s.replaceWith(o)
        }
    }
    function k(e, t) {
        if (e) {
            e = $.parseJSON(e);
            var n = t.split("="), r=!1, i = 0;
            while (!r && i < e.length) {
                var s = e[i].split("=");
                if (s[0] === n[0]) {
                    r=!0;
                    e[i] = t
                } else 
                    s[0] === "likes" && (e[i] = "likes=n");
                i++
            }
        }
        var o = JSON.stringify(e);
        return JSON.stringify(e)
    }
    function L(e, t, n, r) {
        var i, s = (new Date).getTime();
        s - l > 1e3 && (f = Math.floor(Math.random() * 9999888888 + 111111));
        l = s;
        i = Math.floor(Math.random() * 999999 + 1);
        e = e.replace(/viewid%3D(\d+?)%2F/g, "viewid%3D" + f + "%2F");
        e = e.replace(/random%3D(\d+?)%2F/g, "random%3D" + i + "%2F");
        e = e.replace(/viewid=(\d+?)/g, "viewid=" + f + "/");
        e = e.replace(/random=(\d+?)/g, "random=" + i + "/");
        if (t)
            if (t.indexOf("pos=")!==-1) {
                var o = t.replace("pos=", "");
                e = e.replace(/pos%3D([A-Za-z0-9_%\+]+?)%2F/g, "pos%3D" + o + "%2F");
                o === "inf" && (e = A(e))
            } else if (t.indexOf("likes=")!==-1) {
                var u = t.replace("likes=", "");
                e = e.replace(/likes%3D([A-Za-z0-9_%\+]+?)%2F/g, "likes%3D" + u + "%2F")
            } else if (t.indexOf("skinid=")!==-1)
                e = e.replace(/skinid=([A-Za-z0-9_%\+]+?)\//g, t + "/");
            else if (t.indexOf("area=")!==-1) {
                var a = t.replace("area=", "");
                e = e.replace(/area%3D([A-Za-z0-9_%\+]+?)%2F/g, "area%3D" + escape(a) + "%2F");
                e = e.replace(/inv_code=([A-Za-z0-9_%\+]+?)&/g, "inv_code=" + a + "&")
            }
        if (typeof GlamIsAvailable == "function") {
            var c, h, p, d;
            c = GlamIsAvailable(n) ? "1" : "";
            var v = GlamGetSlotObject(n);
            if (v) {
                h = v.adid ? v.adid : "";
                p = v.sz ? v.sz : "";
                d = v.reqid ? v.reqid : ""
            } else {
                h = "";
                p = "";
                d = ""
            }
            e = e.replace(/(\/|%2F)glam[0-9]+(=|%3D)[^\/]/g, escape("/glam" + r + "=" + c));
            e = e.replace(/(\/|%2F)gadadid(=|%3D)[^\/]*/g, escape("/gadadid=" + h));
            e = e.replace(/(\/|%2F)gadsz(=|%3D)[^\/]*/g, escape("/gadsz=" + p));
            e = e.replace(/(\/|%2F)gadreqid(=|%3D)[^\/]*/g, escape("/gadreqid=" + d))
        }
        e = A(e);
        return e
    }
    function A(e) {
        if (typeof amznads != "undefined") {
            var t = ["a300x250p1", "a300x250p2", "a300x250p3", "a300x250p4", "a300x250p5", "a300x250p6", "a300x250p7", "a300x250p8", "a300x250p9", "a300x250p10", "a728x90p1", "a728x90p2", "a728x90p3", "a728x90p4", "a728x90p5", "a728x90p6", "a728x90p7", "a728x90p8", "a728x90p9", "a728x90p10"];
            for (var n = 0; n < t.length; n++) {
                var r = new RegExp("(/|%2F)" + t[n] + "(=|%3D)[^/]*", "g");
                e = e.replace(r, "")
            }
        }
        return e
    }
    function O(e, t) {
        try {
            var n = $("body").contents().find("." + e), r = $("." + t, parent.window.document);
            $("#content", parent.window.document).css("background", "inherit");
            $(r).html($(n).contents());
            $(r).attr("data-debug", $(n).attr("data-debug"));
            $(parent.window.document).find("#" + t + "Frame").remove()
        } catch (i) {
            $(parent.window.document).find("#" + t + "Frame").remove()
        }
    }
    function M(e) {
        var t = $("body").contents().find(".ad.mrec").first(), n = $(t).find(">iframe").first().attr("src");
        e = e || 2;
        var r, i = (new Date).getTime();
        i - l > 1e3 && (f = Math.floor(Math.random() * 9999888888 + 111111));
        l = i;
        r = Math.floor(Math.random() * 999999 + 1);
        n = n.replace(/viewid%3D(\d+?)%2F/g, "viewid%3D" + f + "%2F");
        n = n.replace(/random%3D(\d+?)%2F/g, "random%3D" + r + "%2F");
        n = n.replace(/viewid=(\d+?)/g, "viewid=" + f + "/");
        n = n.replace(/random=(\d+?)/g, "random=" + r + "/");
        n = n.replace(/pos%3D[\w+?]+/g, "pos%3Dinf");
        n = n.replace(/pos=(\w+?)/g, "pos=inf");
        n = A(n);
        var s = $('<iframe src="' + n + '" style="width: 300px; height: 250px" class="mrecIframe" id="mrec' + e + '" frameborder="0" scrolling="no" data-type="RECTANGLE" data-width="300" marginwidth="0" marginheight="0"></iframe>');
        return s
    }
    function _(e, t) {
        var n = $("body").contents().find(".src" + e).first(), r = $(n).text(), i, s, o = (new Date).getTime(), u = "";
        t = t || 2;
        o - l > 1e3 && (f = Math.floor(Math.random() * 9999888888 + 111111));
        l = o;
        i = Math.floor(Math.random() * 999999 + 1);
        r = r.replace(/viewid%3D(\d+?)%2F/g, "viewid%3D" + f + "%2F");
        r = r.replace(/random%3D(\d+?)%2F/g, "random%3D" + i + "%2F");
        r = r.replace(/viewid=(\d+?)/g, "viewid=" + f + "/");
        r = r.replace(/random=(\d+?)/g, "random=" + i + "/");
        if (e === "mrec" && t !== 1) {
            r = r.replace(/pos%3D[\w+?]+/g, "pos%3Dinf");
            r = r.replace(/pos=(\w+?)/g, "pos=inf")
        }
        s = 150;
        e === "mrec" && (s = 250);
        u = $('<iframe style="width: 300px; height: ' + s + 'px" class="' + e + 'Iframe" id="' + e + t + '" frameborder="0" scrolling="no" marginwidth="0" marginheight="0"></iframe>');
        if (!$(".mrecIframe").length) {
            $("#" + e + t).parent().attr("data-loaded", "yes");
            $("#" + e + t).attr("src", r)
        }
        $(window).on("scroll", function(n) {
            var i = $("#" + e + t).parent(), s = $("#" + e + t), o = $(window).scrollTop(), u = o + $(window).height();
            if (s && s.offset())
                if (i.attr("data-loaded") !== "no" || u < s.offset().top || o > s.offset().top + s.height())(u < s.offset().top || o > s.offset().top + s.height()) && i.attr("data-loaded", "no");
            else {
                i.attr("data-loaded", "yes");
                s.attr("src", r)
            }
        });
        return u
    }
    function D(e, t) {
        var n = $("body").contents().find("." + e), r = $("." + t, parent.window.document);
        $(r).attr("data-debug", $(n).attr("data-debug"))
    }
    function P() {
        $(document).on("Pb.Component.Detail.Detail::current_media_update Pb.Component.Filmstrip.Filmstrip::filmstrip_arrow_click", m);
        $(document).on("Pb.Component.Common.Likes::like_click", H)
    }
    function H(e) {
        m("likes=y")
    }
    function B(e) {
        typeof _gaq != "undefined" && _gaq.push(["_trackEvent", "Ads", "Refreshes", e, null, !0])
    }
    function j(e) {
        v = e
    }
    var e = "bannerAd", t = "ad200x70", n = "ad300x250", r = "promounit", i = "mrec", s = "skyscraperAd", o = "billboardAd", u = "skinAd", a = 0, f = null, l = (new Date).getTime(), c = (new Date).getTime(), h = 0, p = null, d=!0, v = 4e3, F = {
        reloadAds: function(e) {
            m(e)
        }
    };
    return {
        AD_BANNER: e,
        AD_200x70: t,
        AD_300x250: n,
        AD_970x250: r,
        AD_MREC: i,
        AD_SKYSCRAPER: s,
        AD_SKIN: u,
        AD_BILLBOARD: o,
        flash: F,
        refresh: S,
        replace: O,
        updateDebug: D,
        initMediaDetail: P,
        reloadAds: m,
        populateFromExistingMrec: M,
        populateFromUrlOnly: _,
        initContinuous: y,
        endContinuous: b,
        setRefreshTimeout: j,
        reloadAdsOnTimer: g
    }
}();
(function() {
    function e() {
        $(document).on("click", function() {
            $(".dropdown-menu.login.active").hide();
            $(".dropdown").removeClass("active")
        });
        $("#login-dropdown").on("click", function() {
            $(".dropdown-menu.login.active").hide();
            $(".dropdown").removeClass("active")
        })
    }
    function t() {
        function t() {
            $("#cartMenu .count").hide();
            if (Pb.Data.get("isLoggedIn")) {
                $("#cartMenu #checkoutIcon").addClass("empty");
                $(".checkoutContainer").removeClass("loggedoutempty")
            } else {
                $(".signUpLink").hide();
                $(".extraSpecialSignUpLink").show();
                $(".checkoutContainer").addClass("loggedoutempty")
            }
        }
        var e = $.cookie("pio");
        if (!e)
            t();
        else {
            $("#cartMenu .count").html(e);
            if (parseInt(e, 10) === 0)
                t();
            else {
                $("#cartMenu .count").css("display", "inline-block");
                $("#cartMenu #checkoutIcon").removeClass("empty");
                $(".checkoutContainer").removeClass("loggedoutempty");
                $(".extraSpecialSignUpLink").hide();
                Pb.Data.get("isLoggedIn") || $(".signUpLink").show()
            }
        }
    }
    function n() {
        var e = ".siteNav li.dropdown", t = "ul.dropdown-menu", n = 150, r = 150;
        Modernizr.touch && $(".dropdown-toggle").on("touchstart", function(e) {
            $(e.currentTarget).parent().attr("id") !== "login-dropdown" && $(e.currentTarget).parent().attr("id") !== "editMenu" && $(e.currentTarget).parent().attr("id") !== "libraryMenu" && e.preventDefault();
            var n = $(e.currentTarget).parent().find(t).get(0);
            $(n).is(":visible") ? $(n).hide() : $(n).show()
        });
        var i = $(e);
        i.hoverIntent({
            timeout: r,
            interval: n,
            over: function() {
                $(t, $(this)).show();
                $(t, $(this)).parent().addClass("active")
            },
            out: function() {
                $(t, $(this)).hide();
                $(t, $(this)).parent().removeClass("active")
            }
        });
        $("p[rel=popover]").popover({
            placement: "bottom",
            trigger: "manual",
            html: !0,
            template: '<div class="popover" onmouseover="$(this).mouseleave(function() {$(this).hide(); });"><div class="arrow"></div><div class="popover-inner"><div class="popover-content"><p></p></div></div></div>',
            container: $("p[rel=popover]")
        }).click(function(e) {
            e.preventDefault()
        }).on("mouseenter", function() {
            var e = this;
            $(this).popover("show");
            $(this).siblings(".popover").on("mouseleave", function() {
                $(e).popover("hide")
            })
        }).on("mouseleave", function() {
            var e = this;
            setTimeout(function() {
                $(".popover:hover").length || $(e).popover("hide")
            }, 100)
        });
        $("#login-dropdown .dropdown-toggle").on("mouseenter", function() {
            $(document).trigger("Pb.Component.Navigation.LoginBox::show");
            $(".login, #loginBox").show().parent().addClass("active")
        });
        $("#userMenu .dropdown-toggle").on("mouseenter", function() {
            $("ul.user").show().parent().addClass("active")
        });
        $(".navbar").on("mouseleave", "ul", function() {
            $("ul.user").hide().parent().removeClass("active")
        })
    }
    Pb.InitEventQueue.addToDomReady(e);
    Pb.InitEventQueue.addToDomReady(t);
    $(document).on("Printio::cartChanged", t);
    Pb.InitEventQueue.addToDomReady(n)
})();
$.getObject("Pb.Component.Common.ProfilePic", !0).ProfilePic = function() {
    function e(e) {
        e && $("img.ownerAvatar:not(.largeProfile)").attr("src", e)
    }
    function t(e) {
        e && $("img.ownerAvatar.largeProfile").attr("src", e)
    }
    return {
        update: e,
        updateLarge: t
    }
}();
(function() {
    function e(e, t) {
        var n = new RegExp(" " + t + "$");
        e.match(n) && (e = e.replace(n, ""));
        return e
    }
    function t() {
        var t = $(this), n, r = t.find("[name=searchUrl]"), i = t.find("[name=type]"), s, o = t.find(".search-query"), u = r.val(), a = o.val();
        if (a) {
            s = t.find(".dropdown-menu .active").text();
            a = e(a, s);
            n = u + i.val() + "/" + encodeURIComponent(a);
            document.location = n
        }
        return !1
    }
    function n() {
        var n = $(".searchbox"), r = n.find(".searchTrigger");
        n.on("click", "li", function(t) {
            var n, r = $(this), i, s = r.closest(".searchbox"), o, u, a = s.find("input.search-query"), f = s.find("input[name=type]");
            t.preventDefault();
            n = s.find(".dropdown-menu .active");
            u = e(a.val(), n.text());
            n.removeClass("active");
            r.addClass("active");
            n = r;
            f.val(n.attr("name"));
            a.attr("placeholder", "Search " + r.text());
            u.length && $(this).closest(".searchbox").submit()
        });
        n.on("submit", t);
        r.on("click", function(e) {
            e.preventDefault();
            $(this).closest(".searchbox").submit()
        })
    }
    Pb.InitEventQueue.addToDomReady(n, 10)
})();
(function() {
    function e(e) {
        this.init(e);
        this.bindAction()
    }
    var t = {
        init: function(e) {
            $(document).trigger($.Event("ActionEventsDispatcher::registerevent", {
                pbEvent: {
                    eventSelector: ".editMediaAction",
                    eventType: "click",
                    event: "EditLink::Edit"
                }
            }));
            $(document).trigger($.Event("ActionEventsDispatcher::registerevent", {
                pbEvent: {
                    eventSelector: ".editEditorAction",
                    eventType: "click",
                    event: "EditLink::Editor"
                }
            }))
        },
        bindAction: function() {
            var e = $(document);
            e.on("EditLink::Edit", $.proxy(function(e) {
                Pb.Track.tr("edit_modal_edit_click");
                var t = e.pbData;
                document.location = "/editor?image=" + escape(t.fullsizeUrl) + "&detailUrl=" + t.linkUrl
            }, this))
        }
    };
    $.extend(e.prototype, t);
    var n = new e
})();
$.getObject("Pb.Component.Library.MediaSelector", !0).MediaSelectorModalClass = function() {
    function e(t) {
        e.superclass.constructor.call(this)
    }
    var t = [], n = {
        MIN_IMAGE_DIMENSIONS: {
            height: 200,
            width: 200
        },
        data: {
            url: "",
            path: ""
        },
        subdomain: null,
        mode: "singleselect",
        warningShown: !1,
        showingUploadModal: !1,
        selectedItems: [],
        init: function(t) {
            var n = {
                eventName: "Pb.Component.Library.MediaSelector::show",
                modalId: "#mediaSelectorModal",
                modalName: "mediaSelectorModal",
                showClass: ".showMediaSelector"
            };
            t && $.extend(n, t);
            e.superclass.init.call(this, n);
            n.showClass && (this.showClass = n.showClass);
            var r = Pb.Data.get(this.modalName);
            if (r) {
                this.subdomain = Pb.Data.get(this.modalName).subdomain;
                this.initEvents()
            }
        },
        initEvents: function() {
            $(document).on("click", this.showClass, $.proxy(function(e) {
                var t = {
                    pbEventTarget: e.currentTarget,
                    mode: "singleselect",
                    eventselector: null,
                    hovertext: "Select",
                    modaltitle: e.modaltitle,
                    bodypretext: e.bodypretext,
                    showdropdown: !1,
                    enforcedimensions: !1,
                    closeonselect: !0,
                    save: !1,
                    tracking: null
                };
                $.extend(t, $(e.currentTarget).data());
                var n = $.Event(this.eventName, t);
                $(document).trigger(n)
            }, this));
            $(document).on("Pb.Component.Upload.UploadModal::uploadComplete", $.proxy(this.addFromUpload, this))
        },
        onTemplateLinked: function() {
            var e = $(this.modalId);
            e.on("click", ".showUploadModal", $.proxy(this.showUploadModal, this));
            e.on("show", 'a[data-toggle="tab"]', $.proxy(this.fetchTabContent, this));
            e.on("click", ".thumbnailLink.album, .breadcrumbs a", Pb.Component.Library.MediaSelector.Tabs.invokeTab);
            e.on("click", ".emptyTab", function(t) {
                t.preventDefault();
                t.stopPropagation();
                t.stopImmediatePropagation();
                e.find(".selectorTabs #albumTab").empty();
                e.find('.selectorTabs li a[href="#albumTab"]').click()
            })
        },
        onShow: function(t) {
            $("#editorpage .avpw_flash").hide();
            if ($(t.pbEventTarget).hasClass("disabled") || $(t.pbEventTarget).hasClass("showLoginModal"))
                return !1;
            e.superclass.onShow.call(this, t);
            var n = $(this.modalId);
            $(".bodypretext").text(t.bodypretext);
            if (t.mode === "multiselect") {
                this.showingUploadModal=!1;
                if (parseInt(Pb.Data.get(this.modalName).userMediaCount, 10) <= 0) {
                    n.find("#uploadBtn").click();
                    return !1
                }
                this.initMultiSelectModal(t)
            } else 
                t.closeonselect && n.on("click", ".thumbnailLink.media, .thumbnailOverlay", function() {
                    n.modal("hide")
                });
            if (t.mode === "singleselect") {
                n.find(".selectPanel").hide();
                n.find("#mediaSelectBtn").hide();
                n.find("#uploadBtn").hide();
                n.off("click", ".thumbnailLink.media")
            }
            t.save && n.find("#mediaSaveBtn").on("click", function(e) {
                e.preventDefault();
                $(document).trigger("MediaSelectorModal::save")
            }).show();
            t.modaltitle && n.find(".modal-header .title").html(t.modaltitle);
            t.showdropdown ? n.find(".modal-header #addFromOptions").removeClass("hide") : n.find(".modal-header #addFromOptions").addClass("hide");
            Pb.Component.Library.MediaSelector.Tabs.setModalSelector(this.modalId);
            Pb.Component.Library.MediaSelector.Tabs.setEventSelector(t.eventselector);
            Pb.Component.Library.MediaSelector.Tabs.setHoverText(t.hovertext);
            Pb.Component.Library.MediaSelector.Tabs.updateThumbs();
            t.filter && (this.data.filter = t.filter);
            var r = n.find(".selectorTabs .tabArea .active > a");
            this.data.path = "";
            t.defaultAlbum && (this.data.path = t.defaultAlbum);
            this.data.prefix = this.modalName;
            this.sendRequest(r.attr("href"));
            r.attr("href") === "#backgroundSelectorTab" && Pb.Track.tr("story_background_modal_pattern_view");
            t.tracking && Pb.Track.tr(t.tracking)
        },
        onHide: function() {
            $("#editorpage .avpw_flash").show();
            this.selectedItems = [];
            if (typeof StoryGraphic != "undefined") {
                var e=!1;
                $.each(StoryGraphic.collection.media, function(t, n) {
                    if (n && n.type !== "text") {
                        e=!0;
                        return 
                    }
                });
                var n = parseInt(Pb.Data.get(this.modalName).userMediaCount, 10);
                n > 0&&!this.showingUploadModal&&!e && t.length === 0 && Pb.Component.Element.Message.show({
                    message: "A story requires at least 1 photo or video.",
                    status: Pb.Component.Element.Message.STATUS_ERROR
                })
            }
            $(document).trigger("MediaSelectorModal::hide");
            t = [];
            this.templateLinked=!1
        },
        initMultiSelectModal: function(e) {
            this.mode = "multiselect";
            $(".mediaSelectorContainer").find(".selectPanel").show();
            $(".mediaSelectorContainer").find("#mediaSelectBtn").show();
            $(".mediaSelectorContainer").find(".uploadContainer").show();
            $(".mediaSelectorContainer").find(".patentPending").show();
            Pb.JSTemplates.link("#mediaSelectorTrayItemTemplate", ".selectPanel", this.selectedItems);
            this.showEmptyTray();
            this.addSelectedHover();
            $(this.modalId).on("click", ".thumbnailLink.media", {
                enforcedimensions: e.enforcedimensions
            }, $.proxy(this.addToTray, this)).on("click", ".selectedThumbnail", $.proxy(this.removeFromTray, this)).on("click", "#mediaSelectBtn", $.proxy(this.addMedia, this))
        },
        addFromUpload: function(e) {
            t = e.media;
            this.addMedia(e);
            Pb.Component.Element.Message.showDefaultSuccessMessage()
        },
        addMedia: function(e) {
            typeof StoryGraphic != "undefined" && t.length && Pb.Ajax.ajax({
                url: "/action/story/add?t=" + (new Date).getTime(),
                data: {
                    mode: "media",
                    destination: Pb.Data.get(Pb.Data.Shared.STORY).fullsizeUrl,
                    sourceData: t,
                    doRedirect: "false"
                },
                messageHandler: function() {
                    return !0
                },
                success: function(e) {
                    if (e.status === 1 && e.data.storyMedias.length > 0) {
                        StoryGraphic.addFromModal(e.data.storyMedias);
                        var t = Pb.Data.Shared.get(Pb.Data.Shared.STORY);
                        t.public = e.data.storyPrivacy;
                        Pb.Data.Shared.put(Pb.Data.Shared.STORY, t);
                        $(document).trigger($.Event("Pb.Component.Common.Privacy::privacy_changed", {
                            privacy: t.public
                        }))
                    }
                    e.messages && $.each(e.messages, function() {
                        Pb.Component.Element.Message.show(this)
                    })
                }
            });
            $(this.modalId).modal("hide")
        },
        fetchTabContent: function(e) {
            var t = e.target, n = e.relatedTarget, r = this.stripTrailingSlash(decodeURI($(t).data("url"))), i = this.stripTrailingSlash($(t.hash).data("url"));
            r === undefined && (this.data.path = "");
            $(this.modalId).find(n.hash).hide();
            if (r === undefined || r !== i || $(this.modalId).find(t.hash).is(":empty")) {
                this.data.url = r;
                this.sendRequest(t.hash)
            } else 
                $(this.modalId).find(t.hash).show();
            t.hash === "#backgroundSelectorTab" && Pb.Track.tr("story_background_modal_pattern_view")
        },
        sendRequest: function(e) {
            this.showProgress();
            var t = "/component/Library-MediaSelector-MediaTab";
            e === "#albumSelectorTab" && (t = "/component/Library-MediaSelector-AlbumsTab");
            e === "#backgroundSelectorTab" && (t = "/component/Library-MediaSelector-BackgroundsTab");
            $.ajax({
                url: t,
                data: this.data,
                type: "get",
                success: $.proxy(function(t) {
                    this.updateProgress("100%");
                    setTimeout($.proxy(function() {
                        this.updateModalUI(t, e)
                    }, this), 250)
                }, this)
            })
        },
        updateModalUI: function(e, t) {
            var n = 0, r = 0, i = "", s = "", o, u = $(this.modalId).find(".selectorTabs"), a = $(this.modalId).find(".breadcrumbs");
            u.find(t).replaceWith(e);
            this.mode === "multiselect" && this.hideFromList();
            o = Pb.Data.Shared.get(Pb.Data.Shared.MEDIA_SELECTOR_ALBUM);
            if (o) {
                n = o.albumCount;
                r = o.mediaCount;
                s = o.location;
                i = o.url
            }
            u.find('li a[href="#mediaSelectorTab"]').text("Your Bucket (" + r + ")");
            u.find('li a[href="#albumSelectorTab"]').text("Albums (" + n + ")");
            var f = s.split("/"), l;
            a.empty();
            if (f[0] !== "") {
                a.append("<a href='" + i.split("/" + encodeURI(s))[0] + "'>Library</a>");
                $.each(f, function(e, t) {
                    l = i.split("/").slice(0, - 1).join("/");
                    e === f.length - 1 ? a.append(" / " + t) : a.append(" / <a href='" + l + "'>" + t + "</a>")
                })
            }
            $(this.modalId).find(".progress").hide()
        },
        stripTrailingSlash: function(e) {
            return e === undefined || e === null ? e : e.substr( - 1) === "/" ? e.substr(0, e.length - 1) : e
        },
        showProgress: function() {
            this.updateProgress("40%");
            $(this.modalId).find(".progress").show()
        },
        updateProgress: function(e) {
            $(this.modalId).find(".progress .bar").css("width", e)
        },
        addToTray: function(e) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            var t = $(e.currentTarget).find("img");
            e.data.enforcedimensions ? this.checkDimensions(t) : this.moveToTray(t)
        },
        moveToTray: function(e) {
            var n = $(this.modalId);
            if (n.find(".selectedThumb").length === 0) {
                n.find(".selectPanel .dragNote").remove();
                n.find("#mediaSelectBtn").removeClass("disabled").removeAttr("disabled")
            }
            var r = e.parents(".thumbnail").attr("id"), i = e.attr("src");
            e.parents(".thumbnail").addClass("hide");
            var s = e.parents(".thumbnail").data("pbdkey"), o, u = Pb.Data.Shared.get(s);
            $.each(u, function(e, t) {
                if (t.id === r) {
                    o = t;
                    return 
                }
            });
            var a = {
                selectedThumbUrl: i,
                selectedId: r,
                mediaUrl: o.linkUrl,
                mediaType: o.mediaType
            };
            t.push(a.mediaUrl);
            if (this.warningShown===!1 && Pb.Data.Shared.get(Pb.Data.Shared.STORY) !== null && Pb.Data.Shared.get(Pb.Data.Shared.STORY).public&&!Pb.Data.Shared.get(Pb.Data.Shared.MEDIA_SELECTOR_ALBUM).isPublic) {
                Pb.Component.Element.Message.show({
                    status: 1,
                    message: "You are about to add a private photo or video to a public Story."
                });
                this.warningShown=!0
            }
            $.observable(this.selectedItems).insert(this.selectedItems.length, a);
            (this.selectedItems.length + 1)%5 === 0 && n.find(".tab-content").trigger("scroll")
        },
        checkDimensions: function(e) {
            if (e.data("fetching"))
                return;
            e.addClass("fade").removeClass("in");
            e.data("fetching", !0);
            var t = e.parents(".thumbnail").data("pbdkey"), n = e.parents(".thumbnail").attr("id"), r, i = Pb.Data.Shared.get(t);
            $.each(i, function(e, t) {
                if (t.id === n) {
                    r = t;
                    return 
                }
            });
            var s = new Pb.Path.PBPath(r.rawpath), o = Pb.Media.MediaIdentifierFactory.getIdentifier(s), u = {
                type: "GET",
                cache: !0,
                url: "/api/user/" + s.ownername + "/media/" + o + "/dimensions",
                success: $.proxy(function(t) {
                    if (t.data && t.data.height && t.data.width)
                        if (this.MIN_IMAGE_DIMENSIONS.height > t.data.height || this.MIN_IMAGE_DIMENSIONS.width > t.data.width) {
                            var n = {
                                status: 0,
                                message: "This photo is smaller than the " + this.MIN_IMAGE_DIMENSIONS.width + "x" + this.MIN_IMAGE_DIMENSIONS.height + " minimum size requirement for Stories.  Try selecting a larger photo.",
                                details: ""
                            };
                            Pb.Component.Element.Message.show(n);
                            e.removeData("fetching");
                            e.addClass("in");
                            return 
                        }
                    this.moveToTray(e);
                    e.removeData("fetching");
                    e.addClass("in")
                }, this),
                error: function() {
                    Pb.Component.Element.Message.showDefaultErrorMessage();
                    e.removeData("fetching");
                    e.addClass("in")
                }
            };
            Pb.Ajax.ajax(u)
        },
        removeFromTray: function(e) {
            var n = $(e.currentTarget).data("id"), r;
            $(this.modalId).find(".thumbnail[id='" + n + "']").removeClass("hide");
            for (var i = 0; i < this.selectedItems.length; i++)
                if (this.selectedItems[i].selectedId === n) {
                    r = i;
                    break
                }
            t.splice(r, 1);
            $.observable(this.selectedItems).remove(r);
            this.selectedItems.length === 0 && this.showEmptyTray()
        },
        showEmptyTray: function() {
            $(this.modalId).find(".selectPanel").append($("#mediaSelectorTrayEmptyTemplate").html());
            $(this.modalId).find("#mediaSelectBtn").addClass("disabled").attr("disabled", "disabled")
        },
        hideFromList: function() {
            for (var e = 0; e < this.selectedItems.length; e++) {
                $(this.modalId).find(".thumbnail[id='" + this.selectedItems[e].selectedId + "']").addClass("hide");
                e%5 === 0 && $(this.modalId).find(".tab-content").trigger("scroll")
            }
        },
        addSelectedHover: function() {
            $(this.modalId).find(".selectedThumb").hoverIntent({
                over: function() {
                    $(this).find(".mediaOverlay").show()
                },
                timeout: 100,
                out: function() {
                    $(this).find(".mediaOverlay").hide()
                }
            })
        },
        showUploadModal: function() {
            if (Pb.Component.Upload === undefined)
                window.location = "/upload?location=" + Pb.Data.Shared.get(Pb.Data.Shared.MEDIA_SELECTOR_ALBUM).location;
            else {
                this.showingUploadModal=!0;
                $(this.modalId).modal("hide")
            }
        }
    };
    Pb.subclass(e, Pb.Component.Element.Modal, n);
    return e
}();
$.getObject("Pb.Component.Library.MediaSelector", !0).MediaSelectorModal = function() {
    var e = new Pb.Component.Library.MediaSelector.MediaSelectorModalClass;
    Pb.InitEventQueue.add(function() {
        e.init()
    });
    return e
}();
$.getObject("Pb.Component.Library.MediaSelector").Tabs = function() {
    function i(t) {
        e = t
    }
    function s(e) {
        r = e
    }
    function o(e) {
        t = e
    }
    function u(e, t, i, s, o) {
        Pb.Data.Shared.put(Pb.Data.Shared.MEDIA_SELECTOR, e.objects);
        var u = {
            query: t,
            scrollContainer: r + " .tab-content",
            contentContainer: r + " .mediaSelectorTab",
            trackablePages: !1,
            pageSize: 20
        }, a = new Pb.Component.Element.InfiniteCollection.InfiniteCollection(u);
        n = new Pb.Component.Library.MediaSelector.ThumbnailList({
            pbDataKey: Pb.Data.Shared.MEDIA_SELECTOR,
            ic: a,
            userId: i,
            albumName: s,
            container: o,
            query: t
        })
    }
    function a() {
        n !== undefined && (n.query = null)
    }
    function f() {
        $(".thumbnailLink.media", r).addClass(e).attr("href", "#").data("link", e);
        var n = $('<div class="mediaOverlay">' + t + "</div>"), i = {
            over: function() {
                $(this).append(n)
            },
            timeout: 100,
            out: function() {
                $(this).find(".mediaOverlay").remove()
            }
        };
        $(r + " .thumbnailLink.media").hoverIntent(i)
    }
    function l(e, t) {
        Pb.JSTemplates.link("albumThumbnailTemplate", t, e.objects)
    }
    function c(e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        var t = $(r), n = t.find(".selectorTabs"), i;
        n.find("li a").data("url", e.currentTarget.href);
        a();
        t.find(".mediaSelectorTab").empty();
        i = n.find('li a[href="#mediaSelectorTab"]');
        i.parent("li").removeClass("active");
        n.find('li a[href="#albumSelectorTab"]').parent("li").addClass("active");
        i.tab("show");
        t.find(".albumSelectorTab").empty();
        return !1
    }
    var e = null, t = "Select", n, r;
    return {
        initMediaThumbs: u,
        initAlbumThumbs: l,
        setModalSelector: s,
        setEventSelector: i,
        setHoverText: o,
        updateThumbs: f,
        invokeTab: c,
        resetQuery: a
    }
}();
$.getObject("Pb.Component.Element.InfiniteCollection", !0).InfiniteCollection = function() {
    function e(e) {
        this.init(e)
    }
    $.extend(e.prototype, {
        query: null,
        offset: 0,
        total: 0,
        pageSize: 24,
        loadProximity: 100,
        working: !1,
        scrollContainer: null,
        contentContainer: null,
        complete: !1,
        scrollEvent: "scroll",
        trackablePages: !0,
        scrollTarget: window,
        loadingIndicator: null,
        fetchOnScroll: !0,
        scrollEventsActive: !1,
        isOrganizer: !1,
        slideshow: !1,
        paused: !1,
        currentPage: 0,
        pagedItemsContainer: null,
        trackContext: null,
        init: function(e) {
            e.query && (this.query = e.query);
            e.startOffset>-1 && (this.offset = e.startOffset);
            this.total = e.total;
            e.pageSize && (this.pageSize = e.pageSize);
            this.onLoadMore = e.onLoadMore;
            this.onLoadMoreViaQuery = e.onLoadMoreViaQuery;
            e.scrollContainer && (this.scrollContainer = e.scrollContainer);
            e.contentContainer && (this.contentContainer = e.contentContainer);
            e.pagedItemsContainer && (this.pagedItemsContainer = e.pagedItemsContainer);
            e.loadProximity && (this.loadProximity = e.loadProximity);
            e.loadingIndicator && (this.loadingIndicator = e.loadingIndicator);
            e.hasOwnProperty("trackablePages") && (this.trackablePages = e.trackablePages);
            e.hasOwnProperty("slideshow") && (this.slideshow = e.slideshow);
            this.forceScrollbar();
            this.initEvents()
        },
        initEvents: function() {
            this.initScrollEvents();
            $(window).on("resize", $.proxy(this.forceScrollbar, this));
            $(document).on("Organizer::active", $.proxy(function() {
                this.isOrganizer=!0
            }, this)).on("Organizer::inactive", $.proxy(function() {
                this.isOrganizer=!1
            }, this))
        },
        initScrollEvents: function() {
            if (!this.scrollEventsActive) {
                if (this.scrollContainer !== null) {
                    this.scrollTarget = this.scrollContainer;
                    $(this.scrollContainer).on("scroll.infiniteCollection", $.throttle(200, $.proxy(this.onScroll, this)))
                } else {
                    this.scrollTarget = window;
                    $(window).on("scroll.infiniteCollection", $.throttle(200, $.proxy(this.onScroll, this)))
                }
                this.scrollEventsActive=!0
            }
        },
        removeScrollEvents: function() {
            if (this.scrollEventsActive) {
                $(this.scrollTarget).off(".infiniteCollection");
                this.scrollEventsActive=!1
            }
        },
        reset: function(e, t, n, r) {
            this.offset = e;
            this.total = t;
            n && (this.query = n);
            this.onLoadMoreViaQuery = r;
            if (!this.total || this.offset < this.total) {
                if (this.complete) {
                    this.complete=!1;
                    this.initScrollEvents()
                }
            } else if (this.offset === this.total) {
                this.complete=!0;
                this.removeScrollEvents()
            }
        },
        pause: function() {
            this.paused=!0
        },
        resume: function() {
            this.paused=!1
        },
        onScroll: function() {
            if (!this.fetchOnScroll || this.complete || this.paused)
                return;
            var e, t;
            if (this.done)
                return;
            if (this.scrollContainer !== null && this.contentContainer !== null) {
                e = $(this.contentContainer).height() - $(this.scrollContainer).height();
                t = Math.abs(e - $(this.scrollContainer).scrollTop())
            } else {
                e = $(document).height() - $(window).height();
                t = Math.abs(e - $(window).scrollTop())
            }
            this.onScrollDone(t)
        },
        onTouchScroll: function() {
            if (!this.fetchOnScroll || this.complete)
                return;
            var e;
            if (this.done)
                return;
            e = $(document).height() - $(window).height() - $(window).scrollTop();
            this.onScrollDone(e)
        },
        onScrollDone: function(e) {
            if (this.working)
                return;
            e <= this.loadProximity && this.loadMore();
            var t = 0, n = $(window).scrollTop(), r = n + $(window).height();
            for (; ;) {
                var i = $(this.pagedItemsContainer).eq(t * this.pageSize);
                if (i.length === 0)
                    break;
                var s = i.offset().top, o = s + i.height(), u = s >= n;
                if (u)
                    break;
                t += 1
            }
            if (t !== this.currentPage) {
                this.currentPage = t;
                this.pageScrolledIntoView(this.currentPage)
            }
        },
        loadMore: function(e) {
            this.working=!0;
            e || this.showLoadingIndicator();
            this.query === null ? this.onLoadMore(this.offset, this.pageSize) : this.onLoadMoreViaQuery(this.offset, this.query);
            this.currentPage = this.offset / this.pageSize;
            this.track()
        },
        pageScrolledIntoView: function(e) {
            this.onPageScrolledIntoView(e)
        },
        forceScrollbar: function() {
            if (this.scrollContainer === null && this.contentContainer === null) {
                var e = $("#footer"), t, n;
                if (e.length) {
                    t = $(window).height();
                    n = t;
                    this.fetchOnScroll && (n += 3);
                    $("#content").css("min-height", n + "px")
                }
            }
        },
        setContainer: function(e) {
            this.container = $("#" + e)
        },
        onLoadMore: function(e, t) {},
        onLoadMoreViaQuery: function(e, t) {},
        onPageScrolledIntoView: function(e) {},
        moreLoaded: function(e) {
            this.fetchFinished();
            this.offset += e;
            if (e === 0 || this.total && e >= this.total) {
                this.removeScrollEvents();
                this.complete=!0
            } else 
                $(this.scrollTarget).trigger(this.scrollEvent)
        },
        fetchFinished: function() {
            this.hideLoadingIndicator();
            this.working=!1
        },
        showLoadingIndicator: function() {
            this.loadingIndicator !== null && $(this.loadingIndicator).removeClass("hide")
        },
        hideLoadingIndicator: function() {
            this.loadingIndicator !== null && $(this.loadingIndicator).addClass("hide")
        },
        setTrackContext: function(e) {
            this.trackContext = e
        },
        track: function() {
            var e = "";
            if (this.slideshow)
                return;
            !this.isOrganizer && this.trackablePages && (this.query && this.query.indexOf("filters[story]")>-1 ? Pb.PageTracker.trackContentRefreshPageView({
                pb_pg: "library:stories"
            }) : this.query && this.query.indexOf("filters[album]")>-1 ? Pb.PageTracker.trackContentRefreshPageView({
                pb_pg: "library:library"
            }) : this.offset > 0 && Pb.PageTracker.trackContentRefreshPageView({
                pb_pg: ""
            }));
            if (this.isOrganizer)
                e = "infinite_scroll_organizer_view";
            else {
                if (!this.query)
                    return;
                this.query.indexOf("filters[album]")>-1 ? e = "infinite_scroll_library_view" : this.query.indexOf("filters[term]=Popular")>-1 ? e = "infinite_scroll_popular_view" : this.query.indexOf("filters[term]")>-1 ? e = "infinite_scroll_search_view" : this.query.indexOf("filters[recentUploads]")>-1 ? e = "infinite_scroll_recentuploads_view" : this.query.indexOf("filters[likes]")>-1 && (e = "infinite_scroll_likes_view");
                this.trackContext && this.offset > 0 && (e = "infinite_scroll_" + this.trackContext + "_view")
            }
            e !== "" && Pb.Track.tr(e)
        }
    });
    return e
}();
$.getObject("Pb", !0).ThumbnailListBase = function() {
    function e(e) {}
    var t = {
        pbDataKey: null,
        fetchServiceUrl: null,
        fetchParams: null,
        linked: !1,
        bannnerAdCreated: !1,
        mrecAdCreated: !1,
        infiniteScrollAds: !0,
        lastXhr: null,
        baseInit: function() {
            var e = Pb.Data.Shared.get(this.pbDataKey);
            this.appendToObjects(e)
        },
        resetObjects: function(e) {
            this.linked=!1;
            Pb.Data.Shared.put(this.pbDataKey, e);
            this.appendToObjects(e)
        },
        linkTemplate: function(e) {
            Pb.JSTemplates.link(this.templateTarget, this.container, e);
            this.linked=!0
        },
        appendToObjects: function(e) {
            if (!this.returnData) {
                Pb.Data.Shared.sync(this.pbDataKey, e);
                if (!this.linked)
                    this.linkTemplate(e);
                else {
                    var t = Pb.Data.Shared.get(this.pbDataKey);
                    $.observable(t).insert(t.length, e)
                }
            }
            typeof this.onObjectsAppended == "function" && this.onObjectsAppended(e)
        },
        abortLastXhr: function() {
            this.lastXhr && this.lastXhr.abort()
        },
        fetchMoreContent: function() {
            var e = this.fetchParams.size === undefined ? "large": this.fetchParams.size, t = Pb.Data.has("gridThumbnailSize") ? Pb.Data.get("gridThumbnailSize"): e, n = $.extend({}, this.fetchParams, {
                size: t
            });
            this.lastXhr = Pb.Ajax.ajax({
                url: this.fetchServiceUrl,
                data: n,
                success: $.proxy(this.handleFetchMoreContent, this),
                error: $.proxy(this.onFetchMoreFailed, this),
                type: "get"
            })
        },
        onFetchMoreFailed: function(e) {},
        insertAd: function(e, t) {
            var n = 0, r=!1;
            if (e !== null && e !== undefined && e.length > 0 && this.infiniteScrollAds) {
                var i = this.ic.pageSize;
                n = Math.ceil(this.ic.offset / i) - 1;
                r=!0
            } else {
                n = t;
                r=!0
            }
            if (!r)
                return;
            this.placeSkyscraperAd();
            n < 1 && this.removeInfiniteScrollAds();
            if (n >= 2 && n%2 == 0) {
                this.refreshSkyscraperAd();
                this.refreshBottomBannerAd();
                this.refreshPaginationMrecAd()
            }
        },
        placeSkyscraperAd: function() {
            var e = $(".skyscraperAd");
            if (e.length) {
                e.css("left", e.offset().left + "px");
                e.css("top", "10px");
                e.css("position", "fixed");
                $(window).on("scroll.infiniteScrollAds", function() {
                    $(window).scrollTop() < $(".ad.bannerAd:first").height() + 50 ? e.css("position", "static") : e.css("position", "fixed")
                })
            }
        },
        refreshSkyscraperAd: $.throttle(5e3, !0, function() {
            var e = $(".skyscraperAd");
            e.length && Pb.Component.Ad.AdRefresh.refresh(Pb.Component.Ad.AdRefresh.AD_SKYSCRAPER, Pb.Component.Ad.AdRefresh.AD_SKYSCRAPER, "pos=inf")
        }),
        createBottomBannerAd: function() {
            function t() {
                $(window).scrollTop() < $(".ad.bannerAd:first").height() ? $("." + e).hide() : $("." + e).show()
            }
            var e = "bannerAd2";
            this.bannerAdCreated=!0;
            $("body").append("<div class='ad bannerAd " + e + "' style='position:fixed; bottom: 0; width: 100%; z-index: 2; padding: 5px 0;'></div>");
            $(window).on("scroll.infiniteScrollAds", $.throttle(250, t));
            t()
        },
        refreshBottomBannerAd: $.throttle(5e3, !0, function() {
            $(".bannerAd").length > 0 && (this.bannerAdCreated || this.createBottomBannerAd());
            var e = "bannerAd2";
            Pb.Component.Ad.AdRefresh.refresh(Pb.Component.Ad.AdRefresh.AD_BANNER, e, "pos=inf")
        }),
        createPaginationMrecAd: function() {
            function n() {
                $(window).scrollTop() < $(".sidebar").position().top + $(".sidebar").height() + 50 ? $("." + e).hide() : $("." + e).show()
            }
            var e = "mrec2";
            this.mrecAdCreated=!0;
            var t = $(this.container).offset().left + $(this.container).width() + 20;
            $("body").append("<div class='ad mrec " + e + "' style='position:fixed; top: 10px; left:  " + t + "px; min-height:250px;width:300px;'></div>");
            $(window).on("resize.infiniteScrollAds", $.proxy(function() {
                var t = $(this.container).offset().left + $(this.container).width() + 20;
                $("div." + e).css("left", t + "px")
            }, this));
            $(window).on("scroll.infiniteScrollAds", $.throttle(250, n));
            n()
        },
        refreshPaginationMrecAd: $.throttle(5e3, !0, function() {
            $(".mrec").length > 0 && (this.mrecAdCreated || this.createPaginationMrecAd());
            var e = "mrec2";
            Pb.Component.Ad.AdRefresh.refresh(Pb.Component.Ad.AdRefresh.AD_MREC, e, "pos=inf")
        }),
        removeInfiniteScrollAds: function() {
            $("div.mrec2").remove();
            this.mrecAdCreated=!1;
            $("div.bannerAd2").remove();
            this.bannerAdCreated=!1;
            $(window).off(".infiniteScrollAds")
        },
        enableInfiniteScrollAds: function() {
            this.infiniteScrollAds=!0
        },
        disableInfiniteScrollAds: function() {
            this.infiniteScrollAds=!1;
            this.removeInfiniteScrollAds()
        }
    };
    $.extend(e.prototype, t);
    return e
}();
$.getObject("Pb.Component.Library.MediaSelector", !0).ThumbnailList = function() {
    function e(e) {
        this.init(e)
    }
    var t = new Pb.ThumbnailListBase, n = {
        fetchParams: null,
        fetchServiceUrl: null,
        container: "",
        templateTarget: null,
        pbDataKey: null,
        ic: null,
        userId: null,
        albumName: null,
        query: null,
        sort: null,
        init: function(e) {
            this.pbDataKey = e.pbDataKey;
            this.ic = e.ic;
            this.query = e.query;
            this.userId = e.userId;
            this.albumName = e.albumName;
            this.sort = e.sort;
            this.container = "#" + e.container;
            this.templateTarget = "#mediaSelectorThumbnailListTemplate";
            this.initEvents();
            this.baseInit()
        },
        initEvents: function() {
            this.ic.onLoadMoreViaQuery = $.proxy(this.onLoadMoreViaQuery, this)
        },
        handleFetchMoreContent: function(e) {
            this.appendToObjects(e.objects)
        },
        onLoadMoreViaQuery: function(e, t) {
            if (this.query) {
                this.fetchParams = {
                    offset: e
                };
                this.fetchServiceUrl = "/component/Library-MediaSelector-MediaThumbnailList?" + t;
                this.fetchMoreContent()
            }
        },
        onObjectsAppended: function(e) {
            var t = e.length;
            this.ic.moreLoaded(t);
            Pb.Component.Library.MediaSelector.Tabs.updateThumbs()
        }
    };
    $.extend(e.prototype, n, t);
    return e
}();
$.getObject("Pb.Component.Thumbnail", !0).Thumbnail = function() {
    function t(t) {
        t.on("mouseenter", ".optionsHandle", $.throttle(100, function() {
            e && e.hide();
            var t = $(this).siblings(".optionsMenu");
            t.show();
            e = t
        })).on("mouseleave", ".optionsMenu", $.throttle(100, function() {
            $(this).hide()
        })).on("mouseenter", ".thumbnail", $.throttle(100, function() {
            $(this).find(".optionsHandle").show()
        })).on("mouseleave", ".thumbnail", $.throttle(100, function() {
            t.find(".optionsMenu:visible").hide();
            t.find(".optionsHandle:visible").hide()
        }))
    }
    function n() {
        $(".thumbContainer").off("mouseenter", ".optionsHandle").off("mouseleave", ".optionsMenu").off("mouseenter", ".thumbnail").off("mouseleave", ".thumbnail")
    }
    function r(e) {
        e.tooltip({
            selector: ".badge"
        })
    }
    function i() {
        if ($(document.documentElement).hasClass("is_iPad"))
            return;
        ($("#fileUploaderWrapper").length&&!$("#fileUploaderWrapper").hasClass("uploading") ||!$("#fileUploaderWrapper").length) && t($(".thumbContainer"));
        r($(".thumbContainer"))
    }
    var e;
    Pb.InitEventQueue.add(i);
    return {
        init: i,
        removeOptionsMenus: n
    }
}();
$.getObject("Pb.Component.Album.Edit", !0).EditAlbum = function() {
    function e() {
        e.superclass.constructor.call(this)
    }
    var t = {
        parentAlbumPath: null,
        albumPath: null,
        mode: null,
        notify: !1,
        notifyOnTitleChange: !1,
        albumPrivacyLevel: null,
        albumPrivacyPassword: null,
        albumVanity: null,
        albumIsRoot: null,
        chooser: null,
        isTagged: !1,
        leftRailClick: !1,
        hashtag: !1,
        passwordMsg: "Passwords must be at least 6 characters long. They can only contain the characters a-z, A-Z, 0-9, -, and _. Spaces are not allowed.",
        init: function(t) {
            var n = {
                eventName: "Pb.Component.Album.Edit::show",
                modalId: "#editAlbumModal",
                modalName: "editAlbumModal",
                submitButtonId: "#createAlbumButton, #editAlbumButton",
                showProgress: !0,
                refreshAdsOnHide: !0
            };
            t && $.extend(n, t);
            e.superclass.init.call(this, n);
            this.initEvents()
        },
        initEvents: function() {
            $(document).on("click", ".editAlbumAction", function(e) {
                e.preventDefault();
                var t = {
                    mode: $(this).attr("data-mode"),
                    notify: $(this).attr("data-notify"),
                    notifyOnTitleChange: $(this).attr("data-notifyontitlechange"),
                    isTagged: $(this).attr("data-istagged"),
                    leftRailClick: $(this).attr("data-leftrailclick")
                };
                if (!Pb.Data.Albums || t.leftRailClick === "true") {
                    t.albumPath = $(this).attr("data-albumpath");
                    t.albumTitle = $(this).attr("data-albumtitle");
                    t.albumDescription = $(this).attr("data-albumdescription");
                    t.albumPrivacyLevel = $(this).attr("data-albumprivacylevel");
                    t.albumPrivacyPassword = $(this).attr("data-albumprivacypassword");
                    t.albumVanity = $(this).attr("data-albumvanity");
                    t.albumIsRoot = $(this).attr("data-albumisroot")
                }
                $(document).trigger($.Event("Pb.Component.Album.Edit::show", t))
            });
            $(document).on("click", "#editAlbumModal #editAlbumShowDescriptionButton", function(e) {
                e.preventDefault();
                $(".descContainerCollapsed").hide();
                $(".descContainer").show()
            })
        },
        buildSubmitData: function() {
            if (this.mode === "create") {
                this.submitUrl = "/action/album/create";
                return this.buildCreateAlbumData()
            }
            this.submitUrl = "/action/album/update";
            return this.buildEditAlbumData()
        },
        getPrivacyPassword: function() {
            var e = "";
            $("#privacyPasswordCheckbox").is(":checked") && (e = $("#privacyPassword").val());
            return e
        },
        buildCreateAlbumData: function() {
            var e = {
                parentAlbumPath: this.parentAlbumPath,
                name: $("#editAlbumTitle").val(),
                description: $("#editAlbumDescription").val(),
                vanityName: $("#editAlbumVanity").val(),
                reloadPage: this.notify?!1: !0,
                hashtag: $(".createHashtagSectionContent #createHashtag").length && $(".createHashtagSectionContent #createHashtag").val().length ? $(".createHashtagSectionContent #createHashtag").val(): !1
            };
            $("input#privacyLevelPrivate").is(":checked") ? e.privacyLevel = $("#privacyPasswordCheckbox").is(":checked") ? "password" : "private" : e.privacyLevel = "public";
            var t = this.getPrivacyPassword();
            if (e.privacyLevel === "password" && t !== this.albumPrivacyPassword) {
                var n, r = /^[\w-]{6,50}$/;
                if ($.trim(t) === "") {
                    n = {
                        status: 0,
                        message: "",
                        details: this.passwordMsg
                    };
                    Pb.Component.Element.Message.show(n);
                    return !1
                }
                if (!r.test(t)) {
                    n = {
                        status: 0,
                        message: "",
                        details: this.passwordMsg
                    };
                    Pb.Component.Element.Message.show(n);
                    this.showProgress=!1;
                    return !1
                }
                e.privacyPassword = t
            }
            return e
        },
        buildEditAlbumData: function() {
            if (this.mode === "privacy")
                var e = {
                    albumPath: this.albumPath,
                    parentAlbumPath: this.parentAlbumPath,
                    name: this.isTagged ? this.albumTitle.replace("#", ""): this.albumTitle,
                    rootHint: this.albumIsRoot ? 1: 0
                };
            else 
                var e = {
                    albumPath: this.albumPath,
                    parentAlbumPath: this.parentAlbumPath,
                    name: this.isTagged ? $("#editAlbumTitle").val().replace("#", ""): $("#editAlbumTitle").val(),
                    description: $("#editAlbumDescription").val(),
                    vanityName: $("#editAlbumVanity").val(),
                    rootHint: this.albumIsRoot ? 1: 0,
                    hashtag: $(".editHashtagSectionContent #createHashtag").length && $(".editHashtagSectionContent #createHashtag").val().length ? $(".editHashtagSectionContent #createHashtag").val(): !1
                };
            $("input#privacyLevelPrivate").is(":checked") ? e.privacyLevel = $("#privacyPasswordCheckbox").is(":checked") ? "password" : "private" : e.privacyLevel = "public";
            var t = this.getPrivacyPassword();
            if (e.privacyLevel === "password" && t !== this.albumPrivacyPassword) {
                var n, r = /^[\w-]{6,50}$/;
                if ($.trim(t) === "") {
                    n = {
                        status: 0,
                        message: "",
                        details: this.passwordMsg
                    };
                    Pb.Component.Element.Message.show(n);
                    return !1
                }
                if (!r.test(t)) {
                    n = {
                        status: 0,
                        message: "",
                        details: this.passwordMsg
                    };
                    Pb.Component.Element.Message.show(n);
                    this.showProgress=!1;
                    return !1
                }
                e.privacyPassword = t
            }
            return e
        },
        onSuccess: function(e) {
            var t = e.data;
            if (e.status === 1)
                if (this.mode === "create")
                    this.notify ? $(document).trigger("EditAlbum::createAlbumCompleted", t) : this.onCreateSuccess(t);
                else if (t.titleChange && this.notifyOnTitleChange)
                    $(document).trigger("EditAlbum::albumTitleChange", t);
                else if (t.titleChange ||!this.notify)
                    this.onEditSuccess(t);
                else {
                    this.mode === "privacy" && (t.privacyType === "public" ? $(".privacyIcon").find("i").removeClass().addClass("icon-unlock") : t.privacyPassword ? $(".privacyIcon").find("i").removeClass().addClass("icon-key") : $(".privacyIcon").find("i").removeClass().addClass("icon-lock"));
                    var n = Pb.Data.Albums.getCurrentAlbum();
                    n.mergeData(t);
                    $(document).trigger($.Event("Pb.Component.Album.Edit.EditAlbum::editAlbumCompleted"))
                }
            $("#editAlbumModal input[type=submit]").attr("disabled", !1)
        },
        onCreateSuccess: function(e) {
            window.location = e.libraryUrl
        },
        onEditSuccess: function(e) {
            document.location.href = e.libraryUrl
        },
        onFail: function(e) {
            $("#editAlbumModal > form").find("input[type=submit]").attr("disabled", !1);
            var t = {
                status: 0,
                message: e.messages[0].message,
                details: ""
            };
            Pb.Component.Element.Message.show(t)
        },
        messageHandler: function(e) {
            if (e.memo && e.memo !== "") {
                $("." + e.memo + "Error").siblings().find(".help-inline").html(e.message).parents(".control-group").addClass("error");
                return !0
            }
            $("#editAlbumModal").find(".control-group").removeClass("error").find(".help-inline").html("");
            return !1
        },
        show: function(e, t, n, r, i, s, o, u, a, f, l) {
            var c = $.Event("Pb.Component.Album.Edit::show", {
                mode: e,
                parentAlbumPath: t,
                albumPath: n,
                albumTitle: r,
                albumDescription: i,
                albumPrivacyLevel: s,
                albumPrivacyPassword: o,
                albumVanity: u,
                albumIsRoot: a,
                isTagged: f,
                hashtag: l
            });
            $(document).trigger(c)
        },
        hideOrShowAlbums: function() {
            if (this.mode === "create") {
                $(".editAlbumSelector").show();
                this.chooser && this.chooser.loadAlbums()
            } else 
                $(".editAlbumSelector").hide()
        },
        onAlbumsLoading: function(e) {
            $("#createAlbumButton").attr("disabled", !0);
            $("#editAlbumChooser .dropdown").hide()
        },
        onAlbumsLoaded: function(e) {
            $("#editAlbumChooser .dropdown").show();
            $("#createAlbumButton").attr("disabled", !1);
            this.parentAlbumPath && this.chooser.setSelectedAlbum(this.parentAlbumPath)
        },
        onAlbumSelected: function(e) {
            this.parentAlbumPath = e.albumData.albumpath
        },
        cleanupErrors: function() {
            $(".help-block").html("");
            $(".error").removeClass("error")
        },
        initializePrivacyDisplay: function() {
            var e = $("#privacyTitleCurrentSetting"), t = $("#privacyPasswordCheckbox"), n = $("#privacyPassword"), r = $("#showOrHide");
            if (this.albumPrivacyLevel === "public" || this.mode === "create") {
                e.html("Public");
                e.removeClass("privacySettingPrivateColor");
                e.addClass("privacySettingPublicColor");
                $("#privacyLevelPublic").prop("checked", !0);
                $("#privacyLevelPrivate").prop("checked", !1);
                t.prop("checked", !1);
                n.val("");
                n.prop("disabled", !0)
            } else {
                $("#privacyTitleCurrentSetting").html("Private");
                e.removeClass("privacySettingPublicColor");
                e.addClass("privacySettingPrivateColor");
                $("#privacyLevelPublic").prop("checked", !1);
                $("#privacyLevelPrivate").prop("checked", !0);
                if (this.albumPrivacyPassword !== null && this.albumPrivacyPassword !== "") {
                    t.prop("checked", !0);
                    t.prop("disabled", !1);
                    n.val(this.albumPrivacyPassword);
                    n.prop("disabled", !1)
                } else {
                    t.prop("checked", !1);
                    t.prop("disabled", !1);
                    n.val("");
                    n.prop("disabled", !0)
                }
                $("#privacyOptionsCollapsed").show();
                $("#privacyOptions").hide()
            }
            r.on("click", function(e) {
                e.preventDefault();
                var t = r, n = $("#privacyPassword");
                if (n.val() !== "")
                    if (t.text() === "Show") {
                        t.text("Hide");
                        n.attr("type", "text")
                    } else {
                        t.text("Show");
                        n.attr("type", "password")
                    }
            })
        },
        onShow: function(e) {
            var t = Pb.Data.get("isLoggedIn");
            if (!t) {
                location.href = "/login?returnUrl=" + encodeURIComponent(document.location.href);
                return 
            }
            e.preventDefault();
            this.mode = e.mode;
            this.notify = e.notify;
            this.notifyOnTitleChange = e.notifyOnTitleChange;
            this.isTagged = e.isTagged;
            if (Pb.Data.Albums && e.leftRailClick !== "true") {
                var n = Pb.Data.Albums.getCurrentAlbum();
                if (!n || this.mode === "create" && n.getProperty("isTagged")&&!Pb.Data.get("loggedInUserData").isHashtagPartner)
                    n = Pb.Data.Albums.getRootAlbum();
                var r = n.hydrate(), i = this;
                $.when(r).then(function() {
                    i.parentAlbumPath = n.getProperty("path");
                    i.albumPath = n.getProperty("path");
                    i.albumTitle = n.data.titleForTemplate.text;
                    i.albumDescription = n.data.description.text;
                    i.albumPrivacyLevel = n.getProperty("privacyType");
                    i.albumPrivacyPassword = n.getProperty("privacyPassword");
                    i.albumVanity = n.getProperty("albumVanity");
                    i.albumIsRoot = n.getProperty("isRootAlbum");
                    i.isTagged = n.getProperty("isTagged");
                    i.hashtag = n.getProperty("hashtag");
                    if (i.hashtag) {
                        var t = i.hashtag.active ? "on": "";
                        e.modalData = $.extend(e.modalData, {
                            toggleHashtag: {
                                on: t,
                                id: "hashtagToggle",
                                name: "hashtagToggle"
                            }
                        })
                    }
                    e.modalData = $.extend(e.modalData, {
                        hashtag: i.hashtag,
                        isTagged: i.isTagged
                    });
                    i.callSuperShow(e)
                })
            } else {
                this.parentAlbumPath = e.parentAlbumPath;
                this.albumPath = e.albumPath;
                this.albumTitle = e.albumTitle;
                this.albumDescription = e.albumDescription;
                this.albumPrivacyLevel = e.albumPrivacyLevel;
                this.albumPrivacyPassword = e.albumPrivacyPassword;
                this.albumVanity = e.albumVanity;
                this.albumIsRoot = e.albumIsRoot;
                this.isTagged = e.isTagged;
                this.hashtag = e.hashtag;
                if (this.hashtag) {
                    var s = this.hashtag.active ? "on": "";
                    e.modalData = $.extend(e.modalData, {
                        toggleHashtag: {
                            on: s,
                            id: "hashtagToggle",
                            name: "hashtagToggle"
                        }
                    })
                }
                e.modalData = $.extend(e.modalData, {
                    hashtag: this.hashtag,
                    isTagged: this.isTagged
                });
                this.callSuperShow(e)
            }
        },
        callSuperShow: function(t) {
            if (this.mode === "create") {
                this.albumTitle = "";
                this.albumDescription = "";
                this.albumPrivacyPassword = "";
                this.albumVanity = "";
                this.albumIsRoot=!1;
                t.headerTitle = "Create New Album";
                e.superclass.onShow.call(this, t);
                $("#editAlbumButton").hide();
                $(".warning").hide();
                $("#createAlbumButton").show();
                $(".createHashtagSectionContent").show();
                $(".editHashtagSectionContent").hide()
            } else if (this.mode === "privacy") {
                t.headerTitle = "Edit Privacy Settings";
                e.superclass.onShow.call(this, t);
                $(".editAlbumTitle").hide();
                $(".editAlbumDescription").hide();
                $("#createAlbumButton").hide();
                $("#editAlbumButton").show()
            } else {
                this.albumIsRoot ? t.headerTitle = "Edit Library Settings" : t.headerTitle = "Edit Album Settings";
                e.superclass.onShow.call(this, t);
                this.albumIsRoot ? $(".editHashtagSectionContent").hide() : $(".editHashtagSectionContent").show();
                $("#createAlbumButton").hide();
                $("#editAlbumButton").show();
                $(".createHashtagSectionContent").hide();
                new Pb.Component.Element.Toggle.Toggle({
                    container: "#hashtagToggle"
                });
                $("#hashtagToggle").on("Pb.Component.Element.Toggle.Toggle::on", $.proxy(this.toggle, this)).on("Pb.Component.Element.Toggle.Toggle::off", $.proxy(this.toggle, this));
                $("#deleteHashtag").on("click", $.proxy(this.deleteHashtag, this))
            }
            var n = $("#editAlbumChooser");
            n.on("AlbumChooserDropdown::albums_loading", $.proxy(this.onAlbumsLoading, this));
            n.on("AlbumChooserDropdown::albums_loaded", $.proxy(this.onAlbumsLoaded, this));
            n.on("AlbumChooserDropdown::album_selected", $.proxy(this.onAlbumSelected, this));
            this.chooser = new Pb.Component.Album.Chooser.AlbumChooserDropdown({
                albumChooserId: "#editAlbumChooser"
            });
            if (this.mode !== "privacy") {
                $("#editAlbumTitle").val(this.albumTitle);
                if (this.albumIsRoot || (this.isTagged&&!Pb.Data.get("loggedInUserData").isHashtagPartner || this.albumTitle === "Facebook" || this.albumTitle === "Mobile Uploads") && this.mode === "edit") {
                    $("#editAlbumTitle").closest(".editalbum-section").hide();
                    $("#editAlbumTitle").attr("disabled", !0)
                } else {
                    $("#editAlbumTitle").closest(".editalbum-section").show();
                    $("#editAlbumTitle").attr("disabled", !1)
                }
                $("#editAlbumDescription").val(this.albumDescription);
                $("#editAlbumVanity").val(this.albumVanity)
            }
            this.initializePrivacyDisplay();
            this.hideOrShowAlbums();
            this.cleanupErrors()
        },
        toggle: function(e) {
            var t = "/action/hashtag/deactivate", n = "Turning off hashtag", r = this;
            if (e.namespace === "Component.Element.Toggle.Toggle::on") {
                n = "Turning on hashtag";
                t = "/action/hashtag/activate"
            }
            var i = {
                data: {
                    hashtag: this.hashtag.hashtag
                },
                url: t,
                showProgressModal: !0,
                progressTitle: n,
                success: function(t) {
                    if (t && t.status === 1 && t.data) {
                        var i = Pb.Data.Albums.getCurrentAlbum(), s;
                        i.mergeData(t.data);
                        $(document).trigger($.Event("Pb.Component.Album.Edit.EditAlbum::editAlbumCompleted"));
                        t.data.hashtag.active ? r.modalTemplateData[0] = $.extend(r.modalTemplateData[0], {
                            toggleHashtag: {
                                on: "on",
                                id: "hashtagToggle",
                                name: "hashtagToggle"
                            }
                        }) : r.modalTemplateData[0] = $.extend(r.modalTemplateData[0], {
                            toggleHashtag: {
                                on: "",
                                id: "hashtagToggle",
                                name: "hashtagToggle"
                            }
                        })
                    } else if (n === "Turning on hashtag") {
                        $(e.target).find(".toggle").removeClass("on");
                        $(e.target).find(".ballOff").show();
                        $(e.target).find(".ballOn").hide();
                        $(e.target).find(".ball").css("left", 3)
                    } else {
                        $(e.target).find(".toggle").addClass("on");
                        $(e.target).find(".ballOff").hide();
                        $(e.target).find(".ballOn").show();
                        $(e.target).find(".ball").css("left", 35)
                    }
                },
                messageHandler: function(e) {
                    return e.status === Pb.Component.Element.Message.STATUS_SUCCESS?!0 : !1
                }
            };
            return Pb.Ajax.ajax(i)
        },
        deleteHashtag: function() {
            var e = {
                data: {
                    hashtag: this.hashtag.hashtag
                },
                url: "/action/hashtag/delete",
                showProgressModal: !0,
                progressTitle: "Deleting hashtag",
                success: function(e) {
                    if (e && e.status === 1 && e.data) {
                        var t = Pb.Data.Albums.getCurrentAlbum();
                        t.mergeData(e.data);
                        $(document).trigger($.Event("Pb.Component.Album.Edit.EditAlbum::editAlbumCompleted"));
                        $("#editAlbumModal").find(".close").click()
                    }
                },
                messageHandler: function(e) {
                    return e.status === Pb.Component.Element.Message.STATUS_SUCCESS?!0 : !1
                }
            };
            Pb.Ajax.ajax(e)
        }
    };
    Pb.subclass(e, Pb.Component.Element.Modal, t);
    var n = new e;
    n.init();
    return n
}();
$.getObject("Pb.Component.Album.Edit", !0).Privacy = function() {
    function e(e) {
        this.init(e)
    }
    var t = {
        init: function(e) {
            this.initEvents()
        },
        initEvents: function() {
            $(document).on("click", "#editAlbumModalWrapper #privacyEditButton", $.proxy(this.onShow, this));
            $(document).on("change", "#editAlbumModalWrapper input:radio[name=privacyLevel]", $.proxy(this.onRadioChange, this));
            $(document).on("change", "#editAlbumModalWrapper #privacyPasswordCheckbox", $.proxy(this.onCheckboxChange, this));
            $(document).on("click", "#editAlbumModalWrapper #editAlbumButton", $.proxy(this.validatePw, this));
            $(document).on("blur", "#editAlbumModalWrapper #privacyPassword", $.proxy(this.validatePw, this));
            $(document).on("keyup", ".error #privacyPassword", $.proxy(this.validatePw, this))
        },
        onShow: function() {
            $("#privacyOptionsCollapsed").hide();
            $("#privacyOptions").show()
        },
        onCheckboxChange: function(e) {
            var t = $("#privacyPassword"), n = $("#privacyPasswordCheckbox");
            if (n.is(":checked")) {
                t.prop("disabled", !1);
                t.val("")
            } else {
                $(".passwordContainer").removeClass("error");
                t.prop("disabled", !0);
                t.val("")
            }
        },
        onRadioChange: function(e) {
            var t = e.target.value, n = $("#privacyPassword"), r = $("#privacyPasswordCheckbox");
            if (t !== $("#privacyLevelPrivate").val()) {
                r.prop("disabled", !0);
                r.prop("checked", !1);
                n.prop("disabled", !0);
                n.val("")
            } else {
                r.prop("disabled", !1);
                n.prop("disabled", !0)
            }
        },
        validatePw: function(e) {
            var t = $("#privacyPassword"), n = t.parent();
            n.removeClass("error");
            if ($("#privacyPasswordCheckbox").attr("checked") === "checked") {
                var r = new Pb.InputValidator, i = t.val(), s = r.validPassword(i);
                if (!s) {
                    n.addClass("error");
                    e.preventDefault();
                    e.stopPropagation()
                }
            }
        }
    };
    Pb.subclass(e, Pb.InputValidator, t);
    var n = new e;
    n.init();
    return n
}();
$.getObject("Pb.Component.Album.Chooser", !0).AlbumChooserDropdown = function() {
    function e(e) {
        this.init(e)
    }
    $.extend(e.prototype, {
        albumChooserId: null,
        listSelector: null,
        dropdownSelector: null,
        spanSelector: null,
        anchorSelector: null,
        filter: null,
        element: null,
        init: function(e) {
            e.filter && (this.filter = e.filter);
            this.albumChooserId = e.albumChooserId;
            this.element = $(this.albumChooserId);
            this.listSelector = this.albumChooserId + " .albumChooserList";
            this.dropdownSelector = this.albumChooserId + " .dropdown";
            this.spanSelector = this.albumChooserId + " a[data-toggle] > span";
            this.anchorSelector = this.albumChooserId + " a[data-toggle]";
            this.initEvents()
        },
        initEvents: function() {
            if ($(this.listSelector).length) {
                var e;
                Pb.Data.has("albumChooser.data." + this.element.attr("id")) ? e = Pb.Data.get("albumChooser.data." + this.element.attr("id")) : e = Pb.Data.add("albumChooser.data." + this.element.attr("id"), {});
                Pb.JSTemplates.link("#albumChooserDropdownItemTemplate", this.listSelector, e)
            }
            this.bindElements()
        },
        bindElements: function() {
            this.element.on("click", ".dropdown > .dropdown-menu li a.available", $.proxy(function(e) {
                this.onChooserClick(e)
            }, this));
            this.element.on("click", ".dropdown .createAlbum", $.proxy(this.createNewAlbum, this));
            $(document).on("EditAlbum::createAlbumCompleted", $.proxy(this.onAlbumCreated, this))
        },
        onChooserClick: function(e) {
            e.preventDefault();
            var t = $(e.currentTarget), n = {
                albumpath: t.attr("data-albumpath"),
                albumshortpath: t.attr("data-albumshortpath"),
                albumurl: t.attr("data-albumurl"),
                name: t.attr("data-name"),
                displayName: t.html(),
                privacy: t.attr("data-privacy")
            };
            if (t.hasClass("disabled"))
                return;
            this.setSelectedAlbum(n.albumpath);
            this.triggerSelectedAlbumEvent(n)
        },
        setSelectedAlbum: function(e) {
            if (e) {
                e.match(/\/$/) && (e = e.substring(0, e.length - 1));
                var t = $(this.listSelector).find('a[data-albumpath="' + e + '"]');
                if (t.length) {
                    $(this.spanSelector).html(t.html());
                    $(this.anchorSelector).attr("data-location", t.attr("data-location"));
                    $(this.anchorSelector).attr("data-albumpath", e)
                }
            } else {
                $(this.spanSelector).html("Select Album");
                $(this.anchorSelector).attr("data-location", "");
                $(this.anchorSelector).attr("data-albumpath", "")
            }
        },
        triggerSelectedAlbumEvent: function(e) {
            var t = $.Event("AlbumChooserDropdown::album_selected");
            t.albumData = e;
            this.element.trigger(t)
        },
        loadAlbums: function(e) {
            if (Pb.Data.get(this.fetchedKey(e))) {
                this.syncData(Pb.Data.get(this.dataKey(e)));
                return 
            }
            this.showProgress();
            var t = "/component/Album-Chooser-AlbumChooserDropdown";
            this.element.trigger($.Event("AlbumChooserDropdown::albums_loading"));
            var n = {};
            this.filter !== null && (n.filter = this.filter);
            typeof e != "undefined" && (n.path = e);
            if (this.albumChooserId === "#editAlbumChooser" || this.albumChooserId === "#moveAlbumAlbumChooser")
                n.editCreateMode=!0;
            var r = {
                url: t,
                data: n,
                success: $.proxy(function(t) {
                    this.onAlbumsLoadSuccess(t, e)
                }, this),
                error: $.proxy(this.onAlbumsLoadFail, this),
                type: "get"
            };
            Pb.Ajax.ajax(r)
        },
        onAlbumsLoadSuccess: function(e, t) {
            this.hideProgress();
            Pb.Data.add(this.fetchedKey(t), !0);
            Pb.Data.add(this.dataKey(t), e);
            this.syncData(e)
        },
        onAlbumsLoadFail: function(e) {
            this.hideProgress();
            typeof console != "undefined" && console.log(e)
        },
        syncData: function(e) {
            Pb.JSTemplates.link("#albumChooserDropdownItemTemplate", this.listSelector, e);
            this.element.show().trigger($.Event("AlbumChooserDropdown::albums_loaded"))
        },
        createNewAlbum: function(e) {
            var t = $(this.anchorSelector).attr("data-albumpath"), n = {
                type: "Pb.Component.Album.Edit::show",
                mode: "create",
                notify: !0
            };
            t && (n.parentAlbumPath = t);
            $(document).trigger(n)
        },
        onAlbumCreated: function(e, t) {
            var n = Pb.Data.get(this.dataKey()), r = t.path.substring(0, t.path.lastIndexOf("/")), i = $(this.listSelector).find('a[data-albumpath="' + r + '"]').parent(), s, o, u, a = 0, f, l, c = 0;
            if (i.length && n) {
                s = $(this.listSelector + " li").index(i);
                if (s < n.length) {
                    c = parseInt(n[s].optionClass.match(/depth(\d)+/, "")[1], 10) + 1;
                    for (f = s + 1; f < n.length; f++)
                        if (n[f].optionClass) {
                            l = n[f].optionClass.match(/depth(\d)+/);
                            if (l === null)
                                break;
                                if (!(parseInt(l[1], 10) >= s))
                                    break;
                                    a++
                        }
                    o = n[s].optionClass.replace(/depth(\d)+/, "depth" + c);
                    u = {
                        displayName: t.dirName,
                        name: t.dirName,
                        optionClass: o,
                        path: t.path,
                        shortPath: t.location,
                        url: t.url
                    };
                    $.observable(n).insert(s + a + 1, u);
                    var h = "private";
                    t.isPublic ? h = "public" : t.hasVisitorPassword && (h = "password");
                    this.setSelectedAlbum(t.path);
                    this.triggerSelectedAlbumEvent({
                        albumpath: t.path,
                        albumshortpath: t.location,
                        albumurl: t.url,
                        displayName: t.dirName,
                        name: t.dirName,
                        privacy: h
                    })
                }
            }
        },
        hideProgress: function() {
            $(this.albumChooserId).find(".progress").hide()
        },
        showProgress: function() {
            $(this.albumChooserId).find(".progress").show()
        },
        fetchedKey: function(e) {
            var t = "albumChooser.fetched." + this.element.attr("id");
            this.filter && (t += this.filter);
            e && (t += e);
            return t
        },
        dataKey: function(e) {
            var t = "albumChooser.data." + this.element.attr("id");
            this.filter && (t += this.filter);
            e && (t += e);
            return t
        }
    });
    return e
}();
$.getObject("Pb.Component.Common.Login", !0).LoginModal = function() {
    function e(t) {
        e.superclass.constructor.call(this)
    }
    $.extend(e.prototype, {
        init: function(t) {
            var n = {
                eventName: "LoginModal::show",
                modalId: "#loginModal",
                modalName: "loginModal"
            };
            t && $.extend(n, t);
            this.authReturnTargetKey = "authReturnTarget";
            this.authReturnTargetEventName = "LoginModal::authReturnTarget";
            e.superclass.init.call(this, n);
            this.initEvents(n)
        },
        initEvents: function(e) {
            $(document).trigger($.Event("ActionEventsDispatcher::registerevent", {
                pbEvent: {
                    eventSelector: ".showLoginModal",
                    eventType: "click",
                    event: e.eventName
                }
            }));
            this.initAuthReturnTargetAction()
        },
        initAuthReturnTargetAction: function() {
            this.hasAuthReturnTargetCookie() && $(document).on(this.authReturnTargetEventName, $.proxy(function() {
                var e = "#" + Pb.Utilities.cookie(this.authReturnTargetKey);
                if ($(e).length > 0) {
                    $(e).click();
                    Pb.Utilities.cookie(this.authReturnTargetKey, null)
                }
            }, this))
        },
        hasAuthReturnTargetCookie: function() {
            var e = Pb.Utilities.cookie(this.authReturnTargetKey);
            return e !== null?!0 : !1
        },
        onShow: function(t) {
            var n = $(t.pbEventTarget), r, i, s, o = $(n).data("authreturntarget");
            this.hasAuthReturnTargetCookie() && Pb.Utilities.cookie(this.authReturnTargetKey, null);
            o !== undefined && Pb.Utilities.cookie(this.authReturnTargetKey, o, {
                expires: ""
            });
            var u = Pb.Data.get(this.modalName);
            u.loginBox.originalIframeUrl || (u.loginBox.originalIframeUrl = u.loginBox.iframeUrl);
            u.loginBox.iframeUrl = u.loginBox.originalIframeUrl;
            i = Pb.Data.get("authReturnUrl");
            i !== null && (u.loginBox.iframeUrl += "&returnUrl=" + encodeURIComponent(i));
            e.superclass.onShow.call(this, t)
        }
    });
    Pb.subclass(e, Pb.Component.Element.Modal, e.prototype);
    var t = new e;
    Pb.InitEventQueue.add(function() {
        t.init()
    });
    return t
}();
(function() {
    function e() {
        var e = Pb.Data.get("authReturnUrl"), n;
        $(document).on("Pb.Component.Navigation.LoginBox::show", t);
        if (e !== null && $("p.signup a").length > 0) {
            n = $("p.signup a").attr("href");
            n.indexOf("returnUrl")===-1 && $("p.signup a").attr("href", n + "?returnUrl=" + encodeURIComponent(e))
        }
    }
    function t() {
        var e = $("#loginBoxFrame");
        e.on("load", function(t) {
            e.css("display", "block")
        });
        e.attr("src") || e.attr("src", e.data("src"))
    }
    Pb.InitEventQueue.add(e, Pb.InitEventQueue.LOW_PRIORITY)
})();
$.getObject("Pb.Component.ExternalSite", !0).Register = function() {
    function t(t) {
        if (t.status === 1)
            t.connectInfo && t.connectInfo.redirect ? location.href = t.connectInfo.redirect : t.connectInfo && t.connectInfo.forwardto ? location.href = t.connectInfo.forwardto : location.href = "/";
        else if (t.connectInfo && t.connectInfo.redirect)
            location.href = t.connectInfo.redirect;
        else if (t.connectInfo&&!t.connectInfo.loggedIn) {
            var n = "";
            t.connectInfo && t.connectInfo.forwardto && (n = "?returnUrl=" + t.connectInfo.forwardto);
            if (e) {
                var r = $.parseJSON(e), i = r.brand + "?token=" + r.brandAffinityToken + "&sku=" + r.brandSku;
                location.href = "/register/" + i + n
            } else 
                location.href = "/register" + n
        } else 
            t.connectInfo && t.connectInfo.forwardto && (location.href = t.connectInfo.forwardto)
    }
    function n(e, n) {
        if (Pb.Data.get("isMobile")) {
            n && Pb.Track.tr(n);
            return !0
        }
        var r = Pb.Data.get("authReturnUrl");
        if (r !== null) {
            var i = $(".register-facebook:first").attr("href");
            i && i.indexOf("forwardto")===-1 && $(".register-facebook").attr("href", i + "&forwardto=" + encodeURIComponent(r));
            var s = $(".register-twitter:first").attr("href");
            s && s.indexOf("forwardto")===-1 && $(".register-twitter").attr("href", s + "&forwardto=" + encodeURIComponent(r))
        }
        $(document).one("Pb.Component.ExternalSite.Connect::connect", t);
        Pb.Component.ExternalSite.Connect.connect(e.href, !0);
        n && Pb.Track.tr(n);
        return !1
    }
    var e = $.cookie("pb_ba_reg_params");
    Pb.InitEventQueue.add(function() {
        $.views.helpers({
            getExternalSiteRegisterClickHandler: function(e) {
                e || (e = "");
                return 'Pb.Component.ExternalSite.Register.register(this, "' + e + '");return false;'
            }
        })
    });
    return {
        register: n
    }
}();
$.getObject("Pb.Component.ExternalSite", !0).Connect = function() {
    function n(n, r) {
        e = open(n, "auth", "width=650,height=350,scrollbars=no,location=no,modal=yes,alwaysRaised=yes");
        if (!e)
            return !1;
        e.focus();
        r && (t = setInterval(function() {
            var n = Pb.Data.get("connectResponse");
            if (n) {
                clearInterval(t);
                r.call(this);
                e.close()
            } else if (e.closed) {
                clearInterval(t);
                r.call(this)
            }
        }, 300));
        return !0
    }
    function r(e) {
        e.messages && e.messages.length > 0 && Pb.Component.Element.Message.show(e.messages[0]);
        $(document).trigger({
            type: "Pb.Component.ExternalSite.Connect::connect",
            connectInfo: e.data,
            status: e.status
        })
    }
    function i(e, t) {
        Pb.Component.Element.Message.remove();
        Pb.Data.add("connectResponse", "");
        n(e, function() {
            var e = Pb.Data.get("connectResponse");
            if (e) {
                e.status === 0 && Pb.Track.tr("regpage_externallogin_view");
                if (!t && e.status && e.data.siteAuth && e.data.siteAuth.siteId === 2 && e.data.siteAuth.publishState.toLowerCase() !== "auto") {
                    $(document).one("AutoPublishModal::hide", $.proxy(function() {
                        r(e)
                    }, this)).one("AutoPublishModal::success", $.proxy(function() {
                        e.data.siteAuth.publishState = "Auto"
                    }, this));
                    Pb.Component.ExternalSite.AutoPublishModal.setConnected();
                    Pb.Component.ExternalSite.AutoPublishModal.onShow()
                } else 
                    r(e)
            } else 
                $(document).trigger({
                    type: "Pb.Component.ExternalSite.Connect::connect",
                    status: 0
                });
            e && e.data && e.data.profilePic && Pb.Component.Common.ProfilePic.ProfilePic.update(e.data.profilePic)
        })
    }
    function s(e) {
        e.messages && e.messages.length > 0 && Pb.Component.Element.Message.show(e.messages[0]);
        $(document).trigger({
            type: "Pb.Component.ExternalSite.Connect::connect",
            connectInfo: e.data,
            status: e.status
        })
    }
    function o(e, t) {
        Pb.Component.Element.Message.remove();
        Pb.Data.add("connectResponse", "");
        n(e, function() {
            var e = Pb.Data.get("connectResponse"), n, r, i;
            if (e && e.status && e.data && e.data.permissions) {
                for (n = 0; n < t.length; n++) {
                    r=!1;
                    for (i in e.data.permissions)
                        e.data.permissions.hasOwnProperty(i) && i === t[n] && (r=!0);
                    r || $(document).trigger({
                        type: "Pb.Component.ExternalSite.Connect::permissions",
                        status: 0
                    })
                }
                $(document).trigger({
                    type: "Pb.Component.ExternalSite.Connect::permissions",
                    connectInfo: e.data,
                    status: e.status
                });
                return 
            }
            $(document).trigger({
                type: "Pb.Component.ExternalSite.Connect::permissions",
                status: 0
            })
        })
    }
    var e, t;
    return {
        connect: i,
        getPermissions: o
    }
}();
$.getObject("Pb.Component.ExternalSite", !0).AutoPublishModal = function() {
    function e() {
        e.superclass.constructor.call(this)
    }
    if (!Pb.Component.Element ||!Pb.Component.Element.Modal)
        return null;
    var t, n = {
        init: function(t) {
            var n = {
                shownEvent: "AutoPublishModal::shown",
                modalId: "#autoPublishModal",
                modalName: "autoPublishModal",
                submitButtonId: "#autoPublishYesButton",
                submitUrl: "/action/externalSite/updateAutoShare",
                hideEvent: "AutoPublishModal::hide",
                showProgress: !0
            };
            t && $.extend(n, t);
            e.superclass.init.call(this, n);
            $(document).on("DisconnectConfirm::success", $.proxy(function(e) {
                var t = $(this.modalId).find("form [name=siteId]").val(), n = Pb.Data.get(this.modalName);
                if (e.siteId === t) {
                    n.sites[t]=!1;
                    $(this.submitButtonId).attr("onclick", "Pb.Component.ExternalSite.AutoPublishModal.connectToService(this); return false;")
                }
            }, this));
            $(document).on("Pb.Component.ExternalSite.Connect::connect", $.proxy(function(e) {
                var t = $(this.modalId).find("form [name=siteId]").val();
                e.status === 1 && e.connectInfo.siteId === t && this.setConnected()
            }, this))
        },
        setConnected: function() {
            $(this.submitButtonId).data("connected", !0).attr("onclick", "")
        },
        onHide: function(t) {
            e.superclass.onHide.call(this, t);
            Pb.Track.tr("autopost_modal_close");
            var n = Pb.Data.get(this.modalName)
        },
        onShow: function(t) {
            t || (t = {});
            var n = Pb.Data.get(this.modalName), r = n.siteId;
            if (n && n.sites && n.sites[r])
                return;
            e.superclass.onShow.call(this, t);
            $(this.submitButtonId).on("click", $.proxy(this.onSubmit, this));
            Pb.Track.tr("autopost_modal_view")
        },
        buildSubmitData: function() {
            var e = $(this.modalId).find("form"), t = {
                siteId: e.find("[name=siteId]").val(),
                autoshare: "true"
            };
            return t
        },
        onSuccess: function() {
            $(document).trigger("AutoPublishModal::success");
            var e = Pb.Data.get(this.modalName), t = $(this.modalId).find("form [name=siteId]").val();
            e.sites[t]=!0;
            $("#autoShareUploadCbfacebook").length && $("#autoShareUploadCbfacebook").prop("checked", !0)
        },
        onFail: function(e) {
            if (e && e.data && e.data.requiredPermissions) {
                var t = $(this.submitButtonId);
                if (t.length) {
                    $(document).one("Pb.Component.ExternalSite.Connect::permissions", $.proxy(function(e) {
                        e.status === 1 ? this.submit() : Pb.Component.Element.Message.show({
                            status: Pb.Component.Element.Message.STATUS_ERROR,
                            message: "You must grant us permission to publish to your Timeline."
                        })
                    }, this));
                    Pb.Component.ExternalSite.Connect.getPermissions(t.attr("href"), e.data.requiredPermissions)
                }
            }
        },
        onSubmit: function(t) {
            t.preventDefault();
            $(t.currentTarget).data("connected") && e.superclass.onSubmit.call(this, t)
        },
        connectToService: function(e) {
            $(document).one("Pb.Component.ExternalSite.Connect::connect", $.proxy(this.connectHandler, this));
            Pb.Component.ExternalSite.Connect.connect(e.href, !0)
        },
        connectHandler: function(e) {
            e.status === 1 && this.submit()
        }
    };
    Pb.subclass(e, Pb.Component.Element.Modal, n);
    t = new e;
    Pb.InitEventQueue.add(function() {
        t.init()
    });
    return t
}();
(function() {
    function t() {
        typeof MBP != "undefined" && MBP.hideUrlBarOnLoad();
        e = Pb.Data.get(Pb.Data.Shared.DATA);
        !e && $(".lowerHeader").length ? n() : $(".sign-up").css("display", "none");
        $("#mobileCloseButton").click(function() {
            var t = $(".dropdown-search");
            t.is(":visible") ? Pb.Track.tr("loggedin_nav_menu_search_close_click") : Pb.Track.tr("nav_menu_close_click");
            $("#mobileCloseButton").css("display", "none");
            $("#mobileMenuButton").css("display", "inline");
            $(".dropdown-search").css("display", "none");
            $(".dropdown-loggedIn").css("display", "none");
            $(".dropdown-loggedOut").css("display", "none");
            e || $(".lowerHeader").length === 0 ? $("header").css("height", "52px") : $("header").css("height", "100px")
        });
        $("#mobileMenuButton").click(function() {
            $("#mobileMenuButton").css("display", "none");
            $(".dropdown-search").css("display", "none");
            $(".dropdown-loggedIn").css("display", "inline");
            $(".dropdown-loggedOut").css("display", "inline");
            $("#mobileCloseButton").css("display", "inline");
            $("#mobileCloseButton").css("right", "auto");
            $("#mobileCloseButton").css("left", "10px");
            e ? $("header").css("height", "370px") : $("header").css("height", "375px")
        });
        $("#searchBox").on("focus", function() {
            Pb.Track.tr("nav_search_focus")
        });
        var t = Pb.Utilities.cookie("PB_JSENABLED");
        if (!t) {
            var r = new Date;
            r.setTime(r.getTime() + 31536e6);
            Pb.Utilities.cookie("PB_JSENABLED", "1", {
                expires: r
            })
        }
    }
    function n() {
        $("nav.navbar").css("height", "100px")
    }
    var e = null;
    Pb.InitEventQueue.add(t, 1)
})();
$(document).ready(function() {
    Pb.Form.addHash("#logoutForm");
    $(document).on("click", ".logoutLink", function() {
        $("#logoutForm").submit()
    })
});
$.getObject("Pb.Component.Editor", !0).Aviary = function() {
    function r() {
        var e = $("body").contents().find(".srcbannerAd").first(), t = $(e).text(), n = $("body").contents().find(".bannerAd").first(), r = $('<iframe style="width: 728px; height:90px" class="bannerAdIframe" src="' + t + '" frameborder="0" scrolling="no" marginwidth="0" marginheight="0"></iframe>'), i = $("body").contents().find(".srcskyscraperAd").first(), s = $(i).text(), o = $("body").contents().find(".skyscraperAd").first(), u = $('<iframe style="width: 160px; height:600px" class="skyscraperAdIframe" src="' + s + '" frameborder="0" scrolling="no" marginwidth="0" marginheight="0"></iframe>');
        e.length && ($(".bannerAdIframe").length || n.append(r));
        o.length && ($(".skyscraperAdIframe").length || o.append(u))
    }
    function i() {
        if (this.fileChooserInited)
            return;
        var e = $("#aviaryEditor");
        if (window.FileReader && Modernizr.draganddrop) {
            e.addClass("upload");
            $(document).on("click", ".chooseLink", function(e) {
                e.stopImmediatePropagation();
                $("#chooseFiles").click()
            }).on("click", ".editLink", function(t) {
                e.removeClass("postupload")
            }).on("click", ".postSignupLink", o).on("click", ".postLoginLink", o);
            $.inArray("dataTransfer", $.event.props)===-1 && $.event.props.push("dataTransfer");
            e.on("dragover", $.proxy(C, this)).on("dragleave", $.proxy(k, this)).on("drop", $.proxy(x, this));
            e.find("input[type=file]").on("change keyup blur", $.proxy(S, this))
        } else 
            Pb.Data.get("isLoggedIn") ? e.addClass("nodragdrop") : e.addClass("notsupported");
        $(document).on("EditLink::Editor", $.proxy(function(e) {
            e.preventDefault();
            var t = e.pbData;
            $("#editableImage").attr("src", t.fullsizeUrl);
            this.settings.filename = t.name;
            this.settings.filetype = /[^.]+$/.exec(t.name);
            this.settings.url = t.fullsizeUrl;
            this.settings.isOwner=!0;
            this.settings.location = t.location;
            this.settings.hiresUrl = t.orig;
            this.initEditor()
        }, this));
        this.fileChooserInited=!0
    }
    function s() {
        this.initFileChooser();
        if (!Pb.Component.Editor.Aviary.settings.hasPermission) {
            Pb.Component.Element.Message.show({
                status: 0,
                message: "That file is not available for editing."
            });
            Pb.Component.Editor.Aviary.settings.hasPermission=!0;
            return 
        }
        if (!e.url) {
            this.loadAds();
            return 
        }
        $(".sponsorSkinClick").height($(window).height() - $(".navbar-inner").height());
        e.filetype = /[^.]+$/.exec(e.filename);
        t = (new Date).getTime();
        $(".aviaryEditorContainer").addClass("loading");
        a(e.hiresUrl, function(t) {
            var i = new Date;
            b("Aviary initialized", i, 0);
            n = new Aviary.Feather({
                apiKey: e.apiKey,
                salt: e.salt,
                timestamp: e.timestamp,
                encryptionMethod: "sha1",
                signature: e.signature,
                isPremiumPartner: 1,
                apiVersion: 3,
                theme: "light",
                tools: "all",
                appendTo: "aviaryEditor",
                noCloseButton: !0,
                displayImageSize: !1,
                initTool: e.tool,
                track: e.track,
                uploadDesc: e.uploadDesc,
                enableCORS: !0,
                onSave: function(e, t) {
                    this.onSaveHiRes(e, t)
                },
                onSaveHiRes: function(e, t) {
                    Pb.Data.get("isLoggedIn") ? Pb.Track.tr("editor_save_success") : Pb.Track.tr("editor_save_loggedout_success");
                    var n = new Date;
                    b("Aviary save END", n, 60);
                    return u(t)
                },
                onSaveButtonClicked: function() {
                    var e = $("#saveTypeCheckbox").is(":checked") ? "replace": "copy";
                    Pb.Data.get("isLoggedIn") || (e = "download");
                    $(document).trigger($.Event("Pb.Component.Editor.SaveModal::selected", {
                        saveType: e
                    }));
                    var t = new Date;
                    b("Aviary save START", t, 40);
                    return !1
                },
                onError: function(e) {
                    var t;
                    switch (e.code) {
                    case 1:
                        t = "An error occured while loading your photo. Want to give it another shot? ";
                        break;
                    case 2:
                        t = "We were unable to detect a Flash plugin or HTML5 Canvas support. Please try editing in another browser.";
                        break;
                    case 4:
                        t = "We're sorry, the tool you've selected is unavailable. Please try another editing tool.";
                        break;
                    case 3:
                    case 5:
                    case 6:
                        t = "We were unable to access your photo. Please try editing another photo";
                        break;
                    case 7:
                        t = "An error occurred while saving your photo. Please click 'Save' again.";
                        break;
                    default:
                        t = Pb.Component.Element.Message.defaultErrorMessage.message
                    }
                    Pb.Track.tr("editor_error_" + e.code);
                    Pb.Component.Element.Message.show({
                        status: 0,
                        message: t
                    });
                    $("#ajaxProgressModalSave").modal("hide");
                    $(".aviaryEditorContainer").removeClass("loading")
                },
                onReady: function() {
                    $(".aviaryEditorContainer").removeClass("loading");
                    r()
                },
                onLoad: function() {
                    function t(e, t, n) {
                        Pb.Modal.ModalProgressBar.updateProgressIndicator("#ajaxProgressModalSave", n);
                        if (typeof console != "undefined") {
                            var r = e + " (" + window.location + ")";
                            window.jsStartTime !== undefined && (r += " " + (t.getTime() - window.jsStartTime.getTime() + "ms"));
                            console.log(r)
                        }
                    }
                    var r = e.width > e.height ? e.width: e.height, i = r > 800 ? r: 800, s = {
                        image: "editableImage",
                        url: e.url
                    }, o = new Date, u, a = ["bow", "goldframe", "happynewyear", "happyyear", "merryny", "merryxmas", "ornaments", "reframe", "seasongreetings", "snowflakes"], f = [];
                    if (Pb.Data.get("version") === "development") {
                        a.push("aviators");
                        a.push("disguise")
                    }
                    t("Aviary editor onload ", o, 0);
                    e.fileFormat && $.extend(s, {
                        fileFormat: e.fileFormat
                    });
                    e.hiresUrl && $.extend(s, {
                        maxSize: i,
                        hiresUrl: e.hiresUrl,
                        hiresHeight: e.height,
                        hiresWidth: e.width,
                        hiresMaxSize: i
                    });
                    n.launch(s);
                    o = new Date;
                    t("Aviary editor called launch ", o, 0);
                    n.on("usage:tool", function(t, n, r) {
                        if (n === "applied") {
                            Pb.Track.tr("editor_" + t + "_applied");
                            if (t === "overlay") {
                                u = r.split(",");
                                f = $(u).filter(a);
                                if (f.length) {
                                    e.sponsoredEdit=!0;
                                    $.each(f, function(e, t) {
                                        Pb.Track.tr("editor_sticker_" + t + "_applied")
                                    })
                                }
                            }
                        }
                    });
                    e.filetype && e.filetype[0] === "jpg" && e.isOwner && $("#avpw_save_button").parent().append('<div id="saveType"><input type="checkbox" id="saveTypeCheckbox"/> <span id="saveTypeCheckboxLabel">Replace Original</span></div>');
                    $("#avpw_save_button").addClass("btn btn-primary");
                    $("#avpw_apply_container").addClass("btn btn-primary").on("touchstart", function(e) {
                        e.stopPropagation();
                        e.preventDefault()
                    });
                    var l = $(window);
                    $(".editorContainer").css("height", l.height() - 200 - $(".header").height() - $(".bannerAd").height());
                    l.resize(function() {
                        var e = $(".editorContainer").height(), t = l.height() - 200 - $(".header").height() - $(".bannerAd").height();
                        if (e !== t) {
                            $(".editorContainer").css("height", t);
                            $(".sponsorSkinClick").height(l.height() - $(".navbar-inner").height())
                        }
                    })
                }
            })
        });
        $("html").hasClass("is_ie") || $(window).on("beforeunload.aviaryPageUnload", g);
        $(".editorContainer").on("mouseup", v);
        $(document).on("Pb.Component.Editor.SaveModal::selected", m);
        $("#saveEditedImageForm").on("submit", function() {
            Pb.Modal.ModalProgressBar.updateProgressIndicator("#ajaxProgressModalSave", 100)
        })
    }
    function o(t) {
        t.preventDefault();
        var n = $(t.currentTarget).hasClass("postLoginLink") ? "login": "register";
        Pb.Track.tr("editor_" + n + "_postedit_click");
        e.files.length && Pb.Utilities.cookie("pbedits", JSON.stringify(e.files), {
            domain: ".photobucket.com",
            path: "/",
            expires: null
        });
        document.location = t.currentTarget.href
    }
    function u(e) {
        $("#savedImageUrl").val(e);
        $(window).off(".aviaryPageUnload");
        var t = new Date;
        b("Pb Upload START", t, 80);
        f(e);
        return !1
    }
    function a(t, n) {
        e.height && e.width && n({
            width: e.width,
            height: e.height
        });
        var r = new Image;
        t || n({
            width: 0,
            height: 0
        });
        r.onload = function() {
            n({
                width: r.width,
                height: r.height
            })
        };
        r.onerror = function() {
            n({
                width: 0,
                height: 0
            })
        };
        r.src = t
    }
    function f(t) {
        if (e.saveType === "download" ||!Pb.Data.get("isLoggedIn")) {
            var r = /[^.]+$/.exec(t)[0], i = /[^.]+$/.exec(e.filename)[0], s = e.filename.replace("." + i, "." + r);
            e.files.push(t);
            $("#ajaxProgressModalSave").modal("hide");
            $("#downloadLink, #saveLink, #thumbnailLink").attr("href", t);
            $("#downloadLink, #saveLink, #thumbnailLink").attr("download", "pb_" + s);
            $("#thumbnailLink").find("img").remove();
            $("#thumbnailLink").append('<img src="' + t + '" />');
            $("#aviaryEditor").addClass("postupload");
            n.close();
            return 
        }
        var o = {
            url: t,
            location: e.location,
            isEdit: !0
        };
        if (e.saveType !== "copy") {
            o.overwrite=!0;
            o.name = e.filename
        }
        e.uploadDesc && (o.desc = e.uploadDesc);
        Pb.Ajax.ajax({
            url: "/action/upload/url",
            data: o,
            success: function(t) {
                if (t.status === 1) {
                    e.saveType === "copy" && l();
                    Pb.Track.tr("editor_upload_success");
                    var n = new Date;
                    b("Pb Upload END", n, 100);
                    window.location = t.data.media.linkUrl + "?state=" + e.saveType + "&sp=" + e.sponsoredEdit
                } else {
                    Pb.Track.tr("editor_upload_error");
                    c()
                }
            },
            error: function() {
                Pb.Track.tr("editor_upload_error");
                c()
            },
            messageHandler: function() {
                return !0
            }
        })
    }
    function l() {
        $("#ajaxProgressModalSave").modal("hide");
        Pb.Component.Element.Message.addMessageToCookie(3, e.success)
    }
    function c() {
        $("#ajaxProgressModalSave").modal("hide");
        Pb.Component.Element.Message.show({
            status: 0,
            message: e.failure
        })
    }
    function h() {
        $("#ajaxProgressModalSave").modal("hide");
        Pb.Component.Element.Message.show({
            status: 0,
            message: "Please select an album. Try again now."
        })
    }
    function p() {
        $("#ajaxProgressModalSave").modal("hide");
        Pb.Component.Element.Message.show({
            status: 0,
            message: "Hmmm. Something didn't click. You image could not be loaded.  Try again now."
        })
    }
    function d() {
        $("#ajaxProgressModalSave").modal("hide");
        Pb.Component.Element.Message.show({
            status: 0,
            message: "Hmmm. Something didn't click.  Please select a valid image.  Try again now."
        })
    }
    function v(e) {
        if ($(e.target).attr("id") === "avpw_save_button")
            return;
        var n = (new Date).getTime(), r = 15;
        if ((n - t) / 1e3 > r) {
            $(".sponsored").removeAttr("style").removeClass("sponsored");
            var i = Pb.Component.Ad.AdRefresh;
            i.refresh(i.AD_SKYSCRAPER, i.AD_SKYSCRAPER);
            i.refresh(i.AD_BANNER, i.AD_BANNER);
            i.refresh(i.AD_SKIN, i.AD_SKIN);
            t = n
        }
    }
    function m(t) {
        e.saveType = t.saveType;
        Pb.Track.tr("editor_save_button_click");
        Pb.Track.tr("editor_save_" + t.saveType);
        e.track && Pb.Track.tr("editor_save_" + e.track);
        Pb.Ajax.ajax({
            url: "/component/Editor-AviaryImageEditor",
            type: "GET",
            success: function(r) {
                n.updateConfig(r);
                e.hiresUrl ? n.saveHiRes() : n.save();
                y(t.saveType)
            }
        })
    }
    function g(e) {
        if (n.getIsDirty())
            return e.returnValue = "This Image has unsaved changes. If you leave this page, your changes will be lost."
    }
    function y(e) {
        Modernizr.canvas || $(".avpw_flash").hide();
        var t = $("#ajaxProgressModalSave"), n;
        if (!t.length) {
            n = e === "replace" ? "Replacing original photo" : "Saving";
            t = $('<div class="modal hide fade no-footer" id="ajaxProgressModalSave" data-backdrop="static" data-keyboard="true"><div class="modal-header"><h2 class="title">' + n + "...</h2></div>" + '<div class="modal-body"></div></div>');
            $(document.body).append(t);
            Pb.Modal.ModalProgressBar.addProgressIndicator("#ajaxProgressModalSave")
        }
        t.modal("show")
    }
    function b(e, t, n) {
        Pb.Modal.ModalProgressBar.updateProgressIndicator("#ajaxProgressModalSave", n);
        if (typeof console != "undefined") {
            var r = e + " (" + window.location + ")";
            window.jsStartTime !== undefined && (r += " " + (t.getTime() - window.jsStartTime.getTime() + "ms"));
            console.log(r)
        }
    }
    function w(e) {
        $(document.body).removeClass("dragOver");
        var t = new FileReader;
        if (!e) {
            d();
            return !1
        }
        if (!T(e.name)) {
            d();
            return !1
        }
        if (!e.type.indexOf("image") === 0) {
            d();
            return !1
        }
        t.onerror = d;
        t.onabort = d;
        t.onload = $.proxy(function(t) {
            Pb.Track.tr("editor_localfile_loaded");
            $("#editableImage").attr("src", t.target.result);
            this.settings.filename = e.name;
            this.settings.url = t.target.result;
            this.settings.fileFormat = e.type.indexOf("jpeg") || e.type.indexOf("jpg") ? "jpg" : "png";
            n ? n.launch({
                url: t.target.result,
                fileFormat: this.settings.fileFormat
            }) : this.initEditor()
        }, this);
        t.readAsDataURL(e)
    }
    function E(e) {
        $(document.body).removeClass("dragOver");
        $(document).one("LocationModal::hide", $.proxy(function() {
            Pb.Data.get("location") || Pb.Data.add("location", "");
            if (!e) {
                d();
                return !1
            }
            if (!T(e.name)) {
                d();
                return !1
            }
            if (!e.type.indexOf("image") === 0) {
                d();
                return !1
            }
            var t = new FormData;
            t.append("file", e);
            t.append("json", !0);
            t.append("location", Pb.Data.get("location"));
            var n = {
                url: "/action/upload/file",
                type: "POST",
                data: t,
                cache: !1,
                json: !0,
                contentType: !1,
                showProgressModal: !0,
                progressTitle: "Uploading...",
                processData: !1,
                success: $.proxy(function(e) {
                    if (e.status === 1) {
                        Pb.Track.tr("editor_localfile_loaded");
                        $("#editableImage").attr("src", e.data.media.fullsizeUrl);
                        this.settings.filename = e.data.media.name;
                        this.settings.url = e.data.media.fullsizeUrl;
                        this.settings.hiresUrl = e.data.media.orig;
                        this.settings.isOwner=!0;
                        this.initEditor()
                    } else 
                        this.showNotLoadedMessage()
                }, this),
                error: $.proxy(this.showNotLoadedMessage, this)
            };
            Pb.Ajax.ajax(n)
        }, this));
        Pb.Component.Editor.LocationModal.onShow()
    }
    function S(e) {
        var t = e.target.files[0];
        Pb.Data.get("isLoggedIn") ? this.uploadFile(t) : this.loadFile(t)
    }
    function x(e) {
        if ($("#__pio_f").length)
            return !1;
        e.stopPropagation();
        e.preventDefault();
        if (e.dataTransfer && e.dataTransfer.files) {
            var t = e.dataTransfer.files[0];
            if (Pb.Data.get("isLoggedIn")) {
                Pb.Track.tr("editor_fileupload_dropped");
                this.uploadFile(t)
            } else {
                Pb.Track.tr("editor_file_dropped");
                this.loadFile(t)
            }
        }
    }
    function T(e) {
        if (!e)
            return !1;
        var t = e.split(".").pop().toLowerCase();
        return N(t)
    }
    function N(e) {
        return $.inArray(e, ["jpg", "jpeg", "gif", "png", "bmp", "tif", "tiff"])===-1?!1 : !0
    }
    function C(e) {
        if ($("#__pio_f").length)
            return !1;
        e.preventDefault();
        e.originalEvent.dataTransfer.dropEffect = "copy";
        var t = e.originalEvent.dataTransfer;
        if (t.types !== null && (t.types.indexOf ? t.types.indexOf("Files")!==-1 : t.types.contains("application/x-moz-file"))) {
            if (!$(document.body).hasClass("dragOver")&&!this.settings.fadingLabel) {
                this.settings.fadingLabel=!0;
                setTimeout($.proxy(function() {
                    setTimeout($.proxy(function() {
                        this.settings.fadingLabel=!1
                    }, this), 200);
                    $(document.body).addClass("dragOver")
                }, this), 200)
            }
            clearTimeout(this.settings.dragTimer)
        }
    }
    function k(e) {
        if ($("#__pio_f").length)
            return !1;
        this.settings.dragTimer = setTimeout($.proxy(function() {
            if ($(document.body).hasClass("dragOver")&&!this.settings.fadingLabel) {
                this.settings.fadingLabel=!0;
                setTimeout($.proxy(function() {
                    setTimeout($.proxy(function() {
                        this.settings.fadingLabel=!1
                    }, this), 200);
                    $(document.body).removeClass("dragOver")
                }, this), 200)
            }
        }, this), 300)
    }
    var e = {
        failure: null,
        success: null,
        filename: null,
        location: null,
        apiKey: null,
        salt: null,
        timestamp: null,
        signature: null,
        saveType: "copy",
        tool: null,
        isOwner: null,
        files: [],
        fileChooserInited: !1,
        dragTimer: null,
        fadingLabel: !1,
        sponsoredEdit: !1
    }, t, n;
    return {
        settings: e,
        initEditor: s,
        initFileChooser: i,
        loadFile: w,
        uploadFile: E,
        loadAds: r
    }
}();
