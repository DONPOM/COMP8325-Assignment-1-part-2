/*
 jQuery JavaScript Library v1.6.4
 http://jquery.com/

 Copyright 2011, John Resig
 Dual licensed under the MIT or GPL Version 2 licenses.
 http://jquery.org/license

 Includes Sizzle.js
 http://sizzlejs.com/
 Copyright 2011, The Dojo Foundation
 Released under the MIT, BSD, and GPL Licenses.

 Amazon elects to use jQuery and Sizzle under the MIT license.

 Date: Mon Sep 12 18:54:48 2011 -0400
*/
(function(p) {
    p.execute(function() {
        p.register("jQuery", function() {
            function c(d, r, a) {
                if (a === void 0 && d.nodeType === 1)
                    if (a = "data-" + r.replace(R, "-$1").toLowerCase(), a = d.getAttribute(a), typeof a === "string") {
                        try {
                            a = a === "true"?!0 : a === "false"?!1 : a === "null" ? null : !h.isNaN(a) ? parseFloat(a) : O.test(a) ? h.parseJSON(a) : a
                        } catch (b) {}
                        h.data(d, r, a)
                    } else 
                        a = void 0;
                return a
            }
            function k(d) {
                for (var r in d)
                    if (r !== "toJSON")
                        return !1;
                return !0
            }
            function o(d, r, a) {
                var b = r + "defer", e = r + "queue", f = r + "mark", c = h.data(d, b, void 0, !0);
                c && (a ===
                "queue" ||!h.data(d, e, void 0, !0)) && (a === "mark" ||!h.data(d, f, void 0, !0)) && setTimeout(function() {
                    !h.data(d, e, void 0, !0)&&!h.data(d, f, void 0, !0) && (h.removeData(d, b, !0), c.resolve())
                }, 0)
            }
            function i() {
                return !1
            }
            function m() {
                return !0
            }
            function j(d, r, a) {
                var b = h.extend({}, a[0]);
                b.type = d;
                b.originalEvent = {};
                b.liveFired = void 0;
                h.event.handle.call(r, b);
                b.isDefaultPrevented() && a[0].preventDefault()
            }
            function n(d) {
                var r, a, b, e, f, c, g, l, q, s, j, n = [];
                e = [];
                f = h._data(this, "events");
                if (!(d.liveFired === this ||!f ||!f.live || d.target.disabled ||
                d.button && d.type === "click")) {
                    d.namespace && (j = RegExp("(^|\\.)" + d.namespace.split(".").join("\\.(?:.*\\.)?") + "(\\.|$)"));
                    d.liveFired = this;
                    var v = f.live.slice(0);
                    for (g = 0; g < v.length; g++)
                        f = v[g], f.origType.replace(ea, "") === d.type ? e.push(f.selector) : v.splice(g--, 1);
                    e = h(d.target).closest(e, d.currentTarget);
                    for (l = 0, q = e.length; l < q; l++) {
                        s = e[l];
                        for (g = 0; g < v.length; g++)
                            if (f = v[g], s.selector === f.selector && (!j || j.test(f.namespace))&&!s.elem.disabled) {
                                c = s.elem;
                                b = null;
                                if (f.preType === "mouseenter" || f.preType === "mouseleave")
                                    d.type =
                                    f.preType, (b = h(d.relatedTarget).closest(f.selector)[0]) && h.contains(c, b) && (b = c);
                                    (!b || b !== c) && n.push({
                                        elem: c,
                                        handleObj: f,
                                        level: s.level
                                    })
                            }
                    }
                    for (l = 0, q = n.length; l < q; l++) {
                        e = n[l];
                        if (a && e.level > a)
                            break;
                        d.currentTarget = e.elem;
                        d.data = e.handleObj.data;
                        d.handleObj = e.handleObj;
                        j = e.handleObj.origHandler.apply(e.elem, arguments);
                        if (j===!1 || d.isPropagationStopped())
                            if (a = e.level, j===!1 && (r=!1), d.isImmediatePropagationStopped())
                                break
                    }
                    return r
                }
            }
            function g(d, r) {
                return (d && d !== "*" ? d + "." : "") + r.replace(Ia, "`").replace(Ja,
                "&")
            }
            function f(d, r, a) {
                r = r || 0;
                if (h.isFunction(r))
                    return h.grep(d, function(d, h) {
                        return !!r.call(d, h, d) === a
                    });
                else if (r.nodeType)
                    return h.grep(d, function(d) {
                        return d === r === a
                    });
                else if (typeof r === "string") {
                    var b = h.grep(d, function(d) {
                        return d.nodeType === 1
                    });
                    if (Ka.test(r))
                        return h.filter(r, b, !a);
                    else 
                        r = h.filter(r, b)
                }
                return h.grep(d, function(d) {
                    return h.inArray(d, r) >= 0 === a
                })
            }
            function b(d, r) {
                if (r.nodeType === 1 && h.hasData(d)) {
                    var a = h.expando, b = h.data(d), e = h.data(r, b);
                    if (b = b[a]) {
                        var f = b.events, e = e[a] = h.extend({},
                        b);
                        if (f) {
                            delete e.handle;
                            e.events = {};
                            for (var c in f) {
                                a = 0;
                                for (b = f[c].length; a < b; a++)
                                    h.event.add(r, c + (f[c][a].namespace ? "." : "") + f[c][a].namespace, f[c][a], f[c][a].data)
                                }
                        }
                    }
                }
            }
            function e(d, r) {
                var a;
                if (r.nodeType === 1) {
                    r.clearAttributes && r.clearAttributes();
                    r.mergeAttributes && r.mergeAttributes(d);
                    a = r.nodeName.toLowerCase();
                    if (a === "object")
                        r.outerHTML = d.outerHTML;
                    else if (a === "input" && (d.type === "checkbox" || d.type === "radio")) {
                        if (d.checked)
                            r.defaultChecked = r.checked = d.checked;
                        if (r.value !== d.value)
                            r.value = d.value
                    } else if (a ===
                    "option")
                        r.selected = d.defaultSelected;
                    else if (a === "input" || a === "textarea")
                        r.defaultValue = d.defaultValue;
                    r.removeAttribute(h.expando)
                }
            }
            function a(d) {
                return "getElementsByTagName"in d ? d.getElementsByTagName("*") : "querySelectorAll"in d ? d.querySelectorAll("*") : []
            }
            function l(d) {
                if (d.type === "checkbox" || d.type === "radio")
                    d.defaultChecked = d.checked
            }
            function q(d) {
                h.nodeName(d, "input") ? l(d) : "getElementsByTagName"in d && h.grep(d.getElementsByTagName("input"), l)
            }
            function s(d, r) {
                r.src ? h.ajax({
                    url: r.src,
                    async: !1,
                    dataType: "script"
                }) : h.globalEval((r.text || r.textContent || r.innerHTML || "").replace(La, "/*$0*/"));
                r.parentNode && r.parentNode.removeChild(r)
            }
            function x(d, r, a) {
                var b = r === "width" ? d.offsetWidth: d.offsetHeight, e = r === "width" ? Ma: Na;
                if (b > 0)
                    return a !== "border" && h.each(e, function() {
                        a || (b -= parseFloat(h.css(d, "padding" + this)) || 0);
                        a === "margin" ? b += parseFloat(h.css(d, a + this)) || 0 : b -= parseFloat(h.css(d, "border" + this + "Width")) || 0
                    }), b + "px";
                b = W(d, r, r);
                if (b < 0 || b == null)
                    b = d.style[r] || 0;
                b = parseFloat(b) || 0;
                a && h.each(e, function() {
                    b +=
                    parseFloat(h.css(d, "padding" + this)) || 0;
                    a !== "padding" && (b += parseFloat(h.css(d, "border" + this + "Width")) || 0);
                    a === "margin" && (b += parseFloat(h.css(d, a + this)) || 0)
                });
                return b + "px"
            }
            function w(d) {
                return function(r, a) {
                    var B;
                    typeof r !== "string" && (a = r, r = "*");
                    if (h.isFunction(a))
                        for (var b = r.toLowerCase().split(ma), e = 0, f = b.length, c, g; e < f; e++)
                            c = b[e], (g = /^\+/.test(c)) && (c = c.substr(1) || "*"), B = d[c] = d[c] || [], c = B, c[g ? "unshift": "push"](a)
                }
            }
            function y(d, r, a, h, b, e) {
                b = b || r.dataTypes[0];
                e = e || {};
                e[b]=!0;
                for (var b = d[b], f = 0, c =
                b ? b.length : 0, g = d === fa, l; f < c && (g ||!l); f++)
                    l = b[f](r, a, h), typeof l === "string" && (!g || e[l] ? l = void 0 : (r.dataTypes.unshift(l), l = y(d, r, a, h, l, e)));
                if ((g ||!l)&&!e["*"])
                    l = y(d, r, a, h, "*", e);
                return l
            }
            function t(d, r) {
                var a, b, e = h.ajaxSettings.flatOptions || {};
                for (a in r)
                    r[a] !== void 0 && ((e[a] ? d : b || (b = {}))[a] = r[a]);
                b && h.extend(!0, d, b)
            }
            function v(d, r, a, b) {
                if (h.isArray(r))
                    h.each(r, function(r, e) {
                        a || Oa.test(d) ? b(d, e) : v(d + "[" + (typeof e === "object" || h.isArray(e) ? r : "") + "]", e, a, b)
                    });
                else if (!a && r != null && typeof r === "object")
                    for (var e in r)
                        v(d +
                        "[" + e + "]", r[e], a, b);
                else 
                    b(d, r)
            }
            function z() {
                try {
                    return new window.XMLHttpRequest
                } catch (d) {}
            }
            function A() {
                setTimeout(C, 0);
                return aa = h.now()
            }
            function C() {
                aa = void 0
            }
            function F(d, r) {
                var a = {};
                h.each(na.concat.apply([], na.slice(0, r)), function() {
                    a[this] = d
                });
                return a
            }
            function p(d) {
                if (!ga[d]) {
                    var r = u.body, a = h("<" + d + ">").appendTo(r), b = a.css("display");
                    a.remove();
                    if (b === "none" || b === "") {
                        if (!Q)
                            Q = u.createElement("iframe"), Q.frameBorder = Q.width = Q.height = 0;
                        r.appendChild(Q);
                        if (!X ||!Q.createElement)
                            X = (Q.contentWindow ||
                            Q.contentDocument).document, X.write((h.support.boxModel ? "<!doctype html>" : "") + "<html><body>"), X.close();
                        a = X.createElement(d);
                        X.body.appendChild(a);
                        b = h.css(a, "display");
                        r.removeChild(Q)
                    }
                    ga[d] = b
                }
                return ga[d]
            }
            function J(d) {
                return h.isWindow(d) ? d : d.nodeType === 9 ? d.defaultView || d.parentWindow : !1
            }
            var u = window.document, E = window.navigator, G = window.location, h = function() {
                function d() {
                    if (!r.isReady) {
                        try {
                            u.documentElement.doScroll("left")
                        } catch (a) {
                            setTimeout(d, 1);
                            return 
                        }
                        r.ready()
                    }
                }
                var r = function(d, a) {
                    return new r.fn.init(d,
                    a, b)
                }, a = window.jQuery, h = window.$, b, e = /^(?:[^#<]*(<[\w\W]+>)[^>]*$|#([\w\-]*)$)/, f = /\S/, c = /^\s+/, g = /\s+$/, l = /\d/, q = /^<(\w+)\s*\/?>(?:<\/\1>)?$/, s = /^[\],:{}\s]*$/, j = /\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, n = /"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, v = /(?:^|:|,)(?:\s*\[)+/g, i = /(webkit)[ \/]([\w.]+)/, m = /(opera)(?:.*version)?[ \/]([\w.]+)/, w = /(msie) ([\w.]+)/, k = /(mozilla)(?:.*? rv:([\w.]+))?/, x = /-([a-z]|[0-9])/ig, o = /^-ms-/, z = function(d, r) {
                    return (r + "").toUpperCase()
                }, t = E.userAgent,
                A, C, F = Object.prototype.toString, y = Object.prototype.hasOwnProperty, p = Array.prototype.push, D = Array.prototype.slice, J = String.prototype.trim, G = Array.prototype.indexOf, R = {};
                r.fn = r.prototype = {
                    constructor: r,
                    init: function(d, a, h) {
                        var b;
                        if (!d)
                            return this;
                        if (d.nodeType)
                            return this.context = this[0] = d, this.length = 1, this;
                        if (d === "body"&&!a && u.body)
                            return this.context = u, this[0] = u.body, this.selector = d, this.length = 1, this;
                        if (typeof d === "string")
                            if ((b = d.charAt(0) === "<" && d.charAt(d.length - 1) === ">" && d.length >= 3 ? [null, d,
                            null] : e.exec(d)) && (b[1] ||!a))
                                if (b[1])
                                    return h = (a = a instanceof r ? a[0] : a) ? a.ownerDocument || a : u, (d = q.exec(d)) ? r.isPlainObject(a) ? (d = [u.createElement(d[1])], r.fn.attr.call(d, a, !0)) : d = [h.createElement(d[1])] : (d = r.buildFragment([b[1]], [h]), d = (d.cacheable ? r.clone(d.fragment) : d.fragment).childNodes), r.merge(this, d);
                                else {
                                    if ((a = u.getElementById(b[2])) && a.parentNode) {
                                        if (a.id !== b[2])
                                            return h.find(d);
                                            this.length = 1;
                                            this[0] = a
                                    }
                                    this.context = u;
                                    this.selector = d;
                                    return this
                                } else 
                                    return !a || a.jquery ? (a || h).find(d) : this.constructor(a).find(d);
                                else if (r.isFunction(d))
                            return h.ready(d);
                        if (d.selector !== void 0)
                            this.selector = d.selector, this.context = d.context;
                        return r.makeArray(d, this)
                    },
                    selector: "",
                    jquery: "1.6.4",
                    length: 0,
                    size: function() {
                        return this.length
                    },
                    toArray: function() {
                        return D.call(this, 0)
                    },
                    get: function(d) {
                        return d == null ? this.toArray() : d < 0 ? this[this.length + d] : this[d]
                    },
                    pushStack: function(d, a, h) {
                        var b = this.constructor();
                        r.isArray(d) ? p.apply(b, d) : r.merge(b, d);
                        b.prevObject = this;
                        b.context = this.context;
                        if (a === "find")
                            b.selector = this.selector +
                            (this.selector ? " " : "") + h;
                        else if (a)
                            b.selector = this.selector + "." + a + "(" + h + ")";
                        return b
                    },
                    each: function(d, a) {
                        return r.each(this, d, a)
                    },
                    ready: function(d) {
                        r.bindReady();
                        A.done(d);
                        return this
                    },
                    eq: function(d) {
                        return d===-1 ? this.slice(d) : this.slice(d, + d + 1)
                    },
                    first: function() {
                        return this.eq(0)
                    },
                    last: function() {
                        return this.eq( - 1)
                    },
                    slice: function() {
                        return this.pushStack(D.apply(this, arguments), "slice", D.call(arguments).join(","))
                    },
                    map: function(d) {
                        return this.pushStack(r.map(this, function(r, a) {
                            return d.call(r, a,
                            r)
                        }))
                    },
                    end: function() {
                        return this.prevObject || this.constructor(null)
                    },
                    push: p,
                    sort: [].sort,
                    splice: [].splice
                };
                r.fn.init.prototype = r.fn;
                r.extend = r.fn.extend = function() {
                    var d, a, h, b, e, f = arguments[0] || {}, c = 1, B = arguments.length, g=!1;
                    typeof f === "boolean" && (g = f, f = arguments[1] || {}, c = 2);
                    typeof f !== "object"&&!r.isFunction(f) && (f = {});
                    B === c && (f = this, --c);
                    for (; c < B; c++)
                        if ((d = arguments[c]) != null)
                            for (a in d)
                                h = f[a], b = d[a], f !== b && (g && b && (r.isPlainObject(b) || (e = r.isArray(b))) ? (e ? (e=!1, h = h && r.isArray(h) ? h : []) : h = h && r.isPlainObject(h) ?
                                h : {}, f[a] = r.extend(g, h, b)) : b !== void 0 && (f[a] = b));
                    return f
                };
                r.extend({
                    noConflict: function(d) {
                        if (window.$ === r)
                            window.$ = h;
                        if (d && window.jQuery === r)
                            window.jQuery = a;
                        return r
                    },
                    isReady: !1,
                    readyWait: 1,
                    holdReady: function(d) {
                        d ? r.readyWait++ : r.ready(!0)
                    },
                    ready: function(d) {
                        if (d===!0&&!--r.readyWait || d!==!0&&!r.isReady) {
                            if (!u.body)
                                return setTimeout(r.ready, 1);
                            r.isReady=!0;
                            d!==!0&&--r.readyWait > 0 || (A.resolveWith(u, [r]), r.fn.trigger && r(u).trigger("ready").unbind("ready"))
                        }
                    },
                    bindReady: function() {
                        if (!A) {
                            A = r._Deferred();
                            if (u.readyState === "complete")
                                return setTimeout(r.ready, 1);
                            if (u.addEventListener)
                                u.addEventListener("DOMContentLoaded", C, !1), window.addEventListener("load", r.ready, !1);
                            else if (u.attachEvent) {
                                u.attachEvent("onreadystatechange", C);
                                window.attachEvent("onload", r.ready);
                                var a=!1;
                                try {
                                    a = window.frameElement == null
                                } catch (h) {}
                                u.documentElement.doScroll && a && d()
                            }
                        }
                    },
                    isFunction: function(d) {
                        return r.type(d) === "function"
                    },
                    isArray: Array.isArray || function(d) {
                        return r.type(d) === "array"
                    },
                    isWindow: function(d) {
                        return d &&
                        typeof d === "object" && "setInterval"in d
                    },
                    isNaN: function(d) {
                        return d == null ||!l.test(d) || isNaN(d)
                    },
                    type: function(d) {
                        return d == null ? String(d) : R[F.call(d)] || "object"
                    },
                    isPlainObject: function(d) {
                        if (!d || r.type(d) !== "object" || d.nodeType || r.isWindow(d))
                            return !1;
                        try {
                            if (d.constructor&&!y.call(d, "constructor")&&!y.call(d.constructor.prototype, "isPrototypeOf"))
                                return !1
                        } catch (a) {
                            return !1
                        }
                        for (var h in d);
                        return h === void 0 || y.call(d, h)
                    },
                    isEmptyObject: function(d) {
                        for (var r in d)
                            return !1;
                        return !0
                    },
                    error: function(d) {
                        throw d;
                    },
                    parseJSON: function(d) {
                        if (typeof d !== "string" ||!d)
                            return null;
                        d = r.trim(d);
                        if (window.JSON && window.JSON.parse)
                            return window.JSON.parse(d);
                        if (s.test(d.replace(j, "@").replace(n, "]").replace(v, "")))
                            return (new Function("return " + d))();
                        r.error("Invalid JSON: " + d)
                    },
                    parseXML: function(d) {
                        var a, h;
                        try {
                            window.DOMParser ? (h = new DOMParser, a = h.parseFromString(d, "text/xml")) : (a = new ActiveXObject("Microsoft.XMLDOM"), a.async = "false", a.loadXML(d))
                        } catch (b) {
                            a = void 0
                        }(!a ||!a.documentElement || a.getElementsByTagName("parsererror").length) &&
                        r.error("Invalid XML: " + d);
                        return a
                    },
                    noop: function() {},
                    globalEval: function(d) {
                        d && f.test(d) && (window.execScript || function(d) {
                            window.eval.call(window, d)
                        })(d)
                    },
                    camelCase: function(d) {
                        return d.replace(o, "ms-").replace(x, z)
                    },
                    nodeName: function(d, r) {
                        return d.nodeName && d.nodeName.toUpperCase() === r.toUpperCase()
                    },
                    each: function(d, a, h) {
                        var b, e = 0, f = d.length, c = f === void 0 || r.isFunction(d);
                        if (h)
                            if (c)
                                for (b in d) {
                                    if (a.apply(d[b], h)===!1)
                                        break
                                } else 
                                    for (; e < f;) {
                                        if (a.apply(d[e++], h)===!1)
                                            break
                                    } else if (c)
                            for (b in d) {
                                if (a.call(d[b],
                                b, d[b])===!1)
                                    break
                            } else 
                                for (; e < f;)
                                    if (a.call(d[e], e, d[e++])===!1)
                                        break;
                        return d
                    },
                    trim: J ? function(d) {
                        return d == null ? "" : J.call(d)
                    }
                    : function(d) {
                        return d == null ? "" : d.toString().replace(c, "").replace(g, "")
                    },
                    makeArray: function(d, a) {
                        var h = a || [];
                        if (d != null) {
                            var b = r.type(d);
                            d.length == null || b === "string" || b === "function" || b === "regexp" || r.isWindow(d) ? p.call(h, d) : r.merge(h, d)
                        }
                        return h
                    },
                    inArray: function(d, r) {
                        if (!r)
                            return - 1;
                        if (G)
                            return G.call(r, d);
                        for (var a = 0, h = r.length; a < h; a++)
                            if (r[a] === d)
                                return a;
                        return - 1
                    },
                    merge: function(d,
                    r) {
                        var a = d.length, h = 0;
                        if (typeof r.length === "number")
                            for (var b = r.length; h < b; h++)
                                d[a++] = r[h];
                        else 
                            for (; r[h] !== void 0;)
                                d[a++] = r[h++];
                        d.length = a;
                        return d
                    },
                    grep: function(d, r, a) {
                        for (var h = [], b, a=!!a, e = 0, f = d.length; e < f; e++)
                            b=!!r(d[e], e), a !== b && h.push(d[e]);
                        return h
                    },
                    map: function(d, a, h) {
                        var b, e, f = [], c = 0, B = d.length;
                        if (d instanceof r || B !== void 0 && typeof B === "number" && (B > 0 && d[0] && d[B - 1] || B === 0 || r.isArray(d)
                            ))for (; c < B; c++)
                            b = a(d[c], c, h), b != null && (f[f.length] = b);
                        else 
                            for (e in d)
                                b = a(d[e], e, h), b != null && (f[f.length] =
                                b);
                        return f.concat.apply([], f)
                    },
                    guid: 1,
                    proxy: function(d, a) {
                        if (typeof a === "string")
                            var h = d[a], a = d, d = h;
                        if (r.isFunction(d)) {
                            var b = D.call(arguments, 2), h = function() {
                                return d.apply(a, b.concat(D.call(arguments)))
                            };
                            h.guid = d.guid = d.guid || h.guid || r.guid++;
                            return h
                        }
                    },
                    access: function(d, a, h, b, e, f) {
                        var c = d.length;
                        if (typeof a === "object") {
                            for (var B in a)
                                r.access(d, B, a[B], b, e, h);
                            return d
                        }
                        if (h !== void 0) {
                            b=!f && b && r.isFunction(h);
                            for (B = 0; B < c; B++)
                                e(d[B], a, b ? h.call(d[B], B, e(d[B], a)) : h, f);
                            return d
                        }
                        return c ? e(d[0], a) : void 0
                    },
                    now: function() {
                        return (new Date).getTime()
                    },
                    uaMatch: function(d) {
                        d = d.toLowerCase();
                        d = i.exec(d) || m.exec(d) || w.exec(d) || d.indexOf("compatible") < 0 && k.exec(d) || [];
                        return {
                            browser: d[1] || "",
                            version: d[2] || "0"
                        }
                    },
                    sub: function() {
                        function d(r, a) {
                            return new d.fn.init(r, a)
                        }
                        r.extend(!0, d, this);
                        d.superclass = this;
                        d.fn = d.prototype = this ();
                        d.fn.constructor = d;
                        d.sub = this.sub;
                        d.fn.init = function(h, b) {
                            b && b instanceof r&&!(b instanceof d) && (b = d(b));
                            return r.fn.init.call(this, h, b, a)
                        };
                        d.fn.init.prototype = d.fn;
                        var a = d(u);
                        return d
                    },
                    browser: {}
                });
                r.each("Boolean Number String Function Array Date RegExp Object".split(" "), function(d, r) {
                    R["[object " + r + "]"] = r.toLowerCase()
                });
                t = r.uaMatch(t);
                if (t.browser)
                    r.browser[t.browser]=!0, r.browser.version = t.version;
                if (r.browser.webkit)
                    r.browser.safari=!0;
                f.test("\u00a0") && (c = /^[\s\xA0]+/, g = /[\s\xA0]+$/);
                b = r(u);
                u.addEventListener ? C = function() {
                    u.removeEventListener("DOMContentLoaded", C, !1);
                    r.ready()
                } : u.attachEvent && (C = function() {
                    u.readyState === "complete" && (u.detachEvent("onreadystatechange", C),
                    r.ready())
                });
                return r
            }(), H = "done fail isResolved isRejected promise then always pipe".split(" "), L = [].slice;
            h.extend({
                _Deferred: function() {
                    var d = [], r, a, b, e = {
                        done: function() {
                            if (!b) {
                                var a = arguments, f, c, B, g, l;
                                r && (l = r, r = 0);
                                for (f = 0, c = a.length; f < c; f++)
                                    B = a[f], g = h.type(B), g === "array" ? e.done.apply(e, B) : g === "function" && d.push(B);
                                l && e.resolveWith(l[0], l[1])
                            }
                            return this
                        },
                        resolveWith: function(h, e) {
                            if (!b&&!r&&!a) {
                                e = e || [];
                                a = 1;
                                try {
                                    for (; d[0];)
                                        d.shift().apply(h, e)
                                    } finally {
                                    r = [h, e], a = 0
                                }
                            }
                            return this
                        },
                        resolve: function() {
                            e.resolveWith(this,
                            arguments);
                            return this
                        },
                        isResolved: function() {
                            return !(!a&&!r)
                        },
                        cancel: function() {
                            b = 1;
                            d = [];
                            return this
                        }
                    };
                    return e
                },
                Deferred: function(d) {
                    var r = h._Deferred(), a = h._Deferred(), b;
                    h.extend(r, {
                        then: function(d, a) {
                            r.done(d).fail(a);
                            return this
                        },
                        always: function() {
                            return r.done.apply(r, arguments).fail.apply(this, arguments)
                        },
                        fail: a.done,
                        rejectWith: a.resolveWith,
                        reject: a.resolve,
                        isRejected: a.isResolved,
                        pipe: function(d, a) {
                            return h.Deferred(function(b) {
                                h.each({
                                    done: [d, "resolve"],
                                    fail: [a, "reject"]
                                }, function(d, a) {
                                    var e =
                                    a[0], f = a[1], c;
                                    if (h.isFunction(e))
                                        r[d](function() {
                                            if ((c = e.apply(this, arguments)) && h.isFunction(c.promise))
                                                c.promise().then(b.resolve, b.reject);
                                            else 
                                                b[f + "With"](this === r ? b : this, [c])
                                            });
                                    else 
                                        r[d](b[f])
                                })
                            }).promise()
                        },
                        promise: function(d) {
                            if (d == null) {
                                if (b)
                                    return b;
                                b = d = {}
                            }
                            for (var a = H.length; a--;)
                                d[H[a]] = r[H[a]];
                            return d
                        }
                    });
                    r.done(a.cancel).fail(r.cancel);
                    delete r.cancel;
                    d && d.call(r, r);
                    return r
                },
                when: function(d) {
                    function a(d) {
                        return function(a) {
                            b[d] = arguments.length > 1 ? L.call(arguments, 0) : a;
                            --c || g.resolveWith(g,
                            L.call(b, 0))
                        }
                    }
                    var b = arguments, e = 0, f = b.length, c = f, g = f <= 1 && d && h.isFunction(d.promise) ? d: h.Deferred();
                    if (f > 1) {
                        for (; e < f; e++)
                            b[e] && h.isFunction(b[e].promise) ? b[e].promise().then(a(e), g.reject) : --c;
                        c || g.resolveWith(g, b)
                    } else 
                        g !== d && g.resolveWith(g, f ? [d] : []);
                    return g.promise()
                }
            });
            h.support = function() {
                var d = u.createElement("div"), a = u.documentElement, b, e, f, c, g, l;
                d.setAttribute("className", "t");
                d.innerHTML = "   <link/><table></table><a href='/a' style='top:1px;float:left;opacity:.55;'>a</a><input type='checkbox'/>";
                b = d.getElementsByTagName("*");
                e = d.getElementsByTagName("a")[0];
                if (!b ||!b.length ||!e)
                    return {};
                f = u.createElement("select");
                c = f.appendChild(u.createElement("option"));
                b = d.getElementsByTagName("input")[0];
                g = {
                    leadingWhitespace: d.firstChild.nodeType === 3,
                    tbody: !d.getElementsByTagName("tbody").length,
                    htmlSerialize: !!d.getElementsByTagName("link").length,
                    style: /top/.test(e.getAttribute("style")),
                    hrefNormalized: e.getAttribute("href") === "/a",
                    opacity: /^0.55$/.test(e.style.opacity),
                    cssFloat: !!e.style.cssFloat,
                    checkOn: b.value === "on",
                    optSelected: c.selected,
                    getSetAttribute: d.className !== "t",
                    submitBubbles: !0,
                    changeBubbles: !0,
                    focusinBubbles: !1,
                    deleteExpando: !0,
                    noCloneEvent: !0,
                    inlineBlockNeedsLayout: !1,
                    shrinkWrapBlocks: !1,
                    reliableMarginRight: !0
                };
                b.checked=!0;
                g.noCloneChecked = b.cloneNode(!0).checked;
                f.disabled=!0;
                g.optDisabled=!c.disabled;
                try {
                    delete d.test
                } catch (q) {
                    g.deleteExpando=!1
                }
                !d.addEventListener && d.attachEvent && d.fireEvent && (d.attachEvent("onclick", function() {
                    g.noCloneEvent=!1
                }), d.cloneNode(!0).fireEvent("onclick"));
                b = u.createElement("input");
                b.value = "t";
                b.setAttribute("type", "radio");
                g.radioValue = b.value === "t";
                b.setAttribute("checked", "checked");
                d.appendChild(b);
                e = u.createDocumentFragment();
                e.appendChild(d.firstChild);
                g.checkClone = e.cloneNode(!0).cloneNode(!0).lastChild.checked;
                d.innerHTML = "";
                d.style.width = d.style.paddingLeft = "1px";
                f = u.getElementsByTagName("body")[0];
                e = u.createElement(f ? "div" : "body");
                c = {
                    visibility: "hidden",
                    width: 0,
                    height: 0,
                    border: 0,
                    margin: 0,
                    background: "none"
                };
                f && h.extend(c, {
                    position: "absolute",
                    left: "-1000px",
                    top: "-1000px"
                });
                for (l in c)
                    e.style[l] = c[l];
                e.appendChild(d);
                a = f || a;
                a.insertBefore(e, a.firstChild);
                g.appendChecked = b.checked;
                h.boxModel = g.boxModel = u.compatMode === "CSS1Compat";
                if ("zoom"in d.style)
                    d.style.display = "inline", d.style.zoom = 1, g.inlineBlockNeedsLayout = d.offsetWidth === 2, d.style.display = "", d.innerHTML = "<div style='width:4px;'></div>", g.shrinkWrapBlocks = d.offsetWidth !== 2;
                d.innerHTML = "<table><tr><td style='padding:0;border:0;display:none'></td><td>t</td></tr></table>";
                f = d.getElementsByTagName("td");
                b = f[0].offsetHeight === 0;
                f[0].style.display = "";
                f[1].style.display = "none";
                g.reliableHiddenOffsets = b && f[0].offsetHeight === 0;
                d.innerHTML = "";
                if (u.defaultView && u.defaultView.getComputedStyle)
                    b = u.createElement("div"), b.style.width = "0", b.style.marginRight = "0", d.appendChild(b), g.reliableMarginRight = (parseInt((u.defaultView.getComputedStyle(b, null) || {
                        marginRight: 0
                    }).marginRight, 10) || 0) === 0;
                e.innerHTML = "";
                a.removeChild(e);
                if (d.attachEvent)
                    for (l in{
                        submit: 1,
                        change: 1,
                        focusin: 1
                    })
                        a = "on" + l, b = a in d, b || (d.setAttribute(a,
                        "return;"), b = typeof d[a] === "function"), g[l + "Bubbles"] = b;
                e = e = f = c = f = b = d = b = null;
                return g
            }();
            var O = /^(?:\{.*\}|\[.*\])$/, R = /([A-Z])/g;
            h.extend({
                cache: {},
                uuid: 0,
                expando: "jQuery" + (h.fn.jquery + Math.random()).replace(/\D/g, ""),
                noData: {
                    embed: !0,
                    object: "clsid:D27CDB6E-AE6D-11cf-96B8-444553540000",
                    applet: !0
                },
                hasData: function(d) {
                    d = d.nodeType ? h.cache[d[h.expando]] : d[h.expando];
                    return !!d&&!k(d)
                },
                data: function(d, a, b, e) {
                    if (h.acceptData(d)) {
                        var f = h.expando, c = typeof a === "string", g = d.nodeType, l = g ? h.cache: d, q = g ? d[h.expando]:
                        d[h.expando] && h.expando;
                        if (q && (!e ||!q ||!l[q] || l[q][f]) ||!(c && b === void 0)) {
                            if (!q)
                                g ? d[h.expando] = q=++h.uuid : q = h.expando;
                            if (!l[q] && (l[q] = {}, !g))
                                l[q].toJSON = h.noop;
                            if (typeof a === "object" || typeof a === "function")
                                e ? l[q][f] = h.extend(l[q][f], a) : l[q] = h.extend(l[q], a);
                            d = l[q];
                            e && (d[f] || (d[f] = {}), d = d[f]);
                            b !== void 0 && (d[h.camelCase(a)] = b);
                            if (a === "events"&&!d[a])
                                return d[f] && d[f].events;
                            c ? (b = d[a], b == null && (b = d[h.camelCase(a)])) : b = d;
                            return b
                        }
                    }
                },
                removeData: function(d, a, b) {
                    if (h.acceptData(d)) {
                        var e, f = h.expando, c = d.nodeType,
                        g = c ? h.cache: d, l = c ? d[h.expando]: h.expando;
                        if (g[l]) {
                            if (a && (e = b ? g[l][f] : g[l]))
                                if (e[a] || (a = h.camelCase(a)), delete e[a], !k(e))
                                    return;
                            if (b && (delete g[l][f], !k(g[l])))
                                return;
                            a = g[l][f];
                            h.support.deleteExpando ||!g.setInterval ? delete g[l] : g[l] = null;
                            if (a) {
                                g[l] = {};
                                if (!c)
                                    g[l].toJSON = h.noop;
                                g[l][f] = a
                            } else 
                                c && (h.support.deleteExpando ? delete d[h.expando] : d.removeAttribute ? d.removeAttribute(h.expando) : d[h.expando] = null)
                            }
                    }
                },
                _data: function(d, a, b) {
                    return h.data(d, a, b, !0)
                },
                acceptData: function(d) {
                    if (d.nodeName) {
                        var a =
                        h.noData[d.nodeName.toLowerCase()];
                        if (a)
                            return !(a===!0 || d.getAttribute("classid") !== a)
                    }
                    return !0
                }
            });
            h.fn.extend({
                data: function(d, a) {
                    var b = null;
                    if (typeof d === "undefined") {
                        if (this.length && (b = h.data(this[0]), this[0].nodeType === 1))
                            for (var e = this[0].attributes, f, g = 0, l = e.length; g < l; g++)
                                f = e[g].name, f.indexOf("data-") === 0 && (f = h.camelCase(f.substring(5)), c(this[0], f, b[f]));
                        return b
                    } else if (typeof d === "object")
                        return this.each(function() {
                            h.data(this, d)
                        });
                    var q = d.split(".");
                    q[1] = q[1] ? "." + q[1] : "";
                    return a === void 0 ?
                    (b = this.triggerHandler("getData" + q[1] + "!", [q[0]]), b === void 0 && this.length && (b = h.data(this[0], d), b = c(this[0], d, b)), b === void 0 && q[1] ? this.data(q[0]) : b) : this.each(function() {
                        var b = h(this), e = [q[0], a];
                        b.triggerHandler("setData" + q[1] + "!", e);
                        h.data(this, d, a);
                        b.triggerHandler("changeData" + q[1] + "!", e)
                    })
                },
                removeData: function(d) {
                    return this.each(function() {
                        h.removeData(this, d)
                    })
                }
            });
            h.extend({
                _mark: function(d, a) {
                    d && (a = (a || "fx") + "mark", h.data(d, a, (h.data(d, a, void 0, !0) || 0) + 1, !0))
                },
                _unmark: function(d, a, b) {
                    d !==
                    !0 && (b = a, a = d, d=!1);
                    if (a) {
                        var b = b || "fx", e = b + "mark";
                        (d = d ? 0 : (h.data(a, e, void 0, !0) || 1) - 1) ? h.data(a, e, d, !0) : (h.removeData(a, e, !0), o(a, b, "mark"))
                    }
                },
                queue: function(d, a, b) {
                    if (d) {
                        var a = (a || "fx") + "queue", e = h.data(d, a, void 0, !0);
                        b && (!e || h.isArray(b) ? e = h.data(d, a, h.makeArray(b), !0) : e.push(b));
                        return e || []
                    }
                },
                dequeue: function(d, a) {
                    var a = a || "fx", b = h.queue(d, a), e = b.shift();
                    e === "inprogress" && (e = b.shift());
                    e && (a === "fx" && b.unshift("inprogress"), e.call(d, function() {
                        h.dequeue(d, a)
                    }));
                    b.length || (h.removeData(d, a + "queue",
                    !0), o(d, a, "queue"))
                }
            });
            h.fn.extend({
                queue: function(d, a) {
                    typeof d !== "string" && (a = d, d = "fx");
                    return a === void 0 ? h.queue(this[0], d) : this.each(function() {
                        var b = h.queue(this, d, a);
                        d === "fx" && b[0] !== "inprogress" && h.dequeue(this, d)
                    })
                },
                dequeue: function(d) {
                    return this.each(function() {
                        h.dequeue(this, d)
                    })
                },
                delay: function(d, a) {
                    d = h.fx ? h.fx.speeds[d] || d : d;
                    a = a || "fx";
                    return this.queue(a, function() {
                        var b = this;
                        setTimeout(function() {
                            h.dequeue(b, a)
                        }, d)
                    })
                },
                clearQueue: function(d) {
                    return this.queue(d || "fx", [])
                },
                promise: function(d) {
                    function a() {
                        --c ||
                        b.resolveWith(e, [e])
                    }
                    typeof d !== "string" && (d = void 0);
                    var d = d || "fx", b = h.Deferred(), e = this, f = e.length, c = 1, g = d + "defer", l = d + "queue";
                    d += "mark";
                    for (var q; f--;)
                        if (q = h.data(e[f], g, void 0, !0) || (h.data(e[f], l, void 0, !0) || h.data(e[f], d, void 0, !0)) && h.data(e[f], g, h._Deferred(), !0))
                            c++, q.done(a);
                    a();
                    return b.promise()
                }
            });
            var M = /[\n\t\r]/g, I = /\s+/, K = /\r/g, Z = /^(?:button|input)$/i, Pa = /^(?:button|input|object|select|textarea)$/i, Qa = /^a(?:rea)?$/i, oa = /^(?:autofocus|autoplay|async|checked|controls|defer|disabled|hidden|loop|multiple|open|readonly|required|scoped|selected)$/i,
            S, pa;
            h.fn.extend({
                attr: function(d, a) {
                    return h.access(this, d, a, !0, h.attr)
                },
                removeAttr: function(d) {
                    return this.each(function() {
                        h.removeAttr(this, d)
                    })
                },
                prop: function(d, a) {
                    return h.access(this, d, a, !0, h.prop)
                },
                removeProp: function(d) {
                    d = h.propFix[d] || d;
                    return this.each(function() {
                        try {
                            this[d] = void 0, delete this[d]
                        } catch (a) {}
                    })
                },
                addClass: function(d) {
                    var a, b, e, f, c, g, l;
                    if (h.isFunction(d))
                        return this.each(function(a) {
                            h(this).addClass(d.call(this, a, this.className))
                        });
                    if (d && typeof d === "string") {
                        a = d.split(I);
                        for (b =
                        0, e = this.length; b < e; b++)
                            if (f = this[b], f.nodeType === 1)
                                if (!f.className && a.length === 1)
                                    f.className = d;
                                else {
                                    c = " " + f.className + " ";
                                    for (g = 0, l = a.length; g < l; g++)
                                        ~c.indexOf(" " + a[g] + " ") || (c += a[g] + " ");
                                        f.className = h.trim(c)
                                }
                    }
                    return this
                },
                removeClass: function(d) {
                    var a, b, e, f, c, g, l;
                    if (h.isFunction(d))
                        return this.each(function(a) {
                            h(this).removeClass(d.call(this, a, this.className))
                        });
                    if (d && typeof d === "string" || d === void 0) {
                        a = (d || "").split(I);
                        for (b = 0, e = this.length; b < e; b++)
                            if (f = this[b], f.nodeType === 1 && f.className)
                                if (d) {
                                    c =
                                    (" " + f.className + " ").replace(M, " ");
                                    for (g = 0, l = a.length; g < l; g++)
                                        c = c.replace(" " + a[g] + " ", " ");
                                        f.className = h.trim(c)
                                } else 
                                    f.className = ""
                    }
                    return this
                },
                toggleClass: function(d, a) {
                    var b = typeof d, e = typeof a === "boolean";
                    return h.isFunction(d) ? this.each(function(b) {
                        h(this).toggleClass(d.call(this, b, this.className, a), a)
                    }) : this.each(function() {
                        if (b === "string")
                            for (var f, c = 0, g = h(this), l = a, q = d.split(I); f = q[c++];)
                                l = e ? l : !g.hasClass(f), g[l ? "addClass": "removeClass"](f);
                        else if (b === "undefined" || b === "boolean")
                            this.className &&
                            h._data(this, "__className__", this.className), this.className = this.className || d===!1 ? "" : h._data(this, "__className__") || ""
                    })
                },
                hasClass: function(d) {
                    for (var d = " " + d + " ", a = 0, b = this.length; a < b; a++)
                        if (this[a].nodeType === 1 && (" " + this[a].className + " ").replace(M, " ").indexOf(d)>-1)
                            return !0;
                    return !1
                },
                val: function(d) {
                    var a, b, e = this[0];
                    if (arguments.length) {
                        var f = h.isFunction(d);
                        return this.each(function(b) {
                            var e = h(this);
                            if (this.nodeType === 1 && (b = f ? d.call(this, b, e.val()) : d, b == null ? b = "" : typeof b === "number" ? b += "" :
                            h.isArray(b) && (b = h.map(b, function(d) {
                                return d == null ? "" : d + ""
                            })), a = h.valHooks[this.nodeName.toLowerCase()] || h.valHooks[this.type], !a ||!("set"in a) || a.set(this, b, "value") === void 0))
                                this.value = b
                        })
                    } else if (e) {
                        if ((a = h.valHooks[e.nodeName.toLowerCase()] || h.valHooks[e.type]) && "get"in a && (b = a.get(e, "value")) !== void 0)
                            return b;
                        b = e.value;
                        return typeof b === "string" ? b.replace(K, "") : b == null ? "" : b
                    }
                }
            });
            h.extend({
                valHooks: {
                    option: {
                        get: function(d) {
                            var a = d.attributes.value;
                            return !a || a.specified ? d.value : d.text
                        }
                    },
                    select: {
                        get: function(d) {
                            var a,
                            b = d.selectedIndex, e = [], f = d.options, d = d.type === "select-one";
                            if (b < 0)
                                return null;
                            for (var c = d ? b : 0, g = d ? b + 1 : f.length; c < g; c++)
                                if (a = f[c], a.selected && (h.support.optDisabled?!a.disabled : a.getAttribute("disabled") === null) && (!a.parentNode.disabled ||!h.nodeName(a.parentNode, "optgroup"))
                                    ) {
                                a = h(a).val();
                                if (d)
                                    return a;
                                e.push(a)
                            }
                            return d&&!e.length && f.length ? h(f[b]).val() : e
                        },
                        set: function(d, a) {
                            var b = h.makeArray(a);
                            h(d).find("option").each(function() {
                                this.selected = h.inArray(h(this).val(), b) >= 0
                            });
                            if (!b.length)
                                d.selectedIndex =
                                - 1;
                            return b
                        }
                    }
                },
                attrFn: {
                    val: !0,
                    css: !0,
                    html: !0,
                    text: !0,
                    data: !0,
                    width: !0,
                    height: !0,
                    offset: !0
                },
                attrFix: {
                    tabindex: "tabIndex"
                },
                attr: function(d, a, b, e) {
                    var f = d.nodeType;
                    if (d&&!(f === 3 || f === 8 || f === 2)) {
                        if (e && a in h.attrFn)
                            return h(d)[a](b);
                        if (!("getAttribute"in d))
                            return h.prop(d, a, b);
                        var c, g;
                        if (e = f !== 1 ||!h.isXMLDoc(d))
                            a = h.attrFix[a] || a, (g = h.attrHooks[a]) || (oa.test(a) ? g = pa : S && (g = S));
                        if (b !== void 0)
                            if (b === null)
                                h.removeAttr(d, a);
                            else 
                                return g && "set"in g && e && (c = g.set(d, b, a)) !== void 0 ? c : (d.setAttribute(a, "" + b), b);
                        else 
                            return g && "get"in g && e && (c = g.get(d, a)) !== null ? c : (c = d.getAttribute(a), c === null ? void 0 : c)
                    }
                },
                removeAttr: function(d, a) {
                    var b;
                    if (d.nodeType === 1 && (a = h.attrFix[a] || a, h.attr(d, a, ""), d.removeAttribute(a), oa.test(a) && (b = h.propFix[a] || a)in d))
                        d[b]=!1
                },
                attrHooks: {
                    type: {
                        set: function(d, a) {
                            if (Z.test(d.nodeName) && d.parentNode)
                                h.error("type property can't be changed");
                            else if (!h.support.radioValue && a === "radio" && h.nodeName(d, "input")) {
                                var b = d.value;
                                d.setAttribute("type", a);
                                if (b)
                                    d.value = b;
                                return a
                            }
                        }
                    },
                    value: {
                        get: function(d,
                        a) {
                            return S && h.nodeName(d, "button") ? S.get(d, a) : a in d ? d.value : null
                        },
                        set: function(d, a, b) {
                            if (S && h.nodeName(d, "button"))
                                return S.set(d, a, b);
                            d.value = a
                        }
                    }
                },
                propFix: {
                    tabindex: "tabIndex",
                    readonly: "readOnly",
                    "for": "htmlFor",
                    "class": "className",
                    maxlength: "maxLength",
                    cellspacing: "cellSpacing",
                    cellpadding: "cellPadding",
                    rowspan: "rowSpan",
                    colspan: "colSpan",
                    usemap: "useMap",
                    frameborder: "frameBorder",
                    contenteditable: "contentEditable"
                },
                prop: function(d, a, b) {
                    var e = d.nodeType;
                    if (d&&!(e === 3 || e === 8 || e === 2)) {
                        var f, c;
                        if (e !==
                        1 ||!h.isXMLDoc(d))
                            a = h.propFix[a] || a, c = h.propHooks[a];
                        return b !== void 0 ? c && "set"in c && (f = c.set(d, b, a)) !== void 0 ? f : d[a] = b : c && "get"in c && (f = c.get(d, a)) !== null ? f : d[a]
                    }
                },
                propHooks: {
                    tabIndex: {
                        get: function(d) {
                            var a = d.getAttributeNode("tabindex");
                            return a && a.specified ? parseInt(a.value, 10) : Pa.test(d.nodeName) || Qa.test(d.nodeName) && d.href ? 0 : void 0
                        }
                    }
                }
            });
            h.attrHooks.tabIndex = h.propHooks.tabIndex;
            pa = {
                get: function(d, a) {
                    var b;
                    return h.prop(d, a)===!0 || (b = d.getAttributeNode(a)) && b.nodeValue!==!1 ? a.toLowerCase() : void 0
                },
                set: function(d, a, b) {
                    a===!1 ? h.removeAttr(d, b) : (a = h.propFix[b] || b, a in d && (d[a]=!0), d.setAttribute(b, b.toLowerCase()));
                    return b
                }
            };
            if (!h.support.getSetAttribute)
                S = h.valHooks.button = {
                    get: function(d, a) {
                        var b;
                        return (b = d.getAttributeNode(a)) && b.nodeValue !== "" ? b.nodeValue : void 0
                    },
                    set: function(d, a, b) {
                        var e = d.getAttributeNode(b);
                        e || (e = u.createAttribute(b), d.setAttributeNode(e));
                        return e.nodeValue = a + ""
                    }
                }, h.each(["width", "height"], function(d, a) {
                    h.attrHooks[a] = h.extend(h.attrHooks[a], {
                        set: function(d, b) {
                            if (b ===
                            "")
                                return d.setAttribute(a, "auto"), b
                            }
                        })
                    });
            h.support.hrefNormalized || h.each(["href", "src", "width", "height"], function(d, a) {
                h.attrHooks[a] = h.extend(h.attrHooks[a], {
                    get: function(d) {
                        d = d.getAttribute(a, 2);
                        return d === null ? void 0 : d
                    }
                })
            });
            if (!h.support.style)
                h.attrHooks.style = {
                    get: function(d) {
                        return d.style.cssText.toLowerCase() || void 0
                    },
                    set: function(d, a) {
                        return d.style.cssText = "" + a
                    }
                };
            if (!h.support.optSelected)
                h.propHooks.selected = h.extend(h.propHooks.selected, {
                    get: function() {
                        return null
                    }
                });
            h.support.checkOn ||
            h.each(["radio", "checkbox"], function() {
                h.valHooks[this] = {
                    get: function(d) {
                        return d.getAttribute("value") === null ? "on" : d.value
                    }
                }
            });
            h.each(["radio", "checkbox"], function() {
                h.valHooks[this] = h.extend(h.valHooks[this], {
                    set: function(d, a) {
                        if (h.isArray(a))
                            return d.checked = h.inArray(h(d).val(), a) >= 0
                    }
                })
            });
            var ea = /\.(.*)$/, ha = /^(?:textarea|input|select)$/i, Ia = /\./g, Ja = / /g, Ra = /[^\w\s.|`]/g, Sa = function(d) {
                return d.replace(Ra, "\\$&")
            };
            h.event = {
                add: function(d, a, b, e) {
                    if (!(d.nodeType === 3 || d.nodeType === 8)) {
                        if (b===!1)
                            b =
                            i;
                        else if (!b)
                            return;
                        var f, c;
                        if (b.handler)
                            f = b, b = f.handler;
                        if (!b.guid)
                            b.guid = h.guid++;
                        if (c = h._data(d)) {
                            var g = c.events, l = c.handle;
                            if (!g)
                                c.events = g = {};
                            if (!l)
                                c.handle = l = function(d) {
                                    return typeof h !== "undefined" && (!d || h.event.triggered !== d.type) ? h.event.handle.apply(l.elem, arguments) : void 0
                                };
                            l.elem = d;
                            for (var a = a.split(" "), q, s = 0, j; q = a[s++];) {
                                c = f ? h.extend({}, f) : {
                                    handler: b,
                                    data: e
                                };
                                q.indexOf(".")>-1 ? (j = q.split("."), q = j.shift(), c.namespace = j.slice(0).sort().join(".")) : (j = [], c.namespace = "");
                                c.type = q;
                                if (!c.guid)
                                    c.guid =
                                    b.guid;
                                var n = g[q], v = h.event.special[q] || {};
                                if (!n && (n = g[q] = [], !v.setup || v.setup.call(d, e, j, l)===!1))
                                    d.addEventListener ? d.addEventListener(q, l, !1) : d.attachEvent && d.attachEvent("on" + q, l);
                                if (v.add && (v.add.call(d, c), !c.handler.guid))
                                    c.handler.guid = b.guid;
                                n.push(c);
                                h.event.global[q]=!0
                            }
                            d = null
                        }
                    }
                },
                global: {},
                remove: function(d, a, b, e) {
                    if (!(d.nodeType === 3 || d.nodeType === 8)) {
                        b===!1 && (b = i);
                        var f, c, g = 0, l, q, s, j, n, v, m = h.hasData(d) && h._data(d), w = m && m.events;
                        if (m && w) {
                            if (a && a.type)
                                b = a.handler, a = a.type;
                            if (!a || typeof a ===
                            "string" && a.charAt(0) === ".")
                                for (f in a = a || "", w)
                                    h.event.remove(d, f + a);
                            else {
                                for (a = a.split(" "); f = a[g++];)
                                    if (j = f, l = f.indexOf(".") < 0, q = [], l || (q = f.split("."), f = q.shift(), s = RegExp("(^|\\.)" + h.map(q.slice(0).sort(), Sa).join("\\.(?:.*\\.)?") + "(\\.|$)")), n = w[f])
                                        if (b) {
                                            j = h.event.special[f] || {};
                                            for (c = e || 0; c < n.length; c++)
                                                if (v = n[c], b.guid === v.guid) {
                                                    if (l || s.test(v.namespace))
                                                        e == null && n.splice(c--, 1), j.remove && j.remove.call(d, v);
                                                        if (e != null)
                                                            break
                                                }
                                                if (n.length === 0 || e != null && n.length === 1)(!j.teardown || j.teardown.call(d,
                                                q)===!1) && h.removeEvent(d, f, m.handle), delete w[f]
                                        } else 
                                            for (c = 0; c < n.length; c++)
                                                if (v = n[c], l || s.test(v.namespace))
                                                    h.event.remove(d, j, v.handler, c), n.splice(c--, 1);
                                                    if (h.isEmptyObject(w)) {
                                                        if (a = m.handle)
                                                            a.elem = null;
                                                            delete m.events;
                                                            delete m.handle;
                                                            h.isEmptyObject(m) && h.removeData(d, void 0, !0)
                                                        }
                                                }
                        }
                    }
                }, customEvent: {
                    getData: !0, setData : !0, changeData : !0
                }, trigger: function(d, a, b, e) {
                    var f = d.type || d, c = [], g;
                    f.indexOf("!") >= 0 && (f = f.slice(0, - 1), g=!0);
                    f.indexOf(".") >= 0 && (c = f.split("."), f = c.shift(), c.sort());
                    if (b&&!h.event.customEvent[f] ||
                    h.event.global[f]) {
                        d = typeof d === "object" ? d[h.expando] ? d : new h.Event(f, d) : new h.Event(f);
                        d.type = f;
                        d.exclusive = g;
                        d.namespace = c.join(".");
                        d.namespace_re = RegExp("(^|\\.)" + c.join("\\.(?:.*\\.)?") + "(\\.|$)");
                        if (e ||!b)
                            d.preventDefault(), d.stopPropagation();
                        if (b) {
                            if (!(b.nodeType === 3 || b.nodeType === 8)) {
                                d.result = void 0;
                                d.target = b;
                                a = a != null ? h.makeArray(a) : [];
                                a.unshift(d);
                                c = b;
                                e = f.indexOf(":") < 0 ? "on" + f : "";
                                do {
                                    g = h._data(c, "handle");
                                    d.currentTarget = c;
                                    g && g.apply(c, a);
                                    if (e && h.acceptData(c) && c[e] && c[e].apply(c, a) ===
                                    !1)
                                        d.result=!1, d.preventDefault();
                                    c = c.parentNode || c.ownerDocument || c === d.target.ownerDocument && window
                                }
                                while (c&&!d.isPropagationStopped());
                                if (!d.isDefaultPrevented()) {
                                    var l, c = h.event.special[f] || {};
                                    if ((!c._default || c._default.call(b.ownerDocument, d)===!1)&&!(f === "click" && h.nodeName(b, "a")) && h.acceptData(b)) {
                                        try {
                                            if (e && b[f])(l = b[e]) && (b[e] = null), h.event.triggered = f, b[f]()
                                            } catch (q) {}
                                        l && (b[e] = l);
                                        h.event.triggered = void 0
                                    }
                                }
                                return d.result
                            }
                        } else 
                            h.each(h.cache, function() {
                                var b = this[h.expando];
                                b && b.events &&
                                b.events[f] && h.event.trigger(d, a, b.handle.elem)
                            })
                        }
                }, handle: function(d) {
                    var d = h.event.fix(d || window.event), a = ((h._data(this, "events") || {})[d.type] || []).slice(0), b=!d.exclusive&&!d.namespace, e = Array.prototype.slice.call(arguments, 0);
                    e[0] = d;
                    d.currentTarget = this;
                    for (var f = 0, c = a.length; f < c; f++) {
                        var g = a[f];
                        if (b || d.namespace_re.test(g.namespace)) {
                            d.handler = g.handler;
                            d.data = g.data;
                            d.handleObj = g;
                            g = g.handler.apply(this, e);
                            if (g !== void 0)
                                d.result = g, g===!1 && (d.preventDefault(), d.stopPropagation());
                            if (d.isImmediatePropagationStopped())
                                break
                        }
                    }
                    return d.result
                },
                props: "altKey attrChange attrName bubbles button cancelable charCode clientX clientY ctrlKey currentTarget data detail eventPhase fromElement handler keyCode layerX layerY metaKey newValue offsetX offsetY pageX pageY prevValue relatedNode relatedTarget screenX screenY shiftKey srcElement target toElement view wheelDelta which".split(" "), fix : function(d) {
                    if (d[h.expando])
                        return d;
                    for (var a = d, d = h.Event(a), b = this.props.length, e; b;)
                        e = this.props[--b], d[e] = a[e];
                    if (!d.target)
                        d.target = d.srcElement || u;
                    if (d.target.nodeType ===
                    3)
                        d.target = d.target.parentNode;
                    if (!d.relatedTarget && d.fromElement)
                        d.relatedTarget = d.fromElement === d.target ? d.toElement : d.fromElement;
                    if (d.pageX == null && d.clientX != null)
                        b = d.target.ownerDocument || u, a = b.documentElement, b = b.body, d.pageX = d.clientX + (a && a.scrollLeft || b && b.scrollLeft || 0) - (a && a.clientLeft || b && b.clientLeft || 0), d.pageY = d.clientY + (a && a.scrollTop || b && b.scrollTop || 0) - (a && a.clientTop || b && b.clientTop || 0);
                    if (d.which == null && (d.charCode != null || d.keyCode != null))
                        d.which = d.charCode != null ? d.charCode : d.keyCode;
                    if (!d.metaKey && d.ctrlKey)
                        d.metaKey = d.ctrlKey;
                    if (!d.which && d.button !== void 0)
                        d.which = d.button & 1 ? 1 : d.button & 2 ? 3 : d.button & 4 ? 2 : 0;
                    return d
                }, guid: 1E8, proxy : h.proxy, special : {
                    ready: {
                        setup: h.bindReady, teardown : h.noop
                    }, live: {
                        add: function(d) {
                            h.event.add(this, g(d.origType, d.selector), h.extend({}, d, {
                                handler: n,
                                guid: d.handler.guid
                            }))
                        }, remove: function(d) {
                            h.event.remove(this, g(d.origType, d.selector), d)
                        }
                    }, beforeunload: {
                        amazonOriginal: null, setup : function(d, a, b) {
                            if (h.isWindow(this)) {
                                var e = function() {};
                                if (typeof this.onbeforeunload ===
                                "function")
                                    e = h.event.special.beforeunload.amazonOriginal = this.onbeforeunload;
                                this.onbeforeunload = function() {
                                    var d = Array.prototype.slice.call(arguments);
                                    e.apply(this, d);
                                    b.apply(this, d)
                                }
                            }
                        }, teardown: function() {
                            this.onbeforeunload = h.event.special.beforeunload.amazonOriginal
                        }
                    }
                }
            };
            h.removeEvent = u.removeEventListener ? function(d, a, b) {
                d.removeEventListener && d.removeEventListener(a, b, !1)
            } : function(d, a, b) {
                d.detachEvent && d.detachEvent("on" + a, b)
            };
            h.Event = function(d, a) {
                if (!this.preventDefault)
                    return new h.Event(d,
                    a);
                d && d.type ? (this.originalEvent = d, this.type = d.type, this.isDefaultPrevented = d.defaultPrevented || d.returnValue===!1 || d.getPreventDefault && d.getPreventDefault() ? m : i) : this.type = d;
                a && h.extend(this, a);
                this.timeStamp = h.now();
                this[h.expando]=!0
            };
            h.Event.prototype = {
                preventDefault: function() {
                    this.isDefaultPrevented = m;
                    var d = this.originalEvent;
                    if (d)
                        d.preventDefault ? d.preventDefault() : d.returnValue=!1
                },
                stopPropagation: function() {
                    this.isPropagationStopped = m;
                    var d = this.originalEvent;
                    if (d)
                        d.stopPropagation && d.stopPropagation(),
                        d.cancelBubble=!0
                },
                stopImmediatePropagation: function() {
                    this.isImmediatePropagationStopped = m;
                    this.stopPropagation()
                },
                isDefaultPrevented: i,
                isPropagationStopped: i,
                isImmediatePropagationStopped: i
            };
            var qa = function(d) {
                var a = d.relatedTarget, b=!1, e = d.type;
                d.type = d.data;
                if (a !== this && (a && (b = h.contains(this, a)), !b))
                    h.event.handle.apply(this, arguments), d.type = e
            }, ra = function(d) {
                d.type = d.data;
                h.event.handle.apply(this, arguments)
            };
            h.each({
                mouseenter: "mouseover",
                mouseleave: "mouseout"
            }, function(d, a) {
                h.event.special[d] =
                {
                    setup: function(b) {
                        h.event.add(this, a, b && b.selector ? ra : qa, d)
                    },
                    teardown: function(d) {
                        h.event.remove(this, a, d && d.selector ? ra : qa)
                    }
                }
            });
            if (!h.support.submitBubbles)
                h.event.special.submit = {
                    setup: function() {
                        if (h.nodeName(this, "form"))
                            return !1;
                        else 
                            h.event.add(this, "click.specialSubmit", function(d) {
                                var a = d.target, b = h.nodeName(a, "input") || h.nodeName(a, "button") ? a.type: "";
                                (b === "submit" || b === "image") && h(a).closest("form").length && j("submit", this, arguments)
                            }), h.event.add(this, "keypress.specialSubmit", function(d) {
                                var a =
                                d.target, b = h.nodeName(a, "input") || h.nodeName(a, "button") ? a.type: "";
                                (b === "text" || b === "password") && h(a).closest("form").length && d.keyCode === 13 && j("submit", this, arguments)
                            })
                        },
                        teardown: function() {
                            h.event.remove(this, ".specialSubmit")
                        }
                    };
            if (!h.support.changeBubbles) {
                var $, sa = function(d) {
                    var a = h.nodeName(d, "input") ? d.type: "", b = d.value;
                    if (a === "radio" || a === "checkbox")
                        b = d.checked;
                    else if (a === "select-multiple")
                        b = d.selectedIndex>-1 ? h.map(d.options, function(d) {
                            return d.selected
                        }).join("-") : "";
                    else if (h.nodeName(d,
                    "select"))
                        b = d.selectedIndex;
                    return b
                }, ba = function(d, a) {
                    var b = d.target, e, f;
                    if (ha.test(b.nodeName)&&!b.readOnly && (e = h._data(b, "_change_data"), f = sa(b), (d.type !== "focusout" || b.type !== "radio") && h._data(b, "_change_data", f), !(e === void 0 || f === e)))
                        if (e != null || f)
                            d.type = "change", d.liveFired = void 0, h.event.trigger(d, a, b)
                };
                h.event.special.change = {
                    filters: {
                        focusout: ba,
                        beforedeactivate: ba,
                        click: function(d) {
                            var a = d.target, b = h.nodeName(a, "input") ? a.type: "";
                            (b === "radio" || b === "checkbox" || h.nodeName(a, "select")) && ba.call(this,
                            d)
                        },
                        keydown: function(d) {
                            var a = d.target, b = h.nodeName(a, "input") ? a.type: "";
                            (d.keyCode === 13&&!h.nodeName(a, "textarea") || d.keyCode === 32 && (b === "checkbox" || b === "radio") || b === "select-multiple") && ba.call(this, d)
                        },
                        beforeactivate: function(d) {
                            d = d.target;
                            h._data(d, "_change_data", sa(d))
                        }
                    },
                    setup: function() {
                        if (this.type === "file")
                            return !1;
                        for (var d in $)
                            h.event.add(this, d + ".specialChange", $[d]);
                        return ha.test(this.nodeName)
                    },
                    teardown: function() {
                        h.event.remove(this, ".specialChange");
                        return ha.test(this.nodeName)
                    }
                };
                $ = h.event.special.change.filters;
                $.focus = $.beforeactivate
            }
            h.support.focusinBubbles || h.each({
                focus: "focusin",
                blur: "focusout"
            }, function(d, a) {
                function b(d) {
                    var e = h.event.fix(d);
                    e.type = a;
                    e.originalEvent = {};
                    h.event.trigger(e, null, e.target);
                    e.isDefaultPrevented() && d.preventDefault()
                }
                var e = 0;
                h.event.special[a] = {
                    setup: function() {
                        e++===0 && u.addEventListener(d, b, !0)
                    },
                    teardown: function() {
                        --e === 0 && u.removeEventListener(d, b, !0)
                    }
                }
            });
            h.each(["bind", "one"], function(d, a) {
                h.fn[a] = function(d, b, e) {
                    var f;
                    if (typeof d ===
                    "object") {
                        for (var c in d)
                            this[a](c, b, d[c], e);
                        return this
                    }
                    if (arguments.length === 2 || b===!1)
                        e = b, b = void 0;
                    a === "one" ? (f = function(d) {
                        h(this).unbind(d, f);
                        return e.apply(this, arguments)
                    }, f.guid = e.guid || h.guid++) : f = e;
                    if (d === "unload" && a !== "one")
                        this.one(d, b, e);
                    else {
                        c = 0;
                        for (var g = this.length; c < g; c++)
                            h.event.add(this[c], d, f, b)
                    }
                    return this
                }
            });
            h.fn.extend({
                unbind: function(d, a) {
                    if (typeof d === "object"&&!d.preventDefault)
                        for (var b in d)
                            this.unbind(b, d[b]);
                    else {
                        b = 0;
                        for (var e = this.length; b < e; b++)
                            h.event.remove(this[b],
                            d, a)
                    }
                    return this
                },
                delegate: function(d, a, b, e) {
                    return this.live(a, b, e, d)
                },
                undelegate: function(d, a, b) {
                    return arguments.length === 0 ? this.unbind("live") : this.die(a, null, b, d)
                },
                trigger: function(d, a) {
                    return this.each(function() {
                        h.event.trigger(d, a, this)
                    })
                },
                triggerHandler: function(d, a) {
                    if (this[0])
                        return h.event.trigger(d, a, this[0], !0)
                },
                toggle: function(d) {
                    var a = arguments, b = d.guid || h.guid++, e = 0, f = function(b) {
                        var f = (h.data(this, "lastToggle" + d.guid) || 0)%e;
                        h.data(this, "lastToggle" + d.guid, f + 1);
                        b.preventDefault();
                        return a[f].apply(this, arguments) ||!1
                    };
                    for (f.guid = b; e < a.length;)
                        a[e++].guid = b;
                    return this.click(f)
                },
                hover: function(d, a) {
                    return this.mouseenter(d).mouseleave(a || d)
                }
            });
            var ia = {
                focus: "focusin",
                blur: "focusout",
                mouseenter: "mouseover",
                mouseleave: "mouseout"
            };
            h.each(["live", "die"], function(d, a) {
                h.fn[a] = function(d, b, e, f) {
                    var c = 0, l, q, s = f || this.selector, j = f ? this: h(this.context);
                    if (typeof d === "object"&&!d.preventDefault) {
                        for (l in d)
                            j[a](l, b, d[l], s);
                        return this
                    }
                    if (a === "die"&&!d && f && f.charAt(0) === ".")
                        return j.unbind(f),
                        this;
                    if (b===!1 || h.isFunction(b))
                        e = b || i, b = void 0;
                    for (d = (d || "").split(" "); (f = d[c++]) != null;)
                        if (l = ea.exec(f), q = "", l && (q = l[0], f = f.replace(ea, "")), f === "hover")
                            d.push("mouseenter" + q, "mouseleave" + q);
                        else if (l = f, ia[f] ? (d.push(ia[f] + q), f += q) : f = (ia[f] || f) + q, a === "live") {
                            q = 0;
                            for (var n = j.length; q < n; q++)
                                h.event.add(j[q], "live." + g(f, s), {
                                    data: b,
                                    selector: s,
                                    handler: e,
                                    origType: f,
                                    origHandler: e,
                                    preType: l
                                })
                        } else 
                            j.unbind("live." + g(f, s), e);
                    return this
                }
            });
            h.each("blur focus focusin focusout load resize scroll unload click dblclick mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave change select submit keydown keypress keyup error".split(" "),
            function(d, a) {
                h.fn[a] = function(d, b) {
                    b == null && (b = d, d = null);
                    return arguments.length > 0 ? this.bind(a, d, b) : this.trigger(a)
                };
                h.attrFn && (h.attrFn[a]=!0)
            });
            (function() {
                function d(d, a, b, e, h, f) {
                    for (var h = 0, c = e.length; h < c; h++) {
                        var g = e[h];
                        if (g) {
                            for (var l=!1, g = g[d]; g;) {
                                if (g.sizcache === b) {
                                    l = e[g.sizset];
                                    break
                                }
                                if (g.nodeType === 1&&!f)
                                    g.sizcache = b, g.sizset = h;
                                if (g.nodeName.toLowerCase() === a) {
                                    l = g;
                                    break
                                }
                                g = g[d]
                            }
                            e[h] = l
                        }
                    }
                }
                function a(d, b, e, h, f, c) {
                    for (var f = 0, g = h.length; f < g; f++) {
                        var l = h[f];
                        if (l) {
                            for (var r=!1, l = l[d]; l;) {
                                if (l.sizcache ===
                                e) {
                                    r = h[l.sizset];
                                    break
                                }
                                if (l.nodeType === 1) {
                                    if (!c)
                                        l.sizcache = e, l.sizset = f;
                                    if (typeof b !== "string") {
                                        if (l === b) {
                                            r=!0;
                                            break
                                        }
                                    } else if (s.filter(b, [l]).length > 0) {
                                        r = l;
                                        break
                                    }
                                }
                                l = l[d]
                            }
                            h[f] = r
                        }
                    }
                }
                var b = /((?:\((?:\([^()]+\)|[^()]+)+\)|\[(?:\[[^\[\]]*\]|['"][^'"]*['"]|[^\[\]'"]+)+\]|\\.|[^ >+~,(\[\\]+)+|[>+~])(\s*,\s*)?((?:.|\r|\n)*)/g, e = 0, f = Object.prototype.toString, c=!1, g=!0, l = /\\/g, q = /\W/;
                [0, 0].sort(function() {
                    g=!1;
                    return 0
                });
                var s = function(d, a, e, h) {
                    var e = e || [], c = a = a || u;
                    if (a.nodeType !== 1 && a.nodeType !== 9)
                        return [];
                    if (!d || typeof d !== "string")
                        return e;
                    var g, l, r, q, v, i=!0, w = s.isXML(a), k = [], x = d;
                    do 
                        if (b.exec(""), g = b.exec(x))
                            if (x = g[3], k.push(g[1]), g[2]) {
                                q = g[3];
                                break
                            }
                    while (g);
                    if (k.length > 1 && n.exec(d))
                        if (k.length === 2 && j.relative[k[0]])
                            l = o(k[0] + k[1], a);
                        else 
                            for (l = j.relative[k[0]] ? [a] : s(k.shift(), a); k.length;)
                                d = k.shift(), j.relative[d] && (d += k.shift()), l = o(d, l);
                    else if (!h && k.length > 1 && a.nodeType === 9&&!w && j.match.ID.test(k[0])&&!j.match.ID.test(k[k.length - 1]) && (g = s.find(k.shift(), a, w), a = g.expr ? s.filter(g.expr, g.set)[0] :
                    g.set[0]), a) {
                        g = h ? {
                            expr: k.pop(),
                            set: m(h)
                        } : s.find(k.pop(), k.length === 1 && (k[0] === "~" || k[0] === "+") && a.parentNode ? a.parentNode : a, w);
                        l = g.expr ? s.filter(g.expr, g.set) : g.set;
                        for (k.length > 0 ? r = m(l) : i=!1; k.length;)
                            g = v = k.pop(), j.relative[v] ? g = k.pop() : v = "", g == null && (g = a), j.relative[v](r, g, w)
                    } else 
                        r = [];
                    r || (r = l);
                    r || s.error(v || d);
                    if (f.call(r) === "[object Array]")
                        if (i)
                            if (a && a.nodeType === 1)
                                for (d = 0; r[d] != null; d++)
                                    r[d] && (r[d]===!0 || r[d].nodeType === 1 && s.contains(a, r[d])) && e.push(l[d]);
                            else 
                                for (d = 0; r[d] != null; d++)
                                    r[d] &&
                                    r[d].nodeType === 1 && e.push(l[d]);
                    else 
                        e.push.apply(e, r);
                    else 
                        m(r, e);
                    q && (s(q, c, e, h), s.uniqueSort(e));
                    return e
                };
                s.uniqueSort = function(d) {
                    if (k && (c = g, d.sort(k), c))
                        for (var a = 1; a < d.length; a++)
                            d[a] === d[a - 1] && d.splice(a--, 1);
                    return d
                };
                s.matches = function(d, a) {
                    return s(d, null, null, a)
                };
                s.matchesSelector = function(d, a) {
                    return s(a, null, null, [d]).length > 0
                };
                s.find = function(d, a, b) {
                    var e;
                    if (!d)
                        return [];
                    for (var h = 0, f = j.order.length; h < f; h++) {
                        var c, g = j.order[h];
                        if (c = j.leftMatch[g].exec(d)) {
                            var r = c[1];
                            c.splice(1, 1);
                            if (r.substr(r.length -
                            1) !== "\\" && (c[1] = (c[1] || "").replace(l, ""), e = j.find[g](c, a, b), e != null)) {
                                d = d.replace(j.match[g], "");
                                break
                            }
                        }
                    }
                    e || (e = typeof a.getElementsByTagName !== "undefined" ? a.getElementsByTagName("*") : []);
                    return {
                        set: e,
                        expr: d
                    }
                };
                s.filter = function(d, a, b, e) {
                    for (var h, f, c = d, g = [], l = a, r = a && a[0] && s.isXML(a[0]); d && a.length;) {
                        for (var q in j.filter)
                            if ((h = j.leftMatch[q].exec(d)) != null && h[2]) {
                                var v, n, i = j.filter[q];
                                n = h[1];
                                f=!1;
                                h.splice(1, 1);
                                if (n.substr(n.length - 1) !== "\\") {
                                    l === g && (g = []);
                                    if (j.preFilter[q])
                                        if (h = j.preFilter[q](h, l,
                                        b, g, e, r)) {
                                            if (h===!0)
                                                continue
                                        } else 
                                            f = v=!0;
                                            if (h)
                                                for (var m = 0; (n = l[m]) != null; m++)
                                                    if (n) {
                                                        v = i(n, h, m, l);
                                                        var k = e^!!v;
                                                        b && v != null ? k ? f=!0 : l[m]=!1 : k && (g.push(n), f=!0)
                                                    }
                                                    if (v !== void 0) {
                                                        b || (l = g);
                                                        d = d.replace(j.match[q], "");
                                                        if (!f)
                                                            return [];
                                                            break
                                                    }
                                }
                            }
                        if (d === c)
                            if (f == null)
                                s.error(d);
                            else 
                                break;
                        c = d
                    }
                    return l
                };
                s.error = function(d) {
                    throw "Syntax error, unrecognized expression: " + d;
                };
                var j = s.selectors = {
                    order: ["ID", "NAME", "TAG"],
                    match: {
                        ID: /#((?:[\w\u00c0-\uFFFF\-]|\\.)+)/,
                        CLASS: /\.((?:[\w\u00c0-\uFFFF\-]|\\.)+)/,
                        NAME: /\[name=['"]*((?:[\w\u00c0-\uFFFF\-]|\\.)+)['"]*\]/,
                        ATTR: /\[\s*((?:[\w\u00c0-\uFFFF\-]|\\.)+)\s*(?:(\S?=)\s*(?:(['"])(.*?)\3|(#?(?:[\w\u00c0-\uFFFF\-]|\\.)*)|)|)\s*\]/,
                        TAG: /^((?:[\w\u00c0-\uFFFF\*\-]|\\.)+)/,
                        CHILD: /:(only|nth|last|first)-child(?:\(\s*(even|odd|(?:[+\-]?\d+|(?:[+\-]?\d*)?n\s*(?:[+\-]\s*\d+)?))\s*\))?/,
                        POS: /:(nth|eq|gt|lt|first|last|even|odd)(?:\((\d*)\))?(?=[^\-]|$)/,
                        PSEUDO: /:((?:[\w\u00c0-\uFFFF\-]|\\.)+)(?:\((['"]?)((?:\([^\)]+\)|[^\(\)]*)+)\2\))?/
                    },
                    leftMatch: {},
                    attrMap: {
                        "class": "className",
                        "for": "htmlFor"
                    },
                    attrHandle: {
                        href: function(d) {
                            return d.getAttribute("href")
                        },
                        type: function(d) {
                            return d.getAttribute("type")
                        }
                    },
                    relative: {
                        "+": function(d, a) {
                            var b = typeof a === "string", e = b&&!q.test(a), b = b&&!e;
                            e && (a = a.toLowerCase());
                            for (var e = 0, h = d.length, f; e < h; e++)
                                if (f = d[e]) {
                                    for (; (f = f.previousSibling) && f.nodeType !== 1;);
                                    d[e] = b || f && f.nodeName.toLowerCase() === a ? f ||!1 : f === a
                                }
                            b && s.filter(a, d, !0)
                        },
                        ">": function(d, a) {
                            var b, e = typeof a === "string", h = 0, f = d.length;
                            if (e&&!q.test(a))
                                for (a = a.toLowerCase(); h < f; h++) {
                                    if (b = d[h])
                                        b = b.parentNode, d[h] = b.nodeName.toLowerCase() === a ? b : !1
                                } else {
                                for (; h < f; h++)(b =
                                d[h]) 
                                    && (d[h] = e ? b.parentNode : b.parentNode === a);
                                e && s.filter(a, d, !0)
                            }
                        },
                        "": function(b, h, f) {
                            var c, g = e++, l = a;
                            typeof h === "string"&&!q.test(h) && (c = h = h.toLowerCase(), l = d);
                            l("parentNode", h, g, b, c, f)
                        },
                        "~": function(b, h, f) {
                            var c, g = e++, l = a;
                            typeof h === "string"&&!q.test(h) && (c = h = h.toLowerCase(), l = d);
                            l("previousSibling", h, g, b, c, f)
                        }
                    },
                    find: {
                        ID: function(d, a, b) {
                            if (typeof a.getElementById !== "undefined"&&!b)
                                return (d = a.getElementById(d[1])) && d.parentNode ? [d] : []
                            },
                        NAME: function(d, a) {
                            if (typeof a.getElementsByName !== "undefined") {
                                for (var b =
                                [], e = a.getElementsByName(d[1]), h = 0, f = e.length; h < f; h++)
                                    e[h].getAttribute("name") === d[1] && b.push(e[h]);
                                return b.length === 0 ? null : b
                            }
                        },
                        TAG: function(d, a) {
                            if (typeof a.getElementsByTagName !== "undefined")
                                return a.getElementsByTagName(d[1])
                        }
                    },
                    preFilter: {
                        CLASS: function(d, a, b, e, h, f) {
                            d = " " + d[1].replace(l, "") + " ";
                            if (f)
                                return d;
                            for (var f = 0, c; (c = a[f]) != null; f++)
                                c && (h^(c.className && (" " + c.className + " ").replace(/[\t\n\r]/g, " ").indexOf(d) >= 0) ? b || e.push(c) : b && (a[f]=!1));
                            return !1
                        },
                        ID: function(d) {
                            return d[1].replace(l,
                            "")
                        },
                        TAG: function(d) {
                            return d[1].replace(l, "").toLowerCase()
                        },
                        CHILD: function(d) {
                            if (d[1] === "nth") {
                                d[2] || s.error(d[0]);
                                d[2] = d[2].replace(/^\+|\s*/g, "");
                                var a = /(-?)(\d*)(?:n([+\-]?\d*))?/.exec(d[2] === "even" && "2n" || d[2] === "odd" && "2n+1" ||!/\D/.test(d[2]) && "0n+" + d[2] || d[2]);
                                d[2] = a[1] + (a[2] || 1) - 0;
                                d[3] = a[3] - 0
                            } else 
                                d[2] && s.error(d[0]);
                            d[0] = e++;
                            return d
                        },
                        ATTR: function(d, a, b, e, h, f) {
                            a = d[1] = d[1].replace(l, "");
                            !f && j.attrMap[a] && (d[1] = j.attrMap[a]);
                            d[4] = (d[4] || d[5] || "").replace(l, "");
                            d[2] === "~=" && (d[4] = " " + d[4] +
                            " ");
                            return d
                        },
                        PSEUDO: function(d, a, e, h, f) {
                            if (d[1] === "not")
                                if ((b.exec(d[3]) || "").length > 1 || /^\w/.test(d[3]))
                                    d[3] = s(d[3], null, null, a);
                                else 
                                    return d = s.filter(d[3], a, e, 1^f), e || h.push.apply(h, d), !1;
                            else if (j.match.POS.test(d[0]) || j.match.CHILD.test(d[0]))
                                return !0;
                            return d
                        },
                        POS: function(d) {
                            d.unshift(!0);
                            return d
                        }
                    },
                    filters: {
                        enabled: function(d) {
                            return d.disabled===!1 && d.type !== "hidden"
                        },
                        disabled: function(d) {
                            return d.disabled===!0
                        },
                        checked: function(d) {
                            return d.checked===!0
                        },
                        selected: function(d) {
                            return d.selected ===
                            !0
                        },
                        parent: function(d) {
                            return !!d.firstChild
                        },
                        empty: function(d) {
                            return !d.firstChild
                        },
                        has: function(d, a, b) {
                            return !!s(b[3], d).length
                        },
                        header: function(d) {
                            return /h\d/i.test(d.nodeName)
                        },
                        text: function(d) {
                            var a = d.getAttribute("type"), b = d.type;
                            return d.nodeName.toLowerCase() === "input" && "text" === b && (a === b || a === null)
                        },
                        radio: function(d) {
                            return d.nodeName.toLowerCase() === "input" && "radio" === d.type
                        },
                        checkbox: function(d) {
                            return d.nodeName.toLowerCase() === "input" && "checkbox" === d.type
                        },
                        file: function(d) {
                            return d.nodeName.toLowerCase() ===
                            "input" && "file" === d.type
                        },
                        password: function(d) {
                            return d.nodeName.toLowerCase() === "input" && "password" === d.type
                        },
                        submit: function(d) {
                            var a = d.nodeName.toLowerCase();
                            return (a === "input" || a === "button") && "submit" === d.type
                        },
                        image: function(d) {
                            return d.nodeName.toLowerCase() === "input" && "image" === d.type
                        },
                        reset: function(d) {
                            var a = d.nodeName.toLowerCase();
                            return (a === "input" || a === "button") && "reset" === d.type
                        },
                        button: function(d) {
                            var a = d.nodeName.toLowerCase();
                            return a === "input" && "button" === d.type || a === "button"
                        },
                        input: function(d) {
                            return /input|select|textarea|button/i.test(d.nodeName)
                        },
                        focus: function(d) {
                            return d === d.ownerDocument.activeElement
                        }
                    },
                    setFilters: {
                        first: function(d, a) {
                            return a === 0
                        },
                        last: function(d, a, b, e) {
                            return a === e.length - 1
                        },
                        even: function(d, a) {
                            return a%2 === 0
                        },
                        odd: function(d, a) {
                            return a%2 === 1
                        },
                        lt: function(d, a, b) {
                            return a < b[3] - 0
                        },
                        gt: function(d, a, b) {
                            return a > b[3] - 0
                        },
                        nth: function(d, a, b) {
                            return b[3] - 0 === a
                        },
                        eq: function(d, a, b) {
                            return b[3] - 0 === a
                        }
                    },
                    filter: {
                        PSEUDO: function(d, a, b, e) {
                            var h = a[1], f = j.filters[h];
                            if (f)
                                return f(d, b, a, e);
                            else if (h === "contains")
                                return (d.textContent || d.innerText ||
                                s.getText([d]) || "").indexOf(a[3]) >= 0;
                            else if (h === "not") {
                                a = a[3];
                                b = 0;
                                for (e = a.length; b < e; b++)
                                    if (a[b] === d)
                                        return !1;
                                return !0
                            } else 
                                s.error(h)
                        },
                        CHILD: function(d, a) {
                            var b = a[1], e = d;
                            switch (b) {
                            case "only":
                            case "first":
                                for (; e = e.previousSibling;)
                                    if (e.nodeType === 1)
                                        return !1;
                                if (b === "first")
                                    return !0;
                                e = d;
                            case "last":
                                for (; e = e.nextSibling;)
                                    if (e.nodeType === 1)
                                        return !1;
                                return !0;
                            case "nth":
                                var b = a[2], h = a[3];
                                if (b === 1 && h === 0)
                                    return !0;
                                var f = a[0], c = d.parentNode;
                                if (c && (c.sizcache !== f ||!d.nodeIndex)) {
                                    for (var g = 0, e = c.firstChild; e; e =
                                    e.nextSibling)
                                        if (e.nodeType === 1)
                                            e.nodeIndex=++g;
                                    c.sizcache = f
                                }
                                e = d.nodeIndex - h;
                                return b === 0 ? e === 0 : e%b === 0 && e / b >= 0
                            }
                        },
                        ID: function(d, a) {
                            return d.nodeType === 1 && d.getAttribute("id") === a
                        },
                        TAG: function(d, a) {
                            return a === "*" && d.nodeType === 1 || d.nodeName.toLowerCase() === a
                        },
                        CLASS: function(d, a) {
                            return (" " + (d.className || d.getAttribute("class")) + " ").indexOf(a)>-1
                        },
                        ATTR: function(d, a) {
                            var b = a[1], b = j.attrHandle[b] ? j.attrHandle[b](d): d[b] != null ? d[b]: d.getAttribute(b), e = b + "", h = a[2], f = a[4];
                            return b == null ? h === "!=" : h === "=" ?
                            e === f : h === "*=" ? e.indexOf(f) >= 0 : h === "~=" ? (" " + e + " ").indexOf(f) >= 0 : !f ? e && b!==!1 : h === "!=" ? e !== f : h === "^=" ? e.indexOf(f) === 0 : h === "$=" ? e.substr(e.length - f.length) === f : h === "|=" ? e === f || e.substr(0, f.length + 1) === f + "-" : !1
                        },
                        POS: function(d, a, b, e) {
                            var h = j.setFilters[a[2]];
                            if (h)
                                return h(d, b, a, e)
                        }
                    }
                }, n = j.match.POS, v = function(d, a) {
                    return "\\" + (a - 0 + 1)
                }, i;
                for (i in j.match)
                    j.match[i] = RegExp(j.match[i].source + /(?![^\[]*\])(?![^\(]*\))/.source), j.leftMatch[i] = RegExp(/(^(?:.|\r|\n)*?)/.source + j.match[i].source.replace(/\\(\d+)/g,
                    v));
                var m = function(d, a) {
                    d = Array.prototype.slice.call(d, 0);
                    return a ? (a.push.apply(a, d), a) : d
                };
                try {
                    Array.prototype.slice.call(u.documentElement.childNodes, 0)
                } catch (w) {
                    m = function(d, a) {
                        var b = 0, e = a || [];
                        if (f.call(d) === "[object Array]")
                            Array.prototype.push.apply(e, d);
                        else if (typeof d.length === "number")
                            for (var h = d.length; b < h; b++)
                                e.push(d[b]);
                        else 
                            for (; d[b]; b++)
                                e.push(d[b]);
                        return e
                    }
                }
                var k, x;
                u.documentElement.compareDocumentPosition ? k = function(d, a) {
                    return d === a ? (c=!0, 0) : !d.compareDocumentPosition ||!a.compareDocumentPosition ?
                    d.compareDocumentPosition?-1 : 1 : d.compareDocumentPosition(a) & 4?-1 : 1
                } : (k = function(d, a) {
                    if (d === a)
                        return c=!0, 0;
                    else if (d.sourceIndex && a.sourceIndex)
                        return d.sourceIndex - a.sourceIndex;
                    var b, e, h = [], f = [];
                    b = d.parentNode;
                    e = a.parentNode;
                    var g = b;
                    if (b === e)
                        return x(d, a);
                    else if (b) {
                        if (!e)
                            return 1
                    } else 
                        return - 1;
                    for (; g;)
                        h.unshift(g), g = g.parentNode;
                    for (g = e; g;)
                        f.unshift(g), g = g.parentNode;
                    b = h.length;
                    e = f.length;
                    for (g = 0; g < b && g < e; g++)
                        if (h[g] !== f[g])
                            return x(h[g], f[g]);
                    return g === b ? x(d, f[g], - 1) : x(h[g], a, 1)
                }, x = function(d,
                a, b) {
                    if (d === a)
                        return b;
                    for (d = d.nextSibling; d;) {
                        if (d === a)
                            return - 1;
                        d = d.nextSibling
                    }
                    return 1
                });
                s.getText = function(d) {
                    for (var a = "", b, e = 0; d[e]; e++)
                        b = d[e], b.nodeType === 3 || b.nodeType === 4 ? a += b.nodeValue : b.nodeType !== 8 && (a += s.getText(b.childNodes));
                    return a
                };
                (function() {
                    var d = u.createElement("div"), a = "script" + (new Date).getTime(), b = u.documentElement;
                    d.innerHTML = "<a name='" + a + "'/>";
                    b.insertBefore(d, b.firstChild);
                    if (u.getElementById(a))
                        j.find.ID = function(d, a, b) {
                            if (typeof a.getElementById !== "undefined"&&!b)
                                return (a =
                                a.getElementById(d[1])) ? a.id === d[1] || typeof a.getAttributeNode !== "undefined" && a.getAttributeNode("id").nodeValue === d[1] ? [a] : void 0 : []
                            }, j.filter.ID = function(d, a) {
                                var b = typeof d.getAttributeNode !== "undefined" && d.getAttributeNode("id");
                                return d.nodeType === 1 && b && b.nodeValue === a
                            };
                    b.removeChild(d);
                    b = d = null
                })();
                (function() {
                    var d = u.createElement("div");
                    d.appendChild(u.createComment(""));
                    if (d.getElementsByTagName("*").length > 0)
                        j.find.TAG = function(d, a) {
                            var b = a.getElementsByTagName(d[1]);
                            if (d[1] === "*") {
                                for (var e =
                                [], h = 0; b[h]; h++)
                                    b[h].nodeType === 1 && e.push(b[h]);
                                    b = e
                            }
                            return b
                        };
                    d.innerHTML = "<a href='#'></a>";
                    if (d.firstChild && typeof d.firstChild.getAttribute !== "undefined" && d.firstChild.getAttribute("href") !== "#")
                        j.attrHandle.href = function(d) {
                            return d.getAttribute("href", 2)
                        };
                    d = null
                })();
                u.querySelectorAll && function() {
                    var d = s, a = u.createElement("div");
                    a.innerHTML = "<p class='TEST'></p>";
                    if (!(a.querySelectorAll && a.querySelectorAll(".TEST").length === 0)) {
                        s = function(a, b, e, h) {
                            b = b || u;
                            if (!h&&!s.isXML(b)) {
                                var f = /^(\w+$)|^\.([\w\-]+$)|^#([\w\-]+$)/.exec(a);
                                if (f && (b.nodeType === 1 || b.nodeType === 9))
                                    if (f[1])
                                        return m(b.getElementsByTagName(a), e);
                                    else if (f[2] && j.find.CLASS && b.getElementsByClassName)
                                        return m(b.getElementsByClassName(f[2]), e);
                                if (b.nodeType === 9) {
                                    if (a === "body" && b.body)
                                        return m([b.body], e);
                                    else if (f && f[3]) {
                                        var c = b.getElementById(f[3]);
                                        if (c && c.parentNode) {
                                            if (c.id === f[3])
                                                return m([c], e)
                                            } else 
                                                return m([], e)
                                            }
                                    try {
                                        return m(b.querySelectorAll(a), e)
                                    } catch (g) {}
                                } else if (b.nodeType === 1 && b.nodeName.toLowerCase() !== "object") {
                                    var f = b, l = (c = b.getAttribute("id")) ||
                                    "__sizzle__", r = b.parentNode, q = /^\s*[+~]/.test(a);
                                    c ? l = l.replace(/'/g, "\\$&") : b.setAttribute("id", l);
                                    if (q && r)
                                        b = b.parentNode;
                                    try {
                                        if (!q || r)
                                            return m(b.querySelectorAll("[id='" + l + "'] " + a), e)
                                        } catch (n) {} finally {
                                        c || f.removeAttribute("id")
                                    }
                                }
                            }
                            return d(a, b, e, h)
                        };
                        for (var b in d)
                            s[b] = d[b];
                        a = null
                    }
                }();
                (function() {
                    var d = u.documentElement, a = d.matchesSelector || d.mozMatchesSelector || d.webkitMatchesSelector || d.msMatchesSelector;
                    if (a) {
                        var b=!a.call(u.createElement("div"), "div"), e=!1;
                        try {
                            a.call(u.documentElement, "[test!='']:sizzle")
                        } catch (h) {
                            e =
                            !0
                        }
                        s.matchesSelector = function(d, h) {
                            h = h.replace(/\=\s*([^'"\]]*)\s*\]/g, "='$1']");
                            if (!s.isXML(d))
                                try {
                                    if (e ||!j.match.PSEUDO.test(h)&&!/!=/.test(h)) {
                                        var f = a.call(d, h);
                                        if (f ||!b || d.document && d.document.nodeType !== 11)
                                            return f
                                    }
                            } catch (c) {}
                            return s(h, null, null, [d]).length > 0
                        }
                    }
                })();
                (function() {
                    var d = u.createElement("div");
                    d.innerHTML = "<div class='test e'></div><div class='test'></div>";
                    if (d.getElementsByClassName && d.getElementsByClassName("e").length !== 0 && (d.lastChild.className = "e", d.getElementsByClassName("e").length !==
                    1))
                        j.order.splice(1, 0, "CLASS"), j.find.CLASS = function(d, a, b) {
                            if (typeof a.getElementsByClassName !== "undefined"&&!b)
                                return a.getElementsByClassName(d[1])
                            }, d = null
                })();
                s.contains = u.documentElement.contains ? function(d, a) {
                    return d !== a && (d.contains ? d.contains(a) : !0)
                } : u.documentElement.compareDocumentPosition ? function(d, a) {
                    return !!(d.compareDocumentPosition(a) & 16)
                } : function() {
                    return !1
                };
                s.isXML = function(d) {
                    return (d = (d ? d.ownerDocument || d : 0).documentElement) ? d.nodeName !== "HTML" : !1
                };
                var o = function(d, a) {
                    for (var b,
                    e = [], h = "", f = a.nodeType ? [a] : a; b = j.match.PSEUDO.exec(d);)
                        h += b[0], d = d.replace(j.match.PSEUDO, "");
                    d = j.relative[d] ? d + "*" : d;
                    b = 0;
                    for (var c = f.length; b < c; b++)
                        s(d, f[b], e);
                    return s.filter(h, e)
                };
                h.find = s;
                h.expr = s.selectors;
                h.expr[":"] = h.expr.filters;
                h.unique = s.uniqueSort;
                h.text = s.getText;
                h.isXMLDoc = s.isXML;
                h.contains = s.contains
            })();
            var Ta = /Until$/, Ua = /^(?:parents|prevUntil|prevAll)/, Va = /,/, Ka = /^.[^:#\[\.,]*$/, Wa = Array.prototype.slice, ta = h.expr.match.POS, Xa = {
                children: !0,
                contents: !0,
                next: !0,
                prev: !0
            };
            h.fn.extend({
                find: function(d) {
                    var a =
                    this, b, e;
                    if (typeof d !== "string")
                        return h(d).filter(function() {
                            for (b = 0, e = a.length; b < e; b++)
                                if (h.contains(a[b], this))
                                    return !0
                                });
                    var f = this.pushStack("", "find", d), c, g, l;
                    for (b = 0, e = this.length; b < e; b++)
                        if (c = f.length, h.find(d, this[b], f), b > 0)
                            for (g = c; g < f.length; g++)
                                for (l = 0; l < c; l++)
                                    if (f[l] === f[g]) {
                                        f.splice(g--, 1);
                                        break
                                    }
                    return f
                },
                has: function(d) {
                    var a = h(d);
                    return this.filter(function() {
                        for (var d = 0, b = a.length; d < b; d++)
                            if (h.contains(this, a[d]))
                                return !0
                    })
                },
                not: function(d) {
                    return this.pushStack(f(this, d, !1), "not",
                    d)
                },
                filter: function(d) {
                    return this.pushStack(f(this, d, !0), "filter", d)
                },
                is: function(d) {
                    return !!d && (typeof d === "string" ? h.filter(d, this).length > 0 : this.filter(d).length > 0)
                },
                closest: function(d, a) {
                    var b = [], e, f, c = this[0];
                    if (h.isArray(d)) {
                        var g, l = {}, q = 1;
                        if (c && d.length) {
                            for (e = 0, f = d.length; e < f; e++)
                                g = d[e], l[g] || (l[g] = ta.test(g) ? h(g, a || this.context) : g);
                            for (; c && c.ownerDocument && c !== a;) {
                                for (g in l)
                                    e = l[g], (e.jquery ? e.index(c)>-1 : h(c).is(e)) && b.push({
                                        selector: g,
                                        elem: c,
                                        level: q
                                    });
                                c = c.parentNode;
                                q++
                            }
                        }
                        return b
                    }
                    g = ta.test(d) ||
                    typeof d !== "string" ? h(d, a || this.context) : 0;
                    for (e = 0, f = this.length; e < f; e++)
                        for (c = this[e]; c;)
                            if (g ? g.index(c)>-1 : h.find.matchesSelector(c, d)) {
                                b.push(c);
                                break
                            } else if (c = c.parentNode, !c ||!c.ownerDocument || c === a || c.nodeType === 11)
                                break;
                    b = b.length > 1 ? h.unique(b) : b;
                    return this.pushStack(b, "closest", d)
                },
                index: function(d) {
                    return !d ? this[0] && this[0].parentNode ? this.prevAll().length : - 1 : typeof d === "string" ? h.inArray(this[0], h(d)) : h.inArray(d.jquery ? d[0] : d, this)
                },
                add: function(d, a) {
                    var b = typeof d === "string" ? h(d, a):
                    h.makeArray(d && d.nodeType ? [d] : d), e = h.merge(this.get(), b);
                    return this.pushStack(!b[0] ||!b[0].parentNode || b[0].parentNode.nodeType === 11 ||!e[0] ||!e[0].parentNode || e[0].parentNode.nodeType === 11 ? e : h.unique(e))
                },
                andSelf: function() {
                    return this.add(this.prevObject)
                }
            });
            h.each({
                parent: function(d) {
                    return (d = d.parentNode) && d.nodeType !== 11 ? d : null
                },
                parents: function(d) {
                    return h.dir(d, "parentNode")
                },
                parentsUntil: function(d, a, b) {
                    return h.dir(d, "parentNode", b)
                },
                next: function(d) {
                    return h.nth(d, 2, "nextSibling")
                },
                prev: function(d) {
                    return h.nth(d,
                    2, "previousSibling")
                },
                nextAll: function(d) {
                    return h.dir(d, "nextSibling")
                },
                prevAll: function(d) {
                    return h.dir(d, "previousSibling")
                },
                nextUntil: function(d, a, b) {
                    return h.dir(d, "nextSibling", b)
                },
                prevUntil: function(d, a, b) {
                    return h.dir(d, "previousSibling", b)
                },
                siblings: function(d) {
                    return h.sibling(d.parentNode.firstChild, d)
                },
                children: function(d) {
                    return h.sibling(d.firstChild)
                },
                contents: function(d) {
                    return h.nodeName(d, "iframe") ? d.contentDocument || d.contentWindow.document : h.makeArray(d.childNodes)
                }
            }, function(d,
            a) {
                h.fn[d] = function(b, e) {
                    var f = h.map(this, a, b), c = Wa.call(arguments);
                    Ta.test(d) || (e = b);
                    e && typeof e === "string" && (f = h.filter(e, f));
                    f = this.length > 1&&!Xa[d] ? h.unique(f) : f;
                    if ((this.length > 1 || Va.test(e)) && Ua.test(d))
                        f = f.reverse();
                    return this.pushStack(f, d, c.join(","))
                }
            });
            h.extend({
                filter: function(d, a, b) {
                    b && (d = ":not(" + d + ")");
                    return a.length === 1 ? h.find.matchesSelector(a[0], d) ? [a[0]] : [] : h.find.matches(d, a)
                },
                dir: function(d, a, b) {
                    for (var e = [], d = d[a]; d && d.nodeType !== 9 && (b === void 0 || d.nodeType !== 1 ||!h(d).is(b));
                    )d.nodeType ===
                    1 && e.push(d), d = d[a];
                    return e
                },
                nth: function(d, a, b) {
                    for (var a = a || 1, e = 0; d; d = d[b])
                        if (d.nodeType === 1&&++e === a)
                            break;
                    return d
                },
                sibling: function(d, a) {
                    for (var b = []; d; d = d.nextSibling)
                        d.nodeType === 1 && d !== a && b.push(d);
                    return b
                }
            });
            var Ya = / jQuery\d+="(?:\d+|null)"/g, ja = /^\s+/, ua = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\w:]+)[^>]*)\/>/ig, va = /<([\w:]+)/, Za = /<tbody/i, $a = /<|&#?\w+;/, wa = /<(?:script|object|embed|option|style)/i, xa = /checked\s*(?:[^=]|=\s*.checked.)/i, ab = /\/(java|ecma)script/i, La = /^\s*<!(?:\[CDATA\[|\-\-)/,
            N = {
                option: [1, "<select multiple='multiple'>", "</select>"],
                legend: [1, "<fieldset>", "</fieldset>"],
                thead: [1, "<table>", "</table>"],
                tr: [2, "<table><tbody>", "</tbody></table>"],
                td: [3, "<table><tbody><tr>", "</tr></tbody></table>"],
                col: [2, "<table><tbody></tbody><colgroup>", "</colgroup></table>"],
                area: [1, "<map>", "</map>"],
                _default: [0, "", ""]
            };
            N.optgroup = N.option;
            N.tbody = N.tfoot = N.colgroup = N.caption = N.thead;
            N.th = N.td;
            if (!h.support.htmlSerialize)
                N._default = [1, "div<div>", "</div>"];
            h.fn.extend({
                text: function(d) {
                    return h.isFunction(d) ?
                    this.each(function(a) {
                        var b = h(this);
                        b.text(d.call(this, a, b.text()))
                    }) : typeof d !== "object" && d !== void 0 ? this.empty().append((this[0] && this[0].ownerDocument || u).createTextNode(d)) : h.text(this)
                },
                wrapAll: function(d) {
                    if (h.isFunction(d))
                        return this.each(function(a) {
                            h(this).wrapAll(d.call(this, a))
                        });
                    if (this[0]) {
                        var a = h(d, this[0].ownerDocument).eq(0).clone(!0);
                        this[0].parentNode && a.insertBefore(this[0]);
                        a.map(function() {
                            for (var d = this; d.firstChild && d.firstChild.nodeType === 1;)
                                d = d.firstChild;
                            return d
                        }).append(this)
                    }
                    return this
                },
                wrapInner: function(d) {
                    return h.isFunction(d) ? this.each(function(a) {
                        h(this).wrapInner(d.call(this, a))
                    }) : this.each(function() {
                        var a = h(this), b = a.contents();
                        b.length ? b.wrapAll(d) : a.append(d)
                    })
                },
                wrap: function(d) {
                    return this.each(function() {
                        h(this).wrapAll(d)
                    })
                },
                unwrap: function() {
                    return this.parent().each(function() {
                        h.nodeName(this, "body") || h(this).replaceWith(this.childNodes)
                    }).end()
                },
                append: function() {
                    return this.domManip(arguments, !0, function(d) {
                        this.nodeType === 1 && this.appendChild(d)
                    })
                },
                prepend: function() {
                    return this.domManip(arguments,
                    !0, function(d) {
                        this.nodeType === 1 && this.insertBefore(d, this.firstChild)
                    })
                },
                before: function() {
                    if (this[0] && this[0].parentNode)
                        return this.domManip(arguments, !1, function(d) {
                            this.parentNode.insertBefore(d, this)
                        });
                    else if (arguments.length) {
                        var d = h(arguments[0]);
                        d.push.apply(d, this.toArray());
                        return this.pushStack(d, "before", arguments)
                    }
                },
                after: function() {
                    if (this[0] && this[0].parentNode)
                        return this.domManip(arguments, !1, function(d) {
                            this.parentNode.insertBefore(d, this.nextSibling)
                        });
                    else if (arguments.length) {
                        var d =
                        this.pushStack(this, "after", arguments);
                        d.push.apply(d, h(arguments[0]).toArray());
                        return d
                    }
                },
                remove: function(d, a) {
                    for (var b = 0, e; (e = this[b]) != null; b++)
                        if (!d || h.filter(d, [e]).length)!a && e.nodeType === 1 && (h.cleanData(e.getElementsByTagName("*")), h.cleanData([e])), e.parentNode && e.parentNode.removeChild(e);
                    return this
                },
                empty: function() {
                    for (var d = 0, a; (a = this[d]) != null; d++)
                        for (a.nodeType === 1 && h.cleanData(a.getElementsByTagName("*")); a.firstChild;)
                            a.removeChild(a.firstChild);
                    return this
                },
                clone: function(d, a) {
                    d =
                    d == null?!1 : d;
                    a = a == null ? d : a;
                    return this.map(function() {
                        return h.clone(this, d, a)
                    })
                },
                html: function(d) {
                    if (d === void 0)
                        return this[0] && this[0].nodeType === 1 ? this[0].innerHTML.replace(Ya, "") : null;
                    else if (typeof d === "string"&&!wa.test(d) && (h.support.leadingWhitespace ||!ja.test(d))&&!N[(va.exec(d) || ["", ""])[1].toLowerCase()]) {
                        d = d.replace(ua, "<$1></$2>");
                        try {
                            for (var a = 0, b = this.length; a < b; a++)
                                if (this[a].nodeType === 1)
                                    h.cleanData(this[a].getElementsByTagName("*")), this[a].innerHTML = d
                        } catch (e) {
                            this.empty().append(d)
                        }
                    } else 
                        h.isFunction(d) ?
                        this.each(function(a) {
                            var b = h(this);
                            b.html(d.call(this, a, b.html()))
                        }) : this.empty().append(d);
                    return this
                },
                replaceWith: function(d) {
                    if (this[0] && this[0].parentNode) {
                        if (h.isFunction(d))
                            return this.each(function(a) {
                                var b = h(this), e = b.html();
                                b.replaceWith(d.call(this, a, e))
                            });
                        typeof d !== "string" && (d = h(d).detach());
                        return this.each(function() {
                            var a = this.nextSibling, b = this.parentNode;
                            h(this).remove();
                            a ? h(a).before(d) : h(b).append(d)
                        })
                    } else 
                        return this.length ? this.pushStack(h(h.isFunction(d) ? d() : d), "replaceWith",
                        d) : this
                },
                detach: function(d) {
                    return this.remove(d, !0)
                },
                domManip: function(d, a, b) {
                    var e, f, c, g = d[0], l = [];
                    if (!h.support.checkClone && arguments.length === 3 && typeof g === "string" && xa.test(g))
                        return this.each(function() {
                            h(this).domManip(d, a, b, !0)
                        });
                    if (h.isFunction(g))
                        return this.each(function(e) {
                            var f = h(this);
                            d[0] = g.call(this, e, a ? f.html() : void 0);
                            f.domManip(d, a, b)
                        });
                    if (this[0]) {
                        e = g && g.parentNode;
                        e = h.support.parentNode && e && e.nodeType === 11 && e.childNodes.length === this.length ? {
                            fragment: e
                        } : h.buildFragment(d, this,
                        l);
                        c = e.fragment;
                        if (f = c.childNodes.length === 1 ? c = c.firstChild : c.firstChild) {
                            a = a && h.nodeName(f, "tr");
                            f = 0;
                            for (var q = this.length, j = q - 1; f < q; f++)
                                b.call(a ? h.nodeName(this[f], "table") ? this[f].getElementsByTagName("tbody")[0] || this[f].appendChild(this[f].ownerDocument.createElement("tbody")) : this[f] : this[f], e.cacheable || q > 1 && f < j ? h.clone(c, !0, !0) : c)
                            }
                        l.length && h.each(l, s)
                    }
                    return this
                }
            });
            h.buildFragment = function(d, a, b) {
                var e, f, c, g;
                a && a[0] && (g = a[0].ownerDocument || a[0]);
                g.createDocumentFragment || (g = u);
                if (d.length ===
                1 && typeof d[0] === "string" && d[0].length < 512 && g === u && d[0].charAt(0) === "<"&&!wa.test(d[0]) && (h.support.checkClone ||!xa.test(d[0])))
                    f=!0, (c = h.fragments[d[0]]) && c !== 1 && (e = c);
                e || (e = g.createDocumentFragment(), h.clean(d, g, e, b));
                f && (h.fragments[d[0]] = c ? e : 1);
                return {
                    fragment: e,
                    cacheable: f
                }
            };
            h.fragments = {};
            h.each({
                appendTo: "append",
                prependTo: "prepend",
                insertBefore: "before",
                insertAfter: "after",
                replaceAll: "replaceWith"
            }, function(d, a) {
                h.fn[d] = function(b) {
                    var e = [], b = h(b), f = this.length === 1 && this[0].parentNode;
                    if (f &&
                    f.nodeType === 11 && f.childNodes.length === 1 && b.length === 1)
                        return b[a](this[0]), this;
                    else {
                        for (var f = 0, c = b.length; f < c; f++) {
                            var g = (f > 0 ? this.clone(!0) : this).get();
                            h(b[f])[a](g);
                            e = e.concat(g)
                        }
                        return this.pushStack(e, d, b.selector)
                    }
                }
            });
            h.extend({
                clone: function(d, f, c) {
                    var g = d.cloneNode(!0), l, q, s;
                    if ((!h.support.noCloneEvent ||!h.support.noCloneChecked) && (d.nodeType === 1 || d.nodeType === 11)&&!h.isXMLDoc(d)) {
                        e(d, g);
                        l = a(d);
                        q = a(g);
                        for (s = 0; l[s]; ++s)
                            q[s] && e(l[s], q[s])
                    }
                    if (f && (b(d, g), c)) {
                        l = a(d);
                        q = a(g);
                        for (s = 0; l[s]; ++s)
                            b(l[s],
                            q[s])
                    }
                    return g
                },
                clean: function(d, a, b, e) {
                    a = a || u;
                    typeof a.createElement === "undefined" && (a = a.ownerDocument || a[0] && a[0].ownerDocument || u);
                    for (var f = [], c, g = 0, l; (l = d[g]) != null; g++)
                        if (typeof l === "number" && (l += ""), l) {
                            if (typeof l === "string")
                                if ($a.test(l)) {
                                    l = l.replace(ua, "<$1></$2>");
                                    c = (va.exec(l) || ["", ""])[1].toLowerCase();
                                    var s = N[c] || N._default, j = s[0], n = a.createElement("div");
                                    for (n.innerHTML = s[1] + l + s[2]; j--;)
                                        n = n.lastChild;
                                        if (!h.support.tbody) {
                                            j = Za.test(l);
                                            s = c === "table"&&!j ? n.firstChild && n.firstChild.childNodes :
                                            s[1] === "<table>"&&!j ? n.childNodes : [];
                                            for (c = s.length - 1; c >= 0; --c)
                                                h.nodeName(s[c], "tbody")&&!s[c].childNodes.length && s[c].parentNode.removeChild(s[c])
                                            }
                                            !h.support.leadingWhitespace && ja.test(l) && n.insertBefore(a.createTextNode(ja.exec(l)[0]), n.firstChild);
                                            l = n.childNodes
                                    } else 
                                        l = a.createTextNode(l);
                                        var v;
                                        if (!h.support.appendChecked)
                                            if (l[0] && typeof(v = l.length) === "number")
                                                for (c = 0; c < v; c++)
                                                    q(l[c]);
                                            else 
                                                q(l);
                                                l.nodeType ? f.push(l) : f = h.merge(f, l)
                            }
                    if (b) {
                        d = function(d) {
                            return !d.type || ab.test(d.type)
                        };
                        for (g = 0; f[g]; g++)
                            e &&
                            h.nodeName(f[g], "script") && (!f[g].type || f[g].type.toLowerCase() === "text/javascript") ? e.push(f[g].parentNode ? f[g].parentNode.removeChild(f[g]) : f[g]) : (f[g].nodeType === 1 && (a = h.grep(f[g].getElementsByTagName("script"), d), f.splice.apply(f, [g + 1, 0].concat(a))), b.appendChild(f[g]))
                    }
                    return f
                },
                cleanData: function(d) {
                    for (var a, b, e = h.cache, f = h.expando, c = h.event.special, g = h.support.deleteExpando, l = 0, q; (q = d[l]) != null; l++)
                        if (!q.nodeName ||!h.noData[q.nodeName.toLowerCase()])
                            if (b = q[h.expando]) {
                                if ((a = e[b] && e[b][f]) &&
                                a.events) {
                                    for (var s in a.events)
                                        c[s] ? h.event.remove(q, s) : h.removeEvent(q, s, a.handle);
                                        if (a.handle)
                                            a.handle.elem = null
                                }
                                g ? delete q[h.expando] : q.removeAttribute && q.removeAttribute(h.expando);
                                delete e[b]
                            }
                    }
            });
            var ka = /alpha\([^)]*\)/i, bb = /opacity=([^)]*)/, cb = /([A-Z]|^ms)/g, ya = /^-?\d+(?:px)?$/i, db = /^-?\d/, eb = /^([\-+])=([\-+.\de]+)/, fb = {
                position: "absolute",
                visibility: "hidden",
                display: "block"
            }, Ma = ["Left", "Right"], Na = ["Top", "Bottom"], W, za, Aa;
            h.fn.css = function(d, a) {
                return arguments.length === 2 && a === void 0 ? this :
                h.access(this, d, a, !0, function(d, a, b) {
                    return b !== void 0 ? h.style(d, a, b) : h.css(d, a)
                })
            };
            h.extend({
                cssHooks: {
                    opacity: {
                        get: function(d, a) {
                            if (a) {
                                var b = W(d, "opacity", "opacity");
                                return b === "" ? "1" : b
                            } else 
                                return d.style.opacity
                        }
                    }
                },
                cssNumber: {
                    fillOpacity: !0,
                    fontWeight: !0,
                    lineHeight: !0,
                    opacity: !0,
                    orphans: !0,
                    widows: !0,
                    zIndex: !0,
                    zoom: !0
                },
                cssProps: {
                    "float": h.support.cssFloat ? "cssFloat": "styleFloat"
                },
                style: function(d, a, b, e) {
                    if (d&&!(d.nodeType === 3 || d.nodeType === 8 ||!d.style)) {
                        var f, c = h.camelCase(a), g = d.style, l = h.cssHooks[c],
                        a = h.cssProps[c] || c;
                        if (b !== void 0) {
                            e = typeof b;
                            if (e === "string" && (f = eb.exec(b)))
                                b =+ (f[1] + 1)*+f[2] + parseFloat(h.css(d, a)), e = "number";
                            if (!(b == null || e === "number" && isNaN(b)))
                                if (e === "number"&&!h.cssNumber[c] && (b += "px"), !l ||!("set"in l) || (b = l.set(d, b)) !== void 0)
                                    try {
                                        g[a] = b
                            } catch (q) {}
                        } else 
                            return l && "get"in l && (f = l.get(d, !1, e)) !== void 0 ? f : g[a]
                    }
                },
                css: function(d, a, b) {
                    var e, f, a = h.camelCase(a);
                    f = h.cssHooks[a];
                    a = h.cssProps[a] || a;
                    a === "cssFloat" && (a = "float");
                    if (f && "get"in f && (e = f.get(d, !0, b)) !== void 0)
                        return e;
                    else if (W)
                        return W(d,
                        a)
                },
                swap: function(d, a, b) {
                    var e = {}, f;
                    for (f in a)
                        e[f] = d.style[f], d.style[f] = a[f];
                    b.call(d);
                    for (f in a)
                        d.style[f] = e[f]
                }
            });
            h.curCSS = h.css;
            h.each(["height", "width"], function(d, a) {
                h.cssHooks[a] = {
                    get: function(d, b, e) {
                        var f;
                        if (b) {
                            if (d.offsetWidth !== 0)
                                return x(d, a, e);
                            else 
                                h.swap(d, fb, function() {
                                    f = x(d, a, e)
                                });
                            return f
                        }
                    },
                    set: function(d, a) {
                        if (ya.test(a)) {
                            if (a = parseFloat(a), a >= 0)
                                return a + "px"
                        } else 
                            return a
                    }
                }
            });
            if (!h.support.opacity)
                h.cssHooks.opacity = {
                    get: function(d, a) {
                        return bb.test((a && d.currentStyle ? d.currentStyle.filter :
                        d.style.filter) || "") ? parseFloat(RegExp.$1) / 100 + "" : a ? "1" : ""
                    },
                    set: function(d, a) {
                        var b = d.style, e = d.currentStyle, f = h.isNaN(a) ? "": "alpha(opacity=" + a * 100 + ")", c = e && e.filter || b.filter || "";
                        b.zoom = 1;
                        if (a >= 1 && h.trim(c.replace(ka, "")) === "" && (b.removeAttribute("filter"), e&&!e.filter))
                            return;
                            b.filter = ka.test(c) ? c.replace(ka, f) : c + " " + f
                        }
                    };
            h(function() {
                if (!h.support.reliableMarginRight)
                    h.cssHooks.marginRight = {
                        get: function(d, a) {
                            var b;
                            h.swap(d, {
                                display: "inline-block"
                            }, function() {
                                b = a ? W(d, "margin-right", "marginRight") :
                                d.style.marginRight
                            });
                            return b
                        }
                    }
            });
            u.defaultView && u.defaultView.getComputedStyle && (za = function(d, a) {
                var b, e, a = a.replace(cb, "-$1").toLowerCase();
                if (e = d.ownerDocument.defaultView) {
                    if (e = e.getComputedStyle(d, null))
                        b = e.getPropertyValue(a), b === ""&&!h.contains(d.ownerDocument.documentElement, d) && (b = h.style(d, a));
                    return b
                }
            });
            u.documentElement.currentStyle && (Aa = function(d, a) {
                var b, e = d.currentStyle && d.currentStyle[a], f = d.runtimeStyle && d.runtimeStyle[a], h = d.style;
                if (!ya.test(e) && db.test(e)) {
                    b = h.left;
                    if (f)
                        d.runtimeStyle.left =
                        d.currentStyle.left;
                    h.left = a === "fontSize" ? "1em" : e || 0;
                    e = h.pixelLeft + "px";
                    h.left = b;
                    if (f)
                        d.runtimeStyle.left = f
                }
                return e === "" ? "auto" : e
            });
            W = za || Aa;
            if (h.expr && h.expr.filters)
                h.expr.filters.hidden = function(d) {
                    var a = d.offsetHeight;
                    return d.offsetWidth === 0 && a === 0 ||!h.support.reliableHiddenOffsets && (d.style.display || h.css(d, "display")) === "none"
                }, h.expr.filters.visible = function(d) {
                    return !h.expr.filters.hidden(d)
                };
            var gb = /%20/g, Oa = /\[\]$/, Ba = /\r?\n/g, hb = /#.*$/, ib = /^(.*?):[ \t]*([^\r\n]*)\r?$/mg, jb = /^(?:color|date|datetime|datetime-local|email|hidden|month|number|password|range|search|tel|text|time|url|week)$/i,
            kb = /^(?:GET|HEAD)$/, lb = /^\/\//, Ca = /\?/, mb = /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, nb = /^(?:select|textarea)/i, ma = /\s+/, ob = /([?&])_=[^&]*/, Da = /^([\w\+\.\-]+:)(?:\/\/([^\/?#:]*)(?::(\d+))?)?/, Ea = h.fn.load, fa = {}, Fa = {}, T, U, Ga = ["*/"] + ["*"];
            try {
                T = G.href
            } catch (ub) {
                T = u.createElement("a"), T.href = "", T = T.href
            }
            U = Da.exec(T.toLowerCase()) || [];
            h.fn.extend({
                load: function(d, a, b) {
                    if (typeof d !== "string" && Ea)
                        return Ea.apply(this, arguments);
                    else if (!this.length)
                        return this;
                    var e = d.indexOf(" ");
                    if (e >= 0)
                        var f =
                        d.slice(e, d.length), d = d.slice(0, e);
                    e = "GET";
                    a && (h.isFunction(a) ? (b = a, a = void 0) : typeof a === "object" && (a = h.param(a, h.ajaxSettings.traditional), e = "POST"));
                    var c = this;
                    h.ajax({
                        url: d,
                        type: e,
                        dataType: "html",
                        data: a,
                        complete: function(d, a, e) {
                            e = d.responseText;
                            d.isResolved() && (d.done(function(d) {
                                e = d
                            }), c.html(f ? h("<div>").append(e.replace(mb, "")).find(f) : e));
                            b && c.each(b, [e, a, d])
                        }
                    });
                    return this
                },
                serialize: function() {
                    return h.param(this.serializeArray())
                },
                serializeArray: function() {
                    return this.map(function() {
                        return this.elements ?
                        h.makeArray(this.elements) : this
                    }).filter(function() {
                        return this.name&&!this.disabled && (this.checked || nb.test(this.nodeName) || jb.test(this.type))
                    }).map(function(d, a) {
                        var b = h(this).val();
                        return b == null ? null : h.isArray(b) ? h.map(b, function(d) {
                            return {
                                name: a.name,
                                value: d.replace(Ba, "\r\n")
                            }
                        }) : {
                            name: a.name,
                            value: b.replace(Ba, "\r\n")
                        }
                    }).get()
                }
            });
            h.each("ajaxStart ajaxStop ajaxComplete ajaxError ajaxSuccess ajaxSend".split(" "), function(d, a) {
                h.fn[a] = function(d) {
                    return this.bind(a, d)
                }
            });
            h.each(["get", "post"],
            function(d, a) {
                h[a] = function(d, b, e, f) {
                    h.isFunction(b) && (f = f || e, e = b, b = void 0);
                    return h.ajax({
                        type: a,
                        url: d,
                        data: b,
                        success: e,
                        dataType: f
                    })
                }
            });
            h.extend({
                getScript: function(d, a) {
                    return h.get(d, void 0, a, "script")
                },
                getJSON: function(d, a, b) {
                    return h.get(d, a, b, "json")
                },
                ajaxSetup: function(d, a) {
                    a ? t(d, h.ajaxSettings) : (a = d, d = h.ajaxSettings);
                    t(d, a);
                    return d
                },
                ajaxSettings: {
                    url: T,
                    isLocal: /^(?:about|app|app\-storage|.+\-extension|file|res|widget):$/.test(U[1]),
                    global: !0,
                    type: "GET",
                    contentType: "application/x-www-form-urlencoded",
                    processData: !0,
                    async: !0,
                    accepts: {
                        xml: "application/xml, text/xml",
                        html: "text/html",
                        text: "text/plain",
                        json: "application/json, text/javascript",
                        "*": Ga
                    },
                    contents: {
                        xml: /xml/,
                        html: /html/,
                        json: /json/
                    },
                    responseFields: {
                        xml: "responseXML",
                        text: "responseText"
                    },
                    converters: {
                        "* text": window.String,
                        "text html": !0,
                        "text json": h.parseJSON,
                        "text xml": h.parseXML
                    },
                    flatOptions: {
                        context: !0,
                        url: !0
                    }
                },
                ajaxPrefilter: w(fa),
                ajaxTransport: w(Fa),
                ajax: function(d, a) {
                    function b(d, a, j, n) {
                        if (x !== 2) {
                            x = 2;
                            k && clearTimeout(k);
                            m = void 0;
                            v =
                            n || "";
                            t.readyState = d > 0 ? 4 : 0;
                            var i, r, w, n = a;
                            if (j) {
                                var z = e, A = t, C = z.contents, F = z.dataTypes, y = z.responseFields, p, u, D, B;
                                for (u in y)
                                    u in j && (A[y[u]] = j[u]);
                                for (; F[0] === "*";)
                                    F.shift(), p === void 0 && (p = z.mimeType || A.getResponseHeader("content-type"));
                                if (p)
                                    for (u in C)
                                        if (C[u] && C[u].test(p)) {
                                            F.unshift(u);
                                            break
                                        }
                                if (F[0]in j)
                                    D = F[0];
                                else {
                                    for (u in j) {
                                        if (!F[0] || z.converters[u + " " + F[0]]) {
                                            D = u;
                                            break
                                        }
                                        B || (B = u)
                                    }
                                    D = D || B
                                }
                                D ? (D !== F[0] && F.unshift(D), j = j[D]) : j = void 0
                            } else 
                                j = void 0;
                            if (d >= 200 && d < 300 || d === 304) {
                                if (e.ifModified) {
                                    if (p = t.getResponseHeader("Last-Modified"))
                                        h.lastModified[s] =
                                        p;
                                    if (p = t.getResponseHeader("Etag"))
                                        h.etag[s] = p
                                }
                                if (d === 304)
                                    n = "notmodified", i=!0;
                                else 
                                    try {
                                        p = e;
                                        p.dataFilter && (j = p.dataFilter(j, p.dataType));
                                        var J = p.dataTypes;
                                        u = {};
                                        var E, G, R = J.length, H, I = J[0], K, O, M, L, V;
                                        for (E = 1; E < R; E++) {
                                            if (E === 1)
                                                for (G in p.converters)
                                                    typeof G === "string" && (u[G.toLowerCase()] = p.converters[G]);
                                                    K = I;
                                                    I = J[E];
                                                    if (I === "*")
                                                        I = K;
                                                    else if (K !== "*" && K !== I) {
                                                        O = K + " " + I;
                                                        M = u[O] || u["* " + I];
                                                        if (!M)
                                                            for (L in V = void 0, u)
                                                                if (H = L.split(" "), H[0] === K || H[0] === "*")
                                                                    if (V = u[H[1] + " " + I]) {
                                                                        L = u[L];
                                                                        L===!0 ? M = V : V===!0 && (M = L);
                                                                        break
                                                                    }
                                                                    !M &&
                                                                    !V && h.error("No conversion from " + O.replace(" ", " to "));
                                                                    M!==!0 && (j = M ? M(j) : V(L(j)))
                                                                }
                                                }
                                                r = j;
                                                n = "success";
                                                i=!0
                                    } catch (N) {
                                    n = "parsererror", w = N
                                }
                            } else if (w = n, !n || d)
                                n = "error", d < 0 && (d = 0);
                            t.status = d;
                            t.statusText = "" + (a || n);
                            i ? g.resolveWith(f, [r, n, t]) : g.rejectWith(f, [t, n, w]);
                            t.statusCode(q);
                            q = void 0;
                            o && c.trigger("ajax" + (i ? "Success" : "Error"), [t, e, i ? r: w]);
                            l.resolveWith(f, [t, n]);
                            o && (c.trigger("ajaxComplete", [t, e]), --h.active || h.event.trigger("ajaxStop"))
                        }
                    }
                    typeof d === "object" && (a = d, d = void 0);
                    var a = a || {}, e = h.ajaxSetup({},
                    a), f = e.context || e, c = f !== e && (f.nodeType || f instanceof h) ? h(f): h.event, g = h.Deferred(), l = h._Deferred(), q = e.statusCode || {}, s, j = {}, n = {}, v, i, m, k, w, x = 0, o, z, t = {
                        readyState: 0,
                        setRequestHeader: function(d, a) {
                            if (!x) {
                                var b = d.toLowerCase(), d = n[b] = n[b] || d;
                                j[d] = a
                            }
                            return this
                        },
                        getAllResponseHeaders: function() {
                            return x === 2 ? v : null
                        },
                        getResponseHeader: function(d) {
                            var a;
                            if (x === 2) {
                                if (!i)
                                    for (i = {}; a = ib.exec(v);)
                                        i[a[1].toLowerCase()] = a[2];
                                a = i[d.toLowerCase()]
                            }
                            return a === void 0 ? null : a
                        },
                        overrideMimeType: function(d) {
                            if (!x)
                                e.mimeType =
                                d;
                            return this
                        },
                        abort: function(d) {
                            d = d || "abort";
                            m && m.abort(d);
                            b(0, d);
                            return this
                        }
                    };
                    g.promise(t);
                    t.success = t.done;
                    t.error = t.fail;
                    t.complete = l.done;
                    t.statusCode = function(d) {
                        if (d) {
                            var a;
                            if (x < 2)
                                for (a in d)
                                    q[a] = [q[a], d[a]];
                            else 
                                a = d[t.status], t.then(a, a)
                        }
                        return this
                    };
                    e.url = ((d || e.url) + "").replace(hb, "").replace(lb, U[1] + "//");
                    e.dataTypes = h.trim(e.dataType || "*").toLowerCase().split(ma);
                    if (e.crossDomain == null)
                        w = Da.exec(e.url.toLowerCase()), e.crossDomain=!(!w ||!(w[1] != U[1] || w[2] != U[2] || (w[3] || (w[1] === "http:" ?
                        80 : 443)) != (U[3] || (U[1] === "http:" ? 80 : 443))));
                    if (e.data && e.processData && typeof e.data !== "string")
                        e.data = h.param(e.data, e.traditional);
                    y(fa, e, a, t);
                    if (x === 2)
                        return !1;
                    o = e.global;
                    e.type = e.type.toUpperCase();
                    e.hasContent=!kb.test(e.type);
                    o && h.active++===0 && h.event.trigger("ajaxStart");
                    if (!e.hasContent && (e.data && (e.url += (Ca.test(e.url) ? "&" : "?") + e.data, delete e.data), s = e.url, e.cache===!1)) {
                        w = h.now();
                        var A = e.url.replace(ob, "$1_=" + w);
                        e.url = A + (A === e.url ? (Ca.test(e.url) ? "&" : "?") + "_=" + w : "")
                    }(e.data && e.hasContent &&
                    e.contentType!==!1 || a.contentType) && t.setRequestHeader("Content-Type", e.contentType);
                    e.ifModified && (s = s || e.url, h.lastModified[s] && t.setRequestHeader("If-Modified-Since", h.lastModified[s]), h.etag[s] && t.setRequestHeader("If-None-Match", h.etag[s]));
                    t.setRequestHeader("Accept", e.dataTypes[0] && e.accepts[e.dataTypes[0]] ? e.accepts[e.dataTypes[0]] + (e.dataTypes[0] !== "*" ? ", " + Ga + "; q=0.01" : "") : e.accepts["*"]);
                    for (z in e.headers)
                        t.setRequestHeader(z, e.headers[z]);
                    if (e.beforeSend && (e.beforeSend.call(f, t, e) ===
                    !1 || x === 2))
                        return t.abort(), !1;
                    for (z in{
                        success: 1,
                        error: 1,
                        complete: 1
                    })
                        t[z](e[z]);
                    if (m = y(Fa, e, a, t)) {
                        t.readyState = 1;
                        o && c.trigger("ajaxSend", [t, e]);
                        e.async && e.timeout > 0 && (k = setTimeout(function() {
                            t.abort("timeout")
                        }, e.timeout));
                        try {
                            x = 1, m.send(j, b)
                        } catch (C) {
                            x < 2 ? b( - 1, C) : h.error(C)
                        }
                    } else 
                        b( - 1, "No Transport");
                    return t
                },
                param: function(d, a) {
                    var b = [], e = function(d, a) {
                        a = h.isFunction(a) ? a() : a;
                        b[b.length] = encodeURIComponent(d) + "=" + encodeURIComponent(a)
                    };
                    if (a === void 0)
                        a = h.ajaxSettings.traditional;
                    if (h.isArray(d) ||
                    d.jquery&&!h.isPlainObject(d))
                        h.each(d, function() {
                            e(this.name, this.value)
                        });
                    else 
                        for (var f in d)
                            v(f, d[f], a, e);
                    return b.join("&").replace(gb, "+")
                }
            });
            h.extend({
                active: 0,
                lastModified: {},
                etag: {}
            });
            var pb = h.now(), ca = /(\=)\?(&|$)|\?\?/i;
            h.ajaxSetup({
                jsonp: "callback",
                jsonpCallback: function() {
                    return h.expando + "_" + pb++
                }
            });
            h.ajaxPrefilter("json jsonp", function(d, a, b) {
                a = d.contentType === "application/x-www-form-urlencoded" && typeof d.data === "string";
                if (d.dataTypes[0] === "jsonp" || d.jsonp!==!1 && (ca.test(d.url) || a &&
                ca.test(d.data))) {
                    var e, f = d.jsonpCallback = h.isFunction(d.jsonpCallback) ? d.jsonpCallback(): d.jsonpCallback, c = window[f], g = d.url, l = d.data, q = "$1" + f + "$2";
                    d.jsonp!==!1 && (g = g.replace(ca, q), d.url === g && (a && (l = l.replace(ca, q)), d.data === l && (g += (/\?/.test(g) ? "&" : "?") + d.jsonp + "=" + f)));
                    d.url = g;
                    d.data = l;
                    window[f] = function(d) {
                        e = [d]
                    };
                    b.always(function() {
                        window[f] = c;
                        if (e && h.isFunction(c))
                            window[f](e[0])
                    });
                    d.converters["script json"] = function() {
                        e || h.error(f + " was not called");
                        return e[0]
                    };
                    d.dataTypes[0] = "json";
                    return "script"
                }
            });
            h.ajaxSetup({
                accepts: {
                    script: "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript"
                },
                contents: {
                    script: /javascript|ecmascript/
                },
                converters: {
                    "text script": function(d) {
                        h.globalEval(d);
                        return d
                    }
                }
            });
            h.ajaxPrefilter("script", function(d) {
                if (d.cache === void 0)
                    d.cache=!1;
                if (d.crossDomain)
                    d.type = "GET", d.global=!1
            });
            h.ajaxTransport("script", function(d) {
                if (d.crossDomain) {
                    var a, b = u.head || u.getElementsByTagName("head")[0] || u.documentElement;
                    return {
                        send: function(e, f) {
                            a = u.createElement("script");
                            a.async = "async";
                            if (d.scriptCharset)
                                a.charset = d.scriptCharset;
                            a.src = d.url;
                            a.onload = a.onreadystatechange = function(d, e) {
                                if (e ||!a.readyState || /loaded|complete/.test(a.readyState))
                                    a.onload = a.onreadystatechange = null, b && a.parentNode && b.removeChild(a), a = void 0, e || f(200, "success")
                            };
                            b.insertBefore(a, b.firstChild)
                        },
                        abort: function() {
                            if (a)
                                a.onload(0, 1)
                        }
                    }
                }
            });
            var la = window.ActiveXObject ? function() {
                for (var d in Y)
                    Y[d](0, 1)
            }
            : !1, qb = 0, Y;
            h.ajaxSettings.xhr = window.ActiveXObject ? function() {
                var d;
                if (!(d=!this.isLocal &&
                z()))
                    a: {
                    try {
                        d = new window.ActiveXObject("Microsoft.XMLHTTP");
                        break a
                    } catch (a) {}
                    d = void 0
                }
                return d
            } : z;
            (function(d) {
                h.extend(h.support, {
                    ajax: !!d,
                    cors: !!d && "withCredentials"in d
                })
            })(h.ajaxSettings.xhr());
            h.support.ajax && h.ajaxTransport(function(d) {
                if (!d.crossDomain || h.support.cors) {
                    var a;
                    return {
                        send: function(b, e) {
                            var f = d.xhr(), c, g;
                            d.username ? f.open(d.type, d.url, d.async, d.username, d.password) : f.open(d.type, d.url, d.async);
                            if (d.xhrFields)
                                for (g in d.xhrFields)
                                    f[g] = d.xhrFields[g];
                            d.mimeType && f.overrideMimeType &&
                            f.overrideMimeType(d.mimeType);
                            !d.crossDomain&&!b["X-Requested-With"] && (b["X-Requested-With"] = "XMLHttpRequest");
                            try {
                                for (g in b)
                                    f.setRequestHeader(g, b[g])
                            } catch (l) {}
                            f.send(d.hasContent && d.data || null);
                            a = function(b, g) {
                                var l, q, s, j, n;
                                try {
                                    if (a && (g || f.readyState === 4)) {
                                        a = void 0;
                                        if (c)
                                            f.onreadystatechange = h.noop, la && delete Y[c];
                                        if (g)
                                            f.readyState !== 4 && f.abort();
                                        else {
                                            l = f.status;
                                            s = f.getAllResponseHeaders();
                                            j = {};
                                            if ((n = f.responseXML) && n.documentElement)
                                                j.xml = n;
                                            j.text = f.responseText;
                                            try {
                                                q = f.statusText
                                            } catch (v) {
                                                q =
                                                ""
                                            }
                                            !l && d.isLocal&&!d.crossDomain ? l = j.text ? 200 : 404 : l === 1223 && (l = 204)
                                        }
                                    }
                                } catch (i) {
                                    g || e( - 1, i)
                                }
                                j && e(l, q, j, s)
                            };
                            !d.async || f.readyState === 4 ? a() : (c=++qb, la && (Y || (Y = {}, h(window).unload(la)), Y[c] = a), f.onreadystatechange = a)
                        },
                        abort: function() {
                            a && a(0, 1)
                        }
                    }
                }
            });
            var ga = {}, Q, X, rb = /^(?:toggle|show|hide)$/, sb = /^([+\-]=)?([\d+.\-]+)([a-z%]*)$/i, da, na = [["height", "marginTop", "marginBottom", "paddingTop", "paddingBottom"], ["width", "marginLeft", "marginRight", "paddingLeft", "paddingRight"], ["opacity"]], aa;
            h.fn.extend({
                show: function(d,
                a, b) {
                    if (d || d === 0)
                        return this.animate(F("show", 3), d, a, b);
                    else {
                        for (var b = 0, e = this.length; b < e; b++)
                            if (d = this[b], d.style) {
                                a = d.style.display;
                                if (!h._data(d, "olddisplay") && a === "none")
                                    a = d.style.display = "";
                                    a === "" && h.css(d, "display") === "none" && h._data(d, "olddisplay", p(d.nodeName))
                            }
                        for (b = 0; b < e; b++)
                            if (d = this[b], d.style && (a = d.style.display, a === "" || a === "none"))
                                d.style.display = h._data(d, "olddisplay") || "";
                        return this
                    }
                },
                hide: function(d, a, b) {
                    if (d || d === 0)
                        return this.animate(F("hide", 3), d, a, b);
                    else {
                        d = 0;
                        for (a = this.length; d <
                        a; d++)
                            this[d].style && (b = h.css(this[d], "display"), b !== "none"&&!h._data(this[d], "olddisplay") && h._data(this[d], "olddisplay", b));
                        for (d = 0; d < a; d++)
                            if (this[d].style)
                                this[d].style.display = "none";
                        return this
                    }
                },
                _toggle: h.fn.toggle,
                toggle: function(d, a, b) {
                    var e = typeof d === "boolean";
                    h.isFunction(d) && h.isFunction(a) ? this._toggle.apply(this, arguments) : d == null || e ? this.each(function() {
                        var a = e ? d: h(this).is(":hidden");
                        h(this)[a ? "show": "hide"]()
                    }) : this.animate(F("toggle", 3), d, a, b);
                    return this
                },
                fadeTo: function(d, a, b,
                e) {
                    return this.filter(":hidden").css("opacity", 0).show().end().animate({
                        opacity: a
                    }, d, b, e)
                },
                animate: function(d, a, b, e) {
                    var f = h.speed(a, b, e);
                    if (h.isEmptyObject(d))
                        return this.each(f.complete, [!1]);
                    d = h.extend({}, d);
                    return this[f.queue===!1 ? "each": "queue"](function() {
                        var r;
                        f.queue===!1 && h._mark(this);
                        var a = h.extend({}, f), b = this.nodeType === 1, e = b && h(this).is(":hidden"), c, g, l, q, s;
                        a.animatedProperties = {};
                        for (l in d) {
                            c = h.camelCase(l);
                            l !== c && (d[c] = d[l], delete d[l]);
                            g = d[c];
                            h.isArray(g) ? (a.animatedProperties[c] =
                            g[1], r = d[c] = g[0], g = r) : a.animatedProperties[c] = a.specialEasing && a.specialEasing[c] || a.easing || "swing";
                            if (g === "hide" && e || g === "show"&&!e)
                                return a.complete.call(this);
                            if (b && (c === "height" || c === "width"))
                                if (a.overflow = [this.style.overflow, this.style.overflowX, this.style.overflowY], h.css(this, "display") === "inline" && h.css(this, "float") === "none")
                                    h.support.inlineBlockNeedsLayout ? (g = p(this.nodeName), g === "inline" ? this.style.display = "inline-block" : (this.style.display = "inline", this.style.zoom = 1)) : this.style.display =
                                    "inline-block"
                        }
                        if (a.overflow != null)
                            this.style.overflow = "hidden";
                        for (l in d)
                            if (b = new h.fx(this, a, l), g = d[l], rb.test(g))
                                b[g === "toggle" ? e ? "show": "hide": g]();
                            else 
                                c = sb.exec(g), q = b.cur(), c ? (g = parseFloat(c[2]), s = c[3] || (h.cssNumber[l] ? "" : "px"), s !== "px" && (h.style(this, l, (g || 1) + s), q*=(g || 1) / b.cur(), h.style(this, l, q + s)), c[1] && (g = (c[1] === "-="?-1 : 1) * g + q), b.custom(q, g, s)) : b.custom(q, g, "");
                        return !0
                    })
                },
                stop: function(d, a) {
                    d && this.queue([]);
                    this.each(function() {
                        var d = h.timers, b = d.length;
                        for (a || h._unmark(!0, this); b--;)
                            if (d[b].elem ===
                            this) {
                                if (a)
                                    d[b](!0);
                                    d.splice(b, 1)
                            }
                    });
                    a || this.dequeue();
                    return this
                }
            });
            h.each({
                slideDown: F("show", 1),
                slideUp: F("hide", 1),
                slideToggle: F("toggle", 1),
                fadeIn: {
                    opacity: "show"
                },
                fadeOut: {
                    opacity: "hide"
                },
                fadeToggle: {
                    opacity: "toggle"
                }
            }, function(d, a) {
                h.fn[d] = function(d, b, e) {
                    return this.animate(a, d, b, e)
                }
            });
            h.extend({
                speed: function(d, a, b) {
                    var e = d && typeof d === "object" ? h.extend({}, d): {
                        complete: b ||!b && a || h.isFunction(d) && d,
                        duration: d,
                        easing: b && a || a&&!h.isFunction(a) && a
                    };
                    e.duration = h.fx.off ? 0 : typeof e.duration ===
                    "number" ? e.duration : e.duration in h.fx.speeds ? h.fx.speeds[e.duration] : h.fx.speeds._default;
                    e.old = e.complete;
                    e.complete = function(d) {
                        h.isFunction(e.old) && e.old.call(this);
                        e.queue!==!1 ? h.dequeue(this) : d!==!1 && h._unmark(this)
                    };
                    return e
                },
                easing: {
                    linear: function(d, a, b, e) {
                        return b + e * d
                    },
                    swing: function(d, a, b, e) {
                        return ( - Math.cos(d * Math.PI) / 2 + 0.5) * e + b
                    }
                },
                timers: [],
                fx: function(d, a, b) {
                    this.options = a;
                    this.elem = d;
                    this.prop = b;
                    a.orig = a.orig || {}
                }
            });
            h.fx.prototype = {
                update: function() {
                    this.options.step && this.options.step.call(this.elem,
                    this.now, this);
                    (h.fx.step[this.prop] || h.fx.step._default)(this)
                },
                cur: function() {
                    if (this.elem[this.prop] != null && (!this.elem.style || this.elem.style[this.prop] == null))
                        return this.elem[this.prop];
                    var d, a = h.css(this.elem, this.prop);
                    return isNaN(d = parseFloat(a))?!a || a === "auto" ? 0 : a : d
                },
                custom: function(d, a, b) {
                    function e(d) {
                        return f.step(d)
                    }
                    var f = this, c = h.fx;
                    this.startTime = aa || A();
                    this.start = d;
                    this.end = a;
                    this.unit = b || this.unit || (h.cssNumber[this.prop] ? "" : "px");
                    this.now = this.start;
                    this.pos = this.state = 0;
                    e.elem =
                    this.elem;
                    e() && h.timers.push(e)&&!da && (da = setInterval(c.tick, c.interval))
                },
                show: function() {
                    this.options.orig[this.prop] = h.style(this.elem, this.prop);
                    this.options.show=!0;
                    this.custom(this.prop === "width" || this.prop === "height" ? 1 : 0, this.cur());
                    h(this.elem).show()
                },
                hide: function() {
                    this.options.orig[this.prop] = h.style(this.elem, this.prop);
                    this.options.hide=!0;
                    this.custom(this.cur(), 0)
                },
                step: function(d) {
                    var a = aa || A(), b=!0, e = this.elem, f = this.options, c;
                    if (d || a >= f.duration + this.startTime) {
                        this.now = this.end;
                        this.pos = this.state = 1;
                        this.update();
                        f.animatedProperties[this.prop]=!0;
                        for (c in f.animatedProperties)
                            f.animatedProperties[c]!==!0 && (b=!1);
                        if (b) {
                            f.overflow != null&&!h.support.shrinkWrapBlocks && h.each(["", "X", "Y"], function(d, a) {
                                e.style["overflow" + a] = f.overflow[d]
                            });
                            f.hide && h(e).hide();
                            if (f.hide || f.show)
                                for (var g in f.animatedProperties)
                                    h.style(e, g, f.orig[g]);
                            f.complete.call(e)
                        }
                        return !1
                    } else 
                        f.duration == Infinity ? this.now = a : (d = a - this.startTime, this.state = d / f.duration, this.pos = h.easing[f.animatedProperties[this.prop]](this.state,
                        d, 0, 1, f.duration), this.now = this.start + (this.end - this.start) * this.pos), this.update();
                    return !0
                }
            };
            h.extend(h.fx, {
                tick: function() {
                    for (var d = h.timers, a = 0; a < d.length; ++a)
                        d[a]() || d.splice(a--, 1);
                    d.length || h.fx.stop()
                },
                interval: 13,
                stop: function() {
                    clearInterval(da);
                    da = null
                },
                speeds: {
                    slow: 600,
                    fast: 200,
                    _default: 400
                },
                step: {
                    opacity: function(d) {
                        h.style(d.elem, "opacity", d.now)
                    },
                    _default: function(d) {
                        d.elem.style && d.elem.style[d.prop] != null ? d.elem.style[d.prop] = (d.prop === "width" || d.prop === "height" ? Math.max(0, d.now) :
                        d.now) + d.unit : d.elem[d.prop] = d.now
                    }
                }
            });
            if (h.expr && h.expr.filters)
                h.expr.filters.animated = function(d) {
                    return h.grep(h.timers, function(a) {
                        return d === a.elem
                    }).length
                };
            var tb = /^t(?:able|d|h)$/i, Ha = /^(?:body|html)$/i;
            h.fn.offset = "getBoundingClientRect"in u.documentElement ? function(d) {
                var a = this[0], b;
                if (d)
                    return this.each(function(a) {
                        h.offset.setOffset(this, d, a)
                    });
                if (!a ||!a.ownerDocument)
                    return null;
                if (a === a.ownerDocument.body)
                    return h.offset.bodyOffset(a);
                try {
                    b = a.getBoundingClientRect()
                } catch (e) {}
                var f =
                a.ownerDocument, c = f.documentElement;
                if (!b ||!h.contains(c, a))
                    return b ? {
                        top: b.top,
                        left: b.left
                    } : {
                        top: 0,
                        left: 0
                    };
                a = f.body;
                f = J(f);
                return {
                    top: b.top + (f.pageYOffset || h.support.boxModel && c.scrollTop || a.scrollTop) - (c.clientTop || a.clientTop || 0),
                    left: b.left + (f.pageXOffset || h.support.boxModel && c.scrollLeft || a.scrollLeft) - (c.clientLeft || a.clientLeft || 0)
                }
            } : function(d) {
                var a = this[0];
                if (d)
                    return this.each(function(a) {
                        h.offset.setOffset(this, d, a)
                    });
                if (!a ||!a.ownerDocument)
                    return null;
                if (a === a.ownerDocument.body)
                    return h.offset.bodyOffset(a);
                h.offset.initialize();
                var b, e = a.offsetParent, f = a.ownerDocument, c = f.documentElement, g = f.body;
                b = (f = f.defaultView) ? f.getComputedStyle(a, null) : a.currentStyle;
                for (var l = a.offsetTop, q = a.offsetLeft; (a = a.parentNode) && a !== g && a !== c;) {
                    if (h.offset.supportsFixedPosition && b.position === "fixed")
                        break;
                    b = f ? f.getComputedStyle(a, null) : a.currentStyle;
                    l -= a.scrollTop;
                    q -= a.scrollLeft;
                    if (a === e) {
                        l += a.offsetTop;
                        q += a.offsetLeft;
                        if (h.offset.doesNotAddBorder && (!h.offset.doesAddBorderForTableAndCells ||!tb.test(a.nodeName)))
                            l +=
                            parseFloat(b.borderTopWidth) || 0, q += parseFloat(b.borderLeftWidth) || 0;
                        e = a.offsetParent
                    }
                    h.offset.subtractsBorderForOverflowNotVisible && b.overflow !== "visible" && (l += parseFloat(b.borderTopWidth) || 0, q += parseFloat(b.borderLeftWidth) || 0)
                }
                if (b.position === "relative" || b.position === "static")
                    l += g.offsetTop, q += g.offsetLeft;
                h.offset.supportsFixedPosition && b.position === "fixed" && (l += Math.max(c.scrollTop, g.scrollTop), q += Math.max(c.scrollLeft, g.scrollLeft));
                return {
                    top: l,
                    left: q
                }
            };
            h.offset = {
                initialize: function() {
                    var d =
                    u.body, a = u.createElement("div"), b, e, f, c = parseFloat(h.css(d, "marginTop")) || 0;
                    h.extend(a.style, {
                        position: "absolute",
                        top: 0,
                        left: 0,
                        margin: 0,
                        border: 0,
                        width: "1px",
                        height: "1px",
                        visibility: "hidden"
                    });
                    a.innerHTML = "<div style='position:absolute;top:0;left:0;margin:0;border:5px solid #000;padding:0;width:1px;height:1px;'><div></div></div><table style='position:absolute;top:0;left:0;margin:0;border:5px solid #000;padding:0;width:1px;height:1px;' cellpadding='0' cellspacing='0'><tr><td></td></tr></table>";
                    d.insertBefore(a,
                    d.firstChild);
                    b = a.firstChild;
                    e = b.firstChild;
                    f = b.nextSibling.firstChild.firstChild;
                    this.doesNotAddBorder = e.offsetTop !== 5;
                    this.doesAddBorderForTableAndCells = f.offsetTop === 5;
                    e.style.position = "fixed";
                    e.style.top = "20px";
                    this.supportsFixedPosition = e.offsetTop === 20 || e.offsetTop === 15;
                    e.style.position = e.style.top = "";
                    b.style.overflow = "hidden";
                    b.style.position = "relative";
                    this.subtractsBorderForOverflowNotVisible = e.offsetTop===-5;
                    this.doesNotIncludeMarginInBodyOffset = d.offsetTop !== c;
                    d.removeChild(a);
                    h.offset.initialize =
                    h.noop
                },
                bodyOffset: function(d) {
                    var a = d.offsetTop, b = d.offsetLeft;
                    h.offset.initialize();
                    h.offset.doesNotIncludeMarginInBodyOffset && (a += parseFloat(h.css(d, "marginTop")) || 0, b += parseFloat(h.css(d, "marginLeft")) || 0);
                    return {
                        top: a,
                        left: b
                    }
                },
                setOffset: function(d, a, b) {
                    var e = h.css(d, "position");
                    if (e === "static")
                        d.style.position = "relative";
                    var f = h(d), c = f.offset(), g = h.css(d, "top"), l = h.css(d, "left"), q = {}, s = {};
                    (e === "absolute" || e === "fixed") && h.inArray("auto", [g, l])>-1 ? (s = f.position(), e = s.top, l = s.left) : (e = parseFloat(g) ||
                    0, l = parseFloat(l) || 0);
                    h.isFunction(a) && (a = a.call(d, b, c));
                    if (a.top != null)
                        q.top = a.top - c.top + e;
                    if (a.left != null)
                        q.left = a.left - c.left + l;
                    "using"in a ? a.using.call(d, q) : f.css(q)
                }
            };
            h.fn.extend({
                position: function() {
                    if (!this[0])
                        return null;
                    var d = this[0], a = this.offsetParent(), b = this.offset(), e = Ha.test(a[0].nodeName) ? {
                        top: 0,
                        left: 0
                    }
                    : a.offset();
                    b.top -= parseFloat(h.css(d, "marginTop")) || 0;
                    b.left -= parseFloat(h.css(d, "marginLeft")) || 0;
                    e.top += parseFloat(h.css(a[0], "borderTopWidth")) || 0;
                    e.left += parseFloat(h.css(a[0],
                    "borderLeftWidth")) || 0;
                    return {
                        top: b.top - e.top,
                        left: b.left - e.left
                    }
                },
                offsetParent: function() {
                    return this.map(function() {
                        for (var d = this.offsetParent || u.body; d&&!Ha.test(d.nodeName) && h.css(d, "position") === "static";)
                            d = d.offsetParent;
                        return d
                    })
                }
            });
            h.each(["Left", "Top"], function(d, a) {
                var b = "scroll" + a;
                h.fn[b] = function(a) {
                    var e, f;
                    if (a === void 0) {
                        e = this[0];
                        return !e ? null : (f = J(e)) ? "pageXOffset"in f ? f[d ? "pageYOffset": "pageXOffset"] : h.support.boxModel && f.document.documentElement[b] || f.document.body[b] : e[b]
                    }
                    return this.each(function() {
                        (f =
                        J(this)) ? f.scrollTo(!d ? a : h(f).scrollLeft(), d ? a : h(f).scrollTop()) : this[b] = a
                    })
                }
            });
            h.each(["Height", "Width"], function(a, b) {
                var e = b.toLowerCase();
                h.fn["inner" + b] = function() {
                    var a = this[0];
                    return a && a.style ? parseFloat(h.css(a, e, "padding")) : null
                };
                h.fn["outer" + b] = function(a) {
                    var d = this[0];
                    return d && d.style ? parseFloat(h.css(d, e, a ? "margin" : "border")) : null
                };
                h.fn[e] = function(a) {
                    var d = this[0];
                    if (!d)
                        return a == null ? null : this;
                    if (h.isFunction(a))
                        return this.each(function(d) {
                            var b = h(this);
                            b[e](a.call(this, d, b[e]()))
                        });
                    if (h.isWindow(d)) {
                        var f = d.document.documentElement["client" + b], d = d.document.body;
                        return h.support.boxModel && f || d && d["client" + b] || f
                    } else 
                        return d.nodeType === 9 ? Math.max(d.documentElement["client" + b], d.body["scroll" + b], d.documentElement["scroll" + b], d.body["offset" + b], d.documentElement["offset" + b]) : a === void 0 ? (f = h.css(d, e), d = parseFloat(f), h.isNaN(d) ? f : d) : this.css(e, typeof a === "string" ? a : a + "px")
                }
            });
            h.fn.offsetNoIPadFix = h.fn.offset;
            h.fn.offsetIPadFix = h.fn.offset;
            if (/webkit.*mobile/i.test(E.userAgent) &&
            parseFloat(h.browser.version) < 532.9 && "getBoundingClientRect"in u.documentElement)
                h.fn.offsetIPadFix = function() {
                    var d = this.offsetNoIPadFix();
                    d && (d.top -= window.scrollY, d.left -= window.scrollX);
                    return d
                }, h.fn.offset = h.fn.offsetIPadFix;
            return h
        })
    })
})(function() {
    var p = window.AmazonUIPageJS || P, c = p.attributeErrors;
    return c ? c("AmazonUIjQuery") : p
}());
(function(p) {
    p.execute(function() {
        p.when("jQuery").register("a-base", function(c) {
            return {
                version: function() {
                    return "3.0"
                },
                $: c
            }
        });
        p.when("a-base", "p-weblab").register("a-util", function(c, k) {
            var o = c.$, i = function() {
                function a(a, e, f) {
                    if (a !== null)
                        if (Array.prototype.forEach && a.forEach === Array.prototype.forEach)
                            a.forEach(e, f);
                        else if (a.length ===+ a.length)
                            for (var c = 0, g = a.length; c < g; c++) {
                                if (c in a && e.call(f, a[c], c, a) === b)
                                    break
                            } else 
                                for (c in a)
                                    if (a.hasOwnProperty(c) && e.call(f, a[c], c, a) === {})
                                        break
                }
                var b = {};
                return {
                    each: a,
                    map: function(b, e, f) {
                        var c = [];
                        if (b === null)
                            return c;
                        if (Array.prototype.map && b.map === Array.prototype.map)
                            return b.map(e, f);
                        a(b, function(a, b, g) {
                            c[c.length] = e.call(f, a, b, g)
                        });
                        if (b.length ===+ b.length)
                            c.length = b.length;
                        return c
                    },
                    reduce: function(b, e, f, c) {
                        var g = arguments.length > 2;
                        b === null && (b = []);
                        if (Array.prototype.reduce && b.reduce === Array.prototype.reduce)
                            return g ? b.reduce(e, f) : b.reduce(e);
                        a(b, function(a, b, l) {
                            g ? f = e.call(c, f, a, b, l) : (f = a, g=!0)
                        });
                        g || p.error("Reduce of empty array with no initial value", "A.util",
                        "reduce");
                        return f
                    },
                    breaker: b
                }
            }(), m = function() {
                return {
                    throttle: function(a, e, f) {
                        var c, g, j, n = null, i = 0;
                        f || (f = {});
                        var v = function() {
                            i = f.leading===!1 ? 0 : b.now();
                            n = null;
                            j = a.apply(c, g);
                            c = g = null
                        };
                        return function() {
                            var m = b.now();
                            !i && f.leading===!1 && (i = m);
                            var k = e - (m - i);
                            c = this;
                            g = arguments;
                            k <= 0 ? (clearTimeout(n), n = null, i = m, j = a.apply(c, g), c = g = null) : !n && f.trailing!==!1 && (n = setTimeout(v, k));
                            return j
                        }
                    },
                    debounce: function(a, e, f) {
                        var c, g, j, n, i, v = function() {
                            var m = b.now() - n;
                            m < e ? c = setTimeout(v, e - m) : (c = null, f || (i = a.apply(j,
                            g), j = g = null))
                        };
                        return function() {
                            j = this;
                            g = arguments;
                            n = b.now();
                            var m = f&&!c;
                            c || (c = setTimeout(v, e));
                            m && (i = a.apply(j, g), j = g = null);
                            return i
                        }
                    },
                    delay: function(a, b) {
                        var e = Array.prototype.slice.call(arguments, 2);
                        return setTimeout(function() {
                            return a.apply(null, e)
                        }, b)
                    },
                    interval: function(a, b) {
                        return setInterval(a, b)
                    }
                }
            }(), j = function() {
                var a, b;
                a = /^\s+/;
                b = /\s+$/;
                return {
                    trim: function(e) {
                        return String.prototype.trim ? String.prototype.trim.call(e) : e.replace(a, "").replace(b, "")
                    },
                    contains: function(a, b) {
                        return ("" + a).indexOf(b) !==
                        - 1
                    }
                }
            }(), n = function() {
                function a() {
                    for (var b = arguments[0], e = 1, f = arguments.length; e < f; e++)
                        i.each(arguments[e], function(e, f) {
                            var c;
                            o.isArray(e) ? (c=!o.isArray(b[f])&&!o.isPlainObject(b[f]) ? [] : b[f], b[f] = a(c, e)) : o.isPlainObject(e) ? (c=!o.isPlainObject(b[f]) ? {} : b[f], b[f] = a(c, e)) : b[f] = e
                        });
                    return b
                }
                function b(a, e) {
                    var a = a || {}, e = e || {}, f = {}, c;
                    for (c in a)
                        a.hasOwnProperty(c) && (f[c] = typeof a[c] === "object" && a[c] ? b(a[c], e[c]) : a[c] !== e[c]);
                    for (c in e)
                        e.hasOwnProperty(c)&&!f[c] && (f[c] = typeof e[c] === "object" && e[c] ? b(e[c],
                        a[c]) : e[c] !== a[c]);
                    return f
                }
                function e(a, b) {
                    var f;
                    if (a === b)
                        return !0;
                    if (o.isArray(a)) {
                        if (!o.isArray(b) || a.length !== b.length)
                            return !1;
                        for (f = a.length; f--;)
                            if (!e(a[f], b[f]))
                                return !1;
                        return !0
                    }
                    if (o.isPlainObject(a)) {
                        if (!o.isPlainObject(b))
                            return !1;
                        for (f in a)
                            if (!e(a[f], b[f]))
                                return !1;
                        return !0
                    }
                    return !1
                }
                return {
                    extend: a,
                    diff: b,
                    equals: e,
                    copy: function(b) {
                        return o.isArray(b) ? b.slice(0) : o.isPlainObject(b) ? a({}, b) : b
                    },
                    indexOfArray: function(a, b, e) {
                        if (Array.prototype.indexOf && a.indexOf === Array.prototype.indexOf)
                            return a.indexOf(b,
                            e);
                        a && a instanceof Array || p.error("Invalid arr passed to A.indexOfArray: " + a, "A.util", "indexOfArray");
                        e = parseInt(e, 10);
                        e = isNaN(e) ? 0 : e;
                        if (!isFinite(e))
                            return - 1;
                        for (var f = a.length; e < f; e++)
                            if (a[e] === b)
                                return e;
                        return - 1
                    },
                    isFiniteNumber: function(a) {
                        return typeof a === "number"&&!isNaN(a) && isFinite(a)
                    },
                    objectIsEmpty: function(a) {
                        for (var b in a)
                            if (a.hasOwnProperty(b))
                                return !1;
                        return !0
                    }
                }
            }(), g = function() {
                return {
                    onScreen: function(a) {
                        var b = o(a);
                        if (!b.is(":visible"))
                            return !1;
                        var e = o(window), a = e.scrollTop(), e = window.innerHeight ?
                        window.innerHeight: e.height(), f = a + e;
                        a -= 100;
                        f += 100;
                        var c = b.offset().top, b = b.height(), g = c + b;
                        return c >= a && c < f || g > a && g <= f || b > e && c <= a && g >= f
                    }
                }
            }(), f = function() {
                return {
                    setCssImportant: function(a, b, e) {
                        a = a.jquery ? a[0] : a;
                        if (typeof a !== "undefined")
                            a = a.style, a.cssText = a.cssText.replace(RegExp(b + "\\s*:\\s*[.^;]*(\\s*;)?", "gmi"), ""), a.cssText += b + ": " + e + " !important;"
                    }
                }
            }(), b = function() {
                return {
                    now: function() {
                        return Date.now()
                    }
                }
            }(), e = function() {
                return {
                    hasWeblab: function(a, b) {
                        var e = a.toUpperCase(), f = b.toUpperCase();
                        return k[e] ? k[e] === f : !1
                    }
                }
            }();
            return {
                now: b.now,
                extend: n.extend,
                copy: n.copy,
                diff: n.diff,
                equals: n.equals,
                objectIsEmpty: n.objectIsEmpty,
                indexOfArray: n.indexOfArray,
                isFiniteNumber: n.isFiniteNumber,
                parseJSON: function(a) {
                    return o.parseJSON(a)
                },
                throttle: m.throttle,
                debounce: m.debounce,
                delay: m.delay,
                interval: m.interval,
                trim: j.trim,
                contains: j.contains,
                map: i.map,
                reduce: i.reduce,
                each: i.each,
                breaker: i.breaker,
                onScreen: g.onScreen,
                setCssImportant: f.setCssImportant,
                hasWeblab: e.hasWeblab
            }
        });
        p.when("p-detect", "a-util").register("a-detect",
        function(c, k) {
            var o = String.fromCharCode(92), i;
            try {
                i = navigator.userAgent
            } catch (m) {
                i = ""
            }
            var j = k.copy(c), n = {};
            k.each({
                isAmazonApp: function() {
                    return /Windowshop/.test(i)
                },
                isGen5App: function() {
                    return /Windowshop.*(?:KFOT|KFTH|KFJWA|KFJWI|KFTT)/.test(i)
                },
                androidVersion: function() {
                    var f = /(?:Android\s+|Windowshop.*Android\/)(\d+\.\d+(?:\.\d+)*)/.exec(i);
                    if (f[1])
                        return f[1]
                },
                isChrome: function() {
                    return /Chrome/.test(i)
                },
                chromeVersion: function() {
                    var f = /Chrome\/(\d+\.\d+(?:\.\d+)*)/.exec(i);
                    if (f[1])
                        return f[1]
                },
                isAndroidStockGuess: function() {
                    var f=!1;
                    j.capabilities.android&&!/Chrome|Opera|Firefox|UCBrowser/.test(i) && (f = /AppleWebKit\/(\d+\.\d+)/.exec(i), f = f[1] && f[1] < "535");
                    return f
                },
                isFirefox: function() {
                    return /Firefox/.test(i)
                },
                isOldAndroid: function() {
                    return /Android\s[12]/.test(i)
                },
                isIE10: function() {
                    return /Trident/.test(i) && "onmspointerup"in document&&!("onpointerup"in document)
                },
                isIE10Plus: function() {
                    return /Trident/.test(i) && ("onpointerup"in document || "onmspointerup"in document)
                },
                isIE11Plus: function() {
                    return /Trident/.test(i) &&
                    "onpointerup"in document
                },
                isiOS8: function() {
                    var f = RegExp("Version/8" + o + ".");
                    return j.capabilities.ios && f.test(i)
                },
                isIETouchCapable: function() {
                    return j.capabilities.isIE10Plus && /Touch;/.test(i)
                },
                isMetroIEGuess: function() {
                    var f=!0;
                    try {
                        f = new ActiveXObject("htmlfile")
                    } catch (b) {
                        f=!1
                    }
                    return j.capabilities.isIE10Plus&&!j.capabilities.mobile&&!f
                },
                pointerPrefix: function() {
                    return "onmspointerup"in document || "onpointerup"in document ? "onpointerup"in document ? "pointer" : "MSPointer" : !1
                },
                actionMode: function() {
                    var f =
                    j.capabilities.pointerPrefix;
                    return f ? f : j.capabilities.touch ? "touch" : "mouse"
                }
            }, function(f, b) {
                var e = j.capabilities, a;
                try {
                    a = f()
                } catch (c) {
                    a=!1
                }
                e[b] = a
            });
            k.each({
                start: {
                    mouse: "down",
                    touch: "start",
                    pointer: "down",
                    MSPointer: "Down"
                },
                end: {
                    mouse: "up",
                    touch: "end",
                    pointer: "up",
                    MSPointer: "Up"
                },
                move: {
                    mouse: "move",
                    touch: "move",
                    pointer: "move",
                    MSPointer: "Move"
                },
                enter: {
                    mouse: "enter",
                    touch: "enter",
                    pointer: "enter"
                },
                leave: {
                    mouse: "leave",
                    touch: "leave",
                    pointer: "leave"
                },
                cancel: {
                    touch: "cancel",
                    pointer: "cancel",
                    MSPointer: "Cancel"
                },
                over: {
                    mouse: "over",
                    pointer: "over",
                    MSPointer: "Over"
                },
                out: {
                    mouse: "out",
                    pointer: "out",
                    MSPointer: "Out"
                }
            }, function(f, b) {
                var e = j.capabilities.actionMode, a = typeof f === "string" ? f: f[e];
                n[b] = a ? e + a : f.mouse === void 0 ? "" : "mouse" + f.mouse
            });
            j.action = n;
            var g = {};
            if (j.capabilities.pointerPrefix === "pointer")
                g.touch = "touch", g.pen = "pen", g.mouse = "mouse", g.unknown = "";
            else if (j.capabilities.pointerPrefix === "MSPointer")
                g.touch = 2, g.pen = 3, g.mouse = 4;
            j.pointerType = g;
            return j
        });
        p.when("a-util", "a-events").register("a-prefix", function(c,
        k) {
            function o(c) {
                return c.toLowerCase().replace(/-(.)/g, function(f, b) {
                    return b.toUpperCase()
                })
            }
            var i = {
                transitionend: null
            }, m = document.createElement("div").style, j = {}, n = ["o", "ms", "moz", "webkit"];
            k.on("beforeReady", function() {
                if (window.addEventListener) {
                    var g = document.createElement("div"), f = function(b) {
                        i.transitionend = b.type;
                        this.removeEventListener("webkitTransitionEnd", f, !1);
                        this.removeEventListener("transitionend", f, !1);
                        this.removeEventListener("otransitionend", f, !1);
                        this.removeEventListener("oTransitionEnd",
                        f, !1)
                    };
                    g.setAttribute("style", "position:absolute;top:0px;z-index:-1;transition:top 1ms ease;-webkit-transition:top 1ms ease;-moz-transition:top 1ms ease;-o-transition:top 1ms ease;");
                    g.addEventListener("transitionend", f, !1);
                    g.addEventListener("webkitTransitionEnd", f, !1);
                    g.addEventListener("otransitionend", f, !1);
                    this.addEventListener("oTransitionEnd", f, !1);
                    document.body.appendChild(g);
                    c.delay(function() {
                        g.style.top = "100px";
                        c.delay(function() {
                            g.parentNode.removeChild(g);
                            g = f = null;
                            c.each(i, function() {})
                        },
                        100)
                    }, 0)
                }
            });
            return {
                prefixes: {
                    getStyle: function(c) {
                        if (!j[c]) {
                            var f = o(c);
                            if (f in m)
                                j[c] = f;
                            else 
                                for (var f = f.charAt(0).toUpperCase() + f.slice(1), b = n.length; b--;) {
                                    var e = n[b] + f;
                                    e in m && (j[c] = e)
                                }
                            }
                        return j[c]
                    },
                    getEvent: function(c) {
                        return c ? i[c.toLowerCase()] : void 0
                    }
                }
            }
        });
        p.register("a-defer", function() {
            function c(g) {
                var f = 0, b = setTimeout(function() {
                    c(g)
                }, 0);
                if (i || g.length === 0)
                    clearTimeout(b), o=!1;
                else {
                    var e = Date.now();
                    try {
                        g.shift().call()
                    } catch (a) {
                        p.error("Deferred execution failed: " + a.message, "A.defer", "partialExecute")
                    }
                    n +=
                    Date.now() - e;
                    n > m && (f = j, n = 0);
                    setTimeout(function() {
                        clearTimeout(b);
                        c(g)
                    }, f)
                }
            }
            var k = [], o=!1, i=!1, m = 50, j = 50, n = 0;
            return {
                defer: function(g) {
                    k.push(g);
                    !o&&!i && (o=!0, setTimeout(function() {
                        c(k)
                    }, 0))
                },
                pauseDeferred: function() {
                    i=!0
                },
                executeDeferred: function() {
                    i=!1;
                    o=!0;
                    setTimeout(function() {
                        c(k)
                    }, 0)
                }
            }
        });
        p.when("a-base", "a-util", "p-detect").register("a-events", function(c, k, o) {
            function i(a, b) {
                b!==!1 && f.makeTopicTimeSliced(a);
                n[a]=!0;
                f.subscribe(a, function() {
                    p.register(a, function() {
                        var a = window.aPageStart || k.now();
                        return {
                            time: k.now() - a
                        }
                    })
                });
                f.publish(a);
                f.unsubscribe(a)
            }
            function m() {
                e || (e=!0, o.responsiveGridEnabled() && o.toggleResponsiveGrid(!0), i("beforeReady"), i("ready"), i("afterReady"), document.readyState === "complete" && j())
            }
            function j() {
                if (!n.beforeLoad ||!n.load)
                    i("beforeLoad"), i("load"), setTimeout(function() {
                        i("beforeAfterLoad");
                        i("afterLoad")
                    }, 1500)
            }
            var c = c.$, n = {}, g = c(window), f = function() {
                function a(b) {
                    for (var e = setTimeout(a, 0), l = k.now(), q; (q = c.shift()) !== void 0;)
                        try {
                            if (q.fn.apply(window, q.args)===!1) {
                                clearTimeout(e);
                                for (var n = q.id, i = void 0, m = [], o = 0, u = c.length; o < u; o++)
                                    i = c[o].id, n !== i && m.push(c[o]);
                                    c = m
                            }
                            if (b===!0 && k.now() - l >= g) {
                                clearTimeout(e);
                                k.delay(a, j);
                                return 
                            }
                    } catch (E) {
                        p.error("Event execution failed for event " + q.topic + ": " + (E && E.message || E), "A.events", "partialPublish")
                    }
                    clearTimeout(e);
                    f=!1
                }
                var b = {}, e = 0, f=!1, c = [], g = 50, j = 15;
                return {
                    publish: function(g) {
                        var j = b[g];
                        if (j) {
                            var n = j.isTimeSliced, i = Array.prototype.slice.call(arguments, 1), m = e++, k = function(a) {
                                c.push({
                                    topic: g,
                                    id: m,
                                    fn: a,
                                    args: i
                                })
                            };
                            if (n)
                                for (var o; (o = j.shift()) !==
                                void 0;)
                                    k(o);
                            else {
                                o = 0;
                                for (var w = j.length; o < w; o++)
                                    k(j[o])
                                }
                            f || (f=!0, a(n))
                        }
                    },
                    subscribe: function(a, e) {
                        b[a] || (b[a] = []);
                        if (typeof e === "function")
                            return b[a].unshift(e), {
                                event: a,
                                callback: e
                            }
                    },
                    unsubscribe: function(a, e) {
                        for (var f, c = a.split(" "); (f = c.pop()) !== void 0;)
                            if (b[f])
                                if (e) {
                                    f = b[f];
                                    for (var g = 0, q = f.length; g < q; g++)
                                        f[g] === e && f.splice(g--, 1)
                                } else 
                                    b[f] = []
                    },
                    makeTopicTimeSliced: function(a) {
                        b[a] || (b[a] = []);
                        b[a].isTimeSliced=!0
                    }
                }
            }(), b = function() {
                var a = function(a, b, e) {
                    var c = a.split(" "), g = [], j = b;
                    for (e===!0 && (j =
                    function() {
                        b.apply(window, arguments);
                        f.unsubscribe(a, j)
                    }); (e = c.pop()) !== void 0;)
                        n[e] ? (f.subscribe(e, b), f.publish(e), f.unsubscribe(e)) : g.push(f.subscribe(e, j).event);
                    return {
                        event: g.join(" "),
                        callback: j
                    }
                };
                k.each("ready,load,unload,afterLoad,scroll,resize,orientationchange,zoom".split(","), function(b) {
                    a[b] = function() {
                        a.apply(window, [b].concat([].slice.call(arguments, 0)))
                    }
                });
                return a
            }(), e=!1;
            c(document).ready(m);
            p.when("a-domready").execute(m);
            g.load(j);
            g.unload(function() {
                i("unload")
            });
            return {
                on: b,
                one: function(a,
                e) {
                    var f = a.split(" ");
                    if (f.length > 1)
                        p.error("A.one only accepts a single event name, but was provided with: " + f.length + ", (" + a + ")", "A.events", "one");
                    else 
                        return b(a, e, !0)
                },
                off: function(a, b) {
                    if (typeof a === "object")
                        a = a.event, b = a.callback;
                    return f.unsubscribe(a, b)
                },
                trigger: function() {
                    f.publish.apply(window, arguments)
                },
                events: {
                    defaults: {
                        input: "change",
                        select: "change",
                        a: "click",
                        button: "click",
                        form: "submit"
                    }
                }
            }
        });
        p.when("a-base", "a-util", "a-events").register("a-declarative", function(c, k, o) {
            function i(c) {
                var f =
                j(c.currentTarget), b = j(c.target);
                if (c.type === "submit") {
                    var e = b.closest("form");
                    e.length && (b = e)
                }
                if (e = f.data("action"))
                    e = e.split(" "), k.each(e, function(a) {
                        var e = n[a] || {}, j = f.data(a), s = c.type, s = {
                            $target: b,
                            $currentTarget: f,
                            targetTag: b.prop("tagName").toLowerCase(),
                            type: s,
                            action: a,
                            data: j,
                            $event: c,
                            $declarativeParent: f
                        }, a = "a:declarative:" + a;
                        o.trigger(a, s);
                        o.trigger(a + ":" + c.type, s);
                        a=!1;
                        j ? a=!!j.allowLinkDefault : e && (a=!!e.allowLinkDefault);
                        c.type === "click"&&!a ? (e = b.closest("a"), e = e.length && (e[0].href === "#" ||
                        c.currentTarget === e[0] || e.parent(".a-declarative").length)) : e=!1;
                        e && c.preventDefault()
                    })
            }
            function m() {
                var c, f, b, e;
                switch (arguments.length) {
                case 2:
                    c = arguments[0];
                    e = arguments[1];
                    break;
                case 3:
                    c = arguments[0];
                    f = arguments[1];
                    e = arguments[2];
                    break;
                case 4:
                    c = arguments[0], f = arguments[1], b = arguments[2], e = arguments[3]
                }
                c && (typeof c === "string" && (c = k.trim(c).split(" ")), k.each(c, function(a) {
                    var c = "a:declarative:" + a;
                    n[a] = b || {};
                    if (f)
                        f = typeof f === "string" ? k.trim(f).split(" ") : f, k.each(f, function(a) {
                            o.on(c + ":" + a, e)
                        });
                    else 
                        o.on(c, e)
                }))
            }
            var j = c.$, n = {};
            j(document).delegate(".a-declarative", "click dblclick mousedown mouseup mouseenter mouseleave mousemove change submit touchstart touchend touchmove touchcancel keydown keyup keypress MSPointerDown pointerdown MSPointerUp pointerup MSPointerMove pointermove MSPointerCancel pointercancel MSPointerOver pointerenter MSPointerOut pointerleave", i).delegate(".a-gesture", "tap swipe swipe-horizontal swipe-vertical pan-horizontal pan-vertical doubleTap", i);
            m.create = function(c,
            f, b) {
                var e = c.jquery && c.length ? c: j(c);
                if (e.length && f) {
                    var a = e.data("action");
                    e.data("action", a ? a + " " + f : f).data(f, b ? b : {});
                    e.addClass("a-declarative")
                }
                return c
            };
            m.remove = function(c, f) {
                var b = c.jquery && c.length ? c: j(c), e = b.data("action"), f = f || e;
                if (b.length && e) {
                    for (var e = e.split(" "), a = e.length; a--;)(e[a] === f || e[a] === "") 
                        && e.splice(a, 1);
                    e.length ? b.data("action", e.join("")) : b.data("action", null).removeClass("a-declarative");
                    b.data(f, null)
                }
                return c
            };
            return {
                declarative: m
            }
        });
        p.when("a-base", "a-util", "a-events",
        "a-declarative").register("a-state", function(c, k, o, i) {
            function m(b, e, a, c) {
                if (e === null || g.isArray(e) || g.isPlainObject(e)) {
                    var j = k.copy(f[b]);
                    !j ||!e || c ? f[b] = k.copy(e) : k.extend(f[b], e);
                    c = k.diff(j, f[b]);
                    j = k.copy(f[b]);
                    a || o.trigger("a:state:update:" + b, j, c, e);
                    return j
                } else 
                    p.error("Invalid value passed to A.state with a namespace of " + b + ".  Value: " + e, "A.state", "updateNamespace")
            }
            function j(b, e, a) {
                if (e.length === 1)
                    return b[e.shift()] = a, b;
                b[e.shift()] = j({}, e, a);
                return b
            }
            function n() {
                for (var b = document.getElementsByTagName("script"),
                e = 0, a = b.length; e < a; e++)
                    if (!g.data(b[e], "a-eval")) {
                        var c = g(b[e]), j = c.attr("data-a-state");
                        if (j && (j = k.parseJSON(j), j.key)) {
                            c = k.parseJSON(c.html());
                            g.data(b[e], "a-eval", !0);
                            var n = f[j.key];
                            n && k.extend(c, n);
                            m(j.key, c)
                        }
                    }
                }
            var g = c.$, f = {};
            i.declarative("a-state", function(b) {
                var e = b.$target, a = b.data.key, f = b.data[b.type];
                !f && o.events.defaults[b.targetTag] === b.type && (f = e.attr("name"));
                f && a && (e.is("select") && (e = e.find(":selected")), typeof e.val() !== void 0 && typeof f === "string" && (b = e.val(), e.is("input[type=checkbox]") &&
                !e.prop("checked") && (b = null), f = j({}, f.split("."), b)), m(a, f))
            });
            c = function(b, e, a) {
                return e === void 0 ? k.copy(f[b]) : m(b, e, !!a)
            };
            c.bind = function(b, e) {
                o.on("a:state:update:" + b, e)
            };
            c.replace = function(b, e, a) {
                return m(b, e, !!a, !0)
            };
            o.on("beforeReady", n);
            c.parse = n;
            return {
                state: c
            }
        });
        p.when("a-base", "a-util", "a-events", "a-declarative", "a-state").register("a-ajax", function(c, k, o, i, m) {
            function j(a) {
                return !a || typeof a === "string" ? "" : f.param(a)
            }
            function n(b, e) {
                b && b.length !== 0 && (typeof b === "string" && k.trim(b) === "" ?
                e && e(b) : (b[0]instanceof Array || (b = [b]), k.each(b, function(f) {
                    var c = a[f[0]];
                    c ? c.apply(window, f) : ((c = e) || p.error("There is no handler for the streaming ajax command: " + b[0], "A.ajax", "chunkHandler"), c(f))
                })))
            }
            function g(a, e) {
                var e = e || {}, f = e.headers || {};
                if (e.accepts !== b)
                    f.Accept = e.accepts;
                if (e.contentType !== b)
                    f["Content-Type"] = e.contentType;
                return l["a-ajax-update"]({
                    url: a,
                    cache: e.cache,
                    params: j(e.params),
                    method: e.method,
                    chunk: e.chunk,
                    success: e.success,
                    failure: e.failure || e.error,
                    abort: e.abort,
                    indicator: e.indicator,
                    timeout: e.timeout,
                    headers: f,
                    withCredentials: !!e.withCredentials
                })
            }
            var f = c.$, b, e = function() {
                var a, b;
                if (!window.XMLHttpRequest)
                    window.XMLHttpRequest = function() {
                        return new ActiveXObject("Microsoft.XMLHTTP")
                    };
                var e = function() {
                    function a() {
                        b.length > 0 ? b.pop().send() : e--
                    }
                    var b = [], e = 0, f = 0, c = 0;
                    return {
                        add: function(a) {
                            if (e < 4)
                                a.send(), e++;
                            else {
                                b.push(a);
                                f++;
                                if (b.length > c)
                                    c = b.length;
                                if ((a = window.ue) && a.count)
                                    a.count("aui:ajax:queued", f), a.count("aui:ajax:maxQueued", c)
                            }
                        },
                        complete: a,
                        abort: function(e) {
                            k.each(b, function(a,
                            f) {
                                if (a === e)
                                    return b.splice(f, 1), k.breaker
                            });
                            a()
                        }
                    }
                }();
                a = 4;
                b = function() {};
                var f = function(b) {
                    var f = b.http, c = b.responsePosition, g = f.readyState === 3, l = f.readyState === a && (f.status === 200 || f.status === 304);
                    if (g || l)
                        if (l = f.responseText, c < l.length) {
                            var c = l.substring(c, l.length), l = c.split("&&&"), j = c.lastIndexOf("&&&");
                            if (j===-1 && g)
                                return;
                                j < c.length - 3 && g && l.pop();
                                k.each(l, function(a) {
                                    var e;
                                    if (k.trim(a) !== "")
                                        try {
                                            e = k.parseJSON(a)
                                        } catch (f) {
                                            k.delay(function() {
                                                p.error("Invalid streaming ajax JSON response: " + a, "A.ajax",
                                                "streamingResponseHandler")
                                            }, 0)
                                        } else 
                                            e = a;
                                            b.callbacks.chunk(e)
                                        });
                                        b.responsePosition += j
                        }
                    f.readyState === a && (clearInterval(b.pollTimer), clearTimeout(b.timeoutTimer), e.complete(), f.status !== 200 && f.status !== 304 ? b.callbacks.failure(b, f.statusText, f.statusText) : b.callbacks.success(null, f.statusText, b), o.trigger("a:pageUpdate"), o.trigger("a:ajax:complete"))
                }, c = function(b) {
                    var f = b.http;
                    if (f.readyState === a) {
                        clearInterval(b.pollTimer);
                        clearTimeout(b.timeoutTimer);
                        e.complete();
                        var c = f.responseText;
                        try {
                            var g = k.parseJSON(c);
                            g && (c = g)
                        } catch (l) {}
                        f.status !== 200 && f.status !== 304 ? b.callbacks.failure(b, f.statusText, f.statusText) : b.callbacks.success(c, f.statusText, b);
                        o.trigger("a:ajax:complete")
                    }
                };
                return function() {
                    function g(b) {
                        if (b.http.readyState < a) {
                            clearInterval(b.pollTimer);
                            var f = "Request Timeout";
                            try {
                                f = b.http.statusText
                            } catch (c) {}
                            b.callbacks.failure(b, f, f);
                            e.complete()
                        }
                    }
                    function l(a, b, e) {
                        e = e || {};
                        e = k.extend({}, n.all, n[b], e);
                        k.each(e, function(b, e) {
                            (b || b === "") && a.setRequestHeader(e, b)
                        });
                        return a
                    }
                    function j(a, b, f, c, g, q, h, n,
                    i, m) {
                        var s = a.http;
                        s.open(b, f);
                        l(s, b, i);
                        a.timeout = c;
                        a.callbacks.chunk = g || a.callbacks.chunk;
                        a.callbacks.success = q || a.callbacks.success;
                        a.callbacks.failure = h || a.callbacks.failure;
                        a.callbacks.abort = n || a.callbacks.abort;
                        if (m)
                            s.withCredentials=!0;
                        e.add(a);
                        return {
                            abort: function() {
                                a.abort()
                            }
                        }
                    }
                    var n = {
                        all: {
                            "X-Requested-With": "XMLHttpRequest"
                        },
                        get: {
                            Accept: "text/html,*/*"
                        },
                        post: {
                            Accept: "text/html,*/*",
                            "Content-Type": "application/x-www-form-urlencoded"
                        }
                    }, i = function() {
                        var a = new XMLHttpRequest;
                        this.pollTimer = null;
                        this.http = a;
                        this.responsePosition = 0;
                        this.buffer = "";
                        this.callbacks = {
                            success: b,
                            failure: b,
                            chunk: b,
                            abort: b
                        }
                    };
                    i.prototype = {
                        send: function() {
                            var a = this;
                            a.http.send(a.params);
                            a.pollTimer = setInterval(function() {
                                if (a.http.readyState >= 2 && typeof a.http.responseText !== "unknown") {
                                    var b = a.http.getResponseHeader("Content-Type"), b = b ? b.toLowerCase(): "";
                                    (b.indexOf("application/json-amazonui-streaming")!==-1 || b.indexOf("application/amazonui-streaming-json")!==-1 ? f : c)(a)
                                }
                            }, 25);
                            a.timeout = typeof a.timeout === "undefined" ?
                            2E4 : a.timeout;
                            a.timeoutTimer = setTimeout(g, a.timeout, a)
                        },
                        get: function(a, b, e, f, c, g, h, l, q) {
                            if (b) {
                                var n = a.indexOf("?"), i = a.charAt(a.length - 1);
                                n>-1 ? i !== "?" && i !== "&" && (a += "&") : a += "?";
                                a += b
                            }
                            return j(this, "get", a, e, f, c, g, h, l, q)
                        },
                        abort: function() {
                            this.http && this.http.abort();
                            clearInterval(this.pollTimer);
                            clearTimeout(this.timeoutTimer);
                            e.abort(this);
                            this.callbacks.abort(this)
                        },
                        post: function(a, b, e, f, c, g, h, l, q) {
                            this.params = b;
                            return j(this, "post", a, e, f, c, g, h, l, q)
                        }
                    };
                    return i
                }()
            }(), a = {
                update: function(a, b, e) {
                    f(b).html(e)
                },
                append: function(a, b, e) {
                    a = f(b);
                    a.html(a.html() + e)
                },
                prepend: function(a, b, e) {
                    a = f(b);
                    a.html(e + a.html())
                },
                state: function(a, b, e) {
                    m.state(b, e)
                },
                script: function(a, b) {
                    eval(b)
                }
            }, l = {
                "a-ajax-update": function(a) {
                    var b = new e, c = function() {
                        var a = window.ue;
                        a && a.tag && (a.tag("aui"), a.tag("aui:ajax"))
                    }, g = a.abort, l = f(a.indicator);
                    l.show();
                    var j = typeof a.method === "string" && a.method.toLowerCase() === "post" ? "post": "get";
                    j === "get" && a.cache===!1 && (a.params += [a.params === "" ? "": "&", "_=", (new Date).getTime()].join(""));
                    return b[j](a.url,
                    a.params, a.timeout, function(b) {
                        c();
                        n(b, a.chunk)
                    }, function() {
                        l.hide();
                        c();
                        a.success && a.success.apply(window, arguments)
                    }, function() {
                        l.hide();
                        c();
                        a.failure && a.failure.apply(window, arguments)
                    }, g, a.headers, a.withCredentials)
                }
            };
            i.declarative("a-ajax-update", function(a) {
                var b = a.$target, e = a.action, f = a.data;
                if (f || o.events.defaults[a.targetTag] === a.type)
                    if (typeof f !== "object" || f[a.type]) {
                        var f = f || {}, c = f.url || b.attr("href") || b.attr("action"), g = j(f.params), n = b.attr("method") || f.method, i = f.indicator, f = f.timeout;
                        c || p.error("No ajax url provided.", "A.ajax", "declarativeHandler");
                        a.targetTag === "form" && a.type === o.events.defaults.form && (b = b.serialize(), g += b);
                        a.$event.preventDefault();
                        return l[e]({
                            url: c,
                            params: g,
                            method: n,
                            indicator: i,
                            operation: e,
                            timeout: f
                        })
                    }
            });
            return {
                ajax: g,
                get: function(a, b) {
                    b = b || {};
                    b.method = "get";
                    return g(a, b)
                },
                post: function(a, b) {
                    b = b || {};
                    b.method = "post";
                    return g(a, b)
                }
            }
        });
        p.when("a-base", "a-util", "p-detect", "a-prefix").register("a-animate", function(c, k, o, i) {
            function m(a, b, e) {
                a = a.jquery ? a[0] : a;
                b =
                i.prefixes.getStyle(b);
                a.style[b] = e
            }
            function j(a) {
                var b = "", e = o.capabilities.transform3d;
                a.top !== f && a.left !== f ? (b = "translate", e && (b += "3d"), b += "(" + a.left + ", " + a.top, e && (b += ", 0"), b += ")") : (a.top !== f ? b = "translateY(" + a.top + ")" : a.left !== f && (b = "translateX(" + a.left + ")"), e && (b += " translateZ(0)"));
                a.scale !== f && (b += " scale(" + a.scale + ")");
                return b
            }
            function n(a) {
                var e = {}, f=!1;
                k.each(b, function(b) {
                    b in a && (f=!0, e[b] = a[b], delete a[b])
                });
                return f ? e : null
            }
            function g(a, b, e) {
                o.capabilities.transform ? (typeof e === "string" &&
                (e = parseInt(e, 10)), k.isFiniteNumber(e) || (e = 0), a = parseInt(a.css(b), 10), k.isFiniteNumber(a) || (a = 0), e -= a, e += "px") : k.isFiniteNumber(e) && (e += "px");
                return e
            }
            var f, b = ["top", "left", "scale"], c = {
                animate: function(a, b, e, f, c) {
                    if (!a._a)
                        a._a = 0;
                    a._a++;
                    var g = function() {
                        a._a--;
                        c && c()
                    };
                    a.queue("fx", [function() {
                        a.animate(b, {
                            duration: e,
                            easing: f === "linear" ? f: "swing",
                            complete: g,
                            queue: !1
                        })
                    }
                    ])
                },
                fadeIn: function(a, b, e, f) {
                    a.fadeIn({
                        duration: b,
                        easing: e === "linear" ? e: "swing",
                        complete: f,
                        queue: !1
                    })
                },
                fadeOut: function(a, b, e, f) {
                    var c =
                    a.css("opacity");
                    a.fadeOut({
                        duration: b,
                        easing: e === "linear" ? e: "swing",
                        complete: function() {
                            + c < 0.95 && a.css("opacity", c);
                            f && f()
                        },
                        queue: !1
                    })
                },
                fadeToggle: function(a, b, e, f) {
                    a.fadeToggle({
                        duration: b,
                        easing: e === "linear" ? e: "swing",
                        complete: f,
                        queue: !1
                    })
                },
                slideUp: function(a, b, e, f) {
                    a.slideUp({
                        duration: b,
                        easing: e === "linear" ? e: "swing",
                        complete: f,
                        queue: !1
                    })
                },
                slideDown: function(a, b, e, f) {
                    a.slideDown({
                        duration: b,
                        easing: e === "linear" ? e: "swing",
                        complete: f,
                        queue: !1
                    })
                },
                slideToggle: function(a, b, e, f) {
                    a.slideToggle({
                        duration: b,
                        easing: e === "linear" ? e: "swing",
                        complete: f,
                        queue: !1
                    })
                },
                isAnimated: function(a) {
                    a = a.jquery ? a[0] : a;
                    return a._a && a._a > 0
                }
            }, e = {
                animate: function(a, b, e, c, i) {
                    var w = a[0], b = k.copy(b), e = e === f ? 250: e, c = c || "linear";
                    if (b.top !== f)
                        b.top = g(a, "top", b.top);
                    if (b.left !== f)
                        b.left = g(a, "left", b.left);
                    m(a, "transition", e < 4 ? "all 0ms" : "all " + e + "ms " + c);
                    if (e > 4) {
                        if (w._a === f)
                            w._a = 0;
                        w._a++;
                        var p = function() {
                            w._a--;
                            w._a || m(a, "transition", "");
                            i && i()
                        };
                        k.delay(function() {
                            k.delay(p, e)
                        }, 0)
                    }
                    o.capabilities.transform && (c = n(b)) && m(a, "transform",
                    j({
                        top: c.top,
                        left: c.left,
                        scale: c.scale
                    }));
                    k.objectIsEmpty(b) || a.css(b);
                    e <= 4 && i && k.delay(i, 0)
                },
                fadeIn: function(a, b, e, f) {
                    if (a.css("display") === "none"||+a.css("opacity") < 0.05) {
                        var c = a.css("opacity") || 1;
                        if (!c||+c < 0.05)
                            c = 1;
                        a.css("opacity", "0").show();
                        var g = this.animate;
                        k.delay(function() {
                            g(a, {
                                opacity: c
                            }, b, e, function() {
                                f && f()
                            })
                        }, 0)
                    } else 
                        f && f()
                },
                fadeOut: function(a, b, e, f) {
                    var c = a.css("opacity");
                    a.css("display") !== "none"&&+c > 0.05 ? this.animate(a, {
                        opacity: 0
                    }, b, e, function() {
                        a.hide().css("opacity", c);
                        f && f()
                    }) :
                    f && f()
                },
                fadeToggle: function(a, b, e, f) {
                    (a.css("display") === "none"||+a.css("opacity") < 0.05 ? this.fadeIn : this.fadeOut).call(this, a, b, e, f)
                },
                slideUp: function(a, b, e, f) {
                    var c = this.animate;
                    a.css({
                        height: a.innerHeight(),
                        overflow: "hidden"
                    });
                    k.delay(function() {
                        c(a, {
                            height: 0
                        }, b, e, function() {
                            a.hide();
                            a.css({
                                height: "",
                                overflow: ""
                            });
                            f && f()
                        })
                    }, 0)
                },
                slideDown: function(a, b, e, f) {
                    var c = a.innerHeight(), g = this.animate;
                    a.css({
                        height: 0,
                        overflow: "hidden"
                    });
                    a.show();
                    k.delay(function() {
                        g(a, {
                            height: c
                        }, b, e, function() {
                            f && f();
                            a.css({
                                height: "",
                                overflow: ""
                            })
                        })
                    }, 0)
                },
                slideToggle: function(a, b, e, f) {
                    (a.is(":visible") ? this.slideUp : this.slideDown).call(this, a, b, e, f)
                },
                isAnimated: function(a) {
                    a = a.jquery ? a[0] : a;
                    return a._a && a._a > 0
                }
            };
            return o.capabilities.transition ? e : c
        });
        p.when("a-base", "a-util", "a-events", "a-defer", "p-detect").register("a-image", function(c, k, o, i, m) {
            function j(b) {
                var b = g(b), e = b.data("a-dynamic-image");
                if (e && typeof e === "object") {
                    var f = b.data("a-dynamic-image-container");
                    typeof f === "undefined" && (f = b.closest(".a-dynamic-image-container"),
                    f.length === 0 && (f = b.parent()), b.data("a-dynamic-image-container", f));
                    var c = window.devicePixelRatio ? window.devicePixelRatio: 1, l = f.width() * c, j = f.height() * c, n = Number.MAX_VALUE, i = Number.MAX_VALUE, m = b.attr("src") || "", o, p = l / j;
                    k.each(e, function(a, b) {
                        var e = parseInt(a[0], 10), f = parseInt(a[1], 10);
                        e -= j;
                        f -= l;
                        f = p >= 1 ? f : e;
                        Math.abs(f) < i && f >= 0 && (i = Math.abs(f), o = b);
                        Math.abs(f) < n && (n = Math.abs(f), m = b)
                    });
                    o && (m = o);
                    a.schedule(m, b);
                    a.fill();
                    return m
                }
            }
            function n() {
                g("img.a-dynamic-image").each(function() {
                    g(this).data("a-manual-replacement") ||
                    j(this)
                })
            }
            var g = c.$;
            g(window);
            var f = document.getElementsByTagName("img"), b = {}, e = function() {
                return {
                    canLoad: function() {
                        return !0
                    },
                    inFlight: 0,
                    limit: function() {
                        return 2
                    }
                }
            }(), a = function() {
                var a = [], f = {};
                return {
                    schedule: function(e, c) {
                        f[e] || (a.push(e), f[e]=!0);
                        b[e] = b[e] || [];
                        for (var g = 0; g < b[e].length; g++)
                            if (c.is(b[e][g]))
                                return;
                        b[e].push(c)
                    },
                    fill: function() {
                        if (e.canLoad())
                            for (var b = 0; b < e.limit() - e.inFlight; b++)
                                if (a.length > 0) {
                                    var c = a.shift();
                                    f[c]=!1;
                                    l.load(c)
                                }
                    }
                }
            }(), l = function() {
                function c(a) {
                    var e = b[a];
                    e && e.length >
                    0 && k.each(e, function(b) {
                        i.defer(function() {
                            b.attr("src", a)
                        })
                    })
                }
                var l = {};
                return {
                    load: function(b) {
                        l[b] && c(b);
                        if (l[b]!==!1) {
                            var f = new Image;
                            f.onload = function() {
                                e.inFlight--;
                                c(b);
                                l[b]=!0;
                                a.fill()
                            };
                            f.onerror = function() {
                                e.inFlight--;
                                l[b]=!1;
                                a.fill()
                            };
                            e.inFlight++;
                            i.defer(function() {
                                f.src = b
                            })
                        }
                    },
                    poll: function() {
                        k.each(f, function(b) {
                            b = g(b);
                            !b.data("a-hires-loaded")&&!b.is(":hidden") && k.onScreen(b)&&!b.data("a-manual-replacement") && (b.data("a-hires") && a.schedule(b.data("a-hires"), b), b.data("a-hires-loaded",
                            !0))
                        })
                    }
                }
            }();
            m.capabilities.hires && o.on.ready(function() {
                k.interval(function() {
                    l.poll();
                    a.fill()
                }, 2E3)
            });
            o.on.ready(n);
            g(window).resize(n);
            return {
                loadHiResImage: function(b) {
                    var e = [];
                    g(b).each(function() {
                        var b = g(this), f = b.data("a-hires");
                        f && (a.schedule(f, b), a.fill(), e.push(f));
                        b.data("a-hires-loaded", !0)
                    });
                    return e
                },
                loadDynamicImage: function(a) {
                    var b = [];
                    g(a).each(function() {
                        b.push(j(this))
                    });
                    return b
                },
                loadImageManually: function(a, b) {
                    var e = [];
                    g(a, b).each(function() {
                        var a = g(this);
                        if (!a.data("a-image-already-loaded")) {
                            a.data("a-image-already-loaded",
                            !0);
                            var b = j(a), f = g("<img>").attr("src", b || a.data("a-image-source"));
                            e.push(b);
                            var b = "" + this.className, c = a.data("a-extra-classes");
                            c && (b += " " + c);
                            f.attr("class", b);
                            f.attr("id", this.id);
                            f.attr("style", a.attr("style"));
                            f.attr("alt", a.attr("alt"));
                            f.attr("usemap", a.attr("usemap"));
                            f.attr("title", a.attr("title"));
                            k.each(this.attributes, function(a) {
                                a && a.name && a.name.indexOf("data-") === 0 && f.attr(a.name, a.value)
                            });
                            f.data(a.data());
                            a.replaceWith(f)
                        }
                        return e
                    })
                },
                loadDescendantImagesManually: function(a, b) {
                    return this.loadImageManually(g(a,
                    b).find("div.a-manually-loaded").filter(function() {
                        return !g(this).data("a-image-already-loaded")
                    }))
                }
            }
        });
        p.when("a-util", "a-defer", "a-base", "a-events", "a-declarative", "a-state", "a-ajax", "a-animate", "a-image", "a-constants", "a-detect", "a-browser-events", "a-preload", "a-prefix", "a-request-animation-frame").register("A", function(c) {
            var k = {};
            c.each(arguments, function(o) {
                c.extend(k, o)
            });
            return k
        });
        p.register("a-constants", function() {
            return {
                constants: {
                    keycodes: {
                        BACKSPACE: 8,
                        TAB: 9,
                        ENTER: 13,
                        ESCAPE: 27,
                        SPACE: 32,
                        LEFT_ARROW: 37,
                        UP_ARROW: 38,
                        RIGHT_ARROW: 39,
                        DOWN_ARROW: 40,
                        DELETE: 46
                    },
                    NOOP: function() {}
                }
            }
        });
        p.when("jQuery", "a-detect", "a-events", "a-util", "a-defer").register("a-browser-events", function(c, k, o, i) {
            var m, j, n;
            function g() {
                return window.orientation === s ? b() > f() ? 90 : 0 : window.orientation
            }
            function f() {
                return window.innerHeight ? window.innerHeight : document.documentElement.clientHeight
            }
            function b() {
                return window.innerWidth ? window.innerWidth : document.documentElement.clientWidth
            }
            function e() {
                return window.innerWidth ?
                Math.round(document.documentElement.clientWidth / window.innerWidth * 10) / 10 : 1
            }
            function a(a) {
                switch (a) {
                case m:
                    a = "orientation,height,width,zoom,scrollLeft,scrollTop".split(",");
                    break;
                case j:
                    a = ["scrollLeft", "scrollTop"];
                    break;
                case n:
                    a = ["height", "width", "zoom", "scrollLeft", "scrollTop"];
                    break;
                default:
                    a = ["orientation", "height", "width", "scrollLeft", "scrollTop"]
                }
                for (var c = {}, l, i; (i = a.pop()) !== s;)
                    l = w[i], i === "orientation" ? w[i] = g() : i === "height" ? w[i] = f() : i === "width" ? w[i] = b() : i === "scrollTop" ? w[i] = window.scrollY ?
                    window.scrollY : x.scrollTop() : i === "scrollLeft" ? w[i] = window.scrollX ? window.scrollX : x.scrollLeft() : i === "zoom" && (w[i] = e()), w[i] !== l && (c[i] = l);
                return c
            }
            function l(a) {
                if (a = A[a])
                    if (a.pollCounter = a.maxPollCount, !a.intervalId)
                        a.intervalId = setInterval(a.handler, a.pollInterval)
            }
            function q(a) {
                if ((a = A[a]) && a.intervalId)
                    clearInterval(a.intervalId), a.intervalId = 0
            }
            var s, x = c(window);
            j = "scroll";
            n = "zoom";
            m = "all";
            var w = {
                scrollLeft: 0,
                scrollTop: 0,
                height: f(),
                width: b(),
                orientation: g(),
                zoom: e()
            };
            o.on("beforeReady", function() {
                a(m)
            });
            var p, t = {
                speed: 0,
                degree: 0,
                direction: "",
                positionX: 0,
                positionY: 0
            }, v = [], z;
            x.bind("mousemove", i.throttle(function(a) {
                a = {
                    x: a.clientX,
                    y: a.clientY
                };
                if (z) {
                    var b = z, e = Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2)) / 50 * 10, f = Math.atan2(a.y - b.y, a.x - b.x) / (Math.PI / 180), c = b = 0;
                    v.push({
                        speed: e,
                        degree: f
                    });
                    v.length > 4 && (v = v.slice( - 4));
                    e = v.length;
                    for (f = 0; f < e; f++)
                        b += v[f].speed, c += v[f].degree;
                    b = Number((b / e).toFixed(2));
                    c = Math.round(c / e);
                    t = {
                        speed: b,
                        degree: c,
                        direction: c >= 0 ? c > 157.5 ? "W": c > 112.5 ? "SW": c > 67.5 ? "S": c > 22.5 ? "SE":
                        "E": c<-157.5 ? "W": c<-112.5 ? "NW": c<-67.5 ? "N": c<-22.5 ? "NE": "E",
                        positionX: a.x,
                        positionY: a.y
                    };
                    z = a
                } else 
                    a && (z = a)
            }, 50));
            x.bind(j, i.throttle(function() {
                var b = a(j);
                o.trigger(j, w, b)
            }, 100));
            var A = {};
            i.each(["resize", n], function(a) {
                A[a] = {
                    handler: function() {},
                    lastViewport: i.copy(w),
                    maxPollCount: 5,
                    pollCounter: 5,
                    pollInterval: 100,
                    intervalId: 0
                }
            });
            A.resize.handler = function() {
                var b = [], e = A.resize;
                a("resize");
                var f = i.diff(w, e.lastViewport);
                f.orientation && b.push("orientationchange");
                f.width || f.height ? b.push("resize") :
                k.capabilities.isIETouchCapable && f.scrollTop && b.push("resize");
                if (b.length)
                    e.lastViewport = i.copy(w), i.each(b, function(a) {
                        o.trigger(a, w, f)
                    });
                --e.pollCounter === 0 && q("resize")
            };
            A.resize.pollInterval = 100;
            A.resize.maxPollCount = 10;
            x.bind("resize", function() {
                l("resize")
            });
            A.zoom.handler = function() {
                a(n);
                var b = A.zoom, e = i.diff(w, b.lastViewport);
                if (e.zoom)
                    b.lastViewport = i.copy(w), o.trigger(n, w, e);
                --b.pollCounter === 0 && q(n)
            };
            A.zoom.pollInterval = 200;
            k.capabilities.android && x.bind("touchcancel", function(a) {
                if (a.originalEvent.changedTouches.length ===
                2)
                    A.zoom.maxPollCount = 15, l(n)
            });
            k.capabilities.ios && x.bind("touchend", function(a) {
                if (a.originalEvent.touches.length === 1)
                    A.zoom.maxPollCount = 1, l(n)
            });
            !k.capabilities.ios&&!k.capabilities.android && x.bind("resize", function() {
                A.zoom.maxPollCount = 5;
                l(n)
            });
            return {
                viewport: function(b) {
                    b && a(m);
                    return i.copy(w)
                },
                cursor: function() {
                    return i.copy(t)
                },
                scrollBarWidth: function(a) {
                    if (a || (document && document.body && document.body.scrollHeight ? document.body.scrollHeight : 0) > f()) {
                        if (p === s) {
                            var a = c('<div style="visibility:hidden"><div style="width:100%"></div></div>'),
                            b = a.children(), e;
                            a.appendTo("body");
                            e = a[0].offsetWidth;
                            a.css("overflow", "scroll");
                            b = b[0].offsetWidth;
                            a.remove();
                            p = e - b
                        }
                        return p > 0 ? p : 15
                    } else 
                        return 0
                }
            }
        });
        p.when("a-base", "a-events").register("a-preload", function(c, k) {
            function o() {
                if (p.length > 0)
                    x = p;
                else if (x = w, x.length === 0 ||!t)
                    return !1;
                if (u >= J)
                    return !1;
                u++;
                return !0
            }
            function i(a, e) {
                var f = function() {
                    if (a) {
                        var b = a.parentElement;
                        b && b.removeChild(a);
                        a = null
                    }
                };
                e && clearTimeout(e);
                u = u < 1 ? 0 : u - 1;
                v ? setTimeout(f, 5) : f();
                C ? b(g, 0) : g()
            }
            function m(a) {
                var b, e, f = a.indexOf("?"),
                f = f > 0 ? f: a.length;
                (e = a.lastIndexOf(".", f)) && (b = a.substring(e + 1, f).toLowerCase());
                return b === "gz" ? m(a.substring(0, e)) : b
            }
            function j(a, e, f) {
                var c;
                if (v || z&&!f)
                    c = b(function() {
                        i(a, c)
                    }, 2500 + Math.random() * 100);
                a.onerror = function() {
                    i(a, c)
                };
                a.onload = function() {
                    i(a, c)
                }
            }
            function n(a) {
                O && O.count && O.count(a, (O.count(a) || 0) + 1)
            }
            function g() {
                if (o()) {
                    var b = x.pop(), f = m(b);
                    a:
                    {
                        for (var c = L.length; c--;)
                            if (L[c] === f) {
                                f=!0;
                                break a
                            }
                        f=!1
                    }
                    v ? c = h : C ? (c = G, f || (D >= 10 ? c = E : D === 9 && (c = H))) : c = A ? f ? G : null : G;
                    var l;
                    if (c) {
                        l = c === G ? new Image : e.createElement(c);
                        l.style.display = "none";
                        if (c === G)
                            j(l, c, f, b), l.src = b;
                        else if (c === h)
                            l.data = b, j(l, c, f, b);
                        else if (c === H)
                            l.type = "text/cache", j(l, c, f, b), l.src = b;
                        else if (c === E)
                            l.rel = "stylesheet", l.media = "speech", j(l, c, f, b), l.href = b;
                        if (!(C && c === G))
                            try {
                                a.appendChild(l)
                            } catch (k) {
                            i();
                            return 
                        }
                        n("aui:preload_fulfilled");
                        u < J && g()
                    } else 
                        i()
                }
            }
            function f(a, b) {
                if (F) {
                    if (typeof a === "string")
                        a = [a];
                    else if (typeof a !== "object" || a === null)
                        return;
                    var e, c;
                    for (e = 0; e < a.length; e++)(c = a[e]) 
                        && typeof c !== "string" ? f(c, b) : c && c[0] !== " " && (b.splice(Math.round(Math.random() *
                        b.length), 0, c), n("aui:preload_asks"))
                }
            }
            var b = setTimeout, e = document, a = e.documentElement, l = a.style, q = navigator, s = q.userAgent, x = [], w = [], p = [], t=!1, v = "MozAppearance"in l, z=!v && "webkitAppearance"in l, A = z && q.vendor.indexOf("Apple") === 0, C=!v&&!z && (q.appName.indexOf("Microsoft") === 0 || s.indexOf("Trident/")>-1), F = z || v || C, l=!C?-1 : /Trident\/([\d]+)/.exec(s) !== null ? parseFloat(RegExp.$1) : null, D=!C?-1 : !l ? 6 : l + 4, J = typeof window.plCount !== "undefined" ? window.plCount() : C && D < 8 ? 2 : 5, u = 0, E = "LINK", G = "IMG", h = "OBJECT", H = "SCRIPT",
            L = ["gif", "jpeg", "jpg", "png"], O = window.ue;
            k.on.afterLoad(function() {
                t=!0;
                g()
            });
            return {
                preload: function(a, b) {
                    b ? b(a) : (f(a, w), g())
                }
            }
        });
        p.when("a-util").register("a-request-animation-frame", function(c) {
            for (var k = 0, o = ["ms", "moz", "webkit", "o"], i = 0; i < o.length&&!window.requestAnimationFrame; ++i)
                window.requestAnimationFrame = window[o[i] + "RequestAnimationFrame"], window.cancelAnimationFrame = window[o[i] + "CancelAnimationFrame"] || window[o[i] + "CancelRequestAnimationFrame"];
            if (!window.requestAnimationFrame)
                window.requestAnimationFrame =
                function(i) {
                    var j = c.now(), n = Math.max(0, 16 - (j - k)), g = window.setTimeout(function() {
                        i(j + n)
                    }, n);
                    k = j + n;
                    return g
                };
            if (!window.cancelAnimationFrame)
                window.cancelAnimationFrame = function(c) {
                    clearTimeout(c)
                };
            return {
                requestAnimationFrame: function(c, j) {
                    return window.requestAnimationFrame(c, j)
                },
                cancelAnimationFrame: function(c) {
                    window.cancelAnimationFrame(c)
                }
            }
        })
    })
})(function() {
    var p = window.AmazonUIPageJS || P, c = p.attributeErrors;
    return c ? c("AmazonUIBaseJS") : p
}());
(function(p) {
    p.execute(function() {})
})(function() {
    var p = window.AmazonUIPageJS || P, c = p.attributeErrors;
    return c ? c("AmazonUITouchJS") : p
}());
(function(p) {
    p.execute(function() {
        p.when("A", "a-popover-base-factory").register("a-popover-base-apis", function(c, k) {
            return {
                show: function(c) {
                    var i = k.get(c.$trigger ? c.$trigger : c);
                    if (i)
                        return i.show.apply(i, arguments)
                },
                hide: function(c) {
                    var i = k.get(c);
                    if (i)
                        return i.unlock(1), i.hide.apply(i, arguments)
                },
                get: function(c) {
                    return k.get(c)
                },
                remove: function(c) {
                    return k.remove(c)
                }
            }
        });
        p.when("A", "a-popover-iframes", "a-popover-util", "a-popover-objectclass", "a-popover-data").register("a-popover-base-factory", function(c,
        k, o, i, m) {
            function j(b) {
                return a[b] ? a[b] : null
            }
            function n(a, b) {
                var c = null;
                if (typeof a === "number")
                    c = j(a);
                else if (typeof a === "string")(c = e[a] ? e[a] : null) || (c = j(a));
                else if (typeof a === "object")
                    if (a.$popover)
                        c = a;
                    else {
                        var g = f(a), c = g.data("a-popover-id");
                        c || (c = g.find(".a-declarative").eq(0), c = c.length ? c.data("a-popover-id") : null);
                        c = j(c);
                        if (!c) {
                            var l = g.data("action");
                            if ((l = l ? g.data(l) : null) && l.name)
                                if ((c = e[l.name] ? e[l.name] : null) && (!b || c.type === b))
                                    if ((l = (l = c.attrs("currentDataStrategy")) ? m.getStrategyByName(l) :
                                    m.guessStrategyByAttrs(c.attrs())) && l.reusePopover) {
                                        if (c.$trigger[0] !== g[0])
                                            c.$trigger.data("a-popover-id", null), c.$trigger = g
                                    } else 
                                        c = null;
                                    else 
                                        c = null
                        }
                    }
                return c
            }
            function g() {
                q || (q = new i.PopoverClass({
                    id: - 1,
                    $popover: l,
                    $trigger: l,
                    immersive: !0
                }, {
                    isActive: function() {
                        return !0
                    },
                    hideMethod: function() {
                        this.hideChildren()
                    },
                    showMethod: c.constants.NOOP
                }));
                return q
            }
            var f = c.$, b = 1, e = {}, a = {}, l = f("<div style='z-index:-1;position:absolute;' />").appendTo("body"), q;
            return {
                getRoot: g,
                get: function(a, b) {
                    var b = b ? b: this ?
                    this.type: null, e = n(a, b);
                    return e && b && e.type !== b ? null : e
                },
                create: function(l, m) {
                    var q = f(l), o = m.attributes || {}, t = m.variant || {}, v = m.actionCheck ||!1;
                    q.data("a-popover-id");
                    var z = o.type, A = null;
                    if (z && (!q.hasClass("a-declarative") ||!q.data("action") || q.data("action").indexOf(z)===-1))
                        q = c.declarative.create(q, "a-" + z), l = q[0];
                    if (v && q.data("action") && q.data("action").indexOf(z)===-1)
                        return null;
                    z && q && (A = n(q));
                    return A ? A.type !== z ? null : A : (v = f(l), q = null, o.type?!v ||!v.length ? t = null : (o = c.extend({
                        id: b++,
                        $trigger: v,
                        $triggerWrapper: null,
                        $iframe: k.get()
                    }, o), t = c.copy(t), q = new i.PopoverClass(o, t), a[q.id] = q, q.name && (e[q.name] = q), v.data("a-popover-id", q.id), t = q.$trigger.closest(".a-popover"), t=!q.attrs("immersive") && t.length ? j(t.data("a-popover-id")) || g() : g(), q.parent = t, t.children.push(q), t = q) : t = null, t)
                },
                remove: function(b, f) {
                    var g = this.get(b), l=!1;
                    if (g) {
                        l = g.id;
                        if (g && l>-1) {
                            var j = c.indexOfArray(g.parent.children, g), n = g.$container, i = g.$trigger;
                            g.parent.children.splice(j, 1);
                            g.unlock().hide();
                            g.update({
                                content: ""
                            });
                            k.release(g.$iframe);
                            n && g.$container.remove();
                            i.data("a-popover-id", "");
                            g.name && delete e[g.name];
                            delete a[l];
                            l=!0
                        } else 
                            l=!1;
                        f && c.declarative.remove(g.$trigger[0], "a-" + f)
                    }
                    return l
                }
            }
        });
        p.when("A", "a-popover-util", "a-popover-base-factory").register("a-popover-base-handlers", function(c, k, o) {
            function i(c) {
                for (var n; c.length;) {
                    if (n = c.data("a-popover-id"))
                        break;
                    c = c.parent()
                }
                return o.get(n)
            }
            var m = c.$;
            m(document).bind("click " + c.action.start, function(j) {
                var n = m(j.target), g = j.originalEvent;
                if (!g ||!g.pointerType ||!(g.pointerType ===
                c.pointerType.touch && g.type === "click"))
                    if (!n.hasClass("a-modal-scroller")&&!(n[0].id === "a-popover-lgtbox" || n[0].nodeName.toLowerCase() === "html")) {
                        var f = function(b) {
                            return k.eventOccursWithin(j, b)
                        };
                        c.each(o.getRoot().children, function(b) {
                            if (b.isLoaded()) {
                                var e = k.search(b, f);
                                e ? e.hideChildren() : b.attrs("lightboxOptions") === null&&!b.attrs("immersive") && b.unlock(1).hide()
                            }
                        })
                    }
            });
            c.declarative("a-popover-close", ["click", c.action.start], function(c) {
                var n = i(c.$target);
                n && (n.unlock().hide(), k.trigger("dismiss",
                n));
                c.$event.preventDefault()
            });
            c.declarative("a-popover-a11y", "keydown", function(j) {
                var n = j.$event, g = n.keyCode;
                if (g === c.constants.keycodes.TAB || c.constants.keycodes.ESCAPE)
                    if (j = i(j.$target))
                        if (g === c.constants.keycodes.ESCAPE)
                            j.hide();
                        else if (g === c.constants.keycodes.TAB)
                            g = document.activeElement, j.$startAnchor.length && g === j.$startAnchor[0] && n.shiftKey ? j.$endAnchor.focus() : j.$endAnchor.length && g === j.$endAnchor[0]&&!n.shiftKey && j.$startAnchor.focus()
            });
            c.on("resize zoom", function() {
                o.getRoot().updatePosition()
            })
        });
        p.when("A", "a-popover-base-apis", "a-popover-base-handlers").register("a-popover-base", function(c, k) {
            return k
        });
        p.when("A", "a-popover-util", "a-popover-data", "a-popover-position", "a-popover-iframes", "a-popover-lightbox").register("a-popover-objectclass", function(c, k, o, i, m, j) {
            function n(n, i) {
                var v =- 1, o = [1], p =- 2;
                this.parent = null;
                this.children = [];
                this.dependencies = {};
                this.attributes = {
                    position: "triggerVertical",
                    alone: !1,
                    immersive: !1
                };
                var x = function(a, b) {
                    var e = this.isActive(), l = this.getDataStrategy(), j =
                    !this.$popover, n = b || j, i=!1, a = a || n;
                    if (!n)
                        for (var m = o.length; m--&&!n;)
                            n=!s[o[m]];
                    if (n)
                        n = t.apply(this), n = f(n), j || (l.unloadContent(this), this.$popover.remove(), i=!0), f("body").append(n), this.$container = n, this.$popover = this.$container.hasClass("a-popover") ? this.$container : this.$container.find(".a-popover"), this.$startAnchor = this.$popover.hasClass("a-popover-start") ? this.$popover : this.$popover.find(".a-popover-start"), this.$endAnchor = this.$popover.find(".a-popover-end"), this.$popover.attr("id", "a-popover-" +
                        this.id).data("a-popover-id", this.id);
                    this.attrs("immersive") || (n = parseInt(this.parent.$popover.css("z-index"), 10), c.isFiniteNumber(n) || (n = this.parent.attrs("immersive") ? 1010 : 0), this.$popover.css("z-index", Math.max(299, 100 + n)));
                    if (l.shouldRefreshContent(this) || a)
                        i || l.unloadContent(this), l.loadContent(this, j);
                    this.dependencies.updateDimensions !== g && this.dependencies.updateDimensions.apply(this);
                    o = [];
                    e && u.call(this, [], !1);
                    return this
                }, u = function(b, e) {
                    function f() {
                        j.updatePosition();
                        var a = j.attrs("navigate");
                        !e && a && j.attrs("navigate", !1);
                        i.apply(j, b);
                        e && k.trigger("show", j);
                        q && q.apply(j, b);
                        e && k.trigger("afterShow", j);
                        j.$popover.attr("aria-hidden", "false");
                        !e && a && j.attrs("navigate", a);
                        p = l
                    }
                    var j = this, e=!!e, n = j.dependencies, i = n.showMethod !== g ? n.showMethod : w, m = n.beforeShowMethod !== g ? n.beforeShowMethod : null, q = n.afterShowMethod !== g ? n.afterShowMethod : null;
                    p = a;
                    j.attrs("originalFocus", document.activeElement);
                    j.$popover.css("visibility", "hidden").addClass("a-popover-hidden").show();
                    m && m.apply(j, b);
                    j.attrs("synchronous") ?
                    f() : c.delay(function() {
                        f()
                    }, 0)
                };
                this.show = function() {
                    var a = this, e = a.attrs("lightboxOptions") || null;
                    a.isActive() || (a.lock(b), e && j.lock(b), a.attrs("alone") && c.each(a.parent.children, function(b) {
                        b.isActive() && b.id !== a.id && b.unlock().hide()
                    }), k.trigger("beforeShow", a), (!a.$container || a.isDirty() || a.getDataStrategy().shouldRefreshContent(a)) && x.call(a), e && j.show(c.extend({
                        popover: a
                    }, e)), u.call(a, arguments, !0), c.delay(function() {
                        a.unlock(b);
                        e && j.unlock(b)
                    }, 0));
                    return this
                };
                this.hide = function() {
                    var a = this,
                    b = a.dependencies, e = b.hideMethod !== g ? b.hideMethod: y, f = b.beforeHideMethod !== g ? b.beforeHideMethod: null, l = b.afterHideMethod !== g ? b.afterHideMethod: null, n = a.attrs("lightboxOptions") || null;
                    !a.isLocked() && a.isActive() && (p = q, a.hideChildren(), k.trigger("beforeHide", a), f && f.apply(a, arguments), e.apply(a, arguments), m.release(a.$iframe), k.trigger("hide", a), c.delay(function() {
                        l && l.apply(a, arguments);
                        a.$popover.attr("aria-hidden", "true");
                        n && (a.parent.attrs("lightboxOptions") ? j.show(c.extend({
                            popover: a.parent
                        }, n)) :
                        j.hide(n));
                        k.trigger("afterHide", a);
                        p =- 2
                    }, 0));
                    return this
                };
                this.update = function(a) {
                    var b = typeof a === "string" ? {
                        content: a
                    }
                    : c.copy(a), e = this.attrs(), a = this.getDataStrategy();
                    c.each(b, function(a, b) {
                        (a&&!e[b] || e[b] && e[b] !== a) && o.push(b)
                    });
                    this.isDirty() && (b = c.extend({}, e, b), this.attrs(b), this.getDataStrategy(b), this.$popover && a.unloadContent(this), this.isActive() && x.call(this, !0));
                    return this
                };
                this.refresh = function(a, b) {
                    return x.call(this, a ||!0, b ||!1)
                };
                this.isActive = function() {
                    return p >= a
                };
                this.isLoaded =
                function() {
                    return p === l
                };
                this.isDirty = function() {
                    return o.length > 0
                };
                this.lock = function(a) {
                    a || (a = e);
                    v < a && (v = a);
                    return this
                };
                this.unlock = function(a) {
                    a || (a = e);
                    v <= a && (v =- 1);
                    return this
                };
                this.isLocked = function() {
                    return v!==-1
                };
                this.dependencies = i;
                this.attrs(n);
                c.extend(this, this.attributes)
            }
            var g, f = c.$, b = 1, e = 10, a = 1, l = 2, q =- 1, s = {
                name: !0,
                url: !0,
                content: !0,
                width: !0,
                height: !0,
                "max-width": !0,
                "max-height": !0,
                "min-width": !0,
                "min-height": !0
            }, p = c.capabilities.mobile && c.capabilities.isIE10Plus, w = function() {
                this.$popover.css({
                    visibility: "visible"
                }).removeClass("a-popover-hidden");
                this.focus()
            }, y = function() {
                var a = this.attrs("originalFocus");
                this.$popover.hide().find(".a-lgtbox-vertical-scroll").removeClass("a-lgtbox-vertical-scroll");
                a && a.focus()
            }, t = function() {
                var a = this.dependencies;
                return a.skin !== g ? a.skin(this) : ""
            }, v = n.prototype;
            v.getDataStrategy = function(a) {
                var b = this.dependencies;
                !a&&!this.attrs("currentDataStrategy") && (a = this.attrs());
                if (a && (a = a.dataStrategy ? o.getStrategyByName(a.dataStrategy) : o.guessStrategyByAttrs(a)))
                    b.dataStrategy = a, this.attrs("currentDataStrategy",
                    a.name);
                return b.dataStrategy
            };
            v.getContent = function() {
                return this.dependencies.getContent !== g ? this.dependencies.getContent.apply(this, arguments) : null
            };
            v.updateContent = function(a) {
                this.dependencies.updateContent !== g && this.dependencies.updateContent.apply(this, arguments);
                return this
            };
            v.ajax = function(a) {
                return this.update({
                    url: a
                })
            };
            v.updateChildrenPosition = function() {
                c.each(this.children, function(a) {
                    a.isActive() && a.updatePosition()
                });
                return this
            };
            v.updatePosition = function() {
                var a = this;
                if (a.id===-1)
                    c.each(a.children,
                    function(a) {
                        a.isActive() && a.updatePosition()
                    });
                else {
                    var b = a.$popover;
                    c.capabilities.isMetroIEGuess && c.capabilities.isIETouchCapable ? b.css("opacity", 0.01) : b.css("visibility", "hidden");
                    var e = function() {
                        var e = b.find(".a-popover-inner").css({
                            height: "auto",
                            "overflow-y": "auto"
                        }), g = a.attrs("position"), l = {}, l = a.dependencies.positionStrategy ? i.customPosition(a, a.dependencies.positionStrategy): i[g](a);
                        k.trigger("beforeUpdatePosition", a);
                        g = {
                            top: l.top + "px",
                            left: l.left + "px"
                        };
                        c.capabilities.isMetroIEGuess && c.capabilities.isIETouchCapable ?
                        g.opacity = 1 : g.visibility = "visible";
                        b.css(g);
                        f(document.activeElement).closest(a.$popover).length === 0 && a.focus();
                        if (e.length && (!e[0].style.height || e[0].style.height === "auto")) {
                            var j = b.outerHeight() || 0, n = b.find(".a-popover-header").outerHeight(!0) || 0, m = b.find(".a-popover-footer").outerHeight(!0) || 0, g = e.outerHeight(!0) || 0, j = j - n - m;
                            g > j && e.css({
                                height: j + "px",
                                "overflow-y": "scroll"
                            })
                        }
                        a.$iframe.height(b.outerHeight()).width(b.outerWidth()).css("z-index", parseInt(b.css("z-index"), 10) - 2).show().offset({
                            top: l.top,
                            left: l.left
                        });
                        k.trigger("afterUpdatePosition", a);
                        k.trigger("positionUpdated", a);
                        c.each(a.children, function(a) {
                            a.isActive() && a.updatePosition()
                        })
                    };
                    a.attrs("immersive") && (c.capabilities.mobile || c.capabilities.tablet) ? (b.css({
                        top: 0,
                        left: 0
                    }), c.delay(function() {
                        e()
                    }, 0)) : e()
                }
                return a
            };
            v.attrs = function(a, b) {
                var e = this;
                return b === g && typeof a !== "object" ? a ? typeof a === "string" ? this.attributes[a] !== g ? this.attributes[a] : null : null : this.attributes : (typeof a === "object" ? c.each(a, function(a, b) {
                    e.attrs(b, a)
                }) : typeof a ===
                "string" && (this.attributes[a] = b, e[a] = b), this)
            };
            v.hideChildren = function() {
                c.each(this.children, function(a) {
                    a.unlock(b);
                    a.hide()
                });
                return this
            };
            v.focus = function() {
                var a = this, b = f(window), e = b.scrollTop(), g = a.$popover.offset().top;
                p && e > g && b.scrollTop(g);
                c.delay(function() {
                    a.$startAnchor.focus()
                }, 0);
                return this
            };
            return {
                PopoverClass: n
            }
        });
        p.when("jQuery", "ready").register("a-changeover", function(c) {
            c(document).delegate(".a-changeover:not(.a-changeover-manual)", "webkitAnimationEnd animationend click touchstart",
            function() {
                this.style.display = "none"
            })
        });
        p.when("A", "a-popover-util").register("a-popover-ajax-strategy", function(c, k) {
            var o = c.$, i = o("html").hasClass("a-lt-ie8");
            return {
                name: "ajax",
                reusePopover: !1,
                loadContent: function(m, j) {
                    var n = m.attrs("url"), g = m.attrs("timeout") || 1E4, f = m.attrs("ajaxFailMsg") || "Sorry, content is not available.", b=!!m.attrs("cache"), e = m.attrs("spinnerTimer"), a = m.attrs("ajaxHandler"), l = m.attrs("content");
                    m.attrs("content", null);
                    l&&!j ? (m.updateContent(l), e && clearTimeout(e), a && a.abort &&
                    a.abort()) : (e = c.delay(function() {
                        !m.attrs("content") && m.attrs("currentDataStrategy") === "ajax" && k.showSpinner(m)
                    }, i ? 0 : 100), a = c.ajax(n, {
                        type: "GET",
                        timeout: g,
                        cache: b,
                        success: function(a) {
                            !m.attrs("content") && m.attrs("currentDataStrategy") === "ajax" && (clearTimeout(e), k.trigger("ajaxSuccess", m), m.updateContent(a), m.isActive() && m.updatePosition(), k.trigger("ajaxContentLoaded", m))
                        },
                        error: function() {
                            !m.attrs("content") && m.attrs("currentDataStrategy") === "ajax" && (clearTimeout(e), k.trigger("ajaxFail", m), m.updateContent(f),
                            m.isActive() && m.updatePosition())
                        }
                    }), m.attrs({
                        spinnerTimer: e,
                        ajaxHandler: a
                    }));
                    return this
                },
                unloadContent: function(c) {
                    k.clearContent(c);
                    return this
                },
                shouldRefreshContent: function(c) {
                    return !c.attrs("manualRefresh")
                },
                isValidStrategy: function(c) {
                    return !!c.url
                }
            }
        });
        p.when("A", "a-popover-util").register("a-popover-inline-strategy", function(c, k) {
            return {
                name: "inline",
                reusePopover: !1,
                loadContent: function(c) {
                    var i = c.attrs("content");
                    i && c.attrs("content", null);
                    if (!i)
                        var i = c.$trigger, m = i.data("action"), i = i.data(m) ||
                        {}, i = i.inlineContent ? i.inlineContent: null;
                    i || (i = c.attrs("inlineContent"));
                    c.updateContent(i);
                    return this
                },
                unloadContent: function(c) {
                    var i = c.getContent(), i = i && i.length > 0 ? i.html(): c.attrs("inlineContent"), m = c.$trigger, j = m.data("action"), n = m.data(j) || {};
                    n.inlineContent = i;
                    m.data(j, n);
                    k.clearContent(c);
                    return this
                },
                shouldRefreshContent: function(c) {
                    return c.isDirty()
                },
                isValidStrategy: function() {
                    return !0
                }
            }
        });
        p.when("A", "a-popover-util").register("a-popover-preload-strategy", function(c, k) {
            var o = c.$;
            return {
                name: "preload",
                reusePopover: !0,
                loadContent: function(c) {
                    var m = c.attrs("name"), j = c.attrs("content");
                    c.attrs("content", null);
                    var n;
                    n = o("#a-popover-" + m);
                    n.detach();
                    if (n.length) {
                        n = n[0];
                        for (var g = document.createDocumentFragment(); n.firstChild;)
                            g.appendChild(n.firstChild);
                        n = g
                    } else 
                        n=!1;
                    j ? c.updateContent(j) : m && c.updateContent(n);
                    return this
                },
                unloadContent: function(i) {
                    var m = i.attrs("name");
                    if (m) {
                        var j = i.getContent();
                        if (j && j.html()) {
                            var m = "a-popover-" + m, n = o("#" + m);
                            n.length ? n = n[0] : (n = document.createElement("div"), n.id = m, n.className =
                            "a-popover-preload", document.body.appendChild(n));
                            m = n;
                            if (!c.trim(m.innerHTML))
                                if (j = j[0], typeof j === "string")
                                    o(m).html(j);
                                else {
                                    for (n = document.createDocumentFragment(); j.firstChild;)
                                        n.appendChild(j.firstChild);
                                        m.appendChild(n)
                                    }
                            k.clearContent(i)
                        }
                    }
                    return this
                },
                shouldRefreshContent: function(c) {
                    var a;
                    a = (c = c.attrs("name")) ? o("#a-popover-" + c) : null, c = a;
                    return !(!c ||!(c.length && c.html() !== ""))
                },
                isValidStrategy: function(c) {
                    return !c.name?!1 : c.currentDataStrategy === "preload"?!0 : !!o("#a-popover-" + c.name).length
                }
            }
        });
        p.when("A").register("a-dropdown-base-positions", function(c) {
            return {
                positionStrategy: function(k) {
                    var o = k.$popover, i = k.$trigger, m = k.measure, k = o.find(".a-popover-inner");
                    k.css({
                        "min-width": "0px",
                        width: "auto",
                        height: "auto"
                    });
                    var j = i.closest(".a-button-dropdown");
                    j.length || (j = i.closest(".a-button-group"));
                    var i = m(o, j), m = i.windowWidth - (i.triggerLeft + i.popoverWidth), j = i.windowWidth - i.triggerLeft - i.triggerWidth, n = i.triggerLeft, g = {}, f = i.triggerTop - i.windowTop, b = i.windowBottom - i.triggerBottom, e = o.find(".a-popover-inner");
                    f > b && b < i.popoverHeight ? (b = f, g.top = f < i.popoverHeight ? i.triggerBottom - f : i.triggerBottom - i.popoverHeight) : g.top = i.triggerTop;
                    e.css("height", b < i.popoverHeight ? b - i.headerHeight + "px" : "auto");
                    i.popoverHeight > b ? e.addClass("a-lgtbox-vertical-scroll") : e.removeClass("a-lgtbox-vertical-scroll");
                    k.hasClass("a-lgtbox-vertical-scroll") && navigator.appVersion.indexOf("Windows")>-1 ? (f = Math.max(i.popoverWidth, i.triggerWidth) + c.scrollBarWidth(!0), k.width(f)) : k.css("min-width", i.triggerWidth + "px");
                    i.popoverWidth = o.width();
                    g.left = m < 50 && n > j ? i.triggerRight - i.popoverWidth : i.triggerLeft;
                    return g
                }
            }
        });
        p.when("A", "a-dropdown-base-positions").register("a-dropdown-base-variant-base", function(c, k) {
            return c.extend(k, {
                updateContent: function(c) {
                    typeof c === "string" ? this.$popover.find(".a-popover-inner").html(c) : c && this.$popover.find(".a-popover-inner").html("").append(c)
                },
                beforeShowMethod: function() {
                    var c = this.id + "_dropdown_combobox";
                    this.parent.lock(1);
                    this.$trigger.attr("aria-pressed", !0).attr("aria-owns", c);
                    this.$popover.find("ol,ul").eq(0).attr("id",
                    c)
                },
                afterShowMethod: function() {
                    var k = this.$popover, i = k.find(".a-active");
                    c.delay(function() {
                        i.length ? i.closest("li").focus() : i = k.find("li").first().focus()
                    }, 0)
                },
                beforeHideMethod: function() {
                    this.parent.unlock(1)
                },
                afterHideMethod: function() {
                    this.$trigger.attr("aria-pressed", !1).focus();
                    this.$popover.css("width", "auto")
                }
            })
        });
        p.when("A", "a-dropdown-base-variant-base").register("a-dropdown-base-variant", function(c, k) {
            return k
        });
        p.when("A", "a-popover-base-factory", "a-dropdown-base-variant").register("a-dropdown-base-factory",
        function(c, k, o) {
            function i(f, b, e) {
                var a = ['<li tabIndex="0" role="option" aria-selected="true"'], g = f.data("aCssClass"), j = f.data("aId"), i = f.data("aHtmlContent"), m = f.data("aImageSource"), k = ['<a tabIndex="-1" href="javascript:void(0)" data-value="{&quot;stringVal&quot;:&quot;', f.val().replace(/"/g, "\\&quot;"), '&quot;}"'], o = ["a-dropdown-link"], t = ["a-dropdown-item"];
                b && (o.push("a-active"), a.push(' aria-checked="true"'));
                n && (c.capabilities.mobile || c.capabilities.tablet) && o.push("a-list-link-after-group");
                n=!1;
                g && t.push(g);
                j && a.push(' id="' + j + '"');
                a.push('aria-labelledby="');
                a.push(e);
                a.push('"');
                k.push(' id="');
                k.push(e);
                k.push('"');
                a.push(' class="' + t.join(" ") + '"');
                a.push(">");
                i ? b = i : (b = [], m && (o.push("a-option-has-image"), b.push('<img src="' + m + '" class="a-rich-option-image" />')), b.push(f.html()), b = b.join(""));
                k.push(' class="');
                k.push(o.join(" "));
                k.push('">');
                k.push(b);
                k.push("</a>");
                a.push(k.join(""));
                a.push("</li>");
                return a.join("")
            }
            function m(f) {
                f.jquery || (f = j(f));
                var b = f.children("optgroup,option:not(.a-prompt)"),
                e=!1, a = f[0], c = f.attr("id") ? f.attr("id") : "dropdown" + g++, m, k;
                if (a.selectedIndex>-1)
                    m = a.options[a.selectedIndex].value;
                k = ['<ul tabIndex="-1" class="a-nostyle a-list-link', f.data("a-has-images") ? " a-box-list": "", '" role="listbox" aria-multiselectable="false">'];
                var o = 0;
                b.each(function() {
                    var a = j(this);
                    a.is("optgroup") ? (a.children().each(function() {
                        k.push(i(j(this), m === this.value, c + "_" + o++))
                    }), k.push('<li tabIndex="-1" class="divider"><hr /></li>'), e = n=!0) : (k.push(i(a, m === this.value, c + "_" + o++)), e=!1)
                });
                e && k.pop();
                k.push("</ul>");
                return k.join("")
            }
            var j = c.$, n=!1, g = 1;
            return c.extend({
                create: k.create,
                remove: k.remove,
                get: k.get
            }, {
                type: "dropdown",
                create: function(f, b, e) {
                    var a = b.$button, g = b.$sourceSelect, j = g[0], n = a.find(".a-dropdown-label"), i = g.data("aTouchHeader");
                    if (!i ||!i.length && n.length)
                        i = n.text();
                    return k.create(f, {
                        attributes: {
                            type: "dropdown",
                            header: i,
                            closeButtonLabel: b.closeButtonLabel ? b.closeButtonLabel: "Close",
                            inlineContent: g,
                            position: b.position,
                            alone: !0,
                            sourceSelect: g,
                            sourceButton: a,
                            name: g[0].name,
                            preventNameReuse: !0,
                            lightboxOptions: c.capabilities.mobile || c.capabilities.tablet ? {
                                showDuration: c.capabilities.ios ? null: 0,
                                hideDuration: 0
                            }
                            : null
                        },
                        variant: c.extend({}, o, e, {
                            skin: function(a) {
                                var b = e.subskin ? e.subskin(j): m(j);
                                a.attrs("inlineContent", b);
                                return e.skin(a)
                            }
                        }),
                        actionCheck: !1
                    })
                }
            })
        });
        p.when("A").register("a-dropdown-base-handlers", function() {});
        p.when("A", "a-dropdown-select-apis", "a-dropdown-base-factory", "a-dropdown-base-handlers", "a-popover-base").register("a-dropdown-base", function(c, k, o) {
            function i(f,
            b, e) {
                try {
                    f.$event.preventDefault()
                } catch (a) {}
                c.delay(function() {
                    var a = b.$button ? b.$button: b.getButtonFromEvent(f), i = b.$select ? b.$select: b.getSelectFromEvent(f);
                    if (!a.hasClass("a-button-disabled")) {
                        n(i, b).isSynced() || j(g.extend({
                            $button: a,
                            $select: i
                        }, b));
                        var m = f.$declarativeParent, i = c.extend({}, b, {
                            $button: a,
                            $sourceSelect: i
                        }), k = o.create(m, i, e);
                        if (k && (k.show(), a.data("a-popover-id", k.id).data("popover", k).data("isPressed", !0), !k.hasOnLoad)) {
                            k.hasOnLoad=!0;
                            var p = [], a = k.$popover.find("img");
                            a.length && (a.each(function(a,
                            b) {
                                if (!b.complete ||!b.naturalWidth) {
                                    var e = g.Deferred();
                                    p.push(e);
                                    g(b).bind("load error", function() {
                                        e.resolve()
                                    })
                                }
                            }), p.length ? g.when.apply(g, p).done(function() {
                                k.updatePosition()
                            }) : k.updatePosition())
                        }
                    }
                })
            }
            function m(f) {
                var b = f.$button, f = f.$select;
                b || (b = f.nextAll(".a-button-dropdown"));
                return f.length ? ((b = o.get(b)) && b.hide(), !0) : !1
            }
            function j(f) {
                var b = f.$button, f = f.$select;
                b || (b = f.nextAll(".a-button-dropdown"));
                return f.length ? ((b = o.get(b)) && o.remove(b.id), f.data("a-info", null), !0) : !1
            }
            function n(f,
            b) {
                var e, a;
                e = b.$select ? b.$select : typeof f === "string" ? g("select#" + f) : f.jquery ? f : g(f);
                if (!e.length)
                    return null;
                a = b.$button ? b.$button : b.getButtonFromSelect(e);
                e.data("a-select") ? a = e.data("a-select") : (a = c.extend({
                    hidePopover: m,
                    refreshPopover: j,
                    options: c.extend({
                        $select: e,
                        $button: a
                    }, b)
                }, k), e.data("a-select", a));
                return a
            }
            var g = c.$;
            return {
                toggleDropdown: function(f, b) {
                    var e = (b.$button ? b.$button : b.getButtonFromEvent(f)).data("popover");
                    e && e.$popover.is(":visible") ? e.hide() : i(f, b)
                },
                showDropdown: i,
                getSelect: n
            }
        });
        p.when("A", "jQuery").register("a-dropdown-options-apis", function() {
            return {
                update: function(c) {
                    typeof c !== "object" && p.error("input of options.update() function must be a hash");
                    this.hidePopover(this.options);
                    for (var k = 0, o = this.size(); k < o; k++) {
                        var i = this.options.elements[k], m = i[0];
                        c.value && i.val(c.value);
                        c.selected !== void 0 && (!m.selected && c.selected ? this.options.$select.val(m.value) : m.selected&&!c.selected && this.options.$select.val(""));
                        c.html_content && i.data("a-html-content", c.html_content);
                        c.image_source &&
                        i.data("a-image-source", c.image_source);
                        if (c.native_css_class)
                            m.className = c.native_css_class;
                        c.css_class && i.data("a-css-class", c.css_class);
                        if (c.native_id)
                            m.id = c.native_id;
                        c.id && i.data("a-id", c.id);
                        c.text && (i.text(c.text), m.selected && this.setSelectValue(m.value))
                    }
                    this.refreshPopover(this.options);
                    return this
                },
                remove: function() {
                    this.hidePopover(this.options);
                    for (var c = 0, k = this.size(); c < k; c++) {
                        var o = this.options.elements[c];
                        o.is(":selected") && this.setSelectValue("");
                        o.remove()
                    }
                    this.refreshPopover(this.options);
                    return !0
                },
                info: function() {
                    for (var c = [], k = 0, o = this.size(); k < o; k++) {
                        var i = this.options.elements[k];
                        c.push({
                            value: i[0].value,
                            text: i.text(),
                            selected: i[0].selected,
                            html_content: i.data("a-html-content"),
                            image_source: i.data("a-image-source"),
                            native_css_class: i[0].className,
                            css_class: i.data("a-css-class"),
                            native_id: i[0].id,
                            id: i.data("a-id")
                        })
                    }
                    return c
                },
                size: function() {
                    return this.options.elements.length
                }
            }
        });
        p.when("A", "jQuery", "a-dropdown-options-apis").register("a-dropdown-select-apis", function(c, k, o) {
            function i(c) {
                var g =
                this.options.$select, f = this.options.$button, b = g[0];
                typeof c === "number" && (c = c.toString());
                for (var e = 0, a = b.options.length; e < a; e++)
                    if (b.options[e].value === c)
                        break;
                e === a && c === "" && (e = 0);
                e < a && (f.find(".a-dropdown-prompt").html(b.options[e].innerHTML), f.css("min-width", e / b.options.length + "%"), g.val() !== c && (g.val(c), g.trigger("change", [j, !0])));
                return this
            }
            function m(c) {
                return c === j ? this.options.$select.val() : (this.setValue = i, this.setValue(c))
            }
            var j;
            return {
                isSynced: function() {
                    var j = this.options.$select, g =
                    j.data("a-info"), f = this.getOptions().info();
                    j.data("a-info", f);
                    return g ? c.equals(g, f) : !0
                },
                update: function(c) {
                    typeof c !== "object" && p.error("input of select.update() function must be a hash");
                    this.hidePopover(this.options);
                    var g = {
                        none: !0,
                        micro: !0,
                        mini: !0,
                        small: !0,
                        base: !0,
                        medium: !0,
                        large: !0,
                        "extra-large": !0,
                        "double-large": !0,
                        block: !0
                    }, f = this.options.$select, b = f[0], e = this.options.$button, a = e[0], l = f.siblings("label");
                    if (c.name)
                        b.name = c.name;
                    if (c.option_prompt) {
                        var i = f.find(".a-prompt");
                        i.length ? (i.text(c.option_prompt),
                        i.prop("selected") && e.find(".a-dropdown-prompt").text(c.option_prompt)) : (f.prepend(k("<option class='a-prompt' />").text(c.option_prompt)), e.find(".a-dropdown-prompt").text(c.option_prompt))
                    }
                    c.has_images !== j && f.data("a-has-images", !!c.has_images);
                    c.button_size !== j && e.length && (c.button_size === "small" ? e.addClass("a-button-small") : e.removeClass("a-button-small"));
                    if (c.spacing !== j && g.hasOwnProperty(c.spacing))
                        g = /\ba-spacing-[a-z]+\b/g, b.className = b.className.replace(g, ""), a.className = a.className.replace(g,
                        ""), f.addClass("a-spacing-" + c.spacing), e.addClass("a-spacing-" + c.spacing);
                    if (c.grid_units !== j)
                        g = /\ba-button-span\d{1,2}\b/g, b.className = b.className.replace(g, ""), a.className = a.className.replace(g, ""), isFinite(c.grid_units) && c.grid_units > 0 && c.grid_units < 13 && (f.addClass("a-button-span" + c.grid_units), e.addClass("a-button-span" + c.grid_units));
                    c.width_name && (c.width_named === "base" ? e.addClass("a-button-width-normal") : e.removeClass("a-button-width-normal"));
                    if (c.status)
                        c.status === "disabled" ? (b.disabled =
                        !0, e.addClass("a-button-disabled")) : (b.disabled=!1, e.removeClass("a-button-disabled"));
                    if (c.native_id && (b.id = c.native_id, l.length))
                        l[0].htmlFor = c.native_id;
                    if (c.id)
                        a.id = c.id;
                    c.native_css_class && ((a = f.data("a-native-class")) && f.removeClass(a), f.addClass(c.native_css_class).data("a-native-class", c.native_css_class));
                    c.css_class && ((a = e.data("a-class")) && e.removeClass(a), e.addClass(c.css_class).data("a-class", c.css_class));
                    c.label_text !== j && (c.label_text === "" ? (e.find(".a-dropdown-label").remove(), f.siblings("label").remove()) :
                    (a = e.find(".a-dropdown-label"), a.length ? a.text(c.label_text) : e.find(".a-dropdown-prompt").before(k("<span class='a-dropdown-label' />").text(c.label_text)), l.length ? l.text(c.label_text) : f.before(k("<label for='" + b.id + "' class='a-native-dropdown' />").text(c.label_text))), e.css("min-width", c.label_text === "" ? "0.1%" : "0%"));
                    this.refreshPopover(this.options);
                    return this
                },
                setValue: i,
                val: m,
                getOptions: function(i) {
                    for (var g = [], f = this.options.$select, b = [], g = i === j ? f.children("optgroup, option:not(.a-prompt)") :
                    k.isArray(i) ? i : [i], i = 0, e = g.length; i < e; i++) {
                        var a = g[i], l = [];
                        c.isFiniteNumber(a) ? l = f.children("optgroup, option:not(.a-prompt)").eq(a) : typeof a === "string" ? l = f.children("option#" + a) : typeof a === "object" && (l = a.jquery ? a : k(a));
                        l.length && b.push(l)
                    }
                    return c.extend({
                        hidePopover: this.hidePopover,
                        refreshPopover: this.refreshPopover,
                        setSelectValue: m,
                        options: c.extend({
                            elements: b
                        }, this.options)
                    }, o)
                },
                getOption: function(c) {
                    return this.getOptions(c)
                },
                addOptions: function(c, g) {
                    k.isArray(c) || (c = [c]);
                    for (var f = c.length; f--;)
                        this.addOption(c[f],
                        g);
                    return this
                },
                addOption: function(c, g) {
                    var f = this.options.$select;
                    if (!c.native_id ||!f.find("option#" + c.native_id).length) {
                        var b = f.children("optgroup, option:not(.a-prompt)"), e = document.createElement("option"), a = g && g > 0 && g <= b.length ? g: 0;
                        if (c.native_id)
                            e.id = c.native_id;
                        b.length === 0 || a === b.length ? f[0].appendChild(e) : b.eq(a).before(e);
                        this.getOption(e).update(c)
                    }
                    return this
                },
                removeOptions: function(c) {
                    this.getOptions(c).remove();
                    return this
                },
                removeOption: function(c) {
                    return this.removeOptions(c)
                },
                appendOption: function(c) {
                    return this.addOption(c,
                    this.options.$select.children("optgroup, option:not(.a-prompt)").length)
                },
                appendOptions: function(c) {
                    if (k.isArray(c))
                        for (var g = 0, f = c.length; g < f; g++)
                            this.addOption(c[g]);
                    return this
                }
            }
        });
        p.when("A", "a-dropdown-options", "a-dropdown-apis", "a-dropdown-handlers").register("a-dropdown", function(c, k, o) {
            var i = c.$;
            i(document).delegate(".a-native-dropdown", "change", function(m, j, n) {
                var g = k.getButtonFromEvent(m), f = "", b = this.options.length, e = this.selectedIndex>-1 ? this.options[this.selectedIndex].value: "", a = g.data("popover"),
                m=!1, l;
                if (g.length) {
                    for (g = g.eq(0); b--;)
                        if (l = this.options[b], l.value === e) {
                            f = l.innerHTML;
                            break
                        }
                    a && a.$popover && (a.$popover.find(".a-active").removeClass("a-active").closest("li").attr("aria-checked", !1), j === void 0 && (j = a.$popover.find('a[data-value="' + ('{"stringVal":"' + e + '"}') + '"]')));
                    j && j.length && (m=!0, j.addClass("a-active").closest("li").attr("aria-checked", !0));
                    g.find(".a-dropdown-prompt").html(f);
                    g.css("min-width", this.selectedIndex / this.options.length + "%");
                    a && (a.hide(), (g = o.getSelect(this)) && i(this).data("a-info",
                    g.getOptions().info()));
                    if (!n)
                        n = this.name, g = this.id, j = {
                            auiItemNode: m ? j[0]: null,
                            nativeItemNode: this.options[this.selectedIndex],
                            selectNode: this,
                            id: g,
                            name: n,
                            value: this.value
                        }, n && n !== "" && (c.trigger("a:dropdown:" + n + ":select", j), c.trigger("a:dropdown:selected:" + n, j)), g && g !== "" && c.trigger("a:dropdown:" + g + ":select", j), c.trigger("a:dropdown:select", j)
                }
            }).delegate(".a-button-dropdown:not(.a-button-disabled)", "focusin", function() {
                i(this).find(".a-button-text").focus()
            });
            return o
        });
        p.when("A", "a-dropdown-base",
        "a-dropdown-options").register("a-dropdown-apis", function(c, k, o) {
            function i(c) {
                return k.getSelect(c, o)
            }
            var m = c.$;
            c.on("beforeReady", function() {
                m(".a-dropdown-container select").each(function() {
                    var c = i(this);
                    c && c.val(c.val())
                })
            });
            return {
                getSelect: i,
                updateOption: function(c, n) {
                    var g = m("option#" + c).closest("select");
                    i(g).getOption(c).update(n)
                },
                updateSelect: function(c, m) {
                    i(c).update(m)
                },
                setValue: function(c, m) {
                    i(c).setValue(m)
                }
            }
        });
        p.when("A", "a-popover-accessibility").register("a-dropdown-variant", function(c,
        k) {
            return {
                skin: function(c) {
                    var i = c.attrs("header") || "", c = c.id;
                    return ['<div class="a-popover a-dropdown a-dropdown-common">', k.getStartAnchorHtml(c, "", i, !0), '<div class="a-popover-wrapper"><div class="a-popover-inner"></div></div>', k.getEndAnchorHtml(c, "", i, !0), "</div>"].join("")
                }
            }
        });
        p.when("A", "a-dropdown-base", "a-dropdown-variant", "a-dropdown-options", "a-dropdown-apis", "a-dropdown-base-factory").register("a-dropdown-handlers", function(c, k, o, i, m, j) {
            function n(f) {
                f.preventDefault();
                var f = g(this), b =
                j.get(f.closest(".a-popover"));
                if (f.hasClass("a-active"))
                    b.hide();
                else {
                    var e = f.data("value").stringVal;
                    b.sourceSelect.val(e).trigger("change", [f])
                }
            }
            var g = c.$;
            c.declarative("a-dropdown-button", "click", function(f) {
                var b = i.getButtonFromEvent(f);
                k.showDropdown(f, c.extend({
                    $button: b
                }, i), o)
            });
            c.declarative("a-dropdown-button", "keydown", function(f) {
                var b = i.getButtonFromEvent(f), e = c.constants.keycodes, a = f.$event.which;
                (a === e.DOWN_ARROW || a === e.ENTER || a === e.SPACE) && k.showDropdown(f, c.extend({
                    $button: b
                }, i),
                o)
            });
            g(document).delegate(".a-popover.a-dropdown a", "click", n).delegate(".a-dropdown li", "keydown", function(f) {
                var b = g(this), e = c.constants.keycodes;
                switch (f.which) {
                case e.UP_ARROW:
                    f.preventDefault();
                    b.prev().focus();
                    break;
                case e.DOWN_ARROW:
                    f.preventDefault();
                    b.next().focus();
                    break;
                case e.ENTER:
                case e.SPACE:
                    n.call(b.find("a")[0], f);
                    break;
                case e.ESCAPE:
                    f.preventDefault(), f = j.get(b.closest(".a-popover")), f.sourceButton.find(".a-button-text").focus(), f.hide()
                }
            })
        });
        p.when("A").register("a-dropdown-options",
        function(c) {
            function k(c) {
                return c.popover ? c.popover.$trigger.closest(".a-button-dropdown") : c.$target ? c.$target.closest(".a-button-dropdown") : o(c.target).nextAll(".a-button-dropdown")
            }
            var o = c.$;
            return {
                getButtonFromEvent: k,
                getButtonFromSelect: function(c) {
                    return c.nextAll(".a-button-dropdown")
                },
                getSelectFromEvent: function(c) {
                    c = k(c).prevAll("select").eq(0);
                    c.length || p.error("Cannot locate the <select> of dropdown");
                    return c
                },
                triggerSelector: ".a-button-dropdown"
            }
        });
        p.when("A", "a-popover-accessibility").register("a-dropdown-split-variant",
        function(c, k) {
            return {
                skin: function(c) {
                    var i = c.attrs("header") || "", c = c.id;
                    return ['<div class="a-popover a-splitdropdown a-dropdown-common">', k.getStartAnchorHtml(c, "", i, !0), '<div class="a-popover-wrapper">\n<div class="a-popover-inner"></div>\n</div>', k.getEndAnchorHtml(c, "", i, !0), "</div>"].join("\n")
                }
            }
        });
        p.when("A", "a-dropdown-base", "a-dropdown-split-utils", "a-dropdown-split-variant", "a-dropdown-split-options", "a-dropdown-base-factory").register("a-dropdown-split-handlers", function(c, k, o, i, m, j) {
            var n =
            c.$;
            c.declarative("a-splitdropdown-button", "click", function(g) {
                var f = m.getButtonFromEvent(g);
                k.showDropdown(g, c.extend({
                    $button: f
                }, m), i)
            });
            c.declarative("a-splitdropdown-main", "click", function(c) {
                var f = c.$target.closest(".a-splitdropdown-container").find("select"), b = f.attr("id"), e = f.val();
                o.triggerEvent(b, f, e);
                c.$event.preventDefault()
            });
            c.declarative("a-splitdropdown-button", "keydown", function(g) {
                var f = m.getButtonFromEvent(g), b = c.constants.keycodes, e = g.$event.which;
                (e === b.DOWN_ARROW || e === b.ENTER ||
                e === b.SPACE) && k.showDropdown(g, n.extend({
                    $button: f
                }, m), i)
            });
            n(document).delegate(".a-popover.a-splitdropdown a", "click", function(c) {
                var f = n(this), b = f.data("value").stringVal, f = j.get(f.closest(".a-popover")), e = f.sourceSelect, a = e.attr("id");
                f.hide();
                o.triggerEvent(a, e, b);
                c.preventDefault()
            }).delegate(".a-splitdropdown li", "keydown", function(g) {
                var f = n(this), b = c.constants.keycodes;
                switch (g.which) {
                case b.UP_ARROW:
                    g.preventDefault();
                    f.prev().focus();
                    break;
                case b.DOWN_ARROW:
                    g.preventDefault();
                    f.next().focus();
                    break;
                case b.ENTER:
                case b.SPACE:
                    g.preventDefault();
                    f.find("a").trigger("click");
                    break;
                case b.ESCAPE:
                    g.preventDefault(), g = j.get(f.closest(".a-popover")), g.sourceButton.find(".a-button-text").focus(), g.hide()
                }
            })
        });
        p.when("A").register("a-dropdown-split-options", function(c) {
            function k(c) {
                return c.popover ? c.popover.$trigger.closest(".a-button-splitdropdown") : c.$target ? c.$target.closest(".a-button-splitdropdown") : o(c.target).nextAll(".a-button-splitdropdown")
            }
            var o = c.$;
            return {
                getButtonFromEvent: k,
                getButtonFromSelect: function(c) {
                    return c.next(".a-button-group-splitdropdown").find(".a-button-splitdropdown")
                },
                getSelectFromEvent: function(c) {
                    c = k(c).closest(".a-splitdropdown-container").find("select");
                    c.length || p.error("cannot locate the <select> of the split dropdown");
                    return c
                }
            }
        });
        p.when("A").register("a-dropdown-split-utils", function(c) {
            return {
                triggerEvent: function(k, o, i) {
                    o = {
                        $select: o,
                        value: i,
                        id: k
                    };
                    c.trigger("a:splitdropdown:" + k + ":select", o);
                    c.trigger("a:splitdropdown:select", o)
                }
            }
        });
        p.when("A", "a-dropdown-base", "a-dropdown-split-options", "a-dropdown-split-utils", "a-dropdown-split-handlers").register("a-splitdropdown",
        function(c, k, o, i) {
            var m = c.$;
            m(document).delegate(".a-native-splitdropdown", "change", function(c, n, g) {
                var c = m(this), n = c.val(), f = c.attr("id");
                g || i.triggerEvent(f, c, n)
            }).delegate(".a-button-splitdropdown:not(.a-button-disabled)", "focusin", function() {
                m(this).find(".a-button-text").focus()
            });
            return {
                getSelect: function(c) {
                    return k.getSelect(c, o)
                }
            }
        });
        p.when("A").register("a-popover-accessibility", function() {
            var c = function(c, o, i, m, j) {
                if (!o)
                    return "";
                var n = {
                    "{{ANCHOR_NAME}}": "a-popover-" + c,
                    "{{AIRA_LABEL}}": m ?
                    'aria-label="' + m + '"': 'aria-labelledby="a-popover-' + (i ? "header" : "content") + "-" + o + '"'
                };
                return (j ? '<span role="dialog" data-action="a-popover-a11y" class="{{ANCHOR_NAME}} a-offscreen a-declarative" tabindex="0" {{AIRA_LABEL}}></span>' : '<span role="dialog" class="{{ANCHOR_NAME}} a-offscreen" tabindex="0" {{AIRA_LABEL}}></span>').replace(/\{\{[\w_]*\}\}/g, function(c) {
                    return n[c]
                })
            };
            return {
                getStartAnchorHtml: function() {
                    [].splice.call(arguments, 0, 0, "start");
                    return c.apply(null, arguments)
                },
                getEndAnchorHtml: function() {
                    [].splice.call(arguments,
                    0, 0, "end");
                    return c.apply(null, arguments)
                }
            }
        });
        p.when("A", "a-popover-util").register("a-popover-ajax", function(c, k) {
            return {
                update: function(c, i, m) {
                    var j = {};
                    j.url = i;
                    if (m.timeout)
                        j.timeout = m.timeout;
                    if (m.ajaxFailMsg)
                        j.ajaxFailMsg = m.ajaxFailMsg;
                    if (m.cache)
                        j.cache = m.cache;
                    c.update(j)
                },
                showSpinner: function(c) {
                    return k.showSpinner(c)
                }
            }
        });
        p.when("A", "a-popover-util", "a-popover-inline-strategy", "a-popover-preload-strategy", "a-popover-ajax-strategy").register("a-popover-data", function(c, k, o, i, m) {
            var j = [m, i,
            o];
            return {
                guessStrategyByAttrs: function(c) {
                    for (var g = 0, f = j.length; g < f; g++) {
                        var b = j[g];
                        if (b.isValidStrategy(c))
                            return b
                    }
                },
                getStrategyByName: function(c) {
                    for (var g = 0, f = j.length; g < f; g++) {
                        var b = j[g];
                        if (b.name === c)
                            return b
                    }
                    return null
                },
                showSpinner: k.showSpinner
            }
        });
        p.when("A", "ready").register("a-popover-iframes", function(c) {
            function k() {
                if (i) {
                    m++;
                    var c = document.createElement("iframe");
                    c.src = "javascript:false;";
                    c.id = "a-popover-iframe-" + m;
                    c.className = "a-popover-iframe";
                    c.style.cssText = "width: 0; height: 0; border: 0; opacity: 0; filter: alpha(opacity = 0); zoom: 1; display: none; position: absolute;";
                    c.frameborder = "0";
                    c.tabindex = "-1";
                    document.body.appendChild(c);
                    return o(c)
                } else 
                    return o()
            }
            var o = c.$;
            o("body");
            var i = document.documentElement.className.indexOf("a-ie6")>-1, m = 0, j = [], n = [];
            c.on.ready(function() {
                i && j.push(k())
            });
            return {
                get: function() {
                    if (i) {
                        var c = j.length === 0 ? k(): j.shift();
                        n.push(c);
                        return c
                    } else 
                        return o()
                },
                release: function(g) {
                    i && (g.hide(), c.each(n, function(c, b) {
                        g.is(c) && n.splice(1, b)
                    }), j.push(g))
                }
            }
        });
        p.when("A", "a-popover-lightbox-markup", "ready").register("a-popover-lightbox", function(c,
        k) {
            function o(a) {
                a.preventDefault();
                a.stopPropagation();
                a.stopImmediatePropagation();
                return !1
            }
            function i() {
                document.body.addEventListener("click", o, !0);
                q=!0
            }
            function m() {
                document.body.removeEventListener("click", o, !0);
                q=!1
            }
            var j = c.$, n = document.documentElement.className, g = n.indexOf("-ie")>-1, f = n.indexOf("a-lt-ie7")>-1, b = c.capabilities.isIE10Plus && c.capabilities.mobile, e = (c.capabilities.androidVersion + "").indexOf("4.") === 0, a = k.id, l = k.div, q=!1, s = null, p =- 1, w =- 1, y =- 1, t = null;
            j(document).delegate("#" + a,
            "click " + c.action.start + " " + c.action.move, function(a) {
                a.preventDefault()
            });
            c.declarative("a-popover-floating-close", c.capabilities.touch ? "touchend" : "click", function(a) {
                !q && a.$target.data("action") && a.$target.data("action").indexOf("a-popover-floating-close")>-1 && t && t.isActive() && (t.unlock().hide(), a.$event.preventDefault())
            });
            if (c.capabilities.isiOS8)
                c.on("a:popover:afterUpdatePosition", function(b) {
                    var b = b.popover, e = j("#" + a), c = e.length ? e.offset().top: - 1, f = j(window), g, l;
                    b.isActive() && b.attrs("lightboxOptions") &&
                    c && (g = 0, l = setInterval(function() {
                        f.scrollTop(c);
                        ++g > 5 && clearInterval(l)
                    }, 200))
                });
            return {
                show: function(n) {
                    var k = j(window);
                    s || (j("body").append(l), s = j("#" + a));
                    n = n || {};
                    (c.capabilities.touch || c.capabilities.pointerPrefix) && i();
                    n.lockScroll && (w===-1 && (w = k.scrollTop(), y = k.scrollLeft()), c.setCssImportant(j("body"), "margin-right", c.scrollBarWidth() + "px"), b || (g ? j("html, body").css("overflow", "hidden") : j("body").css("overflow", "hidden")));
                    var q = (t = n.popover || null) ? t.$popover.css("z-index") - 2: - 1;
                    q > 0 && (s.css("z-index",
                    q), e && k.width());
                    if (typeof n.showDuration !== "number")
                        n.showDuration = 200;
                    f && (w>-1 && s.css("top", w + "px"), y>-1 && s.css("left", y + "px"));
                    n.showDuration > 0 ? c.fadeIn(s, n.showDuration) : s.css("display", "block");
                    (c.capabilities.touch || c.capabilities.pointerPrefix) && c.delay(m, n.showDuration + 300)
                },
                hide: function(a) {
                    var b = j(window);
                    if (!(p>-1) && s) {
                        a = a || {};
                        (c.capabilities.touch || c.capabilities.pointerPrefix) && i();
                        if (typeof a.hideDuration !== "number")
                            a.hideDuration = 250;
                        a.hideDuration > 0 ? c.fadeOut(s, a.duration, "linear",
                        function() {
                            a.lockScroll && (j("html, body").css("overflow", ""), j("body").css("margin-right", ""), c.delay(function() {
                                w > 0 && (b.scrollTop(w), w =- 1);
                                y > 0 && (b.scrollLeft(y), y =- 1)
                            }, 100));
                            t = null
                        }) : (s.css("display", "none"), a.lockScroll && (j("html, body").css("overflow", ""), j("body").css("margin-right", ""), w > 0 && (b.scrollTop(w), w =- 1)), t = null);
                        s.css({
                            height: "",
                            width: ""
                        });
                        (c.capabilities.touch || c.capabilities.pointerPrefix) && c.delay(m, a.hideDuration + 350)
                    }
                },
                lock: function(a) {
                    a || (a = 10);
                    p < a && (p = a)
                },
                unlock: function(a) {
                    a ||
                    (a = 10);
                    p <= a && (p =- 1)
                },
                LIGHTBOX_ID: a
            }
        });
        p.when("A", "ready").register("a-popover-lightbox-markup", function() {
            return {
                id: "a-popover-lgtbox",
                div: '<div id="a-popover-lgtbox" class="a-declarative" data-action="a-popover-floating-close" />'
            }
        });
        p.when("A", "ready").register("a-popover-navigate", function(c) {
            function k(c) {
                if (typeof c === "string")
                    j=!0, window.location.hash = c;
                return window.location.hash || ""
            }
            var o = c.$, o = o(window), i = [], m = [], j=!1, n = {}, g=!1;
            m.push(k());
            o.bind("hashchange", function(f) {
                f.preventDefault();
                g ? g=!1 : m.push(k());
                m.length >= 32 && m.shift();
                j ? j=!1 : c.trigger("a:popover:navigate", n[m[m.length - 1]])
            });
            c.on("a:popover:navigate", function(c) {
                c ? c.show({
                    preventNavigate: !0
                }) : (c = i.length - 1 >= 0 ? i[i.length - 1] : null) && c.popover.hide({
                    preventNavigate: !0
                })
            });
            c.on("a:popover:show", function(c) {
                i.push(c)
            });
            c.on("a:popover:hide", function() {
                i.pop()
            });
            return {
                forward: function(f) {
                    var b = f.name + "_" + c.now();
                    n["#" + b] = f;
                    k(b)
                },
                back: function() {
                    m.length > 0 && m.pop();
                    g=!0;
                    window.history.back()
                }
            }
        });
        p.when("A").register("a-popover-position",
        function(c) {
            function k(b, e) {
                var a = b.outerHeight(!0), g = b.outerWidth(!0), i = b.offset(), m = e.offset(), k;
                c.viewport().zoom === 1 ? k = {
                    top: 0,
                    left: 0
                } : (n || (n = j('<span id="a-popover-offset-tracker"></span>'), j("body").prepend(n)), k = n.offset());
                var o = b.find(".a-popover-header"), o = o.length ? o.outerHeight(!0): 0;
                if (f) {
                    var p = window.pageYOffset - document.documentElement.scrollTop;
                    i.top -= p;
                    m.top -= p
                }
                i.left -= k.left;
                i.top -= k.top;
                m.left -= k.left;
                m.top -= k.top;
                var t, v;
                t = e[0];
                var p = c.viewport(), z = p.scrollLeft, A = p.scrollTop;
                t.getBoundingClientRect ?
                (v = t.getBoundingClientRect(), t = v.right - v.left, v = v.bottom - v.top) : (t = e.width(), v = e.height());
                return {
                    popoverHeight: a,
                    popoverWidth: g,
                    popoverLeft: i.left,
                    popoverTop: i.top,
                    popoverRight: i.left + g,
                    popoverBottom: i.top + a,
                    popoverVerticalCenter: i.top + a / 2,
                    popoverHorizontalCenter: i.left + g / 2,
                    headerHeight: o,
                    triggerHeight: v,
                    triggerWidth: t + 1,
                    triggerLeft: m.left - 1,
                    triggerTop: m.top - 1,
                    triggerRight: m.left + t + 1,
                    triggerBottom: m.top + v + 1,
                    triggerVerticalCenter: m.top + v / 2,
                    triggerHorizontalCenter: m.left + t / 2,
                    windowHeight: p.height,
                    windowWidth: p.width,
                    windowTop: A,
                    windowLeft: z,
                    windowRight: z + p.width,
                    windowBottom: A + p.height,
                    zoomTop: k.top,
                    zoomLeft: k.left
                }
            }
            function o(b) {
                return b.removeClass("a-arrow-top a-arrow-bottom a-arrow-left a-arrow-right")
            }
            function i(b) {
                var e = {
                    deltaTop: 0
                }, a = 0;
                e.top = b.triggerVerticalCenter - b.popoverHeight / 2;
                if (e.top < b.windowTop + g)
                    a = Math.min(b.windowTop + g, b.triggerTop - g), e.deltaTop = e.top - a, e.top = a;
                else if (e.top + b.popoverHeight > b.windowBottom - g)
                    a = Math.min(g, b.windowBottom - b.triggerBottom + g), e.deltaTop = e.top +
                    b.popoverHeight - (b.windowBottom - a), e.top = b.windowBottom - a - b.popoverHeight;
                return e
            }
            function m(b) {
                var e = {
                    deltaLeft: 0
                }, a = 0;
                e.left = b.triggerHorizontalCenter - b.popoverWidth / 2;
                if (e.left < g)
                    a = Math.min(g, b.triggerLeft - g), e.deltaLeft = e.left - a, e.left = a;
                else if (e.left + b.popoverWidth > b.windowRight - g)
                    a = Math.min(g, b.windowWidth - b.triggerRight + g), e.deltaLeft = e.left + b.popoverWidth - (b.windowRight - a), e.left = b.windowRight - a - b.popoverWidth;
                return e
            }
            var j = c.$;
            j(window);
            var n = null, g = 20, f = c.capabilities.mobile && c.capabilities.isIE10Plus;
            return {
                windowCenter: function(b) {
                    var b = k(b.$popover, b.$trigger), e = {};
                    e.top = (b.windowHeight - b.popoverHeight) / 2;
                    e.left = (b.windowWidth - b.popoverWidth) / 2;
                    if (e.top < 0)
                        e.top = 0;
                    return e
                },
                windowTop: function(b) {
                    var b = k(b.$popover, b.$trigger), e = {
                        top: 0
                    };
                    e.left = b.windowWidth / 2 - b.popoverWidth / 2;
                    return e
                },
                windowFullWidth: function() {
                    return {
                        top: 0,
                        left: 0
                    }
                },
                triggerRight: function(b, e) {
                    var a = b.$popover, c = b.$trigger;
                    e || (e = k(a, c));
                    c = i(e);
                    c.left = e.triggerRight;
                    o(a).addClass("a-arrow-right");
                    a.find(".a-arrow-border").css("top",
                    e.popoverHeight / 2 + c.deltaTop);
                    return c
                },
                triggerLeft: function(b, e) {
                    var a = b.$popover, c = b.$trigger;
                    e || (e = k(a, c));
                    c = i(e);
                    c.left = e.triggerLeft - e.popoverWidth;
                    o(a).addClass("a-arrow-left");
                    a.find(".a-arrow-border").css("top", e.popoverHeight / 2 + c.deltaTop);
                    return c
                },
                triggerTop: function(b, e) {
                    var a = b.$popover, c = b.$trigger;
                    e || (e = k(a, c));
                    c = m(e);
                    c.top = e.triggerTop - e.popoverHeight;
                    o(a).addClass("a-arrow-top");
                    a.find(".a-arrow-border").css("left", e.popoverWidth / 2 + c.deltaLeft);
                    return c
                },
                triggerBottom: function(b,
                e) {
                    var a = b.$popover, c = b.$trigger;
                    e || (e = k(a, c));
                    c = m(e);
                    c.top = e.triggerBottom;
                    o(a).addClass("a-arrow-bottom");
                    a.find(".a-arrow-border").css("left", e.popoverWidth / 2 + c.deltaLeft);
                    return c
                },
                triggerHorizontal: function(b, e) {
                    var a = b.$popover, c = b.$trigger;
                    e || (e = k(a, c));
                    return e.triggerLeft - e.windowLeft > e.popoverWidth + g ? this.triggerLeft(b, e) : this.triggerRight(b, e)
                },
                triggerVertical: function(b, e) {
                    var a = b.$popover, c = b.$trigger, a = e ? e: k(a, c);
                    return a.triggerTop - a.windowTop > a.popoverHeight + g ? this.triggerTop(b, a) :
                    this.triggerBottom(b, a)
                },
                triggerVerticalAlignLeft: function(b, e) {
                    var a = b.$popover, c = b.$trigger;
                    e || (e = k(a, c));
                    var c = {}, f = 0, j = 0, i = e.windowBottom - e.triggerBottom;
                    c.left = e.triggerLeft;
                    c.top = i > e.popoverHeight ? e.triggerBottom + 3 : e.triggerTop - e.popoverHeight - 3;
                    if (c.left < g)
                        j = Math.min(g, e.triggerLeft - g), f = c.left - j, c.left = j;
                    else if (c.left + e.popoverWidth > e.windowRight - g)
                        j = Math.min(g, e.windowWidth - e.triggerRight + g), f = c.left + e.popoverWidth - (e.windowRight - j), c.left = e.windowRight - j - e.popoverWidth;
                    o(a).addClass(i >
                    e.popoverHeight ? "a-arrow-bottom" : "a-arrow-top");
                    a.find(".a-arrow-border").css("left", e.triggerWidth / 2 + f);
                    return c
                },
                customPosition: function(b, e) {
                    return e.call(this, {
                        popover: b,
                        $popover: b.$popover,
                        $trigger: b.$trigger,
                        measure: k
                    })
                }
            }
        });
        p.when("A").register("a-popover-util", function(c) {
            function k(c, j) {
                for (var i = c.children.length; i--;) {
                    var g = k(c.children[i], j);
                    if (g)
                        return g
                }
                if (j(c))
                    return c
            }
            var o = c.$, i = /^-?\d+(?:\.\d+)?$/;
            return {
                trigger: function(i, j) {
                    c.trigger("a:popover:" + i, {
                        popover: j
                    });
                    j.name && c.trigger("a:popover:" +
                    i + ":" + j.name, {
                        popover: j
                    })
                },
                extractDeclarativeParams: function(c, j) {
                    var i = o(c), i = i.hasClass("a-declarative") ? i: i.find(".a-declarative").eq(0), g = "a-" + j, f = i.length ? i.data("action").indexOf(g)!==-1: !1, g = i ? i.data(g): null;
                    return f ? {
                        attributes: g,
                        $trigger: i
                    } : null
                },
                eventOccursWithin: function(c, j) {
                    var i = o(c.target);
                    return i.closest(j.$trigger).length > 0 || i.closest(j.$popover).length > 0
                },
                search: k,
                getCSSHash: function(m) {
                    var j = {};
                    c.each("height,width,max-height,max-width,min-height,min-width".split(","), function(k) {
                        if (m[k]) {
                            var g =
                            m[k];
                            if (c.isFiniteNumber(g) || i.test(g))
                                g += "px";
                            j[k] = g
                        }
                    });
                    j.height&&!j["max-height"] && (j["max-height"] = "none");
                    j.width&&!j["max-width"] && (j["max-width"] = "none");
                    return j
                },
                clearContent: function(c) {
                    (c = c.getContent()) && c.empty()
                },
                showSpinner: function(c) {
                    c.updateContent('<div class="a-popover-loading-wrapper a-text-center"><div class="a-box a-color-base-background a-popover-loading"></div></div>');
                    c.updatePosition();
                    return c
                }
            }
        });
        p.when("A", "a-popover-util").register("a-modal-variant-base", function(c, k) {
            var o =
            c.$, i = o("html").hasClass("a-lt-ie9");
            return {
                updateContent: function(c) {
                    typeof c === "string" ? this.$popover.find(".a-popover-inner").html(c) : c && this.$popover.find(".a-popover-inner").html("").append(c)
                },
                updateDimensions: function() {
                    var c = this.$popover, j = k.getCSSHash(this.attrs());
                    c.css(j);
                    j.height ? c.addClass("a-popover-modal-fixed-height") : c.removeClass("a-popover-modal-fixed-height");
                    this.isActive() && this.updatePosition();
                    return this
                },
                getContent: function() {
                    return this.$popover ? this.$popover.find(".a-popover-inner") :
                    null
                },
                showMethod: function() {
                    var k = this, j = k.$popover;
                    !i && k.attrs("currentDataStrategy") !== "ajax" ? (j.css({
                        opacity: 0,
                        visibility: "visible"
                    }).removeClass("a-popover-hidden"), c.delay(function() {
                        c.animate(j, {
                            opacity: 1
                        }, 500, "linear");
                        k.focus()
                    }, 0)) : c.delay(function() {
                        j.css({
                            visibility: "visible"
                        }).removeClass("a-popover-hidden");
                        k.focus()
                    }, 0)
                },
                hideMethod: function() {
                    var k = this.attrs("originalFocus");
                    i ? this.$popover.hide().css("visibility", "hidden").find(".a-lgtbox-vertical-scroll").removeClass("a-lgtbox-vertical-scroll") :
                    c.fadeOut(this.$popover, 250, "linear");
                    k && k.focus()
                }
            }
        });
        p.when("A", "a-modal-variant-base", "a-modal-positions", "a-popover-accessibility").register("a-modal-variant", function(c, k, o, i) {
            var m = c.$, m = m("html").hasClass("a-lt-ie9");
            return c.extend(k, c.capabilities.touch || c.capabilities.mobile || c.capabilities.tablet || m ? o.innerScroll : o.modalScroll, {
                skin: function(c) {
                    var k = c.attrs("id"), g = c.attrs("header") || "", f = c.attrs("footer"), b = c.attrs("closeButton"), e = c.attrs("closeButtonLabel") ? c.attrs("closeButtonLabel"):
                    "", c = c.attrs("popoverLabel") ? c.attrs("popoverLabel"): "", b = b ? '<button data-action="a-popover-close" class="a-button-close a-declarative" aria-label="' + e + '"><i class="a-icon a-icon-close"></i></button>': "", f = f ? ['<div class="a-popover-footer">', f, "</div>"].join(""): "";
                    return ['<div class="a-modal-scroller a-declarative" data-action="a-popover-floating-close"><div class="a-popover a-popover-modal a-declarative" data-action="a-popover-a11y">', i.getStartAnchorHtml(k, g, c), '<div class="a-popover-wrapper"><div class="a-popover-header">',
                    b, '<h4 class="a-popover-header-content" id="a-popover-header-', k, '">', g, '</h4></div><div class="a-popover-inner" id="a-popover-content-', k, '"></div>', f, "</div>", i.getEndAnchorHtml(k, g, c), "</div></div>"].join("")
                }
            })
        });
        p.when("A", "a-popover-lightbox").register("a-modal-positions", function(c, k) {
            function o(c) {
                var f = c.$popover.closest(".a-modal-scroller");
                f.scrollTop(0).css("visibility", "visible");
                f.bind("scroll", function() {
                    c.updateChildrenPosition()
                })
            }
            function i() {
                var g = c.viewport();
                g.width / g.height >
                2 && c.delay(function() {
                    document.activeElement.scrollIntoView();
                    window.scrollTo(window.pageXOffset, 0)
                }, 0)
            }
            var m = c.$, j = document.documentElement.className.indexOf("a-lt-ie7")>-1, n = c.capabilities.isIE10Plus && c.capabilities.mobile;
            return {
                innerScroll: {
                    positionStrategy: function(g) {
                        var f = g.popover, b = g.$popover, e = g.$trigger, a = b.find(".a-popover-inner").css("height", "auto"), l = b.closest(".a-modal-scroller"), q = {}, o = c.viewport(!0), p = o.height * 0.1, o = o.height * 0.8, w = f.attrs("height"), f = f.attrs("min-height");
                        b.css({
                            height: w ?
                            w: "",
                            "min-height": f ? f: ""
                        });
                        g = g.measure(b, e);
                        q.left = (g.windowWidth - g.popoverWidth) / 2;
                        if (q.left < 0)
                            q.left = p, b.css("padding-right", p);
                        g.popoverHeight > o ? (e = b.find(".a-popover-header").outerHeight() || 0, f = b.find(".a-popover-footer").outerHeight() || 0, a.css({
                            height: o - e - f + "px",
                            "overflow-y": "auto"
                        }), b.css({
                            height: o,
                            "min-height": 0
                        }), q.top = p) : (q.top = (g.windowHeight - g.popoverHeight) / 2, a.css("height", "auto"));
                        q.left += g.zoomLeft;
                        q.top += g.zoomTop;
                        j && l.css("top", document.getElementById(k.LIGHTBOX_ID).style.top);
                        n &&
                        (l.css("top", m(window).scrollTop()), b.removeClass("a-popover-pan-y").addClass("a-popover-pan-x"), b = m(document).height(), a = m(document).width(), m("#" + k.LIGHTBOX_ID).css({
                            height: b,
                            width: a > g.popoverWidth ? a: g.popoverWidth + p
                        }));
                        c.capabilities.isMetroIEGuess && c.capabilities.isIETouchCapable && i();
                        return q
                    },
                    beforeShowMethod: c.constants.NOOP,
                    beforeHideMethod: c.constants.NOOP
                },
                modalScroll: {
                    positionStrategy: function(g) {
                        var f = g.$popover, b = g.$trigger, e = f.closest(".a-modal-scroller"), a = f.find(".a-popover-inner").css("height",
                        "auto");
                        if (f.hasClass("a-popover-modal-fixed-height")) {
                            var l = f.find(".a-popover-footer");
                            a.css("padding-bottom", l.height() + 15)
                        }
                        var a = {}, j = c.viewport(!0), k = j.height, l = k * 0.1;
                        k*=0.8;
                        var j = j.width * 0.8, m = f.height(), n = f.width(), g = g.measure(f, b);
                        a.left = (g.windowWidth - n) / 2;
                        a.top = (g.windowHeight - m) / 2;
                        if (n > j)
                            a.left = l, f.css("padding-right", l);
                        if (m > k)
                            if (e.length)
                                a.top = 0, f.css({
                                    position: "static",
                                    margin: g.zoomTop + l + "px 0 " + l + "px " + (g.zoomLeft + a.left) + "px"
                                }), e.css("padding-bottom", "1px");
                            else {
                                if (n > j)
                                    a.left = l;
                                    if (m >
                                    k)
                                        a.top = l
                            } else 
                                e.length && (f.css({
                                    position: "absolute",
                                    margin: "0px"
                                }), e.css("padding-bottom", "0px"));
                        a.left += g.zoomLeft;
                        a.top += g.zoomTop;
                        c.capabilities.isMetroIEGuess && c.capabilities.isIETouchCapable && i();
                        return a
                    },
                    beforeShowMethod: function() {
                        o(this)
                    },
                    beforeHideMethod: function() {
                        this.$popover.closest(".a-modal-scroller").css("visibility", "hidden").unbind("scroll")
                    }
                }
            }
        });
        p.when("A", "a-popover-base-factory", "a-modal-variant", "a-popover-util").register("a-modal-factory", function(c, k, o, i) {
            function m(c, b) {
                b =
                {
                    type: j,
                    alone: !0,
                    immersive: !0,
                    position: "windowCenter",
                    header: b.header,
                    footer: b.footer,
                    width: b.width,
                    height: b.height,
                    "max-width": b["max-width"],
                    "max-height": b["max-height"],
                    "min-width": b["min-width"],
                    "min-height": b["min-height"],
                    closeButton: b.closeButton === "false" || b.closeButton===!1?!1: !0,
                    timeout: b.timeout,
                    lightboxOptions: {
                        lockScroll: !0,
                        showDuration: n || g ? 0: null
                    }, data : b.data || {}, dataStrategy : b.dataStrategy, url : b.url, manualRefresh : !!b.manualRefresh, ajaxFailMsg : b.ajaxFailMsg, cache : b.cache === "false" ||
                    b.cache===!1?!1 : !0, inlineContent : b.inlineContent ? b.inlineContent : b.content, name : b.name, closeButtonLabel : b.closeButtonLabel ? b.closeButtonLabel : "Close", popoverLabel : b.popoverLabel
                };
                return k.create(c, {
                    attributes: b,
                    variant: o,
                    actionCheck: !0
                })
            }
            var j = "modal", n = document.documentElement.className.indexOf("a-lt-ie9")>-1, g = c.capabilities.mobile && c.capabilities.isIE10Plus;
            return {
                type: j,
                create: m,
                get: function(c) {
                    var b = k.get(c, j);
                    !b && typeof c === "object" && (c = i.extractDeclarativeParams(c, j)) && (b = m(c.$trigger, c.attributes ||
                    {}));
                    return b
                },
                remove: function(c) {
                    return k.remove(c, j)
                }
            }
        });
        p.when("A", "a-popover-base-factory", "a-modal-factory").register("a-modal-handlers", function(c, k, o) {
            var i = c.$;
            c.declarative("a-modal", "click", function(c) {
                o.get(c.$declarativeParent).show();
                c.$event.preventDefault()
            });
            i(document).keydown(function(i) {
                if (i.keyCode === c.constants.keycodes.ESCAPE)
                    for (var i = k.getRoot(), j = i.children.length, n=!1; j--&&!n;) {
                        var g = i.children[j];
                        g.type === "modal" && g.isActive() && (g.unlock().hide(), n=!0)
                    }
            });
            i(document).delegate(".a-modal-scroller",
            "click " + c.action.start + " " + c.action.move, function(c) {
                c.target === this && c.preventDefault()
            })
        });
        p.when("A", "a-modal-factory", "a-popover-base", "a-modal-handlers").register("a-modal", function(c, k) {
            return k
        });
        p.when("A", "a-popover-util").register("a-popover-variant-base", function(c, k) {
            return {
                updateContent: function(c) {
                    typeof c === "string" ? this.$popover.find(".a-popover-content").html(c) : c && this.$popover.find(".a-popover-content").html("").append(c)
                },
                updateDimensions: function() {
                    this.$popover.css(k.getCSSHash(this.attrs()));
                    this.isActive() && this.updatePosition();
                    return this
                },
                getContent: function() {
                    return this.$popover ? this.$popover.find(".a-popover-content") : null
                },
                hideMethod: function() {
                    c.fadeOut(this.$popover, 250, "linear")
                }
            }
        });
        p.when("A", "a-popover-variant-base", "a-popover-util", "a-popover-accessibility").register("a-popover-variant", function(c, k, o, i) {
            return c.extend(k, {
                skin: function(c) {
                    var j = c.attrs("id"), k = c.attrs("header"), g = c.attrs("closeButton"), f = c.attrs("closeButtonLabel") ? c.attrs("closeButtonLabel"): "", c = c.attrs("popoverLabel") ?
                    c.attrs("popoverLabel"): "", g = g ? '<button data-action="a-popover-close" class="a-button-close a-declarative" aria-label="' + f + '"><i class="a-icon a-icon-close"></i></button>': "", f = k && k !== "" ? ['<div class="a-popover-header">', g, '<h4 class="a-popover-header-content" id="a-popover-header-', j, '">', k, "</h4></div>"].join(""): "";
                    return ['<div class="a-popover a-declarative" data-action="a-popover-container a-popover-a11y"><div class="a-popover-wrapper">', i.getStartAnchorHtml(j, k, c), f, '<div class="a-popover-inner">',
                    f === "" ? g: "", '<div class="a-popover-content" id="a-popover-content-', j, '"></div></div>', i.getEndAnchorHtml(j, k, c), '<div class="a-arrow-border"><div class="a-arrow"></div></div></div></div>'].join("")
                }
            })
        });
        p.when("A", "a-popover-base-factory", "a-popover-variant", "a-popover-util").register("a-popover-factory", function(c, k, o, i) {
            function m(c, g) {
                g = {
                    type: j,
                    alone: !0,
                    header: g.header,
                    width: g.width,
                    height: g.height,
                    "max-width": g["max-width"],
                    "max-height": g["max-height"],
                    "min-width": g["min-width"],
                    "min-height": g["min-height"],
                    closeButton: g.closeButton === "false" || g.closeButton===!1?!1: !0,
                    position: g.position || "triggerVertical",
                    activate: g.activate || "onmouseover",
                    timeout: g.timeout,
                    data: g.data || {}, dataStrategy : g.dataStrategy, url : g.url, manualRefresh : !!g.manualRefresh, ajaxFailMsg : g.ajaxFailMsg, cache : g.cache === "false" || g.cache===!1?!1 : !0, inlineContent : g.inlineContent ? g.inlineContent : g.content, name : g.name, closeButtonLabel : g.closeButtonLabel ? g.closeButtonLabel : "Close", popoverLabel : g.popoverLabel
                };
                return k.create(c, {
                    attributes: g,
                    variant: o,
                    actionCheck: !0
                })
            }
            var j = "popover"; return {
                type : j, create : m, get : function(c) {
                    var g = k.get(c, j);
                    !g && typeof c === "object" && (c = i.extractDeclarativeParams(c, j)) && (g = m(c.$trigger, c.attributes || {}));
                    return g
                }, remove: function(c) {
                    return k.remove(c, j)
                }
            }
        });
        p.when("A", "a-popover-factory").register("a-popover-handlers", function(c, k) {
            function o(f) {
                if (f&&!f.destroyTimer)
                    f.destroyTimer = c.delay(function() {
                        f.hide()
                    }, 250)
            }
            function i(c) {
                if (c)
                    clearTimeout(c.destroyTimer), c.destroyTimer = null, clearTimeout(c.parent.destroyTimer),
                    c.parent.destroyTimer = null
            }
            var m = c.$, j = 2, n = 200, j = 2, n = 200;
            c.declarative("a-popover", "click", function(c) {
                var b = k.get(c.$declarativeParent);
                b && b.attrs("activate") === "onclick" && (b.show(), c.$event.preventDefault())
            });
            c.declarative("a-popover", "keydown", function(f) {
                f.$event.keyCode === c.constants.keycodes.ENTER && k.get(f.$declarativeParent).show()
            });
            var g;
            c.declarative("a-popover", "mouseenter", function(f) {
                var b = k.get(f.$declarativeParent);
                b && b.attrs("activate") === "onmouseover" && (i(b), g = c.delay(function() {
                    k.get(f.$declarativeParent).show()
                },
                n))
            });
            c.declarative("a-popover", "mousemove", function(f) {
                c.cursor().speed < j && (f = k.get(f.$declarativeParent)) && f.attrs("activate") === "onmouseover" && f.show()
            });
            c.declarative("a-popover", "mouseleave", function(c) {
                if ((c = k.get(c.$declarativeParent)) && c.attrs("activate") === "onmouseover")
                    c && o(c), g && clearTimeout(g)
            });
            c.declarative("a-popover-container", "mouseenter", function(c) {
                (c = k.get(c.$declarativeParent)) && c.attrs("activate") === "onmouseover" && i(c)
            });
            c.declarative("a-popover-container", "mouseleave", function(f) {
                var b =
                k.get(f.$declarativeParent), e=!0, a = m(f.$event.relatedTarget);
                b && b.attrs("activate") === "onmouseover" && b.isActive() && (c.each(b.children, function(b) {
                    if (a.closest(b.$popover).length)
                        return e=!1
                }), e && (o(b), !b.parent.immersive && a.closest(b.parent.$popover).length === 0 && o(b.parent)))
            })
        });
        p.when("A", "a-popover-factory", "a-popover-base", "a-popover-handlers").register("a-popover", function(c, k) {
            return k
        });
        p.when("A", "a-popover-base-factory", "a-secondary-view-variant", "p-detect", "a-popover-util").register("a-secondary-view-factory",
        function(c, k, o, i, m) {
            function j(e, a) {
                a.disableAnimation = a.disableAnimation || f;
                return k.create(e, {
                    attributes: {
                        type: g,
                        immersive: !0,
                        disableAnimation: b || a.disableAnimation,
                        synchronous: !!(b || a.synchronous && a.synchronous !== "false"),
                        animationLength: a.disableAnimation ? 0: 300,
                        alternateBackground: a.alternateBackground ||!1,
                        hideHeader: b || a.hideHeader ||!1,
                        scrollable: a.scrollable ||!0,
                        header: a.header,
                        backButtonText: a.backButtonText,
                        position: "windowFullWidth",
                        timeout: a.timeout,
                        dataStrategy: a.dataStrategy,
                        inlineContent: a.inlineContent ?
                        a.inlineContent: a.content,
                        url: a.url,
                        manualRefresh: !!a.manualRefresh,
                        name: a.name,
                        cache: a.cache === "false" || a.cache===!1?!1: !0,
                        data: a.data || {},
                        popoverLabel: a.popoverLabel
                    }, variant : o, actionCheck : !0
                })
            }
            var n = c.$, g = "secondary-view", f = navigator.userAgent.match(/Android\s[12]/), b=!1;
            p.when("mash-will-load").execute(function() {
                b=!0
            });
            return {
                type: g,
                create: j,
                get: function(b) {
                    var a = k.get(b, g);
                    if (!a && typeof b === "object") {
                        var c = m.extractDeclarativeParams(b, g);
                        c && (a = j(c.$trigger, c.attributes || {}))
                    }
                    if (a && typeof b ===
                    "object")
                        b = n(b), b = b.hasClass("a-declarative") ? b : b.find(".a-declarative").eq(0), c = "a-" + g, b = b ? b.data(c) : null, a.data = b.data;
                    return a
                },
                remove: function(b) {
                    return k.remove(b, g)
                }
            }
        });
        p.when("A", "a-popover-base", "a-secondary-view-factory", "a-secondary-view-handlers").register("a-secondary-view", function(c, k, o) {
            return o
        });
        p.when("A").register("a-tooltip-variant-base", function(c) {
            return {
                updateContent: function(c) {
                    this.$popover.find(".a-tooltip-inner").html(c)
                },
                getContent: function() {
                    return this.$popover ? this.$popover.find(".a-tooltip-inner") :
                    null
                },
                hideMethod: function() {
                    c.fadeOut(this.$popover, 250, "linear")
                }
            }
        });
        p.when("A", "a-tooltip-variant-base").register("a-tooltip-variant", function(c, k) {
            return c.extend(k, {
                skin: function() {
                    return '<div role="tooltip" class="a-popover a-tooltip a-declarative" data-action="a-popover-close"><div class="a-tooltip-inner"></div><div class="a-arrow-border"><div class="a-arrow"></div></div></div>'
                }
            })
        });
        p.when("A", "a-popover-base-factory", "a-tooltip-variant", "a-popover-util").register("a-tooltip-factory", function(c,
        k, o, i) {
            function m(c, g) {
                g = {
                    type: j,
                    inlineContent: g.inlineContent ? g.inlineContent: g.content,
                    position: g.position || "triggerVertical",
                    activate: g.activate || "onmouseover"
                };
                return k.create(c, {
                    attributes: g,
                    variant: o,
                    actionCheck: !0
                })
            }
            var j = "tooltip";
            return {
                type: j,
                create: m,
                get: function(c) {
                    var g = k.get(c, j);
                    !g && typeof c === "object" && (c = i.extractDeclarativeParams(c, j)) && (g = m(c.$trigger, c.attributes || {}));
                    return g
                },
                remove: function(c) {
                    return k.remove(c, j)
                }
            }
        });
        p.when("A", "a-tooltip-factory").register("a-tooltip-handlers",
        function(c, k) {
            c.declarative("a-tooltip", "click", function(c) {
                var i = k.get(c.$declarativeParent);
                i && i.attrs("activate") === "onclick" && (i.show(), c.$event.preventDefault())
            });
            c.declarative("a-tooltip", "mouseenter", function(c) {
                if ((c = k.get(c.$declarativeParent)) && c.attrs("activate") === "onmouseover")
                    if (c.show(), c.destroyTimer)
                        clearTimeout(c.destroyTimer), c.destroyTimer = null
            });
            c.declarative("a-tooltip", "mouseleave", function(o) {
                var i = k.get(o.$declarativeParent);
                if (i && i.attrs("activate") === "onmouseover")
                    i.destroyTimer =
                    c.delay(function() {
                        i.hide()
                    }, 125)
            })
        });
        p.when("A", "a-tooltip-factory", "a-popover-base", "a-tooltip-handlers").register("a-tooltip", function(c, k) {
            return k
        })
    })
})(function() {
    var p = window.AmazonUIPageJS || P, c = p.attributeErrors;
    return c ? c("AmazonUIPopoverJS") : p
}());
(function(p) {
    p.execute(function() {})
})(function() {
    var p = window.AmazonUIPageJS || P, c = p.attributeErrors;
    return c ? c("AmazonUIPopover") : p
}());
(function(p) {
    p.execute(function() {
        p.when("A").register("a-tabs", function(c) {
            var k = c.$;
            c.declarative("a-tabs", ["click"], function(o) {
                var i = o.$target.closest("li"), m = o.data.name, j = i.data("a-tab-name"), n = i.closest(".a-tab-container").find(".a-box-tab");
                j && (k("li.a-active", i.closest(".a-tabs")).removeClass("a-active"), i.closest("li").addClass("a-active"), n.addClass("a-hidden"), n.filter('[data-a-name="' + j + '"]').removeClass("a-hidden"), i = {
                    $tab: i,
                    tabName: j,
                    tabSetName: m
                }, c.trigger("a:tabs:" + m + ":select", {
                    selectedTab: i
                }),
                c.trigger("a:tabs:" + m + ":" + j + ":select", {
                    selectedTab: i
                }), o.$event.preventDefault())
            })
        });
        p.when("A").register("a-accordion", function(c) {
            var k = "slideDown", o = "slideUp", i = 300;
            if (c.capabilities.mobile || c.capabilities.tablet)
                k = "show", o = "hide", i = 0;
            c.declarative("a-accordion", ["click"], function(m) {
                var j = m.$target.closest(".a-accordion"), n = m.$target.closest(".a-box"), g = j.find(".a-box").not(n), f = j.data("a-accordion-name"), b = n.data("a-accordion-row-name"), j = j.hasClass("a-accordion-collapse");
                if (b) {
                    var e = n.find(".a-accordion-inner"),
                    a=!0;
                    if (n.hasClass("a-accordion-active"))
                        if (j)
                            e[o]({
                                duration: i,
                                complete: function() {
                                    n.removeClass("a-accordion-active").attr("aria-expanded", "false");
                                    n.find(".a-icon.a-accordion-radio").removeClass("a-icon-radio-active").addClass("a-icon-radio-inactive")
                                }
                            });
                        else 
                            a=!1;
                    else 
                        g.find(".a-accordion-inner").attr("aria-expanded", "false")[o]({
                            duration: i,
                            complete: function() {
                                g.removeClass("a-accordion-active")
                            }
                        }), e.attr("aria-expanded", "true")[k]({
                            duration: i,
                            complete: function() {
                                n.addClass("a-accordion-active");
                                g.find(".a-icon.a-accordion-radio").removeClass("a-icon-radio-active").addClass("a-icon-radio-inactive");
                                n.find(".a-icon.a-accordion-radio").removeClass("a-icon-radio-inactive").addClass("a-icon-radio-active")
                            }
                        });
                    a && (j = {
                        $row: n,
                        rowName: b,
                        accordionName: f
                    }, c.trigger("a:accordion:select", {
                        selectedRow: j
                    }), c.trigger("a:accordion:" + f + ":select", {
                        selectedRow: j
                    }), c.trigger("a:accordion:" + f + ":" + b + ":select", {
                        selectedRow: j
                    }))
                }
                m.$event.preventDefault()
            })
        });
        p.when("A", "jQuery").register("a-expander", function(c,
        k) {
            var o, i, m;
            function j(b, e) {
                var a = b.closest("." + o), c = a.data("a-expander-collapsed-height"), f = b.attr("aria-expanded") === "true";
                b.toggleClass(i + "-expanded");
                c ? (a.css("height", f ? c : "auto"), b.attr("aria-expanded", f ? "false" : "true"), e()) : b.toggle(0, function() {
                    b.attr("aria-expanded", f ? "false" : "true");
                    e()
                })
            }
            function n() {
                k(".a-expander-partial-collapse-container").each(function() {
                    var b = k(this), c = b.children("." + i), a = b.data("a-expander-collapsed-height"), f = b.children("." + m);
                    c.height() <= a ? f.css({
                        opacity: "0",
                        display: "none"
                    }) : (f.css({
                        opacity: "1",
                        display: "block"
                    }), c.css("padding-bottom", f.height()), c.attr("aria-expanded") !== "true" && b.css({
                        height: a,
                        "max-height": "none"
                    }))
                })
            }
            var g = {
                inline: {
                    expand: "a-icon-expand",
                    collapse: "a-icon-collapse"
                },
                section: {
                    expand: "a-icon-section-expand",
                    collapse: "a-icon-section-collapse"
                },
                extender: {
                    expand: "a-icon-extender-expand",
                    collapse: "a-icon-extender-collapse"
                }
            };
            o = "a-expander-container";
            i = "a-expander-content";
            m = "a-expander-header";
            var f = {};
            c.each(g, function(b, e) {
                f[e] = {};
                c.each(b,
                function(a, b) {
                    f[e][b] = RegExp("\\b" + a + "\\b", "g")
                })
            });
            c.declarative("a-expander-toggle", "click", function(b) {
                var e = b.$target.closest("." + o), a = e.find("." + o), l = e.data("a-expander-name"), k;
                k = b.$currentTarget.hasClass(m) ? b.$currentTarget : e.find("." + m).not(a.find("." + m));
                var n = e.find("." + i).not(a.find("." + i));
                j(n, function() {
                    var j = k.find(".a-icon")[0], i = null, m = k.children(".a-expander-content-fade");
                    if (n.attr("aria-expanded") === "false") {
                        if (j)
                            j.className = j.className.replace(f.inline.collapse, g.inline.expand).replace(f.section.collapse,
                            g.section.expand).replace(f.extender.collapse, g.extender.expand);
                        b.data && b.data.expand_prompt && (i = b.data.expand_prompt);
                        m.show();
                        j = "collapse"
                    } else {
                        if (j)
                            j.className = j.className.replace(f.inline.expand, g.inline.collapse).replace(f.section.expand, g.section.collapse).replace(f.extender.expand, g.extender.collapse);
                        b.data && b.data.collapse_prompt && (i = b.data.collapse_prompt);
                        m.hide();
                        j = "expand"
                    }
                    i && i !== "" && k.find(".a-expander-prompt").not(a.find(".a-expander-prompt")).html(i);
                    i = {
                        $expander: e,
                        expanderName: l
                    };
                    c.trigger("a:expander:toggle", {
                        expander: i
                    });
                    c.trigger("a:expander:toggle:" + j, {
                        expander: i
                    });
                    l && (c.trigger("a:expander:" + l + ":toggle", {
                        expander: i
                    }), c.trigger("a:expander:" + l + ":toggle:" + j, {
                        expander: i
                    }))
                })
            });
            c.on("ready resize orientationchange a:popover:afterShow a:popover:ajaxContentLoaded", n);
            return {
                initializeExpanders: n
            }
        });
        p.when("A").register("a-form-controls-api", function(c) {
            var k = c.$, o = {
                checkboxes: {
                    "multi-select": {
                        active: "a-icon-touch-multi-select-active",
                        inactive: "a-icon-touch-multi-select"
                    },
                    checkbox: {
                        active: "a-icon-checkbox-active" +
                        (c.capabilities.mobile || c.capabilities.tablet ? " a-icon-touch-checkbox-active" : ""),
                        inactive: "a-icon-checkbox-inactive" + (c.capabilities.mobile || c.capabilities.tablet ? " a-icon-touch-checkbox-inactive" : ""),
                        disabled: "a-icon-checkbox-disabled"
                    }
                },
                radio: {
                    active: "a-icon-radio-active" + (c.capabilities.mobile || c.capabilities.tablet ? " a-icon-touch-radio-active" : ""),
                    inactive: "a-icon-radio-inactive" + (c.capabilities.mobile || c.capabilities.tablet ? " a-icon-touch-radio-inactive" : ""),
                    disabled: "a-icon-radio-disabled"
                }
            },
            i = 0, m = function(b) {
                return b && b.jquery ? b : b && b.nodeType === 1 ? k(b) : null
            }, j = function(b) {
                return b.find("input").first()
            }, n = function(b) {
                if ((b = m(b)) && b.length === 1) {
                    var c = b.closest("form");
                    c.length === 0 && (c = b.closest("fieldset"), c.length === 0 && (c = k(document)));
                    return c
                }
            }, g = function(b, c, a) {
                var b = m(b), f=!!c, g=!!a;
                if (b && b.length === 1) {
                    var i = j(b), k = b.find("i.a-icon").first(), n = b.hasClass("a-touch-multi-select") ? "multi-select": "checkbox";
                    b.data("a-checkbox-tabindex");
                    if (c !== void 0) {
                        var c = o.checkboxes[n].active, p = o.checkboxes[n].inactive,
                        t = i.prop("checked");
                        k.toggleClass(c, f).toggleClass(p, !f);
                        b.attr("aria-checked", f);
                        t !== f && i.prop("checked", f).trigger("change")
                    }
                    if (a !== void 0 && n === "checkbox")
                        a = o.checkboxes[n].disabled, f = i.prop("disabled"), k.toggleClass(a, g), b.attr("aria-disabled", g), f !== g && i.prop("disabled", g)
                }
            }, f = function(b, c, a) {
                var b = m(b), f=!!c, g=!!a;
                if (b && b.length === 1) {
                    var i = j(b), k = b.find("i.a-icon").first(), p = b.data("a-radio-tabindex"), y = n(i), t = i[0].name, v = y.find('i[data-a-input-name="' + t + '"]');
                    y.find('input[name="' + t + '"]');
                    y =
                    v.closest(".a-radio-fancy, .a-touch-radio");
                    if (c !== void 0) {
                        var c = o.radio.active, t = o.radio.inactive, z = i.prop("checked");
                        f && (y.attr({
                            "aria-checked": !1,
                            tabindex: - 1
                        }), v.addClass(t).removeClass(c));
                        b.attr({
                            "aria-checked": f,
                            tabindex: f ? p ? p: 0: - 1
                        });
                        k.toggleClass(c, f).toggleClass(t, !f);
                        z !== f && i.prop("checked", f).trigger("change")
                    }
                    if (a !== void 0)
                        a = o.radio.disabled, f = i.prop("disabled"), k.toggleClass(a, g), b.attr("aria-disabled", g), f !== g && i.prop("disabled", g);
                    y.filter('[aria-checked="true"]').length === 0 && y.first().attr("tabindex",
                    p ? p : 0)
                }
            };
            return {
                findFormElementContainer: n,
                toggleCheckboxState: function(b) {
                    if ((b = m(b)) && b.length === 1) {
                        var c = j(b);
                        g(b, !c[0].checked)
                    }
                },
                setCheckboxState: g,
                setRadioState: f,
                normalizeElement: function(b) {
                    if ((b = (b = m(b)) ? b : m(this)) && b.length === 1) {
                        var c = j(b), a = c.attr("type"), l = b.hasClass("a-touch-multi-select");
                        if (!b.attr("id")&&!c.attr("id") && (!l || l&&!b.parent().attr("id")))
                            l = "a-form-controls-autoid-" + i, b.attr("aria-labelledby", l).find(".a-checkbox-label, .a-radio-label, .a-touch-multi-select-item-label").attr("id",
                            l), i++;
                        switch (a) {
                        case "checkbox":
                            a = b.attr("tabindex");
                            b.data("a-checkbox-tabindex", a>-1 ? a : 0);
                            g(b, c[0].checked, c[0].disabled);
                            break;
                        case "radio":
                            f(b, c[0].checked, c[0].disabled)
                        }
                    }
                },
                normalizeRadioGroups: function(b) {
                    var c = {};
                    k(b).each(function(a, f) {
                        var g = k(f), g = j(g), i = g[0].name;
                        if (!(i === void 0 || c[i])) {
                            c[i]=!0;
                            var g = n(g).find('input[name="' + i + '"]').closest(b), m = g.length, o = 0, p;
                            g.each(function(a, b) {
                                var c = k(b), e = c.attr("tabindex");
                                c.attr({
                                    "aria-posinset": a + 1,
                                    "aria-setsize": m
                                });
                                if (e > 0 && (o === 0 || e < o))
                                    o = e;
                                c.attr("aria-checked") ===
                                "true" && (p = c)
                            }).data("a-radio-tabindex", o);
                            p ? p.attr("tabindex", o) : g.first().attr("tabindex", o)
                        }
                    })
                },
                normalizeFieldsets: function(b) {
                    k(b).closest("fieldset").each(function(b, a) {
                        var c = k(a), f = c.find("legend").first();
                        if (f.length) {
                            var g = f.attr("id");
                            g || (g = "a-form-controls-autoid-" + i, f.attr("id", g), i++);
                            c.attr("aria-describedby", g)
                        }
                    })
                }
            }
        });
        p.when("A", "a-form-controls-api").register("a-form-controls-handlers", function(c, k) {
            var o = c.$, i = c.constants.keycodes.UP_ARROW, m = c.constants.keycodes.DOWN_ARROW, j = c.constants.keycodes.LEFT_ARROW,
            n = c.constants.keycodes.RIGHT_ARROW, g = function() {
                o(this).removeClass("a-hover-disable")
            }, f = function(b) {
                var e = o(this), a = e.find("input"), f = a.attr("type");
                if (!a[0].disabled) {
                    switch (f) {
                    case "checkbox":
                        !c.capabilities.mobile&&!c.capabilities.tablet && a[0].checked && b && b.type !== "keydown" && (e.addClass("a-hover-disable"), e.one("mouseleave", g));
                        k.setCheckboxState(e, !a[0].checked);
                        break;
                    case "radio":
                        k.setRadioState(e, !0)
                    }
                    b && b.preventDefault && b.preventDefault()
                }
            };
            return {
                elementClicked: f,
                navigateRadiosWithArrows: function(b) {
                    var c =
                    o(this), a = c.find("input")[0].name, g = b.keyCode, q;
                    if (g === i || g === m || g === j || g === n) {
                        var a = k.findFormElementContainer(c).find('.a-radio-fancy[data-a-input-name="' + a + '"], .a-touch-radio[data-a-input-name="' + a + '"]'), s = a.length, c = a.index(c);
                        if (c > 0 && (g === i || g === j))
                            q = a.eq(c - 1);
                        else if (s > c + 1 && (g === m || g === n))
                            q = a.eq(c + 1)
                    }
                    q && (f.call(q, b), q.focus(), q.attr("aria-disabled") === "true" && b && b.preventDefault && b.preventDefault())
                },
                accessibilityKeyPress: function(b) {
                    b.keyCode === c.constants.keycodes.SPACE && (b.preventDefault(),
                    b.stopPropagation())
                },
                formReset: function(b, e) {
                    var a = k.findFormElementContainer(b);
                    c.delay(function() {
                        a.find(e).each(k.normalizeElement)
                    }, 0)
                }
            }
        });
        p.when("A", "a-form-controls-api", "a-form-controls-handlers", "ready").register("a-form-controls", function(c, k, o) {
            function i(c) {
                o.formReset(c.currentTarget, j)
            }
            var m = c.$, j = ".a-radio-fancy, .a-checkbox-fancy";
            m("html").hasClass("a-lt-ie8") || (m(document).delegate(j, "click", o.elementClicked).delegate(j, "keypress", o.accessibilityKeyPress).delegate(j, "keydown", function(j) {
                j.keyCode ===
                c.constants.keycodes.SPACE && o.elementClicked.call(this, j)
            }).delegate(".a-radio-fancy", "keydown", o.navigateRadiosWithArrows), c.on("a:pageUpdate beforeReady", function() {
                k.normalizeRadioGroups(".a-radio-fancy");
                k.normalizeFieldsets(j);
                m(document.body).find(j).each(k.normalizeElement);
                m("form").unbind("reset.a-form-controls-reset").bind("reset.a-form-controls-reset", i)
            }))
        });
        p.when("A", "jQuery").register("a-buttons", function(c, k) {
            var k = c.$, o = 0;
            c.declarative("a-button-group", ["click"], function(i) {
                var k =
                i.$target.closest(".a-button:not(.a-button-disabled)");
                if (k.length) {
                    var j = i.$declarativeParent.find(".a-button"), n = i.data && i.data.name ? i.data.name: !1, i = i.$target.attr("name");
                    j.removeClass("a-button-selected").attr("aria-checked", "false");
                    k.addClass("a-button-selected").attr("aria-checked", "true");
                    if (i || n)
                        k = {
                            $button: k,
                            buttonName: i,
                            buttonGroupName: n
                        }, n && (c.trigger("a:button-group:" + n + ":toggle", {
                            selectedButton: k
                        }), i && c.trigger("a:button-group:" + n + ":" + i + ":toggle", {
                            selectedButton: k
                        }))
                }
            });
            c.on("a:pageUpdate beforeReady",
            function() {
                var c = k(".a-button:not([id])"), m = k(".a-button-group,.a-button-toggle-group");
                c.each(function() {
                    var c = k(this), i = c.find(".a-button-text"), g = c.find(".a-button-input"), f = "a-autoid-" + o++;
                    c.attr("id", f);
                    i.length && (f = (c = i.attr("id")) ? c : f + "-announce", g.attr("aria-labelledby", f), i.attr("id", f))
                });
                m.each(function() {
                    var c = k(this).find(".a-button[role='radio']"), i = c.length, g = 1;
                    c.each(function() {
                        k(this).attr({
                            "aria-posinset": g++,
                            "aria-setsize": i
                        })
                    })
                })
            });
            k(document).delegate(".a-button-input, .a-button-text",
            "focusin", function() {
                var c = k(this).closest(".a-button");
                c.hasClass("a-button-disabled") || c.addClass("a-button-focus")
            }).delegate(".a-button-input, .a-button-text", "focusout " + c.action.cancel, function() {
                k(this).closest(".a-button").removeClass("a-button-focus")
            })
        });
        p.when("a-switch-framework", "jQuery").register("a-switch", function(c, k) {
            var o = c.SWITCH_STATE, i = c.SWITCH_CONTAINER_CLASS, m = c.SWITCH_CLASS;
            return {
                getSwitch: function(j) {
                    function n(b) {
                        var e = f.data(o);
                        if (b === void 0)
                            return e.isOn;
                        if (!e.isEnabled ||
                        g(f))
                            return !1;
                        c.setOnState(f, b);
                        return !0
                    }
                    function g() {
                        return f.data(o).isDragging
                    }
                    j.jquery || (j = k(j));
                    if (j.length === 0)
                        return null;
                    j = j.eq(0);
                    j = j.closest("." + i);
                    if (j.length === 0)
                        return null;
                    var f = j.children("." + m);
                    c.ensureInitialized(f);
                    return {
                        toggle: function() {
                            return n(!f.data(o).isOn)
                        },
                        isOn: n,
                        enabled: function(b) {
                            var e = f.data(o);
                            if (b === void 0)
                                return e.isEnabled;
                            if (e.isEnabled === b)
                                return !1;
                            c.setEnabled(f, b);
                            return !0
                        },
                        isDragging: g,
                        label: function(b) {
                            var c = f.data(o).label;
                            if (b === void 0)
                                return c.text();
                            c.text(b)
                        }
                    }
                }
            }
        });
        p.when("A", "jQuery").register("a-switch-framework", function(c, k) {
            function o(a) {
                a.preventDefault();
                var b = a.data.$switch.data("a-switch-state"), e = b.control;
                if (!c.isAnimated(e)) {
                    a = y(a) - b.initialX;
                    b.isOn && (a += b.rightBoundary);
                    var f = b.leftBoundary, g = b.rightBoundary;
                    a < f ? a = f : a > g && (a = g);
                    if (a !== b.leftOffset)
                        c.animate(e, {
                            left: a
                        }, 0), b.leftOffset = a, b.isDragging=!0, b.dragCount++
                }
            }
            function i(a) {
                a.preventDefault();
                if (c.capabilities.touch || a.which === 1) {
                    var a = a.data.$switch, b = a.data("a-switch-state");
                    n(a, b.isDragging &&
                    b.dragCount > 1 ? b.leftOffset > b.midPoint : !b.isOn);
                    b.isDragging=!1;
                    w(a)
                }
            }
            function m(a, b, e) {
                e = {
                    switchState: a,
                    previousState: e
                };
                c.trigger("a:switch:" + b, e);
                a.name && c.trigger("a:switch:" + a.name + ":" + b, e)
            }
            function j(a) {
                if (!a.data("a-switch-state")) {
                    var b = a.siblings("input"), c = a.closest(".a-switch-row"), e = a.children("." + g), j = a.siblings("." + f), l = b.attr("name"), i = c.hasClass("a-active"), k=!c.hasClass("a-disabled"), m = a.width() - e.width() - 1;
                    a.data("a-switch-state", {
                        input: b,
                        container: c,
                        control: e,
                        label: j,
                        isDragging: !1,
                        rightBoundary: m,
                        leftBoundary: - 1,
                        midPoint: m / 2,
                        initialX: null,
                        leftOffset: i ? m: - 1,
                        isOn: i,
                        isEnabled: k,
                        name: l,
                        dragCount: 0
                    })
                }
            }
            function n(a, b) {
                j(a);
                var e = a.data("a-switch-state"), f = e.isOn, g = b !== e.isOn;
                e.isOn = b;
                var l = e.isOn ? e.rightBoundary: e.leftBoundary;
                c.animate(e.control, {
                    left: l
                }, 300, "ease-out");
                e.leftOffset = l;
                l = e.container;
                e.isOn ? l.addClass("a-active") : l.removeClass("a-active");
                l = e.input;
                e.isOn ? l.attr("checked", "checked") : l.removeAttr("checked");
                g && m(e, "flip", f);
                b ? m(e, "on", f) : m(e, "off", f)
            }
            var g = "a-switch-control",
            f = "a-switch-label", b = function(a) {
                a.bind("touchmove.a-switch-component", {
                    $switch: a
                }, o);
                a.bind("touchend.a-switch-component", {
                    $switch: a
                }, i);
                a.bind("touchcancel.a-switch-component", {
                    $switch: a
                }, i);
                a.bind("mouseup.a-switch-component", {
                    $switch: a
                }, i)
            }, e = function(a) {
                a.unbind("touchmove.a-switch-component");
                a.unbind("touchend.a-switch-component");
                a.unbind("touchcancel.a-switch-component");
                a.unbind("mouseup.a-switch-component")
            }, a = function(a) {
                return (a.originalEvent.touches[0] || a.originalEvent.changedTouches[0]).pageX
            },
            l = function(a) {
                k("body").bind("mousemove.a-switch-component", {
                    $switch: a
                }, o);
                k("body").bind("mouseup.a-switch-component", {
                    $switch: a
                }, i)
            }, q = function() {
                k("body").unbind("mousemove.a-switch-component", o);
                k("body").unbind("mouseup.a-switch-component", i)
            }, s = function(a) {
                return a.pageX
            }, p = null, w = null, y = null;
            c.capabilities.touch ? (p = b, w = e, y = a) : (p = l, w = q, y = s);
            c.declarative("a-switch", c.capabilities.touch ? "touchstart" : "mousedown", function(a) {
                var b = a.$event;
                b.preventDefault();
                if (c.capabilities.touch || b.which ===
                1) {
                    a = a.$declarativeParent;
                    j(a);
                    var e = a.data("a-switch-state");
                    e.dragCount = 0;
                    e.isDragging=!1;
                    if (e.isEnabled)
                        e.initialX = y(b), p(a)
                }
            });
            c.declarative("a-switch-label", "click", function(a) {
                a.$event.preventDefault();
                a = a.$target.siblings(".a-switch");
                j(a);
                var b = a.data("a-switch-state");
                n(a, !b.isOn)
            });
            return {
                ensureInitialized: j,
                setOnState: n,
                setEnabled: function(a, b) {
                    j(a);
                    var c = a.data("a-switch-state"), e = c.container;
                    b ? e.removeClass("a-disabled") : e.addClass("a-disabled");
                    c.isEnabled = b
                },
                SWITCH_STATE: "a-switch-state",
                SWITCH_CONTAINER_CLASS: "a-switch-row",
                SWITCH_CLASS: "a-switch"
            }
        })
    })
})(function() {
    var p = window.AmazonUIPageJS || P, c = p.attributeErrors;
    return c ? c("AmazonUIComponents") : p
}());
(function(p) {
    p.execute(function() {})
})(function() {
    var p = window.AmazonUIPageJS || P, c = p.attributeErrors;
    return c ? c("AmazonUICompatJS") : p
}());
(function(p) {
    p.execute(function() {
        p.register("a-carousel-utils", function() {
            function c(c) {
                return typeof c === "string"
            }
            function k(c) {
                return c && c.nodeType !== void 0
            }
            function o(i) {
                return !i ? null : c(i) || k(i) ? i : o(i.content)
            }
            function i(m) {
                if (m)
                    c(m) || k(m) ? m=!0 : m.content = i(m.content);
                return m
            }
            return {
                isString: c,
                isElement: k,
                getElementFromItem: o,
                clearElementFromItem: i,
                addElementToDom: function(i, j) {
                    j && (c(j) ? i.html(j) : k(j) && i.empty().append(j), j!==!0 && i.removeClass("a-carousel-card-empty"))
                }
            }
        });
        p.when("A", "jQuery").register("a-carousel-measure",
        function(c, k) {
            return function(o) {
                function i(j, i, g) {
                    var f, b, e, a;
                    i.jquery || (i = k(i));
                    for (c.each(g, function(a) {
                        if (a === "top" || a === "left")
                            return b = i.offset(), !1
                    }); (f = g.pop()) !== void 0;)
                        e = j[f], j[f] = f === "left" || f === "top" ? b[f] : f.indexOf("outer")>-1 ? i[f](!0) : i["outer" + f.charAt(0).toUpperCase() + f.substr(1)](), j[f] !== e && (a === void 0 && (a = {}), a[f] = e);
                    return a
                }
                var m = {
                    carousel: {
                        height: 0,
                        width: 0,
                        outerHeight: 0,
                        outerWidth: 0
                    },
                    viewport: {
                        height: 0,
                        width: 0,
                        outerHeight: 0,
                        outerWidth: 0
                    },
                    items: [],
                    getFirstCardWidth: function() {
                        return this.items[0] ===
                        void 0 ||!c.isFiniteNumber(this.items[0].width) ? 160 : this.items[0].width
                    }
                };
                o.measure = function(j) {
                    var k = this.dom.$carousel, g = this.dom.$viewport, f = {};
                    j && (j = j.split(" "));
                    if (!j || c.indexOfArray(j, "carousel")>-1)
                        f.carousel = i(m.carousel, k, "top,left,height,width,outerHeight,outerWidth".split(","));
                    if (!j || c.indexOfArray(j, "viewport")>-1)
                        f.viewport = i(m.viewport, g, ["height", "width", "outerHeight", "outerWidth"]);
                    if (!j || c.indexOfArray(j, "items")>-1)
                        m.items = [], f.items = {}, k.children("li").each(function(b, c) {
                            m.items[b] =
                            {};
                            var a = i(m.items[b], c, "top,left,height,width,outerHeight,outerWidth".split(","));
                            a !== void 0 && (f.items[b] = a)
                        });
                    return f
                };
                o.getItemOffset = function(c) {
                    var i = m.items;
                    c--;
                    if (i && i.length) {
                        if (c < i.length) {
                            for (var g = 0, f = i[0].outerWidth, b = 0; b < c; b++)
                                g += i[b] ? i[b].outerWidth : f;
                            c > 0 && this.getAttr("first_item_flush_left") && (g += o.getAttr("currentGutter"));
                            return g
                        }
                    } else 
                        return 0
                };
                o.getDimensions = function() {
                    return c.copy(m)
                };
                o.updateDimensionsCache = function(j) {
                    c.extend(m, j)
                };
                o.getViewportWidth = function() {
                    try {
                        return m.viewport.width
                    } catch (c) {}
                }
            }
        });
        p.when("A", "jQuery").register("a-carousel-attributes", function(c, k) {
            return function(o, i) {
                var m = {}, j = {}, n = {};
                c.extend(m, i);
                o.onChange = function(g, f) {
                    for (var b = g.split(" "), e = b.length, a; e--;)
                        a = b[e], j[a] || (j[a] = []), k.isFunction(f) && c.indexOfArray(j[a], f)===-1 && j[a].push(f);
                    return this
                };
                o.unbind = function(g, f) {
                    if (j[g] && f) {
                        var b = c.indexOfArray(j[g], f);
                        b>-1 && j[g].splice(b, 1)
                    }
                    return this
                };
                o.once = function(c, f) {
                    var b = function() {
                        f.apply(null, arguments);
                        o.unbind(c, b)
                    };
                    return o.onChange(c, b)
                };
                o.setAttr = function(g,
                f, b) {
                    var e = m[g];
                    m[g] = f;
                    if (!b&&!n[g]&&!c.equals(f, e)) {
                        n[g]=!0;
                        f = c.copy(f);
                        e = c.copy(e);
                        if (j[g])
                            for (var b = c.copy(j[g]), a = 0, l = b.length; a < l; a++)
                                b[a](f, e, o, g);
                        f = {
                            newValue: f,
                            oldValue: e,
                            carousel: o
                        };
                        c.trigger("a:carousel:change:" + g, f);
                        m.name && c.trigger("a:carousel:" + m.name + ":change:" + g, f);
                        n[g]=!1
                    }
                    return this
                };
                o.getAttr = function(g) {
                    return c.copy(m[g])
                }
            }
        });
        p.when("A", "jQuery", "a-carousel-measure", "a-carousel-attributes", "a-carousel-strategies").register("a-carousel-base", function(c, k, o, i, m) {
            function j(a) {
                a.onChange(f,
                function(c, e) {
                    var f = a.getAttr(b), j = Math.ceil(f / c);
                    j === 1 && f > 1 ? j = 2 : j < 1 && (j = 1);
                    a.setAttr("pageNumber", j);
                    a.setAttr(g, Math.ceil(a.getAttr("set_size") / c));
                    f = a.getAttr("ajax");
                    c > e && (f && f.prefetch_next_page ? a.strategies.ajax.wantNextPage(a) : a.strategies.ajax.wantCurrentPage(a))
                });
                a.onChange("set_size", function(b, c) {
                    var j = a.getAttr(f), i = a.getAttr(e);
                    a.setAttr(g, Math.ceil(b / j));
                    b < c ? (i.splice(b, Number.MAX_VALUE), a.setAttr(e, i)) : a.strategies.ajax.wantCurrentPage && a.strategies.ajax.wantCurrentPage(a)
                });
                a.onChange(b,
                function(b) {
                    a.dom.$container.find("input[name='firstVisibleItem']").val(b)
                });
                a.onChange("pageNumber", function(b) {
                    b > 0 && b <= a.getAttr(g) && a.setAttr("currentlyWrapping", !1)
                })
            }
            function n(a, b, e) {
                if (arguments.length !== 0) {
                    a.jquery || (a = k(a));
                    this.dom = {
                        $container: a,
                        $viewport: a.hasClass("a-carousel-viewport") ? a: a.find(".a-carousel-viewport"),
                        $carousel: a.find(".a-carousel")
                    };
                    var f = {
                        totalPages: 1E3,
                        pageNumber: 1,
                        pageSize: 0,
                        firstVisibleItem: 1,
                        maintain_state: !0,
                        px: 0,
                        auto_adjust_height: !0,
                        ajax: {}
                    };
                    c.extend(f, e);
                    f.maintain_state =
                    !!f.maintain_state;
                    if (f.id_list) {
                        if (!f.set_size)
                            f.set_size = f.id_list.length
                    } else 
                        f.id_list = [];
                    if (!f.set_size) {
                        var g = this.dom.$carousel.children("li"), j = parseInt(g.first().attr("aria-setsize"), 10);
                        f.set_size = c.isFiniteNumber(j) && j > 0 ? j : g.length
                    }
                    var m = [];
                    this.dom.$carousel.children("li").each(function(a, b) {
                        k(b).attr({
                            "aria-posinset": a + 1,
                            "aria-setsize": j
                        });
                        m.push(b.innerHTML)
                    });
                    f.fetchedItems = m;
                    o(this);
                    i(this, f);
                    this.strategies = b;
                    return this
                }
            }
            var g = "totalPages", f = "pageSize", b = "firstVisibleItem", e = "fetchedItems";
            c.each(m, function(a, b) {
                n.prototype["set" + b.charAt(0).toUpperCase() + b.slice(1) + "Strategy"] = function(a) {
                    this.strategies[name] = a;
                    typeof a.init === "function" && a.init(this)
                }
            });
            m = n.prototype;
            m.gotoNextPage = function(a) {
                this.getAttr("transitionPaused") || this.strategies.transition.gotoNextPage(this, a)
            };
            m.gotoPrevPage = function(a) {
                this.getAttr("transitionPaused") || this.strategies.transition.gotoPrevPage(this, a)
            };
            m.gotoPage = function(a, b) {
                this.getAttr("transitionPaused") || this.strategies.transition.gotoPage(this,
                a, b)
            };
            m.gotoIndex = function(a, b) {
                this.getAttr("transitionPaused") || this.strategies.transition.gotoIndex(this, a, b)
            };
            m.gotoPixel = function(a, b) {
                this.getAttr("transitionPaused") || this.strategies.transition.gotoPixel(this, a, b)
            };
            m.resize = function() {
                this.strategies.display.resize(this, this.measure("carousel viewport"))
            };
            m.pause = function() {
                this.setAttr("transitionPaused", !0)
            };
            m.resume = function() {
                this.setAttr("transitionPaused", !1)
            };
            m.triggerEvent = function(a, b) {
                b = b || {};
                b.carousel = this;
                c.trigger("a:carousel:" +
                a, b);
                var e = this.getAttr("name");
                e && c.trigger("a:carousel:" + e + ":" + a, b)
            };
            m.getStaticLoader = function() {
                return ""
            };
            m.getEmptyCard = function(a, b) {
                return ['<li class="a-carousel-card a-carousel-card-empty" role="listitem" aria-setsize="', b, '" aria-posinset="', a, '">', this.getStaticLoader(), "</li>"].join("")
            };
            m.init = function() {
                var a = this, e = a.strategies, i = a.dom.$viewport[0];
                if (i&&!i.id)
                    i.id = "anonCarousel" + a.__id;
                for (var i = a.dom.$carousel[0], k = i.childNodes, m = k.length; m--;)
                    k[m].tagName && k[m].tagName.toLowerCase() ===
                    "li" || i.removeChild(k[m]);
                if (a.getAttr("set_size") < 1)
                    return e.ajax.init(a), !1;
                a.measure();
                c.each(a.strategies, function(b) {
                    b.init(a)
                });
                e = a.getAttr(f);
                i = a.getAttr("set_size");
                e = Math.ceil(i / e);
                a.setAttr(g, e);
                j(a);
                a.triggerEvent("init");
                c.each(a.strategies, function(b) {
                    b.afterInit && b.afterInit(a)
                });
                a.triggerEvent("afterInit");
                e = a.getAttr(b);
                if (e === 1 && a.getAttr("maintain_state") && (e = parseInt(a.dom.$container.find("input[name='firstVisibleItem']").val(), 10), !c.isFiniteNumber(e) ||!(e > 0 && e <= i)))
                    e = 1;
                if (e > 1) {
                    i =
                    700;
                    k = Math.ceil(e / a.getAttr(f));
                    for (m = 2; m < k; m++)
                        i += 700 / m;
                    a.gotoIndex(e, {
                        animationDuration: i,
                        easingFunction: "ease"
                    })
                }
                return !0
            };
            return n
        });
        p.when("A", "jQuery", "a-carousel-base").register("a-carousel-mobile", function(c, k, o) {
            function i(c) {
                var g = c.getAttr("loaderHeight");
                g || ((g = c.getAttr("maxHeight")) ? (g*=0.9, g = Math.min(g, 90), g = Math.max(g, 120)) : g = 90, c.setAttr("loaderHeight", g));
                return g
            }
            function m(k, g, f) {
                o.call(this, k, g, f);
                if (arguments.length !== 0)
                    return this.getAttr("circular") === j && this.setAttr("circular",
                    !1), this.getAttr("show_partial_next") === j && this.setAttr("show_partial_next", !0), this.getAttr("hide_off_screen") === j && this.setAttr("hide_off_screen", !1), this.getAttr("springiness") === j && this.setAttr("springiness", 0.8), this.getAttr("touch_easing") === j && this.setAttr("touch_easing", "cubic-bezier(0.215, 0.610, 0.355, 1.000)"), this.init = function() {
                        var b = this;
                        if (o.prototype.init.call(b)) {
                            b.dom.$carousel.children("li").children(".a-loading-static").css("height", i(b) + "px");
                            var e = b.dom.$viewport;
                            (c.capabilities.touch ||
                            c.capabilities.pointerPrefix) && b.getAttr("transitionStrategy") !== "none" && p.when("a-touch").execute(function() {
                                e.addClass("a-gesture a-gesture-horizontal").bind("pan-horizontal swipe-horizontal", function() {
                                    return !1
                                });
                                c.on("a:swipe-horizontal:" + e[0].id, function(a) {
                                    if (!b.getAttr("transitionPaused") && b.strategies.transition.onSwipe)
                                        b.strategies.transition.onSwipe(b, a)
                                    });
                                    if (!b.getAttr("disable_panning"))
                                        c.on("a:pan-horizontal:" + e[0].id, function(a) {
                                            if (!b.getAttr("transitionPaused") && b.strategies.transition.onPan)
                                                b.strategies.transition.onPan(b,
                                                a)
                                            })
                                        });
                                        if ((c.capabilities.isIE10 || c.capabilities.isIE11Plus) && b.getAttr("transitionStrategy") !== "none") {
                                            var a = function(b) {
                                                b.stopPropagation();
                                                b.preventDefault();
                                                document.body.removeEventListener("click", a, !0)
                                            };
                                            e.bind(c.action.start, function() {
                                                e.bind("swipe-horizontal.a-ssiec pan-horizontal.a-ssiec", function() {
                                                    e.unbind(".a-ssiec");
                                                    e.bind(c.action.end + ".a-ssiec", function() {
                                                        e.unbind(".a-ssiec");
                                                        document.body && document.body.addEventListener("click", a, !0)
                                                    })
                                                })
                                            })
                                        }
                                        return !0
                                    } else 
                                        return !1
                    }, this
            }
            var j;
            m.prototype =
            new o;
            m.prototype.constructor = m;
            m.prototype.getStaticLoader = function() {
                return '<div class="a-loading-static" style="height:' + i(this) + 'px"><div class="a-loading-static-inner"></div></div>'
            };
            return m
        });
        p.when("A", "jQuery", "a-carousel-base").register("a-carousel-desktop", function(c, k, o) {
            function i(c) {
                var f = c.getAttr("set_size") <= c.getAttr("pageSize"), b = c.getAttr("transitionStrategy") === "none";
                c.getAttr("totalPages") === 1 && c.getAttr(n) > 1 && c.gotoPage(1, {
                    startover: !0,
                    animationDuration: 0
                });
                c.dom.$container.find(".a-carousel-left, .a-carousel-right, .a-carousel-pagination").css("visibility",
                f || b ? "hidden" : "visible")
            }
            function m(g, f, b) {
                o.call(this, g, f, b);
                if (arguments.length !== 0) {
                    var e = this;
                    e.getAttr("circular") === j && this.setAttr("circular", !0);
                    e.getAttr("hide_off_screen") === j && this.setAttr("hide_off_screen", !0);
                    e.onChange("totalPages", function(a) {
                        e.dom.$container.find(".a-carousel-page-max").html(a);
                        a < e.getAttr(n) && e.gotoPage(a)
                    });
                    e.onChange(n, function(a) {
                        var b = e.dom.$container, c = b.find(".a-carousel-restart-container");
                        a > 1 ? c.show() : c.hide();
                        b.find(".a-carousel-page-current").html(a)
                    });
                    e.init = function() {
                        var a = this;
                        if (o.prototype.init.call(a)) {
                            i(this);
                            a.onChange("pageSize set_size", function() {
                                i(a)
                            });
                            a.getAttr(n) < 2 && a.dom.$container.find(".a-carousel-restart-container").hide();
                            var b = a.dom.$container.find(".a-carousel-button");
                            if (b.length) {
                                var e = b.eq(0).position().top + "px";
                                b.css("top", e)
                            }
                            var f=!1;
                            a.dom.$container.delegate(".a-carousel-goto-nextpage", "click dblclick", function(b) {
                                f || (f=!0, b.preventDefault(), a.gotoNextPage(), c.delay(function() {
                                    f=!1
                                }, 5))
                            }).delegate(".a-carousel-goto-prevpage",
                            "click dblclick", function(b) {
                                f || (f=!0, b.preventDefault(), a.gotoPrevPage(), c.delay(function() {
                                    f=!1
                                }, 5))
                            }).delegate(".a-carousel-goto-nextpage", "keydown", function(b) {
                                if (b.which === c.constants.keycodes.ENTER || b.which === c.constants.keycodes.SPACE)
                                    b.preventDefault(), a.gotoNextPage(), a.dom.$viewport.focus()
                            }).delegate(".a-carousel-goto-prevpage", "keydown", function(b) {
                                if (b.which === c.constants.keycodes.ENTER || b.which === c.constants.keycodes.SPACE)
                                    b.preventDefault(), a.gotoPrevPage(), a.dom.$viewport.focus()
                            }).delegate(".a-carousel-restart",
                            "click", function(b) {
                                b.preventDefault();
                                b = {
                                    startover: !0
                                };
                                a.getAttr(n) > 5 ? b.animationDuration = 1250 : b.animationSpeed = a.getDimensions().viewport.width * 5;
                                a.gotoPage(1, b)
                            });
                            a.dom.$container.find(".a-carousel-page-max").html(this.getAttr("totalPages"));
                            var g = a.dom.$viewport;
                            (c.capabilities.touch || c.capabilities.pointerPrefix) && a.getAttr("transitionStrategy") !== "none" && p.when("a-touch").execute(function() {
                                g.addClass("a-gesture a-gesture-horizontal").bind("pan-horizontal swipe-horizontal", function() {
                                    return !1
                                });
                                c.on("a:swipe-horizontal:" + g[0].id, function(b) {
                                    if (!a.getAttr("transitionPaused") && a.strategies.transition.onSwipe)
                                        a.strategies.transition.onSwipe(a, b)
                                });
                                if (!a.getAttr("disable_panning"))
                                    c.on("a:pan-horizontal:" + g[0].id, function(b) {
                                        if (!a.getAttr("transitionPaused") && a.strategies.transition.onPan)
                                            a.strategies.transition.onPan(a, b)
                                        })
                            });
                            if ((c.capabilities.isIE10 || c.capabilities.isIE11Plus) && a.getAttr("transitionStrategy") !== "none") {
                                var j = function(a) {
                                    a.stopPropagation();
                                    a.preventDefault();
                                    document.body.removeEventListener("click",
                                    j, !0)
                                };
                                g.bind(c.action.start, function() {
                                    g.bind("swipe-horizontal.a-ssiec pan-horizontal.a-ssiec", function() {
                                        g.unbind(".a-ssiec");
                                        g.bind(c.action.end + ".a-ssiec", function() {
                                            g.unbind(".a-ssiec");
                                            document.body && document.body.addEventListener("click", j, !0)
                                        })
                                    })
                                })
                            }
                            return !0
                        } else 
                            return !1
                    };
                    return e
                }
            }
            var j, n = "pageNumber";
            m.prototype = new o;
            return m.prototype.constructor = m
        });
        p.when("A", "a-carousel-desktop", "a-carousel-mobile").register("a-carousel-classes", function(c, k, o) {
            return {
                desktop: k,
                mobile: o,
                "default": c.capabilities.mobile ||
                c.capabilities.tablet ? "mobile": "desktop"
            }
        });
        p.when("A", "jQuery", "p-detect").register("a-carousel-stretchygoodness", function(c, k) {
            function o(a, b, e, j) {
                a.getAttr(g) && (b -= e / 10);
                var i = a.getAttr(f);
                a.getAttr("set_size");
                for (var a = 0, l=!0; b > 0;)
                    a++, b -= j && l ? e : e + i, l=!1;
                b < 0 && a--;
                return c.isFiniteNumber(a) && a > 0 ? a : 1
            }
            function i(a, b, e, f, j, i, l) {
                a.getAttr(y) === "stretch" && f > i && (f = i);
                b -= e * f;
                a.getAttr(g) ? (a = b - j * (f + 1), l && (a += j), l = a / e, b -= e * (l > 0.5 ? 0.5 : l)) : l && (b += j);
                e = Math.ceil(b / (f + 1));
                if (!c.isFiniteNumber(e) || e < j)
                    e = j;
                return e
            }
            function m(b) {
                if (b.getAttr("auto_adjust_height"))
                    if (b.getAttr(x))
                        b.once(x, function() {
                            m(b)
                        });
                    else {
                        var f = b.getAttr(q), j = b.getDimensions();
                        if (!f ||!c.isFiniteNumber(f))
                            f = 1;
                            var i = f, k = b.getAttr(e), n = k * (b.getAttr(a) - 1), k = n + k - 1, j = j.items, o = j.length, p;
                            for (b.getAttr(g) && k++; n <= k && n < o; n++)(p = j[n]) 
                                && p.outerHeight > i && (i = j[n].outerHeight || j[n].height);
                                i > f && (b.updateDimensionsCache({
                                    viewport: {
                                        height: i,
                                        outerHeight: i
                                    }
                                }), b.setAttr(q, i), f === 1 ? b.dom.$viewport.height(i) : c.animate(b.dom.$viewport, {
                                    height: i
                                }, b.getAttr(l),
                                "linear"))
                    } else 
                        b.dom.$viewport.css("height", "")
            }
            function j(b) {
                b.onChange(a, function() {
                    b.getAttr(w) && b.dom.$carousel.children("li").css("visibility", "")
                });
                b.onChange(e, function(a, c) {
                    a > c && m(b)
                });
                b.onChange("loading", function(a) {
                    a || m(b)
                });
                b.onChange(p, function() {
                    m(b)
                });
                b.onChange(x, function(a) {
                    if (!a && b.getAttr(w)) {
                        var c = b.getAttr(p) - 1, f = c + b.getAttr(e) - 1;
                        b.getAttr(g) && f++;
                        b.dom.$carousel.children("li").each(function(a, b) {
                            var e = a >= c && a <= f;
                            k(b).css("visibility", e ? "" : "hidden").attr("aria-hidden", e ? "false" :
                            "true")
                        })
                    }
                });
                b.onChange(y + " " + f, function() {
                    n(b)
                });
                b.onChange(f, function() {
                    n(b)
                })
            }
            function n(a) {
                var c = a.getDimensions(), g = c.viewport.width, c = c.getFirstCardWidth(), j = a.getAttr(f), l = a.getAttr("set_size"), k = a.getAttr(t), m = o(a, g, c, k), n = i(a, g, c, m, j, l, k);
                a.setAttr(b, n);
                a.setAttr(e, m);
                var q = a.dom.$carousel, x = q.children("li"), j = x.length, h = a.getAttr(p) - 1, H = h + m - 1;
                a.getAttr(w) && H++;
                var L = a.getAttr(w), O = n + "px", R = c + "px", M;
                x.each(function(a, b) {
                    M=!L || a >= h && a < H;
                    b.style.marginLeft = k && a === 0 ? 0 : O;
                    b.style.visibility = M ?
                    "" : "hidden";
                    b.style.width = R;
                    b.setAttribute("aria-hidden", M ? "false" : "true")
                });
                var I, K;
                a.getAttr(t) ? (n = x.first().outerWidth(!0), x.length > 1 && (I = x.eq(1).outerWidth(!0)), K = (j - 1) * I + n) : (n = I = x.first().outerWidth(!0), K = j * I);
                if (m >= l) {
                    K = g;
                    var Z = a.getAttr(y);
                    q.toggleClass("a-text-right", Z === "right");
                    q.toggleClass("a-text-center", Z === "center");
                    Z === "center" && x.first().css("margin-left", 0)
                } else 
                    q.removeClass("a-text-right a-text-center");
                K = m >= l ? g : K;
                q.css("width", K + "px");
                g = {
                    carousel: {
                        width: K,
                        outerWidth: q.outerWidth()
                    },
                    items: []
                };
                for (l = 0; l < j; l++)
                    g.items.push({
                        width: c,
                        outerWidth: l === 0 ? n: I
                    });
                a.updateDimensionsCache(g);
                a.gotoIndex(a.getAttr(p), {
                    animationDuration: 0
                });
                a.triggerEvent("repaint")
            }
            var g = "show_partial_next", f = "minimum_gutter_width", b = "currentGutter", e = "pageSize", a = "pageNumber", l = "height_animation_speed", q = "maxHeight", p = "firstVisibleItem", x = "animating", w = "hide_off_screen", y = "single_page_align", t = "first_item_flush_left";
            return {
                repaint: n,
                init: function(a) {
                    var i = a.getAttr(f);
                    c.isFiniteNumber(i) || (i = 15, a.setAttr(f,
                    i));
                    a.setAttr(b, i);
                    i = a.getAttr(l);
                    c.isFiniteNumber(i) || a.setAttr(l, 200);
                    a.setAttr(t, !!a.getAttr(t));
                    a.setAttr(g, !!a.getAttr(g));
                    n(a);
                    a.getAttr("transitionStrategy") !== "none" && m(a);
                    i = a.getDimensions();
                    a.dom.$container.find(".a-carousel-left, .a-carousel-right, .a-carousel-viewport").css("height", Math.max(i.viewport.height, i.items[0] ? i.items[0].height : 0) + "px");
                    i = a.getAttr(p);
                    i > 1 && (a.setAttr(p, i), i = Math.ceil(i / a.getAttr(e)), a.gotoPage(i));
                    j(a)
                },
                resize: function(a, b) {
                    b.viewport && b.viewport.width !== void 0 &&
                    n(a)
                }
            }
        });
        p.when("A", "jQuery", "p-detect", "a-carousel-utils").register("a-carousel-display-swap", function(c, k, o, i) {
            function m(a) {
                if (a.getAttr("auto_adjust_height")) {
                    var b = a.getAttr(q);
                    if (!b ||!c.isFiniteNumber(b))
                        b = 1;
                    var e = b;
                    a.dom.$carousel.children("li").not(".a-carousel-card-empty").each(function(a, b) {
                        var c = k(b).outerHeight();
                        e = Math.max(c, e)
                    });
                    e > b && (a.setAttr(q, e), o.capabilities.transition ? b === 1 ? a.dom.$viewport.height(e) : c.animate(a.dom.$viewport, {
                        height: e
                    }, a.getAttr(l), "linear") : a.dom.$viewport.height(e),
                    a.updateDimensionsCache({
                        viewport: {
                            height: e,
                            outerHeight: e
                        }
                    }))
                } else 
                    a.dom.$viewport.css("height", "")
            }
            function j(a, b) {
                for (var c = a.dom.$carousel[0], e = a.dom.$carousel.children("li").get(), f; e.length > b;)
                    f = e.pop(), c.removeChild(f)
            }
            function n(l) {
                l.onChange(p, function() {
                    m(l)
                });
                l.onChange(y, function(a) {
                    a || m(l)
                });
                l.onChange(e, function(a, c) {
                    if (a > c) {
                        c === 0 && j(l);
                        var e = l.getAttr("set_size"), f = l.getDimensions().getFirstCardWidth(), g = l.getAttr(b), n = l.getAttr(x), o = l.getAttr(p) - 1, q = l.dom.$carousel.children("li"), t =
                        document.createDocumentFragment(), h;
                        if (k.isArray(n)) {
                            for (var w = q.length; w < a; w++)
                                h = w + o, q = k(['<li class="a-carousel-card a-carousel-card-empty" role="listitem" aria-setsize="', e, '" aria-posinset="', h + 1, '" style="width:', f, "px; margin-left:", g, 'px;">', l.getStaticLoader()].join("")), n[h] && i.addElementToDom(q, i.getElementFromItem(n[h])), h >= e && q.removeClass("a-carousel-card-empty"), t.appendChild(q[0]);
                            l.dom.$carousel.append(t)
                        }
                        m(l)
                    } else 
                        a < c && j(l)
                });
                l.onChange("set_size", function(b, e) {
                    var f = l.getAttr(a),
                    j = l.getAttr(w), i = l.dom.$carousel.children("li");
                    i.attr("aria-setsize", b);
                    f === j && b > e && (i.length && l.dom.$carousel.children("li").each(function(a, b) {
                        if (!c.trim(b.innerHTML))
                            b.className += " a-carousel-card-empty", b.innerHTML = l.getStaticLoader()
                    }), e === 0 && g(l))
                });
                l.onChange(t + " " + f, function() {
                    g(l)
                });
                l.onChange(f, function() {
                    g(l)
                })
            }
            function g(a) {
                var g = a.getDimensions(), i = g.viewport.width, l = g.getFirstCardWidth(), k = a.getAttr(f), g = a.getAttr("set_size"), m;
                m = a.getAttr(f);
                a.getAttr("set_size");
                m = Math.max(Math.floor(i /
                (l + m)), 1);
                m = c.isFiniteNumber(m) ? m : 1;
                var n, o = m;
                a.getAttr(t) === "stretch" && o > g && (o = g);
                o = Math.ceil((i - l * o) / (o + 1));
                c.isFiniteNumber(o) || (o = k);
                n = o;
                a.setAttr(b, n);
                a.setAttr(e, m);
                var o = a.dom.$carousel, q = o.children("li"), k = q.length;
                l += n;
                var p = k * l;
                j(a, Math.min(m, g));
                q.css("margin-left", n + "px");
                m >= g ? (p = i, i = a.getAttr(t), o.toggleClass("a-text-right", i === "right"), o.toggleClass("a-text-center", i === "center"), i === "center" && q.first().css("margin-left", 0)) : o.removeClass("a-text-right a-text-center");
                for (i = {
                    carousel: {
                        width: p,
                        outerWidth: o.outerWidth()
                    },
                    items: []
                }; k--;)
                    i.items.push({
                        outerWidth: l
                    });
                a.updateDimensionsCache(i);
                a.triggerEvent("repaint")
            }
            var f = "minimum_gutter_width", b = "currentGutter", e = "pageSize", a = "pageNumber", l = "height_animation_speed", q = "maxHeight", p = "firstVisibleItem", x = "fetchedItems", w = "totalPages", y = "loading", t = "single_page_align";
            return {
                repaint: g,
                init: function(a) {
                    var i = a.getAttr(f);
                    i || (i = 15, a.setAttr(f, i));
                    a.setAttr(b, i);
                    i = a.getAttr(l);
                    c.isFiniteNumber(i) || a.setAttr(l, 200);
                    n(a);
                    g(a);
                    i = a.getDimensions();
                    a.dom.$container.find(".a-carousel-left, .a-carousel-right, .a-carousel-viewport").css("height", Math.max(i.viewport.height, i.items[0] ? i.items[0].height : 0) + "px");
                    i = a.getAttr(p);
                    i > 1 && (a.setAttr(p, i), i = Math.ceil(i / a.getAttr(e)), a.gotoPage(i))
                },
                resize: function(a, b) {
                    b.viewport && b.viewport.width !== void 0 && g(a)
                }
            }
        });
        p.when("A", "jQuery").register("a-carousel-display-single", function(c) {
            function k(b) {
                if (b.getAttr("auto_adjust_height"))
                    if (b.getAttr(g))
                        b.once(g, function() {
                            k(b)
                        });
                    else 
                        b.dom.$viewport.css("height",
                        "auto"), c.delay(function() {
                            b.dom.$viewport.height(b.dom.$viewport.height())
                        }, 0);
                else 
                    b.dom.$viewport.css("height", "")
            }
            function o(b) {
                b.getAttr(f) || (b.dom.$viewport.delegate("img", "load", function() {
                    k(b)
                }), b.onChange("loading", function(c) {
                    c || k(b)
                }), b.onChange(j, function() {
                    i(b)
                }))
            }
            function i(b) {
                var c = b.getDimensions(), a = c.viewport.width, f = b.getAttr(m), g = b.getAttr(j), i = b.getAttr("set_size");
                a -= g * 2;
                f && (a -= g + c.viewport.width / 3);
                b.dom.$carousel.children("li").css({
                    width: a + "px",
                    margin: "0 " + g + "px"
                });
                c = a + g *
                2;
                f = c * i;
                b.dom.$carousel.width(f);
                for (f = {
                    carousel: {
                        width: f
                    },
                    items: []
                }; i--;)
                    f.items[i] = {
                        width: a,
                        outerWidth: c
                    };
                b.updateDimensionsCache(f);
                b.gotoIndex(b.getAttr(n), {
                    animationDuration: 0
                });
                b.triggerEvent("repaint")
            }
            var m = "show_partial_next", j = "minimum_gutter_width", n = "firstVisibleItem", g = "animating", f = "fixed_height";
            return {
                repaint: i,
                init: function(b) {
                    var e = b.getAttr(j);
                    b.setAttr(j, c.isFiniteNumber(e) ? e : 14);
                    b.setAttr(m, !!b.getAttr(m));
                    b.setAttr("pageSize", 1);
                    b.dom.$carousel.children(":not(li:first-child)").remove();
                    b.setAttr("pageSize", 1);
                    e = b.getAttr(f);
                    c.isFiniteNumber(e) ? b.dom.$viewport.height(e) : b.setAttr(f, !1);
                    b.dom.$carousel.children("li").css("visibility", "visible");
                    o(b);
                    this.repaint(b);
                    k(b)
                },
                resize: function(b, c) {
                    c.viewport && c.viewport.width !== void 0 && (this.repaint(b), b.getAttr(f) || k(b))
                }
            }
        });
        p.when("a-carousel-stretchygoodness", "a-carousel-display-swap", "a-carousel-display-single").register("a-carousel-strategies-display", function(c, k, o) {
            return {
                swap: k,
                single: o,
                stretchyGoodness: c,
                "default": "stretchyGoodness"
            }
        });
        p.when("A", "jQuery", "a-carousel-utils").register("a-carousel-transition-swap", function(c, k, o) {
            function i(a, b) {
                var e = a.getAttr(q);
                e || (e = []);
                for (var f = [], g = b.length - 1; g >= 0; g--)
                    if (b[g]&&!e[g]) {
                        var i = o.getElementFromItem(b[g]);
                        i && k("img", i).each(function() {
                            f.push(this.src)
                        });
                        e[g]=!0
                    }
                c.preload(f);
                a.setAttr(q, e)
            }
            function m(f, g) {
                f.getAttr(a);
                f.getAttr(l);
                var i = f.getAttr(b), j = f.getAttr("delay_time");
                f.setAttr(e, !0);
                var m = f.dom.$carousel.children("li");
                m.each(function(a, b) {
                    var l = k(b), n = g[a + i - 1];
                    l.hasClass(y) &&
                    n && c.delay(function() {
                        o.addElementToDom(l, o.getElementFromItem(n));
                        a === m.length - 1 && f.setAttr(e, !1)
                    }, 0 + j)
                })
            }
            function j(e, i, j) {
                var j = j || {}, k = e.getAttr(a);
                if (i !== k) {
                    var m = e.getAttr(p), q = e.getAttr(w), D = e.getAttr(n), J = e.getAttr(l), u = j.delayTime || e.getAttr(g), E = (typeof j.direction === "number" ? j.direction ? j.direction < 0?-1 : 1 : isNaN(j.direction) ? NaN : 0 : NaN) || NaN;
                    !D && i < 1 ? i = 1 : !D && i > q ? i = q : D && i < 1 ? i = q : D && i > q && (i = 1);
                    E || (E = k < i ? 1 : - 1);
                    j.startover && (E = u = 1);
                    var G = J * (i - 1), h = E === 1 ? 0: J - 1;
                    e.setAttr(a, i);
                    e.setAttr(b, G + 1);
                    e.setAttr("animating",
                    !0);
                    var H = c.interval(function() {
                        var a = G + h;
                        if (H !== e.getAttr(x))
                            clearInterval(H);
                        else if (E===-1 && h < 0 || E === 1 && h >= J)
                            e.setAttr(x, void 0), e.setAttr("animating", !1);
                        else {
                            var b = e.dom.$carousel.children("li").eq(h);
                            b.attr("aria-posinset", a + 1);
                            var c = e.getAttr(f)[a];
                            c ? o.addElementToDom(b, o.getElementFromItem(c)) : a < m ? b.html(e.getStaticLoader()).addClass(y) : b.empty().removeClass(y);
                            h += E
                        }
                    }, u);
                    e.setAttr(x, H)
                }
            }
            var n = "circular", g = "delay_time", f = "fetchedItems", b = "firstVisibleItem", e = "loading", a = "pageNumber", l = "pageSize",
            q = "preloadedImages", p = "set_size", x = "responsiveTimerId", w = "totalPages", y = "a-carousel-card-empty";
            return {
                init: function(a) {
                    var b = a.getAttr(g);
                    c.isFiniteNumber(b) || a.setAttr(g, 30);
                    a.onChange(x, function(a, b) {
                        b !== a && clearInterval(b)
                    });
                    a.onChange(f, function(b) {
                        m(a, b);
                        i(a, b)
                    });
                    i(a, a.getAttr(f))
                },
                gotoIndex: function(a, b, c) {
                    var c = c || {}, e = a.getAttr(l), b = Math.ceil(b / e);
                    j(a, b, c)
                },
                gotoNextPage: function(b, c) {
                    var c = c || {}, e = b.getAttr(a);
                    c.direction =- 1;
                    j(b, ++e, c)
                },
                gotoPrevPage: function(b, c) {
                    var c = c || {}, e = b.getAttr(a);
                    c.direction = 1;
                    j(b, --e, c)
                },
                gotoPage: j
            }
        });
        p.when("A", "jQuery", "a-carousel-utils").register("a-carousel-transition-slide", function(c, k, o) {
            function i(c) {
                var a = c.dom.$carousel.children("li").length, i = a + 1, j = c.getAttr(b), k = j - a;
                if (k > 0) {
                    for (var k = i + k - 1, m = []; i <= k; i++)
                        m.push(c.getEmptyCard(i, j));
                    c.dom.$carousel.append(m.join(""));
                    c.setAttr(f, !0);
                    for (var k = c.getAttr(g), m = c.dom.$carousel.children("li"), n, i = a; i < j; i++)
                        if (n = k[i]) {
                            var p = o.getElementFromItem(n), a = m.eq(i);
                            o.addElementToDom(a, p);
                            k[i] = o.clearElementFromItem(n)
                        }
                    c.strategies.display.repaint &&
                    c.strategies.display.repaint(c);
                    c.setAttr(g, k, !0);
                    c.setAttr(f, !1)
                }
            }
            function m(b, a, c) {
                if (b.getAttr(n))
                    b.once(n, function() {
                        m(b, a, c)
                    });
                else {
                    var i = b.getDimensions().items;
                    if (!c || a.length >= c.length) {
                        b.setAttr(f, !0);
                        for (var j = b.dom.$carousel.children("li"), k = a.length, p, y; k--;)
                            if (y = a[k], y !== c[k] && y) {
                                var t = o.getElementFromItem(y);
                                p = j.eq(k);
                                p.length && (o.addElementToDom(p, t), i[k] = {
                                    width: p.outerWidth(),
                                    outerWidth: p.outerWidth(!0),
                                    height: p.outerHeight(),
                                    outerHeight: p.outerHeight(!0)
                                }, a[k] = o.clearElementFromItem(y))
                            }
                    }
                    b.setAttr(g,
                    a);
                    b.updateDimensionsCache({
                        items: i
                    });
                    b.setAttr(f, !1)
                }
            }
            var j = c.capabilities.touch ? 2E3: 3E3, n = "animating", g = "fetchedItems", f = "loading", b = "set_size";
            return {
                wrapToFirst: function(b) {
                    var a = b.getAttr("pageSize"), c = b.getDimensions().getFirstCardWidth(), f = this;
                    b.gotoPixel(a * c*-1, {
                        animationDuration: 0,
                        callback: function() {
                            b.setAttr("currentlyWrapping", !1);
                            f.gotoPage(b, 1)
                        }
                    })
                },
                wrapToLast: function(b) {
                    b.getAttr("pageSize");
                    var a = b.getAttr("totalPages"), c = this, f = b.getDimensions().carousel.width;
                    b.gotoPixel(f, {
                        animationDuration: 0,
                        callback: function() {
                            b.setAttr("currentlyWrapping", !1);
                            c.gotoPage(b, a)
                        }
                    })
                },
                gotoPage: function(b, a, c) {
                    c = c || {};
                    (c.animationDuration === void 0 || c.animationDuration > 0)&&!c.silent && b.setAttr(n, !0);
                    var f = b.getAttr("totalPages");
                    a > 0 && a <= f && b.setAttr("pageNumber", a);
                    var g = b.getAttr("circular");
                    if (!g && a < 1)
                        a = 1, c.animationDuration = Math.pow(b.getAttr("animation_speed") * b.getAttr("springiness"));
                    else if (!g && a > f)
                        a = f, c.animationDuration = Math.pow(b.getAttr("animation_speed"), b.getAttr("springiness"));
                    this.gotoIndex(b,
                    b.getAttr("pageSize") * (a - 1) + 1, c)
                },
                gotoIndex: function(e, a, f) {
                    f = f || {};
                    (f.animationDuration === void 0 || f.animationDuration > 0)&&!f.silent && e.setAttr(n, !0);
                    var g = e.getAttr("circular")&&!e.getAttr("currentlyWrapping"), i = f.callback, j = this, k=!1, m = e.getViewportWidth(), o = Math.ceil(a / e.getAttr("pageSize")), p;
                    o !== e.getAttr("pageNumber") && o > 0 && o <= e.getAttr("totalPages") && e.setAttr("pageNumber", o);
                    e.setAttr("firstVisibleItem", a);
                    a < 1 ? g && (k = m*-1, p = function() {
                        i && i();
                        j.wrapToLast(e)
                    }) : a > e.getAttr(b) ? g && (k = e.getAttr("px") +
                    m, p = function() {
                        i && i();
                        j.wrapToFirst(e)
                    }) : k = e.getItemOffset(a);
                    p ? (e.setAttr("currentlyWrapping", !0), f.callback = p, f.easingFunction = f.easingFunction || e.getAttr("wrap_easing"), f.animationSpeed = (c.isFiniteNumber(f.animationSpeed) ? f.animationSpeed : e.getAttr("animation_speed")) * 1.3) : f.callback = i;
                    k!==!1 && this.gotoPixel(e, k, f)
                },
                gotoPixel: function(b, a, f) {
                    var g = b.getAttr("px");
                    if (a !== g) {
                        var f = f || {}, i = f.easingFunction || "ease-out", j = f.callback;
                        b.getViewportWidth();
                        var k;
                        f.animationDuration !== void 0 ? k = f.animationDuration :
                        (k = c.isFiniteNumber(f.animationSpeed) ? f.animationSpeed : b.getAttr("animation_speed"), g = Math.abs(a - g), k = k === 0 ? 0 : Math.floor(g / k * 1E3));
                        k > 0&&!f.silent && b.setAttr(n, !0);
                        c.isFiniteNumber(a) ? (g = k > 0 ? function() {
                            j && j();
                            b.setAttr(n, c.isAnimated(b.dom.$carousel), f.silent)
                        } : j, b.setAttr("px", a), c.animate(b.dom.$carousel, {
                            left: a*-1
                        }, k, i, g)) : p.error("Target pixel is not a finite number", "a-carousel-transition-slide", "gotoPixel")
                    }
                },
                gotoNextPage: function(b, a) {
                    var c = b.getAttr("pageNumber");
                    this.gotoPage(b, ++c, a)
                },
                gotoPrevPage: function(b,
                a) {
                    var c = b.getAttr("pageNumber");
                    this.gotoPage(b, --c, a)
                },
                onSwipe: function(b, a) {
                    if (!b.getAttr("currentlyWrapping")) {
                        var f = b.getAttr("firstVisibleItem"), g = b.getAttr("pageSize"), i = b.getAttr("pageNumber"), j = a.velocityX < 0, k = f;
                        j && i < b.getAttr("totalPages") ? k = f + g : !j && i > 1 && (k = f - g);
                        g = b.getAttr("px");
                        i = b.getItemOffset(k);
                        g = Math.abs((j ? g - i : g + i) * 1E3 / a.velocityX);
                        g = Math.max(g, 300);
                        g = Math.min(g, c.viewport().width * 1.2);
                        g = {
                            animationDuration: g,
                            easingFunction: b.getAttr("touch_easing")
                        };
                        k === f&&!b.getAttr("circular") ?
                        (g.animationSpeed = c.viewport().width * 0.95, delete g.animationDuration, b.gotoIndex(k, g)) : j ? b.gotoNextPage(g) : b.gotoPrevPage(g)
                    }
                },
                onPan: function(b, a) {
                    if (!b.getAttr("currentlyWrapping")) {
                        b.setAttr(n, !0);
                        var f = b.getItemOffset(b.getAttr("firstVisibleItem")), g = f - a.touchDeltaX, i = b.getAttr("circular"), j = b.getAttr("pageNumber"), k = b.getAttr("totalPages");
                        if (a.ended) {
                            var f = {
                                easingFunction: b.getAttr("touch_easing"),
                                animationSpeed: c.viewport().width * 0.95,
                                silent: !0
                            }, g = a.touchDeltaX, m = Math.abs(g) < b.getViewportWidth() *
                            0.4;
                            !i && (g < 0 && k === j || g > 0 && j === 1) || m ? b.gotoPage(j, f) : g < 0 ? b.gotoNextPage(f) : b.gotoPrevPage(f)
                        } else {
                            if (!i && (i = b.getAttr("springiness"), g < 0 && a.touchDeltaX > 0 || j === k && a.touchDeltaX < 0))
                                j = Math.pow(Math.abs(a.touchDeltaX), i), g = g <= 0 ? j*-1 : f + j;
                            b.gotoPixel(g, {
                                easingFunction: b.getAttr("touch_easing"),
                                animationDuration: 0,
                                silent: !0
                            })
                        }
                    }
                },
                init: function(e) {
                    var a = e.getAttr("animation_speed");
                    c.isFiniteNumber(a) || e.setAttr("animation_speed", j);
                    e.getAttr("wrap_easing") === void 0 && e.setAttr("wrap_easing", "linear");
                    i(e);
                    e.onChange(g, function(a, b) {
                        m(e, a, b)
                    });
                    e.onChange(b, function(a, b) {
                        a > b && i(e)
                    })
                }
            }
        });
        p.when("A", "jQuery", "a-carousel-transition-slide", "a-carousel-transition-swap").register("a-carousel-strategies-transition", function(c, k, o, i) {
            return {
                slideHorizontal: o,
                swap: i,
                none: {
                    gotoIndex: c.constants.NOOP,
                    gotoNextpage: c.constants.NOOP,
                    gotoPrevPage: c.constants.NOOP,
                    gotoPage: c.constants.NOOP,
                    init: function(c) {
                        c.setAttr("hide_off_screen", !1);
                        c.setAttr("auto_adjust_height", !1);
                        c.dom.$carousel.children("li").css("visibility",
                        "").attr("aria-hidden", "false")
                    }
                },
                "default": "slideHorizontal"
            }
        });
        p.when("A").register("a-carousel-ajax-standard", function(c) {
            function k(f, b, e) {
                f.triggerEvent("beforeAjax", {
                    url: b,
                    params: e
                });
                c.get(b, {
                    cache: !1,
                    success: function(a) {
                        e.needSetSize && ((!a ||!a.length) && p.error("Carousel requires a set_size and none was returned by the fallback AJAX request", "a-carousel-ajax-standard", "sendRequest"), f.setAttr(n, a[0].setSize ? a[0].setSize : a.length));
                        for (var c = f.getAttr("fetchedItems"), j; (j = a.pop()) !== i;)
                            c[e.offset +
                            a.length] = j;
                        e.needSetSize && f.init();
                        f.setAttr("fetchedItems", c);
                        f.setAttr(g, !1);
                        e.needSetSize && f.getAttr(m) >= c.length && f.strategies.ajax.wantCurrentPage(f);
                        f.triggerEvent("ajaxSuccess", {
                            url: b,
                            params: e
                        })
                    },
                    params: e
                })
            }
            function o(c) {
                var b = c.getAttr(j);
                b && (clearTimeout(b), c.setAttr(j, null))
            }
            var i, m = "pageSize", j = "requestTimer", n = "set_size", g = "ajaxLock";
            return {
                getItems: function(f, b, e) {
                    var a = f.getAttr("ajax");
                    f.setAttr(j, c.delay(k, a.fetch_delay, f, b, e))
                },
                wantNextPage: function(c) {
                    o(c);
                    if (c.getAttr("ajax").prefetch_next_page) {
                        var b =
                        c.getAttr(m), e = b * 2;
                        c.getAttr("show_partial_next") && e++;
                        this.want(c, (c.getAttr("pageNumber") - 1) * b, e)
                    } else 
                        this.wantCurrentPage(c)
                },
                wantPrevPage: function(c) {
                    o(c);
                    if (c.getAttr("ajax").prefetch_next_page) {
                        var b = c.getAttr(m), e = b * 2;
                        c.getAttr("show_partial_next") && e++;
                        this.want(c, (c.getAttr("pageNumber") - 2) * b, e)
                    } else 
                        this.wantCurrentPage(c)
                },
                wantCurrentPage: function(c) {
                    o(c);
                    var b = c.getAttr(m), e = c.getAttr("show_partial_next") ? b + 1: b;
                    this.want(c, (c.getAttr("pageNumber") - 1) * b, e)
                },
                want: function(c, b, e) {
                    if (!c.getAttr(g)) {
                        o(c);
                        var a = c.getAttr("ajax"), i = c.getAttr(n);
                        if (a.url) {
                            var j = c.getAttr("fetchedItems"), k = a.id_list;
                            k || (k = []);
                            var m = b>-1 ? b: 0, b = b + e - 1, p = a.params || {}, y = [], t = [];
                            if (i === 0)
                                k.length && (i = k), p.needSetSize = "true", c.setAttr(g, !0);
                            for (e===-1 && i && (b = i); m <= b && m < i;)
                                j[m] || ((e = k[m]) && y.push(e), t.push(m), j[m]=!1), m++;
                            c.setAttr("fetchedItems", j, {
                                silent: !0
                            });
                            p.count = t.length;
                            p.offset = t[0] || 0;
                            y.length > 0 && (p[a.id_param_name] = y.join(","));
                            (t.length > 0 || p.needSetSize) && this.getItems(c, a.url, p)
                        }
                    }
                },
                init: function(f) {
                    var b = f.getAttr("ajax");
                    if (!c.isFiniteNumber(b.fetch_delay))
                        b.fetch_delay = 500;
                    b.id_param_name = b.id_param_name || "ids";
                    b.prefetch_next_page = b.prefetch_next_page === i?!0 : !!b.prefetch_next_page;
                    f.setAttr("ajax", b);
                    f.getAttr(n) || this.want(f, 0, - 1)
                },
                afterInit: function(c) {
                    c.strategies.ajax.wantCurrentPage(c);
                    c.onChange("pageNumber", function(b, e) {
                        b > e ? c.strategies.ajax.wantNextPage(c) : c.strategies.ajax.wantPrevPage(c)
                    })
                }
            }
        });
        p.when("A", "a-carousel-ajax-standard").register("a-carousel-strategies-ajax", function(c, k) {
            return {
                standard: k,
                none: {
                    wantNextPage: c.constants.NOOP,
                    wantPrevPage: c.constants.NOOP,
                    wantCurrentPage: c.constants.NOOP,
                    want: c.constants.NOOP,
                    init: c.constants.NOOP
                },
                "default": "standard"
            }
        });
        p.when("a-carousel-strategies-display", "a-carousel-strategies-transition", "a-carousel-strategies-ajax").register("a-carousel-strategies", function(c, k, o) {
            return {
                display: c,
                transition: k,
                ajax: o
            }
        });
        p.when("A", "jQuery", "a-carousel-classes", "a-carousel-strategies").register("a-carousel-framework", function(c, k, o, i) {
            function m(a, b, e, f) {
                b = new b(a,
                e, f);
                b.__id=++v;
                a.data("a-carousel", b);
                a.removeClass("a-carousel-static");
                c.onScreen(a) ? c.delay(j, 10, b) : s.push(b);
                f.name && (w[f.name] = b)
            }
            function j(a) {
                for (var b = s.length; b--;)
                    if (s[b] === a) {
                        s.splice(b, 1);
                        break
                    }
                a.init();
                q.push(a);
                a.__initialized=!0;
                a.dom.$container.addClass("a-carousel-initialized");
                (b = a.getAttr("name")) && t[b] && c.each(t[b], function(b) {
                    b(a)
                })
            }
            function n(a, b) {
                var c = b[a + "Strategy"];
                c || (c = i[a]["default"]);
                return i[a][c]
            }
            function g(a) {
                for (var b = a.length, c; b--;)
                    if (c = a[b], !c.dom.$container.length ||
                    !y.find(c.dom.$container).length)(c = c.getAttr("name")) && delete w[c], a.splice(b, 1)
                    }
            function f() {
                g(s);
                g(q)
            }
            function b(a) {
                var b = a.data("a-carousel-options") || {};
                b.displayStrategy = a.data("a-display-strategy");
                b.transitionStrategy = a.data("a-transition-strategy");
                b.ajaxStrategy = a.data("a-ajax-strategy");
                b.carouselClass = a.data("a-class");
                var a = n("display", b), c = n("transition", b), e = n("ajax", b), f = b.carouselClass;
                f || (f = o["default"]);
                f = o[f];
                if (!(f === l || a === l || c === l || e === l))
                    return {
                        carouselClass: f,
                        strategies: {
                            display: a,
                            transition: c,
                            ajax: e
                        },
                        opts: b
                    }
            }
            function e() {
                k(".a-carousel-static").each(function() {
                    var a = k(this), c = b(a);
                    c && m(a, c.carouselClass, c.strategies, c.opts)
                })
            }
            function a() {
                c.each(s, function(a) {
                    c.onScreen(a.dom.$container) && j(a)
                })
            }
            var l, q = [], s = [], x=!1, w = {}, y = k(document), t = {}, v = 0;
            c.on("resize orientationchange", function(a, b) {
                f();
                if (b.height || b.width)
                    c.delay(function() {
                        c.each(q, function(a) {
                            a.resize()
                        })
                    }, c.capabilities.mobile || c.capabilities.tablet ? 100 : 0)
            });
            c.on("a:popover:afterSlideOut", function() {
                c.each(q, function(a) {
                    a.resize()
                })
            });
            c.on("a:carousel:change:name", function(a) {
                if (a.newValue)
                    w[a.newValue] = a.carousel;
                a.oldValue && delete w[a.oldValue]
            });
            c.on("a:pageUpdate beforeReady", e);
            c.on("a:pageUpdate", f);
            c.on("scroll", function() {
                e();
                a()
            });
            c.declarative("a-tabs", "click", function() {
                c.delay(a, 50)
            });
            c.on("a:popover:afterShow", function() {
                c.delay(a, 50)
            });
            c.on("a:popover:ajaxContentLoaded", function() {
                c.delay(function() {
                    f();
                    e()
                }, 50)
            });
            c.on.ready(function() {
                x=!0
            });
            var z = {
                getCarousel: function(a) {
                    a.jquery || (a = k(a));
                    var c = a.closest(".a-carousel-container").data("a-carousel");
                    if (!c) {
                        var e = b(a);
                        e && (c = m(a, e.carouselClass, e.strategies, e.opts))
                    }
                    return c
                },
                getCarouselByName: function(a) {
                    return w[a]
                },
                createAll: function() {
                    f();
                    e()
                },
                initializeAll: function() {
                    f();
                    a()
                },
                kill: function(a) {
                    a.jquery || (a = k(a));
                    if (a.length && (a = a.closest(".a-carousel-container"), a.length)) {
                        var b = a.data("a-carousel");
                        if (b) {
                            var e = c.indexOfArray(q, b);
                            e>-1 ? (q[e].name && delete w[q[e].name], q.splice(e, 1)) : (e = c.indexOfArray(s, b), e>-1 && (s[e].name && delete w[s[e].name], s.splice(e, 1)))
                        }
                        a.remove()
                    }
                },
                registerStrategy: function(a,
                b, c) {
                    if (!i[a])
                        i.type = {};
                    i[a][b] && p.error("Attempted to register a " + a + " strategy which already exists: " + b, "a-carousel-framework", "registerStrategy");
                    i[a][b] = c;
                    x && e()
                },
                registerCarouselClass: function(a, b) {
                    o[a] && p.error("Attempted to register a carousel class which already exists: " + a, "a-carousel-framework", "registerCarouselClass");
                    k.isFunction(b) || p.error("Attempted to register carousel class " + a + " without a constructor function.", "a-carousel-framework", "registerCarouselClass");
                    o[a] = b;
                    x && e()
                },
                getAllCarousels: function() {
                    return q.concat(s)
                },
                onInit: function(a, b) {
                    if (a && (t[a] || (t[a] = []), k.isFunction(b))) {
                        t[a].push(b);
                        var c = w[a];
                        c && c.__initialized && b(c)
                    }
                }
            };
            Object.freeze !== l && Object.freeze(z);
            return z
        })
    })
})(function() {
    var p = window.AmazonUIPageJS || P, c = p.attributeErrors;
    return c ? c("AmazonUICarousel") : p
}());
(function(p) {
    p.execute(function() {})
})(function() {
    var p = window.AmazonUIPageJS || P, c = p.attributeErrors;
    return c ? c("AmazonUI") : p
}());
