if (!this.JSON) {
    JSON = function() {
        function f(n) {
            return n < 10 ? "0" + n : n
        }
        Date.prototype.toJSON = function() {
            return this.getUTCFullYear() + "-" + f(this.getUTCMonth() + 1) + "-" + f(this.getUTCDate()) + "T" + f(this.getUTCHours()) + ":" + f(this.getUTCMinutes()) + ":" + f(this.getUTCSeconds()) + "Z"
        };
        var m = {
            "\b": "\\b",
            "\t": "\\t",
            "\n": "\\n",
            "\f": "\\f",
            "\r": "\\r",
            '"': '\\"',
            "\\": "\\\\"
        };
        function stringify(value, whitelist) {
            var a, i, k, l, v;
            switch (typeof value) {
            case"string":
                return /["\\\x00-\x1f]/.test(value) ? '"' + value.replace(/[\x00-\x1f\\"]/g, function(a) {
                    var c = m[a];
                    if (c) {
                        return c
                    }
                    c = a.charCodeAt();
                    return "\\u00" + Math.floor(c / 16).toString(16) + (c%16).toString(16)
                }) + '"' : '"' + value + '"';
            case"number":
                return isFinite(value) ? String(value) : "null";
            case"boolean":
                return String(value);
            case"null":
                return "null";
            case"object":
                if (!value) {
                    return "null"
                }
                if (typeof value.toJSON === "function") {
                    return stringify(value.toJSON())
                }
                a = [];
                if (typeof value.length === "number"&&!(value.propertyIsEnumerable("length"))) {
                    l = value.length;
                    for (i = 0; i < l; i += 1) {
                        a.push(stringify(value[i], whitelist) || "null")
                    }
                    return "[" + a.join(",") + "]"
                }
                if (whitelist) {
                    l = whitelist.length;
                    for (i = 0; i < l; i += 1) {
                        k = whitelist[i];
                        if (typeof k === "string") {
                            v = stringify(value[k], whitelist);
                            if (v) {
                                a.push(stringify(k) + ":" + v)
                            }
                        }
                    }
                } else {
                    for (k in value) {
                        if (typeof k === "string") {
                            v = stringify(value[k], whitelist);
                            if (v) {
                                a.push(stringify(k) + ":" + v)
                            }
                        }
                    }
                }
                return "{" + a.join(",") + "}"
            }
        }
        return {
            stringify: stringify,
            parse: function(text, filter) {
                var j;
                function walk(k, v) {
                    var i, n;
                    if (v && typeof v === "object") {
                        for (i in v) {
                            if (Object.prototype.hasOwnProperty.apply(v, [i])) {
                                n = walk(i, v[i]);
                                if (n !== undefined) {
                                    v[i] = n
                                }
                            }
                        }
                    }
                    return filter(k, v)
                }
                if (/^[\],:{}\s]*$/.test(text.replace(/\\./g, "@").replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(:?[eE][+\-]?\d+)?/g, "]").replace(/(?:^|:|,)(?:\s*\[)+/g, ""))) {
                    j = eval("(" + text + ")");
                    return typeof filter === "function" ? walk("", j) : j
                }
                throw new SyntaxError("parseJSON")
            }
        }
    }()
}
Scroller = {
    speed: 10,
    gy: function(b) {
        gy = b.offsetTop;
        if (b.offsetParent) {
            while (b = b.offsetParent) {
                gy += b.offsetTop
            }
        }
        return gy
    },
    scrollTop: function() {
        body = document.body;
        d = document.documentElement;
        if (body && body.scrollTop) {
            return body.scrollTop
        }
        if (d && d.scrollTop) {
            return d.scrollTop
        }
        if (window.pageYOffset) {
            return window.pageYOffset
        }
        return 0
    },
    add: function(e, b, c) {
        if (e.addEventListener) {
            return e.addEventListener(b, c, false)
        }
        if (e.attachEvent) {
            return e.attachEvent("on" + b, c)
        }
    },
    end: function(b) {
        if (window.event) {
            window.event.cancelBubble = true;
            window.event.returnValue = false;
            return 
        }
        if (b.preventDefault && b.stopPropagation) {
            b.preventDefault();
            b.stopPropagation()
        }
    },
    scroll: function(b) {
        i = window.innerHeight || document.documentElement.clientHeight;
        h = document.body.scrollHeight;
        a = Scroller.scrollTop();
        if (b > a) {
            if (h - b > i) {
                a += Math.ceil((b - a) / Scroller.speed)
            } else {
                a += Math.ceil((b - a - (h - b)) / Scroller.speed)
            }
        } else {
            a = a + (b - a) / Scroller.speed
        }
        window.scrollTo(0, a);
        if (a == b || Scroller.offsetTop == a) {
            clearInterval(Scroller.interval)
        }
        Scroller.offsetTop = a
    },
    init: function() {
        Scroller.add(window, "load", Scroller.render)
    },
    render: function() {
        a = document.getElementsByTagName("a");
        Scroller.end(this);
        window.onscroll;
        for (i = 0; i < a.length; i++) {
            l = a[i];
            if (l.href && l.href.indexOf("#")!=-1 && ((l.pathname == location.pathname) || ("/" + l.pathname == location.pathname))) {
                Scroller.add(l, "click", Scroller.end);
                l.onclick = function() {
                    Scroller.end(this);
                    l = this.hash.substr(1);
                    a = document.getElementsByTagName("a");
                    for (i = 0; i < a.length; i++) {
                        if (a[i].name == l) {
                            clearInterval(Scroller.interval);
                            Scroller.interval = setInterval("Scroller.scroll(" + Scroller.gy(a[i]) + ")", 10)
                        }
                    }
                }
            }
        }
    }
};
Scroller.init();
/*! jQuery v1.6.4 http://jquery.com/ | http://jquery.org/license */
(function(a, b) {
    function cu(a) {
        return f.isWindow(a) ? a : a.nodeType === 9 ? a.defaultView || a.parentWindow : !1
    }
    function cr(a) {
        if (!cg[a]) {
            var b = c.body, d = f("<" + a + ">").appendTo(b), e = d.css("display");
            d.remove();
            if (e === "none" || e === "") {
                ch || (ch = c.createElement("iframe"), ch.frameBorder = ch.width = ch.height = 0), b.appendChild(ch);
                if (!ci ||!ch.createElement) {
                    ci = (ch.contentWindow || ch.contentDocument).document, ci.write((c.compatMode === "CSS1Compat" ? "<!doctype html>" : "") + "<html><body>"), ci.close()
                }
                d = ci.createElement(a), ci.body.appendChild(d), e = f.css(d, "display"), b.removeChild(ch)
            }
            cg[a] = e
        }
        return cg[a]
    }
    function cq(a, b) {
        var c = {};
        f.each(cm.concat.apply([], cm.slice(0, b)), function() {
            c[this] = a
        });
        return c
    }
    function cp() {
        cn = b
    }
    function co() {
        setTimeout(cp, 0);
        return cn = f.now()
    }
    function cf() {
        try {
            return new a.ActiveXObject("Microsoft.XMLHTTP")
        } catch (b) {}
    }
    function ce() {
        try {
            return new a.XMLHttpRequest
        } catch (b) {}
    }
    function b$(a, c) {
        a.dataFilter && (c = a.dataFilter(c, a.dataType));
        var d = a.dataTypes, e = {}, g, h, i = d.length, j, k = d[0], l, m, n, o, p;
        for (g = 1; g < i; g++) {
            if (g === 1) {
                for (h in a.converters) {
                    typeof h == "string" && (e[h.toLowerCase()] = a.converters[h])
                }
            }
            l = k, k = d[g];
            if (k === "*") {
                k = l
            } else {
                if (l !== "*" && l !== k) {
                    m = l + " " + k, n = e[m] || e["* " + k];
                    if (!n) {
                        p = b;
                        for (o in e) {
                            j = o.split(" ");
                            if (j[0] === l || j[0] === "*") {
                                p = e[j[1] + " " + k];
                                if (p) {
                                    o = e[o], o===!0 ? n = p : p===!0 && (n = o);
                                    break
                                }
                            }
                        }
                    }
                    !n&&!p && f.error("No conversion from " + m.replace(" ", " to ")), n!==!0 && (c = n ? n(c) : p(o(c)))
                }
            }
        }
        return c
    }
    function bZ(a, c, d) {
        var e = a.contents, f = a.dataTypes, g = a.responseFields, h, i, j, k;
        for (i in g) {
            i in d && (c[g[i]] = d[i])
        }
        while (f[0] === "*") {
            f.shift(), h === b && (h = a.mimeType || c.getResponseHeader("content-type"))
        }
        if (h) {
            for (i in e) {
                if (e[i] && e[i].test(h)) {
                    f.unshift(i);
                    break
                }
            }
        }
        if (f[0] in d) {
            j = f[0]
        } else {
            for (i in d) {
                if (!f[0] || a.converters[i + " " + f[0]]) {
                    j = i;
                    break
                }
                k || (k = i)
            }
            j = j || k
        }
        if (j) {
            j !== f[0] && f.unshift(j);
            return d[j]
        }
    }
    function bY(a, b, c, d) {
        if (f.isArray(b)) {
            f.each(b, function(b, e) {
                c || bA.test(a) ? d(a, e) : bY(a + "[" + (typeof e == "object" || f.isArray(e) ? b : "") + "]", e, c, d)
            })
        } else {
            if (!c && b != null && typeof b == "object") {
                for (var e in b) {
                    bY(a + "[" + e + "]", b[e], c, d)
                }
            } else {
                d(a, b)
            }
        }
    }
    function bX(a, c) {
        var d, e, g = f.ajaxSettings.flatOptions || {};
        for (d in c) {
            c[d] !== b && ((g[d] ? a : e || (e = {}))[d] = c[d])
        }
        e && f.extend(!0, a, e)
    }
    function bW(a, c, d, e, f, g) {
        f = f || c.dataTypes[0], g = g || {}, g[f]=!0;
        var h = a[f], i = 0, j = h ? h.length: 0, k = a === bP, l;
        for (; i < j && (k ||!l); i++) {
            l = h[i](c, d, e), typeof l == "string" && (!k || g[l] ? l = b : (c.dataTypes.unshift(l), l = bW(a, c, d, e, l, g)))
        }(k ||!l)&&!g["*"] && (l = bW(a, c, d, e, "*", g));
        return l
    }
    function bV(a) {
        return function(b, c) {
            typeof b != "string" && (c = b, b = "*");
            if (f.isFunction(c)) {
                var d = b.toLowerCase().split(bL), e = 0, g = d.length, h, i, j;
                for (; e < g; e++) {
                    h = d[e], j = /^\+/.test(h), j && (h = h.substr(1) || "*"), i = a[h] = a[h] || [], i[j ? "unshift": "push"](c)
                }
            }
        }
    }
    function by(a, b, c) {
        var d = b === "width" ? a.offsetWidth: a.offsetHeight, e = b === "width" ? bt: bu;
        if (d > 0) {
            c !== "border" && f.each(e, function() {
                c || (d -= parseFloat(f.css(a, "padding" + this)) || 0), c === "margin" ? d += parseFloat(f.css(a, c + this)) || 0 : d -= parseFloat(f.css(a, "border" + this + "Width")) || 0
            });
            return d + "px"
        }
        d = bv(a, b, b);
        if (d < 0 || d == null) {
            d = a.style[b] || 0
        }
        d = parseFloat(d) || 0, c && f.each(e, function() {
            d += parseFloat(f.css(a, "padding" + this)) || 0, c !== "padding" && (d += parseFloat(f.css(a, "border" + this + "Width")) || 0), c === "margin" && (d += parseFloat(f.css(a, c + this)) || 0)
        });
        return d + "px"
    }
    function bl(a, b) {
        b.src ? f.ajax({
            url: b.src,
            async: !1,
            dataType: "script"
        }) : f.globalEval((b.text || b.textContent || b.innerHTML || "").replace(bd, "/*$0*/")), b.parentNode && b.parentNode.removeChild(b)
    }
    function bk(a) {
        f.nodeName(a, "input") ? bj(a) : "getElementsByTagName" in a && f.grep(a.getElementsByTagName("input"), bj)
    }
    function bj(a) {
        if (a.type === "checkbox" || a.type === "radio") {
            a.defaultChecked = a.checked
        }
    }
    function bi(a) {
        return "getElementsByTagName" in a ? a.getElementsByTagName("*") : "querySelectorAll" in a ? a.querySelectorAll("*") : []
    }
    function bh(a, b) {
        var c;
        if (b.nodeType === 1) {
            b.clearAttributes && b.clearAttributes(), b.mergeAttributes && b.mergeAttributes(a), c = b.nodeName.toLowerCase();
            if (c === "object") {
                b.outerHTML = a.outerHTML
            } else {
                if (c !== "input" || a.type !== "checkbox" && a.type !== "radio") {
                    if (c === "option") {
                        b.selected = a.defaultSelected
                    } else {
                        if (c === "input" || c === "textarea") {
                            b.defaultValue = a.defaultValue
                        }
                    }
                } else {
                    a.checked && (b.defaultChecked = b.checked = a.checked), b.value !== a.value && (b.value = a.value)
                }
            }
            b.removeAttribute(f.expando)
        }
    }
    function bg(a, b) {
        if (b.nodeType === 1&&!!f.hasData(a)) {
            var c = f.expando, d = f.data(a), e = f.data(b, d);
            if (d = d[c]) {
                var g = d.events;
                e = e[c] = f.extend({}, d);
                if (g) {
                    delete e.handle, e.events = {};
                    for (var h in g) {
                        for (var i = 0, j = g[h].length; i < j; i++) {
                            f.event.add(b, h + (g[h][i].namespace ? "." : "") + g[h][i].namespace, g[h][i], g[h][i].data)
                        }
                    }
                }
            }
        }
    }
    function bf(a, b) {
        return f.nodeName(a, "table") ? a.getElementsByTagName("tbody")[0] || a.appendChild(a.ownerDocument.createElement("tbody")) : a
    }
    function V(a, b, c) {
        b = b || 0;
        if (f.isFunction(b)) {
            return f.grep(a, function(a, d) {
                var e=!!b.call(a, d, a);
                return e === c
            })
        }
        if (b.nodeType) {
            return f.grep(a, function(a, d) {
                return a === b === c
            })
        }
        if (typeof b == "string") {
            var d = f.grep(a, function(a) {
                return a.nodeType === 1
            });
            if (Q.test(b)) {
                return f.filter(b, d, !c)
            }
            b = f.filter(b, d)
        }
        return f.grep(a, function(a, d) {
            return f.inArray(a, b) >= 0 === c
        })
    }
    function U(a) {
        return !a ||!a.parentNode || a.parentNode.nodeType === 11
    }
    function M(a, b) {
        return (a && a !== "*" ? a + "." : "") + b.replace(y, "`").replace(z, "&")
    }
    function L(a) {
        var b, c, d, e, g, h, i, j, k, l, m, n, o, p = [], q = [], r = f._data(this, "events");
        if (!(a.liveFired === this ||!r ||!r.live || a.target.disabled || a.button && a.type === "click")) {
            a.namespace && (n = new RegExp("(^|\\.)" + a.namespace.split(".").join("\\.(?:.*\\.)?") + "(\\.|$)")), a.liveFired = this;
            var s = r.live.slice(0);
            for (i = 0; i < s.length; i++) {
                g = s[i], g.origType.replace(w, "") === a.type ? q.push(g.selector) : s.splice(i--, 1)
            }
            e = f(a.target).closest(q, a.currentTarget);
            for (j = 0, k = e.length; j < k; j++) {
                m = e[j];
                for (i = 0; i < s.length; i++) {
                    g = s[i];
                    if (m.selector === g.selector && (!n || n.test(g.namespace))&&!m.elem.disabled) {
                        h = m.elem, d = null;
                        if (g.preType === "mouseenter" || g.preType === "mouseleave") {
                            a.type = g.preType, d = f(a.relatedTarget).closest(g.selector)[0], d && f.contains(h, d) && (d = h)
                        }(!d || d !== h) && p.push({
                            elem: h,
                            handleObj: g,
                            level: m.level
                        })
                    }
                }
            }
            for (j = 0, k = p.length; j < k; j++) {
                e = p[j];
                if (c && e.level > c) {
                    break
                }
                a.currentTarget = e.elem, a.data = e.handleObj.data, a.handleObj = e.handleObj, o = e.handleObj.origHandler.apply(e.elem, arguments);
                if (o===!1 || a.isPropagationStopped()) {
                    c = e.level, o===!1 && (b=!1);
                    if (a.isImmediatePropagationStopped()) {
                        break
                    }
                }
            }
            return b
        }
    }
    function J(a, c, d) {
        var e = f.extend({}, d[0]);
        e.type = a, e.originalEvent = {}, e.liveFired = b, f.event.handle.call(c, e), e.isDefaultPrevented() && d[0].preventDefault()
    }
    function D() {
        return !0
    }
    function C() {
        return !1
    }
    function m(a, c, d) {
        var e = c + "defer", g = c + "queue", h = c + "mark", i = f.data(a, e, b, !0);
        i && (d === "queue" ||!f.data(a, g, b, !0)) && (d === "mark" ||!f.data(a, h, b, !0)) && setTimeout(function() {
            !f.data(a, g, b, !0)&&!f.data(a, h, b, !0) && (f.removeData(a, e, !0), i.resolve())
        }, 0)
    }
    function l(a) {
        for (var b in a) {
            if (b !== "toJSON") {
                return !1
            }
        }
        return !0
    }
    function k(a, c, d) {
        if (d === b && a.nodeType === 1) {
            var e = "data-" + c.replace(j, "-$1").toLowerCase();
            d = a.getAttribute(e);
            if (typeof d == "string") {
                try {
                    d = d === "true"?!0 : d === "false"?!1 : d === "null" ? null : f.isNaN(d) ? i.test(d) ? f.parseJSON(d) : d : parseFloat(d)
                } catch (g) {}
                f.data(a, c, d)
            } else {
                d = b
            }
        }
        return d
    }
    var c = a.document, d = a.navigator, e = a.location, f = function() {
        function K() {
            if (!e.isReady) {
                try {
                    c.documentElement.doScroll("left")
                } catch (a) {
                    setTimeout(K, 1);
                    return 
                }
                e.ready()
            }
        }
        var e = function(a, b) {
            return new e.fn.init(a, b, h)
        }, f = a.jQuery, g = a.$, h, i = /^(?:[^#<]*(<[\w\W]+>)[^>]*$|#([\w\-]*)$)/, j = /\S/, k = /^\s+/, l = /\s+$/, m = /\d/, n = /^<(\w+)\s*\/?>(?:<\/\1>)?$/, o = /^[\],:{}\s]*$/, p = /\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, q = /"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, r = /(?:^|:|,)(?:\s*\[)+/g, s = /(webkit)[ \/]([\w.]+)/, t = /(opera)(?:.*version)?[ \/]([\w.]+)/, u = /(msie) ([\w.]+)/, v = /(mozilla)(?:.*? rv:([\w.]+))?/, w = /-([a-z]|[0-9])/ig, x = /^-ms-/, y = function(a, b) {
            return (b + "").toUpperCase()
        }, z = d.userAgent, A, B, C, D = Object.prototype.toString, E = Object.prototype.hasOwnProperty, F = Array.prototype.push, G = Array.prototype.slice, H = String.prototype.trim, I = Array.prototype.indexOf, J = {};
        e.fn = e.prototype = {
            constructor: e,
            init: function(a, d, f) {
                var g, h, j, k;
                if (!a) {
                    return this
                }
                if (a.nodeType) {
                    this.context = this[0] = a, this.length = 1;
                    return this
                }
                if (a === "body"&&!d && c.body) {
                    this.context = c, this[0] = c.body, this.selector = a, this.length = 1;
                    return this
                }
                if (typeof a == "string") {
                    a.charAt(0) !== "<" || a.charAt(a.length - 1) !== ">" || a.length < 3 ? g = i.exec(a) : g = [null, a, null];
                    if (g && (g[1] ||!d)) {
                        if (g[1]) {
                            d = d instanceof e ? d[0] : d, k = d ? d.ownerDocument || d : c, j = n.exec(a), j ? e.isPlainObject(d) ? (a = [c.createElement(j[1])], e.fn.attr.call(a, d, !0)) : a = [k.createElement(j[1])] : (j = e.buildFragment([g[1]], [k]), a = (j.cacheable ? e.clone(j.fragment) : j.fragment).childNodes);
                            return e.merge(this, a)
                        }
                        h = c.getElementById(g[2]);
                        if (h && h.parentNode) {
                            if (h.id !== g[2]) {
                                return f.find(a)
                            }
                            this.length = 1, this[0] = h
                        }
                        this.context = c, this.selector = a;
                        return this
                    }
                    return !d || d.jquery ? (d || f).find(a) : this.constructor(d).find(a)
                }
                if (e.isFunction(a)) {
                    return f.ready(a)
                }
                a.selector !== b && (this.selector = a.selector, this.context = a.context);
                return e.makeArray(a, this)
            },
            selector: "",
            jquery: "1.6.4",
            length: 0,
            size: function() {
                return this.length
            },
            toArray: function() {
                return G.call(this, 0)
            },
            get: function(a) {
                return a == null ? this.toArray() : a < 0 ? this[this.length + a] : this[a]
            },
            pushStack: function(a, b, c) {
                var d = this.constructor();
                e.isArray(a) ? F.apply(d, a) : e.merge(d, a), d.prevObject = this, d.context = this.context, b === "find" ? d.selector = this.selector + (this.selector ? " " : "") + c : b && (d.selector = this.selector + "." + b + "(" + c + ")");
                return d
            },
            each: function(a, b) {
                return e.each(this, a, b)
            },
            ready: function(a) {
                e.bindReady(), B.done(a);
                return this
            },
            eq: function(a) {
                return a===-1 ? this.slice(a) : this.slice(a, + a + 1)
            },
            first: function() {
                return this.eq(0)
            },
            last: function() {
                return this.eq( - 1)
            },
            slice: function() {
                return this.pushStack(G.apply(this, arguments), "slice", G.call(arguments).join(","))
            },
            map: function(a) {
                return this.pushStack(e.map(this, function(b, c) {
                    return a.call(b, c, b)
                }))
            },
            end: function() {
                return this.prevObject || this.constructor(null)
            },
            push: F,
            sort: [].sort,
            splice: [].splice
        }, e.fn.init.prototype = e.fn, e.extend = e.fn.extend = function() {
            var a, c, d, f, g, h, i = arguments[0] || {}, j = 1, k = arguments.length, l=!1;
            typeof i == "boolean" && (l = i, i = arguments[1] || {}, j = 2), typeof i != "object"&&!e.isFunction(i) && (i = {}), k === j && (i = this, --j);
            for (; j < k; j++) {
                if ((a = arguments[j]) != null) {
                    for (c in a) {
                        d = i[c], f = a[c];
                        if (i === f) {
                            continue
                        }
                        l && f && (e.isPlainObject(f) || (g = e.isArray(f))) ? (g ? (g=!1, h = d && e.isArray(d) ? d : []) : h = d && e.isPlainObject(d) ? d : {}, i[c] = e.extend(l, h, f)) : f !== b && (i[c] = f)
                    }
                }
            }
            return i
        }, e.extend({
            noConflict: function(b) {
                a.$ === e && (a.$ = g), b && a.jQuery === e && (a.jQuery = f);
                return e
            },
            isReady: !1,
            readyWait: 1,
            holdReady: function(a) {
                a ? e.readyWait++ : e.ready(!0)
            },
            ready: function(a) {
                if (a===!0&&!--e.readyWait || a!==!0&&!e.isReady) {
                    if (!c.body) {
                        return setTimeout(e.ready, 1)
                    }
                    e.isReady=!0;
                    if (a!==!0&&--e.readyWait > 0) {
                        return 
                    }
                    B.resolveWith(c, [e]), e.fn.trigger && e(c).trigger("ready").unbind("ready")
                }
            },
            bindReady: function() {
                if (!B) {
                    B = e._Deferred();
                    if (c.readyState === "complete") {
                        return setTimeout(e.ready, 1)
                    }
                    if (c.addEventListener) {
                        c.addEventListener("DOMContentLoaded", C, !1), a.addEventListener("load", e.ready, !1)
                    } else {
                        if (c.attachEvent) {
                            c.attachEvent("onreadystatechange", C), a.attachEvent("onload", e.ready);
                            var b=!1;
                            try {
                                b = a.frameElement == null
                            } catch (d) {}
                            c.documentElement.doScroll && b && K()
                        }
                    }
                }
            },
            isFunction: function(a) {
                return e.type(a) === "function"
            },
            isArray: Array.isArray || function(a) {
                return e.type(a) === "array"
            },
            isWindow: function(a) {
                return a && typeof a == "object" && "setInterval" in a
            },
            isNaN: function(a) {
                return a == null ||!m.test(a) || isNaN(a)
            },
            type: function(a) {
                return a == null ? String(a) : J[D.call(a)] || "object"
            },
            isPlainObject: function(a) {
                if (!a || e.type(a) !== "object" || a.nodeType || e.isWindow(a)) {
                    return !1
                }
                try {
                    if (a.constructor&&!E.call(a, "constructor")&&!E.call(a.constructor.prototype, "isPrototypeOf")) {
                        return !1
                    }
                } catch (c) {
                    return !1
                }
                var d;
                for (d in a) {}
                return d === b || E.call(a, d)
            },
            isEmptyObject: function(a) {
                for (var b in a) {
                    return !1
                }
                return !0
            },
            error: function(a) {
                throw a
            },
            parseJSON: function(b) {
                if (typeof b != "string" ||!b) {
                    return null
                }
                b = e.trim(b);
                if (a.JSON && a.JSON.parse) {
                    return a.JSON.parse(b)
                }
                if (o.test(b.replace(p, "@").replace(q, "]").replace(r, ""))) {
                    return (new Function("return " + b))()
                }
                e.error("Invalid JSON: " + b)
            },
            parseXML: function(c) {
                var d, f;
                try {
                    a.DOMParser ? (f = new DOMParser, d = f.parseFromString(c, "text/xml")) : (d = new ActiveXObject("Microsoft.XMLDOM"), d.async = "false", d.loadXML(c))
                } catch (g) {
                    d = b
                }(!d ||!d.documentElement || d.getElementsByTagName("parsererror").length) && e.error("Invalid XML: " + c);
                return d
            },
            noop: function() {},
            globalEval: function(b) {
                b && j.test(b) && (a.execScript || function(b) {
                    a.eval.call(a, b)
                })(b)
            },
            camelCase: function(a) {
                return a.replace(x, "ms-").replace(w, y)
            },
            nodeName: function(a, b) {
                return a.nodeName && a.nodeName.toUpperCase() === b.toUpperCase()
            },
            each: function(a, c, d) {
                var f, g = 0, h = a.length, i = h === b || e.isFunction(a);
                if (d) {
                    if (i) {
                        for (f in a) {
                            if (c.apply(a[f], d)===!1) {
                                break
                            }
                        }
                    } else {
                        for (; g < h;) {
                            if (c.apply(a[g++], d)===!1) {
                                break
                            }
                        }
                    }
                } else {
                    if (i) {
                        for (f in a) {
                            if (c.call(a[f], f, a[f])===!1) {
                                break
                            }
                        }
                    } else {
                        for (; g < h;) {
                            if (c.call(a[g], g, a[g++])===!1) {
                                break
                            }
                        }
                    }
                }
                return a
            },
            trim: H ? function(a) {
                return a == null ? "" : H.call(a)
            }
            : function(a) {
                return a == null ? "" : (a + "").replace(k, "").replace(l, "")
            },
            makeArray: function(a, b) {
                var c = b || [];
                if (a != null) {
                    var d = e.type(a);
                    a.length == null || d === "string" || d === "function" || d === "regexp" || e.isWindow(a) ? F.call(c, a) : e.merge(c, a)
                }
                return c
            },
            inArray: function(a, b) {
                if (!b) {
                    return - 1
                }
                if (I) {
                    return I.call(b, a)
                }
                for (var c = 0, d = b.length; c < d; c++) {
                    if (b[c] === a) {
                        return c
                    }
                }
                return - 1
            },
            merge: function(a, c) {
                var d = a.length, e = 0;
                if (typeof c.length == "number") {
                    for (var f = c.length; e < f; e++) {
                        a[d++] = c[e]
                    }
                } else {
                    while (c[e] !== b) {
                        a[d++] = c[e++]
                    }
                }
                a.length = d;
                return a
            },
            grep: function(a, b, c) {
                var d = [], e;
                c=!!c;
                for (var f = 0, g = a.length; f < g; f++) {
                    e=!!b(a[f], f), c !== e && d.push(a[f])
                }
                return d
            },
            map: function(a, c, d) {
                var f, g, h = [], i = 0, j = a.length, k = a instanceof e || j !== b && typeof j == "number" && (j > 0 && a[0] && a[j - 1] || j === 0 || e.isArray(a));
                if (k) {
                    for (; i < j; i++) {
                        f = c(a[i], i, d), f != null && (h[h.length] = f)
                    }
                } else {
                    for (g in a) {
                        f = c(a[g], g, d), f != null && (h[h.length] = f)
                    }
                }
                return h.concat.apply([], h)
            },
            guid: 1,
            proxy: function(a, c) {
                if (typeof c == "string") {
                    var d = a[c];
                    c = a, a = d
                }
                if (!e.isFunction(a)) {
                    return b
                }
                var f = G.call(arguments, 2), g = function() {
                    return a.apply(c, f.concat(G.call(arguments)))
                };
                g.guid = a.guid = a.guid || g.guid || e.guid++;
                return g
            },
            access: function(a, c, d, f, g, h) {
                var i = a.length;
                if (typeof c == "object") {
                    for (var j in c) {
                        e.access(a, j, c[j], f, g, d)
                    }
                    return a
                }
                if (d !== b) {
                    f=!h && f && e.isFunction(d);
                    for (var k = 0; k < i; k++) {
                        g(a[k], c, f ? d.call(a[k], k, g(a[k], c)) : d, h)
                    }
                    return a
                }
                return i ? g(a[0], c) : b
            },
            now: function() {
                return (new Date).getTime()
            },
            uaMatch: function(a) {
                a = a.toLowerCase();
                var b = s.exec(a) || t.exec(a) || u.exec(a) || a.indexOf("compatible") < 0 && v.exec(a) || [];
                return {
                    browser: b[1] || "",
                    version: b[2] || "0"
                }
            },
            sub: function() {
                function a(b, c) {
                    return new a.fn.init(b, c)
                }
                e.extend(!0, a, this), a.superclass = this, a.fn = a.prototype = this (), a.fn.constructor = a, a.sub = this.sub, a.fn.init = function(d, f) {
                    f && f instanceof e&&!(f instanceof a) && (f = a(f));
                    return e.fn.init.call(this, d, f, b)
                }, a.fn.init.prototype = a.fn;
                var b = a(c);
                return a
            },
            browser: {}
        }), e.each("Boolean Number String Function Array Date RegExp Object".split(" "), function(a, b) {
            J["[object " + b + "]"] = b.toLowerCase()
        }), A = e.uaMatch(z), A.browser && (e.browser[A.browser]=!0, e.browser.version = A.version), e.browser.webkit && (e.browser.safari=!0), j.test(" ") && (k = /^[\s\xA0]+/, l = /[\s\xA0]+$/), h = e(c), c.addEventListener ? C = function() {
            c.removeEventListener("DOMContentLoaded", C, !1), e.ready()
        } : c.attachEvent && (C = function() {
            c.readyState === "complete" && (c.detachEvent("onreadystatechange", C), e.ready())
        });
        return e
    }(), g = "done fail isResolved isRejected promise then always pipe".split(" "), h = [].slice;
    f.extend({
        _Deferred: function() {
            var a = [], b, c, d, e = {
                done: function() {
                    if (!d) {
                        var c = arguments, g, h, i, j, k;
                        b && (k = b, b = 0);
                        for (g = 0, h = c.length; g < h; g++) {
                            i = c[g], j = f.type(i), j === "array" ? e.done.apply(e, i) : j === "function" && a.push(i)
                        }
                        k && e.resolveWith(k[0], k[1])
                    }
                    return this
                },
                resolveWith: function(e, f) {
                    if (!d&&!b&&!c) {
                        f = f || [], c = 1;
                        try {
                            while (a[0]) {
                                a.shift().apply(e, f)
                            }
                        } finally {
                            b = [e, f], c = 0
                        }
                    }
                    return this
                },
                resolve: function() {
                    e.resolveWith(this, arguments);
                    return this
                },
                isResolved: function() {
                    return !!c||!!b
                },
                cancel: function() {
                    d = 1, a = [];
                    return this
                }
            };
            return e
        },
        Deferred: function(a) {
            var b = f._Deferred(), c = f._Deferred(), d;
            f.extend(b, {
                then: function(a, c) {
                    b.done(a).fail(c);
                    return this
                },
                always: function() {
                    return b.done.apply(b, arguments).fail.apply(this, arguments)
                },
                fail: c.done,
                rejectWith: c.resolveWith,
                reject: c.resolve,
                isRejected: c.isResolved,
                pipe: function(a, c) {
                    return f.Deferred(function(d) {
                        f.each({
                            done: [a, "resolve"],
                            fail: [c, "reject"]
                        }, function(a, c) {
                            var e = c[0], g = c[1], h;
                            f.isFunction(e) ? b[a](function() {
                                h = e.apply(this, arguments), h && f.isFunction(h.promise) ? h.promise().then(d.resolve, d.reject) : d[g + "With"](this === b ? d : this, [h])
                            }) : b[a](d[g])
                        })
                    }).promise()
                },
                promise: function(a) {
                    if (a == null) {
                        if (d) {
                            return d
                        }
                        d = a = {}
                    }
                    var c = g.length;
                    while (c--) {
                        a[g[c]] = b[g[c]]
                    }
                    return a
                }
            }), b.done(c.cancel).fail(b.cancel), delete b.cancel, a && a.call(b, b);
            return b
        },
        when: function(a) {
            function i(a) {
                return function(c) {
                    b[a] = arguments.length > 1 ? h.call(arguments, 0) : c, --e || g.resolveWith(g, h.call(b, 0))
                }
            }
            var b = arguments, c = 0, d = b.length, e = d, g = d <= 1 && a && f.isFunction(a.promise) ? a: f.Deferred();
            if (d > 1) {
                for (; c < d; c++) {
                    b[c] && f.isFunction(b[c].promise) ? b[c].promise().then(i(c), g.reject) : --e
                }
                e || g.resolveWith(g, b)
            } else {
                g !== a && g.resolveWith(g, d ? [a] : [])
            }
            return g.promise()
        }
    }), f.support = function() {
        var a = c.createElement("div"), b = c.documentElement, d, e, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u;
        a.setAttribute("className", "t"), a.innerHTML = "   <link/><table></table><a href='/a' style='top:1px;float:left;opacity:.55;'>a</a><input type='checkbox'/>", d = a.getElementsByTagName("*"), e = a.getElementsByTagName("a")[0];
        if (!d ||!d.length ||!e) {
            return {}
        }
        g = c.createElement("select"), h = g.appendChild(c.createElement("option")), i = a.getElementsByTagName("input")[0], k = {
            leadingWhitespace: a.firstChild.nodeType === 3,
            tbody: !a.getElementsByTagName("tbody").length,
            htmlSerialize: !!a.getElementsByTagName("link").length,
            style: /top/.test(e.getAttribute("style")),
            hrefNormalized: e.getAttribute("href") === "/a",
            opacity: /^0.55$/.test(e.style.opacity),
            cssFloat: !!e.style.cssFloat,
            checkOn: i.value === "on",
            optSelected: h.selected,
            getSetAttribute: a.className !== "t",
            submitBubbles: !0,
            changeBubbles: !0,
            focusinBubbles: !1,
            deleteExpando: !0,
            noCloneEvent: !0,
            inlineBlockNeedsLayout: !1,
            shrinkWrapBlocks: !1,
            reliableMarginRight: !0
        }, i.checked=!0, k.noCloneChecked = i.cloneNode(!0).checked, g.disabled=!0, k.optDisabled=!h.disabled;
        try {
            delete a.test
        } catch (v) {
            k.deleteExpando=!1
        }
        !a.addEventListener && a.attachEvent && a.fireEvent && (a.attachEvent("onclick", function() {
            k.noCloneEvent=!1
        }), a.cloneNode(!0).fireEvent("onclick")), i = c.createElement("input"), i.value = "t", i.setAttribute("type", "radio"), k.radioValue = i.value === "t", i.setAttribute("checked", "checked"), a.appendChild(i), l = c.createDocumentFragment(), l.appendChild(a.firstChild), k.checkClone = l.cloneNode(!0).cloneNode(!0).lastChild.checked, a.innerHTML = "", a.style.width = a.style.paddingLeft = "1px", m = c.getElementsByTagName("body")[0], o = c.createElement(m ? "div" : "body"), p = {
            visibility: "hidden",
            width: 0,
            height: 0,
            border: 0,
            margin: 0,
            background: "none"
        }, m && f.extend(p, {
            position: "absolute",
            left: "-1000px",
            top: "-1000px"
        });
        for (t in p) {
            o.style[t] = p[t]
        }
        o.appendChild(a), n = m || b, n.insertBefore(o, n.firstChild), k.appendChecked = i.checked, k.boxModel = a.offsetWidth === 2, "zoom" in a.style && (a.style.display = "inline", a.style.zoom = 1, k.inlineBlockNeedsLayout = a.offsetWidth === 2, a.style.display = "", a.innerHTML = "<div style='width:4px;'></div>", k.shrinkWrapBlocks = a.offsetWidth !== 2), a.innerHTML = "<table><tr><td style='padding:0;border:0;display:none'></td><td>t</td></tr></table>", q = a.getElementsByTagName("td"), u = q[0].offsetHeight === 0, q[0].style.display = "", q[1].style.display = "none", k.reliableHiddenOffsets = u && q[0].offsetHeight === 0, a.innerHTML = "", c.defaultView && c.defaultView.getComputedStyle && (j = c.createElement("div"), j.style.width = "0", j.style.marginRight = "0", a.appendChild(j), k.reliableMarginRight = (parseInt((c.defaultView.getComputedStyle(j, null) || {
            marginRight: 0
        }).marginRight, 10) || 0) === 0), o.innerHTML = "", n.removeChild(o);
        if (a.attachEvent) {
            for (t in {
                submit: 1,
                change: 1,
                focusin: 1
            }) {
                s = "on" + t, u = s in a, u || (a.setAttribute(s, "return;"), u = typeof a[s] == "function"), k[t + "Bubbles"] = u
            }
        }
        o = l = g = h = m = j = a = i = null;
        return k
    }(), f.boxModel = f.support.boxModel;
    var i = /^(?:\{.*\}|\[.*\])$/, j = /([A-Z])/g;
    f.extend({
        cache: {},
        uuid: 0,
        expando: "jQuery" + (f.fn.jquery + Math.random()).replace(/\D/g, ""),
        noData: {
            embed: !0,
            object: "clsid:D27CDB6E-AE6D-11cf-96B8-444553540000",
            applet: !0
        },
        hasData: function(a) {
            a = a.nodeType ? f.cache[a[f.expando]] : a[f.expando];
            return !!a&&!l(a)
        },
        data: function(a, c, d, e) {
            if (!!f.acceptData(a)) {
                var g, h, i = f.expando, j = typeof c == "string", k = a.nodeType, l = k ? f.cache: a, m = k ? a[f.expando]: a[f.expando] && f.expando;
                if ((!m || e && m && l[m]&&!l[m][i]) && j && d === b) {
                    return 
                }
                m || (k ? a[f.expando] = m=++f.uuid : m = f.expando), l[m] || (l[m] = {}, k || (l[m].toJSON = f.noop));
                if (typeof c == "object" || typeof c == "function") {
                    e ? l[m][i] = f.extend(l[m][i], c) : l[m] = f.extend(l[m], c)
                }
                g = l[m], e && (g[i] || (g[i] = {}), g = g[i]), d !== b && (g[f.camelCase(c)] = d);
                if (c === "events"&&!g[c]) {
                    return g[i] && g[i].events
                }
                j ? (h = g[c], h == null && (h = g[f.camelCase(c)])) : h = g;
                return h
            }
        },
        removeData: function(a, b, c) {
            if (!!f.acceptData(a)) {
                var d, e = f.expando, g = a.nodeType, h = g ? f.cache: a, i = g ? a[f.expando]: f.expando;
                if (!h[i]) {
                    return 
                }
                if (b) {
                    d = c ? h[i][e] : h[i];
                    if (d) {
                        d[b] || (b = f.camelCase(b)), delete d[b];
                        if (!l(d)) {
                            return 
                        }
                    }
                }
                if (c) {
                    delete h[i][e];
                    if (!l(h[i])) {
                        return 
                    }
                }
                var j = h[i][e];
                f.support.deleteExpando ||!h.setInterval ? delete h[i] : h[i] = null, j ? (h[i] = {}, g || (h[i].toJSON = f.noop), h[i][e] = j) : g && (f.support.deleteExpando ? delete a[f.expando] : a.removeAttribute ? a.removeAttribute(f.expando) : a[f.expando] = null)
            }
        },
        _data: function(a, b, c) {
            return f.data(a, b, c, !0)
        },
        acceptData: function(a) {
            if (a.nodeName) {
                var b = f.noData[a.nodeName.toLowerCase()];
                if (b) {
                    return b!==!0 && a.getAttribute("classid") === b
                }
            }
            return !0
        }
    }), f.fn.extend({
        data: function(a, c) {
            var d = null;
            if (typeof a == "undefined") {
                if (this.length) {
                    d = f.data(this[0]);
                    if (this[0].nodeType === 1) {
                        var e = this[0].attributes, g;
                        for (var h = 0, i = e.length; h < i; h++) {
                            g = e[h].name, g.indexOf("data-") === 0 && (g = f.camelCase(g.substring(5)), k(this[0], g, d[g]))
                        }
                    }
                }
                return d
            }
            if (typeof a == "object") {
                return this.each(function() {
                    f.data(this, a)
                })
            }
            var j = a.split(".");
            j[1] = j[1] ? "." + j[1] : "";
            if (c === b) {
                d = this.triggerHandler("getData" + j[1] + "!", [j[0]]), d === b && this.length && (d = f.data(this[0], a), d = k(this[0], a, d));
                return d === b && j[1] ? this.data(j[0]) : d
            }
            return this.each(function() {
                var b = f(this), d = [j[0], c];
                b.triggerHandler("setData" + j[1] + "!", d), f.data(this, a, c), b.triggerHandler("changeData" + j[1] + "!", d)
            })
        },
        removeData: function(a) {
            return this.each(function() {
                f.removeData(this, a)
            })
        }
    }), f.extend({
        _mark: function(a, c) {
            a && (c = (c || "fx") + "mark", f.data(a, c, (f.data(a, c, b, !0) || 0) + 1, !0))
        },
        _unmark: function(a, c, d) {
            a!==!0 && (d = c, c = a, a=!1);
            if (c) {
                d = d || "fx";
                var e = d + "mark", g = a ? 0: (f.data(c, e, b, !0) || 1) - 1;
                g ? f.data(c, e, g, !0) : (f.removeData(c, e, !0), m(c, d, "mark"))
            }
        },
        queue: function(a, c, d) {
            if (a) {
                c = (c || "fx") + "queue";
                var e = f.data(a, c, b, !0);
                d && (!e || f.isArray(d) ? e = f.data(a, c, f.makeArray(d), !0) : e.push(d));
                return e || []
            }
        },
        dequeue: function(a, b) {
            b = b || "fx";
            var c = f.queue(a, b), d = c.shift(), e;
            d === "inprogress" && (d = c.shift()), d && (b === "fx" && c.unshift("inprogress"), d.call(a, function() {
                f.dequeue(a, b)
            })), c.length || (f.removeData(a, b + "queue", !0), m(a, b, "queue"))
        }
    }), f.fn.extend({
        queue: function(a, c) {
            typeof a != "string" && (c = a, a = "fx");
            if (c === b) {
                return f.queue(this[0], a)
            }
            return this.each(function() {
                var b = f.queue(this, a, c);
                a === "fx" && b[0] !== "inprogress" && f.dequeue(this, a)
            })
        },
        dequeue: function(a) {
            return this.each(function() {
                f.dequeue(this, a)
            })
        },
        delay: function(a, b) {
            a = f.fx ? f.fx.speeds[a] || a : a, b = b || "fx";
            return this.queue(b, function() {
                var c = this;
                setTimeout(function() {
                    f.dequeue(c, b)
                }, a)
            })
        },
        clearQueue: function(a) {
            return this.queue(a || "fx", [])
        },
        promise: function(a, c) {
            function m() {
                --h || d.resolveWith(e, [e])
            }
            typeof a != "string" && (c = a, a = b), a = a || "fx";
            var d = f.Deferred(), e = this, g = e.length, h = 1, i = a + "defer", j = a + "queue", k = a + "mark", l;
            while (g--) {
                if (l = f.data(e[g], i, b, !0) || (f.data(e[g], j, b, !0) || f.data(e[g], k, b, !0)) && f.data(e[g], i, f._Deferred(), !0)) {
                    h++, l.done(m)
                }
            }
            m();
            return d.promise()
        }
    });
    var n = /[\n\t\r]/g, o = /\s+/, p = /\r/g, q = /^(?:button|input)$/i, r = /^(?:button|input|object|select|textarea)$/i, s = /^a(?:rea)?$/i, t = /^(?:autofocus|autoplay|async|checked|controls|defer|disabled|hidden|loop|multiple|open|readonly|required|scoped|selected)$/i, u, v;
    f.fn.extend({
        attr: function(a, b) {
            return f.access(this, a, b, !0, f.attr)
        },
        removeAttr: function(a) {
            return this.each(function() {
                f.removeAttr(this, a)
            })
        },
        prop: function(a, b) {
            return f.access(this, a, b, !0, f.prop)
        },
        removeProp: function(a) {
            a = f.propFix[a] || a;
            return this.each(function() {
                try {
                    this[a] = b, delete this[a]
                } catch (c) {}
            })
        },
        addClass: function(a) {
            var b, c, d, e, g, h, i;
            if (f.isFunction(a)) {
                return this.each(function(b) {
                    f(this).addClass(a.call(this, b, this.className))
                })
            }
            if (a && typeof a == "string") {
                b = a.split(o);
                for (c = 0, d = this.length; c < d; c++) {
                    e = this[c];
                    if (e.nodeType === 1) {
                        if (!e.className && b.length === 1) {
                            e.className = a
                        } else {
                            g = " " + e.className + " ";
                            for (h = 0, i = b.length; h < i; h++) {
                                ~g.indexOf(" " + b[h] + " ") || (g += b[h] + " ")
                            }
                            e.className = f.trim(g)
                        }
                    }
                }
            }
            return this
        },
        removeClass: function(a) {
            var c, d, e, g, h, i, j;
            if (f.isFunction(a)) {
                return this.each(function(b) {
                    f(this).removeClass(a.call(this, b, this.className))
                })
            }
            if (a && typeof a == "string" || a === b) {
                c = (a || "").split(o);
                for (d = 0, e = this.length; d < e; d++) {
                    g = this[d];
                    if (g.nodeType === 1 && g.className) {
                        if (a) {
                            h = (" " + g.className + " ").replace(n, " ");
                            for (i = 0, j = c.length; i < j; i++) {
                                h = h.replace(" " + c[i] + " ", " ")
                            }
                            g.className = f.trim(h)
                        } else {
                            g.className = ""
                        }
                    }
                }
            }
            return this
        },
        toggleClass: function(a, b) {
            var c = typeof a, d = typeof b == "boolean";
            if (f.isFunction(a)) {
                return this.each(function(c) {
                    f(this).toggleClass(a.call(this, c, this.className, b), b)
                })
            }
            return this.each(function() {
                if (c === "string") {
                    var e, g = 0, h = f(this), i = b, j = a.split(o);
                    while (e = j[g++]) {
                        i = d ? i : !h.hasClass(e), h[i ? "addClass": "removeClass"](e)
                    }
                } else {
                    if (c === "undefined" || c === "boolean") {
                        this.className && f._data(this, "__className__", this.className), this.className = this.className || a===!1 ? "" : f._data(this, "__className__") || ""
                    }
                }
            })
        },
        hasClass: function(a) {
            var b = " " + a + " ";
            for (var c = 0, d = this.length; c < d; c++) {
                if (this[c].nodeType === 1 && (" " + this[c].className + " ").replace(n, " ").indexOf(b)>-1) {
                    return !0
                }
            }
            return !1
        },
        val: function(a) {
            var c, d, e = this[0];
            if (!arguments.length) {
                if (e) {
                    c = f.valHooks[e.nodeName.toLowerCase()] || f.valHooks[e.type];
                    if (c && "get" in c && (d = c.get(e, "value")) !== b) {
                        return d
                    }
                    d = e.value;
                    return typeof d == "string" ? d.replace(p, "") : d == null ? "" : d
                }
                return b
            }
            var g = f.isFunction(a);
            return this.each(function(d) {
                var e = f(this), h;
                if (this.nodeType === 1) {
                    g ? h = a.call(this, d, e.val()) : h = a, h == null ? h = "" : typeof h == "number" ? h += "" : f.isArray(h) && (h = f.map(h, function(a) {
                        return a == null ? "" : a + ""
                    })), c = f.valHooks[this.nodeName.toLowerCase()] || f.valHooks[this.type];
                    if (!c ||!("set" in c) || c.set(this, h, "value") === b) {
                        this.value = h
                    }
                }
            })
        }
    }), f.extend({
        valHooks: {
            option: {
                get: function(a) {
                    var b = a.attributes.value;
                    return !b || b.specified ? a.value : a.text
                }
            },
            select: {
                get: function(a) {
                    var b, c = a.selectedIndex, d = [], e = a.options, g = a.type === "select-one";
                    if (c < 0) {
                        return null
                    }
                    for (var h = g ? c : 0, i = g ? c + 1 : e.length; h < i; h++) {
                        var j = e[h];
                        if (j.selected && (f.support.optDisabled?!j.disabled : j.getAttribute("disabled") === null) && (!j.parentNode.disabled ||!f.nodeName(j.parentNode, "optgroup"))
                            ) {
                            b = f(j).val();
                            if (g) {
                                return b
                            }
                            d.push(b)
                        }
                    }
                    if (g&&!d.length && e.length) {
                        return f(e[c]).val()
                    }
                    return d
                },
                set: function(a, b) {
                    var c = f.makeArray(b);
                    f(a).find("option").each(function() {
                        this.selected = f.inArray(f(this).val(), c) >= 0
                    }), c.length || (a.selectedIndex =- 1);
                    return c
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
        attr: function(a, c, d, e) {
            var g = a.nodeType;
            if (!a || g === 3 || g === 8 || g === 2) {
                return b
            }
            if (e && c in f.attrFn) {
                return f(a)[c](d)
            }
            if (!("getAttribute" in a)) {
                return f.prop(a, c, d)
            }
            var h, i, j = g !== 1 ||!f.isXMLDoc(a);
            j && (c = f.attrFix[c] || c, i = f.attrHooks[c], i || (t.test(c) ? i = v : u && (i = u)));
            if (d !== b) {
                if (d === null) {
                    f.removeAttr(a, c);
                    return b
                }
                if (i && "set" in i && j && (h = i.set(a, d, c)) !== b) {
                    return h
                }
                a.setAttribute(c, "" + d);
                return d
            }
            if (i && "get" in i && j && (h = i.get(a, c)) !== null) {
                return h
            }
            h = a.getAttribute(c);
            return h === null ? b : h
        },
        removeAttr: function(a, b) {
            var c;
            a.nodeType === 1 && (b = f.attrFix[b] || b, f.attr(a, b, ""), a.removeAttribute(b), t.test(b) && (c = f.propFix[b] || b) in a && (a[c]=!1))
        },
        attrHooks: {
            type: {
                set: function(a, b) {
                    if (q.test(a.nodeName) && a.parentNode) {
                        f.error("type property can't be changed")
                    } else {
                        if (!f.support.radioValue && b === "radio" && f.nodeName(a, "input")) {
                            var c = a.value;
                            a.setAttribute("type", b), c && (a.value = c);
                            return b
                        }
                    }
                }
            },
            value: {
                get: function(a, b) {
                    if (u && f.nodeName(a, "button")) {
                        return u.get(a, b)
                    }
                    return b in a ? a.value : null
                },
                set: function(a, b, c) {
                    if (u && f.nodeName(a, "button")) {
                        return u.set(a, b, c)
                    }
                    a.value = b
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
        prop: function(a, c, d) {
            var e = a.nodeType;
            if (!a || e === 3 || e === 8 || e === 2) {
                return b
            }
            var g, h, i = e !== 1 ||!f.isXMLDoc(a);
            i && (c = f.propFix[c] || c, h = f.propHooks[c]);
            return d !== b ? h && "set" in h && (g = h.set(a, d, c)) !== b ? g : a[c] = d : h && "get" in h && (g = h.get(a, c)) !== null ? g : a[c]
        },
        propHooks: {
            tabIndex: {
                get: function(a) {
                    var c = a.getAttributeNode("tabindex");
                    return c && c.specified ? parseInt(c.value, 10) : r.test(a.nodeName) || s.test(a.nodeName) && a.href ? 0 : b
                }
            }
        }
    }), f.attrHooks.tabIndex = f.propHooks.tabIndex, v = {
        get: function(a, c) {
            var d;
            return f.prop(a, c)===!0 || (d = a.getAttributeNode(c)) && d.nodeValue!==!1 ? c.toLowerCase() : b
        },
        set: function(a, b, c) {
            var d;
            b===!1 ? f.removeAttr(a, c) : (d = f.propFix[c] || c, d in a && (a[d]=!0), a.setAttribute(c, c.toLowerCase()));
            return c
        }
    }, f.support.getSetAttribute || (u = f.valHooks.button = {
        get: function(a, c) {
            var d;
            d = a.getAttributeNode(c);
            return d && d.nodeValue !== "" ? d.nodeValue : b
        },
        set: function(a, b, d) {
            var e = a.getAttributeNode(d);
            e || (e = c.createAttribute(d), a.setAttributeNode(e));
            return e.nodeValue = b + ""
        }
    }, f.each(["width", "height"], function(a, b) {
        f.attrHooks[b] = f.extend(f.attrHooks[b], {
            set: function(a, c) {
                if (c === "") {
                    a.setAttribute(b, "auto");
                    return c
                }
            }
        })
    })), f.support.hrefNormalized || f.each(["href", "src", "width", "height"], function(a, c) {
        f.attrHooks[c] = f.extend(f.attrHooks[c], {
            get: function(a) {
                var d = a.getAttribute(c, 2);
                return d === null ? b : d
            }
        })
    }), f.support.style || (f.attrHooks.style = {
        get: function(a) {
            return a.style.cssText.toLowerCase() || b
        },
        set: function(a, b) {
            return a.style.cssText = "" + b
        }
    }), f.support.optSelected || (f.propHooks.selected = f.extend(f.propHooks.selected, {
        get: function(a) {
            var b = a.parentNode;
            b && (b.selectedIndex, b.parentNode && b.parentNode.selectedIndex);
            return null
        }
    })), f.support.checkOn || f.each(["radio", "checkbox"], function() {
        f.valHooks[this] = {
            get: function(a) {
                return a.getAttribute("value") === null ? "on" : a.value
            }
        }
    }), f.each(["radio", "checkbox"], function() {
        f.valHooks[this] = f.extend(f.valHooks[this], {
            set: function(a, b) {
                if (f.isArray(b)) {
                    return a.checked = f.inArray(f(a).val(), b) >= 0
                }
            }
        })
    });
    var w = /\.(.*)$/, x = /^(?:textarea|input|select)$/i, y = /\./g, z = / /g, A = /[^\w\s.|`]/g, B = function(a) {
        return a.replace(A, "\\$&")
    };
    f.event = {
        add: function(a, c, d, e) {
            if (a.nodeType !== 3 && a.nodeType !== 8) {
                if (d===!1) {
                    d = C
                } else {
                    if (!d) {
                        return 
                    }
                }
                var g, h;
                d.handler && (g = d, d = g.handler), d.guid || (d.guid = f.guid++);
                var i = f._data(a);
                if (!i) {
                    return 
                }
                var j = i.events, k = i.handle;
                j || (i.events = j = {}), k || (i.handle = k = function(a) {
                    return typeof f != "undefined" && (!a || f.event.triggered !== a.type) ? f.event.handle.apply(k.elem, arguments) : b
                }), k.elem = a, c = c.split(" ");
                var l, m = 0, n;
                while (l = c[m++]) {
                    h = g ? f.extend({}, g) : {
                        handler: d,
                        data: e
                    }, l.indexOf(".")>-1 ? (n = l.split("."), l = n.shift(), h.namespace = n.slice(0).sort().join(".")) : (n = [], h.namespace = ""), h.type = l, h.guid || (h.guid = d.guid);
                    var o = j[l], p = f.event.special[l] || {};
                    if (!o) {
                        o = j[l] = [];
                        if (!p.setup || p.setup.call(a, e, n, k)===!1) {
                            a.addEventListener ? a.addEventListener(l, k, !1) : a.attachEvent && a.attachEvent("on" + l, k)
                        }
                    }
                    p.add && (p.add.call(a, h), h.handler.guid || (h.handler.guid = d.guid)), o.push(h), f.event.global[l]=!0
                }
                a = null
            }
        },
        global: {},
        remove: function(a, c, d, e) {
            if (a.nodeType !== 3 && a.nodeType !== 8) {
                d===!1 && (d = C);
                var g, h, i, j, k = 0, l, m, n, o, p, q, r, s = f.hasData(a) && f._data(a), t = s && s.events;
                if (!s ||!t) {
                    return 
                }
                c && c.type && (d = c.handler, c = c.type);
                if (!c || typeof c == "string" && c.charAt(0) === ".") {
                    c = c || "";
                    for (h in t) {
                        f.event.remove(a, h + c)
                    }
                    return 
                }
                c = c.split(" ");
                while (h = c[k++]) {
                    r = h, q = null, l = h.indexOf(".") < 0, m = [], l || (m = h.split("."), h = m.shift(), n = new RegExp("(^|\\.)" + f.map(m.slice(0).sort(), B).join("\\.(?:.*\\.)?") + "(\\.|$)")), p = t[h];
                    if (!p) {
                        continue
                    }
                    if (!d) {
                        for (j = 0; j < p.length; j++) {
                            q = p[j];
                            if (l || n.test(q.namespace)) {
                                f.event.remove(a, r, q.handler, j), p.splice(j--, 1)
                            }
                        }
                        continue
                    }
                    o = f.event.special[h] || {};
                    for (j = e || 0; j < p.length; j++) {
                        q = p[j];
                        if (d.guid === q.guid) {
                            if (l || n.test(q.namespace)) {
                                e == null && p.splice(j--, 1), o.remove && o.remove.call(a, q)
                            }
                            if (e != null) {
                                break
                            }
                        }
                    }
                    if (p.length === 0 || e != null && p.length === 1) {
                        (!o.teardown || o.teardown.call(a, m)===!1) && f.removeEvent(a, h, s.handle), g = null, delete t[h]
                    }
                }
                if (f.isEmptyObject(t)) {
                    var u = s.handle;
                    u && (u.elem = null), delete s.events, delete s.handle, f.isEmptyObject(s) && f.removeData(a, b, !0)
                }
            }
        },
        customEvent: {
            getData: !0,
            setData: !0,
            changeData: !0
        },
        trigger: function(c, d, e, g) {
            var h = c.type || c, i = [], j;
            h.indexOf("!") >= 0 && (h = h.slice(0, - 1), j=!0), h.indexOf(".") >= 0 && (i = h.split("."), h = i.shift(), i.sort());
            if (!!e&&!f.event.customEvent[h]||!!f.event.global[h]) {
                c = typeof c == "object" ? c[f.expando] ? c : new f.Event(h, c) : new f.Event(h), c.type = h, c.exclusive = j, c.namespace = i.join("."), c.namespace_re = new RegExp("(^|\\.)" + i.join("\\.(?:.*\\.)?") + "(\\.|$)");
                if (g ||!e) {
                    c.preventDefault(), c.stopPropagation()
                }
                if (!e) {
                    f.each(f.cache, function() {
                        var a = f.expando, b = this[a];
                        b && b.events && b.events[h] && f.event.trigger(c, d, b.handle.elem)
                    });
                    return 
                }
                if (e.nodeType === 3 || e.nodeType === 8) {
                    return 
                }
                c.result = b, c.target = e, d = d != null ? f.makeArray(d) : [], d.unshift(c);
                var k = e, l = h.indexOf(":") < 0 ? "on" + h: "";
                do {
                    var m = f._data(k, "handle");
                    c.currentTarget = k, m && m.apply(k, d), l && f.acceptData(k) && k[l] && k[l].apply(k, d)===!1 && (c.result=!1, c.preventDefault()), k = k.parentNode || k.ownerDocument || k === c.target.ownerDocument && a
                }
                while (k&&!c.isPropagationStopped());
                if (!c.isDefaultPrevented()) {
                    var n, o = f.event.special[h] || {};
                    if ((!o._default || o._default.call(e.ownerDocument, c)===!1) && (h !== "click" ||!f.nodeName(e, "a")) && f.acceptData(e)) {
                        try {
                            l && e[h] && (n = e[l], n && (e[l] = null), f.event.triggered = h, e[h]())
                        } catch (p) {}
                        n && (e[l] = n), f.event.triggered = b
                    }
                }
                return c.result
            }
        },
        handle: function(c) {
            c = f.event.fix(c || a.event);
            var d = ((f._data(this, "events") || {})[c.type] || []).slice(0), e=!c.exclusive&&!c.namespace, g = Array.prototype.slice.call(arguments, 0);
            g[0] = c, c.currentTarget = this;
            for (var h = 0, i = d.length; h < i; h++) {
                var j = d[h];
                if (e || c.namespace_re.test(j.namespace)) {
                    c.handler = j.handler, c.data = j.data, c.handleObj = j;
                    var k = j.handler.apply(this, g);
                    k !== b && (c.result = k, k===!1 && (c.preventDefault(), c.stopPropagation()));
                    if (c.isImmediatePropagationStopped()) {
                        break
                    }
                }
            }
            return c.result
        },
        props: "altKey attrChange attrName bubbles button cancelable charCode clientX clientY ctrlKey currentTarget data detail eventPhase fromElement handler keyCode layerX layerY metaKey newValue offsetX offsetY pageX pageY prevValue relatedNode relatedTarget screenX screenY shiftKey srcElement target toElement view wheelDelta which".split(" "),
        fix: function(a) {
            if (a[f.expando]) {
                return a
            }
            var d = a;
            a = f.Event(d);
            for (var e = this.props.length, g; e;) {
                g = this.props[--e], a[g] = d[g]
            }
            a.target || (a.target = a.srcElement || c), a.target.nodeType === 3 && (a.target = a.target.parentNode), !a.relatedTarget && a.fromElement && (a.relatedTarget = a.fromElement === a.target ? a.toElement : a.fromElement);
            if (a.pageX == null && a.clientX != null) {
                var h = a.target.ownerDocument || c, i = h.documentElement, j = h.body;
                a.pageX = a.clientX + (i && i.scrollLeft || j && j.scrollLeft || 0) - (i && i.clientLeft || j && j.clientLeft || 0), a.pageY = a.clientY + (i && i.scrollTop || j && j.scrollTop || 0) - (i && i.clientTop || j && j.clientTop || 0)
            }
            a.which == null && (a.charCode != null || a.keyCode != null) && (a.which = a.charCode != null ? a.charCode : a.keyCode), !a.metaKey && a.ctrlKey && (a.metaKey = a.ctrlKey), !a.which && a.button !== b && (a.which = a.button & 1 ? 1 : a.button & 2 ? 3 : a.button & 4 ? 2 : 0);
            return a
        },
        guid: 100000000,
        proxy: f.proxy,
        special: {
            ready: {
                setup: f.bindReady,
                teardown: f.noop
            },
            live: {
                add: function(a) {
                    f.event.add(this, M(a.origType, a.selector), f.extend({}, a, {
                        handler: L,
                        guid: a.handler.guid
                    }))
                },
                remove: function(a) {
                    f.event.remove(this, M(a.origType, a.selector), a)
                }
            },
            beforeunload: {
                setup: function(a, b, c) {
                    f.isWindow(this) && (this.onbeforeunload = c)
                },
                teardown: function(a, b) {
                    this.onbeforeunload === b && (this.onbeforeunload = null)
                }
            }
        }
    }, f.removeEvent = c.removeEventListener ? function(a, b, c) {
        a.removeEventListener && a.removeEventListener(b, c, !1)
    } : function(a, b, c) {
        a.detachEvent && a.detachEvent("on" + b, c)
    }, f.Event = function(a, b) {
        if (!this.preventDefault) {
            return new f.Event(a, b)
        }
        a && a.type ? (this.originalEvent = a, this.type = a.type, this.isDefaultPrevented = a.defaultPrevented || a.returnValue===!1 || a.getPreventDefault && a.getPreventDefault() ? D : C) : this.type = a, b && f.extend(this, b), this.timeStamp = f.now(), this[f.expando]=!0
    }, f.Event.prototype = {
        preventDefault: function() {
            this.isDefaultPrevented = D;
            var a = this.originalEvent;
            !a || (a.preventDefault ? a.preventDefault() : a.returnValue=!1)
        },
        stopPropagation: function() {
            this.isPropagationStopped = D;
            var a = this.originalEvent;
            !a || (a.stopPropagation && a.stopPropagation(), a.cancelBubble=!0)
        },
        stopImmediatePropagation: function() {
            this.isImmediatePropagationStopped = D, this.stopPropagation()
        },
        isDefaultPrevented: C,
        isPropagationStopped: C,
        isImmediatePropagationStopped: C
    };
    var E = function(a) {
        var b = a.relatedTarget, c=!1, d = a.type;
        a.type = a.data, b !== this && (b && (c = f.contains(this, b)), c || (f.event.handle.apply(this, arguments), a.type = d))
    }, F = function(a) {
        a.type = a.data, f.event.handle.apply(this, arguments)
    };
    f.each({
        mouseenter: "mouseover",
        mouseleave: "mouseout"
    }, function(a, b) {
        f.event.special[a] = {
            setup: function(c) {
                f.event.add(this, b, c && c.selector ? F : E, a)
            },
            teardown: function(a) {
                f.event.remove(this, b, a && a.selector ? F : E)
            }
        }
    }), f.support.submitBubbles || (f.event.special.submit = {
        setup: function(a, b) {
            if (!f.nodeName(this, "form")) {
                f.event.add(this, "click.specialSubmit", function(a) {
                    var b = a.target, c = f.nodeName(b, "input") || f.nodeName(b, "button") ? b.type: "";
                    (c === "submit" || c === "image") && f(b).closest("form").length && J("submit", this, arguments)
                }), f.event.add(this, "keypress.specialSubmit", function(a) {
                    var b = a.target, c = f.nodeName(b, "input") || f.nodeName(b, "button") ? b.type: "";
                    (c === "text" || c === "password") && f(b).closest("form").length && a.keyCode === 13 && J("submit", this, arguments)
                })
            } else {
                return !1
            }
        },
        teardown: function(a) {
            f.event.remove(this, ".specialSubmit")
        }
    });
    if (!f.support.changeBubbles) {
        var G, H = function(a) {
            var b = f.nodeName(a, "input") ? a.type: "", c = a.value;
            b === "radio" || b === "checkbox" ? c = a.checked : b === "select-multiple" ? c = a.selectedIndex>-1 ? f.map(a.options, function(a) {
                return a.selected
            }).join("-") : "" : f.nodeName(a, "select") && (c = a.selectedIndex);
            return c
        }, I = function(c) {
            var d = c.target, e, g;
            if (!!x.test(d.nodeName)&&!d.readOnly) {
                e = f._data(d, "_change_data"), g = H(d), (c.type !== "focusout" || d.type !== "radio") && f._data(d, "_change_data", g);
                if (e === b || g === e) {
                    return 
                }
                if (e != null || g) {
                    c.type = "change", c.liveFired = b, f.event.trigger(c, arguments[1], d)
                }
            }
        };
        f.event.special.change = {
            filters: {
                focusout: I,
                beforedeactivate: I,
                click: function(a) {
                    var b = a.target, c = f.nodeName(b, "input") ? b.type: "";
                    (c === "radio" || c === "checkbox" || f.nodeName(b, "select")) && I.call(this, a)
                },
                keydown: function(a) {
                    var b = a.target, c = f.nodeName(b, "input") ? b.type: "";
                    (a.keyCode === 13&&!f.nodeName(b, "textarea") || a.keyCode === 32 && (c === "checkbox" || c === "radio") || c === "select-multiple") && I.call(this, a)
                },
                beforeactivate: function(a) {
                    var b = a.target;
                    f._data(b, "_change_data", H(b))
                }
            },
            setup: function(a, b) {
                if (this.type === "file") {
                    return !1
                }
                for (var c in G) {
                    f.event.add(this, c + ".specialChange", G[c])
                }
                return x.test(this.nodeName)
            },
            teardown: function(a) {
                f.event.remove(this, ".specialChange");
                return x.test(this.nodeName)
            }
        }, G = f.event.special.change.filters, G.focus = G.beforeactivate
    }
    f.support.focusinBubbles || f.each({
        focus: "focusin",
        blur: "focusout"
    }, function(a, b) {
        function e(a) {
            var c = f.event.fix(a);
            c.type = b, c.originalEvent = {}, f.event.trigger(c, null, c.target), c.isDefaultPrevented() && a.preventDefault()
        }
        var d = 0;
        f.event.special[b] = {
            setup: function() {
                d++===0 && c.addEventListener(a, e, !0)
            },
            teardown: function() {
                --d === 0 && c.removeEventListener(a, e, !0)
            }
        }
    }), f.each(["bind", "one"], function(a, c) {
        f.fn[c] = function(a, d, e) {
            var g;
            if (typeof a == "object") {
                for (var h in a) {
                    this[c](h, d, a[h], e)
                }
                return this
            }
            if (arguments.length === 2 || d===!1) {
                e = d, d = b
            }
            c === "one" ? (g = function(a) {
                f(this).unbind(a, g);
                return e.apply(this, arguments)
            }, g.guid = e.guid || f.guid++) : g = e;
            if (a === "unload" && c !== "one") {
                this.one(a, d, e)
            } else {
                for (var i = 0, j = this.length; i < j; i++) {
                    f.event.add(this[i], a, g, d)
                }
            }
            return this
        }
    }), f.fn.extend({
        unbind: function(a, b) {
            if (typeof a == "object"&&!a.preventDefault) {
                for (var c in a) {
                    this.unbind(c, a[c])
                }
            } else {
                for (var d = 0, e = this.length; d < e; d++) {
                    f.event.remove(this[d], a, b)
                }
            }
            return this
        },
        delegate: function(a, b, c, d) {
            return this.live(b, c, d, a)
        },
        undelegate: function(a, b, c) {
            return arguments.length === 0 ? this.unbind("live") : this.die(b, null, c, a)
        },
        trigger: function(a, b) {
            return this.each(function() {
                f.event.trigger(a, b, this)
            })
        },
        triggerHandler: function(a, b) {
            if (this[0]) {
                return f.event.trigger(a, b, this[0], !0)
            }
        },
        toggle: function(a) {
            var b = arguments, c = a.guid || f.guid++, d = 0, e = function(c) {
                var e = (f.data(this, "lastToggle" + a.guid) || 0)%d;
                f.data(this, "lastToggle" + a.guid, e + 1), c.preventDefault();
                return b[e].apply(this, arguments) ||!1
            };
            e.guid = c;
            while (d < b.length) {
                b[d++].guid = c
            }
            return this.click(e)
        },
        hover: function(a, b) {
            return this.mouseenter(a).mouseleave(b || a)
        }
    });
    var K = {
        focus: "focusin",
        blur: "focusout",
        mouseenter: "mouseover",
        mouseleave: "mouseout"
    };
    f.each(["live", "die"], function(a, c) {
        f.fn[c] = function(a, d, e, g) {
            var h, i = 0, j, k, l, m = g || this.selector, n = g ? this: f(this.context);
            if (typeof a == "object"&&!a.preventDefault) {
                for (var o in a) {
                    n[c](o, d, a[o], m)
                }
                return this
            }
            if (c === "die"&&!a && g && g.charAt(0) === ".") {
                n.unbind(g);
                return this
            }
            if (d===!1 || f.isFunction(d)) {
                e = d || C, d = b
            }
            a = (a || "").split(" ");
            while ((h = a[i++]) != null) {
                j = w.exec(h), k = "", j && (k = j[0], h = h.replace(w, ""));
                if (h === "hover") {
                    a.push("mouseenter" + k, "mouseleave" + k);
                    continue
                }
                l = h, K[h] ? (a.push(K[h] + k), h = h + k) : h = (K[h] || h) + k;
                if (c === "live") {
                    for (var p = 0, q = n.length; p < q; p++) {
                        f.event.add(n[p], "live." + M(h, m), {
                            data: d,
                            selector: m,
                            handler: e,
                            origType: h,
                            origHandler: e,
                            preType: l
                        })
                    }
                } else {
                    n.unbind("live." + M(h, m), e)
                }
            }
            return this
        }
    }), f.each("blur focus focusin focusout load resize scroll unload click dblclick mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave change select submit keydown keypress keyup error".split(" "), function(a, b) {
        f.fn[b] = function(a, c) {
            c == null && (c = a, a = null);
            return arguments.length > 0 ? this.bind(b, a, c) : this.trigger(b)
        }, f.attrFn && (f.attrFn[b]=!0)
    }), function() {
        function u(a, b, c, d, e, f) {
            for (var g = 0, h = d.length; g < h; g++) {
                var i = d[g];
                if (i) {
                    var j=!1;
                    i = i[a];
                    while (i) {
                        if (i.sizcache === c) {
                            j = d[i.sizset];
                            break
                        }
                        if (i.nodeType === 1) {
                            f || (i.sizcache = c, i.sizset = g);
                            if (typeof b != "string") {
                                if (i === b) {
                                    j=!0;
                                    break
                                }
                            } else {
                                if (k.filter(b, [i]).length > 0) {
                                    j = i;
                                    break
                                }
                            }
                        }
                        i = i[a]
                    }
                    d[g] = j
                }
            }
        }
        function t(a, b, c, d, e, f) {
            for (var g = 0, h = d.length; g < h; g++) {
                var i = d[g];
                if (i) {
                    var j=!1;
                    i = i[a];
                    while (i) {
                        if (i.sizcache === c) {
                            j = d[i.sizset];
                            break
                        }
                        i.nodeType === 1&&!f && (i.sizcache = c, i.sizset = g);
                        if (i.nodeName.toLowerCase() === b) {
                            j = i;
                            break
                        }
                        i = i[a]
                    }
                    d[g] = j
                }
            }
        }
        var a = /((?:\((?:\([^()]+\)|[^()]+)+\)|\[(?:\[[^\[\]]*\]|['"][^'"]*['"]|[^\[\]'"]+)+\]|\\.|[^ >+~,(\[\\]+)+|[>+~])(\s*,\s*)?((?:.|\r|\n)*)/g, d = 0, e = Object.prototype.toString, g=!1, h=!0, i = /\\/g, j = /\W/;
        [0, 0].sort(function() {
            h=!1;
            return 0
        });
        var k = function(b, d, f, g) {
            f = f || [], d = d || c;
            var h = d;
            if (d.nodeType !== 1 && d.nodeType !== 9) {
                return []
            }
            if (!b || typeof b != "string") {
                return f
            }
            var i, j, n, o, q, r, s, t, u=!0, w = k.isXML(d), x = [], y = b;
            do {
                a.exec(""), i = a.exec(y);
                if (i) {
                    y = i[3], x.push(i[1]);
                    if (i[2]) {
                        o = i[3];
                        break
                    }
                }
            }
            while (i);
            if (x.length > 1 && m.exec(b)) {
                if (x.length === 2 && l.relative[x[0]]) {
                    j = v(x[0] + x[1], d)
                } else {
                    j = l.relative[x[0]] ? [d] : k(x.shift(), d);
                    while (x.length) {
                        b = x.shift(), l.relative[b] && (b += x.shift()), j = v(b, j)
                    }
                }
            } else {
                !g && x.length > 1 && d.nodeType === 9&&!w && l.match.ID.test(x[0])&&!l.match.ID.test(x[x.length - 1]) && (q = k.find(x.shift(), d, w), d = q.expr ? k.filter(q.expr, q.set)[0] : q.set[0]);
                if (d) {
                    q = g ? {
                        expr: x.pop(),
                        set: p(g)
                    } : k.find(x.pop(), x.length === 1 && (x[0] === "~" || x[0] === "+") && d.parentNode ? d.parentNode : d, w), j = q.expr ? k.filter(q.expr, q.set) : q.set, x.length > 0 ? n = p(j) : u=!1;
                    while (x.length) {
                        r = x.pop(), s = r, l.relative[r] ? s = x.pop() : r = "", s == null && (s = d), l.relative[r](n, s, w)
                    }
                } else {
                    n = x = []
                }
            }
            n || (n = j), n || k.error(r || b);
            if (e.call(n) === "[object Array]") {
                if (!u) {
                    f.push.apply(f, n)
                } else {
                    if (d && d.nodeType === 1) {
                        for (t = 0; n[t] != null; t++) {
                            n[t] && (n[t]===!0 || n[t].nodeType === 1 && k.contains(d, n[t])) && f.push(j[t])
                        }
                    } else {
                        for (t = 0; n[t] != null; t++) {
                            n[t] && n[t].nodeType === 1 && f.push(j[t])
                        }
                    }
                }
            } else {
                p(n, f)
            }
            o && (k(o, h, f, g), k.uniqueSort(f));
            return f
        };
        k.uniqueSort = function(a) {
            if (r) {
                g = h, a.sort(r);
                if (g) {
                    for (var b = 1; b < a.length; b++) {
                        a[b] === a[b - 1] && a.splice(b--, 1)
                    }
                }
            }
            return a
        }, k.matches = function(a, b) {
            return k(a, null, null, b)
        }, k.matchesSelector = function(a, b) {
            return k(b, null, null, [a]).length > 0
        }, k.find = function(a, b, c) {
            var d;
            if (!a) {
                return []
            }
            for (var e = 0, f = l.order.length; e < f; e++) {
                var g, h = l.order[e];
                if (g = l.leftMatch[h].exec(a)) {
                    var j = g[1];
                    g.splice(1, 1);
                    if (j.substr(j.length - 1) !== "\\") {
                        g[1] = (g[1] || "").replace(i, ""), d = l.find[h](g, b, c);
                        if (d != null) {
                            a = a.replace(l.match[h], "");
                            break
                        }
                    }
                }
            }
            d || (d = typeof b.getElementsByTagName != "undefined" ? b.getElementsByTagName("*") : []);
            return {
                set: d,
                expr: a
            }
        }, k.filter = function(a, c, d, e) {
            var f, g, h = a, i = [], j = c, m = c && c[0] && k.isXML(c[0]);
            while (a && c.length) {
                for (var n in l.filter) {
                    if ((f = l.leftMatch[n].exec(a)) != null && f[2]) {
                        var o, p, q = l.filter[n], r = f[1];
                        g=!1, f.splice(1, 1);
                        if (r.substr(r.length - 1) === "\\") {
                            continue
                        }
                        j === i && (i = []);
                        if (l.preFilter[n]) {
                            f = l.preFilter[n](f, j, d, i, e, m);
                            if (!f) {
                                g = o=!0
                            } else {
                                if (f===!0) {
                                    continue
                                }
                            }
                        }
                        if (f) {
                            for (var s = 0; (p = j[s]) != null; s++) {
                                if (p) {
                                    o = q(p, f, s, j);
                                    var t = e^!!o;
                                    d && o != null ? t ? g=!0 : j[s]=!1 : t && (i.push(p), g=!0)
                                }
                            }
                        }
                        if (o !== b) {
                            d || (j = i), a = a.replace(l.match[n], "");
                            if (!g) {
                                return []
                            }
                            break
                        }
                    }
                }
                if (a === h) {
                    if (g == null) {
                        k.error(a)
                    } else {
                        break
                    }
                }
                h = a
            }
            return j
        }, k.error = function(a) {
            throw "Syntax error, unrecognized expression: " + a
        };
        var l = k.selectors = {
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
                href: function(a) {
                    return a.getAttribute("href")
                },
                type: function(a) {
                    return a.getAttribute("type")
                }
            },
            relative: {
                "+": function(a, b) {
                    var c = typeof b == "string", d = c&&!j.test(b), e = c&&!d;
                    d && (b = b.toLowerCase());
                    for (var f = 0, g = a.length, h; f < g; f++) {
                        if (h = a[f]) {
                            while ((h = h.previousSibling) && h.nodeType !== 1) {}
                            a[f] = e || h && h.nodeName.toLowerCase() === b ? h ||!1 : h === b
                        }
                    }
                    e && k.filter(b, a, !0)
                },
                ">": function(a, b) {
                    var c, d = typeof b == "string", e = 0, f = a.length;
                    if (d&&!j.test(b)) {
                        b = b.toLowerCase();
                        for (; e < f; e++) {
                            c = a[e];
                            if (c) {
                                var g = c.parentNode;
                                a[e] = g.nodeName.toLowerCase() === b ? g : !1
                            }
                        }
                    } else {
                        for (; e < f; e++) {
                            c = a[e], c && (a[e] = d ? c.parentNode : c.parentNode === b)
                        }
                        d && k.filter(b, a, !0)
                    }
                },
                "": function(a, b, c) {
                    var e, f = d++, g = u;
                    typeof b == "string"&&!j.test(b) && (b = b.toLowerCase(), e = b, g = t), g("parentNode", b, f, a, e, c)
                },
                "~": function(a, b, c) {
                    var e, f = d++, g = u;
                    typeof b == "string"&&!j.test(b) && (b = b.toLowerCase(), e = b, g = t), g("previousSibling", b, f, a, e, c)
                }
            },
            find: {
                ID: function(a, b, c) {
                    if (typeof b.getElementById != "undefined"&&!c) {
                        var d = b.getElementById(a[1]);
                        return d && d.parentNode ? [d] : []
                    }
                },
                NAME: function(a, b) {
                    if (typeof b.getElementsByName != "undefined") {
                        var c = [], d = b.getElementsByName(a[1]);
                        for (var e = 0, f = d.length; e < f; e++) {
                            d[e].getAttribute("name") === a[1] && c.push(d[e])
                        }
                        return c.length === 0 ? null : c
                    }
                },
                TAG: function(a, b) {
                    if (typeof b.getElementsByTagName != "undefined") {
                        return b.getElementsByTagName(a[1])
                    }
                }
            },
            preFilter: {
                CLASS: function(a, b, c, d, e, f) {
                    a = " " + a[1].replace(i, "") + " ";
                    if (f) {
                        return a
                    }
                    for (var g = 0, h; (h = b[g]) != null; g++) {
                        h && (e^(h.className && (" " + h.className + " ").replace(/[\t\n\r]/g, " ").indexOf(a) >= 0) ? c || d.push(h) : c && (b[g]=!1))
                    }
                    return !1
                },
                ID: function(a) {
                    return a[1].replace(i, "")
                },
                TAG: function(a, b) {
                    return a[1].replace(i, "").toLowerCase()
                },
                CHILD: function(a) {
                    if (a[1] === "nth") {
                        a[2] || k.error(a[0]), a[2] = a[2].replace(/^\+|\s*/g, "");
                        var b = /(-?)(\d*)(?:n([+\-]?\d*))?/.exec(a[2] === "even" && "2n" || a[2] === "odd" && "2n+1" ||!/\D/.test(a[2]) && "0n+" + a[2] || a[2]);
                        a[2] = b[1] + (b[2] || 1) - 0, a[3] = b[3] - 0
                    } else {
                        a[2] && k.error(a[0])
                    }
                    a[0] = d++;
                    return a
                },
                ATTR: function(a, b, c, d, e, f) {
                    var g = a[1] = a[1].replace(i, "");
                    !f && l.attrMap[g] && (a[1] = l.attrMap[g]), a[4] = (a[4] || a[5] || "").replace(i, ""), a[2] === "~=" && (a[4] = " " + a[4] + " ");
                    return a
                },
                PSEUDO: function(b, c, d, e, f) {
                    if (b[1] === "not") {
                        if ((a.exec(b[3]) || "").length > 1 || /^\w/.test(b[3])) {
                            b[3] = k(b[3], null, null, c)
                        } else {
                            var g = k.filter(b[3], c, d, !0^f);
                            d || e.push.apply(e, g);
                            return !1
                        }
                    } else {
                        if (l.match.POS.test(b[0]) || l.match.CHILD.test(b[0])) {
                            return !0
                        }
                    }
                    return b
                },
                POS: function(a) {
                    a.unshift(!0);
                    return a
                }
            },
            filters: {
                enabled: function(a) {
                    return a.disabled===!1 && a.type !== "hidden"
                },
                disabled: function(a) {
                    return a.disabled===!0
                },
                checked: function(a) {
                    return a.checked===!0
                },
                selected: function(a) {
                    a.parentNode && a.parentNode.selectedIndex;
                    return a.selected===!0
                },
                parent: function(a) {
                    return !!a.firstChild
                },
                empty: function(a) {
                    return !a.firstChild
                },
                has: function(a, b, c) {
                    return !!k(c[3], a).length
                },
                header: function(a) {
                    return /h\d/i.test(a.nodeName)
                },
                text: function(a) {
                    var b = a.getAttribute("type"), c = a.type;
                    return a.nodeName.toLowerCase() === "input" && "text" === c && (b === c || b === null)
                },
                radio: function(a) {
                    return a.nodeName.toLowerCase() === "input" && "radio" === a.type
                },
                checkbox: function(a) {
                    return a.nodeName.toLowerCase() === "input" && "checkbox" === a.type
                },
                file: function(a) {
                    return a.nodeName.toLowerCase() === "input" && "file" === a.type
                },
                password: function(a) {
                    return a.nodeName.toLowerCase() === "input" && "password" === a.type
                },
                submit: function(a) {
                    var b = a.nodeName.toLowerCase();
                    return (b === "input" || b === "button") && "submit" === a.type
                },
                image: function(a) {
                    return a.nodeName.toLowerCase() === "input" && "image" === a.type
                },
                reset: function(a) {
                    var b = a.nodeName.toLowerCase();
                    return (b === "input" || b === "button") && "reset" === a.type
                },
                button: function(a) {
                    var b = a.nodeName.toLowerCase();
                    return b === "input" && "button" === a.type || b === "button"
                },
                input: function(a) {
                    return /input|select|textarea|button/i.test(a.nodeName)
                },
                focus: function(a) {
                    return a === a.ownerDocument.activeElement
                }
            },
            setFilters: {
                first: function(a, b) {
                    return b === 0
                },
                last: function(a, b, c, d) {
                    return b === d.length - 1
                },
                even: function(a, b) {
                    return b%2 === 0
                },
                odd: function(a, b) {
                    return b%2 === 1
                },
                lt: function(a, b, c) {
                    return b < c[3] - 0
                },
                gt: function(a, b, c) {
                    return b > c[3] - 0
                },
                nth: function(a, b, c) {
                    return c[3] - 0 === b
                },
                eq: function(a, b, c) {
                    return c[3] - 0 === b
                }
            },
            filter: {
                PSEUDO: function(a, b, c, d) {
                    var e = b[1], f = l.filters[e];
                    if (f) {
                        return f(a, c, b, d)
                    }
                    if (e === "contains") {
                        return (a.textContent || a.innerText || k.getText([a]) || "").indexOf(b[3]) >= 0
                    }
                    if (e === "not") {
                        var g = b[3];
                        for (var h = 0, i = g.length; h < i; h++) {
                            if (g[h] === a) {
                                return !1
                            }
                        }
                        return !0
                    }
                    k.error(e)
                },
                CHILD: function(a, b) {
                    var c = b[1], d = a;
                    switch (c) {
                    case"only":
                    case"first":
                        while (d = d.previousSibling) {
                            if (d.nodeType === 1) {
                                return !1
                            }
                        }
                        if (c === "first") {
                            return !0
                        }
                        d = a;
                    case"last":
                        while (d = d.nextSibling) {
                            if (d.nodeType === 1) {
                                return !1
                            }
                        }
                        return !0;
                    case"nth":
                        var e = b[2], f = b[3];
                        if (e === 1 && f === 0) {
                            return !0
                        }
                        var g = b[0], h = a.parentNode;
                        if (h && (h.sizcache !== g ||!a.nodeIndex)) {
                            var i = 0;
                            for (d = h.firstChild; d; d = d.nextSibling) {
                                d.nodeType === 1 && (d.nodeIndex=++i)
                            }
                            h.sizcache = g
                        }
                        var j = a.nodeIndex - f;
                        return e === 0 ? j === 0 : j%e === 0 && j / e >= 0
                    }
                },
                ID: function(a, b) {
                    return a.nodeType === 1 && a.getAttribute("id") === b
                },
                TAG: function(a, b) {
                    return b === "*" && a.nodeType === 1 || a.nodeName.toLowerCase() === b
                },
                CLASS: function(a, b) {
                    return (" " + (a.className || a.getAttribute("class")) + " ").indexOf(b)>-1
                },
                ATTR: function(a, b) {
                    var c = b[1], d = l.attrHandle[c] ? l.attrHandle[c](a): a[c] != null ? a[c]: a.getAttribute(c), e = d + "", f = b[2], g = b[4];
                    return d == null ? f === "!=" : f === "=" ? e === g : f === "*=" ? e.indexOf(g) >= 0 : f === "~=" ? (" " + e + " ").indexOf(g) >= 0 : g ? f === "!=" ? e !== g : f === "^=" ? e.indexOf(g) === 0 : f === "$=" ? e.substr(e.length - g.length) === g : f === "|=" ? e === g || e.substr(0, g.length + 1) === g + "-" : !1 : e && d!==!1
                },
                POS: function(a, b, c, d) {
                    var e = b[2], f = l.setFilters[e];
                    if (f) {
                        return f(a, c, b, d)
                    }
                }
            }
        }, m = l.match.POS, n = function(a, b) {
            return "\\" + (b - 0 + 1)
        };
        for (var o in l.match) {
            l.match[o] = new RegExp(l.match[o].source + /(?![^\[]*\])(?![^\(]*\))/.source), l.leftMatch[o] = new RegExp(/(^(?:.|\r|\n)*?)/.source + l.match[o].source.replace(/\\(\d+)/g, n))
        }
        var p = function(a, b) {
            a = Array.prototype.slice.call(a, 0);
            if (b) {
                b.push.apply(b, a);
                return b
            }
            return a
        };
        try {
            Array.prototype.slice.call(c.documentElement.childNodes, 0)[0].nodeType
        } catch (q) {
            p = function(a, b) {
                var c = 0, d = b || [];
                if (e.call(a) === "[object Array]") {
                    Array.prototype.push.apply(d, a)
                } else {
                    if (typeof a.length == "number") {
                        for (var f = a.length; c < f; c++) {
                            d.push(a[c])
                        }
                    } else {
                        for (; a[c]; c++) {
                            d.push(a[c])
                        }
                    }
                }
                return d
            }
        }
        var r, s;
        c.documentElement.compareDocumentPosition ? r = function(a, b) {
            if (a === b) {
                g=!0;
                return 0
            }
            if (!a.compareDocumentPosition ||!b.compareDocumentPosition) {
                return a.compareDocumentPosition?-1 : 1
            }
            return a.compareDocumentPosition(b) & 4?-1 : 1
        } : (r = function(a, b) {
            if (a === b) {
                g=!0;
                return 0
            }
            if (a.sourceIndex && b.sourceIndex) {
                return a.sourceIndex - b.sourceIndex
            }
            var c, d, e = [], f = [], h = a.parentNode, i = b.parentNode, j = h;
            if (h === i) {
                return s(a, b)
            }
            if (!h) {
                return - 1
            }
            if (!i) {
                return 1
            }
            while (j) {
                e.unshift(j), j = j.parentNode
            }
            j = i;
            while (j) {
                f.unshift(j), j = j.parentNode
            }
            c = e.length, d = f.length;
            for (var k = 0; k < c && k < d; k++) {
                if (e[k] !== f[k]) {
                    return s(e[k], f[k])
                }
            }
            return k === c ? s(a, f[k], - 1) : s(e[k], b, 1)
        }, s = function(a, b, c) {
            if (a === b) {
                return c
            }
            var d = a.nextSibling;
            while (d) {
                if (d === b) {
                    return - 1
                }
                d = d.nextSibling
            }
            return 1
        }), k.getText = function(a) {
            var b = "", c;
            for (var d = 0; a[d]; d++) {
                c = a[d], c.nodeType === 3 || c.nodeType === 4 ? b += c.nodeValue : c.nodeType !== 8 && (b += k.getText(c.childNodes))
            }
            return b
        }, function() {
            var a = c.createElement("div"), d = "script" + (new Date).getTime(), e = c.documentElement;
            a.innerHTML = "<a name='" + d + "'/>", e.insertBefore(a, e.firstChild), c.getElementById(d) && (l.find.ID = function(a, c, d) {
                if (typeof c.getElementById != "undefined"&&!d) {
                    var e = c.getElementById(a[1]);
                    return e ? e.id === a[1] || typeof e.getAttributeNode != "undefined" && e.getAttributeNode("id").nodeValue === a[1] ? [e] : b : []
                }
            }, l.filter.ID = function(a, b) {
                var c = typeof a.getAttributeNode != "undefined" && a.getAttributeNode("id");
                return a.nodeType === 1 && c && c.nodeValue === b
            }), e.removeChild(a), e = a = null
        }(), function() {
            var a = c.createElement("div");
            a.appendChild(c.createComment("")), a.getElementsByTagName("*").length > 0 && (l.find.TAG = function(a, b) {
                var c = b.getElementsByTagName(a[1]);
                if (a[1] === "*") {
                    var d = [];
                    for (var e = 0; c[e]; e++) {
                        c[e].nodeType === 1 && d.push(c[e])
                    }
                    c = d
                }
                return c
            }), a.innerHTML = "<a href='#'></a>", a.firstChild && typeof a.firstChild.getAttribute != "undefined" && a.firstChild.getAttribute("href") !== "#" && (l.attrHandle.href = function(a) {
                return a.getAttribute("href", 2)
            }), a = null
        }(), c.querySelectorAll && function() {
            var a = k, b = c.createElement("div"), d = "__sizzle__";
            b.innerHTML = "<p class='TEST'></p>";
            if (!b.querySelectorAll || b.querySelectorAll(".TEST").length !== 0) {
                k = function(b, e, f, g) {
                    e = e || c;
                    if (!g&&!k.isXML(e)) {
                        var h = /^(\w+$)|^\.([\w\-]+$)|^#([\w\-]+$)/.exec(b);
                        if (h && (e.nodeType === 1 || e.nodeType === 9)) {
                            if (h[1]) {
                                return p(e.getElementsByTagName(b), f)
                            }
                            if (h[2] && l.find.CLASS && e.getElementsByClassName) {
                                return p(e.getElementsByClassName(h[2]), f)
                            }
                        }
                        if (e.nodeType === 9) {
                            if (b === "body" && e.body) {
                                return p([e.body], f)
                            }
                            if (h && h[3]) {
                                var i = e.getElementById(h[3]);
                                if (!i ||!i.parentNode) {
                                    return p([], f)
                                }
                                if (i.id === h[3]) {
                                    return p([i], f)
                                }
                            }
                            try {
                                return p(e.querySelectorAll(b), f)
                            } catch (j) {}
                        } else {
                            if (e.nodeType === 1 && e.nodeName.toLowerCase() !== "object") {
                                var m = e, n = e.getAttribute("id"), o = n || d, q = e.parentNode, r = /^\s*[+~]/.test(b);
                                n ? o = o.replace(/'/g, "\\$&") : e.setAttribute("id", o), r && q && (e = e.parentNode);
                                try {
                                    if (!r || q) {
                                        return p(e.querySelectorAll("[id='" + o + "'] " + b), f)
                                    }
                                } catch (s) {} finally {
                                    n || m.removeAttribute("id")
                                }
                            }
                        }
                    }
                    return a(b, e, f, g)
                };
                for (var e in a) {
                    k[e] = a[e]
                }
                b = null
            }
        }(), function() {
            var a = c.documentElement, b = a.matchesSelector || a.mozMatchesSelector || a.webkitMatchesSelector || a.msMatchesSelector;
            if (b) {
                var d=!b.call(c.createElement("div"), "div"), e=!1;
                try {
                    b.call(c.documentElement, "[test!='']:sizzle")
                } catch (f) {
                    e=!0
                }
                k.matchesSelector = function(a, c) {
                    c = c.replace(/\=\s*([^'"\]]*)\s*\]/g, "='$1']");
                    if (!k.isXML(a)) {
                        try {
                            if (e ||!l.match.PSEUDO.test(c)&&!/!=/.test(c)) {
                                var f = b.call(a, c);
                                if (f ||!d || a.document && a.document.nodeType !== 11) {
                                    return f
                                }
                            }
                        } catch (g) {}
                    }
                    return k(c, null, null, [a]).length > 0
                }
            }
        }(), function() {
            var a = c.createElement("div");
            a.innerHTML = "<div class='test e'></div><div class='test'></div>";
            if (!!a.getElementsByClassName && a.getElementsByClassName("e").length !== 0) {
                a.lastChild.className = "e";
                if (a.getElementsByClassName("e").length === 1) {
                    return 
                }
                l.order.splice(1, 0, "CLASS"), l.find.CLASS = function(a, b, c) {
                    if (typeof b.getElementsByClassName != "undefined"&&!c) {
                        return b.getElementsByClassName(a[1])
                    }
                }, a = null
            }
        }(), c.documentElement.contains ? k.contains = function(a, b) {
            return a !== b && (a.contains ? a.contains(b) : !0)
        } : c.documentElement.compareDocumentPosition ? k.contains = function(a, b) {
            return !!(a.compareDocumentPosition(b) & 16)
        } : k.contains = function() {
            return !1
        }, k.isXML = function(a) {
            var b = (a ? a.ownerDocument || a : 0).documentElement;
            return b ? b.nodeName !== "HTML" : !1
        };
        var v = function(a, b) {
            var c, d = [], e = "", f = b.nodeType ? [b]: b;
            while (c = l.match.PSEUDO.exec(a)) {
                e += c[0], a = a.replace(l.match.PSEUDO, "")
            }
            a = l.relative[a] ? a + "*" : a;
            for (var g = 0, h = f.length; g < h; g++) {
                k(a, f[g], d)
            }
            return k.filter(e, d)
        };
        f.find = k, f.expr = k.selectors, f.expr[":"] = f.expr.filters, f.unique = k.uniqueSort, f.text = k.getText, f.isXMLDoc = k.isXML, f.contains = k.contains
    }();
    var N = /Until$/, O = /^(?:parents|prevUntil|prevAll)/, P = /,/, Q = /^.[^:#\[\.,]*$/, R = Array.prototype.slice, S = f.expr.match.POS, T = {
        children: !0,
        contents: !0,
        next: !0,
        prev: !0
    };
    f.fn.extend({
        find: function(a) {
            var b = this, c, d;
            if (typeof a != "string") {
                return f(a).filter(function() {
                    for (c = 0, d = b.length; c < d; c++) {
                        if (f.contains(b[c], this)) {
                            return !0
                        }
                    }
                })
            }
            var e = this.pushStack("", "find", a), g, h, i;
            for (c = 0, d = this.length; c < d; c++) {
                g = e.length, f.find(a, this[c], e);
                if (c > 0) {
                    for (h = g; h < e.length; h++) {
                        for (i = 0; i < g; i++) {
                            if (e[i] === e[h]) {
                                e.splice(h--, 1);
                                break
                            }
                        }
                    }
                }
            }
            return e
        },
        has: function(a) {
            var b = f(a);
            return this.filter(function() {
                for (var a = 0, c = b.length; a < c; a++) {
                    if (f.contains(this, b[a])) {
                        return !0
                    }
                }
            })
        },
        not: function(a) {
            return this.pushStack(V(this, a, !1), "not", a)
        },
        filter: function(a) {
            return this.pushStack(V(this, a, !0), "filter", a)
        },
        is: function(a) {
            return !!a && (typeof a == "string" ? f.filter(a, this).length > 0 : this.filter(a).length > 0)
        },
        closest: function(a, b) {
            var c = [], d, e, g = this[0];
            if (f.isArray(a)) {
                var h, i, j = {}, k = 1;
                if (g && a.length) {
                    for (d = 0, e = a.length; d < e; d++) {
                        i = a[d], j[i] || (j[i] = S.test(i) ? f(i, b || this.context) : i)
                    }
                    while (g && g.ownerDocument && g !== b) {
                        for (i in j) {
                            h = j[i], (h.jquery ? h.index(g)>-1 : f(g).is(h)) && c.push({
                                selector: i,
                                elem: g,
                                level: k
                            })
                        }
                        g = g.parentNode, k++
                    }
                }
                return c
            }
            var l = S.test(a) || typeof a != "string" ? f(a, b || this.context): 0;
            for (d = 0, e = this.length; d < e; d++) {
                g = this[d];
                while (g) {
                    if (l ? l.index(g)>-1 : f.find.matchesSelector(g, a)) {
                        c.push(g);
                        break
                    }
                    g = g.parentNode;
                    if (!g ||!g.ownerDocument || g === b || g.nodeType === 11) {
                        break
                    }
                }
            }
            c = c.length > 1 ? f.unique(c) : c;
            return this.pushStack(c, "closest", a)
        },
        index: function(a) {
            if (!a) {
                return this[0] && this[0].parentNode ? this.prevAll().length : - 1
            }
            if (typeof a == "string") {
                return f.inArray(this[0], f(a))
            }
            return f.inArray(a.jquery ? a[0] : a, this)
        },
        add: function(a, b) {
            var c = typeof a == "string" ? f(a, b): f.makeArray(a && a.nodeType ? [a] : a), d = f.merge(this.get(), c);
            return this.pushStack(U(c[0]) || U(d[0]) ? d : f.unique(d))
        },
        andSelf: function() {
            return this.add(this.prevObject)
        }
    }), f.each({
        parent: function(a) {
            var b = a.parentNode;
            return b && b.nodeType !== 11 ? b : null
        },
        parents: function(a) {
            return f.dir(a, "parentNode")
        },
        parentsUntil: function(a, b, c) {
            return f.dir(a, "parentNode", c)
        },
        next: function(a) {
            return f.nth(a, 2, "nextSibling")
        },
        prev: function(a) {
            return f.nth(a, 2, "previousSibling")
        },
        nextAll: function(a) {
            return f.dir(a, "nextSibling")
        },
        prevAll: function(a) {
            return f.dir(a, "previousSibling")
        },
        nextUntil: function(a, b, c) {
            return f.dir(a, "nextSibling", c)
        },
        prevUntil: function(a, b, c) {
            return f.dir(a, "previousSibling", c)
        },
        siblings: function(a) {
            return f.sibling(a.parentNode.firstChild, a)
        },
        children: function(a) {
            return f.sibling(a.firstChild)
        },
        contents: function(a) {
            return f.nodeName(a, "iframe") ? a.contentDocument || a.contentWindow.document : f.makeArray(a.childNodes)
        }
    }, function(a, b) {
        f.fn[a] = function(c, d) {
            var e = f.map(this, b, c), g = R.call(arguments);
            N.test(a) || (d = c), d && typeof d == "string" && (e = f.filter(d, e)), e = this.length > 1&&!T[a] ? f.unique(e) : e, (this.length > 1 || P.test(d)) && O.test(a) && (e = e.reverse());
            return this.pushStack(e, a, g.join(","))
        }
    }), f.extend({
        filter: function(a, b, c) {
            c && (a = ":not(" + a + ")");
            return b.length === 1 ? f.find.matchesSelector(b[0], a) ? [b[0]] : [] : f.find.matches(a, b)
        },
        dir: function(a, c, d) {
            var e = [], g = a[c];
            while (g && g.nodeType !== 9 && (d === b || g.nodeType !== 1 ||!f(g).is(d))) {
                g.nodeType === 1 && e.push(g), g = g[c]
            }
            return e
        },
        nth: function(a, b, c, d) {
            b = b || 1;
            var e = 0;
            for (; a; a = a[c]) {
                if (a.nodeType === 1&&++e === b) {
                    break
                }
            }
            return a
        },
        sibling: function(a, b) {
            var c = [];
            for (; a; a = a.nextSibling) {
                a.nodeType === 1 && a !== b && c.push(a)
            }
            return c
        }
    });
    var W = / jQuery\d+="(?:\d+|null)"/g, X = /^\s+/, Y = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\w:]+)[^>]*)\/>/ig, Z = /<([\w:]+)/, $ = /<tbody/i, _ = /<|&#?\w+;/, ba = /<(?:script|object|embed|option|style)/i, bb = /checked\s*(?:[^=]|=\s*.checked.)/i, bc = /\/(java|ecma)script/i, bd = /^\s*<!(?:\[CDATA\[|\-\-)/, be = {
        option: [1, "<select multiple='multiple'>", "</select>"],
        legend: [1, "<fieldset>", "</fieldset>"],
        thead: [1, "<table>", "</table>"],
        tr: [2, "<table><tbody>", "</tbody></table>"],
        td: [3, "<table><tbody><tr>", "</tr></tbody></table>"],
        col: [2, "<table><tbody></tbody><colgroup>", "</colgroup></table>"],
        area: [1, "<map>", "</map>"],
        _default: [0, "", ""]
    };
    be.optgroup = be.option, be.tbody = be.tfoot = be.colgroup = be.caption = be.thead, be.th = be.td, f.support.htmlSerialize || (be._default = [1, "div<div>", "</div>"]), f.fn.extend({
        text: function(a) {
            if (f.isFunction(a)) {
                return this.each(function(b) {
                    var c = f(this);
                    c.text(a.call(this, b, c.text()))
                })
            }
            if (typeof a != "object" && a !== b) {
                return this.empty().append((this[0] && this[0].ownerDocument || c).createTextNode(a))
            }
            return f.text(this)
        },
        wrapAll: function(a) {
            if (f.isFunction(a)) {
                return this.each(function(b) {
                    f(this).wrapAll(a.call(this, b))
                })
            }
            if (this[0]) {
                var b = f(a, this[0].ownerDocument).eq(0).clone(!0);
                this[0].parentNode && b.insertBefore(this[0]), b.map(function() {
                    var a = this;
                    while (a.firstChild && a.firstChild.nodeType === 1) {
                        a = a.firstChild
                    }
                    return a
                }).append(this)
            }
            return this
        },
        wrapInner: function(a) {
            if (f.isFunction(a)) {
                return this.each(function(b) {
                    f(this).wrapInner(a.call(this, b))
                })
            }
            return this.each(function() {
                var b = f(this), c = b.contents();
                c.length ? c.wrapAll(a) : b.append(a)
            })
        },
        wrap: function(a) {
            return this.each(function() {
                f(this).wrapAll(a)
            })
        },
        unwrap: function() {
            return this.parent().each(function() {
                f.nodeName(this, "body") || f(this).replaceWith(this.childNodes)
            }).end()
        },
        append: function() {
            return this.domManip(arguments, !0, function(a) {
                this.nodeType === 1 && this.appendChild(a)
            })
        },
        prepend: function() {
            return this.domManip(arguments, !0, function(a) {
                this.nodeType === 1 && this.insertBefore(a, this.firstChild)
            })
        },
        before: function() {
            if (this[0] && this[0].parentNode) {
                return this.domManip(arguments, !1, function(a) {
                    this.parentNode.insertBefore(a, this)
                })
            }
            if (arguments.length) {
                var a = f(arguments[0]);
                a.push.apply(a, this.toArray());
                return this.pushStack(a, "before", arguments)
            }
        },
        after: function() {
            if (this[0] && this[0].parentNode) {
                return this.domManip(arguments, !1, function(a) {
                    this.parentNode.insertBefore(a, this.nextSibling)
                })
            }
            if (arguments.length) {
                var a = this.pushStack(this, "after", arguments);
                a.push.apply(a, f(arguments[0]).toArray());
                return a
            }
        },
        remove: function(a, b) {
            for (var c = 0, d; (d = this[c]) != null; c++) {
                if (!a || f.filter(a, [d]).length) {
                    !b && d.nodeType === 1 && (f.cleanData(d.getElementsByTagName("*")), f.cleanData([d])), d.parentNode && d.parentNode.removeChild(d)
                }
            }
            return this
        },
        empty: function() {
            for (var a = 0, b; (b = this[a]) != null; a++) {
                b.nodeType === 1 && f.cleanData(b.getElementsByTagName("*"));
                while (b.firstChild) {
                    b.removeChild(b.firstChild)
                }
            }
            return this
        },
        clone: function(a, b) {
            a = a == null?!1 : a, b = b == null ? a : b;
            return this.map(function() {
                return f.clone(this, a, b)
            })
        },
        html: function(a) {
            if (a === b) {
                return this[0] && this[0].nodeType === 1 ? this[0].innerHTML.replace(W, "") : null
            }
            if (typeof a == "string"&&!ba.test(a) && (f.support.leadingWhitespace ||!X.test(a))&&!be[(Z.exec(a) || ["", ""])[1].toLowerCase()]) {
                a = a.replace(Y, "<$1></$2>");
                try {
                    for (var c = 0, d = this.length; c < d; c++) {
                        this[c].nodeType === 1 && (f.cleanData(this[c].getElementsByTagName("*")), this[c].innerHTML = a)
                    }
                } catch (e) {
                    this.empty().append(a)
                }
            } else {
                f.isFunction(a) ? this.each(function(b) {
                    var c = f(this);
                    c.html(a.call(this, b, c.html()))
                }) : this.empty().append(a)
            }
            return this
        },
        replaceWith: function(a) {
            if (this[0] && this[0].parentNode) {
                if (f.isFunction(a)) {
                    return this.each(function(b) {
                        var c = f(this), d = c.html();
                        c.replaceWith(a.call(this, b, d))
                    })
                }
                typeof a != "string" && (a = f(a).detach());
                return this.each(function() {
                    var b = this.nextSibling, c = this.parentNode;
                    f(this).remove(), b ? f(b).before(a) : f(c).append(a)
                })
            }
            return this.length ? this.pushStack(f(f.isFunction(a) ? a() : a), "replaceWith", a) : this
        },
        detach: function(a) {
            return this.remove(a, !0)
        },
        domManip: function(a, c, d) {
            var e, g, h, i, j = a[0], k = [];
            if (!f.support.checkClone && arguments.length === 3 && typeof j == "string" && bb.test(j)) {
                return this.each(function() {
                    f(this).domManip(a, c, d, !0)
                })
            }
            if (f.isFunction(j)) {
                return this.each(function(e) {
                    var g = f(this);
                    a[0] = j.call(this, e, c ? g.html() : b), g.domManip(a, c, d)
                })
            }
            if (this[0]) {
                i = j && j.parentNode, f.support.parentNode && i && i.nodeType === 11 && i.childNodes.length === this.length ? e = {
                    fragment: i
                } : e = f.buildFragment(a, this, k), h = e.fragment, h.childNodes.length === 1 ? g = h = h.firstChild : g = h.firstChild;
                if (g) {
                    c = c && f.nodeName(g, "tr");
                    for (var l = 0, m = this.length, n = m - 1; l < m; l++) {
                        d.call(c ? bf(this[l], g) : this[l], e.cacheable || m > 1 && l < n ? f.clone(h, !0, !0) : h)
                    }
                }
                k.length && f.each(k, bl)
            }
            return this
        }
    }), f.buildFragment = function(a, b, d) {
        var e, g, h, i;
        b && b[0] && (i = b[0].ownerDocument || b[0]), i.createDocumentFragment || (i = c), a.length === 1 && typeof a[0] == "string" && a[0].length < 512 && i === c && a[0].charAt(0) === "<"&&!ba.test(a[0]) && (f.support.checkClone ||!bb.test(a[0])) && (g=!0, h = f.fragments[a[0]], h && h !== 1 && (e = h)), e || (e = i.createDocumentFragment(), f.clean(a, i, e, d)), g && (f.fragments[a[0]] = h ? e : 1);
        return {
            fragment: e,
            cacheable: g
        }
    }, f.fragments = {}, f.each({
        appendTo: "append",
        prependTo: "prepend",
        insertBefore: "before",
        insertAfter: "after",
        replaceAll: "replaceWith"
    }, function(a, b) {
        f.fn[a] = function(c) {
            var d = [], e = f(c), g = this.length === 1 && this[0].parentNode;
            if (g && g.nodeType === 11 && g.childNodes.length === 1 && e.length === 1) {
                e[b](this[0]);
                return this
            }
            for (var h = 0, i = e.length; h < i; h++) {
                var j = (h > 0 ? this.clone(!0) : this).get();
                f(e[h])[b](j), d = d.concat(j)
            }
            return this.pushStack(d, a, e.selector)
        }
    }), f.extend({
        clone: function(a, b, c) {
            var d = a.cloneNode(!0), e, g, h;
            if ((!f.support.noCloneEvent ||!f.support.noCloneChecked) && (a.nodeType === 1 || a.nodeType === 11)&&!f.isXMLDoc(a)) {
                bh(a, d), e = bi(a), g = bi(d);
                for (h = 0; e[h]; ++h) {
                    g[h] && bh(e[h], g[h])
                }
            }
            if (b) {
                bg(a, d);
                if (c) {
                    e = bi(a), g = bi(d);
                    for (h = 0; e[h]; ++h) {
                        bg(e[h], g[h])
                    }
                }
            }
            e = g = null;
            return d
        },
        clean: function(a, b, d, e) {
            var g;
            b = b || c, typeof b.createElement == "undefined" && (b = b.ownerDocument || b[0] && b[0].ownerDocument || c);
            var h = [], i;
            for (var j = 0, k; (k = a[j]) != null; j++) {
                typeof k == "number" && (k += "");
                if (!k) {
                    continue
                }
                if (typeof k == "string") {
                    if (!_.test(k)) {
                        k = b.createTextNode(k)
                    } else {
                        k = k.replace(Y, "<$1></$2>");
                        var l = (Z.exec(k) || ["", ""])[1].toLowerCase(), m = be[l] || be._default, n = m[0], o = b.createElement("div");
                        o.innerHTML = m[1] + k + m[2];
                        while (n--) {
                            o = o.lastChild
                        }
                        if (!f.support.tbody) {
                            var p = $.test(k), q = l === "table"&&!p ? o.firstChild && o.firstChild.childNodes: m[1] === "<table>"&&!p ? o.childNodes: [];
                            for (i = q.length - 1; i >= 0; --i) {
                                f.nodeName(q[i], "tbody")&&!q[i].childNodes.length && q[i].parentNode.removeChild(q[i])
                            }
                        }
                        !f.support.leadingWhitespace && X.test(k) && o.insertBefore(b.createTextNode(X.exec(k)[0]), o.firstChild), k = o.childNodes
                    }
                }
                var r;
                if (!f.support.appendChecked) {
                    if (k[0] && typeof(r = k.length) == "number") {
                        for (i = 0; i < r; i++) {
                            bk(k[i])
                        }
                    } else {
                        bk(k)
                    }
                }
                k.nodeType ? h.push(k) : h = f.merge(h, k)
            }
            if (d) {
                g = function(a) {
                    return !a.type || bc.test(a.type)
                };
                for (j = 0; h[j]; j++) {
                    if (e && f.nodeName(h[j], "script") && (!h[j].type || h[j].type.toLowerCase() === "text/javascript")) {
                        e.push(h[j].parentNode ? h[j].parentNode.removeChild(h[j]) : h[j])
                    } else {
                        if (h[j].nodeType === 1) {
                            var s = f.grep(h[j].getElementsByTagName("script"), g);
                            h.splice.apply(h, [j + 1, 0].concat(s))
                        }
                        d.appendChild(h[j])
                    }
                }
            }
            return h
        },
        cleanData: function(a) {
            var b, c, d = f.cache, e = f.expando, g = f.event.special, h = f.support.deleteExpando;
            for (var i = 0, j; (j = a[i]) != null; i++) {
                if (j.nodeName && f.noData[j.nodeName.toLowerCase()]) {
                    continue
                }
                c = j[f.expando];
                if (c) {
                    b = d[c] && d[c][e];
                    if (b && b.events) {
                        for (var k in b.events) {
                            g[k] ? f.event.remove(j, k) : f.removeEvent(j, k, b.handle)
                        }
                        b.handle && (b.handle.elem = null)
                    }
                    h ? delete j[f.expando] : j.removeAttribute && j.removeAttribute(f.expando), delete d[c]
                }
            }
        }
    });
    var bm = /alpha\([^)]*\)/i, bn = /opacity=([^)]*)/, bo = /([A-Z]|^ms)/g, bp = /^-?\d+(?:px)?$/i, bq = /^-?\d/, br = /^([\-+])=([\-+.\de]+)/, bs = {
        position: "absolute",
        visibility: "hidden",
        display: "block"
    }, bt = ["Left", "Right"], bu = ["Top", "Bottom"], bv, bw, bx;
    f.fn.css = function(a, c) {
        if (arguments.length === 2 && c === b) {
            return this
        }
        return f.access(this, a, c, !0, function(a, c, d) {
            return d !== b ? f.style(a, c, d) : f.css(a, c)
        })
    }, f.extend({
        cssHooks: {
            opacity: {
                get: function(a, b) {
                    if (b) {
                        var c = bv(a, "opacity", "opacity");
                        return c === "" ? "1" : c
                    }
                    return a.style.opacity
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
            "float": f.support.cssFloat ? "cssFloat": "styleFloat"
        },
        style: function(a, c, d, e) {
            if (!!a && a.nodeType !== 3 && a.nodeType !== 8&&!!a.style) {
                var g, h, i = f.camelCase(c), j = a.style, k = f.cssHooks[i];
                c = f.cssProps[i] || i;
                if (d === b) {
                    if (k && "get" in k && (g = k.get(a, !1, e)) !== b) {
                        return g
                    }
                    return j[c]
                }
                h = typeof d, h === "string" && (g = br.exec(d)) && (d =+ (g[1] + 1)*+g[2] + parseFloat(f.css(a, c)), h = "number");
                if (d == null || h === "number" && isNaN(d)) {
                    return 
                }
                h === "number"&&!f.cssNumber[i] && (d += "px");
                if (!k ||!("set" in k) || (d = k.set(a, d)) !== b) {
                    try {
                        j[c] = d
                    } catch (l) {}
                }
            }
        },
        css: function(a, c, d) {
            var e, g;
            c = f.camelCase(c), g = f.cssHooks[c], c = f.cssProps[c] || c, c === "cssFloat" && (c = "float");
            if (g && "get" in g && (e = g.get(a, !0, d)) !== b) {
                return e
            }
            if (bv) {
                return bv(a, c)
            }
        },
        swap: function(a, b, c) {
            var d = {};
            for (var e in b) {
                d[e] = a.style[e], a.style[e] = b[e]
            }
            c.call(a);
            for (e in b) {
                a.style[e] = d[e]
            }
        }
    }), f.curCSS = f.css, f.each(["height", "width"], function(a, b) {
        f.cssHooks[b] = {
            get: function(a, c, d) {
                var e;
                if (c) {
                    if (a.offsetWidth !== 0) {
                        return by(a, b, d)
                    }
                    f.swap(a, bs, function() {
                        e = by(a, b, d)
                    });
                    return e
                }
            },
            set: function(a, b) {
                if (!bp.test(b)) {
                    return b
                }
                b = parseFloat(b);
                if (b >= 0) {
                    return b + "px"
                }
            }
        }
    }), f.support.opacity || (f.cssHooks.opacity = {
        get: function(a, b) {
            return bn.test((b && a.currentStyle ? a.currentStyle.filter : a.style.filter) || "") ? parseFloat(RegExp.$1) / 100 + "" : b ? "1" : ""
        },
        set: function(a, b) {
            var c = a.style, d = a.currentStyle, e = f.isNaN(b) ? "": "alpha(opacity=" + b * 100 + ")", g = d && d.filter || c.filter || "";
            c.zoom = 1;
            if (b >= 1 && f.trim(g.replace(bm, "")) === "") {
                c.removeAttribute("filter");
                if (d&&!d.filter) {
                    return 
                }
            }
            c.filter = bm.test(g) ? g.replace(bm, e) : g + " " + e
        }
    }), f(function() {
        f.support.reliableMarginRight || (f.cssHooks.marginRight = {
            get: function(a, b) {
                var c;
                f.swap(a, {
                    display: "inline-block"
                }, function() {
                    b ? c = bv(a, "margin-right", "marginRight") : c = a.style.marginRight
                });
                return c
            }
        })
    }), c.defaultView && c.defaultView.getComputedStyle && (bw = function(a, c) {
        var d, e, g;
        c = c.replace(bo, "-$1").toLowerCase();
        if (!(e = a.ownerDocument.defaultView)) {
            return b
        }
        if (g = e.getComputedStyle(a, null)) {
            d = g.getPropertyValue(c), d === ""&&!f.contains(a.ownerDocument.documentElement, a) && (d = f.style(a, c))
        }
        return d
    }), c.documentElement.currentStyle && (bx = function(a, b) {
        var c, d = a.currentStyle && a.currentStyle[b], e = a.runtimeStyle && a.runtimeStyle[b], f = a.style;
        !bp.test(d) && bq.test(d) && (c = f.left, e && (a.runtimeStyle.left = a.currentStyle.left), f.left = b === "fontSize" ? "1em" : d || 0, d = f.pixelLeft + "px", f.left = c, e && (a.runtimeStyle.left = e));
        return d === "" ? "auto" : d
    }), bv = bw || bx, f.expr && f.expr.filters && (f.expr.filters.hidden = function(a) {
        var b = a.offsetWidth, c = a.offsetHeight;
        return b === 0 && c === 0 ||!f.support.reliableHiddenOffsets && (a.style.display || f.css(a, "display")) === "none"
    }, f.expr.filters.visible = function(a) {
        return !f.expr.filters.hidden(a)
    });
    var bz = /%20/g, bA = /\[\]$/, bB = /\r?\n/g, bC = /#.*$/, bD = /^(.*?):[ \t]*([^\r\n]*)\r?$/mg, bE = /^(?:color|date|datetime|datetime-local|email|hidden|month|number|password|range|search|tel|text|time|url|week)$/i, bF = /^(?:about|app|app\-storage|.+\-extension|file|res|widget):$/, bG = /^(?:GET|HEAD)$/, bH = /^\/\//, bI = /\?/, bJ = /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, bK = /^(?:select|textarea)/i, bL = /\s+/, bM = /([?&])_=[^&]*/, bN = /^([\w\+\.\-]+:)(?:\/\/([^\/?#:]*)(?::(\d+))?)?/, bO = f.fn.load, bP = {}, bQ = {}, bR, bS, bT = ["*/"] + ["*"];
    try {
        bR = e.href
    } catch (bU) {
        bR = c.createElement("a"), bR.href = "", bR = bR.href
    }
    bS = bN.exec(bR.toLowerCase()) || [], f.fn.extend({
        load: function(a, c, d) {
            if (typeof a != "string" && bO) {
                return bO.apply(this, arguments)
            }
            if (!this.length) {
                return this
            }
            var e = a.indexOf(" ");
            if (e >= 0) {
                var g = a.slice(e, a.length);
                a = a.slice(0, e)
            }
            var h = "GET";
            c && (f.isFunction(c) ? (d = c, c = b) : typeof c == "object" && (c = f.param(c, f.ajaxSettings.traditional), h = "POST"));
            var i = this;
            f.ajax({
                url: a,
                type: h,
                dataType: "html",
                data: c,
                complete: function(a, b, c) {
                    c = a.responseText, a.isResolved() && (a.done(function(a) {
                        c = a
                    }), i.html(g ? f("<div>").append(c.replace(bJ, "")).find(g) : c)), d && i.each(d, [c, b, a])
                }
            });
            return this
        },
        serialize: function() {
            return f.param(this.serializeArray())
        },
        serializeArray: function() {
            return this.map(function() {
                return this.elements ? f.makeArray(this.elements) : this
            }).filter(function() {
                return this.name&&!this.disabled && (this.checked || bK.test(this.nodeName) || bE.test(this.type))
            }).map(function(a, b) {
                var c = f(this).val();
                return c == null ? null : f.isArray(c) ? f.map(c, function(a, c) {
                    return {
                        name: b.name,
                        value: a.replace(bB, "\r\n")
                    }
                }) : {
                    name: b.name,
                    value: c.replace(bB, "\r\n")
                }
            }).get()
        }
    }), f.each("ajaxStart ajaxStop ajaxComplete ajaxError ajaxSuccess ajaxSend".split(" "), function(a, b) {
        f.fn[b] = function(a) {
            return this.bind(b, a)
        }
    }), f.each(["get", "post"], function(a, c) {
        f[c] = function(a, d, e, g) {
            f.isFunction(d) && (g = g || e, e = d, d = b);
            return f.ajax({
                type: c,
                url: a,
                data: d,
                success: e,
                dataType: g
            })
        }
    }), f.extend({
        getScript: function(a, c) {
            return f.get(a, b, c, "script")
        },
        getJSON: function(a, b, c) {
            return f.get(a, b, c, "json")
        },
        ajaxSetup: function(a, b) {
            b ? bX(a, f.ajaxSettings) : (b = a, a = f.ajaxSettings), bX(a, b);
            return a
        },
        ajaxSettings: {
            url: bR,
            isLocal: bF.test(bS[1]),
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
                "*": bT
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
                "* text": a.String,
                "text html": !0,
                "text json": f.parseJSON,
                "text xml": f.parseXML
            },
            flatOptions: {
                context: !0,
                url: !0
            }
        },
        ajaxPrefilter: bV(bP),
        ajaxTransport: bV(bQ),
        ajax: function(a, c) {
            function w(a, c, l, m) {
                if (s !== 2) {
                    s = 2, q && clearTimeout(q), p = b, n = m || "", v.readyState = a > 0 ? 4 : 0;
                    var o, r, u, w = c, x = l ? bZ(d, v, l): b, y, z;
                    if (a >= 200 && a < 300 || a === 304) {
                        if (d.ifModified) {
                            if (y = v.getResponseHeader("Last-Modified")) {
                                f.lastModified[k] = y
                            }
                            if (z = v.getResponseHeader("Etag")) {
                                f.etag[k] = z
                            }
                        }
                        if (a === 304) {
                            w = "notmodified", o=!0
                        } else {
                            try {
                                r = b$(d, x), w = "success", o=!0
                            } catch (A) {
                                w = "parsererror", u = A
                            }
                        }
                    } else {
                        u = w;
                        if (!w || a) {
                            w = "error", a < 0 && (a = 0)
                        }
                    }
                    v.status = a, v.statusText = "" + (c || w), o ? h.resolveWith(e, [r, w, v]) : h.rejectWith(e, [v, w, u]), v.statusCode(j), j = b, t && g.trigger("ajax" + (o ? "Success" : "Error"), [v, d, o ? r: u]), i.resolveWith(e, [v, w]), t && (g.trigger("ajaxComplete", [v, d]), --f.active || f.event.trigger("ajaxStop"))
                }
            }
            typeof a == "object" && (c = a, a = b), c = c || {};
            var d = f.ajaxSetup({}, c), e = d.context || d, g = e !== d && (e.nodeType || e instanceof f) ? f(e): f.event, h = f.Deferred(), i = f._Deferred(), j = d.statusCode || {}, k, l = {}, m = {}, n, o, p, q, r, s = 0, t, u, v = {
                readyState: 0,
                setRequestHeader: function(a, b) {
                    if (!s) {
                        var c = a.toLowerCase();
                        a = m[c] = m[c] || a, l[a] = b
                    }
                    return this
                },
                getAllResponseHeaders: function() {
                    return s === 2 ? n : null
                },
                getResponseHeader: function(a) {
                    var c;
                    if (s === 2) {
                        if (!o) {
                            o = {};
                            while (c = bD.exec(n)) {
                                o[c[1].toLowerCase()] = c[2]
                            }
                        }
                        c = o[a.toLowerCase()]
                    }
                    return c === b ? null : c
                },
                overrideMimeType: function(a) {
                    s || (d.mimeType = a);
                    return this
                },
                abort: function(a) {
                    a = a || "abort", p && p.abort(a), w(0, a);
                    return this
                }
            };
            h.promise(v), v.success = v.done, v.error = v.fail, v.complete = i.done, v.statusCode = function(a) {
                if (a) {
                    var b;
                    if (s < 2) {
                        for (b in a) {
                            j[b] = [j[b], a[b]]
                        }
                    } else {
                        b = a[v.status], v.then(b, b)
                    }
                }
                return this
            }, d.url = ((a || d.url) + "").replace(bC, "").replace(bH, bS[1] + "//"), d.dataTypes = f.trim(d.dataType || "*").toLowerCase().split(bL), d.crossDomain == null && (r = bN.exec(d.url.toLowerCase()), d.crossDomain=!(!r || r[1] == bS[1] && r[2] == bS[2] && (r[3] || (r[1] === "http:" ? 80 : 443)) == (bS[3] || (bS[1] === "http:" ? 80 : 443)))), d.data && d.processData && typeof d.data != "string" && (d.data = f.param(d.data, d.traditional)), bW(bP, d, c, v);
            if (s === 2) {
                return !1
            }
            t = d.global, d.type = d.type.toUpperCase(), d.hasContent=!bG.test(d.type), t && f.active++===0 && f.event.trigger("ajaxStart");
            if (!d.hasContent) {
                d.data && (d.url += (bI.test(d.url) ? "&" : "?") + d.data, delete d.data), k = d.url;
                if (d.cache===!1) {
                    var x = f.now(), y = d.url.replace(bM, "$1_=" + x);
                    d.url = y + (y === d.url ? (bI.test(d.url) ? "&" : "?") + "_=" + x : "")
                }
            }(d.data && d.hasContent && d.contentType!==!1 || c.contentType) && v.setRequestHeader("Content-Type", d.contentType), d.ifModified && (k = k || d.url, f.lastModified[k] && v.setRequestHeader("If-Modified-Since", f.lastModified[k]), f.etag[k] && v.setRequestHeader("If-None-Match", f.etag[k])), v.setRequestHeader("Accept", d.dataTypes[0] && d.accepts[d.dataTypes[0]] ? d.accepts[d.dataTypes[0]] + (d.dataTypes[0] !== "*" ? ", " + bT + "; q=0.01" : "") : d.accepts["*"]);
            for (u in d.headers) {
                v.setRequestHeader(u, d.headers[u])
            }
            if (d.beforeSend && (d.beforeSend.call(e, v, d)===!1 || s === 2)) {
                v.abort();
                return !1
            }
            for (u in {
                success: 1,
                error: 1,
                complete: 1
            }) {
                v[u](d[u])
            }
            p = bW(bQ, d, c, v);
            if (!p) {
                w( - 1, "No Transport")
            } else {
                v.readyState = 1, t && g.trigger("ajaxSend", [v, d]), d.async && d.timeout > 0 && (q = setTimeout(function() {
                    v.abort("timeout")
                }, d.timeout));
                try {
                    s = 1, p.send(l, w)
                } catch (z) {
                    s < 2 ? w( - 1, z) : f.error(z)
                }
            }
            return v
        },
        param: function(a, c) {
            var d = [], e = function(a, b) {
                b = f.isFunction(b) ? b() : b, d[d.length] = encodeURIComponent(a) + "=" + encodeURIComponent(b)
            };
            c === b && (c = f.ajaxSettings.traditional);
            if (f.isArray(a) || a.jquery&&!f.isPlainObject(a)) {
                f.each(a, function() {
                    e(this.name, this.value)
                })
            } else {
                for (var g in a) {
                    bY(g, a[g], c, e)
                }
            }
            return d.join("&").replace(bz, "+")
        }
    }), f.extend({
        active: 0,
        lastModified: {},
        etag: {}
    });
    var b_ = f.now(), ca = /(\=)\?(&|$)|\?\?/i;
    f.ajaxSetup({
        jsonp: "callback",
        jsonpCallback: function() {
            return f.expando + "_" + b_++
        }
    }), f.ajaxPrefilter("json jsonp", function(b, c, d) {
        var e = b.contentType === "application/x-www-form-urlencoded" && typeof b.data == "string";
        if (b.dataTypes[0] === "jsonp" || b.jsonp!==!1 && (ca.test(b.url) || e && ca.test(b.data))) {
            var g, h = b.jsonpCallback = f.isFunction(b.jsonpCallback) ? b.jsonpCallback(): b.jsonpCallback, i = a[h], j = b.url, k = b.data, l = "$1" + h + "$2";
            b.jsonp!==!1 && (j = j.replace(ca, l), b.url === j && (e && (k = k.replace(ca, l)), b.data === k && (j += (/\?/.test(j) ? "&" : "?") + b.jsonp + "=" + h))), b.url = j, b.data = k, a[h] = function(a) {
                g = [a]
            }, d.always(function() {
                a[h] = i, g && f.isFunction(i) && a[h](g[0])
            }), b.converters["script json"] = function() {
                g || f.error(h + " was not called");
                return g[0]
            }, b.dataTypes[0] = "json";
            return "script"
        }
    }), f.ajaxSetup({
        accepts: {
            script: "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript"
        },
        contents: {
            script: /javascript|ecmascript/
        },
        converters: {
            "text script": function(a) {
                f.globalEval(a);
                return a
            }
        }
    }), f.ajaxPrefilter("script", function(a) {
        a.cache === b && (a.cache=!1), a.crossDomain && (a.type = "GET", a.global=!1)
    }), f.ajaxTransport("script", function(a) {
        if (a.crossDomain) {
            var d, e = c.head || c.getElementsByTagName("head")[0] || c.documentElement;
            return {
                send: function(f, g) {
                    d = c.createElement("script"), d.async = "async", a.scriptCharset && (d.charset = a.scriptCharset), d.src = a.url, d.onload = d.onreadystatechange = function(a, c) {
                        if (c ||!d.readyState || /loaded|complete/.test(d.readyState)) {
                            d.onload = d.onreadystatechange = null, e && d.parentNode && e.removeChild(d), d = b, c || g(200, "success")
                        }
                    }, e.insertBefore(d, e.firstChild)
                },
                abort: function() {
                    d && d.onload(0, 1)
                }
            }
        }
    });
    var cb = a.ActiveXObject ? function() {
        for (var a in cd) {
            cd[a](0, 1)
        }
    }
    : !1, cc = 0, cd;
    f.ajaxSettings.xhr = a.ActiveXObject ? function() {
        return !this.isLocal && ce() || cf()
    } : ce, function(a) {
        f.extend(f.support, {
            ajax: !!a,
            cors: !!a && "withCredentials" in a
        })
    }(f.ajaxSettings.xhr()), f.support.ajax && f.ajaxTransport(function(c) {
        if (!c.crossDomain || f.support.cors) {
            var d;
            return {
                send: function(e, g) {
                    var h = c.xhr(), i, j;
                    c.username ? h.open(c.type, c.url, c.async, c.username, c.password) : h.open(c.type, c.url, c.async);
                    if (c.xhrFields) {
                        for (j in c.xhrFields) {
                            h[j] = c.xhrFields[j]
                        }
                    }
                    c.mimeType && h.overrideMimeType && h.overrideMimeType(c.mimeType), !c.crossDomain&&!e["X-Requested-With"] && (e["X-Requested-With"] = "XMLHttpRequest");
                    try {
                        for (j in e) {
                            h.setRequestHeader(j, e[j])
                        }
                    } catch (k) {}
                    h.send(c.hasContent && c.data || null), d = function(a, e) {
                        var j, k, l, m, n;
                        try {
                            if (d && (e || h.readyState === 4)) {
                                d = b, i && (h.onreadystatechange = f.noop, cb && delete cd[i]);
                                if (e) {
                                    h.readyState !== 4 && h.abort()
                                } else {
                                    j = h.status, l = h.getAllResponseHeaders(), m = {}, n = h.responseXML, n && n.documentElement && (m.xml = n), m.text = h.responseText;
                                    try {
                                        k = h.statusText
                                    } catch (o) {
                                        k = ""
                                    }
                                    !j && c.isLocal&&!c.crossDomain ? j = m.text ? 200 : 404 : j === 1223 && (j = 204)
                                }
                            }
                        } catch (p) {
                            e || g( - 1, p)
                        }
                        m && g(j, k, m, l)
                    }, !c.async || h.readyState === 4 ? d() : (i=++cc, cb && (cd || (cd = {}, f(a).unload(cb)), cd[i] = d), h.onreadystatechange = d)
                },
                abort: function() {
                    d && d(0, 1)
                }
            }
        }
    });
    var cg = {}, ch, ci, cj = /^(?:toggle|show|hide)$/, ck = /^([+\-]=)?([\d+.\-]+)([a-z%]*)$/i, cl, cm = [["height", "marginTop", "marginBottom", "paddingTop", "paddingBottom"], ["width", "marginLeft", "marginRight", "paddingLeft", "paddingRight"], ["opacity"]], cn;
    f.fn.extend({
        show: function(a, b, c) {
            var d, e;
            if (a || a === 0) {
                return this.animate(cq("show", 3), a, b, c)
            }
            for (var g = 0, h = this.length; g < h; g++) {
                d = this[g], d.style && (e = d.style.display, !f._data(d, "olddisplay") && e === "none" && (e = d.style.display = ""), e === "" && f.css(d, "display") === "none" && f._data(d, "olddisplay", cr(d.nodeName)))
            }
            for (g = 0; g < h; g++) {
                d = this[g];
                if (d.style) {
                    e = d.style.display;
                    if (e === "" || e === "none") {
                        d.style.display = f._data(d, "olddisplay") || ""
                    }
                }
            }
            return this
        },
        hide: function(a, b, c) {
            if (a || a === 0) {
                return this.animate(cq("hide", 3), a, b, c)
            }
            for (var d = 0, e = this.length; d < e; d++) {
                if (this[d].style) {
                    var g = f.css(this[d], "display");
                    g !== "none"&&!f._data(this[d], "olddisplay") && f._data(this[d], "olddisplay", g)
                }
            }
            for (d = 0; d < e; d++) {
                this[d].style && (this[d].style.display = "none")
            }
            return this
        },
        _toggle: f.fn.toggle,
        toggle: function(a, b, c) {
            var d = typeof a == "boolean";
            f.isFunction(a) && f.isFunction(b) ? this._toggle.apply(this, arguments) : a == null || d ? this.each(function() {
                var b = d ? a: f(this).is(":hidden");
                f(this)[b ? "show": "hide"]()
            }) : this.animate(cq("toggle", 3), a, b, c);
            return this
        },
        fadeTo: function(a, b, c, d) {
            return this.filter(":hidden").css("opacity", 0).show().end().animate({
                opacity: b
            }, a, c, d)
        },
        animate: function(a, b, c, d) {
            var e = f.speed(b, c, d);
            if (f.isEmptyObject(a)) {
                return this.each(e.complete, [!1])
            }
            a = f.extend({}, a);
            return this[e.queue===!1 ? "each": "queue"](function() {
                e.queue===!1 && f._mark(this);
                var b = f.extend({}, e), c = this.nodeType === 1, d = c && f(this).is(":hidden"), g, h, i, j, k, l, m, n, o;
                b.animatedProperties = {};
                for (i in a) {
                    g = f.camelCase(i), i !== g && (a[g] = a[i], delete a[i]), h = a[g], f.isArray(h) ? (b.animatedProperties[g] = h[1], h = a[g] = h[0]) : b.animatedProperties[g] = b.specialEasing && b.specialEasing[g] || b.easing || "swing";
                    if (h === "hide" && d || h === "show"&&!d) {
                        return b.complete.call(this)
                    }
                    c && (g === "height" || g === "width") && (b.overflow = [this.style.overflow, this.style.overflowX, this.style.overflowY], f.css(this, "display") === "inline" && f.css(this, "float") === "none" && (f.support.inlineBlockNeedsLayout ? (j = cr(this.nodeName), j === "inline" ? this.style.display = "inline-block" : (this.style.display = "inline", this.style.zoom = 1)) : this.style.display = "inline-block"))
                }
                b.overflow != null && (this.style.overflow = "hidden");
                for (i in a) {
                    k = new f.fx(this, b, i), h = a[i], cj.test(h) ? k[h === "toggle" ? d ? "show": "hide": h]() : (l = ck.exec(h), m = k.cur(), l ? (n = parseFloat(l[2]), o = l[3] || (f.cssNumber[i] ? "" : "px"), o !== "px" && (f.style(this, i, (n || 1) + o), m = (n || 1) / k.cur() * m, f.style(this, i, m + o)), l[1] && (n = (l[1] === "-="?-1 : 1) * n + m), k.custom(m, n, o)) : k.custom(m, h, ""))
                }
                return !0
            })
        },
        stop: function(a, b) {
            a && this.queue([]), this.each(function() {
                var a = f.timers, c = a.length;
                b || f._unmark(!0, this);
                while (c--) {
                    a[c].elem === this && (b && a[c](!0), a.splice(c, 1))
                }
            }), b || this.dequeue();
            return this
        }
    }), f.each({
        slideDown: cq("show", 1),
        slideUp: cq("hide", 1),
        slideToggle: cq("toggle", 1),
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
        f.fn[a] = function(a, c, d) {
            return this.animate(b, a, c, d)
        }
    }), f.extend({
        speed: function(a, b, c) {
            var d = a && typeof a == "object" ? f.extend({}, a): {
                complete: c ||!c && b || f.isFunction(a) && a,
                duration: a,
                easing: c && b || b&&!f.isFunction(b) && b
            };
            d.duration = f.fx.off ? 0 : typeof d.duration == "number" ? d.duration : d.duration in f.fx.speeds ? f.fx.speeds[d.duration] : f.fx.speeds._default, d.old = d.complete, d.complete = function(a) {
                f.isFunction(d.old) && d.old.call(this), d.queue!==!1 ? f.dequeue(this) : a!==!1 && f._unmark(this)
            };
            return d
        },
        easing: {
            linear: function(a, b, c, d) {
                return c + d * a
            },
            swing: function(a, b, c, d) {
                return ( - Math.cos(a * Math.PI) / 2 + 0.5) * d + c
            }
        },
        timers: [],
        fx: function(a, b, c) {
            this.options = b, this.elem = a, this.prop = c, b.orig = b.orig || {}
        }
    }), f.fx.prototype = {
        update: function() {
            this.options.step && this.options.step.call(this.elem, this.now, this), (f.fx.step[this.prop] || f.fx.step._default)(this)
        },
        cur: function() {
            if (this.elem[this.prop] != null && (!this.elem.style || this.elem.style[this.prop] == null)) {
                return this.elem[this.prop]
            }
            var a, b = f.css(this.elem, this.prop);
            return isNaN(a = parseFloat(b))?!b || b === "auto" ? 0 : b : a
        },
        custom: function(a, b, c) {
            function g(a) {
                return d.step(a)
            }
            var d = this, e = f.fx;
            this.startTime = cn || co(), this.start = a, this.end = b, this.unit = c || this.unit || (f.cssNumber[this.prop] ? "" : "px"), this.now = this.start, this.pos = this.state = 0, g.elem = this.elem, g() && f.timers.push(g)&&!cl && (cl = setInterval(e.tick, e.interval))
        },
        show: function() {
            this.options.orig[this.prop] = f.style(this.elem, this.prop), this.options.show=!0, this.custom(this.prop === "width" || this.prop === "height" ? 1 : 0, this.cur()), f(this.elem).show()
        },
        hide: function() {
            this.options.orig[this.prop] = f.style(this.elem, this.prop), this.options.hide=!0, this.custom(this.cur(), 0)
        },
        step: function(a) {
            var b = cn || co(), c=!0, d = this.elem, e = this.options, g, h;
            if (a || b >= e.duration + this.startTime) {
                this.now = this.end, this.pos = this.state = 1, this.update(), e.animatedProperties[this.prop]=!0;
                for (g in e.animatedProperties) {
                    e.animatedProperties[g]!==!0 && (c=!1)
                }
                if (c) {
                    e.overflow != null&&!f.support.shrinkWrapBlocks && f.each(["", "X", "Y"], function(a, b) {
                        d.style["overflow" + b] = e.overflow[a]
                    }), e.hide && f(d).hide();
                    if (e.hide || e.show) {
                        for (var i in e.animatedProperties) {
                            f.style(d, i, e.orig[i])
                        }
                    }
                    e.complete.call(d)
                }
                return !1
            }
            e.duration == Infinity ? this.now = b : (h = b - this.startTime, this.state = h / e.duration, this.pos = f.easing[e.animatedProperties[this.prop]](this.state, h, 0, 1, e.duration), this.now = this.start + (this.end - this.start) * this.pos), this.update();
            return !0
        }
    }, f.extend(f.fx, {
        tick: function() {
            for (var a = f.timers, b = 0; b < a.length; ++b) {
                a[b]() || a.splice(b--, 1)
            }
            a.length || f.fx.stop()
        },
        interval: 13,
        stop: function() {
            clearInterval(cl), cl = null
        },
        speeds: {
            slow: 600,
            fast: 200,
            _default: 400
        },
        step: {
            opacity: function(a) {
                f.style(a.elem, "opacity", a.now)
            },
            _default: function(a) {
                a.elem.style && a.elem.style[a.prop] != null ? a.elem.style[a.prop] = (a.prop === "width" || a.prop === "height" ? Math.max(0, a.now) : a.now) + a.unit : a.elem[a.prop] = a.now
            }
        }
    }), f.expr && f.expr.filters && (f.expr.filters.animated = function(a) {
        return f.grep(f.timers, function(b) {
            return a === b.elem
        }).length
    });
    var cs = /^t(?:able|d|h)$/i, ct = /^(?:body|html)$/i;
    "getBoundingClientRect" in c.documentElement ? f.fn.offset = function(a) {
        var b = this[0], c;
        if (a) {
            return this.each(function(b) {
                f.offset.setOffset(this, a, b)
            })
        }
        if (!b ||!b.ownerDocument) {
            return null
        }
        if (b === b.ownerDocument.body) {
            return f.offset.bodyOffset(b)
        }
        try {
            c = b.getBoundingClientRect()
        } catch (d) {}
        var e = b.ownerDocument, g = e.documentElement;
        if (!c ||!f.contains(g, b)) {
            return c ? {
                top: c.top,
                left: c.left
            } : {
                top: 0,
                left: 0
            }
        }
        var h = e.body, i = cu(e), j = g.clientTop || h.clientTop || 0, k = g.clientLeft || h.clientLeft || 0, l = i.pageYOffset || f.support.boxModel && g.scrollTop || h.scrollTop, m = i.pageXOffset || f.support.boxModel && g.scrollLeft || h.scrollLeft, n = c.top + l - j, o = c.left + m - k;
        return {
            top: n,
            left: o
        }
    } : f.fn.offset = function(a) {
        var b = this[0];
        if (a) {
            return this.each(function(b) {
                f.offset.setOffset(this, a, b)
            })
        }
        if (!b ||!b.ownerDocument) {
            return null
        }
        if (b === b.ownerDocument.body) {
            return f.offset.bodyOffset(b)
        }
        f.offset.initialize();
        var c, d = b.offsetParent, e = b, g = b.ownerDocument, h = g.documentElement, i = g.body, j = g.defaultView, k = j ? j.getComputedStyle(b, null): b.currentStyle, l = b.offsetTop, m = b.offsetLeft;
        while ((b = b.parentNode) && b !== i && b !== h) {
            if (f.offset.supportsFixedPosition && k.position === "fixed") {
                break
            }
            c = j ? j.getComputedStyle(b, null) : b.currentStyle, l -= b.scrollTop, m -= b.scrollLeft, b === d && (l += b.offsetTop, m += b.offsetLeft, f.offset.doesNotAddBorder && (!f.offset.doesAddBorderForTableAndCells ||!cs.test(b.nodeName)) && (l += parseFloat(c.borderTopWidth) || 0, m += parseFloat(c.borderLeftWidth) || 0), e = d, d = b.offsetParent), f.offset.subtractsBorderForOverflowNotVisible && c.overflow !== "visible" && (l += parseFloat(c.borderTopWidth) || 0, m += parseFloat(c.borderLeftWidth) || 0), k = c
        }
        if (k.position === "relative" || k.position === "static") {
            l += i.offsetTop, m += i.offsetLeft
        }
        f.offset.supportsFixedPosition && k.position === "fixed" && (l += Math.max(h.scrollTop, i.scrollTop), m += Math.max(h.scrollLeft, i.scrollLeft));
        return {
            top: l,
            left: m
        }
    }, f.offset = {
        initialize: function() {
            var a = c.body, b = c.createElement("div"), d, e, g, h, i = parseFloat(f.css(a, "marginTop")) || 0, j = "<div style='position:absolute;top:0;left:0;margin:0;border:5px solid #000;padding:0;width:1px;height:1px;'><div></div></div><table style='position:absolute;top:0;left:0;margin:0;border:5px solid #000;padding:0;width:1px;height:1px;' cellpadding='0' cellspacing='0'><tr><td></td></tr></table>";
            f.extend(b.style, {
                position: "absolute",
                top: 0,
                left: 0,
                margin: 0,
                border: 0,
                width: "1px",
                height: "1px",
                visibility: "hidden"
            }), b.innerHTML = j, a.insertBefore(b, a.firstChild), d = b.firstChild, e = d.firstChild, h = d.nextSibling.firstChild.firstChild, this.doesNotAddBorder = e.offsetTop !== 5, this.doesAddBorderForTableAndCells = h.offsetTop === 5, e.style.position = "fixed", e.style.top = "20px", this.supportsFixedPosition = e.offsetTop === 20 || e.offsetTop === 15, e.style.position = e.style.top = "", d.style.overflow = "hidden", d.style.position = "relative", this.subtractsBorderForOverflowNotVisible = e.offsetTop===-5, this.doesNotIncludeMarginInBodyOffset = a.offsetTop !== i, a.removeChild(b), f.offset.initialize = f.noop
        },
        bodyOffset: function(a) {
            var b = a.offsetTop, c = a.offsetLeft;
            f.offset.initialize(), f.offset.doesNotIncludeMarginInBodyOffset && (b += parseFloat(f.css(a, "marginTop")) || 0, c += parseFloat(f.css(a, "marginLeft")) || 0);
            return {
                top: b,
                left: c
            }
        },
        setOffset: function(a, b, c) {
            var d = f.css(a, "position");
            d === "static" && (a.style.position = "relative");
            var e = f(a), g = e.offset(), h = f.css(a, "top"), i = f.css(a, "left"), j = (d === "absolute" || d === "fixed") && f.inArray("auto", [h, i])>-1, k = {}, l = {}, m, n;
            j ? (l = e.position(), m = l.top, n = l.left) : (m = parseFloat(h) || 0, n = parseFloat(i) || 0), f.isFunction(b) && (b = b.call(a, c, g)), b.top != null && (k.top = b.top - g.top + m), b.left != null && (k.left = b.left - g.left + n), "using" in b ? b.using.call(a, k) : e.css(k)
        }
    }, f.fn.extend({
        position: function() {
            if (!this[0]) {
                return null
            }
            var a = this[0], b = this.offsetParent(), c = this.offset(), d = ct.test(b[0].nodeName) ? {
                top: 0,
                left: 0
            }
            : b.offset();
            c.top -= parseFloat(f.css(a, "marginTop")) || 0, c.left -= parseFloat(f.css(a, "marginLeft")) || 0, d.top += parseFloat(f.css(b[0], "borderTopWidth")) || 0, d.left += parseFloat(f.css(b[0], "borderLeftWidth")) || 0;
            return {
                top: c.top - d.top,
                left: c.left - d.left
            }
        },
        offsetParent: function() {
            return this.map(function() {
                var a = this.offsetParent || c.body;
                while (a&&!ct.test(a.nodeName) && f.css(a, "position") === "static") {
                    a = a.offsetParent
                }
                return a
            })
        }
    }), f.each(["Left", "Top"], function(a, c) {
        var d = "scroll" + c;
        f.fn[d] = function(c) {
            var e, g;
            if (c === b) {
                e = this[0];
                if (!e) {
                    return null
                }
                g = cu(e);
                return g ? "pageXOffset" in g ? g[a ? "pageYOffset": "pageXOffset"] : f.support.boxModel && g.document.documentElement[d] || g.document.body[d] : e[d]
            }
            return this.each(function() {
                g = cu(this), g ? g.scrollTo(a ? f(g).scrollLeft() : c, a ? c : f(g).scrollTop()) : this[d] = c
            })
        }
    }), f.each(["Height", "Width"], function(a, c) {
        var d = c.toLowerCase();
        f.fn["inner" + c] = function() {
            var a = this[0];
            return a && a.style ? parseFloat(f.css(a, d, "padding")) : null
        }, f.fn["outer" + c] = function(a) {
            var b = this[0];
            return b && b.style ? parseFloat(f.css(b, d, a ? "margin" : "border")) : null
        }, f.fn[d] = function(a) {
            var e = this[0];
            if (!e) {
                return a == null ? null : this
            }
            if (f.isFunction(a)) {
                return this.each(function(b) {
                    var c = f(this);
                    c[d](a.call(this, b, c[d]()))
                })
            }
            if (f.isWindow(e)) {
                var g = e.document.documentElement["client" + c], h = e.document.body;
                return e.document.compatMode === "CSS1Compat" && g || h && h["client" + c] || g
            }
            if (e.nodeType === 9) {
                return Math.max(e.documentElement["client" + c], e.body["scroll" + c], e.documentElement["scroll" + c], e.body["offset" + c], e.documentElement["offset" + c])
            }
            if (a === b) {
                var i = f.css(e, d), j = parseFloat(i);
                return f.isNaN(j) ? i : j
            }
            return this.css(d, typeof a == "string" ? a : a + "px")
        }
    }), a.jQuery = a.$ = f
})(window);
(function() {
    function v(b) {
        console.log("$f.fireEvent", [].slice.call(b))
    }
    function r(b) {
        if (!b || typeof b != "object") {
            return b
        }
        var e = new b.constructor();
        for (var c in b) {
            if (b.hasOwnProperty(c)) {
                e[c] = r(b[c])
            }
        }
        return e
    }
    function p(b, f) {
        if (!b) {
            return 
        }
        var j, g = 0, e = b.length;
        if (e === undefined) {
            for (j in b) {
                if (f.call(b[j], j, b[j]) === false) {
                    break
                }
            }
        } else {
            for (var c = b[0]; g < e && f.call(c, g, c) !== false; c = b[++g]) {}
        }
        return b
    }
    function z(b) {
        return document.getElementById(b)
    }
    function t(b, c, e) {
        if (typeof c != "object") {
            return b
        }
        if (b && c) {
            p(c, function(g, f) {
                if (!e || typeof f != "function") {
                    b[g] = f
                }
            })
        }
        return b
    }
    function o(b) {
        var e = b.indexOf(".");
        if (e!=-1) {
            var f = b.substring(0, e) || "*";
            var g = b.substring(e + 1, b.length);
            var c = [];
            p(document.getElementsByTagName(f), function() {
                if (this.className && this.className.indexOf(g)!=-1) {
                    c.push(this)
                }
            });
            return c
        }
    }
    function w(b) {
        b = b || window.event;
        if (b.preventDefault) {
            b.stopPropagation();
            b.preventDefault()
        } else {
            b.returnValue = false;
            b.cancelBubble = true
        }
        return false
    }
    function s(b, e, c) {
        b[e] = b[e] || [];
        b[e].push(c)
    }
    function x() {
        return "_" + ("" + Math.random()).substring(2, 10)
    }
    var u = function(c, f, e) {
        var g = this;
        var j = {};
        var b = {};
        g.index = f;
        if (typeof c == "string") {
            c = {
                url: c
            }
        }
        t(this, c, true);
        p(("Begin*,Start,Pause*,Resume*,Seek*,Stop*,Finish*,LastSecond,Update,BufferFull,BufferEmpty,BufferStop").split(","), function() {
            var n = "on" + this;
            if (n.indexOf("*")!=-1) {
                n = n.substring(0, n.length - 1);
                var m = "onBefore" + n.substring(2);
                g[m] = function(C) {
                    s(b, m, C);
                    return g
                }
            }
            g[n] = function(C) {
                s(b, n, C);
                return g
            };
            if (f==-1) {
                if (g[m]) {
                    e[m] = g[m]
                }
                if (g[n]) {
                    e[n] = g[n]
                }
            }
        });
        t(this, {
            onCuepoint: function(m, n) {
                if (arguments.length == 1) {
                    j.embedded = [null, m];
                    return g
                }
                if (typeof m == "number") {
                    m = [m]
                }
                var C = x();
                j[C] = [m, n];
                if (e.isLoaded()) {
                    e._api().fp_addCuepoints(m, f, C)
                }
                return g
            },
            update: function(n) {
                t(g, n);
                if (e.isLoaded()) {
                    e._api().fp_updateClip(n, f)
                }
                var C = e.getConfig();
                var m = (f==-1) ? C.clip: C.playlist[f];
                t(m, n, true)
            },
            _fireEvent: function(D, F, C, n) {
                if (D == "onLoad") {
                    p(j, function(H, G) {
                        if (G[0]) {
                            e._api().fp_addCuepoints(G[0], f, H)
                        }
                    });
                    return false
                }
                n = n || g;
                if (D == "onCuepoint") {
                    var E = j[F];
                    if (E) {
                        return E[1].call(e, n, C)
                    }
                }
                if (F && "onBeforeBegin,onMetaData,onStart,onUpdate,onResume".indexOf(D)!=-1) {
                    t(n, F);
                    if (F.metaData) {
                        if (!n.duration) {
                            n.duration = F.metaData.duration
                        } else {
                            n.fullDuration = F.metaData.duration
                        }
                    }
                }
                var m = true;
                p(b[D], function() {
                    m = this.call(e, n, F, C)
                });
                return m
            }
        });
        if (c.onCuepoint) {
            var k = c.onCuepoint;
            g.onCuepoint.apply(g, typeof k == "function" ? [k] : k);
            delete c.onCuepoint
        }
        p(c, function(n, m) {
            if (typeof m == "function") {
                s(b, n, m);
                delete c[n]
            }
        });
        if (f==-1) {
            e.onCuepoint = this.onCuepoint
        }
    };
    var q = function(j, f, g, c) {
        var e = {};
        var k = this;
        var b = false;
        if (c) {
            t(e, c)
        }
        p(f, function(n, m) {
            if (typeof m == "function") {
                e[n] = m;
                delete f[n]
            }
        });
        t(this, {
            animate: function(E, D, m) {
                if (!E) {
                    return k
                }
                if (typeof D == "function") {
                    m = D;
                    D = 500
                }
                if (typeof E == "string") {
                    var n = E;
                    E = {};
                    E[n] = D;
                    D = 500
                }
                if (m) {
                    var C = x();
                    e[C] = m
                }
                if (D === undefined) {
                    D = 500
                }
                f = g._api().fp_animate(j, E, D, C);
                return k
            },
            css: function(n, m) {
                if (m !== undefined) {
                    var C = {};
                    C[n] = m;
                    n = C
                }
                f = g._api().fp_css(j, n);
                t(k, f);
                return k
            },
            show: function() {
                this.display = "block";
                g._api().fp_showPlugin(j);
                return k
            },
            hide: function() {
                this.display = "none";
                g._api().fp_hidePlugin(j);
                return k
            },
            toggle: function() {
                this.display = g._api().fp_togglePlugin(j);
                return k
            },
            fadeTo: function(D, m, n) {
                if (typeof m == "function") {
                    n = m;
                    m = 500
                }
                if (n) {
                    var C = x();
                    e[C] = n
                }
                this.display = g._api().fp_fadeTo(j, D, m, C);
                this.opacity = D;
                return k
            },
            fadeIn: function(m, n) {
                return k.fadeTo(1, m, n)
            },
            fadeOut: function(m, n) {
                return k.fadeTo(0, m, n)
            },
            getName: function() {
                return j
            },
            getPlayer: function() {
                return g
            },
            _fireEvent: function(n, C, m) {
                if (n == "onUpdate") {
                    var E = g._api().fp_getPlugin(j);
                    if (!E) {
                        return 
                    }
                    t(k, E);
                    delete k.methods;
                    if (!b) {
                        p(E.methods, function() {
                            var F = "" + this;
                            k[F] = function() {
                                var H = [].slice.call(arguments);
                                var G = g._api().fp_invoke(j, F, H);
                                return G === "undefined" || G === undefined ? k : G
                            }
                        });
                        b = true
                    }
                }
                var D = e[n];
                if (D) {
                    D.apply(k, C);
                    if (n.substring(0, 1) == "_") {
                        delete e[n]
                    }
                }
            }
        })
    };
    function A(m, K, n) {
        var M = this, F = null, G, J, j = [], L = {}, g = {}, c, I, H, b, k, f;
        t(M, {
            id: function() {
                return c
            },
            isLoaded: function() {
                return (F !== null)
            },
            getParent: function() {
                return m
            },
            hide: function(C) {
                if (C) {
                    m.style.height = "0px"
                }
                if (F) {
                    F.style.height = "0px"
                }
                return M
            },
            show: function() {
                m.style.height = f + "px";
                if (F) {
                    F.style.height = k + "px"
                }
                return M
            },
            isHidden: function() {
                return F && parseInt(F.style.height, 10) === 0
            },
            load: function(C) {
                if (!F && M._fireEvent("onBeforeLoad") !== false) {
                    p(B, function() {
                        this.unload()
                    });
                    G = m.innerHTML;
                    if (G&&!flashembed.isSupported(K.version)) {
                        m.innerHTML = ""
                    }
                    flashembed(m, K, {
                        config: n
                    });
                    if (C) {
                        C.cached = true;
                        s(g, "onLoad", C)
                    }
                }
                return M
            },
            unload: function() {
                if (G.replace(/\s/g, "") !== "") {
                    if (M._fireEvent("onBeforeUnload") === false) {
                        return M
                    }
                    try {
                        if (F) {
                            F.fp_close()
                        }
                    } catch (C) {}
                    F = null;
                    m.innerHTML = G;
                    M._fireEvent("onUnload")
                }
                return M
            },
            getClip: function(C) {
                if (C === undefined) {
                    C = b
                }
                return j[C]
            },
            getCommonClip: function() {
                return J
            },
            getPlaylist: function() {
                return j
            },
            getPlugin: function(E) {
                var C = L[E];
                if (!C && M.isLoaded()) {
                    var D = M._api().fp_getPlugin(E);
                    if (D) {
                        C = new q(E, D, M);
                        L[E] = C
                    }
                }
                return C
            },
            getScreen: function() {
                return M.getPlugin("screen")
            },
            getControls: function() {
                return M.getPlugin("controls")
            },
            getConfig: function(C) {
                return C ? r(n) : n
            },
            getFlashParams: function() {
                return K
            },
            loadPlugin: function(N, O, D, E) {
                if (typeof D == "function") {
                    E = D;
                    D = {}
                }
                var P = E ? x(): "_";
                M._api().fp_loadPlugin(N, O, D, P);
                var Q = {};
                Q[P] = E;
                var C = new q(N, null, M, Q);
                L[N] = C;
                return C
            },
            getState: function() {
                return F ? F.fp_getState() : - 1
            },
            play: function(D, E) {
                function C() {
                    if (D !== undefined) {
                        M._api().fp_play(D, E)
                    } else {
                        M._api().fp_play()
                    }
                }
                if (F) {
                    C()
                } else {
                    M.load(function() {
                        C()
                    })
                }
                return M
            },
            getVersion: function() {
                var C = "flowplayer.js 3.1.4";
                if (F) {
                    var D = F.fp_getVersion();
                    D.push(C);
                    return D
                }
                return C
            },
            _api: function() {
                if (!F) {
                    throw "Flowplayer " + M.id() + " not loaded when calling an API method"
                }
                return F
            },
            setClip: function(C) {
                M.setPlaylist([C]);
                return M
            },
            getIndex: function() {
                return H
            }
        });
        p(("Click*,Load*,Unload*,Keypress*,Volume*,Mute*,Unmute*,PlaylistReplace,ClipAdd,Fullscreen*,FullscreenExit,Error,MouseOver,MouseOut").split(","), function() {
            var D = "on" + this;
            if (D.indexOf("*")!=-1) {
                D = D.substring(0, D.length - 1);
                var C = "onBefore" + D.substring(2);
                M[C] = function(E) {
                    s(g, C, E);
                    return M
                }
            }
            M[D] = function(E) {
                s(g, D, E);
                return M
            }
        });
        p(("pause,resume,mute,unmute,stop,toggle,seek,getStatus,getVolume,setVolume,getTime,isPaused,isPlaying,startBuffering,stopBuffering,isFullscreen,toggleFullscreen,reset,close,setPlaylist,addClip,playFeed").split(","), function() {
            var C = this;
            M[C] = function(E, N) {
                if (!F) {
                    return M
                }
                var D = null;
                if (E !== undefined && N !== undefined) {
                    D = F["fp_" + C](E, N)
                } else {
                    D = (E === undefined) ? F["fp_" + C]() : F["fp_" + C](E)
                }
                return D === "undefined" || D === undefined ? M : D
            }
        });
        M._fireEvent = function(D) {
            if (typeof D == "string") {
                D = [D]
            }
            var C = D[0], Q = D[1], S = D[2], T = D[3], U = 0;
            if (n.debug) {
                v(D)
            }
            if (!F && C == "onLoad" && Q == "player") {
                F = F || z(I);
                k = F.clientHeight;
                p(j, function() {
                    this._fireEvent("onLoad")
                });
                p(L, function(O, N) {
                    N._fireEvent("onUpdate")
                });
                J._fireEvent("onLoad")
            }
            if (C == "onLoad" && Q != "player") {
                return 
            }
            if (C == "onError") {
                if (typeof Q == "string" || (typeof Q == "number" && typeof S == "number")) {
                    Q = S;
                    S = T
                }
            }
            if (C == "onContextMenu") {
                p(n.contextMenu[Q], function(O, N) {
                    N.call(M)
                });
                return 
            }
            if (C == "onPluginEvent") {
                var X = Q.name || Q;
                var W = L[X];
                if (W) {
                    W._fireEvent("onUpdate", Q);
                    W._fireEvent(S, D.slice(3))
                }
                return 
            }
            if (C == "onPlaylistReplace") {
                j = [];
                var R = 0;
                p(Q, function() {
                    j.push(new u(this, R++, M))
                })
            }
            if (C == "onClipAdd") {
                if (Q.isInStream) {
                    return 
                }
                Q = new u(Q, S, M);
                j.splice(S, 0, Q);
                for (U = S + 1; U < j.length; U++) {
                    j[U].index++
                }
            }
            var E = true;
            if (typeof Q == "number" && Q < j.length) {
                b = Q;
                var V = j[Q];
                if (V) {
                    E = V._fireEvent(C, S, T)
                }
                if (!V || E !== false) {
                    E = J._fireEvent(C, S, T, V)
                }
            }
            p(g[C], function() {
                E = this.call(M, Q, S);
                if (this.cached) {
                    g[C].splice(U, 1)
                }
                if (E === false) {
                    return false
                }
                U++
            });
            return E
        };
        function e() {
            if ($f(m)) {
                $f(m).getParent().innerHTML = "";
                H = $f(m).getIndex();
                B[H] = M
            } else {
                B.push(M);
                H = B.length - 1
            }
            f = parseInt(m.style.height, 10) || m.clientHeight;
            if (typeof K == "string") {
                K = {
                    src: K
                }
            }
            c = m.id || "fp" + x();
            I = K.id || c + "_api";
            K.id = I;
            K.cachebusting = true;
            n.playerId = c;
            if (typeof n == "string") {
                n = {
                    clip: {
                        url: n
                    }
                }
            }
            if (typeof n.clip == "string") {
                n.clip = {
                    url: n.clip
                }
            }
            n.clip = n.clip || {};
            if (m.getAttribute("href", 2)&&!n.clip.url) {
                n.clip.url = m.getAttribute("href", 2)
            }
            J = new u(n.clip, - 1, M);
            n.playlist = n.playlist || [n.clip];
            var D = 0;
            p(n.playlist, function() {
                var E = this;
                if (typeof E == "object" && E.length) {
                    E = {
                        url: "" + E
                    }
                }
                p(n.clip, function(O, N) {
                    if (N !== undefined && E[O] === undefined && typeof N != "function") {
                        E[O] = N
                    }
                });
                n.playlist[D] = E;
                E = new u(E, D, M);
                j.push(E);
                D++
            });
            p(n, function(N, E) {
                if (typeof E == "function") {
                    if (J[N]) {
                        J[N](E)
                    } else {
                        s(g, N, E)
                    }
                    delete n[N]
                }
            });
            p(n.plugins, function(N, E) {
                if (E) {
                    L[N] = new q(N, E, M)
                }
            });
            if (!n.plugins || n.plugins.controls === undefined) {
                L.controls = new q("controls", null, M)
            }
            L.canvas = new q("canvas", null, M);
            K.bgcolor = K.bgcolor || "#000000";
            K.version = K.version || [9, 0];
            K.expressInstall = "http://www.flowplayer.org/swf/expressinstall.swf";
            function C(E) {
                if (!M.isLoaded() && M._fireEvent("onBeforeClick") !== false) {
                    M.load()
                }
                return w(E)
            }
            G = m.innerHTML;
            if (G.replace(/\s/g, "") !== "") {
                if (m.addEventListener) {
                    m.addEventListener("click", C, false)
                } else {
                    if (m.attachEvent) {
                        m.attachEvent("onclick", C)
                    }
                }
            } else {
                if (m.addEventListener) {
                    m.addEventListener("click", w, false)
                }
                M.load()
            }
        }
        if (typeof m == "string") {
            flashembed.domReady(function() {
                var C = z(m);
                if (!C) {
                    throw "Flowplayer cannot access element: " + m
                } else {
                    m = C;
                    e()
                }
            })
        } else {
            e()
        }
    }
    var B = [];
    function y(b) {
        this.length = b.length;
        this.each = function(c) {
            p(b, c)
        };
        this.size = function() {
            return b.length
        }
    }
    window.flowplayer = window.$f = function() {
        var g = null;
        var j = arguments[0];
        if (!arguments.length) {
            p(B, function() {
                if (this.isLoaded()) {
                    g = this;
                    return false
                }
            });
            return g || B[0]
        }
        if (arguments.length == 1) {
            if (typeof j == "number") {
                return B[j]
            } else {
                if (j == "*") {
                    return new y(B)
                }
                p(B, function() {
                    if (this.id() == j.id || this.id() == j || this.getParent() == j) {
                        g = this;
                        return false
                    }
                });
                return g
            }
        }
        if (arguments.length > 1) {
            var e = arguments[1];
            var f = (arguments.length == 3) ? arguments[2]: {};
            if (typeof j == "string") {
                if (j.indexOf(".")!=-1) {
                    var b = [];
                    p(o(j), function() {
                        b.push(new A(this, r(e), r(f)))
                    });
                    return new y(b)
                } else {
                    var c = z(j);
                    return new A(c !== null ? c : j, e, f)
                }
            } else {
                if (j) {
                    return new A(j, e, f)
                }
            }
        }
        return null
    };
    t(window.$f, {
        fireEvent: function() {
            var c = [].slice.call(arguments);
            var b = $f(c[0]);
            return b ? b._fireEvent(c.slice(1)) : null
        },
        addPlugin: function(c, b) {
            A.prototype[c] = b;
            return $f
        },
        each: p,
        extend: t
    });
    if (typeof jQuery == "function") {
        jQuery.prototype.flowplayer = function(b, c) {
            if (!arguments.length || typeof arguments[0] == "number") {
                var e = [];
                this.each(function() {
                    var f = $f(this);
                    if (f) {
                        e.push(f)
                    }
                });
                return arguments.length ? e[arguments[0]] : new y(e)
            }
            return this.each(function() {
                $f(this, r(b), c ? r(c) : {})
            })
        }
    }
})();
(function() {
    var q = typeof jQuery == "function";
    var m = {
        width: "100%",
        height: "100%",
        allowfullscreen: true,
        allowscriptaccess: "always",
        quality: "high",
        version: null,
        onFail: null,
        expressInstall: null,
        w3c: false,
        cachebusting: false
    };
    if (q) {
        jQuery.tools = jQuery.tools || {};
        jQuery.tools.flashembed = {
            version: "1.0.4",
            conf: m
        }
    }
    function k() {
        if (s.done) {
            return false
        }
        var b = document;
        if (b && b.getElementsByTagName && b.getElementById && b.body) {
            clearInterval(s.timer);
            s.timer = null;
            for (var c = 0; c < s.ready.length; c++) {
                s.ready[c].call()
            }
            s.ready = null;
            s.done = true
        }
    }
    var s = q ? jQuery: function(b) {
        if (s.done) {
            return b()
        }
        if (s.timer) {
            s.ready.push(b)
        } else {
            s.ready = [b];
            s.timer = setInterval(k, 13)
        }
    };
    function p(b, c) {
        if (c) {
            for (key in c) {
                if (c.hasOwnProperty(key)) {
                    b[key] = c[key]
                }
            }
        }
        return b
    }
    function o(e) {
        switch (n(e)) {
        case"string":
            e = e.replace(new RegExp('(["\\\\])', "g"), "\\$1");
            e = e.replace(/^\s?(\d+)%/, "$1pct");
            return '"' + e + '"';
        case"array":
            return "[" + t(e, function(f) {
                return o(f)
            }).join(",") + "]";
        case"function":
            return '"function()"';
        case"object":
            var c = [];
            for (var b in e) {
                if (e.hasOwnProperty(b)) {
                    c.push('"' + b + '":' + o(e[b]))
                }
            }
            return "{" + c.join(",") + "}"
        }
        return String(e).replace(/\s/g, " ").replace(/\'/g, '"')
    }
    function n(b) {
        if (b === null || b === undefined) {
            return false
        }
        var c = typeof b;
        return (c == "object" && b.push) ? "array" : c
    }
    if (window.attachEvent) {
        window.attachEvent("onbeforeunload", function() {
            __flash_unloadHandler = function() {};
            __flash_savedUnloadHandler = function() {}
        })
    }
    function t(e, f) {
        var b = [];
        for (var c in e) {
            if (e.hasOwnProperty(c)) {
                b[c] = f(e[c])
            }
        }
        return b
    }
    function u(g, e) {
        var j = p({}, g);
        var f = document.all;
        var w = '<object width="' + j.width + '" height="' + j.height + '"';
        if (f&&!j.id) {
            j.id = "_" + ("" + Math.random()).substring(9)
        }
        if (j.id) {
            w += ' id="' + j.id + '"'
        }
        if (j.cachebusting) {
            j.src += ((j.src.indexOf("?")!=-1 ? "&" : "?") + Math.random())
        }
        if (j.w3c ||!f) {
            w += ' data="' + j.src + '" type="application/x-shockwave-flash"'
        } else {
            w += ' classid="clsid:D27CDB6E-AE6D-11cf-96B8-444553540000"'
        }
        w += ">";
        if (j.w3c || f) {
            w += '<param name="movie" value="' + j.src + '" />'
        }
        j.width = j.height = j.id = j.w3c = j.src = null;
        for (var c in j) {
            if (j[c] !== null) {
                w += '<param name="' + c + '" value="' + j[c] + '" />'
            }
        }
        var v = "";
        if (e) {
            for (var b in e) {
                if (e[b] !== null) {
                    v += b + "=" + (typeof e[b] == "object" ? o(e[b]) : e[b]) + "&"
                }
            }
            v = v.substring(0, v.length - 1);
            w += '<param name="flashvars" value=\'' + v + "' />"
        }
        w += "</object>";
        return w
    }
    function r(b, j, c) {
        var e = flashembed.getVersion();
        p(this, {
            getContainer: function() {
                return b
            },
            getConf: function() {
                return j
            },
            getVersion: function() {
                return e
            },
            getFlashvars: function() {
                return c
            },
            getApi: function() {
                return b.firstChild
            },
            getHTML: function() {
                return u(j, c)
            }
        });
        var g = j.version;
        var f = j.expressInstall;
        var v=!g || flashembed.isSupported(g);
        if (v) {
            j.onFail = j.version = j.expressInstall = null;
            b.innerHTML = u(j, c)
        } else {
            if (g && f && flashembed.isSupported([6, 65])) {
                p(j, {
                    src: f
                });
                c = {
                    MMredirectURL: location.href,
                    MMplayerType: "PlugIn",
                    MMdoctitle: document.title
                };
                b.innerHTML = u(j, c)
            } else {
                if (b.innerHTML.replace(/\s/g, "") !== "") {} else {
                    b.innerHTML = "<h2>Flash version " + g + " or greater is required</h2><h3>" + (e[0] > 0 ? "Your version is " + e : "You have no flash plugin installed") + "</h3>" + (b.tagName == "A" ? "<p>Click here to download latest version</p>" : "<p>Download latest version from <a href='http://www.adobe.com/go/getflashplayer'>here</a></p>");
                    if (b.tagName == "A") {
                        b.onclick = function() {
                            location.href = "http://www.adobe.com/go/getflashplayer"
                        }
                    }
                }
            }
        }
        if (!v && j.onFail) {
            var w = j.onFail.call(this);
            if (typeof w == "string") {
                b.innerHTML = w
            }
        }
        if (document.all) {
            window[j.id] = document.getElementById(j.id)
        }
    }
    window.flashembed = function(c, b, e) {
        if (typeof c == "string") {
            var g = document.getElementById(c);
            if (g) {
                c = g
            } else {
                s(function() {
                    flashembed(c, b, e)
                });
                return 
            }
        }
        if (!c) {
            return 
        }
        if (typeof b == "string") {
            b = {
                src: b
            }
        }
        var f = p({}, m);
        p(f, b);
        return new r(c, f, e)
    };
    p(window.flashembed, {
        getVersion: function() {
            var b = [0, 0];
            if (navigator.plugins && typeof navigator.plugins["Shockwave Flash"] == "object") {
                var c = navigator.plugins["Shockwave Flash"].description;
                if (typeof c != "undefined") {
                    c = c.replace(/^.*\s+(\S+\s+\S+$)/, "$1");
                    var w = parseInt(c.replace(/^(.*)\..*$/, "$1"), 10);
                    var f = /r/.test(c) ? parseInt(c.replace(/^.*r(.*)$/, "$1"), 10): 0;
                    b = [w, f]
                }
            } else {
                if (window.ActiveXObject) {
                    try {
                        var j = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.7")
                    } catch (g) {
                        try {
                            j = new ActiveXObject("ShockwaveFlash.ShockwaveFlash.6");
                            b = [6, 0];
                            j.AllowScriptAccess = "always"
                        } catch (e) {
                            if (b[0] == 6) {
                                return b
                            }
                        }
                        try {
                            j = new ActiveXObject("ShockwaveFlash.ShockwaveFlash")
                        } catch (v) {}
                    }
                    if (typeof j == "object") {
                        c = j.GetVariable("$version");
                        if (typeof c != "undefined") {
                            c = c.replace(/^\S+\s+(.*)$/, "$1").split(",");
                            b = [parseInt(c[0], 10), parseInt(c[2], 10)]
                        }
                    }
                }
            }
            return b
        },
        isSupported: function(e) {
            var b = flashembed.getVersion();
            var c = (b[0] > e[0]) || (b[0] == e[0] && b[1] >= e[1]);
            return c
        },
        domReady: s,
        asString: o,
        getHTML: u
    });
    if (q) {
        jQuery.fn.flashembed = function(c, e) {
            var b = null;
            this.each(function() {
                b = flashembed(this, c, e)
            });
            return c.api === false ? this : b
        }
    }
})();
(function(e) {
    var b = e.scrollTo = function(f, g, j) {
        e(window).scrollTo(f, g, j)
    };
    b.defaults = {
        axis: "xy",
        duration: parseFloat(e.fn.jquery) >= 1.3 ? 0: 1
    };
    b.window = function(f) {
        return e(window)._scrollable()
    };
    e.fn._scrollable = function() {
        return this.map(function() {
            var f = this, g=!f.nodeName || e.inArray(f.nodeName.toLowerCase(), ["iframe", "#document", "html", "body"])!=-1;
            if (!g) {
                return f
            }
            var j = (f.contentWindow || f).document || f.ownerDocument || f;
            return e.browser.safari || j.compatMode == "BackCompat" ? j.body : j.documentElement
        })
    };
    e.fn.scrollTo = function(k, g, f) {
        if (typeof g == "object") {
            f = g;
            g = 0
        }
        if (typeof f == "function") {
            f = {
                onAfter: f
            }
        }
        if (k == "max") {
            k = 9000000000
        }
        f = e.extend({}, b.defaults, f);
        g = g || f.speed || f.duration;
        f.queue = f.queue && f.axis.length > 1;
        if (f.queue) {
            g/=2
        }
        f.offset = c(f.offset);
        f.over = c(f.over);
        return this._scrollable().each(function() {
            var w = this, p = e(w), v = k, n, o = {}, j = p.is("html,body");
            switch (typeof v) {
            case"number":
            case"string":
                if (/^([+-]=)?\d+(\.\d+)?(px|%)?$/.test(v)) {
                    v = c(v);
                    break
                }
                v = e(v, this);
            case"object":
                if (v.is || v.style) {
                    n = (v = e(v)).offset()
                }
            }
            e.each(f.axis.split(""), function(s, t) {
                var x = t == "x" ? "Left": "Top", u = x.toLowerCase(), z = "scroll" + x, r = w[z], q = b.max(w, t);
                if (n) {
                    o[z] = n[u] + (j ? 0 : r - p.offset()[u]);
                    if (f.margin) {
                        o[z] -= parseInt(v.css("margin" + x)) || 0;
                        o[z] -= parseInt(v.css("border" + x + "Width")) || 0
                    }
                    o[z] += f.offset[u] || 0;
                    if (f.over[u]) {
                        o[z] += v[t == "x" ? "width": "height"]() * f.over[u]
                    }
                } else {
                    var y = v[u];
                    o[z] = y.slice && y.slice( - 1) == "%" ? parseFloat(y) / 100 * q : y
                }
                if (/^\d+$/.test(o[z])) {
                    o[z] = o[z] <= 0 ? 0 : Math.min(o[z], q)
                }
                if (!s && f.queue) {
                    if (r != o[z]) {
                        m(f.onAfterFirst)
                    }
                    delete o[z]
                }
            });
            m(f.onAfter);
            function m(q) {
                p.animate(o, g, f.easing, q && function() {
                    q.call(this, k, f)
                })
            }
        }).end()
    };
    b.max = function(j, k) {
        var o = k == "x" ? "Width": "Height", n = "scroll" + o;
        if (!e(j).is("html,body")) {
            return j[n] - e(j)[o.toLowerCase()]()
        }
        var p = "client" + o, g = j.ownerDocument.documentElement, f = j.ownerDocument.body;
        return Math.max(g[n], f[n]) - Math.min(g[p], f[p])
    };
    function c(f) {
        return typeof f == "object" ? f : {
            top: f,
            left: f
        }
    }
})(jQuery);
(function(b) {
    b.fn.lightBox = function(v) {
        v = jQuery.extend({
            overlayBgColor: "#000",
            overlayOpacity: 0.8,
            fixedNavigation: false,
            imageLoading: "/static/images/global/lightbox/lightbox-ico-loading.gif",
            imageBtnPrev: "/static/images/global/lightbox/lightbox-btn-prev.gif",
            imageBtnNext: "/static/images/global/lightbox/lightbox-btn-next.gif",
            imageBtnClose: "/static/images/global/lightbox/lightbox-btn-close.gif",
            imageBlank: "/static/images/global/lightbox/lightbox-blank.gif",
            containerBorderSize: 10,
            containerResizeSpeed: 400,
            txtImage: "Image",
            txtOf: "of",
            keyToClose: "c",
            keyToPrev: "p",
            keyToNext: "n",
            imageArray: [],
            activeImage: 0,
            showDescription: 2,
            parent_div_id: null
        }, v);
        var n = this;
        var t = b("#" + v.parent_div_id).parent();
        function x() {
            u(this, n);
            return false
        }
        function u(B, A) {
            b("embed, object, select").css({
                visibility: "hidden"
            });
            e();
            v.imageArray.length = 0;
            v.activeImage = 0;
            if (A.length == 1) {
                v.imageArray.push(new Array(B.getAttribute("href"), B.getAttribute("title")))
            } else {
                for (var z = 0; z < A.length; z++) {
                    v.imageArray.push(new Array(A[z].getAttribute("href"), A[z].getAttribute("title")))
                }
            }
            while (v.imageArray[v.activeImage][0] != B.getAttribute("href")) {
                v.activeImage++
            }
            q()
        }
        function e() {
            b("body").append('<div id="jquery-overlay" class="jquery-overlay"></div><div id="jquery-lightbox"><div id="lightbox-container-image-box"><div id="lightbox-container-image"><img id="lightbox-image"><div style="" id="lightbox-nav"><a href="#" id="lightbox-nav-btnPrev"></a><a href="#" id="lightbox-nav-btnNext"></a></div><div id="lightbox-loading"><a href="#" id="lightbox-loading-link"><img src="' + v.imageLoading + '"></a></div></div></div><div id="lightbox-container-image-data-box"><div id="lightbox-container-image-data"><div id="lightbox-image-details"><span id="lightbox-image-details-caption"></span></div><div id="lightbox-secNav"><span id="lightbox-image-details-currentNumber"></span><a href="#" id="lightbox-secNav-btnClose"><img src="' + v.imageBtnClose + '"></a></div></div></div></div>');
            var z = j();
            b("#jquery-overlay").css({
                backgroundColor: v.overlayBgColor,
                opacity: v.overlayOpacity,
                width: z[0],
                height: z[1] + 5000
            }).fadeIn();
            var A = m();
            b("#jquery-lightbox").css({
                top: A[1] + (z[3] / 10),
                left: A[0]
            }).show();
            b("#jquery-overlay,#jquery-lightbox").click(function() {
                c()
            });
            b("#lightbox-loading-link,#lightbox-secNav-btnClose").click(function() {
                c();
                return false
            });
            b(window).resize(function() {
                var B = j();
                b("#jquery-overlay").css({
                    width: B[0],
                    height: B[1] + 5000
                });
                var C = m();
                b("#jquery-lightbox").css({
                    top: C[1] + (B[3] / 10),
                    left: C[0]
                })
            })
        }
        function q() {
            b("#lightbox-loading").show();
            if (v.fixedNavigation) {
                b("#lightbox-image,#lightbox-container-image-data-box,#lightbox-image-details-currentNumber").hide()
            } else {
                b("#lightbox-image,#lightbox-nav,#lightbox-nav-btnPrev,#lightbox-nav-btnNext,#lightbox-container-image-data-box,#lightbox-image-details-currentNumber").hide()
            }
            var z = new Image();
            z.onload = function() {
                b("#lightbox-image").attr("src", v.imageArray[v.activeImage][0]);
                o(z.width, z.height);
                z.onload = function() {}
            };
            z.src = v.imageArray[v.activeImage][0]
        }
        function o(C, F) {
            var z = b("#lightbox-container-image-box").width();
            var E = b("#lightbox-container-image-box").height();
            var D = (C + (v.containerBorderSize * 2));
            var B = (F + (v.containerBorderSize * 2));
            var A = z - D;
            var G = E - B;
            b("#lightbox-container-image-box").animate({
                width: D,
                height: B
            }, v.containerResizeSpeed, function() {
                k()
            });
            if ((A == 0) && (G == 0)) {
                if (b.browser.msie) {
                    s(250)
                } else {
                    s(100)
                }
            }
            b("#lightbox-container-image-data-box").css({
                width: C
            });
            b("#lightbox-nav-btnPrev,#lightbox-nav-btnNext").css({
                height: F + (v.containerBorderSize * 2)
            })
        }
        function k() {
            b("#lightbox-loading").hide();
            b("#lightbox-image").fadeIn(function() {
                p();
                y();
                b("#lightbox-image-details-currentNumber").fadeIn()
            });
            w()
        }
        function p() {
            b("#lightbox-container-image-data-box").slideDown("fast");
            b("#lightbox-image-details-caption").hide();
            if (v.imageArray[v.activeImage][1]) {
                if (v.showDescription == 1) {
                    var H = t.css("font-family");
                    var D = t.css("color");
                    var I = t.css("font-size");
                    var J = t.css("font-weight");
                    var z = t.css("font-style");
                    var A = t.css("font-variant");
                    var B = t.css("text-align");
                    var F = t.css("text-decoration");
                    var G = t.css("text-transform");
                    var E = t.css("white-space");
                    var C = b("#lightbox-image-details-caption");
                    if (H) {
                        C.css("font-family", H)
                    }
                    if (D) {
                        C.css("color", D)
                    }
                    if (I) {
                        C.css("font-size", I)
                    }
                    if (J) {
                        C.css("font-weight", J)
                    }
                    if (z) {
                        C.css("font-style", z)
                    }
                    if (A) {
                        C.css("font-variant", A)
                    }
                    if (B) {
                        C.css("text-align", B)
                    }
                    if (F) {
                        C.css("text-decoration", F)
                    }
                    if (G) {
                        C.css("text-transform", G)
                    }
                    if (E) {
                        C.css("white-space", E)
                    }
                    C.html(v.imageArray[v.activeImage][1]).show()
                } else {
                    b("#lightbox-image-details-caption").html("&nbsp;").show()
                }
            }
            if (v.imageArray.length > 1) {
                b("#lightbox-image-details-currentNumber").html(v.txtImage + " " + (v.activeImage + 1) + " " + v.txtOf + " " + v.imageArray.length)
            }
        }
        function y() {
            b("#lightbox-nav").show();
            b("#lightbox-nav-btnPrev,#lightbox-nav-btnNext").css({
                background: "transparent url(" + v.imageBlank + ") no-repeat"
            });
            if (v.activeImage != 0) {
                if (v.fixedNavigation) {
                    b("#lightbox-nav-btnPrev").css({
                        background: "url(" + v.imageBtnPrev + ") left 15% no-repeat"
                    }).unbind().bind("click", function() {
                        v.activeImage = v.activeImage - 1;
                        q();
                        return false
                    })
                } else {
                    b("#lightbox-nav-btnPrev").unbind().hover(function() {
                        b(this).css({
                            background: "url(" + v.imageBtnPrev + ") left 15% no-repeat"
                        })
                    }, function() {
                        b(this).css({
                            background: "transparent url(" + v.imageBlank + ") no-repeat"
                        })
                    }).show().bind("click", function() {
                        v.activeImage = v.activeImage - 1;
                        q();
                        return false
                    })
                }
            }
            if (v.activeImage != (v.imageArray.length - 1)) {
                if (v.fixedNavigation) {
                    b("#lightbox-nav-btnNext").css({
                        background: "url(" + v.imageBtnNext + ") right 15% no-repeat"
                    }).unbind().bind("click", function() {
                        v.activeImage = v.activeImage + 1;
                        q();
                        return false
                    })
                } else {
                    b("#lightbox-nav-btnNext").unbind().hover(function() {
                        b(this).css({
                            background: "url(" + v.imageBtnNext + ") right 15% no-repeat"
                        })
                    }, function() {
                        b(this).css({
                            background: "transparent url(" + v.imageBlank + ") no-repeat"
                        })
                    }).show().bind("click", function() {
                        v.activeImage = v.activeImage + 1;
                        q();
                        return false
                    })
                }
            }
            r()
        }
        function r() {
            b(document).keydown(function(z) {
                f(z)
            })
        }
        function g() {
            b(document).unbind()
        }
        function f(z) {
            if (z == null) {
                keycode = event.keyCode;
                escapeKey = 27
            } else {
                keycode = z.keyCode;
                escapeKey = z.DOM_VK_ESCAPE
            }
            key = String.fromCharCode(keycode).toLowerCase();
            if ((key == v.keyToClose) || (key == "x") || (keycode == escapeKey)) {
                c()
            }
            if ((key == v.keyToPrev) || (keycode == 37)) {
                if (v.activeImage != 0) {
                    v.activeImage = v.activeImage - 1;
                    q();
                    g()
                }
            }
            if ((key == v.keyToNext) || (keycode == 39)) {
                if (v.activeImage != (v.imageArray.length - 1)) {
                    v.activeImage = v.activeImage + 1;
                    q();
                    g()
                }
            }
        }
        function w() {
            if ((v.imageArray.length - 1) > v.activeImage) {
                objNext = new Image();
                objNext.src = v.imageArray[v.activeImage + 1][0]
            }
            if (v.activeImage > 0) {
                objPrev = new Image();
                objPrev.src = v.imageArray[v.activeImage - 1][0]
            }
        }
        function c() {
            b("#jquery-lightbox").remove();
            b("#jquery-overlay").fadeOut(function() {
                b("#jquery-overlay").remove()
            });
            b("embed, object, select").css({
                visibility: "visible"
            })
        }
        function j() {
            var B, z;
            if (window.innerHeight && window.scrollMaxY) {
                B = window.innerWidth + window.scrollMaxX;
                z = window.innerHeight + window.scrollMaxY
            } else {
                if (document.body.scrollHeight > document.body.offsetHeight) {
                    B = document.body.scrollWidth;
                    z = document.body.scrollHeight
                } else {
                    B = document.body.offsetWidth;
                    z = document.body.offsetHeight
                }
            }
            var A, C;
            if (self.innerHeight) {
                if (document.documentElement.clientWidth) {
                    A = document.documentElement.clientWidth
                } else {
                    A = self.innerWidth
                }
                C = self.innerHeight
            } else {
                if (document.documentElement && document.documentElement.clientHeight) {
                    A = document.documentElement.clientWidth;
                    C = document.documentElement.clientHeight
                } else {
                    if (document.body) {
                        A = document.body.clientWidth;
                        C = document.body.clientHeight
                    }
                }
            }
            if (z < C) {
                pageHeight = C
            } else {
                pageHeight = z
            }
            if (B < A) {
                pageWidth = B
            } else {
                pageWidth = A
            }
            arrayPageSize = new Array(pageWidth, pageHeight, A, C);
            return arrayPageSize
        }
        function m() {
            var A, z;
            if (self.pageYOffset) {
                z = self.pageYOffset;
                A = self.pageXOffset
            } else {
                if (document.documentElement && document.documentElement.scrollTop) {
                    z = document.documentElement.scrollTop;
                    A = document.documentElement.scrollLeft
                } else {
                    if (document.body) {
                        z = document.body.scrollTop;
                        A = document.body.scrollLeft
                    }
                }
            }
            arrayPageScroll = new Array(A, z);
            return arrayPageScroll
        }
        function s(B) {
            var A = new Date();
            z = null;
            do {
                var z = new Date()
            }
            while (z - A < B)
            }
        return this.unbind("click").click(x)
    }
})(jQuery);
(function(b) {
    if (!(b.iv)) {
        b.extend({
            iv: {}
        })
    }
    b.fn.iv_photogallery = function(c) {
        return this.each(function() {
            new b.iv.photogallery(this, c)
        })
    };
    b.iv.photogallery = function(ak, J) {
        J = b.extend({
            photogallery_id: null,
            display_options: null,
            image_list: null,
            gallery_dropdowns: null,
            left_button_name: null,
            right_button_name: null,
            pause_button_name: null,
            order_num_container_name: null,
            images_container_name: null,
            ul_name: null,
            description_container_name: null,
            p_name: null,
            loading_container_name: null,
            current_page_num_name: null,
            max_page_num_name: null,
            show_description: null,
            autoplay: null,
            type: null,
            txtImage: null,
            txtOf: null
        }, J);
        b.data(ak, "photogallery_" + J.photogallery_id, this);
        var ah = b(ak);
        var E = ah.parent().parent();
        var ao = b("div." + J.images_container_name, ah);
        var ad = b("ul." + J.ul_name, ah);
        var P = b("div." + J.description_container_name, ah);
        var X = b("p." + J.p_name, ah);
        var k = b("div." + J.loading_container_name, ah);
        var f = b("#" + J.order_num_container_name, ah);
        var W = b("div.pagination_container", ah);
        var A = b("span.pagination_current", ah);
        var r = b("span.pagination_total", ah);
        var ag = b("#" + J.current_page_num_name, ah);
        var aj = b("#" + J.max_page_num_name, ah);
        var Q = b("div.pagination_controls", ah);
        var S = b("span.images_showing_start", ah);
        var aa = b("span.images_showing_end", ah);
        var C = b("div.images_container", ah);
        var U = b("span." + J.left_button_name, ah);
        var ai = b("span." + J.right_button_name, ah);
        var Y = b("span." + J.pause_button_name, ah);
        var af = J.photogallery_id;
        var G = JSON.parse(J.display_options);
        var T = JSON.parse(J.image_list);
        var aq = JSON.parse(J.gallery_dropdowns);
        var at = J.show_description;
        var al = J.autoplay;
        var au = J.type;
        var ac;
        if (T.length > 0) {
            var ae = T[0];
            if (au == "preview") {
                ac = ae.preview_url
            } else {
                if (au == "final_design") {
                    ac = ae.final_design_url
                } else {
                    ac = ae.publish_url
                }
            }
        }
        var p = T.length || 0;
        var q = f.val();
        var w = ag.val();
        var an = aj.val();
        var R = 0;
        var Z = null;
        var ap = 1;
        var M = parseInt(S.html());
        var t = parseInt(aa.html());
        var D = G.gallery_type_selected;
        var V = aq.gallery_types[D].toLowerCase();
        var o = new RegExp(/\s+/g);
        var u = V.replace(o, "_");
        var H = aq.gallery_types[D];
        var s = G[H]["large_image_style"];
        switch (u) {
        case ("single_image"):
            y();
            break;
        case ("grid"):
            v();
            break;
        case ("filmstrip_1"):
            B(u);
            break;
        case ("filmstrip_2"):
            B(u);
            break;
        case ("slideshow"):
            c();
            if (al == "1") {
                var n = G[H]["delay"] || "5";
                var am = n * 1000;
                if (am < 850) {
                    am = "850"
                }
                setTimeout(function() {
                    N()
                }, am)
            }
            break
        }
        function y() {
            F();
            k.show();
            if (T.length > 0) {
                U.unbind("click").click(function() {
                    z("left", "single_image");
                    return false
                });
                ai.unbind("click").click(function() {
                    z("right", "single_image");
                    return false
                });
                if (q > 0 && p > 1) {
                    ai.show()
                }
                if ((q < (p - 1)) && (!(q == 0))) {
                    U.show()
                }
            }
            k.hide()
        }
        function v() {
            F();
            if (s == 1) {
                b("img", ao).each(function(aw) {
                    var ax = T[aw];
                    b(this).click(function() {
                        ar(ax);
                        return false
                    })
                })
            } else {
                b(function() {
                    var aw = {
                        txtImage: J.txtImage,
                        txtOf: J.txtOf,
                        showDescription: at,
                        parent_div_id: "photogallery_" + J.photogallery_id
                    };
                    b('img[rel*="lightbox"]', ah).lightBox(aw)
                })
            }
            var av = b("li.page_selected", ad);
            if (av.next().length) {
                ai.unbind("click").css("color", "blue").click(function() {
                    e("right");
                    return false
                })
            } else {
                ai.css("color", "black");
                ai.unbind("click")
            }
            if (av.prev().length) {
                U.unbind("click").css("color", "blue").click(function() {
                    e("left");
                    return false
                })
            } else {
                U.css("color", "black");
                U.unbind("click")
            }
            k.hide()
        }
        function c() {
            F();
            k.show();
            if (T.length > 0) {
                Y.unbind("click").click(function() {
                    g();
                    return false
                });
                ai.unbind("click").click(function() {
                    N();
                    return false
                })
            }
            k.hide()
        }
        function B(av) {
            F();
            k.show();
            if (T.length > 0) {
                if (av == "filmstrip_1") {
                    b("img", ao).each(function(ax) {
                        var ay = T[ax];
                        b(this).click(function() {
                            j(ay);
                            b("li.img_selected", ah).removeClass("img_selected");
                            b(this).parent().parent().addClass("img_selected");
                            f.val(ax);
                            return false
                        })
                    })
                } else {
                    if (av == "filmstrip_2") {
                        if (s == 1) {
                            b("img", ao).each(function(ax) {
                                var ay = T[ax];
                                b(this).click(function() {
                                    K(ay);
                                    b("li.img_selected", ah).removeClass("img_selected");
                                    b(this).parent().parent().addClass("img_selected");
                                    f.val(ax);
                                    return false
                                })
                            })
                        } else {
                            b(function() {
                                var ax = {
                                    txtImage: J.txtImage,
                                    txtOf: J.txtOf,
                                    showDescription: at,
                                    parent_div_id: "photogallery_" + J.photogallery_id
                                };
                                b('img[rel*="lightbox"]', ah).lightBox(ax)
                            })
                        }
                    }
                }
                var aw = b("li.page_selected", ad);
                if (aw.next().length) {
                    ai.unbind("click").click(function() {
                        e("right");
                        return false
                    })
                } else {
                    ai.unbind("click")
                }
                if (aw.prev().length) {
                    U.unbind("click").click(function() {
                        e("left");
                        return false
                    })
                } else {
                    U.unbind("click")
                }
            }
            k.hide()
        }
        function F() {
            w = 1;
            ag.val(w);
            q = 0;
            f.val(q)
        }
        function N() {
            ai.unbind("click");
            ap = 0;
            ai.hide();
            Y.show();
            if (f.val() < p - 1) {
                z("right", "slideshow")
            } else {
                m()
            }
            var av = G[H]["delay"] || "5";
            var aw = av * 1000;
            if (aw < 850) {
                aw = "850"
            }
            setInterval(function() {
                if (ap == 0) {
                    Y.unbind("click");
                    if (f.val() == p - 1) {
                        m()
                    } else {
                        z("right", "slideshow")
                    }
                    Y.click(function() {
                        g();
                        return false
                    })
                }
            }, aw)
        }
        function m() {
            var av = b("li.img_selected", ad);
            av.removeClass("img_selected");
            f.val(0);
            var ay = b("li:eq(0)", ad);
            ay.addClass("img_selected");
            var ax = T[0];
            if (ax.description) {
                X.fadeOut("slow", function() {
                    X.html(ax.description)
                }).fadeIn("slow")
            } else {
                X.fadeOut("slow", function() {
                    X.empty()
                })
            }
            var aw = function() {
                k.hide()
            };
            k.show();
            ao.scrollTo(ay, 800, aw)
        }
        function g() {
            Y.unbind("click");
            ai.unbind("click").click(function() {
                L();
                return false
            });
            ap = 1;
            Y.hide();
            ai.show()
        }
        function L() {
            ai.unbind("click");
            Y.unbind("click").click(function() {
                g();
                return false
            });
            ap = 0;
            Y.show();
            ai.hide()
        }
        function e(aw) {
            if ((R) || (w == 1 && aw == "left") || ((an == w) && aw == "right")) {
                return (1)
            }
            R = 1;
            var ax = null;
            var av = w;
            var aC = b("li.page_selected", ad);
            var aB = 0;
            var aA = 0;
            if (aw == "left") {
                ax = aC.prev();
                av--;
                var az = b("img", ax);
                aB = parseInt(M) - az.length;
                aA = parseInt(M) - 1
            } else {
                ax = aC.next();
                av++;
                var az = b("img", ax);
                aB = parseInt(t) + 1;
                aA = parseInt(t) + az.length
            }
            S.html(aB);
            M = aB;
            aa.html(aA);
            t = aA;
            ag.val(av);
            w = av;
            aC.removeClass("page_selected");
            ax.addClass("page_selected");
            var ay = function() {
                R = 0
            };
            ao.scrollTo(ax, 800, ay);
            if (ax.next().length) {
                ai.unbind("click").css("color", "blue").click(function() {
                    e("right");
                    return false
                })
            } else {
                ai.css("color", "black");
                ai.unbind("click")
            }
            if (ax.prev().length) {
                U.unbind("click").css("color", "blue").click(function() {
                    e("left");
                    return false
                })
            } else {
                U.unbind("click").css("color", "black")
            }
        }
        function j(aD) {
            var av = ac + aD.thumbnail_thumb_path + u + "_thumbnail_" + aD.filecabinet_file_id + "_" + aD.unique_id + aD.thumbnail_extension;
            var aC = aD.description;
            var aE = aD.image_url_link;
            var aA = {
                id: "photogallery_image_" + aD.filecabinet_file_id,
                src: av
            };
            var aw = b('<img style="padding:0px; margin:0px;"></img>').attr(aA);
            var ax = b("div.main_image_container", ah);
            var aG = b("div.sub_image_container", ax);
            if (aE) {
                var ay = aE;
                var az = "";
                var aB = "_self";
                if (!(aE.match(/http:\/\//) || aE.match(/https:\/\//))) {
                    az = "http://"
                }
                if (aD.click_option == 2) {
                    aB = "_blank"
                }
                var aF = b('<a style="text-decoration:none;" href="' + az + ay + '" target="' + aB + '"></a>');
                aF.html(aw);
                aw = aF
            } else {}
            aG.empty();
            aG.html(aw);
            if (aC && (at == 1)) {
                P.empty();
                P.html('<p style="padding:0px; height:30px; overflow-x: hidden; overflow-y: auto; margin:0px;">' + aC + "</p>")
            } else {
                if (at == 1) {
                    P.html('<p style="padding:0px; margin:0px; height:30px; overflow-x: hidden; overflow-y: auto;"></p>')
                } else {
                    P.html('<p style="padding:0px; margin:0px; height:30px; overflow-x: hidden; overflow-y: auto; display:none;"></p>')
                }
            }
        }
        function K(av) {
            ar(av)
        }
        function ar(av) {
            var aw = new Image();
            aw.onload = function() {
                O(av, aw)
            };
            if (av.original_image_path) {
                aw.src = ac + av.original_image_path + "/" + av.original_image_name
            } else {
                aw.src = ac + av.original_image_name
            }
            b("<div></div>").addClass("photogallery_overlay").addClass("photogallery_thumbnail_overlay").css("z-index", 10100).appendTo("body").unbind("click").click(x)
        }
        function O(ax, aK) {
            aK.onload = null;
            var aM = ac + ax.original_image_name;
            if (ax.original_image_path) {
                aM = ac + ax.original_image_path + "/" + ax.original_image_name
            }
            var aL = ax.description;
            if (Z) {
                Z == null
            }
            if (!Z) {
                Z = b('<div class="photogallery_preview_window" style="font-size:0px"><img></img></div>').appendTo("body");
                var aJ = b('<div class="photogallery_preview_close"></div>');
                var ay = b('<div class="photogallery_preview_description"></div>');
                var aC = E.css("font-family");
                var aE = E.css("color");
                var av = E.css("font-size");
                var aB = E.css("font-weight");
                var aG = E.css("font-style");
                var aQ = E.css("font-variant");
                var aP = E.css("text-align");
                var aR = E.css("text-decoration");
                var aA = E.css("text-transform");
                var az = E.css("white-space");
                if (aC) {
                    ay.css("font-family", aC)
                }
                if (aE) {
                    ay.css("color", aE)
                }
                if (av) {
                    ay.css("font-size", av)
                }
                if (aB) {
                    ay.css("font-weight", aB)
                }
                if (aG) {
                    ay.css("font-style", aG)
                }
                if (aQ) {
                    ay.css("font-variant", aQ)
                }
                if (aP) {
                    ay.css("text-align", aP)
                }
                if (aR) {
                    ay.css("text-decoration", aR)
                }
                if (aA) {
                    ay.css("text-transform", aA)
                }
                if (az) {
                    ay.css("white-space", az)
                }
                aJ.appendTo(Z);
                ay.appendTo(Z);
                b("div.photogallery_preview_close", Z).unbind("click").click(x)
            }
            var aI = aK.width - 20;
            if (aL && at == 1) {
                b("div.photogallery_preview_description", Z).html('<p style="padding:0px; width:' + aI + 'px; height:auto; margin:0px;">' + aL + "</p>")
            } else {
                if (at == 1) {
                    b("div.photogallery_preview_description", Z).html('<p style="padding:0px; width:' + aI + 'px; height:auto; margin:0px;"></p>')
                } else {
                    b("div.photogallery_preview_description", Z).html('<p style="padding:0px; width:' + aI + 'px; height:auto; margin:0px; display:none;"></p>')
                }
            }
            var aO = aK.width;
            var aN = aK.height;
            var aF = b(window).width();
            var aD = b(window).height();
            if (aN > aD) {
                aN = aD - 40
            }
            b("img", Z).attr("src", aM).unbind("click").click(x).attr("height", aN);
            var aw = 0 - (parseInt(Z.width()) / 2);
            var aH = 0 - (parseInt(Z.height()) / 2);
            if (b.browser.msie && (b.browser.version < 7)) {
                aH += b(window).scrollTop();
                b("select").hide()
            }
            Z.css({
                "margin-top": aH + "px",
                "margin-left": aw + "px"
            }).show()
        }
        function x() {
            Z.hide();
            b("div.photogallery_thumbnail_overlay").remove()
        }
        function ab(aw) {
            var ax = ac + "/" + aw.original_image_path + "/" + aw.original_image_name;
            var aw = {
                id: "photogallery_image_" + aw.filecabinet_file_id,
                src: ax
            };
            var av = b('<img style="padding:5px;"></img>').attr(aw);
            return (av)
        }
        function I(az) {
            var aC = ac + az.thumbnail_thumb_path;
            var av = az.original_image_name;
            var aB = az.description;
            var ay = ac + az.original_image_path + "/" + az.original_image_name;
            var ax = "photogallery_image_" + az.filecabinet_file_id;
            var aA = {
                id: ax,
                src: aC
            };
            if (aB) {
                aA.title = aB
            } else {
                aA.title = av
            }
            var aw = b('<img rel="lightbox[gallery]" href="' + ay + '"></img>').attr(aA);
            return (aw)
        }
        function z(av, ax) {
            if ((R) || (q == 0 && av == "left") || ((p == q) && av == "right")) {
                return (1)
            }
            R = 1;
            q = f.val();
            var aB = null;
            var az = q;
            var aw = b("li.img_selected", ad);
            if (av == "left") {
                aB = aw.prev();
                az--;
                q--
            } else {
                aB = aw.next();
                az++;
                q++
            }
            f.val(az);
            aw.removeClass("img_selected");
            aB.addClass("img_selected");
            var ay = function() {
                R = 0
            };
            ao.scrollTo(aB, 800, ay);
            var aA = T[az];
            if (ax != "grid") {
                if (aA.description) {
                    if (at == 1) {
                        X.fadeOut("slow", function() {
                            X.html(aA.description)
                        }).fadeIn("slow")
                    }
                } else {
                    X.fadeOut("slow", function() {
                        X.empty()
                    })
                }
            }
            if ((ax != "slideshow") && (ax != "grid")) {
                if (aB.next().length) {
                    ai.show()
                } else {
                    ai.hide()
                }
                if (aB.prev().length) {
                    U.show()
                } else {
                    U.hide()
                }
            }
        }
    };
    b.fn.iv_socialnetwork = function(c) {
        return this.each(function() {
            new b.iv.socialnetwork(this, c)
        })
    };
    b.iv.socialnetwork = function(m, w) {
        w = b.extend({
            name: null,
            id: null,
            url: null,
            path: null,
            title: null,
            browser: null,
            ws_name: null
        }, w);
        b.data(m, w.name + "_" + w.id, this);
        var t = b(m);
        var q = w.id;
        var g = w.name;
        var j = w.url;
        var v = w.path;
        var c = b("img", t);
        var f = w.name + "_" + w.id;
        var s = w.title;
        var p = w.ws_name;
        if ((g == "socialnetwork") || (g == "socialnetwork5")) {
            var o = "_blank";
            if (g == "socialnetwork") {
                var u = b('<a target="' + o + '" style="text-decoration:none; padding:0px; margin:0px; border:0px; display:block;" href="http://www.facebook.com/share.php?u=' + j + '"></a>');
                u.html(c);
                t.empty();
                t.html(u)
            } else {
                var r = /\//;
                v = v.replace(r, "");
                var k = "url=" + j;
                k = k + "&srcURL=" + j;
                if (s) {
                    if (s.match(p)) {
                        k = k + "&title=" + s;
                        k = k + "&srcTitle=" + s
                    } else {
                        k = k + "&title=" + p + " - " + s;
                        k = k + "&srcTitle=" + p + " - " + s
                    }
                } else {
                    if (p) {
                        k = k + "&title=" + p;
                        k = k + "&srcTitle=" + p
                    } else {
                        k = k + "&title=" + j;
                        k = k + "&srcTitle=" + j
                    }
                }
                var u = b('<a target="_blank" style="text-decoration:none; padding:0px; margin:0px; border:0px; display:block;" href="http://www.google.com/buzz/post?' + k + '"></a>');
                u.html(c);
                t.empty();
                t.html(u)
            }
        } else {
            j = decodeURI(j);
            var n = {
                version: "2.0.1",
                login: "ivenue",
                apiKey: "R_064320bbb6b53267e152fa2317b0e72b",
                history: "0",
                longUrl: j
            };
            var e = "http://api.bit.ly/shorten?version=" + n.version + "&longUrl=" + n.longUrl + "&login=" + n.login + "&apiKey=" + n.apiKey + "&history=" + n.history + "&format=json&callback=?";
            b.getJSON(e, function(B) {
                var A = B.errorCode;
                var y;
                var x;
                if (A > 0) {
                    var y = B.errorMessage
                } else {
                    var D = B.results[n.longUrl].errorCode;
                    if (D > 0) {
                        y = B.results[n.longUrl].errorMessage
                    } else {
                        x = B.results[n.longUrl].shortUrl
                    }
                }
                if (y) {
                    c.click(function() {
                        alert(y);
                        return false
                    })
                } else {
                    var C = "_blank";
                    var z = null;
                    if (g == "socialnetwork3") {
                        var z = b('<a target="' + C + '" style="text-decoration:none; padding:0px; margin:0px; border:0px; display:block;" href="http://twitter.com/home/?status=' + x + '"></a>')
                    } else {
                        var z = b('<a target="' + C + '" style="text-decoration:none; padding:0px; margin:0px; border:0px; display:block;" href="http://www.plurk.com/m?qualifier=shares&content=' + x + '"></a>')
                    }
                    z.html(c);
                    t.empty();
                    t.html(z)
                }
            })
        }
    };
    $.fn.iv_unselectable = function() {
        return (this.each(function() {
            $(this).css("-moz-user-select", "none").css("-khtml-user-select", "none").css("-o-user-select", "none").css("user-select", "none");
            if ($.browser.msie) {
                $(this).each(function() {
                    this.ondrag = function() {
                        return (false)
                    };
                    this.onselectstart = function() {
                        return (false)
                    }
                })
            } else {
                if ($.browser.opera) {
                    $(this).attr("unselectable", "on")
                }
            }
        }))
    };
    $.fn.iv_initial_text = function(c) {
        return (this.each(function() {
            new $.iv.initial_text(this, c)
        }))
    };
    $.iv.initial_text = function(f, e) {
        var e = $.extend({
            initial_text: ""
        }, e);
        var c = $(f);
        c.focus(function() {
            if (c.val() == e.initial_text) {
                c.val("")
            }
        });
        c.blur(function() {
            if (c.val() == "") {
                c.val(e.initial_text)
            }
        })
    }
})(jQuery);
function num(b, c) {
    return b[0] && parseInt(jQuery.curCSS(b[0], c, true), 10) || 0
}
var open_sub_menus = [];
var current_menu_item;
var current_parent_menu;
var hide_timeouts = [];
function _show_sub_menu(b, k) {
    function p() {
        var A = document;
        return Math.max(Math.max(A.body.scrollHeight, A.documentElement.scrollHeight), Math.max(A.body.offsetHeight, A.documentElement.offsetHeight), Math.max(A.body.clientHeight, A.documentElement.clientHeight))
    }
    var u = jQuery(b).attr("id");
    while (hide_timeouts.length) {
        clearTimeout(hide_timeouts.pop())
    }
    if (open_sub_menus.length) {
        if (k == current_parent_menu) {
            var c = open_sub_menus.pop();
            c.css("visibility", "hidden")
        } else {
            if (!k) {
                while (open_sub_menus.length) {
                    var c = open_sub_menus.pop();
                    c.css("visibility", "hidden")
                }
            }
        }
    }
    current_parent_menu = k;
    var t = jQuery("#sub_" + u);
    open_sub_menus.push(t);
    var f = t.parents("div:first");
    var v = parseInt(jQuery(b).css("padding-left").replace(/px/, ""));
    var s = parseInt(jQuery(b).css("padding-right").replace(/px/, ""));
    var g = parseInt(jQuery(b).css("padding-top").replace(/px/, ""));
    var z = parseInt(jQuery(b).css("padding-bottom").replace(/px/, ""));
    var e = parseInt(t.css("padding-left").replace(/px/, ""));
    var y = parseInt(t.css("padding-right").replace(/px/, ""));
    var q = parseInt(t.css("padding-top").replace(/px/, ""));
    var m = parseInt(t.css("padding-bottom").replace(/px/, ""));
    var o;
    if (t.attr("menu_position")) {
        o = t.attr("menu_position")
    }
    if (o == "top") {
        if (k) {
            if (jQuery(b).offset().top + jQuery(b).height() + q + m + t.height() > jQuery(document).height()) {
                t.css("top", jQuery(b).position().top - t.height() + jQuery(b).height() + g + z)
            } else {
                t.css("top", jQuery(b).position().top - q)
            }
            if (jQuery(b).offset().left + jQuery(b).width() + v + s + t.width() > document.documentElement.clientWidth) {
                t.css("left", jQuery(b).position().left - t.width())
            } else {
                t.css("left", jQuery(b).position().left + jQuery(b).width() + v + s)
            }
        } else {
            var x;
            if (f.css("position")) {
                var r = jQuery(b);
                if (r.position().top + r.height() + g + z + t.height() + 35 > jQuery(document).height()) {
                    x = r.position().top - t.height()
                } else {
                    x = r.position().top + r.height() + g + z
                }
            }
            t.css("top", x);
            if (jQuery(b).offset().left - e + t.width() > document.documentElement.clientWidth) {
                t.css("left", document.documentElement.clientWidth - t.width())
            } else {
                t.css("left", jQuery(b).position().left - e)
            }
        }
    } else {
        if (o == "left") {
            if (k) {
                if (jQuery(b).offset().top + m + q + t.height() > p()) {
                    t.css("top", jQuery(b).position().top - t.height() + jQuery(b).height() + g + z)
                } else {
                    t.css("top", jQuery(b).position().top - q)
                }
            } else {
                if (jQuery(b).position().top - g + t.height() > p()) {
                    t.css("top", p() - t.height() - g)
                } else {
                    t.css("top", jQuery(b).position().top)
                }
            }
            if (jQuery(b).offset().left + jQuery(b).width() + v + s + t.width() > document.documentElement.clientWidth) {
                t.css("left", jQuery(b).position().left - t.width())
            } else {
                t.css("left", jQuery(b).position().left + jQuery(b).width() + v + s)
            }
        } else {
            if (o == "right") {
                var w = jQuery(t).width();
                var n = 400;
                if (w >= n) {
                    w = n;
                    jQuery(t).width(n)
                }
                t.css("top", jQuery(b).position().top - q);
                var j;
                if (jQuery(b).offset().left - w - e - y < 0) {
                    j = jQuery(b).position().left + w + e + y
                } else {
                    j = jQuery(b).position().left - w - e - y
                }
                t.css("left", j)
            } else {
                if (o == "bottom") {
                    if (k) {
                        t.css("top", jQuery(b).position().top - t.height() + jQuery(b).height());
                        if (jQuery(b).offset().left + jQuery(b).width() + v + s + t.width() > document.documentElement.clientWidth) {
                            t.css("left", jQuery(b).position().left - t.width())
                        } else {
                            t.css("left", jQuery(b).position().left + jQuery(b).width() + v + s)
                        }
                    } else {
                        var x;
                        if (f.css("position")) {
                            var r = jQuery(b);
                            x = r.position().top - g - t.height()
                        }
                        t.css("top", x);
                        if (jQuery(b).offset().left - e + t.width() > document.documentElement.clientWidth) {
                            t.css("left", document.documentElement.clientWidth - t.width())
                        } else {
                            t.css("left", jQuery(b).position().left - e)
                        }
                    }
                }
            }
        }
    }
    t.css("visibility", "visible");
    t.show();
    t.css("z-index", 50000);
    $(".video video").each(function() {
        $(this).get(0).pause()
    });
    $(".video video").addClass("hidden");
    $(".video img.video_preview").removeClass("hidden")
}
function _hide_sub_menus(b) {
    var c;
    if (b) {
        c = 1
    } else {
        c = 500
    }
    hide_timeouts.push(setTimeout(function() {
        if (!current_menu_item) {
            while (open_sub_menus.length) {
                var e = open_sub_menus.pop();
                e.css("visibility", "hidden");
                $(".video video").removeClass("hidden");
                $(".video img.video_preview").addClass("hidden")
            }
        }
    }, c))
}(function(b) {
    b.fn.iv_button_select = function(c) {
        return this.each(function() {
            new b.iv.button_select(this, c)
        })
    };
    b.iv.button_select = function(g, q) {
        b.data(g, "button_select", this);
        q = b.extend({
            button_label: null,
            classes: null,
            on_button_click: null,
            options: [],
            select_id: null,
            select_name: null,
            selected: null
        }, q);
        select_options = JSON.parse(q.options || []);
        var p = b(g);
        var k = q.selected;
        var j = false;
        this.refresh_position = o;
        m();
        function o() {
            var v = b(".button_select_container", p);
            var t = b(".button_select_dropdown_container", p);
            var z = v.parent().position();
            t.css({
                "min-width": null
            });
            var s = t.width();
            var A = t.height();
            var u = z.left;
            var x = z.top;
            var y = v.width();
            var w = v.height();
            if (v.width() > t.width()) {
                s = v.width()
            }
            t.css({
                "min-width": s
            });
            s = t.width();
            if (b(window).width() < (u + s)) {
                u = u + y - s
            }
            t.css({
                left: u
            });
            if (b(window).height() < (w + x + A - b(window).scrollTop())) {
                x = x - A
            } else {
                x = x + w
            }
            t.css({
                top: x
            })
        }
        function m() {
            r();
            c();
            p.click(function(s) {
                b("body").one("click", function() {
                    n()
                });
                s.stopPropagation()
            });
            b(window).unbind("resize", o);
            b(window).bind("resize", o)
        }
        function r() {
            p.empty();
            var t = b('<span class="button_select_container"></span>');
            var u = b('<span class="button_select_label"><span>' + q.button_label + "</span></span>");
            if (q.on_button_click) {
                u.click(q.on_button_click)
            }
            u.hover(function() {
                b(this).addClass("hover")
            }, function() {
                b(this).removeClass("hover")
            });
            var s;
            if (select_options.length > 1) {
                s = b('<span class="button_select_expand_btn"><span>▼</span></span>').click(f);
                s.hover(function() {
                    b(this).addClass("hover")
                }, function() {
                    b(this).removeClass("hover")
                })
            }
            p.append(t);
            t.append(u);
            if (select_options.length > 1) {
                t.append(s)
            }
            t.append('<input class="button_select_value" id="' + q.select_id + '" name="' + ((q.select_name) ? q.select_name : q.select_id) + '" type="hidden" value="' + k + '"/>');
            var v = b('<div class="button_select_dropdown_container"></div>');
            v.iv_unselectable();
            var w = t.position();
            v.css({
                position: "absolute",
                left: w.left
            });
            v.addClass("hidden");
            p.append(v)
        }
        function f() {
            o();
            if (!j) {
                e()
            } else {
                n()
            }
        }
        function n() {
            b(".button_select_expand_btn", p).removeClass("expanded");
            var s = b(".button_select_dropdown_container", p);
            s.addClass("hidden");
            j = false
        }
        function e() {
            b(".button_select_expand_btn", p).addClass("expanded");
            var s = b(".button_select_dropdown_container", p);
            s.removeClass("hidden");
            j = true
        }
        function c() {
            var s = b(".button_select_dropdown_container", p);
            s.empty();
            b.each(select_options, function() {
                var u = this;
                var t = b('<div class="button_select_option" value="' + u.id + '"><span unselectable="on">' + u.string + "</span></div>");
                t.click(function() {
                    var v = b(this).attr("value");
                    var w = b(".button_select_label span", p);
                    w.empty().html(b(this).html());
                    k = v;
                    b("#" + q.select_id).val(k);
                    n()
                });
                t.hover(function() {
                    b(this).addClass("hover")
                }, function() {
                    b(this).removeClass("hover")
                });
                s.append(t);
                s.iv_unselectable()
            })
        }
    }
})(jQuery);
(function(au, y) {
    var N = {
        defaultView: "month",
        aspectRatio: 1.35,
        header: {
            left: "title",
            center: "",
            right: "today prev,next"
        },
        weekends: true,
        allDayDefault: true,
        ignoreTimezone: true,
        lazyFetching: true,
        startParam: "start",
        endParam: "end",
        titleFormat: {
            month: "MMMM yyyy",
            week: "MMM d[ yyyy]{ '&#8212;'[ MMM] d yyyy}",
            day: "dddd, MMM d, yyyy"
        },
        columnFormat: {
            month: "ddd",
            week: "ddd M/d",
            day: "dddd M/d"
        },
        timeFormat: {
            "": "h(:mm)t"
        },
        isRTL: false,
        firstDay: 0,
        monthNames: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        monthNamesShort: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        dayNames: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
        dayNamesShort: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
        buttonText: {
            prev: "&nbsp;&#9668;&nbsp;",
            next: "&nbsp;&#9658;&nbsp;",
            prevYear: "&nbsp;&lt;&lt;&nbsp;",
            nextYear: "&nbsp;&gt;&gt;&nbsp;",
            today: "today",
            month: "month",
            week: "week",
            day: "day"
        },
        theme: false,
        buttonIcons: {
            prev: "circle-triangle-w",
            next: "circle-triangle-e"
        },
        unselectAuto: true,
        dropAccept: "*"
    };
    var T = {
        header: {
            left: "next,prev today",
            center: "",
            right: "title"
        },
        buttonText: {
            prev: "&nbsp;&#9658;&nbsp;",
            next: "&nbsp;&#9668;&nbsp;",
            prevYear: "&nbsp;&gt;&gt;&nbsp;",
            nextYear: "&nbsp;&lt;&lt;&nbsp;"
        },
        buttonIcons: {
            prev: "circle-triangle-e",
            next: "circle-triangle-w"
        }
    };
    var aA = au.fullCalendar = {
        version: "1.5.4"
    };
    var af = aA.views = {};
    au.fn.fullCalendar = function(aM) {
        if (typeof aM == "string") {
            var aL = Array.prototype.slice.call(arguments, 1);
            var aN;
            this.each(function() {
                var aP = au.data(this, "fullCalendar");
                if (aP && au.isFunction(aP[aM])) {
                    var aO = aP[aM].apply(aP, aL);
                    if (aN === y) {
                        aN = aO
                    }
                    if (aM == "destroy") {
                        au.removeData(this, "fullCalendar")
                    }
                }
            });
            if (aN !== y) {
                return aN
            }
            return this
        }
        var aK = aM.eventSources || [];
        delete aM.eventSources;
        if (aM.events) {
            aK.push(aM.events);
            delete aM.events
        }
        aM = au.extend(true, {}, N, (aM.isRTL || aM.isRTL === y && N.isRTL) ? T : {}, aM);
        this.each(function(aQ, aO) {
            var aP = au(aO);
            var aR = new v(aP, aM, aK);
            aP.data("fullCalendar", aR);
            aR.render()
        });
        return this
    };
    function ak(aK) {
        au.extend(true, N, aK)
    }
    function v(a2, a9, bc) {
        var br = this;
        br.options = a9;
        br.render = bi;
        br.destroy = bt;
        br.refetchEvents = aW;
        br.reportEvents = a0;
        br.reportEventChange = bv;
        br.rerenderEvents = aO;
        br.changeView = aT;
        br.select = bs;
        br.unselect = aV;
        br.prev = be;
        br.next = aN;
        br.prevYear = bu;
        br.nextYear = aL;
        br.today = a3;
        br.gotoDate = bh;
        br.incrementDate = a6;
        br.formatDate = function(bC, bB) {
            return C(bC, bB, a9)
        };
        br.formatDates = function(bD, bC, bB) {
            return m(bD, bC, bB, a9)
        };
        br.getDate = a7;
        br.getView = bd;
        br.option = bo;
        br.trigger = bg;
        u.call(br, a9, bc);
        var aK = br.isFetchNeeded;
        var by = br.fetchEvents;
        var bp = a2[0];
        var aP;
        var bj;
        var aY;
        var bA;
        var aX;
        var bz = {};
        var bf;
        var aR;
        var aU;
        var bk = 0;
        var bq = 0;
        var bw = new Date();
        var bb = [];
        var aQ;
        A(bw, a9.year, a9.month, a9.date);
        function bi(bB) {
            if (!aY) {
                ba()
            } else {
                bl();
                bx();
                aM();
                a8(bB)
            }
        }
        function ba() {
            bA = a9.theme ? "ui" : "fc";
            a2.addClass("fc");
            if (a9.isRTL) {
                a2.addClass("fc-rtl")
            }
            if (a9.theme) {
                a2.addClass("ui-widget")
            }
            aY = au("<div class='fc-content' style='position:relative'/>").prependTo(a2);
            aP = new U(br, a9);
            bj = aP.render();
            if (bj) {
                a2.prepend(bj)
            }
            aT(a9.defaultView);
            au(window).one("resize", aZ);
            if (!a1()) {
                aS()
            }
        }
        function aS() {
            setTimeout(function() {
                if (!aX.start && a1()) {
                    a8()
                }
            }, 0)
        }
        function bt() {
            au(window).unbind("resize", aZ);
            aP.destroy();
            aY.remove();
            a2.removeClass("fc fc-rtl ui-widget")
        }
        function bm() {
            return bp.offsetWidth !== 0
        }
        function a1() {
            return au("body")[0].offsetWidth !== 0
        }
        function aT(bD) {
            if (!aX || bD != aX.name) {
                bq++;
                aV();
                var bC = aX;
                var bB;
                if (bC) {
                    (bC.beforeHide || ar)();
                    K(aY, aY.height());
                    bC.element.hide()
                } else {
                    K(aY, 1)
                }
                aY.css("overflow", "hidden");
                aX = bz[bD];
                if (aX) {
                    aX.element.show()
                } else {
                    aX = bz[bD] = new af[bD](bB = aU = au("<div class='fc-view fc-view-" + bD + "' style='position:absolute'/>").appendTo(aY), br)
                }
                if (bC) {
                    aP.deactivateButton(bC.name)
                }
                aP.activateButton(bD);
                a8();
                aY.css("overflow", "");
                if (bC) {
                    K(aY, 1)
                }
                if (!bB) {
                    (aX.afterShow || ar)()
                }
                bq--
            }
        }
        function a8(bD) {
            if (bm()) {
                bq++;
                aV();
                if (aR === y) {
                    bl()
                }
                var bC = false;
                if (!aX.start || bD || bw < aX.start || bw >= aX.end) {
                    aX.render(bw, bD || 0);
                    bn(true);
                    bC = true
                } else {
                    if (aX.sizeDirty) {
                        aX.clearEvents();
                        bn();
                        bC = true
                    } else {
                        if (aX.eventsDirty) {
                            aX.clearEvents();
                            bC = true
                        }
                    }
                }
                aX.sizeDirty = false;
                aX.eventsDirty = false;
                a5(bC);
                bf = a2.outerWidth();
                aP.updateTitle(aX.title);
                var bB = new Date();
                if (bB >= aX.start && bB < aX.end) {
                    aP.disableButton("today")
                } else {
                    aP.enableButton("today")
                }
                bq--;
                aX.trigger("viewDisplay", bp)
            }
        }
        function a4() {
            bx();
            if (bm()) {
                bl();
                bn();
                aV();
                aX.clearEvents();
                aX.renderEvents(bb);
                aX.sizeDirty = false
            }
        }
        function bx() {
            au.each(bz, function(bB, bC) {
                bC.sizeDirty = true
            })
        }
        function bl() {
            if (a9.contentHeight) {
                aR = a9.contentHeight
            } else {
                if (a9.height) {
                    aR = a9.height - (bj ? bj.height() : 0) - G(aY)
                } else {
                    aR = Math.round(aY.width() / Math.max(a9.aspectRatio, 0.5))
                }
            }
        }
        function bn(bB) {
            bq++;
            aX.setHeight(aR, bB);
            if (aU) {
                aU.css("position", "relative");
                aU = null
            }
            aX.setWidth(aY.width(), bB);
            bq--
        }
        function aZ() {
            au(window).unbind("resize", aZ);
            if (!bq) {
                if (aX.start) {
                    var bB=++bk;
                    setTimeout(function() {
                        if (bB == bk&&!bq && bm()) {
                            if (bf != (bf = a2.outerWidth())) {
                                bq++;
                                a4();
                                aX.trigger("windowResize", bp);
                                bq--
                            }
                        }
                    }, 200)
                } else {
                    aS()
                }
            }
            setTimeout(function() {
                au(window).unbind("resize", aZ).one("resize", aZ)
            }, 200)
        }
        function a5(bB) {
            if (!a9.lazyFetching || aK(aX.visStart, aX.visEnd)) {
                aW()
            } else {
                if (bB) {
                    aO()
                }
            }
        }
        function aW() {
            by(aX.visStart, aX.visEnd)
        }
        function a0(bB) {
            bb = bB;
            aO()
        }
        function bv(bB) {
            aO(bB)
        }
        function aO(bB) {
            aM();
            if (bm()) {
                aX.clearEvents();
                aX.renderEvents(bb, bB);
                aX.eventsDirty = false
            }
        }
        function aM() {
            au.each(bz, function(bB, bC) {
                bC.eventsDirty = true
            })
        }
        function bs(bD, bB, bC) {
            aX.select(bD, bB, bC === y ? true : bC)
        }
        function aV() {
            if (aX) {
                aX.unselect()
            }
        }
        function be() {
            a8( - 1)
        }
        function aN() {
            a8(1)
        }
        function bu() {
            aj(bw, - 1);
            a8()
        }
        function aL() {
            aj(bw, 1);
            a8()
        }
        function a3() {
            bw = new Date();
            a8()
        }
        function bh(bC, bD, bB) {
            if (bC instanceof Date) {
                bw = O(bC)
            } else {
                A(bw, bC, bD, bB)
            }
            a8()
        }
        function a6(bC, bB, bD) {
            if (bC !== y) {
                aj(bw, bC)
            }
            if (bB !== y) {
                r(bw, bB)
            }
            if (bD !== y) {
                aC(bw, bD)
            }
            a8()
        }
        function a7() {
            return O(bw)
        }
        function bd() {
            return aX
        }
        function bo(bB, bC) {
            if (bC === y) {
                return a9[bB]
            }
            if (bB == "height" || bB == "contentHeight" || bB == "aspectRatio") {
                a9[bB] = bC;
                a4()
            }
        }
        function bg(bB, bC) {
            if (a9[bB]) {
                return a9[bB].apply(bC || bp, Array.prototype.slice.call(arguments, 2))
            }
        }
        if (a9.droppable) {
            au(document).bind("dragstart", function(bD, bE) {
                var bB = bD.target;
                var bF = au(bB);
                if (!bF.parents(".fc").length) {
                    var bC = a9.dropAccept;
                    if (au.isFunction(bC) ? bC.call(bB, bF) : bF.is(bC)) {
                        aQ = bB;
                        aX.dragStart(aQ, bD, bE)
                    }
                }
            }).bind("dragstop", function(bB, bC) {
                if (aQ) {
                    aX.dragStop(aQ, bB, bC);
                    aQ = null
                }
            })
        }
    }
    function U(aN, aW) {
        var aV = this;
        aV.render = aL;
        aV.destroy = aR;
        aV.updateTitle = aP;
        aV.activateButton = aK;
        aV.deactivateButton = aT;
        aV.disableButton = aM;
        aV.enableButton = aQ;
        var aO = au([]);
        var aS;
        function aL() {
            aS = aW.theme ? "ui" : "fc";
            var aX = aW.header;
            if (aX) {
                aO = au("<table class='fc-header' style='width:100%'/>").append(au("<tr/>").append(aU("left")).append(aU("center")).append(aU("right")));
                return aO
            }
        }
        function aR() {
            aO.remove()
        }
        function aU(aX) {
            var aZ = au("<td class='fc-header-" + aX + "'/>");
            var aY = aW.header[aX];
            if (aY) {
                au.each(aY.split(" "), function(a1) {
                    if (a1 > 0) {
                        aZ.append("<span class='fc-header-space'/>")
                    }
                    var a0;
                    au.each(this.split(","), function(a4, a3) {
                        if (a3 == "title") {
                            aZ.append("<span class='fc-header-title'><h2>&nbsp;</h2></span>");
                            if (a0) {
                                a0.addClass(aS + "-corner-right")
                            }
                            a0 = null
                        } else {
                            var a2;
                            if (aN[a3]) {
                                a2 = aN[a3]
                            } else {
                                if (af[a3]) {
                                    a2 = function() {
                                        a5.removeClass(aS + "-state-hover");
                                        aN.changeView(a3)
                                    }
                                }
                            }
                            if (a2) {
                                var a6 = aW.theme ? F(aW.buttonIcons, a3): null;
                                var a7 = F(aW.buttonText, a3);
                                var a5 = au("<span class='fc-button fc-button-" + a3 + " " + ((a3 == "prev" || a3 == "next") ? "fc-button-nav " : "") + aS + "-state-default'><span class='fc-button-inner'><span class='fc-button-content'>" + (a6 ? "<span class='fc-icon-wrap'><span class='ui-icon ui-icon-" + a6 + "'/></span>" : a7) + "</span><span class='fc-button-effect'><span></span></span></span></span>");
                                if (a5) {
                                    a5.click(function() {
                                        if (!a5.hasClass(aS + "-state-disabled")) {
                                            a2()
                                        }
                                    }).mousedown(function() {
                                        a5.not("." + aS + "-state-active").not("." + aS + "-state-disabled").addClass(aS + "-state-down")
                                    }).mouseup(function() {
                                        a5.removeClass(aS + "-state-down")
                                    }).hover(function() {
                                        a5.not("." + aS + "-state-active").not("." + aS + "-state-disabled").addClass(aS + "-state-hover")
                                    }, function() {
                                        a5.removeClass(aS + "-state-hover").removeClass(aS + "-state-down")
                                    }).appendTo(aZ);
                                    if (!a0) {
                                        a5.addClass(aS + "-corner-left")
                                    }
                                    a0 = a5
                                }
                            }
                        }
                    });
                    if (a0) {
                        a0.addClass(aS + "-corner-right")
                    }
                })
            }
            return aZ
        }
        function aP(aX) {
            aO.find("h2").html(aX)
        }
        function aK(aX) {
            aO.find("span.fc-button-" + aX).addClass(aS + "-state-active")
        }
        function aT(aX) {
            aO.find("span.fc-button-" + aX).removeClass(aS + "-state-active")
        }
        function aM(aX) {
            aO.find("span.fc-button-" + aX).addClass(aS + "-state-disabled")
        }
        function aQ(aX) {
            aO.find("span.fc-button-" + aX).removeClass(aS + "-state-disabled")
        }
    }
    aA.sourceNormalizers = [];
    aA.sourceFetchers = [];
    var p = {
        dataType: "json",
        cache: false
    };
    var X = 1;
    function u(aQ, aY) {
        var a0 = this;
        a0.isFetchNeeded = aS;
        a0.fetchEvents = aZ;
        a0.addEventSource = aN;
        a0.removeEventSource = aO;
        a0.updateEvent = a3;
        a0.renderEvent = aW;
        a0.removeEvents = bf;
        a0.clientEvents = a9;
        a0.normalizeEvent = aV;
        var a2 = a0.trigger;
        var aK = a0.getView;
        var a5 = a0.reportEvents;
        var a7 = {
            events: []
        };
        var aL = [a7];
        var bc, a8;
        var aM = 0;
        var bd = 0;
        var bb = 0;
        var a1 = [];
        for (var ba = 0; ba < aY.length; ba++) {
            a4(aY[ba])
        }
        function aS(bh, bg) {
            return !bc || bh < bc || bg > a8
        }
        function aZ(bk, bh) {
            bc = bk;
            a8 = bh;
            a1 = [];
            var bj=++aM;
            var bg = aL.length;
            bd = bg;
            for (var bi = 0; bi < bg; bi++) {
                aR(aL[bi], bj)
            }
        }
        function aR(bh, bg) {
            a6(bh, function(bj) {
                if (bg == aM) {
                    if (bj) {
                        for (var bi = 0; bi < bj.length; bi++) {
                            bj[bi].source = bh;
                            aV(bj[bi])
                        }
                        a1 = a1.concat(bj)
                    }
                    bd--;
                    if (!bd) {
                        a5(a1)
                    }
                }
            })
        }
        function a6(bg, bq) {
            var bl;
            var bp = aA.sourceFetchers;
            var bn;
            for (bl = 0; bl < bp.length; bl++) {
                bn = bp[bl](bg, bc, a8, bq);
                if (bn === true) {
                    return 
                } else {
                    if (typeof bn == "object") {
                        a6(bn, bq);
                        return 
                    }
                }
            }
            var bs = bg.events;
            if (bs) {
                if (au.isFunction(bs)) {
                    aU();
                    bs(O(bc), O(a8), function(bt) {
                        bq(bt);
                        be()
                    })
                } else {
                    if (au.isArray(bs)) {
                        bq(bs)
                    } else {
                        bq()
                    }
                }
            } else {
                var bh = bg.url;
                if (bh) {
                    var br = bg.success;
                    var bo = bg.error;
                    var bi = bg.complete;
                    var bk = au.extend({}, bg.data || {});
                    var bm = aI(bg.startParam, aQ.startParam);
                    var bj = aI(bg.endParam, aQ.endParam);
                    if (bm) {
                        bk[bm] = Math.round( + bc / 1000)
                    }
                    if (bj) {
                        bk[bj] = Math.round( + a8 / 1000)
                    }
                    aU();
                    au.ajax(au.extend({}, p, bg, {
                        data: bk,
                        success: function(bu) {
                            bu = bu || [];
                            var bt = J(br, this, arguments);
                            if (au.isArray(bt)) {
                                bu = bt
                            }
                            bq(bu)
                        },
                        error: function() {
                            J(bo, this, arguments);
                            bq()
                        },
                        complete: function() {
                            J(bi, this, arguments);
                            be()
                        }
                    }))
                } else {
                    bq()
                }
            }
        }
        function aN(bg) {
            bg = a4(bg);
            if (bg) {
                bd++;
                aR(bg, aM)
            }
        }
        function a4(bg) {
            if (au.isFunction(bg) || au.isArray(bg)) {
                bg = {
                    events: bg
                }
            } else {
                if (typeof bg == "string") {
                    bg = {
                        url: bg
                    }
                }
            }
            if (typeof bg == "object") {
                aX(bg);
                aL.push(bg);
                return bg
            }
        }
        function aO(bg) {
            aL = au.grep(aL, function(bh) {
                return !aT(bh, bg)
            });
            a1 = au.grep(a1, function(bh) {
                return !aT(bh.source, bg)
            });
            a5(a1)
        }
        function a3(bl) {
            var bj, bh = a1.length, bm, bg = aK().defaultEventEnd, bk = bl.start - bl._start, bi = bl.end ? (bl.end - (bl._end || bg(bl))): 0;
            for (bj = 0; bj < bh; bj++) {
                bm = a1[bj];
                if (bm._id == bl._id && bm != bl) {
                    bm.start = new Date( + bm.start + bk);
                    if (bl.end) {
                        if (bm.end) {
                            bm.end = new Date( + bm.end + bi)
                        } else {
                            bm.end = new Date( + bg(bm) + bi)
                        }
                    } else {
                        bm.end = null
                    }
                    bm.title = bl.title;
                    bm.url = bl.url;
                    bm.allDay = bl.allDay;
                    bm.className = bl.className;
                    bm.editable = bl.editable;
                    bm.color = bl.color;
                    bm.backgroudColor = bl.backgroudColor;
                    bm.borderColor = bl.borderColor;
                    bm.textColor = bl.textColor;
                    aV(bm)
                }
            }
            aV(bl);
            a5(a1)
        }
        function aW(bh, bg) {
            aV(bh);
            if (!bh.source) {
                if (bg) {
                    a7.events.push(bh);
                    bh.source = a7
                }
                a1.push(bh)
            }
            a5(a1)
        }
        function bf(bh) {
            if (!bh) {
                a1 = [];
                for (var bg = 0; bg < aL.length; bg++) {
                    if (au.isArray(aL[bg].events)) {
                        aL[bg].events = []
                    }
                }
            } else {
                if (!au.isFunction(bh)) {
                    var bi = bh + "";
                    bh = function(bj) {
                        return bj._id == bi
                    }
                }
                a1 = au.grep(a1, bh, true);
                for (var bg = 0; bg < aL.length; bg++) {
                    if (au.isArray(aL[bg].events)) {
                        aL[bg].events = au.grep(aL[bg].events, bh, true)
                    }
                }
            }
            a5(a1)
        }
        function a9(bg) {
            if (au.isFunction(bg)) {
                return au.grep(a1, bg)
            } else {
                if (bg) {
                    bg += "";
                    return au.grep(a1, function(bh) {
                        return bh._id == bg
                    })
                }
            }
            return a1
        }
        function aU() {
            if (!bb++) {
                a2("loading", null, true)
            }
        }
        function be() {
            if (!--bb) {
                a2("loading", null, false)
            }
        }
        function aV(bh) {
            var bi = bh.source || {};
            var bg = aI(bi.ignoreTimezone, aQ.ignoreTimezone);
            bh._id = bh._id || (bh.id === y ? "_fc" + X++ : bh.id + "");
            if (bh.date) {
                if (!bh.start) {
                    bh.start = bh.date
                }
                delete bh.date
            }
            bh._start = O(bh.start = ab(bh.start, bg));
            bh.end = ab(bh.end, bg);
            if (bh.end && bh.end <= bh.start) {
                bh.end = null
            }
            bh._end = bh.end ? O(bh.end) : null;
            if (bh.allDay === y) {
                bh.allDay = aI(bi.allDayDefault, aQ.allDayDefault)
            }
            if (bh.className) {
                if (typeof bh.className == "string") {
                    bh.className = bh.className.split(/\s+/)
                }
            } else {
                bh.className = []
            }
        }
        function aX(bi) {
            if (bi.className) {
                if (typeof bi.className == "string") {
                    bi.className = bi.className.split(/\s+/)
                }
            } else {
                bi.className = []
            }
            var bh = aA.sourceNormalizers;
            for (var bg = 0; bg < bh.length; bg++) {
                bh[bg](bi)
            }
        }
        function aT(bh, bg) {
            return bh && bg && aP(bh) == aP(bg)
        }
        function aP(bg) {
            return ((typeof bg == "object") ? (bg.events || bg.url) : "") || bg
        }
    }
    aA.addDays = aC;
    aA.cloneDate = O;
    aA.parseDate = ab;
    aA.parseISO8601 = q;
    aA.parseTime = aG;
    aA.formatDate = C;
    aA.formatDates = m;
    var R = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"], aw = 86400000, al = 3600000, W = 60000;
    function aj(aL, aM, aK) {
        aL.setFullYear(aL.getFullYear() + aM);
        if (!aK) {
            c(aL)
        }
        return aL
    }
    function r(aN, aO, aM) {
        if ( + aN) {
            var aK = aN.getMonth() + aO, aL = O(aN);
            aL.setDate(1);
            aL.setMonth(aK);
            aN.setMonth(aK);
            if (!aM) {
                c(aN)
            }
            while (aN.getMonth() != aL.getMonth()) {
                aN.setDate(aN.getDate() + (aN < aL ? 1 : - 1))
            }
        }
        return aN
    }
    function aC(aN, aO, aM) {
        if ( + aN) {
            var aK = aN.getDate() + aO, aL = O(aN);
            aL.setHours(9);
            aL.setDate(aK);
            aN.setDate(aK);
            if (!aM) {
                c(aN)
            }
            aE(aN, aL)
        }
        return aN
    }
    function aE(aL, aK) {
        if ( + aL) {
            while (aL.getDate() != aK.getDate()) {
                aL.setTime( + aL + (aL < aK ? 1 : - 1) * al)
            }
        }
    }
    function k(aK, aL) {
        aK.setMinutes(aK.getMinutes() + aL);
        return aK
    }
    function c(aK) {
        aK.setHours(0);
        aK.setMinutes(0);
        aK.setSeconds(0);
        aK.setMilliseconds(0);
        return aK
    }
    function O(aK, aL) {
        if (aL) {
            return c(new Date( + aK))
        }
        return new Date( + aK)
    }
    function f() {
        var aK = 0, aL;
        do {
            aL = new Date(1970, aK++, 1)
        }
        while (aL.getHours());
        return aL
    }
    function aH(aK, aL, aM) {
        aL = aL || 1;
        while (!aK.getDay() || (aM && aK.getDay() == 1 ||!aM && aK.getDay() == 6)) {
            aC(aK, aL)
        }
        return aK
    }
    function az(aL, aK) {
        return Math.round((O(aL, true) - O(aK, true)) / aw)
    }
    function A(aL, aN, aK, aM) {
        if (aN !== y && aN != aL.getFullYear()) {
            aL.setDate(1);
            aL.setMonth(0);
            aL.setFullYear(aN)
        }
        if (aK !== y && aK != aL.getMonth()) {
            aL.setDate(1);
            aL.setMonth(aK)
        }
        if (aM !== y) {
            aL.setDate(aM)
        }
    }
    function ab(aL, aK) {
        if (typeof aL == "object") {
            return aL
        }
        if (typeof aL == "number") {
            return new Date(aL * 1000)
        }
        if (typeof aL == "string") {
            if (aL.match(/^\d+(\.\d+)?$/)) {
                return new Date(parseFloat(aL) * 1000)
            }
            if (aK === y) {
                aK = true
            }
            return q(aL, aK) || (aL ? new Date(aL) : null)
        }
        return null
    }
    function q(aO, aL) {
        var aK = aO.match(/^([0-9]{4})(-([0-9]{2})(-([0-9]{2})([T ]([0-9]{2}):([0-9]{2})(:([0-9]{2})(\.([0-9]+))?)?(Z|(([-+])([0-9]{2})(:?([0-9]{2}))?))?)?)?)?$/);
        if (!aK) {
            return null
        }
        var aN = new Date(aK[1], 0, 1);
        if (aL ||!aK[13]) {
            var aM = new Date(aK[1], 0, 1, 9, 0);
            if (aK[3]) {
                aN.setMonth(aK[3] - 1);
                aM.setMonth(aK[3] - 1)
            }
            if (aK[5]) {
                aN.setDate(aK[5]);
                aM.setDate(aK[5])
            }
            aE(aN, aM);
            if (aK[7]) {
                aN.setHours(aK[7])
            }
            if (aK[8]) {
                aN.setMinutes(aK[8])
            }
            if (aK[10]) {
                aN.setSeconds(aK[10])
            }
            if (aK[12]) {
                aN.setMilliseconds(Number("0." + aK[12]) * 1000)
            }
            aE(aN, aM)
        } else {
            aN.setUTCFullYear(aK[1], aK[3] ? aK[3] - 1 : 0, aK[5] || 1);
            aN.setUTCHours(aK[7] || 0, aK[8] || 0, aK[10] || 0, aK[12] ? Number("0." + aK[12]) * 1000 : 0);
            if (aK[14]) {
                var aP = Number(aK[16]) * 60 + (aK[18] ? Number(aK[18]) : 0);
                aP*=aK[15] == "-" ? 1 : - 1;
                aN = new Date( + aN + (aP * 60 * 1000))
            }
        }
        return aN
    }
    function aG(aM) {
        if (typeof aM == "number") {
            return aM * 60
        }
        if (typeof aM == "object") {
            return aM.getHours() * 60 + aM.getMinutes()
        }
        var aK = aM.match(/(\d+)(?::(\d+))?\s*(\w+)?/);
        if (aK) {
            var aL = parseInt(aK[1], 10);
            if (aK[3]) {
                aL%=12;
                if (aK[3].toLowerCase().charAt(0) == "p") {
                    aL += 12
                }
            }
            return aL * 60 + (aK[2] ? parseInt(aK[2], 10) : 0)
        }
    }
    function C(aL, aM, aK) {
        return m(aL, null, aM, aK)
    }
    function m(aW, aV, aU, aX) {
        aX = aX || N;
        var aL = aW, aN = aV, aO, aP = aU.length, aR, aM, aT, aQ = "";
        for (aO = 0; aO < aP; aO++) {
            aR = aU.charAt(aO);
            if (aR == "'") {
                for (aM = aO + 1; aM < aP; aM++) {
                    if (aU.charAt(aM) == "'") {
                        if (aL) {
                            if (aM == aO + 1) {
                                aQ += "'"
                            } else {
                                aQ += aU.substring(aO + 1, aM)
                            }
                            aO = aM
                        }
                        break
                    }
                }
            } else {
                if (aR == "(") {
                    for (aM = aO + 1; aM < aP; aM++) {
                        if (aU.charAt(aM) == ")") {
                            var aK = C(aL, aU.substring(aO + 1, aM), aX);
                            if (parseInt(aK.replace(/\D/, ""), 10)) {
                                aQ += aK
                            }
                            aO = aM;
                            break
                        }
                    }
                } else {
                    if (aR == "[") {
                        for (aM = aO + 1; aM < aP; aM++) {
                            if (aU.charAt(aM) == "]") {
                                var aS = aU.substring(aO + 1, aM);
                                var aK = C(aL, aS, aX);
                                if (aK != C(aN, aS, aX)) {
                                    aQ += aK
                                }
                                aO = aM;
                                break
                            }
                        }
                    } else {
                        if (aR == "{") {
                            aL = aV;
                            aN = aW
                        } else {
                            if (aR == "}") {
                                aL = aW;
                                aN = aV
                            } else {
                                for (aM = aP; aM > aO; aM--) {
                                    if (aT = ay[aU.substring(aO, aM)]) {
                                        if (aL) {
                                            aQ += aT(aL, aX)
                                        }
                                        aO = aM - 1;
                                        break
                                    }
                                }
                                if (aM == aO) {
                                    if (aL) {
                                        aQ += aR
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return aQ
    }
    var ay = {
        s: function(aK) {
            return aK.getSeconds()
        },
        ss: function(aK) {
            return ac(aK.getSeconds())
        },
        m: function(aK) {
            return aK.getMinutes()
        },
        mm: function(aK) {
            return ac(aK.getMinutes())
        },
        h: function(aK) {
            return aK.getHours()%12 || 12
        },
        hh: function(aK) {
            return ac(aK.getHours()%12 || 12)
        },
        H: function(aK) {
            return aK.getHours()
        },
        HH: function(aK) {
            return ac(aK.getHours())
        },
        d: function(aK) {
            return aK.getDate()
        },
        dd: function(aK) {
            return ac(aK.getDate())
        },
        ddd: function(aL, aK) {
            return aK.dayNamesShort[aL.getDay()]
        },
        dddd: function(aL, aK) {
            return aK.dayNames[aL.getDay()]
        },
        M: function(aK) {
            return aK.getMonth() + 1
        },
        MM: function(aK) {
            return ac(aK.getMonth() + 1)
        },
        MMM: function(aL, aK) {
            return aK.monthNamesShort[aL.getMonth()]
        },
        MMMM: function(aL, aK) {
            return aK.monthNames[aL.getMonth()]
        },
        yy: function(aK) {
            return (aK.getFullYear() + "").substring(2)
        },
        yyyy: function(aK) {
            return aK.getFullYear()
        },
        t: function(aK) {
            return aK.getHours() < 12 ? Date.CultureInfo.amDesignator : Date.CultureInfo.pmDesignator
        },
        tt: function(aK) {
            return aK.getHours() < 12 ? Date.CultureInfo.amDesignator : Date.CultureInfo.pmDesignator
        },
        T: function(aK) {
            return aK.getHours() < 12 ? Date.CultureInfo.amDesignator : Date.CultureInfo.pmDesignator
        },
        TT: function(aK) {
            return aK.getHours() < 12 ? Date.CultureInfo.amDesignator : Date.CultureInfo.pmDesignator
        },
        u: function(aK) {
            return C(aK, "yyyy-MM-dd'T'HH:mm:ss'Z'")
        },
        S: function(aL) {
            var aK = aL.getDate();
            if (aK > 10 && aK < 20) {
                return "th"
            }
            return ["st", "nd", "rd"][aK%10 - 1] || "th"
        }
    };
    aA.applyAll = J;
    function ao(aK) {
        if (aK.end) {
            return t(aK.end, aK.allDay)
        } else {
            return aC(O(aK.start), 1)
        }
    }
    function t(aK, aL) {
        aK = O(aK);
        return aL || aK.getHours() || aK.getMinutes() ? aC(aK, 1) : c(aK)
    }
    function w(aL, aK) {
        return (aK.msLength - aL.msLength) * 100 + (aL.event.start - aK.event.start)
    }
    function P(aL, aK) {
        return aL.end > aK.start && aL.start < aK.end
    }
    function ad(aW, aQ, aM, aP) {
        var aN = [], aR, aT = aW.length, aL, aU, aS, aV, aX, aK, aO;
        for (aR = 0; aR < aT; aR++) {
            aL = aW[aR];
            aU = aL.start;
            aS = aQ[aR];
            if (aS > aM && aU < aP) {
                if (aU < aM) {
                    aV = O(aM);
                    aK = false
                } else {
                    aV = aU;
                    aK = true
                }
                if (aS > aP) {
                    aX = O(aP);
                    aO = false
                } else {
                    aX = aS;
                    aO = true
                }
                aN.push({
                    event: aL,
                    start: aV,
                    end: aX,
                    isStart: aK,
                    isEnd: aO,
                    msLength: aX - aV
                })
            }
        }
        return aN.sort(w)
    }
    function ai(aM) {
        var aQ = [], aP, aK = aM.length, aL, aO, aR, aN;
        for (aP = 0; aP < aK; aP++) {
            aL = aM[aP];
            aO = 0;
            while (true) {
                aR = false;
                if (aQ[aO]) {
                    for (aN = 0; aN < aQ[aO].length; aN++) {
                        if (P(aQ[aO][aN], aL)) {
                            aR = true;
                            break
                        }
                    }
                }
                if (aR) {
                    aO++
                } else {
                    break
                }
            }
            if (aQ[aO]) {
                aQ[aO].push(aL)
            } else {
                aQ[aO] = [aL]
            }
        }
        return aQ
    }
    function x(aL, aK, aM) {
        aL.unbind("mouseover").mouseover(function(aQ) {
            var aP = aQ.target, aR, aO, aN;
            while (aP != this) {
                aR = aP;
                aP = aP.parentNode
            }
            if ((aO = aR._fci) !== y) {
                aR._fci = y;
                aN = aK[aO];
                aM(aN.event, aN.element, aN);
                au(aQ.target).trigger(aQ)
            }
            aQ.stopPropagation()
        })
    }
    function av(aM, aN, aK) {
        for (var aL = 0, aO; aL < aM.length; aL++) {
            aO = au(aM[aL]);
            aO.width(Math.max(0, aN - j(aO, aK)))
        }
    }
    function aa(aN, aK, aL) {
        for (var aM = 0, aO; aM < aN.length; aM++) {
            aO = au(aN[aM]);
            aO.height(Math.max(0, aK - G(aO, aL)))
        }
    }
    function j(aL, aK) {
        return ah(aL) + s(aL) + (aK ? ag(aL) : 0)
    }
    function ah(aK) {
        return (parseFloat(au.css(aK[0], "paddingLeft", true)) || 0) + (parseFloat(au.css(aK[0], "paddingRight", true)) || 0)
    }
    function ag(aK) {
        return (parseFloat(au.css(aK[0], "marginLeft", true)) || 0) + (parseFloat(au.css(aK[0], "marginRight", true)) || 0)
    }
    function s(aK) {
        return (parseFloat(au.css(aK[0], "borderLeftWidth", true)) || 0) + (parseFloat(au.css(aK[0], "borderRightWidth", true)) || 0)
    }
    function G(aL, aK) {
        return z(aL) + aq(aL) + (aK ? n(aL) : 0)
    }
    function z(aK) {
        return (parseFloat(au.css(aK[0], "paddingTop", true)) || 0) + (parseFloat(au.css(aK[0], "paddingBottom", true)) || 0)
    }
    function n(aK) {
        return (parseFloat(au.css(aK[0], "marginTop", true)) || 0) + (parseFloat(au.css(aK[0], "marginBottom", true)) || 0)
    }
    function aq(aK) {
        return (parseFloat(au.css(aK[0], "borderTopWidth", true)) || 0) + (parseFloat(au.css(aK[0], "borderBottomWidth", true)) || 0)
    }
    function K(aL, aK) {
        aK = (typeof aK == "number" ? aK + "px" : aK);
        aL.each(function(aN, aM) {
            aM.style.cssText += ";min-height:" + aK + ";_height:" + aK
        })
    }
    function ar() {}
    function Y(aL, aK) {
        return aL - aK
    }
    function at(aK) {
        return Math.max.apply(Math, aK)
    }
    function ac(aK) {
        return (aK < 10 ? "0" : "") + aK
    }
    function F(aO, aK) {
        if (aO[aK] !== y) {
            return aO[aK]
        }
        var aN = aK.split(/(?=[A-Z])/), aM = aN.length - 1, aL;
        for (; aM >= 0; aM--) {
            aL = aO[aN[aM].toLowerCase()];
            if (aL !== y) {
                return aL
            }
        }
        return aO[""]
    }
    function aF(aK) {
        return aK.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/'/g, "&#039;").replace(/"/g, "&quot;").replace(/\n/g, "<br />")
    }
    function I(aK) {
        return aK.id + "/" + aK.className + "/" + aK.style.cssText.replace(/(^|;)\s*(top|left|width|height)\s*:[^;]*/ig, "")
    }
    function aJ(aK) {
        aK.attr("unselectable", "on").css("MozUserSelect", "none").bind("selectstart.ui", function() {
            return false
        })
    }
    function e(aK) {
        aK.children().removeClass("fc-first fc-last").filter(":first-child").addClass("fc-first").end().filter(":last-child").addClass("fc-last")
    }
    function M(aK, aL) {
        aK.each(function(aN, aO) {
            aO.className = aO.className.replace(/(weekday|weekend)/g, "");
            var aM = (R[aL.getDay()] == "sat" || R[aL.getDay()] == "sun") ? "weekend": "weekday";
            aO.className = aO.className.replace(/^fc-\w*/, "fc-" + R[aL.getDay()] + " " + aM)
        })
    }
    function S(aL, aM) {
        var aK = aL.source || {};
        var aR = aL.color;
        var aP = aK.color;
        var aO = aM("eventColor");
        var aS = aL.backgroundColor || aR || aK.backgroundColor || aP || aM("eventBackgroundColor") || aO;
        var aN = aL.borderColor || aR || aK.borderColor || aP || aM("eventBorderColor") || aO;
        var aT = aL.textColor || aK.textColor || aM("eventTextColor");
        var aQ = [];
        if (aS) {
            aQ.push("background-color:" + aS)
        }
        if (aN) {
            aQ.push("border-color:" + aN)
        }
        if (aT) {
            aQ.push("color:" + aT)
        }
        return aQ.join(";")
    }
    function J(aN, aO, aL) {
        if (au.isFunction(aN)) {
            aN = [aN]
        }
        if (aN) {
            var aM;
            var aK;
            for (aM = 0; aM < aN.length; aM++) {
                aK = aN[aM].apply(aO, aL) || aK
            }
            return aK
        }
    }
    function aI() {
        for (var aK = 0; aK < arguments.length; aK++) {
            if (arguments[aK] !== y) {
                return arguments[aK]
            }
        }
    }
    af.month = ae;
    function ae(aN, aQ) {
        var aM = this;
        aM.render = aO;
        B.call(aM, aN, aQ, "month");
        var aL = aM.opt;
        var aK = aM.renderBasic;
        var aP = aQ.formatDate;
        function aO(aU, aY) {
            if (aY) {
                r(aU, aY);
                aU.setDate(1)
            }
            var aS = O(aU, true);
            aS.setDate(1);
            var aV = r(O(aS), 1);
            var aZ = O(aS);
            var aW = O(aV);
            var aR = aL("firstDay");
            var aT = aL("weekends") ? 0: 1;
            if (aT) {
                aH(aZ);
                aH(aW, - 1, true)
            }
            aC(aZ, - ((aZ.getDay() - Math.max(aR, aT) + 7)%7));
            aC(aW, (7 - aW.getDay() + Math.max(aR, aT))%7);
            var aX = Math.round((aW - aZ) / (aw * 7));
            if (aL("weekMode") == "fixed") {
                aC(aW, (6 - aX) * 7);
                aX = 6
            }
            aM.title = aP(aS, aL("titleFormat"));
            aM.start = aS;
            aM.end = aV;
            aM.visStart = aZ;
            aM.visEnd = aW;
            aK(6, aX, aT ? 5 : 7, true)
        }
    }
    af.basicWeek = aB;
    function aB(aO, aQ) {
        var aN = this;
        aN.render = aP;
        B.call(aN, aO, aQ, "basicWeek");
        var aM = aN.opt;
        var aL = aN.renderBasic;
        var aK = aQ.formatDates;
        function aP(aV, aX) {
            if (aX) {
                aC(aV, aX * 7)
            }
            var aW = aC(O(aV), - ((aV.getDay() - aM("firstDay") + 7)%7));
            var aT = aC(O(aW), 7);
            var aS = O(aW);
            var aR = O(aT);
            var aU = aM("weekends");
            if (!aU) {
                aH(aS);
                aH(aR, - 1, true)
            }
            aN.title = aK(aS, aC(O(aR), - 1), aM("titleFormat"));
            aN.start = aW;
            aN.end = aT;
            aN.visStart = aS;
            aN.visEnd = aR;
            aL(1, 1, aU ? 7 : 5, false)
        }
    }
    af.basicDay = L;
    function L(aN, aQ) {
        var aM = this;
        aM.render = aO;
        B.call(aM, aN, aQ, "basicDay");
        var aL = aM.opt;
        var aK = aM.renderBasic;
        var aP = aQ.formatDate;
        function aO(aR, aS) {
            if (aS) {
                aC(aR, aS);
                if (!aL("weekends")) {
                    aH(aR, aS < 0?-1 : 1)
                }
            }
            aM.title = aP(aR, aL("titleFormat"));
            aM.start = aM.visStart = O(aR, true);
            aM.end = aM.visEnd = aC(O(aM.start), 1);
            aK(1, 1, 1, false)
        }
    }
    ak({
        weekMode: "fixed"
    });
    function B(a0, bq, aT) {
        var bp = this;
        bp.renderBasic = bi;
        bp.setHeight = bn;
        bp.setWidth = bx;
        bp.renderDayOverlay = bd;
        bp.defaultSelectionEnd = bc;
        bp.renderSelection = br;
        bp.clearSelection = aK;
        bp.reportDayClick = aY;
        bp.dragStart = aM;
        bp.dragStop = aU;
        bp.defaultEventEnd = bE;
        bp.getHoverListener = function() {
            return aN
        };
        bp.colContentLeft = bm;
        bp.colContentRight = bj;
        bp.dayOfWeekCol = a8;
        bp.dateCell = bk;
        bp.cellDate = bD;
        bp.cellIsAllDay = function() {
            return true
        };
        bp.allDayRow = aX;
        bp.allDayBounds = bu;
        bp.getRowCnt = function() {
            return a6
        };
        bp.getColCnt = function() {
            return aP
        };
        bp.getColWidth = function() {
            return a9
        };
        bp.getDaySegmentContainer = function() {
            return aW
        };
        aD.call(bp, a0, bq, aT);
        b.call(bp);
        an.call(bp);
        am.call(bp);
        var a3 = bp.opt;
        var bf = bp.trigger;
        var bb = bp.clearEvents;
        var bA = bp.renderOverlay;
        var aS = bp.clearOverlays;
        var bl = bp.daySelectionMousedown;
        var bo = bq.formatDate;
        var aZ;
        var aV;
        var a7;
        var aR;
        var a1;
        var ba;
        var bh;
        var aW;
        var bB;
        var bG;
        var a9;
        var a6, aP;
        var bt;
        var aN;
        var bv;
        var bs, bF, bC;
        var a5;
        var a2;
        var bH;
        var aQ;
        aJ(a0.addClass("fc-grid"));
        function bi(bJ, bK, bM, bI) {
            a6 = bK;
            aP = bM;
            by();
            var bL=!a7;
            if (bL) {
                be(bJ, bI)
            } else {
                bb()
            }
            aL(bL)
        }
        function by() {
            bs = a3("isRTL");
            if (bs) {
                bF =- 1;
                bC = aP - 1
            } else {
                bF = 1;
                bC = 0
            }
            a5 = a3("firstDay");
            a2 = a3("weekends") ? 0 : 1;
            bH = a3("theme") ? "ui" : "fc";
            aQ = a3("columnFormat")
        }
        function be(bO, bL) {
            var bN;
            var bM = bH + "-widget-header";
            var bJ = bH + "-widget-content";
            var bK, bI;
            var bP;
            bN = "<table class='fc-border-separate' style='width:100%' cellspacing='0'><thead><tr>";
            for (bK = 0; bK < aP; bK++) {
                bN += "<th class='fc- " + bM + "'/>"
            }
            bN += "</tr></thead><tbody>";
            for (bK = 0; bK < bO; bK++) {
                bN += "<tr class='fc-week" + bK + "'>";
                for (bI = 0; bI < aP; bI++) {
                    bN += "<td class='fc- " + bJ + " fc-day" + (bK * aP + bI) + "'><div>" + (bL ? "<div class='fc-day-number'/>" : "") + "<div class='fc-day-content'><div style='position:relative'>&nbsp;</div></div></div></td>"
                }
                bN += "</tr>"
            }
            bN += "</tbody></table>";
            bP = au(bN).appendTo(a0);
            aZ = bP.find("thead");
            aV = aZ.find("th");
            a7 = bP.find("tbody");
            aR = a7.find("tr");
            a1 = a7.find("td");
            ba = a1.filter(":first-child");
            bh = aR.eq(0).find("div.fc-day-content div");
            e(aZ.add(aZ.find("tr")));
            e(aR);
            aR.eq(0).addClass("fc-first");
            aO(a1);
            aW = au("<div style='position:absolute;z-index:8;top:0;left:0'/>").appendTo(a0)
        }
        function aL(bO) {
            var bL = bO || a6 == 1;
            var bM = bp.start.getMonth();
            var bJ = c(new Date());
            var bI;
            var bK;
            var bN;
            if (bL) {
                aV.each(function(bP, bQ) {
                    bI = au(bQ);
                    bK = bg(bP);
                    bI.html(bo(bK, aQ));
                    M(bI, bK)
                })
            }
            a1.each(function(bP, bQ) {
                bI = au(bQ);
                bK = bg(bP);
                if (bK.getMonth() == bM) {
                    bI.removeClass("fc-other-month")
                } else {
                    bI.addClass("fc-other-month")
                }
                if ( + bK ==+ bJ) {
                    bI.addClass(bH + "-state-highlight fc-today");
                    bI.removeClass("fc-not-today")
                } else {
                    bI.removeClass(bH + "-state-highlight fc-today");
                    bI.addClass("fc-not-today")
                }
                bI.find("div.fc-day-number").text(bK.getDate());
                if (bL) {
                    M(bI, bK)
                }
            });
            aR.each(function(bP, bQ) {
                bN = au(bQ);
                if (bP < a6) {
                    bN.show();
                    if (bP == a6 - 1) {
                        bN.addClass("fc-last")
                    } else {
                        bN.removeClass("fc-last")
                    }
                } else {
                    bN.hide()
                }
            })
        }
        function bn(bJ) {
            bG = bJ;
            var bM = bG - aZ.height();
            var bL;
            var bK;
            var bI;
            if (a3("weekMode") == "variable") {
                bL = bK = Math.floor(bM / (a6 == 1 ? 2 : 6))
            } else {
                bL = Math.floor(bM / a6);
                bK = bM - bL * (a6 - 1)
            }
            ba.each(function(bN, bO) {
                if (bN < a6) {
                    bI = au(bO);
                    K(bI.find("> div"), (bN == a6 - 1 ? bK : bL) - G(bI))
                }
            })
        }
        function bx(bI) {
            bB = bI;
            bv.clear();
            a9 = Math.floor(bB / aP);
            av(aV.slice(0, - 1), a9)
        }
        function aO(bI) {
            bI.click(bz).mousedown(bl)
        }
        function bz(bK) {
            if (!a3("selectable")) {
                var bJ = parseInt(this.className.match(/fc\-day(\d+)/)[1]);
                var bI = bg(bJ);
                bf("dayClick", this, bI, true, bK)
            }
        }
        function bd(bN, bR, bL) {
            if (bL) {
                bt.build()
            }
            var bI = O(bp.visStart);
            var bO = aC(O(bI), aP);
            for (var bK = 0; bK < a6; bK++) {
                var bM = new Date(Math.max(bI, bN));
                var bQ = new Date(Math.min(bO, bR));
                if (bM < bQ) {
                    var bJ, bP;
                    if (bs) {
                        bJ = az(bQ, bI) * bF + bC + 1;
                        bP = az(bM, bI) * bF + bC + 1
                    } else {
                        bJ = az(bM, bI);
                        bP = az(bQ, bI)
                    }
                    aO(bw(bK, bJ, bK, bP - 1))
                }
                aC(bI, 7);
                aC(bO, 7)
            }
        }
        function bw(bL, bM, bJ, bK) {
            var bI = bt.rect(bL, bM, bJ, bK, a0);
            return bA(bI, a0)
        }
        function bc(bI, bJ) {
            return O(bI)
        }
        function br(bI, bK, bJ) {
            bd(bI, aC(O(bK), 1), true)
        }
        function aK() {
            aS()
        }
        function aY(bK, bM, bL) {
            var bI = bk(bK);
            var bJ = a1[bI.row * aP + bI.col];
            bf("dayClick", bJ, bK, bM, bL)
        }
        function aM(bK, bI, bJ) {
            aN.start(function(bL) {
                aS();
                if (bL) {
                    bw(bL.row, bL.col, bL.row, bL.col)
                }
            }, bI)
        }
        function aU(bM, bJ, bK) {
            var bI = aN.stop();
            aS();
            if (bI) {
                var bL = bD(bI);
                bf("drop", bM, bL, true, bJ, bK)
            }
        }
        function bE(bI) {
            return O(bI.start)
        }
        bt = new Q(function(bI, bL) {
            var bK, bM, bJ;
            aV.each(function(bO, bN) {
                bK = au(bN);
                bM = bK.offset().left;
                if (bO) {
                    bJ[1] = bM
                }
                bJ = [bM];
                bL[bO] = bJ
            });
            bJ[1] = bM + bK.outerWidth();
            aR.each(function(bO, bN) {
                if (bO < a6) {
                    bK = au(bN);
                    bM = bK.offset().top;
                    if (bO) {
                        bJ[1] = bM
                    }
                    bJ = [bM];
                    bI[bO] = bJ
                }
            });
            bJ[1] = bM + bK.outerHeight()
        });
        aN = new ap(bt);
        bv = new o(function(bI) {
            return bh.eq(bI)
        });
        function bm(bI) {
            return bv.left(bI)
        }
        function bj(bI) {
            return bv.right(bI)
        }
        function bk(bI) {
            return {
                row: Math.floor(az(bI, bp.visStart) / 7),
                col: a8(bI.getDay())
            }
        }
        function bD(bI) {
            return a4(bI.row, bI.col)
        }
        function a4(bJ, bI) {
            return aC(O(bp.visStart), bJ * 7 + bI * bF + bC)
        }
        function bg(bI) {
            return a4(Math.floor(bI / aP), bI%aP)
        }
        function a8(bI) {
            return ((bI - Math.max(a5, a2) + aP)%aP) * bF + bC
        }
        function aX(bI) {
            return aR.eq(bI)
        }
        function bu(bI) {
            return {
                left: 0,
                right: bB
            }
        }
    }
    function am() {
        var aV = this;
        aV.renderEvents = aQ;
        aV.compileDaySegs = aU;
        aV.clearEvents = aT;
        aV.bindDaySeg = aP;
        Z.call(aV);
        var aM = aV.opt;
        var aX = aV.trigger;
        var a1 = aV.isEventDraggable;
        var a3 = aV.isEventResizable;
        var a2 = aV.reportEvents;
        var a6 = aV.reportEventClear;
        var aL = aV.eventElementHandlers;
        var a4 = aV.showEvents;
        var aK = aV.hideEvents;
        var aZ = aV.eventDrop;
        var a7 = aV.getDaySegmentContainer;
        var a5 = aV.getHoverListener;
        var aS = aV.renderDayOverlay;
        var aY = aV.clearOverlays;
        var aO = aV.getRowCnt;
        var a0 = aV.getColCnt;
        var aW = aV.renderDaySegs;
        var aR = aV.resizableDayEvent;
        function aQ(a9, a8) {
            a2(a9);
            aW(aU(a9), a8)
        }
        function aT() {
            a6();
            a7().empty()
        }
        function aU(bj) {
            var bg = aO(), bi = a0(), ba = O(aV.visStart), a8 = aC(O(ba), bi), bh = au.map(bj, ao), bf, bk, be, a9, bc, bd, bb = [];
            for (bf = 0; bf < bg; bf++) {
                bk = ai(ad(bj, bh, ba, a8));
                for (be = 0; be < bk.length; be++) {
                    a9 = bk[be];
                    for (bc = 0; bc < a9.length; bc++) {
                        bd = a9[bc];
                        bd.row = bf;
                        bd.level = be;
                        bb.push(bd)
                    }
                }
                aC(ba, 7);
                aC(a8, 7)
            }
            return bb
        }
        function aP(ba, a9, a8) {
            if (a1(ba)) {
                aN(ba, a9)
            }
            if (a8.isEnd && a3(ba)) {
                aR(ba, a9, a8)
            }
            aL(ba, a9)
        }
        function aN(bb, ba) {
            var a9 = a5();
            var a8;
            ba.draggable({
                zIndex: 9,
                delay: 50,
                opacity: aM("dragOpacity"),
                revertDuration: aM("dragRevertDuration"),
                start: function(bc, bd) {
                    aX("eventDragStart", ba, bb, bc, bd);
                    aK(bb, ba);
                    a9.start(function(bf, be, bg, bh) {
                        ba.draggable("option", "revert", !bf ||!bg&&!bh);
                        aY();
                        if (bf) {
                            a8 = bg * 7 + bh * (aM("isRTL")?-1 : 1);
                            aS(aC(O(bb.start), a8), aC(ao(bb), a8))
                        } else {
                            a8 = 0
                        }
                    }, bc, "drag")
                },
                stop: function(bc, bd) {
                    a9.stop();
                    aY();
                    aX("eventDragStop", ba, bb, bc, bd);
                    if (a8) {
                        aZ(this, bb, a8, 0, bb.allDay, bc, bd)
                    } else {
                        ba.css("filter", "");
                        a4(bb, ba)
                    }
                }
            })
        }
    }
    af.agendaWeek = g;
    function g(aO, aQ) {
        var aN = this;
        aN.render = aP;
        H.call(aN, aO, aQ, "agendaWeek");
        var aM = aN.opt;
        var aL = aN.renderAgenda;
        var aK = aQ.formatDates;
        function aP(aV, aX) {
            if (aX) {
                aC(aV, aX * 7)
            }
            var aW = aC(O(aV), - ((aV.getDay() - aM("firstDay") + 7)%7));
            var aT = aC(O(aW), 7);
            var aS = O(aW);
            var aR = O(aT);
            var aU = aM("weekends");
            if (!aU) {
                aH(aS);
                aH(aR, - 1, true)
            }
            aN.title = aK(aS, aC(O(aR), - 1), aM("titleFormat"));
            aN.start = aW;
            aN.end = aT;
            aN.visStart = aS;
            aN.visEnd = aR;
            aL(aU ? 7 : 5)
        }
    }
    af.agendaDay = ax;
    function ax(aN, aQ) {
        var aM = this;
        aM.render = aO;
        H.call(aM, aN, aQ, "agendaDay");
        var aL = aM.opt;
        var aK = aM.renderAgenda;
        var aP = aQ.formatDate;
        function aO(aS, aU) {
            if (aU) {
                aC(aS, aU);
                if (!aL("weekends")) {
                    aH(aS, aU < 0?-1 : 1)
                }
            }
            var aT = O(aS, true);
            var aR = aC(O(aT), 1);
            aM.title = aP(aS, aL("titleFormat"));
            aM.start = aM.visStart = aT;
            aM.end = aM.visEnd = aR;
            aK(1)
        }
    }
    ak({
        allDaySlot: true,
        allDayText: "all-day",
        firstHour: 6,
        slotMinutes: 30,
        defaultEventMinutes: 120,
        axisFormat: "h(:mm)tt",
        timeFormat: {
            agenda: "h:mm{ - h:mm}"
        },
        dragOpacity: {
            agenda: 0.5
        },
        minTime: 0,
        maxTime: 24
    });
    function H(b7, a2, bf) {
        var bb = this;
        bb.renderAgenda = aW;
        bb.setWidth = bq;
        bb.setHeight = bm;
        bb.beforeHide = bk;
        bb.afterShow = bY;
        bb.defaultEventEnd = bS;
        bb.timePosition = ca;
        bb.dayOfWeekCol = aX;
        bb.dateCell = b3;
        bb.cellDate = aO;
        bb.cellIsAllDay = b2;
        bb.allDayRow = bF;
        bb.allDayBounds = bu;
        bb.getHoverListener = function() {
            return b6
        };
        bb.colContentLeft = cb;
        bb.colContentRight = aT;
        bb.getDaySegmentContainer = function() {
            return a7
        };
        bb.getSlotSegmentContainer = function() {
            return aR
        };
        bb.getMinMinute = function() {
            return bi
        };
        bb.getMaxMinute = function() {
            return aM
        };
        bb.getBodyContent = function() {
            return bE
        };
        bb.getRowCnt = function() {
            return 1
        };
        bb.getColCnt = function() {
            return bH
        };
        bb.getColWidth = function() {
            return aU
        };
        bb.getSlotHeight = function() {
            return bI
        };
        bb.defaultSelectionEnd = bp;
        bb.renderDayOverlay = aP;
        bb.renderSelection = b1;
        bb.clearSelection = bV;
        bb.reportDayClick = ba;
        bb.dragStart = aV;
        bb.dragStop = b8;
        aD.call(bb, b7, a2, bf);
        b.call(bb);
        an.call(bb);
        V.call(bb);
        var bD = bb.opt;
        var bG = bb.trigger;
        var bC = bb.clearEvents;
        var bt = bb.renderOverlay;
        var aN = bb.clearOverlays;
        var bv = bb.reportSelection;
        var bZ = bb.unselect;
        var bz = bb.daySelectionMousedown;
        var a3 = bb.slotSegHtml;
        var a8 = a2.formatDate;
        var bQ;
        var b4;
        var bU;
        var bT;
        var be;
        var aQ;
        var bR;
        var bO;
        var b9;
        var a7;
        var b5;
        var bX;
        var br;
        var bE;
        var aR;
        var aY;
        var bN;
        var bj;
        var a9;
        var aS;
        var bw;
        var bL;
        var bn;
        var aU;
        var bA;
        var bI;
        var aK;
        var bH;
        var bM;
        var bJ;
        var b6;
        var bx;
        var bg = {};
        var bs;
        var bl;
        var bB;
        var aL, a5, a4;
        var bi, aM;
        var a0;
        aJ(b7.addClass("fc-agenda"));
        function aW(cc) {
            bH = cc;
            bK();
            if (!bQ) {
                bd()
            } else {
                bC()
            }
            a1()
        }
        function bK() {
            bs = bD("theme") ? "ui" : "fc";
            bB = bD("weekends") ? 0 : 1;
            bl = bD("firstDay");
            if (aL = bD("isRTL")) {
                a5 =- 1;
                a4 = bH - 1
            } else {
                a5 = 1;
                a4 = 0
            }
            bi = aG(bD("minTime"));
            aM = aG(bD("maxTime"));
            a0 = bD("columnFormat")
        }
        function bd() {
            var ch = bs + "-widget-header";
            var cc = bs + "-widget-content";
            var cg;
            var cf;
            var cj;
            var ci;
            var ce;
            var cd = bD("slotMinutes")%15 == 0;
            cg = "<table style='width:100%' class='fc-agenda-days fc-border-separate' cellspacing='0'><thead><tr><th class='fc-agenda-axis " + ch + "'>&nbsp;</th>";
            for (cf = 0; cf < bH; cf++) {
                cg += "<th class='fc- fc-col" + cf + " " + ch + "'/>"
            }
            cg += "<th class='fc-agenda-gutter " + ch + "'>&nbsp;</th></tr></thead><tbody><tr><th class='fc-agenda-axis " + ch + "'>&nbsp;</th>";
            for (cf = 0; cf < bH; cf++) {
                cg += "<td class='fc- fc-col" + cf + " " + cc + "'><div><div class='fc-day-content'><div style='position:relative'>&nbsp;</div></div></div></td>"
            }
            cg += "<td class='fc-agenda-gutter " + cc + "'>&nbsp;</td></tr></tbody></table>";
            bQ = au(cg).appendTo(b7);
            b4 = bQ.find("thead");
            bU = b4.find("th").slice(1, - 1);
            bT = bQ.find("tbody");
            be = bT.find("td").slice(0, - 1);
            aQ = be.find("div.fc-day-content div");
            bR = be.eq(0);
            bO = bR.find("> div");
            e(b4.add(b4.find("tr")));
            e(bT.add(bT.find("tr")));
            bj = b4.find("th:first");
            a9 = bQ.find(".fc-agenda-gutter");
            b9 = au("<div style='position:absolute;z-index:2;left:0;width:100%'/>").appendTo(b7);
            if (bD("allDaySlot")) {
                a7 = au("<div style='position:absolute;z-index:8;top:0;left:0'/>").appendTo(b9);
                cg = "<table style='width:100%' class='fc-agenda-allday' cellspacing='0'><tr><th class='" + ch + " fc-agenda-axis'>" + bD("allDayText") + "</th><td><div class='fc-day-content'><div style='position:relative'/></div></td><th class='" + ch + " fc-agenda-gutter'>&nbsp;</th></tr></table>";
                b5 = au(cg).appendTo(b9);
                bX = b5.find("tr");
                bo(bX.find("td"));
                bj = bj.add(b5.find("th:first"));
                a9 = a9.add(b5.find("th.fc-agenda-gutter"));
                b9.append("<div class='fc-agenda-divider " + ch + "'><div class='fc-agenda-divider-inner'/></div>")
            } else {
                a7 = au([])
            }
            br = au("<div style='position:absolute;width:100%;overflow-x:hidden;overflow-y:auto'/>").appendTo(b9);
            bE = au("<div style='position:relative;width:100%;overflow:hidden'/>").appendTo(br);
            aR = au("<div style='position:absolute;z-index:8;top:0;left:0'/>").appendTo(bE);
            cg = "<table class='fc-agenda-slots' style='width:100%' cellspacing='0'><tbody>";
            cj = f();
            ci = k(O(cj), aM);
            k(cj, bi);
            bM = 0;
            for (cf = 0; cj < ci; cf++) {
                ce = cj.getMinutes();
                cg += "<tr class='fc-slot" + cf + " " + (!ce ? "" : "fc-minor") + "'><th class='fc-agenda-axis " + ch + "'>" + ((!cd ||!ce) ? a8(cj, bD("axisFormat")) : "&nbsp;") + "</th><td class='" + cc + "'><div style='position:relative'>&nbsp;</div></td></tr>";
                k(cj, bD("slotMinutes"));
                bM++
            }
            cg += "</tbody></table>";
            aY = au(cg).appendTo(bE);
            bN = aY.find("div:first");
            bW(aY.find("td"));
            bj = bj.add(aY.find("th:first"))
        }
        function a1() {
            var cf;
            var cc;
            var cg;
            var ce;
            var cd = c(new Date());
            for (cf = 0; cf < bH; cf++) {
                ce = by(cf);
                cc = bU.eq(cf);
                cc.html(a8(ce, a0));
                cg = be.eq(cf);
                if ( + ce ==+ cd) {
                    cg.addClass(bs + "-state-highlight fc-today");
                    cg.removeClass("fc-not-today")
                } else {
                    cg.removeClass(bs + "-state-highlight fc-today");
                    cg.addClass("fc-not-today")
                }
                M(cc.add(cg), ce)
            }
        }
        function bm(cc, cd) {
            if (cc === y) {
                cc = bL
            }
            bL = cc;
            bg = {};
            var cf = bT.position().top;
            var ce = br.position().top;
            var cg = Math.min(cc - cf, aY.height() + ce + 1);
            bO.height(cg - G(bR));
            b9.css("top", cf);
            br.height(cg - ce - 1);
            bI = bN.height() + 1;
            if (cd) {
                bh()
            }
        }
        function bq(cd) {
            bw = cd;
            bx.clear();
            bn = 0;
            av(bj.width("").each(function(ce, cf) {
                bn = Math.max(bn, au(cf).outerWidth())
            }), bn);
            var cc = br[0].clientWidth;
            bA = br.width() - cc;
            if (bA) {
                av(a9, bA);
                a9.show().prev().removeClass("fc-last")
            } else {
                a9.hide().prev().addClass("fc-last")
            }
            aU = Math.floor((cc - bn) / bH);
            av(bU.slice(0, - 1), aU)
        }
        function bh() {
            var cf = f();
            var cd = O(cf);
            cd.setHours(bD("firstHour"));
            var ce = ca(cf, cd) + 1;
            function cc() {
                br.scrollTop(ce)
            }
            cc();
            setTimeout(cc, 0)
        }
        function bk() {
            aK = br.scrollTop()
        }
        function bY() {
            br.scrollTop(aK)
        }
        function bo(cc) {
            cc.click(bP).mousedown(bz)
        }
        function bW(cc) {
            cc.click(bP).mousedown(aZ)
        }
        function bP(cg) {
            if (!bD("selectable")) {
                var ce = Math.min(bH - 1, Math.floor((cg.pageX - bQ.offset().left - bn) / aU));
                var cd = by(ce);
                var ch = this.parentNode.className.match(/fc-slot(\d+)/);
                if (ch) {
                    var cf = parseInt(ch[1]) * bD("slotMinutes");
                    var cc = Math.floor(cf / 60);
                    cd.setHours(cc);
                    cd.setMinutes(cf%60 + bi);
                    bG("dayClick", be[ce], cd, false, cg)
                } else {
                    bG("dayClick", be[ce], cd, true, cg)
                }
            }
        }
        function aP(cc, ch, ce) {
            if (ce) {
                bJ.build()
            }
            var cd = O(bb.visStart);
            var cg, cf;
            if (aL) {
                cg = az(ch, cd) * a5 + a4 + 1;
                cf = az(cc, cd) * a5 + a4 + 1
            } else {
                cg = az(cc, cd);
                cf = az(ch, cd)
            }
            cg = Math.max(0, cg);
            cf = Math.min(bH, cf);
            if (cg < cf) {
                bo(a6(0, cg, 0, cf - 1))
            }
        }
        function a6(cf, cg, cd, ce) {
            var cc = bJ.rect(cf, cg, cd, ce, b9);
            return bt(cc, b9)
        }
        function bc(ch, cm) {
            var ci = O(bb.visStart);
            var cd = aC(O(ci), 1);
            for (var cf = 0; cf < bH; cf++) {
                var cg = new Date(Math.max(ci, ch));
                var cl = new Date(Math.min(cd, cm));
                if (cg < cl) {
                    var ce = cf * a5 + a4;
                    var ck = bJ.rect(0, ce, 0, ce, bE);
                    var cj = ca(ci, cg);
                    var cc = ca(ci, cl);
                    ck.top = cj;
                    ck.height = cc - cj;
                    bW(bt(ck, bE))
                }
                aC(ci, 1);
                aC(cd, 1)
            }
        }
        bJ = new Q(function(ck, ci) {
            var cg, cd, cc;
            bU.each(function(cn, cm) {
                cg = au(cm);
                cd = cg.offset().left;
                if (cn) {
                    cc[1] = cd
                }
                cc = [cd];
                ci[cn] = cc
            });
            cc[1] = cd + cg.outerWidth();
            if (bD("allDaySlot")) {
                cg = bX;
                cd = cg.offset().top;
                ck[0] = [cd, cd + cg.outerHeight()]
            }
            var cj = bE.offset().top;
            var cl = br.offset().top;
            var cf = cl + br.outerHeight();
            function ch(cm) {
                return Math.max(cl, Math.min(cf, cm))
            }
            for (var ce = 0; ce < bM; ce++) {
                ck.push([ch(cj + bI * ce), ch(cj + bI * (ce + 1))])
            }
        });
        b6 = new ap(bJ);
        bx = new o(function(cc) {
            return aQ.eq(cc)
        });
        function cb(cc) {
            return bx.left(cc)
        }
        function aT(cc) {
            return bx.right(cc)
        }
        function b3(cc) {
            return {
                row: Math.floor(az(cc, bb.visStart) / 7),
                col: aX(cc.getDay())
            }
        }
        function aO(cc) {
            var ce = by(cc.col);
            var cd = cc.row;
            if (bD("allDaySlot")) {
                cd--
            }
            if (cd >= 0) {
                k(ce, bi + cd * bD("slotMinutes"))
            }
            return ce
        }
        function by(cc) {
            return aC(O(bb.visStart), cc * a5 + a4)
        }
        function b2(cc) {
            return bD("allDaySlot")&&!cc.row
        }
        function aX(cc) {
            return ((cc - Math.max(bl, bB) + bH)%bH) * a5 + a4
        }
        function ca(cd, ch) {
            cd = O(cd, true);
            if (ch < k(O(cd), bi)) {
                return 0
            }
            if (ch >= k(O(cd), aM)) {
                return aY.height()
            }
            var cc = bD("slotMinutes"), cg = ch.getHours() * 60 + ch.getMinutes() - bi, cf = Math.floor(cg / cc), ce = bg[cf];
            if (ce === y) {
                ce = bg[cf] = aY.find("tr:eq(" + cf + ") td div")[0].offsetTop
            }
            return Math.max(0, Math.round(ce - 1 + bI * ((cg%cc) / cc)))
        }
        function bu() {
            return {
                left: bn,
                right: bw - bA
            }
        }
        function bF(cc) {
            return bX
        }
        function bS(cc) {
            var cd = O(cc.start);
            if (cc.allDay) {
                return cd
            }
            return k(cd, bD("defaultEventMinutes"))
        }
        function bp(cc, cd) {
            if (cd) {
                return O(cc)
            }
            return k(O(cc), bD("slotMinutes"))
        }
        function b1(cc, ce, cd) {
            if (cd) {
                if (bD("allDaySlot")) {
                    aP(cc, aC(O(ce), 1), true)
                }
            } else {
                b0(cc, ce)
            }
        }
        function b0(cc, ci) {
            var cg = bD("selectHelper");
            bJ.build();
            if (cg) {
                var ce = az(cc, bb.visStart) * a5 + a4;
                if (ce >= 0 && ce < bH) {
                    var cf = bJ.rect(0, ce, 0, ce, bE);
                    var ch = ca(cc, cc);
                    var cd = ca(cc, ci);
                    if (cd > ch) {
                        cf.top = ch;
                        cf.height = cd - ch;
                        cf.left += 2;
                        cf.width -= 5;
                        if (au.isFunction(cg)) {
                            var cj = cg(cc, ci);
                            if (cj) {
                                cf.position = "absolute";
                                cf.zIndex = 8;
                                aS = au(cj).css(cf).appendTo(bE)
                            }
                        } else {
                            cf.isStart = true;
                            cf.isEnd = true;
                            aS = au(a3({
                                title: "",
                                start: cc,
                                end: ci,
                                className: ["fc-select-helper"],
                                editable: false
                            }, cf));
                            aS.css("opacity", bD("dragOpacity"))
                        }
                        if (aS) {
                            bW(aS);
                            bE.append(aS);
                            av(aS, cf.width, true);
                            aa(aS, cf.height, true)
                        }
                    }
                }
            } else {
                bc(cc, ci)
            }
        }
        function bV() {
            aN();
            if (aS) {
                aS.remove();
                aS = null
            }
        }
        function aZ(cc) {
            if (cc.which == 1 && bD("selectable")) {
                bZ(cc);
                var cd;
                b6.start(function(cf, ce) {
                    bV();
                    if (cf && cf.col == ce.col&&!b2(cf)) {
                        var ch = aO(ce);
                        var cg = aO(cf);
                        cd = [ch, k(O(ch), bD("slotMinutes")), cg, k(O(cg), bD("slotMinutes"))].sort(Y);
                        b0(cd[0], cd[3])
                    } else {
                        cd = null
                    }
                }, cc);
                au(document).one("mouseup", function(ce) {
                    b6.stop();
                    if (cd) {
                        if ( + cd[0] ==+ cd[1]) {
                            ba(cd[0], false, ce)
                        }
                        bv(cd[0], cd[3], false, ce)
                    }
                })
            }
        }
        function ba(cc, ce, cd) {
            bG("dayClick", be[aX(cc.getDay())], cc, ce, cd)
        }
        function aV(ce, cc, cd) {
            b6.start(function(cf) {
                aN();
                if (cf) {
                    if (b2(cf)) {
                        a6(cf.row, cf.col, cf.row, cf.col)
                    } else {
                        var ch = aO(cf);
                        var cg = k(O(ch), bD("defaultEventMinutes"));
                        bc(ch, cg)
                    }
                }
            }, cc)
        }
        function b8(cf, cd, ce) {
            var cc = b6.stop();
            aN();
            if (cc) {
                bG("drop", cf, aO(cc), b2(cc), cd, ce)
            }
        }
    }
    function V() {
        var bk = this;
        bk.renderEvents = aQ;
        bk.compileDaySegs = aL;
        bk.clearEvents = a6;
        bk.slotSegHtml = a7;
        bk.bindDaySeg = a4;
        Z.call(bk);
        var aZ = bk.opt;
        var a8 = bk.trigger;
        var aO = bk.isEventDraggable;
        var bp = bk.isEventResizable;
        var bq = bk.eventEnd;
        var aU = bk.reportEvents;
        var bt = bk.reportEventClear;
        var bb = bk.eventElementHandlers;
        var bg = bk.setHeight;
        var aP = bk.getDaySegmentContainer;
        var bs = bk.getSlotSegmentContainer;
        var aN = bk.getHoverListener;
        var aM = bk.getMaxMinute;
        var bd = bk.getMinMinute;
        var aT = bk.timePosition;
        var bf = bk.colContentLeft;
        var be = bk.colContentRight;
        var bi = bk.renderDaySegs;
        var bc = bk.resizableDayEvent;
        var bh = bk.getColCnt;
        var a0 = bk.getColWidth;
        var aK = bk.getSlotHeight;
        var a5 = bk.getBodyContent;
        var aX = bk.reportEventElement;
        var aR = bk.showEvents;
        var aY = bk.hideEvents;
        var bn = bk.eventDrop;
        var a2 = bk.eventResize;
        var a9 = bk.renderDayOverlay;
        var aS = bk.clearOverlays;
        var bl = bk.calendar;
        var bj = bl.formatDate;
        var a1 = bl.formatDates;
        function aQ(by, bw) {
            aU(by);
            var bx, bv = by.length, bz = [], bu = [];
            for (bx = 0; bx < bv; bx++) {
                if (by[bx].allDay) {
                    bz.push(by[bx])
                } else {
                    bu.push(by[bx])
                }
            }
            if (aZ("allDaySlot")) {
                bi(aL(bz), bw);
                bg()
            }
            aV(a3(bu), bw)
        }
        function a6() {
            bt();
            aP().empty();
            bs().empty()
        }
        function aL(by) {
            var bA = ai(ad(by, au.map(by, ao), bk.visStart, bk.visEnd)), bx, bz = bA.length, bB, bw, bv, bu = [];
            for (bx = 0; bx < bz; bx++) {
                bB = bA[bx];
                for (bw = 0; bw < bB.length; bw++) {
                    bv = bB[bw];
                    bv.row = 0;
                    bv.level = bx;
                    bu.push(bv)
                }
            }
            return bu
        }
        function a3(bG) {
            var bF = bh(), bD = bd(), bx = aM(), bE = k(O(bk.visStart), bD), bB = au.map(bG, br), bC, bw, bA, bu, by, bz, bv = [];
            for (bC = 0; bC < bF; bC++) {
                bw = ai(ad(bG, bB, bE, k(O(bE), bx - bD)));
                E(bw);
                for (bA = 0; bA < bw.length; bA++) {
                    bu = bw[bA];
                    for (by = 0; by < bu.length; by++) {
                        bz = bu[by];
                        bz.col = bC;
                        bz.level = bA;
                        bv.push(bz)
                    }
                }
                aC(bE, 1, true)
            }
            return bv
        }
        function br(bu) {
            if (bu.end) {
                return O(bu.end)
            } else {
                return k(O(bu.start), aZ("defaultEventMinutes"))
            }
        }
        function aV(bK, bL) {
            var bQ, bT = bK.length, bS, bO, bU, bG, bD, bC, bw, bE, bB, bN, bu, bx, bF = "", bV, bR, bz, bv = {}, bY = {}, bX, bW, bA, bI, bP = bs(), by, bJ, bH, bM = bh();
            if (by = aZ("isRTL")) {
                bJ =- 1;
                bH = bM - 1
            } else {
                bJ = 1;
                bH = 0
            }
            for (bQ = 0; bQ < bT; bQ++) {
                bS = bK[bQ];
                bO = bS.event;
                bG = aT(bS.start, bS.start);
                bD = aT(bS.start, bS.end);
                bC = bS.col;
                bw = bS.level;
                bE = bS.forward || 0;
                bB = bf(bC * bJ + bH);
                bN = be(bC * bJ + bH) - bB;
                bN = Math.min(bN - 6, bN * 0.95);
                if (bw) {
                    bu = bN / (bw + bE + 1)
                } else {
                    if (bE) {
                        bu = ((bN / (bE + 1)) - (12 / 2)) * 2
                    } else {
                        bu = bN
                    }
                }
                bx = bB + (bN / (bw + bE + 1) * bw) * bJ + (by ? bN - bu : 0);
                bS.top = bG;
                bS.left = bx;
                bS.outerWidth = bu;
                bS.outerHeight = bD - bG;
                bF += a7(bO, bS)
            }
            bP[0].innerHTML = bF;
            bV = bP.children();
            for (bQ = 0; bQ < bT; bQ++) {
                bS = bK[bQ];
                bO = bS.event;
                bR = au(bV[bQ]);
                bz = a8("eventRender", bO, bO, bR);
                if (bz === false) {
                    bR.remove()
                } else {
                    if (bz && bz !== true) {
                        bR.remove();
                        bR = au(bz).css({
                            position: "absolute",
                            top: bS.top,
                            left: bS.left
                        }).appendTo(bP)
                    }
                    bS.element = bR;
                    if (bO._id === bL) {
                        bm(bO, bR, bS)
                    } else {
                        bR[0]._fci = bQ
                    }
                    aX(bO, bR)
                }
            }
            x(bP, bK, bm);
            for (bQ = 0; bQ < bT; bQ++) {
                bS = bK[bQ];
                if (bR = bS.element) {
                    bW = bv[bX = bS.key = I(bR[0])];
                    bS.vsides = bW === y ? (bv[bX] = G(bR, true)) : bW;
                    bW = bY[bX];
                    bS.hsides = bW === y ? (bY[bX] = j(bR, true)) : bW;
                    bA = bR.find("div.fc-event-content");
                    if (bA.length) {
                        bS.contentTop = bA[0].offsetTop
                    }
                }
            }
            for (bQ = 0; bQ < bT; bQ++) {
                bS = bK[bQ];
                if (bR = bS.element) {
                    bR[0].style.width = Math.max(0, bS.outerWidth - bS.hsides) + "px";
                    bI = Math.max(0, bS.outerHeight - bS.vsides);
                    bR[0].style.height = bI + "px";
                    bO = bS.event;
                    if (bS.contentTop !== y && bI - bS.contentTop < 10) {
                        bR.find("div.fc-event-time").text(bj(bO.start, aZ("timeFormat")) + " - " + bO.title);
                        bR.find("div.fc-event-title").remove()
                    }
                    a8("eventAfterRender", bO, bO, bR)
                }
            }
        }
        function a7(bz, bu) {
            var by = "<";
            var bw = bz.url;
            var bv = S(bz, aZ);
            var bA = (bv ? " style='" + bv + "'" : "");
            var bx = ["fc-event", "fc-event-skin", "fc-event-vert"];
            if (aO(bz)) {
                bx.push("fc-event-draggable")
            }
            if (bu.isStart) {
                bx.push("fc-corner-top")
            }
            if (bu.isEnd) {
                bx.push("fc-corner-bottom")
            }
            bx = bx.concat(bz.className);
            if (bz.source) {
                bx = bx.concat(bz.source.className || [])
            }
            if (bw) {
                by += "a href='" + aF(bz.url) + "'"
            } else {
                by += "div"
            }
            by += " class='" + bx.join(" ") + "' style='position:absolute;z-index:8;top:" + bu.top + "px;left:" + bu.left + "px;" + bv + "'><div class='fc-event-inner fc-event-skin'" + bA + "><div class='fc-event-head fc-event-skin'" + bA + "><div class='fc-event-time'>" + aF(a1(bz.start, bz.end, aZ("timeFormat"))) + "</div></div><div class='fc-event-content'><div class='fc-event-title'>" + aF(bz.title) + "</div></div><div class='fc-event-bg'></div></div>";
            if (bu.isEnd && bp(bz)) {
                by += "<div class='ui-resizable-handle ui-resizable-s'>=</div>"
            }
            by += "</" + (bw ? "a" : "div") + ">";
            return by
        }
        function a4(bw, bv, bu) {
            if (aO(bw)) {
                aW(bw, bv, bu.isStart)
            }
            if (bu.isEnd && bp(bw)) {
                bc(bw, bv, bu)
            }
            bb(bw, bv)
        }
        function bm(bw, bv, bu) {
            var bx = bv.find("div.fc-event-time");
            if (aO(bw)) {
                ba(bw, bv, bx)
            }
            if (bu.isEnd && bp(bw)) {
                bo(bw, bv, bx)
            }
            bb(bw, bv)
        }
        function aW(bv, bD, bu) {
            var bE;
            var bC;
            var bG = true;
            var by;
            var bw = aZ("isRTL")?-1 : 1;
            var bx = aN();
            var bz = a0();
            var bF = aK();
            var bB = bd();
            bD.draggable({
                zIndex: 9,
                opacity: aZ("dragOpacity", "month"),
                revertDuration: aZ("dragRevertDuration"),
                start: function(bH, bI) {
                    a8("eventDragStart", bD, bv, bH, bI);
                    aY(bv, bD);
                    bE = bD.width();
                    bx.start(function(bK, bJ, bL, bM) {
                        aS();
                        if (bK) {
                            bC = false;
                            by = bM * bw;
                            if (!bK.row) {
                                a9(aC(O(bv.start), by), aC(ao(bv), by));
                                bA()
                            } else {
                                if (bu) {
                                    if (bG) {
                                        bD.width(bz - 10);
                                        aa(bD, bF * Math.round((bv.end ? ((bv.end - bv.start) / W) : aZ("defaultEventMinutes")) / aZ("slotMinutes")));
                                        bD.draggable("option", "grid", [bz, 1]);
                                        bG = false
                                    }
                                } else {
                                    bC = true
                                }
                            }
                            bC = bC || (bG&&!by)
                        } else {
                            bA();
                            bC = true
                        }
                        bD.draggable("option", "revert", bC)
                    }, bH, "drag")
                },
                stop: function(bI, bJ) {
                    bx.stop();
                    aS();
                    a8("eventDragStop", bD, bv, bI, bJ);
                    if (bC) {
                        bA();
                        bD.css("filter", "");
                        aR(bv, bD)
                    } else {
                        var bH = 0;
                        if (!bG) {
                            bH = Math.round((bD.offset().top - a5().offset().top) / bF) * aZ("slotMinutes") + bB - (bv.start.getHours() * 60 + bv.start.getMinutes())
                        }
                        bn(this, bv, by, bH, bG, bI, bJ)
                    }
                }
            });
            function bA() {
                if (!bG) {
                    bD.width(bE).height("").draggable("option", "grid", null);
                    bG = true
                }
            }
        }
        function ba(bu, bE, bF) {
            var by;
            var bI = false;
            var bz;
            var bD;
            var bB;
            var bv = aZ("isRTL")?-1 : 1;
            var bx = aN();
            var bG = bh();
            var bA = a0();
            var bH = aK();
            bE.draggable({
                zIndex: 9,
                scroll: false,
                grid: [bA, bH],
                axis: bG == 1 ? "y": false,
                opacity: aZ("dragOpacity"),
                revertDuration: aZ("dragRevertDuration"),
                start: function(bJ, bK) {
                    a8("eventDragStart", bE, bu, bJ, bK);
                    aY(bu, bE);
                    by = bE.position();
                    bD = bB = 0;
                    bx.start(function(bM, bL, bN, bO) {
                        bE.draggable("option", "revert", !bM);
                        aS();
                        if (bM) {
                            bz = bO * bv;
                            if (aZ("allDaySlot")&&!bM.row) {
                                if (!bI) {
                                    bI = true;
                                    bF.hide();
                                    bE.draggable("option", "grid", null)
                                }
                                a9(aC(O(bu.start), bz), aC(ao(bu), bz))
                            } else {
                                bC()
                            }
                        }
                    }, bJ, "drag")
                },
                drag: function(bJ, bK) {
                    bD = Math.round((bK.position.top - by.top) / bH) * aZ("slotMinutes");
                    if (bD != bB) {
                        if (!bI) {
                            bw(bD)
                        }
                        bB = bD
                    }
                },
                stop: function(bK, bL) {
                    var bJ = bx.stop();
                    aS();
                    a8("eventDragStop", bE, bu, bK, bL);
                    if (bJ && (bz || bD || bI)) {
                        bn(this, bu, bz, bI ? 0 : bD, bI, bK, bL)
                    } else {
                        bC();
                        bE.css("filter", "");
                        bE.css(by);
                        bw(0);
                        aR(bu, bE)
                    }
                }
            });
            function bw(bK) {
                var bJ = k(O(bu.start), bK);
                var bL;
                if (bu.end) {
                    bL = k(O(bu.end), bK)
                }
                bF.text(a1(bJ, bL, aZ("timeFormat")))
            }
            function bC() {
                if (bI) {
                    bF.css("display", "");
                    bE.draggable("option", "grid", [bA, bH]);
                    bI = false
                }
            }
        }
        function bo(bw, bv, by) {
            var bz, bu;
            var bx = aK();
            bv.resizable({
                handles: {
                    s: "div.ui-resizable-s"
                },
                grid: bx,
                start: function(bA, bB) {
                    bz = bu = 0;
                    aY(bw, bv);
                    bv.css("z-index", 9);
                    a8("eventResizeStart", this, bw, bA, bB)
                },
                resize: function(bA, bB) {
                    bz = Math.round((Math.max(bx, bv.height()) - bB.originalSize.height) / bx);
                    if (bz != bu) {
                        by.text(a1(bw.start, (!bz&&!bw.end) ? null : k(bq(bw), aZ("slotMinutes") * bz), aZ("timeFormat")));
                        bu = bz
                    }
                },
                stop: function(bA, bB) {
                    a8("eventResizeStop", this, bw, bA, bB);
                    if (bz) {
                        a2(this, bw, 0, aZ("slotMinutes") * bz, bA, bB)
                    } else {
                        bv.css("z-index", 8);
                        aR(bw, bv)
                    }
                }
            })
        }
    }
    function E(aO) {
        var aM, aL, aK, aQ, aP, aN;
        for (aM = aO.length - 1; aM > 0; aM--) {
            aQ = aO[aM];
            for (aL = 0; aL < aQ.length; aL++) {
                aP = aQ[aL];
                for (aK = 0; aK < aO[aM - 1].length; aK++) {
                    aN = aO[aM - 1][aK];
                    if (P(aP, aN)) {
                        aN.forward = Math.max(aN.forward || 0, (aP.forward || 0) + 1)
                    }
                }
            }
        }
    }
    function aD(aO, aV, a0) {
        var aU = this;
        aU.element = aO;
        aU.calendar = aV;
        aU.name = a0;
        aU.opt = aM;
        aU.trigger = aW;
        aU.isEventDraggable = aZ;
        aU.isEventResizable = a3;
        aU.reportEvents = a1;
        aU.eventEnd = a4;
        aU.reportEventElement = a2;
        aU.reportEventClear = ba;
        aU.eventElementHandlers = aL;
        aU.showEvents = a7;
        aU.hideEvents = aK;
        aU.eventDrop = aY;
        aU.eventResize = aN;
        var bb = aU.defaultEventEnd;
        var aT = aV.normalizeEvent;
        var aS = aV.reportEventChange;
        var a5 = {};
        var a9 = [];
        var aR = {};
        var aP = aV.options;
        function aM(be, bc) {
            var bd = aP[be];
            if (typeof bd == "object") {
                return F(bd, bc || a0)
            }
            return bd
        }
        function aW(bc, bd) {
            return aV.trigger.apply(aV, [bc, bd || aU].concat(Array.prototype.slice.call(arguments, 2), [aU]))
        }
        function aZ(bc) {
            return a6(bc)&&!aM("disableDragging")
        }
        function a3(bc) {
            return a6(bc)&&!aM("disableResizing")
        }
        function a6(bc) {
            return aI(bc.editable, (bc.source || {}).editable, aM("editable"))
        }
        function a1(be) {
            a5 = {};
            var bd, bc = be.length, bf;
            for (bd = 0; bd < bc; bd++) {
                bf = be[bd];
                if (a5[bf._id]) {
                    a5[bf._id].push(bf)
                } else {
                    a5[bf._id] = [bf]
                }
            }
        }
        function a4(bc) {
            return bc.end ? O(bc.end) : bb(bc)
        }
        function a2(bd, bc) {
            a9.push(bc);
            if (aR[bd._id]) {
                aR[bd._id].push(bc)
            } else {
                aR[bd._id] = [bc]
            }
        }
        function ba() {
            a9 = [];
            aR = {}
        }
        function aL(bd, bc) {
            bc.click(function(be) {
                if (!bc.hasClass("ui-draggable-dragging")&&!bc.hasClass("ui-resizable-resizing")) {
                    return aW("eventClick", this, bd, be)
                }
            }).hover(function(be) {
                aW("eventMouseover", this, bd, be)
            }, function(be) {
                aW("eventMouseout", this, bd, be)
            })
        }
        function a7(bd, bc) {
            aX(bd, bc, "show")
        }
        function aK(bd, bc) {
            aX(bd, bc, "hide")
        }
        function aX(bf, be, bh) {
            var bg = aR[bf._id], bd, bc = bg.length;
            for (bd = 0; bd < bc; bd++) {
                if (!be || bg[bd][0] != be[0]) {
                    bg[bd][bh]()
                }
            }
        }
        function aY(bf, bd, be, bg, bj, bi, bh) {
            var bk = bd.allDay;
            var bc = bd._id;
            a8(a5[bc], be, bg, bj);
            aW("eventDrop", bf, bd, be, bg, bj, function() {
                a8(a5[bc], - be, - bg, bk);
                aS(bc)
            }, bi, bh);
            aS(bc)
        }
        function aN(bi, bg, bd, bc, bf, bh) {
            var be = bg._id;
            aQ(a5[be], bd, bc);
            aW("eventResize", bi, bg, bd, bc, function() {
                aQ(a5[be], - bd, - bc);
                aS(be)
            }, bf, bh);
            aS(be)
        }
        function a8(bg, be, bd, bh) {
            bd = bd || 0;
            for (var bi, bc = bg.length, bf = 0; bf < bc; bf++) {
                bi = bg[bf];
                if (bh !== y) {
                    bi.allDay = bh
                }
                k(aC(bi.start, be, true), bd);
                if (bi.end) {
                    bi.end = k(aC(bi.end, be, true), bd)
                }
                aT(bi, aP)
            }
        }
        function aQ(bg, be, bd) {
            bd = bd || 0;
            for (var bh, bc = bg.length, bf = 0; bf < bc; bf++) {
                bh = bg[bf];
                bh.end = k(aC(a4(bh), be, true), bd);
                aT(bh, aP)
            }
        }
    }
    function Z() {
        var bd = this;
        bd.renderDaySegs = bc;
        bd.resizableDayEvent = a6;
        var aV = bd.opt;
        var a2 = bd.trigger;
        var aM = bd.isEventDraggable;
        var bi = bd.isEventResizable;
        var bj = bd.eventEnd;
        var aT = bd.reportEventElement;
        var aP = bd.showEvents;
        var aU = bd.hideEvents;
        var aY = bd.eventResize;
        var bm = bd.getRowCnt;
        var bb = bd.getColCnt;
        var aW = bd.getColWidth;
        var aR = bd.allDayRow;
        var bg = bd.allDayBounds;
        var a9 = bd.colContentLeft;
        var a7 = bd.colContentRight;
        var a0 = bd.dayOfWeekCol;
        var a8 = bd.dateCell;
        var aL = bd.compileDaySegs;
        var aN = bd.getDaySegmentContainer;
        var a1 = bd.bindDaySeg;
        var aX = bd.calendar.formatDates;
        var a3 = bd.renderDayOverlay;
        var aQ = bd.clearOverlays;
        var aK = bd.clearSelection;
        function bc(bp, bo) {
            var bn = aN();
            var bv;
            var bu = bm();
            var bA = bb();
            var bt = 0;
            var bz;
            var by;
            var bw;
            var bs;
            var bB = bp.length;
            var br;
            var bx;
            var bq;
            bn[0].innerHTML = a4(bp);
            a5(bp, bn.children());
            bf(bp);
            ba(bp, bn, bo);
            bl(bp);
            aO(bp);
            be(bp);
            bv = aZ();
            for (bz = 0; bz < bu; bz++) {
                by = 0;
                bw = [];
                for (bs = 0; bs < bA; bs++) {
                    bw[bs] = 0
                }
                while (bt < bB && (br = bp[bt]).row == bz) {
                    bx = at(bw.slice(br.startCol, br.endCol));
                    br.top = bx;
                    bx += br.outerHeight;
                    for (bq = br.startCol; bq < br.endCol; bq++) {
                        bw[bq] = bx
                    }
                    bt++
                }
                bv[bz].height(at(bw))
            }
            bk(bp, aS(bv))
        }
        function bh(bq, bp, bu) {
            var bt = au("<div/>");
            var bn;
            var bo = aN();
            var bs;
            var bv = bq.length;
            var br;
            bt[0].innerHTML = a4(bq);
            bn = bt.children();
            bo.append(bn);
            a5(bq, bn);
            bl(bq);
            aO(bq);
            be(bq);
            bk(bq, aS(aZ()));
            bn = [];
            for (bs = 0; bs < bv; bs++) {
                br = bq[bs].element;
                if (br) {
                    if (bq[bs].row === bp) {
                        br.css("top", bu)
                    }
                    bn.push(br[0])
                }
            }
            return au(bn)
        }
        function a4(bs) {
            var bz = aV("isRTL");
            var bx;
            var bD = bs.length;
            var bw;
            var bq;
            var br;
            var bu;
            var bn = bg();
            var bv = bn.left;
            var bp = bn.right;
            var bB;
            var bo;
            var bt;
            var bC;
            var bA;
            var by = "";
            for (bx = 0; bx < bD; bx++) {
                bw = bs[bx];
                bq = bw.event;
                bu = ["fc-event", "fc-event-skin", "fc-event-hori"];
                if (aM(bq)) {
                    bu.push("fc-event-draggable")
                }
                if (bz) {
                    if (bw.isStart) {
                        bu.push("fc-corner-right")
                    }
                    if (bw.isEnd) {
                        bu.push("fc-corner-left")
                    }
                    bB = a0(bw.end.getDay() - 1);
                    bo = a0(bw.start.getDay());
                    bt = bw.isEnd ? a9(bB) : bv;
                    bC = bw.isStart ? a7(bo) : bp
                } else {
                    if (bw.isStart) {
                        bu.push("fc-corner-left")
                    }
                    if (bw.isEnd) {
                        bu.push("fc-corner-right")
                    }
                    bB = a0(bw.start.getDay());
                    bo = a0(bw.end.getDay() - 1);
                    bt = bw.isStart ? a9(bB) : bv;
                    bC = bw.isEnd ? a7(bo) : bp
                }
                bu = bu.concat(bq.className);
                if (bq.source) {
                    bu = bu.concat(bq.source.className || [])
                }
                br = bq.url;
                bA = S(bq, aV);
                if (br) {
                    by += "<a href='" + aF(br) + "'"
                } else {
                    by += "<div"
                }
                by += " class='" + bu.join(" ") + "' style='position:absolute;z-index:8;left:" + bt + "px;" + bA + "'><div class='fc-event-inner fc-event-skin'" + (bA ? " style='" + bA + "'" : "") + ">";
                if (!bq.allDay && bw.isStart) {
                    by += "<span class='fc-event-time'>" + aF(aX(bq.start, bq.end, aV("timeFormat"))) + "</span>"
                }
                by += "<span class='fc-event-title'>" + aF(bq.title) + "</span></div>";
                if (bw.isEnd && bi(bq)) {
                    by += "<div class='ui-resizable-handle ui-resizable-" + (bz ? "w" : "e") + "'>&nbsp;&nbsp;&nbsp;</div>"
                }
                by += "</" + (br ? "a" : "div") + ">";
                bw.left = bt;
                bw.outerWidth = bC - bt;
                bw.startCol = bB;
                bw.endCol = bo + 1
            }
            return by
        }
        function a5(bo, bt) {
            var br;
            var bu = bo.length;
            var bn;
            var bs;
            var bq;
            var bp;
            for (br = 0; br < bu; br++) {
                bn = bo[br];
                bs = bn.event;
                bq = au(bt[br]);
                bp = a2("eventRender", bs, bs, bq);
                if (bp === false) {
                    bq.remove()
                } else {
                    if (bp && bp !== true) {
                        bp = au(bp).css({
                            position: "absolute",
                            left: bn.left
                        });
                        bq.replaceWith(bp);
                        bq = bp
                    }
                    bn.element = bq
                }
            }
        }
        function bf(bo) {
            var bq;
            var br = bo.length;
            var bn;
            var bp;
            for (bq = 0; bq < br; bq++) {
                bn = bo[bq];
                bp = bn.element;
                if (bp) {
                    aT(bn.event, bp)
                }
            }
        }
        function ba(bo, bt, bp) {
            var br;
            var bu = bo.length;
            var bn;
            var bq;
            var bs;
            for (br = 0; br < bu; br++) {
                bn = bo[br];
                bq = bn.element;
                if (bq) {
                    bs = bn.event;
                    if (bs._id === bp) {
                        a1(bs, bq, bn)
                    } else {
                        bq[0]._fci = br
                    }
                }
            }
            x(bt, bo, a1)
        }
        function bl(bo) {
            var bs;
            var bt = bo.length;
            var bn;
            var br;
            var bq, bu;
            var bp = {};
            for (bs = 0; bs < bt; bs++) {
                bn = bo[bs];
                br = bn.element;
                if (br) {
                    bq = bn.key = I(br[0]);
                    bu = bp[bq];
                    if (bu === y) {
                        bu = bp[bq] = j(br, true)
                    }
                    bn.hsides = bu
                }
            }
        }
        function aO(bo) {
            var bq;
            var br = bo.length;
            var bn;
            var bp;
            for (bq = 0; bq < br; bq++) {
                bn = bo[bq];
                bp = bn.element;
                if (bp) {
                    bp[0].style.width = Math.max(0, bn.outerWidth - bn.hsides) + "px"
                }
            }
        }
        function be(bo) {
            var br;
            var bt = bo.length;
            var bn;
            var bq;
            var bp, bu;
            var bs = {};
            for (br = 0; br < bt; br++) {
                bn = bo[br];
                bq = bn.element;
                if (bq) {
                    bp = bn.key;
                    bu = bs[bp];
                    if (bu === y) {
                        bu = bs[bp] = n(bq)
                    }
                    bn.outerHeight = bq[0].offsetHeight + bu
                }
            }
        }
        function aZ() {
            var bn;
            var bp = bm();
            var bo = [];
            for (bn = 0; bn < bp; bn++) {
                bo[bn] = aR(bn).find("td:first div.fc-day-content > div")
            }
            return bo
        }
        function aS(bo) {
            var bn;
            var bq = bo.length;
            var bp = [];
            for (bn = 0; bn < bq; bn++) {
                bp[bn] = bo[bn][0].offsetTop
            }
            return bp
        }
        function bk(bo, bt) {
            var bq;
            var bs = bo.length;
            var bn;
            var bp;
            var br;
            for (bq = 0; bq < bs; bq++) {
                bn = bo[bq];
                bp = bn.element;
                if (bp) {
                    bp[0].style.top = bt[bn.row] + (bn.top || 0) + "px";
                    br = bn.event;
                    a2("eventAfterRender", br, br, bp)
                }
            }
        }
        function a6(bp, bo, bn) {
            var bs = aV("isRTL");
            var br = bs ? "w": "e";
            var bq = bo.find("div.ui-resizable-" + br);
            var bt = false;
            aJ(bo);
            bo.mousedown(function(bu) {
                bu.preventDefault()
            }).click(function(bu) {
                if (bt) {
                    bu.preventDefault();
                    bu.stopImmediatePropagation()
                }
            });
            bq.mousedown(function(bD) {
                if (bD.which != 1) {
                    return 
                }
                bt = true;
                var by = bd.getHoverListener();
                var bC = bm();
                var bE = bb();
                var bw = bs?-1 : 1;
                var bv = bs ? bE - 1: 0;
                var bx = bo.css("top");
                var bz;
                var bu;
                var bB = au.extend({}, bp);
                var bF = a8(bp.start);
                aK();
                au("body").css("cursor", br + "-resize").one("mouseup", bA);
                a2("eventResizeStart", this, bp, bD);
                by.start(function(bH, bG) {
                    if (bH) {
                        var bK = Math.max(bF.row, bH.row);
                        var bL = bH.col;
                        if (bC == 1) {
                            bK = 0
                        }
                        if (bK == bF.row) {
                            if (bs) {
                                bL = Math.min(bF.col, bL)
                            } else {
                                bL = Math.max(bF.col, bL)
                            }
                        }
                        bz = (bK * 7 + bL * bw + bv) - (bG.row * 7 + bG.col * bw + bv);
                        var bJ = aC(bj(bp), bz, true);
                        if (bz) {
                            bB.end = bJ;
                            var bI = bu;
                            bu = bh(aL([bB]), bn.row, bx);
                            bu.find("*").css("cursor", br + "-resize");
                            if (bI) {
                                bI.remove()
                            }
                            aU(bp)
                        } else {
                            if (bu) {
                                aP(bp);
                                bu.remove();
                                bu = null
                            }
                        }
                        aQ();
                        a3(bp.start, aC(O(bJ), 1))
                    }
                }, bD);
                function bA(bG) {
                    a2("eventResizeStop", this, bp, bG);
                    au("body").css("cursor", "");
                    by.stop();
                    aQ();
                    if (bz) {
                        aY(this, bp, bz, 0, bG)
                    }
                    setTimeout(function() {
                        bt = false
                    }, 0)
                }
            })
        }
    }
    function an() {
        var aU = this;
        aU.select = aS;
        aU.unselect = aO;
        aU.reportSelection = aK;
        aU.daySelectionMousedown = aR;
        var aM = aU.opt;
        var aN = aU.trigger;
        var aP = aU.defaultSelectionEnd;
        var aL = aU.renderSelection;
        var aT = aU.clearSelection;
        var aQ = false;
        if (aM("selectable") && aM("unselectAuto")) {
            au(document).mousedown(function(aV) {
                var aW = aM("unselectCancel");
                if (aW) {
                    if (au(aV.target).parents(aW).length) {
                        return 
                    }
                }
                aO(aV)
            })
        }
        function aS(aV, aX, aW) {
            aO();
            if (!aX) {
                aX = aP(aV, aW)
            }
            aL(aV, aX, aW);
            aK(aV, aX, aW)
        }
        function aO(aV) {
            if (aQ) {
                aQ = false;
                aT();
                aN("unselect", null, aV)
            }
        }
        function aK(aV, aY, aX, aW) {
            aQ = true;
            aN("select", null, aV, aY, aX, aW)
        }
        function aR(aY) {
            var a1 = aU.cellDate;
            var aW = aU.cellIsAllDay;
            var aV = aU.getHoverListener();
            var aZ = aU.reportDayClick;
            if (aY.which == 1 && aM("selectable")) {
                aO(aY);
                var aX = this;
                var a0;
                aV.start(function(a3, a2) {
                    aT();
                    if (a3 && aW(a3)) {
                        a0 = [a1(a2), a1(a3)].sort(Y);
                        aL(a0[0], a0[1], true)
                    } else {
                        a0 = null
                    }
                }, aY);
                au(document).one("mouseup", function(a2) {
                    aV.stop();
                    if (a0) {
                        if ( + a0[0] ==+ a0[1]) {
                            aZ(a0[0], true, a2)
                        }
                        aK(a0[0], a0[1], true, a2)
                    }
                })
            }
        }
    }
    function b() {
        var aN = this;
        aN.renderOverlay = aL;
        aN.clearOverlays = aK;
        var aM = [];
        var aO = [];
        function aL(aQ, aP) {
            var aR = aO.shift();
            if (!aR) {
                aR = au("<div class='fc-cell-overlay' style='position:absolute;z-index:3'/>")
            }
            if (aR[0].parentNode != aP[0]) {
                aR.appendTo(aP)
            }
            aM.push(aR.css(aQ).show());
            return aR
        }
        function aK() {
            var aP;
            while (aP = aM.shift()) {
                aO.push(aP.hide().unbind())
            }
        }
    }
    function Q(aK) {
        var aL = this;
        var aM;
        var aN;
        aL.build = function() {
            aM = [];
            aN = [];
            aK(aM, aN)
        };
        aL.cell = function(aO, aU) {
            var aT = aM.length;
            var aP = aN.length;
            var aQ, aR =- 1, aS =- 1;
            for (aQ = 0; aQ < aT; aQ++) {
                if (aU >= aM[aQ][0] && aU < aM[aQ][1]) {
                    aR = aQ;
                    break
                }
            }
            for (aQ = 0; aQ < aP; aQ++) {
                if (aO >= aN[aQ][0] && aO < aN[aQ][1]) {
                    aS = aQ;
                    break
                }
            }
            return (aR >= 0 && aS >= 0) ? {
                row: aR,
                col: aS
            } : null
        };
        aL.rect = function(aR, aT, aP, aQ, aS) {
            var aO = aS.offset();
            return {
                top: aM[aR][0] - aO.top,
                left: aN[aT][0] - aO.left,
                width: aN[aQ][1] - aN[aT][0],
                height: aM[aP][1] - aM[aR][0]
            }
        }
    }
    function ap(aP) {
        var aN = this;
        var aO;
        var aQ;
        var aL;
        var aK;
        aN.start = function(aR, aS, aT) {
            aQ = aR;
            aL = aK = null;
            aP.build();
            aM(aS);
            aO = aT || "mousemove";
            au(document).bind(aO, aM)
        };
        function aM(aR) {
            D(aR);
            var aS = aP.cell(aR.pageX, aR.pageY);
            if (!aS!=!aK || aS && (aS.row != aK.row || aS.col != aK.col)) {
                if (aS) {
                    if (!aL) {
                        aL = aS
                    }
                    aQ(aS, aL, aS.row - aL.row, aS.col - aL.col)
                } else {
                    aQ(aS, aL)
                }
                aK = aS
            }
        }
        aN.stop = function() {
            au(document).unbind(aO, aM);
            return aK
        }
    }
    function D(aK) {
        if (aK.pageX === y) {
            aK.pageX = aK.originalEvent.pageX;
            aK.pageY = aK.originalEvent.pageY
        }
    }
    function o(aL) {
        var aK = this, aM = {}, aP = {}, aO = {};
        function aN(aQ) {
            return aM[aQ] = aM[aQ] || aL(aQ)
        }
        aK.left = function(aQ) {
            return aP[aQ] = aP[aQ] === y ? aN(aQ).position().left : aP[aQ]
        };
        aK.right = function(aQ) {
            return aO[aQ] = aO[aQ] === y ? aK.left(aQ) + aN(aQ).width() : aO[aQ]
        };
        aK.clear = function() {
            aM = {};
            aP = {};
            aO = {}
        }
    }
})(jQuery);
(function(b) {
    b.fn.iv_mini_month_calendar = function(c) {
        return (this.each(function() {
            new b.iv.mini_month_calendar(this, c)
        }))
    };
    b.fn.iv_mini_month_calendar_add_events = function(c) {
        return (this.each(function() {
            var e = b.data(this, "mini_month_calendar");
            e.add_events(c)
        }))
    };
    b.fn.iv_mini_month_calendar_clear_events = function() {
        return (this.each(function() {
            var c = b.data(this, "mini_month_calendar");
            c.clear_events()
        }))
    };
    b.fn.iv_mini_month_calendar_replace_events = function(c) {
        return (this.each(function() {
            var e = b.data(this, "mini_month_calendar");
            e.replace_events(c)
        }))
    };
    b.iv.mini_month_calendar = function(g, k) {
        b.data(g, "mini_month_calendar", this);
        k = b.extend({
            events: [],
            event_uri: null,
            first_day_of_week: 0,
            labels: {
                next: "&rsaquo;",
                previous: "&lsaquo;"
            },
            mobile: false,
            on_event_click: null,
            on_event_over: null,
            on_month_change: null,
            on_month_changing: null,
            on_event_render: null,
            start_date: Date.today(),
            trigger_click: false
        }, k);
        var E = b(g);
        var y = b(".minimonthcalendar_wrapper", E);
        var w = b(".minimonthcalendar_title", E);
        var p = (k.start_date.getUTCOffset());
        var B = jstz.determine().name();
        var m;
        var o = k.trigger_click || (k.mobile && k.on_event_click);
        y.height(E.height - w.height());
        var n;
        var v = k.events;
        var s = k.event_uri;
        var F = k.first_day_of_week%7;
        var f = (k.first_day_of_week + 6)%7;
        this.add_events = u;
        this.clear_events = c;
        this.redraw = x;
        this.replace_events = z;
        var t = {
            count: 0,
            start: null
        };
        q();
        function D(K) {
            if (K == undefined) {
                if (n == undefined) {
                    var G = k.start_date.clone();
                    G.setDate(1);
                    G.clearTime();
                    n = G
                }
            } else {
                n = K.clone();
                n.setDate(1);
                n.clearTime()
            }
            var L = n.clone();
            L.addMonths(1).addDays( - 1);
            var I = F - n.getDay();
            if (I > 0) {
                I -= 7
            }
            var J = f - L.getDay();
            if (J <= 0) {
                J += 7
            }
            var H = L.getDate() + Math.abs(I) + Math.abs(J);
            t = {
                count: H,
                start: n.clone().addDays(I)
            }
        }
        function c() {
            var G = b(".event_date", y);
            b.each(G, function() {
                b(this).removeClass("event_date");
                b(this).unbind("click")
            })
        }
        function j() {
            var G = n.clone().addMonths( - 1);
            var J = b('<th class="month_navigation link-prev">' + k.labels.previous + "</th>");
            J.unbind("click").click(function() {
                A(G)
            });
            var I = n.clone().addMonths(1);
            var H = b('<th class="month_navigation link-next">' + k.labels.next + "</th>");
            H.unbind("click").click(function() {
                A(I)
            });
            var K = b('<tr class="mini_month_header"></tr>');
            K.append(J);
            K.append(b('<th class="calendar_date" colspan="5">' + n.datetostring(Date.CultureInfo.formatPatterns.yearMonth) + "</th>"));
            K.append(H);
            if (k.mobile) {
                b(g).unbind("swipeleft").bind("swipeleft", function() {
                    A(I)
                });
                b(g).unbind("swiperight").bind("swiperight", function() {
                    A(G)
                })
            }
            return (K)
        }
        function r() {
            var M = Date.today();
            var H = b('<table class="mini_month_entry_container"></table>');
            var I = b('<tr class="weekdays"></tr>');
            for (var K = 0; K < 7; K++) {
                var P = b("<th>" + Date.CultureInfo.shortestDayNames[(F + K)%7] + "</th>");
                I.append(P)
            }
            var L = b('<thead class="mini_month_calendar_header"></thead>');
            var J = b('<tbody class="mini_month_content"></tbody>');
            var O;
            for (var K = 0; K < t.count; K++) {
                if (K%7 == 0) {
                    O = b("<tr></tr>")
                }
                var G = t.start.clone();
                G.addDays(K);
                var N = (G.getFullYear() == n.getFullYear() && G.getMonth() == n.getMonth()) ? "active_date": "inactive_date";
                if (M.getFullYear() == G.getFullYear() && M.getMonth() == G.getMonth() && M.getDate() == G.getDate()) {
                    N += " today"
                } else {
                    N += " not_today"
                }
                if (((F + K)%7) == 0 || ((F + K)%7) == 6) {
                    N += " weekend"
                } else {
                    N += " weekday"
                }
                O.append(b('<td class="' + N + '" date="' + G.toUTCString() + '">' + G.getDate() + "</td>"));
                if (K%7 == 6) {
                    J.append(O)
                }
            }
            L.append(j()).append(I);
            H.append(L).append(J);
            return (H)
        }
        function x() {
            var G = r();
            y.empty().append(G);
            C()
        }
        function C() {
            var G = b("td[date]", y);
            b.each(G, function(I, H) {
                b.each(v, function() {
                    var L = new Date(this["start"] * 1000);
                    var N = new Date(this["end"] * 1000);
                    var J = new Date(b(H).attr("date"));
                    var M = J.clone().add({
                        days: 1,
                        seconds: - 1
                    });
                    var K = ((L.getTime() < J.getTime() && N.getTime() > J.getTime()) || (L.getTime() >= J.getTime() && L.getTime() <= M.getTime()) || (N.getTime() >= J.getTime() && N.getTime() <= M.getTime()));
                    if (K) {
                        b(H).addClass("marked");
                        if (k.on_event_click) {
                            b(H).unbind("click").click(function() {
                                k.on_event_click.call(H, {}, J, v)
                            })
                        }
                        if (k.on_event_render) {
                            k.on_event_render.call(H, {}, J, v)
                        }
                        return (false)
                    } else {
                        if (k.mobile) {
                            if (!b(H).hasClass("marked")) {
                                if (k.on_event_click) {
                                    b(H).unbind("click").click(function() {
                                        k.on_event_click.call(H, {}, J, [])
                                    })
                                }
                                if (k.on_event_render) {
                                    k.on_event_render.call(H, {}, J, [])
                                }
                            }
                        }
                    }
                })
            })
        }
        function u(G) {
            if (G && G.length > 0) {
                b.merge(v, G)
            } else {
                if (G) {
                    v.push(G)
                }
            }
            c();
            C()
        }
        function z(G) {
            if (G) {
                v = G
            }
            c();
            C()
        }
        function e(G) {
            if (!G) {
                G = n
            }
            if (s) {
                var H = {
                    start: Math.round(t.start.getTime() / 1000),
                    end: Math.round(t.start.clone().add({
                        days: t.count
                    }).getTime() / 1000),
                    offset: p,
                    tz: B,
                    _output: "json"
                };
                if (m && m.readystate != 4) {
                    m.abort()
                }
                m = b.ajax({
                    url: s,
                    type: "post",
                    data: H,
                    success: function(I) {
                        z(I.events);
                        if (o) {
                            b(b("td.today[date]"), y).click();
                            o = false
                        }
                    },
                    dataType: "json"
                })
            } else {
                if (o) {
                    b(b("td.today[date]"), y).click();
                    o = false
                }
            }
        }
        function A(H) {
            var G;
            D(H);
            if (k.on_month_changing) {
                G = k.on_month_changing.call(this, H)
            }
            if (!G) {
                x();
                e(H);
                if (k.on_month_change) {
                    k.on_month_change.call(this, H)
                }
            }
        }
        function q() {
            D(n);
            x();
            e()
        }
    }
})(jQuery);
/*! qTip2 - Pretty powerful tooltips - v2.0.0 - 2012-11-01
* http://craigsworks.com/projects/qtip2/
* Copyright (c) 2012 Craig Michael Thompson; Licensed MIT, GPL */
(function(c, b, e) {
    (function(f) {
        if (typeof define === "function" && define.amd) {
            define(["jquery"], f)
        } else {
            if (jQuery&&!jQuery.fn.qtip) {
                f(jQuery)
            }
        }
    }(function(E) {
        var y = true, S = false, A = null, g = "x", f = "y", k = "width", D = "height", H = "top", v = "left", C = "bottom", T = "right", B = "center", r = "flip", M = "flipinvert", I = "shift", O, z, J, o = {}, K = "ui-tooltip", F = "ui-widget", G = "ui-state-disabled", n = "div.qtip." + K, L = K + "-default", R = K + "-focus", q = K + "-hover", s = "_replacedByqTip", p = "oldtitle", Q;
        function x(U) {
            var V = function(X) {
                return X === A || "object" !== typeof X
            }, W = function(X) {
                return !E.isFunction(X) && ((!X&&!X.attr) || X.length < 1 || ("object" === typeof X&&!X.jquery))
            };
            if (!U || "object" !== typeof U) {
                return S
            }
            if (V(U.metadata)) {
                U.metadata = {
                    type: U.metadata
                }
            }
            if ("content" in U) {
                if (V(U.content) || U.content.jquery) {
                    U.content = {
                        text: U.content
                    }
                }
                if (W(U.content.text || S)) {
                    U.content.text = S
                }
                if ("title" in U.content) {
                    if (V(U.content.title)) {
                        U.content.title = {
                            text: U.content.title
                        }
                    }
                    if (W(U.content.title.text || S)) {
                        U.content.title.text = S
                    }
                }
            }
            if ("position" in U && V(U.position)) {
                U.position = {
                    my: U.position,
                    at: U.position
                }
            }
            if ("show" in U && V(U.show)) {
                U.show = U.show.jquery ? {
                    target: U.show
                } : {
                    event: U.show
                }
            }
            if ("hide" in U && V(U.hide)) {
                U.hide = U.hide.jquery ? {
                    target: U.hide
                } : {
                    event: U.hide
                }
            }
            if ("style" in U && V(U.style)) {
                U.style = {
                    classes: U.style
                }
            }
            E.each(z, function() {
                if (this.sanitize) {
                    this.sanitize(U)
                }
            });
            return U
        }
        function j(aq, W, ak, al) {
            var aj = this, af = b.body, ac = K + "-" + ak, X = 0, ap = 0, Y = E(), ag = ".qtip-" + ak, ah, ai;
            aj.id = ak;
            aj.rendered = S;
            aj.destroyed = S;
            aj.elements = ah = {
                target: aq
            };
            aj.timers = {
                img: {}
            };
            aj.options = W;
            aj.checks = {};
            aj.plugins = {};
            aj.cache = ai = {
                event: {},
                target: E(),
                disabled: S,
                attr: al,
                onTarget: S,
                lastClass: ""
            };
            function Z(au) {
                var ar = 0, aw, at = W, av = au.split(".");
                while (at = at[av[ar++]]) {
                    if (ar < av.length) {
                        aw = at
                    }
                }
                return [aw || W, av.pop()]
            }
            function ao() {
                var ar = W.style.widget;
                Y.toggleClass("ui-helper-reset " + F, ar).toggleClass(L, W.style.def&&!ar);
                if (ah.content) {
                    ah.content.toggleClass(F + "-content", ar)
                }
                if (ah.titlebar) {
                    ah.titlebar.toggleClass(F + "-header", ar)
                }
                if (ah.button) {
                    ah.button.toggleClass(K + "-icon", !ar)
                }
            }
            function U(ar) {
                if (ah.title) {
                    ah.titlebar.remove();
                    ah.titlebar = ah.title = ah.button = A;
                    if (ar !== S) {
                        aj.reposition()
                    }
                }
            }
            function am() {
                var at = W.content.title.button, ar = typeof at === "string", au = ar ? at: "Close tooltip";
                if (ah.button) {
                    ah.button.remove()
                }
                if (at.jquery) {
                    ah.button = at
                } else {
                    ah.button = E("<a />", {
                        "class": "ui-state-default ui-tooltip-close " + (W.style.widget ? "" : K + "-icon"),
                        title: au,
                        "aria-label": au
                    }).prepend(E("<span />", {
                        "class": "ui-icon ui-icon-close",
                        html: "&times;"
                    }))
                }
                ah.button.appendTo(ah.titlebar).attr("role", "button").click(function(av) {
                    if (!Y.hasClass(G)) {
                        aj.hide(av)
                    }
                    return S
                })
            }
            function ab() {
                var ar = ac + "-title";
                if (ah.titlebar) {
                    U()
                }
                ah.titlebar = E("<div />", {
                    "class": K + "-titlebar " + (W.style.widget ? "ui-widget-header" : "")
                }).append(ah.title = E("<div />", {
                    id: ar,
                    "class": K + "-title",
                    "aria-atomic": y
                })).insertBefore(ah.content).delegate(".ui-tooltip-close", "mousedown keydown mouseup keyup mouseout", function(at) {
                    E(this).toggleClass("ui-state-active ui-state-focus", at.type.substr( - 4) === "down")
                }).delegate(".ui-tooltip-close", "mouseover mouseout", function(at) {
                    E(this).toggleClass("ui-state-hover", at.type === "mouseover")
                });
                if (W.content.title.button) {
                    am()
                }
            }
            function ae(ar) {
                var at = ah.button, au = ah.title;
                if (!aj.rendered) {
                    return S
                }
                if (!ar) {
                    at.remove()
                } else {
                    if (!au) {
                        ab()
                    }
                    am()
                }
            }
            function an(au, ar) {
                var at = ah.title;
                if (!aj.rendered ||!au) {
                    return S
                }
                if (E.isFunction(au)) {
                    au = au.call(aq, ai.event, aj)
                }
                if (au === S || (!au && au !== "")) {
                    return U(S)
                } else {
                    if (au.jquery && au.length > 0) {
                        at.empty().append(au.css({
                            display: "block"
                        }))
                    } else {
                        at.html(au)
                    }
                }
                if (ar !== S && aj.rendered && Y[0].offsetWidth > 0) {
                    aj.reposition(ai.event)
                }
            }
            function ad(au, ar) {
                var at = ah.content;
                if (!aj.rendered ||!au) {
                    return S
                }
                if (E.isFunction(au)) {
                    au = au.call(aq, ai.event, aj) || ""
                }
                if (au.jquery && au.length > 0) {
                    at.empty().append(au.css({
                        display: "block"
                    }))
                } else {
                    at.html(au)
                }
                function av(ax) {
                    var aw, ay = {};
                    function az(aA) {
                        if (aA) {
                            delete ay[aA.src];
                            clearTimeout(aj.timers.img[aA.src]);
                            E(aA).unbind(ag)
                        }
                        if (E.isEmptyObject(ay)) {
                            if (ar !== S) {
                                aj.reposition(ai.event)
                            }
                            ax()
                        }
                    }
                    if ((aw = at.find("img[src]:not([height]):not([width])")).length === 0) {
                        return az()
                    }
                    aw.each(function(aB, aD) {
                        if (ay[aD.src] !== e) {
                            return 
                        }
                        var aC = 0, aA = 3;
                        (function aE() {
                            if (aD.height || aD.width || (aC > aA)) {
                                return az(aD)
                            }
                            aC += 1;
                            aj.timers.img[aD.src] = setTimeout(aE, 700)
                        }());
                        E(aD).bind("error" + ag + " load" + ag, function() {
                            az(this)
                        });
                        ay[aD.src] = aD
                    })
                }
                if (aj.rendered < 0) {
                    Y.queue("fx", av)
                } else {
                    ap = 0;
                    av(E.noop)
                }
                return aj
            }
            function V() {
                var av = W.position, at = {
                    show: W.show.target,
                    hide: W.hide.target,
                    viewport: E(av.viewport),
                    document: E(b),
                    body: E(b.body),
                    window: E(c)
                }, au = {
                    show: E.trim("" + W.show.event).split(" "),
                    hide: E.trim("" + W.hide.event).split(" ")
                }, ar = E.browser.msie && parseInt(E.browser.version, 10) === 6;
                function ax(aA) {
                    if (Y.hasClass(G)) {
                        return S
                    }
                    clearTimeout(aj.timers.show);
                    clearTimeout(aj.timers.hide);
                    var aB = function() {
                        aj.toggle(y, aA)
                    };
                    if (W.show.delay > 0) {
                        aj.timers.show = setTimeout(aB, W.show.delay)
                    } else {
                        aB()
                    }
                }
                function aw(aD) {
                    if (Y.hasClass(G) || X || ap) {
                        return S
                    }
                    var aB = E(aD.relatedTarget || aD.target), aA = aB.closest(n)[0] === Y[0], aC = aB[0] === at.show[0];
                    clearTimeout(aj.timers.show);
                    clearTimeout(aj.timers.hide);
                    if ((av.target === "mouse" && aA) || (W.hide.fixed && ((/mouse(out|leave|move)/).test(aD.type) && (aA || aC)))) {
                        try {
                            aD.preventDefault();
                            aD.stopImmediatePropagation()
                        } catch (aE) {}
                        return 
                    }
                    if (W.hide.delay > 0) {
                        aj.timers.hide = setTimeout(function() {
                            aj.hide(aD)
                        }, W.hide.delay)
                    } else {
                        aj.hide(aD)
                    }
                }
                function ay(aA) {
                    if (Y.hasClass(G)) {
                        return S
                    }
                    clearTimeout(aj.timers.inactive);
                    aj.timers.inactive = setTimeout(function() {
                        aj.hide(aA)
                    }, W.hide.inactive)
                }
                function az(aA) {
                    if (aj.rendered && Y[0].offsetWidth > 0) {
                        aj.reposition(aA)
                    }
                }
                Y.bind("mouseenter" + ag + " mouseleave" + ag, function(aA) {
                    var aB = aA.type === "mouseenter";
                    if (aB) {
                        aj.focus(aA)
                    }
                    Y.toggleClass(q, aB)
                });
                if (/mouse(out|leave)/i.test(W.hide.event)) {
                    if (W.hide.leave === "window") {
                        at.window.bind("mouseout" + ag + " blur" + ag, function(aA) {
                            if (!/select|option/.test(aA.target.nodeName)&&!aA.relatedTarget) {
                                aj.hide(aA)
                            }
                        })
                    }
                }
                if (W.hide.fixed) {
                    at.hide = at.hide.add(Y);
                    Y.bind("mouseover" + ag, function() {
                        if (!Y.hasClass(G)) {
                            clearTimeout(aj.timers.hide)
                        }
                    })
                } else {
                    if (/mouse(over|enter)/i.test(W.show.event)) {
                        at.hide.bind("mouseleave" + ag, function(aA) {
                            clearTimeout(aj.timers.show)
                        })
                    }
                }
                if (("" + W.hide.event).indexOf("unfocus")>-1) {
                    av.container.closest("html").bind("mousedown" + ag + " touchstart" + ag, function(aD) {
                        var aC = E(aD.target), aB = aj.rendered&&!Y.hasClass(G) && Y[0].offsetWidth > 0, aA = aC.parents(n).filter(Y[0]).length > 0;
                        if (aC[0] !== aq[0] && aC[0] !== Y[0]&&!aA&&!aq.has(aC[0]).length&&!aC.attr("disabled")) {
                            aj.hide(aD)
                        }
                    })
                }
                if ("number" === typeof W.hide.inactive) {
                    at.show.bind("qtip-" + ak + "-inactive", ay);
                    E.each(O.inactiveEvents, function(aA, aB) {
                        at.hide.add(ah.tooltip).bind(aB + ag + "-inactive", ay)
                    })
                }
                E.each(au.hide, function(aB, aC) {
                    var aA = E.inArray(aC, au.show), aD = E(at.hide);
                    if ((aA>-1 && aD.add(at.show).length === aD.length) || aC === "unfocus") {
                        at.show.bind(aC + ag, function(aE) {
                            if (Y[0].offsetWidth > 0) {
                                aw(aE)
                            } else {
                                ax(aE)
                            }
                        });
                        delete au.show[aA]
                    } else {
                        at.hide.bind(aC + ag, aw)
                    }
                });
                E.each(au.show, function(aA, aB) {
                    at.show.bind(aB + ag, ax)
                });
                if ("number" === typeof W.hide.distance) {
                    at.show.add(Y).bind("mousemove" + ag, function(aD) {
                        var aC = ai.origin || {}, aB = W.hide.distance, aA = Math.abs;
                        if (aA(aD.pageX - aC.pageX) >= aB || aA(aD.pageY - aC.pageY) >= aB) {
                            aj.hide(aD)
                        }
                    })
                }
                if (av.target === "mouse") {
                    at.show.bind("mousemove" + ag, function(aA) {
                        J = {
                            pageX: aA.pageX,
                            pageY: aA.pageY,
                            type: "mousemove"
                        }
                    });
                    if (av.adjust.mouse) {
                        if (W.hide.event) {
                            Y.bind("mouseleave" + ag, function(aA) {
                                if ((aA.relatedTarget || aA.target) !== at.show[0]) {
                                    aj.hide(aA)
                                }
                            });
                            ah.target.bind("mouseenter" + ag + " mouseleave" + ag, function(aA) {
                                ai.onTarget = aA.type === "mouseenter"
                            })
                        }
                        at.document.bind("mousemove" + ag, function(aA) {
                            if (aj.rendered && ai.onTarget&&!Y.hasClass(G) && Y[0].offsetWidth > 0) {
                                aj.reposition(aA || J)
                            }
                        })
                    }
                }
                if (av.adjust.resize || at.viewport.length) {
                    (E.event.special.resize ? at.viewport : at.window).bind("resize" + ag, az)
                }
                if (at.viewport.length || (ar && Y.css("position") === "fixed")) {
                    at.viewport.bind("scroll" + ag, az)
                }
            }
            function aa() {
                var ar = [W.show.target[0], W.hide.target[0], aj.rendered && ah.tooltip[0], W.position.container[0], W.position.viewport[0], W.position.container.closest("html")[0], c, b];
                if (aj.rendered) {
                    E([]).pushStack(E.grep(ar, function(at) {
                        return typeof at === "object"
                    })).unbind(ag)
                } else {
                    W.show.target.unbind(ag + "-create")
                }
            }
            aj.checks.builtin = {
                "^id$": function(au, av, ar) {
                    var aw = ar === y ? O.nextid: ar, at = K + "-" + aw;
                    if (aw !== S && aw.length > 0&&!E("#" + at).length) {
                        Y[0].id = at;
                        ah.content[0].id = at + "-content";
                        ah.title[0].id = at + "-title"
                    }
                },
                "^content.text$": function(at, au, ar) {
                    ad(ar)
                },
                "^content.title.text$": function(at, au, ar) {
                    if (!ar) {
                        return U()
                    }
                    if (!ah.title && ar) {
                        ab()
                    }
                    an(ar)
                },
                "^content.title.button$": function(at, au, ar) {
                    ae(ar)
                },
                "^position.(my|at)$": function(at, au, ar) {
                    if ("string" === typeof ar) {
                        at[au] = new z.Corner(ar)
                    }
                },
                "^position.container$": function(at, au, ar) {
                    if (aj.rendered) {
                        Y.appendTo(ar)
                    }
                },
                "^show.ready$": function() {
                    if (!aj.rendered) {
                        aj.render(1)
                    } else {
                        aj.toggle(y)
                    }
                },
                "^style.classes$": function(at, au, ar) {
                    Y.attr("class", K + " qtip " + ar)
                },
                "^style.width|height": function(at, au, ar) {
                    Y.css(au, ar)
                },
                "^style.widget|content.title": ao,
                "^events.(render|show|move|hide|focus|blur)$": function(at, au, ar) {
                    Y[(E.isFunction(ar) ? "" : "un") + "bind"]("tooltip" + au, ar)
                },
                "^(show|hide|position).(event|target|fixed|inactive|leave|distance|viewport|adjust)": function() {
                    var ar = W.position;
                    Y.attr("tracking", ar.target === "mouse" && ar.adjust.mouse);
                    aa();
                    V()
                }
            };
            E.extend(aj, {
                _triggerEvent: function(at, ar, au) {
                    var av = E.Event("tooltip" + at);
                    av.originalEvent = (au ? E.extend({}, au) : A) || ai.event || A;
                    Y.trigger(av, [aj].concat(ar || []));
                    return !av.isDefaultPrevented()
                },
                render: function(ar) {
                    if (aj.rendered) {
                        return aj
                    }
                    var av = W.content.text, au = W.content.title.text, at = W.position;
                    E.attr(aq[0], "aria-describedby", ac);
                    Y = ah.tooltip = E("<div/>", {
                        id: ac,
                        "class": K + " qtip " + L + " " + W.style.classes + " " + K + "-pos-" + W.position.my.abbrev(),
                        width: W.style.width || "",
                        height: W.style.height || "",
                        tracking: at.target === "mouse" && at.adjust.mouse,
                        role: "alert",
                        "aria-live": "polite",
                        "aria-atomic": S,
                        "aria-describedby": ac + "-content",
                        "aria-hidden": y
                    }).toggleClass(G, ai.disabled).data("qtip", aj).appendTo(W.position.container).append(ah.content = E("<div />", {
                        "class": K + "-content",
                        id: ac + "-content",
                        "aria-atomic": y
                    }));
                    aj.rendered =- 1;
                    X = 1;
                    if (au) {
                        ab();
                        if (!E.isFunction(au)) {
                            an(au, S)
                        }
                    }
                    if (!E.isFunction(av)) {
                        ad(av, S)
                    }
                    aj.rendered = y;
                    ao();
                    E.each(W.events, function(aw, ax) {
                        if (E.isFunction(ax)) {
                            Y.bind(aw === "toggle" ? "tooltipshow tooltiphide" : "tooltip" + aw, ax)
                        }
                    });
                    E.each(z, function() {
                        if (this.initialize === "render") {
                            this (aj)
                        }
                    });
                    V();
                    Y.queue("fx", function(aw) {
                        aj._triggerEvent("render");
                        X = 0;
                        if (W.show.ready || ar) {
                            aj.toggle(y, ai.event, S)
                        }
                        aw()
                    });
                    return aj
                },
                get: function(at) {
                    var ar, au;
                    switch (at.toLowerCase()) {
                    case"dimensions":
                        ar = {
                            height: Y.outerHeight(S),
                            width: Y.outerWidth(S)
                        };
                        break;
                    case"offset":
                        ar = z.offset(Y, W.position.container);
                        break;
                    default:
                        au = Z(at.toLowerCase());
                        ar = au[0][au[1]];
                        ar = ar.precedance ? ar.string() : ar;
                        break
                    }
                    return ar
                },
                set: function(ax, ay) {
                    var aw = /^position\.(my|at|adjust|target|container)|style|content|show\.ready/i, at = /^content\.(title|attr)|style/i, ar = S, av = aj.checks, au;
                    function az(aD, aB) {
                        var aC, aE, aA;
                        for (aC in av) {
                            for (aE in av[aC]) {
                                if (aA = (new RegExp(aE, "i")).exec(aD)) {
                                    aB.push(aA);
                                    av[aC][aE].apply(aj, aB)
                                }
                            }
                        }
                    }
                    if ("string" === typeof ax) {
                        au = ax;
                        ax = {};
                        ax[au] = ay
                    } else {
                        ax = E.extend(y, {}, ax)
                    }
                    E.each(ax, function(aB, aC) {
                        var aD = Z(aB.toLowerCase()), aA;
                        aA = aD[0][aD[1]];
                        aD[0][aD[1]] = "object" === typeof aC && aC.nodeType ? E(aC) : aC;
                        ax[aB] = [aD[0], aD[1], aC, aA];
                        ar = aw.test(aB) || ar
                    });
                    x(W);
                    X = 1;
                    E.each(ax, az);
                    X = 0;
                    if (aj.rendered && Y[0].offsetWidth > 0 && ar) {
                        aj.reposition(W.position.target === "mouse" ? A : ai.event)
                    }
                    return aj
                },
                toggle: function(au, av) {
                    if (av) {
                        if ((/over|enter/).test(av.type) && (/out|leave/).test(ai.event.type) && W.show.target.add(av.target).length === W.show.target.length && Y.has(av.relatedTarget).length) {
                            return aj
                        }
                        ai.event = E.extend({}, av)
                    }
                    if (!aj.rendered) {
                        return au ? aj.render(1) : aj
                    }
                    var aC = au ? "show": "hide", ar = W[aC], ax = W[!au ? "show": "hide"], aE = W.position, aA = W.content, ay = Y[0].offsetWidth > 0, aw = au || ar.target.length === 1, az=!av || ar.target.length < 2 || ai.target[0] === av.target, aD, aB;
                    if ((typeof au).search("boolean|number")) {
                        au=!ay
                    }
                    if (!Y.is(":animated") && ay === au && az) {
                        return aj
                    }
                    if (!aj._triggerEvent(aC, [90])) {
                        return aj
                    }
                    E.attr(Y[0], "aria-hidden", !!!au);
                    if (au) {
                        ai.origin = E.extend({}, J);
                        aj.focus(av);
                        if (E.isFunction(aA.text)) {
                            ad(aA.text, S)
                        }
                        if (E.isFunction(aA.title.text)) {
                            an(aA.title.text, S)
                        }
                        if (!Q && aE.target === "mouse" && aE.adjust.mouse) {
                            E(b).bind("mousemove.qtip", function(aF) {
                                J = {
                                    pageX: aF.pageX,
                                    pageY: aF.pageY,
                                    type: "mousemove"
                                }
                            });
                            Q = y
                        }
                        aj.reposition(av, arguments[2]);
                        if (!!ar.solo) {
                            E(n, ar.solo).not(Y).qtip("hide", E.Event("tooltipsolo"))
                        }
                    } else {
                        clearTimeout(aj.timers.show);
                        delete ai.origin;
                        if (Q&&!E(n + '[tracking="true"]:visible', ar.solo).not(Y).length) {
                            E(b).unbind("mousemove.qtip");
                            Q = S
                        }
                        aj.blur(av)
                    }
                    function at() {
                        if (au) {
                            if (E.browser.msie) {
                                Y[0].style.removeAttribute("filter")
                            }
                            Y.css("overflow", "");
                            if ("string" === typeof ar.autofocus) {
                                E(ar.autofocus, Y).focus()
                            }
                            ar.target.trigger("qtip-" + ak + "-inactive")
                        } else {
                            Y.css({
                                display: "",
                                visibility: "",
                                opacity: "",
                                left: "",
                                top: ""
                            })
                        }
                        aj._triggerEvent(au ? "visible" : "hidden")
                    }
                    if (ar.effect === S || aw === S) {
                        Y[aC]();
                        at.call(Y)
                    } else {
                        if (E.isFunction(ar.effect)) {
                            Y.stop(1, 1);
                            ar.effect.call(Y, aj);
                            Y.queue("fx", function(aF) {
                                at();
                                aF()
                            })
                        } else {
                            Y.fadeTo(90, au ? 1 : 0, at)
                        }
                    }
                    if (au) {
                        ar.target.trigger("qtip-" + ak + "-inactive")
                    }
                    return aj
                },
                show: function(ar) {
                    return aj.toggle(y, ar)
                },
                hide: function(ar) {
                    return aj.toggle(S, ar)
                },
                focus: function(aw) {
                    if (!aj.rendered) {
                        return aj
                    }
                    var ax = E(n), au = parseInt(Y[0].style.zIndex, 10), at = O.zindex + ax.length, av = E.extend({}, aw), ar;
                    if (!Y.hasClass(R)) {
                        if (aj._triggerEvent("focus", [at], av)) {
                            if (au !== at) {
                                ax.each(function() {
                                    if (this.style.zIndex > au) {
                                        this.style.zIndex = this.style.zIndex - 1
                                    }
                                });
                                ax.filter("." + R).qtip("blur", av)
                            }
                            Y.addClass(R)[0].style.zIndex = at
                        }
                    }
                    return aj
                },
                blur: function(ar) {
                    Y.removeClass(R);
                    aj._triggerEvent("blur", [Y.css("zIndex")], ar);
                    return aj
                },
                reposition: function(aH, aE) {
                    if (!aj.rendered || X) {
                        return aj
                    }
                    X = 1;
                    var aL = W.position.target, aK = W.position, aC = aK.my, aD = aK.at, aF = aK.adjust, au = aF.method.split(" "), aI = Y.outerWidth(S), aG = Y.outerHeight(S), az = 0, aA = 0, aw = Y.css("position") === "fixed", aJ = aK.viewport, aM = {
                        left: 0,
                        top: 0
                    }, aB = aK.container, ar = Y[0].offsetWidth > 0, av, ay, ax;
                    if (E.isArray(aL) && aL.length === 2) {
                        aD = {
                            x: v,
                            y: H
                        };
                        aM = {
                            left: aL[0],
                            top: aL[1]
                        }
                    } else {
                        if (aL === "mouse" && ((aH && aH.pageX) || ai.event.pageX)) {
                            aD = {
                                x: v,
                                y: H
                            };
                            aH = J && J.pageX && (aF.mouse ||!aH ||!aH.pageX) ? {
                                pageX: J.pageX,
                                pageY: J.pageY
                            } : (aH && (aH.type === "resize" || aH.type === "scroll") ? ai.event : aH && aH.pageX && aH.type === "mousemove" ? aH : !aF.mouse && ai.origin && ai.origin.pageX && W.show.distance ? ai.origin : aH) || aH || ai.event || J || {};
                            aM = {
                                top: aH.pageY,
                                left: aH.pageX
                            }
                        } else {
                            if (aL === "event" && aH && aH.target && aH.type !== "scroll" && aH.type !== "resize") {
                                ai.target = E(aH.target)
                            } else {
                                if (aL !== "event") {
                                    ai.target = E(aL.jquery ? aL : ah.target)
                                }
                            }
                            aL = ai.target;
                            aL = E(aL).eq(0);
                            if (aL.length === 0) {
                                return aj
                            } else {
                                if (aL[0] === b || aL[0] === c) {
                                    az = z.iOS ? c.innerWidth : aL.width();
                                    aA = z.iOS ? c.innerHeight : aL.height();
                                    if (aL[0] === c) {
                                        aM = {
                                            top: (aJ || aL).scrollTop(),
                                            left: (aJ || aL).scrollLeft()
                                        }
                                    }
                                } else {
                                    if (z.imagemap && aL.is("area")) {
                                        av = z.imagemap(aj, aL, aD, z.viewport ? au : S)
                                    } else {
                                        if (z.svg && typeof aL[0].xmlbase === "string") {
                                            av = z.svg(aj, aL, aD, z.viewport ? au : S)
                                        } else {
                                            az = aL.outerWidth(S);
                                            aA = aL.outerHeight(S);
                                            aM = z.offset(aL, aB)
                                        }
                                    }
                                }
                            }
                            if (av) {
                                az = av.width;
                                aA = av.height;
                                ay = av.offset;
                                aM = av.position
                            }
                            if ((z.iOS > 3.1 && z.iOS < 4.1) || (z.iOS >= 4.3 && z.iOS < 4.33) || (!z.iOS && aw) || (z.iOS >= 5 && aw)) {
                                ax = E(c);
                                aM.left -= ax.scrollLeft();
                                aM.top -= ax.scrollTop()
                            }
                            aM.left += aD.x === T ? az : aD.x === B ? az / 2 : 0;
                            aM.top += aD.y === C ? aA : aD.y === B ? aA / 2 : 0
                        }
                    }
                    aM.left += aF.x + (aC.x === T?-aI : aC.x === B?-aI / 2 : 0);
                    aM.top += aF.y + (aC.y === C?-aG : aC.y === B?-aG / 2 : 0);
                    if (z.viewport) {
                        aM.adjusted = z.viewport(aj, aM, aK, az, aA, aI, aG);
                        if (ay && aM.adjusted.left) {
                            aM.left += ay.left
                        }
                        if (ay && aM.adjusted.top) {
                            aM.top += ay.top
                        }
                    } else {
                        aM.adjusted = {
                            left: 0,
                            top: 0
                        }
                    }
                    if (!aj._triggerEvent("move", [aM, aJ.elem || aJ], aH)) {
                        return aj
                    }
                    delete aM.adjusted;
                    if (aE === S ||!ar || isNaN(aM.left) || isNaN(aM.top) || aL === "mouse" ||!E.isFunction(aK.effect)) {
                        Y.css(aM)
                    } else {
                        if (E.isFunction(aK.effect)) {
                            aK.effect.call(Y, aj, E.extend({}, aM));
                            Y.queue(function(at) {
                                E(this).css({
                                    opacity: "",
                                    height: ""
                                });
                                if (E.browser.msie) {
                                    this.style.removeAttribute("filter")
                                }
                                at()
                            })
                        }
                    }
                    X = 0;
                    return aj
                },
                disable: function(ar) {
                    if ("boolean" !== typeof ar) {
                        ar=!(Y.hasClass(G) || ai.disabled)
                    }
                    if (aj.rendered) {
                        Y.toggleClass(G, ar);
                        E.attr(Y[0], "aria-disabled", ar)
                    } else {
                        ai.disabled=!!ar
                    }
                    return aj
                },
                enable: function() {
                    return aj.disable(S)
                },
                destroy: function() {
                    var ar = aq[0], at = E.attr(ar, p), au = aq.data("qtip");
                    aj.destroyed = y;
                    if (aj.rendered) {
                        Y.stop(1, 0).remove();
                        E.each(aj.plugins, function() {
                            if (this.destroy) {
                                this.destroy()
                            }
                        })
                    }
                    clearTimeout(aj.timers.show);
                    clearTimeout(aj.timers.hide);
                    aa();
                    if (!au || aj === au) {
                        E.removeData(ar, "qtip");
                        if (W.suppress && at) {
                            E.attr(ar, "title", at);
                            aq.removeAttr(p)
                        }
                        aq.removeAttr("aria-describedby")
                    }
                    aq.unbind(".qtip-" + ak);
                    delete o[aj.id];
                    return aq
                }
            })
        }
        function P(V, U) {
            var Y, ah, ac, W, af, X = E(this), Z = E(b.body), ae = this === b ? Z: X, ad = (X.metadata) ? X.metadata(U.metadata): A, ag = U.metadata.type === "html5" && ad ? ad[U.metadata.name]: A, aa = X.data(U.metadata.name || "qtipopts");
            try {
                aa = typeof aa === "string" ? E.parseJSON(aa) : aa
            } catch (ab) {}
            W = E.extend(y, {}, O.defaults, U, typeof aa === "object" ? x(aa) : A, x(ag || ad));
            ah = W.position;
            W.id = V;
            if ("boolean" === typeof W.content.text) {
                ac = X.attr(W.content.attr);
                if (W.content.attr !== S && ac) {
                    W.content.text = ac
                } else {
                    return S
                }
            }
            if (!ah.container.length) {
                ah.container = Z
            }
            if (ah.target === S) {
                ah.target = ae
            }
            if (W.show.target === S) {
                W.show.target = ae
            }
            if (W.show.solo === y) {
                W.show.solo = ah.container.closest("body")
            }
            if (W.hide.target === S) {
                W.hide.target = ae
            }
            if (W.position.viewport === y) {
                W.position.viewport = ah.container
            }
            ah.container = ah.container.eq(0);
            ah.at = new z.Corner(ah.at);
            ah.my = new z.Corner(ah.my);
            if (E.data(this, "qtip")) {
                if (W.overwrite) {
                    X.qtip("destroy")
                } else {
                    if (W.overwrite === S) {
                        return S
                    }
                }
            }
            if (W.suppress && (af = E.attr(this, "title"))) {
                E(this).removeAttr("title").attr(p, af).attr("title", "")
            }
            Y = new j(X, W, V, !!ac);
            E.data(this, "qtip", Y);
            X.bind("remove.qtip-" + V + " removeqtip.qtip-" + V, function() {
                Y.destroy()
            });
            return Y
        }
        O = E.fn.qtip = function(V, Z, aa) {
            var ab = ("" + V).toLowerCase(), Y = A, U = E.makeArray(arguments).slice(1), X = U[U.length - 1], W = this[0] ? E.data(this[0], "qtip"): A;
            if ((!arguments.length && W) || ab === "api") {
                return W
            } else {
                if ("string" === typeof V) {
                    this.each(function() {
                        var ac = E.data(this, "qtip");
                        if (!ac) {
                            return y
                        }
                        if (X && X.timeStamp) {
                            ac.cache.event = X
                        }
                        if ((ab === "option" || ab === "options") && Z) {
                            if (E.isPlainObject(Z) || aa !== e) {
                                ac.set(Z, aa)
                            } else {
                                Y = ac.get(Z);
                                return S
                            }
                        } else {
                            if (ac[ab]) {
                                ac[ab].apply(ac[ab], U)
                            }
                        }
                    });
                    return Y !== A ? Y : this
                } else {
                    if ("object" === typeof V ||!arguments.length) {
                        W = x(E.extend(y, {}, V));
                        return O.bind.call(this, W, X)
                    }
                }
            }
        };
        O.bind = function(V, U) {
            return this.each(function(Z) {
                var X, W, Y, ab, aa, ad;
                ad = E.isArray(V.id) ? V.id[Z] : V.id;
                ad=!ad || ad === S || ad.length < 1 || o[ad] ? O.nextid++ : (o[ad] = ad);
                ab = ".qtip-" + ad + "-create";
                aa = P.call(this, ad, V);
                if (aa === S) {
                    return y
                }
                X = aa.options;
                E.each(z, function() {
                    if (this.initialize === "initialize") {
                        this (aa)
                    }
                });
                W = {
                    show: X.show.target,
                    hide: X.hide.target
                };
                Y = {
                    show: E.trim("" + X.show.event).replace(/ /g, ab + " ") + ab,
                    hide: E.trim("" + X.hide.event).replace(/ /g, ab + " ") + ab
                };
                if (/mouse(over|enter)/i.test(Y.show)&&!/mouse(out|leave)/i.test(Y.hide)) {
                    Y.hide += " mouseleave" + ab
                }
                W.show.bind("mousemove" + ab, function(ae) {
                    J = {
                        pageX: ae.pageX,
                        pageY: ae.pageY,
                        type: "mousemove"
                    };
                    aa.cache.onTarget = y
                });
                function ac(af) {
                    function ae() {
                        aa.render(typeof af === "object" || X.show.ready);
                        W.show.add(W.hide).unbind(ab)
                    }
                    if (aa.cache.disabled) {
                        return S
                    }
                    aa.cache.event = E.extend({}, af);
                    aa.cache.target = af ? E(af.target) : [e];
                    if (X.show.delay > 0) {
                        clearTimeout(aa.timers.show);
                        aa.timers.show = setTimeout(ae, X.show.delay);
                        if (Y.show !== Y.hide) {
                            W.hide.bind(Y.hide, function() {
                                clearTimeout(aa.timers.show)
                            })
                        }
                    } else {
                        ae()
                    }
                }
                W.show.bind(Y.show, ac);
                if (X.show.ready || X.prerender) {
                    ac(U)
                }
            })
        };
        z = O.plugins = {
            Corner: function(U) {
                U = ("" + U).replace(/([A-Z])/, " $1").replace(/middle/gi, B).toLowerCase();
                this.x = (U.match(/left|right/i) || U.match(/center/) || ["inherit"])[0].toLowerCase();
                this.y = (U.match(/top|bottom|center/i) || ["inherit"])[0].toLowerCase();
                var V = U.charAt(0);
                this.precedance = (V === "t" || V === "b" ? f : g);
                this.string = function() {
                    return this.precedance === f ? this.y + this.x : this.x + this.y
                };
                this.abbrev = function() {
                    var W = this.x.substr(0, 1), X = this.y.substr(0, 1);
                    return W === X ? W : this.precedance === f ? X + W : W + X
                };
                this.invertx = function(W) {
                    this.x = this.x === v ? T : this.x === T ? v : W || this.x
                };
                this.inverty = function(W) {
                    this.y = this.y === H ? C : this.y === C ? H : W || this.y
                };
                this.clone = function() {
                    return {
                        x: this.x,
                        y: this.y,
                        precedance: this.precedance,
                        string: this.string,
                        abbrev: this.abbrev,
                        clone: this.clone,
                        invertx: this.invertx,
                        inverty: this.inverty
                    }
                }
            },
            offset: function(X, U) {
                var aa = X.offset(), Z = X.closest("body")[0], ac = U, V, W, Y;
                function ab(ae, ad) {
                    aa.left += ad * ae.scrollLeft();
                    aa.top += ad * ae.scrollTop()
                }
                if (ac) {
                    do {
                        if (ac.css("position") !== "static") {
                            W = ac.position();
                            aa.left -= W.left + (parseInt(ac.css("borderLeftWidth"), 10) || 0) + (parseInt(ac.css("marginLeft"), 10) || 0);
                            aa.top -= W.top + (parseInt(ac.css("borderTopWidth"), 10) || 0) + (parseInt(ac.css("marginTop"), 10) || 0);
                            if (!V && (Y = ac.css("overflow")) !== "hidden" && Y !== "visible") {
                                V = ac
                            }
                        }
                    }
                    while ((ac = E(ac[0].offsetParent)).length);
                    if (V && V[0] !== Z) {
                        ab(V, 1)
                    }
                }
                return aa
            },
            iOS: parseFloat(("" + (/CPU.*OS ([0-9_]{1,5})|(CPU like).*AppleWebKit.*Mobile/i.exec(navigator.userAgent) || [0, ""])[1]).replace("undefined", "3_2").replace("_", ".").replace("_", "")) || S,
            fn: {
                attr: function(U, Y) {
                    if (this.length) {
                        var V = this[0], X = "title", W = E.data(V, "qtip");
                        if (U === X && W && "object" === typeof W && W.options.suppress) {
                            if (arguments.length < 2) {
                                return E.attr(V, p)
                            }
                            if (W && W.options.content.attr === X && W.cache.attr) {
                                W.set("content.text", Y)
                            }
                            return this.attr(p, Y)
                        }
                    }
                    return E.fn["attr" + s].apply(this, arguments)
                },
                clone: function(V) {
                    var X = E([]), W = "title", U = E.fn["clone" + s].apply(this, arguments);
                    if (!V) {
                        U.filter("[" + p + "]").attr("title", function() {
                            return E.attr(this, p)
                        }).removeAttr(p)
                    }
                    return U
                }
            }
        };
        E.each(z.fn, function(V, W) {
            if (!W || E.fn[V + s]) {
                return y
            }
            var U = E.fn[V + s] = E.fn[V];
            E.fn[V] = function() {
                return W.apply(this, arguments) || U.apply(this, arguments)
            }
        });
        if (!E.ui) {
            E["cleanData" + s] = E.cleanData;
            E.cleanData = function(U) {
                for (var V = 0, W; (W = U[V]) !== e; V++) {
                    try {
                        E(W).triggerHandler("removeqtip")
                    } catch (X) {}
                }
                E["cleanData" + s](U)
            }
        }
        O.version = "@VERSION";
        O.nextid = 0;
        O.inactiveEvents = "click dblclick mousedown mouseup mousemove mouseleave mouseenter".split(" ");
        O.zindex = 15000;
        O.defaults = {
            prerender: S,
            id: S,
            overwrite: y,
            suppress: y,
            content: {
                text: y,
                attr: "title",
                title: {
                    text: S,
                    button: S
                }
            },
            position: {
                my: "top left",
                at: "bottom right",
                target: S,
                container: S,
                viewport: S,
                adjust: {
                    x: 0,
                    y: 0,
                    mouse: y,
                    resize: y,
                    method: "flip flip"
                },
                effect: function(V, W, U) {
                    E(this).animate(W, {
                        duration: 200,
                        queue: S
                    })
                }
            },
            show: {
                target: S,
                event: "mouseenter",
                effect: y,
                delay: 90,
                solo: S,
                ready: S,
                autofocus: S
            },
            hide: {
                target: S,
                event: "mouseleave",
                effect: y,
                delay: 0,
                fixed: S,
                inactive: S,
                leave: "window",
                distance: S
            },
            style: {
                classes: "",
                widget: S,
                width: S,
                height: S,
                def: y
            },
            events: {
                render: A,
                move: A,
                show: A,
                hide: A,
                toggle: A,
                visible: A,
                hidden: A,
                focus: A,
                blur: A
            }
        };
        z.svg = function(aa, Z, ae, X) {
            var ad = E(b), W = Z[0], af = {
                width: 0,
                height: 0,
                position: {
                    top: 10000000000,
                    left: 10000000000
                }
            }, Y, U, ab, ac, V;
            while (!W.getBBox) {
                W = W.parentNode
            }
            if (W.getBBox && W.parentNode) {
                Y = W.getBBox();
                U = W.getScreenCTM();
                ab = W.farthestViewportElement || W;
                if (!ab.createSVGPoint) {
                    return af
                }
                ac = ab.createSVGPoint();
                ac.x = Y.x;
                ac.y = Y.y;
                V = ac.matrixTransform(U);
                af.position.left = V.x;
                af.position.top = V.y;
                ac.x += Y.width;
                ac.y += Y.height;
                V = ac.matrixTransform(U);
                af.width = V.x - af.position.left;
                af.height = V.y - af.position.top;
                af.position.left += ad.scrollLeft();
                af.position.top += ad.scrollTop()
            }
            return af
        };
        function w(Y) {
            var ac = this, ad = Y.elements.tooltip, U = Y.options.content.ajax, W = O.defaults.content.ajax, V = ".qtip-ajax", Z = /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, X = y, aa = S, ab;
            Y.checks.ajax = {
                "^content.ajax": function(ag, af, ae) {
                    if (af === "ajax") {
                        U = ae
                    }
                    if (af === "once") {
                        ac.init()
                    } else {
                        if (U && U.url) {
                            ac.load()
                        } else {
                            ad.unbind(V)
                        }
                    }
                }
            };
            E.extend(ac, {
                init: function() {
                    if (U && U.url) {
                        ad.unbind(V)[U.once ? "one": "bind"]("tooltipshow" + V, ac.load)
                    }
                    return ac
                },
                load: function(af) {
                    if (aa) {
                        aa = S;
                        return 
                    }
                    var ai = U.url.lastIndexOf(" "), ag = U.url, ah, am=!U.loading && X;
                    if (am) {
                        try {
                            af.preventDefault()
                        } catch (ak) {}
                    } else {
                        if (af && af.isDefaultPrevented()) {
                            return ac
                        }
                    }
                    if (ab && ab.abort) {
                        ab.abort()
                    }
                    if (ai>-1) {
                        ah = ag.substr(ai);
                        ag = ag.substr(0, ai)
                    }
                    function ae() {
                        var an;
                        if (Y.destroyed) {
                            return 
                        }
                        X = S;
                        if (am) {
                            aa = y;
                            Y.show(af.originalEvent)
                        }
                        if ((an = W.complete || U.complete) && E.isFunction(an)) {
                            an.apply(U.context || Y, arguments)
                        }
                    }
                    function al(ap, an, ao) {
                        var aq;
                        if (Y.destroyed) {
                            return 
                        }
                        if (ah && "string" === typeof ap) {
                            ap = E("<div/>").append(ap.replace(Z, "")).find(ah)
                        }
                        if ((aq = W.success || U.success) && E.isFunction(aq)) {
                            aq.call(U.context || Y, ap, an, ao)
                        } else {
                            Y.set("content.text", ap)
                        }
                    }
                    function aj(ap, an, ao) {
                        if (Y.destroyed || ap.status === 0) {
                            return 
                        }
                        Y.set("content.text", an + ": " + ao)
                    }
                    ab = E.ajax(E.extend({
                        error: W.error || aj,
                        context: Y
                    }, U, {
                        url: ag,
                        success: al,
                        complete: ae
                    }))
                },
                destroy: function() {
                    if (ab && ab.abort) {
                        ab.abort()
                    }
                    Y.destroyed = y
                }
            });
            ac.init()
        }
        z.ajax = function(V) {
            var U = V.plugins.ajax;
            return "object" === typeof U ? U : (V.plugins.ajax = new w(V))
        };
        z.ajax.initialize = "render";
        z.ajax.sanitize = function(U) {
            var W = U.content, V;
            if (W && "ajax" in W) {
                V = W.ajax;
                if (typeof V !== "object") {
                    V = U.content.ajax = {
                        url: V
                    }
                }
                if ("boolean" !== typeof V.once && V.once) {
                    V.once=!!V.once
                }
            }
        };
        E.extend(y, O.defaults, {
            content: {
                ajax: {
                    loading: y,
                    once: y
                }
            }
        });
        function t(Z, X, V) {
            var W = Math.ceil(X / 2), Y = Math.ceil(V / 2), U = {
                bottomright: [[0, 0], [X, V], [X, 0]],
                bottomleft: [[0, 0], [X, 0], [0, V]],
                topright: [[0, V], [X, 0], [X, V]],
                topleft: [[0, 0], [0, V], [X, V]],
                topcenter: [[0, V], [W, 0], [X, V]],
                bottomcenter: [[0, 0], [X, 0], [W, V]],
                rightcenter: [[0, 0], [X, Y], [0, V]],
                leftcenter: [[X, 0], [X, V], [0, Y]]
            };
            U.lefttop = U.bottomright;
            U.righttop = U.bottomleft;
            U.leftbottom = U.topright;
            U.rightbottom = U.topleft;
            return U[Z.string()]
        }
        function N(af, V) {
            var ad = this, Z = af.options.style.tip, ap = af.elements, W = ap.tooltip, ab = {
                top: 0,
                left: 0
            }, ac = {
                width: Z.width,
                height: Z.height
            }, ak = {}, ai = Z.border || 0, Y = ".qtip-tip", ag=!!(E("<canvas />")[0] || {}).getContext, ao;
            ad.corner = A;
            ad.mimic = A;
            ad.border = ai;
            ad.offset = Z.offset;
            ad.size = ac;
            af.checks.tip = {
                "^position.my|style.tip.(corner|mimic|border)$": function() {
                    if (!ad.init()) {
                        ad.destroy()
                    }
                    af.reposition()
                },
                "^style.tip.(height|width)$": function() {
                    ac = {
                        width: Z.width,
                        height: Z.height
                    };
                    ad.create();
                    ad.update();
                    af.reposition()
                },
                "^content.title.text|style.(classes|widget)$": function() {
                    if (ap.tip && ap.tip.length) {
                        ad.update()
                    }
                }
            };
            function aq(at) {
                var ar = W.is(":visible");
                W.show();
                at();
                W.toggle(ar)
            }
            function X() {
                ac.width = Z.height;
                ac.height = Z.width
            }
            function U() {
                ac.width = Z.width;
                ac.height = Z.height
            }
            function aj(at, az, aC, aA) {
                if (!ap.tip) {
                    return 
                }
                var aE = ad.corner.clone(), aD = aC.adjusted, ar = af.options.position.adjust.method.split(" "), au = ar[0], aw = ar[1] || ar[0], av = {
                    left: S,
                    top: S,
                    x: 0,
                    y: 0
                }, ax, ay = {}, aB;
                if (ad.corner.fixed !== y) {
                    if (au === I && aE.precedance === g && aD.left && aE.y !== B) {
                        aE.precedance = aE.precedance === g ? f : g
                    } else {
                        if (au !== I && aD.left) {
                            aE.x = aE.x === B ? (aD.left > 0 ? v : T) : (aE.x === v ? T : v)
                        }
                    }
                    if (aw === I && aE.precedance === f && aD.top && aE.x !== B) {
                        aE.precedance = aE.precedance === f ? g : f
                    } else {
                        if (aw !== I && aD.top) {
                            aE.y = aE.y === B ? (aD.top > 0 ? H : C) : (aE.y === H ? C : H)
                        }
                    }
                    if (aE.string() !== ab.corner.string() && (ab.top !== aD.top || ab.left !== aD.left)) {
                        ad.update(aE, S)
                    }
                }
                ax = ad.position(aE, aD);
                ax[aE.x] += al(aE, aE.x);
                ax[aE.y] += al(aE, aE.y);
                if (ax.right !== e) {
                    ax.left =- ax.right
                }
                if (ax.bottom !== e) {
                    ax.top =- ax.bottom
                }
                ax.user = Math.max(0, Z.offset);
                if (av.left = (au === I&&!!aD.left)) {
                    if (aE.x === B) {
                        ay["margin-left"] = av.x = ax["margin-left"] - aD.left
                    } else {
                        aB = ax.right !== e ? [aD.left, - ax.left] : [ - aD.left, ax.left];
                        if ((av.x = Math.max(aB[0], aB[1])) > aB[0]) {
                            aC.left -= aD.left;
                            av.left = S
                        }
                        ay[ax.right !== e ? T: v] = av.x
                    }
                }
                if (av.top = (aw === I&&!!aD.top)) {
                    if (aE.y === B) {
                        ay["margin-top"] = av.y = ax["margin-top"] - aD.top
                    } else {
                        aB = ax.bottom !== e ? [aD.top, - ax.top] : [ - aD.top, ax.top];
                        if ((av.y = Math.max(aB[0], aB[1])) > aB[0]) {
                            aC.top -= aD.top;
                            av.top = S
                        }
                        ay[ax.bottom !== e ? C: H] = av.y
                    }
                }
                ap.tip.css(ay).toggle(!((av.x && av.y) || (aE.x === B && av.y) || (aE.y === B && av.x)));
                aC.left -= ax.left.charAt ? ax.user : au !== I || av.top ||!av.left&&!av.top ? ax.left : 0;
                aC.top -= ax.top.charAt ? ax.user : aw !== I || av.left ||!av.left&&!av.top ? ax.top : 0;
                ab.left = aD.left;
                ab.top = aD.top;
                ab.corner = aE.clone()
            }
            function ah() {
                var av = Z.corner, au = af.options.position, ar = au.at, aw = au.my.string ? au.my.string(): au.my;
                if (av === S || (aw === S && ar === S)) {
                    return S
                } else {
                    if (av === y) {
                        ad.corner = new z.Corner(aw)
                    } else {
                        if (!av.string) {
                            ad.corner = new z.Corner(av);
                            ad.corner.fixed = y
                        }
                    }
                }
                ab.corner = new z.Corner(ad.corner.string());
                return ad.corner.string() !== "centercenter"
            }
            function al(ay, aw, at) {
                aw=!aw ? ay[ay.precedance] : aw;
                var ar = ap.titlebar && ay.y === H, ax = ar ? ap.titlebar: W, av = "border-" + aw + "-width", au = function(aA) {
                    return parseInt(aA.css(av), 10)
                }, az;
                aq(function() {
                    az = (at ? au(at) : (au(ap.content) || au(ax) || au(W))) || 0
                });
                return az
            }
            function aa(aA) {
                var au = ap.titlebar && aA.y === H, at = au ? ap.titlebar: ap.content, az = E.browser.mozilla, ax = az ? "-moz-": E.browser.webkit ? "-webkit-": "", aw = "border-radius-" + aA.y + aA.x, av = "border-" + aA.y + "-" + aA.x + "-radius", ay = function(aB) {
                    return parseInt(at.css(aB), 10) || parseInt(W.css(aB), 10)
                }, ar;
                aq(function() {
                    ar = ay(av) || ay(ax + av) || ay(ax + aw) || ay(aw) || 0
                });
                return ar
            }
            function an(aE) {
                var aw, aF, av, aC = ap.tip.css("cssText", ""), aD = aE || ad.corner, ay = /rgba?\(0, 0, 0(, 0)?\)|transparent|#123456/i, ar = "border-" + aD[aD.precedance] + "-color", aA = "background-color", aG = "transparent", au = " !important", az = ap.titlebar, aB = az && (aD.y === H || (aD.y === B && aC.position().top + (ac.height / 2) + Z.offset < az.outerHeight(y))), at = aB ? az: ap.content;
                function ax(aH, aK, aI) {
                    var aJ = aH.css(aK) || aG;
                    if (aI && aJ === aH.css(aI)) {
                        return S
                    } else {
                        return ay.test(aJ) ? S : aJ
                    }
                }
                aq(function() {
                    ak.fill = ax(aC, aA) || ax(at, aA) || ax(ap.content, aA) || ax(W, aA) || aC.css(aA);
                    ak.border = ax(aC, ar, "color") || ax(at, ar, "color") || ax(ap.content, ar, "color") || ax(W, ar, "color") || W.css(ar);
                    E("*", aC).add(aC).css("cssText", aA + ":" + aG + au + ";border:0" + au + ";")
                })
            }
            function am(aB) {
                var az = aB.precedance === f, at = ac[az ? k: D], aC = ac[az ? D: k], ay = aB.string().indexOf(B)>-1, ar = at * (ay ? 0.5 : 1), av = Math.pow, aD = Math.round, aA, ax, aE, au = Math.sqrt(av(ar, 2) + av(aC, 2)), aw = [(ai / ar) * au, (ai / aC) * au];
                aw[2] = Math.sqrt(av(aw[0], 2) - av(ai, 2));
                aw[3] = Math.sqrt(av(aw[1], 2) - av(ai, 2));
                aA = au + aw[2] + aw[3] + (ay ? 0 : aw[0]);
                ax = aA / au;
                aE = [aD(ax * aC), aD(ax * at)];
                return {
                    height: aE[az ? 0: 1],
                    width: aE[az ? 1: 0]
                }
            }
            function ae(ar, au, at) {
                return "<qvml:" + ar + ' xmlns="urn:schemas-microsoft.com:vml" class="qtip-vml" ' + (au || "") + ' style="behavior: url(#default#VML); ' + (at || "") + '" />'
            }
            E.extend(ad, {
                init: function() {
                    var ar = ah() && (ag || E.browser.msie);
                    if (ar) {
                        ad.create();
                        ad.update();
                        W.unbind(Y).bind("tooltipmove" + Y, aj)
                    }
                    return ar
                },
                create: function() {
                    var au = ac.width, at = ac.height, ar;
                    if (ap.tip) {
                        ap.tip.remove()
                    }
                    ap.tip = E("<div />", {
                        "class": "ui-tooltip-tip"
                    }).css({
                        width: au,
                        height: at
                    }).prependTo(W);
                    if (ag) {
                        E("<canvas />").appendTo(ap.tip)[0].getContext("2d").save()
                    } else {
                        ar = ae("shape", 'coordorigin="0,0"', "position:absolute;");
                        ap.tip.html(ar + ar);
                        E("*", ap.tip).bind("click mousedown", function(av) {
                            av.stopPropagation()
                        })
                    }
                },
                update: function(aA, aw) {
                    var az = ap.tip, aD = az.children(), au = ac.width, aB = ac.height, aE = Z.mimic, aC = Math.round, ar, at, ay, av, ax;
                    if (!aA) {
                        aA = ab.corner || ad.corner
                    }
                    if (aE === S) {
                        aE = aA
                    } else {
                        aE = new z.Corner(aE);
                        aE.precedance = aA.precedance;
                        if (aE.x === "inherit") {
                            aE.x = aA.x
                        } else {
                            if (aE.y === "inherit") {
                                aE.y = aA.y
                            } else {
                                if (aE.x === aE.y) {
                                    aE[aA.precedance] = aA[aA.precedance]
                                }
                            }
                        }
                    }
                    ar = aE.precedance;
                    if (aA.precedance === g) {
                        X()
                    } else {
                        U()
                    }
                    ap.tip.css({
                        width: (au = ac.width),
                        height: (aB = ac.height)
                    });
                    an(aA);
                    if (ak.border !== "transparent") {
                        ai = al(aA, A);
                        if (Z.border === 0 && ai > 0) {
                            ak.fill = ak.border
                        }
                        ad.border = ai = Z.border !== y ? Z.border : ai
                    } else {
                        ad.border = ai = 0
                    }
                    ay = t(aE, au, aB);
                    ad.size = ax = am(aA);
                    az.css(ax);
                    if (aA.precedance === f) {
                        av = [aC(aE.x === v ? ai : aE.x === T ? ax.width - au - ai : (ax.width - au) / 2), aC(aE.y === H ? ax.height - aB : 0)]
                    } else {
                        av = [aC(aE.x === v ? ax.width - au : 0), aC(aE.y === H ? ai : aE.y === C ? ax.height - aB - ai : (ax.height - aB) / 2)]
                    }
                    if (ag) {
                        aD.attr(ax);
                        at = aD[0].getContext("2d");
                        at.restore();
                        at.save();
                        at.clearRect(0, 0, 3000, 3000);
                        at.fillStyle = ak.fill;
                        at.strokeStyle = ak.border;
                        at.lineWidth = ai * 2;
                        at.lineJoin = "miter";
                        at.miterLimit = 100;
                        at.translate(av[0], av[1]);
                        at.beginPath();
                        at.moveTo(ay[0][0], ay[0][1]);
                        at.lineTo(ay[1][0], ay[1][1]);
                        at.lineTo(ay[2][0], ay[2][1]);
                        at.closePath();
                        if (ai) {
                            if (W.css("background-clip") === "border-box") {
                                at.strokeStyle = ak.fill;
                                at.stroke()
                            }
                            at.strokeStyle = ak.border;
                            at.stroke()
                        }
                        at.fill()
                    } else {
                        ay = "m" + ay[0][0] + "," + ay[0][1] + " l" + ay[1][0] + "," + ay[1][1] + " " + ay[2][0] + "," + ay[2][1] + " xe";
                        av[2] = ai && /^(r|b)/i.test(aA.string()) ? parseFloat(E.browser.version, 10) === 8 ? 2 : 1 : 0;
                        aD.css({
                            coordsize: (au + ai) + " " + (aB + ai),
                            antialias: "" + (aE.string().indexOf(B)>-1),
                            left: av[0],
                            top: av[1],
                            width: au + ai,
                            height: aB + ai
                        }).each(function(aF) {
                            var aG = E(this);
                            aG[aG.prop ? "prop": "attr"]({
                                coordsize: (au + ai) + " " + (aB + ai),
                                path: ay,
                                fillcolor: ak.fill,
                                filled: !!aF,
                                stroked: !aF
                            }).toggle(!!(ai || aF));
                            if (!aF && aG.html() === "") {
                                aG.html(ae("stroke", 'weight="' + (ai * 2) + 'px" color="' + ak.border + '" miterlimit="1000" joinstyle="miter"'))
                            }
                        })
                    }
                    if (aw !== S) {
                        ad.position(aA)
                    }
                },
                position: function(ax) {
                    var ay = ap.tip, at = {}, ar = Math.max(0, Z.offset), au, aw, av;
                    if (Z.corner === S ||!ay) {
                        return S
                    }
                    ax = ax || ad.corner;
                    au = ax.precedance;
                    aw = am(ax);
                    av = [ax.x, ax.y];
                    if (au === g) {
                        av.reverse()
                    }
                    E.each(av, function(aC, aB) {
                        var az, aD, aA;
                        if (aB === B) {
                            az = au === f ? v : H;
                            at[az] = "50%";
                            at["margin-" + az] =- Math.round(aw[au === f ? k: D] / 2) + ar
                        } else {
                            az = al(ax, aB);
                            aD = al(ax, aB, ap.content);
                            aA = aa(ax);
                            at[aB] = aC ? aD : (ar + (aA > az ? aA : - az))
                        }
                    });
                    at[ax[au]] -= aw[au === g ? k: D];
                    ay.css({
                        top: "",
                        bottom: "",
                        left: "",
                        right: "",
                        margin: ""
                    }).css(at);
                    return at
                },
                destroy: function() {
                    if (ap.tip) {
                        ap.tip.remove()
                    }
                    ap.tip = false;
                    W.unbind(Y)
                }
            });
            ad.init()
        }
        z.tip = function(V) {
            var U = V.plugins.tip;
            return "object" === typeof U ? U : (V.plugins.tip = new N(V))
        };
        z.tip.initialize = "render";
        z.tip.sanitize = function(U) {
            var V = U.style, W;
            if (V && "tip" in V) {
                W = U.style.tip;
                if (typeof W !== "object") {
                    U.style.tip = {
                        corner: W
                    }
                }
                if (!(/string|boolean/i).test(typeof W.corner)) {
                    W.corner = y
                }
                if (typeof W.width !== "number") {
                    delete W.width
                }
                if (typeof W.height !== "number") {
                    delete W.height
                }
                if (typeof W.border !== "number" && W.border !== y) {
                    delete W.border
                }
                if (typeof W.offset !== "number") {
                    delete W.offset
                }
            }
        };
        E.extend(y, O.defaults, {
            style: {
                tip: {
                    corner: y,
                    mimic: S,
                    width: 6,
                    height: 6,
                    border: y,
                    offset: 0
                }
            }
        });
        function m(aa) {
            var ah = this, aj = aa.options.show.modal, U = aa.elements, ai = U.tooltip, W = "#qtip-overlay", V = ".qtipmodal", X = V + aa.id, ab = "is-modal-qtip", Z = E(b.body), ag = z.modal.focusable.join(","), ad = {}, Y;
            aa.checks.modal = {
                "^show.modal.(on|blur)$": function() {
                    ah.init();
                    U.overlay.toggle(ai.is(":visible"))
                },
                "^content.text$": function() {
                    af()
                }
            };
            function af() {
                ad = E(ag, ai).not("[disabled]").map(function() {
                    return typeof this.focus === "function" ? this : null
                })
            }
            function ac(ak) {
                if (ad.length < 1 && ak.length) {
                    ak.not("body").blur()
                } else {
                    ad.first().focus()
                }
            }
            function ae(al) {
                var am = E(al.target), ak = am.closest(".qtip"), an;
                an = ak.length < 1 ? S : (parseInt(ak[0].style.zIndex, 10) > parseInt(ai[0].style.zIndex, 10));
                if (!an && (E(al.target).closest(n)[0] !== ai[0])) {
                    ac(am)
                }
            }
            E.extend(ah, {
                init: function() {
                    if (!aj.on) {
                        return ah
                    }
                    Y = ah.create();
                    ai.attr(ab, y).css("z-index", z.modal.zindex + E(n + "[" + ab + "]").length).unbind(V).unbind(X).bind("tooltipshow" + V + " tooltiphide" + V, function(am, al, ao) {
                        var ak = am.originalEvent;
                        if (am.target === ai[0]) {
                            if (ak && am.type === "tooltiphide" && /mouse(leave|enter)/.test(ak.type) && E(ak.relatedTarget).closest(Y[0]).length) {
                                try {
                                    am.preventDefault()
                                } catch (an) {}
                            } else {
                                if (!ak || (ak&&!ak.solo)) {
                                    ah[am.type.replace("tooltip", "")](am, ao)
                                }
                            }
                        }
                    }).bind("tooltipfocus" + V, function(am) {
                        if (am.isDefaultPrevented() || am.target !== ai[0]) {
                            return 
                        }
                        var an = E(n).filter("[" + ab + "]"), al = z.modal.zindex + an.length, ak = parseInt(ai[0].style.zIndex, 10);
                        Y[0].style.zIndex = al - 2;
                        an.each(function() {
                            if (this.style.zIndex > ak) {
                                this.style.zIndex -= 1
                            }
                        });
                        an.end().filter("." + R).qtip("blur", am.originalEvent);
                        ai.addClass(R)[0].style.zIndex = al;
                        try {
                            am.preventDefault()
                        } catch (ao) {}
                    }).bind("tooltiphide" + V, function(ak) {
                        if (ak.target === ai[0]) {
                            E("[" + ab + "]").filter(":visible").not(ai).last().qtip("focus", ak)
                        }
                    });
                    if (aj.escape) {
                        E(b).unbind(X).bind("keydown" + X, function(ak) {
                            if (ak.keyCode === 27 && ai.hasClass(R)) {
                                aa.hide(ak)
                            }
                        })
                    }
                    if (aj.blur) {
                        U.overlay.unbind(X).bind("click" + X, function(ak) {
                            if (ai.hasClass(R)) {
                                aa.hide(ak)
                            }
                        })
                    }
                    af();
                    return ah
                },
                create: function() {
                    var al = E(W);
                    if (al.length) {
                        return (U.overlay = al.insertAfter(E(n).last()))
                    }
                    Y = U.overlay = E("<div />", {
                        id: W.substr(1),
                        html: "<div></div>",
                        mousedown: function() {
                            return S
                        }
                    }).hide().insertAfter(E(n).last());
                    function ak() {
                        Y.css({
                            height: E(c).height(),
                            width: E(c).width()
                        })
                    }
                    E(c).unbind(V).bind("resize" + V, ak);
                    ak();
                    return Y
                },
                toggle: function(ao, ap, aq) {
                    if (ao && ao.isDefaultPrevented()) {
                        return ah
                    }
                    var an = aj.effect, am = ap ? "show": "hide", ar = Y.is(":visible"), al = E("[" + ab + "]").filter(":visible").not(ai), ak;
                    if (!Y) {
                        Y = ah.create()
                    }
                    if ((Y.is(":animated") && ar === ap) || (!ap && al.length)) {
                        return ah
                    }
                    if (ap) {
                        Y.css({
                            left: 0,
                            top: 0
                        });
                        Y.toggleClass("blurs", aj.blur);
                        if (aj.stealfocus !== S) {
                            Z.bind("focusin" + X, ae);
                            ac(E("body :focus"))
                        }
                    } else {
                        Z.unbind("focusin" + X)
                    }
                    Y.stop(y, S);
                    if (E.isFunction(an)) {
                        an.call(Y, ap)
                    } else {
                        if (an === S) {
                            Y[am]()
                        } else {
                            Y.fadeTo(parseInt(aq, 10) || 90, ap ? 1 : 0, function() {
                                if (!ap) {
                                    E(this).hide()
                                }
                            })
                        }
                    }
                    if (!ap) {
                        Y.queue(function(at) {
                            Y.css({
                                left: "",
                                top: ""
                            });
                            at()
                        })
                    }
                    return ah
                },
                show: function(ak, al) {
                    return ah.toggle(ak, y, al)
                },
                hide: function(ak, al) {
                    return ah.toggle(ak, S, al)
                },
                destroy: function() {
                    var ak = Y;
                    if (ak) {
                        ak = E("[" + ab + "]").not(ai).length < 1;
                        if (ak) {
                            U.overlay.remove();
                            E(b).unbind(V)
                        } else {
                            U.overlay.unbind(V + aa.id)
                        }
                        Z.undelegate("*", "focusin" + X)
                    }
                    return ai.removeAttr(ab).unbind(V)
                }
            });
            ah.init()
        }
        z.modal = function(V) {
            var U = V.plugins.modal;
            return "object" === typeof U ? U : (V.plugins.modal = new m(V))
        };
        z.modal.initialize = "render";
        z.modal.sanitize = function(U) {
            if (U.show) {
                if (typeof U.show.modal !== "object") {
                    U.show.modal = {
                        on: !!U.show.modal
                    }
                } else {
                    if (typeof U.show.modal.on === "undefined") {
                        U.show.modal.on = y
                    }
                }
            }
        };
        z.modal.zindex = O.zindex - 200;
        z.modal.focusable = ["a[href]", "area[href]", "input", "select", "textarea", "button", "iframe", "object", "embed", "[tabindex]", "[contenteditable]"];
        E.extend(y, O.defaults, {
            show: {
                modal: {
                    on: S,
                    effect: y,
                    blur: y,
                    stealfocus: y,
                    escape: y
                }
            }
        });
        z.viewport = function(aj, ar, ap, Y, Z, an, am) {
            var aq = ap.target, X = aj.elements.tooltip, ah = ap.my, ak = ap.at, al = ap.adjust, U = al.method.split(" "), ae = U[0], ac = U[1] || U[0], ao = ap.viewport, af = ap.container, ai = aj.cache, ag = aj.plugins.tip, W = {
                left: 0,
                top: 0
            }, V, ab, aa;
            if (!ao.jquery || aq[0] === c || aq[0] === b.body || al.method === "none") {
                return W
            }
            V = X.css("position") === "fixed";
            ao = {
                elem: ao,
                height: ao[(ao[0] === c ? "h" : "outerH") + "eight"](),
                width: ao[(ao[0] === c ? "w" : "outerW") + "idth"](),
                scrollleft: V ? 0: ao.scrollLeft(),
                scrolltop: V ? 0: ao.scrollTop(),
                offset: ao.offset() || {
                    left: 0,
                    top: 0
                }
            };
            af = {
                elem: af,
                scrollLeft: af.scrollLeft(),
                scrollTop: af.scrollTop(),
                offset: af.offset() || {
                    left: 0,
                    top: 0
                }
            };
            function ad(au, at, ay, aK, aC, aA, aJ, aM, aE) {
                var az = ar[aC], aF = ah[au], aL = ak[au], aN = ay === I, aH =- af.offset[aC] + ao.offset[aC] + ao["scroll" + aC], aB = aF === aC ? aE : aF === aA?-aE : - aE / 2, aG = aL === aC ? aM : aL === aA?-aM : - aM / 2, av = ag && ag.size ? ag.size[aJ] || 0 : 0, aI = ag && ag.corner && ag.corner.precedance === au&&!aN ? av : 0, ax = aH - az + aI, aw = az + aE - ao[aJ] - aH + aI, aD = aB - (ah.precedance === au || aF === ah[at] ? aG : 0) - (aL === B ? aM / 2 : 0);
                if (aN) {
                    aI = ag && ag.corner && ag.corner.precedance === at ? av : 0;
                    aD = (aF === aC ? 1 : - 1) * aB - aI;
                    ar[aC] += ax > 0 ? ax : aw > 0?-aw : 0;
                    ar[aC] = Math.max( - af.offset[aC] + ao.offset[aC] + (aI && ag.corner[au] === B ? ag.offset : 0), az - aD, Math.min(Math.max( - af.offset[aC] + ao.offset[aC] + ao[aJ], az + aD), ar[aC]))
                } else {
                    aK*=(ay === M ? 2 : 0);
                    if (ax > 0 && (aF !== aC || aw > 0)) {
                        ar[aC] -= aD + aK;
                        ab["invert" + au](aC)
                    } else {
                        if (aw > 0 && (aF !== aA || ax > 0)) {
                            ar[aC] -= (aF === B?-aD : aD) + aK;
                            ab["invert" + au](aA)
                        }
                    }
                    if (ar[aC] < aH&&-ar[aC] > aw) {
                        ar[aC] = az;
                        ab = ah.clone()
                    }
                }
                return ar[aC] - az
            }
            if (ae !== "shift" || ac !== "shift") {
                ab = ah.clone()
            }
            W = {
                left: ae !== "none" ? ad(g, f, ae, al.x, v, T, k, Y, an): 0,
                top: ac !== "none" ? ad(f, g, ac, al.y, H, C, D, Z, am): 0
            };
            if (ab && ai.lastClass !== (aa = K + "-pos-" + ab.abbrev())) {
                X.removeClass(aj.cache.lastClass).addClass((aj.cache.lastClass = aa))
            }
            return W
        };
        z.imagemap = function(ad, W, ah, aa) {
            if (!W.jquery) {
                W = E(W)
            }
            var V = (ad.cache.areas = {}), af = (W[0].shape || W.attr("shape")).toLowerCase(), ae = W[0].coords || W.attr("coords"), Z = ae.split(","), ag = [], Y = E('img[usemap="#' + W.parent("map").attr("name") + '"]'), aj = Y.offset(), ai = {
                width: 0,
                height: 0,
                position: {
                    top: 10000000000,
                    right: 0,
                    bottom: 0,
                    left: 10000000000
                }
            }, ab = 0, ac = 0, U;
            function X(au, ar, at) {
                var ao = 0, aq = 1, ap = 1, an = 0, al = 0, am = au.width, ak = au.height;
                while (am > 0 && ak > 0 && aq > 0 && ap > 0) {
                    am = Math.floor(am / 2);
                    ak = Math.floor(ak / 2);
                    if (at.x === v) {
                        aq = am
                    } else {
                        if (at.x === T) {
                            aq = au.width - am
                        } else {
                            aq += Math.floor(am / 2)
                        }
                    }
                    if (at.y === H) {
                        ap = ak
                    } else {
                        if (at.y === C) {
                            ap = au.height - ak
                        } else {
                            ap += Math.floor(ak / 2)
                        }
                    }
                    ao = ar.length;
                    while (ao--) {
                        if (ar.length < 2) {
                            break
                        }
                        an = ar[ao][0] - au.position.left;
                        al = ar[ao][1] - au.position.top;
                        if ((at.x === v && an >= aq) || (at.x === T && an <= aq) || (at.x === B && (an < aq || an > (au.width - aq))) || (at.y === H && al >= ap) || (at.y === C && al <= ap) || (at.y === B && (al < ap || al > (au.height - ap)))) {
                            ar.splice(ao, 1)
                        }
                    }
                }
                return {
                    left: ar[0][0],
                    top: ar[0][1]
                }
            }
            aj.left += Math.ceil((Y.outerWidth() - Y.width()) / 2);
            aj.top += Math.ceil((Y.outerHeight() - Y.height()) / 2);
            if (af === "poly") {
                ab = Z.length;
                while (ab--) {
                    ac = [parseInt(Z[--ab], 10), parseInt(Z[ab + 1], 10)];
                    if (ac[0] > ai.position.right) {
                        ai.position.right = ac[0]
                    }
                    if (ac[0] < ai.position.left) {
                        ai.position.left = ac[0]
                    }
                    if (ac[1] > ai.position.bottom) {
                        ai.position.bottom = ac[1]
                    }
                    if (ac[1] < ai.position.top) {
                        ai.position.top = ac[1]
                    }
                    ag.push(ac)
                }
            } else {
                ab =- 1;
                while (ab++<Z.length) {
                    ag.push(parseInt(Z[ab], 10))
                }
            }
            switch (af) {
            case"rect":
                ai = {
                    width: Math.abs(ag[2] - ag[0]),
                    height: Math.abs(ag[3] - ag[1]),
                    position: {
                        left: Math.min(ag[0], ag[2]),
                        top: Math.min(ag[1], ag[3])
                    }
                };
                break;
            case"circle":
                ai = {
                    width: ag[2] + 2,
                    height: ag[2] + 2,
                    position: {
                        left: ag[0],
                        top: ag[1]
                    }
                };
                break;
            case"poly":
                ai.width = Math.abs(ai.position.right - ai.position.left);
                ai.height = Math.abs(ai.position.bottom - ai.position.top);
                if (ah.abbrev() === "c") {
                    ai.position = {
                        left: ai.position.left + (ai.width / 2),
                        top: ai.position.top + (ai.height / 2)
                    }
                } else {
                    if (!V[ah + ae]) {
                        ai.position = X(ai, ag.slice(), ah);
                        if (aa && (aa[0] === "flip" || aa[1] === "flip")) {
                            ai.offset = X(ai, ag.slice(), {
                                x: ah.x === v ? T: ah.x === T ? v: B,
                                y: ah.y === H ? C: ah.y === C ? H: B
                            });
                            ai.offset.left -= ai.position.left;
                            ai.offset.top -= ai.position.top
                        }
                        V[ah + ae] = ai
                    }
                    ai = V[ah + ae]
                }
                ai.width = ai.height = 0;
                break
            }
            ai.position.left += aj.left;
            ai.position.top += aj.top;
            return ai
        };
        function u(X) {
            var aa = this, U = X.elements, ac = X.options, ab = U.tooltip, W = ".ie6-" + X.id, Z = E("select, object").length < 1, V = 0, Y;
            X.checks.ie6 = {
                "^content|style$": function(ae, af, ad) {
                    redraw()
                }
            };
            E.extend(aa, {
                init: function() {
                    if (Z) {
                        U.bgiframe = E('<iframe class="ui-tooltip-bgiframe" frameborder="0" tabindex="-1" src="javascript:\'\';"  style="display:block; position:absolute; z-index:-1; filter:alpha(opacity=0); -ms-filter:"progid:DXImageTransform.Microsoft.Alpha(Opacity=0)";"></iframe>');
                        U.bgiframe.appendTo(ab);
                        ab.bind("tooltipmove" + W, aa.adjustBGIFrame)
                    }
                    Y = E("<div/>", {
                        id: "qtip-rcontainer"
                    }).appendTo(b.body);
                    aa.redraw()
                },
                adjustBGIFrame: function() {
                    var af = X.get("dimensions"), ae = X.plugins.tip, ag = U.tip, ad, ah;
                    ah = parseInt(ab.css("border-left-width"), 10) || 0;
                    ah = {
                        left: - ah,
                        top: - ah
                    };
                    if (ae && ag) {
                        ad = (ae.corner.precedance === "x") ? ["width", "left"] : ["height", "top"];
                        ah[ad[1]] -= ag[ad[0]]()
                    }
                    U.bgiframe.css(ah).css(af)
                },
                redraw: function() {
                    if (X.rendered < 1 || V) {
                        return aa
                    }
                    var ai = ac.style, ae = ac.position.container, ag, ah, ad, af;
                    V = 1;
                    if (ai.height) {
                        ab.css(D, ai.height)
                    }
                    if (ai.width) {
                        ab.css(k, ai.width)
                    } else {
                        ab.css(k, "").appendTo(Y);
                        ah = ab.width();
                        if (ah%2 < 1) {
                            ah += 1
                        }
                        ad = ab.css("max-width") || "";
                        af = ab.css("min-width") || "";
                        ag = (ad + af).indexOf("%")>-1 ? ae.width() / 100 : 0;
                        ad = ((ad.indexOf("%")>-1 ? ag : 1) * parseInt(ad, 10)) || ah;
                        af = ((af.indexOf("%")>-1 ? ag : 1) * parseInt(af, 10)) || 0;
                        ah = ad + af ? Math.min(Math.max(ah, af), ad) : ah;
                        ab.css(k, Math.round(ah)).appendTo(ae)
                    }
                    V = 0;
                    return aa
                },
                destroy: function() {
                    if (Z) {
                        U.bgiframe.remove()
                    }
                    ab.unbind(W)
                }
            });
            aa.init()
        }
        z.ie6 = function(W) {
            var V = E.browser, U = W.plugins.ie6;
            if (!(V.msie && ("" + V.version).charAt(0) === "6")) {
                return S
            }
            return "object" === typeof U ? U : (W.plugins.ie6 = new u(W))
        };
        z.ie6.initialize = "render"
    }))
}(window, document));
var jstz = function() {
    var g = function(b) {
        b =- b.getTimezoneOffset();
        return null !== b ? b : 0
    }, m = function() {
        return g(new Date(2010, 0, 1, 0, 0, 0, 0))
    }, j = function() {
        return g(new Date(2010, 5, 1, 0, 0, 0, 0))
    }, k = function() {
        var e = m(), f = j(), c = m() - j();
        return new jstz.TimeZone(jstz.olson.timezones[0 > c ? e + ",1": 0 < c ? f + ",1,s": e + ",0"])
    };
    return {
        determine_timezone: function() {
            "undefined" !== typeof console && console.log("jstz.determine_timezone() is deprecated and will be removed in an upcoming version. Please use jstz.determine() instead.");
            return k()
        },
        determine: k,
        date_is_dst: function(b) {
            var c = 5 < b.getMonth() ? j(): m(), b = g(b);
            return 0 !== c - b
        }
    }
}();
jstz.TimeZone = function(e) {
    var f = null, f = e;
    "undefined" !== typeof jstz.olson.ambiguity_list[f] && function() {
        for (var c = jstz.olson.ambiguity_list[f], j = c.length, g = 0, k = c[0]; g < j; g += 1) {
            if (k = c[g], jstz.date_is_dst(jstz.olson.dst_start_dates[k])) {
                f = k;
                break
            }
        }
    }();
    return {
        name: function() {
            return f
        }
    }
};
jstz.olson = {};
jstz.olson.timezones = {
    "-720,0": "Etc/GMT+12",
    "-660,0": "Pacific/Pago_Pago",
    "-600,1": "America/Adak",
    "-600,0": "Pacific/Honolulu",
    "-570,0": "Pacific/Marquesas",
    "-540,0": "Pacific/Gambier",
    "-540,1": "America/Anchorage",
    "-480,1": "America/Los_Angeles",
    "-480,0": "Pacific/Pitcairn",
    "-420,0": "America/Phoenix",
    "-420,1": "America/Denver",
    "-360,0": "America/Guatemala",
    "-360,1": "America/Chicago",
    "-360,1,s": "Pacific/Easter",
    "-300,0": "America/Bogota",
    "-300,1": "America/New_York",
    "-270,0": "America/Caracas",
    "-240,1": "America/Halifax",
    "-240,0": "America/Santo_Domingo",
    "-240,1,s": "America/Asuncion",
    "-210,1": "America/St_Johns",
    "-180,1": "America/Godthab",
    "-180,0": "America/Argentina/Buenos_Aires",
    "-180,1,s": "America/Montevideo",
    "-120,0": "America/Noronha",
    "-120,1": "Etc/GMT+2",
    "-60,1": "Atlantic/Azores",
    "-60,0": "Atlantic/Cape_Verde",
    "0,0": "Etc/UTC",
    "0,1": "Europe/London",
    "60,1": "Europe/Berlin",
    "60,0": "Africa/Lagos",
    "60,1,s": "Africa/Windhoek",
    "120,1": "Asia/Beirut",
    "120,0": "Africa/Johannesburg",
    "180,1": "Europe/Moscow",
    "180,0": "Asia/Baghdad",
    "210,1": "Asia/Tehran",
    "240,0": "Asia/Dubai",
    "240,1": "Asia/Yerevan",
    "270,0": "Asia/Kabul",
    "300,1": "Asia/Yekaterinburg",
    "300,0": "Asia/Karachi",
    "330,0": "Asia/Kolkata",
    "345,0": "Asia/Kathmandu",
    "360,0": "Asia/Dhaka",
    "360,1": "Asia/Omsk",
    "390,0": "Asia/Rangoon",
    "420,1": "Asia/Krasnoyarsk",
    "420,0": "Asia/Jakarta",
    "480,0": "Asia/Shanghai",
    "480,1": "Asia/Irkutsk",
    "525,0": "Australia/Eucla",
    "525,1,s": "Australia/Eucla",
    "540,1": "Asia/Yakutsk",
    "540,0": "Asia/Tokyo",
    "570,0": "Australia/Darwin",
    "570,1,s": "Australia/Adelaide",
    "600,0": "Australia/Brisbane",
    "600,1": "Asia/Vladivostok",
    "600,1,s": "Australia/Sydney",
    "630,1,s": "Australia/Lord_Howe",
    "660,1": "Asia/Kamchatka",
    "660,0": "Pacific/Noumea",
    "690,0": "Pacific/Norfolk",
    "720,1,s": "Pacific/Auckland",
    "720,0": "Pacific/Tarawa",
    "765,1,s": "Pacific/Chatham",
    "780,0": "Pacific/Tongatapu",
    "780,1,s": "Pacific/Apia",
    "840,0": "Pacific/Kiritimati"
};
jstz.olson.dst_start_dates = {
    "America/Denver": new Date(2011, 2, 13, 3, 0, 0, 0),
    "America/Mazatlan": new Date(2011, 3, 3, 3, 0, 0, 0),
    "America/Chicago": new Date(2011, 2, 13, 3, 0, 0, 0),
    "America/Mexico_City": new Date(2011, 3, 3, 3, 0, 0, 0),
    "Atlantic/Stanley": new Date(2011, 8, 4, 7, 0, 0, 0),
    "America/Asuncion": new Date(2011, 9, 2, 3, 0, 0, 0),
    "America/Santiago": new Date(2011, 9, 9, 3, 0, 0, 0),
    "America/Campo_Grande": new Date(2011, 9, 16, 5, 0, 0, 0),
    "America/Montevideo": new Date(2011, 9, 2, 3, 0, 0, 0),
    "America/Sao_Paulo": new Date(2011, 9, 16, 5, 0, 0, 0),
    "America/Los_Angeles": new Date(2011, 2, 13, 8, 0, 0, 0),
    "America/Santa_Isabel": new Date(2011, 3, 5, 8, 0, 0, 0),
    "America/Havana": new Date(2011, 2, 13, 2, 0, 0, 0),
    "America/New_York": new Date(2011, 2, 13, 7, 0, 0, 0),
    "Asia/Gaza": new Date(2011, 2, 26, 23, 0, 0, 0),
    "Asia/Beirut": new Date(2011, 2, 27, 1, 0, 0, 0),
    "Europe/Minsk": new Date(2011, 2, 27, 2, 0, 0, 0),
    "Europe/Helsinki": new Date(2011, 2, 27, 4, 0, 0, 0),
    "Europe/Istanbul": new Date(2011, 2, 28, 5, 0, 0, 0),
    "Asia/Damascus": new Date(2011, 3, 1, 2, 0, 0, 0),
    "Asia/Jerusalem": new Date(2011, 3, 1, 6, 0, 0, 0),
    "Africa/Cairo": new Date(2010, 3, 30, 4, 0, 0, 0),
    "Asia/Yerevan": new Date(2011, 2, 27, 4, 0, 0, 0),
    "Asia/Baku": new Date(2011, 2, 27, 8, 0, 0, 0),
    "Pacific/Auckland": new Date(2011, 8, 26, 7, 0, 0, 0),
    "Pacific/Fiji": new Date(2010, 11, 29, 23, 0, 0, 0),
    "America/Halifax": new Date(2011, 2, 13, 6, 0, 0, 0),
    "America/Goose_Bay": new Date(2011, 2, 13, 2, 1, 0, 0),
    "America/Miquelon": new Date(2011, 2, 13, 5, 0, 0, 0),
    "America/Godthab": new Date(2011, 2, 27, 1, 0, 0, 0)
};
jstz.olson.ambiguity_list = {
    "America/Denver": ["America/Denver", "America/Mazatlan"],
    "America/Chicago": ["America/Chicago", "America/Mexico_City"],
    "America/Asuncion": ["Atlantic/Stanley", "America/Asuncion", "America/Santiago", "America/Campo_Grande"],
    "America/Montevideo": ["America/Montevideo", "America/Sao_Paulo"],
    "Asia/Beirut": "Asia/Gaza Asia/Beirut Europe/Minsk Europe/Helsinki Europe/Istanbul Asia/Damascus Asia/Jerusalem Africa/Cairo".split(" "),
    "Asia/Yerevan": ["Asia/Yerevan", "Asia/Baku"],
    "Pacific/Auckland": ["Pacific/Auckland", "Pacific/Fiji"],
    "America/Los_Angeles": ["America/Los_Angeles", "America/Santa_Isabel"],
    "America/New_York": ["America/Havana", "America/New_York"],
    "America/Halifax": ["America/Goose_Bay", "America/Halifax"],
    "America/Godthab": ["America/Miquelon", "America/Godthab"]
};
/*! 
* FitVids 1.0
*
* Copyright 2011, Chris Coyier - http://css-tricks.com + Dave Rupert - http://daverupert.com
* Credit to Thierry Koblentz - http://www.alistapart.com/articles/creating-intrinsic-ratios-for-video/
* Released under the WTFPL license - http://sam.zoy.org/wtfpl/
*
* Date: Thu Sept 01 18:00:00 2011 -0500
*/
(function(b) {
    b.fn.fitVids = function(c) {
        var e = {
            customSelector: null
        };
        var g = document.createElement("div"), f = document.getElementsByTagName("base")[0] || document.getElementsByTagName("script")[0];
        g.className = "fit-vids-style";
        g.innerHTML = "&shy;<style>               .fluid-width-video-wrapper {                 width: 100%;                              position: relative;                       padding: 0;                            }                                                                                   .fluid-width-video-wrapper iframe,        .fluid-width-video-wrapper object,        .fluid-width-video-wrapper embed {           position: absolute;                       top: 0;                                   left: 0;                                  width: 100%;                              height: 100%;                          }                                       </style>";
        f.parentNode.insertBefore(g, f);
        if (c) {
            b.extend(e, c)
        }
        return this.each(function() {
            var j = ["iframe[src^='http://player.vimeo.com']", "iframe[src^='http://www.youtube.com']", "iframe[src^='http://www.kickstarter.com']", "object", "embed"];
            if (e.customSelector) {
                j.push(e.customSelector)
            }
            var k = b(this).find(j.join(","));
            k.each(function() {
                var p = b(this);
                if (this.tagName.toLowerCase() == "embed" && p.parent("object").length || p.parent(".fluid-width-video-wrapper").length) {
                    return 
                }
                var m = this.tagName.toLowerCase() == "object" ? p.attr("height"): p.height(), n = m / p.width();
                if (!p.attr("id")) {
                    var o = "fitvid" + Math.floor(Math.random() * 999999);
                    p.attr("id", o)
                }
                p.wrap('<div class="fluid-width-video-wrapper"></div>').parent(".fluid-width-video-wrapper").css("padding-top", (n * 100) + "%");
                p.removeAttr("height").removeAttr("width")
            })
        })
    }
})(jQuery);
!function(w, v) {
    function n(D, C) {
        function B() {}
        B[s] = this[s];
        var A = this, z = new B, y = r(D), x = y ? D: this, f = y ? {}
        : D, e = function() {
            this.initialize ? this.initialize.apply(this, arguments) : (C || y && A.apply(this, arguments), x.apply(this, arguments))
        };
        e.methods = function(b) {
            o(z, b, A), e[s] = z;
            return this
        }, e.methods.call(e, f).prototype.constructor = e, e.extend = arguments.callee, e[s].implement = e.statics = function(g, c) {
            g = typeof g == "string" ? function() {
                var b = {};
                b[g] = c;
                return b
            }() : g, o(this, g, A);
            return this
        };
        return e
    }
    function o(e, c, j) {
        for (var f in c) {
            c.hasOwnProperty(f) && (e[f] = r(c[f]) && r(j[s][f]) && u.test(c[f]) ? p(f, c[f], j) : c[f])
        }
    }
    function p(f, e, g) {
        return function() {
            var c = this.supr;
            this.supr = g[s][f];
            var b = e.apply(this, arguments);
            this.supr = c;
            return b
        }
    }
    function q(b) {
        return n.call(r(b) ? b : t, b, 1)
    }
    var u = /xyz/.test(function() {
        xyz
    }) ? /\bsupr\b/: /.*/, t = function() {}, s = "prototype", r = function(b) {
        return typeof b === v
    };
    if (typeof module != "undefined" && module.exports) {
        module.exports = q
    } else {
        var m = w.klass;
        q.noConflict = function() {
            w.klass = m;
            return this
        }, w.klass = q
    }
}(this, "function");
