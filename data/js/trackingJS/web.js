/*
 HTML5 Shiv 3.7.2 | @afarkas @jdalton @jon_neal @rem | MIT/GPL2 Licensed

 Lo-Dash 2.4.1 (Custom Build) <http://lodash.com/>
 Build: `lodash modern -o ./dist/lodash.js`
 Copyright 2012-2013 The Dojo Foundation <http://dojofoundation.org/>
 Based on Underscore.js 1.5.2 <http://underscorejs.org/LICENSE>
 Copyright 2009-2013 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
 Available under MIT license <http://lodash.com/license>
*/
(function() {
    Function.prototype.bind || (Function.prototype.bind = function(c) {
        if ("function" !== typeof this)
            throw new TypeError("Function.prototype.bind - what is trying to be bound is not callable");
        var f = Array.prototype.slice.call(arguments, 1), g = this, k = function() {}, n = function() {
            return g.apply(this instanceof k && c ? this : c, f.concat(Array.prototype.slice.call(arguments)))
        };
        k.prototype = this.prototype;
        n.prototype = new k;
        return n
    })
})();
(function(c, f) {
    function g() {
        var c = D.elements;
        return "string" == typeof c ? c.split(" ") : c
    }
    function k(c) {
        var f = I[c[A]];
        f || (f = {}, x++, c[A] = x, I[x] = f);
        return f
    }
    function n(c, g, n) {
        g || (g = f);
        if (K)
            return g.createElement(c);
        n || (n = k(g));
        g = n.cache[c] ? n.cache[c].cloneNode() : B.test(c) ? (n.cache[c] = n.createElem(c)).cloneNode() : n.createElem(c);
        return !g.canHaveChildren || w.test(c) || g.tagUrn ? g : n.frag.appendChild(g)
    }
    function v(c, f) {
        f.cache || (f.cache = {}, f.createElem = c.createElement, f.createFrag = c.createDocumentFragment, f.frag =
        f.createFrag());
        c.createElement = function(g) {
            return D.shivMethods ? n(g, c, f) : f.createElem(g)
        };
        c.createDocumentFragment = Function("h,f", "return function(){var n\x3df.cloneNode(),c\x3dn.createElement;h.shivMethods\x26\x26(" + g().join().replace(/[\w\-:]+/g, function(c) {
            f.createElem(c);
            f.frag.createElement(c);
            return 'c("' + c + '")'
        }) + ");return n}")(D, f.frag)
    }
    function t(c) {
        c || (c = f);
        var g = k(c);
        if (D.shivCSS&&!E&&!g.hasCSS) {
            var n, P = c;
            n = P.createElement("p");
            P = P.getElementsByTagName("head")[0] || P.documentElement;
            n.innerHTML =
            "x\x3cstyle\x3earticle,aside,dialog,figcaption,figure,footer,header,hgroup,main,nav,section{display:block}mark{background:#FF0;color:#000}template{display:none}\x3c/style\x3e";
            n = P.insertBefore(n.lastChild, P.firstChild);
            g.hasCSS=!!n
        }
        K || v(c, g);
        return c
    }
    var z = c.html5 || {}, w = /^<|^(?:button|map|select|textarea|object|iframe|option|optgroup)$/i, B = /^(?:a|b|code|div|fieldset|h1|h2|h3|h4|h5|h6|i|label|li|ol|p|q|span|strong|style|table|tbody|td|th|tr|ul)$/i, E, A = "_html5shiv", x = 0, I = {}, K;
    (function() {
        try {
            var c =
            f.createElement("a");
            c.innerHTML = "\x3cxyz\x3e\x3c/xyz\x3e";
            E = "hidden"in c;
            var g;
            if (!(g = 1 == c.childNodes.length)) {
                f.createElement("a");
                var n = f.createDocumentFragment();
                g = "undefined" == typeof n.cloneNode || "undefined" == typeof n.createDocumentFragment || "undefined" == typeof n.createElement
            }
            K = g
        } catch (P) {
            K = E=!0
        }
    })();
    var D = {
        elements: z.elements || "abbr article aside audio bdi canvas data datalist details dialog figcaption figure footer header hgroup main mark meter nav output picture progress section summary template time video",
        version: "3.7.2",
        shivCSS: !1 !== z.shivCSS,
        supportsUnknownElements: K,
        shivMethods: !1 !== z.shivMethods,
        type: "default",
        shivDocument: t,
        createElement: n,
        createDocumentFragment: function(c, n) {
            c || (c = f);
            if (K)
                return c.createDocumentFragment();
            n = n || k(c);
            for (var v = n.frag.cloneNode(), P = 0, Z = g(), t = Z.length; P < t; P++)
                v.createElement(Z[P]);
            return v
        },
        addElements: function(c, f) {
            var g = D.elements;
            "string" != typeof g && (g = g.join(" "));
            "string" != typeof c && (c = c.join(" "));
            D.elements = g + " " + c;
            t(f)
        }
    };
    c.html5 = D;
    t(f)
})(this, document);
(function(c, f) {
    function g(a) {
        var b = Ga[a] = {};
        d.each(a.split(aa), function(a, d) {
            b[d]=!0
        });
        return b
    }
    function k(a, b, e) {
        if (e === f && 1 === a.nodeType)
            if (e = "data-" + b.replace(Pb, "-$1").toLowerCase(), e = a.getAttribute(e), "string" === typeof e) {
                try {
                    e = "true" === e?!0 : "false" === e?!1 : "null" === e ? null : + e + "" === e?+e : pc.test(e) ? d.parseJSON(e) : e
                } catch (l) {}
                d.data(a, b, e)
            } else 
                e = f;
        return e
    }
    function n(a) {
        for (var b in a)
            if (("data" !== b ||!d.isEmptyObject(a[b])) && "toJSON" !== b)
                return !1;
        return !0
    }
    function v() {
        return !1
    }
    function t() {
        return !0
    }
    function z(a) {
        return !a ||!a.parentNode || 11 === a.parentNode.nodeType
    }
    function w(a, b) {
        do 
            a = a[b];
        while (a && 1 !== a.nodeType);
        return a
    }
    function B(a, b, e) {
        b = b || 0;
        if (d.isFunction(b))
            return d.grep(a, function(a, d) {
                return !!b.call(a, d, a) === e
            });
        if (b.nodeType)
            return d.grep(a, function(a, d) {
                return a === b === e
            });
        if ("string" === typeof b) {
            var l = d.grep(a, function(a) {
                return 1 === a.nodeType
            });
            if (Qb.test(b))
                return d.filter(b, l, !e);
            b = d.filter(b, l)
        }
        return d.grep(a, function(a, l) {
            return 0 <= d.inArray(a, b) === e
        })
    }
    function E(a) {
        var b =
        ha.split("|");
        a = a.createDocumentFragment();
        if (a.createElement)
            for (; b.length;)
                a.createElement(b.pop());
        return a
    }
    function A(a, b) {
        if (1 === b.nodeType && d.hasData(a)) {
            var e, l, m;
            l = d._data(a);
            var h = d._data(b, l), c = l.events;
            if (c)
                for (e in delete h.handle, h.events = {}, c)
                    for (l = 0, m = c[e].length; l < m; l++)
                        d.event.add(b, e, c[e][l]);
            h.data && (h.data = d.extend({}, h.data))
        }
    }
    function x(a, b) {
        var e;
        1 === b.nodeType && (b.clearAttributes && b.clearAttributes(), b.mergeAttributes && b.mergeAttributes(a), e = b.nodeName.toLowerCase(), "object" ===
        e ? (b.parentNode && (b.outerHTML = a.outerHTML), d.support.html5Clone && a.innerHTML&&!d.trim(b.innerHTML) && (b.innerHTML = a.innerHTML)) : "input" === e && mb.test(a.type) ? (b.defaultChecked = b.checked = a.checked, b.value !== a.value && (b.value = a.value)) : "option" === e ? b.selected = a.defaultSelected : "input" === e || "textarea" === e ? b.defaultValue = a.defaultValue : "script" === e && b.text !== a.text && (b.text = a.text), b.removeAttribute(d.expando))
    }
    function I(a) {
        return "undefined" !== typeof a.getElementsByTagName ? a.getElementsByTagName("*") :
        "undefined" !== typeof a.querySelectorAll ? a.querySelectorAll("*") : []
    }
    function K(a) {
        mb.test(a.type) && (a.defaultChecked = a.checked)
    }
    function D(a, b) {
        if (b in a)
            return b;
        for (var e = b.charAt(0).toUpperCase() + b.slice(1), d = b, m = Ua.length; m--;)
            if (b = Ua[m] + e, b in a)
                return b;
        return d
    }
    function O(a, b) {
        a = b || a;
        return "none" === d.css(a, "display") ||!d.contains(a.ownerDocument, a)
    }
    function W(a, b) {
        for (var e, l, m = [], h = 0, c = a.length; h < c; h++)
            e = a[h], e.style && (m[h] = d._data(e, "olddisplay"), b ? (m[h] || "none" !== e.style.display || (e.style.display =
            ""), "" === e.style.display && O(e) && (m[h] = d._data(e, "olddisplay", G(e.nodeName)))) : (l = da(e, "display"), m[h] || "none" === l || d._data(e, "olddisplay", l)));
        for (h = 0; h < c; h++)
            e = a[h], !e.style || b && "none" !== e.style.display && "" !== e.style.display || (e.style.display = b ? m[h] || "" : "none");
        return a
    }
    function N(a, b, e) {
        return (a = nb.exec(b)) ? Math.max(0, a[1] - (e || 0)) + (a[2] || "px") : b
    }
    function P(a, b, e, l) {
        b = e === (l ? "border" : "content") ? 4 : "width" === b ? 1 : 0;
        for (var m = 0; 4 > b; b += 2)
            "margin" === e && (m += d.css(a, e + ia[b], !0)), l ? ("content" === e && (m -= parseFloat(da(a,
            "padding" + ia[b])) || 0), "margin" !== e && (m -= parseFloat(da(a, "border" + ia[b] + "Width")) || 0)) : (m += parseFloat(da(a, "padding" + ia[b])) || 0, "padding" !== e && (m += parseFloat(da(a, "border" + ia[b] + "Width")) || 0));
        return m
    }
    function Z(a, b, e) {
        var l = "width" === b ? a.offsetWidth: a.offsetHeight, m=!0, h = d.support.boxSizing && "border-box" === d.css(a, "boxSizing");
        if (0 >= l || null == l) {
            l = da(a, b);
            if (0 > l || null == l)
                l = a.style[b];
            if (Na.test(l))
                return l;
            m = h && (d.support.boxSizingReliable || l === a.style[b]);
            l = parseFloat(l) || 0
        }
        return l + P(a, b, e || (h ?
        "border" : "content"), m) + "px"
    }
    function G(a) {
        if (Oa[a])
            return Oa[a];
        var b = d("\x3c" + a + "\x3e").appendTo(C.body), e = b.css("display");
        b.remove();
        if ("none" === e || "" === e)
            Pa = C.body.appendChild(Pa || d.extend(C.createElement("iframe"), {
                frameBorder: 0,
                width: 0,
                height: 0
            })), Va && Pa.createElement || (Va = (Pa.contentWindow || Pa.contentDocument).document, Va.write("\x3c!doctype html\x3e\x3chtml\x3e\x3cbody\x3e"), Va.close()), b = Va.body.appendChild(Va.createElement(a)), e = da(b, "display"), C.body.removeChild(Pa);
        return Oa[a] = e
    }
    function ja(a,
    b, e, l) {
        var m;
        if (d.isArray(b))
            d.each(b, function(b, d) {
                e || pa.test(a) ? l(a, d) : ja(a + "[" + ("object" === typeof d ? b : "") + "]", d, e, l)
            });
        else if (e || "object" !== d.type(b))
            l(a, b);
        else 
            for (m in b)
                ja(a + "[" + m + "]", b[m], e, l)
    }
    function U(a) {
        return function(b, e) {
            "string" !== typeof b && (e = b, b = "*");
            var l, m, h = b.toLowerCase().split(aa), c = 0, q = h.length;
            if (d.isFunction(e))
                for (; c < q; c++)
                    l = h[c], (m = /^\+/.test(l)) && (l = l.substr(1) || "*"), l = a[l] = a[l] || [], l[m ? "unshift": "push"](e)
        }
    }
    function R(a, b, e, d, m, h) {
        m = m || b.dataTypes[0];
        h = h || {};
        h[m]=!0;
        var c;
        m = a[m];
        for (var q = 0, p = m ? m.length : 0, r = a === Fb; q < p && (r ||!c); q++)
            c = m[q](b, e, d), "string" === typeof c && (!r || h[c] ? c = f : (b.dataTypes.unshift(c), c = R(a, b, e, d, c, h)));
        !r && c || h["*"] || (c = R(a, b, e, d, "*", h));
        return c
    }
    function qa(a, b) {
        var e, l, m = d.ajaxSettings.flatOptions || {};
        for (e in b)
            b[e] !== f && ((m[e] ? a : l || (l = {}))[e] = b[e]);
        l && d.extend(!0, a, l)
    }
    function V() {
        try {
            return new c.XMLHttpRequest
        } catch (a) {}
    }
    function Za() {
        setTimeout(function() {
            ob = f
        }, 0);
        return ob = d.now()
    }
    function L(a, b) {
        d.each(b, function(b, d) {
            for (var m = (Qa[b] ||
            []).concat(Qa["*"]), h = 0, c = m.length; h < c&&!m[h].call(a, b, d); h++);
        })
    }
    function M(a, b, e) {
        var l = 0, m = pb.length, c = d.Deferred().always(function() {
            delete u.elem
        }), u = function() {
            for (var b = ob || Za(), b = Math.max(0, q.startTime + q.duration - b), e = 1 - (b / q.duration || 0), d = 0, l = q.tweens.length; d < l; d++)
                q.tweens[d].run(e);
            c.notifyWith(a, [q, e, b]);
            if (1 > e && l)
                return b;
            c.resolveWith(a, [q]);
            return !1
        }, q = c.promise({
            elem: a,
            props: d.extend({}, b),
            opts: d.extend(!0, {
                specialEasing: {}
            }, e),
            originalProperties: b,
            originalOptions: e,
            startTime: ob ||
            Za(),
            duration: e.duration,
            tweens: [],
            createTween: function(b, e, s) {
                b = d.Tween(a, q.opts, b, e, q.opts.specialEasing[b] || q.opts.easing);
                q.tweens.push(b);
                return b
            },
            stop: function(b) {
                for (var e = 0, d = b ? q.tweens.length : 0; e < d; e++)
                    q.tweens[e].run(1);
                b ? c.resolveWith(a, [q, b]) : c.rejectWith(a, [q, b]);
                return this
            }
        });
        e = q.props;
        for (ga(e, q.opts.specialEasing);
        l < m;
        l++)if (b = pb[l].call(q, a, e, q.opts))
            return b;
        L(q, e);
        d.isFunction(q.opts.start) && q.opts.start.call(a, q);
        d.fx.timer(d.extend(u, {
            anim: q,
            queue: q.opts.queue,
            elem: a
        }));
        return q.progress(q.opts.progress).done(q.opts.done,
        q.opts.complete).fail(q.opts.fail).always(q.opts.always)
    }
    function ga(a, b) {
        var e, l, m, c, u;
        for (e in a)
            if (l = d.camelCase(e), m = b[l], c = a[e], d.isArray(c) && (m = c[1], c = a[e] = c[0]), e !== l && (a[l] = c, delete a[e]), (u = d.cssHooks[l]) && "expand"in u)
                for (e in c = u.expand(c), delete a[l], c)
                    e in a || (a[e] = c[e], b[e] = m);
            else 
                b[l] = m
    }
    function Q(a, b, e, d, c) {
        return new Q.prototype.init(a, b, e, d, c)
    }
    function T(a, b) {
        var e, d = {
            height: a
        }, c = 0;
        for (b = b ? 1 : 0; 4 > c; c += 2 - b)
            e = ia[c], d["margin" + e] = d["padding" + e] = a;
        b && (d.opacity = d.width = a);
        return d
    }
    function ka(a) {
        return d.isWindow(a) ?
        a : 9 === a.nodeType ? a.defaultView || a.parentWindow : !1
    }
    var Ha, Aa, C = c.document, Ia = c.location, Ra = c.navigator, ya = c.jimdoGen002, X = c.$, Y = Array.prototype.push, $ = Array.prototype.slice, la = Array.prototype.indexOf, va = Object.prototype.toString, Ja = Object.prototype.hasOwnProperty, ba = String.prototype.trim, d = function(a, b) {
        return new d.fn.init(a, b, Ha)
    }, ra = /[\-+]?(?:\d*\.|)\d+(?:[eE][\-+]?\d+|)/.source, ma = /\S/, aa = /\s+/, na = /^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g, wa = /^(?:[^#<]*(<[\w\W]+>)[^>]*$|#([\w\-]*)$)/, Ka = /^<(\w+)\s*\/?>(?:<\/\1>|)$/,
    ca = /^[\],:{}\s]*$/, p = /(?:^|:|,)(?:\s*\[)+/g, qb = /\\(?:["\\\/bfnrt]|u[\da-fA-F]{4})/g, Sa = /"[^"\\\r\n]*"|true|false|null|-?(?:\d\d*\.|)\d+(?:[eE][\-+]?\d+|)/g, rc = /^-ms-/, sc = /-([\da-z])/gi, xa = function(a, b) {
        return (b + "").toUpperCase()
    }, Rb = function() {
        C.addEventListener ? (C.removeEventListener("DOMContentLoaded", Rb, !1), d.ready()) : "complete" === C.readyState && (C.detachEvent("onreadystatechange", Rb), d.ready())
    }, nc = {};
    d.fn = d.prototype = {
        constructor: d,
        init: function(a, b, e) {
            var l;
            if (!a)
                return this;
            if (a.nodeType)
                return this.context =
                this[0] = a, this.length = 1, this;
            if ("string" === typeof a) {
                l = "\x3c" === a.charAt(0) && "\x3e" === a.charAt(a.length - 1) && 3 <= a.length ? [null, a, null] : wa.exec(a);
                if (!l ||!l[1] && b)
                    return !b || b.jquery ? (b || e).find(a) : this.constructor(b).find(a);
                if (l[1])
                    return a = (b = b instanceof d ? b[0] : b) && b.nodeType ? b.ownerDocument || b : C, a = d.parseHTML(l[1], a, !0), Ka.test(l[1]) && d.isPlainObject(b) && this.attr.call(a, b, !0), d.merge(this, a);
                if ((b = C.getElementById(l[2])) && b.parentNode) {
                    if (b.id !== l[2])
                        return e.find(a);
                    this.length = 1;
                    this[0] = b
                }
                this.context =
                C;
                this.selector = a;
                return this
            }
            if (d.isFunction(a))
                return e.ready(a);
            a.selector !== f && (this.selector = a.selector, this.context = a.context);
            return d.makeArray(a, this)
        },
        selector: "",
        jquery: "1.8.1",
        length: 0,
        size: function() {
            return this.length
        },
        toArray: function() {
            return $.call(this)
        },
        get: function(a) {
            return null == a ? this.toArray() : 0 > a ? this[this.length + a] : this[a]
        },
        pushStack: function(a, b, e) {
            a = d.merge(this.constructor(), a);
            a.prevObject = this;
            a.context = this.context;
            "find" === b ? a.selector = this.selector + (this.selector ?
            " " : "") + e : b && (a.selector = this.selector + "." + b + "(" + e + ")");
            return a
        },
        each: function(a, b) {
            return d.each(this, a, b)
        },
        ready: function(a) {
            d.ready.promise().done(a);
            return this
        },
        eq: function(a) {
            a =+ a;
            return - 1 === a ? this.slice(a) : this.slice(a, a + 1)
        },
        first: function() {
            return this.eq(0)
        },
        last: function() {
            return this.eq( - 1)
        },
        slice: function() {
            return this.pushStack($.apply(this, arguments), "slice", $.call(arguments).join(","))
        },
        map: function(a) {
            return this.pushStack(d.map(this, function(b, e) {
                return a.call(b, e, b)
            }))
        },
        end: function() {
            return this.prevObject ||
            this.constructor(null)
        },
        push: Y,
        sort: [].sort,
        splice: [].splice
    };
    d.fn.init.prototype = d.fn;
    d.extend = d.fn.extend = function() {
        var a, b, e, l, c, h = arguments[0] || {}, u = 1, q = arguments.length, p=!1;
        "boolean" === typeof h && (p = h, h = arguments[1] || {}, u = 2);
        "object" === typeof h || d.isFunction(h) || (h = {});
        q === u && (h = this, --u);
        for (; u < q; u++)
            if (null != (a = arguments[u]))
                for (b in a)
                    e = h[b], l = a[b], h !== l && (p && l && (d.isPlainObject(l) || (c = d.isArray(l))) ? (c ? (c=!1, e = e && d.isArray(e) ? e : []) : e = e && d.isPlainObject(e) ? e : {}, h[b] = d.extend(p, e, l)) : l !==
                    f && (h[b] = l));
        return h
    };
    d.extend({
        noConflict: function(a) {
            c.$ === d && (c.$ = X);
            a && c.jimdoGen002 === d && (c.jimdoGen002 = ya);
            return d
        },
        isReady: !1,
        readyWait: 1,
        holdReady: function(a) {
            a ? d.readyWait++ : d.ready(!0)
        },
        ready: function(a) {
            if (!0 === a?!--d.readyWait : !d.isReady) {
                if (!C.body)
                    return setTimeout(d.ready, 1);
                d.isReady=!0;
                !0 !== a && 0<--d.readyWait || (Aa.resolveWith(C, [d]), d.fn.trigger && d(C).trigger("ready").off("ready"))
            }
        },
        isFunction: function(a) {
            return "function" === d.type(a)
        },
        isArray: Array.isArray || function(a) {
            return "array" === d.type(a)
        },
        isWindow: function(a) {
            return null != a && a == a.window
        },
        isNumeric: function(a) {
            return !isNaN(parseFloat(a)) && isFinite(a)
        },
        type: function(a) {
            return null == a ? String(a) : nc[va.call(a)] || "object"
        },
        isPlainObject: function(a) {
            if (!a || "object" !== d.type(a) || a.nodeType || d.isWindow(a))
                return !1;
            try {
                if (a.constructor&&!Ja.call(a, "constructor")&&!Ja.call(a.constructor.prototype, "isPrototypeOf"))
                    return !1
            } catch (b) {
                return !1
            }
            for (var e in a);
            return e === f || Ja.call(a, e)
        },
        isEmptyObject: function(a) {
            for (var b in a)
                return !1;
            return !0
        },
        error: function(a) {
            throw Error(a);
        },
        parseHTML: function(a, b, e) {
            var l;
            if (!a || "string" !== typeof a)
                return null;
            "boolean" === typeof b && (e = b, b = 0);
            b = b || C;
            if (l = Ka.exec(a))
                return [b.createElement(l[1])];
            l = d.buildFragment([a], b, e ? null : []);
            return d.merge([], (l.cacheable ? d.clone(l.fragment) : l.fragment).childNodes)
        },
        parseJSON: function(a) {
            if (!a || "string" !== typeof a)
                return null;
            a = d.trim(a);
            if (c.JSON && c.JSON.parse)
                return c.JSON.parse(a);
            if (ca.test(a.replace(qb, "@").replace(Sa, "]").replace(p, "")))
                return (new Function("return " +
                a))();
            d.error("Invalid JSON: " + a)
        },
        parseXML: function(a) {
            var b, e;
            if (!a || "string" !== typeof a)
                return null;
            try {
                c.DOMParser ? (e = new DOMParser, b = e.parseFromString(a, "text/xml")) : (b = new ActiveXObject("Microsoft.XMLDOM"), b.async = "false", b.loadXML(a))
            } catch (l) {
                b = f
            }
            b && b.documentElement&&!b.getElementsByTagName("parsererror").length || d.error("Invalid XML: " + a);
            return b
        },
        noop: function() {},
        globalEval: function(a) {
            a && ma.test(a) && (c.execScript || function(a) {
                c.eval.call(c, a)
            })(a)
        },
        camelCase: function(a) {
            return a.replace(rc,
            "ms-").replace(sc, xa)
        },
        nodeName: function(a, b) {
            return a.nodeName && a.nodeName.toUpperCase() === b.toUpperCase()
        },
        each: function(a, b, e) {
            var l, c = 0, h = a.length, u = h === f || d.isFunction(a);
            if (e)
                if (u)
                    for (l in a) {
                        if (!1 === b.apply(a[l], e))
                            break
                    } else 
                        for (; c < h&&!1 !== b.apply(a[c++], e););
                    else if (u)
                for (l in a) {
                    if (!1 === b.call(a[l], l, a[l]))
                        break
                } else 
                    for (; c < h&&!1 !== b.call(a[c], c, a[c++]););
            return a
        },
        trim: ba&&!ba.call("\ufeff\u00a0") ? function(a) {
            return null == a ? "" : ba.call(a)
        }
        : function(a) {
            return null == a ? "" : a.toString().replace(na,
            "")
        },
        makeArray: function(a, b) {
            var e, l = b || [];
            null != a && (e = d.type(a), null == a.length || "string" === e || "function" === e || "regexp" === e || d.isWindow(a) ? Y.call(l, a) : d.merge(l, a));
            return l
        },
        inArray: function(a, b, e) {
            var d;
            if (b) {
                if (la)
                    return la.call(b, a, e);
                d = b.length;
                for (e = e ? 0 > e ? Math.max(0, d + e) : e : 0; e < d; e++)
                    if (e in b && b[e] === a)
                        return e
            }
            return - 1
        },
        merge: function(a, b) {
            var e = b.length, d = a.length, c = 0;
            if ("number" === typeof e)
                for (; c < e; c++)
                    a[d++] = b[c];
            else 
                for (; b[c] !== f;)
                    a[d++] = b[c++];
            a.length = d;
            return a
        },
        grep: function(a, b, e) {
            var d,
            c = [], h = 0, u = a.length;
            for (e=!!e; h < u; h++)
                d=!!b(a[h], h), e !== d && c.push(a[h]);
            return c
        },
        map: function(a, b, e) {
            var l, c, h = [], u = 0, q = a.length;
            if (a instanceof d || q !== f && "number" === typeof q && (0 < q && a[0] && a[q - 1] || 0 === q || d.isArray(a)))
                for (; u < q; u++)
                    l = b(a[u], u, e), null != l && (h[h.length] = l);
            else 
                for (c in a)
                    l = b(a[c], c, e), null != l && (h[h.length] = l);
            return h.concat.apply([], h)
        },
        guid: 1,
        proxy: function(a, b) {
            var e, l;
            "string" === typeof b && (e = a[b], b = a, a = e);
            if (!d.isFunction(a))
                return f;
            l = $.call(arguments, 2);
            e = function() {
                return a.apply(b,
                l.concat($.call(arguments)))
            };
            e.guid = a.guid = a.guid || e.guid || d.guid++;
            return e
        },
        access: function(a, b, e, l, c, h, u) {
            var q, p = null == e, r = 0, s = a.length;
            if (e && "object" === typeof e) {
                for (r in e)
                    d.access(a, b, r, e[r], 1, h, l);
                c = 1
            } else if (l !== f) {
                q = u === f && d.isFunction(l);
                p && (q ? (q = b, b = function(a, b, e) {
                    return q.call(d(a), e)
                }) : (b.call(a, l), b = null));
                if (b)
                    for (; r < s; r++)
                        b(a[r], e, q ? l.call(a[r], r, b(a[r], e)) : l, u);
                c = 1
            }
            return c ? a : p ? b.call(a) : s ? b(a[0], e) : h
        },
        now: function() {
            return (new Date).getTime()
        }
    });
    d.ready.promise = function(a) {
        if (!Aa)
            if (Aa =
            d.Deferred(), "complete" === C.readyState)
                setTimeout(d.ready, 1);
            else if (C.addEventListener)
                C.addEventListener("DOMContentLoaded", Rb, !1), c.addEventListener("load", d.ready, !1);
            else {
                C.attachEvent("onreadystatechange", Rb);
                c.attachEvent("onload", d.ready);
                var b=!1;
                try {
                    b = null == c.frameElement && C.documentElement
                } catch (e) {}
                b && b.doScroll && function m() {
                    if (!d.isReady) {
                        try {
                            b.doScroll("left")
                        } catch (a) {
                            return setTimeout(m, 50)
                        }
                        d.ready()
                    }
                }()
            }
        return Aa.promise(a)
    };
    d.each("Boolean Number String Function Array Date RegExp Object".split(" "),
    function(a, b) {
        nc["[object " + b + "]"] = b.toLowerCase()
    });
    Ha = d(C);
    var Ga = {};
    d.Callbacks = function(a) {
        a = "string" === typeof a ? Ga[a] || g(a) : d.extend({}, a);
        var b, e, l, c, h, u, q = [], p=!a.once && [], r = function(d) {
            b = a.memory && d;
            e=!0;
            u = c || 0;
            c = 0;
            h = q.length;
            for (l=!0; q && u < h; u++)
                if (!1 === q[u].apply(d[0], d[1]) && a.stopOnFalse) {
                    b=!1;
                    break
                }
            l=!1;
            q && (p ? p.length && r(p.shift()) : b ? q = [] : s.disable())
        }, s = {
            add: function() {
                if (q) {
                    var e = q.length;
                    (function F(b) {
                        d.each(b, function(b, e) {
                            var r = d.type(e);
                            "function" !== r || a.unique && s.has(e) ? e && e.length &&
                            "string" !== r && F(e) : q.push(e)
                        })
                    })(arguments);
                    l ? h = q.length : b && (c = e, r(b))
                }
                return this
            },
            remove: function() {
                q && d.each(arguments, function(a, b) {
                    for (var e; - 1 < (e = d.inArray(b, q, e));)
                        q.splice(e, 1), l && (e <= h && h--, e <= u && u--)
                });
                return this
            },
            has: function(a) {
                return - 1 < d.inArray(a, q)
            },
            empty: function() {
                q = [];
                return this
            },
            disable: function() {
                q = p = b = f;
                return this
            },
            disabled: function() {
                return !q
            },
            lock: function() {
                p = f;
                b || s.disable();
                return this
            },
            locked: function() {
                return !p
            },
            fireWith: function(a, b) {
                b = b || [];
                b = [a, b.slice ? b.slice():
                b];
                !q || e&&!p || (l ? p.push(b) : r(b));
                return this
            },
            fire: function() {
                s.fireWith(this, arguments);
                return this
            },
            fired: function() {
                return !!e
            }
        };
        return s
    };
    d.extend({
        Deferred: function(a) {
            var b = [["resolve", "done", d.Callbacks("once memory"), "resolved"], ["reject", "fail", d.Callbacks("once memory"), "rejected"], ["notify", "progress", d.Callbacks("memory")]], e = "pending", l = {
                state: function() {
                    return e
                },
                always: function() {
                    c.done(arguments).fail(arguments);
                    return this
                },
                then: function() {
                    var a = arguments;
                    return d.Deferred(function(e) {
                        d.each(b,
                        function(b, l) {
                            var r = l[0], s = a[b];
                            c[l[1]](d.isFunction(s) ? function() {
                                var a = s.apply(this, arguments);
                                if (a && d.isFunction(a.promise))
                                    a.promise().done(e.resolve).fail(e.reject).progress(e.notify);
                                else 
                                    e[r + "With"](this === c ? e : this, [a])
                            } : e[r])
                        });
                        a = null
                    }).promise()
                },
                promise: function(a) {
                    return "object" === typeof a ? d.extend(a, l) : l
                }
            }, c = {};
            l.pipe = l.then;
            d.each(b, function(a, d) {
                var q = d[2], f = d[3];
                l[d[1]] = q.add;
                f && q.add(function() {
                    e = f
                }, b[a^1][2].disable, b[2][2].lock);
                c[d[0]] = q.fire;
                c[d[0] + "With"] = q.fireWith
            });
            l.promise(c);
            a && a.call(c, c);
            return c
        },
        when: function(a) {
            var b = 0, e = $.call(arguments), l = e.length, c = 1 !== l || a && d.isFunction(a.promise) ? l: 0, h = 1 === c ? a: d.Deferred(), u = function(a, b, e) {
                return function(d) {
                    b[a] = this;
                    e[a] = 1 < arguments.length ? $.call(arguments) : d;
                    e === q ? h.notifyWith(b, e) : --c || h.resolveWith(b, e)
                }
            }, q, f, r;
            if (1 < l)
                for (q = Array(l), f = Array(l), r = Array(l); b < l; b++)
                    e[b] && d.isFunction(e[b].promise) ? e[b].promise().done(u(b, r, e)).fail(h.reject).progress(u(b, f, q)) : --c;
            c || h.resolveWith(r, e);
            return h.promise()
        }
    });
    d.support = function() {
        var a,
        b, e, l, m, h, u, q = C.createElement("div");
        q.setAttribute("className", "t");
        q.innerHTML = "  \x3clink/\x3e\x3ctable\x3e\x3c/table\x3e\x3ca href\x3d'/a'\x3ea\x3c/a\x3e\x3cinput type\x3d'checkbox'/\x3e";
        b = q.getElementsByTagName("*");
        e = q.getElementsByTagName("a")[0];
        e.style.cssText = "top:1px;float:left;opacity:.5";
        if (!b ||!b.length ||!e)
            return {};
        l = C.createElement("select");
        m = l.appendChild(C.createElement("option"));
        b = q.getElementsByTagName("input")[0];
        a = {
            leadingWhitespace: 3 === q.firstChild.nodeType,
            tbody: !q.getElementsByTagName("tbody").length,
            htmlSerialize: !!q.getElementsByTagName("link").length,
            style: /top/.test(e.getAttribute("style")),
            hrefNormalized: "/a" === e.getAttribute("href"),
            opacity: /^0.5/.test(e.style.opacity),
            cssFloat: !!e.style.cssFloat,
            checkOn: "on" === b.value,
            optSelected: m.selected,
            getSetAttribute: "t" !== q.className,
            enctype: !!C.createElement("form").enctype,
            html5Clone: "\x3c:nav\x3e\x3c/:nav\x3e" !== C.createElement("nav").cloneNode(!0).outerHTML,
            boxModel: "CSS1Compat" === C.compatMode,
            submitBubbles: !0,
            changeBubbles: !0,
            focusinBubbles: !1,
            deleteExpando: !0,
            noCloneEvent: !0,
            inlineBlockNeedsLayout: !1,
            shrinkWrapBlocks: !1,
            reliableMarginRight: !0,
            boxSizingReliable: !0,
            pixelPosition: !1
        };
        b.checked=!0;
        a.noCloneChecked = b.cloneNode(!0).checked;
        l.disabled=!0;
        a.optDisabled=!m.disabled;
        try {
            delete q.test
        } catch (f) {
            a.deleteExpando=!1
        }
        !q.addEventListener && q.attachEvent && q.fireEvent && (q.attachEvent("onclick", e = function() {
            a.noCloneEvent=!1
        }), q.cloneNode(!0).fireEvent("onclick"), q.detachEvent("onclick", e));
        b = C.createElement("input");
        b.value = "t";
        b.setAttribute("type",
        "radio");
        a.radioValue = "t" === b.value;
        b.setAttribute("checked", "checked");
        b.setAttribute("name", "t");
        q.appendChild(b);
        e = C.createDocumentFragment();
        e.appendChild(q.lastChild);
        a.checkClone = e.cloneNode(!0).cloneNode(!0).lastChild.checked;
        a.appendChecked = b.checked;
        e.removeChild(b);
        e.appendChild(q);
        if (q.attachEvent)
            for (h in{
                submit: !0,
                change: !0,
                focusin: !0
            })
                b = "on" + h, u = b in q, u || (q.setAttribute(b, "return;"), u = "function" === typeof q[b]), a[h + "Bubbles"] = u;
        d(function() {
            var b, e, d, l = C.getElementsByTagName("body")[0];
            l && (b = C.createElement("div"), b.style.cssText = "visibility:hidden;border:0;width:0;height:0;position:static;top:0;margin-top:1px", l.insertBefore(b, l.firstChild), e = C.createElement("div"), b.appendChild(e), e.innerHTML = "\x3ctable\x3e\x3ctr\x3e\x3ctd\x3e\x3c/td\x3e\x3ctd\x3et\x3c/td\x3e\x3c/tr\x3e\x3c/table\x3e", d = e.getElementsByTagName("td"), d[0].style.cssText = "padding:0;margin:0;border:0;display:none", u = 0 === d[0].offsetHeight, d[0].style.display = "", d[1].style.display = "none", a.reliableHiddenOffsets = u && 0 ===
            d[0].offsetHeight, e.innerHTML = "", e.style.cssText = "box-sizing:border-box;-moz-box-sizing:border-box;-webkit-box-sizing:border-box;padding:1px;border:1px;display:block;width:4px;margin-top:1%;position:absolute;top:1%;", a.boxSizing = 4 === e.offsetWidth, a.doesNotIncludeMarginInBodyOffset = 1 !== l.offsetTop, c.getComputedStyle && (a.pixelPosition = "1%" !== (c.getComputedStyle(e, null) || {}).top, a.boxSizingReliable = "4px" === (c.getComputedStyle(e, null) || {
                width: "4px"
            }).width, d = C.createElement("div"), d.style.cssText = e.style.cssText =
            "padding:0;margin:0;border:0;display:block;overflow:hidden;", d.style.marginRight = d.style.width = "0", e.style.width = "1px", e.appendChild(d), a.reliableMarginRight=!parseFloat((c.getComputedStyle(d, null) || {}).marginRight)), "undefined" !== typeof e.style.zoom && (e.innerHTML = "", e.style.cssText = "padding:0;margin:0;border:0;display:block;overflow:hidden;width:1px;padding:1px;display:inline;zoom:1", a.inlineBlockNeedsLayout = 3 === e.offsetWidth, e.style.display = "block", e.style.overflow = "visible", e.innerHTML = "\x3cdiv\x3e\x3c/div\x3e",
            e.firstChild.style.width = "5px", a.shrinkWrapBlocks = 3 !== e.offsetWidth, b.style.zoom = 1), l.removeChild(b))
        });
        e.removeChild(q);
        b = e = l = m = b = e = q = null;
        return a
    }();
    var pc = /(?:\{[\s\S]*\}|\[[\s\S]*\])$/, Pb = /([A-Z])/g;
    d.extend({
        cache: {},
        deletedIds: [],
        uuid: 0,
        expando: "jimdoGen002" + (d.fn.jquery + Math.random()).replace(/\D/g, ""),
        noData: {
            embed: !0,
            object: "clsid:D27CDB6E-AE6D-11cf-96B8-444553540000",
            applet: !0
        },
        hasData: function(a) {
            a = a.nodeType ? d.cache[a[d.expando]] : a[d.expando];
            return !!a&&!n(a)
        },
        data: function(a, b, e, l) {
            if (d.acceptData(a)) {
                var c =
                d.expando, h = "string" === typeof b, u = a.nodeType, q = u ? d.cache: a, p = u ? a[c]: a[c] && c;
                if (p && q[p] && (l || q[p].data) ||!h || e !== f) {
                    p || (u ? a[c] = p = d.deletedIds.pop()||++d.uuid : p = c);
                    q[p] || (q[p] = {}, u || (q[p].toJSON = d.noop));
                    if ("object" === typeof b || "function" === typeof b)
                        l ? q[p] = d.extend(q[p], b) : q[p].data = d.extend(q[p].data, b);
                    a = q[p];
                    l || (a.data || (a.data = {}), a = a.data);
                    e !== f && (a[d.camelCase(b)] = e);
                    h ? (e = a[b], null == e && (e = a[d.camelCase(b)])) : e = a;
                    return e
                }
            }
        },
        removeData: function(a, b, e) {
            if (d.acceptData(a)) {
                var l, c, h, u = a.nodeType,
                q = u ? d.cache: a, f = u ? a[d.expando]: d.expando;
                if (q[f]) {
                    if (b && (l = e ? q[f] : q[f].data)) {
                        d.isArray(b) || (b in l ? b = [b] : (b = d.camelCase(b), b = b in l ? [b] : b.split(" ")));
                        c = 0;
                        for (h = b.length; c < h; c++)
                            delete l[b[c]];
                        if (!(e ? n : d.isEmptyObject)(l))
                            return 
                    }
                    if (!e && (delete q[f].data, !n(q[f])))
                        return;
                    u ? d.cleanData([a], !0) : d.support.deleteExpando || q != q.window ? delete q[f] : q[f] = null
                }
            }
        },
        _data: function(a, b, e) {
            return d.data(a, b, e, !0)
        },
        acceptData: function(a) {
            var b = a.nodeName && d.noData[a.nodeName.toLowerCase()];
            return !b ||!0 !== b && a.getAttribute("classid") ===
            b
        }
    });
    d.fn.extend({
        data: function(a, b) {
            var e, l, c, h, u, q = this[0], p = 0, r = null;
            if (a === f) {
                if (this.length && (r = d.data(q), 1 === q.nodeType&&!d._data(q, "parsedAttrs"))) {
                    c = q.attributes;
                    for (u = c.length; p < u; p++)
                        h = c[p].name, 0 === h.indexOf("data-") && (h = d.camelCase(h.substring(5)), k(q, h, r[h]));
                    d._data(q, "parsedAttrs", !0)
                }
                return r
            }
            if ("object" === typeof a)
                return this.each(function() {
                    d.data(this, a)
                });
            e = a.split(".", 2);
            e[1] = e[1] ? "." + e[1] : "";
            l = e[1] + "!";
            return d.access(this, function(b) {
                if (b === f)
                    return r = this.triggerHandler("getData" +
                    l, [e[0]]), r === f && q && (r = d.data(q, a), r = k(q, a, r)), r === f && e[1] ? this.data(e[0]) : r;
                e[1] = b;
                this.each(function() {
                    var r = d(this);
                    r.triggerHandler("setData" + l, e);
                    d.data(this, a, b);
                    r.triggerHandler("changeData" + l, e)
                })
            }, null, b, 1 < arguments.length, null, !1)
        },
        removeData: function(a) {
            return this.each(function() {
                d.removeData(this, a)
            })
        }
    });
    d.extend({
        queue: function(a, b, e) {
            var l;
            if (a)
                return b = (b || "fx") + "queue", l = d._data(a, b), e && (!l || d.isArray(e) ? l = d._data(a, b, d.makeArray(e)) : l.push(e)), l || []
        },
        dequeue: function(a, b) {
            b = b || "fx";
            var e = d.queue(a, b), l = e.length, c = e.shift(), h = d._queueHooks(a, b), u = function() {
                d.dequeue(a, b)
            };
            "inprogress" === c && (c = e.shift(), l--);
            c && ("fx" === b && e.unshift("inprogress"), delete h.stop, c.call(a, u, h));
            !l && h && h.empty.fire()
        },
        _queueHooks: function(a, b) {
            var e = b + "queueHooks";
            return d._data(a, e) || d._data(a, e, {
                empty: d.Callbacks("once memory").add(function() {
                    d.removeData(a, b + "queue", !0);
                    d.removeData(a, e, !0)
                })
            })
        }
    });
    d.fn.extend({
        queue: function(a, b) {
            var e = 2;
            "string" !== typeof a && (b = a, a = "fx", e--);
            return arguments.length <
            e ? d.queue(this[0], a) : b === f ? this : this.each(function() {
                var e = d.queue(this, a, b);
                d._queueHooks(this, a);
                "fx" === a && "inprogress" !== e[0] && d.dequeue(this, a)
            })
        },
        dequeue: function(a) {
            return this.each(function() {
                d.dequeue(this, a)
            })
        },
        delay: function(a, b) {
            a = d.fx ? d.fx.speeds[a] || a : a;
            return this.queue(b || "fx", function(b, d) {
                var c = setTimeout(b, a);
                d.stop = function() {
                    clearTimeout(c)
                }
            })
        },
        clearQueue: function(a) {
            return this.queue(a || "fx", [])
        },
        promise: function(a, b) {
            var e, l = 1, c = d.Deferred(), h = this, u = this.length, q = function() {
                --l ||
                c.resolveWith(h, [h])
            };
            "string" !== typeof a && (b = a, a = f);
            for (a = a || "fx"; u--;)(e = d._data(h[u], a + "queueHooks")
                ) && e.empty && (l++, e.empty.add(q));
            q();
            return c.promise(b)
        }
    });
    var Da, rb, sb, Ba = /[\t\r\n]/g, gc = /\r/g, tb = /^(?:button|input)$/i, Ta = /^(?:button|input|object|select|textarea)$/i, Sb = /^a(?:rea|)$/i, Tb = /^(?:autofocus|autoplay|async|checked|controls|defer|disabled|hidden|loop|multiple|open|readonly|required|scoped|selected)$/i, db = d.support.getSetAttribute;
    d.fn.extend({
        attr: function(a, b) {
            return d.access(this,
            d.attr, a, b, 1 < arguments.length)
        },
        removeAttr: function(a) {
            return this.each(function() {
                d.removeAttr(this, a)
            })
        },
        prop: function(a, b) {
            return d.access(this, d.prop, a, b, 1 < arguments.length)
        },
        removeProp: function(a) {
            a = d.propFix[a] || a;
            return this.each(function() {
                try {
                    this[a] = f, delete this[a]
                } catch (b) {}
            })
        },
        addClass: function(a) {
            var b, e, l, c, h, u, f;
            if (d.isFunction(a))
                return this.each(function(b) {
                    d(this).addClass(a.call(this, b, this.className))
                });
            if (a && "string" === typeof a)
                for (b = a.split(aa), e = 0, l = this.length; e < l; e++)
                    if (c =
                    this[e], 1 === c.nodeType)
                        if (c.className || 1 !== b.length) {
                            h = " " + c.className + " ";
                            u = 0;
                            for (f = b.length; u < f; u++)
                                ~h.indexOf(" " + b[u] + " ") || (h += b[u] + " ");
                                c.className = d.trim(h)
                        } else 
                            c.className = a;
            return this
        },
        removeClass: function(a) {
            var b, e, l, c, h, u, q;
            if (d.isFunction(a))
                return this.each(function(b) {
                    d(this).removeClass(a.call(this, b, this.className))
                });
            if (a && "string" === typeof a || a === f)
                for (b = (a || "").split(aa), u = 0, q = this.length; u < q; u++)
                    if (l = this[u], 1 === l.nodeType && l.className) {
                        e = (" " + l.className + " ").replace(Ba, " ");
                        c = 0;
                        for (h = b.length; c < h; c++)
                            for (; - 1 < e.indexOf(" " + b[c] + " ");)
                                e = e.replace(" " + b[c] + " ", " ");
                                l.className = a ? d.trim(e) : ""
                    }
            return this
        },
        toggleClass: function(a, b) {
            var e = typeof a, l = "boolean" === typeof b;
            return d.isFunction(a) ? this.each(function(e) {
                d(this).toggleClass(a.call(this, e, this.className, b), b)
            }) : this.each(function() {
                if ("string" === e)
                    for (var c, h = 0, u = d(this), f = b, p = a.split(aa); c = p[h++];)
                        f = l ? f : !u.hasClass(c), u[f ? "addClass": "removeClass"](c);
                else if ("undefined" === e || "boolean" === e)
                    this.className && d._data(this,
                    "__className__", this.className), this.className = this.className ||!1 === a ? "" : d._data(this, "__className__") || ""
            })
        },
        hasClass: function(a) {
            a = " " + a + " ";
            for (var b = 0, e = this.length; b < e; b++)
                if (1 === this[b].nodeType&&-1 < (" " + this[b].className + " ").replace(Ba, " ").indexOf(a))
                    return !0;
            return !1
        },
        val: function(a) {
            var b, e, l, c = this[0];
            if (arguments.length)
                return l = d.isFunction(a), this.each(function(e) {
                    var c = d(this);
                    1 === this.nodeType && (e = l ? a.call(this, e, c.val()) : a, null == e ? e = "" : "number" === typeof e ? e += "" : d.isArray(e) && (e =
                    d.map(e, function(a) {
                        return null == a ? "" : a + ""
                    })), b = d.valHooks[this.type] || d.valHooks[this.nodeName.toLowerCase()], b && "set"in b && b.set(this, e, "value") !== f || (this.value = e))
                });
            if (c) {
                if ((b = d.valHooks[c.type] || d.valHooks[c.nodeName.toLowerCase()]) && "get"in b && (e = b.get(c, "value")) !== f)
                    return e;
                e = c.value;
                return "string" === typeof e ? e.replace(gc, "") : null == e ? "" : e
            }
        }
    });
    d.extend({
        valHooks: {
            option: {
                get: function(a) {
                    var b = a.attributes.value;
                    return !b || b.specified ? a.value : a.text
                }
            },
            select: {
                get: function(a) {
                    var b, e, c = a.selectedIndex,
                    m = [], h = a.options, u = "select-one" === a.type;
                    if (0 > c)
                        return null;
                    a = u ? c : 0;
                    for (e = u ? c + 1 : h.length; a < e; a++)
                        if (b = h[a], b.selected&&!((d.support.optDisabled ? b.disabled : null !== b.getAttribute("disabled")) || b.parentNode.disabled && d.nodeName(b.parentNode, "optgroup"))) {
                            b = d(b).val();
                            if (u)
                                return b;
                                m.push(b)
                        }
                    return u&&!m.length && h.length ? d(h[c]).val() : m
                },
                set: function(a, b) {
                    var e = d.makeArray(b);
                    d(a).find("option").each(function() {
                        this.selected = 0 <= d.inArray(d(this).val(), e)
                    });
                    e.length || (a.selectedIndex =- 1);
                    return e
                }
            }
        },
        attrFn: {},
        attr: function(a, b, e, c) {
            var m, h, u = a.nodeType;
            if (a && 3 !== u && 8 !== u && 2 !== u) {
                if (c && d.isFunction(d.fn[b]))
                    return d(a)[b](e);
                if ("undefined" === typeof a.getAttribute)
                    return d.prop(a, b, e);
                if (c = 1 !== u ||!d.isXMLDoc(a))
                    b = b.toLowerCase(), h = d.attrHooks[b] || (Tb.test(b) ? rb : Da);
                if (e !== f)
                    if (null === e)
                        d.removeAttr(a, b);
                    else {
                        if (h && "set"in h && c && (m = h.set(a, e, b)) !== f)
                            return m;
                            a.setAttribute(b, "" + e);
                            return e
                    } else {
                    if (h && "get"in h && c && null !== (m = h.get(a, b)))
                        return m;
                    m = a.getAttribute(b);
                    return null === m ? f : m
                }
            }
        },
        removeAttr: function(a,
        b) {
            var e, c, m, h, u = 0;
            if (b && 1 === a.nodeType)
                for (c = b.split(aa); u < c.length; u++)
                    if (m = c[u])
                        e = d.propFix[m] || m, (h = Tb.test(m)) || d.attr(a, m, ""), a.removeAttribute(db ? m : e), h && e in a && (a[e]=!1)
            },
        attrHooks: {
            type: {
                set: function(a, b) {
                    if (tb.test(a.nodeName) && a.parentNode)
                        d.error("type property can't be changed");
                    else if (!d.support.radioValue && "radio" === b && d.nodeName(a, "input")) {
                        var e = a.value;
                        a.setAttribute("type", b);
                        e && (a.value = e);
                        return b
                    }
                }
            },
            value: {
                get: function(a, b) {
                    return Da && d.nodeName(a, "button") ? Da.get(a, b) : b in
                    a ? a.value : null
                },
                set: function(a, b, e) {
                    if (Da && d.nodeName(a, "button"))
                        return Da.set(a, b, e);
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
        prop: function(a, b, e) {
            var c, m, h;
            h = a.nodeType;
            if (a && 3 !== h && 8 !== h && 2 !== h) {
                if (h = 1 !== h ||!d.isXMLDoc(a))
                    b = d.propFix[b] || b, m = d.propHooks[b];
                return e !== f ? m && "set"in m && (c = m.set(a, e, b)) !== f ? c : a[b] = e : m && "get"in m && null !== (c = m.get(a, b)) ? c : a[b]
            }
        },
        propHooks: {
            tabIndex: {
                get: function(a) {
                    var b = a.getAttributeNode("tabindex");
                    return b && b.specified ? parseInt(b.value, 10) : Ta.test(a.nodeName) || Sb.test(a.nodeName) && a.href ? 0 : f
                }
            }
        }
    });
    rb = {
        get: function(a, b) {
            var e, c = d.prop(a, b);
            return !0 === c || "boolean" !== typeof c && (e = a.getAttributeNode(b))&&!1 !== e.nodeValue ? b.toLowerCase() : f
        },
        set: function(a, b, e) {
            !1 === b ? d.removeAttr(a, e) : (b = d.propFix[e] || e, b in a && (a[b]=!0), a.setAttribute(e,
            e.toLowerCase()));
            return e
        }
    };
    db || (sb = {
        name: !0,
        id: !0,
        coords: !0
    }, Da = d.valHooks.button = {
        get: function(a, b) {
            var e;
            return (e = a.getAttributeNode(b)) && (sb[b] ? "" !== e.value : e.specified) ? e.value : f
        },
        set: function(a, b, e) {
            var d = a.getAttributeNode(e);
            d || (d = C.createAttribute(e), a.setAttributeNode(d));
            return d.value = b + ""
        }
    }, d.each(["width", "height"], function(a, b) {
        d.attrHooks[b] = d.extend(d.attrHooks[b], {
            set: function(a, d) {
                if ("" === d)
                    return a.setAttribute(b, "auto"), d
            }
        })
    }), d.attrHooks.contenteditable = {
        get: Da.get,
        set: function(a,
        b, e) {
            "" === b && (b = "false");
            Da.set(a, b, e)
        }
    });
    d.support.hrefNormalized || d.each(["href", "src", "width", "height"], function(a, b) {
        d.attrHooks[b] = d.extend(d.attrHooks[b], {
            get: function(a) {
                a = a.getAttribute(b, 2);
                return null === a ? f : a
            }
        })
    });
    d.support.style || (d.attrHooks.style = {
        get: function(a) {
            return a.style.cssText.toLowerCase() || f
        },
        set: function(a, b) {
            return a.style.cssText = "" + b
        }
    });
    d.support.optSelected || (d.propHooks.selected = d.extend(d.propHooks.selected, {
        get: function(a) {
            if (a = a.parentNode)
                a.selectedIndex, a.parentNode &&
                a.parentNode.selectedIndex;
            return null
        }
    }));
    d.support.enctype || (d.propFix.enctype = "encoding");
    d.support.checkOn || d.each(["radio", "checkbox"], function() {
        d.valHooks[this] = {
            get: function(a) {
                return null === a.getAttribute("value") ? "on" : a.value
            }
        }
    });
    d.each(["radio", "checkbox"], function() {
        d.valHooks[this] = d.extend(d.valHooks[this], {
            set: function(a, b) {
                if (d.isArray(b))
                    return a.checked = 0 <= d.inArray(d(a).val(), b)
            }
        })
    });
    var ub = /^(?:textarea|input|select)$/i, $a = /^([^\.]*|)(?:\.(.+)|)$/, Ub = /(?:^|\s)hover(\.\S+|)\b/,
    ea = /^key/, sa = /^(?:mouse|contextmenu)|click/, Gb = /^(?:focusinfocus|focusoutblur)$/, ab = function(a) {
        return d.event.special.hover ? a : a.replace(Ub, "mouseenter$1 mouseleave$1")
    };
    d.event = {
        add: function(a, b, e, c, m) {
            var h, u, q, p, r, s, y, H, F;
            if (3 !== a.nodeType && 8 !== a.nodeType && b && e && (h = d._data(a))) {
                e.handler && (y = e, e = y.handler, m = y.selector);
                e.guid || (e.guid = d.guid++);
                q = h.events;
                q || (h.events = q = {});
                u = h.handle;
                u || (h.handle = u = function(a) {
                    return "undefined" === typeof d || a && d.event.triggered === a.type ? f : d.event.dispatch.apply(u.elem,
                    arguments)
                }, u.elem = a);
                b = d.trim(ab(b)).split(" ");
                for (h = 0; h < b.length; h++)
                    p = $a.exec(b[h]) || [], r = p[1], s = (p[2] || "").split(".").sort(), F = d.event.special[r] || {}, r = (m ? F.delegateType : F.bindType) || r, F = d.event.special[r] || {}, p = d.extend({
                        type: r,
                        origType: p[1],
                        data: c,
                        handler: e,
                        guid: e.guid,
                        selector: m,
                        namespace: s.join(".")
                    }, y), H = q[r], H || (H = q[r] = [], H.delegateCount = 0, F.setup&&!1 !== F.setup.call(a, c, s, u) || (a.addEventListener ? a.addEventListener(r, u, !1) : a.attachEvent && a.attachEvent("on" + r, u))), F.add && (F.add.call(a, p),
                    p.handler.guid || (p.handler.guid = e.guid)), m ? H.splice(H.delegateCount++, 0, p) : H.push(p), d.event.global[r]=!0;
                a = null
            }
        },
        global: {},
        remove: function(a, b, e, c, m) {
            var h, u, f, p, r, s, y, H, F, S, g = d.hasData(a) && d._data(a);
            if (g && (y = g.events)) {
                b = d.trim(ab(b || "")).split(" ");
                for (h = 0; h < b.length; h++)
                    if (u = $a.exec(b[h]) || [], f = p = u[1], u = u[2], f) {
                        H = d.event.special[f] || {};
                        f = (c ? H.delegateType : H.bindType) || f;
                        F = y[f] || [];
                        r = F.length;
                        u = u ? RegExp("(^|\\.)" + u.split(".").sort().join("\\.(?:.*\\.|)") + "(\\.|$)") : null;
                        for (s = 0; s < F.length; s++)
                            S =
                            F[s], !m && p !== S.origType || e && e.guid !== S.guid || u&&!u.test(S.namespace) || c&&!(c === S.selector || "**" === c && S.selector) || (F.splice(s--, 1), S.selector && F.delegateCount--, H.remove && H.remove.call(a, S));
                            0 === F.length && r !== F.length && (H.teardown&&!1 !== H.teardown.call(a, u, g.handle) || d.removeEvent(a, f, g.handle), delete y[f])
                    } else 
                        for (f in y)
                            d.event.remove(a, f + b[h], e, c, !0);
                d.isEmptyObject(y) && (delete g.handle, d.removeData(a, "events", !0))
            }
        },
        customEvent: {
            getData: !0,
            setData: !0,
            changeData: !0
        },
        trigger: function(a, b, e, l) {
            if (!e ||
            3 !== e.nodeType && 8 !== e.nodeType) {
                var m, h, u, p, g, r, s = a.type || a;
                u = [];
                if (!Gb.test(s + d.event.triggered) && (0 <= s.indexOf("!") && (s = s.slice(0, - 1), m=!0), 0 <= s.indexOf(".") && (u = s.split("."), s = u.shift(), u.sort()), e&&!d.event.customEvent[s] || d.event.global[s]))
                    if (a = "object" === typeof a ? a[d.expando] ? a : new d.Event(s, a) : new d.Event(s), a.type = s, a.isTrigger=!0, a.exclusive = m, a.namespace = u.join("."), a.namespace_re = a.namespace ? RegExp("(^|\\.)" + u.join("\\.(?:.*\\.|)") + "(\\.|$)") : null, u = 0 > s.indexOf(":") ? "on" + s : "", e) {
                        if (a.result =
                        f, a.target || (a.target = e), b = null != b ? d.makeArray(b) : [], b.unshift(a), p = d.event.special[s] || {}, !p.trigger ||!1 !== p.trigger.apply(e, b)) {
                            r = [[e, p.bindType || s]];
                            if (!l&&!p.noBubble&&!d.isWindow(e)) {
                                g = p.delegateType || s;
                                m = Gb.test(g + s) ? e : e.parentNode;
                                for (h = e; m; m = m.parentNode)
                                    r.push([m, g]), h = m;
                                    h === (e.ownerDocument || C) && r.push([h.defaultView || h.parentWindow || c, g])
                                }
                                for (h = 0; h < r.length&&!a.isPropagationStopped(); h++)
                                    m = r[h][0], a.type = r[h][1], (g = (d._data(m, "events") || {})[a.type] && d._data(m, "handle")) && g.apply(m, b), (g =
                                    u && m[u]) && d.acceptData(m)&&!1 === g.apply(m, b) && a.preventDefault();
                                    a.type = s;
                                    l || a.isDefaultPrevented() || p._default&&!1 !== p._default.apply(e.ownerDocument, b) || "click" === s && d.nodeName(e, "a") ||!d.acceptData(e) ||!u ||!e[s] || ("focus" === s || "blur" === s) && 0 === a.target.offsetWidth || d.isWindow(e) || ((h = e[u]) && (e[u] = null), d.event.triggered = s, e[s](), d.event.triggered = f, h && (e[u] = h));
                                    return a.result
                        }
                    } else 
                        for (h in e = d.cache, e)
                            e[h].events && e[h].events[s] && d.event.trigger(a, b, e[h].handle.elem, !0)
                }
        },
        dispatch: function(a) {
            a =
            d.event.fix(a || c.event);
            var b, e, l, m, h, u, p = (d._data(this, "events") || {})[a.type] || [], g = p.delegateCount, r = [].slice.call(arguments), s=!a.exclusive&&!a.namespace, y = d.event.special[a.type] || {}, H = [];
            r[0] = a;
            a.delegateTarget = this;
            if (!y.preDispatch ||!1 !== y.preDispatch.call(this, a)) {
                if (g && (!a.button || "click" !== a.type))
                    for (e = a.target; e != this; e = e.parentNode || this)
                        if (!0 !== e.disabled || "click" !== a.type) {
                            m = {};
                            h = [];
                            for (b = 0; b < g; b++)
                                l = p[b], u = l.selector, m[u] === f && (m[u] = 0 <= d(u, this).index(e)), m[u] && h.push(l);
                                h.length &&
                                H.push({
                                    elem: e,
                                    matches: h
                                })
                        }
                p.length > g && H.push({
                    elem: this,
                    matches: p.slice(g)
                });
                for (b = 0; b < H.length&&!a.isPropagationStopped(); b++)
                    for (m = H[b], a.currentTarget = m.elem, e = 0; e < m.matches.length&&!a.isImmediatePropagationStopped(); e++)
                        if (l = m.matches[e], s ||!a.namespace&&!l.namespace || a.namespace_re && a.namespace_re.test(l.namespace))
                            a.data = l.data, a.handleObj = l, l = ((d.event.special[l.origType] || {}).handle || l.handler).apply(m.elem, r), l !== f && (a.result = l, !1 === l && (a.preventDefault(), a.stopPropagation()));
                y.postDispatch &&
                y.postDispatch.call(this, a);
                return a.result
            }
        },
        props: "attrChange attrName relatedNode srcElement altKey bubbles cancelable ctrlKey currentTarget eventPhase metaKey relatedTarget shiftKey target timeStamp view which".split(" "),
        fixHooks: {},
        keyHooks: {
            props: ["char", "charCode", "key", "keyCode"],
            filter: function(a, b) {
                null == a.which && (a.which = null != b.charCode ? b.charCode : b.keyCode);
                return a
            }
        },
        mouseHooks: {
            props: "button buttons clientX clientY fromElement offsetX offsetY pageX pageY screenX screenY toElement".split(" "),
            filter: function(a, b) {
                var e, d, c = b.button, h = b.fromElement;
                null == a.pageX && null != b.clientX && (e = a.target.ownerDocument || C, d = e.documentElement, e = e.body, a.pageX = b.clientX + (d && d.scrollLeft || e && e.scrollLeft || 0) - (d && d.clientLeft || e && e.clientLeft || 0), a.pageY = b.clientY + (d && d.scrollTop || e && e.scrollTop || 0) - (d && d.clientTop || e && e.clientTop || 0));
                !a.relatedTarget && h && (a.relatedTarget = h === a.target ? b.toElement : h);
                a.which || c === f || (a.which = c & 1 ? 1 : c & 2 ? 3 : c & 4 ? 2 : 0);
                return a
            }
        },
        fix: function(a) {
            if (a[d.expando])
                return a;
            var b,
            e, c = a, m = d.event.fixHooks[a.type] || {}, h = m.props ? this.props.concat(m.props): this.props;
            a = d.Event(c);
            for (b = h.length; b;)
                e = h[--b], a[e] = c[e];
            a.target || (a.target = c.srcElement || C);
            3 === a.target.nodeType && (a.target = a.target.parentNode);
            a.metaKey=!!a.metaKey;
            return m.filter ? m.filter(a, c) : a
        },
        special: {
            load: {
                noBubble: !0
            },
            focus: {
                delegateType: "focusin"
            },
            blur: {
                delegateType: "focusout"
            },
            beforeunload: {
                setup: function(a, b, e) {
                    d.isWindow(this) && (this.onbeforeunload = e)
                },
                teardown: function(a, b) {
                    this.onbeforeunload === b && (this.onbeforeunload =
                    null)
                }
            }
        },
        simulate: function(a, b, e, c) {
            a = d.extend(new d.Event, e, {
                type: a,
                isSimulated: !0,
                originalEvent: {}
            });
            c ? d.event.trigger(a, null, b) : d.event.dispatch.call(b, a);
            a.isDefaultPrevented() && e.preventDefault()
        }
    };
    d.event.handle = d.event.dispatch;
    d.removeEvent = C.removeEventListener ? function(a, b, e) {
        a.removeEventListener && a.removeEventListener(b, e, !1)
    } : function(a, b, e) {
        b = "on" + b;
        a.detachEvent && ("undefined" === typeof a[b] && (a[b] = null), a.detachEvent(b, e))
    };
    d.Event = function(a, b) {
        if (!(this instanceof d.Event))
            return new d.Event(a,
            b);
        a && a.type ? (this.originalEvent = a, this.type = a.type, this.isDefaultPrevented = a.defaultPrevented ||!1 === a.returnValue || a.getPreventDefault && a.getPreventDefault() ? t : v) : this.type = a;
        b && d.extend(this, b);
        this.timeStamp = a && a.timeStamp || d.now();
        this[d.expando]=!0
    };
    d.Event.prototype = {
        preventDefault: function() {
            this.isDefaultPrevented = t;
            var a = this.originalEvent;
            a && (a.preventDefault ? a.preventDefault() : a.returnValue=!1)
        },
        stopPropagation: function() {
            this.isPropagationStopped = t;
            var a = this.originalEvent;
            a && (a.stopPropagation &&
            a.stopPropagation(), a.cancelBubble=!0)
        },
        stopImmediatePropagation: function() {
            this.isImmediatePropagationStopped = t;
            this.stopPropagation()
        },
        isDefaultPrevented: v,
        isPropagationStopped: v,
        isImmediatePropagationStopped: v
    };
    d.each({
        mouseenter: "mouseover",
        mouseleave: "mouseout"
    }, function(a, b) {
        d.event.special[a] = {
            delegateType: b,
            bindType: b,
            handle: function(a) {
                var c, m = a.relatedTarget, h = a.handleObj;
                if (!m || m !== this&&!d.contains(this, m))
                    a.type = h.origType, c = h.handler.apply(this, arguments), a.type = b;
                return c
            }
        }
    });
    d.support.submitBubbles ||
    (d.event.special.submit = {
        setup: function() {
            if (d.nodeName(this, "form"))
                return !1;
            d.event.add(this, "click._submit keypress._submit", function(a) {
                a = a.target;
                (a = d.nodeName(a, "input") || d.nodeName(a, "button") ? a.form : f)&&!d._data(a, "_submit_attached") && (d.event.add(a, "submit._submit", function(a) {
                    a._submit_bubble=!0
                }), d._data(a, "_submit_attached", !0))
            })
        },
        postDispatch: function(a) {
            a._submit_bubble && (delete a._submit_bubble, this.parentNode&&!a.isTrigger && d.event.simulate("submit", this.parentNode, a, !0))
        },
        teardown: function() {
            if (d.nodeName(this,
            "form"))
                return !1;
            d.event.remove(this, "._submit")
        }
    });
    d.support.changeBubbles || (d.event.special.change = {
        setup: function() {
            if (ub.test(this.nodeName)) {
                if ("checkbox" === this.type || "radio" === this.type)
                    d.event.add(this, "propertychange._change", function(a) {
                        "checked" === a.originalEvent.propertyName && (this._just_changed=!0)
                    }), d.event.add(this, "click._change", function(a) {
                        this._just_changed&&!a.isTrigger && (this._just_changed=!1);
                        d.event.simulate("change", this, a, !0)
                    });
                return !1
            }
            d.event.add(this, "beforeactivate._change",
            function(a) {
                a = a.target;
                ub.test(a.nodeName)&&!d._data(a, "_change_attached") && (d.event.add(a, "change._change", function(a) {
                    !this.parentNode || a.isSimulated || a.isTrigger || d.event.simulate("change", this.parentNode, a, !0)
                }), d._data(a, "_change_attached", !0))
            })
        },
        handle: function(a) {
            var b = a.target;
            if (this !== b || a.isSimulated || a.isTrigger || "radio" !== b.type && "checkbox" !== b.type)
                return a.handleObj.handler.apply(this, arguments)
        },
        teardown: function() {
            d.event.remove(this, "._change");
            return !ub.test(this.nodeName)
        }
    });
    d.support.focusinBubbles || d.each({
        focus: "focusin",
        blur: "focusout"
    }, function(a, b) {
        var e = 0, c = function(a) {
            d.event.simulate(b, a.target, d.event.fix(a), !0)
        };
        d.event.special[b] = {
            setup: function() {
                0 === e++&&C.addEventListener(a, c, !0)
            },
            teardown: function() {
                0===--e && C.removeEventListener(a, c, !0)
            }
        }
    });
    d.fn.extend({
        on: function(a, b, e, c, m) {
            var h, u;
            if ("object" === typeof a) {
                "string" !== typeof b && (e = e || b, b = f);
                for (u in a)
                    this.on(u, b, e, a[u], m);
                return this
            }
            null == e && null == c ? (c = b, e = b = f) : null == c && ("string" === typeof b ? (c = e, e =
            f) : (c = e, e = b, b = f));
            if (!1 === c)
                c = v;
            else if (!c)
                return this;
            1 === m && (h = c, c = function(a) {
                d().off(a);
                return h.apply(this, arguments)
            }, c.guid = h.guid || (h.guid = d.guid++));
            return this.each(function() {
                d.event.add(this, a, c, e, b)
            })
        },
        one: function(a, b, e, d) {
            return this.on(a, b, e, d, 1)
        },
        off: function(a, b, e) {
            var c;
            if (a && a.preventDefault && a.handleObj)
                return c = a.handleObj, d(a.delegateTarget).off(c.namespace ? c.origType + "." + c.namespace : c.origType, c.selector, c.handler), this;
            if ("object" === typeof a) {
                for (c in a)
                    this.off(c, b, a[c]);
                return this
            }
            if (!1 === b || "function" === typeof b)
                e = b, b = f;
            !1 === e && (e = v);
            return this.each(function() {
                d.event.remove(this, a, e, b)
            })
        },
        bind: function(a, b, e) {
            return this.on(a, null, b, e)
        },
        unbind: function(a, b) {
            return this.off(a, null, b)
        },
        live: function(a, b, e) {
            d(this.context).on(a, this.selector, b, e);
            return this
        },
        die: function(a, b) {
            d(this.context).off(a, this.selector || "**", b);
            return this
        },
        delegate: function(a, b, e, d) {
            return this.on(b, a, e, d)
        },
        undelegate: function(a, b, e) {
            return 1 == arguments.length ? this.off(a, "**") : this.off(b,
            a || "**", e)
        },
        trigger: function(a, b) {
            return this.each(function() {
                d.event.trigger(a, b, this)
            })
        },
        triggerHandler: function(a, b) {
            if (this[0])
                return d.event.trigger(a, b, this[0], !0)
        },
        toggle: function(a) {
            var b = arguments, e = a.guid || d.guid++, c = 0, m = function(e) {
                var m = (d._data(this, "lastToggle" + a.guid) || 0)%c;
                d._data(this, "lastToggle" + a.guid, m + 1);
                e.preventDefault();
                return b[m].apply(this, arguments) ||!1
            };
            for (m.guid = e; c < b.length;)
                b[c++].guid = e;
            return this.click(m)
        },
        hover: function(a, b) {
            return this.mouseenter(a).mouseleave(b ||
            a)
        }
    });
    d.each("blur focus focusin focusout load resize scroll unload click dblclick mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave change select submit keydown keypress keyup error contextmenu".split(" "), function(a, b) {
        d.fn[b] = function(a, d) {
            null == d && (d = a, a = null);
            return 0 < arguments.length ? this.on(b, null, a, d) : this.trigger(b)
        };
        ea.test(b) && (d.event.fixHooks[b] = d.event.keyHooks);
        sa.test(b) && (d.event.fixHooks[b] = d.event.mouseHooks)
    });
    (function(a, b) {
        function e(a, b, e, d) {
            e = e || [];
            b = b ||
            x;
            var r, c, s, y, l = b.nodeType;
            if (1 !== l && 9 !== l)
                return [];
            if (!a || "string" !== typeof a)
                return e;
            s = k(b);
            if (!s&&!d && (r = O.exec(a)))
                if (y = r[1])
                    if (9 === l)
                        if ((c = b.getElementById(y)) && c.parentNode) {
                            if (c.id === y)
                                return e.push(c), e
                        } else 
                            return e;
                        else {
                            if (b.ownerDocument && (c = b.ownerDocument.getElementById(y)) && Z(b, c) && c.id === y)
                                return e.push(c), e
                        } else {
                            if (r[2])
                                return A.apply(e, ca.call(b.getElementsByTagName(a), 0)), e;
                                if ((y = r[3]) && Y && b.getElementsByClassName)
                                    return A.apply(e, ca.call(b.getElementsByClassName(y), 0)), e
                        }
            return F(a,
            b, e, d, s)
        }
        function c(a) {
            return function(b) {
                return "input" === b.nodeName.toLowerCase() && b.type === a
            }
        }
        function m(a) {
            return function(b) {
                var e = b.nodeName.toLowerCase();
                return ("input" === e || "button" === e) && b.type === a
            }
        }
        function h(a, b, e) {
            if (a === b)
                return e;
            for (a = a.nextSibling; a;) {
                if (a === b)
                    return - 1;
                a = a.nextSibling
            }
            return 1
        }
        function f(a, b, d, r) {
            var c, s, y, l, h, H, m, u, F, p=!d && b !== x, q = (p ? "\x3cs\x3e" : "") + a.replace(D, "$1\x3cs\x3e");
            if (h = W[G][q])
                return r ? 0 : ca.call(h, 0);
            h = a;
            H = [];
            u = J.preFilter;
            for (F = J.filter; h;) {
                if (!c || (s =
                I.exec(h)))
                    s && (h = h.slice(s[0].length), y.selector = m), H.push(y = []), m = "", p && (h = " " + h);
                c=!1;
                if (s = qb.exec(h))
                    m += s[0], h = h.slice(s[0].length), c = y.push({
                        part: s.pop().replace(D, " "),
                        string: s[0],
                        captures: s
                    });
                for (l in F)
                    !(s = Q[l].exec(h)) || u[l]&&!(s = u[l](s, b, d)) || (m += s[0], h = h.slice(s[0].length), c = y.push({
                        part: l,
                        string: s.shift(),
                        captures: s
                    }));
                if (!c)
                    break
            }
            m && (y.selector = m);
            return r ? h.length : h ? e.error(a) : ca.call(W(q, H), 0)
        }
        function p(a, b, e, d) {
            var r = b.dir, c = N++;
            a || (a = function(a) {
                return a === e
            });
            return b.first ? function(b) {
                for (; b =
                b[r];)
                    if (1 === b.nodeType)
                        return a(b) && b
            } : d ? function(b) {
                for (; b = b[r];)
                    if (1 === b.nodeType && a(b))
                        return b
            } : function(b) {
                for (var e, d = c + "." + S, s = d + "." + Ca; b = b[r];)
                    if (1 === b.nodeType) {
                        if ((e = b[G]) === s)
                            return b.sizset;
                            if ("string" === typeof e && 0 === e.indexOf(d)) {
                                if (b.sizset)
                                    return b
                            } else {
                                b[G] = s;
                                if (a(b))
                                    return b.sizset=!0, b;
                                    b.sizset=!1
                            }
                    }
            }
        }
        function g(a, b) {
            return a ? function(e) {
                var d = b(e);
                return d && a(!0 === d ? e : d)
            } : b
        }
        function r(a) {
            return function(b) {
                for (var e, d = 0; e = a[d]; d++)
                    if (e(b))
                        return !0;
                return !1
            }
        }
        function s(a, b, d, r) {
            for (var c =
            0, s = b.length; c < s; c++)
                e(a, b[c], d, r)
        }
        function y(a, b, d, r, c, y) {
            var l, h = J.setFilters[b.toLowerCase()];
            h || e.error(b);
            !a && (l = c) || s(a || "*", r, l = [], c);
            return 0 < l.length ? h(l, d, y) : []
        }
        function H(a, d, r, c) {
            for (var l, h, H, m, f, u, F, p, q, S, g, Ca = 0, n = a.length, J = Q.POS, P = RegExp("^" + J.source + "(?!" + U + ")", "i"), tc = function() {
                for (var a = 1, e = arguments.length - 2; a < e; a++)
                    arguments[a] === b && (p[a] = b)
            }; Ca < n; Ca++) {
                l = a[Ca];
                h = "";
                F = c;
                H = 0;
                for (m = l.length; H < m; H++) {
                    f = l[H];
                    u = f.string;
                    if ("PSEUDO" === f.part)
                        for (J.exec(""), f = 0; p = J.exec(u);) {
                            q=!0;
                            S =
                            J.lastIndex = p.index + p[0].length;
                            if (S > f) {
                                h += u.slice(f, p.index);
                                f = S;
                                S = [d];
                                qb.test(h) && (F && (S = F), F = c);
                                if (g = M.test(h))
                                    h = h.slice(0, - 5).replace(qb, "$\x26*"), f++;
                                    1 < p.length && p[0].replace(P, tc);
                                    F = y(h, p[1], p[2], S, F, g)
                                }
                                h = ""
                        }
                    q || (h += u);
                    q=!1
                }
                h ? qb.test(h) ? s(h, F || [d], r, c) : e(h, d, r, c ? c.concat(F) : F) : A.apply(r, F)
            }
            return 1 === n ? r : e.uniqueSort(r)
        }
        function F(a, b, e, d, r) {
            a = a.replace(D, "$1");
            var c, s, y, l, h, m;
            s = f(a, b, r);
            h = b.nodeType;
            if (Q.POS.test(a))
                return H(s, b, e, d);
            if (d)
                c = ca.call(d, 0);
            else if (1 === s.length) {
                if (2 < (y = ca.call(s[0],
                0)).length && "ID" === (l = y[0]).part && 9 === h&&!r && J.relative[y[1].part]) {
                    b = J.find.ID(l.captures[0].replace(na, ""), b, r)[0];
                    if (!b)
                        return e;
                    a = a.slice(y.shift().string.length)
                }
                d = (s = L.exec(y[0].string))&&!s.index && b.parentNode || b;
                h = "";
                for (s = y.length - 1; 0 <= s; s--) {
                    l = y[s];
                    m = l.part;
                    h = l.string + h;
                    if (J.relative[m])
                        break;
                    if (J.order.test(m) && (c = J.find[m](l.captures[0].replace(na, ""), d, r), null != c)) {
                        (a = a.slice(0, a.length - h.length) + h.replace(Q[m], "")) || A.apply(e, ca.call(c, 0));
                        break
                    }
                }
            }
            if (a)
                for (r = v(a, b, r), S = r.dirruns++, null ==
                c && (c = J.find.TAG("*", L.test(a) && b.parentNode || b)), s = 0; a = c[s]; s++)
                    Ca = r.runs++, r(a) && e.push(a);
            return e
        }
        var S, Ca, n, J, P, k, Z, v, t, z, w=!0, G = ("sizcache" + Math.random()).replace(".", ""), x = a.document, B = x.documentElement, N = 0, ca = [].slice, A = [].push, ja = function(a, b) {
            a[G] = b ||!0;
            return a
        }, C = function() {
            var a = {}, b = [];
            return ja(function(e, d) {
                b.push(e) > J.cacheLength && delete a[b.shift()];
                return a[e] = d
            }, a)
        }, E = C(), W = C(), qa = C(), U = "[\\x20\\t\\r\\n\\f]", C = "(?:\\\\.|[-\\w]|[^\\x00-\\xa0])+".replace("w", "w#"), C = "\\[" + U + "*((?:\\\\.|[-\\w]|[^\\x00-\\xa0])+)" +
        U + "*(?:([*^$|!~]?\x3d)" + U + "*(?:(['\"])((?:\\\\.|[^\\\\])*?)\\3|(" + C + ")|)|)" + U + "*\\]", K = ":((?:\\\\.|[-\\w]|[^\\x00-\\xa0])+)(?:\\((?:(['\"])((?:\\\\.|[^\\\\])*?)\\2|([^()[\\]]*|(?:(?:" + C + ")|[^:]|\\\\.)*|.*))\\)|)", D = RegExp("^" + U + "+|((?:^|[^\\\\])(?:\\\\.)*)" + U + "+$", "g"), I = RegExp("^" + U + "*," + U + "*"), qb = RegExp("^" + U + "*([\\x20\\t\\r\\n\\f\x3e+~])" + U + "*"), R = RegExp(K), O = /^(?:#([\w\-]+)|(\w+)|\.([\w\-]+))$/, L = /[\x20\t\r\n\f]*[+~]/, M = /:not\($/, V = /h\d/i, T = /input|select|textarea|button/i, na = /\\(?!\\)/g, Q = {
            ID: /^#((?:\\.|[-\w]|[^\x00-\xa0])+)/,
            CLASS: /^\.((?:\\.|[-\w]|[^\x00-\xa0])+)/,
            NAME: /^\[name=['"]?((?:\\.|[-\w]|[^\x00-\xa0])+)['"]?\]/,
            TAG: RegExp("^(" + "(?:\\\\.|[-\\w]|[^\\x00-\\xa0])+".replace("w", "w*") + ")"),
            ATTR: RegExp("^" + C),
            PSEUDO: RegExp("^" + K),
            CHILD: RegExp("^:(only|nth|last|first)-child(?:\\(" + U + "*(even|odd|(([+-]|)(\\d*)n|)" + U + "*(?:([+-]|)" + U + "*(\\d+)|))" + U + "*\\)|)", "i"),
            POS: /:(nth|eq|gt|lt|first|last|even|odd)(?:\(((?:-\d)?\d*)\)|)(?=[^-]|$)/ig,
            needsContext: RegExp("^" + U + "*[\x3e+~]|:(nth|eq|gt|lt|first|last|even|odd)(?:\\(((?:-\\d)?\\d*)\\)|)(?\x3d[^-]|$)",
            "i")
        }, Sa = function(a) {
            var b = x.createElement("div");
            try {
                return a(b)
            } catch (e) {
                return !1
            } finally {}
        }, C = Sa(function(a) {
            a.appendChild(x.createComment(""));
            return !a.getElementsByTagName("*").length
        }), K = Sa(function(a) {
            a.innerHTML = "\x3ca href\x3d'#'\x3e\x3c/a\x3e";
            return a.firstChild && "undefined" !== typeof a.firstChild.getAttribute && "#" === a.firstChild.getAttribute("href")
        }), X = Sa(function(a) {
            a.innerHTML = "\x3cselect\x3e\x3c/select\x3e";
            a = typeof a.lastChild.getAttribute("multiple");
            return "boolean" !== a && "string" !==
            a
        }), Y = Sa(function(a) {
            a.innerHTML = "\x3cdiv class\x3d'hidden e'\x3e\x3c/div\x3e\x3cdiv class\x3d'hidden'\x3e\x3c/div\x3e";
            if (!a.getElementsByClassName ||!a.getElementsByClassName("e").length)
                return !1;
            a.lastChild.className = "e";
            return 2 === a.getElementsByClassName("e").length
        }), $ = Sa(function(a) {
            a.id = G + 0;
            a.innerHTML = "\x3ca name\x3d'" + G + "'\x3e\x3c/a\x3e\x3cdiv name\x3d'" + G + "'\x3e\x3c/div\x3e";
            B.insertBefore(a, B.firstChild);
            var b = x.getElementsByName && x.getElementsByName(G).length === 2 + x.getElementsByName(G +
            0).length;
            n=!x.getElementById(G);
            B.removeChild(a);
            return b
        });
        try {
            ca.call(B.childNodes, 0)[0].nodeType
        } catch (ba) {
            ca = function(a) {
                for (var b, e = []; b = this[a]; a++)
                    e.push(b);
                return e
            }
        }
        e.matches = function(a, b) {
            return e(a, null, null, b)
        };
        e.matchesSelector = function(a, b) {
            return 0 < e(b, null, null, [a]).length
        };
        P = e.getText = function(a) {
            var b, e = "", d = 0;
            if (b = a.nodeType)
                if (1 === b || 9 === b || 11 === b) {
                    if ("string" === typeof a.textContent)
                        return a.textContent;
                        for (a = a.firstChild; a; a = a.nextSibling)
                            e += P(a)
                } else {
                    if (3 === b || 4 === b)
                        return a.nodeValue
                } else 
                    for (; b =
                    a[d]; d++)
                        e += P(b);
            return e
        };
        k = e.isXML = function(a) {
            return (a = a && (a.ownerDocument || a).documentElement) ? "HTML" !== a.nodeName : !1
        };
        Z = e.contains = B.contains ? function(a, b) {
            var e = 9 === a.nodeType ? a.documentElement: a, d = b && b.parentNode;
            return a === d||!!(d && 1 === d.nodeType && e.contains && e.contains(d))
        } : B.compareDocumentPosition ? function(a, b) {
            return b&&!!(a.compareDocumentPosition(b) & 16)
        } : function(a, b) {
            for (; b = b.parentNode;)
                if (b === a)
                    return !0;
            return !1
        };
        e.attr = function(a, b) {
            var e;
            (e = k(a)) || (b = b.toLowerCase());
            return J.attrHandle[b] ?
            J.attrHandle[b](a) : X || e ? a.getAttribute(b) : (e = a.getAttributeNode(b)) ? "boolean" === typeof a[b] ? a[b] ? b : null : e.specified ? e.value : null : null
        };
        J = e.selectors = {
            cacheLength: 50,
            createPseudo: ja,
            match: Q,
            order: RegExp("ID|TAG" + ($ ? "|NAME" : "") + (Y ? "|CLASS" : "")),
            attrHandle: K ? {}
            : {
                href: function(a) {
                    return a.getAttribute("href", 2)
                },
                type: function(a) {
                    return a.getAttribute("type")
                }
            },
            find: {
                ID: n ? function(a, b, e) {
                    if ("undefined" !== typeof b.getElementById&&!e)
                        return (a = b.getElementById(a)) && a.parentNode ? [a] : []
                }
                : function(a, e, d) {
                    if ("undefined" !==
                    typeof e.getElementById&&!d)
                        return (e = e.getElementById(a)) ? e.id === a || "undefined" !== typeof e.getAttributeNode && e.getAttributeNode("id").value === a ? [e] : b : []
                },
                TAG: C ? function(a, b) {
                    if ("undefined" !== typeof b.getElementsByTagName)
                        return b.getElementsByTagName(a)
                }
                : function(a, b) {
                    var e = b.getElementsByTagName(a);
                    if ("*" === a) {
                        for (var d, r = [], c = 0; d = e[c]; c++)
                            1 === d.nodeType && r.push(d);
                        return r
                    }
                    return e
                },
                NAME: function(a, b) {
                    if ("undefined" !== typeof b.getElementsByName)
                        return b.getElementsByName(name)
                },
                CLASS: function(a,
                b, e) {
                    if ("undefined" !== typeof b.getElementsByClassName&&!e)
                        return b.getElementsByClassName(a)
                }
            },
            relative: {
                "\x3e": {
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
                    a[1] = a[1].replace(na, "");
                    a[3] = (a[4] || a[5] || "").replace(na, "");
                    "~\x3d" === a[2] && (a[3] = " " + a[3] + " ");
                    return a.slice(0, 4)
                },
                CHILD: function(a) {
                    a[1] = a[1].toLowerCase();
                    "nth" === a[1] ? (a[2] || e.error(a[0]), a[3] =+ (a[3] ? a[4] + (a[5] || 1) : 2 * ("even" === a[2] || "odd" ===
                    a[2])), a[4] =+ (a[6] + a[7] || "odd" === a[2])) : a[2] && e.error(a[0]);
                    return a
                },
                PSEUDO: function(a, b, e) {
                    var d, r;
                    if (Q.CHILD.test(a[0]))
                        return null;
                    if (a[3])
                        a[2] = a[3];
                    else if (d = a[4])
                        R.test(d) && (r = f(d, b, e, !0)) && (r = d.indexOf(")", d.length - r) - d.length) && (d = d.slice(0, r), a[0] = a[0].slice(0, r)), a[2] = d;
                    return a.slice(0, 3)
                }
            },
            filter: {
                ID: n ? function(a) {
                    a = a.replace(na, "");
                    return function(b) {
                        return b.getAttribute("id") === a
                    }
                }
                : function(a) {
                    a = a.replace(na, "");
                    return function(b) {
                        return (b = "undefined" !== typeof b.getAttributeNode && b.getAttributeNode("id")) &&
                        b.value === a
                    }
                },
                TAG: function(a) {
                    if ("*" === a)
                        return function() {
                            return !0
                        };
                    a = a.replace(na, "").toLowerCase();
                    return function(b) {
                        return b.nodeName && b.nodeName.toLowerCase() === a
                    }
                },
                CLASS: function(a) {
                    var b = E[G][a];
                    b || (b = E(a, RegExp("(^|" + U + ")" + a + "(" + U + "|$)")));
                    return function(a) {
                        return b.test(a.className || "undefined" !== typeof a.getAttribute && a.getAttribute("class") || "")
                    }
                },
                ATTR: function(a, b, d) {
                    return b ? function(r) {
                        r = e.attr(r, a);
                        var c = r + "";
                        if (null == r)
                            return "!\x3d" === b;
                        switch (b) {
                        case "\x3d":
                            return c === d;
                        case "!\x3d":
                            return c !==
                            d;
                        case "^\x3d":
                            return d && 0 === c.indexOf(d);
                        case "*\x3d":
                            return d&&-1 < c.indexOf(d);
                        case "$\x3d":
                            return d && c.substr(c.length - d.length) === d;
                        case "~\x3d":
                            return - 1 < (" " + c + " ").indexOf(d);
                        case "|\x3d":
                            return c === d || c.substr(0, d.length + 1) === d + "-"
                        }
                    } : function(b) {
                        return null != e.attr(b, a)
                    }
                },
                CHILD: function(a, b, e, d) {
                    if ("nth" === a) {
                        var r = N++;
                        return function(a) {
                            var b, c = 0, s = a;
                            if (1 === e && 0 === d)
                                return !0;
                            if ((b = a.parentNode) && (b[G] !== r ||!a.sizset)) {
                                for (s = b.firstChild; s && (1 !== s.nodeType || (s.sizset=++c, s !== a)); s = s.nextSibling);
                                b[G] = r
                            }
                            a = a.sizset - d;
                            return 0 === e ? 0 === a : 0 === a%e && 0 <= a / e
                        }
                    }
                    return function(b) {
                        var e = b;
                        switch (a) {
                        case "only":
                        case "first":
                            for (; e = e.previousSibling;)
                                if (1 === e.nodeType)
                                    return !1;
                            if ("first" === a)
                                return !0;
                            e = b;
                        case "last":
                            for (; e = e.nextSibling;)
                                if (1 === e.nodeType)
                                    return !1;
                            return !0
                        }
                    }
                },
                PSEUDO: function(a, b, d, r) {
                    var c, s = J.pseudos[a] || J.pseudos[a.toLowerCase()];
                    s || e.error("unsupported pseudo: " + a);
                    return s[G] ? s(b, d, r) : 1 < s.length ? (c = [a, a, "", b], function(a) {
                        return s(a, 0, c)
                    }) : s
                }
            },
            pseudos: {
                not: ja(function(a, b, e) {
                    var d =
                    v(a.replace(D, "$1"), b, e);
                    return function(a) {
                        return !d(a)
                    }
                }),
                enabled: function(a) {
                    return !1 === a.disabled
                },
                disabled: function(a) {
                    return !0 === a.disabled
                },
                checked: function(a) {
                    var b = a.nodeName.toLowerCase();
                    return "input" === b&&!!a.checked || "option" === b&&!!a.selected
                },
                selected: function(a) {
                    a.parentNode && a.parentNode.selectedIndex;
                    return !0 === a.selected
                },
                parent: function(a) {
                    return !J.pseudos.empty(a)
                },
                empty: function(a) {
                    var b;
                    for (a = a.firstChild; a;) {
                        if ("@" < a.nodeName || 3 === (b = a.nodeType) || 4 === b)
                            return !1;
                        a = a.nextSibling
                    }
                    return !0
                },
                contains: ja(function(a) {
                    return function(b) {
                        return - 1 < (b.textContent || b.innerText || P(b)).indexOf(a)
                    }
                }),
                has: ja(function(a) {
                    return function(b) {
                        return 0 < e(a, b).length
                    }
                }),
                header: function(a) {
                    return V.test(a.nodeName)
                },
                text: function(a) {
                    var b, e;
                    return "input" === a.nodeName.toLowerCase() && "text" === (b = a.type) && (null == (e = a.getAttribute("type")) || e.toLowerCase() === b)
                },
                radio: c("radio"),
                checkbox: c("checkbox"),
                file: c("file"),
                password: c("password"),
                image: c("image"),
                submit: m("submit"),
                reset: m("reset"),
                button: function(a) {
                    var b =
                    a.nodeName.toLowerCase();
                    return "input" === b && "button" === a.type || "button" === b
                },
                input: function(a) {
                    return T.test(a.nodeName)
                },
                focus: function(a) {
                    var b = a.ownerDocument;
                    return a === b.activeElement && (!b.hasFocus || b.hasFocus())&&!(!a.type&&!a.href)
                },
                active: function(a) {
                    return a === a.ownerDocument.activeElement
                }
            },
            setFilters: {
                first: function(a, b, e) {
                    return e ? a.slice(1) : [a[0]]
                },
                last: function(a, b, e) {
                    b = a.pop();
                    return e ? a : [b]
                },
                even: function(a, b, e) {
                    b = [];
                    e = e ? 1 : 0;
                    for (var d = a.length; e < d; e += 2)
                        b.push(a[e]);
                    return b
                },
                odd: function(a,
                b, e) {
                    b = [];
                    e = e ? 0 : 1;
                    for (var d = a.length; e < d; e += 2)
                        b.push(a[e]);
                    return b
                },
                lt: function(a, b, e) {
                    return e ? a.slice( + b) : a.slice(0, + b)
                },
                gt: function(a, b, e) {
                    return e ? a.slice(0, + b + 1) : a.slice( + b + 1)
                },
                eq: function(a, b, e) {
                    b = a.splice( + b, 1);
                    return e ? a : b
                }
            }
        };
        t = B.compareDocumentPosition ? function(a, b) {
            return a === b ? (z=!0, 0) : (a.compareDocumentPosition && b.compareDocumentPosition ? a.compareDocumentPosition(b) & 4 : a.compareDocumentPosition)?-1 : 1
        } : function(a, b) {
            if (a === b)
                return z=!0, 0;
            if (a.sourceIndex && b.sourceIndex)
                return a.sourceIndex -
                b.sourceIndex;
            var e, d, r = [], c = [];
            e = a.parentNode;
            d = b.parentNode;
            var s = e;
            if (e === d)
                return h(a, b);
            if (!e)
                return - 1;
            if (!d)
                return 1;
            for (; s;)
                r.unshift(s), s = s.parentNode;
            for (s = d; s;)
                c.unshift(s), s = s.parentNode;
            e = r.length;
            d = c.length;
            for (s = 0; s < e && s < d; s++)
                if (r[s] !== c[s])
                    return h(r[s], c[s]);
            return s === e ? h(a, c[s], - 1) : h(r[s], b, 1)
        };
        [0, 0].sort(t);
        w=!z;
        e.uniqueSort = function(a) {
            var b, e = 1;
            z = w;
            a.sort(t);
            if (z)
                for (; b = a[e]; e++)
                    b === a[e - 1] && a.splice(e--, 1);
            return a
        };
        e.error = function(a) {
            throw Error("Syntax error, unrecognized expression: " +
            a);
        };
        v = e.compile = function(a, b, e) {
            var d, c, s;
            if ((d = qa[G][a]) && d.context === b)
                return d;
            d = f(a, b, e);
            c = 0;
            for (s = d.length; c < s; c++) {
                for (var y = d, l = c, h = d[c], H = b, m = e, F = void 0, S = void 0, Ca = 0; F = h[Ca]; Ca++)
                    S = J.relative[F.part] ? p(S, J.relative[F.part], H, m) : g(S, J.filter[F.part].apply(null, F.captures.concat(H, m)));
                y[l] = S
            }
            d = qa(a, r(d));
            d.context = b;
            d.runs = d.dirruns = 0;
            return d
        };
        x.querySelectorAll && function() {
            var a, b = F, d = /'|\\/g, r = /\=[\x20\t\r\n\f]*([^'"\]]*)[\x20\t\r\n\f]*\]/g, c = [], s = [":active"], y = B.matchesSelector || B.mozMatchesSelector ||
            B.webkitMatchesSelector || B.oMatchesSelector || B.msMatchesSelector;
            Sa(function(a) {
                a.innerHTML = "\x3cselect\x3e\x3coption selected\x3d''\x3e\x3c/option\x3e\x3c/select\x3e";
                a.querySelectorAll("[selected]").length || c.push("\\[" + U + "*(?:checked|disabled|ismap|multiple|readonly|selected|value)");
                a.querySelectorAll(":checked").length || c.push(":checked")
            });
            Sa(function(a) {
                a.innerHTML = "\x3cp test\x3d''\x3e\x3c/p\x3e";
                a.querySelectorAll("[test^\x3d'']").length && c.push("[*^$]\x3d" + U + "*(?:\"\"|'')");
                a.innerHTML =
                "\x3cinput type\x3d'hidden'/\x3e";
                a.querySelectorAll(":enabled").length || c.push(":enabled", ":disabled")
            });
            c = c.length && RegExp(c.join("|"));
            F = function(a, e, r, s, y) {
                if (!(s || y || c && c.test(a)))
                    if (9 === e.nodeType)
                        try {
                            return A.apply(r, ca.call(e.querySelectorAll(a), 0)), r
                } catch (l) {} else if (1 === e.nodeType && "object" !== e.nodeName.toLowerCase()) {
                    var h, H, m, F = e.getAttribute("id"), p = F || G, q = L.test(a) && e.parentNode || e;
                    F ? p = p.replace(d, "\\$\x26") : e.setAttribute("id", p);
                    h = f(a, e, y);
                    p = "[id\x3d'" + p + "']";
                    H = 0;
                    for (m = h.length; H <
                    m; H++)
                        h[H] = p + h[H].selector;
                        try {
                            return A.apply(r, ca.call(q.querySelectorAll(h.join(",")), 0)), r
                        } catch (S) {} finally {
                            F || e.removeAttribute("id")
                        }
                }
                return b(a, e, r, s, y)
            };
            y && (Sa(function(b) {
                a = y.call(b, "div");
                try {
                    y.call(b, "[test!\x3d'']:sizzle"), s.push(Q.PSEUDO.source, Q.POS.source, "!\x3d")
                } catch (e) {}
            }), s = RegExp(s.join("|")), e.matchesSelector = function(b, d) {
                d = d.replace(r, "\x3d'$1']");
                if (!(k(b) || s.test(d) || c && c.test(d)))
                    try {
                        var l = y.call(b, d);
                        if (l || a || b.document && 11 !== b.document.nodeType)
                            return l
                } catch (h) {}
                return 0 <
                e(d, null, null, [b]).length
            })
        }();
        J.setFilters.nth = J.setFilters.eq;
        J.filters = J.pseudos;
        e.attr = d.attr;
        d.find = e;
        d.expr = e.selectors;
        d.expr[":"] = d.expr.pseudos;
        d.unique = e.uniqueSort;
        d.text = e.getText;
        d.isXMLDoc = e.isXML;
        d.contains = e.contains
    })(c);
    var vb = /Until$/, Vb = /^(?:parents|prev(?:Until|All))/, Qb = /^.[^:#\[\.,]*$/, eb = d.expr.match.needsContext, Hb = {
        children: !0,
        contents: !0,
        next: !0,
        prev: !0
    };
    d.fn.extend({
        find: function(a) {
            var b, e, c, m, h, f, p = this;
            if ("string" !== typeof a)
                return d(a).filter(function() {
                    b = 0;
                    for (e = p.length; b <
                    e; b++)
                        if (d.contains(p[b], this))
                            return !0
                        });
            f = this.pushStack("", "find", a);
            b = 0;
            for (e = this.length; b < e; b++)
                if (c = f.length, d.find(a, this[b], f), 0 < b)
                    for (m = c; m < f.length; m++)
                        for (h = 0; h < c; h++)
                            if (f[h] === f[m]) {
                                f.splice(m--, 1);
                                break
                            }
            return f
        },
        has: function(a) {
            var b, e = d(a, this), c = e.length;
            return this.filter(function() {
                for (b = 0; b < c; b++)
                    if (d.contains(this, e[b]))
                        return !0
            })
        },
        not: function(a) {
            return this.pushStack(B(this, a, !1), "not", a)
        },
        filter: function(a) {
            return this.pushStack(B(this, a, !0), "filter", a)
        },
        is: function(a) {
            return !!a &&
            ("string" === typeof a ? eb.test(a) ? 0 <= d(a, this.context).index(this[0]) : 0 < d.filter(a, this).length : 0 < this.filter(a).length)
        },
        closest: function(a, b) {
            for (var e, c = 0, m = this.length, h = [], f = eb.test(a) || "string" !== typeof a ? d(a, b || this.context) : 0; c < m; c++)
                for (e = this[c]; e && e.ownerDocument && e !== b && 11 !== e.nodeType;) {
                    if (f?-1 < f.index(e) : d.find.matchesSelector(e, a)
                        ) {
                            h.push(e);
                            break
                        }
                        e = e.parentNode
                }
            h = 1 < h.length ? d.unique(h) : h;
            return this.pushStack(h, "closest", a)
        },
        index: function(a) {
            return a ? "string" === typeof a ? d.inArray(this[0],
            d(a)) : d.inArray(a.jquery ? a[0] : a, this) : this[0] && this[0].parentNode ? this.prevAll().length : - 1
        },
        add: function(a, b) {
            var e = "string" === typeof a ? d(a, b): d.makeArray(a && a.nodeType ? [a] : a), c = d.merge(this.get(), e);
            return this.pushStack(z(e[0]) || z(c[0]) ? c : d.unique(c))
        },
        addBack: function(a) {
            return this.add(null == a ? this.prevObject : this.prevObject.filter(a))
        }
    });
    d.fn.andSelf = d.fn.addBack;
    d.each({
        parent: function(a) {
            return (a = a.parentNode) && 11 !== a.nodeType ? a : null
        },
        parents: function(a) {
            return d.dir(a, "parentNode")
        },
        parentsUntil: function(a,
        b, e) {
            return d.dir(a, "parentNode", e)
        },
        next: function(a) {
            return w(a, "nextSibling")
        },
        prev: function(a) {
            return w(a, "previousSibling")
        },
        nextAll: function(a) {
            return d.dir(a, "nextSibling")
        },
        prevAll: function(a) {
            return d.dir(a, "previousSibling")
        },
        nextUntil: function(a, b, e) {
            return d.dir(a, "nextSibling", e)
        },
        prevUntil: function(a, b, e) {
            return d.dir(a, "previousSibling", e)
        },
        siblings: function(a) {
            return d.sibling((a.parentNode || {}).firstChild, a)
        },
        children: function(a) {
            return d.sibling(a.firstChild)
        },
        contents: function(a) {
            return d.nodeName(a,
            "iframe") ? a.contentDocument || a.contentWindow.document : d.merge([], a.childNodes)
        }
    }, function(a, b) {
        d.fn[a] = function(e, c) {
            var m = d.map(this, b, e);
            vb.test(a) || (c = e);
            c && "string" === typeof c && (m = d.filter(c, m));
            m = 1 < this.length&&!Hb[a] ? d.unique(m) : m;
            1 < this.length && Vb.test(a) && (m = m.reverse());
            return this.pushStack(m, a, $.call(arguments).join(","))
        }
    });
    d.extend({
        filter: function(a, b, e) {
            e && (a = ":not(" + a + ")");
            return 1 === b.length ? d.find.matchesSelector(b[0], a) ? [b[0]] : [] : d.find.matches(a, b)
        },
        dir: function(a, b, e) {
            var c = [];
            for (a = a[b]; a && 9 !== a.nodeType && (e === f || 1 !== a.nodeType ||!d(a).is(e));)
                1 === a.nodeType && c.push(a), a = a[b];
            return c
        },
        sibling: function(a, b) {
            for (var e = []; a; a = a.nextSibling)
                1 === a.nodeType && a !== b && e.push(a);
            return e
        }
    });
    var ha = "abbr|article|aside|audio|bdi|canvas|data|datalist|details|figcaption|figure|footer|header|hgroup|mark|meter|nav|output|progress|section|summary|time|video", wb = / jimdoGen002\d+="(?:null|\d+)"/g, Wa = /^\s+/, Ib = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\w:]+)[^>]*)\/>/gi, xb = /<([\w:]+)/,
    Wb = /<tbody/i, Xb = /<|&#?\w+;/, Yb = /<(?:script|style|link)/i, Jb = /<(?:script|object|embed|option|style)/i, yb = RegExp("\x3c(?:" + ha + ")[\\s/\x3e]", "i"), mb = /^(?:checkbox|radio)$/, Kb = /checked\s*(?:[^=]|=\s*.checked.)/i, Zb = /\/(java|ecma)script/i, $b = /^\s*<!(?:\[CDATA\[|\-\-)|[\]\-]{2}>\s*$/g, ta = {
        option: [1, "\x3cselect multiple\x3d'multiple'\x3e", "\x3c/select\x3e"],
        legend: [1, "\x3cfieldset\x3e", "\x3c/fieldset\x3e"],
        thead: [1, "\x3ctable\x3e", "\x3c/table\x3e"],
        tr: [2, "\x3ctable\x3e\x3ctbody\x3e", "\x3c/tbody\x3e\x3c/table\x3e"],
        td: [3, "\x3ctable\x3e\x3ctbody\x3e\x3ctr\x3e", "\x3c/tr\x3e\x3c/tbody\x3e\x3c/table\x3e"],
        col: [2, "\x3ctable\x3e\x3ctbody\x3e\x3c/tbody\x3e\x3ccolgroup\x3e", "\x3c/colgroup\x3e\x3c/table\x3e"],
        area: [1, "\x3cmap\x3e", "\x3c/map\x3e"],
        _default: [0, "", ""]
    }, Lb = E(C), zb = Lb.appendChild(C.createElement("div"));
    ta.optgroup = ta.option;
    ta.tbody = ta.tfoot = ta.colgroup = ta.caption = ta.thead;
    ta.th = ta.td;
    d.support.htmlSerialize || (ta._default = [1, "X\x3cdiv\x3e", "\x3c/div\x3e"]);
    d.fn.extend({
        text: function(a) {
            return d.access(this,
            function(a) {
                return a === f ? d.text(this) : this.empty().append((this[0] && this[0].ownerDocument || C).createTextNode(a))
            }, null, a, arguments.length)
        },
        wrapAll: function(a) {
            if (d.isFunction(a))
                return this.each(function(b) {
                    d(this).wrapAll(a.call(this, b))
                });
            if (this[0]) {
                var b = d(a, this[0].ownerDocument).eq(0).clone(!0);
                this[0].parentNode && b.insertBefore(this[0]);
                b.map(function() {
                    for (var a = this; a.firstChild && 1 === a.firstChild.nodeType;)
                        a = a.firstChild;
                    return a
                }).append(this)
            }
            return this
        },
        wrapInner: function(a) {
            return d.isFunction(a) ?
            this.each(function(b) {
                d(this).wrapInner(a.call(this, b))
            }) : this.each(function() {
                var b = d(this), e = b.contents();
                e.length ? e.wrapAll(a) : b.append(a)
            })
        },
        wrap: function(a) {
            var b = d.isFunction(a);
            return this.each(function(e) {
                d(this).wrapAll(b ? a.call(this, e) : a)
            })
        },
        unwrap: function() {
            return this.parent().each(function() {
                d.nodeName(this, "body") || d(this).replaceWith(this.childNodes)
            }).end()
        },
        append: function() {
            return this.domManip(arguments, !0, function(a) {
                1 !== this.nodeType && 11 !== this.nodeType || this.appendChild(a)
            })
        },
        prepend: function() {
            return this.domManip(arguments, !0, function(a) {
                1 !== this.nodeType && 11 !== this.nodeType || this.insertBefore(a, this.firstChild)
            })
        },
        before: function() {
            if (!z(this[0]))
                return this.domManip(arguments, !1, function(a) {
                    this.parentNode.insertBefore(a, this)
                });
            if (arguments.length) {
                var a = d.clean(arguments);
                return this.pushStack(d.merge(a, this), "before", this.selector)
            }
        },
        after: function() {
            if (!z(this[0]))
                return this.domManip(arguments, !1, function(a) {
                    this.parentNode.insertBefore(a, this.nextSibling)
                });
            if (arguments.length) {
                var a = d.clean(arguments);
                return this.pushStack(d.merge(this, a), "after", this.selector)
            }
        },
        remove: function(a, b) {
            for (var e, c = 0; null != (e = this[c]); c++)
                if (!a || d.filter(a, [e]).length)
                    b || 1 !== e.nodeType || (d.cleanData(e.getElementsByTagName("*")), d.cleanData([e])), e.parentNode && e.parentNode.removeChild(e);
            return this
        },
        empty: function() {
            for (var a, b = 0; null != (a = this[b]); b++)
                for (1 === a.nodeType && d.cleanData(a.getElementsByTagName("*")); a.firstChild;)
                    a.removeChild(a.firstChild);
            return this
        },
        clone: function(a,
        b) {
            a = null == a?!1 : a;
            b = null == b ? a : b;
            return this.map(function() {
                return d.clone(this, a, b)
            })
        },
        html: function(a) {
            return d.access(this, function(a) {
                var e = this[0] || {}, c = 0, m = this.length;
                if (a === f)
                    return 1 === e.nodeType ? e.innerHTML.replace(wb, "") : f;
                if ("string" === typeof a&&!(Yb.test(a) ||!d.support.htmlSerialize && yb.test(a) ||!d.support.leadingWhitespace && Wa.test(a) || ta[(xb.exec(a) || ["", ""])[1].toLowerCase()])) {
                    a = a.replace(Ib, "\x3c$1\x3e\x3c/$2\x3e");
                    try {
                        for (; c < m; c++)
                            e = this[c] || {}, 1 === e.nodeType && (d.cleanData(e.getElementsByTagName("*")),
                            e.innerHTML = a);
                        e = 0
                    } catch (h) {}
                }
                e && this.empty().append(a)
            }, null, a, arguments.length)
        },
        replaceWith: function(a) {
            if (!z(this[0])) {
                if (d.isFunction(a))
                    return this.each(function(b) {
                        var e = d(this), c = e.html();
                        e.replaceWith(a.call(this, b, c))
                    });
                "string" !== typeof a && (a = d(a).detach());
                return this.each(function() {
                    var b = this.nextSibling, e = this.parentNode;
                    d(this).remove();
                    b ? d(b).before(a) : d(e).append(a)
                })
            }
            return this.length ? this.pushStack(d(d.isFunction(a) ? a() : a), "replaceWith", a) : this
        },
        detach: function(a) {
            return this.remove(a,
            !0)
        },
        domManip: function(a, b, e) {
            a = [].concat.apply([], a);
            var c, m, h, p = 0, q = a[0], g = [], r = this.length;
            if (!d.support.checkClone && 1 < r && "string" === typeof q && Kb.test(q))
                return this.each(function() {
                    d(this).domManip(a, b, e)
                });
            if (d.isFunction(q))
                return this.each(function(r) {
                    var c = d(this);
                    a[0] = q.call(this, r, b ? c.html() : f);
                    c.domManip(a, b, e)
                });
            if (this[0]) {
                c = d.buildFragment(a, this, g);
                h = c.fragment;
                m = h.firstChild;
                1 === h.childNodes.length && (h = m);
                if (m)
                    for (b = b && d.nodeName(m, "tr"), c = c.cacheable || r - 1; p < r; p++)
                        e.call(b && d.nodeName(this[p],
                        "table") ? this[p].getElementsByTagName("tbody")[0] || this[p].appendChild(this[p].ownerDocument.createElement("tbody")) : this[p], p === c ? h : d.clone(h, !0, !0));
                h = m = null;
                g.length && d.each(g, function(a, b) {
                    b.src ? d.ajax ? d.ajax({
                        url: b.src,
                        type: "GET",
                        dataType: "script",
                        async: !1,
                        global: !1,
                        "throws": !0
                    }) : d.error("no ajax") : d.globalEval((b.text || b.textContent || b.innerHTML || "").replace($b, ""));
                    b.parentNode && b.parentNode.removeChild(b)
                })
            }
            return this
        }
    });
    d.buildFragment = function(a, b, e) {
        var c, m, h, p = a[0];
        b = b || C;
        b=!b.nodeType &&
        b[0] || b;
        b = b.ownerDocument || b;
        1 === a.length && "string" === typeof p && 512 > p.length && b === C && "\x3c" === p.charAt(0)&&!(Jb.test(p) ||!d.support.checkClone && Kb.test(p) ||!d.support.html5Clone && yb.test(p)) && (m=!0, c = d.fragments[p], h = c !== f);
        c || (c = b.createDocumentFragment(), d.clean(a, b, c, e), m && (d.fragments[p] = h && c));
        return {
            fragment: c,
            cacheable: m
        }
    };
    d.fragments = {};
    d.each({
        appendTo: "append",
        prependTo: "prepend",
        insertBefore: "before",
        insertAfter: "after",
        replaceAll: "replaceWith"
    }, function(a, b) {
        d.fn[a] = function(e) {
            var c,
            m = 0, h = [];
            e = d(e);
            var f = e.length;
            c = 1 === this.length && this[0].parentNode;
            if ((null == c || c && 11 === c.nodeType && 1 === c.childNodes.length) && 1 === f)
                return e[b](this[0]), this;
            for (; m < f; m++)
                c = (0 < m ? this.clone(!0) : this).get(), d(e[m])[b](c), h = h.concat(c);
            return this.pushStack(h, a, e.selector)
        }
    });
    d.extend({
        clone: function(a, b, e) {
            var c, m, h, f;
            d.support.html5Clone || d.isXMLDoc(a) ||!yb.test("\x3c" + a.nodeName + "\x3e") ? f = a.cloneNode(!0) : (zb.innerHTML = a.outerHTML, zb.removeChild(f = zb.firstChild));
            if (!(d.support.noCloneEvent && d.support.noCloneChecked ||
            1 !== a.nodeType && 11 !== a.nodeType || d.isXMLDoc(a)))
                for (x(a, f)
                    , c = I(a), m = I(f), h = 0;
            c[h];
            ++h)m[h] && x(c[h], m[h]);
            if (b && (A(a, f), e))
                for (c = I(a), m = I(f), h = 0; c[h]; ++h)
                    A(c[h], m[h]);
            return f
        },
        clean: function(a, b, e, c) {
            var m, h, f, p, g, r, s = b === C && Lb, y = [];
            b && "undefined" !== typeof b.createDocumentFragment || (b = C);
            for (m = 0; null != (f = a[m]); m++)
                if ("number" === typeof f && (f += ""), f) {
                    if ("string" === typeof f)
                        if (Xb.test(f)) {
                            s = s || E(b);
                            r = b.createElement("div");
                            s.appendChild(r);
                            f = f.replace(Ib, "\x3c$1\x3e\x3c/$2\x3e");
                            h = (xb.exec(f) || ["", ""])[1].toLowerCase();
                            p = ta[h] || ta._default;
                            g = p[0];
                            for (r.innerHTML = p[1] + f + p[2]; g--;)
                                r = r.lastChild;
                                if (!d.support.tbody)
                                    for (g = Wb.test(f), p = "table" !== h || g ? "\x3ctable\x3e" !== p[1] || g ? [] : r.childNodes : r.firstChild && r.firstChild.childNodes, h = p.length - 1; 0 <= h; --h)
                                        d.nodeName(p[h], "tbody")&&!p[h].childNodes.length && p[h].parentNode.removeChild(p[h]);
                                        !d.support.leadingWhitespace && Wa.test(f) && r.insertBefore(b.createTextNode(Wa.exec(f)[0]), r.firstChild);
                                        f = r.childNodes;
                                        r.parentNode.removeChild(r)
                        } else 
                            f = b.createTextNode(f);
                            f.nodeType ? y.push(f) :
                            d.merge(y, f)
                }
            r && (f = r = s = null);
            if (!d.support.appendChecked)
                for (m = 0; null != (f = y[m]); m++)
                    d.nodeName(f, "input") ? K(f) : "undefined" !== typeof f.getElementsByTagName && d.grep(f.getElementsByTagName("input"), K);
            if (e)
                for (a = function(a) {
                    if (!a.type || Zb.test(a.type))
                        return c ? c.push(a.parentNode ? a.parentNode.removeChild(a) : a) : e.appendChild(a)
                    }, m = 0; null != (f = y[m]); m++)
                        d.nodeName(f, "script") && a(f) || (e.appendChild(f), "undefined" !== typeof f.getElementsByTagName && (f = d.grep(d.merge([], f.getElementsByTagName("script")), a),
                        y.splice.apply(y, [m + 1, 0].concat(f)), m += f.length));
            return y
        },
        cleanData: function(a, b) {
            for (var e, c, f, h, p = 0, q = d.expando, g = d.cache, r = d.support.deleteExpando, s = d.event.special; null != (f = a[p]); p++)
                if (b || d.acceptData(f))
                    if (e = (c = f[q]) && g[c]) {
                        if (e.events)
                            for (h in e.events)
                                s[h] ? d.event.remove(f, h) : d.removeEvent(f, h, e.handle);
                                g[c] && (delete g[c], r ? delete f[q] : f.removeAttribute ? f.removeAttribute(q) : f[q] = null, d.deletedIds.push(c))
                    }
        }
    });
    (function() {
        var a, b;
        d.uaMatch = function(a) {
            a = a.toLowerCase();
            a = /(chrome)[ \/]([\w.]+)/.exec(a) ||
            /(webkit)[ \/]([\w.]+)/.exec(a) || /(opera)(?:.*version|)[ \/]([\w.]+)/.exec(a) || /(msie) ([\w.]+)/.exec(a) || 0 > a.indexOf("compatible") && /(mozilla)(?:.*? rv:([\w.]+)|)/.exec(a) || [];
            return {
                browser: a[1] || "",
                version: a[2] || "0"
            }
        };
        a = d.uaMatch(Ra.userAgent);
        b = {};
        a.browser && (b[a.browser]=!0, b.version = a.version);
        b.chrome ? b.webkit=!0 : b.webkit && (b.safari=!0);
        d.browser = b;
        d.sub = function() {
            function a(b, d) {
                return new a.fn.init(b, d)
            }
            d.extend(!0, a, this);
            a.superclass = this;
            a.fn = a.prototype = this ();
            a.fn.constructor = a;
            a.sub =
            this.sub;
            a.fn.init = function(c, h) {
                h && h instanceof d&&!(h instanceof a) && (h = a(h));
                return d.fn.init.call(this, c, h, b)
            };
            a.fn.init.prototype = a.fn;
            var b = a(C);
            return a
        }
    })();
    var da, Pa, Va, Ab = /alpha\([^)]*\)/i, ac = /opacity=([^)]*)/, za = /^(top|right|bottom|left)$/, hc = /^(none|table(?!-c[ea]).+)/, Bb = /^margin/, nb = RegExp("^(" + ra + ")(.*)$", "i"), Na = RegExp("^(" + ra + ")(?!px)[a-z%]+$", "i"), ic = RegExp("^([-+])\x3d(" + ra + ")", "i"), Oa = {}, fb = {
        position: "absolute",
        visibility: "hidden",
        display: "block"
    }, La = {
        letterSpacing: 0,
        fontWeight: 400
    },
    ia = ["Top", "Right", "Bottom", "Left"], Ua = ["Webkit", "O", "Moz", "ms"], bc = d.fn.toggle;
    d.fn.extend({
        css: function(a, b) {
            return d.access(this, function(a, b, c) {
                return c !== f ? d.style(a, b, c) : d.css(a, b)
            }, a, b, 1 < arguments.length)
        },
        show: function() {
            return W(this, !0)
        },
        hide: function() {
            return W(this)
        },
        toggle: function(a, b) {
            var e = "boolean" === typeof a;
            return d.isFunction(a) && d.isFunction(b) ? bc.apply(this, arguments) : this.each(function() {
                (e ? a : O(this)) ? d(this).show() : d(this).hide()
            })
        }
    });
    d.extend({
        cssHooks: {
            opacity: {
                get: function(a,
                b) {
                    if (b) {
                        var e = da(a, "opacity");
                        return "" === e ? "1" : e
                    }
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
            "float": d.support.cssFloat ? "cssFloat": "styleFloat"
        },
        style: function(a, b, e, c) {
            if (a && 3 !== a.nodeType && 8 !== a.nodeType && a.style) {
                var p, h, u, q = d.camelCase(b), g = a.style;
                b = d.cssProps[q] || (d.cssProps[q] = D(g, q));
                u = d.cssHooks[b] || d.cssHooks[q];
                if (e !== f) {
                    if (h = typeof e, "string" === h && (p = ic.exec(e)) && (e = (p[1] + 1) * p[2] + parseFloat(d.css(a, b)), h = "number"),
                    !(null == e || "number" === h && isNaN(e) || ("number" !== h || d.cssNumber[q] || (e += "px"), u && "set"in u && (e = u.set(a, e, c)) === f)))
                        try {
                            g[b] = e
                    } catch (r) {}
                } else 
                    return u && "get"in u && (p = u.get(a, !1, c)) !== f ? p : g[b]
            }
        },
        css: function(a, b, e, c) {
            var p, h;
            h = d.camelCase(b);
            b = d.cssProps[h] || (d.cssProps[h] = D(a.style, h));
            (h = d.cssHooks[b] || d.cssHooks[h]) && "get"in h && (p = h.get(a, !0, c));
            p === f && (p = da(a, b));
            "normal" === p && b in La && (p = La[b]);
            return e || c !== f ? (a = parseFloat(p), e || d.isNumeric(a) ? a || 0 : p) : p
        },
        swap: function(a, b, e) {
            var d, c = {};
            for (d in b)
                c[d] =
                a.style[d], a.style[d] = b[d];
            e = e.call(a);
            for (d in b)
                a.style[d] = c[d];
            return e
        }
    });
    c.getComputedStyle ? da = function(a, b) {
        var e, f, p, h, u = c.getComputedStyle(a, null), q = a.style;
        u && (e = u[b], "" !== e || d.contains(a.ownerDocument, a) || (e = d.style(a, b)), Na.test(e) && Bb.test(b) && (f = q.width, p = q.minWidth, h = q.maxWidth, q.minWidth = q.maxWidth = q.width = e, e = u.width, q.width = f, q.minWidth = p, q.maxWidth = h));
        return e
    } : C.documentElement.currentStyle && (da = function(a, b) {
        var d, c, f = a.currentStyle && a.currentStyle[b], h = a.style;
        null == f && h && h[b] &&
        (f = h[b]);
        if (Na.test(f)&&!za.test(b)) {
            d = h.left;
            if (c = a.runtimeStyle && a.runtimeStyle.left)
                a.runtimeStyle.left = a.currentStyle.left;
            h.left = "fontSize" === b ? "1em" : f;
            f = h.pixelLeft + "px";
            h.left = d;
            c && (a.runtimeStyle.left = c)
        }
        return "" === f ? "auto" : f
    });
    d.each(["height", "width"], function(a, b) {
        d.cssHooks[b] = {
            get: function(a, c, f) {
                if (c)
                    return 0 === a.offsetWidth && hc.test(da(a, "display")) ? d.swap(a, fb, function() {
                        return Z(a, b, f)
                    }) : Z(a, b, f)
            },
            set: function(a, c, f) {
                return N(a, c, f ? P(a, b, f, d.support.boxSizing && "border-box" === d.css(a,
                "boxSizing")) : 0)
            }
        }
    });
    d.support.opacity || (d.cssHooks.opacity = {
        get: function(a, b) {
            return ac.test((b && a.currentStyle ? a.currentStyle.filter : a.style.filter) || "") ? 0.01 * parseFloat(RegExp.$1) + "" : b ? "1" : ""
        },
        set: function(a, b) {
            var e = a.style, c = a.currentStyle, f = d.isNumeric(b) ? "alpha(opacity\x3d" + 100 * b + ")": "", h = c && c.filter || e.filter || "";
            e.zoom = 1;
            if (1 <= b && "" === d.trim(h.replace(Ab, "")) && e.removeAttribute && (e.removeAttribute("filter"), c&&!c.filter))
                return;
            e.filter = Ab.test(h) ? h.replace(Ab, f) : h + " " + f
        }
    });
    d(function() {
        d.support.reliableMarginRight ||
        (d.cssHooks.marginRight = {
            get: function(a, b) {
                return d.swap(a, {
                    display: "inline-block"
                }, function() {
                    if (b)
                        return da(a, "marginRight")
                })
            }
        });
        !d.support.pixelPosition && d.fn.position && d.each(["top", "left"], function(a, b) {
            d.cssHooks[b] = {
                get: function(a, c) {
                    if (c) {
                        var f = da(a, b);
                        return Na.test(f) ? d(a).position()[b] + "px" : f
                    }
                }
            }
        })
    });
    d.expr && d.expr.filters && (d.expr.filters.hidden = function(a) {
        return 0 === a.offsetWidth && 0 === a.offsetHeight ||!d.support.reliableHiddenOffsets && "none" === (a.style && a.style.display || da(a, "display"))
    },
    d.expr.filters.visible = function(a) {
        return !d.expr.filters.hidden(a)
    });
    d.each({
        margin: "",
        padding: "",
        border: "Width"
    }, function(a, b) {
        d.cssHooks[a + b] = {
            expand: function(d) {
                var c = "string" === typeof d ? d.split(" "): [d], f = {};
                for (d = 0; 4 > d; d++)
                    f[a + ia[d] + b] = c[d] || c[d - 2] || c[0];
                return f
            }
        };
        Bb.test(a) || (d.cssHooks[a + b].set = N)
    });
    var jc = /%20/g, pa = /\[\]$/, cc = /\r?\n/g, kc = /^(?:color|date|datetime|datetime-local|email|hidden|month|number|password|range|search|tel|text|time|url|week)$/i, Cb = /^(?:select|textarea)/i;
    d.fn.extend({
        serialize: function() {
            return d.param(this.serializeArray())
        },
        serializeArray: function() {
            return this.map(function() {
                return this.elements ? d.makeArray(this.elements) : this
            }).filter(function() {
                return this.name&&!this.disabled && (this.checked || Cb.test(this.nodeName) || kc.test(this.type))
            }).map(function(a, b) {
                var e = d(this).val();
                return null == e ? null : d.isArray(e) ? d.map(e, function(a, d) {
                    return {
                        name: b.name,
                        value: a.replace(cc, "\r\n")
                    }
                }) : {
                    name: b.name,
                    value: e.replace(cc, "\r\n")
                }
            }).get()
        }
    });
    d.param = function(a, b) {
        var e, c = [], p = function(a, b) {
            b = d.isFunction(b) ? b() : null == b ? "" : b;
            c[c.length] =
            encodeURIComponent(a) + "\x3d" + encodeURIComponent(b)
        };
        b === f && (b = d.ajaxSettings && d.ajaxSettings.traditional);
        if (d.isArray(a) || a.jquery&&!d.isPlainObject(a))
            d.each(a, function() {
                p(this.name, this.value)
            });
        else 
            for (e in a)
                ja(e, a[e], b, p);
        return c.join("\x26").replace(jc, "+")
    };
    var Xa, Ya, bb = /#.*$/, ua = /^(.*?):[ \t]*([^\r\n]*)\r?$/mg, gb = /^(?:GET|HEAD)$/, hb = /^\/\//, Mb = /\?/, lc = /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, dc = /([?&])_=[^&]*/, ib = /^([\w\+\.\-]+:)(?:\/\/([^\/?#:]*)(?::(\d+)|)|)/, Db = d.fn.load,
    Fb = {}, ec = {}, jb = ["*/"] + ["*"];
    try {
        Xa = Ia.href
    } catch (Ea) {
        Xa = C.createElement("a"), Xa.href = "", Xa = Xa.href
    }
    Ya = ib.exec(Xa.toLowerCase()) || [];
    d.fn.load = function(a, b, e) {
        if ("string" !== typeof a && Db)
            return Db.apply(this, arguments);
        if (!this.length)
            return this;
        var c, p, h, g = this, q = a.indexOf(" ");
        0 <= q && (c = a.slice(q, a.length), a = a.slice(0, q));
        d.isFunction(b) ? (e = b, b = f) : b && "object" === typeof b && (p = "POST");
        d.ajax({
            url: a,
            type: p,
            dataType: "html",
            data: b,
            complete: function(a, b) {
                e && g.each(e, h || [a.responseText, b, a])
            }
        }).done(function(a) {
            h =
            arguments;
            g.html(c ? d("\x3cdiv\x3e").append(a.replace(lc, "")).find(c) : a)
        });
        return this
    };
    d.each("ajaxStart ajaxStop ajaxComplete ajaxError ajaxSuccess ajaxSend".split(" "), function(a, b) {
        d.fn[b] = function(a) {
            return this.on(b, a)
        }
    });
    d.each(["get", "post"], function(a, b) {
        d[b] = function(a, c, p, h) {
            d.isFunction(c) && (h = h || p, p = c, c = f);
            return d.ajax({
                type: b,
                url: a,
                data: c,
                success: p,
                dataType: h
            })
        }
    });
    d.extend({
        getScript: function(a, b) {
            return d.get(a, f, b, "script")
        },
        getJSON: function(a, b, e) {
            return d.get(a, b, e, "json")
        },
        ajaxSetup: function(a,
        b) {
            b ? qa(a, d.ajaxSettings) : (b = a, a = d.ajaxSettings);
            qa(a, b);
            return a
        },
        ajaxSettings: {
            url: Xa,
            isLocal: /^(?:about|app|app\-storage|.+\-extension|file|res|widget):$/.test(Ya[1]),
            global: !0,
            type: "GET",
            contentType: "application/x-www-form-urlencoded; charset\x3dUTF-8",
            processData: !0,
            async: !0,
            accepts: {
                xml: "application/xml, text/xml",
                html: "text/html",
                text: "text/plain",
                json: "application/json, text/javascript",
                "*": jb
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
                "* text": c.String,
                "text html": !0,
                "text json": d.parseJSON,
                "text xml": d.parseXML
            },
            flatOptions: {
                context: !0,
                url: !0
            }
        },
        ajaxPrefilter: U(Fb),
        ajaxTransport: U(ec),
        ajax: function(a, b) {
            function e(a, b, e, s) {
                var h, n, J, k, v = b;
                if (2 !== Z) {
                    Z = 2;
                    q && clearTimeout(q);
                    g = f;
                    p = s || "";
                    t.readyState = 0 < a ? 4 : 0;
                    if (e) {
                        k = y;
                        s = t;
                        var z, G, w, x, B = k.contents, ca = k.dataTypes, oc = k.responseFields;
                        for (G in oc)
                            G in e && (s[oc[G]] = e[G]);
                        for (; "*" === ca[0];)
                            ca.shift(), z === f && (z = k.mimeType || s.getResponseHeader("content-type"));
                        if (z)
                            for (G in B)
                                if (B[G] &&
                                B[G].test(z)) {
                                    ca.unshift(G);
                                    break
                                }
                        if (ca[0]in e)
                            w = ca[0];
                        else {
                            for (G in e) {
                                if (!ca[0] || k.converters[G + " " + ca[0]]) {
                                    w = G;
                                    break
                                }
                                x || (x = G)
                            }
                            w = w || x
                        }
                        w ? (w !== ca[0] && ca.unshift(w), k = e[w]) : k = void 0
                    }
                    if (200 <= a && 300 > a || 304 === a)
                        if (y.ifModified && ((e = t.getResponseHeader("Last-Modified")) && (d.lastModified[c] = e), (e = t.getResponseHeader("Etag")) && (d.etag[c] = e)), 304 === a)
                            v = "notmodified", h=!0;
                        else {
                            a:
                            {
                                n = y;
                                J = k;
                                var A, N, v = n.dataTypes.slice();
                                z = v[0];
                                G = {};
                                w = 0;
                                n.dataFilter && (J = n.dataFilter(J, n.dataType));
                                if (v[1])
                                    for (A in n.converters)
                                        G[A.toLowerCase()] =
                                        n.converters[A];
                                        for (; e = v[++w];)
                                            if ("*" !== e) {
                                                if ("*" !== z && z !== e) {
                                                    A = G[z + " " + e] || G["* " + e];
                                                    if (!A)
                                                        for (N in G)
                                                            if (h = N.split(" "), h[1] === e && (A = G[z + " " + h[0]] || G["* " + h[0]])) {
                                                                !0 === A ? A = G[N] : !0 !== G[N] && (e = h[0], v.splice(w--, 0, e));
                                                                break
                                                            }
                                                            if (!0 !== A)
                                                                if (A && n["throws"])
                                                                    J = A(J);
                                                                else 
                                                                    try {
                                                                        J = A(J)
                                                                    } catch (U) {
                                                                        h = {
                                                                            state: "parsererror",
                                                                            error: A ? U: "No conversion from " + z + " to " + e
                                                                        };
                                                                        break a
                                                                    }
                                                                }
                                                                z = e
                                            }
                                            h = {
                                                state: "success",
                                                data: J
                                            }
                                        }
                                        v = h.state;
                                        n = h.data;
                                        J = h.error;
                                        h=!J
                    } else if (J = v, !v || a)
                        v = "error", 0 > a && (a = 0);
                    t.status = a;
                    t.statusText = "" + (b || v);
                    h ? S.resolveWith(H,
                    [n, v, t]) : S.rejectWith(H, [t, v, J]);
                    t.statusCode(P);
                    P = f;
                    r && F.trigger("ajax" + (h ? "Success" : "Error"), [t, y, h ? n: J]);
                    Ca.fireWith(H, [t, v]);
                    r && (F.trigger("ajaxComplete", [t, y]), --d.active || d.event.trigger("ajaxStop"))
                }
            }
            "object" === typeof a && (b = a, a = f);
            b = b || {};
            var c, p, h, g, q, n, r, s, y = d.ajaxSetup({}, b), H = y.context || y, F = H !== y && (H.nodeType || H instanceof d) ? d(H): d.event, S = d.Deferred(), Ca = d.Callbacks("once memory"), P = y.statusCode || {}, J = {}, k = {}, Z = 0, v = "canceled", t = {
                readyState: 0,
                setRequestHeader: function(a, b) {
                    if (!Z) {
                        var d =
                        a.toLowerCase();
                        a = k[d] = k[d] || a;
                        J[a] = b
                    }
                    return this
                },
                getAllResponseHeaders: function() {
                    return 2 === Z ? p : null
                },
                getResponseHeader: function(a) {
                    var b;
                    if (2 === Z) {
                        if (!h)
                            for (h = {}; b = ua.exec(p);)
                                h[b[1].toLowerCase()] = b[2];
                        b = h[a.toLowerCase()]
                    }
                    return b === f ? null : b
                },
                overrideMimeType: function(a) {
                    Z || (y.mimeType = a);
                    return this
                },
                abort: function(a) {
                    a = a || v;
                    g && g.abort(a);
                    e(0, a);
                    return this
                }
            };
            S.promise(t);
            t.success = t.done;
            t.error = t.fail;
            t.complete = Ca.add;
            t.statusCode = function(a) {
                if (a) {
                    var b;
                    if (2 > Z)
                        for (b in a)
                            P[b] = [P[b], a[b]];
                    else 
                        b = a[t.status], t.always(b)
                }
                return this
            };
            y.url = ((a || y.url) + "").replace(bb, "").replace(hb, Ya[1] + "//");
            y.dataTypes = d.trim(y.dataType || "*").toLowerCase().split(aa);
            null == y.crossDomain && (n = ib.exec(y.url.toLowerCase()), y.crossDomain=!(!n || n[1] == Ya[1] && n[2] == Ya[2] && (n[3] || ("http:" === n[1] ? 80 : 443)) == (Ya[3] || ("http:" === Ya[1] ? 80 : 443))));
            y.data && y.processData && "string" !== typeof y.data && (y.data = d.param(y.data, y.traditional));
            R(Fb, y, b, t);
            if (2 === Z)
                return t;
            r = y.global;
            y.type = y.type.toUpperCase();
            y.hasContent =
            !gb.test(y.type);
            r && 0 === d.active++&&d.event.trigger("ajaxStart");
            if (!y.hasContent && (y.data && (y.url += (Mb.test(y.url) ? "\x26" : "?") + y.data, delete y.data), c = y.url, !1 === y.cache)) {
                n = d.now();
                var z = y.url.replace(dc, "$1_\x3d" + n);
                y.url = z + (z === y.url ? (Mb.test(y.url) ? "\x26" : "?") + "_\x3d" + n : "")
            }(y.data && y.hasContent&&!1 !== y.contentType || b.contentType) && t.setRequestHeader("Content-Type", y.contentType);
            y.ifModified && (c = c || y.url, d.lastModified[c] && t.setRequestHeader("If-Modified-Since", d.lastModified[c]), d.etag[c] &&
            t.setRequestHeader("If-None-Match", d.etag[c]));
            t.setRequestHeader("Accept", y.dataTypes[0] && y.accepts[y.dataTypes[0]] ? y.accepts[y.dataTypes[0]] + ("*" !== y.dataTypes[0] ? ", " + jb + "; q\x3d0.01" : "") : y.accepts["*"]);
            for (s in y.headers)
                t.setRequestHeader(s, y.headers[s]);
            if (y.beforeSend && (!1 === y.beforeSend.call(H, t, y) || 2 === Z))
                return t.abort();
            v = "abort";
            for (s in{
                success: 1,
                error: 1,
                complete: 1
            })
                t[s](y[s]);
            if (g = R(ec, y, b, t)) {
                t.readyState = 1;
                r && F.trigger("ajaxSend", [t, y]);
                y.async && 0 < y.timeout && (q = setTimeout(function() {
                    t.abort("timeout")
                },
                y.timeout));
                try {
                    Z = 1, g.send(J, e)
                } catch (G) {
                    if (2 > Z)
                        e( - 1, G);
                    else 
                        throw G;
                }
            } else 
                e( - 1, "No Transport");
            return t
        },
        active: 0,
        lastModified: {},
        etag: {}
    });
    var cb = [], Nb = /\?/, kb = /(=)\?(?=&|$)|\?\?/, Ma = d.now();
    d.ajaxSetup({
        jsonp: "callback",
        jsonpCallback: function() {
            var a = cb.pop() || d.expando + "_" + Ma++;
            this[a]=!0;
            return a
        }
    });
    d.ajaxPrefilter("json jsonp", function(a, b, e) {
        var p, m, h, g = a.data, q = a.url, n=!1 !== a.jsonp, r = n && kb.test(q), s = n&&!r && "string" === typeof g&&!(a.contentType || "").indexOf("application/x-www-form-urlencoded") &&
        kb.test(g);
        if ("jsonp" === a.dataTypes[0] || r || s)
            return p = a.jsonpCallback = d.isFunction(a.jsonpCallback) ? a.jsonpCallback() : a.jsonpCallback, m = c[p], r ? a.url = q.replace(kb, "$1" + p) : s ? a.data = g.replace(kb, "$1" + p) : n && (a.url += (Nb.test(q) ? "\x26" : "?") + a.jsonp + "\x3d" + p), a.converters["script json"] = function() {
                h || d.error(p + " was not called");
                return h[0]
            }, a.dataTypes[0] = "json", c[p] = function() {
                h = arguments
            }, e.always(function() {
                c[p] = m;
                a[p] && (a.jsonpCallback = b.jsonpCallback, cb.push(p));
                h && d.isFunction(m) && m(h[0]);
                h = m = f
            }),
            "script"
    });
    d.ajaxSetup({
        accepts: {
            script: "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript"
        },
        contents: {
            script: /javascript|ecmascript/
        },
        converters: {
            "text script": function(a) {
                d.globalEval(a);
                return a
            }
        }
    });
    d.ajaxPrefilter("script", function(a) {
        a.cache === f && (a.cache=!1);
        a.crossDomain && (a.type = "GET", a.global=!1)
    });
    d.ajaxTransport("script", function(a) {
        if (a.crossDomain) {
            var b, d = C.head || C.getElementsByTagName("head")[0] || C.documentElement;
            return {
                send: function(c, p) {
                    b =
                    C.createElement("script");
                    b.async = "async";
                    a.scriptCharset && (b.charset = a.scriptCharset);
                    b.src = a.url;
                    b.onload = b.onreadystatechange = function(a, c) {
                        if (c ||!b.readyState || /loaded|complete/.test(b.readyState))
                            b.onload = b.onreadystatechange = null, d && b.parentNode && d.removeChild(b), b = f, c || p(200, "success")
                    };
                    d.insertBefore(b, d.firstChild)
                },
                abort: function() {
                    if (b)
                        b.onload(0, 1)
                }
            }
        }
    });
    var Fa, lb = c.ActiveXObject ? function() {
        for (var a in Fa)
            Fa[a](0, 1)
    }
    : !1, oa = 0;
    d.ajaxSettings.xhr = c.ActiveXObject ? function() {
        var a;
        if (!(a=!this.isLocal &&
        V()))
            a: {
            try {
                a = new c.ActiveXObject("Microsoft.XMLHTTP");
                break a
            } catch (b) {}
            a = void 0
        }
        return a
    } : V;
    (function(a) {
        d.extend(d.support, {
            ajax: !!a,
            cors: !!a && "withCredentials"in a
        })
    })(d.ajaxSettings.xhr());
    d.support.ajax && d.ajaxTransport(function(a) {
        if (!a.crossDomain || d.support.cors) {
            var b;
            return {
                send: function(e, p) {
                    var m, h, g = a.xhr();
                    a.username ? g.open(a.type, a.url, a.async, a.username, a.password) : g.open(a.type, a.url, a.async);
                    if (a.xhrFields)
                        for (h in a.xhrFields)
                            g[h] = a.xhrFields[h];
                    a.mimeType && g.overrideMimeType &&
                    g.overrideMimeType(a.mimeType);
                    a.crossDomain || e["X-Requested-With"] || (e["X-Requested-With"] = "XMLHttpRequest");
                    try {
                        for (h in e)
                            g.setRequestHeader(h, e[h])
                    } catch (q) {}
                    g.send(a.hasContent && a.data || null);
                    b = function(e, r) {
                        var c, y, h, F, S;
                        try {
                            if (b && (r || 4 === g.readyState))
                                if (b = f, m && (g.onreadystatechange = d.noop, lb && delete Fa[m]), r)
                                    4 !== g.readyState && g.abort();
                                else {
                                    c = g.status;
                                    h = g.getAllResponseHeaders();
                                    F = {};
                                    (S = g.responseXML) && S.documentElement && (F.xml = S);
                                    try {
                                        F.text = g.responseText
                                    } catch (q) {}
                                    try {
                                        y = g.statusText
                                    } catch (n) {
                                        y =
                                        ""
                                    }
                                    c ||!a.isLocal || a.crossDomain ? 1223 === c && (c = 204) : c = F.text ? 200 : 404
                                }
                        } catch (J) {
                            r || p( - 1, J)
                        }
                        F && p(c, y, F, h)
                    };
                    a.async ? 4 === g.readyState ? setTimeout(b, 0) : (m=++oa, lb && (Fa || (Fa = {}, d(c).unload(lb)), Fa[m] = b), g.onreadystatechange = b) : b()
                },
                abort: function() {
                    b && b(0, 1)
                }
            }
        }
    });
    var ob, fa, Ob = /^(?:toggle|show|hide)$/, fc = RegExp("^(?:([-+])\x3d|)(" + ra + ")([a-z%]*)$", "i"), mc = /queueHooks$/, pb = [function(a, b, e) {
        var c, f, h, p, g, n, r = this, s = a.style, y = {}, H = [], F = a.nodeType && O(a);
        e.queue || (g = d._queueHooks(a, "fx"), null == g.unqueued && (g.unqueued =
        0, n = g.empty.fire, g.empty.fire = function() {
            g.unqueued || n()
        }), g.unqueued++, r.always(function() {
            r.always(function() {
                g.unqueued--;
                d.queue(a, "fx").length || g.empty.fire()
            })
        }));
        1 === a.nodeType && ("height"in b || "width"in b) && (e.overflow = [s.overflow, s.overflowX, s.overflowY], "inline" === d.css(a, "display") && "none" === d.css(a, "float") && (d.support.inlineBlockNeedsLayout && "inline" !== G(a.nodeName) ? s.zoom = 1 : s.display = "inline-block"));
        e.overflow && (s.overflow = "hidden", d.support.shrinkWrapBlocks || r.done(function() {
            s.overflow =
            e.overflow[0];
            s.overflowX = e.overflow[1];
            s.overflowY = e.overflow[2]
        }));
        for (c in b)
            f = b[c], Ob.exec(f) && (delete b[c], f !== (F ? "hide" : "show") && H.push(c));
        if (f = H.length)
            for (h = d._data(a, "fxshow") || d._data(a, "fxshow", {}), F ? d(a).show() : r.done(function() {
                d(a).hide()
            }), r.done(function() {
                var b;
                d.removeData(a, "fxshow", !0);
                for (b in y)
                    d.style(a, b, y[b])
                }), c = 0; c < f; c++)
                    b = H[c], p = r.createTween(b, F ? h[b] : 0), y[b] = h[b] || d.style(a, b), b in h || (h[b] = p.start, F && (p.end = p.start, p.start = "width" === b || "height" === b ? 1 : 0))
    }
    ], Qa = {
        "*": [function(a,
        b) {
            var e, c, f, h = this.createTween(a, b), p = fc.exec(b), g = h.cur(), n =+ g || 0, r = 1;
            if (p) {
                e =+ p[2];
                c = p[3] || (d.cssNumber[a] ? "" : "px");
                if ("px" !== c && n) {
                    n = d.css(h.elem, a, !0) || e || 1;
                    do 
                        f = r = r || ".5", n/=r, d.style(h.elem, a, n + c), r = h.cur() / g;
                    while (1 !== r && r !== f)
                    }
                h.unit = c;
                h.start = n;
                h.end = p[1] ? n + (p[1] + 1) * e : e
            }
            return h
        }
        ]
    };
    d.Animation = d.extend(M, {
        tweener: function(a, b) {
            d.isFunction(a) ? (b = a, a = ["*"]): a = a.split(" ");
            for (var e,
            c = 0,
            f = a.length;
            c < f;
            c++)e = a[c],
            Qa[e] = Qa[e] || [],
            Qa[e].unshift(b)
        }, prefilter : function(a, b) {
            b ? pb.unshift(a) : pb.push(a)
        }
    });
    d.Tween = Q;
    Q.prototype = {
        constructor: Q,
        init: function(a, b, e, c, f, h) {
            this.elem = a;
            this.prop = e;
            this.easing = f || "swing";
            this.options = b;
            this.start = this.now = this.cur();
            this.end = c;
            this.unit = h || (d.cssNumber[e] ? "" : "px")
        },
        cur: function() {
            var a = Q.propHooks[this.prop];
            return a && a.get ? a.get(this) : Q.propHooks._default.get(this)
        },
        run: function(a) {
            var b, e = Q.propHooks[this.prop];
            this.pos = this.options.duration ? b = d.easing[this.easing](a, this.options.duration * a, 0, 1, this.options.duration) : b = a;
            this.now = (this.end - this.start) *
            b + this.start;
            this.options.step && this.options.step.call(this.elem, this.now, this);
            e && e.set ? e.set(this) : Q.propHooks._default.set(this);
            return this
        }
    };
    Q.prototype.init.prototype = Q.prototype;
    Q.propHooks = {
        _default: {
            get: function(a) {
                return null == a.elem[a.prop] || a.elem.style && null != a.elem.style[a.prop] ? (a = d.css(a.elem, a.prop, !1, "")) && "auto" !== a ? a : 0 : a.elem[a.prop]
            },
            set: function(a) {
                if (d.fx.step[a.prop])
                    d.fx.step[a.prop](a);
                else 
                    a.elem.style && (null != a.elem.style[d.cssProps[a.prop]] || d.cssHooks[a.prop]) ? d.style(a.elem,
                    a.prop, a.now + a.unit) : a.elem[a.prop] = a.now
            }
        }
    };
    Q.propHooks.scrollTop = Q.propHooks.scrollLeft = {
        set: function(a) {
            a.elem.nodeType && a.elem.parentNode && (a.elem[a.prop] = a.now)
        }
    };
    d.each(["toggle", "show", "hide"], function(a, b) {
        var e = d.fn[b];
        d.fn[b] = function(c, f, h) {
            return null == c || "boolean" === typeof c ||!a && d.isFunction(c) && d.isFunction(f) ? e.apply(this, arguments) : this.animate(T(b, !0), c, f, h)
        }
    });
    d.fn.extend({
        fadeTo: function(a, b, d, c) {
            return this.filter(O).css("opacity", 0).show().end().animate({
                opacity: b
            }, a, d, c)
        },
        animate: function(a,
        b, e, c) {
            var f = d.isEmptyObject(a), h = d.speed(b, e, c);
            b = function() {
                var b = M(this, d.extend({}, a), h);
                f && b.stop(!0)
            };
            return f ||!1 === h.queue ? this.each(b) : this.queue(h.queue, b)
        },
        stop: function(a, b, e) {
            var c = function(a) {
                var b = a.stop;
                delete a.stop;
                b(e)
            };
            "string" !== typeof a && (e = b, b = a, a = f);
            b&&!1 !== a && this.queue(a || "fx", []);
            return this.each(function() {
                var b=!0, h = null != a && a + "queueHooks", f = d.timers, p = d._data(this);
                if (h)
                    p[h] && p[h].stop && c(p[h]);
                else 
                    for (h in p)
                        p[h] && p[h].stop && mc.test(h) && c(p[h]);
                for (h = f.length; h--;)
                    f[h].elem !==
                    this || null != a && f[h].queue !== a || (f[h].anim.stop(e), b=!1, f.splice(h, 1));
                !b && e || d.dequeue(this, a)
            })
        }
    });
    d.each({
        slideDown: T("show"),
        slideUp: T("hide"),
        slideToggle: T("toggle"),
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
        d.fn[a] = function(a, d, c) {
            return this.animate(b, a, d, c)
        }
    });
    d.speed = function(a, b, e) {
        var c = a && "object" === typeof a ? d.extend({}, a): {
            complete: e ||!e && b || d.isFunction(a) && a,
            duration: a,
            easing: e && b || b&&!d.isFunction(b) && b
        };
        c.duration = d.fx.off ? 0 : "number" ===
        typeof c.duration ? c.duration : c.duration in d.fx.speeds ? d.fx.speeds[c.duration] : d.fx.speeds._default;
        if (null == c.queue ||!0 === c.queue)
            c.queue = "fx";
        c.old = c.complete;
        c.complete = function() {
            d.isFunction(c.old) && c.old.call(this);
            c.queue && d.dequeue(this, c.queue)
        };
        return c
    };
    d.easing = {
        linear: function(a) {
            return a
        },
        swing: function(a) {
            return 0.5 - Math.cos(a * Math.PI) / 2
        }
    };
    d.timers = [];
    d.fx = Q.prototype.init;
    d.fx.tick = function() {
        for (var a, b = d.timers, e = 0; e < b.length; e++)
            a = b[e], a() || b[e] !== a || b.splice(e--, 1);
        b.length || d.fx.stop()
    };
    d.fx.timer = function(a) {
        a() && d.timers.push(a)&&!fa && (fa = setInterval(d.fx.tick, d.fx.interval))
    };
    d.fx.interval = 13;
    d.fx.stop = function() {
        clearInterval(fa);
        fa = null
    };
    d.fx.speeds = {
        slow: 600,
        fast: 200,
        _default: 400
    };
    d.fx.step = {};
    d.expr && d.expr.filters && (d.expr.filters.animated = function(a) {
        return d.grep(d.timers, function(b) {
            return a === b.elem
        }).length
    });
    var Eb = /^(?:body|html)$/i;
    d.fn.offset = function(a) {
        if (arguments.length)
            return a === f ? this : this.each(function(b) {
                d.offset.setOffset(this, a, b)
            });
        var b, e, c, p, h, g;
        if (h =
        (b = this[0]) && b.ownerDocument) {
            if ((c = h.body) === b)
                return d.offset.bodyOffset(b);
            e = h.documentElement;
            if (!d.contains(e, b))
                return {
                    top: 0,
                    left: 0
                };
            b = b.getBoundingClientRect();
            p = ka(h);
            h = e.clientTop || c.clientTop || 0;
            c = e.clientLeft || c.clientLeft || 0;
            g = p.pageYOffset || e.scrollTop;
            e = p.pageXOffset || e.scrollLeft;
            h = b.top + g - h;
            b = b.left + e - c;
            return {
                top: h,
                left: b
            }
        }
    };
    d.offset = {
        bodyOffset: function(a) {
            var b = a.offsetTop, e = a.offsetLeft;
            d.support.doesNotIncludeMarginInBodyOffset && (b += parseFloat(d.css(a, "marginTop")) || 0, e += parseFloat(d.css(a,
            "marginLeft")) || 0);
            return {
                top: b,
                left: e
            }
        },
        setOffset: function(a, b, e) {
            var c = d.css(a, "position");
            "static" === c && (a.style.position = "relative");
            var f = d(a), h = f.offset(), p = d.css(a, "top"), g = d.css(a, "left"), n = {}, r = {};
            ("absolute" === c || "fixed" === c)&&-1 < d.inArray("auto", [p, g]) ? (r = f.position(), c = r.top, g = r.left) : (c = parseFloat(p) || 0, g = parseFloat(g) || 0);
            d.isFunction(b) && (b = b.call(a, e, h));
            null != b.top && (n.top = b.top - h.top + c);
            null != b.left && (n.left = b.left - h.left + g);
            "using"in b ? b.using.call(a, n) : f.css(n)
        }
    };
    d.fn.extend({
        position: function() {
            if (this[0]) {
                var a =
                this[0], b = this.offsetParent(), e = this.offset(), c = Eb.test(b[0].nodeName) ? {
                    top: 0,
                    left: 0
                }
                : b.offset();
                e.top -= parseFloat(d.css(a, "marginTop")) || 0;
                e.left -= parseFloat(d.css(a, "marginLeft")) || 0;
                c.top += parseFloat(d.css(b[0], "borderTopWidth")) || 0;
                c.left += parseFloat(d.css(b[0], "borderLeftWidth")) || 0;
                return {
                    top: e.top - c.top,
                    left: e.left - c.left
                }
            }
        },
        offsetParent: function() {
            return this.map(function() {
                for (var a = this.offsetParent || C.body; a&&!Eb.test(a.nodeName) && "static" === d.css(a, "position");)
                    a = a.offsetParent;
                return a ||
                C.body
            })
        }
    });
    d.each({
        scrollLeft: "pageXOffset",
        scrollTop: "pageYOffset"
    }, function(a, b) {
        var e = /Y/.test(b);
        d.fn[a] = function(c) {
            return d.access(this, function(a, c, p) {
                var g = ka(a);
                if (p === f)
                    return g ? b in g ? g[b] : g.document.documentElement[c] : a[c];
                g ? g.scrollTo(e ? d(g).scrollLeft() : p, e ? p : d(g).scrollTop()) : a[c] = p
            }, a, c, arguments.length, null)
        }
    });
    d.each({
        Height: "height",
        Width: "width"
    }, function(a, b) {
        d.each({
            padding: "inner" + a,
            content: b,
            "": "outer" + a
        }, function(e, c) {
            d.fn[c] = function(c, h) {
                var p = arguments.length && (e || "boolean" !==
                typeof c), g = e || (!0 === c ||!0 === h ? "margin" : "border");
                return d.access(this, function(b, e, c) {
                    return d.isWindow(b) ? b.document.documentElement["client" + a] : 9 === b.nodeType ? (e = b.documentElement, Math.max(b.body["scroll" + a], e["scroll" + a], b.body["offset" + a], e["offset" + a], e["client" + a])) : c === f ? d.css(b, e, c, g) : d.style(b, e, c, g)
                }, b, p ? c : f, p, null)
            }
        })
    });
    c.jimdoGen002 = c.$ = d;
    "function" === typeof define && define.amd && define.amd.jimdoGen002 && define("jquery", [], function() {
        return d
    })
})(window);
(function() {
    (function(c, f, g) {
        return "object" === typeof exports && "object" === typeof module ? module.exports = g() : "function" === typeof define && define.amd ? define(g) : c[f] = g()
    })("object" === typeof window ? window : this, "ReplayJS", function() {
        return new (function() {
            function c() {}
            c.prototype.num = 0;
            c.prototype.recordings = {};
            c.prototype.recorder = function() {
                var c;
                this.recordings[this.num] = [];
                c = function(c) {
                    return function() {
                        return c.recordings[arguments.callee.num].push(arguments)
                    }
                }(this);
                c.num = this.num++;
                c.replay = this;
                return c
            };
            c.prototype.play = function(c, g, k) {
                var n;
                null == k && (k = null);
                n = this.recordings[null != c ? c.num: void 0];
                if ("undefined" === typeof n)
                    throw {
                        message: "Given recorder unknown"
                    };
                if (c.replay !== this)
                    throw {
                        message: "Given recorder from different ReplayJS"
                    };
                if ("function" !== typeof g)
                    throw {
                        message: "Given target function is not a function"
                    };
                for (; c = n.shift();)
                    null != g && g.apply(k, c)
            };
            return c
        }())
    })
}).call(this);
(function(c, f) {
    c.jimdom = {
        event: f.recorder(),
        onLoad: f.recorder()
    }
})(jimdoGen002, window.ReplayJS);
this.Class = function(c, f, g) {
    function k(n) {
        function v() {
            for (var t = c.call(arguments, 0), k = null, v = 0, B = n.classes.length; v < B; ++v)
                k ? (f.prototype = k, 1 === v && t.unshift(g), t[0] = new f, k = new f) : k = {}, n.classes[v].apply(k, t);
            "function" === typeof k.init && k.init();
            return k
        }
        v.extend = function(c) {
            var f = k(c);
            c.classes.unshift.apply(c.classes, n.classes);
            return f
        };
        n.classes = [n];
        return v
    }
    return k
}(Array.prototype.slice, function() {});
(function(c, f) {
    function g() {
        function g(c, f) {
            B[c](f, function(c) {
                "error" == c ? x.push(f) : A.push(f) && E.each(f);
                t()
            }, "lazy-loaded-" + (E.name ? E.name : (new Date).getTime()), E[c + "Key"] ? "?key\x3d" + E[c + "Key"] : "")
        }
        function v(c) {
            E.complete(c, A, x);
            clearTimeout(I);
            clearTimeout(K);
            try {
                E[c]("error" === c ? x : A)
            } catch (g) {
                f.console && f.console.debug && f.console.debug(g.toString(), "stack:", g.stack)
            }
        }
        function t() {
            A.length == O.length ? v("success") : A.length + x.length === O.length && v("error")
        }
        function z() {
            x.push(this.src);
            t()
        }
        function w(f) {
            var g =
            [];
            c.each(f, function(c, f) {
                f && g.push(f)
            });
            return g
        }
        var B = this, E, A = [], x = [], I, K, D, O = [];
        this.init = function(f) {
            f && (E = c.extend({}, c.xLazyLoader.defaults, f), D = {
                js: w(E.js),
                css: E.css,
                img: E.img
            }, c.each(D, function(c, f) {
                "string" === typeof f && (f = f.split(","));
                O = O.concat(f)
            }), O.length ? (E.timeout && (I = setTimeout(function() {
                var f = A.concat(x);
                c.each(O, function(g, n) {
                    - 1 === c.inArray(n, f) && x.push(n)
                });
                v("error")
            }, E.timeout)), c.each(D, function(f, k) {
                c.isArray(k) ? c.each(k, function(c, k) {
                    g(f, k)
                }) : "string" === typeof k && g(f, k)
            })) :
            v("error"))
        };
        this.js = function(g, n, t, v) {
            var G = c('script[src*\x3d"' + g + '"]');
            if (G.length)
                G.attr("pending") ? G.bind("scriptload", n) : n();
            else {
                f && f.location.search && null !== f.location.search.match(/(\?|&)debug/) && (v = v&&-1 !== v.indexOf("?") ? v + "\x26debug" : "?debug");
                var w = document.createElement("script");
                w.setAttribute("type", "text/javascript");
                w.setAttribute("charset", "UTF-8");
                w.setAttribute("src", g + v);
                w.setAttribute("id", t);
                w.setAttribute("pending", 1);
                w.onerror = z;
                c(w).bind("scriptload", function() {
                    c(this).removeAttr("pending");
                    n();
                    setTimeout(function() {
                        c(w).unbind("scriptload")
                    }, 10)
                });
                var x=!1;
                w.onload = w.onreadystatechange = function() {
                    x || this.readyState&&!/loaded|complete/.test(this.readyState) || (x=!0, w.onload = w.onreadystatechange = null, c(w).trigger("scriptload"))
                };
                k.appendChild(w)
            }
        };
        this.css = function(f, g, n, t) {
            if (c('link[href*\x3d"' + f + '"]').length)
                "function" === typeof g && g();
            else {
                var v = c('\x3clink rel\x3d"stylesheet" type\x3d"text/css" media\x3d"all" href\x3d"' + f + t + '" id\x3d"' + n + '"\x3e\x3c/link\x3e')[0];
                c.browser.msie ? v.onreadystatechange =
                function() {
                    /loaded|complete/.test(v.readyState) && g()
                } : c.browser.opera ? v.onload = g : (location.hostname.replace("www.", ""), /http:/.test(f) && /^(\w+:)?\/\/([^\/?#]+)/.exec(f), g());
                k.appendChild(v)
            }
        };
        this.img = function(c, f, g, n) {
            g = new Image;
            g.onload = f;
            g.onerror = z;
            g.src = c + n
        };
        this.disable = function(f) {
            c("#lazy-loaded-" + f, k).attr("disabled", "disabled")
        };
        this.enable = function(f) {
            c("#lazy-loaded-" + f, k).removeAttr("disabled")
        };
        this.destroy = function(f) {
            c("#lazy-loaded-" + f, k).remove()
        }
    }
    c.xLazyLoader = function(c, f) {
        "object" ===
        typeof c && (f = c, c = "init");
        (new g)[c](f)
    };
    c.xLazyLoader.defaults = {
        js: [],
        css: [],
        img: [],
        jsKey: null,
        cssKey: null,
        imgKey: null,
        name: null,
        timeout: 2E4,
        success: c.noop,
        error: c.noop,
        complete: c.noop,
        each: c.noop
    };
    var k = document.getElementsByTagName("head")[0]
})(jimdoGen002, window);
(function(c) {
    function f(f, B) {
        function E() {
            M.on("load", function() {
                var d = V.hasClass("lb-cycling"), c = A();
                ma || M.removeClass("lb-stretch");
                Ra = M.width();
                ya = M.height();
                Ja = Ra / ya;
                ka.html(qa.attr(B.titleProp) || "");
                C.text(R.index(qa) + 1);
                x();
                M.addClass("lb-stretch");
                L.add(T).animate({
                    opacity: 1
                }, v);
                I();
                d && (d = A(), L.width(c.container.width).height(c.container.height).css("margin-top", c.container.marginTop).css("margin-left", c.container.marginLeft).animate(d.container, v / 3), T.width(c.controls.width).height(c.controls.height).css("margin-bottom",
                c.controls.marginBottom).css("margin-left", c.controls.marginLeft).animate(d.controls, v / 3).promise().done(function() {
                    Ia.animate({
                        opacity: 1
                    }, v / 3).promise().done(function() {
                        T.height("auto");
                        V.removeClass("lb-cycling")
                    })
                }))
            }).click(function() {
                ra && (ma=!ma, V.toggleClass("lb-zoomed"), L.scrollTop(0), x())
            }).on("contextmenu", function() {
                return !1
            });
            Za.add(Ha).click(G.destroy);
            Aa.click(function() {
                O(!aa)
            });
            ga.click(function() {
                K( - 1)
            });
            Q.click(function() {
                K(1)
            });
            ja.on("keyup.lightbox", function(d) {
                switch (d.which) {
                case 37:
                    K( - 1);
                    break;
                case 39:
                    K(1);
                    break;
                case 27:
                    G.destroy()
                }
            });
            ja.on("resize.lightbox", function() {
                x();
                I()
            })
        }
        function A() {
            return {
                container: {
                    width: L.width(),
                    height: L.height(),
                    marginTop: L.css("margin-top"),
                    marginLeft: L.css("margin-left")
                },
                controls: {
                    width: T.width(),
                    height: T.height(),
                    marginBottom: T.css("margin-bottom"),
                    marginLeft: T.css("margin-left")
                }
            }
        }
        function x() {
            la = ja.width() - n;
            va = ja.height() - n;
            X = Math.min(la, Ra);
            Y = Math.min(va, ya);
            ma || (Y = Math.round(X / Ja));
            X = Math.max(k, X);
            L.add(T).width(X);
            $ = T.height();
            0 < Y + $ - va && (Y =
            va - $);
            L.height(Y);
            if (ma)
                Ra > X && (Y = Math.min(va - $, Y + L[0].offsetHeight - L[0].clientHeight), L.height(Y)), ya > Y && (X = Math.min(la, X + L[0].offsetWidth - L[0].clientWidth), L.add(T).width(X));
            else 
                for (var d = 0; X / Y !== Ja && 5 > d++;) {
                    X = Math.max(k, Math.floor(Y * Ja));
                    L.add(T).width(X);
                    if (ya < va)
                        break;
                        $ = T.height();
                        Y = va - $;
                        L.height(Y)
                }
            L.css("margin", (0 - $ - Y) / 2 + "px 0 0 " +- X / 2 + "px");
            T.css("margin", "0 0 " + (va - $ - Y) / 2 + "px " +- X / 2 + "px")
        }
        function I() {
            ra = Ra > L.width() || ya > L.height();
            V.toggleClass("lb-zoomable", ra);
            ra || (ma=!1, V.removeClass("lb-zoomed"))
        }
        function K(c, f) {
            d&&!V.hasClass("lb-cycling") && (f || O(!1), qa = W(c), V.addClass("lb-cycling"), Ia.animate({
                opacity: 0
            }, v / 3).promise().done(function() {
                M.prop("src", qa.data("href"));
                Z()
            }), N())
        }
        function D() {
            K(1, !0)
        }
        function O(d) {
            V.toggleClass("lb-slideshow-active", d);
            d ? aa = window.setInterval(D, t) : aa && (clearInterval(aa), aa = null)
        }
        function W(d) {
            d = R.index(qa) + d;
            - 1 === d && (d = R.length - 1);
            d === R.length && (d = 0);
            return R.eq(d)
        }
        function N() {
            d = 1 < ba;
            V.toggleClass("lb-cyclable", d);
            d && (P(W( - 1).data("href")), P(W(1).data("href")))
        }
        function P(d) {
            c("\x3cimg /\x3e").attr("src", d).hide().appendTo(V).remove()
        }
        function Z() {
            B.pinterest && M.pinterest({
                wrapper: L
            }).pinterest("enable")
        }
        var G = this, ja = c(window), U = c(f), R = c(z, U), qa, V, Za, L, M, ga, Q, T, ka, Ha, Aa, C, Ia, Ra, ya, X, Y, $, la, va, Ja, ba = R.length, d=!1, ra=!1, ma=!1, aa = null;
        this.init = function() {
            B = c.extend({}, c.fn.lightbox.defaults, B);
            U.click(function(d) {
                qa = c(d.target).closest(z);
                if (!qa.length)
                    return !0;
                V = c(g).appendTo("body");
                Za = c("#lb-overlay", V);
                L = c("#lb-container", V);
                M = c("img", L).prop("src", qa.data("href"));
                ga = c("#lb-prev", L);
                Q = c("#lb-next", L);
                T = c("#lb-controls", V);
                ka = c("#lb-title", T);
                Ha = c("#lb-close", T);
                Aa = c("#lb-play", T);
                C = c("#lb-current", T);
                c("#lb-count", T).text(ba);
                Ia = c(".lb-gutter", T).add(M);
                Z();
                N();
                E();
                return !1
            })
        };
        this.destroy = function() {
            O(!1);
            V.remove();
            ja.off("resize.lightbox keyup.lightbox");
            ma=!1
        }
    }
    c.fn.lightbox = function(g) {
        return this.each(function() {
            var n = c.data(this, "lightbox");
            n && n.destroy();
            c.data(this, "lightbox", new f(this, g)).init()
        })
    };
    c.fn.lightbox.defaults = {
        pinterest: !1,
        titleProp: "title"
    };
    var g = '        \x3cdiv id\x3d"lb-wrapper"\x3e            \x3cdiv id\x3d"lb-overlay"\x3e\x3c/div\x3e            \x3cdiv class\x3d"lb-loading"\x3e\x3c/div\x3e            \x3cdiv id\x3d"lb-container"\x3e                \x3cdiv class\x3d"lb-loading"\x3e\x3c/div\x3e                \x3cdiv class\x3d"cc-pinterest-overlay"\x3e                    \x3ca class\x3d"cc-pinterest-link" href\x3d"javascript:" title\x3d"Pin It"\x3e\x3c/a\x3e                \x3c/div\x3e                \x3cimg /\x3e                \x3cdiv id\x3d"lb-next" class\x3d"lb-cycler"\x3e                    \x3cdiv\x3e\x3c/div\x3e                \x3c/div\x3e                \x3cdiv id\x3d"lb-prev" class\x3d"lb-cycler"\x3e                    \x3cdiv\x3e\x3c/div\x3e                \x3c/div\x3e            \x3c/div\x3e            \x3cdiv id\x3d"lb-controls"\x3e                \x3cdiv class\x3d"lb-gutter"\x3e                    \x3cdiv id\x3d"lb-play" class\x3d"lb-button"\x3e\x3c/div\x3e                    \x3cdiv id\x3d"lb-close" class\x3d"lb-button"\x3e\x3c/div\x3e                    \x3cdiv id\x3d"lb-title"\x3e\x3c/div\x3e                    \x3cdiv id\x3d"lb-footer"\x3e                        \x3cspan id\x3d"lb-current"\x3e\x3c/span\x3e/\x3cspan id\x3d"lb-count"\x3e\x3c/span\x3e                    \x3c/div\x3e                \x3c/div\x3e            \x3c/div\x3e        \x3c/div\x3e    ',
    k = 250, n = 60, v = 300, t = 5E3, z = '[rel*\x3d"lightbox"]'
})(jimdoGen002);
(function(c) {
    var f = [], g = 0, k = {}, n = {}, v=!1, t=!1;
    c.sync = function(z, w) {
        function B() {
            clearTimeout(k[I.url]);
            "high" === x.prio ? (f.splice(g, 0, E), g++) : f.push(E);
            if (!t&&!v)
                f[0]()
        }
        function E() {
            v=!0;
            I.complete = A;
            c.isFunction(I.data) && (I.data = I.data());
            c.ajax(I);
            f.shift();
            g = "high" === x.prio ? g - 1 : g
        }
        function A() {
            "function" === typeof K && K.apply(this, arguments);
            v=!1;
            delete n[I.url];
            if (!t && f.length)
                f[0]()
        }
        var x = c.extend({}, c.sync.defaults, w), I = c.extend({}, x.ajaxOptions, z), K = I.complete;
        x.oneAtATime && n[I.url] || (n[I.url] =
        !0, x.queue && (x.delay && 0 < x.delay ? (clearTimeout(k[I.url]), k[I.url] = setTimeout(B, x.delay)) : B()))
    };
    c.extend(c.sync, {
        pause: function() {
            t=!0
        },
        run: function() {
            t=!1;
            if (!v && f && "function" === typeof f[0])
                f[0]()
        },
        defaults: {
            oneAtATime: !1,
            queue: !0,
            delay: 300,
            prio: "high",
            ajaxOptions: {
                dataType: "json"
            }
        }
    })
})(jimdoGen002);
(function(c, f) {
    function g(f, g, v) {
        this.enabled=!0;
        this.$container = f;
        this.controller = g;
        g = this.settings = c.extend({}, c.fn.actionController.defaults, v);
        g.history && c.fn.historyManager && f.historyManager();
        f.on(g.events, "[" + g.actionAttr + "]", c.proxy(this, "handler"))
    }
    c.fn.actionController = function(f, n) {
        return this.each(function() {
            var v = c.data(this, "actionController") || c.data(this, "actionController", new g(c(this), f, n));
            if ("string" === typeof f)
                v[f](n)
        })
    };
    c.fn.actionController.defaults = {
        actionPrefix: "_",
        actionAttr: "data-action",
        paramsAttr: "data-params",
        historyAttr: "data-history",
        defaultAction: "action",
        editEventAttr: "editAction",
        editEvents: "click",
        events: "click",
        history: !1,
        handleEvent: null,
        context: "element"
    };
    g.prototype = {
        destroy: function() {
            c(this.$container).off(this.events, "[" + this.settings.actionAttr + "]", this.handler);
            this.$container.removeData("actionController")
        },
        disable: function() {
            this.enabled=!1
        },
        enable: function() {
            this.enabled=!0
        },
        handler: function(g) {
            if (this.enabled) {
                var n = this.settings, v = c(g.currentTarget), t = arguments,
                z = v.attr(n.paramsAttr), w = g.type.substr(0, 1).toUpperCase() + g.type.substr(1), B = n.actionPrefix + c.trim(v.attr(n.actionAttr)) + w, E = this.controller[B], w = this.controller[n.actionPrefix + n.defaultAction + w], A = "controller" === n.context ? this.controller: v[0];
                z && (z = c.map(z.split(","), function(f) {
                    return c.trim(f)
                }), t = f.call(t, 0).concat(z));
                if (w && c.isFunction(w)&&!1 === w.apply(A, t))
                    return !1;
                if (E && c.isFunction(E)) {
                    if (c.isFunction(n.handleEvent))
                        try {
                            n.handleEvent.call(this, g)
                        } catch (x) {}
                    return n.history && (n = c.trim(v.attr(n.historyAttr)),
                    !g.historyHandled && n) ? (v.historyManager(n, function() {
                        E.apply(A, t)
                    }), !1) : E.apply(A, t)
                }
            }
        }
    }
})(jimdoGen002, Array.prototype.slice);
(function(c) {
    function f(f, k, n) {
        function v(f) {
            if ("confirm" == t.status&&!c(f.target).hasClass("x-message-action"))
                return !1;
            clearTimeout(D);
            var g = f && f.target && /ok/.test(f.target.className) ? "ok": "abort";
            if (!1 === t.onClose.apply(w[0], [f, g, z]))
                return !1;
            z.destroy();
            return !1
        }
        "object" === typeof k && (n = k, k = null);
        var t = c.extend(!0, {}, c.fn.message.defaults, n), z = this, w, B = c(f), E, A, x, I, K = "", D;
        this.init = function() {
            k = k || t.message;
            if (!k)
                return this;
            c.event.trigger("messageshow");
            B.data("place-error") && (t.insertMethod =
            B.data("place-error"));
            if ("FORM" == B[0].nodeName) {
                var f = c('[type\x3d"submit"]', B);
                B = f.length ? f : B
            }
            "confirm" == t.status && (K += t.buttonsTpl.replace("${ok}", t.labels.ok).replace("${abort}", t.labels.abort), t.autoHide = null);
            w = c(t.template.replace("${message}", k || t.message).replace("${buttons}", K)).addClass("x-message message-" + t.status + " " + t.addClass)[t.insertMethod](B);
            t.closable && w.addClass("x-message-closable");
            I = B.position();
            E = w.height();
            A = w.outerWidth();
            E > t.maxHeight && (E = t.maxHeight, w.css("height", E).find(".x-message-content").css("display",
            "block"));
            x = t.left || I.left;
            f = c(window).width();
            x + A > f && (x -= x + A - f);
            f = t.top || I.top - 2 * E;
            t.resetOffset && 0 > f + E && (f =- 1 * E);
            w.css({
                display: "none",
                visibility: "visible",
                top: f,
                left: x,
                width: t.width
            }).animate({
                top: "+\x3d" + E,
                opacity: "show"
            }, t.speed, function() {
                t.autoHide && (D = setTimeout(v, t.autoHide))
            }).click(function(f) {
                c(f.target).is(".x-message-close") || t.onClick.call(w[0], f);
                v(f)
            })
        };
        this.destroy = function() {
            w.remove();
            c.removeData(f, "message");
            c.event.trigger("messageclose")
        }
    }
    c.fn.message = function(g, k) {
        return this.each(function() {
            c.data(this,
            "message") || c.data(this, "message", new f(this, g, k)).init()
        })
    };
    c.fn.message.defaults = {
        message: null,
        closable: !0,
        status: "success",
        autoHide: 5E3,
        addClass: "",
        template: '        \x3cdiv\x3e            \x3cspan class\x3d"x-message-close x-message-action"\x3ex\x3c/span\x3e            \x3cp class\x3d"x-message-content"\x3e${message}\x3c/p\x3e            ${buttons}        \x3c/div\x3e',
        buttonsTpl: '        \x3cbr /\x3e\x3cbutton type\x3d"button" class\x3d"x-message-button-ok x-message-action" selected\x3d"selected"\x3e${ok}\x3c/button\x3e        \x3cbutton type\x3d"button" class\x3d"x-message-button-abort x-message-action"\x3e${abort}\x3c/button\x3e    ',
        speed: 300,
        width: "auto",
        maxHeight: 150,
        top: null,
        left: null,
        onClose: c.noop,
        onClick: c.noop,
        labels: {
            ok: "Ok",
            abort: "Abort"
        },
        insertMethod: "insertBefore",
        resetOffset: !1
    }
})(jimdoGen002);
(function(c) {
    var f = {};
    c.defer = function(g, k, n) {
        c.isFunction(g) && (n = k, k = g, g = k + "");
        f[g] && clearTimeout(f[g]);
        f[g] = setTimeout(k, n);
        return function() {
            clearTimeout(f[g]);
            delete f[g]
        }
    }
})(jimdoGen002);
(function(c) {
    (function(c, g) {
        if ("object" === typeof exports && exports)
            g(exports);
        else {
            var k = {};
            g(k);
            "function" === typeof define && define.amd ? define(k) : c.Mustache = k
        }
    })(this, function(c) {
        function g(c) {
            return c.replace(/[\-\[\]{}()*+?.,\\\^$|#\s]/g, "\\$\x26")
        }
        function k(c) {
            this.tail = this.string = c;
            this.pos = 0
        }
        function n(c, f) {
            this.view = c || {};
            this.parent = f;
            this._cache = {}
        }
        function v() {
            this.clearCache()
        }
        function t(g, n, k, v) {
            for (var w = "", z, x, B = 0, A = g.length; B < A; ++B)
                switch (z = g[B], x = z[1], z[0]) {
                case "#":
                    x = k.lookup(x);
                    if ("object" === typeof x)
                        if (O(x))
                            for (var E = 0, K = x.length; E < K; ++E)
                                w += t(z[4], n, k.push(x[E]), v);
                        else 
                            x && (w += t(z[4], n, k.push(x), v));
                else 
                    "function" === typeof x ? (z = null == v ? null : v.slice(z[3], z[5]), x = x.call(k.view, z, function(c) {
                        return n.render(c, k)
                    }), null != x && (w += x)) : x && (w += t(z[4], n, k, v));
                    break;
                case "^":
                    x = k.lookup(x);
                    if (!x || O(x) && 0 === x.length)
                        w += t(z[4], n, k, v);
                        break;
                    case "\x3e":
                        x = n.getPartial(x);
                        "function" === typeof x && (w += x(k));
                        break;
                    case "\x26":
                        x = k.lookup(x);
                        null != x && (w += x);
                        break;
                    case "name":
                        x = k.lookup(x);
                        null !=
                        x && (w += c.escape(x));
                        break;
                    case "text":
                        w += x
                }
            return w
        }
        function z(c) {
            return [RegExp(g(c[0]) + "\\s*"), RegExp("\\s*" + g(c[1]))]
        }
        var w = /\s*/, B = /\s+/, E = /\S/, A = /\s*=/, x = /\s*\}/, I = /#|\^|\/|>|\{|&|=|!/, K = RegExp.prototype.test, D = Object.prototype.toString, O = Array.isArray || function(c) {
            return "[object Array]" === D.call(c)
        }, W = {
            "\x26": "\x26amp;",
            "\x3c": "\x26lt;",
            "\x3e": "\x26gt;",
            '"': "\x26quot;",
            "'": "\x26#39;",
            "/": "\x26#x2F;"
        };
        k.prototype.eos = function() {
            return "" === this.tail
        };
        k.prototype.scan = function(c) {
            return (c = this.tail.match(c)) &&
            0 === c.index ? (this.tail = this.tail.substring(c[0].length), this.pos += c[0].length, c[0]) : ""
        };
        k.prototype.scanUntil = function(c) {
            var f = this.tail.search(c);
            switch (f) {
            case - 1:
                c = this.tail;
                this.pos += this.tail.length;
                this.tail = "";
                break;
            case 0:
                c = "";
                break;
            default:
                c = this.tail.substring(0, f), this.tail = this.tail.substring(f), this.pos += f
            }
            return c
        };
        n.make = function(c) {
            return c instanceof n ? c : new n(c)
        };
        n.prototype.push = function(c) {
            return new n(c, this)
        };
        n.prototype.lookup = function(c) {
            var f = this._cache[c];
            if (!f) {
                if ("." ==
                c)
                    f = this.view;
                else 
                    for (var g = this; g;) {
                        if (0 < c.indexOf("."))
                            for (var f = g.view, n = c.split("."), k = 0; f && k < n.length;)
                                f = f[n[k++]];
                        else 
                            f = g.view[c];
                            if (null != f)
                                break;
                                g = g.parent
                    }
                this._cache[c] = f
            }
            "function" === typeof f && (f = f.call(this.view));
            return f
        };
        v.prototype.clearCache = function() {
            this._cache = {};
            this._partialCache = {}
        };
        v.prototype.compile = function(g, n) {
            var k = this._cache[g];
            k || (k = c.parse(g, n), k = this._cache[g] = this.compileTokens(k, g));
            return k
        };
        v.prototype.compilePartial = function(c, f, g) {
            f = this.compile(f, g);
            return this._partialCache[c] =
            f
        };
        v.prototype.getPartial = function(c) {
            c in this._partialCache ||!this._loadPartial || this.compilePartial(c, this._loadPartial(c));
            return this._partialCache[c]
        };
        v.prototype.compileTokens = function(c, f) {
            var g = this;
            return function(k, v) {
                if (v)
                    if ("function" === typeof v)
                        g._loadPartial = v;
                    else 
                        for (var w in v)
                            g.compilePartial(w, v[w]);
                return t(c, g, n.make(k), f)
            }
        };
        v.prototype.render = function(c, f, g) {
            return this.compile(c)(f, g)
        };
        c.name = "mustache.js";
        c.version = "0.7.2";
        c.tags = ["{{", "}}"];
        c.Scanner = k;
        c.Context = n;
        c.Writer =
        v;
        c.parse = function(n, v) {
            n = n || "";
            v = v || c.tags;
            "string" === typeof v && (v = v.split(B));
            if (2 !== v.length)
                throw Error("Invalid tags: " + v.join(", "));
            for (var t = z(v), D = new k(n), N = [], R = [], O = [], V=!1, W=!1, L, M, ga, Q; !D.eos();) {
                L = D.pos;
                if (ga = D.scanUntil(t[0])) {
                    Q = 0;
                    for (var T = ga.length; Q < T; ++Q)
                        if (M = ga.charAt(Q), K.call(E, M) ? W=!0 : O.push(R.length), R.push(["text", M, L, L + 1]), L += 1, "\n" == M) {
                            if (V&&!W)
                                for (; O.length;)
                                    delete R[O.pop()];
                            else 
                                O = [];
                                W = V=!1
                        }
                }
                if (!D.scan(t[0]))
                    break;
                V=!0;
                M = D.scan(I) || "name";
                D.scan(w);
                "\x3d" === M ? (ga = D.scanUntil(A),
                D.scan(A), D.scanUntil(t[1])) : "{" === M ? (ga = D.scanUntil(RegExp("\\s*" + g("}" + v[1]))), D.scan(x), D.scanUntil(t[1]), M = "\x26") : ga = D.scanUntil(t[1]);
                if (!D.scan(t[1]))
                    throw Error("Unclosed tag at " + D.pos);
                Q = [M, ga, L, D.pos];
                R.push(Q);
                if ("#" === M || "^" === M)
                    N.push(Q);
                else if ("/" === M) {
                    if (0 === N.length)
                        throw Error('Unopened section "' + ga + '" at ' + L);
                    M = N.pop();
                    if (M[1] !== ga)
                        throw Error('Unclosed section "' + M[1] + '" at ' + L);
                } else if ("name" === M || "{" === M || "\x26" === M)
                    W=!0;
                else if ("\x3d" === M) {
                    v = ga.split(B);
                    if (2 !== v.length)
                        throw Error("Invalid tags at " +
                        L + ": " + v.join(", "));
                    t = z(v)
                }
            }
            if (M = N.pop())
                throw Error('Unclosed section "' + M[1] + '" at ' + D.pos);
            for (var t = R, D = [], ka, R = 0, O = t.length; R < O; ++R)
                if (N = t[R])
                    "text" === N[0] && ka && "text" === ka[0] ? (ka[1] += N[1], ka[3] = N[3]) : (ka = N, D.push(N));
            ka = D;
            D = t = [];
            N = [];
            O = 0;
            for (V = ka.length; O < V; ++O)
                switch (R = ka[O], R[0]) {
                case "#":
                case "^":
                    N.push(R);
                    D.push(R);
                    D = R[4] = [];
                    break;
                case "/":
                    N.pop()[5] = R[2];
                    D = 0 < N.length ? N[N.length - 1][4] : t;
                    break;
                default:
                    D.push(R)
                }
            return t
        };
        c.escape = function(c) {
            return String(c).replace(/[&<>"'\/]/g, function(c) {
                return W[c]
            })
        };
        var N = new v;
        c.clearCache = function() {
            return N.clearCache()
        };
        c.compile = function(c, f) {
            return N.compile(c, f)
        };
        c.compilePartial = function(c, f, g) {
            return N.compilePartial(c, f, g)
        };
        c.compileTokens = function(c, f) {
            return N.compileTokens(c, f)
        };
        c.render = function(c, f, g) {
            return N.render(c, f, g)
        };
        c.to_html = function(g, n, v, k) {
            g = c.render(g, n, v);
            if ("function" === typeof k)
                k(g);
            else 
                return g
        }
    });
    c.mustache = function(c, g, k) {
        return Mustache.render(c, g, k)
    };
    c.fn.mustache = function(f, g) {
        return c(this).map(function(k, n) {
            var v = c.trim(c(n).html()),
            v = c.mustache(v, f, g);
            return c(v).get()
        })
    }
})(jimdoGen002);
(function(c) {
    "function" === typeof define && define.amd ? define(["jquery"], c) : "object" === typeof exports ? c(require("jquery")) : c(jimdoGen002)
})(function(c) {
    function f(c) {
        return n.raw ? c : encodeURIComponent(c)
    }
    function g(f, g) {
        var z;
        if (n.raw)
            z = f;
        else 
            a: {
            var w = f;
            0 === w.indexOf('"') && (w = w.slice(1, - 1).replace(/\\"/g, '"').replace(/\\\\/g, "\\"));
            try {
                w = decodeURIComponent(w.replace(k, " "));
                z = n.json ? JSON.parse(w) : w;
                break a
            } catch (B) {}
            z = void 0
        }
        return c.isFunction(g) ? g(z) : z
    }
    var k = /\+/g, n = c.cookie = function(k, t, z) {
        if (void 0 !==
        t&&!c.isFunction(t)
            ) {
            z = c.extend({}, n.defaults, z);
            if ("number" === typeof z.expires) {
                var w = z.expires, B = z.expires = new Date;
                B.setTime( + B + 864E5 * w)
            }
            return document.cookie = [f(k), "\x3d", f(n.json ? JSON.stringify(t) : String(t)), z.expires ? "; expires\x3d" + z.expires.toUTCString(): "", z.path ? "; path\x3d" + z.path: "", z.domain ? "; domain\x3d" + z.domain: "", z.secure ? "; secure": ""].join("")
        }
        z = k ? void 0 : {};
        for (var w = document.cookie ? document.cookie.split("; ") : [], B = 0, E = w.length; B < E; B++) {
            var A = w[B].split("\x3d"), x;
            x = A.shift();
            x =
            n.raw ? x : decodeURIComponent(x);
            A = A.join("\x3d");
            if (k && k === x) {
                z = g(A, t);
                break
            }
            k || void 0 === (A = g(A)) || (z[x] = A)
        }
        return z
    };
    n.defaults = {};
    c.removeCookie = function(f, g) {
        if (void 0 === c.cookie(f)
            )return !1;
        c.cookie(f, "", c.extend({}, g, {
            expires: - 1
        }));
        return !c.cookie(f)
    }
});
(function(c, f) {
    var g = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw\x3d\x3d";
    c.fn.imagesLoaded = function(k) {
        function n() {
            var f = c(x), g = c(I);
            w && (I.length ? w.reject(E, f, g) : w.resolve(E));
            c.isFunction(k) && k.call(z, E, f, g)
        }
        function v(c) {
            t(c.target, "error" === c.type)
        }
        function t(f, k) {
            f.src !== g&&-1 === c.inArray(f, A) && (A.push(f), k ? I.push(f) : x.push(f), c.data(f, "imagesLoaded", {
                isBroken: k,
                src: f.src
            }), B && w.notifyWith(c(f), [k, E, c(x), c(I)]), E.length === A.length && (setTimeout(n), E.unbind(".imagesLoaded",
            v)))
        }
        var z = this, w = c.isFunction(c.Deferred) ? c.Deferred(): 0, B = c.isFunction(w.notify), E = z.find("img").add(z.filter("img")), A = [], x = [], I = [];
        c.isPlainObject(k) && c.each(k, function(c, f) {
            if ("callback" === c)
                k = f;
            else if (w)
                w[c](f)
        });
        E.length ? E.bind("load.imagesLoaded error.imagesLoaded", v).each(function(n, k) {
            var v = k.src, w = c.data(k, "imagesLoaded");
            if (w && w.src === v)
                t(k, w.isBroken);
            else if (k.complete && k.naturalWidth !== f)
                t(k, 0 === k.naturalWidth || 0 === k.naturalHeight);
            else if (k.readyState || k.complete)
                k.src = g, k.src = v
        }) :
        n();
        return w ? w.promise(z) : z
    }
})(jimdoGen002);
(function(c, f, g) {
    c.noConflict();
    f.jimdoData && (c.jimdoData = f.jimdoData);
    f.Class && (c.Class = f.Class)
})(jimdoGen002, this);
(function(c) {
    function f(f, g) {
        return function(k, z) {
            c.isArray(z) ? z.unshift(f) : z = [f];
            var w = g.getClass(k);
            return w ? w.apply({}, z) : c.error("Class " + k + " doesn't exist.")
        }
    }
    if (c.frameManager && c.frameManager.inSiteadmin())
        c.frameManager.getCmsJQuery().then(function(g) {
            c.reg = g.reg;
            c.factory = f(c, g.reg)
        });
    else {
        var g = {}, k = {};
        c.reg = function(f, k) {
            if ("string" === typeof f || c.isFunction(k)) {
                f = f.match(/^\s*([\w\.-]+)(\s+extends\s+([\w\.-]+))?\s*$/);
                var t = f[3];
                if (t && "undefined" === typeof g[t])
                    throw Error("Could not find class named '" +
                    t + "'");
                g[f[1]] = t ? g[t].extend(k) : c.Class(k)
            } else 
                c.error("Couldn't create class. \n " + JSON.stringify(arguments));
            return k
        };
        c.reg.get = function(f, g) {
            var t = k[f];
            t || "function" !== typeof g || (t = c.reg.set(f, g()));
            return t
        };
        c.reg.set = function(c, f) {
            return k[c] = f
        };
        c.reg.del = function(c) {
            return "undefined" !== typeof k[c] ? (k[c] = null, delete k[c], !0) : !1
        };
        c.factory = f(c, c.reg);
        c.reg.singleton = function(f, g) {
            return c.reg.get("singleton|" + f, function() {
                return c.factory(f, g)
            })
        };
        c.reg.getClass = function(c) {
            return g[c]
        };
        c.reg.delClass =
        function(c) {
            return delete g[c]
        }
    }
})(jimdoGen002);
(function(c, f) {
    function g(c) {
        t || n.triggerHandler("exception.report", {
            exDescription: c[0],
            appName: k.isCMS ? "cms": "web",
            appVersion: c[1]
        })
    }
    var k = c.jimdoData, n = c(f);
    f._jmdlg = c.noop;
    c.ajaxSetup({
        timeout: 6E4,
        error: function(k) {
            k.channel && f._jmdlg(k.responseText, k.channel.name);
            g([c.trim(this.url), k.status])
        }
    });
    (function() {
        var f = c.ajax, g = c.param({
            cstok: k.cstok,
            websiteid: k.websiteId,
            pageid: k.pageId
        });
        c.ajax = function(k) {
            "GET" !== (k.type || c.ajaxSettings.type).toUpperCase() && ("object" === typeof k.data && (k.data = c.param(k.data)),
            k.data = k.data ? k.data + "\x26" : "", k.data += g);
            return f(k)
        }
    })();
    var v = /debug/.test(location.search);
    c.extend(c.xLazyLoader.defaults, {
        error: g,
        jsKey: k.cacheJsKey + (v ? "\x26debug" : ""),
        cssKey: k.cacheCssKey + (v ? "\x26debug" : "")
    });
    c.extend(c.fn.message.defaults, {
        labels: {
            ok: k.tr.common.ok,
            abort: k.tr.common.abort
        }
    });
    var t;
    c(f).unload(function() {
        t=!0
    });
    (function() {
        k.isCucumber && k.isCMS && (c(function() {
            f.loaded_for_cucumber=!0
        }), f._jmdlg = function(c, g) {
            netscape.security.PrivilegeManager.enablePrivilege("UniversalBrowserAccess UniversalXPConnect");
            var k = Components.classes["@mozilla.org/file/local;1"].createInstance(Components.interfaces.nsILocalFile);
            k.initWithPath(f._jmdlgPath);
            k.exists() || k.createUnique(Components.interfaces.nsIFile.NORMAL_FILE_TYPE, - 1);
            var n = Components.classes["@mozilla.org/network/file-output-stream;1"].createInstance(Components.interfaces.nsIFileOutputStream);
            n.init(k, 16, - 1, 0);
            g = g || "";
            var k = "[" + (new Date).toLocaleString() + "] " + g + "\n" + c + "\n", t = Components.classes["@mozilla.org/intl/converter-output-stream;1"].createInstance(Components.interfaces.nsIConverterOutputStream);
            t.init(n, "UTF-8", 0, 0);
            t.writeString(k);
            t.close()
        })
    })();
    c(function() {
        c.browser.msie && c(document.body).addClass("ie ie-" + parseInt(c.browser.version, 10))
    });
    k.isMobile = function() {
        return k.mobile ||!1
    };
    k.isDebug = v
})(jimdoGen002, window);
(function(c, f) {
    c.modules = {};
    var g = {}, k = /debug/.test(location.search)&&!c.jimdoData.isTestserver ? c.jimdoData.cdnUrl + "s/": c.jimdoData.minUrl, n = k + "js/web/", v = k + "js/common/", t = k + "css/web/", z = n + "classes/", w = n + "modules/", B = n + "shop/", E = k + "js/libs/jquery/", A = k + "js/libs/jquery-ui/", x = k + "js/plugins/", I = k + "js/external/plugins/", K = c(f), D = k + "bower_components/", O = D + "jquery.scrollTo/jquery.scrollTo.js";
    g.header = {
        exfont: {
            js: [x + "exfont/jquery.exfont.js", x + "flash/jquery.flash.js", w + "header.js"]
        }
    };
    g.login = {
        "default": {
            js: [n +
            "login/login.js"]
        }
    };
    g.scrolltotop = {
        "default": {
            js: [A + "ui/jquery.ui.effect.js", v + "scrolltotop.js"]
        }
    };
    g.awesomebackground = {
        "default": {
            js: [x + "cycle/jquery.cycle.js", D + "imagesloaded/jquery.imagesloaded.js", x + "fullscreensize/jquery.fullscreensize.js", k + "js/external/scripts/froogaloop.js", v + "awesomebackground.js"]
        }
    };
    g.emotionHeader = {
        exfont: {
            js: [x + "exfont/jquery.exfont.js", x + "flash/jquery.flash.js", w + "emotionHeader.js"]
        }
    };
    g.gallery = {
        "default": {
            js: [x + "pinterest/jquery.pinterest.js", w + "gallery.js"],
            css: [x +
            "pinterest/jquery.pinterest.css"]
        },
        flash: {
            js: [x + "flash/jquery.flash.js", w + "gallery.js"]
        }
    };
    g.flickr = g.gallery;
    g.gallery.flickr = g.gallery["default"];
    g.video = {
        "default": {
            js: [x + "flash/jquery.flash.js", w + "video.js"],
            css: [t + "module/functional/oldmoduleinterface/video.css"]
        }
    };
    g.video.youtube = g.video.myvideo = g.video["default"];
    g.downloadDocument = {
        "default": {
            js: [w + "downloadDocument.js"]
        }
    };
    g.guestbook = {
        "default": {
            js: [x + "form/jquery.ajaxForm.js", O, E + "helpers/jquery.jimdoForm.js", w + "guestbook.js"]
        }
    };
    g.flash =
    {
        "default": {
            js: [x + "flash/jquery.flash.js", w + "flash.js"]
        }
    };
    g.twitter = {
        "default": {
            js: [w + "twitter.js"]
        }
    };
    g.textWithImage = {
        "default": {
            js: [x + "pinterest/jquery.pinterest.js", w + "textWithImage.js"],
            css: [x + "pinterest/jquery.pinterest.css"]
        }
    };
    g.imageSubtitle = {
        "default": {
            js: [x + "pinterest/jquery.pinterest.js", w + "imageSubtitle.js"],
            css: [x + "pinterest/jquery.pinterest.css"]
        }
    };
    g.product = {
        "default": {
            js: [O, w + "product.js", I + "jqzoom/js/jquery.jqzoom1.0.1.js", x + "superzoom/jquery.superzoom.js"],
            css: [I + "jqzoom/css/jqzoom.css",
            x + "superzoom/jquery.superzoom.css"]
        }
    };
    g.catalog = {
        "default": {
            js: [w + "catalog/catalogoptions.js", w + "catalog.js"]
        }
    };
    g.googlemaps = {
        "default": {
            js: [w + "googlemaps.js", w + "googlemaps/maputils.js", w + "googlemaps/maproutes.js", x + "message/jquery.message.js"],
            css: [x + "message/jquery.message.css"]
        }
    };
    g.googleplus = {
        "default": {
            js: [w + "googleplus.js"]
        }
    };
    g.externalSource = {
        "default": {
            js: [w + "externalSource.js"]
        }
    };
    v = c.jimdoData.newModuleSystem ? [x + "defer/jquery.defer.js", x + "numericInput/jquery.numericInput.js", A + "ui/jquery.ui.core.js",
    A + "ui/jquery.ui.datepicker.js", x + "jimdoDatePicker/jquery.jimdoDatePicker.js", w + "formnew.js"] : [w + "formnew/specialFields.js", x + "defer/jquery.defer.js", z + "Spinner.js", A + "ui/jquery.ui.core.js", A + "ui/jquery.ui.datepicker.js", x + "jimdoDatePicker/jquery.jimdoDatePicker.js", w + "formnew.js"];
    g.formnew = {
        "default": {
            js: v,
            css: [A + "themes/base/jquery.ui.datepicker.css"]
        }
    };
    g.facebook = {
        "default": {
            js: ["http://connect.facebook.net/" + (c.jimdoData.cmsLanguage || "en_US") + "/all.js#xfbml\x3d1"]
        }
    };
    g.promotion = {
        packages: {
            js: [x +
            "contentBox/jquery.contentBox.js", w + "promotion.js"],
            css: [x + "contentBox/jquery.content-box.css"]
        },
        shopoverview: {
            js: [w + "promotion.js"]
        }
    };
    g.promotion.features = g.promotion.sidebarteaser = g.promotion.packages;
    g.jimdoSlider = {
        "default": {
            js: [x + "xSlider/jquery.xSlider.js"]
        }
    };
    g.shoppingcart = {
        "default": {
            js: [x + "numericInput/jquery.numericInput.js", B + "saleutils.js", B + "shoppingcart.js", x + "browserMessage/jquery.browserMessage.js", k + "js/libs/mustache/mustache.js"],
            css: [x + "browserMessage/jquery.browserMessage.css"]
        }
    };
    g.checkout = {
        "default": {
            js: [x + "numericInput/jquery.numericInput.js", B + "saleutils.js", B + "checkout.js"]
        }
    };
    g.checkoutpaymill = {
        "default": {
            js: [B + "checkout/paymentform.js", B + "checkout/paymill.js"]
        }
    };
    g.checkoutpaypal = {
        "default": {
            js: [B + "checkout/paymentform.js", B + "checkout/paypal.js"]
        }
    };
    g.checkoutoverview = {
        "default": {
            js: [B + "checkout/overview.js"]
        }
    };
    g.cookielaw = {
        "default": {
            js: [n + "footer/cookielaw.js"]
        }
    };
    g.passwordRecovery = {
        "default": {
            js: [n + "standardPage/passwordRecovery.js", x + "form/jquery.form2Obj.js", O, x +
            "jimdoForm/jquery.jimdoForm.js", x + "form/jquery.ajaxForm.js"]
        }
    };
    g.comment = {
        "default": {
            js: [w + "comments.js"]
        }
    };
    K.bind("register", function(f, k, n, t) {
        t.variant || (t.variant = "default");
        g[n] && g[n][t.variant] ? c.xLazyLoader(c.extend({}, g[n][t.variant], {
            success: function() {
                W.removeQueue(t, n);
                if ("undefined" !== typeof c.modules[n])
                    c.modules[n](t)
            },
            error: function() {
                W.removeQueue(t);
                W.errs.push(n + "_LOAD_ERROR")
            },
            jsKey: c.jimdoData.cacheJsKey,
            cssKey: c.jimdoData.cacheCssKey
        })) : W.removeQueue(t)
    });
    c.angularInitializationData =
    function() {
        var c = {};
        return {
            setValue: function(f, g) {
                f && g && g.initializationData && (c[f] = g.initializationData)
            },
            getValue: function(f) {
                return c[f]
            }
        }
    }();
    c.regModule = function(f, g) {
        var k = g.id, n;
        k && (n = document.getElementById("cc-m-reg-" + k)) && n.parentNode && n.parentNode.removeChild(n);
        W.addModule(g, f);
        c.angularInitializationData.setValue(k, g);
        f = f.split("_");
        K.triggerHandler("register", [f[0], f[1], g])
    };
    setTimeout(function() {
        var g = f.__regModuleBuffer;
        if (g)
            for (var k = 0, n = g.length; k < n; k += 1)
                c.regModule.apply(c.regModule,
                g[k])
    }, 100);
    var W = {
        modulesQueue: {},
        errs: [],
        uniqeIdAdd: 1,
        uniqeIdRemove: 1,
        checkLoop: 0,
        loopInterval: 200,
        loopsToCheck: 40,
        addModule: function(c, f) {
            var g = this.getId(c);
            this.modulesQueue[g] = f
        },
        checkAllModulesLoaded: function() {
            var f=!1, g=!0;
            this.checkLoop++;
            for (var k in this.modulesQueue)
                if (f=!0, this.modulesQueue[k]) {
                    g=!1;
                    break
                }
            if (g && f)
                this.checkLoop = 60, c.event.trigger({
                    type: "allModulesLoaded"
                });
            else if (this.checkLoop < this.loopsToCheck) {
                var n = this;
                setTimeout(function() {
                    n.checkAllModulesLoaded()
                }, this.loopInterval,
                n)
            } else if (!g) {
                for (var t in this.modulesQueue)
                    this.modulesQueue[t] && this.errs.push(this.modulesQueue[t] + "_" + this.loopsToCheck + "_TIMES_NOT_LOADED");
                c.event.trigger({
                    type: "allModulesLoaded"
                });
                this.errs.length && c.jimdoData.isCMS && K.triggerHandler("exception.report", {
                    exDescription: this.errs.toString(),
                    appName: "unloaded_modules",
                    appVersion: "_"
                })
            }
        },
        getId: function(c, f) {
            var g = c.id || c.moduleId || (c.data ? c.data.id : 0) || 0;
            0 === g && (g = f ? this.uniqeIdRemove++ : this.uniqeIdAdd++);
            return g.toString()
        },
        removeQueue: function(c) {
            c =
            this.getId(c, !0);
            this.modulesQueue[c] ? this.modulesQueue[c]=!1 : this.errs.push(name + "_CALLED_REMOVE_NOT_IN_LIST");
            return c
        }
    };
    c(function() {
        W.checkAllModulesLoaded()
    });
    f._ccGmap = new function() {
        var c = [], f=!1;
        this.load = function() {
            f=!0;
            jimdoGen002.each(c, function(c, f) {
                f.call()
            })
        };
        this.add = function(g) {
            f ? g.call() : c.push(g)
        }
    }
})(jimdoGen002, window);
(function(c, f) {
    f.showLoginBox = function(f, k, n) {
        n ? document.location = c.jimdoData.authUrl + "app/auth/signin/" : c.sync({
            url: c.jimdoData.webPath + "cc/getloginbox.php",
            type: "post",
            dataType: "html",
            data: {
                url: f,
                page: k
            },
            success: function(f) {
                c("#loginboxOuter").html(f);
                c("#loginbox, #loginbox-darklayer").attr("class", "show");
                document.location.href = "#login";
                c("#loginPasswd").focus();
                navigator.cookieEnabled || (c("#login_container").hide(), c("#js_note").show());
                c.jimdoData.maintenance && c("#login_container").remove()
            }
        })
    };
    f.CC_cancelSendEmail = function() {
        c("#ask-for-email").hide();
        c("#login_container").show()
    };
    f.CC_close = function() {
        c("#loginbox, #loginbox-darklayer").attr("class", "hidden")
    };
    f.PopupFenster = function(c) {
        f.open(c, "Druckversion", "width\x3d700,height\x3d600,scrollbars\x3dyes,top\x3d30,left\x3d30,toolbar\x3dyes,menubar\x3dyes")
    };
    f.tellafriend = function(f) {
        c.sync({
            url: c.jimdoData.webPath + "cc/templates/tellafriend.php",
            type: "get",
            data: {
                url: location.href,
                pageID: f
            },
            dataType: "html",
            success: function(f) {
                c("#content_start").html(f)
            }
        });
        c("html,body").stop().animate({
            scrollTop: 0
        }, "slow");
        c("#content_start").show()
    };
    f.tellafriend_send = function() {
        c("#tellafriend_absenden").prop("disabled", !0);
        c.sync({
            url: c.jimdoData.webPath + "cc/tellafriend.php",
            type: "post",
            data: c("#tellafriend_form").serialize(),
            dataType: "json",
            success: f.tellafriend_check
        });
        return !1
    };
    f.tellafriend_check = function(f) {
        "ok" == f.body ? (c("#tellafriend_error").hide(), c("#tellsurl").text(f.thanks), c("#tellstatus").text(f.sent), c("#tellbody").hide()) : c("#tellafriend_error").text(f.body).show();
        c("#captchaImage" + f.captchaId).attr("src", "/app/common/captcha/index/captchaId/" + f.captchaId + "/w/160/h/29/t/" + (new Date).getTime());
        c("#tellafriend_absenden").prop("disabled", null)
    }
})(jimdoGen002, window);
"use strict";
var ModalWindow = {
    rmiClose: function(c, f) {
        var g = ModalWindow.callback;
        delete ModalWindow.callback;
        ModalWindow.returnValue = {};
        for (var k in f)
            ModalWindow.returnValue[k] = f[k];
        c.close();
        g();
        delete ModalWindow.returnValue
    },
    check: function() {
        if (ModalWindow.popup) {
            var c=!1;
            try {
                c = ModalWindow.popup.closed
            } catch (f) {}
            c ? (delete ModalWindow.popup, ModalWindow.callback && (ModalWindow.callback(), delete ModalWindow.callback)) : setTimeout(ModalWindow.check, 800)
        }
    },
    hide: function() {
        null !== ModalWindow.popup && (ModalWindow.popup.close(),
        ModalWindow.popup = null)
    },
    show: function(c, f, g) {
        if (ModalWindow.popup)
            ModalWindow.popup.focus();
        else {
            delete ModalWindow.returnValue;
            ModalWindow.callback = g;
            g = {
                height: 80,
                width: 80,
                directories: "no",
                location: "yes",
                menu: "no",
                resizable: "yes",
                scroll: "yes",
                status: "yes",
                title: "yes",
                tools: "no"
            };
            for (var k in f)
                g[k] = f[k];
            f = "undefined" !== typeof window.showModalDialog;
            "undefined" === typeof window.attachEvent && f ? (f = "dialogHeight:" + g.height + "px;", f += "dialogWidth:" + g.width + "px;", f += "resizable:" + g.resizable + ";", f += "scroll:" +
            g.scroll + ";", f += "status:" + g.status, window.showModalDialog(c, null, f), ModalWindow.callback && ModalWindow.callback()) : (f = "height\x3d" + g.height + ",", f += "width\x3d" + g.width + ",", f += "directories\x3d" + g.directories + ",", f += "location\x3d" + g.location + ",", f += "menubar\x3d" + g.menu + ",", f += "resizable\x3d" + g.resizable + ",", f += "scrollbars\x3d" + g.scroll + ",", f += "status\x3d" + g.status + ",", f += "titlebar\x3d" + g.status + ",", f += "toolbar\x3d" + g.tools, ModalWindow.popup = window.open(c, "ModalWindow", f), window.addEventListener ? window.addEventListener("unload",
            ModalWindow.hide, !0) : window.attachEvent && window.attachEvent("unload", ModalWindow.hide), setTimeout(ModalWindow.check, 800))
        }
    }
};
function JimdoSigninWindow(c, f) {
    if (JimdoSigninWindow._instance)
        return JimdoSigninWindow._instance;
    this.popupLocation = window.jimdoGen002 && jimdoGen002.jimdoData && jimdoGen002.jimdoData.authUrl ? jimdoGen002.jimdoData.authUrl : window.webserver ? window.webserver : "/";
    this.popupLocation += "app/auth/signin/index/popup/1/";
    f && (this.popupLocation += "url/" + f + "/");
    this.callback = c;
    this.show = function() {
        ModalWindow.show(this.popupLocation, {
            width: 1E3,
            height: 500
        }, this.callback)
    };
    JimdoSigninWindow._instance = this;
    return JimdoSigninWindow._instance
}
function JimdoXAjaxRequest() {
    JimdoXAjaxRequest._instances.push(this);
    for (var c = 0; c < JimdoXAjaxRequest._instances.length; c++)
        if (this === JimdoXAjaxRequest._instances[c]) {
            this._instanceId = c;
            break
        }
}
JimdoXAjaxRequest._instances = [];
JimdoXAjaxRequest.response = function(c) {
    if ("undefined" !== typeof c._id && "" !== c._id.toString())
        if ("undefined" !== typeof JimdoXAjaxRequest._instances[c._id] && null != JimdoXAjaxRequest._instances[c._id]) {
            JimdoXAjaxRequest._instances[c._id]._response = c;
            var f = navigator.userAgent, f = parseInt(f.substr(f.indexOf("WebKit") + 7, 7), 10);
            Prototype.Browser.WebKit && 523 > f && (document.getElementsByTagName("head")[0].removeChild(JimdoXAjaxRequest._instances[c._id]._scriptNode), JimdoXAjaxRequest._instances[c._id].onreadystatechange(c),
            JimdoXAjaxRequest._instances[c._id] = null)
        } else 
            window.alert("request assignment error")
};
JimdoXAjaxRequest._dispatch = function(c) {
    document.getElementsByTagName("head")[0].removeChild(c._scriptNode);
    if ("undefined" !== typeof c.onreadystatechange)
        c.onreadystatechange(c._response);
    else 
        window.alert("no callback error");
    JimdoXAjaxRequest._instances[c._instanceId] = null
};
JimdoXAjaxRequest.prototype._instanceId = null;
JimdoXAjaxRequest.prototype._scriptNode = null;
JimdoXAjaxRequest.prototype._response = null;
JimdoXAjaxRequest.prototype.open = function(c) {
    c += (0 < c.indexOf("?") ? "\x26" : "?") + "_requestId\x3d" + this._instanceId;
    this._scriptNode = document.createElement("script");
    this._scriptNode.request = this;
    this._scriptNode.type = "text/javascript";
    this._scriptNode.onload = function() {
        JimdoXAjaxRequest._dispatch(this.request)
    };
    this._scriptNode.onreadystatechange = function() {
        "complete" != this.readyState && "loaded" != this.readyState || JimdoXAjaxRequest._dispatch(this.request)
    };
    this._scriptNode.src = c;
    document.getElementsByTagName("head")[0].appendChild(this._scriptNode)
};
(function(c, f) {
    f.changeCaptcha = function(f) {
        var k = new Date;
        c("#captchaImage" + f).prop("src", "/app/common/captcha/index/captchaId/" + f + "/t/" + k.getTime())
    }
})(jimdoGen002, window);
(function() {
    function c(c, d, f) {
        f = (f || 0) - 1;
        for (var g = c ? c.length : 0; ++f < g;)
            if (c[f] === d)
                return f;
        return - 1
    }
    function f(d, f) {
        var g = typeof f;
        d = d.cache;
        if ("boolean" == g || null == f)
            return d[f] ? 0 : - 1;
        "number" != g && "string" != g && (g = "object");
        var k = "number" == g ? f: W + f;
        d = (d = d[g]) && d[k];
        return "object" == g ? d&&-1 < c(d, f) ? 0 : - 1 : d ? 0 : - 1
    }
    function g(c) {
        var d = this.cache, f = typeof c;
        if ("boolean" == f || null == c)
            d[c]=!0;
        else {
            "number" != f && "string" != f && (f = "object");
            var g = "number" == f ? c: W + c, d = d[f] || (d[f] = {});
            "object" == f ? (d[g] || (d[g] = [])).push(c) :
            d[g]=!0
        }
    }
    function k(c) {
        return c.charCodeAt(0)
    }
    function n(c, d) {
        for (var f = c.criteria, g = d.criteria, k =- 1, n = f.length; ++k < n;) {
            var t = f[k], v = g[k];
            if (t !== v) {
                if (t > v || "undefined" == typeof t)
                    return 1;
                if (t < v || "undefined" == typeof v)
                    return - 1
            }
        }
        return c.index - d.index
    }
    function v(c) {
        var d =- 1, f = c.length, k = c[0], n = c[f / 2 | 0], t = c[f - 1];
        if (k && "object" == typeof k && n && "object" == typeof n && t && "object" == typeof t)
            return !1;
        k = w();
        k["false"] = k["null"] = k["true"] = k.undefined=!1;
        n = w();
        n.array = c;
        n.cache = k;
        for (n.push = g; ++d < f;)
            n.push(c[d]);
        return n
    }
    function t(c) {
        return "\\" + d[c]
    }
    function z() {
        return K.pop() || []
    }
    function w() {
        return D.pop() || {
            array: null,
            cache: null,
            criteria: null,
            "false": !1,
            index: 0,
            "null": !1,
            number: null,
            object: null,
            push: null,
            string: null,
            "true": !1,
            undefined: !1,
            value: null
        }
    }
    function B(c) {
        c.length = 0;
        K.length < P && K.push(c)
    }
    function E(c) {
        var d = c.cache;
        d && E(d);
        c.array = c.cache = c.criteria = c.object = c.number = c.string = c.value = null;
        D.length < P && D.push(c)
    }
    function A(c, d, f) {
        d || (d = 0);
        "undefined" == typeof f && (f = c ? c.length : 0);
        var g =- 1;
        f = f - d ||
        0;
        for (var k = Array(0 > f ? 0 : f); ++g < f;)
            k[g] = c[d + g];
        return k
    }
    function x(d) {
        function g(a) {
            return a && "object" == typeof a&&!oa(a) && ua.call(a, "__wrapped__") ? a : new D(a)
        }
        function D(a, b) {
            this.__chain__=!!b;
            this.__wrapped__ = a
        }
        function K(a) {
            function b() {
                if (d) {
                    var a = A(d);
                    gb.apply(a, arguments)
                }
                if (this instanceof b) {
                    var r = aa(c.prototype), a = c.apply(r, a || arguments);
                    return sa(a) ? a : r
                }
                return c.apply(e, a || arguments)
            }
            var c = a[0], d = a[2], e = a[4];
            lb(b, a);
            return b
        }
        function P(a, c, d, e, f) {
            if (d) {
                var g = d(a);
                if ("undefined" != typeof g)
                    return g
            }
            if (sa(a)) {
                var h =
                pa.call(a);
                if (!la[h])
                    return a;
                var k = Ma[h];
                switch (h) {
                case C:
                case Ia:
                    return new k( + a);
                case ya:
                case $:
                    return new k(a);
                case Y:
                    return g = k(a.source, qa.exec(a)), g.lastIndex = a.lastIndex, g
                }
            } else 
                return a;
            h = oa(a);
            if (c) {
                var l=!e;
                e || (e = z());
                f || (f = z());
                for (var m = e.length; m--;)
                    if (e[m] == a)
                        return f[m];
                g = h ? k(a.length) : {}
            } else 
                g = h ? A(a) : Qa({}, a);
            h && (ua.call(a, "index") && (g.index = a.index), ua.call(a, "input") && (g.input = a.input));
            if (!c)
                return g;
            e.push(a);
            f.push(g);
            (h ? ha : b)(a, function(a, b) {
                g[b] = P(a, c, d, e, f)
            });
            l && (B(e), B(f));
            return g
        }
        function aa(a, b) {
            return sa(a) ? ib(a) : {}
        }
        function xa(a, b, c) {
            if ("function" != typeof a)
                return da;
            if ("undefined" == typeof b ||!("prototype"in a))
                return a;
            var d = a.__bindData__;
            if ("undefined" == typeof d && (Fa.funcNames && (d=!a.name), d = d ||!Fa.funcDecomp, !d)) {
                var e = Ya.call(a);
                Fa.funcNames || (d=!V.test(e));
                d || (d = ga.test(e), lb(a, d))
            }
            if (!1 === d ||!0 !== d && d[1] & 1)
                return a;
            switch (c) {
            case 1:
                return function(c) {
                    return a.call(b, c)
                };
            case 2:
                return function(c, d) {
                    return a.call(b, c, d)
                };
            case 3:
                return function(c, d, e) {
                    return a.call(b,
                    c, d, e)
                };
            case 4:
                return function(c, d, e, f) {
                    return a.call(b, c, d, e, f)
                }
            }
            return Lb(a, b)
        }
        function ma(a) {
            function b() {
                var a = k ? g: this;
                if (e) {
                    var r = A(e);
                    gb.apply(r, arguments)
                }
                if (f || m)
                    if (r || (r = A(arguments)), f && gb.apply(r, f), m && r.length < h)
                        return d|=16, ma([c, n ? d: d&-4, r, null, g, h]);
                r || (r = arguments);
                l && (c = a[p]);
                return this instanceof b ? (a = aa(c.prototype), r = c.apply(a, r), sa(r) ? r : a) : c.apply(a, r)
            }
            var c = a[0], d = a[1], e = a[2], f = a[3], g = a[4], h = a[5], k = d & 1, l = d & 2, m = d & 4, n = d & 8, p = c;
            lb(b, a);
            return b
        }
        function na(a, b) {
            var d =- 1, e = tb(),
            g = a ? a.length : 0, h = g >= N && e === c, k = [];
            if (h) {
                var l = v(b);
                l ? (e = f, b = l) : h=!1
            }
            for (; ++d < g;)
                l = a[d], 0 > e(b, l) && k.push(l);
            h && E(b);
            return k
        }
        function Ga(a, b, c, d) {
            d = (d || 0) - 1;
            for (var e = a ? a.length : 0, f = []; ++d < e;) {
                var g = a[d];
                if (g && "object" == typeof g && "number" == typeof g.length && (oa(g) || db(g))) {
                    b || (g = Ga(g, b, c));
                    var h =- 1, k = g.length, l = f.length;
                    for (f.length += k; ++h < k;)
                        f[l++] = g[h]
                } else 
                    c || f.push(g)
            }
            return f
        }
        function wa(b, c, d, e, f, g) {
            if (d) {
                var h = d(b, c);
                if ("undefined" != typeof h)
                    return !!h
            }
            if (b === c)
                return 0 !== b || 1 / b == 1 / c;
            if (b === b &&
            !(b && ba[typeof b] || c && ba[typeof c]))
                return !1;
            if (null == b || null == c)
                return b === c;
            var k = pa.call(b), l = pa.call(c);
            k == Ha && (k = X);
            l == Ha && (l = X);
            if (k != l)
                return !1;
            switch (k) {
            case C:
            case Ia:
                return + b ==+ c;
            case ya:
                return b!=+b ? c!=+c : 0 == b ? 1 / b == 1 / c : b ==+ c;
            case Y:
            case $:
                return b == La(c)
            }
            l = k == Aa;
            if (!l) {
                var m = ua.call(b, "__wrapped__"), n = ua.call(c, "__wrapped__");
                if (m || n)
                    return wa(m ? b.__wrapped__ : b, n ? c.__wrapped__ : c, d, e, f, g);
                if (k != X)
                    return !1;
                k = b.constructor;
                m = c.constructor;
                if (k != m&&!(ea(k) && k instanceof k && ea(m) && m instanceof
                m) && "constructor"in b && "constructor"in c)
                    return !1
            }
            k=!f;
            f || (f = z());
            g || (g = z());
            for (m = f.length; m--;)
                if (f[m] == b)
                    return g[m] == c;
            var p = 0, h=!0;
            f.push(b);
            g.push(c);
            if (l) {
                if (m = b.length, p = c.length, (h = p == m) || e)
                    for (; p--;)
                        if (l = m, n = c[p], e)
                            for (; l--&&!(h = wa(b[l], n, d, e, f, g)););
                        else if (!(h = wa(b[p], n, d, e, f, g)))
                            break
            } else 
                a(c, function(a, c, s) {
                    if (ua.call(s, c))
                        return p++, h = ua.call(b, c) && wa(b[c], a, d, e, f, g)
                    }), h&&!e && a(b, function(a, b, c) {
                        if (ua.call(c, b))
                            return h =- 1<--p
                        });
            f.pop();
            g.pop();
            k && (B(f), B(g));
            return h
        }
        function Pb(a,
        c, d, f, g) {
            (oa(c) ? ha : b)(c, function(b, c) {
                var s, h, k = b, l = a[c];
                if (b && ((h = oa(b)) || e(b))) {
                    for (k = f.length; k--;)
                        if (s = f[k] == b) {
                            l = g[k];
                            break
                        }
                    if (!s) {
                        var m;
                        d && (k = d(l, b), m = "undefined" != typeof k) && (l = k);
                        m || (l = h ? oa(l) ? l : [] : e(l) ? l : {});
                        f.push(b);
                        g.push(l);
                        m || Pb(l, b, d, f, g)
                    }
                } else 
                    d && (k = d(l, b), "undefined" == typeof k && (k = b)), "undefined" != typeof k && (l = k);
                a[c] = l
            })
        }
        function Da(a, b) {
            return a + Xa(kb() * (b - a + 1))
        }
        function rb(a, b, d) {
            var e =- 1, g = tb(), h = a ? a.length : 0, k = [], l=!b && h >= N && g === c, m = d || l ? z() : k;
            l && (m = v(m), g = f);
            for (; ++e < h;) {
                var n =
                a[e], p = d ? d(n, e, a): n;
                if (b?!e || m[m.length - 1] !== p : 0 > g(m, p)
                    )(d || l) && m.push(p), k.push(n)
            }
            l ? (B(m.array), E(m)) : d && B(m);
            return k
        }
        function sb(a) {
            return function(c, d, e) {
                var f = {};
                d = g.createCallback(d, e, 3);
                e =- 1;
                var h = c ? c.length: 0;
                if ("number" == typeof h)
                    for (; ++e < h;) {
                        var k = c[e];
                        a(f, k, d(k, e, c), c)
                    } else 
                        b(c, function(b, c, e) {
                            a(f, b, d(b, c, e), e)
                        });
                return f
            }
        }
        function Ba(a, b, c, d, e, f) {
            var g = b & 1, h = b & 4, k = b & 16, l = b & 32;
            if (!(b & 2 || ea(a)))
                throw new ia;
            k&&!c.length && (b&=-17, k = c=!1);
            l&&!d.length && (b&=-33, l = d=!1);
            var m = a && a.__bindData__;
            return m&&!0 !== m ? (m = A(m), m[2] && (m[2] = A(m[2])), m[3] && (m[3] = A(m[3])), !g || m[1] & 1 || (m[4] = e), !g && m[1] & 1 && (b|=8), !h || m[1] & 4 || (m[5] = f), k && gb.apply(m[2] || (m[2] = []), c), l && lc.apply(m[3] || (m[3] = []), d), m[1]|=b, Ba.apply(null, m)) : (1 == b || 17 === b ? K : ma)([a, b, c, d, e, f])
        }
        function gc(a) {
            return Ob[a]
        }
        function tb() {
            var a = (a = g.indexOf) === yb ? c: a;
            return a
        }
        function Ta(a) {
            return "function" == typeof a && cc.test(a)
        }
        function Sb(b) {
            var c, d;
            if (!b || pa.call(b) != X || (c = b.constructor, ea(c)&&!(c instanceof c)))
                return !1;
            a(b, function(a, b) {
                d =
                b
            });
            return "undefined" == typeof d || ua.call(b, d)
        }
        function Tb(a) {
            return fc[a]
        }
        function db(a) {
            return a && "object" == typeof a && "number" == typeof a.length && pa.call(a) == Ha ||!1
        }
        function ub(a, b, c) {
            var d = fa(a), e = d.length;
            for (b = xa(b, c, 3); e--&&(c = d[e], !1 !== b(a[c], c, a)););
            return a
        }
        function $a(b) {
            var c = [];
            a(b, function(a, b) {
                ea(a) && c.push(b)
            });
            return c.sort()
        }
        function Ub(a) {
            for (var b =- 1, c = fa(a), d = c.length, e = {}; ++b < d;) {
                var f = c[b];
                e[a[f]] = f
            }
            return e
        }
        function ea(a) {
            return "function" == typeof a
        }
        function sa(a) {
            return !(!a ||
            !ba[typeof a])
        }
        function Gb(a) {
            return "number" == typeof a || a && "object" == typeof a && pa.call(a) == ya ||!1
        }
        function ab(a) {
            return "string" == typeof a || a && "object" == typeof a && pa.call(a) == $ ||!1
        }
        function vb(a) {
            for (var b =- 1, c = fa(a), d = c.length, e = za(d); ++b < d;)
                e[b] = a[c[b]];
            return e
        }
        function Vb(a, c, d) {
            var e =- 1, f = tb(), g = a ? a.length : 0, h=!1;
            d = (0 > d ? Ea(0, g + d) : d) || 0;
            oa(a) ? h =- 1 < f(a, c, d) : "number" == typeof g ? h =- 1 < (ab(a) ? a.indexOf(c, d) : f(a, c, d)) : b(a, function(a) {
                if (++e >= d)
                    return !(h = a === c)
            });
            return h
        }
        function Qb(a, c, d) {
            var e=!0;
            c =
            g.createCallback(c, d, 3);
            d =- 1;
            var f = a ? a.length: 0;
            if ("number" == typeof f)
                for (; ++d < f && (e=!!c(a[d], d, a)););
            else 
                b(a, function(a, b, d) {
                    return e=!!c(a, b, d)
                });
            return e
        }
        function eb(a, c, d) {
            var e = [];
            c = g.createCallback(c, d, 3);
            d =- 1;
            var f = a ? a.length: 0;
            if ("number" == typeof f)
                for (; ++d < f;) {
                    var h = a[d];
                    c(h, d, a) && e.push(h)
                } else 
                    b(a, function(a, b, d) {
                        c(a, b, d) && e.push(a)
                    });
            return e
        }
        function Hb(a, c, d) {
            c = g.createCallback(c, d, 3);
            d =- 1;
            var e = a ? a.length: 0;
            if ("number" == typeof e)
                for (; ++d < e;) {
                    var f = a[d];
                    if (c(f, d, a))
                        return f
                } else {
                var h;
                b(a, function(a, b, d) {
                    if (c(a, b, d))
                        return h = a, !1
                    });
                return h
            }
        }
        function ha(a, c, d) {
            var e =- 1, f = a ? a.length : 0;
            c = c && "undefined" == typeof d ? c : xa(c, d, 3);
            if ("number" == typeof f)
                for (; ++e < f&&!1 !== c(a[e], e, a););
            else 
                b(a, c);
            return a
        }
        function wb(a, c, d) {
            var e = a ? a.length: 0;
            c = c && "undefined" == typeof d ? c : xa(c, d, 3);
            if ("number" == typeof e)
                for (; e--&&!1 !== c(a[e], e, a););
            else {
                var f = fa(a), e = f.length;
                b(a, function(a, b, d) {
                    b = f ? f[--e] : --e;
                    return c(d[b], b, d)
                })
            }
            return a
        }
        function Wa(a, c, d) {
            var e =- 1, f = a ? a.length : 0;
            c = g.createCallback(c, d,
            3);
            if ("number" == typeof f)
                for (var h = za(f); ++e < f;)
                    h[e] = c(a[e], e, a);
            else 
                h = [], b(a, function(a, b, d) {
                    h[++e] = c(a, b, d)
                });
            return h
        }
        function Ib(a, b, c) {
            var d =- Infinity, e = d;
            "function" != typeof b && c && c[b] === a && (b = null);
            if (null == b && oa(a)) {
                c =- 1;
                for (var f = a.length; ++c < f;) {
                    var h = a[c];
                    h > e && (e = h)
                }
            } else 
                b = null == b && ab(a) ? k : g.createCallback(b, c, 3), ha(a, function(a, c, f) {
                    c = b(a, c, f);
                    c > d && (d = c, e = a)
                });
            return e
        }
        function xb(a, c, d, e) {
            if (!a)
                return d;
            var f = 3 > arguments.length;
            c = g.createCallback(c, e, 4);
            var h =- 1, k = a.length;
            if ("number" ==
            typeof k)
                for (f && (d = a[++h]); ++h < k;)
                    d = c(d, a[h], h, a);
            else 
                b(a, function(a, b, e) {
                    d = f ? (f=!1, a) : c(d, a, b, e)
                });
            return d
        }
        function Wb(a, b, c, d) {
            var e = 3 > arguments.length;
            b = g.createCallback(b, d, 4);
            wb(a, function(a, d, f) {
                c = e ? (e=!1, a) : b(c, a, d, f)
            });
            return c
        }
        function Xb(a) {
            var b =- 1, c = a ? a.length : 0, d = za("number" == typeof c ? c : 0);
            ha(a, function(a) {
                var c = Da(0, ++b);
                d[b] = d[c];
                d[c] = a
            });
            return d
        }
        function Yb(a, c, d) {
            var e;
            c = g.createCallback(c, d, 3);
            d =- 1;
            var f = a ? a.length: 0;
            if ("number" == typeof f)
                for (; ++d < f&&!(e = c(a[d], d, a)););
            else 
                b(a,
                function(a, b, d) {
                    return !(e = c(a, b, d))
                });
            return !!e
        }
        function Jb(a, b, c) {
            var d = 0, e = a ? a.length: 0;
            if ("number" != typeof b && null != b) {
                var f =- 1;
                for (b = g.createCallback(b, c, 3); ++f < e && b(a[f], f, a);)
                    d++
            } else if (d = b, null == d || c)
                return a ? a[0] : I;
            return A(a, 0, cb(Ea(0, d), e))
        }
        function yb(a, b, d) {
            if ("number" == typeof d) {
                var e = a ? a.length: 0;
                d = 0 > d ? Ea(0, e + d) : d || 0
            } else if (d)
                return d = Kb(a, b), a[d] === b ? d : - 1;
            return c(a, b, d)
        }
        function mb(a, b, c) {
            if ("number" != typeof b && null != b) {
                var d = 0, e =- 1, f = a ? a.length : 0;
                for (b = g.createCallback(b, c, 3); ++e <
                f && b(a[e], e, a);)
                    d++
            } else 
                d = null == b || c ? 1 : Ea(0, b);
            return A(a, d)
        }
        function Kb(a, b, c, d) {
            var e = 0, f = a ? a.length: e;
            c = c ? g.createCallback(c, d, 1) : da;
            for (b = c(b); e < f;)
                d = e + f>>>1, c(a[d]) < b ? e = d + 1 : f = d;
            return e
        }
        function Zb(a, b, c, d) {
            "boolean" != typeof b && null != b && (d = c, c = "function" != typeof b && d && d[b] === a ? null : b, b=!1);
            null != c && (c = g.createCallback(c, d, 3));
            return rb(a, b, c)
        }
        function $b() {
            for (var a = 1 < arguments.length ? arguments : arguments[0], b =- 1, c = a ? Ib(u(a, "length")) : 0, d = za(0 > c ? 0 : c); ++b < c;)
                d[b] = u(a, b);
            return d
        }
        function ta(a,
        b) {
            var c =- 1, d = a ? a.length : 0, e = {};
            for (b ||!d || oa(a[0]) || (b = []); ++c < d;) {
                var f = a[c];
                b ? e[f] = b[c] : f && (e[f[0]] = f[1])
            }
            return e
        }
        function Lb(a, b) {
            return 2 < arguments.length ? Ba(a, 17, A(arguments, 2), null, b) : Ba(a, 1, null, null, b)
        }
        function zb(a, b, c) {
            var d, e, f, g, h, k, l, m = 0, n=!1, p=!0;
            if (!ea(a))
                throw new ia;
            b = Ea(0, b) || 0;
            if (!0 === c)
                var t=!0, p=!1;
            else 
                sa(c) && (t = c.leading, n = "maxWait"in c && (Ea(b, c.maxWait) || 0), p = "trailing"in c ? c.trailing : p);
            var u = function() {
                var c = b - (q() - g);
                0 >= c ? (e && Cb(e), c = l, e = k = l = I, c && (m = q(), f = a.apply(h, d),
                k || e || (d = h = null))) : k = hb(u, c)
            }, v = function() {
                k && Cb(k);
                e = k = l = I;
                if (p || n !== b)
                    m = q(), f = a.apply(h, d), k || e || (d = h = null)
            };
            return function() {
                d = arguments;
                g = q();
                h = this;
                l = p && (k ||!t);
                if (!1 === n)
                    var c = t&&!k;
                else {
                    e || t || (m = g);
                    var y = n - (g - m), w = 0 >= y;
                    w ? (e && (e = Cb(e)), m = g, f = a.apply(h, d)) : e || (e = hb(v, y))
                }
                w && k ? k = Cb(k) : k || b === n || (k = hb(u, b));
                c && (w=!0, f = a.apply(h, d));
                !w || k || e || (d = h = null);
                return f
            }
        }
        function da(a) {
            return a
        }
        function Pa(a, b, c) {
            var d=!0, e = b && $a(b);
            b && (c || e.length) || (null == c && (c = b), f = D, b = a, a = g, e = $a(b));
            !1 === c ? d=!1 : sa(c) &&
            "chain"in c && (d = c.chain);
            var f = a, h = ea(f);
            ha(e, function(c) {
                var e = a[c] = b[c];
                h && (f.prototype[c] = function() {
                    var b = this.__chain__, c = this.__wrapped__, g = [c];
                    gb.apply(g, arguments);
                    g = e.apply(a, g);
                    if (d || b) {
                        if (c === g && sa(g))
                            return this;
                        g = new f(g);
                        g.__chain__ = b
                    }
                    return g
                })
            })
        }
        function Va() {}
        function Ab(a) {
            return function(b) {
                return b[a]
            }
        }
        function ac() {
            return this.__wrapped__
        }
        d = d ? Ka.defaults(ra.Object(), d, Ka.pick(ra, T)) : ra;
        var za = d.Array, hc = d.Boolean, Bb = d.Date, nb = d.Function, Na = d.Math, ic = d.Number, Oa = d.Object, fb = d.RegExp,
        La = d.String, ia = d.TypeError, Ua = [], bc = Oa.prototype, jc = d._, pa = bc.toString, cc = fb("^" + La(pa).replace(/[.*+?^${}()|[\]\\]/g, "\\$\x26").replace(/toString| for [^\]]+/g, ".*?") + "$"), kc = Na.ceil, Cb = d.clearTimeout, Xa = Na.floor, Ya = nb.prototype.toString, bb = Ta(bb = Oa.getPrototypeOf) && bb, ua = bc.hasOwnProperty, gb = Ua.push, hb = d.setTimeout, Mb = Ua.splice, lc = Ua.unshift, dc = function() {
            try {
                var a = {}, b = Ta(b = Oa.defineProperty) && b, c = b(a, a, a) && b
            } catch (d) {}
            return c
        }(), ib = Ta(ib = Oa.create) && ib, Db = Ta(Db = za.isArray) && Db, Fb = d.isFinite,
        ec = d.isNaN, jb = Ta(jb = Oa.keys) && jb, Ea = Na.max, cb = Na.min, Nb = d.parseInt, kb = Na.random, Ma = {};
        Ma[Aa] = za;
        Ma[C] = hc;
        Ma[Ia] = Bb;
        Ma[Ra] = nb;
        Ma[X] = Oa;
        Ma[ya] = ic;
        Ma[Y] = fb;
        Ma[$] = La;
        D.prototype = g.prototype;
        var Fa = g.support = {};
        Fa.funcDecomp=!Ta(d.WinRTError) && ga.test(x);
        Fa.funcNames = "string" == typeof nb.name;
        g.templateSettings = {
            escape: /<%-([\s\S]+?)%>/g,
            evaluate: /<%([\s\S]+?)%>/g,
            interpolate: Za,
            variable: "",
            imports: {
                _: g
            }
        };
        ib || (aa = function() {
            function a() {}
            return function(b) {
                if (sa(b)) {
                    a.prototype = b;
                    var c = new a;
                    a.prototype =
                    null
                }
                return c || d.Object()
            }
        }());
        var lb = dc ? function(a, b) {
            Ja.value = b;
            dc(a, "__bindData__", Ja)
        }
        : Va, oa = Db || function(a) {
            return a && "object" == typeof a && "number" == typeof a.length && pa.call(a) == Aa ||!1
        }, ob = function(a) {
            var b, c = [];
            if (!a ||!ba[typeof a])
                return c;
            for (b in a)
                ua.call(a, b) && c.push(b);
            return c
        }, fa = jb ? function(a) {
            return sa(a) ? jb(a) : []
        }
        : ob, Ob = {
            "\x26": "\x26amp;",
            "\x3c": "\x26lt;",
            "\x3e": "\x26gt;",
            '"': "\x26quot;",
            "'": "\x26#39;"
        }, fc = Ub(Ob), mc = fb("(" + fa(fc).join("|") + ")", "g"), pb = fb("[" + fa(Ob).join("") + "]", "g"),
        Qa = function(a, b, c) {
            var d, e = a, f = e;
            if (!e)
                return f;
            var g = arguments, h = 0, k = "number" == typeof c ? 2: g.length;
            if (3 < k && "function" == typeof g[k - 2])
                var l = xa(g[--k - 1], g[k--], 2);
            else 
                2 < k && "function" == typeof g[k - 1] && (l = g[--k]);
            for (; ++h < k;)
                if ((e = g[h]) && ba[typeof e])
                    for (var m =- 1, n = ba[typeof e] && fa(e), p = n ? n.length : 0; ++m < p;)
                        d = n[m], f[d] = l ? l(f[d], e[d]) : e[d];
            return f
        }, Eb = function(a, b, c) {
            var d, e = a, f = e;
            if (!e)
                return f;
            for (var g = arguments, h = 0, k = "number" == typeof c ? 2 : g.length; ++h < k;)
                if ((e = g[h]) && ba[typeof e])
                    for (var l =- 1, m = ba[typeof e] &&
                    fa(e), n = m ? m.length : 0; ++l < n;)
                        d = m[l], "undefined" == typeof f[d] && (f[d] = e[d]);
            return f
        }, a = function(a, b, c) {
            var d;
            if (!a ||!ba[typeof a])
                return a;
            b = b && "undefined" == typeof c ? b : xa(b, c, 3);
            for (d in a)
                if (!1 === b(a[d], d, a))
                    break;
            return a
        }, b = function(a, b, c) {
            if (!a ||!ba[typeof a])
                return a;
            b = b && "undefined" == typeof c ? b : xa(b, c, 3);
            for (var d =- 1, e = ba[typeof a] && fa(a), f = e ? e.length : 0; ++d < f && (c = e[d], !1 !== b(a[c], c, a)););
            return a
        }, e = bb ? function(a) {
            if (!a || pa.call(a) != X)
                return !1;
            var b = a.valueOf, c = Ta(b) && (c = bb(b)) && bb(c);
            return c ?
            a == c || bb(a) == c : Sb(a)
        }
        : Sb, l = sb(function(a, b, c) {
            ua.call(a, c) ? a[c]++ : a[c] = 1
        }), m = sb(function(a, b, c) {
            (ua.call(a, c) ? a[c] : a[c] = []).push(b)
        }), h = sb(function(a, b, c) {
            a[c] = b
        }), u = Wa, q = Ta(q = Bb.now) && q || function() {
            return (new Bb).getTime()
        }, qc = 8 == Nb(Z + "08") ? Nb: function(a, b) {
            return Nb(ab(a) ? a.replace(L, "") : a, b || 0)
        };
        g.after = function(a, b) {
            if (!ea(b))
                throw new ia;
            return function() {
                if (1>--a)
                    return b.apply(this, arguments)
            }
        };
        g.assign = Qa;
        g.at = function(a) {
            for (var b = arguments, c =- 1, d = Ga(b, !0, !1, 1), b = b[2] && b[2][b[1]] === a ? 1 : d.length,
            e = za(b); ++c < b;)
                e[c] = a[d[c]];
            return e
        };
        g.bind = Lb;
        g.bindAll = function(a) {
            for (var b = 1 < arguments.length ? Ga(arguments, !0, !1, 1) : $a(a), c =- 1, d = b.length; ++c < d;) {
                var e = b[c];
                a[e] = Ba(a[e], 1, null, null, a)
            }
            return a
        };
        g.bindKey = function(a, b) {
            return 2 < arguments.length ? Ba(b, 19, A(arguments, 2), null, a) : Ba(b, 3, null, null, a)
        };
        g.chain = function(a) {
            a = new D(a);
            a.__chain__=!0;
            return a
        };
        g.compact = function(a) {
            for (var b =- 1, c = a ? a.length : 0, d = []; ++b < c;) {
                var e = a[b];
                e && d.push(e)
            }
            return d
        };
        g.compose = function() {
            for (var a = arguments, b = a.length; b--;)
                if (!ea(a[b]))
                    throw new ia;
            return function() {
                for (var b = arguments, c = a.length; c--;)
                    b = [a[c].apply(this, b)];
                return b[0]
            }
        };
        g.constant = function(a) {
            return function() {
                return a
            }
        };
        g.countBy = l;
        g.create = function(a, b) {
            var c = aa(a);
            return b ? Qa(c, b) : c
        };
        g.createCallback = function(a, b, c) {
            var d = typeof a;
            if (null == a || "function" == d)
                return xa(a, b, c);
            if ("object" != d)
                return Ab(a);
            var e = fa(a), f = e[0], g = a[f];
            return 1 != e.length || g !== g || sa(g) ? function(b) {
                for (var c = e.length, d=!1; c--&&(d = wa(b[e[c]], a[e[c]], null, !0)););
                return d
            } : function(a) {
                a = a[f];
                return g ===
                a && (0 !== g || 1 / g == 1 / a)
            }
        };
        g.curry = function(a, b) {
            b = "number" == typeof b ? b : + b || a.length;
            return Ba(a, 4, null, null, null, b)
        };
        g.debounce = zb;
        g.defaults = Eb;
        g.defer = function(a) {
            if (!ea(a))
                throw new ia;
            var b = A(arguments, 1);
            return hb(function() {
                a.apply(I, b)
            }, 1)
        };
        g.delay = function(a, b) {
            if (!ea(a))
                throw new ia;
            var c = A(arguments, 2);
            return hb(function() {
                a.apply(I, c)
            }, b)
        };
        g.difference = function(a) {
            return na(a, Ga(arguments, !0, !0, 1))
        };
        g.filter = eb;
        g.flatten = function(a, b, c, d) {
            "boolean" != typeof b && null != b && (d = c, c = "function" != typeof b &&
            d && d[b] === a ? null : b, b=!1);
            null != c && (a = Wa(a, c, d));
            return Ga(a, b)
        };
        g.forEach = ha;
        g.forEachRight = wb;
        g.forIn = a;
        g.forInRight = function(b, c, d) {
            var e = [];
            a(b, function(a, b) {
                e.push(b, a)
            });
            var f = e.length;
            for (c = xa(c, d, 3); f--&&!1 !== c(e[f--], e[f], b);
            );
            return b
        };
        g.forOwn = b;
        g.forOwnRight = ub;
        g.functions = $a;
        g.groupBy = m;
        g.indexBy = h;
        g.initial = function(a, b, c) {
            var d = 0, e = a ? a.length: 0;
            if ("number" != typeof b && null != b) {
                var f = e;
                for (b = g.createCallback(b, c, 3); f--&&b(a[f], f, a);
                )d++
            } else 
                d = null == b || c ? 1 : b || d;
            return A(a, 0, cb(Ea(0, e -
            d), e))
        };
        g.intersection = function() {
            for (var a = [], b =- 1, d = arguments.length, e = z(), g = tb(), h = g === c, k = z(); ++b < d;) {
                var l = arguments[b];
                if (oa(l) || db(l))
                    a.push(l), e.push(h && l.length >= N && v(b ? a[b] : k))
            }
            var h = a[0], m =- 1, n = h ? h.length : 0, p = [];
            a: for (; ++m < n;) {
                var q = e[0], l = h[m];
                if (0 > (q ? f(q, l) : g(k, l))) {
                    b = d;
                    for ((q || k).push(l); --b;)
                        if (q = e[b], 0 > (q ? f(q, l) : g(a[b], l)))
                            continue a;
                    p.push(l)
                }
            }
            for (; d--;)(q = e[d]) 
                && E(q);
            B(e);
            B(k);
            return p
        };
        g.invert = Ub;
        g.invoke = function(a, b) {
            var c = A(arguments, 2), d =- 1, e = "function" == typeof b, f = a ? a.length :
            0, g = za("number" == typeof f ? f : 0);
            ha(a, function(a) {
                g[++d] = (e ? b : a[b]).apply(a, c)
            });
            return g
        };
        g.keys = fa;
        g.map = Wa;
        g.mapValues = function(a, c, d) {
            var e = {};
            c = g.createCallback(c, d, 3);
            b(a, function(a, b, d) {
                e[b] = c(a, b, d)
            });
            return e
        };
        g.max = Ib;
        g.memoize = function(a, b) {
            if (!ea(a))
                throw new ia;
            var c = function() {
                var d = c.cache, e = b ? b.apply(this, arguments): W + arguments[0];
                return ua.call(d, e) ? d[e] : d[e] = a.apply(this, arguments)
            };
            c.cache = {};
            return c
        };
        g.merge = function(a) {
            var b = arguments, c = 2;
            if (!sa(a))
                return a;
            "number" != typeof b[2] &&
            (c = b.length);
            if (3 < c && "function" == typeof b[c - 2])
                var d = xa(b[--c - 1], b[c--], 2);
            else 
                2 < c && "function" == typeof b[c - 1] && (d = b[--c]);
            for (var b = A(arguments, 1, c), e =- 1, f = z(), g = z(); ++e < c;)
                Pb(a, b[e], d, f, g);
            B(f);
            B(g);
            return a
        };
        g.min = function(a, b, c) {
            var d = Infinity, e = d;
            "function" != typeof b && c && c[b] === a && (b = null);
            if (null == b && oa(a)) {
                c =- 1;
                for (var f = a.length; ++c < f;) {
                    var h = a[c];
                    h < e && (e = h)
                }
            } else 
                b = null == b && ab(a) ? k : g.createCallback(b, c, 3), ha(a, function(a, c, f) {
                    c = b(a, c, f);
                    c < d && (d = c, e = a)
                });
            return e
        };
        g.omit = function(b, c, d) {
            var e =
            {};
            if ("function" != typeof c) {
                var f = [];
                a(b, function(a, b) {
                    f.push(b)
                });
                for (var f = na(f, Ga(arguments, !0, !1, 1)), h =- 1, k = f.length; ++h < k;) {
                    var l = f[h];
                    e[l] = b[l]
                }
            } else 
                c = g.createCallback(c, d, 3), a(b, function(a, b, d) {
                    c(a, b, d) || (e[b] = a)
                });
            return e
        };
        g.once = function(a) {
            var b, c;
            if (!ea(a))
                throw new ia;
            return function() {
                if (b)
                    return c;
                b=!0;
                c = a.apply(this, arguments);
                a = null;
                return c
            }
        };
        g.pairs = function(a) {
            for (var b =- 1, c = fa(a), d = c.length, e = za(d); ++b < d;) {
                var f = c[b];
                e[b] = [f, a[f]]
            }
            return e
        };
        g.partial = function(a) {
            return Ba(a,
            16, A(arguments, 1))
        };
        g.partialRight = function(a) {
            return Ba(a, 32, null, A(arguments, 1))
        };
        g.pick = function(b, c, d) {
            var e = {};
            if ("function" != typeof c)
                for (var f =- 1, h = Ga(arguments, !0, !1, 1), k = sa(b) ? h.length : 0; ++f < k;) {
                    var l = h[f];
                    l in b && (e[l] = b[l])
                } else 
                    c = g.createCallback(c, d, 3), a(b, function(a, b, d) {
                        c(a, b, d) && (e[b] = a)
                    });
            return e
        };
        g.pluck = u;
        g.property = Ab;
        g.pull = function(a) {
            for (var b = arguments, c = 0, d = b.length, e = a ? a.length : 0; ++c < d;)
                for (var f =- 1, g = b[c]; ++f < e;)
                    a[f] === g && (Mb.call(a, f--, 1), e--);
            return a
        };
        g.range = function(a,
        b, c) {
            a =+ a || 0;
            c = "number" == typeof c ? c : + c || 1;
            null == b && (b = a, a = 0);
            var d =- 1;
            b = Ea(0, kc((b - a) / (c || 1)));
            for (var e = za(b); ++d < b;)
                e[d] = a, a += c;
            return e
        };
        g.reject = function(a, b, c) {
            b = g.createCallback(b, c, 3);
            return eb(a, function(a, c, d) {
                return !b(a, c, d)
            })
        };
        g.remove = function(a, b, c) {
            var d =- 1, e = a ? a.length : 0, f = [];
            for (b = g.createCallback(b, c, 3); ++d < e;)
                c = a[d], b(c, d, a) && (f.push(c), Mb.call(a, d--, 1), e--);
            return f
        };
        g.rest = mb;
        g.shuffle = Xb;
        g.sortBy = function(a, b, c) {
            var d =- 1, e = oa(b), f = a ? a.length : 0, h = za("number" == typeof f ? f : 0);
            e ||
            (b = g.createCallback(b, c, 3));
            ha(a, function(a, c, f) {
                var g = h[++d] = w();
                e ? g.criteria = Wa(b, function(b) {
                    return a[b]
                }) : (g.criteria = z())[0] = b(a, c, f);
                g.index = d;
                g.value = a
            });
            f = h.length;
            for (h.sort(n); f--;)
                a = h[f], h[f] = a.value, e || B(a.criteria), E(a);
            return h
        };
        g.tap = function(a, b) {
            b(a);
            return a
        };
        g.throttle = function(a, b, c) {
            var d=!0, e=!0;
            if (!ea(a))
                throw new ia;
            !1 === c ? d=!1 : sa(c) && (d = "leading"in c ? c.leading : d, e = "trailing"in c ? c.trailing : e);
            va.leading = d;
            va.maxWait = b;
            va.trailing = e;
            return zb(a, b, va)
        };
        g.times = function(a, b,
        c) {
            a =- 1 < (a =+ a) ? a : 0;
            var d =- 1, e = za(a);
            for (b = xa(b, c, 1); ++d < a;)
                e[d] = b(d);
            return e
        };
        g.toArray = function(a) {
            return a && "number" == typeof a.length ? A(a) : vb(a)
        };
        g.transform = function(a, c, d, e) {
            var f = oa(a);
            if (null == d)
                if (f)
                    d = [];
                else {
                    var h = a && a.constructor;
                    d = aa(h && h.prototype)
                }
            c && (c = g.createCallback(c, e, 4), (f ? ha : b)(a, function(a, b, e) {
                return c(d, a, b, e)
            }));
            return d
        };
        g.union = function() {
            return rb(Ga(arguments, !0, !0))
        };
        g.uniq = Zb;
        g.values = vb;
        g.where = eb;
        g.without = function(a) {
            return na(a, A(arguments, 1))
        };
        g.wrap = function(a,
        b) {
            return Ba(b, 16, [a])
        };
        g.xor = function() {
            for (var a =- 1, b = arguments.length; ++a < b;) {
                var c = arguments[a];
                if (oa(c) || db(c))
                    var d = d ? rb(na(d, c).concat(na(c, d))): c
            }
            return d || []
        };
        g.zip = $b;
        g.zipObject = ta;
        g.collect = Wa;
        g.drop = mb;
        g.each = ha;
        g.eachRight = wb;
        g.extend = Qa;
        g.methods = $a;
        g.object = ta;
        g.select = eb;
        g.tail = mb;
        g.unique = Zb;
        g.unzip = $b;
        Pa(g);
        g.clone = function(a, b, c, d) {
            "boolean" != typeof b && null != b && (d = c, c = b, b=!1);
            return P(a, b, "function" == typeof c && xa(c, d, 1))
        };
        g.cloneDeep = function(a, b, c) {
            return P(a, !0, "function" == typeof b &&
            xa(b, c, 1))
        };
        g.contains = Vb;
        g.escape = function(a) {
            return null == a ? "" : La(a).replace(pb, gc)
        };
        g.every = Qb;
        g.find = Hb;
        g.findIndex = function(a, b, c) {
            var d =- 1, e = a ? a.length : 0;
            for (b = g.createCallback(b, c, 3); ++d < e;)
                if (b(a[d], d, a))
                    return d;
            return - 1
        };
        g.findKey = function(a, c, d) {
            var e;
            c = g.createCallback(c, d, 3);
            b(a, function(a, b, d) {
                if (c(a, b, d))
                    return e = b, !1
            });
            return e
        };
        g.findLast = function(a, b, c) {
            var d;
            b = g.createCallback(b, c, 3);
            wb(a, function(a, c, e) {
                if (b(a, c, e))
                    return d = a, !1
            });
            return d
        };
        g.findLastIndex = function(a, b, c) {
            var d =
            a ? a.length: 0;
            for (b = g.createCallback(b, c, 3); d--;)
                if (b(a[d], d, a))
                    return d;
            return - 1
        };
        g.findLastKey = function(a, b, c) {
            var d;
            b = g.createCallback(b, c, 3);
            ub(a, function(a, c, e) {
                if (b(a, c, e))
                    return d = c, !1
            });
            return d
        };
        g.has = function(a, b) {
            return a ? ua.call(a, b) : !1
        };
        g.identity = da;
        g.indexOf = yb;
        g.isArguments = db;
        g.isArray = oa;
        g.isBoolean = function(a) {
            return !0 === a ||!1 === a || a && "object" == typeof a && pa.call(a) == C ||!1
        };
        g.isDate = function(a) {
            return a && "object" == typeof a && pa.call(a) == Ia ||!1
        };
        g.isElement = function(a) {
            return a && 1 ===
            a.nodeType ||!1
        };
        g.isEmpty = function(a) {
            var c=!0;
            if (!a)
                return c;
            var d = pa.call(a), e = a.length;
            if (d == Aa || d == $ || d == Ha || d == X && "number" == typeof e && ea(a.splice))
                return !e;
            b(a, function() {
                return c=!1
            });
            return c
        };
        g.isEqual = function(a, b, c, d) {
            return wa(a, b, "function" == typeof c && xa(c, d, 2))
        };
        g.isFinite = function(a) {
            return Fb(a)&&!ec(parseFloat(a))
        };
        g.isFunction = ea;
        g.isNaN = function(a) {
            return Gb(a) && a!=+a
        };
        g.isNull = function(a) {
            return null === a
        };
        g.isNumber = Gb;
        g.isObject = sa;
        g.isPlainObject = e;
        g.isRegExp = function(a) {
            return a &&
            "object" == typeof a && pa.call(a) == Y ||!1
        };
        g.isString = ab;
        g.isUndefined = function(a) {
            return "undefined" == typeof a
        };
        g.lastIndexOf = function(a, b, c) {
            var d = a ? a.length: 0;
            for ("number" == typeof c && (d = (0 > c ? Ea(0, d + c) : cb(c, d - 1)) + 1); d--;)
                if (a[d] === b)
                    return d;
            return - 1
        };
        g.mixin = Pa;
        g.noConflict = function() {
            d._ = jc;
            return this
        };
        g.noop = Va;
        g.now = q;
        g.parseInt = qc;
        g.random = function(a, b, c) {
            var d = null == a, e = null == b;
            null == c && ("boolean" == typeof a && e ? (c = a, a = 1) : e || "boolean" != typeof b || (c = b, e=!0));
            d && e && (b = 1);
            a =+ a || 0;
            e ? (b = a, a = 0) : b =+ b ||
            0;
            return c || a%1 || b%1 ? (c = kb(), cb(a + c * (b - a + parseFloat("1e-" + ((c + "").length - 1))), b)) : Da(a, b)
        };
        g.reduce = xb;
        g.reduceRight = Wb;
        g.result = function(a, b) {
            if (a) {
                var c = a[b];
                return ea(c) ? a[b]() : c
            }
        };
        g.runInContext = x;
        g.size = function(a) {
            var b = a ? a.length: 0;
            return "number" == typeof b ? b : fa(a).length
        };
        g.some = Yb;
        g.sortedIndex = Kb;
        g.template = function(a, b, c) {
            var d = g.templateSettings;
            a = La(a || "");
            c = Eb({}, c, d);
            var e = Eb({}, c.imports, d.imports), d = fa(e), e = vb(e), f, h = 0, k = c.interpolate || M, l = "__p +\x3d '", k = fb((c.escape || M).source + "|" +
            k.source + "|" + (k === Za ? R : M).source + "|" + (c.evaluate || M).source + "|$", "g");
            a.replace(k, function(b, c, d, e, g, k) {
                d || (d = e);
                l += a.slice(h, k).replace(Q, t);
                c && (l += "' +\n__e(" + c + ") +\n'");
                g && (f=!0, l += "';\n" + g + ";\n__p +\x3d '");
                d && (l += "' +\n((__t \x3d (" + d + ")) \x3d\x3d null ? '' : __t) +\n'");
                h = k + b.length;
                return b
            });
            var l = l + "';\n", m = k = c.variable;
            m || (k = "obj", l = "with (" + k + ") {\n" + l + "\n}\n");
            l = (f ? l.replace(G, "") : l).replace(ja, "$1").replace(U, "$1;");
            l = "function(" + k + ") {\n" + (m ? "" : k + " || (" + k + " \x3d {});\n") + "var __t, __p \x3d '', __e \x3d _.escape" +
            (f ? ", __j \x3d Array.prototype.join;\nfunction print() { __p +\x3d __j.call(arguments, '') }\n" : ";\n") + l + "return __p\n}";
            c = "\n/*\n//# sourceURL\x3d" + (c.sourceURL || "/lodash/template/source[" + ka++ + "]") + "\n*/";
            try {
                var n = nb(d, "return " + l + c).apply(I, e)
            } catch (q) {
                throw q.source = l, q;
            }
            if (b)
                return n(b);
            n.source = l;
            return n
        };
        g.unescape = function(a) {
            return null == a ? "" : La(a).replace(mc, Tb)
        };
        g.uniqueId = function(a) {
            var b=++O;
            return La(null == a ? "" : a) + b
        };
        g.all = Qb;
        g.any = Yb;
        g.detect = Hb;
        g.findWhere = Hb;
        g.foldl = xb;
        g.foldr =
        Wb;
        g.include = Vb;
        g.inject = xb;
        Pa(function() {
            var a = {};
            b(g, function(b, c) {
                g.prototype[c] || (a[c] = b)
            });
            return a
        }(), !1);
        g.first = Jb;
        g.last = function(a, b, c) {
            var d = 0, e = a ? a.length: 0;
            if ("number" != typeof b && null != b) {
                var f = e;
                for (b = g.createCallback(b, c, 3); f--&&b(a[f], f, a);
                )d++
            } else if (d = b, null == d || c)
                return a ? a[e - 1] : I;
            return A(a, Ea(0, e - d))
        };
        g.sample = function(a, b, c) {
            a && "number" != typeof a.length && (a = vb(a));
            if (null == b || c)
                return a ? a[Da(0, a.length - 1)] : I;
            a = Xb(a);
            a.length = cb(Ea(0, b), a.length);
            return a
        };
        g.take = Jb;
        g.head = Jb;
        b(g, function(a, b) {
            var c = "sample" !== b;
            g.prototype[b] || (g.prototype[b] = function(b, d) {
                var e = this.__chain__, f = a(this.__wrapped__, b, d);
                return e || null != b && (!d || c && "function" == typeof b) ? new D(f, e) : f
            })
        });
        g.VERSION = "2.4.1";
        g.prototype.chain = function() {
            this.__chain__=!0;
            return this
        };
        g.prototype.toString = function() {
            return La(this.__wrapped__)
        };
        g.prototype.value = ac;
        g.prototype.valueOf = ac;
        ha(["join", "pop", "shift"], function(a) {
            var b = Ua[a];
            g.prototype[a] = function() {
                var a = this.__chain__, c = b.apply(this.__wrapped__,
                arguments);
                return a ? new D(c, a) : c
            }
        });
        ha(["push", "reverse", "sort", "unshift"], function(a) {
            var b = Ua[a];
            g.prototype[a] = function() {
                b.apply(this.__wrapped__, arguments);
                return this
            }
        });
        ha(["concat", "slice", "splice"], function(a) {
            var b = Ua[a];
            g.prototype[a] = function() {
                return new D(b.apply(this.__wrapped__, arguments), this.__chain__)
            }
        });
        return g
    }
    var I, K = [], D = [], O = 0, W =+ new Date + "", N = 75, P = 40, Z = " \t\x0B\f\u00a0\ufeff\n\r\u2028\u2029\u1680\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000",
    G = /\b__p \+= '';/g, ja = /\b(__p \+=) '' \+/g, U = /(__e\(.*?\)|\b__t\)) \+\n'';/g, R = /\$\{([^\\}]*(?:\\.[^\\}]*)*)\}/g, qa = /\w*$/, V = /^\s*function[ \n\r\t]+\w/, Za = /<%=([\s\S]+?)%>/g, L = RegExp("^[" + Z + "]*0+(?\x3d.$)"), M = /($^)/, ga = /\bthis\b/, Q = /['\n\r\t\u2028\u2029\\]/g, T = "Array Boolean Date Function Math Number Object RegExp String _ attachEvent clearTimeout isFinite isNaN parseInt setTimeout".split(" "), ka = 0, Ha = "[object Arguments]", Aa = "[object Array]", C = "[object Boolean]", Ia = "[object Date]", Ra = "[object Function]",
    ya = "[object Number]", X = "[object Object]", Y = "[object RegExp]", $ = "[object String]", la = {};
    la[Ra]=!1;
    la[Ha] = la[Aa] = la[C] = la[Ia] = la[ya] = la[X] = la[Y] = la[$]=!0;
    var va = {
        leading: !1,
        maxWait: 0,
        trailing: !1
    }, Ja = {
        configurable: !1,
        enumerable: !1,
        value: null,
        writable: !1
    }, ba = {
        "boolean": !1,
        "function": !0,
        object: !0,
        number: !1,
        string: !1,
        undefined: !1
    }, d = {
        "\\": "\\",
        "'": "'",
        "\n": "n",
        "\r": "r",
        "\t": "t",
        "\u2028": "u2028",
        "\u2029": "u2029"
    }, ra = ba[typeof window] && window || this, ma = ba[typeof exports] && exports&&!exports.nodeType && exports,
    aa = ba[typeof module] && module&&!module.nodeType && module, na = aa && aa.exports === ma && ma, wa = ba[typeof global] && global;
    !wa || wa.global !== wa && wa.window !== wa || (ra = wa);
    var Ka = x();
    "function" == typeof define && "object" == typeof define.amd && define.amd ? (ra._ = Ka, define(function() {
        return Ka
    })) : ma && aa ? na ? (aa.exports = Ka)._ = Ka : ma._ = Ka : ra._ = Ka
}).call(this);
(function(c, f) {
    c._ = f.noConflict()
})(jimdoGen002, window._);
(function(c, f, g, k) {
    c.fn.doubleTapToGo = function(k) {
        if (!("ontouchstart"in f || navigator.msMaxTouchPoints || navigator.userAgent.toLowerCase().match(/windows phone os 7/i)))
            return !1;
        this.each(function() {
            var f=!1;
            c(this).on("click", function(g) {
                var k = c(this);
                k[0] != f[0] && (g.preventDefault(), f = k)
            });
            c(g).on("click touchstart MSPointerDown", function(g) {
                var k=!0;
                g = c(g.target).parents();
                for (var n = 0; n < g.length; n++)
                    g[n] == f[0] && (k=!1);
                k && (f=!1)
            })
        });
        return this
    }
})(jimdoGen002, window, document);
