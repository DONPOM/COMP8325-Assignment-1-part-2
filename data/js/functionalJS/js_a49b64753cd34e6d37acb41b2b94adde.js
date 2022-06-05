/*!
 * jQuery JavaScript Library v1.4.4
 * http://jquery.com/
 *
 * Copyright 2010, John Resig
 * Dual licensed under the MIT or GPL Version 2 licenses.
 * http://jquery.org/license
 *
 * Includes Sizzle.js
 * http://sizzlejs.com/
 * Copyright 2010, The Dojo Foundation
 * Released under the MIT, BSD, and GPL Licenses.
 *
 * Date: Thu Nov 11 19:04:53 2010 -0500
 */
(function(K, D) {
    function Qa(a, b, d) {
        if (d === D && a.nodeType === 1) {
            d = a.getAttribute("data-" + b);
            if (typeof d === "string") {
                try {
                    d = d === "true" ? true : d === "false" ? false : d === "null" ? null : !c.isNaN(d) ? parseFloat(d) : tb.test(d) ? c.parseJSON(d) : d
                } catch (e) {}
                c.data(a, b, d)
            } else 
                d = D
        }
        return d
    }
    function sa() {
        return false
    }
    function Fa() {
        return true
    }
    function Ra(a, b, d) {
        d[0].type = a;
        return c.event.handle.apply(b, d)
    }
    function ub(a) {
        var b, d, e, f, g, j, n, q, A, s, B, F = [];
        f = [];
        g = c.data(this, this.nodeType ? "events" : "__events__");
        if (typeof g === "function")
            g =
            g.events;
        if (!(a.liveFired === this ||!g ||!g.live || a.button && a.type === "click")) {
            if (a.namespace)
                B = RegExp("(^|\\.)" + a.namespace.split(".").join("\\.(?:.*\\.)?") + "(\\.|$)");
            a.liveFired = this;
            var P = g.live.slice(0);
            for (n = 0; n < P.length; n++) {
                g = P[n];
                g.origType.replace(va, "") === a.type ? f.push(g.selector) : P.splice(n--, 1)
            }
            f = c(a.target).closest(f, a.currentTarget);
            q = 0;
            for (A = f.length; q < A; q++) {
                s = f[q];
                for (n = 0; n < P.length; n++) {
                    g = P[n];
                    if (s.selector === g.selector && (!B || B.test(g.namespace))) {
                        j = s.elem;
                        e = null;
                        if (g.preType === "mouseenter" ||
                        g.preType === "mouseleave") {
                            a.type = g.preType;
                            e = c(a.relatedTarget).closest(g.selector)[0]
                        }
                        if (!e || e !== j)
                            F.push({
                                elem: j,
                                handleObj: g,
                                level: s.level
                            })
                        }
                }
            }
            q = 0;
            for (A = F.length; q < A; q++) {
                f = F[q];
                if (d && f.level > d)
                    break;
                a.currentTarget = f.elem;
                a.data = f.handleObj.data;
                a.handleObj = f.handleObj;
                B = f.handleObj.origHandler.apply(f.elem, arguments);
                if (B === false || a.isPropagationStopped()) {
                    d = f.level;
                    if (B === false)
                        b = false;
                    if (a.isImmediatePropagationStopped())
                        break
                }
            }
            return b
        }
    }
    function wa(a, b) {
        return (a && a !== "*" ? a + "." : "") + b.replace(vb,
        "`").replace(wb, "&")
    }
    function Sa(a, b, d) {
        if (c.isFunction(b))
            return c.grep(a, function(f, g) {
                return !!b.call(f, g, f) === d
            });
        else if (b.nodeType)
            return c.grep(a, function(f) {
                return f === b === d
            });
        else if (typeof b === "string") {
            var e = c.grep(a, function(f) {
                return f.nodeType === 1
            });
            if (xb.test(b))
                return c.filter(b, e, !d);
            else 
                b = c.filter(b, e)
        }
        return c.grep(a, function(f) {
            return c.inArray(f, b) >= 0 === d
        })
    }
    function Ta(a, b) {
        var d = 0;
        b.each(function() {
            if (this.nodeName === (a[d] && a[d].nodeName)) {
                var e = c.data(a[d++]), f = c.data(this,
                e);
                if (e = e && e.events) {
                    delete f.handle;
                    f.events = {};
                    for (var g in e)
                        for (var j in e[g])
                            c.event.add(this, g, e[g][j], e[g][j].data)
                        }
            }
        })
    }
    function yb(a, b) {
        b.src ? c.ajax({
            url: b.src,
            async: false,
            dataType: "script"
        }) : c.globalEval(b.text || b.textContent || b.innerHTML || "");
        b.parentNode && b.parentNode.removeChild(b)
    }
    function Ua(a, b, d) {
        var e = b === "width" ? a.offsetWidth: a.offsetHeight;
        if (d === "border")
            return e;
        c.each(b === "width" ? zb : Ab, function() {
            d || (e -= parseFloat(c.css(a, "padding" + this)) || 0);
            if (d === "margin")
                e += parseFloat(c.css(a,
                "margin" + this)) || 0;
            else 
                e -= parseFloat(c.css(a, "border" + this + "Width")) || 0
        });
        return e
    }
    function Ga(a, b, d, e) {
        if (c.isArray(b) && b.length)
            c.each(b, function(f, g) {
                d || Bb.test(a) ? e(a, g) : Ga(a + "[" + (typeof g === "object" || c.isArray(g) ? f : "") + "]", g, d, e)
            });
        else if (!d && b != null && typeof b === "object")
            c.isEmptyObject(b) ? e(a, "") : c.each(b, function(f, g) {
                Ga(a + "[" + f + "]", g, d, e)
            });
        else 
            e(a, b)
    }
    function ma(a, b) {
        var d = {};
        c.each(Va.concat.apply([], Va.slice(0, b)), function() {
            d[this] = a
        });
        return d
    }
    function Wa(a) {
        if (!Ha[a]) {
            var b = c("<" +
            a + ">").appendTo("body"), d = b.css("display");
            b.remove();
            if (d === "none" || d === "")
                d = "block";
            Ha[a] = d
        }
        return Ha[a]
    }
    function Ia(a) {
        return c.isWindow(a) ? a : a.nodeType === 9 ? a.defaultView || a.parentWindow : false
    }
    var y = K.document, c = function() {
        function a() {
            if (!b.isReady) {
                try {
                    y.documentElement.doScroll("left")
                } catch (l) {
                    setTimeout(a, 1);
                    return 
                }
                b.ready()
            }
        }
        var b = function(l, r) {
            return new b.fn.init(l, r)
        }, d = K.jQuery, e = K.$, f, g = /^(?:[^<]*(<[\w\W]+>)[^>]*$|#([\w\-]+)$)/, j = /\S/, n = /^\s+/, q = /\s+$/, A = /\W/, s = /\d/, B = /^<(\w+)\s*\/?>(?:<\/\1>)?$/,
        F = /^[\],:{}\s]*$/, P = /\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, z = /"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, Q = /(?:^|:|,)(?:\s*\[)+/g, Y = /(webkit)[ \/]([\w.]+)/, $ = /(opera)(?:.*version)?[ \/]([\w.]+)/, xa = /(msie) ([\w.]+)/, na = /(mozilla)(?:.*? rv:([\w.]+))?/, U = navigator.userAgent, aa = false, E = [], ba, ya = Object.prototype.toString, ka = Object.prototype.hasOwnProperty, ca = Array.prototype.push, ha = Array.prototype.slice, ea = String.prototype.trim, V = Array.prototype.indexOf, X = {};
        b.fn = b.prototype = {
            init: function(l,
            r) {
                var u, v, I;
                if (!l)
                    return this;
                if (l.nodeType) {
                    this.context = this[0] = l;
                    this.length = 1;
                    return this
                }
                if (l === "body"&&!r && y.body) {
                    this.context = y;
                    this[0] = y.body;
                    this.selector = "body";
                    this.length = 1;
                    return this
                }
                if (typeof l === "string")
                    if ((u = g.exec(l)) && (u[1] ||!r))
                        if (u[1]) {
                            I = r ? r.ownerDocument || r : y;
                            if (v = B.exec(l))
                                if (b.isPlainObject(r)) {
                                    l = [y.createElement(v[1])];
                                    b.fn.attr.call(l, r, true)
                                } else 
                                    l = [I.createElement(v[1])];
                            else {
                                v = b.buildFragment([u[1]], [I]);
                                l = (v.cacheable ? v.fragment.cloneNode(true) : v.fragment).childNodes
                            }
                            return b.merge(this,
                            l)
                        } else {
                            if ((v = y.getElementById(u[2])) && v.parentNode) {
                                if (v.id !== u[2])
                                    return f.find(l);
                                    this.length = 1;
                                    this[0] = v
                            }
                            this.context = y;
                            this.selector = l;
                            return this
                        } else if (!r&&!A.test(l)) {
                            this.selector = l;
                            this.context = y;
                            l = y.getElementsByTagName(l);
                            return b.merge(this, l)
                        } else 
                            return !r || r.jquery ? (r || f).find(l) : b(r).find(l);
                        else if (b.isFunction(l))
                    return f.ready(l);
                if (l.selector !== D) {
                    this.selector = l.selector;
                    this.context = l.context
                }
                return b.makeArray(l, this)
            },
            selector: "",
            jquery: "1.4.4",
            length: 0,
            size: function() {
                return this.length
            },
            toArray: function() {
                return ha.call(this, 0)
            },
            get: function(l) {
                return l == null ? this.toArray() : l < 0 ? this.slice(l)[0] : this[l]
            },
            pushStack: function(l, r, u) {
                var v = b();
                b.isArray(l) ? ca.apply(v, l) : b.merge(v, l);
                v.prevObject = this;
                v.context = this.context;
                if (r === "find")
                    v.selector = this.selector + (this.selector ? " " : "") + u;
                else if (r)
                    v.selector = this.selector + "." + r + "(" + u + ")";
                return v
            },
            each: function(l, r) {
                return b.each(this, l, r)
            },
            ready: function(l) {
                b.bindReady();
                if (b.isReady)
                    l.call(y, b);
                else 
                    E && E.push(l);
                return this
            },
            eq: function(l) {
                return l ===
                - 1 ? this.slice(l) : this.slice(l, + l + 1)
            },
            first: function() {
                return this.eq(0)
            },
            last: function() {
                return this.eq( - 1)
            },
            slice: function() {
                return this.pushStack(ha.apply(this, arguments), "slice", ha.call(arguments).join(","))
            },
            map: function(l) {
                return this.pushStack(b.map(this, function(r, u) {
                    return l.call(r, u, r)
                }))
            },
            end: function() {
                return this.prevObject || b(null)
            },
            push: ca,
            sort: [].sort,
            splice: [].splice
        };
        b.fn.init.prototype = b.fn;
        b.extend = b.fn.extend = function() {
            var l, r, u, v, I, J = arguments[0] || {}, G = 1, T = arguments.length, S =
            false;
            if (typeof J === "boolean") {
                S = J;
                J = arguments[1] || {};
                G = 2
            }
            if (typeof J !== "object"&&!b.isFunction(J))
                J = {};
            if (T === G) {
                J = this;
                --G
            }
            for (; G < T; G++)
                if ((l = arguments[G]) != null)
                    for (r in l) {
                        u = J[r];
                        v = l[r];
                        if (J !== v)
                            if (S && v && (b.isPlainObject(v) || (I = b.isArray(v)))) {
                                if (I) {
                                    I = false;
                                    u = u && b.isArray(u) ? u : []
                                } else 
                                    u = u && b.isPlainObject(u) ? u : {};
                                    J[r] = b.extend(S, u, v)
                            } else if (v !== D)
                                J[r] = v
                    }
            return J
        };
        b.extend({
            noConflict: function(l) {
                K.$ = e;
                if (l)
                    K.jQuery = d;
                return b
            },
            isReady: false,
            readyWait: 1,
            ready: function(l) {
                l === true && b.readyWait--;
                if (!b.readyWait || l !== true&&!b.isReady) {
                    if (!y.body)
                        return setTimeout(b.ready, 1);
                    b.isReady = true;
                    if (!(l !== true&&--b.readyWait > 0))
                        if (E) {
                            var r = 0, u = E;
                            for (E = null; l = u[r++];)
                                l.call(y, b);
                                b.fn.trigger && b(y).trigger("ready").unbind("ready")
                        }
                }
            },
            bindReady: function() {
                if (!aa) {
                    aa = true;
                    if (y.readyState === "complete")
                        return setTimeout(b.ready, 1);
                    if (y.addEventListener) {
                        y.addEventListener("DOMContentLoaded", ba, false);
                        K.addEventListener("load", b.ready, false)
                    } else if (y.attachEvent) {
                        y.attachEvent("onreadystatechange", ba);
                        K.attachEvent("onload", b.ready);
                        var l = false;
                        try {
                            l = K.frameElement == null
                        } catch (r) {}
                        y.documentElement.doScroll && l && a()
                    }
                }
            },
            isFunction: function(l) {
                return b.type(l) === "function"
            },
            isArray: Array.isArray || function(l) {
                return b.type(l) === "array"
            },
            isWindow: function(l) {
                return l && typeof l === "object" && "setInterval"in l
            },
            isNaN: function(l) {
                return l == null ||!s.test(l) || isNaN(l)
            },
            type: function(l) {
                return l == null ? String(l) : X[ya.call(l)] || "object"
            },
            isPlainObject: function(l) {
                if (!l || b.type(l) !== "object" || l.nodeType || b.isWindow(l))
                    return false;
                if (l.constructor&&!ka.call(l, "constructor")&&!ka.call(l.constructor.prototype, "isPrototypeOf"))
                    return false;
                for (var r in l);
                return r === D || ka.call(l, r)
            },
            isEmptyObject: function(l) {
                for (var r in l)
                    return false;
                return true
            },
            error: function(l) {
                throw l;
            },
            parseJSON: function(l) {
                if (typeof l !== "string" ||!l)
                    return null;
                l = b.trim(l);
                if (F.test(l.replace(P, "@").replace(z, "]").replace(Q, "")))
                    return K.JSON && K.JSON.parse ? K.JSON.parse(l) : (new Function("return " + l))();
                else 
                    b.error("Invalid JSON: " + l)
            },
            noop: function() {},
            globalEval: function(l) {
                if (l && j.test(l)) {
                    var r = y.getElementsByTagName("head")[0] || y.documentElement, u = y.createElement("script");
                    u.type = "text/javascript";
                    if (b.support.scriptEval)
                        u.appendChild(y.createTextNode(l));
                    else 
                        u.text = l;
                    r.insertBefore(u, r.firstChild);
                    r.removeChild(u)
                }
            },
            nodeName: function(l, r) {
                return l.nodeName && l.nodeName.toUpperCase() === r.toUpperCase()
            },
            each: function(l, r, u) {
                var v, I = 0, J = l.length, G = J === D || b.isFunction(l);
                if (u)
                    if (G)
                        for (v in l) {
                            if (r.apply(l[v], u) === false)
                                break
                        } else 
                            for (; I < J;) {
                                if (r.apply(l[I++],
                                u) === false)
                                    break
                            } else if (G)
                    for (v in l) {
                        if (r.call(l[v], v, l[v]) === false)
                            break
                    } else 
                        for (u = l[0]; I < J && r.call(u, I, u) !== false; u = l[++I]);
                return l
            },
            trim: ea ? function(l) {
                return l == null ? "" : ea.call(l)
            }
            : function(l) {
                return l == null ? "" : l.toString().replace(n, "").replace(q, "")
            },
            makeArray: function(l, r) {
                var u = r || [];
                if (l != null) {
                    var v = b.type(l);
                    l.length == null || v === "string" || v === "function" || v === "regexp" || b.isWindow(l) ? ca.call(u, l) : b.merge(u, l)
                }
                return u
            },
            inArray: function(l, r) {
                if (r.indexOf)
                    return r.indexOf(l);
                for (var u = 0,
                v = r.length; u < v; u++)
                    if (r[u] === l)
                        return u;
                return - 1
            },
            merge: function(l, r) {
                var u = l.length, v = 0;
                if (typeof r.length === "number")
                    for (var I = r.length; v < I; v++)
                        l[u++] = r[v];
                else 
                    for (; r[v] !== D;)
                        l[u++] = r[v++];
                l.length = u;
                return l
            },
            grep: function(l, r, u) {
                var v = [], I;
                u=!!u;
                for (var J = 0, G = l.length; J < G; J++) {
                    I=!!r(l[J], J);
                    u !== I && v.push(l[J])
                }
                return v
            },
            map: function(l, r, u) {
                for (var v = [], I, J = 0, G = l.length; J < G; J++) {
                    I = r(l[J], J, u);
                    if (I != null)
                        v[v.length] = I
                }
                return v.concat.apply([], v)
            },
            guid: 1,
            proxy: function(l, r, u) {
                if (arguments.length ===
                2)
                    if (typeof r === "string") {
                        u = l;
                        l = u[r];
                        r = D
                    } else if (r&&!b.isFunction(r)) {
                        u = r;
                        r = D
                    }
                if (!r && l)
                    r = function() {
                        return l.apply(u || this, arguments)
                    };
                if (l)
                    r.guid = l.guid = l.guid || r.guid || b.guid++;
                return r
            },
            access: function(l, r, u, v, I, J) {
                var G = l.length;
                if (typeof r === "object") {
                    for (var T in r)
                        b.access(l, T, r[T], v, I, u);
                    return l
                }
                if (u !== D) {
                    v=!J && v && b.isFunction(u);
                    for (T = 0; T < G; T++)
                        I(l[T], r, v ? u.call(l[T], T, I(l[T], r)) : u, J);
                    return l
                }
                return G ? I(l[0], r) : D
            },
            now: function() {
                return (new Date).getTime()
            },
            uaMatch: function(l) {
                l = l.toLowerCase();
                l = Y.exec(l) || $.exec(l) || xa.exec(l) || l.indexOf("compatible") < 0 && na.exec(l) || [];
                return {
                    browser: l[1] || "",
                    version: l[2] || "0"
                }
            },
            browser: {}
        });
        b.each("Boolean Number String Function Array Date RegExp Object".split(" "), function(l, r) {
            X["[object " + r + "]"] = r.toLowerCase()
        });
        U = b.uaMatch(U);
        if (U.browser) {
            b.browser[U.browser] = true;
            b.browser.version = U.version
        }
        if (b.browser.webkit)
            b.browser.safari = true;
        if (V)
            b.inArray = function(l, r) {
                return V.call(r, l)
            };
        if (!/\s/.test("\u00a0")) {
            n = /^[\s\xA0]+/;
            q = /[\s\xA0]+$/
        }
        f = b(y);
        if (y.addEventListener)
            ba =
            function() {
                y.removeEventListener("DOMContentLoaded", ba, false);
                b.ready()
            };
        else if (y.attachEvent)
            ba = function() {
                if (y.readyState === "complete") {
                    y.detachEvent("onreadystatechange", ba);
                    b.ready()
                }
            };
        return K.jQuery = K.$ = b
    }();
    (function() {
        c.support = {};
        var a = y.documentElement, b = y.createElement("script"), d = y.createElement("div"), e = "script" + c.now();
        d.style.display = "none";
        d.innerHTML = "   <link/><table></table><a href='/a' style='color:red;float:left;opacity:.55;'>a</a><input type='checkbox'/>";
        var f = d.getElementsByTagName("*"),
        g = d.getElementsByTagName("a")[0], j = y.createElement("select"), n = j.appendChild(y.createElement("option"));
        if (!(!f ||!f.length ||!g)) {
            c.support = {
                leadingWhitespace: d.firstChild.nodeType === 3,
                tbody: !d.getElementsByTagName("tbody").length,
                htmlSerialize: !!d.getElementsByTagName("link").length,
                style: /red/.test(g.getAttribute("style")),
                hrefNormalized: g.getAttribute("href") === "/a",
                opacity: /^0.55$/.test(g.style.opacity),
                cssFloat: !!g.style.cssFloat,
                checkOn: d.getElementsByTagName("input")[0].value === "on",
                optSelected: n.selected,
                deleteExpando: true,
                optDisabled: false,
                checkClone: false,
                scriptEval: false,
                noCloneEvent: true,
                boxModel: null,
                inlineBlockNeedsLayout: false,
                shrinkWrapBlocks: false,
                reliableHiddenOffsets: true
            };
            j.disabled = true;
            c.support.optDisabled=!n.disabled;
            b.type = "text/javascript";
            try {
                b.appendChild(y.createTextNode("window." + e + "=1;"))
            } catch (q) {}
            a.insertBefore(b, a.firstChild);
            if (K[e]) {
                c.support.scriptEval = true;
                delete K[e]
            }
            try {
                delete b.test
            } catch (A) {
                c.support.deleteExpando = false
            }
            a.removeChild(b);
            if (d.attachEvent && d.fireEvent) {
                d.attachEvent("onclick",
                function s() {
                    c.support.noCloneEvent = false;
                    d.detachEvent("onclick", s)
                });
                d.cloneNode(true).fireEvent("onclick")
            }
            d = y.createElement("div");
            d.innerHTML = "<input type='radio' name='radiotest' checked='checked'/>";
            a = y.createDocumentFragment();
            a.appendChild(d.firstChild);
            c.support.checkClone = a.cloneNode(true).cloneNode(true).lastChild.checked;
            c(function() {
                var s = y.createElement("div");
                s.style.width = s.style.paddingLeft = "1px";
                y.body.appendChild(s);
                c.boxModel = c.support.boxModel = s.offsetWidth === 2;
                if ("zoom"in s.style) {
                    s.style.display =
                    "inline";
                    s.style.zoom = 1;
                    c.support.inlineBlockNeedsLayout = s.offsetWidth === 2;
                    s.style.display = "";
                    s.innerHTML = "<div style='width:4px;'></div>";
                    c.support.shrinkWrapBlocks = s.offsetWidth !== 2
                }
                s.innerHTML = "<table><tr><td style='padding:0;display:none'></td><td>t</td></tr></table>";
                var B = s.getElementsByTagName("td");
                c.support.reliableHiddenOffsets = B[0].offsetHeight === 0;
                B[0].style.display = "";
                B[1].style.display = "none";
                c.support.reliableHiddenOffsets = c.support.reliableHiddenOffsets && B[0].offsetHeight === 0;
                s.innerHTML =
                "";
                y.body.removeChild(s).style.display = "none"
            });
            a = function(s) {
                var B = y.createElement("div");
                s = "on" + s;
                var F = s in B;
                if (!F) {
                    B.setAttribute(s, "return;");
                    F = typeof B[s] === "function"
                }
                return F
            };
            c.support.submitBubbles = a("submit");
            c.support.changeBubbles = a("change");
            a = b = d = f = g = null
        }
    })();
    var Xa = {}, tb = /^(?:\{.*\}|\[.*\])$/;
    c.extend({
        cache: {},
        uuid: 0,
        expando: "jQuery" + c.now(),
        noData: {
            embed: true,
            object: "clsid:D27CDB6E-AE6D-11cf-96B8-444553540000",
            applet: true
        },
        data: function(a, b, d) {
            if (c.acceptData(a)) {
                a = a == K ? Xa : a;
                var e =
                a.nodeType, f = e ? a[c.expando]: null, g = c.cache;
                if (!(e&&!f && typeof b === "string" && d === D)) {
                    if (e)
                        f || (a[c.expando] = f=++c.uuid);
                    else 
                        g = a;
                    if (typeof b === "object")
                        if (e)
                            g[f] = c.extend(g[f], b);
                        else 
                            c.extend(g, b);
                    else if (e&&!g[f])
                        g[f] = {};
                    a = e ? g[f] : g;
                    if (d !== D)
                        a[b] = d;
                    return typeof b === "string" ? a[b] : a
                }
            }
        },
        removeData: function(a, b) {
            if (c.acceptData(a)) {
                a = a == K ? Xa : a;
                var d = a.nodeType, e = d ? a[c.expando]: a, f = c.cache, g = d ? f[e]: e;
                if (b) {
                    if (g) {
                        delete g[b];
                        d && c.isEmptyObject(g) && c.removeData(a)
                    }
                } else if (d && c.support.deleteExpando)
                    delete a[c.expando];
                else if (a.removeAttribute)
                    a.removeAttribute(c.expando);
                else if (d)
                    delete f[e];
                else 
                    for (var j in a)
                        delete a[j]
            }
        },
        acceptData: function(a) {
            if (a.nodeName) {
                var b = c.noData[a.nodeName.toLowerCase()];
                if (b)
                    return !(b === true || a.getAttribute("classid") !== b)
            }
            return true
        }
    });
    c.fn.extend({
        data: function(a, b) {
            var d = null;
            if (typeof a === "undefined") {
                if (this.length) {
                    var e = this[0].attributes, f;
                    d = c.data(this[0]);
                    for (var g = 0, j = e.length; g < j; g++) {
                        f = e[g].name;
                        if (f.indexOf("data-") === 0) {
                            f = f.substr(5);
                            Qa(this[0], f, d[f])
                        }
                    }
                }
                return d
            } else if (typeof a ===
            "object")
                return this.each(function() {
                    c.data(this, a)
                });
            var n = a.split(".");
            n[1] = n[1] ? "." + n[1] : "";
            if (b === D) {
                d = this.triggerHandler("getData" + n[1] + "!", [n[0]]);
                if (d === D && this.length) {
                    d = c.data(this[0], a);
                    d = Qa(this[0], a, d)
                }
                return d === D && n[1] ? this.data(n[0]) : d
            } else 
                return this.each(function() {
                    var q = c(this), A = [n[0], b];
                    q.triggerHandler("setData" + n[1] + "!", A);
                    c.data(this, a, b);
                    q.triggerHandler("changeData" + n[1] + "!", A)
                })
        },
        removeData: function(a) {
            return this.each(function() {
                c.removeData(this, a)
            })
        }
    });
    c.extend({
        queue: function(a,
        b, d) {
            if (a) {
                b = (b || "fx") + "queue";
                var e = c.data(a, b);
                if (!d)
                    return e || [];
                if (!e || c.isArray(d))
                    e = c.data(a, b, c.makeArray(d));
                else 
                    e.push(d);
                return e
            }
        },
        dequeue: function(a, b) {
            b = b || "fx";
            var d = c.queue(a, b), e = d.shift();
            if (e === "inprogress")
                e = d.shift();
            if (e) {
                b === "fx" && d.unshift("inprogress");
                e.call(a, function() {
                    c.dequeue(a, b)
                })
            }
        }
    });
    c.fn.extend({
        queue: function(a, b) {
            if (typeof a !== "string") {
                b = a;
                a = "fx"
            }
            if (b === D)
                return c.queue(this[0], a);
            return this.each(function() {
                var d = c.queue(this, a, b);
                a === "fx" && d[0] !== "inprogress" &&
                c.dequeue(this, a)
            })
        },
        dequeue: function(a) {
            return this.each(function() {
                c.dequeue(this, a)
            })
        },
        delay: function(a, b) {
            a = c.fx ? c.fx.speeds[a] || a : a;
            b = b || "fx";
            return this.queue(b, function() {
                var d = this;
                setTimeout(function() {
                    c.dequeue(d, b)
                }, a)
            })
        },
        clearQueue: function(a) {
            return this.queue(a || "fx", [])
        }
    });
    var Ya = /[\n\t]/g, Ja = /\s+/, Cb = /\r/g, Db = /^(?:href|src|style)$/, Eb = /^(?:button|input)$/i, Fb = /^(?:button|input|object|select|textarea)$/i, Gb = /^a(?:rea)?$/i, Za = /^(?:radio|checkbox)$/i;
    c.props = {
        "for": "htmlFor",
        "class": "className",
        readonly: "readOnly",
        maxlength: "maxLength",
        cellspacing: "cellSpacing",
        rowspan: "rowSpan",
        colspan: "colSpan",
        tabindex: "tabIndex",
        usemap: "useMap",
        frameborder: "frameBorder"
    };
    c.fn.extend({
        attr: function(a, b) {
            return c.access(this, a, b, true, c.attr)
        },
        removeAttr: function(a) {
            return this.each(function() {
                c.attr(this, a, "");
                this.nodeType === 1 && this.removeAttribute(a)
            })
        },
        addClass: function(a) {
            if (c.isFunction(a))
                return this.each(function(A) {
                    var s = c(this);
                    s.addClass(a.call(this, A, s.attr("class")))
                });
            if (a && typeof a === "string")
                for (var b =
                (a || "").split(Ja), d = 0, e = this.length; d < e; d++) {
                    var f = this[d];
                    if (f.nodeType === 1)
                        if (f.className) {
                            for (var g = " " + f.className + " ", j = f.className, n = 0, q = b.length; n < q; n++)
                                if (g.indexOf(" " + b[n] + " ") < 0)
                                    j += " " + b[n];
                                    f.className = c.trim(j)
                        } else 
                            f.className = a
                }
            return this
        },
        removeClass: function(a) {
            if (c.isFunction(a))
                return this.each(function(q) {
                    var A = c(this);
                    A.removeClass(a.call(this, q, A.attr("class")))
                });
            if (a && typeof a === "string" || a === D)
                for (var b = (a || "").split(Ja), d = 0, e = this.length; d < e; d++) {
                    var f = this[d];
                    if (f.nodeType ===
                    1 && f.className)
                        if (a) {
                            for (var g = (" " + f.className + " ").replace(Ya, " "), j = 0, n = b.length; j < n; j++)
                                g = g.replace(" " + b[j] + " ", " ");
                                f.className = c.trim(g)
                        } else 
                            f.className = ""
                }
            return this
        },
        toggleClass: function(a, b) {
            var d = typeof a, e = typeof b === "boolean";
            if (c.isFunction(a))
                return this.each(function(f) {
                    var g = c(this);
                    g.toggleClass(a.call(this, f, g.attr("class"), b), b)
                });
            return this.each(function() {
                if (d === "string")
                    for (var f, g = 0, j = c(this), n = b, q = a.split(Ja); f = q[g++];) {
                        n = e ? n : !j.hasClass(f);
                        j[n ? "addClass": "removeClass"](f)
                    } else if (d ===
                "undefined" || d === "boolean") {
                    this.className && c.data(this, "__className__", this.className);
                    this.className = this.className || a === false ? "" : c.data(this, "__className__") || ""
                }
            })
        },
        hasClass: function(a) {
            a = " " + a + " ";
            for (var b = 0, d = this.length; b < d; b++)
                if ((" " + this[b].className + " ").replace(Ya, " ").indexOf(a)>-1)
                    return true;
            return false
        },
        val: function(a) {
            if (!arguments.length) {
                var b = this[0];
                if (b) {
                    if (c.nodeName(b, "option")) {
                        var d = b.attributes.value;
                        return !d || d.specified ? b.value : b.text
                    }
                    if (c.nodeName(b, "select")) {
                        var e =
                        b.selectedIndex;
                        d = [];
                        var f = b.options;
                        b = b.type === "select-one";
                        if (e < 0)
                            return null;
                        var g = b ? e: 0;
                        for (e = b ? e + 1 : f.length; g < e; g++) {
                            var j = f[g];
                            if (j.selected && (c.support.optDisabled?!j.disabled : j.getAttribute("disabled") === null) && (!j.parentNode.disabled ||!c.nodeName(j.parentNode, "optgroup"))
                                ) {
                                a = c(j).val();
                                if (b)
                                    return a;
                                d.push(a)
                            }
                        }
                        return d
                    }
                    if (Za.test(b.type)&&!c.support.checkOn)
                        return b.getAttribute("value") === null ? "on" : b.value;
                    return (b.value || "").replace(Cb, "")
                }
                return D
            }
            var n = c.isFunction(a);
            return this.each(function(q) {
                var A =
                c(this), s = a;
                if (this.nodeType === 1) {
                    if (n)
                        s = a.call(this, q, A.val());
                    if (s == null)
                        s = "";
                    else if (typeof s === "number")
                        s += "";
                    else if (c.isArray(s))
                        s = c.map(s, function(F) {
                            return F == null ? "" : F + ""
                        });
                    if (c.isArray(s) && Za.test(this.type))
                        this.checked = c.inArray(A.val(), s) >= 0;
                    else if (c.nodeName(this, "select")) {
                        var B = c.makeArray(s);
                        c("option", this).each(function() {
                            this.selected = c.inArray(c(this).val(), B) >= 0
                        });
                        if (!B.length)
                            this.selectedIndex =- 1
                    } else 
                        this.value = s
                }
            })
        }
    });
    c.extend({
        attrFn: {
            val: true,
            css: true,
            html: true,
            text: true,
            data: true,
            width: true,
            height: true,
            offset: true
        },
        attr: function(a, b, d, e) {
            if (!a || a.nodeType === 3 || a.nodeType === 8)
                return D;
            if (e && b in c.attrFn)
                return c(a)[b](d);
            e = a.nodeType !== 1 ||!c.isXMLDoc(a);
            var f = d !== D;
            b = e && c.props[b] || b;
            var g = Db.test(b);
            if ((b in a || a[b] !== D) && e&&!g) {
                if (f) {
                    b === "type" && Eb.test(a.nodeName) && a.parentNode && c.error("type property can't be changed");
                    if (d === null)
                        a.nodeType === 1 && a.removeAttribute(b);
                    else 
                        a[b] = d
                }
                if (c.nodeName(a, "form") && a.getAttributeNode(b))
                    return a.getAttributeNode(b).nodeValue;
                if (b === "tabIndex")
                    return (b = a.getAttributeNode("tabIndex")) && b.specified ? b.value : Fb.test(a.nodeName) || Gb.test(a.nodeName) && a.href ? 0 : D;
                return a[b]
            }
            if (!c.support.style && e && b === "style") {
                if (f)
                    a.style.cssText = "" + d;
                return a.style.cssText
            }
            f && a.setAttribute(b, "" + d);
            if (!a.attributes[b] && a.hasAttribute&&!a.hasAttribute(b))
                return D;
            a=!c.support.hrefNormalized && e && g ? a.getAttribute(b, 2) : a.getAttribute(b);
            return a === null ? D : a
        }
    });
    var va = /\.(.*)$/, Ka = /^(?:textarea|input|select)$/i, vb = /\./g, wb = / /g, Hb = /[^\w\s.|`]/g,
    Ib = function(a) {
        return a.replace(Hb, "\\$&")
    }, $a = {
        focusin: 0,
        focusout: 0
    };
    c.event = {
        add: function(a, b, d, e) {
            if (!(a.nodeType === 3 || a.nodeType === 8)) {
                if (c.isWindow(a) && a !== K&&!a.frameElement)
                    a = K;
                if (d === false)
                    d = sa;
                else if (!d)
                    return;
                var f, g;
                if (d.handler) {
                    f = d;
                    d = f.handler
                }
                if (!d.guid)
                    d.guid = c.guid++;
                if (g = c.data(a)) {
                    var j = a.nodeType ? "events": "__events__", n = g[j], q = g.handle;
                    if (typeof n === "function") {
                        q = n.handle;
                        n = n.events
                    } else if (!n) {
                        a.nodeType || (g[j] = g = function() {});
                        g.events = n = {}
                    }
                    if (!q)
                        g.handle = q = function() {
                            return typeof c !==
                            "undefined"&&!c.event.triggered ? c.event.handle.apply(q.elem, arguments) : D
                        };
                    q.elem = a;
                    b = b.split(" ");
                    for (var A = 0, s; j = b[A++];) {
                        g = f ? c.extend({}, f) : {
                            handler: d,
                            data: e
                        };
                        if (j.indexOf(".")>-1) {
                            s = j.split(".");
                            j = s.shift();
                            g.namespace = s.slice(0).sort().join(".")
                        } else {
                            s = [];
                            g.namespace = ""
                        }
                        g.type = j;
                        if (!g.guid)
                            g.guid = d.guid;
                        var B = n[j], F = c.event.special[j] || {};
                        if (!B) {
                            B = n[j] = [];
                            if (!F.setup || F.setup.call(a, e, s, q) === false)
                                if (a.addEventListener)
                                    a.addEventListener(j, q, false);
                                else 
                                    a.attachEvent && a.attachEvent("on" + j, q)
                                }
                        if (F.add) {
                            F.add.call(a,
                            g);
                            if (!g.handler.guid)
                                g.handler.guid = d.guid
                        }
                        B.push(g);
                        c.event.global[j] = true
                    }
                    a = null
                }
            }
        },
        global: {},
        remove: function(a, b, d, e) {
            if (!(a.nodeType === 3 || a.nodeType === 8)) {
                if (d === false)
                    d = sa;
                var f, g, j = 0, n, q, A, s, B, F, P = a.nodeType ? "events": "__events__", z = c.data(a), Q = z && z[P];
                if (z && Q) {
                    if (typeof Q === "function") {
                        z = Q;
                        Q = Q.events
                    }
                    if (b && b.type) {
                        d = b.handler;
                        b = b.type
                    }
                    if (!b || typeof b === "string" && b.charAt(0) === ".") {
                        b = b || "";
                        for (f in Q)
                            c.event.remove(a, f + b)
                        } else {
                        for (b = b.split(" "); f = b[j++];) {
                            s = f;
                            n = f.indexOf(".") < 0;
                            q = [];
                            if (!n) {
                                q =
                                f.split(".");
                                f = q.shift();
                                A = RegExp("(^|\\.)" + c.map(q.slice(0).sort(), Ib).join("\\.(?:.*\\.)?") + "(\\.|$)")
                            }
                            if (B = Q[f])
                                if (d) {
                                    s = c.event.special[f] || {};
                                    for (g = e || 0; g < B.length; g++) {
                                        F = B[g];
                                        if (d.guid === F.guid) {
                                            if (n || A.test(F.namespace)) {
                                                e == null && B.splice(g--, 1);
                                                s.remove && s.remove.call(a, F)
                                            }
                                            if (e != null)
                                                break
                                        }
                                    }
                                    if (B.length === 0 || e != null && B.length === 1) {
                                        if (!s.teardown || s.teardown.call(a, q) === false)
                                            c.removeEvent(a, f, z.handle);
                                            delete Q[f]
                                    }
                                } else 
                                    for (g = 0; g < B.length; g++) {
                                        F = B[g];
                                        if (n || A.test(F.namespace)) {
                                            c.event.remove(a,
                                            s, F.handler, g);
                                            B.splice(g--, 1)
                                        }
                                    }
                                }
                        if (c.isEmptyObject(Q)) {
                            if (b = z.handle)
                                b.elem = null;
                            delete z.events;
                            delete z.handle;
                            if (typeof z === "function")
                                c.removeData(a, P);
                            else 
                                c.isEmptyObject(z) && c.removeData(a)
                            }
                    }
                }
            }
        },
        trigger: function(a, b, d, e) {
            var f = a.type || a;
            if (!e) {
                a = typeof a === "object" ? a[c.expando] ? a : c.extend(c.Event(f), a) : c.Event(f);
                if (f.indexOf("!") >= 0) {
                    a.type = f = f.slice(0, - 1);
                    a.exclusive = true
                }
                if (!d) {
                    a.stopPropagation();
                    c.event.global[f] && c.each(c.cache, function() {
                        this.events && this.events[f] && c.event.trigger(a,
                        b, this.handle.elem)
                    })
                }
                if (!d || d.nodeType === 3 || d.nodeType === 8)
                    return D;
                a.result = D;
                a.target = d;
                b = c.makeArray(b);
                b.unshift(a)
            }
            a.currentTarget = d;
            (e = d.nodeType ? c.data(d, "handle") : (c.data(d, "__events__") || {}).handle) && e.apply(d, b);
            e = d.parentNode || d.ownerDocument;
            try {
                if (!(d && d.nodeName && c.noData[d.nodeName.toLowerCase()]))
                    if (d["on" + f] && d["on" + f].apply(d, b) === false) {
                        a.result = false;
                        a.preventDefault()
                    }
            } catch (g) {}
            if (!a.isPropagationStopped() && e)
                c.event.trigger(a, b, e, true);
            else if (!a.isDefaultPrevented()) {
                var j;
                e = a.target;
                var n = f.replace(va, ""), q = c.nodeName(e, "a") && n === "click", A = c.event.special[n] || {};
                if ((!A._default || A._default.call(d, a) === false)&&!q&&!(e && e.nodeName && c.noData[e.nodeName.toLowerCase()])) {
                    try {
                        if (e[n]) {
                            if (j = e["on" + n])
                                e["on" + n] = null;
                            c.event.triggered = true;
                            e[n]()
                        }
                    } catch (s) {}
                    if (j)
                        e["on" + n] = j;
                    c.event.triggered = false
                }
            }
        },
        handle: function(a) {
            var b, d, e, f;
            d = [];
            var g = c.makeArray(arguments);
            a = g[0] = c.event.fix(a || K.event);
            a.currentTarget = this;
            b = a.type.indexOf(".") < 0&&!a.exclusive;
            if (!b) {
                e = a.type.split(".");
                a.type = e.shift();
                d = e.slice(0).sort();
                e = RegExp("(^|\\.)" + d.join("\\.(?:.*\\.)?") + "(\\.|$)")
            }
            a.namespace = a.namespace || d.join(".");
            f = c.data(this, this.nodeType ? "events" : "__events__");
            if (typeof f === "function")
                f = f.events;
            d = (f || {})[a.type];
            if (f && d) {
                d = d.slice(0);
                f = 0;
                for (var j = d.length; f < j; f++) {
                    var n = d[f];
                    if (b || e.test(n.namespace)) {
                        a.handler = n.handler;
                        a.data = n.data;
                        a.handleObj = n;
                        n = n.handler.apply(this, g);
                        if (n !== D) {
                            a.result = n;
                            if (n === false) {
                                a.preventDefault();
                                a.stopPropagation()
                            }
                        }
                        if (a.isImmediatePropagationStopped())
                            break
                    }
                }
            }
            return a.result
        },
        props: "altKey attrChange attrName bubbles button cancelable charCode clientX clientY ctrlKey currentTarget data detail eventPhase fromElement handler keyCode layerX layerY metaKey newValue offsetX offsetY pageX pageY prevValue relatedNode relatedTarget screenX screenY shiftKey srcElement target toElement view wheelDelta which".split(" "),
        fix: function(a) {
            if (a[c.expando])
                return a;
            var b = a;
            a = c.Event(b);
            for (var d = this.props.length, e; d;) {
                e = this.props[--d];
                a[e] = b[e]
            }
            if (!a.target)
                a.target = a.srcElement || y;
            if (a.target.nodeType === 3)
                a.target = a.target.parentNode;
            if (!a.relatedTarget && a.fromElement)
                a.relatedTarget = a.fromElement === a.target ? a.toElement : a.fromElement;
            if (a.pageX == null && a.clientX != null) {
                b = y.documentElement;
                d = y.body;
                a.pageX = a.clientX + (b && b.scrollLeft || d && d.scrollLeft || 0) - (b && b.clientLeft || d && d.clientLeft || 0);
                a.pageY = a.clientY + (b && b.scrollTop || d && d.scrollTop || 0) - (b && b.clientTop || d && d.clientTop || 0)
            }
            if (a.which == null && (a.charCode != null || a.keyCode != null))
                a.which = a.charCode != null ? a.charCode : a.keyCode;
            if (!a.metaKey && a.ctrlKey)
                a.metaKey = a.ctrlKey;
            if (!a.which && a.button !== D)
                a.which = a.button & 1 ? 1 : a.button & 2 ? 3 : a.button & 4 ? 2 : 0;
            return a
        },
        guid: 1E8,
        proxy: c.proxy,
        special: {
            ready: {
                setup: c.bindReady,
                teardown: c.noop
            },
            live: {
                add: function(a) {
                    c.event.add(this, wa(a.origType, a.selector), c.extend({}, a, {
                        handler: ub,
                        guid: a.handler.guid
                    }))
                },
                remove: function(a) {
                    c.event.remove(this, wa(a.origType, a.selector), a)
                }
            },
            beforeunload: {
                setup: function(a, b, d) {
                    if (c.isWindow(this))
                        this.onbeforeunload = d
                },
                teardown: function(a, b) {
                    if (this.onbeforeunload ===
                    b)
                        this.onbeforeunload = null
                }
            }
        }
    };
    c.removeEvent = y.removeEventListener ? function(a, b, d) {
        a.removeEventListener && a.removeEventListener(b, d, false)
    } : function(a, b, d) {
        a.detachEvent && a.detachEvent("on" + b, d)
    };
    c.Event = function(a) {
        if (!this.preventDefault)
            return new c.Event(a);
        if (a && a.type) {
            this.originalEvent = a;
            this.type = a.type
        } else 
            this.type = a;
        this.timeStamp = c.now();
        this[c.expando] = true
    };
    c.Event.prototype = {
        preventDefault: function() {
            this.isDefaultPrevented = Fa;
            var a = this.originalEvent;
            if (a)
                if (a.preventDefault)
                    a.preventDefault();
                else 
                    a.returnValue = false
        },
        stopPropagation: function() {
            this.isPropagationStopped = Fa;
            var a = this.originalEvent;
            if (a) {
                a.stopPropagation && a.stopPropagation();
                a.cancelBubble = true
            }
        },
        stopImmediatePropagation: function() {
            this.isImmediatePropagationStopped = Fa;
            this.stopPropagation()
        },
        isDefaultPrevented: sa,
        isPropagationStopped: sa,
        isImmediatePropagationStopped: sa
    };
    var ab = function(a) {
        var b = a.relatedTarget;
        try {
            for (; b && b !== this;)
                b = b.parentNode;
            if (b !== this) {
                a.type = a.data;
                c.event.handle.apply(this, arguments)
            }
        } catch (d) {}
    },
    bb = function(a) {
        a.type = a.data;
        c.event.handle.apply(this, arguments)
    };
    c.each({
        mouseenter: "mouseover",
        mouseleave: "mouseout"
    }, function(a, b) {
        c.event.special[a] = {
            setup: function(d) {
                c.event.add(this, b, d && d.selector ? bb : ab, a)
            },
            teardown: function(d) {
                c.event.remove(this, b, d && d.selector ? bb : ab)
            }
        }
    });
    if (!c.support.submitBubbles)
        c.event.special.submit = {
            setup: function() {
                if (this.nodeName.toLowerCase() !== "form") {
                    c.event.add(this, "click.specialSubmit", function(a) {
                        var b = a.target, d = b.type;
                        if ((d === "submit" || d === "image") &&
                        c(b).closest("form").length) {
                            a.liveFired = D;
                            return Ra("submit", this, arguments)
                        }
                    });
                    c.event.add(this, "keypress.specialSubmit", function(a) {
                        var b = a.target, d = b.type;
                        if ((d === "text" || d === "password") && c(b).closest("form").length && a.keyCode === 13) {
                            a.liveFired = D;
                            return Ra("submit", this, arguments)
                        }
                    })
                } else 
                    return false
                },
                teardown: function() {
                    c.event.remove(this, ".specialSubmit")
                }
            };
    if (!c.support.changeBubbles) {
        var ta, cb = function(a) {
            var b = a.type, d = a.value;
            if (b === "radio" || b === "checkbox")
                d = a.checked;
            else if (b === "select-multiple")
                d =
                a.selectedIndex>-1 ? c.map(a.options, function(e) {
                    return e.selected
                }).join("-") : "";
            else if (a.nodeName.toLowerCase() === "select")
                d = a.selectedIndex;
            return d
        }, za = function(a, b) {
            var d = a.target, e, f;
            if (!(!Ka.test(d.nodeName) || d.readOnly)) {
                e = c.data(d, "_change_data");
                f = cb(d);
                if (a.type !== "focusout" || d.type !== "radio")
                    c.data(d, "_change_data", f);
                if (!(e === D || f === e))
                    if (e != null || f) {
                        a.type = "change";
                        a.liveFired = D;
                        return c.event.trigger(a, b, d)
                    }
            }
        };
        c.event.special.change = {
            filters: {
                focusout: za,
                beforedeactivate: za,
                click: function(a) {
                    var b =
                    a.target, d = b.type;
                    if (d === "radio" || d === "checkbox" || b.nodeName.toLowerCase() === "select")
                        return za.call(this, a)
                },
                keydown: function(a) {
                    var b = a.target, d = b.type;
                    if (a.keyCode === 13 && b.nodeName.toLowerCase() !== "textarea" || a.keyCode === 32 && (d === "checkbox" || d === "radio") || d === "select-multiple")
                        return za.call(this, a)
                },
                beforeactivate: function(a) {
                    a = a.target;
                    c.data(a, "_change_data", cb(a))
                }
            },
            setup: function() {
                if (this.type === "file")
                    return false;
                for (var a in ta)
                    c.event.add(this, a + ".specialChange", ta[a]);
                return Ka.test(this.nodeName)
            },
            teardown: function() {
                c.event.remove(this, ".specialChange");
                return Ka.test(this.nodeName)
            }
        };
        ta = c.event.special.change.filters;
        ta.focus = ta.beforeactivate
    }
    y.addEventListener && c.each({
        focus: "focusin",
        blur: "focusout"
    }, function(a, b) {
        function d(e) {
            e = c.event.fix(e);
            e.type = b;
            return c.event.trigger(e, null, e.target)
        }
        c.event.special[b] = {
            setup: function() {
                $a[b]++===0 && y.addEventListener(a, d, true)
            },
            teardown: function() {
                --$a[b] === 0 && y.removeEventListener(a, d, true)
            }
        }
    });
    c.each(["bind", "one"], function(a, b) {
        c.fn[b] = function(d,
        e, f) {
            if (typeof d === "object") {
                for (var g in d)
                    this[b](g, e, d[g], f);
                return this
            }
            if (c.isFunction(e) || e === false) {
                f = e;
                e = D
            }
            var j = b === "one" ? c.proxy(f, function(q) {
                c(this).unbind(q, j);
                return f.apply(this, arguments)
            }): f;
            if (d === "unload" && b !== "one")
                this.one(d, e, f);
            else {
                g = 0;
                for (var n = this.length; g < n; g++)
                    c.event.add(this[g], d, j, e)
            }
            return this
        }
    });
    c.fn.extend({
        unbind: function(a, b) {
            if (typeof a === "object"&&!a.preventDefault)
                for (var d in a)
                    this.unbind(d, a[d]);
            else {
                d = 0;
                for (var e = this.length; d < e; d++)
                    c.event.remove(this[d],
                    a, b)
            }
            return this
        },
        delegate: function(a, b, d, e) {
            return this.live(b, d, e, a)
        },
        undelegate: function(a, b, d) {
            return arguments.length === 0 ? this.unbind("live") : this.die(b, null, d, a)
        },
        trigger: function(a, b) {
            return this.each(function() {
                c.event.trigger(a, b, this)
            })
        },
        triggerHandler: function(a, b) {
            if (this[0]) {
                var d = c.Event(a);
                d.preventDefault();
                d.stopPropagation();
                c.event.trigger(d, b, this[0]);
                return d.result
            }
        },
        toggle: function(a) {
            for (var b = arguments, d = 1; d < b.length;)
                c.proxy(a, b[d++]);
            return this.click(c.proxy(a, function(e) {
                var f =
                (c.data(this, "lastToggle" + a.guid) || 0)%d;
                c.data(this, "lastToggle" + a.guid, f + 1);
                e.preventDefault();
                return b[f].apply(this, arguments) || false
            }))
        },
        hover: function(a, b) {
            return this.mouseenter(a).mouseleave(b || a)
        }
    });
    var db = {
        focus: "focusin",
        blur: "focusout",
        mouseenter: "mouseover",
        mouseleave: "mouseout"
    };
    c.each(["live", "die"], function(a, b) {
        c.fn[b] = function(d, e, f, g) {
            var j, n = 0, q, A, s = g || this.selector;
            g = g ? this : c(this.context);
            if (typeof d === "object"&&!d.preventDefault) {
                for (j in d)
                    g[b](j, e, d[j], s);
                return this
            }
            if (c.isFunction(e)) {
                f =
                e;
                e = D
            }
            for (d = (d || "").split(" "); (j = d[n++]) != null;) {
                q = va.exec(j);
                A = "";
                if (q) {
                    A = q[0];
                    j = j.replace(va, "")
                }
                if (j === "hover")
                    d.push("mouseenter" + A, "mouseleave" + A);
                else {
                    q = j;
                    if (j === "focus" || j === "blur") {
                        d.push(db[j] + A);
                        j += A
                    } else 
                        j = (db[j] || j) + A;
                    if (b === "live") {
                        A = 0;
                        for (var B = g.length; A < B; A++)
                            c.event.add(g[A], "live." + wa(j, s), {
                                data: e,
                                selector: s,
                                handler: f,
                                origType: j,
                                origHandler: f,
                                preType: q
                            })
                        } else 
                            g.unbind("live." + wa(j, s), f)
                        }
            }
            return this
        }
    });
    c.each("blur focus focusin focusout load resize scroll unload click dblclick mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave change select submit keydown keypress keyup error".split(" "),
    function(a, b) {
        c.fn[b] = function(d, e) {
            if (e == null) {
                e = d;
                d = null
            }
            return arguments.length > 0 ? this.bind(b, d, e) : this.trigger(b)
        };
        if (c.attrFn)
            c.attrFn[b] = true
    });
    K.attachEvent&&!K.addEventListener && c(K).bind("unload", function() {
        for (var a in c.cache)
            if (c.cache[a].handle)
                try {
                    c.event.remove(c.cache[a].handle.elem)
        } catch (b) {}
    });
    (function(a, b) {
        function d(h) {
            return Jb.test(h + "")
        }
        function e() {
            var h, k = [];
            return h = function(i, m) {
                if (k.push(i += " ") > E.cacheLength)
                    delete h[k.shift()];
                return h[i] = m
            }
        }
        function f(h) {
            h[G] = true;
            return h
        }
        function g(h) {
            var k = V.createElement("div");
            try {
                return h(k)
            } catch (i) {
                return false
            } finally {}
        }
        function j(h, k, i, m) {
            var o, t, p, w, x;
            if ((k ? k.ownerDocument || k : T) !== V)
                ea(k);
            k = k || V;
            i = i || [];
            if (!h || typeof h !== "string")
                return i;
            if ((w = k.nodeType) !== 1 && w !== 9)
                return [];
            if (!l&&!m) {
                if (o = Kb.exec(h))
                    if (p = o[1])
                        if (w === 9)
                            if ((t = k.getElementById(p)) && t.parentNode) {
                                if (t.id === p) {
                                    i.push(t);
                                    return i
                                }
                            } else 
                                return i;
                        else {
                            if (k.ownerDocument && (t = k.ownerDocument.getElementById(p)) && I(k, t) && t.id === p) {
                                i.push(t);
                                return i
                            }
                        } else if (o[2]) {
                            oa.apply(i,
                            pa.call(k.getElementsByTagName(h), 0));
                            return i
                        } else if ((p = o[3]) && S.getByClassName && k.getElementsByClassName) {
                            oa.apply(i, pa.call(k.getElementsByClassName(p), 0));
                            return i
                        }
                if (S.qsa&&!r.test(h)) {
                    o = true;
                    t = G;
                    p = k;
                    x = w === 9 && h;
                    if (w === 1 && k.nodeName.toLowerCase() !== "object") {
                        w = B(h);
                        if (o = k.getAttribute("id"))
                            t = o.replace(Lb, "\\$&");
                        else 
                            k.setAttribute("id", t);
                        t = "[id='" + t + "'] ";
                        for (p = w.length; p--;)
                            w[p] = t + F(w[p]);
                        p = La.test(h) && k.parentNode || k;
                        x = w.join(",")
                    }
                    if (x)
                        try {
                            oa.apply(i, pa.call(p.querySelectorAll(x), 0));
                            return i
                    } catch (L) {} finally {
                        o ||
                        k.removeAttribute("id")
                    }
                }
            }
            var C;
            a: {
                h = h.replace(Aa, "$1");
                k = k;
                i = i;
                m = m;
                t = B(h);
                if (!m)
                    if (t.length === 1) {
                        o = t[0] = t[0].slice(0);
                        if (o.length > 2 && (C = o[0]).type === "ID" && k.nodeType === 9&&!l && E.relative[o[1].type]) {
                            k = E.find.ID(C.matches[0].replace(ia, ja), k)[0];
                            if (!k) {
                                C = i;
                                break a
                            }
                            h = h.slice(o.shift().value.length)
                        }
                        for (w = Ba.needsContext.test(h) ? 0 : o.length; w--;) {
                            C = o[w];
                            if (E.relative[p = C.type])
                                break;
                                if (p = E.find[p])
                                    if (m = p(C.matches[0].replace(ia, ja), La.test(o[0].type) && k.parentNode || k)) {
                                        o.splice(w, 1);
                                        h = m.length && F(o);
                                        if (!h) {
                                            oa.apply(i, pa.call(m, 0));
                                            C = i;
                                            break a
                                        }
                                        break
                                    }
                        }
                    }
                ka(h, t)(m, k, l, i, La.test(h));
                C = i
            }
            return C
        }
        function n(h, k) {
            var i = k && h;
            if (i)
                for (; i = i.nextSibling;)
                    if (i === k)
                        return - 1;
            return h ? 1 : - 1
        }
        function q(h) {
            return function(k) {
                return k.nodeName.toLowerCase() === "input" && k.type === h
            }
        }
        function A(h) {
            return function(k) {
                var i = k.nodeName.toLowerCase();
                return (i === "input" || i === "button") && k.type === h
            }
        }
        function s(h) {
            return f(function(k) {
                k =+ k;
                return f(function(i, m) {
                    for (var o, t = h([], i.length, k), p = t.length; p--;)
                        if (i[o = t[p]])
                            i[o] =
                            !(m[o] = i[o])
                    })
            })
        }
        function B(h, k) {
            var i, m, o, t, p, w, x;
            if (p = eb[h + " "])
                return k ? 0 : p.slice(0);
            p = h;
            w = [];
            for (x = E.preFilter; p;) {
                if (!i || (m = Mb.exec(p))) {
                    if (m)
                        p = p.slice(m[0].length) || p;
                    w.push(o = [])
                }
                i = false;
                if (m = Nb.exec(p)) {
                    i = m.shift();
                    o.push({
                        value: i,
                        type: m[0].replace(Aa, " ")
                    });
                    p = p.slice(i.length)
                }
                for (t in E.filter)
                    if ((m = Ba[t].exec(p)) && (!x[t] || (m = x[t](m)))) {
                        i = m.shift();
                        o.push({
                            value: i,
                            type: t,
                            matches: m
                        });
                        p = p.slice(i.length)
                    }
                if (!i)
                    break
            }
            return k ? p.length : p ? j.error(h) : eb(h, w).slice(0)
        }
        function F(h) {
            for (var k = 0,
            i = h.length, m = ""; k < i; k++)
                m += h[k].value;
            return m
        }
        function P(h, k, i) {
            var m = k.dir, o = i && m === "parentNode", t = Ob++;
            return k.first ? function(p, w, x) {
                for (; p = p[m];)
                    if (p.nodeType === 1 || o)
                        return h(p, w, x)
            } : function(p, w, x) {
                var L, C, M, H = fa + " " + t;
                if (x)
                    for (; p = p[m];) {
                        if (p.nodeType === 1 || o)
                            if (h(p, w, x))
                                return true
                    } else 
                        for (; p = p[m];)
                            if (p.nodeType === 1 || o) {
                                M = p[G] || (p[G] = {});
                                if ((C = M[m]) && C[0] === H) {
                                    if ((L = C[1]) === true || L === aa)
                                        return L === true
                                } else {
                                    C = M[m] = [H];
                                    C[1] = h(p, w, x) || aa;
                                    if (C[1] === true)
                                        return true
                                }
                            }
            }
        }
        function z(h) {
            return h.length >
            1 ? function(k, i, m) {
                for (var o = h.length; o--;)
                    if (!h[o](k, i, m))
                        return false;
                return true
            } : h[0]
        }
        function Q(h, k, i, m, o) {
            for (var t, p = [], w = 0, x = h.length, L = k != null; w < x; w++)
                if (t = h[w])
                    if (!i || i(t, m, o)) {
                        p.push(t);
                        L && k.push(w)
                    }
            return p
        }
        function Y(h, k, i, m, o, t) {
            if (m&&!m[G])
                m = Y(m);
            if (o&&!o[G])
                o = Y(o, t);
            return f(function(p, w, x, L) {
                var C, M, H = [], R = [], W = w.length, O;
                if (!(O = p)) {
                    O = k || "*";
                    for (var N = x.nodeType ? [x] : x, da = [], qa = 0, Ma = N.length; qa < Ma; qa++)
                        j(O, N[qa], da);
                    O = da
                }
                O = O;
                O = h && (p ||!k) ? Q(O, H, h, x, L) : O;
                N = i ? o || (p ? h : W || m) ? [] : w : O;
                i && i(O,
                N, x, L);
                if (m) {
                    C = Q(N, R);
                    m(C, [], x, L);
                    for (x = C.length; x--;)
                        if (M = C[x])
                            N[R[x]]=!(O[R[x]] = M)
                }
                if (p) {
                    if (o || h) {
                        if (o) {
                            C = [];
                            for (x = N.length; x--;)
                                if (M = N[x])
                                    C.push(O[x] = M);
                            o(null, N = [], C, L)
                        }
                        for (x = N.length; x--;)
                            if ((M = N[x]) && (C = o ? Na.call(p, M) : H[x])>-1)
                                p[C]=!(w[C] = M)
                            }
                } else {
                    N = Q(N === w ? N.splice(W, N.length) : N);
                    o ? o(null, w, N, L) : oa.apply(w, N)
                }
            })
        }
        function $(h) {
            var k, i, m, o = h.length, t = E.relative[h[0].type];
            i = t || E.relative[" "];
            for (var p = t ? 1 : 0, w = P(function(C) {
                return C === k
            }, i, true), x = P(function(C) {
                return Na.call(k, C)>-1
            }, i, true),
            L = [function(C, M, H) {
                return !t && (H || M !== ha) || ((k = M).nodeType ? w(C, M, H) : x(C, M, H))
            }
            ]; p < o; p++)
                if (i = E.relative[h[p].type])
                    L = [P(z(L), i)];
                else {
                    i = E.filter[h[p].type].apply(null, h[p].matches);
                    if (i[G]) {
                        for (m=++p; m < o; m++)
                            if (E.relative[h[m].type])
                                break;
                                return Y(p > 1 && z(L), p > 1 && F(h.slice(0, p - 1)).replace(Aa, "$1"), i, p < m && $(h.slice(p, m)), m < o && $(h = h.slice(m)), m < o && F(h))
                            }
                            L.push(i)
                }
            return z(L)
        }
        function xa(h, k) {
            var i = 0, m = k.length > 0, o = h.length > 0, t = function(p, w, x, L, C) {
                var M, H, R = [], W = 0, O = "0", N = p && [], da = C != null, qa = ha, Ma = p ||
                o && E.find.TAG("*", C && w.parentNode || w), fb = fa += qa == null ? 1: Math.random() || 0.1;
                if (da) {
                    ha = w !== V && w;
                    aa = i
                }
                for (; (C = Ma[O]) != null; O++) {
                    if (o && C) {
                        for (M = 0; H = h[M++];)
                            if (H(C, w, x)) {
                                L.push(C);
                                break
                            }
                        if (da) {
                            fa = fb;
                            aa=++i
                        }
                    }
                    if (m) {
                        if (C=!H && C)
                            W--;
                        p && N.push(C)
                    }
                }
                W += O;
                if (m && O !== W) {
                    for (M = 0; H = k[M++];)
                        H(N, R, w, x);
                    if (p) {
                        if (W > 0)
                            for (; O--;)
                                N[O] || R[O] || (R[O] = Pb.call(L));
                        R = Q(R)
                    }
                    oa.apply(L, R);
                    da&&!p && R.length > 0 && W + k.length > 1 && j.uniqueSort(L)
                }
                if (da) {
                    fa = fb;
                    ha = qa
                }
                return N
            };
            return m ? f(t) : t
        }
        function na() {}
        var U, aa, E, ba, ya, ka, ca, ha, ea, V, X,
        l, r, u, v, I, J, G = "sizzle" +- new Date, T = a.document, S = {}, fa = 0, Ob = 0, gb = e(), eb = e(), hb = e(), ga = typeof b, la = [], Pb = la.pop, oa = la.push, pa = la.slice, Na = la.indexOf || function(h) {
            for (var k = 0, i = this.length; k < i; k++)
                if (this[k] === h)
                    return k;
            return - 1
        };
        la = "(?:\\\\.|[\\w-]|[^\\x00-\\xa0])+".replace("w", "w#");
        var ib = "\\[[\\x20\\t\\r\\n\\f]*((?:\\\\.|[\\w-]|[^\\x00-\\xa0])+)[\\x20\\t\\r\\n\\f]*(?:([*^$|!~]?=)[\\x20\\t\\r\\n\\f]*(?:(['\"])((?:\\\\.|[^\\\\])*?)\\3|(" + la + ")|)|)[\\x20\\t\\r\\n\\f]*\\]", Oa = ":((?:\\\\.|[\\w-]|[^\\x00-\\xa0])+)(?:\\(((['\"])((?:\\\\.|[^\\\\])*?)\\3|((?:\\\\.|[^\\\\()[\\]]|" +
        ib.replace(3, 8) + ")*)|.*)\\)|)", Aa = RegExp("^[\\x20\\t\\r\\n\\f]+|((?:^|[^\\\\])(?:\\\\.)*)[\\x20\\t\\r\\n\\f]+$", "g"), Mb = /^[\x20\t\r\n\f]*,[\x20\t\r\n\f]*/, Nb = /^[\x20\t\r\n\f]*([\x20\t\r\n\f>+~])[\x20\t\r\n\f]*/, Qb = RegExp(Oa), Rb = RegExp("^" + la + "$"), Ba = {
            ID: /^#((?:\\.|[\w-]|[^\x00-\xa0])+)/,
            CLASS: /^\.((?:\\.|[\w-]|[^\x00-\xa0])+)/,
            NAME: /^\[name=['"]?((?:\\.|[\w-]|[^\x00-\xa0])+)['"]?\]/,
            TAG: RegExp("^(" + "(?:\\\\.|[\\w-]|[^\\x00-\\xa0])+".replace("w", "w*") + ")"),
            ATTR: RegExp("^" + ib),
            PSEUDO: RegExp("^" + Oa),
            CHILD: RegExp("^:(only|first|last|nth|nth-last)-(child|of-type)(?:\\([\\x20\\t\\r\\n\\f]*(even|odd|(([+-]|)(\\d*)n|)[\\x20\\t\\r\\n\\f]*(?:([+-]|)[\\x20\\t\\r\\n\\f]*(\\d+)|))[\\x20\\t\\r\\n\\f]*\\)|)", "i"),
            needsContext: RegExp("^[\\x20\\t\\r\\n\\f]*[>+~]|:(even|odd|eq|gt|lt|nth|first|last)(?:\\([\\x20\\t\\r\\n\\f]*((?:-\\d)?\\d*)[\\x20\\t\\r\\n\\f]*\\)|)(?=[^-]|$)", "i")
        }, La = /[\x20\t\r\n\f]*[+~]/, Jb = /^[^{]+\{\s*\[native code/, Kb = /^(?:#([\w-]+)|(\w+)|\.([\w-]+))$/, Sb = /^(?:input|select|textarea|button)$/i,
        Tb = /^h\d$/i, Lb = /'|\\/g, Ub = /\=[\x20\t\r\n\f]*([^'"\]]*)[\x20\t\r\n\f]*\]/g, ia = /\\([\da-fA-F]{1,6}[\x20\t\r\n\f]?|.)/g, ja = function(h, k) {
            var i = "0x" + k - 65536;
            return i !== i ? k : i < 0 ? String.fromCharCode(i + 65536) : String.fromCharCode(i>>10 | 55296, i & 1023 | 56320)
        };
        try {
            pa.call(T.documentElement.childNodes, 0)
        } catch (uc) {
            pa = function(h) {
                for (var k, i = []; k = this[h++];)
                    i.push(k);
                return i
            }
        }
        ya = j.isXML = function(h) {
            return (h = h && (h.ownerDocument || h).documentElement) ? h.nodeName !== "HTML" : false
        };
        ea = j.setDocument = function(h) {
            var k = h ?
            h.ownerDocument || h: T;
            if (k === V || k.nodeType !== 9 ||!k.documentElement)
                return V;
            V = k;
            X = k.documentElement;
            l = ya(k);
            S.tagNameNoComments = g(function(i) {
                i.appendChild(k.createComment(""));
                return !i.getElementsByTagName("*").length
            });
            S.attributes = g(function(i) {
                i.innerHTML = "<select></select>";
                i = typeof i.lastChild.getAttribute("multiple");
                return i !== "boolean" && i !== "string"
            });
            S.getByClassName = g(function(i) {
                i.innerHTML = "<div class='hidden e'></div><div class='hidden'></div>";
                if (!i.getElementsByClassName ||!i.getElementsByClassName("e").length)
                    return false;
                i.lastChild.className = "e";
                return i.getElementsByClassName("e").length === 2
            });
            S.getByName = g(function(i) {
                i.id = G + 0;
                i.innerHTML = "<a name='" + G + "'></a><div name='" + G + "'></div>";
                X.insertBefore(i, X.firstChild);
                var m = k.getElementsByName && k.getElementsByName(G).length === 2 + k.getElementsByName(G + 0).length;
                S.getIdNotName=!k.getElementById(G);
                X.removeChild(i);
                return m
            });
            E.attrHandle = g(function(i) {
                i.innerHTML = "<a href='#'></a>";
                return i.firstChild && typeof i.firstChild.getAttribute !== ga && i.firstChild.getAttribute("href") ===
                "#"
            }) ? {} : {
                href: function(i) {
                    return i.getAttribute("href", 2)
                },
                type: function(i) {
                    return i.getAttribute("type")
                }
            };
            if (S.getIdNotName) {
                E.find.ID = function(i, m) {
                    if (typeof m.getElementById !== ga&&!l) {
                        var o = m.getElementById(i);
                        return o && o.parentNode ? [o] : []
                    }
                };
                E.filter.ID = function(i) {
                    var m = i.replace(ia, ja);
                    return function(o) {
                        return o.getAttribute("id") === m
                    }
                }
            } else {
                E.find.ID = function(i, m) {
                    if (typeof m.getElementById !== ga&&!l) {
                        var o = m.getElementById(i);
                        return o ? o.id === i || typeof o.getAttributeNode !== ga && o.getAttributeNode("id").value ===
                        i ? [o] : b : []
                    }
                };
                E.filter.ID = function(i) {
                    var m = i.replace(ia, ja);
                    return function(o) {
                        return (o = typeof o.getAttributeNode !== ga && o.getAttributeNode("id")) && o.value === m
                    }
                }
            }
            E.find.TAG = S.tagNameNoComments ? function(i, m) {
                if (typeof m.getElementsByTagName !== ga)
                    return m.getElementsByTagName(i)
            } : function(i, m) {
                var o, t = [], p = 0, w = m.getElementsByTagName(i);
                if (i === "*") {
                    for (; o = w[p++];)
                        o.nodeType === 1 && t.push(o);
                    return t
                }
                return w
            };
            E.find.NAME = S.getByName && function(i, m) {
                if (typeof m.getElementsByName !== ga)
                    return m.getElementsByName(name)
            };
            E.find.CLASS = S.getByClassName && function(i, m) {
                if (typeof m.getElementsByClassName !== ga&&!l)
                    return m.getElementsByClassName(i)
            };
            u = [];
            r = [":focus"];
            if (S.qsa = d(k.querySelectorAll)) {
                g(function(i) {
                    i.innerHTML = "<select><option selected=''></option></select>";
                    i.querySelectorAll("[selected]").length || r.push("\\[[\\x20\\t\\r\\n\\f]*(?:checked|disabled|ismap|multiple|readonly|selected|value)");
                    i.querySelectorAll(":checked").length || r.push(":checked")
                });
                g(function(i) {
                    i.innerHTML = "<input type='hidden' i=''/>";
                    i.querySelectorAll("[i^='']").length &&
                    r.push("[*^$]=[\\x20\\t\\r\\n\\f]*(?:\"\"|'')");
                    i.querySelectorAll(":enabled").length || r.push(":enabled", ":disabled");
                    i.querySelectorAll("*,:x");
                    r.push(",.*:")
                })
            }
            if (S.matchesSelector = d(v = X.matchesSelector || X.mozMatchesSelector || X.webkitMatchesSelector || X.oMatchesSelector || X.msMatchesSelector))
                g(function(i) {
                    S.disconnectedMatch = v.call(i, "div");
                    v.call(i, "[s!='']:x");
                    u.push("!=", Oa)
                });
            r = RegExp(r.join("|"));
            u = RegExp(u.join("|"));
            I = d(X.contains) || X.compareDocumentPosition ? function(i, m) {
                var o = i.nodeType ===
                9 ? i.documentElement: i, t = m && m.parentNode;
                return i === t||!!(t && t.nodeType === 1 && (o.contains ? o.contains(t) : i.compareDocumentPosition && i.compareDocumentPosition(t) & 16))
            } : function(i, m) {
                if (m)
                    for (; m = m.parentNode;)
                        if (m === i)
                            return true;
                return false
            };
            J = X.compareDocumentPosition ? function(i, m) {
                var o;
                if (i === m) {
                    ca = true;
                    return 0
                }
                if (o = m.compareDocumentPosition && i.compareDocumentPosition && i.compareDocumentPosition(m)) {
                    if (o & 1 || i.parentNode && i.parentNode.nodeType === 11) {
                        if (i === k || I(T, i))
                            return - 1;
                        if (m === k || I(T, m))
                            return 1;
                        return 0
                    }
                    return o & 4?-1 : 1
                }
                return i.compareDocumentPosition?-1 : 1
            } : function(i, m) {
                var o, t = 0;
                o = i.parentNode;
                var p = m.parentNode, w = [i], x = [m];
                if (i === m) {
                    ca = true;
                    return 0
                } else if (i.sourceIndex && m.sourceIndex)
                    return (~m.sourceIndex||-2147483648) - (I(T, i)&&~i.sourceIndex||-2147483648);
                else if (!o ||!p)
                    return i === k?-1 : m === k ? 1 : o?-1 : p ? 1 : 0;
                else if (o === p)
                    return n(i, m);
                for (o = i; o = o.parentNode;)
                    w.unshift(o);
                for (o = m; o = o.parentNode;)
                    x.unshift(o);
                for (; w[t] === x[t];)
                    t++;
                return t ? n(w[t], x[t]) : w[t] === T?-1 : x[t] === T ? 1 : 0
            };
            ca = false;
            [0, 0].sort(J);
            S.detectDuplicates = ca;
            return V
        };
        j.matches = function(h, k) {
            return j(h, null, null, k)
        };
        j.matchesSelector = function(h, k) {
            if ((h.ownerDocument || h) !== V)
                ea(h);
            k = k.replace(Ub, "='$1']");
            if (S.matchesSelector&&!l && (!u ||!u.test(k))&&!r.test(k))
                try {
                    var i = v.call(h, k);
                    if (i || S.disconnectedMatch || h.document && h.document.nodeType !== 11)
                        return i
            } catch (m) {}
            return j(k, V, null, [h]).length > 0
        };
        j.contains = function(h, k) {
            if ((h.ownerDocument || h) !== V)
                ea(h);
            return I(h, k)
        };
        j.attr = function(h, k) {
            var i;
            if ((h.ownerDocument ||
            h) !== V)
                ea(h);
            l || (k = k.toLowerCase());
            if (i = E.attrHandle[k])
                return i(h);
            if (l || S.attributes)
                return h.getAttribute(k);
            return ((i = h.getAttributeNode(k)) || h.getAttribute(k)) && h[k] === true ? k : i && i.specified ? i.value : null
        };
        j.error = function(h) {
            throw Error("Syntax error, unrecognized expression: " + h);
        };
        j.uniqueSort = function(h) {
            var k, i = [], m = 1, o = 0;
            ca=!S.detectDuplicates;
            h.sort(J);
            if (ca) {
                for (; k = h[m]; m++)
                    if (k === h[m - 1])
                        o = i.push(m);
                for (; o--;)
                    h.splice(i[o], 1)
            }
            return h
        };
        ba = j.getText = function(h) {
            var k, i = "", m = 0;
            if (k = h.nodeType)
                if (k ===
                1 || k === 9 || k === 11)
                    if (typeof h.textContent === "string")
                        return h.textContent;
                    else 
                        for (h = h.firstChild; h; h = h.nextSibling)
                            i += ba(h);
            else {
                if (k === 3 || k === 4)
                    return h.nodeValue
            } else 
                for (; k = h[m]; m++)
                    i += ba(k);
            return i
        };
        E = j.selectors = {
            cacheLength: 50,
            createPseudo: f,
            match: Ba,
            find: {},
            relative: {
                ">": {
                    dir: "parentNode",
                    first: true
                },
                " ": {
                    dir: "parentNode"
                },
                "+": {
                    dir: "previousSibling",
                    first: true
                },
                "~": {
                    dir: "previousSibling"
                }
            },
            preFilter: {
                ATTR: function(h) {
                    h[1] = h[1].replace(ia, ja);
                    h[3] = (h[4] || h[5] || "").replace(ia, ja);
                    if (h[2] ===
                    "~=")
                        h[3] = " " + h[3] + " ";
                    return h.slice(0, 4)
                },
                CHILD: function(h) {
                    h[1] = h[1].toLowerCase();
                    if (h[1].slice(0, 3) === "nth") {
                        h[3] || j.error(h[0]);
                        h[4] =+ (h[4] ? h[5] + (h[6] || 1) : 2 * (h[3] === "even" || h[3] === "odd"));
                        h[5] =+ (h[7] + h[8] || h[3] === "odd")
                    } else 
                        h[3] && j.error(h[0]);
                    return h
                },
                PSEUDO: function(h) {
                    var k, i=!h[5] && h[2];
                    if (Ba.CHILD.test(h[0]))
                        return null;
                    if (h[4])
                        h[2] = h[4];
                    else if (i && Qb.test(i) && (k = B(i, true)) && (k = i.indexOf(")", i.length - k) - i.length)) {
                        h[0] = h[0].slice(0, k);
                        h[2] = i.slice(0, k)
                    }
                    return h.slice(0, 3)
                }
            },
            filter: {
                TAG: function(h) {
                    if (h ===
                    "*")
                        return function() {
                            return true
                        };
                    h = h.replace(ia, ja).toLowerCase();
                    return function(k) {
                        return k.nodeName && k.nodeName.toLowerCase() === h
                    }
                },
                CLASS: function(h) {
                    var k = gb[h + " "];
                    return k || (k = RegExp("(^|[\\x20\\t\\r\\n\\f])" + h + "([\\x20\\t\\r\\n\\f]|$)")) && gb(h, function(i) {
                        return k.test(i.className || typeof i.getAttribute !== ga && i.getAttribute("class") || "")
                    })
                },
                ATTR: function(h, k, i) {
                    return function(m) {
                        m = j.attr(m, h);
                        if (m == null)
                            return k === "!=";
                        if (!k)
                            return true;
                        m += "";
                        return k === "=" ? m === i : k === "!=" ? m !== i : k === "^=" ?
                        i && m.indexOf(i) === 0 : k === "*=" ? i && m.indexOf(i)>-1 : k === "$=" ? i && m.slice( - i.length) === i : k === "~=" ? (" " + m + " ").indexOf(i)>-1 : k === "|=" ? m === i || m.slice(0, i.length + 1) === i + "-" : false
                    }
                },
                CHILD: function(h, k, i, m, o) {
                    var t = h.slice(0, 3) !== "nth", p = h.slice( - 4) !== "last", w = k === "of-type";
                    return m === 1 && o === 0 ? function(x) {
                        return !!x.parentNode
                    } : function(x, L, C) {
                        var M, H, R, W, O;
                        L = t !== p ? "nextSibling" : "previousSibling";
                        var N = x.parentNode, da = w && x.nodeName.toLowerCase();
                        C=!C&&!w;
                        if (N) {
                            if (t) {
                                for (; L;) {
                                    for (H = x; H = H[L];)
                                        if (w ? H.nodeName.toLowerCase() ===
                                        da : H.nodeType === 1)
                                            return false;
                                    O = L = h === "only"&&!O && "nextSibling"
                                }
                                return true
                            }
                            O = [p ? N.firstChild: N.lastChild];
                            if (p && C) {
                                C = N[G] || (N[G] = {});
                                M = C[h] || [];
                                W = M[0] === fa && M[1];
                                R = M[0] === fa && M[2];
                                for (H = W && N.childNodes[W]; H=++W && H && H[L] || (R = W = 0) 
                                    || O.pop();
                                )if (H.nodeType === 1&&++R && H === x) {
                                    C[h] = [fa, W, R];
                                    break
                                }
                            } else if (C && (M = (x[G] || (x[G] = {}))[h]) && M[0] === fa)
                                R = M[1];
                            else 
                                for (; H=++W && H && H[L] || (R = W = 0) || O.pop();)
                                    if ((w ? H.nodeName.toLowerCase() === da : H.nodeType === 1)&&++R) {
                                        if (C)(H[G] || (H[G] = {}))
                                            [h] = [fa, R];
                                            if (H === x)
                                                break
                                    }
                            R -= o;
                            return R === m || R%m === 0 && R / m >= 0
                        }
                    }
                },
                PSEUDO: function(h, k) {
                    var i, m = E.pseudos[h] || E.setFilters[h.toLowerCase()] || j.error("unsupported pseudo: " + h);
                    if (m[G])
                        return m(k);
                    if (m.length > 1) {
                        i = [h, h, "", k];
                        return E.setFilters.hasOwnProperty(h.toLowerCase()) ? f(function(o, t) {
                            for (var p, w = m(o, k), x = w.length; x--;) {
                                p = Na.call(o, w[x]);
                                o[p]=!(t[p] = w[x])
                            }
                        }) : function(o) {
                            return m(o, 0, i)
                        }
                    }
                    return m
                }
            },
            pseudos: {
                not: f(function(h) {
                    var k = [], i = [], m = ka(h.replace(Aa, "$1"));
                    return m[G] ? f(function(o, t, p, w) {
                        w = m(o, null, w, []);
                        for (var x = o.length; x--;)
                            if (p =
                            w[x])
                                o[x]=!(t[x] = p)
                    }) : function(o, t, p) {
                        k[0] = o;
                        m(k, null, p, i);
                        return !i.pop()
                    }
                }),
                has: f(function(h) {
                    return function(k) {
                        return j(h, k).length > 0
                    }
                }),
                contains: f(function(h) {
                    return function(k) {
                        return (k.textContent || k.innerText || ba(k)).indexOf(h)>-1
                    }
                }),
                lang: f(function(h) {
                    Rb.test(h || "") || j.error("unsupported lang: " + h);
                    h = h.replace(ia, ja).toLowerCase();
                    return function(k) {
                        var i;
                        do 
                            if (i = l ? k.getAttribute("xml:lang") || k.getAttribute("lang") : k.lang) {
                                i = i.toLowerCase();
                                return i === h || i.indexOf(h + "-") === 0
                            }
                        while ((k = k.parentNode) &&
                        k.nodeType === 1);
                        return false
                    }
                }),
                target: function(h) {
                    var k = a.location && a.location.hash;
                    return k && k.slice(1) === h.id
                },
                root: function(h) {
                    return h === X
                },
                focus: function(h) {
                    return h === V.activeElement && (!V.hasFocus || V.hasFocus())&&!!(h.type || h.href||~h.tabIndex)
                },
                enabled: function(h) {
                    return h.disabled === false
                },
                disabled: function(h) {
                    return h.disabled === true
                },
                checked: function(h) {
                    var k = h.nodeName.toLowerCase();
                    return k === "input"&&!!h.checked || k === "option"&&!!h.selected
                },
                selected: function(h) {
                    return h.selected === true
                },
                empty: function(h) {
                    for (h = h.firstChild; h; h = h.nextSibling)
                        if (h.nodeName > "@" || h.nodeType === 3 || h.nodeType === 4)
                            return false;
                    return true
                },
                parent: function(h) {
                    return !E.pseudos.empty(h)
                },
                header: function(h) {
                    return Tb.test(h.nodeName)
                },
                input: function(h) {
                    return Sb.test(h.nodeName)
                },
                button: function(h) {
                    var k = h.nodeName.toLowerCase();
                    return k === "input" && h.type === "button" || k === "button"
                },
                text: function(h) {
                    var k;
                    return h.nodeName.toLowerCase() === "input" && h.type === "text" && ((k = h.getAttribute("type")) == null || k.toLowerCase() ===
                    h.type)
                },
                first: s(function() {
                    return [0]
                }),
                last: s(function(h, k) {
                    return [k - 1]
                }),
                eq: s(function(h, k, i) {
                    return [i < 0 ? i + k: i]
                }),
                even: s(function(h, k) {
                    for (var i = 0; i < k; i += 2)
                        h.push(i);
                    return h
                }),
                odd: s(function(h, k) {
                    for (var i = 1; i < k; i += 2)
                        h.push(i);
                    return h
                }),
                lt: s(function(h, k, i) {
                    for (k = i < 0 ? i + k : i; --k >= 0;)
                        h.push(k);
                    return h
                }),
                gt: s(function(h, k, i) {
                    for (i = i < 0 ? i + k : i; ++i < k;)
                        h.push(i);
                    return h
                })
            }
        };
        for (U in{
            radio: true,
            checkbox: true,
            file: true,
            password: true,
            image: true
        })
            E.pseudos[U] = q(U);
        for (U in{
            submit: true,
            reset: true
        })
            E.pseudos[U] =
            A(U);
        ka = j.compile = function(h, k) {
            var i, m = [], o = [], t = hb[h + " "];
            if (!t) {
                k || (k = B(h));
                for (i = k.length; i--;) {
                    t = $(k[i]);
                    t[G] ? m.push(t) : o.push(t)
                }
                t = hb(h, xa(o, m))
            }
            return t
        };
        E.pseudos.nth = E.pseudos.eq;
        E.filters = na.prototype = E.pseudos;
        E.setFilters = new na;
        ea();
        c.find = j;
        c.expr = j.selectors;
        c.expr[":"] = c.expr.filters;
        c.unique = j.uniqueSort;
        c.text = j.getText;
        c.isXMLDoc = j.isXML;
        c.contains = j.contains;
        typeof define === "function" && define.amd && define(function() {
            return j
        });
        c.find = j;
        c.expr = j.selectors;
        c.expr[":"] = c.expr.filters;
        c.unique = j.uniqueSort;
        c.text = j.getText;
        c.isXMLDoc = j.isXML;
        c.contains = j.contains
    })(K);
    var Vb = /Until$/, Wb = /^(?:parents|prevUntil|prevAll)/, Xb = /,/, xb = /^.[^:#\[\.,]*$/, Yb = Array.prototype.slice, Zb = c.expr.match.POS;
    c.fn.extend({
        find: function(a) {
            for (var b = this.pushStack("", "find", a), d = 0, e = 0, f = this.length; e < f; e++) {
                d = b.length;
                c.find(a, this[e], b);
                if (e > 0)
                    for (var g = d; g < b.length; g++)
                        for (var j = 0; j < d; j++)
                            if (b[j] === b[g]) {
                                b.splice(g--, 1);
                                break
                            }
            }
            return b
        },
        has: function(a) {
            var b = c(a);
            return this.filter(function() {
                for (var d =
                0, e = b.length; d < e; d++)
                    if (c.contains(this, b[d]))
                        return true
            })
        },
        not: function(a) {
            return this.pushStack(Sa(this, a, false), "not", a)
        },
        filter: function(a) {
            return this.pushStack(Sa(this, a, true), "filter", a)
        },
        is: function(a) {
            return !!a && c.filter(a, this).length > 0
        },
        closest: function(a, b) {
            var d = [], e, f, g = this[0];
            if (c.isArray(a)) {
                var j, n = {}, q = 1;
                if (g && a.length) {
                    e = 0;
                    for (f = a.length; e < f; e++) {
                        j = a[e];
                        n[j] || (n[j] = c.expr.match.POS.test(j) ? c(j, b || this.context) : j)
                    }
                    for (; g && g.ownerDocument && g !== b;) {
                        for (j in n) {
                            e = n[j];
                            if (e.jquery ?
                            e.index(g)>-1 : c(g).is(e))
                                d.push({
                                    selector: j,
                                    elem: g,
                                    level: q
                                })
                            }
                        g = g.parentNode;
                        q++
                    }
                }
                return d
            }
            j = Zb.test(a) ? c(a, b || this.context) : null;
            e = 0;
            for (f = this.length; e < f; e++)
                for (g = this[e]; g;)
                    if (j ? j.index(g)>-1 : c.find.matchesSelector(g, a)) {
                        d.push(g);
                        break
                    } else {
                        g = g.parentNode;
                        if (!g ||!g.ownerDocument || g === b)
                            break
                    }
            d = d.length > 1 ? c.unique(d) : d;
            return this.pushStack(d, "closest", a)
        },
        index: function(a) {
            if (!a || typeof a === "string")
                return c.inArray(this[0], a ? c(a) : this.parent().children());
            return c.inArray(a.jquery ? a[0] : a, this)
        },
        add: function(a, b) {
            var d = typeof a === "string" ? c(a, b || this.context): c.makeArray(a), e = c.merge(this.get(), d);
            return this.pushStack(!d[0] ||!d[0].parentNode || d[0].parentNode.nodeType === 11 ||!e[0] ||!e[0].parentNode || e[0].parentNode.nodeType === 11 ? e : c.unique(e))
        },
        andSelf: function() {
            return this.add(this.prevObject)
        }
    });
    c.each({
        parent: function(a) {
            return (a = a.parentNode) && a.nodeType !== 11 ? a : null
        },
        parents: function(a) {
            return c.dir(a, "parentNode")
        },
        parentsUntil: function(a, b, d) {
            return c.dir(a, "parentNode", d)
        },
        next: function(a) {
            return c.nth(a,
            2, "nextSibling")
        },
        prev: function(a) {
            return c.nth(a, 2, "previousSibling")
        },
        nextAll: function(a) {
            return c.dir(a, "nextSibling")
        },
        prevAll: function(a) {
            return c.dir(a, "previousSibling")
        },
        nextUntil: function(a, b, d) {
            return c.dir(a, "nextSibling", d)
        },
        prevUntil: function(a, b, d) {
            return c.dir(a, "previousSibling", d)
        },
        siblings: function(a) {
            return c.sibling(a.parentNode.firstChild, a)
        },
        children: function(a) {
            return c.sibling(a.firstChild)
        },
        contents: function(a) {
            return c.nodeName(a, "iframe") ? a.contentDocument || a.contentWindow.document :
            c.makeArray(a.childNodes)
        }
    }, function(a, b) {
        c.fn[a] = function(d, e) {
            var f = c.map(this, b, d);
            Vb.test(a) || (e = d);
            if (e && typeof e === "string")
                f = c.filter(e, f);
            f = this.length > 1 ? c.unique(f) : f;
            if ((this.length > 1 || Xb.test(e)) && Wb.test(a))
                f = f.reverse();
            return this.pushStack(f, a, Yb.call(arguments).join(","))
        }
    });
    c.extend({
        filter: function(a, b, d) {
            if (d)
                a = ":not(" + a + ")";
            return b.length === 1 ? c.find.matchesSelector(b[0], a) ? [b[0]] : [] : c.find.matches(a, b)
        },
        dir: function(a, b, d) {
            var e = [];
            for (a = a[b]; a && a.nodeType !== 9 && (d === D || a.nodeType !==
            1 ||!c(a).is(d));) {
                a.nodeType === 1 && e.push(a);
                a = a[b]
            }
            return e
        },
        nth: function(a, b, d) {
            b = b || 1;
            for (var e = 0; a; a = a[d])
                if (a.nodeType === 1&&++e === b)
                    break;
            return a
        },
        sibling: function(a, b) {
            for (var d = []; a; a = a.nextSibling)
                a.nodeType === 1 && a !== b && d.push(a);
            return d
        }
    });
    var jb = / jQuery\d+="(?:\d+|null)"/g, Ca = /^\s+/, kb = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\w:]+)[^>]*)\/>/ig, lb = /<([\w:]+)/, $b = /<tbody/i, ac = /<|&#?\w+;/, mb = /<(?:script|object|embed|option|style)/i, nb = /checked\s*(?:[^=]|=\s*.checked.)/i, bc =
    /\=([^="'>\s]+\/)>/g, Z = {
        option: [1, "<select multiple='multiple'>", "</select>"],
        legend: [1, "<fieldset>", "</fieldset>"],
        thead: [1, "<table>", "</table>"],
        tr: [2, "<table><tbody>", "</tbody></table>"],
        td: [3, "<table><tbody><tr>", "</tr></tbody></table>"],
        col: [2, "<table><tbody></tbody><colgroup>", "</colgroup></table>"],
        area: [1, "<map>", "</map>"],
        _default: [0, "", ""]
    };
    Z.optgroup = Z.option;
    Z.tbody = Z.tfoot = Z.colgroup = Z.caption = Z.thead;
    Z.th = Z.td;
    if (!c.support.htmlSerialize)
        Z._default = [1, "div<div>", "</div>"];
    c.fn.extend({
        text: function(a) {
            if (c.isFunction(a))
                return this.each(function(b) {
                    var d =
                    c(this);
                    d.text(a.call(this, b, d.text()))
                });
            if (typeof a !== "object" && a !== D)
                return this.empty().append((this[0] && this[0].ownerDocument || y).createTextNode(a));
            return c.text(this)
        },
        wrapAll: function(a) {
            if (c.isFunction(a))
                return this.each(function(d) {
                    c(this).wrapAll(a.call(this, d))
                });
            if (this[0]) {
                var b = c(a, this[0].ownerDocument).eq(0).clone(true);
                this[0].parentNode && b.insertBefore(this[0]);
                b.map(function() {
                    for (var d = this; d.firstChild && d.firstChild.nodeType === 1;)
                        d = d.firstChild;
                    return d
                }).append(this)
            }
            return this
        },
        wrapInner: function(a) {
            if (c.isFunction(a))
                return this.each(function(b) {
                    c(this).wrapInner(a.call(this, b))
                });
            return this.each(function() {
                var b = c(this), d = b.contents();
                d.length ? d.wrapAll(a) : b.append(a)
            })
        },
        wrap: function(a) {
            return this.each(function() {
                c(this).wrapAll(a)
            })
        },
        unwrap: function() {
            return this.parent().each(function() {
                c.nodeName(this, "body") || c(this).replaceWith(this.childNodes)
            }).end()
        },
        append: function() {
            return this.domManip(arguments, true, function(a) {
                this.nodeType === 1 && this.appendChild(a)
            })
        },
        prepend: function() {
            return this.domManip(arguments, true, function(a) {
                this.nodeType === 1 && this.insertBefore(a, this.firstChild)
            })
        },
        before: function() {
            if (this[0] && this[0].parentNode)
                return this.domManip(arguments, false, function(b) {
                    this.parentNode.insertBefore(b, this)
                });
            else if (arguments.length) {
                var a = c(arguments[0]);
                a.push.apply(a, this.toArray());
                return this.pushStack(a, "before", arguments)
            }
        },
        after: function() {
            if (this[0] && this[0].parentNode)
                return this.domManip(arguments, false, function(b) {
                    this.parentNode.insertBefore(b,
                    this.nextSibling)
                });
            else if (arguments.length) {
                var a = this.pushStack(this, "after", arguments);
                a.push.apply(a, c(arguments[0]).toArray());
                return a
            }
        },
        remove: function(a, b) {
            for (var d = 0, e; (e = this[d]) != null; d++)
                if (!a || c.filter(a, [e]).length) {
                    if (!b && e.nodeType === 1) {
                        c.cleanData(e.getElementsByTagName("*"));
                        c.cleanData([e])
                    }
                    e.parentNode && e.parentNode.removeChild(e)
                }
            return this
        },
        empty: function() {
            for (var a = 0, b; (b = this[a]) != null; a++)
                for (b.nodeType === 1 && c.cleanData(b.getElementsByTagName("*")); b.firstChild;)
                    b.removeChild(b.firstChild);
            return this
        },
        clone: function(a) {
            var b = this.map(function() {
                if (!c.support.noCloneEvent&&!c.isXMLDoc(this)) {
                    var d = this.outerHTML, e = this.ownerDocument;
                    if (!d) {
                        d = e.createElement("div");
                        d.appendChild(this.cloneNode(true));
                        d = d.innerHTML
                    }
                    return c.clean([d.replace(jb, "").replace(bc, '="$1">').replace(Ca, "")], e)[0]
                } else 
                    return this.cloneNode(true)
            });
            if (a === true) {
                Ta(this, b);
                Ta(this.find("*"), b.find("*"))
            }
            return b
        },
        html: function(a) {
            if (a === D)
                return this[0] && this[0].nodeType === 1 ? this[0].innerHTML.replace(jb, "") : null;
            else if (typeof a === "string"&&!mb.test(a) && (c.support.leadingWhitespace ||!Ca.test(a))&&!Z[(lb.exec(a) || ["", ""])[1].toLowerCase()]) {
                a = a.replace(kb, "<$1></$2>");
                try {
                    for (var b = 0, d = this.length; b < d; b++)
                        if (this[b].nodeType === 1) {
                            c.cleanData(this[b].getElementsByTagName("*"));
                            this[b].innerHTML = a
                        }
                } catch (e) {
                    this.empty().append(a)
                }
            } else 
                c.isFunction(a) ? this.each(function(f) {
                    var g = c(this);
                    g.html(a.call(this, f, g.html()))
                }) : this.empty().append(a);
            return this
        },
        replaceWith: function(a) {
            if (this[0] && this[0].parentNode) {
                if (c.isFunction(a))
                    return this.each(function(b) {
                        var d =
                        c(this), e = d.html();
                        d.replaceWith(a.call(this, b, e))
                    });
                if (typeof a !== "string")
                    a = c(a).detach();
                return this.each(function() {
                    var b = this.nextSibling, d = this.parentNode;
                    c(this).remove();
                    b ? c(b).before(a) : c(d).append(a)
                })
            } else 
                return this.pushStack(c(c.isFunction(a) ? a() : a), "replaceWith", a)
        },
        detach: function(a) {
            return this.remove(a, true)
        },
        domManip: function(a, b, d) {
            var e, f, g, j = a[0], n = [];
            if (!c.support.checkClone && arguments.length === 3 && typeof j === "string" && nb.test(j))
                return this.each(function() {
                    c(this).domManip(a,
                    b, d, true)
                });
            if (c.isFunction(j))
                return this.each(function(A) {
                    var s = c(this);
                    a[0] = j.call(this, A, b ? s.html() : D);
                    s.domManip(a, b, d)
                });
            if (this[0]) {
                e = j && j.parentNode;
                e = c.support.parentNode && e && e.nodeType === 11 && e.childNodes.length === this.length ? {
                    fragment: e
                } : c.buildFragment(a, this, n);
                g = e.fragment;
                if (f = g.childNodes.length === 1 ? g = g.firstChild : g.firstChild) {
                    b = b && c.nodeName(f, "tr");
                    f = 0;
                    for (var q = this.length; f < q; f++)
                        d.call(b ? c.nodeName(this[f], "table") ? this[f].getElementsByTagName("tbody")[0] || this[f].appendChild(this[f].ownerDocument.createElement("tbody")) :
                        this[f] : this[f], f > 0 || e.cacheable || this.length > 1 ? g.cloneNode(true) : g)
                    }
                n.length && c.each(n, yb)
            }
            return this
        }
    });
    c.buildFragment = function(a, b, d) {
        var e, f, g;
        b = b && b[0] ? b[0].ownerDocument || b[0] : y;
        if (a.length === 1 && typeof a[0] === "string" && a[0].length < 512 && b === y&&!mb.test(a[0]) && (c.support.checkClone ||!nb.test(a[0]))) {
            f = true;
            if (g = c.fragments[a[0]])
                if (g !== 1)
                    e = g
        }
        if (!e) {
            e = b.createDocumentFragment();
            c.clean(a, b, e, d)
        }
        if (f)
            c.fragments[a[0]] = g ? e : 1;
        return {
            fragment: e,
            cacheable: f
        }
    };
    c.fragments = {};
    c.each({
        appendTo: "append",
        prependTo: "prepend",
        insertBefore: "before",
        insertAfter: "after",
        replaceAll: "replaceWith"
    }, function(a, b) {
        c.fn[a] = function(d) {
            var e = [];
            d = c(d);
            var f = this.length === 1 && this[0].parentNode;
            if (f && f.nodeType === 11 && f.childNodes.length === 1 && d.length === 1) {
                d[b](this[0]);
                return this
            } else {
                f = 0;
                for (var g = d.length; f < g; f++) {
                    var j = (f > 0 ? this.clone(true) : this).get();
                    c(d[f])[b](j);
                    e = e.concat(j)
                }
                return this.pushStack(e, a, d.selector)
            }
        }
    });
    c.extend({
        clean: function(a, b, d, e) {
            b = b || y;
            if (typeof b.createElement === "undefined")
                b = b.ownerDocument ||
                b[0] && b[0].ownerDocument || y;
            for (var f = [], g = 0, j; (j = a[g]) != null; g++) {
                if (typeof j === "number")
                    j += "";
                if (j) {
                    if (typeof j === "string"&&!ac.test(j))
                        j = b.createTextNode(j);
                    else if (typeof j === "string") {
                        j = j.replace(kb, "<$1></$2>");
                        var n = (lb.exec(j) || ["", ""])[1].toLowerCase(), q = Z[n] || Z._default, A = q[0], s = b.createElement("div");
                        for (s.innerHTML = q[1] + j + q[2]; A--;)
                            s = s.lastChild;
                        if (!c.support.tbody) {
                            A = $b.test(j);
                            n = n === "table"&&!A ? s.firstChild && s.firstChild.childNodes : q[1] === "<table>"&&!A ? s.childNodes : [];
                            for (q = n.length -
                            1; q >= 0; --q)
                                c.nodeName(n[q], "tbody")&&!n[q].childNodes.length && n[q].parentNode.removeChild(n[q])
                            }
                        !c.support.leadingWhitespace && Ca.test(j) && s.insertBefore(b.createTextNode(Ca.exec(j)[0]), s.firstChild);
                        j = s.childNodes
                    }
                    if (j.nodeType)
                        f.push(j);
                    else 
                        f = c.merge(f, j)
                    }
            }
            if (d)
                for (g = 0; f[g]; g++)
                    if (e && c.nodeName(f[g], "script") && (!f[g].type || f[g].type.toLowerCase() === "text/javascript"))
                        e.push(f[g].parentNode ? f[g].parentNode.removeChild(f[g]) : f[g]);
                    else {
                        f[g].nodeType === 1 && f.splice.apply(f, [g + 1, 0].concat(c.makeArray(f[g].getElementsByTagName("script"))));
                        d.appendChild(f[g])
                    }
            return f
        },
        cleanData: function(a) {
            for (var b, d, e = c.cache, f = c.event.special, g = c.support.deleteExpando, j = 0, n; (n = a[j]) != null; j++)
                if (!(n.nodeName && c.noData[n.nodeName.toLowerCase()]))
                    if (d = n[c.expando]) {
                        if ((b = e[d]) && b.events)
                            for (var q in b.events)
                                f[q] ? c.event.remove(n, q) : c.removeEvent(n, q, b.handle);
                                if (g)
                                    delete n[c.expando];
                                else 
                                    n.removeAttribute && n.removeAttribute(c.expando);
                                    delete e[d]
                    }
        }
    });
    var ob = /alpha\([^)]*\)/i, cc = /opacity=([^)]*)/, dc = /-([a-z])/ig, ec = /([A-Z])/g, pb = /^-?\d+(?:px)?$/i,
    fc = /^-?\d/, gc = {
        position: "absolute",
        visibility: "hidden",
        display: "block"
    }, zb = ["Left", "Right"], Ab = ["Top", "Bottom"], ua, qb, Da, hc = function(a, b) {
        return b.toUpperCase()
    };
    c.fn.css = function(a, b) {
        if (arguments.length === 2 && b === D)
            return this;
        return c.access(this, a, b, true, function(d, e, f) {
            return f !== D ? c.style(d, e, f) : c.css(d, e)
        })
    };
    c.extend({
        cssHooks: {
            opacity: {
                get: function(a, b) {
                    if (b) {
                        var d = ua(a, "opacity", "opacity");
                        return d === "" ? "1" : d
                    } else 
                        return a.style.opacity
                }
            }
        },
        cssNumber: {
            zIndex: true,
            fontWeight: true,
            opacity: true,
            zoom: true,
            lineHeight: true
        },
        cssProps: {
            "float": c.support.cssFloat ? "cssFloat": "styleFloat"
        },
        style: function(a, b, d, e) {
            if (!(!a || a.nodeType === 3 || a.nodeType === 8 ||!a.style)) {
                var f, g = c.camelCase(b), j = a.style, n = c.cssHooks[g];
                b = c.cssProps[g] || g;
                if (d !== D) {
                    if (!(typeof d === "number" && isNaN(d) || d == null)) {
                        if (typeof d === "number"&&!c.cssNumber[g])
                            d += "px";
                        if (!n ||!("set"in n) || (d = n.set(a, d)) !== D)
                            try {
                                j[b] = d
                        } catch (q) {}
                    }
                } else {
                    if (n && "get"in n && (f = n.get(a, false, e)) !== D)
                        return f;
                    return j[b]
                }
            }
        },
        css: function(a, b, d) {
            var e, f = c.camelCase(b),
            g = c.cssHooks[f];
            b = c.cssProps[f] || f;
            if (g && "get"in g && (e = g.get(a, true, d)) !== D)
                return e;
            else if (ua)
                return ua(a, b, f)
        },
        swap: function(a, b, d) {
            var e = {}, f;
            for (f in b) {
                e[f] = a.style[f];
                a.style[f] = b[f]
            }
            d.call(a);
            for (f in b)
                a.style[f] = e[f]
        },
        camelCase: function(a) {
            return a.replace(dc, hc)
        }
    });
    c.curCSS = c.css;
    c.each(["height", "width"], function(a, b) {
        c.cssHooks[b] = {
            get: function(d, e, f) {
                var g;
                if (e) {
                    if (d.offsetWidth !== 0)
                        g = Ua(d, b, f);
                    else 
                        c.swap(d, gc, function() {
                            g = Ua(d, b, f)
                        });
                    if (g <= 0) {
                        g = ua(d, b, b);
                        if (g === "0px" && Da)
                            g = Da(d, b,
                            b);
                        if (g != null)
                            return g === "" || g === "auto" ? "0px" : g
                    }
                    if (g < 0 || g == null) {
                        g = d.style[b];
                        return g === "" || g === "auto" ? "0px" : g
                    }
                    return typeof g === "string" ? g : g + "px"
                }
            },
            set: function(d, e) {
                if (pb.test(e)) {
                    e = parseFloat(e);
                    if (e >= 0)
                        return e + "px"
                } else 
                    return e
            }
        }
    });
    if (!c.support.opacity)
        c.cssHooks.opacity = {
            get: function(a, b) {
                return cc.test((b && a.currentStyle ? a.currentStyle.filter : a.style.filter) || "") ? parseFloat(RegExp.$1) / 100 + "" : b ? "1" : ""
            },
            set: function(a, b) {
                var d = a.style;
                d.zoom = 1;
                var e = c.isNaN(b) ? "": "alpha(opacity=" + b * 100 + ")",
                f = d.filter || "";
                d.filter = ob.test(f) ? f.replace(ob, e) : d.filter + " " + e
            }
        };
    if (y.defaultView && y.defaultView.getComputedStyle)
        qb = function(a, b, d) {
            var e;
            d = d.replace(ec, "-$1").toLowerCase();
            if (!(b = a.ownerDocument.defaultView))
                return D;
                if (b = b.getComputedStyle(a, null)) {
                    e = b.getPropertyValue(d);
                    if (e === ""&&!c.contains(a.ownerDocument.documentElement, a))
                        e = c.style(a, d)
                }
                return e
            };
    if (y.documentElement.currentStyle)
        Da = function(a, b) {
            var d, e, f = a.currentStyle && a.currentStyle[b], g = a.style;
            if (!pb.test(f) && fc.test(f)) {
                d = g.left;
                e = a.runtimeStyle.left;
                a.runtimeStyle.left = a.currentStyle.left;
                g.left = b === "fontSize" ? "1em" : f || 0;
                f = g.pixelLeft + "px";
                g.left = d;
                a.runtimeStyle.left = e
            }
            return f === "" ? "auto" : f
        };
    ua = qb || Da;
    if (c.expr && c.expr.filters) {
        c.expr.filters.hidden = function(a) {
            var b = a.offsetHeight;
            return a.offsetWidth === 0 && b === 0 ||!c.support.reliableHiddenOffsets && (a.style.display || c.css(a, "display")) === "none"
        };
        c.expr.filters.visible = function(a) {
            return !c.expr.filters.hidden(a)
        }
    }
    var ic = c.now(), jc = /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
    kc = /^(?:select|textarea)/i, lc = /^(?:color|date|datetime|email|hidden|month|number|password|range|search|tel|text|time|url|week)$/i, mc = /^(?:GET|HEAD)$/, Bb = /\[\]$/, ra = /\=\?(&|$)/, Pa = /\?/, nc = /([?&])_=[^&]*/, oc = /^(\w+:)?\/\/([^\/?#]+)/, pc = /%20/g, qc = /#.*$/, rb = c.fn.load;
    c.fn.extend({
        load: function(a, b, d) {
            if (typeof a !== "string" && rb)
                return rb.apply(this, arguments);
            else if (!this.length)
                return this;
            var e = a.indexOf(" ");
            if (e >= 0) {
                var f = a.slice(e, a.length);
                a = a.slice(0, e)
            }
            e = "GET";
            if (b)
                if (c.isFunction(b)) {
                    d = b;
                    b = null
                } else if (typeof b ===
                "object") {
                    b = c.param(b, c.ajaxSettings.traditional);
                    e = "POST"
                }
            var g = this;
            c.ajax({
                url: a,
                type: e,
                dataType: "html",
                data: b,
                complete: function(j, n) {
                    if (n === "success" || n === "notmodified")
                        g.html(f ? c("<div>").append(j.responseText.replace(jc, "")).find(f) : j.responseText);
                    d && g.each(d, [j.responseText, n, j])
                }
            });
            return this
        },
        serialize: function() {
            return c.param(this.serializeArray())
        },
        serializeArray: function() {
            return this.map(function() {
                return this.elements ? c.makeArray(this.elements) : this
            }).filter(function() {
                return this.name &&
                !this.disabled && (this.checked || kc.test(this.nodeName) || lc.test(this.type))
            }).map(function(a, b) {
                var d = c(this).val();
                return d == null ? null : c.isArray(d) ? c.map(d, function(e) {
                    return {
                        name: b.name,
                        value: e
                    }
                }) : {
                    name: b.name,
                    value: d
                }
            }).get()
        }
    });
    c.each("ajaxStart ajaxStop ajaxComplete ajaxError ajaxSuccess ajaxSend".split(" "), function(a, b) {
        c.fn[b] = function(d) {
            return this.bind(b, d)
        }
    });
    c.extend({
        get: function(a, b, d, e) {
            if (c.isFunction(b)) {
                e = e || d;
                d = b;
                b = null
            }
            return c.ajax({
                type: "GET",
                url: a,
                data: b,
                success: d,
                dataType: e
            })
        },
        getScript: function(a, b) {
            return c.get(a, null, b, "script")
        },
        getJSON: function(a, b, d) {
            return c.get(a, b, d, "json")
        },
        post: function(a, b, d, e) {
            if (c.isFunction(b)) {
                e = e || d;
                d = b;
                b = {}
            }
            return c.ajax({
                type: "POST",
                url: a,
                data: b,
                success: d,
                dataType: e
            })
        },
        ajaxSetup: function(a) {
            c.extend(c.ajaxSettings, a)
        },
        ajaxSettings: {
            url: location.href,
            global: true,
            type: "GET",
            contentType: "application/x-www-form-urlencoded",
            processData: true,
            async: true,
            xhr: function() {
                return new K.XMLHttpRequest
            },
            accepts: {
                xml: "application/xml, text/xml",
                html: "text/html",
                script: "text/javascript, application/javascript",
                json: "application/json, text/javascript",
                text: "text/plain",
                _default: "*/*"
            }
        },
        ajax: function(a) {
            var b = c.extend(true, {}, c.ajaxSettings, a), d, e, f, g = b.type.toUpperCase(), j = mc.test(g);
            b.url = b.url.replace(qc, "");
            b.context = a && a.context != null ? a.context : b;
            if (b.data && b.processData && typeof b.data !== "string")
                b.data = c.param(b.data, b.traditional);
            if (b.dataType === "jsonp") {
                if (g === "GET")
                    ra.test(b.url) || (b.url += (Pa.test(b.url) ? "&" : "?") + (b.jsonp || "callback") + "=?");
                else if (!b.data ||
                !ra.test(b.data))
                    b.data = (b.data ? b.data + "&" : "") + (b.jsonp || "callback") + "=?";
                b.dataType = "json"
            }
            if (b.dataType === "json" && (b.data && ra.test(b.data) || ra.test(b.url))) {
                d = b.jsonpCallback || "jsonp" + ic++;
                if (b.data)
                    b.data = (b.data + "").replace(ra, "=" + d + "$1");
                b.url = b.url.replace(ra, "=" + d + "$1");
                b.dataType = "script";
                var n = K[d];
                K[d] = function(U) {
                    if (c.isFunction(n))
                        n(U);
                    else {
                        K[d] = D;
                        try {
                            delete K[d]
                        } catch (aa) {}
                    }
                    f = U;
                    c.handleSuccess(b, z, e, f);
                    c.handleComplete(b, z, e, f);
                    s && s.removeChild(B)
                }
            }
            if (b.dataType === "script" && b.cache ===
            null)
                b.cache = false;
            if (b.cache === false && j) {
                var q = c.now(), A = b.url.replace(nc, "$1_=" + q);
                b.url = A + (A === b.url ? (Pa.test(b.url) ? "&" : "?") + "_=" + q : "")
            }
            if (b.data && j)
                b.url += (Pa.test(b.url) ? "&" : "?") + b.data;
            b.global && c.active++===0 && c.event.trigger("ajaxStart");
            q = (q = oc.exec(b.url)) && (q[1] && q[1].toLowerCase() !== location.protocol || q[2].toLowerCase() !== location.host);
            if (b.dataType === "script" && g === "GET" && q) {
                var s = y.getElementsByTagName("head")[0] || y.documentElement, B = y.createElement("script");
                if (b.scriptCharset)
                    B.charset =
                    b.scriptCharset;
                B.src = b.url;
                if (!d) {
                    var F = false;
                    B.onload = B.onreadystatechange = function() {
                        if (!F && (!this.readyState || this.readyState === "loaded" || this.readyState === "complete")) {
                            F = true;
                            c.handleSuccess(b, z, e, f);
                            c.handleComplete(b, z, e, f);
                            B.onload = B.onreadystatechange = null;
                            s && B.parentNode && s.removeChild(B)
                        }
                    }
                }
                s.insertBefore(B, s.firstChild);
                return D
            }
            var P = false, z = b.xhr();
            if (z) {
                b.username ? z.open(g, b.url, b.async, b.username, b.password) : z.open(g, b.url, b.async);
                try {
                    if (b.data != null&&!j || a && a.contentType)
                        z.setRequestHeader("Content-Type",
                        b.contentType);
                    if (b.ifModified) {
                        c.lastModified[b.url] && z.setRequestHeader("If-Modified-Since", c.lastModified[b.url]);
                        c.etag[b.url] && z.setRequestHeader("If-None-Match", c.etag[b.url])
                    }
                    q || z.setRequestHeader("X-Requested-With", "XMLHttpRequest");
                    z.setRequestHeader("Accept", b.dataType && b.accepts[b.dataType] ? b.accepts[b.dataType] + ", */*; q=0.01" : b.accepts._default)
                } catch (Q) {}
                if (b.beforeSend && b.beforeSend.call(b.context, z, b) === false) {
                    b.global && c.active--===1 && c.event.trigger("ajaxStop");
                    z.abort();
                    return false
                }
                b.global &&
                c.triggerGlobal(b, "ajaxSend", [z, b]);
                var Y = z.onreadystatechange = function(U) {
                    if (!z || z.readyState === 0 || U === "abort") {
                        P || c.handleComplete(b, z, e, f);
                        P = true;
                        if (z)
                            z.onreadystatechange = c.noop
                    } else if (!P && z && (z.readyState === 4 || U === "timeout")) {
                        P = true;
                        z.onreadystatechange = c.noop;
                        e = U === "timeout" ? "timeout" : !c.httpSuccess(z) ? "error" : b.ifModified && c.httpNotModified(z, b.url) ? "notmodified" : "success";
                        var aa;
                        if (e === "success")
                            try {
                                f = c.httpData(z, b.dataType, b)
                            } catch (E) {
                            e = "parsererror";
                            aa = E
                        }
                        if (e === "success" || e === "notmodified")
                            d ||
                            c.handleSuccess(b, z, e, f);
                        else 
                            c.handleError(b, z, e, aa);
                        d || c.handleComplete(b, z, e, f);
                        U === "timeout" && z.abort();
                        if (b.async)
                            z = null
                    }
                };
                try {
                    var $ = z.abort;
                    z.abort = function() {
                        z && Function.prototype.call.call($, z);
                        Y("abort")
                    }
                } catch (xa) {}
                b.async && b.timeout > 0 && setTimeout(function() {
                    z&&!P && Y("timeout")
                }, b.timeout);
                try {
                    z.send(j || b.data == null ? null : b.data)
                } catch (na) {
                    c.handleError(b, z, null, na);
                    c.handleComplete(b, z, e, f)
                }
                b.async || Y();
                return z
            }
        },
        param: function(a, b) {
            var d = [], e = function(g, j) {
                j = c.isFunction(j) ? j() : j;
                d[d.length] =
                encodeURIComponent(g) + "=" + encodeURIComponent(j)
            };
            if (b === D)
                b = c.ajaxSettings.traditional;
            if (c.isArray(a) || a.jquery)
                c.each(a, function() {
                    e(this.name, this.value)
                });
            else 
                for (var f in a)
                    Ga(f, a[f], b, e);
            return d.join("&").replace(pc, "+")
        }
    });
    c.extend({
        active: 0,
        lastModified: {},
        etag: {},
        handleError: function(a, b, d, e) {
            a.error && a.error.call(a.context, b, d, e);
            a.global && c.triggerGlobal(a, "ajaxError", [b, a, e])
        },
        handleSuccess: function(a, b, d, e) {
            a.success && a.success.call(a.context, e, d, b);
            a.global && c.triggerGlobal(a, "ajaxSuccess",
            [b, a])
        },
        handleComplete: function(a, b, d) {
            a.complete && a.complete.call(a.context, b, d);
            a.global && c.triggerGlobal(a, "ajaxComplete", [b, a]);
            a.global && c.active--===1 && c.event.trigger("ajaxStop")
        },
        triggerGlobal: function(a, b, d) {
            (a.context && a.context.url == null ? c(a.context) : c.event).trigger(b, d)
        },
        httpSuccess: function(a) {
            try {
                return !a.status && location.protocol === "file:" || a.status >= 200 && a.status < 300 || a.status === 304 || a.status === 1223
            } catch (b) {}
            return false
        },
        httpNotModified: function(a, b) {
            var d = a.getResponseHeader("Last-Modified"),
            e = a.getResponseHeader("Etag");
            if (d)
                c.lastModified[b] = d;
            if (e)
                c.etag[b] = e;
            return a.status === 304
        },
        httpData: function(a, b, d) {
            var e = a.getResponseHeader("content-type") || "", f = b === "xml" ||!b && e.indexOf("xml") >= 0;
            a = f ? a.responseXML : a.responseText;
            f && a.documentElement.nodeName === "parsererror" && c.error("parsererror");
            if (d && d.dataFilter)
                a = d.dataFilter(a, b);
            if (typeof a === "string")
                if (b === "json" ||!b && e.indexOf("json") >= 0)
                    a = c.parseJSON(a);
                else if (b === "script" ||!b && e.indexOf("javascript") >= 0)
                    c.globalEval(a);
            return a
        }
    });
    if (K.ActiveXObject)
        c.ajaxSettings.xhr = function() {
            if (K.location.protocol !== "file:")
                try {
                    return new K.XMLHttpRequest
                } catch (a) {}
                try {
                    return new K.ActiveXObject("Microsoft.XMLHTTP")
                } catch (b) {}
            };
    c.support.ajax=!!c.ajaxSettings.xhr();
    var Ha = {}, rc = /^(?:toggle|show|hide)$/, sc = /^([+\-]=)?([\d+.\-]+)(.*)$/, Ea, Va = [["height", "marginTop", "marginBottom", "paddingTop", "paddingBottom"], ["width", "marginLeft", "marginRight", "paddingLeft", "paddingRight"], ["opacity"]];
    c.fn.extend({
        show: function(a, b, d) {
            if (a || a === 0)
                return this.animate(ma("show",
                3), a, b, d);
            else {
                d = 0;
                for (var e = this.length; d < e; d++) {
                    a = this[d];
                    b = a.style.display;
                    if (!c.data(a, "olddisplay") && b === "none")
                        b = a.style.display = "";
                    b === "" && c.css(a, "display") === "none" && c.data(a, "olddisplay", Wa(a.nodeName))
                }
                for (d = 0; d < e; d++) {
                    a = this[d];
                    b = a.style.display;
                    if (b === "" || b === "none")
                        a.style.display = c.data(a, "olddisplay") || ""
                }
                return this
            }
        },
        hide: function(a, b, d) {
            if (a || a === 0)
                return this.animate(ma("hide", 3), a, b, d);
            else {
                a = 0;
                for (b = this.length; a < b; a++) {
                    d = c.css(this[a], "display");
                    d !== "none" && c.data(this[a], "olddisplay",
                    d)
                }
                for (a = 0; a < b; a++)
                    this[a].style.display = "none";
                return this
            }
        },
        _toggle: c.fn.toggle,
        toggle: function(a, b, d) {
            var e = typeof a === "boolean";
            if (c.isFunction(a) && c.isFunction(b))
                this._toggle.apply(this, arguments);
            else 
                a == null || e ? this.each(function() {
                    var f = e ? a: c(this).is(":hidden");
                    c(this)[f ? "show": "hide"]()
                }) : this.animate(ma("toggle", 3), a, b, d);
            return this
        },
        fadeTo: function(a, b, d, e) {
            return this.filter(":hidden").css("opacity", 0).show().end().animate({
                opacity: b
            }, a, d, e)
        },
        animate: function(a, b, d, e) {
            var f = c.speed(b,
            d, e);
            if (c.isEmptyObject(a))
                return this.each(f.complete);
            return this[f.queue === false ? "each": "queue"](function() {
                var g = c.extend({}, f), j, n = this.nodeType === 1, q = n && c(this).is(":hidden"), A = this;
                for (j in a) {
                    var s = c.camelCase(j);
                    if (j !== s) {
                        a[s] = a[j];
                        delete a[j];
                        j = s
                    }
                    if (a[j] === "hide" && q || a[j] === "show"&&!q)
                        return g.complete.call(this);
                    if (n && (j === "height" || j === "width")) {
                        g.overflow = [this.style.overflow, this.style.overflowX, this.style.overflowY];
                        if (c.css(this, "display") === "inline" && c.css(this, "float") === "none")
                            if (c.support.inlineBlockNeedsLayout)
                                if (Wa(this.nodeName) ===
                                "inline")
                                    this.style.display = "inline-block";
                                else {
                                    this.style.display = "inline";
                                    this.style.zoom = 1
                                } else 
                                    this.style.display = "inline-block"
                    }
                    if (c.isArray(a[j])) {
                        (g.specialEasing = g.specialEasing || {})[j] = a[j][1];
                        a[j] = a[j][0]
                    }
                }
                if (g.overflow != null)
                    this.style.overflow = "hidden";
                g.curAnim = c.extend({}, a);
                c.each(a, function(B, F) {
                    var P = new c.fx(A, g, B);
                    if (rc.test(F))
                        P[F === "toggle" ? q ? "show": "hide": F](a);
                    else {
                        var z = sc.exec(F), Q = P.cur() || 0;
                        if (z) {
                            var Y = parseFloat(z[2]), $ = z[3] || "px";
                            if ($ !== "px") {
                                c.style(A, B, (Y || 1) + $);
                                Q = (Y ||
                                1) / P.cur() * Q;
                                c.style(A, B, Q + $)
                            }
                            if (z[1])
                                Y = (z[1] === "-="?-1 : 1) * Y + Q;
                            P.custom(Q, Y, $)
                        } else 
                            P.custom(Q, F, "")
                    }
                });
                return true
            })
        },
        stop: function(a, b) {
            var d = c.timers;
            a && this.queue([]);
            this.each(function() {
                for (var e = d.length - 1; e >= 0; e--)
                    if (d[e].elem === this) {
                        b && d[e](true);
                        d.splice(e, 1)
                    }
            });
            b || this.dequeue();
            return this
        }
    });
    c.each({
        slideDown: ma("show", 1),
        slideUp: ma("hide", 1),
        slideToggle: ma("toggle", 1),
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
        c.fn[a] = function(d, e,
        f) {
            return this.animate(b, d, e, f)
        }
    });
    c.extend({
        speed: function(a, b, d) {
            var e = a && typeof a === "object" ? c.extend({}, a): {
                complete: d ||!d && b || c.isFunction(a) && a,
                duration: a,
                easing: d && b || b&&!c.isFunction(b) && b
            };
            e.duration = c.fx.off ? 0 : typeof e.duration === "number" ? e.duration : e.duration in c.fx.speeds ? c.fx.speeds[e.duration] : c.fx.speeds._default;
            e.old = e.complete;
            e.complete = function() {
                e.queue !== false && c(this).dequeue();
                c.isFunction(e.old) && e.old.call(this)
            };
            return e
        },
        easing: {
            linear: function(a, b, d, e) {
                return d + e * a
            },
            swing: function(a, b, d, e) {
                return ( - Math.cos(a * Math.PI) / 2 + 0.5) * e + d
            }
        },
        timers: [],
        fx: function(a, b, d) {
            this.options = b;
            this.elem = a;
            this.prop = d;
            if (!b.orig)
                b.orig = {}
        }
    });
    c.fx.prototype = {
        update: function() {
            this.options.step && this.options.step.call(this.elem, this.now, this);
            (c.fx.step[this.prop] || c.fx.step._default)(this)
        },
        cur: function() {
            if (this.elem[this.prop] != null && (!this.elem.style || this.elem.style[this.prop] == null))
                return this.elem[this.prop];
            var a = parseFloat(c.css(this.elem, this.prop));
            return a && a>-1E4 ? a : 0
        },
        custom: function(a, b, d) {
            function e(j) {
                return f.step(j)
            }
            var f = this, g = c.fx;
            this.startTime = c.now();
            this.start = a;
            this.end = b;
            this.unit = d || this.unit || "px";
            this.now = this.start;
            this.pos = this.state = 0;
            e.elem = this.elem;
            if (e() && c.timers.push(e)&&!Ea)
                Ea = setInterval(g.tick, g.interval)
        },
        show: function() {
            this.options.orig[this.prop] = c.style(this.elem, this.prop);
            this.options.show = true;
            this.custom(this.prop === "width" || this.prop === "height" ? 1 : 0, this.cur());
            c(this.elem).show()
        },
        hide: function() {
            this.options.orig[this.prop] =
            c.style(this.elem, this.prop);
            this.options.hide = true;
            this.custom(this.cur(), 0)
        },
        step: function(a) {
            var b = c.now(), d = true;
            if (a || b >= this.options.duration + this.startTime) {
                this.now = this.end;
                this.pos = this.state = 1;
                this.update();
                this.options.curAnim[this.prop] = true;
                for (var e in this.options.curAnim)
                    if (this.options.curAnim[e] !== true)
                        d = false;
                if (d) {
                    if (this.options.overflow != null&&!c.support.shrinkWrapBlocks) {
                        var f = this.elem, g = this.options;
                        c.each(["", "X", "Y"], function(n, q) {
                            f.style["overflow" + q] = g.overflow[n]
                        })
                    }
                    this.options.hide &&
                    c(this.elem).hide();
                    if (this.options.hide || this.options.show)
                        for (var j in this.options.curAnim)
                            c.style(this.elem, j, this.options.orig[j]);
                    this.options.complete.call(this.elem)
                }
                return false
            } else {
                a = b - this.startTime;
                this.state = a / this.options.duration;
                b = this.options.easing || (c.easing.swing ? "swing" : "linear");
                this.pos = c.easing[this.options.specialEasing && this.options.specialEasing[this.prop] || b](this.state, a, 0, 1, this.options.duration);
                this.now = this.start + (this.end - this.start) * this.pos;
                this.update()
            }
            return true
        }
    };
    c.extend(c.fx, {
        tick: function() {
            for (var a = c.timers, b = 0; b < a.length; b++)
                a[b]() || a.splice(b--, 1);
            a.length || c.fx.stop()
        },
        interval: 13,
        stop: function() {
            clearInterval(Ea);
            Ea = null
        },
        speeds: {
            slow: 600,
            fast: 200,
            _default: 400
        },
        step: {
            opacity: function(a) {
                c.style(a.elem, "opacity", a.now)
            },
            _default: function(a) {
                if (a.elem.style && a.elem.style[a.prop] != null)
                    a.elem.style[a.prop] = (a.prop === "width" || a.prop === "height" ? Math.max(0, a.now) : a.now) + a.unit;
                else 
                    a.elem[a.prop] = a.now
            }
        }
    });
    if (c.expr && c.expr.filters)
        c.expr.filters.animated =
        function(a) {
            return c.grep(c.timers, function(b) {
                return a === b.elem
            }).length
        };
    var tc = /^t(?:able|d|h)$/i, sb = /^(?:body|html)$/i;
    c.fn.offset = "getBoundingClientRect"in y.documentElement ? function(a) {
        var b = this[0], d;
        if (a)
            return this.each(function(j) {
                c.offset.setOffset(this, a, j)
            });
        if (!b ||!b.ownerDocument)
            return null;
        if (b === b.ownerDocument.body)
            return c.offset.bodyOffset(b);
        try {
            d = b.getBoundingClientRect()
        } catch (e) {}
        var f = b.ownerDocument, g = f.documentElement;
        if (!d ||!c.contains(g, b))
            return d || {
                top: 0,
                left: 0
            };
        b =
        f.body;
        f = Ia(f);
        return {
            top: d.top + (f.pageYOffset || c.support.boxModel && g.scrollTop || b.scrollTop) - (g.clientTop || b.clientTop || 0),
            left: d.left + (f.pageXOffset || c.support.boxModel && g.scrollLeft || b.scrollLeft) - (g.clientLeft || b.clientLeft || 0)
        }
    } : function(a) {
        var b = this[0];
        if (a)
            return this.each(function(A) {
                c.offset.setOffset(this, a, A)
            });
        if (!b ||!b.ownerDocument)
            return null;
        if (b === b.ownerDocument.body)
            return c.offset.bodyOffset(b);
        c.offset.initialize();
        var d, e = b.offsetParent, f = b.ownerDocument, g = f.documentElement,
        j = f.body;
        d = (f = f.defaultView) ? f.getComputedStyle(b, null) : b.currentStyle;
        for (var n = b.offsetTop, q = b.offsetLeft; (b = b.parentNode) && b !== j && b !== g;) {
            if (c.offset.supportsFixedPosition && d.position === "fixed")
                break;
            d = f ? f.getComputedStyle(b, null) : b.currentStyle;
            n -= b.scrollTop;
            q -= b.scrollLeft;
            if (b === e) {
                n += b.offsetTop;
                q += b.offsetLeft;
                if (c.offset.doesNotAddBorder&&!(c.offset.doesAddBorderForTableAndCells && tc.test(b.nodeName))) {
                    n += parseFloat(d.borderTopWidth) || 0;
                    q += parseFloat(d.borderLeftWidth) || 0
                }
                e = b.offsetParent
            }
            if (c.offset.subtractsBorderForOverflowNotVisible &&
            d.overflow !== "visible") {
                n += parseFloat(d.borderTopWidth) || 0;
                q += parseFloat(d.borderLeftWidth) || 0
            }
            d = d
        }
        if (d.position === "relative" || d.position === "static") {
            n += j.offsetTop;
            q += j.offsetLeft
        }
        if (c.offset.supportsFixedPosition && d.position === "fixed") {
            n += Math.max(g.scrollTop, j.scrollTop);
            q += Math.max(g.scrollLeft, j.scrollLeft)
        }
        return {
            top: n,
            left: q
        }
    };
    c.offset = {
        initialize: function() {
            var a = y.body, b = y.createElement("div"), d, e, f, g = parseFloat(c.css(a, "marginTop")) || 0;
            c.extend(b.style, {
                position: "absolute",
                top: 0,
                left: 0,
                margin: 0,
                border: 0,
                width: "1px",
                height: "1px",
                visibility: "hidden"
            });
            b.innerHTML = "<div style='position:absolute;top:0;left:0;margin:0;border:5px solid #000;padding:0;width:1px;height:1px;'><div></div></div><table style='position:absolute;top:0;left:0;margin:0;border:5px solid #000;padding:0;width:1px;height:1px;' cellpadding='0' cellspacing='0'><tr><td></td></tr></table>";
            a.insertBefore(b, a.firstChild);
            d = b.firstChild;
            e = d.firstChild;
            f = d.nextSibling.firstChild.firstChild;
            this.doesNotAddBorder = e.offsetTop !==
            5;
            this.doesAddBorderForTableAndCells = f.offsetTop === 5;
            e.style.position = "fixed";
            e.style.top = "20px";
            this.supportsFixedPosition = e.offsetTop === 20 || e.offsetTop === 15;
            e.style.position = e.style.top = "";
            d.style.overflow = "hidden";
            d.style.position = "relative";
            this.subtractsBorderForOverflowNotVisible = e.offsetTop===-5;
            this.doesNotIncludeMarginInBodyOffset = a.offsetTop !== g;
            a.removeChild(b);
            c.offset.initialize = c.noop
        },
        bodyOffset: function(a) {
            var b = a.offsetTop, d = a.offsetLeft;
            c.offset.initialize();
            if (c.offset.doesNotIncludeMarginInBodyOffset) {
                b +=
                parseFloat(c.css(a, "marginTop")) || 0;
                d += parseFloat(c.css(a, "marginLeft")) || 0
            }
            return {
                top: b,
                left: d
            }
        },
        setOffset: function(a, b, d) {
            var e = c.css(a, "position");
            if (e === "static")
                a.style.position = "relative";
            var f = c(a), g = f.offset(), j = c.css(a, "top"), n = c.css(a, "left"), q = e === "absolute" && c.inArray("auto", [j, n])>-1;
            e = {};
            var A = {};
            if (q)
                A = f.position();
            j = q ? A.top : parseInt(j, 10) || 0;
            n = q ? A.left : parseInt(n, 10) || 0;
            if (c.isFunction(b))
                b = b.call(a, d, g);
            if (b.top != null)
                e.top = b.top - g.top + j;
            if (b.left != null)
                e.left = b.left - g.left + n;
            "using"in
            b ? b.using.call(a, e) : f.css(e)
        }
    };
    c.fn.extend({
        position: function() {
            if (!this[0])
                return null;
            var a = this[0], b = this.offsetParent(), d = this.offset(), e = sb.test(b[0].nodeName) ? {
                top: 0,
                left: 0
            }
            : b.offset();
            d.top -= parseFloat(c.css(a, "marginTop")) || 0;
            d.left -= parseFloat(c.css(a, "marginLeft")) || 0;
            e.top += parseFloat(c.css(b[0], "borderTopWidth")) || 0;
            e.left += parseFloat(c.css(b[0], "borderLeftWidth")) || 0;
            return {
                top: d.top - e.top,
                left: d.left - e.left
            }
        },
        offsetParent: function() {
            return this.map(function() {
                for (var a = this.offsetParent ||
                y.body; a&&!sb.test(a.nodeName) && c.css(a, "position") === "static";)
                    a = a.offsetParent;
                return a
            })
        }
    });
    c.each(["Left", "Top"], function(a, b) {
        var d = "scroll" + b;
        c.fn[d] = function(e) {
            var f = this[0], g;
            if (!f)
                return null;
            if (e !== D)
                return this.each(function() {
                    if (g = Ia(this))
                        g.scrollTo(!a ? e : c(g).scrollLeft(), a ? e : c(g).scrollTop());
                    else 
                        this[d] = e
                    });
            else 
                return (g = Ia(f)) ? "pageXOffset"in g ? g[a ? "pageYOffset": "pageXOffset"] : c.support.boxModel && g.document.documentElement[d] || g.document.body[d] : f[d]
        }
    });
    c.each(["Height", "Width"],
    function(a, b) {
        var d = b.toLowerCase();
        c.fn["inner" + b] = function() {
            return this[0] ? parseFloat(c.css(this[0], d, "padding")) : null
        };
        c.fn["outer" + b] = function(e) {
            return this[0] ? parseFloat(c.css(this[0], d, e ? "margin" : "border")) : null
        };
        c.fn[d] = function(e) {
            var f = this[0];
            if (!f)
                return e == null ? null : this;
            if (c.isFunction(e))
                return this.each(function(j) {
                    var n = c(this);
                    n[d](e.call(this, j, n[d]()))
                });
            if (c.isWindow(f))
                return f.document.compatMode === "CSS1Compat" && f.document.documentElement["client" + b] || f.document.body["client" +
                b];
            else if (f.nodeType === 9)
                return Math.max(f.documentElement["client" + b], f.body["scroll" + b], f.documentElement["scroll" + b], f.body["offset" + b], f.documentElement["offset" + b]);
            else if (e === D) {
                f = c.css(f, d);
                var g = parseFloat(f);
                return c.isNaN(g) ? f : g
            } else 
                return this.css(d, typeof e === "string" ? e : e + "px")
        }
    })
})(window);

jQ144 = jQuery.noConflict(true);;
/*
* Slides, A Slideshow Plugin for jQuery
* Intructions: http://slidesjs.com
* By: Nathan Searles, http://nathansearles.com
* Version: 1.1.9
* Updated: September 5th, 2011
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
(function(a) {
    a.fn.slides = function(b) {
        return b = a.extend({}, a.fn.slides.option, b), this.each(function() {
            function w(g, h, i) {
                if (!p && o) {
                    p=!0, b.animationStart(n + 1);
                    switch (g) {
                    case"next":
                        l = n, k = n + 1, k = e === k ? 0 : k, r = f * 2, g =- f * 2, n = k;
                        break;
                    case"prev":
                        l = n, k = n - 1, k = k===-1 ? e - 1 : k, r = 0, g = 0, n = k;
                        break;
                    case"pagination":
                        k = parseInt(i, 10), l = a("." + b.paginationClass + " li." + b.currentClass + " a", c).attr("href").match("[^#/]+$"), k > l ? (r = f * 2, g =- f * 2) : (r = 0, g = 0), n = k
                    }
                    h === "fade" ? b.crossfade ? d.children(":eq(" + k + ")", c).css({
                        zIndex: 10
                    }).fadeIn(b.fadeSpeed, b.fadeEasing, function() {
                        b.autoHeight ? d.animate({
                            height: d.children(":eq(" + k + ")", c).outerHeight()
                        }, b.autoHeightSpeed, function() {
                            d.children(":eq(" + l + ")", c).css({
                                display: "none",
                                zIndex: 0
                            }), d.children(":eq(" + k + ")", c).css({
                                zIndex: 0
                            }), b.animationComplete(k + 1), p=!1
                        }) : (d.children(":eq(" + l + ")", c).css({
                            display: "none",
                            zIndex: 0
                        }), d.children(":eq(" + k + ")", c).css({
                            zIndex: 0
                        }), b.animationComplete(k + 1), p=!1)
                    }) : d.children(":eq(" + l + ")", c).fadeOut(b.fadeSpeed, b.fadeEasing, function() {
                        b.autoHeight ? d.animate({
                            height: d.children(":eq(" + k + ")", c).outerHeight()
                        }, b.autoHeightSpeed, function() {
                            d.children(":eq(" + k + ")", c).fadeIn(b.fadeSpeed, b.fadeEasing)
                        }) : d.children(":eq(" + k + ")", c).fadeIn(b.fadeSpeed, b.fadeEasing, function() {
                            a.browser.msie && a(this).get(0).style.removeAttribute("filter")
                        }), b.animationComplete(k + 1), p=!1
                    }) : (d.children(":eq(" + k + ")").css({
                        left: r,
                        display: "block"
                    }), b.autoHeight ? d.animate({
                        left: g,
                        height: d.children(":eq(" + k + ")").outerHeight()
                    }, b.slideSpeed, b.slideEasing, function() {
                        d.css({
                            left: - f
                        }), d.children(":eq(" + k + ")").css({
                            left: f,
                            zIndex: 5
                        }), d.children(":eq(" + l + ")").css({
                            left: f,
                            display: "none",
                            zIndex: 0
                        }), b.animationComplete(k + 1), p=!1
                    }) : d.animate({
                        left: g
                    }, b.slideSpeed, b.slideEasing, function() {
                        d.css({
                            left: - f
                        }), d.children(":eq(" + k + ")").css({
                            left: f,
                            zIndex: 5
                        }), d.children(":eq(" + l + ")").css({
                            left: f,
                            display: "none",
                            zIndex: 0
                        }), b.animationComplete(k + 1), p=!1
                    })), b.pagination && (a("." + b.paginationClass + " li." + b.currentClass, c).removeClass(b.currentClass), a("." + b.paginationClass + " li:eq(" + k + ")", c).addClass(b.currentClass))
                }
            }
            function x() {
                clearInterval(c.data("interval"))
            }
            function y() {
                b.pause ? (clearTimeout(c.data("pause")), clearInterval(c.data("interval")), u = setTimeout(function() {
                    clearTimeout(c.data("pause")), v = setInterval(function() {
                        w("next", i)
                    }, b.play), c.data("interval", v)
                }, b.pause), c.data("pause", u)) : x()
            }
            a("." + b.container, a(this)).children().wrapAll('<div class="slides_control"/>');
            var c = a(this), d = a(".slides_control", c), e = d.children().size(), f = d.children().outerWidth(), g = d.children().outerHeight(), h = b.start - 1, i = b.effect.indexOf(",") < 0 ? b.effect: b.effect.replace(" ", "").split(",")[0], j = b.effect.indexOf(",") < 0 ? i: b.effect.replace(" ", "").split(",")[1], k = 0, l = 0, m = 0, n = 0, o, p, q, r, s, t, u, v;
            if (e < 2)
                return a("." + b.container, a(this)).fadeIn(b.fadeSpeed, b.fadeEasing, function() {
                    o=!0, b.slidesLoaded()
                }), a("." + b.next + ", ." + b.prev).fadeOut(0), !1;
            if (e < 2)
                return;
            h < 0 && (h = 0), h > e && (h = e - 1), b.start && (n = h), b.randomize && d.randomize(), a("." + b.container, c).css({
                overflow: "hidden",
                position: "relative"
            }), d.children().css({
                position: "absolute",
                top: 0,
                left: d.children().outerWidth(),
                zIndex: 0,
                display: "none"
            }), d.css({
                position: "relative",
                width: f * 3,
                height: g,
                left: - f
            }), a("." + b.container, c).css({
                display: "block"
            }), b.autoHeight && (d.children().css({
                height: "auto"
            }), d.animate({
                height: d.children(":eq(" + h + ")").outerHeight()
            }, b.autoHeightSpeed));
            if (b.preload && d.find("img:eq(" + h + ")").length) {
                a("." + b.container, c).css({
                    background: "url(" + b.preloadImage + ") no-repeat 50% 50%"
                });
                var z = d.find("img:eq(" + h + ")").attr("src") + "?" + (new Date).getTime();
                a("img", c).parent().attr("class") != "slides_control" ? t = d.children(":eq(0)")[0].tagName.toLowerCase() : t = d.find("img:eq(" + h + ")"), d.find("img:eq(" + h + ")").attr("src", z).load(function() {
                    d.find(t + ":eq(" + h + ")").fadeIn(b.fadeSpeed, b.fadeEasing, function() {
                        a(this).css({
                            zIndex: 5
                        }), a("." + b.container, c).css({
                            background: ""
                        }), o=!0, b.slidesLoaded()
                    })
                })
            } else 
                d.children(":eq(" + h + ")").fadeIn(b.fadeSpeed, b.fadeEasing, function() {
                    o=!0, b.slidesLoaded()
                });
            b.bigTarget && (d.children().css({
                cursor: "pointer"
            }), d.children().click(function() {
                return w("next", i), !1
            })), b.hoverPause && b.play && (d.bind("mouseover", function() {
                x()
            }), d.bind("mouseleave", function() {
                y()
            })), b.generateNextPrev && (a("." + b.container, c).after('<a href="#" class="' + b.prev + '">Prev</a>'), a("." + b.prev, c).after('<a href="#" class="' + b.next + '">Next</a>')), a("." + b.next, c).click(function(a) {
                a.preventDefault(), b.play && y(), w("next", i)
            }), a("." + b.prev, c).click(function(a) {
                a.preventDefault(), b.play && y(), w("prev", i)
            }), b.generatePagination ? (b.prependPagination ? c.prepend("<ul class=" + b.paginationClass + "></ul>") : c.append("<ul class=" + b.paginationClass + "></ul>"), d.children().each(function() {
                a("." + b.paginationClass, c).append('<li><a href="#' + m + '">' + (m + 1) + "</a></li>"), m++
            })) : a("." + b.paginationClass + " li a", c).each(function() {
                a(this).attr("href", "#" + m), m++
            }), a("." + b.paginationClass + " li:eq(" + h + ")", c).addClass(b.currentClass), a("." + b.paginationClass + " li a", c).click(function() {
                return b.play && y(), q = a(this).attr("href").match("[^#/]+$"), n != q && w("pagination", j, q), !1
            }), a("a.link", c).click(function() {
                return b.play && y(), q = a(this).attr("href").match("[^#/]+$") - 1, n != q && w("pagination", j, q), !1
            }), b.play && (v = setInterval(function() {
                w("next", i)
            }, b.play), c.data("interval", v))
        })
    }, a.fn.slides.option = {
        preload: !1,
        preloadImage: "/img/loading.gif",
        container: "slides_container",
        generateNextPrev: !1,
        next: "next",
        prev: "prev",
        pagination: !0,
        generatePagination: !0,
        prependPagination: !1,
        paginationClass: "pagination",
        currentClass: "current",
        fadeSpeed: 350,
        fadeEasing: "",
        slideSpeed: 350,
        slideEasing: "",
        start: 1,
        effect: "slide",
        crossfade: !1,
        randomize: !1,
        play: 0,
        pause: 0,
        hoverPause: !1,
        autoHeight: !1,
        autoHeightSpeed: 350,
        bigTarget: !1,
        animationStart: function() {},
        animationComplete: function() {},
        slidesLoaded: function() {}
    }, a.fn.randomize = function(b) {
        function c() {
            return Math.round(Math.random()) - .5
        }
        return a(this).each(function() {
            var d = a(this), e = d.children(), f = e.length;
            if (f > 1) {
                e.hide();
                var g = [];
                for (i = 0; i < f; i++)
                    g[g.length] = i;
                g = g.sort(c), a.each(g, function(a, c) {
                    var f = e.eq(c), g = f.clone(!0);
                    g.show().appendTo(d), b !== undefined && b(f, g), f.remove()
                })
            }
        })
    }
})(jQ144)
;
(function($) {
    $(document).ready(function() {
        $('#slides').slides({
            preload: true,
            preloadImage: '/misc/loading.gif',
            play: 5000,
            pause: 2500,
            hoverPause: true,
            //generateNextPrev: true,
            //prev: 'icon-chevron-left icon-large',
            //next: 'icon-chevron-right icon-large',
            animationStart: function(current) {
                $('.caption').animate({
                    bottom: 0
                }, 100);
                /*
                        if (window.console && console.log) {
                          // example return of current slide number
                          console.log('animationStart on slide: ', current);
                        };
                        */
            },
            animationComplete: function(current) {
                $('.caption').animate({
                    bottom: 115
                }, 200);
                /*
                        if (window.console && console.log) {
                          // example return of current slide number
                          console.log('animationComplete on slide: ', current);
                        };
                        */
            },
            slidesLoaded: function() {
                $('.caption').animate({
                    bottom: 115
                }, 200);
            }
        });
    });
})(jQ144);;

